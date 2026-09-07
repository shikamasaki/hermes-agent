from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_MODEL = "hermes-stub-primary"
JUDGE_MODEL = "hermes-stub-judge"
COMPRESSOR_MODEL = "hermes-stub-compressor"


def _reset_hermes_caches(home: Path) -> None:
    """Point already-imported Hermes modules at the isolated test home."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    reset_hermes_home_override(set_hermes_home_override(home))

    try:
        from hermes_cli import kanban_db as kb

        kb._INITIALIZED_PATHS.clear()
    except Exception:
        pass

    try:
        from hermes_cli import goals

        goals._DB_CACHE.clear()
    except Exception:
        pass

    try:
        import hermes_cli.loops as loops

        getattr(loops, "_DB_CACHE", {}).clear()
    except Exception:
        pass

    try:
        from hermes_cli import config as cfg

        for name in ("_config_cache", "_CONFIG_CACHE", "_cached_config"):
            obj = getattr(cfg, name, None)
            if isinstance(obj, dict):
                obj.clear()
    except Exception:
        pass


@pytest.fixture()
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    _reset_hermes_caches(home)

    hermes_state_mod = sys.modules.get("hermes_state")
    if hermes_state_mod is not None and hasattr(hermes_state_mod, "DEFAULT_DB_PATH"):
        monkeypatch.setattr(hermes_state_mod, "DEFAULT_DB_PATH", home / "state.db")

    from hermes_cli import kanban_db as kb

    kb.init_db()
    yield home
    _reset_hermes_caches(home)


class _StubHandler(BaseHTTPRequestHandler):
    server_version = "HermesScenarioStub/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - keep pytest quiet
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        length = int(self.headers.get("content-length") or "0")
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"_raw": body}
        authorization = self.headers.get("authorization") or ""
        with self.server.request_lock:  # type: ignore[attr-defined]
            self.server.request_seq += 1  # type: ignore[attr-defined]
            request_seq = int(self.server.request_seq)  # type: ignore[attr-defined]
        record = {
            "seq": request_seq,
            "recorded_at": time.time(),
            "path": self.path,
            "authorization": authorization,
            "payload": payload,
        }
        self.server.requests.put(record)  # type: ignore[attr-defined]
        if authorization != "Bearer test-key":
            raw = json.dumps({"error": {"message": "unauthorized"}}).encode("utf-8")
            record["stub_kind"] = "auth"
            record["response_status"] = 401
            self.send_response(401)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        messages = payload.get("messages") or []
        model = str(payload.get("model") or "")
        fail_compressor = bool(getattr(self.server, "fail_compressor_empty", False))  # type: ignore[attr-defined]
        if model == COMPRESSOR_MODEL and fail_compressor:
            content = ""
            record["stub_kind"] = "compression"
            record["stub_failure"] = "empty_content"
            record["response_status"] = 200
        else:
            record["stub_kind"] = "completion"
            record["response_status"] = 200
        wants_judge = model == JUDGE_MODEL or any(
            isinstance(m, dict)
            and "JSON" in str(m.get("content") or "")
            and "verdict" in str(m.get("content") or "").lower()
            for m in messages
        ) or any(
            isinstance(m, dict)
            and "goal" in str(m.get("content") or "").lower()
            and "judge" in str(m.get("content") or "").lower()
            for m in messages
        )
        if model == COMPRESSOR_MODEL and fail_compressor:
            pass
        elif wants_judge:
            record["stub_kind"] = "judge"
            content = json.dumps({"verdict": "done", "reason": "localhost judge accepted child success"})
        else:
            if model == PRIMARY_MODEL:
                record["stub_kind"] = "primary"
            primary_reply = getattr(self.server, "primary_reply", None)  # type: ignore[attr-defined]
            content = str(primary_reply or "ok")
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            chunk = {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": payload.get("model") or "hermes-stub",
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            done = {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": payload.get("model") or "hermes-stub",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(f"data: {json.dumps(done)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return

        response = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or "hermes-stub",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class LocalOpenAIStub:
    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self._httpd.requests = queue.Queue()  # type: ignore[attr-defined]
        self._httpd.request_lock = threading.Lock()  # type: ignore[attr-defined]
        self._httpd.request_seq = 0  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="hermes-scenario-stub", daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}/v1"

    def requests(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        q = self._httpd.requests  # type: ignore[attr-defined]
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                return out

    def set_primary_reply(self, *, task_id: str, reason: str) -> str:
        text = f"{task_id}: {reason}"
        self._httpd.primary_reply = text  # type: ignore[attr-defined]
        return text

    def fail_compressor_with_empty_content(self) -> None:
        self._httpd.fail_compressor_empty = True  # type: ignore[attr-defined]

    def assert_rejects_bad_auth(self) -> None:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({"model": PRIMARY_MODEL, "messages": []}).encode("utf-8"),
            headers={"authorization": "Bearer wrong-key", "content-type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=2)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                return
            raise AssertionError(f"bad auth rejected with unexpected status {exc.code}") from exc
        raise AssertionError("bad auth was accepted by localhost stub")

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2)


@pytest.fixture()
def openai_stub() -> LocalOpenAIStub:
    stub = LocalOpenAIStub()
    try:
        yield stub
    finally:
        stub.close()


def write_stub_config(home: Path, base_url: str, *, threshold_tokens: int | None = None, max_attempts: int = 3) -> None:
    if threshold_tokens is None:
        (home / "config.yaml").write_text(
            "\n".join(
                [
                    "model:",
                    "  provider: custom",
                    f"  model: {PRIMARY_MODEL}",
                    f"  base_url: {base_url}",
                    "auxiliary:",
                    "  goal_judge:",
                    "    provider: custom",
                    f"    model: {JUDGE_MODEL}",
                    f"    base_url: {base_url}",
                    "    api_key: test-key",
                    "    api_mode: chat_completions",
                    "    timeout: 5",
                    "    max_tokens: 128",
                    "goals:",
                    "  max_turns: 3",
                    "display:",
                    "  interim_assistant_messages: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return

    (home / "config.yaml").write_text(
        "\n".join(
            [
                "model:",
                "  provider: custom",
                f"  model: {PRIMARY_MODEL}",
                f"  base_url: {base_url}",
                "auxiliary:",
                "  goal_judge:",
                "    provider: custom",
                f"    model: {JUDGE_MODEL}",
                f"    base_url: {base_url}",
                "    api_key: test-key",
                "    api_mode: chat_completions",
                "    timeout: 5",
                "    max_tokens: 128",
                "  compression:",
                "    provider: custom",
                f"    model: {COMPRESSOR_MODEL}",
                f"    base_url: {base_url}",
                "    api_key: test-key",
                "    api_mode: chat_completions",
                "    timeout: 5",
                "    context_length: 131072",
                "compression:",
                "  enabled: true",
                "  in_place: true",
                f"  threshold_tokens: {threshold_tokens}",
                "  protect_first_n: 1",
                "  protect_last_n: 4",
                "  abort_on_summary_failure: true",
                f"  max_attempts: {max_attempts}",
                "goals:",
                "  max_turns: 3",
                "display:",
                "  interim_assistant_messages: false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_real_agent(base_url: str, session_key: str, session_db: Any = None):
    from run_agent import AIAgent
    from hermes_state_registry import get_shared_session_db

    if session_db is None:
        session_db = get_shared_session_db()

    return AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider="custom",
        api_mode="chat_completions",
        model=PRIMARY_MODEL,
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        session_id=session_key,
        gateway_session_key=session_key,
        platform="tui",
        session_db=session_db,
    )


@pytest.fixture()
def server_module(hermes_home: Path):
    import importlib

    mod = importlib.import_module("tui_gateway.server")
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    yield mod
    for sess in list(mod._sessions.values()):
        try:
            mod._teardown_session(sess)
        except Exception:
            pass
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()


def make_tui_session(server: Any, home: Path, sid: str, session_key: str, agent: Any) -> dict[str, Any]:
    """Create a real TUI session through server._init_session.

    _init_session is the entry point under test: it starts the daemon
    notification poller and stores its stop event on the session.
    """
    history = []
    for idx in range(20):
        history.append({"role": "user", "content": f"long history user {idx}: " + ("x" * 220)})
        history.append({"role": "assistant", "content": f"long history assistant {idx}: " + ("y" * 220)})
    server._init_session(
        sid,
        session_key,
        agent,
        history,
        cols=100,
        cwd=str(home),
        source="tui",
        profile_home=str(home),
        explicit_cwd=True,
    )
    session = server._sessions[sid]
    session["model_override"] = {"model": PRIMARY_MODEL, "provider": "custom"}
    return session


def secretless_child_env(home: Path) -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL"}
    env = {k: v for k, v in os.environ.items() if k in allowed and v}
    env.update(
        {
            "HOME": str(home.parent),
            "HERMES_HOME": str(home),
            "HERMES_KANBAN_DB": str(home / "kanban.db"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(SOURCE_ROOT),
        }
    )
    return env


def run_real_child_process(
    home: Path,
    session_key: str,
    *,
    task_id: str,
    alive_marker: Path,
    release_fifo: Path,
    result: str = "child result",
    summary: str = "child success",
) -> subprocess.Popen[str]:
    """Start a real child process that blocks on a real FIFO barrier."""
    script = r'''
import json
import os
import sys
import time
from pathlib import Path
from hermes_cli import kanban_db as kb

task_id = os.environ["HERMES_KANBAN_TASK"]
result = os.environ.get("HERMES_CHILD_RESULT", "child result")
summary = os.environ.get("HERMES_CHILD_SUMMARY", "child success")
alive_marker = Path(os.environ["HERMES_CHILD_ALIVE_MARKER"])
release_fifo = os.environ["HERMES_CHILD_RELEASE_FIFO"]
alive_marker.write_text(json.dumps({"pid": os.getpid(), "task_id": task_id}), encoding="utf-8")
with open(release_fifo, "rb", buffering=0) as gate:
    token = gate.read(1)
if token != b"1":
    print(json.dumps({"task_id": task_id, "error": "release token missing"}), file=sys.stderr)
    sys.exit(2)

conn = kb.connect()
try:
    before = kb.get_task(conn, task_id)
    claimed = kb.claim_task(conn, task_id, claimer="isolated-child", ttl_seconds=60)
    after_claim = kb.get_task(conn, task_id)
    ok = kb.complete_task(conn, task_id, result=result, summary=summary)
    after = kb.get_task(conn, task_id)
    row = conn.execute(
        "SELECT id, kind, payload FROM task_events WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    print(json.dumps({
        "task_id": task_id,
        "result": result,
        "summary": summary,
        "complete_ok": ok,
        "completed_at": time.time(),
        "status_before": before.status if before else None,
        "claim_ok": claimed is not None,
        "status_after_claim": after_claim.status if after_claim else None,
        "status_after": after.status if after else None,
        "event_id": int(row["id"]),
        "event_kind": row["kind"],
        "event_payload": row["payload"],
        "kanban_db_file": kb.__file__,
        "home_env": os.environ.get("HOME"),
    }))
finally:
    conn.close()
'''
    env = secretless_child_env(home)
    env["HERMES_SESSION_KEY"] = session_key
    env["HERMES_KANBAN_TASK"] = task_id
    env["HERMES_CHILD_ALIVE_MARKER"] = str(alive_marker)
    env["HERMES_CHILD_RELEASE_FIFO"] = str(release_fifo)
    env["HERMES_CHILD_RESULT"] = result
    env["HERMES_CHILD_SUMMARY"] = summary
    return subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(home),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def collect_real_child_process(proc: subprocess.Popen[str], timeout: float = 10.0) -> dict[str, Any]:
    killed = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        killed = True
        proc.kill()
        stdout, stderr = proc.communicate(timeout=2)
    record = {
        "pid": proc.pid,
        "returncode": proc.returncode,
        "stdout": (stdout or "").strip(),
        "stderr": (stderr or "").strip(),
        "killed": killed,
    }
    if proc.returncode != 0:
        raise AssertionError(f"child process failed: {record}")
    payload = json.loads(record["stdout"])
    record.update(payload)
    return record


def wait_for_turn_to_finish(session: dict[str, Any], timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with session["history_lock"]:
            running = bool(session.get("running"))
            inflight = session.get("inflight_turn")
        if not running and not (isinstance(inflight, dict) and inflight.get("streaming")):
            return
        time.sleep(0.05)
    raise AssertionError(f"turn did not finish; running={session.get('running')} inflight={session.get('inflight_turn')}")


def run_abnormal_kanban_child(home: Path, task_id: str, *, exit_code: int = 42) -> subprocess.Popen[str]:
    """Start a real subprocess that exits non-zero for crash detection tests."""
    script = r'''
import os
import sys
import time
from pathlib import Path

marker = Path(os.environ["HERMES_CHILD_ALIVE_MARKER"])
marker.write_text(str(os.getpid()), encoding="utf-8")
time.sleep(0.05)
sys.exit(int(os.environ.get("HERMES_CHILD_EXIT_CODE", "42")))
'''
    env = secretless_child_env(home)
    marker = home / f"{task_id}.abnormal-child.pid"
    env["HERMES_CHILD_ALIVE_MARKER"] = str(marker)
    env["HERMES_CHILD_EXIT_CODE"] = str(exit_code)
    return subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(home),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_isolated_parent_process(
    home: Path,
    base_url: str,
    *,
    sid: str,
    session_key: str,
    marker: Path,
    mode: str,
    expected_text: str = "",
    timeout: float = 30.0,
    initial_history: list[dict[str, Any]] | None = None,
) -> subprocess.Popen[str]:
    """Run a real isolated parent process that owns a TUI session."""
    script = r'''
import hashlib
import json
import os
import sys
import time
import threading
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import kanban_db as kb
from hermes_state_registry import get_shared_session_db, release_or_close
from mcp_serve import _load_session_messages
from run_agent import AIAgent
import tui_gateway.server as server

PRIMARY_MODEL = "hermes-stub-primary"
base_url = os.environ["HERMES_PARENT_BASE_URL"]
home = Path(os.environ["HERMES_HOME"])
sid = os.environ["HERMES_PARENT_SID"]
session_key = os.environ["HERMES_PARENT_SESSION_KEY"]
mode = os.environ["HERMES_PARENT_MODE"]
marker = Path(os.environ["HERMES_PARENT_MARKER"])
expected_text = os.environ.get("HERMES_PARENT_EXPECTED_TEXT", "")
timeout = float(os.environ.get("HERMES_PARENT_TIMEOUT", "30"))
temp_root = Path(os.environ["HERMES_PARENT_TEMP_ROOT"])
initial_history_raw = os.environ.get("HERMES_PARENT_INITIAL_HISTORY", "")


def _transcript_signature(messages):
    normalized = [
        {"role": str(msg.get("role", "")), "content": str(msg.get("content", ""))}
        for msg in messages
    ]
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return {
        "roles_content": normalized,
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }


def _inside(path, root):
    return Path(path).expanduser().resolve().is_relative_to(Path(root).expanduser().resolve())


def _guard_evidence():
    return {
        "home_env": os.environ.get("HOME"),
        "hermes_home_env": os.environ.get("HERMES_HOME"),
        "state_db": str(home / "state.db"),
        "temp_root": str(temp_root),
        "bypass": os.environ.get("HERMES_STATE_DB_GUARD_BYPASS"),
    }


guard = _guard_evidence()
if guard["bypass"] is not None:
    raise RuntimeError(f"state DB guard bypass must not be set: {guard}")
if not _inside(Path(guard["home_env"]), temp_root):
    raise RuntimeError(f"HOME escaped parent temp root: {guard}")
if not _inside(home, temp_root):
    raise RuntimeError(f"HERMES_HOME escaped parent temp root: {guard}")
if not _inside(home / "state.db", temp_root):
    raise RuntimeError(f"state.db escaped parent temp root: {guard}")
if (Path(guard["home_env"]) / ".hermes").resolve() == home.resolve():
    raise RuntimeError(f"HOME/.hermes aliases HERMES_HOME: {guard}")
initial_history = json.loads(initial_history_raw) if initial_history_raw else []
if initial_history and mode != "hold":
    raise RuntimeError("initial history may only be injected into the old hold process")

reset_hermes_home_override(set_hermes_home_override(home))
kb.init_db()
server._sessions.clear()
server._pending.clear()
server._answers.clear()
session_db = get_shared_session_db()
try:
    restored_history = []
    restored_history_signature = None
    if not initial_history and mode == "recover":
        restored_history = session_db.get_messages_as_conversation(
            session_key,
            repair_alternation=True,
            include_row_ids=True,
        )
        if not restored_history:
            raise RuntimeError("new parent recovered no durable history from SessionDB")
        restored_history_signature = _transcript_signature(restored_history)
    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        provider="custom",
        api_mode="chat_completions",
        model=PRIMARY_MODEL,
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        session_id=session_key,
        gateway_session_key=session_key,
        platform="tui",
        session_db=session_db,
    )
    history = list(initial_history or restored_history)
    server._init_session(
        sid,
        session_key,
        agent,
        history,
        cols=100,
        cwd=str(home),
        source="tui",
        profile_home=str(home),
        explicit_cwd=True,
    )
    session = server._sessions[sid]
    session["model_override"] = {"model": PRIMARY_MODEL, "provider": "custom"}
    initial_history_signature = None
    if initial_history:
        initial_history_signature = _transcript_signature(initial_history)
        agent._session_messages = list(initial_history)
        if not server._flush_session_messages(session):
            raise RuntimeError("failed to persist old parent initial history")
        saved_initial, saved_initial_err = _load_session_messages(agent.session_id)
        if saved_initial_err is not None or saved_initial is None:
            raise RuntimeError(f"failed to reload old parent initial history: {saved_initial_err}")
        if _transcript_signature(saved_initial[:len(initial_history)]) != initial_history_signature:
            raise RuntimeError("old parent initial history did not round-trip through SessionDB")
    marker.write_text(json.dumps({
        "pid": os.getpid(),
        "sid": sid,
        "session_key": session_key,
        "guard": guard,
        "initial_history_signature": initial_history_signature,
        "restored_history_signature": restored_history_signature,
    }, ensure_ascii=False), encoding="utf-8")
    if mode == "hold":
        while True:
            time.sleep(0.2)
    if mode != "recover":
        raise SystemExit(f"unknown mode {mode}")
    deadline = time.time() + timeout
    evidence = {}
    while time.time() < deadline:
        messages, err = _load_session_messages(agent.session_id)
        assistant_texts = [str(m.get("content", "")) for m in (messages or []) if m.get("role") == "assistant"]
        post_turn_initial_history_signature = (
            _transcript_signature((messages or [])[:len(restored_history)])
            if mode == "recover" and restored_history_signature is not None
            else None
        )
        with kb.connect() as conn:
            subs = kb.list_notify_subs(conn)
            events = [dict(r) for r in conn.execute("SELECT id, task_id, kind, payload FROM task_events ORDER BY id").fetchall()]
        if err is None and expected_text and expected_text in assistant_texts:
            evidence = {
                "evidence": True,
                "pid": os.getpid(),
                "sid": sid,
                "session_key": session_key,
                "assistant_texts": assistant_texts,
                "messages": messages,
                "subs": subs,
                "events": events,
                "running": bool(session.get("running")),
                "guard": guard,
                "initial_history_signature": initial_history_signature,
                "restored_history_signature": restored_history_signature,
                "post_turn_initial_history_signature": post_turn_initial_history_signature,
            }
            break
        time.sleep(0.05)
    else:
        messages, err = _load_session_messages(agent.session_id)
        with kb.connect() as conn:
            subs = kb.list_notify_subs(conn)
            events = [dict(r) for r in conn.execute("SELECT id, task_id, kind, payload FROM task_events ORDER BY id").fetchall()]
        evidence = {
            "timeout": True,
            "error": err,
            "messages": messages,
            "subs": subs,
            "events": events,
            "running": bool(session.get("running")),
            "guard": guard,
            "initial_history_signature": initial_history_signature,
            "restored_history_signature": restored_history_signature,
            "post_turn_initial_history_signature": (
                _transcript_signature((messages or [])[:len(restored_history)])
                if mode == "recover" and restored_history_signature is not None
                else None
            ),
        }
        evidence_json = json.dumps(evidence, ensure_ascii=False)
        if mode == "recover":
            try:
                Path(str(marker) + ".evidence").write_text(evidence_json, encoding="utf-8")
            except Exception:
                pass
        print("\n" + evidence_json)
        raise SystemExit(3)
    evidence_json = json.dumps(evidence, ensure_ascii=False)
    if mode == "recover":
        try:
            Path(str(marker) + ".evidence").write_text(evidence_json, encoding="utf-8")
        except Exception:
            pass
    print("\n" + evidence_json)
finally:
    for sess in list(server._sessions.values()):
        try:
            server._teardown_session(sess)
        except Exception:
            pass
    server._sessions.clear()
    server._pending.clear()
    server._answers.clear()
    release_or_close(session_db)
'''
    env = secretless_child_env(home)
    parent_home = home.parent / f".isolated-parent-home-{sid}"
    parent_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(parent_home)
    env.update(
        {
            "HERMES_PARENT_BASE_URL": base_url,
            "HERMES_PARENT_SID": sid,
            "HERMES_PARENT_SESSION_KEY": session_key,
            "HERMES_PARENT_MARKER": str(marker),
            "HERMES_PARENT_MODE": mode,
            "HERMES_PARENT_EXPECTED_TEXT": expected_text,
            "HERMES_PARENT_TIMEOUT": str(timeout),
            "HERMES_PARENT_TEMP_ROOT": str(home.parent),
            "HERMES_PARENT_INITIAL_HISTORY": json.dumps(initial_history or [], ensure_ascii=False),
        }
    )
    return subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(home),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
