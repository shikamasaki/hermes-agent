"""Regression tests for delegate_task isolation from parent Kanban workers."""
from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path

import pytest

# The subprocess-boundary tests below spawn ``sys.executable -c`` with a tmp
# cwd. Without an explicit PYTHONPATH the child resolves ``hermes_cli`` /
# ``agent`` through whatever install is on sys.path (in a worktree that is the
# MAIN checkout's editable install, which may not contain the code under
# test). Pin the repo root so the child always imports the tree being tested.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _python_with_repo_path(code: str) -> str:
    """Build a shell command running *code* with the repo under test on PYTHONPATH."""
    return (
        f"PYTHONPATH={shlex.quote(str(_REPO_ROOT))} "
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    )


def _run_kanban_cli_code(argv: list[str]) -> str:
    return (
        "from hermes_cli import kanban; "
        "import argparse; "
        "p=argparse.ArgumentParser(); "
        "sub=p.add_subparsers(dest='cmd'); "
        "kanban.build_parser(sub); "
        f"args=p.parse_args({argv!r}); "
        "raise SystemExit(kanban.kanban_command(args))"
    )


def _pollute_terminal_snapshot(env, text: str) -> None:
    env._snapshot_ready = True
    Path(env._snapshot_path).write_text(text, encoding="utf-8")


def _make_running_kanban_task(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    attachments_root = tmp_path / "attachments"
    workspace = tmp_path / "parent-workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "parent-worker")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(attachments_root))

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="parent",
            assignee="parent-worker",
            workspace_kind="scratch",
            workspace_path=str(workspace),
        )
        claim = kb.claim_task(conn, tid)
        assert claim is not None
        run_id = claim.id
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return kb, tid, workspace, attachments_root


def test_delegated_child_context_suppresses_env_gated_kanban_tools(monkeypatch, tmp_path):
    """A delegate_task child must not inherit the parent's Kanban tool schema.

    The parent process may be a dispatcher worker with HERMES_KANBAN_TASK set;
    the child is only a subagent, not the run owner.
    """
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import tools.kanban_tools  # noqa: F401 - ensure registered
    from agent.delegation_context import delegated_child_context
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    with delegated_child_context():
        schema = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)

    names = {s["function"].get("name") for s in schema if "function" in s}
    assert "terminal" in names
    assert {n for n in names if n and n.startswith("kanban_")} == set()


def test_build_child_agent_strips_kanban_toolset_even_when_parent_is_worker(monkeypatch):
    """Child construction must fail closed even if the parent exposes kanban."""
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.valid_tool_names = {"terminal"}
            self.session_id = "child-session"

    import run_agent
    from tools import delegate_tool

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})

    class Parent:
        enabled_toolsets = ["terminal", "kanban"]
        valid_tool_names = {"terminal", "kanban_complete", "kanban_comment"}
        model = "test-model"
        provider = "test-provider"
        base_url = "http://example.invalid"
        api_mode = "chat_completions"
        platform = "cli"
        session_id = "parent-session"

    child = delegate_tool._build_child_agent(
        task_index=0,
        goal="review only",
        context=None,
        toolsets=None,
        model=None,
        max_iterations=3,
        task_count=1,
        parent_agent=Parent(),
    )

    assert child.valid_tool_names == {"terminal"}
    assert "kanban" not in captured["enabled_toolsets"]
    assert "kanban" in captured["disabled_toolsets"]


def test_delegate_child_execute_code_env_bridges_contextvar_and_scrubs_kanban(
    monkeypatch,
    tmp_path,
):
    """The real execute_code child-env builder must bridge ContextVar lineage.

    Regression coverage for the vulnerable path: delegate_task marks child
    execution with a ContextVar, while execute_code used to scrub plain
    ``os.environ`` and therefore never wrote HERMES_DELEGATED_CHILD_CONTEXT into
    the sandbox env.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path / "parent-workspace"))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    from agent.delegation_context import delegated_child_context
    from tools.code_execution_tool import _scrub_child_env

    with delegated_child_context():
        env = _scrub_child_env(
            dict(os.environ),
            is_passthrough=lambda k: k.startswith("HERMES_KANBAN_"),
            is_windows=False,
        )

    assert os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT") is None
    assert env["HERMES_HOME"] == str(home)
    assert env["HERMES_DELEGATED_CHILD_CONTEXT"] == "1"
    assert "HERMES_KANBAN_TASK" not in env
    assert "HERMES_KANBAN_RUN_ID" not in env
    assert "HERMES_KANBAN_DB" not in env
    assert "HERMES_KANBAN_WORKSPACE" not in env
    assert "HERMES_KANBAN_CLAIM_LOCK" not in env


def test_delegate_child_kanban_cli_cannot_delete_parent_board(
    monkeypatch,
    tmp_path,
):
    kb, _tid, _workspace, _attachments_root = _make_running_kanban_task(
        monkeypatch,
        tmp_path,
    )
    kb.create_board("victim")
    assert kb.board_exists("victim")

    from agent.delegation_context import delegated_child_context
    from tools.environments.local import LocalEnvironment

    code = _run_kanban_cli_code(['kanban', 'boards', 'rm', 'victim', '--delete'])
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    try:
        with delegated_child_context():
            result = env.execute(
                _python_with_repo_path(code),
                timeout=15,
            )
    finally:
        env.cleanup()

    assert result["returncode"] == 1
    assert "delegate_task child contexts cannot mutate Kanban tasks" in result["output"]
    assert kb.board_exists("victim")
    assert kb.board_dir("victim").is_dir()


def test_parent_kanban_cli_create_and_comment_survive_polluted_snapshot(
    monkeypatch,
    tmp_path,
):
    kb, tid, _workspace, _attachments_root = _make_running_kanban_task(
        monkeypatch,
        tmp_path,
    )

    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    _pollute_terminal_snapshot(
        env,
        'declare -x HERMES_DELEGATED_CHILD_CONTEXT="1"\n',
    )
    try:
        create = env.execute(
            _python_with_repo_path(
                _run_kanban_cli_code(['kanban', 'create', 'tmp parent cli task'])
            ),
            timeout=15,
        )
        comment = env.execute(
            _python_with_repo_path(
                _run_kanban_cli_code(['kanban', 'comment', tid, 'parent comment ok'])
            ),
            timeout=15,
        )
    finally:
        env.cleanup()

    assert create["returncode"] == 0, create["output"]
    assert "delegate_task child contexts cannot mutate Kanban" not in create["output"]
    assert comment["returncode"] == 0, comment["output"]
    assert "delegate_task child contexts cannot mutate Kanban" not in comment["output"]

    conn = kb.connect()
    try:
        tasks = kb.list_tasks(conn)
        comments = kb.list_comments(conn, tid)
    finally:
        conn.close()
    assert any(task.title == "tmp parent cli task" for task in tasks)
    assert any(comment.body == "parent comment ok" for comment in comments)


def test_child_kanban_cli_comment_still_rejected_after_snapshot_restore(
    monkeypatch,
    tmp_path,
):
    kb, tid, _workspace, _attachments_root = _make_running_kanban_task(
        monkeypatch,
        tmp_path,
    )

    from agent.delegation_context import delegated_child_context
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    _pollute_terminal_snapshot(env, "unset HERMES_DELEGATED_CHILD_CONTEXT\n")
    try:
        with delegated_child_context():
            result = env.execute(
                _python_with_repo_path(
                    _run_kanban_cli_code(['kanban', 'comment', tid, 'child should fail'])
                ),
                timeout=15,
            )
    finally:
        env.cleanup()

    assert result["returncode"] == 1
    assert "delegate_task child contexts cannot mutate Kanban tasks" in result["output"]

    conn = kb.connect()
    try:
        comments = kb.list_comments(conn, tid)
    finally:
        conn.close()
    assert all(comment.body != "child should fail" for comment in comments)


def test_same_local_environment_child_cli_rejection_does_not_poison_parent(
    monkeypatch,
    tmp_path,
):
    kb, tid, _workspace, _attachments_root = _make_running_kanban_task(
        monkeypatch,
        tmp_path,
    )
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    from agent.delegation_context import delegated_child_context
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    try:
        parent_before = env.execute(
            _python_with_repo_path(
                _run_kanban_cli_code(["kanban", "comment", tid, "parent before ok"])
            ),
            timeout=15,
        )
        with delegated_child_context():
            child = env.execute(
                _python_with_repo_path(
                    _run_kanban_cli_code(["kanban", "comment", tid, "child denied"])
                ),
                timeout=15,
            )
        parent_after = env.execute(
            _python_with_repo_path(
                _run_kanban_cli_code(["kanban", "comment", tid, "parent after ok"])
            ),
            timeout=15,
        )
    finally:
        env.cleanup()

    assert parent_before["returncode"] == 0, parent_before["output"]
    assert child["returncode"] == 1, child["output"]
    assert "delegate_task child contexts cannot mutate Kanban tasks" in child["output"]
    assert parent_after["returncode"] == 0, parent_after["output"]

    conn = kb.connect()
    try:
        comments = [comment.body for comment in kb.list_comments(conn, tid)]
    finally:
        conn.close()
    assert "parent before ok" in comments
    assert "parent after ok" in comments
    assert "child denied" not in comments


def test_parallel_child_cli_rejections_sharing_snapshot_do_not_poison_parent(
    monkeypatch,
    tmp_path,
):
    kb, tid, _workspace, _attachments_root = _make_running_kanban_task(
        monkeypatch,
        tmp_path,
    )
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    from agent.delegation_context import delegated_child_context
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    barrier = threading.Barrier(3)
    results: list[dict] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run_child(label: str) -> None:
        try:
            with delegated_child_context():
                barrier.wait(timeout=10)
                result = env.execute(
                    _python_with_repo_path(
                        _run_kanban_cli_code(["kanban", "comment", tid, label])
                    ),
                    timeout=15,
                )
            with lock:
                results.append(result)
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=run_child, args=("child denied 1",)),
        threading.Thread(target=run_child, args=("child denied 2",)),
    ]
    try:
        parent_before = env.execute(
            _python_with_repo_path(
                _run_kanban_cli_code(["kanban", "comment", tid, "parent before race ok"])
            ),
            timeout=15,
        )
        for thread in threads:
            thread.start()
        barrier.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=20)
        parent_after = env.execute(
            _python_with_repo_path(
                _run_kanban_cli_code(["kanban", "comment", tid, "parent after race ok"])
            ),
            timeout=15,
        )
    finally:
        env.cleanup()

    assert parent_before["returncode"] == 0, parent_before["output"]
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == 2
    for result in results:
        assert result["returncode"] == 1, result["output"]
        assert "delegate_task child contexts cannot mutate Kanban tasks" in result["output"]
    assert parent_after["returncode"] == 0, parent_after["output"]

    conn = kb.connect()
    try:
        comments = [comment.body for comment in kb.list_comments(conn, tid)]
    finally:
        conn.close()
    assert "parent before race ok" in comments
    assert "parent after race ok" in comments
    assert "child denied 1" not in comments
    assert "child denied 2" not in comments


def test_child_exception_and_interrupt_finally_do_not_poison_parent_cli(
    monkeypatch,
    tmp_path,
):
    kb, tid, _workspace, _attachments_root = _make_running_kanban_task(
        monkeypatch,
        tmp_path,
    )
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    from agent.delegation_context import delegated_child_context
    from tools.environments.local import LocalEnvironment
    from tools.interrupt import set_interrupt

    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    sentinel = tmp_path / "child-subprocess-started"
    finished = threading.Event()
    interrupted: dict[str, dict] = {}
    errors: list[BaseException] = []

    def interrupted_child() -> None:
        try:
            with delegated_child_context():
                interrupted["result"] = env.execute(
                    _python_with_repo_path(
                        "from pathlib import Path; "
                        "import time; "
                        f"Path({str(sentinel)!r}).write_text('started', encoding='utf-8'); "
                        "print('child command started', flush=True); "
                        "time.sleep(30)"
                    ),
                    timeout=15,
                )
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    try:
        with pytest.raises(RuntimeError, match="intentional child failure"):
            with delegated_child_context():
                child_before_exception = env.execute(
                    _python_with_repo_path(
                        _run_kanban_cli_code(
                            ["kanban", "comment", tid, "child before exception denied"]
                        )
                    ),
                    timeout=15,
                )
                assert child_before_exception["returncode"] == 1, child_before_exception["output"]
                assert (
                    "delegate_task child contexts cannot mutate Kanban tasks"
                    in child_before_exception["output"]
                )
                raise RuntimeError("intentional child failure")

        parent_after_exception = env.execute(
            _python_with_repo_path(
                _run_kanban_cli_code(["kanban", "comment", tid, "parent after exception ok"])
            ),
            timeout=15,
        )

        thread = threading.Thread(target=interrupted_child)
        thread.start()
        deadline = time.monotonic() + 10
        while not sentinel.exists() and time.monotonic() < deadline:
            finished.wait(timeout=0.05)
        assert sentinel.read_text(encoding="utf-8") == "started"
        assert thread.ident is not None
        set_interrupt(True, thread.ident, reason="test interrupt")
        try:
            assert finished.wait(timeout=20)
        finally:
            set_interrupt(False, thread.ident)
        thread.join(timeout=1)

        parent_after_interrupt = env.execute(
            _python_with_repo_path(
                _run_kanban_cli_code(["kanban", "comment", tid, "parent after interrupt ok"])
            ),
            timeout=15,
        )
    finally:
        env.cleanup()

    assert parent_after_exception["returncode"] == 0, parent_after_exception["output"]
    assert errors == []
    assert interrupted["result"]["returncode"] == 130, interrupted["result"]["output"]
    assert "[Command interrupted]" in interrupted["result"]["output"]
    assert parent_after_interrupt["returncode"] == 0, parent_after_interrupt["output"]

    conn = kb.connect()
    try:
        comments = [comment.body for comment in kb.list_comments(conn, tid)]
    finally:
        conn.close()
    assert "parent after exception ok" in comments
    assert "parent after interrupt ok" in comments
    assert "child before exception denied" not in comments


def test_delegate_child_attach_url_guard_leaves_no_row_or_file(monkeypatch, tmp_path):
    kb, tid, _workspace, attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)

    from agent.delegation_context import delegated_child_context
    from tools import kanban_tools

    def forbidden_download(*_args, **_kwargs):
        raise AssertionError("delegated child guard must run before URL download")

    monkeypatch.setattr(kanban_tools, "_download_url_with_cap", forbidden_download)

    with delegated_child_context():
        raw = kanban_tools._handle_attach_url({
            "task_id": tid,
            "url": "https://example.com/leak.txt",
        })

    payload = json.loads(raw)
    assert payload["error"]
    assert "delegate_task child" in payload["error"]

    conn = kb.connect()
    try:
        assert kb.list_attachments(conn, tid) == []
    finally:
        conn.close()
    task_dir = attachments_root / tid
    assert not task_dir.exists() or list(task_dir.iterdir()) == []


def test_child_attempting_default_complete_does_not_finish_parent_or_delete_workspace(
    monkeypatch,
    tmp_path,
):
    """Deterministic E2E: a delegated child cannot complete its parent task."""
    kb, tid, workspace, _attachments_root = _make_running_kanban_task(monkeypatch, tmp_path)
    from tools import delegate_tool
    from tools import kanban_tools

    class Parent:
        _current_task_id = tid

        def _touch_activity(self, _desc):
            return None

    class Child:
        tool_progress_callback = None
        _delegate_saved_tool_names = []
        _credential_pool = None
        _subagent_id = "sa-test"
        _delegate_depth = 1
        _parent_subagent_id = None
        model = "test-model"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_estimated_cost_usd = 0.0
        session_reasoning_tokens = 0

        def get_activity_summary(self):
            return {"api_call_count": 0, "max_iterations": 1, "current_tool": None}

        def run_conversation(self, user_message, task_id, **_kwargs):
            attempted = kanban_tools._handle_complete({"summary": "wrong child completion"})
            return {
                "final_response": attempted,
                "completed": True,
                "api_calls": 0,
                "messages": [],
            }

        def close(self):
            return None

    result = delegate_tool._run_single_child(0, "try to complete parent", Child(), Parent())

    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
    finally:
        conn.close()

    assert result["status"] == "completed"
    assert "delegate_task child" in result["summary"]
    assert task.status == "running"
    assert run.status == "running"
    assert workspace.is_dir()
