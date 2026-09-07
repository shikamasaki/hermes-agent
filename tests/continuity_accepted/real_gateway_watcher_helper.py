from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


class _MinimalGatewayDispatcher:
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin


class _RealGatewayWatcher(_MinimalGatewayDispatcher.GatewayKanbanWatchersMixin):
    """Minimal product watcher host for isolated tests."""

    def __init__(self) -> None:
        self._running = True
        self._kanban_dispatcher_lock_handle = None


class RealGatewayWatcherThread:
    """Run the product GatewayKanbanWatchersMixin dispatcher watcher once.

    The helper owns only watcher lifetime and trace collection. It does not
    implement reap/dispatch logic; the product watcher calls kanban_db itself.
    """

    def __init__(self, trace: dict[str, Any]) -> None:
        self.trace = trace
        self.watcher = _RealGatewayWatcher()
        self.ready = threading.Event()
        self.errors: list[str] = []
        self._thread = threading.Thread(
            target=self._run,
            name="isolated-product-gateway-kanban-watcher",
            daemon=True,
        )

    def start(self) -> None:
        self.trace.setdefault("watcher", {})["route"] = (
            "gateway/kanban_watchers.py::GatewayKanbanWatchersMixin._kanban_dispatcher_watcher"
        )
        self.trace["watcher"]["provider"] = (
            "product GatewayKanbanWatchersMixin + hermes_cli.kanban_db.reap_worker_zombies/dispatch_once"
        )
        self.trace["watcher"]["thread_name"] = self._thread.name
        self._thread.start()
        if not self.ready.wait(timeout=2):
            raise AssertionError("real gateway watcher thread did not start")

    def stop(self) -> None:
        self.watcher._running = False
        self._thread.join(timeout=7)
        self.trace.setdefault("watcher", {})["running_flag_after_stop"] = self.watcher._running
        self.trace["watcher"]["thread_alive_after_stop"] = self._thread.is_alive()
        self.trace["watcher"]["lock_handle_after_stop"] = (
            getattr(self.watcher, "_kanban_dispatcher_lock_handle", None) is not None
        )
        if self._thread.is_alive():
            raise AssertionError("real gateway watcher thread did not stop")
        if self.errors:
            raise AssertionError("real gateway watcher failed: " + "; ".join(self.errors))

    def _run(self) -> None:
        async def _runner() -> None:
            self.ready.set()
            await self.watcher._kanban_dispatcher_watcher()

        try:
            asyncio.run(_runner())
        except BaseException as exc:  # keep error available to the test thread
            self.errors.append(repr(exc))


@contextlib.contextmanager
def run_product_gateway_watcher(trace: dict[str, Any]):
    watcher = RealGatewayWatcherThread(trace)
    watcher.start()
    try:
        yield watcher
    finally:
        watcher.stop()


def configure_abnormal_real_watcher_environment(monkeypatch: Any, hermes_home: Path) -> None:
    """Pin all watcher-controlled paths to the test's isolated root."""
    os_home = hermes_home.parent / "os-home"
    os_home.mkdir(parents=True, exist_ok=True)
    # Keep HOME distinct so the live-db guard does not collide with hermes_home.
    monkeypatch.setenv("HOME", str(os_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(hermes_home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "30")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.delenv("HERMES_STATE_DB_GUARD_BYPASS", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)


def append_real_watcher_config(hermes_home: Path) -> None:
    config_path = hermes_home / "config.yaml"
    with config_path.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n"
            "kanban:\n"
            "  dispatch_in_gateway: true\n"
            "  dispatch_interval_seconds: 1\n"
            "  auto_decompose: false\n"
            "  review_dispatch: false\n"
            "  max_spawn: 0\n"
            "  max_in_progress: 1\n"
            "  failure_limit: 2\n"
            "  reconcile_orphans: true\n"
        )


def assert_boards_are_inside_temp_root(hermes_home: Path) -> list[dict[str, Any]]:
    from hermes_cli import kanban_db as kb

    boards = kb.list_boards(include_archived=False)
    root = hermes_home.resolve()
    paths: list[str] = []
    for board in boards:
        db_path = board.get("db_path")
        resolved = Path(db_path).expanduser().resolve() if db_path else kb.kanban_db_path(board.get("slug") or kb.DEFAULT_BOARD).resolve()
        paths.append(str(resolved))
        if not resolved.is_relative_to(root):
            raise AssertionError({"board_paths": paths, "hermes_home": str(root)})
    if not paths:
        raise AssertionError("kanban board discovery returned no boards")
    return boards


def start_claimed_nonzero_worker(hermes_home: Path, task_id: str, *, exit_code: int = 42) -> tuple[dict[str, Any], subprocess.Popen]:
    """Claim a task, start a real child, and record its pid in Kanban.

    After this function returns, callers must only observe durable state. The
    product watcher is responsible for waitpid/reap and crashed-event storage.
    """
    from hermes_cli import kanban_db as kb
    from .harness import run_abnormal_kanban_child

    with kb.connect() as conn:
        claimed = kb.claim_task(conn, task_id, ttl_seconds=30)
        if claimed is None:
            raise AssertionError(f"failed to claim abnormal child task {task_id}")
        claimer_id = kb._claimer_id()
        task_after_claim = kb.get_task(conn, task_id)
        if task_after_claim is None:
            raise AssertionError(f"claimed abnormal child task {task_id} disappeared before pid assignment")
        assert task_after_claim.claim_lock == claimer_id, {
            "task_id": task_id,
            "claim_lock": task_after_claim.claim_lock,
            "claimer_id": claimer_id,
        }
        proc = run_abnormal_kanban_child(hermes_home, task_id, exit_code=exit_code)
        kb._set_worker_pid(conn, task_id, proc.pid)
        task_after_pid = kb.get_task(conn, task_id)
        if task_after_pid is None:
            raise AssertionError(f"claimed abnormal child task {task_id} disappeared after pid assignment")
        assert task_after_pid.claim_lock == claimer_id, {
            "task_id": task_id,
            "claim_lock": task_after_pid.claim_lock,
            "claimer_id": claimer_id,
        }
    return {
        "pid": proc.pid,
        "returncode_expected": exit_code,
        "claimer_id": claimer_id,
        "claim_lock_after_claim": task_after_claim.claim_lock,
        "claim_lock_after_set_worker_pid": task_after_pid.claim_lock,
        "claim_lock_matches_claimer_id": True,
        "status_after_claim": task_after_claim.status,
        "status_after_set_worker_pid": task_after_pid.status,
        "worker_pid_after_set": task_after_pid.worker_pid,
        "relation": "worker pid belongs to a subprocess child of the process running the product watcher; the test never poll()/wait()s it before watcher reclaim",
    }, proc


def ps_snapshot(pid: int) -> dict[str, Any]:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,ppid=,stat=,command="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=3,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def read_crash_state(task_id: str, dependent_task_id: str) -> dict[str, Any] | None:
    from hermes_cli import kanban_db as kb

    with kb.connect() as conn:
        child_after = kb.get_task(conn, task_id)
        dependent_after = kb.get_task(conn, dependent_task_id)
        crashed_row = conn.execute(
            "SELECT id, kind, payload, run_id FROM task_events "
            "WHERE task_id = ? AND kind = 'crashed' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        completed_count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (task_id,),
        ).fetchone()[0]
        run_row = conn.execute(
            "SELECT status, outcome, error FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        sub = kb.list_notify_subs(conn, task_id)[0]
    if crashed_row is None or run_row is None:
        return None
    return {
        "child_after": child_after,
        "dependent_after": dependent_after,
        "crashed_row": crashed_row,
        "crashed_payload": json.loads(crashed_row["payload"] or "{}"),
        "completed_count": completed_count,
        "run_row": run_row,
        "sub": sub,
    }
