from __future__ import annotations

import asyncio
import time

import pytest

from gateway.control_socket import GatewayControlServer, query_gateway_control
from gateway.kanban_event_loop import KanbanSignalBus
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from hermes_cli import kanban_db as kb


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX live socket test")
def test_real_control_socket_delivers_named_outbox_signal(tmp_path):
    async def scenario() -> None:
        bus = KanbanSignalBus(asyncio.get_running_loop())
        server = GatewayControlServer(
            tmp_path,
            verb_handlers={"kanban_outbox_ready": bus.handle_signal},
        )
        assert await server.start()
        payload = {
            "outbox": [
                {
                    "board": "alpha",
                    "outbox_id": 7,
                    "task_id": "t_one",
                    "event_id": 11,
                    "event_kind": "completed",
                    "delivery_key": "alpha:11:1",
                }
            ]
        }
        try:
            reply = await asyncio.to_thread(
                query_gateway_control,
                tmp_path,
                "kanban_outbox_ready",
                payload=payload,
            )
            assert reply == {"accepted": 1}
            batch = await asyncio.wait_for(bus.next_batch(), timeout=1)
            assert batch.boards == {"alpha"}
            assert batch.outbox_ids == {7}
            assert batch.task_ids == {"t_one"}
            assert batch.outbox_by_board == {"alpha": {7}}
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_duplicate_signals_coalesce_without_losing_named_ids():
    async def scenario() -> None:
        bus = KanbanSignalBus(asyncio.get_running_loop())
        payload = {
            "outbox": [
                {"board": "default", "outbox_id": 3, "task_id": "t_a"},
            ]
        }
        assert bus.handle_signal(payload) == {"accepted": 1}
        assert bus.handle_signal(payload) == {"accepted": 1}
        batch = await asyncio.wait_for(bus.next_batch(), timeout=1)
        assert batch.outbox_ids == {3}
        assert batch.task_ids == {"t_a"}
        assert bus.empty()

    asyncio.run(scenario())


def test_persisted_nearest_deadline_rebuild_and_due_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    conn = kb.connect()
    try:
        now = int(time.time())
        task_id = kb.create_task(
            conn,
            title="deadline",
            assignee="default",
            initial_status="running",
            max_runtime_seconds=30,
        )
        conn.execute(
            "UPDATE tasks SET status = 'running', claim_expires = ?, started_at = ?, last_heartbeat_at = ? WHERE id = ?",
            (now + 10, now, now, task_id),
        )
        conn.commit()

        rebuilt = kb.rebuild_kanban_deadlines(
            conn,
            stale_timeout_seconds=60,
            retention_days=30,
            now=now,
        )
        assert rebuilt >= 3
        assert kb.next_kanban_deadline(conn) == now + 10
        assert kb.pop_due_kanban_deadlines(conn, now=now + 9) == []
        due = kb.pop_due_kanban_deadlines(conn, now=now + 10)
        assert [(row["kind"], row["task_id"]) for row in due] == [
            ("claim_expiry", task_id)
        ]
        assert kb.next_kanban_deadline(conn) == now + 30
    finally:
        conn.close()


def test_rate_limit_retry_is_a_persisted_scheduled_start(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "5")
    conn = kb.connect()
    try:
        now = int(time.time())
        task_id = kb.create_task(conn, title="retry", assignee="default")
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome) "
            "VALUES (?, 'default', 'released', ?, ?, 'rate_limited')",
            (task_id, now - 1, now),
        )
        conn.commit()
        kb.rebuild_kanban_deadlines(conn, now=now)
        row = conn.execute(
            "SELECT kind, task_id, due_at FROM kanban_deadlines WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        assert dict(row) == {
            "kind": "scheduled_start",
            "task_id": task_id,
            "due_at": now + 5,
        }
    finally:
        conn.close()


def test_dispatcher_runs_one_immediate_pass_for_signaled_board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.connect().close()

    import gateway.kanban_watchers as watchers
    import hermes_cli.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "auto_decompose": False,
                "reconcile_orphans": False,
            }
        },
    )
    monkeypatch.setattr(
        watchers, "_acquire_singleton_lock", lambda path: (None, "unavailable")
    )
    calls: list[str] = []

    def fake_dispatch_once(conn, *, board=None, **kwargs):
        calls.append(board)
        if len(calls) == 2:
            runner._running = False
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)

    class Runner(GatewayKanbanWatchersMixin):
        _running = True

    async def scenario() -> None:
        nonlocal runner
        runner = Runner()
        bus = KanbanSignalBus(asyncio.get_running_loop())
        runner._kanban_signal_bus = bus
        task = asyncio.create_task(runner._kanban_dispatcher_watcher())
        while len(calls) < 1:
            await asyncio.sleep(0)
        assert calls == ["default"]  # bounded startup recovery
        bus.handle_signal({
            "outbox": [{
                "board": "default",
                "outbox_id": 99,
                "task_id": "t_signal",
            }]
        })
        await asyncio.wait_for(task, timeout=2)
        assert calls == ["default", "default"]

    runner = None
    asyncio.run(scenario())


def test_outbox_adapter_delivery_acks_once_and_duplicate_signal_is_harmless(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    conn = kb.connect()
    task_id = kb.create_task(conn, title="deliver", assignee="default")
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="chat-1",
        notifier_profile="default",
    )
    assert kb.complete_task(conn, task_id, summary="finished")
    row = conn.execute(
        "SELECT * FROM kanban_notification_outbox WHERE task_id = ? AND platform = 'telegram'",
        (task_id,),
    ).fetchone()
    assert row is not None
    outbox_id = int(row["id"])
    conn.close()

    class Adapter:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, chat_id, text, metadata=None):
            self.sent.append(text)
            runner._running = False
            payload = {
                "outbox": [{
                    "board": "default",
                    "outbox_id": outbox_id,
                    "task_id": task_id,
                }]
            }
            bus.handle_signal(payload)
            bus.handle_signal(payload)
            return None

    class Runner(GatewayKanbanWatchersMixin):
        _running = True
        _profile_adapters = {}

        def _active_profile_name(self):
            return "default"

        def _authorization_adapter(self, platform, profile=None):
            return adapter

    async def scenario() -> None:
        nonlocal runner, adapter, bus
        bus = KanbanSignalBus(asyncio.get_running_loop())
        adapter = Adapter()
        runner = Runner()
        runner._kanban_signal_bus = bus
        await asyncio.wait_for(runner._kanban_outbox_notifier_watcher(), timeout=2)
        assert len(adapter.sent) == 1

        # A fresh startup recovery sees the durable ack and sends nothing.
        runner2 = Runner()
        runner2._running = True
        runner2._kanban_signal_bus = KanbanSignalBus(asyncio.get_running_loop())
        runner2._authorization_adapter = lambda platform, profile=None: adapter
        task = asyncio.create_task(runner2._kanban_outbox_notifier_watcher())
        await asyncio.sleep(0.05)
        runner2._running = False
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(adapter.sent) == 1

    runner = None
    adapter = None
    bus = None
    asyncio.run(scenario())
