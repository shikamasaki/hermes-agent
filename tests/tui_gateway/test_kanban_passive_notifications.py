from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_client_delivery import NotificationSignalListener, publish_profile_signal
from hermes_state import SessionDB
from tui_gateway import kanban_notifications as inbox


class FakeTransport:
    def __init__(self):
        self.frames: list[dict] = []

    def write(self, frame: dict) -> bool:
        self.frames.append(frame)
        return True

    def close(self) -> None:
        pass


@pytest.fixture
def owned_board(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "chief"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    session_id = "chief-bot-chat"
    db = SessionDB(db_path=profile_home / "state.db")
    db.create_session(session_id, "tui", profile_name="chief")
    db.set_session_title(session_id, "Bot Chat")
    before = db.get_messages_as_conversation(session_id)
    db.close()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="passive delivery",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=session_id,
            board="default",
        )
        assert kb.complete_task(conn, task_id, result="done", summary="shipped")
        completed = next(
            row for row in kb.list_notification_outbox(
                conn, profile="chief", board="default", consumer="tui:chief"
            ) if row["event_kind"] == "completed"
        )
    inbox._SUBSCRIBERS.clear()
    inbox._SENT.clear()
    yield root, profile_home, session_id, task_id, completed, before
    inbox._SUBSCRIBERS.clear()
    inbox._SENT.clear()


def test_subscribe_replays_card_without_history_or_turn_state_and_ack_is_idempotent(owned_board):
    _, profile_home, session_id, _, row, before = owned_board
    transport = FakeTransport()

    assert inbox.subscribe(transport, surface="tui", session_id=session_id) == 1
    event = transport.frames[0]["params"]
    assert event["type"] == "kanban.notification"
    assert event["session_id"] == session_id
    assert event["payload"]["event_kind"] == "completed"
    assert "shipped" in (event["payload"]["summary"] or "")

    db = SessionDB(db_path=profile_home / "state.db")
    assert db.get_messages_as_conversation(session_id) == before
    db.close()

    assert inbox.acknowledge(
        surface="tui", board="default", outbox_id=row["id"], delivery_key=row["delivery_key"]
    )
    assert inbox.acknowledge(
        surface="tui", board="default", outbox_id=row["id"], delivery_key=row["delivery_key"]
    )
    inbox.unsubscribe(transport)
    assert inbox.subscribe(FakeTransport(), surface="tui", session_id=session_id) == 0


def test_duplicate_signal_is_suppressed_and_other_profile_cannot_consume(owned_board):
    _, _, session_id, _, row, _ = owned_board
    transport = FakeTransport()
    inbox.subscribe(transport, surface="desktop", session_id=session_id)
    transport.frames.clear()
    signal = {"outbox": [{"board": "default", "outbox_id": row["id"], "task_id": row["task_id"]}]}

    # Already replayed on subscribe, so duplicate committed signals do not add a card.
    assert inbox.deliver_signal(signal) == 0
    assert inbox.deliver_signal(signal) == 0
    assert transport.frames == []

    # Rebinding the process to a different profile yields no cross-profile replay.
    old = __import__("os").environ["HERMES_HOME"]
    __import__("os").environ["HERMES_HOME"] = str(Path(old).parent / "welby")
    try:
        other = FakeTransport()
        with pytest.raises(ValueError, match="canonical Bot Chat"):
            inbox.subscribe(other, surface="tui", session_id=session_id)
        assert other.frames == []
    finally:
        __import__("os").environ["HERMES_HOME"] = old


def test_loopback_signal_pushes_immediately_without_polling(tmp_path, monkeypatch):
    home = tmp_path / ".hermes" / "profiles" / "chief"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    received: list[dict] = []
    listener = NotificationSignalListener(received.append, home=home)
    try:
        payload = {"outbox": [{"board": "default", "outbox_id": 7, "task_id": "t_7"}]}
        assert publish_profile_signal("chief", payload, root=tmp_path / ".hermes") == 1
        deadline = time.monotonic() + 2
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received == [payload]
    finally:
        listener.close()


def test_blocked_review_and_completed_cards_are_scoped_to_exact_session(owned_board):
    _, _, session_id, _, _, _ = owned_board
    with kb.connect() as conn:
        blocked_id = kb.create_task(
            conn,
            title="blocked",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=session_id,
            board="default",
        )
        review_id = kb.create_task(
            conn,
            title="review",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=session_id,
            board="default",
        )
        assert kb.block_task(conn, blocked_id, reason="human decision", kind="needs_input")
        assert kb.request_review(conn, review_id, summary="inspect me", force=True)

    wrong = FakeTransport()
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        inbox.subscribe(wrong, surface="tui", session_id="ordinary-chat")
    assert wrong.frames == []

    exact = FakeTransport()
    assert inbox.subscribe(exact, surface="tui", session_id=session_id) == 3
    kinds = [frame["params"]["payload"]["event_kind"] for frame in exact.frames]
    assert kinds == ["completed", "blocked", "review_requested"]


def test_json_rpc_transport_replays_then_acks_across_reconnect(owned_board):
    from tui_gateway import server

    _, _, session_id, _, row, _ = owned_board
    first = FakeTransport()
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "subscribe-1",
            "method": "kanban.notifications.subscribe",
            "params": {"surface": "desktop", "session_id": session_id},
        },
        first,
    )
    assert response == {
        "jsonrpc": "2.0",
        "id": "subscribe-1",
        "result": {"subscribed": True, "replayed": 1},
    }
    payload = first.frames[0]["params"]["payload"]
    assert payload["delivery_key"] == row["delivery_key"]
    assert payload["task_title"] == "passive delivery"

    ack = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "ack-1",
            "method": "kanban.notifications.ack",
            "params": {
                "surface": "desktop",
                "board": "default",
                "outbox_id": row["id"],
                "delivery_key": row["delivery_key"],
            },
        },
        first,
    )
    assert ack is not None
    assert ack["result"] == {"acknowledged": True}
    inbox.unsubscribe(first)

    second = FakeTransport()
    replay = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "subscribe-2",
            "method": "kanban.notifications.subscribe",
            "params": {"surface": "desktop", "session_id": session_id},
        },
        second,
    )
    assert replay is not None
    assert replay["result"]["replayed"] == 0
    assert second.frames == []
