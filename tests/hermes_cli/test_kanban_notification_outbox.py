"""Persistence-foundation tests for Kanban notification outbox (issue #19)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_cli import kanban_db as kb

BOT_CHAT_TITLE = "Bot Chat"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _session_with_title(home: Path, profile: str, session_id: str, title: str) -> str:
    db_path = home / "profiles" / profile / "state.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id, "tui", profile_name=profile)
        assert db.set_session_title(session_id, title)
        row = db.get_session_by_title(title)
        assert row is not None
        return row["id"]
    finally:
        db.close()


def _canonical_bot_chat(home: Path, profile: str, session_id: str) -> str:
    return _session_with_title(home, profile, session_id, BOT_CHAT_TITLE)


def _default_canonical_bot_chat(home: Path, session_id: str) -> str:
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id, "tui", profile_name="default")
        assert db.set_session_title(session_id, BOT_CHAT_TITLE)
        return session_id
    finally:
        db.close()


def _outbox_rows(conn: sqlite3.Connection):
    return [dict(row) for row in conn.execute("SELECT * FROM kanban_notification_outbox ORDER BY id")]


def _event_kinds(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return [row["kind"] for row in conn.execute("SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,))]


def test_outbox_rows_are_inserted_in_the_same_transaction_as_task_events(kanban_home, monkeypatch):
    session_id = _canonical_bot_chat(kanban_home, "chief", "chief-bot-chat")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="ship persistence foundation",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=session_id,
            board="default",
        )
        assert kb.complete_task(conn, task_id, result="done") is True
        rows = _outbox_rows(conn)
        assert [row["event_kind"] for row in rows] == ["created", "completed"]
        assert [row["delivery_seq"] for row in rows] == [1, 2]
        completed = rows[-1]
        assert completed["task_id"] == task_id
        assert completed["origin_profile"] == "chief"
        assert completed["origin_session_id"] == session_id
        assert completed["board"] == "default"
        assert completed["task_event_id"] is not None

        def boom(*_args, **_kwargs):
            raise RuntimeError("outbox insert refused")

        retry_id = kb.create_task(
            conn,
            title="must roll back completion if outbox fails",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=session_id,
            board="default",
        )
        monkeypatch.setattr(kb, "_insert_notification_outbox", boom)
        with pytest.raises(RuntimeError, match="outbox insert refused"):
            kb.complete_task(conn, retry_id, result="should not commit")
        assert kb.get_task(conn, retry_id).status == "ready"
        assert "completed" not in _event_kinds(conn, retry_id)


def test_outbox_routing_is_fail_closed_by_profile_board_and_canonical_session(kanban_home):
    chief_session = _canonical_bot_chat(kanban_home, "chief", "chief-bot-chat")
    welby_session = _canonical_bot_chat(kanban_home, "welby", "welby-bot-chat")
    other_session = _session_with_title(kanban_home, "chief", "ordinary-chat", "Ordinary Chat")

    with kb.connect() as conn:
        no_origin = kb.create_task(conn, title="no origin", assignee="worker")
        kb.complete_task(conn, no_origin, result="silent")
        assert _outbox_rows(conn) == []

        with pytest.raises(ValueError, match="canonical Bot Chat"):
            kb.create_task(
                conn,
                title="wrong session",
                assignee="worker",
                origin_profile="chief",
                origin_session_id=other_session,
                board="default",
            )
        with pytest.raises(ValueError, match="does not match connection board"):
            kb.create_task(
                conn,
                title="spoofed board",
                assignee="worker",
                origin_profile="chief",
                origin_session_id=chief_session,
                board="other",
            )

        chief_task = kb.create_task(
            conn,
            title="chief owned",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=chief_session,
            board="default",
        )
        welby_task = kb.create_task(
            conn,
            title="welby owned",
            assignee="worker",
            origin_profile="welby",
            origin_session_id=welby_session,
            board="default",
        )
        kb.complete_task(conn, chief_task, result="chief done")
        kb.complete_task(conn, welby_task, result="welby done")

        chief_rows = kb.list_notification_outbox(conn, profile="chief", board="default")
        welby_rows = kb.list_notification_outbox(conn, profile="welby", board="default")
        assert {row["task_id"] for row in chief_rows} == {chief_task}
        assert {row["task_id"] for row in welby_rows} == {welby_task}
        assert kb.list_notification_outbox(conn, profile="chief", board="other") == []


def test_default_profile_origin_uses_root_session_db_and_conflicting_parents_fail_closed(kanban_home):
    default_session = _default_canonical_bot_chat(kanban_home, "default-bot-chat")
    chief_session = _canonical_bot_chat(kanban_home, "chief", "chief-bot-chat")
    with kb.connect() as conn:
        default_parent = kb.create_task(
            conn,
            title="default parent",
            assignee="worker",
            origin_profile="default",
            origin_session_id=default_session,
            board="default",
        )
        chief_parent = kb.create_task(
            conn,
            title="chief parent",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=chief_session,
            board="default",
        )
        default_task = kb.get_task(conn, default_parent)
        assert default_task is not None
        assert default_task.origin_profile == "default"
        assert default_task.origin_session_id == default_session
        assert default_task.origin_board == "default"
        with pytest.raises(ValueError, match="conflicting notification origins"):
            kb.create_task(
                conn,
                title="ambiguous child",
                assignee="worker",
                parents=[default_parent, chief_parent],
            )


def test_lifecycle_events_linked_children_and_legacy_subscriptions_feed_outbox(kanban_home):
    chief_session = _canonical_bot_chat(kanban_home, "chief", "chief-bot-chat")
    welby_session = _canonical_bot_chat(kanban_home, "welby", "welby-bot-chat")
    with kb.connect() as conn:
        parent = kb.create_task(
            conn,
            title="parent",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=chief_session,
            board="default",
        )
        assert kb.complete_task(conn, parent, result="parent prerequisite done") is True
        inherited_child = kb.create_task(conn, title="child inherits", assignee="worker", parents=[parent])
        override_child = kb.create_task(
            conn,
            title="child override",
            assignee="worker",
            parents=[parent],
            origin_profile="welby",
            origin_session_id=welby_session,
            board="default",
        )

        assert kb.block_task(conn, inherited_child, reason="need input", kind="needs_input") is True
        assert kb.request_review(conn, override_child, summary="please review", force=True) is True
        kb.unblock_task(conn, inherited_child)
        assert kb.block_task(conn, inherited_child, reason="again", kind="needs_input") is True

        legacy_task = kb.create_task(
            conn,
            title="legacy subscription",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=chief_session,
            board="default",
        )
        kb.add_notify_sub(
            conn,
            task_id=legacy_task,
            platform="telegram",
            chat_id="chat-1",
            thread_id="",
            user_id="user-1",
            notifier_profile="chief",
        )
        assert kb.complete_task(conn, legacy_task, result="legacy done") is True

        chief_rows = kb.list_notification_outbox(conn, profile="chief", board="default")
        welby_rows = kb.list_notification_outbox(conn, profile="welby", board="default")
        assert {"blocked", "block_loop_detected", "completed"}.issubset({row["event_kind"] for row in chief_rows})
        assert {row["task_id"] for row in chief_rows if row["event_kind"] in {"blocked", "block_loop_detected"}} == {inherited_child}
        assert any(row["event_kind"] == "review_requested" and row["task_id"] == override_child for row in welby_rows)
        assert any(row["platform"] == "telegram" and row["chat_id"] == "chat-1" and row["event_kind"] == "completed" for row in chief_rows)


def test_notification_ack_and_cursor_primitives_are_durable(kanban_home):
    session_id = _canonical_bot_chat(kanban_home, "chief", "chief-bot-chat")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="ack me",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=session_id,
            board="default",
        )
        kb.complete_task(conn, task_id, result="done")
        rows = kb.list_notification_outbox(conn, profile="chief", board="default")
        completed = rows[-1]
        assert kb.ack_notification_outbox(
            conn,
            completed["id"],
            consumer="desktop:main",
            delivery_key=completed["delivery_key"],
        ) is True
        kb.set_notification_cursor(conn, "desktop:main", "chief", "default", completed["id"])

    with kb.connect() as reopened:
        assert kb.get_notification_cursor(reopened, "desktop:main", "chief", "default") == completed["id"]
        acked = kb.list_notification_outbox(reopened, profile="chief", board="default", include_delivered=True)[-1]
        assert acked["delivered_at"] is not None
        assert acked["delivered_to"] == "desktop:main"
        assert kb.ack_notification_outbox(
            reopened,
            completed["id"],
            consumer="desktop:main",
            delivery_key=completed["delivery_key"],
        ) is True
        assert not any(
            row["id"] == completed["id"]
            for row in kb.list_notification_outbox(
                reopened,
                profile="chief",
                board="default",
                consumer="desktop:main",
            )
        )
        assert any(
            row["id"] == completed["id"]
            for row in kb.list_notification_outbox(
                reopened,
                profile="chief",
                board="default",
                consumer="tui:main",
            )
        )
        assert kb.ack_notification_outbox(
            reopened,
            completed["id"],
            consumer="tui:main",
            delivery_key=completed["delivery_key"],
        ) is True
        acknowledgements = reopened.execute(
            "SELECT consumer FROM kanban_notification_acks WHERE outbox_id = ? ORDER BY consumer",
            (completed["id"],),
        ).fetchall()
        assert [row["consumer"] for row in acknowledgements] == ["desktop:main", "tui:main"]


def test_every_owned_task_event_is_transactionally_represented_once_per_destination(kanban_home):
    session_id = _canonical_bot_chat(kanban_home, "chief", "chief-bot-chat")
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="all events",
            assignee="worker",
            origin_profile="chief",
            origin_session_id=session_id,
            board="default",
        )
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="bot-chat",
            chat_id="",
            thread_id="",
            notifier_profile="chief",
        )
        kb.add_comment(conn, task_id, "chief", "durable but silent by default")
        events = conn.execute(
            "SELECT id, kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
        outbox = conn.execute(
            "SELECT task_event_id, event_kind FROM kanban_notification_outbox WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
        assert [(row["task_event_id"], row["event_kind"]) for row in outbox] == [
            (row["id"], row["kind"]) for row in events
        ]
