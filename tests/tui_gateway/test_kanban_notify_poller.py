"""Tests for the TUI-side kanban notification poller (issue #59890).

``kanban_create`` auto-subscribes TUI/desktop sessions with
``platform="tui"`` / ``chat_id=HERMES_SESSION_KEY``, but no component ever
read those rows back: the gateway notifier skips them (no "tui" messaging
adapter) and the TUI notification poller only watched process completions.
``last_event_id`` stayed 0 forever and no notification was ever delivered.

These tests cover the delivery half that now lives in tui_gateway/server.py:
``_collect_kanban_notifications`` (non-destructive read + formatting + ack-time
unsubscribe) and ``_format_kanban_event_text``.
"""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import kanban_db as kb
from tui_gateway.server import (
    _collect_kanban_notifications,
    _collect_kanban_notifications_with_claims,
    _format_kanban_event_text,
)

SESSION_KEY = "tui-session-key-1"


def _session(key: str = SESSION_KEY) -> dict:
    return {"session_key": key}


def _create_subscribed_task(*, chat_id: str = SESSION_KEY, platform: str = "tui"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify tui", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform=platform, chat_id=chat_id)
        return tid
    finally:
        conn.close()


def _complete(tid: str, summary: str = "all done") -> None:
    conn = kb.connect()
    try:
        kb.complete_task(conn, tid, summary=summary)
    finally:
        conn.close()


def _sub_rows(tid: str) -> list:
    conn = kb.connect()
    try:
        return kb.list_notify_subs(conn, task_id=tid)
    finally:
        conn.close()


class TestCollectKanbanNotifications:
    def test_zero_sub_board_is_never_opened_writable(self):
        conn = kb.connect()
        conn.close()
        kb.create_board("second-board")

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()

    def test_done_reopen_notifies_once_per_event_until_archive(self):
        tid = _create_subscribed_task()
        _complete(tid, summary="shipped the fix")

        first = _collect_kanban_notifications(_session())

        assert len(first) == 1
        assert tid in first[0]
        assert "done" in first[0]
        assert "shipped the fix" in first[0]
        rows = _sub_rows(tid)
        assert len(rows) == 1, "done must retain the originating session"
        first_cursor = rows[0]["last_event_id"]

        # The retained subscription must not replay the completed event.
        assert _collect_kanban_notifications(_session()) == []

        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,)
                )
                kb._append_event(conn, tid, "status", {"status": "ready"})
            assert kb.complete_task(conn, tid, summary="review corrections")
        finally:
            conn.close()

        reopened = _collect_kanban_notifications(_session())

        assert len(reopened) == 2
        assert "ready" in reopened[0]
        assert "review corrections" in reopened[1]
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["chat_id"] == SESSION_KEY
        assert rows[0]["last_event_id"] > first_cursor
        assert _collect_kanban_notifications(_session()) == []

        conn = kb.connect()
        try:
            assert kb.archive_task(conn, tid)
        finally:
            conn.close()

        # Archive is notification-terminal and removes the retained route.
        assert _collect_kanban_notifications(_session()) == []
        assert _sub_rows(tid) == []

    def test_matching_tui_sub_delivers_and_advances_cursor(self):
        tid = _create_subscribed_task()
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        conn = kb.connect()
        try:
            kb.block_task(conn, tid, reason="waiting on review", kind="needs_input")
        finally:
            conn.close()

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            first = _collect_kanban_notifications(_session())
            second = _collect_kanban_notifications(_session())

        assert len(first) == 1
        assert "blocked" in first[0]
        assert "waiting on review" in first[0]
        assert second == []
        assert spy_connect.called
        # Blocked is not a final status -> subscription stays alive so a
        # respawned task's next terminal event still reaches the user.
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] > pre_cursor

    def test_non_tui_subscription_does_not_open_board_writable(self):
        tid = _create_subscribed_task(platform="telegram", chat_id="chat-1")
        # New subs start caught up at creation time (issue #29905); record the
        # pre-completion cursors so we can assert they were never claimed.
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        _complete(tid)

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] == pre_cursor

    def test_other_tui_session_does_not_open_board_writable(self):
        tid = _create_subscribed_task(chat_id="some-other-session")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        _complete(tid)

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] == pre_cursor

    def test_probe_error_falls_back_to_writable_delivery(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="fallback delivery")

        def fail_probe(*args, **kwargs):
            raise OSError("probe unavailable")

        monkeypatch.setattr(kb, "count_notify_subs", fail_probe)
        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert len(texts) == 1
        assert tid in texts[0]
        spy_connect.assert_called_once()

    def test_no_session_key_is_a_noop(self):
        tid = _create_subscribed_task()
        _complete(tid)

        assert _collect_kanban_notifications({"session_key": ""}) == []
        assert _collect_kanban_notifications({"session_key": None}) == []
        assert len(_sub_rows(tid)) == 1

    def test_profile_scoped_session_reads_the_shared_board(self, tmp_path):
        """The kanban board is shared across profiles BY DESIGN (see the
        hermes_cli/kanban_db.py module docstring): ``kanban_home()`` anchors on
        ``get_default_hermes_root()``, which resolves the process env and
        ignores context-local profile overrides. A Desktop session bound to a
        non-launch profile (``session["profile_home"]``) must therefore still
        have its subscription claimed from the one shared board — the poller
        needs no per-profile home binding.
        """
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        tid = _create_subscribed_task()
        _complete(tid, summary="cross-profile delivery")

        other_profile_home = tmp_path / "profiles" / "reviewer"
        other_profile_home.mkdir(parents=True)
        session = {
            "session_key": SESSION_KEY,
            "profile_home": str(other_profile_home),
        }
        # Simulate the strictest case: a context-local profile override is
        # active while the poller collects (as a profile-bound RPC would set).
        token = set_hermes_home_override(str(other_profile_home))
        try:
            texts = _collect_kanban_notifications(session)
        finally:
            reset_hermes_home_override(token)

        assert len(texts) == 1
        assert tid in texts[0]
        assert "cross-profile delivery" in texts[0]
        # Completion is reversible, so the shared-board subscription remains
        # owned by this exact Desktop session until the task is archived.
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["chat_id"] == SESSION_KEY

    def test_unaccepted_claim_survives_session_loss_in_sqlite(self):
        import os
        from pathlib import Path

        hermes_home = Path(os.environ["HERMES_HOME"]).resolve()
        assert kb.kanban_db_path().resolve().is_relative_to(hermes_home)

        tid = _create_subscribed_task()
        _complete(tid, summary="session vanished before accept")

        claimed = _collect_kanban_notifications_with_claims(_session())
        assert len(claimed) == 1
        assert tid in claimed[0].text

        recovered = _collect_kanban_notifications(_session())
        assert len(recovered) == 1
        assert tid in recovered[0]
        assert "session vanished before accept" in recovered[0]

    def test_unaccepted_archived_task_notification_survives_session_loss(self):
        tid = _create_subscribed_task()
        _complete(tid, summary="done before archive")
        conn = kb.connect()
        try:
            assert kb.archive_task(conn, tid)
        finally:
            conn.close()

        claimed = _collect_kanban_notifications_with_claims(_session())
        assert len(claimed) == 1
        assert tid in claimed[0].text
        assert "done before archive" in claimed[0].text
        assert len(_sub_rows(tid)) == 1

        recovered = _collect_kanban_notifications(_session())
        assert len(recovered) == 1
        assert tid in recovered[0]
        assert "done before archive" in recovered[0]
        assert _sub_rows(tid) == []


class TestFormatKanbanEventText:
    SUB = {"task_id": "t_abc123"}
    TASK = SimpleNamespace(title="build the thing", assignee="worker", result=None)

    def test_silent_kinds_return_none(self):
        for kind in ("archived", "unblocked"):
            ev = SimpleNamespace(kind=kind, payload={})
            assert _format_kanban_event_text(self.SUB, self.TASK, ev, "main") is None

    def test_blocked_includes_reason(self):
        ev = SimpleNamespace(kind="blocked", payload={"reason": "needs creds"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "main")
        assert "t_abc123" in text
        assert "blocked" in text
        assert "needs creds" in text
        assert "[main]" in text
        assert "@worker" in text

    def test_completed_prefers_payload_summary(self):
        ev = SimpleNamespace(kind="completed", payload={"summary": "first line\nsecond"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "")
        assert "done" in text
        assert "first line" in text
        assert "second" not in text

    def test_timed_out_with_bad_payload_does_not_raise(self):
        ev = SimpleNamespace(kind="timed_out", payload={"limit_seconds": "not-a-number"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "")
        assert "timed out" in text


class TestNotificationPollerLoopKanbanWiring:
    """Drive a real TUI subscription through ``_notification_poller_loop``.

    Covers the wiring above ``_collect_kanban_notifications``: status.update
    emission, agent-turn dispatch when the session is idle, and the
    busy-session pending buffer that flushes once the session goes idle.
    """

    def _start_poller(self, session: dict, monkeypatch):
        import threading
        import tui_gateway.server as server

        emits: list = []
        submits: list = []
        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            server, "_emit", lambda event, sid, payload=None: emits.append((event, payload))
        )
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda rid, sid, sess, text: submits.append(text),
        )
        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        return stop, thread, emits, submits

    @staticmethod
    def _wait_for(predicate, timeout: float = 5.0) -> bool:
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if predicate():
                return True
            _time.sleep(0.02)
        return False

    def _poller_session(self, *, running: bool = False) -> dict:
        import threading

        return {
            "session_key": SESSION_KEY,
            "history_lock": threading.Lock(),
            "running": running,
        }

    def test_idle_session_gets_status_update_and_agent_turn(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="poller e2e done")
        session = self._poller_session(running=False)

        stop, thread, emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(lambda: submits), "agent turn was never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        status_texts = [p["text"] for e, p in emits if e == "status.update" and p]
        assert any(tid in t for t in status_texts), status_texts
        assert any(e == "message.start" for e, _ in emits)
        assert any(tid in text for text in submits), submits
        assert session["running"] is True  # poller claimed the turn
        assert not session.get("_kanban_pending")

    def test_busy_session_does_not_claim_until_idle(self, monkeypatch):
        import time as _time

        tid = _create_subscribed_task()
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        _complete(tid, summary="not claimed while busy")
        session = self._poller_session(running=True)

        stop, thread, emits, submits = self._start_poller(session, monkeypatch)
        try:
            _time.sleep(0.15)
            assert not any(e == "status.update" for e, _ in emits)
            assert not submits
            assert not session.get("_kanban_pending")
            assert _sub_rows(tid)[0]["last_event_id"] == pre_cursor

            with session["history_lock"]:
                session["running"] = False

            assert self._wait_for(lambda: submits), "idle session never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert any(tid in text for text in submits), submits
        assert "not claimed while busy" in submits[0]
        assert not session.get("_kanban_pending")
        assert session["running"] is True

    def test_failed_agent_turn_rewinds_claim_for_reconnected_session(self, monkeypatch):
        import threading
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="submit returned false")
        session = self._poller_session(running=False)
        submits: list[str] = []

        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

        def fail_submit(rid, sid, sess, text):
            submits.append(text)
            return False

        monkeypatch.setattr(server, "_run_prompt_submit", fail_submit)

        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        try:
            assert self._wait_for(lambda: submits), "agent turn was never attempted"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert not session.get("_kanban_pending")
        assert session["running"] is False

        reconnect_session = self._poller_session(running=False)
        reconnect_submits: list[str] = []
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda rid, sid, sess, text: reconnect_submits.append(text),
        )
        reconnect_stop = threading.Event()
        reconnect_thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(reconnect_stop, "sid-poller-reconnect", reconnect_session),
            daemon=True,
        )
        reconnect_thread.start()
        try:
            assert self._wait_for(lambda: reconnect_submits), "reconnected session did not retry"
        finally:
            reconnect_stop.set()
            reconnect_thread.join(timeout=5)

        assert len(reconnect_submits) == 1
        assert tid in reconnect_submits[0]
        assert "submit returned false" in reconnect_submits[0]
        assert _collect_kanban_notifications(_session()) == []

    def test_same_sid_rejected_turn_does_not_repeat_status_update(self, monkeypatch):
        import threading
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="same sid keeps refusing")
        session = self._poller_session(running=False)
        emits: list[tuple[str, dict | None]] = []
        submits: list[str] = []

        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            server, "_emit", lambda event, sid, payload=None: emits.append((event, payload))
        )
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda sid, sess: "owned")

        def refuse_submit(rid, sid, sess, text):
            submits.append(text)
            return False

        monkeypatch.setattr(server, "_run_prompt_submit", refuse_submit)

        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        try:
            assert self._wait_for(lambda: submits), submits
            import time as _time

            _time.sleep(0.15)
        finally:
            stop.set()
            thread.join(timeout=5)

        status_texts = [p["text"] for e, p in emits if e == "status.update" and p]
        repeated_status_texts = [text for text in status_texts if tid in text]
        assert len(repeated_status_texts) == 1
        assert "same sid keeps refusing" in repeated_status_texts[0]
        assert len([e for e, _ in emits if e == "message.start"]) == 1
        assert len(submits) == 1
        assert tid in submits[0]
        assert session["running"] is False
        assert not session.get("_kanban_pending")

    def test_same_sid_refusal_can_recover_without_repeating_display(self, monkeypatch):
        import threading
        import time as _time
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="same sid ownership recovers")
        session = self._poller_session(running=False)
        session.update(
            {
                "agent": SimpleNamespace(
                    session_id="agent-session",
                    clear_interrupt=lambda: None,
                ),
                "history": [],
                "history_version": 0,
                "cwd": "/Users/shikama",
            }
        )
        emits: list[tuple[str, dict | None]] = []
        ensure_calls = 0

        def ensure_then_recover(sid, sess):
            nonlocal ensure_calls
            ensure_calls += 1
            if ensure_calls <= 2:
                return "owned by another screen"
            sess["active_session_lease"] = object()
            return None

        class FakeThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        real_thread = threading.Thread
        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            server, "_emit", lambda event, sid, payload=None: emits.append((event, payload))
        )
        monkeypatch.setattr(server, "_ensure_active_session_slot", ensure_then_recover)
        stop = threading.Event()
        thread = real_thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        with server._sessions_lock:
            old_registered = server._sessions.get("sid-poller-test")
            server._sessions["sid-poller-test"] = session
        monkeypatch.setattr(server.threading, "Thread", FakeThread)
        thread.start()
        try:
            assert self._wait_for(lambda: ensure_calls >= 3)
            _time.sleep(0.05)
        finally:
            stop.set()
            thread.join(timeout=5)
            with server._sessions_lock:
                if old_registered is None:
                    server._sessions.pop("sid-poller-test", None)
                else:
                    server._sessions["sid-poller-test"] = old_registered

        status_texts = [p["text"] for e, p in emits if e == "status.update" and p]
        repeated_status_texts = [text for text in status_texts if tid in text]
        assert len(repeated_status_texts) == 1
        assert "same sid ownership recovers" in repeated_status_texts[0]
        assert len([e for e, _ in emits if e == "error"]) == 1
        assert len([e for e, _ in emits if e == "message.start"]) == 2
        assert ensure_calls >= 3
        assert _collect_kanban_notifications(_session()) == []

    def test_status_emit_exception_keeps_db_unaccepted_and_releases_running(self, monkeypatch):
        import threading
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="status emit failed")
        session = self._poller_session(running=False)

        def fail_status_emit(event, sid, payload=None):
            if event == "status.update":
                raise RuntimeError("status failed")

        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(server, "_emit", fail_status_emit)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda rid, sid, sess, text: (_ for _ in ()).throw(
                AssertionError("submit must not run after status emit failure")
            ),
        )

        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        try:
            assert self._wait_for(lambda: not thread.is_alive() or not session.get("running"))
        finally:
            stop.set()
            thread.join(timeout=5)

        assert session["running"] is False

        reconnect_session = self._poller_session(running=False)
        reconnect_submits: list[str] = []
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda rid, sid, sess, text: reconnect_submits.append(text),
        )
        reconnect_stop = threading.Event()
        reconnect_thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(reconnect_stop, "sid-poller-reconnect", reconnect_session),
            daemon=True,
        )
        reconnect_thread.start()
        try:
            assert self._wait_for(lambda: reconnect_submits), "reconnected session did not retry"
        finally:
            reconnect_stop.set()
            reconnect_thread.join(timeout=5)

        assert len(reconnect_submits) == 1
        assert tid in reconnect_submits[0]
        assert "status emit failed" in reconnect_submits[0]
        assert _collect_kanban_notifications(_session()) == []

    def test_agent_turn_exception_rewinds_claim_for_reconnected_session(self, monkeypatch):
        import threading
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="submit raised")
        session = self._poller_session(running=False)
        submits: list[str] = []

        def fail_submit(rid, sid, sess, text):
            submits.append(text)
            sess["_finalized"] = True
            raise RuntimeError("submit failed")

        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
        monkeypatch.setattr(server, "_run_prompt_submit", fail_submit)

        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        try:
            assert self._wait_for(lambda: submits), "agent turn was never attempted"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert not session.get("_kanban_pending")
        assert session["running"] is False

        reconnect_session = self._poller_session(running=False)
        reconnect_submits: list[str] = []
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda rid, sid, sess, text: reconnect_submits.append(text),
        )
        reconnect_stop = threading.Event()
        reconnect_thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(reconnect_stop, "sid-poller-reconnect", reconnect_session),
            daemon=True,
        )
        reconnect_thread.start()
        try:
            assert self._wait_for(lambda: reconnect_submits), "reconnected session did not retry"
        finally:
            reconnect_stop.set()
            reconnect_thread.join(timeout=5)

        assert len(reconnect_submits) == 1
        assert tid in reconnect_submits[0]
        assert "submit raised" in reconnect_submits[0]
        assert _collect_kanban_notifications(_session()) == []
