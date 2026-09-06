import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals
    import hermes_cli.loops as loops

    goals._DB_CACHE.clear()
    getattr(loops, "_DB_CACHE", {}).clear()
    goals._get_session_db()
    yield home
    goals._DB_CACHE.clear()
    getattr(loops, "_DB_CACHE", {}).clear()


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


@pytest.fixture()
def session(server):
    sid = "tui-session-1"
    session_key = "agent:main:tui:tui-session-1"
    s = {
        "session_key": session_key,
        "running": False,
        "history_lock": threading.RLock(),
        "input_queue": [],
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _expired_waiting_goal(session_key: str):
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_key)
    mgr.set("do something")
    mgr.wait_for_seconds(120, reason="backoff")
    assert mgr.state is not None
    mgr.state.waiting_until = time.time() - 1
    from hermes_cli.goals import save_goal

    save_goal(session_key, mgr.state)
    return mgr


def test_tui_loop_then_goal_order_keeps_cleared_wait_wakeup(server, session):
    sid, session_key, s = session
    from hermes_cli.goals import GoalManager
    from hermes_cli.loops import LoopManager

    _expired_waiting_goal(session_key)
    loop_mgr = LoopManager(session_key)
    loop_mgr.set("loop work", interval_seconds=1)

    fired = {}

    def fake_submit(rid, sid_, session_, text, **kwargs):
        fired.setdefault("texts", []).append(text)
        return True

    with patch.object(server, "_run_prompt_submit", fake_submit), patch.object(server, "_emit"):
        server._maybe_fire_tui_loop_tick(sid, s)
        server._maybe_wake_tui_parked_goal(sid, s)

    assert len(fired.get("texts", [])) == 1
    assert "The wait barrier has cleared" in fired["texts"][0]
    reloaded = GoalManager(session_key)
    assert reloaded.state is not None
    assert reloaded.state.wakeup_pending is False


def test_tui_goal_wakeup_busy_then_idle_retries(server, session):
    sid, session_key, s = session
    from hermes_cli.goals import GoalManager

    _expired_waiting_goal(session_key)
    s["running"] = True
    with patch.object(server, "_run_prompt_submit") as submit, patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)
    submit.assert_not_called()

    mgr = GoalManager(session_key)
    assert mgr.check_wakeup() is not None
    assert mgr.state is not None
    assert mgr.state.wakeup_pending is True

    s["running"] = False
    with patch.object(server, "_run_prompt_submit", return_value=True) as submit, patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)
    submit.assert_called_once()
    state = GoalManager(session_key).state
    assert state is not None
    assert state.wakeup_pending is False


def test_tui_goal_wakeup_dispatch_exception_retries_next_poll(server, session):
    sid, session_key, s = session
    from hermes_cli.goals import GoalManager

    _expired_waiting_goal(session_key)

    def boom(*args, **kwargs):
        raise RuntimeError("delivery failed")

    with patch.object(server, "_run_prompt_submit", boom), patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)

    assert s["running"] is False
    state = GoalManager(session_key).state
    assert state is not None
    assert state.wakeup_pending is True

    with patch.object(server, "_run_prompt_submit", return_value=True) as submit, patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)
    submit.assert_called_once()
    state = GoalManager(session_key).state
    assert state is not None
    assert state.wakeup_pending is False


def test_tui_goal_wakeup_false_refusal_survives_reload_then_sends_once(server, session):
    sid, session_key, s = session
    from hermes_cli.goals import GoalManager

    _expired_waiting_goal(session_key)

    accepted = []

    def submit_refuses_then_accepts(rid, sid_, session_, text, **kwargs):
        accepted.append(text)
        if len(accepted) == 1:
            return False
        return True

    with patch.object(server, "_run_prompt_submit", submit_refuses_then_accepts), patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)

    reloaded_after_refusal = GoalManager(session_key)
    assert reloaded_after_refusal.state is not None
    assert reloaded_after_refusal.state.wakeup_pending is True
    assert s["running"] is False

    reloaded_server_mgr = GoalManager(session_key)
    assert reloaded_server_mgr.check_wakeup() is not None

    with patch.object(server, "_run_prompt_submit", submit_refuses_then_accepts), patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)
        s["running"] = False
        server._maybe_wake_tui_parked_goal(sid, s)

    assert len(accepted) == 2
    assert GoalManager(session_key).check_wakeup() is None


def test_tui_goal_wakeup_pending_survives_reload_and_sends_once(server, session):
    sid, session_key, s = session
    from hermes_cli.goals import GoalManager

    mgr = _expired_waiting_goal(session_key)
    assert mgr.check_wakeup() is not None
    assert mgr.state is not None
    assert mgr.state.wakeup_pending is True

    reloaded = GoalManager(session_key)
    assert reloaded.check_wakeup() is not None

    with patch.object(server, "_run_prompt_submit", return_value=True) as submit, patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)
        s["running"] = False
        server._maybe_wake_tui_parked_goal(sid, s)

    submit.assert_called_once()
    assert GoalManager(session_key).check_wakeup() is None


def test_tui_goal_wakeup_noop_when_not_cleared(server, session):
    sid, session_key, s = session
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_key)
    mgr.set("do something")
    mgr.wait_for_seconds(100)  # still waiting

    with patch.object(server, "_run_prompt_submit") as submit, patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)

    submit.assert_not_called()
    assert s["running"] is False


def test_tui_goal_wakeup_real_submit_refusal_keeps_pending(server, session):
    sid, session_key, s = session
    from hermes_cli.goals import GoalManager

    _expired_waiting_goal(session_key)

    class Refusal:
        reason = "other owner"

        def __str__(self):
            return "session has another live owner"

    with patch.object(server, "_ensure_active_session_slot", return_value=Refusal()), patch.object(server, "_emit"):
        server._maybe_wake_tui_parked_goal(sid, s)

    state = GoalManager(session_key).state
    assert state is not None
    assert state.wakeup_pending is True
    assert s["running"] is False



def test_goal_wakeup_refusal_does_not_clear_new_prompt_submit_claim(server, session):
    sid, session_key, s = session

    _expired_waiting_goal(session_key)
    prompt_submit_claimed = threading.Event()

    class Refusal:
        reason = "other owner"

        def __str__(self):
            return "session has another live owner"

    def emit_then_prompt_submit_claims(event, sid_, payload=None):
        if event == "error":
            with s["history_lock"]:
                assert s["running"] is False
                s["running"] = True
            prompt_submit_claimed.set()

    with patch.object(server, "_ensure_active_session_slot", return_value=Refusal()), patch.object(
        server, "_emit", emit_then_prompt_submit_claims
    ):
        server._maybe_wake_tui_parked_goal(sid, s)

    assert prompt_submit_claimed.is_set()
    assert s["running"] is True
