"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_stuck_in_blocked_no_rerun_fresh_block_still_shows_intentional_stop():
    now = int(time.time())
    task = _task(
        status="blocked",
        no_rerun=1,
        no_rerun_reason="intentionally stopped",
    )
    events = [_event("blocked", ts=now - 60, reason="policy stop")]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert [d.kind for d in diags] == ["stuck_in_blocked"]
    assert diags[0].title == "Task intentionally stopped (no rerun)"
    assert diags[0].data["blocked_at"] == now - 60


def test_stuck_in_blocked_no_rerun_without_block_event_uses_task_timestamp():
    now = int(time.time())
    task = _task(
        status="blocked",
        no_rerun=1,
        no_rerun_reason="intentionally stopped",
        created_at=now - 60,
    )

    diags = kd.compute_task_diagnostics(task, [], [], now=now)

    assert [d.kind for d in diags] == ["stuck_in_blocked"]
    assert diags[0].title == "Task intentionally stopped (no rerun)"
    assert diags[0].data["blocked_at"] == now - 60


def test_stuck_in_blocked_no_rerun_newer_unblocked_keeps_saved_stop():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(
        status="blocked",
        no_rerun=1,
        no_rerun_reason="successor owns the work",
    )
    events = [
        _event("blocked", ts=blocked_at, reason="policy stop"),
        _event("unblocked", ts=blocked_at + 60, text="later event"),
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert [d.kind for d in diags] == ["stuck_in_blocked"]
    assert diags[0].title == "Task intentionally stopped (no rerun)"
    assert "successor owns the work" in diags[0].detail


def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48
    assert "waiting for human input" not in d.detail
    assert not any(action.suggested for action in d.actions)


def test_stuck_in_blocked_capability_does_not_recommend_blind_reopen():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "prohibited command", "kind": "capability"},
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["block_kind"] == "capability"
    assert not any(
        action.suggested and "unblock" in action.label.lower()
        for action in d.actions
    )
    assert "waiting for human input" not in d.detail
    assert "no automatic retry" in d.detail
    assert "without reviewing the saved block reason" in d.detail


def test_stuck_in_blocked_transient_does_not_recommend_blind_reopen():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "temporary outage", "kind": "transient"},
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["block_kind"] == "transient"
    assert not any(
        action.suggested and "unblock" in action.label.lower()
        for action in d.actions
    )
    assert "waiting for human input" not in d.detail
    assert "no automatic retry" in d.detail
    assert "without reviewing the saved block reason" in d.detail


def test_stuck_in_blocked_needs_input_still_recommends_unblock():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "needs approval", "kind": "needs_input"},
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert any(
        action.suggested and action.label == "Add a comment / unblock the task"
        for action in d.actions
    )
    assert "waiting for human input" in d.detail


def test_stuck_in_blocked_no_rerun_suppresses_reopen_recommendations():
    now = int(time.time())
    task = _task(
        status="blocked",
        no_rerun=1,
        no_rerun_reason="superseded by follow-up task",
    )
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "needs approval", "kind": "needs_input"},
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["no_rerun"] is True
    assert d.data["no_rerun_reason"] == "superseded by follow-up task"
    assert "superseded by follow-up task" in d.detail
    assert "再実行禁止" in d.detail or "no rerun" in d.detail.lower()
    assert not any(action.suggested for action in d.actions)
    assert not any(action.kind in {"unblock", "reassign"} for action in d.actions)


def test_stuck_in_blocked_mentions_successor_status_without_hiding_needs_input_branch():
    now = int(time.time())
    task = _task(
        status="blocked",
        successor_task_id="t_successor",
    )
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "needs approval", "kind": "needs_input"},
        },
    ]
    graph = {"successor": {"id": "t_successor", "status": "running"}}

    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)

    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.data["block_kind"] == "needs_input"
    assert any(
        action.suggested and action.label == "Add a comment / unblock the task"
        for action in d.actions
    )
    assert "waiting for human input" in d.detail
    assert "t_successor" in d.detail
    assert "running" in d.detail
    assert "check on the successor" in d.detail


def test_stuck_in_blocked_no_rerun_mentions_done_successor_without_reopening():
    now = int(time.time())
    task = _task(
        status="blocked",
        no_rerun=1,
        no_rerun_reason="superseded by follow-up task",
        successor_task_id="t_successor",
    )
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "temporary outage", "kind": "capability"},
        },
    ]
    graph = {"successor": {"id": "t_successor", "status": "done"}}

    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)

    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.data["no_rerun"] is True
    assert d.data["successor_task_id"] == "t_successor"
    assert "再実行禁止" in d.detail or "no rerun" in d.detail.lower()
    assert "t_successor" in d.detail
    assert "done" in d.detail or "completed" in d.detail
    assert not any(action.suggested for action in d.actions)


def test_stuck_in_blocked_mentions_unresolvable_successor_needs_checking():
    now = int(time.time())
    task = _task(
        status="blocked",
        successor_task_id="t_missing_successor",
    )
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "needs approval", "kind": "dependency"},
        },
    ]
    graph = {
        "successor": {"id": "t_missing_successor", "status": None, "error": "not_found"}
    }

    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)

    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.data["block_kind"] == "dependency"
    assert "t_missing_successor" in d.detail
    assert "not found" in d.detail.lower() or "missing" in d.detail.lower()
    assert "check on the successor" in d.detail


@pytest.mark.parametrize("graph", [None, {}, {"successor": None}, {"successor": {}}])
def test_stuck_in_blocked_mentions_successor_status_unavailable_when_graph_missing(graph):
    now = int(time.time())
    task = _task(
        status="blocked",
        no_rerun=1,
        no_rerun_reason="superseded by successor",
        successor_task_id="t_unreadable_successor",
    )
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "new prohibited work", "kind": "capability"},
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)

    assert [d.kind for d in diags] == ["stuck_in_blocked"]
    detail = diags[0].detail
    assert "再実行禁止" in detail or "no rerun" in detail.lower()
    assert "superseded by successor" in detail
    assert "t_unreadable_successor" in detail
    assert "unavailable" in detail.lower()
    assert "not found" not in detail.lower()
    assert "check on the successor" in detail
    assert diags[0].actions == []


def test_stuck_in_blocked_no_rerun_comment_keeps_saved_stop_visible():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(
        status="blocked",
        no_rerun=1,
        no_rerun_reason="new comment confirms successor owns the work",
        successor_task_id="t_successor",
    )
    events = [
        {
            "kind": "blocked",
            "created_at": blocked_at,
            "payload": {"reason": "needs approval", "kind": "needs_input"},
        },
        _event("commented", ts=blocked_at + 60, text="do not reopen original"),
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert [d.kind for d in diags] == ["stuck_in_blocked"]
    detail = diags[0].detail
    assert "再実行禁止" in detail or "no rerun" in detail.lower()
    assert "new comment confirms successor owns the work" in detail
    assert "unblock with feedback" not in detail
    assert diags[0].actions == []


@pytest.mark.parametrize("terminal_status", ["done", "archived"])
def test_stuck_in_blocked_no_rerun_terminal_status_keeps_saved_stop_visible(terminal_status):
    now = int(time.time())
    task = _task(
        status=terminal_status,
        no_rerun=1,
        no_rerun_reason="successor completed the replacement work",
        successor_task_id="t_successor",
    )
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "prohibited original", "kind": "capability"},
        },
    ]
    graph = {"successor": {"id": "t_successor", "status": "done"}}

    diags = kd.compute_task_diagnostics(task, events, [], now=now, graph=graph)

    assert [d.kind for d in diags] == ["stuck_in_blocked"]
    detail = diags[0].detail
    assert "再実行禁止" in detail or "no rerun" in detail.lower()
    assert "successor completed the replacement work" in detail
    assert "t_successor" in detail
    assert "done" in detail
    assert "reassign" not in detail.lower()
    assert "unblock with feedback" not in detail
    assert diags[0].actions == []


def test_stuck_in_blocked_capability_comment_does_not_clear_warning():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": blocked_at,
            "payload": {"reason": "prohibited command", "kind": "capability"},
        },
        _event("commented", ts=blocked_at + 60, text="reviewed but still blocked"),
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert [d.kind for d in diags] == ["stuck_in_blocked"]
    assert diags[0].data["block_kind"] == "capability"
    assert "waiting for human input" not in diags[0].detail

def test_stuck_in_blocked_transient_comment_does_not_clear_warning():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": blocked_at,
            "payload": {"reason": "temporary outage", "kind": "transient"},
        },
        _event("commented", ts=blocked_at + 60, text="still waiting on outage"),
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert [d.kind for d in diags] == ["stuck_in_blocked"]
    assert diags[0].data["block_kind"] == "transient"
    assert "waiting for human input" not in diags[0].detail


def test_stuck_in_blocked_capability_unblocked_clears_warning():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": blocked_at,
            "payload": {"reason": "prohibited command", "kind": "capability"},
        },
        _event("unblocked", ts=blocked_at + 60, text="human explicitly reopened"),
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert [d.kind for d in diags] == []


def test_stuck_in_blocked_unknown_kind_uses_neutral_warning():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {
                "reason": "unexpected block classification",
                "kind": "weird_unknown_kind",
            },
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    d = diags[0]
    assert d.data["block_kind"] == "weird_unknown_kind"
    assert "waiting for human input" not in d.detail
    assert "no automatic retry" in d.detail
    assert not any(action.suggested for action in d.actions)
    assert not any(action.kind in {"write", "unblock"} for action in d.actions)


@pytest.mark.parametrize("bad_kind", [123, ["capability"], {"kind": "capability"}])
def test_stuck_in_blocked_non_string_kind_uses_neutral_warning(bad_kind):
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": now - 3600 * 48,
            "payload": {"reason": "bad payload", "kind": bad_kind},
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    d = diags[0]
    assert d.data["block_kind"] == bad_kind
    assert "waiting for human input" not in d.detail
    assert "no automatic retry" in d.detail
    assert not any(action.suggested for action in d.actions)


def test_stuck_in_blocked_dependency_comment_does_not_clear_warning():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": blocked_at,
            "payload": {"reason": "waiting for predecessor", "kind": "dependency"},
        },
        _event("commented", ts=blocked_at + 60, text="not input"),
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    d = diags[0]
    assert d.data["block_kind"] == "dependency"
    assert "waiting for human input" not in d.detail
    assert not any(action.suggested for action in d.actions)


def test_stuck_in_blocked_missing_kind_comment_does_not_clear_warning():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=blocked_at, reason="legacy missing kind"),
        _event("commented", ts=blocked_at + 60, text="ambiguous comment"),
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    d = diags[0]
    assert "block_kind" not in d.data
    assert "waiting for human input" not in d.detail
    assert not any(action.suggested for action in d.actions)


def test_stuck_in_blocked_needs_input_comment_clears_warning():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(status="blocked")
    events = [
        {
            "kind": "blocked",
            "created_at": blocked_at,
            "payload": {"reason": "needs approval", "kind": "needs_input"},
        },
        _event("commented", ts=blocked_at + 60, text="approved"),
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert [d.kind for d in diags] == []


def test_stuck_in_blocked_same_timestamp_uses_later_event_in_list_order():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(status="blocked")
    events = [
        {
            "kind": "block_loop_detected",
            "created_at": blocked_at,
            "payload": {"reason": "loop", "kind": "capability"},
        },
        {
            "kind": "blocked",
            "created_at": blocked_at,
            "payload": {"reason": "needs approval", "kind": "needs_input"},
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    assert diags[0].data["block_kind"] == "needs_input"
    assert "waiting for human input" in diags[0].detail


def test_stuck_in_blocked_same_timestamp_prefers_highest_event_id():
    now = int(time.time())
    blocked_at = now - 3600 * 48
    task = _task(status="blocked")
    events = [
        {
            "id": 2,
            "kind": "block_loop_detected",
            "created_at": blocked_at,
            "payload": {"reason": "loop", "kind": "capability"},
        },
        {
            "id": 1,
            "kind": "blocked",
            "created_at": blocked_at,
            "payload": {"reason": "needs approval", "kind": "needs_input"},
        },
    ]

    diags = kd.compute_task_diagnostics(task, events, [], now=now)

    assert len(diags) == 1
    assert diags[0].data["block_kind"] == "capability"
    assert "waiting for human input" not in diags[0].detail



def test_repeated_crashes_truncates_huge_tracebacks():
    """Full Python tracebacks can be tens of KB. The title stays one
    line (≤160 chars); the detail caps at 500 chars + ellipsis so the
    card doesn't explode visually."""
    huge = "Traceback (most recent call last):\n" + ("  File\n" * 500)
    task = _task(status="ready")
    runs = [
        _run(outcome="crashed", run_id=1, error=huge),
        _run(outcome="crashed", run_id=2, error=huge),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    d = diags[0]
    # Title only the first line, capped.
    assert "\n" not in d.title
    assert len(d.title) < 250
    # Detail contains the snippet with ellipsis.
    assert d.detail.endswith("…") or len(d.detail) < 700


# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True
