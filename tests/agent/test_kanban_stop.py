"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.delegation_context import (
    delegated_child_context,
    non_dispatcher_owned_context,
)
from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.


def test_no_nudge_for_delegated_child_sharing_dispatcher_env(clear_kanban_env):
    """A delegate_task child running inside the dispatcher worker's own
    process must never be nudged toward kanban_complete/kanban_block: it
    inherits HERMES_KANBAN_TASK from the parent's os.environ but does not
    own the parent's card."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent_owns_this")
    with delegated_child_context(session_id="child-session"):
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None


def test_no_nudge_for_in_process_cron_sharing_dispatcher_env(clear_kanban_env):
    """cronjob(action="run") executes run_job() in-process inside a kanban
    worker; that cron agent must not be nudged to close the worker's card."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_parent_owns_this")
    with non_dispatcher_owned_context():
        assert kanban_stop_nudge_enabled() is False
        assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_still_fires_for_real_dispatcher_worker(clear_kanban_env):
    """Baseline: outside any delegated/non-owner context, the real
    dispatcher-owned worker still gets nudged as before."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_real_worker")
    assert kanban_stop_nudge_enabled() is True
    messages = [
        {"role": "user", "content": "work kanban task"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "t_real_worker" in nudge




