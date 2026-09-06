"""Persistence-only tests for kanban no-rerun and successor fields."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _connect(db_path: Path) -> sqlite3.Connection:
    return kb.connect(db_path=db_path)


def _event_payloads(conn: sqlite3.Connection, task_id: str, kind: str) -> list[dict | None]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()
    out: list[dict | None] = []
    for row in rows:
        if row["payload"]:
            import json

            out.append(json.loads(row["payload"]))
        else:
            out.append(None)
    return out


def test_no_rerun_persists_with_reason_without_successor_coupling(tmp_path):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        stopped = kb.create_task(conn, title="stopped")
        successor = kb.create_task(conn, title="successor")
        assert kb.complete_task(conn, stopped, result="done")

        assert kb.set_no_rerun(conn, stopped, True, reason="superseded by manual fix")
        assert kb.set_successor_task_id(conn, stopped, successor)

        task = kb.get_task(conn, stopped)
        next_task = kb.get_task(conn, successor)

        assert task is not None
        assert task.no_rerun == 1
        assert task.no_rerun_reason == "superseded by manual fix"
        assert task.successor_task_id == successor
        assert next_task is not None
        assert next_task.no_rerun == 0
        assert next_task.no_rerun_reason is None
        assert next_task.successor_task_id is None
        assert _event_payloads(conn, stopped, "no_rerun_set") == [
            {"no_rerun": True, "reason": "superseded by manual fix"}
        ]
        assert _event_payloads(conn, stopped, "successor_task_set") == [
            {"successor_task_id": successor}
        ]


def test_no_rerun_enable_and_clear_require_reason_and_preserve_status(tmp_path):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        task_id = kb.create_task(conn, title="needs-stop")

        with pytest.raises(RuntimeError, match="blocked, done, or archived"):
            kb.set_no_rerun(conn, task_id, True, reason="not stopped")
        assert kb.get_task(conn, task_id).no_rerun == 0  # type: ignore[union-attr]

        assert kb.block_task(conn, task_id, reason="waiting")
        with pytest.raises(ValueError, match="reason"):
            kb.set_no_rerun(conn, task_id, True, reason=" ")

        assert kb.set_no_rerun(conn, task_id, True, reason="do not retry this blocked card")
        before_clear = kb.get_task(conn, task_id)
        assert before_clear is not None
        assert before_clear.status == "blocked"
        assert before_clear.no_rerun == 1

        with pytest.raises(ValueError, match="reason"):
            kb.set_no_rerun(conn, task_id, False)
        assert kb.get_task(conn, task_id).no_rerun == 1  # type: ignore[union-attr]

        assert kb.set_no_rerun(conn, task_id, False, reason="operator reopened retry path")
        after_clear = kb.get_task(conn, task_id)
        assert after_clear is not None
        assert after_clear.status == "blocked"
        assert after_clear.no_rerun == 0
        assert after_clear.no_rerun_reason is None
        assert _event_payloads(conn, task_id, "no_rerun_cleared") == [
            {
                "no_rerun": False,
                "reason": "operator reopened retry path",
                "previous_reason": "do not retry this blocked card",
            }
        ]


def test_successor_reference_is_validated_atomically_and_independent(tmp_path):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        first = kb.create_task(conn, title="first")
        second = kb.create_task(conn, title="second")
        third = kb.create_task(conn, title="third")

        with pytest.raises(ValueError, match="unknown task"):
            kb.set_successor_task_id(conn, first, "t_missing")
        with pytest.raises(ValueError, match="own successor"):
            kb.set_successor_task_id(conn, first, first)

        assert kb.set_successor_task_id(conn, first, second)
        assert kb.set_successor_task_id(conn, second, third)
        with pytest.raises(ValueError, match="cycle"):
            kb.set_successor_task_id(conn, third, first)
        with pytest.raises(ValueError, match="cycle"):
            kb.set_successor_task_id(conn, second, first)

        first_task = kb.get_task(conn, first)
        second_task = kb.get_task(conn, second)
        third_task = kb.get_task(conn, third)
        assert first_task is not None
        assert second_task is not None
        assert third_task is not None
        assert first_task.successor_task_id == second
        assert second_task.successor_task_id == third
        assert third_task.successor_task_id is None
        assert first_task.no_rerun == 0
        assert second_task.no_rerun == 0


def test_legacy_db_migrates_no_rerun_columns_without_guessing(tmp_path):
    db_path = tmp_path / "legacy-kanban.db"
    raw = sqlite3.connect(db_path)
    try:
        raw.execute(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT,
                assignee TEXT,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                created_by TEXT,
                created_at INTEGER NOT NULL,
                started_at INTEGER,
                completed_at INTEGER,
                workspace_kind TEXT NOT NULL DEFAULT 'scratch',
                workspace_path TEXT,
                claim_lock TEXT,
                claim_expires INTEGER
            )
            """
        )
        raw.execute(
            """
            INSERT INTO tasks (
                id, title, body, assignee, status, priority, created_by,
                created_at, started_at, completed_at, workspace_kind,
                workspace_path, claim_lock, claim_expires
            ) VALUES ('t_legacy', 'legacy', 'do not infer from body', NULL,
                      'blocked', 0, NULL, 1, NULL, NULL, 'scratch', NULL, NULL, NULL)
            """
        )
        raw.commit()
    finally:
        raw.close()

    kb.init_db(db_path=db_path)

    with _connect(db_path) as conn:
        task = kb.get_task(conn, "t_legacy")
        assert task is not None
        assert task.no_rerun == 0
        assert task.no_rerun_reason is None
        assert task.successor_task_id is None
        row = conn.execute(
            "SELECT no_rerun, no_rerun_reason, successor_task_id FROM tasks WHERE id = 't_legacy'"
        ).fetchone()
        assert tuple(row) == (0, None, None)


def test_two_connections_reject_concurrent_mutual_successors(tmp_path):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as setup:
        left = kb.create_task(setup, title="left")
        right = kb.create_task(setup, title="right")

    conn1 = _connect(db_path)
    conn2 = _connect(db_path)
    try:
        assert kb.set_successor_task_id(conn1, left, right)
        with pytest.raises(ValueError, match="cycle"):
            kb.set_successor_task_id(conn2, right, left)
        left_task = kb.get_task(conn1, left)
        right_task = kb.get_task(conn2, right)
        assert left_task is not None
        assert right_task is not None
        assert left_task.successor_task_id == right
        assert right_task.successor_task_id is None
    finally:
        conn1.close()
        conn2.close()


def test_delegated_child_context_cannot_mutate_no_rerun_or_successor(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        task_id = kb.create_task(conn, title="child-guard")
        successor = kb.create_task(conn, title="successor")
        assert kb.complete_task(conn, task_id)

        monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
        with pytest.raises(PermissionError):
            kb.set_no_rerun(conn, task_id, True, reason="child must not mutate")
        with pytest.raises(PermissionError):
            kb.set_successor_task_id(conn, task_id, successor)
        monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT")

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.no_rerun == 0
        assert task.no_rerun_reason is None
        assert task.successor_task_id is None
