"""Guard tests for kanban no-rerun execution and promotion paths."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _connect(db_path: Path) -> sqlite3.Connection:
    return kb.connect(db_path=db_path)


def _set_no_rerun_state(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    status: str,
    reason: str = "do not rerun",
    assignee: str | None = None,
) -> None:
    conn.execute(
        "UPDATE tasks SET status = ?, no_rerun = 1, no_rerun_reason = ?, assignee = ? WHERE id = ?",
        (status, reason, assignee, task_id),
    )


def _event_kinds(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [row["kind"] for row in rows]


def test_no_rerun_blocks_claim_and_review_claim_without_side_effects(tmp_path):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        ready = kb.create_task(conn, title="ready-no-rerun")
        review = kb.create_task(conn, title="review-no-rerun", assignee="reviewer")
        _set_no_rerun_state(conn, ready, status="ready", assignee="builder")
        _set_no_rerun_state(conn, review, status="review", assignee="reviewer")

        claimed = kb.claim_task(conn, ready, claimer="claimer:ready")
        claimed_review = kb.claim_review_task(conn, review, claimer="claimer:review")

        ready_task = kb.get_task(conn, ready)
        review_task = kb.get_task(conn, review)

    assert claimed is None
    assert claimed_review is None
    assert ready_task is not None
    assert review_task is not None
    assert ready_task.status == "ready"
    assert ready_task.claim_lock is None
    assert ready_task.current_run_id is None
    assert ready_task.no_rerun == 1
    assert review_task.status == "review"
    assert review_task.claim_lock is None
    assert review_task.current_run_id is None
    assert review_task.no_rerun == 1
    with _connect(db_path) as check_conn:
        assert _event_kinds(check_conn, ready) == ["created"]
        assert _event_kinds(check_conn, review) == ["created"]


def test_no_rerun_recompute_ready_skips_promotable_tasks(tmp_path):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        _set_no_rerun_state(conn, child, status="blocked", reason="superseded")

        promoted = kb.recompute_ready(conn)
        child_task = kb.get_task(conn, child)

    assert promoted == 0
    assert child_task is not None
    assert child_task.status == "blocked"
    assert child_task.no_rerun == 1
    with _connect(db_path) as check_conn:
        assert _event_kinds(check_conn, child) == ["created"]


@pytest.mark.parametrize(
    "status, action",
    [
        ("blocked", "unblock_task"),
        ("review", "reopen_review_task"),
    ],
)
def test_no_rerun_reactivation_paths_refuse_to_move_state(tmp_path, status, action):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        task = kb.create_task(conn, title=f"{action}-target")
        if status == "blocked":
            conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task,))
        else:
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task,))
        _set_no_rerun_state(conn, task, status=status, reason="keep stopped")

        if action == "unblock_task":
            result = kb.unblock_task(conn, task)
        else:
            result = kb.reopen_review_task(conn, task)

        task_row = kb.get_task(conn, task)

    assert result is False
    assert task_row is not None
    assert task_row.status == status
    assert task_row.no_rerun == 1
    with _connect(db_path) as check_conn:
        assert _event_kinds(check_conn, task) == ["created"]


def test_no_rerun_promote_refuses_to_reactivate(tmp_path):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        parent = kb.create_task(conn, title="parent")
        task = kb.create_task(conn, title="todo-no-rerun", parents=[parent])
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (task,))
        _set_no_rerun_state(conn, task, status="todo", reason="do not retry")

        ok, reason = kb.promote_task(conn, task, actor="operator")
        task_row = kb.get_task(conn, task)

    assert ok is False
    assert reason == f"task {task} is marked no_rerun"
    assert task_row is not None
    assert task_row.status == "todo"
    assert task_row.no_rerun == 1
    with _connect(db_path) as check_conn:
        assert _event_kinds(check_conn, task) == ["created"]


def test_no_rerun_promote_rechecks_inside_write_transaction(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as setup:
        task = kb.create_task(setup, title="blocked-race")
        setup.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (task,))

    conn1 = _connect(db_path)
    conn2 = _connect(db_path)
    original_write_txn = kb.write_txn
    injected = False

    def inject_no_rerun_before_promote_write(conn):
        nonlocal injected
        if conn is conn1 and not injected:
            injected = True
            assert kb.set_no_rerun(conn2, task, True, reason="superseded while promoting")
        return original_write_txn(conn)

    monkeypatch.setattr(kb, "write_txn", inject_no_rerun_before_promote_write)
    try:
        ok, reason = kb.promote_task(conn1, task, actor="operator")
        task_row = kb.get_task(conn1, task)
    finally:
        conn1.close()
        conn2.close()

    assert injected is True
    assert ok is False
    assert reason == f"task {task} is marked no_rerun"
    assert task_row is not None
    assert task_row.status == "blocked"
    assert task_row.no_rerun == 1
    with _connect(db_path) as check_conn:
        assert _event_kinds(check_conn, task) == ["created", "no_rerun_set"]


def test_dispatch_skips_no_rerun_before_claim_or_spawn_side_effects(
    tmp_path,
    all_assignees_spawnable,
    monkeypatch,
):
    db_path = tmp_path / "kanban.db"
    with _connect(db_path) as conn:
        ready = kb.create_task(conn, title="ready-no-rerun")
        review = kb.create_task(conn, title="review-no-rerun", assignee="reviewer")
        _set_no_rerun_state(conn, ready, status="ready", reason="skip dispatch")
        _set_no_rerun_state(conn, review, status="review", reason="skip dispatch", assignee="reviewer")

        def fail_workspace(*_args, **_kwargs):
            raise AssertionError("resolve_workspace must not be called for no_rerun tasks")

        def fail_spawn(*_args, **_kwargs):
            raise AssertionError("spawn_fn must not be called for no_rerun tasks")

        monkeypatch.setattr(kb, "resolve_workspace", fail_workspace)
        monkeypatch.setattr(kb, "_resolve_worktree_workspace", fail_workspace)

        result = kb.dispatch_once(
            conn,
            spawn_fn=fail_spawn,
            default_assignee="reviewer",
        )

        ready_task = kb.get_task(conn, ready)
        review_task = kb.get_task(conn, review)
        task_events = conn.execute(
            "SELECT task_id, kind FROM task_events WHERE task_id IN (?, ?) ORDER BY id",
            (ready, review),
        ).fetchall()

    assert not result.spawned
    assert not result.respawn_guarded
    assert ready_task is not None
    assert review_task is not None
    assert ready_task.status == "ready"
    assert ready_task.assignee is None
    assert review_task.status == "review"
    assert review_task.assignee == "reviewer"
    assert [row["kind"] for row in task_events] == ["created", "created"]
