from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import threading
import time

from .harness import (
    COMPRESSOR_MODEL,
    collect_real_child_process,
    JUDGE_MODEL,
    PRIMARY_MODEL,
    SOURCE_ROOT,
    hermes_home,
    make_real_agent,
    make_tui_session,
    openai_stub,
    run_isolated_parent_process,
    run_real_child_process,
    server_module,
    wait_for_turn_to_finish,
    write_stub_config,
)
from .real_gateway_watcher_helper import (
    append_real_watcher_config,
    assert_boards_are_inside_temp_root,
    configure_abnormal_real_watcher_environment,
    ps_snapshot,
    read_crash_state,
    run_product_gateway_watcher,
    start_claimed_nonzero_worker,
)


def assert_eventually(label, probe, *, timeout: float = 25.0, interval: float = 0.05):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = probe()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out waiting for {label}; last={last!r}")


def goal_snapshot(session_key: str) -> dict:
    from hermes_cli.goals import GoalManager

    state = GoalManager(session_key).state
    if state is None:
        return {}
    return {
        "status": state.status,
        "waiting_on_pid": state.waiting_on_pid,
        "waiting_on_session": state.waiting_on_session,
        "waiting_until": state.waiting_until,
        "waiting_reason": state.waiting_reason,
        "wakeup_pending": state.wakeup_pending,
        "last_verdict": state.last_verdict,
        "last_reason": state.last_reason,
        "turns_used": state.turns_used,
    }


def message_roles_and_content(messages: list[dict]) -> list[tuple[str, str]]:
    return [(str(msg.get("role", "")), str(msg.get("content", ""))) for msg in messages]


def record_server_emits(monkeypatch, server_module):
    """Observe the real TUI transport frames without depending on capfd timing."""
    frames = []

    class ObservedStdout(io.StringIO):
        def __init__(self):
            super().__init__()
            self._pending = ""

        def write(self, text):
            written = super().write(text)
            self._pending += text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                if line.strip():
                    frames.append(json.loads(line))
            return written

        def flush(self):
            return super().flush()

    monkeypatch.setattr(server_module, "_real_stdout", ObservedStdout())
    return frames


def load_session_messages_from_home(home: Path, session_id: str):
    """Read SessionDB through the explicit isolated state.db path."""
    from hermes_state import SessionDB

    db = SessionDB(home / "state.db")
    try:
        return db.get_messages(session_id), None
    except Exception as exc:
        return None, f"Failed to read messages: {exc}"
    finally:
        db.close()


def test_normal_continuity_flow_is_driven_by_real_tui_notification_poller_thread(
    tmp_path,
    hermes_home,
    openai_stub,
    server_module,
    monkeypatch,
    capfd,
):
    """通常系1本: _init_session起動の実pollerだけが子完了/Goal wake/通知turnを駆動する。"""
    emitted_frames = record_server_emits(monkeypatch, server_module)
    write_stub_config(hermes_home, openai_stub.base_url)
    openai_stub.assert_rejects_bad_auth()

    from hermes_cli import kanban_db as kb
    from hermes_state_registry import get_shared_session_db, release_or_close
    from mcp_serve import _load_session_messages

    parent_module_files = {
        "server": Path(server_module.__file__).resolve(),
        "goals": Path(__import__("hermes_cli.goals", fromlist=["dummy"]).__file__).resolve(),
        "kanban_db": Path(kb.__file__).resolve(),
        "run_agent": Path(__import__("run_agent", fromlist=["dummy"]).__file__).resolve(),
    }
    for name, module_file in parent_module_files.items():
        assert module_file.is_relative_to(SOURCE_ROOT), f"{name} escaped source root: {module_file}"

    session_db = get_shared_session_db()
    try:
        sid = "isolated-normal-sid"
        session_key = "agent:main:tui:isolated-normal-session"
        agent = make_real_agent(openai_stub.base_url, session_key, session_db=session_db)
        assert agent._session_db is session_db
        assert Path(agent._session_db.db_path).resolve() == Path(hermes_home / "state.db").resolve()
        assert Path(session_db.db_path).resolve() == Path(hermes_home / "state.db").resolve()

        session = make_tui_session(server_module, hermes_home, sid, session_key, agent)
        assert session.get("_notif_stop") is not None
        assert any(t.name == f"tui-notif-poller-{sid}" and t.is_alive() for t in threading.enumerate())

        from hermes_cli.goals import GoalManager

        with session["history_lock"]:
            initial_history = list(session["history"])
        agent._session_messages = initial_history
        assert server_module._flush_session_messages(session) is True
        pre_release_messages, pre_release_error = _load_session_messages(agent.session_id)
        assert pre_release_error is None
        assert pre_release_messages is not None
        assert any("long history user 19" in str(msg.get("content", "")) for msg in pre_release_messages)
        assert any("long history assistant 19" in str(msg.get("content", "")) for msg in pre_release_messages)

        mgr = GoalManager(session_key)
        long_goal = "長い履歴から委譲した子の正常終了を確認して会話を再開する。" + ("履歴 " * 1200)
        mgr.set(long_goal, max_turns=3)

        with kb.connect() as conn:
            child_task_id = kb.create_task(
                conn,
                title="isolated continuity child",
                body="parent-seeded child, claimed and completed by the real subprocess",
                assignee="worker",
                created_by="parent",
            )
            dependent_task_id = kb.create_task(
                conn,
                title="dependent starts todo and becomes ready via recompute_ready",
                body="must remain todo until the child task completes",
                assignee=None,
                created_by="parent",
                parents=[child_task_id],
            )
            kb.add_notify_sub(conn, task_id=child_task_id, platform="tui", chat_id=session_key)
            child_seed = kb.get_task(conn, child_task_id)
            dependent_seed = kb.get_task(conn, dependent_task_id)
            sub_before = kb.list_notify_subs(conn, child_task_id)[0]
            dependency_link = conn.execute(
                "SELECT parent_id, child_id FROM task_links WHERE parent_id = ? AND child_id = ?",
                (child_task_id, dependent_task_id),
            ).fetchone()
        assert child_seed is not None and child_seed.status == "ready"
        assert dependent_seed is not None and dependent_seed.status == "todo"
        assert dependency_link is not None
        cursor_before = int(sub_before["last_event_id"])

        expected_primary_text = openai_stub.set_primary_reply(
            task_id=dependent_task_id,
            reason="readyだが未割当のため開始しない",
        )

        alive_marker = tmp_path / "child.alive.json"
        release_fifo = tmp_path / "child.release.fifo"
        os.mkfifo(release_fifo)
        proc = run_real_child_process(
            hermes_home,
            session_key,
            task_id=child_task_id,
            alive_marker=alive_marker,
            release_fifo=release_fifo,
        )
        assert_eventually("child alive marker", lambda: alive_marker.exists(), timeout=5)
        alive_payload = json.loads(alive_marker.read_text(encoding="utf-8"))
        assert int(alive_payload["pid"]) == proc.pid

        waiting_state = mgr.wait_on(proc.pid, reason="waiting for delegated child pid")
        assert waiting_state.waiting_on_pid == proc.pid
        assert waiting_state.waiting_until == 0
        assert waiting_state.wakeup_pending is False
        parked_snapshot = goal_snapshot(session_key)
        assert parked_snapshot["waiting_on_pid"] == proc.pid

        observed_goal_snapshots: list[dict] = []
        watcher_stop = threading.Event()

        def watch_goal_state() -> None:
            while not watcher_stop.is_set():
                snap = goal_snapshot(session_key)
                if snap:
                    observed_goal_snapshots.append(snap)
                time.sleep(0.001)

        watcher = threading.Thread(target=watch_goal_state, name="isolated-goal-state-watcher", daemon=True)
        watcher.start()

        try:
            with open(release_fifo, "wb", buffering=0) as gate:
                gate.write(b"1")
            child = collect_real_child_process(proc, timeout=10)
            assert child["pid"] > 0
            assert child["returncode"] == 0
            assert child["complete_ok"] is True
            assert child["claim_ok"] is True
            assert child["status_before"] == "ready"
            assert child["status_after_claim"] == "running"
            assert child["status_after"] == "done"
            assert child["event_kind"] == "completed"
            assert child["event_id"] > cursor_before

            def dependent_ready_and_cursor_advanced():
                with kb.connect() as conn:
                    child_row = kb.get_task(conn, child_task_id)
                    dependent_row = kb.get_task(conn, dependent_task_id)
                    sub = kb.list_notify_subs(conn, child_task_id)[0]
                    completed_row = conn.execute(
                        "SELECT id, kind, payload FROM task_events WHERE id = ?",
                        (child["event_id"],),
                    ).fetchone()
                if (
                    child_row is not None
                    and child_row.status == "done"
                    and dependent_row is not None
                    and dependent_row.status == "ready"
                    and dependent_row.assignee is None
                    and int(sub["last_event_id"]) >= int(child["event_id"])
                ):
                    return {"task": child_row, "dependent": dependent_row, "sub": sub, "event": completed_row}
                return None

            db_after = assert_eventually("dependent ready and notify cursor advanced", dependent_ready_and_cursor_advanced, timeout=30)
            completed_payload = json.loads(db_after["event"]["payload"])
            assert db_after["event"]["kind"] == "completed"
            assert completed_payload["summary"] == "child success"

            assert_eventually(
                "goal wakeup_pending observed from durable state",
                lambda: any(s.get("wakeup_pending") is True for s in observed_goal_snapshots),
                timeout=5,
                interval=0.01,
            )

            def goal_done_and_session_idle():
                snap = goal_snapshot(session_key)
                if snap.get("status") == "done" and snap.get("wakeup_pending") is False and session.get("running") is False:
                    return snap
                return None

            final_goal = assert_eventually("goal done and session idle", goal_done_and_session_idle, timeout=30)
            wait_for_turn_to_finish(session, timeout=5)
        finally:
            watcher_stop.set()
            watcher.join(timeout=2)
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2)

        saved_messages, saved_error = _load_session_messages(agent.session_id)
        assert saved_error is None
        assert saved_messages is not None
        saved_assistant_texts = [str(msg.get("content", "")) for msg in saved_messages if msg.get("role") == "assistant"]
        assert expected_primary_text in saved_assistant_texts

        capfd.readouterr()
        complete_texts = [
            str(frame["params"]["payload"]["text"])
            for frame in emitted_frames
            if frame.get("method") == "event" and frame.get("params", {}).get("type") == "message.complete"
        ]
        assert expected_primary_text in complete_texts

        model_requests = openai_stub.requests()
        chat_requests = [req for req in model_requests if req["path"].endswith("/chat/completions")]
        good_chat_requests = [req for req in chat_requests if req["authorization"] == "Bearer test-key"]
        bad_auth_requests = [req for req in chat_requests if req["authorization"] == "Bearer wrong-key"]
        assert bad_auth_requests, chat_requests
        assert good_chat_requests, model_requests
        assert any((req["payload"].get("stream") is True) for req in good_chat_requests)
        assert any("verdict" in str(req["payload"].get("messages", "")).lower() for req in good_chat_requests)
        assert session.get("running") is False

        notification_texts = [str(req["payload"].get("messages", "")) for req in good_chat_requests]
        assert any(child_task_id in text and "child success" in text for text in notification_texts)

        assert child["kanban_db_file"]
        assert Path(child["kanban_db_file"]).resolve().is_relative_to(SOURCE_ROOT)
        assert child["home_env"] == str(hermes_home.parent)
    finally:
        release_or_close(session_db)


def test_compression_summary_empty_content_aborts_without_losing_history_and_continuity_still_runs(
    tmp_path,
    hermes_home,
    openai_stub,
    server_module,
    monkeypatch,
    capfd,
):
    """子完了通知の自動再開turn内で、preflight圧縮の空応答abortを観測する。"""
    emitted_frames = record_server_emits(monkeypatch, server_module)
    threshold_tokens = 1000
    write_stub_config(hermes_home, openai_stub.base_url, threshold_tokens=threshold_tokens, max_attempts=1)
    openai_stub.fail_compressor_with_empty_content()
    openai_stub.assert_rejects_bad_auth()

    from agent.turn_context import _preflight_request_tokens
    from hermes_cli import kanban_db as kb
    from hermes_cli.goals import GoalManager
    from hermes_state_registry import get_shared_session_db, release_or_close
    from mcp_serve import _load_session_messages

    session_db = get_shared_session_db()
    proc = None
    try:
        sid = "isolated-compression-failure-sid"
        session_key = "agent:main:tui:isolated-compression-failure-session"
        agent = make_real_agent(openai_stub.base_url, session_key, session_db=session_db)
        session = make_tui_session(server_module, hermes_home, sid, session_key, agent)
        assert session.get("_notif_stop") is not None
        assert any(t.name == f"tui-notif-poller-{sid}" and t.is_alive() for t in threading.enumerate())

        with session["history_lock"]:
            before_history = list(session["history"])
        before_session_id = agent.session_id
        system_prompt = agent._build_system_prompt(None)
        estimated_tokens = _preflight_request_tokens(agent, before_history, system_prompt)
        compressor_threshold = int(getattr(agent.context_compressor, "threshold_tokens"))
        assert getattr(agent.context_compressor, "threshold_tokens_cap", None) == threshold_tokens
        assert compressor_threshold == threshold_tokens
        assert estimated_tokens > compressor_threshold
        assert agent.context_compressor.should_compress(estimated_tokens) is True

        agent._session_messages = before_history
        assert server_module._flush_session_messages(session) is True
        saved_initial, saved_initial_error = _load_session_messages(agent.session_id)
        assert saved_initial_error is None
        assert saved_initial is not None
        assert message_roles_and_content(saved_initial) == message_roles_and_content(before_history)
        assert any("long history user 19" in str(msg.get("content", "")) for msg in saved_initial)
        assert any("long history assistant 19" in str(msg.get("content", "")) for msg in saved_initial)

        mgr = GoalManager(session_key)
        mgr.set("子完了通知の自動再開turnでpreflight圧縮失敗を観測する。", max_turns=3)
        with kb.connect() as conn:
            child_task_id = kb.create_task(
                conn,
                title="compression failure continuity child",
                body="completed before autonomous preflight compression fails with empty content",
                assignee="worker",
                created_by="parent",
            )
            dependent_task_id = kb.create_task(
                conn,
                title="compression failure dependent remains unassigned",
                body="ready after parent task completes",
                assignee=None,
                created_by="parent",
                parents=[child_task_id],
            )
            kb.add_notify_sub(conn, task_id=child_task_id, platform="tui", chat_id=session_key)
            child_seed = kb.get_task(conn, child_task_id)
            dependent_seed = kb.get_task(conn, dependent_task_id)
            sub_before = kb.list_notify_subs(conn, child_task_id)[0]
            dependency_link = conn.execute(
                "SELECT parent_id, child_id FROM task_links WHERE parent_id = ? AND child_id = ?",
                (child_task_id, dependent_task_id),
            ).fetchone()
        assert child_seed is not None and child_seed.status == "ready"
        assert dependent_seed is not None and dependent_seed.status == "todo"
        assert dependency_link is not None
        cursor_before = int(sub_before["last_event_id"])
        expected_primary_text = openai_stub.set_primary_reply(
            task_id=dependent_task_id,
            reason="compression abort後もprimary応答は成功",
        )

        alive_marker = tmp_path / "compression-child.alive.json"
        release_fifo = tmp_path / "compression-child.release.fifo"
        os.mkfifo(release_fifo)
        proc = run_real_child_process(
            hermes_home,
            session_key,
            task_id=child_task_id,
            alive_marker=alive_marker,
            release_fifo=release_fifo,
        )
        assert_eventually("compression child alive marker", lambda: alive_marker.exists(), timeout=5)
        alive_payload = json.loads(alive_marker.read_text(encoding="utf-8"))
        assert int(alive_payload["pid"]) == proc.pid
        waiting_state = mgr.wait_on(proc.pid, reason="waiting for delegated child pid before compression preflight")
        assert waiting_state.waiting_on_pid == proc.pid
        assert waiting_state.wakeup_pending is False

        pre_release_requests = openai_stub.requests()
        assert not [
            req for req in pre_release_requests
            if req["path"].endswith("/chat/completions") and req["authorization"] == "Bearer test-key"
        ], pre_release_requests

        observed_goal_snapshots: list[dict] = []
        watcher_stop = threading.Event()

        def watch_goal_state() -> None:
            while not watcher_stop.is_set():
                snap = goal_snapshot(session_key)
                if snap:
                    observed_goal_snapshots.append(snap)
                time.sleep(0.001)

        watcher = threading.Thread(target=watch_goal_state, name="isolated-compression-goal-watcher", daemon=True)
        watcher.start()

        try:
            with open(release_fifo, "wb", buffering=0) as gate:
                gate.write(b"1")
            child = collect_real_child_process(proc, timeout=10)
            proc = None
            assert child["pid"] > 0
            assert child["returncode"] == 0
            assert child["complete_ok"] is True
            assert child["claim_ok"] is True
            assert child["status_before"] == "ready"
            assert child["status_after_claim"] == "running"
            assert child["status_after"] == "done"
            assert child["event_kind"] == "completed"
            assert child["event_id"] > cursor_before

            def dependent_ready_and_cursor_advanced():
                with kb.connect() as conn:
                    child_row = kb.get_task(conn, child_task_id)
                    dependent_row = kb.get_task(conn, dependent_task_id)
                    sub = kb.list_notify_subs(conn, child_task_id)[0]
                    completed_row = conn.execute(
                        "SELECT id, kind, payload FROM task_events WHERE id = ?",
                        (child["event_id"],),
                    ).fetchone()
                if (
                    child_row is not None
                    and child_row.status == "done"
                    and dependent_row is not None
                    and dependent_row.status == "ready"
                    and dependent_row.assignee is None
                    and int(sub["last_event_id"]) >= int(child["event_id"])
                ):
                    return {"task": child_row, "dependent": dependent_row, "sub": sub, "event": completed_row}
                return None

            db_after = assert_eventually("dependent ready and notify cursor advanced", dependent_ready_and_cursor_advanced, timeout=30)
            completed_payload = json.loads(db_after["event"]["payload"])
            assert db_after["event"]["kind"] == "completed"
            assert completed_payload["summary"] == "child success"
            assert_eventually(
                "goal wakeup_pending observed from durable state",
                lambda: any(s.get("wakeup_pending") is True for s in observed_goal_snapshots),
                timeout=5,
                interval=0.01,
            )

            def compressor_failure_observed():
                requests = openai_stub.requests()
                good = [
                    req for req in requests
                    if req["path"].endswith("/chat/completions") and req["authorization"] == "Bearer test-key"
                ]
                compressors = [req for req in good if req["payload"].get("model") == COMPRESSOR_MODEL]
                if compressors:
                    return {"all": requests, "good": good, "compressors": compressors}
                return None

            observed_requests = assert_eventually("autonomous preflight compression request", compressor_failure_observed, timeout=30)
            assert all(req.get("stub_kind") == "compression" for req in observed_requests["compressors"])
            assert all(req.get("stub_failure") == "empty_content" for req in observed_requests["compressors"])
            assert min(float(req["recorded_at"]) for req in observed_requests["compressors"]) >= float(child["completed_at"])

            def goal_done_and_dependent_ready():
                snap = goal_snapshot(session_key)
                with kb.connect() as conn:
                    dependent_row = kb.get_task(conn, dependent_task_id)
                    sub = kb.list_notify_subs(conn, child_task_id)[0]
                if (
                    snap.get("status") == "done"
                    and snap.get("wakeup_pending") is False
                    and session.get("running") is False
                    and dependent_row is not None
                    and dependent_row.status == "ready"
                    and int(sub["last_event_id"]) >= int(child["event_id"])
                ):
                    return {"goal": snap, "dependent": dependent_row, "sub": sub}
                return None

            final_state = assert_eventually("goal done after autonomous compression abort", goal_done_and_dependent_ready, timeout=30)
            wait_for_turn_to_finish(session, timeout=5)
        finally:
            watcher_stop.set()
            watcher.join(timeout=2)

        assert final_state["goal"]["last_verdict"] == "done"
        assert getattr(agent.context_compressor, "_last_compress_aborted", None) is True
        assert getattr(agent.context_compressor, "_last_summary_empty_content_failure", None) is True
        assert "empty content" in str(getattr(agent.context_compressor, "_last_summary_error", "")).lower()
        assert getattr(agent.context_compressor, "_last_summary_network_failure", None) is False
        assert getattr(agent.context_compressor, "_last_summary_truncated_failure", None) is False
        assert agent.session_id == before_session_id

        with session["history_lock"]:
            in_memory_history_after = list(session["history"])
        assert in_memory_history_after[:len(before_history)] == before_history

        saved_messages, saved_error = _load_session_messages(agent.session_id)
        assert saved_error is None
        assert saved_messages is not None
        assert message_roles_and_content(saved_messages[:len(saved_initial)]) == message_roles_and_content(saved_initial)
        assert expected_primary_text in [
            str(msg.get("content", "")) for msg in saved_messages if msg.get("role") == "assistant"
        ]

        capfd.readouterr()
        complete_texts = [
            str(frame["params"]["payload"]["text"])
            for frame in emitted_frames
            if frame.get("method") == "event" and frame.get("params", {}).get("type") == "message.complete"
        ]
        assert expected_primary_text in complete_texts

        later_requests = openai_stub.requests()
        model_requests = observed_requests["all"] + later_requests
        good_chat_requests = [
            req for req in model_requests
            if req["path"].endswith("/chat/completions") and req["authorization"] == "Bearer test-key"
        ]
        compressor_requests = [req for req in good_chat_requests if req["payload"].get("model") == COMPRESSOR_MODEL]
        primary_requests = [req for req in good_chat_requests if req["payload"].get("model") == PRIMARY_MODEL]
        judge_requests = [req for req in good_chat_requests if req["payload"].get("model") == JUDGE_MODEL]
        assert compressor_requests, model_requests
        assert primary_requests, model_requests
        assert judge_requests, model_requests
        assert all(req.get("stub_kind") == "primary" and req.get("response_status") == 200 for req in primary_requests)
        assert all(req.get("stub_kind") == "judge" and req.get("response_status") == 200 for req in judge_requests)
        assert max(req["seq"] for req in compressor_requests) < min(req["seq"] for req in primary_requests)
        assert min(req["seq"] for req in primary_requests) < min(req["seq"] for req in judge_requests)
        notification_texts = [str(req["payload"].get("messages", "")) for req in primary_requests]
        assert any(child_task_id in text and "child success" in text for text in notification_texts)

        print(json.dumps({
            "compression_failure_autonomous_evidence": {
                "child": {
                    "pid": child["pid"],
                    "event_id": child["event_id"],
                    "completed_at": child["completed_at"],
                    "status_after": child["status_after"],
                },
                "config_and_estimate": {
                    "estimated_tokens": estimated_tokens,
                    "threshold_tokens": compressor_threshold,
                    "threshold_tokens_cap": getattr(agent.context_compressor, "threshold_tokens_cap", None),
                    "max_attempts": getattr(agent, "max_compression_attempts", None),
                },
                "request_order": [
                    {
                        "seq": req["seq"],
                        "recorded_at": req["recorded_at"],
                        "model": req["payload"].get("model"),
                        "stub_kind": req.get("stub_kind"),
                        "stub_failure": req.get("stub_failure"),
                        "response_status": req.get("response_status"),
                        "stream": req["payload"].get("stream"),
                    }
                    for req in good_chat_requests
                ],
                "db": {
                    "completed_event_id": child["event_id"],
                    "notify_cursor": int(final_state["sub"]["last_event_id"]),
                    "dependent_status": final_state["dependent"].status,
                    "goal_status": final_state["goal"]["status"],
                    "goal_last_verdict": final_state["goal"]["last_verdict"],
                },
                "history": {
                    "session_id_unchanged": agent.session_id == before_session_id,
                    "initial_len": len(saved_initial),
                    "saved_len_after": len(saved_messages),
                    "initial_prefix_preserved": message_roles_and_content(saved_messages[:len(saved_initial)]) == message_roles_and_content(saved_initial),
                },
                "transport": {
                    "message_complete_texts": complete_texts,
                },
                "compression_attrs": {
                    "last_compress_aborted": getattr(agent.context_compressor, "_last_compress_aborted", None),
                    "last_summary_empty_content_failure": getattr(agent.context_compressor, "_last_summary_empty_content_failure", None),
                    "last_summary_error": getattr(agent.context_compressor, "_last_summary_error", None),
                },
            }
        }, ensure_ascii=False), file=sys.__stdout__)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        release_or_close(session_db)


def test_ownership_conflict_refuses_notification_admission_without_advancing_cursor(
    tmp_path,
    hermes_home,
    openai_stub,
    server_module,
    monkeypatch,
    capfd,
):
    """所有権競合: 実active-session leaseが通知turn受付を拒否し、通知を失わない。"""
    emitted_frames = record_server_emits(monkeypatch, server_module)
    write_stub_config(hermes_home, openai_stub.base_url)

    from hermes_cli import kanban_db as kb
    from hermes_cli.active_sessions import SESSION_NOT_OWNED, try_acquire_active_session
    from hermes_state_registry import get_shared_session_db, release_or_close
    from mcp_serve import _load_session_messages

    session_db = get_shared_session_db()
    conflict_lease = None
    proc = None
    try:
        sid = "isolated-ownership-conflict-sid"
        session_key = "agent:main:tui:isolated-ownership-conflict-session"
        agent = make_real_agent(openai_stub.base_url, session_key, session_db=session_db)
        session = make_tui_session(server_module, hermes_home, sid, session_key, agent)
        assert session.get("_notif_stop") is not None

        conflict_lease, refusal = try_acquire_active_session(
            session_id=session_key,
            surface="tui",
            config={},
            metadata={"live_session_id": "already-owning-parent-runtime"},
            registry_home=hermes_home,
        )
        assert conflict_lease is not None
        assert refusal is None

        with kb.connect() as conn:
            child_task_id = kb.create_task(
                conn,
                title="ownership conflict child",
                body="completion notification is claimed while another live owner holds the session",
                assignee="personal",
                created_by="parent",
            )
            dependent_task_id = kb.create_task(
                conn,
                title="ownership conflict dependent",
                body="must not be started during refused admission",
                assignee=None,
                created_by="parent",
                parents=[child_task_id],
            )
            kb.add_notify_sub(conn, task_id=child_task_id, platform="tui", chat_id=session_key)
            cursor_before = int(kb.list_notify_subs(conn, child_task_id)[0]["last_event_id"])

        alive_marker = tmp_path / "ownership-child.alive.json"
        release_fifo = tmp_path / "ownership-child.release.fifo"
        os.mkfifo(release_fifo)
        proc = run_real_child_process(
            hermes_home,
            session_key,
            task_id=child_task_id,
            alive_marker=alive_marker,
            release_fifo=release_fifo,
            result="child result",
            summary="ownership child success",
        )
        assert_eventually("ownership child alive marker", lambda: alive_marker.exists(), timeout=5)
        alive_payload = json.loads(alive_marker.read_text(encoding="utf-8"))
        assert int(alive_payload["pid"]) == proc.pid

        with open(release_fifo, "wb", buffering=0) as gate:
            gate.write(b"1")
        child = collect_real_child_process(proc, timeout=10)
        assert child["pid"] == proc.pid
        assert child["returncode"] == 0
        assert child["complete_ok"] is True
        assert child["claim_ok"] is True
        assert child["status_before"] == "ready"
        assert child["status_after_claim"] == "running"
        assert child["status_after"] == "done"
        assert child["event_kind"] == "completed"
        assert child["summary"] == "ownership child success"
        completed_event_id = int(child["event_id"])
        assert completed_event_id > cursor_before

        pending_claims = assert_eventually(
            "kanban notification admission refusal leaves pending claims",
            lambda: list(session.get("_kanban_pending_claims") or []),
            timeout=12,
            interval=0.05,
        )
        assert len(pending_claims) == 1
        assert pending_claims[0].task_id == child_task_id
        assert pending_claims[0].old_cursor == cursor_before
        assert pending_claims[0].new_cursor == completed_event_id
        with kb.connect() as conn:
            sub_after_refusal = kb.list_notify_subs(conn, child_task_id)[0]
            child_after_refusal = kb.get_task(conn, child_task_id)
            dependent_after_refusal = kb.get_task(conn, dependent_task_id)
        assert int(sub_after_refusal["last_event_id"]) == cursor_before
        assert child_after_refusal.status == "done"
        assert dependent_after_refusal.status == "ready"
        assert dependent_after_refusal.assignee is None
        assert not [
            req for req in openai_stub.requests()
            if req["authorization"] == "Bearer test-key" and req["path"].endswith("/chat/completions")
        ]

        capfd.readouterr()
        error_frames = [
            frame for frame in emitted_frames
            if frame.get("method") == "error" or frame.get("params", {}).get("type") == "error"
        ]
        assert any(SESSION_NOT_OWNED in str(frame) or "already has a live owner" in str(frame) for frame in error_frames)

        conflict_lease.release()
        conflict_lease = None
        expected_primary_text = openai_stub.set_primary_reply(
            task_id=dependent_task_id,
            reason="ownership refusal後に保存済み通知から復帰",
        )

        def delivered_after_release():
            with kb.connect() as conn:
                sub = kb.list_notify_subs(conn, child_task_id)[0]
            messages, err = _load_session_messages(agent.session_id)
            assistant_texts = [str(msg.get("content", "")) for msg in (messages or []) if msg.get("role") == "assistant"]
            if err is None and int(sub["last_event_id"]) >= completed_event_id and expected_primary_text in assistant_texts:
                return {"sub": sub, "assistant_texts": assistant_texts}
            return None

        delivered = assert_eventually("notification delivered after ownership lease release", delivered_after_release, timeout=20)
        wait_for_turn_to_finish(session, timeout=5)
        assert delivered["assistant_texts"].count(expected_primary_text) == 1
        assert session.get("running") is False
        assert session.get("_kanban_pending_claims") == []

        print(json.dumps({
            "ownership_conflict_evidence": {
                "task_id": child_task_id,
                "dependent_task_id": dependent_task_id,
                "refused_old_cursor": cursor_before,
                "completed_event_id": completed_event_id,
                "child_pid": child["pid"],
                "child_returncode": child["returncode"],
                "child_complete_ok": child["complete_ok"],
                "cursor_after_refusal": int(sub_after_refusal["last_event_id"]),
                "cursor_after_release": int(delivered["sub"]["last_event_id"]),
                "pending_claim_count": len(pending_claims),
                "assistant_delivery_count": delivered["assistant_texts"].count(expected_primary_text),
            }
        }, ensure_ascii=False), file=sys.__stdout__)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        if conflict_lease is not None:
            conflict_lease.release()
        release_or_close(session_db)


def test_abnormal_child_exit_is_reclaimed_by_product_gateway_watcher_and_wakes_tui_parent(
    tmp_path,
    monkeypatch,
    hermes_home,
    openai_stub,
    server_module,
    capfd,
):
    """子異常終了: 製品gateway watcherのreapからTUI通知/Goal判断/保存受信まで通す。"""
    configure_abnormal_real_watcher_environment(monkeypatch, hermes_home)
    write_stub_config(hermes_home, openai_stub.base_url)
    append_real_watcher_config(hermes_home)

    from hermes_cli import kanban_db as kb
    from hermes_cli.goals import GoalManager
    from hermes_state_registry import get_shared_session_db, release_or_close
    from mcp_serve import _load_session_messages

    session_db = get_shared_session_db()
    emitted_frames = record_server_emits(monkeypatch, server_module)
    emitted_frame_cursor = 0
    trace_path = tmp_path / "abnormal-real-watcher-integrated-trace.json"
    trace: dict = {
        "route": "abnormal child test integrates TUI poller + GoalManager + SessionDB + product gateway watcher",
        "paths": {
            "home_env": os.environ.get("HOME"),
            "hermes_home_env": os.environ.get("HERMES_HOME"),
            "kanban_home_env": os.environ.get("HERMES_KANBAN_HOME"),
            "kanban_db_env": os.environ.get("HERMES_KANBAN_DB"),
            "bypass": os.environ.get("HERMES_STATE_DB_GUARD_BYPASS"),
        },
    }
    trace["stage"] = "setup"
    worker_proc = None
    worker = None
    captured_frame_records: list[dict] = []
    captured_reads: list[dict] = []

    def drain_captured_frames(label: str) -> None:
        nonlocal emitted_frame_cursor
        captured = capfd.readouterr()
        read_record = {
            "label": label,
            "stdout_len": len(captured.out),
            "stderr_len": len(captured.err),
            "first_frame_order": len(captured_frame_records),
        }
        captured_reads.append(read_record)
        for frame in emitted_frames[emitted_frame_cursor:]:
            params = frame.get("params", {}) if isinstance(frame, dict) else {}
            payload = params.get("payload", {}) if isinstance(params, dict) else {}
            captured_frame_records.append({
                "order": len(captured_frame_records),
                "capture_label": label,
                "line_index": None,
                "raw_line": json.dumps(frame, ensure_ascii=False),
                "frame": frame,
                "method": frame.get("method") if isinstance(frame, dict) else None,
                "type": params.get("type") if isinstance(params, dict) else None,
                "session_id": params.get("session_id") if isinstance(params, dict) else None,
                "session_key": params.get("session_key") if isinstance(params, dict) else None,
                "payload_message_id": payload.get("message_id") if isinstance(payload, dict) else None,
                "payload_id": payload.get("id") if isinstance(payload, dict) else None,
                "payload_turn_id": payload.get("turn_id") if isinstance(payload, dict) else None,
                "expected_sid": trace.get("sid"),
                "expected_session_key": trace.get("session_key"),
            })
        emitted_frame_cursor = len(emitted_frames)
        for line_index, raw_line in enumerate(captured.out.splitlines()):
            stripped = raw_line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                frame = json.loads(stripped)
            except json.JSONDecodeError as exc:
                captured_frame_records.append({
                    "order": len(captured_frame_records),
                    "capture_label": label,
                    "line_index": line_index,
                    "raw_line": raw_line,
                    "json_error": str(exc),
                    "expected_sid": trace.get("sid"),
                    "expected_session_key": trace.get("session_key"),
                })
                continue
            params = frame.get("params", {}) if isinstance(frame, dict) else {}
            payload = params.get("payload", {}) if isinstance(params, dict) else {}
            captured_frame_records.append({
                "order": len(captured_frame_records),
                "capture_label": label,
                "line_index": line_index,
                "raw_line": raw_line,
                "frame": frame,
                "method": frame.get("method") if isinstance(frame, dict) else None,
                "type": params.get("type") if isinstance(params, dict) else None,
                "session_id": params.get("session_id") if isinstance(params, dict) else None,
                "session_key": params.get("session_key") if isinstance(params, dict) else None,
                "payload_message_id": payload.get("message_id") if isinstance(payload, dict) else None,
                "payload_id": payload.get("id") if isinstance(payload, dict) else None,
                "payload_turn_id": payload.get("turn_id") if isinstance(payload, dict) else None,
                "expected_sid": trace.get("sid"),
                "expected_session_key": trace.get("session_key"),
            })
        read_record["last_frame_order"] = len(captured_frame_records) - 1

    try:
        assert os.environ.get("HERMES_STATE_DB_GUARD_BYPASS") is None
        assert Path(os.environ["HOME"]).resolve().is_relative_to(tmp_path.resolve())
        assert Path(os.environ["HERMES_HOME"]).resolve() == hermes_home.resolve()
        assert Path(os.environ["HERMES_KANBAN_HOME"]).resolve() == hermes_home.resolve()
        assert Path(os.environ["HERMES_KANBAN_DB"]).resolve() == (hermes_home / "kanban.db").resolve()
        trace["boards_before_watcher"] = assert_boards_are_inside_temp_root(hermes_home)

        sid = "isolated-abnormal-child-sid"
        session_key = "agent:main:tui:isolated-abnormal-child-session"
        trace["sid"] = sid
        trace["session_key"] = session_key
        agent = make_real_agent(openai_stub.base_url, session_key, session_db=session_db)
        session = make_tui_session(server_module, hermes_home, sid, session_key, agent)
        assert session.get("_notif_stop") is not None
        assert any(t.name == f"tui-notif-poller-{sid}" and t.is_alive() for t in threading.enumerate())

        with session["history_lock"]:
            initial_history = list(session["history"])
        agent._session_messages = initial_history
        assert server_module._flush_session_messages(session) is True

        mgr = GoalManager(session_key)
        mgr.set("異常終了した子を正常完了と区別し、依存未起動のまま結果を保存して受信する。", max_turns=3)

        with kb.connect() as conn:
            child_task_id = kb.create_task(
                conn,
                title="abnormal child exits nonzero",
                body="real watcher must reap this code 42 child and record crashed",
                assignee="default",
                created_by="parent",
            )
            dependent_task_id = kb.create_task(
                conn,
                title="dependent must stay blocked by crashed parent",
                body="must not become ready because parent never completed",
                assignee=None,
                created_by="parent",
                parents=[child_task_id],
            )
            kb.add_notify_sub(conn, task_id=child_task_id, platform="tui", chat_id=session_key)
            child_seed = kb.get_task(conn, child_task_id)
            dependent_seed = kb.get_task(conn, dependent_task_id)
            sub_before = kb.list_notify_subs(conn, child_task_id)[0]
        assert child_seed is not None and child_seed.status == "ready"
        assert dependent_seed is not None and dependent_seed.status == "todo"
        cursor_before = int(sub_before["last_event_id"])
        trace["task_id"] = child_task_id
        trace["dependent_task_id"] = dependent_task_id
        trace["notify_cursor_before"] = cursor_before

        expected_primary_text = openai_stub.set_primary_reply(
            task_id=child_task_id,
            reason="crashedとして保存され、completed0で依存未起動",
        )
        worker, worker_proc = start_claimed_nonzero_worker(hermes_home, child_task_id, exit_code=42)
        trace["stage"] = "worker_claimed"
        trace["worker"] = worker
        trace["worker_pid"] = worker_proc.pid
        trace["worker_claim_lock_matches_claimer_id"] = worker["claim_lock_matches_claimer_id"]
        assert worker["claim_lock_matches_claimer_id"] is True
        assert worker["claim_lock_after_claim"] == worker["claimer_id"]
        assert worker["claim_lock_after_set_worker_pid"] == worker["claimer_id"]
        assert worker["status_after_claim"] == "running"
        assert worker["status_after_set_worker_pid"] == "running"
        assert worker["worker_pid_after_set"] == worker["pid"]
        trace["pre_watcher_model_requests"] = openai_stub.requests()
        assert not [
            req for req in trace["pre_watcher_model_requests"]
            if req["path"].endswith("/chat/completions") and req["authorization"] == "Bearer test-key"
        ]

        waiting_state = mgr.wait_on(worker["pid"], reason="waiting for delegated abnormal child pid")
        trace["stage"] = "waiting_on_worker"
        assert waiting_state.waiting_on_pid == worker["pid"]
        assert waiting_state.wakeup_pending is False

        observed_goal_snapshots: list[dict] = []
        goal_watcher_stop = threading.Event()

        def watch_goal_state() -> None:
            while not goal_watcher_stop.is_set():
                snap = goal_snapshot(session_key)
                if snap:
                    observed_goal_snapshots.append(snap)
                time.sleep(0.001)

        goal_watcher = threading.Thread(target=watch_goal_state, name="isolated-abnormal-goal-watcher", daemon=True)
        goal_watcher.start()
        try:
            trace["stage"] = "watcher_running"
            with run_product_gateway_watcher(trace):
                def crash_reclaimed_by_product_watcher():
                    state = read_crash_state(child_task_id, dependent_task_id)
                    if state is None:
                        return None
                    child_after = state["child_after"]
                    dependent_after = state["dependent_after"]
                    crashed_payload = state["crashed_payload"]
                    run_row = state["run_row"]
                    if (
                        child_after is not None
                        and child_after.status == "ready"
                        and child_after.worker_pid is None
                        and dependent_after is not None
                        and dependent_after.status == "todo"
                        and state["completed_count"] == 0
                        and run_row["status"] == "crashed"
                        and run_row["outcome"] == "crashed"
                        and crashed_payload.get("exit_code") == 42
                        and crashed_payload.get("exit_kind") == "nonzero_exit"
                        and crashed_payload.get("pid") == worker["pid"]
                    ):
                        return state
                    return None

                crash = assert_eventually(
                    "abnormal child crash reclaimed by product gateway watcher",
                    crash_reclaimed_by_product_watcher,
                    timeout=18,
                    interval=0.05,
                )
                crashed_row = crash["crashed_row"]
                crashed_payload = crash["crashed_payload"]
                child_after = crash["child_after"]
                dependent_after = crash["dependent_after"]
                run_row = crash["run_row"]
                completed_count = crash["completed_count"]
                trace["stage"] = "crash_reclaimed"
                trace["crash"] = {
                    "worker_pid": crashed_payload["pid"],
                    "exit_kind": crashed_payload["exit_kind"],
                    "exit_code": crashed_payload["exit_code"],
                    "crashed_event_id": int(crashed_row["id"]),
                    "task_status_after_crash": child_after.status,
                    "dependent_status_after_crash": dependent_after.status,
                    "completed_events": int(completed_count),
                }
                assert "code 42" in child_after.last_failure_error
                assert "code 42" in run_row["error"]
                assert int(crashed_row["id"]) > cursor_before

                assert_eventually(
                    "goal wakeup_pending observed from durable state after crash",
                    lambda: any(s.get("wakeup_pending") is True for s in observed_goal_snapshots),
                    timeout=5,
                    interval=0.01,
                )

                def crashed_notification_delivered():
                    with kb.connect() as conn:
                        sub = kb.list_notify_subs(conn, child_task_id)[0]
                    messages, err = _load_session_messages(agent.session_id)
                    assistant_texts = [str(msg.get("content", "")) for msg in (messages or []) if msg.get("role") == "assistant"]
                    if err is None and int(sub["last_event_id"]) >= int(crashed_row["id"]) and expected_primary_text in assistant_texts:
                        return {"sub": sub, "messages": messages, "assistant_texts": assistant_texts}
                    return None

                delivered = assert_eventually("crashed notification delivered through TUI poller", crashed_notification_delivered, timeout=25)
                trace["stage"] = "notification_delivered"
                trace["notification"] = {
                    "notify_cursor_before": cursor_before,
                    "notify_cursor_after_delivered_snapshot": int(delivered["sub"]["last_event_id"]),
                    "delivered_snapshot_message_count": len(delivered["messages"] or []),
                }
                wait_for_turn_to_finish(session, timeout=5)
                drain_captured_frames("after_notification_turn")
                pre_assert_message_complete_frames = [
                    record
                    for record in captured_frame_records
                    if record.get("method") == "event" and record.get("type") == "message.complete"
                ]
                trace["raw_transport_frames"] = {
                    "expected_sid": sid,
                    "expected_session_key": session_key,
                    "capture_reads": captured_reads,
                    "all_frames": captured_frame_records,
                }
                trace["raw_message_complete_frames_pre_assert"] = {
                    "expected_sid": sid,
                    "expected_session_key": session_key,
                    "all_message_complete_frames": pre_assert_message_complete_frames,
                    "matching_sid_frames": [
                        record for record in pre_assert_message_complete_frames if record.get("session_id") == sid
                    ],
                    "saved_before_identity_and_notification_asserts": True,
                }
        finally:
            goal_watcher_stop.set()
            goal_watcher.join(timeout=2)

        final_goal = goal_snapshot(session_key)
        trace["stage"] = "finished"
        assert final_goal["status"] == "done"
        assert final_goal["wakeup_pending"] is False
        assert final_goal["last_verdict"] == "done"
        assert session.get("running") is False

        good_requests = [req for req in openai_stub.requests() if req["authorization"] == "Bearer test-key"]
        notification_texts = [str(req["payload"].get("messages", "")) for req in good_requests]
        assert any(child_task_id in text and "crashed" in text.lower() for text in notification_texts)
        assert any(
            "worker crashed" in text.lower() and "dispatcher will retry" in text.lower()
            for text in notification_texts
        )
        assert not any("child success" in text for text in notification_texts)

        saved_messages, saved_error = _load_session_messages(agent.session_id)
        assert saved_error is None
        assert saved_messages is not None
        saved_records = [
            {
                "order": order,
                "id": msg.get("id"),
                "session_id": msg.get("session_id"),
                "role": str(msg.get("role", "")),
                "content": str(msg.get("content", "")),
                "finish_reason": msg.get("finish_reason"),
                "platform_message_id": msg.get("platform_message_id"),
            }
            for order, msg in enumerate(saved_messages)
        ]
        initial_len = len(initial_history)
        assert message_roles_and_content(saved_messages[:initial_len]) == message_roles_and_content(initial_history)
        post_initial_records = saved_records[initial_len:]
        assert post_initial_records, "no saved messages after initial history"

        expected_wait_input = "[Automated continuation] The wait barrier has cleared. Please evaluate the current state and take the next step."
        expected_crashed_input = f"✖ [default] @default Kanban {child_task_id} worker crashed (pid gone); dispatcher will retry"
        post_initial_user_records = [record for record in post_initial_records if record["role"] == "user"]
        post_initial_assistant_records = [record for record in post_initial_records if record["role"] == "assistant"]
        assert len(post_initial_user_records) in (1, 2), post_initial_user_records
        assert len(post_initial_assistant_records) == len(post_initial_user_records), post_initial_records

        input_origins = []
        if len(post_initial_user_records) == 2:
            assert post_initial_user_records[0]["content"] == expected_wait_input, post_initial_user_records
            assert post_initial_user_records[1]["content"] == expected_crashed_input, post_initial_user_records
            input_origins = [
                {"input_id": post_initial_user_records[0]["id"], "origins": ["goal_wait_barrier_cleared"]},
                {"input_id": post_initial_user_records[1]["id"], "origins": ["kanban_crashed_notification"]},
            ]
        else:
            merged_content = post_initial_user_records[0]["content"]
            assert expected_wait_input in merged_content and expected_crashed_input in merged_content, post_initial_user_records
            input_origins = [{
                "input_id": post_initial_user_records[0]["id"],
                "origins": ["goal_wait_barrier_cleared", "kanban_crashed_notification"],
                "merged_from_observed_content": True,
            }]

        response_pairings = []
        used_input_ids = set()
        used_response_ids = set()
        for user_record in post_initial_user_records:
            next_user_order = next(
                (record["order"] for record in post_initial_user_records if record["order"] > user_record["order"]),
                len(saved_records),
            )
            responses = [
                record
                for record in post_initial_assistant_records
                if user_record["order"] < record["order"] < next_user_order
            ]
            assert len(responses) == 1, {"input": user_record, "responses_before_next_user": responses}
            response = responses[0]
            assert user_record["id"] not in used_input_ids, response_pairings
            assert response["id"] not in used_response_ids, response_pairings
            used_input_ids.add(user_record["id"])
            used_response_ids.add(response["id"])
            response_pairings.append({
                "input_id": user_record["id"],
                "input_order": user_record["order"],
                "input_content": user_record["content"],
                "response_id": response["id"],
                "response_order": response["order"],
                "response_content": response["content"],
                "response_finish_reason": response["finish_reason"],
            })
        assert len(response_pairings) == len(post_initial_user_records) == len(post_initial_assistant_records)
        assert all(pair["response_content"] == expected_primary_text for pair in response_pairings), response_pairings

        with kb.connect() as conn:
            final_child = kb.get_task(conn, child_task_id)
            final_dependent = kb.get_task(conn, dependent_task_id)
            final_sub = kb.list_notify_subs(conn, child_task_id)[0]
            final_event_rows = conn.execute(
                "SELECT id, kind, payload, run_id FROM task_events WHERE task_id = ? ORDER BY id",
                (child_task_id,),
            ).fetchall()
        final_events = [
            {"id": int(row["id"]), "kind": row["kind"], "payload": json.loads(row["payload"] or "{}"), "run_id": row["run_id"]}
            for row in final_event_rows
        ]
        spawned_events = [event for event in final_events if event["kind"] == "spawned"]
        crashed_events = [event for event in final_events if event["kind"] == "crashed"]
        completed_events = [event for event in final_events if event["kind"] == "completed"]
        assert len(spawned_events) == 1, final_events
        assert len(crashed_events) == 1, final_events
        assert len(completed_events) == 0, final_events
        assert crashed_events[0]["id"] == int(crashed_row["id"]), final_events
        assert crashed_events[0]["payload"].get("pid") == worker["pid"]
        assert final_child is not None and final_child.status == "ready" and final_child.worker_pid is None
        assert final_dependent is not None and final_dependent.status == "todo" and final_dependent.assignee is None
        assert int(final_sub["last_event_id"]) >= int(crashed_row["id"])

        message_complete_frame_records = [
            record
            for record in captured_frame_records
            if record.get("method") == "event" and record.get("type") == "message.complete"
        ]
        matching_sid_frames = [record for record in message_complete_frame_records if record.get("session_id") == sid]
        complete_texts = [
            str(record.get("frame", {}).get("params", {}).get("payload", {}).get("text", ""))
            for record in message_complete_frame_records
        ]
        matching_sid_complete_texts = [
            str(record.get("frame", {}).get("params", {}).get("payload", {}).get("text", ""))
            for record in matching_sid_frames
        ]
        assert message_complete_frame_records, "no message.complete frames captured"
        assert matching_sid_frames, {"expected_sid": sid, "message_complete_frames": message_complete_frame_records}
        assert expected_primary_text in matching_sid_complete_texts
        complete_identity_fields = [
            key
            for key in ("payload_message_id", "payload_id", "payload_turn_id")
            if any(record.get(key) is not None for record in matching_sid_frames)
        ]
        transport_response_identity_checks = []
        for pair in response_pairings:
            checks = []
            for field in complete_identity_fields:
                matched = [record for record in matching_sid_frames if str(record.get(field)) == str(pair["response_id"])]
                checks.append({"field": field, "matched_orders": [record["order"] for record in matched]})
                assert matched, {"response_pair": pair, "field": field, "matching_sid_frames": matching_sid_frames}
            transport_response_identity_checks.append({
                "response_id": pair["response_id"],
                "identity_fields_present": complete_identity_fields,
                "checks": checks,
                "limitation": None if complete_identity_fields else (
                    "captured message.complete frames expose sid and text but no SessionDB message id or turn id; "
                    "transport receipt cannot be paired to DB responses without using text as the key"
                ),
            })

        trace["raw_transport_frames"] = {
            "expected_sid": sid,
            "expected_session_key": session_key,
            "capture_reads": captured_reads,
            "all_frames": captured_frame_records,
        }
        trace["raw_message_complete_frames"] = {
            "expected_sid": sid,
            "expected_session_key": session_key,
            "all_message_complete_frames": message_complete_frame_records,
            "matching_sid_frames": matching_sid_frames,
            "complete_texts": complete_texts,
            "matching_sid_complete_texts": matching_sid_complete_texts,
            "transport_response_identity_checks": transport_response_identity_checks,
        }
        trace["input_response_identity"] = {
            "basis": "final SessionDB read after turn completion; initial history prefix is preserved and excluded only by exact prefix comparison",
            "saved_session_id": agent.session_id,
            "initial_history_len": initial_len,
            "saved_message_count": len(saved_messages),
            "post_initial_message_count": len(post_initial_records),
            "post_initial_user_count": len(post_initial_user_records),
            "post_initial_assistant_count": len(post_initial_assistant_records),
            "expected_wait_input": expected_wait_input,
            "expected_crashed_input": expected_crashed_input,
            "input_origins": input_origins,
            "pairings": response_pairings,
            "saved_records": saved_records,
        }
        trace["final_event_identity"] = {
            "notify_cursor_before": cursor_before,
            "notify_cursor_after": int(final_sub["last_event_id"]),
            "spawned_event_ids": [event["id"] for event in spawned_events],
            "crashed_event_ids": [event["id"] for event in crashed_events],
            "completed_event_ids": [event["id"] for event in completed_events],
            "events": final_events,
            "task_status_after_crash": final_child.status,
            "dependent_status_after_crash": final_dependent.status,
        }

        trace.update({
            "task_id": child_task_id,
            "dependent_task_id": dependent_task_id,
            "crash": {
                "worker_pid": crashed_payload["pid"],
                "exit_kind": crashed_payload["exit_kind"],
                "exit_code": crashed_payload["exit_code"],
                "crashed_event_id": int(crashed_row["id"]),
                "task_status_after_crash": child_after.status,
                "dependent_status_after_crash": dependent_after.status,
                "completed_events": int(completed_count),
            },
            "notification": {
                "notify_cursor_before": cursor_before,
                "notify_cursor_after": int(final_sub["last_event_id"]),
                "assistant_delivery_count": len(response_pairings),
                "message_complete_texts": complete_texts,
            },
            "goal": final_goal,
            "worker_ps_after_watcher": ps_snapshot(worker["pid"]),
            "model_requests": [
                {
                    "seq": req["seq"],
                    "model": req["payload"].get("model"),
                    "stub_kind": req.get("stub_kind"),
                    "stream": req["payload"].get("stream"),
                    "authorization": req["authorization"],
                }
                for req in good_requests
            ],
        })
    except BaseException as exc:
        trace["failure"] = {
            "stage": trace.get("stage"),
            "type": exc.__class__.__name__,
            "message": str(exc),
            "repr": repr(exc),
        }
        raise
    finally:
        try:
            drain_captured_frames("finally")
            trace["raw_transport_frames"] = {
                "expected_sid": trace.get("sid"),
                "expected_session_key": trace.get("session_key"),
                "capture_reads": captured_reads,
                "all_frames": captured_frame_records,
            }
            if worker_proc is not None:
                worker_ps_after_watcher = ps_snapshot(worker_proc.pid)
                trace["worker_ps_after_watcher"] = worker_ps_after_watcher
                if worker_ps_after_watcher["stdout"]:
                    worker_cleanup: dict[str, object] = {"action": "wait_after_watcher_stop", "pid": worker_proc.pid}
                    try:
                        worker_proc.wait(timeout=5)
                    except BaseException as cleanup_exc:
                        worker_cleanup["wait_error"] = repr(cleanup_exc)
                        try:
                            worker_proc.kill()
                        except BaseException as kill_exc:
                            worker_cleanup["kill_error"] = repr(kill_exc)
                        else:
                            try:
                                worker_proc.wait(timeout=2)
                            except BaseException as wait_exc:
                                worker_cleanup["wait_after_kill_error"] = repr(wait_exc)
                    worker_cleanup["returncode"] = worker_proc.returncode
                    trace["worker_cleanup"] = worker_cleanup
                else:
                    trace["worker_cleanup"] = {
                        "action": "already_reaped_by_product_watcher",
                        "pid": worker_proc.pid,
                        "returncode": worker_proc.returncode,
                    }
            trace["final_stage"] = trace.get("stage")
            trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

            crash_trace = trace.get("crash") or {}
            notification_trace = trace.get("notification") or {}
            watcher_trace = trace.get("watcher") or {}
            print(json.dumps({
                "abnormal_child_exit_evidence": {
                    "trace_path": str(trace_path),
                    "task_id": trace.get("task_id"),
                    "dependent_task_id": trace.get("dependent_task_id"),
                    "worker_pid": crash_trace.get("worker_pid"),
                    "exit_kind": crash_trace.get("exit_kind"),
                    "exit_code": crash_trace.get("exit_code"),
                    "crashed_event_id": crash_trace.get("crashed_event_id"),
                    "notify_cursor_before": notification_trace.get("notify_cursor_before", trace.get("notify_cursor_before")),
                    "notify_cursor_after": notification_trace.get("notify_cursor_after"),
                    "task_status_after_crash": crash_trace.get("task_status_after_crash"),
                    "dependent_status_after_crash": crash_trace.get("dependent_status_after_crash"),
                    "completed_events": crash_trace.get("completed_events"),
                    "assistant_delivery_count": notification_trace.get("assistant_delivery_count"),
                    "watcher_route": watcher_trace.get("route"),
                }
            }, ensure_ascii=False), file=sys.__stdout__)
        except BaseException as diagnostic_exc:
            trace["diagnostic_failure"] = {
                "type": diagnostic_exc.__class__.__name__,
                "message": str(diagnostic_exc),
                "repr": repr(diagnostic_exc),
            }
            try:
                trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            except BaseException:
                pass
        finally:
            release_or_close(session_db)


def test_saved_notification_survives_parent_process_exit_and_recovers_in_new_process(
    tmp_path,
    hermes_home,
    openai_stub,
):
    """親再起動復帰: 保存済み通知を新しい実親プロセスが読み込み後続turnまで進める。"""
    write_stub_config(hermes_home, openai_stub.base_url)

    from hermes_cli import kanban_db as kb
    from mcp_serve import _load_session_messages
    import hashlib

    def transcript_signature(messages):
        normalized = [
            {"role": str(msg.get("role", "")), "content": str(msg.get("content", ""))}
            for msg in messages
        ]
        raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return {
            "roles_content": normalized,
            "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }

    initial_history = []
    unique_seed = "restart recovery unique durable history t_54f9559c " + ("復帰履歴 " * 180)
    for idx in range(12):
        initial_history.append({"role": "user", "content": f"{unique_seed} user {idx}: " + ("u" * 160)})
        initial_history.append({"role": "assistant", "content": f"{unique_seed} assistant {idx}: " + ("a" * 160)})
    initial_signature = transcript_signature(initial_history)

    sid_old = "isolated-restart-old-sid"
    sid_new = "isolated-restart-new-sid"
    session_key = "agent:main:tui:isolated-restart-session"
    old_marker = tmp_path / "old-parent-ready.json"
    new_marker = tmp_path / "new-parent-ready.json"

    old_parent = run_isolated_parent_process(
        hermes_home,
        openai_stub.base_url,
        sid=sid_old,
        session_key=session_key,
        marker=old_marker,
        mode="hold",
        initial_history=initial_history,
    )
    new_parent = None
    child_proc = None
    try:
        def old_parent_ready():
            if old_marker.exists():
                return True
            if old_parent.poll() is not None:
                out, err = old_parent.communicate(timeout=1)
                raise AssertionError({"old_parent_exited": old_parent.returncode, "stdout": out, "stderr": err})
            return False

        assert_eventually("old isolated parent process ready", old_parent_ready, timeout=15)
        old_payload = json.loads(old_marker.read_text(encoding="utf-8"))
        assert int(old_payload["pid"]) == old_parent.pid
        old_guard = old_payload["guard"]
        assert old_guard["bypass"] is None
        assert Path(old_guard["home_env"]).resolve().is_relative_to(tmp_path.resolve())
        assert Path(old_guard["hermes_home_env"]).resolve() == hermes_home.resolve()
        assert Path(old_guard["state_db"]).resolve() == (hermes_home / "state.db").resolve()
        assert not (Path(old_guard["home_env"]) / ".hermes").resolve() == hermes_home.resolve()
        assert old_payload["initial_history_signature"] == initial_signature
        saved_initial, saved_initial_error = load_session_messages_from_home(hermes_home, session_key)
        assert saved_initial_error is None
        assert saved_initial is not None
        assert transcript_signature(saved_initial[: len(initial_history)]) == initial_signature

        with kb.connect() as conn:
            child_task_id = kb.create_task(
                conn,
                title="restart recovery child",
                body="completion is saved before the old parent process exits",
                assignee="personal",
                created_by="parent",
            )
            dependent_task_id = kb.create_task(
                conn,
                title="restart recovery dependent remains unassigned",
                body="new parent evaluates this after reading saved notification",
                assignee=None,
                created_by="parent",
                parents=[child_task_id],
            )
            kb.add_notify_sub(conn, task_id=child_task_id, platform="tui", chat_id=session_key)
            cursor_before = int(kb.list_notify_subs(conn, child_task_id)[0]["last_event_id"])

        alive_marker = tmp_path / "restart-child.alive.json"
        release_fifo = tmp_path / "restart-child.release.fifo"
        os.mkfifo(release_fifo)
        child_proc = run_real_child_process(
            hermes_home,
            session_key,
            task_id=child_task_id,
            alive_marker=alive_marker,
            release_fifo=release_fifo,
            result="child result",
            summary="restart child success",
        )
        assert_eventually("restart child alive marker", lambda: alive_marker.exists(), timeout=5)
        alive_payload = json.loads(alive_marker.read_text(encoding="utf-8"))
        assert int(alive_payload["pid"]) == child_proc.pid

        with open(release_fifo, "wb", buffering=0) as gate:
            gate.write(b"1")
        child = collect_real_child_process(child_proc, timeout=10)
        child_proc = None
        assert child["pid"] > 0
        assert child["pid"] == int(alive_payload["pid"])
        assert child["returncode"] == 0
        assert child["complete_ok"] is True
        assert child["claim_ok"] is True
        assert child["status_before"] == "ready"
        assert child["status_after_claim"] == "running"
        assert child["status_after"] == "done"
        assert child["event_kind"] == "completed"
        assert child["summary"] == "restart child success"
        completed_event_id = int(child["event_id"])
        assert completed_event_id > cursor_before

        old_parent.terminate()
        old_stdout, old_stderr = old_parent.communicate(timeout=5)
        assert old_parent.returncode is not None
        with kb.connect() as conn:
            sub_after_old_exit = kb.list_notify_subs(conn, child_task_id)[0]
        assert int(sub_after_old_exit["last_event_id"]) == cursor_before

        expected_primary_text = openai_stub.set_primary_reply(
            task_id=dependent_task_id,
            reason="新親プロセスが保存済み通知から復帰",
        )
        new_parent = run_isolated_parent_process(
            hermes_home,
            openai_stub.base_url,
            sid=sid_new,
            session_key=session_key,
            marker=new_marker,
            mode="recover",
            expected_text=expected_primary_text,
            timeout=35,
        )
        assert_eventually("new isolated parent process ready", lambda: new_marker.exists(), timeout=15)
        new_stdout, new_stderr = new_parent.communicate(timeout=45)
        assert new_parent.returncode == 0, {"stdout": new_stdout, "stderr": new_stderr, "returncode": new_parent.returncode}
        evidence = None
        evidence_path = Path(str(new_marker) + ".evidence")
        if evidence_path.exists():
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        else:
            marker_idx = new_stdout.rfind('{"evidence"')
            if marker_idx != -1:
                evidence = json.loads(new_stdout[marker_idx:].strip())
        assert evidence is not None, {"stdout": new_stdout, "stderr": new_stderr, "returncode": new_parent.returncode}
        delivered_subs = [sub for sub in evidence["subs"] if sub["task_id"] == child_task_id and sub["chat_id"] == session_key]
        new_guard = evidence["guard"]
        assert new_guard["bypass"] is None
        assert Path(new_guard["home_env"]).resolve().is_relative_to(tmp_path.resolve())
        assert Path(new_guard["hermes_home_env"]).resolve() == hermes_home.resolve()
        assert Path(new_guard["state_db"]).resolve() == (hermes_home / "state.db").resolve()
        assert not (Path(new_guard["home_env"]) / ".hermes").resolve() == hermes_home.resolve()
        assert evidence["initial_history_signature"] is None
        assert evidence["restored_history_signature"] == initial_signature
        assert evidence["post_turn_initial_history_signature"] == initial_signature
        assert "restart recovery seed" not in json.dumps(evidence["messages"], ensure_ascii=False)
        assert expected_primary_text not in json.dumps(evidence["restored_history_signature"], ensure_ascii=False)
        assert len(delivered_subs) == 1
        assert int(delivered_subs[0]["last_event_id"]) >= completed_event_id
        assert expected_primary_text in evidence["assistant_texts"]
        assert evidence["assistant_texts"].count(expected_primary_text) == 1
        completed_events = [event for event in evidence["events"] if event["task_id"] == child_task_id and event["kind"] == "completed"]
        assert len(completed_events) == 1
        assert int(completed_events[0]["id"]) == completed_event_id
        with kb.connect() as conn:
            child_after = kb.get_task(conn, child_task_id)
            dependent_after = kb.get_task(conn, dependent_task_id)
            sub_after_new = kb.list_notify_subs(conn, child_task_id)[0]
        assert child_after.status == "done"
        assert dependent_after.status == "ready"
        assert dependent_after.assignee is None
        assert int(sub_after_new["last_event_id"]) >= completed_event_id

        model_requests = openai_stub.requests()
        primary_requests = [
            req for req in model_requests
            if req["authorization"] == "Bearer test-key" and req["payload"].get("model") == PRIMARY_MODEL
        ]
        notification_texts = [str(req["payload"].get("messages", "")) for req in primary_requests]
        assert any(child_task_id in text and "restart child success" in text for text in notification_texts)

        print(json.dumps({
            "parent_restart_recovery_evidence": {
                "old_parent_pid": old_parent.pid,
                "old_parent_returncode": old_parent.returncode,
                "new_parent_pid": evidence["pid"],
                "task_id": child_task_id,
                "dependent_task_id": dependent_task_id,
                "completed_event_id": completed_event_id,
                "child_pid": child["pid"],
                "child_returncode": child["returncode"],
                "child_complete_ok": child["complete_ok"],
                "child_claim_ok": child["claim_ok"],
                "child_event_kind": child["event_kind"],
                "child_summary": child["summary"],
                "cursor_before": cursor_before,
                "cursor_after_old_exit": int(sub_after_old_exit["last_event_id"]),
                "cursor_after_new_process": int(sub_after_new["last_event_id"]),
                "assistant_delivery_count": evidence["assistant_texts"].count(expected_primary_text),
            }
        }, ensure_ascii=False), file=sys.__stdout__)
    finally:
        if child_proc is not None and child_proc.poll() is None:
            child_proc.kill()
            child_proc.wait(timeout=2)
        if old_parent.poll() is None:
            old_parent.terminate()
            try:
                old_parent.wait(timeout=3)
            except Exception:
                old_parent.kill()
                old_parent.wait(timeout=3)
        if new_parent is not None and new_parent.poll() is None:
            new_parent.terminate()
            try:
                new_parent.wait(timeout=3)
            except Exception:
                new_parent.kill()
                new_parent.wait(timeout=3)
