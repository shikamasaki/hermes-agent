"""delegate_task dispatch for the claude-p backend (PR4).

Asserts that claude-p routes bypass native child construction entirely, that
mixed native+Claude batches keep deterministic result ordering, that a task
which has started running tools is never rerouted, and that the native and
legacy dispatch paths are untouched.

Child construction and the ``claude`` subprocess are both stubbed: no real
CLI is invoked, no credentials are read, no network is used.
"""

import json
import threading
import weakref

import pytest

from agent import claude_p_backend as cb
from agent import delegation_usage_cache as duc
import tools.delegate_tool as dt


CLAUDE_ROUTE = {
    "id": "claude-sub",
    "backend": "claude-p",
    "provider": "claude-p",
    "model": "claude-opus-5",
    "model_class": "frontier",
    "task_difficulties": ["complex", "frontier"],
    "capabilities": ["reasoning", "review"],
    "tool_profile": "review",
    "priority": 10,
}

CODEX_ROUTE = {
    "id": "codex-standard",
    "backend": "native",
    "provider": "openai-codex",
    "model": "gpt-5.6-sol",
    "model_class": "advanced",
    "task_difficulties": ["standard", "routine"],
    "capabilities": ["coding", "reasoning", "tool_use"],
    "priority": 20,
}

MIXED_CFG = {
    "routing": {"enabled": True},
    "routes": [CLAUDE_ROUTE, CODEX_ROUTE],
    "max_concurrent_children": 4,
}

LEGACY_CFG = {"provider": "openai-codex", "model": "gpt-5.6-sol"}


class _Parent:
    provider = "anthropic"
    model = "claude-opus-5"
    api_key = "parent-secret-key"
    base_url = "https://api.anthropic.com"
    session_id = "parent-session"
    _delegate_depth = 0


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
    duc.reset_refresh_state()
    monkeypatch.setattr(duc, "_spawn_refresh", lambda provider: None)
    monkeypatch.setattr(dt, "_load_config", lambda: MIXED_CFG)
    monkeypatch.setattr(
        dt,
        "_available_route_providers",
        lambda routes: frozenset({"claude-p", "openai-codex"}),
    )
    cb.reset_cooldowns()
    cb.reset_workdir_locks()

    def _resolve(requested=None, target_model=None, **kwargs):
        if requested != "openai-codex":
            raise RuntimeError(f"claude-p must never resolve a runtime provider: {requested}")
        return {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-key",
            "api_mode": "codex_responses",
            "model": target_model,
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _resolve)

    built = []

    class _Child:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.session_id = f"child-{len(built)}"

    def _fake_build(**kwargs):
        built.append(kwargs)
        return _Child(kwargs)

    monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", _fake_build)

    ran = []

    def _fake_run(task_index, goal, child=None, parent_agent=None, **kw):
        ran.append((task_index, child))
        if isinstance(child, dt.ClaudePChildSpec):
            return dt._run_claude_p_child(task_index, goal, child, parent_agent)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": f"native done {task_index}",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": child.kwargs.get("model"),
            "exit_reason": "completed",
            "tokens": {"input": 1, "output": 1},
            "tool_trace": [],
        }

    monkeypatch.setattr(dt, "_run_single_child", _fake_run)
    monkeypatch.setattr(dt, "_finalize_child_results", lambda *a, **kw: None)

    return {"built": built, "ran": ran}


def _stub_claude_success(monkeypatch, summary="claude done"):
    import json as _json

    def _fake_task(request, *, write_capable, cancel_event=None):
        return cb.ClaudePRunResult(
            status="completed",
            summary=summary,
            error=None,
            session_id=request.session_id,
            claude_session_id="sess-remote",
            num_turns=3,
            cost_usd=0.02,
            model=request.model,
            duration_seconds=1.0,
            exit_reason="completed",
            tokens={"input": 100, "output": 50},
            model_usage={request.model: {"inputTokens": 100}},
        )

    monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
    return _json


def _run(**kwargs):
    return dt.delegate_task(parent_agent=_Parent(), background=False, **kwargs)


# ---------------------------------------------------------------------------
# claude-p dispatch bypasses native construction
# ---------------------------------------------------------------------------


class TestClaudePDispatch:
    def test_child_prompt_forbids_external_publication_and_credentials(self):
        prompt = dt._build_claude_p_prompt("Implement the requested change", None)
        for forbidden_action in (
            "commit",
            "push",
            "pull requests",
            "publish",
            "deploy",
            "send messages",
            "make payments",
            "access credentials",
        ):
            assert forbidden_action in prompt

    def test_claude_route_builds_no_native_child(self, harness, monkeypatch):
        _stub_claude_success(monkeypatch)
        _run(goal="Redesign the scheduler to remove the global lock", difficulty="complex")

        # No AIAgent was constructed for the claude-p task.
        assert harness["built"] == []
        assert len(harness["ran"]) == 1
        spec = harness["ran"][0][1]
        assert isinstance(spec, dt.ClaudePChildSpec)
        assert spec.params["tool_profile"] == "review"
        assert spec.params["model"] == "claude-opus-5"
        assert spec.write_capable is False

    def test_nested_claude_child_inherits_parent_identity_and_depth(
        self, harness, monkeypatch
    ):
        _stub_claude_success(monkeypatch)
        monkeypatch.setattr(
            dt,
            "_load_config",
            lambda: {**MIXED_CFG, "max_spawn_depth": 3},
        )
        parent = _Parent()
        parent._delegate_depth = 1
        setattr(parent, "_subagent_id", "parent-subagent")

        dt.delegate_task(
            parent_agent=parent,
            background=False,
            goal="Review the scheduler",
            difficulty="complex",
        )

        spec = harness["ran"][0][1]
        assert isinstance(spec, dt.ClaudePChildSpec)
        assert spec._parent_subagent_id == "parent-subagent"
        assert spec._delegate_depth == 2

    def test_subagent_auto_approve_is_captured_into_claude_p_run_params(self, harness, monkeypatch):
        cfg = {
            **MIXED_CFG,
            "subagent_auto_approve": True,
            "routes": [
                {
                    **CLAUDE_ROUTE,
                    "tool_profile": "default",
                    "capabilities": ["coding", "reasoning", "review"],
                }
            ],
        }
        monkeypatch.setattr(dt, "_load_config", lambda: cfg)
        captured = []

        def _fake_task(request, *, write_capable, cancel_event=None):
            captured.append((request, write_capable))
            return cb.ClaudePRunResult(
                status="completed", summary="done", error=None,
                session_id=request.session_id, claude_session_id=None,
                num_turns=1, cost_usd=0.0, model=request.model,
                duration_seconds=0.1, exit_reason="completed",
                tokens={"input": 1, "output": 1}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        _run(goal="Edit the code", difficulty="complex", required_capabilities=["coding"])

        assert captured
        request, write_capable = captured[0]
        assert request.tool_profile == "default"
        assert request.auto_approve is True
        assert request.max_budget_usd is None
        assert write_capable is True

    def test_result_is_normalized_into_the_delegate_entry_shape(self, harness, monkeypatch):
        _stub_claude_success(monkeypatch, summary="reviewed the scheduler")
        out = _run(goal="Redesign the scheduler to remove the global lock", difficulty="complex")

        import json

        payload = json.loads(out)
        entry = payload["results"][0]
        assert entry["task_index"] == 0
        assert entry["status"] == "completed"
        assert entry["summary"] == "reviewed the scheduler"
        assert entry["tokens"] == {"input": 100, "output": 50}
        assert entry["api_calls"] == 3
        assert entry["exit_reason"] == "completed"
        assert entry["cost_status"] == "notional"
        assert entry["route"]["backend"] == "claude-p"
        assert entry["route"]["id"] == "claude-sub"

    def test_session_id_is_opaque_and_never_enters_the_usage_cache(
        self, harness, monkeypatch, tmp_path
    ):
        _stub_claude_success(monkeypatch)
        _run(goal="Redesign the scheduler to remove the global lock", difficulty="complex")

        spec = harness["ran"][0][1]
        # An opaque per-task UUID, not a reused/global identifier.
        assert len(spec.session_id) == 36 and spec.session_id.count("-") == 4

        cache_file = tmp_path / "usage.json"
        if cache_file.exists():
            assert spec.session_id not in cache_file.read_text()
            assert "sess-remote" not in cache_file.read_text()

    def test_claude_p_never_resolves_an_api_key_credential(self, harness, monkeypatch):
        # _resolve raises for anything but openai-codex, so a claude-p task
        # that tried to resolve credentials would blow up the dispatch.
        _stub_claude_success(monkeypatch)
        out = _run(goal="Redesign the scheduler to remove the global lock", difficulty="complex")
        import json

        assert json.loads(out)["results"][0]["status"] == "completed"

    def test_output_schema_retries_once_in_same_session_and_remaining_budget(
        self, harness, monkeypatch
    ):
        calls = []
        remote_id = "12345678-1234-4234-8234-123456789abc"

        def _fake_task(request, *, write_capable, cancel_event=None):
            calls.append(request)
            summary = '{"ok": true}' if len(calls) == 2 else "not json"
            return cb.ClaudePRunResult(
                status="completed",
                summary=summary,
                error=None,
                session_id=request.session_id,
                claude_session_id=remote_id,
                num_turns=2,
                cost_usd=1.0,
                model=request.model,
                duration_seconds=1.0,
                exit_reason="completed",
                tokens={"input": 10, "output": 5},
                model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        out = _run(
            goal="Return structured review status",
            difficulty="complex",
            output_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            },
        )
        import json

        entry = json.loads(out)["results"][0]
        assert entry["schema_valid"] is True
        assert entry["schema_retries"] == 1
        assert len(calls) == 2
        assert calls[1].resume_session_id == remote_id
        assert calls[1].max_turns < calls[0].max_turns
        assert calls[1].max_budget_usd < calls[0].max_budget_usd
        assert calls[1].workdir == calls[0].workdir
        assert entry["tokens"] == {"input": 20, "output": 10}
        assert entry["cost_usd"] == 2.0

    def test_valid_output_schema_does_not_retry(self, harness, monkeypatch):
        calls = []

        def _fake_task(request, *, write_capable, cancel_event=None):
            calls.append(request)
            return cb.ClaudePRunResult(
                status="completed",
                summary='{"ok": true}',
                error=None,
                session_id=request.session_id,
                claude_session_id="12345678-1234-4234-8234-123456789abc",
                num_turns=1,
                cost_usd=0.1,
                model=request.model,
                duration_seconds=0.1,
                exit_reason="completed",
                tokens={"input": 1, "output": 1},
                model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        out = _run(
            goal="Return structured review status",
            difficulty="complex",
            output_schema={"type": "object", "required": ["ok"]},
        )
        import json

        entry = json.loads(out)["results"][0]
        assert entry["schema_valid"] is True
        assert "schema_retries" not in entry
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Mixed native + Claude batch
# ---------------------------------------------------------------------------


class TestMixedBatch:
    def test_batch_mixes_backends_with_deterministic_ordering(self, harness, monkeypatch):
        _stub_claude_success(monkeypatch)
        out = _run(
            tasks=[
                {
                    "goal": "Rename the deprecated helper across the utils package",
                    "difficulty": "routine",
                },
                {
                    "goal": "Redesign the scheduler to remove the global lock",
                    "difficulty": "complex",
                },
                {
                    "goal": "Document the retry semantics in the client module",
                    "difficulty": "routine",
                },
            ],
        )
        import json

        results = json.loads(out)["results"]
        # Deterministic ordering by task_index regardless of completion order.
        assert [r["task_index"] for r in results] == [0, 1, 2]
        assert results[0]["route"]["backend"] == "native"
        assert results[1]["route"]["backend"] == "claude-p"
        assert results[2]["route"]["backend"] == "native"

        # Exactly the two native tasks built an in-process child.
        assert len(harness["built"]) == 2
        specs = [c for _, c in harness["ran"] if isinstance(c, dt.ClaudePChildSpec)]
        assert len(specs) == 1


# ---------------------------------------------------------------------------
# No reroute after tools have started
# ---------------------------------------------------------------------------


class TestNoPostStartReroute:
    def test_tools_started_marker_is_set_before_execution(self, harness, monkeypatch):
        seen = {}

        def _fake_task(request, *, write_capable, cancel_event=None):
            # By the time the process runs, the spec is already marked as
            # started — the selector must never revisit this task.
            seen["started"] = spec_holder["spec"].tools_started
            return cb.ClaudePRunResult(
                status="completed", summary="ok", error=None,
                session_id=request.session_id, claude_session_id=None,
                num_turns=1, cost_usd=0.0, model=request.model,
                duration_seconds=0.1, exit_reason="completed",
                tokens={"input": 0, "output": 0}, model_usage={},
            )

        spec_holder = {}
        original = dt._run_claude_p_child

        def _capture(task_index, goal, spec, parent_agent=None):
            spec_holder["spec"] = spec
            return original(task_index, goal, spec, parent_agent)

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        monkeypatch.setattr(dt, "_run_claude_p_child", _capture)

        _run(goal="Redesign the scheduler to remove the global lock", difficulty="complex")
        assert seen["started"] is True

    def test_mid_flight_failure_is_reported_not_rerouted(self, harness, monkeypatch):
        def _fake_task(request, *, write_capable, cancel_event=None):
            return cb.ClaudePRunResult(
                status="error", summary=None,
                error="claude -p reported an error",
                session_id=request.session_id, claude_session_id="sess-x",
                num_turns=5, cost_usd=0.01, model=request.model,
                duration_seconds=2.0, exit_reason="error_during_execution",
                tokens={"input": 10, "output": 5}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        out = _run(goal="Redesign the scheduler to remove the global lock", difficulty="complex")

        import json

        entry = json.loads(out)["results"][0]
        # The failure surfaces as a blocker on the SAME route; no second
        # (native) route ran, and no extra child was built.
        assert entry["status"] == "error"
        assert entry["route"]["id"] == "claude-sub"
        assert harness["built"] == []
        assert len(harness["ran"]) == 1

        # A mid-execution failure must NOT cool the route down — cooldown is
        # only for pre-start failures.
        assert cb.is_route_in_cooldown("claude-sub") is False

    def test_pre_start_failure_cools_the_route_down(self, harness, monkeypatch):
        def _fake_task(request, *, write_capable, cancel_event=None):
            return cb.ClaudePRunResult(
                status="error", summary=None,
                error="claude -p failed to start (OSError)",
                session_id=request.session_id, claude_session_id=None,
                num_turns=None, cost_usd=None, model=request.model,
                duration_seconds=0.2, exit_reason="spawn_error",
                tokens={"input": 0, "output": 0}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        _run(goal="Redesign the scheduler to remove the global lock", difficulty="complex")

        assert cb.is_route_in_cooldown("claude-sub") is True

        # The next pre-execution selection now excludes the cooling-down
        # route, so a subsequent task falls through to the native one.
        from agent import delegation_routing as _dr

        catalog = _dr.load_route_catalog(MIXED_CFG)
        assert dt._claude_p_cooling_down(catalog.routes) == frozenset({"claude-sub"})

        decision = _dr.select_route(
            catalog,
            _dr.RouteRequest(difficulty=_dr.TaskDifficulty.COMPLEX),
            usage=_dr.UsageView({}),
            available_providers=frozenset({"claude-p", "openai-codex"}),
            cooling_down_route_ids=dt._claude_p_cooling_down(catalog.routes),
        )
        assert decision.route_id != "claude-sub"


# ---------------------------------------------------------------------------
# Live registry / control lifecycle
# ---------------------------------------------------------------------------


class _LiveParent:
    session_id = "parent-live-session"

    def __init__(self):
        self.touches = []

    def _touch_activity(self, desc):
        self.touches.append(desc)


class TestClaudePLiveRegistryLifecycle:
    def _spec(self, parent):
        spec = dt.ClaudePChildSpec(
            task_index=0,
            goal="Audit the scheduler",
            context=None,
            params={
                "model": "claude-opus-5",
                "difficulty": "complex",
                "tool_profile": "review",
                "max_turns": 5,
                "max_budget_usd": 1.0,
                "timeout_seconds": 30,
            },
            route_decision={"route_id": "claude-sub", "provider": "claude-p"},
            output_schema=None,
            workdir=".",
        )
        spec._delegate_parent_ref = weakref.ref(parent)
        spec._parent_session_id = parent.session_id
        spec._delegation_id = "deleg-live"
        return spec

    def test_running_claude_p_child_is_listed_until_run_returns(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)
        started = threading.Event()
        release = threading.Event()

        def _fake_task(request, *, write_capable, cancel_event=None):
            started.set()
            assert release.wait(timeout=2), "test did not release mocked claude-p run"
            return cb.ClaudePRunResult(
                status="completed",
                summary="done",
                error=None,
                session_id=request.session_id,
                claude_session_id=None,
                num_turns=1,
                cost_usd=0.0,
                model=request.model,
                duration_seconds=0.1,
                exit_reason="completed",
                tokens={"input": 1, "output": 1},
                model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        result_holder = {}
        thread = threading.Thread(
            target=lambda: result_holder.setdefault(
                "result", dt._run_single_child(0, "Audit the scheduler", spec, parent)
            )
        )
        thread.start()
        try:
            assert started.wait(timeout=2)
            active = dt.list_active_subagents()
            assert [entry["subagent_id"] for entry in active] == [spec._subagent_id]
            listed = json.loads(dt._handle_control_action("list", None, None, parent))
            assert listed["count"] == 1
            assert listed["subagents"][0]["subagent_id"] == spec._subagent_id
            assert listed["subagents"][0]["goal"] == "Audit the scheduler"
            assert listed["subagents"][0]["accepting_steer"] is False
            release.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert result_holder["result"]["status"] == "completed"
            after = json.loads(dt._handle_control_action("list", None, None, parent))
            assert after["count"] == 0
        finally:
            release.set()
            thread.join(timeout=2)
            dt._unregister_subagent(spec._subagent_id)

    def test_stop_requests_claude_p_cancel_event(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)
        started = threading.Event()
        cancelled = threading.Event()

        def _fake_task(request, *, write_capable, cancel_event=None):
            started.set()
            assert cancel_event is not None
            assert cancel_event.wait(timeout=2), "stop did not signal claude-p cancel"
            cancelled.set()
            return cb.ClaudePRunResult(
                status="interrupted",
                summary=None,
                error="claude -p was interrupted",
                session_id=request.session_id,
                claude_session_id=None,
                num_turns=0,
                cost_usd=0.0,
                model=request.model,
                duration_seconds=0.1,
                exit_reason="interrupted",
                tokens={"input": 0, "output": 0},
                model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        thread = threading.Thread(
            target=lambda: dt._run_single_child(0, "Audit the scheduler", spec, parent)
        )
        thread.start()
        try:
            assert started.wait(timeout=2)
            out = json.loads(
                dt._handle_control_action("stop", spec._subagent_id, None, parent)
            )
            assert out["status"] == "interrupt_requested"
            assert cancelled.wait(timeout=2)
            thread.join(timeout=2)
            assert not thread.is_alive()
        finally:
            thread.join(timeout=2)
            dt._unregister_subagent(spec._subagent_id)


    def test_parent_interrupt_propagates_to_claude_p_child_and_unregisters(self, monkeypatch):
        parent = _LiveParent()
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        spec = self._spec(parent)
        started = threading.Event()
        cancelled = threading.Event()

        def _fake_task(request, *, write_capable, cancel_event=None):
            started.set()
            assert cancel_event is not None
            assert cancel_event.wait(timeout=2), "parent interrupt did not signal claude-p cancel"
            cancelled.set()
            return cb.ClaudePRunResult(
                status="interrupted", summary="partial", error="interrupted",
                session_id=request.session_id, claude_session_id=None,
                num_turns=1, cost_usd=0.0, model=request.model,
                duration_seconds=0.1, exit_reason="interrupted",
                tokens={"input": 1, "output": 1}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        thread = threading.Thread(target=lambda: dt._run_single_child(0, "Audit the scheduler", spec, parent))
        thread.start()
        try:
            assert started.wait(timeout=2)
            assert spec in parent._active_children
            spec.interrupt("ctrl-c")
            assert cancelled.wait(timeout=2)
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert spec not in parent._active_children
            assert json.loads(dt._handle_control_action("list", None, None, parent))["count"] == 0
        finally:
            spec.interrupt("cleanup")
            thread.join(timeout=2)
            dt._unregister_subagent(spec._subagent_id)

    def test_claude_p_spec_has_activity_summary_for_async_progress(self):
        parent = _LiveParent()
        spec = self._spec(parent)

        summary = spec.get_activity_summary()

        assert summary["api_call_count"] == 0
        assert summary["current_tool"] == "claude-p"
        assert isinstance(summary["last_activity_ts"], float)

    def test_claude_p_parent_heartbeat_runs_while_child_is_waiting(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)
        started = threading.Event()
        release = threading.Event()
        monkeypatch.setattr(dt, "_HEARTBEAT_INTERVAL", 0.01)

        def _fake_task(request, *, write_capable, cancel_event=None):
            started.set()
            assert release.wait(timeout=2), "test did not release mocked claude-p run"
            return cb.ClaudePRunResult(
                status="completed", summary="done", error=None,
                session_id=request.session_id, claude_session_id=None,
                num_turns=1, cost_usd=0.0, model=request.model,
                duration_seconds=0.1, exit_reason="completed",
                tokens={"input": 1, "output": 1}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        thread = threading.Thread(
            target=lambda: dt._run_single_child(0, "Audit the scheduler", spec, parent)
        )
        thread.start()
        try:
            assert started.wait(timeout=2)
            for _ in range(100):
                if parent.touches:
                    break
                threading.Event().wait(0.01)
            assert any("claude-p" in desc for desc in parent.touches)
        finally:
            release.set()
            thread.join(timeout=2)
            dt._unregister_subagent(spec._subagent_id)

    def test_claude_p_emits_completion_progress_event(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)
        events = []
        spec.tool_progress_callback = lambda event_type, **kwargs: events.append(
            (event_type, kwargs)
        )

        def _fake_task(request, *, write_capable, cancel_event=None):
            return cb.ClaudePRunResult(
                status="completed",
                summary="done",
                error=None,
                session_id=request.session_id,
                claude_session_id=None,
                num_turns=1,
                cost_usd=0.0,
                model=request.model,
                duration_seconds=0.1,
                exit_reason="completed",
                tokens={"input": 1, "output": 2},
                model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        result = dt._run_single_child(0, "Audit the scheduler", spec, parent)

        assert result["status"] == "completed"
        complete_events = [event for event in events if event[0] == "subagent.complete"]
        assert complete_events
        assert complete_events[-1][1]["status"] == "completed"
        assert complete_events[-1][1]["summary"] == "done"

    def test_claude_p_emits_start_and_single_completion_progress_events(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)
        events = []
        spec.tool_progress_callback = lambda event_type, **kwargs: events.append(event_type)

        def _fake_task(request, *, write_capable, cancel_event=None):
            return cb.ClaudePRunResult(
                status="completed", summary="done", error=None,
                session_id=request.session_id, claude_session_id=None,
                num_turns=1, cost_usd=0.0, model=request.model,
                duration_seconds=0.1, exit_reason="completed",
                tokens={"input": 1, "output": 2}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        dt._run_single_child(0, "Audit the scheduler", spec, parent)

        assert events.count("subagent.start") == 1
        assert events.count("subagent.complete") == 1

    def test_claude_p_exception_is_structured_and_unregistered(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)

        def _fake_task(request, *, write_capable, cancel_event=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        result = dt._run_single_child(0, "Audit the scheduler", spec, parent)

        assert result["status"] == "error"
        assert "boom" in result["error"]

        after = json.loads(dt._handle_control_action("list", None, None, parent))
        assert after["count"] == 0

    def test_schema_retry_is_skipped_if_stop_requested_after_good_summary(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)
        spec._delegate_output_schema = {"type": "object", "required": ["ok"]}
        calls = []

        def _fake_task(request, *, write_capable, cancel_event=None):
            calls.append(request)
            if len(calls) == 1:
                assert cancel_event is not None
                cancel_event.set()
                return cb.ClaudePRunResult(
                    status="completed", summary='{"ok": true}', error=None,
                    session_id=request.session_id, claude_session_id="12345678-1234-4234-8234-123456789abc",
                    num_turns=1, cost_usd=0.0, model=request.model,
                    duration_seconds=0.1, exit_reason="completed",
                    tokens={"input": 1, "output": 2}, model_usage={},
                )
            return cb.ClaudePRunResult(
                status="interrupted", summary=None, error="interrupted",
                session_id=request.session_id, claude_session_id=None,
                num_turns=0, cost_usd=0.0, model=request.model,
                duration_seconds=0.1, exit_reason="interrupted",
                tokens={"input": 0, "output": 0}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        result = dt._run_single_child(0, "Audit the scheduler", spec, parent)

        assert result["summary"] == '{"ok": true}'
        assert result["schema_valid"] is True
        assert len(calls) == 1


    def test_schema_retry_interrupt_preserves_first_summary(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)
        spec._delegate_output_schema = {"type": "object", "required": ["ok"]}
        calls = []

        def _fake_task(request, *, write_capable, cancel_event=None):
            calls.append(request)
            if len(calls) == 1:
                return cb.ClaudePRunResult(
                    status="completed", summary="not json but useful", error=None,
                    session_id=request.session_id, claude_session_id="12345678-1234-4234-8234-123456789abc",
                    num_turns=1, cost_usd=0.1, model=request.model,
                    duration_seconds=0.1, exit_reason="completed",
                    tokens={"input": 1, "output": 2}, model_usage={},
                )
            return cb.ClaudePRunResult(
                status="interrupted", summary=None, error="interrupted",
                session_id=request.session_id, claude_session_id=None,
                num_turns=0, cost_usd=0.0, model=request.model,
                duration_seconds=0.1, exit_reason="interrupted",
                tokens={"input": 0, "output": 0}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        result = dt._run_single_child(0, "Audit the scheduler", spec, parent)

        assert len(calls) == 2
        assert result["summary"] == "not json but useful"
        assert result["schema_valid"] is False
        assert result["schema_retries"] == 1

    def test_claude_p_steer_rejection_names_unsupported_channel(self, monkeypatch):
        parent = _LiveParent()
        spec = self._spec(parent)
        started = threading.Event()
        release = threading.Event()

        def _fake_task(request, *, write_capable, cancel_event=None):
            started.set()
            assert release.wait(timeout=2), "test did not release mocked claude-p run"
            return cb.ClaudePRunResult(
                status="completed", summary="done", error=None,
                session_id=request.session_id, claude_session_id=None,
                num_turns=1, cost_usd=0.0, model=request.model,
                duration_seconds=0.1, exit_reason="completed",
                tokens={"input": 1, "output": 1}, model_usage={},
            )

        monkeypatch.setattr(cb, "run_claude_p_task", _fake_task)
        thread = threading.Thread(
            target=lambda: dt._run_single_child(0, "Audit the scheduler", spec, parent)
        )
        thread.start()
        try:
            assert started.wait(timeout=2)
            payload = json.loads(
                dt._handle_control_action("steer", spec._subagent_id, "focus", parent)
            )
            assert "does not support live steering" in payload["error"]
            assert "finishing" not in payload["error"]
        finally:
            release.set()
            thread.join(timeout=2)
            dt._unregister_subagent(spec._subagent_id)


# ---------------------------------------------------------------------------
# Native / legacy paths unchanged
# ---------------------------------------------------------------------------


class TestNativeAndLegacyUnchanged:
    def test_native_route_still_builds_an_in_process_child(self, harness, monkeypatch):
        _stub_claude_success(monkeypatch)
        _run(goal="Rename the deprecated helper across the utils package", difficulty="routine")

        assert len(harness["built"]) == 1
        assert harness["built"][0]["override_provider"] == "openai-codex"
        assert harness["built"][0]["override_api_key"] == "codex-key"
        assert not isinstance(harness["ran"][0][1], dt.ClaudePChildSpec)

    def test_legacy_config_path_is_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
        monkeypatch.setattr(dt, "_load_config", lambda: LEGACY_CFG)

        def _resolve(requested=None, target_model=None, **kwargs):
            return {
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "codex-key",
                "api_mode": "codex_responses",
                "model": target_model,
            }

        monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _resolve)

        built = []

        class _Child:
            def __init__(self, kwargs):
                self.kwargs = kwargs
                self.session_id = "legacy-child"

        def _fake_build(**kwargs):
            built.append(kwargs)
            return _Child(kwargs)

        monkeypatch.setattr(dt, "_build_child_preserving_parent_tools", _fake_build)
        monkeypatch.setattr(
            dt,
            "_run_single_child",
            lambda task_index, goal, child=None, parent_agent=None, **kw: {
                "task_index": task_index,
                "status": "completed",
                "summary": "legacy done",
                "api_calls": 1,
                "duration_seconds": 0.1,
                "model": "gpt-5.6-sol",
                "exit_reason": "completed",
                "tokens": {"input": 1, "output": 1},
                "tool_trace": [],
            },
        )
        monkeypatch.setattr(dt, "_finalize_child_results", lambda *a, **kw: None)

        out = _run(goal="Rename the deprecated helper across the utils package")

        import json

        entry = json.loads(out)["results"][0]
        assert entry["status"] == "completed"
        # Legacy path carries no route metadata at all.
        assert "route" not in entry
        assert len(built) == 1
        assert not isinstance(built[0], dt.ClaudePChildSpec)
