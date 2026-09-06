"""T1-24: structured-output schema on delegate_task.

Per-task ``output_schema`` (JSON Schema object): the child receives the
schema as an explicit output contract, the parent validates the child's
final answer with jsonschema, and on failure sends exactly ONE bounded
retry turn carrying the validation errors. Result entries gain
``schema_valid`` / ``schema_errors`` / ``schema_retries`` ONLY when a
schema was requested — schema-less calls keep a byte-identical result
shape (wire-shape pinning).

Pattern from: github/copilot-cli ctx.agent(prompt, {schema}) — PATTERN
ONLY, zero code/prompt text copied (proprietary).
"""

import json
import shlex
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.delegation_context import (
    get_delegated_child_snapshot_scope,
    is_delegated_child_context,
)
from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _run_single_child,
    delegate_task,
)
from tools.delegation_output_schema import (
    append_output_contract,
    build_retry_message,
    coerce_output_schema,
    validate_output,
)

ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "zip": {"type": "string"},
    },
    "required": ["city"],
}


# ---------------------------------------------------------------------------
# Helper-module unit tests
# ---------------------------------------------------------------------------


class TestValidateOutput:
    def test_valid_json_matching_schema(self):
        ok, errors = validate_output('{"city": "Berlin"}', ADDRESS_SCHEMA)
        assert ok is True
        assert errors == []

    def test_json_violating_schema_reports_errors(self):
        ok, errors = validate_output('{"zip": "10115"}', ADDRESS_SCHEMA)
        assert ok is False
        assert errors
        assert any("city" in e for e in errors)

    def test_non_json_text_reports_parse_error(self):
        ok, errors = validate_output("I could not produce JSON, sorry.", ADDRESS_SCHEMA)
        assert ok is False
        assert errors

    def test_code_fenced_json_is_accepted(self):
        text = '```json\n{"city": "Oslo"}\n```'
        ok, errors = validate_output(text, ADDRESS_SCHEMA)
        assert ok is True
        assert errors == []

    def test_json_embedded_in_prose_is_extracted(self):
        text = 'Here is the result:\n{"city": "Lima"}\nHope that helps!'
        ok, _ = validate_output(text, ADDRESS_SCHEMA)
        assert ok is True

    def test_empty_text_is_invalid(self):
        ok, errors = validate_output("", ADDRESS_SCHEMA)
        assert ok is False
        assert errors


class TestCoerceOutputSchema:
    def test_valid_schema_passes(self):
        schema, err = coerce_output_schema(ADDRESS_SCHEMA)
        assert schema == ADDRESS_SCHEMA
        assert err is None

    def test_none_passes_through(self):
        schema, err = coerce_output_schema(None)
        assert schema is None
        assert err is None

    def test_non_dict_is_rejected(self):
        schema, err = coerce_output_schema("not a schema")
        assert schema is None
        assert err

    def test_invalid_json_schema_is_rejected(self):
        schema, err = coerce_output_schema({"type": 42})
        assert schema is None
        assert err


class TestPromptPlumbing:
    def test_contract_block_carries_schema(self):
        out = append_output_contract("base context", ADDRESS_SCHEMA)
        assert "base context" in out
        assert "OUTPUT CONTRACT" in out
        assert '"city"' in out

    def test_contract_block_without_prior_context(self):
        out = append_output_contract(None, ADDRESS_SCHEMA)
        assert "OUTPUT CONTRACT" in out

    def test_retry_message_carries_verbatim_errors(self):
        msg = build_retry_message(["'city' is a required property"])
        assert "'city' is a required property" in msg
        assert "JSON" in msg


# ---------------------------------------------------------------------------
# Tool-schema surface (one-time static field)
# ---------------------------------------------------------------------------


class TestToolSchemaSurface:
    def test_output_schema_on_task_items(self):
        item_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"][
            "items"
        ]["properties"]
        assert "output_schema" in item_props
        assert item_props["output_schema"]["type"] == "object"
        # never required
        assert "output_schema" not in DELEGATE_TASK_SCHEMA["parameters"][
            "properties"
        ]["tasks"]["items"]["required"]

    def test_output_schema_advertised_per_task_only(self):
        """output_schema is advertised inside tasks[] items (the only spawn
        shape); the legacy top-level param stays handler-accepted but out
        of the schema."""
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        assert "output_schema" not in props
        task_props = props["tasks"]["items"]["properties"]
        assert task_props["output_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# _run_single_child validation + bounded retry
# ---------------------------------------------------------------------------


class _StubChild:
    """Minimal child agent double (mirrors test_delegate_kanban_isolation)."""

    tool_progress_callback = None
    _delegate_saved_tool_names: list = []
    _credential_pool = None
    _subagent_id = None  # skip registry
    _delegate_depth = 1
    _parent_subagent_id = None
    _delegate_output_schema: dict | None = None
    model = "test-model"
    session_id = "stub-child-session"
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list = []

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, **_kwargs):
        self.calls.append(user_message)
        text = self.responses.pop(0)
        return {
            "final_response": text,
            "completed": True,
            "api_calls": 1,
            "messages": [],
        }

    def close(self):
        return None


class _StubParent:
    _current_task_id = None
    _delegate_depth = 0

    def _touch_activity(self, _desc):
        return None


def _run(child):
    return _run_single_child(0, "produce the address", child, _StubParent())


class TestRunSingleChildSchemaValidation:
    def test_valid_first_try_no_retry(self):
        child = _StubChild(['{"city": "Berlin"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "completed"
        assert entry["schema_valid"] is True
        assert "schema_errors" not in entry
        assert len(child.calls) == 1

    def test_invalid_then_retry_then_valid(self):
        child = _StubChild(["not json at all", '{"city": "Oslo"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is True
        assert entry["schema_retries"] == 1
        # retry turn carried the validation errors
        assert len(child.calls) == 2
        assert "rejected" in child.calls[1] or "JSON" in child.calls[1]
        # final summary is the retried (valid) answer
        assert json.loads(entry["summary"])["city"] == "Oslo"

    def test_invalid_twice_surfaces_errors_and_stops(self):
        child = _StubChild(["nope", "still nope"])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert entry["schema_retries"] == 1
        # exactly ONE retry — bounded
        assert len(child.calls) == 2

    def test_retry_exception_degrades_to_invalid(self):
        child = _StubChild(["nope"])
        child._delegate_output_schema = ADDRESS_SCHEMA

        original = child.run_conversation

        def flaky(user_message, task_id=None, **kw):
            if child.calls:
                raise RuntimeError("child died on retry")
            return original(user_message, task_id=task_id, **kw)

        child.run_conversation = flaky
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]

    def test_no_schema_keeps_legacy_result_shape(self):
        """Schema-less calls must not gain new keys (wire-shape pinning)."""
        child = _StubChild(['{"city": "Berlin"}'])
        entry = _run(child)
        assert "schema_valid" not in entry
        assert "schema_errors" not in entry
        assert "schema_retries" not in entry
        assert len(child.calls) == 1

    def test_failed_child_skips_validation(self):
        """A child with no output never gets a schema retry turn."""
        child = _StubChild([""])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "failed"
        assert len(child.calls) == 1
        assert entry.get("schema_valid") is False

    def test_schema_failure_reported_as_failed_not_completed(self):
        """Regression: a final answer that still violates the declared
        output contract after the bounded retry (here the classic empty
        ``{}`` fallback) must be reported status="failed", not
        "completed". Otherwise the batch report prints a ✓ and
        orchestrators that read only status/icon accept an empty verdict
        — schema_valid/schema_errors carry the detail, but status must
        agree with them."""
        child = _StubChild(["not json at all", "{}"])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["schema_errors"]
        assert entry["status"] == "failed"
        # the failed entry names the schema violation, not the generic
        # "no response" error — the child DID respond, unusably
        assert "output_schema" in entry.get("error", "")
        # the invalid final text is still propagated for debugging
        assert entry["summary"] == "{}"

    def test_schema_failure_without_retry_reported_as_failed(self):
        """Same class, first-try path: retry turn raises, leaving the
        original non-JSON answer in place — status must still be failed."""
        child = _StubChild(["nope"])
        child._delegate_output_schema = ADDRESS_SCHEMA

        original = child.run_conversation

        def flaky(user_message, task_id=None, **kw):
            if child.calls:
                raise RuntimeError("child died on retry")
            return original(user_message, task_id=task_id, **kw)

        child.run_conversation = flaky
        entry = _run(child)
        assert entry["schema_valid"] is False
        assert entry["status"] == "failed"

    def test_schema_valid_entry_still_completed(self):
        """Guard: schema_valid=True keeps status="completed" untouched."""
        child = _StubChild(['{"city": "Berlin"}'])
        child._delegate_output_schema = ADDRESS_SCHEMA
        entry = _run(child)
        assert entry["status"] == "completed"
        assert "error" not in entry


# ---------------------------------------------------------------------------
# delegate_task dispatch-time schema handling
# ---------------------------------------------------------------------------


def _make_mock_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    return parent


class TestDelegateTaskDispatch:
    def test_schema_retry_preserves_child_context_scope_and_restores_parent(self):
        records = []

        child = _StubChild(["not json", '{"city": "Kyoto"}'])
        child.session_id = "schema-retry-child-session"
        child._delegate_output_schema = ADDRESS_SCHEMA

        original = child.run_conversation

        def recording_run(user_message, task_id=None, **kw):
            records.append(
                {
                    "is_child": is_delegated_child_context(),
                    "scope": get_delegated_child_snapshot_scope(),
                }
            )
            return original(user_message, task_id=task_id, **kw)

        child.run_conversation = recording_run

        def fake_build(**_kwargs):
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
            patch(
                "tools.delegate_tool._build_child_preserving_parent_tools",
                side_effect=fake_build,
            ),
        ):
            out = delegate_task(
                goal="produce the address",
                output_schema=ADDRESS_SCHEMA,
                parent_agent=_make_mock_parent(),
            )

        payload = json.loads(out)
        assert payload["results"][0]["schema_valid"] is True
        assert [record["is_child"] for record in records] == [True, True]
        assert records[0]["scope"] == "schema-retry-child-session"
        assert records[1]["scope"] == records[0]["scope"]
        assert is_delegated_child_context() is False
        assert get_delegated_child_snapshot_scope() is None

    def test_schema_retry_kanban_complete_is_rejected_as_child_mutation(
        self,
        monkeypatch,
        tmp_path,
    ):
        from tests.tools.test_delegate_kanban_isolation import _make_running_kanban_task
        from tools import kanban_tools

        kb, tid, workspace, _attachments_root = _make_running_kanban_task(
            monkeypatch,
            tmp_path,
        )
        attempted = []

        child = _StubChild(["not json", '{"city": "Kyoto"}'])
        child.session_id = "schema-retry-kanban-child-session"
        child._delegate_output_schema = ADDRESS_SCHEMA
        original = child.run_conversation

        def retry_attempts_kanban_complete(user_message, task_id=None, **kw):
            if child.calls:
                attempted.append(
                    kanban_tools._handle_complete({"summary": "retry child completion leak"})
                )
            return original(user_message, task_id=task_id, **kw)

        child.run_conversation = retry_attempts_kanban_complete

        class Parent:
            _current_task_id = tid
            _delegate_depth = 0

            def _touch_activity(self, _desc):
                return None

        entry = _run_single_child(0, "produce the address", child, Parent())

        assert entry["schema_valid"] is True
        assert attempted
        assert "delegate_task child" in attempted[0]

        conn = kb.connect()
        try:
            task = kb.get_task(conn, tid)
            run = kb.latest_run(conn, tid)
        finally:
            conn.close()

        assert task.status == "running"
        assert run.status == "running"
        assert workspace.is_dir()

    def test_schema_retry_uses_same_real_local_environment_child_snapshot(
        self,
        monkeypatch,
        tmp_path,
    ):
        from tools.environments.local import LocalEnvironment

        parent_home = tmp_path / "parent_home"
        child_home = tmp_path / "child_home"
        parent_cwd = tmp_path / "parent_cwd"
        child_cwd = tmp_path / "child_cwd"
        terminal_tmp = tmp_path / "terminal_tmp"
        for directory in (parent_home, child_home, parent_cwd, child_cwd, terminal_tmp):
            directory.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(parent_home))
        monkeypatch.setenv("TERMINAL_TEMP_DIR", str(terminal_tmp))
        monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

        env = LocalEnvironment(cwd=str(parent_cwd), timeout=10)
        child = _StubChild([])
        child.session_id = "schema-retry-localenv-child"
        child._delegate_output_schema = ADDRESS_SCHEMA
        seen = []

        def run_with_terminal_state(user_message, task_id=None, **_kw):
            seen.append(
                {
                    "is_child": is_delegated_child_context(),
                    "scope": get_delegated_child_snapshot_scope(),
                }
            )
            if len(seen) == 1:
                result = env.execute(
                    f"export HERMES_HOME={shlex.quote(str(child_home))}; "
                    f"cd {shlex.quote(str(child_cwd))}; "
                    'printf "%s:%s" "$HERMES_HOME" "$(pwd -P)"',
                    timeout=10,
                )
                assert result["returncode"] == 0, result["output"]
                home_text, cwd_text = result["output"].strip().split(":", 1)
                assert home_text == str(child_home)
                assert Path(cwd_text).resolve() == child_cwd.resolve()
                return {
                    "final_response": "not json",
                    "completed": True,
                    "api_calls": 1,
                    "messages": [],
                }

            result = env.execute('printf "%s:%s" "$HERMES_HOME" "$(pwd -P)"', timeout=10)
            assert result["returncode"] == 0, result["output"]
            retry_state = result["output"].strip()
            home_text, cwd_text = retry_state.split(":", 1)
            assert home_text == str(child_home)
            assert Path(cwd_text).resolve() == child_cwd.resolve()
            return {
                "final_response": '{"city": "Kyoto"}',
                "completed": True,
                "api_calls": 1,
                "messages": [],
            }

        child.run_conversation = run_with_terminal_state

        try:
            entry = _run_single_child(0, "produce the address", child, _StubParent())
            assert entry["schema_valid"] is True
            assert [record["is_child"] for record in seen] == [True, True]
            assert seen[0]["scope"] == "schema-retry-localenv-child"
            assert seen[1]["scope"] == seen[0]["scope"]
            parent_state = env.execute('printf "%s:%s" "$HERMES_HOME" "$(pwd -P)"', timeout=10)
            assert parent_state["returncode"] == 0, parent_state["output"]
            home_text, cwd_text = parent_state["output"].strip().split(":", 1)
            assert home_text == str(parent_home)
            assert Path(cwd_text).resolve() == parent_cwd.resolve()
            assert is_delegated_child_context() is False
            assert get_delegated_child_snapshot_scope() is None
        finally:
            env.cleanup()

    def test_schema_retry_exception_restores_parent_context(self, monkeypatch, tmp_path):
        from tools.environments.local import LocalEnvironment

        parent_home = tmp_path / "exception_parent_home"
        retry_home = tmp_path / "exception_retry_home"
        parent_cwd = tmp_path / "exception_parent_cwd"
        retry_cwd = tmp_path / "exception_retry_cwd"
        terminal_tmp = tmp_path / "exception_terminal_tmp"
        for directory in (parent_home, retry_home, parent_cwd, retry_cwd, terminal_tmp):
            directory.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(parent_home))
        monkeypatch.setenv("TERMINAL_TEMP_DIR", str(terminal_tmp))
        monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

        env = LocalEnvironment(cwd=str(parent_cwd), timeout=10)
        child = _StubChild(["not json"])
        child.session_id = "schema-retry-exception-child"
        child._delegate_output_schema = ADDRESS_SCHEMA
        original = child.run_conversation
        seen = []

        def retry_raises(user_message, task_id=None, **kw):
            seen.append(
                {
                    "is_child": is_delegated_child_context(),
                    "scope": get_delegated_child_snapshot_scope(),
                }
            )
            if len(seen) == 2:
                result = env.execute(
                    f"export HERMES_HOME={shlex.quote(str(retry_home))}; "
                    f"cd {shlex.quote(str(retry_cwd))}; "
                    'printf "%s:%s" "$HERMES_HOME" "$(pwd -P)"',
                    timeout=10,
                )
                assert result["returncode"] == 0, result["output"]
                home_text, cwd_text = result["output"].strip().split(":", 1)
                assert home_text == str(retry_home)
                assert Path(cwd_text).resolve() == retry_cwd.resolve()
                raise RuntimeError("intentional retry failure")
            return original(user_message, task_id=task_id, **kw)

        child.run_conversation = retry_raises

        try:
            entry = _run_single_child(0, "produce the address", child, _StubParent())
            assert entry["schema_valid"] is False
            assert [record["is_child"] for record in seen] == [True, True]
            assert seen[1]["scope"] == "schema-retry-exception-child"
            parent_state = env.execute('printf "%s:%s" "$HERMES_HOME" "$(pwd -P)"', timeout=10)
            assert parent_state["returncode"] == 0, parent_state["output"]
            home_text, cwd_text = parent_state["output"].strip().split(":", 1)
            assert home_text == str(parent_home)
            assert Path(cwd_text).resolve() == parent_cwd.resolve()
            assert is_delegated_child_context() is False
            assert get_delegated_child_snapshot_scope() is None
        finally:
            env.cleanup()

    def test_non_dict_output_schema_rejected(self):
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
        ):
            out = delegate_task(
                tasks=[
                    {"goal": "Summarize the release notes for module A", "output_schema": "not-a-dict"},
                    {"goal": "Summarize the release notes for module B"},
                ],
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert payload.get("error")
        assert "output_schema" in payload["error"]

    def test_invalid_json_schema_rejected_at_dispatch(self):
        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
        ):
            out = delegate_task(
                tasks=[
                    {"goal": "Summarize the release notes for module A", "output_schema": {"type": 42}},
                    {"goal": "Summarize the release notes for module B"},
                ],
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert payload.get("error")
        assert "output_schema" in payload["error"]

    def test_child_receives_contract_and_schema_attr(self):
        """The built child carries the schema attr and its context gains
        the output-contract block."""
        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            child = _StubChild(['{"city": "Rio"}'])
            return child

        with (
            patch("tools.delegate_tool._load_config", return_value={}),
            patch(
                "tools.delegate_tool._resolve_delegation_credentials",
                return_value={
                    "provider": None,
                    "model": None,
                    "base_url": None,
                    "api_key": None,
                    "api_mode": None,
                },
            ),
            patch(
                "tools.delegate_tool._build_child_preserving_parent_tools",
                side_effect=fake_build,
            ),
        ):
            out = delegate_task(
                goal="produce the address",
                context="base context",
                output_schema=ADDRESS_SCHEMA,
                parent_agent=_make_mock_parent(),
            )
        payload = json.loads(out)
        assert "OUTPUT CONTRACT" in (captured.get("context") or "")
        results = payload.get("results") or []
        assert results and results[0].get("schema_valid") is True
