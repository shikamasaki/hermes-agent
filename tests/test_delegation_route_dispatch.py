"""delegate_task must resolve credentials PER TASK when routing is enabled.

Child construction is stubbed out; these tests assert only which
provider:model each task was built with, and that parent/child isolation
holds. No network, OAuth, auth store, or credential values are involved.
"""

import json

import pytest

from agent import delegation_usage_cache as duc
import tools.delegate_tool as dt


ROUTE_A = {
    "id": "route-a-standard",
    "backend": "native",
    "provider": "provider-a",
    "model": "model-a-advanced",
    "model_class": "advanced",
    "task_difficulties": ["standard", "complex"],
    "capabilities": ["coding", "reasoning", "tool_use"],
    "priority": 20,
    "reserve_remaining_percent": 15,
}

ROUTE_B = {
    "id": "route-b-routine",
    "backend": "native",
    "provider": "provider-b",
    "model": "model-b-balanced",
    "model_class": "balanced",
    "task_difficulties": ["routine", "standard"],
    "capabilities": ["reasoning", "tool_use", "long_context"],
    "priority": 30,
    "reserve_remaining_percent": 10,
    "usage_window_prefixes": ["Route B Models"],
}

ROUTED_CFG = {
    "routing": {"enabled": True},
    "routes": [ROUTE_A, ROUTE_B],
    "max_concurrent_children": 4,
}


class _Parent:
    """Minimal stand-in for the parent agent."""

    provider = "parent-provider"
    model = "parent-model"
    api_key = "parent-secret-key"
    base_url = "https://api.parent-provider.com"
    session_id = "parent-session"
    _delegate_depth = 0


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
    duc.reset_refresh_state()
    monkeypatch.setattr(duc, "_spawn_refresh", lambda provider: None)
    monkeypatch.setattr(dt, "_load_config", lambda: ROUTED_CFG)
    monkeypatch.setattr(
        dt,
        "_available_route_providers",
        lambda routes: frozenset({"provider-a", "provider-b"}),
    )

    def _resolve(requested=None, target_model=None, **kwargs):
        table = {
            "provider-a": {
                "provider": "provider-a",
                "base_url": "https://provider-a.example/api",
                "api_key": "provider-a-key",
                "api_mode": "mode_a",
            },
            "provider-b": {
                "provider": "provider-b",
                "base_url": "https://provider-b.example/api",
                "api_key": "provider-b-key",
                "api_mode": "mode_b",
                "project_id": "project-b-123",
            },
        }
        if requested not in table:
            raise RuntimeError(f"unknown provider {requested}")
        return {**table[requested], "model": target_model}

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

    def _fake_finalize(results, *a, **kw):
        return results

    monkeypatch.setattr(
        dt, "_run_child_lifecycle",
        lambda *a, **kw: {"task": 0, "status": "success", "summary": "done"},
    )
    return built


def _run(harness, **kwargs):
    dt.delegate_task(parent_agent=_Parent(), background=False, **kwargs)
    return harness


class TestPerTaskRouting:
    def test_batch_tasks_build_children_on_independent_routes(self, harness):
        built = _run(
            harness,
            tasks=[
                {
                    "goal": "Rename the deprecated helper across the utils package",
                    "difficulty": "routine",
                },
                {
                    "goal": "Redesign the scheduler to remove the global lock",
                    "difficulty": "complex",
                },
            ],
        )
        assert len(built) == 2
        assert built[0]["override_provider"] == "provider-b"
        assert built[0]["model"] == "model-b-balanced"
        assert built[1]["override_provider"] == "provider-a"
        assert built[1]["model"] == "model-a-advanced"

    def test_batch_shares_one_availability_and_usage_preflight(
        self, harness, monkeypatch
    ):
        availability_calls = []
        usage_calls = []
        original_build_usage_view = duc.build_route_usage_view

        def _available(routes):
            availability_calls.append(tuple(route.id for route in routes))
            return frozenset({"provider-a", "provider-b"})

        def _usage(*args, **kwargs):
            usage_calls.append(tuple(args[0]))
            return original_build_usage_view(*args, **kwargs)

        monkeypatch.setattr(dt, "_available_route_providers", _available)
        monkeypatch.setattr(duc, "build_route_usage_view", _usage)
        _run(
            harness,
            tasks=[
                {"goal": "Rename the deprecated helper everywhere", "difficulty": "routine"},
                {"goal": "Redesign the scheduler without global locks", "difficulty": "complex"},
            ],
        )
        assert len(availability_calls) == 1
        assert len(usage_calls) == 1

    def test_single_goal_uses_top_level_routing_fields(self, harness):
        built = _run(
            harness,
            goal="Summarize the changelog entries for the release notes",
            difficulty="routine",
            difficulty_reason="mechanical summarization",
        )
        assert built[0]["override_provider"] == "provider-b"

    def test_explicit_per_task_route_overrides_difficulty(self, harness):
        built = _run(
            harness,
            tasks=[
                {
                    "goal": "Rename the deprecated helper across the utils package",
                    "difficulty": "routine",
                    "route": "route-a-standard",
                },
                {
                    "goal": "Redesign the scheduler to remove the global lock",
                    "difficulty": "complex",
                },
            ],
        )
        assert built[0]["override_provider"] == "provider-a"

    def test_capabilities_route_the_task(self, harness):
        built = _run(
            harness,
            goal="Read the entire 400k-token log bundle and extract failures",
            difficulty="standard",
            required_capabilities=["long_context"],
        )
        assert built[0]["override_provider"] == "provider-b"


class TestChildIsolation:
    def test_child_never_inherits_parent_credentials_when_routed(self, harness):
        built = _run(
            harness,
            goal="Redesign the scheduler to remove the global lock cleanly",
            difficulty="complex",
        )
        kwargs = built[0]
        assert kwargs["override_api_key"] == "provider-a-key"
        assert kwargs["override_api_key"] != _Parent.api_key
        assert kwargs["override_base_url"] == "https://provider-a.example/api"
        assert kwargs["override_provider"] != _Parent.provider

    def test_public_route_metadata_omits_raw_usage_and_reason_text(self):
        class _Child:
            _delegate_route_decision = {
                "route_id": "route-a-standard",
                "provider": "provider-a",
                "model": "model-a-advanced",
                "model_class": "advanced",
                "difficulty": "standard",
                "usage_freshness": "stale",
                "usage_remaining_percent": 12.3456,
                "usage_age_seconds": 678.9,
                "reason": "account alice@example.com has 12.3456% remaining",
                "exclusions": (
                    "route 'other' has 2% remaining, below its 10% reserve",
                ),
            }

        metadata = dt._public_route_metadata(_Child())
        assert metadata == {
            "id": "route-a-standard",
            "provider": "provider-a",
            "model": "model-a-advanced",
            "model_class": "advanced",
            "difficulty": "standard",
            "usage_freshness": "stale",
            "explicit_override": False,
            "reserve_bypassed": False,
            "exclusion_codes": ["usage_reserve"],
        }
        blob = json.dumps(metadata)
        assert "alice@example.com" not in blob
        assert "12.3456" not in blob
        assert "678.9" not in blob

    def test_cross_provider_children_do_not_share_api_mode(self, harness):
        built = _run(
            harness,
            tasks=[
                {
                    "goal": "Rename the deprecated helper across the utils package",
                    "difficulty": "routine",
                },
                {
                    "goal": "Redesign the scheduler to remove the global lock",
                    "difficulty": "complex",
                },
            ],
        )
        assert built[0]["override_api_mode"] == "mode_b"
        assert built[1]["override_api_mode"] == "mode_a"


class TestLegacyDispatchUnchanged:
    def test_legacy_config_resolves_once_for_all_tasks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
        legacy_cfg = {"provider": "provider-a", "model": "pinned-model"}
        monkeypatch.setattr(dt, "_load_config", lambda: legacy_cfg)

        calls = []

        def _resolve(requested=None, target_model=None, **kwargs):
            calls.append(requested)
            return {
                "provider": "provider-a",
                "model": target_model,
                "base_url": "https://provider-a.example/api",
                "api_key": "provider-a-key",
                "api_mode": "mode_a",
            }

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", _resolve
        )
        built = []
        monkeypatch.setattr(
            dt,
            "_build_child_preserving_parent_tools",
            lambda **kw: built.append(kw) or type("C", (), {"session_id": "c"})(),
        )
        monkeypatch.setattr(
            dt, "_run_child_lifecycle",
            lambda *a, **kw: {"task": 0, "status": "success", "summary": "done"},
        )

        dt.delegate_task(
            parent_agent=_Parent(),
            background=False,
            tasks=[
                {"goal": "Rename the deprecated helper across the utils package"},
                {"goal": "Redesign the scheduler to remove the global lock"},
            ],
        )
        assert [b["model"] for b in built] == ["pinned-model", "pinned-model"]
        assert calls == ["provider-a"]  # resolved once, not per task
