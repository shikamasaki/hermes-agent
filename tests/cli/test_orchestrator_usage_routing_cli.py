"""CLI integration tests for main-orchestrator usage-aware routing (Fallback fallback).

``HermesCLI._resolve_turn_agent_config`` (hermes_cli/cli_agent_setup_mixin.py)
is the per-turn hook that decides model/runtime/signature before
``_init_agent`` rebuilds ``self.agent``. These tests prove that when
``agent.orchestrator_usage_routing`` is enabled and configured, that hook:

* switches to the Fallback fallback (native provider-fallback /
  model-fallback) when the cached Primary remaining is <= the configured
  threshold;
* fails CLOSED to the current/primary Primary runtime if Fallback runtime/auth
  resolution raises (never exposes the exception, never raises out of the
  turn hook);
* recovers back to Primary on a fresh good reading after being on Fallback;
* preserves the existing `/fast` request_overrides behavior for whichever
  model ends up selected.

All usage-cache reads and runtime/auth resolution are mocked — no real
network/auth is exercised.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from agent.delegation_routing import ProviderUsage

PRIMARY_PROVIDER = "provider-primary"
PRIMARY_MODEL = "model-primary"
FALLBACK_PROVIDER = "provider-fallback"
FALLBACK_MODEL = "model-fallback"

_ROUTING_CONFIG = {
    "agent": {
        "orchestrator_usage_routing": {
            "enabled": True,
            "primary_provider": PRIMARY_PROVIDER,
            "primary_model": PRIMARY_MODEL,
            "fallback_provider": FALLBACK_PROVIDER,
            "fallback_model": FALLBACK_MODEL,
            "switch_at_remaining_percent": 10,
            "restore_above_remaining_percent": 10,
            "usage_ttl_seconds": 900,
            "usage_stale_seconds": 7200,
            "primary_usage_window_prefixes": ["Session", "Weekly"],
        }
    }
}


def _make_shell(provider=PRIMARY_PROVIDER, model=PRIMARY_MODEL, service_tier=None, agent=None):
    return SimpleNamespace(
        model=model,
        api_key="sk-test",
        base_url="https://example.invalid",
        provider=provider,
        requested_provider=provider,
        api_mode="primary_responses",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        provider_project_id=None,
        service_tier=service_tier,
        agent=agent,
    )


def _bound(shell):
    from cli import HermesCLI

    # _resolve_turn_agent_config calls self._apply_orchestrator_usage_routing,
    # a sibling mixin method — bind it too so a bare SimpleNamespace shell
    # (no real MRO) can resolve it, matching how test_fast_command.py and
    # test_credential_pool_routing.py bind single unbound methods onto a
    # SimpleNamespace stand-in for HermesCLI.
    shell._apply_orchestrator_usage_routing = (
        HermesCLI._apply_orchestrator_usage_routing.__get__(shell)
    )
    return HermesCLI._resolve_turn_agent_config.__get__(shell)


def _fallback_runtime():
    return {
        "provider": FALLBACK_PROVIDER,
        "requested_provider": FALLBACK_PROVIDER,
        "api_key": "fallback-key",
        "base_url": "https://fallback.example.invalid",
        "api_mode": "antigravity",
        "command": None,
        "args": [],
        "credential_pool": None,
        "project_id": "fallback-project",
        "model": FALLBACK_MODEL,
    }


def test_cli_route_signature_is_stable_and_project_sensitive():
    from hermes_cli.cli_agent_setup_mixin import _agent_route_signature

    runtime = _fallback_runtime()
    first = _agent_route_signature(FALLBACK_MODEL, runtime)
    second = _agent_route_signature(FALLBACK_MODEL, dict(runtime))
    changed = _agent_route_signature(
        FALLBACK_MODEL, {**runtime, "project_id": "other-project"}
    )

    assert first == second
    assert first != changed
    assert len(first) == 8


class TestCliSwitchesToFallbackOnLowUsage:
    def test_low_remaining_switches_signature_and_provider(self):
        shell = _make_shell()
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=5.0, freshness="fresh", age_seconds=10
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=_fallback_runtime(),
            ),
        ):
            route = _bound(shell)("hi")

        assert route["model"] == FALLBACK_MODEL
        assert route["runtime"]["provider"] == FALLBACK_PROVIDER
        assert route["runtime"]["project_id"] == "fallback-project"
        assert FALLBACK_MODEL in route["signature"]
        assert FALLBACK_PROVIDER in route["signature"]

    def test_explicit_non_policy_model_is_not_overridden(self):
        shell = _make_shell(model="gpt-5.4")
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER,
            remaining_percent=5.0,
            freshness="fresh",
            age_seconds=10,
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider") as resolver,
        ):
            route = _bound(shell)("hi")

        assert route["model"] == "gpt-5.4"
        assert route["runtime"]["provider"] == PRIMARY_PROVIDER
        resolver.assert_not_called()

    def test_healthy_remaining_stays_on_primary(self):
        shell = _make_shell()
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=80.0, freshness="fresh", age_seconds=10
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_resolve,
        ):
            route = _bound(shell)("hi")

        assert route["model"] == PRIMARY_MODEL
        assert route["runtime"]["provider"] == PRIMARY_PROVIDER
        # Primary path must not need to re-resolve a different runtime.
        mock_resolve.assert_not_called()


class TestCliFailsClosedOnFallbackAuthFailure:
    def test_fallback_resolution_raises_falls_back_to_current_primary_runtime(self):
        shell = _make_shell()
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=2.0, freshness="fresh", age_seconds=10
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=RuntimeError("secret-token-xyz should never leak"),
            ),
        ):
            route = _bound(shell)("hi")

        # Fail CLOSED: stay on the CLI's current/primary runtime.
        assert route["model"] == PRIMARY_MODEL
        assert route["runtime"]["provider"] == PRIMARY_PROVIDER
        assert route["runtime"]["api_key"] == "sk-test"

    def test_fallback_resolution_raise_message_never_reaches_route(self):
        """The raw exception text must not leak into the route dict at all."""
        shell = _make_shell()
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=2.0, freshness="fresh", age_seconds=10
        )
        secret_text = "sk-should-not-appear-in-any-string-field"
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                side_effect=RuntimeError(secret_text),
            ),
        ):
            route = _bound(shell)("hi")

        serialized = repr(route)
        assert secret_text not in serialized


class TestCliRecoversToPrimaryOnFreshGoodReading:
    def test_on_fallback_fresh_good_reading_recovers(self):
        shell = _make_shell(provider=FALLBACK_PROVIDER, model=FALLBACK_MODEL)
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=90.0, freshness="fresh", age_seconds=5
        )
        primary_runtime = {
            "provider": PRIMARY_PROVIDER,
            "requested_provider": PRIMARY_PROVIDER,
            "api_key": "primary-key",
            "base_url": "https://primary.example.invalid",
            "api_mode": "primary_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
            "model": PRIMARY_MODEL,
        }
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=primary_runtime,
            ),
        ):
            route = _bound(shell)("hi")

        assert route["model"] == PRIMARY_MODEL
        assert route["runtime"]["provider"] == PRIMARY_PROVIDER

    def test_on_fallback_stale_good_reading_does_not_recover(self):
        shell = _make_shell(provider=FALLBACK_PROVIDER, model=FALLBACK_MODEL)
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=90.0, freshness="stale", age_seconds=5000
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_resolve,
        ):
            route = _bound(shell)("hi")

        assert route["model"] == FALLBACK_MODEL
        assert route["runtime"]["provider"] == FALLBACK_PROVIDER
        mock_resolve.assert_not_called()


class TestCliCurrentIdentityFollowsActiveAgentNotShellConfig:
    """The shell's own ``self.provider``/``self.model`` stay pinned to the

    configured PRIMARY for the life of the process (see
    ``hermes_cli.model_switch``) — they are never mutated by orchestrator
    usage routing. The only place the *actually active* provider/model is
    recorded is ``self.agent.provider`` / ``self.agent.model``, set when the
    live ``AIAgent`` was constructed on a prior (possibly switched) turn.
    ``_apply_orchestrator_usage_routing`` must read identity from
    ``self.agent`` when present so it can tell "already on Fallback" apart from
    "still on the configured primary", while still using ``self.provider`` /
    ``self.model`` (i.e. ``runtime``) as the BASE target to recover to.
    """

    def test_unknown_reading_while_active_agent_is_fallback_stays_on_fallback(self):
        # Shell config (self.provider/self.model) is still the primary
        # Primary — only self.agent reflects the actual live route (Fallback).
        shell = _make_shell(
            provider=PRIMARY_PROVIDER,
            model=PRIMARY_MODEL,
            agent=SimpleNamespace(**_fallback_runtime()),
        )
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=None, freshness="unknown", age_seconds=None
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider"
            ) as mock_resolve,
        ):
            route = _bound(shell)("hi")

        # Preserve the already-active Fallback runtime. Unknown usage must not
        # re-resolve auth and must never fall back to Primary.
        assert route["model"] == FALLBACK_MODEL
        assert route["runtime"]["provider"] == FALLBACK_PROVIDER
        assert route["runtime"]["api_key"] == "fallback-key"
        mock_resolve.assert_not_called()

    def test_fresh_good_reading_while_active_agent_is_fallback_restores_base_primary(self):
        shell = _make_shell(
            provider=PRIMARY_PROVIDER,
            model=PRIMARY_MODEL,
            agent=SimpleNamespace(provider=FALLBACK_PROVIDER, model=FALLBACK_MODEL),
        )
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=95.0, freshness="fresh", age_seconds=5
        )
        primary_runtime = {
            "provider": PRIMARY_PROVIDER,
            "requested_provider": PRIMARY_PROVIDER,
            "api_key": "primary-key",
            "base_url": "https://primary.example.invalid",
            "api_mode": "primary_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
            "model": PRIMARY_MODEL,
        }
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=primary_runtime,
            ) as mock_resolve,
        ):
            route = _bound(shell)("hi")

        assert route["model"] == PRIMARY_MODEL
        assert route["runtime"]["provider"] == PRIMARY_PROVIDER
        # The decision target (Primary) matches the BASE runtime already in
        # hand (self.provider/self.model, unaffected by the earlier Fallback
        # switch) — no re-resolution is needed to recover.
        mock_resolve.assert_not_called()

    def test_no_active_agent_yet_uses_shell_config_as_current(self):
        # No self.agent constructed yet (startup / first turn): identity
        # must fall back to the shell's own provider/model, exactly as
        # before the agent-derived-identity fix.
        shell = _make_shell(provider=PRIMARY_PROVIDER, model=PRIMARY_MODEL, agent=None)
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=80.0, freshness="fresh", age_seconds=5
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_resolve,
        ):
            route = _bound(shell)("hi")

        assert route["model"] == PRIMARY_MODEL
        assert route["runtime"]["provider"] == PRIMARY_PROVIDER
        mock_resolve.assert_not_called()


class TestCliPreservesFastModeOverrides:
    def test_fast_overrides_preserved_when_switched_to_fallback(self):
        shell = _make_shell(service_tier="priority")
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=1.0, freshness="fresh", age_seconds=1
        )
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=_ROUTING_CONFIG),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=_fallback_runtime(),
            ),
            patch(
                "hermes_cli.models.resolve_fast_mode_overrides",
                return_value={"service_tier": "priority"},
            ),
        ):
            route = _bound(shell)("hi")

        assert route["model"] == FALLBACK_MODEL
        assert route["request_overrides"] == {"service_tier": "priority"}


class TestCliRoutingDisabledIsInert:
    def test_disabled_config_never_switches(self):
        shell = _make_shell()
        usage = ProviderUsage(
            provider=PRIMARY_PROVIDER, remaining_percent=0.0, freshness="fresh", age_seconds=1
        )
        disabled_cfg = {"agent": {"orchestrator_usage_routing": {"enabled": False}}}
        with (
            patch("hermes_cli.config.load_config_readonly", return_value=disabled_cfg),
            patch(
                "agent.delegation_usage_cache.build_usage_view",
                return_value=SimpleNamespace(entries={PRIMARY_PROVIDER: usage}),
            ) as mock_build,
            patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_resolve,
        ):
            route = _bound(shell)("hi")

        assert route["model"] == PRIMARY_MODEL
        assert route["runtime"]["provider"] == PRIMARY_PROVIDER
        mock_resolve.assert_not_called()
        # Being inert doesn't strictly forbid a cache read, but must never
        # trigger a runtime/auth resolution call.
        assert mock_build.call_count in (0, 1)
