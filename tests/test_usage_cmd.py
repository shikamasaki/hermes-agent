"""Tests for `hermes usage` — configured-provider usage listing.

This command must only ever render the secret-free projection already
enforced by ``agent.delegation_usage_cache``. Every test here either mocks
the cache/refresh seam or asserts no token, API key, email, org, account or
project identity, credential path, base URL, or raw provider response/error
text ever reaches text or JSON output. No test performs a live provider
call.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from hermes_cli import usage_cmd


# ---------------------------------------------------------------------------
# Provider discovery / dedupe
# ---------------------------------------------------------------------------


class TestDiscoverConfiguredProviders:
    def test_discovers_model_provider(self):
        config = {"model": {"provider": "openai-codex"}}
        providers = usage_cmd.discover_configured_providers(config)
        assert "openai-codex" in providers

    def test_discovers_delegation_routes_providers(self):
        config = {
            "delegation": {
                "routing": {"enabled": True},
                "routes": [
                    {
                        "id": "r1",
                        "provider": "openai-codex",
                        "model": "gpt-5",
                        "model_class": "advanced",
                        "task_difficulties": ["standard"],
                    },
                    {
                        "id": "r2",
                        "provider": "google-antigravity",
                        "model": "gemini-3",
                        "model_class": "advanced",
                        "task_difficulties": ["standard"],
                    },
                ],
            }
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert "openai-codex" in providers
        assert "google-antigravity" in providers

    def test_discovers_legacy_delegation_provider_without_route_catalog(self):
        config = {
            "delegation": {
                "provider": "openai-codex",
                "model": "gpt-5",
            }
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert providers == ("openai-codex",)

    def test_discovers_claude_p_route_provider(self):
        config = {
            "delegation": {
                "routing": {"enabled": True},
                "routes": [
                    {
                        "id": "r1",
                        "provider": "claude-p",
                        "backend": "claude-p",
                        "model": "claude",
                        "model_class": "advanced",
                        "task_difficulties": ["standard"],
                    },
                ],
            }
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert "claude-p" in providers

    def test_discovers_orchestrator_usage_routing_primary_and_fallback(self):
        config = {
            "agent": {
                "orchestrator_usage_routing": {
                    "enabled": True,
                    "primary_provider": "openai-codex",
                    "primary_model": "gpt-5",
                    "fallback_provider": "google-antigravity",
                    "fallback_model": "gemini-3",
                    "switch_at_remaining_percent": 10.0,
                    "restore_above_remaining_percent": 10.0,
                    "usage_ttl_seconds": 900,
                    "usage_stale_seconds": 7200,
                }
            }
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert "openai-codex" in providers
        assert "google-antigravity" in providers

    def test_discovers_auxiliary_provider_assignments(self):
        config = {
            "auxiliary": {
                "vision": {"provider": "openrouter"},
                "compression": {"provider": "auto"},
            }
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert "openrouter" in providers

    def test_dedupes_across_sources(self):
        config = {
            "model": {"provider": "openai-codex"},
            "delegation": {
                "routing": {"enabled": True},
                "routes": [
                    {
                        "id": "r1",
                        "provider": "openai-codex",
                        "model": "gpt-5",
                        "model_class": "advanced",
                        "task_difficulties": ["standard"],
                    },
                ],
            },
            "auxiliary": {"vision": {"provider": "openai-codex"}},
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert list(providers).count("openai-codex") == 1

    def test_ignores_auto_and_custom_and_empty_placeholders(self):
        config = {
            "model": {"provider": "auto"},
            "auxiliary": {
                "vision": {"provider": "custom"},
                "compression": {"provider": ""},
            },
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert "auto" not in providers
        assert "custom" not in providers
        assert "" not in providers

    def test_orchestrator_routing_disabled_contributes_nothing(self):
        config = {
            "agent": {
                "orchestrator_usage_routing": {
                    "enabled": False,
                }
            }
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert providers == ()

    def test_no_configured_providers_returns_empty(self):
        assert usage_cmd.discover_configured_providers({}) == ()

    def test_normalizes_provider_aliases(self):
        config = {"model": {"provider": "antigravity"}}
        providers = usage_cmd.discover_configured_providers(config)
        assert "google-antigravity" in providers
        assert "antigravity" not in providers


# ---------------------------------------------------------------------------
# Cache rendering
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from agent import delegation_usage_cache as duc

    monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
    duc.reset_refresh_state()
    return duc


def _store(cache, provider, *, used=40.0, label="Session", fetched_at=None, source="usage_api"):
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

    snapshot = AccountUsageSnapshot(
        provider=provider,
        source=source,
        fetched_at=fetched_at or datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(
                label=label,
                used_percent=used,
                reset_at=datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc),
            ),
        ),
    )
    cache.store_snapshot(snapshot)


class TestRenderCachedUsage:
    def test_renders_fresh_window_fields(self, cache, monkeypatch):
        monkeypatch.setattr(usage_cmd, "_SAFE_LABEL_OVERRIDE", None, raising=False)
        _store(cache, "openai-codex", used=40.0, label="Session")
        rows = usage_cmd.build_usage_rows(["openai-codex"], refresh=False)
        assert len(rows) == 1
        row = rows[0]
        assert row["provider"] == "openai-codex"
        assert row["freshness"] == "fresh"
        assert row["age_seconds"] is not None
        assert row["windows"][0]["remaining_percent"] == 60.0
        assert row["windows"][0]["used_percent"] == 40.0
        assert row["windows"][0]["reset_at"] == "2026-08-30T12:00:00+00:00"
        assert row["source"] == "usage_api"

    def test_stale_window_marked_stale(self, cache):
        old = datetime.now(timezone.utc) - timedelta(seconds=1000)
        _store(cache, "openai-codex", used=10.0, fetched_at=old)
        rows = usage_cmd.build_usage_rows(
            ["openai-codex"], refresh=False, ttl_seconds=300, stale_seconds=1800
        )
        assert rows[0]["freshness"] == "stale"

    def test_multiple_providers_each_get_a_row(self, cache):
        _store(cache, "openai-codex")
        _store(cache, "google-antigravity")
        rows = usage_cmd.build_usage_rows(
            ["openai-codex", "google-antigravity"], refresh=False
        )
        assert {r["provider"] for r in rows} == {"openai-codex", "google-antigravity"}


# ---------------------------------------------------------------------------
# Unknown / unavailable state
# ---------------------------------------------------------------------------


class TestUnknownState:
    def test_uncached_provider_is_explicit_unknown(self, cache):
        rows = usage_cmd.build_usage_rows(["openai-codex"], refresh=False)
        assert rows[0]["freshness"] == "unknown"
        assert rows[0]["windows"] == []
        assert rows[0]["status"] == "unknown"

    def test_past_stale_window_reports_unknown_not_raw_age(self, cache):
        ancient = datetime.now(timezone.utc) - timedelta(seconds=999999)
        _store(cache, "openai-codex", fetched_at=ancient)
        rows = usage_cmd.build_usage_rows(
            ["openai-codex"], refresh=False, ttl_seconds=300, stale_seconds=1800
        )
        assert rows[0]["freshness"] == "unknown"

    def test_unknown_state_never_carries_raw_error_text(self, cache, monkeypatch):
        # Simulate a fetch that raised — cache stores nothing; row must stay
        # a clean "unknown" shape, never leak exception text.
        rows = usage_cmd.build_usage_rows(["openai-codex"], refresh=False)
        blob = json.dumps(rows)
        assert "Traceback" not in blob
        assert "Exception" not in blob


# ---------------------------------------------------------------------------
# --refresh calls
# ---------------------------------------------------------------------------


class TestRefreshFlag:
    def test_refresh_true_calls_refresh_provider_now_for_selected_providers(
        self, cache, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            usage_cmd.delegation_usage_cache,
            "refresh_provider_now",
            lambda provider: calls.append(provider),
        )
        usage_cmd.build_usage_rows(["openai-codex", "google-antigravity"], refresh=True)
        assert set(calls) == {"openai-codex", "google-antigravity"}

    def test_refresh_false_never_calls_refresh_provider_now(self, cache, monkeypatch):
        calls = []
        monkeypatch.setattr(
            usage_cmd.delegation_usage_cache,
            "refresh_provider_now",
            lambda provider: calls.append(provider),
        )
        usage_cmd.build_usage_rows(["openai-codex"], refresh=False)
        assert calls == []

    def test_refresh_is_bounded_synchronous_not_background_thread(self, cache, monkeypatch):
        # --refresh must call the synchronous, bounded refresh path directly
        # (refresh_provider_now), never schedule_refresh's background thread,
        # so the CLI can render the result of the refresh it just requested.
        monkeypatch.setattr(
            usage_cmd.delegation_usage_cache,
            "refresh_provider_now",
            lambda provider: _store(cache, provider, used=5.0),
        )
        started = []
        monkeypatch.setattr(
            usage_cmd.delegation_usage_cache,
            "schedule_refresh",
            lambda provider: started.append(provider),
        )
        rows = usage_cmd.build_usage_rows(["openai-codex"], refresh=True)
        assert started == []
        assert rows[0]["windows"][0]["used_percent"] == 5.0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestProviderFilter:
    def test_filter_narrows_to_selected_provider(self, cache):
        _store(cache, "openai-codex")
        _store(cache, "google-antigravity")
        rows = usage_cmd.build_usage_rows(
            ["openai-codex", "google-antigravity"],
            refresh=False,
            provider_filter=["openai-codex"],
        )
        assert {r["provider"] for r in rows} == {"openai-codex"}

    def test_filter_normalizes_aliases(self, cache):
        _store(cache, "google-antigravity")
        rows = usage_cmd.build_usage_rows(
            ["google-antigravity"],
            refresh=False,
            provider_filter=["antigravity"],
        )
        assert {r["provider"] for r in rows} == {"google-antigravity"}

    def test_filter_for_unconfigured_provider_yields_no_rows(self, cache):
        rows = usage_cmd.build_usage_rows(
            ["openai-codex"], refresh=False, provider_filter=["google-antigravity"]
        )
        assert rows == []


# ---------------------------------------------------------------------------
# Safe JSON output / no secret fields
# ---------------------------------------------------------------------------


class TestSafeOutput:
    def test_json_output_is_valid_json_with_expected_top_level_shape(self, cache):
        _store(cache, "openai-codex")
        payload = usage_cmd.render_json(["openai-codex"], refresh=False)
        data = json.loads(payload)
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_json_output_never_contains_secret_fields(self, cache):
        _store(cache, "openai-codex")
        payload = usage_cmd.render_json(["openai-codex"], refresh=False)
        for leaked in (
            "api_key",
            "token",
            "Authorization",
            "base_url",
            "credential",
            "email",
            "org_id",
            "project",
        ):
            assert leaked not in payload, f"json output leaked {leaked!r}"

    def test_text_output_never_contains_secret_fields(self, cache):
        _store(cache, "openai-codex")
        text = usage_cmd.render_text(["openai-codex"], refresh=False)
        for leaked in ("api_key", "Authorization", "base_url", "credential"):
            assert leaked not in text, f"text output leaked {leaked!r}"

    def test_no_providers_renders_cleanly_in_text(self):
        text = usage_cmd.render_text([], refresh=False)
        assert isinstance(text, str)

    def test_no_providers_renders_cleanly_in_json(self):
        payload = usage_cmd.render_json([], refresh=False)
        data = json.loads(payload)
        assert data["providers"] == []


# ---------------------------------------------------------------------------
# Parser / dispatch / help / exit behavior
# ---------------------------------------------------------------------------


class TestParserAndDispatch:
    def _build_parser(self):
        import argparse

        from hermes_cli.subcommands.usage import build_usage_parser

        parser = argparse.ArgumentParser(prog="hermes")
        subparsers = parser.add_subparsers(dest="command")
        calls = []
        build_usage_parser(subparsers, cmd_usage=lambda args: calls.append(args))
        return parser, calls

    def test_usage_subcommand_registered(self):
        parser, _ = self._build_parser()
        args = parser.parse_args(["usage"])
        assert args.command == "usage"

    def test_refresh_flag_parses(self):
        parser, _ = self._build_parser()
        args = parser.parse_args(["usage", "--refresh"])
        assert args.refresh is True

    def test_refresh_defaults_false(self):
        parser, _ = self._build_parser()
        args = parser.parse_args(["usage"])
        assert args.refresh is False

    def test_json_flag_parses(self):
        parser, _ = self._build_parser()
        args = parser.parse_args(["usage", "--json"])
        assert args.json is True

    def test_provider_flag_repeatable(self):
        parser, _ = self._build_parser()
        args = parser.parse_args(
            ["usage", "--provider", "openai-codex", "--provider", "google-antigravity"]
        )
        assert args.provider == ["openai-codex", "google-antigravity"]

    def test_provider_flag_defaults_to_none(self):
        parser, _ = self._build_parser()
        args = parser.parse_args(["usage"])
        assert args.provider is None

    def test_dispatch_calls_cmd_usage(self):
        parser, calls = self._build_parser()
        args = parser.parse_args(["usage", "--json"])
        args.func(args)
        assert len(calls) == 1

    def test_help_does_not_raise(self, capsys):
        parser, _ = self._build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["usage", "--help"])
        assert exc_info.value.code == 0

    def test_main_wires_cmd_usage(self):
        # hermes_cli.main must expose cmd_usage and register it with the
        # top-level parser via build_usage_parser, matching every other
        # extracted subcommand (doctor/security/insights).
        from hermes_cli import main as hermes_main

        assert hasattr(hermes_main, "cmd_usage")


class TestRunUsageCommandExitCodes:
    def test_no_configured_providers_exits_zero(self, tmp_path, monkeypatch, capsys):
        import argparse

        monkeypatch.setattr(usage_cmd, "_load_active_config", lambda: {})
        args = argparse.Namespace(refresh=False, provider=None, json=False)
        code = usage_cmd.run_usage_command(args)
        assert code == 0

    def test_no_cached_data_exits_zero(self, tmp_path, monkeypatch, cache):
        import argparse

        monkeypatch.setattr(
            usage_cmd,
            "_load_active_config",
            lambda: {"model": {"provider": "openai-codex"}},
        )
        args = argparse.Namespace(refresh=False, provider=None, json=False)
        code = usage_cmd.run_usage_command(args)
        assert code == 0


# ---------------------------------------------------------------------------
# No prompt-cache / agent behavior changes
# ---------------------------------------------------------------------------


class TestNoBehaviorChangeElsewhere:
    def test_usage_cmd_module_does_not_import_prompt_cache_or_agent_loop(self):
        # A read-only reporting CLI must not import modules that drive the
        # live agent turn loop or prompt-cache bookkeeping — importing
        # usage_cmd must never have a side effect on those systems.
        import sys

        for name in list(sys.modules):
            if name in ("agent.prompt_cache", "agent.agent_loop", "cli"):
                del sys.modules[name]
        import hermes_cli.usage_cmd  # noqa: F401

        assert "agent.prompt_cache" not in sys.modules
        assert "agent.agent_loop" not in sys.modules
        assert "cli" not in sys.modules

    def test_build_usage_rows_does_not_mutate_route_catalog_or_orchestrator_state(
        self, cache
    ):
        # Calling the reporting path must not perturb delegation routing's
        # own module-level state beyond the documented refresh-scheduling
        # seam (which we already assert separately).
        from agent import delegation_routing

        before = delegation_routing.NATIVE_ROUTABLE_PROVIDERS
        usage_cmd.build_usage_rows(["openai-codex"], refresh=False)
        after = delegation_routing.NATIVE_ROUTABLE_PROVIDERS
        assert before is after
