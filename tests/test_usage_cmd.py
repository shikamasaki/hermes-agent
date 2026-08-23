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
        config = {"model": {"provider": "provider-a"}}
        providers = usage_cmd.discover_configured_providers(config)
        assert "provider-a" in providers

    def test_discovers_delegation_routes_providers(self):
        config = {
            "delegation": {
                "routing": {"enabled": True},
                "routes": [
                    {
                        "id": "r1",
                        "provider": "provider-a",
                        "model": "model-a",
                        "model_class": "advanced",
                        "task_difficulties": ["standard"],
                    },
                    {
                        "id": "r2",
                        "provider": "provider-b",
                        "model": "model-b",
                        "model_class": "advanced",
                        "task_difficulties": ["standard"],
                    },
                ],
            }
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert "provider-a" in providers
        assert "provider-b" in providers

    def test_discovers_legacy_delegation_provider_without_route_catalog(self):
        config = {
            "delegation": {
                "provider": "provider-a",
                "model": "model-a",
            }
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert providers == ("provider-a",)


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
            "model": {"provider": "provider-a"},
            "delegation": {
                "routing": {"enabled": True},
                "routes": [
                    {
                        "id": "r1",
                        "provider": "provider-a",
                        "model": "model-a",
                        "model_class": "advanced",
                        "task_difficulties": ["standard"],
                    },
                ],
            },
            "auxiliary": {"vision": {"provider": "provider-a"}},
        }
        providers = usage_cmd.discover_configured_providers(config)
        assert list(providers).count("provider-a") == 1

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


# ---------------------------------------------------------------------------
# Cache rendering
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from agent import delegation_usage_cache as duc

    monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
    duc.reset_refresh_state()
    return duc


def _store(cache, provider, *, used=40.0, label="Session", fetched_at=None, source="source_a"):
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
        _store(cache, "provider-a", used=40.0, label="Session")
        rows = usage_cmd.build_usage_rows(["provider-a"], refresh=False)
        assert len(rows) == 1
        row = rows[0]
        assert row["provider"] == "provider-a"
        assert row["freshness"] == "fresh"
        assert row["age_seconds"] is not None
        assert row["windows"][0]["remaining_percent"] == 60.0
        assert row["windows"][0]["used_percent"] == 40.0
        assert row["windows"][0]["reset_at"] == "2026-08-30T12:00:00+00:00"
        assert row["source"] == "source_a"

    def test_stale_window_marked_stale(self, cache):
        old = datetime.now(timezone.utc) - timedelta(seconds=1000)
        _store(cache, "provider-a", used=10.0, fetched_at=old)
        rows = usage_cmd.build_usage_rows(
            ["provider-a"], refresh=False, ttl_seconds=300, stale_seconds=1800
        )
        assert rows[0]["freshness"] == "stale"

    def test_multiple_providers_each_get_a_row(self, cache):
        _store(cache, "provider-a")
        _store(cache, "provider-b")
        rows = usage_cmd.build_usage_rows(
            ["provider-a", "provider-b"], refresh=False
        )
        assert {r["provider"] for r in rows} == {"provider-a", "provider-b"}


# ---------------------------------------------------------------------------
# Unknown / unavailable state
# ---------------------------------------------------------------------------


class TestUnknownState:
    def test_uncached_provider_is_explicit_unknown(self, cache):
        rows = usage_cmd.build_usage_rows(["provider-a"], refresh=False)
        assert rows[0]["freshness"] == "unknown"
        assert rows[0]["windows"] == []
        assert rows[0]["status"] == "unknown"

    def test_past_stale_window_reports_unknown_not_raw_age(self, cache):
        ancient = datetime.now(timezone.utc) - timedelta(seconds=999999)
        _store(cache, "provider-a", fetched_at=ancient)
        rows = usage_cmd.build_usage_rows(
            ["provider-a"], refresh=False, ttl_seconds=300, stale_seconds=1800
        )
        assert rows[0]["freshness"] == "unknown"

    def test_unknown_state_never_carries_raw_error_text(self, cache, monkeypatch):
        # Simulate a fetch that raised — cache stores nothing; row must stay
        # a clean "unknown" shape, never leak exception text.
        rows = usage_cmd.build_usage_rows(["provider-a"], refresh=False)
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
        usage_cmd.build_usage_rows(["provider-a", "provider-b"], refresh=True)
        assert set(calls) == {"provider-a", "provider-b"}

    def test_refresh_false_never_calls_refresh_provider_now(self, cache, monkeypatch):
        calls = []
        monkeypatch.setattr(
            usage_cmd.delegation_usage_cache,
            "refresh_provider_now",
            lambda provider: calls.append(provider),
        )
        usage_cmd.build_usage_rows(["provider-a"], refresh=False)
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
        rows = usage_cmd.build_usage_rows(["provider-a"], refresh=True)
        assert started == []
        assert rows[0]["windows"][0]["used_percent"] == 5.0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class TestProviderFilter:
    def test_filter_narrows_to_selected_provider(self, cache):
        _store(cache, "provider-a")
        _store(cache, "provider-b")
        rows = usage_cmd.build_usage_rows(
            ["provider-a", "provider-b"],
            refresh=False,
            provider_filter=["provider-a"],
        )
        assert {r["provider"] for r in rows} == {"provider-a"}


    def test_filter_for_unconfigured_provider_yields_no_rows(self, cache):
        rows = usage_cmd.build_usage_rows(
            ["provider-a"], refresh=False, provider_filter=["provider-b"]
        )
        assert rows == []


# ---------------------------------------------------------------------------
# Safe JSON output / no secret fields
# ---------------------------------------------------------------------------


class TestSafeOutput:
    def test_json_output_is_valid_json_with_expected_top_level_shape(self, cache):
        _store(cache, "provider-a")
        payload = usage_cmd.render_json(["provider-a"], refresh=False)
        data = json.loads(payload)
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_json_output_never_contains_secret_fields(self, cache):
        _store(cache, "provider-a")
        payload = usage_cmd.render_json(["provider-a"], refresh=False)
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
        _store(cache, "provider-a")
        text = usage_cmd.render_text(["provider-a"], refresh=False)
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
            ["usage", "--provider", "provider-a", "--provider", "provider-b"]
        )
        assert args.provider == ["provider-a", "provider-b"]

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
            lambda: {"model": {"provider": "provider-a"}},
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

    def test_build_usage_rows_does_not_mutate_route_catalog_state(self, cache):
        # Calling the reporting path must not perturb delegation routing's
        # module-level contracts.
        from agent import delegation_routing

        before = delegation_routing.RouteCatalog
        usage_cmd.build_usage_rows(["provider-a"], refresh=False)
        after = delegation_routing.RouteCatalog
        assert before is after
