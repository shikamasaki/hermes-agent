"""RED tests for `/usage providers [refresh]` on the interactive CLI/TUI.

`cli.py::HermesCLI._handle_usage_command` dispatches `/usage`, `/usage reset
[--force]` (unchanged), and the new `/usage providers [refresh]` subcommand,
which renders text produced entirely by the shared, secret-safe
`hermes_cli.usage_cmd` service — no provider discovery/rendering logic is
duplicated in cli.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import cli


def _bare_cli() -> cli.HermesCLI:
    """A HermesCLI instance built without running __init__."""
    return object.__new__(cli.HermesCLI)


class TestHandleUsageCommandDispatch:
    def test_bare_usage_calls_show_usage(self, monkeypatch):
        instance = _bare_cli()
        calls = []
        monkeypatch.setattr(instance, "_show_usage", lambda: calls.append("show"))

        instance._handle_usage_command("/usage")

        assert calls == ["show"]

    def test_reset_dispatch_unchanged(self, monkeypatch):
        instance = _bare_cli()
        calls = []
        monkeypatch.setattr(
            instance, "_usage_reset", lambda force=False: calls.append(force)
        )

        instance._handle_usage_command("/usage reset")
        instance._handle_usage_command("/usage reset --force")

        assert calls == [False, True]

    def test_providers_dispatches_to_usage_providers_with_refresh_false(self, monkeypatch):
        instance = _bare_cli()
        calls = []
        monkeypatch.setattr(
            instance, "_usage_providers", lambda refresh=False: calls.append(refresh)
        )

        instance._handle_usage_command("/usage providers")

        assert calls == [False]

    def test_providers_refresh_dispatches_with_refresh_true(self, monkeypatch):
        instance = _bare_cli()
        calls = []
        monkeypatch.setattr(
            instance, "_usage_providers", lambda refresh=False: calls.append(refresh)
        )

        instance._handle_usage_command("/usage providers refresh")

        assert calls == [True]

    def test_refresh_is_explicit_only(self, monkeypatch):
        """A second token that isn't literally 'refresh' must not enable it."""
        instance = _bare_cli()
        calls = []
        monkeypatch.setattr(
            instance, "_usage_providers", lambda refresh=False: calls.append(refresh)
        )

        instance._handle_usage_command("/usage providers now")

        assert calls == [False]

    def test_unknown_subcommand_prints_help_mentioning_providers(self, monkeypatch, capsys):
        instance = _bare_cli()
        instance._handle_usage_command("/usage bogus")

        out = capsys.readouterr().out
        assert "Unknown /usage subcommand" in out
        assert "providers" in out


class TestUsageProvidersRendering:
    """`_usage_providers` invokes the shared hermes_cli.usage_cmd service."""

    def test_calls_shared_service_with_refresh_false(self, monkeypatch, capsys):
        instance = _bare_cli()
        calls = {}

        from hermes_cli import usage_cmd

        monkeypatch.setattr(
            usage_cmd, "_load_active_config", lambda: {"model": {"provider": "openai-codex"}}
        )
        monkeypatch.setattr(
            usage_cmd, "discover_configured_providers", lambda config: ("openai-codex",)
        )

        def fake_render_text(providers, *, refresh, **kwargs):
            calls["providers"] = tuple(providers)
            calls["refresh"] = refresh
            return "openai-codex: unknown"

        monkeypatch.setattr(usage_cmd, "render_text", fake_render_text)

        instance._usage_providers(refresh=False)

        assert calls["providers"] == ("openai-codex",)
        assert calls["refresh"] is False
        assert "openai-codex: unknown" in capsys.readouterr().out

    def test_calls_shared_service_with_refresh_true(self, monkeypatch, capsys):
        instance = _bare_cli()
        calls = {}

        from hermes_cli import usage_cmd

        monkeypatch.setattr(
            usage_cmd, "_load_active_config", lambda: {"model": {"provider": "openai-codex"}}
        )
        monkeypatch.setattr(
            usage_cmd, "discover_configured_providers", lambda config: ("openai-codex",)
        )

        def fake_render_text(providers, *, refresh, **kwargs):
            calls["refresh"] = refresh
            return "refreshed output"

        monkeypatch.setattr(usage_cmd, "render_text", fake_render_text)

        instance._usage_providers(refresh=True)

        assert calls["refresh"] is True
        assert "refreshed output" in capsys.readouterr().out

    def test_no_configured_providers_renders_cleanly(self, monkeypatch, capsys):
        instance = _bare_cli()
        from hermes_cli import usage_cmd

        monkeypatch.setattr(usage_cmd, "_load_active_config", lambda: {})

        instance._usage_providers(refresh=False)

        out = capsys.readouterr().out
        assert "No configured providers with usage data." in out

    def test_output_never_leaks_secret_fields(self, monkeypatch, capsys):
        """End-to-end through the real hermes_cli.usage_cmd.render_text —
        asserts the safe-output contract holds through the CLI surface too.
        """
        instance = _bare_cli()
        from hermes_cli import usage_cmd

        monkeypatch.setattr(
            usage_cmd, "_load_active_config", lambda: {"model": {"provider": "openai-codex"}}
        )

        instance._usage_providers(refresh=False)

        out = capsys.readouterr().out
        for leaked in ("api_key", "Authorization", "base_url", "credential", "token"):
            assert leaked not in out, f"providers output leaked {leaked!r}"
