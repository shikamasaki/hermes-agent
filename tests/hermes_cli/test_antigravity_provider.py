from __future__ import annotations

import json
import time

import pytest


def test_provider_plugin_registers_google_antigravity_profile():
    from providers import get_provider_profile
    from hermes_cli.auth import PROVIDER_REGISTRY, OAUTH_PROVIDER_FLOWS

    profile = get_provider_profile("google-antigravity")

    assert profile is not None
    assert profile.name == "google-antigravity"
    assert "antigravity" in profile.aliases
    assert profile.api_mode == "chat_completions"
    assert profile.auth_type == "oauth_external"
    assert profile.base_url == "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"
    assert PROVIDER_REGISTRY["google-antigravity"].auth_type == "oauth_external"
    assert "google-antigravity" in OAUTH_PROVIDER_FLOWS


def test_runtime_uses_valid_google_antigravity_token_and_project(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: {"model": {"provider": "google-antigravity"}})

    from hermes_cli.antigravity_auth import ANTIGRAVITY_BASE_URL
    from hermes_cli.runtime_provider import resolve_runtime_provider

    (tmp_path / "auth.json").write_text(json.dumps({
        "providers": {
            "google-antigravity": {
                "access_token": "access-token-secret",
                "refresh_token": "refresh-token-secret",
                "expires_at": time.time() + 3600,
                "auth_type": "oauth_pkce",
                "base_url": ANTIGRAVITY_BASE_URL,
                "project_id": "project-123",
            }
        },
        "active_provider": "google-antigravity",
    }))

    runtime = resolve_runtime_provider(requested="google-antigravity")

    assert runtime["provider"] == "google-antigravity"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["base_url"] == ANTIGRAVITY_BASE_URL
    assert runtime["api_key"] == "access-token-secret"
    assert runtime["project_id"] == "project-123"
    assert runtime["source"] == "hermes-auth-store"


def test_runtime_rejects_expired_google_antigravity_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("hermes_cli.runtime_provider.load_config", lambda: {"model": {"provider": "google-antigravity"}})

    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    (tmp_path / "auth.json").write_text(json.dumps({
        "providers": {
            "google-antigravity": {
                "access_token": "expired-access-token-secret",
                "refresh_token": "refresh-token-secret",
                "expires_at": time.time() - 1,
                "auth_type": "oauth_pkce",
                "base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
                "project_id": "project-123",
            }
        },
        "active_provider": "google-antigravity",
        "credential_pool": {
            "google-antigravity": [
                {
                    "id": "expired-pool",
                    "label": "expired",
                    "auth_type": "oauth",
                    "source": "manual:antigravity_pkce",
                    "access_token": "expired-access-token-secret",
                    "refresh_token": "refresh-token-secret",
                    "expires_at": time.time() - 1,
                    "base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal",
                }
            ]
        },
    }))

    with pytest.raises(AuthError) as excinfo:
        resolve_runtime_provider(requested="google-antigravity")

    message = str(excinfo.value)
    assert excinfo.value.provider == "google-antigravity"
    assert excinfo.value.relogin_required is True
    assert "expired-access-token-secret" not in message
    assert "refresh-token-secret" not in message
