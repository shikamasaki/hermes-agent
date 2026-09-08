from __future__ import annotations

import json
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest


SECRET_VALUES = {
    "access-token-secret",
    "refresh-token-secret",
    "auth-code-secret",
    "client-secret-secret",
    "id-token-secret",
}


def test_pkce_no_browser_prints_safe_authorize_url_and_persists_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.antigravity_auth import ANTIGRAVITY_BASE_URL, run_pkce_login

    seen: dict[str, str] = {}

    def wait_for_callback(redirect_uri, *, timeout_seconds, on_ready):
        on_ready()
        printed = capsys.readouterr().out
        assert "Open this Google Antigravity sign-in URL" in printed
        query = parse_qs(urlparse(printed.splitlines()[-1]).query)
        seen["state"] = query["state"][0]
        seen["challenge"] = query["code_challenge"][0]
        assert query["code_challenge_method"] == ["S256"]
        assert query["response_type"] == ["code"]
        assert query["redirect_uri"] == [redirect_uri]
        return {"state": seen["state"], "code": "auth-code-secret"}

    def exchange_code(**kwargs):
        assert kwargs["code"] == "auth-code-secret"
        assert kwargs["client_id"] == "client-id"
        assert kwargs["client_secret"] == "client-secret-secret"
        assert kwargs["redirect_uri"] == "http://127.0.0.1:8765/callback"
        assert re.fullmatch(r"[A-Za-z0-9_-]{43,128}", kwargs["code_verifier"])
        assert kwargs["code_verifier"] not in capsys.readouterr().out
        return {
            "access_token": "access-token-secret",
            "refresh_token": "refresh-token-secret",
            "id_token": "id-token-secret",
            "expires_in": 3600,
            "project_id": "project-123",
        }

    saved = run_pkce_login(
        open_browser=False,
        wait_for_callback=wait_for_callback,
        exchange_code=exchange_code,
        discover_project=lambda **_: pytest.fail("project discovery must not run in slice 1"),
        _client_identity_override={"client_id": "client-id", "client_secret": "client-secret-secret"},
    )

    output = capsys.readouterr().out
    for secret in SECRET_VALUES | {saved["access_token"], saved["refresh_token"], saved["id_token"]}:
        assert secret not in output
    assert seen["challenge"]
    auth_store = json.loads((tmp_path / "auth.json").read_text())
    state = auth_store["providers"]["google-antigravity"]
    assert auth_store["active_provider"] == "google-antigravity"
    assert state["access_token"] == "access-token-secret"
    assert state["refresh_token"] == "refresh-token-secret"
    assert state["project_id"] == "project-123"
    assert state["base_url"] == ANTIGRAVITY_BASE_URL


def _login_with_callback(*, tmp_path, monkeypatch, capsys, exchange_payload, discover_project):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.antigravity_auth import run_pkce_login

    seen: dict[str, str] = {}

    def wait_for_callback(redirect_uri, *, timeout_seconds, on_ready):
        on_ready()
        printed = capsys.readouterr().out
        query = parse_qs(urlparse(printed.splitlines()[-1]).query)
        seen["state"] = query["state"][0]
        return {"state": seen["state"], "code": "auth-code-secret"}

    def exchange_code(**kwargs):
        assert kwargs["code"] == "auth-code-secret"
        return dict(exchange_payload)

    return run_pkce_login(
        open_browser=False,
        wait_for_callback=wait_for_callback,
        exchange_code=exchange_code,
        discover_project=discover_project,
        _client_identity_override={"client_id": "client-id", "client_secret": "client-secret-secret"},
    )


def test_pkce_discovers_project_after_token_exchange_and_persists_it(tmp_path, monkeypatch, capsys):
    from hermes_cli.antigravity_auth import ANTIGRAVITY_BASE_URL

    calls: list[dict[str, str]] = []

    def discover_project(**kwargs):
        calls.append(kwargs)
        assert kwargs["access_token"] == "access-token-secret"
        assert kwargs["base_url"] == ANTIGRAVITY_BASE_URL
        return "discovered-project-123"

    saved = _login_with_callback(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        exchange_payload={
            "access_token": "access-token-secret",
            "refresh_token": "refresh-token-secret",
            "id_token": "id-token-secret",
            "expires_in": 3600,
        },
        discover_project=discover_project,
    )

    output = capsys.readouterr().out + capsys.readouterr().err
    assert calls
    assert saved["project_id"] == "discovered-project-123"
    state = json.loads((tmp_path / "auth.json").read_text())["providers"]["google-antigravity"]
    assert state["project_id"] == "discovered-project-123"
    for secret in SECRET_VALUES | {"access-token-secret", "refresh-token-secret", "id-token-secret"}:
        assert secret not in output


def test_pkce_project_discovery_failure_keeps_login_success_without_project(tmp_path, monkeypatch, capsys):
    def discover_project(**_kwargs):
        raise RuntimeError("discovery failed with access-token-secret")

    saved = _login_with_callback(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        exchange_payload={
            "access_token": "access-token-secret",
            "refresh_token": "refresh-token-secret",
            "id_token": "id-token-secret",
            "expires_in": 3600,
        },
        discover_project=discover_project,
    )

    output = capsys.readouterr().out + capsys.readouterr().err
    state = json.loads((tmp_path / "auth.json").read_text())["providers"]["google-antigravity"]
    assert "project_id" not in saved
    assert "project_id" not in state
    assert state["access_token"] == "access-token-secret"
    assert "access-token-secret" not in output


def test_pkce_rejects_callback_state_mismatch_without_leaking_secrets(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.antigravity_auth import AntigravityAuthError, run_pkce_login

    def wait_for_callback(_redirect_uri, *, timeout_seconds, on_ready):
        on_ready()
        capsys.readouterr()
        return {"state": "attacker-state", "code": "auth-code-secret"}

    def exchange_code(**_kwargs):
        pytest.fail("state mismatch must stop before token exchange")

    with pytest.raises(AntigravityAuthError) as excinfo:
        run_pkce_login(
            open_browser=False,
            wait_for_callback=wait_for_callback,
            exchange_code=exchange_code,
            _client_identity_override={"client_id": "client-id", "client_secret": "client-secret-secret"},
        )

    combined = capsys.readouterr().out + capsys.readouterr().err + str(excinfo.value)
    for secret in SECRET_VALUES:
        assert secret not in combined
    assert not (tmp_path / "auth.json").exists()


def test_auth_add_google_antigravity_uses_pkce_login_and_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import hermes_cli.auth_commands as auth_commands
    import hermes_cli.antigravity_auth as antigravity_auth
    from hermes_cli.auth import get_auth_status

    def fake_login(**kwargs):
        assert kwargs["open_browser"] is False
        antigravity_auth.save_state(
            {
                "access_token": "access-token-secret",
                "refresh_token": "refresh-token-secret",
                "expires_at": 4102444800.0,
                "auth_type": "oauth_pkce",
                "redirect_uri": "http://127.0.0.1:8765/callback",
                "base_url": antigravity_auth.ANTIGRAVITY_BASE_URL,
                "project_id": "project-123",
            },
            set_active=True,
        )
        return {"access_token": "access-token-secret", "expires_at": 4102444800.0, "project_id": "project-123"}

    monkeypatch.setattr(antigravity_auth, "run_pkce_login", fake_login)

    auth_commands.auth_add_command(
        SimpleNamespace(provider="google-antigravity", auth_type="oauth", no_browser=True, timeout=1, label=None)
    )

    output = capsys.readouterr().out
    assert "Added google-antigravity OAuth credential" in output
    assert "access-token-secret" not in output
    assert get_auth_status("google-antigravity")["logged_in"] is True
