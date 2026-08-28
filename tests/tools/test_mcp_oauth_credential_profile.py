import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.mcp_config import _oauth_tokens_present
from mcp.shared.auth import OAuthToken
from tools.mcp_oauth import HermesTokenStorage, resolve_credential_home
from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry


def _install_profile(monkeypatch, profile_home: Path) -> None:
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "welby")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profile_home)


def test_resolve_credential_home_defaults_to_caller() -> None:
    assert resolve_credential_home("tracery", None) is None
    assert resolve_credential_home("tracery", {}) is None
    assert resolve_credential_home("tracery", {"credential_profile": None}) is None


@pytest.mark.parametrize("value", ["", "   ", 0, False, [], {}])
def test_resolve_credential_home_rejects_non_profile_values(value) -> None:
    with pytest.raises(ValueError, match="credential_profile"):
        resolve_credential_home("tracery", {"credential_profile": value})


@pytest.mark.parametrize("value", ["../welby", "welby/..", "/tmp/welby", "a\\b"])
def test_resolve_credential_home_rejects_paths(monkeypatch, value) -> None:
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    with pytest.raises(ValueError, match="invalid credential_profile"):
        resolve_credential_home("tracery", {"credential_profile": value})


def test_token_presence_uses_credential_profile(monkeypatch, tmp_path) -> None:
    caller_home = tmp_path / "caller"
    owner_home = tmp_path / "owner"
    _install_profile(monkeypatch, owner_home)
    monkeypatch.setenv("HERMES_HOME", str(caller_home))

    token_path = owner_home / "mcp-tokens" / "tracery.json"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("{}")

    cfg = {"oauth": {"credential_profile": "welby"}}
    assert _oauth_tokens_present("tracery", cfg) is True
    assert not (caller_home / "mcp-tokens" / "tracery.json").exists()


def test_token_presence_rejects_missing_credential_profile(monkeypatch) -> None:
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)
    cfg = {"oauth": {"credential_profile": "missing"}}

    with pytest.raises(ValueError, match="does not exist"):
        _oauth_tokens_present("tracery", cfg)


def test_storage_round_trip_stays_in_credential_profile(monkeypatch, tmp_path) -> None:
    caller_home = tmp_path / "caller"
    owner_home = tmp_path / "owner"
    _install_profile(monkeypatch, owner_home)
    monkeypatch.setenv("HERMES_HOME", str(caller_home))

    credential_home = resolve_credential_home(
        "tracery",
        {"credential_profile": "welby"},
    )
    storage = HermesTokenStorage("tracery", hermes_home=credential_home)
    token = OAuthToken(
        access_token="owner-access-token",
        token_type="Bearer",
        refresh_token="owner-refresh-token",
        expires_in=3600,
    )

    asyncio.run(storage.set_tokens(token))
    loaded = asyncio.run(storage.get_tokens())

    assert loaded is not None
    assert loaded.access_token == "owner-access-token"
    assert (owner_home / "mcp-tokens" / "tracery.json").exists()
    assert not (caller_home / "mcp-tokens" / "tracery.json").exists()


def test_manager_remove_deletes_owner_only(monkeypatch, tmp_path) -> None:
    caller_home = tmp_path / "caller"
    owner_home = tmp_path / "owner"
    _install_profile(monkeypatch, owner_home)

    caller_token = caller_home / "mcp-tokens" / "tracery.json"
    owner_token = owner_home / "mcp-tokens" / "tracery.json"
    for path in (caller_token, owner_token):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    manager = MCPOAuthManager()
    manager.remove(
        "tracery",
        hermes_home=caller_home,
        oauth_config={"credential_profile": "welby"},
    )

    assert caller_token.exists()
    assert not owner_token.exists()


def test_manager_remove_prefers_explicit_config_over_stale_cache(
    monkeypatch,
    tmp_path,
) -> None:
    caller_home = tmp_path / "caller"
    owner_home = tmp_path / "owner"
    _install_profile(monkeypatch, owner_home)

    caller_token = caller_home / "mcp-tokens" / "tracery.json"
    owner_token = owner_home / "mcp-tokens" / "tracery.json"
    for path in (caller_token, owner_token):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    manager = MCPOAuthManager()
    manager._entries[manager._key("tracery", caller_home)] = _ProviderEntry(
        "https://beproud.tracery.jp/mcp",
        {},
    )
    manager.remove(
        "tracery",
        hermes_home=caller_home,
        oauth_config={"credential_profile": "welby"},
    )

    assert caller_token.exists()
    assert not owner_token.exists()


def test_disk_watch_observes_owner_profile(monkeypatch, tmp_path) -> None:
    caller_home = tmp_path / "caller"
    owner_home = tmp_path / "owner"
    _install_profile(monkeypatch, owner_home)

    token_path = owner_home / "mcp-tokens" / "tracery.json"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("{}")

    provider = SimpleNamespace(_initialized=True)
    entry = _ProviderEntry(
        "https://beproud.tracery.jp/mcp",
        {"credential_profile": "welby"},
        provider=provider,
    )
    manager = MCPOAuthManager()
    manager._entries[manager._key("tracery", caller_home)] = entry

    changed = asyncio.run(
        manager.invalidate_if_disk_changed("tracery", hermes_home=caller_home)
    )

    assert changed is True
    assert provider._initialized is False
    assert entry.last_mtime_ns == token_path.stat().st_mtime_ns
