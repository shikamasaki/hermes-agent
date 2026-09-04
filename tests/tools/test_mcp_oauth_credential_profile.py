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


def test_resolve_credential_home_canonicalizes_symlinked_profile(monkeypatch, tmp_path) -> None:
    real_owner = tmp_path / "real-owner"
    real_owner.mkdir()
    linked_owner = tmp_path / "linked-owner"
    linked_owner.symlink_to(real_owner, target_is_directory=True)
    _install_profile(monkeypatch, linked_owner)

    resolved = resolve_credential_home("tracery", {"credential_profile": "welby"})

    assert resolved == real_owner.resolve(strict=False)


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


def test_storage_rollback_replaces_partial_new_state(tmp_path) -> None:
    storage = HermesTokenStorage("tracery", hermes_home=tmp_path)
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    storage._tokens_path().write_bytes(b"old-token")
    storage._client_info_path().write_bytes(b"old-client")
    storage._meta_path().write_bytes(b"old-meta")
    backup = storage.snapshot()

    storage.remove()
    storage._client_info_path().write_bytes(b"partial-new-client")
    storage.restore(backup)

    assert storage._tokens_path().read_bytes() == b"old-token"
    assert storage._client_info_path().read_bytes() == b"old-client"
    assert storage._meta_path().read_bytes() == b"old-meta"


def test_storage_rollback_preserves_later_token_writer_with_absent_guard(tmp_path) -> None:
    storage = HermesTokenStorage("tracery", hermes_home=tmp_path)
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    storage._tokens_path().write_bytes(b"old-token")
    storage._client_info_path().write_bytes(b"old-client")
    storage._meta_path().write_bytes(b"old-meta")
    backup = storage.snapshot()

    storage.remove()
    storage._tokens_path().write_bytes(b'{"access_token":"new-concurrent-token","token_type":"Bearer"}')
    storage.restore(backup, only_if_absent=True)

    assert storage._tokens_path().read_bytes() == b'{"access_token":"new-concurrent-token","token_type":"Bearer"}'
    assert storage._client_info_path().read_bytes() == b"old-client"
    assert storage._meta_path().read_bytes() == b"old-meta"


def test_storage_rollback_replaces_partial_new_state_with_absent_guard(tmp_path) -> None:
    storage = HermesTokenStorage("tracery", hermes_home=tmp_path)
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    storage._tokens_path().write_bytes(b"old-token")
    storage._client_info_path().write_bytes(b"old-client")
    storage._meta_path().write_bytes(b"old-meta")
    backup = storage.snapshot()

    storage.remove()
    storage._client_info_path().write_bytes(b"partial-new-client")
    failed_flow = storage.snapshot()
    storage.restore(backup, only_if_absent=True, expected_current=failed_flow)

    assert storage._tokens_path().read_bytes() == b"old-token"
    assert storage._client_info_path().read_bytes() == b"old-client"
    assert storage._meta_path().read_bytes() == b"old-meta"


def test_storage_content_rollback_restores_own_partial_client_before_token(tmp_path) -> None:
    storage = HermesTokenStorage("tracery", hermes_home=tmp_path)
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    storage._tokens_path().write_bytes(b"old-token")
    storage._client_info_path().write_bytes(b"old-client")
    storage._meta_path().write_bytes(b"old-meta")
    backup = storage.snapshot()

    storage.remove()
    storage._client_info_path().write_bytes(b"failed-client")
    failed_flow = storage.snapshot()
    storage.restore(backup, only_if_absent=True, expected_current=failed_flow)

    assert storage._tokens_path().read_bytes() == b"old-token"
    assert storage._client_info_path().read_bytes() == b"old-client"
    assert storage._meta_path().read_bytes() == b"old-meta"


def test_storage_content_rollback_preserves_later_client_without_token(tmp_path) -> None:
    storage = HermesTokenStorage("tracery", hermes_home=tmp_path)
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    storage._tokens_path().write_bytes(b"old-token")
    storage._client_info_path().write_bytes(b"old-client")
    storage._meta_path().write_bytes(b"old-meta")
    backup = storage.snapshot()

    storage.remove()
    storage._client_info_path().write_bytes(b"failed-client")
    failed_flow = storage.snapshot()
    storage._client_info_path().write_bytes(b"later-client")
    storage.restore(backup, only_if_absent=True, expected_current=failed_flow)

    assert storage._tokens_path().read_bytes() == b"old-token"
    assert storage._client_info_path().read_bytes() == b"later-client"
    assert storage._meta_path().read_bytes() == b"old-meta"


def test_storage_content_rollback_preserves_later_token(tmp_path) -> None:
    storage = HermesTokenStorage("tracery", hermes_home=tmp_path)
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    storage._tokens_path().write_bytes(b"old-token")
    storage._client_info_path().write_bytes(b"old-client")
    storage._meta_path().write_bytes(b"old-meta")
    backup = storage.snapshot()

    storage.remove()
    storage._client_info_path().write_bytes(b"failed-client")
    failed_flow = storage.snapshot()
    storage._tokens_path().write_bytes(b"later-token")
    storage.restore(backup, only_if_absent=True, expected_current=failed_flow)

    assert storage._tokens_path().read_bytes() == b"later-token"
    assert storage._client_info_path().read_bytes() == b"old-client"
    assert storage._meta_path().read_bytes() == b"old-meta"


def test_storage_content_rollback_handles_per_file_mixed_state(tmp_path) -> None:
    storage = HermesTokenStorage("tracery", hermes_home=tmp_path)
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    storage._tokens_path().write_bytes(b"old-token")
    storage._client_info_path().write_bytes(b"old-client")
    storage._meta_path().write_bytes(b"old-meta")
    backup = storage.snapshot()

    storage.remove()
    storage._tokens_path().write_bytes(b"failed-token")
    storage._client_info_path().write_bytes(b"failed-client")
    storage._meta_path().write_bytes(b"failed-meta")
    failed_flow = storage.snapshot()
    storage._client_info_path().write_bytes(b"later-client")
    storage._meta_path().unlink()
    storage.restore(backup, only_if_absent=True, expected_current=failed_flow)

    assert storage._tokens_path().read_bytes() == b"old-token"
    assert storage._client_info_path().read_bytes() == b"later-client"
    assert not storage._meta_path().exists()


def test_storage_content_rollback_cleans_up_empty_snapshot(tmp_path) -> None:
    storage = HermesTokenStorage("tracery", hermes_home=tmp_path)
    backup = storage.snapshot()

    storage._tokens_path().parent.mkdir(parents=True)
    storage._client_info_path().write_bytes(b"failed-client")
    storage._meta_path().write_bytes(b"failed-meta")
    failed_flow = storage.snapshot()
    storage.restore(backup, only_if_absent=True, expected_current=failed_flow)

    assert not storage._tokens_path().exists()
    assert not storage._client_info_path().exists()
    assert not storage._meta_path().exists()



def test_web_transaction_lock_is_keyed_by_credential_owner() -> None:
    from hermes_cli.web_server import _mcp_oauth_transaction

    first = SimpleNamespace(
        hermes_home="/profiles/caller-a",
        credential_home="/profiles/owner",
        server_name="tracery",
    )
    second = SimpleNamespace(
        hermes_home="/profiles/caller-b",
        credential_home="/profiles/owner",
        server_name="tracery",
    )

    assert _mcp_oauth_transaction(first) is _mcp_oauth_transaction(second)


def test_tui_flow_rejects_duplicate_shared_owner(monkeypatch, tmp_path) -> None:
    from tui_gateway import mcp_oauth_sessions

    owner_home = tmp_path / "owner"
    _install_profile(monkeypatch, owner_home)
    first_flow = SimpleNamespace(worker_done=False)
    mcp_oauth_sessions._sessions["existing"] = {
        "server_name": "tracery",
        "credential_home": str(owner_home),
        "flow": first_flow,
        "created_at": float("inf"),
    }
    try:
        with pytest.raises(RuntimeError, match="already in progress"):
            mcp_oauth_sessions.start_flow(
                str(tmp_path / "caller-b"),
                "tracery",
                {
                    "url": "https://beproud.tracery.jp/mcp",
                    "oauth": {"credential_profile": "welby"},
                },
            )
    finally:
        mcp_oauth_sessions._sessions.pop("existing", None)


def test_manager_rebuilds_provider_when_credential_profile_changes(monkeypatch) -> None:
    manager = MCPOAuthManager()
    built = []

    def _build(server_name, entry):
        provider = SimpleNamespace(build_index=len(built))
        built.append((server_name, dict(entry.oauth_config or {}), provider))
        return provider

    monkeypatch.setattr(manager, "_build_provider", _build)

    first = manager.get_or_build_provider(
        "tracery",
        "https://beproud.tracery.jp/mcp",
        {},
    )
    second = manager.get_or_build_provider(
        "tracery",
        "https://beproud.tracery.jp/mcp",
        {"credential_profile": "welby"},
    )

    assert first is not second
    assert [config for _, config, _ in built] == [
        {},
        {"credential_profile": "welby"},
    ]


def test_manager_remove_borrower_skips_shared_owner_tokens(monkeypatch, tmp_path) -> None:
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
    assert owner_token.exists()


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
    assert owner_token.exists()


def test_manager_remove_owner_evicts_all_borrower_caches_for_same_token_home(monkeypatch, tmp_path) -> None:
    owner_home = tmp_path / "owner"
    borrower_a = tmp_path / "borrower-a"
    borrower_b = tmp_path / "borrower-b"
    _install_profile(monkeypatch, owner_home)
    token_path = owner_home / "mcp-tokens" / "tracery.json"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("{}")

    manager = MCPOAuthManager()
    manager._entries[manager._key("tracery", borrower_a)] = _ProviderEntry(
        "https://beproud.tracery.jp/mcp",
        {"credential_profile": "welby"},
        provider=SimpleNamespace(name="a"),
    )
    manager._entries[manager._key("tracery", borrower_b)] = _ProviderEntry(
        "https://beproud.tracery.jp/mcp",
        {"credential_profile": "welby"},
        provider=SimpleNamespace(name="b"),
    )

    manager.remove("tracery", hermes_home=owner_home, oauth_config=None)

    assert not token_path.exists()
    assert manager._key("tracery", borrower_a) not in manager._entries
    assert manager._key("tracery", borrower_b) not in manager._entries


def test_manager_remove_owner_skips_invalid_cached_profile_and_continues(monkeypatch, tmp_path) -> None:
    owner_home = tmp_path / "owner"
    borrower_good = tmp_path / "borrower-good"
    borrower_bad = tmp_path / "borrower-bad"
    unrelated = tmp_path / "unrelated"
    _install_profile(monkeypatch, owner_home)
    token_path = owner_home / "mcp-tokens" / "tracery.json"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("{}")

    manager = MCPOAuthManager()
    manager._entries[manager._key("tracery", borrower_bad)] = _ProviderEntry(
        "https://beproud.tracery.jp/mcp",
        {"credential_profile": "missing"},
        provider=SimpleNamespace(name="bad"),
    )
    manager._entries[manager._key("tracery", borrower_good)] = _ProviderEntry(
        "https://beproud.tracery.jp/mcp",
        {"credential_profile": "welby"},
        provider=SimpleNamespace(name="good"),
    )
    manager._entries[manager._key("other", unrelated)] = _ProviderEntry(
        "https://example.invalid/mcp",
        {},
        provider=SimpleNamespace(name="unrelated"),
    )

    manager.remove("tracery", hermes_home=owner_home, oauth_config=None)

    assert not token_path.exists()
    assert manager._key("tracery", borrower_good) not in manager._entries
    assert manager._key("tracery", borrower_bad) not in manager._entries
    assert manager._key("other", unrelated) in manager._entries


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
