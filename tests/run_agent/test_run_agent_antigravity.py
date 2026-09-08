from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class _Unauthorized(RuntimeError):
    status_code = 401


def _chat_response(text: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="gemini-test")


def _new_antigravity_agent(
    tmp_path,
    monkeypatch,
    *,
    access_token="old-access-token",
    refresh_token: str | None = "refresh-token",
    project_id: str = "project-before-refresh",
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_ID", "client-id")
    monkeypatch.setenv("HERMES_ANTIGRAVITY_CLIENT_SECRET", "client-secret")

    from hermes_cli.antigravity_auth import ANTIGRAVITY_BASE_URL, save_state
    from run_agent import AIAgent

    state = {
        "access_token": access_token,
        "expires_at": 4_102_444_800.0,
        "project_id": project_id,
        "auth_type": "oauth_pkce",
        "base_url": ANTIGRAVITY_BASE_URL,
    }
    if refresh_token is not None:
        state["refresh_token"] = refresh_token
    save_state(state)

    created_clients: list[dict] = []

    def fake_openai(**kwargs):
        created_clients.append(dict(kwargs))
        client = MagicMock()
        client.api_key = kwargs.get("api_key")
        client.close = MagicMock()
        return client

    monkeypatch.setattr("agent.process_bootstrap.OpenAI", fake_openai)
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            api_key=access_token,
            base_url=ANTIGRAVITY_BASE_URL,
            provider="google-antigravity",
            api_mode="chat_completions",
            model="gemini-test",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent, created_clients


def test_antigravity_first_401_refreshes_store_rebuilds_client_and_retries_once(tmp_path, monkeypatch, capsys):
    from hermes_cli import auth_constants
    from hermes_cli.antigravity_auth import ANTIGRAVITY_BASE_URL

    agent, created_clients = _new_antigravity_agent(tmp_path, monkeypatch)
    calls: list[str] = []
    token_posts: list[dict] = []

    class TokenResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
                "project_id": "project-after-refresh",
            }

    class FakeTokenHttp:
        @staticmethod
        def post(url, **kwargs):
            token_posts.append({"url": url, **kwargs})
            return TokenResponse()

    def fake_api_call(_api_kwargs):
        calls.append(agent.api_key)
        if len(calls) == 1:
            raise _Unauthorized("expired")
        return _chat_response("recovered")

    monkeypatch.setattr(auth_constants, "httpx", FakeTokenHttp())
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert result["final_response"] == "recovered"
    assert calls == ["old-access-token", "new-access-token"]
    assert len(token_posts) == 1
    assert token_posts[0]["data"]["grant_type"] == "refresh_token"
    assert token_posts[0]["data"]["refresh_token"] == "refresh-token"
    assert created_clients[-1]["api_key"] == "new-access-token"
    assert created_clients[-1]["base_url"] == ANTIGRAVITY_BASE_URL
    assert created_clients[-1]["default_headers"]["x-goog-user-project"] == "project-after-refresh"

    stored = json.loads((tmp_path / "auth.json").read_text())["providers"]["google-antigravity"]
    assert stored["access_token"] == "new-access-token"
    assert stored["refresh_token"] == "new-refresh-token"
    assert stored["project_id"] == "project-after-refresh"

    observed = capsys.readouterr().out + capsys.readouterr().err
    for secret in ("old-access-token", "new-access-token", "refresh-token", "new-refresh-token", "client-secret"):
        assert secret not in observed


def test_antigravity_refresh_rebuild_uses_refreshed_project_header(tmp_path, monkeypatch):
    from hermes_cli import auth_constants

    agent, created_clients = _new_antigravity_agent(tmp_path, monkeypatch, project_id="old-project")
    client_kwargs = getattr(agent, "_client_kwargs")
    client_kwargs["default_headers"] = {"x-goog-user-project": "old-project", "x-other": "keep"}
    calls: list[str] = []

    class TokenResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"access_token": "new-access-token", "expires_in": 3600, "project_id": "new-project"}

    class FakeTokenHttp:
        @staticmethod
        def post(url, **kwargs):
            return TokenResponse()

    def fake_api_call(_api_kwargs):
        calls.append(agent.api_key)
        if len(calls) == 1:
            raise _Unauthorized("expired")
        return _chat_response("recovered")

    monkeypatch.setattr(auth_constants, "httpx", FakeTokenHttp())
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["completed"] is True
    assert calls == ["old-access-token", "new-access-token"]
    assert created_clients[-1]["api_key"] == "new-access-token"
    assert created_clients[-1]["default_headers"]["x-goog-user-project"] == "new-project"
    assert created_clients[-1]["default_headers"]["x-other"] == "keep"


def test_antigravity_refresh_failure_stops_without_resending_expired_token(tmp_path, monkeypatch):
    from hermes_cli import auth_constants

    agent, _created_clients = _new_antigravity_agent(tmp_path, monkeypatch)
    calls: list[str] = []

    class RejectedTokenResponse:
        status_code = 400

        @staticmethod
        def json():
            return {"error": "invalid_grant"}

    class FakeTokenHttp:
        @staticmethod
        def post(url, **kwargs):
            return RejectedTokenResponse()

    def fake_api_call(_api_kwargs):
        calls.append(agent.api_key)
        raise _Unauthorized("expired")

    monkeypatch.setattr(auth_constants, "httpx", FakeTokenHttp())
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["failed"] is True
    assert calls == ["old-access-token"]


def test_antigravity_missing_refresh_token_stops_without_second_send(tmp_path, monkeypatch):
    agent, _created_clients = _new_antigravity_agent(tmp_path, monkeypatch, refresh_token=None)
    calls: list[str] = []

    def fake_api_call(_api_kwargs):
        calls.append(agent.api_key)
        raise _Unauthorized("expired")

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["failed"] is True
    assert calls == ["old-access-token"]


def test_antigravity_second_401_does_not_refresh_or_retry_again(tmp_path, monkeypatch):
    from hermes_cli import auth_constants

    agent, _created_clients = _new_antigravity_agent(tmp_path, monkeypatch)
    calls: list[str] = []
    token_posts: list[dict] = []

    class TokenResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"access_token": "new-access-token", "expires_in": 3600, "project_id": "project-after-refresh"}

    class FakeTokenHttp:
        @staticmethod
        def post(url, **kwargs):
            token_posts.append({"url": url, **kwargs})
            return TokenResponse()

    def fake_api_call(_api_kwargs):
        calls.append(agent.api_key)
        raise _Unauthorized("still rejected")

    monkeypatch.setattr(auth_constants, "httpx", FakeTokenHttp())
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")

    assert result["failed"] is True
    assert calls == ["old-access-token", "new-access-token"]
    assert len(token_posts) == 1
