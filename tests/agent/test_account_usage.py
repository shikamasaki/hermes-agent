from types import SimpleNamespace

import pytest

from agent import account_usage


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, calls, payload):
        self.calls = calls
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers):
        self.calls.append({"url": url, "headers": headers})
        return _FakeResponse(self.payload)


@pytest.fixture
def codex_usage_payload():
    return {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {
                "used_percent": 21,
                "reset_at": 1779846359,
            },
            "secondary_window": {
                "used_percent": 4,
                "reset_at": 1780230796,
            },
        },
        "credits": {"has_credits": False},
    }


def test_codex_usage_prefers_explicit_live_agent_credentials(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy auth should not be used")),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    assert snapshot.provider == "openai-codex"
    assert snapshot.plan == "Plus"
    assert [w.label for w in snapshot.windows] == ["Session", "Weekly"]
    assert snapshot.windows[0].used_percent == 21
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer live-agent-token"


def test_codex_usage_falls_back_to_native_credential_pool(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    # Pool fallback fires only on AuthError (the documented "no creds" mode of
    # the resolver), NOT on arbitrary exceptions — see the transient-error guard
    # test below.
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(
            account_usage.AuthError("no singleton auth", provider="openai-codex", code="codex_auth_missing")
        ),
    )

    pool_entry = SimpleNamespace(
        runtime_api_key="pooled-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
    )
    pool = SimpleNamespace(select=lambda: pool_entry)

    import agent.credential_pool as credential_pool

    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert snapshot.windows[0].label == "Session"
    assert snapshot.windows[1].label == "Weekly"
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer pooled-token"
    # Pool creds have no account_id concept — the ChatGPT-Account-Id header must
    # be omitted rather than sent stale/wrong.
    assert "ChatGPT-Account-Id" not in calls[0]["headers"]




def test_codex_usage_account_id_read_failure_keeps_singleton_token(monkeypatch, codex_usage_payload):
    """When the resolver succeeds but the separate account_id read raises, the
    working singleton token must still be used (best-effort account_id), NOT
    abandoned in favor of a header-less pool credential."""
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: {
            "api_key": "singleton-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )
    monkeypatch.setattr(
        account_usage,
        "_read_codex_tokens",
        lambda *a, **k: (_ for _ in ()).throw(
            account_usage.AuthError("partial store", provider="openai-codex", code="codex_auth_invalid_shape")
        ),
    )

    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda provider: (_ for _ in ()).throw(AssertionError("pool must not be consulted")),
    )

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert calls[0]["headers"]["Authorization"] == "Bearer singleton-token"
    # account_id read failed → header omitted, but the singleton token is kept.
    assert "ChatGPT-Account-Id" not in calls[0]["headers"]



class _FakeStatusResponse(_FakeResponse):
    def __init__(self, payload, status_code=200):
        super().__init__(payload)
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = account_usage.httpx.Request("GET", "https://example.invalid")
            response = account_usage.httpx.Response(self.status_code, request=request)
            raise account_usage.httpx.HTTPStatusError("rejected", request=request, response=response)


class _FakeAntigravityClient:
    def __init__(self, calls, response):
        self.calls = calls
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers):
        self.calls.append({"url": url, "headers": headers})
        return self.response


def _save_antigravity_auth(tmp_path, monkeypatch, *, project_id="project-123"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.antigravity_auth import ANTIGRAVITY_BASE_URL, save_state

    state = {
        "access_token": "access-token-secret",
        "refresh_token": "refresh-token-secret",
        "expires_at": 4102444800.0,
        "auth_type": "oauth_pkce",
        "base_url": ANTIGRAVITY_BASE_URL,
    }
    if project_id:
        state["project_id"] = project_id
    save_state(state, set_active=True)


def test_antigravity_usage_fetches_quota_summary(monkeypatch, tmp_path):
    _save_antigravity_auth(tmp_path, monkeypatch, project_id="project-123")
    calls = []
    payload = {
        "planType": "pro",
        "quotaSummary": {
            "limits": [
                {"label": "Session", "usedPercent": 20, "resetAt": "2026-09-09T00:00:00Z"},
                {"label": "Daily", "usagePercentage": 0.35, "resetTime": 1780230796},
            ],
            "details": ["Included requests: 1000"],
        },
    }
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeAntigravityClient(calls, _FakeStatusResponse(payload)),
    )

    snapshot = account_usage.fetch_account_usage("google-antigravity")

    assert snapshot is not None
    assert snapshot.provider == "google-antigravity"
    assert snapshot.plan == "Pro"
    assert [w.label for w in snapshot.windows] == ["Session", "Daily"]
    assert snapshot.windows[0].used_percent == 20
    assert snapshot.windows[1].used_percent == 35
    assert snapshot.details == ("Included requests: 1000",)
    assert calls[0]["url"].endswith("/projects/project-123/quotaSummary")
    assert calls[0]["headers"]["Authorization"] == "Bearer access-token-secret"


def test_antigravity_usage_missing_project_fails_open(monkeypatch, tmp_path):
    _save_antigravity_auth(tmp_path, monkeypatch, project_id="")
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: (_ for _ in ()).throw(AssertionError("HTTP must not run without project_id")),
    )

    assert account_usage.fetch_account_usage("google-antigravity") is None


@pytest.mark.parametrize(
    "response",
    [
        _FakeStatusResponse({}, status_code=401),
        _FakeStatusResponse({}, status_code=403),
        _FakeStatusResponse(ValueError("bad json")),
    ],
)
def test_antigravity_usage_auth_rejection_and_invalid_json_fail_open(monkeypatch, tmp_path, response):
    _save_antigravity_auth(tmp_path, monkeypatch, project_id="project-123")
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeAntigravityClient(calls, response),
    )

    assert account_usage.fetch_account_usage("google-antigravity") is None


def test_antigravity_usage_network_error_fails_open(monkeypatch, tmp_path):
    _save_antigravity_auth(tmp_path, monkeypatch, project_id="project-123")

    class _NetworkErrorClient(_FakeAntigravityClient):
        def get(self, url, headers):
            raise account_usage.httpx.ConnectError("connect failed")

    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _NetworkErrorClient([], _FakeStatusResponse({})),
    )

    assert account_usage.fetch_account_usage("google-antigravity") is None



# ── Banked rate-limit reset credits (`/usage reset`) ─────────────────────────


class _FakeResetClient:
    """GET returns the usage payload; POST returns the consume payload."""

    def __init__(self, calls, usage_payload, consume_payload=None):
        self.calls = calls
        self.usage_payload = usage_payload
        self.consume_payload = consume_payload or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers):
        self.calls.append({"method": "GET", "url": url, "headers": headers})
        return _FakeResponse(self.usage_payload)

    def post(self, url, headers=None, json=None):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json})
        return _FakeResponse(self.consume_payload)


def _usage_payload_with_resets(primary_used, secondary_used, banked):
    return {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {"used_percent": primary_used, "reset_at": 1779846359},
            "secondary_window": {"used_percent": secondary_used, "reset_at": 1780230796},
        },
        "rate_limit_reset_credits": {"available_count": banked},
        "credits": {"has_credits": False},
    }
















def test_redeem_missing_credentials_reports_unavailable(monkeypatch):
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key: (_ for _ in ()).throw(RuntimeError("no creds")),
    )

    result = account_usage.redeem_codex_reset_credit()

    assert result.status == "unavailable"
    assert "hermes auth" in result.message
