"""Minimal Google Antigravity OAuth PKCE auth support."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict
from urllib.parse import parse_qs, urlencode, urlparse

from hermes_cli.auth_constants import AuthError

PROVIDER_ID = "google-antigravity"
ANTIGRAVITY_BASE_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8765/callback"
DEFAULT_TIMEOUT_SECONDS = 180.0
SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)
_SECRET_RE = re.compile(
    r"\b(access_token|refresh_token|id_token|client_secret|code|code_verifier|verifier)\s*[=:]\s*[^\s&]+",
    re.IGNORECASE,
)


def discover_antigravity_project(*, access_token: str, base_url: str = ANTIGRAVITY_BASE_URL) -> str | None:
    """Best-effort Code Assist project discovery. Never raises credential-bearing details."""
    token = str(access_token or "").strip()
    if not token:
        return None
    from hermes_cli.auth_constants import httpx

    normalized = str(base_url or ANTIGRAVITY_BASE_URL).rstrip("/")
    try:
        response = httpx.get(
            f"{normalized}/projects",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=15.0,
        )
        if response.status_code in (401, 403) or response.status_code >= 500:
            return None
        response.raise_for_status()
        payload = response.json() or {}
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    candidates = []
    for key in ("project_id", "projectId", "project"):
        candidates.append(payload.get(key))
    projects = payload.get("projects") or payload.get("projectIds") or []
    if isinstance(projects, list):
        for item in projects:
            if isinstance(item, dict):
                candidates.extend(item.get(key) for key in ("project_id", "projectId", "id", "name"))
            else:
                candidates.append(item)
    for candidate in candidates:
        project_id = str(candidate or "").strip()
        if project_id:
            return project_id.rsplit("/", 1)[-1]
    return None


class AntigravityAuthError(Exception):
    """Google Antigravity auth error with credential material redacted."""

    def __init__(self, message: str, *, code: str):
        self.code = code
        super().__init__(_SECRET_RE.sub(lambda m: f"{m.group(1)}=[redacted]", str(message)))


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _pkce_code_verifier() -> str:
    return _b64url(os.urandom(64))


def _pkce_code_challenge(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def _new_state() -> str:
    return _b64url(os.urandom(24))


def _client_identity() -> Dict[str, str]:
    client_id = (os.getenv("HERMES_ANTIGRAVITY_CLIENT_ID") or "").strip()
    if not client_id:
        raise AntigravityAuthError(
            "Google Antigravity OAuth client ID is unavailable.",
            code="antigravity_client_identity_missing",
        )
    client_secret = (os.getenv("HERMES_ANTIGRAVITY_CLIENT_SECRET") or "").strip()
    identity = {"client_id": client_id}
    if client_secret:
        identity["client_secret"] = client_secret
    return identity


def _validate_redirect_uri(redirect_uri: str) -> None:
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.port:
        raise AntigravityAuthError(
            "Google Antigravity callback must use a loopback HTTP URL with a port.",
            code="antigravity_redirect_invalid",
        )


def build_authorize_url(*, client_id: str, redirect_uri: str, state: str, code_challenge: str) -> str:
    _validate_redirect_uri(redirect_uri)
    return GOOGLE_AUTH_URL + "?" + urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(SCOPES),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    })


def _wait_for_callback(redirect_uri: str, *, timeout_seconds: float, on_ready: Callable[[], None]) -> Dict[str, Any]:
    parsed = urlparse(redirect_uri)
    result: Dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            request = urlparse(self.path)
            if request.path != (parsed.path or "/"):
                self.send_error(404)
                return
            query = parse_qs(request.query)
            result.update({key: values[0] for key, values in query.items() if values})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Authorization received. You may close this tab.")

        def log_message(self, format: str, *args: Any) -> None:
            return

    try:
        server = HTTPServer((parsed.hostname or "127.0.0.1", parsed.port or 0), Handler)
    except OSError as exc:
        raise AntigravityAuthError(
            f"Could not start Google Antigravity callback listener ({type(exc).__name__}).",
            code="antigravity_callback_bind_failed",
        ) from None
    try:
        on_ready()
        deadline = time.monotonic() + timeout_seconds
        server.timeout = min(0.25, timeout_seconds)
        while time.monotonic() < deadline and not result:
            server.handle_request()
    finally:
        server.server_close()
    if not result:
        raise AntigravityAuthError(
            "Google Antigravity authorization timed out waiting for the local callback.",
            code="antigravity_callback_timeout",
        )
    return result


def _exchange_code_for_tokens(
    *, client_id: str, client_secret: str = "", code: str, redirect_uri: str, code_verifier: str,
) -> Dict[str, Any]:
    from hermes_cli.auth_constants import httpx

    data = {
        "client_id": client_id,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier,
    }
    if client_secret:
        data["client_secret"] = client_secret
    try:
        response = httpx.post(GOOGLE_TOKEN_URL, data=data, timeout=30.0)
    except Exception as exc:
        raise AntigravityAuthError(
            f"Google Antigravity token exchange failed ({type(exc).__name__}).",
            code="antigravity_exchange_network_error",
        ) from None
    if response.status_code >= 400:
        raise AntigravityAuthError(
            f"Google Antigravity token exchange was rejected (HTTP {response.status_code}).",
            code="antigravity_exchange_rejected",
        )
    try:
        payload = response.json()
    except Exception:
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
        raise AntigravityAuthError(
            "Google Antigravity token exchange returned no access token.",
            code="antigravity_exchange_malformed",
        )
    return payload


def save_state(state: Dict[str, Any], *, set_active: bool = True) -> None:
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store, _store_provider_state

    saved = dict(state)
    saved["base_url"] = str(saved.get("base_url") or ANTIGRAVITY_BASE_URL).rstrip("/")
    with _auth_store_lock():
        store = _load_auth_store()
        _store_provider_state(store, PROVIDER_ID, saved, set_active=set_active)
        _save_auth_store(store)


def run_pkce_login(
    *,
    open_browser: bool = True,
    redirect_uri: str = DEFAULT_REDIRECT_URI,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    wait_for_callback: Callable[..., Dict[str, Any]] | None = None,
    exchange_code: Callable[..., Dict[str, Any]] | None = None,
    discover_project: Callable[..., str | None] | None = None,
    browser_open: Callable[[str], Any] | None = None,
    _client_identity_override: Dict[str, str] | None = None,
    **_: Any,
) -> Dict[str, Any]:
    _validate_redirect_uri(redirect_uri)
    identity = dict(_client_identity_override or _client_identity())
    verifier = _pkce_code_verifier()
    state = _new_state()
    authorize_url = build_authorize_url(
        client_id=identity["client_id"],
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=_pkce_code_challenge(verifier),
    )

    def start_authorization() -> None:
        if open_browser:
            (browser_open or webbrowser.open)(authorize_url)
        else:
            print("Open this Google Antigravity sign-in URL in a browser, then return here:\n" + authorize_url)

    try:
        callback = (wait_for_callback or _wait_for_callback)(
            redirect_uri,
            timeout_seconds=timeout_seconds,
            on_ready=start_authorization,
        )
    except AntigravityAuthError:
        raise
    except Exception as exc:
        raise AntigravityAuthError(
            f"Google Antigravity callback failed ({type(exc).__name__}).",
            code="antigravity_callback_failed",
        ) from None
    if callback.get("state") != state:
        raise AntigravityAuthError(
            "Google Antigravity authorization state did not match. Please retry.",
            code="antigravity_state_mismatch",
        )
    code = callback.get("code")
    if not isinstance(code, str) or not code:
        raise AntigravityAuthError(
            "Google Antigravity authorization callback carried no code.",
            code="antigravity_callback_missing_code",
        )
    payload = (exchange_code or _exchange_code_for_tokens)(
        client_id=identity["client_id"],
        client_secret=identity.get("client_secret", ""),
        code=code,
        redirect_uri=redirect_uri,
        code_verifier=verifier,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
        raise AntigravityAuthError(
            "Google Antigravity token exchange returned no access token.",
            code="antigravity_exchange_malformed",
        )
    try:
        expires_in = max(0.0, float(payload.get("expires_in", 3600)))
    except (TypeError, ValueError):
        expires_in = 3600.0
    saved = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", ""),
        "id_token": payload.get("id_token", ""),
        "expires_at": time.time() + expires_in,
        "redirect_uri": redirect_uri,
        "auth_type": "oauth_pkce",
        "base_url": ANTIGRAVITY_BASE_URL,
    }
    project_id = str(payload.get("project_id") or payload.get("project") or "").strip()
    if not project_id:
        try:
            discovered = (discover_project or discover_antigravity_project)(
                access_token=payload["access_token"],
                base_url=ANTIGRAVITY_BASE_URL,
            )
            project_id = str(discovered or "").strip()
        except Exception:
            project_id = ""
    if project_id:
        saved["project_id"] = project_id
    save_state(saved, set_active=True)
    return saved


def _expires_at_from_payload(payload: Dict[str, Any]) -> float:
    try:
        expires_in = max(0.0, float(payload.get("expires_in", 3600)))
    except (TypeError, ValueError):
        expires_in = 3600.0
    return time.time() + expires_in


def _runtime_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "api_key": str((state or {}).get("access_token") or "").strip(),
        "base_url": str((state or {}).get("base_url") or ANTIGRAVITY_BASE_URL).rstrip("/"),
        "expires_at": float((state or {}).get("expires_at") or 0),
        "project_id": str((state or {}).get("project_id") or "").strip(),
        "source": "hermes-auth-store",
    }


def _missing_or_expired_error() -> AuthError:
    return AuthError(
        "Google Antigravity sign-in is missing or expired. Run `hermes auth add google-antigravity`.",
        provider=PROVIDER_ID,
        code="antigravity_auth_missing_or_expired",
        relogin_required=True,
    )


def _refresh_token_for_access_token(*, stale_access_token: str | None = None) -> Dict[str, Any]:
    """Refresh the stored Google Antigravity bearer once and persist it back to auth.json."""
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store, _store_provider_state

    with _auth_store_lock():
        store = _load_auth_store()
        state = dict(((store.get("providers") or {}).get(PROVIDER_ID)) or {})
        token = str(state.get("access_token") or "").strip()
        if stale_access_token and token and token != stale_access_token:
            return _runtime_from_state(state)
        refresh_token = str(state.get("refresh_token") or "").strip()
        if not refresh_token:
            raise _missing_or_expired_error()
        identity = _client_identity()
        from hermes_cli.auth_constants import httpx

        data = {
            "client_id": identity["client_id"],
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        if identity.get("client_secret"):
            data["client_secret"] = identity["client_secret"]
        try:
            response = httpx.post(GOOGLE_TOKEN_URL, data=data, timeout=30.0)
        except Exception as exc:
            raise AntigravityAuthError(
                f"Google Antigravity token refresh failed ({type(exc).__name__}).",
                code="antigravity_refresh_network_error",
            ) from None
        if response.status_code >= 400:
            raise AntigravityAuthError(
                f"Google Antigravity token refresh was rejected (HTTP {response.status_code}).",
                code="antigravity_refresh_rejected",
            )
        try:
            payload = response.json()
        except Exception:
            payload = None
        if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str):
            raise AntigravityAuthError(
                "Google Antigravity token refresh returned no access token.",
                code="antigravity_refresh_malformed",
            )
        saved = dict(state)
        saved["access_token"] = payload["access_token"]
        saved["refresh_token"] = payload.get("refresh_token") or refresh_token
        saved["id_token"] = payload.get("id_token") or state.get("id_token", "")
        saved["expires_at"] = _expires_at_from_payload(payload)
        saved["base_url"] = str(payload.get("base_url") or state.get("base_url") or ANTIGRAVITY_BASE_URL).rstrip("/")
        project_id = str(payload.get("project_id") or payload.get("project") or state.get("project_id") or "").strip()
        if project_id:
            saved["project_id"] = project_id
        _store_provider_state(store, PROVIDER_ID, saved, set_active=store.get("active_provider") == PROVIDER_ID)
        _save_auth_store(store)
        return _runtime_from_state(saved)


def refresh_antigravity_runtime_credentials(*, stale_access_token: str | None = None) -> Dict[str, Any]:
    return _refresh_token_for_access_token(stale_access_token=stale_access_token)


def resolve_antigravity_runtime_credentials() -> Dict[str, Any]:
    from hermes_cli.auth import get_provider_auth_state

    state = get_provider_auth_state(PROVIDER_ID)
    token = str((state or {}).get("access_token") or "").strip()
    try:
        expires_at = float((state or {}).get("expires_at") or 0)
    except (TypeError, ValueError):
        expires_at = 0.0
    if not token or expires_at <= time.time():
        raise _missing_or_expired_error()
    return _runtime_from_state({**(state or {}), "expires_at": expires_at})


def get_antigravity_auth_status() -> Dict[str, Any]:
    from hermes_cli.auth import get_provider_auth_state

    state = get_provider_auth_state(PROVIDER_ID)
    status = {
        "provider": PROVIDER_ID,
        "logged_in": False,
        "auth_type": None,
        "redirect_uri": None,
        "expires_at": None,
        "api_base_url": ANTIGRAVITY_BASE_URL,
    }
    if not isinstance(state, dict) or not state.get("access_token"):
        return status
    status.update({
        "auth_type": state.get("auth_type") or "oauth_pkce",
        "redirect_uri": state.get("redirect_uri"),
        "expires_at": state.get("expires_at"),
    })
    try:
        expires_at = float(state.get("expires_at") or 0)
    except (TypeError, ValueError):
        expires_at = 0.0
    status["logged_in"] = bool(expires_at > time.time())
    return status
