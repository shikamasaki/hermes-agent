"""OAuth + credential storage for the ``google-antigravity`` provider.

Hermes owns this OAuth state end to end: it runs its own loopback PKCE
flow, stores the grant in Hermes' profile-aware ``auth.json`` (0o600, via
``hermes_cli.auth._save_auth_store``), and refreshes it itself. It never
reads the Antigravity CLI's own credential store, keychain, or logs.

Security contract
-----------------
* Access/refresh/ID tokens and the OAuth client secret are credentials.
  They are never logged, never placed in exception messages, never written
  to ``config.yaml``, and never surfaced to model context. ``redact_state``
  and ``AntigravityAuthError`` enforce this at the two egress points.
* The desktop OAuth client secret is not checked in. It comes from
  ``HERMES_ANTIGRAVITY_CLIENT_ID`` / ``HERMES_ANTIGRAVITY_CLIENT_SECRET``,
  or is discovered at runtime from the locally installed ``agy`` binary.
  Discovery result is held in memory only.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

logger = logging.getLogger(__name__)

PROVIDER_ID = "google-antigravity"

ANTIGRAVITY_BASE_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

ANTIGRAVITY_SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
)

#: Fields that must never leave this module in cleartext.
_SENSITIVE_KEYS = frozenset({
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "code",
    "code_verifier",
})

#: Refresh this many seconds before the token actually expires.
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120

_OAUTH_CALLBACK_TIMEOUT_SECONDS = 180.0
_TOKEN_HTTP_TIMEOUT_SECONDS = 30.0

# Patterns for scrubbing credentials that leaked into a free-form string
# (an upstream error body, a traceback, a URL with a code= param).
_REDACT_PATTERNS = (
    re.compile(r"ya29\.[A-Za-z0-9._\-]+"),
    re.compile(r"1//[A-Za-z0-9._\-]+"),
    re.compile(r"GOCSPX-[A-Za-z0-9._\-]+"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9._\-]+"),
    re.compile(
        r"((?:access_token|refresh_token|id_token|client_secret|code|code_verifier)"
        r"\s*[=:]\s*\"?)([A-Za-z0-9._\-/+]{6,})"
    ),
)

_REDACTED = "[redacted]"


def redact_text(text: Any) -> str:
    """Scrub anything token-shaped out of a free-form string."""
    out = str(text or "")
    for pattern in _REDACT_PATTERNS:
        if pattern.groups >= 2:
            out = pattern.sub(lambda m: m.group(1) + _REDACTED, out)
        else:
            out = pattern.sub(_REDACTED, out)
    return out


class AntigravityAuthError(Exception):
    """Auth failure whose message is always scrubbed of credentials.

    Every raise site in this module funnels through here, so a token can
    never reach a log handler, a CLI stderr dump, or model context via an
    exception string.
    """

    def __init__(self, message: str, *, code: str = "antigravity_auth_error"):
        self.code = code
        super().__init__(redact_text(message))


def redact_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a status-safe view of a stored grant.

    Sensitive fields are replaced with a presence marker so ``hermes auth
    status`` can still say "a refresh token exists" without printing it.
    """
    safe: Dict[str, Any] = {}
    for key, value in (state or {}).items():
        if key in _SENSITIVE_KEYS:
            safe[f"has_{key}"] = bool(value)
        else:
            safe[key] = value
    return safe


# ── Client identity ──────────────────────────────────────────────────


#: Markers emitted near the Cloud Code CLI client id in the ``agy`` binary.
_CLIENT_ID_MARKERS = ("CLOUD_CODE_URL",)

#: Markers emitted near the Cloud Code CLI client secret. Go's length-sorted
#: string table keeps these auth-host literals in the secret's neighbourhood,
#: which is what lets us pair a secret without relying on candidate ordinal.
_CLIENT_SECRET_MARKERS = (
    "https://auth.cloud.google/authorize",
    "https://cloudcode-pa.googleapis.com",
)

#: How far before/after a candidate we will look for its marker.
_MARKER_WINDOW_BYTES = 512

#: How many bytes of pure separator (whitespace/newlines) may sit between a
#: client and a secret for them to still count as "directly adjacent" —
#: i.e. compiled back-to-back in the binary's string table, not merely in
#: the same neighbourhood.
_ADJACENCY_GAP_BYTES = 2


def _direct_adjacent_secret(blob: str, client_match, secret_matches):
    """Return the secret compiled immediately after *client_match*, if any.

    This is the strongest possible evidence of pairing: the two literals
    are back-to-back in the string table, so there is no room for another
    client/secret to have been interleaved. Only applied to the *selected*
    client — never used to pick between competing clients.
    """
    for secret_match in secret_matches:
        gap = secret_match.start() - client_match.end()
        if 0 <= gap <= _ADJACENCY_GAP_BYTES and not blob[client_match.end():secret_match.start()].strip():
            return secret_match
    return None


def _select_by_marker(matches, markers, distance_fn):
    """Pick the candidate closest to any *marker*, or None if none is marked.

    Returning None when nothing is marked is deliberate: it is the signal
    that locality could not justify a pairing.
    """
    scored = []
    for match in matches:
        for marker in markers:
            distance = distance_fn(match, marker, _MARKER_WINDOW_BYTES)
            if distance is not None:
                scored.append((distance, match.start(), match))
    if not scored:
        return None
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def _extract_client_identity_from_strings(blob: str) -> Optional[Dict[str, str]]:
    """Extract the installed-app identity from non-secret ``agy`` strings.

    Google's installed client secrets use a fixed 28-character payload after
    ``GOCSPX-``. Matching that exact shape is important because current agy
    binaries can place two compiled copies directly adjacent to one another.

    Pairing rule
    ------------
    Client IDs and secrets must each be located by *evidence* — direct
    adjacency or a neighbouring marker — never by candidate ordinal. Go
    sorts its string table by length, so agy's two ``GOCSPX-`` secrets can
    land back-to-back in the 35-character bucket while the two client IDs
    live in an unrelated region (~565 KB away in 1.1.18). The two lists
    therefore share no ordering, and index pairing can silently yield a
    mismatched client_id/client_secret whose token exchange Google rejects
    with an opaque HTTP 401.

    Once the client is selected (by its ``CLOUD_CODE_URL`` marker, or as
    the sole candidate), the secret is chosen in this order:

    1. A secret compiled directly adjacent to the selected client — the
       strongest possible evidence, since nothing could be interleaved.
    2. A secret located by its own auth/cloudcode marker locality.
    3. A single unambiguous secret candidate.

    When none of these establish a pairing, we return None so the caller
    can raise actionable guidance instead of guessing.
    """
    client_matches = list(
        re.finditer(
            r"(\d{8,}-[a-z0-9]{20,}\.apps\.googleusercontent\.com)",
            blob,
        )
    )
    secret_matches = list(
        re.finditer(r"(GOCSPX-[A-Za-z0-9_-]{28})", blob)
    )
    if not client_matches or not secret_matches:
        return None

    def _nearest_preceding(match: "re.Match[str]", marker: str, window: int) -> Optional[int]:
        """Distance from *marker* to *match*, or None when absent."""
        pos = blob.rfind(marker, max(0, match.start() - window), match.start())
        return None if pos < 0 else match.start() - pos

    # The agy binary contains both the desktop and the Cloud Code CLI OAuth
    # client. The CLI client is emitted next to its Cloud Code server
    # override marker; prefer that local association.
    client_match = _select_by_marker(client_matches, _CLIENT_ID_MARKERS, _nearest_preceding)
    if client_match is None:
        if len(client_matches) > 1:
            # Two indistinguishable clients and no marker: guessing risks
            # pairing the wrong tenant. Fail loudly instead.
            return None
        client_match = client_matches[0]

    # Strongest evidence first: a secret compiled directly next to the
    # selected client needs no marker to justify the pairing.
    secret_match = _direct_adjacent_secret(blob, client_match, secret_matches)
    if secret_match is None:
        # Otherwise the CLI secret is emitted next to its own Cloud Code
        # auth-host markers.
        secret_match = _select_by_marker(secret_matches, _CLIENT_SECRET_MARKERS, _nearest_preceding)
    if secret_match is None:
        if len(secret_matches) > 1:
            return None
        secret_match = secret_matches[0]

    return {
        "client_id": client_match.group(1),
        "client_secret": secret_match.group(1),
        "source": "agy",
    }


def _discover_client_identity_from_agy() -> Optional[Dict[str, str]]:
    """Best-effort discovery of the OAuth client from the installed ``agy``.

    Reads only the application's own compiled-in configuration strings from
    the executable on PATH. It does NOT touch the user's credential store,
    keychain, logs, or conversation state. Returns None when ``agy`` is
    absent or its identity cannot be located, so the caller can fail with a
    clear message instead of guessing.
    """
    exe = shutil.which("agy")
    if not exe:
        return None
    try:
        blob = subprocess.run(
            ["strings", "-a", exe],
            capture_output=True,
            timeout=30,
            check=False,
        ).stdout.decode("utf-8", "replace")
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.debug("antigravity: client discovery failed: %s", type(exc).__name__)
        return None

    identity = _extract_client_identity_from_strings(blob)
    if identity is None:
        return None
    # Held in memory for this process only — never persisted, never logged.
    return identity


def resolve_client_identity() -> Dict[str, str]:
    """Resolve the OAuth desktop client identity.

    Order: explicit env overrides, then discovery from the installed ``agy``
    executable. Raises with actionable guidance when neither yields a
    complete identity — we never fall back to a checked-in secret.
    """
    env_id = (os.environ.get("HERMES_ANTIGRAVITY_CLIENT_ID") or "").strip()
    env_secret = (os.environ.get("HERMES_ANTIGRAVITY_CLIENT_SECRET") or "").strip()
    if env_id and env_secret:
        return {"client_id": env_id, "client_secret": env_secret, "source": "env"}

    discovered = _discover_client_identity_from_agy()
    if discovered and discovered.get("client_id") and discovered.get("client_secret"):
        return dict(discovered)

    raise AntigravityAuthError(
        "No Antigravity OAuth client identity available. Set "
        "HERMES_ANTIGRAVITY_CLIENT_ID and HERMES_ANTIGRAVITY_CLIENT_SECRET, "
        "or install the Antigravity CLI (agy) so Hermes can discover it.",
        code="antigravity_client_identity_missing",
    )


# ── Token lifecycle ──────────────────────────────────────────────────


def _refresh_access_token(
    *,
    refresh_token: str,
    client_id: str,
    client_secret: str,
    timeout: float = _TOKEN_HTTP_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Exchange a refresh token for a new access token (pure network call).

    Split out as its own function so tests can substitute it without
    touching disk — the same seam the xAI/Codex providers use.
    """
    try:
        resp = httpx.post(
            GOOGLE_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
    except Exception as exc:
        raise AntigravityAuthError(
            f"Antigravity token refresh failed: {type(exc).__name__}",
            code="antigravity_refresh_network_error",
        ) from None
    if resp.status_code >= 400:
        # Status only — the body can echo the token we just sent.
        raise AntigravityAuthError(
            f"Antigravity token refresh rejected (HTTP {resp.status_code}). "
            "Run 'hermes auth add google-antigravity' to sign in again.",
            code="antigravity_refresh_rejected",
        )
    try:
        payload = resp.json()
    except Exception:
        raise AntigravityAuthError(
            "Antigravity token refresh returned a malformed response.",
            code="antigravity_refresh_malformed",
        ) from None
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise AntigravityAuthError(
            "Antigravity token refresh returned no access token.",
            code="antigravity_refresh_missing_token",
        )
    return payload


def ensure_fresh_access_token(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return *state* with a non-expired access token, refreshing if needed."""
    state = dict(state or {})
    expires_at = state.get("expires_at") or 0
    try:
        expires_at = float(expires_at)
    except (TypeError, ValueError):
        expires_at = 0.0

    if state.get("access_token") and expires_at > (
        time.time() + ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    ):
        return state

    refresh_token = str(state.get("refresh_token") or "").strip()
    if not refresh_token:
        raise AntigravityAuthError(
            "Antigravity session has no refresh token. Run "
            "'hermes auth add google-antigravity' to sign in again.",
            code="antigravity_refresh_token_missing",
        )

    identity = resolve_client_identity()
    payload = _refresh_access_token(
        refresh_token=refresh_token,
        client_id=identity["client_id"],
        client_secret=identity["client_secret"],
    )

    state["access_token"] = payload["access_token"]
    # Google omits refresh_token on refresh — keep the existing one.
    if payload.get("refresh_token"):
        state["refresh_token"] = payload["refresh_token"]
    try:
        expires_in = float(payload.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600.0
    state["expires_at"] = time.time() + expires_in
    return state


# ── Hermes auth store (profile-aware, 0o600) ─────────────────────────


def load_state_with_source() -> tuple[Optional[Dict[str, Any]], Any]:
    """Read the grant and the auth.json path that supplied it."""
    from hermes_cli.auth import _load_auth_store, _load_provider_state_with_source

    try:
        return _load_provider_state_with_source(_load_auth_store(), PROVIDER_ID)
    except Exception:
        return None, None


def load_state() -> Optional[Dict[str, Any]]:
    """Read this provider's grant from Hermes' profile-aware auth chain."""
    state, _source_path = load_state_with_source()
    return state


def save_state(state: Dict[str, Any], *, set_active: bool = False) -> None:
    """Persist the grant via the shared atomic 0o600 auth-store writer."""
    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        _store_provider_state,
    )

    with _auth_store_lock():
        auth_store = _load_auth_store()
        pool = auth_store.get("credential_pool")
        if isinstance(pool, dict):
            pool.pop(PROVIDER_ID, None)
        _store_provider_state(auth_store, PROVIDER_ID, state, set_active=set_active)
        _save_auth_store(auth_store)


def save_state_to_source(state: Dict[str, Any], source_path: Any) -> None:
    """Persist a refreshed token chain back to the store it came from."""
    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _save_provider_state_to_source,
    )

    with _auth_store_lock():
        auth_store = _load_auth_store()
        _save_provider_state_to_source(
            auth_store, PROVIDER_ID, state, source_path
        )


def resolve_antigravity_runtime_credentials() -> Dict[str, Any]:
    """Return per-request credentials for the agent, refreshing if needed."""
    state, source_path = load_state_with_source()
    if not isinstance(state, dict) or not state.get("refresh_token"):
        raise AntigravityAuthError(
            "Not signed in to Google Antigravity. Run "
            "'hermes auth add google-antigravity'.",
            code="antigravity_auth_missing",
        )

    fresh = ensure_fresh_access_token(state)
    if fresh != state:
        try:
            save_state_to_source(fresh, source_path)
        except Exception as exc:
            # A persistence failure must not block the in-flight request,
            # and must not surface the token in the log line.
            logger.debug("antigravity: token persist failed: %s", type(exc).__name__)

    return {
        "provider": PROVIDER_ID,
        "api_mode": "chat_completions",
        "base_url": ANTIGRAVITY_BASE_URL,
        "api_key": fresh.get("access_token", ""),
        "project_id": fresh.get("project_id"),
        "source": "hermes-auth-store",
        "auth_mode": "oauth_pkce",
    }


DEFAULT_ANTIGRAVITY_REDIRECT_URI = "http://127.0.0.1:8765/callback"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


# ── Loopback PKCE login ───────────────────────────────────────────────


def _pkce_code_verifier(length: int = 64) -> str:
    raw = base64.urlsafe_b64encode(os.urandom(length)).decode("ascii")
    return raw.rstrip("=")[:128]


def _pkce_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _validate_antigravity_redirect_uri(redirect_uri: str) -> tuple:
    """Enforce loopback-only PKCE redirects (RFC 8252 native-app guidance)."""
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "http":
        raise AntigravityAuthError(
            "Antigravity PKCE redirect_uri must use http://127.0.0.1 or "
            "http://localhost (never https, and never a remote host).",
            code="antigravity_redirect_invalid",
        )
    host = parsed.hostname or ""
    if host not in {"127.0.0.1", "localhost"}:
        raise AntigravityAuthError(
            "Antigravity PKCE redirect_uri must point to 127.0.0.1 or localhost.",
            code="antigravity_redirect_invalid",
        )
    if not parsed.port:
        raise AntigravityAuthError(
            "Antigravity PKCE redirect_uri must include an explicit port.",
            code="antigravity_redirect_invalid",
        )
    return host, parsed.port, (parsed.path or "/")


def build_authorize_url(
    *, redirect_uri: str, code_challenge: str, state: str
) -> str:
    identity = resolve_client_identity()
    _validate_antigravity_redirect_uri(redirect_uri)
    query = urlencode({
        "client_id": identity["client_id"],
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(ANTIGRAVITY_SCOPES),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    })
    return f"{GOOGLE_AUTH_URL}?{query}"


def _make_antigravity_callback_handler(expected_path: str):
    result: Dict[str, Any] = {
        "code": None, "state": None, "error": None, "error_description": None,
    }

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found.")
                return
            params = parse_qs(parsed.query)
            result["code"] = params.get("code", [None])[0]
            result["state"] = params.get("state", [None])[0]
            result["error"] = params.get("error", [None])[0]
            result["error_description"] = params.get("error_description", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if result["error"]:
                body = "<html><body><h1>Antigravity authorization failed.</h1>You can close this tab.</body></html>"
            else:
                body = "<html><body><h1>Antigravity authorization received.</h1>You can close this tab.</body></html>"
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A002
            return

    return _Handler, result


def _wait_for_antigravity_callback(
    redirect_uri: str,
    *,
    timeout_seconds: float = _OAUTH_CALLBACK_TIMEOUT_SECONDS,
    on_ready: Any = None,
) -> Dict[str, Any]:
    """Block until the browser redirects back, or time out.

    Bounded by ``timeout_seconds`` (clamped to at least 5s so a caller
    passing 0 can't spin forever) — mirrors the Spotify loopback flow.
    """
    host, port, path = _validate_antigravity_redirect_uri(redirect_uri)
    handler_cls, result = _make_antigravity_callback_handler(path)

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    try:
        server = _ReuseHTTPServer((host, port), handler_cls)
    except OSError as exc:
        raise AntigravityAuthError(
            f"Could not bind Antigravity callback server on {host}:{port}: {exc}",
            code="antigravity_callback_bind_failed",
        ) from None

    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
    )
    thread.start()
    if callable(on_ready):
        try:
            on_ready()
        except Exception as exc:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)
            raise AntigravityAuthError(
                f"Could not start Antigravity authorization: {type(exc).__name__}",
                code="antigravity_authorization_start_failed",
            ) from None
    deadline = time.monotonic() + max(5.0, timeout_seconds)
    try:
        while time.monotonic() < deadline:
            if result["code"] or result["error"]:
                return result
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
    raise AntigravityAuthError(
        "Antigravity authorization timed out waiting for the local callback.",
        code="antigravity_callback_timeout",
    )


def _exchange_code_for_tokens(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
    timeout: float = _TOKEN_HTTP_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    try:
        resp = httpx.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=timeout,
        )
    except Exception as exc:
        raise AntigravityAuthError(
            f"Antigravity token exchange failed: {type(exc).__name__}",
            code="antigravity_exchange_network_error",
        ) from None
    if resp.status_code >= 400:
        raise AntigravityAuthError(
            f"Antigravity token exchange rejected (HTTP {resp.status_code}).",
            code="antigravity_exchange_rejected",
        )
    try:
        payload = resp.json()
    except Exception:
        raise AntigravityAuthError(
            "Antigravity token exchange returned a malformed response.",
            code="antigravity_exchange_malformed",
        ) from None
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise AntigravityAuthError(
            "Antigravity token exchange returned no access token.",
            code="antigravity_exchange_missing_token",
        )
    return payload


def _fetch_userinfo(access_token: str) -> Dict[str, Any]:
    """Best-effort profile lookup — used only to label the credential.

    Never raises: a userinfo failure must not fail the login, since the
    grant itself is already valid at this point.
    """
    try:
        resp = httpx.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        return {"email": data.get("email")} if isinstance(data, dict) else {}
    except Exception:
        return {}


def run_pkce_login(
    *,
    open_browser: bool = True,
    redirect_uri: str = DEFAULT_ANTIGRAVITY_REDIRECT_URI,
    timeout_seconds: float = _OAUTH_CALLBACK_TIMEOUT_SECONDS,
    _state_override: Optional[str] = None,
    _code_verifier_override: Optional[str] = None,
    _client_identity_override: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run the full localhost PKCE login and persist the resulting grant.

    Returns the persisted (in-memory) state dict on success. Every failure
    path raises :class:`AntigravityAuthError`, whose message is scrubbed —
    the authorization code and any token are never echoed back to the
    caller, a log line, or stdout.

    The ``_*_override`` parameters exist solely so tests can pin the PKCE
    values and skip live network/browser calls; production callers should
    never pass them.
    """
    identity = _client_identity_override or resolve_client_identity()
    code_verifier = _code_verifier_override or _pkce_code_verifier()
    code_challenge = _pkce_code_challenge(code_verifier)
    state = _state_override or base64.urlsafe_b64encode(os.urandom(24)).decode("ascii").rstrip("=")

    _validate_antigravity_redirect_uri(redirect_uri)
    authorize_url = build_authorize_url(
        redirect_uri=redirect_uri, code_challenge=code_challenge, state=state,
    ) if not _client_identity_override else None

    def _start_authorization() -> None:
        if not authorize_url:
            return
        if open_browser:
            import webbrowser

            print(f"Waiting for the local callback on {redirect_uri} ...")
            webbrowser.open(authorize_url)
        else:
            # Manual mode must expose the URL so the user can open it, but
            # automatic browser mode keeps client identity/state/challenge
            # out of terminal logs and captured transcripts.
            print(f"Antigravity sign-in URL: {authorize_url}")
            print(f"Waiting for the local callback on {redirect_uri} ...")

    callback = _wait_for_antigravity_callback(
        redirect_uri,
        timeout_seconds=timeout_seconds,
        on_ready=_start_authorization,
    )

    if callback.get("error"):
        # error_description is a Google-controlled string, not a credential,
        # but is still passed through redact_text defensively.
        detail = callback.get("error_description") or ""
        raise AntigravityAuthError(
            f"Antigravity authorization failed: {callback['error']} {detail}".strip(),
            code="antigravity_authorization_denied",
        )
    if callback.get("state") != state:
        raise AntigravityAuthError(
            "Antigravity authorization failed: state mismatch (possible "
            "CSRF or a stale callback). Please retry.",
            code="antigravity_state_mismatch",
        )
    code = callback.get("code")
    if not code:
        raise AntigravityAuthError(
            "Antigravity authorization callback carried no code.",
            code="antigravity_missing_code",
        )

    token_payload = _exchange_code_for_tokens(
        client_id=identity["client_id"],
        client_secret=identity["client_secret"],
        code=code,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
    )

    access_token = token_payload["access_token"]
    userinfo = _fetch_userinfo(access_token)

    try:
        expires_in = float(token_payload.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600.0

    state_dict: Dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": token_payload.get("refresh_token", ""),
        "id_token": token_payload.get("id_token", ""),
        "expires_at": time.time() + expires_in,
        "redirect_uri": redirect_uri,
    }
    state_dict.update(userinfo)

    save_state(state_dict, set_active=False)
    return state_dict


def revoke_and_logout() -> bool:
    """Best-effort remote revoke, then clear source and profile-local state."""
    state, source_path = load_state_with_source()
    token = ""
    if isinstance(state, dict):
        token = str(state.get("refresh_token") or state.get("access_token") or "").strip()
    if token:
        try:
            httpx.post(
                GOOGLE_REVOKE_URL,
                data={"token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10.0,
            )
        except Exception as exc:
            logger.debug("antigravity: revoke failed: %s", type(exc).__name__)

    from hermes_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
        clear_provider_auth,
    )

    source_cleared = False
    if source_path is not None:
        with _auth_store_lock(target_path=source_path):
            source_store = _load_auth_store(source_path)
            providers = source_store.get("providers")
            pool = source_store.get("credential_pool")
            if isinstance(providers, dict) and providers.pop(PROVIDER_ID, None) is not None:
                source_cleared = True
            if isinstance(pool, dict) and pool.pop(PROVIDER_ID, None) is not None:
                source_cleared = True
            if source_store.get("active_provider") == PROVIDER_ID:
                source_store["active_provider"] = None
                source_cleared = True
            if source_cleared:
                _save_auth_store(source_store, target_path=source_path)

    # Remove a profile-local shadow or legacy pool copy outside the source lock.
    active_cleared = clear_provider_auth(PROVIDER_ID)
    return source_cleared or active_cleared


def get_antigravity_auth_status() -> Dict[str, Any]:
    """Return a redacted status payload for `hermes auth status`."""
    state = load_state()
    if not isinstance(state, dict) or not state.get("refresh_token"):
        return {
            "provider": PROVIDER_ID,
            "authenticated": False,
            "logged_in": False,
        }
    safe = redact_state(state)
    safe.update({
        "provider": PROVIDER_ID,
        "authenticated": True,
        "logged_in": True,
        "api_base_url": ANTIGRAVITY_BASE_URL,
        "auth_mode": "oauth_pkce",
    })
    return safe
