"""Code Assist (Antigravity subscription) transport for Gemini models.

Google's Code Assist endpoint speaks the *same* ``GenerateContentRequest`` /
``GenerateContentResponse`` schema as the native Gemini API, but:

* it lives at ``daily-cloudcode-pa.sandbox.googleapis.com/v1internal`` instead of
  ``generativelanguage.googleapis.com/v1beta``;
* the method is a POST to ``:generateContent`` / ``:streamGenerateContent``
  on the root (not on ``models/{model}``), with the model name and the
  Code Assist project carried *in the body*;
* the Gemini payload is nested one level down under ``request``, and the
  response is nested under ``response``;
* auth is an OAuth bearer token, not an ``?key=`` API key.

So this module is deliberately thin: it wraps/unwraps that envelope and
delegates every message, tool, and streaming translation to
``agent.gemini_native_adapter``. That keeps tool-call IDs, thought
signatures, message alternation, and prompt caching behaving identically
to the API-key Gemini path.
"""

from __future__ import annotations

import json
import logging
import platform
import uuid
from typing import Any, Dict, List, Optional

from agent.gemini_native_adapter import (
    GeminiNativeClient,
    build_gemini_request,
    translate_gemini_response,
    translate_stream_event,
)

logger = logging.getLogger(__name__)

CODE_ASSIST_BASE_URL = "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal"

#: Only Gemini models are surfaced by this provider.
_ENTITLED_MODEL_PREFIX = "gemini-"
_CLIENT_METADATA = {
    "ideType": "ANTIGRAVITY",
    "platform": "PLATFORM_UNSPECIFIED",
    "pluginType": "GEMINI",
}


def build_antigravity_headers(
    access_token: str, *, accept: str = "application/json"
) -> Dict[str, str]:
    """Build provider headers without logging or persisting credentials."""
    system = platform.system().lower() or "unknown"
    machine = platform.machine().lower() or "unknown"
    return {
        "Content-Type": "application/json",
        "Accept": accept,
        "Authorization": f"Bearer {access_token}",
        "User-Agent": f"antigravity/1.0.0 {system}/{machine}",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps(_CLIENT_METADATA, separators=(",", ":")),
        "x-activity-request-id": str(uuid.uuid4()),
    }


def resolve_project_context(
    *,
    access_token: str,
    configured_project: Optional[str] = None,
    base_url: str = CODE_ASSIST_BASE_URL,
    http_client: Any = None,
    timeout: float = 8.0,
) -> Optional[str]:
    """Return an explicit project or discover the managed Code Assist project.

    Discovery is intentionally fail-closed: response bodies are never logged or
    copied into exceptions because the service may reflect request metadata.
    """
    project = str(configured_project or "").strip()
    if project:
        return project

    token = str(access_token or "").strip()
    if not token:
        return None

    import httpx

    client = http_client or httpx
    try:
        response = client.post(
            f"{str(base_url or CODE_ASSIST_BASE_URL).rstrip('/')}:loadCodeAssist",
            headers=build_antigravity_headers(token),
            json={"metadata": {"pluginType": "GEMINI"}},
            timeout=timeout,
        )
        if response.status_code >= 400:
            logger.debug("loadCodeAssist failed: HTTP %s", response.status_code)
            return None
        payload = response.json()
    except Exception as exc:
        logger.debug("loadCodeAssist error: %s", type(exc).__name__)
        return None

    if not isinstance(payload, dict):
        return None
    return str(
        payload.get("cloudaicompanionProject") or payload.get("project") or ""
    ).strip() or None


def build_code_assist_request(
    *,
    model: str,
    gemini_request: Dict[str, Any],
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap a native Gemini request body in the Code Assist envelope."""
    envelope: Dict[str, Any] = {
        "model": model,
        "request": gemini_request,
    }
    if project:
        envelope["project"] = project
    return envelope


def unwrap_code_assist_response(raw: Any) -> Dict[str, Any]:
    """Strip the ``response`` envelope, tolerating already-bare payloads."""
    if isinstance(raw, dict):
        inner = raw.get("response")
        if isinstance(inner, dict):
            return inner
        return raw
    return {}


#: Streaming events carry the same envelope, one chunk at a time.
unwrap_code_assist_stream_event = unwrap_code_assist_response


def translate_code_assist_response(raw: Dict[str, Any], model: str) -> Any:
    """Unwrap then reuse the native Gemini response translation.

    Delegating means tool calls keep the ids, argument encoding, and
    thought-signature round-tripping that Hermes' tool loop already relies
    on — there is no second, divergent implementation to keep in sync.
    """
    return translate_gemini_response(unwrap_code_assist_response(raw), model)


def translate_code_assist_stream_event(
    event: Dict[str, Any],
    model: str,
    tool_call_indices: Dict[str, Dict[str, Any]],
) -> List[Any]:
    """Unwrap a streamed Code Assist event and translate it as Gemini."""
    return translate_stream_event(
        unwrap_code_assist_stream_event(event), model, tool_call_indices
    )


def parse_entitled_models(payload: Any) -> List[str]:
    """Extract deduplicated Gemini model ids from Code Assist catalogs."""
    if not isinstance(payload, dict):
        return []

    candidates: List[Any] = []
    default_model = payload.get("defaultAgentModelId")
    if default_model is not None:
        candidates.append(default_model)

    allowed = payload.get("allowedModels")
    if isinstance(allowed, list):
        candidates.extend(allowed)

    sorts = payload.get("agentModelSorts")
    if isinstance(sorts, list):
        recommended = [
            item
            for item in sorts
            if isinstance(item, dict)
            and "recommended"
            in " ".join(
                str(item.get(key) or "")
                for key in ("name", "displayName", "title", "category", "group")
            ).lower()
        ]
        rest = [
            item
            for item in sorts
            if isinstance(item, dict) and item not in recommended
        ]
        for sort in recommended + rest:
            for key in ("modelIds", "model_ids", "models", "modelSorts"):
                value = sort.get(key)
                if isinstance(value, list):
                    candidates.extend(value)
                elif value is not None:
                    candidates.append(value)
            groups = sort.get("groups")
            if isinstance(groups, list):
                for group in groups:
                    if not isinstance(group, dict):
                        continue
                    for key in ("modelIds", "model_ids", "models"):
                        value = group.get(key)
                        if isinstance(value, list):
                            candidates.extend(value)
                        elif value is not None:
                            candidates.append(value)

    model_catalog = payload.get("models")
    if isinstance(model_catalog, dict):
        candidates.extend(model_catalog.keys())
    elif not candidates and isinstance(model_catalog, list):
        candidates.extend(model_catalog)

    models: List[str] = []
    for entry in candidates:
        if isinstance(entry, dict):
            model_id = (
                entry.get("modelId")
                or entry.get("model_id")
                or entry.get("model")
                or entry.get("id")
                or entry.get("name")
            )
        elif isinstance(entry, str):
            model_id = entry
        else:
            continue
        model_id = str(model_id or "").strip()
        if not model_id or not model_id.lower().startswith(_ENTITLED_MODEL_PREFIX):
            continue
        if model_id not in models:
            models.append(model_id)
    return models


def discover_entitled_models(
    *,
    access_token: str,
    project: Optional[str] = None,
    base_url: str = CODE_ASSIST_BASE_URL,
    timeout: float = 8.0,
    http_client: Any = None,
) -> Optional[List[str]]:
    """Fetch the subscription's Gemini-only agent model catalog."""
    token = (access_token or "").strip()
    if not token:
        return None

    import httpx

    client = http_client or httpx
    if not project:
        project = resolve_project_context(
            access_token=token,
            configured_project=None,
            base_url=base_url,
            http_client=client,
            timeout=timeout,
        )

    url = f"{str(base_url or CODE_ASSIST_BASE_URL).rstrip('/')}:fetchAvailableModels"
    body: Dict[str, Any] = {}
    if project:
        body["project"] = project

    try:
        resp = client.post(
            url,
            headers=build_antigravity_headers(token),
            json=body,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            logger.debug("fetchAvailableModels failed: HTTP %s", resp.status_code)
            return None
        models = parse_entitled_models(resp.json())
    except Exception as exc:
        logger.debug("fetchAvailableModels error: %s", type(exc).__name__)
        return None
    return models or None


def _safe_code_assist_http_error(response: Any) -> Exception:
    """Return a retry-classifiable error without retaining provider payloads."""
    from agent.gemini_native_adapter import GeminiAPIError

    status = int(getattr(response, "status_code", 0) or 0)
    retry_after = None
    try:
        retry_after = float(response.headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        pass
    return GeminiAPIError(
        f"Antigravity Code Assist returned HTTP {status or 'error'}.",
        code=f"antigravity_http_{status or 'error'}",
        status_code=status or None,
        response=None,
        retry_after=retry_after,
        details={},
    )


class CodeAssistClient(GeminiNativeClient):
    """OpenAI-shaped client speaking Code Assist with an OAuth bearer token.

    Subclasses the native Gemini client so the request building, tool-call
    translation, streaming, and error mapping are literally the same code.
    Only three things differ and are overridden here: the auth header (bearer
    instead of ``?key=``), the URL shape (root-level method, model in body),
    and the request/response envelope.
    """

    def __init__(self, *args: Any, project: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._code_assist_project = project

    def _headers(self) -> Dict[str, str]:
        headers = build_antigravity_headers(self.api_key)
        headers.update(getattr(self, "_default_headers", {}) or {})
        return headers

    @property
    def _project(self) -> Optional[str]:
        return getattr(self, "_code_assist_project", None) or None

    def _ensure_project(self) -> Optional[str]:
        if self._project:
            return self._project
        self._code_assist_project = resolve_project_context(
            access_token=self.api_key,
            configured_project=None,
            base_url=self.base_url,
            http_client=self._http,
        )
        return self._project

    def _create_chat_completion(self, *, model: str = "gemini-3-pro-preview", stream: bool = False, timeout: Any = None, **kwargs: Any) -> Any:
        """Build the Gemini request via the base class, then post it wrapped."""
        from agent.gemini_native_adapter import (
            GeminiAPIError,
            bare_gemini_model_id,
        )

        extra_body = kwargs.get("extra_body")
        thinking_config = None
        if isinstance(extra_body, dict):
            thinking_config = extra_body.get("thinking_config") or extra_body.get("thinkingConfig")

        model = bare_gemini_model_id(model)
        request = build_gemini_request(
            messages=kwargs.get("messages") or [],
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            stop=kwargs.get("stop"),
            thinking_config=thinking_config,
            model=model,
        )
        envelope = build_code_assist_request(
            model=model, gemini_request=request, project=self._ensure_project()
        )

        if stream:
            return self._stream_completion(model=model, request=envelope, timeout=timeout)

        url = f"{self.base_url.rstrip('/')}:generateContent"
        response = self._http.post(
            url, json=envelope, headers=self._headers(), timeout=timeout
        )
        if response.status_code != 200:
            raise _safe_code_assist_http_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeminiAPIError(
                f"Invalid JSON from Code Assist API: {exc}",
                code="antigravity_invalid_json",
                status_code=response.status_code,
                response=None,
            ) from exc
        return translate_code_assist_response(payload, model)

    def _stream_completion(self, *, model: str, request: Dict[str, Any], timeout: Any = None):
        """Stream Code Assist SSE, unwrapping each event before translation."""
        import httpx as _httpx

        from agent.bounded_response import read_streaming_error_body
        from agent.gemini_native_adapter import (
            GeminiAPIError,
            _iter_sse_events,
        )

        url = f"{self.base_url.rstrip('/')}:streamGenerateContent?alt=sse"
        stream_headers = dict(self._headers())
        stream_headers["Accept"] = "text/event-stream"

        def _generator():
            transport_error = None
            try:
                with self._http.stream(
                    "POST", url, json=request, headers=stream_headers, timeout=timeout
                ) as response:
                    if response.status_code != 200:
                        read_streaming_error_body(response)
                        raise _safe_code_assist_http_error(response)
                    tool_call_indices: Dict[str, Dict[str, Any]] = {}
                    for event in _iter_sse_events(response):
                        for chunk in translate_code_assist_stream_event(
                            event, model, tool_call_indices
                        ):
                            yield chunk
            except _httpx.HTTPError:
                transport_error = GeminiAPIError(
                    "Code Assist streaming request failed.",
                    code="antigravity_stream_error",
                )
            if transport_error is not None:
                # Raise after leaving the handler so neither __cause__ nor
                # __context__ retains an httpx.Request with bearer headers.
                raise transport_error from None

        return _generator()

    def _unwrap_response(self, payload: Any) -> Dict[str, Any]:
        return unwrap_code_assist_response(payload)


__all__ = [
    "CODE_ASSIST_BASE_URL",
    "CodeAssistClient",
    "discover_entitled_models",
    "build_code_assist_request",
    "build_gemini_request",
    "parse_entitled_models",
    "translate_code_assist_response",
    "translate_code_assist_stream_event",
    "unwrap_code_assist_response",
    "unwrap_code_assist_stream_event",
]
