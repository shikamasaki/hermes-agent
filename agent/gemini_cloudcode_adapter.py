"""Google Cloud Code / Antigravity HTTP boundary helpers."""

from __future__ import annotations

from typing import Any

from utils import base_url_host_matches


_PROVIDER = "google-antigravity"
_PROJECT_HEADER = "x-goog-user-project"


def is_google_antigravity_agent(agent: Any) -> bool:
    """Return True for the Google Antigravity OpenAI-style route only."""
    return (
        getattr(agent, "api_mode", "") == "chat_completions"
        and str(getattr(agent, "provider", "") or "").strip().lower() == _PROVIDER
    )


def is_cloudcode_base_url(base_url: Any) -> bool:
    """Return True for the Google Cloud Code sandbox endpoint used by Antigravity."""
    return base_url_host_matches(str(base_url or ""), "sandbox.googleapis.com")


def cloudcode_default_headers(project_id: Any) -> dict[str, str]:
    """Headers that bind requests to the refreshed Google Cloud project, if present."""
    project = str(project_id or "").strip()
    return {_PROJECT_HEADER: project} if project else {}


def apply_cloudcode_context(client_kwargs: dict[str, Any], *, project_id: Any) -> None:
    """Merge Cloud Code project headers into OpenAI client kwargs, refreshing project-scoped values."""
    headers = cloudcode_default_headers(project_id)
    if not headers:
        return
    existing = dict(client_kwargs.get("default_headers") or {})
    for key, value in headers.items():
        for existing_key in list(existing):
            if str(existing_key).lower() == key.lower():
                existing.pop(existing_key)
        existing[key] = value
    client_kwargs["default_headers"] = existing
