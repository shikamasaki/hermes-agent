"""``hermes usage`` — read-only usage listing for every configured provider.

Discovers which providers are configured (main model, delegation routes,
and auxiliary task assignments),
then renders whatever :mod:`agent.delegation_usage_cache` already has
cached for each — never a live provider call on this path. ``--refresh``
opts into one bounded, synchronous ``refresh_provider_now`` call per
selected provider before rendering; it does not touch the cache's
background-refresh scheduling.

Every field rendered here already passed through
``agent.delegation_usage_cache.project_snapshot``'s allowlist, so this
module does not need its own redaction pass — it must simply never add a
field back in from a raw snapshot, error, or config value (a base URL, a
credential path, an account/org identity, raw exception text). Unknown or
unavailable is always an explicit status, never an error string.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Iterable, Mapping, Optional

from agent import delegation_usage_cache
from agent.delegation_routing import (
    DEFAULT_USAGE_STALE_SECONDS,
    DEFAULT_USAGE_TTL_SECONDS,
    load_route_catalog,
)


__all__ = [
    "discover_configured_providers",
    "build_usage_rows",
    "render_text",
    "render_json",
    "run_usage_command",
]

_PLACEHOLDER_PROVIDERS = frozenset({"", "auto", "custom"})


def _load_active_config() -> Mapping[str, Any]:
    from hermes_cli.config import load_config_readonly

    return load_config_readonly()


def _normalize(provider: Optional[str]) -> str:
    return delegation_usage_cache.normalize_provider(provider)


def _is_real_provider(provider: str) -> bool:
    return bool(provider) and provider not in _PLACEHOLDER_PROVIDERS


def discover_configured_providers(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Generically discover every provider slug named by active config.

    Sources, in order: ``model.provider``, ``delegation.routes[].provider``
    (via the same validated catalog parser routing uses), and every ``auxiliary.
    <task>.provider`` assignment. Placeholders (``""``/``"auto"``/
    ``"custom"``) are dropped; aliases are normalized so the same account
    never appears twice under different spellings. Order is stable
    (first-seen) for readable, deterministic output.
    """
    cfg = config if isinstance(config, Mapping) else {}
    seen: list[str] = []

    def _add(raw: Any) -> None:
        provider = _normalize(str(raw) if raw is not None else "")
        if _is_real_provider(provider) and provider not in seen:
            seen.append(provider)

    model_cfg = cfg.get("model")
    if isinstance(model_cfg, Mapping):
        _add(model_cfg.get("provider"))

    delegation_cfg = cfg.get("delegation")
    if isinstance(delegation_cfg, Mapping):
        # Preserve legacy delegation.provider/model configurations when no
        # route catalog is enabled. The catalog path below remains the source
        # for modern per-route providers; _add() deduplicates when both exist.
        _add(delegation_cfg.get("provider"))
        try:
            catalog = load_route_catalog(delegation_cfg)
        except Exception:
            # A malformed delegation block is a config-validation concern
            # handled elsewhere; usage discovery degrades to "no routes"
            # rather than raising out of a read-only reporting command.
            catalog = None
        if catalog is not None:
            for route in catalog.routes:
                _add(route.provider)


    auxiliary_cfg = cfg.get("auxiliary")
    if isinstance(auxiliary_cfg, Mapping):
        for task_cfg in auxiliary_cfg.values():
            if isinstance(task_cfg, Mapping) and "provider" in task_cfg:
                _add(task_cfg.get("provider"))

    return tuple(seen)


# ---------------------------------------------------------------------------
# Cache rendering
# ---------------------------------------------------------------------------


def build_usage_rows(
    providers: Iterable[str],
    *,
    refresh: bool,
    ttl_seconds: int = DEFAULT_USAGE_TTL_SECONDS,
    stale_seconds: int = DEFAULT_USAGE_STALE_SECONDS,
    provider_filter: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """Render one safe row per selected, deduped provider.

    ``refresh=True`` performs a bounded synchronous
    ``refresh_provider_now`` call per selected provider (never the
    background-thread scheduling path) before reading the cache, so the
    render reflects the refresh just requested.
    """
    normalized = []
    seen: set[str] = set()
    for provider in providers:
        p = _normalize(provider)
        if _is_real_provider(p) and p not in seen:
            seen.add(p)
            normalized.append(p)

    if provider_filter is not None:
        allowed = {_normalize(p) for p in provider_filter}
        normalized = [p for p in normalized if p in allowed]

    if refresh:
        for provider in normalized:
            delegation_usage_cache.refresh_provider_now(provider)

    rows: list[dict[str, Any]] = []
    raw_providers = delegation_usage_cache.read_raw().get("providers") or {}
    for provider in normalized:
        usage = delegation_usage_cache.read_provider_usage(
            provider, ttl_seconds=ttl_seconds, stale_seconds=stale_seconds
        )
        raw_record = raw_providers.get(provider)
        windows: list[dict[str, Any]] = []
        source = ""
        if usage.freshness in ("fresh", "stale") and isinstance(raw_record, dict):
            source = str(raw_record.get("source") or "")
            for window in raw_record.get("windows") or []:
                if not isinstance(window, dict):
                    continue
                windows.append(
                    {
                        "label": window.get("label") or "",
                        "used_percent": window.get("used_percent"),
                        "remaining_percent": window.get("remaining_percent"),
                        "reset_at": window.get("reset_at"),
                    }
                )
        status = "available" if usage.freshness in ("fresh", "stale") and windows else "unknown"
        rows.append(
            {
                "provider": provider,
                "status": status,
                "freshness": usage.freshness,
                "age_seconds": usage.age_seconds,
                "remaining_percent": usage.remaining_percent,
                "source": source,
                "windows": windows,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_age(age_seconds: Optional[float]) -> str:
    if age_seconds is None:
        return "unknown"
    age = int(age_seconds)
    if age < 60:
        return f"{age}s"
    if age < 3600:
        return f"{age // 60}m"
    return f"{age // 3600}h"


def render_text(providers: Iterable[str], *, refresh: bool, provider_filter=None) -> str:
    rows = build_usage_rows(providers, refresh=refresh, provider_filter=provider_filter)
    if not rows:
        return "No configured providers with usage data."
    lines = []
    for row in rows:
        source_str = f" [source: {row['source']}]" if row.get("source") else ""
        lines.append(f"{row['provider']}: {row['status']} ({row['freshness']}, age={_format_age(row['age_seconds'])}){source_str}")
        if not row["windows"]:
            lines.append("  no cached usage windows")
            continue
        for window in row["windows"]:
            label = window["label"] or "(window)"
            remaining = window.get("remaining_percent")
            used = window.get("used_percent")

            rem_str = f"{remaining:.1f}% remaining" if remaining is not None else "unknown remaining"
            used_str = f"{used:.1f}% used" if used is not None else "unknown used"

            reset_at = window.get("reset_at") or "unknown"
            lines.append(f"  {label}: {rem_str} / {used_str}, reset_at={reset_at}")
    return "\n".join(lines)


def render_json(providers: Iterable[str], *, refresh: bool, provider_filter=None) -> str:
    rows = build_usage_rows(providers, refresh=refresh, provider_filter=provider_filter)
    return json.dumps({"providers": rows}, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI adapter
# ---------------------------------------------------------------------------


def run_usage_command(args: argparse.Namespace) -> int:
    """Argparse-facing entry point. Returns a process exit code."""
    config = _load_active_config()
    providers = discover_configured_providers(config)
    provider_filter = getattr(args, "provider", None)
    as_json = bool(getattr(args, "json", False))
    refresh = bool(getattr(args, "refresh", False))

    if not providers:
        if as_json:
            print(json.dumps({"providers": []}, indent=2, sort_keys=True))
        else:
            print("No configured providers found.")
        return 0

    if as_json:
        print(render_json(providers, refresh=refresh, provider_filter=provider_filter))
    else:
        print(render_text(providers, refresh=refresh, provider_filter=provider_filter))
    return 0
