"""External execution backend: ``claude -p`` (Claude Code print mode).

This is the ONLY external-CLI delegation backend Hermes supports, and it
exists for exactly one reason: to let a delegated task run against the
user's Claude Pro/Max *subscription* rather than an Anthropic API key. It is
deliberately narrow — no ``agy -p``, no ``codex exec``, no OpenCode, no
arbitrary shell-configured command, no substituting an API-key provider for
what should be a subscription call.

Hard safety rules enforced throughout this module:

* The child process is spawned with an argv list via
  ``asyncio.create_subprocess_exec`` — never ``shell=True``, and the task
  prompt is never interpolated into a shell string.
* The executable is resolved once via ``shutil.which("claude")``; there is
  no config knob to point at an arbitrary path in this revision.
* ``--bare`` is never passed (it skips OAuth and forces API-key auth,
  defeating the entire point of this backend). ``--dangerously-skip-permissions``
  is passed only when the caller captured an explicit validated
  ``delegation.subagent_auto_approve=true`` decision. Plugins, MCP overrides,
  browser integration, and push/publish/deploy/PR commands are never passed.
* The child's environment is a fresh, minimal projection of the parent's —
  every ``*_API_KEY``/``*_TOKEN``/cloud-credential variable is stripped so a
  parent's (or a native provider's) credentials can never leak into the
  Claude child, forcing it onto subscription auth.
* The subscription-auth probe (``claude auth status --json``) is projected
  down to booleans only — no email, org, account id, token, or raw JSON ever
  leaves this module.
* Only the FINAL JSON object on stdout is parsed; raw stderr, auth output,
  environment, and the full command line (which contains the task body) are
  never logged or returned.
"""

from __future__ import annotations

import asyncio
import json
import codecs
import logging
import math
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "BACKEND_ID",
    "ClaudeAvailability",
    "ClaudePToolProfile",
    "ClaudePRunRequest",
    "ClaudePRunResult",
    "TOOL_PROFILES",
    "build_claude_p_argv",
    "build_scrubbed_environment",
    "check_claude_availability",
    "is_route_in_cooldown",
    "note_route_failure",
    "normalize_claude_p_output",
    "normalize_claude_p_stream",
    "resolve_claude_executable",
    "run_claude_p_task",
    "workdir_lock_for",
]

BACKEND_ID = "claude-p"

# ---------------------------------------------------------------------------
# Executable resolution — fixed to `claude`, no config override in this PR.
# ---------------------------------------------------------------------------


def resolve_claude_executable() -> Optional[str]:
    """Resolve the ``claude`` CLI from PATH. Returns None if not installed."""
    return shutil.which("claude")


# ---------------------------------------------------------------------------
# Tool profiles — fixed, least-privilege allowlists. Never an arbitrary string.
# ---------------------------------------------------------------------------

CODING_BASH_ALLOWED_TOOLS: tuple[str, ...] = (
    "Bash(git status)",
    "Bash(git status *)",
    "Bash(git diff)",
    "Bash(git diff *)",
    "Bash(git log)",
    "Bash(git log *)",
    "Bash(git show)",
    "Bash(git show *)",
    "Bash(uv run pytest)",
    "Bash(uv run pytest *)",
    "Bash(uv run mypy)",
    "Bash(uv run mypy *)",
    "Bash(uv run ruff)",
    "Bash(uv run ruff *)",
    "Bash(pytest)",
    "Bash(pytest *)",
    "Bash(python -m pytest)",
    "Bash(python -m pytest *)",
    "Bash(scripts/run_tests.sh)",
    "Bash(scripts/run_tests.sh *)",
    "Bash(venv/bin/ruff check *)",
    "Bash(venv/bin/python -m compileall *)",
    "Bash(npm test)",
    "Bash(npm run test)",
    "Bash(npm run lint)",
    "Bash(npm run typecheck)",
    "Bash(npm run build)",
    "Bash(pnpm test)",
    "Bash(pnpm run test)",
    "Bash(pnpm run lint)",
    "Bash(pnpm run typecheck)",
    "Bash(pnpm run build)",
    "Bash(yarn test)",
    "Bash(yarn run test)",
    "Bash(yarn run lint)",
    "Bash(yarn run typecheck)",
    "Bash(yarn run build)",
    "Bash(cargo test)",
    "Bash(cargo test *)",
    "Bash(cargo check)",
    "Bash(cargo check *)",
    "Bash(cargo clippy)",
    "Bash(cargo clippy *)",
    "Bash(go test)",
    "Bash(go test *)",
    "Bash(make test)",
    "Bash(make lint)",
    "Bash(make check)",
)

TOOL_PROFILES: Mapping[str, Optional[str]] = {
    # Claude Code normal Task passthrough.  No Hermes-added CLI tool/settings
    # restrictions are emitted for this profile; Claude's own defaults apply.
    "default": None,
    "read_only": "Read(./**)",
    "review": "Read(./**)",
    "coding": ",".join(
        ("Read(./**)", "Edit(./**)", "Write(./**)", *CODING_BASH_ALLOWED_TOOLS)
    ),
}

PROFILE_TOOL_SETS: Mapping[str, Optional[str]] = {
    "default": None,
    "read_only": "Read",
    "review": "Read",
    "coding": "Read,Edit,Write,Bash",
}

# Fixed negative rules provide defense in depth for the coding profile's
# least-privilege Bash rules while read_only/review expose no shell tool.
DISALLOWED_TOOLS: tuple[str, ...] = (
    "Bash(git commit *)",
    "Bash(git push *)",
    "Bash(gh pr *)",
    "Bash(gh release *)",
    "Bash(npm publish *)",
    "Bash(curl *)",
    "Bash(wget *)",
    "Bash(ssh *)",
    "Bash(scp *)",
    "Bash(rsync *)",
    "Bash(aws *)",
    "Bash(gcloud *)",
    "Bash(kubectl *)",
    "Bash(terraform apply *)",
    "Bash(vercel *)",
)

DEFAULT_TOOL_PROFILE = "read_only"


@dataclass(frozen=True)
class ClaudePToolProfile:
    """Validated tool profile resolved from route config."""

    name: str
    allowed_tools: Optional[str]
    tools: Optional[str]

    @property
    def passthrough(self) -> bool:
        return self.name == "default"


def resolve_tool_profile(name: Optional[str]) -> ClaudePToolProfile:
    """Resolve a profile name to its fixed allowlist string.

    Defaults to ``read_only`` when unset. Raises ``ValueError`` for an
    unknown profile name rather than silently falling back — a typo'd
    profile must not silently grant a different privilege level.
    """
    key = (name or DEFAULT_TOOL_PROFILE).strip().lower()
    if key not in TOOL_PROFILES:
        valid = ", ".join(sorted(TOOL_PROFILES))
        raise ValueError(f"unknown claude-p tool profile {name!r} (expected one of: {valid})")
    return ClaudePToolProfile(
        name=key,
        allowed_tools=TOOL_PROFILES[key],
        tools=PROFILE_TOOL_SETS[key],
    )


# ---------------------------------------------------------------------------
# Difficulty -> --effort mapping
# ---------------------------------------------------------------------------

_DIFFICULTY_TO_EFFORT: Mapping[str, str] = {
    "routine": "low",
    "standard": "medium",
    "complex": "high",
    "frontier": "max",
}


def map_difficulty_to_effort(difficulty: str) -> str:
    return _DIFFICULTY_TO_EFFORT.get((difficulty or "").strip().lower(), "medium")


# ---------------------------------------------------------------------------
# Bounded execution parameters
# ---------------------------------------------------------------------------

#: Hard ceilings — a route/task may only request values at or below these.
MAX_TURNS_CEILING = 200
MAX_BUDGET_USD_CEILING = 50.0
MAX_TIMEOUT_SECONDS_CEILING = 3600
MAX_STDOUT_BYTES = 8 * 1024 * 1024
MAX_STDERR_BYTES = 256 * 1024
MAX_STREAM_EVENT_BYTES = 512 * 1024
PROCESS_TERMINATE_GRACE_SECONDS = 5.0
PROCESS_KILL_GRACE_SECONDS = 5.0
PIPE_DRAIN_GRACE_SECONDS = 5.0

DEFAULT_MAX_TURNS = 40
DEFAULT_MAX_BUDGET_USD = 5.0
DEFAULT_TIMEOUT_SECONDS = 900

# Rate-limit/overload/billing cooldown when startup fails before any tool ran.
DEFAULT_COOLDOWN_SECONDS = 60


def _bounded_int(value: Any, *, default: int, ceiling: int, floor: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(floor, min(ceiling, parsed))


def _bounded_float(value: Any, *, default: float, ceiling: float, floor: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed or parsed in (float("inf"), float("-inf")):  # NaN/inf guard
        return default
    return max(floor, min(ceiling, parsed))


# ---------------------------------------------------------------------------
# Environment scrubbing
# ---------------------------------------------------------------------------

#: Strict allowlist for the child process.  A denylist is insufficient here:
#: variables such as ``AWS_PROFILE``, ``GOOGLE_CLOUD_PROJECT``, custom provider
#: headers, or an askpass helper can carry account/credential context without
#: containing ``KEY`` or ``TOKEN`` in their name.
_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LANGUAGE",
        "TZ",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SHELL",
        "TERM",
        # Windows equivalents required to find the CLI and its OAuth state.
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "COMSPEC",
        "PATHEXT",
    }
)


def build_scrubbed_environment(
    base_env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Build a minimal allowlisted environment for the ``claude`` child.

    Only process basics needed to locate the executable, locale, temporary
    directory, and Claude's own subscription OAuth state survive.  Provider
    credentials, cloud profiles/projects, account identifiers, custom headers,
    git/SSH askpass helpers, and all unrelated variables are omitted even when
    their names do not look secret.  Git prompting is disabled explicitly.
    """
    source = dict(base_env) if base_env is not None else dict(os.environ)
    scrubbed = {
        name: source[name]
        for name in _ENV_ALLOWLIST
        if name in source and isinstance(source[name], str)
    }
    scrubbed["GIT_TERMINAL_PROMPT"] = "0"
    scrubbed["GCM_INTERACTIVE"] = "never"
    scrubbed["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] = "1"
    return scrubbed


# ---------------------------------------------------------------------------
# Subscription availability probe — booleans only, never raw identity/JSON.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaudeAvailability:
    """Secret-free projection of ``claude auth status --json``.

    Carries no email, organization, account id, token, or credential path —
    only what the selector needs to decide whether this route can compete.
    """

    installed: bool = False
    authenticated: bool = False
    auth_method_class: str = "unknown"  # "subscription" | "api_key" | "unknown"
    checked_at: Optional[float] = None

    @property
    def available(self) -> bool:
        return (
            self.installed
            and self.authenticated
            and self.auth_method_class == "subscription"
        )


def _project_auth_status(raw: Any) -> ClaudeAvailability:
    if not isinstance(raw, dict):
        return ClaudeAvailability(installed=True, authenticated=False)
    authenticated = bool(raw.get("authenticated") or raw.get("loggedIn") or raw.get("logged_in"))
    method = str(raw.get("authMethod") or raw.get("auth_method") or "").strip().lower()
    subscription_type = str(
        raw.get("subscriptionType") or raw.get("subscription_type") or ""
    ).strip().lower()
    api_provider = str(raw.get("apiProvider") or raw.get("api_provider") or "").strip().lower()
    api_methods = {"api", "apikey", "api-key", "api_key"}
    api_providers = {"api", "anthropic", "thirdparty", "third-party", "third_party"}
    if method in api_methods or api_provider in api_providers:
        method_class = "api_key"
    elif method in {"subscription", "claude.ai"} or subscription_type in {
        "pro",
        "max",
    }:
        method_class = "subscription"
    else:
        method_class = "unknown"
    return ClaudeAvailability(
        installed=True,
        authenticated=authenticated,
        auth_method_class=method_class,
    )


def check_claude_availability(*, timeout: float = 10.0) -> ClaudeAvailability:
    """Bounded probe of ``claude auth status --json``. Never raises.

    Only booleans/auth-method class survive into the return value — raw
    stdout/stderr, email, org, account id, token, and credential path are
    read and immediately discarded.
    """
    executable = resolve_claude_executable()
    if not executable:
        return ClaudeAvailability(installed=False, authenticated=False, checked_at=time.time())

    from hermes_cli._subprocess_compat import bounded_probe_run

    result = bounded_probe_run(
        [executable, "auth", "status", "--json"],
        timeout=timeout,
        env=build_scrubbed_environment(),
    )
    checked_at = time.time()
    if result is None or result.returncode != 0:
        return ClaudeAvailability(
            installed=True, authenticated=False, checked_at=checked_at
        )
    try:
        raw = json.loads(result.stdout or "")
    except ValueError:
        return ClaudeAvailability(installed=True, authenticated=False, checked_at=checked_at)
    projected = _project_auth_status(raw)
    return ClaudeAvailability(
        installed=projected.installed,
        authenticated=projected.authenticated,
        auth_method_class=projected.auth_method_class,
        checked_at=checked_at,
    )


# ---------------------------------------------------------------------------
# Per-route startup cooldown (rate-limit/overload/billing pre-start failures)
# ---------------------------------------------------------------------------

_cooldown_lock = threading.Lock()
_cooldown_until: dict[str, float] = {}


def note_route_failure(route_id: str, *, retry_after_seconds: Optional[float] = None) -> None:
    """Place *route_id* into a bounded cooldown after a pre-start failure.

    Only meant for failures that happen BEFORE any tool call runs (rate
    limit / overload / billing / nonzero-exit-with-no-progress) — never
    call this once a write-capable task has started modifying files.
    """
    duration = retry_after_seconds if (retry_after_seconds and retry_after_seconds > 0) else DEFAULT_COOLDOWN_SECONDS
    duration = min(duration, 3600.0)
    with _cooldown_lock:
        _cooldown_until[route_id] = time.time() + duration


def is_route_in_cooldown(route_id: str) -> bool:
    with _cooldown_lock:
        until = _cooldown_until.get(route_id)
        if until is None:
            return False
        if time.time() >= until:
            _cooldown_until.pop(route_id, None)
            return False
        return True


def reset_cooldowns() -> None:
    """Test seam."""
    with _cooldown_lock:
        _cooldown_until.clear()


# ---------------------------------------------------------------------------
# Per-workdir serialization — write-capable tasks never race on one workdir.
# ---------------------------------------------------------------------------

_workdir_locks_guard = threading.Lock()
_workdir_locks: dict[str, threading.Lock] = {}


def workdir_lock_for(workdir: str) -> threading.Lock:
    """Return the process-wide lock for *workdir* (created on first use)."""
    key = os.path.abspath(workdir) if workdir else ""
    with _workdir_locks_guard:
        lock = _workdir_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _workdir_locks[key] = lock
        return lock


def reset_workdir_locks() -> None:
    """Test seam."""
    with _workdir_locks_guard:
        _workdir_locks.clear()


def _create_restricted_settings(profile: str, workdir: str) -> tuple[str, str]:
    """Create a private one-run PreToolUse policy settings file."""
    directory = tempfile.mkdtemp(prefix="hermes-claude-p-")
    os.chmod(directory, 0o700)
    settings_path = os.path.join(directory, "settings.json")
    gate_script = str(Path(__file__).with_name("claude_p_tool_gate.py").resolve())
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": sys.executable,
                            "args": [
                                gate_script,
                                "--profile",
                                profile,
                                "--workdir",
                                os.path.abspath(workdir),
                            ],
                        }
                    ],
                }
            ]
        }
    }
    fd = os.open(settings_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, json.dumps(settings, separators=(",", ":")).encode("utf-8"))
    finally:
        os.close(fd)
    return settings_path, directory


# ---------------------------------------------------------------------------
# Argv construction — pure, shell-free, one argv element per logical field.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaudePRunRequest:
    """Everything needed to build and run one bounded ``claude -p`` task."""

    prompt: str
    model: str
    difficulty: str = "standard"
    workdir: str = "."
    tool_profile: str = DEFAULT_TOOL_PROFILE
    max_turns: Optional[int] = DEFAULT_MAX_TURNS
    max_budget_usd: Optional[float] = DEFAULT_MAX_BUDGET_USD
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resume_session_id: Optional[str] = None
    auto_approve: bool = False
    settings_path: Optional[str] = None


def build_claude_p_argv(request: ClaudePRunRequest, *, executable: str) -> list[str]:
    """Build the exact, shell-free argv for a ``claude -p`` invocation.

    The prompt is exactly one argv element — never split, never
    interpolated into a shell string. Never includes ``--bare``,
    ``--permission-mode bypassPermissions``, plugin/MCP overrides, browser
    integration, or push/publish/deploy/PR commands. Full permission bypass is
    confined to the explicit ``default`` passthrough profile. Restricted
    profiles use ``dontAsk`` plus a fixed allowlist, so unmatched tools fail
    closed instead of waiting for an interactive approval.
    """
    profile = resolve_tool_profile(request.tool_profile)
    effort = map_difficulty_to_effort(request.difficulty)

    argv = [
        executable,
        "-p",
        request.prompt,
        "--model",
        request.model,
        "--effort",
        effort,
    ]
    if request.max_turns is not None:
        max_turns = _bounded_int(
            request.max_turns, default=DEFAULT_MAX_TURNS, ceiling=MAX_TURNS_CEILING
        )
        argv += ["--max-turns", str(max_turns)]
    if request.max_budget_usd is not None:
        max_budget = _bounded_float(
            request.max_budget_usd,
            default=DEFAULT_MAX_BUDGET_USD,
            ceiling=MAX_BUDGET_USD_CEILING,
            floor=0.01,
        )
        argv += ["--max-budget-usd", f"{max_budget:.2f}"]
    if profile.passthrough:
        if request.auto_approve:
            argv.append("--dangerously-skip-permissions")
    else:
        argv += [
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            profile.allowed_tools or "",
            "--tools",
            profile.tools or "",
            "--disallowedTools",
            *DISALLOWED_TOOLS,
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
        ]
        if request.settings_path:
            argv += ["--settings", request.settings_path]
    argv += [
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--include-hook-events",
    ]
    if request.resume_session_id:
        resume_id = _safe_uuid(request.resume_session_id)
        if resume_id is None:
            raise ValueError("claude-p resume session id must be an opaque UUID")
        argv += ["--resume", resume_id]
    else:
        session_id = _safe_uuid(request.session_id)
        if session_id is None:
            raise ValueError("claude-p session id must be an opaque UUID")
        argv += ["--session-id", session_id]
    return argv


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaudePRunResult:
    """Normalized outcome, shaped to slot into the existing delegate result."""

    status: str  # "completed" | "error" | "timeout" | "interrupted"
    summary: Optional[str]
    error: Optional[str]
    session_id: str
    claude_session_id: Optional[str]
    num_turns: Optional[int]
    cost_usd: Optional[float]
    model: Optional[str]
    duration_seconds: float
    exit_reason: str
    tokens: dict[str, int]
    model_usage: dict[str, Any]


def _find_final_json_object(text: str) -> Optional[dict]:
    """Parse only the LAST top-level JSON object in *text*.

    ``claude -p --output-format json`` writes exactly one JSON object on
    success, but defensively scan from the end so any incidental leading
    noise never gets treated as the result — and a malformed/partial tail
    never raises past this function.
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    # Scan candidate object starts left-to-right and keep the LAST one that
    # decodes and consumes the rest of the text (modulo trailing whitespace).
    # Scanning from the right would latch onto a nested object such as
    # ``modelUsage``'s value and silently drop the real result payload.
    best: Optional[dict] = None
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
        except ValueError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict) and not text[end:].strip():
            best = obj
            idx = text.find("{", end)
        else:
            idx = text.find("{", idx + 1)
    return best


def _safe_nonnegative_int(value: Any, *, ceiling: int = 2**63 - 1) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return parsed if 0 <= parsed <= ceiling else 0


def _safe_nonnegative_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _safe_uuid(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError):
        return None


def _safe_model_usage(raw: Any, model: Optional[str]) -> dict[str, Any]:
    """Project Claude telemetry to fixed numeric fields under the reported model.

    Claude Code may key modelUsage by a full model id even when Hermes routed
    through an alias. Prefer an exact match, otherwise accept a single
    Claude-looking reported model id; never preserve identity-looking arbitrary
    keys.
    """
    if not isinstance(raw, dict):
        return {}
    selected_key: Optional[str] = None
    if model and isinstance(raw.get(model), dict):
        selected_key = model
    else:
        candidates = [
            str(key)
            for key, value in raw.items()
            if isinstance(key, str)
            and key.startswith("claude-")
            and isinstance(value, dict)
        ]
        if len(candidates) == 1:
            selected_key = candidates[0]
    if not selected_key:
        return {}
    candidate = raw.get(selected_key)
    if not isinstance(candidate, dict):
        return {}
    projected: dict[str, int | float] = {}
    for key in (
        "inputTokens",
        "outputTokens",
        "cacheReadInputTokens",
        "cacheCreationInputTokens",
    ):
        value = _safe_nonnegative_int(candidate.get(key))
        if value or candidate.get(key) == 0:
            projected[key] = value
    cost = _safe_nonnegative_float(candidate.get("costUSD"))
    if cost is not None:
        projected["costUSD"] = cost
    return {selected_key: projected} if projected else {}


def normalize_claude_p_output(
    raw_stdout: str,
    *,
    session_id: str,
    duration_seconds: float,
    fallback_model: Optional[str] = None,
) -> ClaudePRunResult:
    """Normalize raw ``claude -p --output-format json`` stdout.

    Only the final JSON object is trusted. Any malformed/oversized/missing
    payload degrades to a safe error result — never raises, never leaks the
    raw text (which could contain task-body echoes) into the error message.
    """
    obj = _find_final_json_object(raw_stdout)
    if obj is None:
        return ClaudePRunResult(
            status="error",
            summary=None,
            error="claude -p produced no parseable JSON result",
            session_id=session_id,
            claude_session_id=None,
            num_turns=None,
            cost_usd=None,
            model=fallback_model,
            duration_seconds=duration_seconds,
            exit_reason="malformed_output",
            tokens={"input": 0, "output": 0},
            model_usage={},
        )

    raw_subtype = str(obj.get("subtype") or "").strip().lower()
    allowed_subtypes = {
        "success",
        "error_max_turns",
        "error_budget",
        "error_max_budget_usd",
        "error_during_execution",
    }
    subtype = raw_subtype if raw_subtype in allowed_subtypes else ""
    is_error = bool(obj.get("is_error")) or raw_subtype.startswith("error")
    result_text = obj.get("result")
    usage_raw = obj.get("usage")
    usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
    model_usage = _safe_model_usage(obj.get("modelUsage"), fallback_model)
    cost = obj.get("total_cost_usd", obj.get("cost_usd"))
    cost_val = _safe_nonnegative_float(cost)
    num_turns_raw = obj.get("num_turns")
    num_turns_val = (
        _safe_nonnegative_int(num_turns_raw, ceiling=MAX_TURNS_CEILING)
        if num_turns_raw is not None
        else None
    )

    input_tokens = _safe_nonnegative_int(usage.get("input_tokens", 0))
    output_tokens = _safe_nonnegative_int(usage.get("output_tokens", 0))

    status = "error" if is_error else "completed"
    exit_reason = subtype if subtype else ("error" if is_error else "completed")

    return ClaudePRunResult(
        status=status,
        summary=result_text if not is_error and isinstance(result_text, str) else None,
        error=(
            f"claude -p reported {subtype}"
            if is_error and subtype
            else ("claude -p reported an error" if is_error else None)
        ),
        session_id=session_id,
        claude_session_id=_safe_uuid(obj.get("session_id")),
        num_turns=num_turns_val,
        cost_usd=cost_val,
        model=fallback_model,
        duration_seconds=duration_seconds,
        exit_reason=exit_reason,
        tokens={
            "input": input_tokens,
            "output": output_tokens,
        },
        model_usage=model_usage,
    )


ClaudePEventCallback = Callable[..., None]

_SECRET_MARKERS = ("SECRET", "TOKEN", "API_KEY", "PASSWORD", "PRIVATE_KEY", "CREDENTIAL")


def _contains_secret_marker(text: Any) -> bool:
    if not isinstance(text, str) or not text:
        return False
    upper = text.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def _safe_event_text(text: Any, *, max_chars: int = 4000) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    # Assistant text is already Claude's user-visible output. Do not suppress
    # legitimate prose merely because it mentions words like "token" or
    # "secret"; raw stderr/env/argv/thinking are the channels we never relay.
    return text[:max_chars]


def _emit_safe_event(event_callback: Optional[ClaudePEventCallback], event_type: str, **kwargs: Any) -> None:
    if event_callback is None:
        return
    try:
        event_callback(event_type, **kwargs)
    except Exception:
        logger.debug("claude-p stream callback failed for %s", event_type)


def _safe_tool_input_summary(raw_input: Any) -> dict[str, Any]:
    if not isinstance(raw_input, dict):
        return {"argument_keys": [], "targets": {}}
    keys = sorted(str(key)[:128] for key in raw_input)[:64]
    target_keys = {
        "cwd", "destination_path", "directory", "dst", "endpoint", "file_path",
        "new_path", "old_path", "path", "source_path", "src", "target_path", "url", "urls",
    }
    url_keys = {"endpoint", "url", "urls"}

    def _clean(key: str, value: Any) -> Any:
        if isinstance(value, list):
            cleaned = [_clean(key, item) for item in value[:16]]
            cleaned = [item for item in cleaned if item is not None]
            return cleaned or None
        if not isinstance(value, str) or not value or _contains_secret_marker(value):
            return None
        bounded = value[:1024]
        if key in url_keys:
            from urllib.parse import urlsplit, urlunsplit

            try:
                parsed = urlsplit(bounded)
                if parsed.scheme and parsed.netloc and parsed.hostname:
                    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
                    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
                    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
            except ValueError:
                return None
        return bounded

    targets: dict[str, Any] = {}
    for key, value in raw_input.items():
        k = str(key).lower()
        if k in target_keys:
            cleaned = _clean(k, value)
            if cleaned is not None:
                targets[k] = cleaned
    return {"argument_keys": keys, "targets": targets}


def _iter_content_blocks(obj: Mapping[str, Any]) -> list[Any]:
    message = obj.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            return content
    content = obj.get("content")
    return content if isinstance(content, list) else []


def _handle_stream_event_obj(
    obj: Mapping[str, Any],
    *,
    event_callback: Optional[ClaudePEventCallback],
    tool_names: dict[str, str],
) -> Optional[dict[str, Any]]:
    """Project one Claude stream-json object to safe live events.

    Returns a final result object when *obj* is the terminal Claude result.
    Raw malformed lines, raw stderr, thinking/reasoning deltas, tool payloads,
    and credential-looking text are deliberately ignored.
    """
    obj_type = str(obj.get("type") or "")
    _emit_safe_event(event_callback, "subagent.activity")
    if obj_type == "result" or obj.get("subtype") in {
        "success", "error_max_turns", "error_budget", "error_max_budget_usd", "error_during_execution"
    }:
        usage_raw = obj.get("usage")
        usage = usage_raw if isinstance(usage_raw, dict) else {}
        _emit_safe_event(
            event_callback,
            "subagent.activity",
            input_tokens=_safe_nonnegative_int(usage.get("input_tokens", 0)),
            output_tokens=_safe_nonnegative_int(usage.get("output_tokens", 0)),
        )
        return dict(obj)

    event = obj.get("event") if isinstance(obj.get("event"), dict) else obj
    delta = event.get("delta") if isinstance(event, dict) and isinstance(event.get("delta"), dict) else None
    if delta and delta.get("type") == "text_delta":
        text = _safe_event_text(delta.get("text"))
        if text is not None:
            _emit_safe_event(event_callback, "subagent.text", preview=text)
        return None

    for block in _iter_content_blocks(obj):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            tool_id = str(block.get("id") or "")
            tool_name = str(block.get("name") or "tool")[:256]
            if tool_id:
                tool_names[tool_id] = tool_name
            _emit_safe_event(
                event_callback,
                "tool.started",
                tool=tool_name,
                tool_id=tool_id or None,
                input_summary=_safe_tool_input_summary(block.get("input")),
            )
        elif block.get("type") == "tool_result":
            tool_id = str(block.get("tool_use_id") or "")
            tool_name = tool_names.get(tool_id, "tool")
            _emit_safe_event(
                event_callback,
                "tool.completed",
                tool=tool_name,
                tool_id=tool_id or None,
                status="error" if bool(block.get("is_error")) else "ok",
            )
    return None


def normalize_claude_p_stream(
    raw_stdout: str,
    *,
    session_id: str,
    duration_seconds: float,
    fallback_model: Optional[str] = None,
    event_callback: Optional[ClaudePEventCallback] = None,
) -> ClaudePRunResult:
    """Normalize Claude Code ``stream-json`` stdout and relay safe live events."""
    tool_names: dict[str, str] = {}
    final: Optional[dict[str, Any]] = None
    text_chunks: list[str] = []
    for raw_line in raw_stdout.splitlines():
        if not raw_line:
            continue
        oversized = len(raw_line.encode("utf-8", errors="ignore")) > MAX_STREAM_EVENT_BYTES
        if oversized and '"result"' not in raw_line and '"subtype"' not in raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except ValueError:
            obj = _find_final_json_object(raw_line) if oversized else None
            if obj is None:
                continue
        if not isinstance(obj, dict):
            continue
        event = obj.get("event") if isinstance(obj.get("event"), dict) else obj
        delta = event.get("delta") if isinstance(event, dict) and isinstance(event.get("delta"), dict) else None
        if delta and delta.get("type") == "text_delta":
            text = _safe_event_text(delta.get("text"), max_chars=8000)
            if text:
                text_chunks.append(text)
        maybe_final = _handle_stream_event_obj(
            obj, event_callback=event_callback, tool_names=tool_names
        )
        if maybe_final is not None:
            final = maybe_final
    if final is None:
        if text_chunks:
            summary = "".join(text_chunks)[:16000]
            return ClaudePRunResult(
                status="error",
                summary=summary,
                error="claude -p produced no parseable JSON result",
                session_id=session_id,
                claude_session_id=None,
                num_turns=None,
                cost_usd=None,
                model=fallback_model,
                duration_seconds=duration_seconds,
                exit_reason="malformed_output",
                tokens={"input": 0, "output": 0},
                model_usage={},
            )
        return normalize_claude_p_output(
            "", session_id=session_id, duration_seconds=duration_seconds, fallback_model=fallback_model
        )
    return normalize_claude_p_output(
        json.dumps(final),
        session_id=session_id,
        duration_seconds=duration_seconds,
        fallback_model=fallback_model,
    )


# ---------------------------------------------------------------------------
# Process execution — bounded, cancellable, byte-capped, group-terminated.
# ---------------------------------------------------------------------------


async def _read_capped(stream: asyncio.StreamReader, cap: int) -> tuple[bytes, bool]:
    """Drain *stream* to EOF while retaining at most *cap* bytes.

    Continuing to drain after the cap is essential: stopping the reader lets
    the child fill the OS pipe and deadlock until the outer timeout.
    """
    retained = bytearray()
    overflow = False
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        room = cap - len(retained)
        if room > 0:
            retained.extend(chunk[:room])
        if len(chunk) > room:
            overflow = True
    return bytes(retained), overflow


async def _read_stream_json_capped(
    stream: asyncio.StreamReader,
    cap: int,
    *,
    event_callback: Optional[ClaudePEventCallback] = None,
) -> tuple[bytes, bool]:
    """Drain stdout while incrementally relaying safe stream-json events.

    Retains the bounded head for diagnostics but also keeps the last terminal
    result line separately.  Claude Code emits newline-delimited JSON; parsing
    only complete newline records avoids gluing unrelated objects together, and
    an incremental decoder preserves UTF-8 characters split across pipe reads.
    """
    retained = bytearray()
    overflow = False
    pending = ""
    final_line: Optional[str] = None
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    tool_names: dict[str, str] = {}

    def _process_line(line: str) -> None:
        nonlocal final_line
        if not line:
            return
        line_bytes_len = len(line.encode("utf-8", errors="ignore"))
        try:
            obj = json.loads(line)
        except ValueError:
            obj = None
            # If an oversized/malformed unterminated line precedes Claude's
            # terminal result, salvage the last embedded result object instead
            # of discarding the already-observed final summary.
            for marker in ('{"type": "result"', '{"type":"result"'):
                idx = line.rfind(marker)
                if idx != -1:
                    try:
                        candidate = json.loads(line[idx:])
                    except ValueError:
                        candidate = None
                    if isinstance(candidate, dict):
                        obj = candidate
                        break
            if obj is None:
                return
        if not isinstance(obj, dict):
            return
        maybe_final = _handle_stream_event_obj(
            obj, event_callback=event_callback, tool_names=tool_names
        )
        if maybe_final is not None:
            # Preserve the terminal result even if stdout already exceeded the
            # aggregate cap or the result line itself exceeds the event cap.
            final_line = json.dumps(maybe_final, ensure_ascii=False)
        elif line_bytes_len > MAX_STREAM_EVENT_BYTES:
            return

    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        room = cap - len(retained)
        if room > 0:
            retained.extend(chunk[:room])
        if len(chunk) > room:
            overflow = True
        pending += decoder.decode(chunk, final=False)
        pending_bytes = pending.encode("utf-8", errors="ignore")
        pending_cap = cap + MAX_STREAM_EVENT_BYTES
        if len(pending_bytes) > pending_cap:
            # Bound malformed/no-newline records while retaining the tail where
            # a later terminal result record can still be recovered.
            overflow = True
            pending = pending_bytes[-pending_cap:].decode("utf-8", errors="ignore")
        *lines, pending = pending.split("\n")
        for line in lines:
            # Non-final oversized stream events are ignored after parsing fails;
            # terminal results are still preserved by _process_line.
            if len(line.encode("utf-8", errors="ignore")) > MAX_STREAM_EVENT_BYTES and '"result"' not in line and '"subtype"' not in line:
                continue
            _process_line(line)

    pending += decoder.decode(b"", final=True)
    if pending:
        _process_line(pending)

    if final_line is not None and final_line not in retained.decode("utf-8", errors="ignore"):
        retained.extend(b"\n")
        retained.extend(final_line.encode("utf-8"))
    return bytes(retained), overflow

def _signal_process_group(process_group_id: int, sig: signal.Signals) -> None:
    if os.name == "nt":
        return
    try:
        os.killpg(process_group_id, sig)  # windows-footgun: ok — guarded above
    except (ProcessLookupError, PermissionError, OSError):
        pass


async def _run_claude_p_async(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    workdir: str,
    timeout_seconds: float,
    cancel_event: Optional[threading.Event] = None,
    event_callback: Optional[ClaudePEventCallback] = None,
) -> tuple[int, bytes, bytes, bool, bool, bool]:
    """Return (returncode, stdout, stderr, timed_out, output_overflow, interrupted).

    Never ``shell=True``. On POSIX the child leads its own process group
    (``start_new_session=True``) so a timeout/cancellation can terminate the
    whole group instead of orphaning descendants. Escalates SIGTERM ->
    SIGKILL if the group does not exit within a short grace window.
    """
    popen_kwargs: dict[str, Any] = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=workdir,
        env=dict(env),
        **popen_kwargs,
    )
    # start_new_session=True makes the child's pid its process-group id. Keep
    # this value even after the direct process exits; os.getpgid(pid) may no
    # longer work while descendants still retain stdout/stderr pipe handles.
    process_group_id = proc.pid if os.name != "nt" else None

    assert proc.stdout is not None
    assert proc.stderr is not None
    if hasattr(proc.stdout, "read"):
        stdout_task = asyncio.ensure_future(
            _read_stream_json_capped(proc.stdout, MAX_STDOUT_BYTES, event_callback=event_callback)
        )
    else:
        # Test doubles and defensive fallback for non-StreamReader-like stdout.
        stdout_task = asyncio.ensure_future(_read_capped(proc.stdout, MAX_STDOUT_BYTES))
    stderr_task = asyncio.ensure_future(_read_capped(proc.stderr, MAX_STDERR_BYTES))

    timed_out = False
    interrupted = False
    try:
        if cancel_event is None:
            await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
        else:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))
            while proc.returncode is None:
                if cancel_event.is_set():
                    interrupted = True
                    await _cancel_process(proc, process_group_id=process_group_id)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    await _cancel_process(proc, process_group_id=process_group_id)
                    break
                try:
                    await asyncio.wait_for(proc.wait(), timeout=min(0.1, remaining))
                except asyncio.TimeoutError:
                    continue
    except asyncio.TimeoutError:
        timed_out = True
        await _cancel_process(proc, process_group_id=process_group_id)
    except asyncio.CancelledError:
        await _cancel_process(proc, process_group_id=process_group_id)
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise

    try:
        stdout, stdout_overflow, stderr, stderr_overflow = await _drain_output_tasks(
            proc,
            stdout_task,
            stderr_task,
            process_group_id=process_group_id,
        )
    except asyncio.CancelledError:
        await _cancel_process(proc, process_group_id=process_group_id)
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise
    returncode = proc.returncode if proc.returncode is not None else -1
    if interrupted and returncode == 0:
        returncode = -1
    return (
        returncode,
        stdout,
        stderr,
        timed_out,
        (stdout_overflow or stderr_overflow),
        interrupted,
    )


async def _cancel_process(
    proc: "asyncio.subprocess.Process",
    *,
    process_group_id: Optional[int] = None,
) -> None:
    """Terminate-then-kill escalation on the process group, bounded grace."""
    if os.name == "nt":
        # taskkill /T must run while the direct parent PID still identifies the
        # tree. A graceful direct-parent terminate can orphan descendants and
        # make the later tree lookup ineffective.
        from hermes_cli._subprocess_compat import kill_process_tree

        kill_process_tree(proc)  # type: ignore[arg-type]
        try:
            await asyncio.wait_for(proc.wait(), timeout=PROCESS_KILL_GRACE_SECONDS)
        except asyncio.TimeoutError:
            pass
        return

    _signal_process_group(process_group_id or proc.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(
            proc.wait(), timeout=PROCESS_TERMINATE_GRACE_SECONDS
        )
    except asyncio.TimeoutError:
        pass
    # Do not return merely because the direct process exited. A descendant can
    # ignore SIGTERM and retain duplicated pipe handles. Escalate the saved
    # process-group id so output drains can reach EOF.
    _signal_process_group(
        process_group_id or proc.pid,
        signal.SIGKILL,  # windows-footgun: ok — Windows returns above
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=PROCESS_KILL_GRACE_SECONDS)
    except asyncio.TimeoutError:
        pass


async def _drain_output_tasks(
    proc: "asyncio.subprocess.Process",
    stdout_task: "asyncio.Task[tuple[bytes, bool]]",
    stderr_task: "asyncio.Task[tuple[bytes, bool]]",
    *,
    process_group_id: Optional[int],
) -> tuple[bytes, bool, bytes, bool]:
    """Drain both pipes without ever waiting indefinitely for descendant EOF."""
    combined = asyncio.gather(stdout_task, stderr_task)
    try:
        (stdout, stdout_overflow), (stderr, stderr_overflow) = await asyncio.wait_for(
            asyncio.shield(combined), timeout=PIPE_DRAIN_GRACE_SECONDS
        )
        return stdout, stdout_overflow, stderr, stderr_overflow
    except asyncio.TimeoutError:
        await _cancel_process(proc, process_group_id=process_group_id)
    except asyncio.CancelledError:
        combined.cancel()
        await asyncio.gather(combined, return_exceptions=True)
        raise

    try:
        (stdout, stdout_overflow), (stderr, stderr_overflow) = await asyncio.wait_for(
            asyncio.shield(combined), timeout=PIPE_DRAIN_GRACE_SECONDS
        )
        return stdout, stdout_overflow, stderr, stderr_overflow
    except asyncio.TimeoutError:
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        combined.cancel()
        await asyncio.gather(combined, return_exceptions=True)
        return b"", True, b"", True
    except asyncio.CancelledError:
        combined.cancel()
        await asyncio.gather(combined, return_exceptions=True)
        raise


def run_claude_p_task(
    request: ClaudePRunRequest,
    *,
    write_capable: bool,
    cancel_event: Optional[threading.Event] = None,
    event_callback: Optional[ClaudePEventCallback] = None,
) -> ClaudePRunResult:
    """Run one bounded ``claude -p`` task synchronously (drives its own loop).

    Safe to call from a plain worker thread (each call gets its own asyncio
    event loop via ``asyncio.run``). Write-capable tasks are serialized per
    workdir; read-only tasks may run concurrently against the same workdir.
    """
    start = time.monotonic()
    executable = resolve_claude_executable()
    if not executable:
        return ClaudePRunResult(
            status="error",
            summary=None,
            error="claude CLI is not installed or not on PATH",
            session_id=request.session_id,
            claude_session_id=None,
            num_turns=None,
            cost_usd=None,
            model=request.model,
            duration_seconds=0.0,
            exit_reason="unavailable",
            tokens={"input": 0, "output": 0},
            model_usage={},
        )

    lock = workdir_lock_for(request.workdir) if write_capable else None
    acquired = False
    settings_dir: Optional[str] = None
    try:
        if lock is not None:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    duration = time.monotonic() - start
                    return ClaudePRunResult(
                        status="interrupted",
                        summary=None,
                        error="claude -p was interrupted before acquiring the workdir lock",
                        session_id=request.session_id,
                        claude_session_id=None,
                        num_turns=None,
                        cost_usd=None,
                        model=request.model,
                        duration_seconds=duration,
                        exit_reason="interrupted",
                        tokens={"input": 0, "output": 0},
                        model_usage={},
                    )
                if lock.acquire(timeout=0.1):
                    acquired = True
                    break

        profile = resolve_tool_profile(request.tool_profile)
        if not profile.passthrough:
            settings_path, settings_dir = _create_restricted_settings(
                profile.name, request.workdir
            )
            request = replace(request, settings_path=settings_path)
        argv = build_claude_p_argv(request, executable=executable)
        env = build_scrubbed_environment()

        async def _runner() -> tuple:
            run_kwargs = {
                "env": env,
                "workdir": request.workdir,
                "timeout_seconds": request.timeout_seconds,
            }
            if cancel_event is not None:
                run_kwargs["cancel_event"] = cancel_event
            if event_callback is not None:
                run_kwargs["event_callback"] = event_callback
            task = asyncio.ensure_future(_run_claude_p_async(argv, **run_kwargs))
            return await task

        run_result = asyncio.run(_runner())
        if len(run_result) == 6:
            (
                returncode,
                stdout_bytes,
                _stderr_bytes,
                timed_out,
                output_overflow,
                interrupted,
            ) = run_result
        else:
            # Backward compatibility for third-party/test doubles that still
            # implement the historical five-item runner contract.
            returncode, stdout_bytes, _stderr_bytes, timed_out, output_overflow = run_result
            interrupted = bool(
                cancel_event is not None and cancel_event.is_set() and not timed_out
            )
    except Exception as exc:
        duration = time.monotonic() - start
        logger.debug("claude-p spawn failed: %s", type(exc).__name__)
        return ClaudePRunResult(
            status="error",
            summary=None,
            error=f"claude -p failed to start ({type(exc).__name__})",
            session_id=request.session_id,
            claude_session_id=None,
            num_turns=None,
            cost_usd=None,
            model=request.model,
            duration_seconds=duration,
            exit_reason="spawn_error",
            tokens={"input": 0, "output": 0},
            model_usage={},
        )
    finally:
        if settings_dir:
            shutil.rmtree(settings_dir, ignore_errors=True)
        if acquired and lock is not None:
            lock.release()

    duration = time.monotonic() - start
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")

    if interrupted:
        parsed = normalize_claude_p_stream(
            stdout_text,
            session_id=request.session_id,
            duration_seconds=duration,
            fallback_model=request.model,
        ) if stdout_text.strip() else None
        return ClaudePRunResult(
            status="interrupted",
            summary=parsed.summary if parsed is not None else None,
            error="claude -p was interrupted and terminated",
            session_id=request.session_id,
            claude_session_id=(parsed.claude_session_id if parsed is not None else None),
            num_turns=(parsed.num_turns if parsed is not None else None),
            cost_usd=(parsed.cost_usd if parsed is not None else None),
            model=request.model,
            duration_seconds=duration,
            exit_reason="interrupted",
            tokens=(parsed.tokens if parsed is not None else {"input": 0, "output": 0}),
            model_usage=(parsed.model_usage if parsed is not None else {}),
        )

    if output_overflow:
        parsed_overflow = normalize_claude_p_stream(
            stdout_text,
            session_id=request.session_id,
            duration_seconds=duration,
            fallback_model=request.model,
        )
        if parsed_overflow.exit_reason != "malformed_output":
            return parsed_overflow
        return ClaudePRunResult(
            status="error",
            summary=None,
            error="claude -p output exceeded its bounded size limit",
            session_id=request.session_id,
            claude_session_id=None,
            num_turns=None,
            cost_usd=None,
            model=request.model,
            duration_seconds=duration,
            exit_reason="output_limit",
            tokens={"input": 0, "output": 0},
            model_usage={},
        )

    if timed_out:
        return ClaudePRunResult(
            status="timeout",
            summary=None,
            error="claude -p exceeded its bounded timeout and was terminated",
            session_id=request.session_id,
            claude_session_id=None,
            num_turns=None,
            cost_usd=None,
            model=request.model,
            duration_seconds=duration,
            exit_reason="timeout",
            tokens={"input": 0, "output": 0},
            model_usage={},
        )

    if returncode != 0:
        parsed = _find_final_json_object(stdout_text)
        if parsed is not None:
            result = normalize_claude_p_stream(
                stdout_text, session_id=request.session_id, duration_seconds=duration, fallback_model=request.model
            )
            return ClaudePRunResult(**{**result.__dict__, "status": "error"})
        return ClaudePRunResult(
            status="error",
            summary=None,
            error=f"claude -p exited with status {returncode}",
            session_id=request.session_id,
            claude_session_id=None,
            num_turns=None,
            cost_usd=None,
            model=request.model,
            duration_seconds=duration,
            exit_reason="nonzero_exit",
            tokens={"input": 0, "output": 0},
            model_usage={},
        )

    return normalize_claude_p_stream(
        stdout_text,
        session_id=request.session_id,
        duration_seconds=duration,
        fallback_model=request.model,
    )
