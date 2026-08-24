"""Fail-closed PreToolUse policy for restricted ``claude -p`` profiles."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

BASH_ALLOWED_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("git", "show"),
    ("uv", "run", "pytest"),
    ("uv", "run", "mypy"),
    ("uv", "run", "ruff"),
    ("pytest",),
    ("python", "-m", "pytest"),
    ("scripts/run_tests.sh",),
    ("venv/bin/ruff", "check"),
    ("venv/bin/python", "-m", "compileall"),
    ("npm", "test"),
    ("npm", "run", "test"),
    ("npm", "run", "lint"),
    ("npm", "run", "typecheck"),
    ("npm", "run", "build"),
    ("pnpm", "test"),
    ("pnpm", "run", "test"),
    ("pnpm", "run", "lint"),
    ("pnpm", "run", "typecheck"),
    ("pnpm", "run", "build"),
    ("yarn", "test"),
    ("yarn", "run", "test"),
    ("yarn", "run", "lint"),
    ("yarn", "run", "typecheck"),
    ("yarn", "run", "build"),
    ("cargo", "test"),
    ("cargo", "check"),
    ("cargo", "clippy"),
    ("go", "test"),
    ("make", "test"),
    ("make", "lint"),
    ("make", "check"),
)

_SHELL_META = frozenset(";&|`$><\n\r")


def _contained(raw_path: Any, workdir: Path) -> bool:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = workdir / candidate
    try:
        return candidate.resolve(strict=False).is_relative_to(workdir)
    except (OSError, RuntimeError, ValueError):
        return False


def _bash_allowed(command: Any, workdir: Path) -> bool:
    if not isinstance(command, str) or not command.strip():
        return False
    if any(char in command for char in _SHELL_META):
        return False
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return False
    if not argv or "=" in argv[0]:
        return False
    if not any(tuple(argv[: len(prefix)]) == prefix for prefix in BASH_ALLOWED_PREFIXES):
        return False
    for token in argv:
        if token.startswith("/") and not _contained(token, workdir):
            return False
        if token == ".." or token.startswith("../") or "/../" in token:
            return False
    return True


def evaluate_tool_call(
    payload: Mapping[str, Any], profile: str, workdir: str | Path
) -> tuple[bool, str]:
    root = Path(workdir).expanduser().resolve(strict=False)
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if payload.get("hook_event_name") != "PreToolUse" or not isinstance(tool_input, Mapping):
        return False, "malformed PreToolUse request"
    if profile not in {"read_only", "review", "coding"}:
        return False, "unknown restricted tool profile"
    if tool_name == "Read":
        allowed = _contained(tool_input.get("file_path"), root)
        return allowed, "Read must stay within the delegated workspace"
    if tool_name in {"Edit", "Write"}:
        if profile != "coding":
            return False, f"{profile} is read-only"
        allowed = _contained(tool_input.get("file_path"), root)
        return allowed, f"{tool_name} must stay within the delegated workspace"
    if tool_name == "Bash":
        if profile != "coding":
            return False, f"{profile} has no Bash capability"
        allowed = _bash_allowed(tool_input.get("command"), root)
        return allowed, "Bash command is outside the fixed verification allowlist"
    return False, f"tool {tool_name!r} is not available in restricted profile {profile!r}"


def _decision(allowed: bool, reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if allowed else "deny",
            "permissionDecisionReason": reason,
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)
    try:
        payload = json.load(sys.stdin)
    except (TypeError, ValueError):
        payload = {}
    allowed, reason = evaluate_tool_call(payload, args.profile, args.workdir)
    json.dump(_decision(allowed, reason), sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
