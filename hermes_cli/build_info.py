"""
Baked-in build metadata for Hermes Agent.

Source installs report their git revision live via ``git rev-parse`` (see
``hermes_cli/dump.py`` and ``hermes_cli/banner.py``).  That doesn't work inside
the published Docker image because ``.dockerignore`` excludes ``.git``, so
those callsites fall back to ``"(unknown)"`` / drop the banner suffix entirely.

To make ``hermes dump`` and the startup banner identify the exact commit the
image was built from, the Docker build writes the build-time ``$HERMES_GIT_SHA``
arg into ``<project_root>/.hermes_build_sha``.  This module is the single
read-side helper consumed by both callsites — keeping the lookup in one place
so the file path and missing-file behaviour stay consistent.

Behaviour:

- Returns ``None`` when the file is absent.  Source installs and dev images
  built without the ``HERMES_GIT_SHA`` build-arg fall through to live-git
  resolution in the caller, so non-Docker installs are unaffected.
- Returns ``None`` on any IO / decoding error.  The build-sha is a nice-to-have
  for support triage; nothing in the CLI is allowed to crash because of it.
- Truncates to ``short`` characters (default 8) to match the format used by
  ``git rev-parse --short=8`` throughout the codebase.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path
from threading import RLock
from typing import Optional

# Path is resolved relative to this module so it works regardless of cwd —
# matches the pattern used by ``banner._resolve_repo_dir``.
_PROJECT_ROOT = Path(__file__).parent.parent
_BUILD_SHA_FILE = _PROJECT_ROOT / ".hermes_build_sha"
_GIT_LS_FILES_TIMEOUT_SECONDS = 5


_code_identity_cache: Optional[dict] = None
_startup_code_identity_cache: Optional[dict] = None
_startup_code_identity_pid: Optional[int] = None
_startup_code_identity_attempted_pid: Optional[int] = None
_startup_code_identity_lock = RLock()


def _iter_repo_content_paths(project_root: Path) -> Optional[list[Path]]:
    """Return git-reported paths that should participate in content identity.

    The identity must see tracked files, untracked files, and binary files.
    If git cannot enumerate the tree cleanly, or if the repo is unreadable,
    return ``None`` rather than pretending two unknown trees are equivalent.
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_LS_FILES_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    paths: list[Path] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = os.fsdecode(raw)
        path = project_root / rel
        try:
            st = path.lstat()
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        paths.append(path)

    return paths


def _safe_read_repo_file_bytes(project_root: Path, rel_path: Path) -> Optional[bytes]:
    """Read a tracked file without following symlinks in any path component."""
    opened_dir_fds: list[int] = []
    directory_flags = getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_RDONLY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow_flag:
        return None

    try:
        current_fd = os.open(os.fspath(project_root), directory_flags | nofollow_flag)
        opened_dir_fds.append(current_fd)

        parts = rel_path.parts
        if not parts:
            return None

        for part in parts[:-1]:
            current_fd = os.open(part, directory_flags | nofollow_flag, dir_fd=current_fd)
            opened_dir_fds.append(current_fd)

        file_fd: Optional[int] = None
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | nofollow_flag,
            dir_fd=current_fd,
        )
        try:
            st = os.fstat(file_fd)
            if not stat.S_ISREG(st.st_mode):
                return None
            digest = hashlib.sha256()
            with os.fdopen(file_fd, "rb", closefd=False) as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            return digest.digest()
        finally:
            if file_fd is not None:
                try:
                    os.close(file_fd)
                except OSError:
                    pass
    except OSError:
        return None
    finally:
        for fd in reversed(opened_dir_fds):
            try:
                os.close(fd)
            except OSError:
                pass


def _resolve_repo_content_sha(project_root: Path) -> Optional[str]:
    """Hash the current checkout contents in a stable, path-aware way."""
    paths = _iter_repo_content_paths(project_root)
    if paths is None:
        return None

    digest = hashlib.sha256()
    try:
        for path in sorted(paths, key=lambda p: p.as_posix()):
            rel = path.relative_to(project_root)
            rel_bytes = rel.as_posix().encode("utf-8", errors="surrogateescape")
            content = _safe_read_repo_file_bytes(project_root, rel)
            if content is None:
                return None
            content_digest = hashlib.sha256(content).digest()
            digest.update(len(rel_bytes).to_bytes(8, "big"))
            digest.update(rel_bytes)
            digest.update(content_digest)
    except Exception:
        return None
    return digest.hexdigest()


def _resolve_git_head_sha(project_root: Path) -> Optional[str]:
    """Resolve the checkout's HEAD commit sha by reading .git directly.

    Deliberately NOT ``git rev-parse`` in a subprocess: this helper runs
    inside library paths (gateway runtime-status writes, update receipts)
    where spawning processes is both slow and hostile to tests that mock
    ``subprocess.run`` tightly (call-count asserts, sequenced side effects).
    Handles regular checkouts, worktrees/submodules (``.git`` file with a
    ``gitdir:`` pointer + ``commondir``), loose refs, and packed-refs.
    Returns None on any failure.
    """
    try:
        git_path = project_root / ".git"
        if git_path.is_file():
            # Worktree/submodule: ".git" is a pointer file.
            pointer = git_path.read_text(encoding="utf-8", errors="replace").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = Path(pointer[len("gitdir:"):].strip())
            if not git_dir.is_absolute():
                git_dir = (project_root / git_dir).resolve()
        elif git_path.is_dir():
            git_dir = git_path
        else:
            return None

        # Refs live in the COMMON git dir for worktrees.
        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.is_file():
            rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
            common = Path(rel)
            if not common.is_absolute():
                common = (git_dir / common).resolve()
            common_dir = common

        head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        if not head.startswith("ref:"):
            # Detached HEAD: the file holds the sha itself.
            return head if len(head) == 40 else None
        ref_name = head[len("ref:"):].strip()

        loose = common_dir / ref_name
        if loose.is_file():
            sha = loose.read_text(encoding="utf-8", errors="replace").strip()
            return sha if len(sha) == 40 else None

        packed = common_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "^")):
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[1].strip() == ref_name:
                    sha = parts[0].strip()
                    return sha if len(sha) == 40 else None
    except Exception:
        return None
    return None


def get_code_identity(refresh: bool = False, probe_content: bool = True) -> dict:
    """Return the running checkout's code identity as a dict.

    Shape: ``{"sha": full-or-short sha | None, "short_sha": str | None,
    "version": pyproject version | None, "source": "git" | "build-file" |
    "unknown", "content_sha": content digest | None, "content_short_sha":
    first 8 chars of content digest | None}``.

    Resolution order mirrors the banner/dump callsites: live ``git
    rev-parse`` for source installs, the baked ``.hermes_build_sha`` for
    Docker images (no ``.git`` inside the published image), else unknown.

    ``probe_content=False`` skips ``_resolve_repo_content_sha`` entirely —
    that helper shells out to ``git ls-files`` (see
    ``_iter_repo_content_paths``), which a package-managed-install refusal
    receipt must never invoke (#91277 admission contract: a refused update
    performs zero git/subprocess work). Callers that don't need the content
    digest (e.g. the refusal receipt) should pass ``probe_content=False``;
    the result is never cached in that mode, so it can't shadow the full
    identity a later ``probe_content=True`` call needs.

    Cached per process — code identity cannot change while a process is
    running (an updated checkout requires a restart to take effect, which
    is exactly the property the fleet version verification relies on).
    Never raises; every field degrades to ``None`` independently.
    """
    global _code_identity_cache
    if probe_content and _code_identity_cache is not None and not refresh:
        return dict(_code_identity_cache)

    sha: Optional[str] = None
    source = "unknown"
    project_root = _PROJECT_ROOT
    resolved = _resolve_git_head_sha(project_root)
    if resolved:
        sha = resolved
        source = "git"
    if sha is None:
        baked = get_build_sha(short=0)
        if baked:
            sha = baked
            source = "build-file"

    content_sha = _resolve_repo_content_sha(project_root) if probe_content else None

    version: Optional[str] = None
    try:
        import tomllib

        with open(project_root / "pyproject.toml", "rb") as fh:  # windows-footgun: ok — binary mode, tomllib requires bytes
            raw_version = tomllib.load(fh).get("project", {}).get("version")
        version = str(raw_version) if raw_version else None
    except Exception:
        version = None

    identity = {
        "sha": sha,
        "short_sha": sha[:8] if sha else None,
        "version": version,
        "source": source,
        "content_sha": content_sha,
        "content_short_sha": content_sha[:8] if content_sha else None,
    }
    if not probe_content:
        return dict(identity)
    _code_identity_cache = identity
    return dict(_code_identity_cache)


def record_startup_code_identity(identity: Optional[dict] = None) -> Optional[dict]:
    """Capture the first code identity seen by this process.

    The startup snapshot is intentionally separate from ``get_code_identity``'s
    refreshable cache: callers can refresh the live checkout view without
    rewriting the boot-time snapshot used by startup comparisons.
    """
    global _startup_code_identity_cache, _startup_code_identity_pid, _startup_code_identity_attempted_pid
    current_pid = os.getpid()
    with _startup_code_identity_lock:
        if (
            _startup_code_identity_cache is not None
            and _startup_code_identity_pid == current_pid
        ):
            return dict(_startup_code_identity_cache)
        if _startup_code_identity_attempted_pid == current_pid:
            return None
        if identity is None:
            try:
                identity = get_code_identity(refresh=True)
            except Exception:
                _startup_code_identity_attempted_pid = current_pid
                return None
            _startup_code_identity_attempted_pid = current_pid
        _startup_code_identity_cache = dict(identity)
        _startup_code_identity_pid = current_pid
        return dict(_startup_code_identity_cache)


def get_startup_code_identity() -> Optional[dict]:
    """Return the captured startup code identity for this process, if any."""
    current_pid = os.getpid()
    with _startup_code_identity_lock:
        if (
            _startup_code_identity_cache is None
            or _startup_code_identity_pid != current_pid
        ):
            return None
        return dict(_startup_code_identity_cache)


def get_build_sha(short: int = 8) -> Optional[str]:
    """Return the baked-in build SHA, truncated to ``short`` chars, or None.

    Reads ``<project_root>/.hermes_build_sha`` if present.  The file is
    written by the Dockerfile's ``HERMES_GIT_SHA`` build-arg and contains
    the full 40-character commit hash on a single line.
    """
    try:
        if not _BUILD_SHA_FILE.is_file():
            return None
        sha = _BUILD_SHA_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not sha:
        return None
    return sha[:short] if short and short > 0 else sha
