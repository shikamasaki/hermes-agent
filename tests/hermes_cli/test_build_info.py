"""Tests for hermes_cli.build_info — baked-in build metadata resolution.

The build SHA is written by the Dockerfile's ``HERMES_GIT_SHA`` build-arg
into ``<project_root>/.hermes_build_sha``.  These tests cover the read-side
helpers: missing file, truncation, content identity stability, binary/untracked
coverage, and failure tolerance.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


def _init_repo(root: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)


def _make_repo_with_files(root: Path, files: dict[str, bytes]) -> Path:
    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    repo = root / "repo"
    repo.mkdir()
    _init_repo(repo)
    for rel, data in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return repo


def test_get_build_sha_returns_none_when_file_absent(tmp_path):
    """Source installs: no file present → None, callers fall back to git."""
    from hermes_cli import build_info

    missing = tmp_path / ".hermes_build_sha"  # never created

    with patch.object(build_info, "_BUILD_SHA_FILE", missing):
        assert build_info.get_build_sha() is None


def test_get_build_sha_respects_short_argument(tmp_path):
    """``short=N`` truncates to N chars; ``short<=0`` returns full SHA."""
    from hermes_cli import build_info

    sha_file = tmp_path / ".hermes_build_sha"
    full_sha = "abcdef1234567890abcdef1234567890abcdef12"
    sha_file.write_text(full_sha + "\n")

    with patch.object(build_info, "_BUILD_SHA_FILE", sha_file):
        assert build_info.get_build_sha(short=12) == "abcdef123456"
        assert build_info.get_build_sha(short=0) == full_sha
        assert build_info.get_build_sha(short=-1) == full_sha


@pytest.mark.parametrize("captured_pid", [111, 222])
def test_startup_code_identity_is_captured_once_and_survives_refresh(tmp_path, monkeypatch, captured_pid):
    from hermes_cli import build_info

    first = {"sha": "a" * 40, "short_sha": "a" * 8, "version": "1.0", "source": "git"}
    second = {"sha": "b" * 40, "short_sha": "b" * 8, "version": "2.0", "source": "git"}
    calls = []

    monkeypatch.setattr(build_info.os, "getpid", lambda: captured_pid)
    monkeypatch.setattr(build_info, "get_code_identity", lambda refresh=False: calls.append(refresh) or first)

    stored = build_info.record_startup_code_identity()
    assert stored == first
    assert build_info.get_startup_code_identity() == first

    stored["sha"] = "mutated"
    assert build_info.get_startup_code_identity() == first

    monkeypatch.setattr(build_info, "get_code_identity", lambda refresh=False: second)
    assert build_info.record_startup_code_identity() == first
    assert build_info.get_startup_code_identity() == first
    assert calls == [True]


@pytest.mark.parametrize("captured_pid", [311, 422])
def test_startup_code_identity_stays_unconfirmed_after_initial_capture_failure(monkeypatch, captured_pid):
    from hermes_cli import build_info

    later = {"sha": "f" * 40, "short_sha": "f" * 8, "version": "5.0", "source": "git"}
    calls = []

    monkeypatch.setattr(build_info.os, "getpid", lambda: captured_pid)

    def fake_get_code_identity(refresh=False):
        calls.append(refresh)
        if len(calls) == 1:
            raise OSError("startup identity unavailable")
        return later

    monkeypatch.setattr(build_info, "get_code_identity", fake_get_code_identity)

    assert build_info.record_startup_code_identity() is None
    assert build_info.get_startup_code_identity() is None
    assert build_info.record_startup_code_identity() is None
    assert build_info.get_startup_code_identity() is None

    monkeypatch.setattr(build_info.os, "getpid", lambda: captured_pid + 1)
    assert build_info.record_startup_code_identity() == later
    assert build_info.get_startup_code_identity() == later
    assert calls == [True, True]


def test_startup_code_identity_failure_blocks_later_explicit_identity_same_pid(monkeypatch):
    from hermes_cli import build_info

    monkeypatch.setattr(build_info.os, "getpid", lambda: 533)
    monkeypatch.setattr(
        build_info,
        "get_code_identity",
        lambda refresh=False: (_ for _ in ()).throw(OSError("startup identity unavailable")),
    )

    assert build_info.record_startup_code_identity() is None
    assert build_info.record_startup_code_identity({"content_sha": "later-disk"}) is None
    assert build_info.get_startup_code_identity() is None


@pytest.mark.parametrize("other_pid", [112, 223])
def test_startup_code_identity_is_pid_scoped(monkeypatch, other_pid):
    from hermes_cli import build_info

    first = {"sha": "c" * 40, "short_sha": "c" * 8, "version": "3.0", "source": "git"}
    second = {"sha": "d" * 40, "short_sha": "d" * 8, "version": "4.0", "source": "git"}

    monkeypatch.setattr(build_info.os, "getpid", lambda: 101)
    monkeypatch.setattr(build_info, "get_code_identity", lambda refresh=False: first)
    assert build_info.record_startup_code_identity() == first
    assert build_info.get_startup_code_identity() == first

    monkeypatch.setattr(build_info.os, "getpid", lambda: other_pid)
    assert build_info.get_startup_code_identity() is None
    monkeypatch.setattr(build_info, "get_code_identity", lambda refresh=False: second)
    assert build_info.record_startup_code_identity() == second
    assert build_info.get_startup_code_identity() == second


@pytest.mark.parametrize("before_capture", [True, False])
def test_startup_code_identity_read_does_not_capture(monkeypatch, before_capture):
    from hermes_cli import build_info

    monkeypatch.setattr(build_info.os, "getpid", lambda: 777)
    monkeypatch.setattr(
        build_info,
        "get_code_identity",
        lambda refresh=False: (_ for _ in ()).throw(AssertionError("should not capture")),
    )

    if before_capture:
        assert build_info.get_startup_code_identity() is None
    else:
        assert build_info.record_startup_code_identity({"sha": "e" * 40}) == {"sha": "e" * 40}
        assert build_info.get_startup_code_identity() == {"sha": "e" * 40}



def _init_git_repo(root: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)


@pytest.fixture()
def content_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    (repo / "binary.bin").write_bytes(b"\x00\x01binary\xff\x00")
    import subprocess

    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    return repo


@pytest.fixture(autouse=True)
def _reset_startup_code_identity_state(monkeypatch):
    from hermes_cli import build_info

    monkeypatch.setattr(build_info, "_startup_code_identity_cache", None, raising=False)
    monkeypatch.setattr(build_info, "_startup_code_identity_pid", None, raising=False)
    monkeypatch.setattr(build_info, "_startup_code_identity_attempted_pid", None, raising=False)


def test_get_code_identity_content_sha_is_stable_for_same_content(content_repo):
    from hermes_cli import build_info

    with patch.object(build_info, "_PROJECT_ROOT", content_repo):
        first = build_info.get_code_identity(refresh=True)
    # Touch metadata without changing bytes.
    (content_repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    (content_repo / "binary.bin").write_bytes(b"\x00\x01binary\xff\x00")
    with patch.object(build_info, "_PROJECT_ROOT", content_repo):
        second = build_info.get_code_identity(refresh=True)

    assert first["content_sha"] == second["content_sha"]
    assert first["content_short_sha"] == first["content_sha"][:8]


def test_get_code_identity_content_sha_changes_when_content_changes(content_repo):
    from hermes_cli import build_info

    with patch.object(build_info, "_PROJECT_ROOT", content_repo):
        first = build_info.get_code_identity(refresh=True)
    (content_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    with patch.object(build_info, "_PROJECT_ROOT", content_repo):
        second = build_info.get_code_identity(refresh=True)

    assert first["content_sha"] != second["content_sha"]


def test_get_code_identity_content_sha_includes_untracked_and_binary_files(content_repo):
    from hermes_cli import build_info

    with patch.object(build_info, "_PROJECT_ROOT", content_repo):
        first = build_info.get_code_identity(refresh=True)
    (content_repo / "untracked.txt").write_text("untracked changed\n", encoding="utf-8")
    (content_repo / "binary.bin").write_bytes(b"\x00\x01binary-changed\xff\x00")
    with patch.object(build_info, "_PROJECT_ROOT", content_repo):
        second = build_info.get_code_identity(refresh=True)

    assert first["content_sha"] != second["content_sha"]


def test_iter_repo_content_paths_sets_git_timeout_and_returns_none_on_timeout(
    tmp_path, monkeypatch
):
    from hermes_cli import build_info

    seen = {}

    def fake_run(args, **kwargs):
        seen["kwargs"] = kwargs
        raise build_info.subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    monkeypatch.setattr(build_info.subprocess, "run", fake_run)

    assert build_info._iter_repo_content_paths(tmp_path) is None
    assert seen["kwargs"]["timeout"] > 0


def test_get_code_identity_content_sha_distinguishes_nul_boundary_collisions(tmp_path):
    from hermes_cli import build_info

    repo1 = _make_repo_with_files(
        tmp_path / "one",
        {"a": bytes([120, 0, 98, 0, 121]), "b": bytes([122])},
    )
    repo2 = _make_repo_with_files(
        tmp_path / "two",
        {"a": bytes([120]), "b": bytes([121, 0, 98, 0, 122])},
    )

    with patch.object(build_info, "_PROJECT_ROOT", repo1):
        first = build_info.get_code_identity(refresh=True)
    with patch.object(build_info, "_PROJECT_ROOT", repo2):
        second = build_info.get_code_identity(refresh=True)

    assert first["content_sha"] != second["content_sha"]


def test_get_code_identity_content_sha_rejects_parent_symlink_race(tmp_path, monkeypatch):
    from hermes_cli import build_info
    import os
    import subprocess

    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    _init_repo(repo)
    (repo / "dir").mkdir()
    (repo / "dir" / "inside.txt").write_text("safe\n", encoding="utf-8")
    (outside / "inside.txt").write_text("outside\n", encoding="utf-8")
    subprocess.run(["git", "add", "dir/inside.txt"], cwd=repo, check=True)

    original_lstat = Path.lstat
    original_open = Path.open
    original_fdopen = os.fdopen
    swapped = False
    outside_reads = 0

    class _Probe:
        def __init__(self, fh):
            self._fh = fh

        def read(self, *args, **kwargs):
            nonlocal outside_reads
            outside_reads += 1
            return self._fh.read(*args, **kwargs)

        def __enter__(self):
            self._fh.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._fh.__exit__(exc_type, exc, tb)

        def __getattr__(self, name):
            return getattr(self._fh, name)

    def _lstat(self, *args, **kwargs):
        nonlocal swapped
        result = original_lstat(self, *args, **kwargs)
        if self == repo / "dir" / "inside.txt" and not swapped:
            swapped = True
            (repo / "dir").rename(repo / "dir.real")
            (repo / "dir").symlink_to(outside, target_is_directory=True)
        return result

    def _open(self, *args, **kwargs):
        fh = original_open(self, *args, **kwargs)
        if self == repo / "dir" / "inside.txt":
            return _Probe(fh)
        return fh

    def _fdopen(fd, *args, **kwargs):
        fh = original_fdopen(fd, *args, **kwargs)
        return _Probe(fh)

    monkeypatch.setattr(Path, "lstat", _lstat)
    monkeypatch.setattr(Path, "open", _open)
    monkeypatch.setattr(os, "fdopen", _fdopen)

    with patch.object(build_info, "_PROJECT_ROOT", repo):
        identity = build_info.get_code_identity(refresh=True)

    assert swapped is True
    assert outside_reads == 0
    assert identity["content_sha"] is None
    assert identity["content_short_sha"] is None


def test_get_code_identity_content_sha_degrades_to_none_on_git_failure(content_repo, monkeypatch):
    from hermes_cli import build_info

    monkeypatch.setattr(
        build_info.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    with patch.object(build_info, "_PROJECT_ROOT", content_repo):
        identity = build_info.get_code_identity(refresh=True)

    assert identity["content_sha"] is None
    assert identity["content_short_sha"] is None


@pytest.mark.parametrize("opened_name", ["tracked.txt", "untracked.txt"])
def test_get_code_identity_content_sha_degrades_to_none_on_unreadable_file(
    content_repo, monkeypatch, opened_name
):
    from hermes_cli import build_info

    original_open = build_info.os.open

    def _open(path, flags, *args, **kwargs):
        if path == opened_name and kwargs.get("dir_fd") is not None:
            raise OSError("permission denied")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(build_info.os, "open", _open)
    with patch.object(build_info, "_PROJECT_ROOT", content_repo):
        identity = build_info.get_code_identity(refresh=True)

    assert identity["content_sha"] is None
    assert identity["content_short_sha"] is None


