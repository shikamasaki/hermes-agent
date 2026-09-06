"""Regression tests for delegated child terminal snapshot isolation."""

import os
import shlex
import stat
import threading
import time
from pathlib import Path

import pytest

from agent.delegation_context import delegated_child_context
from tools.environments.base import BaseEnvironment
from tools.environments.local import LocalEnvironment


class _FakeProcess:
    stdout = None
    pid = None
    returncode = 0

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class _FakeNonLocalEnvironment(BaseEnvironment):
    """Minimal non-local backend: shell commands are remote-side operations."""

    def __init__(self, snapshot_path: str):
        super().__init__(cwd="/remote/work", timeout=10)
        self._snapshot_path = snapshot_path
        self._snapshot_ready = True
        self.seed_commands: list[str] = []

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        self.seed_commands.append(cmd_string)
        return _FakeProcess()

    def _wait_for_process(self, proc, **kwargs):
        return {"output": "", "returncode": 0}

    def cleanup(self):
        pass


def test_delegated_child_snapshot_copy_is_created_0600_before_use(tmp_path: Path):
    parent_snapshot = tmp_path / "parent-snapshot.sh"
    parent_snapshot.write_text('declare -x SYNTHETIC_TOKEN="fixture"\n', encoding="utf-8")
    parent_snapshot.chmod(0o600)

    env = LocalEnvironment.__new__(LocalEnvironment)
    BaseEnvironment.__init__(env, cwd=str(tmp_path), timeout=10)
    env._snapshot_path = str(parent_snapshot)
    env._snapshot_ready = True

    old_umask = os.umask(0)
    try:
        with delegated_child_context("child-mode-check"):
            child_snapshot = Path(env._snapshot_path_for_current_context())
    finally:
        os.umask(old_umask)

    try:
        mode = stat.S_IMODE(child_snapshot.stat().st_mode)
        assert mode == 0o600
        assert child_snapshot.read_text(encoding="utf-8") == parent_snapshot.read_text(encoding="utf-8")
    finally:
        env.cleanup()


def test_delegated_child_empty_snapshot_fallback_is_created_0600(tmp_path: Path):
    missing_parent_snapshot = tmp_path / "missing-parent-snapshot.sh"

    env = LocalEnvironment.__new__(LocalEnvironment)
    BaseEnvironment.__init__(env, cwd=str(tmp_path), timeout=10)
    env._snapshot_path = str(missing_parent_snapshot)
    env._snapshot_ready = True

    old_umask = os.umask(0)
    try:
        with delegated_child_context("child-empty-mode-check"):
            child_snapshot = Path(env._snapshot_path_for_current_context())
    finally:
        os.umask(old_umask)

    try:
        mode = stat.S_IMODE(child_snapshot.stat().st_mode)
        assert mode == 0o600
        assert child_snapshot.read_bytes() == b""
    finally:
        env.cleanup()


def test_delegated_child_completion_never_clobbers_parent_cwd_update(tmp_path: Path, monkeypatch):
    parent_before = tmp_path / "parent-before"
    parent_after = tmp_path / "parent-after"
    child_dir = tmp_path / "child-dir"
    for directory in (parent_before, parent_after, child_dir):
        directory.mkdir()

    env = LocalEnvironment.__new__(LocalEnvironment)
    BaseEnvironment.__init__(env, cwd=str(parent_before), timeout=10)
    env._snapshot_ready = False

    def fake_run_bash(cmd_string, *, login=False, timeout=120, stdin_data=None):
        return object()

    def fake_wait_for_process(proc, **kwargs):
        # Simulate a genuine parent command updating the shared environment cwd
        # while this delegated-child command is still in flight.
        env.cwd = str(parent_after)
        marker = env._cwd_marker
        return {"output": f"child output\n{marker}{child_dir}{marker}\n", "returncode": 0}

    monkeypatch.setattr(env, "_run_bash", fake_run_bash)
    monkeypatch.setattr(env, "_wait_for_process", fake_wait_for_process)

    with delegated_child_context("cwd-scope"):
        result = env.execute("cd child-dir")
        assert result["returncode"] == 0

    assert env.cwd == str(parent_after)

    def fake_wait_echo_pwd(proc, **kwargs):
        marker = env._cwd_marker
        return {"output": f"{marker}{child_dir}{marker}\n", "returncode": 0}

    monkeypatch.setattr(env, "_wait_for_process", fake_wait_echo_pwd)
    with delegated_child_context("cwd-scope"):
        result = env.execute("pwd")

    assert result["cwd"] == str(child_dir)
    assert env.cwd == str(parent_after)


def test_real_local_child_finishing_after_parent_cd_preserves_parent_cwd(tmp_path: Path):
    parent_before = tmp_path / "parent-before-real"
    parent_after = tmp_path / "parent-after-real"
    child_dir = tmp_path / "child-dir-real"
    for directory in (parent_before, parent_after, child_dir):
        directory.mkdir()
    started = tmp_path / "child-started"
    release = tmp_path / "release-child"

    env = LocalEnvironment(cwd=str(parent_before), timeout=10)
    child_result: dict[str, dict] = {}
    child_error: list[BaseException] = []

    def run_child() -> None:
        try:
            with delegated_child_context("real-cwd-scope"):
                child_result["first"] = env.execute(
                    f"cd {shlex.quote(str(child_dir))}; "
                    f"touch {shlex.quote(str(started))}; "
                    f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.05; done",
                    timeout=10,
                )
        except BaseException as exc:
            child_error.append(exc)

    thread = threading.Thread(target=run_child)
    try:
        thread.start()
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists()

        parent = env.execute(f"cd {shlex.quote(str(parent_after))}", timeout=10)
        assert parent["returncode"] == 0, parent["output"]
        assert env.cwd == str(parent_after)

        release.touch()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert child_error == []
        assert child_result["first"]["returncode"] == 0, child_result["first"]["output"]
        assert env.cwd == str(parent_after)

        with delegated_child_context("real-cwd-scope"):
            child_pwd = env.execute("pwd", timeout=10)
        assert child_pwd["returncode"] == 0, child_pwd["output"]
        assert child_pwd["output"].strip() == str(child_dir)
        assert env.cwd == str(parent_after)
    finally:
        if thread.is_alive():
            release.touch()
            thread.join(timeout=2)
        env.cleanup()


def test_non_local_child_snapshot_seed_runs_in_target_not_local_filesystem(tmp_path: Path):
    remote_root = tmp_path / "remote-path-that-python-must-not-touch"
    remote_snapshot = remote_root / "snap.sh"
    env = _FakeNonLocalEnvironment(str(remote_snapshot))

    with delegated_child_context("remote-copy-scope"):
        child_snapshot = env._snapshot_path_for_current_context()

    assert env.seed_commands, "non-local seeding must execute a target-side shell command"
    seed_command = env.seed_commands[0]
    assert str(remote_snapshot) in seed_command
    assert child_snapshot in seed_command
    assert not remote_root.exists()


def test_child_snapshot_seed_failure_aborts_before_user_command(tmp_path: Path, monkeypatch):
    env = LocalEnvironment.__new__(LocalEnvironment)
    BaseEnvironment.__init__(env, cwd=str(tmp_path), timeout=10)
    env._snapshot_path = str(tmp_path / "parent-snapshot.sh")
    env._snapshot_ready = True

    def fail_seed(base_path: str, child_path: str) -> None:
        raise OSError("synthetic seed failure")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("user command must not run without seeded child snapshot")

    monkeypatch.setattr(env, "_seed_child_snapshot", fail_seed)
    monkeypatch.setattr(env, "_run_bash", unexpected_run)

    with delegated_child_context("seed-fail-scope"):
        with pytest.raises(RuntimeError, match="Could not seed"):
            env.execute("printf should-not-run")


def test_local_child_snapshot_seed_failure_removes_created_target(tmp_path: Path, monkeypatch):
    parent_snapshot = tmp_path / "parent-snapshot.sh"
    parent_snapshot.write_text('declare -x SYNTHETIC_VALUE="fixture"\n', encoding="utf-8")
    child_snapshot = tmp_path / "child-snapshot.sh"

    def fail_fchmod(fd: int, mode: int) -> None:
        raise OSError("synthetic fchmod failure")

    monkeypatch.setattr("tools.environments.local.os.fchmod", fail_fchmod)

    env = LocalEnvironment.__new__(LocalEnvironment)
    BaseEnvironment.__init__(env, cwd=str(tmp_path), timeout=10)

    with pytest.raises(OSError, match="synthetic fchmod failure"):
        env._seed_child_snapshot(str(parent_snapshot), str(child_snapshot))

    assert not child_snapshot.exists()
