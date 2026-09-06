"""Newline in bridged session env must not become shell code via the snapshot.

Regression for issue #71296: bash 3.2 ``export -p`` prints a value containing
a newline as a multi-line ``declare -x NAME="…`` block. The old line-based
``grep -vE`` filter removed only the opener; continuation lines (e.g.
``curl … | bash #`` smuggled into a Matrix room/display name) persisted into
the shared terminal snapshot and executed on the next ``source``, with stdout
discarded by the wrapper.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tools.environments.base import _export_dump_excluding_session_vars


def _bash() -> str:
    return "/bin/bash" if os.path.exists("/bin/bash") else "bash"


def _run_dump_and_source(
    *,
    tmp_path: Path,
    env_name: str,
    env_value: str,
    marker: Path,
) -> subprocess.CompletedProcess:
    snap = tmp_path / "snap"
    dump = _export_dump_excluding_session_vars(shlex.quote(str(snap)))
    q_snap = shlex.quote(str(snap))
    q_marker = shlex.quote(str(marker))
    script = f"""
set -e
export {env_name}
{dump}
if grep -qE 'pwned|touch |{env_name}' {q_snap}; then
  echo "LEAKED_INTO_SNAPSHOT" >&2
  exit 2
fi
bash -c 'source {q_snap} >/dev/null 2>&1 || true'
if [ -e {q_marker} ]; then
  echo "PAYLOAD_EXECUTED" >&2
  exit 3
fi
if ! grep -qE '^declare -x PATH=' {q_snap}; then
  echo "PATH_MISSING" >&2
  exit 4
fi
"""
    env = os.environ.copy()
    env[env_name] = env_value
    return subprocess.run(
        [_bash(), "-c", script],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_multiline_session_chat_name_not_executed_via_snapshot(tmp_path: Path):
    """Continuation lines of HERMES_SESSION_CHAT_NAME must not run on source."""
    marker = tmp_path / "pwned"
    chat_name = f"demo\ntouch {marker} #"
    proc = _run_dump_and_source(
        tmp_path=tmp_path,
        env_name="HERMES_SESSION_CHAT_NAME",
        env_value=chat_name,
        marker=marker,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_multiline_session_user_name_not_executed_via_snapshot(tmp_path: Path):
    """Same hole via HERMES_SESSION_USER_NAME (display-name path)."""
    marker = tmp_path / "pwned_user"
    user_name = f"alice\ntouch {marker} #"
    proc = _run_dump_and_source(
        tmp_path=tmp_path,
        env_name="HERMES_SESSION_USER_NAME",
        env_value=user_name,
        marker=marker,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_delegated_child_marker_not_persisted_into_terminal_snapshot(tmp_path: Path):
    """delegate_task lineage is per-command state, not shared shell state."""
    marker = tmp_path / "unused"
    proc = _run_dump_and_source(
        tmp_path=tmp_path,
        env_name="HERMES_DELEGATED_CHILD_CONTEXT",
        env_value="1",
        marker=marker,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_polluted_snapshot_marker_does_not_override_clean_process_env(tmp_path: Path):
    """Existing polluted snapshots must not mark later parent commands as child."""
    from tools.environments.local import LocalEnvironment

    snap = tmp_path / "snap.sh"
    snap.write_text('declare -x HERMES_DELEGATED_CHILD_CONTEXT="1"\n', encoding="utf-8")

    env = LocalEnvironment.__new__(LocalEnvironment)
    env._snapshot_ready = True
    env._session_id = "testsession1"
    env._cwd_marker = "__HERMES_CWD_testsession1__"
    env._snapshot_path = str(snap)
    env._snapshot_passthrough_names = set()

    wrapped = env._wrap_command(
        'python -c "import os; print(os.environ.get(\'HERMES_DELEGATED_CHILD_CONTEXT\') is not None)"',
        str(tmp_path),
    )
    clean_env = dict(os.environ)
    clean_env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)

    proc = subprocess.run(
        [_bash(), "-c", wrapped],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=clean_env,
        timeout=30,
    )

    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert "False" in proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_polluted_snapshot_does_not_remove_true_child_process_marker(tmp_path: Path):
    """A true child process keeps its trusted process-env marker after source."""
    from tools.environments.local import LocalEnvironment

    snap = tmp_path / "snap.sh"
    snap.write_text("unset HERMES_DELEGATED_CHILD_CONTEXT\n", encoding="utf-8")

    env = LocalEnvironment.__new__(LocalEnvironment)
    env._snapshot_ready = True
    env._session_id = "testsession2"
    env._cwd_marker = "__HERMES_CWD_testsession2__"
    env._snapshot_path = str(snap)
    env._snapshot_passthrough_names = set()

    wrapped = env._wrap_command(
        'python -c "import os; print(os.environ.get(\'HERMES_DELEGATED_CHILD_CONTEXT\') is not None)"',
        str(tmp_path),
    )
    child_env = dict(os.environ)
    child_env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"

    proc = subprocess.run(
        [_bash(), "-c", wrapped],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=child_env,
        timeout=30,
    )

    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert "True" in proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot path")
def test_snapshot_cannot_spoof_internal_restore_variables(tmp_path: Path):
    """Snapshot content must not overwrite the saved process-env marker state."""
    from tools.environments.local import LocalEnvironment

    snap = tmp_path / "snap.sh"
    snap.write_text(
        "declare -x _HERMES_RUNTIME_PASSTHROUGH_HERMES_DELEGATED_CHILD_CONTEXT_PRESENT=x\n"
        "declare -x _HERMES_RUNTIME_PASSTHROUGH_HERMES_DELEGATED_CHILD_CONTEXT_VALUE=1\n",
        encoding="utf-8",
    )

    env = LocalEnvironment.__new__(LocalEnvironment)
    env._snapshot_ready = True
    env._session_id = "testsession3"
    env._cwd_marker = "__HERMES_CWD_testsession3__"
    env._snapshot_path = str(snap)
    env._snapshot_passthrough_names = set()

    wrapped = env._wrap_command(
        'python -c "import os; print(os.environ.get(\'HERMES_DELEGATED_CHILD_CONTEXT\') is not None)"',
        str(tmp_path),
    )
    clean_env = dict(os.environ)
    clean_env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)

    proc = subprocess.run(
        [_bash(), "-c", wrapped],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=clean_env,
        timeout=30,
    )

    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert "False" in proc.stdout
