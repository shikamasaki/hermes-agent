"""Regression coverage for delegated child terminal env isolation."""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


def _make_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    parent_home = tmp_path / "parent_home"
    parent_home.mkdir()
    terminal_tmp = tmp_path / "terminal_tmp"
    terminal_tmp.mkdir()
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(parent_home))
    monkeypatch.setenv("TERMINAL_TEMP_DIR", str(terminal_tmp))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)

    from tools.environments.local import LocalEnvironment

    return LocalEnvironment(cwd=str(cwd), timeout=10), parent_home, cwd


def _home(result: dict) -> str:
    assert result["returncode"] == 0, result["output"]
    return result["output"].strip()


def test_child_exported_hermes_home_does_not_rewrite_parent_snapshot(monkeypatch, tmp_path):
    from agent.delegation_context import delegated_child_context

    env, parent_home, _cwd = _make_env(monkeypatch, tmp_path)
    child_home = tmp_path / "child_home"
    child_home.mkdir()
    child_home_later = tmp_path / "child_home_later"
    child_home_later.mkdir()
    child_cwd = tmp_path / "child_cwd"
    child_cwd.mkdir()
    try:
        assert _home(env.execute('printf "%s" "$HERMES_HOME"')) == str(parent_home)
        assert _home(env.execute('pwd -P')) == str(_cwd)

        with delegated_child_context("child-a"):
            assert _home(env.execute(f'export HERMES_HOME={child_home}; cd {child_cwd}; printf "%s:%s" "$HERMES_HOME" "$(pwd -P)"')) == f"{child_home}:{child_cwd}"

        assert _home(env.execute('printf "%s" "$HERMES_HOME"')) == str(parent_home)
        assert _home(env.execute('pwd -P')) == str(_cwd)

        with delegated_child_context("child-a"):
            assert _home(env.execute('printf "%s:%s" "$HERMES_HOME" "$(pwd -P)"')) == f"{child_home}:{child_cwd}"
            assert _home(env.execute(f'export HERMES_HOME={child_home_later}; printf "%s" "$HERMES_HOME"')) == str(child_home_later)

        assert _home(env.execute('printf "%s" "$HERMES_HOME"')) == str(parent_home)
        with delegated_child_context("child-a"):
            assert _home(env.execute('printf "%s" "$HERMES_HOME"')) == str(child_home_later)
    finally:
        env.cleanup()


def test_parallel_sibling_child_exports_do_not_cross_or_poison_parent(monkeypatch, tmp_path):
    from agent.delegation_context import delegated_child_context

    env, parent_home, _cwd = _make_env(monkeypatch, tmp_path)
    child_a_home = tmp_path / "child_a_home"
    child_b_home = tmp_path / "child_b_home"
    child_a_home.mkdir()
    child_b_home.mkdir()
    barrier = threading.Barrier(3)
    results: dict[str, str] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run_child(name: str, home: Path) -> None:
        try:
            with delegated_child_context(name):
                barrier.wait(timeout=10)
                first = _home(env.execute(f'export HERMES_HOME={home}; printf "%s" "$HERMES_HOME"'))
                second = _home(env.execute('printf "%s" "$HERMES_HOME"'))
            with lock:
                results[name] = f"{first}\n{second}"
        except BaseException as exc:  # pragma: no cover - surfaced below
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=run_child, args=("child-a", child_a_home)),
        threading.Thread(target=run_child, args=("child-b", child_b_home)),
    ]
    try:
        for thread in threads:
            thread.start()
        barrier.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=20)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert results["child-a"].splitlines() == [str(child_a_home), str(child_a_home)]
        assert results["child-b"].splitlines() == [str(child_b_home), str(child_b_home)]
        assert _home(env.execute('printf "%s" "$HERMES_HOME"')) == str(parent_home)
    finally:
        env.cleanup()


def test_child_exception_and_interrupted_export_do_not_poison_parent(monkeypatch, tmp_path):
    from agent.delegation_context import delegated_child_context
    from tools.interrupt import set_interrupt

    env, parent_home, _cwd = _make_env(monkeypatch, tmp_path)
    child_home = tmp_path / "child_exception_home"
    child_home.mkdir()
    started = tmp_path / "started"
    finished = threading.Event()
    interrupted: dict[str, dict] = {}
    errors: list[BaseException] = []

    def interrupted_child() -> None:
        try:
            with delegated_child_context("child-interrupted"):
                interrupted["result"] = env.execute(
                    f'export HERMES_HOME={child_home}; touch {started}; sleep 30',
                    timeout=15,
                )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            finished.set()

    try:
        with pytest.raises(RuntimeError, match="intentional child exception"):
            with delegated_child_context("child-exception"):
                assert _home(env.execute(f'export HERMES_HOME={child_home}; printf "%s" "$HERMES_HOME"')) == str(child_home)
                raise RuntimeError("intentional child exception")

        assert _home(env.execute('printf "%s" "$HERMES_HOME"')) == str(parent_home)

        thread = threading.Thread(target=interrupted_child)
        thread.start()
        for _ in range(100):
            if started.exists():
                break
            finished.wait(timeout=0.05)
        assert started.exists()
        assert thread.ident is not None
        set_interrupt(True, thread.ident, reason="test interrupt")
        try:
            assert finished.wait(timeout=20)
        finally:
            set_interrupt(False, thread.ident)
        thread.join(timeout=1)

        assert errors == []
        assert interrupted["result"]["returncode"] == 130, interrupted["result"]["output"]
        assert _home(env.execute('printf "%s" "$HERMES_HOME"')) == str(parent_home)
    finally:
        env.cleanup()


def test_parent_reuses_prepopulated_snapshot_without_child_home_contamination(monkeypatch, tmp_path):
    from agent.delegation_context import delegated_child_context

    env, parent_home, _cwd = _make_env(monkeypatch, tmp_path)
    child_home = tmp_path / "child_prepopulated_home"
    child_home.mkdir()
    preexisting = tmp_path / "preexisting_home"
    preexisting.mkdir()
    Path(env._snapshot_path).write_text(
        f'declare -x HERMES_HOME="{preexisting}"\n'
        'declare -x HERMES_PARENT_SENTINEL="kept"\n',
        encoding="utf-8",
    )
    env._snapshot_ready = True
    try:
        assert _home(env.execute('printf "%s:%s" "$HERMES_HOME" "$HERMES_PARENT_SENTINEL"')) == f"{preexisting}:kept"
        with delegated_child_context("child-prepopulated"):
            assert _home(env.execute(f'export HERMES_HOME={child_home}; printf "%s" "$HERMES_HOME"')) == str(child_home)
        assert _home(env.execute('printf "%s:%s" "$HERMES_HOME" "$HERMES_PARENT_SENTINEL"')) == f"{preexisting}:kept"
    finally:
        env.cleanup()
