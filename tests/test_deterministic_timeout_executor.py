"""Contract tests for deterministic timeout executor helpers."""

from __future__ import annotations

import threading

from concurrent.futures import TimeoutError as FuturesTimeoutError

from tests.deterministic_timeout_executor import DelayedStartExecutor


def test_synchronized_delayed_start_future_starts_before_result_timeout() -> None:
    """Document old unsynchronized RED and synchronized start-before-timeout."""
    old_executor = DelayedStartExecutor(synchronize_result_start=False)
    old_future = old_executor.submit(lambda: "unreachable-before-release")
    assert old_executor.submitted.wait(timeout=5)
    try:
        try:
            old_future.result(timeout=0.01)
        except FuturesTimeoutError:
            pass
        else:  # pragma: no cover - documents the deterministic RED contract
            raise AssertionError("unsynchronized future unexpectedly completed")
        assert not old_executor.started.is_set()
    finally:
        old_executor.allow_start.set()
        old_executor.shutdown()

    new_executor = DelayedStartExecutor(synchronize_result_start=True)
    new_future = new_executor.submit(lambda: "started-before-timeout")
    callback_futures = []
    new_future.add_done_callback(callback_futures.append)
    assert new_executor.submitted.wait(timeout=5)
    assert new_future.result(timeout=0.01) == "started-before-timeout"
    assert new_executor.started.is_set()
    assert new_future.done()
    assert callback_futures and callback_futures[0] is not new_future
    assert callback_futures[0].result() == "started-before-timeout"
    assert new_future.result(timeout=0.01) == "started-before-timeout"
    new_executor.shutdown()


def test_synchronized_future_waits_for_external_started_event_before_timeout() -> None:
    """External start events must be observed before delegating result timeout."""
    publish_start = threading.Event()
    external_started = threading.Event()
    caller_entered = threading.Event()
    caller_done = threading.Event()
    result_box: dict[str, object] = {}
    executor = DelayedStartExecutor(
        synchronize_result_start=True,
        started_event=external_started,
    )

    def submitted_fn() -> str:
        assert publish_start.wait(timeout=5)
        external_started.set()
        return "external-started-before-result-timeout"

    future = executor.submit(submitted_fn)
    assert executor.submitted.wait(timeout=5)

    def call_result() -> None:
        caller_entered.set()
        try:
            result_box["result"] = future.result(timeout=0.01)
        except BaseException as exc:  # pragma: no cover - failure detail
            result_box["exc"] = exc
        finally:
            caller_done.set()

    caller = threading.Thread(target=call_result)
    caller.start()
    try:
        assert caller_entered.wait(timeout=5)
        assert not caller_done.wait(timeout=0.05)
        assert not result_box
        publish_start.set()
        assert caller_done.wait(timeout=5)
        assert result_box == {
            "result": "external-started-before-result-timeout"
        }
    finally:
        publish_start.set()
        executor.shutdown()
        caller.join(timeout=5)
