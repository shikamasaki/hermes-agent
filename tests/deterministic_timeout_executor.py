"""Deterministic Future wrappers for timeout tests."""

from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any, Callable


class StartOnResultFuture:
    """Minimal Future proxy used by timeout tests.

    Deliberately not a ``Future`` subclass: it only supports the call sites
    under test (``result``, ``add_done_callback``, and attribute forwarding).
    """

    def __init__(
        self,
        future: Future,
        *,
        allow_start: threading.Event,
        started: threading.Event,
        start_wait_timeout: float = 5.0,
    ) -> None:
        self._future = future
        self._allow_start = allow_start
        self._started = started
        self._start_wait_timeout = start_wait_timeout
        self._start_lock = threading.Lock()
        self._start_waited = False

    def result(self, timeout: float | None = None) -> Any:
        if not self._start_waited:
            with self._start_lock:
                if not self._start_waited:
                    self._allow_start.set()
                    if not self._started.wait(timeout=self._start_wait_timeout):
                        if not self._future.done():
                            raise TimeoutError("delayed test worker did not start")
                    self._start_waited = True
        return self._future.result(timeout=timeout)

    def add_done_callback(self, fn: Callable[[Future], Any]) -> None:
        self._future.add_done_callback(fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._future, name)


class DelayedStartExecutor:
    """Small test executor whose workers wait behind a gate before running."""

    def __init__(
        self,
        *,
        synchronize_result_start: bool,
        started_event: threading.Event | None = None,
        initializer: Callable[..., Any] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        self.synchronize_result_start = synchronize_result_start
        self.allow_start = threading.Event()
        self.submitted = threading.Event()
        self.started = started_event or threading.Event()
        self._auto_mark_started = started_event is None
        self._initializer = initializer
        self._initargs = initargs
        self._threads: list[threading.Thread] = []

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        future: Future = Future()

        def runner() -> None:
            self.submitted.set()
            self.allow_start.wait()
            if self._initializer is not None:
                self._initializer(*self._initargs)
            if self._auto_mark_started:
                self.started.set()
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

        thread = threading.Thread(target=runner, daemon=True)
        self._threads.append(thread)
        thread.start()
        if not self.synchronize_result_start:
            return future
        return StartOnResultFuture(
            future,
            allow_start=self.allow_start,
            started=self.started,
        )

    def shutdown(self, wait: bool = True, **_kwargs: Any) -> None:
        if wait:
            self.allow_start.set()
            for thread in self._threads:
                thread.join(timeout=5)
