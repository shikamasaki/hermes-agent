"""Notifier production path never falls back to board polling."""

import asyncio
from unittest.mock import MagicMock, patch

from gateway.config import Platform
from gateway.run import GatewayRunner


def _make_runner(with_adapter=False):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: MagicMock()} if with_adapter else {}
    runner._kanban_sub_fail_counts = {}
    return runner


def test_notifier_without_signal_bus_does_not_poll_boards():
    runner = _make_runner(with_adapter=True)
    board_scans = []
    sleep_calls = []

    async def fake_sleep(delay):
        sleep_calls.append(delay)

    import hermes_cli.kanban_db as _kb

    with patch.object(
        _kb, "list_boards",
        side_effect=lambda *a, **kw: board_scans.append(True) or [],
    ):
        with patch("asyncio.sleep", side_effect=fake_sleep):
            asyncio.run(runner._kanban_notifier_watcher())

    assert board_scans == []
    assert sleep_calls == []
