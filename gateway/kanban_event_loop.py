"""Event-driven Kanban wake coordination for the Gateway.

The transactional outbox is the durability boundary.  This module only turns
bounded post-commit control-socket hints into asyncio wakeups; duplicate hints
are coalesced by their durable identities and no background polling thread is
created.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KanbanSignalBatch:
    boards: set[str]
    outbox_ids: set[int]
    task_ids: set[str]
    outbox_by_board: dict[str, set[int]]


class KanbanSignalBus:
    """Thread-safe bridge from synchronous control handlers to asyncio.

    ``GatewayControlServer`` executes handlers in an executor.  The lock guards
    the small identity sets there; ``Event.set`` is always scheduled onto the
    owning loop.  Consumers get independent notification and dispatch streams
    so one watcher cannot steal the other's wakeup.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._lock = threading.Lock()
        self._pending = {
            "notification": {"boards": set(), "outbox_ids": set(), "task_ids": set(), "outbox_by_board": {}},
            "dispatch": {"boards": set(), "outbox_ids": set(), "task_ids": set(), "outbox_by_board": {}},
        }
        self._events = {
            "notification": asyncio.Event(),
            "dispatch": asyncio.Event(),
        }

    def handle_signal(self, payload: dict[str, Any]) -> dict[str, int]:
        rows = payload.get("outbox") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("payload.outbox must be a list")
        accepted: list[tuple[str, int, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            board = str(row.get("board") or "").strip()
            task_id = str(row.get("task_id") or "").strip()
            raw_outbox_id = row.get("outbox_id")
            try:
                outbox_id = int(str(raw_outbox_id))
            except (TypeError, ValueError):
                continue
            if not board or not task_id or outbox_id < 1:
                continue
            accepted.append((board, outbox_id, task_id))
        dispatch_rows = payload.get("dispatch", []) if isinstance(payload, dict) else []
        if not isinstance(dispatch_rows, list):
            raise ValueError("payload.dispatch must be a list")
        dispatch_accepted: list[tuple[str, str]] = []
        for row in dispatch_rows:
            if not isinstance(row, dict):
                continue
            board = str(row.get("board") or "").strip()
            task_id = str(row.get("task_id") or "").strip()
            try:
                event_id = int(str(row.get("event_id")))
            except (TypeError, ValueError):
                continue
            if board and task_id and event_id > 0:
                dispatch_accepted.append((board, task_id))
        if not accepted and not dispatch_accepted:
            return {"accepted": 0}
        with self._lock:
            notification = self._pending["notification"]
            dispatch = self._pending["dispatch"]
            for board, outbox_id, task_id in accepted:
                for stream in (notification, dispatch):
                    stream["boards"].add(board)
                    stream["outbox_ids"].add(outbox_id)
                    stream["task_ids"].add(task_id)
                    stream["outbox_by_board"].setdefault(board, set()).add(outbox_id)
            for board, task_id in dispatch_accepted:
                dispatch["boards"].add(board)
                dispatch["task_ids"].add(task_id)
        if accepted:
            self._loop.call_soon_threadsafe(self._events["notification"].set)
        if accepted or dispatch_accepted:
            self._loop.call_soon_threadsafe(self._events["dispatch"].set)
        return {"accepted": len(accepted) + len(dispatch_accepted)}

    async def _next(self, stream_name: str) -> KanbanSignalBatch:
        event = self._events[stream_name]
        await event.wait()
        with self._lock:
            stream = self._pending[stream_name]
            batch = KanbanSignalBatch(
                boards=set(stream["boards"]),
                outbox_ids=set(stream["outbox_ids"]),
                task_ids=set(stream["task_ids"]),
                outbox_by_board={
                    board: set(ids)
                    for board, ids in stream["outbox_by_board"].items()
                },
            )
            stream["boards"].clear()
            stream["outbox_ids"].clear()
            stream["task_ids"].clear()
            stream["outbox_by_board"].clear()
            event.clear()
        return batch

    async def next_batch(self) -> KanbanSignalBatch:
        return await self._next("notification")

    async def next_dispatch_batch(self) -> KanbanSignalBatch:
        return await self._next("dispatch")

    def empty(self) -> bool:
        """Whether the notification stream has no pending signal identities."""
        with self._lock:
            return not self._pending["notification"]["outbox_ids"]
