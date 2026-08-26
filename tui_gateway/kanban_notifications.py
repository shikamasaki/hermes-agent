"""Durable passive Kanban inbox exposed over TUI/Desktop JSON-RPC."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_client_delivery import NotificationSignalListener

_VISIBLE_KINDS = frozenset({
    "blocked", "block_loop_detected", "completed", "crashed", "gave_up",
    "review_requested", "timed_out",
})
_SUBSCRIBERS: dict[Any, tuple[str, str, str]] = {}
_SENT: dict[Any, set[str]] = {}
_LOCK = threading.RLock()
_LISTENER: Optional[NotificationSignalListener] = None


def _active_profile() -> str:
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return home.name if home.parent.name == "profiles" else "default"


def _consumer(surface: str, profile: str) -> str:
    normalized = str(surface or "").strip().lower()
    if normalized not in {"desktop", "shared", "tui"}:
        raise ValueError("surface must be desktop, shared, or tui")
    return f"{normalized}:{profile}"


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    try:
        parsed = json.loads(row.get("payload") or "{}")
        if isinstance(parsed, dict):
            detail = parsed
    except (TypeError, ValueError):
        pass
    return {
        "outbox_id": int(row["id"]),
        "delivery_key": str(row["delivery_key"]),
        "delivery_seq": int(row["delivery_seq"]),
        "board": str(row["board"]),
        "task_id": str(row["task_id"]),
        "task_title": str(row.get("task_title") or ""),
        "event_kind": str(row["event_kind"]),
        "created_at": int(row["created_at"]),
        "summary": detail.get("summary"),
        "reason": detail.get("reason"),
        "status": detail.get("status"),
    }


def _frame(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {
            "type": "kanban.notification",
            "session_id": str(row.get("origin_session_id") or ""),
            "payload": _payload(row),
        },
    }


def _load_rows(profile: str, requested: Optional[dict[str, set[int]]] = None, *, consumer: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        boards = kb.list_boards(include_archived=False)
    except Exception:
        boards = [{"slug": kb.DEFAULT_BOARD}]
    for meta in boards:
        board = str(meta.get("slug") or kb.DEFAULT_BOARD)
        ids = requested.get(board) if requested is not None else None
        if requested is not None and not ids:
            continue
        conn = kb.connect(board=board)
        try:
            candidates = (
                kb.get_notification_outbox_by_ids(conn, ids or set())
                if ids is not None
                else kb.list_notification_outbox(
                    conn, profile=profile, board=board, consumer=consumer, limit=500
                )
            )
            task_ids = sorted({str(row.get("task_id") or "") for row in candidates if row.get("task_id")})
            titles: dict[str, str] = {}
            if task_ids:
                placeholders = ",".join("?" for _ in task_ids)
                titles = {
                    str(task["id"]): str(task["title"])
                    for task in conn.execute(
                        f"SELECT id, title FROM tasks WHERE id IN ({placeholders})", task_ids
                    ).fetchall()
                }
            for row in candidates:
                if row.get("platform") != "bot-chat":
                    continue
                if row.get("origin_profile") != profile or not row.get("origin_session_id"):
                    continue
                if row.get("event_kind") not in _VISIBLE_KINDS:
                    continue
                if kb.notification_outbox_acknowledged(conn, int(row["id"]), consumer=consumer):
                    continue
                row["task_title"] = titles.get(str(row.get("task_id") or ""))
                rows.append(row)
        finally:
            conn.close()
    rows.sort(key=lambda row: (int(row.get("created_at") or 0), str(row.get("board") or ""), int(row["id"])))
    return rows


def subscribe(transport: Any, *, surface: str, session_id: str) -> int:
    profile = _active_profile()
    consumer = _consumer(surface, profile)
    canonical_session = str(session_id or "").strip()
    if not canonical_session:
        raise ValueError("session_id is required")
    # The client may name a public session id, but that is only a lookup hint;
    # re-validate it against this profile's canonical Bot Chat before exposing
    # any durable ownership row.
    kb._validate_canonical_bot_chat_origin(profile, canonical_session)
    with _LOCK:
        _SUBSCRIBERS[transport] = (profile, consumer, canonical_session)
        _SENT[transport] = set()
    count = 0
    for row in _load_rows(profile, consumer=consumer):
        if str(row.get("origin_session_id") or "") != canonical_session:
            continue
        key = str(row["delivery_key"])
        if key in _SENT[transport]:
            continue
        if transport.write(_frame(row)):
            _SENT[transport].add(key)
            count += 1
    return count


def unsubscribe(transport: Any) -> None:
    with _LOCK:
        _SUBSCRIBERS.pop(transport, None)
        _SENT.pop(transport, None)


def deliver_signal(payload: dict[str, Any]) -> int:
    requested: dict[str, set[int]] = {}
    for item in payload.get("outbox") or []:
        if not isinstance(item, dict):
            continue
        try:
            board = str(item.get("board") or "").strip()
            outbox_id = int(item.get("outbox_id") or 0)
        except (TypeError, ValueError):
            continue
        if board and outbox_id > 0:
            requested.setdefault(board, set()).add(outbox_id)
    if not requested:
        return 0
    with _LOCK:
        subscribers = list(_SUBSCRIBERS.items())
    sent = 0
    for transport, (profile, consumer, canonical_session) in subscribers:
        for row in _load_rows(profile, requested, consumer=consumer):
            if str(row.get("origin_session_id") or "") != canonical_session:
                continue
            key = str(row["delivery_key"])
            with _LOCK:
                if key in _SENT.setdefault(transport, set()):
                    continue
            try:
                if transport.write(_frame(row)):
                    with _LOCK:
                        _SENT.setdefault(transport, set()).add(key)
                    sent += 1
            except Exception:
                unsubscribe(transport)
                break
    return sent


def acknowledge(*, surface: str, board: str, outbox_id: int, delivery_key: str) -> bool:
    profile = _active_profile()
    consumer = _consumer(surface, profile)
    conn = kb.connect(board=board)
    try:
        rows = kb.get_notification_outbox_by_ids(conn, [outbox_id])
        if len(rows) != 1:
            return False
        row = rows[0]
        if row.get("platform") != "bot-chat" or row.get("origin_profile") != profile:
            return False
        return kb.ack_notification_outbox(
            conn, outbox_id, consumer=consumer, delivery_key=delivery_key
        )
    finally:
        conn.close()


def ensure_listener() -> NotificationSignalListener:
    global _LISTENER
    with _LOCK:
        if _LISTENER is None:
            def _deliver(payload: dict[str, Any]) -> None:
                deliver_signal(payload)

            _LISTENER = NotificationSignalListener(_deliver)
        return _LISTENER
