"""Passive Kanban inbox JSON-RPC methods."""

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("kanban.notifications.subscribe")
def _(rid, params: dict) -> dict:
    try:
        from tui_gateway.kanban_notifications import ensure_listener, subscribe

        transport = current_transport() or _stdio_transport
        ensure_listener()
        count = subscribe(
            transport,
            surface=str(params.get("surface") or ""),
            session_id=str(params.get("session_id") or ""),
        )
        return _ok(rid, {"subscribed": True, "replayed": count})
    except Exception as exc:
        return _err(rid, 5095, str(exc))


@method("kanban.notifications.ack")
def _(rid, params: dict) -> dict:
    try:
        from tui_gateway.kanban_notifications import acknowledge

        transport = current_transport() or _stdio_transport
        ok = acknowledge(
            transport=transport,
            surface=str(params.get("surface") or ""),
            board=str(params.get("board") or ""),
            outbox_id=int(params.get("outbox_id") or 0),
            delivery_key=str(params.get("delivery_key") or ""),
        )
        return _ok(rid, {"acknowledged": ok})
    except Exception as exc:
        return _err(rid, 5096, str(exc))


def register(server) -> None:
    _registry.install(server)
