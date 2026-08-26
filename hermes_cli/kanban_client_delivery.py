"""Passive Kanban notification delivery between gateway and local chat clients.

The transactional outbox remains authoritative.  This module only carries exact
outbox identities from the gateway process to TUI-gateway processes.  Delivery
uses a per-process loopback datagram registration; reconnect recovery always
reads SQLite, so a dropped datagram cannot lose a notification.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Optional

_MAX_DATAGRAM = 64 * 1024
_RUNTIME_DIR = "kanban-client-notify"


def hermes_root(home: Optional[Path] = None) -> Path:
    value = Path(home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return value.parent.parent if value.parent.name == "profiles" else value


def profile_home(profile: str, *, root: Optional[Path] = None) -> Path:
    base = hermes_root(root)
    name = str(profile or "default").strip() or "default"
    return base if name == "default" else base / "profiles" / name


def _registry_dir(profile: str, *, root: Optional[Path] = None) -> Path:
    return profile_home(profile, root=root) / _RUNTIME_DIR


class NotificationSignalListener:
    """One process-local datagram endpoint advertised by a small descriptor."""

    def __init__(self, callback: Callable[[dict[str, Any]], None], *, home: Optional[Path] = None) -> None:
        self._callback = callback
        self._home = Path(home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
        self._profile = "default" if self._home.parent.name != "profiles" else self._home.name
        self._dir = _registry_dir(self._profile, root=hermes_root(self._home))
        self._dir.mkdir(parents=True, exist_ok=True)
        with __import__("contextlib").suppress(OSError):
            os.chmod(self._dir, 0o700)
        self._token = f"{os.getpid()}-{secrets.token_hex(6)}"
        self._descriptor = self._dir / f"{self._token}.json"
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("127.0.0.1", 0))
        host, port = self._socket.getsockname()
        self._address = (host, port)
        tmp = self._descriptor.with_suffix(".tmp")
        tmp.write_text(json.dumps({"host": host, "port": port, "pid": os.getpid()}), encoding="utf-8")
        os.replace(tmp, self._descriptor)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="kanban-client-notify", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                raw, _ = self._socket.recvfrom(_MAX_DATAGRAM)
            except OSError:
                return
            if self._closed.is_set():
                return
            try:
                payload = json.loads(raw.decode("utf-8"))
                if isinstance(payload, dict):
                    self._callback(payload)
            except Exception:
                continue

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # Wake the blocking recv without a timer/poll loop.
        try:
            wake = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            wake.sendto(b"{}", self._address)
            wake.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)
        try:
            self._socket.close()
        except OSError:
            pass
        self._descriptor.unlink(missing_ok=True)


def publish_profile_signal(profile: str, payload: dict[str, Any], *, root: Optional[Path] = None) -> int:
    """Best-effort immediate fanout; durable replay covers absent/dead clients."""
    directory = _registry_dir(profile, root=root)
    try:
        descriptors = list(directory.glob("*.json"))
    except OSError:
        return 0
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_DATAGRAM:
        raise ValueError("Kanban client notification signal is too large")
    sent = 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for descriptor in descriptors:
            try:
                target = json.loads(descriptor.read_text(encoding="utf-8"))
                host = str(target.get("host") or "")
                port = int(target.get("port") or 0)
                if host != "127.0.0.1" or port < 1:
                    raise ValueError("invalid endpoint")
                sock.sendto(encoded, (host, port))
                sent += 1
            except Exception:
                descriptor.unlink(missing_ok=True)
    finally:
        sock.close()
    return sent
