"""Post-commit Gateway control-socket signaling for Kanban outbox rows."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gateway import control_socket as control
from hermes_cli import kanban_db as kb
from hermes_state import SessionDB


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_home = home / "profiles" / "chief"
    profile_home.mkdir(parents=True)
    db = SessionDB(db_path=profile_home / "state.db")
    try:
        db.create_session("chief-bot-chat", "tui", profile_name="chief")
        assert db.set_session_title("chief-bot-chat", "Bot Chat")
    finally:
        db.close()
    kb.init_db()
    return home


def _create_owned_task(conn, title: str = "owned") -> str:
    return kb.create_task(
        conn,
        title=title,
        assignee="worker",
        origin_profile="chief",
        origin_session_id="chief-bot-chat",
        board="default",
    )


def test_nested_writes_signal_once_only_after_outer_commit(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, str, dict]] = []

    def record(home: Path, verb: str, payload: dict, **_kwargs) -> bool:
        calls.append((home, verb, payload))
        return True

    monkeypatch.setattr(control, "signal_gateway_control", record)
    with kb.connect() as conn:
        with kb.write_txn(conn):
            task_id = _create_owned_task(conn)
            kb.add_comment(conn, task_id, "chief", "second outbox event")
            assert calls == []

        assert len(calls) == 1
        home, verb, payload = calls[0]
        assert home == kanban_home
        assert verb == "kanban_outbox_ready"
        identities = payload["outbox"]
        assert len(identities) == 2
        assert {item["board"] for item in identities} == {"default"}
        assert {item["task_id"] for item in identities} == {task_id}
        assert {item["event_kind"] for item in identities} == {"created", "commented"}
        assert all(item["outbox_id"] > 0 for item in identities)
        assert all(item["event_id"] > 0 for item in identities)
        assert all(item["delivery_key"] for item in identities)


def test_unowned_task_create_still_wakes_dispatch_after_commit(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(
        control,
        "signal_gateway_control",
        lambda _home, _verb, payload, **_kwargs: calls.append(payload) or True,
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="dispatch without a notification destination",
            assignee="worker",
        )

    assert len(calls) == 1
    assert calls[0]["outbox"] == []
    assert calls[0]["dispatch"] == [
        {
            "board": "default",
            "task_id": task_id,
            "event_id": calls[0]["dispatch"][0]["event_id"],
            "event_kind": "created",
        }
    ]
    assert calls[0]["dispatch"][0]["event_id"] > 0


def test_rollback_sends_no_signal_and_leaves_no_event_or_outbox(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        control,
        "signal_gateway_control",
        lambda *_args, **_kwargs: calls.append(object()) or True,
    )
    with kb.connect() as conn:
        with pytest.raises(RuntimeError, match="abort outer transaction"):
            with kb.write_txn(conn):
                task_id = _create_owned_task(conn, "rolled back")
                raise RuntimeError("abort outer transaction")
        assert calls == []
        assert kb.get_task(conn, task_id) is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM kanban_notification_outbox WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0


def test_unavailable_gateway_never_rolls_back_committed_outbox(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*_args, **_kwargs):
        raise OSError("gateway exited before signal")

    monkeypatch.setattr(control, "signal_gateway_control", unavailable)
    with kb.connect() as conn:
        task_id = _create_owned_task(conn, "survives lost signal")

    with kb.connect() as reopened:
        task = kb.get_task(reopened, task_id)
        assert task is not None
        rows = kb.list_notification_outbox(
            reopened, profile="chief", board="default", include_delivered=True
        )
        assert [row["task_id"] for row in rows] == [task_id]


def test_separate_connections_signal_only_their_committed_identity_batch(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads: list[dict] = []
    monkeypatch.setattr(
        control,
        "signal_gateway_control",
        lambda _home, _verb, payload, **_kwargs: payloads.append(payload) or True,
    )
    with kb.connect() as first, kb.connect() as second:
        first_task = _create_owned_task(first, "first connection")
        second_task = _create_owned_task(second, "second connection")

    assert len(payloads) == 2
    assert [{item["task_id"] for item in payload["outbox"]} for payload in payloads] == [
        {first_task},
        {second_task},
    ]


def test_generic_control_signal_carries_payload_over_real_socket(kanban_home: Path) -> None:
    received: list[dict] = []

    async def scenario() -> bool:
        server = control.GatewayControlServer(
            kanban_home,
            verb_handlers={
                "kanban_outbox_ready": lambda payload: received.append(payload)
                or {"accepted": True}
            },
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: control.signal_gateway_control(
                    kanban_home,
                    "kanban_outbox_ready",
                    {
                        "outbox": [
                            {
                                "board": "default",
                                "outbox_id": 41,
                                "event_id": 17,
                                "event_kind": "completed",
                                "task_id": "t_signal",
                                "delivery_key": "chief:default:41",
                            }
                        ]
                    },
                    timeout=0.5,
                ),
            )
        finally:
            await server.stop()

    assert asyncio.run(scenario()) is True
    assert received == [
        {
            "outbox": [
                {
                    "board": "default",
                    "outbox_id": 41,
                    "event_id": 17,
                    "event_kind": "completed",
                    "task_id": "t_signal",
                    "delivery_key": "chief:default:41",
                }
            ]
        }
    ]


def test_generic_control_signal_is_bounded_and_failure_safe(kanban_home: Path) -> None:
    assert control.signal_gateway_control(
        kanban_home,
        "kanban_outbox_ready",
        {"outbox": []},
        timeout=0.01,
    ) is False
