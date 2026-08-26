import hashlib
import hmac
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kdb
from hermes_cli.github_sync import (
    GitHubIntakeError,
    GitHubIntakeService,
    GitHubProjector,
    initialize_github_sync,
)


def _payload(*, sender="shikamasaki", installation=77, repo="shikamasaki/demo", assignee="shikamasaki"):
    owner, name = repo.split("/", 1)
    return {
        "action": "assigned",
        "installation": {"id": installation},
        "repository": {
            "id": 101,
            "node_id": "R_repo",
            "name": name,
            "full_name": repo,
            "owner": {"login": owner, "type": "User"},
            "html_url": f"https://github.com/{repo}",
        },
        "issue": {
            "id": 202,
            "node_id": "I_issue",
            "number": 20,
            "title": "Implement sync",
            "state": "open",
            "html_url": f"https://github.com/{repo}/issues/20",
            "assignees": [{"login": assignee}],
        },
        "assignee": {"login": assignee},
        "sender": {"login": sender},
    }


def _headers(payload, secret="whsec", delivery="delivery-1", event="issues"):
    raw = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return raw, {
        "X-GitHub-Delivery": delivery,
        "X-GitHub-Event": event,
        "X-Hub-Signature-256": f"sha256={signature}",
    }


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(kdb, "kanban_home", lambda: tmp_path)
    c = kdb.connect(tmp_path / "kanban.db")
    initialize_github_sync(c)
    yield c
    c.close()


@pytest.fixture
def route(tmp_path):
    return {
        "installation_id": 77,
        "owner": "shikamasaki",
        "repositories": ["demo"],
        "assignee": "shikamasaki",
        "profile": "personal",
        "board": "default",
        "tenant": "personal",
        "webhook_secret_ref": "personal-webhook",
        "project_token_ref": "personal-project-token",
        "project": {
            "id": "PVT_project",
            "fields": {
                "task_id": "PVTF_task",
                "execution_status": "PVTF_status",
                "tenant_board": "PVTF_tenant",
                "assignee_profile": "PVTF_profile",
                "last_summary": "PVTF_summary",
                "linked_pr": "PVTF_pr",
                "last_event_at": "PVTF_at",
            },
        },
        "workspace": {"mode": "repository", "clone_root": str(tmp_path / "clones")},
        "auto_start": "self_assigned",
        "skills": ["github-issue-to-pr", "test-driven-development"],
    }


def _service(route):
    return GitHubIntakeService(
        [route],
        secret_resolver=lambda ref: {"personal-webhook": "whsec"}[ref],
        repository_resolver=lambda resolved, payload: (None, "/repos/demo"),
        origin_resolver=lambda profile: (None, None),
    )


def test_signed_self_assignment_creates_one_routed_worktree_task_and_projection(conn, route):
    payload = _payload()
    raw, headers = _headers(payload)
    result = _service(route).process(conn, headers=headers, body=raw)

    task = kdb.get_task(conn, result.task_id)
    assert result.disposition == "created"
    assert task.assignee == "personal"
    assert task.tenant == "personal"
    assert task.workspace_path == "/repos/demo/.worktrees/github-issue-20"
    assert task.workspace_kind == "worktree"
    assert task.idempotency_key == "github:R_repo:issue:I_issue"
    assert task.status == "ready"
    assert "https://github.com/shikamasaki/demo/issues/20" in task.body
    assert conn.execute("SELECT COUNT(*) FROM github_project_outbox").fetchone()[0] == 1


def test_duplicate_delivery_and_reassignment_are_idempotent(conn, route):
    payload = _payload()
    raw, headers = _headers(payload)
    service = _service(route)
    first = service.process(conn, headers=headers, body=raw)
    duplicate = service.process(conn, headers=headers, body=raw)
    _, retry_headers = _headers(payload, delivery="delivery-2")
    reassigned = service.process(conn, headers=retry_headers, body=raw)

    assert duplicate.disposition == "duplicate_delivery"
    assert reassigned.task_id == first.task_id
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM github_project_outbox").fetchone()[0] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["installation"].update(id=78),
        lambda p: p["repository"].update(full_name="shikamasaki/other", name="other"),
        lambda p: p["assignee"].update(login="someone-else"),
        lambda p: p["repository"]["owner"].update(type="Organization"),
    ],
)
def test_route_mismatch_fails_closed_without_task(conn, route, mutation):
    payload = _payload()
    mutation(payload)
    raw, headers = _headers(payload)
    with pytest.raises(GitHubIntakeError):
        _service(route).process(conn, headers=headers, body=raw)
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_bad_signature_fails_before_delivery_or_task_write(conn, route):
    payload = _payload()
    raw, headers = _headers(payload)
    headers["X-Hub-Signature-256"] = "sha256=" + "0" * 64
    with pytest.raises(GitHubIntakeError, match="signature"):
        _service(route).process(conn, headers=headers, body=raw)
    assert conn.execute("SELECT COUNT(*) FROM github_webhook_deliveries").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_other_sender_is_passive_triage(conn, route):
    payload = _payload(sender="trusted-collaborator")
    raw, headers = _headers(payload)
    result = _service(route).process(conn, headers=headers, body=raw)
    assert result.disposition == "triage"
    assert kdb.get_task(conn, result.task_id).status == "triage"


def test_exact_repo_route_can_allow_team_assignment(conn, route):
    route["allow_team_assignments"] = True
    payload = _payload(sender="trusted-collaborator")
    raw, headers = _headers(payload)
    result = _service(route).process(conn, headers=headers, body=raw)
    assert result.disposition == "created"
    assert kdb.get_task(conn, result.task_id).status == "ready"


def test_unassign_queues_to_triage_but_running_work_requires_human_decision(conn, route):
    payload = _payload()
    raw, headers = _headers(payload)
    service = _service(route)
    created = service.process(conn, headers=headers, body=raw)

    unassigned = _payload()
    unassigned["action"] = "unassigned"
    unassigned["issue"]["assignees"] = []
    unassigned_raw, unassigned_headers = _headers(unassigned, delivery="delivery-2")
    queued = service.process(conn, headers=unassigned_headers, body=unassigned_raw)
    assert queued.disposition == "triage"
    assert kdb.get_task(conn, created.task_id).status == "triage"

    conn.execute("UPDATE tasks SET status='running' WHERE id=?", (created.task_id,))
    closed = _payload()
    closed["action"] = "closed"
    closed["issue"]["state"] = "closed"
    closed_raw, closed_headers = _headers(closed, delivery="delivery-3")
    running = service.process(conn, headers=closed_headers, body=closed_raw)
    assert running.disposition == "decision_required"
    assert kdb.get_task(conn, created.task_id).status == "running"
    assert conn.execute(
        "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (created.task_id,),
    ).fetchone()[0] == "github_intake_decision_required"


def test_lifecycle_event_enqueues_one_projection_without_accepting_card_state(conn, route):
    payload = _payload()
    raw, headers = _headers(payload)
    result = _service(route).process(conn, headers=headers, body=raw)
    with kdb.write_txn(conn):
        kdb._append_event(conn, result.task_id, "blocked", {"reason": "human approval"})
    rows = conn.execute(
        "SELECT event_kind FROM github_project_outbox ORDER BY id"
    ).fetchall()
    assert [row[0] for row in rows] == ["created", "blocked"]

    project_event = dict(payload)
    project_event["action"] = "edited"
    project_raw, project_headers = _headers(project_event, delivery="delivery-project", event="projects_v2_item")
    ignored = _service(route).process(conn, headers=project_headers, body=project_raw)
    assert ignored.disposition == "ignored"
    assert kdb.get_task(conn, result.task_id).status == "ready"


def test_projector_uses_explicit_route_token_and_recovers_undelivered_rows(conn, route):
    payload = _payload()
    raw, headers = _headers(payload)
    result = _service(route).process(conn, headers=headers, body=raw)
    with kdb.write_txn(conn):
        kdb._append_event(conn, result.task_id, "blocked", {"reason": "approval"})
    calls = []

    def graphql(token, query, variables):
        calls.append((token, query, variables))
        if "addProjectV2ItemById" in query:
            return {"data": {"addProjectV2ItemById": {"item": {"id": "PVTI_item"}}}}
        return {"data": {"updateProjectV2ItemFieldValue": {"projectV2Item": {"id": "PVTI_item"}}}}

    projector = GitHubProjector(
        token_resolver=lambda ref: {"personal-project-token": "tenant-token"}[ref],
        graphql=graphql,
    )
    assert projector.drain(conn, limit=10) == 2
    assert calls and {call[0] for call in calls} == {"tenant-token"}
    add_calls = [call for call in calls if "addProjectV2ItemById" in call[1]]
    assert len(add_calls) == 1
    assert add_calls[0][2]["content"] == "I_issue"
    assert conn.execute(
        "SELECT COUNT(*) FROM github_project_outbox WHERE delivered_at IS NOT NULL"
    ).fetchone()[0] == 2
    assert projector.drain(conn, limit=10) == 0
