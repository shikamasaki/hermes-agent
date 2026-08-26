"""Signed GitHub Issue intake and one-way Projects v2 projection.

This module deliberately keeps GitHub outside the Kanban execution state machine:
GitHub webhooks may create/triage work, while every later Hermes lifecycle event
is projected through a durable outbox. Project card movement is never imported
as execution state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Optional
import urllib.request

from hermes_cli import kanban_db as kdb


class GitHubIntakeError(ValueError):
    """A delivery failed authentication or explicit routing policy."""


@dataclass(frozen=True)
class IntakeResult:
    disposition: str
    task_id: Optional[str] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS github_webhook_deliveries (
    delivery_id TEXT PRIMARY KEY,
    event TEXT NOT NULL,
    action TEXT NOT NULL,
    repository TEXT,
    installation_id INTEGER,
    task_id TEXT,
    received_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS github_issue_links (
    task_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    repository_node_id TEXT NOT NULL,
    issue_node_id TEXT NOT NULL,
    repository TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    issue_url TEXT NOT NULL,
    project_id TEXT,
    project_token_ref TEXT,
    project_fields TEXT,
    project_item_id TEXT,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS github_project_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    delivery_key TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    task_event_id INTEGER,
    event_kind TEXT NOT NULL,
    project_id TEXT NOT NULL,
    project_token_ref TEXT NOT NULL,
    project_fields TEXT NOT NULL,
    payload TEXT NOT NULL,
    project_item_id TEXT,
    created_at INTEGER NOT NULL,
    delivered_at INTEGER,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_github_project_pending
    ON github_project_outbox(delivered_at, id);
"""


def initialize_github_sync(conn: sqlite3.Connection) -> None:
    """Install the additive GitHub intake/projection tables."""
    conn.executescript(_SCHEMA)
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(github_issue_links)")
    }
    if "project_item_id" not in columns:
        conn.execute(
            "ALTER TABLE github_issue_links ADD COLUMN project_item_id TEXT"
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _header(headers: Mapping[str, str], name: str) -> str:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value).strip()
    return ""


def _route_value(route: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = route.get(key, default)
    return value


def _canonical_repo_path(root: str, owner: str, name: str) -> Path:
    base = Path(root).expanduser().resolve()
    candidate = (base / owner / name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:  # pragma: no cover - names are shape-checked too
        raise GitHubIntakeError("repository path escapes configured clone_root") from exc
    return candidate


def _safe_component(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value or "") or value in {".", ".."}:
        raise GitHubIntakeError(f"invalid {label}")
    return value


def default_repository_resolver(
    route: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[Optional[str], str]:
    """Resolve an existing Project or clone into the route's controlled root.

    This function runs only after signature, installation, repository, assignee,
    and sender policy have passed. It never uses the caller's cwd.
    """
    repository = payload["repository"]
    owner = _safe_component(str(repository["owner"]["login"]), "owner")
    name = _safe_component(str(repository["name"]), "repository")
    workspace = route.get("workspace") or {}
    clone_root = str(workspace.get("clone_root") or "").strip()
    if not clone_root:
        raise GitHubIntakeError("route has no controlled workspace.clone_root")
    repo_path = _canonical_repo_path(clone_root, owner, name)

    from hermes_cli import projects_db

    expected_remote = f"https://github.com/{owner}/{name}.git"
    with projects_db.connect_closing() as project_conn:
        for project in projects_db.list_projects(project_conn):
            primary = Path(project.primary_path).expanduser().resolve() if project.primary_path else None
            if primary is None or not primary.exists():
                continue
            remote = subprocess.run(
                ["git", "-C", str(primary), "remote", "get-url", "origin"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            accepted_remotes = {
                expected_remote,
                expected_remote.removesuffix(".git"),
                f"git@github.com:{owner}/{name}.git",
            }
            if primary == repo_path or remote in accepted_remotes:
                return project.id, str(primary)

    if not repo_path.exists():
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        url = str(repository.get("clone_url") or "")
        expected = f"https://github.com/{owner}/{name}.git"
        if url != expected:
            raise GitHubIntakeError("repository clone URL does not match routed repository")
        completed = subprocess.run(
            ["git", "clone", "--", url, str(repo_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise GitHubIntakeError("controlled repository clone failed")
    if not (repo_path / ".git").exists():
        raise GitHubIntakeError("controlled repository path is not a git checkout")

    with projects_db.connect_closing() as project_conn:
        existing = projects_db.find_by_primary_path(project_conn, str(repo_path))
        if existing is not None:
            return existing.id, str(repo_path)
        project_id = projects_db.create_project(
            project_conn,
            name=f"{owner}/{name}",
            slug=f"{owner}-{name}",
            primary_path=str(repo_path),
            board_slug=str(route.get("board") or "default"),
        )
    return project_id, str(repo_path)


def canonical_bot_chat_origin(profile: str) -> tuple[str, str]:
    """Resolve the exact profile's canonical Bot Chat; never borrow another."""
    from hermes_cli.profiles import get_profile_dir
    from hermes_state import SessionDB

    profile_dir = Path(get_profile_dir(profile))
    if not profile_dir.is_dir():
        raise GitHubIntakeError(f"routed profile {profile!r} does not exist")
    state_path = profile_dir / "state.db"
    if not state_path.exists():
        raise GitHubIntakeError("routed profile has no canonical Bot Chat state")
    db = SessionDB(db_path=state_path)
    try:
        row = db.get_session_by_title("Bot Chat")
    finally:
        db.close()
    session_id = str((row or {}).get("id") or "")
    if not session_id:
        raise GitHubIntakeError("routed profile has no canonical Bot Chat")
    return profile, session_id


class GitHubIntakeService:
    def __init__(
        self,
        routes: Iterable[Mapping[str, Any]],
        *,
        secret_resolver: Callable[[str], str],
        repository_resolver: Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[Optional[str], str]] = default_repository_resolver,
        origin_resolver: Optional[Callable[[str], tuple[Optional[str], Optional[str]]]] = None,
    ) -> None:
        self._routes = [dict(route) for route in routes]
        self._secret_resolver = secret_resolver
        self._repository_resolver = repository_resolver
        self._origin_resolver = origin_resolver or canonical_bot_chat_origin

    def _resolve_route(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        installation = str((payload.get("installation") or {}).get("id") or "")
        repository = payload.get("repository") or {}
        full_name = str(repository.get("full_name") or "").lower()
        owner_obj = repository.get("owner") or {}
        owner = str(owner_obj.get("login") or "").lower()
        name = str(repository.get("name") or "").lower()
        exact: list[Mapping[str, Any]] = []
        wildcard: list[Mapping[str, Any]] = []
        for route in self._routes:
            if str(route.get("installation_id") or "") != installation:
                continue
            configured_repo = str(route.get("repository") or "").lower()
            if configured_repo and configured_repo == full_name:
                exact.append(route)
                continue
            if str(route.get("owner") or "").lower() != owner:
                continue
            enabled = {str(value).lower() for value in route.get("repositories") or ()}
            if name in enabled or full_name in enabled or "*" in enabled:
                wildcard.append(route)
        matches = exact or wildcard
        if len(matches) != 1:
            raise GitHubIntakeError("delivery does not resolve to exactly one explicit route")
        route = matches[0]
        if not exact and str(owner_obj.get("type") or "") != "User":
            raise GitHubIntakeError("owner wildcard is limited to personal repositories")
        required = ("profile", "board", "tenant", "assignee", "webhook_secret_ref")
        if any(not str(route.get(key) or "").strip() for key in required):
            raise GitHubIntakeError("route is incomplete")
        return route

    def _verify(self, route: Mapping[str, Any], headers: Mapping[str, str], body: bytes) -> None:
        signature = _header(headers, "X-Hub-Signature-256")
        if not signature.startswith("sha256="):
            raise GitHubIntakeError("missing GitHub signature")
        ref = str(route["webhook_secret_ref"])
        secret = self._secret_resolver(ref)
        if not secret:
            raise GitHubIntakeError("configured webhook secret is unavailable")
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise GitHubIntakeError("invalid GitHub signature")

    def process(
        self,
        conn: sqlite3.Connection,
        *,
        headers: Mapping[str, str],
        body: bytes,
    ) -> IntakeResult:
        initialize_github_sync(conn)
        delivery_id = _header(headers, "X-GitHub-Delivery")
        event = _header(headers, "X-GitHub-Event")
        if not delivery_id:
            raise GitHubIntakeError("missing GitHub delivery id")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GitHubIntakeError("invalid GitHub payload") from exc
        if not isinstance(payload, dict):
            raise GitHubIntakeError("invalid GitHub payload")

        route = self._resolve_route(payload)
        self._verify(route, headers, body)
        if event != "issues":
            return IntakeResult("ignored")
        action = str(payload.get("action") or "")
        if action not in {"assigned", "unassigned", "closed"}:
            return IntakeResult("ignored")
        issue = payload.get("issue") or {}
        repository = payload.get("repository") or {}
        if action != "closed" and str(issue.get("state") or "") != "open":
            raise GitHubIntakeError("issue is not open")
        expected_assignee = str(route["assignee"]).lower()
        assigned_login = str((payload.get("assignee") or {}).get("login") or "").lower()
        assignees = {str(row.get("login") or "").lower() for row in issue.get("assignees") or []}
        if action == "assigned" and (
            assigned_login != expected_assignee or expected_assignee not in assignees
        ):
            raise GitHubIntakeError("assignee does not match route")
        if action == "unassigned" and assigned_login != expected_assignee:
            raise GitHubIntakeError("unassigned login does not match route")

        sender = str((payload.get("sender") or {}).get("login") or "").lower()
        owner = str((repository.get("owner") or {}).get("login") or "").lower()
        self_assigned = sender == owner == expected_assignee
        auto_start = self_assigned and route.get("auto_start") == "self_assigned"
        if route.get("allow_team_assignments") is True:
            auto_start = True
        triage = not auto_start

        repo_node = str(repository.get("node_id") or "")
        issue_node = str(issue.get("node_id") or "")
        if not repo_node or not issue_node:
            raise GitHubIntakeError("repository and issue node ids are required")
        idempotency_key = f"github:{repo_node}:issue:{issue_node}"

        existing_delivery = conn.execute(
            "SELECT task_id FROM github_webhook_deliveries WHERE delivery_id = ?",
            (delivery_id,),
        ).fetchone()
        if existing_delivery is not None:
            return IntakeResult("duplicate_delivery", existing_delivery["task_id"])
        existing_link = conn.execute(
            "SELECT task_id FROM github_issue_links WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if action in {"unassigned", "closed"}:
            if existing_link is None:
                return IntakeResult("ignored")
            task_id = str(existing_link["task_id"])
            task = kdb.get_task(conn, task_id)
            queued = bool(
                task
                and task.status in {"triage", "todo", "ready", "blocked", "scheduled"}
            )
            with kdb.write_txn(conn):
                conn.execute(
                    "INSERT INTO github_webhook_deliveries "
                    "(delivery_id,event,action,repository,installation_id,task_id,received_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (delivery_id, event, action, repository.get("full_name"), int(payload["installation"]["id"]), task_id, int(time.time())),
                )
                if queued:
                    conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (task_id,))
                    kdb._append_event(conn, task_id, "github_intake_revoked", {"action": action})
                elif task:
                    kdb._append_event(conn, task_id, "github_intake_decision_required", {"action": action})
            return IntakeResult("triage" if queued else "decision_required", task_id)
        if existing_link is not None:
            with kdb.write_txn(conn):
                conn.execute(
                    "INSERT INTO github_webhook_deliveries "
                    "(delivery_id,event,action,repository,installation_id,task_id,received_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (delivery_id, event, action, repository.get("full_name"), int(payload["installation"]["id"]), existing_link["task_id"], int(time.time())),
                )
            return IntakeResult("existing", existing_link["task_id"])

        project_id, repo_path = self._repository_resolver(route, payload)
        origin_profile, origin_session_id = self._origin_resolver(str(route["profile"]))
        issue_number = int(issue.get("number") or 0)
        worktree_path = str(Path(repo_path) / ".worktrees" / f"github-issue-{issue_number}")
        project = route.get("project") or {}
        project_node_id = str(project.get("id") or "")
        project_token_ref = str(route.get("project_token_ref") or "")
        project_fields = project.get("fields") or {}
        now = int(time.time())
        body_text = (
            f"Primary source: {issue.get('html_url')}\n"
            f"Repository: {repository.get('full_name')}\n"
            "Carry this Issue through isolated-worktree TDD, independent different-provider review, "
            "push, and a read-back PR. Never merge, deploy, or broaden credentials without approval."
        )
        with kdb.write_txn(conn):
            task_id = kdb.create_task(
                conn,
                title=f"GitHub #{issue_number}: {issue.get('title')}",
                body=body_text,
                assignee=str(route["profile"]),
                created_by="github-app",
                workspace_kind="worktree",
                workspace_path=worktree_path,
                branch_name=f"github/issue-{issue_number}",
                tenant=str(route["tenant"]),
                triage=triage,
                idempotency_key=idempotency_key,
                skills=route.get("skills") or ["github-issue-to-pr", "test-driven-development"],
                goal_mode=True,
                initial_status="running" if triage else "running",
                board=str(route["board"]),
                project_id=project_id,
                origin_profile=origin_profile,
                origin_session_id=origin_session_id,
            )
            conn.execute(
                "INSERT INTO github_issue_links "
                "(task_id,idempotency_key,repository_node_id,issue_node_id,repository,issue_number,issue_url,project_id,project_token_ref,project_fields,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, idempotency_key, repo_node, issue_node, repository.get("full_name"), issue_number, issue.get("html_url"), project_node_id or None, project_token_ref or None, _json(project_fields), now),
            )
            conn.execute(
                "INSERT INTO github_webhook_deliveries "
                "(delivery_id,event,action,repository,installation_id,task_id,received_at) VALUES (?,?,?,?,?,?,?)",
                (delivery_id, event, action, repository.get("full_name"), int(payload["installation"]["id"]), task_id, now),
            )
            event_row = conn.execute(
                "SELECT id, kind, created_at FROM task_events WHERE task_id=? ORDER BY id LIMIT 1",
                (task_id,),
            ).fetchone()
            if project_node_id and project_token_ref and event_row:
                enqueue_project_event(conn, task_id, int(event_row["id"]), str(event_row["kind"]), created_at=int(event_row["created_at"]))
        return IntakeResult("triage" if triage else "created", task_id)


def enqueue_project_event(
    conn: sqlite3.Connection,
    task_id: str,
    event_id: int,
    kind: str,
    *,
    created_at: Optional[int] = None,
) -> None:
    """Append an idempotent projection row inside the caller's task txn."""
    try:
        link = conn.execute(
            "SELECT * FROM github_issue_links WHERE task_id=?", (task_id,)
        ).fetchone()
    except sqlite3.OperationalError:
        return
    if link is None or not link["project_id"] or not link["project_token_ref"]:
        return
    task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if task is None:
        return
    event_payload: dict[str, Any] = {}
    event_row = conn.execute(
        "SELECT payload FROM task_events WHERE id=? AND task_id=?",
        (event_id, task_id),
    ).fetchone()
    if event_row and event_row["payload"]:
        try:
            decoded = json.loads(event_row["payload"])
            if isinstance(decoded, dict):
                event_payload = decoded
        except (TypeError, ValueError):
            pass
    payload = {
        "task_id": task_id,
        "status": task["status"],
        "tenant": task["tenant"],
        "board": kdb._board_for_connection(conn),
        "assignee": task["assignee"],
        "summary": event_payload.get("summary")
        or event_payload.get("reason")
        or task["result"],
        "issue_url": link["issue_url"],
        "issue_node_id": link["issue_node_id"],
        "event_kind": kind,
        "event_at": int(created_at or time.time()),
    }
    conn.execute(
        "INSERT OR IGNORE INTO github_project_outbox "
        "(delivery_key,task_id,task_event_id,event_kind,project_id,project_token_ref,project_fields,payload,created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (f"github-project:{task_id}:event:{event_id}", task_id, event_id, kind, link["project_id"], link["project_token_ref"], link["project_fields"], _json(payload), int(created_at or time.time())),
    )


_ADD_ITEM = """mutation AddHermesItem($project: ID!, $content: ID!) { addProjectV2ItemById(input:{projectId:$project,contentId:$content}) { item { id } } }"""
_UPDATE_TEXT = """mutation UpdateHermesField($project: ID!, $item: ID!, $field: ID!, $value: String!) { updateProjectV2ItemFieldValue(input:{projectId:$project,itemId:$item,fieldId:$field,value:{text:$value}}) { projectV2Item { id } } }"""


class GitHubProjector:
    """Bounded durable outbox drain using only an explicitly routed token."""

    def __init__(self, *, token_resolver: Callable[[str], str], graphql: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]) -> None:
        self._token_resolver = token_resolver
        self._graphql = graphql

    def drain(self, conn: sqlite3.Connection, *, limit: int = 100) -> int:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='github_project_outbox'"
        ).fetchone()
        if table is None:
            return 0
        rows = conn.execute(
            "SELECT * FROM github_project_outbox WHERE delivered_at IS NULL ORDER BY id LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        delivered = 0
        for row in rows:
            try:
                token = self._token_resolver(str(row["project_token_ref"]))
                if not token:
                    raise RuntimeError("route token unavailable")
                payload = json.loads(row["payload"])
                fields = json.loads(row["project_fields"])
                link = conn.execute(
                    "SELECT project_item_id FROM github_issue_links WHERE task_id=?",
                    (row["task_id"],),
                ).fetchone()
                item_id = link["project_item_id"] if link else None
                if not item_id:
                    added = self._graphql(token, _ADD_ITEM, {"project": row["project_id"], "content": payload["issue_node_id"]})
                    item_id = (((added.get("data") or {}).get("addProjectV2ItemById") or {}).get("item") or {}).get("id")
                    if not item_id:
                        raise RuntimeError("GitHub did not return a project item id")
                    conn.execute("UPDATE github_project_outbox SET project_item_id=? WHERE id=?", (item_id, row["id"]))
                    conn.execute(
                        "UPDATE github_issue_links SET project_item_id=? WHERE task_id=?",
                        (item_id, row["task_id"]),
                    )
                values = {
                    "task_id": payload["task_id"],
                    "execution_status": payload.get("status") or "",
                    "tenant_board": f"{payload.get('tenant') or ''} / {payload.get('board') or ''}",
                    "assignee_profile": payload.get("assignee") or "",
                    "last_summary": payload.get("summary") or "",
                    "linked_pr": payload.get("linked_pr") or "",
                    "last_event_at": str(payload.get("event_at") or ""),
                }
                for name, value in values.items():
                    field_id = fields.get(name)
                    if field_id:
                        self._graphql(token, _UPDATE_TEXT, {"project": row["project_id"], "item": item_id, "field": field_id, "value": str(value)})
                conn.execute("UPDATE github_project_outbox SET delivered_at=?, last_error=NULL WHERE id=?", (int(time.time()), row["id"]))
                delivered += 1
            except Exception as exc:
                conn.execute("UPDATE github_project_outbox SET last_error=? WHERE id=?", (str(exc)[:500], row["id"]))
        return delivered


def env_secret_resolver(reference: str) -> str:
    """Resolve only the exact configured variable; never borrow gh/default auth."""
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", reference or ""):
        raise GitHubIntakeError("credential reference must be an explicit environment variable name")
    return os.environ.get(reference, "")


def github_graphql(
    token: str, query: str, variables: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Execute one GitHub GraphQL request with an explicitly supplied token."""
    request = urllib.request.Request(
        "https://api.github.com/graphql",
        data=_json({"query": query, "variables": dict(variables)}).encode("utf-8"),
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "hermes-agent-github-projector",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict) or result.get("errors"):
        raise RuntimeError("GitHub GraphQL mutation failed")
    return result


def configured_routes() -> list[Mapping[str, Any]]:
    """Load the explicit GitHub routes from config.yaml."""
    from hermes_cli.config import cfg_get, load_config

    routes = cfg_get(load_config(), "github_intake", "routes", default=[])
    return [dict(route) for route in routes if isinstance(route, Mapping)]


def process_configured_delivery(
    *, headers: Mapping[str, str], body: bytes
) -> IntakeResult:
    """Resolve the route, open its exact board, and process one delivery."""
    routes = configured_routes()
    service = GitHubIntakeService(routes, secret_resolver=env_secret_resolver)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GitHubIntakeError("invalid GitHub payload") from exc
    route = service._resolve_route(payload)
    from gateway.run import _profile_runtime_scope
    from hermes_cli.profiles import get_profile_dir

    profile = str(route["profile"])
    profile_dir = Path(get_profile_dir(profile))
    if not profile_dir.is_dir():
        raise GitHubIntakeError(f"routed profile {profile!r} does not exist")
    with _profile_runtime_scope(profile_dir):
        with kdb.connect_closing(board=str(route["board"])) as conn:
            result = service.process(conn, headers=headers, body=body)
            GitHubProjector(
                token_resolver=env_secret_resolver, graphql=github_graphql
            ).drain(conn, limit=100)
            return result


def recover_configured_project_outboxes() -> int:
    """Perform one bounded startup drain for configured boards."""
    routes = configured_routes()
    boards = sorted({str(route.get("board") or "") for route in routes if route.get("board")})
    delivered = 0
    for board in boards:
        with kdb.connect_closing(board=board) as conn:
            delivered += GitHubProjector(
                token_resolver=env_secret_resolver, graphql=github_graphql
            ).drain(conn, limit=100)
    return delivered
