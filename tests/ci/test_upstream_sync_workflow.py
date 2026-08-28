"""Contract tests for the fork's self-completing upstream-sync workflows."""

from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yaml"
SYNC_WORKFLOW = ROOT / ".github" / "workflows" / "upstream-sync.yml"


def _workflow(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    # PyYAML 1.1 parses the key `on` as boolean True.
    if True in data and "on" not in data:
        data["on"] = data[True]
    return data


def _run_scripts(path: Path) -> str:
    jobs = _workflow(path)["jobs"]
    return "\n".join(
        step.get("run", "")
        for job in jobs.values()
        for step in job.get("steps", [])
        if isinstance(step, dict)
    )


def test_ci_accepts_identifiable_manual_sync_dispatches():
    dispatch = _workflow(CI_WORKFLOW)["on"]["workflow_dispatch"]
    assert "sync_token" in dispatch["inputs"]
    assert dispatch["inputs"]["sync_token"]["required"] is False
    text = CI_WORKFLOW.read_text()
    assert "inputs.sync_token" in text
    # The local detect-changes action owns checkout now and exposes only
    # github-token; stale sparse-checkout inputs make hosted CI fail at load.
    assert "sparse-checkout:" not in text
    assert "sparse-checkout-cone-mode:" not in text


def test_sync_keeps_fetch_merge_and_fail_closed_conflict_repair_path():
    workflow = _workflow(SYNC_WORKFLOW)
    steps = workflow["jobs"]["sync"]["steps"]
    names = [step["name"] for step in steps]
    assert names[:4] == [
        "Checkout repository",
        "Configure Git",
        "Add upstream remote",
        "Merge upstream into sync branch",
    ]
    assert "Notify Hermes Gateway to resolve conflict" in names
    scripts = _run_scripts(SYNC_WORKFLOW)
    assert "git fetch upstream main" in scripts
    assert "git merge --no-edit upstream/main" in scripts
    assert "git merge --abort" in scripts
    assert "X-Webhook-Signature-V2" in scripts
    assert "Hermes Gateway accepted the conflict event; failing this run visibly." in scripts
    assert "exit 1" in scripts


def test_clean_sync_dispatches_and_selects_ci_by_token_and_exact_sha():
    workflow = _workflow(SYNC_WORKFLOW)
    assert workflow["permissions"]["actions"] == "write"
    scripts = _run_scripts(SYNC_WORKFLOW)
    assert "gh workflow run ci.yaml" in scripts
    assert "sync_token" in scripts
    assert "displayTitle" in scripts
    assert "headSha" in scripts
    assert "SYNC_SHA" in scripts
    assert "gh run watch" in scripts


def test_clean_sync_scopes_gh_workflow_and_run_commands_to_the_fork():
    scripts = _run_scripts(SYNC_WORKFLOW)
    assert re.search(
        r'gh workflow run ci\.yaml \\\n\s+--repo "\$GITHUB_REPOSITORY"', scripts
    )
    assert re.search(r'gh run list \\\n\s+--repo "\$GITHUB_REPOSITORY"', scripts)
    assert 'gh run watch "$RUN_ID" --repo "$GITHUB_REPOSITORY" --exit-status' in scripts


def test_merge_is_exact_sha_guarded_and_preserves_upstream_ancestry():
    scripts = _run_scripts(SYNC_WORKFLOW)
    assert 'sha="$SYNC_SHA"' in scripts
    assert "merge_method=merge" in scripts
    assert "merge_method=squash" not in scripts
    assert "gh pr merge" not in scripts
    assert "remote_head" in scripts
    assert "base.ref" in scripts
    assert "head.sha" in scripts
