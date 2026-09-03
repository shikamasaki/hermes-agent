"""Contract tests for the fork's self-completing upstream-sync workflows."""

from pathlib import Path
import json
import re
import subprocess

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
    assert "Hermes Gateway accepted the conflict event." in scripts
    assert "Fail closed after upstream sync conflict" in names
    assert "exit 1" in scripts


def test_merge_only_enters_recovery_path_when_merge_head_confirms_a_conflict():
    workflow = _workflow(SYNC_WORKFLOW)
    merge_step = next(
        step
        for step in workflow["jobs"]["sync"]["steps"]
        if step["name"] == "Merge upstream into sync branch"
    )
    script = merge_step["run"]
    assert 'git rev-parse -q --verify MERGE_HEAD >/dev/null' in script
    assert 'echo "Upstream merge failed without MERGE_HEAD; not treating it as a conflict." >&2' in script
    assert 'git merge --abort || {' in script
    assert 'echo "git merge --abort failed after confirmed conflict." >&2' in script
    assert 'exit 1' in script


def test_conflict_creates_or_updates_one_sha_keyed_recovery_issue_before_webhook():
    workflow = _workflow(SYNC_WORKFLOW)
    assert workflow["permissions"]["issues"] == "write"
    assert workflow["concurrency"] == {
        "group": "upstream-sync-main",
        "cancel-in-progress": False,
    }
    steps = workflow["jobs"]["sync"]["steps"]
    names = [step["name"] for step in steps]
    issue_index = names.index("Create or update upstream sync recovery issue")
    webhook_index = names.index("Notify Hermes Gateway to resolve conflict")
    assert issue_index < webhook_index

    issue_step = steps[issue_index]
    assert issue_step["if"] == "${{ !cancelled() && steps.merge.outputs.has_conflict == 'true' }}"
    assert issue_step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert issue_step["env"]["UPSTREAM_SHA"] == "${{ steps.merge.outputs.upstream_sha }}"
    scripts = issue_step["run"]
    assert 'ISSUE_TITLE="upstream-sync-conflict:$UPSTREAM_SHA"' in scripts
    assert 'repos/${GITHUB_REPOSITORY}/issues?state=all&per_page=100' in scripts
    assert 'select(.pull_request == null and .title == $title)' in scripts
    assert 'sort_by(.number)' in scripts
    assert "ISSUE_NUMBER=\"$(jq -r '.[0].number // empty'" in scripts
    assert 'state=open' in scripts
    assert 'gh api --method PATCH "repos/${GITHUB_REPOSITORY}/issues/$ISSUE_NUMBER"' in scripts
    assert 'gh api --method PATCH "repos/${GITHUB_REPOSITORY}/issues/$DUPLICATE_NUMBER" -f state=closed' in scripts
    assert 'gh api --method POST "repos/${GITHUB_REPOSITORY}/issues"' in scripts



def test_recovery_issue_body_is_limited_to_recovery_data_and_conflict_still_fails():
    workflow = _workflow(SYNC_WORKFLOW)
    steps = workflow["jobs"]["sync"]["steps"]
    issue_step = next(
        step
        for step in steps
        if step["name"] == "Create or update upstream sync recovery issue"
    )
    issue_script = issue_step["run"]
    assert "upstream_sha:" in issue_script
    assert "fork_main_sha:" in issue_script
    assert "conflict_files:" in issue_script
    assert "run_url:" in issue_script
    assert "WEBHOOK_SECRET" not in issue_script
    assert "PAYLOAD" not in issue_script
    assert "SIGNATURE" not in issue_script

    webhook_step = next(
        step for step in steps if step["name"] == "Notify Hermes Gateway to resolve conflict"
    )
    assert webhook_step["continue-on-error"] is True
    assert '[[ -n "$WEBHOOK_URL" && -n "$WEBHOOK_SECRET" ]] || exit 0' in webhook_step["run"]
    assert "exit 1" not in webhook_step["run"]

    failure_step = next(
        step for step in steps if step["name"] == "Fail closed after upstream sync conflict"
    )
    conflict_after_failure = "${{ !cancelled() && steps.merge.outputs.has_conflict == 'true' }}"
    assert issue_step["if"] == conflict_after_failure
    assert webhook_step["if"] == conflict_after_failure
    assert failure_step["if"] == conflict_after_failure
    assert failure_step["run"].strip() == "exit 1"


def test_success_closes_only_recovery_issues_whose_upstream_sha_is_merged():
    workflow = _workflow(SYNC_WORKFLOW)
    steps = workflow["jobs"]["sync"]["steps"]
    close_step = next(
        step for step in steps if step["name"] == "Close resolved upstream sync recovery issues"
    )
    assert close_step["if"] == "steps.merge.outputs.has_conflict == 'false'"
    assert close_step["env"]["GH_TOKEN"] == "${{ github.token }}"
    scripts = close_step["run"]
    assert 'repos/${GITHUB_REPOSITORY}/issues?state=open&per_page=100' in scripts
    assert 'capture("^upstream-sync-conflict:(?<sha>[0-9a-f]{40})$")' in scripts
    assert 'git merge-base --is-ancestor "$TRACKED_SHA" origin/main' in scripts
    assert 'gh api --method PATCH "repos/${GITHUB_REPOSITORY}/issues/$ISSUE_NUMBER" -f state=closed' in scripts


def test_clean_sync_dispatches_and_selects_ci_by_token_and_exact_sha():
    workflow = _workflow(SYNC_WORKFLOW)
    assert workflow["permissions"]["actions"] == "write"
    scripts = _run_scripts(SYNC_WORKFLOW)
    assert "gh workflow run ci.yaml" in scripts
    assert "sync_token" in scripts
    assert "displayTitle" in scripts
    assert "headSha" in scripts
    assert "| head -n 1" not in scripts
    assert '[.[] | select(.headSha == $sha and .displayTitle == $title) | .databaseId][0] // empty' in scripts
    assert "gh run watch" in scripts


def test_clean_sync_scopes_gh_workflow_and_run_commands_to_the_fork():
    scripts = _run_scripts(SYNC_WORKFLOW)
    assert re.search(
        r'gh workflow run ci\.yaml \\\n\s+--repo "\$GITHUB_REPOSITORY"', scripts
    )
    assert re.search(r'gh run list \\\n\s+--repo "\$GITHUB_REPOSITORY"', scripts)
    assert 'gh run watch "$RUN_ID" --repo "$GITHUB_REPOSITORY" --exit-status' in scripts


def test_clean_sync_reruns_failed_ci_once_before_failing_closed():
    scripts = _run_scripts(SYNC_WORKFLOW)
    watch = 'gh run watch "$RUN_ID" --repo "$GITHUB_REPOSITORY" --exit-status'
    rerun = 'gh run rerun "$RUN_ID" --repo "$GITHUB_REPOSITORY" --failed'
    assert f"if ! {watch}; then" in scripts
    assert scripts.count(rerun) == 1
    assert scripts.count(watch) == 2
    assert re.search(
        r'previous_attempt="\$\(\s+gh run view "\$RUN_ID"', scripts
    )
    assert "current_attempt > previous_attempt" in scripts
    assert '[[ "$rerun_started" == true ]]' in scripts


def test_merge_is_exact_sha_guarded_and_preserves_upstream_ancestry():
    scripts = _run_scripts(SYNC_WORKFLOW)
    assert 'sha="$SYNC_SHA"' in scripts
    assert "merge_method=merge" in scripts
    assert "merge_method=squash" not in scripts
    assert "gh pr merge" not in scripts
    assert "remote_head" in scripts
    assert "base.ref" in scripts
    assert "head.sha" in scripts


RECOVERY_MATCH_FILTER = """add
| [ .[] | select(.pull_request == null and .title == $title) ]
| sort_by(.number)"""
RECOVERY_DUPLICATE_FILTER = (
    '.[] | select(.number != $canonical and .state == "open") | .number'
)
RECOVERY_CLOSE_FILTER = r'''.[].[]
| select(.pull_request == null)
| . as $issue
| try (.title | capture("^upstream-sync-conflict:(?<sha>[0-9a-f]{40})$").sha) catch empty
| "\($issue.number) \(.)"'''


def _run_jq(
    filter_: str, pages: list[list[dict]], *args: str, slurp: bool = True
) -> str:
    command = ["jq", "-r"]
    if slurp:
        command.append("-s")
    result = subprocess.run(
        [*command, *args, filter_],
        input="".join(json.dumps(page) + "\n" for page in pages),
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_recovery_issue_jq_selects_no_issue_for_zero_matches():
    assert _run_jq(
        RECOVERY_MATCH_FILTER,
        [[{"number": 1, "title": "unrelated", "pull_request": None}]],
        "--arg",
        "title",
        "upstream-sync-conflict:" + "a" * 40,
    ) == "[]\n"


def test_recovery_issue_jq_selects_one_matching_issue():
    sha = "b" * 40
    output = _run_jq(
        RECOVERY_MATCH_FILTER,
        [[
            {"number": 9, "title": f"upstream-sync-conflict:{sha}", "pull_request": None},
            {"number": 2, "title": f"upstream-sync-conflict:{sha}", "pull_request": {"url": "pr"}},
        ]],
        "--arg",
        "title",
        f"upstream-sync-conflict:{sha}",
    )
    assert json.loads(output) == [
        {"number": 9, "title": f"upstream-sync-conflict:{sha}", "pull_request": None}
    ]


def test_recovery_issue_jq_canonicalizes_paginated_duplicates_and_filters_closes():
    sha = "c" * 40
    pages = [
        [{"number": 9, "state": "open", "title": f"upstream-sync-conflict:{sha}", "pull_request": None}],
        [
            {"number": 4, "state": "closed", "title": f"upstream-sync-conflict:{sha}", "pull_request": None},
            {"number": 8, "state": "open", "title": f"upstream-sync-conflict:{sha}", "pull_request": None},
        ],
    ]
    matching = _run_jq(
        RECOVERY_MATCH_FILTER, pages, "--arg", "title", f"upstream-sync-conflict:{sha}"
    )
    assert [issue["number"] for issue in json.loads(matching)] == [4, 8, 9]
    assert _run_jq(
        RECOVERY_DUPLICATE_FILTER,
        [json.loads(matching)],
        "--argjson",
        "canonical",
        "4",
        slurp=False,
    ) == "8\n9\n"


def test_resolved_recovery_jq_filter_excludes_prs_and_non_sha_titles():
    sha = "d" * 40
    output = _run_jq(
        RECOVERY_CLOSE_FILTER,
        [[
            {"number": 1, "title": f"upstream-sync-conflict:{sha}", "pull_request": None},
            {"number": 2, "title": "upstream-sync-conflict:not-a-sha", "pull_request": None},
            {"number": 3, "title": f"upstream-sync-conflict:{sha}", "pull_request": {"url": "pr"}},
        ]],
    )
    assert output == f"1 {sha}\n"


def _normalized_jq(filter_: str) -> str:
    return "\n".join(line.strip() for line in filter_.splitlines())


def test_workflow_uses_the_executable_jq_filters_and_delimits_conflict_filenames():
    workflow = _workflow(SYNC_WORKFLOW)
    issue_script = next(
        step["run"]
        for step in workflow["jobs"]["sync"]["steps"]
        if step["name"] == "Create or update upstream sync recovery issue"
    )
    close_script = next(
        step["run"]
        for step in workflow["jobs"]["sync"]["steps"]
        if step["name"] == "Close resolved upstream sync recovery issues"
    )
    assert _normalized_jq(RECOVERY_MATCH_FILTER) in _normalized_jq(issue_script)
    assert RECOVERY_DUPLICATE_FILTER in issue_script
    assert _normalized_jq(RECOVERY_CLOSE_FILTER) in _normalized_jq(close_script)
    assert "--- begin conflict filenames ---" in issue_script
    assert "--- end conflict filenames ---" in issue_script
