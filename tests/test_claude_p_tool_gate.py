from pathlib import Path

from agent.claude_p_tool_gate import evaluate_tool_call


def _payload(tool: str, **tool_input):
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}


def test_coding_file_tools_are_contained_to_workspace(tmp_path: Path):
    inside = tmp_path / "src" / "x.py"
    outside = tmp_path.parent / "outside.py"

    assert evaluate_tool_call(_payload("Write", file_path=str(inside)), "coding", tmp_path)[0]
    assert not evaluate_tool_call(_payload("Write", file_path=str(outside)), "coding", tmp_path)[0]
    assert not evaluate_tool_call(_payload("Read", file_path="../outside.py"), "coding", tmp_path)[0]


def test_review_is_read_only_and_workspace_scoped(tmp_path: Path):
    inside = tmp_path / "README.md"

    assert evaluate_tool_call(_payload("Read", file_path=str(inside)), "review", tmp_path)[0]
    assert not evaluate_tool_call(_payload("Write", file_path=str(inside)), "review", tmp_path)[0]
    assert not evaluate_tool_call(_payload("Bash", command="git status"), "review", tmp_path)[0]


def test_coding_bash_is_exact_allowlist_and_shell_syntax_fails_closed(tmp_path: Path):
    allowed = (
        "git status --short",
        "scripts/run_tests.sh tests/test_x.py -v",
        "uv run pytest tests/test_x.py",
        "venv/bin/ruff check agent/x.py",
    )
    denied = (
        "uname -a",
        "touch marker",
        "git commit --dry-run",
        "git status && touch marker",
        "scripts/run_tests.sh tests/test_x.py; touch marker",
        "scripts/run_tests.sh $(whoami)",
        "scripts/run_tests.sh tests/test_x.py > out.txt",
        "FOO=bar scripts/run_tests.sh tests/test_x.py",
        "bash -c 'scripts/run_tests.sh tests/test_x.py'",
    )

    for command in allowed:
        assert evaluate_tool_call(_payload("Bash", command=command), "coding", tmp_path)[0]
    for command in denied:
        assert not evaluate_tool_call(_payload("Bash", command=command), "coding", tmp_path)[0]


def test_unknown_profile_and_tool_fail_closed(tmp_path: Path):
    assert not evaluate_tool_call(_payload("Read", file_path="README.md"), "unknown", tmp_path)[0]
    assert not evaluate_tool_call(_payload("WebFetch", url="https://example.com"), "coding", tmp_path)[0]
