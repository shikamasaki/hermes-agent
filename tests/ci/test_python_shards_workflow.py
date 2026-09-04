"""Contracts for fork-only four-way Python test sharding."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yaml"
RUNNER = ROOT / "scripts" / "run_tests_parallel.py"
CANONICAL_REPOSITORY = "NousResearch/hermes-agent"


def _workflow() -> dict:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML 1.1 parses the YAML key `on` as boolean True.
    if True in data and "on" not in data:
        data["on"] = data[True]
    return data


def _runner_module():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix_for(repository: str) -> list[dict[str, str]]:
    expression = _workflow()["jobs"]["test"]["strategy"]["matrix"]["include"]
    assert isinstance(expression, str)
    canonical_json, fork_json = expression.split(" && ", 1)[1].split(" || ")
    selected = canonical_json if repository == CANONICAL_REPOSITORY else fork_json
    return json.loads(selected.strip().removesuffix(") }}").strip("'"))


def test_canonical_matrix_is_one_unsliced_96_worker_job() -> None:
    matrix = _matrix_for(CANONICAL_REPOSITORY)
    assert matrix == [{"slice": "", "workers": "96"}]


def test_fork_matrix_is_four_unsliced_standard_runner_jobs() -> None:
    workflow = _workflow()
    test = workflow["jobs"]["test"]
    matrix = _matrix_for("someone/hermes-agent")

    assert test["runs-on"] == "${{ github.repository == 'NousResearch/hermes-agent' && 'ubuntu-latest-96-core' || 'ubuntu-latest' }}"
    assert test["strategy"]["fail-fast"] is False
    assert matrix == [
        {"slice": "1/4", "workers": "4"},
        {"slice": "2/4", "workers": "4"},
        {"slice": "3/4", "workers": "4"},
        {"slice": "4/4", "workers": "4"},
    ]


def test_test_job_display_slice_and_env_forwarding_are_repository_safe() -> None:
    test = _workflow()["jobs"]["test"]
    run_step = next(step for step in test["steps"] if step["name"] == "Run tests")

    assert test["name"] == "${{ github.repository == 'NousResearch/hermes-agent' && 'Run tests' || format('Run tests (slice {0})', matrix.slice) }}"
    assert test["strategy"]["matrix"]["include"]
    assert "HERMES_TEST_WORKERS: ${{ matrix.workers }}" in WORKFLOW.read_text(encoding="utf-8")
    # Canonical jobs invoke the runner with no HERMES_TEST_SLICE assignment;
    # fork jobs inject exactly their matrix slice before the same script.
    assert 'if [ -n "${{ matrix.slice }}" ]; then' in run_step["run"]
    assert "HERMES_TEST_SLICE=${{ matrix.slice }} scripts/run_tests.sh" in run_step["run"]
    assert "scripts/run_tests.sh" in run_step["run"]


def test_four_slices_are_an_exact_deterministic_inventory_without_durations() -> None:
    runner = _runner_module()
    files = runner._discover_files([ROOT / "tests"])
    assert files

    first = runner._compute_lpt_slices(files, 4, {}, ROOT)
    repeated = runner._compute_lpt_slices(files, 4, {}, ROOT)
    flattened = [path for bucket in first for path in bucket]

    assert first == repeated
    assert len(first) == 4
    assert len(flattened) == len(files)
    assert len(set(flattened)) == len(files)
    assert set(flattened) == set(files)


def test_reusable_python_result_remains_required_by_all_checks_gate() -> None:
    ci = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    tests = ci["jobs"]["tests"]
    gate = ci["jobs"]["all-checks-pass"]

    assert tests["uses"] == "./.github/workflows/tests.yml"
    assert "tests" in gate["needs"]
    assert gate["if"] == "always()"
