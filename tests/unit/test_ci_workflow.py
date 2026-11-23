"""Offline unit tests validating PR CI workflow gates and PR template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PR_CHECKS_PATH = REPO_ROOT / ".github" / "workflows" / "pr_checks.yaml"
PR_TEMPLATE_PATH = REPO_ROOT / ".github" / "pull_request_template.md"


def _load_pr_checks() -> dict[str, Any]:
    parsed = yaml.safe_load(PR_CHECKS_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_pr_checks_defines_offline_lint_and_unit_job() -> None:
    workflow = _load_pr_checks()
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert "pull_request" in triggers

    jobs = workflow["jobs"]
    assert "lint-and-unit" in jobs

    lint_job = jobs["lint-and-unit"]
    assert "needs" not in lint_job
    assert lint_job.get("permissions", {}).get("id-token") is None

    steps = lint_job["steps"]
    assert not any(
        "google-github-actions/auth" in str(step.get("uses", "")) for step in steps
    )

    run_commands = [
        str(step["run"]).strip() for step in steps if isinstance(step.get("run"), str)
    ]
    expected_commands = [
        "uv sync --locked --extra lint",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run ty check",
        "uv run pytest tests/unit",
        "uv run python -m evals.runner --dry-run",
    ]
    for command in expected_commands:
        assert command in run_commands


def test_pr_checks_guards_gcp_integration_job() -> None:
    workflow = _load_pr_checks()
    jobs = workflow["jobs"]
    assert "integration" in jobs

    integration_job = jobs["integration"]
    condition = str(integration_job.get("if", ""))
    assert "vars.CICD_PROJECT_ID" in condition

    run_commands = [
        str(step["run"]).strip()
        for step in integration_job["steps"]
        if isinstance(step.get("run"), str)
    ]
    assert "uv run pytest tests/integration" in run_commands


def test_pull_request_template_covers_required_checklists() -> None:
    assert PR_TEMPLATE_PATH.is_file()
    content = PR_TEMPLATE_PATH.read_text(encoding="utf-8").lower()
    assert "local" in content
    assert "remote" in content
    assert "parity" in content
    assert "tests/unit" in content
    assert "benchmark" in content
