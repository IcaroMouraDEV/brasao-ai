"""Antigravity `run_workflow` pipeline: executes GitHub issues sequentially with fresh context.

Each issue runs inside an isolated `agent(..., workspace="share")` so context never bloats
across issues. Runtime primitives (`args`, `phase`, `log`, `agent`, `pipeline`) are injected
by Antigravity's `run_workflow` runtime.
"""

issues = args.get("issues", [])
base_branch = args.get("base_branch", "main")
auto_push = bool(args.get("auto_push", True))
stop_on_block = bool(args.get("stop_on_block", True))

if not issues:
    raise ValueError("`args['issues']` must be a non-empty list of GitHub issue numbers.")

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_number": {"type": "integer"},
        "status": {"type": "string", "enum": ["completed", "blocked", "failed"]},
        "branch": {"type": "string"},
        "pr_url": {"type": "string"},
        "verification": {"type": "string"},
        "summary": {"type": "string"},
        "blocker": {"type": "string"},
    },
    "required": ["issue_number", "status", "branch", "pr_url", "verification", "summary"],
}

push_instructions = (
    f"6. Push the branch (never use --no-verify) and open a Pull Request against `{base_branch}` "
    "with `gh pr create` including `Closes #<N>` in the PR body."
    if auto_push
    else "6. Leave the commit on the local feature branch without pushing (auto_push=false)."
)


async def run_single_issue(issue_number: int) -> dict:
    phase(f"Issue #{issue_number}")
    log(f"Starting isolated fresh-context worker for GitHub Issue #{issue_number}...")

    prompt = f"""\
Implement GitHub Issue #{issue_number} for Brasão AI in this isolated workspace.

Mandatory procedure:
1. Read `AGENTS.md`, `.agents/rules/adk-architecture.md`, and `.agents/rules/pedagogy-sdg4.md`.
2. Fetch issue details via `gh issue view {issue_number}`.
   - SAFETY STOP: If the issue has label `infra` or asks to execute `terraform apply` or `agents-cli deploy`, do NOT execute cloud mutations; return status `"blocked"` explaining human approval is required.
3. Ensure you branch off `{base_branch}` onto a descriptive branch (e.g., `feat/issue-{issue_number}-...` or `ci/issue-{issue_number}-...`).
4. Implement all requirements cleanly in English-first code (pt-BR only for student prompts, Brazilian History corpus, and eval cases) preserving strict local (Ollama) vs. remote (Gemini) parity.
5. Run full offline verification:
   - `uv run ruff check .`
   - `uv run ruff format --check .`
   - `uv run ty check`
   - `uv run pytest tests/unit -v`
   - `uv run python -m evals.runner --dry-run --reports-dir /tmp/eval_smoke_issue_{issue_number}`
   Commit all changes with an English Conventional Commit message.
{push_instructions}
7. Call `finish` with the exact JSON schema reporting `issue_number`, `status`, `branch`, `pr_url` (empty string if not pushed), `verification`, `summary`, and `blocker` (if any).
"""

    result = await agent(
        prompt,
        schema=RESULT_SCHEMA,
        role=f"Issue #{issue_number} Worker",
        type_name="self",
        workspace="share",
    )
    log(
        f"Issue #{issue_number} finished -> status={result.get('status')} "
        f"branch={result.get('branch')} pr={result.get('pr_url') or 'none'}"
    )
    return result


results = []
for issue_num in issues:
    outcome = await run_single_issue(int(issue_num))
    results.append(outcome)
    if stop_on_block and outcome.get("status") != "completed":
        log(
            f"Stopping issue pipeline after Issue #{issue_num} "
            f"(status={outcome.get('status')}, blocker={outcome.get('blocker', 'unspecified')})."
        )
        break

phase("Summary")
for item in results:
    log(
        f"#{item['issue_number']} | {item['status']} | {item['branch']} | "
        f"{item['pr_url'] or '-'} | {item['summary']}"
    )

results
