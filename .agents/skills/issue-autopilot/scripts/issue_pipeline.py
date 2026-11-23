"""Antigravity `run_workflow` pipeline: executes GitHub issues sequentially with fresh context.

Each issue runs inside an isolated `agent(..., workspace="share")` so context never bloats
across issues. Supports `chain_branches=True` (stacked PRs) so later issues (#7, #8) build
cleanly on top of earlier unmerged PR branches (#2-#6) without touching `main`.

Runtime primitives (`args`, `phase`, `log`, `agent`, `pipeline`) are injected by Antigravity's
`run_workflow` runtime.
"""

issues = args.get("issues", [])
base_branch = args.get("base_branch", "main")
auto_push = bool(args.get("auto_push", True))
chain_branches = bool(args.get("chain_branches", True))
stop_on_block = bool(args.get("stop_on_block", True))
worker_model = args.get("model")

if not issues:
    raise ValueError("`args['issues']` must be a non-empty list of GitHub issue numbers.")

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_number": {"type": "integer"},
        "status": {"type": "string", "enum": ["completed", "blocked", "failed"]},
        "branch": {"type": "string"},
        "base_branch": {"type": "string"},
        "pr_url": {"type": "string"},
        "verification": {"type": "string"},
        "summary": {"type": "string"},
        "blocker": {"type": "string"},
    },
    "required": [
        "issue_number",
        "status",
        "branch",
        "base_branch",
        "pr_url",
        "verification",
        "summary",
    ],
}


async def run_single_issue(issue_number: int, current_base: str) -> dict:
    phase(f"Issue #{issue_number} (base: {current_base})")
    log(
        f"Starting fresh-context worker for Issue #{issue_number} "
        f"(branching from `{current_base}`)..."
    )

    push_instructions = (
        f"6. Push the branch (`git -c core.sshCommand=\"ssh -o BatchMode=yes\" push -u origin <branch>`, "
        f"never use `--no-verify`) and open a Pull Request with `gh pr create --base {current_base} "
        f"--head <branch>` referencing `Closes #{issue_number}` in the PR body."
        if auto_push
        else "6. Leave the commit on the local feature branch without pushing (`auto_push=false`)."
    )

    prompt = f"""\
Implement GitHub Issue #{issue_number} for Brasão AI in this isolated workspace.

Mandatory procedure:
1. Read `AGENTS.md`, `.agents/rules/adk-architecture.md`, and `.agents/rules/pedagogy-sdg4.md`.
2. Fetch issue details via `gh issue view {issue_number}`.
   - SAFETY STOP: If the issue has label `infra` or asks to execute `terraform apply` or `agents-cli deploy`, do NOT execute cloud mutations; return status `"blocked"` explaining human approval is required.
3. Ensure your workspace starts from `{current_base}` (run `git fetch origin {current_base}` if needed and checkout `{current_base}` cleanly), then create a descriptive branch (e.g., `feat/issue-{issue_number}-...` or `ci/issue-{issue_number}-...`).
4. Implement all requirements cleanly in English-first code (`pt-BR` only for student prompts, Brazilian History corpus, and eval cases) preserving strict local (Ollama) vs. remote (Gemini) parity.
   - If running live Gemini Free Tier benchmarks (`#7`, `#8`), respect `REMOTE_REQUEST_DELAY_S` (e.g., `--remote-delay 6.0`) and built-in 429 retry backoff so free-tier RPM limits are not exceeded.
5. Run full offline verification before committing:
   - `uv run --offline ruff check .`
   - `uv run --offline ruff format --check .`
   - `uv run --offline ty check`
   - `uv run --offline pytest tests/unit -v`
   - `uv run --offline python -m evals.runner --dry-run --reports-dir /tmp/eval_smoke_issue_{issue_number}`
   Commit all changes with an English Conventional Commit message.
{push_instructions}
7. Call `finish` with the exact JSON schema reporting `issue_number`, `status`, `branch`, `base_branch` (`"{current_base}"`), `pr_url` (empty string if not pushed), `verification`, `summary`, and `blocker` (if any).
"""

    kwargs = {
        "schema": RESULT_SCHEMA,
        "role": f"Issue #{issue_number} Worker",
        "type_name": "self",
        "workspace": "share",
    }
    if worker_model:
        kwargs["model"] = worker_model

    result = await agent(prompt, **kwargs)
    log(
        f"Issue #{issue_number} finished -> status={result.get('status')} "
        f"branch={result.get('branch')} base={result.get('base_branch')} "
        f"pr={result.get('pr_url') or 'none'}"
    )
    return result


results = []
active_base = base_branch

for issue_num in issues:
    outcome = await run_single_issue(int(issue_num), active_base)
    results.append(outcome)

    if outcome.get("status") == "completed":
        if chain_branches and outcome.get("branch"):
            active_base = outcome["branch"]
    elif stop_on_block:
        log(
            f"Stopping issue pipeline after Issue #{issue_num} "
            f"(status={outcome.get('status')}, blocker={outcome.get('blocker', 'unspecified')})."
        )
        break

phase("Summary")
for item in results:
    log(
        f"#{item['issue_number']} | {item['status']} | {item['base_branch']} -> {item['branch']} | "
        f"{item['pr_url'] or '-'} | {item['summary']}"
    )

results
