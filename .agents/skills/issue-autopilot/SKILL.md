---
name: issue-autopilot
description: >-
  Execute a sequence of GitHub issues automatically with a fresh, isolated context per issue
  using a `run_workflow` pipeline. Use when the user asks to work through multiple issues in
  order, auto-advance to the next issue after finishing one, or run an autonomous issue queue.
---

# Issue Autopilot (Fresh Context Per Issue)

Instead of accumulating multiple issues inside one conversation and compacting manually, this
skill runs each GitHub issue in its own **fresh-context agent** via `run_workflow` + `pipeline`.
When Issue N finishes and opens a Pull Request, the pipeline automatically launches Issue N+1
with a clean context window.

## Safety Gates (Mandatory)

- **Never auto-run human-gated issues:** Skip issues labeled `infra` or mentioning `terraform apply`
  or `agents-cli deploy` unless the user explicitly approves them in advance.
- **Strict verification:** Every worker agent MUST pass `make check` (`ruff` + `pytest tests/unit`)
  and `uv run ty check` before committing or opening a PR.
- **No hook bypass:** Never use `git push --no-verify` or `SECURITY_GATE_SKIP=1`.
- **No silent model changes:** Preserve `DEFAULT_GEMINI_MODEL` and strict `local` vs. `remote` parity.

## Execution Steps

1. **Discover the Issue Queue:**
   List open issues and filter out human-gated ones (such as `infra`) unless explicitly approved:
   ```bash
   gh issue list --state open --json number,title,labels --limit 30
   ```
2. **Confirm Base Branch State:**
   Ensure `main` is up to date so each isolated worker workspace branches cleanly from the latest
   merged code.
3. **Launch the Pipeline Workflow:**
   Call `run_workflow` using the bundled script:
   - `ScriptPath`: `<workspace>/.agents/skills/issue-autopilot/scripts/issue_pipeline.py`
   - `Description`: `"Run Brasão AI GitHub issue pipeline with fresh context per issue"`
   - `Args`:
     ```json
     {
       "issues": [2, 3, 4],
       "base_branch": "main",
       "auto_push": true,
       "stop_on_block": true
     }
     ```
   - `issues` (required): Ordered list of GitHub issue numbers to execute sequentially.
   - `base_branch` (optional, default `"main"`): Base branch for new feature branches and PRs.
   - `auto_push` (optional, default `true`): Whether each worker pushes its branch and opens a PR via `gh pr create`.
   - `stop_on_block` (optional, default `true`): Stop the remaining queue if an issue returns `"blocked"` or `"failed"` (recommended when later issues depend on earlier ones).
4. **Review Results:**
   When `run_workflow` completes, present a concise table of `(issue_number, status, branch, pr_url, summary)` and surface any blockers requiring human input.

## Tips

- For unattended multi-issue runs, recommend the user trigger `/goal` alongside this skill.
- If an issue requires interactive credentials (e.g., SSH passphrase, Gemini API key, or Ollama download), the worker returns `status: "blocked"` with clear instructions instead of hanging.
