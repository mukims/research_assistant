# Instructions for Antigravity: execute the Technique transfer plan

You are implementing `notimportant/superpowers/plans/2026-10-21-technique-transfer.md`.
Its design and reasons are in
`notimportant/superpowers/specs/2026-10-21-technique-transfer-design.md`. Read both in
full before changing anything. The plan is the source of truth. Where this page and
the plan disagree, follow the plan and say so in your report.

## 0. What changed

- **The old plan is superseded.** The hypothesis-search plan (`2026-10-21-hypothesis-search*.md`) is replaced. Do not resume it.
- **Only its Task 0 carries over.** That work is already on `feat/hypothesis-search`: the MCTS draft is on `wip/mcts-draft` and `.ENV` is ignored. This plan branches from there.
- **The old draft stays where it is.** Do not reuse, copy or delete the code on `wip/mcts-draft`. Task 15 may read it for reference only.

## 1. Preconditions — check them, don't fix them

Stop and report to the maintainer if any of these fails:

1. You are in `/run/media/shardul/storage/marvin/migration-source/research_assistant`. The parent folder is not a git repository.
2. `git branch --show-current` prints `feat/hypothesis-search`.
3. `git status --short` prints nothing.
4. `git check-ignore -v .ENV` prints a line naming `.gitignore`.

## 2. Rules for the whole run

- **Python:** use `/home/shardul/miniconda3/envs/ml/bin/python` wherever the plan writes `python`. Set `CITATION_LOG_FILE=0` for every test run.
- **Staging:** stage only the exact paths each task lists. Never run `git add -A`, `git add .` or `git commit -a`.
- **History:** never push, merge or rebase. Commit locally with the message each task gives.
- **Off limits:**
  - `agents/agent0_discoverer.py`, `agents/agent5_batch_citer.py`, `agents/agent8_verifier.py`
  - `research_assistant/judgement/`
  - `research_assistant/shared/ingestion.py`
  - `data/`, except where a task's live step writes there
  - the running migration process

  Call these; never edit them. Add no dependencies.
- **Test first.** For every task, write the tests exactly as the plan gives them and watch them fail. Then implement and watch them pass. If a test in the plan looks wrong, stop and report it. Never edit a test just to make it pass.
- **One task at a time, in order.** After each task:
  1. Run the task's own test command.
  2. Run the whole suite: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/ -q`.
  3. Commit only if both pass.
  4. If something you didn't touch fails, stop and report it.
- **No live steps.** Never run a model, card build, case-corpus build, retrodiction run or network call yourself. Live steps are marked "maintainer" in the plan: list each one in your report with its exact command, and move on only where this page allows. The unit tests use fakes, and are always safe to run.
- **Private material stays private.** Never write anything about the maintainer's unpublished work into a tracked file. The answer keys live only under `data/eval/transfer/answer_keys/`, and the maintainer writes them.
- **Ambiguity:** if a step can be read two ways, take the reading that matches the spec, and note the choice in your report.

## 3. Order and stopping points

| Step | Tasks | Then |
|---|---|---|
| A | Task 0 | Continue. |
| B | Tasks 1–5 (Steps 1–5 of each) | **STOP at Gate F1.** Report. The maintainer builds 20 cards, rates them, and tells you whether F1 passed. |
| C | Tasks 6–12 (Steps 1–5 of each) | Continue without stopping while each task's tests and the full suite pass. Skip Task 12, Step 6 (live), and list it. |
| D | Task 13, Steps 1–5 | Continue. List Step 6 (live) for the maintainer. |
| E | Task 14, Steps 1–5 | **STOP at Gates F2 and F3.** The maintainer runs them and calibrates. |
| F | Task 15 | **Only if the maintainer says F1–F3 passed and asks for Phase 2.** Then **STOP at its gate.** |
| G | Tasks 16–17 | Only when the maintainer says to go on. If Task 15 was skipped, drop `research_assistant.transfer.mcts` from the Task 17 sweep. |

If the maintainer reports a gate as failed, do not continue to the next phase. Wait for
the fix they ask for; it will usually change cards, prompts or thresholds.

## 4. Report at each stopping point

Keep it short, and include:

- The commits made: `git log --oneline feat/hypothesis-search..HEAD`.
- The test counts from the last full-suite run, and any failure verbatim.
- Every place you departed from the plan, and why.
- Every live step you skipped, with the exact command the maintainer should run, in order.
