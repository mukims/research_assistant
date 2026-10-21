# Instructions for Antigravity: execute the Hypothesis search plan

> **Superseded** by `2026-10-21-technique-transfer-design.md` / `2026-10-21-technique-transfer.md`.

You are implementing `notimportant/superpowers/plans/2026-10-21-hypothesis-search.md`.
Its design and the reasons behind it are in
`notimportant/superpowers/specs/2026-10-21-hypothesis-search-design.md`. Read both in
full before you change anything. The plan is the source of truth: where this page and
the plan disagree, follow the plan and say so in your report.

## 0. Stop the previous work first

The earlier "MCTS Scientific Hypothesis Tree Engine" plan is **superseded**. Do not
continue it, extend it, or fix its tests. The files it produced stay exactly as they are
until Task 0 moves them to a side branch:

- `research_assistant/brainstorm/`
- `research_assistant/ui/mcts_tab.py`
- `scripts/mcts_hypothesis.py`
- `tests/test_mcts.py`
- `output/`
- the tab edit in `app.py`

## 1. Preconditions — check them, don't fix them

Stop and report to the maintainer if any of these fails:

1. You are in `/run/media/shardul/storage/marvin/migration-source/research_assistant`. The parent folder is not a git repository.
2. `git status --short` shows no modification to `research_assistant/shared/llm.py` or `scripts/migrate_to_v3.py`. Those are the maintainer's v3 changes; they commit or stash them, never you.
3. The current branch is `feat/v3-gemma-images`.

## 2. Rules for the whole run

- **Python:** `/home/shardul/miniconda3/envs/ml/bin/python`. Where the plan writes `python -m pytest`, run `/home/shardul/miniconda3/envs/ml/bin/python -m pytest`, with `CITATION_LOG_FILE=0` set.
- **Staging:** stage the exact paths each task lists. Never run `git add -A`, `git add .` or `git commit -a`. `.ENV` holds the Gemini key and is not ignored until Task 0, Step 4.
- **Pushing:** never push, merge or rebase. Commit locally only, with the message each task gives.
- **Off limits:** `agents/agent5_batch_citer.py`, `agents/agent8_verifier.py`, `research_assistant/judgement/`, `data/` and the running migration process. Don't add dependencies.
- **Test-first:** for every task, write the tests exactly as the plan gives them and run them to see them fail. Then implement and run them to see them pass. If a test in the plan looks wrong, stop and report it; don't edit a test to make it pass.
- **One task at a time, in order.** After each task:
  1. Run that task's test command.
  2. Run the whole suite: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/ -q`.
  3. Commit only if both pass.
  4. If something you didn't touch fails, stop and report it.
- **No live model calls** unless both are true: the maintainer has said the migration is finished, and `pipeline_status` shows no active run. Check with `PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python -c "from research_assistant.shared import pipeline_status as p; print(p.get_status()['active'])"`, which must print `False`. All unit tests use fakes and are fine to run at any time.
- **Ambiguity:** if a step can be read two ways, take the reading that matches the spec. Note the choice in your report rather than inventing behaviour.

## 3. Order and stopping points

| Step | Tasks | Then |
|---|---|---|
| A | Task 0 | **Stop.** Report `git branch`, `git status --short` and the result of `git check-ignore -v .ENV`. Wait for the maintainer before continuing. |
| B | Tasks 1–6 | Continue without stopping, as long as each task's tests and the full suite pass. |
| C | Task 7, Steps 1–4 and 6 | Skip Step 5 (the live check) and list it in your report as "waiting for the maintainer". |
| D | Task 8, Steps 1–5 | Skip Step 6 (the live eval). **Stop at the checkpoint.** The maintainer runs the eval and decides whether Tasks 9–11 go ahead. |
| E | Tasks 9–11 | Only when the maintainer says to go on. |

Do not start Phase 2 (MCTS). The plan does not schedule it.

## 4. Report after each stopping point

Keep it short, and include:

- The commits made: `git log --oneline feat/v3-gemma-images..HEAD`.
- The test counts from the last full-suite run, and any failure verbatim.
- Every place you departed from the plan, and why.
- Every step you skipped (the live checks), with the exact command the maintainer should run.
