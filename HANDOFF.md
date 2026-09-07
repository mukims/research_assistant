# Handoff — merged Research Assistant

*Written 2026-09-07, mid-execution. Read this first when picking the work back up.*

## What this project is

One project merged from two existing repos that share a lineage but diverged:

- **`/Users/shardul/Downloads/tech_ireland`** — better *infrastructure*. Its `shared/`
  layer is a strict improvement on the other repo's in every module the two have in
  common, and it adds four modules the other lacks. Also has GROBID extraction, a
  LangGraph orchestrator, a Streamlit UI and Docker packaging.
- **`/Users/shardul/Downloads/citation_builder`** — better *features*. Agents 5 (batch
  citer), 6 (manual ingestor) and 7 (research chat), plus a real test suite and CI,
  none of which survived into `tech_ireland`.

The merge takes `tech_ireland` as the base and ports `citation_builder`'s agents and
tests onto it, restructured as an installable package.

**Both source repos are read-only in this work. Nothing modifies them.**

## Where things stand

**Location:** `/Users/shardul/Downloads/research_assistant`
**Branch:** `build/merged-pipeline` (branched from `main` @ `c727d06`)
**Tests:** green — 10 passed
**Working tree:** clean

```
8ca2419  feat(schemas): Add typed dataclass contracts for pipeline stages   ← Task 2
7474bd6  feat: scaffold research_assistant package, packaging, and merged config  ← Task 1
c727d06  docs: restore per-task commits to the plan
da7af8d  docs: design spec and implementation plan for the merged pipeline
```

### Done

| Task | What landed | State |
|---|---|---|
| **1 — Scaffold + config** | Package skeleton, `pyproject.toml` (editable install), `config.py`, `prompts.py`, `shared/log.py`, `shared/retry.py`, requirements files, `model_final.pth` symlink | ✅ committed, reviewed clean |
| **2 — `schemas.py`** | `Reference`, `DownloadedPaper`, `SeedPaper` dataclasses + `SchemaError`, with `to_dict`/`from_dict`; `tests/test_schemas.py` | ✅ committed, tests green, **review not yet run** |

### Not done

Tasks 3–14. Nothing has been started on them.

## Two real bugs already fixed

1. **`DETECTRON_CONFIG` had inverted `os.environ.get` arguments** in `tech_ireland`'s
   newest commit — the model-zoo URL was being used as the *environment variable name*,
   so the value was always `"publaynet_config.yaml"` and the `lp://` path was
   unreachable. Fixed in Task 1; `config.py` now reads
   `os.environ.get("CITATION_DETECTRON_CONFIG", "lp://PubLayNet/...")`.
2. **`retry.py` had a stale `from shared.log import get_logger`** that the plan
   wrongly claimed needed no changes. Caught during Task 1.

## The two documents that govern the work

Read these in order. They are committed in the repo.

1. **Spec** — `docs/superpowers/specs/2026-09-07-merged-research-assistant-design.md`
   *Why* each component comes from which repo, and the design of the new pieces.
   This is the binding authority.
2. **Plan** — `docs/superpowers/plans/2026-09-07-merged-research-assistant.md`
   14 tasks, each with the actual code to write and a runnable verification.
   The plan argues from the spec; where they conflict, the spec wins.

## Remaining tasks

| # | Task | Notes |
|---|---|---|
| 3 | `shared/manifest.py` | Replaces an append-only text file that lived *inside* the ChromaDB directory, and a full collection scan that ran on every ingest. TDD, complete code in the plan. |
| 4 | `shared/llm.py` + `chat_stream()` | The one genuinely new piece of infrastructure. Agent 7 streams; the backend abstraction needs to as well, or an agent bypasses it. |
| 5 | Port `db`, `fetch`, `source_key`, `search`, `retrieve` | Mechanical — copy from `tech_ireland`, rewrite imports. Watch the *indented* function-local imports; the `sed` rule only catches line-start ones. |
| 6 | `shared/ingestion.py` rewired to the manifest | |
| 7 | **Agent 1 — GROBID + regex fallback** | The hardest task. See below. |
| 8 | Agents 0 and 4 | Mechanical port. |
| 9 | Agents 2 and 3 + schema wiring | Deletes agent 3's legacy-shape shim. |
| 10 | Agents 5, 6, 7 ported onto `shared.llm` | After this, no module outside `shared/llm.py` imports `ollama` or `openai`. |
| 11 | `orchestrate.py` | Removes the GROBID dead-end branch. |
| 12 | `watch.py` | From `master_orchestrator.py`. |
| 13 | `app.py` + 2 new tabs | Batch-cite tab and research-chat tab. |
| 14 | Docs, CI, final verification sweep | |

### The interesting one: Task 7

The two repos extract references incompatibly and each has the other's weakness.
`tech_ireland` posts PDFs to a GROBID server and parses TEI — much richer output
(per-reference DOI, authors, year, plus a confidence score for each consolidated DOI),
but it needs a Java service, and when that service is down the pipeline *dead-ends*.
`citation_builder` runs `pdftotext` and matches three reference-line regexes — weaker,
but no external dependency.

Merged: **GROBID primary, regex fallback per-PDF.** The pipeline degrades instead of
stopping. Task 11 then deletes the `GROBID down → END` edge from the orchestrator.

## Rulings made so far

These were decisions taken on your behalf during a pre-flight scan of the plan.
Each is recorded in the ledger with what it costs if wrong. Review them — anything you
disagree with is cheap to reverse now and expensive later.

| # | Ruling | Cost if wrong |
|---|---|---|
| — | Work on branch `build/merged-pipeline`, not a git worktree — the repo is new and `main` holds only docs, so there is nothing to isolate from | None material |
| F1 | **Task 7 must batch GROBID.** As written, the plan would make one round trip per PDF, losing `GROBID_BATCH_CONCURRENCY`. `run_extractor()` should make one batch pre-pass, after which the per-PDF calls find cached TEI | Slower extraction, no correctness impact |
| F2 | **Task 7's return key renamed to `reference_count`.** The plan used `references` for both a *list* (in the JSON) and an *int* (in the return dict) in one function | Confusing but working overload |
| F3 | **Task 9 must establish a `provider` variable.** `DownloadedPaper` requires it, but the plan never verified the copied fetcher tracks which source succeeded | One fix round |
| F4 | **Task 6 introduces `ManifestBackedTestCase`** in the ported test file — the plan's new test names a base class that may not exist there | Cosmetic test churn |
| F5 | Task 1's brief showed a placeholder import line; apply the import rule mechanically instead | Caught immediately by verification |
| F6 | **Test counts in the plan are estimates, not targets.** Assert on green/red, never on a number | None |

## Known open items

- **Task 2's review was never run.** The code is committed and its tests pass, but it
  has not been through the review gate the other tasks get. Run that before treating
  Task 2 as fully done.
- **One deferred minor from Task 1:** `prompts.py`'s docstring lost a clause explaining
  *why* prompts were centralised. Cosmetic.
- **Nothing has been verified against a live model or a real GROBID server.** All
  verification so far is offline — tests plus import sweeps. A live end-to-end run
  needs your Ollama daemon up.

## How to resume

The work was running under a task-by-task loop: dispatch an implementer per task,
review the diff, then commit. Progress is tracked in a ledger at:

```
.superpowers/sdd/2026-09-07-merged-research-assistant/progress.md
```

That directory is git-ignored scratch, but the ledger is the recovery map — it names
every commit and every ruling. Trust it and `git log` over memory.

To continue, pick up at **Task 3** (or run Task 2's outstanding review first). Each
task's brief can be regenerated from the plan.

Verify the current state at any point with:

```bash
cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 .venv/bin/python -m pytest tests/ -v
```
