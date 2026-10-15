# Pipeline test-bench notebook — design

*2026-10-16 · branch `feat/pipeline-notebook` · `notebooks/pipeline.ipynb`*

## Purpose

One notebook that runs the real pipeline (Agents 0–8) on **a single paper** and lets
the user swap **embedding models** and **inference models** to see what changes.
It is a test bench, not a demo: edit a settings cell, Run All, read a comparison.

Non-goals: batch evaluation across many papers (that is `scripts/evaluate_*.py`),
a new pipeline, a UI, model downloads, the task planner (Pipeline 2).

## Decisions (made, not asked — each is a one-line change)

| Decision | Choice | Why |
|---|---|---|
| Seed | `SEED` is an arXiv id, URL or local PDF; default `arXiv:0909.0418` (6 references) | Run size is the seed's reference list; no cap knob exists. A free-text idea would make discovery LLM-dependent and the shared stages unfair. |
| Models | Local Ollama by default; `LLM_BACKEND` / `EMBED_BACKEND` are settings | Matches the repo default. An OpenAI-compatible endpoint works by flipping one line. |
| Missing model | Fail in preflight with the exact `ollama pull <name>` | The notebook never downloads a model. |
| Index | `CITATION_INDEX_VERSION=2` | v2 chunks with sentence windows (no embedding model involved), so chunk boundaries are identical across embedders. v1's semantic chunker uses the embedder and would confound the comparison. |
| Corpus | One corpus **per embedding model**, built from the same downloaded PDFs; summary model held fixed | The embedding model must stay fixed per corpus (ARCHITECTURE §6.2). Holding summaries fixed means a corpus depends on the embedder alone. |
| "Inference model" | One value sets `CITATION_LLM_MODEL`, `CITATION_CHAT_MODEL` and `CITATION_JUDGEMENT_MODEL` | Gate, notes, synthesis, need-check, citer, chat and judge all follow it. Without the judge variable a stray `GEMINI_API_KEY` would route the judge to Gemini. |
| Prompts | None. No `input()`; a bad setting raises with the fix | The user does not want to be prompted. |
| Output | Committed with outputs cleared | No bloat, no run data in git. |

## Mechanism

`config.py` reads every model name and path from the environment **at import time**,
and many modules copy those constants by name. So switching models in one kernel
means: set the environment, drop every `research_assistant*` (and `orchestrate`)
entry from `sys.modules`, import afresh. A 10-line helper, `use(embed, llm, data_dir)`,
does exactly that, and every stage cell calls it. `CITATION_LOG_FILE=0` keeps logging
on the console, because `get_logger` keeps the handlers of its first import.

Shared, model-independent stages (**Agents 0, 1, 2**: discover, extract, fetch) run once in
`notebooks/.data/shared`. Each embedder `E` then gets `notebooks/.data/<E>` with `raw/`
and `pulled_pdfs/` symlinked to the shared ones and the three manifests copied (their
paths are absolute, so they stay valid). The nodes `ingest_seed` and `ingest_refs`
(**Agent 3**) build the index there. Every `(E, L)` pair then runs **synthesis**,
**Agent 4** (one sentence), **Agent 5** (cite `DRAFT`), **Agent 8** (verify) and **Agent 7**
(one chat turn). **Agent 6** runs only if `MANUAL_PDF` is set.

## Cells (about 20)

1. Markdown: purpose, how to use, prerequisites, runtime note.
2. Settings: `SEED`, `EMBEDDERS`, `LLMS`, `DRAFT`, `QUESTION`, `MANUAL_PDF`, backends.
3. Isolation and preflight: data dir guard, Ollama reachable, models present, GROBID status.
4. `use()` and a 3-line `stage()` timer.
5–7. Agents 0, 1, 2 (shared).
8. Agent 3 per embedder.
9. Embedder comparison (retrieval, no LLM).
10–15. Per `(E, L)`: synthesis, Agent 4, Agent 5, Agent 8, Agent 7, optional Agent 6.
16. Comparison table, and the full outputs side by side.

Every stage cell ends with an `assert` that names its success criterion.

## What is compared

- **Per embedder (no LLM involved):** ingest seconds, documents, chunks; retrieval **hit@5 and MRR** against
  the seed's own citations (`research_assistant/eval/citer_gold.py`) when a GROBID TEI is available.
  Without a TEI the cell says so and reports top-5 document overlap between embedders instead.
- **Per (embedder, model):** seconds per stage; synthesis length, keys cited, unverified keys; citations
  inserted by Agent 5; Agent 8 verdict counts; chat answer length and `[S#]` keys. The texts themselves are shown.
- Author citations are a floor, not truth (the metrics module says so). The notebook repeats that caveat.

## GEMINI.md, applied

- **Think before coding:** assumptions are the table above and a first-cell checklist.
- **Simplicity first:** one helper, one timer, no config system, no pandas (a plain text table).
- **Surgical:** no existing file is edited. New: the notebook, `notebooks/.gitignore`, `tests/test_notebook.py`.
  The unrelated `research_assistant/agents/lang_orch.ipynb` is left alone.
- **Goal-driven:** per-stage asserts; a static test; two executed cycles.
- **Invariants:** own data dir under `notebooks/.data`, set before import, refuses the real `data/`; no
  `num_ctx` or thread overrides; the real store is never opened.

## Build order and checks

1. `tests/test_notebook.py` first (red): valid nbformat 4, no saved outputs, every code cell parses on
   Python 3.10 grammar, every `from research_assistant… import name` resolves, the data-dir guard refuses
   the real `data/`, and `use()` really switches `config` between two model names.
2. Build the notebook from cell sources.
3. Execute headless on `arXiv:0909.0418`: **cold** (empty `.data`), then **warm** (same `.data`, must resume).
4. Edge cases: empty `SEED`, missing model, data dir pointing at the real store, GROBID down.
5. Read the outputs, not the exit code.

## Known limits

- Only one chat model is installed on the development machine, so the inference axis was verified
  by the `use()` unit test and a one-model run; add models to `LLMS` after an `ollama pull`.
- Running two embedders doubles ingest time. Runtime is dominated by local LLM calls.
- Module purging is pragmatic, not airtight: after an unexpected error, restart the kernel.
