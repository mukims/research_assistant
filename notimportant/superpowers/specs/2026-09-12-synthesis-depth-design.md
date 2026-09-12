# Synthesis depth — read every shortlisted paper, then argue

*2026-09-12*

## 1. Purpose

Tab 1's related-work synthesis (`shared/retrieve.research_answer`, also the
`respond` node of `orchestrate.py`) is the feature Marvin was built for —
"tell me what's already been done on this idea" — and it is shallow in ways
that are visible in one run on the v2 index (2026-09-12, query *"Anderson
localization and transport in disordered quasi-one-dimensional wires"*,
174 s, 475 words):

1. **It reads two papers and calls it six.** Stage 1 shortlists six papers;
   stage 2 (`deep_search`) returns the global top-6 chunks *across* the
   shortlist, and six chunks came from two papers. Four shortlisted papers —
   including the canonical Evers–Mirlin review it had retrieved — contribute
   nothing and are never cited. One paper is cited nine times.
2. **The gap is generic.** "Your research could add significant value by
   applying these established theories…" — the prompt asks for "one or two
   sentences on where this idea might still add something", and that is
   what it gets.
3. **It samples at temperature 1.0.** `research_answer` calls `chat()` with
   no temperature and no options; `gemma4:e2b`'s modelfile sets
   `temperature 1`. A grounded synthesis is being written with full creative
   sampling, and its context window is whatever Ollama's default is rather
   than a chosen number.
4. **The shortlist has duplicates.** The same paper under two document keys
   (an arXiv copy and a DOI copy) takes two of six slots.
5. **Six serial gate calls, six corpus reloads.** `gate_documents` makes one
   model call per summary; `deep_search` calls `load_search_resources()` —
   a full page of the collection out of Chroma plus a 13 MB pickle — every
   time it is called.

`gemma4:e2b` is 5.1 B parameters with a 131k context; the 4k window is a
config choice for CPU sanity, not a model limit. Depth has to come from
reading more, in more calls, and from a prompt that demands an argument
rather than a list.

## 2. Decisions

### A. Per-paper retrieval: every shortlisted paper is read

`passages_per_paper(query, docs, per_paper)` runs one `hybrid_search`
restricted to each shortlisted document (`doc_filter={doc}`,
`exclude_types={"figure_description"}`) and returns `per_paper` chunks for
each. The synthesis sees every paper it shortlisted. This replaces the global
top-k in both modes below. The shortlist is de-duplicated by normalised
title before retrieval, keeping the higher stage-1 score.

### B. Two modes, `map_reduce` the default

`CITATION_SYNTHESIS_MODE=map_reduce|single`.

**`map_reduce`** (default after this spec — the operator asked for depth and
accepted the time):

1. Stage 1: `rank_documents` → batched gate (§D) → de-duplicated shortlist.
   Each paper gets a short key `P1`…`Pn` in shortlist order.
2. **Map** — per paper: its `SYNTHESIS_PER_PAPER_CHUNKS` (4) passages, capped
   at `SYNTHESIS_PER_PAPER_MAX_CHARS` (5,000), and one call with
   `PAPER_NOTES_USER`: at most 150 words in three labelled parts —
   *Establishes* (the specific result with system, conditions, numbers),
   *Method*, *Limits* — or the single line `Not relevant: …`. The paper is
   referred to as its key.
3. **Reduce** — one call with `SYNTHESIS_USER` over all notes, 350–500
   words under three fixed headings: **What is established** (by theme,
   every sentence cites keys), **Where the papers differ** (both sides
   cited, or "none" in one sentence), **The gap** (concretely what none of
   the notes cover that the idea needs, and what evidence would close it —
   no general statements of the idea's value). Papers whose notes say *Not
   relevant* are not cited.
4. `1 + N + 1` model calls. With six papers on the CPU machine: roughly
   1 + 6×40 s + 90 s ≈ 6 minutes, against ~3 today. The map calls run at
   `num_ctx` 4,096; the reduce at 8,192 (as the judge already does), because
   eight papers' notes plus a 600-token answer do not fit in 4k.

**`single`** — one call, as today, but over the per-paper passages from §A
(one or two per paper) instead of the global top-k, with the same headings.
The fast path, and a fair comparison for the live check.

Both modes run at `SYNTHESIS_TEMPERATURE` (0.2) with explicit `options`.

### C. Citations are keys, and keys are checked

The model cites `[P3]`, not `[Long paper title, p.16]`. Short keys are what a
5 B model reproduces exactly, and they are checkable: after the reduce,
`check_citation_keys(text, valid_keys)` extracts every `[Pn]` and reports
those that are not on the shortlist. The result carries `keys` (key →
title), `citations` (titles actually cited, in key order), and
`unverified_citations` (keys the model invented). The UI shows a legend and
a warning when the list is non-empty. A cited key whose notes say *Not
relevant* is also flagged.

### D. Batched gate (rag-engine-v2 spec §3.2)

One call listing the shortlisted summaries numbered, answered as `N: YES` /
`N: NO`. Parsing requires exactly one verdict per input, in order; on any
misalignment the existing per-summary calls run instead. Same rule as Agent
5's batched citation-need check: never guess which verdict belongs to which
item. Falls back to the top summary when everything is rejected, as today.

### E. Search-resource cache (rag-engine-v2 spec §3.1)

`shared/db.load_search_resources()` is memoised per process, keyed on
`(VECTORDB_PATH, COLLECTION_NAME, BM25_INDEX_PATH, mtime of the pickle)`; a
rebuilt index invalidates it; `clear_search_cache()` exists for tests and
for the ingest path. `passages_per_paper` calls `hybrid_search` N times per
query; without the cache that is N full reloads.

### F. Progress and the record

`research_answer(query, top_k=None, mode=None, on_progress=None)`. The
callback receives `(stage, payload)` events — `shortlist`, `notes` (per
paper, with the key and title), `synthesis` — so the app's `st.status`
shows each paper being read. The return keeps today's keys (`suggestion`,
`citations`, `passages`, `selected`) and adds `mode`, `keys`, `notes`
(`[{key, document, citation, notes, relevant, passages_used}]`),
`unverified_citations`, `timings` (`gate`, `map`, `reduce`, `total`
seconds). `orchestrate.respond` passes no callback and is otherwise
unchanged. The Tab 1 renderer shows the notes under each shortlisted paper
and the key legend under the synthesis.

## 3. Not changed

Research chat (Agent 7) — its depth problem is the window, not the prompt,
and it is a different design (query condensation, history budget). The
reranker (spec §3.3) and neighbour expansion on this path (§3.4) — the
per-paper budget is spent on more chunks of the paper rather than
neighbours of one. Stage-1 ranking and `DOC_SELECT_K` (still 6; the operator
can raise it — the cost is linear in N).

## 4. Configuration

| variable | default | meaning |
|---|---|---|
| `CITATION_SYNTHESIS_MODE` | `map_reduce` | or `single` |
| `CITATION_SYNTHESIS_PER_PAPER_CHUNKS` | `4` | chunks retrieved per shortlisted paper (map); `single` uses 2 |
| `CITATION_SYNTHESIS_PER_PAPER_MAX_CHARS` | `5000` | cap on a paper's passages in the map prompt |
| `CITATION_SYNTHESIS_TEMPERATURE` | `0.2` | both modes, both stages |
| `CITATION_GATE_BATCHED` | `1` | batched relevance gate; `0` restores per-summary calls |
| `SYNTHESIS_OLLAMA_OPTIONS` (code) | `{"num_ctx": 8192}` | the reduce call; map calls use `CHAT_OLLAMA_OPTIONS` (4,096) |

## 5. Risks

- **Slower Tab 1.** ~2× today's synthesis time by construction. Accepted;
  `single` remains one env var away, and the UI shows progress per paper so
  the wait is legible.
- **Notes can be wrong** — a 5 B model summarising four chunks can misstate
  a number. The notes are shown per paper next to the passages they came
  from, so a reader can check; and the judge (Agent 8) exists for the draft
  that comes later.
- **Invented keys** are caught by §C. Invented *content* under a real key
  is not — that is what the judge evaluation set measures, downstream.
- **`num_ctx` 8,192 for one call per query on the CPU VM.** The judge already
  runs at 8,192 there; GEMINI.md's 4,096 rule is about the chat model's
  per-turn cost, and this is one call.

## 6. Verification

No harness exists for synthesis quality. The live check compares `single`
and `map_reduce` on the same three ideas, on the v2 index, against
code-checkable criteria: papers cited ≥ 4 of the shortlist (today: 2 of 6),
the three headings present, no unverified keys, no *Not relevant* paper
cited, word count in range, timings recorded — and the operator reads both.
