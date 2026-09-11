# RAG engine v2 — measured retrieval, clean ingestion, depth

*2026-09-11*

## 1. Purpose

Three symptoms, reported from use: the retrieved passages are often wrong or
missing; the synthesis and chat answers are shallow; Tab 1 is slow on the VM.
Two constraints: `gemma4:e2b` stays the generator, and the VM stays CPU-only
(4 vCPU / 12 GB). Re-ingestion and new Python dependencies are allowed.

The diagnosis, from the local 323-paper corpus (36,609 chunks):

- **Chunks are fragments.** Median chunk is 242 characters; 45% are under 200.
  A random sample of the small half is bibliography entries chopped into
  pieces and font-map garbage. BM25's length normalisation ranks those
  fragments above real passages on exact-term hits; their vectors sit near any
  query in the same field.
- **The index is partly blind to its own words.** 31% of chunks contain a
  ligature-broken word (`ﬁeld` ×2,379, `diﬀerent`, `ﬁrst`, `deﬁned`, `ﬁnite`,
  `ﬂuctuations`, `eﬀect`) because `\w+` treats U+FB01 as a letter; 23% contain
  a mid-word hyphen break (`quan- tum`). The keyword half of hybrid search
  cannot match "field" against `ﬁeld`.
- **The generator is starving.** `research_answer()` hands the model at most 6
  chunks — ~400 tokens of evidence in a 4,096-token window. Agent 8 judges
  each citation from one ~240-character chunk.
- **Figures are indexed badly.** 3,769 figure/table entries (10% of the
  index): all 548 tables are caption-less placeholders (`find_caption` has no
  table branch); 966 of 3,221 figure "captions" are body text that happened to
  sit below the crop; nothing consumes the 4,029 PNG crops (239 MB).
- **Nothing is measured.** `ARCHITECTURE.md §10` deferred the evaluation
  harness. No change to retrieval can currently be shown to help.

A spike (GROBID `processFulltextDocument` vs PyMuPDF on four of the corpus's
PDFs, same chunker on both) established that GROBID separates the bibliography
on 4/4 papers where the heading heuristic managed 1/3, removes hyphenation and
ligature artefacts entirely, and yields section headings and figure/table
captions that PyMuPDF cannot; it does not rescue a PDF whose font map is
broken (neither does anything else — the alphabetic-ratio filter drops it),
and it lifts formulas out of paragraphs (which were glyph soup inline anyway).

### Scope

Four phases, in this order, each measured against the harness built in the
first:

1. **Harness** — `evaluate_rag.py`, synthetic + golden sets, metrics.
2. **Query-time layer** — no re-ingest: resource cache, batched gate,
   cross-encoder reranker, neighbour expansion, NFKC/stemmed BM25 tokenizer,
   nomic task prefixes (v2 only).
3. **Ingestion v2** — GROBID-first extraction, filters, sentence-window
   chunker, figure/table handling, versioned index.
4. **Depth** — map-reduce synthesis on the Tab 1 path.

Each phase gets its own implementation plan. **Order as decided on
2026-09-11: ingestion v2 (§4, including figure analysis §4.6) is built
first**, carrying with it the two index-construction items from §3 — the
BM25 tokenizer (§3.5) and the nomic task prefixes (§3.6) — so the v2 index
is built once. The harness (§2) is deferred; until it exists, v2's gains are
asserted from the diagnosis above and judged in the app, not measured. The
remaining query-time items (§3.1–3.4) and depth (§5) follow. All work is
local against the 323-paper corpus. **Deploying to the VM is a
separate decision at the end of each phase**, taken by the operator, not folded
into any phase. Nothing in this spec writes to the v1 index; v1 collections,
BM25 pickle and manifest are read-only to v2 code paths.

### Out of scope

Chat-query condensation across turns; multi-hop reference walking (the
corpus stays one hop from the seed); a larger or hosted generator; equation
search; parent/child hierarchical chunks; UI changes beyond the one checkbox
in §4.6 and progress reporting.

---

## 2. Evaluation harness — `evaluate_rag.py`

A root-level script (alongside `orchestrate.py`, `watch.py`), because it is
something a person runs. Its pure helpers live in
`research_assistant/eval/` so they are importable and testable.

### 2.1 Synthetic set

`data/eval/synthetic.jsonl`, generated once and cached; regenerated only with
`--regenerate`. Procedure:

1. Take every PDF in `data/pulled_pdfs/` and `data/raw/`; run
   `shared.extract.extract(pdf)` (§4.1) — GROBID with PyMuPDF fallback.
2. Sample 200 body paragraphs of 300–1,500 characters from ~100 papers, at
   most 3 per paper, excluding abstract and any section whose normalised name
   is `other` when a named section is available.
3. For each, one `gemma4:e2b` call: *"Write one specific question a physicist
   might ask that this passage answers. Return only the question."*
   Temperature 0.
4. Record `{id, question, document, section, truth_text, truth_page}`.

Ground truth is the paper plus the paragraph text — **not** a chunk id — so
v1 and v2 indexes, which chunk differently, are scored on the same footing.

### 2.2 Golden set

`data/eval/golden.jsonl`, hand-written by the operator:
`{query, expected_documents: [...], notes}`. The harness ships a template with
two worked examples and treats the file as optional; results for it are
reported separately and never averaged into the synthetic numbers.

### 2.3 Figure subset

When the v2 index contains caption chunks, an additional 50 synthetic
questions are generated from captions (`{…, kind: "figure"}`), reported as
their own row, so the contribution of §4.6 is a number.

### 2.4 Metrics

Per query, over the ranked list returned by the path under test:

- **Document recall@1/3/5** — the truth document appears in the top-k.
- **MRR** over documents.
- **Passage hit@3** — some chunk in the top-3 shares ≥50% of its token set
  (NFKC-normalised, lowercased, `\w+`) with the truth paragraph, *and* is from
  the truth document.
- **Latency** — median and p90 wall-clock per query, retrieval only (no
  generation), measured after one warm-up query.

Paths: **flat** (`hybrid_search`, `top_k=5`) and **two-stage**
(`rank_documents → gate → deep_search`, `top_k=6`).

### 2.5 Runs and output

```
python evaluate_rag.py --index v1 --rerank off --expand off
python evaluate_rag.py --index v2 --rerank on  --expand on --model minilm
python evaluate_rag.py --compare results/a.json results/b.json
```

Each run prints a markdown table and writes
`data/eval/results/<timestamp>-<index>-<flags>.json` with every per-query
result, so two runs can be diffed query by query. `--compare` prints the two
tables side by side with deltas. Reranker model choice is a flag so the
harness picks between the two candidates in §3.3.

---

## 3. Query-time layer (phase 2 — no re-ingest)

Every item here is behind an env var and is a no-op when it is off.

### 3.1 Search-resource cache

`shared/db.load_search_resources()` is memoised per process, keyed on
`(VECTORDB_PATH, COLLECTION_NAME, BM25_INDEX_PATH, mtime(BM25_INDEX_PATH))`.
A rebuilt index (new mtime) invalidates it. `retrieve.deep_search()` today
calls it on every query and pages the whole collection out of Chroma; this
is the single largest fixed cost on the Tab 1 path.

Agent 7 already caches the tuple on its object; it keeps doing so and simply
gets the cached tuple.

### 3.2 Batched relevance gate

`retrieve.gate_documents()` sends **one** prompt listing the shortlisted
summaries as a numbered list and asks for one line per item, `N: YES` or
`N: NO`. Parsing requires exactly one verdict per input in order; on any
misalignment (missing number, duplicate, extra) it logs and falls back to the
existing per-summary calls. Same principle as Agent 5's batched
citation-need check: never guess which verdict belongs to which item.

New prompt `DOC_RELEVANCE_GATE_BATCH` in `prompts.py`; the single-item prompt
stays for the fallback.

### 3.3 Cross-encoder reranker

In `shared/search.hybrid_search()`, after RRF fusion and before the final
cut: take the top `RERANK_CANDIDATES` fused results, score `(query, text)`
pairs with a cross-encoder, and return the top `top_k` ordered by that score.
Each result gains `rerank_score`; `rrf_score` is kept.

- Implementation: `fastembed.rerank.cross_encoder.TextCrossEncoder`
  (ONNX runtime; no torch). Wrapped in `shared/rerank.py` behind a
  process-wide singleton, imported function-locally so the package imports
  without `fastembed` installed.
- Models, measured on the development machine for 20 × 1,200-char pairs:
  `Xenova/ms-marco-MiniLM-L-6-v2` — 0.29 s, ~440 MB RSS;
  `jinaai/jina-reranker-v1-turbo-en` — 0.38 s, ~490 MB.
  `BAAI/bge-reranker-base` (2.0 s, ~2 GB) is excluded on VM memory.
  Default MiniLM; the harness decides whether jina earns its extra cost.
- `RERANK_CANDIDATES` defaults to 20 and is never less than `top_k`.
- The model is downloaded on first use to `CITATION_DATA_DIR/models/`
  (persistent disk on the VM). Absent model or import failure → warn once,
  return the RRF order. A query never fails because of the reranker.
- Callers are unchanged. Agent 5's per-sentence loop pays the reranker cost
  per sentence; that is accepted (citations are where precision matters) and
  `CITATION_RERANK=0` turns it off globally.

### 3.4 Neighbour expansion

`shared/search.expand_neighbours(results, texts, metadatas, window=1)`:
for each result, attach the chunks at `chunk_index ± window` **if they have
the same `document`**. Returned as `result["context_before"]` /
`result["context_after"]` (text only), so callers that don't want them see
no change; `metadata["seq"]` is used when present (v2), else the chunk index
(v1 — a paper's chunks are minted contiguously in ingest order, so an
adjacent index with the same `document` is the next chunk of that paper:
page order for text, with that paper's figure entries preceding its text).

Used at context assembly by `retrieve.research_answer()`, Agent 7 and
Agent 8. Not used by Agent 4 or Agent 5 (one sentence, speed).

### 3.5 BM25 tokenizer

`shared/tokenize.py`:

```
tokenize(text) -> list[str]
```

NFKC normalisation → lowercase → `\w+` → drop tokens shorter than 2 chars or
in a small English stopword list → Snowball (`snowballstemmer`, pure Python)
English stem. **The same function tokenises queries** in `hybrid_search`.

Applied in `ingestion.rebuild_bm25()`, which reads chunk text back out of
Chroma — so **one rebuild repairs the keyword side of the v1 index without
re-embedding anything**. The BM25 pickle becomes
`{"tokenizer": "v2", "built_at": …, "bm25": <BM25Okapi>}`;
`load_search_resources()` accepts both this and today's bare `BM25Okapi`
object (treated as tokenizer `v1`) and warns when the loaded tokenizer
does not match the code's, since a mismatch silently degrades keyword
recall.

### 3.6 nomic task prefixes

`nomic-embed-text` expects `search_document: ` on indexed text and
`search_query: ` on queries. `shared/llm.get_embeddings()` returns a thin
wrapper that prepends them when the model name starts with `nomic-embed`
**and `INDEX_VERSION >= 2`**. v1 vectors were computed without the prefix;
mixing prefixed queries with unprefixed documents is worse than neither, so
the gate on version is load-bearing.

---

## 4. Ingestion v2 (phase 3)

### 4.1 Extraction — `shared/extract.py`

One entry point, one output shape:

```python
@dataclass
class Paragraph:  text: str; page: int
@dataclass
class Section:    heading: str; kind: str; paragraphs: list[Paragraph]
@dataclass
class Figure:     label: str; caption: str; page: int; bbox: tuple | None; kind: str  # figure|table; refs: list[str]
@dataclass
class Document:   key: str; title: str; abstract: list[Paragraph]; sections: list[Section]
                  figures: list[Figure]; extraction: str   # "grobid" | "pymupdf"

def extract(pdf_path: str) -> Document
```

**GROBID path.** `POST /api/processFulltextDocument` with
repeated `teiCoordinates` form fields for `p`, `head` and `figure` (paragraph coordinates give the page as the
first field of `coords`; `<graphic coords>` gives the bitmap box for §4.6).
TEI cached at `raw/grobid_output/<pdf>.tei.xml` — the location Agent 1
already uses, so a seed paper is never sent twice. `<listBibl>` is read for
its count and otherwise ignored. Formulas are dropped from paragraph text.
Section `kind` is normalised from the heading by keyword
(`abstract, introduction, background, methods, results, discussion,
conclusion, other`); `heading` keeps the raw text. Body sentences that
contain `<ref type="figure" target="#fig_N">` are collected into
`Figure.refs` for that figure (§4.6).

**PyMuPDF path** (GROBID unreachable, timeout, non-200, or empty body):
`get_text("blocks")` per page, NFKC-normalised, then: blocks repeated on
≥30% of pages dropped (running headers/footers); text cut at the last
block matching `^(references|bibliography|literature cited)$`; one section
of kind `other`; no figures. Recorded as `extraction="pymupdf"`.

Which path ran is recorded per chunk (`extraction` metadata) and counted in
the ingest report, exactly as Agent 1 reports `grobid | regex | none`.

**GROBID concurrency.** Ingestion already serialises on `ingest.lock`; within
a run, PDFs go to GROBID with at most `GROBID_BATCH_CONCURRENCY` in flight
(existing setting). A 300 s per-document timeout; a timeout is a fallback,
not a failure.

### 4.2 Filters — `shared/chunking.py`

Applied before windowing, to paragraphs:

- alphabetic ratio (`isalpha() or isspace()` over length) < 0.6 → dropped
  (font-map garbage);
- section `kind` in `{acknowledgements, funding, author contributions,
  competing interests, data availability}` → dropped (matched by heading);
- GROBID `<head>` without an `n` attribute whose text matches a running
  header pattern (`^\(\d+ of \d+\)$`, `et al\.$`, a bare page number) →
  dropped as a section boundary, its paragraphs joined to the previous
  section;
- sections whose total text is under 200 characters → merged into the next
  section.

Applied after windowing, to chunks: under 200 characters → dropped unless
`type=caption`.

### 4.3 Chunker — `shared/chunking.py`

Pure functions, no I/O, no model calls:

```python
def sentences(text: str) -> list[str]          # pysbd, regex fallback per paragraph
def windows(paragraphs: list[Paragraph], target=1200, hard_max=1800) -> list[Window]
def chunk_document(doc: Document, cfg) -> list[Chunk]
```

Within one section, sentences are packed into windows:

- close a window when its length reaches `target` (1,200 chars ≈ 300
  tokens); never exceed `hard_max` (1,800) — a single sentence longer than
  `hard_max` becomes its own window rather than being cut;
- prefer to close at a paragraph boundary once the window is ≥ 70% of
  `target`;
- **overlap of one sentence**: the last sentence of window *n* opens window
  *n+1*;
- never cross a section boundary.

`pysbd` is the sentence splitter (it handles `et al.`, `Fig. 3`,
`Phys. Rev. B`); on any exception for a paragraph, a regex splitter with
the same abbreviation guard is used for that paragraph and a debug line is
logged.

Expected result on the current corpus: ~10k chunks instead of 36.6k, median
~1,100 chars; six of them fill ~1,800 tokens of the 4k window.

### 4.4 Embedded text vs stored text

Stored (Chroma `documents`, what is quoted, cited and verified) is the window
verbatim. Embedded is:

```
search_document: Title: {doc.title}. Section: {section.heading}. {window}
```

The `search_document:` prefix comes from §3.6; the title/section header is
the contextual-chunk-header technique — a chunk that says "our approach
outperforms the baseline" gets something to be about. Queries are embedded
as `search_query: {query}`. `EMBED_MAX_CHARS` applies to the embedded string.

### 4.5 Metadata

Every chunk: `document, citation_source, type (text|caption|figure_description),
section, section_raw, page_first, page_last, seq, extraction`. Captions and
descriptions add `figure_id, figure_kind (figure|table), figure_label,
image_path, described (bool)`.

`seq` is the chunk's position within its document. Chroma ids remain
`chunk_N`, contiguous, never deleted — the BM25 alignment invariant from
`ARCHITECTURE.md §4.4` is unchanged.

### 4.6 Figures and tables

Detectron2 and `find_caption()` are **not consulted by v2**. They remain in
the codebase for v1 and are untouched.

**Always, for every figure/table GROBID reports:**

- a caption chunk: `type=caption`, text `"{label}: {caption}"`
  (`"Figure 2: Density of states of the first site …"`), section = the
  section the figure sits in, `figure_id` = GROBID's `xml:id`;
- a crop: PyMuPDF `page.get_pixmap(clip=bbox, dpi=PDF_RENDER_DPI)` on the
  `<graphic coords>` box (falling back to the union of the figure's `coords`
  when there is no `<graphic>`), written to
  `IMAGES_DIR/<doc>_p<page>_f<i>.png` — same naming as today — and recorded
  as `image_path`. Milliseconds per figure; no torch, cv2 or detectron2.

**When figure analysis is on for the run:**

- the switch is made **once, at the start of a run**: a checkbox on Tab 1
  beside *Process and Index Paper(s)* and *Build corpus* — *"Analyse figures
  and tables with the model (adds roughly a minute per figure)"* — the CLI
  flag `--describe-figures` on Agent 3 / Agent 6 / `orchestrate.py`, and
  `CITATION_FIGURE_VLM` as the checkbox's default state. There is no
  per-image choice anywhere;
- for every figure and table in every PDF of that run, inline, before the
  run reports done: `describe_figure(image_path, kind, context)` where
  **context is the GROBID caption plus the body sentences that reference the
  figure** (`Figure.refs`, e.g. *"Δσ_ph of MoS₂-BDT networks increases by
  ≈10% compared to pristine films (Figure 1b)"*). Prompt `FIGURE_DESCRIPTION`
  is revised to ask for what the figure shows qualitatively, to quote a
  number only when it is legible on an axis or in a cell, and to say when it
  cannot read one. Temperature 0;
- the result is written as its own chunk, `type=figure_description`,
  `extraction="vlm"`, `figure_id` linking it to the caption chunk, stored
  text opening with *"Auto-generated description of {label} — verify values
  against the figure: "*, embedded with the same title/section header;
- the caption chunk's `described` flips to true. Progress is reported per
  figure in the UI and the log.

**When it is off:** caption chunk and crop only; `described=false`. A later
run with the switch on describes whatever is still `described=false` in the
PDFs it is given — resumability after an interrupted run, not selection.

**Trust.** Description chunks are retrievable for synthesis and chat. **Agent
8 excludes `extraction="vlm"` chunks as evidence** (`doc_filter` plus a
metadata `where` on `extraction != "vlm"`): a citation verdict rests on what
the paper says, not on what a 2B model read off a plot. In the spike the
model placed a plot's peaks at ±0.5 where the axis shows ±1.0.

**Cost, stated:** ~33 s per figure on the development CPU, so a 12-page paper
with 10 figures adds ~5 minutes to its ingest; the corpus's ~2,600 real
figures/tables would be a day of CPU. The operator has accepted this in
exchange for the descriptions being correct-context and actually present.

### 4.7 Summaries

Unchanged shape (one `DOCUMENT_SUMMARY` call per paper into the summary
collection), but the input is `abstract + introduction + conclusion` sections
(in that order, up to `SUMMARY_MAX_CHARS`) instead of the first 8,000
characters of raw page text, which is frequently the title page,
affiliations and half an introduction.

### 4.8 Versioning and migration

`CITATION_INDEX_VERSION` (default `1`). `config.py` derives from it:

| | v1 (today) | v2 |
|---|---|---|
| chunk collection | `physics_papers` | `physics_papers_v2` |
| summary collection | `physics_summaries` | `physics_summaries_v2` |
| BM25 pickle | `bm25_index.pkl` | `bm25_index_v2.pkl` |
| manifest | `ingested.json` | `ingested_v2.json` |
| extraction | PyMuPDF (+ optional Detectron2) | GROBID → PyMuPDF fallback |
| chunker | `SemanticChunker` per page | §4.3 |
| nomic prefixes | off | on |

`process_pdf()` and `upsert_corpus()` dispatch on the version; Agent 3,
Agent 6, the batch uploader and `orchestrate.py` inherit it. A v2 re-ingest
of the existing corpus is
`CITATION_INDEX_VERSION=2 python -m research_assistant.agents.agent3_ingestor`
over the PDFs already in `pulled_pdfs/` and `raw/` — nothing is
re-downloaded. v1 files are never opened for writing by v2 code. `data/` is
never wiped.

---

## 5. Depth (phase 4) — the synthesis path

`retrieve.research_answer()` gains `CITATION_SYNTHESIS_MODE=map_reduce|single`
(default `single` until the harness and the operator have seen both).

**map_reduce:**

1. Stage 1 as today: `rank_documents()` → batched gate (§3.2) → shortlist.
2. **Map** — per shortlisted paper: `deep_search(query, [doc], top_k=4)`
   with neighbour expansion, then one call with prompt `PAPER_NOTES`:
   *"In at most 120 words: what this paper establishes that bears on the idea
   — method, key result, stated limitation. Refer to it as [key]. Use only the
   passages given."* Output stored as `notes[doc]`.
3. **Reduce** — one call with prompt `RELATED_WORK_SYNTHESIS` over all notes:
   themes; where papers agree and where they disagree; what is established;
   **what is missing — the gap the idea could fill**. 300–450 words, every
   claim carrying a `[key]`. Only the notes are in context, so the reduce step
   fits the 4k window with `DOC_SELECT_K` up to 10.
4. Returned dict keeps today's shape (`suggestion, citations, passages,
   selected`) and adds `notes` so the UI can show them under each selected
   paper.

**single** is the current behaviour with §3 applied (larger chunks,
neighbours, reranking).

**Cost, stated:** `1 + N + 1` model calls instead of `6 + 1`. On the VM's CPU
a shortlist of 6 is plausibly 1–2 minutes for Tab 1. The chat, cite and verify
paths are unchanged in call count and get their depth from §3/§4 alone.

Agent 8's `JUDGEMENT_TOP_K=1` is unchanged; one chunk is now ~1,100
characters plus neighbours rather than ~240.

---

## 6. Configuration

New in `config.py`, all env-overridable. **Phase-2 switches default on**:
they are safe on the v1 index, each degrades to today's behaviour when its
dependency is absent, and they are the point of phase 2 — the harness
confirms each before phase 2 ships. **Phase-3 and phase-4 switches default
off**: nothing re-ingests, re-embeds or changes the synthesis until set.

| variable | default | phase |
|---|---|---|
| `CITATION_INDEX_VERSION` | `1` | 3 |
| `CITATION_RERANK` | `1` | 2 |
| `CITATION_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | 2 |
| `CITATION_RERANK_CANDIDATES` | `20` | 2 |
| `CITATION_NEIGHBOUR_WINDOW` | `1` | 2 |
| `CITATION_GATE_BATCHED` | `1` | 2 |
| `CITATION_CHUNK_TARGET_CHARS` | `1200` | 3 |
| `CITATION_CHUNK_MAX_CHARS` | `1800` | 3 |
| `CITATION_SYNTHESIS_MODE` | `single` | 4 |
| `CITATION_DOC_SELECT_K` | `6` (existing) | — |
| `CITATION_FIGURE_VLM` | `0` (existing; now the checkbox default) | 3 |

New dependencies in `requirements.txt`: `fastembed`, `pysbd`,
`snowballstemmer`. `langchain_experimental` (SemanticChunker) stays for v1.

---

## 7. Data flow

```
PDF ─► extract() ─► Document ─► filters ─► windows ─► chunks ─┬─► Chroma (physics_papers_v2)
       GROBID/PyMuPDF                                        ├─► BM25 (bm25_index_v2.pkl)
                       └─ figures ─► caption chunk + crop ───┤
                                     └─(switch on)─► VLM ─► figure_description chunk
                       └─ abstract+intro+conclusion ─► summary ─► physics_summaries_v2

query ─► search_query: ─► summary kNN ─► batched gate ─► hybrid (tokenize v2) ─► RRF ─► rerank top-20 ─► top_k
                                                                                        └─► expand ±1 ─► context
       (map_reduce) per paper: 4 chunks + neighbours ─► notes ─► reduce ─► synthesis + gap
```

---

## 8. Error handling

| failure | behaviour |
|---|---|
| GROBID unreachable / timeout / empty body | PyMuPDF path for that PDF; `extraction="pymupdf"`; counted in the run report |
| `pysbd` raises on a paragraph | regex splitter for that paragraph; debug log |
| reranker model missing, `fastembed` not installed, ONNX error | RRF order returned; one warning per process |
| batched gate misaligned | per-summary gate calls (existing) |
| VLM call fails for a figure | caption chunk stays `described=false`; error logged; run continues |
| BM25 pickle tokenizer mismatch | warning on load; results still returned |
| v2 collection absent | same `RuntimeError` path as today ("ingest papers first"), naming the v2 collection |
| `map_reduce` map call fails for one paper | that paper's notes are its summary; the reduce step proceeds |

---

## 9. Testing

Unit tests under `tests/`, none requiring Chroma, Ollama, GROBID, the
reranker model, `fastembed` or `pysbd` at import time (the CI import sweep
runs without them):

- `test_chunking.py` — window sizes, hard max, one-sentence overlap, paragraph-
  boundary preference, no window crosses a section, short-section merge,
  alphabetic-ratio filter, caption chunks exempt from the length floor, a
  long single sentence becomes its own window.
- `test_extract.py` — TEI parser on a fixture `.tei.xml` (sections, kinds,
  paragraph pages from `coords`, figures with `graphic` box and refs,
  bibliography count); PyMuPDF fallback heuristics on synthetic block lists
  (running-header removal, references cut).
- `test_tokenize.py` — NFKC (`ﬁeld` → `field`), stemming, stopwords, query and
  document tokenised identically.
- `test_rerank.py` — wrapper with a stub scorer: reorders, keeps `rrf_score`,
  respects `top_k`, no-op when disabled or when the scorer raises.
- `test_search.py` (extend) — neighbour expansion stays within a document;
  v1 (index) and v2 (`seq`) paths.
- `test_retrieve.py` — batched gate parsing incl. every misalignment case
  falling back; `map_reduce` assembly with a stub `chat`; failed map call
  substitutes the summary.
- `test_db.py` — cache hit, invalidation on mtime change.
- `test_eval.py` — passage-hit and MRR arithmetic on hand-built results.

Then the harness, before and after each phase, with the numbers recorded in
the plan's completion notes. The repository's two-cycle protocol
(`.agents/rules/test_and_retest.md`) applies to each phase's deployment when
the operator chooses to deploy; the local cycle is: unit tests → harness →
the app against the local corpus, all five tabs.

---

## 10. Risks

- **VM memory.** Reranker adds ~450 MB beside GROBID (3 GB heap) and Ollama;
  fits in 12 GB, to be confirmed on the standalone image before flipping
  `CITATION_RERANK` there.
- **Re-ingest time on the VM.** ~10 s GROBID + embedding per paper, ≈ 1–1.5 h
  for 323 papers without figure analysis; with it, a day. Must not overlap a
  demo (`GEMINI.md`).
- **GROBID variance.** Section detection is good on journal PDFs and weaker on
  preprints with unusual layouts; unnumbered heads still yield clean
  paragraphs, so the floor is "one section of kind other", not garbage.
- **Synthetic bias.** Questions come from paragraphs GROBID parsed cleanly,
  which favours v2. The golden set and the passage-hit metric on v1's own
  chunks are the checks; the spec does not claim v2 wins until the table says
  so.
- **VLM hallucination.** Mitigated by provenance marking, the revised prompt,
  and Agent 8's exclusion — not eliminated. Descriptions can still mislead
  chat and synthesis; the marker is in the text the model sees.
- **`chunk_N` invariant.** Unchanged, still load-bearing, still only partly
  enforced. v2 adds `seq` but does not remove the dependence.
