# Ingestion v2 — verification record

*2026-09-12. Branch `ingestion-v2`, head after `b303179`. Local only; nothing deployed.*

## Build

Tasks 1–15 of `2026-09-11-ingestion-v2.md` implemented by the build agent
(`d4c0e9d`…`cbfdd90`), plus two of its own fixes (`21b2617` path resolution,
`552da24` stats script) and two regression fixes from the takeover
(`0c5837a`, `b303179`). Test suite: 461 passed, 1 skipped before the fixes
with one failure (`test_watch`), now green; `tests/test_app_names.py` added.

## Task 16 gates

| Gate | Result |
|---|---|
| 16.1 baseline | v1: 36,609 chunks / 323 docs; p10 45, median 242, p90 1,398; 45% under 200 chars |
| 16.2 one paper, figures ON | `doi_10.1002_adma.202211157.pdf`: GROBID, 17 crops, 10 descriptions. Crop `p4_f2` opened: it is Figure 2 (σ vs T, ln W vs ln T with hopping exponents, R_hop bars); its description says the same. `ingested.json` untouched. |
| 16.3 full corpus, figures OFF | 303 processed, 1 skipped, 15,946 chunks inserted, 0 unreadable. GROBID 11 min (2.2 s/paper), chunking 6 s, embedding 52 min (5 chunks/s), summaries ~2.7 h (~33 s/paper), BM25 rebuild 21 s. 1 of 303 fell back to PyMuPDF (GROBID HTTP 500 on `title_eafd2d1a64e8d30a.pdf`). |
| 16.4 compare | v2: 16,006 chunks / 304 docs; p10 352, median 1,062, p90 1,498; 5% under 200 chars (captions). Types: 13,196 text, 2,800 caption, 10 description. v1 files byte-untouched (`bm25_index.pkl`, `ingested.json` mtimes 2026-09-09; `physics_papers` count 36,609). |
| 16.5 app on v2 | Tab 1: sidebar reads 16,006 chunks; toggle *Analyse figures and tables with the model* present, default off (seen in the browser). Tabs 2/3/4 exercised through the functions the tabs call, on `CITATION_INDEX_VERSION=2` — below. |

### Tab 2 — `suggest_citation`, 63 s

Sentence: *Anderson localization suppresses diffusion in one-dimensional
disordered wires.* Cited *Anderson transitions* (Rev. Mod. Phys. 80, 1355)
and the SciPost nanoribbon paper. Passages 941 / 420 / 1,120 chars, all
prose, no bibliography text.

### Tab 3 — `run_batch_citer` then `verify_draft`, 143 s + 240 s

Two-sentence draft on covalent MoS₂ networks. Agent 5 cited sentence 1 to
*Covalent MoS2 networks* (right) and sentence 2 to *Charge transport in
semiconducting carbon nanotube networks* (wrong — same retrieval-miss class
as the comparison below). Agent 8: `cite_1 → Partially supports / partial`,
`cite_2 → Does not support / insufficient`. Both evidence spans real
paragraphs (1,045 and 1,227 chars); neither a `figure_description`.

### Tab 4 — `ResearchChat.stream_turn`, 3.5 s retrieval + 83 s answer

Question on hopping transport vs linker in the MoS₂ paper. Five chunks from
that paper (7,096 chars of context). Answer names BDT vs PDT, 3D-VRH, and
how R_hop changes — specific, grounded.

## Retrieval comparison (same six queries, both indexes, this machine)

| | v1 | v2 |
|---|---|---|
| cold load | 4.5 s | 3.1 s |
| text in memory / BM25 pickle | 19.0 MB / 21.9 MB | 16.2 MB / 13.1 MB |
| hybrid search median / max | 91 / 180 ms | 84 / 128 ms |
| stage-1 rank | 121 ms | 122 ms |
| top-3 chars (median) | 1,862 | 3,239 |
| BM25 query tokens for "finite size effects on the density of states" | 8 incl. `on the of` | 5 stems |

Top-1 on four queries: v2 better on three (canonical review for Anderson
localization; a real paragraph instead of equation soup; prose instead of a
bibliography fragment), worse on one (MoS₂ hopping: intro paragraph
out-scored the hopping paragraph). Reranking (spec §3.3) is the fix for that
class; the harness (spec §2) is how it gets measured.

## Regressions found in the build agent's work

1. `app.py` — the lazy `from research_assistant.shared.ingestion import
   ingest_pdfs` was deleted while wiring the toggle into the multi-PDF/ZIP
   branch → `NameError` on first use of that branch. Not visible to the suite
   or the import sweep. Fixed in `0c5837a` with a static guard test.
2. `watch.py` — after `21b2617`, `_labeled_pdfs` looked papers up by the
   manifest's recorded path while `_pdfs_from_manifest` now keys by the local
   path, so labels fell back to filename stems. Fixed in `b303179`.

## Rulings made during takeover

- `clean_text()` was changed in `c896fd9` to preserve paragraph breaks — not
  in the plan; done to satisfy a plan test that assumed the wrong thing.
  **Accepted:** both v1 extraction paths flatten newlines per block before
  `clean_text` runs, so the change is inert on real input. Cost if wrong: none
  observed; the v1 upsert tests are green.
- The build agent's 16.5 note in `552da24` was generic and written four
  minutes after the ingest finished. **Treated as not done**; re-verified
  above.
- Tabs 2/3/4 were verified through the tab's own functions rather than the
  browser, because the browser pane was in use by a person mid-check.

## Deferred minors

- A `figure_description` chunk shares its caption's `seq`; `seq` is therefore
  not unique per document. Matters only when neighbour expansion (spec §3.4)
  lands.
- Sidebar still shows "Layout — on" (a v1 concept) under v2; it should show
  the index version and extraction mode.
- Two chunks reach `EMBED_MAX_CHARS` (4,000): a sentence the splitter could
  not break stood alone. Rare (p90 is 1,498); worth a look at which.
- Chat context per turn is ~1,800 tokens on v2 — the window fills after two
  or three turns (accepted loss #6 in the handoff).
- `tests/test_ingestion.py` cannot run while another process holds
  `data/ingest.lock`; the suite hangs rather than failing.
