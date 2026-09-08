# Decisions and open items

*Recorded 2026-09-07 at the end of the merge; updated 2026-09-08 after the
PDF-upload feature.*

The [spec](specs/2026-09-07-merged-research-assistant-design.md) describes the design as
it was drafted. Reading the code closely changed several parts of it. This file records
what changed and why, so the spec's stale claims don't mislead, and lists what was
knowingly left undone.

## Where the code departs from the spec

| # | Decision | Why |
|---|---|---|
| 1 | `Reference.source_file`, not `source_document`; plus `container` and `xml_id` | The spec invented a field name. The real TEI parser emits `source_file`, and Agent 2 reads it. `container` and `xml_id` existed too and the spec's schema would have dropped them. |
| 2 | `DownloadedPaper` carries `doi_source`, `authoritative`, `cited_by`, `xml_id` | The fetcher already wrote all four. Since `from_dict` ignores unknown keys by design, the spec's narrower schema would have silently discarded them on every round trip — `cited_by` being the link from a downloaded paper back to the paper that cited it. |
| 3 | `provider` and `fetched_at` are new fields the fetcher had to be taught to record | Nothing tracked *which* of Unpaywall / Europe PMC / arXiv served a PDF — `saved` was a bare boolean reused across all three branches, and `doi_source` answers a different question (how the DOI was resolved). |
| 4 | `run_extractor()` returns `reference_count`, not `references` | The JSON payload already uses `references` for a *list*. Had the plan's literal `result.get("references")` shipped, it would have returned `None` on every call, the orchestrator would have believed extraction always fails, and "GROBID down" would have become "the pipeline always stops" — silently. |
| 5 | `extracted_citations.json` keeps its `articles` key | The plan's payload dropped it. Nothing reads it today, which is exactly why it was easy to miss. |
| 6 | Agent 1 batches GROBID once up front, then reads cached TEI per PDF | The plan implied a per-PDF upload, which would have lost `GROBID_BATCH_CONCURRENCY`. |
| 7 | `_unique_destination` and the `processed/`/`failed/` filing are ported from `citation_builder` | They don't exist in the other source repo, and the plan called them without saying where they came from. |
| 8 | `paper_filename` deleted; filename identity is `filename_for(source_key(ref))` | `paper_filename` was ported to satisfy a stale interface line and was reachable only from its own tests — eleven green tests giving false confidence while the function that really names files had none. |
| 9 | `schemas.py` rejects a bare string for `authors`, and non-`str` for `str | None` fields | `"Smith, J."` was becoming a 9-tuple of characters (and therefore a different `source_key`), and an operator-precedence slip put a tuple into `doi_source` with nothing to catch it. |
| 10 | Logs live under `DATA_DIR`, not the repo root | Spec §4.2 required it; the code didn't. `CITATION_DATA_DIR` couldn't redirect logs, and the unguarded `makedirs` would break a read-only host. |

## Added after the merge: seeding from an uploaded PDF

| # | Decision | Why |
|---|---|---|
| 11 | Uploads join Agent 0's existing path rather than paralleling it | `discover_from_file()` derives the same `source_key` (`arxiv:` → `doi:` → `file:<sha1>`) and writes into `RAW_DIR`, so nothing downstream knows an upload happened. |
| 12 | An *inferred* seed query is disambiguated by the paper's key; a *supplied* one is not | `seed_papers.json` is query-keyed, which is an identity only while the caller types it. Inferred from a title or filename, two papers can collide — and the second silently overwrote the first's record. Supplied queries keep their slot because `discover()` and `discover_from_url()` depend on that for resume. |
| 13 | `ingest_pdfs()` reports `{"extraction": {"layout": n, "text_only": n}}` | Layout detection failing degrades the whole corpus to text-only with only per-PDF warnings. Agent 1 already records its GROBID-vs-regex fallback for exactly this reason; ingestion now matches it. |
| 14 | A failed Detectron2 load is cached and re-raised, not retried | `process_pdf()` swallows the failure into a text-only fallback, so nothing stopped the next document retrying. A broken config made every paper in a batch pay for a build guaranteed to fail — measured at 8 attempts across 4 calls where 2 was correct. |
| 15 | Uploads are staged to a temp file, deleted after the run | Agent 0 copies the paper into `RAW_DIR` under its own key, so the staging copy is pure duplication. It previously lived in `DATA_DIR/uploads` under the user's filename, grew without bound, and let two uploads sharing a name overwrite each other. The intermediate cannot be removed entirely — the graph's state carries a path, not bytes. |

## The recurring bug class

Six instances of **silent data loss at a boundary** were found and fixed: a field
dropped because a schema didn't declare it (#2), a payload key deleted in a rewrite (#5),
a manifest key meaning two different things across modules, an outcome recorded as
success when the model had declined, `xml_id` reading `None` forever because
BeautifulSoup stores `xml:id` under a literal key rather than Clark notation, and — after
the merge — a seed record overwritten because an inferred query was treated as an
identity (#12).

Two more were caught as *tests that couldn't fail*: one asserted on a hand-built object
while claiming to guard a pipeline function; another used a fixture the old buggy code
also handled correctly. Both were replaced with tests proven to go red under mutation.

If you change this codebase, the highest-value habit is: after writing a test, break the
code it guards and confirm the test actually fails.

## Known open items

None block use. Ordered roughly by value.

**Worth doing**
- `Reference.from_dict` has no production caller — Agent 1 constructs directly and Agent 2
  reads raw dicts, so that half of the contract is write-only and validates nothing.
- `manifest.rebuild_from_collection()` is documented and named in a user-facing warning
  but has no CLI or flag. Add `--rebuild-manifest`.
- `extracted_citations.json` is overwritten rather than merged, so a second research idea
  discards the first's `articles` and reference metadata. Every other manifest merges.
- Re-running `orchestrate.py` for a completed query re-does the seed search, download and
  GROBID pass, adding a `_2`, `_3` copy under `raw/processed/` each time. Ingestion
  correctly no-ops, so the expensive stage is safe, but the earlier stages aren't idempotent.

**Cosmetic or low-risk**
- `config.py`'s inline comment on `GROBID_SERVER` still claims a public Hugging Face Space
  default; the actual default is `http://localhost:8070`.
- The spec's §6 diagram hangs Agents 4/5/7 off two-stage retrieval; all three call
  `hybrid_search` directly. `ARCHITECTURE.md` §5.4 documents and justifies this — the
  diagram is what's stale.
- `UNPAYWALL_EMAIL` falls back to a placeholder address sent in the `User-Agent` on every
  Crossref / Unpaywall / arXiv / Semantic Scholar request. Documented, but failing loudly
  would be better than sending an invented address.
- `agent1_extractor.py`'s `force=True` disables the GROBID client's own TEI cache.
- Agent 6's watchdog behaviour (debounce, file-move handling) has no test coverage in
  either source repo or here.
- `grobid_alive()`'s success path is untested; only the unreachable branch is covered.

## Environment note

The project declares `>=3.10,<3.13`. The upper bound is real: `lxml==4.9.4` has no cp313
wheel and won't compile against 3.13's C API. If the repo's `.venv` is ever rebuilt, use a
3.10–3.12 interpreter.
