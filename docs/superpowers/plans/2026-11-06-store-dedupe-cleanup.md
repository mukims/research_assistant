# Store Duplicate Cleanup Implementation Plan

> **For agentic workers (Antigravity):** This file is your complete brief. Read all of it before
> touching code. If you have the superpowers skills, use `superpowers:executing-plans`; otherwise follow
> the tasks in order exactly as written. Track progress in a task artifact mirroring the `- [ ]` steps.
> The reviewer (Claude) checks each task; the owner gives the go for anything that touches real data.

**Goal:** Remove the duplicate papers already in LanceDB (14 PDFs stored twice, 41 abstract-only
records stored more than once, 1 test record), and stop new abstract-only duplicates from being ingested.

**Architecture:** One new module, `shared/store_dedupe.py`, holds the duplicate rule (same folded title;
two PDFs must also share a first-author surname), the keeper rule, and the apply step. Agent 3's
abstract-only ingest uses the same title rule to skip a record whose paper is already in the store.
A new script `scripts/dedupe_store.py` runs a dry run by default and writes a plan file for the owner to
review. With `--apply` it deletes the extra copies' rows from the papers and summaries tables, removes
them from the ingest manifest, retires them in the paper index, and moves their PDFs to quarantine.

**Tech Stack:** Python 3.12, LanceDB through the existing `LanceStoreClient` adapter
(`collection.get/delete`, `collection.table.version/restore`), `unittest` tests under pytest.
No new dependencies.

**Spec:** the "Approved Design" section below, plus the reviewer's read-only check of the live store on
2026-11-06 (numbers in Task 5). Where a task and the design disagree, the design wins: stop and log it.

---

## Approved Design

**Why.** The two-layer guard (plan `2026-11-06-two-layer-dedupe.md`, merged to `main` at `5ab7994`)
stops *new* PDF duplicates. The store still holds the duplicates made before it landed. When a search
returns the same paper under two keys, a citation audit or research answer can list one source twice.
Abstract-only records (`doi:…`, `raw:…`, `title:…` keys, one chunk each, written by Agent 3 when no PDF
could be fetched) bypass both guard layers, so new duplicates of that kind keep appearing.

**Owner's decision (2026-11-06):** clean up the existing duplicates and close the abstract-only gap.
This lifts the earlier "never delete from LanceDB" rule **for this cleanup only**, and only through
`scripts/dedupe_store.py --apply` after the owner approves the dry-run plan.

**What counts as a duplicate.** Two store documents are the same paper when their folded titles
(`source_key.fold_title`) are equal and the folded title is at least 15 characters long. Two **PDFs**
must also share a first-author surname key (`paper_index.surname_key`), which is the layer-2 rule.
A DOI match alone is never enough: the index has two different papers ("Lipschitz GANs" and
"Deep Learning") sharing one wrong DOI. Documents with no usable title are left alone.

**Where titles come from.** A PDF's title and surnames come from its `paper_index.json` entry. An
abstract-only record's title is the `citation_source` of its row in the store (Agent 3 writes the
record's title there, falling back to the raw reference).

**Which copy is kept (planner's call; the owner can override per group with `--keep` / `--skip`):**
- A group with any PDF keeps the PDF with the most chunks. Ties go to the PDF whose index entry has a
  DOI, then to the smallest document key. Every abstract-only record in that group is removed.
- A group of abstract-only records keeps the `doi:` key. If there is none, it keeps the record whose
  `downloaded.json` entry has a DOI, then the smallest key.
- Most chunks wins because it means more evidence text: an arXiv version with appendices beats a
  shorter published version for citation checking. The removed copy's DOIs and arXiv IDs are merged
  into the keeper's index entry, so layer 1 still recognises both versions.

**What removing a copy does (apply only):**
1. Deletes its rows from the papers table (`config.COLLECTION_NAME`) and the summaries table
   (`config.SUMMARY_COLLECTION_NAME`).
2. Removes its key from `data/ingested_v3.json`.
3. For a PDF: merges its DOIs and arXiv IDs into the keeper, deletes its paper-index entry and records
   an alias `removed → keeper`. It also moves the file to `data/quarantine/store_dedupe/`. Moving is
   not deleting, and `build_paper_index._find_pdfs` already skips `quarantine/`.
4. `downloaded.json` is not edited. A removed PDF's record now points at a missing file, which
   `agent3_ingestor._pdfs_from_manifest` already skips. A removed abstract-only record is skipped by the
   new Agent 3 check (Task 2), because its title matches the keeper.

**Prevention (abstract-only).** Before Agent 3 ingests an abstract-only record, it skips the record if
its folded title (≥ 15 chars) matches a document already in the store, or one accepted earlier in the
same run. Skips are logged. Nothing is deleted at ingest time. A stub that arrives *before* the PDF of the
same paper still ends up next to it. The end-of-run sweep doesn't report that case; rerunning
`scripts/dedupe_store.py` removes it. That stays a manual step on purpose: the live pipeline never
deletes from the store.

**Out of scope:** fuzzy title matching; records with no title; images or Docling cache files belonging
to removed PDFs (harmless leftovers); the `batch_citer` ordering flake; `scripts/rebuild_corpus_docling.py`'s
own abstract-only path.

## Global Constraints

- Interpreter `$PY` = `/home/shardul/miniconda3/envs/ml/bin/python`. Test command, from
  `migration-source/research_assistant/`: `CITATION_LOG_FILE=0 $PY -m pytest <targets> -q -p no:cacheprovider`.
- Baseline on `main` at `5ab7994`: `2309 passed, 18 skipped, 1 failed`. The failure is
  `tests/test_batch_citer.py::TestRunBatchCiterPassesContextAndQuery::test_context_and_paragraph_always_reach_the_seam`,
  a known ordering race that predates this work. Any other failure is yours.
- Branch `fix/store-dedupe-cleanup` from `main`. One commit per task, `feat(dedupe): <what>`.
  **Do not push. Do not merge.**
- Tests never touch the real `data/` directory (conftest already redirects `DATA_DIR`; patch every
  other path you use).
- Rows may be deleted from LanceDB **only** in Task 5 Step 6, after the owner's explicit go. Never delete
  a file; PDFs are moved to quarantine.
- Do not run the app, `fetch_papers` or `ingest_pdfs` against real data.
- Touch only the files each task lists. If reality differs from the plan, stop, log it in the Handoff Log
  and ask.

## Review Focus

1. A group whose PDFs share a title but no surname (two different papers with a generic title, for
   example "A Survey of Large Language Models"): not a duplicate, nothing removed (Task 1).
2. Two different papers sharing one wrong DOI and different titles: not grouped (Task 1).
3. A stub whose title is shorter than 15 folded chars, or missing: never grouped and never skipped by
   Agent 3 (Tasks 1, 2).
4. `--apply` while an ingest holds the ingest lock: it waits for the same lock, never runs alongside
   (Task 4).
5. A delete that raises partway through: the manifest and paper index are not touched for that
   document, the error is reported, and the run exits non-zero (Task 4).

---

## File Map

| File | Responsibility |
|---|---|
| `research_assistant/shared/store_dedupe.py` (new) | `title_key`, `StoreDoc`, `scan_store`, `plan_cleanup`, `ingested_titles`, `apply_cleanup` |
| `research_assistant/shared/manifest.py` | `discard_many` |
| `research_assistant/shared/paper_index.py` | `PaperIndex.retire` |
| `research_assistant/agents/agent3_ingestor.py` | Skip abstract-only records already in the store |
| `scripts/dedupe_store.py` (new) | CLI: dry run → `data/store_dedupe_plan.json`; `--apply` |
| `PIPELINE.md` | One paragraph on the abstract-only check and the cleanup script |
| `tests/test_store_dedupe.py` (new) | Tasks 1, 3, 4 |
| `tests/test_agent3_abstract_dedupe.py` (new) | Task 2 |

---

### Task 1: Duplicate rule, keeper rule, and the two small helpers

**Files:** create `research_assistant/shared/store_dedupe.py`, `tests/test_store_dedupe.py`; modify
`research_assistant/shared/manifest.py`, `research_assistant/shared/paper_index.py`, `tests/test_paper_index.py`.

**Interfaces (produces):**
```python
TITLE_MIN = 15
def title_key(title: str | None) -> str | None          # fold_title(title) if len >= TITLE_MIN else None

@dataclass(frozen=True)
class StoreDoc:
    document: str
    kind: Literal["pdf", "stub"]       # "pdf" iff document.endswith(".pdf")
    title: str | None
    surnames: frozenset[str]           # surname_key()s of the index entry's first_authors; empty for stubs
    chunks: int
    has_doi: bool                      # pdf: index entry has dois; stub: key starts "doi:" or downloaded record has "doi"

@dataclass(frozen=True)
class Group:
    title_key: str
    keep: str
    remove: tuple[str, ...]
    reason: str                        # "pdf+pdf" / "pdf+stub" / "stub+stub" etc: sorted kinds joined by "+"

def plan_cleanup(docs: Iterable[StoreDoc], keep: Iterable[str] = (), skip: Iterable[str] = (),
                 drop: Iterable[str] = ()) -> list[Group]
# manifest.py
def discard_many(keys: Iterable[str]) -> int             # one read-modify-write; returns number removed
# paper_index.py
def retire(self, document: str, keeper: str) -> None     # union document's dois/arxiv_ids into keeper's
                                                         # entry (if keeper is indexed), delete document's
                                                         # entry, add_alias(document, keeper); no-op if absent.
                                                         # keeper == "" (a --drop): delete the entry, no alias
```

`plan_cleanup` grouping: bucket by `title_key(title)` and drop `None`. Inside a bucket, link two docs
unless both are PDFs with disjoint `surnames`. Take connected components of size ≥ 2 and choose the
keeper by the design's rule. `keep` forces that document as its group's keeper; `skip` drops the whole
group containing that document. `drop` documents become single-member groups with `keep=""`,
`remove=(doc,)`, `reason="drop"`. Groups are returned sorted by `title_key`.

- [ ] **Step 1: Write the failing tests** in `tests/test_store_dedupe.py` (class `TestPlanCleanup`):
  - `test_two_pdfs_same_title_shared_surname_keeps_more_chunks`: pdf A (59 chunks, {"zhang"}) and
    pdf B (40, {"zhang"}) with the same title → one group, `keep == A`, `remove == (B,)`, `reason == "pdf+pdf"`.
  - `test_two_pdfs_same_title_no_shared_surname_not_grouped` → `[]`.
  - `test_chunk_tie_prefers_doi_then_smallest_key`: equal chunks, only B has a DOI → keep B. If both
    have DOIs → keep the lexicographically smaller key.
  - `test_pdf_and_stubs_keep_pdf`: pdf (67) + `doi:x` + `raw:y`, all with the same title →
    `keep == pdf`, `remove == ("doi:x", "raw:y")` (sorted), `reason == "pdf+stub+stub"`.
  - `test_stubs_keep_doi_key`: `raw:a` (has_doi) + `doi:b` → keep `doi:b`. `raw:a` (has_doi) +
    `raw:b` → keep `raw:a`.
  - `test_short_or_missing_title_never_grouped`: two stubs titled "Introduction" → `[]`, and two with
    `title=None` → `[]`.
  - `test_same_doi_different_titles_not_grouped`: two PDFs, different titles, shared surname → `[]`.
  - `test_keep_override` and `test_skip_override` behave as specified.
  - `test_drop_makes_single_group`: `drop=["doi:10.1000/999"]` → a group with `keep == ""` and `reason == "drop"`.
  In `tests/test_paper_index.py`: `test_retire_merges_ids_and_aliases` (the keeper gains the retired
  entry's DOI and arXiv ID, `get(retired) is None`, `resolve(retired).document == keeper`) and
  `test_retire_missing_is_noop`.
  In `tests/test_store_dedupe.py` (class `TestManifestDiscard`): `discard_many(["a", "zz"])` on
  `{"a", "b"}` returns 1 and leaves `{"b"}`; on no match it returns 0 and does not rewrite the file
  (assert the mtime is unchanged).
- [ ] **Step 2:** Run the tests and check they fail on missing names.
- [ ] **Step 3:** Implement the interfaces above.
- [ ] **Step 4:** Run the tests and check they pass. Then run the full suite: baseline plus your tests.
- [ ] **Step 5:** Commit: `feat(dedupe): store duplicate rule and keeper choice`.

### Task 2: Agent 3 skips abstract-only records already in the store

**Files:** modify `research_assistant/shared/store_dedupe.py` and
`research_assistant/agents/agent3_ingestor.py` (the abstract-only loop, currently around L114–152);
create `tests/test_agent3_abstract_dedupe.py`.

**Interfaces (produces):**
```python
def ingested_titles(collection, index: PaperIndex, ingested: Iterable[str]) -> dict[str, str]
# title_key -> document, for every document in `ingested`. PDFs: index title. Stubs: citation_source of
# collection.get(where={"type": "abstract_only"}, include=["metadatas"]) rows (paged, limit 5000).
```
In Agent 3, build the map once before the loop. Use `open_store().get_collection(name=COLLECTION_NAME)`,
`PaperIndex.load(PAPER_INDEX_PATH)` and `manifest.load()`. If the store can't be opened, log a warning
and use an empty map: the check is skipped, and ingest must never fail because of it. For each record
that passes the existing `manifest.contains` check, compute `tk = title_key(record.get("title"))`.
If `tk` is in the map, skip the record and log `"Skipping abstract-only %s: same paper as %s"`.
Otherwise ingest it and add `map[tk] = doc_key`.

- [ ] **Step 1: Write the failing tests** (patch the store and paper index with fakes; `downloaded.json`
  in a temp dir):
  - `test_stub_matching_ingested_pdf_title_is_skipped`: the index has PDF `p.pdf` titled T and in the
    manifest. A record `doi:x` titled T → `upsert_corpus` is not called with any `doi:x` entry.
  - `test_stub_matching_ingested_stub_is_skipped`: the store has an abstract-only row
    `doi:x`/`citation_source` T. A record `raw:y` titled T is skipped.
  - `test_two_new_stubs_same_title_only_first_ingested`: both new, same title → only the first key
    is upserted and marked.
  - `test_stub_with_short_title_is_ingested`: a record titled "Results" is ingested as today.
  - `test_store_unavailable_ingests_as_before`: the store raises `StoreUnavailable` → the record is
    ingested.
- [ ] **Step 2:** Run the tests and check they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests and check they pass. Then run the full suite.
- [ ] **Step 5:** Add one paragraph to `PIPELINE.md` next to the duplicate-guard paragraph, covering
  the abstract-only check and `scripts/dedupe_store.py`. Commit: `feat(dedupe): skip abstract-only records already in the store`.

### Task 3: Read the store and write the dry-run plan

**Files:** modify `research_assistant/shared/store_dedupe.py`; create `scripts/dedupe_store.py`; extend
`tests/test_store_dedupe.py`.

**Interfaces (produces):**
```python
def scan_store(collection, index: PaperIndex, downloaded: dict) -> list[StoreDoc]
# one paged pass over collection.get(include=["metadatas"], limit=5000, offset=…): count chunks per
# document, take citation_source of the first row seen. Stub has_doi uses downloaded records keyed by
# record.get("key") or their dict key.
```
`scripts/dedupe_store.py` (sys.path fix as in `scripts/build_paper_index.py`):
- Options: `--apply`, `--keep DOC` (repeatable), `--skip DOC` (repeatable), `--drop DOC` (repeatable),
  `--plan PATH` (default `data/store_dedupe_plan.json`).
- **Dry run (default):** scan the store, call `plan_cleanup`, and write the plan file with
  `atomic_write_json`: `{"created": iso, "groups": [asdict(g)...], "totals": {"groups", "remove_docs",
  "remove_pdf_docs", "remove_stub_docs", "remove_chunks"}, "args": {...}}`. Print the totals and one line
  per group (`reason  keep  ←  removed`). Exit 0. Writes nothing else.

- [ ] **Step 1: Write the failing tests:** `test_scan_store_counts_chunks_and_titles` uses a fake
  collection whose `get` pages over a list of metadata dicts. `test_dry_run_writes_plan_only` runs the
  script's `main(argv)` against fakes and asserts the plan file exists, has the expected totals, and that
  no fake `delete` was called and the manifest is unchanged.
- [ ] **Step 2:** Check the tests fail. **Step 3:** Implement. **Step 4:** Check the tests pass, then run the full suite.
- [ ] **Step 5:** Commit: `feat(dedupe): dry-run store cleanup plan`.

### Task 4: Apply

**Files:** modify `research_assistant/shared/store_dedupe.py` and `scripts/dedupe_store.py`; extend
`tests/test_store_dedupe.py`.

**Interfaces (produces):**
```python
def apply_cleanup(groups: list[Group], papers, summaries, index_path: str, quarantine_dir: str) -> dict
# returns {"versions_before": {"papers": int, "summaries": int}, "removed": [doc...],
#          "moved": [[src, dst]...], "errors": [{"document", "error"}...]}
```
Order per removed document:
1. `papers.delete(where={"document": doc})`, then `summaries.delete(where={"document": doc})`.
2. Only if both deletes succeeded: `manifest.discard_many([doc])`. For a PDF, inside
   `with paper_index.edit(index_path) as idx`: read the entry's `path`, then `idx.retire(doc, keep)`.
   Then, if the path exists, move the
   file into `quarantine_dir` with a unique name.
3. If a delete raises, record the error, leave the manifest and index alone for that document, and
   continue with the next one.
Record `papers.table.version` and `summaries.table.version` before the first delete.

`--apply` in the script:
- Refuse unless `--plan` exists and was written by a dry run with the same `--keep/--skip/--drop`.
  Re-scan, re-plan and compare against the file. If the groups differ, refuse with "store changed since
  the dry run — rerun the dry run".
- Hold `ingestion._ingest_lock()` for the whole apply.
- After the apply: `db.clear_search_cache()`, then `manifest.rebuild_from_collection()`. Print the
  rebuild count and compare it with the manifest count from before minus the removed documents.
  Then run `run_sweep()`.
- Write `data/store_dedupe_applied-<YYYYMMDD-HHMM>.json` holding the `apply_cleanup` result.
- Exit 1 if `errors` is non-empty.

- [ ] **Step 1: Write the failing tests** (fake collections that record `delete` calls and can be set to
  raise; temp manifest, index and quarantine dir):
  - `test_apply_deletes_both_tables_and_updates_manifest_and_index`.
  - `test_apply_moves_pdf_to_quarantine_never_deletes_file`: the file exists at the new path with the
    same bytes.
  - `test_delete_error_leaves_manifest_and_index_for_that_doc`: covers Review Focus 5.
  - `test_drop_group_removes_index_entry_without_alias`.
  - `test_apply_refuses_when_plan_differs`.
  - `test_apply_holds_ingest_lock`: patch `_ingest_lock` with a recording context manager.
- [ ] **Step 2:** Check the tests fail. **Step 3:** Implement. **Step 4:** Check the tests pass, then run the full suite.
- [ ] **Step 5:** Commit: `feat(dedupe): apply store cleanup`.
- [ ] **Step 6:** Stop. Log "Tasks 1–4 done, <sha>, suite <numbers>" and wait for the reviewer.

### Task 5: Real run (owner's go required at two points)

Expected numbers come from the reviewer's read-only check of the live store on 2026-11-06. It holds
1,781 documents and 125,148 chunks, and finds 49 groups (14 pdf+pdf, 4 pdf+stub, 1 pdf+stub+stub,
26 stub+stub, 3 stub+stub+stub, 1 stub+stub+stub+stub).

- [ ] **Step 1:** Wait for the reviewer to accept Tasks 1–4 and the owner to say go. The owner stops the
  Marvin app first; `pgrep -af "streamlit run"` must print nothing.
- [ ] **Step 2:** Back up: `cp -a data/lancedb data/lancedb.bak-2026-11-06` (1.3 GB; 36 GB free),
  `cp data/ingested_v3.json data/ingested_v3.json.bak-cleanup`, `cp data/paper_index.json data/paper_index.json.bak-cleanup`.
- [ ] **Step 3:** Dry run: `$PY scripts/dedupe_store.py --drop "doi:10.1000/999"`.
  Expected: 50 groups (49 plus the drop), 56 documents to remove (14 PDFs, 42 abstract-only), about
  1,425 chunks.
- [ ] **Step 4:** Log the totals and paste the per-group lines into the Handoff Log. **Stop.**
- [ ] **Step 5:** The reviewer checks the plan file group by group. The owner decides any keeper
  overrides; for example, the owner's own papers `p3.pdf` and `p6.pdf` are each in a pair, and under the
  rule `p3.pdf` is removed and `p6.pdf` kept. If there are overrides, rerun the dry run with
  `--keep`/`--skip` and stop again.
- [ ] **Step 6:** After the owner's explicit go, run the same command with `--apply` and identical
  `--keep/--skip/--drop`. Expected: no errors; manifest rebuild 1,725; store 1,725 documents and about
  123,723 chunks; 14 PDFs in `data/quarantine/store_dedupe/`.
- [ ] **Step 7:** Rerun the dry run. Expected: 0 groups.
- [ ] **Step 8:** Log all the numbers, the `versions_before` values and the applied-file name. Commit the
  log: `docs(dedupe): record store cleanup`. Stop and report. Do not restart the app.

**Rollback** (if the owner asks): stop the app, then
`rm -rf data/lancedb && cp -a data/lancedb.bak-2026-11-06 data/lancedb`. Restore the two `.bak-cleanup`
files and move the quarantined PDFs back using the `moved` list in the applied file. Without the copy,
`table.restore(<versions_before>)` on each table also works for at least 7 days: ingest's `optimize()`
only removes versions older than that.

---

## Review Round 1 (reviewer, 2026-11-06, against `5d40e1c`)

Tasks 1–4 mostly match the plan, and the tests are honest. Three fixes are needed before Task 5;
R8 blocks it. Do them on the same branch, test first, with one commit each
(`fix(dedupe): <what>`). Then run the full suite, log, and stop.

### R8 (blocking): `--apply` always refuses, even right after a matching dry run

`scripts/dedupe_store.py` compares `[asdict(g) for g in groups]` with the plan file's `groups`. After
the JSON round trip, `remove` is a list in the file but a tuple in memory, and `("a",) != ["a"]`.
So any plan with at least one group is always reported as "store changed since the dry run". The only
apply test that reaches success uses `"groups": []`, which hides this.
- Fix: compare like with like, e.g. `json.loads(json.dumps(current_group_dicts)) != plan_data["groups"]`.
- Test `test_dry_run_then_apply_same_store_succeeds`: on one fake store with one duplicate pair, run
  `main([])` and then `main(["--apply", ...])`. Assert the second returns 0 and the fake `delete` was
  called for the removed document in both tables.

### R9: Agent 3 with `force=True` skips an ingested stub as a duplicate of itself

With `force`, the `manifest.contains` check is bypassed. But `titles_map` already maps the stub's own
title to its own key, so the record is skipped with "same paper as <itself>" and never re-ingested.
- Fix: skip only when `titles_map[tk] != doc_key`.
- Test `test_force_reingests_stub_already_in_store`: the store has `doi:x` titled T and the manifest
  has `doi:x`. `run_ingestor(force=True)` with record `doi:x` titled T → `doi:x` is upserted.

### R10: a failure after the deletes aborts the apply and loses the rollback record

In `apply_cleanup`, only the two deletes are inside `try`. If `manifest.discard_many`, the index
`edit`, or `shutil.move` raises (a full disk, a permission error), the exception escapes. The rest of
the plan doesn't run, and the script never writes `store_dedupe_applied-*.json`. That file holds
`versions_before` and the `moved` list the rollback needs.
- Fix: wrap the post-delete steps for each document in their own `try` and record
  `{"document", "stage": "manifest" | "index" | "move", "error"}` in `errors`. In the script, print
  `versions_before` before the first delete. Write the applied file in a `finally` so it exists
  even if something unexpected escapes.
- Test `test_move_failure_is_recorded_and_apply_continues`: two PDF removals, and `shutil.move` raises
  for the first. Assert the second is still removed and moved, `errors` has one entry with
  `stage == "move"`, and the first PDF is still at its original path.

### Accepted deviation

`scan_store` falls back to the store's `citation_source` when a PDF's index entry has no title. That is
broader than the design, but PDF+PDF links still need a shared surname, so it can only add PDF+stub
groups. On the live store it adds exactly one real duplicate: `doi_10.1103_revmodphys.81.109.pdf`
(no index title) and the abstract-only `title:976524d097dc9807`, both "The electronic properties of
graphene". Keep it.

### Corrected Task 5 numbers (reviewer's in-memory run of `scan_store` + `plan_cleanup` on the live store)

- Step 3 (with `--drop "doi:10.1000/999"`): **51 groups** (14 pdf+pdf, 5 pdf+stub, 1 pdf+stub+stub,
  26 stub+stub, 3 stub+stub+stub, 1 stub+stub+stub+stub, 1 drop), **57 documents** (14 PDFs,
  43 abstract-only), **1,426 chunks**.
- Step 6: manifest rebuild **1,724**; store **1,724** documents and **123,722** chunks; 14 PDFs quarantined.
- `p3.pdf` is removed (kept: `title_b780af29ce603766.pdf`, 50 chunks vs 38); `p6.pdf` is kept.

**Reviewer verdict on R8–R10 (`91e8c30`, `770ab29`, `4d2d1cc`):** accepted. Each fix matches the
request and has the requested test.
- R8: the plan comparison now goes through a JSON round trip.
- R9: a stub is skipped only when the matching title belongs to a different document.
- R10: the manifest, index and move steps each record their own errors, versions are printed before
  the deletes, and the applied file is written in a `finally`.

Residual, accepted: if `apply_cleanup` itself raised unexpectedly, the applied file would hold only
`versions_before`, with no `moved` list. The per-step guards make that very unlikely, and the full
`data/lancedb.bak-2026-11-06` copy (Task 5 Step 2) covers it.
**Cleared for Task 5 once the owner says go.**

---

## Handoff Log (append-only)

Format: `date time · task · sha · result · discrepancies`.

2026-11-06 20:17 · Task 1 · 0103397 · tests/test_store_dedupe.py + test_paper_index.py (33 passed), full suite (2323 passed, 18 skipped) · none
2026-11-06 20:25 · Task 2 · cf6237b · tests/test_agent3_abstract_dedupe.py (5 passed), full suite (2328 passed, 18 skipped) · none
2026-11-06 20:30 · Task 3 · 1e1bb5d · tests/test_store_dedupe.py (13 passed), full suite (2329 passed, 18 skipped, 1 failed known flake in test_batch_citer.py) · none
2026-11-06 20:36 · Task 4 · 2a78353 · tests/test_store_dedupe.py (19 passed), full suite (2336 passed, 18 skipped) · none
2026-11-06 20:37 · Tasks 1–4 done · 2a78353 · full suite: 2336 passed, 18 skipped · none
2026-11-07 10:35 · R8 · 91e8c30 · tests/test_store_dedupe.py (20 passed) · none
2026-11-07 10:36 · R9 · 770ab29 · tests/test_agent3_abstract_dedupe.py (6 passed) · none
2026-11-07 10:41 · R10 · 4d2d1cc · tests/test_store_dedupe.py (21 passed), full suite (2339 passed, 18 skipped) · none
2026-11-07 11:08 · Task 5 Steps 1–4 (dry run) · 98d72ce · 51 groups, 57 remove docs (14 PDFs, 43 stubs), 1426 chunks · none
pdf+stub+stub  doi_10.1039_c7cp04186a.pdf  ←  doi:10.1039/c7cp04186a, raw:eb3ab3ce8300bf76
stub+stub  raw:45a0ba4360acff20  ←  raw:abbd524ca05ffc5d
stub+stub  doi:10.1063/1.4952647  ←  raw:cdbc9d416cfd324e
stub+stub  doi:10.1103/physrevb.44.943  ←  raw:a7624afc99390855
stub+stub  doi:10.1088/1361-648x/aa6667  ←  raw:85d5e12c84f3b54a
stub+stub  doi:10.1063/1.5008981  ←  raw:8f942bb066bdb33f
stub+stub  doi:10.1063/1.4958636  ←  raw:48f421e577fb082e
pdf+pdf  title_5d1b8104d3220524.pdf  ←  arxiv_2406.10833.pdf
pdf+pdf  title_b780af29ce603766.pdf  ←  p3.pdf
stub+stub  doi:10.1103/physrevlett.84.5172  ←  raw:59c24c0a55949cd2
stub+stub  doi:10.1103/physreva.38.3098  ←  raw:e6ef27586ada03b2
stub+stub  doi:10.1002/qua.560280846  ←  raw:7688a1a8c28771a9
stub+stub  doi:10.1103/physrevlett.49.1691  ←  raw:d5ceb456935ba561
stub+stub  doi:10.1103/physrevb.48.16929  ←  raw:7fa00e9ac4df2bd4
stub+stub  doi:10.1103/physrevb.52.r5467  ←  raw:870c69fec536b64d
stub+stub  doi:10.1063/1.464913  ←  raw:dc42d5a4cf98fa72
pdf+stub  raw_d8b73794c1cae008.pdf  ←  raw:d8b73794c1cae008
stub+stub  doi:10.1103/physrevb.37.785  ←  raw:aed6c091df7ce8c0
stub+stub+stub  doi:10.1063/1.5090445  ←  raw:7e51f38bfa66e7c7, raw:93fe2293a90b3774
pdf+pdf  p6.pdf  ←  doi_10.1088_1361-648x_adc52c.pdf
stub+stub  doi:10.1103/physrevb.57.1505  ←  raw:f5b7186b6518615a
pdf+stub  doi_10.1103_revmodphys.81.109.pdf  ←  title:976524d097dc9807
stub+stub  doi:10.1017/cbo9780511805776  ←  title:a2addbed1ba36587
stub+stub  doi:10.1103/physrevb.82.165109  ←  raw:82ac9d797f71c3bd
pdf+stub  raw_7820c1093d7afde6.pdf  ←  raw:7820c1093d7afde6
pdf+pdf  doi_10.1038_nature12952.pdf  ←  nature12952.pdf
stub+stub+stub  doi:10.1103/physrevlett.77.3865  ←  raw:5b598f8dbbbc5443, title:c0f36c3f5177b602
stub+stub  doi:10.1063/1.4959882  ←  raw:25baf2f103d70960
stub+stub  title:8cb78f24b8e4d890  ←  title:43ecb61cf2c4b336
pdf+stub  title_2bbfb8ffb5918041.pdf  ←  title:2bbfb8ffb5918041
stub+stub  doi:10.1063/1.1564060  ←  raw:fbaccd3fafb1e586
stub+stub  doi:10.1103/physrev.136.b864  ←  raw:f8da2006732551ed
stub+stub  doi:10.1103/physrevlett.68.2512  ←  title:36de9d75ff887cac
stub+stub  doi:10.1021/acs.jctc.9b00322  ←  raw:b71c6d1878fbd1b3
pdf+pdf  title_179bb67320a2268f.pdf  ←  title_960f865668cf20c7.pdf
pdf+pdf  nmat3687.pdf  ←  title_5489394486b132f6.pdf
stub+stub+stub+stub  raw:3da7b8294a88831a  ←  raw:783e2ab566f88bdb, raw:c3107662494b31b6, raw:c396c26df039eaeb
stub+stub  doi:10.1063/1.5025723  ←  raw:28c72208f05e6d58
pdf+pdf  title_d86ca14c50306fe2.pdf  ←  title_bad77363b9702ff2.pdf
stub+stub  doi:10.1063/1.477923  ←  raw:3a16d3e83751e74e
stub+stub+stub  raw:0ada165f5fd9704f  ←  raw:3f2a23a2776b7698, raw:9b39f883ea24d31e
pdf+pdf  doi_10.1038_nature13831.pdf  ←  title_bca16e6249d6aeb0.pdf
pdf+pdf  title_31c5a0663f14c5b5.pdf  ←  title_f5645863e6a98f2e.pdf
pdf+pdf  title_81c4479574c12c4a.pdf  ←  title_575bddaa45fac00a.pdf
stub+stub  doi:10.1063/1.2387954  ←  raw:912217fcffbd46f2
pdf+stub  raw_c49332877bb83d0e.pdf  ←  raw:c49332877bb83d0e
pdf+pdf  doi_10.1038_nature17151.pdf  ←  nature17151.pdf
pdf+pdf  title_0c3716fd7b2d10dd.pdf  ←  doi_10.1007_s11704-026-60308-3.pdf
pdf+pdf  doi_10.4208_cicp.110113.010813a.pdf  ←  title_1b5352b9273b5bf5.pdf
drop    ←  doi:10.1000/999
pdf+pdf  doi_10.1002_adma.202211157.pdf  ←  title_c93a05ab11221828.pdf

**Reviewer verdict on the dry run (`98d72ce`):** accepted.
- **Totals:** 51 groups, 57 documents, 1,426 chunks, matching Review Round 1. All 51 groups
  equal the reviewer's independent title grouping, plus the accepted graphene pair and the drop.
- **Consistency:** every keeper and every removed document is in the manifest. All 14 PDFs to be
  removed exist on disk: 12 in `pulled_pdfs/` and 2 in `raw/processed/`
  (`arxiv_2406.10833.pdf`, `doi_10.1007_s11704-026-60308-3.pdf`). Both are old seed papers, and the UI
  reads `seed_papers.json` for display metadata only, so moving them is safe.
- **Backup:** `data/lancedb.bak-2026-11-06` is byte-identical to `data/lancedb` (93 files,
  1,340,839,993 bytes, `diff -rq` clean).
**Task 5 Step 6 waits for the owner's go, including the `p3.pdf` keeper choice.**

2026-11-07 11:36 · Task 5 Steps 6–8 (apply) · (this commit) · versions_before={'papers': 2720, 'summaries': 102}, rebuild_count=1724 (expected 1724), applied_file=data/store_dedupe_applied-20261107-1135.json, 14 PDFs quarantined, after_dry_run=0 groups (0 docs, 0 chunks), store=1724 docs, 123722 chunks · none
