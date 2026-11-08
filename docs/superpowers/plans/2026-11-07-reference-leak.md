# Reference-List Leak Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep bibliography entries that Docling neither labels as references nor puts under a recognised
"References" heading out of the text chunks, and repair the papers already stored with leaked reference text.

**Architecture:** A new `shared/bibliography.py` decides what looks like a reference entry and finds *runs*
of consecutive entries. `docling_extract.extract_docling` drops those runs from each section before
chunking, and a wider `_BIB_HEAD` catches heading variants it misses today. A dry-run/apply script
re-chunks only the affected stored papers from their Docling cache and replaces their `text_chunk` rows.

**Tech Stack:** Python 3.12 (`ml` conda env), Docling cache JSON, LanceDB, unittest/pytest.

**Spec:** this file, section "Approved Design" (agreed with the owner in chat on 2026-11-07).

## Approved Design

**Why.** Marvin's citer matches claims against stored chunks. A chunk that is really a reference list is
full of other papers' titles and topics, so it can score high and credit a paper for work it only cites.
Measured on 2026-11-07 (read-only): **1,508 stored text chunks in 227 papers** carry three or more
citation marks; in 6 random samples every one was a plain reference list.

**Where they come from.** `extract_docling` (`shared/docling_extract.py:252-319`) skips an item only if
Docling labels it `reference` or it sits under a heading matching `_BIB_HEAD`
(`^\s*(references?|bibliography|literature cited)\s*$`). In the 206 leaked papers measured, the leaked
items were Docling `list_item`s (6,792) or `text` (130) sitting under whatever heading came last
(Methods, Conclusion, appendices, "Online content", "Supplementary references"). References that follow
an Acknowledgments heading already never reach the store: `chunking.DROP_KINDS` drops that section kind.

**The rule (calibrated on the Docling cache, reviewer, 2026-11-07).**
- A **reference mark** is a match of any of these five patterns:
  1. volume, page (year): `\b\d{1,4}\s*,\s*\d{1,6}(?:\s*[-–]\s*\d+)?\s*\((?:18|19|20)\d{2}\)` (e.g. `82, 165109 (2010)`)
  2. volume (year) page: `\b\d{1,4}\s*\((?:18|19|20)\d{2}\)\s*\d{1,6}` (Elsevier, e.g. `102 (2007) 093511`)
  3. arXiv id: `\barXiv:\s?\d{4}\.\d{4,5}` (case-insensitive)
  4. DOI: `\bdoi(?:\.org/|:\s?)10\.\d{4,}` (case-insensitive)
  5. numbered entry with a year: `^\s*\[?\d{1,3}[\].)]\s+\S.{0,400}\b(?:18|19|20)\d{2}\b`
- A paragraph is a **reference entry** if it has ≥1 mark and is shorter than 600 characters, or has ≥3 marks
  at any length (several entries Docling merged into one item).
- A paragraph is a **continuation** if it is shorter than 600 characters and contains a year
  (`\b(?:18|19|20)\d{2}[a-z]?\b`); this keeps entries Docling split across items inside the run.
- A **run** starts at a reference entry and extends over following paragraphs that are entries or
  continuations. A run is dropped only if it holds **≥3 reference entries**. Runs never cross a heading.
- `_BIB_HEAD` also accepts a leading section number, an optional `supplementary`/`additional`/`cited`
  prefix, `references and notes`, `notes and references`, a bare `literature`, `литература`,
  `works cited`, and a trailing colon.
- **Calibration result:** on the 206 flagged papers the rule drops runs in 189 (9,986 paragraphs). On 250
  random unflagged papers it drops runs in 18. In every sampled case (12 printed) the dropped run was a
  reference list. The one non-reference case seen was three PLOS figure-link lines
  (`https://doi.org/10.1371/journal.pone.0206753.g005`), which are no loss.

**Repair (owner's go needed for `--apply`).** Only papers that have at least one stored `text_chunk` with
≥3 reference marks are candidates (227 today). Each candidate is re-chunked from its **Docling cache only**
(never a new conversion) with the fixed extractor. It is repaired only if the fresh chunks carry fewer
reference-heavy chunks than the stored ones. A repair replaces that paper's `text_chunk` rows and nothing
else: captions, formulas, figure images and the summary stay. Neighbour context is fetched within one
`stored_form` (`lance_store.py:410-430`), so mixed old/new `seq` values across forms are harmless.

**Non-goals.** Reference styles with no mark (e.g. ML papers citing `Venue, 2021.` with no arXiv id or DOI);
the legacy PyMuPDF/GROBID extraction path; regenerating summaries of repaired papers; the `batch_citer`
ordering flake.

## Global Constraints

- Work only in a worktree: from the main checkout run
  `git worktree add -b fix/reference-leak ../research_assistant-refleak 282f654`, then work in
  `../research_assistant-refleak`. Copy this plan file into the worktree and commit it first as
  `docs(refleak): add reference-leak plan`. Never run `git checkout`, `switch` or `commit` in
  `/home/shardul/marvin/migration-source/research_assistant` (Marvin may be running from it). No push, no merge.
- Interpreter: `/home/shardul/miniconda3/envs/ml/bin/python` (`$PY`). Tests:
  `CITATION_LOG_FILE=0 $PY -m pytest tests/ -q -p no:cacheprovider`. Tests never touch the real `data/`.
- Test first for every change: write the test, watch it fail for the stated reason, then implement.
- One commit per task, with the message given in the task.
- The repair script never runs Docling conversion, never deletes a paper's rows without fresh rows to write,
  and `--apply` runs only after the owner's go on a reviewed dry run, with Marvin stopped and
  `data/lancedb` backed up.
- Stop after each task and add a Handoff Log line (commit SHA, tests run, counts). Do not start Task 2's
  real dry run on `data/`; the reviewer gives that go.

## Review Focus

1. **A review article whose prose cites in full inline** (three paragraphs in a row, each with
   `Phys. Rev. B 82, 165109 (2010)`): the paragraphs are long, so they are not entries and must survive.
   Test: `test_long_prose_paragraphs_with_inline_citations_survive` (Task 1).
2. **A numbered experimental procedure** (`1. Anneal at 300 C for 2 h.`) has no year, so no run forms.
   Test: `test_numbered_procedure_without_years_survives` (Task 1).
3. **A candidate whose fresh extraction yields zero text chunks** (empty or broken cache): the stored rows
   must stay. Test: `test_a_paper_with_no_fresh_chunks_is_skipped_not_wiped` (Task 2).
4. **The store changed between dry run and apply** (e.g. the rerun ingested more papers): apply refuses.
   Test: `test_apply_refuses_when_the_scan_differs_from_the_plan` (Task 2).
5. **An upsert that fails after a paper's old rows were deleted**: the applied report must name that paper
   under `errors` as `deleted_not_reinserted` so the backup can restore it; other papers continue.
   Test: `test_a_failed_upsert_is_reported_as_deleted_not_reinserted` (Task 2).

## File Map

- Create `research_assistant/shared/bibliography.py`: reference marks, entries, runs, reference-heavy chunks.
- Modify `research_assistant/shared/extract.py:379`: wider `_BIB_HEAD`.
- Modify `research_assistant/shared/docling_extract.py:243-319, 620`: drop runs per section; real `n_bib`.
- Create `research_assistant/shared/reference_repair.py`: scan and apply logic (testable without `data/`).
- Create `scripts/repair_reference_leak.py`: CLI (dry run default, `--apply`).
- Tests: create `tests/test_bibliography.py`, `tests/test_reference_repair.py`; extend
  `tests/test_docling_phase5.py`.
- `PIPELINE.md`: one paragraph under ingestion on how references are kept out, and the repair script.

---

### Task 1: Keep reference runs out of extracted sections

**Files:**
- Create: `research_assistant/shared/bibliography.py`
- Modify: `research_assistant/shared/extract.py:379`, `research_assistant/shared/docling_extract.py:243-319, 620`
- Test: `tests/test_bibliography.py`, `tests/test_docling_phase5.py` (class `TestSyntheticDoclingExtraction`)

**Interfaces:**
- Produces (in `shared/bibliography.py`):
  - `REFERENCE_MARKS: tuple[re.Pattern, ...]` (the five patterns above, in that order)
  - `reference_marks(text: str) -> int` (total matches over all five patterns)
  - `is_reference_entry(text: str) -> bool`
  - `reference_runs(texts: Sequence[str], min_entries: int = 3) -> list[tuple[int, int]]` (half-open `[start, end)`)
  - `is_reference_heavy_chunk(text: str) -> bool` (`reference_marks(text) >= 3`)
- Produces: `Document.n_bib` from `extract_docling` = items skipped for a `reference` label + items under a
  bibliography heading + paragraphs dropped by runs.

Shared fixtures for the tests below (put them at the top of `tests/test_bibliography.py`):

```python
APS = "F. Bloch, Phys. Rev. A 7, 2187 (1973)."
APS2 = "A. Sartori, J. Marino, S. Stringari, and A. Recati, New J. Phys. 17, 093036 (2015)."
ELS = "S. Torquato, F. H. Stillinger, Tunneled crystals, J. Appl. Phys. 102 (2007) 093511."
ARX = "A. Gallemí, A. Muñoz Mateo, R. Mayol, and M. Guilleumas, arXiv:1509.04418."
DOI = "Cai, J. et al. Graphene nanoribbon heterojunctions. doi:10.1038/nnano.2014.184"
NUM = "[12] E. Sandier and S. Serfaty, 2D Coulomb gases and the renormalized energy, Ann. Probab., 2015."
SPLIT = "M. Benakli and S. Raghavan, Europhys. Lett. (1999)"     # a continuation, no mark
PROSE = "We grew the samples by chemical vapour deposition and measured them at 4.2 K."
```

- [ ] **Step 1: Write the failing tests in `tests/test_bibliography.py`**

```python
def test_each_reference_style_is_an_entry(self):
    for text in (APS, ELS, ARX, DOI, NUM):
        self.assertTrue(is_reference_entry(text), text)

def test_prose_and_bare_years_are_not_entries(self):
    for text in (PROSE, "In 2019 the group repeated the measurement.", SPLIT,
                 "The gap closes at 4.2 K, as shown in earlier work [12]."):
        self.assertFalse(is_reference_entry(text), text)

def test_a_run_needs_three_entries(self):
    self.assertEqual(reference_runs([PROSE, APS, APS2, PROSE]), [])
    self.assertEqual(reference_runs([PROSE, APS, APS2, ELS, PROSE]), [(1, 4)])

def test_a_split_entry_stays_inside_the_run(self):
    self.assertEqual(reference_runs([APS, SPLIT, APS2, ARX]), [(0, 4)])

def test_a_continuation_cannot_start_a_run(self):
    self.assertEqual(reference_runs([SPLIT, APS, APS2, ARX]), [(1, 4)])

def test_a_long_paragraph_ends_the_run(self):
    long_prose = PROSE * 9 + " (2010)"            # > 600 chars, one year, no mark
    self.assertEqual(reference_runs([APS, APS2, ARX, long_prose]), [(0, 3)])

def test_merged_entries_count_at_any_length(self):
    merged = " ".join([APS, APS2, ELS] * 4)        # > 600 chars, 12 marks
    self.assertTrue(is_reference_entry(merged))

def test_long_prose_paragraphs_with_inline_citations_survive(self):
    p = PROSE * 8 + " This agrees with Phys. Rev. B 82, 165109 (2010)."   # > 600 chars, 1 mark
    self.assertEqual(reference_runs([p, p, p]), [])

def test_numbered_procedure_without_years_survives(self):
    steps = ["1. Anneal at 300 C for 2 h.", "2. Cool to room temperature.", "3. Etch in HF for 30 s."]
    self.assertEqual(reference_runs(steps), [])

def test_reference_heavy_chunk_threshold(self):
    self.assertTrue(is_reference_heavy_chunk(" ".join([APS, APS2, ELS])))
    self.assertFalse(is_reference_heavy_chunk(" ".join([APS, APS2])))

def test_bibliography_heading_variants(self):
    from research_assistant.shared.extract import _BIB_HEAD
    for h in ("References", "REFERENCES", "VI. References", "5 References", "Supplementary references",
              "References and Notes", "Notes and references", "Bibliography", "Literature cited",
              "Literature", "Литература", "Works cited", "References:"):
        self.assertTrue(_BIB_HEAD.match(h), h)
    for h in ("Literature review", "Reference frames", "Notes", "Methods", "Referenced work"):
        self.assertFalse(_BIB_HEAD.match(h), h)
```

And in `tests/test_docling_phase5.py::TestSyntheticDoclingExtraction`, reusing the existing `FakeItem` /
`FakeDoc` pattern from `test_references_skipped_and_appendices_retained`:

```python
def test_a_reference_list_under_methods_is_kept_out(self):
    # items: section_header "Methods"; text PROSE; list_item APS, APS2, ELS, ARX; section_header "Extended Data"; text "Table S1 lists the samples."
    methods = next(s for s in doc.sections if s.heading == "Methods")
    self.assertEqual([p.text for p in methods.paragraphs], [PROSE])
    self.assertIn("Extended Data", [s.heading for s in doc.sections])
    self.assertEqual(doc.n_bib, 4)

def test_supplementary_references_heading_is_a_bibliography(self):
    # items: section_header "Results"; text PROSE; section_header "Supplementary references"; list_item APS; list_item "Unrelated note."
    self.assertNotIn("Supplementary references", [s.heading for s in doc.sections])
    self.assertNotIn("Unrelated note.", " ".join(p.text for s in doc.sections for p in s.paragraphs))

def test_n_bib_counts_labelled_headed_and_run_references(self):
    # the existing fixture's items, with section_header "Methods" + list_item APS, APS2, ELS inserted
    # after "Introductory discussion.": 2 labelled `reference` items + a run of 3 => 5
    self.assertEqual(doc.n_bib, 5)
    self.assertIn("Detailed derivation step.", " ".join(p.text for s in doc.sections for p in s.paragraphs))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `CITATION_LOG_FILE=0 $PY -m pytest tests/test_bibliography.py tests/test_docling_phase5.py -q -p no:cacheprovider`
Expected: `test_bibliography.py` fails to import `research_assistant.shared.bibliography`; the three new
extraction tests fail (Methods keeps the four references; `n_bib` is 0).

- [ ] **Step 3: Implement `shared/bibliography.py` and widen `_BIB_HEAD`**

Patterns, thresholds (600 characters, ≥3 marks, ≥3 entries) and the run algorithm exactly as in "The rule".
`_BIB_HEAD` stays anchored at both ends and case-insensitive.

- [ ] **Step 4: Apply runs in `extract_docling`**

Add `_drop_reference_runs(paras: list[Paragraph]) -> tuple[list[Paragraph], int]` in `docling_extract.py`
and pass `current_paras` through it at every place a section (or the abstract) is closed: the
bibliography-heading branch, the normal heading branch, the end-of-document branch, and the flat-text
fallback. Count label skips, items under a bibliography heading and run drops into `n_bib`, and pass it
to `Document(...)` instead of `n_bib=0`.

- [ ] **Step 5: Run the tests, then the full suite**

Run: `CITATION_LOG_FILE=0 $PY -m pytest tests/test_bibliography.py tests/test_docling_phase5.py tests/test_extract.py -q -p no:cacheprovider`
Expected: all pass. Then the full suite; report every failure by name (the `test_batch_citer` and
`test_app_render::TestDraftJobs` flakes are known; name them if they appear).

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/bibliography.py research_assistant/shared/extract.py research_assistant/shared/docling_extract.py tests/test_bibliography.py tests/test_docling_phase5.py
git commit -m "fix(extract): keep unlabelled reference lists out of the text chunks"
```

---

### Task 2: Repair the stored papers with leaked references

**Files:**
- Create: `research_assistant/shared/reference_repair.py`, `scripts/repair_reference_leak.py`
- Modify: `PIPELINE.md`
- Test: `tests/test_reference_repair.py`

**Interfaces:**
- Consumes: `is_reference_heavy_chunk` (Task 1); `process_pdf_v2` (`shared/ingest_v2.py`);
  `upsert_corpus`, `rebuild_bm25`, `_ingest_lock` (`shared/ingestion.py`); `db.open_store`,
  `db.clear_search_cache`; `PaperIndex.load(config.PAPER_INDEX_PATH)`; `_docling_cache_path`
  (`shared/docling_extract.py`); `atomic_write_json`.
- Produces (in `shared/reference_repair.py`):
  - `@dataclass LeakRepair: document: str; pdf_path: str; stored_chunks: int; stored_heavy: int; fresh_chunks: int; fresh_heavy: int`
  - `@dataclass LeakSkip: document: str; reason: str` with `reason` one of `"no_docling_cache"`,
    `"no_fresh_chunks"`, `"not_improved"`, `"extract_failed: <error>"`
  - `scan_reference_leaks(col, index, *, fresh_chunks=None) -> tuple[list[LeakRepair], list[LeakSkip], dict[str, list[dict]]]`
    — the third value maps each repaired document to its fresh `text_chunk` entries. `fresh_chunks` is an
    injectable `Callable[[str, str], list[dict]]` (document, pdf_path → text_chunk entries); the default
    calls `process_pdf_v2(pdf_path, citation_string=<stored citation_source>, images_dir=<a temp dir>,
    extractor="docling")` with `ingest_v2.EMBED_IMAGES` off and keeps `type == "text_chunk"`, and refuses
    (reason `no_docling_cache`) when `_docling_cache_path(pdf_path)` does not exist.
  - `apply_reference_repair(repairs, fresh, col, *, upsert=upsert_corpus) -> dict` with keys
    `versions_before`, `repaired` (list of documents), `errors` (list of `{"document", "error"}`).
- Candidates: every `.pdf` document with ≥1 stored `text_chunk` where `is_reference_heavy_chunk(text)`.
  Repaired only if `fresh_heavy < stored_heavy` and `fresh_chunks > 0`. PDF path:
  `index.resolve(document).path` if that file exists, else `os.path.join(config.PULLED_PDFS_DIR, document)`.
- Apply, per repaired document, inside `ingestion._ingest_lock()`: delete rows where
  `document = <doc> AND stored_form = 'text_chunk'`, then `upsert(fresh[doc])`. Once after all documents:
  `rebuild_bm25()` and `db.clear_search_cache()`.
- CLI `scripts/repair_reference_leak.py [--apply] [--plan PATH]`, plan default
  `<DATA_DIR>/reference_repair_plan.json`. Dry run writes the plan (repairs, skips, totals) and prints one
  line per repair plus totals (`N papers, X stored chunks → Y fresh chunks, H reference-heavy chunks removed`).
  `--apply` re-scans, refuses with `store changed since the dry run — rerun the dry run` (exit 1) if the
  repairs differ from the plan, otherwise applies and writes `<DATA_DIR>/reference_repair_applied-<UTC yyyymmdd-HHMM>.json`.
  Mirror `scripts/dedupe_store.py` for structure.

- [ ] **Step 1: Write the failing tests in `tests/test_reference_repair.py`**

Use a small in-memory fake collection (like `tests/test_store_dedupe.py::TestScanStore`'s `PagedCollection`)
holding rows `{"document", "stored_form", "text", "citation_source"}`, and a `fresh_chunks` stub.

```python
def test_only_papers_with_reference_heavy_chunks_are_candidates(self):
    # paper a.pdf: one heavy chunk; paper b.pdf: clean chunks only
    repairs, skips, fresh = scan_reference_leaks(col, index, fresh_chunks=stub_clean)
    self.assertEqual([r.document for r in repairs], ["a.pdf"])
    self.assertNotIn("b.pdf", [s.document for s in skips])

def test_a_paper_is_repaired_only_when_fresh_chunks_are_cleaner(self):
    # stub returns the same heavy chunk again for a.pdf
    repairs, skips, _ = scan_reference_leaks(col, index, fresh_chunks=stub_same)
    self.assertEqual(repairs, [])
    self.assertEqual([(s.document, s.reason) for s in skips], [("a.pdf", "not_improved")])

def test_a_paper_with_no_fresh_chunks_is_skipped_not_wiped(self):
    repairs, skips, _ = scan_reference_leaks(col, index, fresh_chunks=lambda d, p: [])
    self.assertEqual([(s.document, s.reason) for s in skips], [("a.pdf", "no_fresh_chunks")])
    result = apply_reference_repair(repairs, {}, col, upsert=fake_upsert)
    self.assertEqual(col.count(document="a.pdf", stored_form="text_chunk"), 2)   # untouched

def test_apply_replaces_only_text_chunk_rows(self):
    # a.pdf also has a caption and a figure_image row
    apply_reference_repair(repairs, fresh, col, upsert=fake_upsert)
    self.assertEqual(col.texts(document="a.pdf", stored_form="text_chunk"), ["clean body text"])
    self.assertEqual(col.count(document="a.pdf", stored_form="caption"), 1)
    self.assertEqual(col.count(document="a.pdf", stored_form="figure_image"), 1)

def test_a_failed_upsert_is_reported_as_deleted_not_reinserted(self):
    # two repairs, a.pdf and c.pdf; upsert raises for a.pdf only
    result = apply_reference_repair(repairs, fresh, col, upsert=upsert_failing_for_a)
    self.assertEqual(result["repaired"], ["c.pdf"])
    self.assertEqual(result["errors"], [{"document": "a.pdf", "error": "deleted_not_reinserted: boom"}])

def test_missing_docling_cache_is_skipped_without_conversion(self):
    # default fresh_chunks, cache path absent; patch process_pdf_v2 to fail the test if called
    _, skips, _ = scan_reference_leaks(col, index)
    self.assertEqual([(s.document, s.reason) for s in skips], [("a.pdf", "no_docling_cache")])

def test_apply_refuses_when_the_scan_differs_from_the_plan(self):
    # run main([]) to write the plan, add a heavy chunk for a new paper, run main(["--apply"])
    self.assertEqual(rc, 1)
    self.assertIn("store changed since the dry run", out)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `CITATION_LOG_FILE=0 $PY -m pytest tests/test_reference_repair.py -q -p no:cacheprovider`
Expected: fails to import `research_assistant.shared.reference_repair`.

- [ ] **Step 3: Implement `shared/reference_repair.py` and `scripts/repair_reference_leak.py`**

As in Interfaces. Read stored chunks per document with the collection's `get(where=..., include=[...])`
paging pattern `scan_store` already uses; never load vectors.

- [ ] **Step 4: Run the tests, then the full suite**

Run: `CITATION_LOG_FILE=0 $PY -m pytest tests/test_reference_repair.py tests/test_bibliography.py -q -p no:cacheprovider`
Expected: all pass. Then the full suite, failures reported by name.

- [ ] **Step 5: Document it in `PIPELINE.md`**

One paragraph under ingestion: how references are kept out (label, heading, runs of ≥3 entries) and that
`scripts/repair_reference_leak.py` repairs stored papers (dry run by default; `--apply` needs Marvin
stopped and a backup of `data/lancedb`).

- [ ] **Step 6: Commit and stop**

```bash
git add research_assistant/shared/reference_repair.py scripts/repair_reference_leak.py tests/test_reference_repair.py PIPELINE.md
git commit -m "feat(store): repair stored papers whose chunks hold reference lists"
```

Stop here. The reviewer runs or approves the real dry run on `data/`; `--apply` waits for the owner.

---

## Handoff Log

(Antigravity: one line per task — commit SHA, tests run with counts, anything that surprised you.)
