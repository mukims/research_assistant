# Pre-Parse Duplicate Check Implementation Plan

> **For agentic workers (Antigravity):** This file is your complete brief. Read all of it before
> touching code. If you have the superpowers skills, use `superpowers:executing-plans`; otherwise follow
> the tasks in order exactly as written. Track progress in a task artifact mirroring the `- [ ]` steps.
> The reviewer (Claude) checks each task; the owner decides merges.

**Goal:** Drop duplicate PDFs *before* the Docling full-text parse, including two copies of one paper
inside the same batch, so no parse time is spent on a paper that will be thrown away.

**Architecture:** `ingest_pdfs` gains a pre-parse step: read every candidate's header (GROBID first,
Docling first-pages conversion only when GROBID gives no title or authors), then drop each candidate that
matches an already-ingested paper or a copy accepted earlier in the same batch. The drop rule and the drop
actions are the existing layer-2 ones, pulled out of `_drop_title_duplicates` into two helpers shared by
the pre-parse step and the existing post-parse check, which stays as a backstop and reuses the headers.

**Tech Stack:** Python 3.12, existing GROBID `processHeaderDocument` and Docling helpers in
`shared/paper_header.py`, `unittest` tests under pytest. No new dependencies.

**Spec:** the "Approved Design" section below (agreed with the owner in chat on 2026-11-07).
Where a task and the design disagree, the design wins: stop and log it.

---

## Approved Design

**Why.** Today `_drop_title_duplicates` ([ingestion.py:964](../../../research_assistant/shared/ingestion.py))
runs only after *every* candidate has been parsed by Docling (about 2 minutes per uncached PDF). The
2026-11-07 run had 418 candidates, of which about 79 looked like duplicates, so up to ~2.5 hours of
parsing was thrown away. Its title/author lookup needs only the header, not the parse. It also cannot see
two copies inside one batch: a match counts only when the other copy is already in the ingest manifest,
and the manifest is written after the batch, so both copies get stored.

**Owner's decisions (2026-11-07):**
- A fixed rule, not an LLM agent.
- Deduplication uses the header's **title + first-author surnames**. The header comes from **GROBID**;
  **Docling is the fallback** when GROBID gives no usable title or authors.
- Docling then parses the full text of every PDF that survives the check, as today.

**The rule (unchanged from layer 2).** Two PDFs are the same paper when their folded titles
(`source_key.fold_title`) are equal and they share at least one first-author surname key
(`paper_index.surname_key`). A PDF without a usable header (no title or no authors from either GROBID or
Docling) is never dropped; it is parsed.

**Header read, per candidate (pre-parse):**
1. `read_header(path)` as today: GROBID header, merged with the Docling cache if one exists. No conversion.
2. Only if that returns `None` **and** the PDF has no Docling cache file: `read_header(path,
   convert_if_missing=True)`, which converts pages 1–2 with Docling.
3. If GROBID is down (`is_grobid_alive()` is false once, at the start), skip the pre-parse step entirely,
   log one warning, and leave everything to the post-parse check as today. Otherwise every PDF would get a
   first-pages Docling conversion and the batch would get slower, not faster.

**Which match drops a candidate:**
- **Store match:** an index entry with the same title + surname whose document is in the ingest manifest,
  whose file exists, and whose path is not the candidate's own. Search *all* title/author matches for an
  ingested one: today's `find_by_title_author` returns the first match only, which can be a pending,
  never-ingested copy hiding the ingested one behind it.
- **Batch match:** a candidate already accepted earlier in this batch with the same title + surname.
  Candidates are visited in keeper order, so the copy kept is: a `doi_…` file first, then an `arxiv_…`
  file, then the smallest document key (planner's call; matches the store cleanup's stub-group rule).
- A store match wins over a batch match.

**What dropping does (unchanged):** delete the PDF if it is inside `PULLED_PDFS_DIR` (never outside it),
alias the dropped document to the keeper in the paper index, repoint its `downloaded.json` entry to the
keeper's path, log `Duplicate dropped: …`, add the pipeline event, and add a record to
`result["duplicates"]`.

**Post-parse check (backstop):** stays where it is, uses the same rule, and reuses the pre-parse header
for every path that had one. Only paths whose pre-parse header was `None` are read again (by then a
Docling cache exists).

**Out of scope:** the match rule itself, Agent 2, the store-cleanup script, the `n_bib=0` log count.

**Accepted risk:** a batch keeper that later fails to parse leaves its dropped copy deleted. The keeper is
not marked as ingested, so the next run retries it.

## Global Constraints

- Interpreter `$PY` = `/home/shardul/miniconda3/envs/ml/bin/python`. Test command, from the worktree root:
  `CITATION_LOG_FILE=0 $PY -m pytest <targets> -q -p no:cacheprovider`.
- Baseline on `main` at `507a85a`: `2339 passed, 18 skipped`. The known flake
  `tests/test_batch_citer.py::TestRunBatchCiterPassesContextAndQuery::test_context_and_paragraph_always_reach_the_seam`
  may fail on some runs. Any other failure is yours.
- **Work in a separate git worktree.** The app is running a long ingest from the main checkout, and a
  branch switch there would change files under it. From
  `/home/shardul/marvin/migration-source/research_assistant`:
  `git worktree add ../research_assistant-preparse -b fix/preparse-dedupe main`, then work only in
  `../research_assistant-preparse`. Never `git checkout` in the main checkout. This plan file is
  untracked in the main checkout: copy it into the worktree at the same path, commit it with Task 1, and
  keep the Handoff Log in the worktree copy.
- One commit per task, `feat(dedupe): <what>`. **Do not push. Do not merge.**
- Tests never touch the real `data/` directory and never reach GROBID or run Docling: patch
  `research_assistant.shared.ingestion.read_header` and `research_assistant.shared.ingestion.is_grobid_alive`
  in every test that exercises the pre-parse step. (conftest already blocks the network.)
- Do not run the app, `fetch_papers` or `ingest_pdfs` against real data.
- Touch only the files each task lists. If reality differs from the plan, stop, log it in the Handoff Log
  and ask.

## Review Focus

1. GROBID down for the whole run: no first-pages Docling conversions, one warning, behaviour exactly as
   today (Task 2, `test_grobid_down_skips_the_pre_parse_check`).
2. An ingested match sitting behind a pending, never-ingested index entry with the same title and author:
   the candidate is still dropped against the ingested one (Tasks 1, 2).
3. Two copies in one batch where the non-`doi_` copy is listed first: the `doi_` copy is the one parsed
   and kept (Task 2).
4. Same title, different first authors, in one batch (two different papers with a generic title): both
   parsed (Task 2).
5. A cancel request during the header reads: nothing is parsed, nothing is marked, and the function
   returns without error (Task 2).

---

## File Map

| File | Responsibility |
|---|---|
| `research_assistant/shared/paper_index.py` | `PaperIndex.find_all_by_title_author`; `find_by_title_author` delegates to it |
| `research_assistant/shared/ingestion.py` | `_read_header_for_dedupe`, `_keeper_rank`, `_plan_drops`, `_execute_drops`, `_drop_duplicates_before_parse`; `_drop_title_duplicates` reuses headers; wiring in `_ingest_pdfs_locked` |
| `PIPELINE.md` | One paragraph: duplicates are now dropped before parsing |
| `tests/test_paper_index.py` | Task 1 |
| `tests/test_ingest_dedupe.py` | Task 2 (new tests in the existing `TestIngestDedupe` class, reusing its `setUp`) |

---

### Task 1: Find every title/author match in the paper index

**Files:**
- Modify: `research_assistant/shared/paper_index.py:166-187`
- Test: `tests/test_paper_index.py`

**Interfaces:**
- Produces: `PaperIndex.find_all_by_title_author(title: str | None, first_authors: Iterable[str],
  exclude: str | None = None) -> list[PaperEntry]`, in index order. `find_by_title_author` keeps its
  signature and returns the first element or `None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_find_all_by_title_author_returns_every_match(self):
    # index holds A ("Spin Transport in Graphene Bilayers", ("Tombros",)),
    # B (same title, ("Tombros", "Popinciuc")), C (same title, ("Geim",))
    found = idx.find_all_by_title_author("Spin transport in graphene bilayers", ["Tombros"])
    assert [e.document for e in found] == ["a.pdf", "b.pdf"]

def test_find_all_by_title_author_honours_exclude(self):
    found = idx.find_all_by_title_author("Spin Transport in Graphene Bilayers", ["Tombros"], exclude="a.pdf")
    assert [e.document for e in found] == ["b.pdf"]

def test_find_all_by_title_author_without_title_or_authors_is_empty(self):
    assert idx.find_all_by_title_author(None, ["Tombros"]) == []
    assert idx.find_all_by_title_author("Spin Transport in Graphene Bilayers", []) == []
```

- [ ] **Step 2:** Run `tests/test_paper_index.py`. Expected: the three new tests FAIL with `AttributeError`.
- [ ] **Step 3:** Implement `find_all_by_title_author`; make `find_by_title_author` return its first element or `None`.
- [ ] **Step 4:** Run `tests/test_paper_index.py tests/test_ingest_dedupe.py tests/test_build_paper_index.py`. Expected: all pass.
- [ ] **Step 5:** Commit `feat(dedupe): find every title/author match in the paper index`.

---

### Task 2: Drop duplicates before parsing

**Files:**
- Modify: `research_assistant/shared/ingestion.py` (`_drop_title_duplicates` at 641-724, `_ingest_pdfs_locked` at 782-1010, imports at 60-77)
- Modify: `PIPELINE.md`
- Test: `tests/test_ingest_dedupe.py`

**Interfaces:**
- Consumes: `PaperIndex.find_all_by_title_author` (Task 1); `read_header` and `docling_extract._docling_cache_path`;
  `grobid_manager.is_grobid_alive(timeout: float = 3.0) -> bool`, imported at module level in `ingestion.py`
  so tests patch `research_assistant.shared.ingestion.is_grobid_alive`.
- Produces (all private to `ingestion.py`):
  - `_read_header_for_dedupe(path: str) -> PaperHeader | None`: the two-step header read from the design.
    It calls the module-level name `read_header`, so existing test patches keep working.
  - `_keeper_rank(document: str) -> tuple[int, str]`: `(0, key)` for `doi_…`, `(1, key)` for `arxiv_…`, else `(2, key)`.
  - `_plan_drops(candidates: dict[str, str], headers: dict[str, PaperHeader | None], index: PaperIndex,
    ingested: set[str]) -> dict[str, PaperEntry | str]`: dropped path → the store `PaperEntry` it matches,
    or the keeper's *path* for a batch match. Pure apart from `os.path.isfile` on the match path.
  - `_execute_drops(drops: dict[str, PaperEntry | str], index: PaperIndex, log_prefix: str) -> list[dict]`:
    the existing delete / alias / repoint / log / event actions, returning records shaped exactly like
    today's (`path`, `document`, `kept`, `deleted`).
  - `_drop_duplicates_before_parse(candidates: dict[str, str], log_prefix: str = "") ->
    tuple[dict[str, str], list[dict], dict[str, PaperHeader | None]] | None`: kept candidates (original
    order), duplicate records, and the header read for every candidate. Returns `None` when a cancel is
    requested during the header reads.
  - `_drop_title_duplicates(candidates, corpus, log_prefix="", headers: dict[str, PaperHeader | None] | None = None)`:
    same return shape as today; uses `headers[path]` when present and not `None`, else `read_header(path)`.

**Wiring.** In `_ingest_pdfs_locked`, at the top of the `try:` block (before the parse loop): if
`is_grobid_alive()` is false, log `"%sGROBID is not reachable; skipping the pre-parse duplicate check."`
as a warning and continue as today. Otherwise call `_drop_duplicates_before_parse`; on `None`, add the
cancel event and `return result`; else replace `candidates`, keep the headers for the post-parse call, and
log one line: `"%sPre-parse duplicate check: %d of %d PDF(s) dropped (%d match the store, %d within this
batch); %d without a readable header will be parsed."`. During the header reads, update progress with
`detail=f"Checking for duplicates [{i}/{n}]: {name}"`. `result["duplicates"]` is the pre-parse records
followed by the post-parse ones.

- [ ] **Step 1: Write the failing tests** (in `TestIngestDedupe`; each one patches `ing.is_grobid_alive`
  to `True` unless stated, `process_pdf`, `upsert_corpus` and `rebuild_bm25` as the existing
  `test_duplicate_never_reaches_the_store` does, and records the paths `process_pdf` was called with)

```python
def test_store_duplicate_is_dropped_before_parsing(self):
    # manifest has P; ingest_pdfs([dup_pdf, uniq_pdf]); dup header matches P
    assert parsed_paths == [uniq_pdf]
    assert not os.path.exists(dup_pdf)
    assert [d["kept"] for d in result["duplicates"]] == ["existing_p.pdf"]

def test_two_copies_in_one_batch_keep_the_doi_file(self):
    # pulled/title_abc.pdf listed before pulled/doi_10.1_x.pdf; same header
    # ("Spin Transport in Graphene Bilayers", ("Tombros",)); nothing in the store matches;
    # downloaded.json has {"ref": {"path": title_abc_path}}
    assert parsed_paths == [doi_path]
    assert not os.path.exists(title_abc_path)
    assert result["duplicates"][0]["document"] == "title_abc.pdf"
    assert result["duplicates"][0]["kept"] == "doi_10.1_x.pdf"
    assert PaperIndex.load(self.index_path).resolve("title_abc.pdf").document == "doi_10.1_x.pdf"
    assert json.load(open(self.downloaded_path))["ref"]["path"] == doi_path
    assert "doi_10.1_x.pdf" in manifest_mod.load() and "title_abc.pdf" not in manifest_mod.load()

def test_same_title_different_authors_in_one_batch_are_both_parsed(self):
    # "A Survey of Large Language Models" by ("Zhao",) and by ("Chang",)
    assert sorted(parsed_paths) == sorted([a_path, b_path])
    assert result["duplicates"] == []

def test_ingested_match_behind_a_pending_entry_still_drops(self):
    # index rebuilt with pending Q (file exists, not in manifest) added BEFORE ingested P,
    # both "Half-Metallic Graphene Nanoribbons" / ("Son",); candidate dup header matches both
    assert parsed_paths == []
    assert result["duplicates"][0]["kept"] == "existing_p.pdf"

def test_grobid_without_a_header_falls_back_to_docling_first_pages(self):
    # fake read_header returns None unless convert_if_missing=True, then the P-matching header
    assert parsed_paths == []
    assert any(k.get("convert_if_missing") for k in read_header_kwargs)

def test_readable_grobid_header_never_converts(self):
    assert not any(k.get("convert_if_missing") for k in read_header_kwargs)

def test_pdf_with_a_docling_cache_is_not_converted_again(self):
    # read_header always None; a file exists at docling_extract._docling_cache_path(pdf) (patch DOCLING_CACHE_DIR to tmp)
    assert not any(k.get("convert_if_missing") for k in read_header_kwargs)
    assert parsed_paths == [pdf]

def test_unreadable_header_is_parsed(self):
    # read_header always None, no cache
    assert parsed_paths == [pdf]
    assert result["duplicates"] == []

def test_post_parse_check_reuses_pre_parse_headers(self):
    # one unique PDF with a readable header
    assert read_header_calls_for(pdf) == 1

def test_batch_of_only_duplicates_parses_and_stores_nothing(self):
    assert parsed_paths == [] and upsert_calls == 0
    assert len(result["duplicates"]) == 1

def test_grobid_down_skips_the_pre_parse_check(self):
    # is_grobid_alive -> False; read_header returns the P-matching header
    assert parsed_paths == [dup_pdf]          # parsed, as today
    assert not any(k.get("convert_if_missing") for k in read_header_kwargs)
    assert len(result["duplicates"]) == 1     # still dropped by the post-parse check

def test_cancel_during_header_check_parses_nothing(self):
    # pipeline_status.is_cancel_requested -> True
    assert parsed_paths == []
    assert not manifest_mod.load()
```

- [ ] **Step 2:** Run `tests/test_ingest_dedupe.py`. Expected: the new tests FAIL; the existing ones pass.
- [ ] **Step 3:** Implement the helpers and the wiring per the Interfaces block and the Approved Design.
  `_plan_drops` visits candidates sorted by `_keeper_rank(pdf_key(path))`; for each with a header that has
  a title and authors, look for a store match first (first entry from `find_all_by_title_author(…,
  exclude=doc_key)` that is in `ingested`, has an existing file, and is not the candidate's path), then a
  batch match among already-accepted candidates (same `fold_title`, surname-key overlap). Load the
  ingested set once with `manifest.load()`. Hold `edit(PAPER_INDEX_PATH)` across plan + execute, as today.
- [ ] **Step 4:** Confirm `is_grobid_alive()` returns `False` (does not raise) under conftest's network
  block, so tests that do not patch it keep today's behaviour. Run `tests/test_ingest_dedupe.py`. Expected: all pass.
- [ ] **Step 5:** Add one paragraph to `PIPELINE.md` where the ingest dedupe is described: duplicates are now
  dropped before the Docling parse, using GROBID's header with a Docling first-pages fallback, including
  copies within one batch; the post-parse check remains as a backstop.
- [ ] **Step 6:** Run the full suite. Expected: `2339 + new tests` passed, 18 skipped (the known flake aside).
- [ ] **Step 7:** Commit `feat(dedupe): drop duplicate PDFs before parsing`.

---

### After Task 2 (reviewer and owner, not Antigravity)

The reviewer reviews the branch. Merge and an app restart happen only on the owner's go, and only after
the 2026-11-07 ingest has finished. The first real check is the next seed run's log: the
`Pre-parse duplicate check:` line, and no Docling conversion for any `Duplicate dropped:` document.

---

## Review Round 1 (reviewer, 2026-11-07, against `4ae1e18`)

Everything else matches the plan: the rule, keeper order, GROBID-down skip, header reuse, cancel path,
`PIPELINE.md`. Full suite in the worktree: 2354 passed, 18 skipped, 1 error in
`tests/test_app_render.py::TestDraftJobs::test_a_second_run_while_one_is_active_warns_and_starts_nothing`
that passes on its own (13/13 for the class); not caused by this branch.

### R11 (blocking): a forced re-ingest deletes both stored copies of a paper

`_plan_drops` plans every drop before `_execute_drops` deletes anything. When the batch contains two
copies that are **both already ingested** (any `skip_ingested=False` run: Force in the corpus and audit
tabs, `agent3 --force`, `agent6`, `orchestrate` with `force`), each finds the other as a "store match":
`title_x.pdf` is dropped against `title_y.pdf` and `title_y.pdf` against `title_x.pdf`. Both files are
deleted from `pulled_pdfs/`, both are aliased to each other (a cycle), and neither is re-parsed.
The old post-parse loop was safe here because it deleted as it went, so the second copy's
`os.path.isfile(match.path)` was false.

Reviewer's reproduction (two stored copies, same header, `ingest_pdfs([x, y], skip_ingested=False)`):
`Duplicate dropped: title_x.pdf (matches title_y.pdf, deleted=True)`, then
`Duplicate dropped: title_y.pdf (matches title_x.pdf, deleted=True)`, then
`Pre-parse duplicate check: 2 of 2 PDF(s) dropped (0 match the store, 2 within this batch)`.

**Fix:** in `_plan_drops`, a store-match entry whose path is one of this batch's candidates is not a store
match (compare `os.path.abspath`); those copies are settled by the batch rule, which only matches against
already-accepted candidates and so can never drop both. Keeper order then holds in force mode too.

**Test** (in `TestIngestDedupe`):

```python
def test_force_reingest_of_two_stored_copies_keeps_one(self):
    # title_x.pdf and title_y.pdf in pulled_dir, both in the index and the manifest, same header
    # ("Spin Transport in Graphene Bilayers", ("Tombros",)); ingest_pdfs([y, x], skip_ingested=False)
    assert os.path.exists(x_path) and not os.path.exists(y_path)
    assert parsed_paths == [x_path]
    assert [(d["document"], d["kept"]) for d in result["duplicates"]] == [("title_y.pdf", "title_x.pdf")]
    assert PaperIndex.load(self.index_path).resolve("title_x.pdf").document == "title_x.pdf"
```

The pre-parse log line's store/batch split (which counts a drop as "batch" when its keeper is a
candidate) is correct once R11 is fixed; no separate change needed.

Commit `fix(dedupe): never drop two batch copies against each other`. Append a Handoff Log line, stop.

---

## Handoff Log (append-only)

Format: `date time · task · sha · result · discrepancies`.

2026-11-07 15:38 · Task 1 · b9b71e2 · tests/test_paper_index.py (25 passed), tests/test_ingest_dedupe.py (8 passed), tests/test_build_paper_index.py (6 passed) · none
2026-11-07 15:49 · Task 2 · 1a3e6ea · tests/test_ingest_dedupe.py (20 passed), full suite (2353 passed, 18 skipped, 1 failed known flake in test_batch_citer.py) · none
2026-11-07 17:32 · R11 · 83643cf · test_force_reingest_of_two_stored_copies_keeps_one (passed), tests/test_ingest_dedupe.py + test_paper_index.py (46 passed), full suite (2354 passed, 18 skipped, 1 failed known flake in test_batch_citer.py) · none
