# Two-Layer Duplicate Guard Implementation Plan

> **For agentic workers (Antigravity):** This file is your complete brief. Read all of it before
> touching code. If you have the superpowers skills, use `superpowers:executing-plans`; otherwise follow
> the tasks in order exactly as written. Track progress in a **task artifact** (a markdown checklist
> mirroring the `- [ ]` steps below) and tick each step as you finish it.

**Goal:** Stop Marvin from downloading or ingesting a paper that is already in the corpus: skip the
download when the DOI/arXiv ID is already known (layer 1), and drop + delete a freshly downloaded PDF
whose title and first author match a corpus paper (layer 2), with a report-only sweep at the end of
every run.

**Architecture:** A new on-disk *paper index* (`data/paper_index.json`) records, per corpus paper, its
DOIs, arXiv IDs, title and first-author surname. Both download paths (Agent 0 seed download, Agent 2
reference fetch, pasted DOIs) consult it before downloading. The single ingestion door (`ingest_pdfs`)
reads each new PDF's title and first author with GROBID's header model, falling back to the Docling
conversion it already performs, drops
title+author duplicates before they reach the store, and adds every successfully ingested paper to the
index. A sweep runs at the end of every `ingest_pdfs` and `fetch_papers` call and writes
`data/dedupe_report.json`.

**Tech Stack:** Python 3.12; GROBID 0.8.x at `config.GROBID_SERVER` (already running for Agent 1),
called on `processHeaderDocument`; Docling 2.133 (already the ingestion extractor; it caches every
converted document as `config.DOCLING_CACHE_DIR/<pdf stem>.docling.json`); `requests`, BeautifulSoup
(`"xml"` parser, as Agent 1 uses); `unittest` tests run under pytest. No new dependencies.

**Spec:** the approved design is the next section of this file (agreed in chat with the project owner
on 2026-11-06). Where a task and the design disagree, the design wins: stop and note it in the log.

---

## Approved Design

**Why.** Every Marvin run (Citation auditor, Research idea, Corpus library, Labs) processes a PDF and
downloads the papers it cites when they are not already in the corpus. Today "already have it" is decided
only by `downloaded.json` keys and file names, so the same paper arriving under a different DOI, key or
metadata is downloaded and ingested again. The owner wants that to stop.

**Layer 1 — DOI before download.** If a reference's (or seed candidate's) DOI or arXiv ID is already in
the corpus, do not download. Point the run at the existing file so audit/synthesis still have evidence.
Applies to all three download paths: Agent 2 (`fetch_papers`), Agent 0 (`download_seed_cascade`,
`_download_and_record`) and the paste-a-DOI/arXiv box (`batch_uploader.fetch_and_stage_urls_or_dois`).
Agent 0 must look through the whole corpus, not only its own file name in `data/raw/`.

**Layer 2 — title + first author after download.** At ingestion, read the new PDF's own title and
first author — **GROBID header first, Docling as fallback** (owner's decision after a head-to-head
test, recorded below). If its normalised title matches a corpus paper **and** the two papers share a
first-author surname (from either extractor), do not ingest it, delete the downloaded file, and record
its key as an alias of the existing paper.

**Why both extractors (test of 2026-11-06, 80 random corpus papers):** first author found by GROBID
76/80, by the Docling rules 58/80, by either 78/80. Of the 7 real disagreements GROBID was right 6
times; the Docling rules were right once (GROBID took a Frontiers paper's *editor* as its author).
GROBID header median 1.2 s per paper, max 13.5 s. Titles agreed in 70 of 73 cases where both found one.

**Sweep — end of every run.** Re-run both tests over the whole index and write any duplicate groups that
slipped through to `data/dedupe_report.json`. Report only.

**Decisions made by the planner (owner may override):**
- Only files inside `PULLED_PDFS_DIR` (Agent 2 downloads) are ever deleted. `RAW_DIR` holds seed
  papers *and* user uploads (`batch_uploader` stages uploads there), so a layer-2 match there is skipped
  from ingestion but left on disk. Agent 1 also reads the seed from `RAW_DIR` later in the run.
- Nothing is ever removed from LanceDB.
- Each paper keeps **every** surname either extractor produced (GROBID's first). Two papers match when
  their folded titles are equal and their surname sets share at least one surname key (last word of
  the surname, accents and encoding marks removed). Title comes from GROBID, else Docling.
- Docling has no author label (authors arrive as plain `text` items), so its author comes from
  conservative rules (Task 2); they are a fallback, not the primary source.
- When neither extractor yields a title and at least one surname — GROBID down or failing *and* no
  usable Docling cache — layer 2 does nothing for that PDF and it is ingested as today. Missing a
  duplicate is acceptable; deleting the wrong file is not.
- Layer 2 runs after Docling has converted the PDF (the conversion is already paid for) and before
  anything is written to the store.
- Normalisation reuses the existing rules in `shared/source_key.py` (`_fold_title`, `_fold`,
  `normalise_doi`).

**Out of scope:** per-paper knowledge graphs (next project; will later replace the layer-2 test),
fuzzy/content similarity, cleaning existing duplicates out of LanceDB.

## Global Constraints

- Interpreter: `/home/shardul/miniconda3/envs/ml/bin/python` (call it `$PY`). The base conda `python3`
  lacks `lancedb` and must not be used.
- Test command: `CITATION_LOG_FILE=0 $PY -m pytest <targets> -q -p no:cacheprovider`, run from
  `migration-source/research_assistant/`.
- Baseline before you start: full suite `2238 passed, 18 skipped`. It must end green with only your new
  tests added.
- Work on branch `feat/two-layer-dedupe` created from the current `main`. One commit per task, message
  style `feat(dedupe): <what>` (repo uses conventional commits). **Do not push. Do not merge.**
- The working tree on `main` already has uncommitted changes (the on-disk PDF check in `shared/fetch.py`,
  `agents/agent2_fetcher.py`, `schemas.py`, two test files, `PIPELINE.md`). Build on them; do not revert
  or reformat them. If they are still uncommitted when you start, stop and ask the owner.
- Tests must never read or write the real `data/` directory: patch every path constant into a temp dir.
- GROBID being down, slow (≤ 60 s timeout per call) or wrong, and a header that cannot be read (no
  Docling cache, bad JSON, no title, no author), must never fail or block a run.
- Never delete a file outside `PULLED_PDFS_DIR`. Never delete from LanceDB.
- Do not run `fetch_papers`, `ingest_pdfs` or the app against real data, except the backfill in Task 7.
- Touch only the files each task lists. If reality differs from this plan (a name, a line, a signature),
  stop, append the discrepancy to the log at the end of this file, and ask rather than improvise.

## Review Focus

Most likely to bite a real user; each has a test in the owning task.

1. GROBID down → Docling's reading is used; GROBID down *and* no usable Docling header (no cache, a
   ResearchGate-style cover page, an affiliation where the author line should be) → no author, the PDF
   is kept and ingested, nothing deleted (Tasks 2, 5).
2. A seed paper or user upload in `RAW_DIR` that matches a corpus paper → skipped, never deleted (Task 5).
3. An index entry whose file no longer exists → treated as no match, the paper is downloaded/ingested
   normally (Tasks 3, 5).
4. Two threads updating the index at once (watcher + job thread) → no lost entries (Task 1).
5. A corrupt or missing `paper_index.json` → treated as an empty index, never a crash (Task 1, Task 6).

---

## File Map

| File | Responsibility |
|---|---|
| `research_assistant/shared/paper_index.py` (new) | Index model, normalised lookups, aliases, duplicate groups, locked edit, manifest repoint |
| `research_assistant/shared/paper_header.py` (new) | Title / first-author surnames / DOI / arXiv ID: GROBID `processHeaderDocument`, plus a Docling document (cached JSON, or a pages-1–2 conversion for the backfill), merged |
| `research_assistant/shared/dedupe_sweep.py` (new) | End-of-run report |
| `research_assistant/shared/source_key.py` | Public names for existing title/name folding |
| `research_assistant/config.py` | `PAPER_INDEX_PATH`, `DEDUPE_REPORT_PATH` |
| `research_assistant/agents/agent2_fetcher.py` | Layer 1 for references; sweep hook |
| `research_assistant/agents/agent0_discoverer.py` | Layer 1 for seed papers |
| `research_assistant/shared/batch_uploader.py` | Layer 1 for pasted DOIs / arXiv IDs |
| `research_assistant/shared/ingestion.py` | Layer 2; index update after ingest; sweep hook |
| `scripts/build_paper_index.py` (new) | One-off/resumable backfill of the index from the existing corpus |
| `PIPELINE.md` | One paragraph documenting the guard |

---

### Task 1: Paper index

**Files:**
- Create: `research_assistant/shared/paper_index.py`
- Modify: `research_assistant/config.py` (add two paths right after `INGEST_LOCK_PATH`)
- Modify: `research_assistant/shared/source_key.py` (add public aliases)
- Test: `tests/test_paper_index.py`

**Interfaces — Produces:**
```python
# config.py
PAPER_INDEX_PATH   = os.path.join(DATA_DIR, "paper_index.json")
DEDUPE_REPORT_PATH = os.path.join(DATA_DIR, "dedupe_report.json")

# source_key.py — public names for the existing private helpers (same behaviour)
def fold_title(title: str | None) -> str      # = _fold_title
def fold_name(text: str | None) -> str        # = _fold

# paper_index.py
@dataclass(frozen=True)
class PaperEntry:
    document: str                     # ingest-manifest key: pdf_key(path), e.g. "doi_10.1038_nature05180.pdf"
    path: str
    title: str | None = None          # as read; normalised only for comparison
    first_authors: tuple[str, ...] = ()   # first-author surnames as read by each extractor, GROBID's first
    dois: tuple[str, ...] = ()
    arxiv_ids: tuple[str, ...] = ()

def normalise_arxiv(arxiv_id: str | None) -> str | None   # lowercase, strip "arxiv:" prefix and trailing r"v\d+"
def surname_key(name: str | None) -> str | None
    # NFKD; drop every char whose unicodedata.category is Mn, Sk or Lm (so "Kulyt˙e" and "Kulytė" agree);
    # then fold_name(); return the LAST whitespace token ("Essam Ghareeb" -> "ghareeb"); empty -> None

class PaperIndex:
    path: str
    @classmethod
    def load(cls, path: str) -> "PaperIndex"     # missing or unparseable file -> empty index, logs a warning
    def save(self) -> None                       # atomic_write_json(self.path, ...)
    def add(self, entry: PaperEntry) -> None     # replaces an entry with the same document
    def get(self, document: str) -> PaperEntry | None
    def find_by_ids(self, doi: str | None = None, arxiv_id: str | None = None) -> PaperEntry | None
    def find_by_title_author(self, title: str | None, first_authors: Iterable[str],
                             exclude: str | None = None) -> PaperEntry | None
    def add_alias(self, alias: str, document: str) -> None
    def resolve(self, document_or_alias: str) -> PaperEntry | None
    def duplicate_groups(self) -> list[dict]     # [{"reason": "doi"|"arxiv"|"title_author", "key": str, "documents": [sorted str]}]
    def __len__(self) -> int

@contextmanager
def edit(path: str) -> Iterator[PaperIndex]       # module-level threading.Lock held; load, yield, save on clean exit

def repoint_manifest(manifest_path: str, old_path: str, new_path: str) -> int
    # rewrite downloaded.json records whose "path" basename == basename(old_path) to new_path; returns count
```
On-disk shape: `{"version": 1, "papers": {document: {path, title, first_authors, dois, arxiv_ids}}, "aliases": {alias: document}}`.
Matching: DOIs via `normalise_doi`; arXiv via `normalise_arxiv`; title via `fold_title`; authors via
`surname_key`. `find_by_title_author` matches an entry when the folded titles are equal **and** the
two sets of surname keys intersect; it returns `None` unless the folded title and at least one surname
key are non-empty. `duplicate_groups` uses the same rule (a `title_author` group's `key` is the folded
title). `find_by_ids(None, None)` returns `None`.

- [ ] **Step 1: Write the failing tests** in `tests/test_paper_index.py` (each builds a `PaperIndex` in a temp dir):
  - `test_find_by_ids_matches_doi_regardless_of_prefix_and_case` — entry `dois=("10.1038/nature05180",)`; `find_by_ids(doi="https://doi.org/10.1038/NATURE05180").document == "doi_10.1038_nature05180.pdf"`
  - `test_find_by_ids_matches_arxiv_across_versions` — entry `arxiv_ids=("2007.12504",)`; `find_by_ids(arxiv_id="arXiv:2007.12504v2")` hits
  - `test_find_by_ids_with_no_ids_finds_nothing` — an entry with no ids exists; `find_by_ids(None, None) is None`
  - `test_title_author_match_ignores_case_punctuation_and_filler_words` — entry `title="Half-Metallic Graphene Nanoribbons", first_authors=("Son",)`; `find_by_title_author("half metallic graphene nanoribbons.", ["SON"])` hits
  - `test_same_title_different_first_author_is_not_a_match` — same title, authors `["Geim"]` → `None`
  - `test_title_or_author_missing_never_matches` — `(title, [])` and `(None, ["Son"])` → `None`
  - `test_any_shared_surname_matches` — entry `first_authors=("Zaghouani", "Müller")` (GROBID took the editor, Docling the author); query `["Muller"]` → hits
  - `test_surname_key_handles_compound_and_encoding_artifacts` — `surname_key("Essam Ghareeb") == surname_key("Ghareeb") == "ghareeb"`; `surname_key("Kulyt˙e") == surname_key("Kulytė") == "kulyte"`; `surname_key("") is None`
  - `test_exclude_skips_the_paper_itself` — `exclude=<that document>` → `None`
  - `test_alias_resolves_to_the_kept_paper` — `add_alias("title_x.pdf", "doi_y.pdf")`; `resolve("title_x.pdf").document == "doi_y.pdf"`
  - `test_round_trips_through_disk` — entries + aliases written via `edit()` equal those from `PaperIndex.load()`
  - `test_unreadable_file_loads_as_empty` — file containing `{not json` → `len(PaperIndex.load(p)) == 0`, no exception
  - `test_concurrent_edits_lose_nothing` — 8 threads each `with edit(p) as idx: idx.add(<distinct entry>)`; reload has 8 entries
  - `test_duplicate_groups_reports_shared_doi_and_shared_title_author` — A,B share a DOI; C,D share title+author; E unique → exactly `[{"reason":"doi",...,"documents":["A","B"]}, {"reason":"title_author",...,"documents":["C","D"]}]` (compare reason + documents)
  - `test_repoint_manifest_moves_matching_records_only` — manifest with two records, one whose path basename matches → returns `1`, that record's `path` is the new path, the other unchanged
- [ ] **Step 2: Run them and confirm they fail** — `... tests/test_paper_index.py` → `ModuleNotFoundError` / `ImportError` for `paper_index`.
- [ ] **Step 3: Implement** the config paths, the `source_key` aliases and `paper_index.py` per the interface above.
- [ ] **Step 4: Run them and confirm they pass**, then run `tests/test_fetcher.py tests/test_download.py` (must stay green).
- [ ] **Step 5: Commit** — `feat(dedupe): add corpus paper index`

### Task 2: Header reader

**Files:**
- Create: `research_assistant/shared/paper_header.py`
- Test: `tests/test_paper_header.py`

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class PaperHeader:
    title: str | None
    first_authors: tuple[str, ...]   # first-author surnames, GROBID's first; () when none
    doi: str | None                  # normalise_doi() form
    arxiv_id: str | None             # normalise_arxiv() form (Task 1)

# --- GROBID (primary) ---
def header_from_grobid_tei(tei_xml: str) -> PaperHeader
def grobid_header(pdf_path: str, server: str = GROBID_SERVER, timeout: float = 60) -> PaperHeader | None
    # POST multipart field "input" to f"{server.rstrip('/')}/api/processHeaderDocument",
    # data={"consolidateHeader": "0"} (no outside lookups), headers={"Accept": "application/xml"},
    # via requests.post on this module (tests patch paper_header.requests.post).
    # Exception, non-200 or empty body -> None.

# --- Docling (fallback) ---
def header_from_docling(doc: dict) -> PaperHeader
    # pure; *doc* is a DoclingDocument as JSON, i.e. what extract_docling caches (doc.model_dump(mode="json"))
def docling_header(pdf_path: str, convert_if_missing: bool = False) -> PaperHeader | None
    # loads docling_extract._docling_cache_path(pdf_path); if absent and convert_if_missing, converts
    # pages 1–2 only: get_digital_converter().convert(pdf_path, page_range=(1, 2)) (no OCR, result not cached)
    # and passes .document.model_dump(mode="json") to header_from_docling.
    # Absent cache without convert_if_missing, unreadable JSON, or any exception -> None.

# --- merged (what Tasks 5 and 7 call) ---
def read_header(pdf_path: str, convert_if_missing: bool = False) -> PaperHeader | None
    # g = grobid_header(pdf_path); d = docling_header(pdf_path, convert_if_missing)
    # title = g.title or d.title; doi / arxiv_id = GROBID's else Docling's;
    # first_authors = GROBID's then Docling's, dropping any whose surname_key is already present.
    # None when the merged title is None or first_authors is empty.
```
`header_from_grobid_tei` uses `BeautifulSoup(tei, "xml")`: title = `teiHeader fileDesc titleStmt title`;
surname = first `sourceDesc biblStruct analytic author persName surname`; `doi` = `idno[type="DOI"]`;
`arxiv_id` = `idno[type="arXiv"]`; blank strings become `None`; `first_authors` is `(surname,)` or `()`.

Rules for `header_from_docling` (all decided — these are the algorithm, implement them as written).
Look only at `doc["texts"]` items whose first `prov` entry has `page_no == 1`, in document order.
- **Title:** the first item labelled `title`; otherwise the first `section_header` whose text has at least
  4 words and whose folded text is not one of: `abstract`, `introduction`, `contents`, `article`,
  `research article`, `original article`, `review`, `letter`, and that contains none of the whole words
  `journal proceedings transactions bulletin magazine repository portal` (venue or repository-cover
  banners, not titles). None found → `title=None`.
- **First author:** scan at most the 12 `text` items after the title item. Take the first whose text,
  after removing `* ∗ † ‡ § ¶` (`∗` is U+2217, common in Docling output) and digits, (a) contains none of `@`, `http`, `www.`, and none of the words
  `university department institute laboratory school college centre center faculty abstract doi
  received accepted published article journal vol volume citations reads keywords email correspondence
  copyright` (case-insensitive, whole words), and (b) whose first name segment — the part before the
  first `,` `;` `·` `&` or ` and ` — is 2 to 4 whitespace tokens, each starting with an uppercase letter.
  Surname = the last token of that segment with leading and trailing non-letter characters removed; if
  fewer than 2 letters remain (a bare initial), no surname (`first_authors=()`). No such item, or no
  title → `first_authors=()`.
- **DOI:** first match of `\b10\.\d{4,9}/[^\s"<>]+` in page-1 text (strip trailing `.,;)`), then
  `normalise_doi`. **arXiv:** first match of `arXiv:\s*([0-9]{4}\.[0-9]{4,5})(v\d+)?`, then
  `normalise_arxiv`.

Calibration (planner's probes, 2026-11-06). Docling rules alone, over the 1,553 files in
`data/docling_cache/`: title for 1,469, first author for 1,067. Head-to-head on 80 random corpus papers:
first author from GROBID 76/80, Docling rules 58/80, merged `read_header` 78/80; title and author from
either 77/80. Task 7's backfill should therefore report roughly 95%+ of papers with a title and at least
one surname; if it is far below that, stop and log.

Fixtures are hand-written dicts in `tests/test_paper_header.py` with the DoclingDocument shape
(`{"texts": [{"label": ..., "text": ..., "prov": [{"page_no": 1}]}, ...]}`). The cover-page fixture
is modelled on the real first page of `data/docling_cache/1806-1117-rbef-39-01-e1303.docling.json`
(read it to see the shape; do not load it from tests):
`text "See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/..."`,
`section_header "Pedagogical introduction to equilibrium Green's functions: Condensed-matter examples with numerical implementations"`,
`text "Article in Revista Brasileira de Ensino de Física · October 2016"`, `text "DOI: 10.1590/1806-9126-RBEF-2016-0087"`,
`text "CITATIONS"`, `text "25"`, `text "READS"`, `text "4,187"`, `text "3 authors, including:"`,
`text "Mariana M. Odashima"`, `text "Universidade Federal de Uberlândia (UFU)"`.

- [ ] **Step 1: Write the failing tests.** In the Docling tests, `first_author == "X"` is shorthand for
  `header.first_authors == ("X",)` and `first_author is None` for `header.first_authors == ()`.
  GROBID tests use a hand-written TEI fixture (title "Half-metallic graphene nanoribbons", authors Son
  then Cohen, DOI `10.1038/nature05180`, arXiv `cond-mat/0611602`).
  - `test_grobid_tei_yields_title_surname_doi_and_arxiv` — `PaperHeader("Half-metallic graphene nanoribbons", ("Son",), "10.1038/nature05180", "cond-mat/0611602")`
  - `test_grobid_unreachable_returns_none` — `requests.post` raises `requests.ConnectionError` → `grobid_header(...) is None`
  - `test_grobid_non_200_returns_none` — fake response `status_code=503` → `None`
  - `test_grobid_posts_header_request_without_consolidation` — fake post records URL and data, returns 200 + fixture → URL ends with `/api/processHeaderDocument`, `data["consolidateHeader"] == "0"`
  - `test_read_header_merges_both_sources` — `grobid_header` stubbed to `("T", ("Zaghouani",))`, `docling_header` to `("T", ("Müller",))` → `first_authors == ("Zaghouani", "Müller")`, `title == "T"`
  - `test_read_header_keeps_one_spelling_per_surname` — GROBID `("Essam Ghareeb",)`, Docling `("Ghareeb",)` → `first_authors == ("Essam Ghareeb",)`
  - `test_read_header_falls_back_to_docling_when_grobid_is_down` — `grobid_header` → `None`, Docling has title + `("Odashima",)` → that header
  - `test_read_header_is_none_without_title_or_author` — both `None`; and GROBID title with no author from either → `None`
  - `test_researchgate_cover_page_yields_title_author_and_doi` — cover fixture → `title` starts with `"Pedagogical introduction to equilibrium Green's functions"`, `first_author == "Odashima"`, `doi == "10.1590/1806-9126-rbef-2016-0087"`
  - `test_title_label_wins_over_section_header` — items `section_header "Introduction to the problem space"`, `title "Half-metallic graphene nanoribbons"`, `text "Young-Woo Son, Marvin L. Cohen and Steven G. Louie"` → `title == "Half-metallic graphene nanoribbons"`, `first_author == "Son"`
  - `test_superscript_markers_are_ignored` — author text `"Young-Woo Son1*, Marvin L. Cohen1,2"` → `first_author == "Son"`
  - `test_trailing_marks_are_stripped_from_the_surname` — `"Ulrich Eckern∗"` → `"Eckern"`; `"Shantanu Mishra'+"` → `"Mishra"`
  - `test_bare_initial_is_not_a_surname` — author text `"Alice B"` → `first_author is None`
  - `test_venue_or_repository_banner_is_not_a_title` — `section_header "European Journal of Public Health Studies"`, then `section_header "Institutional Repository - Research Portal"`, then `section_header "Carrier density and magnetism in graphene zigzag nanoribbons"` → `title` is the last one
  - `test_affiliation_is_never_taken_as_the_author` — after the title only `text "Department of Physics, University of California, Berkeley"` and `text "Abstract"` → `first_author is None`
  - `test_generic_section_header_is_not_a_title` — only `section_header "Abstract"` and `section_header "Introduction"` → `title is None and first_author is None`
  - `test_only_page_one_is_read` — the author line is on `page_no == 2` → `first_author is None`
  - `test_docling_header_without_cache_returns_none` — `config.DOCLING_CACHE_DIR` patched to an empty temp dir → `docling_header(pdf) is None` and no converter is created (patch `docling_extract.get_digital_converter` to raise if called)
  - `test_docling_header_with_corrupt_cache_returns_none` — cache file containing `{not json` → `None`
  - `test_docling_header_converts_two_pages_when_asked` — no cache, `convert_if_missing=True`; patched converter records the `page_range` argument and returns an object whose `.document.model_dump(mode="json")` is the cover fixture → `page_range == (1, 2)` and `first_author == "Odashima"`
- [ ] **Step 2: Run and confirm they fail** (`ImportError`).
- [ ] **Step 3: Implement** `paper_header.py` (import `normalise_arxiv` and `surname_key` from `paper_index`, `normalise_doi` and `fold_title` from `source_key`).
- [ ] **Step 4: Run and confirm they pass.**
- [ ] **Step 5: Commit** — `feat(dedupe): read title and first author via GROBID with Docling fallback`

### Task 3: Layer 1 in Agent 2

**Files:**
- Modify: `research_assistant/agents/agent2_fetcher.py` (`fetch_papers`, the per-reference loop after `doi, how = resolve_doi(ref)`; the success-record block that currently uses `path=dest`)
- Test: `tests/test_fetcher.py` (new class `TestDoiAlreadyInCorpus`; reuse the harness style of `TestAlreadyOnDisk`)

**Interfaces — Consumes:** `PaperIndex.load(path)`, `PaperIndex.find_by_ids` (Task 1).
Import `PAPER_INDEX_PATH` into `agent2_fetcher`'s namespace and call `PaperIndex.load(PAPER_INDEX_PATH)`
once before the loop, so tests patch `agent2_fetcher.PAPER_INDEX_PATH`.

Behaviour: a hit whose `path` exists on disk → `saved=True`, `provider="corpus"`, no provider call, and
the manifest record's `path` is the **hit's path** (introduce a `record_path` variable defaulting to
`dest`). The layer-1 check runs **before** the existing `is_saved_pdf(dest)` branch. A hit whose file is
missing is ignored.

- [ ] **Step 1: Write the failing tests** (`try_unpaywall`/`try_europepmc`/`try_arxiv` record calls; index file in the temp dir):
  - `test_doi_already_in_corpus_is_not_downloaded` — index entry `dois=("10.1038/nature05180",)`, `path=<tmp>/corpus/existing.pdf` (file exists, `%PDF-` bytes) → `calls == []`; record `provider == "corpus"`, `path == <that path>`
  - `test_arxiv_id_in_corpus_counts_as_a_match` — `resolve_doi` returns `(None, None)`, ref `arxiv_id="2007.12504v1"`, entry `arxiv_ids=("2007.12504",)` → `calls == []`
  - `test_corpus_entry_whose_file_is_gone_is_downloaded_normally` — entry path does not exist → `calls == ["unpaywall"]`
- [ ] **Step 2: Run and confirm they fail** for the right reason (`["unpaywall"] != []`).
- [ ] **Step 3: Implement** the check and `record_path`.
- [ ] **Step 4: Run** `tests/test_fetcher.py tests/test_download.py` — all green (the existing `TestAlreadyOnDisk` and `TestFetchPapersSuccessRecord` must still pass).
- [ ] **Step 5: Commit** — `feat(dedupe): skip reference download when DOI is already in corpus`

### Task 4: Layer 1 in Agent 0 (seed paper)

**Files:**
- Modify: `research_assistant/agents/agent0_discoverer.py` — `download_seed_cascade(ranked_candidates, force=False)` (check each candidate before its `os.path.exists(dest)` check) and `_download_and_record(query, key, pdf_url, seeds, force, arxiv_id=None, source="manual-url")` (check before downloading)
- Test: `tests/test_discoverer.py` (new class `TestSeedAlreadyInCorpus`)

**Interfaces — Consumes:** `PaperIndex.load`, `find_by_ids` (Task 1). Same `PAPER_INDEX_PATH` import pattern as Task 3.

Behaviour: unless `force`, a candidate whose `doi`/`arxiv_id` hits an index entry with an existing file
returns `(cand, key, hit.path)` with `cand["download_status"] = "in_corpus"` and no `download_pdf` call.
In `_download_and_record`, the DOI is `key[4:]` when `key.startswith("doi:")`, plus the `arxiv_id`
argument; on a hit the seed manifest records `hit.path` and `download_pdf` is not called.

- [ ] **Step 1: Write the failing tests** (patch `agent0_discoverer.download_pdf` to record calls; patch `RAW_DIR`, `SEED_PAPERS_PATH`, `PAPER_INDEX_PATH` into the temp dir):
  - `test_seed_candidate_in_corpus_is_not_downloaded` — candidate `{"doi": "10.1038/nature05180", "pdf_url": "https://x/p.pdf", "title": "..."}`, matching entry → returned path is the corpus file, `download_pdf` not called, `download_status == "in_corpus"`
  - `test_force_downloads_even_when_in_corpus` — same with `force=True` → `download_pdf` called once
  - `test_manual_url_seed_in_corpus_is_not_downloaded` — `_download_and_record("q", "doi:10.1038/nature05180", "https://x/p.pdf", {}, False)` returns the corpus path; `download_pdf` not called
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_discoverer.py tests/test_e2e_discovery.py` — all green.
- [ ] **Step 5: Commit** — `feat(dedupe): agent0 reuses a seed paper already in the corpus`

### Task 4b: Layer 1 for pasted DOIs and arXiv IDs

**Files:**
- Modify: `research_assistant/shared/batch_uploader.py` — `fetch_and_stage_urls_or_dois` (the arXiv-token branch and the DOI-token branch, currently ~L345–380)
- Test: add to the existing batch-uploader tests (`grep -l fetch_and_stage_urls_or_dois tests/`); create `tests/test_batch_uploader_dedupe.py` if none exists

**Interfaces — Consumes:** `PaperIndex.load`, `find_by_ids` (Task 1). Import `PAPER_INDEX_PATH` into the module namespace.

Behaviour: for an arXiv or DOI token whose ID hits an index entry with an existing file, set `data` to
that file's bytes instead of making any network call, then stage exactly as today (callers such as the
Citation auditor still receive a staged record). Plain URL tokens are unchanged. Leave the existing
`ok = agent2_fetcher.try_unpaywall(...)` tuple-truthiness behaviour as it is (owner decides separately).

- [ ] **Step 1: Write the failing tests** (patch `batch_uploader.requests.get` and `agent2_fetcher.try_unpaywall` / `try_europepmc` to record calls; staging into a temp dir):
  - `test_pasted_doi_in_corpus_is_staged_without_downloading` — token `"10.1038/nature05180"`, matching entry → no network call recorded, one staged record whose file bytes equal the corpus file's bytes
  - `test_pasted_arxiv_id_in_corpus_is_staged_without_downloading` — token `"arXiv:2007.12504v1"`, entry `arxiv_ids=("2007.12504",)` → `requests.get` not called, one staged record
  - `test_pasted_doi_not_in_corpus_is_downloaded_as_before` — empty index → `try_unpaywall` called once
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new tests plus every existing test file that mentions `batch_uploader` — green.
- [ ] **Step 5: Commit** — `feat(dedupe): reuse corpus copy for pasted DOIs and arXiv IDs`

### Task 5: Layer 2 at ingestion

**Files:**
- Modify: `research_assistant/shared/ingestion.py` — new helper; call it in `_ingest_pdfs_locked` after the processing loop and the `require_grobid` fallback filter, immediately before `result["inserted"] = upsert_corpus(corpus)` (currently ~L805–807), so Docling has already converted and cached each PDF and nothing has reached the store yet; add ingested papers to the index right after `manifest.add_many(pdf_key(path) for path in marked)` (currently ~L840)
- Test: `tests/test_ingest_dedupe.py` (new). For the integration test, follow the existing stubbing of `process_pdf` / the store in `tests/test_ingestion.py`.

**Interfaces — Consumes:** `read_header` (Task 2); `edit`, `PaperIndex.find_by_title_author`, `add_alias`, `add`, `repoint_manifest` (Task 1). Import `PAPER_INDEX_PATH`, `DOWNLOADED_JSON_PATH` into the module namespace.
**Produces:**
```python
def _drop_title_duplicates(candidates: dict[str, str], corpus: list[dict], log_prefix: str = "") \
        -> tuple[dict[str, str], list[dict], list[dict], dict[str, PaperHeader]]
# returns (kept candidates, kept corpus entries, duplicates, headers of kept paths)
# a dropped path's corpus entries are those with entry["document"] == pdf_key(path)
# duplicate record: {"path": str, "document": str, "kept": str, "deleted": bool}
```
`ingest_pdfs()`'s result dict gains `"duplicates": list[dict]` (empty list when none).

Behaviour per candidate: `read_header(path)`; `None` → keep. Match =
`find_by_title_author(header.title, header.first_authors, exclude=pdf_key(path))` whose `path` exists and is not the
same file. On a match: drop it from `candidates` (so it is never marked ingested) and drop its entries
from `corpus` (so nothing is upserted); delete the file **only** if its real path is inside
`PULLED_PDFS_DIR`; `add_alias(pdf_key(path), match.document)`;
`repoint_manifest(DOWNLOADED_JSON_PATH, path, match.path)`; one `pipeline_status.add_event` line naming
both papers. After `manifest.add_many`, add each marked path to the index with `document=pdf_key(path)`,
title and `first_authors` from its header, DOIs/arXiv IDs from the header plus any `downloaded.json` record whose
`path` basename matches.

- [ ] **Step 1: Write the failing tests:**
  - `test_title_author_duplicate_in_pulled_pdfs_is_dropped_and_deleted` — index has paper P (file exists); a new PDF in a patched `PULLED_PDFS_DIR` whose stubbed header matches P → not in kept, file gone, `deleted is True`
  - `test_duplicate_outside_pulled_pdfs_is_dropped_but_kept_on_disk` — same but the new PDF is in a patched `RAW_DIR` → not in kept, file still exists, `deleted is False`
  - `test_unreadable_header_keeps_the_pdf` — `read_header` returns `None` → kept, file exists
  - `test_match_whose_file_is_gone_does_not_drop` — P's path missing → kept
  - `test_duplicate_is_aliased_and_downloaded_json_repointed` — after the drop, `resolve(<new key>).document == P.document` and the `downloaded.json` record that pointed at the new file now points at `P.path`
  - `test_ingested_pdf_is_added_to_the_index` — through `ingest_pdfs` with processing stubbed: a non-duplicate PDF ends up in the index with its header's title and `first_authors`
  - `test_duplicate_never_reaches_the_store` — through `ingest_pdfs` with processing stubbed to return entries for a duplicate and a non-duplicate: the recorded `upsert_corpus` argument contains only the non-duplicate's `document`, and the duplicate is not in the ingest manifest
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `tests/test_ingest_dedupe.py tests/test_ingestion.py tests/test_ingest_v2.py tests/test_p0_remediation.py` — all green.
- [ ] **Step 5: Commit** — `feat(dedupe): drop title+author duplicates at ingestion`

### Task 6: End-of-run sweep

**Files:**
- Create: `research_assistant/shared/dedupe_sweep.py`
- Modify: `research_assistant/shared/ingestion.py` (`ingest_pdfs`: run the sweep after `_ingest_pdfs_locked` returns, outside the lock)
- Modify: `research_assistant/agents/agent2_fetcher.py` (`fetch_papers`: run the sweep after the final `_checkpoint`)
- Test: `tests/test_dedupe_sweep.py`

**Interfaces — Produces:**
```python
def run_sweep(index_path: str | None = None, report_path: str | None = None) -> dict
# defaults: config.PAPER_INDEX_PATH / config.DEDUPE_REPORT_PATH (resolved at call time)
# report: {"generated_at": ISO-8601 UTC str, "papers": int, "groups": PaperIndex.duplicate_groups()}
# writes report atomically; never raises (logs and returns {"error": str, "groups": []} on failure);
# deletes nothing; adds one pipeline_status event when groups is non-empty
```
Both hooks call `run_sweep()` inside `try/except Exception` so a sweep problem can never fail a run.

- [ ] **Step 1: Write the failing tests:**
  - `test_sweep_reports_papers_sharing_a_doi` — two entries share a DOI → one group, `reason == "doi"`, both documents
  - `test_clean_index_reports_no_groups` — `groups == []`, report file written
  - `test_unreadable_index_does_not_raise` — garbage index file → returns with `groups == []`
  - `test_sweep_deletes_nothing` — files of grouped entries still exist afterwards
  - `test_ingest_pdfs_runs_the_sweep` — patch `ingestion.run_sweep` (module namespace) with a recorder; `ingest_pdfs([])` → called once
  - `test_fetch_papers_runs_the_sweep` — same for `agent2_fetcher.run_sweep` with an empty reference list
- [ ] **Step 2: Run and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new file plus `tests/test_fetcher.py tests/test_ingestion.py` — green.
- [ ] **Step 5: Commit** — `feat(dedupe): end-of-run duplicate report`

### Task 7: Backfill, docs, full verification

**Files:**
- Create: `scripts/build_paper_index.py`
- Modify: `PIPELINE.md` (one paragraph under the Agent 2 row: the two layers, the sweep, where files go)
- Test: `tests/test_build_paper_index.py`

**Interfaces — Produces:** CLI
`python scripts/build_paper_index.py [--dry-run] [--limit N] [--extra-manifest PATH ...]`.
Scans `PULLED_PDFS_DIR` and `RAW_DIR` for `*.pdf` (skip any path containing `quarantine` or `_archive`).
Resumable: documents already in the index are not re-read. Per new PDF:
`read_header(path, convert_if_missing=True)` (GROBID header, merged with the Docling cache when present
or a pages-1–2 Docling conversion otherwise), plus DOIs/arXiv IDs
from `DOWNLOADED_JSON_PATH` and every `--extra-manifest` (match records on `path` basename). Saves every
25 papers. Ends by calling `run_sweep()` and printing papers indexed, headers unreadable, groups found.
`--dry-run` prints counts and writes nothing.

- [x] **Step 1: Write the failing tests:**
  - `test_backfill_skips_papers_already_indexed` — index already holds A; dir holds A and B → `read_header` called only for B
  - `test_backfill_takes_dois_from_manifests` — B's `downloaded.json` record has `doi` → entry `dois == ("10.1038/nature05180",)`
  - `test_dry_run_writes_nothing` — no index file afterwards
- [x] **Step 2: Run and confirm they fail.**
- [x] **Step 3: Implement the script and the `PIPELINE.md` paragraph.**
- [x] **Step 4: Full suite** — `CITATION_LOG_FILE=0 $PY -m pytest tests/ -q -p no:cacheprovider` → `0 failed`; passed = 2238 + your new tests; skipped still 18.
- [x] **Step 5: Commit** — `feat(dedupe): backfill script and docs`
- [x] **Step 6: Real backfill (only after the owner says go).**
  `$PY scripts/build_paper_index.py --dry-run --extra-manifest ../downloaded.json`, then
  `--limit 20` and record seconds per paper in the log, then the full run (safe to stop and resume).
  About 1,800 PDFs; 1,522 already have a Docling cache, and GROBID takes ≈1.2 s each, so expect roughly
  40 minutes. GROBID must answer `curl -s localhost:8070/api/isalive` before you start; if it goes down
  mid-run, stop rather than backfill Docling-only. Paste the final summary lines — papers indexed,
  headers with no author, duplicate groups — into the log below.

---

## Review Round 1 (reviewer, 2026-11-06, against `10f6595`)

Tasks 1–7 match the plan and are well tested. Reviewer's full-suite run: 2297 passed, 18 skipped,
1 failed (`test_batch_citer.py::…test_the_contextualized_query_reaches_the_seam_when_on`). That test
passes alone and in its own file, and the branch touches none of its code, so it looks order-dependent,
not caused by this work; if it fails in your full runs too, log it by name. Four defects found — R1 is a
planner error (Task 4 as written), not an implementation slip. Fix them in order, same rules as above
(test first, watch it fail, one commit per fix, `fix(dedupe): <what>`, log each in the Handoff Log).
**Do not start Task 7 Step 6 (the real backfill) until R1–R4 are committed and the owner says go.**

### R1 (blocking): a seed found in the corpus must still land in `RAW_DIR`

**Why:** `orchestrate.py` runs `discover → ingest_seed → extract`, and `agent1_extractor.run_extractor()`
only globs `RAW_DIR/*.pdf`. Task 4 returns the corpus file's path (usually in `pulled_pdfs/` or
`raw/processed/`), so Agent 1 finds nothing and the run stops with "No references could be extracted".

**Files:** `research_assistant/agents/agent0_discoverer.py` (`download_seed_cascade`, `_download_and_record`);
`tests/test_discoverer.py` (`TestSeedAlreadyInCorpus`).

**Behaviour:** on a corpus hit (not `force`), copy the corpus file to `dest = RAW_DIR/filename_for(key)`
with `shutil.copyfile` unless `dest` already exists (a local copy, no network), keep
`download_status = "in_corpus"`, and return / record **`dest`**, not `hit.path`. Layer 2 then skips the copy
at `ingest_seed` (it lives in `RAW_DIR`, so it is kept), and Agent 1 mines and files it as usual.

- [x] Change the three existing `TestSeedAlreadyInCorpus` tests to expect `dest` (in `RAW_DIR`, bytes equal
  the corpus file's) and still `download_pdf` not called; add
  `test_corpus_seed_is_visible_to_the_extractor` — after `download_seed_cascade` on a hit,
  `glob(RAW_DIR/*.pdf) == [dest]`.
- [x] Red → implement → green (`tests/test_discoverer.py tests/test_e2e_discovery.py tests/test_orchestrate_upload.py`) → commit `fix(dedupe): copy a corpus seed into raw/ so Agent 1 can read it`.

### R2: keep index paths valid when Agent 1 files a seed away

**Why:** the seed is indexed at `ingest_seed` under `RAW_DIR/<name>.pdf`; Agent 1's `_file_away` then
moves it to `raw/processed/`. The entry's path is dead, so both layers ignore it ("file gone → no
match") and a later run re-downloads and re-ingests every earlier seed — the most common duplicate.

**Files:** `research_assistant/shared/paper_index.py`; `research_assistant/agents/agent1_extractor.py`
(`_file_away`); `research_assistant/shared/ingestion.py` (`_drop_title_duplicates`);
tests in `tests/test_paper_index.py`, `tests/test_extractor.py`, `tests/test_ingest_dedupe.py`.

**Produces:** `PaperIndex.repoint_path(old_path: str, new_path: str) -> int` — every entry whose
`os.path.abspath(path) == os.path.abspath(old_path)` gets `path=new_path` (document unchanged); returns count.

- [x] Tests: `test_repoint_path_updates_matching_entry_only` (paper_index);
  `test_filing_a_pdf_repoints_its_index_entry` (extractor: index entry at the RAW_DIR path → after
  `_file_away` its path is the file's new location under `processed/`); `PAPER_INDEX_PATH` patched to a temp dir.
- [x] Implement: in `_file_away`, capture the destination from `_unique_destination`, and after a
  successful `shutil.move` run `with edit(PAPER_INDEX_PATH) as idx: idx.repoint_path(pdf_path, destination)`
  inside `try/except Exception` (log at warning, never fail extraction).
- [x] Because the index lock now has a second user, move the `read_header` calls in
  `_drop_title_duplicates` **before** `with edit(...)`: read every candidate's header first, then take
  the lock only for matching, aliasing and repointing. Existing `test_ingest_dedupe.py` must stay green.
- [x] Red → green (`tests/test_paper_index.py tests/test_extractor.py tests/test_ingest_dedupe.py tests/test_pipeline_status.py`) → commit `fix(dedupe): repoint index entry when Agent 1 files a seed`.

### R3: Docling surnames lose accented edge letters

**Why:** `header_from_docling` strips the surname with `^[^a-zA-Z]+|[^a-zA-Z]+$`, which treats
accented letters as non-letters: `"Rasa Kulytė"` → `"Kulyt"`, `"Øystein Ødegård"` → `"degård"` (verified).
GROBID reads `"Kulytė"`, so the same paper read Docling-only (GROBID down) never matches. The plan
said "non-letter characters"; Unicode letters are letters.

**Files:** `research_assistant/shared/paper_header.py`; `tests/test_paper_header.py`.

- [x] Tests: `test_accented_surnames_keep_their_letters` — author text `"Rasa Kulytė, Jonas Petras"` →
  `first_author == "Kulytė"`; `"Øystein Ødegård and Kari Nord"` → `"Ødegård"`. Existing
  `"Shantanu Mishra'+"` → `"Mishra"` and bare-initial tests must stay green.
- [x] Implement: strip with `^[\W\d_]+|[\W\d_]+$`; the "at least 2 letters" check counts
  `sum(c.isalpha() for c in surname)`.
- [x] Red → green → commit `fix(dedupe): keep accented letters in Docling surnames`.

### R4: the backfill must not overwrite entries written while it runs, and must not run without GROBID

**Why:** `build_index` holds one in-memory index for ~40 minutes and saves it whole every 25 papers.
The `edit()` lock is per-process, so any entry the running app adds meanwhile is overwritten. And if
GROBID dies mid-run, the remaining papers are indexed Docling-only and, being "already indexed",
are never re-read.

**Files:** `scripts/build_paper_index.py`; `tests/test_build_paper_index.py`.

- [x] Tests: `test_backfill_keeps_entries_written_meanwhile` — stub `read_header` so its first call also
  writes an extra entry X to the index file via `edit()`; after `build_index`, the reloaded index has X
  plus every backfilled paper. `test_backfill_refuses_to_start_without_grobid` — GROBID probe stubbed
  `False` → returns `{"error": ...}`, no index file written (`--dry-run` still works without GROBID).
  `test_backfill_stops_cleanly_when_grobid_goes_down` — probe `True` at start, `False` at the first
  checkpoint → papers read so far are saved, the run stops with `{"error": ..., "indexed": n}`.
- [x] Implement: collect new entries in a batch and flush every 25 (and at the end) with
  `with edit(target_index) as fresh: for e in batch: fresh.add(e)` instead of `idx.save()`; probe
  `f"{config.GROBID_SERVER}/api/isalive"` (10 s timeout, `"true"` in body) before starting and at each
  flush; on failure flush what was read and stop with a "GROBID is down — rerun to resume" message.
- [x] Red → green → then the full suite (expect 2298 + new tests passed, 18 skipped) → commit
  `fix(dedupe): backfill merges into the live index and stops without GROBID`.

### Task 7 Step 6 — additions for the real backfill

- Stop the running Marvin app first (even with R4 the two processes have no shared lock), and start it
  again only after the backfill finishes.
- Run with `$PY scripts/build_paper_index.py` from the repo root (commit `10f6595` made this work).

### Noted, no change requested

- Two brand-new copies of one paper in the **same** ingest batch are both ingested (neither is in the
  index yet); the end-of-run sweep reports them. Acceptable per the design (report only).
- Commit `10f6595` is not in the Handoff Log; add a line for it with the R1 entry.

## Review Round 2 (reviewer, 2026-11-06, against `c4bfe5e`)

R1–R4 accepted: each matches its spec, with tests that would fail without the fix. Reviewer's full
suite: 2304 passed, 18 skipped, 1 failed. The failure is again in
`test_batch_citer.py::TestRunBatchCiterPassesContextAndQuery`, which is a pre-existing, unrelated race:
`run_batch_citer` processes sentences in a `ThreadPoolExecutor` with `as_completed`, but the test reads
`seen[1]` as if it were the second sentence. Not this branch's concern; do not fix it here.

Small notes, no code change: the R4 Handoff Log line says `pending` where the sha is `c4bfe5e` (fix it
when you append the backfill line). If GROBID dies mid-backfill, up to 25 papers (one batch) are
saved Docling-only before the run stops; list their documents in the log if that happens.

**Cleared for Task 7 Step 6 once the owner says go.**

## Review Round 3 (reviewer, 2026-11-06, against `main` at `e81fd21`, after the merge)

The branch was merged to `main` and the backfill is done (1,818 papers, 97.4% with title + author;
all 46 title_author groups verified as real duplicates). Two follow-ups, found by checking the real data.
Work on a new branch **`fix/dedupe-round3` from `main`**, same rules as before (test first, watch it
fail, one commit per fix, `fix(dedupe): <what>`, Handoff Log line per fix, no push, no merge).

**Facts the reviewer measured (read-only, 2026-11-06):** the LanceDB store holds **1,781** distinct
`document`s. The ingest manifest `data/ingested_v3.json` lists only **186**, all of them also in the
store (manifest-only: 0; store-only: 1,595 — papers from before the migration). Of the 1,818 papers in
`paper_index.json`, **1,537 are in the store and 281 are not** (on disk, never ingested).

### R5: layer 2 must only count a match that is actually in the store

**Why:** `_drop_title_duplicates` treats any index entry with an existing file as "already in the
corpus". If that entry is one of the 281 on-disk-but-never-ingested PDFs, the new copy is dropped (and
deleted if in `pulled_pdfs/`) while neither copy is in the store, so that run's audit has no evidence
for the paper. (The next run ingests the old copy via the repointed `downloaded.json` record, so
nothing is lost for good, but it is a one-run gap.)

**Files:** `research_assistant/shared/ingestion.py` (`_drop_title_duplicates`);
`tests/test_ingest_dedupe.py`.

**Behaviour:** a match counts only when its file exists, it is not the same file, **and
`manifest.contains(match.document)`** is true. Otherwise the candidate is kept and ingested as today.
Layer 1 needs no change: a layer-1 hit is recorded in `downloaded.json` (or copied / staged), so the
same run ingests it.

- [ ] Tests: `test_match_that_was_never_ingested_does_not_drop` — P in the index with an existing file
  but **not** in the manifest; new PDF in `PULLED_PDFS_DIR` with a matching header → kept, file still
  exists, no alias, `downloaded.json` unchanged. Update the existing drop tests
  (`test_title_author_duplicate_in_pulled_pdfs_is_dropped_and_deleted`,
  `test_duplicate_outside_pulled_pdfs_is_dropped_but_kept_on_disk`,
  `test_duplicate_is_aliased_and_downloaded_json_repointed`, `test_duplicate_never_reaches_the_store`)
  to record P's document with `manifest.add(...)` in their setup (the manifest lives under the test
  `DATA_DIR`; never touch the real one).
- [ ] Red → implement → green (`tests/test_ingest_dedupe.py tests/test_ingestion.py tests/test_ingest_v2.py tests/test_p0_remediation.py`) → commit `fix(dedupe): only drop a duplicate of a paper that is in the store`.

### R6: ignore template placeholder DOIs

**Why:** Six DOIs in the index are template placeholders copied from paper first pages:
`10.1007/sxxxxx-yyy-zzzz-1`, `10.1109/xxx.0000.0000000`, `10.1145/nnnnnnn.nnnnnnn`, `10.1145/xxxxxx`,
`10.1364/ao.xx.xxxxxx`, `10.15607/rss.2024.xx.015`. `10.1145/nnnnnnn.nnnnnnn` alone groups 5 unrelated
ACM papers in the report. They can't block a real download, but they pollute the report and the index.
The rule below matches exactly these 6 and none of the other ~1,000 DOIs in the index.

**Files:** `research_assistant/shared/paper_index.py`; `tests/test_paper_index.py`.

**Produces:** `is_placeholder_doi(doi: str | None) -> bool` in `paper_index.py`: true when
`normalise_doi(doi)` is not None and the part after the first `/` matches `re.search(r"xx|nnnn|yyy|zzzz", suffix)`.
Applied at three points so old index entries are covered without rewriting the file:
`PaperIndex.add` stores the entry with placeholder DOIs removed (`dataclasses.replace`);
`find_by_ids` returns `None` for a placeholder DOI query (an arXiv ID in the same call still counts);
`duplicate_groups` skips placeholder keys when building DOI groups.

- [ ] Tests: `test_placeholder_dois_are_recognised` — all 6 DOIs above → `True`; `10.1038/nature05180`,
  `10.1103/physrevx.10.011001`, `10.1145/3586183.3606763`, `None`, `""` → `False`.
  `test_add_drops_placeholder_dois` — `add(entry with dois=("10.1145/xxxxxx", "10.1038/nature05180"))`
  → `get(...).dois == ("10.1038/nature05180",)`. `test_find_by_ids_ignores_placeholder_doi`.
  `test_duplicate_groups_skip_placeholder_doi` — two entries loaded from disk (bypassing `add`) sharing
  `10.1145/nnnnnnn.nnnnnnn` → no `doi` group.
- [ ] Red → green (`tests/test_paper_index.py tests/test_dedupe_sweep.py tests/test_build_paper_index.py`) → full suite (the known
  `test_batch_citer.py` ordering race may fail; log it by name if so) → commit `fix(dedupe): ignore template placeholder DOIs`.

### R7: repair the ingest manifest (real data — only after the owner says go)

**Why:** R5 relies on the manifest, which lists 186 of the 1,781 papers in the store. Without this
repair layer 2 would almost never fire. `manifest.rebuild_from_collection()` already exists for exactly
this (it rewrites the manifest from the store's `document` metadata). Nothing in the manifest is
missing from the store, so no entry is lost. Side effect (wanted): ingestion stops re-processing the
1,595 pre-migration papers if they come through a run again.

- [ ] Stop and ask the owner for "go". The owner stops the Marvin app first.
- [ ] Copy `data/ingested_v3.json` to `data/ingested_v3.json.bak-2026-11-06` (keep it).
- [ ] `$PY -c "from research_assistant.shared import manifest; print(manifest.rebuild_from_collection())"`
  → expect 1,781 (± papers ingested since 2026-11-06 15:24).
- [ ] `$PY -c "from research_assistant.shared.dedupe_sweep import run_sweep; r = run_sweep(); print(r['papers'], len(r['groups']))"`
  → papers 1,818; groups below 100 (the placeholder group is gone).
- [ ] Log both numbers in the Handoff Log, commit the log (`docs(dedupe): record manifest repair`), stop and report.

## Handoff Log (append-only)

Append one entry per finished task and for every stop/discrepancy:
`YYYY-MM-DD HH:MM · Task N · <commit sha> · <tests run and result> · <anything that differed from the plan>`.

The reviewer (Claude Code session) diffs your branch against the baseline task by task. Do not squash or
amend commits.

2026-11-06 12:21 · Task 1 · 468a9b9 · tests/test_paper_index.py (15 passed), tests/test_fetcher.py + test_download.py (49 passed) → all green · none
2026-11-06 12:24 · Task 2 · cd9a4be · tests/test_paper_header.py (20 passed) → all green · none
2026-11-06 13:03 · Task 3 · 0ecc1de · tests/test_fetcher.py + test_download.py (52 passed) → all green · none
2026-11-06 13:06 · Task 4 · 9a5638c · tests/test_discoverer.py + test_e2e_discovery.py (107 passed) → all green · none
2026-11-06 13:09 · Task 4b · 4bab01d · tests/test_batch_uploader_dedupe.py + 7 existing batch_uploader test files (126 passed) → all green · none
2026-11-06 13:13 · Task 5 · 029c38d · tests/test_ingest_dedupe.py + 3 ingestion test files (68 passed) → all green · none
2026-11-06 13:15 · Task 6 · a356528 · tests/test_dedupe_sweep.py + test_fetcher.py + test_ingestion.py (69 passed) → all green · none
2026-11-06 13:21 · Task 7 · 2cdf2d1 · full suite (2298 passed, 18 skipped) → all green · none
2026-11-06 13:22 · Task 7 fix · 10f6595 · scripts/build_paper_index.py dry-run passed (1818 PDFs) · added package root to sys.path
2026-11-06 14:12 · R1 · bc62b50 · tests/test_discoverer.py + test_e2e_discovery.py + test_orchestrate_upload.py (120 passed) → all green · none
2026-11-06 14:15 · R2 · 40a4a9b · tests/test_paper_index.py + test_extractor.py + test_ingest_dedupe.py + test_pipeline_status.py (113 passed) → all green · none
2026-11-06 14:16 · R3 · 2c12fa5 · tests/test_paper_header.py (21 passed) → all green · none
2026-11-06 14:22 · R4 · c4bfe5e · full suite (2305 passed, 18 skipped) → all green · none
2026-11-06 15:21 · Task 7 Step 6 · (this commit) · backfill completed: 1818 papers indexed, 48 headers unreadable (2.64%), 100 duplicate groups found (doi: 28, arxiv: 26, title_author: 46), avg 1.69s/paper · none


