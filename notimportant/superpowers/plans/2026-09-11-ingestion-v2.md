# Ingestion v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v2 index — GROBID-first extraction, sentence-window chunks, GROBID-sourced figure/table captions with optional inline VLM description chosen once per run, NFKC/stemmed BM25, nomic task prefixes — under `CITATION_INDEX_VERSION=2`, leaving the v1 index untouched.

**Architecture:** Three new pure modules (`shared/tokenize.py`, `shared/extract.py`, `shared/chunking.py`) and two thin I/O modules (`shared/figures.py`, `shared/ingest_v2.py`) produce corpus entries in the same `list[dict]` shape `upsert_corpus()` already consumes, so the existing lock / dedupe / id / manifest / BM25 bookkeeping is reused rather than forked. `config.py` derives every collection, pickle and manifest name from `CITATION_INDEX_VERSION`, so `db.py`, `retrieve.py`, `manifest.py` and the agents follow the version without being edited. `process_pdf()` dispatches on the version; a `describe_figures` flag is threaded from the CLI / orchestrator / UI down to `process_pdf_v2()`.

**Tech Stack:** Python 3.10–3.12, pytest (collecting `unittest.TestCase` classes), GROBID 0.8.1 (`processFulltextDocument`), `lxml`, PyMuPDF (`fitz`), `pysbd`, `snowballstemmer`, ChromaDB, `rank_bm25`, `shared.llm` (Ollama `gemma4:e2b` for descriptions, `nomic-embed-text` for embeddings).

**Spec:** `notimportant/superpowers/specs/2026-09-11-rag-engine-v2-design.md` — §3.5, §3.6, §4 (all of it, §4.6 in particular), §6, §8, §9.

## Global Constraints

- **Python 3.10–3.12.** `pyproject.toml` sets `requires-python = ">=3.10,<3.13"`. No 3.13-only syntax.
- **Working interpreter for this checkout is the `ml` conda env:** `/home/shardul/miniconda3/envs/ml/bin/python` (it has chromadb, fitz, lxml, ollama). Every `python -m pytest` below means that interpreter. GROBID is expected at `http://localhost:8070` and Ollama at `localhost:11434` for the final task only.
- **v1 is read-only to v2 code.** Nothing may open `physics_papers`, `physics_summaries`, `bm25_index.pkl` or `ingested.json` for writing when `CITATION_INDEX_VERSION=2`. `data/` is never wiped. (Spec §4.8)
- **No provider SDK outside `shared/llm.py`.** No module may `import openai` or `import ollama`. (Existing rule, spec §4.6 routes descriptions through `shared.llm.chat`.)
- **Heavy imports stay inside functions.** `chromadb`, `fitz`, `requests` to GROBID, `pysbd` are imported lazily; `lxml` and `snowballstemmer` are pure/pinned and in `requirements-test.txt`, so they may be imported at module scope. CI's import sweep runs with only `requirements-test.txt` installed. (Spec §9)
- **Nothing writes outside `CITATION_DATA_DIR`.** TEI cache goes to `RAW_DIR/grobid_output/`, crops to `IMAGES_DIR`. (Existing rule)
- **All file writes go through `shared/atomic.py`** for manifests and the BM25 pickle.
- **Tests use `unittest.TestCase` classes**, matching every file in `tests/`.
- **The chunk-id invariant stands:** Chroma ids are `chunk_N`, contiguous, never deleted. (`ARCHITECTURE.md §4.4`, spec §4.5)
- **Figure analysis is one switch per run, never per image.** (Spec §4.6)
- Run the suite with `CITATION_LOG_FILE=0 python -m pytest tests/ -v` so it does not write into `data/logs/`.
- **Commit after every task** with the message given; do not push.

---

### Task 1: `config.py` — index version and v2 settings

**Files:**
- Modify: `research_assistant/config.py:69-73` (vector DB block), `:97` (manifest path), `:122-127` (ingestion tunables), `:207-215` (GROBID block)
- Test: `tests/test_config_version.py`

**Interfaces:**
- Produces: `config.INDEX_VERSION: int`; `config.COLLECTION_NAME`, `config.SUMMARY_COLLECTION_NAME`, `config.BM25_INDEX_PATH`, `config.INGESTED_MANIFEST_PATH` now derived from the version; `config.CHUNK_TARGET_CHARS: int`, `config.CHUNK_MAX_CHARS: int`, `config.CHUNK_MIN_CHARS: int = 200`, `config.CHUNK_MIN_ALPHA: float = 0.6`; `config.GROBID_FULLTEXT_TIMEOUT: int`; `config.GROBID_TEI_DIR: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_version.py
"""CITATION_INDEX_VERSION selects every on-disk name the index uses.

Checked in a subprocess: config.py is imported by most of the package, so
reloading it in-process would change module attributes other tests already
bound. A fresh interpreter per case is the honest way to read it.
"""

import json
import os
import subprocess
import sys
import unittest

_PROBE = """
import json
from research_assistant import config as c
print(json.dumps({
    "v": c.INDEX_VERSION,
    "chunks": c.COLLECTION_NAME,
    "summaries": c.SUMMARY_COLLECTION_NAME,
    "bm25": c.BM25_INDEX_PATH.rsplit("/", 1)[-1],
    "manifest": c.INGESTED_MANIFEST_PATH.rsplit("/", 1)[-1],
    "target": c.CHUNK_TARGET_CHARS,
    "max": c.CHUNK_MAX_CHARS,
    "tei_dir": c.GROBID_TEI_DIR.rsplit("/", 1)[-1],
}))
"""


def _probe(**env):
    # Start from a copy of the environment with every CITATION_* knob these
    # tests set removed, so the outer shell cannot leak a value into the
    # "default" case; then apply this case's values.
    full = {k: v for k, v in os.environ.items()
            if k not in ("CITATION_INDEX_VERSION", "CITATION_CHUNK_TARGET_CHARS", "CITATION_CHUNK_MAX_CHARS")}
    full.update({"CITATION_LOG_FILE": "0", **env})
    out = subprocess.run([sys.executable, "-c", _PROBE], env=full, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestIndexVersionNames(unittest.TestCase):
    def test_default_is_v1_with_the_existing_names(self):
        got = _probe()
        self.assertEqual(got["v"], 1)
        self.assertEqual(got["chunks"], "physics_papers")
        self.assertEqual(got["summaries"], "physics_summaries")
        self.assertEqual(got["bm25"], "bm25_index.pkl")
        self.assertEqual(got["manifest"], "ingested.json")

    def test_v2_suffixes_every_name(self):
        got = _probe(CITATION_INDEX_VERSION="2")
        self.assertEqual(got["v"], 2)
        self.assertEqual(got["chunks"], "physics_papers_v2")
        self.assertEqual(got["summaries"], "physics_summaries_v2")
        self.assertEqual(got["bm25"], "bm25_index_v2.pkl")
        self.assertEqual(got["manifest"], "ingested_v2.json")

    def test_chunk_sizes_have_spec_defaults_and_are_overridable(self):
        got = _probe()
        self.assertEqual((got["target"], got["max"]), (1200, 1800))
        got = _probe(CITATION_CHUNK_TARGET_CHARS="900", CITATION_CHUNK_MAX_CHARS="1400")
        self.assertEqual((got["target"], got["max"]), (900, 1400))

    def test_tei_cache_dir_is_the_one_agent1_uses(self):
        self.assertEqual(_probe()["tei_dir"], "grobid_output")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_config_version.py -v`
Expected: FAIL — `AttributeError: module 'research_assistant.config' has no attribute 'INDEX_VERSION'` (from the subprocess, surfaced as `CalledProcessError`).

- [ ] **Step 3: Edit `config.py`**

Replace lines 69–73 (the `# ─── Vector Database` block) with:

```python
# ─── Index version ───────────────────────────────────────────────────────────
# Every on-disk name the index uses is derived from this, so a v2 index (new
# extractor, new chunker, prefixed embeddings) lives beside v1 and never
# opens a v1 file for writing. Default 1: nothing changes until set.
INDEX_VERSION = _env_int("CITATION_INDEX_VERSION", 1)
_INDEX_SUFFIX = "" if INDEX_VERSION == 1 else f"_v{INDEX_VERSION}"

# ─── Vector Database ──────────────────────────────────────────────────────────
VECTORDB_PATH    = os.path.join(DATA_DIR, "physics_vectordb")
COLLECTION_NAME  = f"physics_papers{_INDEX_SUFFIX}"       # detail chunks (stage-2 retrieval)
SUMMARY_COLLECTION_NAME = f"physics_summaries{_INDEX_SUFFIX}"   # one summary per document (stage 1)
BM25_INDEX_PATH  = os.path.join(DATA_DIR, f"bm25_index{_INDEX_SUFFIX}.pkl")
```

Replace line 97 (`INGESTED_MANIFEST_PATH = ...`) with:

```python
INGESTED_MANIFEST_PATH   = os.path.join(DATA_DIR, f"ingested{_INDEX_SUFFIX}.json")
```

After line 127 (`SEMANTIC_CHUNKER_AMOUNT  = 90`) add:

```python

# v2 chunker (INDEX_VERSION >= 2): sentence windows packed within a section.
# ~1,200 chars is ~300 tokens; six of them fill ~1,800 tokens of a 4k window.
CHUNK_TARGET_CHARS       = _env_int("CITATION_CHUNK_TARGET_CHARS", 1200)
CHUNK_MAX_CHARS          = _env_int("CITATION_CHUNK_MAX_CHARS", 1800)
CHUNK_MIN_CHARS          = 200      # shorter windows are dropped (captions exempt)
CHUNK_MIN_ALPHA          = 0.6      # alphabetic ratio below this is font-map garbage
```

After line 215 (`GROBID_START_COMMAND = ...`) add:

```python
# v2 ingestion sends every PDF to processFulltextDocument. One TEI per PDF is
# cached here — the same directory Agent 1 writes, so a seed paper it has
# already processed is never sent twice.
GROBID_TEI_DIR           = os.path.join(RAW_DIR, "grobid_output")
GROBID_FULLTEXT_TIMEOUT  = _env_int("GROBID_FULLTEXT_TIMEOUT", 300)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_config_version.py tests/test_manifest.py -v`
Expected: PASS (manifest tests still pass because the default path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add research_assistant/config.py tests/test_config_version.py
git commit -m "feat(config): CITATION_INDEX_VERSION derives collection, pickle and manifest names; v2 chunk and GROBID settings"
```

---

### Task 2: `shared/tokenize.py` — NFKC, stopwords, stemming

**Files:**
- Create: `research_assistant/shared/tokenize.py`
- Modify: `requirements.txt` (Agent 3 block), `requirements-test.txt`
- Test: `tests/test_tokenize.py`

**Interfaces:**
- Produces: `tokenize.normalize(text: str) -> str` (NFKC + whitespace collapse); `tokenize.tokenize(text: str) -> list[str]`; `tokenize.TOKENIZER_VERSION = "v2"`; `tokenize.legacy_tokenize(text) -> list[str]` (the old `\w+` behaviour, for v1 pickles).

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, under `# ─── Agent 3 — Ingestor (text-only path) + retrieval`, after `langchain-experimental==0.4.1  # SemanticChunker` add:

```
snowballstemmer==3.1.1         # BM25 tokenizer stemming (v2)
pysbd==0.3.4                   # v2 chunker sentence splitter
```

In `requirements-test.txt`, after `watchdog~=6.0` add:

```
# snowballstemmer backs shared/tokenize.py, which shared/search.py imports at
# module scope; pysbd backs shared/chunking.py. Both are pure Python.
snowballstemmer~=3.1
pysbd~=0.3
```

Then install into the working env: `python -m pip install snowballstemmer==3.1.1 pysbd==0.3.4`

- [ ] **Step 2: Write the failing test**

```python
# tests/test_tokenize.py
"""The BM25 tokenizer must see the same word the reader sees.

31% of the v1 index's chunks contain a ligature-broken word (`ﬁeld`) that
`\\w+` indexes as a different token from `field`; 23% contain `quan- tum`.
NFKC fixes the first at rebuild time. Stemming makes `fluctuations` match
`fluctuation`. The same function must run on queries, or none of it helps.
"""

import unittest

from research_assistant.shared import tokenize as tk


class TestNormalize(unittest.TestCase):
    def test_ligatures_become_ascii(self):
        self.assertEqual(tk.normalize("ﬁeld eﬀect diﬀusion"), "field effect diffusion")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(tk.normalize("a  b\n\nc\t d"), "a b c d")


class TestTokenize(unittest.TestCase):
    def test_lowercases_stems_and_drops_stopwords(self):
        self.assertEqual(tk.tokenize("The Fluctuations of the fields"), ["fluctuat", "field"])

    def test_ligature_query_and_document_tokenise_identically(self):
        self.assertEqual(tk.tokenize("ﬁnite-size eﬀects"), tk.tokenize("finite-size effects"))

    def test_single_characters_and_digits_alone_are_dropped(self):
        # "t" and "1" carry nothing for keyword search; "MoS2" survives as one token.
        self.assertEqual(tk.tokenize("t = 1 for MoS2"), ["mos2"])

    def test_version_tag(self):
        self.assertEqual(tk.TOKENIZER_VERSION, "v2")

    def test_legacy_matches_the_old_behaviour(self):
        self.assertEqual(tk.legacy_tokenize("The ﬁeld"), ["the", "ﬁeld"])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_tokenize.py -v`
Expected: FAIL — `ImportError: cannot import name 'tokenize'`.

- [ ] **Step 4: Write the module**

```python
# research_assistant/shared/tokenize.py
"""One tokenizer for BM25, applied to documents at rebuild and to queries at
search time.

Why a version tag: the pickle records which tokenizer built it, and
hybrid_search picks the matching query tokenizer. A v1 pickle (bare
BM25Okapi, built with `\\w+`) keeps working with legacy_tokenize(); anything
rebuilt from now on uses tokenize().
"""

import re
import unicodedata

import snowballstemmer

TOKENIZER_VERSION = "v2"

_WORD = re.compile(r"\w+")
_STEMMER = snowballstemmer.stemmer("english")

# Small on purpose: physics prose is dense with content words, and an
# aggressive list would drop terms like "state" or "order" that matter here.
STOPWORDS = frozenset("""
a an and are as at be been but by for from has have in into is it its of on
or that the their there these this to was we were which with
""".split())


def normalize(text: str) -> str:
    """NFKC (ligatures → letters, full-width → ASCII) and collapsed whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text or "").split())


def legacy_tokenize(text: str) -> list[str]:
    """What rebuild_bm25() and hybrid_search() did before v2. Kept for v1 pickles."""
    return _WORD.findall((text or "").lower())


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(normalize(text).lower())
    kept = [w for w in words if len(w) > 1 and w not in STOPWORDS and not w.isdigit()]
    return _STEMMER.stemWords(kept)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_tokenize.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/tokenize.py tests/test_tokenize.py requirements.txt requirements-test.txt
git commit -m "feat(search): NFKC + stemmed BM25 tokenizer with a version tag"
```

---

### Task 3: BM25 pickle header; `db.py` loads both formats; `search.py` picks the query tokenizer

**Files:**
- Modify: `research_assistant/shared/ingestion.py:753-789` (`rebuild_bm25`)
- Modify: `research_assistant/shared/db.py:84-92` (pickle load)
- Modify: `research_assistant/shared/search.py:12-16` (imports), `:58-59` (query tokenisation)
- Test: `tests/test_db.py` (new), `tests/test_search.py` (extend)

**Interfaces:**
- Consumes: `tokenize.tokenize`, `tokenize.legacy_tokenize`, `tokenize.TOKENIZER_VERSION` (Task 2).
- Produces: BM25 pickle format `{"tokenizer": "v2", "built_at": str, "bm25": BM25Okapi}`; `load_search_resources()` returns a `bm25` object with attribute `tokenizer_version` (`"v1"` for a bare legacy pickle); `hybrid_search` reads `getattr(bm25, "tokenizer_version", "v1")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py
"""The BM25 pickle carries its tokenizer version, and both formats load.

A pickle built by tokenize() searched with legacy_tokenize() silently loses
most keyword recall — the stems don't match the raw words. So the pickle
says which built it, the loader tags the object, and search reads the tag.
"""

import os
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import research_assistant.shared.db as db


class _Bm25Stub:
    def get_scores(self, q):
        return [0.0]


class _FakeCollection:
    def get(self, include=None, limit=None, offset=0):
        if offset:
            return {"ids": [], "documents": [], "metadatas": []}
        return {"ids": ["chunk_0"], "documents": ["x"], "metadatas": [{"document": "a.pdf"}]}


class LoadTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkl = os.path.join(self.tmp.name, "bm25.pkl")
        chroma = types.ModuleType("chromadb")
        chroma.PersistentClient = lambda path: types.SimpleNamespace(
            get_collection=lambda name: _FakeCollection()
        )
        p = patch.dict(sys.modules, {"chromadb": chroma}); p.start(); self.addCleanup(p.stop)
        p = patch.object(db, "BM25_INDEX_PATH", self.pkl); p.start(); self.addCleanup(p.stop)

    def test_legacy_bare_pickle_is_tagged_v1(self):
        with open(self.pkl, "wb") as f:
            pickle.dump(_Bm25Stub(), f)
        _, bm25, texts, _ = db.load_search_resources()
        self.assertEqual(bm25.tokenizer_version, "v1")
        self.assertEqual(texts, ["x"])

    def test_v2_pickle_is_unwrapped_and_tagged(self):
        with open(self.pkl, "wb") as f:
            pickle.dump({"tokenizer": "v2", "built_at": "now", "bm25": _Bm25Stub()}, f)
        _, bm25, _, _ = db.load_search_resources()
        self.assertEqual(bm25.tokenizer_version, "v2")
        self.assertTrue(hasattr(bm25, "get_scores"))

    def test_unknown_tokenizer_warns_but_loads(self):
        with open(self.pkl, "wb") as f:
            pickle.dump({"tokenizer": "v9", "built_at": "now", "bm25": _Bm25Stub()}, f)
        with self.assertLogs("db", level="WARNING") as cm:
            _, bm25, _, _ = db.load_search_resources()
        self.assertEqual(bm25.tokenizer_version, "v9")
        self.assertTrue(any("tokenizer" in line for line in cm.output))
```

Append to `tests/test_search.py`:

```python


class TestQueryTokenizerFollowsThePickle(unittest.TestCase):
    """The query is tokenised the way the index was built."""

    class _RecordingBM25:
        def __init__(self, version):
            self.tokenizer_version = version
            self.seen = None

        def get_scores(self, tokenized_query):
            self.seen = list(tokenized_query)
            return np.asarray([1.0, 0.5])

    def _run(self, bm25):
        texts, metadatas = _corpus(2)
        hybrid_search("The ﬁelds", FakeCollection([]), bm25, texts, metadatas,
                      top_k=1, embeddings_model=FakeEmbeddings())
        return bm25.seen

    def test_v2_pickle_gets_stemmed_normalised_tokens(self):
        self.assertEqual(self._run(self._RecordingBM25("v2")), ["field"])

    def test_v1_pickle_gets_legacy_tokens(self):
        self.assertEqual(self._run(self._RecordingBM25("v1")), ["the", "ﬁelds"])

    def test_untagged_object_is_treated_as_v1(self):
        bm25 = self._RecordingBM25("v1")
        del bm25.tokenizer_version
        self.assertEqual(self._run(bm25), ["the", "ﬁelds"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_db.py tests/test_search.py -v`
Expected: `test_db.py` — FAIL, `AttributeError: '_Bm25Stub' object has no attribute 'tokenizer_version'` / dict has no `get_scores`; `test_search.py::TestQueryTokenizerFollowsThePickle::test_v2_pickle...` — FAIL (gets legacy tokens).

- [ ] **Step 3: `rebuild_bm25()` writes the header and indexes title + text**

In `research_assistant/shared/ingestion.py`, add to the imports (after `from research_assistant.shared import manifest`):

```python
from research_assistant.shared.tokenize import tokenize, TOKENIZER_VERSION
```

Replace the body of `rebuild_bm25()` from `# Use larger page size to reduce round-trips` to the end of the function with:

```python
    # Use larger page size to reduce round-trips
    paired, limit, offset = [], 5000, 0
    while True:
        batch = collection.get(include=["documents", "metadatas"], limit=limit, offset=offset)
        if not batch or not batch["ids"]:
            break
        for doc, meta, cid in zip(batch["documents"], batch["metadatas"] or [], batch["ids"]):
            try:
                idx = int(cid.split("_")[1])
            except (ValueError, IndexError):
                continue
            # The paper title rides along with every chunk (citation_source),
            # so a query that names the paper keyword-matches its chunks.
            title = (meta or {}).get("citation_source") or ""
            paired.append((idx, doc, title))
        offset += limit

    paired.sort(key=lambda x: x[0])
    bm25 = BM25Okapi([tokenize(f"{title} {text}") for _, text, title in paired])

    # Atomic: load_search_resources() unpickles this on every query, so a
    # partial write is not a degraded index but an exception on the next
    # search, recoverable only by re-ingesting. The header records which
    # tokenizer built it so search tokenises queries the same way.
    payload = {
        "tokenizer": TOKENIZER_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bm25": bm25,
    }
    with atomic_write(BM25_INDEX_PATH, binary=True) as f:
        pickle.dump(payload, f)
    elapsed = time.perf_counter() - t0
    logger.info("✓ BM25 index rebuilt (%d documents, tokenizer %s) in %.1fs.",
                len(paired), TOKENIZER_VERSION, elapsed)
```

- [ ] **Step 4: `load_search_resources()` accepts both formats**

In `research_assistant/shared/db.py`, add after `from research_assistant.shared.log import get_logger`:

```python
from research_assistant.shared.tokenize import TOKENIZER_VERSION
```

Replace the final block (from `logger.info("Loading BM25 index from %s…", BM25_INDEX_PATH)` to `return collection, bm25, texts, metadatas`) with:

```python
    logger.info("Loading BM25 index from %s…", BM25_INDEX_PATH)
    with open(BM25_INDEX_PATH, "rb") as f:
        payload = pickle.load(f)

    # Two on-disk shapes: a bare BM25Okapi (built before the tokenizer was
    # versioned, with `\w+`) and a dict carrying the tokenizer tag. The tag is
    # copied onto the object so hybrid_search can pick the matching query
    # tokenizer without a second return value.
    if isinstance(payload, dict) and "bm25" in payload:
        bm25 = payload["bm25"]
        version = payload.get("tokenizer", "v1")
    else:
        bm25 = payload
        version = "v1"
    bm25.tokenizer_version = version
    if version not in ("v1", TOKENIZER_VERSION):
        logger.warning(
            "BM25 index was built with tokenizer %s but this code knows %s — "
            "keyword recall may be degraded; re-run ingestion to rebuild it.",
            version, TOKENIZER_VERSION,
        )

    return collection, bm25, texts, metadatas
```

- [ ] **Step 5: `hybrid_search()` picks the query tokenizer**

In `research_assistant/shared/search.py`, replace the import block:

```python
import re

import numpy as np

from research_assistant.config import RRF_K, DEFAULT_TOP_K
from research_assistant.shared.log import get_logger
```

with:

```python
import numpy as np

from research_assistant.config import RRF_K, DEFAULT_TOP_K
from research_assistant.shared.log import get_logger
from research_assistant.shared.tokenize import tokenize, legacy_tokenize
```

and replace:

```python
    # 1. Sparse (BM25) retrieval
    tokenized_query = re.findall(r'\w+', query.lower())
```

with:

```python
    # 1. Sparse (BM25) retrieval — tokenised the way the pickle was built
    # (db.load_search_resources tags the object; an untagged object is v1).
    if getattr(bm25, "tokenizer_version", "v1") == "v1":
        tokenized_query = legacy_tokenize(query)
    else:
        tokenized_query = tokenize(query)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_db.py tests/test_search.py tests/test_ingestion.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add research_assistant/shared/ingestion.py research_assistant/shared/db.py research_assistant/shared/search.py tests/test_db.py tests/test_search.py
git commit -m "feat(search): BM25 pickle records its tokenizer; queries tokenised to match; title indexed with each chunk"
```

---

### Task 4: `shared/extract.py` — TEI parser and the `Document` shape

**Files:**
- Create: `research_assistant/shared/extract.py`
- Create: `tests/fixtures/sample.grobid.tei.xml`
- Test: `tests/test_extract.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass class Paragraph: text: str; page: int              # page is 0-based
  @dataclass class Figure:    id: str; kind: str; label: str; caption: str; page: int
                              bbox: tuple[float,float,float,float] | None; refs: list[str]
  @dataclass class Section:   heading: str; kind: str; paragraphs: list[Paragraph]
  @dataclass class Document:  key: str; title: str; abstract: list[Paragraph]
                              sections: list[Section]; figures: list[Figure]
                              extraction: str; n_bib: int = 0
  SECTION_KINDS = ("abstract","introduction","background","methods","results","discussion","conclusion","acknowledgements","other")
  def normalise_section_kind(heading: str) -> str
  def parse_tei(xml: bytes, key: str) -> Document
  def is_running_header(heading: str) -> bool
  ```

- [ ] **Step 1: Write the fixture**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- tests/fixtures/sample.grobid.tei.xml — trimmed shape of GROBID 0.8.1
     processFulltextDocument output with teiCoordinates=p,head,figure and
     segmentSentences=1. Pages in coords are 1-based. -->
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="_x">
  <teiHeader>
    <fileDesc>
      <titleStmt><title level="a" type="main">Charge transport in covalent MoS2 networks</title></titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract>
        <div><p coords="1,50.0,100.0,400.0,10.0">We study charge transport in MoS2 networks. Hopping dominates at low temperature.</p></div>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head n="1.">Introduction</head>
        <p coords="1,50.0,200.0,400.0,10.0;1,50.0,212.0,400.0,10.0"><s>Two-dimensional semiconductors are of interest for flexible electronics.</s> <s>We find that the photoconductivity increases by 10% in covalent networks (<ref type="figure" target="#fig_0">Figure 1b</ref>).</s></p>
        <p coords="2,50.0,60.0,400.0,10.0"><s>This paragraph starts on page two.</s></p>
      </div>
      <div><head>(3 of 11)</head>
        <p coords="3,50.0,60.0,400.0,10.0"><s>Text under a running header belongs to the previous section.</s></p>
      </div>
      <div><head n="2.">Methods</head>
        <p coords="3,50.0,300.0,400.0,10.0"><s>Films were prepared by exfoliation.</s></p>
        <formula xml:id="formula_0" coords="3,50.0,320.0,100.0,12.0">σ = n e μ</formula>
        <p coords="3,50.0,340.0,400.0,10.0"><s>Conductivity was measured in a four-probe geometry.</s></p>
      </div>
      <div><head n="3.">Acknowledgements</head>
        <p coords="10,50.0,60.0,400.0,10.0"><s>We thank the funding agency.</s></p>
      </div>
      <figure xml:id="fig_0" coords="3,48.0,653.0,496.0,8.0;3,50.0,379.0,492.0,264.0">
        <head>Figure 1.</head><label>1</label>
        <figDesc>Figure 1. THz spectral analysis on MoS2 films. a) Schematic. b) Photoconductivity dynamics.</figDesc>
        <graphic coords="3,50.0,379.0,492.0,264.0" type="bitmap" />
      </figure>
      <figure xml:id="tab_0" type="table" coords="5,48.0,100.0,496.0,200.0">
        <head>Table 1</head><label>1</label>
        <figDesc>Table 1. Mobility of each sample.</figDesc>
        <table><row><cell>Sample</cell><cell>Mobility</cell></row></table>
      </figure>
    </body>
    <back>
      <div type="references"><listBibl>
        <biblStruct xml:id="b0"><analytic><title>Ref one</title></analytic></biblStruct>
        <biblStruct xml:id="b1"><analytic><title>Ref two</title></analytic></biblStruct>
      </listBibl></div>
    </back>
  </text>
</TEI>
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_extract.py
"""TEI → Document. What the chunker receives from GROBID.

The fixture exercises the things PyMuPDF got wrong on the real corpus:
sections with headings, a running header GROBID mislabels as a section,
formulas as siblings of paragraphs, a figure with a graphic box, a table
with no graphic, in-text figure references, and a bibliography kept apart.
"""

import os
import unittest

from research_assistant.shared import extract as ex

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.grobid.tei.xml")


def _doc():
    with open(FIXTURE, "rb") as f:
        return ex.parse_tei(f.read(), key="sample.pdf")


class TestHeaderAndAbstract(unittest.TestCase):
    def test_title_and_key(self):
        d = _doc()
        self.assertEqual(d.key, "sample.pdf")
        self.assertEqual(d.title, "Charge transport in covalent MoS2 networks")
        self.assertEqual(d.extraction, "grobid")

    def test_abstract_is_paragraphs_with_zero_based_page(self):
        d = _doc()
        self.assertEqual(len(d.abstract), 1)
        self.assertTrue(d.abstract[0].text.startswith("We study charge transport"))
        self.assertEqual(d.abstract[0].page, 0)

    def test_bibliography_is_counted_not_included(self):
        d = _doc()
        self.assertEqual(d.n_bib, 2)
        body = " ".join(p.text for s in d.sections for p in s.paragraphs)
        self.assertNotIn("Ref one", body)


class TestSections(unittest.TestCase):
    def test_headings_kinds_and_running_header_folded_into_previous(self):
        d = _doc()
        self.assertEqual([s.heading for s in d.sections], ["Introduction", "Methods", "Acknowledgements"])
        self.assertEqual([s.kind for s in d.sections], ["introduction", "methods", "acknowledgements"])
        intro = d.sections[0]
        self.assertEqual(len(intro.paragraphs), 3)            # 2 own + 1 folded from "(3 of 11)"
        self.assertEqual([p.page for p in intro.paragraphs], [0, 1, 2])

    def test_paragraph_text_includes_ref_text_and_is_single_spaced(self):
        p = _doc().sections[0].paragraphs[0]
        self.assertIn("(Figure 1b)", p.text)
        self.assertNotIn("\n", p.text)

    def test_formulas_are_not_paragraphs(self):
        methods = _doc().sections[1]
        self.assertEqual(len(methods.paragraphs), 2)
        self.assertNotIn("σ = n e μ", " ".join(p.text for p in methods.paragraphs))


class TestFigures(unittest.TestCase):
    def test_figure_with_graphic_box(self):
        fig = _doc().figures[0]
        self.assertEqual((fig.id, fig.kind, fig.label), ("fig_0", "figure", "Figure 1"))
        self.assertTrue(fig.caption.startswith("Figure 1. THz spectral analysis"))
        self.assertEqual(fig.page, 2)
        self.assertEqual(fig.bbox, (50.0, 379.0, 542.0, 643.0))   # x0, y0, x0+w, y0+h

    def test_table_uses_union_of_figure_coords(self):
        tab = _doc().figures[1]
        self.assertEqual((tab.id, tab.kind, tab.label), ("tab_0", "table", "Table 1"))
        self.assertEqual(tab.page, 4)
        self.assertEqual(tab.bbox, (48.0, 100.0, 544.0, 300.0))

    def test_in_text_references_are_attached_as_sentences(self):
        fig = _doc().figures[0]
        self.assertEqual(len(fig.refs), 1)
        self.assertIn("photoconductivity increases by 10%", fig.refs[0])
        self.assertEqual(_doc().figures[1].refs, [])


class TestSectionKinds(unittest.TestCase):
    def test_keyword_mapping(self):
        cases = {
            "1. Introduction": "introduction", "Related work": "background",
            "Experimental methods": "methods", "Results and discussion": "results",
            "IV. DISCUSSION": "discussion", "Summary and outlook": "conclusion",
            "Conclusions": "conclusion", "Acknowledgments": "acknowledgements",
            "Funding": "acknowledgements", "Author contributions": "acknowledgements",
            "Terahertz spectral analysis": "other", "": "other",
        }
        for heading, kind in cases.items():
            self.assertEqual(ex.normalise_section_kind(heading), kind, heading)

    def test_running_header_detection(self):
        for h in ["(3 of 11)", "Odashima et al.", "12", "Adv. Mater. 2023, 35, 2211157"]:
            self.assertTrue(ex.is_running_header(h), h)
        for h in ["Introduction", "2.1 Green's functions", "Results"]:
            self.assertFalse(ex.is_running_header(h), h)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_extract.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract'`.

- [ ] **Step 4: Write the parser**

```python
# research_assistant/shared/extract.py
"""PDF → Document: the structure the v2 chunker consumes.

Two producers, one shape. parse_tei() reads GROBID's processFulltextDocument
output — sections with headings, paragraphs with page numbers, figures with
captions and bitmap boxes, and the bibliography kept apart. extract_pymupdf()
(Task 5) is the fallback and produces the same shape with one section of
kind "other" and no figures.

Pages are 0-based throughout, matching what v1 stored from PyMuPDF; GROBID's
coords are 1-based and converted here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

from research_assistant.shared.log import get_logger

logger = get_logger("extract")

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

SECTION_KINDS = (
    "abstract", "introduction", "background", "methods", "results",
    "discussion", "conclusion", "acknowledgements", "other",
)


@dataclass
class Paragraph:
    text: str
    page: int


@dataclass
class Figure:
    id: str
    kind: str                       # "figure" | "table"
    label: str                      # "Figure 1", "Table 2"
    caption: str
    page: int
    bbox: tuple | None              # (x0, y0, x1, y1) in PDF points, or None
    refs: list = field(default_factory=list)   # body sentences that cite it


@dataclass
class Section:
    heading: str
    kind: str
    paragraphs: list


@dataclass
class Document:
    key: str
    title: str
    abstract: list
    sections: list
    figures: list
    extraction: str                 # "grobid" | "pymupdf"
    n_bib: int = 0


# ─── Section kinds ───────────────────────────────────────────────────────────

_KIND_PATTERNS = [
    ("acknowledgements", re.compile(r"acknowledg|funding|author contribution|competing interest|conflict of interest|data availability", re.I)),
    ("introduction",     re.compile(r"\bintroduction\b", re.I)),
    ("background",       re.compile(r"\b(background|related work|prior work|preliminar)", re.I)),
    ("conclusion",       re.compile(r"\b(conclusion|summary|outlook)", re.I)),
    ("results",          re.compile(r"\bresults?\b", re.I)),
    ("discussion",       re.compile(r"\bdiscussion\b", re.I)),
    ("methods",          re.compile(r"\b(method|experimental|materials|setup|procedure|model|theory)", re.I)),
]


def normalise_section_kind(heading: str) -> str:
    """Map a raw heading to one of SECTION_KINDS. Order matters: 'Results and
    discussion' is results; 'Summary and outlook' is conclusion."""
    text = heading or ""
    for kind, pat in _KIND_PATTERNS:
        if pat.search(text):
            return kind
    return "other"


_RUNNING = [
    re.compile(r"^\(\d+ of \d+\)$"),
    re.compile(r"\bet al\.?$", re.I),
    re.compile(r"^\d+$"),
    re.compile(r"\b(19|20)\d{2}\b.*\b\d{3,}\b"),      # "Adv. Mater. 2023, 35, 2211157"
]


def is_running_header(heading: str) -> bool:
    """GROBID sometimes emits a page header/footer as a <head> without an n=.
    Those must not start a section — their paragraphs belong to the previous one."""
    h = (heading or "").strip()
    return bool(h) and any(p.search(h) for p in _RUNNING)


# ─── TEI helpers ─────────────────────────────────────────────────────────────

def _text(el) -> str:
    # "".join, not " ".join: GROBID's text nodes carry their own spacing, and
    # inserting one at every element boundary turns "(Figure 1b)" into
    # "( Figure 1b )".
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _first_page(coords: str | None) -> int:
    """'3,48.0,653.0,496.0,8.0;3,...' → 2 (0-based). Unknown → 0."""
    if not coords:
        return 0
    try:
        return max(0, int(coords.split(";")[0].split(",")[0]) - 1)
    except (ValueError, IndexError):
        return 0


def _union_box(coords: str | None):
    """Union of every (x,y,w,h) group in a coords string → (x0,y0,x1,y1)."""
    boxes = []
    for group in (coords or "").split(";"):
        parts = group.split(",")
        if len(parts) != 5:
            continue
        try:
            _, x, y, w, h = (float(v) for v in parts)
        except ValueError:
            continue
        boxes.append((x, y, x + w, y + h))
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _sentence_containing(ref) -> str:
    """The <s> holding a <ref>, else its <p>, as single-spaced text (≤ 400 chars)."""
    node = ref.getparent()
    while node is not None and node.tag not in (f"{TEI}s", f"{TEI}p"):
        node = node.getparent()
    return _text(node)[:400] if node is not None else ""


# ─── Parser ──────────────────────────────────────────────────────────────────

def parse_tei(xml: bytes, key: str) -> Document:
    root = etree.fromstring(xml)

    title_el = root.find(f".//{TEI}titleStmt/{TEI}title")
    title = _text(title_el)

    abstract = [
        Paragraph(_text(p), _first_page(p.get("coords")))
        for p in root.findall(f".//{TEI}abstract//{TEI}p")
        if _text(p)
    ]

    body = root.find(f".//{TEI}body")
    sections: list[Section] = []
    if body is not None:
        for div in body.findall(f"{TEI}div"):
            head = div.find(f"{TEI}head")
            heading = _text(head)
            paras = [
                Paragraph(_text(p), _first_page(p.get("coords")))
                for p in div.findall(f"{TEI}p")
                if _text(p)
            ]
            unnumbered = head is None or head.get("n") is None
            if unnumbered and is_running_header(heading) and sections:
                sections[-1].paragraphs.extend(paras)
                continue
            if not paras and not heading:
                continue
            sections.append(Section(heading, normalise_section_kind(heading), paras))

    figures: list[Figure] = []
    by_id: dict[str, Figure] = {}
    if body is not None:
        for i, fig in enumerate(body.findall(f".//{TEI}figure")):
            fid = fig.get(XML_ID) or f"fig_{i}"
            kind = "table" if fig.get("type") == "table" else "figure"
            head = _text(fig.find(f"{TEI}head")).rstrip(" .")
            label_num = _text(fig.find(f"{TEI}label"))
            label = head or (f"{'Table' if kind == 'table' else 'Figure'} {label_num}".strip())
            caption = _text(fig.find(f"{TEI}figDesc"))
            graphic = fig.find(f"{TEI}graphic")
            coords = graphic.get("coords") if graphic is not None and graphic.get("coords") else fig.get("coords")
            f = Figure(fid, kind, label, caption, _first_page(coords), _union_box(coords))
            figures.append(f)
            by_id[fid] = f

        for ref in body.iter(f"{TEI}ref"):
            if ref.get("type") != "figure":
                continue
            target = (ref.get("target") or "").lstrip("#")
            f = by_id.get(target)
            if f is None:
                continue
            sentence = _sentence_containing(ref)
            if sentence and sentence not in f.refs:
                f.refs.append(sentence)

    n_bib = len(root.findall(f".//{TEI}listBibl/{TEI}biblStruct"))

    return Document(key=key, title=title, abstract=abstract, sections=sections,
                    figures=figures, extraction="grobid", n_bib=n_bib)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_extract.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/extract.py tests/test_extract.py tests/fixtures/sample.grobid.tei.xml
git commit -m "feat(extract): parse GROBID full-text TEI into a Document with sections, pages, figures and refs"
```

---

### Task 5: `shared/extract.py` — GROBID call with TEI cache, PyMuPDF fallback, `extract()`

**Files:**
- Modify: `research_assistant/shared/extract.py`
- Test: `tests/test_extract.py` (extend)

**Interfaces:**
- Consumes: `config.GROBID_SERVER`, `config.GROBID_TEI_DIR`, `config.GROBID_FULLTEXT_TIMEOUT` (Task 1); `tokenize.normalize` (Task 2).
- Produces:
  ```python
  def tei_cache_path(pdf_path: str) -> str            # GROBID_TEI_DIR/<stem>.grobid.tei.xml, or an existing Agent-1 TEI for that stem
  def grobid_fulltext(pdf_path: str, server: str = GROBID_SERVER, timeout: int = GROBID_FULLTEXT_TIMEOUT) -> bytes   # raises on any failure
  def extract_pymupdf(pdf_path: str, key: str) -> Document
  def extract(pdf_path: str, key: str | None = None) -> Document   # cache → GROBID → PyMuPDF
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extract.py`:

```python


import sys
import tempfile
import types
from unittest.mock import patch


class TestPyMuPDFFallback(unittest.TestCase):
    """The same Document shape from page blocks, with the v1 defects fixed
    where a heuristic can fix them: ligatures, running headers, the
    bibliography cut at its heading."""

    def _fake_fitz(self, pages):
        class _Page:
            def __init__(self, blocks):
                self._blocks = blocks

            def get_text(self, kind):
                return [(0, 0, 1, 1, b, i, 0) for i, b in enumerate(self._blocks)]

        class _Doc(list):
            pass

        fitz = types.ModuleType("fitz")
        fitz.open = lambda path: _Doc(_Page(p) for p in pages)
        return patch.dict(sys.modules, {"fitz": fitz})

    def test_ligatures_headers_and_bibliography(self):
        pages = [
            ["Odashima et al.", "The ﬁrst result is deﬁned here."],
            ["Odashima et al.", "Second page text."],
            ["Odashima et al.", "Third page text."],
            ["Odashima et al.", "References", "1. A. Author, Phys. Rev. B 1, 1 (2000)."],
        ]
        with self._fake_fitz(pages):
            d = ex.extract_pymupdf("/x/paper.pdf", key="paper.pdf")
        self.assertEqual(d.extraction, "pymupdf")
        self.assertEqual(len(d.sections), 1)
        self.assertEqual(d.sections[0].kind, "other")
        texts = [p.text for p in d.sections[0].paragraphs]
        self.assertEqual(texts, ["The first result is defined here.", "Second page text.", "Third page text."])
        self.assertEqual([p.page for p in d.sections[0].paragraphs], [0, 1, 2])
        self.assertEqual(d.figures, [])
        self.assertEqual(d.title, "paper.pdf")


class TestExtractDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(ex, "GROBID_TEI_DIR", self.tmp.name); p.start(); self.addCleanup(p.stop)
        with open(FIXTURE, "rb") as f:
            self.tei = f.read()

    def test_cached_tei_is_used_without_calling_grobid(self):
        with open(os.path.join(self.tmp.name, "paper.grobid.tei.xml"), "wb") as f:
            f.write(self.tei)
        with patch.object(ex, "grobid_fulltext", side_effect=AssertionError("must not be called")):
            d = ex.extract("/x/paper.pdf", key="paper.pdf")
        self.assertEqual(d.extraction, "grobid")
        self.assertEqual(d.key, "paper.pdf")

    def test_agent1_tei_for_the_same_stem_is_reused(self):
        with open(os.path.join(self.tmp.name, "paper.fulltext.tei.xml"), "wb") as f:
            f.write(self.tei)
        with patch.object(ex, "grobid_fulltext", side_effect=AssertionError("must not be called")):
            d = ex.extract("/x/paper.pdf", key="paper.pdf")
        self.assertEqual(d.extraction, "grobid")

    def test_grobid_response_is_cached_then_parsed(self):
        with patch.object(ex, "grobid_fulltext", return_value=self.tei) as call:
            d = ex.extract("/x/paper.pdf", key="paper.pdf")
        call.assert_called_once()
        self.assertEqual(d.extraction, "grobid")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "paper.grobid.tei.xml")))

    def test_grobid_failure_falls_back_to_pymupdf(self):
        fallback = ex.Document("paper.pdf", "paper.pdf", [], [], [], "pymupdf")
        with patch.object(ex, "grobid_fulltext", side_effect=RuntimeError("down")), \
             patch.object(ex, "extract_pymupdf", return_value=fallback) as fb:
            d = ex.extract("/x/paper.pdf", key="paper.pdf")
        fb.assert_called_once()
        self.assertEqual(d.extraction, "pymupdf")

    def test_empty_grobid_body_falls_back(self):
        empty = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body/></text></TEI>'
        fallback = ex.Document("paper.pdf", "paper.pdf", [], [], [], "pymupdf")
        with patch.object(ex, "grobid_fulltext", return_value=empty), \
             patch.object(ex, "extract_pymupdf", return_value=fallback):
            d = ex.extract("/x/paper.pdf", key="paper.pdf")
        self.assertEqual(d.extraction, "pymupdf")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_extract.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'extract_pymupdf'` / `'GROBID_TEI_DIR'`.

- [ ] **Step 3: Add the I/O half of the module**

Append to `research_assistant/shared/extract.py` (and add `import os` and the config import at the top, after `import re`):

```python
import os

from research_assistant.config import GROBID_SERVER, GROBID_TEI_DIR, GROBID_FULLTEXT_TIMEOUT
from research_assistant.shared.tokenize import normalize
```

then at the end of the file:

```python


# ─── GROBID call + cache ─────────────────────────────────────────────────────

# Agent 1 caches TEI under these suffixes for seed papers; reuse any of them
# for the same PDF stem before asking GROBID again.
_TEI_SUFFIXES = (".grobid.tei.xml", ".fulltext.tei.xml", ".tei.xml")


def _stem(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]


def tei_cache_path(pdf_path: str) -> str:
    stem = _stem(pdf_path)
    for suffix in _TEI_SUFFIXES:
        candidate = os.path.join(GROBID_TEI_DIR, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(GROBID_TEI_DIR, stem + _TEI_SUFFIXES[0])


def grobid_fulltext(pdf_path: str, server: str = GROBID_SERVER,
                    timeout: int = GROBID_FULLTEXT_TIMEOUT) -> bytes:
    """POST the PDF to processFulltextDocument. Raises on any failure —
    extract() decides what to do about it."""
    import requests

    with open(pdf_path, "rb") as fh:
        # Repeated teiCoordinates fields: paragraph coords give the page,
        # figure coords give the bitmap box. segmentSentences wraps sentences
        # in <s>, which is what lets a figure reference carry its sentence.
        data = [
            ("consolidateHeader", "0"),
            ("segmentSentences", "1"),
            ("teiCoordinates", "p"),
            ("teiCoordinates", "head"),
            ("teiCoordinates", "figure"),
        ]
        resp = requests.post(f"{server.rstrip('/')}/api/processFulltextDocument",
                             files={"input": fh}, data=data, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"GROBID returned HTTP {resp.status_code} for {os.path.basename(pdf_path)}")
    if not resp.content.strip():
        raise RuntimeError(f"GROBID returned an empty body for {os.path.basename(pdf_path)}")
    return resp.content


# ─── PyMuPDF fallback ────────────────────────────────────────────────────────

_BIB_HEAD = re.compile(r"^\s*(references?|bibliography|literature cited)\s*$", re.I)


def extract_pymupdf(pdf_path: str, key: str) -> Document:
    """Page text blocks → one section of kind "other". Ligatures are NFKC-
    normalised, blocks repeated on ≥30% of pages are dropped as running
    headers/footers, and everything from the last References heading on is
    cut. No figures: nothing in a text block says where a figure is."""
    import fitz

    pdf = fitz.open(pdf_path)
    pages: list[list[str]] = []
    for page in pdf:
        blocks = []
        for b in page.get_text("blocks"):
            if b[6] != 0:                       # image block
                continue
            text = normalize(b[4].replace("\n", " "))
            if len(text) > 4:
                blocks.append(text)
        pages.append(blocks)

    npages = len(pages)
    seen_on = {}
    for blocks in pages:
        for b in set(blocks):
            seen_on[b] = seen_on.get(b, 0) + 1
    repeated = {b for b, n in seen_on.items() if npages >= 4 and n >= max(2, 0.3 * npages)}

    flat: list[Paragraph] = []
    for i, blocks in enumerate(pages):
        for b in blocks:
            if b not in repeated:
                flat.append(Paragraph(b, i))

    cut = max((i for i, p in enumerate(flat) if _BIB_HEAD.match(p.text)), default=None)
    body = flat[:cut] if cut is not None else flat

    return Document(key=key, title=key, abstract=[],
                    sections=[Section("", "other", body)], figures=[],
                    extraction="pymupdf")


# ─── Entry point ─────────────────────────────────────────────────────────────

def extract(pdf_path: str, key: str | None = None) -> Document:
    """Cached TEI → GROBID → PyMuPDF. Never raises for a readable PDF; the
    Document says which path produced it."""
    key = key or os.path.basename(pdf_path).strip().replace(" ", "_").lower()
    cache = tei_cache_path(pdf_path)

    xml = None
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            xml = f.read()
    else:
        try:
            xml = grobid_fulltext(pdf_path)
            os.makedirs(GROBID_TEI_DIR, exist_ok=True)
            with open(cache, "wb") as f:
                f.write(xml)
        except Exception as exc:                # noqa: BLE001 — any failure is a fallback
            logger.warning("GROBID full-text failed for %s (%s) — falling back to PyMuPDF.",
                           os.path.basename(pdf_path), exc)

    if xml is not None:
        try:
            doc = parse_tei(xml, key)
        except etree.XMLSyntaxError as exc:
            logger.warning("Cached TEI for %s is not well-formed (%s) — falling back to PyMuPDF.",
                           os.path.basename(pdf_path), exc)
            doc = None
        if doc is not None and (doc.abstract or any(s.paragraphs for s in doc.sections)):
            return doc
        logger.warning("GROBID produced no body text for %s — falling back to PyMuPDF.",
                       os.path.basename(pdf_path))

    return extract_pymupdf(pdf_path, key)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_extract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/extract.py tests/test_extract.py
git commit -m "feat(extract): GROBID full-text with TEI cache, PyMuPDF fallback with NFKC/header/bibliography cleanup"
```

---

### Task 6: `shared/chunking.py` — sentences and windows

**Files:**
- Create: `research_assistant/shared/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `extract.Paragraph` (Task 4).
- Produces:
  ```python
  def sentences(text: str) -> list[str]
  def pack_windows(paragraphs: list[Paragraph], target: int, hard_max: int) -> list[tuple[str, int, int]]   # (text, page_first, page_last)
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunking.py
"""Sentence windows: what replaces per-page SemanticChunker.

The v1 index's median chunk was 242 chars and 45% were under 200 — bibliography
fragments and clauses cut at an embedding-distance breakpoint. These tests pin
the four rules that stop that: pack to a target, never exceed the max, overlap
one sentence, and don't cut a sentence in half.
"""

import unittest

from research_assistant.shared import chunking as ck
from research_assistant.shared.extract import Paragraph


class TestSentences(unittest.TestCase):
    def test_splits_plain_prose(self):
        self.assertEqual(ck.sentences("One here. Two here. Three here."), ["One here.", "Two here.", "Three here."])

    def test_physics_abbreviations_do_not_split(self):
        text = "We follow Ref. [3] and Fig. 2b. Phys. Rev. B 70, 241403 (2004) reports it. The gap is 0.5 eV (see Sec. 3). Done."
        out = ck.sentences(text)
        self.assertEqual(out, [
            "We follow Ref. [3] and Fig. 2b.",
            "Phys. Rev. B 70, 241403 (2004) reports it.",
            "The gap is 0.5 eV (see Sec. 3).",
            "Done.",
        ])

    def test_empty(self):
        self.assertEqual(ck.sentences("   "), [])


class TestPackWindows(unittest.TestCase):
    def _sent(self, i, page=0):
        return f"Sentence number {i} has about sixty characters of text in it, yes."   # ~66 chars

    def _paras(self, n, per_para=5, page=0):
        return [Paragraph(" ".join(self._sent(i * per_para + j) for j in range(per_para)), page) for i in range(n)]

    def test_windows_reach_target_and_never_exceed_max(self):
        out = ck.pack_windows(self._paras(6), target=300, hard_max=450)
        self.assertGreater(len(out), 1)
        for text, _, _ in out:
            self.assertLessEqual(len(text), 450)
        # every window but the last is at least the target or a paragraph close
        for text, _, _ in out[:-1]:
            self.assertGreaterEqual(len(text), 0.7 * 300)

    def test_one_sentence_overlap(self):
        out = ck.pack_windows(self._paras(2, per_para=6), target=300, hard_max=450)
        first, second = out[0][0], out[1][0]
        last_sentence_of_first = ck.sentences(first)[-1]
        self.assertTrue(second.startswith(last_sentence_of_first))

    def test_no_sentence_is_cut(self):
        out = ck.pack_windows(self._paras(4), target=300, hard_max=450)
        for text, _, _ in out:
            for s in ck.sentences(text):
                self.assertTrue(s.endswith("."), s)

    def test_a_sentence_longer_than_max_stands_alone(self):
        giant = "X" + "x" * 600 + "."
        paras = [Paragraph(f"Short one. {giant} Short two.", 0)]
        out = ck.pack_windows(paras, target=300, hard_max=450)
        self.assertEqual([t for t, _, _ in out], ["Short one.", giant, "Short two."])

    def test_prefers_closing_at_a_paragraph_boundary(self):
        # Two paragraphs of ~260 chars: a 300 target must not glue them if the
        # first is already ≥ 70% of target.
        paras = self._paras(2, per_para=4)
        out = ck.pack_windows(paras, target=300, hard_max=450)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], paras[0].text)

    def test_pages_are_tracked_across_the_window(self):
        paras = [Paragraph(self._sent(0) + " " + self._sent(1), 3), Paragraph(self._sent(2) + " " + self._sent(3), 4)]
        out = ck.pack_windows(paras, target=1000, hard_max=1500)
        self.assertEqual(out, [(paras[0].text + " " + paras[1].text, 3, 4)])

    def test_empty_input(self):
        self.assertEqual(ck.pack_windows([], 300, 450), [])

    def test_overlap_alone_is_never_a_second_window(self):
        # 5 sentences ≈ 334 chars close one window at the target; the carried
        # overlap sentence must not be flushed as a window of its own.
        out = ck.pack_windows(self._paras(1, per_para=5), target=300, hard_max=450)
        self.assertEqual(len(out), 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_chunking.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunking'`.

- [ ] **Step 3: Write the module**

```python
# research_assistant/shared/chunking.py
"""Sentence windows packed within a section — the v2 chunker.

Pure functions: no I/O, no model calls, no config reads. chunk_document()
(Task 7) turns a Document into Chunks; this half is the sentence splitter
and the packer.

Rules (spec §4.3): close a window at `target` chars; never exceed `hard_max`
(a lone sentence over the max stands alone rather than being cut); prefer to
close at a paragraph boundary once ≥ 70% of target; the last sentence of one
window opens the next; never cross a section (the caller packs one section
at a time).
"""

from __future__ import annotations

import re

from research_assistant.shared.log import get_logger

logger = get_logger("chunking")

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[“\"])")
# A piece ending in one of these did not end a sentence. Applied after pysbd
# too: it splits "Phys. Rev." and "Ref. [3]" on physics prose.
_ABBR = re.compile(
    r"(\bet al|\bFig|\bFigs|\bRef|\bRefs|\bEq|\bEqs|\bSec|\bSecs|\bTab|\bvs|\bcf|\be\.g|\bi\.e"
    r"|\bPhys|\bRev|\bLett|\bNat|\bAdv|\bMater|\bAppl|\bJ|\bSci|\bChem|\bOpt|\bVol|\bpp"
    r"|\b[A-Z])\.$"
)


def _merge_abbreviations(pieces: list[str]) -> list[str]:
    out: list[str] = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if out and _ABBR.search(out[-1]):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def _regex_sentences(text: str) -> list[str]:
    return _merge_abbreviations(_SPLIT.split(text))


_segmenter = None


def sentences(text: str) -> list[str]:
    """pysbd when available, the regex splitter otherwise; the abbreviation
    merge runs on both."""
    global _segmenter
    text = " ".join((text or "").split())
    if not text:
        return []
    if _segmenter is None:
        try:
            import pysbd
            _segmenter = pysbd.Segmenter(language="en", clean=False)
        except Exception as exc:                        # noqa: BLE001
            logger.debug("pysbd unavailable (%s) — regex sentence splitter in use.", exc)
            _segmenter = False
    if _segmenter:
        try:
            return _merge_abbreviations(_segmenter.segment(text))
        except Exception as exc:                        # noqa: BLE001
            logger.debug("pysbd failed on a paragraph (%s) — regex fallback.", exc)
    return _regex_sentences(text)


def pack_windows(paragraphs, target: int, hard_max: int) -> list[tuple[str, int, int]]:
    """Pack sentences into windows. Returns (text, page_first, page_last)."""
    out: list[tuple[str, int, int]] = []
    cur: list[str] = []          # sentences in the open window
    cur_pages: list[int] = []
    fresh = 0                    # sentences added since the last close, excluding the carried overlap

    def cur_len():
        return sum(len(x) for x in cur) + max(0, len(cur) - 1)

    def close(overlap=True):
        nonlocal cur, cur_pages, fresh
        if not cur:
            return
        out.append((" ".join(cur), min(cur_pages), max(cur_pages)))
        # one-sentence overlap: the last sentence opens the next window
        cur, cur_pages = ([cur[-1]], [cur_pages[-1]]) if overlap else ([], [])
        fresh = 0

    for para in paragraphs:
        for s in sentences(para.text):
            if len(s) > hard_max:
                # A giant sentence stands alone; whatever is pending closes
                # first (without overlap), a bare carried overlap is dropped.
                if fresh:
                    close(overlap=False)
                else:
                    cur, cur_pages, fresh = [], [], 0
                out.append((s, para.page, para.page))
                continue
            if cur and cur_len() + 1 + len(s) > hard_max:
                close()
                if cur and len(cur[0]) + 1 + len(s) > hard_max:   # the overlap itself is too long to carry
                    cur, cur_pages = [], []
            cur.append(s)
            cur_pages.append(para.page)
            fresh += 1
            if cur_len() >= target:
                close()
        # paragraph boundary: a good place to stop once the window is big enough
        if fresh and cur_len() >= 0.7 * target:
            close()

    # Flush only new content: after close() the open window is just the
    # carried overlap, and emitting that alone would duplicate the last window's tail.
    if fresh:
        close(overlap=False)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_chunking.py -v`
Expected: PASS (this implementation was run against exactly these tests, with pysbd installed and without it, before the plan was written).

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/chunking.py tests/test_chunking.py
git commit -m "feat(chunking): sentence splitter with physics abbreviation guard; windows packed to a target with one-sentence overlap"
```

---

### Task 7: `shared/chunking.py` — filters and `chunk_document()`

**Files:**
- Modify: `research_assistant/shared/chunking.py`
- Test: `tests/test_chunking.py` (extend)

**Interfaces:**
- Consumes: `extract.Document/Section/Paragraph/Figure` (Task 4); `pack_windows`, `sentences` (Task 6).
- Produces:
  ```python
  @dataclass class Chunk: text: str; type: str; section: str; section_raw: str
                          page_first: int; page_last: int; seq: int
                          figure_id: str = ""; figure_kind: str = ""; figure_label: str = ""
  DROP_KINDS = frozenset({"acknowledgements"})
  def alpha_ratio(s: str) -> float
  def filter_document(doc: Document, min_alpha: float = 0.6, min_section_chars: int = 200) -> Document
  def chunk_document(doc: Document, target: int, hard_max: int, min_chars: int = 200, min_alpha: float = 0.6) -> list[Chunk]
  ```
  `chunk_document` emits text chunks (`type="text"`) for abstract + every kept section, then one `type="caption"` chunk per figure (`"{label}: {caption}"`, exempt from `min_chars`), with `seq` numbered across all of them in that order.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunking.py`:

```python


from research_assistant.shared.extract import Document, Section, Figure


def _sentence(i):
    return f"Sentence number {i} has about sixty characters of text in it, yes."


def _section(heading, kind, n_sentences, page=0):
    return Section(heading, kind, [Paragraph(" ".join(_sentence(i) for i in range(n_sentences)), page)])


class TestFilters(unittest.TestCase):
    def test_alpha_ratio(self):
        self.assertGreater(ck.alpha_ratio("plain words here"), 0.9)
        self.assertLess(ck.alpha_ratio("2D! Q82A'C#+,!#>! #6!"), 0.6)

    def test_garbage_paragraphs_and_acknowledgements_are_dropped(self):
        doc = Document("k.pdf", "T", [], [
            _section("Introduction", "introduction", 6),
            Section("Results", "results", [Paragraph("2D! Q82A'C#+,!#>! #6! S%86'6! F?;! 5%.6'8%5%+>6", 3),
                                            Paragraph(" ".join(_sentence(i) for i in range(6)), 3)]),
            _section("Acknowledgements", "acknowledgements", 3),
        ], [], "grobid")
        out = ck.filter_document(doc)
        self.assertEqual([s.kind for s in out.sections], ["introduction", "results"])
        self.assertEqual(len(out.sections[1].paragraphs), 1)

    def test_short_sections_merge_forward(self):
        doc = Document("k.pdf", "T", [], [
            Section("Nomenclature", "other", [Paragraph("Ten chars.", 0)]),
            _section("Introduction", "introduction", 6, page=1),
        ], [], "grobid")
        out = ck.filter_document(doc)
        self.assertEqual(len(out.sections), 1)
        self.assertEqual(out.sections[0].kind, "introduction")
        self.assertEqual(out.sections[0].paragraphs[0].text, "Ten chars.")

    def test_trailing_short_section_merges_backward(self):
        doc = Document("k.pdf", "T", [], [
            _section("Introduction", "introduction", 6),
            Section("Note", "other", [Paragraph("Ten chars.", 5)]),
        ], [], "grobid")
        out = ck.filter_document(doc)
        self.assertEqual(len(out.sections), 1)
        self.assertEqual(out.sections[0].paragraphs[-1].text, "Ten chars.")


class TestChunkDocument(unittest.TestCase):
    def _doc(self):
        return Document("k.pdf", "A title", [Paragraph(" ".join(_sentence(i) for i in range(4)), 0)], [
            _section("1. Introduction", "introduction", 10, page=0),
            _section("2. Methods", "methods", 10, page=2),
        ], [
            Figure("fig_0", "figure", "Figure 1", "Density of states of the first site.", 1, (0, 0, 1, 1), ["Seen in Figure 1."]),
            Figure("tab_0", "table", "Table 1", "Mobilities.", 3, None, []),
        ], "grobid")

    def test_chunks_never_cross_a_section(self):
        chunks = ck.chunk_document(self._doc(), target=300, hard_max=450)
        for c in chunks:
            if c.type == "text":
                self.assertIn(c.section, ("abstract", "introduction", "methods"))
        sections_in_order = [c.section for c in chunks if c.type == "text"]
        self.assertEqual(sections_in_order, sorted(sections_in_order, key=["abstract", "introduction", "methods"].index))

    def test_seq_is_contiguous_and_pages_carried(self):
        chunks = ck.chunk_document(self._doc(), target=300, hard_max=450)
        self.assertEqual([c.seq for c in chunks], list(range(len(chunks))))
        methods = [c for c in chunks if c.section == "methods"]
        self.assertTrue(all(c.page_first == 2 for c in methods))
        self.assertEqual(chunks[0].section_raw, "")            # abstract has no heading
        self.assertEqual([c.section_raw for c in chunks if c.section == "methods"][0], "2. Methods")

    def test_captions_are_chunks_exempt_from_the_length_floor(self):
        chunks = ck.chunk_document(self._doc(), target=300, hard_max=450)
        caps = [c for c in chunks if c.type == "caption"]
        self.assertEqual([(c.figure_id, c.figure_kind, c.figure_label) for c in caps],
                         [("fig_0", "figure", "Figure 1"), ("tab_0", "table", "Table 1")])
        self.assertEqual(caps[0].text, "Figure 1: Density of states of the first site.")
        self.assertEqual(caps[1].page_first, 3)
        self.assertLess(len(caps[1].text), 200)

    def test_short_text_windows_are_dropped(self):
        doc = Document("k.pdf", "T", [], [Section("Intro", "introduction", [Paragraph("Tiny.", 0)])], [], "grobid")
        self.assertEqual(ck.chunk_document(doc, target=300, hard_max=450), [])

    def test_figure_without_caption_is_still_a_chunk(self):
        doc = Document("k.pdf", "T", [], [], [Figure("fig_0", "figure", "Figure 2", "", 0, None, [])], "grobid")
        chunks = ck.chunk_document(doc, target=300, hard_max=450)
        self.assertEqual(chunks[0].text, "Figure 2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_chunking.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'alpha_ratio'`.

- [ ] **Step 3: Add filters and `chunk_document`**

Append to `research_assistant/shared/chunking.py` (add `from dataclasses import dataclass` and `from research_assistant.shared.extract import Document, Section, Paragraph` to the imports):

```python


# ─── Filters ─────────────────────────────────────────────────────────────────

DROP_KINDS = frozenset({"acknowledgements"})


def alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(c.isalpha() or c.isspace() for c in s) / len(s)


def filter_document(doc: Document, min_alpha: float = 0.6, min_section_chars: int = 200) -> Document:
    """Drop what should never be indexed, then merge sections too small to
    stand alone into their neighbour (forward; the trailing one backward)."""
    kept: list[Section] = []
    for sec in doc.sections:
        if sec.kind in DROP_KINDS:
            continue
        paras = [p for p in sec.paragraphs if alpha_ratio(p.text) >= min_alpha]
        if not paras:
            continue
        kept.append(Section(sec.heading, sec.kind, paras))

    merged: list[Section] = []
    pending: list[Paragraph] = []
    for sec in kept:
        if sum(len(p.text) for p in sec.paragraphs) < min_section_chars:
            pending.extend(sec.paragraphs)
            continue
        merged.append(Section(sec.heading, sec.kind, pending + sec.paragraphs))
        pending = []
    if pending:
        if merged:
            merged[-1].paragraphs.extend(pending)
        else:
            merged.append(Section("", "other", pending))

    abstract = [p for p in doc.abstract if alpha_ratio(p.text) >= min_alpha]
    return Document(doc.key, doc.title, abstract, merged, doc.figures, doc.extraction, doc.n_bib)


# ─── Document → chunks ───────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    type: str                  # "text" | "caption"
    section: str
    section_raw: str
    page_first: int
    page_last: int
    seq: int
    figure_id: str = ""
    figure_kind: str = ""
    figure_label: str = ""


def chunk_document(doc: Document, target: int, hard_max: int,
                   min_chars: int = 200, min_alpha: float = 0.6) -> list[Chunk]:
    doc = filter_document(doc, min_alpha=min_alpha)
    chunks: list[Chunk] = []

    def emit_text(paragraphs, kind, heading):
        for text, p0, p1 in pack_windows(paragraphs, target, hard_max):
            if len(text) < min_chars or alpha_ratio(text) < min_alpha:
                continue
            chunks.append(Chunk(text, "text", kind, heading, p0, p1, len(chunks)))

    if doc.abstract:
        emit_text(doc.abstract, "abstract", "")
    for sec in doc.sections:
        emit_text(sec.paragraphs, sec.kind, sec.heading)

    for fig in doc.figures:
        text = f"{fig.label}: {fig.caption}" if fig.caption else fig.label
        if not text.strip():
            continue
        chunks.append(Chunk(text, "caption", "caption", fig.label, fig.page, fig.page, len(chunks),
                            figure_id=fig.id, figure_kind=fig.kind, figure_label=fig.label))
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_chunking.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/chunking.py tests/test_chunking.py
git commit -m "feat(chunking): section filters, short-section merge, and Document → Chunk with captions and seq"
```

---

### Task 8: `shared/figures.py` — crop, context, describe; revised prompt

**Files:**
- Create: `research_assistant/shared/figures.py`
- Modify: `research_assistant/prompts.py:80-86` (`FIGURE_DESCRIPTION`)
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: `extract.Figure` (Task 4); `shared.llm.chat(messages, images=..., temperature=...)` (existing).
- Produces:
  ```python
  DESCRIPTION_PREFIX = "Auto-generated description of {label} — verify values against the figure: "
  def crop_name(document_key: str, fig: Figure, index: int) -> str        # "<doc>_p<page>_f<index>.png"
  def crop_figure(pdf_path: str, fig: Figure, out_path: str, dpi: int) -> bool
  def figure_context(fig: Figure, max_refs: int = 4) -> str
  def describe_crop(image_path: str, fig: Figure, context: str) -> str     # raw model text, no prefix
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_figures.py
"""Figures: crop from GROBID's box, describe with the right context.

The v1 path gave the VLM whatever text block sat below a Detectron2 crop —
30% of the time not the caption. Here the context is the caption plus the
sentences that cite the figure, and the output is marked as generated.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from research_assistant.shared import figures as fg
from research_assistant.shared.extract import Figure


def _fig(**kw):
    base = dict(id="fig_0", kind="figure", label="Figure 1", caption="Photoconductivity dynamics.",
                page=2, bbox=(50.0, 379.0, 542.0, 643.0), refs=[])
    base.update(kw)
    return Figure(**base)


class TestContext(unittest.TestCase):
    def test_caption_plus_citing_sentences(self):
        f = _fig(refs=["We find a 10% increase (Figure 1b).", "Figure 1c shows the decay."])
        ctx = fg.figure_context(f)
        self.assertTrue(ctx.startswith("Caption: Figure 1: Photoconductivity dynamics."))
        self.assertIn("Referenced in the text:", ctx)
        self.assertIn("- We find a 10% increase (Figure 1b).", ctx)

    def test_refs_are_capped(self):
        f = _fig(refs=[f"Ref {i}." for i in range(10)])
        self.assertEqual(fg.figure_context(f, max_refs=3).count("\n- "), 3)

    def test_no_refs(self):
        self.assertNotIn("Referenced", fg.figure_context(_fig()))


class TestCropName(unittest.TestCase):
    def test_matches_v1_naming(self):
        self.assertEqual(fg.crop_name("doi_10.1_x.pdf", _fig(page=4), 2), "doi_10.1_x.pdf_p4_f2.png")


class TestCrop(unittest.TestCase):
    def _fake_fitz(self, saved):
        class _Pix:
            def save(self, path):
                saved.append(path)

        class _Page:
            rect = types.SimpleNamespace(width=600.0, height=800.0)

            def get_pixmap(self, clip=None, dpi=72):
                saved.append(("clip", clip.x0, clip.y0, clip.x1, clip.y1))
                return _Pix()

        fitz = types.ModuleType("fitz")
        fitz.Rect = lambda x0, y0, x1, y1: types.SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1)
        fitz.open = lambda path: [_Page(), _Page(), _Page()]
        return patch.dict(sys.modules, {"fitz": fitz})

    def test_crops_with_padding_clamped_to_the_page(self):
        saved = []
        with self._fake_fitz(saved), tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "x.png")
            ok = fg.crop_figure("/x/p.pdf", _fig(bbox=(590.0, 5.0, 620.0, 100.0)), out, dpi=72)
        self.assertTrue(ok)
        self.assertEqual(saved[0], ("clip", 580.0, 0.0, 600.0, 110.0))     # 10pt pad, clamped
        self.assertEqual(saved[1], out)

    def test_no_bbox_means_no_crop(self):
        with self._fake_fitz([]):
            self.assertFalse(fg.crop_figure("/x/p.pdf", _fig(bbox=None), "/x/out.png", dpi=72))

    def test_page_out_of_range_means_no_crop(self):
        with self._fake_fitz([]):
            self.assertFalse(fg.crop_figure("/x/p.pdf", _fig(page=7), "/x/out.png", dpi=72))


class TestDescribe(unittest.TestCase):
    def test_routes_through_shared_llm_with_the_image_and_temperature_zero(self):
        seen = {}

        def fake_chat(messages, images=None, temperature=None, **kw):
            seen.update(messages=messages, images=images, temperature=temperature)
            return types.SimpleNamespace(content="Two peaks at ±1.")

        import research_assistant.shared.llm as llm_mod
        with patch.object(llm_mod, "chat", fake_chat):
            out = fg.describe_crop("/x/crop.png", _fig(), "Caption: ...")
        self.assertEqual(out, "Two peaks at ±1.")
        self.assertEqual(seen["images"], ["/x/crop.png"])
        self.assertEqual(seen["temperature"], 0.0)
        self.assertIn("Caption: ...", seen["messages"][-1]["content"])
        self.assertIn("figure", seen["messages"][-1]["content"].lower())

    def test_prefix_names_the_figure(self):
        self.assertEqual(fg.DESCRIPTION_PREFIX.format(label="Table 2"),
                         "Auto-generated description of Table 2 — verify values against the figure: ")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_figures.py -v`
Expected: FAIL — `ImportError: cannot import name 'figures'`.

- [ ] **Step 3: Revise the prompt**

In `research_assistant/prompts.py`, replace the `FIGURE_DESCRIPTION` block:

```python
# ─── Figure description (ingestion VLM pass — only when CITATION_FIGURE_VLM=1) ─
FIGURE_DESCRIPTION = (
    "You are analysing scientific plots. Describe this {fig_type}. "
    "Extract textual information, data and trends.\n\n"
    "Surrounding Document Context:\n{context}. Answer in 3-5 sentences at max."
)
```

with:

```python
# ─── Figure description (v2 ingestion, when figure analysis is on for the run) ─
# The context is the figure's own caption plus the sentences in the paper that
# refer to it. Numbers are the failure mode: a 2B model reading a plot placed
# peaks at ±0.5 where the axis showed ±1.0. So: describe what is shown, quote a
# value only when it is legible, and say when it is not.
FIGURE_DESCRIPTION = (
    "You are reading a {fig_type} from a physics paper. Using the image and the "
    "context below, describe in 3-5 sentences what it shows: the quantities on "
    "each axis or in each column, the qualitative behaviour (trends, peaks, "
    "crossovers, comparisons between curves or rows), and what the paper uses "
    "it to establish. Quote a numerical value only if you can read it directly "
    "from an axis tick, a label or a table cell; otherwise describe the "
    "behaviour without numbers. If part of the image is unreadable, say so. "
    "Do not repeat the caption verbatim.\n\n{context}"
)
```

- [ ] **Step 4: Write the module**

```python
# research_assistant/shared/figures.py
"""Figures and tables in the v2 ingest: crop from GROBID's box, describe with
the caption and the sentences that cite it.

Nothing here decides *whether* to describe — that is the run's switch,
threaded down from the CLI/UI to process_pdf_v2(). This module only knows how.
"""

from __future__ import annotations

import os

from research_assistant.prompts import FIGURE_DESCRIPTION
from research_assistant.shared.extract import Figure
from research_assistant.shared.log import get_logger

logger = get_logger("figures")

DESCRIPTION_PREFIX = "Auto-generated description of {label} — verify values against the figure: "
_PAD_PT = 10.0


def crop_name(document_key: str, fig: Figure, index: int) -> str:
    """Same shape as the v1 layout path wrote: <doc>_p<page>_f<i>.png."""
    return f"{document_key}_p{fig.page}_f{index}.png"


def crop_figure(pdf_path: str, fig: Figure, out_path: str, dpi: int) -> bool:
    """Render the figure's box (plus a small pad, clamped to the page) to PNG.
    False when there is nothing to crop; never raises for a missing box."""
    if not fig.bbox:
        return False
    import fitz

    pdf = fitz.open(pdf_path)
    if fig.page < 0 or fig.page >= len(pdf):
        return False
    page = pdf[fig.page]
    x0, y0, x1, y1 = fig.bbox
    clip = fitz.Rect(
        max(0.0, x0 - _PAD_PT), max(0.0, y0 - _PAD_PT),
        min(page.rect.width, x1 + _PAD_PT), min(page.rect.height, y1 + _PAD_PT),
    )
    if clip.x1 <= clip.x0 or clip.y1 <= clip.y0:
        return False
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    page.get_pixmap(clip=clip, dpi=dpi).save(out_path)
    return True


def figure_context(fig: Figure, max_refs: int = 4) -> str:
    caption = f"{fig.label}: {fig.caption}" if fig.caption else fig.label
    lines = [f"Caption: {caption}"]
    if fig.refs:
        lines.append("Referenced in the text:")
        lines.extend(f"- {r}" for r in fig.refs[:max_refs])
    return "\n".join(lines)


def describe_crop(image_path: str, fig: Figure, context: str) -> str:
    """One model call. Raw text back; the caller adds DESCRIPTION_PREFIX."""
    from research_assistant.shared.llm import chat

    prompt = FIGURE_DESCRIPTION.format(fig_type=fig.kind, context=context)
    return chat([{"role": "user", "content": prompt}], images=[image_path], temperature=0.0).content.strip()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_figures.py tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/figures.py research_assistant/prompts.py tests/test_figures.py
git commit -m "feat(figures): crop from GROBID coords, caption+refs context, cautious description prompt"
```

---

### Task 9: `shared/ingest_v2.py` — `process_pdf_v2()`

**Files:**
- Create: `research_assistant/shared/ingest_v2.py`
- Test: `tests/test_ingest_v2.py`

**Interfaces:**
- Consumes: `extract.extract` (Task 5); `chunking.chunk_document` (Task 7); `figures.crop_name/crop_figure/figure_context/describe_crop/DESCRIPTION_PREFIX` (Task 8); `config.CHUNK_TARGET_CHARS/CHUNK_MAX_CHARS/CHUNK_MIN_CHARS/CHUNK_MIN_ALPHA/IMAGES_DIR/PDF_RENDER_DPI/SUMMARY_MAX_CHARS` (Task 1 + existing); `ingestion.pdf_key` (existing).
- Produces: `process_pdf_v2(pdf_path: str, citation_string: str, describe_figures: bool = False, images_dir: str | None = None) -> list[dict]`. Every entry is a dict `upsert_corpus()` accepts (Task 10): keys `document, citation, page, type, content, extraction`, plus `embed_text` (text/caption/description entries) and `meta` (flat metadata). Types emitted: `text_chunk`, `caption`, `figure_description`, `summary_source`. `EXTRACTION_MODES_V2 = ("grobid", "pymupdf")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest_v2.py
"""process_pdf_v2: Document → corpus entries.

What matters is the contract with upsert_corpus (Task 10): pre-chunked
entries carry the stored text in `content`, the text to embed in
`embed_text`, and flat metadata in `meta`; captions are always emitted; a
description entry appears only when the run's switch is on; one
summary_source entry per document feeds the stage-1 index.
"""

import unittest
from unittest.mock import patch

from research_assistant.shared import ingest_v2 as iv
from research_assistant.shared.extract import Document, Section, Paragraph, Figure


def _sentence(i):
    return f"Sentence number {i} has about sixty characters of text in it, yes."


def _doc():
    return Document("paper.pdf", "A Title", [Paragraph(" ".join(_sentence(i) for i in range(4)), 0)], [
        Section("1. Introduction", "introduction", [Paragraph(" ".join(_sentence(i) for i in range(12)), 0)]),
        Section("4. Conclusion", "conclusion", [Paragraph(" ".join(_sentence(i) for i in range(6)), 5)]),
    ], [
        Figure("fig_0", "figure", "Figure 1", "Density of states.", 1, (0, 0, 10, 10), ["Seen in Figure 1."]),
    ], "grobid", n_bib=12)


class V2TestCase(unittest.TestCase):
    def setUp(self):
        p = patch.object(iv, "extract", return_value=_doc()); p.start(); self.addCleanup(p.stop)
        p = patch.object(iv, "crop_figure", return_value=True); p.start(); self.addCleanup(p.stop)
        describe = patch.object(iv, "describe_crop", return_value="Two sharp peaks.")
        self.describe_mock = describe.start(); self.addCleanup(describe.stop)


class TestTextEntries(V2TestCase):
    def test_shape_and_header(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=False, images_dir="/tmp/img")
        text = [e for e in entries if e["type"] == "text_chunk"]
        self.assertGreater(len(text), 1)
        e = text[0]
        self.assertEqual(e["document"], "paper.pdf")
        self.assertEqual(e["citation"], "A Title")
        self.assertEqual(e["extraction"], "grobid")
        self.assertEqual(e["page"], e["meta"]["page_first"])
        self.assertTrue(e["embed_text"].startswith("Title: A Title. Section: "))
        self.assertTrue(e["embed_text"].endswith(e["content"]))
        self.assertEqual(set(e["meta"]), {"section", "section_raw", "page_first", "page_last", "seq", "extraction"})
        self.assertEqual(e["meta"]["extraction"], "grobid")

    def test_seq_runs_over_text_then_captions_and_summary_source_is_last(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        seqs = [e["meta"]["seq"] for e in entries if "meta" in e]
        self.assertEqual(seqs, list(range(len(seqs))))
        self.assertEqual(entries[-1]["type"], "summary_source")
        self.assertEqual(entries[-2]["type"], "caption")


class TestCaptionsAndDescriptions(V2TestCase):
    def test_caption_always_present_with_crop_and_described_false(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=False, images_dir="/tmp/img")
        caps = [e for e in entries if e["type"] == "caption"]
        self.assertEqual(len(caps), 1)
        c = caps[0]
        self.assertEqual(c["content"], "Figure 1: Density of states.")
        self.assertEqual(c["meta"]["figure_id"], "fig_0")
        self.assertEqual(c["meta"]["figure_kind"], "figure")
        self.assertEqual(c["meta"]["image_path"], "paper.pdf_p1_f0.png")
        self.assertFalse(c["meta"]["described"])
        self.assertEqual([e for e in entries if e["type"] == "figure_description"], [])
        self.describe_mock.assert_not_called()

    def test_switch_on_describes_every_figure_inline(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
        desc = [e for e in entries if e["type"] == "figure_description"]
        self.assertEqual(len(desc), 1)
        d = desc[0]
        self.assertEqual(d["content"], "Auto-generated description of Figure 1 — verify values against the figure: Two sharp peaks.")
        self.assertEqual(d["meta"]["extraction"], "vlm")
        self.assertEqual(d["meta"]["figure_id"], "fig_0")
        self.assertEqual(d["extraction"], "grobid")          # per-document mode for the run report
        self.assertTrue(d["embed_text"].startswith("Title: A Title. Section: Figure 1."))
        cap = [e for e in entries if e["type"] == "caption"][0]
        self.assertTrue(cap["meta"]["described"])

    def test_description_context_is_caption_plus_refs(self):
        with patch.object(iv, "describe_crop", return_value="x") as d:
            iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
        ctx = d.call_args.args[2]
        self.assertIn("Caption: Figure 1: Density of states.", ctx)
        self.assertIn("- Seen in Figure 1.", ctx)

    def test_description_failure_leaves_caption_undescribed_and_continues(self):
        with patch.object(iv, "describe_crop", side_effect=RuntimeError("model down")):
            entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
        self.assertEqual([e for e in entries if e["type"] == "figure_description"], [])
        self.assertFalse([e for e in entries if e["type"] == "caption"][0]["meta"]["described"])
        self.assertTrue(any(e["type"] == "text_chunk" for e in entries))

    def test_no_crop_means_no_description_but_caption_stays(self):
        with patch.object(iv, "crop_figure", return_value=False):
            entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", describe_figures=True, images_dir="/tmp/img")
        cap = [e for e in entries if e["type"] == "caption"][0]
        self.assertEqual(cap["meta"]["image_path"], "")
        self.assertEqual([e for e in entries if e["type"] == "figure_description"], [])


class TestSummarySource(V2TestCase):
    def test_abstract_intro_conclusion_in_that_order(self):
        entries = iv.process_pdf_v2("/x/paper.pdf", "A Title", images_dir="/tmp/img")
        src = [e for e in entries if e["type"] == "summary_source"]
        self.assertEqual(len(src), 1)
        text = src[0]["content"]
        self.assertLess(text.index("Sentence number 0"), text.index("Sentence number 11"))
        self.assertIn("Sentence number 5 has", text)              # conclusion included
        self.assertEqual(src[0]["document"], "paper.pdf")


class TestPyMuPDFDocument(V2TestCase):
    def test_no_figures_no_descriptions_mode_recorded(self):
        doc = Document("paper.pdf", "paper.pdf", [], [Section("", "other", [Paragraph(" ".join(_sentence(i) for i in range(8)), 0)])], [], "pymupdf")
        with patch.object(iv, "extract", return_value=doc):
            entries = iv.process_pdf_v2("/x/paper.pdf", "L", describe_figures=True, images_dir="/tmp/img")
        self.assertTrue(all(e["extraction"] == "pymupdf" for e in entries))
        self.assertEqual([e["type"] for e in entries if e["type"] != "text_chunk"], ["summary_source"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_ingest_v2.py -v`
Expected: FAIL — `ImportError: cannot import name 'ingest_v2'`.

- [ ] **Step 3: Write the module**

```python
# research_assistant/shared/ingest_v2.py
"""PDF → corpus entries, v2.

extract() → chunk_document() → entries in the dict shape upsert_corpus()
consumes, so the lock, dedupe, id allocation, manifest and BM25 rebuild are
shared with v1 rather than copied. Three entry types are new and carry
`embed_text` (what is embedded) apart from `content` (what is stored, quoted,
cited and verified), and flat `meta`.

Figure analysis is a per-run switch (spec §4.6): when on, every figure and
table with a crop is described here, inline, before this function returns.
"""

from __future__ import annotations

import os

from research_assistant.config import (
    CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_MIN_ALPHA,
    IMAGES_DIR, PDF_RENDER_DPI, SUMMARY_MAX_CHARS,
)
from research_assistant.shared.chunking import chunk_document
from research_assistant.shared.extract import extract, Document
from research_assistant.shared.figures import (
    DESCRIPTION_PREFIX, crop_name, crop_figure, figure_context, describe_crop,
)
from research_assistant.shared.log import get_logger

logger = get_logger("ingest_v2")

EXTRACTION_MODES_V2 = ("grobid", "pymupdf")


def _pdf_key(pdf_path: str) -> str:
    # Same identity as ingestion.pdf_key(); repeated here to avoid importing
    # ingestion (which imports this module).
    return os.path.basename(pdf_path).strip().replace(" ", "_").lower()


def _header(title: str, section_label: str) -> str:
    return f"Title: {title}. Section: {section_label}. "


def _summary_source(doc: Document) -> str:
    parts = [p.text for p in doc.abstract]
    for kind in ("introduction", "conclusion"):
        for sec in doc.sections:
            if sec.kind == kind:
                parts.extend(p.text for p in sec.paragraphs)
    if not parts:                                   # PyMuPDF path: kind "other" only
        parts = [p.text for s in doc.sections for p in s.paragraphs]
    return "\n\n".join(parts)[:SUMMARY_MAX_CHARS]


def process_pdf_v2(pdf_path: str, citation_string: str, describe_figures: bool = False,
                   images_dir: str | None = None) -> list[dict]:
    images_dir = images_dir or IMAGES_DIR
    key = _pdf_key(pdf_path)
    doc = extract(pdf_path, key=key)
    title = doc.title or citation_string
    chunks = chunk_document(doc, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS,
                            min_chars=CHUNK_MIN_CHARS, min_alpha=CHUNK_MIN_ALPHA)
    figures = {f.id: f for f in doc.figures}

    entries: list[dict] = []
    described = 0
    fig_index = 0
    for c in chunks:
        base = {
            "document": key, "citation": citation_string, "page": c.page_first,
            "extraction": doc.extraction,
        }
        meta = {
            "section": c.section, "section_raw": c.section_raw,
            "page_first": c.page_first, "page_last": c.page_last, "seq": c.seq,
            "extraction": doc.extraction,
        }
        if c.type == "text":
            entries.append({**base, "type": "text_chunk", "content": c.text,
                            "embed_text": _header(title, c.section_raw or c.section) + c.text,
                            "meta": meta})
            continue

        # caption chunk — always; crop — cheap, always attempted
        fig = figures.get(c.figure_id)
        image_path = ""
        if fig is not None:
            name = crop_name(key, fig, fig_index)
            fig_index += 1
            try:
                if crop_figure(pdf_path, fig, os.path.join(images_dir, name), PDF_RENDER_DPI):
                    image_path = name
            except Exception as exc:                # noqa: BLE001
                logger.warning("Crop failed for %s %s: %s", key, fig.label, exc)

        cap_meta = {**meta, "figure_id": c.figure_id, "figure_kind": c.figure_kind,
                    "figure_label": c.figure_label, "image_path": image_path, "described": False}
        cap_entry = {**base, "type": "caption", "content": c.text,
                     "embed_text": _header(title, c.figure_label) + c.text, "meta": cap_meta}
        entries.append(cap_entry)

        if describe_figures and fig is not None and image_path:
            context = figure_context(fig)
            try:
                text = describe_crop(os.path.join(images_dir, image_path), fig, context)
            except Exception as exc:                # noqa: BLE001
                logger.error("Description failed for %s %s: %s", key, fig.label, exc)
                continue
            if not text:
                continue
            content = DESCRIPTION_PREFIX.format(label=fig.label) + text
            entries.append({**base, "type": "figure_description", "content": content,
                            "embed_text": _header(title, c.figure_label) + content,
                            "meta": {**cap_meta, "extraction": "vlm", "described": True}})
            cap_meta["described"] = True
            described += 1
            logger.info("  described %s (%d/%d)", fig.label, described, len(doc.figures))

    entries.append({"document": key, "citation": citation_string, "page": 0,
                    "type": "summary_source", "content": _summary_source(doc),
                    "extraction": doc.extraction})

    logger.info("✓ %s: %s, %d text chunk(s), %d caption(s), %d described, %d bib entries kept out",
                key, doc.extraction, sum(e["type"] == "text_chunk" for e in entries),
                sum(e["type"] == "caption" for e in entries), described, doc.n_bib)
    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_ingest_v2.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/ingest_v2.py tests/test_ingest_v2.py
git commit -m "feat(ingest): process_pdf_v2 — extract, chunk, caption, crop, describe when the run says so"
```

---

### Task 10: `upsert_corpus()` accepts pre-chunked entries, `embed_text`, `meta`, `summary_source`

**Files:**
- Modify: `research_assistant/shared/ingestion.py:576-725` (`upsert_corpus`), `:524-575` (`upsert_summaries` embeds with `embed_documents`)
- Test: `tests/test_upsert.py` (extend)

**Interfaces:**
- Consumes: entry dicts from Task 9.
- Produces: `upsert_corpus(corpus)` behaviour: entries whose `type` is in `PRECHUNKED_TYPES = {"text_chunk", "caption", "figure_description"}` bypass the chunker; `entry["embed_text"]` (if present) is what is embedded, `entry["content"]` is what is stored; `entry["meta"]` is merged into Chroma metadata unprefixed; `type == "summary_source"` sets that document's summary input and is not indexed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_upsert.py`:

```python


class TestPrechunkedEntries(UpsertTestCase):
    """v2 hands upsert_corpus chunks it has already made."""

    def _chunk(self, document, content, type="text_chunk", embed_text=None, meta=None):
        return {
            "document": document, "citation": f"{document} title", "page": 2, "type": type,
            "content": content, "embed_text": embed_text, "meta": meta or {},
        }

    def test_prechunked_text_is_stored_verbatim_not_rechunked(self):
        content = "Para one.\n\nPara two."             # the fake chunker would split on the blank line
        ing.upsert_corpus([self._chunk("a.pdf", content)])
        self.assertEqual(self.collection.chunks_for("a.pdf"), [content])

    def test_embed_text_is_embedded_and_content_is_stored(self):
        seen = []
        import research_assistant.shared.llm as llm_mod
        emb = types.SimpleNamespace(embed_documents=lambda texts: (seen.extend(texts), [[0.0] for _ in texts])[1],
                                    embed_query=lambda t: [0.0])
        with patch.object(llm_mod, "get_embeddings", lambda *a, **k: emb):
            ing.upsert_corpus([self._chunk("a.pdf", "stored text", embed_text="Title: T. Section: S. stored text")])
        self.assertEqual(seen, ["Title: T. Section: S. stored text"])
        self.assertEqual(self.collection.documents, ["stored text"])

    def test_meta_is_merged_unprefixed(self):
        ing.upsert_corpus([self._chunk("a.pdf", "x" * 30, meta={"section": "methods", "seq": 3, "described": False})])
        m = self.collection.metadatas[0]
        self.assertEqual((m["section"], m["seq"], m["described"]), ("methods", 3, False))
        self.assertEqual(m["type"], "text_chunk")
        self.assertEqual(m["citation_source"], "a.pdf title")

    def test_summary_source_feeds_the_summary_index_and_is_not_stored(self):
        seen = {}
        with patch.object(ing, "upsert_summaries", lambda per_doc_text, per_doc_citation: seen.update(per_doc_text) or 0):
            ing.upsert_corpus([
                self._chunk("a.pdf", "body chunk " * 5),
                {"document": "a.pdf", "citation": "a.pdf title", "page": 0, "type": "summary_source",
                 "content": "abstract + intro + conclusion"},
            ])
        self.assertEqual(seen, {"a.pdf": "abstract + intro + conclusion"})
        self.assertEqual(self.collection.chunks_for("a.pdf"), ["body chunk " * 5])

    def test_v1_entries_still_go_through_the_chunker(self):
        ing.upsert_corpus([self._entry("a.pdf", "Para one.\n\nPara two.")])
        self.assertEqual(self.collection.chunks_for("a.pdf"), ["Para one.", "Para two."])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_upsert.py -v`
Expected: FAIL — `test_prechunked_text_is_stored_verbatim_not_rechunked` finds the entry ignored (type `text_chunk` is neither `figure/table` nor `text`), others fail on missing metadata keys.

- [ ] **Step 3: Edit `upsert_corpus()`**

Add near the top of `ingestion.py`, after `EXTRACTION_MODES = ("layout", "text_only")`:

```python
# v2 entries arrive already chunked; these pass straight through to the store.
PRECHUNKED_TYPES = ("text_chunk", "caption", "figure_description")
```

In `upsert_corpus()`, replace the grouping loop:

```python
    for entry in corpus:
        if entry["type"] in ["figure", "table"]:
            total_corpus.append(entry)
        elif entry["type"] == "text":
            content = clean_text(entry["content"])
            if len(content) < CHUNK_MIN_LENGTH:
                continue
            key = (entry["document"], entry["citation"], entry["page"])
            page_text_groups.setdefault(key, []).append(content)

    # Per-document full text, for the summary index (populated as we chunk).
    per_doc_text, per_doc_citation = {}, {}
```

with:

```python
    # Per-document text for the summary index. v1 accumulates it from page
    # groups below; v2 sends one explicit summary_source entry per document.
    per_doc_text, per_doc_citation = {}, {}

    for entry in corpus:
        if entry["type"] in ["figure", "table"] or entry["type"] in PRECHUNKED_TYPES:
            total_corpus.append(entry)
        elif entry["type"] == "summary_source":
            per_doc_text[entry["document"]] = entry.get("content") or ""
            per_doc_citation.setdefault(entry["document"], entry["citation"])
        elif entry["type"] == "text":
            content = clean_text(entry["content"])
            if len(content) < CHUNK_MIN_LENGTH:
                continue
            key = (entry["document"], entry["citation"], entry["page"])
            page_text_groups.setdefault(key, []).append(content)
```

Then in the same function, in the page-group loop, change:

```python
        per_doc_text[doc] = per_doc_text.get(doc, "") + "\n\n" + merged
```

to:

```python
        if doc not in per_doc_text or not per_doc_text[doc].strip():
            per_doc_text[doc] = ""
        per_doc_text[doc] = per_doc_text[doc] + "\n\n" + merged
```

Replace the entry → record loop:

```python
    current_index = get_max_chunk_index(collection)
    documents, metadatas, ids, seen = [], [], [], set()

    for entry in total_corpus:
        content = entry["content"].strip()
        # Compare on the same form that reaches the store: add() truncates to
        # EMBED_MAX_CHARS, so comparing full text against stored text would
        # never match for a long chunk and would re-insert it on every run.
        stored_form = content[:EMBED_MAX_CHARS]
        identity = (entry["document"], stored_form)
        if identity in seen or identity in existing_chunks or len(content) < CHUNK_MIN_LENGTH:
            continue
        seen.add(identity)
        meta = {
            "document": entry["document"],
            "page": entry["page"],
            "type": entry["type"],
            "citation_source": entry["citation"],
        }
        if "metadata" in entry:
            for k, v in entry["metadata"].items():
                meta[f"extra_{k}"] = str(v)
        documents.append(content)
        metadatas.append(meta)
        ids.append(f"chunk_{current_index}")
        current_index += 1
```

with:

```python
    current_index = get_max_chunk_index(collection)
    documents, embed_texts, metadatas, ids, seen = [], [], [], [], set()

    for entry in total_corpus:
        content = entry["content"].strip()
        # Compare on the same form that reaches the store: add() truncates to
        # EMBED_MAX_CHARS, so comparing full text against stored text would
        # never match for a long chunk and would re-insert it on every run.
        stored_form = content[:EMBED_MAX_CHARS]
        identity = (entry["document"], stored_form)
        if identity in seen or identity in existing_chunks or len(content) < CHUNK_MIN_LENGTH:
            continue
        seen.add(identity)
        meta = {
            "document": entry["document"],
            "page": entry["page"],
            "type": entry["type"],
            "citation_source": entry["citation"],
        }
        if "metadata" in entry:
            for k, v in entry["metadata"].items():
                meta[f"extra_{k}"] = str(v)
        # v2: flat metadata under its own names (section, seq, figure_id, …).
        # Chroma accepts str / int / float / bool values only.
        for k, v in (entry.get("meta") or {}).items():
            meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        documents.append(stored_form)
        # v2 embeds a contextual header + text; v1 embeds the stored text.
        embed_texts.append((entry.get("embed_text") or content)[:EMBED_MAX_CHARS])
        metadatas.append(meta)
        ids.append(f"chunk_{current_index}")
        current_index += 1
```

and the add loop:

```python
        for i in range(0, len(documents), EMBED_BATCH_SIZE):
            b_docs = [d[:EMBED_MAX_CHARS] for d in documents[i : i + EMBED_BATCH_SIZE]]
            collection.add(
                embeddings=embeddings.embed_documents(b_docs),
                documents=b_docs,
                metadatas=metadatas[i : i + EMBED_BATCH_SIZE],
                ids=ids[i : i + EMBED_BATCH_SIZE],
            )
```

with:

```python
        for i in range(0, len(documents), EMBED_BATCH_SIZE):
            collection.add(
                embeddings=embeddings.embed_documents(embed_texts[i : i + EMBED_BATCH_SIZE]),
                documents=documents[i : i + EMBED_BATCH_SIZE],
                metadatas=metadatas[i : i + EMBED_BATCH_SIZE],
                ids=ids[i : i + EMBED_BATCH_SIZE],
            )
```

- [ ] **Step 4: `upsert_summaries()` embeds a summary as a document**

In `upsert_summaries()`, change:

```python
            embeddings=[embeddings.embed_query(summary)],
```

to:

```python
            # A stored summary is a document, not a query: under the v2 nomic
            # prefixes the two are embedded differently. Identical for v1.
            embeddings=embeddings.embed_documents([summary]),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_upsert.py tests/test_ingestion.py -v`
Expected: PASS (all existing upsert tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/ingestion.py tests/test_upsert.py
git commit -m "feat(ingestion): upsert_corpus takes pre-chunked entries with embed_text, flat meta and a summary_source"
```

---

### Task 11: `process_pdf()` dispatches on the version; `describe_figures` threaded through `ingest_pdfs()`; run report

**Files:**
- Modify: `research_assistant/shared/ingestion.py:304-344` (`process_pdf`), `:792-933` (`ingest_pdfs`, `_ingest_pdfs_locked`), imports
- Test: `tests/test_ingestion.py` (extend)

**Interfaces:**
- Consumes: `config.INDEX_VERSION`, `config.FIGURE_VLM` (existing); `ingest_v2.process_pdf_v2`, `ingest_v2.EXTRACTION_MODES_V2` (Task 9).
- Produces: `process_pdf(pdf_path, citation_string, detectron_weights=None, images_dir=None, detectron_config=None, describe_figures=None)`; `ingest_pdfs(pdfs, workers=1, skip_ingested=True, rebuild_index=True, log_prefix="", describe_figures=None)` — `None` means `config.FIGURE_VLM`. `result["extraction"]` gains `grobid` / `pymupdf` counts under v2; `result["described"]` = number of description entries inserted.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ingestion.py`:

```python


class TestVersionDispatch(ManifestBackedTestCase):
    """Under INDEX_VERSION=2 the v2 processor runs, with the run's figure switch."""

    def _v2_entries(self, path, label, describe_figures=False, images_dir=None):
        key = ing.pdf_key(path)
        out = [{"document": key, "citation": label, "page": 0, "type": "text_chunk",
                "content": "c" * 300, "embed_text": "h c", "meta": {"seq": 0}, "extraction": "grobid"}]
        if describe_figures:
            out.append({"document": key, "citation": label, "page": 0, "type": "figure_description",
                        "content": "d" * 300, "embed_text": "h d", "meta": {"extraction": "vlm"},
                        "extraction": "grobid"})
        return out

    def test_v2_uses_process_pdf_v2_and_reports_modes(self):
        pdf = _touch(self.tmp, "a.pdf")
        with patch.object(ing, "INDEX_VERSION", 2), \
             patch("research_assistant.shared.ingest_v2.process_pdf_v2", side_effect=self._v2_entries) as v2, \
             patch.object(ing, "upsert_corpus", return_value=1), patch.object(ing, "rebuild_bm25"):
            result = ing.ingest_pdfs({pdf: "A"}, describe_figures=True)
        v2.assert_called_once()
        self.assertTrue(v2.call_args.kwargs["describe_figures"])
        self.assertEqual(result["extraction"], {"grobid": 1, "pymupdf": 0})
        self.assertEqual(result["described"], 1)

    def test_describe_defaults_to_config_figure_vlm(self):
        pdf = _touch(self.tmp, "a.pdf")
        with patch.object(ing, "INDEX_VERSION", 2), patch.object(ing, "FIGURE_VLM", True), \
             patch("research_assistant.shared.ingest_v2.process_pdf_v2", side_effect=self._v2_entries) as v2, \
             patch.object(ing, "upsert_corpus", return_value=1), patch.object(ing, "rebuild_bm25"):
            ing.ingest_pdfs({pdf: "A"})
        self.assertTrue(v2.call_args.kwargs["describe_figures"])

    def test_v1_path_is_untouched(self):
        pdf = _touch(self.tmp, "a.pdf")
        with patch.object(ing, "INDEX_VERSION", 1), patch.object(ing, "LAYOUT_DETECTION", False), \
             patch.object(ing, "_extract_text_only", return_value=[]) as v1, \
             patch("research_assistant.shared.ingest_v2.process_pdf_v2") as v2, \
             patch.object(ing, "rebuild_bm25"):
            result = ing.ingest_pdfs({pdf: "A"})
        v1.assert_called_once()
        v2.assert_not_called()
        self.assertEqual(result["extraction"], {"layout": 0, "text_only": 0})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_ingestion.py -v -k VersionDispatch`
Expected: FAIL — `TypeError: ingest_pdfs() got an unexpected keyword argument 'describe_figures'`; `AttributeError: ... has no attribute 'INDEX_VERSION'`.

- [ ] **Step 3: Edit `ingestion.py`**

Add `INDEX_VERSION` to the `from research_assistant.config import (...)` block.

Replace `process_pdf()` with:

```python
def process_pdf(
    pdf_path: str,
    citation_string: str,
    detectron_weights=None,
    images_dir=None,
    detectron_config=None,
    describe_figures=None,
):
    """Extract a single PDF into corpus entries.

    INDEX_VERSION >= 2: GROBID full-text extraction with the PyMuPDF fallback,
    sentence-window chunks, captions, crops, and — when the run's switch is on
    — a VLM description of every figure and table (shared.ingest_v2).

    INDEX_VERSION 1: with ``config.LAYOUT_DETECTION`` on, Detectron2 layout
    detection plus a VLM description of every figure and table; if that is
    unavailable or fails it falls back to text-only PyMuPDF. With it off:
    text-only. Both return the same ``list[dict]`` shape.

    Every entry is tagged with the mode that produced it so ingest_pdfs() can
    report how the batch was actually extracted — see _tag_extraction().
    """
    if INDEX_VERSION >= 2:
        from research_assistant.shared.ingest_v2 import process_pdf_v2

        if describe_figures is None:
            describe_figures = FIGURE_VLM
        return process_pdf_v2(pdf_path, citation_string,
                              describe_figures=describe_figures, images_dir=images_dir)

    if not LAYOUT_DETECTION:
        return _tag_extraction(_extract_text_only(pdf_path, citation_string), "text_only")
    try:
        return _tag_extraction(
            _process_pdf_layout(
                pdf_path,
                citation_string,
                detectron_weights,
                images_dir,
                detectron_config=detectron_config,
            ),
            "layout",
        )
    except Exception as e:
        logger.warning(
            "Layout detection failed (%s) — falling back to text-only extraction "
            "for %s. Set CITATION_LAYOUT_DETECTION=0 to silence this.",
            e, os.path.basename(pdf_path),
        )
        return _tag_extraction(_extract_text_only(pdf_path, citation_string), "text_only")
```

Change the `ingest_pdfs` signature and body:

```python
def ingest_pdfs(
    pdfs,
    workers: int = 1,
    skip_ingested: bool = True,
    rebuild_index: bool = True,
    log_prefix: str = "",
    describe_figures: bool | None = None,
) -> dict:
```

add to its docstring Args:

```
        describe_figures: v2 only. The run's figure-analysis switch: describe
                       every figure and table in every PDF of this batch with
                       the model, inline. None → config.FIGURE_VLM. Ignored
                       under INDEX_VERSION 1.
```

and change its last line to:

```python
    with _ingest_lock():
        return _ingest_pdfs_locked(pdfs, workers, skip_ingested, rebuild_index, log_prefix, describe_figures)
```

Change `_ingest_pdfs_locked`'s signature to `def _ingest_pdfs_locked(pdfs, workers, skip_ingested, rebuild_index, log_prefix, describe_figures=None) -> dict:` and, right after `result = {...}`, add:

```python
    if describe_figures is None:
        describe_figures = FIGURE_VLM
    if INDEX_VERSION >= 2:
        logger.info("%sIndex v%d — figure analysis %s for this run.",
                    log_prefix, INDEX_VERSION, "ON" if describe_figures else "off")
```

Change both `process_pdf` invocations:

```python
            entries = process_pdf(path, label)
```
→
```python
            entries = process_pdf(path, label, describe_figures=describe_figures)
```

and

```python
                executor.submit(process_pdf, path, label): path
```
→
```python
                executor.submit(process_pdf, path, label, None, None, None, describe_figures): path
```

Replace the extraction report block (from `by_mode = {mode: set() for mode in EXTRACTION_MODES}` through the `if LAYOUT_DETECTION and by_mode["text_only"]:` warning) with:

```python
    if INDEX_VERSION >= 2:
        from research_assistant.shared.ingest_v2 import EXTRACTION_MODES_V2 as modes
    else:
        modes = EXTRACTION_MODES
    by_mode = {mode: set() for mode in modes}
    for entry in corpus:
        docs = by_mode.get(entry.get("extraction"))
        if docs is not None and entry.get("document"):
            docs.add(entry["document"])
    result["extraction"] = {mode: len(docs) for mode, docs in by_mode.items()}
    result["described"] = sum(1 for e in corpus if e.get("type") == "figure_description")

    if INDEX_VERSION >= 2 and by_mode["pymupdf"]:
        logger.warning(
            "%s%d of %d document(s) fell back to PyMuPDF extraction — no sections, "
            "captions or figures for them, and the bibliography may be in the text. "
            "Check GROBID at the configured GROBID_SERVER.",
            log_prefix, len(by_mode["pymupdf"]), result["processed"],
        )
    elif INDEX_VERSION == 1 and LAYOUT_DETECTION and by_mode["text_only"]:
        logger.warning(
            "%s%d of %d document(s) fell back to text-only extraction — no figures "
            "or tables were indexed for them. See the warnings above for why layout "
            "detection failed, or set CITATION_LAYOUT_DETECTION=0 if text-only is "
            "what you intended.",
            log_prefix, len(by_mode["text_only"]), result["processed"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_ingestion.py tests/test_upsert.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/ingestion.py tests/test_ingestion.py
git commit -m "feat(ingestion): process_pdf dispatches on INDEX_VERSION; describe_figures is a per-run switch on ingest_pdfs"
```

---

### Task 12: nomic task prefixes on the embedder (v2 only)

**Files:**
- Modify: `research_assistant/shared/llm.py:229-253` (`get_embeddings`), imports
- Test: `tests/test_llm.py` (extend)

**Interfaces:**
- Consumes: `config.INDEX_VERSION` (Task 1).
- Produces: `get_embeddings()` returns the backend object wrapped in `_PrefixedEmbeddings(inner, "search_document: ", "search_query: ")` when `INDEX_VERSION >= 2` and the model name starts with `nomic-embed`; `llm.NOMIC_DOC_PREFIX`, `llm.NOMIC_QUERY_PREFIX`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python


class TestNomicPrefixes(unittest.TestCase):
    """nomic-embed-text expects task prefixes; v1 vectors were made without
    them, so the wrapper applies only under INDEX_VERSION >= 2."""

    def setUp(self):
        llm._embeddings_singleton = None
        self.addCleanup(setattr, llm, "_embeddings_singleton", None)
        self.calls = []
        inner = types.SimpleNamespace(
            embed_documents=lambda texts: (self.calls.append(("docs", list(texts))), [[0.0]] * len(texts))[1],
            embed_query=lambda text: (self.calls.append(("query", text)), [0.0])[1],
        )
        ollama_mod = types.ModuleType("langchain_ollama")
        ollama_mod.OllamaEmbeddings = lambda model: inner
        p = patch.dict("sys.modules", {"langchain_ollama": ollama_mod}); p.start(); self.addCleanup(p.stop)
        p = patch.object(llm, "EMBED_BACKEND", "ollama"); p.start(); self.addCleanup(p.stop)
        p = patch.object(llm, "EMBED_MODEL", "nomic-embed-text"); p.start(); self.addCleanup(p.stop)

    def test_v2_prefixes_documents_and_queries(self):
        with patch.object(llm, "INDEX_VERSION", 2):
            emb = llm.get_embeddings()
            emb.embed_documents(["a", "b"])
            emb.embed_query("q")
        self.assertEqual(self.calls, [("docs", ["search_document: a", "search_document: b"]),
                                      ("query", "search_query: q")])

    def test_v1_is_unprefixed(self):
        with patch.object(llm, "INDEX_VERSION", 1):
            emb = llm.get_embeddings()
            emb.embed_documents(["a"])
            emb.embed_query("q")
        self.assertEqual(self.calls, [("docs", ["a"]), ("query", "q")])

    def test_non_nomic_model_is_unprefixed_even_on_v2(self):
        with patch.object(llm, "INDEX_VERSION", 2), patch.object(llm, "EMBED_MODEL", "mxbai-embed-large"):
            emb = llm.get_embeddings()
            emb.embed_query("q")
        self.assertEqual(self.calls, [("query", "q")])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_llm.py -v -k Nomic`
Expected: FAIL — `AttributeError: module ... has no attribute 'INDEX_VERSION'` / calls unprefixed.

- [ ] **Step 3: Edit `llm.py`**

Add `INDEX_VERSION` to the config import at the top of `llm.py` (it already imports `EMBED_BACKEND`, `EMBED_MODEL`, …).

Before `_embeddings_singleton = None` add:

```python
NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "


class _PrefixedEmbeddings:
    """nomic-embed-text's task prefixes. Applied only to a v2 index: v1 vectors
    were computed without them, and prefixed queries against unprefixed
    documents are worse than neither."""

    def __init__(self, inner, doc_prefix, query_prefix):
        self._inner = inner
        self._dp = doc_prefix
        self._qp = query_prefix

    def embed_documents(self, texts):
        return self._inner.embed_documents([self._dp + t for t in texts])

    def embed_query(self, text):
        return self._inner.embed_query(self._qp + text)
```

At the end of `get_embeddings()`, before `return _embeddings_singleton`, inside the `if _embeddings_singleton is None:` block (after the backend branches):

```python
        if INDEX_VERSION >= 2 and model.startswith("nomic-embed"):
            logger.info("Embeddings: nomic task prefixes on (index v%d)", INDEX_VERSION)
            _embeddings_singleton = _PrefixedEmbeddings(
                _embeddings_singleton, NOMIC_DOC_PREFIX, NOMIC_QUERY_PREFIX
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/llm.py tests/test_llm.py
git commit -m "feat(llm): nomic search_document/search_query prefixes on the v2 index only"
```

---

### Task 13: Agent 8 never uses VLM text as evidence — `exclude_types` on `hybrid_search`

**Files:**
- Modify: `research_assistant/shared/search.py` (`hybrid_search` signature, sparse post-filter, dense `where`)
- Modify: `research_assistant/agents/agent8_verifier.py:241-244`
- Test: `tests/test_search.py` (extend), `tests/test_verifier.py` (extend)

**Interfaces:**
- Produces: `hybrid_search(..., doc_filter=None, exclude_types=None)` — results whose `metadata["type"]` is in `exclude_types` are removed from both retrievers; the dense `where` combines `document $in` and `type $nin` with `$and`. Agent 8 passes `exclude_types={"figure_description"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python


class TestExcludeTypes(unittest.TestCase):
    """A verdict must rest on what the paper says, not on a model's reading of
    a plot: Agent 8 excludes figure_description chunks from evidence."""

    class _Coll(FakeCollection):
        def __init__(self, ids):
            super().__init__(ids)
            self.where = None

        def query(self, **kwargs):
            self.where = kwargs.get("where")
            return super().query(**kwargs)

    def test_sparse_side_drops_excluded_types(self):
        texts = ["a", "b", "c"]
        metas = [{"document": "d.pdf", "type": "text_chunk"},
                 {"document": "d.pdf", "type": "figure_description"},
                 {"document": "d.pdf", "type": "caption"}]
        out = hybrid_search("q", FakeCollection([]), FakeBM25([1.0, 3.0, 2.0]), texts, metas,
                            top_k=3, embeddings_model=FakeEmbeddings(), exclude_types={"figure_description"})
        self.assertEqual([r["chunk_index"] for r in out], [2, 0])

    def test_dense_where_combines_document_and_type_filters(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(), doc_filter={"doc0.pdf"}, exclude_types={"figure_description"})
        self.assertEqual(coll.where, {"$and": [{"document": {"$in": ["doc0.pdf"]}},
                                               {"type": {"$nin": ["figure_description"]}}]})

    def test_dense_where_is_none_when_nothing_is_filtered(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1, embeddings_model=FakeEmbeddings())
        self.assertIsNone(coll.where)
```

Append to `tests/test_verifier.py` (it already defines `VerifyDraftTestCase` with a `_run(draft, mapping, rows, hits, search_side_effect=...)` helper that patches `agent8_verifier.hybrid_search` and `judge`):

```python


class TestVerifierExcludesVlmText(VerifyDraftTestCase):
    """A verdict rests on the paper's text: figure_description chunks are
    never retrieved as evidence."""

    def test_hybrid_search_is_called_with_exclude_types(self):
        seen = {}
        hits = [{"text": "Graphene is highly conductive.", "metadata": {"document": "a.pdf"}}]

        def record(*args, **kwargs):
            seen.update(kwargs)
            return hits

        self._run(
            "Graphene conducts well \\cite{cite_1}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            hits,
            search_side_effect=record,
        )
        self.assertEqual(seen["exclude_types"], {"figure_description"})
        self.assertEqual(seen["doc_filter"], {"a.pdf"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_search.py -v -k ExcludeTypes`
Expected: FAIL — `TypeError: hybrid_search() got an unexpected keyword argument 'exclude_types'`.

- [ ] **Step 3: Edit `search.py`**

Change the signature:

```python
    embeddings_model=None,
    doc_filter: set | None = None,
) -> list[dict]:
```
→
```python
    embeddings_model=None,
    doc_filter: set | None = None,
    exclude_types: set | None = None,
) -> list[dict]:
```

Add to the docstring Args:

```
        exclude_types:    Chunk ``metadata["type"]`` values to leave out of
                          both retrievers (e.g. ``{"figure_description"}``
                          — Agent 8 never takes a model's reading of a plot
                          as evidence).
```

Replace the sparse post-filter:

```python
    if doc_filter:
        sparse_top_indices = [
            i for i in sparse_top_indices
            if (metadatas[i] if i < len(metadatas) else {}).get("document") in doc_filter
        ]
```

with:

```python
    def _meta(i):
        return metadatas[i] if i < len(metadatas) else {}

    if doc_filter:
        sparse_top_indices = [i for i in sparse_top_indices if _meta(i).get("document") in doc_filter]
    if exclude_types:
        sparse_top_indices = [i for i in sparse_top_indices if _meta(i).get("type") not in exclude_types]
```

Replace the dense `where=` argument:

```python
        where={"document": {"$in": list(doc_filter)}} if doc_filter else None,
```

with:

```python
        where=_where(doc_filter, exclude_types),
```

and add above `hybrid_search`:

```python
def _where(doc_filter, exclude_types):
    """ChromaDB where-clause for the dense side. Two conditions need $and;
    one stands alone; none is None (Chroma rejects an empty dict)."""
    clauses = []
    if doc_filter:
        clauses.append({"document": {"$in": list(doc_filter)}})
    if exclude_types:
        clauses.append({"type": {"$nin": sorted(exclude_types)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
```

- [ ] **Step 4: Edit `agent8_verifier.py`**

At line 241–244, change:

```python
            hits = hybrid_search(
                entry["claim"], collection, bm25, texts, metadatas,
                top_k=top_k, doc_filter=documents,
            )
```

to:

```python
            # A verdict rests on what the paper says. A figure_description is
            # a model's reading of a plot (v2, spec §4.6) and is never evidence.
            hits = hybrid_search(
                entry["claim"], collection, bm25, texts, metadatas,
                top_k=top_k, doc_filter=documents,
                exclude_types={"figure_description"},
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_search.py tests/test_verifier.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/search.py research_assistant/agents/agent8_verifier.py tests/test_search.py tests/test_verifier.py
git commit -m "feat(verify): Agent 8 excludes VLM figure descriptions from evidence via hybrid_search exclude_types"
```

---

### Task 14: The switch — CLI flags, orchestrator state, Tab 1 checkbox

**Files:**
- Modify: `research_assistant/agents/agent3_ingestor.py:39-93` (`run_ingestor`, CLI)
- Modify: `research_assistant/agents/agent6_manual_ingestor.py:40-68`, `:113-132` (`ingest_manual_pdf`, CLI)
- Modify: `orchestrate.py:57-75` (`PipelineState`), `:128-135` (`ingest_seed`), `:159-164` (`ingest_refs`), `:235-262` (`run`), `:288-315` (CLI)
- Modify: `app.py:562-577` (upload form), `:609` (batch ingest call), `:688-702` (search form), `:717-724` (pipeline inputs)
- Test: `tests/test_orchestrate_upload.py` (extend; check its fixtures for how the graph is driven)

**Interfaces:**
- Produces: `agent3_ingestor.run_ingestor(workers=1, force=False, describe_figures=None)`; `agent6_manual_ingestor.ingest_manual_pdf(pdf_path, citation_string=None, workers=1, describe_figures=None)`; `PipelineState.describe_figures: Optional[bool]`; `orchestrate.run(..., describe_figures=None)`; CLI flag `--describe-figures` on all three; Tab 1 toggle "Analyse figures and tables with the model" in both forms, default `config.FIGURE_VLM`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrate_upload.py`:

```python


class TestDescribeFiguresFlowsToIngest(unittest.TestCase):
    def test_ingest_seed_passes_the_switch(self):
        import orchestrate
        seen = {}
        with patch.object(orchestrate, "ingest_pdfs", lambda pdfs, **kw: seen.update(kw) or {}):
            orchestrate.ingest_seed({"seed_path": "/x/a.pdf", "seed_label": "A", "workers": 1,
                                     "force": False, "describe_figures": True})
        self.assertTrue(seen["describe_figures"])

    def test_ingest_refs_passes_the_switch(self):
        import orchestrate
        seen = {}
        with patch.object(orchestrate.agent3_ingestor, "run_ingestor", lambda **kw: seen.update(kw)):
            orchestrate.ingest_refs({"workers": 2, "force": False, "describe_figures": False})
        self.assertEqual(seen, {"workers": 2, "force": False, "describe_figures": False})

    def test_absent_key_means_none_so_config_decides(self):
        import orchestrate
        seen = {}
        with patch.object(orchestrate, "ingest_pdfs", lambda pdfs, **kw: seen.update(kw) or {}):
            orchestrate.ingest_seed({"seed_path": "/x/a.pdf", "seed_label": "A"})
        self.assertIsNone(seen["describe_figures"])
```

(Add `import unittest` / `from unittest.mock import patch` at the top of that file if not already present.)

- [ ] **Step 2: Run test to verify it fails**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_orchestrate_upload.py -v -k DescribeFigures`
Expected: FAIL — `KeyError: 'describe_figures'`.

- [ ] **Step 3: Agent 3**

In `agent3_ingestor.py`, change `run_ingestor`'s signature and the `ingest_pdfs` call:

```python
def run_ingestor(workers: int = 1, force: bool = False, describe_figures: bool | None = None):
```

docstring Args add: `describe_figures: v2 — analyse every figure and table with the model for this run (None → config).`

```python
    result = ingest_pdfs(pdfs, workers=workers, skip_ingested=not force, describe_figures=describe_figures)
```

and after the `--force` argument:

```python
    parser.add_argument(
        "--describe-figures",
        action="store_true",
        default=None,
        help="Index v2: describe every figure and table with the model during this run "
             "(one choice for the whole run; ~1 minute per figure on CPU). Default: CITATION_FIGURE_VLM.",
    )
```

and `run_ingestor(workers=args.workers, force=args.force, describe_figures=args.describe_figures)`.

- [ ] **Step 4: Agent 6**

In `agent6_manual_ingestor.py`, `ingest_manual_pdf(pdf_path, citation_string=None, workers=1, describe_figures=None)`; call `ingest_pdfs({pdf_path: citation_string}, workers=workers, describe_figures=describe_figures)`; add the same `--describe-figures` argument to `main()` and pass `describe_figures=args.describe_figures` in the `--once` branch. The watcher branch (`ManualPDFHandler`) leaves it `None` so config decides.

- [ ] **Step 5: Orchestrator**

In `orchestrate.py`, add to `PipelineState`:

```python
    # v2: the run's figure-analysis switch. None → config.FIGURE_VLM.
    describe_figures: Optional[bool]
```

`ingest_seed`:

```python
    ingest_pdfs(
        {state["seed_path"]: state["seed_label"]},
        workers=state.get("workers", 1),
        skip_ingested=not state.get("force", False),
        describe_figures=state.get("describe_figures"),
    )
```

`ingest_refs`:

```python
    agent3_ingestor.run_ingestor(
        workers=state.get("workers", 1), force=state.get("force", False),
        describe_figures=state.get("describe_figures"),
    )
```

`run()` gains `describe_figures: bool | None = None` and puts `"describe_figures": describe_figures` in the stream input dict. The CLI gains:

```python
    parser.add_argument(
        "--describe-figures", action="store_true", default=None,
        help="Index v2: analyse every figure and table with the model during this run.",
    )
```

and passes `describe_figures=args.describe_figures` to `run(...)`.

- [ ] **Step 6: The UI**

In `app.py`, upload form — replace:

```python
            c1, c2 = st.columns(2)
            ask = c1.toggle("Answer my query at the end", value=True)
            force = c2.toggle("Force re-run every stage", value=False)
            submitted = st.form_submit_button("Process and Index Paper(s)", type="primary")
```

with:

```python
            c1, c2, c3 = st.columns(3)
            ask = c1.toggle("Answer my query at the end", value=True)
            force = c2.toggle("Force re-run every stage", value=False)
            describe_figures = c3.toggle(
                "Analyse figures and tables with the model",
                value=config.FIGURE_VLM,
                help="One choice for this whole run: every figure and table in every paper "
                     "is described by the model and the description joins the corpus. "
                     "Adds roughly a minute per figure on CPU.",
            )
            submitted = st.form_submit_button("Process and Index Paper(s)", type="primary")
```

Batch path — change:

```python
                        ingest_res = ingest_pdfs(candidates, workers=1, skip_ingested=not force)
```
to
```python
                        ingest_res = ingest_pdfs(candidates, workers=1, skip_ingested=not force,
                                                 describe_figures=describe_figures)
                        if ingest_res.get("described"):
                            st.write(f"🖼️ Described {ingest_res['described']} figure(s)/table(s).")
```

Search form — the same three-column toggle block replacing its `c1, c2 = st.columns(2)` block (button label `"Build corpus"` unchanged). Also initialise `describe_figures = config.FIGURE_VLM` next to `force = False` at line 559 so the variable exists on every path.

Pipeline inputs — add `"describe_figures": describe_figures,` to the `inputs` dict at line 717.

- [ ] **Step 7: Run tests and the import sweep locally**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/ -v` — Expected: PASS.
Run: `CITATION_LOG_FILE=0 python -c "import app, orchestrate; import research_assistant.agents.agent3_ingestor, research_assistant.agents.agent6_manual_ingestor"` — Expected: no output, exit 0.

- [ ] **Step 8: Commit**

```bash
git add research_assistant/agents/agent3_ingestor.py research_assistant/agents/agent6_manual_ingestor.py orchestrate.py app.py tests/test_orchestrate_upload.py
git commit -m "feat: --describe-figures / Tab 1 toggle — one figure-analysis switch per run, threaded to ingestion"
```

---

### Task 15: CI import sweep, docs

**Files:**
- Modify: `.github/workflows/tests.yml:59-72` (module list)
- Modify: `PIPELINE.md` (Agent 3 row, layout tree, configuration notes), `ARCHITECTURE.md` (new §4.7–4.9 under Ingestion), `README.md` ("What he's worst at" — one sentence noting v2 exists behind a flag)

- [ ] **Step 1: CI sweep**

In `.github/workflows/tests.yml`, add to the `for m in ...` list after `research_assistant.shared.ingestion`:

```
                   research_assistant.shared.tokenize research_assistant.shared.extract \
                   research_assistant.shared.chunking research_assistant.shared.figures \
                   research_assistant.shared.ingest_v2 \
```

and add `research_assistant.agents.agent8_verifier` after `agent7_research_chat` (it was missing).

- [ ] **Step 2: `PIPELINE.md`**

In the Agent 3 row of the pipeline table, append: *"Under `CITATION_INDEX_VERSION=2`: GROBID full-text extraction (PyMuPDF fallback), sentence-window chunks (~1,200 chars, one-sentence overlap, within sections), figure/table captions from GROBID with crops from its coordinates, and — when `--describe-figures` / the Tab 1 toggle is on for the run — a model description of every figure and table, stored as its own `figure_description` chunk marked as generated. Writes to `physics_papers_v2` / `physics_summaries_v2` / `bm25_index_v2.pkl` / `ingested_v2.json`; v1 is never touched."*

In the layout tree under `shared/`, add lines for `tokenize.py`, `extract.py`, `chunking.py`, `figures.py`, `ingest_v2.py` with one-phrase descriptions matching the module docstrings.

In "Configuration notes", add a bullet: `CITATION_INDEX_VERSION` (1 default; 2 selects the v2 index and extractor), `CITATION_CHUNK_TARGET_CHARS`, `CITATION_CHUNK_MAX_CHARS`, `GROBID_FULLTEXT_TIMEOUT`, `CITATION_FIGURE_VLM` (now the default state of the per-run switch).

- [ ] **Step 3: `ARCHITECTURE.md`**

After §4.6 add:

```markdown
### 4.7 v2: GROBID full text, then sentence windows

Under `CITATION_INDEX_VERSION=2` every PDF goes through GROBID's
`processFulltextDocument` (TEI cached in `raw/grobid_output/`), and the
chunker packs `pysbd` sentences into ~1,200-character windows within a
section, with one sentence of overlap. PyMuPDF is the fallback, recorded per
chunk as `extraction=pymupdf`.

**Why.** On the v1 corpus the median chunk was 242 characters and 45% were
under 200 — chopped bibliography entries and font-map garbage, which BM25's
length normalisation ranks above real passages. GROBID keeps the
bibliography out (the heading heuristic found it on 1 of 3 papers; GROBID
on 4 of 4), de-hyphenates (`quan- tum`), normalises ligatures (`ﬁeld`, in
31% of v1 chunks and invisible to a query for "field"), and yields sections
and captions PyMuPDF cannot. Windows give the 4k-window generator ~1,800
tokens of evidence from six chunks instead of ~400.

**Trade-off.** 6–11 s of GROBID per paper at ingest; formulas are dropped
from paragraph text (they were glyph soup inline); a broken-font PDF is
garbage under both paths and is dropped by the alphabetic-ratio filter.

### 4.8 What is embedded is not what is stored

The stored chunk is the window verbatim — quoted, cited, verified. The
embedded text is `search_document: Title: … Section: … <window>`: nomic's
task prefix plus a contextual header. Queries get `search_query:`. Prefixes
apply to the v2 index only; v1's vectors were made without them and mixing
is worse than neither.

### 4.9 Figures: GROBID captions, coordinate crops, descriptions as a run's choice

Captions come from GROBID's `<figDesc>` for figures *and* tables (v1's
`find_caption` had no table branch — all 548 tables were placeholders — and
attached body text to 30% of figures). Crops are PyMuPDF renders of GROBID's
`<graphic coords>` box; no Detectron2. Whether every figure is then described
by the model is one switch per run (`--describe-figures`, the Tab 1 toggle,
default `CITATION_FIGURE_VLM`) — never a per-image choice — and the
description is a separate `figure_description` chunk whose text opens with
"Auto-generated description of … verify values against the figure".

**Why the marker and the exclusion.** In testing, the model placed a plot's
peaks at ±0.5 where the axis read ±1.0. Descriptions are retrievable for
synthesis and chat; Agent 8 excludes them as evidence (`exclude_types`), so
a citation verdict rests on the paper's text.
```

- [ ] **Step 4: `README.md`**

In "What he's worst at", after the bullet list and before "We will give him depth.", add one sentence: *"The first of those is being fixed: `CITATION_INDEX_VERSION=2` builds a second index from GROBID full text with proper chunks and captions, beside the old one — see `PIPELINE.md`."*

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/tests.yml PIPELINE.md ARCHITECTURE.md README.md
git commit -m "docs: ingestion v2 in PIPELINE, ARCHITECTURE §4.7–4.9, README; CI sweeps the new modules"
```

---

### Task 16: Build the v2 index locally and verify it

This task is the live check. It needs GROBID on `localhost:8070` (`curl localhost:8070/api/isalive` → `true`) and Ollama with `gemma4:e2b` and `nomic-embed-text` (`curl localhost:11434/api/tags`). It does not touch the VM. It writes only v2 files under `data/`.

**Files:**
- Create: `scripts/index_stats.py` (small, kept — a maintainer will want it again)

- [ ] **Step 1: The stats script**

```python
# scripts/index_stats.py
"""Chunk-length distribution and type counts for the active index.

    CITATION_INDEX_VERSION=2 python scripts/index_stats.py
"""
import collections
import statistics as st

import chromadb

from research_assistant.config import VECTORDB_PATH, COLLECTION_NAME, INDEX_VERSION

col = chromadb.PersistentClient(path=VECTORDB_PATH).get_collection(COLLECTION_NAME)
lens, types, docs, modes, described = [], collections.Counter(), set(), collections.Counter(), 0
off = 0
while True:
    b = col.get(include=["documents", "metadatas"], limit=5000, offset=off)
    if not b["ids"]:
        break
    for d, m in zip(b["documents"], b["metadatas"]):
        lens.append(len(d)); types[m.get("type")] += 1; docs.add(m.get("document"))
        modes[m.get("extraction", "?")] += 1
        if m.get("type") == "caption" and m.get("described"):
            described += 1
    off += 5000
lens.sort()
q = lambda p: lens[int(p * (len(lens) - 1))] if lens else 0
print(f"index v{INDEX_VERSION} ({COLLECTION_NAME}): {len(lens):,} chunks over {len(docs)} docs")
print(f"chars  p10 {q(.1)}  p25 {q(.25)}  median {q(.5)}  p75 {q(.75)}  p90 {q(.9)}  max {lens[-1] if lens else 0}")
print(f"< 200 chars: {sum(l < 200 for l in lens) / max(1, len(lens)):.0%}")
print("types:", dict(types))
print("extraction:", dict(modes), "| captions described:", described)
```

- [ ] **Step 2: Baseline (v1) numbers for the record**

Run: `CITATION_LOG_FILE=0 python scripts/index_stats.py`
Expected: `index v1 (physics_papers): 36,609 chunks over 323 docs` and `median 242`, `< 200 chars: 45%` (the diagnosis numbers). Paste the output into the commit message of Step 7.

- [ ] **Step 3: One paper, figures ON — proves the whole path**

Run:

```bash
CITATION_LOG_FILE=0 CITATION_INDEX_VERSION=2 python -m research_assistant.agents.agent6_manual_ingestor \
  --once data/pulled_pdfs/doi_10.1002_adma.202211157.pdf --citation "Covalent MoS2 networks" --describe-figures
```

Expected in the log: `Index v2 — figure analysis ON for this run.`, `✓ doi_10.1002_adma.202211157.pdf: grobid, N text chunk(s), 11 caption(s), 11 described, 53 bib entries kept out` (caption/bib counts from the spike; N ≈ 45–55), and `BM25 index rebuilt (..., tokenizer v2)`. Wall-clock: ~6 minutes (11 figures × ~33 s). Then:

```bash
CITATION_LOG_FILE=0 CITATION_INDEX_VERSION=2 python scripts/index_stats.py
ls data/images | grep adma.202211157 | wc -l          # expect 11 crops (5 with <graphic>, 6 from figure coords)
ls data/raw/grobid_output/ | grep adma.202211157      # expect the cached TEI
ls data/ingested_v2.json data/bm25_index_v2.pkl       # expect both; data/ingested.json untouched (check mtime)
```

Open two of the crops and read two of the descriptions out of Chroma:

```bash
CITATION_LOG_FILE=0 CITATION_INDEX_VERSION=2 python - <<'EOF'
import chromadb
from research_assistant.config import VECTORDB_PATH, COLLECTION_NAME
col = chromadb.PersistentClient(path=VECTORDB_PATH).get_collection(COLLECTION_NAME)
b = col.get(where={"type": "figure_description"}, include=["documents", "metadatas"], limit=2)
for d, m in zip(b["documents"], b["metadatas"]):
    print(m["figure_label"], "|", m["image_path"], "\n", d, "\n")
EOF
```

Expected: each description starts with `Auto-generated description of Figure N — verify values against the figure:` and reads as a description of *that* figure (compare against the crop you opened). If a description is about the wrong figure, the crop/caption pairing in `process_pdf_v2` is wrong — stop and fix before the full run.

- [ ] **Step 4: Full corpus, figures OFF**

Run (this is 323 papers × ~6–11 s GROBID + embedding; expect 45–90 minutes; run in the background and check the log):

```bash
CITATION_LOG_FILE=0 CITATION_INDEX_VERSION=2 nohup python -m research_assistant.agents.agent3_ingestor --workers 1 \
  > /tmp/claude-1000/-run-media-shardul-storage1-research-assistant/cb5fde9b-3785-4318-8666-5ac0f1eb2bd7/scratchpad/v2_ingest.log 2>&1 &
```

Watch: `tail -f <that log>`. Expected at the end: `Done. Processed 322, skipped 1, inserted ...` (one skipped: the paper from Step 3) and a fallback warning naming the papers GROBID could not parse (the garbled-font paper at least).

- [ ] **Step 5: Compare**

Run: `CITATION_LOG_FILE=0 CITATION_INDEX_VERSION=2 python scripts/index_stats.py`
Expected, per the spec's estimate: roughly 8–14k chunks, median ~1,000–1,200 chars, `< 200 chars` under 5% (captions), types `{text_chunk, caption, figure_description}`, extraction mostly `grobid` with a handful of `pymupdf`. Record the actual numbers.

Also confirm v1 is untouched: `python scripts/index_stats.py` (no version env) prints exactly the Step 2 numbers, and `stat -c %y data/ingested.json data/bm25_index.pkl` shows mtimes from before this task.

- [ ] **Step 6: The app, on v2**

Run: `CITATION_LOG_FILE=0 CITATION_INDEX_VERSION=2 streamlit run app.py` and, in the browser:

1. Tab 1, *Search for a paper* form: the new toggle is present, default off. Do not build.
2. Tab 2: cite `"Anderson localization suppresses diffusion in one-dimensional disordered wires."` — expect a `\cite{}` and passages that are full paragraphs, not fragments; no bibliography text in the passages.
3. Tab 4: ask `"What does the MoS2 covalent-network paper say about hopping transport?"` — expect an answer citing that paper and at least one source passage of several sentences; if a figure description is among the sources it carries the "Auto-generated description" marker.
4. Tab 3: paste two sentences, cite, then Verify — expect verdicts, and no `figure_description` text in any evidence span.

Record what you saw for each tab in the commit message.

- [ ] **Step 7: Commit the script and the numbers**

```bash
git add scripts/index_stats.py
git commit -m "chore: index_stats.py; v2 index built locally

v1: <paste Step 2 output>
v2: <paste Step 5 output>
Tabs 2/3/4 on v2: <one line each>"
```

---

## Self-review

**Spec coverage.** §3.5 → Tasks 2, 3. §3.6 → Task 12. §4.1 → Tasks 4, 5. §4.2 → Task 7 (`filter_document`, `is_running_header` in Task 4). §4.3 → Task 6, 7. §4.4 → Task 9 (`_header`), Task 10 (`embed_text`), Task 12 (prefix). §4.5 → Tasks 7, 9, 10. §4.6 → Tasks 8, 9, 11, 13, 14. §4.7 → Task 9 (`_summary_source`), Task 10 (`summary_source` entry). §4.8 → Task 1 (names), Task 11 (dispatch), Task 16 (v1 untouched check). §6 config → Task 1 (`CITATION_FIGURE_VLM` as the toggle default: Task 14). §8 error handling: GROBID down → Task 5; pysbd → Task 6; VLM failure → Task 9; tokenizer mismatch → Task 3; v2 collection absent → existing `db.py` path with the versioned name (Task 1). §9 tests → every task. Not in this plan, by decision: §2 harness, §3.1–3.4, §5.

**Type consistency.** `Document/Section/Paragraph/Figure` (Task 4) are used unchanged in Tasks 5–9. `Chunk` (Task 7) fields `figure_id/figure_kind/figure_label/page_first/page_last/seq/section/section_raw` are what Task 9 reads. Entry dict keys from Task 9 (`content`, `embed_text`, `meta`, `type`, `extraction`, `page`) are what Task 10 consumes and Task 11 counts. `describe_figures` is `bool | None` at every layer (Tasks 9, 11, 14) with `None → config.FIGURE_VLM` resolved once, in `_ingest_pdfs_locked`. `tokenizer_version` attribute (Task 3) is read by name in `search.py`. `exclude_types` (Task 13) is a `set`.

**Placeholders.** None. The chunker (Task 6–7) and the TEI parser (Task 4) were run against their planned tests before the plan was written — both pass, the chunker with and without `pysbd`.
