# Merged Research Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one project at `/Users/shardul/Downloads/research_assistant` from `tech_ireland` (infrastructure) and `citation_builder` (features), restructured as an installable package with typed stage contracts.

**Architecture:** `tech_ireland`'s `shared/` layer is the base — it is a strict improvement on `citation_builder`'s in every module the two have in common. `citation_builder` supplies agents 5–7 and the test suite, ported onto `shared.llm` so no agent bypasses the backend abstraction. Agent 1 combines both extraction strategies (GROBID primary, regex fallback) so the pipeline degrades instead of stopping. Everything moves into a `research_assistant` package with three thin entry points at the root.

**Tech Stack:** Python 3.12, ChromaDB, rank-bm25, LangGraph, Streamlit, GROBID, PyMuPDF, optional Detectron2/layoutparser, Ollama or any OpenAI-compatible endpoint.

**Spec:** `docs/superpowers/specs/2026-09-07-merged-research-assistant-design.md`

## Global Constraints

- **No git.** The user chose no version control. Every task ends with a **Checkpoint** step (run the tests) instead of a commit. Nothing in this plan runs `git`. *(If you want revertable tasks, say so and I'll `git init` first — the plan is unaffected otherwise.)*
- **Source repos are read-only.** `/Users/shardul/Downloads/tech_ireland` and `/Users/shardul/Downloads/citation_builder` are copied *from*, never modified.
- **No corpus or weights are copied.** `data/` starts empty. `model_final.pth` is a symlink to `/Users/shardul/Downloads/tech_ireland/model_final.pth`.
- **Import convention:** absolute, always `research_assistant.…`. No relative imports.
- **Heavy imports stay function-local.** ChromaDB, torch, detectron2/layoutparser, OpenCV, PyMuPDF and the LangChain embedding stack are imported *inside* the functions that use them. This is load-bearing: CI installs only `requirements-test.txt`, and module-scope imports make the suite uncollectable there.
- **After this merge, no module outside `research_assistant/shared/llm.py` imports `ollama` or `openai`.**
- **Test command:** `CITATION_LOG_FILE=0 python -m pytest tests/ -v`
- **Import sweep** (the CI gate that catches broken imports no test covers):
  ```bash
  CITATION_LOG_FILE=0 python -c "
  import importlib
  for m in ['research_assistant.config','research_assistant.prompts','research_assistant.schemas','research_assistant.shared.log','research_assistant.shared.retry','research_assistant.shared.manifest','research_assistant.shared.llm','research_assistant.shared.db','research_assistant.shared.search','research_assistant.shared.source_key','research_assistant.shared.fetch','research_assistant.shared.ingestion','research_assistant.agents.agent1_extractor','research_assistant.agents.agent2_fetcher','research_assistant.agents.agent3_ingestor']:
      importlib.import_module(m); print('ok', m)"
  ```

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | package metadata, editable install, pytest config |
| `research_assistant/config.py` | every model name, path, tunable; env parsing |
| `research_assistant/prompts.py` | every prompt sent to a model |
| `research_assistant/schemas.py` | typed contracts between stages |
| `research_assistant/shared/manifest.py` | what has been ingested |
| `research_assistant/shared/llm.py` | backend-agnostic chat / stream / embeddings |
| `research_assistant/shared/ingestion.py` | process → upsert → mark → index |
| `research_assistant/shared/{search,retrieve}.py` | hybrid search; two-stage retrieval |
| `research_assistant/shared/{db,fetch,source_key,log,retry}.py` | infrastructure |
| `research_assistant/agents/agent0…agent7*.py` | the pipeline stages |
| `orchestrate.py` / `watch.py` / `app.py` | entry points: per-query / daemon / UI |
| `tests/` | the ported suite plus two new files |

---

### Task 1: Scaffold, packaging, and config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `research_assistant/__init__.py`, `research_assistant/agents/__init__.py`, `research_assistant/shared/__init__.py`, `tests/__init__.py`
- Create: `research_assistant/config.py`, `research_assistant/shared/log.py`, `research_assistant/shared/retry.py`, `research_assistant/prompts.py`
- Create: `requirements.txt`, `requirements-layout.txt`, `requirements-test.txt`
- Create: `model_final.pth` (symlink), `publaynet_config.yaml`

**Interfaces:**
- Produces: `research_assistant.config` module attributes (`DATA_DIR`, `PROJECT_ROOT`, `VECTORDB_PATH`, `COLLECTION_NAME`, `SUMMARY_COLLECTION_NAME`, `BM25_INDEX_PATH`, `RAW_DIR`, `DRAFTS_DIR`, `PULLED_PDFS_DIR`, `IMAGES_DIR`, all `*_PATH` manifests, `LLM_BACKEND`, `EMBED_BACKEND`, `LLM_MODEL`, `CHAT_MODEL`, `EMBED_MODEL`, `CHAT_OLLAMA_OPTIONS`, `LAYOUT_DETECTION`, `FIGURE_VLM`, `DETECTRON_*`, `RRF_K`, `DEFAULT_TOP_K`, `DOC_SELECT_K`, `DOC_GATE`, `GROBID_SERVER`, `GROBID_BATCH_CONCURRENCY`, `SEARCH_PROVIDERS`, `SEARCH_LIMIT`, `CITATION_CHECK_BATCH_SIZE`, cooldowns); `research_assistant.shared.log.get_logger(name)`; `research_assistant.shared.retry.retry(max_retries, backoff)`
- `_env_bool(name, default) -> bool`, `_env_int(name, default) -> int` are private to config

- [ ] **Step 1: Create the directory tree and package markers**

```bash
cd /Users/shardul/Downloads/research_assistant
mkdir -p research_assistant/agents research_assistant/shared tests .github/workflows data
touch research_assistant/__init__.py research_assistant/agents/__init__.py \
      research_assistant/shared/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "research-assistant"
version = "0.1.0"
description = "Multi-agent pipeline that builds a citable knowledge base from physics papers."
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["research_assistant*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Copy the dependency and asset files**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/requirements.txt ../tech_ireland/requirements-layout.txt .
cp ../citation_builder/requirements-test.txt .
cp ../tech_ireland/publaynet_config.yaml .
ln -s ../tech_ireland/model_final.pth model_final.pth
```

Then append to `requirements.txt` (agents 5–7 and `watch.py` need it; neither source repo's core requirements list it):

```
# ─── Agent 6 / watch.py — directory watching ────────────────────────────────
watchdog==6.0.0
```

- [ ] **Step 4: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/

# All runtime state lives under data/ — see config.DATA_DIR
data/

# Layout-detection checkpoint (symlinked, ~830MB)
model_final.pth
```

- [ ] **Step 5: Copy `log.py` and `retry.py`, rewrite their imports**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/shared/log.py ../tech_ireland/shared/retry.py research_assistant/shared/
```

`retry.py` needs no changes. In `log.py`, change the one config import:

```python
# before
from config import LOG_DIR_DEFAULT  # or however it reads config
# after
from research_assistant.config import ...
```

Open `research_assistant/shared/log.py` and apply the import rule from Global Constraints to every `from config …` / `from shared… ` line. Keep both `logger.propagate = False` and the httpx quieting — they fix real double-logging and log-spam bugs.

- [ ] **Step 6: Copy `prompts.py` and correct its stale comments**

```bash
cp ../tech_ireland/prompts.py research_assistant/prompts.py
```

`tech_ireland`'s version is already a superset (it has `DOCUMENT_SUMMARY`, `DOC_RELEVANCE_GATE`, `NO_CORPUS_FALLBACK`, `RELATED_WORK_USER` that `citation_builder` lacks). Only the comments are wrong — they describe agent 5 and `evaluate_rag.py` as "planned — not yet in this repo", which the merge makes false for agent 5. Edit:

- Module docstring: replace the sentence about "a planned Ragas evaluation script (evaluate_rag.py, not yet in this repo)" with: *"Keeping the citation prompt here in one place means a future evaluation script can score the exact prompt Agent 4 runs rather than a copy that has drifted from it."*
- `# ─── Batched citation-need check (planned Agent 5) ───` → `(Agent 5)`
- `# ─── Per-sentence citation with reasoning (planned Agent 5) ───` → `(Agent 5)`
- `CITATION_SUGGESTION_SYSTEM` comment `(Agent 4; also the planned evaluate_rag generator)` → `(Agent 4)`

- [ ] **Step 7: Write `research_assistant/config.py`**

Start from `../tech_ireland/config.py` and apply four changes. Full head of the file:

```python
"""
Central configuration for the Research Assistant pipeline.

All model names, paths and tunable constants live here so that changing a model
or a path only requires editing one file.
"""

import os

# ─── Roots ───────────────────────────────────────────────────────────────────
# config.py now lives inside the package, so the project root is two levels up.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

# Everything the pipeline *writes* is anchored here, separate from the source
# tree. Set CITATION_DATA_DIR to a writable path (e.g. /data) on a read-only or
# ephemeral host.
DATA_DIR = os.environ.get("CITATION_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))


# ─── Env helpers ─────────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
```

Then copy the rest of `tech_ireland/config.py` verbatim **except**:

1. Delete its `PROJECT_ROOT` / `DATA_DIR` lines (replaced above).
2. Replace every inline env-boolean with the helper:
   ```python
   LAYOUT_DETECTION = _env_bool("CITATION_LAYOUT_DETECTION", True)
   FIGURE_VLM       = _env_bool("CITATION_FIGURE_VLM", False)
   DOC_GATE         = _env_bool("CITATION_DOC_GATE", True)
   ```
   and every `int(os.environ.get(...))` with `_env_int`:
   ```python
   DOC_SELECT_K             = _env_int("CITATION_DOC_SELECT_K", 6)
   SUMMARY_MAX_CHARS        = _env_int("CITATION_SUMMARY_MAX_CHARS", 8000)
   GROBID_BATCH_CONCURRENCY = _env_int("GROBID_BATCH_CONCURRENCY", 2)
   ```
3. **Fix the inverted `os.environ.get` arguments.** `tech_ireland` has the model-zoo URL as the *variable name*, so the value is always the yaml and the lp:// path is unreachable:
   ```python
   # before (bug)
   DETECTRON_CONFIG = os.environ.get(
       "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config", "publaynet_config.yaml")
   # after
   DETECTRON_CONFIG = os.environ.get(
       "CITATION_DETECTRON_CONFIG",
       "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
   )
   ```
4. Add the ingestion manifest path next to the other data files, and drop the
   two evaluation paths (`SAMPLE_INPUTS_PATH`, `EVAL_RESULTS_PATH`) and
   `EVAL_MODEL` — Ragas is out of scope:
   ```python
   INGESTED_MANIFEST_PATH = os.path.join(DATA_DIR, "ingested.json")
   ```
   Keep `CITATION_CHECK_BATCH_SIZE`, `DRAFTS_DIR` and all three cooldown
   constants — agents 5–7 and `watch.py` use them.
   Change the comment on `CITATION_CHECK_BATCH_SIZE` from
   `# ─── Agent 5 — Batch Citer (planned — not yet in this repo) ───` to
   `# ─── Agent 5 — Batch Citer ───`.

- [ ] **Step 8: Install the package**

```bash
cd /Users/shardul/Downloads/research_assistant && python -m pip install -e . -q && python -m pip install -r requirements-test.txt -q
```

- [ ] **Step 9: Verify config resolves correctly**

Run:
```bash
cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -c "
from research_assistant import config as c
assert c.PROJECT_ROOT.endswith('research_assistant'), c.PROJECT_ROOT
assert c.DATA_DIR.endswith('/data'), c.DATA_DIR
assert c.VECTORDB_PATH.startswith(c.DATA_DIR), c.VECTORDB_PATH
assert c.DETECTRON_CONFIG.startswith('lp://'), c.DETECTRON_CONFIG
assert c.LAYOUT_DETECTION is True and c.FIGURE_VLM is False
from research_assistant.shared.log import get_logger; get_logger('x').info('ok')
from research_assistant import prompts; assert prompts.DOCUMENT_SUMMARY
print('config ok')"
```
Expected: `config ok`. The `DETECTRON_CONFIG` assertion is the regression test for the inverted-argument bug — it fails against the unfixed `tech_ireland` code.

- [ ] **Step 10: Checkpoint** — Step 9 passes.

---

### Task 2: `schemas.py` — typed stage contracts

**Files:**
- Create: `research_assistant/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `SchemaError`; `Reference`, `DownloadedPaper`, `SeedPaper` dataclasses, each with `.to_dict() -> dict` and `.from_dict(dict) -> Self`.
- Field names match the JSON already written by both repos. `Reference` uses **`source_file`** (not `source_document` — that was an error in the spec; Agent 2 and `summarise()` both read `source_file`) and carries `container` and `xml_id`, which `tech_ireland`'s TEI parser emits.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_schemas.py
"""Tests for the contracts between pipeline stages.

The failure these are designed out of: agent 3 previously accepted two
incompatible downloaded.json shapes, and the wrong one "made every entry look
like a missing file and ingested nothing at all".
"""

import unittest

from research_assistant.schemas import (
    DownloadedPaper,
    Reference,
    SchemaError,
    SeedPaper,
)


class TestReference(unittest.TestCase):
    def test_round_trip_preserves_every_field(self):
        ref = Reference(
            raw_reference="Smith et al. Nature 2020",
            source_file="seed.pdf",
            title="A paper",
            container="Nature",
            authors=("Smith, J.", "Doe, A."),
            year=2020,
            doi="10.1038/x",
            doi_confidence="high",
            xml_id="b12",
            extraction_method="grobid",
        )
        self.assertEqual(Reference.from_dict(ref.to_dict()), ref)

    def test_authors_from_json_list_becomes_a_tuple(self):
        """JSON has no tuples; a decoded list must not break equality."""
        ref = Reference.from_dict(
            {"raw_reference": "r", "source_file": "s.pdf", "authors": ["A", "B"]}
        )
        self.assertEqual(ref.authors, ("A", "B"))

    def test_unknown_keys_are_ignored(self):
        """A manifest written by an older run still loads."""
        ref = Reference.from_dict(
            {"raw_reference": "r", "source_file": "s.pdf", "legacy_field": 1}
        )
        self.assertEqual(ref.raw_reference, "r")

    def test_missing_required_field_raises(self):
        with self.assertRaises(SchemaError):
            Reference.from_dict({"raw_reference": "r"})

    def test_defaults_fill_in_for_absent_optional_fields(self):
        ref = Reference.from_dict({"raw_reference": "r", "source_file": "s.pdf"})
        self.assertIsNone(ref.doi)
        self.assertEqual(ref.authors, ())
        self.assertEqual(ref.extraction_method, "grobid")


class TestDownloadedPaper(unittest.TestCase):
    def test_missing_path_raises_rather_than_half_building(self):
        """The exact bug this schema exists to prevent."""
        with self.assertRaises(SchemaError):
            DownloadedPaper.from_dict(
                {"key": "doi:10.1/x", "provider": "arxiv", "fetched_at": "now"}
            )

    def test_round_trip(self):
        paper = DownloadedPaper(
            key="doi:10.1/x",
            path="data/pulled_pdfs/doi_10.1_x.pdf",
            provider="unpaywall",
            fetched_at="2026-09-07T00:00:00Z",
            title="A paper",
        )
        self.assertEqual(DownloadedPaper.from_dict(paper.to_dict()), paper)

    def test_non_dict_input_raises(self):
        with self.assertRaises(SchemaError):
            DownloadedPaper.from_dict("not a dict")


class TestSeedPaper(unittest.TestCase):
    def test_round_trip(self):
        seed = SeedPaper(
            key="arxiv:2401.12345",
            path="data/raw/arxiv_2401.12345.pdf",
            fetched_at="2026-09-07T00:00:00Z",
            source="search",
            title="A seed",
        )
        self.assertEqual(SeedPaper.from_dict(seed.to_dict()), seed)

    def test_missing_source_raises(self):
        with self.assertRaises(SchemaError):
            SeedPaper.from_dict({"key": "k", "path": "p", "fetched_at": "t"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'research_assistant.schemas'`

- [ ] **Step 3: Write `research_assistant/schemas.py`**

```python
"""
Contracts between pipeline stages.

The agents hand work to each other through JSON manifests. Those were untyped,
and it cost: agent 3 carried a compatibility shim for two incompatible
downloaded.json shapes because one of them "made every entry look like a
missing file and ingested nothing at all".

Each record validates at the boundary. Unknown keys are ignored so a manifest
written by an older run still loads; a missing *required* field raises, because
a half-built record is exactly the failure being designed out.
"""

from dataclasses import MISSING, asdict, dataclass, fields


class SchemaError(ValueError):
    """A record could not be built from the given mapping."""


def _required_names(cls) -> tuple[str, ...]:
    return tuple(
        f.name
        for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    )


class _Record:
    """to_dict / from_dict shared by every schema below."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise SchemaError(
                f"{cls.__name__}: expected a mapping, got {type(data).__name__}"
            )
        missing = [n for n in _required_names(cls) if data.get(n) is None]
        if missing:
            raise SchemaError(
                f"{cls.__name__}: missing required field(s): {', '.join(missing)}"
            )
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if kwargs.get("authors") is not None:
            kwargs["authors"] = tuple(kwargs["authors"])
        return cls(**kwargs)


@dataclass(frozen=True)
class Reference(_Record):
    """One entry in a paper's reference list. Agent 1 → Agent 2."""

    raw_reference: str
    source_file: str                      # the PDF that cited it
    title: str | None = None
    container: str | None = None          # journal / proceedings
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    doi_confidence: str | None = None     # high | medium | low | unknown
    arxiv_id: str | None = None
    pmid: str | None = None
    xml_id: str | None = None             # GROBID biblStruct id; None for regex
    extraction_method: str = "grobid"     # grobid | regex


@dataclass(frozen=True)
class DownloadedPaper(_Record):
    """A reference whose full text was retrieved. Agent 2 → Agent 3."""

    key: str                              # source_key — the primary identity
    path: str
    provider: str                         # unpaywall | europepmc | arxiv | crossref
    fetched_at: str
    title: str | None = None
    raw_reference: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None


@dataclass(frozen=True)
class SeedPaper(_Record):
    """The paper a research query was seeded from. Agent 0 → orchestrator."""

    key: str
    path: str
    fetched_at: str
    source: str                           # "search" | "manual-url"
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_schemas.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Checkpoint** — full suite green: `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 3: `shared/manifest.py` — what has been ingested

**Files:**
- Create: `research_assistant/shared/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `config.INGESTED_MANIFEST_PATH`, `config.VECTORDB_PATH`, `config.COLLECTION_NAME` (Task 1)
- Produces: `load() -> dict[str, str]`, `contains(pdf_key: str) -> bool`, `add(pdf_key: str) -> None`, `add_many(pdf_keys: Iterable[str]) -> None`, `rebuild_from_collection() -> int`

Replaces `ingestion_manifest.txt` — an append-only, never-deduplicated text file living *inside* the ChromaDB directory — and the full paginated collection scan that ran on every ingest.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest.py
"""Tests for the ingestion manifest.

This records what has already been parsed. Parsing is the expensive stage —
layout detection per page plus a VLM call per figure — so a key that fails to
persist means the PDF is re-parsed in full on every future run.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import research_assistant.shared.manifest as m


class ManifestTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "ingested.json")
        patcher = patch.object(m, "MANIFEST_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestLoadAndAdd(ManifestTestCase):
    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(m.load(), {})
        self.assertFalse(m.contains("paper.pdf"))

    def test_add_then_contains(self):
        m.add("paper.pdf")
        self.assertTrue(m.contains("paper.pdf"))

    def test_add_persists_to_disk(self):
        m.add("paper.pdf")
        with open(self.path) as fh:
            self.assertIn("paper.pdf", json.load(fh))

    def test_adding_twice_does_not_duplicate(self):
        """The old text file appended unconditionally and grew forever."""
        m.add("paper.pdf")
        m.add("paper.pdf")
        self.assertEqual(list(m.load()), ["paper.pdf"])

    def test_add_many_writes_every_key(self):
        m.add_many(["a.pdf", "b.pdf", "c.pdf"])
        self.assertEqual(sorted(m.load()), ["a.pdf", "b.pdf", "c.pdf"])

    def test_add_many_preserves_existing_entries(self):
        m.add("old.pdf")
        m.add_many(["new.pdf"])
        self.assertEqual(sorted(m.load()), ["new.pdf", "old.pdf"])


class TestCorruption(ManifestTestCase):
    def test_unreadable_manifest_is_treated_as_empty(self):
        """Corrupt state must not crash ingestion — it re-parses at worst."""
        with open(self.path, "w") as fh:
            fh.write("{not json")
        self.assertEqual(m.load(), {})

    def test_wrong_toplevel_type_is_treated_as_empty(self):
        with open(self.path, "w") as fh:
            json.dump(["a.pdf"], fh)
        self.assertEqual(m.load(), {})

    def test_writing_over_a_corrupt_manifest_recovers_it(self):
        with open(self.path, "w") as fh:
            fh.write("{not json")
        m.add("paper.pdf")
        self.assertEqual(list(m.load()), ["paper.pdf"])


class TestAtomicity(ManifestTestCase):
    def test_no_temp_files_left_behind(self):
        m.add("paper.pdf")
        leftovers = [f for f in os.listdir(self.tmp.name) if f != "ingested.json"]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_manifest.py -v`
Expected: FAIL — `No module named 'research_assistant.shared.manifest'`

- [ ] **Step 3: Write `research_assistant/shared/manifest.py`**

```python
"""
Which documents have already been ingested.

This used to be two sources of truth unioned on every call: an append-only
newline-delimited text file inside the ChromaDB directory (no deduplication,
growing without bound) and a full paginated scan of every metadata record in
the collection. The scan is now a repair path — rebuild_from_collection() —
rather than something ingestion pays for on every run.

Writes are atomic: a temp file in the same directory, then os.replace, the same
pattern shared/fetch.py uses for downloads. A half-written manifest would make
the pipeline re-parse an entire corpus.
"""

import json
import os
import tempfile
import time
from typing import Iterable

from research_assistant.config import (
    COLLECTION_NAME,
    INGESTED_MANIFEST_PATH,
    VECTORDB_PATH,
)
from research_assistant.shared.log import get_logger

logger = get_logger("manifest")

MANIFEST_PATH = INGESTED_MANIFEST_PATH


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load() -> dict[str, str]:
    """Return ``{pdf_key: iso8601}``. Unreadable or absent means empty."""
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Ingestion manifest at %s is unreadable (%s) — treating as empty. "
            "Already-ingested PDFs may be re-parsed; rebuild_from_collection() "
            "recovers it from the vector store.",
            MANIFEST_PATH, exc,
        )
        return {}

    if not isinstance(data, dict):
        logger.warning(
            "Ingestion manifest at %s is a %s, expected an object — treating as "
            "empty.", MANIFEST_PATH, type(data).__name__,
        )
        return {}
    return data


def _write(data: dict[str, str]) -> None:
    directory = os.path.dirname(MANIFEST_PATH) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, MANIFEST_PATH)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def contains(pdf_key: str) -> bool:
    return pdf_key in load()


def add(pdf_key: str) -> None:
    data = load()
    if pdf_key in data:
        return
    data[pdf_key] = _now()
    _write(data)


def add_many(pdf_keys: Iterable[str]) -> None:
    """Add several keys in one read-modify-write.

    ingest_pdfs() marks every candidate at the end of a batch; doing that one
    at a time would re-read and rewrite the whole manifest per PDF.
    """
    data = load()
    now = _now()
    added = False
    for key in pdf_keys:
        if key not in data:
            data[key] = now
            added = True
    if added:
        _write(data)


def rebuild_from_collection() -> int:
    """Repair the manifest from ChromaDB metadata. Returns the number of keys.

    Only for recovering a lost or corrupt manifest — ingestion does not call
    this. Reads every metadata record in the collection.
    """
    import chromadb

    try:
        collection = chromadb.PersistentClient(path=VECTORDB_PATH).get_collection(
            name=COLLECTION_NAME
        )
    except Exception as exc:  # noqa: BLE001 — no collection yet is not an error
        logger.warning("No collection to rebuild from (%s).", exc)
        return 0

    documents, limit, offset = set(), 5000, 0
    while True:
        batch = collection.get(include=["metadatas"], limit=limit, offset=offset)
        if not batch or not batch["metadatas"]:
            break
        for meta in batch["metadatas"]:
            if meta and meta.get("document"):
                documents.add(meta["document"])
        offset += limit

    now = _now()
    _write({doc: now for doc in sorted(documents)})
    logger.info("Rebuilt ingestion manifest with %d document(s).", len(documents))
    return len(documents)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_manifest.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 4: `shared/llm.py` — add streaming

**Files:**
- Create: `research_assistant/shared/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `config.LLM_BACKEND`, `EMBED_BACKEND`, `LLM_MODEL`, `EMBED_MODEL`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `HF_TOKEN`
- Produces: `ChatResult(content, prompt_tokens, completion_tokens)`; `chat(messages, model=None, images=None) -> ChatResult`; **`chat_stream(messages, model=None, options=None) -> Iterator[str]`**; `get_embeddings(model=None)`

Agent 7 currently streams by calling `ollama.chat(stream=True)` directly. Porting it as-is would leave the merged repo with a backend abstraction one of its agents bypasses, so the abstraction gains streaming.

- [ ] **Step 1: Copy the base file and rewrite imports**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/shared/llm.py research_assistant/shared/llm.py
```

Change the two import blocks:
```python
from research_assistant.config import (
    LLM_BACKEND, EMBED_BACKEND, LLM_MODEL, EMBED_MODEL,
    OPENAI_BASE_URL, OPENAI_API_KEY, HF_TOKEN,
)
from research_assistant.shared.log import get_logger
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_llm.py
"""Tests for the backend-agnostic LLM layer.

chat_stream() exists so agent 7 can stream without importing a provider SDK.
Both backends must yield plain string deltas and skip the empty frames each
API emits, so agent code sees one contract.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import research_assistant.shared.llm as llm


def _ollama_chunk(text):
    return types.SimpleNamespace(message=types.SimpleNamespace(content=text))


def _openai_chunk(text):
    delta = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


class TestChatStreamOllama(unittest.TestCase):
    def test_yields_content_deltas_in_order(self):
        fake = MagicMock()
        fake.chat.return_value = iter(
            [_ollama_chunk("Hel"), _ollama_chunk("lo"), _ollama_chunk("!")]
        )
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "hi"}])),
                ["Hel", "lo", "!"],
            )

    def test_empty_deltas_are_skipped(self):
        fake = MagicMock()
        fake.chat.return_value = iter(
            [_ollama_chunk(""), _ollama_chunk("x"), _ollama_chunk(None)]
        )
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "hi"}])), ["x"]
            )

    def test_options_are_forwarded(self):
        """CHAT_OLLAMA_OPTIONS carries flash-attention / KV-cache tuning."""
        fake = MagicMock()
        fake.chat.return_value = iter([_ollama_chunk("x")])
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            list(llm.chat_stream([{"role": "user", "content": "hi"}],
                                 options={"num_ctx": 4096}))
        self.assertEqual(fake.chat.call_args.kwargs["options"], {"num_ctx": 4096})


class TestChatStreamOpenAI(unittest.TestCase):
    def test_yields_content_deltas_and_skips_role_only_frames(self):
        stream = [_openai_chunk(None), _openai_chunk("Hi"), _openai_chunk(None)]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(stream)
        module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))
        with patch.dict("sys.modules", {"openai": module}), \
             patch.object(llm, "LLM_BACKEND", "openai"):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "hi"}])), ["Hi"]
            )

    def test_chunk_with_no_choices_is_skipped(self):
        """Some gateways emit a usage-only terminal frame."""
        stream = [types.SimpleNamespace(choices=[]), _openai_chunk("ok")]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(stream)
        module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))
        with patch.dict("sys.modules", {"openai": module}), \
             patch.object(llm, "LLM_BACKEND", "openai"):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "hi"}])), ["ok"]
            )


class TestUnknownBackend(unittest.TestCase):
    def test_raises_eagerly_not_on_first_next(self):
        """A bad backend must fail at the call, not when iteration starts."""
        with patch.object(llm, "LLM_BACKEND", "nope"):
            with self.assertRaises(ValueError):
                llm.chat_stream([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_llm.py -v`
Expected: FAIL — `AttributeError: module … has no attribute 'chat_stream'`

- [ ] **Step 4: Add `chat_stream` to `research_assistant/shared/llm.py`**

Insert after `_openai_chat`, before the `# ─── Embeddings ───` section:

```python
# ─── Streaming chat ──────────────────────────────────────────────────────────
# Agent 7 streams token-by-token. Kept behind the same backend switch as chat()
# so no agent has to import a provider SDK to do it.


def chat_stream(messages, model=None, options=None):
    """Yield content deltas as plain strings.

    Args:
        messages: list of ``{"role", "content"}`` dicts.
        model:    model id; defaults to config.LLM_MODEL.
        options:  ollama runtime options (e.g. CHAT_OLLAMA_OPTIONS). The
                  OpenAI-compatible backend has no equivalent and ignores it.

    Deliberately not a generator itself, so an unknown backend raises at the
    call rather than on the first ``next()``.
    """
    model = model or LLM_MODEL
    if LLM_BACKEND == "openai":
        return _openai_stream(messages, model)
    if LLM_BACKEND == "ollama":
        return _ollama_stream(messages, model, options)
    raise ValueError(
        f"Unknown LLM_BACKEND {LLM_BACKEND!r} (expected 'ollama' or 'openai')"
    )


def _ollama_stream(messages, model, options):
    import ollama

    kwargs = {"model": model, "messages": messages, "stream": True}
    if options:
        kwargs["options"] = options
    for chunk in ollama.chat(**kwargs):
        try:
            content = chunk.message.content
        except AttributeError:
            content = (chunk.get("message") or {}).get("content")
        if content:
            yield content


def _openai_stream(messages, model):
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True
    )
    for chunk in stream:
        # Role-only opening frames and usage-only terminal frames carry no text.
        if not getattr(chunk, "choices", None):
            continue
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            yield content
```

Also update the module docstring: after the sentence describing `chat()` and `get_embeddings()`, add `chat_stream()` to the list of entry points.

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_llm.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 6: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 5: Port the unchanged shared modules

**Files:**
- Create: `research_assistant/shared/{db,fetch,source_key,search,retrieve}.py`
- Test: `tests/test_search.py` (ported)

**Interfaces:**
- Consumes: `shared.llm.get_embeddings` (Task 4), `config` (Task 1)
- Produces: `source_key(record, min_doi_confidence=("high","medium")) -> str|None`, `normalise_doi(doi)`, `is_authoritative(key)`, `title_blocking_key(record)`; `download_pdf(url, dest_path) -> (bool, str|None)`, `filename_for(key)`, `HEADERS`; `load_search_resources() -> (collection, bm25, texts, metadatas)`, `get_max_chunk_index(collection)`; `hybrid_search(query, collection, bm25, texts, metadatas, top_k, rrf_k, embeddings_model, doc_filter) -> list[dict]`; `rank_documents`, `gate_documents`, `deep_search`, `research_answer(query, top_k) -> dict|None`

These five are taken from `tech_ireland` unchanged apart from imports. `search.py` there already has the `doc_filter` parameter that `retrieve.py` depends on and `citation_builder`'s copy lacks.

- [ ] **Step 1: Copy the files**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/shared/{db,fetch,source_key,search,retrieve}.py research_assistant/shared/
```

- [ ] **Step 2: Rewrite imports in all five**

Apply to each file:
```bash
cd /Users/shardul/Downloads/research_assistant/research_assistant/shared
sed -i '' \
  -e 's/^from config import/from research_assistant.config import/' \
  -e 's/^from prompts import/from research_assistant.prompts import/' \
  -e 's/^from shared\./from research_assistant.shared./' \
  -e 's/^from shared import/from research_assistant.shared import/' \
  db.py fetch.py source_key.py search.py retrieve.py
```

Then hand-check the **function-local** imports `sed` will not have caught, because they are indented — `retrieve.py` has `from shared.llm import get_embeddings`, `from shared.llm import chat`, `from shared.db import load_search_resources`, `from shared.search import hybrid_search` inside function bodies, and `search.py` has `from shared.llm import get_embeddings` inside `_get_embeddings()`. Fix each to the `research_assistant.shared.…` form. Verify none remain:

```bash
grep -rn "from shared\.\|from config import\|from prompts import" research_assistant/ || echo "clean"
```

- [ ] **Step 3: Correct the stale comments**

- `search.py` module docstring: replace *"Used by Agent 4 (interactive). Agent 5 (batch) and evaluate_rag.py are planned consumers, not yet in this repo."* with *"Used by Agent 4 (interactive), Agent 5 (batch) and Agent 7 (chat)."* In the `hybrid_search` body, the comment *"The planned Agent 5 would call this once per sentence"* → *"Agent 5 calls this once per sentence needing a citation"*.
- `db.py` module docstring: *"used by Agent 4. (Agent 5 and evaluate_rag.py are planned, not yet in this repo.)"* → *"used by Agents 4, 5 and 7."*
- `fetch.py`: the `HEADERS` User-Agent string is `citation_builder/0.1` — change to `research_assistant/0.1`.

- [ ] **Step 4: Port `tests/test_search.py`**

```bash
cp ../../citation_builder/test_search.py ../../tests/test_search.py
```
(from the repo root: `cp ../citation_builder/test_search.py tests/test_search.py`)

Change its one import:
```python
from research_assistant.shared.search import hybrid_search
```

The test passes its own `FakeBM25` / `FakeCollection` and an explicit `embeddings_model`, so it needs no network, no ChromaDB and no Ollama.

- [ ] **Step 5: Run the tests**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_search.py -v`
Expected: PASS — both the partition-vs-true-top-k test and the out-of-range dense-hit test.

- [ ] **Step 6: Checkpoint** — full suite plus the import sweep from Global Constraints (it will still fail on `shared.ingestion` and the agents; confirm the five modules from this task import cleanly).

---

### Task 6: `shared/ingestion.py` — rewire to the manifest

**Files:**
- Create: `research_assistant/shared/ingestion.py`
- Test: `tests/test_ingestion.py` (ported and adjusted)

**Interfaces:**
- Consumes: `shared.manifest` (Task 3), `shared.llm.chat`/`get_embeddings` (Task 4), `shared.db.get_max_chunk_index` (Task 5)
- Produces: `ingest_pdfs(pdfs, workers=1, skip_ingested=True, rebuild_index=True, log_prefix="") -> dict`, `process_pdf`, `upsert_corpus`, `upsert_summaries`, `rebuild_bm25`, `pdf_key(pdf_path) -> str`, `get_ingested_documents() -> set[str]`

- [ ] **Step 1: Copy and rewrite imports**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/shared/ingestion.py research_assistant/shared/ingestion.py
```

Apply the same import rules as Task 5, including the function-local `from shared.llm import chat, get_embeddings` inside `upsert_summaries`, `describe_figure` and `upsert_corpus`.

- [ ] **Step 2: Replace the manifest functions**

Delete `mark_document_ingested()` and the manifest half of `get_ingested_documents()`. Replace both with manifest-backed versions:

```python
# before — two sources of truth, unioned on every call
def get_ingested_documents(collection=None) -> set[str]:
    docs = set()
    manifest_path = os.path.join(VECTORDB_PATH, "ingestion_manifest.txt")
    if os.path.exists(manifest_path):
        ...
    # then a full paginated scan of the whole collection
    ...

def mark_document_ingested(pdf_name: str):
    with open(manifest_path, "a") as f:
        f.write(pdf_name + "\n")
```

```python
# after
def get_ingested_documents() -> set[str]:
    """Documents already parsed, per the manifest.

    The collection scan this used to do on every call is now
    shared.manifest.rebuild_from_collection(), a repair path — reading every
    metadata record in the corpus before ingesting one file was the single
    most expensive thing about a warm run.
    """
    return set(manifest.load())
```

Add `from research_assistant.shared import manifest` to the module imports, and drop the now-unused `VECTORDB_PATH`-based manifest path.

- [ ] **Step 3: Point `ingest_pdfs` at `add_many`**

In `ingest_pdfs`, the marking loop:

```python
# before
for path in candidates:
    mark_document_ingested(pdf_key(path))
```
```python
# after
manifest.add_many(pdf_key(path) for path in candidates)
```

The `skip_ingested` branch calls `get_ingested_documents()`, which now needs no `collection` argument — confirm the call site passes none.

- [ ] **Step 4: Port and adjust `tests/test_ingestion.py`**

```bash
cp ../citation_builder/test_ingestion.py tests/test_ingestion.py
```

Three changes:
1. `import shared.ingestion as ing` → `import research_assistant.shared.ingestion as ing`
2. Any test that patched the text-file manifest path now patches the JSON one. Add at the top of the shared test-case `setUp`:
   ```python
   import research_assistant.shared.manifest as manifest_mod
   patcher = patch.object(
       manifest_mod, "MANIFEST_PATH", os.path.join(self.tmp.name, "ingested.json")
   )
   patcher.start()
   self.addCleanup(patcher.stop)
   ```
3. Any assertion reading `ingestion_manifest.txt` becomes a `manifest_mod.load()` call.

Add one new test for the behaviour Task 3 changed:

```python
class TestMarkingUsesTheManifest(ManifestBackedTestCase):
    def test_every_attempted_pdf_is_marked_even_when_it_yields_nothing(self):
        """A corrupt or empty PDF must still be marked, or it is re-parsed forever."""
        import research_assistant.shared.manifest as manifest_mod

        pdf = _touch(self.tmp.name, "empty.pdf")
        with patch.object(ing, "process_pdf", return_value=[]), \
             patch.object(ing, "rebuild_bm25"):
            ing.ingest_pdfs({pdf: "label"}, rebuild_index=False)
        self.assertIn(ing.pdf_key(pdf), manifest_mod.load())
```

- [ ] **Step 5: Run the tests**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_ingestion.py -v`
Expected: PASS — the `pdf_key` normalisation tests and the check/mark contract tests.

- [ ] **Step 6: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 7: Agent 1 — GROBID primary, regex fallback

**Files:**
- Create: `research_assistant/agents/agent1_extractor.py`
- Test: `tests/test_extractor.py` (ported and extended)

**Interfaces:**
- Consumes: `schemas.Reference` (Task 2), `config.GROBID_SERVER`, `config.GROBID_BATCH_CONCURRENCY`, `config.RAW_DIR`, `config.EXTRACTED_CITATIONS_PATH`
- Produces: `grobid_alive() -> bool`, `extract_references(pdf_path, grobid_ok) -> tuple[list[Reference], str]` where the second element is `"grobid" | "regex" | "none"`, `run_extractor() -> dict`, `_unique_destination(directory, name) -> str`, `parse_reference`, `parse_tei_file`, `summarise`

This is the only place where "take the better repo's version" is the wrong answer. `tech_ireland`'s GROBID path produces far richer references but dies when the server is down; `citation_builder`'s `pdftotext` + regex path is weaker but has no external dependency. Merged, the pipeline degrades instead of stopping.

- [ ] **Step 1: Copy the GROBID implementation as the base**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/agent1_extractor.py research_assistant/agents/agent1_extractor.py
```

Rewrite imports per Global Constraints, and add:
```python
import shutil
import subprocess

from research_assistant.schemas import Reference
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_extractor.py
"""Tests for Agent 1.

Two properties matter. Where a PDF is filed after processing is the whole
record of what happened — a paper under processed/ is treated as done and never
looked at again. And extraction must degrade to the regex path rather than
stopping when GROBID is unreachable, because a dead server used to end the run.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import research_assistant.agents.agent1_extractor as ex
from research_assistant.schemas import Reference


class TestUniqueDestination(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_uses_the_plain_name_when_free(self):
        self.assertEqual(
            ex._unique_destination(self.tmp.name, "paper.pdf"),
            os.path.join(self.tmp.name, "paper.pdf"),
        )

    def test_suffixes_rather_than_overwriting(self):
        """Two different papers with one filename must not silently collide."""
        open(os.path.join(self.tmp.name, "paper.pdf"), "w").close()
        self.assertNotEqual(
            ex._unique_destination(self.tmp.name, "paper.pdf"),
            os.path.join(self.tmp.name, "paper.pdf"),
        )


class TestRegexReferences(unittest.TestCase):
    def _extract(self, text):
        with patch.object(ex, "_pdftotext", return_value=text):
            return ex._references_from_regex("some.pdf")

    def test_bracket_numbered_references(self):
        refs = self._extract(
            "Introduction text\n"
            "References\n"
            "[1] Smith J. A first paper. Nature, 2019.\n"
            "[2] Doe A. A second paper. PRL, 2020.\n"
        )
        self.assertEqual(len(refs), 2)
        self.assertIn("Smith", refs[0].raw_reference)

    def test_dot_numbered_references(self):
        refs = self._extract(
            "References\n"
            "1. Smith J. A first paper. Nature, 2019.\n"
            "2. Doe A. A second paper. PRL, 2020.\n"
        )
        self.assertEqual(len(refs), 2)

    def test_every_reference_is_tagged_as_regex_extracted(self):
        """Provenance must be visible downstream — these are lower quality."""
        refs = self._extract("References\n[1] Smith J. A paper. Nature, 2019.\n")
        self.assertEqual(refs[0].extraction_method, "regex")
        self.assertEqual(refs[0].source_file, "some.pdf")

    def test_structured_fields_are_absent_not_invented(self):
        refs = self._extract("References\n[1] Smith J. A paper. Nature, 2019.\n")
        self.assertIsNone(refs[0].doi)
        self.assertIsNone(refs[0].doi_confidence)

    def test_no_recognisable_references_yields_nothing(self):
        self.assertEqual(self._extract("Just prose, no reference list."), [])


class TestFallbackRouting(unittest.TestCase):
    def _ref(self, method):
        return Reference(
            raw_reference="r", source_file="p.pdf", extraction_method=method
        )

    def test_grobid_used_when_available_and_productive(self):
        with patch.object(ex, "_references_from_grobid",
                          return_value=[self._ref("grobid")]) as grobid, \
             patch.object(ex, "_references_from_regex") as regex:
            refs, method = ex.extract_references("p.pdf", grobid_ok=True)
        self.assertEqual(method, "grobid")
        self.assertEqual(len(refs), 1)
        grobid.assert_called_once()
        regex.assert_not_called()

    def test_regex_used_when_grobid_is_down(self):
        """The case that used to end the whole run."""
        with patch.object(ex, "_references_from_grobid") as grobid, \
             patch.object(ex, "_references_from_regex",
                          return_value=[self._ref("regex")]):
            refs, method = ex.extract_references("p.pdf", grobid_ok=False)
        self.assertEqual(method, "regex")
        self.assertEqual(len(refs), 1)
        grobid.assert_not_called()

    def test_regex_used_when_grobid_returns_nothing_for_this_pdf(self):
        """Per-PDF fallback, not just per-run."""
        with patch.object(ex, "_references_from_grobid", return_value=[]), \
             patch.object(ex, "_references_from_regex",
                          return_value=[self._ref("regex")]):
            refs, method = ex.extract_references("p.pdf", grobid_ok=True)
        self.assertEqual(method, "regex")

    def test_both_paths_empty_reports_none(self):
        with patch.object(ex, "_references_from_grobid", return_value=[]), \
             patch.object(ex, "_references_from_regex", return_value=[]):
            refs, method = ex.extract_references("p.pdf", grobid_ok=True)
        self.assertEqual((refs, method), ([], "none"))


class TestGrobidProbe(unittest.TestCase):
    def test_unreachable_server_is_false_not_an_exception(self):
        import requests

        with patch.object(ex.requests, "get",
                          side_effect=requests.RequestException("refused")):
            self.assertFalse(ex.grobid_alive())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_extractor.py -v`
Expected: FAIL — `_references_from_regex`, `extract_references`, `grobid_alive` do not exist.

- [ ] **Step 4: Add the probe and the regex path**

Insert into `research_assistant/agents/agent1_extractor.py`:

```python
# ─── Regex fallback ──────────────────────────────────────────────────────────
# Ported from the pdftotext-based extractor. Weaker than GROBID — a raw string
# per reference, no structured fields — but it needs no server, which is the
# whole point: an unreachable GROBID used to end the pipeline.

CITATION_PATTERNS = [
    re.compile(r'^\[(\d+)\](?:\s*(.*))?$'),    # [1] Author...
    re.compile(r'^(\d+)\.(?:\s*(.*))?$'),      # 1. Author...
    re.compile(r'^\((\d+)\)(?:\s*(.*))?$'),    # (1) Author...
]


def _match_citation_line(line):
    """Return (id, rest_of_line) for a reference-list line, else None."""
    for pattern in CITATION_PATTERNS:
        match = pattern.match(line)
        if match:
            return int(match.group(1)), match.group(2)
    return None


def _pdftotext(pdf_path: str) -> str:
    """Extract raw text. Returns "" when pdftotext is missing or fails."""
    try:
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("pdftotext unavailable for %s: %s", pdf_path, exc)
        return ""
    if result.returncode != 0:
        logger.warning("pdftotext failed on %s: %s", pdf_path, result.stderr.strip())
        return ""
    return result.stdout


def _references_from_regex(pdf_path: str) -> list[Reference]:
    """Reference strings recovered by numbering pattern alone."""
    text = _pdftotext(pdf_path)
    if not text.strip():
        return []

    source_file = os.path.basename(pdf_path)
    references, current, seen = [], None, set()

    def _flush():
        if current is None:
            return
        joined = " ".join(current).strip()
        if len(joined) > 20 and joined not in seen:
            seen.add(joined)
            references.append(
                Reference(
                    raw_reference=joined,
                    source_file=source_file,
                    extraction_method="regex",
                )
            )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        matched = _match_citation_line(line)
        if matched:
            _flush()
            _, rest = matched
            current = [rest] if rest else []
        elif current is not None:
            # Continuation of the reference started on a previous line.
            current.append(line)
    _flush()

    return references


# ─── Strategy selection ──────────────────────────────────────────────────────


def grobid_alive() -> bool:
    """Probe the GROBID server once per run."""
    try:
        response = requests.get(f"{GROBID_SERVER}/api/isalive", timeout=10)
    except requests.RequestException as exc:
        logger.warning("GROBID at %s is unreachable (%s).", GROBID_SERVER, exc)
        return False
    return response.ok and "true" in response.text.lower()


def extract_references(pdf_path: str, grobid_ok: bool) -> tuple[list[Reference], str]:
    """Return (references, method) for one PDF.

    method is "grobid", "regex" or "none". GROBID is preferred whenever it is
    up and productive; the regex path covers both a dead server and a PDF that
    GROBID parsed without finding a reference list.
    """
    if grobid_ok:
        references = _references_from_grobid(pdf_path)
        if references:
            return references, "grobid"
        logger.info(
            "GROBID found no references in %s — falling back to pattern matching.",
            os.path.basename(pdf_path),
        )

    references = _references_from_regex(pdf_path)
    if references:
        return references, "regex"
    return [], "none"
```

Add `import requests` to the module imports if the copied file lacks it.

- [ ] **Step 5: Wrap the existing TEI parse as `_references_from_grobid`**

The copied file already has `parse_tei_file()` returning `(article, references)` as **dicts**. Add a wrapper that runs the GROBID client for one PDF, parses the resulting TEI, and converts each dict to a `Reference`:

```python
def _references_from_grobid(pdf_path: str) -> list[Reference]:
    """Process one PDF through GROBID and parse its TEI into References.

    The TEI is cached under XML_OUTPUT_DIR, so a re-run parses from disk
    rather than re-uploading the PDF.
    """
    tei_path = _tei_path_for(pdf_path)
    if not os.path.exists(tei_path):
        try:
            _run_grobid([pdf_path])
        except Exception as exc:  # noqa: BLE001 — any client failure means fall back
            logger.warning("GROBID processing failed for %s: %s", pdf_path, exc)
            return []
        if not os.path.exists(tei_path):
            return []

    try:
        _article, raw_references = parse_tei_file(tei_path)
    except Exception as exc:  # noqa: BLE001 — malformed TEI means fall back
        logger.warning("Could not parse TEI for %s: %s", pdf_path, exc)
        return []

    references = []
    for raw in raw_references:
        if not (raw.get("raw_reference") or raw.get("title")):
            continue
        references.append(
            Reference(
                raw_reference=raw.get("raw_reference") or raw.get("title") or "",
                source_file=raw.get("source_file") or os.path.basename(pdf_path),
                title=raw.get("title"),
                container=raw.get("container"),
                authors=tuple(raw.get("authors") or ()),
                year=raw.get("year"),
                doi=raw.get("doi"),
                doi_confidence=raw.get("doi_confidence"),
                xml_id=raw.get("xml_id"),
                extraction_method="grobid",
            )
        )
    return references
```

`_tei_path_for(pdf_path)` returns the cached TEI path — reuse the existing `TEI_SUFFIXES` tuple to find whichever suffix the client wrote:

```python
def _tei_path_for(pdf_path: str) -> str:
    stem = _stem(pdf_path)
    for suffix in TEI_SUFFIXES:
        candidate = os.path.join(XML_OUTPUT_DIR, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(XML_OUTPUT_DIR, stem + TEI_SUFFIXES[0])
```

`_run_grobid(paths)` wraps the `GrobidClient` call the copied `run_extractor()` already makes — extract it from that function so both callers share it.

- [ ] **Step 6: Rewrite `run_extractor()` around the new strategy**

```python
def run_extractor() -> dict:
    """Extract references from every PDF in RAW_DIR.

    Files each source PDF by outcome so a paper is never silently swallowed:
    raw/processed/ when references were read, raw/failed/ when neither strategy
    produced any.
    """
    os.makedirs(XML_OUTPUT_DIR, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(RAW_DIR, "*.pdf")))
    if not pdfs:
        logger.info("No PDFs found in %s.", RAW_DIR)
        return {"references": 0, "processed": 0, "failed": 0}

    grobid_ok = grobid_alive()
    if not grobid_ok:
        logger.warning(
            "GROBID at %s is not responding — falling back to pattern-based "
            "extraction for all %d PDF(s). References will lack DOIs, authors "
            "and years; Agent 2 will resolve what it can via Crossref.",
            GROBID_SERVER, len(pdfs),
        )

    processed_dir = os.path.join(RAW_DIR, "processed")
    failed_dir = os.path.join(RAW_DIR, "failed")
    os.makedirs(processed_dir, exist_ok=True)

    all_references, by_method = [], {"grobid": 0, "regex": 0, "none": 0}

    for pdf in pdfs:
        references, method = extract_references(pdf, grobid_ok)
        by_method[method] += 1
        name = os.path.basename(pdf)

        if references:
            all_references.extend(references)
            logger.info("%s: %d reference(s) via %s.", name, len(references), method)
            shutil.move(pdf, _unique_destination(processed_dir, name))
        else:
            # Nothing came out: a scan with no text layer, an unrecognised
            # reference format, or no reference list at all. Filing it under
            # processed/ would claim a success it did not have.
            logger.warning("%s: no references extracted — filing under failed/.", name)
            os.makedirs(failed_dir, exist_ok=True)
            shutil.move(pdf, _unique_destination(failed_dir, name))

    payload = {
        "references": [r.to_dict() for r in all_references],
        "summary": summarise([r.to_dict() for r in all_references]),
        "extraction": by_method,
    }
    with open(EXTRACTED_CITATIONS_PATH, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Wrote %d reference(s) from %d PDF(s) (%d via GROBID, %d via patterns, "
        "%d yielded nothing) to %s.",
        len(all_references), len(pdfs), by_method["grobid"], by_method["regex"],
        by_method["none"], EXTRACTED_CITATIONS_PATH,
    )
    return {
        "references": len(all_references),
        "processed": by_method["grobid"] + by_method["regex"],
        "failed": by_method["none"],
    }
```

The top-level `{"references": [...], "summary": {...}}` shape is preserved exactly — Agent 2 reads `payload.get("references", [])`. `summarise()` takes dicts, so it is fed `to_dict()` output.

Confirm `summarise()` tolerates a reference with `doi_confidence=None`: it filters on `r["doi"]` first, so regex references (no DOI) never reach the confidence check. No change needed.

- [ ] **Step 7: Run the tests**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_extractor.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 8: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 8: Agents 0 and 4 — mechanical port

**Files:**
- Create: `research_assistant/agents/agent0_discoverer.py`, `research_assistant/agents/agent4_assistant.py`

**Interfaces:**
- Consumes: `shared.fetch`, `shared.source_key`, `shared.db`, `shared.search`, `shared.llm`, `schemas.SeedPaper`
- Produces: `agent0.discover(query, force=False) -> str|None`, `agent0.discover_from_url(query, url, force=False) -> str|None`, `agent0.get_seed(query) -> dict|None`, `agent0.find_and_fetch_seed(query, force, limit)`; `agent4.suggest_citation(text, top_k, search_resources=None) -> dict|None`

- [ ] **Step 1: Copy both files**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/agent0_discoverer.py research_assistant/agents/agent0_discoverer.py
cp ../tech_ireland/agent4_assistant.py research_assistant/agents/agent4_assistant.py
```

- [ ] **Step 2: Rewrite imports in both**

Apply the Global Constraints rule. `agent4_assistant.py` needs `from research_assistant.shared.llm import chat` and four other `shared.*` imports; `agent0_discoverer.py` needs `config`, `shared.fetch`, `shared.log`, `shared.source_key`.

- [ ] **Step 3: Wire Agent 0's manifest through `SeedPaper`**

`_download_and_record()` and `discover()` build the seed record as a bare dict. Route both through the schema so a malformed record fails at write time. In `discover()`:

```python
# before
seeds[query] = {
    "key": key, "title": cand.get("title"), "url": cand["pdf_url"],
    "path": dest, "doi": cand.get("doi"), "arxiv_id": cand.get("arxiv_id"),
    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source": "search",
}
```
```python
# after
seeds[query] = SeedPaper(
    key=key,
    path=dest,
    fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    source="search",
    title=cand.get("title"),
    url=cand["pdf_url"],
    doi=cand.get("doi"),
    arxiv_id=cand.get("arxiv_id"),
).to_dict()
```

In `_download_and_record()`, replace the `{**extra}` splat with explicit `SeedPaper(...)` construction — callers pass `arxiv_id=` and `source="manual-url"`. Change its signature from `**extra` to named `arxiv_id=None, source="manual-url"` so the record cannot grow undeclared keys.

Add `from research_assistant.schemas import SeedPaper`. `get_seed()` keeps returning a plain dict — `orchestrate.py` and `app.py` read it with `.get()`.

- [ ] **Step 4: Verify both import and expose their entry points**

Run:
```bash
cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -c "
from research_assistant.agents import agent0_discoverer as a0, agent4_assistant as a4
assert callable(a0.discover) and callable(a0.discover_from_url) and callable(a0.get_seed)
assert callable(a4.suggest_citation)
from research_assistant.schemas import SeedPaper
s = SeedPaper(key='arxiv:1', path='p.pdf', fetched_at='t', source='search')
assert SeedPaper.from_dict(s.to_dict()) == s
print('agents 0 and 4 ok')"
```
Expected: `agents 0 and 4 ok`

- [ ] **Step 5: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 9: Agents 2 and 3 — schema wiring

**Files:**
- Create: `research_assistant/agents/agent2_fetcher.py`, `research_assistant/agents/agent3_ingestor.py`
- Test: `tests/test_fetcher.py` (ported)

**Interfaces:**
- Consumes: `schemas.Reference`, `schemas.DownloadedPaper`, `shared.fetch`, `shared.source_key`, `shared.ingestion.ingest_pdfs`
- Produces: `agent2.fetch_papers() -> None`, `agent2.paper_filename(title, doi=None, arxiv_id=None) -> str`, `agent2.resolve_doi(ref) -> tuple[str|None, str]`; `agent3.run_ingestor(workers=1, force=False) -> dict|None`

- [ ] **Step 1: Copy both and rewrite imports**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/agent2_fetcher.py research_assistant/agents/agent2_fetcher.py
cp ../tech_ireland/agent3_ingestor.py research_assistant/agents/agent3_ingestor.py
```

- [ ] **Step 2: Have Agent 2 write `DownloadedPaper` records**

Agent 2 keys `downloaded.json` by `source_key` with a dict per paper. Build those through the schema. At the point a download succeeds:

```python
# after a successful fetch, replacing the inline dict
downloaded[key] = DownloadedPaper(
    key=key,
    path=dest,
    provider=provider,          # "unpaywall" | "europepmc" | "arxiv"
    fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    title=ref.get("title"),
    raw_reference=ref.get("raw_reference"),
    doi=doi,
    arxiv_id=ref.get("arxiv_id"),
).to_dict()
```

Add `from research_assistant.schemas import DownloadedPaper`. Leave `failed_downloads.json` as plain dicts — it records *why* a fetch failed and has no downstream consumer that needs a contract.

- [ ] **Step 3: Delete Agent 3's legacy-shape shim**

`_pdfs_from_manifest()` exists only to tolerate two incompatible manifest shapes. With `DownloadedPaper` as the contract, the guessing goes:

```python
# before — accepts both {key: record} and legacy {path: citation}
def _pdfs_from_manifest(downloaded):
    pdfs = {}
    for key, value in downloaded.items():
        if not isinstance(value, dict):
            pdfs[key] = value                      # legacy {path: citation}
            continue
        path = value.get("path")
        if not path:
            logger.warning("No path recorded for %s — skipping.", key)
            continue
        pdfs[path] = value.get("title") or value.get("raw_reference") or key
    return pdfs
```
```python
# after
def _pdfs_from_manifest(downloaded: dict) -> dict[str, str]:
    """Turn Agent 2's manifest into the {pdf_path: citation_label} ingest wants.

    The citation label becomes citation_source on every chunk, and from there
    the key in a cited draft's mapping — so prefer the parsed title over the
    raw reference string, and never fall back to the opaque source key.
    """
    pdfs = {}
    for key, record in downloaded.items():
        try:
            paper = DownloadedPaper.from_dict(record)
        except SchemaError as exc:
            logger.warning("Skipping malformed manifest entry %s: %s", key, exc)
            continue
        pdfs[paper.path] = paper.title or paper.raw_reference or paper.key
    return pdfs
```

Add `from research_assistant.schemas import DownloadedPaper, SchemaError`. A malformed entry is now *named* in a warning instead of silently becoming a missing-file lookup.

- [ ] **Step 4: Port `tests/test_fetcher.py`**

```bash
cp ../citation_builder/test_fetcher.py tests/test_fetcher.py
```
Change its import to `from research_assistant.agents.agent2_fetcher import paper_filename`. The test needs no network — it only exercises filename derivation.

Add one test for the shim's replacement:

```python
class TestManifestParsing(unittest.TestCase):
    def test_malformed_entry_is_skipped_not_treated_as_a_path(self):
        """The old shim turned a record with no path into a missing file."""
        from research_assistant.agents.agent3_ingestor import _pdfs_from_manifest

        pdfs = _pdfs_from_manifest({
            "doi:10.1/good": {
                "key": "doi:10.1/good", "path": "a.pdf",
                "provider": "arxiv", "fetched_at": "t", "title": "Good",
            },
            "doi:10.1/bad": {"key": "doi:10.1/bad", "provider": "arxiv"},
        })
        self.assertEqual(pdfs, {"a.pdf": "Good"})

    def test_title_is_preferred_over_the_opaque_key_as_the_label(self):
        from research_assistant.agents.agent3_ingestor import _pdfs_from_manifest

        pdfs = _pdfs_from_manifest({
            "doi:10.1/x": {
                "key": "doi:10.1/x", "path": "a.pdf", "provider": "arxiv",
                "fetched_at": "t", "title": "Real Title",
                "raw_reference": "Smith et al.",
            },
        })
        self.assertEqual(pdfs, {"a.pdf": "Real Title"})
```

- [ ] **Step 5: Run the tests**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_fetcher.py -v`
Expected: PASS — the identity tests plus the two manifest tests.

- [ ] **Step 6: Checkpoint** — full suite, plus the import sweep from Global Constraints, which should now pass completely.

---

### Task 10: Agents 5, 6 and 7 — port onto `shared.llm`

**Files:**
- Create: `research_assistant/agents/agent5_batch_citer.py`, `agent6_manual_ingestor.py`, `agent7_research_chat.py`
- Test: `tests/test_batch_citer.py` (ported, mock retargeted)

**Interfaces:**
- Consumes: `shared.llm.chat` / `chat_stream` (Task 4), `shared.db.load_search_resources`, `shared.search.hybrid_search`, `shared.ingestion.ingest_pdfs`
- Produces: `agent5.run_batch_citer(file_path, out_path) -> str|None`, `agent5.split_into_sentences(text) -> list[str]`, `agent5._cite_keys(text) -> set[str]`, `agent5._batch_needs_citation(sentences) -> list[bool]`; `agent6.ingest_manual_pdf(pdf_path, citation_string=None, workers=1) -> dict`; `agent7.ResearchChat(top_k=5)` with `.chat_stream(msg)`, `.clear_history()`, `.export_conversation()`, `.last_sources`; `agent7.run_repl(agent, banner=None)`

These come from `citation_builder`, where they call `ollama` directly. After this task no module outside `shared/llm.py` imports a provider SDK.

- [ ] **Step 1: Copy all three**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../citation_builder/agent5_batch_citer.py ../citation_builder/agent6_manual_ingestor.py \
   ../citation_builder/agent7_research_chat.py research_assistant/agents/
```

- [ ] **Step 2: Port Agent 5 off `ollama`**

Two call sites. Replace `import ollama` with `from research_assistant.shared.llm import chat`.

```python
# before — _batch_needs_citation
response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
try:
    answer = response.message.content
except AttributeError:
    answer = response["message"]["content"]
```
```python
# after
answer = chat([{"role": "user", "content": prompt}]).content
```

```python
# before — _cite_sentence_with_reasoning
response = ollama.chat(model=LLM_MODEL, messages=[...])
try:
    raw = response.message.content.strip()
except AttributeError:
    raw = response["message"]["content"].strip()
prompt_tokens = getattr(response, "prompt_eval_count", "N/A")
completion_tokens = getattr(response, "eval_count", "N/A")
```
```python
# after
result = chat([
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": user_prompt},
])
raw = result.content.strip()
prompt_tokens, completion_tokens = result.prompt_tokens, result.completion_tokens
```

`ChatResult` already carries both token counts, so the log line is unchanged. Drop the now-unused `from config import LLM_MODEL` — `chat()` defaults to it.

Preserve exactly: the verdict-alignment check that raises on a missing verdict, and the `run_batch_citer` abort that returns `None` without writing when it fails after retries. A misaligned verdict list attributes one sentence's decision to another.

- [ ] **Step 3: Port Agent 7 onto `chat_stream`**

```python
# before
import ollama
...
@retry(max_retries=3, backoff=2.0)
def _generate(self, messages, stream=False):
    response = ollama.chat(model=CHAT_MODEL, messages=messages,
                           stream=stream, options=CHAT_OLLAMA_OPTIONS)
    ...
```
```python
# after
from research_assistant.shared.llm import chat_stream
...
def _generate_stream(self, messages):
    """Delegates to the backend-agnostic streaming layer.

    No @retry here: the generator has already yielded tokens to the caller by
    the time most failures surface, so a retry would replay a partial answer.
    """
    return chat_stream(messages, model=CHAT_MODEL, options=CHAT_OLLAMA_OPTIONS)
```

In `chat_stream()` (the method), replace the chunk-unwrapping loop — `chat_stream()` (the function) already yields plain strings:

```python
# before
for chunk in self._generate(messages, stream=True):
    try:
        content = chunk.message.content
    except AttributeError:
        content = chunk["message"]["content"]
    full_answer += content
    yield content
```
```python
# after
for content in self._generate_stream(messages):
    full_answer += content
    yield content
```

Rename the method to `stream_turn()` to avoid shadowing the imported `chat_stream`, and update its two call sites in `run_repl()`. Everything else — history trimming to 10 turns, `/clear`, `/sources`, `/export` — is unchanged.

- [ ] **Step 4: Port Agent 6**

Imports only — it already delegates to `shared.ingestion.ingest_pdfs()`. Apply the Global Constraints rule and confirm `watchdog` is importable (added in Task 1 Step 3).

- [ ] **Step 5: Confirm no provider SDK leaks remain**

Run:
```bash
cd /Users/shardul/Downloads/research_assistant && \
grep -rn "^import ollama\|^import openai\|^from ollama\|^from openai" research_assistant/ \
  --include=*.py | grep -v "shared/llm.py" && echo "LEAK FOUND" || echo "clean"
```
Expected: `clean`

- [ ] **Step 6: Port `tests/test_batch_citer.py` with the mock retargeted**

```bash
cp ../citation_builder/test_batch_citer.py tests/test_batch_citer.py
```

The existing helper wraps text in ollama's response shape; it now wraps in `ChatResult`:

```python
# before
def _reply(text):
    """Wrap *text* in the object shape ollama.chat returns."""
    return types.SimpleNamespace(message=types.SimpleNamespace(content=text))
```
```python
# after
from research_assistant.shared.llm import ChatResult

def _reply(text):
    """Wrap *text* in what shared.llm.chat returns."""
    return ChatResult(content=text)
```

Change the import to `from research_assistant.agents.agent5_batch_citer import _batch_needs_citation, _cite_keys`, and retarget every `@patch("agent5_batch_citer.ollama.chat")` to `@patch("research_assistant.agents.agent5_batch_citer.chat")`.

Add a test for the property the port must not lose:

```python
class TestVerdictAlignment(unittest.TestCase):
    def test_missing_verdict_raises_rather_than_padding(self):
        """A short reply must not silently shift verdicts onto wrong sentences."""
        with patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("1. YES\n2. NO")):
            with self.assertRaises(ValueError):
                _batch_needs_citation(["one", "two", "three"])

    def test_verdicts_are_indexed_by_number_not_line_position(self):
        """A preamble line must not shift every verdict by one."""
        with patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("Here you go:\n\n1. YES\n\n2. NO\n")):
            self.assertEqual(_batch_needs_citation(["one", "two"]), [True, False])
```

Note `_batch_needs_citation` is wrapped in `@retry(max_retries=2)`, so the raising test will see three calls before the exception surfaces — that is expected and the assertion is on the final raise.

- [ ] **Step 7: Run the tests**

Run: `cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -m pytest tests/test_batch_citer.py -v`
Expected: PASS.

- [ ] **Step 8: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 11: `orchestrate.py` — remove the GROBID dead end

**Files:**
- Create: `orchestrate.py` (repo root)

**Interfaces:**
- Consumes: agents 0–3 (Tasks 7–9), `shared.ingestion.ingest_pdfs`, `shared.retrieve.research_answer`
- Produces: `build_graph()`, `run(query, workers=1, force=False, ask=False, seed_url=None) -> int`

- [ ] **Step 1: Copy and rewrite imports**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/orchestrate.py orchestrate.py
```

The four bare agent imports become package imports:
```python
from research_assistant.agents import (
    agent0_discoverer, agent1_extractor, agent2_fetcher, agent3_ingestor,
)
from research_assistant.config import EXTRACTED_CITATIONS_PATH, GROBID_SERVER
from research_assistant.shared.ingestion import ingest_pdfs
from research_assistant.shared.log import get_logger
```
and the function-local `from shared import retrieve` in `respond()` becomes `from research_assistant.shared import retrieve`.

- [ ] **Step 2: Make `extract` non-terminal**

Agent 1 now always produces *something* unless both strategies fail on every PDF. The node reports which path was taken instead of ending the run:

```python
# before
def extract(state: PipelineState) -> dict:
    _banner("extract — mining the seed's reference list (GROBID)")
    agent1_extractor.run_extractor()
    if not os.path.exists(EXTRACTED_CITATIONS_PATH):
        return {"references_ok": False, "stopped": (
            "GROBID returned no references — the server at "
            f"{GROBID_SERVER} is down, asleep, or rate-limiting. "
            f"Check: curl {GROBID_SERVER}/api/isalive")}
    return {"references_ok": True}
```
```python
# after
def extract(state: PipelineState) -> dict:
    _banner("extract — mining the seed's reference list")
    result = agent1_extractor.run_extractor()

    if not result or not result.get("references"):
        # Both strategies came up empty: GROBID down *and* no recognisable
        # reference numbering. There is nothing to fetch, so stop — but say
        # which of the two failed.
        return {"references_ok": False, "stopped": (
            "No references could be extracted from the seed paper. GROBID at "
            f"{GROBID_SERVER} is unreachable and pattern-based extraction found "
            "no numbered reference list — the PDF may be a scan with no text "
            f"layer. Check: curl {GROBID_SERVER}/api/isalive")}
    return {"references_ok": True, "extraction": result}
```

Add `extraction: Optional[dict]` to `PipelineState`. The `_after_extract` conditional edge stays — it now fires only when extraction genuinely produced nothing, not merely because GROBID was down.

- [ ] **Step 3: Verify the graph builds and routes correctly**

Run:
```bash
cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -c "
import orchestrate as o
g = o.build_graph(); assert g is not None
assert o._after_extract({'references_ok': True}) == 'fetch'
assert o._after_discover({'seed_path': 'x.pdf'}) == 'ingest_seed'
assert o._after_discover({'ask': True}) == 'fallback'
assert o._after_ingest_refs({'ask': True}) == 'respond'
print('graph ok')"
```
Expected: `graph ok`

- [ ] **Step 4: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 12: `watch.py` — the daemon entry point

**Files:**
- Create: `watch.py` (repo root)

**Interfaces:**
- Consumes: agents 1, 2, 3, 5, `shared.ingestion.ingest_pdfs`, cooldown constants from `config`
- Produces: `sync_database(workers=1)`, `Orchestrator`, `main()`

`citation_builder`'s `master_orchestrator.py`, renamed. A genuinely different mode from `orchestrate.py` — reactive and directory-driven rather than a single query-driven run — so both are kept over the same stage functions.

- [ ] **Step 1: Copy and rename**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../citation_builder/master_orchestrator.py watch.py
```

- [ ] **Step 2: Rewrite imports and agent references**

```python
from research_assistant.agents import (
    agent1_extractor, agent2_fetcher, agent3_ingestor, agent5_batch_citer,
)
from research_assistant.config import (
    RAW_DIR, DRAFTS_DIR, PULLED_PDFS_DIR,
    PDF_COOLDOWN_SECONDS, DRAFT_COOLDOWN_SECONDS, MANUAL_COOLDOWN_SECONDS,
    DEFAULT_WORKERS,
)
from research_assistant.shared.ingestion import ingest_pdfs
from research_assistant.shared.log import get_logger
```

- [ ] **Step 3: Update the docstring and CLI description**

The module docstring references `agent_graph.py` in `attic/`, which does not exist here. Replace the "Orchestration & Control Flow" paragraph with:

```python
"""
watch.py — reactive, directory-driven pipeline runner.

Watches three directories and runs the matching stages when files appear:

    data/raw/          → Agent 1 (extract) → Agent 2 (fetch) → Agent 3 (ingest)
    data/pulled_pdfs/  → batch ingest via shared.ingestion
    data/drafts/       → Agent 5 (batch cite)

Debounce timers and per-directory cooldowns stop a burst of dropped files from
launching several ingests at once. These sequences are fixed, so they are called
directly rather than planned by a model.

For a single research idea end to end, use orchestrate.py instead.
"""
```

- [ ] **Step 4: Point the directories at `DATA_DIR`**

The three watched paths come from `config`, which Task 1 re-anchored under `data/`. Confirm `sync_database()` and each handler use the config constants rather than bare relative strings, and that `main()` creates the directories before the observer starts:

```python
for directory in (RAW_DIR, PULLED_PDFS_DIR, DRAFTS_DIR):
    os.makedirs(directory, exist_ok=True)
```

- [ ] **Step 5: Verify it imports and its handlers are wired**

Run:
```bash
cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -c "
import watch
assert callable(watch.sync_database) and callable(watch.main)
assert watch.Orchestrator is not None
print('watch ok')"
```
Expected: `watch ok`

- [ ] **Step 6: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 13: `app.py` — two new tabs

**Files:**
- Create: `app.py` (repo root)

**Interfaces:**
- Consumes: `orchestrate.build_graph`, `agent4_assistant.suggest_citation`, `agent5_batch_citer.run_batch_citer`, `agent7_research_chat.ResearchChat`, `shared.db.load_search_resources`

- [ ] **Step 1: Copy and rewrite imports**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/app.py app.py
```

`import config` → `from research_assistant import config`. The function-local `import agent4_assistant` and `from shared.db import load_search_resources` become package imports.

- [ ] **Step 2: Extract the duplicated passages block**

The retrieved-context expander is written verbatim in both `_render_suggestion` and `_render_build`. Replace both with one function:

```python
def _render_passages(passages, label="Retrieved context"):
    """The retrieved-chunk expander, shared by every tab that shows one."""
    if not passages:
        return
    with st.expander(f"{label} · {len(passages)} passage(s)"):
        for i, p in enumerate(passages, 1):
            m = p.get("metadata") or {}
            st.caption(
                f"{i}. **{m.get('citation_source', '?')}** — "
                f"{m.get('document', '?')} · p.{m.get('page', '?')}"
            )
            st.text((p.get("text") or "")[:900])
            if i < len(passages):
                st.divider()
```

Call it from both sites.

- [ ] **Step 3: Widen the tab bar**

```python
tab_build, tab_cite, tab_batch, tab_chat, tab_help = st.tabs(
    ["Research a topic", "Cite a draft", "Cite a whole draft", "Research chat",
     "How to use"]
)
```

- [ ] **Step 4: Add the batch-cite tab**

```python
with tab_batch:
    st.caption(
        "Upload a plain-text draft. Every sentence that makes a factual claim "
        "is checked against the corpus and cited where a source supports it."
    )
    if chunks == 0:
        st.info("No corpus yet — build one in **Research a topic** first.", icon="📭")

    uploaded = st.file_uploader("Draft (.txt)", type=["txt"], key="batch_upload")
    pasted = st.text_area("…or paste it here", height=200, key="batch_paste")
    run_batch = st.button("Cite the draft", type="primary", disabled=chunks == 0)

    if run_batch and (uploaded or pasted.strip()):
        import tempfile
        from research_assistant.agents import agent5_batch_citer

        os.makedirs(config.DRAFTS_DIR, exist_ok=True)
        text = uploaded.read().decode("utf-8") if uploaded else pasted
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", dir=config.DRAFTS_DIR, delete=False, encoding="utf-8"
        ) as fh:
            fh.write(text)
            draft_path = fh.name
        out_path = draft_path.replace(".txt", "_cited.txt")

        with st.spinner("Checking each sentence and retrieving sources…"):
            try:
                written = agent5_batch_citer.run_batch_citer(draft_path, out_path)
            except Exception as e:  # noqa: BLE001
                st.exception(e)
                written = None
        st.session_state["batch_result"] = written

    written = st.session_state.get("batch_result")
    if written is None and st.session_state.get("batch_result", "unset") != "unset":
        # run_batch_citer returns None only when it aborted before writing.
        st.error(
            "The citation-need check could not be aligned to the draft's "
            "sentences, so nothing was written. Try again, or shorten the draft "
            "if the model keeps truncating its reply.",
            icon="⚠️",
        )
    elif written:
        with open(written, encoding="utf-8") as fh:
            cited = fh.read()
        st.markdown("###### Cited draft")
        st.text_area("Result", cited, height=260, key="batch_out")
        st.download_button("Download cited draft", cited,
                           file_name=os.path.basename(written))

        mapping_path = written.replace(".txt", "_citations.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, encoding="utf-8") as fh:
                mapping = json.load(fh)
            st.markdown(f"**Sources cited** — {len(mapping)}")
            st.json(mapping, expanded=False)

        report_path = written.replace(".txt", "_report.md")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as fh:
                report = fh.read()
            with st.expander("Per-sentence decisions"):
                st.markdown(report)
```

- [ ] **Step 5: Add the research-chat tab**

```python
with tab_chat:
    st.caption("Multi-turn conversation grounded in the ingested corpus.")
    if chunks == 0:
        st.info("No corpus yet — build one in **Research a topic** first.", icon="📭")
    else:
        if "chat_agent" not in st.session_state:
            from research_assistant.agents.agent7_research_chat import ResearchChat

            st.session_state["chat_agent"] = ResearchChat(top_k=5)
        agent = st.session_state["chat_agent"]

        c1, c2 = st.columns([1, 4])
        if c1.button("Clear"):
            agent.clear_history()
            st.rerun()
        if c2.button("Export conversation"):
            st.success(f"Saved to {agent.export_conversation()}")

        for msg in agent.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question := st.chat_input("Ask about the literature…"):
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    st.write_stream(agent.stream_turn(question))
                except Exception as e:  # noqa: BLE001
                    st.exception(e)
            if agent.last_sources:
                with st.expander(f"Sources · {len(agent.last_sources)}"):
                    for s in agent.last_sources:
                        st.caption(f"**{s['document']}** — {s['citation']}")
```

`stream_turn` is the method renamed in Task 10 Step 3; it appends to `agent.history` itself, so the loop above renders prior turns on the next rerun.

- [ ] **Step 6: Verify the app parses and its helpers resolve**

Streamlit scripts execute top to bottom, so import-checking it would run the UI. Compile-check instead, then confirm the pieces it calls exist:

```bash
cd /Users/shardul/Downloads/research_assistant && python -m py_compile app.py && \
CITATION_LOG_FILE=0 python -c "
from research_assistant.agents.agent7_research_chat import ResearchChat
from research_assistant.agents import agent5_batch_citer
assert hasattr(ResearchChat, 'stream_turn'), 'rename from Task 10 not applied'
assert hasattr(ResearchChat, 'clear_history') and hasattr(ResearchChat, 'export_conversation')
assert callable(agent5_batch_citer.run_batch_citer)
print('app wiring ok')"
```
Expected: `app wiring ok`

- [ ] **Step 7: Checkpoint** — `CITATION_LOG_FILE=0 python -m pytest tests/ -v`

---

### Task 14: Docs, CI, and the final sweep

**Files:**
- Create: `README.md`, `ARCHITECTURE.md`, `HOW_TO_USE.md`, `Dockerfile`, `.dockerignore`, `.github/workflows/tests.yml`

- [ ] **Step 1: Copy the docs and container files**

```bash
cd /Users/shardul/Downloads/research_assistant
cp ../tech_ireland/{README.md,ARCHITECTURE.md,HOW_TO_USE.md,Dockerfile,.dockerignore} .
cp ../citation_builder/.github/workflows/tests.yml .github/workflows/tests.yml
```

- [ ] **Step 2: Update the CI workflow for the new layout**

```yaml
      - name: Run tests
        env:
          CITATION_LOG_FILE: "0"
        run: |
          python -m pip install -e .
          python -m pytest tests/ -v

      - name: Check every module still imports
        env:
          CITATION_LOG_FILE: "0"
        run: |
          for m in research_assistant.config research_assistant.prompts \
                   research_assistant.schemas research_assistant.shared.log \
                   research_assistant.shared.retry research_assistant.shared.manifest \
                   research_assistant.shared.llm research_assistant.shared.db \
                   research_assistant.shared.search research_assistant.shared.source_key \
                   research_assistant.shared.fetch research_assistant.shared.ingestion \
                   research_assistant.agents.agent1_extractor \
                   research_assistant.agents.agent2_fetcher \
                   research_assistant.agents.agent3_ingestor; do
            python -c "import $m" || exit 1
          done
```

Add `pytest` to `requirements-test.txt` if absent (the source suite used `unittest discover`; this plan runs pytest, which collects `unittest.TestCase` classes unchanged).

- [ ] **Step 3: Rewrite `README.md`**

Sections to change from `tech_ireland`'s version:

- **Pipeline table** — add rows for Agent 5 (batch citer), Agent 6 (manual ingestor) and Agent 7 (research chat), and rewrite the Agent 1 row: *"Sends every PDF in `data/raw/` to a GROBID server and parses the TEI output… When GROBID is unreachable, or returns no reference list for a paper, it falls back to pattern-matching a numbered reference list out of the extracted text — weaker (raw strings only, no DOIs), but it keeps the pipeline moving. The path taken is recorded per run in `extracted_citations.json`."*
- **Layout block** — replace with the tree from the spec's §3.
- **Configuration notes** — `CITATION_DATA_DIR` now defaults to `<project>/data` rather than the repo root.
- Delete the closing paragraph beginning *"Configuration in config.py and prompts.py also anticipate further agents…"* — those agents are here now.
- **Usage** — add:
  ```bash
  # 5. Cite a whole draft
  python -m research_assistant.agents.agent5_batch_citer --file draft.txt --out cited.txt

  # 7. Chat with the corpus
  python -m research_assistant.agents.agent7_research_chat

  # Or run the reactive daemon instead of the per-query pipeline
  python watch.py
  ```
- **Development** — document `pip install -e .` and the pytest command.

- [ ] **Step 4: Update `ARCHITECTURE.md`**

Add a section recording the two decisions this merge made that neither source repo documents:

1. **Why Agent 1 keeps both extraction strategies** — GROBID's structured TEI is materially better input for Agent 2's DOI resolution, but an external Java service is a single point of failure for the whole pipeline; the regex path is a floor, not a competitor. Provenance is recorded per reference so downstream consumers can weigh it.
2. **Why the ingestion manifest is a JSON file rather than a collection scan** — parsing is the expensive stage, the check runs on every file, and the previous implementation read every metadata record in the corpus before ingesting one PDF.

Update its file paths to the package layout.

- [ ] **Step 5: Update `HOW_TO_USE.md`**

It is rendered inside the app (tab 5 reads it from `PROJECT_ROOT`), so it must describe all five tabs. Add a section each for *Cite a whole draft* and *Research chat*. Confirm the app still finds it:

```bash
cd /Users/shardul/Downloads/research_assistant && CITATION_LOG_FILE=0 python -c "
import os
from research_assistant import config
p = os.path.join(config.PROJECT_ROOT, 'HOW_TO_USE.md')
assert os.path.exists(p), p
print('HOW_TO_USE.md reachable at', p)"
```

- [ ] **Step 6: Update the Dockerfile**

Two changes: install the package (`RUN pip install -e .`) and set `CITATION_DATA_DIR` to a writable path so a read-only image still runs:

```dockerfile
ENV CITATION_DATA_DIR=/data
RUN mkdir -p /data
```

Confirm the `CMD` still points at `app.py` at the repo root.

- [ ] **Step 7: Final verification sweep**

Run all four gates:

```bash
cd /Users/shardul/Downloads/research_assistant

# 1. Full suite
CITATION_LOG_FILE=0 python -m pytest tests/ -v

# 2. Import sweep (from Global Constraints)
CITATION_LOG_FILE=0 python -c "
import importlib
for m in ['research_assistant.config','research_assistant.prompts','research_assistant.schemas','research_assistant.shared.log','research_assistant.shared.retry','research_assistant.shared.manifest','research_assistant.shared.llm','research_assistant.shared.db','research_assistant.shared.search','research_assistant.shared.source_key','research_assistant.shared.fetch','research_assistant.shared.ingestion','research_assistant.agents.agent1_extractor','research_assistant.agents.agent2_fetcher','research_assistant.agents.agent3_ingestor']:
    importlib.import_module(m); print('ok', m)"

# 3. No stale imports, no provider SDK leaks
grep -rn "from shared\.\|^from config import\|^from prompts import" research_assistant/ *.py && echo "STALE IMPORTS" || echo "imports clean"
grep -rn "^import ollama\|^import openai\|^from ollama\|^from openai" research_assistant/ --include=*.py | grep -v "shared/llm.py" && echo "SDK LEAK" || echo "no leaks"

# 4. Entry points compile
python -m py_compile app.py orchestrate.py watch.py && echo "entry points ok"
```

Expected: suite green; every module `ok`; `imports clean`; `no leaks`; `entry points ok`.

- [ ] **Step 8: Report what was *not* verified**

State plainly in the handoff: the suite and sweeps run without Ollama, GROBID or network, so they confirm the merge is structurally sound but **do not** exercise a live model call, a real GROBID round trip, or an end-to-end corpus build. Offer to run a live `orchestrate.py --query … --ask` if the user's Ollama daemon is up.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2 provenance / not carried over | 1, 5, 8, 9, 10 |
| §3 layout | 1 |
| §4.1 package | 1 |
| §4.2 `DATA_DIR` → `data/` | 1 |
| §5.1 Agent 1 both paths | 7, 11 |
| §5.2 `chat_stream` | 4, 10 |
| §5.3 `schemas.py` | 2, 8, 9 |
| §5.4 `manifest.py` | 3, 6 |
| §5.5 config (`DETECTRON_CONFIG` fix, env helpers) | 1 |
| §5.6 app.py two tabs + dedupe | 13 |
| §5.7 `watch.py` | 12 |
| §6 data flow | 7, 9, 11 |
| §7 error handling preserved | 7, 10 |
| §8 testing | 2, 3, 4, 5, 6, 7, 9, 10, 14 |
| §9 out of scope (no eval) | 1 (paths dropped), 14 |
| §10 risks | 14 Step 7 gates |

No gaps.

**Corrections made against the spec during planning:**
- `Reference.source_document` → **`source_file`**. The spec had it wrong; `parse_reference()` and `summarise()` both emit and read `source_file`, and Agent 2 consumes those dicts.
- `Reference` gains `container` and `xml_id` — both are produced by the existing TEI parser and would have been silently dropped.
- `Reference.raw_reference` is required but `title`-derived fallback is allowed in `_references_from_grobid`, since GROBID sometimes omits the raw note.

**Type consistency:** `pdf_key()` (Task 6) is the key type stored by `manifest.add_many()` (Task 3) and returned by `get_ingested_documents()`. `ChatResult` (Task 4) is what `_reply()` builds in Task 10's tests and what Agent 5 unpacks. `DownloadedPaper.path` (Task 2) is the dict key Agent 3 builds in Task 9. `stream_turn` is renamed once in Task 10 and called in Task 13 — checked in Task 13 Step 6.

**Placeholder scan:** no TBD/TODO; every code step carries the actual code; no "similar to Task N".
