# Citer Eval and Judge Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Agent 5 (the citer) a ground truth built from the seed papers' own citations, then use the auditor's judge as its acceptance test, then fix its query — each gated by the measurement the first step creates.

**Architecture:** The per-sentence citing logic is extracted from `run_batch_citer` into `cite_sentence()` returning a `CiteResult` whose candidates carry the document basename; `hybrid_search` gains `exclude_docs` so an evaluation cannot retrieve the seed's own text. A new `research_assistant/eval/` package builds gold records from a seed's TEI plus the download manifest (cited sentences with in-corpus references, and a matched sample of uncited sentences) and scores a citer run on need, target and end-to-end accuracy; `scripts/evaluate_citer.py` mirrors `evaluate_judge.py`. Behind two config flags, `cite_sentence` then judges each candidate in retrieval order and inserts the first that passes deterministically (A), and `run_batch_citer` becomes paragraph-aware and contextualizes queries once per paragraph (B).

**Tech Stack:** Python 3.12, `unittest.TestCase` collected by pytest, BeautifulSoup (`"xml"` parser), existing `shared.search.hybrid_search`, `judgement.judge`, `shared.seed_audit` helpers, `shared.llm.chat`.

**Spec:** `notimportant/superpowers/specs/2026-09-18-citer-eval-and-judge-gate-design.md` — §2 is the evidence, §3.1–3.4 the four pieces, §3.2's table the metric definitions.

## Global Constraints

- Tests: `python3 -m pytest tests/<file>.py -q` from the repo root (base env; no `chromadb`, model and index always stubbed). Full suite: `python3 -m pytest tests/ -q` — 887 tests at `019d492`, all green, and must stay green after every task.
- Live runs: `CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python …` (call it `$PY`). Gemini key is in `.env`.
- No model call in any test. Live runs are the deliverables of Tasks 6, 9 and 11 only.
- `CITATION_CITER_JUDGE` and `CITATION_CITER_CONTEXTUALIZE` default **False** until their gate passes; the gate task flips the default and records the numbers in its commit.
- With both flags off, `run_batch_citer`'s three output files are byte-identical to today's for the same inputs and model replies — the existing `tests/test_batch_citer.py` pins this.
- `shared.claim_text` imports `split_into_sentences` from `agent5_batch_citer` today, so agent5 cannot import from `claim_text` until Task 10 moves the splitter into `claim_text` (agent5 re-exports it; every existing `from agent5_batch_citer import split_into_sentences` keeps working).
- `agent8_verifier` imports `agent5_batch_citer`, so `agent5_batch_citer` never imports `agent8_verifier` at module level; `assemble_evidence` is imported inside the function that uses it. `judgement.judge` imports no agent and may be imported at module top.
- Author citations are a floor: the metrics `render()` prints that sentence every time.
- Commit after each task. Subject in the repo's style (`feat(citer): …`, `feat(eval): …`, `fix(…): …`), body says why, message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Gold record shape (used by Tasks 3, 4, 5, 10): `{"id", "seed", "kind": "cited"|"uncited", "sentence", "context", "section", "section_heading", "paragraph_id", "sentence_index", "author_documents": [str], "author_refs": [{"index", "title"}], "roles": [str]}`.
- Harness result shape (used by Tasks 4, 5, 10): `{"case": <gold record>, "needs_cite": bool, "cite": <dataclasses.asdict(CiteResult)> | None, "run": int, "error": str | None}`.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `research_assistant/shared/search.py` | `exclude_docs` on `hybrid_search` and `_where` | 1 |
| `tests/test_search.py` | `TestExcludeDocs` | 1 |
| `research_assistant/agents/agent5_batch_citer.py` | `CiteResult`, `cite_sentence`, `_cite_by_judge`, `_insert_cite`, paragraph-aware `run_batch_citer`, report additions | 2, 7, 8, 10 |
| `tests/test_batch_citer.py` | tests for the above | 2, 7, 8, 10 |
| `research_assistant/config.py` | `CITATION_CITER_JUDGE`, `CITATION_CITER_CONTEXTUALIZE` | 7, 10 |
| `research_assistant/eval/__init__.py` (new) | package docstring | 3 |
| `research_assistant/eval/citer_gold.py` (new) | `build(seed_pdf_path)`, `load_cases()`, `CASES_DIR` | 3 |
| `research_assistant/eval/cases/` (new) | `citer_<stem>.jsonl`, tracked | 6 |
| `research_assistant/eval/citer_metrics.py` (new) | `score`, `render`, `compare`, `FLOOR_NOTE` | 4 |
| `scripts/evaluate_citer.py` (new) | `--build`, `--score`, `--compare`; `seed_documents()`, `run()` | 5, 10 |
| `pyproject.toml` | package-data for `research_assistant.eval` | 3 |
| `tests/test_citer_gold.py`, `tests/test_citer_metrics.py`, `tests/test_evaluate_citer.py` (new) | tests | 3, 4, 5, 10 |

---

## Phase 0 — The seam

### Task 1: `hybrid_search` can exclude documents

**Files:**
- Modify: `research_assistant/shared/search.py:28-38` (`_where`), `:41-52` (signature), `:97-101` (sparse filter), `:105` (dense where)
- Test: `tests/test_search.py`

**Interfaces:**
- Produces: `hybrid_search(..., exclude_docs: set | None = None)` — chunks whose `metadata["document"]` is in `exclude_docs` are dropped from both retrievers. `_where(doc_filter, exclude_types, exclude_docs=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py` (the file already defines `FakeBM25`, `FakeCollection`, `FakeEmbeddings`, `_corpus` and imports `hybrid_search`):

```python
class TestExcludeDocs(unittest.TestCase):
    """The citer's evaluation runs on a seed paper's own sentences. Two of the
    three seeds are in the chunk index, so without an exclusion retrieval
    would hand the citer the seed's own text and the score would be a lie."""

    class _Coll(FakeCollection):
        def __init__(self, ids):
            super().__init__(ids)
            self.where = None

        def query(self, **kwargs):
            self.where = kwargs.get("where")
            return super().query(**kwargs)

    def test_sparse_side_drops_excluded_documents(self):
        texts = ["a", "b", "c"]
        metas = [{"document": "seed.pdf"}, {"document": "ref.pdf"}, {"document": "seed.pdf"}]
        out = hybrid_search("q", FakeCollection([]), FakeBM25([3.0, 2.0, 1.0]), texts, metas,
                            top_k=3, embeddings_model=FakeEmbeddings(), exclude_docs={"seed.pdf"})
        self.assertEqual([r["chunk_index"] for r in out], [1])

    def test_dense_where_carries_the_exclusion(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(), exclude_docs={"doc0.pdf"})
        self.assertEqual(coll.where, {"document": {"$nin": ["doc0.pdf"]}})

    def test_exclusion_combines_with_the_other_filters(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(),
                      doc_filter={"doc0.pdf", "doc1.pdf"}, exclude_docs={"doc1.pdf"}, exclude_types={"caption"})
        self.assertEqual(coll.where, {"$and": [
            {"document": {"$in": ["doc0.pdf", "doc1.pdf"]}},
            {"document": {"$nin": ["doc1.pdf"]}},
            {"type": {"$nin": ["caption"]}},
        ]})

    def test_no_exclusion_changes_nothing(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(), exclude_docs=set())
        self.assertIsNone(coll.where)
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_search.py::TestExcludeDocs -q`
Expected: FAIL — `TypeError: hybrid_search() got an unexpected keyword argument 'exclude_docs'`

- [ ] **Step 3: Implement**

In `research_assistant/shared/search.py`:

(a) `_where` becomes:

```python
def _where(doc_filter, exclude_types, exclude_docs=None):
    """ChromaDB where-clause for the dense side. Two or more conditions need
    $and; one stands alone; none is None (Chroma rejects an empty dict).
    doc_filter sorts for a stable clause; the exclusions sort likewise."""
    clauses = []
    if doc_filter:
        clauses.append({"document": {"$in": sorted(doc_filter)}})
    if exclude_docs:
        clauses.append({"document": {"$nin": sorted(exclude_docs)}})
    if exclude_types:
        clauses.append({"type": {"$nin": sorted(exclude_types)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
```

(b) The signature gains a parameter after `exclude_types`:

```python
    exclude_types: set | None = None,
    exclude_docs: set | None = None,
) -> list[dict]:
```

and the docstring's `Args` gains:

```
        exclude_docs:     Chunk ``metadata["document"]`` values to leave out of
                          both retrievers — the citer's evaluation keeps the
                          seed paper out of its own search.
```

(c) After the `if exclude_types:` sparse filter line add:

```python
    if exclude_docs:
        sparse_top_indices = [i for i in sparse_top_indices if _meta(i).get("document") not in exclude_docs]
```

(d) The dense call's `where=_where(doc_filter, exclude_types)` becomes `where=_where(doc_filter, exclude_types, exclude_docs)`.

Note (a) changes `list(doc_filter)` to `sorted(doc_filter)`. `TestExcludeTypes.test_dense_where_combines_document_and_type_filters` passes a one-element set, so it still passes; the new combined test needs the deterministic order.

- [ ] **Step 4: Run the file**

Run: `python3 -m pytest tests/test_search.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/search.py tests/test_search.py
git commit -m "feat(search): hybrid_search can exclude documents

The citer's evaluation runs on a seed paper's own sentences, and two of
the three seeds are in the chunk index. Without an exclusion, retrieval
hands the citer the seed's own text.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `cite_sentence` — the per-sentence seam

**Files:**
- Modify: `research_assistant/agents/agent5_batch_citer.py:1-18` (imports), before `_generate_report` (new dataclass + function), `:333-397` (the loop in `run_batch_citer`)
- Test: `tests/test_batch_citer.py`

**Interfaces:**
- Consumes: `hybrid_search(..., exclude_docs=)` from Task 1; existing `_cite_sentence_with_reasoning`, `_cite_keys`, `_restore_terminal_punctuation`.
- Produces:

```python
@dataclass
class CiteResult:
    original: str
    cited_text: str
    keys: list[str]            # [] when nothing was cited
    candidates: list[dict]     # [{key, citation, document, chunk_index, rrf_score}] in retrieval order
    reasoning: str
    skip_reason: str | None
    query: str
    verdicts: list[dict]       # empty until Task 7
    partial: bool              # False until Task 7
    cited -> bool              # property: bool(keys)

def cite_sentence(sentence: str, resources: tuple, key_registry: dict, *,
                  context: str | None = None, query: str | None = None,
                  exclude_docs: set | None = None, paragraph_id: str | None = None) -> CiteResult
```

`resources` is `(collection, bm25, texts, metadatas)`. `key_registry` maps `citation_source → "cite_N"` and is mutated (keys only ever added; the next N is `len(key_registry) + 1`). A model or retrieval exception propagates; callers decide.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_batch_citer.py`: extend the import to include `cite_sentence`, `CiteResult`, and append:

```python
class TestCiteSentence(unittest.TestCase):
    """The per-sentence seam. Everything the batch loop did inline lives here,
    so it can be called by an evaluation, gated by a judge, and tested."""

    RES = (None, None, [], [])
    HIT = {
        "text": "Ballistic transport has been observed in graphene at cryogenic temperatures.",
        "chunk_index": 7, "rrf_score": 0.03,
        "metadata": {"citation_source": "Doe, J. et al. (2020)", "document": "doe2020.pdf"},
    }

    def test_a_citation_carries_key_and_document(self):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[self.HIT]), \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: Graphene is ballistic \\cite{cite_1}.\nREASON: Directly reported.")):
            res = cite_sentence("Graphene is ballistic.", self.RES, {})
        self.assertIsInstance(res, CiteResult)
        self.assertTrue(res.cited)
        self.assertEqual(res.keys, ["cite_1"])
        self.assertEqual(res.cited_text, "Graphene is ballistic \\cite{cite_1}.")
        self.assertEqual(res.reasoning, "Directly reported.")
        self.assertEqual(res.candidates, [{
            "key": "cite_1", "citation": "Doe, J. et al. (2020)", "document": "doe2020.pdf",
            "chunk_index": 7, "rrf_score": 0.03,
        }])
        self.assertIsNone(res.skip_reason)
        self.assertEqual(res.query, "Graphene is ballistic.")

    def test_registry_keys_are_stable_across_sentences(self):
        registry = {}
        other = dict(self.HIT, metadata={"citation_source": "Roe 2019", "document": "roe.pdf"})
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search",
                   side_effect=[[self.HIT], [other, self.HIT]]), \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: S.\nREASON: no")):
            cite_sentence("First.", self.RES, registry)
            res = cite_sentence("Second.", self.RES, registry)
        self.assertEqual(registry, {"Doe, J. et al. (2020)": "cite_1", "Roe 2019": "cite_2"})
        self.assertEqual([c["key"] for c in res.candidates], ["cite_2", "cite_1"])

    def test_no_context_and_declined_are_told_apart(self):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[]):
            res = cite_sentence("Nothing here.", self.RES, {})
        self.assertFalse(res.cited)
        self.assertEqual(res.cited_text, "Nothing here.")
        self.assertEqual(res.skip_reason, "no relevant context found in database")
        self.assertEqual(res.candidates, [])

        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[self.HIT]), \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: Nothing here.\nREASON: The context is about something else.")):
            res = cite_sentence("Nothing here.", self.RES, {})
        self.assertFalse(res.cited)
        self.assertEqual(res.skip_reason, "context retrieved but the model did not cite it")
        self.assertEqual(res.reasoning, "The context is about something else.")
        self.assertEqual(len(res.candidates), 1)

    def test_query_and_exclusion_reach_retrieval(self):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[]) as hs:
            res = cite_sentence("This approach works.", self.RES, {},
                                query="recursive Green's function inversion works", exclude_docs={"seed.pdf"})
        self.assertEqual(hs.call_args[0][0], "recursive Green's function inversion works")
        self.assertEqual(hs.call_args.kwargs["exclude_docs"], {"seed.pdf"})
        self.assertEqual(res.query, "recursive Green's function inversion works")
        self.assertEqual(res.original, "This approach works.")
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_batch_citer.py::TestCiteSentence -q`
Expected: FAIL — `ImportError: cannot import name 'cite_sentence'`

- [ ] **Step 3: Implement the dataclass and function**

In `research_assistant/agents/agent5_batch_citer.py`:

(a) Add `from dataclasses import dataclass, field` to the imports (after `import re`).

(b) Directly before `def _generate_report(`, insert:

```python
# ─── The per-sentence seam ────────────────────────────────────────────────────

@dataclass
class CiteResult:
    """What citing one sentence produced.

    ``candidates`` are every chunk retrieval offered, in rank order, each
    with the ``document`` it came from — the name the audit and the citer's
    evaluation key on. The batch loop used to keep only ``citation_source``.
    """
    original: str
    cited_text: str
    keys: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    reasoning: str = ""
    skip_reason: str | None = None
    query: str = ""
    verdicts: list = field(default_factory=list)
    partial: bool = False

    @property
    def cited(self) -> bool:
        return bool(self.keys)


def cite_sentence(sentence, resources, key_registry, *, context=None, query=None,
                  exclude_docs=None, paragraph_id=None) -> CiteResult:
    """Retrieve context for one sentence and ask the model to cite it.

    ``key_registry`` maps citation_source → cite key and is shared across a
    draft so a source keeps its key; entries are only ever added, so the
    next key is len + 1. ``query`` is what to search with (the sentence
    when None); ``exclude_docs`` keeps named documents out of retrieval.
    A model or retrieval failure propagates — the caller decides what a
    failed sentence means.
    """
    collection, bm25, texts, metadatas = resources
    query = query or sentence
    results = hybrid_search(query, collection, bm25, texts, metadatas, top_k=3, exclude_docs=exclude_docs)
    if not results:
        return CiteResult(original=sentence, cited_text=sentence, query=query,
                          skip_reason="no relevant context found in database")

    context_str = ""
    candidates = []
    for r in results:
        meta = r.get("metadata") or {}
        cit_source = meta.get("citation_source", "Unknown")
        # Keys are registered for every retrieved chunk so the model has a
        # stable label to reference, but registration is NOT the same as
        # use — the caller filters the final mapping down to keys that
        # actually made it into the draft.
        if cit_source not in key_registry:
            key_registry[cit_source] = f"cite_{len(key_registry) + 1}"
        cite_key = key_registry[cit_source]
        context_str += f"--- Context (Cite Key: {cite_key}) ---\n{r['text']}\n\n"
        candidates.append({
            "key": cite_key, "citation": cit_source, "document": meta.get("document"),
            "chunk_index": r.get("chunk_index"), "rrf_score": r.get("rrf_score"),
        })

    cited_sentence, reasoning = _cite_sentence_with_reasoning(sentence, context_str)
    keys = sorted(_cite_keys(cited_sentence))
    if keys:
        return CiteResult(original=sentence, cited_text=cited_sentence, keys=keys,
                          candidates=candidates, reasoning=reasoning, query=query)
    # A successful call is not a citation: when the model declines, or the
    # reply could not be parsed, the sentence comes back unchanged.
    return CiteResult(original=sentence, cited_text=sentence, candidates=candidates,
                      reasoning=reasoning, query=query,
                      skip_reason="context retrieved but the model did not cite it")


```

(c) In `run_batch_citer`, replace the loop body from `logger.info(" -> Needs citation. Searching context…")` through `citation_entries.append(entry)` (the end of the `for i, sentence in enumerate(sentences):` loop) with:

```python
        logger.info(" -> Needs citation. Searching context…")
        try:
            res = cite_sentence(sentence, (collection, bm25, texts, metadatas), key_registry)
        except Exception as e:
            logger.error(" -> Error during citing: %s", e)
            cited_sentences.append(sentence)
            entry["skip_reason"] = f"LLM error: {e}"
            citation_entries.append(entry)
            continue

        cited_sentences.append(res.cited_text)
        if res.cited:
            logger.info(" -> Cited: %s", res.cited_text[:80])
            entry["cited"] = True
            entry["cited_text"] = res.cited_text
            entry["reasoning"] = res.reasoning
            entry["sources"] = res.candidates
        else:
            if res.candidates:
                logger.info(" -> Declined: retrieved context did not support the claim.")
                entry["candidates"] = res.candidates
            else:
                logger.info(" -> No context found.")
            entry["skip_reason"] = res.skip_reason
            if res.reasoning:
                entry["reasoning"] = res.reasoning
        citation_entries.append(entry)
```

and delete the now-unused `next_cite_idx = 1` line above the loop (the registry sizes itself).

- [ ] **Step 4: Run the file**

Run: `python3 -m pytest tests/test_batch_citer.py -q`
Expected: all pass, including the pre-existing `TestRunBatchCiter` (its `hybrid_search` stub has no `chunk_index` or `document`; the seam uses `.get`).

- [ ] **Step 5: Commit**

```bash
git add research_assistant/agents/agent5_batch_citer.py tests/test_batch_citer.py
git commit -m "refactor(citer): cite_sentence is the per-sentence seam

The batch loop did retrieval, key registration, the model call and the
parse inline, so nothing per-sentence could be called, gated or scored.
Extracted unchanged into cite_sentence() returning a CiteResult whose
candidates carry the document basename — the name the audit and the
citer's evaluation key on, which the loop had been dropping.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 1 — G: the citer's ground truth

### Task 3: `research_assistant/eval/citer_gold.py` — gold records from a seed's own citations

**Files:**
- Create: `research_assistant/eval/__init__.py`, `research_assistant/eval/citer_gold.py`, `research_assistant/eval/cases/.gitkeep`
- Modify: `pyproject.toml:18-19` (package-data)
- Test: `tests/test_citer_gold.py` (new)

**Interfaces:**
- Consumes: `seed_audit.extract_seed_citation_claims(tei_path) -> list[dict]` (fields `claim, context, section, section_heading, paragraph_id, paragraph_index, sentence_index, ref, resolved, role`); `seed_audit.find_tei_for_seed(seed_path) -> str | None`; `seed_audit._load_downloaded_manifest() -> dict`; `seed_audit._match_downloaded_paper(ref, seed_pdf_name, manifest) -> dict | None`; `seed_audit._paragraph_section(p) -> str`; `claim_text.paragraph_sentences(p) -> list[str]`; `claim_text.sentence_context(sentences, idx) -> str`; `tei_structure.section_breadcrumb(p) -> str`.
- Produces: `citer_gold.build(seed_pdf_path, *, negatives_seed=7) -> list[dict]` (records per Global Constraints); `citer_gold.cited_records(claims, seed_name, manifest) -> list[dict]`; `citer_gold.uncited_pool(tei_path) -> list[dict]`; `citer_gold.write_cases(records, path)`; `citer_gold.load_cases(paths=None) -> list[dict]`; `citer_gold.CASES_DIR: Path`; `citer_gold.MIN_WORDS = 8`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_citer_gold.py`:

```python
"""The citer's ground truth is a seed paper's own citations: which sentence
the author had cite which work. These pin how a record is built — an
in-corpus reference keeps the sentence, an unavailable one is dropped from
it, and the negatives are sampled from paragraphs that cite nothing."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from research_assistant.eval import citer_gold as cg

TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt><title>Seed</title></titleStmt><sourceDesc><p></p></sourceDesc></fileDesc></teiHeader>
  <text><body>
    <div><head n="2.">Results</head>
      <p xml:id="p_cited">Graphene shows ballistic transport at low temperature <ref type="bibr" target="#b0">[1]</ref>. A second sentence cites something we do not hold <ref type="bibr" target="#b1">[2]</ref>.</p>
    </div>
    <div><head n="3.">Discussion</head>
      <p xml:id="p_plain">This paragraph makes no citation at all and has enough words to count. Another sentence with plenty of words that also cites nothing at all here.</p>
      <p xml:id="p_short">Too short to count.</p>
    </div>
  </body>
  <back><div type="references"><listBibl>
    <biblStruct xml:id="b0"><analytic><title level="a" type="main">Held paper</title>
      <author><persName><forename>A.</forename><surname>Held</surname></persName></author>
      <idno type="DOI">10.1/held</idno></analytic></biblStruct>
    <biblStruct xml:id="b1"><analytic><title level="a" type="main">Missing paper</title>
      <author><persName><forename>B.</forename><surname>Missing</surname></persName></author>
      <idno type="DOI">10.1/missing</idno></analytic></biblStruct>
  </listBibl></div></back></text>
</TEI>"""


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tei = os.path.join(self.tmp.name, "seed.grobid.tei.xml")
        with open(self.tei, "w", encoding="utf-8") as fh:
            fh.write(TEI)
        self.held_pdf = os.path.join(self.tmp.name, "doi_10.1_held.pdf")
        with open(self.held_pdf, "wb") as fh:
            fh.write(b"%PDF-1.4 stub")
        self.manifest = {"doi:10.1/held": {
            "key": "doi:10.1/held", "path": self.held_pdf, "doi": "10.1/held",
            "title": "Held paper", "cited_by": "seed.pdf", "xml_id": "b0",
        }}

    def _build(self, **kw):
        with patch.object(cg, "find_tei_for_seed", return_value=self.tei), \
             patch.object(cg, "_load_downloaded_manifest", return_value=self.manifest):
            return cg.build(os.path.join(self.tmp.name, "seed.pdf"), **kw)

    def test_a_cited_sentence_with_an_in_corpus_reference_is_a_record(self):
        cited = [r for r in self._build() if r["kind"] == "cited"]
        self.assertEqual(len(cited), 1)
        rec = cited[0]
        self.assertEqual(rec["sentence"], "Graphene shows ballistic transport at low temperature.")
        self.assertEqual(rec["author_documents"], ["doi_10.1_held.pdf"])
        self.assertEqual(rec["author_refs"], [{"index": 1, "title": "Held paper"}])
        self.assertEqual(rec["roles"], ["evidential"])
        self.assertEqual(rec["section"], "results")
        self.assertEqual(rec["section_heading"], "2. Results")
        self.assertEqual(rec["paragraph_id"], "p_cited")
        # The window is the audit's display form: the marked sentence keeps its
        # rendered citation marker, exactly as the judge sees it today.
        self.assertIn("«Graphene shows ballistic transport at low temperature", rec["context"])
        self.assertIn("A second sentence cites something we do not hold", rec["context"])
        self.assertEqual(rec["seed"], "seed")
        self.assertEqual(rec["id"], "c_seed_0001")

    def test_a_sentence_whose_only_reference_is_not_in_the_corpus_is_dropped(self):
        sentences = [r["sentence"] for r in self._build() if r["kind"] == "cited"]
        self.assertNotIn("A second sentence cites something we do not hold.", sentences)

    def test_negatives_come_from_uncited_paragraphs_and_match_the_cited_count(self):
        out = self._build()
        neg = [r for r in out if r["kind"] == "uncited"]
        self.assertEqual(len(neg), 1)                      # as many as cited records
        self.assertEqual(neg[0]["paragraph_id"], "p_plain")
        self.assertEqual(neg[0]["author_documents"], [])
        self.assertGreaterEqual(len(neg[0]["sentence"].split()), cg.MIN_WORDS)
        self.assertEqual(neg[0]["id"], "u_seed_0001")
        self.assertEqual(neg[0]["section"], "discussion")

    def test_the_short_paragraph_is_never_a_negative(self):
        pool = cg.uncited_pool(self.tei)
        self.assertEqual({p["paragraph_id"] for p in pool}, {"p_plain"})
        self.assertEqual(len(pool), 2)

    def test_negatives_are_sampled_deterministically(self):
        self.assertEqual(self._build(), self._build())
        a = [r["sentence"] for r in self._build(negatives_seed=1) if r["kind"] == "uncited"]
        b = [r["sentence"] for r in self._build(negatives_seed=1) if r["kind"] == "uncited"]
        self.assertEqual(a, b)

    def test_no_tei_raises(self):
        with patch.object(cg, "find_tei_for_seed", return_value=None):
            with self.assertRaises(FileNotFoundError):
                cg.build("/nowhere/seed.pdf")


class TestLoadCases(unittest.TestCase):
    def _rec(self, id, **over):
        d = {"id": id, "seed": "s", "kind": "cited", "sentence": "S.", "context": "«S.»",
             "section": "other", "section_heading": "", "paragraph_id": "p_0", "sentence_index": 0,
             "author_documents": ["a.pdf"], "author_refs": [], "roles": ["evidential"]}
        d.update(over)
        return d

    def test_round_trip_and_duplicate_ids_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "citer_s.jsonl")
            cg.write_cases([self._rec("c_1"), self._rec("u_1", kind="uncited", author_documents=[])], path)
            cases = cg.load_cases([path])
            self.assertEqual([c["id"] for c in cases], ["c_1", "u_1"])
            cg.write_cases([self._rec("c_1"), self._rec("c_1")], path)
            with self.assertRaises(ValueError):
                cg.load_cases([path])

    def test_a_record_missing_a_required_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "citer_s.jsonl")
            rec = self._rec("c_1")
            del rec["author_documents"]
            with open(path, "w") as fh:
                fh.write(json.dumps(rec) + "\n")
            with self.assertRaises(ValueError):
                cg.load_cases([path])
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_citer_gold.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'research_assistant.eval'`

- [ ] **Step 3: Create the package**

`research_assistant/eval/__init__.py`:

```python
"""Evaluation of what the pipeline writes.

The judge's evaluation lives in research_assistant/judgement (evalset,
metrics, harvest, labeling): a labelled claim–evidence set. This package
holds the citer's, whose ground truth is a different kind of thing — a
published paper's own citations, which need no labelling session.
"""
```

`research_assistant/eval/cases/.gitkeep`: an empty file.

`pyproject.toml` — the package-data table becomes:

```toml
[tool.setuptools.package-data]
"research_assistant.judgement" = ["prompt.md", "cases/*.jsonl"]
"research_assistant.eval" = ["cases/*.jsonl"]
```

- [ ] **Step 4: Write `citer_gold.py`**

```python
"""The citer's ground truth, from a seed paper's own citations.

A published paper is a labelled dataset for citation: its author decided
which sentence cites which work. For every body sentence whose author-cited
reference is in the corpus, a record — the clean sentence, its three-sentence
window, and the documents the author cited that we hold. A matched sample of
the paper's uncited sentences is the negative class for citation need.

Author citations are a floor, not truth: a correct paper the author did not
cite scores as wrong, and an uncited sentence may merely be under-cited. The
metrics print both caveats.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from bs4 import BeautifulSoup

from research_assistant.shared.claim_text import paragraph_sentences, sentence_context
from research_assistant.shared.seed_audit import (
    _load_downloaded_manifest,
    _match_downloaded_paper,
    _paragraph_section,
    extract_seed_citation_claims,
    find_tei_for_seed,
)
from research_assistant.shared.tei_structure import section_breadcrumb

CASES_DIR = Path(__file__).parent / "cases"
MIN_WORDS = 8
REQUIRED = ("id", "seed", "kind", "sentence", "context", "author_documents")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _short(stem: str) -> str:
    """An id prefix: the arXiv year-month when there is one, else the stem."""
    m = re.search(r"(\d{4})\.\d{4,5}", stem)
    return m.group(1) if m else stem[:16]


def cited_records(claims: list[dict], seed_name: str, manifest: dict) -> list[dict]:
    """One record per sentence with at least one author citation whose
    reference PDF is in the corpus. References not in the corpus are dropped
    from the record: the citer cannot be expected to find them."""
    by_sent: dict[tuple, dict] = {}
    for c in claims:
        key = (c.get("paragraph_index", 0), c.get("sentence_index", 0))
        rec = by_sent.setdefault(key, {
            "sentence": c["claim"],
            "context": c.get("context") or "",
            "section": c.get("section") or "other",
            "section_heading": c.get("section_heading") or "",
            "paragraph_id": c.get("paragraph_id"),
            "sentence_index": c.get("sentence_index", 0),
            "author_documents": [], "author_refs": [], "roles": [],
        })
        if not c.get("resolved"):
            continue
        ref = c.get("ref") or {}
        dl = _match_downloaded_paper(ref, seed_name, manifest)
        if not (dl and dl.get("path") and os.path.exists(dl["path"])):
            continue
        doc = os.path.basename(dl["path"])
        if doc in rec["author_documents"]:
            continue
        rec["author_documents"].append(doc)
        rec["author_refs"].append({"index": ref.get("index"), "title": ref.get("title")})
        rec["roles"].append(c.get("role") or "evidential")
    return [rec for _, rec in sorted(by_sent.items()) if rec["author_documents"]]


def uncited_pool(tei_path: str) -> list[dict]:
    """Body sentences from paragraphs that cite nothing, MIN_WORDS words or
    more. Paragraph-level on purpose: a paragraph with no <ref type="bibr">
    has clean sentences and nothing to strip."""
    with open(tei_path, encoding="utf-8") as fh:
        soup = BeautifulSoup(fh, "xml")
    body = soup.find("body")
    pool = []
    for p_idx, p in enumerate(body.find_all("p") if body else []):
        if p.find_all("ref", type="bibr"):
            continue
        sents = paragraph_sentences(p)
        for s_idx, sent in enumerate(sents):
            if len(sent.split()) < MIN_WORDS:
                continue
            pool.append({
                "sentence": sent,
                "context": sentence_context(sents, s_idx),
                "section": _paragraph_section(p),
                "section_heading": section_breadcrumb(p),
                "paragraph_id": p.get("xml:id") or p.get("id") or f"p_{p_idx}",
                "sentence_index": s_idx,
                "author_documents": [], "author_refs": [], "roles": [],
            })
    return pool


def build(seed_pdf_path: str, *, negatives_seed: int = 7) -> list[dict]:
    """Gold records for one seed: every cited sentence with an in-corpus
    reference, then as many uncited sentences, sampled with a fixed seed so
    two builds agree."""
    tei = find_tei_for_seed(seed_pdf_path)
    if not tei:
        raise FileNotFoundError(f"no GROBID TEI for {seed_pdf_path}")
    seed_name = os.path.basename(seed_pdf_path)
    stem = _stem(seed_pdf_path)
    short = _short(stem)
    manifest = _load_downloaded_manifest()
    cited = cited_records(extract_seed_citation_claims(tei), seed_name, manifest)
    pool = uncited_pool(tei)
    random.Random(negatives_seed).shuffle(pool)
    negatives = pool[: len(cited)]
    out = []
    for n, rec in enumerate(cited, 1):
        out.append({"id": f"c_{short}_{n:04d}", "seed": stem, "kind": "cited", **rec})
    for n, rec in enumerate(negatives, 1):
        out.append({"id": f"u_{short}_{n:04d}", "seed": stem, "kind": "uncited", **rec})
    return out


def write_cases(records: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_cases(paths=None) -> list[dict]:
    """Every citer_*.jsonl in CASES_DIR unless paths are given. A record
    missing a required field, or a duplicate id, raises — a silent gap here
    is a wrong score later."""
    paths = [Path(p) for p in paths] if paths else sorted(CASES_DIR.glob("citer_*.jsonl"))
    out, seen = [], set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                missing = [k for k in REQUIRED if k not in rec]
                if missing:
                    raise ValueError(f"{path.name}: record {rec.get('id', '?')!r} missing {missing}")
                if rec["id"] in seen:
                    raise ValueError(f"duplicate case id {rec['id']!r} in {path.name}")
                seen.add(rec["id"])
                out.append(rec)
    return out
```

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_citer_gold.py -q`
Expected: all pass. If `test_a_cited_sentence_with_an_in_corpus_reference_is_a_record` fails on `section_heading`, check that `section_breadcrumb` is producing `"2. Results"` for `<head n="2.">Results</head>` — it should (Task 1 of the previous plan).

- [ ] **Step 6: Commit**

```bash
git add research_assistant/eval/__init__.py research_assistant/eval/citer_gold.py research_assistant/eval/cases/.gitkeep pyproject.toml tests/test_citer_gold.py
git commit -m "feat(eval): the citer's ground truth is the seed paper's own citations

One record per sentence whose author-cited reference is in the corpus,
and a matched sample of uncited sentences as the negative class. No
labelling session: a published paper already decided which sentence
cites which work. Author citations are a floor, not truth; the metrics
will say so.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `research_assistant/eval/citer_metrics.py` — score a citer run

**Files:**
- Create: `research_assistant/eval/citer_metrics.py`
- Test: `tests/test_citer_metrics.py` (new)

**Interfaces:**
- Consumes: harness result records (Global Constraints) whose `cite` is `dataclasses.asdict(CiteResult)` — in particular `cite["keys"]` and `cite["candidates"][i]["key"|"document"]`, and `cite["skip_reason"]`.
- Produces: `citer_metrics.cited_documents(cite: dict | None) -> set[str]`; `citer_metrics.score(results) -> dict` with keys `n_cited, n_uncited, need_recall, need_specificity, target_precision, target_recall, sentence_hit, end_to_end, declined: {n, reasons: {reason: n}}, by_role, by_section, stability, errors`; `citer_metrics.render(summary) -> str`; `citer_metrics.compare(a, b) -> str`; `citer_metrics.FLOOR_NOTE: str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_citer_metrics.py`:

```python
"""Scoring for the citer evaluation. Every number is defined in the spec's
table (§3.2); these pin each one on hand-built results."""

import unittest

from research_assistant.eval import citer_metrics as cm


def _case(id, kind="cited", author=("a.pdf",), roles=("evidential",), section="results"):
    return {"id": id, "seed": "s", "kind": kind, "sentence": "S.", "context": "«S.»",
            "section": section, "section_heading": "", "paragraph_id": "p_0", "sentence_index": 0,
            "author_documents": list(author), "author_refs": [], "roles": list(roles)}


def _cite(keys, candidates, skip_reason=None):
    return {"original": "S.", "cited_text": "S \\cite{x}.", "keys": list(keys),
            "candidates": [{"key": k, "citation": k, "document": d} for k, d in candidates],
            "reasoning": "", "skip_reason": skip_reason, "query": "S.", "verdicts": [], "partial": False}


def _r(case, needs_cite, cite=None, run=1, error=None):
    return {"case": case, "needs_cite": needs_cite, "cite": cite, "run": run, "error": error}


class TestCitedDocuments(unittest.TestCase):
    def test_only_keys_that_made_the_sentence_count(self):
        cite = _cite(["cite_2"], [("cite_1", "a.pdf"), ("cite_2", "b.pdf"), ("cite_3", None)])
        self.assertEqual(cm.cited_documents(cite), {"b.pdf"})
        self.assertEqual(cm.cited_documents(None), set())


class TestScore(unittest.TestCase):
    def _results(self):
        return [
            # cited, needed, hit: cites the author's paper
            _r(_case("c1"), True, _cite(["cite_1"], [("cite_1", "a.pdf"), ("cite_2", "z.pdf")])),
            # cited, needed, miss: cites a paper the author did not
            _r(_case("c2", author=("a.pdf", "b.pdf"), roles=("evidential", "method"), section="methods"),
               True, _cite(["cite_2"], [("cite_1", "a.pdf"), ("cite_2", "z.pdf")])),
            # cited, needed, declined
            _r(_case("c3"), True, _cite([], [("cite_1", "a.pdf")], skip_reason="context retrieved but the model did not cite it")),
            # cited, need missed
            _r(_case("c4"), False),
            # uncited: one correctly left alone, one wrongly flagged
            _r(_case("u1", kind="uncited", author=(), roles=()), False),
            _r(_case("u2", kind="uncited", author=(), roles=()), True, _cite(["cite_1"], [("cite_1", "a.pdf")])),
        ]

    def test_need_numbers(self):
        s = cm.score(self._results())
        self.assertEqual((s["n_cited"], s["n_uncited"]), (4, 2))
        self.assertAlmostEqual(s["need_recall"], 3 / 4)
        self.assertAlmostEqual(s["need_specificity"], 1 / 2)

    def test_target_numbers_are_micro_over_sentences_that_cited(self):
        s = cm.score(self._results())
        # c1 cited {a} ∩ {a} = 1 of 1; c2 cited {z} ∩ {a,b} = 0 of 1 → precision 1/2
        self.assertAlmostEqual(s["target_precision"], 1 / 2)
        # author docs over all cited cases: c1 1, c2 2, c3 1, c4 1 = 5; found 1 → recall 1/5
        self.assertAlmostEqual(s["target_recall"], 1 / 5)
        self.assertAlmostEqual(s["sentence_hit"], 1 / 2)        # of the two that cited
        self.assertAlmostEqual(s["end_to_end"], 1 / 4)          # of all cited cases

    def test_declined_is_counted_with_its_reasons(self):
        s = cm.score(self._results())
        self.assertEqual(s["declined"]["n"], 1)
        self.assertEqual(s["declined"]["reasons"], {"context retrieved but the model did not cite it": 1})

    def test_slices_by_role_and_section(self):
        s = cm.score(self._results())
        self.assertEqual(s["by_role"]["evidential"]["n"], 4)
        self.assertAlmostEqual(s["by_role"]["evidential"]["end_to_end"], 1 / 4)
        self.assertEqual(s["by_role"]["method"]["n"], 1)
        self.assertEqual(s["by_section"]["methods"]["n"], 1)

    def test_errors_are_excluded_and_counted(self):
        rs = self._results() + [_r(_case("c9"), True, error="boom")]
        s = cm.score(rs)
        self.assertEqual(s["errors"], 1)
        self.assertEqual(s["n_cited"], 4)

    def test_stability_over_runs(self):
        hit = _cite(["cite_1"], [("cite_1", "a.pdf")])
        miss = _cite(["cite_1"], [("cite_1", "z.pdf")])
        rs = [_r(_case("c1"), True, hit, run=1), _r(_case("c1"), True, hit, run=2),
              _r(_case("c2"), True, hit, run=1), _r(_case("c2"), True, miss, run=2)]
        self.assertAlmostEqual(cm.score(rs)["stability"], 1 / 2)
        self.assertIsNone(cm.score(rs[:2:2] + rs[2:3])["stability"])   # one run: undefined

    def test_render_states_the_floor_and_compare_shows_deltas(self):
        s = cm.score(self._results())
        text = cm.render(s)
        self.assertIn(cm.FLOOR_NOTE.split(":")[0], text)
        self.assertIn("end_to_end", text)
        out = cm.compare({"summary": s}, {"summary": s})
        self.assertIn("end_to_end", out)
        self.assertIn("25%", out)
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_citer_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'citer_metrics'`

- [ ] **Step 3: Implement**

Create `research_assistant/eval/citer_metrics.py`:

```python
"""Scoring for the citer evaluation. Pure; the harness collects results.

A result is {case, needs_cite, cite, run, error}; ``cite`` is the
dataclasses.asdict of a CiteResult, or None when nothing was attempted.
"""

from __future__ import annotations

from collections import Counter, defaultdict

FLOOR_NOTE = (
    "Author citations are a floor, not truth: a correct paper the author did not cite scores as "
    "wrong, and an uncited sentence may merely be under-cited. Compare runs against each other; "
    "do not read an absolute number as the citer's accuracy."
)

HEADLINE = ("need_recall", "need_specificity", "target_precision", "target_recall", "sentence_hit", "end_to_end")


def cited_documents(cite: dict | None) -> set:
    """The documents behind the keys that actually made it into the sentence."""
    if not cite:
        return set()
    keys = set(cite.get("keys") or [])
    return {c["document"] for c in (cite.get("candidates") or [])
            if c.get("key") in keys and c.get("document")}


def _rate(num, den):
    return (num / den) if den else None


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def score(results: list[dict]) -> dict:
    scored = [r for r in results if not r.get("error")]
    cited = [r for r in scored if r["case"]["kind"] == "cited"]
    uncited = [r for r in scored if r["case"]["kind"] == "uncited"]

    made = [r for r in cited if r["needs_cite"] and cited_documents(r.get("cite"))]
    hit_ids = {id(r) for r in made if cited_documents(r["cite"]) & set(r["case"]["author_documents"])}
    inter = sum(len(cited_documents(r["cite"]) & set(r["case"]["author_documents"])) for r in made)
    n_cited_docs = sum(len(cited_documents(r["cite"])) for r in made)
    n_author_docs = sum(len(set(r["case"]["author_documents"])) for r in cited)
    declined = [r for r in cited if r["needs_cite"] and not cited_documents(r.get("cite"))]

    def _group(keys):
        groups = defaultdict(list)
        for r in cited:
            for k in keys(r):
                if k:
                    groups[k].append(r)
        return {k: {"n": len(v), "end_to_end": _rate(sum(1 for r in v if id(r) in hit_ids), len(v))}
                for k, v in groups.items()}

    runs = {r["run"] for r in cited}
    stability = None
    if len(runs) > 1:
        outcome = defaultdict(dict)
        for r in cited:
            outcome[r["case"]["id"]][r["run"]] = id(r) in hit_ids
        complete = [o for o in outcome.values() if len(o) == len(runs)]
        stability = _rate(sum(1 for o in complete if len(set(o.values())) == 1), len(complete))

    return {
        "n_cited": len(cited),
        "n_uncited": len(uncited),
        "need_recall": _rate(sum(1 for r in cited if r["needs_cite"]), len(cited)),
        "need_specificity": _rate(sum(1 for r in uncited if not r["needs_cite"]), len(uncited)),
        "target_precision": _rate(inter, n_cited_docs),
        "target_recall": _rate(inter, n_author_docs),
        "sentence_hit": _rate(len(hit_ids), len(made)),
        "end_to_end": _rate(len(hit_ids), len(cited)),
        "declined": {
            "n": len(declined),
            "reasons": dict(Counter((r.get("cite") or {}).get("skip_reason") or "no attempt" for r in declined)),
        },
        "by_role": _group(lambda r: set(r["case"].get("roles") or [])),
        "by_section": _group(lambda r: [r["case"].get("section")]),
        "stability": stability,
        "errors": len(results) - len(scored),
    }


def render(summary: dict) -> str:
    lines = [
        f"_{FLOOR_NOTE}_",
        "",
        f"**Cited sentences:** n={summary['n_cited']}   **uncited (negatives):** n={summary['n_uncited']}"
        f"   errors {summary['errors']}   stability {_pct(summary['stability'])}",
        "",
        "| metric | value |", "|---|---|",
    ]
    for k in HEADLINE:
        lines.append(f"| {k} | {_pct(summary[k])} |")
    d = summary["declined"]
    lines += ["", f"**Needed a citation, none made:** {d['n']}"]
    for reason, n in sorted(d["reasons"].items(), key=lambda x: -x[1]):
        lines.append(f"- {n} × {reason}")
    lines += ["", "| role / section | n | end_to_end |", "|---|---|---|"]
    for k, v in sorted(summary["by_role"].items()):
        lines.append(f"| {k} | {v['n']} | {_pct(v['end_to_end'])} |")
    for k, v in sorted(summary["by_section"].items()):
        lines.append(f"| § {k} | {v['n']} | {_pct(v['end_to_end'])} |")
    return "\n".join(lines)


def compare(a: dict, b: dict) -> str:
    sa, sb = a["summary"], b["summary"]
    lines = [f"_{FLOOR_NOTE}_", ""]
    for k in HEADLINE:
        lines.append(f"{k:18} {_pct(sa[k]):>5} → {_pct(sb[k]):>5}")
    lines.append(f"{'declined':18} {sa['declined']['n']:>5} → {sb['declined']['n']:>5}")
    for tag in sorted(set(sa["by_role"]) | set(sb["by_role"])):
        ta, tb = sa["by_role"].get(tag), sb["by_role"].get(tag)
        lines.append(f"# {tag}: end_to_end {_pct(ta['end_to_end'] if ta else None)} → "
                     f"{_pct(tb['end_to_end'] if tb else None)}  (n {ta['n'] if ta else 0} → {tb['n'] if tb else 0})")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_citer_metrics.py -q`
Expected: all pass. If `test_stability_over_runs`'s second assertion fails, note that `rs[:2:2] + rs[2:3]` is `[c1 run1, c2 run1]` — a single run, so stability must be `None`.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/eval/citer_metrics.py tests/test_citer_metrics.py
git commit -m "feat(eval): score a citer run — need, target, end to end

Six numbers plus the declined breakdown and role/section slices, each
defined against the seed paper's own citations. The floor caveat is
printed by every render and compare.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `scripts/evaluate_citer.py` — build, score, compare

**Files:**
- Create: `scripts/evaluate_citer.py`
- Test: `tests/test_evaluate_citer.py` (new)

**Interfaces:**
- Consumes: `citer_gold.build/write_cases/load_cases/CASES_DIR/_stem`; `citer_metrics.score/render/compare/cited_documents`; `agent5_batch_citer._batch_needs_citation(sentences) -> list[bool]`, `agent5_batch_citer.cite_sentence(...)`, `CiteResult`; `shared.db.load_search_resources()` (live only).
- Produces: `evaluate_citer.seed_documents(seed_stem, metadatas) -> set[str]`; `evaluate_citer.run(cases, resources, runs=1) -> dict` with keys `generated, runs, excluded: {seed: [docs]}, summary, results`; CLI `--build <pdf>...`, `--score [--cases ...] [--runs N] [--out DIR]`, `--compare A B`. Results file `data/eval/citer/results/<stamp>-citer.json` (renamed by Tasks 7 and 10).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluate_citer.py`:

```python
"""The citer harness with a stubbed citer: the seed is excluded from its own
search, the need check is batched per seed, results carry what the metrics
need, and one sentence's failure is recorded rather than fatal."""

import unittest
from unittest.mock import patch

from research_assistant.agents.agent5_batch_citer import CiteResult
from scripts import evaluate_citer as ec


def _case(id, seed="arxiv_2108.10114v3", kind="cited", sentence="A claim with enough words to be eligible.", author=("a.pdf",)):
    return {"id": id, "seed": seed, "kind": kind, "sentence": sentence, "context": f"«{sentence}»",
            "section": "results", "section_heading": "", "paragraph_id": "p_0", "sentence_index": 0,
            "author_documents": list(author), "author_refs": [], "roles": ["evidential"]}


class TestSeedDocuments(unittest.TestCase):
    METAS = [{"document": "doi_10.1038_nature12952.pdf"}, {"document": "arxiv_2007.12504.pdf"},
             {"document": "doi_10.1103_physrevb.102.075409.pdf"}, {}]

    def test_names_that_differ_by_source_still_match(self):
        self.assertEqual(ec.seed_documents("nature12952", self.METAS), {"doi_10.1038_nature12952.pdf"})
        self.assertEqual(ec.seed_documents("arxiv_2007.12504v1", self.METAS), {"arxiv_2007.12504.pdf"})

    def test_a_seed_not_in_the_index_excludes_nothing(self):
        self.assertEqual(ec.seed_documents("arxiv_2108.10114v3", self.METAS), set())

    def test_a_short_document_name_is_not_swallowed_by_the_seed_stem(self):
        # "a" is a substring of "arxiv_2108…"; the match runs one way only.
        self.assertEqual(ec.seed_documents("arxiv_2108.10114v3", self.METAS + [{"document": "a.pdf"}]), set())


class TestRun(unittest.TestCase):
    RESOURCES = (None, None, [], [{"document": "arxiv_2108.10114v3.pdf"}, {"document": "a.pdf"}])

    def test_seed_is_excluded_need_is_batched_and_results_are_scored(self):
        cases = [_case("c_1"), _case("c_2", sentence="Short one."), _case("u_1", kind="uncited", author=())]
        seen = {}

        def fake_cite(sentence, resources, key_registry, **kw):
            seen[sentence] = kw
            key_registry.setdefault("Doe 2020", "cite_1")
            return CiteResult(original=sentence, cited_text=f"{sentence[:-1]} \\cite{{cite_1}}.", keys=["cite_1"],
                              candidates=[{"key": "cite_1", "citation": "Doe 2020", "document": "a.pdf"}], query=sentence)

        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True, False]) as need, \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite):
            out = ec.run(cases, self.RESOURCES, runs=1)

        # "Short one." has 2 words and never reaches the need check; the other two are batched once
        need.assert_called_once()
        self.assertEqual(need.call_args[0][0], [cases[0]["sentence"], cases[2]["sentence"]])
        self.assertEqual(out["excluded"], {"arxiv_2108.10114v3": ["arxiv_2108.10114v3.pdf"]})
        self.assertEqual(seen[cases[0]["sentence"]]["exclude_docs"], {"arxiv_2108.10114v3.pdf"})
        self.assertEqual(seen[cases[0]["sentence"]]["context"], cases[0]["context"])
        by_id = {r["case"]["id"]: r for r in out["results"]}
        self.assertEqual(by_id["c_1"]["needs_cite"], True)
        self.assertEqual(by_id["c_1"]["cite"]["keys"], ["cite_1"])
        self.assertEqual(by_id["c_2"]["needs_cite"], False)
        self.assertIsNone(by_id["c_2"]["cite"])
        self.assertEqual(by_id["u_1"]["needs_cite"], False)
        self.assertAlmostEqual(out["summary"]["end_to_end"], 1 / 2)
        self.assertEqual(out["runs"], 1)

    def test_a_failing_sentence_is_recorded_not_fatal(self):
        cases = [_case("c_1"), _case("c_2")]
        calls = iter([RuntimeError("model down"),
                      CiteResult(original="x", cited_text="x", keys=[], candidates=[], query="x",
                                 skip_reason="no relevant context found in database")])

        def fake_cite(sentence, resources, key_registry, **kw):
            v = next(calls)
            if isinstance(v, Exception):
                raise v
            return v

        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True, True]), \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite):
            out = ec.run(cases, self.RESOURCES, runs=1)
        by_id = {r["case"]["id"]: r for r in out["results"]}
        self.assertIn("model down", by_id["c_1"]["error"])
        self.assertIsNone(by_id["c_2"]["error"])
        self.assertEqual(out["summary"]["errors"], 1)

    def test_a_failed_need_check_marks_every_case_of_that_seed(self):
        with patch.object(ec.citer, "_batch_needs_citation", side_effect=ValueError("misaligned")), \
             patch.object(ec.citer, "cite_sentence") as cite:
            out = ec.run([_case("c_1"), _case("c_2")], self.RESOURCES, runs=1)
        cite.assert_not_called()
        self.assertTrue(all("misaligned" in r["error"] for r in out["results"]))
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_evaluate_citer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.evaluate_citer'`

- [ ] **Step 3: Write the harness**

Create `scripts/evaluate_citer.py`:

```python
"""Evaluate Agent 5's citer against the seed papers' own citations.

    PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/evaluate_citer.py --build data/raw/processed/arxiv_2108.10114v3.pdf
    PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/evaluate_citer.py --score --runs 2
    PYTHONPATH=. python scripts/evaluate_citer.py --compare data/eval/citer/results/A.json data/eval/citer/results/B.json

--build writes research_assistant/eval/cases/citer_<stem>.jsonl from a seed's
TEI and the download manifest. --score runs the real citer on every case —
the batched citation-need check, then cite_sentence with the seed paper
excluded from retrieval — and writes data/eval/citer/results/<stamp>-citer.json.
Author citations are a floor; every report says so.
"""

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from datetime import datetime

from research_assistant.agents import agent5_batch_citer as citer
from research_assistant.config import DATA_DIR
from research_assistant.eval.citer_gold import CASES_DIR, _stem, build, load_cases, write_cases
from research_assistant.eval.citer_metrics import cited_documents, compare, render, score
from research_assistant.shared.atomic import atomic_write_json

RESULTS_DIR = os.path.join(DATA_DIR, "eval", "citer", "results")


def _norm(name: str) -> str:
    return re.sub(r"v\d+$", "", os.path.splitext(os.path.basename(name))[0].lower())


def seed_documents(seed_stem: str, metadatas) -> set:
    """Index documents that are the seed paper itself. Names differ by
    source — 'nature12952.pdf' was ingested as 'doi_10.1038_nature12952.pdf',
    'arxiv_2007.12504v1.pdf' as 'arxiv_2007.12504.pdf' — so a document
    matches when its normalised stem equals the seed's or contains it. Only
    that direction: the seed's stem is the long, specific one, and a short
    document name ('a.pdf') is a substring of almost anything."""
    stem = _norm(seed_stem)
    out = set()
    for m in metadatas:
        doc = (m or {}).get("document")
        if not doc:
            continue
        d = _norm(doc)
        if d == stem or (len(stem) >= 6 and stem in d):
            out.add(doc)
    return out


def run(cases, resources, runs=1):
    """The citer on every case: need check batched per seed file, as
    run_batch_citer batches a draft, then cite_sentence with the seed
    excluded. One sentence's failure is recorded, never fatal."""
    _, _, _, metadatas = resources
    by_seed = {}
    for c in cases:
        by_seed.setdefault(c["seed"], []).append(c)
    excluded = {seed: sorted(seed_documents(seed, metadatas)) for seed in by_seed}

    results = []
    for run_no in range(1, runs + 1):
        for seed, seed_cases in by_seed.items():
            eligible = [c for c in seed_cases if len(c["sentence"].split()) >= 4]
            need_by_id, need_error = {}, None
            if eligible:
                try:
                    verdicts = citer._batch_needs_citation([c["sentence"] for c in eligible])
                    need_by_id = {c["id"]: v for c, v in zip(eligible, verdicts)}
                except Exception as exc:  # noqa: BLE001 — recorded on every case of the seed
                    need_error = f"{type(exc).__name__}: {exc}"

            key_registry = {}
            for c in seed_cases:
                t0 = time.perf_counter()
                rec = {"case": c, "needs_cite": bool(need_by_id.get(c["id"], False)),
                       "cite": None, "run": run_no, "error": need_error}
                if rec["needs_cite"] and not need_error:
                    try:
                        res = citer.cite_sentence(
                            c["sentence"], resources, key_registry,
                            context=c.get("context"), exclude_docs=set(excluded[seed]),
                            paragraph_id=c.get("paragraph_id"),
                        )
                        rec["cite"] = asdict(res)
                    except Exception as exc:  # noqa: BLE001 — recorded, never fatal
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["seconds"] = round(time.perf_counter() - t0, 2)
                results.append(rec)
                got = sorted(cited_documents(rec["cite"]))
                print(f"run {run_no} {c['id']:16s} need={str(rec['needs_cite']):5s} cited={got} "
                      f"author={c['author_documents']}{'  ERROR ' + rec['error'] if rec['error'] else ''}")

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
        "excluded": excluded,
        "summary": score(results),
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="*", metavar="SEED_PDF", help="write gold for these seed PDFs")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0]) as fa, open(args.compare[1]) as fb:
            print(compare(json.load(fa), json.load(fb)))
        return

    if args.build:
        for pdf in args.build:
            records = build(pdf)
            path = CASES_DIR / f"citer_{_stem(pdf)}.jsonl"
            write_cases(records, path)
            n_c = sum(1 for r in records if r["kind"] == "cited")
            print(f"{path}: {n_c} cited + {len(records) - n_c} uncited")
        return

    if args.score:
        from research_assistant.shared.db import load_search_resources

        cases = load_cases(args.cases)
        out = run(cases, load_search_resources(), runs=args.runs)
        print("\n" + render(out["summary"]))
        for seed, docs in out["excluded"].items():
            print(f"(excluded from search for {seed}: {docs or 'nothing — seed not in the index'})")
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-citer.json")
        atomic_write_json(path, out)
        print(f"\nwritten {path}")
        return

    ap.error("one of --build, --score, --compare is required")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests, then the suite**

Run: `python3 -m pytest tests/test_evaluate_citer.py -q && python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_citer.py tests/test_evaluate_citer.py
git commit -m "feat(eval): evaluate_citer — build gold, score the citer, compare runs

Mirrors evaluate_judge. The seed paper is excluded from its own search
by stem, the need check is batched per seed file as a draft is batched,
and a sentence's failure is recorded rather than fatal.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Build the gold and take the baseline

**Files:**
- Create (tracked): `research_assistant/eval/cases/citer_arxiv_2108.10114v3.jsonl`, `citer_arxiv_2007.12504v1.jsonl`, `citer_nature12952.jsonl`
- Create (git-ignored): `data/eval/citer/results/<stamp>-citer.json` (the baseline; keep its name)

**Interfaces:**
- Consumes: Task 5's CLI.
- Produces: the baseline numbers for Tasks 9 and 11 to compare against.

- [ ] **Step 1: Build the three gold files**

```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python scripts/evaluate_citer.py --build \
  data/raw/processed/arxiv_2108.10114v3.pdf data/raw/processed/arxiv_2007.12504v1.pdf data/raw/processed/nature12952.pdf
```
Expected: three lines like `…/citer_arxiv_2108.10114v3.jsonl: 22 cited + 22 uncited`. If a seed reports 0 cited, its TEI or manifest is not where `find_tei_for_seed` / `_load_downloaded_manifest` look — check `data/raw/grobid_output/<stem>*.tei.xml` exists and stop rather than committing an empty file.

- [ ] **Step 2: Read a few records**

```bash
python3 - <<'EOF'
import json, glob
for p in sorted(glob.glob("research_assistant/eval/cases/citer_*.jsonl")):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    c = [r for r in rows if r["kind"] == "cited"]
    print(f"{p.split('/')[-1]}: {len(c)} cited, {len(rows)-len(c)} uncited; "
          f"{sum(len(r['author_documents']) for r in c)} author docs; roles {sorted({x for r in c for x in r['roles']})}")
    print("   e.g.", c[0]["sentence"][:90], "→", c[0]["author_documents"])
EOF
```
Expected: sentences read as clean prose with no `[n]` markers; `author_documents` are basenames present in `data/pulled_pdfs/`.

- [ ] **Step 3: Baseline, two runs**

```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python scripts/evaluate_citer.py --score --runs 2
```
Expected: the rendered table, the `excluded` lines (nature12952 and arxiv_2007.12504v1 each exclude one document; arxiv_2108.10114v3 excludes nothing), and `written data/eval/citer/results/<stamp>-citer.json`. Roughly 130 model calls.

- [ ] **Step 4: Commit the gold and record the baseline**

```bash
git add research_assistant/eval/cases/
git commit -m "eval(citer): gold from three seeds — <N> cited sentences, <N> uncited

Built from the seeds' own citations; nothing labelled by hand.

Baseline, judge off, raw query, 2 runs:
<paste the rendered table>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 2 — A: cite only what the judge would pass

### Task 7: The judge as acceptance test inside `cite_sentence`

**Files:**
- Modify: `research_assistant/config.py` (after `CITATION_AUDIT_MAX_CLAIMS`)
- Modify: `research_assistant/agents/agent5_batch_citer.py` (imports; `cite_sentence` signature and tail; three new functions)
- Modify: `scripts/evaluate_citer.py` (`run` signature, the `cite_sentence` call, `main`)
- Test: `tests/test_batch_citer.py`, `tests/test_evaluate_citer.py`

**Interfaces:**
- Consumes: `judgement.judge.judge(claim, evidence, context=) -> dict` (keys `judgement, model_judgement, confidence, evidence_sufficiency, supporting_span, span_verified, reason, rubric_violations, slots`); `judgement.judge.compose_context(window, section=None, artifacts=None)`; `agent8_verifier.assemble_evidence(hits, max_chars)`; `shared.search.expand_neighbours(results, texts, metadatas, window=)`; config `JUDGEMENT_EVIDENCE_MAX_CHARS`, `JUDGEMENT_NEIGHBOUR_WINDOW`.
- Produces: `config.CITATION_CITER_JUDGE: bool` (env `CITATION_CITER_JUDGE`, default False); `cite_sentence(..., judge_gate: bool | None = None)` — `None` reads the config; `agent5_batch_citer._insert_cite(sentence, key) -> str`; `agent5_batch_citer._cite_by_judge(...)`; `CiteResult.verdicts` populated (one record per judged candidate: `key, document, evidence, judgement, model_judgement, confidence, evidence_sufficiency, supporting_span, span_verified, reason, rubric_violations, slots`, plus `error` when the judge call failed, plus `best: True` on the closest candidate when nothing was accepted); `CiteResult.partial: bool`; skip reasons `"no candidate passed the judge"` and `"judge failed on every candidate"`. `evaluate_citer.run(cases, resources, runs=1, judge_gate=None)`; CLI `--judge {on,off,config}`; results file `<stamp>-judge<on|off>.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_batch_citer.py` (extend the import to include `_insert_cite`):

```python
def _verdict(judgement, span_verified=True, reason="because", span="the evidence sentence"):
    return {"judgement": judgement, "model_judgement": judgement, "confidence": "High",
            "evidence_sufficiency": "sufficient", "supporting_span": span, "span_verified": span_verified,
            "reason": reason, "rubric_violations": [],
            "slots": {"finding": {"assertion": "f", "verdict": "Supports"},
                      "scope": {"assertion": "s", "verdict": "Supports"},
                      "strength": {"assertion": "t", "verdict": "Not applicable"}}}


class TestInsertCite(unittest.TestCase):
    def test_before_terminal_punctuation(self):
        self.assertEqual(_insert_cite("Graphene is ballistic.", "cite_1"), "Graphene is ballistic \\cite{cite_1}.")
        self.assertEqual(_insert_cite("Is it ballistic?", "cite_2"), "Is it ballistic \\cite{cite_2}?")

    def test_appended_when_there_is_none(self):
        self.assertEqual(_insert_cite("Graphene is ballistic", "cite_1"), "Graphene is ballistic \\cite{cite_1}")

    def test_trailing_whitespace_is_dropped(self):
        self.assertEqual(_insert_cite("Ballistic.  ", "cite_1"), "Ballistic \\cite{cite_1}.")


class TestCiteByJudge(unittest.TestCase):
    """The citer and the auditor apply the same test. A candidate is cited
    only if the judge would pass it — Supports with a verbatim span — and the
    key is placed by code, so nothing has to be parsed out of a rewrite."""

    RES = (None, None, [], [])
    HITS = [
        {"text": "Off-topic passage.", "chunk_index": 1, "rrf_score": 0.03,
         "metadata": {"citation_source": "Roe 2019", "document": "roe.pdf"}},
        {"text": "Ballistic transport observed in graphene.", "chunk_index": 2, "rrf_score": 0.02,
         "metadata": {"citation_source": "Doe 2020", "document": "doe.pdf"}},
        {"text": "Another passage.", "chunk_index": 3, "rrf_score": 0.01,
         "metadata": {"citation_source": "Poe 2021", "document": "poe.pdf"}},
    ]

    def _cite(self, verdicts, **kw):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(h) for h in self.HITS]), \
             patch("research_assistant.agents.agent5_batch_citer.judge", side_effect=verdicts) as judge, \
             patch("research_assistant.agents.agent5_batch_citer.chat") as chat:
            res = cite_sentence("Graphene is ballistic.", self.RES, {}, judge_gate=True, **kw)
        return res, judge, chat

    def test_first_supports_with_a_verified_span_is_cited_and_the_rewrite_is_never_asked_for(self):
        res, judge, chat = self._cite([_verdict("Unclear / insufficient evidence"), _verdict("Supports")])
        self.assertEqual(res.keys, ["cite_2"])
        self.assertEqual(res.cited_text, "Graphene is ballistic \\cite{cite_2}.")
        self.assertEqual(res.reasoning, "because")
        self.assertFalse(res.partial)
        self.assertEqual([v["judgement"] for v in res.verdicts], ["Unclear / insufficient evidence", "Supports"])
        self.assertEqual(res.verdicts[1]["document"], "doe.pdf")
        self.assertEqual(judge.call_count, 2)          # stopped at the first pass
        chat.assert_not_called()

    def test_supports_without_a_verified_span_is_not_accepted(self):
        res, judge, _ = self._cite([_verdict("Supports", span_verified=False),
                                    _verdict("Does not support"), _verdict("Unclear / insufficient evidence")])
        self.assertFalse(res.cited)
        self.assertEqual(res.cited_text, "Graphene is ballistic.")
        self.assertEqual(res.skip_reason, "no candidate passed the judge")
        self.assertEqual(judge.call_count, 3)
        # The closest candidate is the unverified Supports, first by verdict rank; it is marked and its reason kept.
        self.assertTrue(res.verdicts[0]["best"])
        self.assertEqual(res.reasoning, "because")

    def test_partially_supports_is_accepted_only_when_nothing_supports_and_is_flagged(self):
        res, _, _ = self._cite([_verdict("Partially supports"), _verdict("Does not support"), _verdict("Unclear / insufficient evidence")])
        self.assertEqual(res.keys, ["cite_1"])
        self.assertTrue(res.partial)
        res, _, _ = self._cite([_verdict("Partially supports"), _verdict("Supports")])
        self.assertEqual(res.keys, ["cite_2"])
        self.assertFalse(res.partial)

    def test_a_failing_judge_call_moves_to_the_next_candidate(self):
        res, _, _ = self._cite([RuntimeError("model down"), _verdict("Supports")])
        self.assertEqual(res.keys, ["cite_2"])
        self.assertIn("model down", res.verdicts[0]["error"])
        self.assertNotIn("judgement", res.verdicts[0])

    def test_every_call_failing_is_its_own_reason(self):
        res, _, _ = self._cite([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
        self.assertFalse(res.cited)
        self.assertEqual(res.skip_reason, "judge failed on every candidate")

    def test_the_paragraph_window_reaches_the_judge_as_context(self):
        _, judge, _ = self._cite([_verdict("Supports")], context="Before. «Graphene is ballistic.» After.")
        self.assertEqual(judge.call_args.kwargs["context"], "Before. «Graphene is ballistic.» After.")

    def test_gate_off_is_the_legacy_path(self):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(self.HITS[1])]), \
             patch("research_assistant.agents.agent5_batch_citer.judge") as judge, \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: Graphene is ballistic \\cite{cite_1}.\nREASON: r")):
            res = cite_sentence("Graphene is ballistic.", self.RES, {}, judge_gate=False)
        judge.assert_not_called()
        self.assertEqual(res.keys, ["cite_1"])
        self.assertEqual(res.verdicts, [])

    def test_gate_none_reads_the_config(self):
        with patch("research_assistant.agents.agent5_batch_citer.CITATION_CITER_JUDGE", True), \
             patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(self.HITS[1])]), \
             patch("research_assistant.agents.agent5_batch_citer.judge", return_value=_verdict("Supports")), \
             patch("research_assistant.agents.agent5_batch_citer.chat") as chat:
            res = cite_sentence("Graphene is ballistic.", self.RES, {})
        self.assertEqual(res.keys, ["cite_1"])
        chat.assert_not_called()
```

And in `tests/test_evaluate_citer.py`, add to `TestRun`:

```python
    def test_judge_gate_is_forwarded_and_named_in_the_result(self):
        seen = {}

        def fake_cite(sentence, resources, key_registry, **kw):
            seen.update(kw)
            return CiteResult(original=sentence, cited_text=sentence, query=sentence, skip_reason="no candidate passed the judge")

        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True]), \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite):
            out = ec.run([_case("c_1")], self.RESOURCES, runs=1, judge_gate=True)
        self.assertIs(seen["judge_gate"], True)
        self.assertIs(out["judge_gate"], True)
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_batch_citer.py::TestInsertCite tests/test_batch_citer.py::TestCiteByJudge tests/test_evaluate_citer.py -q`
Expected: FAIL — `ImportError: cannot import name '_insert_cite'`, and `TypeError: … unexpected keyword argument 'judge_gate'`.

- [ ] **Step 3: The config flag**

In `research_assistant/config.py`, directly after the `CITATION_AUDIT_MAX_CLAIMS` line:

```python
# The citer accepts a candidate source only if the auditor's judge would
# accept the citation: Supports (or Partially supports, flagged) with a
# verbatim supporting span. Off until its gate on the citer evaluation
# passes — scripts/evaluate_citer.py, judge off against on.
CITATION_CITER_JUDGE = _env_bool("CITATION_CITER_JUDGE", False)
```

- [ ] **Step 4: The gate in `cite_sentence`**

In `research_assistant/agents/agent5_batch_citer.py`:

(a) Imports. Replace `from research_assistant.shared.search import hybrid_search` with:

```python
from research_assistant.config import (
    CITATION_CITER_JUDGE,
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_NEIGHBOUR_WINDOW,
)
from research_assistant.judgement.judge import compose_context, judge
from research_assistant.shared.search import expand_neighbours, hybrid_search
```

(`judgement.judge` imports no agent; `agent8_verifier` imports this module, so it is imported inside the function below, never at module level.)

(b) `cite_sentence`'s signature gains `judge_gate=None` after `paragraph_id=None`, and its docstring a line: `judge_gate: True/False forces the judge acceptance test on or off; None reads CITATION_CITER_JUDGE.`

(c) In `cite_sentence`, replace the tail from `cited_sentence, reasoning = _cite_sentence_with_reasoning(sentence, context_str)` to the end of the function with:

```python
    if CITATION_CITER_JUDGE if judge_gate is None else judge_gate:
        return _cite_by_judge(sentence, results, candidates, context=context, query=query,
                              texts=texts, metadatas=metadatas)

    cited_sentence, reasoning = _cite_sentence_with_reasoning(sentence, context_str)
    keys = sorted(_cite_keys(cited_sentence))
    if keys:
        return CiteResult(original=sentence, cited_text=cited_sentence, keys=keys,
                          candidates=candidates, reasoning=reasoning, query=query)
    # A successful call is not a citation: when the model declines, or the
    # reply could not be parsed, the sentence comes back unchanged.
    return CiteResult(original=sentence, cited_text=sentence, candidates=candidates,
                      reasoning=reasoning, query=query,
                      skip_reason="context retrieved but the model did not cite it")
```

(d) Directly after `cite_sentence`, add:

```python
_VERDICT_RANK = {"Supports": 0, "Partially supports": 1, "Contradicts": 2,
                 "Does not support": 3, "Unclear / insufficient evidence": 4}
_VERDICT_FIELDS = ("judgement", "model_judgement", "confidence", "evidence_sufficiency",
                   "supporting_span", "span_verified", "reason", "rubric_violations", "slots")


def _insert_cite(sentence: str, key: str) -> str:
    """\\cite{key} before the terminal punctuation — where the legacy rewrite
    already ends up after _restore_terminal_punctuation — or appended."""
    s = sentence.rstrip()
    if s and s[-1] in _TERMINAL:
        return f"{s[:-1].rstrip()} \\cite{{{key}}}{s[-1]}"
    return f"{s} \\cite{{{key}}}"


def _accepted(sentence, cand, record, candidates, verdicts, query, *, partial) -> CiteResult:
    return CiteResult(original=sentence, cited_text=_insert_cite(sentence, cand["key"]),
                      keys=[cand["key"]], candidates=candidates, reasoning=record.get("reason") or "",
                      query=query, verdicts=verdicts, partial=partial)


def _cite_by_judge(sentence, hits, candidates, *, context, query, texts, metadatas) -> CiteResult:
    """Judge each candidate in retrieval order with the auditor's own judge
    and cite the first that passes: Supports with a verbatim span, else the
    first Partially supports with one, flagged. The key is placed by code —
    the judge chose, nothing has to be parsed out of a rewrite. Nothing
    passing means no citation and a record of what came closest, so the
    report can show the same account the audit would.
    """
    from research_assistant.agents.agent8_verifier import assemble_evidence  # agent8 imports this module

    expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
    ctx = compose_context(context) if context else None
    verdicts, partial = [], None
    for hit, cand in zip(hits, candidates):
        evidence = assemble_evidence([hit], JUDGEMENT_EVIDENCE_MAX_CHARS)
        record = {"key": cand["key"], "document": cand.get("document"), "evidence": evidence}
        try:
            v = judge(sentence, evidence, context=ctx)
        except Exception as exc:  # noqa: BLE001 — one candidate's failure is not the sentence's
            record["error"] = f"{type(exc).__name__}: {exc}"
            verdicts.append(record)
            continue
        record.update({k: v.get(k) for k in _VERDICT_FIELDS})
        verdicts.append(record)
        if v.get("judgement") == "Supports" and v.get("span_verified") is True:
            return _accepted(sentence, cand, record, candidates, verdicts, query, partial=False)
        if partial is None and v.get("judgement") == "Partially supports" and v.get("span_verified") is True:
            partial = (cand, record)
    if partial is not None:
        return _accepted(sentence, partial[0], partial[1], candidates, verdicts, query, partial=True)

    judged = [r for r in verdicts if "judgement" in r]
    if not judged:
        return CiteResult(original=sentence, cited_text=sentence, candidates=candidates,
                          verdicts=verdicts, query=query, skip_reason="judge failed on every candidate")
    best = min(judged, key=lambda r: _VERDICT_RANK.get(r["judgement"], 9))   # ties keep retrieval order
    best["best"] = True
    return CiteResult(original=sentence, cited_text=sentence, candidates=candidates, verdicts=verdicts,
                      query=query, reasoning=best.get("reason") or "",
                      skip_reason="no candidate passed the judge")
```

- [ ] **Step 5: The harness forwards the gate**

In `scripts/evaluate_citer.py`:

(a) `def run(cases, resources, runs=1):` becomes `def run(cases, resources, runs=1, *, judge_gate=None):`, the `cite_sentence(` call gains `judge_gate=judge_gate,` after `paragraph_id=c.get("paragraph_id"),`, and the returned dict gains `"judge_gate": judge_gate,` after `"runs": runs,`.

(b) In `main()`: add `ap.add_argument("--judge", choices=["on", "off", "config"], default="config", help="force the judge acceptance test on or off; config reads CITATION_CITER_JUDGE")`. In the `--score` branch:

```python
        judge_gate = {"on": True, "off": False}.get(args.judge)
        if judge_gate is None:
            from research_assistant.config import CITATION_CITER_JUDGE
            judge_gate = CITATION_CITER_JUDGE
        out = run(cases, load_search_resources(), runs=args.runs, judge_gate=judge_gate)
```

and the filename becomes `f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-judge{'on' if judge_gate else 'off'}.json"`.

- [ ] **Step 6: Run the two files, then the suite**

Run: `python3 -m pytest tests/test_batch_citer.py tests/test_evaluate_citer.py -q && python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add research_assistant/config.py research_assistant/agents/agent5_batch_citer.py scripts/evaluate_citer.py tests/test_batch_citer.py tests/test_evaluate_citer.py
git commit -m "feat(citer): cite only what the judge would pass

Behind CITATION_CITER_JUDGE (off until gated): each retrieved candidate
goes through the auditor's judge in rank order and the first Supports
with a verbatim span is cited — Partially supports accepted and flagged
when nothing supports. The key is placed by code before the terminal
punctuation; the model's rewrite is not asked for. Nothing passing means
no citation and a record of the closest candidate.

The citer and the auditor now apply the same test, so Marvin stops
flagging his own citations.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: The citation report shows the judge's account

**Files:**
- Modify: `research_assistant/agents/agent5_batch_citer.py` (the loop in `run_batch_citer`; `_generate_report`'s cited and declined branches)
- Test: `tests/test_batch_citer.py`

**Interfaces:**
- Consumes: `CiteResult.verdicts`, `CiteResult.partial` from Task 7; `seed_audit.explain_verdict_steps(item) -> list[tuple[str, str]]` (needs an item with `outcome: "judged"`, `judgement`, `slots`, `reason`, `supporting_span`, `span_verified`, `rubric_violations`; imported inside `_generate_report` because `seed_audit` imports this module).
- Produces: report entries carry `verdicts` and `partial`; the Markdown gains **Verified span**, **Why this citation**, **⚠ Partial support**, and **Closest candidate** blocks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_batch_citer.py`:

```python
class TestReportShowsTheJudgesAccount(unittest.TestCase):
    """With the judge on, the citation report is the audit report's twin:
    a cited sentence shows its verified span and how the verdict was
    reached; a declined one shows the closest candidate and why it failed."""

    # Both sentences have ≥ 4 words: shorter ones are skipped as "too short"
    # before the need check and never reach the citer at all.
    DRAFT = "Graphene shows ballistic transport. Nothing in the corpus supports this."
    HIT = {"text": "Ballistic transport observed in graphene.", "chunk_index": 2, "rrf_score": 0.02,
           "metadata": {"citation_source": "Doe 2020", "document": "doe.pdf"}}

    def test_cited_and_declined_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            draft_path = os.path.join(tmpdir, "draft.txt")
            with open(draft_path, "w") as f:
                f.write(self.DRAFT)
            out_path = os.path.join(tmpdir, "cited.txt")
            with patch("research_assistant.agents.agent5_batch_citer.CITATION_CITER_JUDGE", True), \
                 patch("research_assistant.agents.agent5_batch_citer.load_search_resources", return_value=(None, None, [], [])), \
                 patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(self.HIT)]), \
                 patch("research_assistant.agents.agent5_batch_citer.chat", return_value=_reply("1. YES\n2. YES")), \
                 patch("research_assistant.agents.agent5_batch_citer.judge",
                       side_effect=[_verdict("Supports", span="Ballistic transport observed in graphene."),
                                    _verdict("Does not support", span_verified=False, reason="Not about this.")]):
                run_batch_citer(draft_path, out_path)
            with open(out_path) as f:
                self.assertEqual(f.read(), "Graphene shows ballistic transport \\cite{cite_1}. Nothing in the corpus supports this.")
            with open(out_path.replace(".txt", "_report.md")) as f:
                report = f.read()
        self.assertIn("**Verified span:**", report)
        self.assertIn("Ballistic transport observed in graphene.", report)
        self.assertIn("**Why this citation**", report)
        self.assertIn("Attributed to the cited paper", report)
        self.assertIn("**Closest candidate:** `cite_1`", report)
        self.assertIn("Does not support", report)
        self.assertIn("Not about this.", report)
        self.assertNotIn("Partial support", report)

    def test_partial_support_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            draft_path = os.path.join(tmpdir, "draft.txt")
            with open(draft_path, "w") as f:
                f.write("Graphene shows ballistic transport.")
            out_path = os.path.join(tmpdir, "cited.txt")
            with patch("research_assistant.agents.agent5_batch_citer.CITATION_CITER_JUDGE", True), \
                 patch("research_assistant.agents.agent5_batch_citer.load_search_resources", return_value=(None, None, [], [])), \
                 patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(self.HIT)]), \
                 patch("research_assistant.agents.agent5_batch_citer.chat", return_value=_reply("1. YES")), \
                 patch("research_assistant.agents.agent5_batch_citer.judge", return_value=_verdict("Partially supports")):
                run_batch_citer(draft_path, out_path)
            with open(out_path.replace(".txt", "_report.md")) as f:
                report = f.read()
        self.assertIn("⚠ **Partial support**", report)
```

- [ ] **Step 2: Run to see it fail**

Run: `python3 -m pytest tests/test_batch_citer.py::TestReportShowsTheJudgesAccount -q`
Expected: FAIL on `**Verified span:**`.

- [ ] **Step 3: Carry the verdicts into the entries and render them**

(a) In `run_batch_citer`'s loop, after `cited_sentences.append(res.cited_text)` and before `if res.cited:`, add:

```python
        if res.verdicts:
            entry["verdicts"] = res.verdicts
            entry["partial"] = res.partial
```

(b) In `_generate_report`, the cited branch — after the `if entry.get("reasoning"):` block and before the `if entry.get("sources"):` block — add:

```python
            chosen = next((v for v in entry.get("verdicts") or [] if v.get("key") in _cite_keys(entry["cited_text"])), None)
            if chosen:
                if entry.get("partial"):
                    lines.append("⚠ **Partial support** — the judge found the source carries a weaker version of this sentence.\n")
                if chosen.get("supporting_span"):
                    lines.append(f"**Verified span:** \u201c{chosen['supporting_span']}\u201d\n")
                lines.append("**Why this citation**")
                lines.extend(f"- *{label}:* {text}" for label, text in _verdict_steps(chosen, entry["original"]))
                lines.append("")
```

and the declined branch — after the `if entry.get("reasoning"):` block — add:

```python
            best = next((v for v in entry.get("verdicts") or [] if v.get("best")), None)
            if best:
                lines.append(f"**Closest candidate:** `{best['key']}` — judged *{best.get('judgement')}*")
                lines.extend(f"- *{label}:* {text}" for label, text in _verdict_steps(best, entry["original"]))
                lines.append("")
```

(c) Directly before `def _generate_report(`, add:

```python
def _verdict_steps(record: dict, sentence: str) -> list:
    """The audit's 'Why this verdict' lines for one of the citer's judged
    candidates. seed_audit imports this module, so it is imported here."""
    from research_assistant.shared.seed_audit import explain_verdict_steps

    item = {"outcome": "judged", "claim": sentence, "evidence_hits": 1}
    item.update({k: v for k, v in record.items() if k not in ("key", "document", "evidence", "best", "error")})
    return explain_verdict_steps(item)
```

- [ ] **Step 4: Run the file, then the suite**

Run: `python3 -m pytest tests/test_batch_citer.py -q && python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/agents/agent5_batch_citer.py tests/test_batch_citer.py
git commit -m "feat(citer): the citation report shows the judge's account

A cited sentence carries its verified span and the same 'why' lines the
audit prints; a declined one names the closest candidate and why it
failed; partial support is flagged. Same words in both reports, because
it was the same test.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Gate A — judge off against on

**Files:** none tracked unless the gate passes; then `research_assistant/config.py` (the default).

**Interfaces:**
- Consumes: the gold from Task 6, the harness with `--judge` from Task 7.

- [ ] **Step 1: Score both arms, two runs each**

```bash
export CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=.
PY=/home/shardul/miniconda3/envs/ml/bin/python
$PY scripts/evaluate_citer.py --score --runs 2 --judge off
$PY scripts/evaluate_citer.py --score --runs 2 --judge on
ls -t data/eval/citer/results/ | head -2
```
Expected: two result files, `…-judgeoff.json` and `…-judgeon.json`. The on-arm makes up to 3 judge calls per sentence that needs a citation instead of one rewrite call.

- [ ] **Step 2: Compare**

```bash
$PY scripts/evaluate_citer.py --compare data/eval/citer/results/<judgeoff>.json data/eval/citer/results/<judgeon>.json
```

Acceptance (spec §3.3): `target_precision` rises, and `end_to_end` does not fall by more than the noise floor — the noise floor being the number of cited cases whose outcome flips between the two runs *within* one arm, read off `stability`. Expected shape: precision up, `declined` up, `end_to_end` flat or up.

- [ ] **Step 3: Ship or record**

If it passes, flip the default in `research_assistant/config.py` — `CITATION_CITER_JUDGE = _env_bool("CITATION_CITER_JUDGE", True)` — update its comment's last sentence to `On since <date>: <the one-line numbers>.`, run `python3 -m pytest tests/ -q`, and commit:

```bash
git add research_assistant/config.py
git commit -m "feat(citer): the judge acceptance test is on by default

Gate on the citer evaluation, <N> cited sentences from three seeds, 2 runs:
<paste the --compare output>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

If it fails, leave the default off and record the comparison in an empty commit (`git commit --allow-empty`) with the same body, so the measurement is on record and the code stays.

---

## Phase 3 — B: the query is not the bare sentence

### Task 10: Paragraph-aware citing with contextualized queries

**Files:**
- Modify: `research_assistant/config.py` (after `CITATION_CITER_JUDGE`)
- Modify: `research_assistant/agents/agent5_batch_citer.py` (`run_batch_citer`: the split, a contextualization block, the `cite_sentence` call)
- Modify: `scripts/evaluate_citer.py` (`run` signature and loop, `main`)
- Test: `tests/test_batch_citer.py`, `tests/test_evaluate_citer.py`

**Interfaces:**
- Consumes: `seed_audit.contextualize_citation_queries(claims, model=None) -> None` — sets `claim["search_query"]` on each dict in `claims`, batching one model call per distinct `paragraph_id`; tolerates claims without `ref` (imported inside the function — `seed_audit` imports this module); `claim_text.sentence_context(sentences, idx) -> str`; `cite_sentence(..., context=, query=, paragraph_id=)` from Task 2.
- Produces: `claim_text.split_into_sentences(text) -> list[str]` (moved from agent5, which re-exports it) and `claim_text._ABBREVS`; `config.CITATION_CITER_CONTEXTUALIZE: bool` (env, default False); `agent5_batch_citer.split_paragraphs(text) -> list[list[str]]` (sentences per blank-line paragraph); `agent5_batch_citer.contextualized_queries(sentences, contexts, paragraph_ids, needs_cite) -> dict[int, str]`; `evaluate_citer.run(..., contextualize=None)`; CLI `--query {raw,ctx,config}`; results file `<stamp>-judge<on|off>-query<raw|ctx>.json`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_batch_citer.py` (extend the import to include `split_paragraphs`, `contextualized_queries`):

```python
class TestParagraphs(unittest.TestCase):
    def test_blank_lines_split_paragraphs_and_sentences_are_unchanged(self):
        from research_assistant.agents.agent5_batch_citer import split_into_sentences
        text = "A one. A two.\n\nB one.\n\n\n  C one. C two. C three."
        paras = split_paragraphs(text)
        self.assertEqual(paras, [["A one.", "A two."], ["B one."], ["C one.", "C two.", "C three."]])
        # The flat sentence list the draft is rebuilt from is what the whole-text split gave before.
        self.assertEqual([s for p in paras for s in p], split_into_sentences(text))

    def test_a_draft_without_blank_lines_is_one_paragraph(self):
        self.assertEqual(split_paragraphs("A one. A two."), [["A one.", "A two."]])


class TestContextualizedQueries(unittest.TestCase):
    SENTS = ["We use the recursive Green's function method.", "This approach scales linearly.", "Unrelated."]
    CTX = ["«We use the recursive Green's function method.» This approach scales linearly.",
           "We use the recursive Green's function method. «This approach scales linearly.» Unrelated.",
           "This approach scales linearly. «Unrelated.»"]
    PIDS = ["p_0", "p_0", "p_1"]

    def test_one_call_per_paragraph_with_the_sentences_that_need_a_citation(self):
        def fake(claims, model=None):
            for c in claims:
                c["search_query"] = "Q: " + c["claim"]
        with patch("research_assistant.shared.seed_audit.contextualize_citation_queries", side_effect=fake) as ctx:
            out = contextualized_queries(self.SENTS, self.CTX, self.PIDS, needs_cite=[True, True, False])
        self.assertEqual(out, {0: "Q: We use the recursive Green's function method.", 1: "Q: This approach scales linearly."})
        ctx.assert_called_once()
        claims = ctx.call_args[0][0]
        self.assertEqual([c["paragraph_id"] for c in claims], ["p_0", "p_0"])
        self.assertEqual(claims[1]["context"], self.CTX[1])

    def test_failure_falls_back_to_no_queries(self):
        with patch("research_assistant.shared.seed_audit.contextualize_citation_queries", side_effect=RuntimeError("down")):
            self.assertEqual(contextualized_queries(self.SENTS, self.CTX, self.PIDS, needs_cite=[True, True, True]), {})

    def test_nothing_needing_a_citation_makes_no_call(self):
        with patch("research_assistant.shared.seed_audit.contextualize_citation_queries") as ctx:
            self.assertEqual(contextualized_queries(self.SENTS, self.CTX, self.PIDS, needs_cite=[False, False, False]), {})
        ctx.assert_not_called()


class TestRunBatchCiterPassesContextAndQuery(unittest.TestCase):
    DRAFT = "We use the recursive Green's function method. This approach scales linearly.\n\nUnrelated paragraph here."

    def _run(self, contextualize):
        seen = []

        def fake_cite(sentence, resources, key_registry, **kw):
            seen.append((sentence, kw))
            return CiteResult(original=sentence, cited_text=sentence, query=kw.get("query") or sentence,
                              skip_reason="no relevant context found in database")

        def fake_ctx(claims, model=None):
            for c in claims:
                c["search_query"] = "Q: " + c["claim"]

        with tempfile.TemporaryDirectory() as tmpdir:
            draft_path = os.path.join(tmpdir, "draft.txt")
            with open(draft_path, "w") as f:
                f.write(self.DRAFT)
            with patch("research_assistant.agents.agent5_batch_citer.CITATION_CITER_CONTEXTUALIZE", contextualize), \
                 patch("research_assistant.agents.agent5_batch_citer.load_search_resources", return_value=(None, None, [], [])), \
                 patch("research_assistant.agents.agent5_batch_citer.chat", return_value=_reply("1. YES\n2. YES\n3. NO")), \
                 patch("research_assistant.agents.agent5_batch_citer.cite_sentence", side_effect=fake_cite), \
                 patch("research_assistant.shared.seed_audit.contextualize_citation_queries", side_effect=fake_ctx):
                run_batch_citer(draft_path, os.path.join(tmpdir, "cited.txt"))
        return seen

    def test_context_and_paragraph_always_reach_the_seam(self):
        seen = self._run(contextualize=False)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1][1]["context"],
                         "We use the recursive Green's function method. «This approach scales linearly.»")
        self.assertEqual(seen[1][1]["paragraph_id"], "p_0")
        self.assertIsNone(seen[1][1]["query"])

    def test_the_contextualized_query_reaches_the_seam_when_on(self):
        seen = self._run(contextualize=True)
        self.assertEqual(seen[1][1]["query"], "Q: This approach scales linearly.")
```

And in `tests/test_evaluate_citer.py`, add to `TestRun`:

```python
    def test_contextualize_builds_queries_per_seed_and_forwards_them(self):
        seen = {}

        def fake_cite(sentence, resources, key_registry, **kw):
            seen[sentence] = kw
            return CiteResult(original=sentence, cited_text=sentence, query=kw.get("query") or sentence,
                              skip_reason="no relevant context found in database")

        def fake_ctx(claims, model=None):
            for c in claims:
                c["search_query"] = "Q: " + c["claim"]

        cases = [_case("c_1"), _case("c_2", sentence="Another claim with enough words for the check.")]
        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True, True]), \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite), \
             patch("research_assistant.shared.seed_audit.contextualize_citation_queries", side_effect=fake_ctx) as ctx:
            out = ec.run(cases, self.RESOURCES, runs=1, contextualize=True)
        ctx.assert_called_once()
        self.assertEqual(seen[cases[0]["sentence"]]["query"], "Q: " + cases[0]["sentence"])
        self.assertIs(out["contextualize"], True)
        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True, True]), \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite):
            ec.run(cases, self.RESOURCES, runs=1, contextualize=False)
        self.assertIsNone(seen[cases[0]["sentence"]]["query"])
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_batch_citer.py tests/test_evaluate_citer.py -q`
Expected: FAIL — `ImportError: cannot import name 'split_paragraphs'`; `TypeError: … unexpected keyword argument 'contextualize'`.

- [ ] **Step 3: The config flag**

In `research_assistant/config.py`, directly after the `CITATION_CITER_JUDGE` line:

```python
# The citer searches with a query written from the sentence's paragraph —
# pronouns and "this approach" resolved — instead of the bare sentence, the
# way the audit already does. Off until its gate on the citer evaluation
# passes — scripts/evaluate_citer.py, query raw against ctx.
CITATION_CITER_CONTEXTUALIZE = _env_bool("CITATION_CITER_CONTEXTUALIZE", False)
```

- [ ] **Step 4: Paragraphs, contexts and queries in the citer**

(0) **Break the import cycle first.** `research_assistant/shared/claim_text.py` line 23 reads
`from research_assistant.agents.agent5_batch_citer import split_into_sentences`, so agent5 cannot
import `sentence_context` from `claim_text`. The splitter is a text helper and belongs in
`claim_text`; move it there and let agent5 re-export it.

In `research_assistant/shared/claim_text.py`, delete line 23 and, directly after the `import re`
line, paste the `_ABBREVS` constant and the whole `split_into_sentences` function **verbatim** from
`agent5_batch_citer.py` (they begin at its line 24, `_ABBREVS = r"(?:et al|Fig|…`, and end with
`return [s.replace(_TOKEN, '.').strip() for s in sentences if s.strip()]`). Then in
`agent5_batch_citer.py` delete those same lines (the constant, the function, and the
`# ─── Improved sentence splitter` heading above them) and add to its imports:

```python
from research_assistant.shared.claim_text import sentence_context, split_into_sentences  # re-exported: seed_audit, agent8 and tests import it from here
```

Add to `tests/test_batch_citer.py`:

```python
class TestSplitterLivesInClaimText(unittest.TestCase):
    def test_agent5_re_exports_the_shared_splitter(self):
        from research_assistant.shared import claim_text
        from research_assistant.agents import agent5_batch_citer
        self.assertIs(agent5_batch_citer.split_into_sentences, claim_text.split_into_sentences)
        self.assertEqual(claim_text.split_into_sentences("See Fig. 3. Then et al. agreed."), ["See Fig. 3.", "Then et al. agreed."])
```

Run `python3 -m pytest tests/test_batch_citer.py tests/test_seed_audit.py tests/test_extract.py -q` — everything that imported the splitter from agent5 still passes.

In `research_assistant/agents/agent5_batch_citer.py`:

(a) Add `CITATION_CITER_CONTEXTUALIZE,` to the `from research_assistant.config import (` block. (`sentence_context` arrived with (0).)

(b) Directly after `# ─── Citation key extraction` would be, i.e. before `_CITE_RE = re.compile(`, add:

```python
def split_paragraphs(text: str) -> list:
    """Sentences per paragraph, paragraphs being blank-line separated. The
    flat sentence list is what split_into_sentences gave for the whole text
    — a paragraph break is whitespace after a full stop to the splitter —
    so the draft is rebuilt exactly as before; the paragraphs are for the
    context each sentence is cited in."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [split_into_sentences(p) for p in paragraphs] or [[]]


def contextualized_queries(sentences, contexts, paragraph_ids, needs_cite) -> dict:
    """{sentence index: query} for the sentences that need a citation, from
    the audit's query contextualization — one model call per paragraph. A
    failure means no queries: the caller searches with the sentences."""
    from research_assistant.shared.seed_audit import contextualize_citation_queries  # seed_audit imports this module

    indices = [i for i, need in enumerate(needs_cite) if need]
    if not indices:
        return {}
    claims = [{"claim": sentences[i], "context": contexts[i], "paragraph_id": paragraph_ids[i]} for i in indices]
    try:
        contextualize_citation_queries(claims)
    except Exception as exc:  # noqa: BLE001 — the query is better with it, fine without
        logger.warning("Query contextualization failed (%s) — searching with the sentences.", exc)
        return {}
    return {i: c["search_query"] for i, c in zip(indices, claims) if c.get("search_query")}
```

(c) In `run_batch_citer`, replace

```python
    sentences = split_into_sentences(draft_text)
    logger.info("Split draft into %d sentences.", len(sentences))
```

with

```python
    sentences, contexts, paragraph_ids = [], [], []
    for p_idx, para in enumerate(split_paragraphs(draft_text)):
        for s_idx, sent in enumerate(para):
            sentences.append(sent)
            contexts.append(sentence_context(para, s_idx))
            paragraph_ids.append(f"p_{p_idx}")
    logger.info("Split draft into %d sentences in %d paragraph(s).", len(sentences), len(set(paragraph_ids)))
```

(d) Directly after the need-check block (after the `for idx, needs in zip(eligible_indices, batch_results): needs_cite[idx] = needs` loop), add:

```python
    queries = contextualized_queries(sentences, contexts, paragraph_ids, needs_cite) if CITATION_CITER_CONTEXTUALIZE else {}
```

(e) The `cite_sentence(` call in the loop becomes:

```python
            res = cite_sentence(sentence, (collection, bm25, texts, metadatas), key_registry,
                                context=contexts[i], query=queries.get(i), paragraph_id=paragraph_ids[i])
```

- [ ] **Step 5: The harness**

In `scripts/evaluate_citer.py`:

(a) `def run(cases, resources, runs=1, *, judge_gate=None):` becomes `def run(cases, resources, runs=1, *, judge_gate=None, contextualize=None):`.

(b) Inside the per-seed loop, directly after `key_registry = {}`, add:

```python
            queries = {}
            if contextualize:
                needing = [c for c in seed_cases if need_by_id.get(c["id"]) and not need_error]
                q = citer.contextualized_queries(
                    [c["sentence"] for c in needing], [c.get("context") or "" for c in needing],
                    [c.get("paragraph_id") or "p_0" for c in needing], [True] * len(needing),
                )
                queries = {needing[i]["id"]: v for i, v in q.items()}
```

and the `cite_sentence(` call gains `query=queries.get(c["id"]),` after `context=c.get("context"),`. The returned dict gains `"contextualize": contextualize,` after `"judge_gate": judge_gate,`.

(c) In `main()`: add `ap.add_argument("--query", choices=["raw", "ctx", "config"], default="config", help="search with the sentence, a contextualized query, or whatever CITATION_CITER_CONTEXTUALIZE says")`. In the `--score` branch, after `judge_gate` is resolved:

```python
        contextualize = {"ctx": True, "raw": False}.get(args.query)
        if contextualize is None:
            from research_assistant.config import CITATION_CITER_CONTEXTUALIZE
            contextualize = CITATION_CITER_CONTEXTUALIZE
        out = run(cases, load_search_resources(), runs=args.runs, judge_gate=judge_gate, contextualize=contextualize)
```

and the filename becomes `f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-judge{'on' if judge_gate else 'off'}-query{'ctx' if contextualize else 'raw'}.json"`.

- [ ] **Step 6: Run the two files, then the suite**

Run: `python3 -m pytest tests/test_batch_citer.py tests/test_evaluate_citer.py -q && python3 -m pytest tests/ -q`
Expected: all pass — including the pre-existing `TestRunBatchCiter`, whose one-sentence draft is one paragraph.

- [ ] **Step 7: Commit**

```bash
git add research_assistant/config.py research_assistant/agents/agent5_batch_citer.py research_assistant/shared/claim_text.py scripts/evaluate_citer.py tests/test_batch_citer.py tests/test_evaluate_citer.py
git commit -m "feat(citer): paragraph-aware citing; queries written from the paragraph

The citer searched with the bare sentence, so 'this approach outperforms
the baseline' searched for those words. Sentences now carry their
paragraph and window, and behind CITATION_CITER_CONTEXTUALIZE (off
until gated) the query is the audit's contextualized one, one model call
per paragraph. The draft is rebuilt exactly as before.

split_into_sentences moves to shared.claim_text, where it belongs and
where it no longer makes claim_text depend on an agent; agent5 re-exports
it for the modules that import it from there.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Gate B — raw against contextualized

**Files:** none tracked unless the gate passes; then `research_assistant/config.py` (the default).

- [ ] **Step 1: Score both arms with the judge in whatever state Task 9 left it**

```bash
export CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=.
PY=/home/shardul/miniconda3/envs/ml/bin/python
$PY scripts/evaluate_citer.py --score --runs 2 --query raw
$PY scripts/evaluate_citer.py --score --runs 2 --query ctx
ls -t data/eval/citer/results/ | head -2
```

- [ ] **Step 2: Compare**

```bash
$PY scripts/evaluate_citer.py --compare data/eval/citer/results/<queryraw>.json data/eval/citer/results/<queryctx>.json
```

Acceptance (spec §3.4): `sentence_hit` rises and `target_precision` does not fall by more than the noise floor (`stability` within an arm).

- [ ] **Step 3: Ship or record**

If it passes, flip the default — `CITATION_CITER_CONTEXTUALIZE = _env_bool("CITATION_CITER_CONTEXTUALIZE", True)` — update the comment's last sentence to `On since <date>: <numbers>.`, run `python3 -m pytest tests/ -q`, and commit:

```bash
git add research_assistant/config.py
git commit -m "feat(citer): contextualized queries are on by default

Gate on the citer evaluation, <N> cited sentences from three seeds, 2 runs:
<paste the --compare output>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

If it fails, leave the default off and record the comparison in an empty commit with the same body.

---

## Not in this plan

- Citing at the claim rather than the sentence, several cites per sentence (spec idea D); source preference (E); summary-index retrieval (F); two-stage retrieval for the citer (C). Each is a later gate on the evaluation this plan builds.
- Agent 4 (`suggest_citation`) and Agent 7. Agent 4 is a different surface — a suggestion prompt, not a rewrite.
- Moving the judge's eval modules under `research_assistant/eval/`.
