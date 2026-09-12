# Synthesis Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Tab 1 related-work synthesis reads every paper it shortlists, writes per-paper notes and then an argued synthesis — what is established, where the papers differ, the gap — cites papers by checkable keys, runs at a chosen temperature and context, and shows its progress; the batched gate and a search-resource cache make the extra reading affordable.

**Architecture:** `shared/retrieve.py` is restructured around four pure-ish steps — de-duplicated, keyed shortlist → per-paper passages → per-paper notes (map) → synthesis (reduce) — with a `single` mode that skips the map. `shared/db.py` gains a per-process cache. `prompts.py` gains three prompts. `app.py` renders notes and a key legend and feeds a progress callback. No agent, index, or orchestrator node changes shape; `research_answer()` keeps every key it returns today and adds new ones.

**Tech Stack:** Python 3.10–3.12, pytest collecting `unittest.TestCase`, Ollama `gemma4:e2b` via `shared.llm.chat`, ChromaDB + `rank_bm25` via `shared.search.hybrid_search`, Streamlit.

**Spec:** `notimportant/superpowers/specs/2026-09-12-synthesis-depth-design.md` — §2 A–F are the tasks' authority; §4 the config names.

---

## Context you need before Task 1

### The path today

```
Tab 1 (app.py ~line 666, batch-upload path) ─┐
orchestrate.respond node ────────────────────┴─► retrieve.research_answer(query)
    rank_documents(query)            summary kNN → [{document, citation, summary, score}] (DOC_SELECT_K=6)
    gate_documents(query, ranked)    ONE model call PER summary: "relevant prior work? YES/NO"
    deep_search(query, docs, top_k)  load_search_resources() [full reload] → hybrid_search over the shortlist → GLOBAL top-6 chunks
    context = "[title] (from doc, p.N)\n<chunk>" × 6
    chat([RESEARCH_CHAT_SYSTEM, RELATED_WORK_USER.format(query, context)])   ← no temperature, no options
    → {"suggestion": text, "citations": [titles seen in passages], "passages": hits, "selected": shortlist}
```

### What the "before" looked like (2026-09-12, v2 index, `"Anderson localization and transport in disordered quasi-one-dimensional wires"`)

174 s. Six shortlisted (one of them twice: same paper, two document keys). Six passages — from **two** papers. 475 words; nine citations to one paper, three to another, zero to the other four (one of which is Evers & Mirlin, *Metal-Insulator Transitions*, Rev. Mod. Phys.). Gap paragraph: *"Your research could add significant value by applying these established localization theories…"*. Recorded at `/tmp/claude-1000/-run-media-shardul-storage1-research-assistant/cb5fde9b-3785-4318-8666-5ac0f1eb2bd7/scratchpad/synth_before.json` if it still exists.

### The data shapes you will handle

- `rank_documents(query, k)` → `list[{"document": str, "citation": str (paper title), "summary": str, "score": float}]`.
- `hybrid_search(query, collection, bm25, texts, metadatas, top_k, doc_filter=set|None, exclude_types=set|None)` → `list[{"chunk_index", "text", "metadata": {"document", "citation_source", "type", "page", …v2: "section", "page_first", …}, "rrf_score"}]`.
- `chat(messages, model=None, images=None, temperature=None, options=None) -> ChatResult(content, prompt_tokens, completion_tokens)`. `CHAT_OLLAMA_OPTIONS = {"num_ctx": 4096}` in config; `JUDGEMENT_OLLAMA_OPTIONS = {"num_ctx": 8192}` is the precedent for a bigger window on one call.
- `gemma4:e2b`: 5.1 B parameters, 131k native context; the modelfile's default temperature is **1.0** — every call without an explicit temperature samples at 1.0.

### Environment

- Interpreter `/home/shardul/miniconda3/envs/ml/bin/python`; scripts with `PYTHONPATH=.` from the repo root. Tests: `CITATION_LOG_FILE=0 python -m pytest tests/ -v`; `unittest.TestCase`. Ollama on `localhost:11434`; the v2 index via `CITATION_INDEX_VERSION=2` — needed only in Task 7.
- Branch: from `judge-evaluation` if it exists, else `judgement-hardening`, else `ingestion-v2`: `git checkout <base> && git checkout -b synthesis-depth`. **Never push, never merge.** Commit after every task.
- `data/ingest.lock` held by another process blocks `tests/test_ingestion.py`; deselect it if so.

### Pitfalls from previous builds on this repo

1. A lazy import deleted while editing the block around it → `NameError` at runtime, invisible to the suite (`tests/test_app_names.py` guards `app.py`; nothing guards the agents). Keep local imports where you find them.
2. Production code bent to satisfy a synthetic test input. If a planned test contradicts what the code does on real input, fix the test and say so in the report.
3. Verification notes written without the run. Task 7 asks for pasted output.

## Global Constraints

- **Python 3.10–3.12.** No 3.13-only syntax.
- **No provider SDK outside `shared/llm.py`.** Every model call goes through `shared.llm.chat`.
- **`research_answer()` keeps its existing return keys** (`suggestion`, `citations`, `passages`, `selected`) with their meanings; new keys are additive. `orchestrate.respond` is not edited.
- **Every model call in this path passes an explicit `temperature` and `options`.** (Spec §2B.)
- **`figure_description` chunks are never passages here** (`exclude_types={"figure_description"}` on every `hybrid_search` call).
- **Heavy imports stay inside functions** (`chromadb`). `retrieve.py` may import `hybrid_search`, `load_search_resources` and `chat` at module scope — those modules are light at import time — so tests can patch `retrieve.<name>`.
- **`unittest.TestCase`** classes; live-model tests behind `RUN_LLM_TESTS=1`.
- Run the suite with `CITATION_LOG_FILE=0`.
- **Commit after every task; do not push.**

---

### Task 1: `load_search_resources()` is cached per process

**Files:**
- Modify: `research_assistant/shared/db.py` (wrap the loader; add `clear_search_cache()`)
- Test: `tests/test_db.py` (extend)

**Interfaces:**
- Produces: `load_search_resources()` returns the same tuple object on repeated calls while `(VECTORDB_PATH, COLLECTION_NAME, BM25_INDEX_PATH, os.path.getmtime(BM25_INDEX_PATH))` is unchanged; `clear_search_cache()` empties it. `ingestion.rebuild_bm25()` calls `clear_search_cache()` after writing the pickle.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_db.py` (the file's `LoadTestCase` already fakes Chroma and points `db.BM25_INDEX_PATH` at a temp pickle):

```python


class TestSearchResourceCache(LoadTestCase):
    def setUp(self):
        super().setUp()
        db.clear_search_cache()
        self.addCleanup(db.clear_search_cache)
        with open(self.pkl, "wb") as f:
            pickle.dump({"tokenizer": "v2", "built_at": "now", "bm25": _Bm25Stub()}, f)

    def test_second_call_returns_the_same_objects_without_reloading(self):
        first = db.load_search_resources()
        with patch.object(db, "_load_search_resources_uncached", side_effect=AssertionError("reloaded")):
            second = db.load_search_resources()
        self.assertIs(first, second)

    def test_a_rebuilt_pickle_invalidates(self):
        first = db.load_search_resources()
        with open(self.pkl, "wb") as f:
            pickle.dump({"tokenizer": "v2", "built_at": "later", "bm25": _Bm25Stub()}, f)
        os.utime(self.pkl, (time.time() + 5, time.time() + 5))
        second = db.load_search_resources()
        self.assertIsNot(first, second)

    def test_clear_forces_a_reload(self):
        first = db.load_search_resources()
        db.clear_search_cache()
        self.assertIsNot(first, db.load_search_resources())
```

(Add `import time` to the file's imports if absent; `os`, `pickle`, `patch` are already there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_db.py -q -k Cache`
Expected: FAIL — `AttributeError: module … has no attribute 'clear_search_cache'`.

- [ ] **Step 3: Implement**

In `research_assistant/shared/db.py`, rename the existing `def load_search_resources():` to `def _load_search_resources_uncached():` (body unchanged), and add above it:

```python
# One process, one copy of the corpus in memory. The synthesis path calls
# hybrid_search once per shortlisted paper, and each call used to page the
# whole collection out of Chroma and unpickle 13 MB — per query, per paper.
# Keyed on the pickle's mtime so a rebuild (ingest) invalidates it.
_cache: dict = {}


def _cache_key():
    try:
        mtime = os.path.getmtime(BM25_INDEX_PATH)
    except OSError:
        mtime = None
    return (VECTORDB_PATH, COLLECTION_NAME, BM25_INDEX_PATH, mtime)


def clear_search_cache() -> None:
    _cache.clear()


def load_search_resources():
    """Cached: see _load_search_resources_uncached() for what is loaded."""
    key = _cache_key()
    if key not in _cache:
        _cache.clear()
        _cache[key] = _load_search_resources_uncached()
    return _cache[key]
```

In `research_assistant/shared/ingestion.py`, at the end of `rebuild_bm25()` after the pickle is written (before the final `logger.info`), add:

```python
    # The cached (collection, bm25, texts, metadatas) in this process is now
    # stale; the mtime key would catch it, but say so explicitly.
    from research_assistant.shared.db import clear_search_cache
    clear_search_cache()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_db.py tests/test_search.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/db.py research_assistant/shared/ingestion.py tests/test_db.py
git commit -m "perf(search): cache load_search_resources per process, keyed on the BM25 pickle's mtime"
```

---

### Task 2: Batched relevance gate

**Files:**
- Modify: `research_assistant/prompts.py` (add `DOC_RELEVANCE_GATE_BATCH`), `research_assistant/config.py` (add `GATE_BATCHED`), `research_assistant/shared/retrieve.py` (`parse_gate_verdicts`, `gate_documents`)
- Test: `tests/test_retrieve.py` (new)

**Interfaces:**
- Produces: `retrieve.parse_gate_verdicts(text: str, n: int) -> list[bool] | None` (None on any misalignment); `retrieve.gate_documents(query, ranked)` makes one batched call when `GATE_BATCHED` and falls back to the per-summary loop (`_gate_one_by_one`) on `None`; `config.GATE_BATCHED = _env_bool("CITATION_GATE_BATCHED", True)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieve.py
"""The synthesis path: gate, shortlist, per-paper passages, notes, synthesis.

Everything below stubs the model and the index; the live behaviour is
checked in the plan's Task 7. What these pin is the bookkeeping that makes
the output honest: verdicts aligned to their summaries, every shortlisted
paper read, keys checked against the shortlist.
"""

import types
import unittest
from unittest.mock import patch

from research_assistant.shared import retrieve as rt


def _reply(text):
    return types.SimpleNamespace(content=text)


def _ranked(n):
    return [{"document": f"d{i}.pdf", "citation": f"Paper {i}", "summary": f"summary {i}", "score": 1 - i / 10}
            for i in range(1, n + 1)]


class TestParseGateVerdicts(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(rt.parse_gate_verdicts("1: YES\n2: NO\n3: yes", 3), [True, False, True])

    def test_tolerates_punctuation_and_preamble(self):
        self.assertEqual(rt.parse_gate_verdicts("Here you go:\n1) YES\n2. NO\n", 2), [True, False])

    def test_out_of_order_is_indexed_by_number(self):
        self.assertEqual(rt.parse_gate_verdicts("2: NO\n1: YES", 2), [True, False])

    def test_missing_duplicate_or_extra_is_none(self):
        self.assertIsNone(rt.parse_gate_verdicts("1: YES", 2))
        self.assertIsNone(rt.parse_gate_verdicts("1: YES\n1: NO", 2))
        self.assertIsNone(rt.parse_gate_verdicts("1: YES\n2: NO\n3: YES", 2))
        self.assertIsNone(rt.parse_gate_verdicts("YES NO", 2))


class TestGateDocuments(unittest.TestCase):
    def test_batched_gate_is_one_call_and_keeps_the_yeses(self):
        with patch.object(rt, "GATE_BATCHED", True), \
             patch.object(rt, "chat", return_value=_reply("1: YES\n2: NO\n3: YES")) as chat:
            kept = rt.gate_documents("idea", _ranked(3))
        self.assertEqual(chat.call_count, 1)
        self.assertEqual([r["document"] for r in kept], ["d1.pdf", "d3.pdf"])
        prompt = chat.call_args.args[0][-1]["content"]
        self.assertIn("summary 2", prompt); self.assertIn("idea", prompt)

    def test_misaligned_reply_falls_back_to_per_summary_calls(self):
        replies = iter([_reply("1: YES\n1: YES"), _reply("YES"), _reply("NO"), _reply("YES")])
        with patch.object(rt, "GATE_BATCHED", True), patch.object(rt, "chat", side_effect=lambda *a, **k: next(replies)) as chat:
            kept = rt.gate_documents("idea", _ranked(3))
        self.assertEqual(chat.call_count, 4)
        self.assertEqual([r["document"] for r in kept], ["d1.pdf", "d3.pdf"])

    def test_everything_rejected_keeps_the_top_summary(self):
        with patch.object(rt, "GATE_BATCHED", True), patch.object(rt, "chat", return_value=_reply("1: NO\n2: NO")):
            kept = rt.gate_documents("idea", _ranked(2))
        self.assertEqual([r["document"] for r in kept], ["d1.pdf"])

    def test_batched_off_uses_per_summary_calls(self):
        with patch.object(rt, "GATE_BATCHED", False), patch.object(rt, "chat", return_value=_reply("YES")) as chat:
            rt.gate_documents("idea", _ranked(3))
        self.assertEqual(chat.call_count, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_retrieve.py -q`
Expected: FAIL — `AttributeError: … 'parse_gate_verdicts'` (and `retrieve.chat` does not exist yet at module scope).

- [ ] **Step 3: Prompt and config**

In `research_assistant/prompts.py`, after `DOC_RELEVANCE_GATE`:

```python
# One call for the whole shortlist. Verdicts come back numbered so the
# parser can align them to the summaries — or refuse to guess.
DOC_RELEVANCE_GATE_BATCH = (
    "A researcher is exploring this idea:\n\"{query}\"\n\n"
    "Below are summaries of {n} papers, numbered. For each one, decide whether "
    "the paper could be relevant prior work for that idea — even loosely.\n\n"
    "{summaries}\n\n"
    "Answer with exactly {n} lines, one per paper, in order, each of the form "
    "`N: YES` or `N: NO`. Nothing else."
)
```

In `research_assistant/config.py`, after `DOC_GATE = ...`:

```python
# One gate call listing every shortlisted summary instead of one call per
# summary. Falls back to per-summary calls when the reply cannot be aligned.
GATE_BATCHED     = _env_bool("CITATION_GATE_BATCHED", True)
```

- [ ] **Step 4: `retrieve.py` — imports and the gate**

At the top of `research_assistant/shared/retrieve.py`, replace the imports with:

```python
import re
import time

from research_assistant.config import (
    VECTORDB_PATH,
    SUMMARY_COLLECTION_NAME,
    DOC_SELECT_K,
    DOC_GATE,
    GATE_BATCHED,
    DEFAULT_TOP_K,
)
from research_assistant.prompts import (
    DOC_RELEVANCE_GATE,
    DOC_RELEVANCE_GATE_BATCH,
    RESEARCH_CHAT_SYSTEM,
    RELATED_WORK_USER,
)
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.llm import chat
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import hybrid_search
```

and remove the function-local `from research_assistant.shared.llm import chat` / `db` / `search` imports inside `gate_documents`, `deep_search` and `research_answer` (they now come from the module scope; `rank_documents` keeps its local `import chromadb` and `get_embeddings`).

Replace `gate_documents` with:

```python
_VERDICT_RE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(YES|NO)\b", re.I | re.M)


def parse_gate_verdicts(text: str, n: int):
    """Exactly one verdict per summary, indexed by number, or None.

    Same rule as Agent 5's batched citation-need check: a reply that cannot
    be aligned to its inputs is not partially trusted — it is discarded and
    the per-summary path runs.
    """
    found = {}
    for m in _VERDICT_RE.finditer(text or ""):
        idx = int(m.group(1))
        if idx in found:
            return None
        found[idx] = m.group(2).upper() == "YES"
    if set(found) != set(range(1, n + 1)):
        return None
    return [found[i] for i in range(1, n + 1)]


def _gate_one_by_one(query: str, ranked: list[dict]) -> list[dict]:
    kept = []
    for r in ranked:
        try:
            ans = chat([{
                "role": "user",
                "content": DOC_RELEVANCE_GATE.format(query=query, summary=r["summary"]),
            }]).content.strip().upper()
        except Exception as e:
            logger.warning("Gate call failed for %s: %s — keeping it.", r["document"], e)
            kept.append(r)
            continue
        if ans.startswith("Y"):
            kept.append(r)
    return kept


def gate_documents(query: str, ranked: list[dict]) -> list[dict]:
    """Ask the LLM to keep only the summaries that could be relevant prior work."""
    kept = None
    if GATE_BATCHED and ranked:
        summaries = "\n\n".join(f"{i}. {r['summary']}" for i, r in enumerate(ranked, 1))
        try:
            reply = chat([{
                "role": "user",
                "content": DOC_RELEVANCE_GATE_BATCH.format(query=query, n=len(ranked), summaries=summaries),
            }]).content
            verdicts = parse_gate_verdicts(reply, len(ranked))
        except Exception as e:
            logger.warning("Batched gate call failed: %s — falling back to per-summary calls.", e)
            verdicts = None
        if verdicts is None:
            logger.info("Batched gate reply could not be aligned to %d summaries — per-summary calls.", len(ranked))
        else:
            kept = [r for r, keep in zip(ranked, verdicts) if keep]
    if kept is None:
        kept = _gate_one_by_one(query, ranked)

    if not kept:
        logger.info("Gate rejected everything — falling back to the top summary.")
        return ranked[:1]
    logger.info("Gate kept %d/%d documents.", len(kept), len(ranked))
    return kept
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_retrieve.py -q` — Expected: PASS (the `TestGate*` classes; later classes are added in Tasks 3–4).

- [ ] **Step 6: Commit**

```bash
git add research_assistant/prompts.py research_assistant/config.py research_assistant/shared/retrieve.py tests/test_retrieve.py
git commit -m "perf(retrieve): batched relevance gate — one call per shortlist, per-summary fallback on misalignment"
```

---

### Task 3: De-duplicated, keyed shortlist and per-paper passages

**Files:**
- Modify: `research_assistant/shared/retrieve.py`
- Test: `tests/test_retrieve.py` (extend)

**Interfaces:**
- Produces:
  ```python
  def dedupe_shortlist(ranked: list[dict]) -> list[dict]        # by normalised title, higher score kept, score-descending
  def assign_keys(selected: list[dict]) -> dict[str, str]       # sets r["key"] = "P1".. in order; returns {key: title}
  def passages_per_paper(query: str, selected: list[dict], per_paper: int) -> dict[str, list[dict]]   # key → hits (each hit gets "key")
  def format_passages(hits: list[dict], max_chars: int) -> str
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieve.py`:

```python


class TestShortlist(unittest.TestCase):
    def test_dedupe_by_title_keeps_the_higher_score(self):
        ranked = [{"document": "arxiv_1.pdf", "citation": "Conductance Quantization in Ribbons", "summary": "a", "score": 0.7},
                  {"document": "doi_1.pdf", "citation": "conductance quantization in ribbons.", "summary": "b", "score": 0.9},
                  {"document": "d3.pdf", "citation": "Other", "summary": "c", "score": 0.8}]
        out = rt.dedupe_shortlist(ranked)
        self.assertEqual([r["document"] for r in out], ["doi_1.pdf", "d3.pdf"])

    def test_keys_are_assigned_in_order(self):
        sel = _ranked(3)
        keys = rt.assign_keys(sel)
        self.assertEqual([r["key"] for r in sel], ["P1", "P2", "P3"])
        self.assertEqual(keys, {"P1": "Paper 1", "P2": "Paper 2", "P3": "Paper 3"})


class TestPassagesPerPaper(unittest.TestCase):
    def test_one_search_per_paper_restricted_to_it(self):
        sel = _ranked(2); rt.assign_keys(sel)
        calls = []

        def fake_search(query, collection, bm25, texts, metadatas, top_k, doc_filter=None, exclude_types=None, **kw):
            calls.append((top_k, doc_filter, exclude_types))
            doc = next(iter(doc_filter))
            return [{"chunk_index": 1, "text": f"chunk of {doc}", "metadata": {"document": doc, "page": 3, "section": "results"}}]

        with patch.object(rt, "load_search_resources", return_value=(None, None, [], [])), \
             patch.object(rt, "hybrid_search", side_effect=fake_search):
            out = rt.passages_per_paper("idea", sel, per_paper=4)
        self.assertEqual(calls, [(4, {"d1.pdf"}, {"figure_description"}), (4, {"d2.pdf"}, {"figure_description"})])
        self.assertEqual(list(out), ["P1", "P2"])
        self.assertEqual(out["P1"][0]["key"], "P1")

    def test_format_passages_caps_and_labels(self):
        hits = [{"text": "A" * 30, "metadata": {"page": 2, "section": "methods"}},
                {"text": "B" * 30, "metadata": {"page": 5, "section": "results"}}]
        out = rt.format_passages(hits, max_chars=60)
        self.assertTrue(out.startswith("(p.2, methods) " + "A" * 30))
        self.assertLessEqual(len(out), 60 + len("\n\n"))
        self.assertIn("(p.5, results)", out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_retrieve.py -q -k "Shortlist or Passages"`
Expected: FAIL — `AttributeError: … 'dedupe_shortlist'`.

- [ ] **Step 3: Implement**

Add to `research_assistant/shared/retrieve.py` after `gate_documents`:

```python
# ─── Shortlist → keys → per-paper passages ──────────────────────────────────


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def dedupe_shortlist(ranked: list[dict]) -> list[dict]:
    """The same paper under two document keys (an arXiv copy and a DOI copy)
    must not take two slots. Keyed on the normalised title; the higher
    stage-1 score wins; order is score-descending."""
    best = {}
    for r in ranked:
        k = _norm_title(r.get("citation")) or r["document"]
        if k not in best or r.get("score", 0) > best[k].get("score", 0):
            best[k] = r
    return sorted(best.values(), key=lambda r: -r.get("score", 0))


def assign_keys(selected: list[dict]) -> dict:
    """P1..Pn in shortlist order. Short keys are what a 5B model reproduces
    exactly, and what check_citation_keys() can verify."""
    keys = {}
    for i, r in enumerate(selected, 1):
        r["key"] = f"P{i}"
        keys[r["key"]] = r.get("citation") or r["document"]
    return keys


def passages_per_paper(query: str, selected: list[dict], per_paper: int) -> dict:
    """One restricted hybrid search per shortlisted paper, so every paper on
    the shortlist is read — the global top-k left four of six unread."""
    collection, bm25, texts, metadatas = load_search_resources()
    out = {}
    for r in selected:
        hits = hybrid_search(
            query, collection, bm25, texts, metadatas,
            top_k=per_paper, doc_filter={r["document"]},
            exclude_types={"figure_description"},
        )
        for h in hits:
            h["key"] = r["key"]
        out[r["key"]] = hits
    return out


def format_passages(hits: list[dict], max_chars: int) -> str:
    parts, total = [], 0
    for h in hits:
        m = h.get("metadata") or {}
        block = f"(p.{m.get('page_first', m.get('page', '?'))}, {m.get('section') or 'text'}) {h.get('text', '')}"
        room = max_chars - total
        if room <= 0:
            break
        block = block[:room]
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_retrieve.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/retrieve.py tests/test_retrieve.py
git commit -m "feat(retrieve): de-duplicated keyed shortlist and one restricted search per shortlisted paper"
```

---

### Task 4: Notes, synthesis, key checking, and the new `research_answer()`

**Files:**
- Modify: `research_assistant/prompts.py` (three prompts), `research_assistant/config.py` (synthesis settings), `research_assistant/shared/retrieve.py` (`paper_notes`, `synthesise`, `check_citation_keys`, `research_answer`; `deep_search` kept for callers)
- Test: `tests/test_retrieve.py` (extend)

**Interfaces:**
- Produces:
  ```python
  # config
  SYNTHESIS_MODE = os.environ.get("CITATION_SYNTHESIS_MODE", "map_reduce")
  SYNTHESIS_PER_PAPER_CHUNKS = _env_int("CITATION_SYNTHESIS_PER_PAPER_CHUNKS", 4)
  SYNTHESIS_PER_PAPER_MAX_CHARS = _env_int("CITATION_SYNTHESIS_PER_PAPER_MAX_CHARS", 5000)
  SYNTHESIS_TEMPERATURE = float(os.environ.get("CITATION_SYNTHESIS_TEMPERATURE", "0.2"))
  SYNTHESIS_OLLAMA_OPTIONS = {"num_ctx": 8192}
  # retrieve
  def paper_notes(query, key, title, hits) -> tuple[str, bool]          # (notes text, relevant)
  def synthesise(query, material: str, n: int) -> str
  def check_citation_keys(text, valid_keys) -> tuple[list[str], list[str]]   # (used in order, unknown)
  def research_answer(query, top_k=None, mode=None, on_progress=None) -> dict | None
  ```
  Return dict: today's `suggestion, citations, passages, selected` **plus** `mode, keys, notes, unverified_citations, irrelevant_cited, timings`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_retrieve.py`:

```python


class TestCitationKeys(unittest.TestCase):
    def test_used_in_order_and_unknown(self):
        used, unknown = rt.check_citation_keys("A [P2]. B [P1, P3]. C [P2] and [P9].", {"P1", "P2", "P3"})
        self.assertEqual(used, ["P2", "P1", "P3", "P9"])
        self.assertEqual(unknown, ["P9"])

    def test_no_keys(self):
        self.assertEqual(rt.check_citation_keys("no citations here", {"P1"}), ([], []))


class ResearchAnswerTestCase(unittest.TestCase):
    """Stubs: stage 1 returns three papers, retrieval returns one chunk per
    paper, the model returns canned notes and a canned synthesis."""

    def setUp(self):
        self.ranked = _ranked(3)
        p = patch.object(rt, "rank_documents", return_value=[dict(r) for r in self.ranked]); p.start(); self.addCleanup(p.stop)
        p = patch.object(rt, "DOC_GATE", False); p.start(); self.addCleanup(p.stop)
        p = patch.object(rt, "load_search_resources", return_value=(None, None, [], [])); p.start(); self.addCleanup(p.stop)

        def fake_search(query, collection, bm25, texts, metadatas, top_k, doc_filter=None, exclude_types=None, **kw):
            doc = next(iter(doc_filter))
            return [{"chunk_index": 1, "text": f"chunk of {doc}", "metadata": {"document": doc, "citation_source": f"Paper {doc[1]}", "page": 1}}]
        p = patch.object(rt, "hybrid_search", side_effect=fake_search); p.start(); self.addCleanup(p.stop)
        self.calls = []

        def fake_chat(messages, model=None, images=None, temperature=None, options=None):
            content = messages[-1]["content"]
            self.calls.append({"content": content, "temperature": temperature, "options": options})
            if "Paper P2" in content and "Write notes" in content:
                return _reply("Not relevant: it is about something else.")
            if "Write notes" in content:
                key = content.split("Paper ")[1].split(":")[0]
                return _reply(f"Establishes: result of {key}. Method: simulation. Limits: none stated.")
            return _reply("### What is established\nX [P1]. Y [P3].\n### Where the papers differ\nNone.\n### The gap\nZ [P1, P7].")
        p = patch.object(rt, "chat", side_effect=fake_chat); p.start(); self.addCleanup(p.stop)


class TestMapReduce(ResearchAnswerTestCase):
    def test_reads_every_paper_then_synthesises(self):
        events = []
        out = rt.research_answer("idea", mode="map_reduce", on_progress=lambda stage, payload: events.append(stage))
        self.assertEqual(out["mode"], "map_reduce")
        self.assertEqual(len(self.calls), 4)                                  # 3 notes + 1 synthesis
        self.assertTrue(all(c["temperature"] == rt.SYNTHESIS_TEMPERATURE for c in self.calls))
        self.assertEqual(self.calls[-1]["options"], rt.SYNTHESIS_OLLAMA_OPTIONS)
        self.assertEqual(self.calls[0]["options"], rt.CHAT_OLLAMA_OPTIONS)
        self.assertEqual([n["key"] for n in out["notes"]], ["P1", "P2", "P3"])
        self.assertFalse(out["notes"][1]["relevant"])
        self.assertEqual(out["keys"], {"P1": "Paper 1", "P2": "Paper 2", "P3": "Paper 3"})
        self.assertEqual(out["citations"], ["Paper 1", "Paper 3"])
        self.assertEqual(out["unverified_citations"], ["P7"])
        self.assertEqual(out["irrelevant_cited"], [])
        self.assertEqual(len(out["passages"]), 3)
        self.assertEqual(events, ["shortlist", "notes", "notes", "notes", "synthesis"])
        for k in ("gate", "map", "reduce", "total"):
            self.assertIn(k, out["timings"])
        self.assertTrue(out["suggestion"].startswith("### What is established"))
        self.assertEqual([s["key"] for s in out["selected"]], ["P1", "P2", "P3"])

    def test_notes_call_carries_the_papers_passages_and_key(self):
        rt.research_answer("idea", mode="map_reduce")
        first = self.calls[0]["content"]
        self.assertIn("Paper P1: Paper 1", first); self.assertIn("chunk of d1.pdf", first); self.assertNotIn("chunk of d2.pdf", first)

    def test_irrelevant_paper_that_is_cited_is_flagged(self):
        def synth_cites_p2(messages, **kw):
            content = messages[-1]["content"]
            if "Write notes" in content:
                return _reply("Not relevant: x.") if "Paper P2" in content else _reply("Establishes: r.")
            return _reply("### What is established\nX [P2].\n### Where the papers differ\nNone.\n### The gap\nG.")
        with patch.object(rt, "chat", side_effect=synth_cites_p2):
            out = rt.research_answer("idea", mode="map_reduce")
        self.assertEqual(out["irrelevant_cited"], ["P2"])

    def test_a_failed_notes_call_uses_the_summary(self):
        def flaky(messages, **kw):
            content = messages[-1]["content"]
            if "Paper P1" in content and "Write notes" in content:
                raise RuntimeError("down")
            if "Write notes" in content:
                return _reply("Establishes: r.")
            return _reply("### What is established\nX [P1].\n### Where the papers differ\nNone.\n### The gap\nG.")
        with patch.object(rt, "chat", side_effect=flaky):
            out = rt.research_answer("idea", mode="map_reduce")
        self.assertEqual(out["notes"][0]["notes"], "summary 1")
        self.assertTrue(out["notes"][0]["relevant"])
        self.assertEqual(out["notes"][0]["passages_used"], 0)


class TestSingle(ResearchAnswerTestCase):
    def test_one_call_over_per_paper_passages(self):
        out = rt.research_answer("idea", mode="single")
        self.assertEqual(out["mode"], "single")
        self.assertEqual(len(self.calls), 1)
        content = self.calls[0]["content"]
        for doc in ("d1.pdf", "d2.pdf", "d3.pdf"):
            self.assertIn(f"chunk of {doc}", content)
        self.assertIn("[P2]", content)
        self.assertEqual(out["notes"], [])
        self.assertEqual(out["citations"], ["Paper 1", "Paper 3"])

    def test_empty_corpus_returns_none(self):
        with patch.object(rt, "rank_documents", return_value=[]):
            self.assertIsNone(rt.research_answer("idea"))

    def test_deep_search_still_exists_for_other_callers(self):
        with patch.object(rt, "hybrid_search", return_value=[]) as hs:
            rt.deep_search("q", ["d1.pdf"], top_k=2)
        self.assertEqual(hs.call_args.kwargs["doc_filter"], {"d1.pdf"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_retrieve.py -q -k "CitationKeys or MapReduce or Single"`
Expected: FAIL — `AttributeError: … 'check_citation_keys'`.

- [ ] **Step 3: Prompts**

In `research_assistant/prompts.py`, after `RELATED_WORK_USER`:

```python
# ─── Synthesis (Tab 1 / orchestrate respond) ────────────────────────────────
# Map: one call per shortlisted paper over its own passages. Reduce: one call
# over the notes. Both cite by short key ([P3]); retrieve.check_citation_keys
# verifies every key against the shortlist afterwards.
SYNTHESIS_SYSTEM = (
    "You are a physicist writing the related-work section of a research "
    "proposal. You write from the material you are given and nothing else. "
    "Every factual sentence carries at least one citation key in square "
    "brackets, e.g. [P2] or [P1, P4]. You never cite a key for something its "
    "material does not say, and when the material does not cover something "
    "you say so instead of filling the gap from memory."
)

PAPER_NOTES_USER = (
    "Research idea: {query}\n\n"
    "Paper {key}: {title}\n"
    "Passages from this paper:\n{passages}\n\n"
    "Write notes on this paper for the idea above, at most 150 words, as "
    "three short labelled parts:\n"
    "Establishes: the specific result(s) in the passages that bear on the "
    "idea — with the system, conditions and numbers when given.\n"
    "Method: how (experiment, simulation, theory; the setup or model).\n"
    "Limits: a stated limitation, assumption or open question, or 'none stated'.\n"
    "If the passages do not bear on the idea at all, write only: "
    "Not relevant: <one sentence why>.\n"
    "Refer to the paper as {key}. Use only the passages."
)

SYNTHESIS_USER = (
    "Research idea: {query}\n\n"
    "Material on {n} papers (each cited by its key):\n{material}\n\n"
    "Write a related-work synthesis of 350-500 words with exactly these headings:\n"
    "### What is established\n"
    "Group by theme. Every sentence cites the keys it rests on.\n"
    "### Where the papers differ\n"
    "Conditions, systems, magnitudes or conclusions that disagree or do not "
    "overlap — cite both sides. If none, say so in one sentence.\n"
    "### The gap\n"
    "State concretely what none of the material covers that the idea needs — "
    "the system, regime, quantity or comparison — and what evidence would "
    "close it. Do not describe the idea's value in general terms.\n"
    "Use only the material. A paper marked 'Not relevant' is not cited."
)
```

- [ ] **Step 4: Config**

In `research_assistant/config.py`, after the `GATE_BATCHED` line:

```python
# ─── Synthesis (Tab 1 related work) ───────────────────────────────────────────
# map_reduce: per-paper notes, then one synthesis over the notes (1 + N + 1
# calls). single: one call over per-paper passages. Both at a low
# temperature — the model's default of 1.0 is for creative writing, not for a
# grounded overview. The reduce runs at 8k context (as the judge does) so
# eight papers' notes plus a 600-token answer fit.
SYNTHESIS_MODE                = os.environ.get("CITATION_SYNTHESIS_MODE", "map_reduce").lower()
SYNTHESIS_PER_PAPER_CHUNKS    = _env_int("CITATION_SYNTHESIS_PER_PAPER_CHUNKS", 4)
SYNTHESIS_PER_PAPER_MAX_CHARS = _env_int("CITATION_SYNTHESIS_PER_PAPER_MAX_CHARS", 5000)
SYNTHESIS_TEMPERATURE         = float(os.environ.get("CITATION_SYNTHESIS_TEMPERATURE", "0.2"))
SYNTHESIS_OLLAMA_OPTIONS      = {"num_ctx": 8192}
```

- [ ] **Step 5: `retrieve.py` — imports, notes, synthesis, keys, `research_answer`**

Extend the config import in `retrieve.py` with `CHAT_OLLAMA_OPTIONS, SYNTHESIS_MODE, SYNTHESIS_PER_PAPER_CHUNKS, SYNTHESIS_PER_PAPER_MAX_CHARS, SYNTHESIS_TEMPERATURE, SYNTHESIS_OLLAMA_OPTIONS` and the prompts import with `SYNTHESIS_SYSTEM, PAPER_NOTES_USER, SYNTHESIS_USER`.

Keep `deep_search` as it is (other callers may use it). Replace `research_answer` entirely with:

```python
# ─── Map: notes per paper ───────────────────────────────────────────────────


def paper_notes(query: str, key: str, title: str, hits: list[dict]):
    """One call over one paper's passages. Returns (notes, relevant)."""
    passages = format_passages(hits, SYNTHESIS_PER_PAPER_MAX_CHARS)
    text = chat(
        [{"role": "system", "content": SYNTHESIS_SYSTEM},
         {"role": "user", "content": PAPER_NOTES_USER.format(query=query, key=key, title=title, passages=passages)}],
        temperature=SYNTHESIS_TEMPERATURE, options=CHAT_OLLAMA_OPTIONS,
    ).content.strip()
    relevant = not text.lower().startswith("not relevant")
    return text, relevant


# ─── Reduce: the synthesis ──────────────────────────────────────────────────


def synthesise(query: str, material: str, n: int) -> str:
    return chat(
        [{"role": "system", "content": SYNTHESIS_SYSTEM},
         {"role": "user", "content": SYNTHESIS_USER.format(query=query, n=n, material=material)}],
        temperature=SYNTHESIS_TEMPERATURE, options=SYNTHESIS_OLLAMA_OPTIONS,
    ).content.strip()


_KEY_GROUP_RE = re.compile(r"\[(P\d+(?:\s*,\s*P\d+)*)\]")


def check_citation_keys(text: str, valid_keys):
    """Keys the synthesis cites, in first-use order, and the ones that are
    not on the shortlist — a 5B model can invent a [P7] as easily as a fact."""
    used = []
    for m in _KEY_GROUP_RE.finditer(text or ""):
        for k in re.split(r"\s*,\s*", m.group(1)):
            if k not in used:
                used.append(k)
    unknown = [k for k in used if k not in valid_keys]
    return used, unknown


# ─── End to end ─────────────────────────────────────────────────────────────


def research_answer(query: str, top_k: int = DEFAULT_TOP_K, mode: str | None = None, on_progress=None) -> dict | None:
    """Shortlist papers → read each one → synthesise, citing by key.

    Returns None when the corpus is empty. Otherwise a dict shaped as before
    (``suggestion`` / ``citations`` / ``passages`` / ``selected``) plus
    ``mode``, ``keys`` (key → title), ``notes`` (map_reduce only),
    ``unverified_citations`` (keys not on the shortlist), ``irrelevant_cited``
    and ``timings``. ``on_progress(stage, payload)`` is called with
    ``shortlist``, ``notes`` (once per paper) and ``synthesis``.
    """
    mode = (mode or SYNTHESIS_MODE).lower()
    if mode not in ("map_reduce", "single"):
        logger.warning("Unknown synthesis mode %r — using map_reduce.", mode)
        mode = "map_reduce"

    def emit(stage, payload):
        if on_progress:
            try:
                on_progress(stage, payload)
            except Exception as exc:                # noqa: BLE001 — progress must never break the answer
                logger.debug("on_progress raised: %s", exc)

    t_start = time.perf_counter()
    ranked = rank_documents(query)
    if not ranked:
        return None

    t0 = time.perf_counter()
    selected = gate_documents(query, ranked) if DOC_GATE else ranked
    selected = dedupe_shortlist(selected)
    keys = assign_keys(selected)
    t_gate = time.perf_counter() - t0
    emit("shortlist", {"papers": [{"key": r["key"], "citation": keys[r["key"]]} for r in selected]})

    per_paper = SYNTHESIS_PER_PAPER_CHUNKS if mode == "map_reduce" else 2
    try:
        passages = passages_per_paper(query, selected, per_paper)
    except Exception as e:
        logger.warning("Per-paper retrieval unavailable (%s) — using summaries only.", e)
        passages = {}
    all_hits = [h for r in selected for h in passages.get(r["key"], [])]

    notes, t_map = [], 0.0
    if mode == "map_reduce":
        t0 = time.perf_counter()
        for r in selected:
            key, title = r["key"], keys[r["key"]]
            hits = passages.get(key) or []
            t1 = time.perf_counter()
            if hits:
                try:
                    text, relevant = paper_notes(query, key, title, hits)
                except Exception as e:
                    logger.warning("Notes failed for %s (%s) — using its summary.", key, e)
                    text, relevant, hits = r.get("summary", ""), True, []
            else:
                text, relevant = r.get("summary", ""), True
            note = {"key": key, "document": r["document"], "citation": title, "notes": text,
                    "relevant": relevant, "passages_used": len(hits), "seconds": round(time.perf_counter() - t1, 1)}
            notes.append(note)
            emit("notes", note)
        t_map = time.perf_counter() - t0
        material = "\n\n".join(f"[{n['key']}] {n['citation']}\n{n['notes']}" for n in notes)
    else:
        blocks = []
        for r in selected:
            hits = passages.get(r["key"]) or []
            body = format_passages(hits, SYNTHESIS_PER_PAPER_MAX_CHARS) if hits else r.get("summary", "")
            blocks.append(f"[{r['key']}] {keys[r['key']]}\n{body}")
        material = "\n\n".join(blocks)

    t0 = time.perf_counter()
    answer = synthesise(query, material, len(selected))
    t_reduce = time.perf_counter() - t0
    emit("synthesis", {"seconds": round(t_reduce, 1)})

    used, unknown = check_citation_keys(answer, set(keys))
    irrelevant = {n["key"] for n in notes if not n["relevant"]}
    citations = [keys[k] for k in used if k in keys]

    return {
        "suggestion": answer,
        "citations": citations,
        "passages": all_hits,
        "selected": selected,
        "mode": mode,
        "keys": keys,
        "notes": notes,
        "unverified_citations": unknown,
        "irrelevant_cited": [k for k in used if k in irrelevant],
        "timings": {"gate": round(t_gate, 1), "map": round(t_map, 1), "reduce": round(t_reduce, 1),
                    "total": round(time.perf_counter() - t_start, 1)},
    }
```

`RESEARCH_CHAT_SYSTEM` and `RELATED_WORK_USER` are no longer used by `retrieve.py`; leave them in `prompts.py` (Agent 7 uses the first) and drop them from `retrieve.py`'s import if unused.

- [ ] **Step 6: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_retrieve.py tests/test_config_version.py -q` — Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add research_assistant/prompts.py research_assistant/config.py research_assistant/shared/retrieve.py tests/test_retrieve.py
git commit -m "feat(synthesis): map-reduce over every shortlisted paper — notes, argued synthesis, checked citation keys, explicit temperature and context"
```

---

### Task 5: The app shows the notes, the legend, the warnings, and progress

**Files:**
- Modify: `app.py` (`_render_build`, the batch-upload synthesis call ~line 666)
- Test: `tests/test_app_names.py` (existing guard must still pass); `tests/test_app_render.py` (new, pure helper)

**Interfaces:**
- Produces: `app._synthesis_warnings(answer: dict) -> list[str]` (pure; returns the warning strings for unverified / irrelevant keys, empty when clean); `_render_notes(answer)`; `_render_synthesis(answer, heading)` used in both places `_render_build` renders an answer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app_render.py
"""The one pure piece of Tab 1's synthesis rendering: which warnings the
reader sees. Everything else in app.py is Streamlit calls."""

import unittest

import app


class TestSynthesisWarnings(unittest.TestCase):
    def test_clean_answer_has_no_warnings(self):
        self.assertEqual(app._synthesis_warnings({"unverified_citations": [], "irrelevant_cited": []}), [])

    def test_unverified_and_irrelevant_keys_are_named(self):
        w = app._synthesis_warnings({"unverified_citations": ["P7"], "irrelevant_cited": ["P2"], "keys": {"P2": "Paper 2"}})
        self.assertEqual(len(w), 2)
        self.assertIn("P7", w[0]); self.assertIn("P2", w[1]); self.assertIn("Paper 2", w[1])

    def test_old_answers_without_the_fields_are_fine(self):
        self.assertEqual(app._synthesis_warnings({"suggestion": "x"}), [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_app_render.py -q` — Expected: FAIL, `AttributeError: module 'app' has no attribute '_synthesis_warnings'`.

- [ ] **Step 3: Implement in `app.py`**

Add after `_render_shortlist`:

```python
def _synthesis_warnings(answer) -> list:
    """What the reader must be told about the synthesis's citations."""
    warnings = []
    unknown = answer.get("unverified_citations") or []
    if unknown:
        warnings.append(f"The synthesis cites {', '.join(unknown)}, which are not on the shortlist — "
                        "treat those sentences as unsupported.")
    irrelevant = answer.get("irrelevant_cited") or []
    if irrelevant:
        keys = answer.get("keys") or {}
        named = ", ".join(f"{k} ({keys.get(k, '?')})" for k in irrelevant)
        warnings.append(f"The synthesis cites {named}, whose notes say the paper is not relevant to the idea.")
    return warnings


def _render_notes(answer):
    notes = answer.get("notes") or []
    if not notes:
        return
    st.markdown(f"**What each paper says** — {len(notes)} paper(s) read")
    for n in notes:
        flag = "" if n.get("relevant", True) else " · not relevant"
        with st.expander(f"{n['key']} · {n['citation']}{flag}  ·  {n.get('passages_used', 0)} passage(s), {n.get('seconds', '?')}s"):
            st.write(n["notes"])


def _render_synthesis(answer, heading):
    for w in _synthesis_warnings(answer):
        st.warning(w, icon="⚠️")
    with st.container(border=True):
        st.markdown(f"###### {heading}")
        st.markdown(answer["suggestion"])
    keys = answer.get("keys") or {}
    if keys:
        st.caption("Keys: " + " · ".join(f"{k} = {t}" for k, t in keys.items()))
    elif answer.get("citations"):
        st.caption("Sources: " + " · ".join(str(c) for c in answer["citations"]))
    t = answer.get("timings")
    if t:
        st.caption(f"{answer.get('mode', '')}: gate {t['gate']}s · notes {t['map']}s · synthesis {t['reduce']}s · total {t['total']}s")
```

In `_render_build`, replace the two answer-rendering blocks. The batch-upload one:

```python
        answer = final.get("answer")
        if answer:
            _render_shortlist(answer.get("selected"))
            with st.container(border=True):
                st.markdown("###### Related work across collection")
                st.markdown(answer["suggestion"])
            if answer.get("citations"):
                st.caption("Sources: " + " · ".join(str(c) for c in answer["citations"]))
            _render_passages(answer.get("passages") or [])
```
becomes
```python
        answer = final.get("answer")
        if answer:
            _render_shortlist(answer.get("selected"))
            _render_notes(answer)
            _render_synthesis(answer, "Related work across collection")
            _render_passages(answer.get("passages") or [])
```
and the pipeline one:
```python
    answer = final.get("answer")
    if answer:
        _render_shortlist(answer.get("selected"))
        with st.container(border=True):
            st.markdown("###### Related work")
            st.markdown(answer["suggestion"])
        if answer.get("citations"):
            st.caption("Sources: " + " · ".join(str(c) for c in answer["citations"]))
        _render_passages(answer.get("passages") or [])
```
becomes
```python
    answer = final.get("answer")
    if answer:
        _render_shortlist(answer.get("selected"))
        _render_notes(answer)
        _render_synthesis(answer, "Related work")
        _render_passages(answer.get("passages") or [])
```

Progress — in the batch-upload path, the call

```python
                            try:
                                answer = retrieve.research_answer(effective_q)
```
becomes
```python
                            def _progress(stage, payload):
                                if stage == "shortlist":
                                    st.write("📚 Reading " + ", ".join(f"{p['key']} {p['citation'][:50]}" for p in payload["papers"]))
                                elif stage == "notes":
                                    mark = "📝" if payload.get("relevant", True) else "➖"
                                    st.write(f"{mark} {payload['key']} · {payload['citation'][:60]} ({payload.get('seconds', '?')}s)")
                                elif stage == "synthesis":
                                    st.write(f"🧠 Synthesis written ({payload.get('seconds', '?')}s)")

                            try:
                                answer = retrieve.research_answer(effective_q, on_progress=_progress)
```

(`from research_assistant.shared import retrieve` is a lazy import a few lines above that call — **keep it**.)

- [ ] **Step 4: Run tests**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_app_render.py tests/test_app_names.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_render.py
git commit -m "feat(app): Tab 1 shows per-paper notes, the key legend, citation warnings and per-paper progress"
```

---

### Task 6: Docs

**Files:**
- Modify: `ARCHITECTURE.md` (§5.4 rewrite + §5.5), `PIPELINE.md` (two-stage retrieval row, config notes), `README.md` (one bullet)

- [ ] **Step 1: `ARCHITECTURE.md`** — replace §5.4 "Two outputs from one corpus" with:

```markdown
### 5.4 Two outputs from one corpus

- **Related-work synthesis** (`research_answer`, the "Research a topic"
  flow): shortlist → read every shortlisted paper → synthesise (§5.5).
- **Citation insertion** (`agent4_assistant.suggest_citation`, the "Cite a
  draft" flow): flat hybrid search → rewrite the sentence with `\cite{key}`.

**Why two.** A literature overview wants breadth across the shortlist; citing
one sentence wants the single best-matching passage. The citation flow stays
flat because narrowing to a summary shortlist first would over-constrain a
single-sentence lookup.

### 5.5 The synthesis reads every paper it shortlists, then argues

Stage 2 used to return the global top-k chunks across the shortlist; on a
six-paper shortlist, six chunks came from two papers and the other four —
one of them the review the query most needed — were never cited. Now each
shortlisted paper (de-duplicated by title, keyed `P1`…`Pn`) gets its own
restricted search, and in `map_reduce` mode (the default) its own model
call producing ≤150-word notes — *Establishes / Method / Limits*, or *Not
relevant* — before one synthesis call over the notes writes **What is
established**, **Where the papers differ**, and **The gap**, citing by key.
`single` mode is one call over the per-paper passages with the same
headings. Both run at temperature 0.2 with an explicit context (4k for
notes, 8k for the synthesis); the model's default was 1.0.

**Why keys.** `[P3]` is what a 5 B model reproduces exactly and what code
can check: `check_citation_keys` lists every key the synthesis used and
flags any not on the shortlist, or belonging to a paper whose notes said
*Not relevant*; the UI shows both. Invented content under a real key is the
judge's job, downstream.

**Cost.** `1 + N + 1` calls; ~6 minutes for six papers on the CPU machine
versus ~3 before. The batched gate (§5.1, now one call) and the per-process
search-resource cache pay for part of it. Progress is shown per paper.
```

and in §5.1 replace the *Trade-off* paragraph with: *"`DOC_SELECT_K` (default 6) summaries are gated in one batched call (`CITATION_GATE_BATCHED=1`); a reply that cannot be aligned to the summaries falls back to one call per summary. `CITATION_DOC_GATE=0` turns the gate off."*

- [ ] **Step 2: `PIPELINE.md`** — in the "Two-stage retrieval" row, replace the last sentence with: *"`research_answer()` then reads every shortlisted paper — its own restricted search and, in `map_reduce` mode, its own notes — and synthesises what is established, where the papers differ, and the gap, citing by checkable keys."* In configuration notes add `CITATION_SYNTHESIS_MODE`, `CITATION_SYNTHESIS_PER_PAPER_CHUNKS`, `CITATION_SYNTHESIS_PER_PAPER_MAX_CHARS`, `CITATION_SYNTHESIS_TEMPERATURE`, `CITATION_GATE_BATCHED`.

- [ ] **Step 3: `README.md`** — in "What he's worst at", replace the third bullet (*He answers short by design…*) with:

```markdown
- **He used to answer short by design.** The synthesis now reads every paper
  it shortlists and is asked for the argument — what's established, where the
  papers disagree, the gap — rather than a headline list. Whether the gap it
  names is the right one is still yours to judge.
```

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md PIPELINE.md README.md
git commit -m "docs: synthesis depth — per-paper reading, map-reduce, checked keys; batched gate and search cache"
```

---

### Task 7: Live comparison on the v2 index — `single` vs `map_reduce`, three ideas

Needs Ollama (`gemma4:e2b`, `nomic-embed-text`) and the v2 index; nothing else using Ollama.

- [ ] **Step 1: The comparison script** (kept)

```python
# scripts/synth_compare.py
"""Run research_answer in both modes on the same ideas and report the
code-checkable depth criteria (spec §6). Live; not a unit test.

    PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/synth_compare.py "idea one" "idea two"
"""
import json
import os
import re
import sys
import time

from research_assistant.config import DATA_DIR
from research_assistant.shared import retrieve
from research_assistant.shared.atomic import atomic_write_json

OUT_DIR = os.path.join(DATA_DIR, "eval", "synthesis")
HEADINGS = ("### What is established", "### Where the papers differ", "### The gap")


def criteria(out):
    text = out["suggestion"]
    used, _ = retrieve.check_citation_keys(text, set(out["keys"]))
    return {
        "mode": out["mode"], "papers_shortlisted": len(out["selected"]), "papers_cited": len([k for k in used if k in out["keys"]]),
        "words": len(text.split()), "headings_present": all(h in text for h in HEADINGS),
        "unverified": out["unverified_citations"], "irrelevant_cited": out["irrelevant_cited"],
        "not_relevant_notes": sum(1 for n in out["notes"] if not n["relevant"]), "timings": out["timings"],
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for idea in sys.argv[1:]:
        for mode in ("single", "map_reduce"):
            out = retrieve.research_answer(idea, mode=mode)
            c = criteria(out)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(OUT_DIR, f"{stamp}-{mode}-{re.sub(r'[^a-z0-9]+', '-', idea.lower())[:40]}.json")
            atomic_write_json(path, {"idea": idea, "criteria": c, "answer": out}, default=str)
            print(f"\n=== {mode} · {idea}\n{json.dumps(c, indent=1)}\n--- synthesis ---\n{out['suggestion']}\n(saved {path})")


if __name__ == "__main__":
    main()
```

(`atomic_write_json(path, data, **dump_kwargs)` forwards keyword arguments to `json.dump`, so `default=str` is accepted.)

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/synth_compare.py \
  "Anderson localization and transport in disordered quasi-one-dimensional wires" \
  "hopping transport in covalent MoS2 networks and the role of the molecular linker" \
  "analytic continuation of imaginary-time Green's functions from quantum Monte Carlo data"
```

Expect ~3 min per `single` and ~6 min per `map_reduce` run — roughly 30 minutes total.

- [ ] **Step 3: Gates — per idea, `map_reduce` must satisfy all of:**

- `papers_cited ≥ 4` when `papers_shortlisted ≥ 5` (the "before" was 2 of 6), and `papers_cited ≥ papers_shortlisted − not_relevant_notes − 1` otherwise;
- `headings_present` is true;
- `unverified == []` and `irrelevant_cited == []` (if a run violates this, the warning path is what protects the reader — record it, do not hide it);
- `350 ≤ words ≤ 600`;
- the **The gap** section names a specific system/regime/quantity, not "your research could add value" — read it.

And for the record: `single` on the same ideas, with the same criteria, so the reader sees what the extra minutes buy.

- [ ] **Step 4: The app**

`CITATION_INDEX_VERSION=2 streamlit run app.py --server.port 8601`; Tab 1 → *Upload research paper(s)*; upload one PDF from `data/pulled_pdfs/` (e.g. `doi_10.1002_adma.202211157.pdf`) with a research topic; watch the status show `📚 Reading …`, one `📝` line per paper, `🧠 Synthesis written`; then the notes expanders, the synthesis with the three headings, the key legend, and the timings caption. Screenshot or describe.

- [ ] **Step 5: Suite**

`CITATION_LOG_FILE=0 python -m pytest tests/ -q` (deselect `tests/test_ingestion.py` if the ingest lock is held). Expected: green.

- [ ] **Step 6: Commit**

```bash
git add scripts/synth_compare.py
git commit -F - <<'EOF'
chore(synthesis): live comparison single vs map_reduce on three ideas

<paste the three criteria blocks per mode and, for one idea, both syntheses in full>
app: <one line on what Tab 1 showed>
suite: <last line>
EOF
```

---

## Self-review

**Spec coverage.** §2A → Task 3 (`dedupe_shortlist`, `passages_per_paper`). §2B → Task 4 (both modes; `map_reduce` default in config; per-paper chunk counts; temperature and options on every call; 8k reduce). §2C → Task 4 (`check_citation_keys`, `keys`, `citations`, `unverified_citations`, `irrelevant_cited`) + Task 5 (legend, warnings). §2D → Task 2. §2E → Task 1. §2F → Task 4 (`on_progress`, timings, return shape) + Task 5 (progress lines, notes). §3 not-changed list honoured: Agent 7, reranker, neighbours, `orchestrate.respond` untouched (it calls `research_answer(query)` and gets the default mode). §4 config names match Task 2/4. §6 → Task 7.

**Type consistency.** `assign_keys` sets `r["key"]` that `passages_per_paper`, the notes loop, `single`'s material block, and `_render_shortlist`-adjacent `_render_notes` all read. `research_answer`'s return keys are exactly what `criteria()` in Task 7 and `_synthesis_warnings` / `_render_synthesis` in Task 5 read. `on_progress(stage, payload)` payload shapes: `shortlist → {"papers": [{key, citation}]}`, `notes → the note dict`, `synthesis → {"seconds"}` — matched in Task 5's `_progress`.

**Placeholders.** None. Task 7's commit asks for pasted criteria and syntheses.

**Validated before writing.** The retrieve/db/prompts/config edits were assembled into a scratch copy of the package and the plan's tests run against them: 30/30 (gate parser and fallback, shortlist de-dup and keys, per-paper retrieval, key checking, both `research_answer` modes with a stubbed model, the search-resource cache).
