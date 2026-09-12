# Research Chat Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Research chat (Agent 7 / Tab 4) understands follow-ups in context, always sees the paper the conversation is about, fits every turn inside the window by design — folding older turns into a rolling memory instead of losing them — cites keyed sources that are checked, and samples at a chosen temperature.

**Architecture:** A new pure module `shared/chat_context.py` owns the mechanics (token estimate, per-paper cap, merge, keyed context formatting, history fitting, key checking/resolution). `agents/agent7_research_chat.py`'s `ResearchChat` gains condense → focus+global search → budget → memory fold → stream → verify, exposing `last_query`, `last_sources` (keyed, cited flags), `last_warnings`, `last_budget`, `memory`. `shared/llm.chat_stream` gains `temperature`. Tab 4 and the CLI show the query, keyed sources, warnings, memory. Nothing in the index or the other agents changes.

**Tech Stack:** Python 3.10–3.12, pytest collecting `unittest.TestCase`, Ollama `gemma4:e2b` via `shared.llm`, `shared.search.hybrid_search`, Streamlit.

**Spec:** `notimportant/superpowers/specs/2026-09-12-chat-depth-design.md` — §2 A–E are the tasks' authority; §4 the config names.

---

## Context you need before Task 1

### Agent 7 today (`research_assistant/agents/agent7_research_chat.py`)

```python
class ResearchChat:
    def __init__(self, top_k=5, search_resources=None): ...   # loads (collection, bm25, texts, metadatas) unless given
    def _retrieve_context(self, query) -> (context_str, sources)  # hybrid_search(query, ..., top_k) on the RAW question; "--- Source N | Document | Page | Citation ---\n<text>"
    def _generate_stream(self, messages)                          # chat_stream(messages, model=CHAT_MODEL, options=CHAT_OLLAMA_OPTIONS)  ← no temperature
    def stream_turn(self, user_message)                           # yields text; messages = [system] + history + [question + context]; history trimmed to last 10 messages
    def clear_history(self); def export_conversation(self) -> path
```
Module-level imports: `CHAT_MODEL, CHAT_OLLAMA_OPTIONS, DRAFTS_DIR` from config; `RESEARCH_CHAT_SYSTEM` from prompts; `load_search_resources`; `hybrid_search`; `chat_stream`. `SYSTEM_PROMPT = RESEARCH_CHAT_SYSTEM`. A CLI `run_repl(agent)` handles `/clear /sources /export /help`. **There are no tests for this module.**

Tab 4 in `app.py` (lines ~970–1020) creates `ResearchChat(top_k=5, search_resources=cached_res)` in `st.session_state["chat_agent"]`, renders `agent.history`, streams `agent.stream_turn(question)` with `st.write_stream`, and shows `agent.last_sources` (`document` — `citation`) in an expander.

### The "before" (2026-09-12, v2 index, three turns)

| turn | question | retrieved from | prompt (est. tokens / 4,096) | outcome |
|---|---|---|---|---|
| 1 | What does the covalent MoS₂ network paper say about hopping transport? | 5 × the MoS₂ paper | 1,977 | good, 388 words |
| 2 | What about its limitations? | RMP 69 731, nnano.2008.58, Phys. Rep. 1997, ncomms6678 — **not the MoS₂ paper** | 2,273 | "I cannot specify the limitations" |
| 3 | How does that compare with the carbon nanotube network paper? | 5 × the CNT paper | 2,789 | comparison from history only |

Two more turns at this rate exceed `num_ctx` with the answer's ~500 tokens; Ollama then truncates from the front of the prompt (its server log says `truncating input prompt`), where the system message is. Every answer sampled at the model's default temperature 1.0.

### Shapes

- `hybrid_search(query, collection, bm25, texts, metadatas, top_k, doc_filter=set|None, exclude_types=set|None)` → `list[{"chunk_index", "text", "metadata": {"document", "citation_source", "type", "page", …v2: "section", "page_first"}, "rrf_score"}]`.
- `chat(messages, model=None, images=None, temperature=None, options=None) -> ChatResult(content, …)`; `chat_stream(messages, model=None, options=None)` yields strings — **no temperature yet** (Task 1 adds it).
- `CHAT_OLLAMA_OPTIONS = {"num_ctx": 4096}` (config). `GEMINI.md`: keep it at 4,096 on the CPU VM.

### Environment

- Interpreter `/home/shardul/miniconda3/envs/ml/bin/python`; scripts with `PYTHONPATH=.`; tests `CITATION_LOG_FILE=0 python -m pytest tests/ -v`; `unittest.TestCase`. Ollama + the v2 index (`CITATION_INDEX_VERSION=2`) only in Task 7.
- Branch: from `synthesis-depth` if it exists, else the newest of `judge-evaluation` / `judgement-hardening` / `ingestion-v2`: `git checkout <base> && git checkout -b chat-depth`. **Never push, never merge.** Commit after every task.
- `data/ingest.lock` held → deselect `tests/test_ingestion.py`.

### Pitfalls from previous builds on this repo

1. A lazy import deleted while editing the block around it → `NameError` at runtime, invisible to the suite. Keep local imports where you find them (Tab 4 has one: `from research_assistant.agents.agent7_research_chat import ResearchChat`).
2. Production code bent to satisfy a synthetic test. Fix the test and say so instead.
3. Verification notes without the run. Task 7 asks for pasted output.

## Global Constraints

- **Python 3.10–3.12.** No 3.13-only syntax.
- **No provider SDK outside `shared/llm.py`.**
- **`num_ctx` stays 4,096 in config** (`GEMINI.md`); the budget derives from `CHAT_OLLAMA_OPTIONS["num_ctx"]` so it scales if the operator raises it.
- **Every model call in this path passes an explicit `temperature` and `options`.**
- **`figure_description` chunks are never chat context** (`exclude_types={"figure_description"}` on every search).
- **Nothing is dropped from the conversation silently.** Older turns leave the prompt only by being folded into `memory`, and the fold is logged.
- **`ResearchChat.stream_turn` keeps its contract:** a generator of text deltas; `history` remains a list of `{"role","content"}`; `clear_history` / `export_conversation` keep working. Tab 4 and `run_repl` depend on these.
- **`unittest.TestCase`**; live-model tests behind `RUN_LLM_TESTS=1`; run with `CITATION_LOG_FILE=0`.
- **Commit after every task; do not push.**

---

### Task 1: `chat_stream()` takes a temperature

**Files:**
- Modify: `research_assistant/shared/llm.py:137-187`
- Test: `tests/test_llm.py` (extend)

**Interfaces:**
- Produces: `chat_stream(messages, model=None, options=None, temperature=None)`; Ollama gets it via `options["temperature"]` (a copy — never mutates the caller's dict); OpenAI gets `temperature=`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm.py` (the file already defines `_ollama_chunk`, `_openai_chunk`, and patches backends via `sys.modules`; mirror `TestChatStreamOllama`):

```python


class TestChatStreamTemperature(unittest.TestCase):
    def test_ollama_gets_temperature_in_a_copied_options_dict(self):
        fake = MagicMock()
        fake.chat.return_value = iter([_ollama_chunk("x")])
        opts = {"num_ctx": 4096}
        with patch.dict("sys.modules", {"ollama": fake}), patch.object(llm, "LLM_BACKEND", "ollama"):
            list(llm.chat_stream([{"role": "user", "content": "q"}], options=opts, temperature=0.3))
        sent = fake.chat.call_args.kwargs["options"]
        self.assertEqual(sent, {"num_ctx": 4096, "temperature": 0.3})
        self.assertEqual(opts, {"num_ctx": 4096})               # caller's dict untouched

    def test_ollama_without_temperature_sends_options_as_is(self):
        fake = MagicMock()
        fake.chat.return_value = iter([_ollama_chunk("x")])
        with patch.dict("sys.modules", {"ollama": fake}), patch.object(llm, "LLM_BACKEND", "ollama"):
            list(llm.chat_stream([{"role": "user", "content": "q"}], options={"num_ctx": 8192}))
        self.assertEqual(fake.chat.call_args.kwargs["options"], {"num_ctx": 8192})

    def test_openai_gets_temperature(self):
        fake_openai = MagicMock()
        client = fake_openai.OpenAI.return_value
        client.chat.completions.create.return_value = iter([_openai_chunk("x")])
        with patch.dict("sys.modules", {"openai": fake_openai}), patch.object(llm, "LLM_BACKEND", "openai"):
            list(llm.chat_stream([{"role": "user", "content": "q"}], temperature=0.3))
        self.assertEqual(client.chat.completions.create.call_args.kwargs["temperature"], 0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_llm.py -q -k Temperature`
Expected: FAIL — `TypeError: chat_stream() got an unexpected keyword argument 'temperature'`.

- [ ] **Step 3: Implement**

Replace `chat_stream`, `_ollama_stream` and `_openai_stream` with:

```python
def chat_stream(messages, model=None, options=None, temperature=None):
    """Yield content deltas as plain strings.

    Args:
        messages: list of ``{"role", "content"}`` dicts.
        model:    model id; defaults to config.LLM_MODEL.
        options:  ollama runtime options (e.g. CHAT_OLLAMA_OPTIONS). The
                  OpenAI-compatible backend has no equivalent and ignores it.
        temperature: sampling temperature; None leaves the provider default
                  (which for gemma4:e2b's modelfile is 1.0).

    Deliberately not a generator itself, so an unknown backend raises at the
    call rather than on the first ``next()``.
    """
    model = model or LLM_MODEL
    if LLM_BACKEND == "openai":
        return _openai_stream(messages, model, temperature)
    if LLM_BACKEND == "ollama":
        return _ollama_stream(messages, model, options, temperature)
    raise ValueError(
        f"Unknown LLM_BACKEND {LLM_BACKEND!r} (expected 'ollama' or 'openai')"
    )


def _ollama_stream(messages, model, options, temperature=None):
    import ollama

    # Copied, not mutated: callers pass a module-level config dict.
    opts = dict(options) if options else {}
    if temperature is not None:
        opts["temperature"] = temperature
    kwargs = {"model": model, "messages": messages, "stream": True}
    if opts:
        kwargs["options"] = opts
    for chunk in ollama.chat(**kwargs):
        try:
            content = chunk.message.content
        except AttributeError:
            content = (chunk.get("message") or {}).get("content")
        if content:
            yield content


def _openai_stream(messages, model, temperature=None):
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    kwargs = {"model": model, "messages": messages, "stream": True}
    if temperature is not None:
        kwargs["temperature"] = temperature
    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        # Role-only opening frames and usage-only terminal frames carry no text.
        if not getattr(chunk, "choices", None):
            continue
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            yield content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_llm.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/llm.py tests/test_llm.py
git commit -m "feat(llm): chat_stream takes an explicit temperature on both backends"
```

---

### Task 2: `shared/chat_context.py` — the mechanics, pure

**Files:**
- Create: `research_assistant/shared/chat_context.py`
- Test: `tests/test_chat_context.py`

**Interfaces:**
- Produces:
  ```python
  CHARS_PER_TOKEN = 4
  def estimate_tokens(text: str) -> int
  def messages_tokens(messages: list[dict]) -> int
  def cap_per_document(results, per_doc: int) -> list          # order kept; at most per_doc per metadata["document"]
  def merge_results(focus_hits, global_hits) -> list           # focus first; dedupe on (document, chunk_index)
  def format_context(results, max_chars: int) -> tuple[str, list[dict]]
      # "[S1] <citation> — p.<page>, <section>\n<text>" blocks joined by blank lines; blocks past the cap dropped whole
      # sources: [{"key","document","citation","page","section","chunk_index","cited": False}]
  def fit_history(history, max_tokens: int) -> tuple[list, list]   # (kept, dropped) — whole user/assistant pairs, most recent first
  def cited_keys(text: str) -> list[str]                        # [S2], [S1, S3] → in first-use order
  def resolve_keys(text: str, sources) -> str                   # [S2] → [<short title>]; unknown keys left as-is
  def short_title(citation: str, words: int = 6) -> str
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chat_context.py
"""The mechanics of a chat turn, with no model and no index: what fits, what
is kept, how sources are keyed and checked."""

import unittest

from research_assistant.shared import chat_context as cc


def _hit(i, doc, text="t", **meta):
    m = {"document": doc, "citation_source": f"Title of {doc}", "page": 2}
    m.update(meta)
    return {"chunk_index": i, "text": text, "metadata": m, "rrf_score": 0.1}


class TestTokens(unittest.TestCase):
    def test_estimate_rounds_up(self):
        self.assertEqual(cc.estimate_tokens(""), 0)
        self.assertEqual(cc.estimate_tokens("abcd"), 1)
        self.assertEqual(cc.estimate_tokens("abcde"), 2)

    def test_messages_tokens_sums_content(self):
        self.assertEqual(cc.messages_tokens([{"role": "a", "content": "abcd"}, {"role": "b", "content": "abcdefgh"}]), 3)


class TestCapAndMerge(unittest.TestCase):
    def test_cap_per_document_keeps_order_and_limit(self):
        hits = [_hit(1, "a"), _hit(2, "a"), _hit(3, "b"), _hit(4, "a"), _hit(5, "a")]
        self.assertEqual([h["chunk_index"] for h in cc.cap_per_document(hits, 2)], [1, 2, 3])   # a: 1,2 kept, 4,5 over the cap

    def test_merge_puts_focus_first_and_dedupes(self):
        focus = [_hit(9, "a"), _hit(1, "a")]
        glob = [_hit(1, "a"), _hit(3, "b")]
        self.assertEqual([h["chunk_index"] for h in cc.merge_results(focus, glob)], [9, 1, 3])


class TestFormatContext(unittest.TestCase):
    def test_keys_headers_and_sources(self):
        text, sources = cc.format_context([_hit(1, "a.pdf", "AAA", section="results", page_first=4), _hit(2, "b.pdf", "BBB")], max_chars=1000)
        self.assertTrue(text.startswith("[S1] Title of a.pdf — p.4, results\nAAA\n\n[S2] Title of b.pdf — p.2, text\nBBB"))
        self.assertEqual([s["key"] for s in sources], ["S1", "S2"])
        self.assertEqual(sources[0], {"key": "S1", "document": "a.pdf", "citation": "Title of a.pdf", "page": 4,
                                      "section": "results", "chunk_index": 1, "cited": False})

    def test_cap_drops_whole_blocks_from_the_tail(self):
        hits = [_hit(1, "a", "A" * 100), _hit(2, "b", "B" * 100), _hit(3, "c", "C" * 100)]
        text, sources = cc.format_context(hits, max_chars=280)
        self.assertEqual([s["key"] for s in sources], ["S1", "S2"])
        self.assertNotIn("CCC", text)

    def test_first_block_is_never_dropped(self):
        text, sources = cc.format_context([_hit(1, "a", "A" * 500)], max_chars=100)
        self.assertEqual(len(sources), 1)
        self.assertLessEqual(len(text), 100)

    def test_empty(self):
        self.assertEqual(cc.format_context([], 100), ("", []))


class TestFitHistory(unittest.TestCase):
    def _hist(self, n_pairs, chars=40):
        h = []
        for i in range(n_pairs):
            h += [{"role": "user", "content": f"q{i}" + "x" * chars}, {"role": "assistant", "content": f"a{i}" + "y" * chars}]
        return h

    def test_keeps_the_most_recent_whole_pairs(self):
        h = self._hist(4)                       # each pair ≈ 2×42 chars ≈ 22 tokens
        kept, dropped = cc.fit_history(h, max_tokens=50)
        self.assertEqual([m["content"][:2] for m in kept], ["q2", "a2", "q3", "a3"])
        self.assertEqual([m["content"][:2] for m in dropped], ["q0", "a0", "q1", "a1"])

    def test_everything_fits(self):
        h = self._hist(2)
        self.assertEqual(cc.fit_history(h, 10_000), (h, []))

    def test_nothing_fits(self):
        h = self._hist(2)
        self.assertEqual(cc.fit_history(h, 5), ([], h))


class TestKeys(unittest.TestCase):
    def test_cited_keys_in_order(self):
        self.assertEqual(cc.cited_keys("X [S2]. Y [S1, S3]. Z [S2] and [S9]."), ["S2", "S1", "S3", "S9"])

    def test_resolve_keys_uses_short_titles_and_leaves_unknown(self):
        sources = [{"key": "S1", "citation": "Charge transport in covalent MoS2 networks of nanosheets"},
                   {"key": "S2", "citation": "Anderson transitions"}]
        out = cc.resolve_keys("A [S1]. B [S1, S2]. C [S7].", sources)
        self.assertEqual(out, "A [Charge transport in covalent MoS2 networks]. B [Charge transport in covalent MoS2 networks; Anderson transitions]. C [S7].")

    def test_short_title(self):
        self.assertEqual(cc.short_title("one two three four five six seven eight"), "one two three four five six")
        self.assertEqual(cc.short_title(""), "untitled")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_chat_context.py -q` — Expected: FAIL, `ImportError`.

- [ ] **Step 3: Implement**

```python
# research_assistant/shared/chat_context.py
"""The mechanics of a research-chat turn. Pure: no model, no index.

Agent 7 assembles each turn against a token budget derived from num_ctx.
Tokens are estimated at four characters each — this model's tokenizer on
English prose — and the answer reserve carries the margin. Sources are
keyed [S1]… so an answer's citations can be checked against what it was
actually shown, and resolved to titles before the answer enters history.
"""

from __future__ import annotations

import re

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    n = len(text or "")
    return (n + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def messages_tokens(messages) -> int:
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def cap_per_document(results, per_doc: int) -> list:
    counts, out = {}, []
    for r in results:
        doc = (r.get("metadata") or {}).get("document")
        if counts.get(doc, 0) >= per_doc:
            continue
        counts[doc] = counts.get(doc, 0) + 1
        out.append(r)
    return out


def merge_results(focus_hits, global_hits) -> list:
    seen, out = set(), []
    for r in list(focus_hits) + list(global_hits):
        key = ((r.get("metadata") or {}).get("document"), r.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def format_context(results, max_chars: int):
    parts, sources, total = [], [], 0
    for r in results:
        m = r.get("metadata") or {}
        key = f"S{len(sources) + 1}"
        citation = m.get("citation_source") or "Unknown"
        page = m.get("page_first", m.get("page", "?"))
        section = m.get("section") or "text"
        block = f"[{key}] {citation} — p.{page}, {section}\n{r.get('text', '')}"
        if parts and total + len(block) + 2 > max_chars:
            break
        block = block[:max_chars]
        parts.append(block)
        total += len(block) + 2
        sources.append({"key": key, "document": m.get("document"), "citation": citation, "page": page,
                        "section": m.get("section"), "chunk_index": r.get("chunk_index"), "cited": False})
    return "\n\n".join(parts), sources


def fit_history(history, max_tokens: int):
    """Whole (user, assistant) pairs, newest first, until the budget is spent."""
    pairs = [history[i:i + 2] for i in range(0, len(history), 2)]
    kept, used = [], 0
    for pair in reversed(pairs):
        t = sum(estimate_tokens(m.get("content", "")) for m in pair)
        if used + t > max_tokens:
            break
        kept.insert(0, pair)
        used += t
    dropped = pairs[:len(pairs) - len(kept)]
    return [m for p in kept for m in p], [m for p in dropped for m in p]


_KEY_GROUP_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def cited_keys(text: str) -> list:
    used = []
    for m in _KEY_GROUP_RE.finditer(text or ""):
        for k in re.split(r"\s*,\s*", m.group(1)):
            if k not in used:
                used.append(k)
    return used


def short_title(citation: str, words: int = 6) -> str:
    parts = (citation or "").split()
    return " ".join(parts[:words]) if parts else "untitled"


def resolve_keys(text: str, sources) -> str:
    titles = {s["key"]: short_title(s.get("citation", "")) for s in sources}

    def _sub(m):
        keys = re.split(r"\s*,\s*", m.group(1))
        if not all(k in titles for k in keys):
            return m.group(0)
        return "[" + "; ".join(titles[k] for k in keys) + "]"

    return _KEY_GROUP_RE.sub(_sub, text or "")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_chat_context.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/chat_context.py tests/test_chat_context.py
git commit -m "feat(chat): pure turn mechanics — token estimate, per-paper cap, keyed context, history fitting, key checking"
```

---

### Task 3: Prompts and config

**Files:**
- Modify: `research_assistant/prompts.py` (replace `RESEARCH_CHAT_SYSTEM`; add `CHAT_TURN_USER`, `CHAT_TURN_NO_CONTEXT`, `CHAT_CONDENSE_USER`, `CHAT_MEMORY_USER`), `research_assistant/config.py` (chat settings)
- Test: `tests/test_prompts_chat.py` (new, tiny)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts_chat.py
import unittest

from research_assistant import prompts as p
from research_assistant import config as c


class TestChatPrompts(unittest.TestCase):
    def test_placeholders(self):
        self.assertIn("{question}", p.CHAT_TURN_USER); self.assertIn("{context}", p.CHAT_TURN_USER)
        self.assertIn("{question}", p.CHAT_TURN_NO_CONTEXT)
        self.assertIn("{history}", p.CHAT_CONDENSE_USER); self.assertIn("{question}", p.CHAT_CONDENSE_USER)
        self.assertIn("{memory}", p.CHAT_MEMORY_USER); self.assertIn("{turns}", p.CHAT_MEMORY_USER)
        self.assertIn("[S1]", p.RESEARCH_CHAT_SYSTEM)

    def test_config_defaults(self):
        self.assertTrue(c.CHAT_CONDENSE)
        self.assertEqual((c.CHAT_CONTEXT_MAX_CHARS, c.CHAT_ANSWER_RESERVE_TOKENS, c.CHAT_PER_DOC_CAP, c.CHAT_FOCUS_TOP_K), (7000, 700, 3, 2))
        self.assertAlmostEqual(c.CHAT_TEMPERATURE, 0.3)
        self.assertEqual(c.CHAT_OLLAMA_OPTIONS["num_ctx"], 4096)
```

- [ ] **Step 2: Run to verify it fails** — `CITATION_LOG_FILE=0 python -m pytest tests/test_prompts_chat.py -q` → FAIL, `AttributeError`.

- [ ] **Step 3: Prompts** — in `research_assistant/prompts.py`, replace the `RESEARCH_CHAT_SYSTEM = """…"""` block with:

```python
RESEARCH_CHAT_SYSTEM = """\
You are a physicist's research assistant working from a database of papers.
Each turn you receive the researcher's question and numbered source passages
[S1], [S2], … retrieved from that database for this turn.

Answer from the sources:
- Put a source key on every factual sentence, e.g. [S2] or [S1, S3]. Never
  cite a key that is not among this turn's sources.
- Be specific: give the systems, conditions, numbers and mechanisms the
  sources state, not generalities.
- Say plainly what the sources do not cover. If one of the sources looks
  likely to cover it elsewhere in the paper, say which one to ask about next.
- Stay on the researcher's question. No preamble, no restating the question.

Earlier turns of the conversation may be present; use them for continuity,
but cite only this turn's sources."""

CHAT_TURN_USER = "{question}\n\nSources retrieved for this turn:\n{context}"

CHAT_TURN_NO_CONTEXT = (
    "{question}\n\n[No passages were retrieved from the database for this "
    "question. Say so first; then answer from general knowledge if you can, "
    "and mark that answer as not grounded in the database.]"
)

# One short call per follow-up: the question as typed ("what about its
# limitations?") embeds badly; the rewrite names what "its" refers to.
CHAT_CONDENSE_USER = (
    "Recent conversation:\n{history}\n\n"
    "New question: {question}\n\n"
    "Rewrite the new question as a standalone search query for a database of "
    "physics papers. Resolve references like 'it', 'that paper', 'the same "
    "system' using the conversation; keep the specific papers, materials, "
    "quantities and conditions; drop conversational filler. At most 40 words. "
    "Return only the query."
)

# Older turns leave the prompt only by being folded into this.
CHAT_MEMORY_USER = (
    "Summarise these earlier turns of a research conversation in at most 120 "
    "words of plain prose: what was asked, what was established (name the "
    "papers by title), and what remains open. Merge with the previous summary; "
    "keep what is still relevant.\n\n"
    "Previous summary: {memory}\n\n"
    "Earlier turns:\n{turns}"
)
```

- [ ] **Step 4: Config** — in `research_assistant/config.py`, after `CHAT_OLLAMA_OPTIONS = {...}`:

```python
# Research chat (Agent 7). The window is num_ctx above; every turn is
# assembled against a budget derived from it: the answer reserve, then the
# system prompt (+ memory), then retrieved context up to CONTEXT_MAX_CHARS,
# then as many whole recent turns as fit — older ones are folded into a
# rolling memory by a model call, never dropped silently.
CHAT_CONDENSE               = _env_bool("CITATION_CHAT_CONDENSE", True)
CHAT_CONTEXT_MAX_CHARS      = _env_int("CITATION_CHAT_CONTEXT_MAX_CHARS", 7000)
CHAT_ANSWER_RESERVE_TOKENS  = _env_int("CITATION_CHAT_ANSWER_RESERVE_TOKENS", 700)
CHAT_PER_DOC_CAP            = _env_int("CITATION_CHAT_PER_DOC_CAP", 3)
CHAT_FOCUS_TOP_K            = _env_int("CITATION_CHAT_FOCUS_TOP_K", 2)
CHAT_TEMPERATURE            = float(os.environ.get("CITATION_CHAT_TEMPERATURE", "0.3"))
```

- [ ] **Step 5: Run tests** — `CITATION_LOG_FILE=0 python -m pytest tests/test_prompts_chat.py tests/test_config_version.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add research_assistant/prompts.py research_assistant/config.py tests/test_prompts_chat.py
git commit -m "feat(chat): keyed-source system prompt, condense and memory prompts, chat budget settings"
```

---

### Task 4: `ResearchChat` — condense, focus search, budget, memory, verified keys

**Files:**
- Modify: `research_assistant/agents/agent7_research_chat.py` (imports; `ResearchChat`; `run_repl`'s `/sources`, new `/memory`)
- Test: `tests/test_research_chat.py` (new)

**Interfaces:**
- Produces on `ResearchChat`: attributes `memory: str`, `focus_documents: set`, `last_query: str`, `last_sources: list[dict]` (keyed, with `cited`), `last_warnings: list[str]`, `last_budget: dict` (`num_ctx, prompt_tokens_est, reserve, context_chars, history_messages, folded`); methods `_condense(question) -> str`, `_search(query) -> list`, `_fold_memory(dropped_messages) -> None`, `_system_prompt() -> str`; `stream_turn(question)` unchanged in contract. `export_conversation` writes the memory (when any) before the turns.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_research_chat.py
"""One chat turn, with the model and the index stubbed. What is pinned: the
follow-up is condensed and the condensed query is what is searched; focus
papers get their own search and come first; the prompt fits the budget,
with older turns folded into memory rather than dropped; keys are checked
and resolved; temperature and options reach the stream."""

import types
import unittest
from unittest.mock import patch

from research_assistant.agents import agent7_research_chat as a7


def _hit(i, doc, text, page=1):
    return {"chunk_index": i, "text": text, "metadata": {"document": doc, "citation_source": f"Title {doc}", "page": page}, "rrf_score": 0.5}


class ChatTestCase(unittest.TestCase):
    def setUp(self):
        self.searches = []

        def fake_search(query, collection, bm25, texts, metadatas, top_k, doc_filter=None, exclude_types=None, **kw):
            self.searches.append({"query": query, "top_k": top_k, "doc_filter": doc_filter, "exclude_types": exclude_types})
            if doc_filter:
                return [_hit(90 + i, next(iter(doc_filter)), f"focus chunk {i}") for i in range(top_k)]
            return [_hit(1, "a.pdf", "A one"), _hit(2, "a.pdf", "A two"), _hit(3, "a.pdf", "A three"),
                    _hit(4, "a.pdf", "A four"), _hit(5, "b.pdf", "B one"), _hit(6, "c.pdf", "C one")][:top_k]
        p = patch.object(a7, "hybrid_search", side_effect=fake_search); p.start(); self.addCleanup(p.stop)

        self.chats = []

        def fake_chat(messages, model=None, temperature=None, options=None, **kw):
            content = messages[-1]["content"]
            self.chats.append({"content": content, "temperature": temperature, "options": options})
            if "Rewrite the new question" in content:
                return types.SimpleNamespace(content="limitations of the MoS2 covalent network hopping study")
            return types.SimpleNamespace(content="MEMORY: earlier turns summarised")
        p = patch.object(a7, "chat", side_effect=fake_chat); p.start(); self.addCleanup(p.stop)

        self.streams = []
        self.answer = "Hopping dominates [S1]. See also [S2]. Unknown [S9]."

        def fake_stream(messages, model=None, options=None, temperature=None):
            self.streams.append({"messages": messages, "temperature": temperature, "options": options})
            yield from [self.answer[:10], self.answer[10:]]
        p = patch.object(a7, "chat_stream", side_effect=fake_stream); p.start(); self.addCleanup(p.stop)

        self.agent = a7.ResearchChat(top_k=5, search_resources=(None, None, [], []))

    def _turn(self, q):
        return "".join(self.agent.stream_turn(q))


class TestFirstTurn(ChatTestCase):
    def test_no_condense_global_search_capped_per_paper(self):
        self._turn("What does the MoS2 paper say?")
        self.assertEqual([c for c in self.chats if "Rewrite" in c["content"]], [])
        self.assertEqual(self.searches[0]["top_k"], 10)                       # top_k × 2
        self.assertEqual(self.searches[0]["exclude_types"], {"figure_description"})
        docs = [s["document"] for s in self.agent.last_sources]
        self.assertEqual(docs, ["a.pdf", "a.pdf", "a.pdf", "b.pdf", "c.pdf"])   # cap 3 per paper, then top_k 5

    def test_keys_are_checked_and_focus_is_set(self):
        self._turn("q")
        self.assertEqual([s["cited"] for s in self.agent.last_sources], [True, True, False, False, False])
        self.assertEqual(self.agent.last_warnings, ["cited [S9], which is not among this turn's sources"])
        self.assertEqual(self.agent.focus_documents, {"a.pdf"})

    def test_history_stores_resolved_titles_and_the_raw_question(self):
        self._turn("q")
        self.assertEqual(self.agent.history[0], {"role": "user", "content": "q"})
        self.assertIn("[Title a.pdf]", self.agent.history[1]["content"])
        self.assertNotIn("[S1]", self.agent.history[1]["content"])
        self.assertIn("[S9]", self.agent.history[1]["content"])                 # unknown keys are left visible

    def test_stream_gets_temperature_and_options_and_a_keyed_context(self):
        self._turn("q")
        s = self.streams[0]
        self.assertEqual(s["temperature"], a7.CHAT_TEMPERATURE); self.assertEqual(s["options"], a7.CHAT_OLLAMA_OPTIONS)
        self.assertEqual(s["messages"][0]["role"], "system")
        self.assertIn("[S1] Title a.pdf — p.1, text\nA one", s["messages"][-1]["content"])
        self.assertEqual(self.agent.last_budget["num_ctx"], a7.CHAT_OLLAMA_OPTIONS["num_ctx"])


class TestFollowUp(ChatTestCase):
    def test_condensed_query_is_searched_and_focus_search_runs_first(self):
        self._turn("What does the MoS2 paper say?")
        self.searches.clear()
        self._turn("What about its limitations?")
        condense = [c for c in self.chats if "Rewrite the new question" in c["content"]]
        self.assertEqual(len(condense), 1)
        self.assertIn("What does the MoS2 paper say?", condense[0]["content"])
        self.assertEqual(condense[0]["temperature"], 0.0)
        self.assertEqual(self.agent.last_query, "limitations of the MoS2 covalent network hopping study")
        self.assertTrue(all(s["query"] == self.agent.last_query for s in self.searches))
        focus = [s for s in self.searches if s["doc_filter"]]
        self.assertEqual(focus[0]["doc_filter"], {"a.pdf"}); self.assertEqual(focus[0]["top_k"], a7.CHAT_FOCUS_TOP_K)
        self.assertTrue(self.agent.last_sources[0]["citation"].startswith("Title a.pdf"))
        self.assertIn("focus chunk 0", self.streams[-1]["messages"][-1]["content"])

    def test_condense_failure_or_nonsense_falls_back_to_the_question(self):
        self._turn("first")
        with patch.object(a7, "chat", side_effect=RuntimeError("down")):
            self._turn("second question here")
        self.assertEqual(self.agent.last_query, "second question here")
        with patch.object(a7, "chat", return_value=types.SimpleNamespace(content="")):
            self._turn("third question here")
        self.assertEqual(self.agent.last_query, "third question here")

    def test_condense_off_by_config(self):
        self._turn("first")
        with patch.object(a7, "CHAT_CONDENSE", False):
            self._turn("second")
        self.assertEqual(self.agent.last_query, "second")
        self.assertEqual([c for c in self.chats if "Rewrite" in c["content"]], [])


class TestBudget(ChatTestCase):
    def test_older_turns_fold_into_memory_and_the_prompt_fits(self):
        self.answer = "answer " * 120 + "[S1]."                                   # ≈ 850 chars ≈ 215 tokens per answer
        with patch.object(a7, "CHAT_OLLAMA_OPTIONS", {"num_ctx": 1400}), patch.object(a7, "CHAT_CONTEXT_MAX_CHARS", 600):
            for i in range(5):
                self._turn(f"question number {i}")
        self.assertIn("MEMORY", self.agent.memory)
        self.assertTrue(any("Summarise these earlier turns" in c["content"] for c in self.chats))
        last = self.streams[-1]
        self.assertIn("Conversation so far", last["messages"][0]["content"])
        self.assertLessEqual(self.agent.last_budget["prompt_tokens_est"], 1400 - a7.CHAT_ANSWER_RESERVE_TOKENS)
        self.assertLess(len(self.agent.history), 10)
        self.assertTrue(self.agent.last_budget["folded"] >= 1)

    def test_nothing_folds_while_everything_fits(self):
        for i in range(3):
            self._turn(f"q{i}")
        self.assertEqual(self.agent.memory, "")
        self.assertEqual(len(self.agent.history), 6)

    def test_memory_fold_failure_still_drops_to_fit(self):
        self.answer = "answer " * 120 + "[S1]."
        def chat_fails_memory(messages, **kw):
            if "Summarise these earlier turns" in messages[-1]["content"]:
                raise RuntimeError("down")
            return types.SimpleNamespace(content="condensed")
        with patch.object(a7, "chat", side_effect=chat_fails_memory), \
             patch.object(a7, "CHAT_OLLAMA_OPTIONS", {"num_ctx": 1400}), patch.object(a7, "CHAT_CONTEXT_MAX_CHARS", 600):
            for i in range(5):
                self._turn(f"question number {i}")
        self.assertEqual(self.agent.memory, "")
        self.assertLessEqual(self.agent.last_budget["prompt_tokens_est"], 1400 - a7.CHAT_ANSWER_RESERVE_TOKENS)


class TestNoResults(ChatTestCase):
    def test_no_context_message(self):
        with patch.object(a7, "hybrid_search", return_value=[]):
            self._turn("anything")
        self.assertIn("No passages were retrieved", self.streams[-1]["messages"][-1]["content"])
        self.assertEqual(self.agent.last_sources, [])


class TestClearAndExport(ChatTestCase):
    def test_clear_resets_everything(self):
        self._turn("q")
        self.agent.memory = "m"
        self.agent.clear_history()
        self.assertEqual((self.agent.history, self.agent.memory, self.agent.focus_documents, self.agent.last_sources), ([], "", set(), []))

    def test_export_includes_memory(self):
        import os, tempfile
        self._turn("q")
        self.agent.memory = "what we established so far"
        with tempfile.TemporaryDirectory() as d, patch.object(a7, "DRAFTS_DIR", d):
            path = self.agent.export_conversation()
            text = open(path, encoding="utf-8").read()
        self.assertIn("what we established so far", text)
        self.assertIn("[Title a.pdf]", text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_research_chat.py -q`
Expected: FAIL — `AttributeError: module … has no attribute 'chat'` / `'CHAT_TEMPERATURE'`.

- [ ] **Step 3: Implement** — replace the imports and the `ResearchChat` class in `research_assistant/agents/agent7_research_chat.py`:

```python
from research_assistant.config import (
    CHAT_ANSWER_RESERVE_TOKENS,
    CHAT_CONDENSE,
    CHAT_CONTEXT_MAX_CHARS,
    CHAT_FOCUS_TOP_K,
    CHAT_MODEL,
    CHAT_OLLAMA_OPTIONS,
    CHAT_PER_DOC_CAP,
    CHAT_TEMPERATURE,
    DRAFTS_DIR,
)
from research_assistant.prompts import (
    CHAT_CONDENSE_USER,
    CHAT_MEMORY_USER,
    CHAT_TURN_NO_CONTEXT,
    CHAT_TURN_USER,
    RESEARCH_CHAT_SYSTEM,
)
from research_assistant.shared.chat_context import (
    cap_per_document,
    cited_keys,
    estimate_tokens,
    fit_history,
    format_context,
    merge_results,
    messages_tokens,
    resolve_keys,
)
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.llm import chat, chat_stream
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import hybrid_search
```

```python
class ResearchChat:
    """Multi-turn conversational RAG agent backed by the paper database.

    Each turn: condense the follow-up into a standalone query → search the
    corpus (capped per paper) and the focus papers → fit context and history
    into the window, folding older turns into a rolling memory → stream the
    answer → check its [S#] keys against this turn's sources.
    """

    def __init__(self, top_k: int = 5, search_resources: tuple = None):
        self.top_k = top_k
        self.history = []          # list of {"role": ..., "content": ...}; answers stored with keys resolved to titles
        self.memory = ""           # rolling summary of turns that no longer fit
        self.focus_documents = set()
        self.last_sources = []     # this turn's keyed sources, with "cited"
        self.last_query = ""       # what was actually searched
        self.last_warnings = []
        self.last_budget = {}
        self.turn_count = 0

        if search_resources:
            self.collection, self.bm25, self.texts, self.metadatas = search_resources
        else:
            logger.info("Loading search resources…")
            self.collection, self.bm25, self.texts, self.metadatas = load_search_resources()
        logger.info("Ready. %d chunks in database.", len(self.texts))

    # ── Retrieval ────────────────────────────────────────────────────────

    @staticmethod
    def _transcript(messages, limit):
        return "\n".join(
            f"{'Researcher' if m['role'] == 'user' else 'Assistant'}: {m['content'][:limit]}" for m in messages
        )

    def _condense(self, question: str) -> str:
        """The follow-up as a standalone query, or the question itself."""
        if not CHAT_CONDENSE or not self.history:
            return question
        try:
            out = chat(
                [{"role": "user", "content": CHAT_CONDENSE_USER.format(
                    history=self._transcript(self.history[-4:], 600), question=question)}],
                model=CHAT_MODEL, temperature=0.0, options=CHAT_OLLAMA_OPTIONS,
            ).content.strip().strip('"')
        except Exception as exc:                # noqa: BLE001
            logger.warning("Condense call failed (%s) — searching the question as typed.", exc)
            return question
        words = len(out.split())
        if not (3 <= words <= 60):
            return question
        return out

    def _search(self, query: str) -> list:
        common = dict(top_k=self.top_k * 2, exclude_types={"figure_description"})
        global_hits = hybrid_search(query, self.collection, self.bm25, self.texts, self.metadatas, **common)
        global_hits = cap_per_document(global_hits, CHAT_PER_DOC_CAP)[: self.top_k]
        focus_hits = []
        if self.focus_documents and CHAT_FOCUS_TOP_K > 0:
            focus_hits = hybrid_search(
                query, self.collection, self.bm25, self.texts, self.metadatas,
                top_k=CHAT_FOCUS_TOP_K, doc_filter=set(self.focus_documents),
                exclude_types={"figure_description"},
            )
        return merge_results(focus_hits, global_hits)

    # ── Window budget ────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        if self.memory:
            return RESEARCH_CHAT_SYSTEM + "\n\nConversation so far: " + self.memory
        return RESEARCH_CHAT_SYSTEM

    def _fold_memory(self, dropped) -> None:
        """Older turns leave the prompt only through here."""
        try:
            out = chat(
                [{"role": "user", "content": CHAT_MEMORY_USER.format(
                    memory=self.memory or "(none)", turns=self._transcript(dropped, 800))}],
                model=CHAT_MODEL, temperature=0.0, options=CHAT_OLLAMA_OPTIONS,
            ).content.strip()
            if out:
                self.memory = out
        except Exception as exc:                # noqa: BLE001
            logger.warning("Memory fold failed (%s) — %d earlier message(s) leave the window unsummarised.",
                           exc, len(dropped))

    def _fit(self, user_content: str) -> tuple[list, int]:
        """History that fits beside the system prompt, the turn, and the reserve."""
        num_ctx = CHAT_OLLAMA_OPTIONS.get("num_ctx", 4096)
        folded = 0
        for _ in range(2):                      # the fold changes the system prompt; refit once
            fixed = estimate_tokens(self._system_prompt()) + estimate_tokens(user_content) + CHAT_ANSWER_RESERVE_TOKENS
            kept, dropped = fit_history(self.history, max(0, num_ctx - fixed))
            if not dropped:
                break
            logger.info("Folding %d earlier message(s) into memory to fit num_ctx=%d.", len(dropped), num_ctx)
            self._fold_memory(dropped)
            self.history = kept
            folded += 1
        return kept, folded

    # ── Public API ───────────────────────────────────────────────────────

    def stream_turn(self, user_message: str):
        """Process one user turn: retrieve context, stream response, update history."""
        self.turn_count += 1
        query = self._condense(user_message)
        self.last_query = query

        results = self._search(query)
        context, sources = format_context(results, CHAT_CONTEXT_MAX_CHARS)
        if context:
            user_content = CHAT_TURN_USER.format(question=user_message, context=context)
        else:
            user_content = CHAT_TURN_NO_CONTEXT.format(question=user_message)

        kept, folded = self._fit(user_content)
        messages = [{"role": "system", "content": self._system_prompt()}] + kept + [{"role": "user", "content": user_content}]
        num_ctx = CHAT_OLLAMA_OPTIONS.get("num_ctx", 4096)
        self.last_budget = {
            "num_ctx": num_ctx, "prompt_tokens_est": messages_tokens(messages),
            "reserve": CHAT_ANSWER_RESERVE_TOKENS, "context_chars": len(context),
            "history_messages": len(kept), "folded": folded,
        }

        full_answer = ""
        for content in chat_stream(messages, model=CHAT_MODEL, options=CHAT_OLLAMA_OPTIONS, temperature=CHAT_TEMPERATURE):
            full_answer += content
            yield content

        used = cited_keys(full_answer)
        valid = {s["key"] for s in sources}
        for s in sources:
            s["cited"] = s["key"] in used
        self.last_sources = sources
        self.last_warnings = [f"cited [{k}], which is not among this turn's sources" for k in used if k not in valid]
        cited_docs = {s["document"] for s in sources if s["cited"] and s["document"]}
        self.focus_documents = cited_docs or {s["document"] for s in sources[:2] if s["document"]}

        # Store the clean question and the answer with keys resolved to titles:
        # [S3] meant something only in this turn.
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": resolve_keys(full_answer, sources)})

    def clear_history(self):
        """Reset the conversation."""
        self.history.clear()
        self.memory = ""
        self.focus_documents = set()
        self.last_sources = []
        self.last_query = ""
        self.last_warnings = []
        self.last_budget = {}
        self.turn_count = 0

    def export_conversation(self) -> str:
        """Save the full conversation to a timestamped markdown file."""
        os.makedirs(DRAFTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(DRAFTS_DIR, f"research_chat_{timestamp}.md")

        lines = [f"# Research Chat — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
        if self.memory:
            lines.append(f"\n> **Earlier in this conversation:** {self.memory}\n")
        for msg in self.history:
            role = "🧑‍🔬 **Researcher**" if msg["role"] == "user" else "🤖 **Assistant**"
            lines.append(f"\n{role}\n\n{msg['content']}\n")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return filepath
```

Delete the old `_retrieve_context` and `_generate_stream` methods and the `SYSTEM_PROMPT = RESEARCH_CHAT_SYSTEM` alias (search the file for other uses of `SYSTEM_PROMPT` first — `run_repl` does not use it). In `run_repl`, extend the `/sources` handler to print `agent.last_query` and, per source, `[key] citation — p.page (cited|not cited)` plus `agent.last_warnings`; add `/memory` printing `agent.memory or "(nothing folded yet)"`, and add it to `HELP_TEXT`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_research_chat.py tests/test_chat_context.py -q` — Expected: PASS. If `TestBudget` fails on the exact number of folds, check `_fit`'s loop, not the test: after a fold the memory text enters the system prompt and the remaining history must be refit once.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/agents/agent7_research_chat.py tests/test_research_chat.py
git commit -m "feat(chat): condensed follow-ups, focus-paper search, a budgeted window with rolling memory, checked source keys"
```

---

### Task 5: Tab 4 shows the query, keyed sources, warnings and memory

**Files:**
- Modify: `app.py:1016-1019` (sources expander) and the surrounding Tab 4 block
- Test: `tests/test_app_names.py` (existing guard)

- [ ] **Step 1: Implement** — replace

```python
                if agent.last_sources:
                    with st.expander(f"Sources · {len(agent.last_sources)}"):
                        for s in agent.last_sources:
                            st.caption(f"**{s['document']}** — {s['citation']}")
```
with
```python
                for w in agent.last_warnings:
                    st.warning(w, icon="⚠️")
                if agent.last_sources:
                    cited = sum(1 for s in agent.last_sources if s.get("cited"))
                    with st.expander(f"Sources · {cited} cited of {len(agent.last_sources)} · searched for: {agent.last_query[:80]}"):
                        for s in agent.last_sources:
                            mark = "✓" if s.get("cited") else "·"
                            st.caption(f"{mark} **[{s['key']}]** {s['citation']} — p.{s.get('page', '?')}, {s.get('section') or 'text'}")
                        b = agent.last_budget
                        if b:
                            st.caption(f"prompt ≈ {b['prompt_tokens_est']} of {b['num_ctx']} tokens · "
                                       f"{b['history_messages']} history message(s)"
                                       + (f" · {b['folded']} fold(s) into memory" if b.get('folded') else ""))
                if agent.memory:
                    with st.expander("Earlier in this conversation (memory)"):
                        st.write(agent.memory)
```

- [ ] **Step 2: Check** — `CITATION_LOG_FILE=0 python -m pytest tests/test_app_names.py -q` → PASS; `CITATION_LOG_FILE=0 python -c "import app"` → no error.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): research chat shows the searched query, keyed sources with cited marks, warnings, budget and memory"
```

---

### Task 6: Docs

**Files:**
- Modify: `ARCHITECTURE.md` (new §5.6), `PIPELINE.md` (Agent 7 row; config notes), `README.md` (the "keyhole" bullet)

- [ ] **Step 1: `ARCHITECTURE.md`** — after §5.5 (or after §5.4 if the synthesis plan has not landed) add:

```markdown
### 5.6 Research chat: retrieve for the follow-up, fit the window, remember the thread

A follow-up ("what about its limitations?") embedded as typed retrieved
from four unrelated papers and missed the one the conversation was about.
Agent 7 now rewrites every follow-up into a standalone query with one short
model call (shown in the UI as "searched for"), searches both the corpus
(capped at three chunks per paper, so a topic question sees more than one
paper) and the papers cited in the previous answer, and puts the focus hits
first.

The window is a budget, not a hope: answer reserve, system prompt with
memory, context up to a cap, then as many whole recent turns as fit. Older
turns leave the prompt only by being folded into a rolling memory by a
model call, which rides in the system message; nothing is truncated
silently (Ollama truncates from the front, where the system prompt is). The
budget derives from `num_ctx`, so raising it on a capable machine enlarges
every line.

Sources are keyed `[S1]…`; the answer must cite them per sentence; the keys
are checked after the stream and unknown ones reported; in history and
export the keys are resolved to titles. Temperature is explicit (0.3).
```

- [ ] **Step 2: `PIPELINE.md`** — Agent 7 row: append *"Follow-ups are condensed into standalone queries; the papers cited last turn get their own search; each turn fits `num_ctx` by budget with older turns folded into a rolling memory; sources are keyed and checked."* Config notes: add the six `CITATION_CHAT_*` variables.

- [ ] **Step 3: `README.md`** — in "What he's worst at", the *"He reads through a keyhole"* bullet: append one sentence: *"In chat he now remembers the thread — follow-ups are searched for what they mean, and older turns are summarised rather than silently cut."*

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md PIPELINE.md README.md
git commit -m "docs: research chat depth — condensed retrieval, focus papers, budgeted window with memory, keyed sources"
```

---

### Task 7: Live check — the same conversation, six turns, instrumented

Needs Ollama (`gemma4:e2b`, `nomic-embed-text`) and the v2 index.

- [ ] **Step 1: The bench** (kept)

```python
# scripts/chat_bench.py
"""Run a scripted conversation through ResearchChat on the live index and
print, per turn: the condensed query, the papers retrieved, the estimated
prompt size against num_ctx, folds into memory, and the citation check.

    PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/chat_bench.py
"""
import json
import os
import time

from research_assistant.agents.agent7_research_chat import ResearchChat
from research_assistant.config import DATA_DIR
from research_assistant.shared.atomic import atomic_write_json

TURNS = [
    "What does the covalent MoS2 network paper say about hopping transport?",
    "What about its limitations?",
    "How does that compare with what the carbon nanotube network paper found?",
    "Which of the two has the stronger evidence for the mechanism it claims, and why?",
    "What experiment would distinguish variable-range hopping from nearest-neighbour hopping in the MoS2 networks?",
    "Summarise what we have established so far and what is still open.",
]


def main():
    agent = ResearchChat(top_k=5)
    record = []
    for i, q in enumerate(TURNS, 1):
        t0 = time.perf_counter()
        answer = "".join(agent.stream_turn(q))
        secs = time.perf_counter() - t0
        docs = [s["document"] for s in agent.last_sources]
        b = agent.last_budget
        row = {"turn": i, "question": q, "searched_for": agent.last_query, "documents": docs,
               "cited": [s["key"] for s in agent.last_sources if s["cited"]], "warnings": agent.last_warnings,
               "prompt_tokens_est": b["prompt_tokens_est"], "num_ctx": b["num_ctx"], "reserve": b["reserve"],
               "folded": b["folded"], "memory_words": len(agent.memory.split()), "seconds": round(secs), "answer": answer}
        record.append(row)
        print(f"\n=== turn {i} ({row['seconds']}s)\n  asked:        {q}\n  searched for: {agent.last_query}\n  retrieved:    {docs}"
              f"\n  cited: {row['cited']}  warnings: {agent.last_warnings}\n  prompt ≈ {b['prompt_tokens_est']}/{b['num_ctx']} tokens"
              f" (reserve {b['reserve']}), history {b['history_messages']} msgs, folds {b['folded']}, memory {row['memory_words']} words"
              f"\n  answer: {answer[:400].replace(chr(10), ' ')}…")
    out = os.path.join(DATA_DIR, "eval", "chat", time.strftime("%Y%m%d-%H%M%S") + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    atomic_write_json(out, record)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it** — `PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python scripts/chat_bench.py` (≈ 8–10 minutes).

- [ ] **Step 3: Gates**

- Turn 2's `searched for` names the MoS₂ network paper or its subject explicitly, and `retrieved` includes `doi_10.1002_adma.202211157.pdf`. (The "before": four unrelated papers.)
- Every turn: `prompt_tokens_est ≤ num_ctx − reserve`.
- By turn 6, `folded ≥ 1` and `memory_words > 0`, and the turn-6 answer mentions the MoS₂ paper — the thread survived the fold.
- Every turn: `warnings == []`, and `cited` is non-empty whenever `retrieved` is.
- Read the six answers: turn 2 should now actually discuss limitations from the paper; turn 4 should cite both papers.

- [ ] **Step 4: The app** — `CITATION_INDEX_VERSION=2 streamlit run app.py --server.port 8601`, Tab 4: ask turns 1–2; confirm the sources expander shows *searched for: …*, `✓ [S1] …` lines, the budget caption; after enough turns, the memory expander.

- [ ] **Step 5: Suite** — `CITATION_LOG_FILE=0 python -m pytest tests/ -q` (deselect `tests/test_ingestion.py` if the lock is held). Green.

- [ ] **Step 6: Commit**

```bash
git add scripts/chat_bench.py
git commit -F - <<'EOF'
chore(chat): six-turn live check on the v2 index after the depth changes

<paste the six per-turn blocks>
app: <one line>
suite: <last line>
EOF
```

---

## Self-review

**Spec coverage.** §2A → Task 3 (prompt), Task 4 (`_condense`, `last_query`, config off-switch). §2B → Task 4 (`_search`: cap, focus, merge). §2C → Task 2 (`fit_history`, `estimate_tokens`), Task 4 (`_fit`, `_fold_memory`, `_system_prompt`, `last_budget`), Task 3 (config). §2D → Task 2 (`format_context`, `cited_keys`, `resolve_keys`), Task 3 (system prompt), Task 4 (checks, focus, history storage), Task 1 (temperature). §2E → Task 5 (UI), Task 4 (`/sources`, `/memory`, export). §3 honoured: index, synthesis, `num_ctx` untouched. §4 config names match Task 3. §6 → Task 7.

**Type consistency.** `format_context` returns sources with `key/document/citation/page/section/chunk_index/cited` — the fields Task 4 sets (`cited`), Task 5 renders, and Task 7 reads. `fit_history` returns `(kept, dropped)` as Task 4's `_fit` consumes. `chat_stream(..., temperature=)` (Task 1) is what Task 4 calls. `last_budget` keys (`num_ctx, prompt_tokens_est, reserve, context_chars, history_messages, folded`) are what Tasks 5 and 7 read.

**Placeholders.** None. Task 7's commit asks for pasted output.

**Validated before writing.** The `llm.py`, `prompts.py`, `config.py`, `chat_context.py` and `agent7` changes were assembled into a scratch copy of the package and the plan's tests run against them together with the existing `tests/test_llm.py`: 48/48. One test expectation in this plan was corrected during that run (`cap_per_document`: the code was right).
