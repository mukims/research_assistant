# Judgement Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the standalone judgement module as Agent 8 — a post-hoc audit that verifies, per citation, whether the cited source actually supports the claim citing it.

**Architecture:** A new `research_assistant/judgement/` package holds the V1.4 prompt and a `judge()` that routes through `shared.llm.chat()` rather than a provider SDK. A new `agents/agent8_verifier.py` reads Agent 5's cited draft plus its citation mapping, re-retrieves the best chunk from each cited source, judges each claim–evidence pair, and writes a JSON record and a Markdown report. Agents 4 and 5 are not modified.

**Tech Stack:** Python 3.10–3.12, pytest (collecting `unittest.TestCase` classes), ChromaDB, `rank_bm25`, existing `shared/` helpers (`llm`, `db`, `search`, `atomic`, `retry`, `log`).

**Spec:** `notimportant/superpowers/specs/2026-09-09-judgement-integration-design.md`

## Global Constraints

- **Python 3.10–3.12.** `pyproject.toml` sets `requires-python = ">=3.10,<3.13"`. Do not use 3.13+ syntax.
- **No provider SDK outside `shared/llm.py`.** No module may `import openai` or `import ollama`. `LLM_BACKEND` is the single switch. (Spec §4.3)
- **Nothing writes outside `CITATION_DATA_DIR`** except files written next to a user-supplied draft path. (Spec §1)
- **Agents 4 and 5 are not modified.** Import from them; do not edit them. (Spec §4.1)
- **`shared/search.py` is not modified.** (Spec §5.4)
- **All file writes go through `shared/atomic.py`** (`atomic_write` / `atomic_write_json`), like every other manifest.
- **Tests use `unittest.TestCase` classes**, matching every file in `tests/`. pytest collects them unchanged.
- **Live LLM tests stay behind `RUN_LLM_TESTS=1`** so CI is unaffected.
- **Heavy imports stay inside functions.** `chromadb` and the embedding stack are imported lazily so `requirements-test.txt` alone can run the suite.
- Run the suite with `CITATION_LOG_FILE=0 python -m pytest tests/ -v` so it does not write into `data/logs/`.

---

### Task 1: `chat()` gains `temperature` and `options`

**Files:**
- Modify: `research_assistant/shared/llm.py:46-60` (`chat`), `:90-110` (`_openai_chat`), `:63-79` (`_ollama_chat`)
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `chat(messages, model=None, images=None, temperature=None, options=None) -> ChatResult`. Both new parameters default to `None`; when both are `None` the outgoing provider call is identical to today's. Task 2 calls this with `temperature=0.0` and `options={"num_ctx": 8192}`.

Why: `_openai_chat` currently calls `create(model, messages, stream=False)` with nothing else, and `_ollama_chat` passes no options at all — so neither temperature nor `num_ctx` is reachable. Task 2 needs both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
def _openai_reply(text):
    msg = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg)],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _ollama_reply(text):
    return types.SimpleNamespace(
        message=types.SimpleNamespace(content=text),
        prompt_eval_count=1,
        eval_count=1,
    )


class TestChatTemperatureOpenAI(unittest.TestCase):
    def _run(self, **kwargs):
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_reply("ok")
        module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))
        with patch.dict("sys.modules", {"openai": module}), \
             patch.object(llm, "LLM_BACKEND", "openai"):
            llm.chat([{"role": "user", "content": "hi"}], **kwargs)
        return client.chat.completions.create.call_args.kwargs

    def test_temperature_absent_when_not_given(self):
        """The default call must stay byte-for-byte what it is today."""
        self.assertNotIn("temperature", self._run())

    def test_temperature_forwarded(self):
        self.assertEqual(self._run(temperature=0.0)["temperature"], 0.0)

    def test_options_are_ignored_by_openai(self):
        """ollama runtime options have no OpenAI equivalent."""
        self.assertNotIn("options", self._run(options={"num_ctx": 8192}))


class TestChatTemperatureOllama(unittest.TestCase):
    def _run(self, **kwargs):
        fake = MagicMock()
        fake.chat.return_value = _ollama_reply("ok")
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            llm.chat([{"role": "user", "content": "hi"}], **kwargs)
        return fake.chat.call_args.kwargs

    def test_options_absent_when_not_given(self):
        self.assertNotIn("options", self._run())

    def test_temperature_lands_inside_options(self):
        self.assertEqual(self._run(temperature=0.0)["options"], {"temperature": 0.0})

    def test_options_and_temperature_merge(self):
        merged = self._run(temperature=0.0, options={"num_ctx": 8192})["options"]
        self.assertEqual(merged, {"num_ctx": 8192, "temperature": 0.0})

    def test_caller_options_dict_is_not_mutated(self):
        """A module-level config constant must not grow a temperature key."""
        opts = {"num_ctx": 8192}
        fake = MagicMock()
        fake.chat.return_value = _ollama_reply("ok")
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            llm.chat([{"role": "user", "content": "hi"}], temperature=0.0, options=opts)
        self.assertEqual(opts, {"num_ctx": 8192})
```

`test_temperature_forwarded` and `test_temperature_lands_inside_options` both use `0.0` deliberately: an implementation written as `if temperature:` would silently drop it, and `0.0` is the only value this feature actually needs.

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_llm.py -v -k "Temperature"`
Expected: FAIL — `TypeError: chat() got an unexpected keyword argument 'temperature'`

- [ ] **Step 3: Implement**

In `research_assistant/shared/llm.py`, replace `chat`:

```python
def chat(messages, model=None, images=None, temperature=None, options=None) -> ChatResult:
    """Run a chat completion.

    Args:
        messages: list of ``{"role", "content"}`` dicts.
        model:    model id; defaults to config.LLM_MODEL.
        images:   optional list of local image paths for a vision request
                  (attached to the last user message).
        temperature: sampling temperature. None leaves the provider's own
                  default in place, so existing callers are unaffected.
        options:  ollama runtime options (e.g. JUDGEMENT_OLLAMA_OPTIONS).
                  The openai backend has no equivalent and ignores it.
    """
    model = model or LLM_MODEL
    if LLM_BACKEND == "openai":
        return _openai_chat(messages, model, images, temperature)
    if LLM_BACKEND == "ollama":
        return _ollama_chat(messages, model, images, temperature, options)
    raise ValueError(f"Unknown LLM_BACKEND {LLM_BACKEND!r} (expected 'ollama' or 'openai')")
```

Replace `_ollama_chat`:

```python
def _ollama_chat(messages, model, images, temperature=None, options=None) -> ChatResult:
    import ollama

    if images:
        messages = list(messages)
        messages[-1] = {**messages[-1], "images": list(images)}

    # Copied, not mutated: callers pass a module-level config dict.
    opts = dict(options) if options else {}
    if temperature is not None:
        opts["temperature"] = temperature

    kwargs = {"model": model, "messages": messages, "stream": False}
    if opts:
        kwargs["options"] = opts

    resp = ollama.chat(**kwargs)
    try:
        content = resp.message.content
    except AttributeError:
        content = resp["message"]["content"]
    return ChatResult(
        content=content,
        prompt_tokens=getattr(resp, "prompt_eval_count", "N/A"),
        completion_tokens=getattr(resp, "eval_count", "N/A"),
    )
```

In `_openai_chat`, change the signature and the `create` call. Leave the `images` block exactly as it is:

```python
def _openai_chat(messages, model, images, temperature=None) -> ChatResult:
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    if images:
        messages = list(messages)
        last = messages[-1]
        parts = [{"type": "text", "text": last["content"]}]
        parts += [
            {"type": "image_url", "image_url": {"url": _b64_data_url(p)}} for p in images
        ]
        messages[-1] = {**last, "content": parts}

    kwargs = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)
    usage = getattr(resp, "usage", None)
    return ChatResult(
        content=resp.choices[0].message.content,
        prompt_tokens=getattr(usage, "prompt_tokens", "N/A"),
        completion_tokens=getattr(usage, "completion_tokens", "N/A"),
    )
```

- [ ] **Step 4: Run the new tests**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_llm.py -v -k "Temperature"`
Expected: PASS (7 tests)

- [ ] **Step 5: Run the whole suite — the change must be invisible to existing callers**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/ -v`
Expected: PASS, no regressions. If anything in `tests/test_llm.py` that existed before this task now fails, the change is wrong — revert and redo, do not edit the old test to match.

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/llm.py tests/test_llm.py
git commit -m "feat(llm): add temperature and options to chat()"
```

---

### Task 2: The `judgement` package

**Files:**
- Create: `research_assistant/judgement/__init__.py`, `research_assistant/judgement/judge.py`, `research_assistant/judgement/prompt.md`, `research_assistant/judgement/cases/cases.jsonl`
- Modify: `research_assistant/config.py` (append a judgement section)
- Test: `tests/test_judgement.py`

**Interfaces:**
- Consumes: `chat(..., temperature=..., options=...)` from Task 1.
- Produces:
  - `build_prompt(claim: str, citation_evidence: str) -> str`
  - `parse_judgement(raw: str) -> dict` — raises `JudgementParseError`
  - `judge(claim: str, citation_evidence: str, model: str | None = None) -> dict`
  - `JudgementParseError(ValueError)` with a `.raw` attribute
  - Constants `VALID_JUDGEMENTS`, `VALID_CONFIDENCE`, `VALID_SUFFICIENCY`
  - Config names `JUDGEMENT_MODEL`, `JUDGEMENT_TEMPERATURE`, `JUDGEMENT_TOP_K`, `JUDGEMENT_OLLAMA_OPTIONS`

- [ ] **Step 1: Copy the two data files verbatim**

```bash
mkdir -p research_assistant/judgement/cases
cp /home/shardul/Downloads/judgement/prompt.md research_assistant/judgement/prompt.md
cp /home/shardul/Downloads/judgement/cases/cases.jsonl research_assistant/judgement/cases/cases.jsonl
```

Do not edit either file. `prompt.md` is a balanced rubric — the first-match-wins ordering in its Step 3 and the D-vs-E contrast at the end are load-bearing, and the six cases in `cases.jsonl` are the only guard on them.

Verify: `wc -l research_assistant/judgement/prompt.md` → 259, and `wc -l research_assistant/judgement/cases/cases.jsonl` → 6.

- [ ] **Step 2: Add the config section**

Append to `research_assistant/config.py`, after the Agent 5 section:

```python
# ─── Agent 8 — Verifier (judgement) ──────────────────────────────────────────
# None → LLM_MODEL, matching SUMMARY_MODEL's pattern above.
JUDGEMENT_MODEL       = os.environ.get("CITATION_JUDGEMENT_MODEL", "") or None
# The judgement prompt is a rubric, not a generation task, and its regression
# cases assert exact verdicts — sampling makes both meaningless.
JUDGEMENT_TEMPERATURE = 0.0
# Chunks judged per cited source. 1 = judge the single best-matching chunk.
JUDGEMENT_TOP_K       = _env_int("CITATION_JUDGEMENT_TOP_K", 1)
# The prompt is ~3,760 tokens. _ollama_chat sends no options, so a local run
# would otherwise use the model default (commonly 4096, sometimes 2048) and
# Ollama would truncate — silently, from the tail, which is exactly where the
# worked examples live. Ignored by the openai backend.
JUDGEMENT_OLLAMA_OPTIONS = {"num_ctx": 8192}
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_judgement.py`:

```python
"""Tests for the judgement module.

The offline tests here need no API key and no network: parsing and validation
are ordinary Python, and that is what makes them runnable in CI. The live
regression suite against cases.jsonl is opt-in via RUN_LLM_TESTS=1.
"""

import json
import os
import unittest
from pathlib import Path

import research_assistant.judgement.judge as judge_mod
from research_assistant.judgement.judge import (
    JudgementParseError,
    build_prompt,
    parse_judgement,
)

CASES_PATH = Path(judge_mod.__file__).parent / "cases" / "cases.jsonl"

VALID_JUDGEMENTS = {
    "Supports",
    "Partially supports",
    "Contradicts",
    "Does not support",
    "Unclear / insufficient evidence",
}


def load_cases():
    with CASES_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _good(**overrides):
    """A well-formed judgement payload, minus whatever a test overrides."""
    payload = {
        "slots": {
            "finding": {"assertion": "a", "verdict": "Supports"},
            "scope": {"assertion": "b", "verdict": "Supports"},
            "strength": {"assertion": "c", "verdict": "Not applicable"},
        },
        "judgement": "Supports",
        "evidence_sufficiency": "sufficient",
        "confidence": "High",
        "supporting_span": "some span",
        "reason": "because",
    }
    payload.update(overrides)
    return payload


class TestCases(unittest.TestCase):
    def test_cases_are_well_formed(self):
        cases = load_cases()
        self.assertEqual(len(cases), 6)
        for case in cases:
            self.assertTrue(case["claim"])
            self.assertTrue(case["citation_evidence"])
            self.assertIn(case["expected_judgement"], VALID_JUDGEMENTS)


class TestBuildPrompt(unittest.TestCase):
    def test_both_placeholders_are_filled(self):
        prompt = build_prompt("TEST CLAIM", "TEST EVIDENCE")
        self.assertIn("TEST CLAIM", prompt)
        self.assertIn("TEST EVIDENCE", prompt)
        self.assertNotIn("{{CLAIM}}", prompt)
        self.assertNotIn("{{CITATION_EVIDENCE}}", prompt)


class TestParseJudgement(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_judgement(json.dumps(_good()))["judgement"], "Supports")

    def test_json_fence_is_stripped(self):
        raw = "```json\n" + json.dumps(_good()) + "\n```"
        self.assertEqual(parse_judgement(raw)["judgement"], "Supports")

    def test_bare_fence_is_stripped(self):
        raw = "```\n" + json.dumps(_good()) + "\n```"
        self.assertEqual(parse_judgement(raw)["judgement"], "Supports")

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertEqual(
            parse_judgement("\n\n" + json.dumps(_good()) + "\n\n")["judgement"],
            "Supports",
        )

    def test_invalid_json_raises_with_raw_attached(self):
        with self.assertRaises(JudgementParseError) as ctx:
            parse_judgement("not json at all")
        self.assertEqual(ctx.exception.raw, "not json at all")

    def test_missing_field_raises(self):
        payload = _good()
        del payload["confidence"]
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(payload))

    def test_lowercased_judgement_is_rejected(self):
        """A near-miss category would otherwise distort the report's counts."""
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(_good(judgement="supports")))

    def test_unknown_confidence_is_rejected(self):
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(_good(confidence="Very high")))

    def test_unknown_sufficiency_is_rejected(self):
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(_good(evidence_sufficiency="lots")))

    def test_missing_slot_is_rejected(self):
        payload = _good()
        del payload["slots"]["strength"]
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(payload))

    def test_extra_slot_is_rejected(self):
        payload = _good()
        payload["slots"]["vibes"] = {"assertion": "x", "verdict": "Supports"}
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(payload))

    def test_json_array_is_rejected(self):
        with self.assertRaises(JudgementParseError):
            parse_judgement("[1, 2, 3]")


@unittest.skipUnless(
    os.getenv("RUN_LLM_TESTS") == "1",
    "Set RUN_LLM_TESTS=1 to run live LLM tests.",
)
class TestLiveJudgement(unittest.TestCase):
    def test_every_case_reaches_its_expected_judgement(self):
        from research_assistant.judgement.judge import judge

        failures = []
        for case in load_cases():
            result = judge(case["claim"], case["citation_evidence"])
            if result["judgement"] != case["expected_judgement"]:
                failures.append(
                    f"{case['id']} ({case['name']}): "
                    f"expected {case['expected_judgement']!r}, "
                    f"got {result['judgement']!r}"
                )
        self.assertEqual(failures, [], "\n".join(failures))
```

The live test collects every failure before asserting, rather than stopping at the first: when a prompt edit shifts behaviour, which *four* cases moved is the useful signal, not just the first one.

- [ ] **Step 4: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_judgement.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'research_assistant.judgement'`

- [ ] **Step 5: Implement the package**

Create `research_assistant/judgement/__init__.py`:

```python
"""Claim–evidence verification (V1.4)."""
```

Create `research_assistant/judgement/judge.py`:

```python
"""Claim–evidence verification as a pipeline component.

The prompt and its regression cases came from a standalone module that spoke
to OpenAI directly. Routing through shared.llm instead is what lets the same
judgement run against a local Ollama model and a hosted API without the caller
knowing which — the same reason every agent in this repo calls chat().

Parsing is stricter than the source module's bare json.loads. The prompt says
"no code fences" and models emit them anyway, and a near-miss category like
"supports" would otherwise flow through as an unrecognised verdict and quietly
distort the verifier's summary counts.
"""

import json
import re
from pathlib import Path

from research_assistant.config import (
    JUDGEMENT_MODEL,
    JUDGEMENT_OLLAMA_OPTIONS,
    JUDGEMENT_TEMPERATURE,
)
from research_assistant.shared.llm import chat
from research_assistant.shared.log import get_logger

logger = get_logger("judgement")

PROMPT_PATH = Path(__file__).parent / "prompt.md"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")

VALID_JUDGEMENTS = {
    "Supports",
    "Partially supports",
    "Contradicts",
    "Does not support",
    "Unclear / insufficient evidence",
}
VALID_CONFIDENCE = {"High", "Medium", "Low"}
VALID_SUFFICIENCY = {"sufficient", "partial", "insufficient"}

REQUIRED_FIELDS = {
    "slots",
    "judgement",
    "evidence_sufficiency",
    "confidence",
    "supporting_span",
    "reason",
}
REQUIRED_SLOTS = {"finding", "scope", "strength"}

_FENCE_RE = re.compile(r"\A```(?:json)?\s*\n(.*?)\n?```\Z", re.DOTALL)


class JudgementParseError(ValueError):
    """The model's reply was not a usable judgement."""

    def __init__(self, message, raw=""):
        super().__init__(message)
        self.raw = raw


def build_prompt(claim: str, citation_evidence: str) -> str:
    """Fill the prompt input template."""
    return (
        PROMPT_TEMPLATE
        .replace("{{CLAIM}}", claim)
        .replace("{{CITATION_EVIDENCE}}", citation_evidence)
    )


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1) if match else stripped


def _validate(result, raw):
    if not isinstance(result, dict):
        raise JudgementParseError(
            f"Expected a JSON object, got {type(result).__name__}", raw=raw
        )

    missing = REQUIRED_FIELDS - result.keys()
    if missing:
        raise JudgementParseError(f"Missing output fields: {sorted(missing)}", raw=raw)

    for field, allowed in (
        ("judgement", VALID_JUDGEMENTS),
        ("confidence", VALID_CONFIDENCE),
        ("evidence_sufficiency", VALID_SUFFICIENCY),
    ):
        if result[field] not in allowed:
            raise JudgementParseError(
                f"{field} was {result[field]!r}, expected one of {sorted(allowed)}",
                raw=raw,
            )

    slots = result["slots"]
    if not isinstance(slots, dict) or set(slots) != REQUIRED_SLOTS:
        got = sorted(slots) if isinstance(slots, dict) else repr(slots)
        raise JudgementParseError(
            f"slots must be exactly {sorted(REQUIRED_SLOTS)}, got {got}", raw=raw
        )

    return result


def parse_judgement(raw: str) -> dict:
    """Parse and validate one model reply. Raises JudgementParseError."""
    try:
        result = json.loads(_strip_fence(raw))
    except json.JSONDecodeError as exc:
        raise JudgementParseError(f"Reply was not valid JSON: {exc}", raw=raw) from exc
    return _validate(result, raw)


def judge(claim: str, citation_evidence: str, model: str | None = None) -> dict:
    """Verdict on whether *citation_evidence* supports *claim*."""
    result = chat(
        [{"role": "user", "content": build_prompt(claim, citation_evidence)}],
        model=model or JUDGEMENT_MODEL,
        temperature=JUDGEMENT_TEMPERATURE,
        options=JUDGEMENT_OLLAMA_OPTIONS,
    )
    if not result.content:
        raise JudgementParseError("LLM returned an empty response.", raw="")
    return parse_judgement(result.content)
```

`model=model or JUDGEMENT_MODEL` passing `None` is correct: `chat()` falls back to `LLM_MODEL`, which is the documented default.

- [ ] **Step 6: Run the tests**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_judgement.py -v`
Expected: PASS (16 tests, live class skipped)

- [ ] **Step 7: Verify the package imports with only test dependencies**

Run: `CITATION_LOG_FILE=0 python -c "import research_assistant.judgement.judge; print('ok')"`
Expected: `ok`. If this pulls in `chromadb` or `openai`, an import is in the wrong place.

- [ ] **Step 8: Commit**

```bash
git add research_assistant/judgement research_assistant/config.py tests/test_judgement.py
git commit -m "feat(judgement): add claim-evidence verification module"
```

---

### Task 3: Verifier resolution logic

**Files:**
- Create: `research_assistant/agents/agent8_verifier.py`
- Test: `tests/test_verifier.py`

**Interfaces:**
- Consumes: `_cite_keys` and `split_into_sentences` from `agent5_batch_citer` (imported, not copied — the audit's sentence boundaries must match the citer's).
- Produces:
  - `invert_citation_mapping(mapping: dict) -> dict` — `{source: key}` → `{key: source}`
  - `strip_citations(sentence: str) -> str`
  - `resolve_documents(collection, citation_source: str, cache: dict) -> set[str]`
  - `citation_pairs(sentences: list[str], key_to_source: dict) -> list[dict]` — one entry per `(sentence, cite_key)` with keys `sentence_index`, `sentence`, `claim`, `cite_key`, `citation_source`, `outcome`. `outcome` is `"orphaned"` when the key is absent from the mapping, otherwise `None` for Task 4 to fill in.

This task is pure logic — no LLM, no ChromaDB. That is what makes it testable with fakes.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_verifier.py`:

```python
"""Tests for agent 8's resolution logic.

Everything here is pure: mapping inversion, cite stripping, and document
resolution against a fake collection. No network, no ChromaDB, no model.
"""

import unittest

from research_assistant.agents.agent8_verifier import (
    citation_pairs,
    invert_citation_mapping,
    resolve_documents,
    strip_citations,
)


class FakeCollection:
    """Stands in for a ChromaDB collection's .get(where=...)."""

    def __init__(self, rows):
        # rows: list of (citation_source, document)
        self.rows = rows
        self.calls = []

    def get(self, where=None, include=None):
        self.calls.append(where)
        source = where["citation_source"]
        metas = [{"document": doc} for src, doc in self.rows if src == source]
        return {"ids": [str(i) for i in range(len(metas))], "metadatas": metas}


class TestInvertCitationMapping(unittest.TestCase):
    def test_inverts_source_to_key(self):
        self.assertEqual(
            invert_citation_mapping({"Smith 2020": "cite_1", "Jones 2019": "cite_2"}),
            {"cite_1": "Smith 2020", "cite_2": "Jones 2019"},
        )

    def test_empty_mapping(self):
        self.assertEqual(invert_citation_mapping({}), {})

    def test_two_sources_sharing_a_key_keeps_one(self):
        """Shouldn't happen — agent 5 mints a key per source — but inversion
        must not raise if it ever does."""
        inverted = invert_citation_mapping({"Smith 2020": "cite_1", "Jones 2019": "cite_1"})
        self.assertEqual(set(inverted), {"cite_1"})
        self.assertIn(inverted["cite_1"], {"Smith 2020", "Jones 2019"})


class TestStripCitations(unittest.TestCase):
    def test_removes_single_cite(self):
        self.assertEqual(
            strip_citations("Graphene conducts well \\cite{cite_1}."),
            "Graphene conducts well.",
        )

    def test_removes_multi_key_cite(self):
        self.assertEqual(
            strip_citations("Graphene conducts well \\cite{cite_1,cite_2}."),
            "Graphene conducts well.",
        )

    def test_removes_several_cites_in_one_sentence(self):
        self.assertEqual(
            strip_citations("A \\cite{cite_1} and B \\cite{cite_2} differ."),
            "A and B differ.",
        )

    def test_sentence_without_cites_is_unchanged(self):
        self.assertEqual(strip_citations("Nothing here."), "Nothing here.")

    def test_collapses_the_gap_left_behind(self):
        self.assertEqual(strip_citations("A \\cite{x}  B"), "A B")


class TestResolveDocuments(unittest.TestCase):
    def test_returns_documents_for_a_source(self):
        col = FakeCollection([("Smith 2020", "a.pdf"), ("Smith 2020", "a.pdf"),
                              ("Jones 2019", "b.pdf")])
        self.assertEqual(resolve_documents(col, "Smith 2020", {}), {"a.pdf"})

    def test_unknown_source_returns_empty_set(self):
        col = FakeCollection([("Smith 2020", "a.pdf")])
        self.assertEqual(resolve_documents(col, "Nobody 1999", {}), set())

    def test_result_is_cached_across_calls(self):
        """A source cited twenty times must cost one Chroma call, not twenty."""
        col = FakeCollection([("Smith 2020", "a.pdf")])
        cache = {}
        resolve_documents(col, "Smith 2020", cache)
        resolve_documents(col, "Smith 2020", cache)
        self.assertEqual(len(col.calls), 1)

    def test_empty_result_is_cached_too(self):
        col = FakeCollection([])
        cache = {}
        resolve_documents(col, "Nobody 1999", cache)
        resolve_documents(col, "Nobody 1999", cache)
        self.assertEqual(len(col.calls), 1)


class TestCitationPairs(unittest.TestCase):
    def test_one_pair_per_cited_sentence(self):
        pairs = citation_pairs(
            ["Uncited sentence.", "Cited one \\cite{cite_1}."],
            {"cite_1": "Smith 2020"},
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["sentence_index"], 1)
        self.assertEqual(pairs[0]["cite_key"], "cite_1")
        self.assertEqual(pairs[0]["citation_source"], "Smith 2020")
        self.assertEqual(pairs[0]["claim"], "Cited one.")
        self.assertIsNone(pairs[0]["outcome"])

    def test_two_sources_in_one_sentence_yield_two_pairs(self):
        pairs = citation_pairs(
            ["Both agree \\cite{cite_1,cite_2}."],
            {"cite_1": "Smith 2020", "cite_2": "Jones 2019"},
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual({p["cite_key"] for p in pairs}, {"cite_1", "cite_2"})
        # Both judge the same claim, with the LaTeX removed.
        self.assertEqual({p["claim"] for p in pairs}, {"Both agree."})

    def test_key_absent_from_mapping_is_orphaned(self):
        """Agent 5 warns about invented keys; this is where they surface."""
        pairs = citation_pairs(["Invented \\cite{cite_9}."], {"cite_1": "Smith 2020"})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["outcome"], "orphaned")
        self.assertIsNone(pairs[0]["citation_source"])

    def test_pairs_are_ordered_by_sentence_then_key(self):
        pairs = citation_pairs(
            ["B \\cite{cite_2}.", "A \\cite{cite_1}."],
            {"cite_1": "Jones 2019", "cite_2": "Smith 2020"},
        )
        self.assertEqual([p["sentence_index"] for p in pairs], [0, 1])

    def test_no_citations_yields_nothing(self):
        self.assertEqual(citation_pairs(["Plain text."], {"cite_1": "Smith 2020"}), [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_verifier.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'research_assistant.agents.agent8_verifier'`

- [ ] **Step 3: Implement**

Create `research_assistant/agents/agent8_verifier.py`:

```python
"""
Agent 8 — Verifier.

Audits an already-cited draft. For every \\cite{key} Agent 5 inserted, this
re-retrieves the best-matching chunk from the cited source and asks the
judgement prompt whether that evidence actually supports the claim.

It reports; it does not gate. A citation Agent 5 got wrong is still in the
draft — this flags it, with a slot-level verdict and a verbatim span, and
leaves the correction to the author. Making it a gate would put a
per-sentence LLM call back in Agent 5's hot path, which is the cost Agent 5's
batched citation-need check exists to avoid.

Evidence is re-retrieved rather than replayed because Agent 5 does not persist
the chunk text it cited — only the source metadata. The question this answers
is therefore "does this source's strongest evidence for this claim support
it?", which fails in the safe direction: it cannot manufacture support the
source does not contain.
"""

import re

from research_assistant.agents.agent5_batch_citer import (
    _cite_keys,
    split_into_sentences,
)
from research_assistant.shared.log import get_logger

logger = get_logger("agent8")

# " \cite{x}." must become "." and not " .", and the gap the removal leaves
# behind must collapse — the claim is handed to a model as prose.
_CITE_RE = re.compile(r"\\cite\{[^}]*\}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_MULTI_SPACE_RE = re.compile(r"\s+")


def invert_citation_mapping(mapping: dict) -> dict:
    """``{citation_source: cite_key}`` → ``{cite_key: citation_source}``.

    Agent 5 writes _citations.json keyed by source because that is how it
    builds the registry; every consumer here starts from a key found in the
    draft, so the mapping is inverted once up front.
    """
    return {key: source for source, key in mapping.items()}


def strip_citations(sentence: str) -> str:
    """The sentence as prose, with every LaTeX citation removed."""
    text = _CITE_RE.sub("", sentence)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def resolve_documents(collection, citation_source: str, cache: dict) -> set:
    """Documents whose chunks carry *citation_source*.

    hybrid_search filters on ``metadata["document"]``, but a citation key
    resolves to a ``citation_source``. This is the bridge between the two.
    Cached because a source cited twenty times is one lookup, not twenty.
    """
    if citation_source in cache:
        return cache[citation_source]

    try:
        batch = collection.get(
            where={"citation_source": citation_source}, include=["metadatas"]
        )
        documents = {
            meta.get("document")
            for meta in (batch.get("metadatas") or [])
            if meta and meta.get("document")
        }
    except Exception as exc:
        logger.warning("Could not resolve source %r: %s", citation_source, exc)
        documents = set()

    cache[citation_source] = documents
    return documents


def citation_pairs(sentences: list, key_to_source: dict) -> list:
    """One record per (sentence, cited source) pair, in draft order.

    A sentence citing three sources is judged three times: each source is a
    separate claim about a separate paper.
    """
    pairs = []
    for index, sentence in enumerate(sentences):
        keys = _cite_keys(sentence)
        if not keys:
            continue
        claim = strip_citations(sentence)
        for key in sorted(keys):
            source = key_to_source.get(key)
            pairs.append({
                "sentence_index": index,
                "sentence": sentence,
                "claim": claim,
                "cite_key": key,
                "citation_source": source,
                # Agent 5 already warns when the model invents a key; this is
                # where that shows up per sentence instead of once per run.
                "outcome": None if source else "orphaned",
            })
    return pairs
```

- [ ] **Step 4: Run the tests**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_verifier.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add research_assistant/agents/agent8_verifier.py tests/test_verifier.py
git commit -m "feat(agent8): add verifier resolution logic"
```

---

### Task 4: `verify_draft()`, reports, and the CLI

**Files:**
- Modify: `research_assistant/agents/agent8_verifier.py`
- Test: `tests/test_verifier.py`

**Interfaces:**
- Consumes: everything from Task 3; `judge` and `JudgementParseError` from Task 2; `load_search_resources()` from `shared/db.py` returning `(collection, bm25, texts, metadatas)`; `hybrid_search(query, collection, bm25, texts, metadatas, top_k=…, doc_filter=…) -> list[dict]` with keys `chunk_index`, `text`, `metadata`, `rrf_score`; `atomic_write_json` and `atomic_write`.
- Produces: `verify_draft(draft_path, citations_path=None, top_k=None, search_resources=None) -> dict` and a `main()` CLI. The returned dict is exactly what lands in `_verification.json`.

Outcome vocabulary — every pair ends with exactly one, and every one appears in the report:

| `outcome` | Meaning |
|---|---|
| `judged` | A verdict was produced. `judgement` and the other judgement fields are populated. |
| `orphaned` | The cite key is not in `_citations.json`. Agent 5's "model invented a key" case. |
| `unresolved` | The source resolves to no documents — the corpus changed since the draft was cited. |
| `no_evidence` | Retrieval returned nothing from that source for this claim. |
| `retrieval_failed` | `hybrid_search` raised — embedding backends make a network call. `error_type` and `raw` carry the exception. |
| `parse_failed` | Two attempts both produced an unusable reply. `raw` carries the last one. |
| `call_failed` | The judging call raised something other than `JudgementParseError` — a transport, timeout or HTTP error. `error_type` and `raw` carry the exception. |

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_verifier.py`:

```python
import json
import os
import tempfile
from unittest.mock import patch

from research_assistant.agents.agent8_verifier import verify_draft
from research_assistant.judgement.judge import JudgementParseError


def _verdict(judgement="Supports"):
    return {
        "slots": {
            "finding": {"assertion": "a", "verdict": "Supports"},
            "scope": {"assertion": "b", "verdict": "Supports"},
            "strength": {"assertion": "c", "verdict": "Not applicable"},
        },
        "judgement": judgement,
        "evidence_sufficiency": "sufficient",
        "confidence": "High",
        "supporting_span": "span",
        "reason": "because",
    }


class _Resources:
    """The (collection, bm25, texts, metadatas) tuple load_search_resources returns."""

    def __init__(self, rows, hits):
        self.collection = FakeCollection(rows)
        self.hits = hits

    def as_tuple(self):
        return (self.collection, None, [], [])


class VerifyDraftTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, draft, mapping):
        draft_path = os.path.join(self.tmp.name, "cited_draft.txt")
        with open(draft_path, "w", encoding="utf-8") as fh:
            fh.write(draft)
        with open(os.path.join(self.tmp.name, "cited_draft_citations.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(mapping, fh)
        return draft_path

    def _run(self, draft, mapping, rows, hits, judge_side_effect=None):
        draft_path = self._write(draft, mapping)
        res = _Resources(rows, hits)
        with patch("research_assistant.agents.agent8_verifier.hybrid_search",
                   return_value=hits), \
             patch("research_assistant.agents.agent8_verifier.judge",
                   side_effect=judge_side_effect or (lambda c, e, **k: _verdict())):
            return draft_path, verify_draft(draft_path, search_resources=res.as_tuple())


class TestVerifyDraft(VerifyDraftTestCase):
    def test_happy_path_judges_the_citation(self):
        _, report = self._run(
            "Graphene conducts well \\cite{cite_1}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "Graphene is highly conductive.", "metadata": {"document": "a.pdf"}}],
        )
        self.assertEqual(len(report["results"]), 1)
        entry = report["results"][0]
        self.assertEqual(entry["outcome"], "judged")
        self.assertEqual(entry["judgement"], "Supports")
        self.assertEqual(entry["evidence"], "Graphene is highly conductive.")

    def test_missing_citations_file_raises(self):
        draft_path = os.path.join(self.tmp.name, "cited_draft.txt")
        with open(draft_path, "w", encoding="utf-8") as fh:
            fh.write("Text \\cite{cite_1}.")
        with self.assertRaises(FileNotFoundError):
            verify_draft(draft_path, search_resources=(FakeCollection([]), None, [], []))

    def test_orphaned_key_is_not_judged(self):
        _, report = self._run(
            "Invented \\cite{cite_9}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")], [],
        )
        self.assertEqual(report["results"][0]["outcome"], "orphaned")
        self.assertEqual(report["totals"]["orphaned"], 1)

    def test_source_with_no_documents_is_unresolved(self):
        _, report = self._run(
            "Claim \\cite{cite_1}.", {"Ghost 1999": "cite_1"}, [], [],
        )
        self.assertEqual(report["results"][0]["outcome"], "unresolved")

    def test_empty_retrieval_is_no_evidence(self):
        _, report = self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")], [],
        )
        self.assertEqual(report["results"][0]["outcome"], "no_evidence")

    def test_parse_failure_is_recorded_and_does_not_abort(self):
        """One malformed reply must not cost a 120-citation run."""
        calls = {"n": 0}

        def flaky(claim, evidence, **kwargs):
            calls["n"] += 1
            if "first" in claim:
                raise JudgementParseError("bad", raw="garbage")
            return _verdict()

        _, report = self._run(
            "The first claim \\cite{cite_1}. The second claim \\cite{cite_1}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
            judge_side_effect=flaky,
        )
        outcomes = [r["outcome"] for r in report["results"]]
        self.assertEqual(outcomes, ["parse_failed", "judged"])
        self.assertEqual(report["results"][0]["raw"], "garbage")

    def test_both_output_files_are_written(self):
        draft_path, _ = self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
        )
        base = draft_path.replace(".txt", "")
        self.assertTrue(os.path.exists(base + "_verification.json"))
        self.assertTrue(os.path.exists(base + "_verification.md"))

    def test_totals_count_every_category(self):
        _, report = self._run(
            "A \\cite{cite_1}. B \\cite{cite_9}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
        )
        self.assertEqual(report["totals"]["judged"], 1)
        self.assertEqual(report["totals"]["orphaned"], 1)
        self.assertEqual(report["totals"]["total"], 2)

    def test_report_lists_worst_verdicts_first(self):
        verdicts = iter([_verdict("Supports"), _verdict("Contradicts")])
        draft_path, _ = self._run(
            "Fine \\cite{cite_1}. Wrong \\cite{cite_2}.",
            {"Smith 2020": "cite_1", "Jones 2019": "cite_2"},
            [("Smith 2020", "a.pdf"), ("Jones 2019", "b.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
            judge_side_effect=lambda c, e, **k: next(verdicts),
        )
        markdown = open(draft_path.replace(".txt", "_verification.md"),
                        encoding="utf-8").read()
        # The flagged sentence must appear before the clean one. Asserting on
        # the word "Contradicts" instead would pass trivially — it is also a
        # row label in the summary table at the top.
        self.assertLess(markdown.index("Wrong"), markdown.index("Fine"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_verifier.py -v -k "VerifyDraft"`
Expected: FAIL — `ImportError: cannot import name 'verify_draft'`

- [ ] **Step 3: Implement `verify_draft` and the writers**

Append to `research_assistant/agents/agent8_verifier.py`. Add these imports at the top of the file, below the existing ones:

```python
import argparse
import json
import os
from datetime import datetime

from research_assistant.config import JUDGEMENT_TOP_K, LLM_BACKEND, LLM_MODEL
from research_assistant.judgement.judge import JudgementParseError, judge
from research_assistant.shared.atomic import atomic_write, atomic_write_json
from research_assistant.shared.retry import retry
from research_assistant.shared.search import hybrid_search
```

`load_search_resources` is imported inside `verify_draft`, not at module scope, because it pulls in `chromadb` and `tests/test_verifier.py` must run without it.

Then:

```python
# Worst first: the reader is looking for citations to fix, and a report that
# opens with 90 lines of "Supports" buries them.
_SEVERITY = {
    "Contradicts": 0,
    "Does not support": 1,
    "Unclear / insufficient evidence": 2,
    "Partially supports": 3,
    "Supports": 4,
}

# Two attempts: shared.retry counts attempts, not retries.
_JUDGE_ATTEMPTS = 2


@retry(max_retries=_JUDGE_ATTEMPTS, backoff=1.0)
def _judge_once(claim, evidence):
    return judge(claim, evidence)


def _warn_if_context_is_tight():
    """A local model with a small context truncates the prompt silently."""
    if LLM_BACKEND != "ollama":
        return
    from research_assistant.config import JUDGEMENT_OLLAMA_OPTIONS
    from research_assistant.judgement.judge import PROMPT_TEMPLATE

    approx_tokens = len(PROMPT_TEMPLATE) // 4
    logger.warning(
        "Judging with local model %s. The prompt alone is ~%d tokens and "
        "num_ctx is set to %s — if the model cannot honour that, Ollama "
        "truncates the prompt tail (where the worked examples are) without "
        "raising, and verdicts degrade silently.",
        LLM_MODEL, approx_tokens, JUDGEMENT_OLLAMA_OPTIONS.get("num_ctx"),
    )


def verify_draft(draft_path, citations_path=None, top_k=None,
                 search_resources=None) -> dict:
    """Judge every citation in *draft_path* against its cited source.

    Writes ``<draft>_verification.json`` and ``<draft>_verification.md``
    alongside the draft, matching agent 5's output naming.

    Returns:
        dict: the same record written to the JSON file.
    """
    if citations_path is None:
        citations_path = draft_path.replace(".txt", "_citations.json")

    if not os.path.exists(citations_path):
        raise FileNotFoundError(
            f"No citation mapping at {citations_path}. Agent 8 needs it to "
            "resolve \\cite keys back to sources — run Agent 5 first."
        )

    with open(draft_path, encoding="utf-8") as fh:
        draft_text = fh.read()
    with open(citations_path, encoding="utf-8") as fh:
        mapping = json.load(fh)

    if search_resources is None:
        from research_assistant.shared.db import load_search_resources
        search_resources = load_search_resources()
    collection, bm25, texts, metadatas = search_resources

    top_k = top_k or JUDGEMENT_TOP_K
    sentences = split_into_sentences(draft_text)
    pairs = citation_pairs(sentences, invert_citation_mapping(mapping))

    if pairs:
        logger.info(
            "Verifying %d citation(s) across %d sentence(s) — one model call each.",
            len(pairs), len(sentences),
        )
        _warn_if_context_is_tight()

    doc_cache = {}
    results = []

    for i, pair in enumerate(pairs, 1):
        entry = dict(pair)
        logger.info("[%d/%d] %s", i, len(pairs), entry["claim"][:80])

        if entry["outcome"] == "orphaned":
            logger.info(" -> key %s is not in the mapping.", entry["cite_key"])
            results.append(entry)
            continue

        documents = resolve_documents(collection, entry["citation_source"], doc_cache)
        if not documents:
            entry["outcome"] = "unresolved"
            logger.info(" -> source resolves to no documents in the corpus.")
            results.append(entry)
            continue

        hits = hybrid_search(
            entry["claim"], collection, bm25, texts, metadatas,
            top_k=top_k, doc_filter=documents,
        )
        if not hits:
            entry["outcome"] = "no_evidence"
            logger.info(" -> nothing retrieved from that source for this claim.")
            results.append(entry)
            continue

        entry["evidence"] = hits[0]["text"]
        try:
            verdict = _judge_once(entry["claim"], entry["evidence"])
        except JudgementParseError as exc:
            entry["outcome"] = "parse_failed"
            entry["raw"] = exc.raw
            logger.warning(" -> unusable reply after %d attempts.", _JUDGE_ATTEMPTS)
            results.append(entry)
            continue
        except Exception as exc:
            entry["outcome"] = "parse_failed"
            entry["raw"] = str(exc)
            logger.warning(" -> judging failed: %s", exc)
            results.append(entry)
            continue

        entry["outcome"] = "judged"
        entry.update(verdict)
        logger.info(" -> %s (%s confidence)", verdict["judgement"], verdict["confidence"])
        results.append(entry)

    report = {
        "draft": os.path.abspath(draft_path),
        "citations": os.path.abspath(citations_path),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": LLM_MODEL,
        "top_k": top_k,
        "results": results,
        "totals": _totals(results),
    }

    json_path = draft_path.replace(".txt", "_verification.json")
    atomic_write_json(json_path, report)
    logger.info("Saved verification record to %s", json_path)

    md_path = draft_path.replace(".txt", "_verification.md")
    _write_markdown(md_path, report)
    logger.info("Saved verification report to %s", md_path)

    return report


def _totals(results) -> dict:
    totals = {
        "total": len(results),
        "judged": 0, "orphaned": 0, "unresolved": 0,
        "no_evidence": 0, "parse_failed": 0,
    }
    for judgement in _SEVERITY:
        totals[judgement] = 0
    for entry in results:
        totals[entry["outcome"]] = totals.get(entry["outcome"], 0) + 1
        if entry["outcome"] == "judged":
            totals[entry["judgement"]] = totals.get(entry["judgement"], 0) + 1
    return totals


def _write_markdown(path, report) -> None:
    totals = report["totals"]
    lines = [
        f"# Verification Report — `{os.path.basename(report['draft'])}`",
        f"*Generated {report['generated']} · model `{report['model']}`*\n",
        "## Summary\n",
        "| Outcome | Count |",
        "|---------|-------|",
        f"| Citations checked | {totals['total']} |",
        f"| Judged | {totals['judged']} |",
        f"| **Contradicts** | **{totals.get('Contradicts', 0)}** |",
        f"| **Does not support** | **{totals.get('Does not support', 0)}** |",
        f"| Unclear / insufficient evidence | {totals.get('Unclear / insufficient evidence', 0)} |",
        f"| Partially supports | {totals.get('Partially supports', 0)} |",
        f"| Supports | {totals.get('Supports', 0)} |",
        f"| Key not in mapping | {totals['orphaned']} |",
        f"| Source not in corpus | {totals['unresolved']} |",
        f"| No evidence retrieved | {totals['no_evidence']} |",
        f"| Unusable model reply | {totals['parse_failed']} |",
        "",
        "---\n",
    ]

    judged = [e for e in report["results"] if e["outcome"] == "judged"]
    other = [e for e in report["results"] if e["outcome"] != "judged"]
    judged.sort(key=lambda e: (_SEVERITY.get(e["judgement"], 9), e["sentence_index"]))

    flagged = [e for e in judged if e["judgement"] != "Supports"]
    clean = [e for e in judged if e["judgement"] == "Supports"]

    if flagged:
        lines.append("## Citations to review\n")
        for entry in flagged:
            lines.extend(_entry_block(entry))

    if other:
        lines.append("## Not judged\n")
        for entry in other:
            lines.append(f"### Sentence {entry['sentence_index'] + 1} — {entry['outcome']}\n")
            lines.append(f"> {entry['sentence']}\n")
            lines.append(f"Key `{entry['cite_key']}`"
                         + (f" → {entry['citation_source']}" if entry["citation_source"] else "")
                         + "\n")
            if entry.get("raw"):
                lines.append("<details><summary>Raw reply</summary>\n")
                lines.append(f"```\n{entry['raw'][:2000]}\n```\n")
                lines.append("</details>\n")
            lines.append("---\n")

    if clean:
        lines.append("## Verified\n")
        for entry in clean:
            lines.append(
                f"- Sentence {entry['sentence_index'] + 1} — `{entry['cite_key']}` "
                f"({entry['confidence']} confidence, evidence "
                f"{entry['evidence_sufficiency']}): {entry['sentence']}"
            )
        lines.append("")

    with atomic_write(path) as fh:
        fh.write("\n".join(lines))


def _entry_block(entry) -> list:
    slots = entry.get("slots", {})
    block = [
        f"### Sentence {entry['sentence_index'] + 1} — {entry['judgement']}\n",
        f"> {entry['sentence']}\n",
        f"**Cited source:** {entry['citation_source']} (`{entry['cite_key']}`)\n",
        f"**Confidence:** {entry['confidence']} · "
        f"**Evidence sufficiency:** {entry['evidence_sufficiency']}\n",
        "| Slot | Assertion | Verdict |",
        "|------|-----------|---------|",
    ]
    for name in ("finding", "scope", "strength"):
        slot = slots.get(name, {})
        block.append(
            f"| `{name}` | {slot.get('assertion', '—')} | {slot.get('verdict', '—')} |"
        )
    block.append("")
    if entry.get("supporting_span"):
        block.append(f"**Supporting span:**\n> {entry['supporting_span']}\n")
    block.append(f"**Reason:** {entry['reason']}\n")
    block.append("---\n")
    return block


def main():
    parser = argparse.ArgumentParser(
        description="Agent 8 — verify the citations in an already-cited draft."
    )
    parser.add_argument("--draft", required=True, help="Path to the cited draft (.txt).")
    parser.add_argument("--citations", default=None,
                        help="Path to _citations.json. Defaults to the draft's sibling.")
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"Chunks judged per source (default {JUDGEMENT_TOP_K}).")
    args = parser.parse_args()

    report = verify_draft(args.draft, args.citations, args.top_k)
    totals = report["totals"]
    flagged = sum(
        totals.get(j, 0)
        for j in ("Contradicts", "Does not support", "Unclear / insufficient evidence")
    )
    logger.info(
        "Checked %d citation(s): %d judged, %d need review.",
        totals["total"], totals["judged"], flagged,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_verifier.py -v`
Expected: PASS (26 tests)

- [ ] **Step 5: Run the whole suite**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/ -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Verify the CLI's help renders**

Run: `CITATION_LOG_FILE=0 python -m research_assistant.agents.agent8_verifier --help`
Expected: usage text listing `--draft`, `--citations`, `--top-k`.

- [ ] **Step 7: Commit**

```bash
git add research_assistant/agents/agent8_verifier.py tests/test_verifier.py
git commit -m "feat(agent8): add verify_draft, reports, and CLI"
```

---

### Task 5: The Verify button in `tab_batch`

**Files:**
- Modify: `app.py:586-650` (the `with tab_batch:` block, after the existing report expander)

**Interfaces:**
- Consumes: `verify_draft(draft_path) -> dict` from Task 4.
- Produces: no new API.

The button goes in `tab_batch` rather than a sixth tab because it only ever operates on that tab's output; separating them would split one workflow across two screens.

- [ ] **Step 1: Read the surrounding block**

Run: `sed -n '586,655p' app.py`

Note how `written` (the draft path) is held, and how the existing mapping/report blocks guard with `os.path.exists`. Match that shape.

- [ ] **Step 2: Add the import**

At the top of `app.py`, beside the other agent imports:

```python
from research_assistant.agents.agent8_verifier import verify_draft
```

- [ ] **Step 3: Add the button**

Immediately after the existing `report_path` expander block inside `with tab_batch:`, at the same indentation:

```python
        st.divider()
        st.caption(
            "Verification re-checks every inserted citation against the source "
            "it cites — one model call per citation, so a long draft is not free."
        )
        if st.button("Verify citations", key="verify_citations"):
            with st.status("Judging each citation…", expanded=True) as status:
                try:
                    verification = verify_draft(written)
                except FileNotFoundError as exc:
                    status.update(label="Verification failed", state="error")
                    st.error(str(exc))
                except Exception as exc:
                    status.update(label="Verification failed", state="error")
                    st.error(f"Verification failed: {exc}")
                else:
                    status.update(label="Verification complete", state="complete")
                    st.session_state[f"verification:{written}"] = verification

        # Named `verification`, not `report`: a few lines above, `report` is
        # already bound to the text of agent 5's _report.md in this same block.
        #
        # Keyed by `written` (the current draft's path) rather than a bare
        # "verification" key: a fresh draft gets a fresh tempfile path every
        # run, so a result cached under another draft's key is never looked up
        # here, and a failed run for this draft simply never populates this
        # draft's key. Either way, nothing renders that wasn't computed for
        # this exact draft.
        verification = st.session_state.get(f"verification:{written}")
        if verification:
            totals = verification["totals"]
            needs_review = sum(
                totals.get(j, 0)
                for j in ("Contradicts", "Does not support",
                          "Unclear / insufficient evidence")
            )
            cols = st.columns(4)
            cols[0].metric("Checked", totals["total"])
            cols[1].metric("Supports", totals.get("Supports", 0))
            cols[2].metric("Partially", totals.get("Partially supports", 0))
            cols[3].metric("Need review", needs_review)

            md_path = written.replace(".txt", "_verification.md")
            if os.path.exists(md_path):
                with open(md_path, encoding="utf-8") as fh:
                    with st.expander("Full verification report", expanded=needs_review > 0):
                        st.markdown(fh.read())
```

The `encoding="utf-8"` matches the two blocks directly above it, which already pass it.

- [ ] **Step 4: Verify the app still imports**

Run: `CITATION_LOG_FILE=0 python -c "import ast, sys; ast.parse(open('app.py').read()); print('parses')"`
Expected: `parses`

Then, matching CI's import sweep:

Run: `CITATION_LOG_FILE=0 python -c "import app" 2>&1 | tail -5`
Expected: no `ImportError` or `NameError`. Streamlit warnings about bare mode are fine.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(app): add citation verification to the batch tab"
```

---

### Task 6: Documentation

**Files:**
- Modify: `README.md` (pipeline table), `ARCHITECTURE.md` (new section), `HOW_TO_USE.md` (reading a report)

- [ ] **Step 1: Add the Agent 8 row to README's pipeline table**

After the Agent 7 row:

```markdown
| **Agent 8 — Verifier** | [agent8_verifier.py](research_assistant/agents/agent8_verifier.py) | Audits an already-cited draft. For every `\cite{key}` Agent 5 inserted, it re-retrieves the best-matching chunk from that source and asks a claim–evidence rubric whether the evidence actually supports the claim — decomposing the claim into `finding` / `scope` / `strength`, verdicting each, and aggregating into `Supports` / `Partially supports` / `Contradicts` / `Does not support` / `Unclear`. It reports rather than gates: a wrong citation stays in the draft and is flagged, with a slot table and a verbatim supporting span. Separately rates `evidence_sufficiency`, which distinguishes "the paper does not support this" from "retrieval did not return enough to tell". Writes `_verification.json` and `_verification.md`. |
```

- [ ] **Step 2: Add the ARCHITECTURE section**

Insert a new numbered section before *Things deliberately not done*, renumbering the sections after it:

```markdown
## 9. Verification (Agent 8)

### 9.1 An audit pass, not an inline gate

Agent 8 runs after Agent 5, over its output. Agents 4 and 5 are unchanged.

**Why.** Judgement is one claim–evidence pair at a time; the slot
decomposition is what makes it accurate and it cannot be batched without
losing that. Agent 5 already batches its citation-need check
(`CITATION_CHECK_BATCH_SIZE = 20`) because per-sentence calls were too
expensive, and putting judgement inline on every retrieved candidate roughly
triples a batch run's call count — acceptable against a hosted API,
impractical against a local CPU model.

**The trade-off.** A wrong citation is inserted and then flagged rather than
blocked, and the draft is not corrected automatically. Making judgement a gate
is a coherent future change; it is not this one.

### 9.2 Evidence is re-retrieved, not replayed

Agent 5 does not persist the chunk text it cited — `citation_entries` records
only source metadata, and `_report.md` is prose. So the evidence behind a
citation is not recoverable from Agent 5's output, and Agent 8 re-runs
`hybrid_search` restricted to the cited source.

**Why that is acceptable.** The question becomes "does this source's strongest
evidence for this claim support it?" If even the best chunk fails, the citation
is wrong regardless of what Agent 5 saw. The failure runs in the safe
direction: re-retrieval cannot manufacture support the source does not
contain. The report names the chunk that was judged, so the reader can see
which text the verdict rests on.

### 9.3 The prompt is code

`judgement/prompt.md` is a 259-line rubric whose aggregation order and
scope-versus-contradiction distinction are load-bearing, and no unit test
covers them. `judgement/cases/cases.jsonl` is the only guard. Run the live
suite (`RUN_LLM_TESTS=1`) before shipping any prompt edit.
```

- [ ] **Step 3: Add the HOW_TO_USE section**

After the batch-citer section:

```markdown
## Verifying the citations

**Verify citations** in the batch tab re-checks every citation the pipeline
inserted. It costs one model call per citation, so a long draft is not free.

Each citation gets a verdict:

| Verdict | What it means |
|---|---|
| Supports | The source reports what the sentence claims, under the conditions the sentence names. |
| Partially supports | Close, but overstated — a hedged mechanism stated as a cause, or one system generalised to a class. |
| Contradicts | The source reports the opposite result, *inside* the conditions the claim covers. |
| Does not support | The source examined none of what the claim covers. An opposite result under different conditions lands here, not in Contradicts. |
| Unclear / insufficient evidence | The retrieved text cannot decide the question either way. |

**`evidence_sufficiency` is a separate axis, and it is the one to read first.**
It describes the *retrieved text*, not the claim. `insufficient` means
retrieval did not return enough to judge — often because the result lives in a
supplementary table or a figure that was never ingested. That is a retrieval
gap to fix, not a citation to delete. A verdict of "Unclear" with
`evidence_sufficiency: insufficient` means "ingest more of this paper", while
"Does not support" with `sufficient` means "this citation is wrong".
```

- [ ] **Step 4: Check the renumbering is consistent**

Run: `grep -n "^## [0-9]" ARCHITECTURE.md`
Expected: a gapless ascending sequence. Fix any cross-references with `grep -n "§[0-9]" ARCHITECTURE.md README.md`.

- [ ] **Step 5: Commit**

```bash
git add README.md ARCHITECTURE.md HOW_TO_USE.md
git commit -m "docs: document agent 8 verification"
```

---

## Final verification

- [ ] Full suite green: `CITATION_LOG_FILE=0 python -m pytest tests/ -v`
- [ ] Import sweep, matching `.github/workflows/tests.yml`:

```bash
CITATION_LOG_FILE=0 python -c "
import research_assistant.judgement.judge
import research_assistant.agents.agent8_verifier
import app, orchestrate, watch
print('all modules import')
"
```

- [ ] No provider SDK leaked outside `shared/llm.py`:

```bash
grep -rn "^import openai\|^import ollama\|^from openai\|^from ollama" --include="*.py" research_assistant/ app.py orchestrate.py watch.py | grep -v "shared/llm.py"
```

Expected: no output.

- [ ] Agents 4 and 5 untouched: `git diff --stat main -- research_assistant/agents/agent4_assistant.py research_assistant/agents/agent5_batch_citer.py` → empty.
- [ ] `shared/search.py` untouched: `git diff --stat main -- research_assistant/shared/search.py` → empty.
- [ ] Live regression suite, once, before calling it done:

```bash
RUN_LLM_TESTS=1 CITATION_LOG_FILE=0 python -m pytest tests/test_judgement.py -v -k Live
```

Expected: all six cases reach their expected judgement. If some do not, report which — do not edit `prompt.md` or `cases.jsonl` to make them pass without saying so first.
