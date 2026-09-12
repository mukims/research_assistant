# Judgement Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Agent 8's verdicts honest and cheaper: claims stop arriving merged, the rubric is enforced in code, the supporting span is verified, the rubric becomes a cacheable prompt prefix (if measurement says it pays), and the verifier judges what it retrieved and looks wider when its own verdict says it saw too little.

**Architecture:** Four contained changes to existing modules. Agent 5 restores terminal punctuation after `\cite{}`. `judgement/judge.py` gains a pure `enforce_rubric()` pass that derives the aggregate from the slots, records disagreement, and verifies the span. `prompt.md` is reordered (input block last) behind a measured gate. `agent8_verifier.py` assembles evidence from the top hit plus neighbours and escalates once to the top-3 when the first verdict reports insufficient evidence. Every new field is additive; no existing field is renamed.

**Tech Stack:** Python 3.10–3.12, pytest collecting `unittest.TestCase`, Ollama (`gemma4:e2b`, `num_ctx` 8192) via `shared.llm.chat`, ChromaDB + `rank_bm25` via `shared.search.hybrid_search`.

**Spec:** `notimportant/superpowers/specs/2026-09-12-judgement-hardening-design.md` — read it first; §2 A–D are the four tasks' authority.

---

## Context you need before Task 1

### How a citation is judged today (the path these tasks change)

```
draft.txt ──► Agent 5 (agents/agent5_batch_citer.py: run_batch_citer)
              split_into_sentences() ──► per sentence: hybrid_search(top_k=3) ──► model writes "CITED: <sentence> \cite{cite_N}"
              cited_sentences joined with " " ──► cited.txt   +   cited_citations.json  {"<paper title>": "cite_N", …}
                                                                                       │
cited.txt ──► Agent 8 (agents/agent8_verifier.py: verify_draft)                      │
              split_into_sentences()  (imported FROM agent5)                          │
              citation_pairs(): one (sentence, cite_key) pair per key; claim = sentence with \cite{} stripped
              resolve_documents(): cite_key → paper title (citation_source) → set of `document` keys in Chroma
              hybrid_search(claim, …, top_k=JUDGEMENT_TOP_K=1, doc_filter=documents, exclude_types={"figure_description"})
              evidence = hits[0]["text"]                       ◄── Task 6 changes this
              judge(claim, evidence)  (judgement/judge.py)      ◄── Task 2 changes what comes back
                 build_prompt(): prompt.md with {{CLAIM}} / {{CITATION_EVIDENCE}} filled   ◄── Task 4 reorders prompt.md
                 chat(model=JUDGEMENT_MODEL or LLM_MODEL, temperature=0.0, options={"num_ctx": 8192})
                 parse_judgement(): JSON → dict, vocab-checked at the top level only
              entry.update(verdict fields) ──► <draft>_verification.json + .md   ◄── Task 3 shows the new flags
```

### The data shapes

`hybrid_search()` returns `list[dict]`, each:
```python
{"chunk_index": int,            # position in the `texts` list == the N in Chroma id "chunk_N"
 "text": str,                   # the stored chunk (v2: ~1,100 chars; v1: ~240)
 "metadata": {"document": "doi_10.1002_adma.202211157.pdf", "citation_source": "<paper title>",
              "type": "text_chunk" | "caption" | "figure_description" | "figure" | "table",
              "page": int, … v2 adds "section", "seq", "page_first", "page_last", "extraction"},
 "rrf_score": float}
```
`texts` and `metadatas` (from `shared.db.load_search_resources()`) are lists aligned by `chunk_index`; a paper's chunks were minted contiguously, so `chunk_index ± 1` with the same `metadata["document"]` is the adjacent chunk of that paper.

`judge()` returns the parsed model reply — today exactly:
```python
{"slots": {"finding":  {"assertion": str, "verdict": str},
           "scope":    {"assertion": str, "verdict": str},
           "strength": {"assertion": str, "verdict": str}},
 "judgement": "Supports" | "Partially supports" | "Contradicts" | "Does not support" | "Unclear / insufficient evidence",
 "evidence_sufficiency": "sufficient" | "partial" | "insufficient",
 "confidence": "High" | "Medium" | "Low",
 "supporting_span": str | None,
 "reason": str}
```
The verifier's per-citation `entry` (one per `results[]` item in the JSON record) carries `sentence_index, sentence, claim, cite_key, citation_source, outcome` plus, when `outcome == "judged"`, `evidence` and every field of the verdict above.

### What went wrong on 2026-09-12 (the evidence behind each task)

Draft (two sentences): *"Covalent networks of MoS2 flakes show enhanced photoconductivity compared to pristine films. Charge transport in these networks is dominated by hopping between flakes at low temperature."*

Agent 5 wrote: `Covalent networks of MoS2 flakes show enhanced photoconductivity compared to pristine films \cite{cite_1} Charge transport in these networks is dominated by hopping between flakes at low temperature \cite{cite_2}` — **no full stops**. Agent 8 saw one sentence, two keys. Its two records:

```
cite_1 → model: Partially supports (Medium, sufficiency=partial)
   finding   Partially supports   ← NOT in the finding vocabulary; validator accepted it
   scope     Supports
   strength  Insufficient
   supporting_span verbatim in evidence: False
   claim: "Covalent networks … pristine films Charge transport in these networks …"   ← two sentences as one claim
cite_2 → model: Does not support (High, sufficiency=insufficient)      ← correct: Agent 5 cited the wrong paper
```

The hopping paragraph that would have satisfied `cite_1`'s "strength" exists in the same paper (Figure 2d, R_hop) one chunk away; `top_k=1` never showed it to the judge.

On the six regression cases (`judgement/cases/cases.jsonl`, which are the prompt's own examples A–F): 6/6 pass with `gemma4:e2b`; case F's `finding` verdict was again out of vocabulary; per-case time 61–123 s on this CPU.

### Environment

- **Interpreter:** `/home/shardul/miniconda3/envs/ml/bin/python` (has chromadb, ollama, lxml). Every `python` below means that interpreter, run from the repo root with `PYTHONPATH=.` when running scripts (the package is installed editable for `main`'s checkout; `PYTHONPATH=.` makes the branch's code win).
- **Tests:** `CITATION_LOG_FILE=0 python -m pytest tests/ -v`. `unittest.TestCase` classes. Live model tests are behind `RUN_LLM_TESTS=1` and are not run in CI.
- **Ollama:** `localhost:11434`, `gemma4:e2b` and `nomic-embed-text` pulled; Ollama 0.30.6. Needed for Tasks 4 and 8 only.
- **Index:** work on `CITATION_INDEX_VERSION=2` for the live tasks (16,006 chunks, ~1,100-char median). Read `data/`; write nothing there except Agent 5/8 outputs into the scratch draft directory you create.
- **Branch:** start from `ingestion-v2` (`git checkout ingestion-v2 && git checkout -b judgement-hardening`). Commit after every task with the message given. **Never push, never merge.**
- **The ingest lock:** if someone is ingesting, `data/ingest.lock` is held and `tests/test_ingestion.py` blocks. Check with the one-liner in Task 8 before running the full suite; deselect that file if the lock is held.

### Pitfalls from the previous build on this repo (read these — both happened)

1. **A lazy import deleted while editing the block around it** → `NameError` at runtime, invisible to the suite and to CI's import sweep. `tests/test_app_names.py` now guards `app.py`; there is no such guard for the agents. When you edit a function, keep its local imports.
2. **Production code changed to satisfy a test that assumed the wrong thing.** If a planned test contradicts what the code demonstrably does for real inputs, fix the test and say so in the report — do not bend production code to a synthetic input.
3. **Generic verification notes** ("verified tabs 2/3/4") were written without the runs having happened. Task 8 asks for the actual outputs; paste them.

## Global Constraints

- **Python 3.10–3.12.** No 3.13-only syntax.
- **No provider SDK outside `shared/llm.py`.** The judge calls `shared.llm.chat`; nothing imports `ollama` or `openai`.
- **The rubric's rules and vocabularies are not changed.** `prompt.md` may be *reordered* (Task 4) and nothing else. `cases.jsonl` is not edited.
- **Every new record field is additive.** Existing fields in `_verification.json` keep their names and meanings; `judgement` remains the field a reader trusts.
- **Heavy imports stay inside functions** (`chromadb`, `fitz`). CI installs only `requirements-test.txt`.
- **`tests/` uses `unittest.TestCase` classes.** Live-model tests go under `RUN_LLM_TESTS=1` skips like `tests/test_judgement.py::TestLiveJudgement`.
- **All file writes go through `shared/atomic.py`.**
- **`figure_description` chunks are never evidence** — the `exclude_types={"figure_description"}` argument on every `hybrid_search` call in Agent 8 stays.
- Run the suite with `CITATION_LOG_FILE=0` so tests do not write into `data/logs/`.
- **Commit after every task; do not push.**

---

### Task 1: The sentence boundary survives citation; Agent 8 flags a merge that didn't

**Files:**
- Modify: `research_assistant/agents/agent5_batch_citer.py:102-142` (`_cite_sentence_with_reasoning`)
- Modify: `research_assistant/agents/agent8_verifier.py:100-124` (`citation_pairs`)
- Test: `tests/test_batch_citer.py`, `tests/test_verifier.py` (extend both)

**Interfaces:**
- Produces: `agent5_batch_citer._restore_terminal_punctuation(original: str, cited: str) -> str`; `_cite_sentence_with_reasoning` returns a cited sentence that ends the way the original did. `agent8_verifier.citation_pairs` entries gain `compound_sentence: bool`; `COMPOUND_WORDS = 40`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_batch_citer.py`:

```python


from research_assistant.agents.agent5_batch_citer import _restore_terminal_punctuation, split_into_sentences


class TestTerminalPunctuationSurvivesCitation(unittest.TestCase):
    """gemma4:e2b drops the full stop when it appends \\cite{}. The draft is
    joined with spaces and Agent 8 splits on [.!?]+whitespace, so a lost stop
    merges two sentences into one claim. Seen 2026-09-12."""

    def test_dropped_full_stop_is_restored_after_the_cite(self):
        self.assertEqual(
            _restore_terminal_punctuation("Films are good.", "Films are good \\cite{cite_1}"),
            "Films are good \\cite{cite_1}.",
        )

    def test_question_mark_is_restored(self):
        self.assertEqual(_restore_terminal_punctuation("Is it so?", "Is it so \\cite{a}"), "Is it so \\cite{a}?")

    def test_already_terminated_after_the_cite_is_unchanged(self):
        self.assertEqual(_restore_terminal_punctuation("Films are good.", "Films are good \\cite{a}."),
                         "Films are good \\cite{a}.")

    def test_terminated_before_a_trailing_cite_is_unchanged(self):
        self.assertEqual(_restore_terminal_punctuation("Films are good.", "Films are good. \\cite{a}"),
                         "Films are good. \\cite{a}")

    def test_multiple_trailing_cites(self):
        self.assertEqual(_restore_terminal_punctuation("X holds.", "X holds \\cite{a} \\cite{b}"),
                         "X holds \\cite{a} \\cite{b}.")

    def test_original_without_terminal_punctuation_is_left_alone(self):
        self.assertEqual(_restore_terminal_punctuation("a heading", "a heading \\cite{a}"), "a heading \\cite{a}")

    def test_trailing_whitespace_is_trimmed(self):
        self.assertEqual(_restore_terminal_punctuation("Done.", "Done \\cite{a}   "), "Done \\cite{a}.")

    def test_the_rejoined_draft_splits_back_into_two_sentences(self):
        a = _restore_terminal_punctuation("First claim here.", "First claim here \\cite{cite_1}")
        b = _restore_terminal_punctuation("Second claim here.", "Second claim here \\cite{cite_2}")
        self.assertEqual(len(split_into_sentences(" ".join([a, b]))), 2)


class TestCiteSentenceKeepsPunctuation(unittest.TestCase):
    def test_model_reply_without_full_stop_is_repaired(self):
        import research_assistant.agents.agent5_batch_citer as a5
        reply = ChatResult(content="CITED: Films are good \\cite{cite_1}\nREASON: because.")
        with patch.object(a5, "chat", return_value=reply):
            cited, reason = a5._cite_sentence_with_reasoning("Films are good.", "--- Context (Cite Key: cite_1) ---\nx")
        self.assertEqual(cited, "Films are good \\cite{cite_1}.")
        self.assertEqual(reason, "because.")
```

(`tests/test_batch_citer.py` already imports `patch` and `ChatResult`; nothing to add.)

Append to `tests/test_verifier.py`:

```python


class TestCompoundSentenceFlag(unittest.TestCase):
    """Two cite keys in one long 'sentence' is the signature of a merged
    boundary. Flag it so a reader can distrust both verdicts."""

    def test_two_keys_and_many_words_are_flagged(self):
        long = " ".join(["word"] * 45)
        pairs = citation_pairs([f"{long} \\cite{{cite_1}} more {long} \\cite{{cite_2}}"], {"cite_1": "A", "cite_2": "B"})
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(p["compound_sentence"] for p in pairs))

    def test_two_keys_in_a_short_sentence_are_not_flagged(self):
        pairs = citation_pairs(["Both agree \\cite{cite_1} \\cite{cite_2}."], {"cite_1": "A", "cite_2": "B"})
        self.assertFalse(any(p["compound_sentence"] for p in pairs))

    def test_one_key_is_never_flagged(self):
        long = " ".join(["word"] * 60)
        pairs = citation_pairs([f"{long} \\cite{{cite_1}}."], {"cite_1": "A"})
        self.assertFalse(pairs[0]["compound_sentence"])
```

(`citation_pairs` is already imported in `tests/test_verifier.py`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_batch_citer.py tests/test_verifier.py -q`
Expected: FAIL — `ImportError: cannot import name '_restore_terminal_punctuation'`; `KeyError: 'compound_sentence'`.

- [ ] **Step 3: Agent 5**

In `research_assistant/agents/agent5_batch_citer.py`, add after `_CITE_RE = re.compile(r"\\cite\{([^}]*)\}")` (line 46):

```python
# One or more \cite{...} groups at the very end of a sentence, with the
# whitespace before them.
_TRAILING_CITES_RE = re.compile(r"(?:\s*\\cite\{[^}]*\})+\s*$")
_TERMINAL = ".!?"


def _restore_terminal_punctuation(original: str, cited: str) -> str:
    """Give the cited sentence back the full stop the model dropped.

    gemma4:e2b writes "… films \\cite{cite_1}" for "… films." — no terminal
    punctuation. run_batch_citer joins sentences with a space, and Agent 8's
    split_into_sentences needs [.!?] before whitespace, so the next sentence
    is swallowed into this one and both citations get judged against a
    two-sentence claim. Restore the original's terminator after the cite.

    Left alone: a cited sentence that already ends in [.!?], or that ends in
    [.!?] immediately before a trailing \\cite{} group ("… films. \\cite{a}"),
    and any original that had no terminator to restore.
    """
    original = original.rstrip()
    if not original or original[-1] not in _TERMINAL:
        return cited
    stripped = cited.rstrip()
    if stripped and stripped[-1] in _TERMINAL:
        return stripped
    core = _TRAILING_CITES_RE.sub("", stripped)
    if core and core[-1] in _TERMINAL:
        return stripped
    return stripped + original[-1]
```

In `_cite_sentence_with_reasoning`, change the final line

```python
    return cited_sentence, reasoning
```
to
```python
    return _restore_terminal_punctuation(sentence, cited_sentence), reasoning
```

- [ ] **Step 4: Agent 8**

In `research_assistant/agents/agent8_verifier.py`, add above `def citation_pairs`:

```python
# A "sentence" carrying two or more cite keys and more words than any real
# sentence has is almost always two sentences whose boundary was lost when
# the citation was inserted. Both verdicts on it are then about the wrong
# claim; say so in the record.
COMPOUND_WORDS = 40
```

and change the `pairs.append({...})` in `citation_pairs` to:

```python
        compound = len(keys) >= 2 and len(claim.split()) > COMPOUND_WORDS
        if compound:
            logger.warning(
                "Sentence %d carries %d citations across %d words — a lost sentence "
                "boundary? Both verdicts will be about the combined claim.",
                index + 1, len(keys), len(claim.split()),
            )
        for key in sorted(keys):
            source = key_to_source.get(key)
            pairs.append({
                "sentence_index": index,
                "sentence": sentence,
                "claim": claim,
                "cite_key": key,
                "citation_source": source,
                "compound_sentence": compound,
                # Agent 5 already warns when the model invents a key; this is
                # where that shows up per sentence instead of once per run.
                "outcome": None if source else "orphaned",
            })
```

(Replace the existing `for key in sorted(keys): … pairs.append({…})` block; the `claim = strip_citations(sentence)` line stays above it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_batch_citer.py tests/test_verifier.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add research_assistant/agents/agent5_batch_citer.py research_assistant/agents/agent8_verifier.py tests/test_batch_citer.py tests/test_verifier.py
git commit -m "fix(cite): restore the terminal punctuation \\cite{} insertion drops; flag compound multi-cite sentences in Agent 8"
```

---

### Task 2: `enforce_rubric()` — the aggregate is derived, the span is verified, violations are recorded

**Files:**
- Modify: `research_assistant/judgement/judge.py` (add constants + functions; call from `judge()`)
- Test: `tests/test_judgement.py` (extend)

**Interfaces:**
- Produces:
  ```python
  SLOT_VOCAB = {"finding": {...}, "scope": {...}, "strength": {...}}     # from the rubric, Step 2
  DERIVED_FIELDS = {"model_judgement", "rubric_mismatch", "rubric_violations", "span_verified"}
  def derive_judgement(slots: dict) -> str          # Step-3 rules; out-of-vocab verdict counts as "Insufficient"
  def span_is_verbatim(span, evidence) -> bool | None   # None when no span
  def enforce_rubric(result: dict, evidence: str) -> dict   # returns the same dict, mutated and returned
  ```
  `judge()` returns `enforce_rubric(parse_judgement(reply), citation_evidence)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_judgement.py`:

```python


import copy


def _reply(finding="Supports", scope="Supports", strength="Not applicable", judgement="Supports",
           span="the sentence", confidence="High", sufficiency="sufficient"):
    return {
        "slots": {"finding": {"assertion": "f", "verdict": finding},
                  "scope": {"assertion": "s", "verdict": scope},
                  "strength": {"assertion": "t", "verdict": strength}},
        "judgement": judgement, "evidence_sufficiency": sufficiency, "confidence": confidence,
        "supporting_span": span, "reason": "r",
    }


class TestDeriveJudgement(unittest.TestCase):
    """The rubric's Step 3, in code, first match wins."""

    def test_scope_does_not_support_beats_everything(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Contradicts", scope="Does not support")["slots"]),
                         "Does not support")

    def test_contradicts(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Contradicts")["slots"]), "Contradicts")

    def test_finding_does_not_support(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Does not support")["slots"]), "Does not support")

    def test_insufficient_finding_or_scope_is_unclear(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Insufficient")["slots"]),
                         "Unclear / insufficient evidence")
        self.assertEqual(judge_mod.derive_judgement(_reply(scope="Insufficient")["slots"]),
                         "Unclear / insufficient evidence")

    def test_all_supports_or_not_applicable_is_supports(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(strength="Supports")["slots"]), "Supports")
        self.assertEqual(judge_mod.derive_judgement(_reply(strength="Not applicable")["slots"]), "Supports")

    def test_partial_scope_or_strength_is_partially_supports(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(scope="Partially supports")["slots"]), "Partially supports")
        self.assertEqual(judge_mod.derive_judgement(_reply(strength="Insufficient")["slots"]), "Partially supports")

    def test_out_of_vocabulary_slot_counts_as_insufficient(self):
        # "Partially supports" is not a finding verdict — seen from gemma4:e2b twice on 2026-09-12.
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Partially supports")["slots"]),
                         "Unclear / insufficient evidence")


class TestSpanIsVerbatim(unittest.TestCase):
    EVIDENCE = "The ﬁlms showed a 10% increase in Δσ_ph — “as expected”.  Next sentence."

    def test_exact_substring(self):
        self.assertTrue(judge_mod.span_is_verbatim("Next sentence.", self.EVIDENCE))

    def test_whitespace_case_ligature_and_quote_differences_are_tolerated(self):
        self.assertTrue(judge_mod.span_is_verbatim('the films showed a 10% increase in Δσ_ph - "as expected".', self.EVIDENCE))

    def test_paraphrase_is_not_verbatim(self):
        self.assertFalse(judge_mod.span_is_verbatim("Films increased by ten percent.", self.EVIDENCE))

    def test_no_span_is_none(self):
        self.assertIsNone(judge_mod.span_is_verbatim(None, self.EVIDENCE))
        self.assertIsNone(judge_mod.span_is_verbatim("", self.EVIDENCE))
        self.assertIsNone(judge_mod.span_is_verbatim("null", self.EVIDENCE))


class TestEnforceRubric(unittest.TestCase):
    def test_clean_reply_passes_through_with_derived_fields(self):
        out = judge_mod.enforce_rubric(_reply(span="the sentence"), "here is the sentence indeed")
        self.assertEqual(out["judgement"], "Supports")
        self.assertEqual(out["model_judgement"], "Supports")
        self.assertFalse(out["rubric_mismatch"])
        self.assertEqual(out["rubric_violations"], [])
        self.assertTrue(out["span_verified"])
        self.assertEqual(out["confidence"], "High")

    def test_model_aggregate_that_breaks_the_rules_is_overridden_and_recorded(self):
        r = _reply(scope="Does not support", judgement="Supports")
        out = judge_mod.enforce_rubric(r, "the sentence")
        self.assertEqual(out["judgement"], "Does not support")
        self.assertEqual(out["model_judgement"], "Supports")
        self.assertTrue(out["rubric_mismatch"])
        self.assertEqual(out["confidence"], "Medium")

    def test_out_of_vocabulary_slot_is_recorded_and_caps_confidence(self):
        r = _reply(finding="Partially supports", judgement="Partially supports")
        out = judge_mod.enforce_rubric(r, "the sentence")
        self.assertEqual(out["rubric_violations"], ["finding: 'Partially supports'"])
        self.assertEqual(out["judgement"], "Unclear / insufficient evidence")
        self.assertTrue(out["rubric_mismatch"])
        self.assertEqual(out["confidence"], "Medium")

    def test_unverified_span_caps_confidence_but_keeps_the_verdict(self):
        out = judge_mod.enforce_rubric(_reply(span="not in there"), "the evidence text")
        self.assertFalse(out["span_verified"])
        self.assertEqual(out["judgement"], "Supports")
        self.assertEqual(out["confidence"], "Medium")

    def test_low_confidence_is_not_raised_to_medium(self):
        out = judge_mod.enforce_rubric(_reply(span="not in there", confidence="Low"), "the evidence text")
        self.assertEqual(out["confidence"], "Low")

    def test_judge_applies_enforcement(self):
        reply = json.dumps(_reply(scope="Does not support", judgement="Supports", span="evidence"))
        with patch.object(judge_mod, "chat", return_value=types.SimpleNamespace(content=reply)):
            out = judge_mod.judge("claim", "the evidence")
        self.assertEqual(out["judgement"], "Does not support")
        self.assertTrue(out["rubric_mismatch"])
```

(`tests/test_judgement.py` imports `json`, `unittest` and `judge_mod` but **not** `types` or `patch` — add `import types` and `from unittest.mock import patch` at the top of the file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_judgement.py -q`
Expected: FAIL — `AttributeError: module … has no attribute 'derive_judgement'`.

- [ ] **Step 3: Implement**

In `research_assistant/judgement/judge.py`, add `import unicodedata` to the imports, then after `REQUIRED_SLOTS = {...}`:

```python
# Step 2 of the rubric: each slot has its own vocabulary. The model reply is
# validated at the top level only (VALID_JUDGEMENTS etc.); this is per slot.
SLOT_VOCAB = {
    "finding":  {"Supports", "Contradicts", "Does not support", "Insufficient"},
    "scope":    {"Supports", "Partially supports", "Does not support", "Insufficient"},
    "strength": {"Supports", "Partially supports", "Insufficient", "Not applicable"},
}

# Added by enforce_rubric(); merged into the verifier's record alongside
# REQUIRED_FIELDS. Additive: no existing field changes meaning.
DERIVED_FIELDS = {"model_judgement", "rubric_mismatch", "rubric_violations", "span_verified"}

_CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}


def _slot_verdict(slots: dict, name: str) -> str:
    """The verdict as the rules see it: anything outside the slot's vocabulary
    is read as 'Insufficient' — the model did not follow the rubric here, and
    the conservative reading of that is 'cannot tell'."""
    verdict = slots[name]["verdict"]
    return verdict if verdict in SLOT_VOCAB[name] else "Insufficient"


def derive_judgement(slots: dict) -> str:
    """Step 3 of the rubric, in code. First match wins."""
    finding, scope, strength = (_slot_verdict(slots, k) for k in ("finding", "scope", "strength"))
    if scope == "Does not support":
        return "Does not support"
    if finding == "Contradicts":
        return "Contradicts"
    if finding == "Does not support":
        return "Does not support"
    if finding == "Insufficient" or scope == "Insufficient":
        return "Unclear / insufficient evidence"
    if all(v in ("Supports", "Not applicable") for v in (finding, scope, strength)):
        return "Supports"
    return "Partially supports"


_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})
_ELLIPSIS_RE = re.compile(r"(\.\.\.|…|\[\s*\.\.\.\s*\]|\[…\])")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    text = _ELLIPSIS_RE.sub(" ", text)
    return " ".join(text.lower().split())


def span_is_verbatim(span, evidence: str):
    """Is the supporting span a substring of the evidence, up to whitespace,
    case, ligatures, quote style and ellipses? None when there is no span."""
    if span is None or not str(span).strip() or str(span).strip().lower() == "null":
        return None
    return _normalise(str(span)) in _normalise(evidence)


def enforce_rubric(result: dict, evidence: str) -> dict:
    """Make the record say what the rubric says, and where the model differed.

    The model's stated judgement is kept as model_judgement; `judgement`
    becomes the one the Step-3 rules derive from its own slots. Slot verdicts
    outside their vocabulary are listed in rubric_violations. The supporting
    span is checked verbatim. Any of those three caps confidence at Medium:
    the model's High was self-reported about a reply that broke its rules.
    """
    slots = result["slots"]
    violations = [
        f"{name}: {slots[name]['verdict']!r}"
        for name in ("finding", "scope", "strength")
        if slots[name]["verdict"] not in SLOT_VOCAB[name]
    ]
    derived = derive_judgement(slots)
    result["model_judgement"] = result["judgement"]
    result["rubric_mismatch"] = derived != result["judgement"]
    result["rubric_violations"] = violations
    result["judgement"] = derived
    result["span_verified"] = span_is_verbatim(result.get("supporting_span"), evidence)

    if violations or result["rubric_mismatch"] or result["span_verified"] is False:
        if _CONFIDENCE_RANK[result["confidence"]] > _CONFIDENCE_RANK["Medium"]:
            result["confidence"] = "Medium"
    return result
```

and change the end of `judge()`:

```python
    if not result.content:
        raise JudgementParseError("LLM returned an empty response.", raw="")
    return enforce_rubric(parse_judgement(result.content), citation_evidence)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_judgement.py tests/test_verifier.py -q`
Expected: PASS (the verifier's stub `_verdict()` lacks the derived fields; that is handled in Task 3 — if a verifier test fails here on a missing derived key, proceed to Task 3, which fixes the merge).

- [ ] **Step 5: Commit**

```bash
git add research_assistant/judgement/judge.py tests/test_judgement.py
git commit -m "feat(judgement): enforce the rubric in code — derived aggregate, per-slot vocabularies, verbatim span check, capped confidence"
```

---

### Task 3: The verifier records and shows the new fields

**Files:**
- Modify: `research_assistant/agents/agent8_verifier.py:37-41` (import), `:289` (merge), `:314-327` (`_totals`), `:421-450` (`_entry_block`)
- Test: `tests/test_verifier.py` (extend)

**Interfaces:**
- Produces: each judged entry carries `model_judgement, rubric_mismatch, rubric_violations, span_verified` (when the verdict has them; `entry.get` tolerates stubs without them); `totals["rubric_mismatch"]` and `totals["span_unverified"]`; the markdown shows a ⚠ line for each.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_verifier.py`:

```python


class TestDerivedFieldsInTheRecord(VerifyDraftTestCase):
    def _verdict_with(self, **extra):
        v = _verdict()
        v.update({"model_judgement": v["judgement"], "rubric_mismatch": False,
                  "rubric_violations": [], "span_verified": True})
        v.update(extra)
        return v

    def test_derived_fields_are_merged_and_counted(self):
        v = self._verdict_with(judgement="Does not support", model_judgement="Supports",
                               rubric_mismatch=True, span_verified=False, confidence="Medium")
        draft_path, report = self._run(
            "Graphene conducts well \\cite{cite_1}.", {"Smith 2020": "cite_1"}, [("Smith 2020", "a.pdf")],
            [{"text": "Graphene is highly conductive.", "metadata": {"document": "a.pdf"}, "chunk_index": 0}],
            judge_side_effect=lambda c, e, **k: v,
        )
        entry = report["results"][0]
        self.assertEqual(entry["judgement"], "Does not support")
        self.assertEqual(entry["model_judgement"], "Supports")
        self.assertTrue(entry["rubric_mismatch"])
        self.assertFalse(entry["span_verified"])
        self.assertEqual(report["totals"]["rubric_mismatch"], 1)
        self.assertEqual(report["totals"]["span_unverified"], 1)
        md = self._markdown(draft_path)
        self.assertIn("model said Supports", md)
        self.assertIn("not found verbatim", md)

    def test_a_verdict_without_derived_fields_still_works(self):
        # Older stubs and older records have no derived fields.
        _, report = self._run(
            "Graphene conducts well \\cite{cite_1}.", {"Smith 2020": "cite_1"}, [("Smith 2020", "a.pdf")],
            [{"text": "Graphene is highly conductive.", "metadata": {"document": "a.pdf"}, "chunk_index": 0}],
        )
        self.assertEqual(report["results"][0]["outcome"], "judged")
        self.assertEqual(report["totals"]["rubric_mismatch"], 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_verifier.py -q -k Derived`
Expected: FAIL — `KeyError: 'model_judgement'` / `'rubric_mismatch'` in totals.

- [ ] **Step 3: Implement**

Import: change

```python
from research_assistant.judgement.judge import (
    REQUIRED_FIELDS,
    JudgementParseError,
    judge,
)
```
to
```python
from research_assistant.judgement.judge import (
    DERIVED_FIELDS,
    REQUIRED_FIELDS,
    JudgementParseError,
    judge,
)
```

The merge (line 289):

```python
        entry.update({field: verdict[field] for field in sorted(REQUIRED_FIELDS)})
```
becomes
```python
        entry.update({field: verdict[field] for field in sorted(REQUIRED_FIELDS)})
        # Derived by enforce_rubric(); absent from older stubs and records.
        entry.update({field: verdict[field] for field in sorted(DERIVED_FIELDS) if field in verdict})
```

`_totals`: after `for judgement in _SEVERITY: totals[judgement] = 0` add

```python
    totals["rubric_mismatch"] = 0
    totals["span_unverified"] = 0
```
and inside the loop, under `if entry["outcome"] == "judged":`:
```python
            if entry.get("rubric_mismatch"):
                totals["rubric_mismatch"] += 1
            if entry.get("span_verified") is False:
                totals["span_unverified"] += 1
```

`_entry_block`: after the `**Confidence:** … **Evidence sufficiency:** …` line, add

```python
    if entry.get("compound_sentence"):
        block.append("⚠ **Compound sentence:** two or more citations in one long sentence — "
                     "a lost sentence boundary? This verdict is about the combined claim.\n")
    if entry.get("rubric_mismatch"):
        detail = "; ".join(entry.get("rubric_violations") or []) or "aggregate did not follow the slots"
        block.append(f"⚠ **Rubric:** model said {entry['model_judgement']}; the rules derive "
                     f"{entry['judgement']} ({detail}).\n")
    if entry.get("span_verified") is False:
        block.append("⚠ **Supporting span not found verbatim in the evidence** — treat it as a paraphrase.\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_verifier.py tests/test_judgement.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research_assistant/agents/agent8_verifier.py tests/test_verifier.py
git commit -m "feat(verify): record and report rubric mismatches, unverified spans and compound sentences"
```

---

### Task 4: Spike — the rubric as a cacheable prefix (adopt only if measured)

**Files:**
- Create: `scripts/judge_bench.py` (kept)
- Modify (conditionally): `research_assistant/judgement/prompt.md` — move `## Input` to the end; `tests/test_judgement.py::TestBuildPrompt` if it asserts position

**Interfaces:** none new. `build_prompt()` is unchanged: it does `.replace("{{CLAIM}}", …)` / `.replace("{{CITATION_EVIDENCE}}", …)` wherever the placeholders sit.

- [ ] **Step 1: The bench script**

```python
# scripts/judge_bench.py
"""Time and check the judge on the regression cases.

    PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/judge_bench.py [--runs N]

Prints per-case seconds and pass/fail against expected_judgement, then the
median over calls 2..end (call 1 is a cold prompt cache). Live model call;
not a unit test.
"""
import argparse, json, statistics as st, time
from pathlib import Path

from research_assistant.judgement.judge import judge, JudgementParseError

CASES = Path("research_assistant/judgement/cases/cases.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    times, passes = [], 0
    for run in range(args.runs):
        for c in cases:
            t0 = time.perf_counter()
            try:
                v = judge(c["claim"], c["citation_evidence"])
                got = v["judgement"]
            except JudgementParseError as exc:
                got = f"PARSE_FAIL ({exc})"
            secs = time.perf_counter() - t0
            times.append(secs)
            ok = got == c["expected_judgement"]
            passes += ok
            print(f"run {run+1} {c['id']:8s} {secs:6.1f}s  {'PASS' if ok else 'FAIL'}  expected={c['expected_judgement']!r} got={got!r}")
    warm = times[1:] or times
    print(f"\npass {passes}/{len(times)}   first call {times[0]:.1f}s   median of the rest {st.median(warm):.1f}s   max {max(warm):.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Baseline — current prompt order**

Make sure nothing else is using Ollama (no ingest running, no chat open). Run:

`PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/judge_bench.py --runs 1`

Expected: `pass 6/6`, per-call times of the order 60–120 s (2026-09-12 measurement: 68, 61, 109, 123, 113, 115 s). **Record the full output** in the Step 6 commit message.

- [ ] **Step 3: Reorder `prompt.md`**

Cut lines 9–16 of `research_assistant/judgement/prompt.md`:

```
## Input

**Claim:**
{{CLAIM}}

**Citation Evidence:**
{{CITATION_EVIDENCE}}

---
```

and re-insert the same block at the very end of the file (after the last paragraph, `**B against F.** …`), preceded by a `---` separator, so the file ends:

```
…an attribution the paper hedges ("may be due to", "could be attributed to", "probably") never supports a claim that states it as the cause.

---

## Input

Now apply Steps 1–5 to this claim and evidence. Return ONLY the JSON object.

**Claim:**
{{CLAIM}}

**Citation Evidence:**
{{CITATION_EVIDENCE}}
```

Nothing else in the file changes. Line 3–5 (the two-sentence framing) stays at the top.

- [ ] **Step 4: Bench again**

Run: `PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/judge_bench.py --runs 1`

**Gate:** adopt the reorder only if **both** hold: `pass 6/6`, and the median of calls 2–6 is ≤ (baseline median ÷ 1.5). Record the output either way.

- If the gate holds: also run `CITATION_LOG_FILE=0 python -m pytest tests/test_judgement.py -q` (`TestBuildPrompt` checks both placeholders are filled — position-independent; if any test asserts the input comes first, update that assertion and say so).
- If it does not: `git checkout -- research_assistant/judgement/prompt.md` and record the numbers in the commit message with "not adopted".

- [ ] **Step 5: Run the unit suite**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_judgement.py -q` — Expected: PASS.

- [ ] **Step 6: Commit**

If adopted:
```bash
git add scripts/judge_bench.py research_assistant/judgement/prompt.md tests/test_judgement.py
git commit -m "perf(judgement): rubric first, input last — a fixed prompt prefix Ollama can cache

baseline: <paste bench output>
reordered: <paste bench output>"
```
If not adopted:
```bash
git add scripts/judge_bench.py
git commit -m "chore(judgement): judge_bench.py; prompt reorder measured and not adopted

baseline: <paste>
reordered: <paste>"
```

---

### Task 5: `expand_neighbours()` in `shared/search.py`

**Files:**
- Modify: `research_assistant/shared/search.py` (add a function; `hybrid_search` untouched)
- Test: `tests/test_search.py` (extend)

**Interfaces:**
- Produces: `expand_neighbours(results: list[dict], texts: list[str], metadatas: list[dict], window: int = 1) -> list[dict]` — returns the same result dicts with `context_before: str` and `context_after: str` added (empty strings when there is no neighbour). Neighbour = `chunk_index ± i` for `i in 1..window`, only while `metadatas[j]["document"] == metadatas[chunk_index]["document"]`. Pure; no I/O.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python


from research_assistant.shared.search import expand_neighbours


class TestExpandNeighbours(unittest.TestCase):
    def _corpus(self):
        texts = ["a0", "a1", "a2", "b0", "b1"]
        metas = [{"document": "a.pdf"}] * 3 + [{"document": "b.pdf"}] * 2
        return texts, metas

    def test_adds_the_adjacent_chunks_of_the_same_document(self):
        texts, metas = self._corpus()
        out = expand_neighbours([{"chunk_index": 1, "text": "a1", "metadata": metas[1]}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("a0", "a2"))

    def test_never_crosses_into_another_document(self):
        texts, metas = self._corpus()
        out = expand_neighbours([{"chunk_index": 2, "text": "a2", "metadata": metas[2]}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("a1", ""))
        out = expand_neighbours([{"chunk_index": 3, "text": "b0", "metadata": metas[3]}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("", "b1"))

    def test_window_two_joins_both_sides(self):
        texts, metas = self._corpus()
        out = expand_neighbours([{"chunk_index": 2, "text": "a2", "metadata": metas[2]}], texts, metas, window=2)
        self.assertEqual(out[0]["context_before"], "a0\na1")

    def test_window_zero_and_out_of_range_are_safe(self):
        texts, metas = self._corpus()
        out = expand_neighbours([{"chunk_index": 0, "text": "a0", "metadata": metas[0]}], texts, metas, window=0)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("", ""))
        out = expand_neighbours([{"chunk_index": 99, "text": "x", "metadata": {"document": "z"}}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("", ""))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_search.py -q -k Neighbours`
Expected: FAIL — `ImportError: cannot import name 'expand_neighbours'`.

- [ ] **Step 3: Implement**

Append to `research_assistant/shared/search.py`:

```python


def expand_neighbours(results, texts, metadatas, window: int = 1):
    """Attach the adjacent chunks of the same document to each result.

    A paper's chunks are minted contiguously (ARCHITECTURE.md §4.4), so
    chunk_index ± 1 with the same ``metadata["document"]`` is the neighbouring
    text. Adds ``context_before`` / ``context_after`` (joined with newlines,
    nearest last / nearest first respectively); empty strings at a document
    boundary. Callers that ignore the two keys see no change.
    """
    def _meta(i):
        return metadatas[i] if 0 <= i < len(metadatas) else None

    for r in results:
        idx = r.get("chunk_index")
        doc = (r.get("metadata") or {}).get("document")
        before, after = [], []
        if isinstance(idx, int) and 0 <= idx < len(texts) and window > 0:
            for i in range(1, window + 1):
                m = _meta(idx - i)
                if m is None or m.get("document") != doc:
                    break
                before.insert(0, texts[idx - i])
            for i in range(1, window + 1):
                m = _meta(idx + i)
                if m is None or m.get("document") != doc:
                    break
                after.append(texts[idx + i])
        r["context_before"] = "\n".join(before)
        r["context_after"] = "\n".join(after)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_search.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/search.py tests/test_search.py
git commit -m "feat(search): expand_neighbours — adjacent chunks of the same document as context"
```

---

### Task 6: Evidence escalation in `verify_draft`

**Files:**
- Modify: `research_assistant/config.py:160-172` (Agent 8 block)
- Modify: `research_assistant/agents/agent8_verifier.py:31-36` (config import), `:45` (search import), `:205` (`top_k`), `:240-291` (retrieval → judge)
- Test: `tests/test_verifier.py` (extend)

**Interfaces:**
- Consumes: `expand_neighbours` (Task 5); `judge()` returning `evidence_sufficiency` and `judgement`.
- Produces: config `JUDGEMENT_ESCALATE_TOP_K = _env_int("CITATION_JUDGEMENT_ESCALATE_TOP_K", 3)`, `JUDGEMENT_NEIGHBOUR_WINDOW = _env_int("CITATION_JUDGEMENT_NEIGHBOUR_WINDOW", 1)`, `JUDGEMENT_EVIDENCE_MAX_CHARS = _env_int("CITATION_JUDGEMENT_EVIDENCE_MAX_CHARS", 6000)`; in `agent8_verifier`: `assemble_evidence(hits, max_chars) -> str`, `needs_escalation(verdict) -> bool`; judged entries gain `evidence_hits: int`, `escalated: bool`, and when escalated `first_judgement`, `first_sufficiency`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_verifier.py`:

```python


from research_assistant.agents.agent8_verifier import assemble_evidence, needs_escalation


def _hit(i, text, doc="a.pdf", before="", after=""):
    return {"chunk_index": i, "text": text, "metadata": {"document": doc},
            "context_before": before, "context_after": after}


class TestAssembleEvidence(unittest.TestCase):
    def test_neighbours_wrap_the_hit_and_hits_are_separated(self):
        out = assemble_evidence([_hit(1, "MID", before="PRE", after="POST"), _hit(5, "SECOND")], max_chars=10_000)
        self.assertEqual(out, "PRE\nMID\nPOST\n\n[…]\n\nSECOND")

    def test_cap_truncates_the_tail_not_the_top_hit(self):
        out = assemble_evidence([_hit(1, "A" * 50), _hit(2, "B" * 50)], max_chars=60)
        self.assertTrue(out.startswith("A" * 50))
        self.assertLessEqual(len(out), 60)

    def test_duplicate_text_is_not_repeated(self):
        out = assemble_evidence([_hit(1, "same"), _hit(2, "same")], max_chars=1000)
        self.assertEqual(out, "same")


class TestNeedsEscalation(unittest.TestCase):
    def test_insufficient_or_partial_sufficiency_escalates(self):
        self.assertTrue(needs_escalation({"judgement": "Supports", "evidence_sufficiency": "partial"}))
        self.assertTrue(needs_escalation({"judgement": "Supports", "evidence_sufficiency": "insufficient"}))

    def test_unclear_or_does_not_support_escalates_even_when_sufficient(self):
        self.assertTrue(needs_escalation({"judgement": "Unclear / insufficient evidence", "evidence_sufficiency": "sufficient"}))
        self.assertTrue(needs_escalation({"judgement": "Does not support", "evidence_sufficiency": "sufficient"}))

    def test_supported_and_sufficient_does_not(self):
        self.assertFalse(needs_escalation({"judgement": "Supports", "evidence_sufficiency": "sufficient"}))
        self.assertFalse(needs_escalation({"judgement": "Contradicts", "evidence_sufficiency": "sufficient"}))


class TestEscalation(VerifyDraftTestCase):
    HITS = [
        {"chunk_index": 10, "text": "first hit", "metadata": {"document": "a.pdf"}},
        {"chunk_index": 20, "text": "second hit", "metadata": {"document": "a.pdf"}},
        {"chunk_index": 30, "text": "third hit", "metadata": {"document": "a.pdf"}},
    ]

    def _resources(self):
        # texts/metadatas long enough for chunk_index 10/20/30 and their neighbours
        texts = [f"t{i}" for i in range(40)]
        metas = [{"document": "a.pdf"} for _ in range(40)]
        return texts, metas

    def _run_with(self, judge_replies):
        calls = []

        def fake_judge(claim, evidence, **kw):
            calls.append(evidence)
            reply = judge_replies[min(len(calls) - 1, len(judge_replies) - 1)]
            return reply

        draft_path = self._write("Graphene conducts well \\cite{cite_1}.", {"Smith 2020": "cite_1"})
        texts, metas = self._resources()
        import research_assistant.agents.agent8_verifier as a8
        with patch.object(a8, "hybrid_search", return_value=[dict(h) for h in self.HITS]) as hs, \
             patch.object(a8, "judge", side_effect=fake_judge):
            report = verify_draft(draft_path, search_resources=(FakeCollection([("Smith 2020", "a.pdf")]), None, texts, metas))
        return report, calls, hs

    def test_retrieves_the_escalation_budget_in_one_search(self):
        _, _, hs = self._run_with([_verdict()])
        self.assertEqual(hs.call_args.kwargs["top_k"], 3)          # max(JUDGEMENT_TOP_K=1, ESCALATE_TOP_K=3)
        self.assertEqual(hs.call_args.kwargs["exclude_types"], {"figure_description"})

    def test_sufficient_first_verdict_judges_once_on_the_top_hit_with_neighbours(self):
        report, calls, _ = self._run_with([_verdict()])
        self.assertEqual(len(calls), 1)
        self.assertIn("first hit", calls[0])
        self.assertIn("t9", calls[0]); self.assertIn("t11", calls[0])     # neighbours of chunk 10
        self.assertNotIn("second hit", calls[0])
        e = report["results"][0]
        self.assertFalse(e["escalated"]); self.assertEqual(e["evidence_hits"], 1)
        self.assertEqual(e["evidence"], calls[0])

    def test_insufficient_first_verdict_escalates_once_and_keeps_both(self):
        first = _verdict(judgement="Unclear / insufficient evidence"); first["evidence_sufficiency"] = "insufficient"
        second = _verdict(judgement="Supports")
        report, calls, _ = self._run_with([first, second])
        self.assertEqual(len(calls), 2)
        for t in ("first hit", "second hit", "third hit"):
            self.assertIn(t, calls[1])
        e = report["results"][0]
        self.assertTrue(e["escalated"]); self.assertEqual(e["evidence_hits"], 3)
        self.assertEqual(e["judgement"], "Supports")
        self.assertEqual(e["first_judgement"], "Unclear / insufficient evidence")
        self.assertEqual(e["first_sufficiency"], "insufficient")
        self.assertEqual(e["evidence"], calls[1])

    def test_no_escalation_when_there_is_nothing_more_to_show(self):
        first = _verdict(judgement="Does not support"); first["evidence_sufficiency"] = "insufficient"
        draft_path = self._write("Graphene conducts well \\cite{cite_1}.", {"Smith 2020": "cite_1"})
        texts, metas = self._resources()
        import research_assistant.agents.agent8_verifier as a8
        with patch.object(a8, "hybrid_search", return_value=[dict(self.HITS[0])]), \
             patch.object(a8, "judge", return_value=first) as j:
            report = verify_draft(draft_path, search_resources=(FakeCollection([("Smith 2020", "a.pdf")]), None, texts, metas))
        self.assertEqual(j.call_count, 1)
        self.assertFalse(report["results"][0]["escalated"])

    def test_escalation_disabled_by_config(self):
        first = _verdict(judgement="Does not support"); first["evidence_sufficiency"] = "insufficient"
        import research_assistant.agents.agent8_verifier as a8
        with patch.object(a8, "JUDGEMENT_ESCALATE_TOP_K", 0):
            report, calls, hs = self._run_with([first])
        self.assertEqual(len(calls), 1)
        self.assertEqual(hs.call_args.kwargs["top_k"], 1)
```

(`_verdict(judgement=...)` already exists in the file with a `judgement` parameter; `FakeCollection` and `VerifyDraftTestCase` too.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_verifier.py -q -k "Assemble or Escalation or NeedsEscalation"`
Expected: FAIL — `ImportError: cannot import name 'assemble_evidence'`.

- [ ] **Step 3: Config**

In `research_assistant/config.py`, after `JUDGEMENT_TOP_K = ...` (line 167) add:

```python
# Evidence escalation. The top JUDGEMENT_TOP_K hit(s), each with
# JUDGEMENT_NEIGHBOUR_WINDOW adjacent chunks, are judged first. If that
# verdict reports it did not see enough (sufficiency != sufficient, or
# Unclear / Does not support), the top JUDGEMENT_ESCALATE_TOP_K hits are
# judged once more. 0 disables the second look. The assembled evidence is
# capped so prompt (~3.8k tokens) + evidence stays inside num_ctx below.
JUDGEMENT_ESCALATE_TOP_K     = _env_int("CITATION_JUDGEMENT_ESCALATE_TOP_K", 3)
JUDGEMENT_NEIGHBOUR_WINDOW   = _env_int("CITATION_JUDGEMENT_NEIGHBOUR_WINDOW", 1)
JUDGEMENT_EVIDENCE_MAX_CHARS = _env_int("CITATION_JUDGEMENT_EVIDENCE_MAX_CHARS", 6000)
```

- [ ] **Step 4: Verifier**

Imports — extend the config import to

```python
from research_assistant.config import (
    JUDGEMENT_ESCALATE_TOP_K,
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_MODEL,
    JUDGEMENT_NEIGHBOUR_WINDOW,
    JUDGEMENT_TOP_K,
    LLM_BACKEND,
    LLM_MODEL,
)
```
and the search import to
```python
from research_assistant.shared.search import expand_neighbours, hybrid_search
```

Add above `def verify_draft`:

```python
_HIT_SEPARATOR = "\n\n[…]\n\n"


def assemble_evidence(hits, max_chars: int) -> str:
    """The text the judge sees: each hit wrapped in its neighbours, hits in
    rank order, separated, de-duplicated, capped from the tail so the top hit
    is never the one cut."""
    parts, seen = [], set()
    for h in hits:
        block = "\n".join(p for p in (h.get("context_before", ""), h["text"], h.get("context_after", "")) if p)
        if block in seen:
            continue
        seen.add(block)
        parts.append(block)
    out = _HIT_SEPARATOR.join(parts)
    return out[:max_chars]


def needs_escalation(verdict: dict) -> bool:
    """The first verdict says it did not see enough — by its own sufficiency
    rating, or by landing on a judgement that more evidence could overturn."""
    return (
        verdict.get("evidence_sufficiency") != "sufficient"
        or verdict.get("judgement") in ("Unclear / insufficient evidence", "Does not support")
    )
```

In `verify_draft`, change

```python
    top_k = top_k or JUDGEMENT_TOP_K
```
to
```python
    top_k = top_k or JUDGEMENT_TOP_K
    escalate_k = JUDGEMENT_ESCALATE_TOP_K if JUDGEMENT_ESCALATE_TOP_K > top_k else 0
    retrieve_k = max(top_k, escalate_k)
```

change the retrieval call's `top_k=top_k` to `top_k=retrieve_k`, and replace the block from `entry["evidence"] = hits[0]["text"]` through `results.append(entry)` at the end of the judged branch with:

```python
        expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
        entry["evidence"] = assemble_evidence(hits[:top_k], JUDGEMENT_EVIDENCE_MAX_CHARS)
        entry["evidence_hits"] = min(top_k, len(hits))
        entry["escalated"] = False
        try:
            verdict = _judge_once(entry["claim"], entry["evidence"])
            # Second look, once, only when the first verdict says it saw too
            # little and there is more of this paper to show it.
            if escalate_k and len(hits) > top_k and needs_escalation(verdict):
                entry["first_judgement"] = verdict["judgement"]
                entry["first_sufficiency"] = verdict["evidence_sufficiency"]
                entry["evidence"] = assemble_evidence(hits[:escalate_k], JUDGEMENT_EVIDENCE_MAX_CHARS)
                entry["evidence_hits"] = min(escalate_k, len(hits))
                entry["escalated"] = True
                logger.info(" -> %s on %d hit(s), sufficiency %s — judging again with %d hit(s).",
                            verdict["judgement"], top_k, verdict["evidence_sufficiency"], entry["evidence_hits"])
                verdict = _judge_once(entry["claim"], entry["evidence"])
        except JudgementParseError as exc:
            entry["outcome"] = "parse_failed"
            entry["raw"] = exc.raw
            logger.warning(" -> unusable reply after %d attempts.", _JUDGE_ATTEMPTS)
            results.append(entry)
            continue
        except Exception as exc:
            # Connection errors, timeouts and HTTP 4xx/5xx are not "unusable
            # model reply" — filing them as parse_failed sends the reader to
            # prompt.md when the endpoint was simply down.
            entry["outcome"] = "call_failed"
            entry["error_type"] = type(exc).__name__
            entry["raw"] = str(exc)
            logger.warning(" -> judging call failed: %s: %s", type(exc).__name__, exc)
            results.append(entry)
            continue

        entry["outcome"] = "judged"
        # Only the rubric's own fields are merged. _validate permits extra
        # top-level keys in a model reply — harmless in themselves, but a
        # blanket update() lets an echoed `sentence_index`, `claim`, `evidence`
        # or `outcome` overwrite the pipeline's own record of what was judged,
        # which is exactly the provenance §9.2 leans on to justify
        # re-retrieval.
        entry.update({field: verdict[field] for field in sorted(REQUIRED_FIELDS)})
        # Derived by enforce_rubric(); absent from older stubs and records.
        entry.update({field: verdict[field] for field in sorted(DERIVED_FIELDS) if field in verdict})
        logger.info(" -> %s (%s confidence)", verdict["judgement"], verdict["confidence"])
        results.append(entry)
```

Also set `"escalate_top_k": escalate_k` in the `report = {...}` dict next to `"top_k": top_k`.

In `_entry_block`, after the confidence line add:

```python
    if entry.get("escalated"):
        block.append(f"↻ **Escalated:** first verdict {entry['first_judgement']} "
                     f"(sufficiency {entry['first_sufficiency']}) on the top hit; "
                     f"re-judged on {entry['evidence_hits']} hits.\n")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `CITATION_LOG_FILE=0 python -m pytest tests/test_verifier.py tests/test_config_version.py -q`
Expected: PASS. If `TestRetrievalFailure` / `TestJudgingCallFailure` (existing) fail, the exception handling above was not preserved exactly — compare with the original block.

- [ ] **Step 6: Commit**

```bash
git add research_assistant/config.py research_assistant/agents/agent8_verifier.py tests/test_verifier.py
git commit -m "feat(verify): judge the top hit with its neighbours; escalate once to the top-3 when the verdict says it saw too little"
```

---

### Task 7: Docs

**Files:**
- Modify: `ARCHITECTURE.md` (after §9.3), `PIPELINE.md` (Agent 8 row; configuration notes)

- [ ] **Step 1: `ARCHITECTURE.md`** — after §9.3 add:

```markdown
### 9.4 The sentence boundary survives citation

`gemma4:e2b` drops the terminal full stop when it appends `\cite{}`. Agent 5
restores it (`_restore_terminal_punctuation`), because the draft is joined
with spaces and Agent 8 splits on `[.!?]` + whitespace: a lost stop merged
two sentences into one claim, and both citations were judged against it.
Agent 8 still flags any sentence with ≥2 keys and >40 words as
`compound_sentence` — the signature of a merge that got through.

### 9.5 The rubric is enforced in code

The model's reply is evidence, not verdict. `enforce_rubric()` derives the
aggregate from the three slot verdicts by the rubric's own Step-3 rules,
records the model's stated judgement as `model_judgement` and any
disagreement as `rubric_mismatch`, lists slot verdicts outside their
vocabulary in `rubric_violations` (treated as *Insufficient* for
aggregation), and checks `supporting_span` verbatim against the evidence
(`span_verified`). Any of those caps `confidence` at Medium.

**Why derive rather than reject.** A rejected reply is retried at
temperature 0 — the same reply — and the citation ends up unjudged. A derived
verdict with the disagreement on record is more useful, and it is what the
evaluation set (next) will measure.

### 9.6 Evidence: the top hit with its neighbours, then a second look

Agent 8 retrieves `max(JUDGEMENT_TOP_K, JUDGEMENT_ESCALATE_TOP_K)` hits from
the cited paper once, wraps each in its adjacent chunks (`expand_neighbours`),
and judges the top `JUDGEMENT_TOP_K`. If that verdict reports it saw too
little — `evidence_sufficiency` not *sufficient*, or *Unclear* / *Does not
support* — it judges once more on the top `JUDGEMENT_ESCALATE_TOP_K`, capped
at `JUDGEMENT_EVIDENCE_MAX_CHARS`. The second verdict stands; the first is
kept (`first_judgement`, `first_sufficiency`, `escalated`). At most two calls
per citation. `figure_description` chunks are never evidence.

**Why.** `hits[0]` alone reported *strength: Insufficient* on a claim whose
supporting paragraph sat one chunk away in the same paper. The rubric's
sufficiency field exists to separate a retrieval gap from a citation
failure; now it drives one.
```

If Task 4 adopted the reorder, add a one-paragraph §9.7 with the measured before/after numbers and the reason (fixed prefix, Ollama prompt cache).

- [ ] **Step 2: `PIPELINE.md`** — in the Agent 8 row append: *"Claims keep their sentence boundary (Agent 5 restores the full stop after `\cite{}`); the aggregate is derived from the slots in code and disagreement is recorded; the supporting span is verified verbatim; the top hit is judged with its neighbours and, when the verdict says it saw too little, re-judged once on the top 3."* In configuration notes add `CITATION_JUDGEMENT_ESCALATE_TOP_K`, `CITATION_JUDGEMENT_NEIGHBOUR_WINDOW`, `CITATION_JUDGEMENT_EVIDENCE_MAX_CHARS`.

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md PIPELINE.md
git commit -m "docs: judgement hardening — sentence boundary, rubric enforcement, evidence escalation"
```

---

### Task 8: Live check on the v2 index — the run that exposed the defects, repeated

Needs Ollama with `gemma4:e2b` and `nomic-embed-text`, and the v2 index (`CITATION_INDEX_VERSION=2`). Nothing else should be using Ollama.

- [ ] **Step 1: The same two-sentence draft**

```bash
mkdir -p /tmp/judgement-check && cat > /tmp/judgement-check/draft.txt <<'EOF'
Covalent networks of MoS2 flakes show enhanced photoconductivity compared to pristine films. Charge transport in these networks is dominated by hopping between flakes at low temperature.
EOF
PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python -m research_assistant.agents.agent5_batch_citer \
  --file /tmp/judgement-check/draft.txt --out /tmp/judgement-check/cited.txt
cat /tmp/judgement-check/cited.txt
```

**Gate:** the cited draft contains **two** full stops, one after each `\cite{}` group (`… films \cite{cite_1}. Charge transport … temperature \cite{cite_2}.`).

- [ ] **Step 2: Verify it**

```bash
PYTHONPATH=. CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 python -m research_assistant.agents.agent8_verifier \
  --draft /tmp/judgement-check/cited.txt
python - <<'EOF'
import json
rep = json.load(open("/tmp/judgement-check/cited_verification.json"))
print("escalate_top_k:", rep["escalate_top_k"], "| totals:", {k: v for k, v in rep["totals"].items() if v})
for r in rep["results"]:
    print(f"\n{r['cite_key']} → {r['judgement']} ({r['confidence']}, sufficiency={r['evidence_sufficiency']})"
          f" | model said {r.get('model_judgement')} | mismatch={r.get('rubric_mismatch')} violations={r.get('rubric_violations')}"
          f" | span_verified={r.get('span_verified')} | escalated={r.get('escalated')} hits={r.get('evidence_hits')}"
          f" | compound={r.get('compound_sentence')}")
    print("   claim:", r["claim"][:110])
    if r.get("escalated"): print("   first:", r["first_judgement"], "/", r["first_sufficiency"])
EOF
```

**Gates:**
- `results` has two entries with **different** `claim` texts (sentence 1 and sentence 2), `compound_sentence: False` on both.
- `cite_1` (the MoS₂ paper): judgement *Supports* or *Partially supports*; if `escalated`, `first_judgement` is recorded; `span_verified` is `True` or `None` — if `False`, the ⚠ line appears in `cited_verification.md`.
- `cite_2`: whichever paper Agent 5 chose; if it is not the MoS₂ paper, *Does not support* is the correct verdict and it should have `escalated: True` (the first verdict asked for more, more was shown, the verdict held).
- Every judged entry has `rubric_violations == []` **or** the ⚠ Rubric line in the markdown explains what was overridden.

Paste both printouts into the Step 4 commit message.

- [ ] **Step 3: Regression cases and the suite**

```bash
PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/judge_bench.py --runs 1      # expect pass 6/6
python -c "
import fcntl; f=open('data/ingest.lock','w')
try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB); print('ingest lock free — run the full suite')
except OSError: print('ingest lock HELD — run with --deselect tests/test_ingestion.py')"
CITATION_LOG_FILE=0 python -m pytest tests/ -q          # or with the deselect
```

Expected: 6/6; suite green.

- [ ] **Step 4: Commit the record**

```bash
git commit --allow-empty -F - <<'EOF'
chore(judgement): live check on the v2 index after hardening

cited draft: <paste>
verification: <paste the printout>
bench: <paste the pass line and times>
suite: <paste the last line>
EOF
```

---

## Self-review

**Spec coverage.** §2A → Task 1. §2B → Tasks 2, 3. §2C → Task 4 (gated, reversible). §2D → Tasks 5, 6. §4 config → Task 6. §3 "additive fields" → Tasks 2, 3, 6 only add keys; `judgement` keeps its meaning (now rule-derived, which is what the rubric always intended). §5 risks: reorder gated by the bench; escalation off via `ESCALATE_TOP_K=0`; both verdicts recorded.

**Type consistency.** `expand_neighbours` (Task 5) writes `context_before`/`context_after` that `assemble_evidence` (Task 6) reads. `DERIVED_FIELDS` (Task 2) is what Task 3 and Task 6's merge line import. `_verdict(judgement=…)` in the existing verifier tests takes the keyword the new tests pass. `needs_escalation` reads `evidence_sufficiency` and `judgement`, both in `REQUIRED_FIELDS`.

**Placeholders.** None. Each bench/commit message asks for pasted output, not a summary.
