# Judgement hardening — the verdict you read is the verdict the rubric gives

*2026-09-12*

## 1. Purpose

Agent 8 (`agents/agent8_verifier.py` + `judgement/judge.py` + `judgement/prompt.md`)
audits every `\cite{}` in a draft. Four defects, each seen in a real run on
2026-09-12 against the v2 index, mean the verdict a reader sees can differ
from what the rubric — or the evidence — actually says:

1. **Claims arrive merged.** `gemma4:e2b` drops the terminal full stop when it
   appends `\cite{}`; Agent 5 stores its `CITED:` line verbatim and joins
   sentences with a space; Agent 8's splitter needs `[.!?]` before whitespace.
   A two-sentence draft became one claim with two cites; `cite_1` was judged
   on "photoconductivity *and* hopping" and downgraded to *Partially supports*
   for a claim its evidence supports.
2. **Retrieved evidence is discarded.** `verify_draft` judges `hits[0]` only,
   whatever `top_k` is. `cite_1`'s record said `strength: Insufficient,
   sufficiency: partial` while the hopping paragraph sat one chunk down in the
   same paper. The rubric's `evidence_sufficiency` — built to separate a
   retrieval gap from a citation failure — triggers nothing.
3. **The rubric is not enforced in code.** `_validate` checks a slot verdict
   is *a string*; `finding: "Partially supports"` (forbidden by the rubric)
   passed in 2 of 8 judgements. The Step-3 aggregation is deterministic given
   the slots, yet the model's stated `judgement` is trusted as written.
4. **`supporting_span` is unverified.** The rubric says verbatim; `cite_1`'s
   span is not in its evidence.

And one cost: each judgement is 61–123 s on the CPU VM-class machine. The
prompt is ~3,760 tokens with `{{CLAIM}}` at line 12, so consecutive calls
share almost no prefix and Ollama re-processes the whole rubric every time.

The six regression cases pass 6/6 — but they are the prompt's own worked
examples, so that measures copying, not judging. Building a real evaluation
set is the next spec (see §6); this one makes the verdict pipeline honest and
cheaper first, because an evaluation of a pipeline that merges claims and
ignores its own rubric would measure the wrong thing.

## 2. Decisions

### A. The sentence boundary survives citation (Agent 5), and Agent 8 notices when it didn't

`_cite_sentence_with_reasoning()` restores the original sentence's terminal
punctuation when the cited sentence lost it: `"… films \cite{a}"` →
`"… films \cite{a}."`. A cited sentence that already ends in `[.!?]`, or ends
in `[.!?]` immediately before a trailing `\cite{}` group, is left alone.

`citation_pairs()` in Agent 8 flags any sentence carrying ≥2 distinct cite
keys **and** more than 40 words as `compound_sentence: true` and logs a
warning — the signature of a merge that slipped through.

### B. The rubric is enforced in code; the model's reply is evidence, not verdict

`judge()` gains an `enforce_rubric(result, evidence)` pass after parsing:

- **Per-slot vocabularies** from the rubric (`finding`: Supports / Contradicts /
  Does not support / Insufficient; `scope`: Supports / Partially supports /
  Does not support / Insufficient; `strength`: Supports / Partially supports /
  Insufficient / Not applicable). A verdict outside its slot's vocabulary is
  recorded in `rubric_violations` and, for aggregation, treated as
  `Insufficient` — the conservative reading of "the model did not follow the
  rubric here".
- **The aggregate is derived from the slots** by the Step-3 rules, in code.
  The model's stated judgement is kept as `model_judgement`; `judgement` is
  the derived one; `rubric_mismatch` is true when they differ.
- **`supporting_span` is checked verbatim** against the evidence after NFKC
  normalisation, lowercasing, whitespace collapse, and stripping of quotes
  and ellipses. `span_verified` is true / false / null (no span given).
- **Confidence is capped at `Medium`** when any violation, mismatch, or
  unverified span occurred: the model's `High` was self-reported about a
  reply that broke its own rules.

Why derive rather than reject: a rejected reply is retried at temperature 0
(same reply) and the citation ends up unjudged. A derived verdict with the
disagreement on record is more useful to the reader and to the evaluation
that follows.

### C. The rubric is a fixed prefix so Ollama can cache it — measured before adopted

`prompt.md`'s `## Input` block moves from the top to the end, after the
examples. Every call then shares the first ~3,500 tokens byte-for-byte, and
Ollama's prompt cache (llama.cpp prefix reuse, same model, same options) can
skip re-processing them. Expected: prompt processing — the bulk of a 60–120 s
call on CPU — collapses to the claim and evidence.

This is a spike with a gate: adopt only if the six regression cases still
pass 6/6 **and** median per-call time improves ≥1.5× on calls 2–6. Otherwise
revert and record the numbers. Placement of the input relative to the
instructions is a known variable in rubric prompts; the regression cases are
the guard.

### D. Evidence escalation — judge what was retrieved, and look wider when told to

`verify_draft` retrieves `max(JUDGEMENT_TOP_K, JUDGEMENT_ESCALATE_TOP_K)`
hits once, expands each with its neighbour chunks (`expand_neighbours`, spec
`2026-09-11-rag-engine-v2-design.md` §3.4, implemented here), and judges the
top `JUDGEMENT_TOP_K` (default 1) as today. If that verdict says it did not
see enough — `evidence_sufficiency != "sufficient"`, or the judgement is
*Unclear / insufficient evidence* or *Does not support* — and more hits
exist, it judges once more with the top `JUDGEMENT_ESCALATE_TOP_K` (default
3) hits joined, capped at `JUDGEMENT_EVIDENCE_MAX_CHARS` (6,000 ≈ 1.5k tokens;
prompt + evidence stays under `num_ctx` 8,192). The second verdict stands;
the first is kept as `first_judgement` / `first_sufficiency`, with
`escalated: true` and `evidence_hits: N`. At most two calls per citation, and
the second only when the first asked for it.

`figure_description` chunks stay excluded (they are never evidence).

## 3. What is and isn't changed

Changed: `agents/agent5_batch_citer.py`, `agents/agent8_verifier.py`,
`judgement/judge.py`, `judgement/prompt.md` (order only, gated),
`shared/search.py` (`expand_neighbours`), `config.py`, docs, tests.

Not changed: the rubric's rules and vocabularies; the six regression cases;
`shared/llm.py`; anything in ingestion or the indexes. The verifier's JSON
record gains fields; no field is removed or renamed, so existing reports
still parse.

## 4. Configuration

| variable | default | meaning |
|---|---|---|
| `CITATION_JUDGEMENT_TOP_K` | `1` (existing) | hits judged first |
| `CITATION_JUDGEMENT_ESCALATE_TOP_K` | `3` | hits judged on escalation; `0` disables escalation |
| `CITATION_JUDGEMENT_NEIGHBOUR_WINDOW` | `1` | neighbour chunks each side |
| `CITATION_JUDGEMENT_EVIDENCE_MAX_CHARS` | `6000` | cap on assembled evidence |

## 5. Risks

- **Reordering the prompt changes behaviour.** Gated by the regression cases
  and reverted on failure. The cases are the prompt's own examples, so a pass
  is necessary, not sufficient — the evaluation set in §6 is the real guard.
- **Escalation can flip a right verdict wrong.** More evidence can also
  distract a 2B model. Both verdicts are recorded; the evaluation set will
  measure whether escalation nets out positive; `ESCALATE_TOP_K=0` turns it
  off.
- **Derived-vs-model disagreement will be visible.** That is the point: it
  surfaces every place the model broke the rubric. The report shows both.

## 6. Out of scope (next spec)

A held-out judge evaluation: real claim/evidence pairs from the corpus
labelled by the operator plus adversarial transforms of *Supports* cases,
with accuracy, a confusion matrix, the Contradicts-vs-Does-not-support
confusion, and run-to-run stability. Prompt content changes (physics
examples, trimming) wait for it. A second-opinion judge (`qwen2.5:7b`) for
agreement-based confidence is optional after that.
