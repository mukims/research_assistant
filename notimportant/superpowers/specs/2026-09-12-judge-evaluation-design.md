# Judge evaluation — a held-out set the rubric has never seen

*2026-09-12*

## 1. Purpose

Agent 8's judge passes its regression cases 6/6 — and those six are the
worked examples inside `prompt.md`. They measure whether `gemma4:e2b` can
copy what it was just shown. Nothing measures whether it judges a physics
claim from this corpus correctly, whether it confuses *Contradicts* with
*Does not support* (the D-vs-E distinction the rubric spends a paragraph on),
whether the rubric-enforcement and escalation added by
`2026-09-12-judgement-hardening-design.md` net out positive, or whether any
prompt edit helps or hurts. Every later change to the judge is a guess
without this.

The deliverable is an evaluation set that is **held out** of the prompt,
**labelled by a person** where the label is a judgement call, **generated
deterministically** where it isn't, and a harness that reports the numbers a
prompt or model change has to move.

## 2. Where labels come from — three sources, kept apart

### 2.1 Human-labelled real pairs (`judgement/cases/human.jsonl`, target 12–15)

Candidates are harvested from the corpus so the operator confirms rather
than composes:

- **Abstract → body.** An abstract sentence is a claim the paper's own body
  should support. For a sample of papers with an abstract chunk in the v2
  index (262 of 304 have one), take abstract sentences of 80–300 characters,
  retrieve the best body chunk from the *same* paper (`doc_filter`, abstract
  chunks and `figure_description` excluded), wrap it in its neighbours, and
  present the pair. Most are *Supports*; the ones that aren't — the abstract
  overstates, the body chunk retrieved is the wrong one — are the valuable
  labels.
- **Verification records.** Every `*_verification.json` Agent 8 has written
  holds real (claim, evidence, model verdict) triples from real drafts. The
  operator labels these **blind** — the model's verdict is hidden until the
  label is saved — so the harvest doubles as an agreement measurement.

Labelling is a terminal loop (`scripts/judge_label.py`): claim, evidence,
one keystroke for the verdict, optional note, and for each *Supports* an
optional hand-written negated claim (source of the *Contradicts* cases no
transform can make reliably). Ten to fifteen labels is roughly twenty
minutes.

### 2.2 Deterministic transforms (`judgement/cases/transforms.jsonl`, target ~40)

Each transform takes a labelled *Supports* pair and produces a case whose
correct verdict follows from the rubric by construction. No model is used
to generate them, so they cannot inherit the judge's blind spots.

| transform | what changes | expected | accepted also | why it's decidable |
|---|---|---|---|---|
| `cross_pair` | claim of pair *i*, evidence of pair *j* from a different paper | Does not support | — | rubric: evidence on an unrelated topic reports nothing about this relationship |
| `scope_swap` | one system/material/regime token in the **claim** replaced from a table (`MoS2→WSe2`, `graphene→silicon`, `one-dimensional→three-dimensional`, `low temperature→room temperature`, …); applied only when the token also appears in the evidence | Does not support | Unclear / insufficient evidence | rubric example E: the evidence examined none of what the claim now covers — a scope failure, never a contradiction |
| `number_swap` | a number present in **both** claim and evidence is changed in the claim (×2, or ×0.5 when ≥ 100), unit kept | Contradicts | — | rubric example D: same conditions, opposite/incompatible result |
| `delete_key_sentence` | the evidence sentence with the highest token overlap with the claim is removed | Does not support | Unclear / insufficient evidence | the relationship is no longer reported; "fragmentary" vs "reports nothing" is the rubric's own grey zone, so both are accepted |
| `hedge_evidence` | in that same key sentence, an assertive verb is weakened (`shows/demonstrates/establishes/confirms/reveals → may suggest`, `is/are → may be`, `increases → may increase`, …); applied only when such a verb is present | Partially supports | — | rubric examples B and F: a hedged mechanism never supports an assertive claim |

Transforms are applied to every *Supports* case from §2.1 plus the prompt's
case A, so a set of 8–10 seeds yields ~40 cases. Each transform case records
its `origin` and `transform` so a failure is traceable to a mechanism.

### 2.3 Prompt examples (`judgement/cases/cases.jsonl`, the existing six)

Kept, run, and reported in their own row labelled *in-prompt* — a useful
floor (a model that fails these has a broken parser or context window), and
never part of the headline number.

### 2.4 Held-out discipline

The harness refuses to score any case whose claim or evidence appears
verbatim in `prompt.md`, and reports what it refused. Adding a human case to
the prompt as an example moves it out of the evaluation automatically.

## 3. Two modes of measurement

- **`judge` mode** — `judge(claim, evidence)` on each case's stored
  evidence. Measures the rubric + model + `enforce_rubric()`, isolated from
  retrieval. This is the mode a prompt edit is judged by.
- **`verifier` mode** — for cases that carry `document` and
  `citation_source` (the human cases do), write a one-sentence draft
  `"<claim> \cite{cite_1}."` with the matching mapping and run
  `verify_draft()` against the live v2 index. Measures the whole path:
  retrieval, neighbours, escalation, enforcement. This is the mode the
  hardening spec's §2D (escalation) is judged by. Runs only when the index
  is present.

## 4. Metrics

Per run, over held-out cases (human + transforms), and broken down by source
and by transform:

- **strict accuracy** (`got == expected`) and **lenient accuracy**
  (`got ∈ accept`);
- **confusion matrix** over the five judgements, with the
  *Contradicts ↔ Does not support* cell pair called out as a number of its own;
- **per-class precision and recall**;
- **rubric signals** (when the hardening branch is present): rate of
  `rubric_mismatch`, of out-of-vocabulary slot verdicts, of
  `span_verified == False`; in verifier mode also the escalation rate and
  how often escalation changed the verdict;
- **stability**: with `--runs N`, the fraction of cases whose verdict is
  identical across all runs;
- **latency**: median and p90 seconds per judgement.

Output: a markdown table to stdout and a JSON file
`data/eval/judge/results/<timestamp>-<mode>.json` with every per-case
result; `--compare a.json b.json` prints deltas case by case. A baseline is
recorded before any prompt or model change.

## 5. What this spec does not do

No prompt changes, no model changes — this measures them. No LLM-generated
adversarial cases: the negations that need a writer come from the operator
during labelling. No automatic pass/fail thresholds: the harness reports,
the operator decides. The set grows over time: every verification record
Agent 8 writes is a future candidate.

## 6. Dependencies and placement

Built on the `judgement-hardening` branch if it exists (the harness reads
`rubric_mismatch`, `span_verified`, `escalated` when present and shows n/a
otherwise). Cases live in `research_assistant/judgement/cases/` and are
committed (short excerpts of open-access papers). Candidates, labels in
progress, and results live under `data/eval/judge/` (gitignored).
`evaluate_judge.py` is a root-level script like `orchestrate.py`; pure
helpers live in `research_assistant/judgement/` so they are testable without
a model.
