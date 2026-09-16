# Running the citation audit on a local model — gemma4:e2b vs gemini-3.5-flash-lite

Same seed paper (`arxiv_2108.10114v3.pdf`), same corpus, same rubric (V1.5), same retrieval.
Only the judgement model differs. Reports: `data/eval/audit/local_gemma4.json` (local) and
`data/raw/seed_audits/history/arxiv_2108.10114v3_20260916-162114_audit.json` (hosted).

## Headline

| | gemini-3.5-flash-lite | gemma4:e2b (local, CPU) |
|---|---|---|
| judged | 20 | 18 (1 parse failure, 1 no evidence) |
| Supports | 5 | **15** |
| Unclear / insufficient | 12 | 1 |
| Partially / Does not support | 2 / 1 | 1 / 1 |
| evidence rated `sufficient` | 6 | 16 |
| **supporting span not verbatim** | 4 / 20 | **13 / 18** |
| wall clock | ~2 min | ~24 min |

The local model agrees with almost everything, and 13 of its 18 verdicts quote a span that does not
occur in the evidence it was given. The failure is not paraphrase. On several claims it echoed the
**citing** paper's own sentence back as the cited paper's evidence, citation markers included:

> span: `"However, the basic non-spatially-resolving inversion has been shown to work with other
> input signals [40,] [57]."`
> evidence: *"(b) shows the energy-dependent conductance for an armchair-edged GNR of width W = 3a…"*

Keeping only the verdicts whose span is verbatim leaves 4 Supports and 1 Unclear out of 18.

## What this says about the system

1. **`span_is_verbatim` caught every one of them.** The guard works, and it is the only thing
   standing between a weak model and a page of green ticks.
2. **A fabricated span used to still report as `Supports`.** `enforce_rubric` capped confidence at
   Medium and recorded `span_verified: false`, but left the verdict and its green badge alone.
   Fixed in `1c0af02`: Supports, Partially supports and Contradicts are recorded as Unclear when
   the span is not in the evidence — the symmetric half of the absence guard. Replaying these two
   audits through it: hosted Supports 5 → 3, local Supports 15 → 4. The report names the invented
   quote (`a2aa44d`, extended in the follow-up commit).
3. **Model choice changes the audit's answer completely**, in the permissive direction. The audit's
   headline numbers are not comparable across models, and the report names the model but does not
   warn that a local model was used.
4. **num_ctx is a silent cliff.** At the shipped 4096, gemma4 returned a bare code fence for every
   claim — the prompt alone had outgrown the window at V1.5 and Ollama truncates from the tail,
   where the output schema lives. Fixed in `40eb472`, pinned by a test.

## The span check needed hardening before it could carry a verdict

Promoting `span_is_verbatim` from a confidence cap to a verdict rule made its false negatives
expensive, and it had some. PDF text keeps the hyphen of a word split at a line break — `energy-
dependent` where the extractor left the space, `informa-tion` where it closed it up — and a model
quoting that sentence writes the word whole. Three of the hosted run's four unverified spans were
that artefact, not invention. Dropping every hyphen on both sides of the comparison (`c5ffd35`,
hardened in the follow-up) fixes them:

| | before | after |
|---|---|---|
| gemini-3.5-flash-lite | 4 / 20 unverified | **1 / 20** |
| gemma4:e2b (local) | 13 / 18 unverified | **13 / 18** |

Real quotations stopped failing on extraction noise; not one fabrication was excused. The single
remaining hosted failure is a genuine partial invention — the model wrote a plausible opening for a
sentence and continued with the evidence's own words.

## Fit for purpose

gemma4:e2b is not usable as the judge for this task. `qwen2.5:7b` is the other local model installed
and is untested here. Any local candidate should be screened on the 32-case eval set with
`scripts/evaluate_judge.py` before a paper is audited with it — the span-verified rate is the cheapest
single screen.
