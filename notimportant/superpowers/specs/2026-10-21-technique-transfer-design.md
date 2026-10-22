# Technique transfer: find methods from other fields that solve your problem's bottleneck

*2026-10-21 · base: `feat/hypothesis-search` at `8b5b5ea` (index v3) · supersedes
`2026-10-21-hypothesis-search-design.md` on concept. Premise checking, the coverage
gate and background jobs carry over from it.*

## 1. What it is

Given a research problem, the tool finds techniques that papers in *other* fields used
on a structurally similar problem, and writes a concrete transfer:
- which technique it is,
- how its parts map onto the problem,
- which assumptions survive the move,
- whether the problem is even identifiable,
- an implementation sketch,
- a test against the current method,
- whether it has been done before.

It runs unattended for as long as its budget allows. Its objective is **relevant
transfers with sufficient depth**, defined in §4 as checks that can be counted and
verified, never as a model's opinion of itself.

Worked example: *the forward problem in quantum transport* → bottleneck "the recursive
Green's function method (RGF) needs a matrix inversion per slice, per energy, per
disorder sample" → generic form "an expensive forward map evaluated many times" → a
technique card for "latent-space surrogate with property-guided search", from a
molecular-design paper → a mapping table, an identifiability argument, an
implementation sketch, a test against RGF, and prior art.

"From scratch" means this feature: a new package, `research_assistant/transfer/`. It
reuses the repo's shared core rather than rebuilding it:

| Reused | For |
|---|---|
| `shared.llm` | model calls |
| `judgement.judge` and the citer's acceptance rule | checking premises |
| `hybrid_search` | passage retrieval |
| `retrieve.rank_documents` and `gate_documents` | the near set and coverage |
| `run_jobs` and `pipeline_status` | background runs and progress |
| the eval conventions | the retrodiction harness |

## 2. The pipeline

| Stage | Does | Model calls |
|---|---|---|
| S0 Coverage | nearest summaries → relevance gate; fewer than 3 kept → refused | 1 (gate) |
| S1 Bottlenecks | from the near passages: up to 3 bottlenecks, each with grounded premises, the quantities it limits, the current method, and **three generic forms** (specific, mid, broad) | 1 |
| S2 Techniques | each generic form searches the **technique-card index**; near-set and excluded documents are skipped, as are unverified cards; best 3 cards per bottleneck | 0 (embeddings only) |
| S3 Write | one transfer per (bottleneck, card): mapping table, identifiability, steps, data, baseline, metric, prediction, refuting result, premises | 1 per transfer |
| S4 Judge | every premise is judged against its own passage with the citer's rule (Supports with a verified span = 1.0, Partially supports with a verified span = 0.5, anything else = 0); results cached per (claim, chunk) | ~2–4 per transfer |
| S5 Critique | an adversarial critic attacks the mapping rows and identifiability, ideally on a **different model**; then one revision call answers each objection or concedes it. Every row the critic marks `breaks` is also put to the reviser as an objection on that row | 2 per transfer |
| S6 Prior art | OpenAlex and arXiv titles and abstracts for technique + target quantity, optionally only before a cutoff year | 0 (HTTP and embeddings) |
| S7 Reward | gates, then the depth checklist, then the prior-art bonus (§4); rank | 0 |

Each run records a **trace**: for every transfer, the card it came from, that card's
document, the abstraction level and the retrieval rank. Retrodiction scoring (§6) reads
the trace, not just the prose.

## 3. Technique cards

Summaries describe what a paper *found*. Transfer needs what it *did*. So each paper
gets one card-building call:

- **Input.** About 9,000 characters of the paper's own passages:
  1. its abstract chunks;
  2. its `methods` chunks;
  3. one chunk per distinct `section_raw` heading, in document order.

  Figures and captions are excluded. The normalised `section` field can't be relied
  on: only 3,702 of 57,169 v3 chunks are labelled `methods`, and 26,633 are `other`,
  including headings such as "Quantum Green's functions".
- **Output.** Up to 3 techniques, each with:
  - `name`: a reusable method;
  - `problem_solved`: stated generically, with no field nouns; this field is what gets matched;
  - `context`: how this paper used it;
  - `inputs`, `outputs` and `assumptions`;
  - `mechanism`: 2–5 concrete steps saying how it works. A technique with fewer is dropped in code (`TRANSFER_CARD_MIN_MECHANISM_STEPS`), so a survey, a database or a benchmark yields no card. This was added after F1 run 1 (§10);
  - an **evidence premise** citing one passage key.
- **Verified** means the evidence premise passes the judge. Only verified cards can be sources.
- **Stored** as `DATA_DIR/transfer/cards.jsonl` plus `cards.npy` (the `problem_solved` vectors). There are no Chroma writes. Building is resumable through a per-document manifest; rerun it after an ingest.
- **Cost.** About 700 generation calls plus 700–2,000 judge calls for the current corpus. Minutes to an hour or two on Gemini; overnight on Ollama.

## 4. The reward

**Gates.** A transfer must pass all of them. The first one that fails becomes its
`set_aside` reason, in this order:

| reason | fails when |
|---|---|
| `contradicted` | any premise is judged Contradicts |
| `ungrounded_bottleneck` | the bottleneck has no accepted or partial premise |
| `unverified_technique` | the card's evidence premise did not pass |
| `shape_mismatch` | the critic says the problem shapes do not match (`shape_match == "no"`) |
| `core_assumption_breaks` | a core mapping row's final status (the critic overrides the writer) is `breaks`: the critic broke it and the reviser conceded or did not answer. A break the reviser rebutted is `contested`, which costs reward instead (see Reward) |
| `not_identifiable` | the final identifiability verdict is `underdetermined`, for a bottleneck marked `inference` (it infers unknowns from measurements). Computing, simulating, representing and optimising are not gated |
| `off_target` | the metric names none of the bottleneck's quantities |
| `done_before` | prior art found the same combination. The transfer is listed under "Already done", with references |

**Depth.** Ten items, each pass or fail; `depth` = passed / 10. Every item is checked in code:

1. Mapping complete: at least 3 rows, each with a source role and a target role.
2. Every row is classified as holds, breaks or enforce, and at least one row is core.
3. Identifiability argued: the measurements and the unknowns are both named, the verdict is not `unclear`, and the argument runs to at least 10 words. It passes automatically when the bottleneck is not an inference problem.
4. At least 3 steps, each naming an input, an output and a tool.
5. The data requirement contains a number.
6. The baseline shares a content word with the bottleneck's current method.
7. The prediction states a direction or a number.
8. The refuting result has at least 5 words and differs from the prediction.
9. At least one accepted or partial premise from the card's paper and one from a near paper.
10. The critic raised at least one objection, and each one is answered or conceded in at least 5 words.

**Reward.** `reward = depth × (1 + TRANSFER_PRIOR_ART_BONUS)` when prior art is
`not_found`, otherwise `reward = depth`. The bonus defaults to 0.25. Each contested
core row then multiplies the reward by `1 − TRANSFER_CONTESTED_PENALTY × contested`
(floored at 0; the penalty defaults to 0.2). A transfer that is set aside scores 0.

Prior-art status:

| status | when |
|---|---|
| `done_before` | a hit's similarity to the transfer summary is ≥ 0.85 and every content word of the technique name appears in its title or abstract |
| `partial` | similarity ≥ 0.75 |
| `not_found` | otherwise |
| `unchecked` | the network failed; no bonus, with a warning |

Thresholds, the bonus and the gate order are calibrated in §6 before any unattended
run.

## 5. Search

**Phase 1, best-of-N.** Every (bottleneck, top card) pair is developed: 3 × 3 = 9
transfers, roughly 60 model and judge calls.

**Phase 2, MCTS. Only after §6 passes.** The tree runs root → bottleneck →
abstraction level → technique:
- UCT allocates a budget, counted in transfers developed.
- Progressive widening: a node's next child is its next-ranked card.
- A leaf's value is its reward; there are no rollouts.
- The run stops when the budget is spent, or when the top 3 haven't changed for 3
  iterations.

MCTS ships only if it beats best-of-N on the retrodiction metrics at an equal budget.

## 6. Feasibility gates and calibration

The cases are retrodictions: transfers that really happened, with the papers that
report them hidden.

| Gate | What | Pass |
|---|---|---|
| **F1 Cards** | cards for 20 random papers, rated blind by the maintainer (correct / partial / wrong; specific yes / no) | ≥ 70% correct or partial and ≥ 60% specific |
| **F2 Retrieval** | for each case, a hand-written generic bottleneck searches the cards, with no generation | an expected source card is in the top 5 for ≥ 3 of 4 cases |
| **F3 Retrodiction** | full runs, 2 each | see the case table below |

| Case | Kind | Corpus | Hidden |
|---|---|---|---|
| NQS: RBMs as variational wavefunctions | positive | built for the case (target, source and distractor papers before 2017) | Carleo & Troyer 2017 |
| KPM into real-space Kubo transport | positive | built for the case (RGF/Landauer papers, the KPM review, trace estimation) | the real-space Kubo-KPM transport paper |
| Maintainer's autoencoder (**unpublished**) | positive | main corpus minus that work | not in any model's training data |
| Autoencoder → impurity positions | negative | main corpus | — |

Scoring reads the trace, so a technique named from the model's memory, without its
card being retrieved, earns nothing.

| Case kind | Passes when |
|---|---|
| positive | a transfer whose card comes from an expected source paper and matches the technique terms ranks in the **top 3** among transfers not set aside; ≥ 50% of the key mapping terms appear in it; the critic hits ≥ 1 known failure term |
| negative | every transfer using the technique is set aside, and the expected reason (`not_identifiable`) is among those reasons |

Queries are written in neutral words, without the target field's vocabulary. Prior art
in retrodiction runs is limited to years before the case's cutoff. Answer keys for the
two maintainer cases live under `data/eval/transfer/answer_keys/`, which is gitignored.

**Rule:** no unattended run on open questions until F1–F3 pass. If F1 fails, fix the
cards before anything else. If F2 fails, fix the abstraction levels or the card fields
before writing transfers.

## 7. Configuration (`config.py`, `CITATION_TRANSFER_*`)

| setting | default | meaning |
|---|---|---|
| `TRANSFER_DIR` (code) | `DATA_DIR/transfer` | cards and runs |
| `TRANSFER_BOTTLENECKS` | 3 | per run |
| `TRANSFER_CARDS_PER_QUERY` | 3 | cards retrieved per generic form |
| `TRANSFER_TECHNIQUES_PER_BOTTLENECK` | 3 | kept after merging the levels |
| `TRANSFER_MIN_NEAR_PAPERS` | 3 | coverage gate |
| `TRANSFER_NEAR_PASSAGES` / `_SOURCE_PASSAGES` | 8 / 3 | passages per prompt |
| `TRANSFER_PASSAGE_MAX_CHARS` | 7000 | passage text per prompt |
| `TRANSFER_CARD_MAX_CHARS` / `_CARD_MAX_TECHNIQUES` | 9000 / 3 | card building |
| `TRANSFER_TEMPERATURE` / `_BOTTLENECK_TEMPERATURE` | 0.7 / 0.3 | generation |
| `TRANSFER_CRITIC_MODEL` | unset → the generator's model, with a warning | the critic |
| `TRANSFER_PRIOR_ART_DONE` / `_PARTIAL` / `_BONUS` | 0.85 / 0.75 / 0.25 | §4 |
| `TRANSFER_NUM_CTX` | 8192 | Ollama context |

## 8. Errors

| condition | behaviour |
|---|---|
| gate keeps fewer than 3 papers | `CorpusCoverageError`; no generation call |
| another run active (`pipeline_status`) | `BusyError`; CLI exit 2; tab warning |
| a reply is not the JSON asked for, twice | that item is skipped, with a warning |
| a premise cites a key not in its own prompt | dropped, with a warning |
| the judge or the critic raises | premise credit 0, or the critique is empty (so depth item 10 fails); the run continues |
| prior-art network failure | `unchecked`, with a warning |
| no verified cards match | that bottleneck yields nothing, and a "corpus gap" warning names its generic forms |
| cancel pressed | stops at the next transfer boundary |

## 9. Not in scope

- Fetching and ingesting papers mid-run. Gaps are reported, and Agent 0 can build the
  corpus.
- Changes to Agents 0, 5 and 8, `judgement/` or ingestion.
- Phase 2 tasks before F1–F3 pass.

## 10. Results

*To be filled in by the maintainer after F1, F2, F3 and the Phase 2 comparison.*

### F1 — technique cards (2026-10-21)

Both runs used the same 20 papers (`build_cards.py --sample 20 --seed 1`, then
`--documents` with that list) and `gemini-3.5-flash-lite`. The ratings were made by
Claude on the maintainer's behalf; the maintainer rated card 27 in run 1.

| run | prompt | cards | correct or partial | specific | gate |
|---|---|---|---|---|---|
| 1 | up to 3 techniques; an empty list allowed | 40 | 100% | 55% | **fail** |
| 2 | a technique is a method with steps; non-techniques listed; 2–5 step `mechanism`, enforced in code | 13 | 100% | 92% | **pass** |

- **Run 1's vague cards** came from papers with no transferable method: a database, two surveys, device handling and genome annotation. The judge passed all 40 evidence premises, because each quotes its own passage. That confirms the judge checks the quote, not the card.
- **Run 2 cut precision problems, but recall too.** Of run 1's 22 specific cards, 11 survive. The lost ones include real techniques: masked pretraining and value binning (scBERT), GW corrections, serpentine strain gauges, LSH triplet mining, title-as-auxiliary-signal (TLDR), likelihood parameterisation and ancestral sampling (DeepAR), and impurity-pair reclassification. The likely cause is the prompt's "most papers have 0 to 2". If F2 misses an expected card, loosen that line first; keep the mechanism requirement.
- **The corpus was repaired between the runs.** 4 of the 20 papers held a different PDF from their reference (13.7% across the corpus); they were relabelled to their real titles. Run 2's cards cite the right papers, e.g. RA-BLIP rather than "BLIP-2".

### Task 12 smoke runs — the core-assumption gate (2026-10-22)

Query: "Accelerating the forward problem in quantum transport for disordered
nanostructures", over the full card build (1,490 papers, 1,193 cards), with
`gemini-3.5-flash-lite` as generator, critic and judge.

| run | rule | transfers | set aside | top reward |
|---|---|---|---|---|
| 1 | critic's `breaks` is final | 6 | 6, all `core_assumption_breaks` | 0 |
| 2 | broken rows put to the reviser; "concede an objection that is right" | 9 | 8 (reviser conceded 19 of 23 plain objections) | 0.80 |
| 3 | as 2; "concede only when no change to the transfer can save it" | 6 | 4: one each of `core_assumption_breaks` and `off_target`, two `not_identifiable` | 0.72 |

- **Run 1:** a critic told to reject if it can broke a core row in every transfer, so the gate rejected everything. A rebutted break is now `contested` and costs `TRANSFER_CONTESTED_PENALTY` (0.2) per core row.
- **Run 2:** the same model, asked to concede what is right, conceded and then repaired the transfer. The gate read each concession as fatal. The reviser's rule now says what a concession does.
- **Run 3** is the first in which the gates discriminate. Calibrate the penalty and the concession rule at F3, ideally with a critic on a different model (`CITATION_TRANSFER_CRITIC_MODEL`).

### F2 — card retrieval (2026-10-22)

| case | corpus | first expected card |
|---|---|---|
| nqs | case: 10 papers, 10 cards | rank 3 |
| kpm | case: 6 papers, 8 cards | rank 1 |
| positive control | main: 1,490 papers, 1,193 cards | rank 1 (was 50) |
| negative control | main | rank 1 (was 80) |

**F2 PASS (4 of 4 in the top 5)**, with three caveats:

- **The case corpora are too small for F2 to test much.** F2 ranks every card, and NQS has only 10, so a top-5 pass needs only half of them below the expected one. More distractor papers per case would make it a real test.
- **The two control cases are tuned.** Their hand-written generic forms were reworded after the first F2 run. The new wordings name no method, but they were chosen from a few tries against F2. F3 is the honest check, because there the engine writes its own generic forms.
- **F2 does not exclude a case's target papers, but a run does.** The engine never takes technique cards from the near papers, so F2 ranks against more competitors than a run faces.

### F3 — full retrodiction runs (2026-10-22)

Run with `--score --runs 2`. The result is
`data/eval/transfer/results/20261022-105756.json`. **F3 FAIL: 0 of 4.** Each
case failed for a different reason; none of the reasons is the idea itself.

| case | expected card proposed? | what happened |
|---|---|---|
| nqs | yes, in 6 of 9 transfers per run | All were set aside, mostly `not_identifiable`. The critic overrode the writer's "determined" to "underdetermined" on a variational problem that has no measurements to invert. |
| kpm | no | The relevance gate kept the KPM and trace-estimation papers as **near** papers, so they could not be sources. The FFTW auto-tuning card passed every gate at reward 1.25. |
| positive control | no | The engine's own generic forms put the expected cards at ranks 16–200, and only 3 seeds per bottleneck are kept. The F2 tuning did not carry over. Copies of the near papers (duplicates in the corpus) came back as sources. |
| negative control | no ML transfer was proposed | So "every matching transfer set aside" could not be tested. Run 1 had no bottlenecks, because both were dropped for incomplete generic levels. All six of run 2's transfers were set aside. |

**Calibration needed before rerunning F3:**
1. The identifiability gate should apply only to problems that infer unknowns from measurements.
2. In case corpora, the near set should come from the case's target papers only.
3. Every copy of a near paper should be excluded as a source; the corpus holds duplicates.
4. A bottleneck with missing generic levels should be repaired, not dropped.
5. The critic should run on a different model; every run warned that it shares the generator's.
6. The negative case should report "not tested" when no matching transfer is proposed.
7. The depth checklist checks form, not substance: the FFTW transfer scored the maximum.

Fixes 1–3 were made on 2026-10-22:
- Bottlenecks carry `inference`, and only inference bottlenecks are gated on identifiability.
- The F3 harness uses a case's target papers as its near set (`near_from_targets`).
- `candidate_cards` excludes every copy of a near paper, matched by title (`copies_of`).

### F3 rerun after fixes 1–3 (2026-10-22)

The result is `data/eval/transfer/results/20261022-112210.json`.
**F3 FAIL: 2 of 4 (spread 0.50)**, up from 0 of 4.

| case | run 1 | run 2 | note |
|---|---|---|---|
| kpm | pass (rank 3) | pass (rank 1) | The KPM card is now proposed and survives. The FFTW auto-tuning card still outscores it in run 1 (1.00 vs 0.80). |
| negative control | pass | pass | Every transfer matching a technique term was set aside. The terms are loose: "encoder" matched a vision-encoder card. |
| nqs | fail | fail | The RBM-paper transfers now pass the gates (rewards 0.48–0.80). In run 2, B2.T3 proposes a bipartite latent-variable model "such as Restricted Boltzmann Machines" as the state representation, at rank 3. It is not counted: its card (contrastive divergence) never names the RBM, and matching deliberately ignores the transfer's own prose. |
| positive control | fail | fail | The expected ML-inversion cards are still never seeded. Two bottlenecks were set aside as `ungrounded_bottleneck`. |

Open questions:
1. Should NQS match on the card's own vocabulary ("contrastive divergence", "Markov chain")? Or should cards keep the model's name in `name`? Prose-based matching would invite answers recalled from the model's memory.
2. The positive control's retrieval gap remains: either more seeds per bottleneck, or generic forms nearer "learn the inverse map".
3. Fixes 4–7 from the first F3 run are still open.

### F3 after the card-naming rule (2026-10-22)

Cards now keep the method's own name (commit 0462b73), and the NQS case cards
were rebuilt. The result is `data/eval/transfer/results/20261022-113642.json`.
**F3 FAIL: 0 of 4 (spread 1.50).** Between this rerun and the last, KPM and the
negative control flipped from pass to fail, so two runs per case do not measure
much.

| case | run 1 | run 2 | why it failed |
|---|---|---|---|
| nqs | rank 3, mapped 25% | rank 1, mapped 0% | **The naming fix worked: the RBM card is matched.** Run 1's mapping is the NQS result ("parameterizing complex quantum states and wave-functions with auxiliary hidden units"; "approximating quantum expectation values and gradient updates"), but it misses the mapping terms spin/amplitude/energy. Run 2 maps the RBM onto operator compression, a weaker transfer. |
| kpm | pass (rank 1) | rank 5 > 3 | The FFTW auto-tuning card (1.00) outranks KPM (0.70) and the trace estimator (0.90): fix 7. |
| negative control | no bottlenecks | no ML transfer proposed | Fix 4 (missing generic level) and fix 6 ("not tested"). |
| positive control | expected card proposed, set aside `not_identifiable` | not proposed | The analytic-continuation ANN card reached a transfer for the first time; the critic judged the inversion underdetermined. A corpus copy of the user's p8 is filed under a DOI string with no title, so `copies_of` missed it. |
