# Citation audit integrity — what changed, what the live runs showed, what is left

*2026-09-14 · after the nine plan tasks (`32a61e2…cce44ab`) and nine follow-up
commits (`51b30fb…d089e2d`) · suite 724 → 748 passing*

## 1. Was there an improvement? Yes — measured three ways

### 1.1 What the judge is handed (40-paper corpus sample, 2,835 pairs)

| | before (`72af0bf`) | after |
|---|---|---|
| median claim length | 102 words | **27** |
| claims over 120 words | 39 % | **0 %** |
| claims with a glued `word.Word` boundary | 77 % | **0 %** |
| placeholder tokens leaking into claims | present | 0 |
| citation role assigned | — | 2,663 evidential · 85 pointer · 44 method · 43 software (+ definition) |

### 1.2 What the report says (the two reports the user brought)

Every failure mode in them is now structurally impossible: `Confidence:
Medium · partial` on unjudged rows (the `.get()` defaults are gone), 46
"Unclear" rows that were 0 verdicts (`judgement=None` on every failure
path; `judged + Σ not_assessed == total` is a tested invariant), a run
continuing with Ollama down (3-consecutive-failure breaker → `not_attempted`
+ abort banner), LAPACK audited as a claim (`not_a_claim`), 75/83 sources
"peer-reviewed" from a bare DOI (venue required; rationale says heuristic).

### 1.3 A live audit, same paper, three runs tonight (`arxiv_2007.12504v1`, gemini-3.5-flash-lite, v2 index)

| | run 1 (plan as implemented) | run 3 (after follow-ups) |
|---|---|---|
| pairs found / judgeable | 65 / 10 | 64 / **41** |
| `unresolved_ref` | 46 (all-or-nothing position rule refused the whole paper) | 18 (each one names both candidate entries) |
| judged | 6 | 19 |
| `Does not support` | **5**, all `High` confidence with `insufficient` evidence and no span | 1, with `sufficient` evidence |
| `Supports` | 1 — a 119-page review citing someone else, rated HIGH | 2 — both verbatim; the secondhand one flagged |
| fabricated fields | 0 | 0 |
| coverage line | 6 of 6 attempted | 19 of 19 attempted |

Records of all three runs are in `data/raw/seed_audits/history/` for the
evaluation harness to harvest.

## 2. What the follow-up commits fixed, and the evidence for each

| commit | finding | fix |
|---|---|---|
| `51b30fb` | 304/307 `downloaded.json` paths point at the old `storage` mount; every reference counted as missing on this machine | manifest paths resolved by basename under `PULLED_PDFS_DIR`; `recorded_path` kept |
| `fda3eb2` | Tab 3 still defaulted confidence to Medium and relation to Unclear for failed entries | `evaluate_reliability(..., outcome=)` in Agent 8; `orphaned`/`unresolved` explained |
| `fbec53e` | 72 % of pairs sit under topical headings ("DFT-based tight-binding Hamiltonian") → no section rank → budget spent in document order | leading headless div = introduction; unmapped sections ordered later-paragraphs-first |
| `d712a7a` | 5/6 verdicts "Does not support · High · insufficient · no span": *the retrieved passages don't mention it* reported as *the paper doesn't say it*. The one Supports rested on "…has been proved ¹⁸" | absence from insufficient evidence → Unclear, recorded as a violation; `span_cites_others` → secondhand, capped MODERATE; UNSUPPORTED wording says "retrieved passages"; Physics Reports / Rep. Prog. Phys. are reviews |
| `03a1adf` | GROBID's `target` links off by one for `[24]`–`[34]` while list position was right; the plan's rule trusted links and refused position for the whole paper | two witnesses: agreement resolves; two real entries that disagree are `ambiguous`, both named, never judged; a footnote at the position yields to the link. "See SM for details … [26]" is a pointer |
| `e286762` | finding assertion "temperature has little effect on the inversion procedure" for a claim about δE_cor — decomposed from the *context* sentence | assertion drift (< 40 % content-word overlap; finding vs claim, scope vs claim+evidence) recorded and confidence capped; the six rubric examples are the regression floor. "The conductance reads G = …" is a definition, not a claim |
| `8f2a06e` | **89 % of the 16,336 indexed chunks contain glued sentences** (`two-fold.First`): `extract._text` joined `</s><s>` with "". BM25 saw one token, the judge read "corrupted text", a quoted span that repaired the glue failed the verbatim check and a correct Supports rated LOW | a space at every `<s>` boundary and nowhere else. **Takes effect on the next ingest; the live index is unchanged until re-indexed (§3.1)** |
| `d089e2d` | "`[35]` This small set of conditions…" — GROBID closed the previous sentence before its trailing marker; the next sentence was judged against a citation it never carried. And a Supports whose span joined two grounded sentences with "…" rated LOW | a numeric marker before any word moves to the previous sentence (narrative "Smith et al. (2020) showed…" stays); a span is verbatim when every ellipsis-separated piece is |

## 3. What is left, ranked by how much hallucination it removes per unit of work

The judge's own outputs are now the smallest source of error visible in
the runs; the inputs it gets are the largest. Everything below is in
evidence in `data/raw/seed_audits/history/`.

### 3.1 Re-index (operational, highest value) — the extractor fix is in, the index is not

With `CITATION_INDEX_VERSION=3` the new index builds beside v2 and nothing
opens v2 for writing. Expected effect: fewer "corrupted evidence" verdicts,
BM25 matches across sentence boundaries, verbatim-span pass rate up (the
model stops "repairing" glued quotes), and the `too_long`/glue residue in
Agent 8's re-retrieved evidence disappears. Measure with the same paper
before and after: `evidence_sufficiency` distribution and `span_verified`
rate over the 19 judged items.

### 3.2 Retrieval is now the bottleneck, not judgement

18 of 20 judged items in run 2 had `insufficient` evidence; the sources are
a 400-page textbook (Datta), two *Reviews of Modern Physics* articles and a
119-page *Physics Reports*. Three chunks of those is nothing.
- **Scale the evidence budget with source length** (chunks per source, up
  to a cap), and select the section first: the v2 index carries
  `section`; a two-stage pick (which section mentions the claim's terms →
  chunks within it) costs no model calls.
- **Targeted second query for negatives.** When the first verdict is
  Unclear/DNS with `insufficient`, re-retrieve with the judge's own
  `finding` assertion plus the claim's numbers and symbols, then judge
  once more. Today escalation only widens *k* on the same query.
- **Do not let a caption be the only evidence.** 5 of the retrieved
  sections in run 2 were `caption`; one Supports rests on "FIG. 2. 2D
  histograms…". Record it (it is recorded) and cap at MODERATE, or exclude
  captions from the top-k unless the claim is about a figure.

### 3.3 The judge is not deterministic — verify the verdicts that matter

Same input, temperature 0: run 2 quoted one sentence verbatim, run 3 joined
two with "…". Verdicts that a reader acts on (Supports, Contradicts,
Does not support) should be confirmed by a second pass — a second model
(the pitch's "mid-tier validates") or the same model over the widened
evidence — and a disagreement lands as Unclear + expert queue. In run 3
only 3 of 19 verdicts were actionable, so the cost is small. The harness's
`--runs N` stability metric is the number to watch.

### 3.4 Measure — there is still no labelled set

`judgement/evalset.py`, `harvest.py`, `labeling.py`, `metrics.py`,
`transforms.py` exist; `cases/` still holds only the six prompt examples.
The three run records above are the first real (claim, evidence, verdict)
triples. Twenty minutes of labelling gives the first agreement number;
without it, every change above is argued, not measured.

### 3.5 Smaller deterministic guards, in order of value

- **Reason grounding**: every number, unit and chemical formula in `reason`
  must occur in the claim or the evidence; flag `reason_ungrounded`.
- **Evidence quality gate** until the re-index lands: alphabetic/word-like
  ratio of the assembled evidence below a threshold → `evidence_unreadable`
  (one more outcome, four registries), no model call.
- **Group claims** ("numerous ways of extracting TB Hamiltonians
  [15]–[20]"): judge each paper with a softened question ("is this paper
  an instance of X?") rather than the group claim as stated, or keep the
  honest Unclear.
- **Comparison citations** ("similar in spirit to [35]", "akin to",
  "inspired by") are method-like; add the cue words to `_METHOD_RE`.
- **Tab 3 parity**: Agent 8 now passes context and rates honestly, but has
  no citation roles, no cluster cap, no breaker.
- **Source grade** beyond heuristics: Crossref `type` and `update-to` are
  already in the response Agent 2 discards (`agent2_fetcher.py:132`).

## 4. Operational notes

- `data/ingest.lock` is a 0-byte file from 01:56 today; if nothing is
  ingesting, it is stale and blocks `tests/test_ingestion.py`.
- `BRIEFING.md` at the repo root is an untracked copy the implementing
  agent left; the canonical one is `.agents/audit_integrity/BRIEFING.md`.
- No live check with the backend *down* was run tonight (it would have
  meant stopping Ollama while other things may depend on it); the breaker
  is covered by `test_three_consecutive_backend_failures_stop_the_run`.
