# Citation audit integrity — judge real sentences, report only what was judged

*2026-09-14 · against HEAD `72af0bf` (branch `synthesis-depth`)*

## 1. Purpose

Two reports produced by Tab 1's in-text citation audit were examined
(`doi_10.1093_gji_ggag077`, a geophysics paper with author–year citations;
`arxiv_2108.10114v3`, a physics paper with numeric citations). Neither
contained a single verdict the judge actually produced, and both presented
verdict-shaped output anyway. The problems are upstream and downstream of
the judge, not in it:

**Upstream — the claim the judge is handed is not a sentence.**

| measured over 40 corpus seed papers (2,674 claim–reference pairs) | |
|---|---|
| median "claim" length | 102 words |
| claims over 120 words (longest 636) | 39 % |
| claims containing a glued `word.Word` boundary | 77 % |
| pairs from sentences carrying ≥ 3 citations | 71 % |
| pairs `max_claims=20` leaves unjudged | 72 % |
| of the 20 judged per paper, cluster citations | 79 % |
| in-text refs with no GROBID `target` (28,964 refs corpus-wide) | 16 % |

Four mechanisms, all in `extract_seed_citation_claims`:

1. **GROBID's own sentence segmentation is thrown away.** 304 of 324 TEIs
   carry `<s>…</s>`; `_clean(p)` calls `get_text()`, which concatenates
   `</s><s>` with no whitespace, and `split_into_sentences` needs whitespace
   after the full stop. The paragraph survives as one "sentence".
2. **Splitting runs *after* citation placeholders are inserted, and the
   placeholder embeds the citation's own text** (`__CITE_1_b6_C.A.N. da
   Costa et al. 2018__`). The splitter cuts at `C. ` and half the token
   leaks into the report verbatim — four times in the geophysics report.
3. **Stripping a narrative citation deletes the sentence's subject.**
   "explored by Landa et al. (2006) with…" becomes "explored by with…";
   "Following Silva et al. (2021), the…" becomes "Following, the…";
   "(Vasconcelos et al. 2017; Cui et al. 2020)" becomes "(;; )".
4. **A targetless numeric citation is resolved by list position.** In
   `arxiv_2007.12504`, `b20` is a footnote GROBID swept into `listBibl`, so
   from `[21]` on, position ≠ number (`[21]` is `b53`). The `index` shown to
   the reader is wrong for the same reason, and the last-resort surname
   match is a substring test (`"he" in txt`).

Two further upstream findings from the physics report: one introduction
sentence (`"…search engines [15,] [16], genetic algorithms [17] and …
strategies [18]…[27] have been proposed"`) consumed 13 of the 20-claim
budget; and software citations (`LAPACK Users' Guide`, `Julia`, `Algorithm
832`) were audited as claims, so the geophysics report asks the reader to
upload the LAPACK manual.

**Downstream — the report says things the judge never said.**

In the physics report, 46 citations are labelled *⚪ Unclear / Insufficient
Evidence — fragile or fragmentary evidence*. They decompose exactly into 26
`cap_exceeded`, 10 `no_evidence`, 9 `retrieval_failed` (*Failed to connect
to Ollama*), 1 `call_failed` (same). Zero were judged. Every one of the 83
rows shows `Confidence: Medium · Evidence Sufficiency: partial` — the
`.get()` defaults in `_format_entry` — and a reliability line saying the
citation "does not provide sufficient, retrievable evidence". The run
continued through all 20 attempts with the backend down and wrote a full
report. 75 of 83 sources were graded *📄 Standard Primary — peer-reviewed
primary publication*, which `assess_source` returns for any bare DOI; the
set includes a *J. Phys.: Condens. Matter* topical review and an SEG
Encyclopedia chapter.

## 2. Design

### 2.1 Claims are GROBID sentences, citations are tokens (`shared/claim_text.py`)

A new pure module owns the text work; `seed_audit.extract_seed_citation_claims`
calls it and keeps its public signature.

- `paragraph_sentences(p)` returns the cleaned text of each `<s>` child
  when the paragraph has them, else `split_into_sentences(clean_text(p))`.
  The regex splitter is the fallback, never the first choice.
- Each `<ref type="bibr">` is replaced by a period-free token `⟦C{n}⟧`;
  a side table `cites[n] = {"target", "txt", "ref", "resolution"}` carries
  what the old placeholder embedded. Tokens survive any splitter.
- `render_claim(sentence, cites)` builds the judge's claim: a **numeric**
  cite (`[12]`, `[9,`, `10]`) or a cite inside an unclosed `(`/`[` is
  removed; a **narrative** cite (`Landa et al. (2006)` outside brackets)
  is replaced by its own text, so the sentence keeps its subject.
  `tidy_punctuation` then removes `( ; )`, `[ ]`, doubled separators and
  space-before-punctuation. `render_sentence` builds the display form
  (`[Landa et al. (2006)]`).
- `claim_quality(claim)` returns a list from `{placeholder_residue,
  fragment, too_long}` (fragment: fewer than 5 words, or starts with a
  lowercase letter or a closing bracket/comma; too_long: over 80 words).
  `placeholder_residue` or `fragment` means the pair is **not judged**
  (`outcome = "malformed_claim"`); `too_long` is judged and flagged.
- `classify_citation_role(sentence, token, ref_info)` returns one of
  `evidential | software | pointer | method` from the eight words before
  the token and the reference's title/raw string (`software`: *package,
  code, library, Users' Guide, Algorithm N, …*; `pointer`: *see, cf., e.g.,
  for a review, as described in, Ref., and references therein*; `method`:
  *following, according to, adapted from, we use/adopt/employ*). Only
  `evidential` pairs are judged; the others get `outcome = "not_a_claim"`,
  are listed in their own report section, and are excluded from the
  missing-references table and the paragraph deferral ratio.
- `sentence_context(sentences, idx)` returns previous + «claim» + next
  sentence in display form — the context handed to the judge (§2.4).

### 2.2 Reference resolution never guesses silently (`seed_audit.py`)

`_resolve_ref(target, txt, …)` returns `(ref_info | None, resolution)`:

1. `target` → `bib_by_id` (`resolution = "target"`);
2. else numeric `txt` → `number_map` built from every ref in the same
   document that *does* carry a target and a numeric text (`[9]` linked to
   `#b8` teaches `9 → b8`) (`"number_map"`);
3. else numeric `txt`, when every observed `(k, xml_id)` pair satisfies
   `xml_id == f"b{k-1}"` → list position (`"position"`); an inconsistent
   map (the footnote case) disables this step;
4. else non-numeric `txt` containing a surname as a whole word **and** the
   reference's year → that reference (`"surname"`);
5. else `None` (`"unresolved"`) → `outcome = "unresolved_ref"`, never judged.

`ref_info` gains `venue` (`monogr/title[@level="j"]`) and `is_monograph`
(`monogr/title[@level="m"]` with no `analytic` title) for §2.5.

### 2.3 Budget goes to the citations that matter; runs stop when the backend is down

- Each claim records `section` (via `extract.normalise_section_kind` on the
  enclosing `<div><head>`), `cite_count` (tokens in its sentence),
  `paragraph_index`, `sentence_index`.
- `prioritise_claims(downloaded)` orders by
  `(SECTION_RANK[section], cite_count, paragraph_index, sentence_index)`
  with `results/discussion/conclusion → 0, methods → 1, else → 2`, and
  keeps at most `MAX_PAIRS_PER_SENTENCE = 3` pairs per sentence; the rest
  are `outcome = "cluster_skipped"`. `max_claims` still caps attempted
  pairs; the overflow is `cap_exceeded`, as now.
- The judging loop counts consecutive `retrieval_failed`/`call_failed`
  outcomes. At `MAX_CONSECUTIVE_BACKEND_FAILURES = 3` it stops; every
  remaining pair is `outcome = "not_attempted"`, and the report carries
  `aborted = {"reason", "after_attempted"}`. `cross_check_seed_audit`
  re-judges `not_attempted` on the next run.
- Every audit is also written to `data/raw/seed_audits/history/
  <stem>_<timestamp>_audit.json`; the latest overwrites `<stem>_audit.json`
  as now. Real (claim, evidence, verdict) triples accumulate for the judge
  evaluation set.

### 2.4 The judge sees the sentence in its paragraph

`prompt.md` gains a `{{CONTEXT_BLOCK}}` slot immediately before
**Claim:**; `build_prompt(claim, evidence, context=None)` fills it with
"**Context** (the sentences around the claim; the claim is the sentence
between « and »; judge only that sentence and use the rest only to resolve
what its words refer to)" or with nothing. `judge()` and `_judge_once()`
take `context=None`; the seed audit passes `item["context"]`; Agent 8 passes
previous + sentence + next from the draft. The six regression cases run
without context and are unchanged.

Judged items also record `evidence_sections`, `evidence_pages` (from hit
metadata), `escalated`, `evidence_hits`.

### 2.5 A rating is a statement about a verdict, and only a judged item has one

- Failure outcomes leave `judgement = None`. The only source of
  `"Unclear / insufficient evidence"` is the judge.
- `compute_totals(results)` (shared by `audit_seed_citations` and
  `cross_check_seed_audit`) counts verdicts over `outcome == "judged"` only
  and adds `not_assessed: {outcome: count}` and `coverage: {downloaded,
  attempted, judged}`.
- `evaluate_reliability(…, outcome="judged")`: when `outcome != "judged"`
  the rating is `UNRESOLVED` with the explanation for that outcome
  ("Not assessed — the 20-claim budget was reached before this citation",
  "Not assessed — the model backend was unreachable", …). A judged
  `Does not support` becomes the new rating `UNSUPPORTED` ("the cited paper
  does not report this"), no longer folded into `UNRESOLVED`.
- `assess_source`: a monograph is `UNKNOWN` ("book or book chapter — not
  graded"); a DOI with no venue is `UNKNOWN` ("DOI only; venue unknown");
  `STANDARD_PRIMARY`/`RIGOROUS_PRIMARY` require a venue, and every
  rationale states it is a heuristic not verified against Crossref.
- `explain_rubric_verdict` branches on `outcome` first with a specific
  sentence for `cap_exceeded`, `not_attempted`, `cluster_skipped`,
  `not_a_claim` (by role), `malformed_claim`, `unresolved_ref`; the
  "insufficient evidence" sentence is reachable only from a judged item.
- The Markdown report opens with an **Assessment coverage** table
  (downloaded / attempted / judged / not assessed by reason) and, when
  present, an abort banner. Per-citation blocks print `Confidence` and
  `Evidence Sufficiency` only for judged items and otherwise
  "Not assessed — <reason>"; judged blocks add role, section of the citing
  sentence, evidence provenance (sections and pages), and the rubric
  warnings Agent 8 already prints (`span_verified is False`,
  `rubric_mismatch`, `escalated`). Deferred items are grouped one block per
  paragraph. The reliability table gains an *Unsupported* row.
- Tab 1 gains a *Not assessed (N)* filter tab, an *Unsupported* reliability
  tile, an abort banner, and outcome-specific badges for non-judged items.

## 3. Outcome vocabulary (complete)

| outcome | judged? | set where |
|---|---|---|
| `judged` | yes | `_judge_claim_entry` |
| `no_evidence`, `retrieval_failed`, `parse_failed`, `call_failed` | no | `_judge_claim_entry` |
| `not_attempted` | no | breaker in `audit_seed_citations` |
| `cap_exceeded`, `cluster_skipped` | no | `prioritise_claims` / cap |
| `not_downloaded`, `deferred_paywalled` | no | manifest matching (as now) |
| `not_a_claim`, `malformed_claim`, `unresolved_ref` | no | partition before matching |

## 4. Not in scope

Per-slot supporting spans (a prompt-schema change; measure the judge first
with the harness in `judgement/evalset.py`). Abstract-level evidence for
paywalled sources. Crossref/OpenAlex lookups in the source assessor. Model
tiering. Changing the `>50 %` deferral rule.

## 5. Placement

`research_assistant/shared/claim_text.py` (new, pure) with
`tests/test_claim_text.py`; changes in `shared/seed_audit.py`,
`judgement/policy.py`, `judgement/source_assessor.py`, `judgement/judge.py`,
`judgement/prompt.md`, `agents/agent8_verifier.py`, `app.py`; tests extended
in place. TEI fixtures are inline strings in the tests, as the existing
ones are.
