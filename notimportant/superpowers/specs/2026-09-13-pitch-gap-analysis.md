# Citation Needed! — the pitched workflow vs. what is built

*2026-09-13 · against HEAD `bbe26c6` (branch `synthesis-depth`) and the pitch
`CitationNeeded_Pitch_Final.pptx.pdf` (DogPatch Labs, 14 September 2026)*

Legend: ✅ built · ⚠️ partly · ❌ not built · 📋 planned (spec + plan written, not built)

## 1. Slide 4 — "An agent that reads like a careful researcher"

| step (as pitched) | what the code does | gap |
|---|---|---|
| **A scientific paper** | Tab 1: upload a PDF (also: search by idea, or a link) | — |
| **Find its claims** — "this material improves cooling by X%" | ⚠️ `seed_audit.extract_seed_citation_claims`: the sentences that *carry a citation marker*, mapped to the bibliography via GROBID's TEI | Claims **without** a citation — the literal "citation needed" case — are not found in an uploaded paper. Agent 5's citation-need check does exactly this, but only over the user's own draft. One sentence = one claim; compound sentences are flagged (Agent 8) but not split. |
| **Fetch the sources it cites** — "chase the paper trail" | ✅ GROBID reference list → Unpaywall → Europe PMC → arXiv, with retry of transient failures | **The pitch's own numbers show the wall**: 34 of 36 references paywalled (slide 3); 78 of 90 citations "Paywalled / Unchecked" (slide 5). There is no fallback when the PDF is closed — no abstract-level evidence (OpenAlex / Semantic Scholar abstracts and S2 *citation contexts* are open even when the paper is not), no guided "upload the ones you have access to, re-audit" flow (a manual drop into `pulled_pdfs/` works; matching by DOI/title exists in `_match_downloaded_paper`; nothing in the UI leads there). No second hop — "citation chains" (slide 2) stop at one reference level. |
| **Check the source actually backs it up** | ✅ the strongest part: rubric V1.4 with three slots (finding / scope / strength), verdict enforced in code, verbatim-span check, one escalation to more evidence; the seed audit UI with tabs and downloads | Accuracy is unmeasured: the judge-evaluation set (harvest, human labels, transforms, held-out harness) is 📋 written, not built. No number exists to say "X % agreement with an expert". |
| **Judge how much to trust it** — "method quality, sample size, expert input" | ❌ absent. The rubric's `confidence` is the judge's confidence in its *own verdict*, not the source's trustworthiness | **This is the Source Assessor of slide 6 and it does not exist in any form.** |
| **A trustworthy evidence table** — claim → source → how confident to be | ⚠️ per-claim expanders (verdict, confidence, span, reason); `.md` and `.json` downloads; five tiles | It is a *relation* table. There is no reliability column — no HIGH / MODERATE / LOW / UNRESOLVED anywhere in the code. |

## 2. Slide 6 — "Supported ≠ trustworthy. Contradicted ≠ wrong."

| component | state | detail |
|---|---|---|
| **1 · Relation Judge** | ✅ | `judgement/judge.py` + `prompt.md`; `Supports / Partially supports / Contradicts / Does not support / Unclear-insufficient`; slots aggregated by a fixed rule (`derive_judgement`); rubric mismatches recorded. |
| **2 · Source Assessor** — primary or secondary, well-documented method or not | ❌ | Nothing assesses the source. The cheap signals are already within reach: Crossref (Agent 2 resolves DOIs there) returns `type` (journal-article / posted-content / book-chapter) and `is-referenced-by-count`; OpenAlex returns `type`, `cited_by_count`, `is_retracted`; v2 chunks carry `section`, so "has a methods section" is a lookup; a review is recognisable from its title and length. A model pass over the source's methods section for "sample size / conditions stated" is one call per source, once. |
| **3 · Reliability Policy** — a fixed deterministic rule → HIGH / MODERATE / LOW / UNRESOLVED | ❌ | Trivial once 2 exists: a pure function over (relation, source grade) — and the pitch's core rule is a test case: *supported × weak secondary → LOW; contradicted × rigorous primary → HIGH*. |
| **"a low-cost model does the first pass, a mid-tier model validates it, a strong model for the hard cases"** | ❌ | One judgement model (`CITATION_JUDGEMENT_MODEL`, else the chat model); the escalation adds *evidence*, not a stronger model. Since `b6ebf7b` everything runs on `gemini-3.5-flash-lite`. No validation pass, no routing of hard cases. |
| **"a human expert kept in the loop throughout"** | ❌ | No review queue, no override, no record of an expert's decision, nothing shown as "expert-confirmed". The planned `scripts/judge_label.py` (📋 judge-evaluation) is an offline labelling tool for the eval set, not a product loop. |

## 3. Slide 10 — "Where we go from here"

| | claimed | actual |
|---|---|---|
| **Now** — single-paper verification | ✅ | Tab 1 seed → references → fetch → index → audit |
| **Now** — working reliability judgement | ❌ | relation judgement only (§2) |
| **Now** — human expert in the loop | ❌ | none (§2) |
| **Next** — refine judgement accuracy | 📋 | judge-hardening built (`e437ef4…9a26ae4`); the evaluation set that would *measure* accuracy is written, not built |
| **Next** — corpus depth beyond one reference level | ❌ | no second hop; not planned |

## 4. Slide 11 — the vision loop (fair to be unbuilt; what is nearest)

| stage | nearest thing in the code |
|---|---|
| Gather the literature, incl. data from graphs and tables | ⚠️ v2 figure/table *captions* always; model descriptions of figures opt-in per run; table cells are an accepted loss (ingestion-v2 handoff) |
| Interpret conditions — reconcile results across process conditions | ❌ nothing reconciles; the brainstorm board's *established / objections* items "with system, conditions, numbers" (📋) is the seed of it |
| Consult the expert — ambiguous cases go to Gita | ❌ (§2) |
| Recommend the next experiment — Bayesian optimisation | ❌; the brainstorm *Test it* move (📋) proposes one experiment qualitatively |
| Run the real experiment / update the understanding | ❌ no way to add a lab result as evidence |
| Traceable throughout | ✅ keyed sources, verbatim spans, reports; 📋 narrated runs make the *process* traceable too |
| Compounds over time — institutional memory | ⚠️ brainstorm sessions persist (📋); expert decisions, audits and verdicts are not reused across runs |

## 5. What is true as pitched

Free and open source; deployed on Google Cloud; GROBID extraction and
embedding end to end; a real corpus; a working relation judgement with an
explicit rubric and downloadable reports; the audit UI on slide 5 is the
actual application.

## 6. Closing the distance — ordered by distance closed per unit of work

1. **Reliability = Source Assessor + Policy** (slide 6, steps 2–3). The largest gap between the slides and the code, and the one the pitch's title rests on. Metadata grade from Crossref/OpenAlex (type, citations, retracted) + a one-call method-documentation grade per source, cached per document; a pure policy function with the slide's two rules as its first tests; two columns in the table (source grade, reliability) and a sixth tile. *Moderate.*
2. **The expert in the loop.** A review queue in the audit: every *Unclear*, *Low* confidence, rubric-mismatch or *Contradicts* verdict offers *Confirm / Overrule (verdict + note)*; decisions stored in `data/expert_decisions.jsonl` keyed by (claim, source), shown as *expert-confirmed*, and applied automatically when the same pair recurs. That is also the beginning of "compounds over time". *Moderate.*
3. **The paywall.** Abstract-level evidence when the PDF is closed: OpenAlex/S2 abstracts and S2 citation contexts as the evidence, the verdict capped at *abstract only* (reliability ≤ MODERATE, never HIGH); plus a guided *upload what you have access to → re-audit* path. This is what turns 78 of 90 "unchecked" into something. *Moderate; mostly Agent 2 and the audit.*
4. **Tiered judgement.** With Gemini in place it is configuration plus one routing rule: first pass `flash-lite`; re-judge with a mid-tier model when the first pass is *Unclear*, *Low* confidence or a rubric mismatch; the strong model only for expert-queue candidates. *Small.*
5. **Uncited claims in the uploaded paper** — run Agent 5's citation-need check over the seed's sentences and list "citation needed" claims beside the audit. *Small.*
6. **Measure the judge** — build the judge-evaluation plan. Without it "refine judgement accuracy" has no baseline. *Written; build it.*
7. **A second hop** (bounded: the references of the papers the audit marked *Need review*). *Large; later.*

Items 1–2 are what the pitch says exists *now*; 3–4 are what makes the demo's numbers move.
