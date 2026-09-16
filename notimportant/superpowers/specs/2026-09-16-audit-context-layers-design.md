# Audit context layers — what the citing paper points at, what the cited paper is about, and which of it the judge may see

Revision of the Antigravity plan "3-Layer Hierarchical Context Architecture for Citation Audit"
(`~/.gemini/antigravity/brain/ed17e380-7347-4140-a9fe-6a10ec9896b9/implementation_plan.md`),
rewritten against the code at `f0a40f8`.

## 1. Purpose

A citation claim in the seed paper is judged against passages retrieved from the cited paper.
Two things about that claim are known to the TEI but not to the pipeline:

- **what its words point at** — "the discrepancy in Fig. 3b", "this approach", "Table 2" — which the
  retrieval query has to spell out to find the right passage;
- **where it sits** — the section heading it falls under, which is topical in physics papers
  ("DFT-based tight-binding Hamiltonian") and therefore a retrieval keyword in its own right.

And one thing about the cited paper is already computed and unused: the ingest-time summary
(`sum::{doc}` in `SUMMARY_COLLECTION_NAME`, 120–180 words on problem / method / result), which says
what the cited paper calls its own system and method — the vocabulary the query should be in.

The original plan wanted all of this, plus the cited paper's summary, injected into the **judge's**
prompt, and a rubric rule for citations that borrow a method. This spec keeps the data and the
retrieval use, moves the judge-side use behind an evaluation gate, and reframes the rubric rule.

## 2. Findings that shape the design

Each of these was checked against the repository, not assumed.

1. **Two TEI parsers.** `shared/extract.py:parse_tei` (lxml) feeds the corpus chunker and already
   collects, per figure, the body sentences that cite it (`Figure.refs`, lines 207–216). The seed
   audit never calls it: `seed_audit.extract_seed_citation_claims` (line 210) is an independent
   BeautifulSoup walk. Adding fields to `extract.py` does nothing for the audit.
2. **Method citations are not judged.** `claim_text.classify_citation_role` returns `"method"` for
   the eight words before a marker matching *following / according to / as in / adapted from / based
   on / we use|apply|…* (`_METHOD_RE`, line 155), and `"method"` is in `SKIP_ROLES` (line 31), so the
   audit files those as `not_a_claim` before any judge call (`seed_audit.py:1040`). The original
   plan's worked example G ("Following Settnes (2015), we apply…") is such a sentence.
3. **The judge already over-reads context.** Commit `e286762`: on a live run the model decomposed the
   *context* sentence instead of the marked one, with a three-sentence window. `assertion_drift`
   and the confidence cap in `judge.enforce_rubric` exist because of it. More prose in the block
   raises that rate; prose *about the cited paper* (its summary) is the worst case, because it reads
   as evidence and the grounding rule forbids using it as such.
4. **GROBID's structure is flat and noisy.** In `doi_10.1002_adma.202211157.grobid.tei.xml`, "2." and
   "2.1." are sibling top-level `<div>`s; a page header appears as `<head n="2211157">(3 of 11)</head>`
   and two of eleven `<figure>` nodes are page-header fragments with no label; 4 of 25 figure `<ref>`s
   have no `target`. `extract.py` already has `is_running_header` for the first problem;
   `seed_audit._paragraph_section` does not use it.
5. **The eval harness cannot see context.** `scripts/evaluate_judge.py` scores `judge(claim,
   evidence)`; the 19 cases in `judgement/cases/*.jsonl` carry no `context`. A change whose only
   effect is on what the judge reads has no measurement today. The file the original plan named
   for live verification (`doi_10.1093_gji_ggag077`) is not in `data/raw/grobid_output/` (331 files).
6. **The summary lookup is sound.** `ingestion.upsert_summaries` writes id `sum::{doc}` with metadata
   `{"document": doc}`; the audit sets `item["document"]` to the same basename
   (`seed_audit.py:1108`); `retrieve.rank_documents` already opens the collection.

## 3. Design

### 3.1 Three phases, each shippable on its own

| Phase | Adds | Reaches the judge? | Measured by |
|---|---|---|---|
| 1 Retrieval-side context | section breadcrumb, figure/table captions, cited-paper summary → claim fields and the **query-contextualization** prompt; shown in report and UI | No | before/after audit of the same seed: `no_evidence` count, `evidence_sufficiency` mix, queries by eye |
| 2 Judge-side context | `compose_context()`: `[Section: …]` breadcrumb + one caption line per referenced artifact + the window; eval set gains `context`; harness gains `--context none/window/full` and a drift signal | Yes — citing-paper facts only | `evaluate_judge --compare` window vs full on labelled context-bearing cases; drift rate not up |
| 3 Method-transfer scope rule | prompt.md V1.5 rule + example G; `tags` on cases | Yes | `evaluate_judge --compare` on `method_transfer`-tagged cases |

Phase 2 depends on Phase 1's fields. Phase 3 depends on Phase 2's `tags` and `--context`.

### 3.2 Where each fact is allowed

| Fact | Query prompt | Judge context | Report / UI |
|---|---|---|---|
| Section breadcrumb (citing paper) | yes | yes (Phase 2) | yes |
| Figure/table caption (citing paper) | yes, full caption ≤300 chars | yes (Phase 2), one line ≤200 chars, label + caption | yes |
| Cited paper summary | yes | **never** | yes, as "cited paper, in brief" |
| Seed title / abstract / "thesis" | no | no | — (title already shown) |

The cited summary is kept out of the judge because it is model prose about the paper the evidence
comes from: `span_is_verbatim` would catch a span quoted from it, but a slot verdict or `reason`
leaning on it is invisible.

### 3.3 New module `research_assistant/shared/tei_structure.py`

Pure BeautifulSoup functions over the citing paper's TEI; no model calls; hygiene first.

- `section_breadcrumb(p) -> str` — "2. Results and Discussion > 2.1. Terahertz Spectral Analysis".
  Parent = nearest earlier sibling `<div>` whose `<head n=…>` is shallower (fewer dotted components);
  climbing stops at a depth-1 heading; unnumbered headings count as depth 0 and are only ever the
  paragraph's own crumb; headless and running-header divs (`extract.is_running_header`) inherit the
  previous heading; a paragraph before any heading gets `""`.
- `artifact_registry(soup) -> {"by_id": {...}, "by_number": {(kind, int): ...}}` — a `<figure>` counts
  only if it has a numeric `<label>` or a `<head>` reading "Figure N"/"Table N"; caption =
  `figDesc` text ≤ 300 chars; label rendered "Fig. N" / "Table N".
- `paragraph_artifacts(p, registry) -> list[dict]` — targeted `<ref type="figure"|"table">` first,
  then the ref's own number, then "Fig. N"/"Table N" mentions in the paragraph text; each artifact once.

Artifact dict: `{"id": xml_id, "kind": "figure"|"table", "label": "Fig. 1", "caption": "..."}`.

### 3.4 Claim fields (added by `extract_seed_citation_claims`, carried through the report)

- `section_heading: str` — the breadcrumb, `""` when unknown (`section` kind is unchanged)
- `artifacts: list[dict]` — per paragraph, possibly empty
- `cited_summary: str` — set later by `attach_cited_summaries(claims)`, only on claims whose
  `document` has a summary; one Chroma `get` for the whole batch (`retrieve.document_summaries`)

### 3.5 Judge context (Phase 2)

`judge.compose_context(window, section=None, artifacts=None) -> str | None`:

```
[Section: 2. Results and Discussion > 2.1. Terahertz Spectral Analysis]
[Fig. 1: THz spectral analysis on MoS2 films and covalent networks. (a) …]
Before. «The discrepancy in Fig. 1(b) indicates … [14].» After.
```

`None` when there is no window. `_CONTEXT_HEADER` names the three parts and keeps "never as evidence".
`seed_audit._judge_claim_entry` calls it; `scripts/evaluate_judge.py` calls it for `--context full`.

### 3.6 Eval set (Phase 2)

Optional case fields `context`, `section_heading`, `artifacts`, `tags` (list of strings, default
`[]`). Harvest from audit reports carries the first three; the labeller shows the context and asks
for tags. Metrics gain `signals.drift` (rate of verdicts with an "assertion drift" violation) and
`by_tag`. Harness gains `--context {none,window,full}` (default `full`; verifier mode ignores it).

### 3.7 Method-transfer rule (Phase 3)

**Assumption, stated:** `method` stays in `SKIP_ROLES`. The rule therefore targets *evidential*
sentences that attribute a method to the cited paper and apply it to the citing paper's own system
without a `_METHOD_RE` cue — "The electrodes were made by the in situ oxidative polymerisation route
of [12], here with NiFe₂O₄ in place of MnFe₂O₄." Whether method citations should be judged at all is a
separate decision (budget, prioritisation) and is not made here.

Rule, in Step 2 under `scope`: when the claim borrows a method/formulation/model from the cited paper
and applies it to the citing paper's own system, `scope` is the system the cited paper developed the
method for, not the citing paper's application; the exception ends where the claim asserts a *result*
in the new system. Example G on the same MnFe₂O₄@PANI paper as A–F. Version V1.5 in `prompt.md`,
`judgement/__init__.py`, `app.py:1497`, `HOW_TO_USE.md`.

## 4. Not in scope

- Unifying the audit onto `extract.parse_tei` (worth doing; a refactor with its own plan).
- Taking `method` out of `SKIP_ROLES`.
- A seed "thesis" from the introduction's closing paragraph; a "discourse role" layer (the role
  classifier already exists).
- Vision on figure crops.

## 5. Placement

- `research_assistant/shared/tei_structure.py` (new), `tests/test_tei_structure.py` (new)
- `research_assistant/shared/retrieve.py` — `document_summaries`
- `research_assistant/shared/seed_audit.py` — claim fields, `attach_cited_summaries`, query prompt,
  judge context wiring, report line
- `app.py` — two captions in `_render_claim_item`
- `research_assistant/judgement/{judge,evalset,harvest,labeling,metrics}.py`, `prompt.md`, `cases/cases.jsonl`
- `scripts/{evaluate_judge,judge_label}.py`
