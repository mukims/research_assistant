# Audit Context Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The seed-paper citation audit resolves what each claim points at (section, figures, tables) and what the cited paper is about, uses that to write better retrieval queries, and — only once the evaluation set can measure it — lets the judge see the citing-paper parts of it; a rubric rule stops method-borrowing claims from failing `scope` for the citing paper's own application.

**Architecture:** A new pure module `shared/tei_structure.py` reads section breadcrumbs and figure/table artifacts out of the citing paper's TEI (BeautifulSoup, no model calls, page-header hygiene). `seed_audit.extract_seed_citation_claims` attaches them to every claim; `retrieve.document_summaries` fetches the ingest-time summaries of the cited papers in one store read; `contextualize_citation_queries` feeds all three to the query prompt. Phase 2 adds `judge.compose_context` (breadcrumb + caption lines + window; never the cited summary), context-bearing eval cases, and a `--context` switch on the harness so window-vs-full can be compared. Phase 3 adds the method-transfer `scope` rule and example G to the rubric, measured on tagged cases.

**Tech Stack:** Python 3.12, `unittest.TestCase` collected by pytest, BeautifulSoup (`"xml"` parser, lxml), ChromaDB (stubbed in tests via `sys.modules`), existing `shared.llm.chat`, `judgement.judge`, Streamlit `app.py`.

**Spec:** `notimportant/superpowers/specs/2026-09-16-audit-context-layers-design.md` — §2 is the evidence, §3 the design, §3.2 the table of what may reach the judge.

## Global Constraints

- Tests: `python3 -m pytest tests/<file>.py -q` from the repo root (base env; `chromadb` is not installed there and tests stub it via `patch.dict(sys.modules, …)` as in `tests/test_db.py`). Full suite: `python3 -m pytest tests/ -q` — 819 tests at `f0a40f8`, all green, and must stay green after every task.
- Live runs (audit, harness): prefix every command with `CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=.` and use `/home/shardul/miniconda3/envs/ml/bin/python` (the env `.claude/launch.json` runs the app with; it has chromadb 1.4.1 and the Gemini key from `.env`). Call it `$PY` below: `PY=/home/shardul/miniconda3/envs/ml/bin/python`.
- No new model calls in Phase 1. The only model calls in the audit remain the existing per-paragraph query contextualization and the per-claim judge.
- The cited paper's summary (`cited_summary`) never enters the judge prompt — spec §3.2. A test pins this (Task 13).
- `prompt.md` examples A–F are unchanged throughout; the rubric version stays `V1.4` until Task 15 bumps it to `V1.5` in all four places: `research_assistant/judgement/prompt.md:1`, `research_assistant/judgement/__init__.py:1`, `app.py:1497`, `HOW_TO_USE.md` (three mentions).
- Data under `data/` is git-ignored; eval artefacts go to `data/eval/audit/` (new) and `data/eval/judge/results/` (existing). Labelled cases go to `research_assistant/judgement/cases/human.jsonl` (tracked).
- Commit after each task. Subject in the repo's style (`feat(audit): …`, `feat(judge): …`, `test(…): …`), body says why, and the message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Artifact dict shape, used by every task that touches artifacts: `{"id": str, "kind": "figure"|"table", "label": "Fig. N"|"Table N", "caption": str}`.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `research_assistant/shared/extract.py` | `is_running_header` learns the `DOI: …` page header | 1 |
| `research_assistant/shared/tei_structure.py` (new) | `section_breadcrumb`, `artifact_registry`, `paragraph_artifacts` — the citing paper's TEI structure, pure functions | 1, 2 |
| `tests/test_tei_structure.py` (new) | tests for the above | 1, 2 |
| `research_assistant/shared/seed_audit.py` | claim fields `section_heading` / `artifacts`; `attach_cited_summaries`; query prompt; judge context wiring; report line | 3, 5, 6, 7, 13 |
| `tests/test_seed_audit.py` | tests for the above | 3, 5, 6, 7, 13 |
| `research_assistant/shared/retrieve.py` | `document_summaries(documents)` — one Chroma `get` | 4 |
| `tests/test_retrieve.py` | tests for the above | 4 |
| `app.py` | two captions in `_render_claim_item` | 7 |
| `research_assistant/judgement/judge.py` | `compose_context`, `ARTIFACT_CAPTION_CHARS`, `_CONTEXT_HEADER` | 9 |
| `research_assistant/judgement/evalset.py` | optional `context`, `section_heading`, `artifacts`, `tags` | 10 |
| `research_assistant/judgement/harvest.py`, `labeling.py` | carry those fields; tags | 10 |
| `scripts/judge_label.py` | show context, ask tags | 10 |
| `scripts/evaluate_judge.py` | `--context none|window|full` | 11 |
| `research_assistant/judgement/metrics.py` | `signals.drift`, `by_tag` | 11 |
| `research_assistant/judgement/prompt.md`, `cases/cases.jsonl` | V1.5 scope rule, example G | 15 |
| `tests/test_judgement.py`, `test_evalset.py`, `test_harvest.py`, `test_labeling.py`, `test_evaluate_judge.py`, `test_metrics.py`, `test_extract.py` | tests | 1, 9–11, 15 |

---

## Phase 0 — Baseline

### Task 0: Record the baseline audit

**Files:**
- Create (git-ignored): `data/eval/audit/A.json`

**Interfaces:**
- Produces: `data/eval/audit/A.json`, a full audit report of `data/raw/processed/arxiv_2007.12504v1.pdf` at `f0a40f8` (≈20 judged claims; 8 of its cited paragraphs also reference figures). Task 8 compares against it.

- [ ] **Step 1: Confirm the seed and its TEI exist**

Run:
```bash
ls -la data/raw/processed/arxiv_2007.12504v1.pdf data/raw/grobid_output/arxiv_2007.12504v1.grobid.tei.xml
```
Expected: both files listed.

- [ ] **Step 2: Run the audit from scratch and keep the report**

Run:
```bash
mkdir -p data/eval/audit && CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python - <<'EOF'
import json
from research_assistant.shared.seed_audit import audit_seed_citations
r = audit_seed_citations("data/raw/processed/arxiv_2007.12504v1.pdf", force=True, skip_if_cached=False)
json.dump(r, open("data/eval/audit/A.json", "w"), indent=1)
print(r["totals"])
EOF
```
Expected: a totals dict with `judged` ≈ 19–20 and no `error` key; the run takes a few minutes (≈20 judge calls plus ≈10 contextualization calls on `gemini-3.5-flash-lite`).

- [ ] **Step 3: Note the baseline numbers**

Run:
```bash
python3 - <<'EOF'
import json
from collections import Counter
r = json.load(open("data/eval/audit/A.json"))
js = [i for i in r["results"] if i.get("outcome") == "judged"]
print("judged", len(js), "no_evidence", sum(1 for i in r["results"] if i.get("outcome") == "no_evidence"))
print("sufficiency", dict(Counter(i.get("evidence_sufficiency") for i in js)))
print("judgements", dict(Counter(i.get("judgement") for i in js)))
EOF
```
Expected: three lines; write them into the commit message of Task 8 as "baseline".

No commit — nothing tracked changed.

---

## Phase 1 — Retrieval-side context

### Task 1: `section_breadcrumb` — where a paragraph sits

**Files:**
- Modify: `research_assistant/shared/extract.py:87-93` (`_RUNNING`)
- Create: `research_assistant/shared/tei_structure.py`
- Test: `tests/test_extract.py:97-101`, `tests/test_tei_structure.py` (new)

**Interfaces:**
- Consumes: `research_assistant.shared.claim_text.clean_text(node) -> str`; `research_assistant.shared.extract.is_running_header(heading: str) -> bool`.
- Produces: `tei_structure.section_breadcrumb(p) -> str` where `p` is a bs4 `<p>` Tag. Returns `"2. Results and Discussion > 2.1. Terahertz Spectral Analysis"`, or the heading alone at the top, or `""` for a paragraph before any heading.

- [ ] **Step 1: Write the failing test for the DOI page header**

In `tests/test_extract.py`, change the true list in `test_running_header_detection` (line 98):

```python
        for h in ["(3 of 11)", "Odashima et al.", "12", "Adv. Mater. 2023, 35, 2211157", "DOI: 10.1002/adma.202211157"]:
```

- [ ] **Step 2: Run it to see it fail**

Run: `python3 -m pytest tests/test_extract.py::TestSectionKinds::test_running_header_detection -q`
Expected: FAIL — `AssertionError: False is not true : DOI: 10.1002/adma.202211157`

- [ ] **Step 3: Add the pattern**

In `research_assistant/shared/extract.py`, `_RUNNING` (line 87) becomes:

```python
_RUNNING = [
    re.compile(r"^\(\d+ of \d+\)$"),
    re.compile(r"\bet al\.?$", re.I),
    re.compile(r"^\d+$"),
    re.compile(r"\b(19|20)\d{2}\b.*\b\d{3,}\b"),      # "Adv. Mater. 2023, 35, 2211157"
    re.compile(r"^\s*doi\s*:", re.I),                 # "DOI: 10.1002/adma.202211157" — Wiley's first-page header
]
```

- [ ] **Step 4: Run it to see it pass**

Run: `python3 -m pytest tests/test_extract.py -q`
Expected: all pass.

- [ ] **Step 5: Write the failing breadcrumb tests**

Create `tests/test_tei_structure.py`:

```python
"""Where a paragraph sits in the citing paper's TEI and what it points at.

GROBID's sections are flat siblings ("2." and "2.1." side by side) and its
figure list carries page-header fragments; these pin the walks that make a
breadcrumb and an artifact list out of that.
"""

import unittest

from bs4 import BeautifulSoup

from research_assistant.shared import tei_structure as ts

TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><p xml:id="p0">A paragraph before any heading.</p></div>
<div><head n="1.">Introduction</head><p xml:id="p1">Intro.</p></div>
<div><head n="2.">Results and Discussion</head><p xml:id="p2">Top of results.</p></div>
<div><head n="2.1.">Terahertz Spectral Analysis</head>
  <p xml:id="p3">See <ref type="figure" target="#fig_0">1b</ref> and Table <ref type="table">2</ref> and Fig. 3 and Fig. 1 again.</p>
</div>
<div><head n="2211157">(3 of 11)</head><p xml:id="p4">Under a page header.</p></div>
<div><head n="2.2.">Charge Transport</head><p xml:id="p5">More.</p></div>
<div><p xml:id="p6">A headless continuation.</p></div>
<div><head>Conclusions</head><p xml:id="p7">Done.</p></div>
<figure xml:id="fig_0"><head>Figure 1 .</head><label>1</label><figDesc>Figure 1. THz spectra of films. """ + "x" * 400 + """</figDesc></figure>
<figure xml:id="fig_1"><figDesc>Adv. Mater. 2023, 2211157</figDesc></figure>
<figure xml:id="fig_2"><head>Figure 4 .</head><figDesc>Figure 4. Headed but unlabelled.</figDesc></figure>
<figure type="table" xml:id="tab_0"><head>Table 2 .</head><label>2</label><figDesc>Fit parameters.</figDesc></figure>
</body></text></TEI>"""


def _soup():
    return BeautifulSoup(TEI, "xml")


def _p(soup, xml_id):
    return soup.find("p", attrs={"xml:id": xml_id})


class TestSectionBreadcrumb(unittest.TestCase):
    def test_subsection_gets_its_parent(self):
        self.assertEqual(ts.section_breadcrumb(_p(_soup(), "p3")),
                         "2. Results and Discussion > 2.1. Terahertz Spectral Analysis")

    def test_top_level_is_alone(self):
        s = _soup()
        self.assertEqual(ts.section_breadcrumb(_p(s, "p1")), "1. Introduction")
        self.assertEqual(ts.section_breadcrumb(_p(s, "p2")), "2. Results and Discussion")

    def test_unnumbered_heading_is_alone(self):
        self.assertEqual(ts.section_breadcrumb(_p(_soup(), "p7")), "Conclusions")

    def test_before_any_heading_is_empty(self):
        self.assertEqual(ts.section_breadcrumb(_p(_soup(), "p0")), "")

    def test_page_header_and_headless_divs_inherit_the_previous_heading(self):
        s = _soup()
        self.assertEqual(ts.section_breadcrumb(_p(s, "p4")),
                         "2. Results and Discussion > 2.1. Terahertz Spectral Analysis")
        self.assertEqual(ts.section_breadcrumb(_p(s, "p6")),
                         "2. Results and Discussion > 2.2. Charge Transport")

    def test_a_later_sibling_skips_the_page_header(self):
        self.assertEqual(ts.section_breadcrumb(_p(_soup(), "p5")),
                         "2. Results and Discussion > 2.2. Charge Transport")
```

- [ ] **Step 6: Run to see it fail**

Run: `python3 -m pytest tests/test_tei_structure.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'research_assistant.shared.tei_structure'`

- [ ] **Step 7: Create the module with `section_breadcrumb`**

Create `research_assistant/shared/tei_structure.py`:

```python
# research_assistant/shared/tei_structure.py
"""Where a paragraph sits in the citing paper and what it points at.

Two facts GROBID's TEI states but does not hand over cleanly. Sections are
flat: "2." and "2.1." are sibling <div>s, so a heading's parent is the
nearest earlier heading with a shallower number. Figures and tables are
<figure> nodes a paragraph reaches through <ref type="figure"
target="#fig_0"> — when GROBID linked the reference; about one in six it
leaves untargeted, and "Fig. 3" in the text is the only witness.

Both walks are deterministic and cost no model call. Both are hygiene
first: GROBID emits page headers as <head> and as <figure>, and neither
may become a breadcrumb or an artifact.
"""

from __future__ import annotations

import re

from research_assistant.shared.claim_text import clean_text
from research_assistant.shared.extract import is_running_header

_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*$")


def _depth(n) -> int:
    """'2.1.' → 2, '2.' → 1; unnumbered or roman → 0."""
    n = (n or "").strip().rstrip(".")
    return n.count(".") + 1 if n and _NUMBERED_RE.match(n) else 0


def _label(head) -> str:
    return f"{(head.get('n') or '').strip()} {clean_text(head)}".strip()


def section_breadcrumb(p) -> str:
    """'2. Results and Discussion > 2.1. Terahertz Spectral Analysis' for a
    paragraph in the 2.1 div; the heading alone at the top; '' before any
    heading. The parent of a heading is the nearest earlier sibling with a
    shallower number, and the climb stops at depth 1 — an unnumbered
    heading ("Conclusions", "I. INVERSION PROCEDURE") is only ever a
    paragraph's own crumb, never a parent, because GROBID also files the
    journal's page header as an unnumbered <head>. A headless or
    page-header div inherits the heading before it."""
    div = p.find_parent("div")
    if div is None:
        return ""
    crumbs: list[str] = []
    depth = None
    for cand in [div, *div.find_previous_siblings("div")]:
        head = cand.find("head")
        if head is None or is_running_header(clean_text(head)):
            continue
        d = _depth(head.get("n"))
        if depth is None or d < depth:
            crumbs.append(_label(head))
            depth = d
        if depth <= 1:
            break
    return " > ".join(reversed(crumbs))
```

- [ ] **Step 8: Run to see it pass**

Run: `python3 -m pytest tests/test_tei_structure.py tests/test_extract.py -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add research_assistant/shared/extract.py research_assistant/shared/tei_structure.py tests/test_extract.py tests/test_tei_structure.py
git commit -m "feat(audit): a paragraph knows its section breadcrumb

GROBID's <div>s are flat — '2.' and '2.1.' are siblings — so the parent of
a heading is the nearest earlier one with a shallower number. Page headers
('(3 of 11)', 'DOI: 10.1002/…') are skipped and their paragraphs inherit
the heading before them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `artifact_registry` and `paragraph_artifacts` — what a paragraph points at

**Files:**
- Modify: `research_assistant/shared/tei_structure.py`
- Test: `tests/test_tei_structure.py`

**Interfaces:**
- Consumes: the `TEI` fixture and helpers from Task 1's test file.
- Produces: `tei_structure.artifact_registry(soup) -> dict` with keys `"by_id": {xml_id: artifact}` and `"by_number": {(kind, int): artifact}`; `tei_structure.paragraph_artifacts(p, registry) -> list[dict]`; `tei_structure.CAPTION_MAX_CHARS = 300`. Artifact dict per Global Constraints.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tei_structure.py`:

```python
class TestArtifactRegistry(unittest.TestCase):
    def test_labelled_and_headed_figures_are_registered_and_fragments_are_not(self):
        reg = ts.artifact_registry(_soup())
        self.assertEqual(sorted(reg["by_id"]), ["fig_0", "fig_2", "tab_0"])
        self.assertEqual(sorted(reg["by_number"]), [("figure", 1), ("figure", 4), ("table", 2)])
        self.assertEqual(reg["by_id"]["fig_0"]["label"], "Fig. 1")
        self.assertEqual(reg["by_id"]["tab_0"], {"id": "tab_0", "kind": "table", "label": "Table 2", "caption": "Fit parameters."})

    def test_caption_is_capped(self):
        reg = ts.artifact_registry(_soup())
        self.assertEqual(len(reg["by_id"]["fig_0"]["caption"]), ts.CAPTION_MAX_CHARS)
        self.assertTrue(reg["by_id"]["fig_0"]["caption"].startswith("Figure 1. THz spectra of films."))

    def test_no_body_is_empty(self):
        reg = ts.artifact_registry(BeautifulSoup("<TEI/>", "xml"))
        self.assertEqual(reg, {"by_id": {}, "by_number": {}})


class TestParagraphArtifacts(unittest.TestCase):
    def test_targeted_ref_untargeted_ref_and_text_mention_each_once(self):
        s = _soup()
        arts = ts.paragraph_artifacts(_p(s, "p3"), ts.artifact_registry(s))
        # fig_0 via target; tab_0 via the untargeted ref's own number; "Fig. 3" has no figure;
        # "Fig. 1 again" is the same artifact and is not repeated.
        self.assertEqual([a["label"] for a in arts], ["Fig. 1", "Table 2"])
        self.assertEqual(arts[0]["id"], "fig_0")

    def test_paragraph_without_references_is_empty(self):
        s = _soup()
        self.assertEqual(ts.paragraph_artifacts(_p(s, "p1"), ts.artifact_registry(s)), [])
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_tei_structure.py -q`
Expected: FAIL — `AttributeError: module … has no attribute 'artifact_registry'`

- [ ] **Step 3: Implement**

Append to `research_assistant/shared/tei_structure.py`:

```python


# ─── Figures and tables ──────────────────────────────────────────────────────

CAPTION_MAX_CHARS = 300

_HEAD_NUMBER_RE = re.compile(r"^(?:Fig(?:ure)?|Table)\.?\s*(\d+)", re.I)
_MENTION_RE = re.compile(r"\b(Fig(?:ure|s)?|Tables?)\.?\s*(\d+)", re.I)


def artifact_registry(soup) -> dict:
    """Every figure and table the paper numbers, by xml:id and by (kind,
    number). A <figure> counts only when it carries a number — a <label>,
    or a <head> that reads 'Figure 3' / 'Table 2'; GROBID's page-header
    and stray-caption fragments have neither. When two <figure>s claim the
    same number (a split multi-panel figure) the first keeps the number."""
    by_id: dict[str, dict] = {}
    by_number: dict[tuple, dict] = {}
    body = soup.find("body")
    for i, fig in enumerate(body.find_all("figure") if body else []):
        kind = "table" if fig.get("type") == "table" else "figure"
        m = re.match(r"(\d+)", clean_text(fig.find("label"))) or _HEAD_NUMBER_RE.match(clean_text(fig.find("head")))
        if not m:
            continue
        number = int(m.group(1))
        art = {
            "id": fig.get("xml:id") or f"{kind}_{i}",
            "kind": kind,
            "label": f"{'Table' if kind == 'table' else 'Fig.'} {number}",
            "caption": clean_text(fig.find("figDesc"))[:CAPTION_MAX_CHARS],
        }
        by_id[art["id"]] = art
        by_number.setdefault((kind, number), art)
    return {"by_id": by_id, "by_number": by_number}


def paragraph_artifacts(p, registry: dict) -> list[dict]:
    """The figures and tables a paragraph points at, each once, keyed by
    label so a split figure reached two ways is still one entry. Linked
    <ref>s first (GROBID's own resolution), then an untargeted <ref>'s own
    number, then 'Fig. 3' / 'Table 2' mentions in the text."""
    found: dict[str, dict] = {}
    for ref in p.find_all("ref", type=["figure", "table"]):
        art = registry["by_id"].get((ref.get("target") or "").lstrip("#"))
        if art is None:
            kind = "table" if ref.get("type") == "table" else "figure"
            m = re.match(r"(\d+)", clean_text(ref))
            art = registry["by_number"].get((kind, int(m.group(1)))) if m else None
        if art is not None:
            found.setdefault(art["label"], art)
    for m in _MENTION_RE.finditer(clean_text(p)):
        kind = "table" if m.group(1).lower().startswith("tab") else "figure"
        art = registry["by_number"].get((kind, int(m.group(2))))
        if art is not None:
            found.setdefault(art["label"], art)
    return list(found.values())
```

- [ ] **Step 4: Run to see them pass**

Run: `python3 -m pytest tests/test_tei_structure.py -q`
Expected: all pass (11 tests).

- [ ] **Step 5: Try it on a real file**

Run:
```bash
python3 - <<'EOF'
from bs4 import BeautifulSoup
from research_assistant.shared import tei_structure as ts
soup = BeautifulSoup(open("data/raw/grobid_output/doi_10.1002_adma.202211157.grobid.tei.xml", encoding="utf-8"), "xml")
reg = ts.artifact_registry(soup)
print("registered:", sorted(reg["by_number"]))
for p in soup.find("body").find_all("p"):
    arts = ts.paragraph_artifacts(p, reg)
    if arts and p.find_all("ref", type="bibr"):
        print(repr(ts.section_breadcrumb(p))[:60], [a["label"] for a in arts]); break
EOF
```
Expected: `registered:` lists `('figure', 1)…('figure', 5)` and no table (this paper's tables are images); one breadcrumb line like `'2. Results and Discussion > 2.1. Terahertz Spectral Analysi' ['Fig. 1']`.

- [ ] **Step 6: Commit**

```bash
git add research_assistant/shared/tei_structure.py tests/test_tei_structure.py
git commit -m "feat(audit): a paragraph knows which figures and tables it points at

Registered from <figure> nodes that carry a number — GROBID's page-header
and stray-caption fragments do not — and reached through the linked <ref>,
the untargeted <ref>'s own number, or a 'Fig. 3' in the text. Captions are
capped at 300 characters for the query prompt.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Every extracted claim carries `section_heading` and `artifacts`

**Files:**
- Modify: `research_assistant/shared/seed_audit.py:210-393` (`extract_seed_citation_claims`)
- Test: `tests/test_seed_audit.py`

**Interfaces:**
- Consumes: `tei_structure.artifact_registry(soup)`, `tei_structure.paragraph_artifacts(p, registry)`, `tei_structure.section_breadcrumb(p)`.
- Produces: on every claim dict from `extract_seed_citation_claims`: `"section_heading": str` (breadcrumb, `""` when unknown) and `"artifacts": list[dict]` (per paragraph, possibly `[]`). `section` (the kind) is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_seed_audit.py` (module level, before the last class is fine):

```python
STRUCTURED_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader><fileDesc><titleStmt><title>Structured Seed</title></titleStmt><sourceDesc><p></p></sourceDesc></fileDesc></teiHeader>
    <text>
        <body>
            <div><head n="2.">Results and Discussion</head><p xml:id="p_top">Top-level result text cites a paper <ref type="bibr" target="#b0">[1]</ref>.</p></div>
            <div><head n="2.1.">Terahertz Spectral Analysis</head>
                <p xml:id="p_fig">The photoconductivity enhancement in <ref type="figure" target="#fig_0">1b</ref> agrees with the covalent network model <ref type="bibr" target="#b0">[1]</ref>.</p>
            </div>
            <figure xml:id="fig_0"><head>Figure 1 .</head><label>1</label><figDesc>Figure 1. THz spectral analysis on MoS2 films and covalent networks.</figDesc></figure>
        </body>
        <back><div type="references"><listBibl>
            <biblStruct xml:id="b0"><analytic><title level="a" type="main">Covalent MoS2 networks</title>
                <author><persName><forename>A.</forename><surname>Gabbett</surname></persName></author>
                <idno type="DOI">10.1002/adma.202211157</idno></analytic>
                <monogr><title level="j">Adv. Mater.</title><imprint><date type="published" when="2023">2023</date></imprint></monogr></biblStruct>
        </listBibl></div></back>
    </text>
</TEI>"""


class TestClaimStructureFields(unittest.TestCase):
    def test_claims_carry_breadcrumb_and_artifacts(self):
        claims = extract_seed_citation_claims(BeautifulSoup(STRUCTURED_TEI_XML, "xml"))
        by_p = {c["paragraph_id"]: c for c in claims}
        self.assertEqual(by_p["p_top"]["section_heading"], "2. Results and Discussion")
        self.assertEqual(by_p["p_top"]["artifacts"], [])
        self.assertEqual(by_p["p_fig"]["section_heading"], "2. Results and Discussion > 2.1. Terahertz Spectral Analysis")
        self.assertEqual([a["label"] for a in by_p["p_fig"]["artifacts"]], ["Fig. 1"])
        self.assertTrue(by_p["p_fig"]["artifacts"][0]["caption"].startswith("Figure 1. THz spectral analysis"))
        # The kind is what it was: prioritisation and the report still read it.
        self.assertEqual(by_p["p_top"]["section"], "results")
```

- [ ] **Step 2: Run to see it fail**

Run: `python3 -m pytest tests/test_seed_audit.py::TestClaimStructureFields -q`
Expected: FAIL — `KeyError: 'section_heading'`

- [ ] **Step 3: Wire the fields in**

In `research_assistant/shared/seed_audit.py`:

(a) Add the import after `from research_assistant.shared.source_key import normalise_doi` (line 56):

```python
from research_assistant.shared.tei_structure import artifact_registry, paragraph_artifacts, section_breadcrumb
```

(b) In the `extract_seed_citation_claims` docstring, after the `paragraph_refs` line, add:

```
        - section_heading: breadcrumb of numbered headings ("2. Results > 2.1. THz analysis"), "" if unknown
        - artifacts: figures/tables the paragraph points at: [{id, kind, label, caption}]
```

(c) Right after `number_map = _build_number_map(body, bib_by_id)` (line 303), add:

```python
    registry = artifact_registry(soup)
```

(d) Right after `section = _paragraph_section(p)` (line 314), add:

```python
        heading = section_breadcrumb(p)
        p_artifacts = paragraph_artifacts(p, registry)
```

(e) In the `paragraph_claims.append({...})` dict (line 379), after `"section": section,` add:

```python
                    "section_heading": heading,
                    "artifacts": p_artifacts,
```

- [ ] **Step 4: Run to see it pass, then the whole file**

Run: `python3 -m pytest tests/test_seed_audit.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/seed_audit.py tests/test_seed_audit.py
git commit -m "feat(audit): every claim carries its section breadcrumb and the figures its paragraph points at

Deterministic, from the TEI, no model call. Nothing reads them yet; the
query prompt and the report do next.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `retrieve.document_summaries` — the cited papers' summaries in one read

**Files:**
- Modify: `research_assistant/shared/retrieve.py` (after `rank_documents`, line ~85)
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `SUMMARY_COLLECTION_NAME`, `VECTORDB_PATH` (already imported in `retrieve.py`); Chroma ids `sum::{doc}` with metadata `{"document": doc}` as written by `ingestion.upsert_summaries` (`ingestion.py:596-604`).
- Produces: `retrieve.document_summaries(documents: Iterable[str]) -> dict[str, str]` — `{document: summary}` for the documents that have one; `{}` when nothing is wanted or the collection does not exist.

- [ ] **Step 1: Write the failing tests**

In `tests/test_retrieve.py`, add `import sys` to the imports, then append:

```python
class TestDocumentSummaries(unittest.TestCase):
    """One store read for all the cited papers, keyed by document; the
    store is stubbed the way tests/test_db.py stubs it."""

    def _chroma(self, rows, missing_collection=False):
        calls = []

        class _Col:
            def get(self, ids, include):
                calls.append(sorted(ids))
                return {
                    "ids": [f"sum::{d}" for d, _ in rows],
                    "documents": [t for _, t in rows],
                    "metadatas": [{"document": d} for d, _ in rows],
                }

        class _Client:
            def __init__(self, path):
                pass

            def get_collection(self, name):
                if missing_collection:
                    raise ValueError("no such collection")
                return _Col()

        chroma = types.ModuleType("chromadb")
        chroma.PersistentClient = _Client
        return chroma, calls

    def test_one_get_keyed_by_document_and_missing_ones_absent(self):
        chroma, calls = self._chroma([("a.pdf", "A studies X by Y.")])
        with patch.dict(sys.modules, {"chromadb": chroma}):
            out = rt.document_summaries(["a.pdf", "b.pdf", "a.pdf", None, ""])
        self.assertEqual(out, {"a.pdf": "A studies X by Y."})
        self.assertEqual(calls, [["sum::a.pdf", "sum::b.pdf"]])

    def test_no_collection_is_empty(self):
        chroma, _ = self._chroma([], missing_collection=True)
        with patch.dict(sys.modules, {"chromadb": chroma}):
            self.assertEqual(rt.document_summaries(["a.pdf"]), {})

    def test_nothing_wanted_does_not_touch_the_store(self):
        chroma, calls = self._chroma([])
        with patch.dict(sys.modules, {"chromadb": chroma}):
            self.assertEqual(rt.document_summaries([]), {})
        self.assertEqual(calls, [])
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_retrieve.py::TestDocumentSummaries -q`
Expected: FAIL — `AttributeError: module … has no attribute 'document_summaries'`

- [ ] **Step 3: Implement**

In `research_assistant/shared/retrieve.py`, directly after `rank_documents` (before the `# ─── Stage 1` gate section that follows it):

```python
def document_summaries(documents) -> dict[str, str]:
    """The ingest-time summary of each named document, keyed by document.
    One store read for the lot; a document without a summary is absent
    from the result, and no summary collection at all is an empty dict."""
    import chromadb

    wanted = sorted({d for d in documents if d})
    if not wanted:
        return {}
    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    try:
        col = client.get_collection(SUMMARY_COLLECTION_NAME)
    except Exception:
        logger.warning("No summary collection yet — has anything been ingested?")
        return {}
    got = col.get(ids=[f"sum::{d}" for d in wanted], include=["documents", "metadatas"])
    out: dict[str, str] = {}
    for text, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
        doc = (meta or {}).get("document")
        if doc and text:
            out[doc] = text
    return out
```

- [ ] **Step 4: Run to see them pass**

Run: `python3 -m pytest tests/test_retrieve.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/retrieve.py tests/test_retrieve.py
git commit -m "feat(retrieve): document_summaries reads the cited papers' ingest-time summaries in one call

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `attach_cited_summaries` — the summary rides on every claim that cites the paper

**Files:**
- Modify: `research_assistant/shared/seed_audit.py` (new function before `contextualize_citation_queries`, line 463; one call in `audit_seed_citations`, line ~1155)
- Test: `tests/test_seed_audit.py`

**Interfaces:**
- Consumes: `retrieve.document_summaries(documents) -> dict[str, str]`; `claim["document"]` (set at `seed_audit.py:1108` for downloaded claims).
- Produces: `seed_audit.attach_cited_summaries(claims: list[dict]) -> None`; sets `claim["cited_summary"]: str` only where a summary exists. Called in `audit_seed_citations` immediately before `contextualize_citation_queries(claims_to_judge)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_seed_audit.py`:

```python
class TestCitedSummaries(unittest.TestCase):
    def test_summary_lands_on_every_claim_that_cites_the_document(self):
        from research_assistant.shared.seed_audit import attach_cited_summaries
        claims = [{"document": "a.pdf"}, {"document": "b.pdf"}, {"document": "a.pdf"}, {}]
        with patch("research_assistant.shared.retrieve.document_summaries", return_value={"a.pdf": "About A."}) as ds:
            attach_cited_summaries(claims)
        self.assertEqual(ds.call_args[0][0], {"a.pdf", "b.pdf"})
        self.assertEqual([c.get("cited_summary") for c in claims], ["About A.", None, "About A.", None])

    def test_store_failure_costs_only_the_summary(self):
        from research_assistant.shared.seed_audit import attach_cited_summaries
        claims = [{"document": "a.pdf"}]
        with patch("research_assistant.shared.retrieve.document_summaries", side_effect=RuntimeError("store down")):
            attach_cited_summaries(claims)
        self.assertNotIn("cited_summary", claims[0])

    def test_no_documents_no_store_read(self):
        from research_assistant.shared.seed_audit import attach_cited_summaries
        with patch("research_assistant.shared.retrieve.document_summaries") as ds:
            attach_cited_summaries([{"claim": "x"}])
        ds.assert_not_called()

    @patch("research_assistant.shared.seed_audit.contextualize_citation_queries")
    @patch("research_assistant.shared.seed_audit.attach_cited_summaries")
    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_summaries_are_attached_before_queries_are_written(
        self, mock_manifest, mock_find_tei, mock_search, mock_judge, mock_attach, mock_ctx
    ):
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(SAMPLE_TEI_XML); tei_file = tf.name
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            pdf = dummy_pdf.name
        mock_find_tei.return_value = tei_file
        mock_manifest.return_value = {"doi:10.1126/science.1102896": {
            "key": "doi:10.1126/science.1102896", "path": pdf, "doi": "10.1126/science.1102896",
            "title": "Electric field effect in atomically thin carbon films", "cited_by": "seed.pdf", "xml_id": "b0"}}
        mock_search.return_value = [{"text": "Evidence.", "metadata": {"document": os.path.basename(pdf)}}]
        mock_judge.return_value = {
            "judgement": "Supports", "confidence": "High", "supporting_span": "Evidence.", "reason": "r",
            "slots": {"finding": {"assertion": "a", "verdict": "Supports"}, "scope": {"assertion": "s", "verdict": "Supports"},
                      "strength": {"assertion": "t", "verdict": "Not applicable"}},
            "evidence_sufficiency": "sufficient",
        }
        order = []
        mock_attach.side_effect = lambda claims: order.append(("attach", len(claims)))
        mock_ctx.side_effect = lambda claims: order.append(("queries", len(claims)))
        with _audit_dirs():
            audit_seed_citations("seed.pdf", search_resources=(MagicMock(), MagicMock(), [], []), max_claims=5)
        self.assertEqual(order, [("attach", 1), ("queries", 1)])
        os.unlink(tei_file); os.unlink(pdf)
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_seed_audit.py::TestCitedSummaries -q`
Expected: FAIL — `ImportError: cannot import name 'attach_cited_summaries'`

- [ ] **Step 3: Implement and wire**

(a) In `research_assistant/shared/seed_audit.py`, immediately before `def contextualize_citation_queries` (line 463):

```python
def attach_cited_summaries(claims: list[dict]) -> None:
    """The ingest-time summary of each cited document, as 'cited_summary'
    on every claim that cites it. One store read for the whole batch; a
    claim whose document has no summary gets none, and a store that cannot
    be read costs nothing but the summary."""
    docs = {c.get("document") for c in claims if c.get("document")}
    if not docs:
        return
    from research_assistant.shared.retrieve import document_summaries

    try:
        summaries = document_summaries(docs)
    except Exception as exc:  # noqa: BLE001 — the query is better with it, fine without
        logger.warning("Could not read document summaries: %s", exc)
        return
    for c in claims:
        text = summaries.get(c.get("document"))
        if text:
            c["cited_summary"] = text


```

(b) In `audit_seed_citations`, the block

```python
    if claims_to_judge:
        pipeline_status.update_progress(detail="Contextualizing search queries from paragraph context")
        contextualize_citation_queries(claims_to_judge)
```

becomes

```python
    if claims_to_judge:
        pipeline_status.update_progress(detail="Contextualizing search queries from paragraph context")
        attach_cited_summaries(claims_to_judge)
        contextualize_citation_queries(claims_to_judge)
```

- [ ] **Step 4: Run the file**

Run: `python3 -m pytest tests/test_seed_audit.py -q`
Expected: all pass. (Existing end-to-end audit tests now log one "Could not read document summaries" warning each because the test env has no `chromadb`; that is the fallback working.)

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/seed_audit.py tests/test_seed_audit.py
git commit -m "feat(audit): each judged claim carries the cited paper's ingest-time summary

One store read before the queries are written; a store that cannot be read
costs nothing but the summary.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The query prompt reads the section, the captions and the cited summary

**Files:**
- Modify: `research_assistant/shared/seed_audit.py` (`contextualize_citation_queries`, the per-paragraph loop, lines ~496-527)
- Test: `tests/test_seed_audit.py` (class `TestQueryContextualizationAndCompoundCitations`)

**Interfaces:**
- Consumes: `claim["section_heading"]`, `claim["artifacts"]`, `claim["cited_summary"]`, `claim["document"]` from Tasks 3 and 5.
- Produces: the same `claim["search_query"]` contract; a larger prompt. Behaviour without the new fields is unchanged apart from rule wording.

- [ ] **Step 1: Write the failing tests**

Add to `TestQueryContextualizationAndCompoundCitations` in `tests/test_seed_audit.py`:

```python
    @patch("research_assistant.shared.llm.chat")
    def test_query_prompt_carries_section_captions_and_cited_summary(self, mock_chat):
        from research_assistant.shared.seed_audit import contextualize_citation_queries
        from research_assistant.shared.llm import ChatResult

        mock_chat.return_value = ChatResult(content='["q0", "q1"]')
        shared = {
            "context": "Before. «The enhancement in Fig. 1b agrees with the network model.» After.",
            "paragraph_id": "p_fig",
            "section_heading": "2. Results and Discussion > 2.1. Terahertz Spectral Analysis",
            "artifacts": [{"id": "fig_0", "kind": "figure", "label": "Fig. 1",
                           "caption": "THz photoconductivity of MoS2 films and networks."}],
            "ref": {"authors": ["Gabbett"], "year": 2023, "title": "Covalent MoS2 networks"},
            "document": "gabbett.pdf",
            "cited_summary": "Studies covalent MoS2 networks by terahertz spectroscopy.",
        }
        claims = [dict(shared, claim="The enhancement in Fig. 1b agrees with the network model."),
                  dict(shared, claim="The same paper is cited again in this paragraph.")]
        with patch("research_assistant.shared.seed_audit.CITATION_AUDIT_CONTEXTUALIZE_QUERIES", True):
            contextualize_citation_queries(claims)

        prompt = mock_chat.call_args[0][0][0]["content"]
        self.assertIn("Section of the citing paper: 2. Results and Discussion > 2.1. Terahertz Spectral Analysis", prompt)
        self.assertIn("- Fig. 1: THz photoconductivity of MoS2 films and networks.", prompt)
        # The same paper's summary is spelled out once and referred back to after that.
        self.assertEqual(prompt.count("Studies covalent MoS2 networks by terahertz spectroscopy."), 1)
        self.assertIn("What the cited paper is about: as for [0]", prompt)
        self.assertEqual([c["search_query"] for c in claims], ["q0", "q1"])

    @patch("research_assistant.shared.llm.chat")
    def test_query_prompt_without_structure_has_no_empty_headings(self, mock_chat):
        from research_assistant.shared.seed_audit import contextualize_citation_queries
        from research_assistant.shared.llm import ChatResult

        mock_chat.return_value = ChatResult(content='["q0"]')
        claims = [{"claim": "However, this method produces edge distortion.",
                   "context": "«However, this method produces edge distortion.»", "paragraph_id": "p_0",
                   "ref": {"authors": ["Settnes"], "year": 2015, "title": "Wavelet Transforms"}}]
        with patch("research_assistant.shared.seed_audit.CITATION_AUDIT_CONTEXTUALIZE_QUERIES", True):
            contextualize_citation_queries(claims)
        prompt = mock_chat.call_args[0][0][0]["content"]
        for absent in ("Section of the citing paper", "Figures and tables", "What the cited paper is about"):
            self.assertNotIn(absent, prompt)
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_seed_audit.py::TestQueryContextualizationAndCompoundCitations -q`
Expected: the first new test FAILS on the `Section of the citing paper` assertion; the second passes already.

- [ ] **Step 3: Rewrite the loop body**

In `contextualize_citation_queries`, replace everything from `context_text = unqueried[0].get("context") …` through the closing `)` of `prompt = (...)` with:

```python
        first = unqueried[0]
        context_text = first.get("context") or first.get("sentence") or ""
        section = first.get("section_heading") or ""
        artifact_lines = "\n".join(
            f"- {a['label']}: {a['caption']}" for a in (first.get("artifacts") or []) if a.get("caption")
        )

        # One summary per cited paper per paragraph: the second claim on the
        # same paper points back at the first instead of repeating 180 words.
        claim_entries = []
        summarised: dict[str, int] = {}
        for idx, c in enumerate(unqueried):
            ref = c.get("ref") or {}
            ref_str = f"{', '.join((ref.get('authors') or [])[:2])} ({ref.get('year') or 'n.d.'}) - '{ref.get('title') or ''}'"
            entry = f"[{idx}] Cited Reference: {ref_str}\n    Claim Sentence: \"{c.get('claim', '')}\""
            summary, doc = c.get("cited_summary"), c.get("document")
            if summary:
                if doc in summarised:
                    entry += f"\n    What the cited paper is about: as for [{summarised[doc]}]"
                else:
                    summarised[doc] = idx
                    entry += f"\n    What the cited paper is about: {summary}"
            claim_entries.append(entry)

        prompt = (
            "You are a scientific retrieval assistant. For each citation claim extracted from the "
            "paragraph below, generate a focused, standalone search query to find the supporting "
            "passage in the cited paper.\n\n"
            "Rules:\n"
            "1. Resolve pronouns ('this method', 'they', 'the authors', 'this result') to the specific "
            "technique, theory, or findings described in the paragraph.\n"
            "2. Where the sentence points at a figure or table, say what that figure shows (from its "
            "caption below) instead of its number — the cited paper has its own figure numbers.\n"
            "3. Phrase the query in the cited paper's own vocabulary: its summary, where given, says "
            "what it calls its system and method.\n"
            "4. Include the cited author's name, publication year, and essential domain keywords.\n"
            "5. Keep each query concise (10-25 words), focused on concrete technical search terms.\n"
            "6. Return ONLY a valid JSON array of strings in the exact same order as the inputs, e.g.:\n"
            '["query for 0", "query for 1"]\n\n'
            + (f"Section of the citing paper: {section}\n\n" if section else "")
            + f"Paragraph Context:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            + (f"Figures and tables the paragraph refers to:\n{artifact_lines}\n\n" if artifact_lines else "")
            + "Citations to Contextualize:\n" + "\n".join(claim_entries) + "\n\n"
            "JSON array of queries:"
        )
```

The `try: res = chat(...)` block after it is unchanged.

- [ ] **Step 4: Run the file**

Run: `python3 -m pytest tests/test_seed_audit.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/seed_audit.py tests/test_seed_audit.py
git commit -m "feat(audit): retrieval queries are written from the section, the figure captions and the cited paper's summary

'The discrepancy in Fig. 3b' becomes the quantity the caption names; the
query is phrased in the vocabulary the cited paper uses for its own
system. Same one call per paragraph.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The report and the UI show what was resolved

**Files:**
- Modify: `research_assistant/shared/seed_audit.py:1711` (`generate_seed_audit_markdown`, the "Cited in" line)
- Modify: `app.py:684-685` (`_render_claim_item`, after the Retrieval Query caption)
- Test: `tests/test_seed_audit.py` (class `TestHonestMarkdown`)

**Interfaces:**
- Consumes: `item["section_heading"]`, `item["artifacts"]`, `item["cited_summary"]`.
- Produces: nothing new for later tasks.

- [ ] **Step 1: Write the failing test**

Add to `TestHonestMarkdown`:

```python
    def test_cited_in_shows_the_breadcrumb_when_known(self):
        md = generate_seed_audit_markdown(self._report([
            {"outcome": "cap_exceeded", "judgement": None, "downloaded": True, "role": "evidential",
             "section": "results", "section_heading": "2. Results and Discussion > 2.1. Terahertz Spectral Analysis",
             "reason": "Maximum claims evaluation budget reached.",
             "reliability": "UNRESOLVED", "reliability_badge": "⏳ Not Assessed",
             "reliability_explanation": "Not assessed — the per-paper claim budget was reached before this citation."},
            {"outcome": "cap_exceeded", "judgement": None, "downloaded": True, "role": "evidential",
             "section": "results", "section_heading": "",
             "reason": "Maximum claims evaluation budget reached.",
             "reliability": "UNRESOLVED", "reliability_badge": "⏳ Not Assessed",
             "reliability_explanation": "Not assessed — the per-paper claim budget was reached before this citation."},
        ]))
        self.assertIn("**Cited in:** 2. Results and Discussion > 2.1. Terahertz Spectral Analysis", md)
        self.assertIn("**Cited in:** results", md)   # the kind when no heading is known
```

- [ ] **Step 2: Run to see it fail**

Run: `python3 -m pytest tests/test_seed_audit.py::TestHonestMarkdown -q`
Expected: FAIL on the breadcrumb assertion.

- [ ] **Step 3: Change the report line and add the UI captions**

(a) `research_assistant/shared/seed_audit.py`, in `generate_seed_audit_markdown`:

```python
        role = item.get("role"); section = item.get("section_heading") or item.get("section")
```

(b) `app.py`, in `_render_claim_item`, directly after

```python
                if item.get("search_query") and item.get("outcome") == "judged":
                    st.caption(f"🔎 **Retrieval Query:** *{item['search_query']}*")
```

add

```python
                if item.get("artifacts"):
                    st.caption("🖼️ **Refers to:** " + " · ".join(
                        f"{a['label']} — {a['caption'][:120]}" for a in item["artifacts"] if a.get("caption")
                    ))
                if item.get("cited_summary"):
                    st.caption(f"📄 **Cited paper, in brief:** {item['cited_summary']}")
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_seed_audit.py -q && python3 -c "import ast; ast.parse(open('app.py').read()); print('app.py parses')"`
Expected: all pass; `app.py parses`.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/seed_audit.py app.py tests/test_seed_audit.py
git commit -m "feat(audit): the report names the section heading; the UI shows the figures a claim refers to and the cited paper in brief

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Phase 1 sanity run against the baseline

**Files:**
- Create (git-ignored): `data/eval/audit/B.json`

**Interfaces:**
- Consumes: `data/eval/audit/A.json` from Task 0.
- Produces: the before/after numbers, recorded in this task's commit message (an empty commit — nothing tracked changes).

This is a sanity check on ≈20 claims, not a measurement: it shows the new fields flow through a real run and that retrieval did not get worse. Verdict-quality measurement is Phase 2's job.

- [ ] **Step 1: Run the audit again on the same seed**

Run:
```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python - <<'EOF'
import json
from research_assistant.shared.seed_audit import audit_seed_citations
r = audit_seed_citations("data/raw/processed/arxiv_2007.12504v1.pdf", force=True, skip_if_cached=False)
json.dump(r, open("data/eval/audit/B.json", "w"), indent=1)
print(r["totals"])
EOF
```
Expected: totals with `judged` ≈ the baseline's.

- [ ] **Step 2: Compare**

Run:
```bash
python3 - <<'EOF'
import json
from collections import Counter
A, B = (json.load(open(f"data/eval/audit/{x}.json")) for x in "AB")
def key(i): return (i.get("paragraph_id"), i.get("sentence_index"), (i.get("ref") or {}).get("xml_id"))
b = {key(i): i for i in B["results"]}
print(f"{'claim':52} | A outcome/sufficiency/judgement | B outcome/sufficiency/judgement")
for a in A["results"]:
    if a.get("outcome") not in ("judged", "no_evidence"):
        continue
    m = b.get(key(a)) or {}
    print(f"{a['claim'][:50]:52} | {a.get('outcome')}/{a.get('evidence_sufficiency')}/{a.get('judgement')} | {m.get('outcome')}/{m.get('evidence_sufficiency')}/{m.get('judgement')}")
for name, rep in (("A", A), ("B", B)):
    js = [i for i in rep["results"] if i.get("outcome") == "judged"]
    print(name, "judged", len(js), "no_evidence", sum(1 for i in rep["results"] if i.get("outcome") == "no_evidence"),
          "sufficiency", dict(Counter(i.get("evidence_sufficiency") for i in js)))
print("B claims with artifacts:", sum(1 for i in B["results"] if i.get("artifacts")),
      "with cited_summary:", sum(1 for i in B["results"] if i.get("cited_summary")),
      "with a heading:", sum(1 for i in B["results"] if i.get("section_heading")))
print()
for i in B["results"]:
    if i.get("artifacts") and i.get("search_query"):
        print("ARTIFACT", [a["label"] for a in i["artifacts"]], "→", i["search_query"])
EOF
```

Acceptance, all three or the task is not done:
1. `B` has `no_evidence` ≤ `A`'s and the number of `sufficient` judgements ≥ `A`'s.
2. `B claims with artifacts` > 0, `with cited_summary` > 0, `with a heading` > 0.
3. Every `ARTIFACT … →` line's query names what the figure shows (a quantity, a material, a method) rather than "Fig. N". If one still says "Fig.", paste that paragraph's prompt into a scratch file and adjust rule 2's wording in Task 6; re-run the two Task 6 tests and this step.

- [ ] **Step 3: Record the numbers**

```bash
git commit --allow-empty -m "chore(audit): phase-1 sanity run on arxiv_2007.12504v1

baseline (A): <paste the A line from Step 2>
after    (B): <paste the B line from Step 2>
artifacts/summaries/headings on B: <paste>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 2 — Judge-side context, behind an evaluation gate

### Task 9: `judge.compose_context` — the one place the judge's context block is assembled

**Files:**
- Modify: `research_assistant/judgement/judge.py:248-264` (`_CONTEXT_HEADER`, then a new function before `build_prompt`)
- Test: `tests/test_judgement.py`

**Interfaces:**
- Produces: `judge.compose_context(window: str | None, section: str | None = None, artifacts: list[dict] | None = None) -> str | None`; `judge.ARTIFACT_CAPTION_CHARS = 200`. Used by `seed_audit._judge_claim_entry` (Task 13) and `scripts/evaluate_judge.py` (Task 11).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_judgement.py` (the module already imports `judge_mod` and `build_prompt`):

```python
class TestComposeContext(unittest.TestCase):
    """The judge's context block is assembled in one place, from the citing
    paper's parts only. The cited paper's summary is never one of them."""

    def test_window_alone_is_the_window(self):
        self.assertEqual(judge_mod.compose_context("Before. «claim» After."), "Before. «claim» After.")

    def test_section_and_artifacts_precede_the_window(self):
        out = judge_mod.compose_context(
            "Before. «claim» After.",
            section="2. Results > 2.1. THz",
            artifacts=[{"label": "Fig. 1", "caption": "THz  spectra\nof films."}, {"label": "Fig. 2", "caption": ""}],
        )
        self.assertEqual(out, "[Section: 2. Results > 2.1. THz]\n[Fig. 1: THz spectra of films.]\nBefore. «claim» After.")

    def test_caption_is_capped(self):
        out = judge_mod.compose_context("«c»", artifacts=[{"label": "Fig. 1", "caption": "x" * 500}])
        self.assertEqual(out, f"[Fig. 1: {'x' * judge_mod.ARTIFACT_CAPTION_CHARS}]\n«c»")

    def test_no_window_is_none_even_with_a_section(self):
        self.assertIsNone(judge_mod.compose_context("", section="2. Results"))
        self.assertIsNone(judge_mod.compose_context(None, artifacts=[{"label": "Fig. 1", "caption": "c"}]))

    def test_header_names_the_parts_and_keeps_the_never_as_evidence_rule(self):
        prompt = build_prompt("claim", "evidence", context="«claim»")
        self.assertIn("captions of figures", prompt)
        self.assertIn("never as evidence", prompt)
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_judgement.py::TestComposeContext -q`
Expected: FAIL — `AttributeError: module … has no attribute 'compose_context'`

- [ ] **Step 3: Implement**

In `research_assistant/judgement/judge.py`, replace `_CONTEXT_HEADER` (line 248) and add the function above `build_prompt`:

```python
_CONTEXT_HEADER = (
    "**Context** (from the citing paper: the section the claim sits in, the captions of figures "
    "and tables its paragraph points at, and the sentences around it; the claim is the sentence "
    "between « and »; judge only that sentence, and use the rest only to resolve what its words "
    "refer to — never as evidence):\n"
)

ARTIFACT_CAPTION_CHARS = 200


def compose_context(window: str | None, section: str | None = None, artifacts: list | None = None) -> str | None:
    """The judge's context block from its parts: a [Section: …] header, one
    line per figure or table the paragraph points at, then the sentence
    window with the claim marked. None without a window — a header alone
    tells the judge nothing about the claim. The cited paper's summary is
    deliberately not a part: it is model prose about the paper the
    evidence comes from, and the grounding rule forbids using it as such."""
    if not window or not str(window).strip():
        return None
    lines = []
    if section and str(section).strip():
        lines.append(f"[Section: {str(section).strip()}]")
    for a in artifacts or []:
        caption = " ".join(str(a.get("caption") or "").split())[:ARTIFACT_CAPTION_CHARS]
        if a.get("label") and caption:
            lines.append(f"[{a['label']}: {caption}]")
    lines.append(str(window).strip())
    return "\n".join(lines)
```

- [ ] **Step 4: Run the file**

Run: `python3 -m pytest tests/test_judgement.py -q`
Expected: all pass (the six prompt examples still render without a context block).

- [ ] **Step 5: Commit**

```bash
git add research_assistant/judgement/judge.py tests/test_judgement.py
git commit -m "feat(judge): compose_context assembles the context block from the citing paper's parts

Section breadcrumb, one caption line per referenced figure or table, the
sentence window. Not the cited paper's summary: it reads as evidence and
the grounding rule forbids using it as such.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Eval cases can carry context, artifacts and tags

**Files:**
- Modify: `research_assistant/judgement/evalset.py:22-30, 36-56`
- Modify: `research_assistant/judgement/harvest.py:57-71`
- Modify: `research_assistant/judgement/labeling.py:22-51`
- Modify: `scripts/judge_label.py:47-65`
- Test: `tests/test_evalset.py`, `tests/test_harvest.py`, `tests/test_labeling.py`

**Interfaces:**
- Produces: on every validated case, optional `context: str | None`, `section_heading: str | None`, `artifacts: list | None`, `tags: list[str]` (always a list). `labeling.apply_label(candidate, key, existing_ids, note="", negation="", tags=())`. Candidates from `harvest.candidates_from_verification` carry `context`, `section_heading`, `artifacts`.

- [ ] **Step 1: Write the failing tests**

`tests/test_evalset.py`, in `TestValidateCase`:

```python
    def test_context_fields_default_to_none_and_tags_to_a_list(self):
        c = es.validate_case(_case())
        self.assertIsNone(c["context"])
        self.assertIsNone(c["section_heading"])
        self.assertIsNone(c["artifacts"])
        self.assertEqual(c["tags"], [])
        c = es.validate_case(_case(tags=["figure_ref"], context="«x»", section_heading="2. Results"))
        self.assertEqual((c["tags"], c["context"], c["section_heading"]), (["figure_ref"], "«x»", "2. Results"))
```

`tests/test_harvest.py`, in `TestCandidates`:

```python
    def test_candidates_from_verification_carry_the_context_the_judge_saw(self):
        report = {"results": [{
            "outcome": "judged", "claim": "C1", "evidence": "E1", "citation_source": "S1", "judgement": "Supports",
            "context": "B. «C1» A.", "section_heading": "2. Results",
            "artifacts": [{"id": "fig_0", "kind": "figure", "label": "Fig. 1", "caption": "cap"}],
        }, {"outcome": "judged", "claim": "C2", "evidence": "E2", "citation_source": "S2", "judgement": "Supports"}]}
        out = hv.candidates_from_verification(report, prefix="v1")
        self.assertEqual((out[0]["context"], out[0]["section_heading"], out[0]["artifacts"][0]["label"]),
                         ("B. «C1» A.", "2. Results", "Fig. 1"))
        self.assertEqual((out[1]["context"], out[1]["section_heading"], out[1]["artifacts"]), (None, None, None))
```

`tests/test_labeling.py`, in `TestApplyLabel`:

```python
    def test_context_and_tags_ride_along_on_every_case(self):
        cand = _cand(context="B. «claim» A.", section_heading="2. Results",
                     artifacts=[{"label": "Fig. 1", "caption": "cap"}])
        human, negation = lb.apply_label(cand, "s", set(), negation="MoS2 networks do not conduct by hopping.",
                                         tags=["figure_ref", "method_transfer"])
        for case in (human, negation):
            self.assertEqual(case["context"], "B. «claim» A.")
            self.assertEqual(case["section_heading"], "2. Results")
            self.assertEqual(case["artifacts"][0]["label"], "Fig. 1")
            self.assertEqual(case["tags"], ["figure_ref", "method_transfer"])
        self.assertEqual(lb.apply_label(_cand(), "d", set())[0]["tags"], [])
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_evalset.py tests/test_harvest.py tests/test_labeling.py -q`
Expected: three FAILs (`KeyError: 'context'` / `'tags'`, `TypeError: … unexpected keyword argument 'tags'`).

- [ ] **Step 3: Implement**

(a) `research_assistant/judgement/evalset.py` — `_OPTIONAL_DEFAULTS` becomes:

```python
_OPTIONAL_DEFAULTS = {
    "accept": None,
    "transform": None,
    "origin": None,
    "document": None,
    "citation_source": None,
    "notes": "",
    "model_judgement_at_harvest": None,
    # What the judge was shown besides claim and evidence, when the case
    # was harvested from an audit; None for cases that never had it.
    "context": None,
    "section_heading": None,
    "artifacts": None,
    # Operator labels for slicing the score: "figure_ref", "method_transfer", …
    "tags": None,
}
```

and in `validate_case`, after `case["accept"] = accept`:

```python
    case["tags"] = [str(t) for t in (case.get("tags") or [])]
```

(b) `research_assistant/judgement/harvest.py`, in `candidates_from_verification`, the appended dict gains three keys after `"section": None,`:

```python
            "context": entry.get("context"),
            "section_heading": entry.get("section_heading"),
            "artifacts": entry.get("artifacts"),
```

(c) `research_assistant/judgement/labeling.py` — `apply_label` becomes:

```python
def apply_label(candidate: dict, key: str, existing_ids: set, note: str = "", negation: str = "", tags=()) -> list[dict]:
    judgement = KEYS[key.strip().lower()]
    ids = set(existing_ids)
    hid = next_id(ids, "h")
    ids.add(hid)
    carried = {
        "document": candidate.get("document"),
        "citation_source": candidate.get("citation_source"),
        "context": candidate.get("context"),
        "section_heading": candidate.get("section_heading"),
        "artifacts": candidate.get("artifacts"),
        "tags": [str(t) for t in tags],
    }
    human = {
        "id": hid,
        "source": "human",
        "claim": candidate["claim"],
        "citation_evidence": candidate["citation_evidence"],
        "expected_judgement": judgement,
        **carried,
        "notes": note or "",
        "model_judgement_at_harvest": (candidate.get("hidden") or {}).get("model_judgement"),
    }
    out = [human]
    if judgement == "Supports" and negation.strip():
        out.append({
            "id": next_id(ids, "n"),
            "source": "human_negation",
            "claim": negation.strip(),
            "citation_evidence": candidate["citation_evidence"],
            "expected_judgement": "Contradicts",
            "origin": hid,
            **carried,
            "notes": "operator-written negation of " + hid,
        })
    return out
```

(d) `scripts/judge_label.py`, in the loop: after the `EVIDENCE` print add

```python
        if cand.get("context"):
            print("CONTEXT (what the judge saw around the claim):\n" + textwrap.fill(cand["context"], 78) + "\n")
```

and replace the `note = input(...)` / `new = lb.apply_label(...)` pair with

```python
        note = input("note (Enter to skip): ").strip()
        tags = [t.strip() for t in input("tags, comma-separated — figure_ref, method_transfer, … (Enter to skip): ").split(",") if t.strip()]
        new = lb.apply_label(cand, key, {c["id"] for c in cases}, note=note, negation=negation, tags=tags)
```

- [ ] **Step 4: Run the three files, then the suite**

Run: `python3 -m pytest tests/test_evalset.py tests/test_harvest.py tests/test_labeling.py -q && python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/judgement/evalset.py research_assistant/judgement/harvest.py research_assistant/judgement/labeling.py scripts/judge_label.py tests/test_evalset.py tests/test_harvest.py tests/test_labeling.py
git commit -m "feat(eval): cases carry the context the judge saw, and operator tags

A change to what the judge reads had no measurement: the cases had claim
and evidence only. Harvest keeps the window, section and captions from
the audit report; the labeller shows them and asks for tags.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: The harness can run with no context, the window, or the full block; the score reports drift and per-tag accuracy

**Files:**
- Modify: `scripts/evaluate_judge.py:22-27, 44-60, 108-140`
- Modify: `research_assistant/judgement/metrics.py:22-103, 109-143, 144-158`
- Test: `tests/test_evaluate_judge.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `judge.compose_context`; case fields from Task 10.
- Produces: `evaluate_judge.run(cases, mode="judge", runs=1, limit=None, context_mode="full")`, CLI `--context {none,window,full}` (default `full`), result key `"context_mode"`, results file `<stamp>-<mode>-<context>.json`. `metrics.score()["signals"]["drift"]`, `metrics.score()["by_tag"]`; `compare()` prints per-tag strict and drift.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate_judge.py`: every `side_effect=lambda c, e: …` in the file becomes `side_effect=lambda c, e, context=None: …` (three places: lines 37, 49, 55 — the judge is now called with a `context` keyword). Then add to `TestJudgeMode`:

```python
    def test_context_mode_controls_what_the_judge_sees(self):
        case = _case("h_1", context="Before. «claim h_1» After.", section_heading="2. Results",
                     artifacts=[{"label": "Fig. 1", "caption": "cap"}])
        seen = []

        def fake_judge(c, e, context=None):
            seen.append(context)
            return {"judgement": "Supports"}

        with patch.object(ej, "judge", side_effect=fake_judge):
            for mode in ("none", "window", "full"):
                out = ej.run([case], mode="judge", runs=1, context_mode=mode)
                self.assertEqual(out["context_mode"], mode)
        self.assertEqual(seen, [None, "Before. «claim h_1» After.",
                                "[Section: 2. Results]\n[Fig. 1: cap]\nBefore. «claim h_1» After."])
```

`tests/test_metrics.py`: give `_r` a `tags=None` parameter and put `"tags": tags` in the dict it validates; then add to `TestScore`:

```python
    def test_drift_signal_and_tag_groups(self):
        rs = [
            _r("1", "Supports", "Supports", tags=["figure_ref"],
               verdict={"rubric_violations": ["finding: assertion drift — 20% of its content words occur in the claim"]}),
            _r("2", "Supports", "Contradicts", tags=["figure_ref", "method_transfer"], verdict={"rubric_violations": []}),
            _r("3", "Supports", "Supports", verdict={}),
        ]
        s = mx.score(rs)
        self.assertAlmostEqual(s["signals"]["drift"], 0.5)   # case 3 reported no violations field: not counted
        self.assertEqual(s["by_tag"]["figure_ref"]["n"], 2)
        self.assertAlmostEqual(s["by_tag"]["figure_ref"]["strict"], 0.5)
        self.assertEqual(s["by_tag"]["method_transfer"]["n"], 1)
        self.assertIn("figure_ref", mx.render(s, refused=[]))
        self.assertIn("figure_ref", mx.compare({"summary": s, "results": []}, {"summary": s, "results": []}))
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_evaluate_judge.py tests/test_metrics.py -q`
Expected: FAIL — `TypeError: run() got an unexpected keyword argument 'context_mode'`; `KeyError: 'drift'`.

- [ ] **Step 3: Implement the harness switch**

`scripts/evaluate_judge.py`:

(a) import line: `from research_assistant.judgement.judge import PROMPT_TEMPLATE, compose_context, judge`

(b) replace `_judge_one` with:

```python
def _context_for(case, context_mode):
    """What the judge is shown besides claim and evidence: nothing, the
    sentence window the case was harvested with, or the full block the
    audit builds (section, captions, window)."""
    if context_mode == "none":
        return None
    if context_mode == "window":
        return case.get("context")
    return compose_context(case.get("context"), section=case.get("section_heading"), artifacts=case.get("artifacts"))


def _judge_one(case, mode, context_mode="full"):
    if mode == "judge":
        verdict = judge(case["claim"], case["citation_evidence"], context=_context_for(case, context_mode))
        return verdict["judgement"], verdict
```

(the verifier branch below it is unchanged; it drives `verify_draft` and ignores context).

(c) `def run(cases, mode="judge", runs=1, limit=None, context_mode="full"):` — pass `context_mode` in the call `got, verdict = _judge_one(case, mode, context_mode)` and add `"context_mode": context_mode,` to the returned dict after `"runs": runs,`.

(d) in `main()`: `ap.add_argument("--context", choices=["none", "window", "full"], default="full", help="what the judge sees besides claim and evidence (judge mode)")`; call `run(cases, mode=args.mode, runs=args.runs, limit=args.limit, context_mode=args.context)`; the results filename becomes `f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.mode}-{args.context}.json"`.

- [ ] **Step 4: Implement the metrics**

`research_assistant/judgement/metrics.py`, in `score`:

(a) `_group` takes a function returning a *list* of keys:

```python
    def _group(keys):
        groups = defaultdict(list)
        for r in scored:
            for k in keys(r):
                if k:
                    groups[k].append(r)
        return {
            k: {
                "n": len(v),
                "strict": sum(1 for r in v if r["got"] == r["case"]["expected_judgement"]) / len(v),
                "lenient": sum(1 for r in v if r["got"] in r["case"]["accept"]) / len(v),
            }
            for k, v in groups.items()
        }
```

(b) `signals` gains:

```python
        "drift": _rate([
            any("assertion drift" in x for x in (v.get("rubric_violations") or []))
            if "rubric_violations" in v else None
            for v in verdicts
        ]),
```

(c) the returned dict:

```python
        "by_source": _group(lambda r: [r["case"]["source"]]),
        "by_transform": _group(lambda r: [r["case"].get("transform")]),
        "by_tag": _group(lambda r: r["case"].get("tags") or []),
```

(d) `render`: after the `by_transform` rows add

```python
    for k, v in sorted(summary.get("by_tag", {}).items()):
        lines.append(f"| # {k} | {v['n']} | {_pct(v['strict'])} | {_pct(v['lenient'])} |")
```

and the signals line becomes

```python
    lines.append(
        f"rubric_mismatch {_pct(s['rubric_mismatch'])}  span_unverified {_pct(s['span_unverified'])}  "
        f"drift {_pct(s.get('drift'))}  escalated {_pct(s['escalated'])}  escalation changed verdict {_pct(s['escalation_changed'])}"
    )
```

(e) `compare`: after the first line append

```python
        f"drift {_pct(sa['signals'].get('drift'))} → {_pct(sb['signals'].get('drift'))}",
```

and before the `| case | A | B |` table header:

```python
    for tag in sorted(set(sa.get("by_tag", {})) | set(sb.get("by_tag", {}))):
        ta, tb = sa.get("by_tag", {}).get(tag), sb.get("by_tag", {}).get(tag)
        lines.append(
            f"# {tag}: strict {_pct(ta['strict'] if ta else None)} → {_pct(tb['strict'] if tb else None)}"
            f"  (n {ta['n'] if ta else 0} → {tb['n'] if tb else 0})"
        )
```

- [ ] **Step 5: Run the tests, then the suite**

Run: `python3 -m pytest tests/test_evaluate_judge.py tests/test_metrics.py -q && python3 -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/evaluate_judge.py research_assistant/judgement/metrics.py tests/test_evaluate_judge.py tests/test_metrics.py
git commit -m "feat(eval): --context none|window|full; the score reports drift and per-tag accuracy

window-vs-full is the comparison Phase 2 rests on; drift is the failure
mode more context is expected to worsen (e286762).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Label context-bearing cases (operator step)

**Files:**
- Create (git-ignored): `data/eval/audit/C.json`, `data/eval/judge/candidates_context.jsonl`
- Modify (tracked): `research_assistant/judgement/cases/human.jsonl`

**Interfaces:**
- Consumes: `scripts/judge_harvest.py verification`, `scripts/judge_label.py` with Task 10's fields.
- Produces: ≥ 12 human cases tagged `figure_ref` (claim's paragraph points at a figure or table), each with `context`, `section_heading`, `artifacts`; any method-borrowing claims met along the way tagged `method_transfer` (Phase 3 needs ≥ 8 of those; Task 14 tops them up).

- [ ] **Step 1: A wider harvest run**

The default budget judges 20 claims; for a candidate pool, judge more:

```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python - <<'EOF'
import json
from research_assistant.shared.seed_audit import audit_seed_citations
r = audit_seed_citations("data/raw/processed/arxiv_2007.12504v1.pdf", force=True, skip_if_cached=False, max_claims=60)
json.dump(r, open("data/eval/audit/C.json", "w"), indent=1)
print(r["totals"]); print("judged with artifacts:", sum(1 for i in r["results"] if i.get("outcome") == "judged" and i.get("artifacts")))
EOF
```
Expected: `judged with artifacts:` ≥ 12. If it is lower, run the same on `"data/raw/processed/Gabbett et al sub to nat mat.pdf"` into `data/eval/audit/C2.json` and harvest both files in Step 2.

- [ ] **Step 2: Harvest into a dedicated candidates file**

```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python scripts/judge_harvest.py verification data/eval/audit/C.json --out data/eval/judge/candidates_context.jsonl
```
Expected: `added N candidate(s)` with N = the judged count.

- [ ] **Step 3: Label**

```bash
CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=. /home/shardul/miniconda3/envs/ml/bin/python scripts/judge_label.py --candidates data/eval/judge/candidates_context.jsonl
```

Labelling rules for this pass:
- The verdict is yours, from claim and evidence under rubric V1.4; the `CONTEXT` block is there to resolve "this", "Fig. 3b", not to change what the claim says.
- Tag `figure_ref` when the context block shows a `[Fig. N: …]`/`[Table N: …]` line or the claim itself names a figure or table.
- Tag `method_transfer` when the claim attributes a method, model, formulation or synthesis route to the cited paper and applies it to the citing paper's own system (the sentence does not begin with "Following…"/"We use…" — those never reach the judge). Label the verdict as V1.5 will define it (spec §3.7): `Supports` when the evidence shows the cited paper established the method, regardless of the citing paper's new system; `Does not support` only when the claim asserts the cited paper obtained a *result* in that new system.
- `x` skips anything you cannot decide.

- [ ] **Step 4: Check the counts and commit**

```bash
python3 - <<'EOF'
import json
cases = [json.loads(l) for l in open("research_assistant/judgement/cases/human.jsonl") if l.strip()]
from collections import Counter
print("human cases", len(cases), Counter(t for c in cases for t in (c.get("tags") or [])))
EOF
python3 -m pytest tests/test_evalset.py -q
git add research_assistant/judgement/cases/human.jsonl
git commit -m "eval(judge): <N> context-bearing cases from the arxiv_2007.12504v1 audit (<F> figure_ref, <M> method_transfer)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
Expected: `figure_ref` ≥ 12; the evalset tests still load every file.

---

### Task 13: The audit's judge sees the breadcrumb and captions — and the A/B that decides whether it keeps them

**Files:**
- Modify: `research_assistant/shared/seed_audit.py:603-607` (`_judge_claim_entry`, the `claim_context` block)
- Test: `tests/test_seed_audit.py`

**Interfaces:**
- Consumes: `judge.compose_context`; `item["section_heading"]`, `item["artifacts"]`, `item["section"]`, `item["context"]`.
- Produces: the judge's `context` kwarg is `compose_context(window, section=<breadcrumb or kind>, artifacts)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_seed_audit.py`:

```python
class TestJudgeContext(unittest.TestCase):
    def _verdict(self):
        return {
            "judgement": "Supports", "confidence": "High", "supporting_span": "Evidence.", "reason": "r",
            "slots": {"finding": {"assertion": "a", "verdict": "Supports"}, "scope": {"assertion": "s", "verdict": "Supports"},
                      "strength": {"assertion": "t", "verdict": "Not applicable"}},
            "evidence_sufficiency": "sufficient",
        }

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    def test_judge_sees_breadcrumb_and_captions_but_never_the_cited_summary(self, mock_search, mock_judge):
        from research_assistant.shared.seed_audit import _judge_claim_entry
        mock_search.return_value = [{"text": "Evidence.", "metadata": {"document": "x.pdf", "section": "results", "page": 3}}]
        mock_judge.return_value = self._verdict()
        item = {"claim": "The enhancement in Fig. 1b agrees with the model.", "document": "x.pdf",
                "context": "Before. «The enhancement in Fig. 1b agrees with the model.» After.",
                "section": "results", "section_heading": "2. Results > 2.1. THz",
                "artifacts": [{"id": "fig_0", "kind": "figure", "label": "Fig. 1", "caption": "THz photoconductivity."}],
                "cited_summary": "SUMMARY-MUST-NOT-APPEAR"}
        _judge_claim_entry(item, MagicMock(), MagicMock(), [], [])
        ctx = mock_judge.call_args.kwargs["context"]
        self.assertEqual(ctx, "[Section: 2. Results > 2.1. THz]\n[Fig. 1: THz photoconductivity.]\n"
                              "Before. «The enhancement in Fig. 1b agrees with the model.» After.")
        self.assertNotIn("SUMMARY-MUST-NOT-APPEAR", ctx)

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    def test_the_kind_stands_in_when_no_heading_is_known(self, mock_search, mock_judge):
        from research_assistant.shared.seed_audit import _judge_claim_entry
        mock_search.return_value = [{"text": "Evidence.", "metadata": {"document": "x.pdf"}}]
        mock_judge.return_value = self._verdict()
        item = {"claim": "A claim.", "document": "x.pdf", "context": "«A claim.»", "section": "results", "section_heading": ""}
        _judge_claim_entry(item, MagicMock(), MagicMock(), [], [])
        self.assertEqual(mock_judge.call_args.kwargs["context"], "[Section: Results]\n«A claim.»")
```

- [ ] **Step 2: Run to see them fail**

Run: `python3 -m pytest tests/test_seed_audit.py::TestJudgeContext -q`
Expected: the first FAILS (context lacks the `[Fig. 1: …]` line); the second passes already.

- [ ] **Step 3: Wire it**

In `_judge_claim_entry`, replace

```python
    claim_context = item.get("context")
    if item.get("section") and claim_context and not str(claim_context).startswith("[Section:"):
        sec_header = f"[Section: {str(item['section']).replace('_', ' ').title()}]\n"
        claim_context = f"{sec_header}{claim_context}"
```

with

```python
    # The breadcrumb when the TEI gave one, the kind otherwise; the captions
    # of the figures the paragraph points at; never the cited summary.
    from research_assistant.judgement.judge import compose_context

    section = item.get("section_heading") or (
        str(item["section"]).replace("_", " ").title() if item.get("section") else None
    )
    claim_context = compose_context(item.get("context"), section=section, artifacts=item.get("artifacts"))
```

- [ ] **Step 4: Run the file and commit**

Run: `python3 -m pytest tests/test_seed_audit.py -q`
Expected: all pass.

```bash
git add research_assistant/shared/seed_audit.py tests/test_seed_audit.py
git commit -m "feat(audit): the judge's context carries the section breadcrumb and the referenced figures' captions

Assembled by judge.compose_context — the cited paper's summary is not a
part of it. Whether this stays is decided by the window-vs-full run in
the same task.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: The A/B — window against full, on the human cases, two runs each**

```bash
export CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=.
PY=/home/shardul/miniconda3/envs/ml/bin/python
$PY scripts/evaluate_judge.py --mode judge --source human --source human_negation --context window --runs 2
$PY scripts/evaluate_judge.py --mode judge --source human --source human_negation --context full --runs 2
ls -t data/eval/judge/results/ | head -2
```
Then, with the two filenames just listed (`…-judge-window.json` and `…-judge-full.json`):
```bash
$PY scripts/evaluate_judge.py --compare data/eval/judge/results/<window>.json data/eval/judge/results/<full>.json
```

Acceptance — `full` keeps the wiring only if all hold:
1. `# figure_ref: strict` under `full` ≥ under `window`.
2. Overall `strict` under `full` ≥ under `window` minus one case (one case = 1 / (n × runs)).
3. `drift` under `full` ≤ `drift` under `window` + 0.05.

If any fails: `git revert HEAD` (the Step 4 commit; `compose_context` and the harness stay), and put the two `--compare` outputs in the revert message. The audit then keeps the V1.4 context block and the measured reason is on record.

- [ ] **Step 6: Record the result**

```bash
git commit --allow-empty -m "eval(judge): window vs full context on <n> human cases × 2 runs

<paste the --compare output>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Phase 3 — The method-transfer scope rule

**Stated assumption (spec §3.7):** `method` stays in `SKIP_ROLES`. Sentences cued by *following / according to / we use…* keep being filed as `not_a_claim`. The rule below is for *evidential* sentences that attribute a method to the cited paper and apply it to the citing paper's own system. Taking `method` out of `SKIP_ROLES` is a separate decision and not made here.

### Task 14: Label method-transfer cases (operator step)

**Files:**
- Modify (tracked): `research_assistant/judgement/cases/human.jsonl`
- Create (git-ignored, if needed): `data/eval/audit/C2.json`, `data/eval/judge/candidates_context2.jsonl`

**Interfaces:**
- Produces: ≥ 8 human cases tagged `method_transfer`, labelled as V1.5 defines the verdict (spec §3.7).

- [ ] **Step 1: Count what Task 12 already produced**

```bash
python3 - <<'EOF'
import json
cases = [json.loads(l) for l in open("research_assistant/judgement/cases/human.jsonl") if l.strip()]
print("method_transfer:", sum(1 for c in cases if "method_transfer" in (c.get("tags") or [])))
EOF
```

- [ ] **Step 2: If fewer than 8, harvest the second seed**

```bash
export CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=.
PY=/home/shardul/miniconda3/envs/ml/bin/python
$PY - <<'EOF'
import json
from research_assistant.shared.seed_audit import audit_seed_citations
r = audit_seed_citations("data/raw/processed/Gabbett et al sub to nat mat.pdf", force=True, skip_if_cached=False, max_claims=60)
json.dump(r, open("data/eval/audit/C2.json", "w"), indent=1)
print(r["totals"])
EOF
$PY scripts/judge_harvest.py verification data/eval/audit/C2.json --out data/eval/judge/candidates_context2.jsonl
$PY scripts/judge_label.py --candidates data/eval/judge/candidates_context2.jsonl
```
In this pass label **only** candidates that read as a borrowed method (`x` for everything else): the claim names a method, model, formulation, fitting procedure or synthesis route as the cited paper's and uses it on the citing paper's own material or device. Verdict per spec §3.7; tag `method_transfer`; add `figure_ref` too where it applies.

- [ ] **Step 3: Commit what there is**

```bash
python3 -m pytest tests/test_evalset.py -q
git add research_assistant/judgement/cases/human.jsonl
git commit -m "eval(judge): <M> method_transfer cases

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
If `<M>` is under 8 after both seeds, say so in the message; Task 16's comparison is then indicative, not decisive, and the commit message of Task 16 says that too.

---

### Task 15: Rubric V1.5 — the rule, example G, and the version bump

**Files:**
- Modify: `research_assistant/judgement/prompt.md:1, 35-39, 121, 223-249`
- Modify: `research_assistant/judgement/cases/cases.jsonl` (append one line)
- Modify: `research_assistant/judgement/__init__.py:1`, `app.py:1497`, `HOW_TO_USE.md` (three `V1.4` mentions)
- Test: `tests/test_judgement.py:347`, `tests/test_evalset.py:79`

**Interfaces:**
- Consumes: the harness from Task 11 with `--context full` and the tagged cases from Tasks 12 and 14.
- Produces: rubric V1.5; `cases.jsonl` has seven `prompt_example` cases.

- [ ] **Step 1: Score V1.4 first, before touching the prompt**

```bash
export CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=.
PY=/home/shardul/miniconda3/envs/ml/bin/python
$PY scripts/evaluate_judge.py --mode judge --source human --source human_negation --context full --runs 2
ls -t data/eval/judge/results/ | head -1
```
Keep that filename: it is `A` for Task 16.

- [ ] **Step 2: Update the two count assertions so they fail first**

`tests/test_judgement.py`, in `test_the_prompts_own_examples_do_not_trip_the_drift_check`: `self.assertEqual(len(blocks), 7)` and the docstring's "six" becomes "seven".
`tests/test_evalset.py`, in `test_the_shipped_prompt_examples_load`: `self.assertEqual(len(cases), 7)`.

Run: `python3 -m pytest tests/test_judgement.py tests/test_evalset.py -q`
Expected: exactly those two FAIL.

- [ ] **Step 3: Edit `prompt.md`**

(a) Line 1: `# Claim–Evidence Verification — V1.5`

(b) In **Step 2**, directly after the `scope` bullet list (after the `Insufficient — the evidence does not state what it examined` bullet) and before `**strength**`, insert:

```markdown
A borrowed method is scoped to where it was established. When the claim attributes a *method, model, formulation or synthesis route* to the cited paper and applies it to the citing paper's own system — "the electrodes were made by the in situ polymerisation route of the cited work, here with NiFe₂O₄" — `scope` is the system the cited paper developed the method for, not the citing paper's application: `Supports` when the evidence shows the cited paper established that method there. The citing paper's substitution is its own work, not something it attributes to the cited paper. This ends where the claim asserts a *result* in the new system — "the cited work showed the route works for NiFe₂O₄" asserts a finding about NiFe₂O₄, and `scope` is judged on NiFe₂O₄ as usual.
```

(c) The sentence `All five examples below cite the same paper, on a MnFe₂O₄@PANI nanoflower composite supercapacitor electrode.` becomes `All seven examples below cite the same paper, on a MnFe₂O₄@PANI nanoflower composite supercapacitor electrode.`

(d) After example F's closing code fence and before the line `Two contrasts are worth studying before you judge.`, insert (the blank lines and the untagged code fence matter — `tests/test_judgement.py` parses examples with a regex over exactly this shape):

````markdown
### G. Method borrowed for a new system → Supports

Claim: *The electrodes were made by the in situ oxidative polymerisation route of the cited work, here with NiFe₂O₄ in place of MnFe₂O₄.*
Evidence: *MnFe2O4@PANI nanocomposites were synthesized by in situ oxidative polymerization of aniline in the presence of MnFe2O4 nanoparticles, using ammonium persulfate as the oxidant in 1M HCl.*

```
{
  "slots": {
    "finding":  {"assertion": "the electrodes are made by in situ oxidative polymerisation", "verdict": "Supports"},
    "scope":    {"assertion": "the in situ polymerisation route for MnFe2O4@PANI", "verdict": "Supports"},
    "strength": {"assertion": "the route is used", "verdict": "Not applicable"}
  },
  "judgement": "Supports",
  "evidence_sufficiency": "sufficient",
  "confidence": "High",
  "supporting_span": "MnFe2O4@PANI nanocomposites were synthesized by in situ oxidative polymerization of aniline in the presence of MnFe2O4 nanoparticles",
  "reason": "The claim borrows the cited paper's synthesis route and says so; NiFe2O4 is the citing paper's own substitution, not a result attributed to the cited paper, so scope is the route as established for MnFe2O4@PANI."
}
```

---

````

(e) `Two contrasts are worth studying before you judge.` becomes `Three contrasts are worth studying before you judge.`, and after the **B against F** paragraph add:

```markdown
**E against G.** Both claims reach into a system the cited paper never tested. E attributes a *result* there and fails `scope`; G borrows a *method* and scopes it to where the method was established. What the claim attributes to the cited paper decides — not what the citing paper goes on to do with it.
```

- [ ] **Step 4: Add case G to `cases.jsonl`**

Append one line — claim and evidence copied character-for-character from the prompt, since `evalset.held_out` matches on the text:

```json
{"id": "case_G", "name": "method_borrowed", "claim": "The electrodes were made by the in situ oxidative polymerisation route of the cited work, here with NiFe₂O₄ in place of MnFe₂O₄.", "citation_evidence": "MnFe2O4@PANI nanocomposites were synthesized by in situ oxidative polymerization of aniline in the presence of MnFe2O4 nanoparticles, using ammonium persulfate as the oxidant in 1M HCl.", "expected_judgement": "Supports"}
```

- [ ] **Step 5: Bump the version everywhere it is named**

```bash
sed -i '1s/(V1.4)/(V1.5)/' research_assistant/judgement/__init__.py
sed -i '1497s/Rubric V1.4/Rubric V1.5/' app.py
sed -i 's/V1\.4/V1.5/g' HOW_TO_USE.md
grep -rn "V1\.4" --include=*.py --include=*.md . | grep -v notimportant | grep -v "\.agents"
```
Expected: the `grep` prints nothing.

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/test_judgement.py tests/test_evalset.py tests/test_evaluate_judge.py -q && python3 -m pytest tests/ -q`
Expected: all pass — in particular `test_the_prompts_own_examples_do_not_trip_the_drift_check` (G's slot assertions share their content words with the claim and evidence) and `test_prompt_examples_are_not_held_out` (case G is recognised as in-prompt).

- [ ] **Step 7: Commit**

```bash
git add research_assistant/judgement/prompt.md research_assistant/judgement/cases/cases.jsonl research_assistant/judgement/__init__.py app.py HOW_TO_USE.md tests/test_judgement.py tests/test_evalset.py
git commit -m "feat(judge): rubric V1.5 — a borrowed method is scoped to where it was established

A claim that takes the cited paper's method to a new system was failing
scope for the system the cited paper never tested. Scope is now the
method's own system unless the claim asserts a result in the new one.
Example G, on the same paper as A–F.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 16: V1.4 against V1.5 on the tagged cases

**Files:** none tracked.

**Interfaces:**
- Consumes: the `A` results file from Task 15 Step 1; the harness.
- Produces: the comparison, in an empty commit's message.

- [ ] **Step 1: Score V1.5 the same way**

```bash
export CITATION_INDEX_VERSION=2 CITATION_LOG_FILE=0 PYTHONPATH=.
PY=/home/shardul/miniconda3/envs/ml/bin/python
$PY scripts/evaluate_judge.py --mode judge --source human --source human_negation --context full --runs 2
ls -t data/eval/judge/results/ | head -1
```

- [ ] **Step 2: Compare**

```bash
$PY scripts/evaluate_judge.py --compare data/eval/judge/results/<A from Task 15>.json data/eval/judge/results/<B from Step 1>.json
```

Acceptance:
1. `# method_transfer: strict` goes up.
2. Overall `strict` does not go down by more than one case; `# figure_ref: strict` does not go down.
3. `drift` does not go up by more than 0.05.
4. In-prompt examples: n = 14 (7 × 2 runs), strict 100% — printed by the `render` of Step 1's run under **In-prompt examples**.

If 1 holds and 2–4 hold, done. If 1 fails, the rule's wording is not landing: read the `reason` fields of the `method_transfer` cases in the B results file (`results[].verdict.reason` where `results[].expected != results[].got`), adjust the rule paragraph in `prompt.md` (not the example), re-run Steps 1–2 once, and commit the wording change with the second comparison. If 2–4 fail with 1 holding, revert Task 15's commit and record why.

- [ ] **Step 3: Record**

```bash
git commit --allow-empty -m "eval(judge): V1.4 vs V1.5, <n> human cases × 2 runs, context full

<paste the --compare output>

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Not in this plan (and why)

- **Unifying the audit onto `extract.parse_tei`.** The right long-term fix for the two-parser problem (spec §2.1); a refactor of `extract_seed_citation_claims` with its own plan. `tei_structure.py` is written against BeautifulSoup so it drops into the audit today, and its two walks port to lxml in an afternoon when that plan comes.
- **Judging `method`-role citations.** Would change the budget and the prioritisation; decide first, then plan.
- **A seed "thesis" / abstract in the judge prompt.** No evidence it helps; more context is the documented failure mode.
- **The cited summary in the judge prompt.** Ruled out by the grounding rule (spec §3.2), not deferred.
