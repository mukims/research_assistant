# Citation Audit Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The seed-paper citation audit judges real sentences (GROBID's own, with citations handled as tokens), spends its budget on the citations that matter, stops when the model backend is down, and reports only what the judge actually produced — with every unjudged citation labelled by the reason it was not judged.

**Architecture:** A new pure module `research_assistant/shared/claim_text.py` owns sentence segmentation, citation tokens, claim rendering, claim quality and citation role. `seed_audit.extract_seed_citation_claims` is rewired onto it and gains number-map reference resolution and per-claim `section / role / context` fields. `audit_seed_citations` partitions non-claims before judging, prioritises by section and cluster size, applies a per-sentence cap and a consecutive-failure circuit breaker, and leaves `judgement=None` on every failure path. `compute_totals` becomes the single source of counts. `judgement/policy.py` learns the outcome and gains `UNSUPPORTED`; `source_assessor.py` stops grading a bare DOI as peer-reviewed; `judge.py`/`prompt.md` accept a context block; the Markdown report and Tab 1 print only judged fields for judged items.

**Tech Stack:** Python 3.10–3.12, `unittest.TestCase` collected by pytest, BeautifulSoup (`"xml"` parser, lxml), existing `shared.search.hybrid_search`, `judgement.judge`, Streamlit `app.py`.

**Spec:** `notimportant/superpowers/specs/2026-09-14-citation-audit-integrity-design.md` — §2 is the design authority; §3 is the complete outcome vocabulary.

## Global Constraints

- **Python 3.10–3.12.** No 3.13-only syntax. Walrus and `X | None` are fine.
- **Interpreter and test command:** `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest <files> -q` from the repo root. The full suite is `tests/`; anything calling a model or Chroma is skipped unless `RUN_LLM_TESTS=1`.
- **No model in any new test.** `_judge_once` and `hybrid_search` are patched where a test crosses them (the existing tests show the pattern).
- **Public names stay:** `extract_seed_citation_claims`, `audit_seed_citations`, `cross_check_seed_audit`, `generate_seed_audit_markdown`, `explain_rubric_verdict`, `_clean_claim_punctuation`, `_match_downloaded_paper` keep their signatures (new keyword arguments with defaults are allowed).
- **`judgement/cases/cases.jsonl` and the rubric text of `prompt.md` are not edited.** Only the Input section of `prompt.md` gains a slot.
- **All file writes go through `shared/atomic.py`.**
- **Outcome strings are exactly those in spec §3.** No new outcome without adding it to `NOT_ASSESSED_OUTCOMES`, `explain_rubric_verdict` and `policy.NOT_ASSESSED_EXPLANATIONS`.
- **Commit after every task; never push.** Branch: work on `synthesis-depth` (current) or `git checkout -b audit-integrity` from it.

---

## Context you need before Task 1

### What exists

- `research_assistant/shared/seed_audit.py` (≈1,400 lines). `extract_seed_citation_claims(tei_source) -> list[dict]` parses `<listBibl>` into `bib_by_id`/`bib_by_index`, then for each `<p>` replaces every `<ref type="bibr">` with `" __CITE_{i}_{target}_{txt}__ "`, calls `_clean(p)` (which is `get_text()` + whitespace collapse) and `split_into_sentences`, and emits one dict per (sentence, citation). `audit_seed_citations(seed_path, search_resources=None, max_claims=20, top_k=…, force=False, skip_if_cached=True)` groups claims by paragraph, defers paragraphs with >50 % missing references, judges `downloaded_claims[:max_claims]` through `_judge_claim_entry`, marks the rest `cap_exceeded`, runs `assess_source` + `evaluate_reliability` over every result, builds `totals`, writes `data/raw/seed_audits/<stem>_audit.json` and `.md`. `cross_check_seed_audit` re-judges a cached report's unresolved items and rebuilds the same `totals` a second time by hand. `explain_rubric_verdict(item)` and `generate_seed_audit_markdown(report)` render.
- `research_assistant/shared/claim_text.py` does not exist yet.
- `research_assistant/agents/agent5_batch_citer.py:26` — `split_into_sentences(text) -> list[str]`: masks `_ABBREVS` periods, splits on `(?<=[.!?])\s+`.
- `research_assistant/shared/extract.py:85` — `normalise_section_kind(heading) -> str` maps a heading to one of `abstract, introduction, background, methods, results, discussion, conclusion, caption, other`.
- `research_assistant/judgement/judge.py` — `build_prompt(claim, citation_evidence)`, `judge(claim, citation_evidence, model=None)`, `enforce_rubric`, `REQUIRED_FIELDS`, `DERIVED_FIELDS`, `PROMPT_TEMPLATE`. `prompt.md` ends with an `## Input` section holding `{{CLAIM}}` and `{{CITATION_EVIDENCE}}`.
- `research_assistant/agents/agent8_verifier.py:161` — `_judge_once(claim, evidence)` (retry-wrapped `judge`), `assemble_evidence(hits, max_chars)`, `needs_escalation(verdict)`, `citation_pairs(sentences, key_to_source)`.
- `research_assistant/judgement/policy.py` — `ReliabilityRating` (`HIGH, MODERATE, LOW, CONTRADICTED, UNRESOLVED`), `evaluate_reliability(relation, source_grade=None, confidence="High", span_verified=None, rubric_violations=None, rubric_mismatch=False) -> {"rating","badge","rating_label","explanation"}`. Its first branch sends `Does not support`, `Unclear…` and `Deferred…` to `UNRESOLVED`.
- `research_assistant/judgement/source_assessor.py` — `SourceGrade` (`RIGOROUS_PRIMARY, STANDARD_PRIMARY, SECONDARY_REVIEW, PREPRINT_UNREVIEWED, RETRACTED_OR_FLAWED, UNKNOWN`), `assess_source(metadata=None, ref_info=None) -> dict` with keys `grade, grade_label, badge, is_preprint, is_retracted, venue, citation_count, rationale`. Step 4 returns `STANDARD_PRIMARY` whenever `venue or doi`.
- `app.py:466` — `_render_seed_citation_audit(final)`: six metric tiles, a reliability tile row, filter tabs `Supported / Need Review / ⏳ Pending Evidence (Deferred) / All Citations`, and `_render_claim_item` (a nested function) which already prints Confidence only when `outcome == "judged"`.
- Hits from `hybrid_search` are dicts `{"chunk_index", "text", "metadata", "rrf_score"}`; `metadata` carries `document`, `section`, `page` (v2 index). `expand_neighbours` adds `context_before`/`context_after`.
- Tests: `tests/test_seed_audit.py` (inline TEI strings `SAMPLE_TEI_XML`, `MULTI_PARAGRAPH_TEI_XML`; audits are run with `@patch` on `_judge_once`, `hybrid_search`, `find_tei_for_seed`, `_load_downloaded_manifest` and `search_resources=(MagicMock(), MagicMock(), [], [])`), `tests/test_reliability_policy.py`, `tests/test_source_assessor.py`, `tests/test_judgement.py`, `tests/test_verifier.py`, `tests/test_app_render.py` (imports `app` and tests one pure helper).

### Field names this plan adds to a claim dict (Task 2) — later tasks rely on them

`resolved: bool`, `resolution: str`, `role: str`, `claim_quality: list[str]`, `context: str`, `section: str`, `cite_count: int`, `sentence_index: int`; `ref` gains `venue: str | None`, `is_monograph: bool`. Task 7 adds to judged items `evidence_sections: list[str]`, `evidence_pages: list[int]`, `escalated: bool`, `evidence_hits: int`.

### Pitfalls

1. `seed_audit.py` imports `_judge_once` at module level; tests patch `research_assistant.shared.seed_audit._judge_once`. Keep that import where it is.
2. `_clean_claim_punctuation` is imported by `tests/test_seed_audit.py`; Task 1 makes it an alias of `claim_text.tidy_punctuation` — do not delete the name.
3. The `"xml"` BeautifulSoup parser keeps namespaces out of tag names: `p.find_all("s")` works on GROBID TEI.
4. `SAMPLE_TEI_XML` has no `<s>` tags. The regex fallback must keep working or half the existing tests fail.

---

### Task 1: `shared/claim_text.py` — sentences, tokens, claim rendering, quality, role

**Files:**
- Create: `research_assistant/shared/claim_text.py`
- Test: `tests/test_claim_text.py`

**Interfaces:**
- Consumes: `research_assistant.agents.agent5_batch_citer.split_into_sentences`.
- Produces:
  ```python
  CITE_TOKEN_RE: re.Pattern            # matches "⟦C7⟧", group(1) == "7"
  CLAIM_MIN_WORDS = 5; CLAIM_MAX_WORDS = 80
  SKIP_ROLES = frozenset({"software", "pointer", "method"})
  def cite_token(n: int) -> str                            # "⟦C7⟧"
  def clean_text(node) -> str                              # bs4 node or str → whitespace-collapsed
  def paragraph_sentences(p) -> list[str]                  # <s> children if any, else regex split
  def is_numeric_cite(txt: str) -> bool
  def is_parenthetical(sentence: str, token: str) -> bool
  def tidy_punctuation(text: str) -> str
  def render_claim(sentence: str, cites: dict[int, dict]) -> str     # cites[n]["txt"]
  def render_sentence(sentence: str, cites: dict[int, dict]) -> str
  def claim_quality(claim: str) -> list[str]               # ⊆ {"placeholder_residue","fragment","too_long"}
  def classify_citation_role(sentence: str, token: str, ref_info: dict | None) -> str
  def sentence_context(sentences: list[str], idx: int) -> str
  ```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_claim_text.py
"""The deterministic text work that turns a GROBID paragraph into claims.
Every case is a literal paragraph; no model, no index."""

import unittest

from bs4 import BeautifulSoup

from research_assistant.shared import claim_text as ct


def _p(xml: str):
    return BeautifulSoup(f'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>{xml}</body></text></TEI>', "xml").find("p")


class TestParagraphSentences(unittest.TestCase):
    def test_grobid_sentence_tags_are_used_when_present(self):
        p = _p("<p><s>First sentence here.</s><s>Second sentence here.</s></p>")
        self.assertEqual(ct.paragraph_sentences(p), ["First sentence here.", "Second sentence here."])

    def test_glued_boundary_is_not_a_problem_with_sentence_tags(self):
        # get_text() would give "here.Second" — the tags are the boundary, not the whitespace.
        p = _p("<p><s>Ends here.</s><s>Second one.</s></p>")
        self.assertEqual(len(ct.paragraph_sentences(p)), 2)

    def test_regex_fallback_without_sentence_tags(self):
        p = _p("<p>First sentence here. Second sentence here.</p>")
        self.assertEqual(ct.paragraph_sentences(p), ["First sentence here.", "Second sentence here."])

    def test_tokens_inside_sentence_tags_survive(self):
        p = _p("<p><s>Shown by <ref type=\"bibr\">Landa et al. (2006)</ref> here.</s></p>")
        p.find("ref").replace_with(f" {ct.cite_token(0)} ")
        self.assertEqual(ct.paragraph_sentences(p), ["Shown by ⟦C0⟧ here."])


class TestNumericAndParenthetical(unittest.TestCase):
    def test_numeric_markers(self):
        for txt in ("[12]", "[9,", "10]", "3-5", "[13–20]", "7"):
            self.assertTrue(ct.is_numeric_cite(txt), txt)

    def test_author_year_is_not_numeric(self):
        for txt in ("Landa et al. (2006)", "Silva et al. 2021", "C.A.N. da Costa et al. 2018"):
            self.assertFalse(ct.is_numeric_cite(txt), txt)

    def test_parenthetical_detection(self):
        s = "Proposed by several groups (⟦C0⟧; ⟦C1⟧) and later by ⟦C2⟧ here."
        self.assertTrue(ct.is_parenthetical(s, "⟦C0⟧"))
        self.assertTrue(ct.is_parenthetical(s, "⟦C1⟧"))
        self.assertFalse(ct.is_parenthetical(s, "⟦C2⟧"))


class TestRenderClaim(unittest.TestCase):
    def test_narrative_citation_keeps_the_sentence_subject(self):
        cites = {0: {"txt": "Landa et al. (2006)"}}
        s = "A similar idea was explored by ⟦C0⟧ with a path-integral formulation."
        self.assertEqual(ct.render_claim(s, cites),
                         "A similar idea was explored by Landa et al. (2006) with a path-integral formulation.")

    def test_leading_narrative_citation_is_kept(self):
        cites = {0: {"txt": "Silva et al. (2021)"}}
        s = "Following ⟦C0⟧ , the computational time is described next."
        self.assertEqual(ct.render_claim(s, cites),
                         "Following Silva et al. (2021), the computational time is described next.")

    def test_parenthetical_cluster_is_removed_cleanly(self):
        cites = {0: {"txt": "Vasconcelos et al. 2017"}, 1: {"txt": "Cui et al. 2020"}}
        s = "Target-oriented methods have been proposed by several groups ( ⟦C0⟧ ; ⟦C1⟧ )."
        self.assertEqual(ct.render_claim(s, cites),
                         "Target-oriented methods have been proposed by several groups.")

    def test_numeric_citations_are_removed(self):
        cites = {0: {"txt": "[9,"}, 1: {"txt": "10]"}}
        s = "The Landauer conductance reads G = 2e2/h ⟦C0⟧ ⟦C1⟧ ."
        self.assertEqual(ct.render_claim(s, cites), "The Landauer conductance reads G = 2e2/h.")

    def test_display_sentence_brackets_the_citation(self):
        cites = {0: {"txt": "Landa et al. (2006)"}}
        self.assertEqual(ct.render_sentence("By ⟦C0⟧ here.", cites), "By [Landa et al. (2006)] here.")

    def test_tidy_punctuation_existing_behaviour(self):
        self.assertEqual(ct.tidy_punctuation("Experiments confirmed this [ ] ."), "Experiments confirmed this.")
        self.assertEqual(ct.tidy_punctuation("It was shown ( ) , that conductance varies ."),
                         "It was shown, that conductance varies.")
        self.assertEqual(ct.tidy_punctuation("groups (;; ) here"), "groups here")


class TestClaimQuality(unittest.TestCase):
    def test_good_claim_has_no_issues(self):
        self.assertEqual(ct.claim_quality("Graphene shows a linear dispersion near the Dirac point."), [])

    def test_placeholder_residue(self):
        self.assertIn("placeholder_residue", ct.claim_quality("Shown in ⟦C3⟧ and elsewhere in the paper."))
        self.assertIn("placeholder_residue", ct.claim_quality("Shown in __CITE_1_b6_C.A.N. elsewhere too."))

    def test_fragments(self):
        self.assertIn("fragment", ct.claim_quality("with a path-integral formulation of depth migration."))
        self.assertIn("fragment", ct.claim_quality(", the computational time for each method is described next:"))
        self.assertIn("fragment", ct.claim_quality("See SM for M."))

    def test_too_long(self):
        self.assertIn("too_long", ct.claim_quality("word " * 81))
        self.assertNotIn("too_long", ct.claim_quality("word " * 80))


class TestCitationRole(unittest.TestCase):
    def test_software_by_reference_title(self):
        ref = {"title": "LAPACK Users' Guide, 3rd edn"}
        self.assertEqual(ct.classify_citation_role("We call ⟦C0⟧ for dense operations.", "⟦C0⟧", ref), "software")
        ref = {"title": "Algorithm 832"}
        self.assertEqual(ct.classify_citation_role("We use ⟦C0⟧ for LU factorization.", "⟦C0⟧", ref), "software")

    def test_software_by_context(self):
        ref = {"title": "Julia: a fresh approach to numerical computing"}
        self.assertEqual(ct.classify_citation_role("The code is implemented in ⟦C0⟧ and run on CPUs.", "⟦C0⟧", ref), "software")

    def test_pointer(self):
        s = "Multi-terminal experiments (see, e.g, Ref. ⟦C0⟧ for an introduction) have played a role."
        self.assertEqual(ct.classify_citation_role(s, "⟦C0⟧", {"title": "Semiconductor Nanostructures"}), "pointer")
        self.assertEqual(ct.classify_citation_role("See ⟦C0⟧ for a review of the field.", "⟦C0⟧", {}), "pointer")

    def test_method(self):
        s = "Following ⟦C0⟧ , we also apply a taper to the target gradient."
        self.assertEqual(ct.classify_citation_role(s, "⟦C0⟧", {"title": "Target-oriented inversion"}), "method")

    def test_evidential(self):
        s = "Büttiker ⟦C0⟧ has shown that R and V can be cast in terms of the conductance matrix."
        self.assertEqual(ct.classify_citation_role(s, "⟦C0⟧", {"title": "Four-terminal phase-coherent conductance"}), "evidential")
        s = "Multi-terminal experiments have played a key role in graphene ⟦C0⟧ ⟦C1⟧ ."
        self.assertEqual(ct.classify_citation_role(s, "⟦C1⟧", {"title": "Nanoscale direct mapping of noise"}), "evidential")


class TestSentenceContext(unittest.TestCase):
    def test_marks_claim_between_neighbours(self):
        s = ["One.", "Two.", "Three."]
        self.assertEqual(ct.sentence_context(s, 1), "One. «Two.» Three.")
        self.assertEqual(ct.sentence_context(s, 0), "«One.» Two.")
        self.assertEqual(ct.sentence_context(s, 2), "Two. «Three.»")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_claim_text.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'research_assistant.shared.claim_text'`.

- [ ] **Step 3: Write the module**

```python
# research_assistant/shared/claim_text.py
"""Turning a GROBID paragraph into judgeable claims.

Everything here is deterministic string work: which sentences a paragraph
has, where its citations sit, what the sentence says once the citation
marker is dealt with, whether the result is a claim at all, and what kind
of citation it is. None of it touches a model, so all of it is tested
against literal paragraphs.

Why tokens: the old placeholder embedded the citation's own text
("__CITE_1_b6_C.A.N. da Costa et al. 2018__"), so the sentence splitter cut
it at "C. " and half of it leaked into reports verbatim. "⟦C7⟧" contains
nothing any splitter treats as a boundary.

Why narrative citations are kept: "explored by Landa et al. (2006) with…"
loses its subject when the citation is deleted. A numeric marker or a
citation inside parentheses is punctuation and goes; an author–year
citation that is part of the sentence's grammar stays as text.
"""

import re

from research_assistant.agents.agent5_batch_citer import split_into_sentences

CITE_TOKEN_RE = re.compile(r"⟦C(\d+)⟧")

CLAIM_MIN_WORDS = 5
CLAIM_MAX_WORDS = 80

# Roles that are not claims about the cited paper's findings and are not judged.
SKIP_ROLES = frozenset({"software", "pointer", "method"})


def cite_token(n: int) -> str:
    return f"⟦C{n}⟧"


def clean_text(node) -> str:
    """Whitespace-collapsed text of a bs4 node or a string."""
    if node is None:
        return ""
    text = node.get_text() if hasattr(node, "get_text") else str(node)
    return " ".join(text.split()).strip()


def paragraph_sentences(p) -> list[str]:
    """GROBID's own <s> segmentation when the paragraph has it; the regex
    splitter otherwise. get_text() on a <p> glues "</s><s>" with no space,
    which is why the regex must never be the first choice."""
    s_tags = p.find_all("s")
    if s_tags:
        return [t for t in (clean_text(s) for s in s_tags) if t]
    return [t for t in split_into_sentences(clean_text(p)) if t]


_NUMERIC_CITE_RE = re.compile(r"^[\[\(]?\s*\d+[\d,;\s\-–—\]\)]*$")


def is_numeric_cite(txt: str) -> bool:
    """"[12]", "[9,", "10]", "3-5": a marker that carries no words."""
    return bool(_NUMERIC_CITE_RE.match((txt or "").strip()))


def is_parenthetical(sentence: str, token: str) -> bool:
    """Is *token* inside an unclosed ( or [ in *sentence*?"""
    before = sentence.split(token, 1)[0]
    depth = (before.count("(") - before.count(")")) + (before.count("[") - before.count("]"))
    return depth > 0


def tidy_punctuation(text: str) -> str:
    """Remove what a deleted citation leaves behind: "( ; )", "[ ]", doubled
    separators, a space before punctuation."""
    text = re.sub(r"[\(\[]\s*[;,\s]*[\)\]]", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(\s*[,;:])+", r"\1", text)
    return " ".join(text.split()).strip()


def render_claim(sentence: str, cites: dict[int, dict]) -> str:
    """The sentence as the judge should read it: numeric and parenthetical
    citations removed, narrative citations kept as their own text."""
    def _sub(match):
        n = int(match.group(1))
        txt = (cites.get(n) or {}).get("txt", "") or ""
        if is_numeric_cite(txt) or txt.startswith("(") or is_parenthetical(sentence, match.group(0)):
            return ""
        return txt
    return tidy_punctuation(CITE_TOKEN_RE.sub(_sub, sentence))


def render_sentence(sentence: str, cites: dict[int, dict]) -> str:
    """The sentence as the reader should see it: every citation in brackets."""
    def _sub(match):
        n = int(match.group(1))
        txt = (cites.get(n) or {}).get("txt", "") or ""
        return f"[{txt.strip('[]')}]" if txt else ""
    return " ".join(CITE_TOKEN_RE.sub(_sub, sentence).split())


_FRAGMENT_START_RE = re.compile(r"^[a-z\)\],;:]")


def claim_quality(claim: str) -> list[str]:
    """What is wrong with *claim* as a thing to judge. "placeholder_residue"
    and "fragment" mean it is not judged; "too_long" is judged and flagged."""
    issues = []
    if CITE_TOKEN_RE.search(claim) or "__CITE_" in claim:
        issues.append("placeholder_residue")
    words = claim.split()
    stripped = claim.lstrip()
    if len(words) < CLAIM_MIN_WORDS or _FRAGMENT_START_RE.match(stripped):
        issues.append("fragment")
    if len(words) > CLAIM_MAX_WORDS:
        issues.append("too_long")
    return issues


_SOFTWARE_REF_RE = re.compile(
    r"(users?'?\s+guide|user\s+manual|\bmanual\b|\bsoftware\b|\bpackage\b|\blibrary\b|"
    r"\btoolkit\b|\btoolbox\b|\balgorithm\s+\d+|\bversion\s+\d|\bv\d+\.\d|\bgithub\b|"
    r"\bzenodo\b|\bdocumentation\b)",
    re.I,
)
_SOFTWARE_CONTEXT_RE = re.compile(
    r"\b(package|code|software|library|toolkit|toolbox|implementation|solver|routine|"
    r"implemented (?:in|with|using)|written in|computed (?:with|using)|performed (?:with|using)|"
    r"calculated (?:with|using)|simulated (?:with|using))\b",
    re.I,
)
_POINTER_RE = re.compile(
    r"(\bsee\b|\bcf\.|\be\.g\.?,?|\bfor (?:a |an )?(?:review|details|introduction|overview|"
    r"discussion|derivation|proof|survey)\b|\bas (?:described|discussed|explained|reviewed|shown|"
    r"detailed|derived|outlined|summari[sz]ed) in\b|\breviewed in\b|\band references therein\b|"
    r"\bfor (?:instance|example)\b|\brefs?\.?\s*$)",
    re.I,
)
_METHOD_RE = re.compile(
    r"\b(following|according to|as in|adapted from|adopted from|based on|"
    r"we (?:use|used|adopt|adopted|follow|followed|employ|employed|apply|applied))\b",
    re.I,
)


def classify_citation_role(sentence: str, token: str, ref_info: dict | None) -> str:
    """'software' | 'pointer' | 'method' | 'evidential', from the eight words
    before the token and the reference's title. Heuristic and deliberately
    conservative: an unrecognised citation is evidential and gets judged."""
    before = sentence.split(token, 1)[0]
    window = " ".join(before.split()[-8:])
    ref = ref_info or {}
    ref_text = f"{ref.get('title') or ''} {ref.get('raw_reference') or ''}"
    if _SOFTWARE_REF_RE.search(ref_text) or _SOFTWARE_CONTEXT_RE.search(window):
        return "software"
    if _POINTER_RE.search(window):
        return "pointer"
    if _METHOD_RE.search(window):
        return "method"
    return "evidential"


def sentence_context(sentences: list[str], idx: int) -> str:
    """Previous + «claim» + next, for the judge's context block."""
    parts = []
    if idx > 0:
        parts.append(sentences[idx - 1])
    parts.append(f"«{sentences[idx]}»")
    if idx + 1 < len(sentences):
        parts.append(sentences[idx + 1])
    return " ".join(parts)
```

- [ ] **Step 4: Run the tests**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_claim_text.py -q`
Expected: all pass. If `test_pointer`'s second case fails on "See ⟦C0⟧", the window is `"See"` and `\bsee\b` matches case-insensitively — check that `re.I` is on `_POINTER_RE`.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/claim_text.py tests/test_claim_text.py
git commit -m "feat(audit): claim_text — GROBID sentences, citation tokens, narrative-aware claims, quality and role"
```

---

### Task 2: Extraction on `claim_text` with number-map reference resolution

**Files:**
- Modify: `research_assistant/shared/seed_audit.py` — imports (top), `_clean_claim_punctuation` (≈line 55), bibliography parse and paragraph loop inside `extract_seed_citation_claims` (≈lines 91–304)
- Test: `tests/test_seed_audit.py` (append a class)

**Interfaces:**
- Consumes: everything in Task 1; `extract.normalise_section_kind`.
- Produces: each claim dict now carries `resolved, resolution, role, claim_quality, context, section, cite_count, sentence_index` in addition to the existing keys; `ref` carries `venue` and `is_monograph`. Module-level helpers `_paragraph_section(p)`, `_build_number_map(body, bib_by_id)`, `_position_map_consistent(number_map)`, `_resolve_ref(target, txt, bib_by_id, bib_by_index, number_map, position_ok) -> tuple[dict | None, str]`, `_unknown_ref(target, txt) -> dict`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_seed_audit.py`, before `if __name__ == "__main__":`)

```python
AUTHOR_YEAR_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><head>Introduction</head>
<p><s>Target-oriented methods have been proposed by several groups (<ref type="bibr" target="#b27">Vasconcelos et al. 2017</ref>; <ref type="bibr" target="#b6">C.A.N. da Costa et al. 2018</ref>).</s><s>A similar idea was explored by <ref type="bibr" target="#b16">Landa et al. (2006)</ref> with a path-integral formulation of depth migration.</s></p>
</div>
<div><head>Results</head>
<p><s>Following <ref type="bibr" target="#b25">Silva et al. (2021)</ref>, the computational time for each method is described next.</s><s>The PGF time grows with M, while the RPGF time decreases.</s></p>
<p><s>We use UMFPACK (<ref type="bibr" target="#b9">Davis 2004</ref>) for LU factorization of sparse matrices.</s></p>
</div></body>
<back><listBibl>
<biblStruct xml:id="b6"><analytic><title level="a" type="main">Target-level waveform inversion</title><author><persName><forename>C</forename><surname>Costa</surname></persName></author></analytic><monogr><title level="j">Geophysical Prospecting</title><imprint><date when="2018">2018</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b9"><analytic><title level="a" type="main">Algorithm 832</title><author><persName><forename>T</forename><surname>Davis</surname></persName></author></analytic><monogr><title level="j">ACM Trans. Math. Softw.</title><imprint><date when="2004">2004</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b16"><analytic><title level="a" type="main">Path-integral seismic imaging</title><author><persName><forename>E</forename><surname>Landa</surname></persName></author></analytic><monogr><title level="j">Geophysical Prospecting</title><imprint><date when="2006">2006</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b25"><analytic><title level="a" type="main">Target-oriented inversion using the patched green's function method</title><author><persName><forename>D</forename><surname>Silva</surname></persName></author></analytic><monogr><title level="j">Geophysics</title><imprint><date when="2021">2021</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b27"><analytic><title level="a" type="main">Subsurface-domain objective functions</title><author><persName><forename>I</forename><surname>Vasconcelos</surname></persName></author></analytic><monogr><title level="j">Geophysics</title><imprint><date when="2017">2017</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b40"><monogr><title level="m">Semiconductor Nanostructures</title><author><persName><forename>T</forename><surname>Ihn</surname></persName></author><imprint><date when="2010">2010</date></imprint></monogr></biblStruct>
</listBibl></back></text></TEI>
"""

# A footnote GROBID swept into the bibliography: b20 is not reference [21].
# GROBID linked [21] to b53 elsewhere; a targetless [21] must follow that, and
# a targetless number nobody linked must not be guessed from list position.
FOOTNOTE_BIB_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<p><s>We compute everything with the kernel polynomial method <ref type="bibr" target="#b53">[21]</ref> on large flakes.</s></p>
<p><s>The same method was used for the disordered case <ref type="bibr">[21]</ref> and the clean case <ref type="bibr">[22]</ref> respectively.</s></p>
</body>
<back><listBibl>
<biblStruct xml:id="b19"><analytic><title level="a" type="main">Machine learning phases of matter</title></analytic></biblStruct>
<biblStruct xml:id="b20"><note type="raw_reference">An alternative definition of β(E), shown in the SM [21], may extend the validity</note></biblStruct>
<biblStruct xml:id="b21"><analytic><title level="a" type="main">Z2pack</title></analytic></biblStruct>
<biblStruct xml:id="b53"><analytic><title level="a" type="main">KITE: high-performance accurate modelling of electronic structure</title></analytic></biblStruct>
</listBibl></back></text></TEI>
"""


class TestExtractionOnClaimText(unittest.TestCase):
    def _claims(self, xml):
        return extract_seed_citation_claims(BeautifulSoup(xml, "xml"))

    def test_sentence_tags_give_one_claim_per_sentence(self):
        claims = self._claims(AUTHOR_YEAR_TEI_XML)
        by_claim = {c["claim"] for c in claims}
        self.assertIn("Target-oriented methods have been proposed by several groups.", by_claim)
        self.assertIn("A similar idea was explored by Landa et al. (2006) with a path-integral formulation of depth migration.", by_claim)
        self.assertIn("Following Silva et al. (2021), the computational time for each method is described next.", by_claim)

    def test_no_placeholder_ever_leaks(self):
        for c in self._claims(AUTHOR_YEAR_TEI_XML):
            self.assertNotIn("__CITE_", c["claim"]); self.assertNotIn("⟦", c["claim"])
            self.assertNotIn("__CITE_", c["sentence"]); self.assertNotIn("⟦", c["sentence"])
            self.assertEqual(c["claim_quality"], [], c["claim"])

    def test_display_sentence_and_context(self):
        landa = next(c for c in self._claims(AUTHOR_YEAR_TEI_XML) if c["ref"]["xml_id"] == "b16")
        self.assertEqual(landa["sentence"], "A similar idea was explored by [Landa et al. (2006)] with a path-integral formulation of depth migration.")
        self.assertTrue(landa["context"].startswith("Target-oriented methods have been proposed by several groups ( [Vasconcelos"))
        self.assertIn("«A similar idea", landa["context"])
        self.assertEqual(landa["sentence_index"], 1)
        self.assertEqual(landa["cite_count"], 1)

    def test_section_and_role(self):
        claims = self._claims(AUTHOR_YEAR_TEI_XML)
        landa = next(c for c in claims if c["ref"]["xml_id"] == "b16")
        silva = next(c for c in claims if c["ref"]["xml_id"] == "b25")
        davis = next(c for c in claims if c["ref"]["xml_id"] == "b9")
        self.assertEqual(landa["section"], "introduction"); self.assertEqual(landa["role"], "evidential")
        self.assertEqual(silva["section"], "results"); self.assertEqual(silva["role"], "method")
        self.assertEqual(davis["role"], "software")

    def test_paragraph_refs_count_only_evidential_resolved_references(self):
        davis = next(c for c in self._claims(AUTHOR_YEAR_TEI_XML) if c["ref"]["xml_id"] == "b9")
        self.assertEqual(davis["paragraph_refs"], [])

    def test_venue_and_monograph_on_ref(self):
        claims = self._claims(AUTHOR_YEAR_TEI_XML)
        landa = next(c for c in claims if c["ref"]["xml_id"] == "b16")
        self.assertEqual(landa["ref"]["venue"], "Geophysical Prospecting")
        self.assertFalse(landa["ref"]["is_monograph"])

    def test_targetless_number_follows_what_grobid_linked_elsewhere(self):
        claims = self._claims(FOOTNOTE_BIB_TEI_XML)
        second = [c for c in claims if c["sentence"].startswith("The same method")]
        twenty_one = next(c for c in second if c["cite_text"] == "[21]")
        self.assertEqual(twenty_one["ref"]["xml_id"], "b53")
        self.assertEqual(twenty_one["resolution"], "number_map")
        self.assertTrue(twenty_one["resolved"])

    def test_unlinked_number_is_not_guessed_when_positions_are_inconsistent(self):
        claims = self._claims(FOOTNOTE_BIB_TEI_XML)
        twenty_two = next(c for c in claims if c["cite_text"] == "[22]")
        self.assertFalse(twenty_two["resolved"])
        self.assertEqual(twenty_two["resolution"], "unresolved")

    def test_position_fallback_still_works_when_consistent(self):
        # SAMPLE_TEI_XML: [1]→b0 is linked, so [2] at position 2 is safe.
        claims = extract_seed_citation_claims(BeautifulSoup(SAMPLE_TEI_XML, "xml"))
        self.assertEqual(claims[1]["resolution"], "position")
        self.assertTrue(claims[1]["resolved"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_seed_audit.py -q -k "ExtractionOnClaimText"`
Expected: FAIL — `KeyError: 'claim_quality'` / `'resolution'` and placeholder assertions.

- [ ] **Step 3: Rewire the imports and `_clean_claim_punctuation`**

At the top of `seed_audit.py`, after `from research_assistant.agents.agent5_batch_citer import split_into_sentences`, add:

```python
from research_assistant.shared.claim_text import (
    CITE_TOKEN_RE,
    SKIP_ROLES,
    cite_token,
    claim_quality,
    classify_citation_role,
    clean_text,
    is_numeric_cite,
    paragraph_sentences,
    render_claim,
    render_sentence,
    sentence_context,
    tidy_punctuation,
)
from research_assistant.shared.extract import normalise_section_kind
```

Replace the body of `_clean_claim_punctuation` so it is an alias:

```python
def _clean_claim_punctuation(text: str) -> str:
    """Tidy punctuation spaces left behind by stripped citation markers."""
    return tidy_punctuation(text)
```

- [ ] **Step 4: Add the resolution helpers** (module level, directly above `extract_seed_citation_claims`)

```python
_NUM_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _paragraph_section(p) -> str:
    """Section kind of the <div> the paragraph sits in; GROBID often leaves
    the introduction headless, which lands as "other"."""
    div = p.find_parent("div")
    head = div.find("head") if div is not None else None
    return normalise_section_kind(clean_text(head)) if head is not None else "other"


def _build_number_map(body, bib_by_id: dict) -> dict[int, dict]:
    """What GROBID itself linked: every targeted ref with a numeric text
    teaches "[k] means this entry". Consulted for the refs it left untargeted."""
    number_map: dict[int, dict] = {}
    for ref in body.find_all("ref", type="bibr"):
        target = (ref.get("target") or "").lstrip("#")
        txt = clean_text(ref)
        if target in bib_by_id and is_numeric_cite(txt):
            nums = _NUM_RE.findall(txt)
            if len(nums) == 1:
                number_map.setdefault(int(nums[0]), bib_by_id[target])
    return number_map


def _position_map_consistent(number_map: dict[int, dict]) -> bool:
    """True when every number GROBID linked sits at its own list position —
    the only condition under which "[k] is the k-th entry" is safe. A
    footnote swept into the bibliography breaks it for everything after."""
    return all(info.get("xml_id") == f"b{k - 1}" for k, info in number_map.items())


def _unknown_ref(target: str, txt: str) -> dict:
    return {
        "xml_id": target or f"unknown_{txt}",
        "index": None,
        "title": txt or "Unknown reference",
        "authors": [],
        "year": None,
        "doi": None,
        "raw_reference": txt,
        "venue": None,
        "is_monograph": False,
    }


def _resolve_ref(target, txt, bib_by_id, bib_by_index, number_map, position_ok) -> tuple[Optional[dict], str]:
    """(reference, how). Never guesses: a numeric marker nobody linked is
    resolved by position only when positions are known to be consistent, and
    a surname match needs the whole word and, when both sides have one, the
    year."""
    if target in bib_by_id:
        return bib_by_id[target], "target"
    if is_numeric_cite(txt):
        nums = _NUM_RE.findall(txt)
        if nums:
            k = int(nums[0])
            if k in number_map:
                return number_map[k], "number_map"
            if position_ok and k in bib_by_index:
                return bib_by_index[k], "position"
        return None, "unresolved"
    year = _YEAR_RE.search(txt or "")
    for info in bib_by_id.values():
        surnames = [a.split()[-1] for a in info.get("authors", []) if a.split()]
        if any(len(s) > 2 and re.search(rf"\b{re.escape(s)}\b", txt) for s in surnames):
            if year is None or info.get("year") is None or int(year.group(1)) == info["year"]:
                return info, "surname"
    return None, "unresolved"
```

- [ ] **Step 5: Capture venue / monograph in the bibliography parse**

Inside `extract_seed_citation_claims`, in the `for idx, b in enumerate(list_bibl.find_all("biblStruct")):` loop, after `raw_reference = …` and before `ref_info = {`, add:

```python
            journal = monogr.find("title", level="j") if monogr else None
            book = monogr.find("title", level="m") if monogr else None
            venue = _clean(journal) if journal else (_clean(book) if book else None)
            is_monograph = bool(book) and not (analytic and analytic.find("title"))
```

and add two keys to `ref_info`:

```python
                "venue": venue,
                "is_monograph": is_monograph,
```

- [ ] **Step 6: Replace the paragraph loop**

Replace everything from `body = soup.find("body")` to the end of the function with:

```python
    body = soup.find("body")
    if not body:
        return []

    number_map = _build_number_map(body, bib_by_id)
    position_ok = _position_map_consistent(number_map)

    claims = []
    seen_pairs = set()

    for p_idx, p in enumerate(body.find_all("p")):
        refs = p.find_all("ref", type="bibr")
        if not refs:
            continue

        p_id = p.get("xml:id") or p.get("id") or f"p_{p_idx}"
        section = _paragraph_section(p)

        # Citations become period-free tokens; what the old placeholder
        # embedded lives in this table instead.
        cites = {}
        for n, ref in enumerate(refs):
            target = (ref.get("target") or "").lstrip("#")
            txt = clean_text(ref)
            ref_info, resolution = _resolve_ref(
                target, txt, bib_by_id, bib_by_index, number_map, position_ok
            )
            cites[n] = {"target": target, "txt": txt, "ref": ref_info, "resolution": resolution}
            ref.replace_with(f" {cite_token(n)} ")

        sentences = paragraph_sentences(p)
        display = [render_sentence(s, cites) for s in sentences]

        # The deferral ratio counts the references a paragraph leans on for
        # evidence — not the software it used or the review it points to.
        p_refs_dict = {}
        paragraph_claims = []
        for s_idx, sent in enumerate(sentences):
            tokens = [int(m) for m in CITE_TOKEN_RE.findall(sent)]
            if not tokens:
                continue

            claim_text = render_claim(sent, cites)
            if len(claim_text) < 15:
                continue
            quality = claim_quality(claim_text)
            context = sentence_context(display, s_idx)

            for n in tokens:
                cite = cites[n]
                resolved = cite["ref"] is not None
                ref_info = cite["ref"] or _unknown_ref(cite["target"], cite["txt"])
                role = classify_citation_role(sent, cite_token(n), ref_info)

                ref_key = ref_info.get("xml_id") or cite["target"] or cite["txt"]
                dedup_key = (claim_text, ref_key)
                if dedup_key in seen_pairs:
                    continue
                seen_pairs.add(dedup_key)

                if resolved and role == "evidential":
                    p_refs_dict.setdefault(ref_key, ref_info)

                paragraph_claims.append({
                    "sentence": display[s_idx],
                    "claim": claim_text,
                    "cite_text": cite["txt"],
                    "target": cite["target"],
                    "ref": ref_info,
                    "resolved": resolved,
                    "resolution": cite["resolution"],
                    "role": role,
                    "claim_quality": quality,
                    "context": context,
                    "section": section,
                    "cite_count": len(tokens),
                    "sentence_index": s_idx,
                    "paragraph_id": p_id,
                    "paragraph_index": p_idx,
                })

        p_unique_refs = list(p_refs_dict.values())
        for c in paragraph_claims:
            c["paragraph_refs"] = p_unique_refs
        claims.extend(paragraph_claims)

    return claims
```

- [ ] **Step 7: Run the seed-audit tests**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_seed_audit.py tests/test_claim_text.py -q`
Expected: all pass, including the pre-existing `test_extract_seed_citation_claims_numeric_fallback` (SAMPLE_TEI's `[2]` resolves by position because `[1]→b0` is consistent). If `test_paragraph_refs_count_only_evidential_resolved_references` fails because `paragraph_refs` is non-empty, the Davis sentence's role came back `evidential` — check that `_SOFTWARE_REF_RE` matches `"Algorithm 832"` (it needs `\balgorithm\s+\d+`).

- [ ] **Step 8: Commit**

```bash
git add research_assistant/shared/seed_audit.py tests/test_seed_audit.py
git commit -m "feat(audit): extract claims from GROBID sentences with citation tokens; resolve refs by number map, never by guess"
```

---

### Task 3: Reliability policy knows the outcome; `Does not support` is UNSUPPORTED

**Files:**
- Modify: `research_assistant/judgement/policy.py`
- Test: `tests/test_reliability_policy.py`

**Interfaces:**
- Produces:
  ```python
  class ReliabilityRating(str, Enum): HIGH, MODERATE, LOW, CONTRADICTED, UNSUPPORTED, UNRESOLVED
  NOT_ASSESSED_EXPLANATIONS: dict[str, str]     # keyed by outcome
  def evaluate_reliability(relation, source_grade=None, confidence=None, span_verified=None,
                           rubric_violations=None, rubric_mismatch=False, outcome="judged") -> dict
  ```

- [ ] **Step 1: Write the failing tests** (replace `test_does_not_support_is_unresolved` and add the rest)

```python
    def test_does_not_support_is_unsupported(self):
        res = evaluate_reliability(
            relation="Does not support",
            source_grade=SourceGrade.STANDARD_PRIMARY.value,
        )
        self.assertEqual(res["rating"], ReliabilityRating.UNSUPPORTED.value)
        self.assertIn("does not report", res["explanation"])

    def test_judged_unclear_is_unresolved_with_judge_explanation(self):
        res = evaluate_reliability(relation="Unclear / insufficient evidence",
                                   source_grade=SourceGrade.STANDARD_PRIMARY.value)
        self.assertEqual(res["rating"], ReliabilityRating.UNRESOLVED.value)
        self.assertIn("could not decide", res["explanation"])

    def test_not_assessed_outcomes_are_unresolved_with_their_own_reason(self):
        for outcome, phrase in (
            ("cap_exceeded", "budget"),
            ("not_attempted", "backend"),
            ("retrieval_failed", "retrieval"),
            ("call_failed", "backend"),
            ("no_evidence", "no passages"),
            ("cluster_skipped", "cites"),
            ("not_a_claim", "not a verifiable claim"),
            ("malformed_claim", "sentence"),
            ("unresolved_ref", "bibliography"),
            ("not_downloaded", "not in the corpus"),
            ("deferred_paywalled", "deferred"),
        ):
            res = evaluate_reliability(relation=None, source_grade=None, outcome=outcome)
            self.assertEqual(res["rating"], ReliabilityRating.UNRESOLVED.value, outcome)
            self.assertTrue(res["explanation"].startswith("Not assessed"), outcome)
            self.assertIn(phrase, res["explanation"], outcome)

    def test_missing_confidence_does_not_downgrade(self):
        res = evaluate_reliability(relation="Supports", source_grade=SourceGrade.STANDARD_PRIMARY.value,
                                   confidence=None, span_verified=True)
        self.assertEqual(res["rating"], ReliabilityRating.HIGH.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_reliability_policy.py -q`
Expected: FAIL — `AttributeError: UNSUPPORTED`, `TypeError: unexpected keyword 'outcome'`.

- [ ] **Step 3: Implement**

In `policy.py`, add `UNSUPPORTED = "UNSUPPORTED"` to `ReliabilityRating` between `CONTRADICTED` and `UNRESOLVED`. Add after the enum:

```python
# One sentence per way a citation can end up not judged. The rating for all of
# them is UNRESOLVED; the explanation is what distinguishes "the paper was
# silent" from "we never looked".
NOT_ASSESSED_EXPLANATIONS = {
    "cap_exceeded": "Not assessed — the per-paper claim budget was reached before this citation.",
    "not_attempted": "Not assessed — the audit stopped early because the model backend was unreachable.",
    "retrieval_failed": "Not assessed — retrieval from the cited paper failed.",
    "call_failed": "Not assessed — the model backend returned an error.",
    "parse_failed": "Not assessed — the model reply could not be parsed.",
    "no_evidence": "Not assessed — search found no passages in the cited paper for this sentence.",
    "cluster_skipped": "Not assessed — the sentence cites many papers; only the first few were judged.",
    "not_a_claim": "Not assessed — this citation is not a verifiable claim about the cited paper (software, pointer, or method reference).",
    "malformed_claim": "Not assessed — the extracted sentence is a fragment and could not be judged.",
    "unresolved_ref": "Not assessed — the citation could not be matched to a bibliography entry.",
    "not_downloaded": "Not assessed — the cited paper is not in the corpus.",
    "deferred_paywalled": "Not assessed — deferred because most references in this paragraph are missing.",
}
```

Change the signature and the first branch of `evaluate_reliability`:

```python
def evaluate_reliability(
    relation: str | None,
    source_grade: str | None = None,
    confidence: str | None = None,
    span_verified: bool | None = None,
    rubric_violations: list[str] | None = None,
    rubric_mismatch: bool = False,
    outcome: str = "judged",
) -> dict[str, Any]:
    """Derive scientific reliability from the relation verdict and source grade.

    A rating is a statement about a verdict. When there is no verdict
    (*outcome* != "judged") the rating is UNRESOLVED and the explanation says
    why nothing was judged — never that the evidence was insufficient.
    """
    source_grade = source_grade or SourceGrade.UNKNOWN.value
    conf = (confidence or "High").capitalize()
    has_violations = bool(rubric_violations) or rubric_mismatch

    # 0. Nothing was judged.
    if outcome != "judged" or not relation:
        return {
            "rating": ReliabilityRating.UNRESOLVED.value,
            "badge": "⏳ Not Assessed",
            "rating_label": "Not Assessed",
            "explanation": NOT_ASSESSED_EXPLANATIONS.get(
                outcome, f"Not assessed — {outcome}."
            ),
        }
    rel = relation.strip()

    # 1. The judge could not decide from what it saw.
    if rel == "Unclear / insufficient evidence":
        return {
            "rating": ReliabilityRating.UNRESOLVED.value,
            "badge": "⚪ Unresolved",
            "rating_label": "Unresolved Evidence",
            "explanation": "The judge could not decide from the retrieved passages whether the cited paper supports this sentence.",
        }

    # 1b. The paper was read and does not say this.
    if rel == "Does not support":
        return {
            "rating": ReliabilityRating.UNSUPPORTED.value,
            "badge": "🟠 Unsupported",
            "rating_label": "Unsupported Citation",
            "explanation": "The cited paper does not report the finding or the conditions this sentence attributes to it.",
        }
```

Delete the old block that began `if rel in ("Does not support", "Unclear / insufficient evidence") or rel.startswith("Deferred"):`. Leave the Contradicts / retracted / span / partial / supports branches as they are. Remove the old `test_deferred_is_unresolved` test only if it fails — with `relation="Deferred (…)"` and default `outcome="judged"` it now falls through to the final "Unrecognized relation" branch; change that test to pass `outcome="deferred_paywalled"` and assert `UNRESOLVED`.

- [ ] **Step 4: Run the tests**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_reliability_policy.py tests/test_seed_audit.py tests/test_verifier.py -q`
Expected: all pass (`seed_audit` and `agent8` call `evaluate_reliability` positionally/with the old keywords, which still work).

- [ ] **Step 5: Commit**

```bash
git add research_assistant/judgement/policy.py tests/test_reliability_policy.py
git commit -m "feat(policy): rate only judged verdicts; Does-not-support is UNSUPPORTED; explain each not-assessed outcome"
```

---

### Task 4: Source assessor stops grading a bare DOI as peer-reviewed

**Files:**
- Modify: `research_assistant/judgement/source_assessor.py` (step 4 of `assess_source`, and a new step before it)
- Test: `tests/test_source_assessor.py`

- [ ] **Step 1: Write the failing tests**

```python
    def test_doi_without_venue_is_unknown(self):
        res = assess_source({"title": "Nanoscale direct mapping of noise source activities", "doi": "10.1021/acsnano"})
        self.assertEqual(res["grade"], SourceGrade.UNKNOWN.value)
        self.assertIn("venue", res["rationale"].lower())

    def test_monograph_is_unknown_not_primary(self):
        res = assess_source(ref_info={"title": "Semiconductor Nanostructures", "venue": "Semiconductor Nanostructures",
                                      "is_monograph": True, "doi": "10.1093/acprof:oso/9780199534425.001.0001"})
        self.assertEqual(res["grade"], SourceGrade.UNKNOWN.value)
        self.assertIn("book", res["rationale"].lower())

    def test_primary_rationale_says_heuristic(self):
        res = assess_source({"title": "Synthesis of MnFe2O4 nanoparticles", "venue": "Journal of Alloys and Compounds",
                             "doi": "10.1016/j.jallcom.2021.123456"})
        self.assertEqual(res["grade"], SourceGrade.STANDARD_PRIMARY.value)
        self.assertIn("not verified", res["rationale"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_source_assessor.py -q`
Expected: 3 FAIL.

- [ ] **Step 3: Implement**

In `assess_source`, immediately after the `is_review` block returns (i.e. before `# 4. Primary Peer-Reviewed Literature`), insert:

```python
    # 3b. Books and chapters are not primary literature and are not graded.
    if meta.get("is_monograph"):
        return {
            "grade": SourceGrade.UNKNOWN.value,
            "grade_label": "Book / Chapter",
            "badge": "📖 Book / Chapter",
            "is_preprint": False,
            "is_retracted": False,
            "venue": venue,
            "citation_count": citation_count,
            "rationale": "Book or book chapter — not graded as primary literature.",
        }
```

Change step 4's guard from `if venue or doi:` to `if venue:` and append to both rationales the suffix `" (heuristic from venue and title; not verified against Crossref)"` — e.g. `"Peer-reviewed primary experimental or theoretical publication (heuristic from venue and title; not verified against Crossref)."`. Replace the final step-5 return's rationale with:

```python
            "rationale": (
                "DOI present but venue unknown — peer-review status not graded."
                if doi else
                "Insufficient metadata to assess peer review status or venue rigor."
            ),
```

- [ ] **Step 4: Run the tests**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_source_assessor.py tests/test_reliability_policy.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/judgement/source_assessor.py tests/test_source_assessor.py
git commit -m "fix(source-assessor): a bare DOI is not peer review; books are not primary; rationale says heuristic"
```

---

### Task 5: Honest outcomes — `judgement=None` on failure, one `compute_totals`, outcome-first explanations, non-claims partitioned out

**Files:**
- Modify: `research_assistant/shared/seed_audit.py` — `_judge_claim_entry` (≈351–428), `audit_seed_citations` (partition block, reliability loop, totals block), `cross_check_seed_audit` (totals block ≈ lines 575–615), `explain_rubric_verdict` (≈1088)
- Test: `tests/test_seed_audit.py`

**Interfaces:**
- Produces:
  ```python
  VERDICTS = ("Supports", "Partially supports", "Contradicts", "Does not support", "Unclear / insufficient evidence")
  NOT_ASSESSED_OUTCOMES = ("not_downloaded", "deferred_paywalled", "cap_exceeded", "cluster_skipped",
                           "not_attempted", "no_evidence", "retrieval_failed", "parse_failed",
                           "call_failed", "not_a_claim", "malformed_claim", "unresolved_ref")
  def compute_totals(results: list[dict]) -> dict   # keys: total, downloaded, judged, <each VERDICT>,
                                                    # not_downloaded, deferred_paywalled, not_assessed{}, coverage{}, reliability{}
  ```
  `totals["coverage"] = {"downloaded", "attempted", "judged"}`; `totals["reliability"]` gains `"unsupported"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_seed_audit.py`; add `compute_totals` to the import list at the top of the file, and `import contextlib` to the imports)

`audit_seed_citations` writes into `AUDIT_DIR` (the real `data/raw/seed_audits/`). Every new test that runs it goes through this helper so nothing lands in `data/`:

```python
@contextlib.contextmanager
def _audit_dirs():
    """Run an audit with its output redirected to a temporary directory."""
    import research_assistant.shared.seed_audit as sa
    with tempfile.TemporaryDirectory() as tmp, patch.object(sa, "AUDIT_DIR", tmp):
        yield tmp


class TestHonestOutcomes(unittest.TestCase):
    def test_failure_outcomes_carry_no_judgement(self):
        from research_assistant.shared.seed_audit import _judge_claim_entry
        item = {"claim": "Graphene is a semimetal with linear dispersion.", "document": "x.pdf", "context": ""}
        with patch("research_assistant.shared.seed_audit.hybrid_search", side_effect=ConnectionError("Failed to connect to Ollama")):
            _judge_claim_entry(item, MagicMock(), MagicMock(), [], [])
        self.assertEqual(item["outcome"], "retrieval_failed")
        self.assertIsNone(item["judgement"])
        self.assertIn("Ollama", item["reason"])

        item = {"claim": "Graphene is a semimetal with linear dispersion.", "document": "x.pdf", "context": ""}
        with patch("research_assistant.shared.seed_audit.hybrid_search", return_value=[]):
            _judge_claim_entry(item, MagicMock(), MagicMock(), [], [])
        self.assertEqual(item["outcome"], "no_evidence")
        self.assertIsNone(item["judgement"])

    def test_compute_totals_counts_verdicts_over_judged_only(self):
        results = [
            {"outcome": "judged", "judgement": "Supports", "downloaded": True, "reliability": "HIGH"},
            {"outcome": "judged", "judgement": "Does not support", "downloaded": True, "reliability": "UNSUPPORTED"},
            {"outcome": "judged", "judgement": "Unclear / insufficient evidence", "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "retrieval_failed", "judgement": None, "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "cap_exceeded", "judgement": None, "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "not_a_claim", "judgement": None, "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "not_downloaded", "judgement": None, "downloaded": False, "reliability": "UNRESOLVED"},
            {"outcome": "deferred_paywalled", "judgement": None, "downloaded": False, "reliability": "UNRESOLVED"},
        ]
        t = compute_totals(results)
        self.assertEqual(t["total"], 8)
        self.assertEqual(t["judged"], 3)
        self.assertEqual(t["Supports"], 1)
        self.assertEqual(t["Does not support"], 1)
        self.assertEqual(t["Unclear / insufficient evidence"], 1)   # the judged one only
        self.assertEqual(t["not_assessed"], {"retrieval_failed": 1, "cap_exceeded": 1, "not_a_claim": 1,
                                             "not_downloaded": 1, "deferred_paywalled": 1})
        self.assertEqual(t["coverage"], {"downloaded": 6, "attempted": 4, "judged": 3})
        self.assertEqual(t["reliability"]["unsupported"], 1)
        self.assertEqual(t["judged"] + sum(t["not_assessed"].values()), t["total"])

    def test_explain_is_outcome_first(self):
        self.assertIn("budget", explain_rubric_verdict({"outcome": "cap_exceeded", "judgement": None}))
        self.assertIn("backend", explain_rubric_verdict({"outcome": "not_attempted", "judgement": None}))
        self.assertIn("software", explain_rubric_verdict({"outcome": "not_a_claim", "role": "software", "judgement": None}))
        self.assertIn("fragment", explain_rubric_verdict({"outcome": "malformed_claim", "judgement": None}))
        self.assertIn("bibliography", explain_rubric_verdict({"outcome": "unresolved_ref", "judgement": None}))
        self.assertIn("cites", explain_rubric_verdict({"outcome": "cluster_skipped", "judgement": None}))
        # The "insufficient evidence" sentence is only reachable from a judged item.
        self.assertNotIn("fragmentary", explain_rubric_verdict({"outcome": "cap_exceeded", "judgement": None}))
        self.assertIn("fragmentary", explain_rubric_verdict({"outcome": "judged", "judgement": "Unclear / insufficient evidence"}))

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_non_claims_are_partitioned_before_judging(self, mock_manifest, mock_find_tei, mock_search, mock_judge):
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(AUTHOR_YEAR_TEI_XML); tei_file = tf.name
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            pdf = dummy_pdf.name
        mock_find_tei.return_value = tei_file
        mock_manifest.return_value = {
            k: {"key": k, "path": pdf, "xml_id": xid, "cited_by": "seed.pdf", "title": t}
            for k, xid, t in (
                ("a", "b9", "Algorithm 832"), ("b", "b16", "Path-integral seismic imaging"),
                ("c", "b25", "Target-oriented inversion using the patched green's function method"),
                ("d", "b27", "Subsurface-domain objective functions"), ("e", "b6", "Target-level waveform inversion"),
            )
        }
        mock_search.return_value = [{"text": "Evidence.", "metadata": {"document": os.path.basename(pdf), "section": "results", "page": 3}}]
        mock_judge.return_value = {
            "judgement": "Supports", "confidence": "High", "supporting_span": "Evidence.", "reason": "r",
            "slots": {"finding": {"assertion": "a", "verdict": "Supports"}, "scope": {"assertion": "s", "verdict": "Supports"},
                      "strength": {"assertion": "t", "verdict": "Not applicable"}},
            "evidence_sufficiency": "sufficient",
        }
        with _audit_dirs():
            report = audit_seed_citations("seed.pdf", search_resources=(MagicMock(), MagicMock(), [], []),
                                          max_claims=10, skip_if_cached=False)
        by_xid = {r["ref"]["xml_id"]: r for r in report["results"]}
        self.assertEqual(by_xid["b9"]["outcome"], "not_a_claim")     # software
        self.assertEqual(by_xid["b25"]["outcome"], "not_a_claim")    # method
        self.assertEqual(by_xid["b16"]["outcome"], "judged")
        self.assertEqual(mock_judge.call_count, 3)                    # b16, b27, b6
        self.assertEqual(report["totals"]["not_assessed"]["not_a_claim"], 2)
        self.assertEqual(by_xid["b9"]["reliability_explanation"][:12], "Not assessed")
        os.unlink(tei_file); os.unlink(pdf)
```

Also, in the existing `test_audit_seed_citations_totals_math`, after the `expected_sum` assertion add:

```python
            self.assertEqual(t["judged"] + sum(t["not_assessed"].values()), t["total"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_seed_audit.py -q -k "HonestOutcomes or totals_math"`
Expected: FAIL — `ImportError: cannot import name 'compute_totals'`.

- [ ] **Step 3: Failure paths leave no verdict** — in `_judge_claim_entry`, change all four failure branches so `item["judgement"] = None` (retrieval_failed, no_evidence, parse_failed, call_failed). Keep `outcome` and `reason` as they are.

- [ ] **Step 4: Add `VERDICTS`, `NOT_ASSESSED_OUTCOMES`, `compute_totals`** (module level, above `audit_seed_citations`)

```python
VERDICTS = (
    "Supports", "Partially supports", "Contradicts", "Does not support",
    "Unclear / insufficient evidence",
)

# Every way a (sentence, citation) pair ends without a verdict. Spec §3.
NOT_ASSESSED_OUTCOMES = (
    "not_downloaded", "deferred_paywalled", "cap_exceeded", "cluster_skipped",
    "not_attempted", "no_evidence", "retrieval_failed", "parse_failed",
    "call_failed", "not_a_claim", "malformed_claim", "unresolved_ref",
)
_ATTEMPTED_OUTCOMES = ("judged", "no_evidence", "retrieval_failed", "parse_failed", "call_failed")
_RELIABILITY_KEYS = (
    ("high", "HIGH"), ("moderate", "MODERATE"), ("low", "LOW"),
    ("contradicted", "CONTRADICTED"), ("unsupported", "UNSUPPORTED"), ("unresolved", "UNRESOLVED"),
)


def compute_totals(results: list[dict]) -> dict:
    """The one place counts come from. Verdict counts are over judged items
    only; everything else is in not_assessed, keyed by outcome, so
    judged + sum(not_assessed) == total always holds."""
    judged = [r for r in results if r.get("outcome") == "judged"]
    totals = {
        "total": len(results),
        "downloaded": sum(1 for r in results if r.get("downloaded")),
        "judged": len(judged),
    }
    for verdict in VERDICTS:
        totals[verdict] = sum(1 for r in judged if r.get("judgement") == verdict)
    totals["not_downloaded"] = sum(1 for r in results if r.get("outcome") == "not_downloaded")
    totals["deferred_paywalled"] = sum(1 for r in results if r.get("outcome") == "deferred_paywalled")
    not_assessed = {}
    for outcome in NOT_ASSESSED_OUTCOMES:
        n = sum(1 for r in results if r.get("outcome") == outcome)
        if n:
            not_assessed[outcome] = n
    totals["not_assessed"] = not_assessed
    totals["coverage"] = {
        "downloaded": totals["downloaded"],
        "attempted": sum(1 for r in results if r.get("outcome") in _ATTEMPTED_OUTCOMES),
        "judged": len(judged),
    }
    totals["reliability"] = {
        key: sum(1 for r in results if r.get("reliability") == rating)
        for key, rating in _RELIABILITY_KEYS
    }
    return totals
```

- [ ] **Step 5: Partition non-claims and mark `downloaded` on every item** — in `audit_seed_citations`, right after `claims = extract_seed_citation_claims(tei_path)` and its empty-check, and before `downloaded_manifest = _load_downloaded_manifest()`, add:

```python
    # Pairs that are not claims about a paper's findings never reach the
    # judge, the budget, or the missing-references table.
    skipped_claims = []
    judgeable = []
    for c in claims:
        c["judgement"] = None
        if not c.get("resolved", True):
            c["outcome"] = "unresolved_ref"
            c["reason"] = "Citation marker could not be matched to a bibliography entry."
        elif c.get("role") in SKIP_ROLES:
            c["outcome"] = "not_a_claim"
            c["reason"] = f"{c['role']} citation — not a verifiable claim about the cited paper."
        elif {"placeholder_residue", "fragment"} & set(c.get("claim_quality") or []):
            c["outcome"] = "malformed_claim"
            c["reason"] = f"Extracted sentence is not judgeable: {', '.join(c['claim_quality'])}."
        else:
            judgeable.append(c)
            continue
        c["downloaded"] = False
        skipped_claims.append(c)
    claims = judgeable
```

Then, in the `all_results = …` line, append `+ skipped_claims`. In the reliability loop replace the `evaluate_reliability(` call with:

```python
        rel_eval = evaluate_reliability(
            relation=r.get("judgement"),
            source_grade=source_eval["grade"],
            confidence=r.get("confidence"),
            span_verified=r.get("span_verified"),
            rubric_violations=r.get("rubric_violations"),
            rubric_mismatch=r.get("rubric_mismatch", False),
            outcome=r.get("outcome") or "judged",
        )
```

Replace the whole hand-built `totals = { … }` dict in `audit_seed_citations` with `totals = compute_totals(all_results)`. Do the same in `cross_check_seed_audit`: replace its `totals = { … }` block with `totals = compute_totals(results)` and make its `evaluate_reliability(` call identical to the one above. In `cross_check_seed_audit`'s `unresolved_claims` outcome tuple add `"call_failed"` and `"not_attempted"`.

Also in the `cap_exceeded` loop change `c["judgement"] = "Unclear / insufficient evidence"` to `c["judgement"] = None`.

- [ ] **Step 6: Make `explain_rubric_verdict` outcome-first** — replace the function body from its start down to (and including) the `if outcome in ("retrieval_failed", "call_failed", "parse_failed"):` block with:

```python
    judgement = item.get("judgement") or ""
    outcome = item.get("outcome", "")
    slots = item.get("slots") or {}

    if outcome == "deferred_paywalled" or "Deferred" in judgement:
        reason = item.get("reason")
        if reason:
            return f"{reason} Upload the missing reference PDF(s) to verify this claim."
        return (
            "Evaluation deferred: More than 50% of the references cited in this paragraph are missing from the corpus. "
            "Upload the missing reference PDF(s) to enable empirical verification."
        )
    if outcome == "not_downloaded":
        return (
            "This reference paper was paywalled, a book, or otherwise unavailable for open-access download. "
            "Its full text is not in the corpus, so claims citing it could not be empirically verified."
        )
    if outcome == "no_evidence":
        return "The cited reference is in the corpus, but semantic and keyword search found no passages discussing this specific assertion."
    if outcome in ("retrieval_failed", "call_failed", "parse_failed"):
        return f"Evaluation could not complete due to a processing issue: {item.get('reason', outcome)}."
    if outcome == "cap_exceeded":
        return "Not assessed: the per-paper claim budget was reached before this citation. Re-run with a higher budget to judge it."
    if outcome == "not_attempted":
        return "Not assessed: the audit stopped early because the model backend was unreachable. Re-run once the backend is up."
    if outcome == "cluster_skipped":
        return "Not assessed: this sentence cites many papers; only the first few were judged."
    if outcome == "not_a_claim":
        role = item.get("role", "non-evidential")
        return f"Not assessed: this is a {role} citation, not a verifiable claim about the cited paper's findings."
    if outcome == "malformed_claim":
        return f"Not assessed: the extracted sentence is a fragment ({', '.join(item.get('claim_quality') or [])}) and could not be judged."
    if outcome == "unresolved_ref":
        return "Not assessed: the citation marker could not be matched to a bibliography entry."
    if outcome != "judged":
        return item.get("reason") or f"Not assessed ({outcome})."
```

The rest of the function (the `_val` helper and the per-judgement sentences) is unchanged.

- [ ] **Step 7: Run the seed-audit suite**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_seed_audit.py -q`
Expected: all pass. `test_generate_seed_audit_markdown_and_explain` and `test_generate_seed_audit_markdown_with_deferred` may fail if their fixture items carry `outcome` values with `judgement` strings the old explain relied on — read the failing assertion; the fixture item needs `"outcome": "judged"` where it asserts a rubric sentence. Do not weaken the assertion.

- [ ] **Step 8: Commit**

```bash
git add research_assistant/shared/seed_audit.py tests/test_seed_audit.py
git commit -m "fix(audit): failures carry no verdict; one compute_totals; explanations by outcome; non-claims partitioned out"
```

---

### Task 6: Budget goes where it matters; runs stop when the backend is down; history is kept

**Files:**
- Modify: `research_assistant/shared/seed_audit.py` — new `prioritise_claims`, the judging loop in `audit_seed_citations`, the write-out block
- Test: `tests/test_seed_audit.py`

**Interfaces:**
- Produces:
  ```python
  SECTION_RANK = {"results": 0, "discussion": 0, "conclusion": 0, "methods": 1}   # default 2
  MAX_PAIRS_PER_SENTENCE = 3
  MAX_CONSECUTIVE_BACKEND_FAILURES = 3
  BACKEND_FAILURE_OUTCOMES = ("retrieval_failed", "call_failed")
  def prioritise_claims(claims: list[dict]) -> tuple[list[dict], list[dict]]   # (to_judge in order, cluster_skipped)
  ```
  The report gains `report["aborted"]: None | {"reason": str, "after_attempted": int}`. The history copy goes to `os.path.join(AUDIT_DIR, "history")`, computed at write time so a test that patches `AUDIT_DIR` redirects both files.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_seed_audit.py`; the fixture goes at module level next to the other TEI strings)

```python
BREAKER_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><head>Results</head>
<p xml:id="p_0"><s>Graphene shows extraordinary electronic mobility in suspended flakes <ref type="bibr" target="#b0">[1]</ref>.</s><s>Thermal conductance across grain boundaries is strongly suppressed <ref type="bibr" target="#b1">[2]</ref>.</s><s>A negative Poisson ratio was reported in the rippled lattice <ref type="bibr" target="#b2">[3]</ref>.</s><s>Shear strain opens a bandgap of several hundred meV <ref type="bibr" target="#b3">[4]</ref>.</s><s>Phase coherence survives up to room temperature in these devices <ref type="bibr" target="#b4">[5]</ref>.</s></p>
</div></body>
<back><listBibl>
<biblStruct xml:id="b0"><analytic><title level="a" type="main">Paper b0</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
<biblStruct xml:id="b1"><analytic><title level="a" type="main">Paper b1</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
<biblStruct xml:id="b2"><analytic><title level="a" type="main">Paper b2</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
<biblStruct xml:id="b3"><analytic><title level="a" type="main">Paper b3</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
<biblStruct xml:id="b4"><analytic><title level="a" type="main">Paper b4</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
</listBibl></back></text></TEI>
"""


class TestBudgetAndBreaker(unittest.TestCase):
    def _c(self, section, cite_count, p, s, xid):
        return {"section": section, "cite_count": cite_count, "paragraph_index": p, "sentence_index": s,
                "ref": {"xml_id": xid}, "claim": f"claim {xid}"}

    def test_results_before_intro_and_singles_before_clusters(self):
        from research_assistant.shared.seed_audit import prioritise_claims
        intro_cluster = [self._c("other", 8, 0, 0, f"b{i}") for i in range(8)]
        intro_single = self._c("introduction", 1, 1, 0, "b20")
        methods_single = self._c("methods", 1, 5, 0, "b30")
        results_pair = [self._c("results", 2, 9, 2, "b40"), self._c("results", 2, 9, 2, "b41")]
        ordered, skipped = prioritise_claims(intro_cluster + [intro_single, methods_single] + results_pair)
        self.assertEqual([c["ref"]["xml_id"] for c in ordered[:4]], ["b40", "b41", "b30", "b20"])
        self.assertEqual(len(ordered), 4 + 3)          # at most 3 of the 8-cite sentence
        self.assertEqual(len(skipped), 5)
        self.assertTrue(all(c["outcome"] == "cluster_skipped" for c in skipped))

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_three_consecutive_backend_failures_stop_the_run(self, mock_manifest, mock_find_tei, mock_search, mock_judge):
        # Five evidential claims on five references, all "downloaded": three
        # failures trip the breaker and two are left not_attempted.
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(BREAKER_TEI_XML); tei_file = tf.name
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            pdf = dummy_pdf.name
        mock_find_tei.return_value = tei_file
        mock_manifest.return_value = {
            x: {"key": x, "path": pdf, "xml_id": x, "cited_by": "seed.pdf", "title": f"Paper {x}"}
            for x in ("b0", "b1", "b2", "b3", "b4")
        }
        mock_search.side_effect = ConnectionError("Failed to connect to Ollama")
        with _audit_dirs():
            report = audit_seed_citations("seed.pdf", search_resources=(MagicMock(), MagicMock(), [], []),
                                          max_claims=50, skip_if_cached=False)
        outcomes = [r["outcome"] for r in report["results"] if r.get("downloaded")]
        self.assertEqual(outcomes.count("retrieval_failed"), 3)
        self.assertEqual(outcomes.count("not_attempted"), 2)
        self.assertIsNotNone(report["aborted"])
        self.assertEqual(report["aborted"]["after_attempted"], 3)
        self.assertIn("Ollama", report["aborted"]["reason"])
        self.assertEqual(mock_judge.call_count, 0)
        self.assertEqual(report["totals"]["Unclear / insufficient evidence"], 0)
        os.unlink(tei_file); os.unlink(pdf)

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_history_copy_is_written(self, mock_manifest, mock_find_tei, mock_search, mock_judge):
        with _audit_dirs() as tmp:
            with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
                tf.write(SAMPLE_TEI_XML); tei_file = tf.name
            mock_find_tei.return_value = tei_file
            mock_manifest.return_value = {}
            audit_seed_citations("seed.pdf", search_resources=(MagicMock(), MagicMock(), [], []), skip_if_cached=False)
            self.assertTrue(os.path.exists(os.path.join(tmp, "seed_audit.json")))
            history = os.listdir(os.path.join(tmp, "history"))
            self.assertEqual(len(history), 1)
            self.assertTrue(history[0].startswith("seed_") and history[0].endswith("_audit.json"))
            os.unlink(tei_file)
```

- [ ] **Step 2: Run to verify they fail**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_seed_audit.py -q -k "BudgetAndBreaker"`
Expected: FAIL — `ImportError: prioritise_claims`, `KeyError: 'aborted'`, and no `history/` directory for the history test.

- [ ] **Step 3: Add the constants and `prioritise_claims`** (module level, next to `compute_totals`)

```python
SECTION_RANK = {"results": 0, "discussion": 0, "conclusion": 0, "methods": 1}
MAX_PAIRS_PER_SENTENCE = 3
MAX_CONSECUTIVE_BACKEND_FAILURES = 3
BACKEND_FAILURE_OUTCOMES = ("retrieval_failed", "call_failed")


def prioritise_claims(claims: list[dict]) -> tuple[list[dict], list[dict]]:
    """The order the budget is spent in, and the pairs it never reaches.

    Results and discussion before methods before introduction; sentences
    with few citations before "[13]–[27] have been proposed"; document order
    last. At most MAX_PAIRS_PER_SENTENCE pairs per sentence — the rest are
    cluster_skipped, which a re-run with a larger budget does not revisit.
    """
    ordered = sorted(
        claims,
        key=lambda c: (
            SECTION_RANK.get(c.get("section", "other"), 2),
            c.get("cite_count", 1),
            c.get("paragraph_index", 0),
            c.get("sentence_index", 0),
        ),
    )
    per_sentence: dict[tuple, int] = {}
    to_judge, skipped = [], []
    for c in ordered:
        key = (c.get("paragraph_index", 0), c.get("sentence_index", 0))
        per_sentence[key] = per_sentence.get(key, 0) + 1
        if per_sentence[key] > MAX_PAIRS_PER_SENTENCE:
            c["outcome"] = "cluster_skipped"
            c["judgement"] = None
            c["reason"] = f"Sentence cites {c.get('cite_count')} papers; only {MAX_PAIRS_PER_SENTENCE} judged."
            skipped.append(c)
        else:
            to_judge.append(c)
    return to_judge, skipped
```

- [ ] **Step 4: Rewrite the judging block** — in `audit_seed_citations`, replace from `claims_to_judge = downloaded_claims[:max_claims]` through the `for c in downloaded_claims[max_claims:]:` loop with:

```python
    ordered, cluster_skipped = prioritise_claims(downloaded_claims)
    claims_to_judge = ordered[:max_claims]
    overflow = ordered[max_claims:]
    aborted = None

    if claims_to_judge:
        if search_resources is None:
            from research_assistant.shared.db import load_search_resources

            search_resources = load_search_resources()
        collection, bm25, texts, metadatas = search_resources

        escalate_k = JUDGEMENT_ESCALATE_TOP_K if JUDGEMENT_ESCALATE_TOP_K > top_k else 0
        consecutive_failures = 0

        for i, item in enumerate(claims_to_judge, 1):
            if aborted:
                item["outcome"] = "not_attempted"
                item["judgement"] = None
                item["reason"] = f"Audit stopped after {aborted['after_attempted']} attempts: {aborted['reason']}"
                continue
            ref_info = item.get("ref") or {}
            ref_lbl = f"[{ref_info.get('index') or '?'}] {ref_info.get('title') or item.get('cite_text', '')}"
            pipeline_status.update_progress(
                detail=f"Judging citation ({i}/{len(claims_to_judge)}): {ref_lbl[:40]}"
            )
            logger.info("[%d/%d] Auditing citation %s: %s", i, len(claims_to_judge), ref_lbl[:40], item["claim"][:80])
            _judge_claim_entry(item, collection, bm25, texts, metadatas, top_k=top_k, escalate_k=escalate_k)

            # A backend that is down fails every call the same way; three in a
            # row is that, not three unlucky citations. Stop, say so, keep
            # the budget for a run that can use it.
            if item.get("outcome") in BACKEND_FAILURE_OUTCOMES:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_BACKEND_FAILURES:
                    aborted = {"reason": item.get("reason", item["outcome"]), "after_attempted": i}
                    logger.warning("Audit stopped after %d consecutive backend failures: %s", i, aborted["reason"])
                    pipeline_status.add_event(f"⚠️ Citation audit stopped: {aborted['reason'][:80]}")
            else:
                consecutive_failures = 0

    for c in overflow:
        if not c.get("outcome"):
            c["outcome"] = "cap_exceeded"
            c["judgement"] = None
            c["reason"] = "Maximum claims evaluation budget reached."

    all_results = claims_to_judge + overflow + cluster_skipped + undownloaded_claims + deferred_claims + skipped_claims
```

(Remove the older `all_results = …` line from Task 5 so there is exactly one.) Add `"aborted": aborted,` to the `report = {…}` dict.

- [ ] **Step 5: Write the history copy** — after `atomic_write_json(out_file, report)`, add:

```python
    # Every run is kept: the latest overwrites <stem>_audit.json as before,
    # and a timestamped copy accumulates real (claim, evidence, verdict)
    # triples for the judge evaluation set.
    history_dir = os.path.join(AUDIT_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    atomic_write_json(os.path.join(history_dir, f"{stem}_{stamp}_audit.json"), report)
```

- [ ] **Step 6: Run the seed-audit suite**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_seed_audit.py -q`
Expected: all pass. `test_audit_seed_citations_totals_math` uses `mock_judge.side_effect = [five verdicts]` in document order — prioritisation changes which claim gets which verdict but not the counts, so its `assertGreater(..., 0)` checks still hold. If `test_three_consecutive_backend_failures_stop_the_run` finds fewer than 3 `retrieval_failed`, the fixture's deferral rule left fewer than 3 downloaded, non-deferred claims — every reference is in the mocked manifest, so that means `paragraph_refs` was empty for some paragraph; check Task 2's `p_refs_dict` population.

- [ ] **Step 7: Commit**

```bash
git add research_assistant/shared/seed_audit.py tests/test_seed_audit.py
git commit -m "feat(audit): prioritise results over intro and singles over clusters; per-sentence cap; stop on backend failure; keep history"
```

---

### Task 7: The judge sees the claim in context; judged items record evidence provenance

**Files:**
- Modify: `research_assistant/judgement/prompt.md` (Input section only), `research_assistant/judgement/judge.py` (`build_prompt`, `judge`), `research_assistant/agents/agent8_verifier.py` (`_judge_once`, `citation_pairs`, the two `_judge_once` calls), `research_assistant/shared/seed_audit.py` (`_judge_claim_entry`)
- Test: `tests/test_judgement.py`, `tests/test_verifier.py`, `tests/test_seed_audit.py`

**Interfaces:**
- Produces: `build_prompt(claim, citation_evidence, context=None)`, `judge(claim, citation_evidence, model=None, context=None)`, `_judge_once(claim, evidence, context=None)`; Agent 8 pairs carry `context`; judged seed-audit items carry `evidence_sections`, `evidence_pages`, `escalated`, `evidence_hits`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_judgement.py`, class `TestBuildPrompt`, add:

```python
    def test_context_block_is_absent_by_default(self):
        prompt = build_prompt("claim text", "evidence text")
        self.assertNotIn("{{CONTEXT_BLOCK}}", prompt)
        self.assertNotIn("**Context", prompt)
        self.assertLess(prompt.index("**Claim:**"), prompt.index("claim text"))

    def test_context_block_precedes_the_claim(self):
        prompt = build_prompt("claim text", "evidence text", context="Before. «claim text» After.")
        self.assertIn("**Context", prompt)
        self.assertIn("«claim text»", prompt)
        self.assertLess(prompt.index("**Context"), prompt.index("**Claim:**"))
        self.assertIn("judge only that sentence", prompt)
```

In `tests/test_verifier.py`, add a test to whichever class tests `citation_pairs` (search for `citation_pairs(` in the file and put it in that class):

```python
    def test_pairs_carry_neighbouring_sentences_as_context(self):
        sentences = ["First sentence here.", "Graphene is a semimetal \\cite{k1}.", "Last sentence here."]
        pairs = citation_pairs(sentences, {"k1": "Some paper"})
        self.assertEqual(pairs[0]["context"], "First sentence here. «Graphene is a semimetal.» Last sentence here.")
```

In `tests/test_seed_audit.py`, add to `TestHonestOutcomes`:

```python
    def test_judged_item_records_provenance_and_passes_context(self):
        from research_assistant.shared.seed_audit import _judge_claim_entry
        item = {"claim": "Graphene is a semimetal with linear dispersion.", "document": "x.pdf",
                "context": "Before. «Graphene is a semimetal with linear dispersion.» After."}
        hits = [{"text": "t1", "metadata": {"document": "x.pdf", "section": "results", "page": 4}},
                {"text": "t2", "metadata": {"document": "x.pdf", "section": "introduction", "page": 1}}]
        verdict = {"judgement": "Supports", "confidence": "High", "supporting_span": "t1", "reason": "r",
                   "slots": {"finding": {"assertion": "a", "verdict": "Supports"}, "scope": {"assertion": "s", "verdict": "Supports"},
                             "strength": {"assertion": "t", "verdict": "Not applicable"}},
                   "evidence_sufficiency": "sufficient"}
        with patch("research_assistant.shared.seed_audit.hybrid_search", return_value=hits), \
             patch("research_assistant.shared.seed_audit._judge_once", return_value=verdict) as mj:
            _judge_claim_entry(item, MagicMock(), MagicMock(), [], [], top_k=1, escalate_k=0)
        self.assertEqual(item["outcome"], "judged")
        self.assertEqual(item["evidence_sections"], ["results"])
        self.assertEqual(item["evidence_pages"], [4])
        self.assertEqual(item["evidence_hits"], 1)
        self.assertFalse(item["escalated"])
        self.assertEqual(mj.call_args.kwargs.get("context") or mj.call_args.args[2], item["context"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_judgement.py tests/test_verifier.py tests/test_seed_audit.py -q -k "context or provenance"`
Expected: FAIL — `TypeError: build_prompt() got an unexpected keyword argument 'context'`, `KeyError: 'context'`, `KeyError: 'evidence_sections'`.

- [ ] **Step 3: The prompt slot** — in `prompt.md`, change the end of the file from

```
**Claim:**
{{CLAIM}}

**Citation Evidence:**
{{CITATION_EVIDENCE}}
```

to

```
{{CONTEXT_BLOCK}}**Claim:**
{{CLAIM}}

**Citation Evidence:**
{{CITATION_EVIDENCE}}
```

(no blank line between the slot and `**Claim:**`; `build_prompt` supplies its own trailing blank line when there is context).

- [ ] **Step 4: `build_prompt` and `judge`** — in `judge.py`:

```python
_CONTEXT_HEADER = (
    "**Context** (the sentences around the claim, from the citing paper; the claim is the "
    "sentence between « and ». Judge only that sentence, and use the rest only to resolve "
    "what its words refer to — never as evidence):\n"
)


def build_prompt(claim: str, citation_evidence: str, context: str | None = None) -> str:
    """Fill the prompt input template. The context block is present only when
    there is context, so the six regression cases render exactly as before."""
    context_block = f"{_CONTEXT_HEADER}{context.strip()}\n\n" if context and context.strip() else ""
    return (
        PROMPT_TEMPLATE
        .replace("{{CONTEXT_BLOCK}}", context_block)
        .replace("{{CLAIM}}", claim)
        .replace("{{CITATION_EVIDENCE}}", citation_evidence)
    )
```

and

```python
def judge(claim: str, citation_evidence: str, model: str | None = None, context: str | None = None) -> dict:
    """Verdict on whether *citation_evidence* supports *claim*."""
    result = chat(
        [{"role": "user", "content": build_prompt(claim, citation_evidence, context=context)}],
        model=model or JUDGEMENT_MODEL,
        temperature=JUDGEMENT_TEMPERATURE,
        options=JUDGEMENT_OLLAMA_OPTIONS,
    )
```

(the rest of `judge` unchanged).

- [ ] **Step 5: Agent 8** — in `agent8_verifier.py`:

```python
@retry(max_retries=_JUDGE_ATTEMPTS, backoff=1.0)
def _judge_once(claim, evidence, context=None):
    return judge(claim, evidence, context=context)
```

In `citation_pairs`, inside the `for index, sentence in enumerate(sentences):` loop after `claim = strip_citations(sentence)`, add:

```python
        before = strip_citations(sentences[index - 1]) if index > 0 else ""
        after = strip_citations(sentences[index + 1]) if index + 1 < len(sentences) else ""
        context = " ".join(p for p in (before, f"«{claim}»", after) if p)
```

and add `"context": context,` to the appended dict. Change both `_judge_once(entry["claim"], entry["evidence"])` calls in `verify_draft` to `_judge_once(entry["claim"], entry["evidence"], context=entry.get("context"))`.

- [ ] **Step 6: Seed audit** — in `_judge_claim_entry`, after `expand_neighbours(...)` and the `item["evidence"] = …` line, add:

```python
    def _provenance(k):
        used = hits[:k]
        item["evidence_hits"] = len(used)
        item["evidence_sections"] = sorted({(h.get("metadata") or {}).get("section") for h in used} - {None})
        item["evidence_pages"] = sorted({(h.get("metadata") or {}).get("page") for h in used} - {None})

    _provenance(top_k)
    item["escalated"] = False
```

Change the two `_judge_once(item["claim"], item["evidence"])` calls to `_judge_once(item["claim"], item["evidence"], context=item.get("context"))`, and inside the escalation branch, after re-assembling evidence, add `_provenance(escalate_k)` and `item["escalated"] = True`.

- [ ] **Step 7: Run the affected suites**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_judgement.py tests/test_verifier.py tests/test_seed_audit.py tests/test_batch_citer.py -q`
Expected: all pass. `test_both_placeholders_are_filled` still passes because `{{CONTEXT_BLOCK}}` is replaced by the empty string.

- [ ] **Step 8: Commit**

```bash
git add research_assistant/judgement/prompt.md research_assistant/judgement/judge.py research_assistant/agents/agent8_verifier.py research_assistant/shared/seed_audit.py tests/test_judgement.py tests/test_verifier.py tests/test_seed_audit.py
git commit -m "feat(judge): pass the claim's neighbouring sentences as context; record evidence sections and pages"
```

---

### Task 8: The Markdown report prints only what was judged, and says what was not

**Files:**
- Modify: `research_assistant/shared/seed_audit.py` — `generate_seed_audit_markdown` (≈1165) and its nested `_format_entry`
- Test: `tests/test_seed_audit.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestHonestMarkdown(unittest.TestCase):
    def _report(self, results, aborted=None):
        for r in results:
            r.setdefault("ref", {"xml_id": "b0", "index": 1, "title": "T", "authors": ["A B"], "year": 2020, "doi": None})
            r.setdefault("sentence", "S."); r.setdefault("claim", "S.")
            r.setdefault("paragraph_id", "p_0"); r.setdefault("paragraph_refs", [])
        return {"seed_name": "seed.pdf", "generated": "now", "model": "m", "results": results,
                "totals": compute_totals(results), "aborted": aborted}

    def test_not_assessed_items_show_no_confidence(self):
        md = generate_seed_audit_markdown(self._report([
            {"outcome": "cap_exceeded", "judgement": None, "downloaded": True, "reason": "Maximum claims evaluation budget reached.",
             "reliability": "UNRESOLVED", "reliability_badge": "⏳ Not Assessed",
             "reliability_explanation": "Not assessed — the per-paper claim budget was reached before this citation."},
        ]))
        self.assertNotIn("Confidence:", md)
        self.assertNotIn("Evidence Sufficiency", md)
        self.assertIn("Not assessed", md)
        self.assertNotIn("fragmentary", md)

    def test_coverage_table_and_abort_banner(self):
        md = generate_seed_audit_markdown(self._report(
            [{"outcome": "retrieval_failed", "judgement": None, "downloaded": True, "reason": "Retrieval failed: Failed to connect to Ollama"},
             {"outcome": "not_attempted", "judgement": None, "downloaded": True, "reason": "stopped"}],
            aborted={"reason": "Retrieval failed: Failed to connect to Ollama", "after_attempted": 1}))
        self.assertIn("Assessment coverage", md)
        self.assertIn("| Judged by the model | **0**", md)
        self.assertIn("audit stopped", md.lower())
        self.assertIn("Ollama", md)

    def test_judged_item_shows_provenance_role_and_warnings(self):
        md = generate_seed_audit_markdown(self._report([
            {"outcome": "judged", "judgement": "Supports", "downloaded": True, "confidence": "Medium",
             "evidence_sufficiency": "sufficient", "supporting_span": "quoted", "reason": "why",
             "slots": {"finding": {"assertion": "a", "verdict": "Supports"}, "scope": {"assertion": "s", "verdict": "Supports"},
                       "strength": {"assertion": "t", "verdict": "Not applicable"}},
             "role": "evidential", "section": "results", "evidence_sections": ["introduction"], "evidence_pages": [2],
             "span_verified": False, "rubric_mismatch": True, "model_judgement": "Partially supports", "escalated": True,
             "evidence_hits": 3, "reliability": "MODERATE", "reliability_badge": "🟡 Moderate", "reliability_explanation": "e"},
        ]))
        self.assertIn("Confidence:** `Medium`", md)
        self.assertIn("Evidence from:** introduction (p. 2)", md)
        self.assertIn("Cited in:** results", md)
        self.assertIn("not found verbatim", md)
        self.assertIn("model said Partially supports", md)
        self.assertIn("Escalated", md)

    def test_deferred_items_are_grouped_per_paragraph(self):
        missing = [{"xml_id": "b9", "index": 9, "title": "Missing paper", "authors": [], "year": 2019, "doi": None}]
        results = [
            {"outcome": "deferred_paywalled", "judgement": None, "downloaded": False, "paragraph_id": "p_3", "section": "results",
             "sentence": "First deferred sentence [9].", "paragraph_missing_refs": missing, "paywall_ratio": 1.0, "reason": "deferred"},
            {"outcome": "deferred_paywalled", "judgement": None, "downloaded": False, "paragraph_id": "p_3", "section": "results",
             "sentence": "Second deferred sentence [9].", "paragraph_missing_refs": missing, "paywall_ratio": 1.0, "reason": "deferred"},
        ]
        md = generate_seed_audit_markdown(self._report(results))
        self.assertEqual(md.count("### Paragraph p_3"), 1)
        self.assertIn("First deferred sentence", md); self.assertIn("Second deferred sentence", md)
        self.assertIn("Missing paper", md)
```

- [ ] **Step 2: Run to verify they fail**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_seed_audit.py -q -k "HonestMarkdown"`
Expected: FAIL on `Confidence:` present, `Assessment coverage` missing, etc.

- [ ] **Step 3: Coverage table, abort banner, Unsupported row** — in `generate_seed_audit_markdown`, after the `"## 1. Executive Summary"` block's reliability table and before `"---"`, insert (build these lines then `lines.extend(...)`):

```python
    coverage = totals.get("coverage", {})
    not_assessed = totals.get("not_assessed", {})
    coverage_lines = [
        "### Assessment coverage",
        "",
        "| | Count |",
        "| :--- | :--- |",
        f"| Citations whose reference is in the corpus | **{coverage.get('downloaded', 0)}** |",
        f"| Attempted (retrieval + model call) | **{coverage.get('attempted', 0)}** |",
        f"| Judged by the model | **{coverage.get('judged', 0)}** |",
    ]
    for outcome, n in sorted(not_assessed.items(), key=lambda kv: -kv[1]):
        coverage_lines.append(f"| Not assessed — {outcome.replace('_', ' ')} | {n} |")
    coverage_lines.append("")
    if report.get("aborted"):
        ab = report["aborted"]
        coverage_lines += [
            f"> ⚠️ **Audit stopped early** after {ab.get('after_attempted')} attempt(s): {ab.get('reason')}  ",
            "> Every citation after that point is *not attempted*. Re-run once the backend is reachable.",
            "",
        ]
```

Add the Unsupported row to the reliability table, between Contradicted and Unresolved:

```python
        f"| 🟠 **Unsupported** | **{totals.get('reliability', {}).get('unsupported', 0)}** | {pct(totals.get('reliability', {}).get('unsupported', 0))} | The cited paper was read and does not report this |",
```

and change the Unresolved row's description to `"Judged but undecidable, or not assessed (see coverage)"`.

- [ ] **Step 4: `_format_entry` prints only judged fields** — replace the `out_lines = [` … `]` block and the following `if rel_line / if src_line / out_lines.extend([...])` with:

```python
        judged = item.get("outcome") == "judged"
        out_lines = [f"### Citation {index}: {ref_num} {ref_title}{ref_year}", ""]
        if judged:
            out_lines.append(
                f"- **Verdict:** `{badge}` · **Confidence:** `{item.get('confidence')}` · "
                f"**Evidence Sufficiency:** `{item.get('evidence_sufficiency')}`"
            )
        else:
            out_lines.append(f"- **Status:** `{badge}` — *{item.get('reliability_explanation') or explain_rubric_verdict(item)}*")
        if judged and rel_line:
            out_lines.append(rel_line)
        if src_line:
            out_lines.append(src_line)
        role = item.get("role"); section = item.get("section")
        if role or section:
            out_lines.append(f"- **Cited in:** {section or '—'} · **Citation role:** {role or '—'}")
        out_lines.extend([
            f"- **Statement in Paper:**  \n  > \"{item.get('sentence') or item.get('claim', '')}\"",
            (
                f"- **Cited Reference:** {ref_authors}{ref_year}. *{ref_title}*. DOI: [`{doi}`]({doi_str})"
                if doi
                else f"- **Cited Reference:** {ref_authors}{ref_year}. *{ref_title}*"
            ),
        ])
        if judged:
            secs = ", ".join(item.get("evidence_sections") or []) or "—"
            pages = ", ".join(str(p) for p in (item.get("evidence_pages") or []))
            out_lines.append(f"- **Evidence from:** {secs}" + (f" (p. {pages})" if pages else "") +
                             f" · {item.get('evidence_hits', 0)} passage(s)")
            if item.get("escalated"):
                out_lines.append(f"- ↻ **Escalated:** first verdict {item.get('first_judgement')}; re-judged on more evidence.")
            if item.get("span_verified") is False:
                out_lines.append("- ⚠ **Supporting span not found verbatim in the evidence** — treat it as a paraphrase.")
            if item.get("rubric_mismatch"):
                out_lines.append(f"- ⚠ **Rubric:** model said {item.get('model_judgement')}; the rules derive {item.get('judgement')}.")
            if "too_long" in (item.get("claim_quality") or []):
                out_lines.append("- ⚠ **Long sentence:** the claim is over 80 words; the verdict is about the whole sentence.")
```

Keep the slot table, span/evidence, `Assessment` and `Why this verdict?` lines that follow, but wrap the `Assessment` line so it prints only when `judged` (the not-judged reason is already on the Status line).

- [ ] **Step 5: Group deferred items per paragraph** — replace the `if paywalled:` section with:

```python
    not_downloaded = [r for r in results if r.get("outcome") == "not_downloaded"]
    deferred = [r for r in results if r.get("outcome") == "deferred_paywalled"]
    if deferred:
        lines.append(f"## {current_sec}. Deferred Paragraphs ({len({r.get('paragraph_id') for r in deferred})})\n")
        by_para: dict = {}
        for r in deferred:
            by_para.setdefault(r.get("paragraph_id", "p_?"), []).append(r)
        for p_id, items in by_para.items():
            first = items[0]
            missing = first.get("paragraph_missing_refs") or []
            lines.append(f"### Paragraph {p_id} ({first.get('section', '—')}) — {len(missing)} missing reference(s)\n")
            for r in items:
                lines.append(f"> {r.get('sentence')}\n")
            for m in missing:
                doi_val = m.get("doi")
                doi_s = f" · [`{doi_val}`](https://doi.org/{doi_val})" if doi_val else ""
                lines.append(f"- [{m.get('index') or '?'}] {m.get('title') or 'Unknown title'} ({m.get('year') or 'n.d.'}){doi_s}")
            lines.append("\n---\n")
        current_sec += 1
    if not_downloaded:
        lines.append(f"## {current_sec}. Paywalled or Unavailable Citations ({len(not_downloaded)})\n")
        for i, item in enumerate(not_downloaded, 1):
            lines.append(_format_entry(item, i))
        current_sec += 1
```

and make `needs_review` / `supported` / `other` outcome-aware:

```python
    needs_review = [r for r in results if r.get("outcome") == "judged"
                    and r.get("judgement") in ("Contradicts", "Does not support", "Unclear / insufficient evidence")]
    supported = [r for r in results if r.get("outcome") == "judged"
                 and r.get("judgement") in ("Supports", "Partially supports")]
    not_assessed = [r for r in results if r.get("outcome") not in ("judged", "not_downloaded", "deferred_paywalled")]
```

and add, after the supported section, a `## N. Not Assessed (len)` section that calls `_format_entry` on `not_assessed` sorted by `outcome`. Delete the old `other` section.

- [ ] **Step 6: Run the seed-audit suite**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_seed_audit.py -q`
Expected: all pass. The two pre-existing markdown tests assert on section titles — if `test_generate_seed_audit_markdown_with_deferred` asserts the old "Paywalled or Unavailable Citations" heading for deferred items, update that assertion to `"Deferred Paragraphs"` (the section was renamed, not removed).

- [ ] **Step 7: Commit**

```bash
git add research_assistant/shared/seed_audit.py tests/test_seed_audit.py
git commit -m "fix(audit-report): coverage table and abort banner; judged fields only for judged items; provenance and rubric warnings; deferred grouped per paragraph"
```

---

### Task 9: Tab 1 mirrors the report

**Files:**
- Modify: `app.py` — `_render_seed_citation_audit` (≈466–760): tiles, filter tabs, `_render_claim_item` badge and body
- Test: `tests/test_app_render.py`

**Interfaces:**
- Produces: `app._audit_item_badge(item: dict) -> str` (pure, tested).

- [ ] **Step 1: Write the failing test** (append to `tests/test_app_render.py`)

```python
class TestAuditItemBadge(unittest.TestCase):
    def test_badges_follow_outcome_before_judgement(self):
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Supports"}), "🟢 Supports")
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Does not support"}), "🟠 Does Not Support")
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Unclear / insufficient evidence"}), "⚪ Unclear / Insufficient Evidence")
        self.assertEqual(app._audit_item_badge({"outcome": "cap_exceeded", "judgement": None}), "⏸ Not Assessed (budget)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_attempted", "judgement": None}), "⏸ Not Assessed (backend down)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_a_claim", "judgement": None, "role": "software"}), "🔧 Not a Claim (software)")
        self.assertEqual(app._audit_item_badge({"outcome": "deferred_paywalled", "judgement": None}), "⏳ Deferred (Pending Evidence)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_downloaded", "judgement": None}), "🔒 Paywalled / Not In Corpus")
        # No outcome ever falls through to a verdict-looking label.
        self.assertNotIn("Unclear", app._audit_item_badge({"outcome": "retrieval_failed", "judgement": None}))
```

- [ ] **Step 2: Run to verify it fails**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_app_render.py -q`
Expected: FAIL — `AttributeError: module 'app' has no attribute '_audit_item_badge'`.

- [ ] **Step 3: The badge helper** — add at module level in `app.py`, directly above `_render_seed_citation_audit`:

```python
_VERDICT_BADGES = {
    "Supports": "🟢 Supports",
    "Partially supports": "🟡 Partially Supports",
    "Contradicts": "🔴 Contradicts",
    "Does not support": "🟠 Does Not Support",
    "Unclear / insufficient evidence": "⚪ Unclear / Insufficient Evidence",
}
_NOT_ASSESSED_BADGES = {
    "deferred_paywalled": "⏳ Deferred (Pending Evidence)",
    "not_downloaded": "🔒 Paywalled / Not In Corpus",
    "cap_exceeded": "⏸ Not Assessed (budget)",
    "cluster_skipped": "⏸ Not Assessed (cluster)",
    "not_attempted": "⏸ Not Assessed (backend down)",
    "retrieval_failed": "⏸ Not Assessed (retrieval failed)",
    "call_failed": "⏸ Not Assessed (model call failed)",
    "parse_failed": "⏸ Not Assessed (unparseable reply)",
    "no_evidence": "⏸ Not Assessed (no passages found)",
    "malformed_claim": "⏸ Not Assessed (sentence fragment)",
    "unresolved_ref": "⏸ Not Assessed (unmatched reference)",
}


def _audit_item_badge(item: dict) -> str:
    """The label a citation row wears. Outcome first: a verdict label is only
    ever shown for an item the judge produced a verdict for."""
    outcome = item.get("outcome")
    if outcome == "judged":
        return _VERDICT_BADGES.get(item.get("judgement") or "", "⚪ Unclear / Insufficient Evidence")
    if outcome == "not_a_claim":
        return f"🔧 Not a Claim ({item.get('role') or 'non-evidential'})"
    return _NOT_ASSESSED_BADGES.get(outcome or "", f"⏸ Not Assessed ({outcome or 'unknown'})")
```

- [ ] **Step 4: Use it, and add the coverage banner, tile and tab** — inside `_render_seed_citation_audit`:

1. Replace the `outcome = item.get("outcome") … else: badge = "⚪ Unclear / Insufficient Evidence"` chain in `_render_claim_item` with `badge = _audit_item_badge(item)`.
2. Directly above the `# Top metric row` comment, add:

```python
        coverage = totals.get("coverage", {})
        if audit.get("aborted"):
            st.error(
                f"⚠️ Audit stopped early after {audit['aborted'].get('after_attempted')} attempt(s): "
                f"{audit['aborted'].get('reason')} — citations after that point were not attempted. "
                "Re-run once the backend is reachable."
            )
        if coverage:
            st.caption(
                f"Assessed by the model: **{coverage.get('judged', 0)}** of {coverage.get('downloaded', 0)} citations "
                f"whose reference is in the corpus ({coverage.get('attempted', 0)} attempted)."
            )
```

3. In the reliability tile row, change `rcols = st.columns(5)` to `st.columns(6)` and insert after the Contradicted tile:

```python
            rcols[4].metric("Unsupported 🟠", rel_totals.get("unsupported", 0), help="The cited paper was read and does not report this")
```

and renumber the Unresolved tile to `rcols[5]`, with `help="Judged but undecidable, or not assessed — see the caption above"`.

4. In the filter tabs, add a fifth tab. Change the `st.tabs([...])` call to:

```python
        not_assessed_items = [r for r in results if r.get("outcome") not in ("judged", "not_downloaded", "deferred_paywalled")]
        tab_supported, tab_review, tab_not_assessed, tab_deferred, tab_all = st.tabs([
            f"Supported ({supp_count})",
            f"Need Review ({rev_count})",
            f"⏸ Not Assessed ({len(not_assessed_items)})",
            f"⏳ Pending Evidence (Deferred) ({deferred_count})",
            f"All Citations ({all_count})",
        ])
```

and add, after the `with tab_review:` block:

```python
        with tab_not_assessed:
            if not_assessed_items:
                st.info("These citations were found in the paper but the judge produced no verdict for them. "
                        "The badge on each says why.")
                for it in sorted(not_assessed_items, key=lambda r: r.get("outcome") or ""):
                    _render_claim_item(it)
            else:
                st.success("Every citation with an available reference was assessed.")
```

Make the `rev_items` filter in `with tab_review:` outcome-aware: `if r.get("outcome") == "judged" and r.get("judgement") in (...)`.

- [ ] **Step 5: Run the test and the app-name tests**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/test_app_render.py tests/test_app_names.py tests/test_app_ui_stress.py -q`
Expected: all pass.

- [ ] **Step 6: Run the full suite**

Run: `CITATION_LOG_FILE=0 /home/shardul/miniconda3/envs/ml/bin/python -m pytest tests/ -q`
Expected: everything passes except model/Chroma-gated skips. Check `data/ingest.lock` is absent first (`tests/test_ingestion.py` blocks on it).

- [ ] **Step 7: Live check on a real paper** (the repo's two-cycle rule: `.agents/rules/test_and_retest.md`)

Start the app with the Browser pane (`preview_start` on the configured Streamlit entry in `.claude/launch.json`), upload `data/raw/arxiv_2007.12504v1.pdf` in Tab 1, run the audit with the backend **up**, and confirm in the UI and in `data/raw/seed_audits/arxiv_2007.12504v1_audit.md`:
- the coverage caption reads *Assessed by the model: N of M*, with N > 0;
- no row shows `Confidence` unless it has a slot table;
- the `[13]…[20]` sentence contributes at most 3 judged rows and the rest wear *Not Assessed (cluster)*;
- `See SM for a discussion…` → *Not a Claim (pointer)*;
- a results-section citation appears before an introduction one in the report's *Citations Needing Review*.
Then stop Ollama (or set `OLLAMA_HOST` to a dead port), re-run with `force=True`, and confirm the red *Audit stopped early* banner appears, `not_attempted` rows exist, and the summary tiles show 0 verdicts rather than 46 "Unclear". Paste the coverage lines from both `.md` files into the commit message body.

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_app_render.py
git commit -m "feat(tab1): outcome-first badges, Not Assessed tab, Unsupported tile, coverage caption and abort banner"
```

---

## Self-review against the spec

- §2.1 sentences/tokens/narrative/quality/role/context → Task 1 (+ wired in Task 2).
- §2.2 number-map resolution, no silent guess, venue/monograph on `ref` → Task 2.
- §2.3 section/cite_count fields (Task 2), `prioritise_claims`, per-sentence cap, breaker, `not_attempted`, `aborted`, history, cross-check retry set → Tasks 5–6.
- §2.4 context block in prompt, `judge`/`_judge_once` context, Agent 8 context, provenance fields → Task 7.
- §2.5 `judgement=None` on failure, `compute_totals`, policy `outcome` + `UNSUPPORTED`, assessor gating, outcome-first `explain_rubric_verdict`, coverage table, abort banner, judged-only fields, provenance/warnings, per-paragraph deferred, Unsupported row, Tab 1 tab/tile/banner/badges → Tasks 3, 4, 5, 8, 9.
- §3 outcome vocabulary: every outcome string appears in `NOT_ASSESSED_OUTCOMES` (Task 5), `explain_rubric_verdict` (Task 5), `policy.NOT_ASSESSED_EXPLANATIONS` (Task 3) and `_NOT_ASSESSED_BADGES` (Task 9); `not_a_claim` has its own branch in each.
- Type consistency: `_resolve_ref` returns `(dict | None, str)` and Task 2 stores it as `ref`/`resolved`/`resolution`; Task 5 reads `resolved`, `role`, `claim_quality`; Task 6 reads `section`, `cite_count`, `paragraph_index`, `sentence_index`; Task 7 writes `evidence_sections`, `evidence_pages`, `evidence_hits`, `escalated`; Task 8 reads all of those plus `first_judgement` (already written by `_judge_claim_entry`); `compute_totals` keys `coverage` / `not_assessed` / `reliability.unsupported` are read by Tasks 8 and 9.
