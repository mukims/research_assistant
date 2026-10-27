"""
report.py — Verdict explanations, badges, and Markdown scorecard generation for citation audits.
"""

from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Optional

from research_assistant.shared.source_key import normalise_doi

def get_deferred_missing_references(audit_report: dict) -> list[dict]:
    """Aggregates all unique missing references blocking deferred paragraphs.

    Returns a list of dicts with keys:
        - xml_id: str
        - index: Optional[int]
        - title: str
        - authors: list[str]
        - year: Optional[int]
        - doi: Optional[str]
        - raw_reference: Optional[str]
        - affected_claims: list[str] (unique claim sentences requiring this reference)
    """
    if not audit_report:
        return []

    results = audit_report.get("results", [])
    missing_results = [
        r for r in results
        if r.get("outcome") in ("deferred_paywalled", "not_downloaded") or not r.get("downloaded")
    ]

    missing_map = {}

    for item in missing_results:
        sent = item.get("sentence") or item.get("claim") or ""
        missing_refs = item.get("paragraph_missing_refs") or []
        if not missing_refs and item.get("ref"):
            missing_refs = [item["ref"]]

        for m_ref in missing_refs:
            if not isinstance(m_ref, dict):
                continue
            doi = normalise_doi(m_ref.get("doi"))
            xml_id = m_ref.get("xml_id") or ""
            title = (m_ref.get("title") or "").strip()

            key = xml_id or doi or title.lower()
            if not key:
                continue

            if key not in missing_map:
                missing_map[key] = {
                    "xml_id": xml_id,
                    "index": m_ref.get("index"),
                    "title": title or "Unknown Title",
                    "authors": m_ref.get("authors") or [],
                    "year": m_ref.get("year"),
                    "doi": doi or m_ref.get("doi"),
                    "raw_reference": m_ref.get("raw_reference"),
                    "affected_claims": [],
                }

            if sent and sent not in missing_map[key]["affected_claims"]:
                missing_map[key]["affected_claims"].append(sent)

    return list(missing_map.values())
 
 
get_missing_references = get_deferred_missing_references


_GUARD_VIOLATION = "absence asserted from insufficient evidence"
_SPAN_VIOLATION = "span that is not in the evidence"


def _slot(item: dict, name: str) -> tuple[str, str]:
    """(assertion, verdict) for one slot, tolerating the flat shape old
    reports stored ({"finding": "Supports"})."""
    s = (item.get("slots") or {}).get(name)
    if isinstance(s, dict):
        return str(s.get("assertion") or ""), str(s.get("verdict") or "")
    return "", str(s or "")


def explain_verdict_steps(item: dict) -> list[tuple[str, str]]:
    """How this verdict was reached, as (label, text) lines.

    Assembled from what the record already holds — the slot assertions, the
    retrieval provenance, the model's own reason, the rubric violations — so
    it costs no model call and cannot say anything the judgement did not.

    The line that earns the function is the last one. "Unclear" is reached
    two ways that ask the reader for opposite things: the model could not
    tell from the passages (read them yourself), or the model said the paper
    does not report this while rating those same passages insufficient —
    which the rubric refuses, because three silent chunks are not a silent
    paper (retrieve more of it). The old text called both "too fragmentary
    or ambiguous" and left the reader to guess.
    """
    if item.get("outcome") != "judged":
        return []

    judgement = item.get("judgement") or ""
    f_a, f_v = _slot(item, "finding")
    s_a, s_v = _slot(item, "scope")
    steps: list[tuple[str, str]] = []

    attributed = f_a or item.get("claim") or "—"
    if s_a:
        attributed = f"{attributed} — in {s_a}"
    steps.append(("Attributed to the cited paper", attributed))

    hits = item.get("evidence_hits")
    read = f"{hits} passage{'s' if hits != 1 else ''}" if hits else "the retrieved passages"
    sections = ", ".join(item.get("evidence_sections") or [])
    pages = ", ".join(str(p) for p in (item.get("evidence_pages") or []))
    if sections:
        read += f", from {sections}"
    if pages:
        read += f" (p. {pages})"
    steps.append(("Read from the cited paper", read))

    invented = any(_SPAN_VIOLATION in v for v in (item.get("rubric_violations") or []))
    if item.get("reason"):
        # When the span guard fired, the model's account of the passages is
        # the thing that failed. Printing it under "what the passages said"
        # would repeat a discredited claim as fact.
        label = "What the model said, unverified" if invented else "What the passages said"
        text = item["reason"]
        if invented:
            text += " (This account could not be verified against the passages — see below.)"
        steps.append((label, text))

    span = item.get("supporting_span")
    if span and item.get("span_verified") is not False:
        steps.append(("Verbatim in the cited paper", f"\u201c{span}\u201d"))

    if judgement == "Supports":
        return steps

    guard = any(_GUARD_VIOLATION in v for v in (item.get("rubric_violations") or []))
    hits_n = hits or 0
    if invented:
        gap = (
            f"The model quoted a sentence that does not occur in the retrieved passages — "
            f"\u201c{span}\u201d — so nothing shows those passages report this, and the verdict it "
            f"reached ({item.get('model_judgement') or 'support'}) is recorded as unclear instead."
        )
    elif guard:
        gap = (
            f"The model judged that the paper does not report this, but rated the same passages "
            f"insufficient to tell — silence in {hits_n} passages is not silence in the paper, so the "
            f"verdict is recorded as unclear rather than as a citation error."
        )
    elif s_v == "Does not support":
        gap = (
            f"The passages examined none of what the claim covers ({s_a or 'the stated scope'}), which "
            f"decides the verdict on its own"
            + (
                " — a result outside the claim's scope is not a contradiction, however it reads."
                if f_v == "Contradicts" else "."
            )
        )
    elif f_v == "Does not support":
        gap = "The passages are on a related topic but report nothing about this relationship."
    elif f_v == "Insufficient" or s_v == "Insufficient":
        gap = "The model could not tell from these passages whether the paper reports this."
    else:
        _, st_v = _slot(item, "strength")
        weak = [n for n, v in (("scope", s_v), ("strength", st_v)) if v == "Partially supports"]
        gap = (
            f"The passages support a weaker version of the claim: {' and '.join(weak)} "
            f"{'match' if len(weak) > 1 else 'matches'} only in part."
            if weak else "Not every part of the claim is carried by these passages."
        )
    steps.append(("Why the verdict is not stronger", gap))

    if invented:
        settle = (
            "Re-judging this claim — the quote offered is not in the passages, so the account of "
            "them cannot be relied on. Read the passages below, or re-run on a stronger model."
        )
    elif guard or item.get("evidence_sufficiency") == "insufficient":
        settle = (
            "Reading more of the cited paper — raise the passages judged, or check the sections this "
            "claim would live in. The paper may well say it; these passages do not."
        )
    elif s_v == "Does not support":
        settle = f"A passage in the cited paper reporting this for {s_a or 'the scope the claim states'}."
    else:
        settle = f"A passage in the cited paper stating {f_a or 'the asserted finding'} outright."
    steps.append(("What would settle it", settle))
    return steps


def explain_rubric_verdict(item: dict) -> str:
    """Explains why a citation received its verdict based on the 3-slot rubric."""
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
        note = item.get("resolution_note") or item.get("reason")
        return f"Not assessed: {note}" if note else "Not assessed: the citation marker could not be matched to a bibliography entry."
    if outcome != "judged":
        return item.get("reason") or f"Not assessed ({outcome})."

    def _val(name):
        s = slots.get(name)
        if isinstance(s, dict):
            return s.get("verdict", "")
        return str(s or "")

    f_v = _val("finding")
    s_v = _val("scope")
    st_v = _val("strength")

    if judgement == "Supports":
        return (
            "Fully supported: The cited evidence explicitly verifies the asserted finding, matches the tested scope/conditions, and corroborates the claim's strength."
        )

    if judgement == "Contradicts":
        return (
            "Contradiction flagged: The cited paper investigated the same scope and reported an empirical finding directly opposite or inconsistent with the claim."
        )

    if judgement == "Does not support":
        if s_v in ("Does not support", "Insufficient"):
            return (
                "Scope mismatch: The cited paper investigated a different material system, experimental condition, or domain than asserted in the claim."
            )
        if f_v in ("Does not support", "Insufficient"):
            return (
                "Finding mismatch: While the cited paper is in a related area, it does not report the specific effect or relationship asserted."
            )
        return "Does not support: The evidence does not validate the asserted finding or tested conditions."

    if judgement == "Partially supports":
        hedges = []
        if s_v in ("Partially supports", "Insufficient"):
            hedges.append("the paper tested only a narrower subset of the asserted conditions or scope")
        if st_v in ("Partially supports", "Insufficient"):
            hedges.append("the paper proposed a hedged or specific mechanism whereas the claim asserted a definitive or generalized effect")
        detail = " and ".join(hedges) if hedges else "certain slots were qualified"
        return f"Partial support: The core finding is substantiated in the cited text, but {detail}."

    if judgement == "Unclear / insufficient evidence":
        return (
            "Insufficient evidence: The retrieved passages from the cited paper are too fragmentary or ambiguous to determine whether the claim is supported."
        )

    return item.get("reason") or "Evaluation complete."


_VERDICT_BADGES = {
    "Supports": "🟢 Supports",
    "Partially supports": "🟡 Partially Supports",
    "Contradicts": "🔴 Contradicts",
    "Does not support": "🟠 Does Not Support",
    "Unclear / insufficient evidence": "⚪ Unclear / Insufficient Evidence",
}

# An unjudged citation is one of two very different things, and the report
# used to show them the same way. Either its reference PDF is already in the
# corpus and something stopped the judge — the budget, the sentence cap, a
# backend that was down — which another run fixes; or there was never a claim
# to judge, which no re-run changes. A reader who cannot tell them apart reads
# the first as a missing paper.
_IN_CORPUS_UNJUDGED = frozenset({
    "cap_exceeded", "cluster_skipped", "not_attempted",
    "retrieval_failed", "call_failed", "parse_failed", "no_evidence",
})
_NOTHING_TO_JUDGE = frozenset({"not_a_claim", "malformed_claim", "unresolved_ref"})


def unassessed_bucket(item: dict) -> str | None:
    """"in_corpus_unjudged" | "nothing_to_judge" | None.

    None for the outcomes that are not unassessed at all: a judged citation,
    a paper that was never downloaded, a deferred paragraph.
    """
    outcome = item.get("outcome") or ""
    if outcome in _IN_CORPUS_UNJUDGED:
        return "in_corpus_unjudged"
    if outcome in _NOTHING_TO_JUDGE:
        return "nothing_to_judge"
    return None


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


def generate_seed_audit_markdown(report: dict, seed_title: str | None = None) -> str:
    """Generates a complete, comprehensive Markdown audit report for the given report dict."""
    title = (
        seed_title
        or report.get("seed_name")
        or os.path.basename(report.get("seed_path", "Seed Paper"))
    )
    totals = report.get("totals", {})
    results = report.get("results", [])
    generated = report.get("generated", datetime.now().isoformat(timespec="seconds"))
    model = report.get("model", "judgement-engine")

    total_citations = totals.get("total", len(results))
    pct = lambda val: f"{round((val / total_citations) * 100, 1)}%" if total_citations else "0%"

    lines = [
        f"# 🔍 In-Text Citation Audit Report",
        "",
        f"**Document:** `{title}`  ",
        f"**Generated:** `{generated}`  ",
        f"**Evaluation Model:** `{model}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Category | Count | Percentage | Description |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Total In-Text Citations** | **{total_citations}** | 100% | Unique in-text reference instances |",
        f"| **Downloaded / In Corpus** | **{totals.get('downloaded', 0)}** | {pct(totals.get('downloaded', 0))} | Reference PDFs available in library |",
        f"| 🟢 **Supports** | **{totals.get('Supports', 0)}** | {pct(totals.get('Supports', 0))} | Evidence directly validates claim |",
        f"| 🟡 **Partially Supports** | **{totals.get('Partially supports', 0)}** | {pct(totals.get('Partially supports', 0))} | Core finding matches; scope or strength hedged |",
        f"| 🔴 **Contradicts** | **{totals.get('Contradicts', 0)}** | {pct(totals.get('Contradicts', 0))} | Evidence directly opposes claim |",
        f"| 🟠 **Does Not Support** | **{totals.get('Does not support', 0)}** | {pct(totals.get('Does not support', 0))} | Scope or finding mismatch |",
        f"| ⚪ **Unclear / Insufficient** | **{totals.get('Unclear / insufficient evidence', 0)}** | {pct(totals.get('Unclear / insufficient evidence', 0))} | Fragile or inconclusive evidence |",
        f"| ⏳ **Deferred (Pending Evidence)** | **{totals.get('deferred_paywalled', 0)}** | {pct(totals.get('deferred_paywalled', 0))} | Paragraph >50% paywalled; evaluation deferred |",
        f"| 🔒 **Paywalled / Unchecked** | **{totals.get('not_downloaded', 0)}** | {pct(totals.get('not_downloaded', 0))} | Non-OA reference; unavailable |",
        "",
        "### Scientific Evidence Reliability Rating",
        "",
        "| Reliability Tier | Count | Percentage | Standard & Policy |",
        "| :--- | :--- | :--- | :--- |",
        f"| 🟢 **High Reliability** | **{totals.get('reliability', {}).get('high', 0)}** | {pct(totals.get('reliability', {}).get('high', 0))} | Peer-reviewed primary research with verified verbatim evidence span |",
        f"| 🟡 **Moderate Reliability** | **{totals.get('reliability', {}).get('moderate', 0)}** | {pct(totals.get('reliability', {}).get('moderate', 0))} | Supported by unreviewed preprints, secondary surveys, or hedged findings |",
        f"| 🟠 **Low / Flagged** | **{totals.get('reliability', {}).get('low', 0)}** | {pct(totals.get('reliability', {}).get('low', 0))} | Unverified spans, retracted/flawed sources, or rubric violations |",
        f"| 🔴 **Contradicted** | **{totals.get('reliability', {}).get('contradicted', 0)}** | {pct(totals.get('reliability', {}).get('contradicted', 0))} | Cited peer-reviewed literature directly refutes the claim |",
        f"| 🟠 **Unsupported** | **{totals.get('reliability', {}).get('unsupported', 0)}** | {pct(totals.get('reliability', {}).get('unsupported', 0))} | The cited paper was read and does not report this |",
        f"| ⏳ **Unresolved / Pending** | **{totals.get('reliability', {}).get('unresolved', 0)}** | {pct(totals.get('reliability', {}).get('unresolved', 0))} | Judged but undecidable, or not assessed (see coverage) |",
        "",
    ]

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
    lines.extend(coverage_lines)

    lines.extend([
        "---",
        "",
        "## 2. Verification Rubric & Methodology",
        "",
        "Every in-text citation was audited by extracting the exact sentence in the paper, isolating the referenced source, and retrieving the top matching passages from that source. The evidence is evaluated against a formal **three-slot decomposition**:",
        "",
        "- **Finding Slot**: Does the cited evidence assert the specific phenomenon, relationship, or effect claimed?",
        "- **Scope Slot**: Did the cited paper test the same system, material, conditions, or environment?",
        "- **Strength Slot**: Does the evidence establish causation or generality, or only an isolated observation or hypothesis?",
        "",
        "### Paragraph Context & Compound Citation Policy",
        "",
        "Every in-text citation whose reference PDF is available in the local library is audited directly against the 3-slot rubric. When a sentence cites multiple references and some are missing from the corpus, the audit notes that the available reference may only support part of the compound assertion. Citations for unavailable references are cataloged in Section 3 so they can be uploaded directly.",
        "",
        "---",
        "",
    ])

    missing_refs = get_deferred_missing_references(report)
    if totals.get("deferred_paywalled", 0) > 0:
        lines.append(f"## 3. Missing References Required for Deferred Paragraphs ({len(missing_refs)})\n")
    else:
        lines.append(f"## 3. Missing References Needed for Full Verification ({len(missing_refs)})\n")
    if missing_refs:
        req_col = "Required By Deferred Claim(s)" if totals.get("deferred_paywalled", 0) > 0 else "Required By Claim(s)"
        lines.append(f"| Reference / Title | Authors | Year | DOI | {req_col} |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for ref in missing_refs:
            ref_num = f"[{ref.get('index') or '?'}]"
            title_s = f"{ref_num} {ref.get('title') or 'Unknown Title'}"
            authors_list = ref.get("authors", [])
            auth_s = (
                ", ".join(authors_list[:2]) + (" et al." if len(authors_list) > 2 else "")
                if authors_list
                else "Unknown authors"
            )
            yr_s = str(ref.get("year") or "N/A")
            doi_val = ref.get("doi")
            doi_s = f"[`{doi_val}`](https://doi.org/{doi_val})" if doi_val else "N/A"
            affected = ref.get("affected_claims", [])
            claims_s = "<br>• ".join(f'"{c}"' for c in affected[:3]) if affected else "—"
            if len(affected) > 3:
                claims_s += f"<br>*(+{len(affected) - 3} more)*"
            lines.append(f"| {title_s} | {auth_s} | {yr_s} | {doi_s} | {claims_s} |")
        lines.append("")
    else:
        lines.append("*All references cited in the paper are available in the local corpus.*\n")

    needs_review = [
        r for r in results
        if r.get("outcome") == "judged"
        and r.get("judgement") in ("Contradicts", "Does not support", "Unclear / insufficient evidence")
    ]
    supported = [
        r for r in results
        if r.get("outcome") == "judged"
        and r.get("judgement") in ("Supports", "Partially supports")
    ]
    in_corpus_unjudged = [r for r in results if unassessed_bucket(r) == "in_corpus_unjudged"]
    nothing_to_judge = [r for r in results if unassessed_bucket(r) == "nothing_to_judge"]
    # Anything unassessed for a reason neither bucket names still has to appear.
    nothing_to_judge += [
        r for r in results
        if r.get("outcome") not in ("judged", "not_downloaded", "deferred_paywalled")
        and unassessed_bucket(r) is None
    ]
    deferred = [r for r in results if r.get("outcome") == "deferred_paywalled"]
    not_downloaded = [r for r in results if r.get("outcome") == "not_downloaded"]

    def _format_entry(item, index):
        ref_info = item.get("ref") or {}
        ref_num = f"[{ref_info.get('index') or '?'}]"
        ref_title = ref_info.get("title") or item.get("cite_text") or "Unknown Reference"
        ref_year = f" ({ref_info.get('year')})" if ref_info.get("year") else ""
        ref_authors = (
            ", ".join(ref_info.get("authors", [])[:3])
            + (" et al." if len(ref_info.get("authors", [])) > 3 else "")
            if ref_info.get("authors")
            else "Unknown authors"
        )
        doi = ref_info.get("doi")
        doi_str = f"https://doi.org/{doi}" if doi else "N/A"

        badge = _audit_item_badge(item)

        rel_line = (
            f"- **Scientific Reliability:** `{item.get('reliability_badge', '⚪ Unresolved')}` — *{item.get('reliability_explanation', '')}*"
            if item.get("reliability_badge")
            else None
        )
        source_ass = item.get("source_assessment") or {}
        src_line = (
            f"- **Source Quality:** `{source_ass.get('badge', '❓ Unknown')}` ({source_ass.get('grade_label', '')}) · *{source_ass.get('rationale', '')}*"
            if source_ass.get("badge")
            else None
        )

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
        role = item.get("role"); section = item.get("section_heading") or item.get("section")
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

            steps = explain_verdict_steps(item)
            if steps:
                out_lines.append("- **Why this verdict**")
                out_lines.extend(f"    - *{label}:* {text}" for label, text in steps)

        slots = item.get("slots")
        if slots and isinstance(slots, dict):
            out_lines.append("")
            out_lines.append("**Slot Decomposition Analysis:**")
            out_lines.append("")
            out_lines.append("| Slot | Assertion Extracted from Claim | Evaluation |")
            out_lines.append("| :--- | :--- | :--- |")
            for slot_name in ("finding", "scope", "strength"):
                sdata = slots.get(slot_name) or {}
                if isinstance(sdata, dict):
                    asrt = sdata.get("assertion", "—")
                    v = sdata.get("verdict", "—")
                else:
                    asrt = "—"
                    v = str(sdata or "—")
                out_lines.append(f"| **`{slot_name}`** | {asrt} | `{v}` |")

        if item.get("supporting_span"):
            out_lines.append("")
            out_lines.append(f"- **Verbatim Evidence from Cited Paper:**  \n  > \"{item['supporting_span']}\"")
        elif item.get("evidence"):
            out_lines.append("")
            out_lines.append(f"- **Evidence Passages Considered:**  \n  > \"{item['evidence'][:350].strip()}...\"")

        out_lines.append("")
        if judged and item.get("reason"):
            out_lines.append(f"- **Assessment:** {item['reason']}")
        out_lines.append(f"- **Why this verdict?** {explain_rubric_verdict(item)}")
        out_lines.append("")
        out_lines.append("---")
        out_lines.append("")
        return "\n".join(out_lines)

    current_sec = 4
    if needs_review:
        lines.append(f"## {current_sec}. Citations Needing Review ({len(needs_review)})\n")
        for i, item in enumerate(needs_review, 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    if supported:
        lines.append(f"## {current_sec}. Supported Citations ({len(supported)})\n")
        for i, item in enumerate(supported, 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    if in_corpus_unjudged:
        lines.append(f"## {current_sec}. In the Corpus but Not Judged ({len(in_corpus_unjudged)})\n")
        lines.append(
            "The reference PDF for each of these is already in the library — the judge did not reach "
            "them. Raise the per-paper budget (`CITATION_AUDIT_MAX_CLAIMS`, default 50) or re-run the "
            "audit to have them evaluated; nothing needs to be fetched.\n"
        )
        for i, item in enumerate(sorted(in_corpus_unjudged, key=lambda r: str(r.get("outcome") or "")), 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    if nothing_to_judge:
        lines.append(f"## {current_sec}. Not a Verifiable Claim ({len(nothing_to_judge)})\n")
        lines.append(
            "These citations carry no assertion about the cited paper's findings — a software or "
            "dataset citation, a pointer to a review, a sentence fragment, or a marker that could "
            "not be matched to a bibliography entry. Re-running does not change them.\n"
        )
        for i, item in enumerate(sorted(nothing_to_judge, key=lambda r: str(r.get("outcome") or "")), 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

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
                if not isinstance(m, dict):
                    continue
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

    return "\n".join(lines)

