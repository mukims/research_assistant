"""Candidates for labelling. Pure functions; scripts/judge_harvest.py does the I/O.

Two sources (spec §2.1). An abstract sentence is a claim the paper's own body
should support, so the pair (abstract sentence, best body chunk of the same
paper) is a natural Supports candidate — and the ones that aren't are the
labels worth having. A verification record is a real claim/evidence pair
Agent 8 already judged; its verdict rides along hidden so the operator
labels blind.
"""

from __future__ import annotations

import re

from research_assistant.shared.chunking import sentences

# GROBID sometimes drops the space after a full stop: "data.This paper".
# A capital letter straight after [.!?] never happens inside a number
# ("0.5") or an abbreviation followed by a digit ("Fig.2"), so this is safe.
_GLUED_RE = re.compile(r"([.!?])([A-Z])")
_ACK_RE = re.compile(r"\b(we thank|acknowledg|funded by|supported by|grant)\b", re.I)


def resplit_glued(text: str) -> str:
    return _GLUED_RE.sub(r"\1 \2", text or "")


def abstract_sentences(text: str, min_chars: int = 80, max_chars: int = 300) -> list[str]:
    out = []
    for s in sentences(resplit_glued(text)):
        if not (min_chars <= len(s) <= max_chars):
            continue
        if _ACK_RE.search(s):
            continue
        out.append(s)
    return out


def evidence_from_hit(hit: dict) -> str:
    parts = [hit.get("context_before", ""), hit.get("text", ""), hit.get("context_after", "")]
    return "\n".join(p for p in parts if p)


def make_candidate(cid: str, claim: str, hit: dict, document: str, citation_source: str, source: str) -> dict:
    return {
        "id": cid,
        "source": source,
        "claim": claim,
        "citation_evidence": evidence_from_hit(hit),
        "document": document,
        "citation_source": citation_source,
        "section": (hit.get("metadata") or {}).get("section"),
        "hidden": {"model_judgement": None},
    }


def candidates_from_verification(report: dict, prefix: str) -> list[dict]:
    out = []
    for entry in report.get("results", []):
        if entry.get("outcome") != "judged":
            continue
        out.append({
            "id": f"{prefix}_{len(out) + 1:04d}",
            "source": "verification",
            "claim": entry["claim"],
            "citation_evidence": entry["evidence"],
            "document": None,
            "citation_source": entry.get("citation_source"),
            "section": None,
            "context": entry.get("context"),
            "section_heading": entry.get("section_heading"),
            "artifacts": entry.get("artifacts"),
            "hidden": {"model_judgement": entry.get("judgement")},
        })
    return out
