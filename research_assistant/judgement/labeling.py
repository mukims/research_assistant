"""Turn an operator's keystroke into a case. Pure; scripts/judge_label.py loops."""

from __future__ import annotations

import re

KEYS = {
    "s": "Supports",
    "p": "Partially supports",
    "c": "Contradicts",
    "d": "Does not support",
    "u": "Unclear / insufficient evidence",
}


def next_id(existing_ids: set, prefix: str) -> str:
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    n = max((int(m.group(1)) for i in existing_ids if (m := pat.match(i))), default=0)
    return f"{prefix}_{n + 1:03d}"


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
            # The context marks the sentence that was negated, not the
            # negation; shown with the negated claim it contradicts itself.
            **{**carried, "context": None, "section_heading": None, "artifacts": None},
            "notes": "operator-written negation of " + hid,
        })
    return out


def unlabelled(candidates: list[dict], cases: list[dict]) -> list[dict]:
    done = {(c["claim"], c["citation_evidence"]) for c in cases}
    return [c for c in candidates if (c["claim"], c["citation_evidence"]) not in done]
