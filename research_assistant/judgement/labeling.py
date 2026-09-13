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


def apply_label(candidate: dict, key: str, existing_ids: set, note: str = "", negation: str = "") -> list[dict]:
    judgement = KEYS[key.strip().lower()]
    ids = set(existing_ids)
    hid = next_id(ids, "h")
    ids.add(hid)
    human = {
        "id": hid,
        "source": "human",
        "claim": candidate["claim"],
        "citation_evidence": candidate["citation_evidence"],
        "expected_judgement": judgement,
        "document": candidate.get("document"),
        "citation_source": candidate.get("citation_source"),
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
            "document": candidate.get("document"),
            "citation_source": candidate.get("citation_source"),
            "notes": "operator-written negation of " + hid,
        })
    return out


def unlabelled(candidates: list[dict], cases: list[dict]) -> list[dict]:
    done = {(c["claim"], c["citation_evidence"]) for c in cases}
    return [c for c in candidates if (c["claim"], c["citation_evidence"]) not in done]
