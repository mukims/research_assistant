"""Deterministic transforms of a Supports case into cases whose verdict the
rubric fixes by construction (spec §2.2). No model is involved, so these
cannot inherit the judge's blind spots. Each returns None when it cannot
guarantee its label for that seed.
"""

from __future__ import annotations

import re

from research_assistant.judgement.labeling import next_id
from research_assistant.shared.chunking import sentences

UNCLEAR = "Unclear / insufficient evidence"

# Claim-side substitutions. Only applied when the key also appears in the
# evidence, so the swap moves the claim outside what the evidence examined.
SCOPE_TABLE = {
    "MoS2": "WSe2",
    "MoS₂": "WSe₂",
    "WS2": "MoTe2",
    "graphene": "silicon",
    "silicon": "germanium",
    "one-dimensional": "three-dimensional",
    "1D": "3D",
    "two-dimensional": "bulk",
    "2D": "bulk",
    "monolayer": "bulk",
    "nanotube": "nanowire",
    "nanotubes": "nanowires",
    "nanoribbon": "nanotube",
    "low temperature": "room temperature",
    "low temperatures": "room temperature",
    "room temperature": "cryogenic temperature",
    "electron": "phonon",
    "electrons": "phonons",
    "hole": "electron",
    "holes": "electrons",
    "ferromagnetic": "antiferromagnetic",
    "superconducting": "insulating",
    "quantum dot": "quantum well",
    "acidic": "neutral",
    "H2SO4": "KOH",
    "aqueous": "organic",
}

_WORD_RE = re.compile(r"\w+")
_STOP = frozenset(
    "a an and are as at be by for from in is it its of on or that the this to was we were with".split()
)
_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]|\d)")
_HEDGE = [
    (
        re.compile(
            r"\b(shows|show|demonstrates|demonstrate|establishes|establish|confirms|confirm|reveals|reveal|proves|prove|exhibits|exhibit)\b"
        ),
        "may suggest",
    ),
    (
        re.compile(r"\b(increases|decreases|enhances|reduces|improves|suppresses|lowers|raises)\b"),
        None,
    ),  # → "may " + lemma
]
_LEMMA = {
    "increases": "increase",
    "decreases": "decrease",
    "enhances": "enhance",
    "reduces": "reduce",
    "improves": "improve",
    "suppresses": "suppress",
    "lowers": "lower",
    "raises": "raise",
}


def _tokens(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP and len(w) > 1}


def key_sentence(claim: str, evidence: str):
    sents = sentences(evidence)
    if not sents:
        return -1, []
    ct = _tokens(claim)
    scores = [len(ct & _tokens(s)) for s in sents]
    return scores.index(max(scores)), sents


def _base(
    seed: dict,
    transform: str,
    tid: str,
    claim: str,
    evidence: str,
    expected: str,
    accept=None,
    keep_paper=True,
) -> dict:
    return {
        "id": tid,
        "source": "transform",
        "transform": transform,
        "origin": seed["id"],
        "claim": claim,
        "citation_evidence": evidence,
        "expected_judgement": expected,
        "accept": [expected] + [a for a in (accept or []) if a != expected],
        "document": seed.get("document") if keep_paper else None,
        "citation_source": seed.get("citation_source") if keep_paper else None,
        "notes": f"{transform} of {seed['id']}",
    }


def cross_pair(seeds: list[dict], limit: int, existing_ids: set) -> list[dict]:
    ids, out = set(existing_ids), []
    n = len(seeds)
    for i in range(n):
        if len(out) >= limit:
            break
        j = (i + 1) % n
        a, b = seeds[i], seeds[j]
        if i == j or (a.get("document") and a.get("document") == b.get("document")):
            continue
        tid = next_id(ids, "t")
        ids.add(tid)
        out.append(
            _base(
                a,
                "cross_pair",
                tid,
                a["claim"],
                b["citation_evidence"],
                "Does not support",
                keep_paper=False,
            )
        )
    return out


def scope_swap(seed: dict, existing_ids: set):
    for token, replacement in SCOPE_TABLE.items():
        pat = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)")
        if pat.search(seed["claim"]) and pat.search(seed["citation_evidence"]):
            claim = pat.sub(replacement, seed["claim"], count=1)
            tid = next_id(existing_ids, "t")
            return _base(seed, "scope_swap", tid, claim, seed["citation_evidence"], "Does not support", accept=[UNCLEAR])
    return None


def number_swap(seed: dict, existing_ids: set):
    for m in _NUMBER_RE.finditer(seed["claim"]):
        num = m.group(1)
        if not re.search(rf"(?<![\w.]){re.escape(num)}(?![\w.]|\d)", seed["citation_evidence"]):
            continue
        value = float(num)
        new = value / 2 if value >= 100 else value * 2
        new_s = f"{new:g}"
        claim = seed["claim"][:m.start(1)] + new_s + seed["claim"][m.end(1):]
        tid = next_id(existing_ids, "t")
        return _base(seed, "number_swap", tid, claim, seed["citation_evidence"], "Contradicts")
    return None


def delete_key_sentence(seed: dict, existing_ids: set):
    idx, sents = key_sentence(seed["claim"], seed["citation_evidence"])
    if idx < 0 or len(sents) < 2:
        return None
    evidence = " ".join(s for k, s in enumerate(sents) if k != idx)
    tid = next_id(existing_ids, "t")
    return _base(seed, "delete_key_sentence", tid, seed["claim"], evidence, "Does not support", accept=[UNCLEAR])


def hedge_evidence(seed: dict, existing_ids: set):
    idx, sents = key_sentence(seed["claim"], seed["citation_evidence"])
    if idx < 0:
        return None
    target = sents[idx]
    for pat, replacement in _HEDGE:
        m = pat.search(target)
        if not m:
            continue
        rep = replacement if replacement else "may " + _LEMMA[m.group(1).lower()]
        hedged = target[:m.start()] + rep + target[m.end():]
        sents = list(sents)
        sents[idx] = hedged
        tid = next_id(existing_ids, "t")
        return _base(seed, "hedge_evidence", tid, seed["claim"], " ".join(sents), "Partially supports")
    return None


def generate(cases: list[dict], cross_pairs: int = 10) -> list[dict]:
    seeds = [
        c
        for c in cases
        if c["expected_judgement"] == "Supports" and c["source"] in ("human", "prompt_example")
    ]
    ids = {c["id"] for c in cases}
    out = cross_pair(seeds, cross_pairs, ids)
    ids |= {c["id"] for c in out}
    for seed in seeds:
        for fn in (scope_swap, number_swap, delete_key_sentence, hedge_evidence):
            case = fn(seed, ids)
            if case:
                ids.add(case["id"])
                out.append(case)
    return out
