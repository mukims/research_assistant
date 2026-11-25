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
    "bilayer": "bulk",
    "nanotube": "nanowire",
    "nanotubes": "nanowires",
    "nanoribbon": "nanotube",
    "nanoribbons": "nanotubes",
    "low temperature": "room temperature",
    "low temperatures": "room temperature",
    "room temperature": "cryogenic temperature",
    "ballistic": "diffusive",
    "terahertz": "microwave",
    "THz": "GHz",
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
_NUMBER_WITH_UNIT_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)(\s*(?:%|wt%|at\.?%|mol%|F/g|Wh\s*kg-1|W\s*kg-1|A/g|mA|uA|"
    r"eV|meV|keV|GeV|K|°C|C|nm|µm|um|mm|cm|V|mV|Hz|GHz|THz|s|ms|ns|ps|fs))(?=\s|[.,;:)\]]|$)"
)
_FIG_POINTER_VERB_RE = re.compile(
    r"\b(?:panel\s*\([a-z0-9]+\)|fig(?:ure)?s?\.?\s*[a-z0-9\(\)]+|tab(?:le)?s?\.?\s*[a-z0-9]+)\s*$",
    re.IGNORECASE,
)
_PARTICIPLE_HEDGE_RE = re.compile(
    r",\s*\b(demonstrating|confirming|establishing|proving|revealing|showing|indicating)\b",
    re.IGNORECASE,
)
_HEDGE = [
    (
        re.compile(
            r"(?:\bwill\s+)?\b(shows|show|demonstrates|demonstrate|establishes|establish|confirms|confirm|reveals|reveal|proves|prove|exhibits|exhibit)\b"
        ),
        "may suggest",
    ),
    (
        re.compile(r"\b(increases|decreases|enhances|reduces|improves|suppresses|lowers|raises|followed|follows)\b"),
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
    "followed": "roughly follow",
    "follows": "roughly follow",
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
    seen_pairs: set[tuple[str, str]] = set()
    for step in range(1, n):
        if len(out) >= limit:
            break
        for i in range(n):
            if len(out) >= limit:
                break
            j = (i + step) % n
            a, b = seeds[i], seeds[j]
            if i == j or (a.get("document") and a.get("document") == b.get("document")):
                continue
            pair_key = (a["id"], b["id"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
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
                    accept=[UNCLEAR],
                    keep_paper=False,
                )
            )
    return out


def scope_swap(seed: dict, existing_ids: set):
    for token, replacement in SCOPE_TABLE.items():
        pat = re.compile(rf"(?<!\w){re.escape(token)}(?!\w)")
        if pat.search(seed["claim"]) and pat.search(seed["citation_evidence"]):
            claim = pat.sub(replacement, seed["claim"])
            tid = next_id(existing_ids, "t")
            return _base(seed, "scope_swap", tid, claim, seed["citation_evidence"], "Does not support", accept=[UNCLEAR])
    return None


def number_swap(seed: dict, existing_ids: set):
    for m in _NUMBER_WITH_UNIT_RE.finditer(seed["claim"]):
        num, unit = m.group(1), m.group(2)
        value = float(num)
        if value == 0:
            continue
        ev_pat = re.compile(rf"(?<![\w.\[])({re.escape(num)})\s*{re.escape(unit.strip())}")
        if not ev_pat.search(seed["citation_evidence"]):
            continue
        new = value / 2 if value >= 100 else value * 2
        new_s = f"{new:g}"
        claim = seed["claim"][:m.start(1)] + new_s + seed["claim"][m.end(1):]
        tid = next_id(existing_ids, "t")
        return _base(seed, "number_swap", tid, claim, seed["citation_evidence"], "Contradicts")
    return None


def delete_key_sentence(seed: dict, existing_ids: set):
    idx, sents = key_sentence(seed["claim"], seed["citation_evidence"])
    if idx < 0 or len(sents) < 2 or len(sents) > 6:
        return None
    ct = _tokens(seed["claim"])
    key_overlap = len(ct & _tokens(sents[idx]))
    if key_overlap < 2:
        return None
    remaining = [s for k, s in enumerate(sents) if k != idx]
    # Refuse if another sentence in the passage still carries most of the claim's content words
    if any(len(ct & _tokens(s)) >= max(2, int(0.6 * key_overlap)) for s in remaining):
        return None
    evidence = " ".join(remaining)
    tid = next_id(existing_ids, "t")
    return _base(seed, "delete_key_sentence", tid, seed["claim"], evidence, "Does not support", accept=[UNCLEAR], keep_paper=False)


def hedge_evidence(seed: dict, existing_ids: set):
    idx, sents = key_sentence(seed["claim"], seed["citation_evidence"])
    if idx < 0 or len(sents) > 8:
        return None
    target = sents[idx]
    changed = False

    # 1. Weakening participial result clauses (", demonstrating X" -> ", which may suggest X")
    if _PARTICIPLE_HEDGE_RE.search(target):
        target = _PARTICIPLE_HEDGE_RE.sub(", which may suggest", target)
        changed = True

    # 2. Weakening finite assertive verbs (skipping figure-pointer verbs like "Panel (b) shows")
    for pat, replacement in _HEDGE:
        new_pieces = []
        last_end = 0
        local_hit = False
        for m in pat.finditer(target):
            prefix = target[:m.start()]
            if _FIG_POINTER_VERB_RE.search(prefix):
                continue
            rep = replacement if replacement else "may " + _LEMMA[m.group(1).lower()]
            new_pieces.append(target[last_end:m.start()] + rep)
            last_end = m.end()
            local_hit = True
        if local_hit:
            new_pieces.append(target[last_end:])
            target = "".join(new_pieces)
            changed = True
            break

    if not changed:
        return None
    sents = list(sents)
    sents[idx] = target
    tid = next_id(existing_ids, "t")
    return _base(seed, "hedge_evidence", tid, seed["claim"], " ".join(sents), "Partially supports", keep_paper=False)


def generate(cases: list[dict], cross_pairs: int = 10) -> list[dict]:
    # Never draw seeds from prompt_example: their evidence lives in prompt.md
    # and would be refused by split_held_out() as prompt leakage.
    seeds = [
        c
        for c in cases
        if c["expected_judgement"] == "Supports" and c["source"] == "human"
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
