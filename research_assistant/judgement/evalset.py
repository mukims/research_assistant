"""The judge evaluation set: what a case is, how it loads, what counts as held out.

Three files, three sources kept apart (spec §2): cases.jsonl are the prompt's
own worked examples and are never part of the headline number; human.jsonl
is operator-labelled; transforms.jsonl is generated deterministically from
the Supports cases. held_out() is the guard that keeps the first out of the
score: a case whose text appears in prompt.md measures copying, not judging.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from research_assistant.judgement import judge as _judge

VALID_SOURCES = {"prompt_example", "human", "human_negation", "transform"}
CASES_DIR = Path(_judge.__file__).parent / "cases"
DEFAULT_CASE_FILES = [CASES_DIR / "cases.jsonl", CASES_DIR / "human.jsonl", CASES_DIR / "transforms.jsonl"]

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


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text or "").lower().split())


def validate_case(raw: dict) -> dict:
    case = dict(raw)
    for key in ("id", "claim", "citation_evidence", "expected_judgement"):
        if not str(case.get(key) or "").strip():
            raise ValueError(f"case {case.get('id', '?')!r}: missing {key}")
    case.setdefault("source", "prompt_example")
    if case["source"] not in VALID_SOURCES:
        raise ValueError(f"case {case['id']!r}: source {case['source']!r} not in {sorted(VALID_SOURCES)}")
    if case["expected_judgement"] not in _judge.VALID_JUDGEMENTS:
        raise ValueError(f"case {case['id']!r}: expected_judgement {case['expected_judgement']!r} is not a judgement")
    for key, default in _OPTIONAL_DEFAULTS.items():
        case.setdefault(key, default)
    accept = list(case["accept"] or [])
    bad = [a for a in accept if a not in _judge.VALID_JUDGEMENTS]
    if bad:
        raise ValueError(f"case {case['id']!r}: accept contains {bad}")
    if case["expected_judgement"] not in accept:
        accept.insert(0, case["expected_judgement"])
    case["accept"] = accept
    case["tags"] = [str(t) for t in (case.get("tags") or [])]
    return case


def load_cases(paths=None) -> list[dict]:
    paths = [Path(p) for p in (paths or DEFAULT_CASE_FILES)]
    cases, seen = [], set()
    for path in paths:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = validate_case(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if case["id"] in seen:
                raise ValueError(f"{path}:{line_no}: duplicate case id {case['id']!r}")
            seen.add(case["id"])
            cases.append(case)
    return cases


def held_out(case: dict, prompt_text: str) -> bool:
    """Neither the claim nor the evidence appears in the prompt."""
    prompt = normalise(prompt_text)
    return normalise(case["claim"]) not in prompt and normalise(case["citation_evidence"]) not in prompt


def split_held_out(cases, prompt_text: str):
    scored, refused = [], []
    for case in cases:
        (scored if held_out(case, prompt_text) else refused).append(case)
    return scored, refused
