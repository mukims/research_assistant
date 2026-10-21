"""Retrodiction cases and case corpora management for technique transfer."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from research_assistant import config

REQUIRED_CASE_FIELDS = (
    "id",
    "kind",
    "query",
    "generic",
    "cutoff_year",
    "corpus",
    "papers",
    "exclude_documents",
    "expected",
)


@dataclass
class Case:
    id: str
    kind: str  # "positive" or "negative"
    query: str
    generic: str
    cutoff_year: int | None
    corpus: str  # "main" or "case"
    papers: dict[str, list[str]]  # target, source, distractor, hidden
    exclude_documents: list[str]
    expected: dict  # source_titles, technique_terms, mapping_terms, critic_terms, set_aside


def case_data_dir(case_id: str) -> str:
    """Return filesystem path to case data directory."""
    return os.path.join(config.PROJECT_ROOT, "data", "eval", "transfer", "corpora", case_id)


def _validate_and_build_case(d: dict[str, Any]) -> Case:
    case_id = d.get("id", "unknown")
    for f in REQUIRED_CASE_FIELDS:
        if f not in d:
            raise ValueError(f"Case '{case_id}' missing required field: '{f}'")

    kind = d.get("kind")
    if kind not in ("positive", "negative"):
        raise ValueError(f"Case '{case_id}' has invalid kind: '{kind}' (must be positive or negative)")

    return Case(
        id=d["id"],
        kind=d["kind"],
        query=d["query"],
        generic=d["generic"],
        cutoff_year=d["cutoff_year"],
        corpus=d["corpus"],
        papers=d["papers"],
        exclude_documents=d["exclude_documents"],
        expected=d["expected"],
    )


def load_cases(
    public_path: str | None = None,
    private_dir: str | None = None,
) -> list[Case]:
    """Load public benchmark cases and merge any private answer keys."""
    if public_path is None:
        public_path = os.path.join(os.path.dirname(__file__), "cases", "transfer_retro.jsonl")
    if private_dir is None:
        private_dir = os.path.join(config.DATA_DIR, "eval", "transfer", "answer_keys")

    cases: list[Case] = []

    if os.path.exists(public_path):
        with open(public_path, "r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh, start=1):
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                try:
                    data = json.loads(raw)
                    cases.append(_validate_and_build_case(data))
                except Exception as exc:
                    raise ValueError(f"Error parsing line {line_idx} of {public_path}: {exc}")

    if os.path.exists(private_dir) and os.path.isdir(private_dir):
        for fname in sorted(os.listdir(private_dir)):
            if fname.endswith(".json") and fname != "TEMPLATE.json":
                fpath = os.path.join(private_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    cases.append(_validate_and_build_case(data))
                except Exception as exc:
                    if isinstance(exc, ValueError):
                        raise
                    raise ValueError(f"Error reading private case file {fpath}: {exc}")

    return cases
