# research_assistant/shared/bibliography.py
"""Recognise reference-list text that Docling neither labels nor heads as References.

A *reference mark* is a citation fingerprint (volume, page (year); an arXiv id; a DOI; ...).
A paragraph is a *reference entry* when it is short and carries a mark, or carries several
marks at any length (entries Docling merged into one item). A *run* is a stretch of entries,
plus short year-bearing continuations of entries Docling split, and only runs of at least
three entries are treated as a reference list. Calibrated on the Docling cache on 2026-11-07
(docs/superpowers/plans/2026-11-07-reference-leak.md); change the thresholds only with a new
calibration.
"""

from __future__ import annotations

import re
from typing import Sequence

REFERENCE_MARKS: tuple[re.Pattern, ...] = (
    # volume, page (year): "82, 165109 (2010)"
    re.compile(r"\b\d{1,4}\s*,\s*\d{1,6}(?:\s*[-–]\s*\d+)?\s*\((?:18|19|20)\d{2}\)"),
    # volume (year) page, Elsevier style: "102 (2007) 093511"
    re.compile(r"\b\d{1,4}\s*\((?:18|19|20)\d{2}\)\s*\d{1,6}"),
    # arXiv id
    re.compile(r"\barXiv:\s?\d{4}\.\d{4,5}", re.I),
    # DOI
    re.compile(r"\bdoi(?:\.org/|:\s?)10\.\d{4,}", re.I),
    # numbered entry with a year: "[12] A. Author, ..., 2015"
    re.compile(r"^\s*\[?\d{1,3}[\].)]\s+\S.{0,400}\b(?:18|19|20)\d{2}\b"),
)

_YEAR = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b")
_SHORT = 600          # characters; a paragraph this long or longer is prose, not one entry
_MERGED_MARKS = 3     # marks that make a paragraph an entry at any length
_HEAVY_MARKS = 3      # marks that make a stored chunk reference-heavy


def reference_marks(text: str) -> int:
    """Total matches of all reference-mark patterns in ``text``."""
    return sum(len(p.findall(text)) for p in REFERENCE_MARKS)


def is_reference_entry(text: str) -> bool:
    marks = reference_marks(text)
    return (marks >= 1 and len(text) < _SHORT) or marks >= _MERGED_MARKS


def _is_continuation(text: str) -> bool:
    return len(text) < _SHORT and bool(_YEAR.search(text))


def reference_runs(texts: Sequence[str], min_entries: int = 3) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans of ``texts`` that form reference lists.

    A run starts at an entry and extends over entries and continuations; it is
    kept only if it holds at least ``min_entries`` entries. A trailing
    continuation does not extend the span past the run's last entry.
    """
    runs: list[tuple[int, int]] = []
    i, n = 0, len(texts)
    while i < n:
        if not is_reference_entry(texts[i]):
            i += 1
            continue
        start, last_entry, entries = i, i, 1
        j = i + 1
        while j < n:
            if is_reference_entry(texts[j]):
                last_entry, entries = j, entries + 1
            elif not _is_continuation(texts[j]):
                break
            j += 1
        if entries >= min_entries:
            runs.append((start, last_entry + 1))
        i = last_entry + 1
    return runs


def is_reference_heavy_chunk(text: str) -> bool:
    """A stored chunk with at least three reference marks: almost always a reference list."""
    return reference_marks(text) >= _HEAVY_MARKS
