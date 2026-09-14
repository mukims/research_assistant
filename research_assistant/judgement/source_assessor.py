"""Scientific Source Assessor for cited literature.

Classifies scientific evidence sources into tiers based on peer review status,
publication venue impact, citation signals, and retraction status.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any


class SourceGrade(str, Enum):
    RIGOROUS_PRIMARY = "RIGOROUS_PRIMARY"
    STANDARD_PRIMARY = "STANDARD_PRIMARY"
    SECONDARY_REVIEW = "SECONDARY_REVIEW"
    PREPRINT_UNREVIEWED = "PREPRINT_UNREVIEWED"
    RETRACTED_OR_FLAWED = "RETRACTED_OR_FLAWED"
    UNKNOWN = "UNKNOWN"


_PREPRINT_VENUES = re.compile(
    r"\b(arxiv|biorxiv|medrxiv|techrxiv|ssrn|chemrxiv|preprints?|research\s*square)\b",
    re.IGNORECASE,
)

_PREPRINT_DOI_PREFIXES = (
    "10.48550",  # arXiv
    "10.1101",   # bioRxiv / medRxiv
    "10.21203",  # Research Square
    "10.20944",  # Preprints.org
    "10.2139",   # SSRN
)

# Venues that have "Review" in their name but are premier primary research journals
_PHYS_REV_PRIMARY = re.compile(
    r"\bphys(?:ical)?\.?\s*rev(?:iew)?(?:\s+(?:letters|lett|b|a|x|d|e|materials|research|applied))?\b",
    re.IGNORECASE,
)

_REVIEW_VENUES = re.compile(
    r"\b(reviews?\s+of\s+modern\s+physics|chem(?:ical)?\.?\s*rev(?:iews?)?|annual\s+review|"
    r"nature\s+reviews?|trends\s+in|advances\s+in|surveys?\s+in|progress\s+in)\b",
    re.IGNORECASE,
)

_REVIEW_TITLE_INDICATORS = re.compile(
    r"\b(a\s+review|systematic\s+review|literature\s+review|overview\s+and\s+perspectives?|comprehensive\s+review|"
    r"recent\s+progress\s+in|state\s+of\s+the\s+art)\b",
    re.IGNORECASE,
)

_HIGH_IMPACT_VENUES = re.compile(
    r"\b(nature|science|cell|phys(?:ical)?\.?\s*rev(?:iew)?\.?\s*(?:lett|letters|x|b|a)?|prl|pnas|"
    r"acs\s+nano|advanced\s+materials|nano\s+letters|ieee\s+trans|chem(?:ical)?\.?\s*rev|joule|"
    r"energy\s*&\s*environmental\s*science|applied\s+physics\s+letters|apl)\b",
    re.IGNORECASE,
)


def assess_source(metadata: dict[str, Any] | None = None, ref_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assess publication quality, peer-review status, and rigor of a cited document.

    Args:
        metadata: Metadata dictionary from Chroma or downloaded.json.
        ref_info: Optional reference struct from GROBID parsing.

    Returns:
        Structured assessment dict containing grade, grade_label, badge, flags, and rationale.
    """
    meta = dict(metadata or {})
    if ref_info:
        for k, v in ref_info.items():
            if k not in meta or not meta[k]:
                meta[k] = v

    title = str(meta.get("title") or "").strip()
    venue = str(meta.get("venue") or meta.get("journal") or "").strip()
    doi = str(meta.get("doi") or "").strip().lower()
    arxiv_id = str(meta.get("arxiv_id") or "").strip()
    pub_type = str(meta.get("publication_type") or meta.get("type") or "").strip().lower()

    citation_count: int | None = None
    if "citation_count" in meta and meta["citation_count"] is not None:
        try:
            citation_count = int(meta["citation_count"])
        except (ValueError, TypeError):
            pass

    # 1. Retraction Check
    is_retracted = bool(meta.get("is_retracted") or meta.get("retracted"))
    if not is_retracted:
        if re.search(r"\b(retracted|retraction|withdrawn)\b", title, re.IGNORECASE):
            is_retracted = True
        elif re.search(r"\b(retracted|retraction)\b", venue, re.IGNORECASE):
            is_retracted = True

    if is_retracted:
        return {
            "grade": SourceGrade.RETRACTED_OR_FLAWED.value,
            "grade_label": "Retracted / Flawed",
            "badge": "⛔ Retracted / Flawed",
            "is_preprint": False,
            "is_retracted": True,
            "venue": venue,
            "citation_count": citation_count,
            "rationale": "Source is flagged as retracted or withdrawn.",
        }

    # 2. Preprint Check
    is_preprint = bool(meta.get("is_preprint"))
    if not is_preprint:
        if arxiv_id:
            is_preprint = True
        elif any(doi.startswith(prefix) for prefix in _PREPRINT_DOI_PREFIXES):
            is_preprint = True
        elif _PREPRINT_VENUES.search(venue) or _PREPRINT_VENUES.search(title):
            is_preprint = True
        elif "preprint" in pub_type:
            is_preprint = True

    if is_preprint:
        return {
            "grade": SourceGrade.PREPRINT_UNREVIEWED.value,
            "grade_label": "Unreviewed Preprint",
            "badge": "📝 Unreviewed Preprint",
            "is_preprint": True,
            "is_retracted": False,
            "venue": venue or "Preprint Server",
            "citation_count": citation_count,
            "rationale": "Unreviewed preprint (e.g. arXiv / bioRxiv) lacking formal peer review.",
        }

    # 3. Review / Secondary Synthesis Check
    is_review = False
    if pub_type == "review":
        is_review = True
    elif bool(_REVIEW_VENUES.search(venue)):
        is_review = True
    elif bool(_REVIEW_TITLE_INDICATORS.search(title)):
        if not re.search(r"\b(experimental|measurement|observation|fabrication)\b", title, re.IGNORECASE):
            is_review = True
    elif re.search(r"\breview\b", venue, re.IGNORECASE) and not bool(_PHYS_REV_PRIMARY.search(venue)):
        is_review = True

    if is_review:
        return {
            "grade": SourceGrade.SECONDARY_REVIEW.value,
            "grade_label": "Review / Secondary Literature",
            "badge": "📚 Review / Secondary",
            "is_preprint": False,
            "is_retracted": False,
            "venue": venue,
            "citation_count": citation_count,
            "rationale": "Secondary survey, review, or meta-analysis synthesizing existing primary studies.",
        }

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

    # 4. Primary Peer-Reviewed Literature
    if venue:
        is_high_impact = bool(_HIGH_IMPACT_VENUES.search(venue))
        is_well_cited = citation_count is not None and citation_count >= 50
        if is_high_impact or is_well_cited:
            return {
                "grade": SourceGrade.RIGOROUS_PRIMARY.value,
                "grade_label": "Rigorous Primary Literature",
                "badge": "🏛️ Rigorous Primary",
                "is_preprint": False,
                "is_retracted": False,
                "venue": venue,
                "citation_count": citation_count,
                "rationale": "Peer-reviewed primary research published in a leading venue or with substantial citation replication (heuristic from venue and title; not verified against Crossref).",
            }
        return {
            "grade": SourceGrade.STANDARD_PRIMARY.value,
            "grade_label": "Standard Primary Literature",
            "badge": "📄 Standard Primary",
            "is_preprint": False,
            "is_retracted": False,
            "venue": venue,
            "citation_count": citation_count,
            "rationale": "Peer-reviewed primary experimental or theoretical publication (heuristic from venue and title; not verified against Crossref).",
        }

    # 5. Sparse / Unknown Metadata
    return {
        "grade": SourceGrade.UNKNOWN.value,
        "grade_label": "Unknown Source",
        "badge": "❓ Unknown",
        "is_preprint": False,
        "is_retracted": False,
        "venue": "",
        "citation_count": None,
        "rationale": (
            "DOI present but venue unknown — peer-review status not graded."
            if doi else
            "Insufficient metadata to assess peer review status or venue rigor."
        ),
    }
