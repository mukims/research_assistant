"""Scientific Source Assessor for cited literature.

Classifies scientific evidence sources into tiers based on peer review status,
publication venue impact, citation signals, and retraction status.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from research_assistant.shared.source_key import normalise_doi


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
    r"nature\s+reviews?|trends\s+in|"
    r"advances\s+in(?!\s+(?:neural\s+information\s+processing|cryptology))|"
    r"surveys?\s+in|progress\s+in|"
    r"physics\s+reports|phys\.?\s*rep\.?|reports?\s+on\s+progress|rep\.?\s*prog\.?|"
    r"science\s+reports|reviews?\s+in)\b",
    re.IGNORECASE,
)

_REVIEW_TITLE_INDICATORS = re.compile(
    r"\b(a\s+review|systematic\s+review|literature\s+review|overview\s+and\s+perspectives?|comprehensive\s+review|"
    r"recent\s+progress\s+in|state\s+of\s+the\s+art)\b",
    re.IGNORECASE,
)

_HIGH_IMPACT_VENUES = re.compile(
    r"(?:"
    r"^\s*nature(?:\s+(?:physics|materials|nanotechnology|photonics|electronics|communications|"
    r"machine\s+intelligence|medicine|methods|biotechnology|chemistry|energy|catalysis|astronomy|"
    r"geoscience|climate\s+change|genetics|neuroscience|immunology|cell\s+biology|structural|"
    r"computational\s+science|biomedical\s+engineering|human\s+behaviour|chemical\s+biology|"
    r"microbiology|plants|ecology|sustainability|cancer|cardiovascular|metabolism|synthesis|protocols))?\s*$|"
    r"^\s*science(?:\s+(?:advances|robotics|translational\s+medicine|immunology|signaling))?\s*$|"
    r"^\s*(?:(?:molecular|cancer|developmental)\s+)?cell(?:\s+(?:metabolism|stem\s+cell|"
    r"host(?:\s*&\s*microbe)?|reports|systems|chemical\s+biology))?\s*$|"
    r"\b(?:"
    r"phys(?:ical)?\.?\s*rev(?:iew)?\.?\s*(?:lett|letters|x|b|a|d|e|materials|research|applied)?|"
    r"prl|prx|pnas|proceedings\s+of\s+the\s+national\s+academy\s+of\s+sciences|"
    r"acs\s+nano|advanced\s+materials|nano\s+letters|journal\s+of\s+the\s+american\s+chemical\s+society|"
    r"jacs|angewandte\s+chemie|joule|energy\s*&\s*environmental\s*science|applied\s+physics\s+letters|apl|"
    r"neurips|neural\s+information\s+processing\s+systems|nips|"
    r"icml|international\s+conference\s+on\s+machine\s+learning|"
    r"iclr|international\s+conference\s+on\s+learning\s+representations|"
    r"cvpr|computer\s+vision\s+and\s+pattern\s+recognition|iccv|eccv|"
    r"acl|association\s+for\s+computational\s+linguistics|emnlp|naacl|"
    r"aaai|ijcai|kdd|sigmod|vldb|sosp|osdi|nsdi|sigcomm|pldi|popl|stoc|focs|"
    r"acm\s+ccs|computer\s+and\s+communications\s+security|"
    r"ieee\s+s\s*&\s*p|symposium\s+on\s+security\s+and\s+privacy|usenix\s+security|ndss|"
    r"journal\s+of\s+machine\s+learning\s+research|jmlr|"
    r"ieee\s+trans(?:actions)?|acm\s+trans(?:actions)?|communications\s+of\s+the\s+acm|proceedings\s+of\s+the\s+ieee|"
    r"new\s+england\s+journal\s+of\s+medicine|n\.?\s*engl\.?\s*j\.?\s*med\.?|nejm|"
    r"(?:the\s+)?lancet(?:\s+(?:oncology|neurology|infectious\s+diseases|digital\s+health|respiratory\s+medicine|psychiatry|public\s+health|global\s+health))?|"
    r"jama|journal\s+of\s+the\s+american\s+medical\s+association|bmj|british\s+medical\s+journal|annals\s+of\s+internal\s+medicine"
    r")\b"
    r")",
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
    venue = str(meta.get("venue") or meta.get("journal") or meta.get("container") or "").strip()
    raw_doi = str(meta.get("doi") or "").strip()
    doi = normalise_doi(raw_doi) or raw_doi.lower()
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
    has_published_doi = bool(doi and not any(doi.startswith(pfx) for pfx in _PREPRINT_DOI_PREFIXES))
    has_published_venue = bool(venue and not _PREPRINT_VENUES.search(venue))

    if not is_preprint:
        if has_published_doi or has_published_venue:
            is_preprint = False
        elif arxiv_id:
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
    is_well_cited = citation_count is not None and citation_count >= 50
    if venue:
        is_high_impact = bool(_HIGH_IMPACT_VENUES.search(venue))
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

    if is_well_cited:
        return {
            "grade": SourceGrade.RIGOROUS_PRIMARY.value,
            "grade_label": "Rigorous Primary Literature",
            "badge": "🏛️ Rigorous Primary",
            "is_preprint": False,
            "is_retracted": False,
            "venue": "",
            "citation_count": citation_count,
            "rationale": "Published primary work with substantial citation impact (citation_count >= 50; venue string absent).",
        }

    # 5. Sparse / Unknown Metadata
    return {
        "grade": SourceGrade.UNKNOWN.value,
        "grade_label": "Unknown Source",
        "badge": "❓ Unknown",
        "is_preprint": False,
        "is_retracted": False,
        "venue": "",
        "citation_count": citation_count,
        "rationale": (
            "DOI present but venue unknown — peer-review status not graded."
            if doi else
            "Insufficient metadata to assess peer review status or venue rigor."
        ),
    }


def assess_source_online(
    metadata: dict[str, Any] | None = None,
    ref_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess publication quality, enriching metadata via academic APIs (OpenAlex/CrossRef/S2) if needed."""
    meta = dict(metadata or {})
    doi = meta.get("doi") or (ref_info.get("doi") if ref_info else None)
    title = meta.get("title") or (ref_info.get("title") if ref_info else None)

    if (meta.get("citation_count") is None or "is_retracted" not in meta) and (doi or title):
        try:
            from research_assistant.shared.academic_apis import fetch_bibliometrics_and_abstract
            target = doi or title
            bib = fetch_bibliometrics_and_abstract(target)
            if bib:
                if meta.get("citation_count") is None and bib.get("citation_count") is not None:
                    meta["citation_count"] = bib["citation_count"]
                if "is_retracted" not in meta and bib.get("is_retracted") is not None:
                    meta["is_retracted"] = bib["is_retracted"]
                if not meta.get("venue") and bib.get("venue"):
                    meta["venue"] = bib["venue"]
        except Exception:
            pass

    return assess_source(meta, ref_info=ref_info)

