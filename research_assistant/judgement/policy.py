"""Deterministic Reliability Policy for Scientific Citations.

Integrates the relation verdict (factual faithfulness) with source grading
(methodological rigor & publication status) to produce an actionable reliability rating.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from research_assistant.judgement.source_assessor import SourceGrade


class ReliabilityRating(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    CONTRADICTED = "CONTRADICTED"
    UNRESOLVED = "UNRESOLVED"


def evaluate_reliability(
    relation: str,
    source_grade: str | None = None,
    confidence: str = "High",
    span_verified: bool | None = None,
    rubric_violations: list[str] | None = None,
    rubric_mismatch: bool = False,
) -> dict[str, Any]:
    """Derive scientific reliability from relation verdict and source grade.

    Args:
        relation: Verdict from the judgement engine (Supports, Partially supports, etc.).
        source_grade: Evaluated SourceGrade of the cited paper.
        confidence: Model confidence (High, Medium, Low).
        span_verified: Whether the cited quote was verified verbatim in the PDF text.
        rubric_violations: Out-of-vocabulary slot violations.
        rubric_mismatch: Whether model's top-level verdict differed from derived slots.

    Returns:
        Dict with keys: rating, badge, rating_label, explanation.
    """
    source_grade = source_grade or SourceGrade.UNKNOWN.value
    rel = relation.strip()
    conf = confidence.capitalize() if confidence else "High"
    has_violations = bool(rubric_violations) or rubric_mismatch

    # 1. Unresolved / Deferred / Missing Evidence
    if rel in ("Does not support", "Unclear / insufficient evidence") or rel.startswith("Deferred"):
        return {
            "rating": ReliabilityRating.UNRESOLVED.value,
            "badge": "⏳ Pending / Unresolved",
            "rating_label": "Unresolved Evidence",
            "explanation": "Citation does not provide sufficient, retrievable evidence to verify the claim.",
        }

    # 2. Contradicted Claims (Refutations)
    if rel == "Contradicts":
        if source_grade == SourceGrade.RETRACTED_OR_FLAWED.value:
            return {
                "rating": ReliabilityRating.LOW.value,
                "badge": "🟠 Low Reliability",
                "rating_label": "Unreliable Refutation",
                "explanation": "Contradiction originates from a retracted or flawed source; caution advised.",
            }
        return {
            "rating": ReliabilityRating.CONTRADICTED.value,
            "badge": "🔴 Contradicted",
            "rating_label": "Contradicted Finding",
            "explanation": "Cited scientific evidence directly conflicts with the statement under comparable conditions.",
        }

    # 3. Flawed or Retracted Sources
    if source_grade == SourceGrade.RETRACTED_OR_FLAWED.value:
        return {
            "rating": ReliabilityRating.LOW.value,
            "badge": "🟠 Low Reliability",
            "rating_label": "Compromised Source",
            "explanation": "The cited reference is retracted or withdrawn; this citation should be replaced.",
        }

    # 4. Hallucinated / Non-Verbatim Evidence Spans
    if span_verified is False:
        return {
            "rating": ReliabilityRating.LOW.value,
            "badge": "🟠 Low Reliability",
            "rating_label": "Unverified Evidence Span",
            "explanation": "The supporting evidence span could not be matched verbatim in the source document.",
        }

    # 5. Partially Supported Claims
    if rel == "Partially supports":
        return {
            "rating": ReliabilityRating.MODERATE.value,
            "badge": "🟡 Moderate Reliability",
            "rating_label": "Partially Supported",
            "explanation": "The core finding is corroborated, but certainty, methodology, or scope is qualified.",
        }

    # 6. Supported Claims
    if rel == "Supports":
        if source_grade == SourceGrade.PREPRINT_UNREVIEWED.value:
            return {
                "rating": ReliabilityRating.MODERATE.value,
                "badge": "🟡 Moderate Reliability",
                "rating_label": "Preprint Corroboration",
                "explanation": "Supported by cited text, but source is an unreviewed preprint lacking formal peer review.",
            }

        if source_grade == SourceGrade.SECONDARY_REVIEW.value:
            return {
                "rating": ReliabilityRating.MODERATE.value,
                "badge": "🟡 Moderate Reliability",
                "rating_label": "Secondary Review Evidence",
                "explanation": "Supported by a secondary review or survey rather than a primary experimental measurement.",
            }

        if source_grade in (SourceGrade.RIGOROUS_PRIMARY.value, SourceGrade.STANDARD_PRIMARY.value):
            if conf == "Low" or has_violations:
                return {
                    "rating": ReliabilityRating.MODERATE.value,
                    "badge": "🟡 Moderate Reliability",
                    "rating_label": "Qualified Primary",
                    "explanation": "Primary peer-reviewed source, but evaluation confidence is qualified by rubric warnings.",
                }
            return {
                "rating": ReliabilityRating.HIGH.value,
                "badge": "🟢 High Reliability",
                "rating_label": "High Reliability Primary",
                "explanation": "Verified finding supported by peer-reviewed primary literature with verbatim evidence.",
            }

        # Unknown or sparse source
        return {
            "rating": ReliabilityRating.MODERATE.value,
            "badge": "🟡 Moderate Reliability",
            "rating_label": "Moderate Reliability",
            "explanation": "Supported by text, but source publication metadata is unverified.",
        }

    # Fallback default
    return {
        "rating": ReliabilityRating.UNRESOLVED.value,
        "badge": "⚪ Unresolved",
        "rating_label": "Unresolved",
        "explanation": f"Unrecognized relation verdict: {rel}",
    }
