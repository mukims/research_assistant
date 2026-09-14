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
    UNSUPPORTED = "UNSUPPORTED"
    UNRESOLVED = "UNRESOLVED"


# One sentence per way a citation can end up not judged. The rating for all of
# them is UNRESOLVED; the explanation is what distinguishes "the paper was
# silent" from "we never looked".
NOT_ASSESSED_EXPLANATIONS = {
    "cap_exceeded": "Not assessed — the per-paper claim budget was reached before this citation.",
    "not_attempted": "Not assessed — the audit stopped early because the model backend was unreachable.",
    "retrieval_failed": "Not assessed — retrieval from the cited paper failed.",
    "call_failed": "Not assessed — the model backend returned an error.",
    "parse_failed": "Not assessed — the model reply could not be parsed.",
    "no_evidence": "Not assessed — search found no passages in the cited paper for this sentence.",
    "cluster_skipped": "Not assessed — the sentence cites many papers; only the first few were judged.",
    "not_a_claim": "Not assessed — this citation is not a verifiable claim about the cited paper (software, pointer, or method reference).",
    "malformed_claim": "Not assessed — the extracted sentence is a fragment and could not be judged.",
    "unresolved_ref": "Not assessed — the citation could not be matched to a bibliography entry.",
    # Agent 8 (draft verification) outcomes.
    "orphaned": "Not assessed — the citation key in the draft is not in the citation mapping.",
    "unresolved": "Not assessed — the cited source has no chunks in the corpus.",
    "not_downloaded": "Not assessed — the cited paper is not in the corpus.",
    "deferred_paywalled": "Not assessed — deferred because most references in this paragraph are missing.",
}


def evaluate_reliability(
    relation: str | None,
    source_grade: str | None = None,
    confidence: str | None = None,
    span_verified: bool | None = None,
    rubric_violations: list[str] | None = None,
    rubric_mismatch: bool = False,
    outcome: str = "judged",
) -> dict[str, Any]:
    """Derive scientific reliability from the relation verdict and source grade.

    A rating is a statement about a verdict. When there is no verdict
    (*outcome* != "judged") the rating is UNRESOLVED and the explanation says
    why nothing was judged — never that the evidence was insufficient.
    """
    source_grade = source_grade or SourceGrade.UNKNOWN.value
    conf = (confidence or "High").capitalize()
    has_violations = bool(rubric_violations) or rubric_mismatch

    # 0. Nothing was judged.
    if outcome != "judged" or not relation:
        return {
            "rating": ReliabilityRating.UNRESOLVED.value,
            "badge": "⏳ Not Assessed",
            "rating_label": "Not Assessed",
            "explanation": NOT_ASSESSED_EXPLANATIONS.get(
                outcome, f"Not assessed — {outcome}."
            ),
        }
    rel = relation.strip()

    # 1. The judge could not decide from what it saw.
    if rel == "Unclear / insufficient evidence":
        return {
            "rating": ReliabilityRating.UNRESOLVED.value,
            "badge": "⚪ Unresolved",
            "rating_label": "Unresolved Evidence",
            "explanation": "The judge could not decide from the retrieved passages whether the cited paper supports this sentence.",
        }

    # 1b. The paper was read and does not say this.
    if rel == "Does not support":
        return {
            "rating": ReliabilityRating.UNSUPPORTED.value,
            "badge": "🟠 Unsupported",
            "rating_label": "Unsupported Citation",
            "explanation": "The cited paper does not report the finding or the conditions this sentence attributes to it.",
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
