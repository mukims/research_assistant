"""Card review sheet generation and Gate F1 scoring."""

from __future__ import annotations

from typing import Any

from research_assistant.transfer.cards import CardIndex


def review_sheet(index: CardIndex, documents: list[str]) -> tuple[str, str]:
    """
    Generate markdown review sheet and CSV rating template for cards from specified documents.

    Returns:
        (markdown_text, csv_text)
    """
    doc_set = set(documents)
    cards = [c for c in index.cards if c.document in doc_set]

    md_blocks = []
    csv_lines = ["card_id,rating,specific"]

    for c in cards:
        span_str = c.evidence.supporting_span or "N/A"
        assumptions_str = ", ".join(c.assumptions) if c.assumptions else "none"
        mechanism_str = "".join(
            f"   {i}. {step}\n" for i, step in enumerate(c.mechanism, start=1)
        ) or "   none\n"
        block = f"""## Card {c.card_id}
- **Paper:** {c.citation} ({c.document})
- **Technique:** {c.name}
- **Problem Solved:** {c.problem_solved}
- **Context:** {c.context}
- **Inputs:** {c.inputs}
- **Outputs:** {c.outputs}
- **Assumptions:** {assumptions_str}
- **Mechanism:**
{mechanism_str}- **Evidence Claim:** "{c.evidence.claim}" (Key: {c.evidence.key})
- **Supporting Span:** "{span_str}"
- **Verdict:** {c.evidence.judgement} (Verified: {c.evidence.span_verified})
"""
        md_blocks.append(block)
        csv_lines.append(f"{c.card_id},,")

    return "\n".join(md_blocks), "\n".join(csv_lines)


def score_review(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Score a completed review sheet CSV.

    Rating: correct, partial, wrong
    Specific: y, n
    Blank rows are ignored.
    Passed: correct_or_partial >= 0.70 and specific >= 0.60
    """
    valid_rows = [
        r for r in rows
        if str(r.get("rating", "")).strip() != ""
    ]

    n_rated = len(valid_rows)
    if n_rated == 0:
        return {
            "rated": 0,
            "correct_or_partial": 0.0,
            "specific": 0.0,
            "passed": False,
        }

    n_cop = sum(
        1 for r in valid_rows
        if str(r.get("rating", "")).strip().lower() in ("correct", "partial")
    )
    n_spec = sum(
        1 for r in valid_rows
        if str(r.get("specific", "")).strip().lower() == "y"
    )

    cop_ratio = round(n_cop / n_rated, 4)
    spec_ratio = round(n_spec / n_rated, 4)
    passed = bool(cop_ratio >= 0.70 and spec_ratio >= 0.60)

    return {
        "rated": n_rated,
        "correct_or_partial": cop_ratio,
        "specific": spec_ratio,
        "passed": passed,
    }
