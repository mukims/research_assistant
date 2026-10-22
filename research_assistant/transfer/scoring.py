"""Scoring: gates, depth checklist, calibrated reward, and ranking."""

from __future__ import annotations

import re

from research_assistant.config import TRANSFER_CONTESTED_PENALTY, TRANSFER_PRIOR_ART_BONUS, TRANSFER_TUNE_PENALTY
from research_assistant.transfer.models import (
    Bottleneck,
    Card,
    MappingRow,
    Scores,
    Transfer,
)

DEPTH_CHECKS = (
    "mapping_complete",
    "rows_classified",
    "identifiability_argued",
    "steps_concrete",
    "data_quantified",
    "baseline_is_current",
    "prediction_directional",
    "refutable",
    "premises_both_sides",
    "objections_answered",
)

DIRECTION_RE = re.compile(
    r"\b(increase|decrease|higher|lower|larger|smaller|rise|fall|grow|drop|reduce|raise|faster|slower|exceed|more|less|improv|worse)\w*|\d",
    re.IGNORECASE,
)


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalise_phrase(text: str) -> str:
    """Lowercase words, hyphens as spaces, plurals made singular: "Wave-functions" -> "wave function"."""
    return " ".join(_singular(w) for w in re.findall(r"[a-z0-9]+", text.lower()))


def content_words(text: str) -> set[str]:
    """Extract lowercase words with letters only and length >= 4."""
    return {w.lower() for w in re.findall(r"[a-zA-Z]+", text) if len(w) >= 4}


def final_rows(transfer: Transfer) -> list[MappingRow]:
    """Apply the critic's row statuses over the writer's.

    A row the critic breaks becomes "contested" when the reviser answered the
    objection on that row without conceding it.
    """
    rows = [
        MappingRow(
            source_role=r.source_role,
            target_role=r.target_role,
            assumption=r.assumption,
            status=r.status,
            core=r.core,
            note=r.note,
        )
        for r in transfer.mapping
    ]
    if transfer.critique and transfer.critique.rows:
        rebutted = {
            o.get("row")
            for o in transfer.critique.objections
            if isinstance(o, dict) and o.get("answered") and not o.get("conceded")
        }
        for cr in transfer.critique.rows:
            if not isinstance(cr, dict):
                continue
            idx = cr.get("row")
            new_status = cr.get("status")
            if isinstance(idx, int) and 0 <= idx < len(rows) and new_status:
                if new_status.strip().lower() == "breaks" and idx in rebutted:
                    new_status = "contested"
                rows[idx].status = new_status
    return rows


def final_identifiability(transfer: Transfer) -> str:
    """Return the critic's identifiability override if given, else writer's verdict."""
    if transfer.critique and transfer.critique.identifiability:
        val = str(transfer.critique.identifiability).strip()
        if val:
            return val
    return transfer.identifiability.verdict.strip()


def set_aside_reason(transfer: Transfer, bottleneck: Bottleneck, card: Card) -> str | None:
    """
    Check gating criteria in priority order. Return first reason that fails, or None.
    """
    # 1. contradicted: any premise is judged Contradicts
    if any(p.judgement == "Contradicts" or getattr(p, "contradicted", False) for p in transfer.premises):
        return "contradicted"

    # 2. ungrounded_bottleneck: the bottleneck has no accepted or partial premise
    if not any(p.accepted or p.partial for p in bottleneck.premises):
        return "ungrounded_bottleneck"

    # 3. unverified_technique: the card's evidence premise did not pass
    if not (card.evidence.accepted or card.evidence.partial):
        return "unverified_technique"

    # 4. shape_mismatch: the critic says problem shapes do not match
    if transfer.critique and transfer.critique.shape_match.strip().lower() == "no":
        return "shape_mismatch"

    # 5. core_assumption_breaks: a core mapping row's final status is breaks
    f_rows = final_rows(transfer)
    if any(r.core and r.status.strip().lower() == "breaks" for r in f_rows):
        return "core_assumption_breaks"

    # 6. not_identifiable: the final identifiability verdict is underdetermined,
    #    for a bottleneck that infers unknowns from measurements
    if bottleneck.inference and final_identifiability(transfer).strip().lower() == "underdetermined":
        return "not_identifiable"

    # 7. off_target: metric names none of the bottleneck's quantities
    #    (ignoring case, hyphens and plurals)
    metric_norm = f" {normalise_phrase(transfer.metric)} "
    if not any(f" {normalise_phrase(q)} " in metric_norm for q in bottleneck.quantities if q.strip()):
        return "off_target"

    # 8. done_before: prior art found the same combination
    if transfer.prior_art and transfer.prior_art.status.strip().lower() == "done_before":
        return "done_before"

    return None


def depth_checks(transfer: Transfer, bottleneck: Bottleneck, card: Card) -> dict[str, bool]:
    """Check the ten depth items and return a dictionary of pass/fail for each."""
    checks: dict[str, bool] = {}

    # 1. Mapping complete: at least 3 rows, each with source role and target role
    checks["mapping_complete"] = (
        len(transfer.mapping) >= 3
        and all(bool(r.source_role.strip()) and bool(r.target_role.strip()) for r in transfer.mapping)
    )

    # 2. Rows classified: every row is holds, breaks or enforce, and at least one is core
    checks["rows_classified"] = (
        bool(transfer.mapping)
        and all(r.status.strip().lower() in ("holds", "breaks", "enforce") for r in transfer.mapping)
        and any(r.core for r in transfer.mapping)
    )

    # 3. Identifiability argued: measurements & unknowns named, verdict != unclear, argument >= 10 words.
    #    It does not apply when the bottleneck infers nothing from measurements.
    checks["identifiability_argued"] = not bottleneck.inference or (
        bool(transfer.identifiability.measurements.strip())
        and bool(transfer.identifiability.unknowns.strip())
        and transfer.identifiability.verdict.strip().lower() != "unclear"
        and len(transfer.identifiability.argument.split()) >= 10
    )

    # 4. Steps concrete: at least 3 steps, each naming input, output, tool
    checks["steps_concrete"] = (
        len(transfer.steps) >= 3
        and all(bool(s.input.strip()) and bool(s.output.strip()) and bool(s.tool.strip()) for s in transfer.steps)
    )

    # 5. Data quantified: contains a number
    checks["data_quantified"] = bool(re.search(r"\d", transfer.data_requirement))

    # 6. Baseline is current: shares a content word with current method
    checks["baseline_is_current"] = bool(
        content_words(transfer.baseline) & content_words(bottleneck.current_method)
    )

    # 7. Prediction directional: direction or number
    checks["prediction_directional"] = bool(DIRECTION_RE.search(transfer.prediction))

    # 8. Refutable: at least 5 words and differs from prediction
    checks["refutable"] = (
        len(transfer.refuting_result.split()) >= 5
        and transfer.refuting_result.strip().lower() != transfer.prediction.strip().lower()
    )

    # 9. Premises both sides: accepted/partial premise from card paper and from near paper
    checks["premises_both_sides"] = (
        any(p.document == card.document and (p.accepted or p.partial) for p in transfer.premises)
        and any(p.document != card.document and (p.accepted or p.partial) for p in transfer.premises)
    )

    # 10. Objections answered: at least one objection, each answered or conceded in >= 5 words
    checks["objections_answered"] = (
        bool(transfer.critique and transfer.critique.objections)
        and all(
            bool(o.get("answered")) and len(str(o.get("response", "")).split()) >= 5
            for o in transfer.critique.objections
        )
    )

    return checks


def reward(depth: float, prior_art_status: str) -> float:
    """Calculate reward from depth score and prior-art status."""
    if prior_art_status == "not_found":
        return depth * (1.0 + TRANSFER_PRIOR_ART_BONUS)
    return depth


def score_transfer(transfer: Transfer, bottleneck: Bottleneck, card: Card) -> Scores:
    """Score a transfer: compute depth checklist, check gates, and calculate reward."""
    checks = depth_checks(transfer, bottleneck, card)
    passed_count = sum(1.0 for ok in checks.values() if ok)
    depth = passed_count / len(DEPTH_CHECKS)
    set_aside = set_aside_reason(transfer, bottleneck, card)
    contested = sum(1 for r in final_rows(transfer) if r.core and r.status == "contested")
    if set_aside is not None:
        rew = 0.0
    else:
        rew = reward(depth, transfer.prior_art.status) * max(0.0, 1.0 - TRANSFER_CONTESTED_PENALTY * contested)
        # A transfer that only tunes the current method is worth less than one that changes it.
        if transfer.critique and transfer.critique.kind == "tune":
            rew *= 1.0 - TRANSFER_TUNE_PENALTY
    scores = Scores(depth=depth, checks=checks, reward=rew, set_aside=set_aside, contested=contested)
    transfer.scores = scores
    return scores


def rank(transfers: list[Transfer]) -> list[Transfer]:
    """Sort transfers by reward descending; set-aside transfers go last (stable sort)."""
    return sorted(
        transfers,
        key=lambda t: (
            t.scores.set_aside is not None if t.scores else True,
            -(t.scores.reward if t.scores else 0.0),
        ),
    )
