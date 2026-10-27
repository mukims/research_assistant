"""Evidence assembly and preparation for citation judgement.

Provides hit assembly, neighbor chaining, context window sanity checks,
and escalation predicate checks for claim-evidence verification.
"""

from research_assistant.config import (
    JUDGEMENT_MODEL,
    JUDGEMENT_OLLAMA_OPTIONS,
    LLM_BACKEND,
    LLM_MODEL,
)
from research_assistant.shared.log import get_logger

logger = get_logger("judgement.evidence")

_HIT_SEPARATOR = "\n\n[…]\n\n"


def _judgement_model() -> str:
    """The model that actually produces the verdicts."""
    return JUDGEMENT_MODEL or LLM_MODEL


def _warn_if_context_is_tight():
    """A local model with a small context truncates the prompt silently."""
    if LLM_BACKEND != "ollama":
        return
    from research_assistant.judgement.judge import PROMPT_TEMPLATE

    approx_tokens = len(PROMPT_TEMPLATE) // 4
    logger.warning(
        "Judging with local model %s. The prompt alone is ~%d tokens and "
        "num_ctx is set to %s — if the model cannot honour that, Ollama "
        "truncates the prompt tail (where the worked examples are) without "
        "raising, and verdicts degrade silently.",
        _judgement_model(),
        approx_tokens,
        JUDGEMENT_OLLAMA_OPTIONS.get("num_ctx"),
    )


def assemble_evidence(hits, max_chars: int) -> str:
    """The text the judge sees: each hit wrapped in its neighbours, hits in
    rank order, separated, de-duplicated, capped from the tail so the top hit
    is never the one cut."""
    parts, seen = [], set()
    for h in hits:
        block = "\n".join(
            p
            for p in (
                h.get("context_before", ""),
                h.get("text", ""),
                h.get("context_after", ""),
            )
            if p
        )
        if block in seen:
            continue
        seen.add(block)
        parts.append(block)
    out = _HIT_SEPARATOR.join(parts)
    return out[:max_chars]


def needs_escalation(verdict: dict) -> bool:
    """The first verdict says it did not see enough — by its own sufficiency
    rating, or by landing on a judgement that more evidence could overturn."""
    return (
        verdict.get("evidence_sufficiency") != "sufficient"
        or verdict.get("judgement") in ("Unclear / insufficient evidence", "Does not support")
    )
