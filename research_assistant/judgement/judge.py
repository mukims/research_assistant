"""Claim–evidence verification as a pipeline component.

The prompt and its regression cases came from a standalone module that spoke
to OpenAI directly. Routing through shared.llm instead is what lets the same
judgement run against a local Ollama model and a hosted API without the caller
knowing which — the same reason every agent in this repo calls chat().

Parsing is stricter than the source module's bare json.loads. The prompt says
"no code fences" and models emit them anyway, and a near-miss category like
"supports" would otherwise flow through as an unrecognised verdict and quietly
distort the verifier's summary counts.
"""

import json
import re
from pathlib import Path

from research_assistant.config import (
    JUDGEMENT_MODEL,
    JUDGEMENT_OLLAMA_OPTIONS,
    JUDGEMENT_TEMPERATURE,
)
from research_assistant.shared.llm import chat
from research_assistant.shared.log import get_logger

logger = get_logger("judgement")

PROMPT_PATH = Path(__file__).parent / "prompt.md"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")

VALID_JUDGEMENTS = {
    "Supports",
    "Partially supports",
    "Contradicts",
    "Does not support",
    "Unclear / insufficient evidence",
}
VALID_CONFIDENCE = {"High", "Medium", "Low"}
VALID_SUFFICIENCY = {"sufficient", "partial", "insufficient"}

REQUIRED_FIELDS = {
    "slots",
    "judgement",
    "evidence_sufficiency",
    "confidence",
    "supporting_span",
    "reason",
}
REQUIRED_SLOTS = {"finding", "scope", "strength"}

_FENCE_RE = re.compile(r"\A```(?:json)?\s*\n(.*?)\n?```\Z", re.DOTALL)


class JudgementParseError(ValueError):
    """The model's reply was not a usable judgement."""

    def __init__(self, message, raw=""):
        super().__init__(message)
        self.raw = raw


def build_prompt(claim: str, citation_evidence: str) -> str:
    """Fill the prompt input template."""
    return (
        PROMPT_TEMPLATE
        .replace("{{CLAIM}}", claim)
        .replace("{{CITATION_EVIDENCE}}", citation_evidence)
    )


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1) if match else stripped


def _validate(result, raw):
    if not isinstance(result, dict):
        raise JudgementParseError(
            f"Expected a JSON object, got {type(result).__name__}", raw=raw
        )

    missing = REQUIRED_FIELDS - result.keys()
    if missing:
        raise JudgementParseError(f"Missing output fields: {sorted(missing)}", raw=raw)

    for field, allowed in (
        ("judgement", VALID_JUDGEMENTS),
        ("confidence", VALID_CONFIDENCE),
        ("evidence_sufficiency", VALID_SUFFICIENCY),
    ):
        if result[field] not in allowed:
            raise JudgementParseError(
                f"{field} was {result[field]!r}, expected one of {sorted(allowed)}",
                raw=raw,
            )

    slots = result["slots"]
    if not isinstance(slots, dict) or set(slots) != REQUIRED_SLOTS:
        got = sorted(slots) if isinstance(slots, dict) else repr(slots)
        raise JudgementParseError(
            f"slots must be exactly {sorted(REQUIRED_SLOTS)}, got {got}", raw=raw
        )

    for name, slot in slots.items():
        if not isinstance(slot, dict) or "assertion" not in slot or "verdict" not in slot:
            raise JudgementParseError(
                f"slots[{name!r}] must be an object with 'assertion' and "
                f"'verdict', got {slot!r}",
                raw=raw,
            )
        if not isinstance(slot["verdict"], str):
            raise JudgementParseError(
                f"slots[{name!r}]['verdict'] must be a string, "
                f"got {type(slot['verdict']).__name__}",
                raw=raw,
            )

    return result


def parse_judgement(raw: str) -> dict:
    """Parse and validate one model reply. Raises JudgementParseError."""
    try:
        result = json.loads(_strip_fence(raw))
    except json.JSONDecodeError as exc:
        raise JudgementParseError(f"Reply was not valid JSON: {exc}", raw=raw) from exc
    return _validate(result, raw)


def judge(claim: str, citation_evidence: str, model: str | None = None) -> dict:
    """Verdict on whether *citation_evidence* supports *claim*."""
    result = chat(
        [{"role": "user", "content": build_prompt(claim, citation_evidence)}],
        model=model or JUDGEMENT_MODEL,
        temperature=JUDGEMENT_TEMPERATURE,
        options=JUDGEMENT_OLLAMA_OPTIONS,
    )
    if not result.content:
        raise JudgementParseError("LLM returned an empty response.", raw="")
    return parse_judgement(result.content)
