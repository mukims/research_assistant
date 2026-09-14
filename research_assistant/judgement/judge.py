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
import unicodedata
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

# Step 2 of the rubric: each slot has its own vocabulary. The model reply is
# validated at the top level only (VALID_JUDGEMENTS etc.); this is per slot.
SLOT_VOCAB = {
    "finding":  {"Supports", "Contradicts", "Does not support", "Insufficient"},
    "scope":    {"Supports", "Partially supports", "Does not support", "Insufficient"},
    "strength": {"Supports", "Partially supports", "Insufficient", "Not applicable"},
}

# Added by enforce_rubric(); merged into the verifier's record alongside
# REQUIRED_FIELDS. Additive: no existing field changes meaning.
DERIVED_FIELDS = {
    "model_judgement", "rubric_mismatch", "rubric_violations", "span_verified", "span_cites_others",
}

_CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}


def _slot_verdict(slots: dict, name: str) -> str:
    """The verdict as the rules see it: anything outside the slot's vocabulary
    is read as 'Insufficient' — the model did not follow the rubric here, and
    the conservative reading of that is 'cannot tell'."""
    verdict = slots[name]["verdict"]
    return verdict if verdict in SLOT_VOCAB[name] else "Insufficient"


def derive_judgement(slots: dict) -> str:
    """Step 3 of the rubric, in code. First match wins."""
    finding, scope, strength = (_slot_verdict(slots, k) for k in ("finding", "scope", "strength"))
    if scope == "Does not support":
        return "Does not support"
    if finding == "Contradicts":
        return "Contradicts"
    if finding == "Does not support":
        return "Does not support"
    if finding == "Insufficient" or scope == "Insufficient":
        return "Unclear / insufficient evidence"
    if all(v in ("Supports", "Not applicable") for v in (finding, scope, strength)):
        return "Supports"
    return "Partially supports"


_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})
_ELLIPSIS_RE = re.compile(r"(\.\.\.|…|\[\s*\.\.\.\s*\]|\[…\])")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    text = _ELLIPSIS_RE.sub(" ", text)
    return " ".join(text.lower().split())


def span_is_verbatim(span, evidence: str):
    """Is the supporting span a substring of the evidence, up to whitespace,
    case, ligatures, quote style and ellipses? None when there is no span."""
    if span is None or not str(span).strip() or str(span).strip().lower() == "null":
        return None
    return _normalise(str(span)) in _normalise(evidence)


# A supporting sentence that itself carries a citation — "[12]", "Ref. 7",
# "Smith et al.", or GROBID's spaced superscript "proved 18 ." — is the cited
# paper attributing the statement to someone else. The verdict stands; the
# support is secondhand and the reliability policy treats it as such.
_CITES_OTHERS_RE = re.compile(
    r"(\[\d+[^\]]*\]|\bRefs?\.\s*\d|\bet al\.|[A-Za-z]\s\d{1,3}\s[.,;])"
)


def span_cites_others(span) -> bool | None:
    """Does the supporting span attribute its statement to another work?
    None when there is no span."""
    if span is None or not str(span).strip() or str(span).strip().lower() == "null":
        return None
    return bool(_CITES_OTHERS_RE.search(str(span)))


def enforce_rubric(result: dict, evidence: str) -> dict:
    """Make the record say what the rubric says, and where the model differed.

    The model's stated judgement is kept as model_judgement; `judgement`
    becomes the one the Step-3 rules derive from its own slots. Slot verdicts
    outside their vocabulary are listed in rubric_violations. The supporting
    span is checked verbatim. Any of those three caps confidence at Medium:
    the model's High was self-reported about a reply that broke its rules.

    One more rule the slots cannot express: "Does not support" asserts that
    the paper reports nothing about the relationship, and the judge only ever
    sees a few chunks of it. When it also rates that evidence "insufficient"
    — too fragmentary to assess — it has not established absence, only that
    the retrieved passages were silent. That is a retrieval gap, and the
    record says "Unclear", not "the paper does not say this".
    """
    slots = result["slots"]
    violations = [
        f"{name}: {slots[name]['verdict']!r}"
        for name in ("finding", "scope", "strength")
        if slots[name]["verdict"] not in SLOT_VOCAB[name]
    ]
    derived = derive_judgement(slots)
    if derived == "Does not support" and result.get("evidence_sufficiency") == "insufficient":
        violations.append(
            "absence asserted from insufficient evidence: retrieved passages were silent, "
            "which does not show the paper is"
        )
        derived = "Unclear / insufficient evidence"
    result["model_judgement"] = result["judgement"]
    result["rubric_mismatch"] = derived != result["judgement"]
    result["rubric_violations"] = violations
    result["judgement"] = derived
    result["span_verified"] = span_is_verbatim(result.get("supporting_span"), evidence)
    result["span_cites_others"] = span_cites_others(result.get("supporting_span"))

    if violations or result["rubric_mismatch"] or result["span_verified"] is False:
        if _CONFIDENCE_RANK[result["confidence"]] > _CONFIDENCE_RANK["Medium"]:
            result["confidence"] = "Medium"
    return result

_FENCE_RE = re.compile(r"\A```(?:json)?\s*\n(.*?)\n?```\Z", re.DOTALL)


class JudgementParseError(ValueError):
    """The model's reply was not a usable judgement."""

    def __init__(self, message, raw=""):
        super().__init__(message)
        self.raw = raw


_CONTEXT_HEADER = (
    "**Context** (the sentences around the claim, from the citing paper; the claim is the "
    "sentence between « and »; judge only that sentence, and use the rest only to resolve "
    "what its words refer to — never as evidence):\n"
)


def build_prompt(claim: str, citation_evidence: str, context: str | None = None) -> str:
    """Fill the prompt input template. The context block is present only when
    there is context, so the six regression cases render exactly as before."""
    context_block = f"{_CONTEXT_HEADER}{context.strip()}\n\n" if context and context.strip() else ""
    return (
        PROMPT_TEMPLATE
        .replace("{{CONTEXT_BLOCK}}", context_block)
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


def judge(claim: str, citation_evidence: str, model: str | None = None, context: str | None = None) -> dict:
    """Verdict on whether *citation_evidence* supports *claim*."""
    result = chat(
        [{"role": "user", "content": build_prompt(claim, citation_evidence, context=context)}],
        model=model or JUDGEMENT_MODEL,
        temperature=JUDGEMENT_TEMPERATURE,
        options=JUDGEMENT_OLLAMA_OPTIONS,
    )
    if not result.content:
        raise JudgementParseError("LLM returned an empty response.", raw="")
    return enforce_rubric(parse_judgement(result.content), citation_evidence)
