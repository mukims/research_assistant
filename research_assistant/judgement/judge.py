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
    GEMINI_API_KEY,
    LLM_BACKEND,
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

# Verdicts that assert the passages report something, and so rest on the
# supporting span. "Does not support" asserts the opposite and has its own
# guard; "Unclear" asserts nothing.
_REST_ON_THE_SPAN = frozenset({"Supports", "Partially supports", "Contradicts"})


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
# PDF text keeps the hyphen of a word split at a line break, either left open
# ("energy- dependent") or closed up ("informa-tion"), and a model quoting that
# sentence writes the word whole. Dropping every hyphen on both sides makes the
# two agree without loosening the check: the span still has to be the evidence's
# own words, in the evidence's own order, for its whole length — an invented
# sentence opening does not survive it because a hyphen was never the difference.
_LINE_HYPHEN_RE = re.compile(r"-\s*")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    text = _ELLIPSIS_RE.sub(" ", text)
    text = _LINE_HYPHEN_RE.sub("", text)
    return " ".join(text.lower().split())


_SPAN_PIECE_MIN_CHARS = 12


def span_is_verbatim(span, evidence: str):
    """Is the supporting span a substring of the evidence, up to whitespace,
    case, ligatures and quote style? A span the model joined with an
    ellipsis counts when every piece of it is verbatim — grounded is what
    the check is for, contiguity is not. None when there is no span."""
    if span is None or not str(span).strip() or str(span).strip().lower() == "null":
        return None
    haystack = _normalise(evidence)
    pieces = [piece.strip() for piece in _ELLIPSIS_RE.split(str(span)) if piece and piece.strip()]
    pieces = [piece for piece in pieces if len(piece) >= _SPAN_PIECE_MIN_CHARS] or pieces
    return all(_normalise(piece) in haystack for piece in pieces)


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


# Words that carry no content for the drift check: the model rephrases
# freely around them ("the value is achieved", "the improvement occurs").
_DRIFT_STOPWORDS = frozenset("""
the a an and or of in on at to for with by from as is are was were be been being
this that these those it its their there than then which who whom whose what when
where while into onto over under between within without about above below across
does do did done has have had having not no nor can could may might will would
shall should must also only just more most less least very much many such same
other another each every both either neither all any some few several own per
""".split())
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9@_'-]{2,}")
DRIFT_MIN_OVERLAP = 0.4


_SUFFIXES = ("ies", "ing", "es", "ed", "er", "s")


def _stem(word: str) -> str:
    """Just enough to let reach/reaches, capacitance/capacitances,
    large/larger agree."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            base = word[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return word


def _content_words(text: str) -> set:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return {_stem(w) for w in _WORD_RE.findall(text) if w not in _DRIFT_STOPWORDS}


def assertion_drift(assertion: str, *sources: str) -> float | None:
    """Fraction of the assertion's content words that occur in any of the
    *sources* (after NFKC folding and light stemming). None when the
    assertion has no content words to compare."""
    words = _content_words(assertion)
    if not words:
        return None
    pool = set().union(*(_content_words(src) for src in sources))
    return sum(1 for w in words if w in pool) / len(words)


def enforce_rubric(result: dict, evidence: str, claim: str | None = None) -> dict:
    """Make the record say what the rubric says, and where the model differed.

    The model's stated judgement is kept as model_judgement; `judgement`
    becomes the one the Step-3 rules derive from its own slots. Slot verdicts
    outside their vocabulary are listed in rubric_violations. The supporting
    span is checked verbatim. Any of those three caps confidence at Medium:
    the model's High was self-reported about a reply that broke its rules.

    With *claim* given, the decomposition is checked against it: a finding
    assertion that shares fewer than DRIFT_MIN_OVERLAP of its content words
    with the claim was decomposed from somewhere else — seen when the model
    reads the context sentence instead of the marked one. The scope
    assertion may legitimately name the evidence's system where the claim
    is vague (rubric example F), so it is checked against claim and
    evidence together. The verdict stands; the drift is recorded and
    confidence capped.

    Two more rules the slots cannot express. "Does not support" asserts that
    the paper reports nothing about the relationship, and the judge only ever
    sees a few chunks of it. When it also rates that evidence "insufficient"
    — too fragmentary to assess — it has not established absence, only that
    the retrieved passages were silent. That is a retrieval gap, and the
    record says "Unclear", not "the paper does not say this".

    Its mirror image is support asserted from a span that is not in the
    evidence. Supports, Partially supports and Contradicts all assert that
    the passages report something, and the span is where the model shows
    which sentence does. When that sentence does not occur in the evidence
    the verdict rests on nothing: across two live audits every unverified
    span was under 40% verbatim, and a local model quoted the citing
    paper's own sentence back with its citation markers intact. Capping
    confidence left such a verdict reading as a green tick, so it is
    recorded as unclear instead.
    """
    slots = result["slots"]
    violations = [
        f"{name}: {slots[name]['verdict']!r}"
        for name in ("finding", "scope", "strength")
        if slots[name]["verdict"] not in SLOT_VOCAB[name]
    ]
    if claim:
        for name, sources in (("finding", (claim,)), ("scope", (claim, evidence))):
            overlap = assertion_drift(str(slots[name].get("assertion") or ""), *sources)
            if overlap is not None and overlap < DRIFT_MIN_OVERLAP:
                violations.append(
                    f"{name}: assertion drift — {overlap:.0%} of its content words occur in the claim"
                )
    derived = derive_judgement(slots)
    span_verified = span_is_verbatim(result.get("supporting_span"), evidence)
    if derived == "Does not support" and result.get("evidence_sufficiency") == "insufficient":
        violations.append(
            "absence asserted from insufficient evidence: retrieved passages were silent, "
            "which does not show the paper is"
        )
        derived = "Unclear / insufficient evidence"
    elif derived in _REST_ON_THE_SPAN and span_verified is False:
        violations.append(
            "support asserted from a span that is not in the evidence: the quoted sentence does "
            "not occur in the retrieved passages, so nothing shows they report this"
        )
        derived = "Unclear / insufficient evidence"
    result["model_judgement"] = result["judgement"]
    result["rubric_mismatch"] = derived != result["judgement"]
    result["rubric_violations"] = violations
    result["judgement"] = derived
    result["span_verified"] = span_verified
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
    "**Context** (from the citing paper: the section the claim sits in, the captions of figures "
    "and tables its paragraph points at, and the sentences around it; the claim is the sentence "
    "between « and »; judge only that sentence, and use the rest only to resolve what its words "
    "refer to — never as evidence):\n"
)

ARTIFACT_CAPTION_CHARS = 200


def compose_context(window: str | None, section: str | None = None, artifacts: list | None = None) -> str | None:
    """The judge's context block from its parts: a [Section: …] header, one
    line per figure or table the paragraph points at, then the sentence
    window with the claim marked. None without a window — a header alone
    tells the judge nothing about the claim. The cited paper's summary is
    deliberately not a part: it is model prose about the paper the
    evidence comes from, and the grounding rule forbids using it as such."""
    if not window or not str(window).strip():
        return None
    lines = []
    if section and str(section).strip():
        lines.append(f"[Section: {str(section).strip()}]")
    for a in artifacts or []:
        caption = " ".join(str(a.get("caption") or "").split())[:ARTIFACT_CAPTION_CHARS]
        if a.get("label") and caption:
            lines.append(f"[{a['label']}: {caption}]")
    lines.append(str(window).strip())
    return "\n".join(lines)


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


def judge(
    claim: str,
    citation_evidence: str,
    model: str | None = None,
    context: str | None = None,
    backend: str | None = None,
) -> dict:
    """Verdict on whether *citation_evidence* supports *claim*.

    A Gemini key routes the judgement to Gemini's OpenAI-compatible endpoint —
    but only as a default. An operator who sets LLM_BACKEND (to run the audit
    on a local Ollama model, say) means it: a key left in the environment does
    not quietly send the claims to a hosted API. config.LLM_BACKEND is already
    "openai" whenever the key is present and nothing else was asked for.
    """
    effective_backend = backend or (LLM_BACKEND if GEMINI_API_KEY else None)
    gemini = (effective_backend or LLM_BACKEND) == "openai" and GEMINI_API_KEY
    effective_model = model or JUDGEMENT_MODEL or ("gemini-3.5-flash-lite" if gemini else None)
    result = chat(
        [{"role": "user", "content": build_prompt(claim, citation_evidence, context=context)}],
        model=effective_model,
        temperature=JUDGEMENT_TEMPERATURE,
        options=JUDGEMENT_OLLAMA_OPTIONS,
        backend=effective_backend,
    )
    if not result.content:
        raise JudgementParseError("LLM returned an empty response.", raw="")
    return enforce_rubric(parse_judgement(result.content), citation_evidence, claim=claim)
