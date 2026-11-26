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
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_KEY,
    LLM_BACKEND,
    LLM_MODEL,
    JUDGEMENT_MODEL,
    JUDGEMENT_OLLAMA_OPTIONS,
    JUDGEMENT_SELECTIVE_VOTES,
    JUDGEMENT_TEMPERATURE,
    JUDGEMENT_VOTES,
)
from research_assistant.shared.llm import chat
from research_assistant.shared.log import get_logger
from research_assistant.shared.retry import retry

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

# Added by judge() itself: which model produced the verdict, and the vote that
# produced it. `votes` / `vote_split` are absent from a single-vote verdict, so
# every copy site takes these only when present (`if k in verdict`).
PROVENANCE_FIELDS = ("judge_model", "judge_model_version", "votes", "vote_split")

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


_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-", "\u2212": "-"})
_ELLIPSIS_RE = re.compile(r"(\.\.\.|…|\[\s*\.\.\.\s*\]|\[…\])")
# PDF text keeps the hyphen of a word split at a line break, either left open
# ("energy- dependent") or closed up ("informa-tion"), and a model quoting that
# sentence writes the word whole. Matching hyphens between letters unifies line
# breaks and hyphenated words while strictly preserving minus signs preceding
# numbers (e.g. -0.5 eV, 10^-5) so negative quantities are not sign-inverted.
_LINE_HYPHEN_RE = re.compile(r"(?<=[a-zA-Z])-\s*(?=[a-zA-Z])")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:!?)\]]|\.(?=\s|$))")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    text = _ELLIPSIS_RE.sub(" ", text)
    text = _LINE_HYPHEN_RE.sub("", text)
    text = " ".join(text.lower().split())
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    return text


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


def infer_evidence_tier(hits: list[dict], prior_tier: str | None = None) -> str:
    """Determine whether *hits* represent ``'abstract_only'`` or ``'full_text'`` evidence."""
    def _hit_is_abs(h: dict) -> bool:
        m = h.get("metadata") or {}
        return (
            m.get("tier") == "abstract_only"
            or m.get("stored_form") == "abstract_only"
            or m.get("type") == "abstract_only"
            or m.get("evidence_tier") == "abstract_only"
        )

    if hits and all(_hit_is_abs(h) for h in hits):
        return "abstract_only"
    if prior_tier == "abstract_only":
        has_full = any(
            (h.get("metadata") or {}).get("tier") == "full_text"
            or (h.get("metadata") or {}).get("stored_form") in ("text_chunk", "table", "caption")
            for h in hits
        )
        if not has_full:
            return "abstract_only"
    return "full_text"


def guard_abstract_only_verdict(verdict: dict, evidence_tier: str | None) -> dict:
    """Prevent asserting absence ('Does not support') when only a paper's abstract was available.

    A 150-word abstract rarely lists computational parameters, benchmark table rows, or
    secondary controls; silence in an abstract is 'Unclear / insufficient evidence',
    never proof that the full paper omits the finding.
    """
    if evidence_tier != "abstract_only" or not isinstance(verdict, dict):
        return verdict
    if verdict.get("judgement") != "Does not support":
        return verdict
    updated = dict(verdict)
    updated["model_judgement"] = updated.get("model_judgement") or "Does not support"
    updated["rubric_mismatch"] = True
    violations = list(updated.get("rubric_violations") or [])
    violations.append(
        "absence asserted from abstract-only evidence: only the abstract was available, "
        "which cannot establish that the full paper omits this"
    )
    updated["rubric_violations"] = violations
    updated["judgement"] = "Unclear / insufficient evidence"
    base_reason = str(updated.get("reason") or "").rstrip()
    note = "[Only the cited paper's abstract was available; full text is required to verify absence.]"
    updated["reason"] = f"{base_reason} {note}".strip()
    return updated

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
_ARTIFACT_LABEL_RE = re.compile(
    r"^\s*(fig(?:ure)?s?\.?|tab(?:le)?s?\.?)\s*([A-Za-z0-9]+)\s*$",
    re.IGNORECASE,
)


def filter_relevant_artifacts(claim_text: str | None, artifacts: list | None) -> list:
    """Keep only figure/table captions explicitly referenced in *claim_text*.

    Paragraph-level ``artifacts`` often carry captions for four different figures
    discussed across a long paragraph; injecting unreferenced captions distracted
    the judge in 25d6fd9. Matching the exact figure/table identifier mentioned in
    the claim sentence itself gives figure-referencing claims the caption they
    point at while leaving plain prose claims clean.
    """
    if not claim_text or not artifacts:
        return []
    text = str(claim_text)
    selected = []
    for art in artifacts:
        if not isinstance(art, dict):
            continue
        label = str(art.get("label") or "").strip()
        caption = str(art.get("caption") or "").strip()
        if not label or not caption:
            continue
        m = _ARTIFACT_LABEL_RE.match(label)
        if not m:
            if re.search(rf"\b{re.escape(label)}\b", text, re.IGNORECASE):
                selected.append(art)
            continue
        kind_raw, ident = m.group(1).lower(), m.group(2)
        if kind_raw.startswith("fig"):
            # Match 'Fig. 2', 'Fig. 2b', 'Figure 2(a)', or plural 'Figs. 1 and 2' / 'Figs. 1-3'
            pat = re.compile(
                rf"\bfig(?:ure)?s?\.?\s*(?:[A-Za-z0-9\(\)]+\s*(?:,|and|&|[-–—])\s*)*"
                rf"{re.escape(ident)}(?:[a-z]|\([a-z0-9]+\))?\b",
                re.IGNORECASE,
            )
        else:
            pat = re.compile(
                rf"\btab(?:le)?s?\.?\s*(?:[A-Za-z0-9]+\s*(?:,|and|&|[-–—])\s*)*"
                rf"{re.escape(ident)}(?:[a-z]|\([a-z0-9]+\))?\b",
                re.IGNORECASE,
            )
        if pat.search(text):
            selected.append(art)
    return selected


def build_citing_paper_note(
    claim_text: str | None,
    seed_title: str | None = None,
    seed_self_names: list[str] | None = None,
) -> str | None:
    """Build a scoping header when *claim_text* names the citing manuscript's own system/framework.

    Prevents the judge from demanding that a cited reference (e.g. a protocol or
    dataset paper) mention the citing manuscript's newly coined framework name
    (e.g. 'Paper2Agent represents a paper as an MCP server [17]').
    """
    if not claim_text:
        return None
    matched_names = [
        nm for nm in (seed_self_names or [])
        if nm and re.search(rf"\b{re.escape(nm)}\b", claim_text)
    ]
    if not matched_names:
        return None
    names_str = ", ".join(f"'{n}'" for n in matched_names)
    title_part = f" (\"{seed_title}\")" if seed_title else ""
    return (
        f"[Citing Paper Self-Identity: The citing manuscript{title_part} introduces {names_str} "
        f"as its own work. Do NOT require the cited reference to mention {names_str}; evaluate "
        "the cited reference only on the underlying concept, protocol, method, or prior finding "
        "that the sentence attributes to it.]"
    )


def compose_context(
    window: str | None,
    section: str | None = None,
    artifacts: list | None = None,
    citing_paper_note: str | None = None,
) -> str | None:
    """The judge's context block from its parts: a [Section: …] header,
    optional citing-paper self-identity note, one line per figure or table
    referenced, then the sentence window with the claim marked. None without
    a window — a header alone tells the judge nothing about the claim."""
    if not window or not str(window).strip():
        return None
    lines = []
    if section and str(section).strip():
        lines.append(f"[Section: {str(section).strip()}]")
    if citing_paper_note and str(citing_paper_note).strip():
        lines.append(str(citing_paper_note).strip())
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


_SELECTIVE_VOTE_TRIGGERS = frozenset({"Contradicts", "Does not support", "Partially supports"})


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
    effective_model = model or JUDGEMENT_MODEL or (DEFAULT_GEMINI_MODEL if gemini else None)
    messages = [{"role": "user", "content": build_prompt(claim, citation_evidence, context=context)}]
    votes = [
        _judge_vote(messages, effective_model, effective_backend, citation_evidence, claim)
        for _ in range(JUDGEMENT_VOTES)
    ]
    if len(votes) == 1:
        if JUDGEMENT_SELECTIVE_VOTES and votes[0]["judgement"] in _SELECTIVE_VOTE_TRIGGERS:
            try:
                v1 = _judge_vote(messages, effective_model, effective_backend, citation_evidence, claim)
                v2 = _judge_vote(messages, effective_model, effective_backend, citation_evidence, claim)
                return _tally([votes[0], v1, v2])
            except StopIteration:
                # Single-reply test mocks keep their single vote
                return votes[0]
        return votes[0]
    return _tally(votes)


def _judge_vote(messages, effective_model, effective_backend, citation_evidence, claim) -> dict:
    """One model call, parsed and rubric-enforced, stamped with the model."""
    result = chat(
        messages,
        model=effective_model,
        temperature=JUDGEMENT_TEMPERATURE,
        options=JUDGEMENT_OLLAMA_OPTIONS,
        backend=effective_backend,
    )
    if not result.content:
        raise JudgementParseError("LLM returned an empty response.", raw="")
    verdict = enforce_rubric(parse_judgement(result.content), citation_evidence, claim=claim)
    # The model that actually produced the verdict, so a result can be traced
    # (and a silent provider-side model change noticed). judge_model is what
    # was asked for; judge_model_version is what the provider says answered.
    verdict["judge_model"] = effective_model or LLM_MODEL
    verdict["judge_model_version"] = getattr(result, "model", None)
    return verdict


def _tally(votes: list[dict]) -> dict:
    """Combine several rubric-enforced votes into one verdict.

    A strict majority on `judgement` returns the first vote carrying it. With
    none, the votes disagree about the claim, and a coin-flip verdict would be
    reported as a finding: the record says Unclear / Low, with vote_split set.
    """
    counts: dict[str, int] = {}
    for vote in votes:
        counts[vote["judgement"]] = counts.get(vote["judgement"], 0) + 1
    winner, top = max(counts.items(), key=lambda item: item[1])
    if top * 2 > len(votes):
        out = next(v for v in votes if v["judgement"] == winner)
        out["votes"] = counts
        return out
    out = dict(votes[0])
    out["judgement"] = "Unclear / insufficient evidence"
    out["confidence"] = "Low"
    # The copy keeps vote 1's reason and span, which may argue for a verdict
    # the vote did not reach; say so at the head of the reason.
    tally = ", ".join(f"{v}: {n}" for v, n in sorted(counts.items()))
    out["reason"] = f"Votes split ({tally}): {out.get('reason') or ''}".rstrip()
    out["votes"] = counts
    out["vote_split"] = True
    return out


_JUDGE_ATTEMPTS = 2


@retry(max_retries=_JUDGE_ATTEMPTS, backoff=1.0)
def _judge_once(claim: str, evidence: str, context: str | None = None, model: str | None = None):
    return judge(claim, evidence, context=context, model=model)


def _judgement_model() -> str:
    """The model that actually produces the verdicts."""
    return JUDGEMENT_MODEL or LLM_MODEL


judge_once = _judge_once

