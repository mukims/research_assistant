"""Strict reply parsing and premise key resolution for technique transfer."""

from __future__ import annotations

import json
import re
from typing import Callable

from research_assistant.transfer.models import Passage, Premise

JSON_NUDGE = (
    "Your previous reply was not the JSON object asked for. "
    "Reply again with only that JSON object: no prose, no code fence."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


class ReplyParseError(ValueError):
    """Raised when an LLM reply is not valid JSON or misses required keys."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class ReplyValidationError(ReplyParseError):
    """Raised when a parsed JSON reply fails nested schema or enum validation."""


def parse_reply(
    raw: str,
    required: tuple[str, ...],
    validator: Callable[[dict], list[str]] | None = None,
) -> dict:
    """Parse a model output string into a dictionary, enforcing required keys and optional schema rules."""
    if not raw or not raw.strip():
        raise ReplyParseError("Empty reply from model", raw=raw)

    cleaned = raw.strip()
    match = _FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback to outermost { ... }
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

    if not isinstance(parsed, dict):
        raise ReplyParseError(f"Expected JSON object, got {type(parsed).__name__}", raw=raw)

    missing = [k for k in required if k not in parsed]
    if missing:
        raise ReplyParseError(f"Missing required key(s): {missing}", raw=raw)

    if validator is not None:
        errors = validator(parsed)
        if errors:
            raise ReplyValidationError("; ".join(errors), raw=raw)

    return parsed


def ask_json(
    call: Callable[[list[dict]], str],
    messages: list[dict],
    required: tuple[str, ...],
    validator: Callable[[dict], list[str]] | None = None,
) -> dict:
    """Call model for a JSON object with one automatic retry on parse or validation failure."""
    raw = call(messages)
    try:
        return parse_reply(raw, required, validator=validator)
    except ReplyValidationError as exc:
        nudge = f"{JSON_NUDGE} Fix schema error(s): {exc}"
        retry_messages = list(messages) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": nudge},
        ]
        raw_retry = call(retry_messages)
        return parse_reply(raw_retry, required, validator=validator)
    except ReplyParseError:
        retry_messages = list(messages) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": JSON_NUDGE},
        ]
        raw_retry = call(retry_messages)
        return parse_reply(raw_retry, required, validator=validator)


def validate_transfer_reply(parsed: dict) -> list[str]:
    """Validate nested structure of a TASK: transfer reply."""
    errors: list[str] = []
    mapping = parsed.get("mapping")
    if not isinstance(mapping, list) or not mapping or not all(isinstance(r, dict) for r in mapping):
        errors.append("'mapping' must be a non-empty list of objects")
    steps = parsed.get("steps")
    if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
        errors.append("'steps' must be a list of objects")
    ident = parsed.get("identifiability")
    if not isinstance(ident, dict):
        errors.append("'identifiability' must be an object")
    premises = parsed.get("premises")
    if not isinstance(premises, list):
        errors.append("'premises' must be a list")
    return errors


def validate_critique_reply(parsed: dict) -> list[str]:
    """Validate nested structure and enums of a TASK: critique reply."""
    errors: list[str] = []
    sm = parsed.get("shape_match")
    if not isinstance(sm, str) or sm.strip().lower() not in ("yes", "partial", "no", "unknown"):
        errors.append("'shape_match' must be one of 'yes', 'partial', 'no'")
    if "rows" in parsed and not isinstance(parsed["rows"], list):
        errors.append("'rows' must be a list")
    if "objections" in parsed and not isinstance(parsed["objections"], list):
        errors.append("'objections' must be a list")
    return errors


def validate_revise_reply(parsed: dict) -> list[str]:
    """Validate nested structure of a TASK: revise reply."""
    errors: list[str] = []
    responses = parsed.get("responses")
    if not isinstance(responses, list) or not all(isinstance(r, dict) for r in responses):
        errors.append("'responses' must be a list of objects")
    return errors


def validate_verify_rebuttal_reply(parsed: dict) -> list[str]:
    """Validate nested structure of a TASK: verify_rebuttal reply."""
    errors: list[str] = []
    verdicts = parsed.get("verdicts")
    if not isinstance(verdicts, list) or not all(isinstance(v, dict) for v in verdicts):
        errors.append("'verdicts' must be a list of objects")
    return errors


def normalise_key(key: str | None) -> str:
    """Turn '[c7]' or ' C7 ' into 'C7'."""
    if not key:
        return ""
    return key.strip().strip("[]").upper()


def resolve_premises(
    items: list[dict],
    passages: dict[str, Passage],
    limit: int,
) -> tuple[list[Premise], list[str]]:
    """
    Resolve raw premise items against shown passages.

    Copies document, seq and chunk_index from the cited passage.
    Drops unknown keys, empty claims, duplicates, and items beyond limit.
    """
    premises: list[Premise] = []
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()

    for item in items:
        raw_claim = item.get("claim", "")
        claim = raw_claim.strip() if isinstance(raw_claim, str) else ""
        raw_key = item.get("key", "")
        norm_key = normalise_key(raw_key)

        if not claim:
            warnings.append(f"premise for key {raw_key!r} has an empty claim and was dropped")
            continue

        if norm_key not in passages:
            warnings.append(
                f"premise cites {raw_key}, which was not among the passages shown"
            )
            continue

        claim_key_pair = (claim, norm_key)
        if claim_key_pair in seen:
            warnings.append(f"duplicate premise for {norm_key} was dropped")
            continue
        seen.add(claim_key_pair)

        if len(premises) >= limit:
            warnings.append(f"premise beyond limit of {limit} was dropped: {claim[:40]}...")
            continue

        passage = passages[norm_key]
        premises.append(
            Premise(
                claim=claim,
                key=norm_key,
                document=passage.document,
                chunk_index=passage.chunk_index,
                seq=passage.seq,
            )
        )

    return premises, warnings
