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


def parse_reply(raw: str, required: tuple[str, ...]) -> dict:
    """Parse a model output string into a dictionary, enforcing required keys."""
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

    return parsed


def ask_json(
    call: Callable[[list[dict]], str],
    messages: list[dict],
    required: tuple[str, ...],
) -> dict:
    """Call model for a JSON object with one automatic retry on parse failure."""
    raw = call(messages)
    try:
        return parse_reply(raw, required)
    except ReplyParseError:
        retry_messages = list(messages) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": JSON_NUDGE},
        ]
        raw_retry = call(retry_messages)
        return parse_reply(raw_retry, required)


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

    Copies document and chunk_index from the cited passage.
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
            )
        )

    return premises, warnings
