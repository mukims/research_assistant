# research_assistant/shared/chat_context.py
"""The mechanics of a research-chat turn. Pure: no model, no index.

Agent 7 assembles each turn against a token budget derived from num_ctx.
Tokens are estimated at four characters each — this model's tokenizer on
English prose — and the answer reserve carries the margin. Sources are
keyed [S1]… so an answer's citations can be checked against what it was
actually shown, and resolved to titles before the answer enters history.
"""

from __future__ import annotations

import re
from typing import Any

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count at 4 characters per token, rounded up."""
    n = len(text or "")
    return (n + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def messages_tokens(messages: list[dict]) -> int:
    """Sum estimated tokens for a list of message dicts."""
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def cap_per_document(results: list[dict], per_doc: int) -> list[dict]:
    """Retain at most per_doc chunks per metadata['document'], keeping order."""
    counts: dict[str, int] = {}
    out = []
    for r in results:
        meta = r.get("metadata") or {}
        doc = meta.get("document")
        if counts.get(doc, 0) >= per_doc:
            continue
        counts[doc] = counts.get(doc, 0) + 1
        out.append(r)
    return out


def merge_results(focus_hits: list[dict], global_hits: list[dict]) -> list[dict]:
    """Merge focus and global search hits. Focus hits go first, deduplicating on (document, chunk_index)."""
    seen = set()
    out = []
    for r in list(focus_hits) + list(global_hits):
        meta = r.get("metadata") or {}
        key = (meta.get("document"), r.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def format_context(results: list[dict], max_chars: int) -> tuple[str, list[dict]]:
    """Format chunks into [S1], [S2] ... blocks with metadata headers and cap to max_chars."""
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    total = 0

    for r in results:
        m = r.get("metadata") or {}
        key = f"S{len(sources) + 1}"
        citation = m.get("citation_source") or "Unknown"
        page = m.get("page_first", m.get("page", "?"))
        section = m.get("section") or "text"
        raw_text = r.get("text", "")
        block = f"[{key}] {citation} — p.{page}, {section}\n{raw_text}"
        
        if parts and total + len(block) + 2 > max_chars:
            break
        block = block[:max_chars]
        parts.append(block)
        total += len(block) + 2
        sources.append({
            "key": key,
            "document": m.get("document"),
            "citation": citation,
            "page": page,
            "section": m.get("section"),
            "chunk_index": r.get("chunk_index"),
            "rrf_score": r.get("rrf_score"),
            "text": raw_text[:400] if raw_text else "",
            "cited": False,
        })
    return "\n\n".join(parts), sources


def fit_history(history: list[dict], max_tokens: int) -> tuple[list[dict], list[dict]]:
    """Whole (user, assistant) pairs, newest first, until the budget is spent."""
    pairs = [history[i:i + 2] for i in range(0, len(history), 2)]
    kept = []
    used = 0
    for pair in reversed(pairs):
        t = sum(estimate_tokens(m.get("content", "")) for m in pair)
        if used + t > max_tokens:
            break
        kept.insert(0, pair)
        used += t
    dropped = pairs[:len(pairs) - len(kept)]
    return [m for p in kept for m in p], [m for p in dropped for m in p]


_KEY_GROUP_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def cited_keys(text: str) -> list[str]:
    """Extract all cited keys (e.g. [S1], [S2, S3]) in first-use order."""
    used = []
    for m in _KEY_GROUP_RE.finditer(text or ""):
        for k in re.split(r"\s*,\s*", m.group(1)):
            if k not in used:
                used.append(k)
    return used


def short_title(citation: str, words: int = 6) -> str:
    """Extract a short readable title (first N words) from a citation string."""
    parts = (citation or "").split()
    return " ".join(parts[:words]) if parts else "untitled"


def resolve_keys(text: str, sources: list[dict]) -> str:
    """Replace [S1] with [<short title>] using the provided sources list. Unknown keys left as-is."""
    titles = {s["key"]: short_title(s.get("citation", "")) for s in sources}

    def _sub(m):
        keys = re.split(r"\s*,\s*", m.group(1))
        if not all(k in titles for k in keys):
            return m.group(0)
        return "[" + "; ".join(titles[k] for k in keys) + "]"

    return _KEY_GROUP_RE.sub(_sub, text or "")


def parse_brainstorm_suggestions(answer: str) -> list[str]:
    """Parse 2-4 concrete follow-up research questions or exploration directions from the answer.

    Looks for sections like:
    ### 💡 Suggested Next Questions
    - Question 1
    - Question 2
    Or extracts trailing bullet points ending in question marks.
    """
    if not answer:
        return []

    suggestions = []
    
    # Check for explicit suggestion heading
    heading_match = re.search(
        r"(?:###?\s*(?:💡\s*)?(?:Suggested\s+Next\s+Questions|Next\s+Research\s+Directions|Follow-up\s+Questions|Brainstorming\s+Vectors):?\s*)(.*?)(?:\Z|###)",
        answer,
        re.DOTALL | re.IGNORECASE,
    )
    target_block = heading_match.group(1) if heading_match else answer[-1200:]

    lines = target_block.strip().split("\n")
    for line in lines:
        line_clean = line.strip()
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line_clean)
        if m:
            item = m.group(1).strip()
            item = re.sub(r"^\*\*([^*]+)\*\*:?\s*", r"\1: ", item)
            item = item.strip("\"' ")
            if len(item) >= 15 and (item.endswith("?") or not heading_match or len(suggestions) < 3):
                suggestions.append(item)
                if len(suggestions) >= 4:
                    break

    if not suggestions and heading_match:
        for chunk in heading_match.group(1).split("?"):
            c = chunk.strip().lstrip("-*•123456789. ")
            if len(c) > 15:
                suggestions.append(c + "?")
                if len(suggestions) >= 3:
                    break

    return suggestions[:4]
