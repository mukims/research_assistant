"""
Agent 8 — Verifier.

Audits an already-cited draft. For every \\cite{key} Agent 5 inserted, this
re-retrieves the best-matching chunk from the cited source and asks the
judgement prompt whether that evidence actually supports the claim.

It reports; it does not gate. A citation Agent 5 got wrong is still in the
draft — this flags it, with a slot-level verdict and a verbatim span, and
leaves the correction to the author. Making it a gate would put a
per-sentence LLM call back in Agent 5's hot path, which is the cost Agent 5's
batched citation-need check exists to avoid.

Evidence is re-retrieved rather than replayed because Agent 5 does not persist
the chunk text it cited — only the source metadata. The question this answers
is therefore "does this source's strongest evidence for this claim support
it?", which fails in the safe direction: it cannot manufacture support the
source does not contain.
"""

import re

from research_assistant.agents.agent5_batch_citer import (
    _cite_keys,
    split_into_sentences,
)
from research_assistant.shared.log import get_logger

logger = get_logger("agent8")

# " \cite{x}." must become "." and not " .", and the gap the removal leaves
# behind must collapse — the claim is handed to a model as prose.
_CITE_RE = re.compile(r"\\cite\{[^}]*\}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_MULTI_SPACE_RE = re.compile(r"\s+")


def invert_citation_mapping(mapping: dict) -> dict:
    """``{citation_source: cite_key}`` → ``{cite_key: citation_source}``.

    Agent 5 writes _citations.json keyed by source because that is how it
    builds the registry; every consumer here starts from a key found in the
    draft, so the mapping is inverted once up front.
    """
    return {key: source for source, key in mapping.items()}


def strip_citations(sentence: str) -> str:
    """The sentence as prose, with every LaTeX citation removed."""
    text = _CITE_RE.sub("", sentence)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def resolve_documents(collection, citation_source: str, cache: dict) -> set:
    """Documents whose chunks carry *citation_source*.

    hybrid_search filters on ``metadata["document"]``, but a citation key
    resolves to a ``citation_source``. This is the bridge between the two.
    Cached because a source cited twenty times is one lookup, not twenty.
    """
    if citation_source in cache:
        return cache[citation_source]

    try:
        batch = collection.get(
            where={"citation_source": citation_source}, include=["metadatas"]
        )
        documents = {
            meta.get("document")
            for meta in (batch.get("metadatas") or [])
            if meta and meta.get("document")
        }
    except Exception as exc:
        logger.warning("Could not resolve source %r: %s", citation_source, exc)
        documents = set()

    cache[citation_source] = documents
    return documents


def citation_pairs(sentences: list, key_to_source: dict) -> list:
    """One record per (sentence, cited source) pair, in draft order.

    A sentence citing three sources is judged three times: each source is a
    separate claim about a separate paper.
    """
    pairs = []
    for index, sentence in enumerate(sentences):
        keys = _cite_keys(sentence)
        if not keys:
            continue
        claim = strip_citations(sentence)
        for key in sorted(keys):
            source = key_to_source.get(key)
            pairs.append({
                "sentence_index": index,
                "sentence": sentence,
                "claim": claim,
                "cite_key": key,
                "citation_source": source,
                # Agent 5 already warns when the model invents a key; this is
                # where that shows up per sentence instead of once per run.
                "outcome": None if source else "orphaned",
            })
    return pairs
