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

import argparse
import json
import os
import re
from datetime import datetime

from research_assistant.agents.agent5_batch_citer import (
    _cite_keys,
    split_into_sentences,
)
from research_assistant.config import (
    JUDGEMENT_MODEL,
    JUDGEMENT_TOP_K,
    LLM_BACKEND,
    LLM_MODEL,
)
from research_assistant.judgement.judge import (
    DERIVED_FIELDS,
    REQUIRED_FIELDS,
    JudgementParseError,
    judge,
)
from research_assistant.shared.atomic import atomic_write, atomic_write_json
from research_assistant.shared.log import get_logger
from research_assistant.shared.retry import retry
from research_assistant.shared.search import hybrid_search

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


# A "sentence" carrying two or more cite keys and more words than any real
# sentence has is almost always two sentences whose boundary was lost when
# the citation was inserted. Both verdicts on it are then about the wrong
# claim; say so in the record.
COMPOUND_WORDS = 40


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
        compound = len(keys) >= 2 and len(claim.split()) > COMPOUND_WORDS
        if compound:
            logger.warning(
                "Sentence %d carries %d citations across %d words — a lost sentence "
                "boundary? Both verdicts will be about the combined claim.",
                index + 1, len(keys), len(claim.split()),
            )
        for key in sorted(keys):
            source = key_to_source.get(key)
            pairs.append({
                "sentence_index": index,
                "sentence": sentence,
                "claim": claim,
                "cite_key": key,
                "citation_source": source,
                "compound_sentence": compound,
                # Agent 5 already warns when the model invents a key; this is
                # where that shows up per sentence instead of once per run.
                "outcome": None if source else "orphaned",
            })
    return pairs


# Worst first: the reader is looking for citations to fix, and a report that
# opens with 90 lines of "Supports" buries them.
_SEVERITY = {
    "Contradicts": 0,
    "Does not support": 1,
    "Unclear / insufficient evidence": 2,
    "Partially supports": 3,
    "Supports": 4,
}

# Two attempts: shared.retry counts attempts, not retries.
_JUDGE_ATTEMPTS = 2


@retry(max_retries=_JUDGE_ATTEMPTS, backoff=1.0)
def _judge_once(claim, evidence):
    return judge(claim, evidence)


def _judgement_model() -> str:
    """The model that actually produces the verdicts.

    ``judge()`` hands ``JUDGEMENT_MODEL`` to ``chat()``, which falls back to
    ``LLM_MODEL`` only when it is unset. Naming ``LLM_MODEL`` unconditionally
    would attribute every verdict to a model that produced none of them the
    moment ``CITATION_JUDGEMENT_MODEL`` is set — which is the entire point of
    that override, since the audit is meant to be able to run on a stronger
    model than the one that drafted.
    """
    return JUDGEMENT_MODEL or LLM_MODEL


def _warn_if_context_is_tight():
    """A local model with a small context truncates the prompt silently."""
    if LLM_BACKEND != "ollama":
        return
    from research_assistant.config import JUDGEMENT_OLLAMA_OPTIONS
    from research_assistant.judgement.judge import PROMPT_TEMPLATE

    approx_tokens = len(PROMPT_TEMPLATE) // 4
    logger.warning(
        "Judging with local model %s. The prompt alone is ~%d tokens and "
        "num_ctx is set to %s — if the model cannot honour that, Ollama "
        "truncates the prompt tail (where the worked examples are) without "
        "raising, and verdicts degrade silently.",
        _judgement_model(), approx_tokens, JUDGEMENT_OLLAMA_OPTIONS.get("num_ctx"),
    )


def verify_draft(draft_path, citations_path=None, top_k=None,
                 search_resources=None) -> dict:
    """Judge every citation in *draft_path* against its cited source.

    Writes ``<draft>_verification.json`` and ``<draft>_verification.md``
    alongside the draft, matching agent 5's output naming.

    Returns:
        dict: the same record written to the JSON file.
    """
    if citations_path is None:
        citations_path = draft_path.replace(".txt", "_citations.json")

    if not os.path.exists(citations_path):
        raise FileNotFoundError(
            f"No citation mapping at {citations_path}. Agent 8 needs it to "
            "resolve \\cite keys back to sources — run Agent 5 first."
        )

    with open(draft_path, encoding="utf-8") as fh:
        draft_text = fh.read()
    with open(citations_path, encoding="utf-8") as fh:
        mapping = json.load(fh)

    if search_resources is None:
        from research_assistant.shared.db import load_search_resources
        search_resources = load_search_resources()
    collection, bm25, texts, metadatas = search_resources

    top_k = top_k or JUDGEMENT_TOP_K
    sentences = split_into_sentences(draft_text)
    pairs = citation_pairs(sentences, invert_citation_mapping(mapping))

    if pairs:
        logger.info(
            "Verifying %d citation(s) across %d sentence(s) — one model call each.",
            len(pairs), len(sentences),
        )
        _warn_if_context_is_tight()

    doc_cache = {}
    results = []

    for i, pair in enumerate(pairs, 1):
        entry = dict(pair)
        logger.info("[%d/%d] %s", i, len(pairs), entry["claim"][:80])

        if entry["outcome"] == "orphaned":
            logger.info(" -> key %s is not in the mapping.", entry["cite_key"])
            results.append(entry)
            continue

        documents = resolve_documents(collection, entry["citation_source"], doc_cache)
        if not documents:
            entry["outcome"] = "unresolved"
            logger.info(" -> source resolves to no documents in the corpus.")
            results.append(entry)
            continue

        # Retrieval is guarded like everything else in this loop: nothing is
        # persisted until the report is written, so an exception at citation 50
        # of 120 would throw away fifty paid model calls and leave no artefact.
        # hybrid_search embeds the query, which is a network call under
        # EMBED_BACKEND=openai|huggingface.
        try:
            # A verdict rests on what the paper says. A figure_description is
            # a model's reading of a plot (v2, spec §4.6) and is never evidence.
            hits = hybrid_search(
                entry["claim"], collection, bm25, texts, metadatas,
                top_k=top_k, doc_filter=documents,
                exclude_types={"figure_description"},
            )
        except Exception as exc:
            entry["outcome"] = "retrieval_failed"
            entry["error_type"] = type(exc).__name__
            entry["raw"] = str(exc)
            logger.warning(" -> retrieval failed: %s: %s", type(exc).__name__, exc)
            results.append(entry)
            continue

        if not hits:
            entry["outcome"] = "no_evidence"
            logger.info(" -> nothing retrieved from that source for this claim.")
            results.append(entry)
            continue

        entry["evidence"] = hits[0]["text"]
        try:
            verdict = _judge_once(entry["claim"], entry["evidence"])
        except JudgementParseError as exc:
            entry["outcome"] = "parse_failed"
            entry["raw"] = exc.raw
            logger.warning(" -> unusable reply after %d attempts.", _JUDGE_ATTEMPTS)
            results.append(entry)
            continue
        except Exception as exc:
            # Connection errors, timeouts and HTTP 4xx/5xx are not "unusable
            # model reply" — filing them as parse_failed sends the reader to
            # prompt.md when the endpoint was simply down.
            entry["outcome"] = "call_failed"
            entry["error_type"] = type(exc).__name__
            entry["raw"] = str(exc)
            logger.warning(" -> judging call failed: %s: %s", type(exc).__name__, exc)
            results.append(entry)
            continue

        entry["outcome"] = "judged"
        # Only the rubric's own fields are merged. _validate permits extra
        # top-level keys in a model reply — harmless in themselves, but a
        # blanket update() lets an echoed `sentence_index`, `claim`, `evidence`
        # or `outcome` overwrite the pipeline's own record of what was judged,
        # which is exactly the provenance §9.2 leans on to justify
        # re-retrieval.
        entry.update({field: verdict[field] for field in sorted(REQUIRED_FIELDS)})
        # Derived by enforce_rubric(); absent from older stubs and records.
        entry.update({field: verdict[field] for field in sorted(DERIVED_FIELDS) if field in verdict})
        logger.info(" -> %s (%s confidence)", verdict["judgement"], verdict["confidence"])
        results.append(entry)

    report = {
        "draft": os.path.abspath(draft_path),
        "citations": os.path.abspath(citations_path),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": _judgement_model(),
        "top_k": top_k,
        "results": results,
        "totals": _totals(results),
    }

    json_path = draft_path.replace(".txt", "_verification.json")
    atomic_write_json(json_path, report)
    logger.info("Saved verification record to %s", json_path)

    md_path = draft_path.replace(".txt", "_verification.md")
    _write_markdown(md_path, report)
    logger.info("Saved verification report to %s", md_path)

    return report


def _totals(results) -> dict:
    totals = {
        "total": len(results),
        "judged": 0, "orphaned": 0, "unresolved": 0,
        "no_evidence": 0, "retrieval_failed": 0,
        "parse_failed": 0, "call_failed": 0,
    }
    for judgement in _SEVERITY:
        totals[judgement] = 0
    totals["rubric_mismatch"] = 0
    totals["span_unverified"] = 0
    for entry in results:
        totals[entry["outcome"]] = totals.get(entry["outcome"], 0) + 1
        if entry["outcome"] == "judged":
            totals[entry["judgement"]] = totals.get(entry["judgement"], 0) + 1
            if entry.get("rubric_mismatch"):
                totals["rubric_mismatch"] += 1
            if entry.get("span_verified") is False:
                totals["span_unverified"] += 1
    return totals


def _write_markdown(path, report) -> None:
    totals = report["totals"]
    lines = [
        f"# Verification Report — `{os.path.basename(report['draft'])}`",
        f"*Generated {report['generated']} · model `{report['model']}`*\n",
        "## Summary\n",
        "| Outcome | Count |",
        "|---------|-------|",
        f"| Citations checked | {totals['total']} |",
        f"| Judged | {totals['judged']} |",
        f"| **Contradicts** | **{totals.get('Contradicts', 0)}** |",
        f"| **Does not support** | **{totals.get('Does not support', 0)}** |",
        f"| Unclear / insufficient evidence | {totals.get('Unclear / insufficient evidence', 0)} |",
        f"| Partially supports | {totals.get('Partially supports', 0)} |",
        f"| Supports | {totals.get('Supports', 0)} |",
        f"| Key not in mapping | {totals['orphaned']} |",
        f"| Source not in corpus | {totals['unresolved']} |",
        f"| No evidence retrieved | {totals['no_evidence']} |",
        f"| Retrieval failed | {totals['retrieval_failed']} |",
        f"| Unusable model reply | {totals['parse_failed']} |",
        f"| Model call failed | {totals['call_failed']} |",
        "",
        "---\n",
    ]

    judged = [e for e in report["results"] if e["outcome"] == "judged"]
    other = [e for e in report["results"] if e["outcome"] != "judged"]
    judged.sort(key=lambda e: (_SEVERITY.get(e["judgement"], 9), e["sentence_index"]))

    flagged = [e for e in judged if e["judgement"] != "Supports"]
    clean = [e for e in judged if e["judgement"] == "Supports"]

    if flagged:
        lines.append("## Citations to review\n")
        for entry in flagged:
            lines.extend(_entry_block(entry))

    if other:
        lines.append("## Not judged\n")
        for entry in other:
            lines.append(f"### Sentence {entry['sentence_index'] + 1} — {entry['outcome']}\n")
            lines.append(f"> {entry['sentence']}\n")
            lines.append(f"Key `{entry['cite_key']}`"
                         + (f" → {entry['citation_source']}" if entry["citation_source"] else "")
                         + "\n")
            # retrieval_failed and call_failed carry an exception type; naming
            # it is the difference between "fix the prompt" and "the endpoint
            # was down".
            if entry.get("error_type"):
                lines.append(f"**Error:** `{entry['error_type']}`\n")
            if entry.get("raw"):
                # Only parse_failed's `raw` is an actual model reply; the
                # failure outcomes carry an exception message.
                label = ("Raw reply" if entry["outcome"] == "parse_failed"
                         else "Error detail")
                lines.append(f"<details><summary>{label}</summary>\n")
                lines.append(f"```\n{entry['raw'][:2000]}\n```\n")
                lines.append("</details>\n")
            lines.append("---\n")

    if clean:
        lines.append("## Verified\n")
        for entry in clean:
            lines.append(
                f"- Sentence {entry['sentence_index'] + 1} — `{entry['cite_key']}` "
                f"({entry['confidence']} confidence, evidence "
                f"{entry['evidence_sufficiency']}): {entry['sentence']}"
            )
        lines.append("")

    with atomic_write(path) as fh:
        fh.write("\n".join(lines))


# Long enough to judge the verdict against, short enough that a page of
# flagged citations stays readable.
_EVIDENCE_CHARS = 400


def _blockquote(text, limit) -> str:
    """*text* as a single-line Markdown blockquote, truncated to *limit*.

    A retrieved chunk carries newlines, and a bare ``> `` prefix would leave
    every line after the first outside the quote.
    """
    flat = _MULTI_SPACE_RE.sub(" ", str(text)).strip()
    if len(flat) > limit:
        flat = flat[:limit].rstrip() + "…"
    return f"> {flat}"


def _entry_block(entry) -> list:
    slots = entry.get("slots", {})
    block = [
        f"### Sentence {entry['sentence_index'] + 1} — {entry['judgement']}\n",
        f"> {entry['sentence']}\n",
        f"**Cited source:** {entry['citation_source']} (`{entry['cite_key']}`)\n",
        f"**Confidence:** {entry['confidence']} · "
        f"**Evidence sufficiency:** {entry['evidence_sufficiency']}\n",
    ]
    if entry.get("compound_sentence"):
        block.append("⚠ **Compound sentence:** two or more citations in one long sentence — "
                     "a lost sentence boundary? This verdict is about the combined claim.\n")
    if entry.get("rubric_mismatch"):
        detail = "; ".join(entry.get("rubric_violations") or []) or "aggregate did not follow the slots"
        block.append(f"⚠ **Rubric:** model said {entry['model_judgement']}; the rules derive "
                     f"{entry['judgement']} ({detail}).\n")
    if entry.get("span_verified") is False:
        block.append("⚠ **Supporting span not found verbatim in the evidence** — treat it as a paraphrase.\n")
    block.extend([
        "| Slot | Assertion | Verdict |",
        "|------|-----------|---------|",
    ])
    for name in ("finding", "scope", "strength"):
        slot = slots.get(name, {})
        block.append(
            f"| `{name}` | {slot.get('assertion', '—')} | {slot.get('verdict', '—')} |"
        )
    block.append("")
    # The chunk the verdict rests on. §9.2 promises the report names it, and
    # for "Does not support" — the most actionable verdict — supporting_span
    # is legitimately null, so without this the reader gets three assertions
    # and no evidence text at all.
    if entry.get("evidence"):
        block.append("**Evidence judged:**\n"
                     + _blockquote(entry["evidence"], _EVIDENCE_CHARS) + "\n")
    if entry.get("supporting_span"):
        block.append("**Supporting span:**\n"
                     + _blockquote(entry["supporting_span"], _EVIDENCE_CHARS) + "\n")
    block.append(f"**Reason:** {entry['reason']}\n")
    block.append("---\n")
    return block


def main():
    parser = argparse.ArgumentParser(
        description="Agent 8 — verify the citations in an already-cited draft."
    )
    parser.add_argument("--draft", required=True, help="Path to the cited draft (.txt).")
    parser.add_argument("--citations", default=None,
                        help="Path to _citations.json. Defaults to the draft's sibling.")
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"Chunks judged per source (default {JUDGEMENT_TOP_K}).")
    args = parser.parse_args()

    report = verify_draft(args.draft, args.citations, args.top_k)
    totals = report["totals"]
    flagged = sum(
        totals.get(j, 0)
        for j in ("Contradicts", "Does not support", "Unclear / insufficient evidence")
    )
    logger.info(
        "Checked %d citation(s): %d judged, %d need review, %d not judged.",
        totals["total"], totals["judged"], flagged,
        totals["total"] - totals["judged"],
    )


if __name__ == "__main__":
    main()
