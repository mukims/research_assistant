import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

from research_assistant.prompts import (
    CITATION_NEED_CHECK,
    CITE_SENTENCE_SYSTEM,
    CITE_SENTENCE_USER,
)
from research_assistant.shared.log import get_logger
from research_assistant.shared.db import load_search_resources
from research_assistant.config import (
    CITATION_CITER_CONTEXTUALIZE,
    CITATION_CITER_JUDGE,
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_NEIGHBOUR_WINDOW,
)
from research_assistant.judgement.judge import compose_context, judge
from research_assistant.shared.claim_text import sentence_context, split_into_sentences  # re-exported: seed_audit, agent8 and tests import it from here
from research_assistant.shared.search import expand_neighbours, hybrid_search
from research_assistant.shared.retry import retry
from research_assistant.shared.llm import chat

logger = get_logger("agent5")


def split_paragraphs(text: str) -> list:
    """Sentences per paragraph, paragraphs being blank-line separated. The
    flat sentence list is what split_into_sentences gave for the whole text
    — a paragraph break is whitespace after a full stop to the splitter —
    so the draft is rebuilt exactly as before; the paragraphs are for the
    context each sentence is cited in."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [split_into_sentences(p) for p in paragraphs] or [[]]


def contextualized_queries(sentences, contexts, paragraph_ids, needs_cite) -> dict:
    """{sentence index: query} for the sentences that need a citation, from
    the audit's query contextualization — one model call per paragraph. A
    failure means no queries: the caller searches with the sentences."""
    from research_assistant.shared.seed_audit import contextualize_citation_queries  # seed_audit imports this module

    indices = [i for i, need in enumerate(needs_cite) if need]
    if not indices:
        return {}
    claims = [{"claim": sentences[i], "context": contexts[i], "paragraph_id": paragraph_ids[i]} for i in indices]
    try:
        contextualize_citation_queries(claims)
    except Exception as exc:  # noqa: BLE001 — the query is better with it, fine without
        logger.warning("Query contextualization failed (%s) — searching with the sentences.", exc)
        return {}
    return {i: c["search_query"] for i, c in zip(indices, claims) if c.get("search_query")}


# ─── Citation key extraction ─────────────────────────────────────────────────

_CITE_RE = re.compile(r"\\cite\{([^}]*)\}")

# One or more \cite{...} groups at the very end of a sentence, with the
# whitespace before them.
_TRAILING_CITES_RE = re.compile(r"(?:\s*\\cite\{[^}]*\})+\s*$")
_TERMINAL = ".!?"


def _restore_terminal_punctuation(original: str, cited: str) -> str:
    r"""Give the cited sentence back the full stop the model dropped.

    gemma4:e2b writes "… films \cite{cite_1}" for "… films." — no terminal
    punctuation. run_batch_citer joins sentences with a space, and Agent 8's
    split_into_sentences needs [.!?] before whitespace, so the next sentence
    is swallowed into this one and both citations get judged against a
    two-sentence claim. Restore the original's terminator after the cite.

    Left alone: a cited sentence that already ends in [.!?], or that ends in
    [.!?] immediately before a trailing \cite{} group ("… films. \cite{a}"),
    and any original that had no terminator to restore.
    """
    original = original.rstrip()
    if not original or original[-1] not in _TERMINAL:
        return cited
    stripped = cited.rstrip()
    if stripped and stripped[-1] in _TERMINAL:
        return stripped
    core = _TRAILING_CITES_RE.sub("", stripped)
    if core and core[-1] in _TERMINAL:
        return stripped
    return stripped + original[-1]


def _cite_keys(text: str) -> set[str]:
    """Return the set of citation keys actually present in *text*.

    Handles the multi-key form ``\\cite{a,b}`` as well as ``\\cite{a}``.
    An empty set means no citation was made, whatever the model replied.
    """
    keys = set()
    for match in _CITE_RE.finditer(text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


# ─── Batched citation-need check ─────────────────────────────────────────────

@retry(max_retries=2, backoff=2.0)
def _batch_needs_citation(sentences):
    """
    Ask the LLM once whether each sentence in a batch needs a citation.
    Returns a list of booleans aligned with the input list.
    """
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = CITATION_NEED_CHECK.format(numbered=numbered)
    answer = chat([{"role": "user", "content": prompt}]).content

    # Parse the YES/NO list.
    #
    # Index by the number the model emitted rather than by line position: a
    # preamble line, a blank line, or a double-spaced list would otherwise
    # shift every verdict onto the wrong sentence. Padding a short result to
    # length hides exactly that failure, so a missing verdict raises instead
    # (the @retry above gives the model two more attempts first).
    verdicts = {}
    for line in answer.strip().split("\n"):
        match = re.match(r"\s*(\d+)\s*[.)]\s*(YES|NO)\b", line.strip(), re.IGNORECASE)
        if not match:
            continue
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(sentences):
            verdicts[idx] = match.group(2).upper() == "YES"

    missing = [i + 1 for i in range(len(sentences)) if i not in verdicts]
    if missing:
        raise ValueError(
            f"Citation-need check returned no verdict for sentence(s) {missing} "
            f"of {len(sentences)}. Raw response:\n{answer.strip()[:500]}"
        )

    return [verdicts[i] for i in range(len(sentences))]


@retry(max_retries=3, backoff=2.0)
def _cite_sentence_with_reasoning(sentence, context_str):
    """Ask the LLM to rewrite a sentence with \\cite{key} and explain why.

    Returns:
        tuple: (cited_sentence, reasoning)
    """
    sys_prompt = CITE_SENTENCE_SYSTEM
    user_prompt = CITE_SENTENCE_USER.format(sentence=sentence, context=context_str)

    result = chat([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ])
    raw = result.content.strip()
    prompt_tokens, completion_tokens = result.prompt_tokens, result.completion_tokens
    logger.info(
        " -> [Token Stats] Submitted: %s | Generated: %s", prompt_tokens, completion_tokens
    )

    # Parse the structured response
    cited_sentence = sentence  # fallback
    reasoning = ""

    cited_match = re.search(r'CITED:\s*(.+?)(?:\nREASON:|$)', raw, re.DOTALL)
    reason_match = re.search(r'REASON:\s*(.+)', raw, re.DOTALL)

    if cited_match:
        cited_sentence = cited_match.group(1).strip()
    elif "\\cite" in raw:
        # If the model didn't follow format but did produce a citation,
        # use the first line as the sentence
        cited_sentence = raw.split("\n")[0].strip()

    if reason_match:
        reasoning = reason_match.group(1).strip()
    elif not cited_match and len(raw.split("\n")) > 1:
        # Try to extract reasoning from non-formatted response
        reasoning = " ".join(raw.split("\n")[1:]).strip()

    return _restore_terminal_punctuation(sentence, cited_sentence), reasoning


# ─── The per-sentence seam ────────────────────────────────────────────────────

@dataclass
class CiteResult:
    """What citing one sentence produced.

    ``candidates`` are every chunk retrieval offered, in rank order, each
    with the ``document`` it came from — the name the audit and the citer's
    evaluation key on. The batch loop used to keep only ``citation_source``.
    """
    original: str
    cited_text: str
    keys: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    reasoning: str = ""
    skip_reason: str | None = None
    query: str = ""
    verdicts: list = field(default_factory=list)
    partial: bool = False

    @property
    def cited(self) -> bool:
        return bool(self.keys)


def cite_sentence(sentence, resources, key_registry, *, context=None, query=None,
                  exclude_docs=None, paragraph_id=None, judge_gate=None) -> CiteResult:
    """Retrieve context for one sentence and ask the model to cite it.

    ``key_registry`` maps citation_source → cite key and is shared across a
    draft so a source keeps its key; entries are only ever added, so the
    next key is len + 1. ``query`` is what to search with (the sentence
    when None); ``exclude_docs`` keeps named documents out of retrieval.
    ``judge_gate``: True/False forces the judge acceptance test on or off;
    None reads CITATION_CITER_JUDGE.
    A model or retrieval failure propagates — the caller decides what a
    failed sentence means.
    """
    collection, bm25, texts, metadatas = resources
    query = query or sentence
    results = hybrid_search(query, collection, bm25, texts, metadatas, top_k=3, exclude_docs=exclude_docs)
    if not results:
        return CiteResult(original=sentence, cited_text=sentence, query=query,
                          skip_reason="no relevant context found in database")

    context_str = ""
    candidates = []
    for r in results:
        meta = r.get("metadata") or {}
        cit_source = meta.get("citation_source", "Unknown")
        # Keys are registered for every retrieved chunk so the model has a
        # stable label to reference, but registration is NOT the same as
        # use — the caller filters the final mapping down to keys that
        # actually made it into the draft.
        if cit_source not in key_registry:
            key_registry[cit_source] = f"cite_{len(key_registry) + 1}"
        cite_key = key_registry[cit_source]
        context_str += f"--- Context (Cite Key: {cite_key}) ---\n{r['text']}\n\n"
        candidates.append({
            "key": cite_key, "citation": cit_source, "document": meta.get("document"),
            "chunk_index": r.get("chunk_index"), "rrf_score": r.get("rrf_score"),
        })

    if CITATION_CITER_JUDGE if judge_gate is None else judge_gate:
        return _cite_by_judge(sentence, results, candidates, context=context, query=query,
                              texts=texts, metadatas=metadatas)

    cited_sentence, reasoning = _cite_sentence_with_reasoning(sentence, context_str)
    keys = sorted(_cite_keys(cited_sentence))
    if keys:
        return CiteResult(original=sentence, cited_text=cited_sentence, keys=keys,
                          candidates=candidates, reasoning=reasoning, query=query)
    # A successful call is not a citation: decide from the text. A reply with
    # no key is a decline, and a declining model's rewrite is noise — the
    # draft keeps the author's sentence, as the report already says it does.
    # (The judge path never uses the rewrite at all: _insert_cite places the key.)
    return CiteResult(original=sentence, cited_text=sentence, candidates=candidates,
                      reasoning=reasoning, query=query,
                      skip_reason="context retrieved but the model did not cite it")


_VERDICT_RANK = {"Supports": 0, "Partially supports": 1, "Contradicts": 2,
                 "Does not support": 3, "Unclear / insufficient evidence": 4}
_VERDICT_FIELDS = ("judgement", "model_judgement", "confidence", "evidence_sufficiency",
                   "supporting_span", "span_verified", "reason", "rubric_violations", "slots")


def _insert_cite(sentence: str, key: str) -> str:
    """\\cite{key} before the terminal punctuation — where the legacy rewrite
    already ends up after _restore_terminal_punctuation — or appended."""
    s = sentence.rstrip()
    if s and s[-1] in _TERMINAL:
        return f"{s[:-1].rstrip()} \\cite{{{key}}}{s[-1]}"
    return f"{s} \\cite{{{key}}}"


def _accepted(sentence, cand, record, candidates, verdicts, query, *, partial) -> CiteResult:
    return CiteResult(original=sentence, cited_text=_insert_cite(sentence, cand["key"]),
                      keys=[cand["key"]], candidates=candidates, reasoning=record.get("reason") or "",
                      query=query, verdicts=verdicts, partial=partial)


def _cite_by_judge(sentence, hits, candidates, *, context, query, texts, metadatas) -> CiteResult:
    """Judge each candidate in retrieval order with the auditor's own judge
    and cite the first that passes: Supports with a verbatim span, else the
    first Partially supports with one, flagged. The key is placed by code —
    the judge chose, nothing has to be parsed out of a rewrite. Nothing
    passing means no citation and a record of what came closest, so the
    report can show the same account the audit would.
    """
    from research_assistant.agents.agent8_verifier import assemble_evidence  # agent8 imports this module

    expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
    ctx = compose_context(context) if context else None
    verdicts, partial = [], None
    for hit, cand in zip(hits, candidates):
        evidence = assemble_evidence([hit], JUDGEMENT_EVIDENCE_MAX_CHARS)
        record = {"key": cand["key"], "document": cand.get("document"), "evidence": evidence}
        try:
            v = judge(sentence, evidence, context=ctx)
        except Exception as exc:  # noqa: BLE001 — one candidate's failure is not the sentence's
            record["error"] = f"{type(exc).__name__}: {exc}"
            verdicts.append(record)
            continue
        record.update({k: v.get(k) for k in _VERDICT_FIELDS})
        verdicts.append(record)
        if v.get("judgement") == "Supports" and v.get("span_verified") is True:
            return _accepted(sentence, cand, record, candidates, verdicts, query, partial=False)
        if partial is None and v.get("judgement") == "Partially supports" and v.get("span_verified") is True:
            partial = (cand, record)
    if partial is not None:
        return _accepted(sentence, partial[0], partial[1], candidates, verdicts, query, partial=True)

    judged = [r for r in verdicts if "judgement" in r]
    if not judged:
        return CiteResult(original=sentence, cited_text=sentence, candidates=candidates,
                          verdicts=verdicts, query=query, skip_reason="judge failed on every candidate")
    best = min(judged, key=lambda r: _VERDICT_RANK.get(r["judgement"], 9))   # ties keep retrieval order
    best["best"] = True
    return CiteResult(original=sentence, cited_text=sentence, candidates=candidates, verdicts=verdicts,
                      query=query, reasoning=best.get("reason") or "",
                      skip_reason="no candidate passed the judge")


def _verdict_steps(record: dict, sentence: str) -> list:
    """The audit's 'Why this verdict' lines for one of the citer's judged
    candidates. seed_audit imports this module, so it is imported here."""
    from research_assistant.shared.seed_audit import explain_verdict_steps

    item = {"outcome": "judged", "claim": sentence, "evidence_hits": 1}
    item.update({k: v for k, v in record.items() if k not in ("key", "document", "evidence", "best", "error")})
    return explain_verdict_steps(item)


def _generate_report(
    file_path: str,
    report_path: str,
    sentences: list[str],
    cited_sentences: list[str],
    needs_cite: list[bool],
    citation_entries: list[dict],
    citation_mapping: dict,
    key_registry: dict,
):
    """Generate a markdown report explaining every citation decision."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    basename = os.path.basename(file_path)

    # These three are mutually exclusive and cover every sentence, so a reader
    # can see at a glance how often the pipeline wanted a citation and failed
    # to produce one — previously indistinguishable from "none needed".
    n_cited = sum(1 for e in citation_entries if e.get("cited"))
    n_not_needed = sum(
        1 for i, e in enumerate(citation_entries)
        if not e.get("cited") and not needs_cite[i]
    )
    n_declined = sum(
        1 for i, e in enumerate(citation_entries)
        if not e.get("cited") and needs_cite[i]
    )

    lines = [
        f"# Citation Report — `{basename}`",
        f"*Generated on {timestamp}*\n",
        "---\n",
        "## Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total sentences | {n_cited + n_not_needed + n_declined} |",
        f"| Cited | {n_cited} |",
        f"| No citation needed | {n_not_needed} |",
        f"| **Needed a citation, none made** | **{n_declined}** |",
        f"| Unique sources cited | {len(citation_mapping)} |",
        f"| Sources retrieved but never cited | {len(key_registry) - len(citation_mapping)} |",
        "",
        "## Citation Key Mapping\n",
        "| Key | Full Citation |",
        "|-----|---------------|",
    ]

    for full_cit, key in sorted(citation_mapping.items(), key=lambda x: x[1]):
        # Truncate long citations for the table
        short = full_cit[:100] + ("…" if len(full_cit) > 100 else "")
        lines.append(f"| `{key}` | {short} |")

    lines.append("")
    lines.append("---\n")
    lines.append("## Sentence-by-Sentence Analysis\n")

    for i, entry in enumerate(citation_entries):
        lines.append(f"### Sentence {i+1}\n")

        if not entry.get("cited"):
            lines.append(f"> {entry['original']}\n")
            lines.append(f"**Decision:** Not cited — {entry.get('skip_reason', 'skipped')}\n")

            if entry.get("reasoning"):
                lines.append("**Model's explanation:**")
                lines.append(f"{entry['reasoning']}\n")

            best = next((v for v in entry.get("verdicts") or [] if v.get("best")), None)
            if best:
                lines.append(f"**Closest candidate:** `{best['key']}` — judged *{best.get('judgement')}*")
                lines.extend(f"- *{label}:* {text}" for label, text in _verdict_steps(best, entry["original"]))
                lines.append("")

            if entry.get("candidates"):
                lines.append("**Retrieved but not used:**")
                for src in entry["candidates"]:
                    lines.append(f"- `{src['key']}` — {src['citation'][:80]}")
                lines.append("")
        else:
            lines.append("**Original:**")
            lines.append(f"> {entry['original']}\n")
            lines.append("**With citation:**")
            lines.append(f"> {entry['cited_text']}\n")

            if entry.get("reasoning"):
                lines.append("**Reasoning:**")
                lines.append(f"{entry['reasoning']}\n")

            chosen = next((v for v in entry.get("verdicts") or [] if v.get("key") in _cite_keys(entry["cited_text"])), None)
            if chosen:
                if entry.get("partial"):
                    lines.append("⚠ **Partial support** — the judge found the source carries a weaker version of this sentence.\n")
                if chosen.get("supporting_span"):
                    lines.append(f"**Verified span:** “{chosen['supporting_span']}”\n")
                lines.append("**Why this citation**")
                lines.extend(f"- *{label}:* {text}" for label, text in _verdict_steps(chosen, entry["original"]))
                lines.append("")

            # Separate what was actually cited from what was merely retrieved —
            # listing all three candidates as "sources used" overstates the
            # evidence behind the sentence.
            if entry.get("sources"):
                used = _cite_keys(entry["cited_text"])
                chosen = [s for s in entry["sources"] if s["key"] in used]
                rejected = [s for s in entry["sources"] if s["key"] not in used]

                if chosen:
                    lines.append("**Cited:**")
                    for src in chosen:
                        lines.append(f"- `{src['key']}` — {src['citation'][:80]}")
                    lines.append("")
                if rejected:
                    lines.append("**Also retrieved, not cited:**")
                    for src in rejected:
                        lines.append(f"- `{src['key']}` — {src['citation'][:80]}")
                    lines.append("")

        lines.append("---\n")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Saved citation report to %s", report_path)


def run_batch_citer(file_path, out_path="cited_draft.txt", search_resources=None):
    """Cite every claim in *file_path* that the retrieved corpus supports.

    Writes three files alongside *out_path*: the cited draft, a
    ``_citations.json`` key mapping for BibTeX, and a ``_report.md`` explaining
    each decision.

    Returns:
        str | None: *out_path* if the draft was written, None if the run was
        aborted before producing output.
    """
    if not os.path.exists(file_path):
        logger.error("File %s not found.", file_path)
        return None

    with open(file_path, "r") as f:
        draft_text = f.read()

    if search_resources:
        collection, bm25, texts, metadatas = search_resources
    else:
        collection, bm25, texts, metadatas = load_search_resources()

    sentences, contexts, paragraph_ids = [], [], []
    for p_idx, para in enumerate(split_paragraphs(draft_text)):
        for s_idx, sent in enumerate(para):
            sentences.append(sent)
            contexts.append(sentence_context(para, s_idx))
            paragraph_ids.append(f"p_{p_idx}")
    logger.info("Split draft into %d sentences in %d paragraph(s).", len(sentences), len(set(paragraph_ids)))

    # ── Batch citation-need check ────────────────────────────────────────
    eligible_indices = [i for i, s in enumerate(sentences) if len(s.split()) >= 4]
    eligible_sentences = [sentences[i] for i in eligible_indices]

    needs_cite = [False] * len(sentences)
    if eligible_sentences:
        logger.info("Checking %d eligible sentences for citation need (batched)…", len(eligible_sentences))
        try:
            batch_results = _batch_needs_citation(eligible_sentences)
        except Exception as e:
            # Abort rather than guess. A misaligned verdict list attributes one
            # sentence's decision to another, and nothing has been written yet,
            # so stopping here costs no work and avoids a misleading draft.
            logger.error(
                "Citation-need check failed after retries: %s\n"
                "  → Aborting without writing output. Re-run to try again, or "
                "shorten the draft if the model keeps truncating its reply.", e,
            )
            return None
        for idx, needs in zip(eligible_indices, batch_results):
            needs_cite[idx] = needs

    queries = contextualized_queries(sentences, contexts, paragraph_ids, needs_cite) if CITATION_CITER_CONTEXTUALIZE else {}

    # ── Process sentences ────────────────────────────────────────────────
    cited_sentences = []
    key_registry = {}      # every source offered to the model: citation → cite_N
    citation_entries = []  # For the report

    for i, sentence in enumerate(sentences):
        logger.info("[%d/%d] %s", i + 1, len(sentences), sentence[:80])
        entry = {"original": sentence, "cited": False}

        if not needs_cite[i]:
            reason = "too short" if len(sentence.split()) < 4 else "no citation needed"
            logger.info(" -> %s, skipping.", reason)
            cited_sentences.append(sentence)
            entry["skip_reason"] = reason
            citation_entries.append(entry)
            continue

        logger.info(" -> Needs citation. Searching context…")
        try:
            res = cite_sentence(sentence, (collection, bm25, texts, metadatas), key_registry,
                                context=contexts[i], query=queries.get(i), paragraph_id=paragraph_ids[i])
        except Exception as e:
            logger.error(" -> Error during citing: %s", e)
            cited_sentences.append(sentence)
            entry["skip_reason"] = f"LLM error: {e}"
            citation_entries.append(entry)
            continue

        cited_sentences.append(res.cited_text)
        if res.verdicts:
            entry["verdicts"] = res.verdicts
            entry["partial"] = res.partial
        if res.cited:
            logger.info(" -> Cited: %s", res.cited_text[:80])
            entry["cited"] = True
            entry["cited_text"] = res.cited_text
            entry["reasoning"] = res.reasoning
            entry["sources"] = res.candidates
        else:
            if res.candidates:
                logger.info(" -> Declined: retrieved context did not support the claim.")
                entry["candidates"] = res.candidates
            else:
                logger.info(" -> No context found.")
            entry["skip_reason"] = res.skip_reason
            if res.reasoning:
                entry["reasoning"] = res.reasoning
        citation_entries.append(entry)

    # ── Write outputs ────────────────────────────────────────────────────
    final_draft = " ".join(cited_sentences)

    with open(out_path, "w") as f:
        f.write(final_draft)
    logger.info("Saved cited draft to %s", out_path)

    # The mapping is the BibTeX input, so it must describe the draft as written.
    # key_registry holds every source that was offered to the model; only the
    # keys the model actually used belong here.
    used_keys = _cite_keys(final_draft)
    citation_mapping = {
        source: key for source, key in key_registry.items() if key in used_keys
    }

    unknown = used_keys - set(key_registry.values())
    if unknown:
        logger.warning(
            "Draft cites %d key(s) that were never offered as context — the model "
            "likely invented them: %s",
            len(unknown), ", ".join(sorted(unknown)),
        )

    logger.info(
        "%d of %d retrieved sources were actually cited.",
        len(citation_mapping), len(key_registry),
    )

    mapping_file = out_path.replace(".txt", "_citations.json")
    with open(mapping_file, "w") as f:
        json.dump(citation_mapping, f, indent=4)
    logger.info("Saved citation mapping to %s", mapping_file)

    # ── Generate citation reasoning report ───────────────────────────────
    report_path = out_path.replace(".txt", "_report.md")
    _generate_report(
        file_path, report_path, sentences, cited_sentences,
        needs_cite, citation_entries, citation_mapping, key_registry,
    )

    # Returned so callers can tell a completed run from an aborted one. The
    # early returns above yield None; only this path wrote an output file.
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 5 — Batch Citer (Research Assistant)")
    parser.add_argument("--file", type=str, required=True, help="Path to the draft text file.")
    parser.add_argument("--out", type=str, default="cited_draft.txt", help="Path to save the cited draft.")
    args = parser.parse_args()

    run_batch_citer(args.file, args.out)
