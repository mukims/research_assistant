"""
Two-stage retrieval.

Stage 1 — cheap: rank documents by the similarity of their *summary* to the
query, keep the top DOC_SELECT_K, and (if DOC_GATE) let the LLM drop the ones
that aren't plausibly relevant prior work.

Stage 2 — expensive: hybrid chunk search, but only over the documents that
survived stage 1.

As the corpus grows this keeps the detailed search bounded to a handful of
papers instead of the whole store.
"""

import re
import time

from research_assistant.config import (
    VECTORDB_PATH,
    SUMMARY_COLLECTION_NAME,
    DOC_SELECT_K,
    DOC_GATE,
    GATE_BATCHED,
    DEFAULT_TOP_K,
    CHAT_OLLAMA_OPTIONS,
    SYNTHESIS_MODE,
    SYNTHESIS_PER_PAPER_CHUNKS,
    SYNTHESIS_PER_PAPER_MAX_CHARS,
    SYNTHESIS_TEMPERATURE,
    SYNTHESIS_OLLAMA_OPTIONS,
)
from research_assistant.prompts import (
    DOC_RELEVANCE_GATE,
    DOC_RELEVANCE_GATE_BATCH,
    SYNTHESIS_SYSTEM,
    PAPER_NOTES_USER,
    SYNTHESIS_USER,
)
from research_assistant.shared.db import load_search_resources
from research_assistant.shared.llm import chat
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import hybrid_search


logger = get_logger("retrieve")


# ─── Stage 1 ────────────────────────────────────────────────────────────────


def rank_documents(query: str, k: int = DOC_SELECT_K) -> list[dict]:
    """Nearest document summaries. Returns [{document, citation, summary, score}]."""
    import chromadb

    from research_assistant.shared.llm import get_embeddings

    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    try:
        col = client.get_collection(SUMMARY_COLLECTION_NAME)
    except Exception:
        logger.warning("No summary collection yet — has anything been ingested?")
        return []

    n = col.count()
    if not n:
        return []

    res = col.query(
        query_embeddings=[get_embeddings().embed_query(query)],
        n_results=min(k, n),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for summary, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({
            "document": meta.get("document", "?"),
            "citation": meta.get("citation_source", ""),
            "summary": summary,
            "score": round(1.0 - dist, 4),
        })
    return out


def document_summaries(documents) -> dict[str, str]:
    """The ingest-time summary of each named document, keyed by document.
    One store read for the lot; a document without a summary is absent
    from the result, and no summary collection at all is an empty dict."""
    import chromadb

    wanted = sorted({d for d in documents if d})
    if not wanted:
        return {}
    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    try:
        col = client.get_collection(SUMMARY_COLLECTION_NAME)
    except Exception:
        logger.warning("No summary collection yet — has anything been ingested?")
        return {}
    got = col.get(ids=[f"sum::{d}" for d in wanted], include=["documents", "metadatas"])
    out: dict[str, str] = {}
    for text, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
        doc = (meta or {}).get("document")
        if doc and text:
            out[doc] = text
    return out


_VERDICT_RE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(YES|NO)\b", re.I | re.M)


def parse_gate_verdicts(text: str, n: int):
    """Exactly one verdict per summary, indexed by number, or None.

    Same rule as Agent 5's batched citation-need check: a reply that cannot
    be aligned to its inputs is not partially trusted — it is discarded and
    the per-summary path runs.
    """
    found = {}
    for m in _VERDICT_RE.finditer(text or ""):
        idx = int(m.group(1))
        if idx in found:
            return None
        found[idx] = m.group(2).upper() == "YES"
    if set(found) != set(range(1, n + 1)):
        return None
    return [found[i] for i in range(1, n + 1)]


def _gate_one_by_one(query: str, ranked: list[dict]) -> list[dict]:
    kept = []
    for r in ranked:
        try:
            ans = chat([{
                "role": "user",
                "content": DOC_RELEVANCE_GATE.format(query=query, summary=r["summary"]),
            }]).content.strip().upper()
        except Exception as e:
            logger.warning("Gate call failed for %s: %s — keeping it.", r["document"], e)
            kept.append(r)
            continue
        if ans.startswith("Y"):
            kept.append(r)
    return kept


def gate_documents(query: str, ranked: list[dict]) -> list[dict]:
    """Ask the LLM to keep only the summaries that could be relevant prior work."""
    kept = None
    if GATE_BATCHED and ranked:
        summaries = "\n\n".join(f"{i}. {r['summary']}" for i, r in enumerate(ranked, 1))
        try:
            reply = chat([{
                "role": "user",
                "content": DOC_RELEVANCE_GATE_BATCH.format(query=query, n=len(ranked), summaries=summaries),
            }]).content
            verdicts = parse_gate_verdicts(reply, len(ranked))
        except Exception as e:
            logger.warning("Batched gate call failed: %s — falling back to per-summary calls.", e)
            verdicts = None
        if verdicts is None:
            logger.info("Batched gate reply could not be aligned to %d summaries — per-summary calls.", len(ranked))
        else:
            kept = [r for r, keep in zip(ranked, verdicts) if keep]
    if kept is None:
        kept = _gate_one_by_one(query, ranked)

    if not kept:
        logger.info("Gate rejected everything — falling back to the top summary.")
        return ranked[:1]
    logger.info("Gate kept %d/%d documents.", len(kept), len(ranked))
    return kept


# ─── Shortlist → keys → per-paper passages ──────────────────────────────────


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def dedupe_shortlist(ranked: list[dict]) -> list[dict]:
    """The same paper under two document keys (an arXiv copy and a DOI copy)
    must not take two slots. Keyed on the normalised title; the higher
    stage-1 score wins; order is score-descending."""
    best = {}
    for r in ranked:
        k = _norm_title(r.get("citation")) or r["document"]
        if k not in best or r.get("score", 0) > best[k].get("score", 0):
            best[k] = r
    return sorted(best.values(), key=lambda r: -r.get("score", 0))


def assign_keys(selected: list[dict]) -> dict:
    """P1..Pn in shortlist order. Short keys are what a 5B model reproduces
    exactly, and what check_citation_keys() can verify."""
    keys = {}
    for i, r in enumerate(selected, 1):
        r["key"] = f"P{i}"
        keys[r["key"]] = r.get("citation") or r["document"]
    return keys


def passages_per_paper(query: str, selected: list[dict], per_paper: int) -> dict:
    """One restricted hybrid search per shortlisted paper, so every paper on
    the shortlist is read — the global top-k left four of six unread."""
    collection, bm25, texts, metadatas = load_search_resources()
    out = {}
    for r in selected:
        hits = hybrid_search(
            query, collection, bm25, texts, metadatas,
            top_k=per_paper, doc_filter={r["document"]},
            exclude_types={"figure_description"},
        )
        for h in hits:
            h["key"] = r["key"]
        out[r["key"]] = hits
    return out


def format_passages(hits: list[dict], max_chars: int) -> str:
    parts, total = [], 0
    for h in hits:
        m = h.get("metadata") or {}
        block = f"(p.{m.get('page_first', m.get('page', '?'))}, {m.get('section') or 'text'}) {h.get('text', '')}"
        room = max_chars - total
        if room <= 0:
            break
        block = block[:room]
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


# ─── Stage 2 ────────────────────────────────────────────────────────────────



def deep_search(query: str, documents, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Hybrid chunk search restricted to `documents`."""
    collection, bm25, texts, metadatas = load_search_resources()
    return hybrid_search(
        query, collection, bm25, texts, metadatas,
        top_k=top_k, doc_filter=set(documents),
    )



# ─── Map: notes per paper ───────────────────────────────────────────────────


def paper_notes(query: str, key: str, title: str, hits: list[dict]):
    """One call over one paper's passages. Returns (notes, relevant)."""
    passages = format_passages(hits, SYNTHESIS_PER_PAPER_MAX_CHARS)
    text = chat(
        [{"role": "system", "content": SYNTHESIS_SYSTEM},
         {"role": "user", "content": PAPER_NOTES_USER.format(query=query, key=key, title=title, passages=passages)}],
        temperature=SYNTHESIS_TEMPERATURE, options=CHAT_OLLAMA_OPTIONS,
    ).content.strip()
    relevant = not text.lower().startswith("not relevant")
    return text, relevant


# ─── Reduce: the synthesis ──────────────────────────────────────────────────


def synthesise(query: str, material: str, n: int) -> str:
    return chat(
        [{"role": "system", "content": SYNTHESIS_SYSTEM},
         {"role": "user", "content": SYNTHESIS_USER.format(query=query, n=n, material=material)}],
        temperature=SYNTHESIS_TEMPERATURE, options=SYNTHESIS_OLLAMA_OPTIONS,
    ).content.strip()


_KEY_GROUP_RE = re.compile(r"\[(P\d+(?:\s*,\s*P\d+)*)\]")


def check_citation_keys(text: str, valid_keys):
    """Keys the synthesis cites, in first-use order, and the ones that are
    not on the shortlist — a 5B model can invent a [P7] as easily as a fact."""
    used = []
    for m in _KEY_GROUP_RE.finditer(text or ""):
        for k in re.split(r"\s*,\s*", m.group(1)):
            if k not in used:
                used.append(k)
    unknown = [k for k in used if k not in valid_keys]
    return used, unknown


# ─── End to end ─────────────────────────────────────────────────────────────


def research_answer(query: str, top_k: int = DEFAULT_TOP_K, mode: str | None = None, on_progress=None) -> dict | None:
    """Shortlist papers → read each one → synthesise, citing by key.

    Returns None when the corpus is empty. Otherwise a dict shaped as before
    (``suggestion`` / ``citations`` / ``passages`` / ``selected``) plus
    ``mode``, ``keys`` (key → title), ``notes`` (map_reduce only),
    ``unverified_citations`` (keys not on the shortlist), ``irrelevant_cited``
    and ``timings``. ``on_progress(stage, payload)`` is called with
    ``shortlist``, ``notes`` (once per paper) and ``synthesis``.
    """
    mode = (mode or SYNTHESIS_MODE).lower()
    if mode not in ("map_reduce", "single"):
        logger.warning("Unknown synthesis mode %r — using map_reduce.", mode)
        mode = "map_reduce"

    def emit(stage, payload):
        if on_progress:
            try:
                on_progress(stage, payload)
            except Exception as exc:                # noqa: BLE001 — progress must never break the answer
                logger.debug("on_progress raised: %s", exc)

    t_start = time.perf_counter()
    ranked = rank_documents(query)
    if not ranked:
        return None

    t0 = time.perf_counter()
    selected = gate_documents(query, ranked) if DOC_GATE else ranked
    selected = dedupe_shortlist(selected)
    keys = assign_keys(selected)
    t_gate = time.perf_counter() - t0
    emit("shortlist", {"papers": [{"key": r["key"], "citation": keys[r["key"]]} for r in selected]})

    per_paper = SYNTHESIS_PER_PAPER_CHUNKS if mode == "map_reduce" else 2
    try:
        passages = passages_per_paper(query, selected, per_paper)
    except Exception as e:
        logger.warning("Per-paper retrieval unavailable (%s) — using summaries only.", e)
        passages = {}
    all_hits = [h for r in selected for h in passages.get(r["key"], [])]

    notes, t_map = [], 0.0
    if mode == "map_reduce":
        t0 = time.perf_counter()
        for r in selected:
            key, title = r["key"], keys[r["key"]]
            hits = passages.get(key) or []
            t1 = time.perf_counter()
            if hits:
                try:
                    text, relevant = paper_notes(query, key, title, hits)
                except Exception as e:
                    logger.warning("Notes failed for %s (%s) — using its summary.", key, e)
                    text, relevant, hits = r.get("summary", ""), True, []
            else:
                text, relevant = r.get("summary", ""), True
            note = {"key": key, "document": r["document"], "citation": title, "notes": text,
                    "relevant": relevant, "passages_used": len(hits), "seconds": round(time.perf_counter() - t1, 1)}
            notes.append(note)
            emit("notes", note)
        t_map = time.perf_counter() - t0
        material = "\n\n".join(f"[{n['key']}] {n['citation']}\n{n['notes']}" for n in notes)
    else:
        blocks = []
        for r in selected:
            hits = passages.get(r["key"]) or []
            body = format_passages(hits, SYNTHESIS_PER_PAPER_MAX_CHARS) if hits else r.get("summary", "")
            blocks.append(f"[{r['key']}] {keys[r['key']]}\n{body}")
        material = "\n\n".join(blocks)

    t0 = time.perf_counter()
    answer = synthesise(query, material, len(selected))
    t_reduce = time.perf_counter() - t0
    emit("synthesis", {"seconds": round(t_reduce, 1)})

    used, unknown = check_citation_keys(answer, set(keys))
    irrelevant = {n["key"] for n in notes if not n["relevant"]}
    citations = [keys[k] for k in used if k in keys]

    return {
        "suggestion": answer,
        "citations": citations,
        "passages": all_hits,
        "selected": selected,
        "mode": mode,
        "keys": keys,
        "notes": notes,
        "unverified_citations": unknown,
        "irrelevant_cited": [k for k in used if k in irrelevant],
        "timings": {"gate": round(t_gate, 1), "map": round(t_map, 1), "reduce": round(t_reduce, 1),
                    "total": round(time.perf_counter() - t_start, 1)},
    }

