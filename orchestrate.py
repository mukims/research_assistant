"""
orchestrate.py — run the whole pipeline for one research idea, as a LangGraph.

    python orchestrate.py --query "topological protection in disordered wires"
    python orchestrate.py --query "..." --ask --workers 4 --force

The graph:

    START
      │
      ▼
    discover ──(no seed, --ask)──► fallback ──► END   answer from the model alone
      │       ──(no seed, no --ask)──────────► END
      ▼
    ingest_seed        index the seed paper itself, not just what it cites
      │
      ▼
    extract  ──(both strategies found nothing)────► END
      │
      ▼
    fetch              open-access PDFs for the seed's references   (Agent 2)
      │
      ▼
    ingest_refs        chunk, embed, summarise everything fetched   (Agent 3)
      │
      ├──(--ask)──► respond ──► END                 two-stage retrieval + synthesis
      └─────────────────────────► END

Each node wraps the same agent entry point you'd otherwise call by hand, and
every agent keeps its own on-disk state (seed_papers.json,
extracted_citations.json, downloaded.json, the ChromaDB store) — so a run that
dies partway is resumed by simply running this again.
"""

import argparse
import hashlib
import multiprocessing
import os
import time
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

from research_assistant.agents import (
    agent0_discoverer, agent1_extractor, agent2_fetcher, agent3_ingestor,
)
from research_assistant.config import GROBID_SERVER
from research_assistant.shared.ingestion import ingest_pdfs
from research_assistant.shared.log import get_logger
from research_assistant.shared import pipeline_status

logger = get_logger("orchestrate")


# ─── Graph state ─────────────────────────────────────────────────────────────


class _Inputs(TypedDict):
    query: str
    workers: int
    force: bool
    ask: bool


class PipelineState(_Inputs, total=False):
    # Optional: seed straight from this PDF / arXiv link instead of searching
    # (the fallback when the search finds nothing open-access).
    seed_url: Optional[str]
    # Optional: seed directly from an uploaded or local PDF file.
    seed_file: Optional[str]
    # Filled in as the graph runs.
    seed_path: Optional[str]
    seed_label: Optional[str]
    references_ok: bool
    extraction: Optional[dict]
    answer: Optional[dict]
    citation_audit: Optional[dict]
    audit_citations: Optional[bool]
    stopped: Optional[str]   # reason the run ended early, if it did
    # v2: the run's figure-analysis switch. None → config.FIGURE_VLM.
    describe_figures: Optional[bool]


def _banner(title: str) -> None:
    logger.info("─" * 60)
    logger.info(title)
    logger.info("─" * 60)


# ─── Nodes ───────────────────────────────────────────────────────────────────


def discover(state: PipelineState) -> dict:
    query = (state.get("query") or "").strip()
    force = state.get("force", False)
    seed_url = (state.get("seed_url") or "").strip()
    seed_file = (state.get("seed_file") or "").strip()

    with pipeline_status.track_stage("discover", "Finding seed paper", current_step=1, total_steps=5):
        if seed_file:
            _banner("discover — seeding from uploaded / local PDF")
            pipeline_status.update_progress(detail=f"Loading uploaded PDF: {os.path.basename(seed_file)}", current_item_name=os.path.basename(seed_file))
            path = agent0_discoverer.discover_from_file(query, seed_file, force=force)
            fail = f"could not load seed PDF ({os.path.basename(seed_file)}) — file is missing or not a valid PDF"
        elif seed_url:
            _banner("discover — seeding from the supplied link")
            pipeline_status.update_progress(detail=f"Downloading from link: {seed_url[:60]}", current_item_name=seed_url)
            path = agent0_discoverer.discover_from_url(query, seed_url, force=force)
            fail = f"could not download a PDF from {seed_url}"
        else:
            _banner("discover — finding a seed paper")
            pipeline_status.update_progress(detail=f"Searching literature for: {query[:50]}", current_item_name=query)
            path = agent0_discoverer.discover(query, force=force)
            fail = (
                "no open-access PDF found for that query — supply an arXiv or "
                "open-access PDF link or upload a PDF to seed from directly"
            )

        if not path:
            pipeline_status.add_event(f"⚠️ No open-access seed PDF found for query: {query[:40]}")
            return {"seed_path": None, "stopped": fail}

        seed = agent0_discoverer.get_seed(query) or {}
        if not seed:
            seed, q = agent0_discoverer.get_seed_by_path(path)
            if q:
                query = q
            else:
                seed = {}

        label = seed.get("title") or seed.get("key") or os.path.basename(path)
        pipeline_status.add_event(f"✅ Found seed paper: {label[:60]}")
        pipeline_status.update_progress(current_item_name=label, detail=f"Seed paper identified: {label[:60]}")
        result = {"seed_path": path, "seed_label": label}
        if query and query != state.get("query"):
            result["query"] = query
        return result


def ingest_seed(state: PipelineState) -> dict:
    _banner("ingest_seed — indexing the seed paper itself")
    label = state.get("seed_label") or os.path.basename(state.get("seed_path", "seed"))
    with pipeline_status.track_stage("ingest_seed", "Indexing seed paper", current_step=2, total_steps=5):
        pipeline_status.update_progress(current_item_name=label, detail=f"Indexing seed paper: {label[:60]}")
        pipeline_status.add_event(f"📥 Indexing seed paper: {label[:60]}")
        ingest_pdfs(
            {state["seed_path"]: state["seed_label"]},
            workers=state.get("workers", 1),
            skip_ingested=not state.get("force", False),
            describe_figures=state.get("describe_figures"),
        )
        pipeline_status.add_event(f"✅ Seed paper indexed: {label[:60]}")
        return {}


def extract(state: PipelineState) -> dict:
    _banner("extract — mining the seed's reference list")
    with pipeline_status.track_stage("extract", "Extracting reference list", current_step=3, total_steps=5, detail="Mining reference list from seed paper"):
        result = agent1_extractor.run_extractor()

        if not result or not result.get("reference_count"):
            # Both strategies came up empty: GROBID down *and* no recognisable numbered
            # reference list. There is nothing to fetch, so stop — but say which failed.
            fail_msg = (
                "No references could be extracted from the seed paper. GROBID at "
                f"{GROBID_SERVER} is unreachable and pattern-based extraction found no "
                "numbered reference list — the PDF may be a scan with no text layer. "
                f"Check: curl {GROBID_SERVER}/api/isalive"
            )
            pipeline_status.add_event("⚠️ No references extracted from seed paper")
            return {"references_ok": False, "stopped": fail_msg}
        return {"references_ok": True, "extraction": result}


def fetch(state: PipelineState) -> dict:
    _banner("fetch — downloading the referenced papers (Agent 2)")
    with pipeline_status.track_stage("fetch", "Fetching referenced papers", current_step=4, total_steps=5, detail="Downloading open-access reference PDFs"):
        agent2_fetcher.fetch_papers()
        return {}


def ingest_refs(state: PipelineState) -> dict:
    _banner("ingest_refs — ingesting the reference PDFs (Agent 3)")
    with pipeline_status.track_stage("ingest_refs", "Ingesting and summarizing papers", current_step=5, total_steps=5, detail="Ingesting referenced papers into corpus"):
        agent3_ingestor.run_ingestor(
            workers=state.get("workers", 1), force=state.get("force", False),
            describe_figures=state.get("describe_figures"),
        )
        return {}


def respond(state: PipelineState) -> dict:
    _banner("respond — related-work synthesis for the query")
    from research_assistant.shared import retrieve  # imported here so corpus-building stays light

    with pipeline_status.track_stage("respond", "Synthesizing answer", current_step=5, total_steps=5, detail=f"Synthesizing related-work response for: {state.get('query', '')[:50]}"):
        pipeline_status.add_event("✍️ Formulating related-work synthesis…")
        result = retrieve.research_answer(state["query"])
        pipeline_status.add_event("✅ Related-work synthesis complete")

        audit_result = None
        if state.get("audit_citations") and state.get("seed_path"):
            try:
                from research_assistant.shared import seed_audit
                pipeline_status.add_event("🔍 Auditing in-text citations from seed paper…")
                audit_result = seed_audit.audit_seed_citations(state["seed_path"])
                pipeline_status.add_event("✅ Seed citation audit complete")
            except Exception as e:
                logger.warning("Seed citation audit encountered an error: %s", e)
                audit_result = {"error": str(e), "totals": {"total": 0, "judged": 0}, "results": []}

        out = {"answer": result}
        if audit_result is not None:
            out["citation_audit"] = audit_result
        return out


def fallback(state: PipelineState) -> dict:
    """No seed paper — answer the query from the model's own knowledge."""
    _banner("fallback — no corpus, answering from general knowledge")
    from research_assistant.prompts import NO_CORPUS_FALLBACK
    from research_assistant.shared.llm import chat

    with pipeline_status.track_stage("respond", "Synthesizing fallback answer", current_step=5, total_steps=5, detail="Generating ungrounded answer from general knowledge"):
        pipeline_status.add_event("💭 Generating ungrounded fallback answer…")
        try:
            text = chat(
                [{"role": "user", "content": NO_CORPUS_FALLBACK.format(query=state["query"])}]
            ).content
        except Exception as e:  # noqa: BLE001
            logger.error("Fallback answer failed: %s", e)
            return {}
        return {"answer": {"suggestion": text, "citations": [], "passages": [], "ungrounded": True}}



# ─── Edges ───────────────────────────────────────────────────────────────────


def _after_discover(state: PipelineState) -> str:
    if state.get("seed_path"):
        return "ingest_seed"
    # No paper to build on. Answer from general knowledge if the caller wanted
    # an answer and a research query was provided; otherwise just stop.
    return "fallback" if (state.get("ask") and (state.get("query") or "").strip()) else END


def _after_extract(state: PipelineState) -> str:
    return "fetch" if state.get("references_ok") else END


def _after_ingest_refs(state: PipelineState) -> str:
    return "respond" if state.get("ask") else END


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("discover", discover)
    g.add_node("ingest_seed", ingest_seed)
    g.add_node("extract", extract)
    g.add_node("fetch", fetch)
    g.add_node("ingest_refs", ingest_refs)
    g.add_node("respond", respond)
    g.add_node("fallback", fallback)

    g.add_edge(START, "discover")
    g.add_conditional_edges("discover", _after_discover, ["ingest_seed", "fallback", END])
    g.add_edge("ingest_seed", "extract")
    g.add_conditional_edges("extract", _after_extract, ["fetch", END])
    g.add_edge("fetch", "ingest_refs")
    g.add_conditional_edges("ingest_refs", _after_ingest_refs, ["respond", END])
    g.add_edge("respond", END)
    g.add_edge("fallback", END)

    return g.compile(checkpointer=MemorySaver())


# ─── Entry point ─────────────────────────────────────────────────────────────


def run(
    query: str = "",
    workers: int = 1,
    force: bool = False,
    ask: bool = False,
    seed_url: str | None = None,
    seed_file: str | None = None,
    describe_figures: bool | None = None,
    audit_citations: bool = False,
) -> int:
    app = build_graph()
    thread_key = query or (os.path.abspath(seed_file) if seed_file else "") or (seed_url or "") or "run"
    thread_id = hashlib.sha1(thread_key.encode()).hexdigest()[:12]
    config = {"configurable": {"thread_id": thread_id}}

    query_display = query or (os.path.basename(seed_file) if seed_file else "") or (seed_url or "") or "topic"
    pipeline_status.set_status(
        active=True,
        stage="discover",
        stage_label="Finding seed paper",
        current_step=1,
        total_steps=5,
        detail=f"Starting pipeline for: {query_display[:60]}",
    )
    pipeline_status.add_event(f"🚀 Pipeline started for: {query_display[:50]}")

    final: PipelineState = {}
    try:
        for update in app.stream(
            {
                "query": query,
                "workers": workers,
                "force": force,
                "ask": ask,
                "seed_url": seed_url,
                "seed_file": seed_file,
                "describe_figures": describe_figures,
                "audit_citations": audit_citations,
            },
            config=config,
            stream_mode="values",
        ):
            final = update
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            pipeline_status.add_event("⚠️ Pipeline cancelled by user (SIGINT)")
            pipeline_status.set_status(
                active=False,
                stage="idle",
                stage_label="Idle",
                detail="Pipeline cancelled by user",
            )
        else:
            pipeline_status.add_event(f"❌ Pipeline failed: {exc}")
            pipeline_status.set_status(
                active=False,
                stage="idle",
                stage_label="Idle",
                detail=f"Pipeline failed: {exc}",
            )
        raise

    stopped = final.get("stopped")
    if stopped:
        logger.warning("Pipeline stopped: %s", stopped)
        pipeline_status.add_event(f"⚠️ Pipeline stopped early: {stopped[:60]}")
        pipeline_status.set_status(
            active=False,
            stage="idle",
            stage_label="Idle",
            detail=f"Stopped: {stopped[:60]}",
        )
    else:
        effective_query = final.get("query") or query or final.get("seed_label") or "paper"
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pipeline_status.add_event(f"✅ Pipeline completed: {effective_query[:40]}")
        pipeline_status.set_status(
            active=False,
            stage="idle",
            stage_label="Idle",
            detail="Pipeline completed successfully",
            last_completed_at=now_iso,
            last_summary=f"Completed query: {effective_query[:40]}",
        )

    result = final.get("answer")
    if result:
        if result.get("ungrounded"):
            print("\n=== ANSWER (not grounded in retrieved papers) ===")
        else:
            print("\n--- Grounding references ---")
            for cit in result["citations"]:
                print(f" > {cit}")
            print("\n=== SUGGESTION ===")
        print(result["suggestion"])
        print("==================\n")
    elif ask and not stopped:
        logger.info("Nothing in the corpus matched the query yet.")

    effective_query = final.get("query") or query or final.get("seed_label") or "paper"
    logger.info("Pipeline complete for: %s", effective_query)
    return 1 if stopped else 0



def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline for one research idea.")
    parser.add_argument("--query", default="", help="The research idea to seed from.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes for the ingestion stages.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run every stage even if its output already exists.",
    )
    parser.add_argument(
        "--ask", action="store_true",
        help="After building the corpus, answer the original query with Agent 4.",
    )
    parser.add_argument(
        "--seed-url",
        help="Seed from this PDF / arXiv link instead of searching for a paper.",
    )
    parser.add_argument(
        "--seed-file",
        help="Seed directly from a local PDF file instead of searching.",
    )
    parser.add_argument(
        "--describe-figures", action="store_true", default=None,
        help="Index v2: analyse every figure and table with the model during this run.",
    )
    parser.add_argument(
        "--audit-citations", action="store_true", default=False,
        help="Audit in-text citations from the seed paper against fetched references.",
    )
    args = parser.parse_args()

    if not args.query and not args.seed_file and not args.seed_url:
        parser.error("Must provide --query, --seed-file, or --seed-url")

    raise SystemExit(run(
        args.query, workers=args.workers, force=args.force, ask=args.ask,
        seed_url=args.seed_url, seed_file=args.seed_file,
        describe_figures=args.describe_figures,
        audit_citations=args.audit_citations,
    ))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
