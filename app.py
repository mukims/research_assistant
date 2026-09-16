"""
Streamlit front-end for the Research Assistant pipeline (also the container entry
point).

Tab 1 runs the LangGraph orchestrator from a research idea. Tab 2 is the
Agent 4 citation assistant over whatever corpus has been built so far. Tab 3
runs Agent 5 (batch citer) over a whole draft. Tab 4 is Agent 7's multi-turn
research chat. Tab 5 renders HOW_TO_USE.md.
"""

import hashlib
import json
import os
import re
import tempfile
import time

import streamlit as st

from research_assistant import config
from research_assistant.agents.agent8_verifier import verify_draft
from research_assistant.shared import pipeline_status
from research_assistant.shared.atomic import atomic_write_json

st.set_page_config(page_title="Citation Needed! · Marvin the Citebot", page_icon="📚", layout="wide")
os.makedirs(config.DATA_DIR, exist_ok=True)


def _check_auth() -> bool:
    app_pwd = os.environ.get("APP_PASSWORD")
    if not app_pwd:
        try:
            if hasattr(st, "secrets") and "password" in st.secrets:
                app_pwd = st.secrets["password"]
        except Exception:
            app_pwd = None

    if not app_pwd:
        return True
    if st.session_state.get("authenticated", False):
        return True

    st.markdown("### 🔐 Research Assistant — Access Verification")
    st.caption("This research instance is currently protected during review. Please enter the access code to continue.")
    with st.form("auth_form", clear_on_submit=False):
        entered = st.text_input("Access Code", type="password", placeholder="Enter access password")
        submitted = st.form_submit_button("Unlock", type="primary")
        if submitted:
            if entered.strip() == app_pwd.strip():
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid access code. Please verify the code and try again.", icon="🚫")
    return False


if not _check_auth():
    st.stop()


@st.cache_resource(show_spinner=False)
def get_cached_search_resources(mtime: float):
    """Load ChromaDB collection and BM25 index once into memory for all sessions.
    
    Keyed on the modification time of BM25_INDEX_PATH so re-indexing automatically
    refreshes the shared cache without needing server restarts.
    """
    from research_assistant.shared.db import load_search_resources
    return load_search_resources()


def _get_resources_mtime() -> float:
    if os.path.exists(config.BM25_INDEX_PATH):
        return os.path.getmtime(config.BM25_INDEX_PATH)
    return 0.0


def _is_ingest_locked() -> bool:
    return pipeline_status._is_ingest_locked()


# ─── Data helpers ───────────────────────────────────────────────────────────


def _graph():
    # Rebuilt per run so each build starts from a clean checkpointer.
    # orchestrate.py is a root-level entry point, not a package module —
    # it must be imported bare, not as research_assistant.orchestrate.
    import orchestrate

    return orchestrate.build_graph()


@st.cache_data(ttl=30, show_spinner=False)
def _corpus_stats():
    """(chunks, papers) — cheap, tolerant of a missing/empty store."""

    chunks = 0
    try:
        import chromadb

        chunks = (
            chromadb.PersistentClient(path=config.VECTORDB_PATH)
            .get_collection(config.COLLECTION_NAME)
            .count()
        )
    except Exception:
        pass

    papers = 0
    for path in (config.SEED_PAPERS_PATH, config.DOWNLOADED_JSON_PATH):
        try:
            with open(path) as fh:
                papers += len(json.load(fh))
        except Exception:
            pass

    try:
        from research_assistant.shared import manifest
        manifest_count = len(manifest.load())
        papers = max(papers, manifest_count)
    except Exception:
        pass
    return chunks, papers


@st.cache_data(ttl=30, show_spinner=False)
def _grobid_ok():
    try:
        from research_assistant.agents.grobid_controller import GrobidAgent

        return GrobidAgent().is_alive(timeout=3.0)
    except Exception:
        return False


def _clear_grobid_cache():
    if hasattr(_grobid_ok, "clear"):
        _grobid_ok.clear()


def _render_grobid_controls():
    try:
        from research_assistant.agents.grobid_controller import GrobidAgent

        agent = GrobidAgent()

        if "grobid_feedback" in st.session_state:
            fb_type, fb_msg = st.session_state.pop("grobid_feedback")
            if fb_type == "success":
                st.success(fb_msg, icon="✅")
            elif fb_type == "error":
                st.error(fb_msg, icon="🚫")
            elif fb_type == "warning":
                st.warning(fb_msg, icon="⚠️")
            else:
                st.info(fb_msg, icon="ℹ️")

        status = agent.check_status()
        st.session_state["grobid_server_state"] = status.state

        if status.state == "RUNNING":
            st.success(f"🟢 **GROBID Status: Running**\n\n`{status.server_url}`")
        elif status.state == "STARTING":
            st.warning(f"🟡 **GROBID Status: Starting...**\n\n`{status.server_url}`")
            if status.message:
                st.caption(status.message)
        elif status.state == "ERROR":
            st.error(f"⚠️ **GROBID Status: Error**\n\n`{status.server_url}`")
            if status.message:
                st.caption(status.message)
            if status.details:
                st.caption(status.details)
        else:
            st.error(f"🔴 **GROBID Status: Stopped**\n\n`{status.server_url}`")
            st.caption(
                "Reference extraction falls back to regex without metadata (authors, years, DOIs)."
            )

        if status.is_alive:
            c_stop, c_restart, c_chk = st.columns([3, 3, 2])
            if c_stop.button("⏹️ Stop", key="btn_stop_grobid", use_container_width=True, help="Stop GROBID server"):
                with st.spinner("Stopping GROBID server..."):
                    ok, msg = agent.stop_server()
                    _clear_grobid_cache()
                    st.session_state["grobid_feedback"] = ("info" if ok else "error", msg)
                    st.session_state["grobid_server_state"] = "STOPPED" if ok else "ERROR"
                    st.rerun()
            if c_restart.button("🔄 Restart", key="btn_restart_grobid", use_container_width=True, help="Restart GROBID server"):
                with st.spinner("Restarting GROBID server..."):
                    ok, msg = agent.restart_server()
                    _clear_grobid_cache()
                    st.session_state["grobid_feedback"] = ("success" if ok else "error", msg)
                    st.session_state["grobid_server_state"] = "RUNNING" if ok else "ERROR"
                    st.rerun()
            if c_chk.button("🩺 Status", key="btn_check_grobid_running", use_container_width=True, help="Probe server health"):
                _clear_grobid_cache()
                new_status = agent.check_status()
                st.session_state["grobid_server_state"] = new_status.state
                if new_status.is_alive:
                    st.session_state["grobid_feedback"] = ("success", f"GROBID is active at {new_status.server_url}.")
                else:
                    st.session_state["grobid_feedback"] = (
                        "warning" if new_status.state == "STARTING" else "error",
                        new_status.message or f"GROBID server is {new_status.state.lower()}."
                    )
                st.rerun()
        elif status.state == "STARTING":
            c_chk, c_stop = st.columns([3, 2])
            if c_chk.button("🩺 Refresh Status", key="btn_check_grobid_starting", use_container_width=True, help="Probe if JVM has finished initializing"):
                _clear_grobid_cache()
                new_status = agent.check_status()
                st.session_state["grobid_server_state"] = new_status.state
                if new_status.is_alive:
                    st.session_state["grobid_feedback"] = ("success", f"GROBID is active at {new_status.server_url}.")
                else:
                    st.session_state["grobid_feedback"] = (
                        "warning" if new_status.state == "STARTING" else "error",
                        new_status.message or f"GROBID server is {new_status.state.lower()}."
                    )
                st.rerun()
            if c_stop.button("⏹️ Stop", key="btn_stop_grobid_starting", use_container_width=True, help="Stop initializing container"):
                with st.spinner("Stopping GROBID container..."):
                    ok, msg = agent.stop_server()
                    _clear_grobid_cache()
                    st.session_state["grobid_feedback"] = ("info" if ok else "error", msg)
                    st.session_state["grobid_server_state"] = "STOPPED" if ok else "ERROR"
                    st.rerun()
        else:
            c_start, c_chk = st.columns([3, 2])
            if c_start.button("▶️ Start GROBID", key="btn_start_grobid", use_container_width=True, help="Launch GROBID server agent"):
                with st.spinner("Launching GROBID agent & checking health..."):
                    ok, msg = agent.start_server()
                    _clear_grobid_cache()
                    new_status = agent.check_status()
                    st.session_state["grobid_server_state"] = new_status.state
                    if new_status.is_alive:
                        st.session_state["grobid_feedback"] = ("success", msg)
                    elif new_status.state == "STARTING":
                        st.session_state["grobid_feedback"] = ("warning", msg)
                    else:
                        st.session_state["grobid_feedback"] = ("error", msg)
                    st.rerun()
            if c_chk.button("🩺 Check Status", key="btn_check_grobid_stopped", use_container_width=True, help="Probe server health"):
                _clear_grobid_cache()
                new_status = agent.check_status()
                st.session_state["grobid_server_state"] = new_status.state
                if new_status.is_alive:
                    st.session_state["grobid_feedback"] = ("success", f"GROBID is active at {new_status.server_url}.")
                else:
                    st.session_state["grobid_feedback"] = (
                        "warning" if new_status.state == "STARTING" else "error",
                        new_status.message or f"GROBID server is {new_status.state.lower()}."
                    )
                st.rerun()

        with st.expander("🛠️ GROBID Diagnostics"):
            st.markdown(f"**Endpoint:** `{status.server_url}`")
            st.markdown(f"**Container / Target:** `{status.container_name}`")
            docker_ok = status.docker_available
            st.markdown(f"**Docker Status:** {'🟢 Available' if docker_ok else '🔴 Unavailable'}")
            if not docker_ok and status.details:
                st.warning(status.details)
            if status.container_status:
                st.markdown(f"**Container Status:** `{status.container_status}`")
            if status.image:
                st.markdown(f"**Docker Image:** `{status.image}`")
            st.markdown("**Manual launch command:**")
            st.code(status.manual_command or agent.get_manual_command(), language="bash")
    except Exception as exc:
        st.error(f"GROBID Agent encountered an error: {exc}")


def _manifest(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


# ─── Rendering ──────────────────────────────────────────────────────────────

STEPS = {
    "discover":    ("🔍", "Finding a seed paper"),
    "ingest_seed": ("📥", "Indexing the seed paper"),
    "extract":     ("📖", "Extracting the reference list"),
    "fetch":       ("🌐", "Fetching referenced papers"),
    "ingest_refs": ("🧩", "Ingesting reference PDFs"),
    "respond":     ("✍️", "Answering your query"),
    "fallback":    ("💭", "No paper found — answering from general knowledge"),
}


def _render_passages(passages, label="Retrieved context"):
    """The retrieved-chunk expander, shared by every tab that shows one."""
    if not passages:
        return
    with st.expander(f"{label} · {len(passages)} passage(s)"):
        for i, p in enumerate(passages, 1):
            m = p.get("metadata") or {}
            st.caption(
                f"{i}. **{m.get('citation_source', '?')}** — "
                f"{m.get('document', '?')} · p.{m.get('page', '?')}"
            )
            st.text((p.get("text") or "")[:900])
            if i < len(passages):
                st.divider()


def _render_suggestion(result):
    with st.container(border=True):
        st.markdown("###### Suggested text")
        st.markdown(result["suggestion"])

    cits = result.get("citations") or []
    if cits:
        st.markdown("**Grounded in**")
        for c in cits:
            st.markdown(f"- {c}")

    _render_passages(result.get("passages") or [])


def _stage_upload(uploaded_file) -> str:
    """Write an uploaded PDF somewhere the pipeline can seed from, and return it.

    Agent 0 copies the paper into RAW_DIR under a key derived from its DOI,
    arXiv id or content hash, so this staging copy is pure duplication — it
    exists only because the graph's state carries a path, not bytes. So it goes
    to a temp file the caller deletes once the run finishes, rather than under
    DATA_DIR where copies would pile up and two uploads sharing a filename
    would overwrite each other.
    """
    fd, path = tempfile.mkstemp(prefix="seed_upload_", suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return path


def _render_seed_and_downloads(query, final):
    # If the pipeline stopped without establishing a seed, do not render phantom seeds or prior references.
    if not final.get("seed_path") and final.get("stopped"):
        return

    seeds = _manifest(config.SEED_PAPERS_PATH)
    seed = seeds.get(query) or {}
    if not seed and final.get("seed_path"):
        norm_path = os.path.abspath(final["seed_path"])
        base_name = os.path.basename(final["seed_path"])
        for s in reversed(list(seeds.values())):
            p = s.get("path")
            if p and (os.path.abspath(p) == norm_path or os.path.basename(p) == base_name):
                seed = s
                break
    # Query-key mismatch fallback: only if a seed was actually established for this run
    if not seed and final.get("seed_path") and len(seeds) == 1:
        seed = next(iter(seeds.values()))

    title = seed.get("title") or final.get("seed_label")
    if (title or seed) and final.get("seed_path"):
        with st.container(border=True):
            st.markdown(f"**🌱 Seed paper** — {title or seed.get('key', '—')}")
            bits = []
            if seed.get("url"):
                bits.append(f"[PDF]({seed['url']})")
            elif seed.get("source") == "upload":
                bits.append("📁 Uploaded PDF")
            if seed.get("arxiv_id"):
                bits.append(f"arXiv:{seed['arxiv_id']}")
            if seed.get("doi"):
                bits.append(f"doi:{seed['doi']}")
            if seed.get("key"):
                bits.append(f"`{seed['key']}`")
            if bits:
                st.caption(" · ".join(bits))

    if final.get("seed_path") and (final.get("extraction") or final.get("references_ok") or final.get("answer")):
        from research_assistant.agents.agent2_fetcher import source_key

        extracted_data = _manifest(config.EXTRACTED_CITATIONS_PATH)
        extracted_refs = extracted_data.get("references", []) if isinstance(extracted_data, dict) else []
        target_keys = {source_key(r) for r in extracted_refs if isinstance(r, dict) and source_key(r)} if extracted_refs else set()

        all_downloaded = _manifest(config.DOWNLOADED_JSON_PATH)
        all_failed = _manifest(config.FAILED_DOWNLOADS_PATH)

        if target_keys:
            downloaded = [rec for k, rec in all_downloaded.items() if k in target_keys]
            failed = [rec for k, rec in all_failed.items() if k in target_keys]
        else:
            downloaded = list(all_downloaded.values())
            failed = list(all_failed.values())

        if downloaded or failed:
            total_extracted = len(extracted_refs) if extracted_refs else (len(downloaded) + len(failed))
            label = f"📄 Reference PDFs — {len(downloaded)} fetched, {len(failed)} unavailable"
            if total_extracted:
                label += f" ({total_extracted} references found)"
            with st.expander(
                label,
                expanded=bool(downloaded) and not final.get("answer"),
            ):
                for rec in downloaded:
                    t = rec.get("title") or rec.get("raw_reference") or rec.get("key")
                    st.markdown(f"- ✅ {t}")
                for rec in failed:
                    t = rec.get("title") or rec.get("raw_reference") or rec.get("key")
                    st.markdown(
                        f"- ⚠️ {t}  \n  <sub>{rec.get('reason', '')}</sub>",
                        unsafe_allow_html=True,
                    )


def _render_shortlist(selected):
    if not selected:
        return
    st.markdown(f"**Relevant prior work** — {len(selected)} paper(s)")
    for r in selected:
        with st.expander(f"{r.get('citation') or r.get('document')}  ·  score {r.get('score', '')}"):
            st.write(r.get("summary", ""))


def _synthesis_warnings(answer) -> list:
    """What the reader must be told about the synthesis's citations."""
    warnings = []
    unknown = answer.get("unverified_citations") or []
    if unknown:
        warnings.append(f"The synthesis cites {', '.join(unknown)}, which are not on the shortlist — "
                        "treat those sentences as unsupported.")
    irrelevant = answer.get("irrelevant_cited") or []
    if irrelevant:
        keys = answer.get("keys") or {}
        named = ", ".join(f"{k} ({keys.get(k, '?')})" for k in irrelevant)
        warnings.append(f"The synthesis cites {named}, whose notes say the paper is not relevant to the idea.")
    return warnings


def _render_notes(answer):
    notes = answer.get("notes") or []
    if not notes:
        return
    st.markdown(f"**What each paper says** — {len(notes)} paper(s) read")
    for n in notes:
        flag = "" if n.get("relevant", True) else " · not relevant"
        with st.expander(f"{n['key']} · {n['citation']}{flag}  ·  {n.get('passages_used', 0)} passage(s), {n.get('seconds', '?')}s"):
            st.write(n["notes"])


def _render_synthesis(answer, heading):
    for w in _synthesis_warnings(answer):
        st.warning(w, icon="⚠️")
    with st.container(border=True):
        st.markdown(f"###### {heading}")
        st.markdown(answer["suggestion"])
    keys = answer.get("keys") or {}
    if keys:
        st.caption("Keys: " + " · ".join(f"{k} = {t}" for k, t in keys.items()))
    elif answer.get("citations"):
        st.caption("Sources: " + " · ".join(str(c) for c in answer["citations"]))
    t = answer.get("timings")
    if t:
        st.caption(f"{answer.get('mode', '')}: gate {t['gate']}s · notes {t['map']}s · synthesis {t['reduce']}s · total {t['total']}s")


_VERDICT_BADGES = {
    "Supports": "🟢 Supports",
    "Partially supports": "🟡 Partially Supports",
    "Contradicts": "🔴 Contradicts",
    "Does not support": "🟠 Does Not Support",
    "Unclear / insufficient evidence": "⚪ Unclear / Insufficient Evidence",
}
_NOT_ASSESSED_BADGES = {
    "deferred_paywalled": "⏳ Deferred (Pending Evidence)",
    "not_downloaded": "🔒 Paywalled / Not In Corpus",
    "cap_exceeded": "⏸ Not Assessed (budget)",
    "cluster_skipped": "⏸ Not Assessed (cluster)",
    "not_attempted": "⏸ Not Assessed (backend down)",
    "retrieval_failed": "⏸ Not Assessed (retrieval failed)",
    "call_failed": "⏸ Not Assessed (model call failed)",
    "parse_failed": "⏸ Not Assessed (unparseable reply)",
    "no_evidence": "⏸ Not Assessed (no passages found)",
    "malformed_claim": "⏸ Not Assessed (sentence fragment)",
    "unresolved_ref": "⏸ Not Assessed (unmatched reference)",
}


def _audit_item_badge(item: dict) -> str:
    """The label a citation row wears. Outcome first: a verdict label is only
    ever shown for an item the judge produced a verdict for."""
    outcome = item.get("outcome")
    if outcome == "judged":
        return _VERDICT_BADGES.get(item.get("judgement") or "", "⚪ Unclear / Insufficient Evidence")
    if outcome == "not_a_claim":
        return f"🔧 Not a Claim ({item.get('role') or 'non-evidential'})"
    return _NOT_ASSESSED_BADGES.get(outcome or "", f"⏸ Not Assessed ({outcome or 'unknown'})")


def _render_seed_citation_audit(final):

    audit = final.get("citation_audit")
    seed_path = final.get("seed_path") or (audit.get("seed_path") if audit else None)

    # If audit has not been run yet, offer an explicit button when a seed PDF is available
    if not audit and seed_path and os.path.exists(seed_path):
        from research_assistant.shared.seed_audit import find_tei_for_seed

        if find_tei_for_seed(seed_path):
            with st.container(border=True):
                st.markdown("###### 🔍 In-Text Citation Audit")
                st.caption(
                    "Audit whether the references cited inside this uploaded paper actually support "
                    "the statements made in the paper's text."
                )
                if st.button(
                    "Audit Seed Paper Citations",
                    key="btn_run_seed_audit",
                    type="secondary",
                ):
                    with st.status(
                        "Auditing in-text citations from seed paper…", expanded=True
                    ) as status:
                        from research_assistant.shared.seed_audit import audit_seed_citations

                        cached_res = get_cached_search_resources(_get_resources_mtime())
                        audit_res = audit_seed_citations(seed_path, search_resources=cached_res)
                        final["citation_audit"] = audit_res
                        st.session_state["build_result"] = final
                        status.update(label="Citation audit complete!", state="complete")
                        st.rerun()

    if not audit:
        return

    totals = audit.get("totals") or {}
    results = audit.get("results") or []

    if not results:
        if audit.get("error"):
            st.info(f"Seed citation audit: {audit['error']}")
        return

    with st.container(border=True):
        from research_assistant.shared.seed_audit import (
            explain_rubric_verdict,
            generate_seed_audit_markdown,
            get_deferred_missing_references,
            save_and_register_reference_pdf,
        )

        stem = (
            os.path.splitext(os.path.basename(seed_path))[0]
            if seed_path
            else "seed_paper"
        )

        if final.get("from_cache") or audit.get("from_cache"):
            summary = final.get("cross_check_summary") or audit.get("cross_check_summary") or {}
            dur = summary.get("duration_seconds")
            dur_str = f" in **{dur:.1f}s**" if dur is not None else ""
            rejudged = summary.get("newly_judged_count", 0)
            rejudged_str = f" · {rejudged} newly available reference(s) evaluated" if rejudged > 0 else ""
            st.info(
                f"⚡ **Loaded from Persistent Audit Cache{dur_str}**: "
                f"Cross-checked {totals.get('total', 0)} citation claims against current corpus{rejudged_str}. "
                f"Reliability policy and source assessments verified in real time.",
                icon="⚡",
            )

        # Header with Title and Download Buttons
        c_title, c_dl1, c_dl2 = st.columns([3, 1, 1])
        with c_title:
            st.markdown("##### 🔍 Uploaded Paper Citation Audit")
            st.caption(
                "Verifies whether the references cited inside the uploaded paper support "
                "the statements made in its text, evaluated with the formal 3-slot judgement rubric."
            )

        md_report = generate_seed_audit_markdown(audit, seed_title=stem)
        with c_dl1:
            st.download_button(
                "📥 Download Report (.md)",
                data=md_report,
                file_name=f"{stem}_citation_verification_report.md",
                mime="text/markdown",
                key="btn_dl_seed_audit_md",
                help="Download complete, detailed verification report in Markdown with all citations, slots, and reasoning.",
            )
        with c_dl2:
            st.download_button(
                "📥 Download Data (.json)",
                data=json.dumps(audit, indent=2, ensure_ascii=False),
                file_name=f"{stem}_citation_audit.json",
                mime="application/json",
                key="btn_dl_seed_audit_json",
                help="Download raw structured JSON audit data.",
            )

        coverage = totals.get("coverage", {})
        if audit.get("aborted"):
            st.error(
                f"⚠️ Audit stopped early after {audit['aborted'].get('after_attempted')} attempt(s): "
                f"{audit['aborted'].get('reason')} — citations after that point were not attempted. "
                "Re-run once the backend is reachable."
            )
        if coverage:
            st.caption(
                f"Assessed by the model: **{coverage.get('judged', 0)}** of {coverage.get('downloaded', 0)} citations "
                f"whose reference is in the corpus ({coverage.get('attempted', 0)} attempted)."
            )

        # Top metric row
        cols = st.columns(6)
        cols[0].metric("Citations Found", totals.get("total", len(results)))
        cols[1].metric("Supports 🟢", totals.get("Supports", 0))
        cols[2].metric("Partially 🟡", totals.get("Partially supports", 0))
        needs_rev = totals.get("Contradicts", 0) + totals.get("Does not support", 0)
        cols[3].metric("Need Review 🔴", needs_rev)
        cols[4].metric("Deferred (Pending) ⏳", totals.get("deferred_paywalled", 0))
        cols[5].metric("Paywalled / Unchecked ⚪", totals.get("not_downloaded", 0))

        rel_totals = totals.get("reliability") or {}
        if rel_totals:
            st.caption("🛡️ **Scientific Evidence Reliability Assessment:**")
            rcols = st.columns(6)
            rcols[0].metric("High Reliability 🟢", rel_totals.get("high", 0), help="Peer-reviewed primary literature with verified verbatim evidence span")
            rcols[1].metric("Moderate 🟡", rel_totals.get("moderate", 0), help="Preprints, secondary reviews, or qualified claims")
            rcols[2].metric("Low / Flagged 🟠", rel_totals.get("low", 0), help="Unverified spans, retracted sources, or rubric violations")
            rcols[3].metric("Contradicted 🔴", rel_totals.get("contradicted", 0), help="Direct conflict with primary experimental/theoretical findings")
            rcols[4].metric("Unsupported 🟠", rel_totals.get("unsupported", 0), help="The cited paper was read and does not report this")
            rcols[5].metric("Unresolved ⏳", rel_totals.get("unresolved", 0), help="Judged but undecidable, or not assessed — see the caption above")

        # Filter tabs
        supp_count = totals.get("Supports", 0) + totals.get("Partially supports", 0)
        rev_count = needs_rev + totals.get("Unclear / insufficient evidence", 0)
        deferred_count = totals.get("deferred_paywalled", 0)
        all_count = totals.get("total", len(results))

        not_assessed_items = [r for r in results if r.get("outcome") not in ("judged", "not_downloaded", "deferred_paywalled")]
        tab_supported, tab_review, tab_not_assessed, tab_deferred, tab_all = st.tabs([
            f"Supported ({supp_count})",
            f"Need Review ({rev_count})",
            f"⏸ Not Assessed ({len(not_assessed_items)})",
            f"⏳ Pending Evidence (Deferred) ({deferred_count})",
            f"All Citations ({all_count})",
        ])

        def _render_claim_item(item):
            ref_info = item.get("ref") or {}
            ref_num = f"[{ref_info.get('index') or '?'}]"
            ref_title = ref_info.get("title") or item.get("cite_text", "Unknown reference")
            ref_yr = f"({ref_info.get('year')})" if ref_info.get("year") else ""
            ref_auth = (
                (", ".join(ref_info.get("authors", [])[:2]) + " et al.")
                if ref_info.get("authors")
                else ""
            )

            outcome = item.get("outcome")
            judgement = item.get("judgement", "Unclear")
            badge = _audit_item_badge(item)

            header = f"{badge}  ·  {ref_num} {ref_auth} {ref_yr} · *{ref_title[:55]}*"
            with st.expander(header):
                st.markdown("**Original Statement in Uploaded Paper:**")
                st.info(f"\"{item.get('sentence') or item.get('claim')}\"")

                st.markdown("**Cited Reference:**")
                st.write(f"{ref_num} **{ref_title}** {ref_yr}  \n*{ref_auth}*")
                if ref_info.get("doi"):
                    st.caption(f"DOI: [{ref_info['doi']}](https://doi.org/{ref_info['doi']})")

                if item.get("compound_missing_refs"):
                    c_missing = item["compound_missing_refs"]
                    c_names = [m.get("title") or f"[{m.get('index') or '?'}]" for m in c_missing if isinstance(m, dict)]
                    st.warning(
                        f"⚠️ **Compound Citation:** This sentence also cites {len(c_missing)} reference(s) missing from the library: "
                        + "; ".join(c_names[:2])
                        + (" et al." if len(c_names) > 2 else "")
                        + ". This paper was evaluated independently, but may only support part of the compound claim.",
                        icon="⚠️",
                    )

                if item.get("search_query") and item.get("outcome") == "judged":
                    st.caption(f"🔎 **Retrieval Query:** *{item['search_query']}*")
                if item.get("artifacts"):
                    st.caption("🖼️ **Refers to:** " + " · ".join(
                        f"{a['label']} — {a['caption'][:120]}" for a in item["artifacts"] if a.get("caption")
                    ))
                if item.get("cited_summary"):
                    st.caption(f"📄 **Cited paper, in brief:** {item['cited_summary']}")

                if outcome == "deferred_paywalled":
                    st.warning(
                        "⏳ **Evaluation Deferred (Pending Evidence)**: More than 50% of the cited references "
                        "in this body paragraph are missing from the local corpus (paywalled publishers, books, or 404s). "
                        "Evaluation of this statement is deferred until the missing papers are uploaded.",
                        icon="⏳",
                    )
                    missing_refs = item.get("paragraph_missing_refs") or []
                    if missing_refs:
                        missing_titles = [
                            m.get("title") or f"[{m.get('index') or '?'}] {m.get('doi') or 'Unknown'}"
                            for m in missing_refs
                            if isinstance(m, dict)
                        ]
                        if missing_titles:
                            st.caption(
                                f"Missing references in this paragraph ({len(missing_titles)}): "
                                + "; ".join(missing_titles)
                            )
                elif item.get("outcome") == "judged":
                    st.markdown(
                        f"**Verdict:** `{judgement}` · **Confidence:** `{item.get('confidence')}` · "
                        f"**Evidence Sufficiency:** `{item.get('evidence_sufficiency')}`"
                    )
                    if item.get("reliability_badge"):
                        st.markdown(
                            f"**🛡️ Scientific Reliability:** `{item['reliability_badge']}` — *{item.get('reliability_explanation', '')}*"
                        )
                    source_ass = item.get("source_assessment") or {}
                    if source_ass.get("badge"):
                        st.markdown(
                            f"**🏛️ Source Quality:** `{source_ass.get('badge')}` ({source_ass.get('grade_label', '')}) · *{source_ass.get('rationale', '')}*"
                        )

                    # Slot decomposition table
                    slots = item.get("slots") or {}
                    if slots and isinstance(slots, dict):
                        st.markdown("**Claim Decomposition & Slot Analysis:**")
                        slot_rows = []
                        for slot_name in ("finding", "scope", "strength"):
                            sdata = slots.get(slot_name) or {}
                            if isinstance(sdata, dict):
                                asrt = sdata.get("assertion", "—")
                                v = sdata.get("verdict", "—")
                            else:
                                asrt = "—"
                                v = str(sdata or "—")

                            if v == "Supports":
                                vb = "🟢 Supports"
                            elif v == "Partially supports":
                                vb = "🟡 Partially supports"
                            elif v == "Contradicts":
                                vb = "🔴 Contradicts"
                            elif v == "Does not support":
                                vb = "🟠 Does not support"
                            elif v == "Insufficient":
                                vb = "⚪ Insufficient"
                            else:
                                vb = f"`{v}`"

                            slot_rows.append(f"| `{slot_name}` | {asrt} | {vb} |")

                        st.markdown(
                            "| Slot | Assertion Extracted from Claim | Slot Evaluation |\n"
                            "| :--- | :--- | :--- |\n"
                            + "\n".join(slot_rows)
                        )

                    if item.get("supporting_span"):
                        st.markdown("**Verbatim Evidence from Cited Paper:**")
                        st.success(f"\"{item['supporting_span']}\"")
                    elif item.get("evidence"):
                        with st.expander("Show retrieved evidence excerpt from paper"):
                            st.markdown(f"> {item['evidence'][:500]}...")

                    st.markdown("**Detailed Reasoning & Assessment:**")
                    if item.get("reason"):
                        st.markdown(f"> {item['reason']}")

                    rule_expl = explain_rubric_verdict(item)
                    st.info(f"💡 **Why this verdict?** {rule_expl}")

                elif outcome == "not_downloaded":
                    st.warning(
                        "⚠️ **Reference Not Downloaded**: This reference paper was not open-access or could not be downloaded "
                        "(paywalled publisher, book, or 404). Its full text is not in the local library, so claims citing it "
                        "cannot be empirically verified.",
                        icon="🔒",
                    )
                else:
                    st.caption(f"Status: {item.get('reliability_explanation') or item.get('reason') or explain_rubric_verdict(item)}")

        with tab_supported:
            supp_items = [
                r for r in results if r.get("outcome") == "judged" and r.get("judgement") in ("Supports", "Partially supports")
            ]
            if supp_items:
                for it in supp_items:
                    _render_claim_item(it)
            else:
                st.caption("No supported citations to display.")

        with tab_review:
            rev_items = [
                r
                for r in results
                if r.get("outcome") == "judged"
                and r.get("judgement")
                in ("Contradicts", "Does not support", "Unclear / insufficient evidence")
            ]
            if rev_items:
                for it in rev_items:
                    _render_claim_item(it)
            else:
                st.caption("No citations flagged for review.")

        with tab_not_assessed:
            if not_assessed_items:
                st.info("These citations were found in the paper but the judge produced no verdict for them. "
                        "The badge on each says why.")
                for it in sorted(not_assessed_items, key=lambda r: r.get("outcome") or ""):
                    _render_claim_item(it)
            else:
                st.success("Every citation with an available reference was assessed.")

        with tab_deferred:
            if deferred_count == 0:
                st.info(
                    "No citations currently deferred. All paragraphs had sufficient reference coverage (≤50% paywalled).",
                    icon="✅",
                )
            else:
                st.warning(
                    "⏳ **Evaluation Deferred (Pending Evidence)**: One or more paragraphs in this uploaded paper cite "
                    "references where more than 50% are missing from the corpus (paywalled publishers, books, or 404s). "
                    "Evaluation of these statements is deferred until the missing PDFs are uploaded. "
                    "Upload the missing papers below and re-run the audit.",
                    icon="⏳",
                )

                deferred_missing = get_deferred_missing_references(audit)
                if not deferred_missing:
                    st.caption("No missing reference details available.")
                else:
                    st.markdown(f"**Missing References Blocking Evaluation ({len(deferred_missing)}):**")
                    for idx, ref in enumerate(deferred_missing):
                        xid = ref.get("xml_id") or str(ref.get("index") or idx)
                        xid = re.sub(r"[^\w\-]", "_", str(xid))
                        ref_idx = ref.get("index")
                        ref_idx_str = f"[{ref_idx}] " if ref_idx is not None else ""
                        ref_title = ref.get("title") or "Unknown Title"
                        ref_authors = ref.get("authors") or []
                        ref_year = ref.get("year")
                        ref_doi = ref.get("doi")

                        with st.container(border=True):
                            st.markdown(f"##### {ref_idx_str}{ref_title}")

                            meta_line = []
                            if ref_authors:
                                auth_str = ", ".join(ref_authors[:3])
                                if len(ref_authors) > 3:
                                    auth_str += " et al."
                                meta_line.append(f"**Authors:** {auth_str}")
                            if ref_year:
                                meta_line.append(f"**Year:** {ref_year}")
                            if meta_line:
                                st.markdown(" · ".join(meta_line))

                            if ref_doi:
                                st.markdown(f"🔗 **DOI:** [{ref_doi}](https://doi.org/{ref_doi})")
                            elif ref.get("raw_reference"):
                                st.caption(f"Citation: *{ref['raw_reference'][:140]}*")

                            affected = ref.get("affected_claims") or []
                            with st.expander(f"Dependent Statement(s) in Seed Paper ({len(affected)})"):
                                if affected:
                                    for aff_sent in affected:
                                        st.info(f"\"{aff_sent}\"")
                                else:
                                    st.caption("No dependent statements recorded.")

                            uploader_key = f"upload_ref_{stem}_{xid}"
                            uploaded_file = st.file_uploader(
                                f"Upload PDF for {ref_title[:45]}…",
                                type=["pdf"],
                                key=uploader_key,
                                help="Upload the full-text PDF to unblock evaluation of citations to this work.",
                            )

                            if uploaded_file is not None:
                                proc_key = f"processed_upload_{stem}_{xid}_{uploaded_file.size}"
                                legacy_key = f"processed_upload_{stem}_{xid}"
                                if (
                                    proc_key not in st.session_state
                                    and st.session_state.get(legacy_key) != uploaded_file.size
                                ):
                                    seed_name = os.path.basename(seed_path) if seed_path else ""
                                    try:
                                        reg_entry = save_and_register_reference_pdf(
                                            uploaded_file.read(),
                                            ref,
                                            seed_pdf_name=seed_name,
                                            original_filename=uploaded_file.name,
                                        )
                                        st.session_state[proc_key] = reg_entry
                                        st.session_state[legacy_key] = uploaded_file.size
                                        st.toast(f"✅ Uploaded & indexed {ref_title}", icon="📄")
                                    except Exception as exc:
                                        st.error(f"Failed to process uploaded PDF: {exc}")

                                if (
                                    proc_key in st.session_state
                                    or st.session_state.get(legacy_key) == uploaded_file.size
                                ):
                                    st.success(
                                        f"✅ PDF `{uploaded_file.name}` uploaded and indexed in knowledge base. Ready for re-audit."
                                    )

                st.markdown("---")
                if seed_path and os.path.exists(seed_path):
                    if st.button(
                        "⚡ Re-run Audit with Uploaded Papers",
                        key="btn_rerun_audit_uploaded",
                        type="primary",
                    ):
                        with st.status(
                            "Re-auditing seed citations with newly uploaded reference papers…",
                            expanded=True,
                        ) as status:
                            from research_assistant.shared.seed_audit import audit_seed_citations

                            cached_res = get_cached_search_resources(_get_resources_mtime())
                            audit_res = audit_seed_citations(seed_path, search_resources=cached_res)
                            final["citation_audit"] = audit_res
                            st.session_state["build_result"] = final
                            status.update(label="Citation audit complete!", state="complete")
                            st.rerun()

        with tab_all:
            for it in results:
                _render_claim_item(it)

        if seed_path and os.path.exists(seed_path):
            if st.button("🔄 Re-run Seed Citation Audit", key="btn_rerun_seed_audit"):
                with st.status("Re-auditing in-text citations from seed paper…", expanded=True) as status:
                    from research_assistant.shared.seed_audit import audit_seed_citations

                    cached_res = get_cached_search_resources(_get_resources_mtime())
                    audit_res = audit_seed_citations(seed_path, search_resources=cached_res)
                    final["citation_audit"] = audit_res
                    st.session_state["build_result"] = final
                    status.update(label="Citation audit complete!", state="complete")
                    st.rerun()


def _render_build(final, query):
    if final.get("batch_uploaded"):
        staged = final["batch_uploaded"]
        with st.container(border=True):
            st.markdown(f"**📚 Uploaded Paper Collection** — {len(staged)} paper(s) indexed")
            for p in staged:
                title = p.get("title") or p.get("filename")
                bits = [f"📄 `{p.get('filename')}`"]
                if p.get("doi"):
                    bits.append(f"doi:{p['doi']}")
                if p.get("arxiv_id"):
                    bits.append(f"arXiv:{p['arxiv_id']}")
                st.markdown(
                    f"- **{title}**  \n  <small style='color:gray;'>{' · '.join(bits)}</small>",
                    unsafe_allow_html=True,
                )

        answer = final.get("answer")
        if answer:
            _render_shortlist(answer.get("selected"))
            _render_notes(answer)
            _render_synthesis(answer, "Related work across collection")
            _render_passages(answer.get("passages") or [])
        else:
            st.info(
                "Corpus updated with uploaded papers. Switch to **Research chat** or **Cite a draft** to query them.",
                icon="✍️",
            )
        _render_seed_citation_audit(final)
        return

    # Show whatever the run produced — seed, downloads — even if it stopped early.
    _render_seed_and_downloads(query, final)

    if final.get("stopped"):
        st.warning(final["stopped"], icon="⚠️")
        if (
            "supply an arXiv" in final["stopped"]
            or "could not download" in final["stopped"]
            or "could not load seed PDF" in final["stopped"]
        ):
            st.info(
                "Upload a seed PDF or paste an arXiv/PDF link into **Seed paper URL** "
                "above and build again to ground the answer in a real corpus.",
                icon="💡",
            )
        if final.get("answer"):
            with st.container(border=True):
                st.markdown("###### Answer (not grounded in retrieved papers)")
                st.markdown(final["answer"]["suggestion"])
        return

    answer = final.get("answer")
    if answer:
        _render_shortlist(answer.get("selected"))
        _render_notes(answer)
        _render_synthesis(answer, "Related work")
        _render_passages(answer.get("passages") or [])
    else:

        st.info(
            "Corpus updated. Switch to **Cite a draft** to query it.", icon="✍️"
        )

    _render_seed_citation_audit(final)


# ─── Sidebar ────────────────────────────────────────────────────────────────

@st.fragment(run_every="3s")
def _render_sidebar_pipeline_status():
    try:
        from datetime import datetime, timezone

        status = pipeline_status.get_status()

        if status.get("active", False):
            st.info("⚡ **Pipeline Active**")
            if st.button("⏹️ Stop Pipeline", key="sidebar_stop_pipeline_btn", type="secondary", use_container_width=True, help="Immediately halt the active pipeline safely"):
                pipeline_status.request_cancel("User stopped pipeline via sidebar")
                st.rerun()
            stage_lbl = status.get("stage_label") or status.get("stage") or "Processing"
            try:
                curr_step = int(status.get("current_step") or 1)
                tot_steps = int(status.get("total_steps") or 5)
            except (ValueError, TypeError):
                curr_step, tot_steps = 1, 5
            st.markdown(f"**Stage:** {stage_lbl} (Step {curr_step}/{tot_steps})")

            curr_item = status.get("current_item_name", "")
            if curr_item:
                st.markdown(f"**Current:** `{curr_item[:60]}`")

            try:
                item_c = int(status.get("item_current") or 0)
                item_t = int(status.get("item_total") or 0)
            except (ValueError, TypeError):
                item_c, item_t = 0, 0
            if item_t > 0:
                pct = min(1.0, max(0.0, item_c / item_t))
                st.progress(pct, text=f"{item_c} / {item_t} papers ({int(pct * 100)}%)")
            else:
                pct = min(1.0, max(0.0, curr_step / max(1, tot_steps)))
                st.progress(pct, text=f"Step {curr_step} of {tot_steps}")

            started_at = status.get("started_at")
            if started_at:
                try:
                    start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                    elapsed_sec = int((datetime.now(timezone.utc) - start_dt).total_seconds())
                    if elapsed_sec >= 60:
                        mins = elapsed_sec // 60
                        secs = elapsed_sec % 60
                        st.caption(f"⏱️ Elapsed: {mins}m {secs}s")
                    else:
                        st.caption(f"⏱️ Elapsed: {max(0, elapsed_sec)}s")
                except Exception:
                    pass

            detail = status.get("detail", "")
            if detail:
                st.caption(f"⚙️ {detail}")

            events = status.get("recent_events", [])
            with st.expander("📋 Live Activity Log", expanded=False):
                if events:
                    for ev in reversed(events):
                        st.markdown(f"- {ev}")
                else:
                    st.caption("No events recorded yet.")
        else:
            st.markdown("🟢 **Pipeline: Idle** (Ready for new papers)")
            last_completed = status.get("last_completed_at")
            if last_completed:
                try:
                    comp_dt = datetime.fromisoformat(last_completed.replace("Z", "+00:00"))
                    time_str = comp_dt.strftime("%H:%M UTC")
                    summary = (status.get("last_summary") or "").strip()
                    if summary.startswith("(") and summary.endswith(")"):
                        summary = summary[1:-1].strip()
                    summary_part = f" ({summary})" if summary else ""
                    st.caption(f"Last run completed: {time_str}{summary_part}")
                except Exception:
                    pass
            events = status.get("recent_events", [])
            if events:
                with st.expander("📋 Recent Activity Log", expanded=False):
                    for ev in reversed(events):
                        st.caption(ev)
    except Exception as exc:
        st.caption("Pipeline status temporarily unavailable")


with st.sidebar:
    st.subheader("Pipeline Status")
    _render_sidebar_pipeline_status()
    st.divider()

    st.subheader("Corpus")
    chunks, papers = _corpus_stats()
    c1, c2 = st.columns(2)
    c1.metric("Papers", papers)
    c2.metric("Chunks", chunks)

    st.subheader("Backend")
    st.caption(f"**Chat** — `{config.LLM_MODEL}` · {config.LLM_BACKEND}")
    st.caption(f"**Embeddings** — `{config.EMBED_MODEL}` · {config.EMBED_BACKEND}")
    st.caption(
        f"**Layout** — {'on' if config.LAYOUT_DETECTION else 'text-only'}"
    )

    st.subheader("Server & Services")
    _render_grobid_controls()
    if config.LLM_BACKEND == "openai" and not config.OPENAI_API_KEY:
        st.error("No OPENAI_API_KEY / HF_TOKEN set.", icon="🚫")


# ─── Main ───────────────────────────────────────────────────────────────────

st.title("📚 Citation Needed!")
st.caption(
    "Marvin the Citebot — Autonomous AI Research Assistant. Give it a research idea → "
    "it builds a corpus from the literature and tells you what's already been done. "
    "Or hand it a sentence and it finds the citation."
)

tab_audit, tab_idea, tab_draft, tab_chat, tab_help = st.tabs(
    ["Citation auditor", "Research idea", "Cite a draft", "Research chat",
     "How to use"]
)


@st.fragment(run_every="3s")
def _render_live_pipeline_status(key_suffix="tab1"):
    try:
        active_status = pipeline_status.get_status()
        if active_status.get("active", False) or _is_ingest_locked():
            with st.container(border=True):
                col_title, col_stop = st.columns([3, 1])
                with col_title:
                    st.markdown("#### ⚡ Pipeline Active on Server")
                with col_stop:
                    if st.button("⏹️ Stop Pipeline", key=f"stop_pipeline_btn_{key_suffix}", type="secondary", use_container_width=True, help="Immediately halt the active pipeline safely"):
                        pipeline_status.request_cancel(f"User stopped pipeline via {key_suffix}")
                        st.rerun()
                st_stage = active_status.get("stage_label") or "Indexing papers"
                try:
                    st_step = int(active_status.get("current_step") or 1)
                    st_total = int(active_status.get("total_steps") or 5)
                except (ValueError, TypeError):
                    st_step, st_total = 1, 5
                st.markdown(f"**Stage:** {st_stage} (Step {st_step}/{st_total})")
                if active_status.get("current_item_name"):
                    st.markdown(f"**Working on:** `{active_status['current_item_name'][:70]}`")
                if active_status.get("detail"):
                    st.caption(f"⚙️ {active_status['detail']}")
                try:
                    item_c = int(active_status.get("item_current") or 0)
                    item_t = int(active_status.get("item_total") or 0)
                except (ValueError, TypeError):
                    item_c, item_t = 0, 0
                if item_t > 0:
                    pct = min(1.0, max(0.0, item_c / item_t))
                    st.progress(pct, text=f"{item_c} / {item_t} papers ({int(pct * 100)}%)")
                else:
                    pct = min(1.0, max(0.0, st_step / max(1, st_total)))
                    st.progress(pct, text=f"Step {st_step} of {st_total}")
                st.info(
                    "⏳ Another paper indexing process is actively running on the server. "
                    "Your upload or pipeline request will queue safely once the active job finishes.",
                    icon="ℹ️",
                )
                if active_status.get("recent_events"):
                    with st.expander("📋 Live Activity Log", expanded=False):
                        for ev in reversed(active_status["recent_events"][-5:]):
                            st.write(ev)
    except Exception:
        pass


_render_tab1_live_status = _render_live_pipeline_status


def _execute_pipeline(inputs: dict, cfg: dict, display_q: str, on_snapshot=None) -> dict:
    """Run the graph to completion and return its final state.

    Runs in a job thread (see shared/run_jobs.py): it must never touch
    Streamlit. Progress goes through pipeline_status — the file every
    session already polls — and *on_snapshot* receives the partial state
    after each node so a session can show the seed and its downloads while
    the run continues. Cancellation is the pipeline_status flag the sidebar
    button sets. Failure raises; the job records it.
    """
    graph = _graph()
    final: dict = {}
    pipeline_status.set_status(
        active=True,
        stage="discover",
        stage_label="Finding seed paper",
        current_step=1,
        total_steps=5,
        detail=f"Starting pipeline for: {display_q[:50]}",
    )
    pipeline_status.add_event(f"🚀 Pipeline started for: {display_q[:40]}")
    try:
        for update in graph.stream(inputs, cfg, stream_mode="updates"):
            if pipeline_status.is_cancel_requested():
                break
            for node, payload in update.items():
                final.update(payload or {})
                if on_snapshot is not None:
                    on_snapshot({**final, "_last_node": node})
                if payload and payload.get("stopped"):
                    break
            if pipeline_status.is_cancel_requested():
                break
        if pipeline_status.is_cancel_requested():
            pipeline_status.add_event("⏹️ Pipeline stopped by user")
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle",
                                       detail="Pipeline stopped by user")
            raise pipeline_status.PipelineCancelledError("Pipeline stopped by user")
        final = graph.get_state(cfg).values
        if final.get("stopped"):
            pipeline_status.add_event(f"⚠️ Pipeline stopped early: {final['stopped'][:60]}")
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle",
                                       detail=f"Stopped: {final['stopped'][:60]}")
        else:
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            pipeline_status.add_event(f"✅ Pipeline completed: {display_q[:40]}")
            pipeline_status.set_status(
                active=False, stage="idle", stage_label="Idle", detail="Pipeline complete",
                last_completed_at=now_iso, last_summary=f"Completed {display_q[:40]}",
            )
        return final
    except pipeline_status.PipelineCancelledError:
        raise
    except BaseException as e:  # noqa: BLE001
        if pipeline_status.is_cancellation(e):
            pipeline_status.add_event("⚠️ Pipeline cancelled")
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle",
                                       detail="Pipeline cancelled")
        else:
            detail_str = pipeline_status.format_exception_detail(e)
            pipeline_status.add_event(f"❌ Pipeline failed: {detail_str}")
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle",
                                       detail=f"Pipeline failed: {detail_str}")
        raise
    finally:
        seed_file_path = inputs.get("seed_file")
        if seed_file_path:
            try:
                raw_dir_abs = os.path.abspath(config.RAW_DIR)
                seed_abs = os.path.abspath(seed_file_path)
                if not (seed_abs == raw_dir_abs or seed_abs.startswith(raw_dir_abs + os.sep)):
                    if os.path.exists(seed_file_path):
                        os.unlink(seed_file_path)
            except OSError:
                pass


def _start_pipeline_job(
    query: str,
    seed_url_val: str | None = None,
    seed_file_path: str | None = None,
    ask: bool = True,
    force: bool = False,
    describe_figures: bool = False,
    audit_citations: bool = True,
    origin: str = "audit",
) -> str | None:
    """Start the pipeline as a server-side job and remember it in the URL.

    Returns the job id, or None when another run is active (a warning is
    shown). The id is deterministic in the input, so re-submitting the same
    paper while its run is active attaches to that run instead of starting
    a second one.
    """
    from research_assistant.shared import run_jobs

    thread_seed = query or (os.path.basename(seed_file_path) if seed_file_path else "run")
    job_id = run_jobs.job_id_for(thread_seed)
    cfg = {"configurable": {"thread_id": job_id}}
    inputs = {
        "query": query,
        "workers": 1,
        "force": force,
        "ask": ask,
        "seed_url": seed_url_val,
        "seed_file": seed_file_path,
        "describe_figures": describe_figures,
        "audit_citations": audit_citations,
    }
    display_q = query or (os.path.basename(seed_file_path) if seed_file_path else "") or (seed_url_val or "") or "topic"

    def runner(on_snapshot):
        return _execute_pipeline(inputs, cfg, display_q, on_snapshot=on_snapshot)

    try:
        run_jobs.start_job(job_id, display_q, runner, origin=origin)
    except run_jobs.JobBusy as busy:
        st.warning(
            f"A run is already active on the server ({busy.job.label}). "
            "Wait for it to finish, or stop it from the sidebar, then submit again.",
            icon="⏳",
        )
        return None
    st.query_params["job"] = job_id
    st.session_state["active_job"] = job_id
    st.session_state["active_job_origin"] = origin
    return job_id


def _start_audit_job(seed_file_path: str, label: str, origin: str = "audit") -> str | None:
    """Re-run only the citation audit for a paper whose seed and references
    are already in the corpus — the case where a cached report exists but
    the judge never ran on it. Same job machinery as the full pipeline, so
    it too survives a refresh."""
    from research_assistant.shared import run_jobs
    from research_assistant.shared.seed_audit import audit_seed_citations

    job_id = run_jobs.job_id_for(os.path.basename(seed_file_path))

    def runner():
        pipeline_status.set_status(
            active=True, stage="respond", stage_label="Auditing citations",
            current_step=5, total_steps=5, detail=f"Re-running citation audit for: {label[:50]}",
        )
        pipeline_status.add_event(f"🔍 Re-running citation audit: {label[:40]}")
        try:
            audit = audit_seed_citations(seed_file_path, force=True, skip_if_cached=False)
            pipeline_status.add_event("✅ Seed citation audit complete")
            return {"seed_path": seed_file_path, "seed_label": label, "citation_audit": audit, "audit_only": True}
        finally:
            pipeline_status.set_status(active=False, stage="idle", stage_label="Idle", detail="Audit complete")

    try:
        run_jobs.start_job(job_id, label, runner, origin=origin)
    except run_jobs.JobBusy as busy:
        st.warning(
            f"A run is already active on the server ({busy.job.label}). "
            "Wait for it to finish, or stop it from the sidebar, then submit again.",
            icon="⏳",
        )
        return None
    st.query_params["job"] = job_id
    st.session_state["active_job"] = job_id
    st.session_state["active_job_origin"] = origin
    return job_id


def _resolve_job(job_id: str):
    """(state, final, error, origin) for a job id, from the registry first
    and the run record on disk second — the disk is what survives a server
    restart. state is running | done | failed | cancelled | None."""
    from research_assistant.shared import run_jobs

    job = run_jobs.get_job(job_id)
    if job is not None:
        if job.state == "running":
            return "running", None, None, job.origin
        if job.state == "done":
            return "done", job.result, None, job.origin
        return job.state, None, job.error, job.origin
    record = run_jobs.load_run(job_id)
    if record:
        return "done", record.get("final") or {}, None, record.get("origin", "audit")
    return None, None, None, None


def _adopt_job_result(final: dict, origin: str) -> None:
    """Put a finished run where the tabs already look for it."""
    label = final.get("query") or final.get("seed_label") or os.path.basename(final.get("seed_path") or "") or ""
    if origin == "idea":
        st.session_state["idea_result"] = final
        st.session_state["idea_query"] = label
    else:
        st.session_state["audit_result"] = final
        st.session_state["audit_query"] = label
        st.session_state["build_result"] = final
        st.session_state["build_query"] = label
    st.session_state.pop("active_job", None)
    st.session_state.pop("active_job_origin", None)


@st.fragment(run_every="2s")
def _render_job_progress(job_id: str):
    """The live view of a running job. Rerenders every 2 s from
    pipeline_status and the job's snapshot; when the job ends, reruns the
    whole app so the tab renders the result."""
    from research_assistant.shared import run_jobs

    job = run_jobs.get_job(job_id)
    if job is None or job.state != "running":
        st.rerun(scope="app")
        return
    with st.container(border=True):
        st.markdown(f"#### ⏳ Running: `{job.label}`")
        st.caption(
            "This run continues on the server. You can refresh this page, open it in another tab, "
            "or come back later — the URL keeps the job id, and the report will be here when it finishes."
        )
        status = pipeline_status.get_status()
        stage = status.get("stage_label") or "Starting"
        try:
            step, total = int(status.get("current_step") or 1), int(status.get("total_steps") or 5)
        except (ValueError, TypeError):
            step, total = 1, 5
        try:
            item_c, item_t = int(status.get("item_current") or 0), int(status.get("item_total") or 0)
        except (ValueError, TypeError):
            item_c, item_t = 0, 0
        if item_t > 0:
            st.progress(min(1.0, max(0.0, item_c / item_t)), text=f"Step {step}/{total}: {stage} — {item_c}/{item_t}")
        else:
            st.progress(min(1.0, max(0.0, step / max(1, total))), text=f"Step {step}/{total}: {stage}")
        if status.get("detail"):
            st.caption(f"⚙️ {status['detail']}")
        if status.get("recent_events"):
            with st.expander("📋 Live Activity Log", expanded=False):
                for ev in reversed(status["recent_events"][-8:]):
                    st.write(ev)
        if job.snapshot:
            _render_seed_and_downloads(job.snapshot.get("query") or job.label, job.snapshot)


def _handle_job_state(origin: str) -> bool:
    """Show a running job, adopt a finished one, explain a failed one.

    The job id comes from this session (it started the run) or from the
    URL (the page was refreshed, or opened elsewhere). Returns True when a
    running job was rendered and the tab should show nothing else.
    """
    from research_assistant.shared import run_jobs

    job_id = st.session_state.get("active_job") or st.query_params.get("job")
    if not job_id:
        return False
    state, final, error, job_origin = _resolve_job(job_id)
    if job_origin not in (None, origin):
        return False  # belongs to the other tab
    if state == "running":
        _render_job_progress(job_id)
        return True
    if state == "done":
        already = st.session_state.get("idea_result" if origin == "idea" else "audit_result")
        if not already or st.session_state.get("active_job") == job_id:
            _adopt_job_result(final or {}, job_origin or origin)
        return False
    if state in ("failed", "cancelled"):
        st.session_state.pop("active_job", None)
        if state == "failed":
            st.error(f"The run failed: {error}", icon="❌")
        else:
            st.warning("The run was stopped before it finished.", icon="⏹️")
        return False
    st.info(
        f"No record of run `{job_id}` on this server — it may have been started before the last restart "
        "and not finished. Pick a recent run below or start a new one.",
        icon="ℹ️",
    )
    st.session_state.pop("active_job", None)
    return False


def _render_recent_runs(origin: str) -> None:
    """Open a finished run from disk without re-uploading anything."""
    from research_assistant.shared import run_jobs

    runs = [r for r in run_jobs.list_runs() if r.get("origin", "audit") == origin]
    if not runs:
        return
    options = {f"{r['label'] or r['seed_name'] or r['job_id']} — {r['saved_at'][:16].replace('T', ' ')}": r["job_id"] for r in runs}
    with st.container(border=True):
        st.markdown("###### 🗂️ Recent runs on this server")
        choice = st.selectbox("Open a finished run", list(options), key=f"recent_runs_{origin}", label_visibility="collapsed")
        if st.button("Open", key=f"open_recent_run_{origin}"):
            st.query_params["job"] = options[choice]
            st.session_state["active_job"] = options[choice]
            st.session_state["active_job_origin"] = origin
            st.rerun()


# ─── Tab 1: citation auditor ───────────────────────────────────────────────

with tab_audit:
    _render_live_pipeline_status("tab_audit")

    st.subheader("📄 Seed Paper Citation Auditor")
    st.caption(
        "Upload a research manuscript (PDF or ZIP). The pipeline parses in-text citations with GROBID, "
        "retrieves open-access references from Unpaywall, Europe PMC, and arXiv, and audits each citation "
        "against the source text using Rubric V1.5."
    )

    with st.form("upload_papers_form"):
        uploaded_files = st.file_uploader(
            "Select research paper(s) (.pdf or .zip)",
            type=["pdf", "zip"],
            accept_multiple_files=True,
            help="Upload one or multiple PDF papers, or a .zip archive. If 1 paper is uploaded, its in-text citations are audited; if multiple papers are uploaded, all are directly indexed into the corpus.",
        )
        pdf_query = st.text_input(
            "Research topic / question (optional)",
            placeholder="e.g. computational modeling of lipid nanocarriers (leave blank to infer from papers)",
            help="If provided, used to synthesize an answer across the papers at the end.",
        )
        c1, c2, c3, c4 = st.columns(4)
        audit_citations = c1.toggle(
            "Audit citations",
            value=True,
            help="Audit in-text citations in uploaded paper against fetched references using Gemma 4 / Gemini.",
        )
        ask = c2.toggle("Synthesize answer", value=True, help="Formulate related-work synthesis.")
        force = c3.toggle("Force re-run", value=False)
        describe_figures = c4.toggle(
            "Analyse figures",
            value=config.FIGURE_VLM,
            help="Describe figures and tables with VLM during ingestion. Adds ~1 min per figure on CPU.",
        )
        submitted_upload = st.form_submit_button("Audit Citations & Index Paper(s)", type="primary")

    if submitted_upload:
        if not uploaded_files:
            st.warning("Please upload one or more PDF files (or a .zip) to begin.", icon="⚠️")
        else:
            from research_assistant.shared.batch_uploader import unpack_and_stage_uploads

            staged = unpack_and_stage_uploads(uploaded_files, destination_dir=config.RAW_DIR)
            if not staged:
                st.error(
                    "No valid PDF documents found in the uploaded files. Check that files contain valid PDF headers (%PDF-).",
                    icon="⚠️",
                )
            elif len(staged) == 1:
                seed_file_path = staged[0]["path"]
                effective_q = pdf_query.strip()
                from research_assistant.shared.seed_audit import (
                    cross_check_seed_audit,
                    get_cached_seed_audit,
                )

                cached = get_cached_seed_audit(seed_file_path) if not force else None
                # A record exists but the judge never ran on it (backend was
                # down): the corpus already holds the seed and its references,
                # so only the audit is re-run — as a job, not in this session.
                stale = (
                    None if (cached or force)
                    else get_cached_seed_audit(seed_file_path, include_failed=True)
                )
                if stale:
                    for key in ("audit_result", "audit_query", "build_result", "build_query"):
                        st.session_state.pop(key, None)
                    st.info(
                        "The previous audit of this paper never reached the judge (the model backend was down). "
                        "Its references are already indexed — re-running just the audit.",
                        icon="🔁",
                    )
                    _start_audit_job(
                        seed_file_path,
                        label=effective_q or staged[0].get("title") or os.path.basename(seed_file_path),
                        origin="audit",
                    )
                elif cached:
                    with st.status(
                        "⚡ Found existing audit — cross-checking references and reliability…",
                        expanded=True,
                    ) as status:
                        cached_res = get_cached_search_resources(_get_resources_mtime())
                        audit_res, summary = cross_check_seed_audit(
                            seed_file_path, cached, search_resources=cached_res
                        )
                        status.update(
                            label=f"Done — cross-checked in {summary['duration_seconds']:.1f}s!",
                            state="complete",
                        )

                    effective_final_q = (
                        effective_q
                        or staged[0].get("title")
                        or os.path.basename(seed_file_path)
                    )
                    final = {
                        "seed_path": seed_file_path,
                        "seed_label": effective_final_q,
                        "citation_audit": audit_res,
                        "from_cache": True,
                        "cross_check_summary": summary,
                    }
                    st.session_state["audit_result"] = final
                    st.session_state["audit_query"] = effective_final_q
                    st.session_state["build_result"] = final
                    st.session_state["build_query"] = effective_final_q
                else:
                    # The ten-minute path runs as a server-side job: this
                    # session (or any other, after a refresh) follows it by
                    # the job id in the URL and adopts the result when done.
                    for key in ("audit_result", "audit_query", "build_result", "build_query"):
                        st.session_state.pop(key, None)
                    _start_pipeline_job(
                        query=effective_q,
                        seed_url_val=None,
                        seed_file_path=seed_file_path,
                        ask=ask,
                        force=force,
                        describe_figures=describe_figures,
                        audit_citations=audit_citations,
                        origin="audit",
                    )
            else:
                with st.status(f"Batch ingesting {len(staged)} papers into corpus…", expanded=True) as status:
                    batch_progress = st.progress(0.0, text=f"Preparing to ingest {len(staged)} papers…")
                    st.write(f"📁 Unpacked {len(staged)} documents.")
                    candidates = {p["path"]: p.get("title") or os.path.basename(p["path"]) for p in staged}

                    from research_assistant.shared.ingestion import ingest_pdfs

                    def _on_batch_progress(item_c, item_t, name):
                        if item_t > 0:
                            pct = min(1.0, max(0.0, item_c / item_t))
                            batch_progress.progress(pct, text=f"Parsing paper [{item_c}/{item_t}]: {name[:40]}")

                    with pipeline_status.track_stage(
                        "ingest_refs",
                        "Batch Paper Ingestion",
                        current_step=1,
                        total_steps=1,
                        item_total=len(candidates),
                        detail=f"Batch ingesting {len(candidates)} papers",
                    ):
                        try:
                            ingest_res = ingest_pdfs(candidates, workers=1, skip_ingested=not force, describe_figures=describe_figures)
                        finally:
                            pipeline_status.unregister_progress_callback(_on_batch_progress)

                    if pipeline_status.is_cancel_requested():
                        status.update(label="Batch ingestion stopped by user", state="error")
                        st.warning("Batch ingestion stopped by user.", icon="⏹️")
                    else:
                        if ingest_res.get("described"):
                            st.write(f"🖼️ Described {ingest_res['described']} figure(s)/table(s).")
                        scanned_empty = ingest_res.get("scanned_or_empty", [])
                        inserted = ingest_res.get("inserted", 0)
                        processed = ingest_res.get("processed", len(candidates))
                        skipped = ingest_res.get("skipped", 0)
                        st.write(f"✓ Parsed {processed} paper(s), skipped {skipped} duplicates, inserted {inserted} new chunks.")
                        if scanned_empty:
                            st.warning(f"⚠️ {len(scanned_empty)} file(s) had no extractable text: {', '.join(scanned_empty[:5])}")

                        try:
                            dl_manifest = {}
                            if os.path.exists(config.DOWNLOADED_JSON_PATH):
                                with open(config.DOWNLOADED_JSON_PATH, "r", encoding="utf-8") as f:
                                    dl_manifest = json.load(f)
                            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            for p in staged:
                                k = p.get("key") or os.path.basename(p["path"])
                                dl_manifest[k] = {
                                    "key": k,
                                    "title": p.get("title") or os.path.basename(p["path"]),
                                    "doi": p.get("doi"),
                                    "arxiv_id": p.get("arxiv_id"),
                                    "path": p["path"],
                                    "source": "upload",
                                    "fetched_at": now_iso,
                                }
                            atomic_write_json(config.DOWNLOADED_JSON_PATH, dl_manifest, ensure_ascii=False)
                        except Exception as e:
                            logger.warning("Could not update downloaded.json: %s", e)

                        answer = None
                        effective_q = pdf_query.strip() or (staged[0].get("title") if staged else "")
                        if ask and effective_q:
                            st.write("🧠 Formulating related-work synthesis across uploaded collection…")
                            from research_assistant.shared import retrieve

                            def _on_progress(stage, payload):
                                if stage == "shortlist":
                                    st.write("📚 Reading " + ", ".join(f"{p['key']} {p['citation'][:50]}" for p in payload["papers"]))
                                elif stage == "notes":
                                    mark = "📝" if payload.get("relevant", True) else "➖"
                                    st.write(f"{mark} {payload['key']} · {payload['citation'][:60]} ({payload.get('seconds', '?')}s)")
                                elif stage == "synthesis":
                                    st.write(f"🧠 Synthesis written ({payload.get('seconds', '?')}s)")

                            try:
                                answer = retrieve.research_answer(effective_q, on_progress=_on_progress)
                            except Exception as exc:
                                logger.error("Synthesis failed: %s", exc)
                                answer = {
                                    "suggestion": f"Synthesis encountered an error: {exc}",
                                    "citations": [],
                                    "passages": [],
                                }

                        status.update(label=f"Done — {len(staged)} paper(s) indexed!", state="complete")

                    final = {
                        "seed_path": staged[0]["path"] if staged else None,
                        "seed_label": f"Uploaded collection ({len(staged)} papers)",
                        "batch_uploaded": staged,
                        "answer": answer,
                    }
                    effective_final_q = effective_q or "Uploaded paper collection"
                    st.session_state["audit_result"] = final
                    st.session_state["audit_query"] = effective_final_q
                    st.session_state["build_result"] = final
                    st.session_state["build_query"] = effective_final_q
                    _corpus_stats.clear()

    # A running job renders its own live view; a finished one is adopted
    # into session state here, so the read below must come after the call.
    job_running = _handle_job_state("audit")
    audit_res = st.session_state.get("audit_result") or st.session_state.get("build_result")
    if job_running:
        pass
    elif audit_res and (
        audit_res.get("citation_audit")
        or audit_res.get("audit")
        or audit_res.get("batch_uploaded")
        or audit_res.get("seed_path")
    ):
        _render_build(audit_res, st.session_state.get("audit_query") or st.session_state.get("build_query", ""))
    else:
        _render_recent_runs("audit")
        st.markdown("---")
        st.markdown(
            "#### 🔍 How Seed Paper Citation Auditing Works\n\n"
            "1. **Upload your paper** (.pdf) or drop multiple papers (.zip).\n"
            "2. **Reference Mining**: GROBID extracts all in-text citation markers (`[14]`, `(Smith et al., 2020)`) and maps them to the bibliography.\n"
            "3. **Open-Access Retrieval**: Unpaywall, Europe PMC, and arXiv download full-text PDFs of the cited literature.\n"
            "4. **Scientific Claim Audit**: Every citation is independently audited against the source text to verify if the cited evidence actually backs up the claim."
        )


# ─── Tab 2: research idea ──────────────────────────────────────────────────

with tab_idea:
    _render_live_pipeline_status("tab_idea")

    st.subheader("🔍 Explore Research Ideas & Discover Literature")
    st.caption(
        "Give Marvin a topic, hypothesis, or research question. The assistant searches arXiv, Semantic Scholar, and OpenAlex, "
        "downloads relevant open-access papers, indexes them into your local corpus, and synthesizes what has already been done."
    )

    with st.form("idea_form"):
        query = st.text_input(
            "Research idea",
            placeholder="topological protection in disordered quantum wires",
            help="Enter a research topic, question, or hypothesis.",
        )
        seed_url = st.text_input(
            "Seed paper URL (optional)",
            placeholder="https://arxiv.org/abs/2401.12345 — leave blank to search automatically",
            help="Used when the search finds no open-access PDF. Accepts an arXiv link or a direct .pdf URL.",
        )
        c1, c2, c3 = st.columns(3)
        ask = c1.toggle("Synthesize answer", value=True, help="Formulate related-work synthesis across discovered papers.")
        force = c2.toggle("Force re-run", value=False)
        describe_figures = c3.toggle(
            "Analyse figures",
            value=config.FIGURE_VLM,
            help="Describe figures and tables with VLM during ingestion. Adds ~1 min per figure on CPU.",
        )
        submitted_idea = st.form_submit_button("Explore Research Idea", type="primary")

    if submitted_idea:
        if not query.strip():
            st.warning("Please enter a research idea to search.", icon="⚠️")
        else:
            q = query.strip()
            seed_url_val = seed_url.strip() or None
            for key in ("idea_result", "idea_query"):
                st.session_state.pop(key, None)
            _start_pipeline_job(
                query=q,
                seed_url_val=seed_url_val,
                seed_file_path=None,
                ask=ask,
                force=force,
                describe_figures=describe_figures,
                audit_citations=False,
                origin="idea",
            )

    if _handle_job_state("idea"):
        pass  # a run is in progress; its live view is on screen
    elif st.session_state.get("idea_result"):
        idea_res = st.session_state["idea_result"]
        _corpus_stats.clear()
        _render_build(idea_res, st.session_state.get("idea_query", ""))
    else:
        _render_recent_runs("idea")
        st.markdown("---")
        st.markdown(
            "#### 💡 How Literature Discovery Works\n\n"
            "1. **Enter an idea**: Describe a research topic or paste a known arXiv / PDF link.\n"
            "2. **Discovery (Agent 0)**: Searches academic repositories for candidate literature and downloads the seed paper.\n"
            "3. **Citation Chasing (Agents 1 & 2)**: Extracts references and fetches connected open-access papers.\n"
            "4. **Corpus Indexing (Agent 3)**: Chunks, embeds, and builds the BM25 keyword index.\n"
            "5. **Synthesis**: Writes a structured related-work summary highlighting established findings and open gaps."
        )


# ─── Tab 3: cite a draft ───────────────────────────────────────────────────

with tab_draft:
    st.subheader("✍️ Cite a Research Draft")
    st.caption(
        "Upload a plain-text draft (.txt) or paste it below. Every sentence that makes a factual claim "
        "is checked against your indexed corpus, cited where a source supports it, and verified for accuracy."
    )
    if chunks == 0:
        st.info("No corpus yet — build one in **Citation auditor** or **Research idea** first.", icon="📭")

    uploaded = st.file_uploader("Draft (.txt)", type=["txt"], key="batch_upload")
    pasted = st.text_area("…or paste it here", height=200, key="batch_paste")
    run_batch = st.button("Cite the draft", type="primary", disabled=chunks == 0)

    if run_batch and (uploaded or pasted.strip()):
        import tempfile

        from research_assistant.agents import agent5_batch_citer

        os.makedirs(config.DRAFTS_DIR, exist_ok=True)
        text = uploaded.read().decode("utf-8") if uploaded else pasted
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", dir=config.DRAFTS_DIR, delete=False, encoding="utf-8"
        ) as fh:
            fh.write(text)
            draft_path = fh.name
        out_path = draft_path.replace(".txt", "_cited.txt")

        with st.spinner("Checking each sentence and retrieving sources…"):
            try:
                cached_res = get_cached_search_resources(_get_resources_mtime())
                written = agent5_batch_citer.run_batch_citer(
                    draft_path, out_path, search_resources=cached_res
                )
            except Exception as e:  # noqa: BLE001
                st.exception(e)
                written = None
        st.session_state["batch_result"] = written

    written = st.session_state.get("batch_result")
    if written is None and st.session_state.get("batch_result", "unset") != "unset":
        # run_batch_citer returns None only when it aborted before writing.
        st.error(
            "The citation-need check could not be aligned to the draft's "
            "sentences, so nothing was written. Try again, or shorten the draft "
            "if the model keeps truncating its reply.",
            icon="⚠️",
        )
    elif written:
        with open(written, encoding="utf-8") as fh:
            cited = fh.read()
        st.markdown("###### Cited draft")
        st.text_area("Result", cited, height=260, key="batch_out")
        st.download_button("Download cited draft", cited,
                           file_name=os.path.basename(written))

        mapping_path = written.replace(".txt", "_citations.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, encoding="utf-8") as fh:
                mapping = json.load(fh)
            st.markdown(f"**Sources cited** — {len(mapping)}")
            st.json(mapping, expanded=False)

        report_path = written.replace(".txt", "_report.md")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as fh:
                report = fh.read()
            with st.expander("Per-sentence decisions"):
                st.markdown(report)

        st.divider()
        st.caption(
            "Verification re-checks every inserted citation against the source "
            "it cites — one model call per citation, so a long draft is not free."
        )
        if st.button("Verify citations", key="verify_citations"):
            with st.status("Judging each citation…", expanded=True) as status:
                try:
                    verification = verify_draft(written)
                except FileNotFoundError as exc:
                    status.update(label="Verification failed", state="error")
                    st.error(str(exc))
                except Exception as exc:
                    status.update(label="Verification failed", state="error")
                    st.error(f"Verification failed: {exc}")
                else:
                    status.update(label="Verification complete", state="complete")
                    st.session_state[f"verification:{written}"] = verification

        # Named `verification`, not `report`: a few lines above, `report` is
        # already bound to the text of agent 5's _report.md in this same block.
        #
        # Keyed by `written` (the current draft's path) rather than a bare
        # "verification" key: a fresh draft gets a fresh tempfile path every
        # run (see the NamedTemporaryFile above), so a result cached under
        # another draft's key is never looked up here, and a failed run for
        # this draft simply never populates this draft's key. Either way,
        # nothing renders that wasn't computed for this exact draft.
        verification = st.session_state.get(f"verification:{written}")
        if verification:
            totals = verification["totals"]
            needs_review = sum(
                totals.get(j, 0)
                for j in ("Contradicts", "Does not support",
                          "Unclear / insufficient evidence")
            )
            # Without this tile the four verdict counts do not add up to
            # "Checked", and the difference — orphaned keys, sources no longer
            # in the corpus, failed retrieval, failed or unusable model calls —
            # is invisible. A corpus re-ingested since the draft was cited
            # returns every citation `unresolved`, which would otherwise read
            # as "120 checked, nothing flagged".
            not_judged = totals["total"] - totals["judged"]
            cols = st.columns(5)
            cols[0].metric("Checked", totals["total"])
            cols[1].metric("Supports", totals.get("Supports", 0))
            cols[2].metric("Partially", totals.get("Partially supports", 0))
            cols[3].metric("Need review", needs_review)
            cols[4].metric("Not judged", not_judged)

            md_path = written.replace(".txt", "_verification.md")
            json_path = written.replace(".txt", "_verification.json")
            if os.path.exists(md_path):
                with open(md_path, encoding="utf-8") as fh:
                    report_md = fh.read()
                c_ver_h, c_ver_dl_md, c_ver_dl_json = st.columns([3, 1, 1])
                with c_ver_h:
                    st.markdown("##### 📋 Full Verification Report")
                with c_ver_dl_md:
                    st.download_button(
                        "📥 Download Report (.md)",
                        data=report_md,
                        file_name=os.path.basename(md_path),
                        mime="text/markdown",
                        key=f"dl_verif_md_{os.path.basename(written)}",
                        help="Download verification report in Markdown",
                    )
                if os.path.exists(json_path):
                    with open(json_path, encoding="utf-8") as jfh:
                        report_json = jfh.read()
                    with c_ver_dl_json:
                        st.download_button(
                            "📥 Download Data (.json)",
                            data=report_json,
                            file_name=os.path.basename(json_path),
                            mime="application/json",
                            key=f"dl_verif_json_{os.path.basename(written)}",
                            help="Download raw JSON verification results",
                        )
                # Open the report whenever anything is wrong — a run that
                # judged nothing is exactly when the reader needs it.
                with st.expander("Full verification report",
                                 expanded=needs_review > 0 or not_judged > 0):
                    st.markdown(report_md)


# ─── Tab 4: research chat ──────────────────────────────────────────────────

with tab_chat:
    st.markdown("### 🔬 Interactive Research Brainstorming Studio")
    st.caption("Multi-turn research collaboration grounded in your ingested literature. Condenses follow-up queries, retains evidence cards per turn, folds historical context into rolling memory, and suggests next exploration directions.")

    if chunks == 0:
        st.info("No corpus yet — build one in **Research a topic** first.", icon="📭")
    else:
        if "chat_agent" not in st.session_state or st.session_state["chat_agent"] is None:
            from research_assistant.agents.agent7_research_chat import ResearchChat

            try:
                cached_res = get_cached_search_resources(_get_resources_mtime())
                st.session_state["chat_agent"] = ResearchChat(top_k=5, search_resources=cached_res)
            except Exception as e:
                st.warning(f"Could not load search index: {e}")
                st.session_state["chat_agent"] = None

        agent = st.session_state.get("chat_agent")
        if agent is None:
            st.info("Ingest papers to enable research chat.", icon="📭")
        else:
            # Session state defaults
            st.session_state.setdefault("chat_scratchpad", [])
            st.session_state.setdefault("chat_pending_query", None)

            # Brainstorming modes & definitions
            LENS_LABELS = {
                "explore": "🧭 Explore (Broad connections)",
                "gaps": "💡 Gaps (Literature blind spots)",
                "contradictions": "⚔️ Contradictions (Competing theories)",
                "hypotheses": "🧪 Hypotheses (Novel proposals)",
                "methodology": "🔬 Methodology (Protocols & measurement)",
            }

            STARTERS_BY_MODE = {
                "explore": [
                    "What are the central findings and overarching themes across these papers?",
                    "How is the primary physical phenomenon modelled across the corpus?",
                    "What are the latest discoveries and their broader implications?",
                ],
                "gaps": [
                    "What are the major open questions or unaddressed gaps across these papers?",
                    "What experimental conditions or control parameters remain untested?",
                    "Where do the authors identify the need for future theoretical development?",
                ],
                "contradictions": [
                    "Where do the findings or interpretations in the corpus directly disagree?",
                    "Compare differing theoretical assumptions made by different authors.",
                    "Are there conflicting experimental measurements across these studies?",
                ],
                "hypotheses": [
                    "Propose a novel testable hypothesis connecting two or more papers in the corpus.",
                    "What new experiment could resolve the conflicting findings reported here?",
                    "How could the existing model be extended to account for unexplained observations?",
                ],
                "methodology": [
                    "Compare the experimental techniques and measurement setups used across papers.",
                    "What are the sample preparation conditions and measurement limitations?",
                    "Compare the numerical simulation methods and boundary conditions applied.",
                ],
            }

            # Top controls toolbar
            col_mode, col_clear, col_exp_md, col_exp_json = st.columns([3, 1, 1.2, 1.2])
            with col_mode:
                selected_label = st.selectbox(
                    "Brainstorming Lens",
                    options=list(LENS_LABELS.values()),
                    index=0,
                    key="chat_lens_selector",
                    label_visibility="collapsed",
                    help="Adjust the analytical lens and guidance used to evaluate the papers.",
                )
                active_mode = next((k for k, v in LENS_LABELS.items() if v == selected_label), "explore")
                agent.mode = active_mode

            with col_clear:
                if st.button("🗑️ Reset Chat", use_container_width=True, help="Clear conversation turns and memory"):
                    agent.clear_history()
                    st.session_state["chat_pending_query"] = None
                    st.rerun()

            with col_exp_md:
                export_path = agent.export_conversation(scratchpad_notes=st.session_state.get("chat_scratchpad"))
                try:
                    with open(export_path, encoding="utf-8") as _f:
                        export_md_text = _f.read()
                except Exception:
                    export_md_text = "# Research Chat Transcript\n"
                st.download_button(
                    "📥 Export (.md)",
                    data=export_md_text,
                    file_name=f"research_brainstorm_{int(time.time())}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    help="Download complete brainstorming session with sources and notes",
                )

            with col_exp_json:
                export_json_dict = agent.export_conversation_json(scratchpad_notes=st.session_state.get("chat_scratchpad"))
                st.download_button(
                    "💾 Export JSON",
                    data=json.dumps(export_json_dict, indent=2),
                    file_name=f"research_brainstorm_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Download structured JSON session data",
                )

            # Interactive Scratchpad & Pinned Ideas Drawer
            scratchpad = st.session_state.get("chat_scratchpad", [])
            with st.expander(f"📝 Brainstorm Scratchpad & Pinned Ideas ({len(scratchpad)})", expanded=False):
                st.caption("Pin important takeaways, hypotheses, or paper quotes as you brainstorm. These are included when exporting your session.")
                if scratchpad:
                    for idx, note in enumerate(scratchpad):
                        sc1, sc2 = st.columns([9, 1])
                        sc1.markdown(f"- {note}")
                        if sc2.button("✕", key=f"del_note_{idx}", help="Remove this note"):
                            st.session_state["chat_scratchpad"].pop(idx)
                            st.rerun()
                else:
                    st.info("No pinned ideas yet. Click '📌 Pin to Scratchpad' on any response below to save it here.")

                new_note = st.text_input("Add a manual research idea or hypothesis:", key="manual_note_input", placeholder="e.g. Test temperature dependence of variable-range hopping...")
                if st.button("➕ Add Note", key="add_manual_note_btn"):
                    if new_note.strip():
                        st.session_state["chat_scratchpad"].append(new_note.strip())
                        st.rerun()

            # Rolling Memory Display (if memory exists)
            if agent.memory:
                with st.expander("🧠 Rolling Conversation Memory (Context Summary)", expanded=False):
                    st.info(agent.memory)

            # Quick Starter Prompts (when no turns yet)
            if not agent.turns and not agent.history:
                with st.container(border=True):
                    st.markdown(f"##### 💡 Quick-start your brainstorming session ({selected_label}):")
                    starters = STARTERS_BY_MODE.get(active_mode, STARTERS_BY_MODE["explore"])
                    scols = st.columns(len(starters))
                    for i, s in enumerate(starters):
                        if scols[i].button(f"✨ {s}", key=f"starter_btn_{i}", use_container_width=True):
                            st.session_state["chat_pending_query"] = s
                            st.rerun()

            # Turn-by-Turn History Rendering
            if agent.turns:
                for turn in agent.turns:
                    with st.chat_message("user"):
                        st.markdown(turn["user_message"])
                        if turn.get("condensed_query") and turn["condensed_query"] != turn["user_message"]:
                            st.caption(f"🔍 *Searched literature for:* `{turn['condensed_query']}`")

                    with st.chat_message("assistant"):
                        mode_tag = turn.get("mode", "explore").capitalize()
                        st.caption(f"**Lens:** {mode_tag}")
                        st.markdown(turn["content"])

                        # Pin action
                        col_pin, col_empty = st.columns([2, 5])
                        if col_pin.button("📌 Pin to Scratchpad", key=f"pin_btn_{turn['turn_index']}"):
                            summary_snippet = turn["content"][:250].replace("\n", " ") + ("..." if len(turn["content"]) > 250 else "")
                            st.session_state["chat_scratchpad"].append(f"[Turn {turn['turn_index']}] {summary_snippet}")
                            st.toast("Saved to Scratchpad!")

                        # Per-turn Sources Expander
                        if turn.get("sources"):
                            with st.expander(f"📚 Sources & Evidence ({len(turn['sources'])}) · Turn {turn['turn_index']}"):
                                for s in turn["sources"]:
                                    status_badge = "✅ Cited" if s.get("cited") else "⚪ Referenced"
                                    score_text = f" · RRF: {s['rrf_score']:.3f}" if s.get("rrf_score") else ""
                                    st.markdown(f"**[{s['key']}]** `{s.get('document', 'Unknown')}` — {s.get('citation', '')} *(p.{s.get('page', '?')}, {s.get('section', 'text')})* `[{status_badge}{score_text}]`")
                                    if s.get("text"):
                                        with st.container(border=True):
                                            st.caption(f"Snippet: {s['text'][:300]}...")

                        if turn.get("warnings"):
                            for w in turn["warnings"]:
                                st.warning(f"⚠️ {w}")
            else:
                for msg in agent.history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # Clickable Follow-up Suggestion Chips Under Latest Turn
            if agent.last_suggestions and agent.turns:
                st.markdown("##### 💡 Next Exploration Directions (Click to brainstorm):")
                sug_cols = st.columns(min(len(agent.last_suggestions), 3))
                for idx, sug in enumerate(agent.last_suggestions[:3]):
                    if sug_cols[idx].button(f"➡️ {sug}", key=f"sug_btn_{idx}_{agent.turn_count}", use_container_width=True):
                        st.session_state["chat_pending_query"] = sug
                        st.rerun()

            # Chat Input & Processing
            chat_input_text = st.chat_input("Ask about the literature or brainstorm a hypothesis…")
            pending_query = st.session_state.pop("chat_pending_query", None)
            query_to_run = pending_query or chat_input_text

            if query_to_run:
                with st.chat_message("user"):
                    st.markdown(query_to_run)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Searching literature, condensing context, and formulating response…"):
                            stream = agent.stream_turn(query_to_run, mode=active_mode)
                            first_chunk = next(stream, None)
                        if first_chunk is not None:
                            def _generator():
                                yield first_chunk
                                yield from stream
                            st.write_stream(_generator())
                        else:
                            st.info("No response generated.")
                    except Exception as e:  # noqa: BLE001
                        st.exception(e)
                st.rerun()



# ─── Tab 5: how to use ─────────────────────────────────────────────────────

with tab_help:
    # Rendered from the repo's own HOW_TO_USE.md so the doc and the in-app help
    # cannot drift apart.
    guide = os.path.join(config.PROJECT_ROOT, "HOW_TO_USE.md")
    try:
        with open(guide, encoding="utf-8") as f:
            text = f.read()
        # Relative links are correct on GitHub but dead inside the app, which
        # serves no such routes — show them as filenames instead. http(s) links
        # are left alone.
        text = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"`\1`", text)
        st.markdown(text)
    except OSError:
        st.warning(
            "HOW_TO_USE.md is missing from this build — read it in the repo "
            "instead.",
            icon="📄",
        )
