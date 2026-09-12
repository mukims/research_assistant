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

import streamlit as st

from research_assistant import config
from research_assistant.agents.agent8_verifier import verify_draft
from research_assistant.shared import pipeline_status

st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
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
            with st.container(border=True):
                st.markdown("###### Related work across collection")
                st.markdown(answer["suggestion"])
            if answer.get("citations"):
                st.caption("Sources: " + " · ".join(str(c) for c in answer["citations"]))
            _render_passages(answer.get("passages") or [])
        else:
            st.info(
                "Corpus updated with uploaded papers. Switch to **Research chat** or **Cite a draft** to query them.",
                icon="✍️",
            )
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
        with st.container(border=True):
            st.markdown("###### Related work")
            st.markdown(answer["suggestion"])
        if answer.get("citations"):
            st.caption("Sources: " + " · ".join(str(c) for c in answer["citations"]))
        _render_passages(answer.get("passages") or [])
    else:
        st.info(
            "Corpus updated. Switch to **Cite a draft** to query it.", icon="✍️"
        )


# ─── Sidebar ────────────────────────────────────────────────────────────────

@st.fragment(run_every="3s")
def _render_sidebar_pipeline_status():
    try:
        from datetime import datetime, timezone

        status = pipeline_status.get_status()

        if status.get("active", False):
            st.info("⚡ **Pipeline Active**")
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

st.title("📚 Research Assistant")
st.caption(
    "Autonomous AI Research Assistant — Give it a research idea → it builds a "
    "corpus from the literature and tells you what's already been done. Or hand it "
    "a sentence and it finds the citation."
)

tab_build, tab_cite, tab_batch, tab_chat, tab_help = st.tabs(
    ["Research a topic", "Cite a draft", "Cite a whole draft", "Research chat",
     "How to use"]
)


@st.fragment(run_every="3s")
def _render_tab1_live_status():
    try:
        active_status = pipeline_status.get_status()
        if active_status.get("active", False) or _is_ingest_locked():
            with st.container(border=True):
                st.markdown("#### ⚡ Pipeline Active on Server")
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


# ─── Tab 1: build a corpus ─────────────────────────────────────────────────

with tab_build:
    _render_tab1_live_status()

    source_type = st.radio(
        "Start pipeline from",
        ["📄 Upload research paper(s) (PDF or ZIP)", "🔍 Search for a paper"],
        horizontal=True,
    )

    seed_file_path = None
    seed_url_val = None
    q = ""
    submitted = False
    ask = True
    force = False
    describe_figures = config.FIGURE_VLM

    if source_type == "📄 Upload research paper(s) (PDF or ZIP)":
        with st.form("upload_papers_form"):
            uploaded_files = st.file_uploader(
                "Select research paper(s) (.pdf or .zip)",
                type=["pdf", "zip"],
                accept_multiple_files=True,
                help="Upload one or multiple PDF papers, or a .zip archive of papers. If 1 paper is uploaded, its references can be mined; if multiple papers are uploaded, all are directly indexed into the corpus.",
            )
            pdf_query = st.text_input(
                "Research topic / question (optional)",
                placeholder="e.g. computational modeling of lipid nanocarriers (leave blank to infer from papers)",
                help="If provided, used to synthesize an answer across the papers at the end.",
            )
            c1, c2, c3 = st.columns(3)
            ask = c1.toggle("Answer my query at the end", value=True)
            force = c2.toggle("Force re-run every stage", value=False)
            describe_figures = c3.toggle(
                "Analyse figures and tables with the model",
                value=config.FIGURE_VLM,
                help="One choice for this whole run: every figure and table in every paper "
                     "is described by the model and the description joins the corpus. "
                     "Adds roughly a minute per figure on CPU.",
            )
            submitted = st.form_submit_button("Process and Index Paper(s)", type="primary")

        if submitted:
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
                    # Single PDF: execute full LangGraph pipeline (discover, seed ingest, reference extraction & fetch, synthesis)
                    seed_file_path = staged[0]["path"]
                    q = pdf_query.strip()
                else:
                    # Multiple PDFs: execute direct batch ingestion of all papers
                    st.session_state.pop("build_result", None)
                    st.session_state.pop("build_query", None)
                    with st.status(f"Ingesting {len(staged)} research papers into corpus…", expanded=True) as status:
                        batch_progress = st.progress(0.0, text=f"Preparing to ingest {len(staged)} papers…")
                        candidates = {}
                        for p in staged:
                            title_lbl = p.get("title") or p.get("key") or os.path.basename(p["path"])
                            candidates[p["path"]] = title_lbl
                            st.write(f"📄 Found: **{title_lbl}** (`{p.get('filename', os.path.basename(p['path']))}`)")

                        batch_progress.progress(0.2, text="Parsing and chunking papers…")
                        st.write("⚙️ Parsing text chunks, computing embeddings, and building vector index…")
                        from research_assistant.shared.ingestion import ingest_pdfs

                        def _on_batch_progress(st_data):
                            try:
                                try:
                                    ic = int(st_data.get("item_current") or 0)
                                    it = int(st_data.get("item_total") or 0)
                                except (ValueError, TypeError):
                                    ic, it = 0, 0
                                if it > 0:
                                    frac = min(1.0, max(0.0, ic / it))
                                    p_val = 0.2 + 0.75 * frac
                                    txt = f"Ingesting: {ic}/{it} papers ({int(frac * 100)}%)"
                                    if st_data.get("detail"):
                                        txt += f" · {st_data['detail'][:40]}"
                                    batch_progress.progress(min(0.98, max(0.0, p_val)), text=txt)
                            except Exception:
                                pass

                        unreg_batch = pipeline_status.register_progress_callback(_on_batch_progress)
                        try:
                            with pipeline_status.track_stage(
                                "ingest_refs",
                                f"Ingesting {len(staged)} uploaded papers",
                                current_step=5,
                                total_steps=5,
                                item_total=len(candidates),
                                mark_idle_on_exit=True,
                                last_summary=f"+{len(candidates)} uploaded papers indexed",
                            ):
                                ingest_res = ingest_pdfs(candidates, workers=1, skip_ingested=not force,
                                                         describe_figures=describe_figures)
                        finally:
                            unreg_batch()
                        batch_progress.progress(1.0, text="Ingestion complete!")
                        if ingest_res.get("described"):
                            st.write(f"🖼️ Described {ingest_res['described']} figure(s)/table(s).")
                        scanned_empty = ingest_res.get("scanned_or_empty", [])
                        inserted = ingest_res.get("inserted", 0)
                        processed = ingest_res.get("processed", len(candidates))

                        if scanned_empty and inserted == 0 and processed > 0:
                            st.error(
                                f"📄 **No selectable text found in: {', '.join(scanned_empty)}**. "
                                "This PDF appears to be a scanned photocopy or rasterized document without an embedded OCR text layer. "
                                "GROBID and PyMuPDF require digital selectable text. Please run OCR or upload a PDF with digital text.",
                                icon="⚠️",
                            )
                        elif scanned_empty:
                            st.warning(
                                f"⚠️ **{len(scanned_empty)} document(s) had no selectable text** ({', '.join(scanned_empty)}). "
                                f"The remaining documents were indexed successfully ({inserted} chunks inserted).",
                                icon="⚠️",
                            )
                        else:
                            st.write(
                                f"✅ Ingestion complete: {processed} processed, "
                                f"{inserted} chunks inserted into ChromaDB."
                            )

                        # Record in downloaded.json so they appear in manifests and sidebar
                        try:
                            import json, time
                            from research_assistant.shared.atomic import atomic_write_json

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
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Could not update downloaded.json: %s", e)

                        # Formulate synthesis if query or ask is set
                        answer = None
                        effective_q = pdf_query.strip() or (staged[0].get("title") if staged else "")
                        if ask and effective_q:
                            st.write("🧠 Formulating related-work synthesis across uploaded collection…")
                            from research_assistant.shared import retrieve

                            try:
                                answer = retrieve.research_answer(effective_q)
                            except Exception as exc:  # noqa: BLE001
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
                    st.session_state["build_result"] = final
                    st.session_state["build_query"] = effective_final_q
                    _corpus_stats.clear()
    else:
        with st.form("build_form"):
            query = st.text_input(
                "Research idea",
                placeholder="topological protection in disordered quantum wires",
            )
            seed_url = st.text_input(
                "Seed paper URL",
                placeholder="https://arxiv.org/abs/2401.12345 — leave blank to search automatically",
                help="Used when the search finds no open-access PDF. "
                     "Accepts an arXiv link or a direct .pdf URL.",
            )
            c1, c2, c3 = st.columns(3)
            ask = c1.toggle("Answer my query at the end", value=True)
            force = c2.toggle("Force re-run every stage", value=False)
            describe_figures = c3.toggle(
                "Analyse figures and tables with the model",
                value=config.FIGURE_VLM,
                help="One choice for this whole run: every figure and table in every paper "
                     "is described by the model and the description joins the corpus. "
                     "Adds roughly a minute per figure on CPU.",
            )
            submitted = st.form_submit_button("Build corpus", type="primary")

        if submitted:
            if not query.strip():
                st.warning("Please enter a research idea to search.", icon="⚠️")
            else:
                q = query.strip()
                seed_url_val = seed_url.strip() or None

    if submitted and (seed_file_path or q):
        st.session_state.pop("build_result", None)
        st.session_state.pop("build_query", None)
        graph = _graph()
        thread_seed = q or (os.path.basename(seed_file_path) if seed_file_path else "run")
        cfg = {"configurable": {"thread_id": hashlib.sha1(thread_seed.encode()).hexdigest()[:16]}}
        inputs = {
            "query": q,
            "workers": 1,
            "force": force,
            "ask": ask,
            "seed_url": seed_url_val,
            "seed_file": seed_file_path,
            "describe_figures": describe_figures,
        }

        # Filled progressively as nodes complete, so the seed + downloads show
        # up mid-run instead of only at the end.
        live = st.empty()

        with st.status("Running the pipeline…", expanded=True) as status:
            prog_bar = st.progress(0.0, text="Starting pipeline…")
            stage_ranges = {
                "discover": (0.0, 0.2),
                "ingest_seed": (0.2, 0.4),
                "extract": (0.4, 0.6),
                "fetch": (0.6, 0.8),
                "ingest_refs": (0.8, 0.95),
                "respond": (0.95, 1.0),
                "fallback": (0.95, 1.0),
            }

            def _on_pipeline_progress(st_data):
                try:
                    st_stage = st_data.get("stage", "discover")
                    p_low, p_high = stage_ranges.get(st_stage, (0.0, 0.2))
                    try:
                        ic = int(st_data.get("item_current") or 0)
                        it = int(st_data.get("item_total") or 0)
                        cs = int(st_data.get("current_step") or 1)
                        ts = int(st_data.get("total_steps") or 5)
                    except (ValueError, TypeError):
                        ic, it, cs, ts = 0, 0, 1, 5
                    lbl = st_data.get("stage_label") or st_stage
                    if it > 0:
                        frac = min(1.0, max(0.0, ic / it))
                        val = p_low + (p_high - p_low) * frac
                        txt = f"Step {cs}/{ts}: {lbl} — {ic}/{it} papers ({int(frac * 100)}%)"
                    else:
                        val = p_low + (p_high - p_low) * 0.25
                        txt = f"Step {cs}/{ts}: {lbl}"
                    detail = st_data.get("detail")
                    if detail:
                        txt += f" · {detail[:40]}"
                    prog_bar.progress(min(0.98, max(0.0, val)), text=txt)
                    curr_item = st_data.get("current_item_name")
                    if curr_item:
                        status.update(label=f"Pipeline: {lbl} — {curr_item[:40]}")
                except Exception:
                    pass

            unreg_pipeline = pipeline_status.register_progress_callback(_on_pipeline_progress)

            node_weights = {
                "discover": (1, 0.2),
                "ingest_seed": (2, 0.4),
                "extract": (3, 0.6),
                "fetch": (4, 0.8),
                "ingest_refs": (5, 0.95),
                "respond": (5, 1.0),
                "fallback": (5, 1.0),
            }
            display_q = q or (os.path.basename(seed_file_path) if seed_file_path else "") or (seed_url_val or "") or "topic"
            pipeline_status.set_status(
                active=True,
                stage="discover",
                stage_label="Finding seed paper",
                current_step=1,
                total_steps=5,
                detail=f"Starting pipeline for: {display_q[:50]}",
            )
            pipeline_status.add_event(f"🚀 Pipeline started for: {display_q[:40]}")
            final = {}
            try:
                for update in graph.stream(inputs, cfg, stream_mode="updates"):
                    for node, payload in update.items():
                        icon, label = STEPS.get(node, ("•", node))
                        st.write(f"{icon} {label}")
                        step_num, progress_val = node_weights.get(node, (1, 0.2))
                        prog_bar.progress(progress_val, text=f"Step {step_num}/5: {label}")
                        final.update(payload or {})
                        if node in ("discover", "ingest_seed", "extract", "fetch", "ingest_refs"):
                            with live.container():
                                effective_display_q = final.get("query") or q
                                _render_seed_and_downloads(effective_display_q, final)
                final = graph.get_state(cfg).values
                prog_bar.progress(1.0, text="Pipeline complete!")
                if final.get("stopped"):
                    status.update(label="Stopped early", state="error")
                    pipeline_status.add_event(f"⚠️ Pipeline stopped early: {final['stopped'][:60]}")
                    pipeline_status.set_status(
                        active=False,
                        stage="idle",
                        stage_label="Idle",
                        detail=f"Stopped: {final['stopped'][:60]}",
                    )
                else:
                    status.update(label="Done", state="complete")
                    import time
                    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    pipeline_status.add_event(f"✅ Pipeline completed: {display_q[:40]}")
                    pipeline_status.set_status(
                        active=False,
                        stage="idle",
                        stage_label="Idle",
                        detail="Pipeline complete",
                        last_completed_at=now_iso,
                        last_summary=f"Completed {display_q[:40]}",
                    )
            except BaseException as e:  # noqa: BLE001
                status.update(label="Pipeline failed", state="error")
                if isinstance(e, KeyboardInterrupt):
                    pipeline_status.add_event("⚠️ Pipeline cancelled by user (SIGINT)")
                    pipeline_status.set_status(
                        active=False,
                        stage="idle",
                        stage_label="Idle",
                        detail="Pipeline cancelled by user",
                    )
                else:
                    pipeline_status.add_event(f"❌ Pipeline failed: {e}")
                    pipeline_status.set_status(
                        active=False,
                        stage="idle",
                        stage_label="Idle",
                        detail=f"Pipeline failed: {e}",
                    )
                if isinstance(e, Exception):
                    st.exception(e)
                else:
                    raise
            finally:
                unreg_pipeline()
                # Agent 0 has copied the paper into RAW_DIR under its own key by
                # now, so the staging copy has served its purpose either way.
                if seed_file_path:
                    try:
                        os.unlink(seed_file_path)
                    except OSError:
                        pass

        live.empty()
        effective_final_q = final.get("query") or q or final.get("seed_label", "")
        st.session_state["build_result"] = final
        st.session_state["build_query"] = effective_final_q

    if st.session_state.get("build_result"):
        _render_build(
            st.session_state["build_result"],
            st.session_state.get("build_query", ""),
        )
    else:
        st.caption(
            "Agent 0 gets the seed paper → Agent 1 reads its references → "
            "Agent 2 fetches them → Agent 3 indexes everything → "
            "the top papers get summarised and matched to your topic."
        )


# ─── Tab 2: cite a draft ──────────────────────────────────────────────────

with tab_cite:
    if chunks == 0:
        st.info(
            "No corpus yet — build one in **Research a topic** first.", icon="📭"
        )

    with st.form("cite_form"):
        draft = st.text_area(
            "Your sentence",
            placeholder="Anderson localization suppresses diffusive transport in one dimension.",
            height=120,
        )
        top_k = st.slider("Passages to retrieve", 1, 10, config.DEFAULT_TOP_K)
        cite_submitted = st.form_submit_button(
            "Suggest a citation", type="primary", disabled=chunks == 0
        )

    if cite_submitted and draft.strip():
        from research_assistant.agents import agent4_assistant

        try:
            resources = get_cached_search_resources(_get_resources_mtime())
            with st.spinner("Retrieving and drafting…"):
                result = agent4_assistant.suggest_citation(
                    draft.strip(), top_k=top_k, search_resources=resources
                )
            st.session_state["cite_result"] = result or "empty"
        except RuntimeError:
            st.session_state["cite_result"] = "empty"
        except Exception as e:  # noqa: BLE001
            st.exception(e)
            st.session_state["cite_result"] = None

    cr = st.session_state.get("cite_result")
    if cr == "empty":
        st.info("Nothing in the corpus matched that text.", icon="🤷")
    elif isinstance(cr, dict):
        _render_suggestion(cr)


# ─── Tab 3: cite a whole draft ─────────────────────────────────────────────

with tab_batch:
    st.caption(
        "Upload a plain-text draft. Every sentence that makes a factual claim "
        "is checked against the corpus and cited where a source supports it."
    )
    if chunks == 0:
        st.info("No corpus yet — build one in **Research a topic** first.", icon="📭")

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
            if os.path.exists(md_path):
                with open(md_path, encoding="utf-8") as fh:
                    # Open the report whenever anything is wrong — a run that
                    # judged nothing is exactly when the reader needs it.
                    with st.expander("Full verification report",
                                     expanded=needs_review > 0 or not_judged > 0):
                        st.markdown(fh.read())


# ─── Tab 4: research chat ──────────────────────────────────────────────────

with tab_chat:
    st.caption("Multi-turn conversation grounded in the ingested corpus.")
    if chunks == 0:
        st.info("No corpus yet — build one in **Research a topic** first.", icon="📭")
    else:
        if "chat_agent" not in st.session_state:
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
            c1, c2 = st.columns([1, 4])
            if c1.button("Clear"):
                agent.clear_history()
                st.rerun()
            if c2.button("Export conversation"):
                st.success(f"Saved to {agent.export_conversation()}")

            for msg in agent.history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if question := st.chat_input("Ask about the literature…"):
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Searching literature and formulating response…"):
                            stream = agent.stream_turn(question)
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
                if agent.last_sources:
                    with st.expander(f"Sources · {len(agent.last_sources)}"):
                        for s in agent.last_sources:
                            st.caption(f"**{s['document']}** — {s['citation']}")



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
