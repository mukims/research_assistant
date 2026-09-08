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

st.set_page_config(page_title="Research assistant", page_icon="📚", layout="centered")
os.makedirs(config.DATA_DIR, exist_ok=True)


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
    return chunks, papers


@st.cache_data(ttl=60, show_spinner=False)
def _grobid_ok():
    import requests

    try:
        r = requests.get(f"{config.GROBID_SERVER}/api/isalive", timeout=8)
        return r.ok and "true" in r.text.lower()
    except Exception:
        return False


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
        for s in reversed(list(seeds.values())):
            if s.get("path") and os.path.abspath(s["path"]) == norm_path:
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

    if final.get("seed_path"):
        downloaded = _manifest(config.DOWNLOADED_JSON_PATH)
        failed = _manifest(config.FAILED_DOWNLOADS_PATH)
        if downloaded or failed:
            with st.expander(
                f"📄 Reference PDFs — {len(downloaded)} fetched, {len(failed)} unavailable",
                expanded=bool(downloaded) and not final.get("answer"),
            ):
                for rec in downloaded.values():
                    t = rec.get("title") or rec.get("raw_reference") or rec.get("key")
                    st.markdown(f"- ✅ {t}")
                for rec in failed.values():
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

with st.sidebar:
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

    grobid_up = _grobid_ok()
    st.caption(("🟢" if grobid_up else "🔴") + f" **GROBID** — {config.GROBID_SERVER}")
    if not grobid_up:
        st.warning(
            "GROBID isn't responding — Agent 1 will fall back to weaker, "
            "unstructured reference extraction (no DOIs, no authors).",
            icon="⚠️",
        )
    if config.LLM_BACKEND == "openai" and not config.OPENAI_API_KEY:
        st.error("No OPENAI_API_KEY / HF_TOKEN set.", icon="🚫")


# ─── Main ───────────────────────────────────────────────────────────────────

st.title("📚 Research Assistant")
st.caption(
    "Give it a research idea → it builds a corpus from the literature and tells "
    "you what's already been done. Or hand it a sentence and it finds the "
    "citation."
)

tab_build, tab_cite, tab_batch, tab_chat, tab_help = st.tabs(
    ["Research a topic", "Cite a draft", "Cite a whole draft", "Research chat",
     "How to use"]
)


# ─── Tab 1: build a corpus ─────────────────────────────────────────────────

with tab_build:
    source_type = st.radio(
        "Start pipeline from",
        ["📄 Upload a seed PDF", "🔍 Search for a paper"],
        horizontal=True,
    )

    seed_file_path = None
    seed_url_val = None
    q = ""
    submitted = False
    ask = True
    force = False

    if source_type == "📄 Upload a seed PDF":
        with st.form("upload_seed_form"):
            uploaded_pdf = st.file_uploader(
                "Seed paper (.pdf)",
                type=["pdf"],
                help="Start from this PDF as the seed paper. Its references will be extracted, fetched, and indexed.",
            )
            pdf_query = st.text_input(
                "Research topic / question (optional)",
                placeholder="e.g. topological protection in disordered wires (leave blank to infer from paper)",
                help="If provided, used for the final related-work answer. If blank, automatically inferred from the paper title or filename.",
            )
            c1, c2 = st.columns(2)
            ask = c1.toggle("Answer my query at the end", value=True)
            force = c2.toggle("Force re-run every stage", value=False)
            submitted = st.form_submit_button("Build corpus from PDF", type="primary")

        if submitted:
            if not uploaded_pdf:
                st.warning("Please upload a PDF file to begin.", icon="⚠️")
            elif uploaded_pdf.size == 0:
                st.warning("The uploaded PDF is empty (0 bytes).", icon="⚠️")
            else:
                seed_file_path = _stage_upload(uploaded_pdf)
                q = pdf_query.strip()
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
            c1, c2 = st.columns(2)
            ask = c1.toggle("Answer my query at the end", value=True)
            force = c2.toggle("Force re-run every stage", value=False)
            submitted = st.form_submit_button("Build corpus", type="primary")

        if submitted:
            if not query.strip():
                st.warning("Please enter a research idea to search.", icon="⚠️")
            else:
                q = query.strip()
                seed_url_val = seed_url.strip() or None

    if submitted and (seed_file_path or q):
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
        }

        # Filled progressively as nodes complete, so the seed + downloads show
        # up mid-run instead of only at the end.
        live = st.empty()

        with st.status("Running the pipeline…", expanded=True) as status:
            final = {}
            try:
                for update in graph.stream(inputs, cfg, stream_mode="updates"):
                    for node, payload in update.items():
                        icon, label = STEPS.get(node, ("•", node))
                        st.write(f"{icon} {label}")
                        final.update(payload or {})
                        if node in ("discover", "ingest_seed", "fetch", "ingest_refs"):
                            with live.container():
                                effective_display_q = final.get("query") or q
                                _render_seed_and_downloads(effective_display_q, final)
                final = graph.get_state(cfg).values
                if final.get("stopped"):
                    status.update(label="Stopped early", state="error")
                else:
                    status.update(label="Done", state="complete")
            except Exception as e:  # noqa: BLE001
                status.update(label="Pipeline failed", state="error")
                st.exception(e)
            finally:
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
        from research_assistant.shared.db import load_search_resources
        from research_assistant.agents import agent4_assistant

        try:
            resources = load_search_resources()
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
                written = agent5_batch_citer.run_batch_citer(draft_path, out_path)
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


# ─── Tab 4: research chat ──────────────────────────────────────────────────

with tab_chat:
    st.caption("Multi-turn conversation grounded in the ingested corpus.")
    if chunks == 0:
        st.info("No corpus yet — build one in **Research a topic** first.", icon="📭")
    else:
        if "chat_agent" not in st.session_state:
            from research_assistant.agents.agent7_research_chat import ResearchChat

            st.session_state["chat_agent"] = ResearchChat(top_k=5)
        agent = st.session_state["chat_agent"]

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
                    st.write_stream(agent.stream_turn(question))
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
