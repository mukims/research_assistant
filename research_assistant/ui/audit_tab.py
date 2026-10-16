"""Tab: audit the citations of an uploaded manuscript against its references."""

import json
import os
import time

import streamlit as st

from research_assistant import config
from research_assistant.shared import pipeline_status
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.log import get_logger
from research_assistant.ui.audit import _start_audit_job
from research_assistant.ui.common import (
    _corpus_stats,
    _get_resources_mtime,
    get_cached_search_resources,
)
from research_assistant.ui.jobs import _handle_job_state, _render_recent_runs, _start_pipeline_job
from research_assistant.ui.results import _render_build
from research_assistant.ui.status import _render_live_pipeline_status

logger = get_logger("app")


def render_audit_tab():
    """Tab: audit the citations of an uploaded manuscript against its references."""
    _render_live_pipeline_status("tab_audit")

    st.subheader("📄 Seed Paper Citation Auditor")
    st.caption(
        "Upload a research manuscript (PDF or ZIP). The pipeline parses in-text citations with GROBID, "
        "retrieves open-access references from Unpaywall, Europe PMC, and arXiv, and audits each citation "
        "against the source text using Rubric V1.6."
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
