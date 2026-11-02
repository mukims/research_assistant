"""Tab: cite and verify a whole draft. `chunks` is the corpus size shown in the sidebar."""

import json
import os
from typing import Literal

import streamlit as st

from research_assistant import config
from research_assistant.agents.agent8_verifier import verify_draft
from research_assistant.shared.draft_input import DraftInputError, read_draft
from research_assistant.ui.common import _get_resources_mtime, get_cached_search_resources
from research_assistant.ui.jobs import _handle_job_state


def _run_cite(draft_path: str, out_path: str, search_resources) -> dict:
    """Job runner: cite the draft. Runs in a job thread (see shared/run_jobs.py),
    so it must never touch Streamlit; the session resolves *search_resources*
    first because the cache that holds them is a Streamlit one."""
    from research_assistant.agents import agent5_batch_citer

    written = agent5_batch_citer.run_batch_citer(draft_path, out_path, search_resources=search_resources)
    return {"written": written}


def _run_verify(written: str) -> dict:
    """Job runner: judge every citation in a cited draft. No Streamlit (job thread).
    The draft's path rides along so a session that was not there when the run
    started can file the result under the right draft."""
    return {"verification": verify_draft(written), "written": written}


def _start_draft_job(kind: Literal["cite", "verify"], text_or_path: str) -> str | None:
    """Start citing a draft (*text_or_path* is its text) or verifying a cited
    one (*text_or_path* is the cited draft's path) as a server-side job and
    remember it in the URL, so a refresh finds the run again.

    Returns the job id, or None when it could not start (the reason is shown).
    """
    import tempfile

    from research_assistant.shared import pipeline_status, run_jobs

    draft_path = None
    if kind == "cite":
        try:
            cached_res = get_cached_search_resources(_get_resources_mtime())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Citing failed: {exc}", icon="❌")
            return None
        os.makedirs(config.DRAFTS_DIR, exist_ok=True)
        # A fresh path per run: verification results are filed under it, so a
        # re-cited draft never shows an older draft's verification.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", dir=config.DRAFTS_DIR, delete=False, encoding="utf-8"
        ) as fh:
            fh.write(text_or_path)
            draft_path = fh.name
        out_path = draft_path.replace(".txt", "_cited.txt")
        job_id = run_jobs.job_id_for(f"draft-cite:{draft_path}")
        label = "Citing the draft"

        def runner():
            return _run_cite(draft_path, out_path, cached_res)
    else:
        job_id = run_jobs.job_id_for(f"draft-verify:{text_or_path}")
        label = "Verifying citations"

        def runner():
            return _run_verify(text_or_path)

    metadata = {"kind": kind, "label": label, "draft_path": draft_path or text_or_path}
    pipeline_status.clear_cancel_request(job_id=job_id)
    try:
        run_jobs.start_job(job_id, label, runner, origin="draft", metadata=metadata)
    except run_jobs.JobBusy as busy:
        if draft_path:
            try:
                os.unlink(draft_path)
            except OSError:
                pass
        st.warning(
            f"A run is already active on the server ({busy.job.label}). "
            "Wait for it to finish, or stop it from the sidebar, then submit again.",
            icon="⏳",
        )
        return None
    if kind == "cite":
        # The previous draft's result must not sit under this run's outcome.
        st.session_state.pop("batch_result", None)
    st.query_params["job"] = job_id
    st.session_state["active_job"] = job_id
    st.session_state["active_job_origin"] = "draft"
    return job_id


def render_draft_tab(chunks):
    """Tab: cite and verify a whole draft. `chunks` is the corpus size shown in the sidebar."""
    st.subheader("✍️ Cite a Research Draft")
    st.caption(
        "Upload a plain-text draft (.txt or .md) or paste it below. Every sentence that makes a factual claim "
        "is checked against your indexed corpus, cited where a source supports it, and verified for accuracy."
    )
    if chunks == 0:
        st.info("No corpus yet — build one in **Citation auditor** or **Research idea** first.", icon="📭")

    # A running or just-finished citing/verification job (this session's, or one
    # found in the URL after a refresh) is shown here instead of the form.
    if _handle_job_state("draft"):
        return

    uploaded = st.file_uploader("Draft (.txt, .md)", type=["txt", "md"], key="batch_upload")
    pasted = st.text_area("…or paste it here", height=200, key="batch_paste")
    run_batch = st.button("Cite the draft", type="primary", disabled=chunks == 0)

    if run_batch:
        try:
            if uploaded:
                text = read_draft(uploaded.name, uploaded.read())
            else:
                text = read_draft("pasted.txt", pasted.encode("utf-8"))
        except DraftInputError as exc:
            st.error(str(exc), icon="⚠️")
        else:
            if _start_draft_job("cite", text):
                st.rerun()

    written = st.session_state.get("batch_result")
    if written is None and "batch_result" in st.session_state:
        # Only a job that finished and wrote nothing lands here: run_batch_citer
        # returns None when it aborted before writing. A job that failed is
        # reported by _handle_job_state and never sets batch_result.
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
            if _start_draft_job("verify", written):
                st.rerun()

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
