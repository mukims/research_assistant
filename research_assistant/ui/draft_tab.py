"""Tab: cite and verify a whole draft. `chunks` is the corpus size shown in the sidebar."""

import json
import os

import streamlit as st

from research_assistant import config
from research_assistant.agents.agent8_verifier import verify_draft
from research_assistant.ui.common import _get_resources_mtime, get_cached_search_resources

def render_draft_tab(chunks):
    """Tab: cite and verify a whole draft. `chunks` is the corpus size shown in the sidebar."""
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
