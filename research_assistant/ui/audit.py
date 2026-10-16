"""The citation-audit report for a seed paper, and the job that produces it."""

import json
import os
import re

import streamlit as st

from research_assistant.shared import pipeline_status

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
                    label = (
                        final.get("seed_label")
                        or final.get("query")
                        or (os.path.splitext(os.path.basename(seed_path))[0] if seed_path else "seed_paper")
                    )
                    if _start_audit_job(seed_path, label=label, origin="audit", force=True, existing_final=final):
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
            explain_verdict_steps,
            unassessed_bucket,
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

        # An unjudged citation whose PDF is already here asks for a re-run; one
        # with no claim to judge asks for nothing. Same grey badge, opposite
        # actions — so they get their own tabs.
        in_corpus_unjudged = [r for r in results if unassessed_bucket(r) == "in_corpus_unjudged"]
        nothing_to_judge = [
            r for r in results
            if r.get("outcome") not in ("judged", "not_downloaded", "deferred_paywalled")
            and unassessed_bucket(r) != "in_corpus_unjudged"
        ]
        tab_supported, tab_review, tab_not_assessed, tab_not_a_claim, tab_deferred, tab_all = st.tabs([
            f"Supported ({supp_count})",
            f"Need Review ({rev_count})",
            f"⏸ In Corpus, Not Judged ({len(in_corpus_unjudged)})",
            f"🔧 Not a Claim ({len(nothing_to_judge)})",
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

                steps = explain_verdict_steps(item)
                if steps:
                    st.markdown("**Why this verdict**")
                    st.markdown("\n".join(f"- **{label}:** {text}" for label, text in steps))

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
            if in_corpus_unjudged:
                st.info(
                    "The reference PDF for each of these is **already in the library** — the judge did not "
                    "reach them. Nothing needs to be fetched: raise the per-paper budget "
                    "(`CITATION_AUDIT_MAX_CLAIMS`, default 50) or re-run the audit. The badge says which "
                    "of the budget, the per-sentence cap or a backend failure stopped each one.",
                    icon="⏸",
                )
                for it in sorted(in_corpus_unjudged, key=lambda r: r.get("outcome") or ""):
                    _render_claim_item(it)
            else:
                st.success("Every citation with an available reference was assessed.")

        with tab_not_a_claim:
            if nothing_to_judge:
                st.info(
                    "These carry no assertion about the cited paper's findings — a software or dataset "
                    "citation, a pointer to a review, a sentence fragment, or a marker that could not be "
                    "matched to a bibliography entry. Re-running does not change them.",
                    icon="🔧",
                )
                for it in sorted(nothing_to_judge, key=lambda r: r.get("outcome") or ""):
                    _render_claim_item(it)
            else:
                st.success("Every citation in the paper carried a claim that could be judged.")

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
                        label = (
                            final.get("seed_label")
                            or final.get("query")
                            or stem
                        )
                        if _start_audit_job(seed_path, label=label, origin="audit", force=True, existing_final=final):
                            st.rerun()

        with tab_all:
            for it in results:
                _render_claim_item(it)

        if seed_path and os.path.exists(seed_path):
            if st.button("🔄 Re-run Seed Citation Audit", key="btn_rerun_seed_audit"):
                label = (
                    final.get("seed_label")
                    or final.get("query")
                    or stem
                )
                if _start_audit_job(seed_path, label=label, origin="audit", force=True, existing_final=final):
                    st.rerun()


def _start_audit_job(
    seed_file_path: str,
    label: str,
    origin: str = "audit",
    force: bool = True,
    existing_final: dict | None = None,
) -> str | None:
    """Re-run only the citation audit for a paper whose seed and references
    are already in the corpus — the case where a cached report exists but
    the judge never ran on it, or when the user explicitly triggers a re-run.
    Same job machinery as the full pipeline, so it too survives a refresh."""
    from research_assistant.shared import run_jobs
    from research_assistant.shared.seed_audit import audit_seed_citations

    job_id = run_jobs.job_id_for(os.path.basename(seed_file_path))

    def runner():
        pipeline_status.set_status(
            active=True, stage="respond", stage_label="Auditing citations",
            current_step=5, total_steps=5, detail=f"Auditing citations for: {label[:50]}",
        )
        pipeline_status.add_event(f"🔍 Auditing citations: {label[:40]}")
        try:
            audit = audit_seed_citations(seed_file_path, force=force, skip_if_cached=not force)
            pipeline_status.add_event("✅ Seed citation audit complete")
            result = dict(existing_final or {})
            result.update({
                "seed_path": seed_file_path,
                "seed_label": label,
                "citation_audit": audit,
                "audit_only": True,
            })
            return result
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
    for key in ("audit_result", "audit_query", "build_result", "build_query"):
        st.session_state.pop(key, None)
    st.query_params["job"] = job_id
    st.session_state["active_job"] = job_id
    st.session_state["active_job_origin"] = origin
    return job_id
