"""Rendering one finished run: discovery trace, seed and downloads, shortlist, notes, synthesis, audit."""

import os

import streamlit as st

from research_assistant import config
from research_assistant.ui.audit import _render_seed_citation_audit
from research_assistant.ui.common import _manifest, _render_passages

def _render_discovery_trace(query, final):
    from research_assistant.agents import agent0_discoverer
    trace = final.get("discovery_trace")
    if not trace and query:
        trace = agent0_discoverer.get_discovery_trace(query)
    if not trace:
        return

    with st.expander("🔬 Literature Discovery Trace (Context & 3-Stage Search)", expanded=False):
        context = trace.get("context_profile") or final.get("context_profile")
        if context:
            st.markdown("##### 🧠 Formulated Scientific Context")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"**Primary Domain:** {context.get('primary_domain', 'N/A')}")
                st.markdown(f"**Core Problem:** {context.get('core_problem', 'N/A')}")
                subtopics = context.get("subtopics") or []
                if subtopics:
                    st.markdown(f"**Subtopics:** {', '.join(subtopics)}")
            with c2:
                ideal_type = str(context.get("ideal_paper_type") or "N/A").replace("_", " ").title()
                st.markdown(f"**Target Paper Type:** `{ideal_type}`")
                st.markdown(f"**Seed Criteria:** {context.get('seed_criteria', 'N/A')}")
                concepts = context.get("key_concepts") or []
                if concepts:
                    st.markdown(f"**Key Concepts:** {', '.join(concepts)}")

        search_trace = trace.get("search_trace") or {}
        iters = search_trace.get("iterations") or []
        if iters:
            st.markdown("---")
            total_cand = search_trace.get("total_candidates", len(iters))
            st.markdown(f"##### 🔍 3-Stage Iterative Search ({total_cand} candidates pooled)")
            for it in iters:
                it_num = it.get("iteration", 1)
                q_used = it.get("query_used", "")
                cands = it.get("candidates_found", 0)
                oa_cnt = it.get("open_access_count", 0)
                st.markdown(f"**Iteration {it_num}:** `\"{q_used}\"` — *{cands} candidates found ({oa_cnt} open-access)*")
                if it.get("gap_analysis"):
                    st.caption(f"↳ Gap Analysis: {it['gap_analysis']}")
                if it.get("pool_evaluation"):
                    st.caption(f"↳ Pool Evaluation: {it['pool_evaluation']}")

        ranked = trace.get("ranked_candidates") or []
        if ranked:
            st.markdown("---")
            st.markdown(f"##### 🏆 Evaluated Seed Candidates ({len(ranked)} scored)")
            for idx, cand in enumerate(ranked[:5], 1):
                score = cand.get("score")
                score_str = f"Score: {score:.1f}/10" if isinstance(score, (int, float)) else ""
                title = cand.get("title") or "Unknown Candidate"
                reasoning = cand.get("reasoning") or ""
                eval_method = cand.get("eval_method", "heuristic")
                badge = "🥇 Best Match" if idx == 1 else f"#{idx}"
                st.markdown(f"- **{badge}** {title} {f'· `{score_str}`' if score_str else ''}")
                if reasoning:
                    st.caption(f"↳ Reasoning ({eval_method}): {reasoning}")


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

        _render_discovery_trace(query, final)

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
