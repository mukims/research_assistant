"""Streamlit tab for Cross-Disciplinary Technique Transfer."""

import html
import json
import os
import streamlit as st

from research_assistant import config
from research_assistant.shared.log import get_logger
from research_assistant.transfer.models import RunResult, result_from_dict, result_to_dict
from research_assistant.transfer.render import render_markdown, render_tree
from research_assistant.ui.status import _render_live_pipeline_status

logger = get_logger("transfer_tab")


TRANSFER_EXAMPLES = [
    "Inverse problem parameter recovery under correlated noise",
    "Topological protection in disordered one-dimensional lattices",
    "Phase retrieval in high-dimensional non-convex optimization",
    "Diffusion-based generative sampling with physical conservation laws",
]


def _card_store_summary() -> tuple[int, int, int]:
    """Return (total_cards, verified_cards, unique_docs) from TRANSFER_DIR/cards.jsonl."""
    cards_path = os.path.join(config.TRANSFER_DIR, "cards.jsonl")
    if not os.path.isfile(cards_path):
        return 0, 0, 0
    total = 0
    verified = 0
    docs: set[str] = set()
    try:
        with open(cards_path, "r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                total += 1
                item = json.loads(raw)
                doc = item.get("document")
                if doc:
                    docs.add(str(doc))
                ev = item.get("evidence") or {}
                if ev.get("judgement") in ("Supports", "Partially supports") and ev.get("span_verified") is True:
                    verified += 1
    except Exception:
        return 0, 0, 0
    return total, verified, len(docs)


def _saved_runs_map() -> dict[str, str]:
    """Return {label: json_path} for saved transfer runs in TRANSFER_DIR/runs/."""
    runs_dir = os.path.join(config.TRANSFER_DIR, "runs")
    if not os.path.isdir(runs_dir):
        return {}
    mapping: dict[str, str] = {}
    try:
        files = sorted(
            (f for f in os.listdir(runs_dir) if f.endswith(".json")),
            reverse=True,
        )
        for fname in files[:30]:
            fpath = os.path.join(runs_dir, fname)
            run_id = fname[:-5]
            label = run_id
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                q = str(meta.get("query") or "").strip()
                n_t = len(meta.get("transfers") or [])
                if q:
                    short_q = q if len(q) <= 60 else q[:57] + "..."
                    label = f"{run_id} — {short_q} ({n_t} transfers)"
            except Exception:
                pass
            mapping[label] = fpath
    except Exception:
        return {}
    return mapping


def render_transfer_tab():
    """Render the Cross-Disciplinary Technique Transfer tab."""
    _render_live_pipeline_status("tab_transfer")

    st.subheader("🔄 Cross-Disciplinary Technique Transfer")
    st.caption(
        "Bridge methodologies across scientific disciplines. Enter a target bottleneck, problem formulation, "
        "or computational challenge, and Marvin explores transfer cards from other literature domains, "
        "maps foundational analogies, analyzes identifiability, tests critic objections, and generates novel hypotheses."
    )

    total_cards, verified_cards, card_docs = _card_store_summary()
    if total_cards > 0:
        st.caption(
            f"📇 **Card index:** {verified_cards:,} verified / {total_cards:,} total technique cards "
            f"across {card_docs:,} papers (`data/transfer/cards.jsonl`)."
        )
    else:
        st.caption(
            "📇 **Card index:** No technique cards built yet (`python scripts/build_cards.py`)."
        )

    with st.expander("💡 Example Bottlenecks & Problems (Click to prefill)", expanded=False):
        scols = st.columns(len(TRANSFER_EXAMPLES))
        for idx, starter in enumerate(TRANSFER_EXAMPLES):
            if scols[idx].button(f"✨ {starter}", key=f"transfer_starter_{idx}", use_container_width=True):
                st.session_state["transfer_prefilled_query"] = starter
                st.rerun()

    prefill = st.session_state.pop("transfer_prefilled_query", None)
    if prefill is not None:
        st.session_state["transfer_query_input"] = prefill

    with st.form("transfer_form"):
        query = st.text_input(
            "Target Problem or Bottleneck",
            key="transfer_query_input",
            placeholder="e.g. recovering continuous parameters from noisy spectra",
            help="Describe the physical problem, bottleneck, or mathematical challenge.",
        )

        col_b, col_tech, col_cands, col_year, col_temp = st.columns(5)
        with col_b:
            bottlenecks = st.slider(
                "Bottlenecks",
                min_value=1,
                max_value=5,
                value=int(config.TRANSFER_BOTTLENECKS),
                help="Maximum technical bottlenecks to extract from near literature.",
            )
        with col_tech:
            techniques_per_b = st.slider(
                "Techniques / bottleneck",
                min_value=1,
                max_value=6,
                value=int(config.TRANSFER_TECHNIQUES_PER_BOTTLENECK),
                help="Top candidate technique cards to develop per bottleneck.",
            )
        with col_cands:
            candidates = st.slider(
                "Cards per query",
                min_value=1,
                max_value=25,
                value=int(config.TRANSFER_CARDS_PER_QUERY),
                help="Search depth per abstraction level when retrieving candidate cards.",
            )
        with col_year:
            before_year = st.number_input("Prior art cutoff year (optional)", min_value=1900, max_value=2030, value=2024)
        with col_temp:
            temperature = st.slider("Proposal temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        run_btn = st.form_submit_button("🚀 Find Cross-Disciplinary Transfers", type="primary", use_container_width=True)

    if run_btn and not query.strip():
        st.warning("Please enter a target problem or bottleneck.")
    elif run_btn and query.strip():
        with st.status(f"Searching technique transfers for '{query}'…", expanded=True) as status_box:
            st.write("🔍 Identifying candidate problem cards & source methodologies…")
            try:
                from research_assistant.transfer.engine import default_engine
                from research_assistant.transfer import store

                def on_progress(done: int, total: int, label: str):
                    status_box.write(f"⚙️ [{done}/{total}] {label}")

                engine = default_engine()
                # Run the transfer engine search
                result = engine.run(
                    query.strip(),
                    candidates=candidates,
                    temperature=temperature,
                    prior_art_before_year=int(before_year) if before_year else None,
                    bottlenecks=bottlenecks,
                    techniques_per_bottleneck=techniques_per_b,
                    progress_cb=on_progress,
                )
                try:
                    saved_path = store.save_result(result)
                    logger.info("Saved technique transfer run %s to %s", result.run_id, saved_path)
                except Exception as save_exc:
                    logger.warning("Could not persist technique transfer run %s: %s", result.run_id, save_exc)
                st.session_state["latest_transfer_result"] = result_to_dict(result)
                status_box.update(label="✅ Technique transfer search complete!", state="complete", expanded=False)
            except Exception as exc:
                logger.exception("Transfer engine execution error for query %r: %s", query.strip(), exc)
                status_box.update(label=f"❌ Transfer search failed: {exc}", state="error")
                st.error(f"Transfer engine execution error: {exc}")

    runs_map = _saved_runs_map()
    if runs_map:
        options = ["(Current session)"] + list(runs_map.keys())
        chosen = st.selectbox("📂 Load a saved transfer run", options=options, key="transfer_saved_run_picker")
        if isinstance(chosen, str) and chosen in runs_map:
            try:
                with open(runs_map[chosen], "r", encoding="utf-8") as fh:
                    st.session_state["latest_transfer_result"] = json.load(fh)
            except Exception as load_exc:
                logger.warning("Failed to load saved transfer run %s: %s", chosen, load_exc)

    # Display results if present in session state
    saved_result_dict = st.session_state.get("latest_transfer_result")
    if saved_result_dict:
        try:
            result = result_from_dict(saved_result_dict)
            st.divider()
            st.markdown(f"### 📋 Transfer Results: *{html.escape(result.query)}*")

            t_tab_report, t_tab_tree, t_tab_json = st.tabs(["📄 Full Transfer Report", "🌳 Transfer Tree", "💾 Raw JSON"])

            with t_tab_report:
                report_md = render_markdown(result)
                st.markdown(report_md)

            with t_tab_tree:
                mermaid_tree = render_tree(result)
                st.markdown("#### Methodology Search Tree")
                from research_assistant.tasks.ui import _render_mermaid_interactive
                _render_mermaid_interactive(mermaid_tree, height=500)

            with t_tab_json:
                st.json(saved_result_dict)

        except Exception as exc:
            st.warning(f"Could not render stored transfer result: {exc}")
