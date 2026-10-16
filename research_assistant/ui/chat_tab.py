"""Tab: multi-turn research chat. `chunks` is the corpus size shown in the sidebar."""

import json
import time

import streamlit as st

from research_assistant.ui.common import _get_resources_mtime, get_cached_search_resources

def render_chat_tab(chunks):
    """Tab: multi-turn research chat. `chunks` is the corpus size shown in the sidebar."""
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
        else:
            # A session outlives index rebuilds: pick up the current corpus on every rerun.
            try:
                st.session_state["chat_agent"].use_search_resources(
                    get_cached_search_resources(_get_resources_mtime())
                )
            except Exception as e:
                st.warning(f"Could not refresh search index: {e}")

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
