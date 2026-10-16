"""Pipeline progress widgets: the sidebar summary and the live panel at the top of each tab."""

import streamlit as st

from research_assistant.shared import pipeline_status
from research_assistant.ui.common import _is_ingest_locked

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
