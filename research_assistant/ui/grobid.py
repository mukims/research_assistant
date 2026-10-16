"""The sidebar's GROBID status and Start/Stop controls."""

import streamlit as st

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
