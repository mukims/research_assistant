"""First-run setup checks in the page: a banner when something blocking is missing,
and a "Setup checks" expander in the sidebar that lists every check."""

from dataclasses import asdict

import streamlit as st

from research_assistant import config
from research_assistant.shared.preflight import run_checks


def _session_target() -> tuple[str, str]:
    llm_backend = st.session_state.get("llm_backend") or config.LLM_BACKEND
    chat_model = st.session_state.get("chat_model") or config.CHAT_MODEL
    return llm_backend, chat_model


@st.cache_data(ttl=60, show_spinner=False)
def _cached_checks(llm_backend: str, chat_model: str) -> list:
    # dicts, not Check objects: st.cache_data pickles the result, keyed by the
    # session's active backend and chat model so switching the sidebar picker
    # updates the checks immediately; a 60s TTL picks up external daemon fixes.
    return [asdict(c) for c in run_checks(llm_backend=llm_backend, chat_model=chat_model)]


def render_preflight_banner() -> None:
    failing = [c for c in _cached_checks(*_session_target()) if c["blocking"] and not c["ok"]]
    if not failing:
        return
    lines = "\n".join(f"- **{c['name']}**: {c['fix'] or c['detail']}" for c in failing)
    st.error(f"Marvin can't run yet:\n\n{lines}", icon="🚫")


def render_preflight_sidebar() -> None:
    with st.sidebar.expander("Setup checks"):
        for c in _cached_checks(*_session_target()):
            icon = "✅" if c["ok"] else ("❌" if c["blocking"] else "⚠️")
            st.markdown(f"{icon} **{c['name']}**: {c['detail']}")
            if not c["ok"] and c["fix"]:
                st.code(c["fix"], language="bash")
