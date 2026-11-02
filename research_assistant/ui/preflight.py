"""First-run setup checks in the page: a banner when something blocking is missing,
and a "Setup checks" expander in the sidebar that lists every check."""

from dataclasses import asdict

import streamlit as st

from research_assistant.shared.preflight import run_checks


@st.cache_data(ttl=60, show_spinner=False)
def _cached_checks() -> list:
    # dicts, not Check objects: st.cache_data pickles the result, and a minute is
    # short enough that fixing the problem shows up without a restart.
    return [asdict(c) for c in run_checks()]


def render_preflight_banner() -> None:
    failing = [c for c in _cached_checks() if c["blocking"] and not c["ok"]]
    if not failing:
        return
    lines = "\n".join(f"- **{c['name']}**: {c['fix'] or c['detail']}" for c in failing)
    st.error(f"Marvin can't run yet:\n\n{lines}", icon="🚫")


def render_preflight_sidebar() -> None:
    with st.sidebar.expander("Setup checks"):
        for c in _cached_checks():
            icon = "✅" if c["ok"] else ("❌" if c["blocking"] else "⚠️")
            st.markdown(f"{icon} **{c['name']}**: {c['detail']}")
            if not c["ok"] and c["fix"]:
                st.code(c["fix"], language="bash")
