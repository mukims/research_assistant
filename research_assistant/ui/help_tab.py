"""Tab: the in-app copy of HOW_TO_USE.md."""

import os
import re

import streamlit as st

from research_assistant import config

def render_help_tab():
    """Tab: the in-app copy of HOW_TO_USE.md."""
    # Rendered from the repo's own HOW_TO_USE.md so the doc and the in-app help
    # cannot drift apart.
    guide = os.path.join(config.PROJECT_ROOT, "HOW_TO_USE.md")
    try:
        with open(guide, encoding="utf-8") as f:
            text = f.read()
        # Relative links are correct on GitHub but dead inside the app, which
        # serves no such routes — show them as filenames instead. http(s) links
        # are left alone.
        text = re.sub(r"\[([^\]]+)\]\((?!https?:)[^)]+\)", r"`\1`", text)
        st.markdown(text)
    except OSError:
        st.warning(
            "HOW_TO_USE.md is missing from this build — read it in the repo "
            "instead.",
            icon="📄",
        )
