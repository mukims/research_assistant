"""Agents package for research_assistant."""


def __getattr__(name):
    if name in ("GrobidAgent", "GrobidController"):
        from research_assistant.agents.grobid_controller import (
            GrobidAgent,
            GrobidController,
        )

        return GrobidAgent if name == "GrobidAgent" else GrobidController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GrobidAgent", "GrobidController"]
