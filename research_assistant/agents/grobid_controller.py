"""
Agent for managing and monitoring the GROBID server lifecycle.

Re-exports GrobidManager, GrobidStatus, GrobidController, GrobidAgent
from research_assistant.shared.grobid_manager for agent-level access.
"""

from __future__ import annotations

from research_assistant.shared.grobid_manager import (
    GrobidAgent,
    GrobidController,
    GrobidManager,
    GrobidStatus,
    check_grobid_status,
    is_grobid_alive,
    main,
    restart_grobid,
    start_grobid,
    stop_grobid,
)

__all__ = [
    "GrobidAgent",
    "GrobidController",
    "GrobidManager",
    "GrobidStatus",
    "check_grobid_status",
    "is_grobid_alive",
    "main",
    "restart_grobid",
    "start_grobid",
    "stop_grobid",
]


if __name__ == "__main__":
    main()
