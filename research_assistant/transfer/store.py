"""Persistent storage for technique transfer run results."""

from __future__ import annotations

import json
import os

from research_assistant import config
from research_assistant.shared.atomic import atomic_write, atomic_write_json
from research_assistant.transfer.models import RunResult, result_from_dict, result_to_dict
from research_assistant.transfer.render import render_markdown


def save_result(result: RunResult, directory: str | None = None) -> str:
    """Save RunResult as JSON and Markdown under target directory. Returns JSON path."""
    target_dir = directory or os.path.join(config.TRANSFER_DIR, "runs")
    os.makedirs(target_dir, exist_ok=True)
    json_path = os.path.join(target_dir, f"{result.run_id}.json")
    md_path = os.path.join(target_dir, f"{result.run_id}.md")
    atomic_write_json(json_path, result_to_dict(result))
    with atomic_write(md_path) as fh:
        fh.write(render_markdown(result))
    return json_path


def load_result(path: str) -> RunResult:
    """Load RunResult from saved JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return result_from_dict(data)
