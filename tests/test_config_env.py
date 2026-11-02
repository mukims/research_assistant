"""config.load_env_files: the real environment beats every .env file."""

import os
import subprocess
import sys
from pathlib import Path

from research_assistant import config

REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def test_real_environment_beats_dotenv(tmp_path):
    f = tmp_path / ".env"
    f.write_text("A=from_file\nB=from_file\n")
    env = {"A": "real"}
    config.load_env_files([str(f)], protected=frozenset(env), environ=env)
    assert env == {"A": "real", "B": "from_file"}


def test_later_file_overrides_earlier_but_not_real_env(tmp_path):
    p, d = tmp_path / "p.env", tmp_path / "d.env"
    p.write_text("K=project\nR=project\n")
    d.write_text("K=data\nR=data\n")
    env = {"R": "real"}
    config.load_env_files([str(p), str(d)], protected=frozenset(env), environ=env)
    assert env == {"K": "data", "R": "real"}


def test_missing_file_is_ignored(tmp_path):
    env = {}
    config.load_env_files([str(tmp_path / "nope.env")], protected=frozenset(), environ=env)
    assert env == {}


def test_disable_switch_keeps_keys_out(tmp_path):
    # The maintainer's macOS checkout has `.ENV`, which a case-insensitive FS serves as `.env`.
    out = subprocess.run(
        [sys.executable, "-c",
         "from research_assistant import config; print(bool(config.GEMINI_API_KEY), config.LLM_BACKEND)"],
        env={"PATH": os.environ["PATH"], "CITATION_DISABLE_DOTENV": "1"},
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert out.stdout.split() == ["False", "ollama"]
