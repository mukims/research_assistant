"""
Launch settings for running model-written Python from the task planner.

The planner's autonomous mode executes code an LLM wrote after reading an
untrusted paper. Before this module the child inherited ``os.environ``, so
GEMINI_API_KEY, OPENAI_API_KEY and APP_PASSWORD were readable by that code.
Now the child gets an allowlisted environment (``sandbox_env``) and an
interpreter that ignores PYTHON* variables and the user site (``python_argv``).

RESIDUAL RISK -- this is secret hygiene, not a sandbox. Generated code still
runs as the user: it can read any file the user can (including the project
``.env``, ``~/.ssh`` and the corpus), write anywhere the user can write, and
open network connections. ``CITATION_DISABLE_DOTENV`` only stops *our* config
loader from reading ``.env`` if the child imports ``research_assistant``; it
does not stop the code opening the file itself. Real isolation (a container
with no mounted secrets and no network) is strategic bet B-sandbox. Until then
autonomous mode needs explicit consent in the UI and ``--allow-untrusted-code``
on the command line.
"""

import os
import sys
from typing import List, Mapping, Optional

# Only these survive from the parent environment. Everything else, including
# every *_API_KEY, *_TOKEN and APP_PASSWORD, is dropped by omission.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ", "SYSTEMROOT")

# Numeric libraries spawn one worker per core by default; the harness runs many
# short tasks, so cap them all at one thread.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)


def sandbox_env(workspace_dir: str, base: Optional[Mapping[str, str]] = None) -> dict:
    """Environment for a generated-code child: allowlisted keys of ``base`` plus fixed settings."""
    source = os.environ if base is None else base
    env = {key: source[key] for key in ENV_ALLOWLIST if key in source}
    # HOME inside the workspace keeps ~ lookups (config dirs, caches) out of the user's real home.
    env["HOME"] = workspace_dir
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # If the child imports research_assistant.config, do not let it load the project .env.
    env["CITATION_DISABLE_DOTENV"] = "1"
    for name in _THREAD_VARS:
        env[name] = "1"
    return env


def python_argv(*args: str) -> List[str]:
    """argv for the current interpreter in isolated-ish mode: ``-E`` (ignore PYTHON* vars), ``-s`` (no user site).

    Deliberately not ``-I``: that also drops the script directory and the cwd
    from sys.path, and generated code imports its sibling modules.
    """
    return [sys.executable, "-E", "-s", *args]
