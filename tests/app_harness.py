"""Run the real app.py headlessly, in a subprocess, on a scratch data directory.

Streamlit's AppTest executes the actual script. It runs out of process because
executing app.py opens the corpus store to count its chunks, and a test must never
open the real one: CITATION_DATA_DIR is set to a fresh temp directory before
anything imports research_assistant. GROBID's status is stubbed so no Docker or
network call is made.

run_page() takes two snippets of Python that run inside that subprocess:

    before   after `at = AppTest.from_file(...)`, before the first run:
             set at.session_state, create files under `data`.
    after    once the first run has finished: click widgets, run again.

Both see `at`, `data` (the scratch directory) and `out` (a dict that is printed back
as JSON). The page's exceptions, tab labels and so on are added to `out` last.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE_STREAMLIT = find_spec("streamlit") is not None

TEMPLATE = """
import json, os, sys
from unittest import mock

sys.path.insert(0, @ROOT@)
os.environ["CITATION_DATA_DIR"] = @DATA@
os.environ["CITATION_LOG_FILE"] = "0"
for key in ("APP_PASSWORD", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(key, None)

from streamlit.testing.v1 import AppTest
from research_assistant import config
from research_assistant.agents import grobid_controller

data = @DATA@
assert os.path.realpath(config.DATA_DIR) == os.path.realpath(data), config.DATA_DIR

status = grobid_controller.GrobidStatus(
    is_alive=False, state="STOPPED", message="stubbed", server_url="http://localhost:8070",
    docker_available=False, container_name="grobid",
)
out = {}
with mock.patch.object(grobid_controller.GrobidController, "check_status", lambda self: status), \\
     mock.patch.object(grobid_controller.GrobidController, "is_alive", lambda self, *a, **k: False):
    at = AppTest.from_file(@APP@, default_timeout=120)
@BEFORE@
    at.run()
@AFTER@
    out.update({
        "exceptions": [e.value for e in at.exception],
        "tabs": [t.label for t in at.tabs],
        "sidebar_subheaders": [s.value for s in at.sidebar.subheader],
        "title": [t.value for t in at.title],
        "empty_corpus_notices": sum("No corpus yet" in i.value for i in at.info),
    })
print(json.dumps(out, default=str))
"""


def _indent(code):
    return textwrap.indent(textwrap.dedent(code).strip("\n"), "    ") if code.strip() else ""


def run_page(before="", after="", timeout=240):
    """Run app.py once and return the dict the subprocess printed."""
    with tempfile.TemporaryDirectory() as data:
        program = (
            TEMPLATE.replace("@ROOT@", repr(str(ROOT)))
            .replace("@DATA@", repr(data))
            .replace("@APP@", repr(str(ROOT / "app.py")))
            .replace("@BEFORE@", _indent(before))
            .replace("@AFTER@", _indent(after))
        )
        # The hooks run inside the `with` that stubs GROBID, so they are indented to match.
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, cwd=data, timeout=timeout)
    if result.returncode:
        raise AssertionError(f"rendering app.py failed:\n{result.stderr[-1800:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])
