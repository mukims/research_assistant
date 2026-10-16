"""app.py itself: the page lays out and every tab renders.

app.py is only the skeleton now (sidebar, tab bar, one call per tab), so a wrong
name or signature there, say render_draft_tab() losing its `chunks` argument, is
invisible to the tests of the modules it calls. Streamlit's AppTest runs the real
script headlessly.

It runs in a subprocess with a scratch data directory: executing app.py opens the
corpus store to count its chunks, and the real store must never be opened by a test.
GROBID's status is stubbed so no Docker or network call is made.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROGRAM = textwrap.dedent(
    """
    import json, os, sys
    from unittest import mock

    sys.path.insert(0, {root!r})
    os.environ["CITATION_DATA_DIR"] = {data!r}
    os.environ["CITATION_LOG_FILE"] = "0"
    for key in ("APP_PASSWORD", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(key, None)

    from streamlit.testing.v1 import AppTest
    from research_assistant import config
    from research_assistant.agents import grobid_controller

    assert os.path.realpath(config.DATA_DIR) == os.path.realpath({data!r}), config.DATA_DIR

    status = grobid_controller.GrobidStatus(
        is_alive=False, state="STOPPED", message="stubbed", server_url="http://localhost:8070",
        docker_available=False, container_name="grobid",
    )
    with mock.patch.object(grobid_controller.GrobidController, "check_status", lambda self: status), \\
         mock.patch.object(grobid_controller.GrobidController, "is_alive", lambda self, *a, **k: False):
        at = AppTest.from_file({app!r}, default_timeout=120).run()

    print(json.dumps({{
        "exceptions": [e.value for e in at.exception],
        "tabs": [t.label for t in at.tabs],
        "sidebar_subheaders": [s.value for s in at.sidebar.subheader],
        "title": [t.value for t in at.title],
        "empty_corpus_notices": sum("No corpus yet" in i.value for i in at.info),
    }}))
    """
)


@unittest.skipUnless(find_spec("streamlit"), "needs streamlit")
class TestAppRenders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as data:
            program = PROGRAM.format(root=str(ROOT), data=data, app=str(ROOT / "app.py"))
            result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, cwd=data, timeout=240)
        if result.returncode:
            raise AssertionError(f"rendering app.py failed:\n{result.stderr[-1500:]}")
        cls.page = json.loads(result.stdout.strip().splitlines()[-1])

    def test_the_page_renders_without_an_exception(self):
        self.assertEqual(self.page["exceptions"], [])

    def test_the_six_tabs_are_there_in_order(self):
        self.assertEqual(
            self.page["tabs"],
            ["Agent task planner", "Citation auditor", "Research idea", "Cite a draft", "Research chat", "How to use"],
        )

    def test_the_sidebar_has_its_four_sections(self):
        self.assertEqual(self.page["sidebar_subheaders"], ["Pipeline Status", "Corpus", "Backend", "Server & Services"])

    def test_the_title_is_shown(self):
        self.assertEqual(self.page["title"], ["🤖 Marvin the Citebot"])

    def test_the_corpus_size_reaches_the_tabs_that_need_it(self):
        # An empty corpus: the draft tab and the chat tab each say so. They get the count as an argument.
        self.assertEqual(self.page["empty_corpus_notices"], 2)


if __name__ == "__main__":
    unittest.main()
