import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

for mod in [
    "langgraph",
    "langgraph.checkpoint",
    "langgraph.checkpoint.memory",
    "langgraph.graph",
]:
    if mod not in sys.modules:
        m = MagicMock()
        m.START = "START"
        m.END = "END"
        sys.modules[mod] = m

if "streamlit" not in sys.modules:
    st_mock = MagicMock()
    st_mock.columns.return_value = [MagicMock(), MagicMock()]
    st_mock.tabs.return_value = [MagicMock()] * 5
    st_mock.sidebar = MagicMock()
    st_mock.session_state = {}
    st_mock.form_submit_button.return_value = False
    st_mock.button.return_value = False
    st_mock.file_uploader.return_value = None
    st_mock.cache_data = lambda *a, **kw: (lambda f: f)
    sys.modules["streamlit"] = st_mock

import research_assistant.agents.agent0_discoverer as a0
import orchestrate


class TestOrchestrateSeedFileIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_dir = os.path.join(self.tmp.name, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.raw_dir_patch = patch.object(a0, "RAW_DIR", self.raw_dir)
        self.seed_path_patch = patch.object(
            a0, "SEED_PAPERS_PATH", os.path.join(self.tmp.name, "seed_papers.json")
        )
        self.raw_dir_patch.start()
        self.seed_path_patch.start()
        self.addCleanup(self.raw_dir_patch.stop)
        self.addCleanup(self.seed_path_patch.stop)

    def test_discover_node_with_seed_file_and_no_query(self):
        pdf_path = os.path.join(self.tmp.name, "quantum_hall_effect.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 mock pdf content with /Title (Quantum Hall Effect)")

        state = {
            "query": "",
            "workers": 1,
            "force": False,
            "ask": True,
            "seed_file": pdf_path,
            "seed_url": None,
        }
        res = orchestrate.discover(state)
        self.assertIn("seed_path", res)
        self.assertTrue(os.path.exists(res["seed_path"]))
        self.assertEqual(res["seed_label"], "Quantum Hall Effect")
        # Inferred query was added to state
        self.assertEqual(res.get("query"), "Quantum Hall Effect")
        self.assertNotIn("stopped", res)

    def test_discover_node_with_seed_file_and_custom_query(self):
        pdf_path = os.path.join(self.tmp.name, "hall.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 mock pdf content")

        state = {
            "query": "fractional quantum hall states",
            "workers": 1,
            "force": False,
            "ask": True,
            "seed_file": pdf_path,
        }
        res = orchestrate.discover(state)
        self.assertIn("seed_path", res)
        self.assertTrue(os.path.exists(res["seed_path"]))
        # Query unchanged since user provided one
        self.assertNotIn("query", res)  # Only updated when state["query"] was empty
        self.assertNotIn("stopped", res)

    def test_discover_node_with_invalid_seed_file(self):
        fake_pdf = os.path.join(self.tmp.name, "fake.pdf")
        with open(fake_pdf, "wb") as f:
            f.write(b"not a pdf at all")

        state = {
            "query": "",
            "workers": 1,
            "force": False,
            "ask": True,
            "seed_file": fake_pdf,
        }
        res = orchestrate.discover(state)
        self.assertIsNone(res["seed_path"])
        self.assertIn("stopped", res)
        self.assertIn("could not load seed PDF", res["stopped"])

    def test_after_discover_routing(self):
        state_with_seed = {"seed_path": "/path/to/seed.pdf", "ask": True}
        self.assertEqual(orchestrate._after_discover(state_with_seed), "ingest_seed")

        # Fallback only when a real query exists
        state_stopped_with_query = {"seed_path": None, "ask": True, "query": "quantum wires"}
        self.assertEqual(orchestrate._after_discover(state_stopped_with_query), "fallback")

        # Empty or missing query routes to END, not fallback
        state_stopped_empty_query = {"seed_path": None, "ask": True, "query": ""}
        self.assertEqual(orchestrate._after_discover(state_stopped_empty_query), orchestrate.END)

        state_stopped_no_query = {"seed_path": None, "ask": True}
        self.assertEqual(orchestrate._after_discover(state_stopped_no_query), orchestrate.END)

        state_stopped_no_ask = {"seed_path": None, "ask": False, "query": "quantum wires"}
        self.assertEqual(orchestrate._after_discover(state_stopped_no_ask), orchestrate.END)


class TestAppSeedRender(unittest.TestCase):
    def test_render_seed_and_downloads_skips_when_failed(self):
        import app
        # Mock streamlit container and markdown calls
        with patch.object(app, "_manifest", return_value={"old query": {"title": "Old Paper", "path": "/p.pdf"}}):
            with patch("streamlit.container") as mock_container:
                # When run stopped before finding seed
                final = {"seed_path": None, "stopped": "could not load seed PDF"}
                app._render_seed_and_downloads("", final)
                mock_container.assert_not_called()

    def test_render_seed_and_downloads_shows_seed_when_successful(self):
        import app
        with patch.object(
            app,
            "_manifest",
            side_effect=[
                {"uploaded": {"title": "New Uploaded Paper", "path": "/path/raw/paper.pdf", "source": "upload"}},
                {},  # downloaded
                {},  # failed
            ],
        ):
            with patch("streamlit.container") as mock_container:
                with patch("streamlit.markdown") as mock_markdown:
                    with patch("streamlit.caption"):
                        final = {"seed_path": "/path/raw/paper.pdf", "seed_label": "New Uploaded Paper"}
                        app._render_seed_and_downloads("uploaded", final)
                        mock_container.assert_called()
                        # Verify markdown rendered seed paper title
                        rendered = [call.args[0] for call in mock_markdown.call_args_list if call.args]
                        self.assertTrue(any("New Uploaded Paper" in r for r in rendered))


class TestUploadStaging(unittest.TestCase):
    """An uploaded PDF must not accumulate a second copy under DATA_DIR.

    discover_from_file already writes the paper into RAW_DIR under a
    key-derived name, so any staging copy is pure duplication — it exists only
    because the graph's state carries a path rather than bytes. It previously
    landed in DATA_DIR/uploads under the user's own filename and was never
    cleaned up, so the directory grew without bound and re-uploading a
    different paper with the same filename silently overwrote the earlier one.
    """

    def _fake_upload(self, body: bytes, name: str = "paper.pdf"):
        uploaded = MagicMock()
        uploaded.getbuffer.return_value = body
        uploaded.name = name
        return uploaded

    def _stage(self, uploaded):
        import app

        path = app._stage_upload(uploaded)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_staged_upload_holds_the_bytes_and_lives_outside_the_data_dir(self):
        from research_assistant import config

        path = self._stage(self._fake_upload(b"%PDF-1.4 body"))

        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), b"%PDF-1.4 body")
        self.assertFalse(
            os.path.abspath(path).startswith(os.path.abspath(config.DATA_DIR)),
            "a staging copy must not accumulate inside DATA_DIR",
        )

    def test_two_uploads_sharing_a_filename_do_not_overwrite_each_other(self):
        first = self._stage(self._fake_upload(b"%PDF-1.4 first", "paper.pdf"))
        second = self._stage(self._fake_upload(b"%PDF-1.4 second", "paper.pdf"))

        self.assertNotEqual(first, second)
        with open(first, "rb") as fh:
            self.assertEqual(fh.read(), b"%PDF-1.4 first")


if __name__ == "__main__":
    unittest.main()
