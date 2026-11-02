"""First-run setup checks: what is missing, and the command that fixes it.

run_checks() takes its network and store probes as parameters, so none of these tests
touch a socket, a GROBID or a vector store.
"""

import os
import tempfile
import unittest
from unittest import mock

import requests

from research_assistant import config
from research_assistant.shared.preflight import Check, run_checks
from tests.app_harness import HAVE_STREAMLIT, run_page


class _Resp:
    def __init__(self, tags, status_code=200):
        self.status_code = status_code
        self._tags = tags

    def json(self):
        return {"models": [{"name": n} for n in self._tags]}


def _http_with(*tags):
    return lambda url, timeout=None, **kw: _Resp(tags)


def _raises(url, timeout=None, **kw):
    raise requests.ConnectionError("refused")


def _by_name(checks):
    return {c.name: c for c in checks}


class TestRunChecks(unittest.TestCase):
    def setUp(self):
        data = tempfile.TemporaryDirectory()
        self.addCleanup(data.cleanup)
        for target, value in (
            ("LLM_BACKEND", "ollama"),
            ("EMBED_BACKEND", "ollama"),
            ("CHAT_MODEL", "gemma4:e2b"),
            ("EMBED_MODEL", "embeddinggemma-2"),
            ("OPENAI_API_KEY", None),
            ("DATA_DIR", data.name),
            ("UNPAYWALL_EMAIL", "me@example.edu"),
        ):
            patcher = mock.patch.object(config, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self, http_get=None, grobid_alive=lambda: True, open_store=lambda: object()):
        return run_checks(
            http_get=http_get or _http_with("gemma4:e2b", "embeddinggemma-2:latest"),
            grobid_alive=grobid_alive,
            open_store=open_store,
        )

    def test_all_green(self):
        checks = self._run()
        self.assertEqual(
            [c.name for c in checks],
            ["Model backend", "Chat model", "Embedding model", "Vector store", "Data folder", "GROBID", "Contact email"],
        )
        self.assertTrue(all(c.ok for c in checks), [c for c in checks if not c.ok])

    def test_missing_chat_model_says_what_to_pull(self):
        checks = _by_name(self._run(http_get=_http_with("embeddinggemma-2:latest")))
        self.assertFalse(checks["Chat model"].ok)
        self.assertEqual(checks["Chat model"].fix, "ollama pull gemma4:e2b")
        self.assertTrue(checks["Chat model"].blocking)
        self.assertTrue(checks["Embedding model"].ok)

    def test_untagged_name_matches_latest_and_any_tag(self):
        with mock.patch.object(config, "EMBED_MODEL", "embeddinggemma-2"):
            self.assertTrue(_by_name(self._run(http_get=_http_with("gemma4:e2b", "embeddinggemma-2:300m")))["Embedding model"].ok)
        # A tagged name must match exactly: gemma4:e2b is not gemma4:e4b.
        self.assertFalse(_by_name(self._run(http_get=_http_with("gemma4:e4b", "embeddinggemma-2")))["Chat model"].ok)
        # A different model that merely shares a prefix does not match.
        self.assertFalse(_by_name(self._run(http_get=_http_with("gemma4:e2b", "embeddinggemma-22:latest")))["Embedding model"].ok)

    def test_unreachable_backend_omits_model_checks(self):
        checks = _by_name(self._run(http_get=_raises))
        self.assertFalse(checks["Model backend"].ok)
        self.assertIn("refused", checks["Model backend"].detail)
        self.assertEqual(checks["Model backend"].fix, "Start Ollama (ollama serve) or set GEMINI_API_KEY in .env")
        self.assertNotIn("Chat model", checks)
        self.assertNotIn("Embedding model", checks)

    def test_a_non_200_reply_is_an_unreachable_backend(self):
        checks = _by_name(self._run(http_get=lambda url, timeout=None, **kw: _Resp([], status_code=503)))
        self.assertFalse(checks["Model backend"].ok)
        self.assertNotIn("Chat model", checks)

    def test_http_get_is_asked_for_api_tags_within_three_seconds(self):
        calls = []

        def spy(url, timeout=None, **kw):
            calls.append((url, timeout))
            return _Resp(["gemma4:e2b", "embeddinggemma-2"])

        with mock.patch("research_assistant.shared.llm._ollama_host", return_value="http://h:1"):
            self._run(http_get=spy)
        self.assertEqual(calls, [("http://h:1/api/tags", 3)])

    def test_openai_backend_needs_a_key_and_skips_the_ollama_model_checks(self):
        with mock.patch.object(config, "LLM_BACKEND", "openai"), mock.patch.object(config, "EMBED_BACKEND", "openai"):
            with mock.patch.object(config, "OPENAI_API_KEY", None):
                checks = _by_name(self._run(http_get=_raises))
                self.assertFalse(checks["Model backend"].ok)
            with mock.patch.object(config, "OPENAI_API_KEY", "sk-test"):
                checks = _by_name(self._run(http_get=_raises))
                self.assertTrue(checks["Model backend"].ok)
        self.assertNotIn("Chat model", checks)
        self.assertNotIn("Embedding model", checks)

    def test_hosted_chat_still_needs_ollama_for_embeddings(self):
        with mock.patch.object(config, "LLM_BACKEND", "openai"), mock.patch.object(config, "OPENAI_API_KEY", "sk-test"):
            checks = _by_name(self._run(http_get=_raises))
        self.assertTrue(checks["Model backend"].ok)
        self.assertNotIn("Chat model", checks)
        embed = checks["Embedding model"]
        self.assertFalse(embed.ok)
        self.assertTrue(embed.blocking)
        self.assertIn("refused", embed.detail)
        self.assertEqual(embed.fix, "Start Ollama (ollama serve): embeddings run on Ollama even with a hosted chat model")

    def test_hosted_chat_reports_a_missing_embedding_model(self):
        with mock.patch.object(config, "LLM_BACKEND", "openai"), mock.patch.object(config, "OPENAI_API_KEY", "sk-test"):
            checks = _by_name(self._run(http_get=_http_with("gemma4:e2b")))
        self.assertNotIn("Chat model", checks)
        self.assertFalse(checks["Embedding model"].ok)
        self.assertEqual(checks["Embedding model"].fix, "ollama pull embeddinggemma-2")

    def test_an_import_failure_in_the_probe_is_a_failed_check(self):
        from research_assistant import shared

        # `from package import submodule` reads the package attribute first, so drop it as
        # well as blocking the module, or the import still succeeds.
        self.addCleanup(setattr, shared, "llm", shared.llm)
        del shared.llm
        with mock.patch.dict("sys.modules", {"research_assistant.shared.llm": None}):
            checks = _by_name(self._run())
        self.assertFalse(checks["Model backend"].ok)

    def test_a_remote_embedder_skips_only_the_embedding_check(self):
        with mock.patch.object(config, "EMBED_BACKEND", "openai"):
            checks = _by_name(self._run())
        self.assertIn("Chat model", checks)
        self.assertNotIn("Embedding model", checks)

    def test_vector_store_failure_reports_the_exception(self):
        def broken():
            raise ImportError("No module named 'chromadb'")

        checks = _by_name(self._run(open_store=broken))
        self.assertFalse(checks["Vector store"].ok)
        self.assertTrue(checks["Vector store"].blocking)
        self.assertIn("chromadb", checks["Vector store"].detail)
        self.assertEqual(checks["Vector store"].fix, "deploy/local.sh up")

    def test_data_folder_must_be_writable(self):
        with tempfile.TemporaryDirectory() as parent:
            missing = os.path.join(parent, "not", "yet")
            with mock.patch.object(config, "DATA_DIR", missing):
                self.assertTrue(_by_name(self._run())["Data folder"].ok)
                self.assertTrue(os.path.isdir(missing))
            with mock.patch.object(config, "DATA_DIR", parent), mock.patch("os.access", return_value=False):
                check = _by_name(self._run())["Data folder"]
            self.assertFalse(check.ok)
            self.assertTrue(check.blocking)
            self.assertEqual(check.fix, f"Make {parent} writable or set CITATION_DATA_DIR")

    def test_data_folder_that_cannot_be_created_is_a_failed_check(self):
        with mock.patch("os.makedirs", side_effect=PermissionError("denied")):
            with mock.patch.object(config, "DATA_DIR", "/nonexistent/x"):
                check = _by_name(self._run())["Data folder"]
        self.assertFalse(check.ok)
        self.assertIn("denied", check.detail)

    def test_grobid_down_is_not_blocking(self):
        checks = _by_name(self._run(grobid_alive=lambda: False))
        grobid = checks["GROBID"]
        self.assertFalse(grobid.ok)
        self.assertFalse(grobid.blocking)
        self.assertIn("docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1", grobid.fix)
        self.assertIn("references fall back to regex", grobid.fix)

    def test_a_grobid_probe_that_raises_is_a_failed_check_not_a_crash(self):
        def boom():
            raise RuntimeError("docker exploded")

        grobid = _by_name(self._run(grobid_alive=boom))["GROBID"]
        self.assertFalse(grobid.ok)
        self.assertIn("docker exploded", grobid.detail)

    def test_missing_contact_email_is_not_blocking(self):
        with mock.patch.object(config, "UNPAYWALL_EMAIL", None):
            email = _by_name(self._run())["Contact email"]
        self.assertFalse(email.ok)
        self.assertFalse(email.blocking)
        self.assertEqual(email.fix, "export UNPAYWALL_EMAIL=you@university.edu")

    def test_check_is_frozen_and_blocking_by_default(self):
        check = Check("x", True, "d")
        self.assertTrue(check.blocking)
        self.assertEqual(check.fix, "")
        with self.assertRaises(Exception):
            check.ok = False


@unittest.skipUnless(HAVE_STREAMLIT, "needs streamlit")
class TestPreflightInThePage(unittest.TestCase):
    def test_a_dead_backend_shows_the_banner_and_the_sidebar_expander(self):
        # The test environment points OLLAMA_HOST at a closed port, so Model backend fails.
        page = run_page(
            after="""
            out["errors"] = [e.value for e in at.error]
            out["sidebar_expanders"] = [e.label for e in at.sidebar.expander]
            """
        )
        self.assertEqual(page["exceptions"], [])
        banner = [e for e in page["errors"] if e.startswith("Marvin can't run yet:")]
        self.assertEqual(len(banner), 1)
        self.assertIn("- **Model backend**: Start Ollama (ollama serve) or set GEMINI_API_KEY in .env", banner[0])
        self.assertNotIn("GROBID", banner[0])  # non-blocking checks stay out of the banner
        self.assertIn("Setup checks", page["sidebar_expanders"])


if __name__ == "__main__":
    unittest.main()
