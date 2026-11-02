"""One source of truth for default models, the contact address and the User-Agent.

The code once defaulted to "gemma4:e2b-mlx" while every doc told users to
`ollama pull gemma4:e2b`; outbound requests named another project's repository in
their User-Agent; and an unconfigured install sent a placeholder email address to
Crossref, OpenAlex and Unpaywall. These tests hold the line.
"""

import glob
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from research_assistant import config

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "research_assistant"


class TestDefaults(unittest.TestCase):
    def test_docs_pull_only_the_default_models(self):
        paths = [ROOT / name for name in ("README.md", "HOW_TO_USE.md", "PIPELINE.md", "deploy/README.md")]
        paths += [Path(p) for p in glob.glob(str(ROOT / "deploy" / "*.sh"))]
        pulled = set()
        for path in paths:
            text = path.read_text(encoding="utf-8")
            # A script may pull a model it names in a variable (MODEL="embeddinggemma-2").
            variables = dict(re.findall(r'^([A-Z_]+)="([^"$]*)"', text, re.M))
            for name, value in variables.items():
                text = text.replace(f'"${name}"', value).replace(f"${name}", value)
            # Markdown wraps commands in backticks and ends sentences with punctuation.
            pulled.update(m.strip("\"'").rstrip("`.,;:)") for m in re.findall(r"ollama pull (\S+)", text))
        self.assertTrue(pulled, "the docs should tell users which models to pull")
        self.assertLessEqual(pulled, {config.DEFAULT_OLLAMA_MODEL, config.DEFAULT_EMBED_MODEL})

    def test_no_model_literals_outside_config(self):
        pattern = re.compile(r"[\"']gemma4:|[\"']gemini-\d")
        offenders = []
        for path in sorted(PACKAGE.rglob("*.py")):
            if path == PACKAGE / "config.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
        self.assertEqual(offenders, [])

    def test_user_agent_names_this_project(self):
        self.assertIn("mukims/research_assistant", config.USER_AGENT)
        self.assertNotIn("deepmind", config.USER_AGENT)
        offenders = [
            str(path.relative_to(ROOT))
            for path in PACKAGE.rglob("*.py")
            if "abcdef_12345" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def _user_agent_in_fresh_interpreter(self, email):
        # USER_AGENT is computed at import; reloading config in-process would leak
        # constants into other modules, so ask a subprocess with no .env.
        env = {**os.environ, "CITATION_DISABLE_DOTENV": "1"}
        env.pop("UNPAYWALL_EMAIL", None)
        if email is not None:
            env["UNPAYWALL_EMAIL"] = email
        out = subprocess.run(
            [sys.executable, "-c", "from research_assistant import config; print(config.USER_AGENT)"],
            env=env, cwd=str(ROOT), capture_output=True, text=True, timeout=120, check=True,
        )
        return out.stdout.strip()

    def test_user_agent_carries_the_contact_only_when_set(self):
        base = "research_assistant/0.2 (+https://github.com/mukims/research_assistant"
        self.assertEqual(self._user_agent_in_fresh_interpreter(None), base + ")")
        self.assertEqual(
            self._user_agent_in_fresh_interpreter("you@example.org"),
            base + "; mailto:you@example.org)",
        )

    def test_every_module_sends_the_one_user_agent(self):
        from research_assistant.shared import academic_apis, fetch

        self.assertEqual(fetch.HEADERS["User-Agent"], config.USER_AGENT)
        self.assertEqual(academic_apis.DEFAULT_HEADERS["User-Agent"], config.USER_AGENT)

    def test_prior_art_requests_send_the_one_user_agent(self):
        from research_assistant.transfer import prior_art

        response = MagicMock()
        response.json.return_value = {}
        response.text = ""
        with patch("requests.get", return_value=response) as get:
            prior_art.default_get_json("https://api.openalex.org/works", {})
            prior_art.default_get_text("https://export.arxiv.org/api/query", {})
        self.assertEqual(get.call_count, 2)
        for call in get.call_args_list:
            self.assertEqual(call.kwargs["headers"], {"User-Agent": config.USER_AGENT})

    def test_unset_contact_skips_unpaywall(self):
        from research_assistant.agents import agent2_fetcher

        with patch.object(agent2_fetcher, "UNPAYWALL_EMAIL", None), \
             patch.object(agent2_fetcher.requests, "get") as get:
            result = agent2_fetcher.try_unpaywall("10.1000/xyz", "/tmp/never-written.pdf")
        self.assertEqual(result, (False, "Unpaywall skipped: set UNPAYWALL_EMAIL"))
        get.assert_not_called()

    def test_openalex_omits_mailto_when_unset(self):
        from research_assistant.agents import agent0_discoverer

        response = MagicMock()
        response.json.return_value = {"results": []}
        with patch.object(agent0_discoverer, "OPENALEX_MAILTO", None), \
             patch.object(agent0_discoverer.requests, "get", return_value=response) as get:
            agent0_discoverer._search_openalex("tensor networks", 5)
        self.assertNotIn("mailto", get.call_args.kwargs["params"])

    def test_openalex_sends_mailto_when_set(self):
        from research_assistant.agents import agent0_discoverer

        response = MagicMock()
        response.json.return_value = {"results": []}
        with patch.object(agent0_discoverer, "OPENALEX_MAILTO", "me@example.org"), \
             patch.object(agent0_discoverer.requests, "get", return_value=response) as get:
            agent0_discoverer._search_openalex("tensor networks", 5)
        self.assertEqual(get.call_args.kwargs["params"]["mailto"], "me@example.org")

    def test_crossref_omits_mailto_when_unset(self):
        from research_assistant.shared import academic_apis

        response = MagicMock(status_code=200)
        response.json.return_value = {}
        with patch.object(academic_apis, "_rate_limit"), \
             patch.object(academic_apis.requests, "get", return_value=response) as get:
            academic_apis.call_crossref_api("a title", mailto=None)
        self.assertNotIn("mailto", get.call_args.kwargs["params"])

    def test_openalex_api_url_omits_mailto_when_unset(self):
        from research_assistant.shared import academic_apis

        response = MagicMock(status_code=200)
        response.json.return_value = {}
        with patch.object(academic_apis, "_rate_limit"), \
             patch.object(academic_apis.requests, "get", return_value=response) as get:
            academic_apis.call_openalex_api("10.1000/xyz", mailto=None)
        self.assertNotIn("mailto", get.call_args.args[0])

    def test_unpaywall_lookup_skipped_without_contact(self):
        from research_assistant.shared import academic_apis

        with patch.object(academic_apis, "_rate_limit"), \
             patch.object(academic_apis.requests, "get") as get:
            self.assertIsNone(academic_apis.fetch_open_access_pdf_url("10.1000/xyz", mailto=None))
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
