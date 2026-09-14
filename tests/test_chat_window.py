"""Tests for CHAT_WINDOW_TOKENS configuration and Agent 7 window budgeting."""

import json
import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.agents import agent7_research_chat as a7

_PROBE = """
import json
try:
    import dotenv
    dotenv.load_dotenv = lambda *a, **kw: None   # the probe's env is the whole truth
except ImportError:                              # config.py tolerates a missing dotenv too
    pass
from research_assistant import config as c
print(json.dumps({
    "window": c.CHAT_WINDOW_TOKENS,
    "backend": c.LLM_BACKEND,
}))
"""


def _probe(**env):
    full = {k: v for k, v in os.environ.items()
            if k not in ("CITATION_CHAT_WINDOW_TOKENS", "LLM_BACKEND", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY")}
    full.update({"CITATION_LOG_FILE": "0", "GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", **env})
    out = subprocess.run([sys.executable, "-c", _PROBE], env=full, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


class TestChatWindowConfig(unittest.TestCase):
    def test_ollama_backend_defaults_to_4096(self):
        got = _probe(LLM_BACKEND="ollama")
        self.assertEqual(got["window"], 4096)

    def test_openai_backend_defaults_to_32768(self):
        got = _probe(LLM_BACKEND="openai", OPENAI_API_KEY="dummy")
        self.assertEqual(got["window"], 32768)

    def test_explicit_env_override_honored(self):
        got = _probe(CITATION_CHAT_WINDOW_TOKENS="16384")
        self.assertEqual(got["window"], 16384)


class TestAgent7ChatWindowBudgeting(unittest.TestCase):
    def setUp(self):
        self.fake_resources = (MagicMock(), MagicMock(), [], [])

    def test_large_window_retains_more_turns_without_folding(self):
        agent = a7.ResearchChat(search_resources=self.fake_resources)
        agent.history = [
            {"role": "user", "content": f"Turn {i} " * 50}
            for i in range(10)
        ]
        with patch.object(a7, "CHAT_WINDOW_TOKENS", 32768), patch.object(a7, "CHAT_OLLAMA_OPTIONS", {"num_ctx": 4096}):
            kept, folded = agent._fit("New user question")
            self.assertEqual(folded, 0)
            self.assertEqual(len(kept), 10)

    def test_small_window_triggers_folding(self):
        agent = a7.ResearchChat(search_resources=self.fake_resources)
        agent.history = [
            {"role": "user", "content": f"Turn {i} " * 50}
            for i in range(10)
        ]
        with patch.object(a7, "CHAT_WINDOW_TOKENS", 500), patch.object(a7, "CHAT_OLLAMA_OPTIONS", {"num_ctx": 4096}):
            kept, folded = agent._fit("New user question")
            self.assertGreater(folded, 0)
            self.assertLess(len(kept), 10)
