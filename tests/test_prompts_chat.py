# tests/test_prompts_chat.py
import unittest

from research_assistant import prompts as p
from research_assistant import config as c


class TestChatPrompts(unittest.TestCase):
    def test_placeholders(self):
        self.assertIn("{question}", p.CHAT_TURN_USER)
        self.assertIn("{context}", p.CHAT_TURN_USER)
        self.assertIn("{question}", p.CHAT_TURN_NO_CONTEXT)
        self.assertIn("{history}", p.CHAT_CONDENSE_USER)
        self.assertIn("{question}", p.CHAT_CONDENSE_USER)
        self.assertIn("{memory}", p.CHAT_MEMORY_USER)
        self.assertIn("{turns}", p.CHAT_MEMORY_USER)
        self.assertIn("[S1]", p.RESEARCH_CHAT_SYSTEM)
        self.assertIn("💡 Suggested Next Questions", p.RESEARCH_CHAT_SYSTEM)

    def test_brain_lens_instructions(self):
        self.assertIn("explore", p.BRAIN_LENS_INSTRUCTIONS)
        self.assertIn("gaps", p.BRAIN_LENS_INSTRUCTIONS)
        self.assertIn("contradictions", p.BRAIN_LENS_INSTRUCTIONS)
        self.assertIn("hypotheses", p.BRAIN_LENS_INSTRUCTIONS)
        self.assertIn("methodology", p.BRAIN_LENS_INSTRUCTIONS)

    def test_config_defaults(self):
        self.assertTrue(c.CHAT_CONDENSE)
        self.assertEqual(
            (c.CHAT_CONTEXT_MAX_CHARS, c.CHAT_ANSWER_RESERVE_TOKENS, c.CHAT_PER_DOC_CAP, c.CHAT_FOCUS_TOP_K),
            (7000, 700, 3, 2),
        )
        self.assertAlmostEqual(c.CHAT_TEMPERATURE, 0.3)
        self.assertEqual(c.CHAT_OLLAMA_OPTIONS["num_ctx"], 4096)


if __name__ == "__main__":
    unittest.main()
