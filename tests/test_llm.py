"""Tests for the backend-agnostic LLM layer.

chat_stream() exists so agent 7 can stream without importing a provider SDK.
Both backends must yield plain string deltas and skip the empty frames each
API emits, so agent code sees one contract.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import research_assistant.shared.llm as llm


def _ollama_chunk(text):
    return types.SimpleNamespace(message=types.SimpleNamespace(content=text))


def _openai_chunk(text):
    delta = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


class TestChatStreamOllama(unittest.TestCase):
    def test_yields_content_deltas_in_order(self):
        fake = MagicMock()
        fake.chat.return_value = iter(
            [_ollama_chunk("Hel"), _ollama_chunk("lo"), _ollama_chunk("!")]
        )
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "hi"}])),
                ["Hel", "lo", "!"],
            )

    def test_empty_deltas_are_skipped(self):
        fake = MagicMock()
        fake.chat.return_value = iter(
            [_ollama_chunk(""), _ollama_chunk("x"), _ollama_chunk(None)]
        )
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "hi"}])), ["x"]
            )

    def test_options_are_forwarded(self):
        """CHAT_OLLAMA_OPTIONS carries flash-attention / KV-cache tuning."""
        fake = MagicMock()
        fake.chat.return_value = iter([_ollama_chunk("x")])
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            list(llm.chat_stream([{"role": "user", "content": "hi"}],
                                 options={"num_ctx": 4096}))
        self.assertEqual(fake.chat.call_args.kwargs["options"], {"num_ctx": 4096})


class TestChatStreamOpenAI(unittest.TestCase):
    def test_yields_content_deltas_and_skips_role_only_frames(self):
        stream = [_openai_chunk(None), _openai_chunk("Hi"), _openai_chunk(None)]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(stream)
        module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))
        with patch.dict("sys.modules", {"openai": module}), \
             patch.object(llm, "LLM_BACKEND", "openai"):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "hi"}])), ["Hi"]
            )

    def test_chunk_with_no_choices_is_skipped(self):
        """Some gateways emit a usage-only terminal frame."""
        stream = [types.SimpleNamespace(choices=[]), _openai_chunk("ok")]
        client = MagicMock()
        client.chat.completions.create.return_value = iter(stream)
        module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))
        with patch.dict("sys.modules", {"openai": module}), \
             patch.object(llm, "LLM_BACKEND", "openai"):
            self.assertEqual(
                list(llm.chat_stream([{"role": "user", "content": "hi"}])), ["ok"]
            )


class TestUnknownBackend(unittest.TestCase):
    def test_raises_eagerly_not_on_first_next(self):
        """A bad backend must fail at the call, not when iteration starts."""
        with patch.object(llm, "LLM_BACKEND", "nope"):
            with self.assertRaises(ValueError):
                llm.chat_stream([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
