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


def _openai_reply(text):
    msg = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=msg)],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _ollama_reply(text):
    return types.SimpleNamespace(
        message=types.SimpleNamespace(content=text),
        prompt_eval_count=1,
        eval_count=1,
    )


class TestChatTemperatureOpenAI(unittest.TestCase):
    def _run(self, **kwargs):
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_reply("ok")
        module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))
        with patch.dict("sys.modules", {"openai": module}), \
             patch.object(llm, "LLM_BACKEND", "openai"):
            llm.chat([{"role": "user", "content": "hi"}], **kwargs)
        return client.chat.completions.create.call_args.kwargs

    def test_temperature_absent_when_not_given(self):
        """The default call must stay byte-for-byte what it is today."""
        self.assertNotIn("temperature", self._run())

    def test_temperature_forwarded(self):
        self.assertEqual(self._run(temperature=0.0)["temperature"], 0.0)

    def test_options_are_ignored_by_openai(self):
        """ollama runtime options have no OpenAI equivalent."""
        self.assertNotIn("options", self._run(options={"num_ctx": 8192}))


class TestChatTemperatureOllama(unittest.TestCase):
    def _run(self, **kwargs):
        fake = MagicMock()
        fake.chat.return_value = _ollama_reply("ok")
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            llm.chat([{"role": "user", "content": "hi"}], **kwargs)
        return fake.chat.call_args.kwargs

    def test_options_absent_when_not_given(self):
        self.assertNotIn("options", self._run())

    def test_temperature_lands_inside_options(self):
        self.assertEqual(self._run(temperature=0.0)["options"], {"temperature": 0.0})

    def test_options_and_temperature_merge(self):
        merged = self._run(temperature=0.0, options={"num_ctx": 8192})["options"]
        self.assertEqual(merged, {"num_ctx": 8192, "temperature": 0.0})

    def test_caller_options_dict_is_not_mutated(self):
        """A module-level config constant must not grow a temperature key."""
        opts = {"num_ctx": 8192}
        fake = MagicMock()
        fake.chat.return_value = _ollama_reply("ok")
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            llm.chat([{"role": "user", "content": "hi"}], temperature=0.0, options=opts)
        self.assertEqual(opts, {"num_ctx": 8192})


class TestNomicPrefixes(unittest.TestCase):
    """nomic-embed-text expects task prefixes; v1 vectors were made without
    them, so the wrapper applies only under INDEX_VERSION >= 2."""

    def setUp(self):
        llm._embeddings_singleton = None
        self.addCleanup(setattr, llm, "_embeddings_singleton", None)
        self.calls = []
        inner = types.SimpleNamespace(
            embed_documents=lambda texts: (self.calls.append(("docs", list(texts))), [[0.0]] * len(texts))[1],
            embed_query=lambda text: (self.calls.append(("query", text)), [0.0])[1],
        )
        ollama_mod = types.ModuleType("langchain_ollama")
        ollama_mod.OllamaEmbeddings = lambda model: inner
        p = patch.dict("sys.modules", {"langchain_ollama": ollama_mod}); p.start(); self.addCleanup(p.stop)
        p = patch.object(llm, "EMBED_BACKEND", "ollama"); p.start(); self.addCleanup(p.stop)
        p = patch.object(llm, "EMBED_MODEL", "nomic-embed-text"); p.start(); self.addCleanup(p.stop)

    def test_v2_prefixes_documents_and_queries(self):
        with patch.object(llm, "INDEX_VERSION", 2):
            emb = llm.get_embeddings()
            emb.embed_documents(["a", "b"])
            emb.embed_query("q")
        self.assertEqual(self.calls, [("docs", ["search_document: a", "search_document: b"]),
                                      ("query", "search_query: q")])

    def test_v1_is_unprefixed(self):
        with patch.object(llm, "INDEX_VERSION", 1):
            emb = llm.get_embeddings()
            emb.embed_documents(["a"])
            emb.embed_query("q")
        self.assertEqual(self.calls, [("docs", ["a"]), ("query", "q")])

    def test_non_nomic_model_is_unprefixed_even_on_v2(self):
        with patch.object(llm, "INDEX_VERSION", 2), patch.object(llm, "EMBED_MODEL", "mxbai-embed-large"):
            emb = llm.get_embeddings()
            emb.embed_query("q")
        self.assertEqual(self.calls, [("query", "q")])


if __name__ == "__main__":
    unittest.main()
