"""Challenger test suite for Milestone 4: Hardware Guardrails & Test Environment Alignment.

Empirically verifies:
1. chat() and _ollama_chat() clamp num_thread: 16 to 1 (and excessive threads > 2).
2. options={"num_ctx": 4096} is set by default when calling chat().
3. torch.get_num_threads() == 1 and os.environ["OMP_NUM_THREADS"] == "1".
4. TASK_OLLAMA_OPTIONS["num_ctx"] == 4096.
5. ChromaDB chunk count is exactly 75,039.
6. Additional stress tests for TaskRunner persistent storage safety and subprocess thread limits.
"""

import os
import types
import unittest
from unittest.mock import MagicMock, patch

import research_assistant.config as config
import research_assistant.shared.llm as llm


def _ollama_reply(text="ok"):
    return types.SimpleNamespace(
        message=types.SimpleNamespace(content=text),
        prompt_eval_count=10,
        eval_count=10,
    )


def _ollama_chunk(text="ok"):
    return types.SimpleNamespace(message=types.SimpleNamespace(content=text))


class TestChallengerM4HardwareGuardrails(unittest.TestCase):
    """Stress-tests for Milestone 4 hardware CPU guardrails and invariants."""

    # ─────────────────────────────────────────────────────────────────────────
    # 1. num_thread Clamping
    # ─────────────────────────────────────────────────────────────────────────

    def test_chat_clamps_num_thread_16_to_1(self):
        """chat() with options={'num_thread': 16} must clamp num_thread down to 1."""
        fake = MagicMock()
        fake.chat.return_value = _ollama_reply()
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            llm.chat(
                [{"role": "user", "content": "test"}],
                options={"num_thread": 16},
                backend="ollama",
            )
        self.assertIn("options", fake.chat.call_args.kwargs)
        sent_opts = fake.chat.call_args.kwargs["options"]
        self.assertEqual(sent_opts.get("num_thread"), 1)
        self.assertEqual(sent_opts.get("num_ctx"), 4096)

    def test_ollama_chat_clamps_num_thread_16_to_1(self):
        """_ollama_chat() directly called with options={'num_thread': 16} must clamp to 1."""
        fake = MagicMock()
        fake.chat.return_value = _ollama_reply()
        with patch.dict("sys.modules", {"ollama": fake}):
            llm._ollama_chat(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
                images=None,
                options={"num_thread": 16},
            )
        self.assertIn("options", fake.chat.call_args.kwargs)
        sent_opts = fake.chat.call_args.kwargs["options"]
        self.assertEqual(sent_opts.get("num_thread"), 1)
        self.assertEqual(sent_opts.get("num_ctx"), 4096)

    def test_chat_clamps_adversarial_thread_counts(self):
        """Excessive thread counts (4, 32, 64) must all clamp to 1, while safe bounds (1, 2) stay untouched."""
        for excessive in [4, 32, 64]:
            fake = MagicMock()
            fake.chat.return_value = _ollama_reply()
            with patch.dict("sys.modules", {"ollama": fake}), \
                 patch.object(llm, "LLM_BACKEND", "ollama"):
                llm.chat([{"role": "user", "content": "test"}], options={"num_thread": excessive}, backend="ollama")
            self.assertEqual(
                fake.chat.call_args.kwargs["options"]["num_thread"],
                1,
                f"Failed to clamp num_thread={excessive} to 1",
            )

        for safe in [1, 2]:
            fake = MagicMock()
            fake.chat.return_value = _ollama_reply()
            with patch.dict("sys.modules", {"ollama": fake}), \
                 patch.object(llm, "LLM_BACKEND", "ollama"):
                llm.chat([{"role": "user", "content": "test"}], options={"num_thread": safe}, backend="ollama")
            self.assertEqual(
                fake.chat.call_args.kwargs["options"]["num_thread"],
                safe,
                f"Safe thread count {safe} should not be mutated",
            )

    def test_chat_stream_clamps_num_thread_16_to_1(self):
        """chat_stream() with options={'num_thread': 16} must also clamp to 1."""
        fake = MagicMock()
        fake.chat.return_value = iter([_ollama_chunk("chunk")])
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            list(llm.chat_stream([{"role": "user", "content": "test"}], options={"num_thread": 16}, backend="ollama"))
        self.assertIn("options", fake.chat.call_args.kwargs)
        self.assertEqual(fake.chat.call_args.kwargs["options"].get("num_thread"), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Default num_ctx: 4096 on chat()
    # ─────────────────────────────────────────────────────────────────────────

    def test_default_options_num_ctx_4096_when_calling_chat(self):
        """Calling chat() without explicit options must default options to num_ctx=4096."""
        fake = MagicMock()
        fake.chat.return_value = _ollama_reply()
        with patch.dict("sys.modules", {"ollama": fake}), \
             patch.object(llm, "LLM_BACKEND", "ollama"):
            llm.chat([{"role": "user", "content": "hello"}], backend="ollama")
        self.assertIn(
            "options",
            fake.chat.call_args.kwargs,
            "options was not passed to ollama.chat when chat() called without options",
        )
        sent_opts = fake.chat.call_args.kwargs["options"]
        self.assertIsNotNone(sent_opts, "options dictionary sent to ollama.chat must not be None")
        self.assertEqual(
            sent_opts.get("num_ctx"),
            4096,
            f"Expected num_ctx: 4096 by default, got: {sent_opts.get('num_ctx')}",
        )

    def test_default_options_num_ctx_4096_when_calling_ollama_chat(self):
        """Calling _ollama_chat() directly without options must also default options to num_ctx=4096."""
        fake = MagicMock()
        fake.chat.return_value = _ollama_reply()
        with patch.dict("sys.modules", {"ollama": fake}):
            llm._ollama_chat(
                messages=[{"role": "user", "content": "hello"}],
                model="test-model",
                images=None,
            )
        self.assertIn(
            "options",
            fake.chat.call_args.kwargs,
            "options was not passed to ollama.chat when _ollama_chat() called without options",
        )
        sent_opts = fake.chat.call_args.kwargs["options"]
        self.assertIsNotNone(sent_opts, "options dictionary sent to ollama.chat must not be None")
        self.assertEqual(
            sent_opts.get("num_ctx"),
            4096,
            f"Expected num_ctx: 4096 by default, got: {sent_opts.get('num_ctx')}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Torch and OMP Thread Limits
    # ─────────────────────────────────────────────────────────────────────────

    def test_torch_get_num_threads_is_one(self):
        """torch.get_num_threads() must be clamped to 1 to prevent CPU starvation."""
        import torch
        self.assertEqual(
            torch.get_num_threads(),
            1,
            f"torch.get_num_threads() is {torch.get_num_threads()}, expected 1",
        )

    def test_omp_num_threads_env_var_is_one(self):
        """os.environ['OMP_NUM_THREADS'] must be set to '1'."""
        self.assertEqual(
            os.environ.get("OMP_NUM_THREADS"),
            "1",
            f"OMP_NUM_THREADS is {os.environ.get('OMP_NUM_THREADS')!r}, expected '1'",
        )

    def test_all_blas_env_vars_set_to_one(self):
        """All critical BLAS/math backend thread environment variables must be '1'."""
        vars_to_check = [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TORCH_NUM_THREADS",
        ]
        for var in vars_to_check:
            self.assertEqual(
                os.environ.get(var),
                "1",
                f"Expected os.environ['{var}'] == '1', got {os.environ.get(var)!r}",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. TASK_OLLAMA_OPTIONS Configuration
    # ─────────────────────────────────────────────────────────────────────────

    def test_task_ollama_options_num_ctx_4096(self):
        """TASK_OLLAMA_OPTIONS in config.py must define num_ctx=4096."""
        self.assertTrue(
            hasattr(config, "TASK_OLLAMA_OPTIONS"),
            "config module missing TASK_OLLAMA_OPTIONS constant",
        )
        self.assertIsInstance(config.TASK_OLLAMA_OPTIONS, dict)
        self.assertEqual(
            config.TASK_OLLAMA_OPTIONS.get("num_ctx"),
            4096,
            f"TASK_OLLAMA_OPTIONS['num_ctx'] is {config.TASK_OLLAMA_OPTIONS.get('num_ctx')}, expected 4096",
        )

    def test_chat_ollama_options_num_ctx_4096(self):
        """CHAT_OLLAMA_OPTIONS in config.py must define num_ctx=4096."""
        self.assertEqual(
            config.CHAT_OLLAMA_OPTIONS.get("num_ctx"),
            4096,
            f"CHAT_OLLAMA_OPTIONS['num_ctx'] is {config.CHAT_OLLAMA_OPTIONS.get('num_ctx')}, expected 4096",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Persistent ChromaDB Storage Invariant (75,039 chunks)
    # ─────────────────────────────────────────────────────────────────────────

    def test_chromadb_chunk_count_is_75039(self):
        """ChromaDB collection physics_papers must contain exactly 75,039 chunks intact."""
        import chromadb
        client = chromadb.PersistentClient(path=config.VECTORDB_PATH)
        collection = client.get_collection(config.COLLECTION_NAME)
        count = collection.count()
        self.assertEqual(
            count,
            75039,
            f"ChromaDB chunk count mismatch! Expected exactly 75,039 chunks, found: {count}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. TaskRunner Storage Safety Guardrail
    # ─────────────────────────────────────────────────────────────────────────

    def test_task_runner_workspace_storage_protection(self):
        """TaskRunner must reject workspaces pointing directly to or nested inside DATA_DIR or VECTORDB_PATH."""
        from research_assistant.tasks.models import TaskGraph
        from research_assistant.tasks.runner import TaskRunner

        dummy_graph = TaskGraph(paper_id="test_p1", paper_title="Test Paper")
        forbidden_paths = [
            config.VECTORDB_PATH,
            config.DATA_DIR,
            os.path.join(config.VECTORDB_PATH, "nested_dir"),
            os.path.join(config.DATA_DIR, "nested_dir"),
        ]

        for forbidden in forbidden_paths:
            with self.assertRaises(ValueError, msg=f"Should have rejected workspace '{forbidden}'") as ctx:
                TaskRunner(graph=dummy_graph, workspace_dir=forbidden)
            self.assertIn("cannot point to persistent data storage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
