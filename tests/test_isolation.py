"""The test process has no network, no API keys and no shared on-disk state."""

import os
import socket
import tempfile
import unittest
from pathlib import Path

from research_assistant import config
from tests import conftest


# This suite is unittest-compatible: test_challenger_m4_storage forbids pytest imports in tests.
_assert = unittest.TestCase()


def test_outbound_tcp_is_blocked():
    with _assert.assertRaisesRegex(OSError, "network disabled in tests"):
        socket.create_connection(("example.org", 443), timeout=1)


def test_localhost_tcp_is_blocked_too():   # planner tests reached a live Ollama on 11434
    with _assert.assertRaisesRegex(OSError, "network disabled in tests"):
        socket.create_connection(("127.0.0.1", 11434), timeout=1)


def test_no_api_keys_in_the_test_process():
    assert not any(os.environ.get(k) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"))
    assert config.GEMINI_API_KEY is None and config.LLM_BACKEND == "ollama"


# With CITATION_TEST_REAL_STORE=1 (or a preset CITATION_DATA_DIR) conftest does not wipe DATA_DIR,
# which may be the real corpus; writing a marker there would only litter it.
_wipes_data_dir = unittest.skipIf(conftest.TEST_DATA_DIR is None, "conftest did not create DATA_DIR")


@_wipes_data_dir
def test_data_dir_wipe_part_1_writes():
    marker = Path(config.DATA_DIR, "marker")
    marker.write_text("x")
    assert marker.exists()


@_wipes_data_dir
def test_data_dir_wipe_part_2_sees_nothing():
    assert not Path(config.DATA_DIR, "marker").exists()


def test_wipe_refuses_a_dir_that_is_not_the_test_dir(tmp_path):
    with _assert.assertRaisesRegex(RuntimeError, "refusing to wipe"):
        conftest._wipe_contents(str(tmp_path))


def test_workspace_root_is_a_temp_dir():
    assert os.environ["CITATION_WORKSPACE_DIR"].startswith(tempfile.gettempdir())
