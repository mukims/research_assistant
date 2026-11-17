import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from scripts import build_cards


class TestBuildCardsCli(unittest.TestCase):
    def test_a_busy_pipeline_exits_2_before_loading_anything(self):
        with patch.object(
            build_cards.pipeline_status,
            "get_status",
            return_value={"active": True, "stage_label": "Ingesting"},
        ), patch.object(build_cards, "load_search_resources") as lsr, redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(build_cards.main(["--limit", "1"]), 2)
        lsr.assert_not_called()

    def test_repair_links_flag_invokes_store_repair_and_exits_0(self):
        out = io.StringIO()
        with patch.object(build_cards.pipeline_status, "get_status", return_value={"active": False}), \
             patch.object(build_cards, "load_search_resources", return_value=(None, None, [], [])), \
             patch.object(build_cards, "_load_doc_map", return_value={"a.pdf": "Cite A"}), \
             patch.object(build_cards.CardStore, "repair_evidence_links", return_value={"total": 5, "repaired": 4, "already_linked": 1, "unresolved": 0}) as rep_mock, \
             patch("sys.stdout", out):
            rc = build_cards.main(["--repair-links"])
        self.assertEqual(rc, 0)
        rep_mock.assert_called_once()
        self.assertIn("4 repaired", out.getvalue())
