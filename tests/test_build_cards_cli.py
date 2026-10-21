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
