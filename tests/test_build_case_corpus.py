import io
from contextlib import redirect_stderr
import unittest
from unittest.mock import patch

from research_assistant.eval import transfer_cases
from scripts import build_case_corpus as bcc


class TestBuildCaseCorpus(unittest.TestCase):
    def test_it_refuses_to_build_into_any_directory_but_the_cases_own(self):
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(bcc.main(["--case", "nqs", "--resolve"]), 4)      # DATA_DIR is the main data directory in tests
        self.assertIn("CITATION_DATA_DIR", err.getvalue())

    def test_titles_resolve_only_on_a_close_match_with_a_pdf(self):
        hits = [
            {"title": "Training Products of Experts by Minimizing Contrastive Divergence", "year": "2002", "pdf_url": "u"},
            {"title": "Something unrelated", "year": "2002", "pdf_url": "v"},
        ]
        got = bcc.resolve_titles(
            ["Training products of experts by minimizing contrastive divergence", "Nothing like it"],
            search=lambda q, limit: hits,
        )
        self.assertEqual([g["match_title"] is not None for g in got], [True, False])

    def test_hidden_papers_are_never_resolved(self):
        recorded = []

        def fake_resolve(titles, search=None):
            recorded.extend(titles)
            return []

        with patch.object(bcc, "guard_data_dir"), patch.object(bcc, "resolve_titles", side_effect=fake_resolve):
            with patch.object(bcc, "atomic_write_json"):
                bcc.main(["--case", "nqs", "--resolve"])

        case_nqs = next(c for c in transfer_cases.load_cases() if c.id == "nqs")
        for h in case_nqs.papers.get("hidden", []):
            self.assertNotIn(h, recorded)
