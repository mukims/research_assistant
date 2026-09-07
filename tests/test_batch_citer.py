import json
import os
import tempfile
import unittest
from unittest.mock import patch

from research_assistant.agents.agent5_batch_citer import (
    _batch_needs_citation,
    _cite_keys,
    run_batch_citer,
)
from research_assistant.shared.llm import ChatResult


def _reply(text):
    """Wrap *text* in what shared.llm.chat returns."""
    return ChatResult(content=text)


class TestCiteKeyExtraction(unittest.TestCase):
    def test_no_citation(self):
        """A sentence the model declined to cite yields no keys."""
        self.assertEqual(_cite_keys("Disorder is ubiquitous in graphene systems."), set())

    def test_single_and_multi_key(self):
        """Both \\cite{a} and the comma-separated \\cite{a,b} form are recognised."""
        self.assertEqual(_cite_keys("Ballistic transport occurs \\cite{cite_2}."), {"cite_2"})
        self.assertEqual(
            _cite_keys("This is well established \\cite{cite_1, cite_4}."),
            {"cite_1", "cite_4"},
        )

    def test_multiple_citations_in_one_sentence(self):
        """Separate \\cite commands in the same sentence are all collected."""
        text = "Shown in graphene \\cite{cite_1} and in MoS2 \\cite{cite_9}."
        self.assertEqual(_cite_keys(text), {"cite_1", "cite_9"})

    def test_empty_braces_ignored(self):
        """A malformed \\cite{} contributes no key rather than an empty one."""
        self.assertEqual(_cite_keys("A claim \\cite{}."), set())


class TestCitationNeedParsing(unittest.TestCase):
    """Verdicts must bind to the sentence the model numbered, not to line order.

    The earlier parser appended one boolean per line of the response, so any
    preamble or blank line shifted every verdict onto the wrong sentence.
    """

    SENTENCES = ["First claim.", "Second claim.", "Third claim."]
    EXPECTED = [True, False, True]

    def _run(self, response):
        with patch(
            "research_assistant.agents.agent5_batch_citer.chat",
            return_value=_reply(response),
        ):
            return _batch_needs_citation(self.SENTENCES)

    def test_clean_list(self):
        """The well-formed case the prompt asks for."""
        self.assertEqual(self._run("1. YES\n2. NO\n3. YES"), self.EXPECTED)

    def test_preamble_line(self):
        """A conversational opener must not shift the verdicts by one."""
        self.assertEqual(
            self._run("Here are the answers:\n1. YES\n2. NO\n3. YES"), self.EXPECTED
        )

    def test_blank_lines_between_entries(self):
        """A double-spaced list must not interleave phantom False verdicts."""
        self.assertEqual(self._run("1. YES\n\n2. NO\n\n3. YES"), self.EXPECTED)

    def test_trailing_commentary(self):
        """Text after the list is ignored rather than parsed as a verdict."""
        self.assertEqual(
            self._run("1. YES\n2. NO\n3. YES\n\nLet me know if you need more."),
            self.EXPECTED,
        )

    def test_paren_and_lowercase_forms(self):
        """`1) yes` is accepted alongside `1. YES`."""
        self.assertEqual(self._run("1) yes\n2) no\n3) yes"), self.EXPECTED)

    def test_out_of_order_response(self):
        """Verdicts are placed by their number, not by the order received."""
        self.assertEqual(self._run("2. NO\n1. YES\n3. YES"), self.EXPECTED)

    def test_truncated_response_raises(self):
        """A short list must fail loudly rather than be padded with False."""
        with patch("research_assistant.shared.retry.time.sleep"):
            with self.assertRaises(ValueError) as ctx:
                self._run("1. YES\n2. NO")
        self.assertIn("3", str(ctx.exception))

    def test_unparseable_response_raises(self):
        """A response with no verdicts at all raises instead of returning all-False."""
        with patch("research_assistant.shared.retry.time.sleep"):
            with self.assertRaises(ValueError):
                self._run("I cannot determine this without more context.")


class TestVerdictAlignment(unittest.TestCase):
    """Direct regression coverage for the two load-bearing properties of the port.

    ``_batch_needs_citation`` is wrapped in ``@retry(max_retries=2)``, so the
    raising test below sees the underlying ``chat`` mock invoked multiple
    times before the ``ValueError`` finally surfaces — that is expected.
    """

    def test_missing_verdict_raises_rather_than_padding(self):
        """A short reply must not silently shift verdicts onto wrong sentences."""
        with patch(
            "research_assistant.agents.agent5_batch_citer.chat",
            return_value=_reply("1. YES\n2. NO"),
        ):
            with self.assertRaises(ValueError):
                _batch_needs_citation(["one", "two", "three"])

    def test_verdicts_are_indexed_by_number_not_line_position(self):
        """A preamble line must not shift every verdict by one."""
        with patch(
            "research_assistant.agents.agent5_batch_citer.chat",
            return_value=_reply("Here you go:\n\n1. YES\n\n2. NO\n"),
        ):
            self.assertEqual(_batch_needs_citation(["one", "two"]), [True, False])


class TestRunBatchCiter(unittest.TestCase):
    """Regression coverage for run_batch_citer's two safety properties.

    Neither property was exercised anywhere in the suite before this class —
    a reviewer flipped ``if _cite_keys(cited_sentence):`` to ``if True:`` in
    ``run_batch_citer`` and every existing test still passed. That inversion
    would make ``_citations.json`` assert sources that never backed a claim,
    and would make ``_report.md`` report declined citations as successes.

    (a) A successful per-sentence call that *declines* to cite (its reply
        contains no ``\\cite{}``) must not be recorded as a citation.
    (b) A citation-need check that fails after retries must abort the whole
        run and write none of the three output files.

    The retrieval seams (``load_search_resources``, ``hybrid_search``) are
    mocked so the test never touches a real ChromaDB/BM25 index; ``chat`` is
    mocked so it never calls a real model. All output is written under a
    ``tempfile.TemporaryDirectory`` so nothing lands in the repo.
    """

    DRAFT_TEXT = "Graphene exhibits ballistic transport at low temperature."

    def _write_draft(self, tmpdir):
        draft_path = os.path.join(tmpdir, "draft.txt")
        with open(draft_path, "w") as f:
            f.write(self.DRAFT_TEXT)
        return draft_path

    def test_declined_citation_is_not_recorded_as_cited(self):
        """A successful call that declines to cite must not count as cited."""
        with tempfile.TemporaryDirectory() as tmpdir:
            draft_path = self._write_draft(tmpdir)
            out_path = os.path.join(tmpdir, "cited.txt")

            with patch(
                "research_assistant.agents.agent5_batch_citer.load_search_resources",
                return_value=(None, None, [], []),
            ), patch(
                "research_assistant.agents.agent5_batch_citer.hybrid_search",
                return_value=[
                    {
                        "text": "Ballistic transport has been observed in graphene at cryogenic temperatures.",
                        "metadata": {"citation_source": "Doe, J. et al. (2020)"},
                    }
                ],
            ), patch(
                "research_assistant.agents.agent5_batch_citer.chat",
                side_effect=[
                    _reply("1. YES"),
                    _reply(
                        "CITED: The original sentence unchanged.\n"
                        "REASON: The context does not support this claim."
                    ),
                ],
            ):
                result = run_batch_citer(draft_path, out_path)

            self.assertEqual(result, out_path)

            mapping_path = out_path.replace(".txt", "_citations.json")
            report_path = out_path.replace(".txt", "_report.md")
            self.assertTrue(os.path.exists(out_path))
            self.assertTrue(os.path.exists(mapping_path))
            self.assertTrue(os.path.exists(report_path))

            with open(mapping_path) as f:
                mapping = json.load(f)
            # The retrieved source was never actually used in a \cite{}, so
            # it must not appear in the key -> source mapping that drives
            # BibTeX generation.
            self.assertEqual(mapping, {})

            with open(report_path) as f:
                report = f.read()
            # The declined sentence must be tallied as "needed a citation,
            # none made" -- never folded into the "Cited" count.
            self.assertIn("| Cited | 0 |", report)
            self.assertIn("**Needed a citation, none made** | **1**", report)

    def test_failed_citation_need_check_aborts_without_writing(self):
        """A citation-need check that fails after retries must write nothing.

        A misaligned verdict list would attribute one sentence's citation
        decision to another, and a partially-written draft is worse than no
        draft -- so a failure here must abort before any output is written.
        ``_batch_needs_citation`` is wrapped in ``@retry(max_retries=2)``, so
        making its underlying ``chat`` call always raise means it is invoked
        twice (with the retry's backoff sleep stubbed out) before the
        exception finally surfaces to ``run_batch_citer`` -- that is expected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            draft_path = self._write_draft(tmpdir)
            out_path = os.path.join(tmpdir, "cited.txt")

            with patch(
                "research_assistant.agents.agent5_batch_citer.load_search_resources",
                return_value=(None, None, [], []),
            ), patch(
                "research_assistant.agents.agent5_batch_citer.chat",
                side_effect=RuntimeError("simulated LLM failure"),
            ), patch("research_assistant.shared.retry.time.sleep"):
                result = run_batch_citer(draft_path, out_path)

            self.assertIsNone(result)

            mapping_path = out_path.replace(".txt", "_citations.json")
            report_path = out_path.replace(".txt", "_report.md")
            self.assertFalse(os.path.exists(out_path))
            self.assertFalse(os.path.exists(mapping_path))
            self.assertFalse(os.path.exists(report_path))


if __name__ == "__main__":
    unittest.main()
