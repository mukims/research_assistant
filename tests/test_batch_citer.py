import unittest
from unittest.mock import patch

from research_assistant.agents.agent5_batch_citer import _batch_needs_citation, _cite_keys
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


if __name__ == "__main__":
    unittest.main()
