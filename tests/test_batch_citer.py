import json
import os
import tempfile
import unittest
from unittest.mock import patch

from research_assistant.agents.agent5_batch_citer import (
    CiteResult,
    _batch_needs_citation,
    _cite_keys,
    _insert_cite,
    cite_sentence,
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

    def setUp(self):
        # The three output files are pinned byte-for-byte on the rewrite
        # path; the judge default must not leak into that pin.
        p = patch("research_assistant.agents.agent5_batch_citer.CITATION_CITER_JUDGE", False)
        p.start()
        self.addCleanup(p.stop)

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

            with open(out_path) as f:
                draft = f.read()
            # The model's declined reply above is a paraphrase, not the
            # sentence. Nothing of it may reach the draft.
            self.assertEqual(draft, self.DRAFT_TEXT)

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


from research_assistant.agents.agent5_batch_citer import _restore_terminal_punctuation, split_into_sentences


class TestTerminalPunctuationSurvivesCitation(unittest.TestCase):
    r"""gemma4:e2b drops the full stop when it appends \cite{}. The draft is
    joined with spaces and Agent 8 splits on [.!?]+whitespace, so a lost stop
    merges two sentences into one claim. Seen 2026-09-12."""

    def test_dropped_full_stop_is_restored_after_the_cite(self):
        self.assertEqual(
            _restore_terminal_punctuation("Films are good.", "Films are good \\cite{cite_1}"),
            "Films are good \\cite{cite_1}.",
        )

    def test_question_mark_is_restored(self):
        self.assertEqual(_restore_terminal_punctuation("Is it so?", "Is it so \\cite{a}"), "Is it so \\cite{a}?")

    def test_already_terminated_after_the_cite_is_unchanged(self):
        self.assertEqual(_restore_terminal_punctuation("Films are good.", "Films are good \\cite{a}."),
                         "Films are good \\cite{a}.")

    def test_terminated_before_a_trailing_cite_is_unchanged(self):
        self.assertEqual(_restore_terminal_punctuation("Films are good.", "Films are good. \\cite{a}"),
                         "Films are good. \\cite{a}")

    def test_multiple_trailing_cites(self):
        self.assertEqual(_restore_terminal_punctuation("X holds.", "X holds \\cite{a} \\cite{b}"),
                         "X holds \\cite{a} \\cite{b}.")

    def test_original_without_terminal_punctuation_is_left_alone(self):
        self.assertEqual(_restore_terminal_punctuation("a heading", "a heading \\cite{a}"), "a heading \\cite{a}")

    def test_trailing_whitespace_is_trimmed(self):
        self.assertEqual(_restore_terminal_punctuation("Done.", "Done \\cite{a}   "), "Done \\cite{a}.")

    def test_the_rejoined_draft_splits_back_into_two_sentences(self):
        a = _restore_terminal_punctuation("First claim here.", "First claim here \\cite{cite_1}")
        b = _restore_terminal_punctuation("Second claim here.", "Second claim here \\cite{cite_2}")
        self.assertEqual(len(split_into_sentences(" ".join([a, b]))), 2)


class TestCiteSentenceKeepsPunctuation(unittest.TestCase):
    def test_model_reply_without_full_stop_is_repaired(self):
        import research_assistant.agents.agent5_batch_citer as a5
        reply = ChatResult(content="CITED: Films are good \\cite{cite_1}\nREASON: because.")
        with patch.object(a5, "chat", return_value=reply):
            cited, reason = a5._cite_sentence_with_reasoning("Films are good.", "--- Context (Cite Key: cite_1) ---\nx")
        self.assertEqual(cited, "Films are good \\cite{cite_1}.")
        self.assertEqual(reason, "because.")


class TestCiteSentence(unittest.TestCase):
    """The per-sentence seam. Everything the batch loop did inline lives here,
    so it can be called by an evaluation, gated by a judge, and tested.
    Calls pass judge_gate=False: these pin the rewrite path, whatever the default."""

    RES = (None, None, [], [])
    HIT = {
        "text": "Ballistic transport has been observed in graphene at cryogenic temperatures.",
        "chunk_index": 7, "rrf_score": 0.03,
        "metadata": {"citation_source": "Doe, J. et al. (2020)", "document": "doe2020.pdf"},
    }

    def test_a_citation_carries_key_and_document(self):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[self.HIT]), \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: Graphene is ballistic \\cite{cite_1}.\nREASON: Directly reported.")):
            res = cite_sentence("Graphene is ballistic.", self.RES, {}, judge_gate=False)
        self.assertIsInstance(res, CiteResult)
        self.assertTrue(res.cited)
        self.assertEqual(res.keys, ["cite_1"])
        self.assertEqual(res.cited_text, "Graphene is ballistic \\cite{cite_1}.")
        self.assertEqual(res.reasoning, "Directly reported.")
        self.assertEqual(res.candidates, [{
            "key": "cite_1", "citation": "Doe, J. et al. (2020)", "document": "doe2020.pdf",
            "chunk_index": 7, "rrf_score": 0.03,
        }])
        self.assertIsNone(res.skip_reason)
        self.assertEqual(res.query, "Graphene is ballistic.")

    def test_registry_keys_are_stable_across_sentences(self):
        registry = {}
        other = dict(self.HIT, metadata={"citation_source": "Roe 2019", "document": "roe.pdf"})
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search",
                   side_effect=[[self.HIT], [other, self.HIT]]), \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: S.\nREASON: no")):
            cite_sentence("First.", self.RES, registry, judge_gate=False)
            res = cite_sentence("Second.", self.RES, registry, judge_gate=False)
        self.assertEqual(registry, {"Doe, J. et al. (2020)": "cite_1", "Roe 2019": "cite_2"})
        self.assertEqual([c["key"] for c in res.candidates], ["cite_2", "cite_1"])

    def test_no_context_and_declined_are_told_apart(self):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[]):
            res = cite_sentence("Nothing here.", self.RES, {}, judge_gate=False)
        self.assertFalse(res.cited)
        self.assertEqual(res.cited_text, "Nothing here.")
        self.assertEqual(res.skip_reason, "no relevant context found in database")
        self.assertEqual(res.candidates, [])

        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[self.HIT]), \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: Nothing here.\nREASON: The context is about something else.")):
            res = cite_sentence("Nothing here.", self.RES, {}, judge_gate=False)
        self.assertFalse(res.cited)
        self.assertEqual(res.skip_reason, "context retrieved but the model did not cite it")
        self.assertEqual(res.reasoning, "The context is about something else.")
        self.assertEqual(len(res.candidates), 1)

    def test_declined_rewrite_never_reaches_the_draft(self):
        # A decline carries no key, so the model's text is noise: the draft
        # keeps the author's sentence, as the report already claims it does.
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[self.HIT]), \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: Nothing to see here.\nREASON: Off topic.")):
            res = cite_sentence("Nothing here.", self.RES, {}, judge_gate=False)
        self.assertFalse(res.cited)
        self.assertEqual(res.cited_text, "Nothing here.")
        self.assertEqual(res.original, "Nothing here.")
        self.assertEqual(res.reasoning, "Off topic.")

    def test_query_and_exclusion_reach_retrieval(self):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[]) as hs:
            res = cite_sentence("This approach works.", self.RES, {},
                                query="recursive Green's function inversion works", exclude_docs={"seed.pdf"})
        self.assertEqual(hs.call_args[0][0], "recursive Green's function inversion works")
        self.assertEqual(hs.call_args.kwargs["exclude_docs"], {"seed.pdf"})
        self.assertEqual(res.query, "recursive Green's function inversion works")
        self.assertEqual(res.original, "This approach works.")


def _verdict(judgement, span_verified=True, reason="because", span="the evidence sentence"):
    return {"judgement": judgement, "model_judgement": judgement, "confidence": "High",
            "evidence_sufficiency": "sufficient", "supporting_span": span, "span_verified": span_verified,
            "reason": reason, "rubric_violations": [],
            "slots": {"finding": {"assertion": "f", "verdict": "Supports"},
                      "scope": {"assertion": "s", "verdict": "Supports"},
                      "strength": {"assertion": "t", "verdict": "Not applicable"}}}


class TestInsertCite(unittest.TestCase):
    def test_before_terminal_punctuation(self):
        self.assertEqual(_insert_cite("Graphene is ballistic.", "cite_1"), "Graphene is ballistic \\cite{cite_1}.")
        self.assertEqual(_insert_cite("Is it ballistic?", "cite_2"), "Is it ballistic \\cite{cite_2}?")

    def test_appended_when_there_is_none(self):
        self.assertEqual(_insert_cite("Graphene is ballistic", "cite_1"), "Graphene is ballistic \\cite{cite_1}")

    def test_trailing_whitespace_is_dropped(self):
        self.assertEqual(_insert_cite("Ballistic.  ", "cite_1"), "Ballistic \\cite{cite_1}.")


class TestCiteByJudge(unittest.TestCase):
    """The citer and the auditor apply the same test. A candidate is cited
    only if the judge would pass it — Supports with a verbatim span — and the
    key is placed by code, so nothing has to be parsed out of a rewrite."""

    RES = (None, None, [], [])
    HITS = [
        {"text": "Off-topic passage.", "chunk_index": 1, "rrf_score": 0.03,
         "metadata": {"citation_source": "Roe 2019", "document": "roe.pdf"}},
        {"text": "Ballistic transport observed in graphene.", "chunk_index": 2, "rrf_score": 0.02,
         "metadata": {"citation_source": "Doe 2020", "document": "doe.pdf"}},
        {"text": "Another passage.", "chunk_index": 3, "rrf_score": 0.01,
         "metadata": {"citation_source": "Poe 2021", "document": "poe.pdf"}},
    ]

    def _cite(self, verdicts, **kw):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(h) for h in self.HITS]), \
             patch("research_assistant.agents.agent5_batch_citer.judge", side_effect=verdicts) as judge, \
             patch("research_assistant.agents.agent5_batch_citer.chat") as chat:
            res = cite_sentence("Graphene is ballistic.", self.RES, {}, judge_gate=True, **kw)
        return res, judge, chat

    def test_first_supports_with_a_verified_span_is_cited_and_the_rewrite_is_never_asked_for(self):
        res, judge, chat = self._cite([_verdict("Unclear / insufficient evidence"), _verdict("Supports")])
        self.assertEqual(res.keys, ["cite_2"])
        self.assertEqual(res.cited_text, "Graphene is ballistic \\cite{cite_2}.")
        self.assertEqual(res.reasoning, "because")
        self.assertFalse(res.partial)
        self.assertEqual([v["judgement"] for v in res.verdicts], ["Unclear / insufficient evidence", "Supports"])
        self.assertEqual(res.verdicts[1]["document"], "doe.pdf")
        self.assertEqual(judge.call_count, 2)          # stopped at the first pass
        chat.assert_not_called()

    def test_supports_without_a_verified_span_is_not_accepted(self):
        res, judge, _ = self._cite([_verdict("Supports", span_verified=False),
                                    _verdict("Does not support"), _verdict("Unclear / insufficient evidence")])
        self.assertFalse(res.cited)
        self.assertEqual(res.cited_text, "Graphene is ballistic.")
        self.assertEqual(res.skip_reason, "no candidate passed the judge")
        self.assertEqual(judge.call_count, 3)
        # The closest candidate is the unverified Supports, first by verdict rank; it is marked and its reason kept.
        self.assertTrue(res.verdicts[0]["best"])
        self.assertEqual(res.reasoning, "because")

    def test_partially_supports_is_accepted_only_when_nothing_supports_and_is_flagged(self):
        res, _, _ = self._cite([_verdict("Partially supports"), _verdict("Does not support"), _verdict("Unclear / insufficient evidence")])
        self.assertEqual(res.keys, ["cite_1"])
        self.assertTrue(res.partial)
        res, _, _ = self._cite([_verdict("Partially supports"), _verdict("Supports")])
        self.assertEqual(res.keys, ["cite_2"])
        self.assertFalse(res.partial)

    def test_a_failing_judge_call_moves_to_the_next_candidate(self):
        res, _, _ = self._cite([RuntimeError("model down"), _verdict("Supports")])
        self.assertEqual(res.keys, ["cite_2"])
        self.assertIn("model down", res.verdicts[0]["error"])
        self.assertNotIn("judgement", res.verdicts[0])

    def test_every_call_failing_is_its_own_reason(self):
        res, _, _ = self._cite([RuntimeError("a"), RuntimeError("b"), RuntimeError("c")])
        self.assertFalse(res.cited)
        self.assertEqual(res.skip_reason, "judge failed on every candidate")

    def test_the_paragraph_window_reaches_the_judge_as_context(self):
        _, judge, _ = self._cite([_verdict("Supports")], context="Before. «Graphene is ballistic.» After.")
        self.assertEqual(judge.call_args.kwargs["context"], "Before. «Graphene is ballistic.» After.")

    def test_gate_off_is_the_legacy_path(self):
        with patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(self.HITS[1])]), \
             patch("research_assistant.agents.agent5_batch_citer.judge") as judge, \
             patch("research_assistant.agents.agent5_batch_citer.chat",
                   return_value=_reply("CITED: Graphene is ballistic \\cite{cite_1}.\nREASON: r")):
            res = cite_sentence("Graphene is ballistic.", self.RES, {}, judge_gate=False)
        judge.assert_not_called()
        self.assertEqual(res.keys, ["cite_1"])
        self.assertEqual(res.verdicts, [])

    def test_gate_none_reads_the_config(self):
        with patch("research_assistant.agents.agent5_batch_citer.CITATION_CITER_JUDGE", True), \
             patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(self.HITS[1])]), \
             patch("research_assistant.agents.agent5_batch_citer.judge", return_value=_verdict("Supports")), \
             patch("research_assistant.agents.agent5_batch_citer.chat") as chat:
            res = cite_sentence("Graphene is ballistic.", self.RES, {})
        self.assertEqual(res.keys, ["cite_1"])
        chat.assert_not_called()


class TestReportShowsTheJudgesAccount(unittest.TestCase):
    """With the judge on, the citation report is the audit report's twin:
    a cited sentence shows its verified span and how the verdict was
    reached; a declined one shows the closest candidate and why it failed."""

    # Both sentences have ≥ 4 words: shorter ones are skipped as "too short"
    # before the need check and never reach the citer at all.
    DRAFT = "Graphene shows ballistic transport. Nothing in the corpus supports this."
    HIT = {"text": "Ballistic transport observed in graphene.", "chunk_index": 2, "rrf_score": 0.02,
           "metadata": {"citation_source": "Doe 2020", "document": "doe.pdf"}}

    def test_cited_and_declined_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            draft_path = os.path.join(tmpdir, "draft.txt")
            with open(draft_path, "w") as f:
                f.write(self.DRAFT)
            out_path = os.path.join(tmpdir, "cited.txt")
            with patch("research_assistant.agents.agent5_batch_citer.CITATION_CITER_JUDGE", True), \
                 patch("research_assistant.agents.agent5_batch_citer.load_search_resources", return_value=(None, None, [], [])), \
                 patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(self.HIT)]), \
                 patch("research_assistant.agents.agent5_batch_citer.chat", return_value=_reply("1. YES\n2. YES")), \
                 patch("research_assistant.agents.agent5_batch_citer.judge",
                       side_effect=[_verdict("Supports", span="Ballistic transport observed in graphene."),
                                    _verdict("Does not support", span_verified=False, reason="Not about this.")]):
                run_batch_citer(draft_path, out_path)
            with open(out_path) as f:
                self.assertEqual(f.read(), "Graphene shows ballistic transport \\cite{cite_1}. Nothing in the corpus supports this.")
            with open(out_path.replace(".txt", "_report.md")) as f:
                report = f.read()
        self.assertIn("**Verified span:**", report)
        self.assertIn("Ballistic transport observed in graphene.", report)
        self.assertIn("**Why this citation**", report)
        self.assertIn("Attributed to the cited paper", report)
        self.assertIn("**Closest candidate:** `cite_1`", report)
        self.assertIn("Does not support", report)
        self.assertIn("Not about this.", report)
        self.assertNotIn("Partial support", report)

    def test_partial_support_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            draft_path = os.path.join(tmpdir, "draft.txt")
            with open(draft_path, "w") as f:
                f.write("Graphene shows ballistic transport.")
            out_path = os.path.join(tmpdir, "cited.txt")
            with patch("research_assistant.agents.agent5_batch_citer.CITATION_CITER_JUDGE", True), \
                 patch("research_assistant.agents.agent5_batch_citer.load_search_resources", return_value=(None, None, [], [])), \
                 patch("research_assistant.agents.agent5_batch_citer.hybrid_search", return_value=[dict(self.HIT)]), \
                 patch("research_assistant.agents.agent5_batch_citer.chat", return_value=_reply("1. YES")), \
                 patch("research_assistant.agents.agent5_batch_citer.judge", return_value=_verdict("Partially supports")):
                run_batch_citer(draft_path, out_path)
            with open(out_path.replace(".txt", "_report.md")) as f:
                report = f.read()
        self.assertIn("⚠ **Partial support**", report)


if __name__ == "__main__":
    unittest.main()
