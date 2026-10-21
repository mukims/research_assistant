import json
import tempfile
import unittest

import numpy as np

from research_assistant.transfer import cards, models, premises
from tests.transfer_fixtures import card


class TestSelect(unittest.TestCase):
    def test_abstract_then_methods_then_one_chunk_per_heading_without_figures_or_captions(self):
        texts = ["abs", "intro1", "intro2", "meth", "cap", "img", "green1"]
        metas = [
            {"document": "d", "type": "text_chunk", "section": s, "section_raw": r}
            for s, r in [
                ("abstract", ""),
                ("introduction", "Introduction"),
                ("introduction", "Introduction"),
                ("methods", "Model"),
                ("caption", "Fig. 1"),
                ("other", "Fig. 1"),
                ("other", "Quantum Green's functions"),
            ]
        ]
        metas[5]["type"] = "figure_image"
        got = cards.select_card_passages("d", "D", texts, metas, max_chars=10_000)
        self.assertEqual([p.text for p in got], ["abs", "meth", "intro1", "green1"])

    def test_selection_stops_before_the_character_budget(self):
        metas = [
            {"document": "d", "type": "text_chunk", "section": "other", "section_raw": f"H{i}"}
            for i in range(5)
        ]
        self.assertEqual(
            len(cards.select_card_passages("d", "D", ["x" * 400] * 5, metas, max_chars=1000)),
            2,
        )


class TestBuild(unittest.TestCase):
    def test_a_technique_whose_evidence_does_not_resolve_is_dropped_and_the_rest_verified(self):
        ps = [models.Passage("C1", 0, "d", "D", "methods", "We use Chebyshev expansion.", "card")]
        reply = json.dumps({
            "techniques": [
                {
                    "name": "Chebyshev expansion",
                    "problem_solved": "approximate a function of a large sparse matrix",
                    "context": "DOS",
                    "inputs": "sparse matrix",
                    "outputs": "spectral density",
                    "assumptions": ["bounded spectrum"],
                    "mechanism": [
                        "Rescale the matrix so its spectrum lies in [-1, 1].",
                        "Compute Chebyshev moments by the three-term recursion.",
                    ],
                    "evidence": {
                        "claim": "The paper expands the DOS in Chebyshev polynomials.",
                        "key": "C1",
                    },
                },
                {
                    "name": "x",
                    "problem_solved": "y",
                    "context": "",
                    "inputs": "",
                    "outputs": "",
                    "assumptions": [],
                    "mechanism": ["Do a.", "Do b."],
                    "evidence": {"claim": "z", "key": "C9"},
                },
            ]
        })
        checker = premises.PremiseChecker(
            lambda c, e: {
                "judgement": "Supports",
                "span_verified": True,
                "supporting_span": "s",
                "reason": "r",
            },
            ["We use Chebyshev expansion."],
            [{"document": "d"}],
        )
        got, warnings = cards.build_card("d", "D", ps, chat=lambda m: reply, checker=checker)
        self.assertEqual([(c.card_id, c.verified) for c in got], [("d#0", True)])
        self.assertTrue(any("C9" in w for w in warnings))


    def _technique(self, name, mechanism):
        return {
            "name": name,
            "problem_solved": "approximate a function of a large sparse matrix",
            "context": "DOS",
            "inputs": "sparse matrix",
            "outputs": "spectral density",
            "assumptions": [],
            "mechanism": mechanism,
            "evidence": {"claim": "The paper expands the DOS in Chebyshev polynomials.", "key": "C1"},
        }

    def _build(self, reply):
        ps = [models.Passage("C1", 0, "d", "D", "methods", "We use Chebyshev expansion.", "card")]
        checker = premises.PremiseChecker(
            lambda c, e: {"judgement": "Supports", "span_verified": True, "supporting_span": "s", "reason": "r"},
            ["We use Chebyshev expansion."],
            [{"document": "d"}],
        )
        return cards.build_card("d", "D", ps, chat=lambda m: json.dumps(reply), checker=checker)

    def test_a_technique_without_a_concrete_mechanism_is_dropped_with_a_warning(self):
        steps = [
            "Rescale the matrix so its spectrum lies in [-1, 1].",
            "Compute Chebyshev moments by the three-term recursion.",
            "Damp the moments with the Jackson kernel.",
        ]
        got, warnings = self._build({"techniques": [
            self._technique("Kernel polynomial method", steps),
            self._technique("Combine methods", ["Combine the two approaches."]),
            self._technique("Unspecified", []),
        ]})
        self.assertEqual([(c.name, c.mechanism) for c in got], [("Kernel polynomial method", steps)])
        self.assertEqual([c.card_id for c in got], ["d#0"])
        self.assertEqual(sum("mechanism" in w for w in warnings), 2)

    def test_blank_mechanism_steps_do_not_count(self):
        got, _ = self._build({"techniques": [self._technique("T", ["Do one thing.", "  ", ""])]})
        self.assertEqual(got, [])

    def test_a_paper_with_no_reusable_technique_gives_no_cards_and_no_warnings(self):
        self.assertEqual(self._build({"techniques": []}), ([], []))

class TestStoreAndIndex(unittest.TestCase):
    def test_the_store_resumes_and_the_index_excludes_documents_and_unverified_cards(self):
        with tempfile.TemporaryDirectory() as d:
            store = cards.CardStore(d)
            a, b, u = card("a.pdf", verified=True), card("b.pdf", verified=True), card("u.pdf", verified=False)
            store.append("a.pdf", [a], np.array([[1.0, 0.0]]))
            store.append("b.pdf", [b], np.array([[0.8, 0.6]]))
            store.append("u.pdf", [u], np.array([[1.0, 0.01]]))
            self.assertEqual(store.done_documents(), {"a.pdf", "b.pdf", "u.pdf"})
            ix = store.load()
            self.assertEqual([c.document for c, _ in ix.search(np.array([1.0, 0.0]), 3)], ["a.pdf", "b.pdf"])
            self.assertEqual(
                [c.document for c, _ in ix.search(np.array([1.0, 0.0]), 3, exclude_documents={"a.pdf"})],
                ["b.pdf"],
            )

    def test_build_all_skips_documents_already_built(self):
        with tempfile.TemporaryDirectory() as d:
            store = cards.CardStore(d)
            a = card("a.pdf", verified=True)
            store.append("a.pdf", [a], np.array([[1.0, 0.0]]))
            chat_calls = []

            def mock_chat(m):
                chat_calls.append(m)
                return json.dumps({
                    "techniques": [
                        {
                            "name": "T",
                            "problem_solved": "P",
                            "context": "C",
                            "inputs": "I",
                            "outputs": "O",
                            "assumptions": [],
                            "evidence": {"claim": "cl", "key": "C1"},
                        }
                    ]
                })

            checker = premises.PremiseChecker(
                lambda c, e: {"judgement": "Supports", "span_verified": True, "supporting_span": "s", "reason": "r"},
                ["text b"],
                [{"document": "b.pdf"}],
            )
            docs = [("a.pdf", "Citation A"), ("b.pdf", "Citation B")]
            texts = ["text a", "text b"]
            metas = [
                {"document": "a.pdf", "type": "text_chunk", "section": "methods", "section_raw": "M"},
                {"document": "b.pdf", "type": "text_chunk", "section": "methods", "section_raw": "M"},
            ]
            res = cards.build_all(
                docs,
                chat=mock_chat,
                checker=checker,
                embed_documents=lambda ts: [[1.0, 0.0] for _ in ts],
                store=store,
                texts=texts,
                metadatas=metas,
                max_chars=10_000,
            )
            self.assertEqual(len(chat_calls), 1)
            self.assertEqual(res["skipped"], 1)
            self.assertEqual(res["built"], 1)


class TestCardRecords(unittest.TestCase):
    def test_a_card_stored_before_mechanism_existed_loads_with_no_steps(self):
        d = models.card_to_dict(card("a.pdf", True))
        del d["mechanism"]
        self.assertEqual(models.card_from_dict(d).mechanism, [])
