"""A passage is identified by (document, seq) on every store, never by a bare chunk_index.

On LanceDB the texts/metadatas sequences are ordered by paper and seq restarts at 0 for each paper,
while a search hit's chunk_index is that per-paper seq. A card passage is found by enumerating the
corpus, so its chunk_index is a position in texts. These tests build corpora of that shape and check
that nothing compares a position with a seq.
"""

import unittest

from research_assistant.transfer import stages
from research_assistant.transfer.cards import select_card_passages
from research_assistant.transfer.deps import TransferSettings
from research_assistant.transfer.models import (
    Bottleneck,
    Card,
    Premise,
    card_from_dict,
    result_from_dict,
    result_to_dict,
)
from research_assistant.transfer.premises import PremiseChecker, resolve_chunk
from research_assistant.transfer.replies import resolve_premises
from tests.transfer_fixtures import sample_result, world


def lance_corpus(sizes):
    """Rows ordered by paper, seq restarting at 0 per paper, as LanceSequenceProxy serves them."""
    texts, metadatas = [], []
    for doc, n in sizes:
        for seq in range(n):
            texts.append(f"{doc}-seq{seq}")
            metadatas.append({"document": doc, "section": "methods", "seq": seq, "citation_source": f"Cite {doc}"})
    return texts, metadatas


def lance_hit(doc, seq):
    """A search hit as lance_store.search_hybrid returns it: chunk_index is the per-paper seq."""
    return {
        "chunk_index": seq,
        "text": f"{doc}-seq{seq}",
        "metadata": {"document": doc, "section": "methods", "seq": seq, "citation_source": f"Cite {doc}"},
        "score": 1.0,
    }


class TestResolveChunk(unittest.TestCase):
    def setUp(self):
        # Paper A is rows 0-2, paper B rows 3-9 with seq 0-6.
        self.texts, self.metas = lance_corpus([("A", 3), ("B", 7)])

    def test_a_seq_is_never_read_as_a_position_in_the_same_paper(self):
        # Row 5 belongs to B too (its seq 2), which is exactly the chunk that must not be returned.
        idx, text, meta = resolve_chunk(self.texts, self.metas, "B", 5)
        self.assertEqual((idx, text, meta["seq"]), (8, "B-seq5", 5))

    def test_a_position_hint_is_used_only_when_its_row_has_the_seq(self):
        self.assertEqual(resolve_chunk(self.texts, self.metas, "B", 5, position=8)[0], 8)
        self.assertEqual(resolve_chunk(self.texts, self.metas, "B", 5, position=5)[0], 8)

    def test_a_seq_the_paper_does_not_have_resolves_to_nothing(self):
        self.assertIsNone(resolve_chunk(self.texts, self.metas, "B", 7))
        self.assertIsNone(resolve_chunk(self.texts, self.metas, "C", 0))

    def test_a_store_without_seq_numbers_its_chunks_by_position(self):
        # Chroma v1 chunks and list-backed corpora record no seq; their search hits report a position.
        texts = ["a0", "b0", "b1"]
        metas = [{"document": "a"}, {"document": "b"}, {"document": "b"}]
        self.assertEqual(resolve_chunk(texts, metas, "b", 2)[:2], (2, "b1"))
        self.assertIsNone(resolve_chunk(texts, metas, "b", 0))


class TestPremiseCheckerJudgesTheCitedChunk(unittest.TestCase):
    def test_a_premise_from_a_lance_hit_is_judged_against_that_chunk(self):
        texts, metas = lance_corpus([("A", 3), ("B", 7)])
        judged = {}

        def judge(claim, evidence):
            judged[claim] = evidence
            return {"judgement": "Supports", "span_verified": True, "supporting_span": "s", "reason": "r"}

        # The world's bottleneck reply cites C1, C2 and C3: B's seq 5, A's seq 2 and B's seq 0.
        deps, _ = world(search=lambda q, k, docs, excl: [lance_hit("B", 5), lance_hit("A", 2), lance_hit("B", 0)])
        deps.texts, deps.metadatas = texts, metas
        g = stages.ground("q", deps, TransferSettings(), PremiseChecker(judge, texts, metas))

        p1 = g.bottlenecks[0].premises[0]
        self.assertEqual((p1.key, p1.document, p1.seq), ("C1", "B", 5))
        self.assertIn("B-seq5", judged["bottleneck fact one"])
        self.assertNotIn("B-seq2", judged["bottleneck fact one"])
        self.assertIn("B-seq0", judged["bottleneck fact three"])


class TestDevelopComparesLikeWithLike(unittest.TestCase):
    def setUp(self):
        # Paper A is rows 0-2, paper B rows 3-12 with seq 0-9: B's seq 5 sits at position 8.
        self.texts, self.metas = lance_corpus([("A", 3), ("B", 10)])
        self.deps, _ = world()
        self.deps.texts, self.deps.metadatas = self.texts, self.metas
        self.prompts = []
        chat = self.deps.chat

        def recording_chat(messages, temperature=0.7):
            self.prompts.append(messages[-1]["content"])
            return chat(messages, temperature)

        self.deps.chat = recording_chat
        self.checker = PremiseChecker(self.deps.judge, self.texts, self.metas)

    def _card_with_evidence_at(self, seq):
        """A card built the way build_card builds one: evidence cites a card passage."""
        passages = select_card_passages("B", "Cite B", self.texts, self.metas)
        key = next(p.key for p in passages if p.text == f"B-seq{seq}")
        evidence = resolve_premises([{"claim": "B does it", "key": key}], {p.key: p for p in passages}, limit=1)[0][0]
        self.assertEqual((evidence.chunk_index, evidence.seq), (3 + seq, seq))
        return Card(
            card_id="B#0", document="B", citation="Cite B", name="method B", problem_solved="problem B",
            context="", inputs="", outputs="", assumptions=[], evidence=evidence, mechanism=["a", "b"],
        )

    def _grounding(self, near_hits):
        passages = {}
        for n, (doc, seq) in enumerate(near_hits, start=1):
            hit = lance_hit(doc, seq)
            passages[f"C{n}"] = stages.Passage(
                key=f"C{n}", chunk_index=hit["chunk_index"], seq=seq, document=doc,
                citation=f"Cite {doc}", section="methods", text=hit["text"], role="near",
            )
        b = Bottleneck(id="B1", statement="s", quantities=["q"], current_method="m",
                       generic={"specific": "x", "mid": "x", "broad": "x"})
        return stages.Grounding(query="q", near=[], passages=passages, bottlenecks=[b], warnings=[])

    def _develop(self, card, grounding):
        seed = stages.Seed(bottleneck_id="B1", card=card, score=1.0, level="mid", rank=1)
        return stages.develop(seed, 1, grounding, self.deps, TransferSettings(), self.checker)

    def test_card_evidence_is_not_replaced_by_a_passage_whose_seq_equals_its_position(self):
        card = self._card_with_evidence_at(5)          # position 8
        g = self._grounding([("B", 8)])                # a near passage with seq 8
        self.deps.search = lambda *a, **k: []
        self._develop(card, g)
        transfer_prompt = next(p for p in self.prompts if "TASK: transfer" in p)
        self.assertIn("B-seq5", transfer_prompt)
        self.assertTrue(any(p.text == "B-seq5" and p.role == "source" for p in g.passages.values()))

    def test_card_evidence_reuses_the_passage_of_the_same_chunk(self):
        card = self._card_with_evidence_at(5)
        g = self._grounding([("B", 5)])
        self.deps.search = lambda *a, **k: []
        self._develop(card, g)
        self.assertEqual([p.key for p in g.passages.values() if p.text == "B-seq5"], ["C1"])

    def test_a_hit_whose_seq_equals_the_evidence_position_is_not_skipped(self):
        card = self._card_with_evidence_at(5)          # position 8
        g = self._grounding([])
        self.deps.search = lambda *a, **k: [lance_hit("B", 8), lance_hit("B", 5)]
        self._develop(card, g)
        transfer_prompt = next(p for p in self.prompts if "TASK: transfer" in p)
        self.assertIn("B-seq8", transfer_prompt)
        # The hit for the evidence chunk itself is not added a second time.
        self.assertEqual(sum(p.text == "B-seq5" for p in g.passages.values()), 1)


class TestSavedDataWithoutSeq(unittest.TestCase):
    def test_runs_and_cards_saved_before_seq_existed_still_load(self):
        d = result_to_dict(sample_result())
        for p in d["passages"].values():
            p.pop("seq")
        for b in d["bottlenecks"]:
            for p in b["premises"]:
                p.pop("seq")
        r = result_from_dict(d)
        self.assertTrue(all(p.seq is None for p in r.passages.values()))

        card = card_from_dict({
            "card_id": "x#0", "document": "x", "citation": "X", "name": "n", "problem_solved": "p",
            "context": "", "inputs": "", "outputs": "", "assumptions": [],
            "evidence": {"claim": "c", "key": "C1", "document": "x", "chunk_index": 4},
        })
        self.assertIsNone(card.evidence.seq)

    def test_old_card_evidence_is_read_as_the_position_it_always_was(self):
        # Cards were only ever built from enumerated passages, so a seq-less chunk_index is a position.
        texts, metas = lance_corpus([("A", 3), ("B", 7)])
        old = Premise(claim="c", key="C1", document="B", chunk_index=5)
        card = Card(card_id="B#0", document="B", citation="B", name="n", problem_solved="p", context="",
                    inputs="", outputs="", assumptions=[], evidence=old)
        deps, _ = world()
        deps.texts, deps.metadatas = texts, metas
        self.assertEqual(stages.resolve_card_evidence(deps, card)[1], "B-seq2")
        old.chunk_index = 1                             # now A's row: never re-read as B's seq 1
        self.assertIsNone(stages.resolve_card_evidence(deps, card))


if __name__ == "__main__":
    unittest.main()
