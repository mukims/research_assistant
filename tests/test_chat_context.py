# tests/test_chat_context.py
"""The mechanics of a chat turn, with no model and no index: what fits, what
is kept, how sources are keyed and checked, and how suggestions are parsed."""

import unittest

from research_assistant.shared import chat_context as cc


def _hit(i, doc, text="t", **meta):
    m = {"document": doc, "citation_source": f"Title of {doc}", "page": 2}
    m.update(meta)
    return {"chunk_index": i, "text": text, "metadata": m, "rrf_score": 0.1}


class TestTokens(unittest.TestCase):
    def test_estimate_rounds_up(self):
        self.assertEqual(cc.estimate_tokens(""), 0)
        self.assertEqual(cc.estimate_tokens("abcd"), 1)
        self.assertEqual(cc.estimate_tokens("abcde"), 2)

    def test_messages_tokens_sums_content(self):
        self.assertEqual(cc.messages_tokens([{"role": "a", "content": "abcd"}, {"role": "b", "content": "abcdefgh"}]), 3)


class TestCapAndMerge(unittest.TestCase):
    def test_cap_per_document_keeps_order_and_limit(self):
        hits = [_hit(1, "a"), _hit(2, "a"), _hit(3, "b"), _hit(4, "a"), _hit(5, "a")]
        self.assertEqual([h["chunk_index"] for h in cc.cap_per_document(hits, 2)], [1, 2, 3])

    def test_merge_puts_focus_first_and_dedupes(self):
        focus = [_hit(9, "a"), _hit(1, "a")]
        glob = [_hit(1, "a"), _hit(3, "b")]
        self.assertEqual([h["chunk_index"] for h in cc.merge_results(focus, glob)], [9, 1, 3])


class TestFormatContext(unittest.TestCase):
    def test_keys_headers_and_sources(self):
        text, sources = cc.format_context([_hit(1, "a.pdf", "AAA", section="results", page_first=4), _hit(2, "b.pdf", "BBB")], max_chars=1000)
        self.assertTrue(text.startswith("[S1] Title of a.pdf — p.4, results\nAAA\n\n[S2] Title of b.pdf — p.2, text\nBBB"))
        self.assertEqual([s["key"] for s in sources], ["S1", "S2"])
        self.assertEqual(sources[0]["key"], "S1")
        self.assertEqual(sources[0]["document"], "a.pdf")
        self.assertEqual(sources[0]["citation"], "Title of a.pdf")
        self.assertEqual(sources[0]["page"], 4)
        self.assertEqual(sources[0]["section"], "results")
        self.assertEqual(sources[0]["chunk_index"], 1)
        self.assertFalse(sources[0]["cited"])

    def test_cap_drops_whole_blocks_from_the_tail(self):
        hits = [_hit(1, "a", "A" * 100), _hit(2, "b", "B" * 100), _hit(3, "c", "C" * 100)]
        text, sources = cc.format_context(hits, max_chars=280)
        self.assertEqual([s["key"] for s in sources], ["S1", "S2"])
        self.assertNotIn("CCC", text)

    def test_first_block_is_never_dropped(self):
        text, sources = cc.format_context([_hit(1, "a", "A" * 500)], max_chars=100)
        self.assertEqual(len(sources), 1)
        self.assertLessEqual(len(text), 100)

    def test_empty(self):
        self.assertEqual(cc.format_context([], 100), ("", []))


class TestFitHistory(unittest.TestCase):
    def _hist(self, n_pairs, chars=40):
        h = []
        for i in range(n_pairs):
            h += [{"role": "user", "content": f"q{i}" + "x" * chars}, {"role": "assistant", "content": f"a{i}" + "y" * chars}]
        return h

    def test_keeps_the_most_recent_whole_pairs(self):
        h = self._hist(4)
        kept, dropped = cc.fit_history(h, max_tokens=50)
        self.assertEqual([m["content"][:2] for m in kept], ["q2", "a2", "q3", "a3"])
        self.assertEqual([m["content"][:2] for m in dropped], ["q0", "a0", "q1", "a1"])

    def test_everything_fits(self):
        h = self._hist(2)
        self.assertEqual(cc.fit_history(h, 10_000), (h, []))

    def test_nothing_fits(self):
        h = self._hist(2)
        self.assertEqual(cc.fit_history(h, 5), ([], h))


class TestKeys(unittest.TestCase):
    def test_cited_keys_in_order(self):
        self.assertEqual(cc.cited_keys("X [S2]. Y [S1, S3]. Z [S2] and [S9]."), ["S2", "S1", "S3", "S9"])

    def test_resolve_keys_uses_short_titles_and_leaves_unknown(self):
        sources = [{"key": "S1", "citation": "Charge transport in covalent MoS2 networks of nanosheets"},
                   {"key": "S2", "citation": "Anderson transitions"}]
        out = cc.resolve_keys("A [S1]. B [S1, S2]. C [S7].", sources)
        self.assertEqual(out, "A [Charge transport in covalent MoS2 networks]. B [Charge transport in covalent MoS2 networks; Anderson transitions]. C [S7].")

    def test_short_title(self):
        self.assertEqual(cc.short_title("one two three four five six seven eight"), "one two three four five six")
        self.assertEqual(cc.short_title(""), "untitled")


class TestParseSuggestions(unittest.TestCase):
    def test_parses_explicit_section(self):
        answer = """
Based on [S1], the hopping mechanism is dominant below 150K.

### 💡 Suggested Next Questions:
- How does chemical doping modify the hopping exponent?
- What are the primary experimental limitations of this measurement setup?
- Can this model be extended to bilayer MoS2?
"""
        suggestions = cc.parse_brainstorm_suggestions(answer)
        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions[0], "How does chemical doping modify the hopping exponent?")
        self.assertEqual(suggestions[1], "What are the primary experimental limitations of this measurement setup?")
        self.assertEqual(suggestions[2], "Can this model be extended to bilayer MoS2?")

    def test_parses_numbered_or_bold_suggestions(self):
        answer = """
Some findings [S1].

### Next Research Directions
1. **Doping concentration**: Does high doping suppress localization?
2. **Defect engineering**: How do sulfur vacancies alter hopping transport?
"""
        suggestions = cc.parse_brainstorm_suggestions(answer)
        self.assertEqual(len(suggestions), 2)
        self.assertIn("Does high doping suppress localization?", suggestions[0])
        self.assertIn("How do sulfur vacancies alter hopping transport?", suggestions[1])


if __name__ == "__main__":
    unittest.main()
