# tests/test_tokenize.py
"""The BM25 tokenizer must see the same word the reader sees.

31% of the v1 index's chunks contain a ligature-broken word (`ﬁeld`) that
`\\w+` indexes as a different token from `field`; 23% contain `quan- tum`.
NFKC fixes the first at rebuild time. Stemming makes `fluctuations` match
`fluctuation`. The same function must run on queries, or none of it helps.
"""

import unittest

from research_assistant.shared import tokenize as tk


class TestNormalize(unittest.TestCase):
    def test_ligatures_become_ascii(self):
        self.assertEqual(tk.normalize("ﬁeld eﬀect diﬀusion"), "field effect diffusion")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(tk.normalize("a  b\n\nc\t d"), "a b c d")


class TestTokenize(unittest.TestCase):
    def test_lowercases_stems_and_drops_stopwords(self):
        self.assertEqual(tk.tokenize("The Fluctuations of the fields"), ["fluctuat", "field"])

    def test_ligature_query_and_document_tokenise_identically(self):
        self.assertEqual(tk.tokenize("ﬁnite-size eﬀects"), tk.tokenize("finite-size effects"))

    def test_single_characters_and_digits_alone_are_dropped(self):
        # "t" and "1" carry nothing for keyword search; "MoS2" survives as one token.
        self.assertEqual(tk.tokenize("t = 1 for MoS2"), ["mos2"])

    def test_version_tag(self):
        self.assertEqual(tk.TOKENIZER_VERSION, "v2")

    def test_legacy_matches_the_old_behaviour(self):
        self.assertEqual(tk.legacy_tokenize("The ﬁeld"), ["the", "ﬁeld"])
