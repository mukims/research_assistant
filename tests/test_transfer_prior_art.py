import unittest

from research_assistant.transfer import prior_art


class TestPriorArt(unittest.TestCase):
    def test_openalex_abstracts_are_rebuilt_in_word_order(self):
        self.assertEqual(
            prior_art.reconstruct_abstract({"neural": [0], "quantum": [1], "states": [2, 4], "and": [3]}),
            "neural quantum states and states",
        )

    def test_hits_from_the_cutoff_year_or_later_are_ignored(self):
        oa = {
            "results": [
                {
                    "display_name": "Neural network quantum states",
                    "publication_year": 2017,
                    "doi": "d",
                    "abstract_inverted_index": {"restricted": [0], "boltzmann": [1], "machine": [2]},
                }
            ]
        }
        c = prior_art.PriorArtClient(
            lambda u, p: oa,
            lambda u, p: "<feed xmlns='http://www.w3.org/2005/Atom'/>",
            lambda texts: [[1.0, 0.0]] * len(texts),
        )
        self.assertEqual(c.check("restricted boltzmann machine", ["energy"], "s", before_year=2017).status, "not_found")
        self.assertEqual(c.check("restricted boltzmann machine", ["energy"], "s").status, "done_before")

    def test_a_similar_hit_without_the_technique_words_is_only_partial(self):
        oa = {
            "results": [
                {
                    "display_name": "Something else",
                    "publication_year": 2010,
                    "doi": "d",
                    "abstract_inverted_index": {"x": [0]},
                }
            ]
        }
        c = prior_art.PriorArtClient(
            lambda u, p: oa,
            lambda u, p: "<feed xmlns='http://www.w3.org/2005/Atom'/>",
            lambda texts: [[1.0, 0.0]] * len(texts),
        )
        self.assertEqual(c.check("chebyshev expansion", ["conductance"], "s").status, "partial")

    def test_a_network_failure_is_unchecked(self):
        def boom(u, p):
            raise ConnectionError("down")

        self.assertEqual(
            prior_art.PriorArtClient(boom, boom, lambda t: [[1.0]] * len(t)).check("x", [], "s").status,
            "unchecked",
        )
