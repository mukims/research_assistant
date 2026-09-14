# tests/test_claim_text.py
"""The deterministic text work that turns a GROBID paragraph into claims.
Every case is a literal paragraph; no model, no index."""

import unittest

from bs4 import BeautifulSoup

from research_assistant.shared import claim_text as ct


def _p(xml: str):
    return BeautifulSoup(f'<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>{xml}</body></text></TEI>', "xml").find("p")


class TestParagraphSentences(unittest.TestCase):
    def test_grobid_sentence_tags_are_used_when_present(self):
        p = _p("<p><s>First sentence here.</s><s>Second sentence here.</s></p>")
        self.assertEqual(ct.paragraph_sentences(p), ["First sentence here.", "Second sentence here."])

    def test_glued_boundary_is_not_a_problem_with_sentence_tags(self):
        # get_text() would give "here.Second" — the tags are the boundary, not the whitespace.
        p = _p("<p><s>Ends here.</s><s>Second one.</s></p>")
        self.assertEqual(len(ct.paragraph_sentences(p)), 2)

    def test_regex_fallback_without_sentence_tags(self):
        p = _p("<p>First sentence here. Second sentence here.</p>")
        self.assertEqual(ct.paragraph_sentences(p), ["First sentence here.", "Second sentence here."])

    def test_tokens_inside_sentence_tags_survive(self):
        p = _p("<p><s>Shown by <ref type=\"bibr\">Landa et al. (2006)</ref> here.</s></p>")
        p.find("ref").replace_with(f" {ct.cite_token(0)} ")
        self.assertEqual(ct.paragraph_sentences(p), ["Shown by ⟦C0⟧ here."])


class TestNumericAndParenthetical(unittest.TestCase):
    def test_numeric_markers(self):
        for txt in ("[12]", "[9,", "10]", "3-5", "[13–20]", "7"):
            self.assertTrue(ct.is_numeric_cite(txt), txt)

    def test_author_year_is_not_numeric(self):
        for txt in ("Landa et al. (2006)", "Silva et al. 2021", "C.A.N. da Costa et al. 2018"):
            self.assertFalse(ct.is_numeric_cite(txt), txt)

    def test_parenthetical_detection(self):
        s = "Proposed by several groups (⟦C0⟧; ⟦C1⟧) and later by ⟦C2⟧ here."
        self.assertTrue(ct.is_parenthetical(s, "⟦C0⟧"))
        self.assertTrue(ct.is_parenthetical(s, "⟦C1⟧"))
        self.assertFalse(ct.is_parenthetical(s, "⟦C2⟧"))


class TestRenderClaim(unittest.TestCase):
    def test_narrative_citation_keeps_the_sentence_subject(self):
        cites = {0: {"txt": "Landa et al. (2006)"}}
        s = "A similar idea was explored by ⟦C0⟧ with a path-integral formulation."
        self.assertEqual(ct.render_claim(s, cites),
                         "A similar idea was explored by Landa et al. (2006) with a path-integral formulation.")

    def test_leading_narrative_citation_is_kept(self):
        cites = {0: {"txt": "Silva et al. (2021)"}}
        s = "Following ⟦C0⟧ , the computational time is described next."
        self.assertEqual(ct.render_claim(s, cites),
                         "Following Silva et al. (2021), the computational time is described next.")

    def test_parenthetical_cluster_is_removed_cleanly(self):
        cites = {0: {"txt": "Vasconcelos et al. 2017"}, 1: {"txt": "Cui et al. 2020"}}
        s = "Target-oriented methods have been proposed by several groups ( ⟦C0⟧ ; ⟦C1⟧ )."
        self.assertEqual(ct.render_claim(s, cites),
                         "Target-oriented methods have been proposed by several groups.")

    def test_numeric_citations_are_removed(self):
        cites = {0: {"txt": "[9,"}, 1: {"txt": "10]"}}
        s = "The Landauer conductance reads G = 2e2/h ⟦C0⟧ ⟦C1⟧ ."
        self.assertEqual(ct.render_claim(s, cites), "The Landauer conductance reads G = 2e2/h.")

    def test_display_sentence_brackets_the_citation(self):
        cites = {0: {"txt": "Landa et al. (2006)"}}
        self.assertEqual(ct.render_sentence("By ⟦C0⟧ here.", cites), "By [Landa et al. (2006)] here.")

    def test_tidy_punctuation_existing_behaviour(self):
        self.assertEqual(ct.tidy_punctuation("Experiments confirmed this [ ] ."), "Experiments confirmed this.")
        self.assertEqual(ct.tidy_punctuation("It was shown ( ) , that conductance varies ."),
                         "It was shown, that conductance varies.")
        self.assertEqual(ct.tidy_punctuation("groups (;; ) here"), "groups here")


class TestClaimQuality(unittest.TestCase):
    def test_good_claim_has_no_issues(self):
        self.assertEqual(ct.claim_quality("Graphene shows a linear dispersion near the Dirac point."), [])

    def test_placeholder_residue(self):
        self.assertIn("placeholder_residue", ct.claim_quality("Shown in ⟦C3⟧ and elsewhere in the paper."))
        self.assertIn("placeholder_residue", ct.claim_quality("Shown in __CITE_1_b6_C.A.N. elsewhere too."))

    def test_fragments(self):
        self.assertIn("fragment", ct.claim_quality("with a path-integral formulation of depth migration."))
        self.assertIn("fragment", ct.claim_quality(", the computational time for each method is described next:"))
        self.assertIn("fragment", ct.claim_quality("See SM for M."))

    def test_too_long(self):
        self.assertIn("too_long", ct.claim_quality("word " * 81))
        self.assertNotIn("too_long", ct.claim_quality("word " * 80))


class TestCitationRole(unittest.TestCase):
    def test_software_by_reference_title(self):
        ref = {"title": "LAPACK Users' Guide, 3rd edn"}
        self.assertEqual(ct.classify_citation_role("We call ⟦C0⟧ for dense operations.", "⟦C0⟧", ref), "software")
        ref = {"title": "Algorithm 832"}
        self.assertEqual(ct.classify_citation_role("We use ⟦C0⟧ for LU factorization.", "⟦C0⟧", ref), "software")

    def test_software_by_context(self):
        ref = {"title": "Julia: a fresh approach to numerical computing"}
        self.assertEqual(ct.classify_citation_role("The code is implemented in ⟦C0⟧ and run on CPUs.", "⟦C0⟧", ref), "software")

    def test_pointer(self):
        s = "Multi-terminal experiments (see, e.g, Ref. ⟦C0⟧ for an introduction) have played a role."
        self.assertEqual(ct.classify_citation_role(s, "⟦C0⟧", {"title": "Semiconductor Nanostructures"}), "pointer")
        self.assertEqual(ct.classify_citation_role("See ⟦C0⟧ for a review of the field.", "⟦C0⟧", {}), "pointer")

    def test_pointer_at_sentence_start_with_citation_far_away(self):
        s = "See SM for details, including a brief discussion on the choices available to extract TB parameters from DFT calculations ⟦C0⟧ ."
        self.assertEqual(ct.classify_citation_role(s, "⟦C0⟧", {"title": "Wannier90 as a community code"}), "pointer")
        s = "For a review of the multi-terminal formalism and its applications to graphene devices the reader is referred to ⟦C0⟧ ."
        self.assertEqual(ct.classify_citation_role(s, "⟦C0⟧", {}), "pointer")

    def test_method(self):
        s = "Following ⟦C0⟧ , we also apply a taper to the target gradient."
        self.assertEqual(ct.classify_citation_role(s, "⟦C0⟧", {"title": "Target-oriented inversion"}), "method")

    def test_evidential(self):
        s = "Büttiker ⟦C0⟧ has shown that R and V can be cast in terms of the conductance matrix."
        self.assertEqual(ct.classify_citation_role(s, "⟦C0⟧", {"title": "Four-terminal phase-coherent conductance"}), "evidential")
        s = "Multi-terminal experiments have played a key role in graphene ⟦C0⟧ ⟦C1⟧ ."
        self.assertEqual(ct.classify_citation_role(s, "⟦C1⟧", {"title": "Nanoscale direct mapping of noise"}), "evidential")


class TestSentenceContext(unittest.TestCase):
    def test_marks_claim_between_neighbours(self):
        s = ["One.", "Two.", "Three."]
        self.assertEqual(ct.sentence_context(s, 1), "One. «Two.» Three.")
        self.assertEqual(ct.sentence_context(s, 0), "«One.» Two.")
        self.assertEqual(ct.sentence_context(s, 2), "Two. «Three.»")


if __name__ == "__main__":
    unittest.main()
