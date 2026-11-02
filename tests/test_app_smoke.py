"""app.py itself: the page lays out and every tab renders.

app.py is only the skeleton now (sidebar, tab bar, one call per tab), so a wrong
name or signature there, say render_draft_tab() losing its `chunks` argument, is
invisible to the tests of the modules it calls. Streamlit's AppTest runs the real
script headlessly, out of process and on a scratch data directory (tests/app_harness.py).
"""

import unittest

from research_assistant.ui.layout import LAB_TABS, MAIN_TABS
from tests.app_harness import HAVE_STREAMLIT, run_page


@unittest.skipUnless(HAVE_STREAMLIT, "needs streamlit")
class TestAppRenders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = run_page()

    def test_the_page_renders_without_an_exception(self):
        self.assertEqual(self.page["exceptions"], [])

    def test_the_tabs_are_there_in_order(self):
        # The auditor leads; the planner and technique transfer sit under Labs. Nested
        # tabs (Corpus library's sources, the Labs pair) may appear in the flat list.
        self.assertEqual([t for t in self.page["tabs"] if t in MAIN_TABS], MAIN_TABS)
        self.assertTrue(set(LAB_TABS) <= set(self.page["tabs"]))

    def test_the_sidebar_has_its_four_sections(self):
        self.assertEqual(self.page["sidebar_subheaders"], ["Pipeline Status", "Corpus", "Backend", "Server & Services"])

    def test_the_title_is_shown(self):
        self.assertEqual(self.page["title"], ["🤖 Marvin the Citebot"])

    def test_the_corpus_size_reaches_the_tabs_that_need_it(self):
        # An empty corpus: the draft tab and the chat tab each say so. They get the count as an argument.
        self.assertEqual(self.page["empty_corpus_notices"], 2)

    def test_the_audit_form_has_two_explicit_buttons_and_no_audit_toggle(self):
        page = run_page(after='out["buttons"] = [b.label for b in at.button]\nout["toggles"] = [t.label for t in at.toggle]')
        self.assertIn("Audit this paper", page["buttons"])
        self.assertIn("Add papers to library", page["buttons"])
        self.assertNotIn("Audit Citations & Index Paper(s)", page["buttons"])
        self.assertNotIn("Audit citations", page["toggles"])


if __name__ == "__main__":
    unittest.main()
