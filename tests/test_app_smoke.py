"""app.py itself: the page lays out and every tab renders.

app.py is only the skeleton now (sidebar, tab bar, one call per tab), so a wrong
name or signature there, say render_draft_tab() losing its `chunks` argument, is
invisible to the tests of the modules it calls. Streamlit's AppTest runs the real
script headlessly, out of process and on a scratch data directory (tests/app_harness.py).
"""

import unittest

from tests.app_harness import HAVE_STREAMLIT, run_page


@unittest.skipUnless(HAVE_STREAMLIT, "needs streamlit")
class TestAppRenders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = run_page()

    def test_the_page_renders_without_an_exception(self):
        self.assertEqual(self.page["exceptions"], [])

    def test_the_six_tabs_are_there_in_order(self):
        self.assertEqual(
            self.page["tabs"],
            ["Agent task planner", "Citation auditor", "Research idea", "Cite a draft", "Research chat", "How to use"],
        )

    def test_the_sidebar_has_its_four_sections(self):
        self.assertEqual(self.page["sidebar_subheaders"], ["Pipeline Status", "Corpus", "Backend", "Server & Services"])

    def test_the_title_is_shown(self):
        self.assertEqual(self.page["title"], ["🤖 Marvin the Citebot"])

    def test_the_corpus_size_reaches_the_tabs_that_need_it(self):
        # An empty corpus: the draft tab and the chat tab each say so. They get the count as an argument.
        self.assertEqual(self.page["empty_corpus_notices"], 2)


if __name__ == "__main__":
    unittest.main()
