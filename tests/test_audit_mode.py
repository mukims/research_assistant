"""The audit form has two explicit buttons; how many PDFs were staged decides what each one does."""

import unittest

from tests.app_harness import HAVE_STREAMLIT


@unittest.skipUnless(HAVE_STREAMLIT, "needs streamlit")
class TestAuditMode(unittest.TestCase):
    def test_audit_mode(self):
        from research_assistant.ui.audit_tab import _audit_mode

        self.assertEqual(_audit_mode(1, "audit"), "audit")
        self.assertEqual(_audit_mode(2, "audit"), "error")
        self.assertEqual(_audit_mode(0, "audit"), "error")
        self.assertEqual(_audit_mode(3, "library"), "library")

    def test_library_takes_a_single_pdf_too(self):
        from research_assistant.ui.audit_tab import _audit_mode

        self.assertEqual(_audit_mode(1, "library"), "library")


if __name__ == "__main__":
    unittest.main()
