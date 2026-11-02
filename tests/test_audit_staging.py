"""An upload reaches RAW_DIR only when the button pressed is valid for it.

agent1_extractor.run_extractor() processes every PDF in RAW_DIR, so PDFs left there by an
"Audit this paper" press with several files would be processed by the user's next audit.
"""

import os
import tempfile
import unittest

from tests.app_harness import HAVE_STREAMLIT

PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n%%EOF\n"


@unittest.skipUnless(HAVE_STREAMLIT, "needs streamlit")
class TestStageForAction(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.raw = os.path.join(tmp.name, "raw")
        os.makedirs(self.raw)
        with open(os.path.join(self.raw, "library.pdf"), "wb") as fh:
            fh.write(PDF + b"% already here\n")

    def _stage(self, uploads, action):
        from research_assistant.ui.audit_tab import _stage_for_action

        return _stage_for_action(uploads, action, raw_dir=self.raw)

    def _expected_names(self, uploads):
        from research_assistant.shared.batch_uploader import unpack_and_stage_uploads

        with tempfile.TemporaryDirectory() as d:
            return sorted(os.path.basename(r["path"]) for r in unpack_and_stage_uploads(uploads, destination_dir=d))

    def test_several_pdfs_pressed_as_an_audit_leave_nothing_in_raw_dir(self):
        uploads = [(PDF + b"% one\n", "one.pdf"), (PDF + b"% two\n", "two.pdf")]
        staged, mode = self._stage(uploads, "audit")
        self.assertEqual(mode, "error")
        self.assertEqual(len(staged), 2)
        self.assertEqual(os.listdir(self.raw), ["library.pdf"])
        self.assertFalse(any(os.path.exists(r["path"]) for r in staged))

    def test_a_rejected_upload_never_removes_a_library_paper_at_the_same_path(self):
        existing = self._expected_names([(PDF + b"% one\n", "one.pdf")])[0]
        with open(os.path.join(self.raw, existing), "wb") as fh:
            fh.write(PDF + b"% one\n")
        staged, mode = self._stage([(PDF + b"% one\n", "one.pdf"), (PDF + b"% two\n", "two.pdf")], "audit")
        self.assertEqual(mode, "error")
        self.assertEqual(sorted(os.listdir(self.raw)), sorted(["library.pdf", existing]))

    def test_library_uploads_land_in_raw_dir_under_their_staged_names(self):
        uploads = [(PDF + b"% one\n", "one.pdf"), (PDF + b"% two\n", "two.pdf")]
        staged, mode = self._stage(uploads, "library")
        self.assertEqual(mode, "library")
        names = self._expected_names(uploads)
        self.assertEqual(sorted(os.path.basename(r["path"]) for r in staged), names)
        self.assertTrue(all(os.path.dirname(r["path"]) == self.raw and os.path.isfile(r["path"]) for r in staged))
        self.assertEqual(sorted(os.listdir(self.raw)), sorted(names + ["library.pdf"]))

    def test_a_single_pdf_audit_lands_in_raw_dir(self):
        staged, mode = self._stage([(PDF + b"% one\n", "one.pdf")], "audit")
        self.assertEqual(mode, "audit")
        with open(staged[0]["path"], "rb") as fh:
            self.assertEqual(fh.read(), PDF + b"% one\n")
        self.assertEqual(os.path.dirname(staged[0]["path"]), self.raw)


if __name__ == "__main__":
    unittest.main()
