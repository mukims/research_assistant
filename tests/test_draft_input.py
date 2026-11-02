# tests/test_draft_input.py
"""Reading an uploaded draft: any failure is a readable DraftInputError, never a traceback."""

import unittest

from research_assistant.shared.draft_input import DraftInputError, read_draft


class TestReadDraft(unittest.TestCase):
    def test_non_utf8_draft_gets_a_readable_error(self):
        with self.assertRaisesRegex(DraftInputError, "UTF-8") as ctx:
            read_draft("d.txt", "café".encode("latin-1"))
        self.assertIn("Save As", str(ctx.exception))

    def test_bom_is_stripped(self):
        self.assertEqual(read_draft("d.txt", b"\xef\xbb\xbfHello world."), "Hello world.")

    def test_markdown_passes_through(self):
        md = "# Title\n\nA *claim* with `code`.\n"
        self.assertEqual(read_draft("d.md", md.encode("utf-8")), md)

    def test_plain_text_passes_through_unchanged(self):
        txt = "  Leading space kept.\n\nSecond paragraph.\n"
        self.assertEqual(read_draft("d.txt", txt.encode("utf-8")), txt)

    def test_empty_draft_is_refused(self):
        for data in (b"", b"   \n\t ", b"\xef\xbb\xbf"):
            with self.subTest(data=data):
                with self.assertRaisesRegex(DraftInputError, "The draft is empty."):
                    read_draft("d.txt", data)

    def test_draft_input_error_is_a_value_error(self):
        self.assertTrue(issubclass(DraftInputError, ValueError))


if __name__ == "__main__":
    unittest.main()
