# tests/test_draft_input.py
"""Reading an uploaded draft: any failure is a readable DraftInputError, never a traceback."""

import unittest

from research_assistant.shared.draft_input import DraftInputError, read_draft, tex_to_text


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


class TestTexToText(unittest.TestCase):
    def test_comments_dropped_escaped_percent_kept(self):
        self.assertEqual(tex_to_text("50\\% % note"), "50%")

    def test_equations_removed(self):
        src = "Before.\n\\begin{equation}\nE = mc^2\n\\end{equation}\nAfter."
        out = tex_to_text(src)
        self.assertNotIn("mc^2", out)
        self.assertNotIn("equation", out)
        self.assertIn("Before.", out)
        self.assertIn("After.", out)

    def test_existing_cites_kept(self):
        self.assertEqual(tex_to_text("Known \\cite{a,b}."), "Known \\cite{a,b}.")

    def test_section_titles_become_paragraphs(self):
        self.assertEqual(
            tex_to_text("\\section{Intro}\nText here.\n\\subsection{Aim}\nMore."),
            "Intro\n\nText here.\n\nAim\n\nMore.",
        )

    def test_preamble_ignored(self):
        src = (
            "\\documentclass{article}\n\\usepackage{amsmath}\n"
            "\\begin{document}\nBody text.\n\\end{document}\nTrailing."
        )
        self.assertEqual(tex_to_text(src), "Body text.")

    def test_kept_cites_survive_dropping_a_braceless_command(self):
        # Rule 7 must not eat \citep/\cite/\citet, with or without an optional argument.
        src = "\\noindent Water boils \\citep{w} and \\citet[p.~3]{x} and \\cite{y}."
        self.assertEqual(
            tex_to_text(src),
            "Water boils \\citep{w} and \\citet[p.~3]{x} and \\cite{y}.",
        )

    def test_emphasis_unwrapped_and_spaces_collapsed(self):
        self.assertEqual(
            tex_to_text("A \\emph{very}   \\textbf{bold} and \\textit{\\emph{nested}} claim."),
            "A very bold and nested claim.",
        )

    def test_starred_environments_and_tables_removed(self):
        src = "A.\n\\begin{align*}x&=1\\end{align*}\n\\begin{table}\\begin{tabular}{c}1\\end{tabular}\\end{table}\nB."
        out = tex_to_text(src)
        self.assertNotIn("x&=1", out)
        self.assertNotIn("tabular", out)
        self.assertEqual(out.split(), ["A.", "B."])

    def test_comment_only_line_does_not_split_a_paragraph(self):
        self.assertEqual(tex_to_text("One\n% hidden\ntwo."), "One\ntwo.")


class TestReadTexDraft(unittest.TestCase):
    def test_tex_extension_is_converted(self):
        data = b"\\begin{document}\n\\section{Intro}\nClaim \\cite{k}. % private\n\\end{document}"
        self.assertEqual(read_draft("paper.TEX", data), "Intro\n\nClaim \\cite{k}.")

    def test_txt_is_not_converted(self):
        self.assertEqual(read_draft("d.txt", b"\\section{X} 50\\%"), "\\section{X} 50\\%")

    def test_tex_with_no_body_text_is_empty(self):
        data = b"\\begin{document}\n% only a comment\n\\begin{equation}x\\end{equation}\n\\end{document}"
        with self.assertRaisesRegex(DraftInputError, "The draft is empty."):
            read_draft("paper.tex", data)


if __name__ == "__main__":
    unittest.main()
