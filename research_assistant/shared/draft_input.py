"""Reading an uploaded draft into text.

The "Cite a draft" tab used to decode the upload inline, outside its try
block, so a Word-exported or Latin-1 file crashed the page with a
UnicodeDecodeError. Every way an upload can be unusable is a DraftInputError
here, with a message the reader can act on.

Streamlit-free: the tab shows the message, this module only decides it.
"""

import re

NOT_UTF8 = (
    "This file isn't UTF-8 text. Save it as UTF-8 (in Word: Save As → Plain Text → UTF-8) "
    "and upload again."
)
EMPTY = "The draft is empty."


class DraftInputError(ValueError):
    """The draft cannot be used; str() is the message to show the reader."""


def read_draft(name: str, data: bytes) -> str:
    """Decode an uploaded (or pasted) draft.

    *name* is the upload's file name. .txt and .md are both plain text and
    pass through unchanged; a .tex draft is reduced to its body text by
    tex_to_text. The name is not used to reject anything, so a draft saved
    under another extension still gets the same two checks.
    A UTF-8 byte-order mark (Notepad and Word add one) is dropped, because
    it would otherwise become the first character of the first sentence.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DraftInputError(NOT_UTF8) from None
    if name.lower().endswith(".tex"):
        text = tex_to_text(text)
    if not text.strip():
        raise DraftInputError(EMPTY)
    return text


# --- LaTeX ------------------------------------------------------------------

_DOCUMENT_BODY = re.compile(r"\\begin\{document\}(.*)\\end\{document\}", re.DOTALL)
# A whole-line comment goes with its newline, so it cannot split a paragraph.
_COMMENT_LINE = re.compile(r"^[ \t]*%[^\n]*\n?", re.MULTILINE)
# `\\\\` is consumed first so that `\\\\%` (a line break, then a comment) is not read as `\\%`.
_PERCENT_OR_COMMENT = re.compile(r"(\\\\)|\\%|%[^\n]*")
_DROPPED_ENVIRONMENT = re.compile(
    r"\\begin\{(equation|align|gather|multline|figure|table|tabular)(\*?)\}.*?\\end\{\1\2\}",
    re.DOTALL,
)
# The title may hold one level of braces (\section{On \emph{x}}); rule 6 unwraps them afterwards.
_SECTION = re.compile(r"\\(?:sub){0,2}section\*?\{((?:[^{}]|\{[^{}]*\})*)\}")
# One level of nested braces is allowed (\emph{the {BERT} model}); the inner braces are kept.
_EMPHASIS = re.compile(r"\\(?:emph|textbf|textit)\{((?:[^{}]|\{[^{}]*\})*)\}")
# Any \command with no brace after it. `\cite`, `\citep` and `\citet` are excluded by name, so
# they survive with or without an optional argument (\citep[p. 3]{k}); the (?![A-Za-z*{]) tail
# stops the regex backtracking into half a command name (`\noinden` + `t`) or a starred one.
_BRACELESS_COMMAND = re.compile(r"\\(?!cite[pt]?(?![A-Za-z]))[A-Za-z]+\*?(?![A-Za-z*{])")


def tex_to_text(source: str) -> str:
    """Reduce a LaTeX draft to the prose the citer should see.

    Existing \\cite, \\citep and \\citet commands are kept verbatim, so the citer's
    output is plain text with \\cite keys; it is not a compilable .tex file. The
    rules are applied in this order:

    1. keep only the text between \\begin{document} and \\end{document} (when both exist);
    2. drop % comments, keeping \\% as a literal %. Comments are removed before the
       body is cut out, so a commented \\begin{document} or \\end{document} is ignored;
    3. drop equation, align, gather, multline, figure, table and tabular environments
       (starred forms too);
    4. turn \\section, \\subsection and \\subsubsection into their title followed by a
       blank line (and set off from the paragraph before it);
    5. keep \\cite, \\citep and \\citet as they are;
    6. unwrap \\emph, \\textbf and \\textit;
    7. drop any other \\command that has no braces;
    8. collapse runs of spaces.
    """
    # Comments go first (rule 2 before rule 1): a commented-out "% \\begin{document}" in the
    # preamble, or "% \\end{document}" after the real one, must not be taken as the body's edge.
    text = _COMMENT_LINE.sub("", source)
    text = _PERCENT_OR_COMMENT.sub(lambda m: m.group(1) or ("%" if m.group(0) == "\\%" else ""), text)
    body = _DOCUMENT_BODY.search(text)
    if body:
        text = body.group(1)
    text = _DROPPED_ENVIRONMENT.sub("", text)
    # Blank line on both sides, so the title is a paragraph of its own; extra blanks collapse in rule 8.
    text = _SECTION.sub(lambda m: "\n\n" + m.group(1) + "\n\n", text)
    # Rule 5 needs no code: nothing below touches \cite, \citep or \citet.
    previous = None
    while previous != text:  # nested emphasis: \textit{\emph{x}}
        previous = text
        text = _EMPHASIS.sub(r"\1", text)
    text = _BRACELESS_COMMAND.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()
