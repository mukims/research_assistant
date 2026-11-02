"""Reading an uploaded draft into text.

The "Cite a draft" tab used to decode the upload inline, outside its try
block, so a Word-exported or Latin-1 file crashed the page with a
UnicodeDecodeError. Every way an upload can be unusable is a DraftInputError
here, with a message the reader can act on.

Streamlit-free: the tab shows the message, this module only decides it.
"""

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
    pass through unchanged; the name is not used to reject anything, so a
    draft saved under another extension still gets the same two checks.
    A UTF-8 byte-order mark (Notepad and Word add one) is dropped, because
    it would otherwise become the first character of the first sentence.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DraftInputError(NOT_UTF8) from None
    if not text.strip():
        raise DraftInputError(EMPTY)
    return text
