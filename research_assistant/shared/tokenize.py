# research_assistant/shared/tokenize.py
"""One tokenizer for BM25, applied to documents at rebuild and to queries at
search time.

Why a version tag: the pickle records which tokenizer built it, and
hybrid_search picks the matching query tokenizer. A v1 pickle (bare
BM25Okapi, built with `\\w+`) keeps working with legacy_tokenize(); anything
rebuilt from now on uses tokenize().
"""

import re
import unicodedata

import snowballstemmer

TOKENIZER_VERSION = "v2"

_WORD = re.compile(r"\w+")
_STEMMER = snowballstemmer.stemmer("english")

# Small on purpose: physics prose is dense with content words, and an
# aggressive list would drop terms like "state" or "order" that matter here.
STOPWORDS = frozenset("""
a an and are as at be been but by for from has have in into is it its of on
or that the their there these this to was we were which with
""".split())


def normalize(text: str) -> str:
    """NFKC (ligatures → letters, full-width → ASCII) and collapsed whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text or "").split())


def legacy_tokenize(text: str) -> list[str]:
    """What rebuild_bm25() and hybrid_search() did before v2. Kept for v1 pickles."""
    return _WORD.findall((text or "").lower())


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(normalize(text).lower())
    kept = [w for w in words if len(w) > 1 and w not in STOPWORDS and not w.isdigit()]
    return _STEMMER.stemWords(kept)
