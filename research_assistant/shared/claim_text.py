# research_assistant/shared/claim_text.py
"""Turning a GROBID paragraph into judgeable claims.

Everything here is deterministic string work: which sentences a paragraph
has, where its citations sit, what the sentence says once the citation
marker is dealt with, whether the result is a claim at all, and what kind
of citation it is. None of it touches a model, so all of it is tested
against literal paragraphs.

Why tokens: the old placeholder embedded the citation's own text
("__CITE_1_b6_C.A.N. da Costa et al. 2018__"), so the sentence splitter cut
it at "C. " and half of it leaked into reports verbatim. "⟦C7⟧" contains
nothing any splitter treats as a boundary.

Why narrative citations are kept: "explored by Landa et al. (2006) with…"
loses its subject when the citation is deleted. A numeric marker or a
citation inside parentheses is punctuation and goes; an author–year
citation that is part of the sentence's grammar stays as text.
"""

import re

from research_assistant.agents.agent5_batch_citer import split_into_sentences

CITE_TOKEN_RE = re.compile(r"⟦C(\d+)⟧")

CLAIM_MIN_WORDS = 5
CLAIM_MAX_WORDS = 80

# Roles that are not claims about the cited paper's findings and are not judged.
SKIP_ROLES = frozenset({"software", "pointer", "method"})


def cite_token(n: int) -> str:
    return f"⟦C{n}⟧"


def clean_text(node) -> str:
    """Whitespace-collapsed text of a bs4 node or a string."""
    if node is None:
        return ""
    text = node.get_text() if hasattr(node, "get_text") else str(node)
    return " ".join(text.split()).strip()


def paragraph_sentences(p) -> list[str]:
    """GROBID's own <s> segmentation when the paragraph has it; the regex
    splitter otherwise. get_text() on a <p> glues "</s><s>" with no space,
    which is why the regex must never be the first choice."""
    s_tags = p.find_all("s")
    if s_tags:
        return [t for t in (clean_text(s) for s in s_tags) if t]
    return [t for t in split_into_sentences(clean_text(p)) if t]


_NUMERIC_CITE_RE = re.compile(r"^[\[\(]?\s*\d+[\d,;\s\-–—\]\)]*$")


def is_numeric_cite(txt: str) -> bool:
    """"[12]", "[9,", "10]", "3-5": a marker that carries no words."""
    return bool(_NUMERIC_CITE_RE.match((txt or "").strip()))


def is_parenthetical(sentence: str, token: str) -> bool:
    """Is *token* inside an unclosed ( or [ in *sentence*?"""
    before = sentence.split(token, 1)[0]
    depth = (before.count("(") - before.count(")")) + (before.count("[") - before.count("]"))
    return depth > 0


def tidy_punctuation(text: str) -> str:
    """Remove what a deleted citation leaves behind: "( ; )", "[ ]", doubled
    separators, a space before punctuation."""
    text = re.sub(r"[\(\[]\s*[;,\s]*[\)\]]", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(\s*[,;:])+", r"\1", text)
    return " ".join(text.split()).strip()


def render_claim(sentence: str, cites: dict[int, dict]) -> str:
    """The sentence as the judge should read it: numeric and parenthetical
    citations removed, narrative citations kept as their own text."""
    def _sub(match):
        n = int(match.group(1))
        txt = (cites.get(n) or {}).get("txt", "") or ""
        if is_numeric_cite(txt) or txt.startswith("(") or is_parenthetical(sentence, match.group(0)):
            return ""
        return txt
    return tidy_punctuation(CITE_TOKEN_RE.sub(_sub, sentence))


def render_sentence(sentence: str, cites: dict[int, dict]) -> str:
    """The sentence as the reader should see it: every citation in brackets."""
    def _sub(match):
        n = int(match.group(1))
        txt = (cites.get(n) or {}).get("txt", "") or ""
        return f"[{txt.strip('[]')}]" if txt else ""
    return " ".join(CITE_TOKEN_RE.sub(_sub, sentence).split())


_FRAGMENT_START_RE = re.compile(r"^[a-z\)\],;:]")


def claim_quality(claim: str) -> list[str]:
    """What is wrong with *claim* as a thing to judge. "placeholder_residue"
    and "fragment" mean it is not judged; "too_long" is judged and flagged."""
    issues = []
    if CITE_TOKEN_RE.search(claim) or "__CITE_" in claim:
        issues.append("placeholder_residue")
    words = claim.split()
    stripped = claim.lstrip()
    if len(words) < CLAIM_MIN_WORDS or _FRAGMENT_START_RE.match(stripped):
        issues.append("fragment")
    if len(words) > CLAIM_MAX_WORDS:
        issues.append("too_long")
    return issues


_SOFTWARE_REF_RE = re.compile(
    r"(users?'?\s+guide|user\s+manual|\bmanual\b|\bsoftware\b|\bpackage\b|\blibrary\b|"
    r"\btoolkit\b|\btoolbox\b|\balgorithm\s+\d+|\bversion\s+\d|\bv\d+\.\d|\bgithub\b|"
    r"\bzenodo\b|\bdocumentation\b)",
    re.I,
)
_SOFTWARE_CONTEXT_RE = re.compile(
    r"\b(package|code|software|library|toolkit|toolbox|implementation|solver|routine|"
    r"implemented (?:in|with|using)|written in|computed (?:with|using)|performed (?:with|using)|"
    r"calculated (?:with|using)|simulated (?:with|using))\b",
    re.I,
)
_POINTER_RE = re.compile(
    r"(\bsee\b|\bcf\.|\be\.g\.?,?|\bfor (?:a |an )?(?:review|details|introduction|overview|"
    r"discussion|derivation|proof|survey)\b|\bas (?:described|discussed|explained|reviewed|shown|"
    r"detailed|derived|outlined|summari[sz]ed) in\b|\breviewed in\b|\band references therein\b|"
    r"\bfor (?:instance|example)\b|\brefs?\.?\s*$)",
    re.I,
)
# A sentence that opens as a pointer is a pointer wherever its citation sits.
_POINTER_START_RE = re.compile(
    r"^(see\b|cf\.|for (?:a |an )?(?:review|details|introduction|overview|discussion|derivation|"
    r"proof|survey)\b|as (?:described|discussed|explained|reviewed|shown|detailed|derived|outlined) in\b|"
    r"(?:further|more) details\b|the reader is referred\b)",
    re.I,
)
_METHOD_RE = re.compile(
    r"\b(following|according to|as in|adapted from|adopted from|based on|"
    r"we (?:use|used|adopt|adopted|follow|followed|employ|employed|apply|applied))\b",
    re.I,
)


def classify_citation_role(sentence: str, token: str, ref_info: dict | None) -> str:
    """'software' | 'pointer' | 'method' | 'evidential', from the eight words
    before the token and the reference's title. Heuristic and deliberately
    conservative: an unrecognised citation is evidential and gets judged."""
    before = sentence.split(token, 1)[0]
    window = " ".join(before.split()[-8:])
    ref = ref_info or {}
    ref_text = f"{ref.get('title') or ''} {ref.get('raw_reference') or ''}"
    if _SOFTWARE_REF_RE.search(ref_text) or _SOFTWARE_CONTEXT_RE.search(window):
        return "software"
    if _POINTER_RE.search(window) or _POINTER_START_RE.match(sentence.lstrip()):
        return "pointer"
    if _METHOD_RE.search(window):
        return "method"
    return "evidential"


def sentence_context(sentences: list[str], idx: int) -> str:
    """Previous + «claim» + next, for the judge's context block."""
    parts = []
    if idx > 0:
        parts.append(sentences[idx - 1])
    parts.append(f"«{sentences[idx]}»")
    if idx + 1 < len(sentences):
        parts.append(sentences[idx + 1])
    return " ".join(parts)
