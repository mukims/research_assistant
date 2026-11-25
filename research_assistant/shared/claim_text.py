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

# Abbreviations and initials that should NOT trigger a sentence break
_ABBREVS = r"(?:et al|Fig|Figs|Eq|Eqs|Dr|Prof|Mr|Mrs|Ms|Jr|Sr|vs|i\.e|e\.g|cf|approx|Ref|Refs|Vol|No|Ch|Sec|pp|[A-Za-z])"

def split_into_sentences(text):
    """
    Split text into sentences, handling common scientific abbreviations
    and author initials that contain periods (e.g., "et al.", "Fig.", "Eq.", "K.J.").
    """
    _TOKEN = "<PD>"
    def replace_abbrev_period(match):
        return match.group(0).replace('.', _TOKEN)

    # Mask periods in known abbreviations and author initials (case-insensitive)
    pattern = rf'\b({_ABBREVS})\.'
    masked_text = re.sub(pattern, replace_abbrev_period, text, flags=re.IGNORECASE)

    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', masked_text.strip())
    return [s.replace(_TOKEN, '.').strip() for s in sentences if s.strip()]

CITE_TOKEN_RE = re.compile(r"⟦C(\d+)⟧")

CLAIM_MIN_WORDS = 5
CLAIM_MAX_WORDS = 80

# Roles that are not claims about the cited paper's findings and are not judged.
SKIP_ROLES = frozenset({"software", "pointer", "method", "definition"})


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


_NUMERIC_CITE_RE = re.compile(r"^(?:refs?\.?|references?)?\s*[\[\(]?\s*\d+[\d,;\s\-–—\]\)]*$", re.IGNORECASE)
_ROMAN_CITE_RE = re.compile(r"^[\[\(]\s*[ivxlcdmIVXLCDM]+[\s,;\-–—ivxlcdmIVXLCDM]*[\]\)]$")

_DECIMAL_SECTION_OR_EQ_RE = re.compile(
    r"^\s*[\(\[]?\s*(?:eqs?\.?|sec(?:tion)?s?\.?|fig(?:ure)?s?\.?|thm\.?|lem\.?|app\.?)?\s*"
    r"\d+(?:\.\d+)+[a-z]?\s*[\)\]]?\s*$",
    re.IGNORECASE,
)
_BROKEN_DECIMAL_HALF_RE = re.compile(
    r"^(?:\s*[\(\[]\s*\d+\.\d*\s*|\s*\.\d+[a-z]?\s*[\)\]]\s*)$",
    re.IGNORECASE,
)
_INTERNAL_XREF_RE = re.compile(
    r"^\s*[\(\[]?\s*(?:chapter|chap\.?|section|sec\.?|appendix|app\.?|part|theorem|thm\.?|"
    r"lemma|lem\.?|proposition|prop\.?|corollary|cor\.?|exercise|ex\.?)\s+"
    r"[A-Z0-9]+(?:\.[A-Z0-9]+)*\s*[\)\]]?\s*$",
    re.IGNORECASE,
)
_CODE_ARRAY_OR_SLICE_RE = re.compile(
    r"(?:^\s*\[\s*-?\d*\s*:\s*-?\d*(?:\s*:\s*-?\d*)?\s*\]\s*$"
    r"|\btensor\s*\("
    r"|\[\s*\["
    r"|\d+\.\s*,"
    r"|,\s*\d+\.\s*[,\]]"
    r"|^\s*\[\s*0\s*[,\]]\s*$)",
    re.IGNORECASE,
)
_LIFESPAN_RE = re.compile(
    r"(?:\b1[5-9]\d{2}\s*[-–—]\s*(?:1[6-9]\d{2}|20\d{2})\b"
    r"|\(\s*[-–—]\s*(?:1[6-9]\d{2}|20\d{2})\s*\)\s*\)?$)"
)
_YEAR_ANY_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})[a-z]?\b")
_MULTI_AY_SPLIT_RE = re.compile(
    r"(?<=\b(?:1[89]\d{2}|20\d{2}))\s*[;,]\s*(?=[A-ZÀ-ÖØ-Þ])"
    r"|(?<=\b(?:1[89]\d{2}|20\d{2})[a-z])\s*[;,]\s*(?=[A-ZÀ-ÖØ-Þ])"
)


def is_non_citation_ref(txt: str, has_valid_target: bool = False) -> bool:
    """True when a GROBID ``<ref type="bibr">`` node is actually an equation/section
    cross-reference (``(1.3)``, ``1.3.6``, ``(1.191``, ``.198)``), a code token
    (``[__name__]``), a birth–death lifespan (``Fisher (1890-1962)``), or empty."""
    cleaned = (txt or "").strip()
    if not cleaned or not re.search(r"[A-Za-z0-9]", cleaned):
        return True
    if has_valid_target:
        return False
    if _DECIMAL_SECTION_OR_EQ_RE.match(cleaned) or _BROKEN_DECIMAL_HALF_RE.match(cleaned):
        return True
    if _INTERNAL_XREF_RE.match(cleaned) or _CODE_ARRAY_OR_SLICE_RE.search(cleaned):
        return True
    if _LIFESPAN_RE.search(cleaned):
        return True
    # Code / identifier tokens (underscores, scope/member operators) without a year
    if ("_" in cleaned or "::" in cleaned or "->" in cleaned) and not _YEAR_ANY_RE.search(cleaned):
        return True
    return False


def split_author_year_cites(txt: str) -> list[str]:
    """Split a multi-entry parenthetical author–year string into individual citations.

    Example: ``'(Kingma and Welling, 2014, Rezende et al., 2014)'``
    -> ``['(Kingma and Welling, 2014)', '(Rezende et al., 2014)']``.
    Single citations and numeric citations are returned as ``[txt]``.
    """
    cleaned = (txt or "").strip()
    if not cleaned or is_numeric_cite(cleaned):
        return [cleaned] if cleaned else []
    if len(_YEAR_ANY_RE.findall(cleaned)) < 2:
        return [cleaned]
    is_wrapped = (cleaned.startswith("(") and cleaned.endswith(")")) or (
        cleaned.startswith("[") and cleaned.endswith("]")
    )
    open_ch = cleaned[0] if is_wrapped else "("
    close_ch = cleaned[-1] if is_wrapped else ")"
    inner = cleaned[1:-1].strip() if is_wrapped else cleaned.strip("()[] ")
    parts = [p.strip(" ,;()[]") for p in _MULTI_AY_SPLIT_RE.split(inner) if p.strip(" ,;()[]")]
    if len(parts) <= 1 or not all(_YEAR_ANY_RE.search(p) for p in parts):
        return [cleaned]
    if is_wrapped or cleaned.startswith("(") or cleaned.endswith(")"):
        return [f"{open_ch}{p}{close_ch}" for p in parts]
    return parts


def is_numeric_cite(txt: str) -> bool:
    """"[12]", "[9,", "10]", "3-5", "Ref. 11", "[IV]": a marker that carries no author words."""
    cleaned = (txt or "").strip()
    return bool(_NUMERIC_CITE_RE.match(cleaned) or _ROMAN_CITE_RE.match(cleaned))


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
# "the conductance reads G = …", "T(E) is given by …", "is defined as …":
# a definition cites its textbook; there is no finding to verify. Needs an
# equation-like fragment in the sentence so that "the enhancement is given by
# the morphology" (a finding) is not swept up.
_DEFINITION_VERB_RE = re.compile(
    r"\b(reads|is given by|are given by|is defined as|are defined as|can be written as|"
    r"can be expressed as|takes the form|is expressed as|is written as)\b",
    re.I,
)
_EQUATION_RE = re.compile(r"(=|\bTr\[|∂|∫|Σ|\bd[A-Za-z]\(|\^|_\{|†)")
_METHOD_RE = re.compile(
    r"\b(following|according to|as in|adapted from|adopted from|based on|"
    r"we (?:use|used|adopt|adopted|follow|followed|employ|employed|apply|applied))\b",
    re.I,
)


def classify_citation_role(sentence: str, token: str, ref_info: dict | None) -> str:
    """'software' | 'pointer' | 'method' | 'definition' | 'evidential', from the eight words
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
    if _DEFINITION_VERB_RE.search(sentence) and _EQUATION_RE.search(sentence):
        return "definition"
    if _METHOD_RE.search(window):
        return "method"
    return "evidential"


_LEADING_TOKENS_RE = re.compile(r"^\s*((?:⟦C\d+⟧\s*)+)")


def reattach_leading_markers(sentences: list[str], cites: dict[int, dict]) -> list[str]:
    """GROBID sometimes closes a sentence before its trailing citation and
    opens the next with it: "…of the system.</s><s>[35] This small set…".
    A numeric marker before any word belongs to the sentence before it; a
    narrative author–year citation that opens a sentence ("Smith et al.
    (2020) showed…") is the sentence's subject and stays."""
    out: list[str] = []
    for sent in sentences:
        match = _LEADING_TOKENS_RE.match(sent)
        if match and out:
            tokens = CITE_TOKEN_RE.findall(match.group(1))
            movable = [n for n in tokens if is_numeric_cite((cites.get(int(n)) or {}).get("txt", ""))]
            if movable and len(movable) == len(tokens):
                moved = " ".join(cite_token(int(n)) for n in movable)
                prev = out[-1].rstrip()
                # Put the markers before the closing punctuation, as the author did.
                if prev and prev[-1] in ".!?":
                    out[-1] = f"{prev[:-1].rstrip()} {moved}{prev[-1]}"
                else:
                    out[-1] = f"{prev} {moved}"
                sent = sent[match.end():].lstrip()
        out.append(sent)
    return [x for x in out if x]


def sentence_context(sentences: list[str], idx: int) -> str:
    """Previous + «claim» + next, for the judge's context block."""
    parts = []
    if idx > 0:
        parts.append(sentences[idx - 1])
    parts.append(f"«{sentences[idx]}»")
    if idx + 1 < len(sentences):
        parts.append(sentences[idx + 1])
    return " ".join(parts)


_CLAUSE_SPLIT_RE = re.compile(
    r"(?:;\s+|:\s+|,\s+(?=(?:whereas|while|although|though|but|yet|however)\b))",
    re.IGNORECASE,
)
_RANGE_OR_LIST_CITE_RE = re.compile(r"\d+\s*[-–—,;]\s*\d+")


def render_focal_sentence(sentence: str, cites: dict[int, dict], target_n: int) -> str:
    """Render *sentence* marking *target_n* as ``[THIS CITATION: ...]`` and other citations in brackets."""
    def _sub(match):
        n = int(match.group(1))
        txt = ((cites.get(n) or {}).get("txt", "") or "").strip("[] ")
        if n == target_n:
            return f"[THIS CITATION: {txt or f'#{n}'}]"
        return f"[{txt}]" if txt else ""
    return " ".join(CITE_TOKEN_RE.sub(_sub, sentence).split())


def describe_citation_locus(sentence: str, target_n: int, cites: dict[int, dict]) -> dict:
    """Analyze how *target_n* sits within a potentially multi-citation *sentence*.

    Returns a dict with:
      - ``mode``: ``'single'`` | ``'multi_locus'`` | ``'cluster'``
      - ``focal_sentence``: rendered sentence with ``[THIS CITATION: ...]`` when multi/cluster
      - ``target_clause``: clean sub-clause or local phrase anchored by *target_n* (when ``multi_locus``)
      - ``attribution_note``: scoping guidance for the judge context block (None for ``single``)
    """
    matches = list(CITE_TOKEN_RE.finditer(sentence))
    txt_raw = ((cites.get(target_n) or {}).get("txt", "") or "").strip()
    txt_clean = txt_raw.strip("[] ") or f"#{target_n}"

    if len(matches) <= 1:
        is_range = bool(is_numeric_cite(txt_raw) and _RANGE_OR_LIST_CITE_RE.search(txt_raw))
        if not is_range:
            return {
                "mode": "single",
                "focal_sentence": render_sentence(sentence, cites),
                "target_clause": None,
                "attribution_note": None,
            }
        focal = render_focal_sentence(sentence, cites, target_n)
        note = (
            f"[Target Citation Attribution: [THIS CITATION: {txt_clean}] is part of a multi-reference "
            "range/group. If the sentence explicitly enumerates distinct examples (e.g. 'including A, B, and C'), "
            "evaluate the example addressed by this reference; otherwise evaluate the full claim scope as usual.]"
        )
        return {
            "mode": "cluster",
            "focal_sentence": focal,
            "target_clause": None,
            "attribution_note": note,
        }

    # Group consecutive tokens into contiguous citation clusters (loci)
    clusters: list[list[re.Match]] = [[matches[0]]]
    for m_prev, m_curr in zip(matches, matches[1:]):
        between = sentence[m_prev.end():m_curr.start()]
        stripped_between = re.sub(r"\b(?:and|or|refs?\.?)\b", "", between, flags=re.IGNORECASE)
        if not re.search(r"[A-Za-z0-9]{2,}", stripped_between):
            clusters[-1].append(m_curr)
        else:
            clusters.append([m_curr])

    focal = render_focal_sentence(sentence, cites, target_n)
    if len(clusters) == 1:
        total_in_cluster = len(clusters[0])
        note = (
            f"[Target Citation Attribution: [THIS CITATION: {txt_clean}] is 1 of {total_in_cluster} "
            "references cited together at this position. If the sentence explicitly enumerates distinct "
            "examples (e.g. 'such as A, B, and C'), evaluate the example addressed by this reference; "
            "otherwise evaluate the full claim scope as usual.]"
        )
        return {
            "mode": "cluster",
            "focal_sentence": focal,
            "target_clause": None,
            "attribution_note": note,
        }

    # Multiple distinct citation positions across the sentence: isolate the target clause/phrase
    tok_str = cite_token(target_n)
    clause_candidates = _CLAUSE_SPLIT_RE.split(sentence)
    chosen_seg = next((seg for seg in clause_candidates if tok_str in seg), sentence)

    # Check if chosen_seg still holds multiple distinct clusters (e.g. "in imaging [2] and seismology [3]")
    seg_matches = list(CITE_TOKEN_RE.finditer(chosen_seg))
    seg_clusters: list[list[re.Match]] = []
    for sm in seg_matches:
        if not seg_clusters:
            seg_clusters.append([sm])
        else:
            btw = chosen_seg[seg_clusters[-1][-1].end():sm.start()]
            btw_s = re.sub(r"\b(?:and|or|refs?\.?)\b", "", btw, flags=re.IGNORECASE)
            if not re.search(r"[A-Za-z0-9]{2,}", btw_s):
                seg_clusters[-1].append(sm)
            else:
                seg_clusters.append([sm])

    local_phrase = render_claim(chosen_seg, cites).strip(" ,;.")
    if len(seg_clusters) >= 2:
        t_cidx = next(
            (ci for ci, cl in enumerate(seg_clusters) if any(int(m.group(1)) == target_n for m in cl)),
            0,
        )
        start_pos = seg_clusters[t_cidx - 1][-1].end() if t_cidx > 0 else 0
        # If the citation sits at the end of its item (postfix "[2]"), cut right after its cluster;
        # if it is a narrative subject near the start ("Büttiker [2] showed..."), extend to next cluster.
        before_in_item = render_claim(chosen_seg[start_pos:seg_clusters[t_cidx][0].start()], cites).strip(" ,;.")
        if len(before_in_item.split()) >= 2:
            end_pos = seg_clusters[t_cidx][-1].end()
        else:
            end_pos = seg_clusters[t_cidx + 1][0].start() if t_cidx + 1 < len(seg_clusters) else len(chosen_seg)
        cand_phrase = render_claim(chosen_seg[start_pos:end_pos], cites)
        cand_phrase = re.sub(r"^\s*(?:,\s*)?(?:and|or|as well as)\s+", "", cand_phrase, flags=re.IGNORECASE).strip(" ,;.")
        if len(cand_phrase.split()) >= 2:
            local_phrase = cand_phrase

    note = (
        f"[Target Citation Attribution: This sentence cites references at {len(clusters)} distinct "
        f"positions. Evaluate ONLY the sub-clause or item anchored by [THIS CITATION: {txt_clean}] "
        f"(\"{local_phrase}\"), not sibling clauses or items anchored by other citations.]"
    )
    return {
        "mode": "multi_locus",
        "focal_sentence": focal,
        "target_clause": local_phrase or None,
        "attribution_note": note,
    }


# ─── Citation key extraction & punctuation restoration ───────────────────────

_CITE_RE = re.compile(r"\\cite[pt]?(?:\[[^\]]*\])?\{([^}]*)\}")

# One or more \cite / \citep / \citet groups at the very end of a sentence,
# with the whitespace before them.
_TRAILING_CITES_RE = re.compile(r"(?:\s*\\cite[pt]?(?:\[[^\]]*\])?\{[^}]*\})+\s*$")
_TERMINAL = ".!?"


def _restore_terminal_punctuation(original: str, cited: str) -> str:
    r"""Give the cited sentence back the full stop the model dropped.

    gemma4:e2b writes "… films \cite{cite_1}" for "… films." — no terminal
    punctuation. run_batch_citer joins sentences with a space, and Agent 8's
    split_into_sentences needs [.!?] before whitespace, so the next sentence
    is swallowed into this one and both citations get judged against a
    two-sentence claim. Restore the original's terminator after the cite.

    Left alone: a cited sentence that already ends in [.!?], or that ends in
    [.!?] immediately before a trailing \cite{} group ("… films. \cite{a}"),
    and any original that had no terminator to restore.
    """
    original = original.rstrip()
    if not original or original[-1] not in _TERMINAL:
        return cited
    stripped = cited.rstrip()
    if stripped and stripped[-1] in _TERMINAL:
        return stripped
    core = _TRAILING_CITES_RE.sub("", stripped)
    if core and core[-1] in _TERMINAL:
        return stripped
    return stripped + original[-1]


def _cite_keys(text: str) -> set[str]:
    """Return the set of citation keys actually present in *text*.

    Handles the multi-key form ``\\cite{a,b}`` as well as ``\\cite{a}``.
    An empty set means no citation was made, whatever the model replied.
    """
    keys = set()
    for match in _CITE_RE.finditer(text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


cite_keys = _cite_keys
restore_terminal_punctuation = _restore_terminal_punctuation

