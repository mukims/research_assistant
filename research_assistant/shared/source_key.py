import hashlib
import re
import unicodedata

# Words that vary between how a reference is printed and how a paper's own
# header states its title. Stripped before hashing so the two still agree.
_TITLE_NOISE = re.compile(
    r"\b(a|an|the|of|for|and|on|in|to|with|via|using|towards|toward)\b"
)


def _fold(text):
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _fold_title(title):
    folded = _fold(title)
    folded = _TITLE_NOISE.sub(" ", folded)
    return " ".join(folded.split())


fold_title = _fold_title
fold_name = _fold


def _hash(prefix, payload, length=16):
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


_ARXIV_ID_PATTERNS = (
    re.compile(r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.IGNORECASE),
    re.compile(
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+(?:\.[a-z]{2})?/[0-9]{7}(?:v\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\barxiv\s*(?:preprint\s*(?:arxiv\s*)?)?[:/\s]\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+(?:\.[a-z]{2})?/[0-9]{7}(?:v\d+)?)\b",
        re.IGNORECASE,
    ),
)

_EMBEDDED_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>\]\}]+)", re.IGNORECASE)
_DOI_TRAILING_JUNK_RE = re.compile(r"(?:/mm\d+|(?<=\d)(?:website|webpage|online|html|pdf))$", re.IGNORECASE)


def normalise_doi(doi):
    """Strip the common prefixes and trailing extraction noise so the same DOI folds to one string."""
    if not doi:
        return None
    doi = str(doi).strip().lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "dx.doi.org/",
        "doi:",
        "doi ",
    ):
        if doi.startswith(prefix):
            doi = doi[len(prefix):].strip()
    while doi and doi[-1] in ".,;:'\"\\]}>":
        doi = doi[:-1].strip()
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1].strip()
    doi = _DOI_TRAILING_JUNK_RE.sub("", doi).rstrip("./,;")
    return doi if doi.startswith("10.") and "/" in doi and len(doi.split("/", 1)[1]) >= 2 else None


def extract_arxiv_from_text(text: str | None) -> str | None:
    """Extract a canonical arXiv identifier from a reference string, DOI, or URL."""
    if not text:
        return None
    s = str(text)
    for pat in _ARXIV_ID_PATTERNS:
        m = pat.search(s)
        if m:
            aid = m.group(1).strip().rstrip(".,;:)")
            if aid.lower().endswith(".pdf"):
                aid = aid[:-4]
            return aid.lower()
    return None


def extract_doi_from_text(text: str | None) -> str | None:
    """Extract a normalised DOI from raw reference text when structured fields omit it."""
    if not text:
        return None
    for m in _EMBEDDED_DOI_RE.finditer(str(text)):
        cand = normalise_doi(m.group(1))
        if cand and not cand.startswith("10.48550/arxiv."):
            return cand
        if cand:
            return cand
    return None


def source_key(record, min_doi_confidence=("high", "medium")):
    """
    Return a deterministic key for one reference or ingested document.

    The prefix says where the key came from, so a consumer can tell an
    authoritative identifier from a derived one:

        doi:10.1039/d4nr01234a      resolved, trustworthy
        arxiv:2401.12345            resolved, trustworthy
        url:<hash>                  a specific retrievable resource
        title:<hash>                derived from title + first author + year
        raw:<hash>                  nothing parsed; last resort

    Same input always gives the same key, so runs are comparable.
    """
    doi = normalise_doi(record.get("doi"))
    confidence = record.get("doi_confidence")
    if doi and (confidence is None or confidence in min_doi_confidence):
        return f"doi:{doi}"

    arxiv = record.get("arxiv_id")
    if arxiv:
        return f"arxiv:{arxiv.strip().lower().replace('arxiv:', '')}"

    pmid = record.get("pmid")
    if pmid:
        return f"pmid:{str(pmid).strip()}"

    url = record.get("url")
    if url:
        cleaned = url.strip().lower().split("#")[0].rstrip("/")
        cleaned = re.sub(r"^https?://(www\.)?", "", cleaned)
        return _hash("url", cleaned)

    title = _fold_title(record.get("title"))
    if title:
        authors = record.get("authors") or []
        # Surname of the first author only — initials and middle names are
        # the parts most likely to differ between two parses of one paper.
        first_author = _fold(authors[0].split()[-1]) if authors else ""
        year = str(record.get("year") or "")
        return _hash("title", f"{title}|{first_author}|{year}")

    raw = _fold(record.get("raw_reference"))
    if raw:
        return _hash("raw", raw)

    return None


def is_authoritative(key):
    """True when the key came from a real identifier rather than a derived one."""
    return bool(key) and key.split(":", 1)[0] in ("doi", "arxiv", "pmid")


def pdf_key(pdf_path: str) -> str:
    """Return the normalised document key for pdf_path."""
    import os
    return os.path.basename(pdf_path).strip().replace(" ", "_").lower()