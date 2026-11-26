"""
Agent 2 — reference fetcher.

Takes the reference list Agent 1 extracted, resolves a DOI per reference
(trusting Agent 1 when its confidence was high/medium, otherwise asking
Crossref), then tries to download an open-access PDF through a four-source
ladder: Crossref (DOI resolution) -> Unpaywall -> Europe PMC -> arXiv.
Checkpoints downloaded.json / failed_downloads.json after every paper so a
crashed run resumes instead of starting over.

Successful downloads are written through schemas.DownloadedPaper — the typed
contract Agent 3 reads back. failed_downloads.json stays plain dicts: those
records carry a `reason` and no `path`, and have no downstream consumer
needing a contract.
"""

import html
import json
import os
import re
import requests
import time
import urllib.parse
import xml.etree.ElementTree as ET

from research_assistant.config import (
    EXTRACTED_CITATIONS_PATH,
    DOWNLOADED_JSON_PATH,
    FAILED_DOWNLOADS_PATH,
    PULLED_PDFS_DIR,
    UNPAYWALL_EMAIL,
    MAX_CITATION_LEN,
    ARXIV_RATE_LIMIT,
    UNPAYWALL_SLEEP,
    PAPER_INDEX_PATH,
)
from research_assistant.schemas import DownloadedPaper
from research_assistant.shared.log import get_logger
from research_assistant.shared.paper_index import PaperIndex
from research_assistant.shared.dedupe_sweep import run_sweep
from research_assistant.shared.source_key import (
    extract_arxiv_from_text,
    extract_doi_from_text,
    is_authoritative,
    normalise_doi,
    source_key,
)
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.fetch import HEADERS, candidate_pdf_urls, download_pdf, filename_for, is_saved_pdf
from research_assistant.shared import pipeline_status
from research_assistant.shared import academic_apis


def fetch_bibliometrics_and_abstract(*args, **kwargs):
    return academic_apis.fetch_bibliometrics_and_abstract(*args, **kwargs)


logger = get_logger("agent2")

TRUSTED_CONFIDENCE = ("high", "medium")


# ─── Retry policy ────────────────────────────────────────────────────────────
# A failed key used to be excluded from the work list forever, which made a
# ten-second network blip permanently drop those references from the corpus.
# These markers separate "the answer might differ next run" from "this paper is
# settled" — a paywall, a 404 or an absence from an index is a real answer and
# stays cached; a dropped connection, a rate limit or a 5xx is not.
_TRANSIENT_MARKERS = (
    "download error:",
    "incomplete download",
    "unpaywall error:",
    "europe pmc error:",
    "arxiv error:",
    "empty response",
    "malformed json",
)
_TRANSIENT_HTTP = re.compile(r"http (?:429|5\d\d)")


def _is_retryable(reason: str | None) -> bool:
    """True when *reason* describes a fault that could clear on its own.

    `reason` is every rung of the download ladder joined together, so one
    transient leg is enough: the paper may well be reachable next time.
    """
    text = (reason or "").lower()
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return True
    if UNPAYWALL_EMAIL and "unpaywall skipped" in text:
        return True
    return bool(_TRANSIENT_HTTP.search(text))


def _checkpoint(downloaded, failed):
    """Write state to disk after every paper — crash-safe incremental saves.

    Atomic because the checkpoint is the resume point: _load_state() treats an
    unparseable file as empty and starts the whole fetch again, so a truncated
    downloaded.json costs more than no checkpoint at all.
    """
    atomic_write_json(DOWNLOADED_JSON_PATH, downloaded, ensure_ascii=False)
    atomic_write_json(FAILED_DOWNLOADS_PATH, failed, ensure_ascii=False)


def _load_state(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("%s is unreadable; starting fresh.", path)
        return default


def _load_references():
    """Read agent1's output, tolerating the old flat-string format."""
    with open(EXTRACTED_CITATIONS_PATH, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return payload.get("references", [])

    # Legacy format: a list of raw citation strings.
    logger.warning("Legacy citation format detected; only raw strings available.")
    return [{"raw_reference": c, "title": None, "doi": None, "authors": []} for c in payload]


def resolve_doi(ref):
    """
    Return (doi, how). Trust agent1's DOI when it was confidently matched;
    otherwise ask Crossref using the parsed title, checking top candidates.
    """
    doi = normalise_doi(ref.get("doi")) or extract_doi_from_text(ref.get("raw_reference"))
    if doi and ref.get("doi_confidence") in TRUSTED_CONFIDENCE:
        return doi, "grobid"

    from research_assistant.shared.academic_apis import (
        _match_surnames,
        clean_reference_title,
        verify_candidate_match,
    )

    raw_title = ref.get("title")
    cleaned_title = clean_reference_title(raw_title) if raw_title else None
    query = cleaned_title or raw_title or ref.get("raw_reference")
    if not query:
        return None, None

    params = {"query.bibliographic": query, "rows": 5, "select": "DOI,title,author,issued"}
    if ref.get("authors"):
        params["query.author"] = ref["authors"][0]

    try:
        res = requests.get(
            "https://api.crossref.org/works", params=params, headers=HEADERS, timeout=15
        )
        res.raise_for_status()
        items = res.json()["message"]["items"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.warning("Crossref lookup failed: %s", exc)
        return (doi, "grobid-unverified") if doi else (None, None)

    if not items:
        return (doi, "grobid-unverified") if doi else (None, None)

    target_title = raw_title
    target_author = ref.get("authors")[0] if ref.get("authors") else None
    target_year = ref.get("year")

    for cand in items:
        cand_titles = cand.get("title", [])
        cand_title = cand_titles[0] if isinstance(cand_titles, list) and cand_titles else str(cand_titles or "")
        cand_authors_raw = cand.get("author", [])
        cand_all_authors = []
        if cand_authors_raw and isinstance(cand_authors_raw, list):
            for a_obj in cand_authors_raw:
                if isinstance(a_obj, dict):
                    nm = a_obj.get("family") or a_obj.get("name")
                    if nm:
                        cand_all_authors.append(nm)
        cand_author = cand_all_authors[0] if cand_all_authors else None
        issued = cand.get("issued", {}).get("date-parts", [[None]])
        cand_year = issued[0][0] if issued and issued[0] else None

        if target_title:
            if verify_candidate_match(
                target_title,
                cand_title,
                target_author,
                cand_author,
                target_year,
                cand_year,
                candidate_authors=cand_all_authors,
            ):
                resolved = normalise_doi(cand.get("DOI"))
                if resolved:
                    return resolved, "crossref"
        else:
            # Titleless bibliographic string (e.g. APS brief citation): verify author & year if present
            if target_author and cand_author and not _match_surnames(target_author, cand_author):
                continue
            if target_year is not None and cand_year is not None:
                try:
                    if abs(int(target_year) - int(cand_year)) > 1:
                        continue
                except (ValueError, TypeError):
                    pass
            resolved = normalise_doi(cand.get("DOI"))
            if resolved:
                return resolved, "crossref"

    logger.info(
        "No Crossref candidate passed similarity check against target '%s'",
        (target_title or query)[:40],
    )
    return (doi, "grobid-unverified") if doi else (None, None)



def try_unpaywall(doi, dest_path):
    """
    Look for an open-access PDF for this DOI.

    Unpaywall's best_oa_location is usually the publisher's own site, which
    is also the one most likely to block an automated request. Every known
    location is tried, repositories first.

    Unpaywall refuses requests without a contact email, so with UNPAYWALL_EMAIL
    unset this makes no request at all.
    """
    if not UNPAYWALL_EMAIL:
        return False, "Unpaywall skipped: set UNPAYWALL_EMAIL"
    try:
        res = requests.get(
            f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}",
            params={"email": UNPAYWALL_EMAIL},
            headers=HEADERS,
            timeout=15,
        )
    except requests.RequestException as exc:
        return False, f"Unpaywall error: {exc}"

    if res.status_code != 200:
        return False, f"Unpaywall lookup failed (HTTP {res.status_code})"

    try:
        data = res.json()
    except ValueError:
        return False, "Unpaywall returned malformed JSON"

    if not data.get("is_oa"):
        return False, "paywalled / not open access"

    # Collect every location with a PDF, best_oa_location included, then
    # de-duplicate on URL since it usually repeats one of the others.
    locations = list(data.get("oa_locations") or [])
    best = data.get("best_oa_location")
    if best:
        locations.append(best)

    candidates = []
    seen = set()
    for loc in locations:
        raw_url = (loc or {}).get("url_for_pdf") or (loc or {}).get("url")
        for url in candidate_pdf_urls(raw_url):
            if url and url not in seen:
                seen.add(url)
                candidates.append({"url": url, "host_type": (loc or {}).get("host_type") or "?"})

    if not candidates:
        return False, "open access, but no direct PDF URL"

    candidates.sort(key=lambda c: c["host_type"] != "repository")

    reasons = []
    for candidate in candidates:
        ok, reason = download_pdf(candidate["url"], dest_path)
        if ok:
            return True, None
        reasons.append(f"{candidate['host_type']}: {reason}")

    return False, "Unpaywall PDF failed: " + "; ".join(reasons)


def try_europepmc(doi, ref, dest_path):
    """Europe PMC covers biomedical and open-access repository literature."""
    from research_assistant.shared.academic_apis import clean_reference_title, verify_candidate_match

    pmid = ref.get("pmid")
    clean_title = clean_reference_title(ref.get("title"))
    if doi:
        query = f'DOI:"{doi}"'
    elif pmid:
        query = f'EXT_ID:"{pmid}" AND SRC:MED'
    elif clean_title:
        query = f'TITLE:"{clean_title}"'
    else:
        return False, "not in Europe PMC"

    try:
        res = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": 3, "resultType": "core"},
            headers=HEADERS,
            timeout=20,
        )
        res.raise_for_status()
        results = res.json().get("resultList", {}).get("result", [])
    except (requests.RequestException, ValueError) as exc:
        return False, f"Europe PMC error: {exc}"

    if not results:
        return False, "not in Europe PMC"

    target_title = ref.get("title")
    target_author = ref.get("authors")[0] if ref.get("authors") else None
    target_year = ref.get("year")

    hit = None
    for cand in results:
        if doi or pmid:
            hit = cand
            break
        cand_title = cand.get("title") or ""
        cand_yr = cand.get("pubYear")
        cand_author = None
        auth_list = (cand.get("authorList") or {}).get("author") or []
        cand_all_authors = [
            a.get("lastName") or a.get("fullName")
            for a in auth_list
            if isinstance(a, dict) and (a.get("lastName") or a.get("fullName"))
        ]
        if cand_all_authors:
            cand_author = cand_all_authors[0]
        if target_title and verify_candidate_match(
            target_title,
            cand_title,
            target_author,
            cand_author,
            target_year,
            cand_yr,
            candidate_authors=cand_all_authors,
        ):
            hit = cand
            break

    if not hit:
        return False, "not in Europe PMC"

    urls_to_try = []
    pmcid = hit.get("pmcid")
    if pmcid and hit.get("isOpenAccess") == "Y":
        urls_to_try.append(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextPdf")

    ft_list = (hit.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    for ft in ft_list:
        if not isinstance(ft, dict):
            continue
        style = (ft.get("documentStyle") or "").lower()
        avail = (ft.get("availabilityCode") or "").upper()
        u = ft.get("url")
        if u and style == "pdf" and avail in ("OA", "F"):
            for cand_u in candidate_pdf_urls(u):
                if cand_u not in urls_to_try:
                    urls_to_try.append(cand_u)

    if not urls_to_try:
        return False, "in Europe PMC but not open access"

    last_reason = "download failed"
    for url in urls_to_try:
        ok, reason = download_pdf(url, dest_path)
        if ok:
            return True, None
        last_reason = reason or last_reason
    return False, f"Europe PMC PDF failed: {last_reason}"


# Inline formatting is part of the word — "WS<sub>2</sub>" is WS2 on arXiv, not
# "WS 2" — so those tags close up. Everything else stands between two words.
_INLINE_TAG_RE = re.compile(r"</?(?:sub|sup|i|b|em|strong|italic|bold|sc|span)\b[^>]*>", re.I)
_MARKUP_RE = re.compile(r"<[^>]+>")
_ARXIV_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "using", "via", "under", "over", "their", "this", "that",
})


def arxiv_query_title(title) -> str:
    """A title arXiv's phrase index can match."""
    text = _INLINE_TAG_RE.sub("", str(title or ""))
    text = _MARKUP_RE.sub(" ", text)
    text = html.unescape(text).replace('"', " ")
    return " ".join(text.split())


def try_arxiv(ref, dest_path):
    """Fall back to arXiv, searching on explicit arxiv_id, exact title variants, or keyword+author."""
    from research_assistant.shared.academic_apis import (
        _extract_author_surname,
        title_variants,
        verify_candidate_match,
    )

    arxiv_id = (
        ref.get("arxiv_id")
        or extract_arxiv_from_text(ref.get("doi"))
        or extract_arxiv_from_text(ref.get("url"))
        or extract_arxiv_from_text(ref.get("raw_reference"))
    )
    title = arxiv_query_title(ref.get("title"))
    raw = ref.get("raw_reference")
    if not (arxiv_id or title or raw):
        return False, "nothing to search arXiv with"

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    target_title = ref.get("title")
    target_author = ref.get("authors")[0] if ref.get("authors") else None
    target_year = ref.get("year")

    def _query_entries(url: str):
        res = requests.get(url, headers=HEADERS, timeout=20)
        res.raise_for_status()
        root = ET.fromstring(res.content)
        return root.findall("atom:entry", ns)

    urls = []
    if arxiv_id:
        clean_aid = re.sub(r"^arxiv:", "", str(arxiv_id).strip(), flags=re.IGNORECASE)
        urls.append(
            "https://export.arxiv.org/api/query?id_list="
            f"{urllib.parse.quote(clean_aid)}&max_results=2"
        )
    elif title:
        variants = title_variants(title) or [title]
        for v in variants:
            q_str = f'ti:"{arxiv_query_title(v)}"'
            u = (
                "https://export.arxiv.org/api/query?search_query="
                f"{urllib.parse.quote(q_str)}&max_results=5"
            )
            if u not in urls:
                urls.append(u)
        content_words = [
            w for w in re.findall(r"[A-Za-z0-9]{3,}", variants[0])
            if w.lower() not in _ARXIV_STOPWORDS and not w.isdigit()
        ]
        surname = _extract_author_surname(target_author) if target_author else ""
        if len(content_words) >= 4 and surname:
            ti_clause = " AND ".join(f"ti:{w}" for w in content_words[:7])
            kw_query = f"({ti_clause}) AND au:{surname}"
            urls.append(
                "https://export.arxiv.org/api/query?search_query="
                f"{urllib.parse.quote(kw_query)}&max_results=5"
            )
    else:
        query = f'all:"{raw[:200]}"'
        urls.append(
            "https://export.arxiv.org/api/query?search_query="
            f"{urllib.parse.quote(query)}&max_results=3"
        )

    saw_any_entry = False
    matched_entry = None
    for idx_u, url in enumerate(urls):
        try:
            entries = _query_entries(url)
        except (requests.RequestException, ET.ParseError) as exc:
            if idx_u == 0:
                return False, f"arXiv error: {exc}"
            continue
        if not entries:
            continue
        saw_any_entry = True
        for entry in entries:
            cand_title = " ".join((entry.findtext("atom:title", "", ns)).split())
            cand_authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            cand_author = cand_authors[0] if cand_authors else None
            pub = entry.findtext("atom:published", "", ns)
            cand_year = int(pub[:4]) if pub and pub[:4].isdigit() else None
            arxiv_doi = normalise_doi(entry.findtext("{http://arxiv.org/schemas/atom}doi", "", ns))
            ref_doi = normalise_doi(ref.get("doi"))

            if ref_doi and arxiv_doi and ref_doi == arxiv_doi:
                matched_entry = entry
                break
            if arxiv_id and not target_title:
                matched_entry = entry
                break
            if target_title:
                # arXiv preprints are routinely posted 1-3 years before journal publication
                eff_max_yr = 1
                if target_year is not None and cand_year is not None:
                    try:
                        if 0 <= int(target_year) - int(cand_year) <= 3:
                            eff_max_yr = 3
                    except (TypeError, ValueError):
                        pass
                if verify_candidate_match(
                    target_title,
                    cand_title,
                    target_author,
                    cand_author,
                    target_year,
                    cand_year,
                    candidate_authors=cand_authors,
                    max_year_diff=eff_max_yr,
                ):
                    matched_entry = entry
                    break
            elif arxiv_id:
                matched_entry = entry
                break
        if matched_entry is not None:
            break

    if not saw_any_entry:
        return False, "not found on arXiv"
    if matched_entry is None:
        if not target_title and not arxiv_id:
            return False, "arXiv match lacked title/ID verification"
        return False, "arXiv match was a different paper (similarity verification failed)"

    pdf_link = None
    for link in matched_entry.findall("atom:link", ns):
        if link.attrib.get("title") == "pdf":
            pdf_link = link.attrib.get("href", "").replace("http://", "https://")
            break
    if not pdf_link and arxiv_id:
        clean_aid = re.sub(r"^arxiv:", "", str(arxiv_id).strip(), flags=re.IGNORECASE)
        pdf_link = f"https://arxiv.org/pdf/{clean_aid}.pdf"

    if not pdf_link:
        return False, "no PDF link on arXiv"
    if not pdf_link.endswith(".pdf"):
        pdf_link += ".pdf"

    ok, reason = download_pdf(pdf_link, dest_path)
    return ok, None if ok else f"arXiv PDF failed: {reason}"


def fetch_papers(source_file: str | None = None):
    references = _load_references()
    if source_file:
        scoped = [r for r in references if r.get("source_file") == source_file]
        if scoped:
            references = scoped
    os.makedirs(PULLED_PDFS_DIR, exist_ok=True)

    downloaded = _load_state(DOWNLOADED_JSON_PATH, {})
    failed = _load_state(FAILED_DOWNLOADS_PATH, {})
    if isinstance(failed, list):  # legacy shape
        failed = {}

    # One entry per distinct source, even when several papers cite it.
    # Also record cross-paper (source_file, xml_id) links on already-downloaded entries
    # so auditing a new paper immediately resolves references fetched by an earlier run.
    unique = {}
    cited_links_dirty = False
    for ref in references:
        raw = ref.get("raw_reference") or ""
        if raw and len(raw) > MAX_CITATION_LEN:
            continue
        key = source_key(ref)
        if not key:
            continue
        unique.setdefault(key, ref)
        existing = downloaded.get(key)
        src_file = ref.get("source_file")
        xid = ref.get("xml_id")
        if isinstance(existing, dict) and src_file and xid:
            if existing.get("cited_by") != src_file or existing.get("xml_id") != xid:
                cref_key = f"{src_file}:{xid}"
                cited_refs = existing.get("cited_refs")
                if not isinstance(cited_refs, list):
                    cited_refs = []
                if cref_key not in cited_refs:
                    cited_refs.append(cref_key)
                    existing["cited_refs"] = cited_refs
                    cited_links_dirty = True

    logger.info("%d references collapse to %d distinct sources.", len(references), len(unique))

    retryable = {
        k for k, record in failed.items()
        if _is_retryable((record or {}).get("reason") if isinstance(record, dict) else None)
    }
    upgradable_abstracts = {
        k for k, record in downloaded.items()
        if isinstance(record, dict)
        and record.get("evidence_tier") == "abstract_only"
        and not record.get("path")
        and not record.get("oa_exhausted")
    }
    remaining = [
        (k, r) for k, r in unique.items()
        if (k not in downloaded or k in upgradable_abstracts)
        and (k not in failed or k in retryable)
    ]
    if len(remaining) < len(unique):
        logger.info("Skipping %d already processed.", len(unique) - len(remaining))
    retrying = len(retryable & {k for k, _ in remaining})
    if retrying:
        logger.info("Retrying %d that previously failed for a transient reason.", retrying)
    if cited_links_dirty and not remaining:
        _checkpoint(downloaded, failed)

    paper_index = PaperIndex.load(PAPER_INDEX_PATH)
    standalone = not pipeline_status.get_status().get("active")
    with pipeline_status.track_stage(
        "fetch",
        "Fetching referenced papers",
        current_step=4,
        total_steps=5,
        item_total=len(remaining),
        detail=f"0/{len(remaining)} papers checked ({len(downloaded)} downloaded, {len(failed)} unavailable)",
        mark_idle_on_exit=standalone,
        last_summary=f"+{len(downloaded)} papers fetched",
    ):
        for i, (key, ref) in enumerate(remaining, start=1):
            if pipeline_status.is_cancel_requested():
                logger.info("Fetch cancelled by user via UI at paper %d/%d", i, len(remaining))
                pipeline_status.add_event(f"⏹️ Fetch stopped by user ({len(downloaded)} downloaded)")
                break

            label = (ref.get("title") or ref.get("raw_reference") or key)[:70]
            logger.info("[%d/%d] %s", i, len(remaining), label)
            pipeline_status.update_progress(
                item_current=i,
                item_total=len(remaining),
                current_item_name=label,
                detail=f"Checking [{i}/{len(remaining)}]: {label[:50]} ({len(downloaded)} downloaded)",
            )

            dest = os.path.join(PULLED_PDFS_DIR, filename_for(key))
            record_path = dest
            reasons = []
            saved = False
            saved_fulltext = False
            doi = how = None
            provider = None
            arxiv_id = (
                ref.get("arxiv_id")
                or extract_arxiv_from_text(ref.get("doi"))
                or extract_arxiv_from_text(ref.get("raw_reference"))
            )

            try:
                doi, how = resolve_doi(ref)
                if not arxiv_id and doi:
                    arxiv_id = extract_arxiv_from_text(doi)

                corpus_hit = paper_index.find_by_ids(doi=doi, arxiv_id=arxiv_id) if (doi or arxiv_id) else None
                if not corpus_hit and ref.get("title") and ref.get("authors"):
                    corpus_hit = paper_index.find_by_title_author(ref.get("title"), ref.get("authors") or ())
                if corpus_hit and os.path.isfile(corpus_hit.path):
                    saved = saved_fulltext = True
                    provider = "corpus"
                    record_path = corpus_hit.path
                    logger.info("    already in corpus -> %s", corpus_hit.path)
                elif is_saved_pdf(dest) and not (
                    isinstance(downloaded.get(key), dict)
                    and downloaded[key].get("evidence_tier") == "abstract_only"
                    and not downloaded[key].get("path")
                ):
                    saved = saved_fulltext = True
                    provider = "local_file"
                    logger.info("    already on disk -> %s", os.path.basename(dest))
                elif doi:
                    saved, reason = try_unpaywall(doi, dest)
                    if saved:
                        saved_fulltext = True
                        provider = "unpaywall"
                    else:
                        reasons.append(reason)
                        time.sleep(UNPAYWALL_SLEEP)
                else:
                    reasons.append("no DOI resolved")

                if not saved and (doi or ref.get("pmid")):
                    saved, reason = try_europepmc(doi, ref, dest)
                    if saved:
                        saved_fulltext = True
                        provider = "europepmc"
                    else:
                        reasons.append(reason)

                if not saved:
                    ref_for_arxiv = dict(ref, arxiv_id=arxiv_id) if arxiv_id and not ref.get("arxiv_id") else ref
                    saved, reason = try_arxiv(ref_for_arxiv, dest)
                    if saved:
                        saved_fulltext = True
                        provider = "arxiv"
                    else:
                        reasons.append(reason)
                    time.sleep(ARXIV_RATE_LIMIT)

            except Exception as exc:
                logger.exception("Unhandled error on %s", key)
                reasons.append(str(exc))

            biblio = {}
            if not saved:
                try:
                    biblio = fetch_bibliometrics_and_abstract(doi or ref.get("title") or key)
                    if isinstance(biblio, dict):
                        bib_doi = normalise_doi(biblio.get("doi"))
                        bib_arxiv = biblio.get("arxiv_id")

                        # Check if newly discovered DOI/arXiv matches an existing local corpus PDF
                        if (bib_doi and bib_doi != doi) or (bib_arxiv and bib_arxiv != arxiv_id):
                            c_hit = paper_index.find_by_ids(doi=bib_doi or doi, arxiv_id=bib_arxiv or arxiv_id)
                            if c_hit and os.path.isfile(c_hit.path):
                                saved = saved_fulltext = True
                                provider = "corpus"
                                record_path = c_hit.path
                                doi = doi or bib_doi
                                arxiv_id = arxiv_id or bib_arxiv
                                logger.info("    matched corpus via OpenAlex/S2 IDs -> %s", c_hit.path)

                        oa_urls = list(biblio.get("oa_pdf_urls") or [])
                        if biblio.get("oa_pdf_url") and biblio["oa_pdf_url"] not in oa_urls:
                            oa_urls.insert(0, biblio["oa_pdf_url"])

                        if not saved and oa_urls:
                            oa_reasons = []
                            for oa_url in oa_urls:
                                logger.info("    attempting direct OA PDF from bibliometrics: %s", oa_url[:60])
                                ok, reason = download_pdf(oa_url, dest)
                                if ok:
                                    prev_refs = (downloaded.get(key) or {}).get("cited_refs") if isinstance(downloaded.get(key), dict) else None
                                    rec_dict = DownloadedPaper(
                                        key=key,
                                        path=dest,
                                        evidence_tier="full_text",
                                        abstract=(biblio.get("abstract") or "").strip(),
                                        is_retracted=bool(biblio.get("is_retracted", False)),
                                        citation_count=biblio.get("citation_count") if biblio.get("citation_count") is not None else 0,
                                        provider="oa_pdf_url",
                                        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        title=ref.get("title") or biblio.get("title") or "",
                                        venue=(ref.get("venue") or ref.get("container") or ref.get("journal") or biblio.get("venue") or "").strip(),
                                        raw_reference=ref.get("raw_reference") or "",
                                        doi=doi or bib_doi or biblio.get("doi"),
                                        arxiv_id=arxiv_id or bib_arxiv,
                                        doi_source=how or ("openalex" if bib_doi else None),
                                        authoritative=is_authoritative(key),
                                        cited_by=ref.get("source_file"),
                                        xml_id=ref.get("xml_id"),
                                    ).to_dict()
                                    if prev_refs:
                                        rec_dict["cited_refs"] = prev_refs
                                    downloaded[key] = rec_dict
                                    saved = saved_fulltext = True
                                    failed.pop(key, None)
                                    logger.info("    saved full text via OA URL -> %s", label[:50])
                                    pipeline_status.add_event(f"⬇️ Fetched OA PDF: {label[:50]}")
                                    break
                                oa_reasons.append(reason)
                            if not saved and oa_reasons:
                                reasons.append(f"oa_pdf_url failed: {oa_reasons[0]}")

                        # If OpenAlex/S2 uncovered a DOI or PMID we didn't have earlier, try Unpaywall/EuropePMC
                        if not saved and bib_doi and not doi:
                            doi = bib_doi
                            how = how or "openalex"
                            ok_u, r_u = try_unpaywall(doi, dest)
                            if ok_u:
                                saved = saved_fulltext = True
                                provider = "unpaywall"
                            else:
                                ok_e, r_e = try_europepmc(doi, ref, dest)
                                if ok_e:
                                    saved = saved_fulltext = True
                                    provider = "europepmc"

                        if not saved:
                            abstract_cand = (biblio.get("abstract") or "").strip()
                            if len(abstract_cand) >= 50:
                                prev_refs = (downloaded.get(key) or {}).get("cited_refs") if isinstance(downloaded.get(key), dict) else None
                                rec_dict = DownloadedPaper(
                                    key=key,
                                    path=None,
                                    evidence_tier="abstract_only",
                                    abstract=abstract_cand,
                                    is_retracted=bool(biblio.get("is_retracted", False)),
                                    citation_count=biblio.get("citation_count") if biblio.get("citation_count") is not None else 0,
                                    provider="openalex" if (doi or biblio.get("doi")) else "abstract_only",
                                    fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    title=ref.get("title") or biblio.get("title") or "",
                                    venue=(ref.get("venue") or ref.get("container") or ref.get("journal") or biblio.get("venue") or "").strip(),
                                    raw_reference=ref.get("raw_reference") or "",
                                    doi=doi or bib_doi or biblio.get("doi"),
                                    arxiv_id=arxiv_id or bib_arxiv,
                                    doi_source=how,
                                    authoritative=is_authoritative(key),
                                    cited_by=ref.get("source_file"),
                                    xml_id=ref.get("xml_id"),
                                ).to_dict()
                                if not any(_is_retryable(r) for r in reasons):
                                    rec_dict["oa_exhausted"] = True
                                if prev_refs:
                                    rec_dict["cited_refs"] = prev_refs
                                downloaded[key] = rec_dict
                                saved = True
                                failed.pop(key, None)
                                logger.info("    saved abstract-only -> %s", label[:50])
                                pipeline_status.add_event(f"📄 Abstract-only: {label[:50]}")
                except Exception as exc:
                    logger.debug("Abstract fallback check failed for %s: %s", key, exc)

            if saved:
                if saved_fulltext and (key not in downloaded or downloaded[key].get("evidence_tier") != "full_text"):
                    prev_refs = (downloaded.get(key) or {}).get("cited_refs") if isinstance(downloaded.get(key), dict) else None
                    if not biblio:
                        try:
                            biblio = fetch_bibliometrics_and_abstract(doi or ref.get("title") or key)
                        except Exception:
                            biblio = {}
                    rec_dict = DownloadedPaper(
                        key=key,
                        path=record_path,
                        provider=provider,
                        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        title=ref.get("title") or (biblio.get("title") if isinstance(biblio, dict) else None),
                        venue=(
                            ref.get("venue")
                            or ref.get("container")
                            or ref.get("journal")
                            or (biblio.get("venue") if isinstance(biblio, dict) else "")
                            or ""
                        ).strip(),
                        raw_reference=ref.get("raw_reference"),
                        doi=doi or (normalise_doi(biblio.get("doi")) if isinstance(biblio, dict) else None),
                        arxiv_id=arxiv_id or (biblio.get("arxiv_id") if isinstance(biblio, dict) else None),
                        doi_source=how,
                        authoritative=is_authoritative(key),
                        cited_by=ref.get("source_file"),
                        xml_id=ref.get("xml_id"),
                        abstract=ref.get("abstract") or (biblio.get("abstract") if isinstance(biblio, dict) else "") or "",
                        citation_count=ref.get("citation_count") if ref.get("citation_count") is not None else ((biblio.get("citation_count") if isinstance(biblio, dict) and biblio.get("citation_count") is not None else 0)),
                        is_retracted=bool(ref.get("is_retracted", False) or (biblio.get("is_retracted", False) if isinstance(biblio, dict) else False)),
                        evidence_tier="full_text",
                    ).to_dict()
                    if prev_refs:
                        rec_dict["cited_refs"] = prev_refs
                    downloaded[key] = rec_dict
                    failed.pop(key, None)   # a retry that worked is no longer a failure
                    logger.info("    saved -> %s", os.path.basename(record_path))
                    pipeline_status.add_event(f"✅ Downloaded ({provider or 'oa'}): {label[:50]}")
                pipeline_status.update_progress(
                    item_current=i,
                    item_total=len(remaining),
                    current_item_name=label,
                    detail=f"Downloaded {len(downloaded)} / Unavailable {len(failed)}",
                )
            elif key in upgradable_abstracts and key in downloaded:
                if not any(_is_retryable(r) for r in reasons):
                    downloaded[key]["oa_exhausted"] = True
                pipeline_status.update_progress(
                    item_current=i,
                    item_total=len(remaining),
                    current_item_name=label,
                    detail=f"Downloaded {len(downloaded)} / Unavailable {len(failed)}",
                )
            else:
                reason_str = " | ".join(r for r in reasons if r) or "fetch failed"
                record = {
                    "key": key,
                    "doi": doi,
                    "doi_source": how,
                    "authoritative": is_authoritative(key),
                    "title": ref.get("title"),
                    "container": ref.get("container") or ref.get("venue") or ref.get("journal"),
                    "cited_by": ref.get("source_file"),
                    "xml_id": ref.get("xml_id"),
                    "raw_reference": ref.get("raw_reference"),
                    "reason": reason_str,
                }
                failed[key] = record
                logger.info("    unavailable: %s", record["reason"])
                reason_short = reason_str[:35]
                pipeline_status.add_event(f"⚠️ {label[:45]} ({reason_short})")
                pipeline_status.update_progress(
                    item_current=i,
                    item_total=len(remaining),
                    current_item_name=label,
                    detail=f"Downloaded {len(downloaded)} / Unavailable {len(failed)}",
                )

            if i % 5 == 0 or i == len(remaining):
                _checkpoint(downloaded, failed)

        _checkpoint(downloaded, failed)

        try:
            run_sweep()
        except Exception as exc:
            logger.warning("End-of-fetch dedupe sweep failed: %s", exc)

        if pipeline_status.is_cancel_requested():
            logger.info("Fetch aborted early due to user stop request.")
            return

        manual = [r for r in failed.values() if r.get("doi")]
        pipeline_status.add_event(
            f"🌐 Fetch complete: {len(downloaded)} downloaded, {len(failed)} unavailable"
        )
        pipeline_status.update_progress(
            item_current=len(remaining),
            item_total=len(remaining),
            detail=f"Fetch complete: {len(downloaded)} downloaded, {len(failed)} unavailable",
        )
        logger.info(
            "Done. %d downloaded, %d unavailable (%d have a DOI and can be fetched by hand).",
            len(downloaded),
            len(failed),
            len(manual),
        )



def backfill_failed_downloads(
    failed_path: str = FAILED_DOWNLOADS_PATH,
    downloaded_path: str = DOWNLOADED_JSON_PATH,
    output_dir: str = PULLED_PDFS_DIR,
) -> dict[str, int]:
    """Unblock and backfill failed downloads in failed_downloads.json.
    Attempts to resolve via direct oa_pdf_url or fall back to abstract_only.
    """
    failed = _load_state(failed_path, {})
    downloaded = _load_state(downloaded_path, {})
    if not isinstance(failed, dict) or not failed:
        return {"attempted": 0, "resolved_oa": 0, "resolved_abstract": 0, "remaining_failed": 0}

    from research_assistant.shared.academic_apis import fetch_bibliometrics_and_abstract

    os.makedirs(output_dir, exist_ok=True)
    resolved_oa = 0
    resolved_abstract = 0
    to_check = [
        (k, rec) for k, rec in list(failed.items())
        if isinstance(rec, dict) and k not in downloaded and (rec.get("doi") or rec.get("title"))
    ]

    for k, rec in to_check:
        doi = rec.get("doi")
        title = rec.get("title")
        raw = rec.get("raw_reference") or ""
        key = rec.get("key") or k
        dest_path = os.path.join(output_dir, f"{key.replace(':', '_')}.pdf")

        try:
            biblio = fetch_bibliometrics_and_abstract(doi or title or key)
            oa_urls = list(biblio.get("oa_pdf_urls") or [])
            if biblio.get("oa_pdf_url") and biblio["oa_pdf_url"] not in oa_urls:
                oa_urls.insert(0, biblio["oa_pdf_url"])
            saved = False
            for oa_url in oa_urls:
                ok, _ = download_pdf(oa_url, dest_path)
                if ok:
                    downloaded[key] = DownloadedPaper(
                        key=key,
                        path=dest_path,
                        evidence_tier="full_text",
                        abstract=(biblio.get("abstract") or "").strip(),
                        is_retracted=bool(biblio.get("is_retracted", False)),
                        citation_count=biblio.get("citation_count") or 0,
                        provider="oa_pdf_url",
                        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        title=title or biblio.get("title") or "",
                        venue=(rec.get("venue") or rec.get("container") or biblio.get("venue") or "").strip(),
                        raw_reference=raw,
                        doi=doi or biblio.get("doi"),
                        arxiv_id=rec.get("arxiv_id") or biblio.get("arxiv_id"),
                        doi_source="oa_backfill",
                        authoritative=is_authoritative(key),
                        cited_by=rec.get("cited_by"),
                        xml_id=rec.get("xml_id"),
                    ).to_dict()
                    failed.pop(k, None)
                    resolved_oa += 1
                    saved = True
                    break

            if not saved:
                abstract = (biblio.get("abstract") or "").strip()
                if len(abstract) >= 50:
                    downloaded[key] = DownloadedPaper(
                        key=key,
                        path=None,
                        evidence_tier="abstract_only",
                        abstract=abstract,
                        is_retracted=bool(biblio.get("is_retracted", False)),
                        citation_count=biblio.get("citation_count") or 0,
                        provider="openalex" if biblio.get("doi") else "abstract_only",
                        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        title=title or biblio.get("title") or "",
                        venue=(rec.get("venue") or rec.get("container") or biblio.get("venue") or "").strip(),
                        raw_reference=raw,
                        doi=doi or biblio.get("doi"),
                        arxiv_id=rec.get("arxiv_id"),
                        doi_source="abstract_backfill",
                        authoritative=is_authoritative(key),
                        cited_by=rec.get("cited_by"),
                        xml_id=rec.get("xml_id"),
                    ).to_dict()
                    failed.pop(k, None)
                    resolved_abstract += 1
        except Exception as exc:
            logger.debug("Failed backfill for %s: %s", k, exc)

    _checkpoint(downloaded, failed)
    return {
        "attempted": len(to_check),
        "resolved_oa": resolved_oa,
        "resolved_abstract": resolved_abstract,
        "remaining_failed": len(failed),
    }


fetch_missing_citations = fetch_papers


if __name__ == "__main__":
    pipeline_status.clear_cancel_request()
    fetch_papers()
