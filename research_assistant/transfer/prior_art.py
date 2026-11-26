"""Prior-art checking across OpenAlex and arXiv with cutoff years."""

from __future__ import annotations

import re
import threading
import time
import unittest.mock
import xml.etree.ElementTree as ET
from typing import Callable

import numpy as np

from research_assistant import config
from research_assistant.config import (
    ARXIV_RATE_LIMIT,
    OPENALEX_MAILTO,
    OPENALEX_URL,
    TRANSFER_PRIOR_ART_DONE,
    TRANSFER_PRIOR_ART_PARTIAL,
    USER_AGENT,
)
from research_assistant.shared.log import get_logger
from research_assistant.transfer.models import PriorArt

logger = get_logger("transfer.prior_art")

_RATE_LOCKS: dict[str, threading.Lock] = {
    "openalex": threading.Lock(),
    "arxiv": threading.Lock(),
}
_LAST_CALL_TS: dict[str, float] = {}
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _throttle_provider(provider: str, min_interval: float, is_mocked: bool) -> None:
    if is_mocked or min_interval <= 0:
        return
    lock = _RATE_LOCKS.setdefault(provider, threading.Lock())
    with lock:
        now = time.monotonic()
        last = _LAST_CALL_TS.get(provider)
        if last is not None:
            wait = min_interval - (now - last)
            if wait > 0:
                time.sleep(wait)
        _LAST_CALL_TS[provider] = time.monotonic()


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct an abstract from OpenAlex's abstract_inverted_index into word order."""
    if not inverted_index:
        return ""
    words_by_pos: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                words_by_pos.append((pos, word))
    words_by_pos.sort(key=lambda x: x[0])
    return " ".join(word for _, word in words_by_pos)


def parse_openalex(payload: dict) -> list[dict]:
    """Parse OpenAlex works JSON response into standard hit dicts."""
    results: list[dict] = []
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        title = (item.get("display_name") or item.get("title") or "").strip()
        year = item.get("publication_year")
        if year is None:
            pub_date = str(item.get("publication_date") or "")
            if len(pub_date) >= 4 and pub_date[:4].isdigit():
                year = int(pub_date[:4])
        url = item.get("doi") or item.get("id") or ""
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
        results.append({
            "title": title,
            "year": year,
            "url": url,
            "abstract": abstract,
        })
    return results


def parse_arxiv(xml_text: str) -> list[dict]:
    """Parse arXiv Atom XML response into standard hit dicts."""
    results: list[dict] = []
    if not xml_text or not xml_text.strip():
        return results
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if not entries:
        entries = root.findall("entry")

    for entry in entries:
        title_el = entry.find("atom:title", ns)
        if title_el is None:
            title_el = entry.find("title")
        title = " ".join((title_el.text or "").split()) if title_el is not None else ""

        summary_el = entry.find("atom:summary", ns)
        if summary_el is None:
            summary_el = entry.find("summary")
        abstract = " ".join((summary_el.text or "").split()) if summary_el is not None else ""

        pub_el = entry.find("atom:published", ns)
        if pub_el is None:
            pub_el = entry.find("published")
        pub_text = (pub_el.text or "").strip() if pub_el is not None else ""
        year = int(pub_text[:4]) if len(pub_text) >= 4 and pub_text[:4].isdigit() else None

        id_el = entry.find("atom:id", ns)
        if id_el is None:
            id_el = entry.find("id")
        url = (id_el.text or "").strip() if id_el is not None else ""

        results.append({
            "title": title,
            "year": year,
            "url": url,
            "abstract": abstract,
        })
    return results


def _request_with_retry(url: str, params: dict):
    import requests

    is_mocked = isinstance(requests.get, unittest.mock.Base)
    provider = "arxiv" if "arxiv.org" in url else ("openalex" if "openalex.org" in url else "other")
    interval = float(ARXIV_RATE_LIMIT) if provider == "arxiv" else (0.12 if provider == "openalex" else 0.0)

    last_exc: Exception | None = None
    for attempt in range(2):
        _throttle_provider(provider, interval, is_mocked=is_mocked)
        try:
            resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 0 and not is_mocked:
                time.sleep(1.5)
                continue
            raise

        status_code = getattr(resp, "status_code", None)
        if isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES and attempt == 0:
            if not is_mocked:
                retry_after = 1.5
                hdrs = getattr(resp, "headers", None)
                if isinstance(hdrs, dict) and hdrs.get("Retry-After"):
                    try:
                        retry_after = min(10.0, max(0.5, float(hdrs["Retry-After"])))
                    except (ValueError, TypeError):
                        pass
                time.sleep(retry_after)
            continue

        resp.raise_for_status()
        return resp

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Request to {url} failed after retries")


def default_get_json(url: str, params: dict) -> dict:
    return _request_with_retry(url, params).json()


def default_get_text(url: str, params: dict) -> str:
    return _request_with_retry(url, params).text


_QUERY_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "into", "when",
    "where", "which", "over", "under", "between", "through", "using", "used",
    "via", "such", "than", "each", "both", "all", "any", "are", "was", "were",
    "in", "on", "of", "to", "by", "as", "at", "an", "or",
})


def _build_prior_art_query(technique_name: str, quantities: list[str]) -> str:
    """Build a concise search query for OpenAlex and arXiv prior-art lookup."""
    raw = f"{technique_name} {' '.join(quantities[:2])}".strip()
    words = raw.split()
    if len(words) <= 8:
        return raw
    tech_tokens = [
        w for w in re.findall(r"[a-zA-Z0-9\-]+", technique_name)
        if w.lower() not in _QUERY_STOPWORDS and len(w) >= 3
    ][:5]
    quant_tokens = [
        w for w in re.findall(r"[a-zA-Z0-9\-]+", " ".join(quantities[:2]))
        if w.lower() not in _QUERY_STOPWORDS and len(w) >= 3
    ][:3]
    combined = tech_tokens + [w for w in quant_tokens if w.lower() not in {t.lower() for t in tech_tokens}]
    return " ".join(combined) if combined else raw


class PriorArtClient:
    """Client for checking prior art across OpenAlex and arXiv."""

    def __init__(
        self,
        get_json: Callable[[str, dict], dict] = default_get_json,
        get_text: Callable[[str, dict], str] = default_get_text,
        embed_documents: Callable[[list[str]], list[list[float]]] | None = None,
        limit: int = 10,
    ):
        self.get_json = get_json
        self.get_text = get_text
        self.embed_documents = embed_documents
        self.limit = limit
        self._cache_lock = threading.Lock()
        self._hit_cache: dict[tuple[str, int | None], list[dict]] = {}

    def check(
        self,
        technique_name: str,
        quantities: list[str],
        summary: str,
        before_year: int | None = None,
    ) -> PriorArt:
        """
        Check literature for prior art combining the technique and quantities.

        1. Query is built via _build_prior_art_query(technique_name, quantities[:2]).
        2. Searches OpenAlex and arXiv independently (surviving single-provider outages).
        3. Drops hits with year >= before_year.
        4. Scores hits by cosine similarity between summary and title+abstract.
        5. Assigns status: done_before, partial, not_found, or unchecked.
        """
        query = _build_prior_art_query(technique_name, quantities)
        cache_key = (query, before_year)

        with self._cache_lock:
            cached_hits = self._hit_cache.get(cache_key)

        partial_error: str | None = None
        if cached_hits is not None:
            all_hits = list(cached_hits)
        else:
            oa_params: dict[str, str | int] = {
                "search": query,
                "per_page": self.limit,
            }
            mailto = getattr(config, "OPENALEX_MAILTO", None) or OPENALEX_MAILTO
            if mailto:
                oa_params["mailto"] = mailto
            api_key = getattr(config, "OPENALEX_API_KEY", None)
            if api_key:
                oa_params["api_key"] = api_key
            if before_year is not None:
                oa_params["filter"] = f"to_publication_date:{before_year - 1}-12-31"

            oa_hits: list[dict] = []
            arxiv_hits: list[dict] = []
            provider_errors: list[str] = []

            try:
                oa_payload = self.get_json(OPENALEX_URL, oa_params)
                oa_hits = parse_openalex(oa_payload)
            except Exception as exc:
                logger.warning("OpenAlex prior-art lookup failed for query %r: %s", query, exc)
                provider_errors.append(f"openalex: {exc}")

            arxiv_params: dict[str, str | int] = {
                "search_query": f"all:{query}",
                "max_results": self.limit,
            }
            try:
                arxiv_text = self.get_text("https://export.arxiv.org/api/query", arxiv_params)
                arxiv_hits = parse_arxiv(arxiv_text)
            except Exception as exc:
                logger.warning("arXiv prior-art lookup failed for query %r: %s", query, exc)
                provider_errors.append(f"arxiv: {exc}")

            partial_error = "; ".join(provider_errors) if provider_errors else None
            if len(provider_errors) >= 2:
                logger.warning(
                    "Prior-art check failed across all providers for %r (%s) — marking unchecked",
                    technique_name,
                    partial_error,
                )
                return PriorArt(status="unchecked", hits=[], error=partial_error)

            all_hits = oa_hits + arxiv_hits
            if not provider_errors:
                with self._cache_lock:
                    self._hit_cache[cache_key] = list(all_hits)

        try:
            # Filter by cutoff year
            if before_year is not None:
                all_hits = [
                    h for h in all_hits
                    if h.get("year") is not None and h["year"] < before_year
                ]

            if not all_hits or self.embed_documents is None:
                return PriorArt(status="not_found", hits=[], error=partial_error)

            # Deduplicate by title
            seen_titles: set[str] = set()
            unique_hits: list[dict] = []
            for h in all_hits:
                norm_title = h["title"].lower().strip()
                if norm_title and norm_title not in seen_titles:
                    seen_titles.add(norm_title)
                    unique_hits.append(h)
                elif not norm_title:
                    unique_hits.append(h)

            emb_summary_list = self.embed_documents([summary])
            emb_summary = np.asarray(emb_summary_list[0], dtype=float)
            s_norm = np.linalg.norm(emb_summary)
            if s_norm == 0:
                s_norm = 1.0

            hit_texts = [f"{h['title']} {h['abstract']}".strip() for h in unique_hits]
            emb_hits = np.asarray(self.embed_documents(hit_texts), dtype=float)

            scored_hits: list[dict] = []
            for hit, h_vec in zip(unique_hits, emb_hits):
                h_norm = np.linalg.norm(h_vec)
                if h_norm == 0:
                    h_norm = 1.0
                sim = float(np.dot(emb_summary, h_vec) / (s_norm * h_norm))
                hit_copy = dict(hit)
                hit_copy["similarity"] = sim
                scored_hits.append(hit_copy)

            scored_hits.sort(key=lambda x: x["similarity"], reverse=True)
            top_hits = scored_hits[:3]

            # Determine status
            content_words = [
                w.lower()
                for w in re.findall(r"[a-zA-Z]+", technique_name)
                if len(w) >= 4
            ]

            status = "not_found"
            for hit in top_hits:
                hit_text = f"{hit['title']} {hit['abstract']}".lower()
                has_all_words = bool(content_words) and all(w in hit_text for w in content_words)
                if hit["similarity"] >= TRANSFER_PRIOR_ART_DONE and has_all_words:
                    status = "done_before"
                    break

            if status != "done_before":
                for hit in top_hits:
                    if hit["similarity"] >= TRANSFER_PRIOR_ART_PARTIAL:
                        status = "partial"
                        break

            return PriorArt(status=status, hits=top_hits, error=partial_error)

        except Exception as exc:
            logger.warning("Prior-art scoring failed for %r: %s", technique_name, exc)
            err_msg = f"{partial_error}; scoring: {exc}" if partial_error else f"scoring: {exc}"
            return PriorArt(status="unchecked", hits=[], error=err_msg)
