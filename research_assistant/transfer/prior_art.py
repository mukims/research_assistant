"""Prior-art checking across OpenAlex and arXiv with cutoff years."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Callable

import numpy as np

from research_assistant.config import (
    OPENALEX_MAILTO,
    OPENALEX_URL,
    TRANSFER_PRIOR_ART_DONE,
    TRANSFER_PRIOR_ART_PARTIAL,
)
from research_assistant.transfer.models import PriorArt


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


def default_get_json(url: str, params: dict) -> dict:
    import requests
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def default_get_text(url: str, params: dict) -> str:
    import requests
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.text


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

    def check(
        self,
        technique_name: str,
        quantities: list[str],
        summary: str,
        before_year: int | None = None,
    ) -> PriorArt:
        """
        Check literature for prior art combining the technique and quantities.

        1. Query is '{technique_name} {quantities[:2]}'.
        2. Searches OpenAlex and arXiv.
        3. Drops hits with year >= before_year.
        4. Scores hits by cosine similarity between summary and title+abstract.
        5. Assigns status: done_before, partial, not_found, or unchecked.
        """
        try:
            query = f"{technique_name} {' '.join(quantities[:2])}".strip()

            oa_params: dict[str, str | int] = {
                "search": query,
                "per_page": self.limit,
            }
            if OPENALEX_MAILTO:
                oa_params["mailto"] = OPENALEX_MAILTO
            if before_year is not None:
                oa_params["filter"] = f"to_publication_date:{before_year - 1}-12-31"

            oa_payload = self.get_json(OPENALEX_URL, oa_params)
            oa_hits = parse_openalex(oa_payload)

            arxiv_params: dict[str, str | int] = {
                "search_query": f"all:{query}",
                "max_results": self.limit,
            }
            arxiv_text = self.get_text("https://export.arxiv.org/api/query", arxiv_params)
            arxiv_hits = parse_arxiv(arxiv_text)

            all_hits = oa_hits + arxiv_hits

            # Filter by cutoff year
            if before_year is not None:
                all_hits = [
                    h for h in all_hits
                    if h.get("year") is None or h["year"] < before_year
                ]

            if not all_hits or self.embed_documents is None:
                return PriorArt(status="not_found", hits=[])

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

            return PriorArt(status=status, hits=top_hits)

        except Exception:
            return PriorArt(status="unchecked", hits=[])
