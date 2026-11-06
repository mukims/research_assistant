"""Corpus paper index model, lookups, aliases, duplicate groups, and manifest repointing."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import os
import re
import threading
import unicodedata

from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.source_key import fold_name, fold_title, normalise_doi

logger = logging.getLogger(__name__)

_EDIT_LOCK = threading.Lock()


@dataclass(frozen=True)
class PaperEntry:
    document: str                     # ingest-manifest key: pdf_key(path), e.g. "doi_10.1038_nature05180.pdf"
    path: str
    title: str | None = None          # as read; normalised only for comparison
    first_authors: tuple[str, ...] = ()   # first-author surnames as read by each extractor, GROBID's first
    dois: tuple[str, ...] = ()
    arxiv_ids: tuple[str, ...] = ()


def normalise_arxiv(arxiv_id: str | None) -> str | None:
    """Lowercase, strip 'arxiv:' prefix and trailing r'v\\d+'."""
    if not arxiv_id:
        return None
    s = arxiv_id.strip().lower()
    if s.startswith("arxiv:"):
        s = s[len("arxiv:"):].strip()
    s = re.sub(r"v\d+$", "", s)
    return s if s else None


def surname_key(name: str | None) -> str | None:
    """NFKD; drop every char whose unicodedata.category is Mn, Sk or Lm (so 'Kulyt˙e' and 'Kulytė' agree);
    then fold_name(); return the LAST whitespace token ('Essam Ghareeb' -> 'ghareeb'); empty -> None.
    """
    if not name:
        return None
    # Drop modifier symbols (Sk, Lm) before NFKD so compatibility decompositions
    # (e.g. U+02D9 dot above -> space + combining dot) do not insert whitespace
    # into a name like "Kulyt˙e".
    cleaned = "".join(c for c in name if unicodedata.category(c) not in ("Sk", "Lm"))
    norm = unicodedata.normalize("NFKD", cleaned)
    norm_cleaned = "".join(c for c in norm if unicodedata.category(c) not in ("Mn", "Sk", "Lm"))
    folded = fold_name(norm_cleaned)
    tokens = folded.split()
    if not tokens:
        return None
    return tokens[-1]


class PaperIndex:
    def __init__(
        self,
        path: str,
        papers: dict[str, PaperEntry] | None = None,
        aliases: dict[str, str] | None = None,
    ):
        self.path = path
        self._papers: dict[str, PaperEntry] = dict(papers) if papers else {}
        self._aliases: dict[str, str] = dict(aliases) if aliases else {}

    @classmethod
    def load(cls, path: str) -> "PaperIndex":
        """Missing or unparseable file -> empty index, logs a warning."""
        if not os.path.exists(path):
            return cls(path=path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("Paper index at %s is not a dict; using empty index", path)
                return cls(path=path)
            papers_data = data.get("papers", {})
            papers: dict[str, PaperEntry] = {}
            if isinstance(papers_data, dict):
                for doc, entry_data in papers_data.items():
                    if isinstance(entry_data, dict):
                        papers[doc] = PaperEntry(
                            document=entry_data.get("document", doc),
                            path=entry_data.get("path", ""),
                            title=entry_data.get("title"),
                            first_authors=tuple(entry_data.get("first_authors") or ()),
                            dois=tuple(entry_data.get("dois") or ()),
                            arxiv_ids=tuple(entry_data.get("arxiv_ids") or ()),
                        )
            aliases = data.get("aliases", {})
            if not isinstance(aliases, dict):
                aliases = {}
            return cls(path=path, papers=papers, aliases=aliases)
        except Exception as e:
            logger.warning("Failed to load paper index from %s: %s; using empty index", path, e)
            return cls(path=path)

    def save(self) -> None:
        """Atomic write to self.path."""
        data = {
            "version": 1,
            "papers": {
                doc: {
                    "document": entry.document,
                    "path": entry.path,
                    "title": entry.title,
                    "first_authors": list(entry.first_authors),
                    "dois": list(entry.dois),
                    "arxiv_ids": list(entry.arxiv_ids),
                }
                for doc, entry in self._papers.items()
            },
            "aliases": dict(self._aliases),
        }
        atomic_write_json(self.path, data)

    def add(self, entry: PaperEntry) -> None:
        """Replaces an entry with the same document."""
        self._papers[entry.document] = entry

    def get(self, document: str) -> PaperEntry | None:
        return self._papers.get(document)

    def find_by_ids(self, doi: str | None = None, arxiv_id: str | None = None) -> PaperEntry | None:
        norm_doi = normalise_doi(doi) if doi else None
        norm_arxiv = normalise_arxiv(arxiv_id) if arxiv_id else None
        if not norm_doi and not norm_arxiv:
            return None
        for entry in self._papers.values():
            if norm_doi:
                for d in entry.dois:
                    if normalise_doi(d) == norm_doi:
                        return entry
            if norm_arxiv:
                for a in entry.arxiv_ids:
                    if normalise_arxiv(a) == norm_arxiv:
                        return entry
        return None

    def find_by_title_author(
        self,
        title: str | None,
        first_authors: Iterable[str],
        exclude: str | None = None,
    ) -> PaperEntry | None:
        folded_title = fold_title(title)
        if not folded_title:
            return None
        query_keys = {k for a in first_authors if (k := surname_key(a)) is not None}
        if not query_keys:
            return None

        for entry in self._papers.values():
            if exclude and entry.document == exclude:
                continue
            if fold_title(entry.title) != folded_title:
                continue
            entry_keys = {k for a in entry.first_authors if (k := surname_key(a)) is not None}
            if query_keys & entry_keys:
                return entry
        return None

    def add_alias(self, alias: str, document: str) -> None:
        self._aliases[alias] = document

    def resolve(self, document_or_alias: str) -> PaperEntry | None:
        curr = document_or_alias
        visited = set()
        while curr in self._aliases and curr not in visited:
            visited.add(curr)
            curr = self._aliases[curr]
        return self.get(curr)

    def duplicate_groups(self) -> list[dict]:
        groups: list[dict] = []

        # 1. DOI duplicates
        doi_map: dict[str, set[str]] = defaultdict(set)
        for entry in self._papers.values():
            for d in entry.dois:
                norm = normalise_doi(d)
                if norm:
                    doi_map[norm].add(entry.document)
        for key in sorted(doi_map):
            docs = doi_map[key]
            if len(docs) > 1:
                groups.append({"reason": "doi", "key": key, "documents": sorted(docs)})

        # 2. arXiv duplicates
        arxiv_map: dict[str, set[str]] = defaultdict(set)
        for entry in self._papers.values():
            for a in entry.arxiv_ids:
                norm = normalise_arxiv(a)
                if norm:
                    arxiv_map[norm].add(entry.document)
        for key in sorted(arxiv_map):
            docs = arxiv_map[key]
            if len(docs) > 1:
                groups.append({"reason": "arxiv", "key": key, "documents": sorted(docs)})

        # 3. Title + Author duplicates
        title_groups: dict[str, list[PaperEntry]] = defaultdict(list)
        for entry in self._papers.values():
            ft = fold_title(entry.title)
            if ft:
                entry_keys = {k for a in entry.first_authors if (k := surname_key(a)) is not None}
                if entry_keys:
                    title_groups[ft].append(entry)

        for ft in sorted(title_groups):
            entries = title_groups[ft]
            if len(entries) < 2:
                continue
            # Cluster entries sharing at least one surname key using connected components
            adj: dict[str, set[str]] = {e.document: set() for e in entries}
            entry_surnames = {
                e.document: {k for a in e.first_authors if (k := surname_key(a)) is not None}
                for e in entries
            }
            for i, e1 in enumerate(entries):
                for e2 in entries[i + 1:]:
                    if entry_surnames[e1.document] & entry_surnames[e2.document]:
                        adj[e1.document].add(e2.document)
                        adj[e2.document].add(e1.document)

            visited = set()
            for doc in sorted(adj):
                if doc not in visited:
                    component = []
                    q = [doc]
                    visited.add(doc)
                    while q:
                        curr = q.pop()
                        component.append(curr)
                        for neighbor in adj[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                    if len(component) > 1:
                        groups.append({
                            "reason": "title_author",
                            "key": ft,
                            "documents": sorted(component),
                        })

        return groups

    def __len__(self) -> int:
        return len(self._papers)


@contextmanager
def edit(path: str) -> Iterator[PaperIndex]:
    """Module-level threading.Lock held; load, yield, save on clean exit."""
    with _EDIT_LOCK:
        idx = PaperIndex.load(path)
        yield idx
        idx.save()


def repoint_manifest(manifest_path: str, old_path: str, new_path: str) -> int:
    """Rewrite downloaded.json records whose 'path' basename == basename(old_path) to new_path; returns count."""
    if not os.path.exists(manifest_path):
        return 0
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Could not read manifest at %s: %s", manifest_path, e)
        return 0

    if not isinstance(data, (dict, list)):
        return 0

    target_basename = os.path.basename(old_path)
    count = 0
    records = data.values() if isinstance(data, dict) else data
    for record in records:
        if isinstance(record, dict) and os.path.basename(record.get("path", "")) == target_basename:
            record["path"] = new_path
            count += 1

    if count > 0:
        atomic_write_json(manifest_path, data)

    return count
