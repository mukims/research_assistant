"""
Contracts between pipeline stages.

The agents hand work to each other through JSON manifests. Those were untyped,
and it cost: agent 3 carried a compatibility shim for two incompatible
downloaded.json shapes because one of them "made every entry look like a
missing file and ingested nothing at all".

Each record validates at the boundary. Unknown keys are ignored so a manifest
written by an older run still loads; a missing *required* field raises, because
a half-built record is exactly the failure being designed out.
"""

from dataclasses import MISSING, asdict, dataclass, fields


class SchemaError(ValueError):
    """A record could not be built from the given mapping."""


def _required_names(cls) -> tuple[str, ...]:
    return tuple(
        f.name
        for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    )


class _Record:
    """to_dict / from_dict shared by every schema below."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise SchemaError(
                f"{cls.__name__}: expected a mapping, got {type(data).__name__}"
            )
        missing = [n for n in _required_names(cls) if data.get(n) is None]
        if missing:
            raise SchemaError(
                f"{cls.__name__}: missing required field(s): {', '.join(missing)}"
            )
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if kwargs.get("authors") is not None:
            kwargs["authors"] = tuple(kwargs["authors"])
        return cls(**kwargs)


@dataclass(frozen=True)
class Reference(_Record):
    """One entry in a paper's reference list. Agent 1 → Agent 2."""

    raw_reference: str
    source_file: str                      # the PDF that cited it
    title: str | None = None
    container: str | None = None          # journal / proceedings
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    doi_confidence: str | None = None     # high | medium | low | unknown
    arxiv_id: str | None = None
    pmid: str | None = None
    xml_id: str | None = None             # GROBID biblStruct id; None for regex
    extraction_method: str = "grobid"     # grobid | regex


@dataclass(frozen=True)
class DownloadedPaper(_Record):
    """A reference whose full text was retrieved. Agent 2 → Agent 3."""

    key: str                              # source_key — the primary identity
    path: str
    provider: str                         # unpaywall | europepmc | arxiv | crossref
    fetched_at: str
    title: str | None = None
    raw_reference: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None


@dataclass(frozen=True)
class SeedPaper(_Record):
    """The paper a research query was seeded from. Agent 0 → orchestrator."""

    key: str
    path: str
    fetched_at: str
    source: str                           # "search" | "manual-url"
    title: str | None = None
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
