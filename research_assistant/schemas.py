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

# The one union shape this module checks. Finding I3: agent2_fetcher.py wrote
# a tuple into `doi_source` (str | None) because of an operator-precedence
# bug, and nothing here caught it before it reached disk. This is
# deliberately narrow -- str | None is the shape every optional scalar field
# below actually uses -- not a general type system.
_STR_OR_NONE = str | None


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
        # An explicit `null` is dropped rather than passed through: required
        # fields already raised above if null, so this only ever affects
        # optional fields, where dropping the key lets the dataclass default
        # apply instead of storing a bare None where e.g. a tuple is expected.
        kwargs = {k: v for k, v in data.items() if k in known and v is not None}

        # `authors` becomes tuple(value) below. tuple() does not reject a
        # bare string -- it happily explodes "Smith, J." into a 9-tuple of
        # characters, one per letter, which then produces a different
        # source_key than the list-of-names it was supposed to be. Catch
        # that shape specifically, before it reaches tuple().
        if isinstance(kwargs.get("authors"), str):
            raise SchemaError(
                f"{cls.__name__}.authors: expected a list of names, got a bare "
                f"string ({kwargs['authors']!r})"
            )

        # Lightweight scalar check: a field declared `str | None` must be a
        # str whenever it is present (None was already filtered out above).
        # Deliberately narrow -- see _STR_OR_NONE's comment -- this is not a
        # general type system, just the one shape that has actually broken.
        field_types = {f.name: f.type for f in fields(cls)}
        for name, value in kwargs.items():
            if field_types.get(name) == _STR_OR_NONE and not isinstance(value, str):
                raise SchemaError(
                    f"{cls.__name__}.{name}: expected str, got {type(value).__name__} "
                    f"({value!r})"
                )

        try:
            if "authors" in kwargs:
                kwargs["authors"] = tuple(kwargs["authors"])
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{cls.__name__}: {exc}") from exc


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
    provider: str                         # unpaywall | europepmc | arxiv
    fetched_at: str
    title: str | None = None
    raw_reference: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    doi_source: str | None = None         # how the DOI was resolved: grobid | crossref
    authoritative: bool | None = None     # whether the source key came from a real identifier
    cited_by: str | None = None           # source_file of the paper that cited this one
    xml_id: str | None = None             # GROBID biblStruct id of the originating reference


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
