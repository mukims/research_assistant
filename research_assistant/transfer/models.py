"""Data model and serialization for technique transfer."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Passage:
    key: str
    chunk_index: int
    document: str
    citation: str
    section: str
    text: str
    role: str  # "near", "source", or "card"


@dataclass
class Premise:
    claim: str
    key: str
    document: str = ""
    chunk_index: int | None = None
    judgement: str | None = None
    span_verified: bool | None = None
    supporting_span: str | None = None
    reason: str | None = None
    error: str | None = None

    @property
    def accepted(self) -> bool:
        return self.judgement == "Supports" and self.span_verified is True

    @property
    def partial(self) -> bool:
        return self.judgement == "Partially supports" and self.span_verified is True

    @property
    def credit(self) -> float:
        if self.accepted:
            return 1.0
        if self.partial:
            return 0.5
        return 0.0

    @property
    def contradicts(self) -> bool:
        return self.judgement == "Contradicts"


@dataclass
class Bottleneck:
    id: str
    statement: str
    quantities: list[str]
    current_method: str
    generic: dict[str, str]  # "specific", "mid", "broad"
    premises: list[Premise] = field(default_factory=list)


@dataclass
class Card:
    card_id: str
    document: str
    citation: str
    name: str
    problem_solved: str
    context: str
    inputs: str
    outputs: str
    assumptions: list[str]
    evidence: Premise
    mechanism: list[str] = field(default_factory=list)  # how it works, step by step

    @property
    def verified(self) -> bool:
        return self.evidence.accepted or self.evidence.partial


@dataclass
class MappingRow:
    source_role: str
    target_role: str
    assumption: str
    status: str  # "holds", "breaks", "enforce", "unknown"
    core: bool
    note: str = ""


@dataclass
class Identifiability:
    measurements: str = ""
    unknowns: str = ""
    symmetries: list[str] = field(default_factory=list)
    verdict: str = "unclear"  # "determined", "underdetermined", "unclear"
    argument: str = ""


@dataclass
class Step:
    action: str
    input: str
    output: str
    tool: str


@dataclass
class Critique:
    shape_match: str = "unknown"
    rows: list[dict] = field(default_factory=list)
    identifiability: str | None = None
    objections: list[dict] = field(default_factory=list)


@dataclass
class PriorArt:
    status: str = "unchecked"
    hits: list[dict] = field(default_factory=list)


@dataclass
class Scores:
    depth: float
    checks: dict[str, bool]
    reward: float
    set_aside: str | None = None


@dataclass
class Transfer:
    id: str
    bottleneck_id: str
    card_id: str
    card_document: str
    level: str
    retrieval_rank: int
    title: str
    summary: str
    mapping: list[MappingRow]
    identifiability: Identifiability
    steps: list[Step]
    data_requirement: str
    baseline: str
    metric: str
    prediction: str
    refuting_result: str
    premises: list[Premise]
    critique: Critique
    prior_art: PriorArt = field(default_factory=PriorArt)
    scores: Scores | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    query: str
    search: str
    created_at: str
    models: dict[str, str]
    settings: dict
    near: list[dict]
    passages: dict[str, Passage]
    bottlenecks: list[Bottleneck]
    cards: dict[str, Card]
    transfers: list[Transfer]
    calls: dict[str, int]
    seconds: float
    warnings: list[str] = field(default_factory=list)


def new_run_id(query: str) -> str:
    h = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{now}-{h}"


def card_to_dict(c: Card) -> dict[str, Any]:
    return asdict(c)


def card_from_dict(d: dict[str, Any]) -> Card:
    d_copy = dict(d)
    d_copy["evidence"] = Premise(**d_copy["evidence"])
    return Card(**d_copy)


def result_to_dict(r: RunResult) -> dict[str, Any]:
    return asdict(r)


def result_from_dict(d: dict[str, Any]) -> RunResult:
    d_copy = dict(d)
    passages = {k: Passage(**v) for k, v in d_copy.get("passages", {}).items()}
    cards = {k: card_from_dict(v) for k, v in d_copy.get("cards", {}).items()}
    bottlenecks = []
    for b in d_copy.get("bottlenecks", []):
        b_dict = dict(b)
        b_dict["premises"] = [Premise(**p) for p in b_dict.get("premises", [])]
        bottlenecks.append(Bottleneck(**b_dict))
    transfers = []
    for t in d_copy.get("transfers", []):
        t_dict = dict(t)
        t_dict["mapping"] = [MappingRow(**m) for m in t_dict.get("mapping", [])]
        t_dict["identifiability"] = Identifiability(**t_dict["identifiability"]) if t_dict.get("identifiability") else Identifiability()
        t_dict["steps"] = [Step(**s) for s in t_dict.get("steps", [])]
        t_dict["premises"] = [Premise(**p) for p in t_dict.get("premises", [])]
        t_dict["critique"] = Critique(**t_dict["critique"]) if t_dict.get("critique") else Critique()
        t_dict["prior_art"] = PriorArt(**t_dict["prior_art"]) if t_dict.get("prior_art") else PriorArt()
        t_dict["scores"] = Scores(**t_dict["scores"]) if t_dict.get("scores") else None
        transfers.append(Transfer(**t_dict))

    return RunResult(
        run_id=d_copy["run_id"],
        query=d_copy["query"],
        search=d_copy["search"],
        created_at=d_copy["created_at"],
        models=d_copy.get("models", {}),
        settings=d_copy.get("settings", {}),
        near=d_copy.get("near", []),
        passages=passages,
        bottlenecks=bottlenecks,
        cards=cards,
        transfers=transfers,
        calls=d_copy.get("calls", {}),
        seconds=d_copy.get("seconds", 0.0),
        warnings=d_copy.get("warnings", []),
    )
