"""Shared fixtures for technique transfer tests."""

from research_assistant.transfer.models import (
    Bottleneck,
    Card,
    Critique,
    Identifiability,
    MappingRow,
    Passage,
    Premise,
    PriorArt,
    RunResult,
    Scores,
    Step,
    Transfer,
)


def premise(
    judgement: str | None = "Supports",
    key: str = "C1",
    document: str = "a.pdf",
    chunk_index: int = 1,
    claim: str | None = None,
    span: bool | None = True,
) -> Premise:
    return Premise(
        claim=claim or f"Claim for {key}",
        key=key,
        document=document,
        chunk_index=chunk_index,
        judgement=judgement,
        span_verified=span,
        supporting_span="sample supporting span" if span else None,
    )


def good_transfer() -> tuple[Transfer, Bottleneck, Card]:
    b = Bottleneck(
        id="B1",
        statement="Matrix inversion bottleneck in RGF",
        quantities=["conductance"],
        current_method="recursive Green's function",
        generic={
            "specific": "large matrix inversion",
            "mid": "expensive forward map",
            "broad": "repeated simulation",
        },
        premises=[premise("Supports", key="C1", document="near.pdf", chunk_index=1)],
    )

    card = Card(
        card_id="src.pdf#0",
        document="src.pdf",
        citation="Source Author 2020",
        name="latent-space surrogate",
        problem_solved="expensive forward map evaluated many times",
        context="molecular property prediction",
        inputs="graph representation",
        outputs="target property",
        assumptions=["smooth latent space"],
        evidence=premise("Supports", key="C2", document="src.pdf", chunk_index=2),
    )

    t = Transfer(
        id="B1.T1",
        bottleneck_id="B1",
        card_id="src.pdf#0",
        card_document="src.pdf",
        level="specific",
        retrieval_rank=1,
        title="Latent-space surrogate for recursive Green's function",
        summary="Transferring latent surrogate modeling to quantum transport.",
        mapping=[
            MappingRow(
                source_role="molecular graph",
                target_role="Hamiltonian matrix",
                assumption="isomorphic feature map",
                status="holds",
                core=True,
                note="direct mapping",
            ),
            MappingRow(
                source_role="property score",
                target_role="conductance value",
                assumption="scalar prediction",
                status="holds",
                core=False,
            ),
            MappingRow(
                source_role="latent search",
                target_role="parameter optimization",
                assumption="convex landscape",
                status="holds",
                core=False,
            ),
        ],
        identifiability=Identifiability(
            measurements="conductance at Fermi energy",
            unknowns="disorder configuration",
            symmetries=["time reversal"],
            verdict="determined",
            argument="The forward map is well-conditioned and uniquely invertible for physical configurations.",
        ),
        steps=[
            Step(action="train surrogate", input="Hamiltonian matrices", output="conductance", tool="PyTorch"),
            Step(action="sample latent space", input="latent vectors", output="candidate structures", tool="Gaussian process"),
            Step(action="evaluate error", input="predicted conductance", output="accuracy metric", tool="benchmark runner"),
        ],
        data_requirement="about 10000 RGF samples",
        baseline="plain recursive Green's function search",
        metric="conductance error per 1000 RGF calls",
        prediction="the conductance error will decrease twofold at equal cost",
        refuting_result="conductance error increases under equal compute",
        premises=[
            premise("Supports", key="C2", document="src.pdf", chunk_index=2),
            premise("Supports", key="C1", document="near.pdf", chunk_index=1),
        ],
        critique=Critique(
            shape_match="yes",
            rows=[],
            identifiability="determined",
            objections=[
                {
                    "text": "Computational complexity might scale poorly with disorder.",
                    "answered": True,
                    "response": "Disorder configurations are projected onto low-dimensional invariant basis.",
                    "conceded": False,
                }
            ],
        ),
        prior_art=PriorArt(status="not_found", hits=[]),
        scores=Scores(depth=1.0, checks={f"check_{i}": True for i in range(1, 11)}, reward=1.25),
        warnings=[],
    )

    return t, b, card


def sample_result() -> RunResult:
    t1, b1, card1 = good_transfer()

    # Second transfer set aside as not_identifiable
    t2 = Transfer(
        id="B1.T2",
        bottleneck_id="B1",
        card_id="src.pdf#0",
        card_document="src.pdf",
        level="mid",
        retrieval_rank=2,
        title="Autoencoder parameter recovery",
        summary="Autoencoding conductivity to find impurity configurations.",
        mapping=list(t1.mapping),
        identifiability=Identifiability(
            measurements="conductance",
            unknowns="impurity positions",
            verdict="underdetermined",
            argument="Multiple impurity distributions produce identical conductance traces.",
        ),
        steps=list(t1.steps),
        data_requirement="5000 samples",
        baseline="direct search",
        metric="position accuracy",
        prediction="recovers 90 percent of positions",
        refuting_result="position error exceeds random baseline",
        premises=list(t1.premises),
        critique=Critique(shape_match="yes", identifiability="underdetermined"),
        prior_art=PriorArt(status="not_found"),
        scores=Scores(depth=0.5, checks={}, reward=0.0, set_aside="not_identifiable"),
    )

    near_chunk = Passage(
        key="C1",
        chunk_index=1,
        document="near.pdf",
        citation="Near Author 2021",
        section="methods",
        text="RGF method details...",
        role="near",
    )
    src_chunk = Passage(
        key="C2",
        chunk_index=2,
        document="src.pdf",
        citation="Source Author 2020",
        section="methods",
        text="Latent space surrogate details...",
        role="source",
    )

    return RunResult(
        run_id="20261021-120000-abcdef12",
        query="conductance calculation acceleration",
        search="best_of_n",
        created_at="2026-10-21T12:00:00Z",
        models={"generator": "gemma4:e2b", "judge": "gemini-3.5-flash-lite", "critic": "gemma4:e2b"},
        settings={"temperature": 0.7},
        near=[{"document": "near.pdf", "score": 0.95}],
        passages={"C1": near_chunk, "C2": src_chunk},
        bottlenecks=[b1],
        cards={"src.pdf#0": card1},
        transfers=[t1, t2],
        calls={"llm": 5, "judge": 4, "critic": 2, "embed": 3, "prior_art": 1},
        seconds=12.5,
        warnings=[],
    )


def card(document: str = "a.pdf", verified: bool = True) -> Card:
    return Card(
        card_id=f"{document}#0",
        document=document,
        citation=f"Citation for {document}",
        name="test technique",
        problem_solved="generic problem description",
        context="specific context",
        inputs="input data",
        outputs="output data",
        assumptions=["basic assumption"],
        evidence=premise("Supports" if verified else "Unclear / insufficient evidence", key="C1", document=document),
        mechanism=["step 1: compute", "step 2: apply"],
    )


def good_transfer_reply(**overrides) -> str:
    t, _, _ = good_transfer()
    data = {
        "title": t.title,
        "summary": t.summary,
        "mapping": [
            {
                "source_role": r.source_role,
                "target_role": r.target_role,
                "assumption": r.assumption,
                "status": r.status,
                "core": r.core,
                "note": r.note,
            }
            for r in t.mapping
        ],
        "identifiability": {
            "measurements": t.identifiability.measurements,
            "unknowns": t.identifiability.unknowns,
            "symmetries": t.identifiability.symmetries,
            "verdict": t.identifiability.verdict,
            "argument": t.identifiability.argument,
        },
        "steps": [
            {"action": s.action, "input": s.input, "output": s.output, "tool": s.tool}
            for s in t.steps
        ],
        "data_requirement": t.data_requirement,
        "baseline": t.baseline,
        "metric": t.metric,
        "prediction": t.prediction,
        "refuting_result": t.refuting_result,
        "premises": [
            {"claim": "near fact", "key": "C1"},
            {"claim": "source fact", "key": "C2"},
        ],
    }
    data.update(overrides)
    return json.dumps(data)


import json
import math
import re
from dataclasses import dataclass, field
import numpy as np

from research_assistant.transfer.cards import CardIndex
from research_assistant.transfer.deps import Deps


@dataclass
class WorldLog:
    chat_calls: int = 0
    critic_calls: int = 0
    judged: list[str] = field(default_factory=list)
    searches: list[tuple] = field(default_factory=list)
    progress: list[tuple] = field(default_factory=list)


def world(**overrides) -> tuple[Deps, WorldLog]:
    gate_keeps = overrides.get("gate_keeps", 3)
    card_names = overrides.get("card_names", {})
    replies = overrides.get("replies", {})
    cancel_after = overrides.get("cancel_after", 9999)

    log = WorldLog()
    chat_kind_counts: dict[str, int] = {}
    transfers_written = [0]

    all_docs = ["n0", "n1", "n2", "n3", "n4", "n5", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    doc_to_chunk = {doc: i for i, doc in enumerate(all_docs)}
    texts = [f"text of {doc}" for doc in all_docs]
    metadatas = [{"document": doc, "section": "methods", "citation": f"Citation {doc}"} for doc in all_docs]

    def _angle_vec(deg: float) -> list[float]:
        rad = math.radians(deg)
        return [math.cos(rad), math.sin(rad)]

    def embed_query(text: str) -> list[float]:
        t = text.strip()
        if t == "g1-specific":
            return _angle_vec(0.0)
        if t == "g1-mid":
            return _angle_vec(5.0)
        if t == "g1-broad":
            return _angle_vec(10.0)
        if t.startswith("g2-"):
            if "mid" in t or t == "g2-2":
                return _angle_vec(95.0)
            if "broad" in t or t == "g2-3":
                return _angle_vec(100.0)
            return _angle_vec(90.0)
        return _angle_vec(45.0)

    def embed_documents(doc_texts: list[str]) -> list[list[float]]:
        return [embed_query(t) for t in doc_texts]

    card_specs = [
        ("s1", 2.0, True),
        ("s2", 7.0, True),
        ("s3", 12.0, True),
        ("s4", 92.0, True),
        ("s5", 97.0, True),
        ("s6", 102.0, True),
        ("n0", 1.0, True),
        ("s7", 91.0, False),
    ]
    card_list = []
    card_vecs = []
    for doc, ang, ver in card_specs:
        c_name = card_names.get(doc, f"method {doc}")
        ev = Premise(
            claim=f"Claim for {doc}",
            key="C1",
            document=doc,
            chunk_index=doc_to_chunk[doc],
            judgement="Supports" if ver else "Unclear / insufficient evidence",
            span_verified=ver,
            supporting_span="sample span" if ver else None,
        )
        c = Card(
            card_id=f"{doc}#0",
            document=doc,
            citation=f"Citation {doc}",
            name=c_name,
            problem_solved=f"problem solved by {doc}",
            context=f"context for {doc}",
            inputs="input",
            outputs="output",
            assumptions=["assumption"],
            evidence=ev,
            mechanism=["step 1: compute", "step 2: apply"],
        )
        card_list.append(c)
        card_vecs.append(_angle_vec(ang))
    cards_index = CardIndex(card_list, np.array(card_vecs, dtype=float))

    def shortlist_fn(query: str, k: int) -> list[dict]:
        return [
            {"document": f"n{i}", "citation": f"Citation n{i}", "score": 1.0 - i * 0.1}
            for i in range(min(k, 6))
        ]

    def gate_fn(query: str, sl: list[dict]) -> list[dict]:
        return sl[:gate_keeps]

    def search_fn(query: str, top_k: int, doc_filter: set | None, exclude_docs: set | None) -> list[dict]:
        doc_filter_set = set(doc_filter) if doc_filter else set()
        exclude_set = set(exclude_docs) if exclude_docs else set()
        log.searches.append((doc_filter_set, exclude_set))
        hits = []
        for doc in sorted(doc_filter_set):
            if doc in exclude_set:
                continue
            c_idx = doc_to_chunk[doc]
            hits.append({
                "chunk_index": c_idx,
                "document": doc,
                "text": f"text of {doc}",
                "metadata": {"document": doc, "section": "methods", "citation": f"Citation {doc}"},
                "score": 1.0,
            })
        return hits[:top_k]

    def chat_fn(messages: list[dict], temperature: float = 0.7) -> str:
        log.chat_calls += 1
        content = messages[-1].get("content", "")
        kind = "unknown"
        if "TASK: bottlenecks" in content:
            kind = "bottlenecks"
        elif "TASK: transfer" in content:
            kind = "transfer"
        elif "TASK: revise" in content:
            kind = "revise"
        elif "TASK: cards" in content:
            kind = "cards"

        chat_kind_counts[kind] = chat_kind_counts.get(kind, 0) + 1
        call_n = chat_kind_counts[kind]
        if kind == "transfer":
            transfers_written[0] += 1
        if (kind, call_n) in replies:
            rep = replies[(kind, call_n)]
            if isinstance(rep, list):
                if rep:
                    return rep.pop(0) if len(rep) > 1 else rep[0]
                return ""
            return rep

        if kind == "bottlenecks":
            return json.dumps({
                "bottlenecks": [
                    {
                        "statement": "Bottleneck 1",
                        "quantities": ["conductance"],
                        "current_method": "recursive Green's function",
                        "generic": {
                            "specific": "g1-specific",
                            "mid": "g1-mid",
                            "broad": "g1-broad",
                        },
                        "premises": [
                            {"claim": "bottleneck fact one", "key": "C1"},
                            {"claim": "bottleneck fact two", "key": "C2"},
                        ],
                    },
                    {
                        "statement": "Bottleneck 2",
                        "quantities": ["conductance"],
                        "current_method": "recursive Green's function",
                        "generic": {
                            "specific": "g2-specific",
                            "mid": "g2-mid",
                            "broad": "g2-broad",
                        },
                        "premises": [
                            {"claim": "bottleneck fact two", "key": "C2"},
                            {"claim": "bottleneck fact three", "key": "C3"},
                        ],
                    },
                ]
            })

        if kind == "transfer":
            keys = re.findall(r"\[(C\d+)\]", content)
            k1 = keys[0] if keys else "C1"
            k2 = keys[-1] if keys else "C2"
            t, _, _ = good_transfer()
            return json.dumps({
                "title": t.title,
                "summary": t.summary,
                "mapping": [
                    {"source_role": r.source_role, "target_role": r.target_role, "assumption": r.assumption, "status": r.status, "core": r.core, "note": r.note}
                    for r in t.mapping
                ],
                "identifiability": {
                    "measurements": t.identifiability.measurements,
                    "unknowns": t.identifiability.unknowns,
                    "symmetries": t.identifiability.symmetries,
                    "verdict": t.identifiability.verdict,
                    "argument": t.identifiability.argument,
                },
                "steps": [
                    {"action": s.action, "input": s.input, "output": s.output, "tool": s.tool}
                    for s in t.steps
                ],
                "data_requirement": t.data_requirement,
                "baseline": t.baseline,
                "metric": t.metric,
                "prediction": t.prediction,
                "refuting_result": t.refuting_result,
                "premises": [
                    {"claim": f"near fact {k1}", "key": k1},
                    {"claim": f"source fact {k2}", "key": k2},
                ],
            })

        if kind == "revise":
            return json.dumps({
                "responses": [
                    {"objection": 0, "response": "Addressed objection with concrete argument.", "concede": False}
                ]
            })

        return "{}"

    def critic_fn(messages: list[dict]) -> str:
        log.critic_calls += 1
        return json.dumps({
            "shape_match": "yes",
            "rows": [],
            "identifiability": "determined",
            "objections": [
                "Sample critic objection"
            ],
        })

    def judge_fn(claim: str, evidence: str) -> dict:
        log.judged.append(claim)
        return {
            "judgement": "Supports",
            "span_verified": True,
            "supporting_span": "sample supporting span",
            "reason": "verified against text",
        }

    def prior_art_fn(technique_name: str, quantities: list[str], summary: str, before_year: int | None) -> PriorArt:
        status = "done_before" if "known" in technique_name.lower() else "not_found"
        hits = [{"title": technique_name, "similarity": 0.9}] if status == "done_before" else []
        return PriorArt(status=status, hits=hits)

    deps = Deps(
        chat=chat_fn,
        critic=critic_fn,
        judge=judge_fn,
        embed_query=embed_query,
        embed_documents=embed_documents,
        shortlist=shortlist_fn,
        gate=gate_fn,
        search=search_fn,
        prior_art=prior_art_fn,
        texts=texts,
        metadatas=metadatas,
        cards=cards_index,
        cancelled=lambda: transfers_written[0] >= cancel_after,
        progress=lambda d, t, label: log.progress.append((d, t, label)),
    )
    return deps, log

