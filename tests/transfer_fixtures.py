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
    )
