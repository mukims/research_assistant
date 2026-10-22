"""Prompts and passage formatting for technique transfer."""

from __future__ import annotations

from research_assistant.transfer.models import Passage

SYSTEM = """You are an expert scientific methodologist specializing in cross-disciplinary technique transfer.
You analyze research papers to extract reusable computational, mathematical, statistical, experimental, or analytical methods.
You state problem definitions generically without domain-specific jargon so they can be matched across fields."""

CARDS_USER = """TASK: cards
Extract reusable techniques from the following paper.

Paper: {citation}

Passages:
{passages}

Rules:
1. A technique is a method with steps: an algorithm, model, estimator, approximation, training scheme or fabrication process that someone in another field could carry out from your description. List at most {n} that this paper uses. Most papers have 0 to 2.
2. These are not techniques:
   - a finding or a result;
   - a survey's topic, or a research direction the paper discusses;
   - building a database, a benchmark, a user interface or file formats;
   - routine handling, measurement or characterisation;
   - "combining" or "integrating" two things, unless the paper says exactly how.
3. mechanism gives 2 to 5 steps saying what is computed or done, in order, concretely enough to implement. Name the operation, such as "rescale the matrix so its spectrum lies in [-1, 1]", not a goal such as "improve accuracy". If you cannot write the steps from the passages, it is not a technique: leave it out.
4. name is the method's own name as its field knows it, including the model or object it works on, such as "restricted Boltzmann machine trained by contrastive divergence" or "kernel polynomial method with Jackson damping", not a paraphrase such as "truncated sampling gradient approximation".
5. problem_solved states the problem the technique solves with no material, system, organism or field named.
6. context says how this paper used it.
7. evidence is one sentence the cited passage states, citing exactly one key, written like C7.
8. If the paper uses no technique by rule 1, return an empty list. That is the right answer for surveys, databases and benchmarks.
9. Reply with the JSON object only.

Format your response as a JSON object matching this schema:
{{
  "techniques": [
    {{
      "name": "The method's own name, with the model it uses",
      "problem_solved": "Generic statement of the problem solved without field-specific nouns",
      "context": "How this paper used it",
      "inputs": "What input data/structure it requires",
      "outputs": "What outputs/results it yields",
      "assumptions": ["Key assumption 1", "Key assumption 2"],
      "mechanism": ["Step 1: the concrete operation", "Step 2: the next operation"],
      "evidence": {{
        "claim": "One sentence the cited passage states",
        "key": "C1"
      }}
    }}
  ]
}}
"""


def format_passages(passages: list[Passage], max_chars: int) -> str:
    """Format passages into tagged text blocks, capped from tail at max_chars."""
    blocks = []
    for p in passages:
        sec = p.section or "other"
        blocks.append(f"[{p.key}] ({p.citation} · {sec}) {p.text}")
    joined = "\n\n".join(blocks)
    if len(joined) > max_chars:
        return joined[:max_chars]
    return joined


BOTTLENECKS_USER = """TASK: bottlenecks
Identify up to {n} technical bottlenecks in the following problem.

Query: {query}

Passages:
{passages}

Rules:
1. A bottleneck is what makes this problem expensive, ill-posed or data-starved, as the passages show it. List at most {n}.
2. quantities are the quantities the bottleneck limits, as the field names them.
3. current_method is how the passages solve it today.
4. generic restates the bottleneck three ways — specific, mid and broad — naming no material, system, organism or field, so that a paper from another field could match it.
5. Each bottleneck has 1 to 3 premises: one sentence the cited passage states, citing exactly one key, written like C7.
6. inference is true when the problem is to infer unknowns from measurements or observations (an inverse problem), and false when it is to compute, simulate, represent or optimise something.
7. Reply with the JSON object only.

Format your response as a JSON object matching this schema:
{{
  "bottlenecks": [
    {{
      "statement": "What makes this problem difficult",
      "quantities": ["quantity1", "quantity2"],
      "current_method": "How it is solved today",
      "inference": true,
      "generic": {{
        "specific": "Restatement without field nouns at specific level",
        "mid": "Restatement at mid level",
        "broad": "Restatement at broad level"
      }},
      "premises": [
        {{
          "claim": "One sentence the cited passage states",
          "key": "C1"
        }}
      ]
    }}
  ]
}}
"""


TRANSFER_USER = """TASK: transfer
Propose a cross-disciplinary technique transfer from the source technique to solve the target bottleneck.

Target bottleneck:
Statement: {bottleneck_statement}
Quantities: {bottleneck_quantities}
Current method: {bottleneck_current_method}
Premises:
{bottleneck_premises}

Source technique card:
Name: {card_name}
Citation: {card_citation}
Problem solved: {card_problem_solved}
Context: {card_context}
Inputs: {card_inputs}
Outputs: {card_outputs}
Assumptions: {card_assumptions}
Mechanism:
{card_mechanism}

Passages:
{passages}

Rules:
1. Map every role the technique has in its source onto the target problem, one row each; mark each row's assumption holds, breaks or enforce, and core if the transfer fails without it.
2. Identifiability: say what the measurements are, what the unknowns are, any symmetry that makes different unknowns give the same measurements, and whether the measurements determine the unknowns.
3. Steps name an input, an output and a tool. data_requirement gives a number and its basis. baseline is the current method. metric measures one of the quantities. prediction states a direction or a number; refuting_result names the outcome that would refute the transfer.
4. Give 2 to 4 premises: one sentence the cited passage states, citing exactly one key, written like C7; at least one from the source paper.
5. Reply with the JSON object only.

Format your response as a JSON object matching this schema:
{{
  "title": "Title of the proposed transfer",
  "summary": "One-paragraph summary of the proposed transfer and how it works",
  "mapping": [
    {{
      "source_role": "Role in source technique",
      "target_role": "Mapped role in target problem",
      "assumption": "Assumption underlying this role",
      "status": "holds",
      "core": true,
      "note": "Optional note"
    }}
  ],
  "identifiability": {{
    "measurements": "Measurements available",
    "unknowns": "Unknowns to be determined",
    "symmetries": ["Symmetry 1"],
    "verdict": "determined",
    "argument": "Argument for why measurements determine unknowns"
  }},
  "steps": [
    {{
      "action": "Concrete action",
      "input": "Input data or structure",
      "output": "Output data or structure",
      "tool": "Tool or method used"
    }}
  ],
  "data_requirement": "Sample count or training size with justification",
  "baseline": "Current method",
  "metric": "Metric measuring one of the quantities",
  "prediction": "Directional or numerical prediction",
  "refuting_result": "Concrete outcome that would refute the transfer",
  "premises": [
    {{
      "claim": "One sentence the cited passage states",
      "key": "C1"
    }}
  ]
}}
"""


CRITIQUE_USER = """TASK: critique
Review this transfer proposal to reject it if you can.

Transfer:
Title: {title}
Summary: {summary}

Mapping:
{mapping}

Identifiability:
Measurements: {measurements}
Unknowns: {unknowns}
Symmetries: {symmetries}
Verdict: {identifiability_verdict}
Argument: {identifiability_argument}

Steps:
{steps}

Baseline: {baseline}
Metric: {metric}
Prediction: {prediction}
Refuting result: {refuting_result}

Rules:
1. You are reviewing this transfer to reject it if you can. For each mapping row say holds, breaks or enforce, with a reason; say whether the problem shapes match (yes, partial, no); override the identifiability verdict if it is wrong; list your strongest objections.
2. Reply with the JSON object only.

Format your response as a JSON object matching this schema:
{{
  "shape_match": "yes",
  "rows": [
    {{
      "row": 0,
      "status": "holds",
      "reason": "Reason for status"
    }}
  ],
  "identifiability": null,
  "objections": [
    "Strongest objection 1"
  ]
}}
"""


REVISE_USER = """TASK: revise
Address each objection raised by the critic.

Transfer:
Title: {title}
Summary: {summary}

Objections from critic:
{objections}

Rules:
1. Each response addresses one objection by its 0-based index. Rebut it by saying why the assumption holds in the target problem, or what change to the transfer enforces it; restating the claim is not a rebuttal. Concede only when the transfer cannot work on this point whatever is changed: a conceded objection on a core row rejects the transfer.
2. Reply with the JSON object only.

Format your response as a JSON object matching this schema:
{{
  "responses": [
    {{
      "objection": 0,
      "response": "Detailed technical response",
      "concede": false
    }}
  ]
}}
"""


