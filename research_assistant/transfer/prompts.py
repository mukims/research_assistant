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
4. problem_solved states the problem the technique solves with no material, system, organism or field named.
5. context says how this paper used it.
6. evidence is one sentence the cited passage states, citing exactly one key, written like C7.
7. If the paper uses no technique by rule 1, return an empty list. That is the right answer for surveys, databases and benchmarks.
8. Reply with the JSON object only.

Format your response as a JSON object matching this schema:
{{
  "techniques": [
    {{
      "name": "Name of the technique",
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
