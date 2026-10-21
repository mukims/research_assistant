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
1. List at most {n} techniques this paper uses that someone in another field could reuse. A finding is not a technique.
2. problem_solved states the problem the technique solves with no material, system, organism or field named.
3. context says how this paper used it.
4. evidence is one sentence the cited passage states, citing exactly one key, written like C7.
5. If the paper uses no reusable technique, return an empty list.
6. Reply with the JSON object only.

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
