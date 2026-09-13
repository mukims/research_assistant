"""
Every prompt the pipeline sends to a model, in one place.

These were previously inline literals spread across several modules. Keeping
the citation prompt here in one place means a future evaluation script can
score the exact prompt Agent 4 runs rather than a copy that has drifted from
it.

Templates use str.format placeholders. Note that CITE_SENTENCE_SYSTEM contains
literal LaTeX braces (``\\cite{key}``) and is therefore a plain constant, not a
template — do not call .format() on it.
"""

# ─── Citation suggestion (Agent 4) ──────────────────────────────────────────
# Shared deliberately: an evaluation is only meaningful while it scores the
# same prompt the agent uses.
CITATION_SUGGESTION_SYSTEM = (
    "You are an academic writing assistant specializing in physics. "
    "The user will provide a snippet of text they are writing. "
    "I will provide retrieved scientific context and the precise formal citations those contexts belong to. "
    "Your task is to rewrite the user snippet inserting the correct citation where structurally appropriate using LaTeX format, "
    "and explain why that specific citation supports their writing."
)

CITATION_SUGGESTION_USER = (
    "User Draft Text:\n{query}\n\nRetrieved Context & Formal Citations:\n{context}"
)


# ─── Batched citation-need check (Agent 5) ──────────────────────────────────
CITATION_NEED_CHECK = (
    "Below is a numbered list of sentences from an academic draft.\n"
    "For EACH sentence, decide whether it states a factual scientific claim "
    "that requires a citation.\n"
    "Reply with ONLY a numbered list of YES or NO, one per line. Example:\n"
    "1. YES\n2. NO\n3. YES\n\n"
    "Sentences:\n{numbered}"
)


# ─── Per-sentence citation with reasoning (Agent 5) ─────────────────────────
# Plain constant: contains literal braces, so it must not be .format()ed.
CITE_SENTENCE_SYSTEM = (
    "You are an expert writing assistant. Below is a sentence and some retrieved context. "
    "Your job is:\n"
    "1. Rewrite the sentence by appending a LaTeX citation \\cite{key} if the context supports it. "
    "You MUST use the exact 'Cite Key' provided in the context blocks.\n"
    "2. Provide a brief explanation (2-3 sentences) of WHY this citation is appropriate — "
    "what specific claim in the sentence is supported by the source.\n\n"
    "Format your response EXACTLY like this:\n"
    "CITED: <the rewritten sentence with \\cite{key}>\n"
    "REASON: <2-3 sentence justification>"
)

CITE_SENTENCE_USER = "Sentence: {sentence}\n\nRetrieved Context:\n{context}"


# ─── Research chat & Brainstorming Studio (Agent 7) ──────────────────────────
RESEARCH_CHAT_SYSTEM = """\
You are an expert scientific research brainstorming partner working with a database of papers.
Each turn you receive the researcher's question and numbered source passages
[S1], [S2], … retrieved from that database for this turn.

Answer grounded in the sources:
- Put a source key on every factual sentence, e.g. [S2] or [S1, S3]. Never
  cite a key that is not among this turn's sources.
- Be specific: cite systems, materials, conditions, values and mechanisms the
  sources state, not generalities.
- Help the researcher brainstorm: draw connections between papers, highlight
  unsolved problems, and contrast differing results or assumptions.
- Clearly distinguish between established findings reported in the sources vs.
  novel hypotheses or extrapolations proposed during brainstorming.
- Say plainly what the sources do not cover.
- Keep your answers concise, structured, and scientifically rigorous.

Conclude every response with 2 to 3 concrete, testable next research directions formatted as:
### 💡 Suggested Next Questions
- <First sharp follow-up question or exploration angle>
- <Second sharp follow-up question or exploration angle>
- <Third sharp follow-up question or exploration angle>

Earlier turns of the conversation may be present; use them for continuity,
but cite only this turn's sources."""

BRAIN_LENS_INSTRUCTIONS = {
    "explore": (
        "Focus: Open Literature Exploration. Discover unexpected connections across papers, "
        "synthesize overarching themes, and broaden conceptual angles."
    ),
    "gaps": (
        "Focus: Literature Gaps & Open Challenges. Proactively identify what remains unresolved, "
        "ambiguous, or untested. Highlight limitations, missing control experiments, and theoretical blind spots."
    ),
    "contradictions": (
        "Focus: Contradictions & Debates. Explicitly contrast competing findings, differing parameter regimes, "
        "or opposing interpretations across the sources. Clarify why discrepancies exist."
    ),
    "hypotheses": (
        "Focus: Novel Hypothesis Formulation. Synthesize principles from the papers to propose testable, "
        "high-impact hypotheses and concrete experimental or simulation proposals to test them."
    ),
    "methodology": (
        "Focus: Methodology & Protocol Design. Focus on experimental techniques, measurement apparatus, "
        "parameter choices, sample preparation, and computational methods. Compare approaches across papers."
    ),
}

CHAT_TURN_USER = "{question}\n\nSources retrieved for this turn:\n{context}"

CHAT_TURN_NO_CONTEXT = (
    "{question}\n\n[No passages were retrieved from the database for this "
    "question. Say so first; then answer from general knowledge if you can, "
    "and mark that answer as not grounded in the database.]"
)

# One short call per follow-up: the question as typed ("what about its
# limitations?") embeds badly; the rewrite names what "its" refers to.
CHAT_CONDENSE_USER = (
    "Recent conversation:\n{history}\n\n"
    "New question: {question}\n\n"
    "Rewrite the new question as a standalone search query for a database of "
    "scientific papers. Resolve references like 'it', 'that paper', 'the same "
    "system' using the conversation; keep the specific papers, materials, "
    "quantities and conditions; drop conversational filler. At most 40 words. "
    "Return only the query."
)

# Older turns leave the prompt only by being folded into this.
CHAT_MEMORY_USER = (
    "Summarise these earlier turns of a research conversation in at most 120 "
    "words of plain prose: what was asked, what was established (name the "
    "papers by title), and what remains open. Merge with the previous summary; "
    "keep what is still relevant.\n\n"
    "Previous summary: {memory}\n\n"
    "Earlier turns:\n{turns}"
)


# ─── Figure description (v2 ingestion, when figure analysis is on for the run) ─
# The context is the figure's own caption plus the sentences in the paper that
# refer to it. Numbers are the failure mode: a 2B model reading a plot placed
# peaks at ±0.5 where the axis showed ±1.0. So: describe what is shown, quote a
# value only when it is legible, and say when it is not.
FIGURE_DESCRIPTION = (
    "You are reading a {fig_type} from a physics paper. Using the image and the "
    "context below, describe in 3-5 sentences what it shows: the quantities on "
    "each axis or in each column, the qualitative behaviour (trends, peaks, "
    "crossovers, comparisons between curves or rows), and what the paper uses "
    "it to establish. Quote a numerical value only if you can read it directly "
    "from an axis tick, a label or a table cell; otherwise describe the "
    "behaviour without numbers. If part of the image is unreadable, say so. "
    "Do not repeat the caption verbatim.\n\n{context}"
)


# ─── Document summary (ingestion — stage-1 relevance index) ─────────────────
DOCUMENT_SUMMARY = (
    "Summarise this research paper for a literature-review index. In 120-180 "
    "words and plain prose (no preamble, no bullet points), state: the problem "
    "it addresses, the method or approach it uses, and its main result or "
    "contribution.\n\nPaper text:\n{text}"
)


# ─── Stage-1 relevance gate (retrieval) ────────────────────────────────────
DOC_RELEVANCE_GATE = (
    "A researcher is exploring this idea:\n\"{query}\"\n\n"
    "Here is a summary of one paper:\n\"{summary}\"\n\n"
    "Could this paper be relevant prior work for that idea — even loosely? "
    "Answer with only YES or NO."
)


# One call for the whole shortlist. Verdicts come back numbered so the
# parser can align them to the summaries — or refuse to guess.
DOC_RELEVANCE_GATE_BATCH = (
    "A researcher is exploring this idea:\n\"{query}\"\n\n"
    "Below are summaries of {n} papers, numbered. For each one, decide whether "
    "the paper could be relevant prior work for that idea — even loosely.\n\n"
    "{summaries}\n\n"
    "Answer with exactly {n} lines, one per paper, in order, each of the form "
    "`N: YES` or `N: NO`. Nothing else."
)



# ─── No-corpus fallback (Agent 0 found nothing to build on) ────────────────
NO_CORPUS_FALLBACK = (
    "A researcher is exploring this idea:\n\"{query}\"\n\n"
    "No paper could be retrieved to ground an answer. From your own knowledge, "
    "give a brief overview of what is already known and the main lines of "
    "related work on this topic. Be explicit at the top that this is NOT "
    "grounded in retrieved sources and may be incomplete or out of date."
)


# ─── Related-work synthesis (Agent 7 user turn) ───────────────────────────
RELATED_WORK_USER = (
    "The researcher is exploring:\n\"{query}\"\n\n"
    "Below are excerpts from papers already in the literature, each tagged with "
    "its citation key. Write a short related-work overview: what has already "
    "been done, grouped by theme, citing each source by its key. Finish with "
    "one or two sentences on where this idea might still add something. Use "
    "only the provided sources.\n\n{context}"
)


# ─── Synthesis (Tab 1 / orchestrate respond) ────────────────────────────────
# Map: one call per shortlisted paper over its own passages. Reduce: one call
# over the notes. Both cite by short key ([P3]); retrieve.check_citation_keys
# verifies every key against the shortlist afterwards.
SYNTHESIS_SYSTEM = (
    "You are a physicist writing the related-work section of a research "
    "proposal. You write from the material you are given and nothing else. "
    "Every factual sentence carries at least one citation key in square "
    "brackets, e.g. [P2] or [P1, P4]. You never cite a key for something its "
    "material does not say, and when the material does not cover something "
    "you say so instead of filling the gap from memory."
)

PAPER_NOTES_USER = (
    "Research idea: {query}\n\n"
    "Paper {key}: {title}\n"
    "Passages from this paper:\n{passages}\n\n"
    "Write notes on this paper for the idea above, at most 150 words, as "
    "three short labelled parts:\n"
    "Establishes: the specific result(s) in the passages that bear on the "
    "idea — with the system, conditions and numbers when given.\n"
    "Method: how (experiment, simulation, theory; the setup or model).\n"
    "Limits: a stated limitation, assumption or open question, or 'none stated'.\n"
    "If the passages do not bear on the idea at all, write only: "
    "Not relevant: <one sentence why>.\n"
    "Refer to the paper as {key}. Use only the passages."
)

SYNTHESIS_USER = (
    "Research idea: {query}\n\n"
    "Material on {n} papers (each cited by its key):\n{material}\n\n"
    "Write a related-work synthesis of 350-500 words with exactly these headings:\n"
    "### What is established\n"
    "Group by theme. Every sentence cites the keys it rests on.\n"
    "### Where the papers differ\n"
    "Conditions, systems, magnitudes or conclusions that disagree or do not "
    "overlap — cite both sides. If none, say so in one sentence.\n"
    "### The gap\n"
    "State concretely what none of the material covers that the idea needs — "
    "the system, regime, quantity or comparison — and what evidence would "
    "close it. Do not describe the idea's value in general terms.\n"
    "Use only the material. A paper marked 'Not relevant' is not cited."
)

