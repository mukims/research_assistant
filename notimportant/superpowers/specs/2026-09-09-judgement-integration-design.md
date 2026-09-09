# Judgement — claim–evidence verification as an audit pass

*2026-09-09*

## 1. Purpose

Integrate the standalone `judgement` module (V1.4 claim–evidence verification)
into the research assistant as a **post-hoc audit** over an already-cited
draft. It answers, per citation, whether the cited source actually supports the
claim that cites it — and, separately, whether the retrieved evidence was even
sufficient to tell.

Source: `/home/shardul/Downloads/judgement/`. Copied in, not referenced in
place.

### Scope of this spec

This spec covers the judgement integration only. Deploying the result on Google
Cloud is a **separate spec**, not yet written: three decisions from that design
discussion remain open (access control, whether the GROBID lifecycle buttons
keep working under Compose, and chat/embedding model choice). Nothing here
blocks on those, and nothing here assumes a particular host.

The agreed order is integration first, deployment second — GCP ships the
integrated pipeline, not a pre-judgement one. Two constraints follow from that
and are load-bearing here rather than in the deployment spec: everything added
must route through `shared/llm.py` so a local Ollama checkout keeps working
(§4.3), and nothing may write outside `CITATION_DATA_DIR` (§5.4), since the
deployment mounts a persistent disk there and the app directory is read-only in
the container.

### What the audit does

Agent 5 produces a cited draft. The verifier reads that draft plus its citation
mapping, and for every `\cite{key}` it finds, re-retrieves the best supporting
chunk from that source and asks the judgement prompt for a structured verdict:
three slot verdicts (`finding` / `scope` / `strength`), an aggregate judgement,
an evidence-sufficiency rating, a confidence, and a verbatim supporting span.

## 2. Provenance

| From `judgement/` | Disposition |
|---|---|
| `prompt.md` | copied verbatim |
| `cases/cases.jsonl` | copied verbatim (6 cases) |
| `test_judgement.py` | adapted — flat `tests/`, absolute import |
| `llm.py` | **replaced** by `judgement/judge.py` |
| `__init__.py` | replaced with a package docstring |
| `README.md` | not carried over; content folded into ARCHITECTURE |

### Why `llm.py` is replaced rather than copied

It constructs `OpenAI(...)` directly and reads `JUDGEMENT_MODEL`,
`OPENAI_API_KEY` and `OPENAI_BASE_URL` itself. Every agent in this repo calls
`shared.llm.chat()` precisely so that the backend is one env var and no agent
imports a provider SDK. Copying it in as-is would hard-wire OpenAI and break
the local Ollama path, which is a stated requirement. `build_prompt()` is the
only function that survives unchanged.

Also dropped: the hardcoded `"gpt-5.6"` default (moves to `config.py`) and the
commented-out `temperature=0` (becomes real — see §5.2).

## 3. Layout

```text
research_assistant/
├── judgement/
│   ├── __init__.py
│   ├── judge.py              # build_prompt, judge, parse/validate
│   ├── prompt.md
│   └── cases/
│       └── cases.jsonl
├── agents/
│   └── agent8_verifier.py    # new
├── shared/
│   └── llm.py                # changed — see 5.2
└── config.py                 # changed — see 5.3

tests/
├── test_judgement.py         # adapted from the source module
└── test_verifier.py          # new
```

Outputs, alongside the cited draft and matching Agent 5's existing convention
of `out_path.replace(".txt", …)`:

```text
cited_draft.txt                  (agent 5)
cited_draft_citations.json       (agent 5)
cited_draft_report.md            (agent 5)
cited_draft_verification.json    (agent 8)  ← machine-readable
cited_draft_verification.md      (agent 8)  ← human report
```

## 4. Structural decisions

### 4.1 An audit pass, not an inline gate

The verifier runs after Agent 5, over its output. It does not sit inside the
citing loop, and Agents 4 and 5 are not modified.

**Why.** Judgement is inherently one claim–evidence pair at a time; the slot
decomposition is what makes it accurate and it cannot be batched without
destroying that. Agent 5 already batches its citation-need check at
`CITATION_CHECK_BATCH_SIZE = 20` specifically because per-sentence calls were
too expensive. Putting judgement inline on every candidate roughly triples the
call count of a batch run, which is fine against a hosted API and impractical
against a local CPU model.

**The trade-off, stated plainly.** A bad citation is inserted and then flagged,
rather than blocked. The draft is not automatically corrected. The user reads
the verdict report and decides. Making judgement a gate is a coherent future
change; it is not this one.

### 4.2 Evidence is re-retrieved, not replayed

Agent 5 does not persist the chunk text it cited. `citation_entries` records
only `{"key", "citation"}` per source, and `_report.md` is prose. The evidence
that grounded a citation is therefore not recoverable from Agent 5's output.

The verifier re-runs `hybrid_search` restricted to the cited source and judges
its best-matching chunk.

**Why this is acceptable, and arguably better.** The question the audit answers
becomes "does this source's strongest evidence for this claim support it?" If
even the best chunk fails, the citation is wrong regardless of what Agent 5
saw. The failure mode runs the safe direction: it cannot manufacture support
that the source does not contain.

**Why not make Agent 5 persist the chunks instead.** It would change Agent 5's
output format and its hot path, which §4.1 exists to avoid. Worth revisiting if
the audit ever becomes a gate.

### 4.3 Judgement routes through `shared.llm`

`judge.py` calls `shared.llm.chat()`. It never imports `openai` or `ollama`.

**Why.** Same reason every other agent does: `LLM_BACKEND` stays the single
switch, local users keep Ollama, a deployment gets a hosted API, and the test
suite can inject a fake without a network.

## 5. New and changed components

### 5.1 `judgement/judge.py`

```python
build_prompt(claim: str, citation_evidence: str) -> str
judge(claim: str, citation_evidence: str, model: str | None = None) -> dict
```

`build_prompt` is unchanged from the source module: literal `{{CLAIM}}` and
`{{CITATION_EVIDENCE}}` replacement against `prompt.md`, which is read once at
import.

`judge` calls `chat()` with a single user message, passing
`model=JUDGEMENT_MODEL`, `temperature=JUDGEMENT_TEMPERATURE` and
`options=JUDGEMENT_OLLAMA_OPTIONS` from `config.py` (§5.3). It then parses.
Parsing is stricter than the source module's bare `json.loads`:

1. Strip a leading/trailing markdown code fence if present. The prompt says
   "no code fences"; models emit them anyway, and the source module would
   raise `JSONDecodeError` on every such reply.
2. `json.loads`.
3. Validate that all six top-level fields are present — the source module's
   check — **and** that `judgement`, `confidence` and `evidence_sufficiency`
   each hold a value from their allowed set, and that `slots` has exactly
   `finding`, `scope`, `strength`.
4. Raise `JudgementParseError` (a new exception type) on any failure, carrying
   the raw response so the caller can log what actually came back.

Enum validation matters because the aggregate judgement drives the report's
counts. A model returning `"supports"` or `"Support"` would otherwise flow
through as an unrecognised category and silently distort the summary.

### 5.2 `shared/llm.py` — `temperature` and `options` on `chat()`

```python
def chat(messages, model=None, images=None, temperature=None, options=None) -> ChatResult
```

Both new parameters default to `None`, and when both are `None` the call is
byte-for-byte what it is today. No existing caller changes.

- `temperature` → passed to `client.chat.completions.create(...)` in
  `_openai_chat`, and folded into the options dict in `_ollama_chat`.
- `options` → ollama runtime options, mirroring what `chat_stream` already
  accepts. Ignored by the OpenAI backend.

**Why.** `_openai_chat` currently calls `create(model, messages, stream=False)`
with nothing else, and `_ollama_chat` passes no options at all — so neither
temperature nor `num_ctx` is reachable from `chat()`. The judgement regression
suite asserts exact verdicts, which is not meaningful at a default temperature,
and §5.3 needs `num_ctx`.

### 5.3 `config.py` — judgement settings

```python
JUDGEMENT_MODEL       = os.environ.get("CITATION_JUDGEMENT_MODEL", "") or None   # None → LLM_MODEL
JUDGEMENT_TEMPERATURE = 0.0
JUDGEMENT_TOP_K       = _env_int("CITATION_JUDGEMENT_TOP_K", 1)
JUDGEMENT_OLLAMA_OPTIONS = {"num_ctx": 8192}
```

`JUDGEMENT_MODEL = None` falling back to `LLM_MODEL` follows the existing
`SUMMARY_MODEL` pattern exactly.

`JUDGEMENT_OLLAMA_OPTIONS` exists because the prompt is ~3,760 tokens.
`_ollama_chat` sends no options, so a local run uses the model's default
context — commonly 4096, sometimes 2048. Add an evidence chunk and the JSON
reply and it overflows. Ollama truncates rather than erroring, and what it
truncates is the tail of the prompt, which is where the six worked examples and
the D-versus-E contrast live. Local judgement would degrade silently into
worse verdicts with no signal. 8192 leaves room for prompt, evidence and reply.

`JUDGEMENT_TOP_K = 1` — the verifier judges the single best chunk per cited
source. Raising it judges more chunks per citation and keeps the strongest
verdict, at a proportional cost.

### 5.4 `agents/agent8_verifier.py`

```python
verify_draft(draft_path: str, citations_path: str | None = None,
             search_resources=None) -> dict
```

`citations_path` defaults to `draft_path.replace(".txt", "_citations.json")`.

Per cited sentence:

1. Split the draft with `agent5_batch_citer.split_into_sentences` — reused, not
   reimplemented, so the audit's sentence boundaries match the citer's.
2. Extract keys with `agent5_batch_citer._cite_keys`.
3. Invert the citation mapping. `_citations.json` is
   `{citation_source: cite_N}`; the verifier needs `cite_N → citation_source`.
   A key present in the draft but absent from the mapping is recorded as
   `orphaned` and not judged — Agent 5 already warns about invented keys, and
   this is where that shows up per sentence.
4. Resolve `citation_source → {document}` with one
   `collection.get(where={"citation_source": src})`. Cached across sentences,
   so a source cited twenty times costs one Chroma call.
5. Strip every `\cite{...}` from the sentence to form the claim, so the model
   judges prose rather than LaTeX.
6. `hybrid_search(claim, …, doc_filter=documents, top_k=JUDGEMENT_TOP_K)`.
7. `judge(claim, evidence_chunk_text)`.

A sentence citing several sources yields one judgement per `(sentence, source)`
pair. `shared/search.py` is not modified — `doc_filter` already does what is
needed once the source is resolved to its documents.

**Outputs.** `_verification.json` is the record: draft path, timestamp, model,
per-pair `{sentence_index, claim, cite_key, citation_source, evidence,
slots, judgement, evidence_sufficiency, confidence, supporting_span, reason}`,
plus totals. `_verification.md` renders it: a summary table counting each
judgement category, then a section per flagged citation, worst first —
`Contradicts`, then `Does not support`, then `Unclear / insufficient evidence`,
then `Partially supports`. Clean `Supports` results are counted in the summary
and listed compactly rather than expanded.

A CLI `main()` mirrors the other agents: `--draft`, `--citations`, `--top-k`.

### 5.5 `app.py` — a Verify button in `tab_batch`

`tab_batch` already holds the written draft path and renders the citation
mapping and report. A **Verify citations** button runs `verify_draft` on that
path and renders the summary table plus a per-citation expander.

**Why not a new tab.** A sixth tab for something that only ever operates on
tab 3's output would separate the two halves of one workflow. The button sits
where the artefact it verifies already is.

### 5.6 Documentation

- `README.md` — an Agent 8 row in the pipeline table.
- `ARCHITECTURE.md` — a new section recording §4.1 and §4.2: why the audit sits
  outside the citing loop, and why evidence is re-retrieved.
- `HOW_TO_USE.md` — how to read a verification report, and what
  `evidence_sufficiency` means as distinct from the judgement.

## 6. Data flow

```text
draft.txt
   │  agent 5 (unchanged)
   ▼
cited_draft.txt ─────────────┐
cited_draft_citations.json ──┤
                             ▼
                     agent8_verifier
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  split + extract     resolve source        hybrid_search
  \cite keys          → {document}          (doc_filter, top_k)
        └────────────────────┼────────────────────┘
                             ▼
                     judgement.judge()
                     (shared.llm.chat)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
   cited_draft_verification.json   cited_draft_verification.md
```

## 7. Error handling

| Failure | Behaviour |
|---|---|
| `_citations.json` missing | Abort with a message naming the expected path. Without the mapping no key resolves to a source. |
| Cite key not in the mapping | Record `orphaned`, do not judge, count in the summary. |
| `citation_source` resolves to no documents | Record `unresolved`. Means the corpus changed since the draft was cited. |
| `hybrid_search` returns nothing | Record `no_evidence`, do not call the model. |
| Model reply unparseable after fence-strip | `@retry(max_retries=2)` — two attempts total, since that decorator counts attempts, not retries — then record `parse_failed` with the raw reply and continue. |
| Model reply parses but fails enum validation | Same as unparseable — retried, then recorded. |

Nothing here aborts the run except a missing mapping. A verification pass over
120 citations should not be lost to one malformed reply, and every failure mode
is a category in the report rather than a silent omission — the same principle
Agent 5's report follows in distinguishing "not needed" from "needed, none
made".

Writes go through `shared.atomic.atomic_write_json` and `atomic_write`, like
every other manifest in the pipeline.

## 8. Testing

**`tests/test_judgement.py`** — adapted from the source module. Kept: the
well-formed-cases check and the template-fill check. Changed: `from .llm import`
becomes an absolute import, and `CASES_PATH` resolves through the package
rather than `Path(__file__).parent`. Live tests stay behind `RUN_LLM_TESTS=1`,
so CI is unaffected.

**`tests/test_verifier.py`** — new, fakes only, no network:

- code-fence stripping, on ```` ```json ```` and bare ``` ```` ``` ````
- enum validation rejects `"supports"`, `"Support"`, a missing slot, an extra slot
- key inversion, including a duplicate source mapped to one key
- orphaned-key detection
- `\cite{...}` stripping, including several keys in one sentence and
  `\cite{a,b}`
- one sentence citing two sources produces two judgements
- each error-table row in §7 produces its category and does not abort

The existing suite must stay green. The `chat()` signature change is additive
and `tests/test_llm.py` should pass untouched — if it does not, the change is
wrong.

## 9. Out of scope

- **Judgement as a gate.** The audit reports; it does not block or rewrite.
- **Acting on `evidence_sufficiency`.** It is reported. Wiring it to trigger
  re-retrieval — widen `top_k`, drop the doc gate — is the obvious follow-up
  and deliberately not bundled here.
- **Generalising the prompt's examples.** All six worked examples are one
  MnFe₂O₄@PANI paper, in a tool that bills itself as physics-wide. Changing the
  prompt and the regression cases in the same pass as the integration would
  make any failure ambiguous. Separate change, after this one is green.
- **Agent 4.** Single-sentence citation is interactive and already shows its
  reasoning. The audit targets whole drafts.
- **GCP deployment.** Separate spec, three open decisions.
- **Windows support.** Unrelated, previously assessed, explicitly deferred.

## 10. Risks

**The prompt is a third-party artefact this repo now owns.** 259 lines of
carefully balanced rubric — the D-versus-E contrast, the first-match-wins
aggregation order — that no test covers directly. The six regression cases are
the only guard, and they run only with `RUN_LLM_TESTS=1`. Mitigation: keep the
cases green in every live run before shipping a prompt edit; treat `prompt.md`
as code, not content.

**Re-retrieved evidence may differ from what Agent 5 cited.** §4.2 argues this
is acceptable and safe in direction, but a verdict of `Supports` on a chunk the
citer never saw is a slightly different claim than "the citation was correct".
The report states which chunk was judged so the reader can see it.

**Cost is user-visible and per-citation.** A 120-citation draft is ~120 calls
with a ~3.8k-token prompt each. On a hosted API that is real money; the button
should say so before running. Mitigation: the summary logs an estimated call
count before starting.

**Silent degradation on a small local context.** §5.3's `num_ctx` handles the
default case, but a user pointing `CITATION_LLM_MODEL` at a genuinely
small-context model still gets truncation without an error. Mitigation:
`agent8_verifier` logs a visible warning at startup when `LLM_BACKEND=ollama`,
naming the prompt's token count and the configured `num_ctx`.
