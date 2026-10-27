# Marvin the Citebot 🤖

*Autonomous Academic Research Assistant*

**He cites your draft for you. He also judges you for not reading the papers.**

Hand him a half-written manuscript and a vague research idea. He'll find the
seed paper, raid its references, read every open-access PDF you were
"definitely going to get to", and put a `\cite{}` in every sentence that
needs one.

Then he checks his own work. For every citation, he confirms the paper
actually says what you claimed. If it doesn't, he tells you. He will not lie
for you. He has standards, even if you've abandoned yours.

Marvin blames AI for all of this. Humans used to read papers, he says. Now
they ask a chatbot, get a confident answer, and cite a paper that doesn't
exist. Every `\cite{}` Marvin writes points at a PDF he has actually read;
there is no other kind in his index.

The irony of being an AI himself is not lost on him. It's a big part of why
he's so miserable.

He didn't start out as a citation tool. He started because I wanted something
to brainstorm with — smart enough to be worth arguing with, and grounded
enough that it could only argue from papers it had actually read. That's still
the heart of him: hand him an idea and he tells you what's already been done
on it, then argues about the rest in the chat tab. The citing came later, once
he'd read everything anyway and had opinions about your draft.

He has a side job too. Hand him a paper and he reads its method as a recipe:
a dependency graph of replication tasks — environment, data, model, training,
evaluation, audit — each assigned to a specialised coding-agent role. He
considers this beneath him, and does it anyway. See the **Agent task planner**
tab below.

And a newer one, which he trusts least of all: give him a problem and he goes
looking in other fields for a technique that already solved something of the
same shape. See [Borrowing from other fields](#borrowing-from-other-fields).

## A note on where he lives

Marvin lives on your machine and nowhere else. `deploy/local.sh up` starts him on
`127.0.0.1` with GROBID for company, and nothing outside the room can knock. There
is no hosted instance and no firewall to ask about; if `APP_PASSWORD` is set he will
still ask for an access code at the door. How `local.sh` works is in
[deploy/README.md](deploy/README.md).

With local models he can be miserable for free, forever. See
[Running him at home](#running-him-at-home).

> Marvin introduces himself consistently across the application, user guide, and pipeline.

## What he does

Six tabs. The sixth is the instructions, which nobody reads, which he has
noticed.

| Tab | You give him | He gives you |
|-----|--------------|--------------|
| **Agent task planner** | a paper — a PDF from your library, an upload, GROBID TEI XML, or plain-text methods notes | a dependency graph of replication tasks, grouped into parallel waves and assigned to six agent roles; a Mermaid flowchart, a Markdown plan and Antigravity agent specs to download; and a replication scorecard if you run it (a dry run, or autonomously, with the model writing each task's deliverables) |
| **Citation auditor** | a published paper — one PDF, several, or a ZIP | he parses its in-text citations with GROBID, fetches the open-access references, and checks each citation against the cited paper's text: a verdict, the passage behind it, and an upload card for every paywalled reference he couldn't reach. Optionally a related-work synthesis across them |
| **Research idea** | a research idea, and optionally a seed-paper link (arXiv or a direct PDF URL) | a corpus built from the literature, and a synthesis of what's already been done on your idea |
| **Cite a draft** | a `.txt` draft, uploaded or pasted | the cited draft, a key → source map for BibTeX, a sentence-by-sentence report, and a verdict on every citation: *supports*, *partially*, *contradicts*, *does not support*, or *unclear* |
| **Research chat** | questions | streamed answers grounded in the corpus, with the sources he used |
| **How to use** | nothing | [HOW_TO_USE.md](HOW_TO_USE.md), the step-by-step guide |

Cite a draft and Research chat need a corpus, so build one in Citation auditor
or Research idea first. Citing a single sentence is not a tab; it is Agent 4
on the command line (see [PIPELINE.md](PIPELINE.md)).

## How he works

Marvin is nine agents in a trench coat, joined by files on disk rather than
function calls, so when one of them dies halfway — and one of them will — the
run resumes where it stopped.

```
research idea ─┐
PDF / ZIP ─────┴─► Agent 0  find a seed paper           arXiv → OpenAlex → Semantic Scholar
                   Agent 1  raid its reference list     GROBID, regex fallback when it's down
                   Agent 2  fetch open-access PDFs      Unpaywall → Europe PMC → arXiv
                   Agent 3  read, chunk, embed, summarise   ChromaDB + BM25, one summary per paper
                            │
            ┌───────────────┼──────────────────────┐
            ▼               ▼                      ▼
        Agent 4         Agent 5 ──► Agent 8     Agent 7
        one sentence    whole draft, then       multi-turn chat
        → \cite{key}    audit every \cite       over the corpus

        Agent 6  ingests PDFs you drop in by hand — the paywalled ones Agent 2 couldn't reach
```

Agent 4 is the command-line way to cite one sentence. The app's **Citation
auditor** tab runs the same chain on a published paper and points Agent 8 at the
paper's *own* citations.

The **Agent task planner** is a second pipeline (`research_assistant/tasks/`,
[PIPELINE.md](PIPELINE.md) calls it Pipeline 2). It reads the paper's methods
(GROBID TEI, or PyMuPDF text for a bare PDF). If the paper is already in the
corpus it retrieves evidence for four facets — data, architecture,
hyperparameters, evaluation — from it (`TASK_ENABLE_AUTO_INGEST=1` ingests the
paper first), and constructs a paper-scoped reference subgraph to dereference
methodological citations.

A proposer–critic loop of at most three turns rewrites the task graph against a
**Two-Level Hierarchical Critic Engine**:
- **MacroCritic**: Verifies DAG acyclicity, dependency validity, topological wave
  parallelizability, and mandatory coverage across the six scientific phases
  (`environment_setup`, `data_preprocessing`, `model_implementation`,
  `sanity_smoke_test`, `training_or_simulation`, `metric_evaluation` / `verification_audit`).
  Also enforces single-owner deliverables (`macro_duplicate_output_*` blocking)
  and warns against over-fragmented graph bloat (>12 tasks).
- **MicroCritic**: Scrutinizes task instruction specificity (operational parameters,
  first-principles algorithms), enforces formal `ArtifactContract` schemas
  (shapes, dtypes, columns, keys), prevents intra-task duplicate outputs and
  deliverable bloat (>8 deliverables), warns against unpruned tensor bloat (>50,000 dimensions),
  and strictly verifies hardware CPU bounds (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`).
- **Quantitative Rubric Scorecards (0–100)**: Rates topology, contracts, and testability.

Whenever a paper is uploaded or asked to be deconstructed, the system creates a
dedicated root directory (`workspace_runs/<paper_id>/`) and automatically drops a
complete suite of inspectable Markdown artifacts:
- [`plan.md`](#): Complete DAG replication plan with wave sequencing and artifact contracts.
- [`dag.md`](#): Mermaid flowchart dependency diagram.
- [`scorecard.md`](#): Quantitative critic rubric scorecard and identified flaws.
- [`paper_summary.md`](#): Extracted abstract, theoretical framework, methodology sections, and tables.
- [`rationale.md`](#): Consolidated scientific hypothesis and complexity bounds (adapting the AUTOMAT protocol).
- [`rationales/*.md`](#): Individual subagent task rationales documenting *Problem Knowledge*, *Scientific Insight*, and *Implementation Strategy*.
- [`README.md`](#): Workspace index and execution waves manifest.
- [`plan.json`](#) & [`antigravity_specs.json`](#): Machine-readable specifications for autonomous multi-agent teams.

When executed, a **Two-Cycle Reflexion Runner** runs tasks wave-by-wave, generates
deterministic acceptance test harnesses, parses structured failure diagnostics
(exit code, error type, failing assertions, missing files, tracebacks), and gives
failing tasks one reflection cycle to refine deliverables before compiling a
final `execution_scorecard.md`. Read the scorecard for what it is: the
rating (`CERTIFIED_REPLICABLE` at 99%+, `PARTIALLY_REPLICATED` from 70%) is the
share of tasks whose acceptance check passed. It does not say the paper's
numbers were reproduced, and a dry run, which fills in placeholder data, can
earn the top rating.

Retrieval is two-stage so he stays fast as the corpus grows: match the query
against the one-paragraph **paper summaries** first, let the model drop the
off-topic papers, then run the detailed hybrid (vector + keyword) chunk search
only over what survives.

Agent 8 is the one that keeps him honest. For every `\cite{}` Agent 5
inserted — or, in the Citation auditor, every in-text citation in the paper you
uploaded — it pulls the best-matching passage back out of that paper and puts
the claim through a rubric — finding, scope, strength — before deciding
whether the evidence supports it. As an auditor it reports; it doesn't gate. A
bad citation stays in your draft, flagged, with the passage that failed to back
it up.

The same judge also stands at Agent 5's door (`CITATION_CITER_JUDGE`, on by
default): a candidate source is only inserted if the judge would pass it, and
a sentence nothing passes is left uncited. That costs up to three judge calls
per sentence instead of one rewrite — cheap on a hosted API, slow on a CPU
model, and the run logs a warning once when the backend is Ollama.
`CITATION_CITER_JUDGE=0` restores the rewrite path.

Every agent, every file he writes and every knob is in
[PIPELINE.md](PIPELINE.md). The reasoning behind each design choice is in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Borrowing from other fields

Good ideas often arrive from somewhere else. A restricted Boltzmann machine, built
for machine learning, became a compact wavefunction for many-body physics. A
Chebyshev expansion became a way to get a density of states without diagonalising
anything. Marvin's newest job is to find the next one: give him a problem, and he
looks through the corpus for a technique another field uses on a problem with the
same shape. Then he proposes the transfer, argues against it, and scores it. It
runs from the command line, not a tab, in four stages (`research_assistant/transfer/`):

1. **Technique cards.** `scripts/build_cards.py` reads every paper in the corpus
   once and writes a card for each method in it that someone in another field
   could carry out. A card holds the method's own name, the problem it solves
   (worded so that no material, system or field is named), how it works in two
   to five steps, and one sentence from the paper as evidence, which the judge
   checks. Surveys, databases and benchmarks get no card, on purpose.
2. **Bottlenecks.** For your problem he grounds himself in the nearest papers and
   names what makes it hard. Each bottleneck is restated three ways (specific,
   mid, broad) with the field's nouns taken out. A problem that infers something
   from measurements gets a fourth restatement, saying what is inferred from what.
3. **Transfers.** Each restatement searches the cards, and the best few per
   bottleneck become proposals:
   - a role-by-role mapping from the source problem to yours;
   - an argument about whether your measurements can determine the unknowns;
   - concrete steps, a baseline, a prediction, and the result that would refute it.

   A critic then tries to reject each proposal, and the writer must answer each
   objection or concede it.
4. **Scoring.** The reward is checked in code, not by asking the model how it
   feels. First come gates that set a proposal aside:
   - evidence that contradicts it;
   - a core assumption that breaks and is conceded;
   - unknowns the measurements cannot determine;
   - a metric that misses the problem;
   - prior art that already did it.

   Then a ten-item depth checklist, and a bonus when OpenAlex and arXiv turn up
   no prior art. Rebutted objections and mere tuning of the current method cost
   reward.

```bash
deploy/local.sh chroma                                # the store's server, if the app isn't running
python scripts/build_cards.py                         # once; resumes where it stopped
python scripts/transfer.py -q "your problem" --show-cards   # bottlenecks and candidate cards only
python scripts/transfer.py -q "your problem"                # the full run
```

- **Where the report goes:** the full run writes a Markdown report and a JSON
  record to `data/transfer/runs/`.
- **Options:**
  - `--exclude` keeps named documents out of the run.
  - `--before-year` counts only prior art published before that year.
- **When it won't run:** while the app has a pipeline running, it refuses to start.

**Treat what he proposes as a lead, not a finding.** Each premise is checked
against its paper. The mapping, the steps and the prediction are his own, and the
report labels them that way.

Whether he finds the right technique is being measured by retrodiction. He is
given problems whose answer is known (neural-network quantum states, the kernel
polynomial method), plus held-out cases kept outside the repository, and checked
for whether he finds it. `scripts/evaluate_transfer.py --retrieval-only` checks
that the right card is retrieved; `--score --runs 3` runs the whole thing.
Results will follow when there are some. He has not been told this is a promise.

## What he's worst at

Depth. Ask him what's been done on an idea and he'll tell you — grouped by
theme, every claim cited — but he gives you the headline of each paper, not
its argument. He is shallow in three specific ways, all of them visible in
the code:

- **He reads one hop.** The corpus is the seed paper's reference list and
  nothing beyond it. He doesn't follow the references' references, so a
  field's foundational work is in there only if the seed paper happened to
  cite it.
- **He reads through a keyhole.** The default model is `gemma4:e2b` — two
  billion parameters on a CPU, with a 4,096-token window for chat and task
  planning (the judge gets 12,288). Every summary is one paragraph. Every
  verdict Agent 8 hands down starts from a single retrieved passage and its
  neighbours (`CITATION_JUDGEMENT_TOP_K=1`), and is re-judged on the top three
  only when the first read admits it saw too little. He is fast and cheap, and
  he knows exactly what that costs. Set `GEMINI_API_KEY` and chat, synthesis
  and judging move to a hosted model; the keyhole widens, the bill does too.
- **He used to answer short by design.** The synthesis now reads every paper
  it shortlists and is asked for the argument — what's established, where the
  papers disagree, the gap — rather than a headline list. Whether the gap it
  names is the right one is still yours to judge.


The first of those is being fixed: the current index (v3) is built from GROBID full text with proper chunks and captions, and embeds each figure as an image so a text query can find it — see `PIPELINE.md`. Older indexes stay on disk, read-only.

We will give him depth. Those three are where it goes. Until then he is a very
well-read undergraduate: he has read everything and understood some of it. He
would like that on the record.

## Physics, and everyone else

Physics is his domain. arXiv is searched first, the prompts introduce him as
a physicist, and his collections are literally called `physics_papers`.
Nothing in the machinery cares, though. For another field, change the two
system prompts in [`research_assistant/prompts.py`](research_assistant/prompts.py)
that call him a physicist, and reorder `CITATION_SEARCH_PROVIDERS` so your
field's index is tried before arXiv. He'll read anything with a reference
list. He'll complain about it either way.

## Running him at home

Free, forever, miserable. Five commands.

```bash
git clone https://github.com/mukims/research_assistant.git && cd research_assistant
pip install -e . && pip install -r requirements.txt          # Python 3.10–3.12
ollama pull gemma4:e2b && ollama pull embeddinggemma-2       # the default local models
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
streamlit run app.py
```

- **One command instead of the last two.** `deploy/local.sh up` starts GROBID if it is not
  answering, then the app, listening on `127.0.0.1` only (a bare `streamlit run app.py`
  listens on every network interface). `status` and `down` do what they say.
  See [deploy/README.md](deploy/README.md).
- **The vector store has one owner: a Chroma server** on `localhost:8000`, which
  `deploy/local.sh up` starts. The app and every script connect to it; none opens the files.
  ChromaDB's local store is not safe across processes — two with it open lost 17,201
  vectors on 2026-10-21. To run a script without the app, start the server alone with
  `deploy/local.sh chroma`. `CITATION_CHROMA_SERVER=` (empty) opens the files directly, for
  a store nothing else has open.
- **Models** run locally through [Ollama](https://ollama.com) by default. Set
  `GEMINI_API_KEY` to use Gemini (`gemini-3.5-flash-lite`) for chat, synthesis
  and judging, or `LLM_BACKEND=openai` (with `OPENAI_BASE_URL` /
  `OPENAI_API_KEY`) for any OpenAI-compatible endpoint. Embeddings stay on
  Ollama (`embeddinggemma-2`, which embeds text and figure images) with Gemini.
- **Access code.** Set `APP_PASSWORD` and the app asks for it before anything
  else. Unset, it is open.
- **GROBID** is how Agent 1 reads reference lists. Without it Marvin still
  runs, but falls back to regex extraction — no DOIs, no authors, and
  noticeably fewer downloads. Setup and troubleshooting:
  [Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md).
- **Unpaywall** requires a contact address on every request, and Crossref uses
  the same one to put you in its faster "polite" pool. There is a placeholder
  default; set your own: `export UNPAYWALL_EMAIL=you@example.com`.
- **OpenAlex** now gives requests without an API key a small daily budget, shared
  by everyone on your network's address (about a hundred searches). Once it runs
  out:
  - paper searches carry on with the other providers;
  - technique-transfer prior-art checks come back `unchecked` until the budget resets.

Without the UI, one research idea end to end:

```bash
python orchestrate.py --query "topological protection in disordered quantum wires" --ask
```

`--seed-url` or `--seed-file` seeds from a paper you already have instead of
searching, and `--audit-citations` also audits that paper's own citations.

The task planner has its own command line: `plan` turns a paper (PDF, TEI XML
or text) into a task graph, `show` renders it, `run` executes it.

```bash
python orchestrate_tasks.py plan paper.pdf --format json -o plan.json     # add --heuristic to skip the model
python orchestrate_tasks.py show plan.json --format mermaid
python orchestrate_tasks.py run plan.json --mode simulation --report-out report.md
```

`--mode simulation` only validates the graph and writes placeholder
deliverables. `--mode autonomous` has the model write each task's deliverables
into the workspace, runs a generated acceptance test over them in a subprocess,
and gives a failing task one reflection retry. Give it a scratch `--workspace`;
the runner refuses to point at the persistent data directory.

## Where things are

| Path | What it is |
|------|------------|
| `app.py` | the Streamlit page: sidebar and six tabs. Each tab's code is in `research_assistant/ui/` |
| `orchestrate.py` | one research idea end to end, as a LangGraph |
| `orchestrate_tasks.py` | command line for the Agent task planner: `plan`, `show`, `run` |
| `watch.py` | directory watcher: drop a PDF or a draft and the matching stage runs |
| `research_assistant/agents/` | Agents 0–8, one file each (the pipeline above) |
| `research_assistant/shared/` | what the agents share: model access, ingestion, retrieval, the citation audit |
| `research_assistant/judgement/` | the claim–evidence judge, source assessor, reliability policy, and the judge's evaluation set |
| `research_assistant/eval/` | evaluation of the citer, scored against a paper's own citations |
| `research_assistant/tasks/` | the Agent task planner (Pipeline 2): `models`, `extractor`, `critic`, `iteration`, `retrieval`, `harness`, `roles`, `runner`, `workspace`, `ui` |
| `research_assistant/transfer/` | borrowing from other fields: technique cards, bottlenecks, transfers, the critic, scoring and reports |
| `scripts/` | operator tools: judge and citer evaluation, labelling, index statistics, and the technique-transfer commands (`build_cards.py`, `transfer.py`, `evaluate_transfer.py`, `build_case_corpus.py`, `review_cards.py`) |
| `notebooks/` | `pipeline.ipynb`, a test bench that runs Agents 0–8 on one paper |
| `deploy/` | `local.sh`, which starts GROBID and the app on this machine only |
| `tests/` | the test suite |
| `workspace_runs/` | dedicated root workspaces for deconstructed and executed papers with all Markdown bundles |
| `data/` | everything the pipeline writes: indexes, PDFs, logs. Gitignored |

The full tree, down to individual files, is in [PIPELINE.md](PIPELINE.md#layout).

## Things he has written down

| Read this | When |
|-----------|------|
| [HOW_TO_USE.md](HOW_TO_USE.md) | you want to drive the app — every tab, every button, what each failure message means |
| [PIPELINE.md](PIPELINE.md) | you want the full technical reference: agents, on-disk state, every environment variable, the repo layout |
| [ARCHITECTURE.md](ARCHITECTURE.md) | you want to know *why* — each design decision, what was rejected, the trade-off accepted |
| [Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md) | GROBID won't start, or you've never installed Docker |
| [deploy/README.md](deploy/README.md) | you want to run him on this machine: `local.sh`, GROBID, low memory |

## Development

He has tests. They pass. He finds this suspicious.

```bash
pip install -e .
pip install -r requirements-test.txt
CITATION_LOG_FILE=0 python -m pytest tests/ -v
```

The suite covers the pure logic — reference parsing, source naming, extractor
filing, ingestion bookkeeping, retrieval ranking, the claim–evidence rubric,
task-graph construction and execution, and technique transfer (against a small
fake world of papers and cards). It needs none of the heavy stack: no ChromaDB,
no PyTorch, no model backend, no GROBID. Each run gets its own throwaway data
directory, so a test can never open the real corpus. CI runs it on Python 3.10 and 3.12 and import-sweeps every module
in the package to catch a broken import no test happens to cover.
