# Marvin the Citebot 🤖

*Autonomous Academic Research Assistant*

[![tests](https://github.com/mukims/research_assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/mukims/research_assistant/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

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

## A note on his health

Marvin is alive at **https://marvin-the-citebot.duckdns.org** for as long as
Google Cloud and my wallet allow. Neither is infinite.

He lives on a single Compute Engine VM with GROBID for company and his corpus
on a persistent disk. The firewall admits allow-listed IPs only, so ask before
you knock; if `APP_PASSWORD` is set he will also ask for an access code at the
door. By default he runs `gemma4:e2b` and `nomic-embed-text` through Ollama on
CPU, and `deploy/auto_shutdown_idle.sh` switches him off after 45 minutes of
nobody talking to him, which he considers the best part of his day. How the VM
is built, updated and stopped is in [deploy/README.md](deploy/README.md).

He's happiest running on your own machine with local models, where he can be
miserable for free, forever. See [Running him at home](#running-him-at-home).

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
paper first). Then a proposer–critic loop of at most three turns rewrites the
task graph until the critic finds no blocking flaw, and any flaw still left is
repaired deterministically. If the model fails or the graph won't validate,
or you pass `--heuristic`, it falls back to a fixed five-task graph
(`--full-phases` gives all six phases). The result is checked as a DAG (no
cycles, no dangling dependencies), scheduled into topological waves, and run
either as a dry run or autonomously. Read the scorecard for what it is: the
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


The first of those is being fixed: `CITATION_INDEX_VERSION=2` builds a second index from GROBID full text with proper chunks and captions, beside the old one — see `PIPELINE.md`.

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
ollama pull gemma4:e2b && ollama pull nomic-embed-text       # the default local models
docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1
streamlit run app.py
```

- **Models** run locally through [Ollama](https://ollama.com) by default. Set
  `GEMINI_API_KEY` to use Gemini (`gemini-3.5-flash-lite`) for chat, synthesis
  and judging, or `LLM_BACKEND=openai` (with `OPENAI_BASE_URL` /
  `OPENAI_API_KEY`) for any OpenAI-compatible endpoint. Embeddings stay on
  Ollama (`nomic-embed-text`) with Gemini, so an existing index keeps working.
- **Access code.** Set `APP_PASSWORD` and the app asks for it before anything
  else. Unset, it is open.
- **GROBID** is how Agent 1 reads reference lists. Without it Marvin still
  runs, but falls back to regex extraction — no DOIs, no authors, and
  noticeably fewer downloads. Setup and troubleshooting:
  [Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md).
- **Unpaywall** requires a contact address on every request, and OpenAlex and
  Crossref use the same one to put you in their faster "polite" pools. There
  is a placeholder default; set your own: `export UNPAYWALL_EMAIL=you@example.com`.

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

### Docker

Two images, for two situations:

- [`Dockerfile`](Dockerfile) + [`docker-compose.yml`](docker-compose.yml) — the
  app pointed at OpenAI (`gpt-4.1-mini`, `text-embedding-3-small`) with GROBID
  as a second service. This is what `deploy/deploy.sh` builds and pushes to
  Artifact Registry. Needs `OPENAI_API_KEY` in the environment:
  `docker compose up`.
- [`Dockerfile.standalone`](Dockerfile.standalone) — everything in one image:
  Ollama with `gemma4:e2b` and `nomic-embed-text` baked in, GROBID, and the
  app. **This is the one that serves the live instance**, built on the VM
  itself. It expects a pre-pulled `models/` directory beside it, which is not
  committed; `deploy/entrypoint.sh` is. [deploy/README.md](deploy/README.md)
  has the details.

## Things he has written down

| Read this | When |
|-----------|------|
| [HOW_TO_USE.md](HOW_TO_USE.md) | you want to drive the app — every tab, every button, what each failure message means |
| [PIPELINE.md](PIPELINE.md) | you want the full technical reference: agents, on-disk state, every environment variable, the repo layout |
| [ARCHITECTURE.md](ARCHITECTURE.md) | you want to know *why* — each design decision, what was rejected, the trade-off accepted |
| [Research_Assistant_GROBID_Guide.md](Research_Assistant_GROBID_Guide.md) | GROBID won't start, or you've never installed Docker |
| [deploy/README.md](deploy/README.md) | you're putting him on a VM, or paying for one |

## Development

He has tests. They pass. He finds this suspicious.

```bash
pip install -e .
pip install -r requirements-test.txt
CITATION_LOG_FILE=0 python -m pytest tests/ -v
```

The suite covers the pure logic — reference parsing, source naming, extractor
filing, ingestion bookkeeping, retrieval ranking, the claim–evidence rubric,
task-graph construction and execution — and needs none of the heavy stack: no ChromaDB, no PyTorch, no model backend,
no GROBID. CI runs it on Python 3.10 and 3.12 and import-sweeps every module
in the package to catch a broken import no test happens to cover.

## License

[MIT](LICENSE). Marvin would like it noted that this doesn't make him happy
either.
