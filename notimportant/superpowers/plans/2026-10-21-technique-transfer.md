# Technique Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** an unattended search that, for a research problem, finds techniques from other fields that solve the same shape of problem, writes concrete, judge-checked transfers, and ranks them by a reward that is calibrated on known historical transfers before it is trusted.

**Architecture:** a new package, `research_assistant/transfer/`, works in five stages:

1. It builds one **technique card** per paper. A card records what the paper *did*, not what it found.
2. For a query, it grounds **bottlenecks** in the nearest papers.
3. It searches the cards with each bottleneck's generic forms.
4. For each pair it writes a transfer, has every premise judged and the transfer attacked by a critic, then checks prior art.
5. It scores each transfer with code-checked gates and a depth checklist.

Every external call comes in through a `Deps` object, so all tests run on fakes. Search starts as best-of-N; MCTS is a later phase, built only if a retrodiction eval shows it helps.

**Tech Stack:**
- Python 3.10–3.12 and numpy
- ChromaDB 1.5.9, read only
- embeddinggemma-2 through Ollama
- `shared.llm` (Ollama or Gemini) and `judgement.judge`
- the OpenAlex and arXiv APIs
- Streamlit 1.65 (`run_jobs`, AppTest)
- unittest, run under pytest

**Spec:** `notimportant/superpowers/specs/2026-10-21-technique-transfer-design.md`

## Global Constraints

- **Python versions.** Python ≥ 3.10 and < 3.13. CI runs 3.10 and 3.12, and `tests/test_python_compat.py` rejects 3.12-only f-string forms.
- **No new dependencies.** numpy and requests are already pinned.
- **No heavy imports at module level.** No new module imports chromadb, torch or langchain at module level, because CI installs only `requirements-test.txt`.
- **Where files go.**
  - Every write lands under `config.DATA_DIR` (`TRANSFER_DIR`, `data/eval/transfer/`), via `shared.atomic.atomic_write_json` or `atomic_write`.
  - Nothing writes to Chroma.
  - Case corpora are built only under `data/eval/transfer/corpora/<case>/`.
- **Figures are never evidence.** Every `hybrid_search` call passes `exclude_types=EXCLUDED_FROM_EVIDENCE`. Card building also skips `caption` chunks.
- **Premise credit uses the citer's rule:**

  | Verdict | `span_verified` | Credit |
  |---|---|---|
  | Supports | `True` | 1.0 |
  | Partially supports | `True` | 0.5 |
  | anything else | any | 0.0 |

  A failure never earns credit, and Contradicts sets the transfer aside.
- **Code that stays untouched.** Agents 0, 5 and 8, `judgement/` and `shared/ingestion.py` are not modified; they are called.
- **Configuration.** New tunables go in `config.py` under `# ─── Technique transfer ───`, after the Synthesis section. Each has a `CITATION_TRANSFER_*` override and a one-line reason, written with `_env_int` or `float(os.environ.get(...))`.
- **Prompts.** Every user prompt template begins with the line `TASK: <kind>`, where kind is one of `cards`, `bottlenecks`, `transfer`, `critique` or `revise`. The test fakes route on that line.
- **Tests.** Tests are `unittest.TestCase` classes with sentence-named tests. Fixtures live in `tests/transfer_fixtures.py`.
- **Staging.** Stage explicit paths only; never `git add -A` or `git add .`. Never push.
- **Live runs.** Any command that calls a live model, or reads Chroma for a batch job, runs only when `pipeline_status` shows no active run. Builder and eval scripts enforce this with `ensure_idle()`.
- **Private material.** Anything describing the maintainer's unpublished work goes under `data/eval/transfer/answer_keys/`, which is gitignored, and never into a tracked file.
- **Python on the maintainer's machine.** `/home/shardul/miniconda3/envs/ml/bin/python`. The commands below write `python`.

## Review Focus

1. **A transfer named from model memory.** The model can name RBMs or KPM without the card being retrieved. Retrodiction scoring must read the trace (card → document), so such a transfer scores zero. Pinned in Task 14.
2. **A case corpus built into the main data directory.** That would pollute the real corpus. The builder must refuse unless `config.DATA_DIR` is that case's directory. Pinned in Task 13.
3. **A premise citing a passage key that wasn't in its own prompt.** It is dropped with a warning and earns nothing. Pinned in Tasks 2 and 11.
4. **Prior art leaking the answer.** Prior art must respect the case's cutoff year, or every retrodiction is "done before". Pinned in Task 9.
5. **A negative-control transfer that looks good on the surface.** An autoencoder from conductance to impurity positions must be set aside as `not_identifiable` when the critic or writer finds it underdetermined. Pinned in Tasks 10 and 14.

---

## Before Task 0 (maintainer)

- Antigravity stays paused on `feat/hypothesis-search`, after that plan's Task 0. That plan is superseded; its branch is the base for this one.
- The v3 migration (`scripts/migrate_to_v3.py`) must finish before any **live** step: Task 4 Step 6, Task 5 Step 5, and the live steps of Tasks 12, 13, 14 and 15. The unit-test tasks can proceed while it runs.
- Write the two private answer keys (Task 13 gives the template) when you reach Task 13. Until then, nothing private is needed.

## Phase and gate map

| Phase | Tasks | Ends with |
|---|---|---|
| Housekeeping | 0 | branch `feat/technique-transfer` |
| A — Foundations | 1–3 | models, reply parsing, premise checking |
| B — Technique cards | 4–5 | **Gate F1**: you rate 20 cards |
| C — Pipeline (best-of-N) | 6–12 | a working CLI |
| D — Retrodiction | 13–14 | **Gates F2 and F3**, then calibration |
| E — MCTS | 15 | **Gate**: beats best-of-N at equal budget, or it isn't shipped |
| F — UI and docs | 16–17 | the tab, docs and CI |

At a gate, the executor stops and reports. You run the live step and decide whether to go on.

## File map

| File | Responsibility | Task |
|---|---|---|
| `research_assistant/transfer/__init__.py` | package docstring | 1 |
| `research_assistant/transfer/models.py` | dataclasses, ids, JSON round trip | 1 |
| `research_assistant/transfer/replies.py` | JSON replies with one retry; premise key resolution | 2 |
| `research_assistant/transfer/premises.py` | premises judged with the citer's rule, cached | 3 |
| `research_assistant/transfer/prompts.py` | all templates, keyed passage formatting | 4 (cards); 6 and 8 add theirs |
| `research_assistant/transfer/cards.py` | card passage selection, building, store, index | 4 |
| `research_assistant/transfer/review.py` | F1 review sheet and scoring | 5 |
| `research_assistant/transfer/deps.py` | `Deps`, `TransferSettings` | 6 |
| `research_assistant/transfer/stages.py` | `ground`, `candidate_cards`, `develop` | 6, 7, 8 |
| `research_assistant/transfer/prior_art.py` | OpenAlex and arXiv client, with abstracts and a cutoff | 9 |
| `research_assistant/transfer/scoring.py` | gates, depth checklist, reward, ranking | 10 |
| `research_assistant/transfer/engine.py` | guards, `live_deps`, `run_best_of_n` | 11 |
| `research_assistant/transfer/render.py`, `store.py` | Markdown, Mermaid, save and load | 12 |
| `research_assistant/transfer/mcts.py` | Phase 2 budget allocation | 15 |
| `research_assistant/eval/transfer_cases.py` | case schema; public and private loading | 13 |
| `research_assistant/eval/cases/transfer_retro.jsonl` | NQS and KPM cases (public) | 13 |
| `research_assistant/eval/transfer_metrics.py` | pure retrodiction scoring | 14 |
| `scripts/build_cards.py`, `scripts/review_cards.py` | build cards; score F1 | 4, 5 |
| `scripts/transfer.py` | CLI, including `--show-cards` | 12 |
| `scripts/build_case_corpus.py` | resolve, download and ingest a case corpus | 13 |
| `scripts/evaluate_transfer.py` | F2, F3 and the Phase 2 comparison | 14, 15 |
| `research_assistant/shared/pipeline_status.py` | stage `transfer` | 16 |
| `research_assistant/ui/jobs.py`, `ui/common.py`, `tasks/ui.py`, `ui/transfer_tab.py`, `app.py` | the tab | 16 |
| `README.md`, `HOW_TO_USE.md`, `ARCHITECTURE.md`, `.github/workflows/tests.yml` | docs, CI sweep | 17 |
| `research_assistant/config.py` | `TRANSFER_*` settings | 1, 4, 6, 9 |

---

### Task 0: Branch, and retire the superseded plan

**Files:**
- Modify: the first line under the title of each superseded document, adding `> **Superseded** by \`2026-10-21-technique-transfer-design.md\` / \`2026-10-21-technique-transfer.md\`.`:
  - `notimportant/superpowers/specs/2026-10-21-hypothesis-search-design.md`
  - `notimportant/superpowers/plans/2026-10-21-hypothesis-search.md`
  - `notimportant/superpowers/plans/2026-10-21-hypothesis-search-antigravity.md`

**Interfaces:** produces branch `feat/technique-transfer`, created from `feat/hypothesis-search`.

- [ ] **Step 1: Check the starting state**

Run: `git branch --show-current && git status --short`
Expected: `feat/hypothesis-search`, and no output from `git status`.

- [ ] **Step 2: Create the branch**

Run: `git switch -c feat/technique-transfer`

- [ ] **Step 3: Mark the three documents superseded and commit**

```bash
git add notimportant/superpowers/specs/2026-10-21-hypothesis-search-design.md \
        notimportant/superpowers/plans/2026-10-21-hypothesis-search.md \
        notimportant/superpowers/plans/2026-10-21-hypothesis-search-antigravity.md
git commit -m "docs: mark the hypothesis-search plan superseded by technique transfer"
```

- [ ] **Step 4: Verify**

Run: `python -m pytest tests/test_app_smoke.py -q`
Expected: PASS.

---

### Task 1: Package, configuration and data model

**Files:**
- Create: `research_assistant/transfer/__init__.py`, `research_assistant/transfer/models.py`, `tests/transfer_fixtures.py`
- Modify: `research_assistant/config.py`
- Test: `tests/test_transfer_models.py`

**Interfaces.** Every type below is a `@dataclass`.

| Type | Fields | Notes |
|---|---|---|
| `Passage` | `key: str, chunk_index: int, document: str, citation: str, section: str, text: str, role: str` | `key` is `"C<n>"` within one prompt family. `role` is `"near"`, `"source"` or `"card"` |
| `Premise` | `claim: str, key: str, document: str = "", chunk_index: int \| None = None, judgement: str \| None = None, span_verified: bool \| None = None, supporting_span: str \| None = None, reason: str \| None = None, error: str \| None = None` | properties `accepted`, `partial`, `credit` and `contradicts`, per the citer's rule |
| `Bottleneck` | `id: str, statement: str, quantities: list[str], current_method: str, generic: dict[str, str], premises: list[Premise]` | `generic` has the keys `"specific"`, `"mid"` and `"broad"` |
| `Card` | `card_id: str, document: str, citation: str, name: str, problem_solved: str, context: str, inputs: str, outputs: str, assumptions: list[str], evidence: Premise` | `card_id` is `f"{document}#{i}"`. Property `verified`: `evidence.accepted or evidence.partial` |
| `MappingRow` | `source_role: str, target_role: str, assumption: str, status: str, core: bool, note: str = ""` | `status` is `holds`, `breaks`, `enforce` or `unknown` |
| `Identifiability` | `measurements: str = "", unknowns: str = "", symmetries: list[str] = [], verdict: str = "unclear", argument: str = ""` | `verdict` is `determined`, `underdetermined` or `unclear` |
| `Step` | `action: str, input: str, output: str, tool: str` | |
| `Critique` | `shape_match: str = "unknown", rows: list[dict] = [], identifiability: str \| None = None, objections: list[dict] = []` | `rows` entries are `{"row": int, "status", "reason"}`. `objections` entries are `{"text", "answered": bool, "response": str, "conceded": bool}` |
| `PriorArt` | `status: str = "unchecked", hits: list[dict] = []` | `hits` entries are `{"title", "year", "url", "similarity"}` |
| `Scores` | `depth: float, checks: dict[str, bool], reward: float, set_aside: str \| None = None` | |
| `Transfer` | `id: str, bottleneck_id: str, card_id: str, card_document: str, level: str, retrieval_rank: int, title: str, summary: str, mapping: list[MappingRow], identifiability: Identifiability, steps: list[Step], data_requirement: str, baseline: str, metric: str, prediction: str, refuting_result: str, premises: list[Premise], critique: Critique, prior_art: PriorArt, scores: Scores \| None = None, warnings: list[str] = []` | the `id` is `f"{bottleneck_id}.T{n}"` |
| `RunResult` | `run_id: str, query: str, search: str, created_at: str, models: dict[str, str], settings: dict, near: list[dict], passages: dict[str, Passage], bottlenecks: list[Bottleneck], cards: dict[str, Card], transfers: list[Transfer], calls: dict[str, int], seconds: float, warnings: list[str]` | `search` is `"best_of_n"` or `"mcts"`. `models` has the keys `generator`, `judge` and `critic`. `calls` has the keys `llm`, `critic`, `judge`, `embed` and `prior_art`. `cards` holds only the cards that were used |

List defaults use `field(default_factory=list)`, and the empty critique and prior-art defaults use `field(default_factory=…)`.

Functions:
- `new_run_id(query) -> str`, of the form `%Y%m%d-%H%M%S-<sha1[:8]>`.
- `result_to_dict(r) -> dict` and `result_from_dict(d) -> RunResult`; nested dataclasses are rebuilt explicitly.
- `card_to_dict(c) -> dict` and `card_from_dict(d) -> Card`.

Config (Task 1 adds these; later tasks add theirs):
- `TRANSFER_DIR = os.path.join(DATA_DIR, "transfer")`
- `TRANSFER_BOTTLENECKS = _env_int("CITATION_TRANSFER_BOTTLENECKS", 3)`
- `TRANSFER_CARDS_PER_QUERY = 3`
- `TRANSFER_TECHNIQUES_PER_BOTTLENECK = 3`
- `TRANSFER_MIN_NEAR_PAPERS = 3`
- `TRANSFER_NEAR_PASSAGES = 8`
- `TRANSFER_SOURCE_PASSAGES = 3`
- `TRANSFER_PASSAGE_MAX_CHARS = 7000`
- `TRANSFER_TEMPERATURE = 0.7`
- `TRANSFER_BOTTLENECK_TEMPERATURE = 0.3`
- `TRANSFER_CRITIC_MODEL = os.environ.get("CITATION_TRANSFER_CRITIC_MODEL") or None`
- `TRANSFER_OLLAMA_OPTIONS = {"num_ctx": _env_int("CITATION_TRANSFER_NUM_CTX", 8192)}`

Each integer default above gets the same `CITATION_TRANSFER_<NAME>` override.

Fixtures, `tests/transfer_fixtures.py`:
- `premise(judgement="Supports", key="C1", document="a.pdf", chunk_index=1, claim=None, span=True) -> Premise`
- `good_transfer() -> (Transfer, Bottleneck, Card)`, a triple that passes every gate and all ten depth checks:
  - **Bottleneck:** `quantities=["conductance"]`, `current_method="recursive Green's function"`, one accepted premise from `near.pdf`.
  - **Card:** document `src.pdf`, with a verified evidence premise.
  - **Transfer:**
    - mapping: 3 rows, 1 core, all `holds`;
    - identifiability: `determined`, with a 12-word argument;
    - steps: 3 complete ones;
    - `data_requirement="about 10000 RGF samples"`, `baseline="plain recursive Green's function search"`, `metric="conductance error per 1000 RGF calls"`;
    - `prediction="the conductance error will decrease twofold at equal cost"`, with a different 6-word `refuting_result`;
    - premises: accepted, one on `src.pdf` and one on `near.pdf`;
    - critique: `shape_match="yes"` and one answered objection;
    - prior art: `PriorArt(status="not_found")`.
- `sample_result() -> RunResult`, deterministic, with run id `20261021-120000-abcdef12`. It holds two transfers:
  - **B1.T1**, built from `good_transfer()`, with reward 1.25.
  - **B1.T2**, set aside as `not_identifiable`.

- [ ] **Step 1: Write the failing tests** (`tests/test_transfer_models.py`)

```python
class TestModels(unittest.TestCase):
    def test_premise_credit_follows_the_citers_rule(self):
        self.assertEqual(premise("Supports").credit, 1.0); self.assertEqual(premise("Partially supports").credit, 0.5)
        for j, s in [("Supports", None), ("Supports", False), ("Unclear / insufficient evidence", True), (None, None)]:
            self.assertEqual(premise(j, span=s).credit, 0.0, (j, s))
        self.assertTrue(premise("Contradicts").contradicts)

    def test_a_card_is_verified_only_by_an_accepted_or_partial_evidence_premise(self):
        _, _, card = good_transfer(); self.assertTrue(card.verified)
        card.evidence = premise("Unclear / insufficient evidence"); self.assertFalse(card.verified)

    def test_a_run_and_a_card_survive_json(self):
        r = sample_result()
        self.assertEqual(models.result_from_dict(json.loads(json.dumps(models.result_to_dict(r)))), r)
        _, _, c = good_transfer(); self.assertEqual(models.card_from_dict(json.loads(json.dumps(models.card_to_dict(c)))), c)

    def test_the_transfer_directory_is_under_data_dir(self):
        self.assertTrue(os.path.realpath(config.TRANSFER_DIR).startswith(os.path.realpath(config.DATA_DIR)))
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `python -m pytest tests/test_transfer_models.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

- The package docstring is `"""Technique transfer: methods from other fields for a problem's bottlenecks."""`.
- Write the models, the fixtures and the config lines.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_transfer_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/__init__.py research_assistant/transfer/models.py research_assistant/config.py \
        tests/transfer_fixtures.py tests/test_transfer_models.py
git commit -m "feat(transfer): data model for bottlenecks, technique cards and transfers"
```

---

### Task 2: Strict reply parsing and premise key resolution

**Files:**
- Create: `research_assistant/transfer/replies.py`
- Test: `tests/test_transfer_replies.py`

**Interfaces:**
- `JSON_NUDGE = "Your previous reply was not the JSON object asked for. Reply again with only that JSON object: no prose, no code fence."`
- `class ReplyParseError(ValueError)`, carrying `.raw`.
- `parse_reply(raw: str, required: tuple[str, ...]) -> dict`
  - Strips a ```` ``` ```` or ```` ```json ```` fence, then runs `json.loads`.
  - If that fails, tries the outermost `{…}`.
  - The result must be a dict holding every key in `required`.
- `ask_json(call: Callable[[list[dict]], str], messages: list[dict], required: tuple[str, ...]) -> dict`
  - After one `ReplyParseError`, it calls once more with `messages + [assistant: bad reply, user: JSON_NUDGE]`.
  - A second failure raises.
- `normalise_key(key) -> str` turns `"[c7]"` or `" C7 "` into `"C7"`.
- `resolve_premises(items: list, passages: dict[str, Passage], limit: int) -> tuple[list[Premise], list[str]]`
  - Items look like `{"claim", "key"}`.
  - Copies `document` and `chunk_index` from the cited passage.
  - Drops each of these with a warning: an unknown key (`"premise cites C99, which was not among the passages shown"`), an empty claim, a duplicate (claim, key), and anything beyond `limit`.

- [ ] **Step 1: Write the failing tests**

```python
P = {"C1": models.Passage("C1", 4, "a.pdf", "A", "methods", "t", "near"),
     "C2": models.Passage("C2", 9, "b.pdf", "B", "other", "t", "source")}

class TestReplies(unittest.TestCase):
    def test_fenced_bare_and_prose_wrapped_replies_parse(self):
        for raw in ['```json\n{"a": 1}\n```', '{"a": 1}', 'Sure: {"a": 1} done']:
            self.assertEqual(replies.parse_reply(raw, ("a",)), {"a": 1})

    def test_a_missing_required_key_is_rejected(self):
        with self.assertRaises(replies.ReplyParseError): replies.parse_reply('{"a": 1}', ("a", "b"))

    def test_one_retry_with_the_nudge_then_give_up(self):
        sent = []
        def call(m): sent.append(m); return "no json"
        with self.assertRaises(replies.ReplyParseError): replies.ask_json(call, [{"role": "user", "content": "q"}], ("a",))
        self.assertEqual(sent[1][-2:], [{"role": "assistant", "content": "no json"}, {"role": "user", "content": replies.JSON_NUDGE}])

    def test_a_premise_citing_a_passage_not_shown_is_dropped_with_a_warning(self):
        got, w = replies.resolve_premises([{"claim": "X.", "key": "C99"}], P, limit=4)
        self.assertEqual(got, []); self.assertIn("C99", w[0])

    def test_keys_are_normalised_and_carry_provenance_and_extras_are_dropped(self):
        items = [{"claim": "A.", "key": "[c2]"}, {"claim": "A.", "key": "C2"}, {"claim": "", "key": "C1"}, {"claim": "B.", "key": "C1"}]
        got, w = replies.resolve_premises(items, P, limit=1)
        self.assertEqual([(p.key, p.document, p.chunk_index) for p in got], [("C2", "b.pdf", 9)]); self.assertEqual(len(w), 3)
```

- [ ] **Step 2: Run and confirm they fail**

Run: `python -m pytest tests/test_transfer_replies.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `replies.py`**

- [ ] **Step 4: Run and confirm they pass**

Run the same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/replies.py tests/test_transfer_replies.py
git commit -m "feat(transfer): strict JSON replies with one retry; premises keyed to shown passages"
```

---

### Task 3: Premise checking with the citer's rule

**Files:**
- Create: `research_assistant/transfer/premises.py`
- Test: `tests/test_transfer_premises.py`

**Interfaces:**
- Consumes:
  - `agents.agent8_verifier.assemble_evidence(hits, max_chars)`
  - `shared.search.expand_neighbours(results, texts, metadatas, window)`
  - config `JUDGEMENT_NEIGHBOUR_WINDOW` and `JUDGEMENT_EVIDENCE_MAX_CHARS`
- Produces `PremiseChecker(judge: Callable[[str, str], dict], texts: list[str], metadatas: list[dict])`:
  - `.check(p) -> Premise` fills in `judgement`, `span_verified`, `supporting_span` and `reason`. If the judge raises, it sets `error` to `"<Type>: <msg>"` instead. It never raises itself.
  - `.check_all(ps) -> list[Premise]`.
  - `.calls: int` counts judge calls.
  - The cache key is `(" ".join(claim.lower().split()), chunk_index)`. Failures are cached too.
- How evidence is built for each premise:
  1. `hit = {"chunk_index", "text", "metadata"}`, taken from `texts` and `metadatas`.
  2. `expand_neighbours([hit], …, window=JUDGEMENT_NEIGHBOUR_WINDOW)`.
  3. `assemble_evidence([hit], JUDGEMENT_EVIDENCE_MAX_CHARS)`.
  4. `judge(claim, evidence)`.

- [ ] **Step 1: Write the failing tests**

```python
TEXTS = ["the sentence before", "the cited chunk", "a figure", "after"]
METAS = [{"document": "a.pdf", "type": t} for t in ("text_chunk", "text_chunk", "figure_image", "text_chunk")]
OK = {"judgement": "Supports", "span_verified": True, "supporting_span": "the cited chunk", "reason": "r"}

class TestPremiseChecker(unittest.TestCase):
    def test_judged_against_its_chunk_and_its_text_neighbour_not_the_figure(self):
        seen = []
        p = premises.PremiseChecker(lambda c, e: (seen.append(e), OK)[1], TEXTS, METAS).check(models.Premise("A.", "C1", "a.pdf", 1))
        self.assertIn("the cited chunk", seen[0]); self.assertIn("the sentence before", seen[0]); self.assertNotIn("a figure", seen[0])
        self.assertTrue(p.accepted)

    def test_the_same_claim_on_the_same_chunk_is_judged_once(self):
        ch = premises.PremiseChecker(lambda c, e: OK, TEXTS, METAS)
        ch.check_all([models.Premise("A  claim.", "C1", "a.pdf", 1), models.Premise("a claim.", "C1", "a.pdf", 1)])
        self.assertEqual(ch.calls, 1)

    def test_a_judge_failure_earns_nothing_and_the_next_premise_is_still_judged(self):
        def judge(c, e):
            if c == "bad": raise JudgementParseError("Reply was not valid JSON", raw="{")
            return OK
        bad, good = premises.PremiseChecker(judge, TEXTS, METAS).check_all(
            [models.Premise("bad", "C1", "a.pdf", 1), models.Premise("good", "C1", "a.pdf", 1)])
        self.assertEqual(bad.credit, 0.0); self.assertIn("JudgementParseError", bad.error); self.assertTrue(good.accepted)
```

- [ ] **Step 2: Run and confirm they fail.** Then **Step 3: implement**, and **Step 4: run and confirm they pass**.

Run: `python -m pytest tests/test_transfer_premises.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/premises.py tests/test_transfer_premises.py
git commit -m "feat(transfer): judge premises the way the citer judges a candidate"
```

---

### Task 4: Technique cards — select, build, store, search

**Files:**
- Create: `research_assistant/transfer/prompts.py` (with only `SYSTEM`, `CARDS_USER` and `format_passages`), `research_assistant/transfer/cards.py`, `scripts/build_cards.py`
- Modify: `research_assistant/config.py`
- Test: `tests/test_transfer_cards.py`, `tests/test_build_cards_cli.py`

**Interfaces.**

`prompts.py`:
- `format_passages(passages: list[Passage], max_chars: int) -> str` writes one block per passage, `[C7] (Paper B · <section_raw or section>) text`, cut from the tail at `max_chars`.
- `CARDS_USER` has the fields `{citation}`, `{passages}` and `{n}`. Its rules, verbatim:
  1. "List at most {n} techniques this paper uses that someone in another field could reuse. A finding is not a technique."
  2. "problem_solved states the problem the technique solves with no material, system, organism or field named."
  3. "context says how this paper used it."
  4. "evidence is one sentence the cited passage states, citing exactly one key, written like C7."
  5. "If the paper uses no reusable technique, return an empty list."
  6. "Reply with the JSON object only."
- `CARDS_USER` asks for this reply shape: `{"techniques": [{"name", "problem_solved", "context", "inputs", "outputs", "assumptions": [str], "evidence": {"claim", "key"}}]}`.

`cards.py`:
- `select_card_passages(document, citation, texts, metadatas, max_chars) -> list[Passage]`
  - Picks, in this order: chunks with `section == "abstract"`; chunks with `section == "methods"`; then the first chunk under each `section_raw` heading not already used, in document order.
  - Skips `type in EXCLUDED_FROM_EVIDENCE` and `section == "caption"`.
  - Stops before `max_chars` of text is exceeded.
  - Keys are `C1…`; role is `"card"`.
- `build_card(document, citation, passages, *, chat, checker) -> tuple[list[Card], list[str]]`
  - Asks with `ask_json(required=("techniques",))`.
  - Resolves each technique's evidence through `resolve_premises(limit=1)`. A technique whose evidence doesn't resolve is dropped with a warning.
  - Judges the evidence with `checker`.
  - Card ids are `f"{document}#{i}"`, counting only the techniques kept.
- `class CardStore(directory)`
  - Holds `cards.jsonl` (one card per line), `cards.npy` (the `problem_solved` vectors, unit-normalised, row-aligned with the JSONL) and `manifest.json` (`{document: {"cards": n, "at": iso}}`).
  - `.load() -> CardIndex`.
  - `.append(document, cards, vectors) -> None` rewrites all three files atomically.
  - `.done_documents() -> set[str]`.
- `class CardIndex(cards: list[Card], vectors: np.ndarray)`
  - `.search(vector, k, exclude_documents=frozenset(), verified_only=True) -> list[tuple[Card, float]]` returns results by cosine similarity, best first.
- `build_all(documents: list[tuple[str, str]], *, chat, checker, embed_documents, store, texts, metadatas, max_chars, cancelled=lambda: False, progress=lambda d, t, label: None) -> dict` returns `{"built", "skipped", "cards", "warnings"}` and skips documents that are already done.

Config:
- `TRANSFER_CARD_MAX_CHARS = _env_int("CITATION_TRANSFER_CARD_MAX_CHARS", 9000)`
- `TRANSFER_CARD_MAX_TECHNIQUES = _env_int(…, 3)`

`scripts/build_cards.py` is the CLI. `main(argv=None) -> int`:
- **Arguments:** `[--documents doc …] [--sample N --seed S] [--limit N]`.
- **Guard:** checks `pipeline_status.get_status()["active"]` directly, with `pipeline_status` imported by name so the test can patch it. Exit code 2 when busy, before anything loads.
- **Inputs:** documents and citations come from `load_search_resources()` metadata.
- **Model calls:** `chat` is `shared.llm.chat(messages, temperature=0.2, options=TRANSFER_OLLAMA_OPTIONS).content`. The checker's judge is `retry(max_retries=2, backoff=1.0)(judge)`.
- **Output:** writes to `TRANSFER_DIR`, prints progress to stderr and a summary line to stdout.

- [ ] **Step 1: Write the failing tests**

```python
class TestSelect(unittest.TestCase):
    def test_abstract_then_methods_then_one_chunk_per_heading_without_figures_or_captions(self):
        texts = ["abs", "intro1", "intro2", "meth", "cap", "img", "green1"]
        metas = [{"document": "d", "type": "text_chunk", "section": s, "section_raw": r} for s, r in
                 [("abstract", ""), ("introduction", "Introduction"), ("introduction", "Introduction"), ("methods", "Model"),
                  ("caption", "Fig. 1"), ("other", "Fig. 1"), ("other", "Quantum Green's functions")]]
        metas[5]["type"] = "figure_image"
        got = cards.select_card_passages("d", "D", texts, metas, max_chars=10_000)
        self.assertEqual([p.text for p in got], ["abs", "meth", "intro1", "green1"])

    def test_selection_stops_before_the_character_budget(self):
        metas = [{"document": "d", "type": "text_chunk", "section": "other", "section_raw": f"H{i}"} for i in range(5)]
        self.assertEqual(len(cards.select_card_passages("d", "D", ["x" * 400] * 5, metas, max_chars=1000)), 2)

class TestBuild(unittest.TestCase):
    def test_a_technique_whose_evidence_does_not_resolve_is_dropped_and_the_rest_verified(self):
        ps = [models.Passage("C1", 0, "d", "D", "methods", "We use Chebyshev expansion.", "card")]
        reply = json.dumps({"techniques": [
            {"name": "Chebyshev expansion", "problem_solved": "approximate a function of a large sparse matrix", "context": "DOS",
             "inputs": "sparse matrix", "outputs": "spectral density", "assumptions": ["bounded spectrum"],
             "evidence": {"claim": "The paper expands the DOS in Chebyshev polynomials.", "key": "C1"}},
            {"name": "x", "problem_solved": "y", "context": "", "inputs": "", "outputs": "", "assumptions": [],
             "evidence": {"claim": "z", "key": "C9"}}]})
        checker = premises.PremiseChecker(lambda c, e: {"judgement": "Supports", "span_verified": True,
                                                        "supporting_span": "s", "reason": "r"}, ["We use Chebyshev expansion."], [{"document": "d"}])
        got, warnings = cards.build_card("d", "D", ps, chat=lambda m: reply, checker=checker)
        self.assertEqual([(c.card_id, c.verified) for c in got], [("d#0", True)]); self.assertTrue(any("C9" in w for w in warnings))

class TestStoreAndIndex(unittest.TestCase):
    def test_the_store_resumes_and_the_index_excludes_documents_and_unverified_cards(self):
        with tempfile.TemporaryDirectory() as d:
            store = cards.CardStore(d)
            a, b, u = card("a.pdf", verified=True), card("b.pdf", verified=True), card("u.pdf", verified=False)  # fixture helper
            store.append("a.pdf", [a], np.array([[1.0, 0.0]])); store.append("b.pdf", [b], np.array([[0.8, 0.6]]))
            store.append("u.pdf", [u], np.array([[1.0, 0.01]]))
            self.assertEqual(store.done_documents(), {"a.pdf", "b.pdf", "u.pdf"})
            ix = store.load()
            self.assertEqual([c.document for c, _ in ix.search(np.array([1.0, 0.0]), 3)], ["a.pdf", "b.pdf"])
            self.assertEqual([c.document for c, _ in ix.search(np.array([1.0, 0.0]), 3, exclude_documents={"a.pdf"})], ["b.pdf"])

    def test_build_all_skips_documents_already_built(self):
        ...  # store with "a.pdf" done; build_all over [("a.pdf","A"), ("b.pdf","B")] calls chat once; result["skipped"] == 1
```

`card(document, verified)` is added to `tests/transfer_fixtures.py`.

```python
# tests/test_build_cards_cli.py
class TestBuildCardsCli(unittest.TestCase):
    def test_a_busy_pipeline_exits_2_before_loading_anything(self):
        with patch.object(build_cards.pipeline_status, "get_status", return_value={"active": True, "stage_label": "Ingesting"}), \
             patch.object(build_cards, "load_search_resources") as lsr, redirect_stderr(io.StringIO()):
            self.assertEqual(build_cards.main(["--limit", "1"]), 2)
        lsr.assert_not_called()
```

- [ ] **Step 2: Run and confirm they fail**

Run: `python -m pytest tests/test_transfer_cards.py tests/test_build_cards_cli.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `prompts.py`, `cards.py`, `scripts/build_cards.py` and the config lines**

- [ ] **Step 4: Run and confirm they pass**

Run the same command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/prompts.py research_assistant/transfer/cards.py scripts/build_cards.py \
        research_assistant/config.py tests/transfer_fixtures.py tests/test_transfer_cards.py tests/test_build_cards_cli.py
git commit -m "feat(transfer): technique cards: what a paper did, verified by the judge, searchable"
```

- [ ] **Step 6: Live smoke check (maintainer, after the migration finishes)**

Run: `PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/build_cards.py --sample 20 --seed 1`
Expected:
- the summary reports `built` = 20 and at least one card;
- `data/transfer/cards.jsonl` exists.

---

### Task 5: Gate F1 — card review sheet and scoring

**Files:**
- Create: `research_assistant/transfer/review.py`, `scripts/review_cards.py`
- Test: `tests/test_transfer_review.py`

**Interfaces:**
- `review_sheet(index: CardIndex, documents: list[str]) -> tuple[str, str]` returns `(markdown, csv_text)`.
  - The Markdown has one block per card: citation, name, `problem_solved`, context, inputs, outputs, assumptions, the evidence claim with its supporting span, and the verdict.
  - The CSV header is `card_id,rating,specific`, with one row per card and the rating columns left empty.
- `score_review(rows: list[dict]) -> dict` returns `{"rated", "correct_or_partial", "specific", "passed"}`.
  - `rating` is one of `correct`, `partial` or `wrong`; `specific` is `y` or `n`. Blank rows are ignored.
  - `passed` is true when `correct_or_partial >= 0.70` and `specific >= 0.60`.
- `scripts/review_cards.py`:
  - `--sheet [--documents …]` writes `data/eval/transfer/f1/<stamp>-cards.md` and `<stamp>-cards.csv`, defaulting to the documents from the last `--sample` build (recorded in `manifest.json` under `"_last_sample"`).
  - `--score CSV` prints the summary dict and `F1 PASS` or `F1 FAIL`, exiting 0 on a pass and 1 on a fail.

- [ ] **Step 1: Write the failing tests**

```python
class TestReview(unittest.TestCase):
    def test_the_gate_needs_seventy_percent_usable_and_sixty_percent_specific(self):
        rows = [{"card_id": str(i), "rating": r, "specific": s} for i, (r, s) in
                enumerate([("correct", "y")] * 6 + [("partial", "y")] + [("wrong", "n")] * 3)]
        self.assertEqual(review.score_review(rows), {"rated": 10, "correct_or_partial": 0.7, "specific": 0.7, "passed": True})
        rows[0]["rating"] = "wrong"; self.assertFalse(review.score_review(rows)["passed"])

    def test_blank_rows_are_ignored(self):
        self.assertEqual(review.score_review([{"card_id": "1", "rating": "", "specific": ""}])["rated"], 0)

    def test_the_sheet_has_one_block_and_one_csv_row_per_card(self):
        with tempfile.TemporaryDirectory() as d:
            store = cards.CardStore(d); store.append("a.pdf", [card("a.pdf", True)], np.array([[1.0, 0.0]]))
            md, csv_text = review.review_sheet(store.load(), ["a.pdf"])
        self.assertIn("a.pdf#0", md); self.assertEqual(csv_text.splitlines(), ["card_id,rating,specific", "a.pdf#0,,"])
```

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_review.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/review.py scripts/review_cards.py tests/test_transfer_review.py
git commit -m "feat(transfer): F1 review sheet and pass rule for technique cards"
```

- [ ] **Step 6: GATE F1 — stop and report**

The maintainer:
1. Runs `scripts/review_cards.py --sheet`.
2. Rates the CSV.
3. Runs `scripts/review_cards.py --score <csv>`.
4. Records the result in the spec, §10.

On `F1 FAIL`, stop. Card prompts and passage selection get fixed before Phase C. Phase C does not start until the maintainer says F1 passed.

---

### Task 6: Dependencies, settings and stage S1 (coverage and bottlenecks)

**Files:**
- Create: `research_assistant/transfer/deps.py`, `research_assistant/transfer/stages.py`
- Modify: `research_assistant/transfer/prompts.py` (add `BOTTLENECKS_USER`), `tests/transfer_fixtures.py`
- Test: `tests/test_transfer_stages.py`

**Interfaces.**

`deps.py` defines `@dataclass Deps`. Every field except the last two is required:

| Field | Type |
|---|---|
| `chat` | `Callable[[list[dict], float], str]` |
| `critic` | `Callable[[list[dict]], str]` |
| `judge` | `Callable[[str, str], dict]` |
| `embed_query` | `Callable[[str], list[float]]` |
| `embed_documents` | `Callable[[list[str]], list[list[float]]]` |
| `shortlist` | `Callable[[str, int], list[dict]]` |
| `gate` | `Callable[[str, list[dict]], list[dict]]` |
| `search` | `Callable[[str, int, set \| None, set \| None], list[dict]]` |
| `prior_art` | `Callable[[str, list[str], str, int \| None], PriorArt]` |
| `texts` | `list[str]` |
| `metadatas` | `list[dict]` |
| `cards` | `CardIndex` |
| `cancelled` | `Callable[[], bool] = lambda: False` |
| `progress` | `Callable[[int, int, str], None] = lambda d, t, label: None` |

The arguments to `search` are `(query, top_k, doc_filter, exclude_docs)`.

`deps.py` also defines `@dataclass TransferSettings`:
- `bottlenecks`, `cards_per_query`, `techniques_per_bottleneck`, `near_passages`, `source_passages`, `passage_max_chars`, `min_near_papers`, `temperature` and `bottleneck_temperature` each default to the matching `TRANSFER_*` config value;
- `exclude_documents: frozenset = frozenset()`;
- `prior_art_before_year: int | None = None`.

`stages.py`:
- `class CorpusCoverageError(RuntimeError)`, constructed as `CorpusCoverageError(query: str, kept: list[dict], shortlist: list[dict])`. Its message names the shortlisted citations and says to build a corpus in the Research idea tab.
- `@dataclass Grounding(query, near: list[dict], passages: dict[str, Passage], bottlenecks: list[Bottleneck], warnings: list[str])`
- `ground(query, deps, settings, checker) -> Grounding`, which:
  1. Takes `shortlist = [r for r in deps.shortlist(query, DOC_SELECT_K) if r["document"] not in settings.exclude_documents]`.
  2. Runs `kept = deps.gate(query, shortlist)`. If fewer than `settings.min_near_papers` papers are kept, it raises `CorpusCoverageError`, and no other dependency has been called.
  3. Calls `deps.search(query, settings.near_passages, kept_docs, settings.exclude_documents or None)`, giving passages `C1…` with role `"near"`, deduplicated by chunk.
  4. Asks with `BOTTLENECKS_USER` at `settings.bottleneck_temperature`, using `required=("bottlenecks",)`. Up to `settings.bottlenecks` items are kept, with ids `B1…`. Premises use `resolve_premises(limit=3)`.
  5. Drops a bottleneck with no premise, no quantities, or any empty generic level, with a warning. If the reply never parses, the bottleneck list is empty and a warning is added.
  6. Judges all bottleneck premises with `checker`.
- `BOTTLENECKS_USER` has the fields `{query}`, `{passages}` and `{n}`. Its rules, verbatim:
  1. "A bottleneck is what makes this problem expensive, ill-posed or data-starved, as the passages show it."
  2. "quantities are the quantities the bottleneck limits, as the field names them."
  3. "current_method is how the passages solve it today."
  4. "generic restates the bottleneck three ways — specific, mid and broad — naming no material, system, organism or field, so that a paper from another field could match it."
  5. "Each bottleneck has 1 to 3 premises: one sentence the cited passage states, citing exactly one key, written like C7."
  6. "Reply with the JSON object only."
- The reply shape: `{"bottlenecks": [{"statement", "quantities": [str], "current_method", "generic": {"specific", "mid", "broad"}, "premises": [{"claim", "key"}]}]}`.

`tests/transfer_fixtures.py` gets `world(**overrides) -> (deps, log)`:

- **Shortlist:** returns `n0…n5`. The gate keeps the first `gate_keeps` (default 3).
- **Search:** returns one hit per document in `doc_filter`, sorted, with text `text of <doc>` and real `chunk_index` values into the fixture's `texts`. It logs `(doc_filter, exclude_docs)`.
- **Vectors** (2-D, at the angles given):

  | Text | Angle |
  |---|---|
  | `"g1-specific"`, `"g1-mid"`, `"g1-broad"` | 0°, 5°, 10° |
  | `"g2-*"` | 90°, 95°, 100° |
  | everything else | 45° |

- **Cards:**

  | Card | Angle | Note |
  |---|---|---|
  | s1, s2, s3 | 2°, 7°, 12° | verified |
  | s4, s5, s6 | 92°, 97°, 102° | verified |
  | n0 | 1° | verified, but its document is a near paper |
  | s7 | 91° | **unverified** |

  Each card's name is `"method <doc>"`, or as overridden by `card_names`.
- **Chat**, routed on the `TASK:` line:
  - `bottlenecks` returns two bottlenecks:

    | | B1 | B2 |
    |---|---|---|
    | Generic forms | `g1-*` | `g2-*` |
    | Quantities | `["conductance"]` | `["conductance"]` |
    | Current method | `"recursive Green's function"` | `"recursive Green's function"` |
    | Premises | `"bottleneck fact one"` on C1, `"bottleneck fact two"` on C2 | `"bottleneck fact two"` on C2, `"bottleneck fact three"` on C3 |

  - `transfer` returns a reply as complete as `good_transfer()`. Its premises are `"near fact <K1>"` on the **first** `[C…]` key in the prompt and `"source fact <K2>"` on the **last** one.
  - `revise` answers every objection.
  - `replies` maps `(kind, n)` to override replies; `n` is the 1-based call number of that kind.
- **Critic:** returns `shape_match "yes"`, every row `holds`, and one objection. It is counted separately.
- **Judge:** returns Supports with a verified span, and logs the claim.
- **Prior art:** returns `PriorArt("not_found")`, or `"done_before"` when the card name contains `known`.
- **Cancellation:** `cancelled` becomes true once `cancel_after` transfers have been written.
- **Log:** `chat_calls`, `critic_calls`, `judged`, `searches` and `progress`.

- [ ] **Step 1: Write the failing tests**

```python
class TestGround(unittest.TestCase):
    def test_a_question_the_corpus_does_not_cover_is_refused_before_anything_else(self):
        deps, log = world(gate_keeps=1)
        with self.assertRaises(stages.CorpusCoverageError) as cm:
            stages.ground("lithium dendrites with MXenes", deps, TransferSettings(), PremiseChecker(deps.judge, deps.texts, deps.metadatas))
        self.assertEqual((log.chat_calls, log.searches, log.judged), (0, [], [])); self.assertIn("Research idea", str(cm.exception))

    def test_bottlenecks_are_grounded_in_the_kept_papers_and_shared_premises_judged_once(self):
        deps, log = world(); g = stages.ground("q", deps, TransferSettings(), PremiseChecker(deps.judge, deps.texts, deps.metadatas))
        self.assertEqual([b.id for b in g.bottlenecks], ["B1", "B2"])
        self.assertEqual(log.searches[0][0], {"n0", "n1", "n2"})
        self.assertEqual(log.judged.count("bottleneck fact two"), 1)

    def test_excluded_documents_never_reach_the_gate_or_the_search(self):
        deps, log = world(); stages.ground("q", deps, TransferSettings(exclude_documents=frozenset({"n0"})),
                                           PremiseChecker(deps.judge, deps.texts, deps.metadatas))
        self.assertEqual(log.searches[0], ({"n1", "n2", "n3"}, {"n0"}))
```

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_stages.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/deps.py research_assistant/transfer/stages.py research_assistant/transfer/prompts.py \
        tests/transfer_fixtures.py tests/test_transfer_stages.py
git commit -m "feat(transfer): ground bottlenecks in the nearest papers, behind the coverage gate"
```

---

### Task 7: Stage S2 — candidate cards

**Files:**
- Modify: `research_assistant/transfer/stages.py`
- Test: `tests/test_transfer_stages.py`

**Interfaces:**
- `@dataclass Seed(bottleneck_id: str, card: Card, score: float, level: str, rank: int)`
- `candidate_cards(grounding, deps, settings) -> tuple[list[Seed], list[str]]`. For each bottleneck:
  1. For each level in `("specific", "mid", "broad")`, search with `deps.cards.search(embed_query(generic[level]), settings.cards_per_query, exclude_documents=near_docs | settings.exclude_documents)`.
  2. Merge the hits by `card_id`, keeping the best score together with that hit's level and its 1-based rank within that level.
  3. Keep the top `settings.techniques_per_bottleneck` by score.
  4. If no hits remain, warn: `corpus gap for B<i>: no verified card matches '<generic mid>'`.

  The same card may serve two bottlenecks.

- [ ] **Step 1: Write the failing tests**

```python
class TestCandidates(unittest.TestCase):
    def setUp(self):
        self.deps, self.log = world()
        self.g = stages.ground("q", self.deps, TransferSettings(), PremiseChecker(self.deps.judge, self.deps.texts, self.deps.metadatas))

    def test_each_bottleneck_gets_its_nearest_verified_cards_and_never_a_near_paper_or_unverified_card(self):
        seeds, _ = stages.candidate_cards(self.g, self.deps, TransferSettings())
        by_b = {b: sorted(s.card.document for s in seeds if s.bottleneck_id == b) for b in ("B1", "B2")}
        self.assertEqual(by_b, {"B1": ["s1", "s2", "s3"], "B2": ["s4", "s5", "s6"]})

    def test_every_seed_records_its_level_and_rank(self):
        seeds, _ = stages.candidate_cards(self.g, self.deps, TransferSettings())
        s1 = next(s for s in seeds if s.card.document == "s1")
        self.assertEqual((s1.level, s1.rank), ("specific", 1))

    def test_an_empty_card_index_reports_a_corpus_gap(self):
        self.deps.cards = CardIndex([], np.zeros((0, 2)))
        seeds, warnings = stages.candidate_cards(self.g, self.deps, TransferSettings())
        self.assertEqual(seeds, []); self.assertTrue(any(w.startswith("corpus gap for B1") for w in warnings))
```

Why the expected values hold: B1's specific form sits at 0°. The nearest card, n0 at 1°, belongs to a near paper, so the search starts from s1 at 2°, which ranks 1.

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_stages.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/stages.py tests/test_transfer_stages.py
git commit -m "feat(transfer): technique candidates from three abstraction levels per bottleneck"
```

---

### Task 8: Stages S3–S5 — write, judge, critique, revise

**Files:**
- Modify: `research_assistant/transfer/stages.py`, `research_assistant/transfer/prompts.py` (add `TRANSFER_USER`, `CRITIQUE_USER` and `REVISE_USER`)
- Test: `tests/test_transfer_stages.py`

**Interfaces.**

`develop(seed, n, grounding, deps, settings, checker) -> Transfer | None` writes the transfer whose id is `f"{seed.bottleneck_id}.T{n}"`. It does not score it; that is Task 10.

1. **Cancellation.** If `deps.cancelled()` is true, raise `PipelineCancelledError("Technique transfer stopped by user")`.
2. **Source passages.**
   - The card's evidence chunk comes first.
   - Then `deps.search(card.problem_solved, settings.source_passages, {card.document}, exclude or None)` adds more.
   - They are deduplicated by chunk and given role `"source"`. Their keys continue the run's numbering in `grounding.passages`, so the first transfer's source passage is `C4` in the fixture.
3. **The `TRANSFER_USER` prompt** shows, in order:
   1. the bottleneck: statement, quantities, current method, and premises rendered as `- <claim> (C2)`;
   2. the card's fields;
   3. the passages behind the bottleneck's premises;
   4. the source passages, **last**.

   Its rules, verbatim:
   1. "Map every role the technique has in its source onto the target problem, one row each; mark each row's assumption holds, breaks or enforce, and core if the transfer fails without it."
   2. "Identifiability: say what the measurements are, what the unknowns are, any symmetry that makes different unknowns give the same measurements, and whether the measurements determine the unknowns."
   3. "Steps name an input, an output and a tool. data_requirement gives a number and its basis. baseline is the current method. metric measures one of the quantities. prediction states a direction or a number; refuting_result names the outcome that would refute the transfer."
   4. "Give 2 to 4 premises: one sentence the cited passage states, citing exactly one key, written like C7; at least one from the source paper."
   5. "Reply with the JSON object only."

   It is asked with `required = ("title", "summary", "mapping", "identifiability", "steps", "data_requirement", "baseline", "metric", "prediction", "refuting_result", "premises")`.

   If the reply never parses, `develop` returns `None` after recording the warning `"<id>: the model's reply was not usable JSON; skipped"` in `grounding.warnings`.
4. **Premises.** Resolve them with `resolve_premises(limit=4)` against **that prompt's passages only**, then judge them.
5. **Critique.** `CRITIQUE_USER` goes to `deps.critic`. Its rule, verbatim: "You are reviewing this transfer to reject it if you can. For each mapping row say holds, breaks or enforce, with a reason; say whether the problem shapes match (yes, partial, no); override the identifiability verdict if it is wrong; list your strongest objections."

   The reply shape is `{"shape_match", "rows": [{"row", "status", "reason"}], "identifiability": str | null, "objections": [str]}`. If the critic raises or the reply never parses, the result is an empty `Critique()` plus a warning.
6. **Revision.** `REVISE_USER` goes to `deps.chat` with the objections. The reply shape is `{"responses": [{"objection", "response", "concede"}]}`. Each response marks its objection `answered`; `objection` is the 0-based index.
7. **Prior art.** `transfer.prior_art = deps.prior_art(card.name, bottleneck.quantities, transfer.summary, settings.prior_art_before_year)`.

- [ ] **Step 1: Write the failing tests**

```python
class TestDevelop(unittest.TestCase):
    def _go(self, **kw):
        deps, log = world(**kw); s = TransferSettings(); ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        g = stages.ground("q", deps, s, ch); seeds, _ = stages.candidate_cards(g, deps, s)
        return deps, log, g, seeds, s, ch

    def test_a_complete_transfer_has_source_premises_critique_answers_and_prior_art(self):
        deps, log, g, seeds, s, ch = self._go()
        t = stages.develop(seeds[0], 1, g, deps, s, ch)
        self.assertEqual(t.id, "B1.T1"); self.assertEqual(t.card_document, "s1")
        self.assertTrue(any(p.document == "s1" and p.accepted for p in t.premises))
        self.assertTrue(all(o["answered"] for o in t.critique.objections)); self.assertEqual(t.prior_art.status, "not_found")
        self.assertEqual(log.critic_calls, 1)

    def test_a_premise_citing_a_passage_from_another_prompt_is_dropped(self):
        bad = good_transfer_reply(premises=[{"claim": "x", "key": "C3"}, {"claim": "source fact", "key": "C4"}])  # fixture helper
        deps, log, g, seeds, s, ch = self._go(replies={("transfer", 1): [bad]})
        t = stages.develop(seeds[0], 1, g, deps, s, ch)          # B1's prompt shows C1, C2 and the source passage C4
        self.assertEqual([p.key for p in t.premises], ["C4"]); self.assertTrue(any("C3" in w for w in t.warnings))

    def test_a_reply_that_never_parses_is_skipped_with_a_warning(self):
        deps, log, g, seeds, s, ch = self._go(replies={("transfer", 1): ["nope", "still nope"]})
        self.assertIsNone(stages.develop(seeds[0], 1, g, deps, s, ch)); self.assertTrue(any(w.startswith("B1.T1:") for w in g.warnings))

    def test_a_failing_critic_leaves_an_empty_critique_not_a_crash(self):
        deps, log, g, seeds, s, ch = self._go(); deps.critic = lambda m: (_ for _ in ()).throw(RuntimeError("down"))
        t = stages.develop(seeds[0], 1, g, deps, s, ch)
        self.assertEqual(t.critique.objections, []); self.assertTrue(any("critic" in w for w in t.warnings))

    def test_cancelling_stops_before_the_transfer(self):
        deps, log, g, seeds, s, ch = self._go(); deps.cancelled = lambda: True
        with self.assertRaises(pipeline_status.PipelineCancelledError): stages.develop(seeds[0], 1, g, deps, s, ch)
```

`good_transfer_reply(**overrides) -> str` is a fixture helper: the `transfer` reply as JSON, with fields overridden.

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_stages.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/stages.py research_assistant/transfer/prompts.py tests/transfer_fixtures.py tests/test_transfer_stages.py
git commit -m "feat(transfer): write, judge, critique and revise one transfer"
```

---

### Task 9: Prior art, with abstracts and a cutoff year

**Files:**
- Create: `research_assistant/transfer/prior_art.py`
- Modify: `research_assistant/config.py`
- Test: `tests/test_transfer_prior_art.py`

**Interfaces:**
- `reconstruct_abstract(inverted_index: dict | None) -> str` rebuilds OpenAlex's `abstract_inverted_index` into word order; `None` gives `""`.
- `parse_openalex(payload: dict) -> list[dict]` returns `[{"title", "year", "url", "abstract"}]`, where `url` is the DOI or the OpenAlex id.
- `parse_arxiv(xml_text: str) -> list[dict]` returns the same shape. Its `abstract` is the entry `summary`; `year` comes from `published[:4]`.
- `class PriorArtClient(get_json, get_text, embed_documents, limit=10)`:
  - `get_json(url, params) -> dict` and `get_text(url, params) -> str` are injected; the live versions use `requests` with a 20-second timeout and `OPENALEX_MAILTO`.
  - `.check(technique_name, quantities, summary, before_year=None) -> PriorArt`
    1. The query is `f"{technique_name} {' '.join(quantities[:2])}"`.
    2. It searches OpenAlex (`search=<query>`, `per_page=limit`, plus `filter=to_publication_date:<before_year-1>-12-31` when a year is given) and arXiv (`all:<query>`).
    3. It drops hits with `year >= before_year`, which also covers arXiv.
    4. It scores each hit as the cosine similarity between `embed_documents([summary])` and `embed_documents([title + " " + abstract])`.
  - The status follows spec §4. `done_before` also requires every content word of `technique_name` (lowercased, length ≥ 4) to appear in the hit's title or abstract.
  - Any network exception gives `PriorArt("unchecked")`.
  - The top 3 hits by similarity are kept.
- Config:
  - `TRANSFER_PRIOR_ART_DONE = float(os.environ.get("CITATION_TRANSFER_PRIOR_ART_DONE", "0.85"))`
  - `TRANSFER_PRIOR_ART_PARTIAL = 0.75`
  - `TRANSFER_PRIOR_ART_BONUS = 0.25`

  The last two get the same kind of override.

- [ ] **Step 1: Write the failing tests**

```python
class TestPriorArt(unittest.TestCase):
    def test_openalex_abstracts_are_rebuilt_in_word_order(self):
        self.assertEqual(prior_art.reconstruct_abstract({"neural": [0], "quantum": [1], "states": [2, 4], "and": [3]}),
                         "neural quantum states and states")

    def test_hits_from_the_cutoff_year_or_later_are_ignored(self):
        oa = {"results": [{"display_name": "Neural network quantum states", "publication_year": 2017, "doi": "d",
                           "abstract_inverted_index": {"restricted": [0], "boltzmann": [1], "machine": [2]}}]}
        c = prior_art.PriorArtClient(lambda u, p: oa, lambda u, p: "<feed xmlns='http://www.w3.org/2005/Atom'/>",
                                     lambda texts: [[1.0, 0.0]] * len(texts))
        self.assertEqual(c.check("restricted boltzmann machine", ["energy"], "s", before_year=2017).status, "not_found")
        self.assertEqual(c.check("restricted boltzmann machine", ["energy"], "s").status, "done_before")

    def test_a_similar_hit_without_the_technique_words_is_only_partial(self):
        oa = {"results": [{"display_name": "Something else", "publication_year": 2010, "doi": "d", "abstract_inverted_index": {"x": [0]}}]}
        c = prior_art.PriorArtClient(lambda u, p: oa, lambda u, p: "<feed xmlns='http://www.w3.org/2005/Atom'/>",
                                     lambda texts: [[1.0, 0.0]] * len(texts))
        self.assertEqual(c.check("chebyshev expansion", ["conductance"], "s").status, "partial")

    def test_a_network_failure_is_unchecked(self):
        def boom(u, p): raise ConnectionError("down")
        self.assertEqual(prior_art.PriorArtClient(boom, boom, lambda t: [[1.0]] * len(t)).check("x", [], "s").status, "unchecked")
```

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_prior_art.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/prior_art.py research_assistant/config.py tests/test_transfer_prior_art.py
git commit -m "feat(transfer): prior-art check on OpenAlex and arXiv abstracts, honouring a cutoff year"
```

---

### Task 10: Scoring — gates, depth and reward

**Files:**
- Create: `research_assistant/transfer/scoring.py`
- Test: `tests/test_transfer_scoring.py`

**Interfaces:**
- `DEPTH_CHECKS = ("mapping_complete", "rows_classified", "identifiability_argued", "steps_concrete", "data_quantified", "baseline_is_current", "prediction_directional", "refutable", "premises_both_sides", "objections_answered")`. Their rules are in spec §4, items 1–10.
  - **Content word:** lowercase, length ≥ 4, letters only.
  - **Direction:** the regex `\b(increase|decrease|higher|lower|larger|smaller|rise|fall|grow|drop|reduce|raise|faster|slower|exceed|more|less|improv|worse)\w*|\d`, case-insensitive.
- `final_rows(transfer) -> list[MappingRow]` applies the critic's row statuses over the writer's.
- `final_identifiability(transfer) -> str` returns the critic's override if it gave one, otherwise the writer's verdict.
- `depth_checks(transfer, bottleneck, card) -> dict[str, bool]`
- `set_aside_reason(transfer, bottleneck, card) -> str | None` follows the gate order in spec §4.
  - `core_assumption_breaks` and `not_identifiable` use the final values.
  - `off_target` applies when no quantity appears, case-insensitively, in `metric`.
- `reward(depth, prior_art_status) -> float` is `depth × (1 + TRANSFER_PRIOR_ART_BONUS)` when the status is `not_found`, and `depth` otherwise.
- `score_transfer(transfer, bottleneck, card) -> Scores`. A transfer that is set aside gets reward 0.0, but its depth is still computed.
- `rank(transfers) -> list[Transfer]` sorts by reward, descending. Set-aside transfers go last, and the sort is stable.

- [ ] **Step 1: Write the failing tests**

```python
class TestScoring(unittest.TestCase):
    def test_the_fixture_transfer_passes_every_gate_and_every_depth_check(self):
        t, b, c = good_transfer(); s = scoring.score_transfer(t, b, c)
        self.assertEqual((s.set_aside, s.depth, s.reward), (None, 1.0, 1.25))

    def test_each_gate_sets_the_transfer_aside_with_its_reason(self):
        cases = {
            "contradicted": lambda t, b, c: t.premises.append(premise("Contradicts", "C9", "src.pdf", 9)),
            "ungrounded_bottleneck": lambda t, b, c: setattr(b, "premises", [premise("Unclear / insufficient evidence")]),
            "unverified_technique": lambda t, b, c: setattr(c, "evidence", premise("Does not support")),
            "shape_mismatch": lambda t, b, c: setattr(t.critique, "shape_match", "no"),
            "core_assumption_breaks": lambda t, b, c: t.critique.rows.append({"row": 0, "status": "breaks", "reason": "2D has no order"}),
            "not_identifiable": lambda t, b, c: setattr(t.critique, "identifiability", "underdetermined"),
            "off_target": lambda t, b, c: setattr(t, "metric", "wall-clock time"),
            "done_before": lambda t, b, c: setattr(t.prior_art, "status", "done_before"),
        }
        for reason, spoil in cases.items():
            t, b, c = good_transfer(); spoil(t, b, c)       # the fixture's row 0 is its core row
            s = scoring.score_transfer(t, b, c)
            self.assertEqual((s.set_aside, s.reward), (reason, 0.0), reason)

    def test_each_depth_item_fails_on_its_own(self):
        spoils = {"data_quantified": ("data_requirement", "a lot of samples"),
                  "baseline_is_current": ("baseline", "random search"),
                  "prediction_directional": ("prediction", "the method will behave differently overall"),
                  "refutable": ("refuting_result", "the conductance error will decrease twofold at equal cost")}
        for name, (field, value) in spoils.items():
            t, b, c = good_transfer(); setattr(t, field, value)
            checks = scoring.depth_checks(t, b, c)
            self.assertEqual([k for k, ok in checks.items() if not ok], [name], name)

    def test_without_novelty_the_reward_is_the_depth(self):
        t, b, c = good_transfer(); t.prior_art.status = "partial"
        self.assertEqual(scoring.score_transfer(t, b, c).reward, 1.0)

    def test_rank_puts_set_aside_transfers_last(self):
        ...  # three transfers with rewards 0.5, 1.2, and 0.0 set aside → ids ordered 1.2, 0.5, set-aside
```

In the `refutable` case, the refuting result equals the fixture's prediction.

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_scoring.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/scoring.py tests/test_transfer_scoring.py
git commit -m "feat(transfer): reward as code-checked gates, a ten-item depth checklist and a prior-art bonus"
```

---

### Task 11: Engine — guards, live dependencies, best-of-N

**Files:**
- Create: `research_assistant/transfer/engine.py`
- Test: `tests/test_transfer_engine.py`

**Interfaces:**
- `class BusyError(RuntimeError)`.
- `ensure_idle() -> None` raises `BusyError(f"Another run is active: {stage_label}. Wait for it to finish.")` when `pipeline_status.get_status()["active"]` is true.
- `make_search(resources) -> Callable` returns `lambda q, k, docs, excl: hybrid_search(q, collection, bm25, texts, metadatas, top_k=k, doc_filter=set(docs) if docs else None, exclude_types=EXCLUDED_FROM_EVIDENCE, exclude_docs=set(excl) if excl else None)`.
- `live_deps(search_resources=None, card_dir=None) -> Deps`:

  | Field | Live implementation |
  |---|---|
  | `chat` | `shared.llm.chat(m, temperature=t, options=TRANSFER_OLLAMA_OPTIONS).content` |
  | `critic` | `chat(m, model=TRANSFER_CRITIC_MODEL or None, temperature=0.0, options=…)`. When no critic model is set, it adds the run warning "critic shares the generator's model" |
  | `judge` | `retry(max_retries=2, backoff=1.0)(judge)` |
  | `shortlist` | `retrieve.rank_documents` |
  | `gate` | `retrieve.gate_documents` |
  | `prior_art` | `PriorArtClient(...).check` |
  | `cards` | `CardStore(card_dir or TRANSFER_DIR).load()` |
  | `cancelled` | `pipeline_status.is_cancel_requested` |

- `run_best_of_n(query, deps, settings=None) -> RunResult`:
  1. It wraps `deps` in counters (`llm` counts `chat` and `gate` calls; `critic`, `judge`, `embed` and `prior_art` count their own).
  2. It runs `ground`, then `candidate_cards`, then `develop` for every seed in order. Transfers are numbered per bottleneck.
  3. It calls `score_transfer` and `rank`.
  4. `progress(done, total, label)` advances over `total = 2 + len(seeds)`. A skipped transfer still advances it.
  5. It fills in `models`, `settings` and `seconds`. Its `cards` field keeps only the cards it used.
- `estimate_calls(settings) -> dict` returns `{"llm": 2 + 2*b*t, "critic": b*t, "judge_low": 1*b + 2*b*t, "judge_high": 3*b + 4*b*t}`, where `b = settings.bottlenecks` and `t = settings.techniques_per_bottleneck`.

- [ ] **Step 1: Write the failing tests**

```python
class TestEngine(unittest.TestCase):
    def test_two_bottlenecks_times_three_cards_make_six_ranked_transfers(self):
        r = engine.run_best_of_n("q", world()[0])
        self.assertEqual(sorted(t.id for t in r.transfers), ["B1.T1", "B1.T2", "B1.T3", "B2.T1", "B2.T2", "B2.T3"])
        self.assertTrue(all(t.scores.set_aside is None for t in r.transfers))

    def test_calls_are_counted_by_kind(self):
        r = engine.run_best_of_n("q", world()[0])
        self.assertEqual((r.calls["llm"], r.calls["critic"]), (14, 6))      # gate + bottlenecks + 6 × (write + revise)

    def test_an_excluded_document_is_never_used_anywhere(self):
        deps, log = world(); r = engine.run_best_of_n("q", deps, TransferSettings(exclude_documents=frozenset({"s1"})))
        self.assertNotIn("s1", {t.card_document for t in r.transfers})
        self.assertTrue(all("s1" in (excl or set()) for _, excl in log.searches))

    def test_done_before_is_ranked_last_with_its_reason(self):
        r = engine.run_best_of_n("q", world(card_names={"s2": "known method"})[0])
        self.assertEqual((r.transfers[-1].card_document, r.transfers[-1].scores.set_aside), ("s2", "done_before"))

    def test_progress_reaches_its_total(self):
        deps, log = world(); engine.run_best_of_n("q", deps); self.assertEqual(log.progress[-1][:2], (8, 8))

    def test_ensure_idle_refuses_while_another_run_is_active(self):
        with patch.object(engine.pipeline_status, "get_status", return_value={"active": True, "stage_label": "Ingesting and summarizing papers"}):
            with self.assertRaisesRegex(engine.BusyError, "Ingesting"): engine.ensure_idle()

    def test_every_live_search_excludes_figures(self):
        with patch.object(engine, "hybrid_search", return_value=[]) as hs:
            engine.make_search(("col", "bm25", ["t"], [{}]))("q", 3, {"a.pdf"}, {"x.pdf"})
        kw = hs.call_args.kwargs
        self.assertEqual((kw["exclude_types"], kw["doc_filter"], kw["exclude_docs"]), (config.EXCLUDED_FROM_EVIDENCE, {"a.pdf"}, {"x.pdf"}))

    def test_estimate_calls_for_three_by_three(self):
        self.assertEqual(engine.estimate_calls(TransferSettings(bottlenecks=3, techniques_per_bottleneck=3)),
                         {"llm": 20, "critic": 9, "judge_low": 21, "judge_high": 45})
```

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_*.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/engine.py tests/test_transfer_engine.py
git commit -m "feat(transfer): best-of-N over bottlenecks x technique cards, with guards and call counts"
```

---

### Task 12: Rendering, storage and the CLI

**Files:**
- Create: `research_assistant/transfer/render.py`, `research_assistant/transfer/store.py`, `scripts/transfer.py`
- Test: `tests/test_transfer_render.py`, `tests/test_transfer_cli.py`

**Interfaces.**

`render.py`:
- `premise_badge(p) -> str` returns, in order of precedence: `"✅ Supports"`, `"🟡 Partially supports"`, `"❌ Contradicts"`, `"⛔ Judge error"`, or otherwise `"⚠️ Not accepted"`.
- `mermaid_label(text, limit=60) -> str` collapses whitespace, turns `"` into `'`, and removes every character in ``#<>&`{}[]|``. Anything longer than `limit` is cut and ends in `…`.
- `render_tree(result) -> str` produces:
  - `flowchart TD` with a root node `R`;
  - a node per bottleneck, `B1["B1 · <statement>"]`;
  - a node per transfer, `T_B1_T2["<card name> · <reward:.2f>"]`;
  - edges `R --> B1 --> T_…`;
  - `classDef best` / `aside` declarations, with lines `class <id> best` for the top transfer that is not set aside and `class <id> aside` for each set-aside one.
- `render_markdown(result) -> str` contains, in order:
  1. The title, `# Technique transfer: <query>`.
  2. A models and calls line.
  3. The caveat: "Premises are checked against their papers; the mapping, steps and prediction are the tool's proposal."
  4. For each bottleneck: its statement and its premises, with badges.
  5. For each transfer, in ranked order:
     - the title, the technique and its card citation;
     - **Proposal (not stated by any source):** the summary;
     - the mapping table, as a Markdown table with the final statuses;
     - identifiability, steps, data, baseline, metric, prediction and refuting result;
     - premises, with badges and their spans;
     - critic objections and their answers;
     - prior art, with its hits.
  6. `## Set aside`, listing each set-aside transfer with its reason.
  7. `## Already done`, holding the `done_before` transfers.
  8. `## Search tree`, a ```` ```mermaid ```` block.

`store.py`:
- `save_result(result, directory=None) -> str` writes `<run_id>.json` and `.md` under `directory or os.path.join(config.TRANSFER_DIR, "runs")`, and returns the JSON path.
- `load_result(path) -> RunResult`.

`scripts/transfer.py` is the CLI. `main(argv=None) -> int`:
- **Arguments:** `--query/-q` (required), `--bottlenecks`, `--techniques`, `--exclude doc …`, `--before-year Y`, `--show-cards`, `--out`.
- **Exit codes:** 0 on success; 2 when busy, printed before anything loads; 3 when the corpus doesn't cover the query.
- **Output:** it prints the JSON and Markdown paths, then one line per transfer: `rank  reward  depth  set-aside  technique  ← card document`.
- **`--show-cards`:** runs `ground` and `candidate_cards` only, using the `chat` call for bottlenecks but no transfers. It prints each bottleneck's three generic forms and its seeds (level, rank, score, card name, document). This is how F2 checks are made by hand.

- [ ] **Step 1: Write the failing tests**

```python
class TestRender(unittest.TestCase):
    def test_hostile_text_cannot_break_a_label(self):
        label = render.mermaid_label('A "b" [c] {d} #1 <e> & `f` |g|\nh ' + "w" * 80, limit=60)
        for ch in '"[]{}#<>&`|\n': self.assertNotIn(ch, label)
        self.assertLessEqual(len(label), 60)

    def test_the_report_labels_the_proposal_and_sets_aside_with_the_reason(self):
        md = render.render_markdown(sample_result())
        for text in ("**Proposal (not stated by any source):**", "## Set aside", "not_identifiable", "```mermaid", "✅ Supports"):
            self.assertIn(text, md)

    def test_the_tree_marks_the_best_and_the_set_aside(self):
        tree = render.render_tree(sample_result())
        for line in ("R --> B1", "B1 --> T_B1_T1", "class T_B1_T1 best", "class T_B1_T2 aside"): self.assertIn(line, tree)

    def test_save_and_load_round_trip_under_runs(self):
        with tempfile.TemporaryDirectory() as d:
            path = store.save_result(sample_result(), directory=d)
            self.assertEqual(path, os.path.join(d, "20261021-120000-abcdef12.json")); self.assertEqual(store.load_result(path), sample_result())

class TestTransferCli(unittest.TestCase):
    def test_busy_exits_2_and_uncovered_exits_3(self):
        with patch.object(cli.engine, "ensure_idle", side_effect=engine.BusyError("Another run is active: Ingesting")), \
             patch.object(cli.engine, "live_deps") as ld, redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["-q", "x"]), 2)
        ld.assert_not_called()
        refusal = stages.CorpusCoverageError("x", [], [{"document": "a", "citation": "Paper A", "summary": "", "score": 0.4}])
        with patch.object(cli.engine, "ensure_idle"), patch.object(cli.engine, "live_deps"), \
             patch.object(cli.engine, "run_best_of_n", side_effect=refusal), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cli.main(["-q", "x"]), 3)
        self.assertIn("Paper A", err.getvalue())

    def test_show_cards_writes_no_transfer(self):
        deps, log = world()
        with patch.object(cli.engine, "ensure_idle"), patch.object(cli.engine, "live_deps", return_value=deps), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["-q", "q", "--show-cards"]), 0)
        self.assertEqual(log.critic_calls, 0); self.assertIn("s1", out.getvalue())
```

The tests import `from scripts import transfer as cli`.

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_render.py tests/test_transfer_cli.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/render.py research_assistant/transfer/store.py scripts/transfer.py \
        tests/test_transfer_render.py tests/test_transfer_cli.py
git commit -m "feat(transfer): Markdown report, escaped search tree, saved runs and the command line"
```

- [ ] **Step 6: Live smoke check (maintainer, after the migration, with the full card build done)**

Run: `PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/build_cards.py`
This builds the rest of the corpus, resuming where it left off.

Then run: `PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/transfer.py -q "Accelerating the forward problem in quantum transport for disordered nanostructures" --show-cards`
Expected:
- 1–3 bottlenecks, each with three generic forms;
- seeds whose cards come from outside the transport papers.

Finally, run the same command without `--show-cards`.
Expected: a report under `data/transfer/runs/`.

---

### Task 13: Retrodiction cases and case corpora

**Files:**
- Create: `research_assistant/eval/transfer_cases.py`, `research_assistant/eval/cases/transfer_retro.jsonl`, `scripts/build_case_corpus.py`
- Test: `tests/test_transfer_cases.py`, `tests/test_build_case_corpus.py`

**Interfaces.**

`@dataclass Case`, in `transfer_cases.py`:

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | |
| `kind` | `str` | `positive` or `negative` |
| `query` | `str` | neutral wording |
| `generic` | `str` | the hand-written generic bottleneck, used by F2 |
| `cutoff_year` | `int \| None` | |
| `corpus` | `str` | `main` or `case` |
| `papers` | `dict[str, list[str]]` | keys `target`, `source`, `distractor` and `hidden`; titles |
| `exclude_documents` | `list[str]` | |
| `expected` | `dict` | keys `source_titles`, `technique_terms`, `mapping_terms`, `critic_terms` and `set_aside` (a reason or null) |

Functions:
- `load_cases(public_path=<eval/cases/transfer_retro.jsonl>, private_dir=os.path.join(DATA_DIR, "eval", "transfer", "answer_keys")) -> list[Case]`.
  - The private directory holds one JSON file per case, and is read only if it exists.
  - A case missing a required field, or with an unknown `kind`, raises `ValueError` naming the case.
- `case_data_dir(case_id) -> str` returns `os.path.join(PROJECT_ROOT, "data", "eval", "transfer", "corpora", case_id)`.

`transfer_retro.jsonl` holds two lines, `nqs` and `kpm`. The maintainer confirms every title during resolution.

| Field | `nqs` | `kpm` |
|---|---|---|
| kind | positive | positive |
| corpus | case | case |
| cutoff | 2017 | 2015 |
| query | "Represent the state of N interacting two-level systems, whose description needs 2^N complex amplitudes, compactly enough to optimise its energy" | "Compute the conductance and density of states of very large disordered tight-binding systems without inverting or diagonalising the Hamiltonian" |
| generic | "represent and sample from a very high-dimensional probability distribution with few parameters" | "approximate a function of a large sparse matrix, or its trace, from matrix-vector products alone" |
| target | "Density matrix formulation for quantum renormalization groups" (White 1992); "The density-matrix renormalization group in the age of matrix product states" (Schollwöck 2011); "A practical introduction to tensor networks" (Orús 2014); "Green's function Monte Carlo with stochastic reconfiguration" (Sorella 1998) | "The recursive Green's function method for graphene" (already in the corpus); a Landauer–Büttiker review |
| source | "Training products of experts by minimizing contrastive divergence" (Hinton 2002); "Representational power of restricted Boltzmann machines and deep belief networks" (Le Roux & Bengio 2008); "An introduction to restricted Boltzmann machines" (Fischer & Igel 2012) | "The kernel polynomial method" (Weiße, Wellein, Alvermann & Fehske 2006); "Randomized algorithms for estimating the trace of an implicit symmetric positive semi-definite matrix" (Avron & Toledo 2011) |
| distractor | "ImageNet classification with deep convolutional neural networks" (2012); "Efficient estimation of word representations in vector space" (2013); "Auto-encoding variational Bayes" (2013) | one FFT paper and one multigrid paper |
| hidden | "Solving the quantum many-body problem with artificial neural networks" (Carleo & Troyer 2017) | "Real-space calculation of the conductivity tensor for disordered topological matter" (García, Covaci & Rappoport 2015) |
| technique_terms | `["boltzmann", "rbm", "neural network"]` | `["chebyshev", "kernel polynomial", "trace estimat"]` |
| mapping_terms | `["spin", "amplitude", "energy", "sampl"]` | `["spectrum", "rescal", "moment", "random vector"]` |
| critic_terms | `["sign", "phase", "complex", "normalis"]` | `["bound", "resolution", "moments", "noise"]` |

`scripts/build_case_corpus.py`. `main(argv=None) -> int` takes `--case ID` and one of `--resolve` or `--confirm`:
- `guard_data_dir(case_id)` exits with code 4 and the message "set CITATION_DATA_DIR=<path> to build this case" unless `os.path.realpath(config.DATA_DIR) == os.path.realpath(case_data_dir(case_id))`.
- `resolve_titles(titles, search) -> list[dict]` returns `[{"title", "match_title", "year", "pdf_url", "score"}]`.
  - `search` is injected; live it is `agent0_discoverer._search_arxiv` followed by `_search_openalex`.
  - `score` is the token Jaccard similarity between the normalised titles. A match needs a score ≥ 0.8 and a `pdf_url`; otherwise `match_title` is `None`.
- `--resolve` writes `<case dir>/resolution.json` and prints a table. Hidden papers are never resolved or downloaded.
- `--confirm` downloads each matched PDF with `shared.fetch.download_pdf` into `<case dir>/pulled_pdfs/`, ingests them with `ingest_pdfs(paths, workers=1, rebuild_index=True, require_grobid=True)`, then writes `<case dir>/resolved.json` mapping each title to its ingested document name.
- The private answer-key template is written to `data/eval/transfer/answer_keys/TEMPLATE.json` by `--write-key-template`, holding every `Case` field with placeholder values. For the maintainer's two cases, `corpus` is `"main"`, `exclude_documents` lists the documents of the unpublished work, and the negative case has `"set_aside": "not_identifiable"`.

- [ ] **Step 1: Write the failing tests**

```python
class TestCases(unittest.TestCase):
    def test_the_public_cases_load_and_a_private_key_is_merged(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({**MINIMAL_CASE, "id": "ae_neg", "kind": "negative", "corpus": "main"}, open(os.path.join(d, "ae_neg.json"), "w"))
            cases = transfer_cases.load_cases(private_dir=d)
        self.assertEqual([c.id for c in cases], ["nqs", "kpm", "ae_neg"])

    def test_a_case_missing_a_field_is_rejected_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"id": "bad", "kind": "positive"}, open(os.path.join(d, "bad.json"), "w"))
            with self.assertRaisesRegex(ValueError, "bad"): transfer_cases.load_cases(private_dir=d)

class TestBuildCaseCorpus(unittest.TestCase):
    def test_it_refuses_to_build_into_any_directory_but_the_cases_own(self):
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(bcc.main(["--case", "nqs", "--resolve"]), 4)      # DATA_DIR is the main data directory in tests
        self.assertIn("CITATION_DATA_DIR", err.getvalue())

    def test_titles_resolve_only_on_a_close_match_with_a_pdf(self):
        hits = [{"title": "Training Products of Experts by Minimizing Contrastive Divergence", "year": "2002", "pdf_url": "u"},
                {"title": "Something unrelated", "year": "2002", "pdf_url": "v"}]
        got = bcc.resolve_titles(["Training products of experts by minimizing contrastive divergence", "Nothing like it"],
                                 search=lambda q, limit: hits)
        self.assertEqual([g["match_title"] is not None for g in got], [True, False])

    def test_hidden_papers_are_never_resolved(self):
        ...  # patch guard_data_dir to pass and resolve_titles to record its titles; "--resolve" never passes a hidden title
```

`MINIMAL_CASE` is a fixture dict with every required field.

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_cases.py tests/test_build_case_corpus.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/eval/transfer_cases.py research_assistant/eval/cases/transfer_retro.jsonl scripts/build_case_corpus.py \
        tests/test_transfer_cases.py tests/test_build_case_corpus.py
git commit -m "feat(eval): retrodiction cases and a guarded case-corpus builder"
```

- [ ] **Step 6: Live steps (maintainer)**

1. For each of `nqs` and `kpm`, run:
   ```bash
   CITATION_DATA_DIR=data/eval/transfer/corpora/<id> PYTHONPATH=. python scripts/build_case_corpus.py --case <id> --resolve
   ```
   Check `resolution.json`, then rerun the same command with `--confirm` in place of `--resolve`.
2. Build cards inside each case corpus:
   ```bash
   CITATION_DATA_DIR=data/eval/transfer/corpora/<id> PYTHONPATH=. python scripts/build_cards.py
   ```
3. Copy `TEMPLATE.json` and write `ae_pos.json` and `ae_neg.json` under `data/eval/transfer/answer_keys/`. Make sure no draft of the unpublished work is in the main corpus, or that it is listed in `exclude_documents`.

---

### Task 14: Retrodiction metrics and harness — Gates F2 and F3

**Files:**
- Create: `research_assistant/eval/transfer_metrics.py`, `scripts/evaluate_transfer.py`
- Test: `tests/test_transfer_metrics.py`, `tests/test_evaluate_transfer.py`

**Interfaces.**

`transfer_metrics.py` is pure, and works on `RunResult` dicts:

- `CAVEAT_NOTE` = "A positive case counts only when the transfer's technique card came from an expected source paper; naming the technique is not enough. Prior art is limited to before the case's cutoff."
- `expected_documents(case, resolved: dict[str, str]) -> set[str]`, using the `source_titles` that resolved.
- `matches(transfer: dict, cards: dict, case, expected_docs) -> bool`. It is true when the transfer's `card_document` is in `expected_docs` **and** any of `technique_terms` appears, case-insensitively, in the card's name or `problem_solved`. **The transfer's own prose is not consulted.**
- `found_rank(result, case, expected_docs) -> int | None`: the 1-based rank among transfers that were not set aside.
- `mapped_share(transfer, case) -> float`: the share of `mapping_terms` found in the text of the mapping rows.
- `critiqued(transfer, case) -> bool`: whether any of `critic_terms` appears in the critique's objections or row reasons.
- `case_passed(result, case, expected_docs) -> dict` returns `{"passed", "found_rank", "mapped", "critiqued", "reason"}`, following the rules in spec §6.
  - For a negative case, every transfer whose card name contains a technique term must be set aside, and the expected `set_aside` must be among their reasons.
- `retrieval_rank(seeds: list[dict], expected_docs) -> int | None` is for F2. Seeds are `{"card_document", "score"}`, ranked by score.
- `leak_guard(case, resolved, corpus_documents: set[str]) -> None` raises `RuntimeError` naming the paper if any hidden title resolves to a document in the corpus.
- `score(records) -> dict` returns `{"cases": {id: {...per run}}, "f2": {...}, "f3": {"passed": n, "of": m, "spread": …}, "note": CAVEAT_NOTE}`. The spread is the mean |run1 − run2| of `found_rank`, with a missing rank counted as 10.

`scripts/evaluate_transfer.py`. `main(argv=None) -> int`:
- **`--retrieval-only [--cases …]`** is gate F2. For each case:
  1. Set `CITATION_DATA_DIR` per case by re-executing itself with the case's directory for `case` corpora, or use the main directory.
  2. Load its cards.
  3. Embed `case.generic` and search with `exclude_documents` applied.
  4. Record `retrieval_rank`.

  It prints `F2 PASS` when the expected card is in the top 5 for at least 3 of 4 cases. It never calls a model.
- **`--score --runs 2 [--cases …]`** is gate F3.
  1. Call `ensure_idle`, then `leak_guard` for each case.
  2. Run `run_best_of_n(case.query, live_deps(...), TransferSettings(exclude_documents=…, prior_art_before_year=case.cutoff_year))`.
  3. Write `data/eval/transfer/results/<stamp>.json` (records and summary) and `<stamp>-review.md` (each case's top 3 transfers, for the maintainer to read).
  4. Print `F3 PASS` when every case passes in both runs.
- **`--compare A B`** prints the per-case differences. Task 15 uses it.

Running a case under its own data directory means re-executing as `subprocess.run([sys.executable, __file__, …], env={**os.environ, "CITATION_DATA_DIR": case_data_dir(id)})`, because `config.DATA_DIR` is fixed at import. The child writes its record into the parent's results path, which is passed as `--record-to`.

- [ ] **Step 1: Write the failing tests**

Test results are built from `sample_result()` with edited transfers, using helpers `_result_with(transfers=[...])` and `_case(kind=...)`.

```python
class TestTransferMetrics(unittest.TestCase):
    def test_a_technique_named_in_prose_but_not_retrieved_from_an_expected_paper_does_not_count(self):
        r = _result_with([{"card_document": "other.pdf", "card_name": "restricted Boltzmann machine", "summary": "RBM ansatz"}])
        self.assertIsNone(m.found_rank(r, _case(technique_terms=["boltzmann"]), {"hinton2002.pdf"}))

    def test_found_rank_counts_only_transfers_not_set_aside(self):
        r = _result_with([{"card_document": "x.pdf", "card_name": "kernel method", "set_aside": None},
                          {"card_document": "hinton2002.pdf", "card_name": "restricted boltzmann machine", "set_aside": "shape_mismatch"},
                          {"card_document": "hinton2002.pdf", "card_name": "restricted boltzmann machine", "set_aside": None}])
        self.assertEqual(m.found_rank(r, _case(technique_terms=["boltzmann"]), {"hinton2002.pdf"}), 2)

    def test_a_negative_case_passes_only_if_every_matching_transfer_is_set_aside_as_expected(self):
        case = _case(kind="negative", technique_terms=["autoencoder"], set_aside="not_identifiable")
        good = _result_with([{"card_document": "a.pdf", "card_name": "autoencoder", "set_aside": "not_identifiable"}])
        bad = _result_with([{"card_document": "a.pdf", "card_name": "autoencoder", "set_aside": None}])
        self.assertTrue(m.case_passed(good, case, set())["passed"]); self.assertFalse(m.case_passed(bad, case, set())["passed"])

    def test_the_leak_guard_stops_a_run_whose_hidden_paper_is_in_the_corpus(self):
        case = _case(hidden=["Solving the quantum many-body problem with artificial neural networks"])
        with self.assertRaisesRegex(RuntimeError, "many-body"):
            m.leak_guard(case, {"Solving the quantum many-body problem with artificial neural networks": "carleo.pdf"}, {"carleo.pdf"})

    def test_retrieval_rank_and_the_f2_rule(self):
        self.assertEqual(m.retrieval_rank([{"card_document": "x", "score": .9}, {"card_document": "k.pdf", "score": .8}], {"k.pdf"}), 2)
        ...  # score() over four F2 records: ranks [1, 3, 7, None] → f2 not passed (2 of 4 within top 5); [1, 3, 5, None] → passed (3 of 4)
```

```python
class TestEvaluateTransfer(unittest.TestCase):
    def test_retrieval_only_never_calls_a_model(self):
        ...  # patch the case runner to use world() deps; assert chat_calls == 0 and critic_calls == 0 and "F2" in stdout

    def test_score_runs_the_leak_guard_before_any_case(self):
        ...  # leak_guard patched to raise for the first case → exit 1, run_best_of_n never called
```

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_metrics.py tests/test_evaluate_transfer.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/eval/transfer_metrics.py scripts/evaluate_transfer.py tests/test_transfer_metrics.py tests/test_evaluate_transfer.py
git commit -m "feat(eval): retrodiction harness: F2 retrieval and F3 full runs, scored on the retrieval trace"
```

- [ ] **Step 6: GATES F2 and F3 — stop and report**

The maintainer:
1. Runs `--retrieval-only`. On `F2 FAIL`, fix the generic forms or the card fields before running F3.
2. Runs `--score --runs 2`, then reads `<stamp>-review.md`.
3. Calibrates against the result: the prior-art thresholds, the bonus, and whether the identifiability gate catches the negative case.
4. Records everything in spec §10.

No unattended run on open questions happens until F3 passes. Task 15 doesn't start without the maintainer's go-ahead.

---

### Task 15 (Phase 2, gated): MCTS budget allocation

Start only if F1–F3 passed and the maintainer says to go on.

**Files:**
- Create: `research_assistant/transfer/mcts.py`
- Modify: `scripts/evaluate_transfer.py` (add `--search mcts|best_of_n` and `--budget N`)
- Test: `tests/test_transfer_mcts.py`

**Interfaces:**
- `run_mcts(query, deps, settings=None, *, budget: int, c_explore: float = 1.4, patience: int = 3) -> RunResult`, with `search="mcts"`. It reuses `ground`, `candidate_cards`, `develop`, `score_transfer` and `rank`.
- **Tree:** root → bottleneck → level → seeds, in rank order.
- **Each iteration:**
  1. Select a bottleneck by UCT (unvisited first), then a level within it by UCT.
  2. Develop that level's next undeveloped seed (progressive widening).
  3. Back up the transfer's reward to the level, the bottleneck and the root.
- **Stopping:** when `budget` transfers have been developed, when no undeveloped seeds remain, or when the top 3 ids haven't changed for `patience` iterations. The reason goes in `warnings` as `"stopped: budget"`, `"stopped: exhausted"` or `"stopped: stable"`.
- **Deduplication:** a card developed for a bottleneck at one level isn't developed again for that bottleneck at another level.

- [ ] **Step 1: Write the failing tests**

```python
class TestMcts(unittest.TestCase):
    def test_the_budget_is_the_number_of_transfers_developed(self):
        r = mcts.run_mcts("q", world()[0], budget=4); self.assertEqual(len(r.transfers), 4); self.assertIn("stopped: budget", r.warnings)

    def test_unvisited_bottlenecks_are_tried_before_any_is_revisited(self):
        r = mcts.run_mcts("q", world()[0], budget=2); self.assertEqual(sorted(t.bottleneck_id for t in r.transfers), ["B1", "B2"])

    def test_a_better_rewarded_bottleneck_gets_more_of_the_budget(self):
        deps, _ = world(card_names={d: "known method" for d in ("s4", "s5", "s6")})   # every B2 transfer is done_before → reward 0
        r = mcts.run_mcts("q", deps, budget=5, patience=99)
        self.assertGreater(sum(t.bottleneck_id == "B1" for t in r.transfers), sum(t.bottleneck_id == "B2" for t in r.transfers))

    def test_the_search_stops_when_seeds_run_out(self):
        r = mcts.run_mcts("q", world()[0], budget=99, patience=99); self.assertEqual(len(r.transfers), 6)
        self.assertIn("stopped: exhausted", r.warnings)
```

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_transfer_mcts.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/transfer/mcts.py scripts/evaluate_transfer.py tests/test_transfer_mcts.py
git commit -m "feat(transfer): MCTS budget allocation over bottlenecks and abstraction levels"
```

- [ ] **Step 6: GATE — stop and report**

The maintainer runs `--score --runs 2 --search best_of_n --budget 9` and the same with `--search mcts --budget 9`, then `--compare`. MCTS becomes the default only if it passes at least as many cases with a lower mean `found_rank`, by more than the spread. Otherwise best-of-N stays the default, and the result goes in spec §10.

---

### Task 16: The tab, run as a background job

**Files:**
- Create: `research_assistant/ui/transfer_tab.py`
- Modify:
  - `research_assistant/shared/pipeline_status.py` (`VALID_STAGES` and `STAGE_METADATA`)
  - `research_assistant/ui/jobs.py` (`_adopt_job_result` and `_handle_job_state`)
  - `research_assistant/ui/common.py` (add `_render_mermaid`)
  - `research_assistant/tasks/ui.py` (replace lines 59–101 with a re-export)
  - `app.py`
- Test: `tests/test_pipeline_status.py` (class `TestPipelineStatus`), `tests/test_app_render.py` (class `TestJobResolution`), `tests/test_app_smoke.py`, `tests/test_app_transfer_tab.py`

**Interfaces:**
- **Pipeline status:** `"transfer"` joins `VALID_STAGES`, and `STAGE_METADATA["transfer"] = (1, "Searching for technique transfers")`. Without it, `set_status` coerces the stage to `idle`.
- **Job adoption:** `ui/jobs.py` gets `_RESULT_KEYS = {"idea": ("idea_result", "idea_query"), "transfer": ("transfer_result", "transfer_query")}`.
  - `_adopt_job_result` writes those keys for the two listed origins and keeps today's audit branch for every other origin.
  - `_handle_job_state` reads `already` from `_RESULT_KEYS.get(origin, ("audit_result",))[0]`.
- **Mermaid:** `ui/common.py::_render_mermaid(mermaid_code, height=420)` is moved verbatim from `tasks/ui.py::_render_mermaid_interactive`. `tasks/ui.py` then imports it as `from research_assistant.ui.common import _render_mermaid as _render_mermaid_interactive`.
- **`ui/transfer_tab.py`:**
  - `render_transfer_tab()`.
  - `_start_transfer_job(query, bottlenecks, techniques) -> str | None` mirrors `_start_pipeline_job`:
    - the job id is `run_jobs.job_id_for(f"transfer|{query}|{bottlenecks}|{techniques}")`;
    - it calls `start_job(..., origin="transfer")`;
    - on `JobBusy` it shows `st.warning`;
    - it sets `st.query_params["job"]` and the active-job keys.
  - `_transfer_runner(query, bottlenecks, techniques) -> Callable[[], dict]` touches no Streamlit and takes no `on_snapshot` argument.
    1. Set `set_status(active=True, stage="transfer", detail=query[:60], item_current=0, item_total=0)`.
    2. Build `live_deps()`, and wire `deps.progress` to `update_progress`.
    3. Run `run_best_of_n`, then `store.save_result`.
    4. Return `result_to_dict`. On `CorpusCoverageError`, return `{"query": query, "coverage_error": str(e)}` instead.
    5. Finally, set `set_status(active=False, stage="idle", detail="Technique transfer finished")`.
  - The tab lays out:
    - `_render_live_pipeline_status("tab_transfer")`;
    - the subheader "🔀 Technique transfer";
    - the caption, beginning "Experimental.";
    - a form, `transfer_form`, with a "Research problem" text input (`key="transfer_query_input"`), "Bottlenecks" (1–3) and "Techniques per bottleneck" (1–4), an `estimate_calls` caption, and the submit button "Search for transfers" (`key="transfer_submit"`).
  - **Submit:**
    - with no query, it warns;
    - while `pipeline_status.get_status()["active"]`, it shows `st.warning(f"Another run is active ({stage_label}). Wait for it to finish, then search again.")`;
    - otherwise it starts the job.
  - **Result:**
    - one expander per transfer, `f"{rank}. {title} — {reward:.2f}"`, with the top 3 expanded, holding:
      - the depth, reward and prior-art status as metrics;
      - "Proposal (not stated by any source)" as `st.info`;
      - the mapping table as `st.dataframe`;
      - identifiability, steps and the test;
      - premises with badges;
      - critic objections and their answers;
      - the card it came from, with the paper, level and rank;
    - "Set aside (n)" and "Already done (n)" expanders;
    - the Mermaid tree;
    - the downloads "Download report (.md)" and "Download run (.json)".
  - Otherwise it calls `_render_recent_runs("transfer")`.
- **`app.py`:**
  - the tabs become `[..., "Research chat", "Technique transfer", "How to use"]`;
  - `with tab_transfer: render_transfer_tab()`;
  - the docstring says seven tabs and lists `transfer_tab`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_status.py — class TestPipelineStatus
def test_the_transfer_stage_is_kept_and_labelled(self):
    s = pipeline_status.set_status(active=True, stage="transfer")
    self.assertEqual((s["stage"], s["stage_label"], s["current_step"]), ("transfer", "Searching for technique transfers", 1))

# tests/test_app_render.py — class TestJobResolution
def test_a_transfer_result_is_adopted_into_its_own_keys(self):
    with patch.object(ui_jobs.st, "session_state", {"active_job": "j"}) as ss:
        ui_jobs._adopt_job_result({"query": "q", "transfers": []}, "transfer")
    self.assertEqual(ss["transfer_query"], "q"); self.assertFalse({"audit_result", "idea_result", "build_result"} & set(ss))

def test_a_finished_transfer_run_belongs_to_the_transfer_tab_only(self):
    self.run_jobs.save_run("k", {"query": "q", "transfers": []}, label="q", origin="transfer")
    with patch.object(ui_jobs.st, "session_state", {"active_job": "k"}) as ss, patch.object(ui_jobs.st, "query_params", {}):
        self.assertFalse(ui_jobs._handle_job_state("idea")); self.assertNotIn("idea_result", ss)
        self.assertFalse(ui_jobs._handle_job_state("transfer")); self.assertEqual(ss["transfer_result"]["query"], "q")

# tests/test_app_smoke.py — replaces test_the_six_tabs_are_there_in_order
def test_the_seven_tabs_are_there_in_order(self):
    self.assertEqual(self.page["tabs"], ["Agent task planner", "Citation auditor", "Research idea", "Cite a draft",
                                         "Research chat", "Technique transfer", "How to use"])

# tests/test_app_transfer_tab.py — skips without streamlit, like tests/test_app_smoke.py
class TestTransferTab(unittest.TestCase):
    def test_a_finished_search_renders_its_transfers_and_both_downloads(self):
        payload = json.dumps(models.result_to_dict(sample_result()))
        page = run_page(before=f'import json\nat.session_state["transfer_result"] = json.loads({payload!r})',
                        after='out["expanders"] = [e.label for e in at.expander]\n'
                              'out["downloads"] = [e.proto.label for e in at.get("download_button")]')
        self.assertEqual(page["exceptions"], [])
        self.assertTrue(any(sample_result().transfers[0].title in l for l in page["expanders"]))
        self.assertIn("Download report (.md)", page["downloads"]); self.assertIn("Download run (.json)", page["downloads"])

    def test_a_search_is_refused_while_another_run_is_active(self):
        before = 'import os\nfrom research_assistant.shared import pipeline_status\npipeline_status.set_status(active=True, stage="ingest_refs", pid=os.getpid())'
        after = ('at.text_input(key="transfer_query_input").input("graphene disorder")\n'
                 'at.button(key="transfer_submit").click().run()\nout["warnings"] = [w.value for w in at.warning]')
        page = run_page(before=before, after=after)
        self.assertTrue(any("Another run is active (Ingesting and summarizing papers)" in w for w in page["warnings"]))
```

- [ ] **Steps 2–4: run and fail, implement, run and pass.**

Run: `python -m pytest tests/test_pipeline_status.py tests/test_app_render.py tests/test_app_smoke.py tests/test_app_transfer_tab.py tests/test_task_ui.py tests/test_ui_modules.py tests/test_app_audit_in_two_tabs.py -v`

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/pipeline_status.py research_assistant/ui/jobs.py research_assistant/ui/common.py \
        research_assistant/tasks/ui.py research_assistant/ui/transfer_tab.py app.py \
        tests/test_pipeline_status.py tests/test_app_render.py tests/test_app_smoke.py tests/test_app_transfer_tab.py
git commit -m "feat(ui): Technique transfer tab, run as a background job, after Research chat"
```

---

### Task 17: Documentation and the CI import sweep

**Files:**
- Modify:
  - `README.md` (line 56 and its tab table; line 239 and the path table)
  - `HOW_TO_USE.md` (line 9, plus a new section before `### Tab 5 — How to use`)
  - `ARCHITECTURE.md` (a new §5.6)
  - `.github/workflows/tests.yml`

- [ ] **Step 1: Edit the docs**

**`README.md`:**
- Change line 56 to "Seven tabs. The seventh is the instructions, which nobody reads, which he has noticed."
- Add this row after Research chat: `| **Technique transfer** *(experimental)* | a research problem your corpus covers | methods from other fields that solve the same shape of bottleneck, each with a mapping table, an identifiability check, an implementation sketch, a test against the current method and a prior-art check; premises checked by the citation judge, the proposal marked as stated by no source |`
- Change line 239 to "sidebar and seven tabs".
- Add this path-table row: `| research_assistant/transfer/ | Technique transfer: technique cards, bottlenecks, the reward; scripts/build_cards.py, scripts/transfer.py, scripts/evaluate_transfer.py |`

**`HOW_TO_USE.md`:**
- Change the heading to "## 🧭 Interface Overview (The 7 Tabs)".
- Add `### Tab 5 — Technique transfer (experimental)`. It covers:
  - building cards first;
  - what to type;
  - the coverage refusal;
  - how to read the mapping table and identifiability;
  - the badges;
  - "Set aside" and "Already done";
  - that runs continue in the background.
- Rename the old Tab 5 to `### Tab 6 — How to use`.

**`ARCHITECTURE.md`:** add §5.6, "Technique transfer reads technique cards, judges premises, and is calibrated by retrodiction", with one paragraph each on spec §3, §4, §5 and §6, linking the spec.

**`tests.yml`:** add `research_assistant.transfer.engine research_assistant.transfer.mcts research_assistant.transfer.render research_assistant.transfer.store research_assistant.eval.transfer_metrics` before `app orchestrate watch`.

- [ ] **Step 2: Verify**

Run:
```bash
for m in research_assistant.transfer.engine research_assistant.transfer.mcts research_assistant.transfer.render research_assistant.transfer.store research_assistant.eval.transfer_metrics; do CITATION_LOG_FILE=0 python -c "import $m" || echo "FAIL $m"; done
```
Expected: no output.

Run: `CITATION_LOG_FILE=0 python -m pytest tests/ -q`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md HOW_TO_USE.md ARCHITECTURE.md .github/workflows/tests.yml
git commit -m "docs: the Technique transfer tab, its design, and the CI import sweep"
```

If Task 15 was skipped at its gate, drop `research_assistant.transfer.mcts` from the sweep list.
