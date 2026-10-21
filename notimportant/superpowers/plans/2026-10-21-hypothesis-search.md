# Hypothesis Search Implementation Plan

> **Superseded** by `2026-10-21-technique-transfer-design.md` / `2026-10-21-technique-transfer.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "Hypothesis search" tab, CLI and eval that propose ranked conjectures for a question the corpus covers. Each conjecture is built from premises the judge has checked. MCTS waits until an eval shows that best-of-N earns it.

**Architecture:** A new package, `research_assistant/ideation/`, works in five steps:
- It reads the summary vectors once.
- It picks the near papers (through the existing relevance gate) and bridge papers from a percentile band of the distance ranking.
- It asks the model for mechanisms, then for one proposal per (mechanism, bridge).
- It judges every premise with the citer's acceptance rule.
- It scores novelty, relevance and testability in code.

Every model, judge, embedding and search call comes in through a `Deps` object, so the tests run on fakes. The tab runs the search as a `run_jobs` background job. The CLI and the eval call the same engine.

**Tech Stack:** Python 3.10–3.12, numpy, ChromaDB 1.5.9 (summary collection), embeddinggemma-2 via Ollama, `shared.llm` (Ollama or Gemini), `judgement.judge`, Streamlit 1.65 (`run_jobs`, AppTest), unittest under pytest.

**Spec:** `notimportant/superpowers/specs/2026-10-21-hypothesis-search-design.md`

## Global Constraints

- Python ≥ 3.10, < 3.13 (`pyproject.toml`). CI runs 3.10 and 3.12, and `tests/test_python_compat.py` rejects 3.12-only f-string forms.
- No new dependencies; numpy is already pinned (`numpy~=2.2.6`).
- No module-level import of chromadb, torch or langchain in a new module. CI installs `requirements-test.txt` only, so chromadb is imported inside `bands.load_summary_index`.
- Every write goes under `config.DATA_DIR` (`IDEATION_DIR`, `data/eval/ideation/`), through `shared.atomic.atomic_write_json` or `atomic_write`.
- Every `hybrid_search` call passes `exclude_types=EXCLUDED_FROM_EVIDENCE`.
- Premise credit follows the citer's rule. A failure never earns credit, and a Contradicts verdict sets the candidate aside.

  | verdict | `span_verified` | credit |
  |---|---|---|
  | Supports | `True` | 1.0 |
  | Partially supports | `True` | 0.5 |
  | anything else | — | 0.0 |

- Agent 5 (`agents/agent5_batch_citer.py`), Agent 8 (`agents/agent8_verifier.py`) and `judgement/` are not modified.
- New tunables go in `config.py` under `# ─── Ideation (Hypothesis search tab) ───`, after the Synthesis section. Each gets a `CITATION_IDEATION_*` override and a one-line reason, written with `_env_int` or `float(os.environ.get(...))` as the file already does.
- Tests are `unittest.TestCase` classes with sentence-named tests. The model, judge, embeddings and search are fakes injected through `Deps`.
- Stage explicit paths only. Never `git add -A` or `git add .`: `.ENV` is a byte-identical copy of `.env` (it holds the Gemini key), and nothing ignores it until Task 0.
- No live-model command (a CLI run, the eval) while `pipeline_status` shows a run active. `scripts/migrate_to_v3.py` is running as this is written.
- UI copy:
  - the tab label is `Hypothesis search`;
  - its caption starts `Experimental.`;
  - a conjecture is always labelled `not stated by any source`.
- Commands below write `python -m pytest`. On the maintainer's machine that is `/home/shardul/miniconda3/envs/ml/bin/python -m pytest`.

## Review Focus

1. **A question the corpus doesn't cover.** Example: "lithium dendrites with MXenes", which has 0 papers. The search must be refused before any generation call, naming the nearest papers. Pinned in Task 5.
2. **The model cites a passage key it was never shown** (`C99`, or a key from another prompt). That premise is dropped with a warning and earns nothing. Pinned in Tasks 2 and 5.
3. **The judge raises or replies unparseably on a premise.** That premise earns 0 (never a neutral score), and the run continues. Pinned in Task 3.
4. **A search starts while another run is active** (the migration, an ingest). It is refused, with that run's stage named, in the engine, the CLI and the tab. Pinned in Tasks 5, 7 and 10.
5. **Model text with quotes, brackets, `#`, backticks or newlines in a title.** The Mermaid tree still renders. Pinned in Task 6.

---

## Before Task 0 (maintainer)

- Pause the Antigravity run that wrote the MCTS draft, so nothing else edits this checkout.
- Commit or stash your own in-progress v3 changes (`research_assistant/shared/llm.py`, `scripts/migrate_to_v3.py`) on `feat/v3-gemma-images`. Task 0 never stages them.
- Task 0 is safe while the migration runs. It touches only the draft's files and `app.py`, and the migration imports neither. The live steps in Tasks 7 and 8 wait for the migration to finish.

## File map

| File | Responsibility | Task |
|---|---|---|
| `.gitignore` | also ignore `.ENV` | 0 |
| `research_assistant/config.py` | `IDEATION_*` settings | 1, 4, 5, 6 |
| `research_assistant/ideation/__init__.py` | package docstring | 1 |
| `research_assistant/ideation/bands.py` | summary vectors, near set, percentile band, bridge ranking, novelty percentile | 1 |
| `research_assistant/ideation/models.py` | dataclasses, run ids, JSON round trip | 2 |
| `research_assistant/ideation/replies.py` | JSON reply parsing with one retry, premise key resolution | 2 |
| `research_assistant/ideation/premises.py` | premises judged with the citer's rule, cached | 3 |
| `research_assistant/ideation/scoring.py` | testability checks, set-aside reasons, composite score, ranking | 4 |
| `research_assistant/ideation/prompts.py` | templates, keyed passage formatting | 5 |
| `research_assistant/ideation/engine.py` | `Deps`, coverage and busy guards, `run_best_of_n`, `run_sampled`, `describe_bands` | 5 |
| `research_assistant/ideation/render.py` | premise badges, Mermaid tree, Markdown report | 6 |
| `research_assistant/ideation/store.py` | save and load under `IDEATION_DIR` | 6 |
| `scripts/ideate.py` | CLI, with `--show-bands` for calibration | 7 |
| `research_assistant/eval/cases/ideation_queries.jsonl` | six questions the corpus covers | 8 |
| `research_assistant/eval/ideation_metrics.py` | pure scoring of eval records, blind sheet, ratings | 8 |
| `scripts/evaluate_ideation.py` | eval harness | 8 |
| `research_assistant/shared/pipeline_status.py` | the `ideate` stage | 9 |
| `research_assistant/ui/jobs.py` | adopt `ideation` results into their own session keys | 9 |
| `research_assistant/ui/common.py` | `_render_mermaid`, moved from `tasks/ui.py` | 10 |
| `research_assistant/tasks/ui.py` | re-export the moved helper under its old name | 10 |
| `research_assistant/ui/ideation_tab.py` | the tab | 10 |
| `app.py` | seventh tab, docstring | 10 |
| `README.md`, `HOW_TO_USE.md`, `ARCHITECTURE.md`, `.github/workflows/tests.yml` | docs, CI import sweep | 11 |

Test files:
- one new file per new module (`tests/test_ideation_*.py`, `tests/test_ideate_cli.py`, `tests/test_evaluate_ideation.py`, `tests/test_app_ideation_tab.py`);
- a shared fixture module, `tests/ideation_fixtures.py`;
- edits to `tests/test_app_smoke.py`, `tests/test_app_render.py` and `tests/test_pipeline_status.py`.

---

### Task 0: Keep the MCTS draft on a side branch and start clean

**Files:**
- Modify: `.gitignore` (after line 25, `.env.*`)
- Move to branch `wip/mcts-draft`:
  - `research_assistant/brainstorm/__init__.py`, `mcts.py`, `prompts.py`
  - `research_assistant/ui/mcts_tab.py`
  - `scripts/mcts_hypothesis.py`
  - `tests/test_mcts.py`
  - `output/mcts_test.md`
  - `app.py` (the draft's tab edit)

**Interfaces:**
- Produces: branch `feat/hypothesis-search`, cut from `feat/v3-gemma-images`, with none of the draft in its working tree.
- Phase 2 reads `wip/mcts-draft` for the draft's UCT and backpropagation code.

- [ ] **Step 1: Confirm the starting state**

Run: `git status --short`
Expected:
- the draft paths above as `??`, plus ` M app.py`;
- `?? .ENV`;
- nothing else except your own v3 files.

Run: `python -m pytest tests/test_app_smoke.py -q`
Expected: FAIL on `test_the_six_tabs_are_there_in_order`, because the draft's tab is in `app.py`.

- [ ] **Step 2: Commit the draft to its own branch (explicit paths only)**

```bash
git switch -c wip/mcts-draft
git add research_assistant/brainstorm/__init__.py research_assistant/brainstorm/mcts.py \
        research_assistant/brainstorm/prompts.py research_assistant/ui/mcts_tab.py \
        scripts/mcts_hypothesis.py tests/test_mcts.py output/mcts_test.md app.py
git commit -m "wip: MCTS hypothesis-tree draft from the Antigravity plan (superseded; kept for Phase 2)"
```

- [ ] **Step 3: Start the feature branch and remove what git leaves behind**

```bash
git switch -c feat/hypothesis-search feat/v3-gemma-images
rm -rf research_assistant/brainstorm output
```

Git leaves `brainstorm/` behind because it still holds an ignored `__pycache__/`. Left in place, the directory would act as a namespace package.

- [ ] **Step 4: Ignore `.ENV`**

Add `.ENV` under the `.env.*` line, with the comment `# a case-variant copy of .env: this filesystem is case-sensitive, so .env does not cover it`.

Run: `git check-ignore -v .ENV`
Expected: one line ending in `.ENV	.ENV`, naming `.gitignore`.

- [ ] **Step 5: Verify**

Run: `git status --short`
Expected: ` M .gitignore`, plus your own v3 files if you did not commit them.

Run: `python -m pytest tests/test_app_smoke.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore .ENV, a case-variant copy of .env"
```

---

### Task 1: Paper-level bands over the summary vectors

**Files:**
- Create: `research_assistant/ideation/__init__.py`, `research_assistant/ideation/bands.py`, `tests/ideation_fixtures.py`
- Modify: `research_assistant/config.py`
- Test: `tests/test_ideation_bands.py`

**Interfaces:**

Produces, in `bands.py`:
- `@dataclass SummaryIndex(documents: list[str], citations: list[str], summaries: list[str], vectors: np.ndarray, nn_distances: np.ndarray)`
  - `vectors`: rows L2-normalised.
  - `nn_distances[i]`: cosine distance from paper i to its nearest other paper; `[1.0]` for a one-paper corpus.
- `build_index(documents, citations, summaries, vectors) -> SummaryIndex`
- `load_summary_index() -> SummaryIndex`
  - Imports chromadb inside the function.
  - Reads `PersistentClient(VECTORDB_PATH).get_collection(SUMMARY_COLLECTION_NAME)`, runs `check_embedder(col)`, then `col.get(include=["embeddings", "metadatas", "documents"])`.
  - Document and citation come from `metadata["document"]` and `metadata["citation_source"]`.
  - Cached per process on `(VECTORDB_PATH, SUMMARY_COLLECTION_NAME, col.count())`.
  - With no collection: `RuntimeError("No summary collection yet — ingest papers first.")`.
- `distances_to(index, vector) -> np.ndarray`: `1 − V·v̂`.
- `rank_percentiles(distances) -> np.ndarray`: rank by ascending distance, divided by (n − 1); `[0.0]` when n = 1.
- `nearest(distances, k) -> list[int]`: nearest first.
- `band(index, distances, lo_pct, hi_pct, exclude_documents=frozenset()) -> list[int]`: papers with `lo_pct ≤ percentile ≤ hi_pct`, minus excluded documents, nearest first.
- `rank_bridges(index, candidates: list[int], vector, k, exclude_documents=frozenset()) -> list[int]`: the k candidates most similar to `vector`; ties keep candidate order.
- `novelty_percentile(index, distance) -> float`: the share of `nn_distances` strictly below `distance`.
- `as_ranked(index, idx, distances) -> list[dict]`: `[{"document", "citation", "summary", "score": 1 − distance}]`, the shape `retrieve.gate_documents` reads.

Produces, in `tests/ideation_fixtures.py`:
- `at(deg) -> np.ndarray`: a unit vector at `deg`.
- `five_paper_index() -> SummaryIndex`: papers `d0…d4` at 0°, 10°, 30°, 60°, 120°, deliberately unnormalised (×3); citations `Paper 0…4`.

Config:
- `IDEATION_BAND_LO_PCT = float(os.environ.get("CITATION_IDEATION_BAND_LO_PCT", "0.10"))`
- `IDEATION_BAND_HI_PCT = float(os.environ.get("CITATION_IDEATION_BAND_HI_PCT", "0.50"))`
- Comment: these are percentiles of the query's distance ranking, because cosine distances are model-specific; calibrate with `scripts/ideate.py --show-bands`.

- [ ] **Step 1: Write the failing tests** (imports: `math`, `unittest`, `numpy as np`, `bands`, `five_paper_index`, `at`)

```python
class TestBands(unittest.TestCase):
    def test_rows_are_normalised_and_nearest_neighbour_distances_are_exact(self):
        ix = five_paper_index()
        np.testing.assert_allclose(np.linalg.norm(ix.vectors, axis=1), 1.0)
        np.testing.assert_allclose(ix.nn_distances, [1 - math.cos(math.radians(d)) for d in (10, 10, 20, 30, 60)], atol=1e-9)

    def test_nearest_returns_the_k_closest_papers_closest_first(self):
        self.assertEqual(bands.nearest(bands.distances_to(five_paper_index(), at(0)), 2), [0, 1])

    def test_band_keeps_the_middle_percentiles_and_drops_excluded_documents(self):
        ix = five_paper_index(); d = bands.distances_to(ix, at(0))
        self.assertEqual(bands.band(ix, d, 0.25, 0.75), [1, 2, 3])
        self.assertEqual(bands.band(ix, d, 0.25, 0.75, exclude_documents={"d0", "d1"}), [2, 3])

    def test_rank_bridges_orders_candidates_by_similarity_and_skips_used_documents(self):
        ix = five_paper_index()
        self.assertEqual(bands.rank_bridges(ix, [2, 3], at(60), 1), [3])
        self.assertEqual(bands.rank_bridges(ix, [2, 3], at(60), 2, exclude_documents={"d3"}), [2])

    def test_novelty_percentile_is_the_share_of_papers_closer_to_their_neighbour(self):
        self.assertAlmostEqual(bands.novelty_percentile(five_paper_index(), 0.1), 0.6)

    def test_as_ranked_has_the_shape_the_gate_reads(self):
        ix = five_paper_index(); row = bands.as_ranked(ix, [1], bands.distances_to(ix, at(0)))[0]
        self.assertEqual(set(row), {"document", "citation", "summary", "score"})
        self.assertAlmostEqual(row["score"], math.cos(math.radians(10)))

    def test_a_one_paper_corpus_has_no_band(self):
        ix = bands.build_index(["d0"], ["P"], ["s"], np.array([[1.0, 0.0]]))
        self.assertEqual(list(ix.nn_distances), [1.0])
        self.assertEqual(bands.band(ix, bands.distances_to(ix, at(0)), 0.1, 0.5), [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ideation_bands.py -v`
Expected: FAIL with `No module named 'research_assistant.ideation'`.

- [ ] **Step 3: Implement `bands.py`, the package docstring and the config lines**

- Package docstring: `"""Hypothesis search: grounded conjectures from the corpus (the Hypothesis search tab)."""`
- `nn_distances` is `1 −` the row maximum of `V @ V.T` with the diagonal set to `-inf`, special-cased to `[1.0]` when n = 1.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ideation_bands.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ideation/__init__.py research_assistant/ideation/bands.py research_assistant/config.py \
        tests/ideation_fixtures.py tests/test_ideation_bands.py
git commit -m "feat(ideation): paper-level percentile bands over the summary vectors"
```

---

### Task 2: Data model and model-reply parsing

**Files:**
- Create: `research_assistant/ideation/models.py`, `research_assistant/ideation/replies.py`
- Modify: `tests/ideation_fixtures.py`
- Test: `tests/test_ideation_models.py`, `tests/test_ideation_replies.py`

**Interfaces:**

Produces, in `models.py` (all `@dataclass`):
- `Passage(key: str, chunk_index: int, document: str, citation: str, section: str, text: str, role: str)`
  - `key` is `"C<n>"`, unique within a run.
  - `role` is `"near"` or `"bridge"`.
- `Premise(claim: str, key: str, document: str = "", chunk_index: int | None = None, judgement: str | None = None, span_verified: bool | None = None, supporting_span: str | None = None, reason: str | None = None, error: str | None = None)`, with properties:
  - `accepted`: Supports and `span_verified is True`
  - `partial`: Partially supports and `span_verified is True`
  - `credit`: 1.0 / 0.5 / 0.0
  - `contradicts`: `judgement == "Contradicts"`
- `Mechanism(id: str, statement: str, abstraction: str, premises: list[Premise])`, with ids `M1…`.
- `Protocol(system: str = "", observable: str = "", prediction: str = "", null_result: str = "", method: str = "", method_key: str = "")`
- `Scores(grounded: float, novelty: float, relevance: float, testable: float, composite: float, nearest_document: str, nearest_similarity: float, checks: dict[str, bool], set_aside: str | None = None)`
- `Candidate(id: str, title: str, conjecture: str, premises: list[Premise], protocol: Protocol, mechanism_id: str | None = None, bridge: dict | None = None, scores: Scores | None = None, warnings: list[str] = field(default_factory=list))`
  - ids are `M1.B2` for best_of_n and `S3` for sampled.
  - `bridge` is `{"document", "citation", "distance", "percentile", "similarity"}`.
- `IdeationResult(run_id: str, query: str, arm: str, preset: str, created_at: str, model: str, judge_model: str, settings: dict, near: list[dict], passages: dict[str, Passage], mechanisms: list[Mechanism], candidates: list[Candidate], calls: dict[str, int], seconds: float, warnings: list[str])`
  - `arm` is `"best_of_n"` or `"sampled"`.
  - `near` is `[{"document", "citation", "similarity"}]` after the gate.
  - `calls` has the keys `"llm"`, `"judge"` and `"embed"`.
- `new_run_id(query: str) -> str`: `f"{datetime.now():%Y%m%d-%H%M%S}-{sha1(query)[:8]}"`.
- `result_to_dict(result) -> dict` (JSON-safe) and `result_from_dict(data) -> IdeationResult`, which rebuilds the nested dataclasses explicitly.

Produces, in `replies.py`:
- `JSON_NUDGE = "Your previous reply was not the JSON object asked for. Reply again with only that JSON object: no prose, no code fence."`
- `class ReplyParseError(ValueError)`, with `.raw`.
- `parse_reply(raw: str, required: tuple[str, ...]) -> dict`
  - Strips a ```` ``` ```` or ```` ```json ```` fence, then `json.loads`; on failure, tries the outermost `{…}`.
  - The result must be a dict holding every `required` key, else `ReplyParseError`.
- `ask_json(call: Callable[[list[dict]], str], messages: list[dict], required: tuple[str, ...]) -> dict`
  - After a first `ReplyParseError`, calls once more with `messages + [{"role": "assistant", "content": bad}, {"role": "user", "content": JSON_NUDGE}]`.
  - A second failure raises.
- `normalise_key(key) -> str`: `"[c7]"` and `" C7 "` both become `"C7"`.
- `resolve_premises(items: list, passages: dict[str, Passage], limit: int) -> tuple[list[Premise], list[str]]`
  - `items` are `{"claim", "key"}`; `document` and `chunk_index` are copied from the passage.
  - Each of these is dropped with a warning: an unknown key (`"premise cites C99, which was not among the passages shown"`), an empty claim, a duplicate (claim, key), and anything past `limit`.

Produces, in `tests/ideation_fixtures.py`: `sample_result() -> IdeationResult`, fully deterministic:
- run id `20261021-120000-abcdef12`; query `graphene disorder from conductance`; arm `best_of_n`;
- passages `C1` (near, `a.pdf`, "Paper A") and `C2` (bridge, `b.pdf`, "Paper B");
- `M1`, with one accepted premise on `C1`;
- candidate `M1.B1`, titled "Disorder maps from noise spectra": accepted premise on `C2`, a full protocol with `method_key="C2"`, composite 0.71;
- candidate `M1.B2`, titled "Second idea": a Contradicts premise on `C2` (a different claim from `M1.B1`'s), composite 0.0, `set_aside="contradicted"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ideation_models.py
class TestPremiseCredit(unittest.TestCase):
    def test_credit_follows_the_citers_acceptance_rule(self):
        P = lambda j, s: models.Premise("c", "C1", judgement=j, span_verified=s)
        self.assertEqual(P("Supports", True).credit, 1.0); self.assertTrue(P("Supports", True).accepted)
        self.assertEqual(P("Partially supports", True).credit, 0.5); self.assertTrue(P("Partially supports", True).partial)
        for j, s in [("Supports", None), ("Supports", False), ("Partially supports", None),
                     ("Unclear / insufficient evidence", True), ("Does not support", True), (None, None)]:
            self.assertEqual(P(j, s).credit, 0.0, (j, s))
        self.assertTrue(P("Contradicts", True).contradicts); self.assertEqual(P("Contradicts", True).credit, 0.0)

class TestRoundTrip(unittest.TestCase):
    def test_a_result_survives_json(self):
        r = sample_result()
        self.assertEqual(models.result_from_dict(json.loads(json.dumps(models.result_to_dict(r)))), r)

    def test_run_ids_carry_the_time_and_differ_by_query(self):
        a, b = models.new_run_id("q one"), models.new_run_id("q two")
        self.assertRegex(a, r"^\d{8}-\d{6}-[0-9a-f]{8}$"); self.assertNotEqual(a[-8:], b[-8:])
```

```python
# tests/test_ideation_replies.py
PASSAGES = {"C1": models.Passage("C1", 4, "a.pdf", "Paper A", "results", "t", "near"),
            "C2": models.Passage("C2", 9, "b.pdf", "Paper B", "methods", "t", "bridge")}

class TestParseReply(unittest.TestCase):
    def test_fenced_bare_and_prose_wrapped_replies_parse(self):
        for raw in ['```json\n{"a": 1}\n```', '{"a": 1}', 'Here it is: {"a": 1} hope that helps']:
            self.assertEqual(replies.parse_reply(raw, ("a",)), {"a": 1})

    def test_a_reply_missing_a_required_key_is_rejected(self):
        with self.assertRaises(replies.ReplyParseError):
            replies.parse_reply('{"a": 1}', ("a", "b"))

class TestAskJson(unittest.TestCase):
    def test_one_retry_with_the_nudge_then_give_up(self):
        sent = []
        def call(msgs): sent.append(msgs); return "no json here"
        with self.assertRaises(replies.ReplyParseError):
            replies.ask_json(call, [{"role": "user", "content": "q"}], ("a",))
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[1][-2:], [{"role": "assistant", "content": "no json here"},
                                        {"role": "user", "content": replies.JSON_NUDGE}])

    def test_the_second_reply_is_used_when_it_parses(self):
        out = iter(["nope", '{"a": 2}'])
        self.assertEqual(replies.ask_json(lambda m: next(out), [], ("a",)), {"a": 2})

class TestResolvePremises(unittest.TestCase):
    def test_a_premise_citing_a_passage_that_was_never_shown_is_dropped_with_a_warning(self):
        got, warnings = replies.resolve_premises([{"claim": "X holds.", "key": "C99"}], PASSAGES, limit=4)
        self.assertEqual(got, []); self.assertIn("C99", warnings[0])

    def test_keys_are_normalised_and_carry_the_passages_provenance(self):
        got, _ = replies.resolve_premises([{"claim": "Y holds.", "key": "[c2]"}], PASSAGES, limit=4)
        self.assertEqual((got[0].key, got[0].document, got[0].chunk_index), ("C2", "b.pdf", 9))

    def test_empty_duplicate_and_over_the_cap_premises_are_dropped(self):
        items = [{"claim": "", "key": "C1"}, {"claim": "A.", "key": "C1"},
                 {"claim": "A.", "key": "C1"}, {"claim": "B.", "key": "C2"}]
        got, warnings = replies.resolve_premises(items, PASSAGES, limit=1)
        self.assertEqual([p.claim for p in got], ["A."]); self.assertEqual(len(warnings), 3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ideation_models.py tests/test_ideation_replies.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `models.py`, `replies.py` and `sample_result()`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ideation_models.py tests/test_ideation_replies.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ideation/models.py research_assistant/ideation/replies.py tests/ideation_fixtures.py \
        tests/test_ideation_models.py tests/test_ideation_replies.py
git commit -m "feat(ideation): premise/conjecture data model and strict reply parsing"
```

---

### Task 3: Premise checking with the citer's rule

**Files:**
- Create: `research_assistant/ideation/premises.py`
- Test: `tests/test_ideation_premises.py`

**Interfaces:**
- Consumes:
  - `models.Premise`
  - `agents.agent8_verifier.assemble_evidence(hits, max_chars)`
  - `shared.search.expand_neighbours(results, texts, metadatas, window)`
  - config `JUDGEMENT_NEIGHBOUR_WINDOW` and `JUDGEMENT_EVIDENCE_MAX_CHARS`
- Produces: `PremiseChecker(judge: Callable[[str, str], dict], texts: list[str], metadatas: list[dict])`, with:
  - `.check(premise) -> Premise`: fills `judgement`, `span_verified`, `supporting_span` and `reason` from the verdict. When `judge` raises, it fills `error` (`"<Type>: <message>"`) instead. It never raises itself.
  - `.check_all(premises) -> list[Premise]`
  - `.calls: int`: the number of judge invocations.
  - A cache keyed on `(" ".join(claim.lower().split()), chunk_index)`. Failures are cached too, which keeps the cost bounded.
- Evidence, as `_cite_by_judge` builds it:
  1. `hit = {"chunk_index": i, "text": texts[i], "metadata": metadatas[i]}`
  2. `expand_neighbours([hit], texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)`
  3. `assemble_evidence([hit], JUDGEMENT_EVIDENCE_MAX_CHARS)`
  4. `judge(claim, evidence)`

- [ ] **Step 1: Write the failing tests**

```python
TEXTS = ["the sentence before", "the cited chunk", "a figure", "the sentence after"]
METAS = [{"document": "a.pdf", "type": t} for t in ("text_chunk", "text_chunk", "figure_image", "text_chunk")]
SUPPORTS = {"judgement": "Supports", "span_verified": True, "supporting_span": "the cited chunk", "reason": "r"}

class TestPremiseChecker(unittest.TestCase):
    def test_the_premise_is_judged_against_its_chunk_and_its_text_neighbour(self):
        seen = []
        checker = premises.PremiseChecker(lambda c, e: (seen.append(e), SUPPORTS)[1], TEXTS, METAS)
        p = checker.check(models.Premise("A claim.", "C1", "a.pdf", 1))
        self.assertIn("the cited chunk", seen[0]); self.assertIn("the sentence before", seen[0])
        self.assertNotIn("a figure", seen[0]); self.assertTrue(p.accepted)

    def test_the_same_claim_on_the_same_chunk_is_judged_once(self):
        checker = premises.PremiseChecker(lambda c, e: SUPPORTS, TEXTS, METAS)
        a, b = checker.check_all([models.Premise("A  claim.", "C1", "a.pdf", 1), models.Premise("a claim.", "C1", "a.pdf", 1)])
        self.assertEqual(checker.calls, 1); self.assertTrue(a.accepted and b.accepted)

    def test_a_judge_failure_earns_nothing_and_the_next_premise_is_still_judged(self):
        def judge(claim, evidence):
            if claim == "bad":
                raise JudgementParseError("Reply was not valid JSON", raw="{")
            return SUPPORTS
        checker = premises.PremiseChecker(judge, TEXTS, METAS)
        bad, good = checker.check_all([models.Premise("bad", "C1", "a.pdf", 1), models.Premise("good", "C1", "a.pdf", 1)])
        self.assertEqual(bad.credit, 0.0); self.assertIn("JudgementParseError", bad.error); self.assertTrue(good.accepted)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ideation_premises.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `premises.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ideation_premises.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ideation/premises.py tests/test_ideation_premises.py
git commit -m "feat(ideation): judge premises the way the citer judges a candidate"
```

---

### Task 4: Scoring

**Files:**
- Create: `research_assistant/ideation/scoring.py`
- Modify: `research_assistant/config.py`
- Test: `tests/test_ideation_scoring.py`

**Interfaces:**

Consumes: `models.*`, `bands.distances_to`, `bands.novelty_percentile`.

Produces, in `scoring.py`:
- `TESTABILITY_CHECKS = ("system", "observable", "prediction", "null_result", "method")`
- `testability_checks(protocol: Protocol, accepted_keys: set[str]) -> dict[str, bool]`
  - `system`: at least 3 words.
  - `observable`: at least 3 words.
  - `prediction`: at least 5 words and matches `r"\b(increase|decrease|higher|lower|larger|smaller|rise|fall|grow|drop|enhance|suppress|reduce|raise|shift|faster|slower|exceed|more|less)\w*|\d"` (case-insensitive).
  - `null_result`: at least 5 words, and it differs from the prediction once both are lowercased with punctuation and repeated spaces removed.
  - `method`: `normalise_key(method_key)` is in `accepted_keys`.
- `unique_premises(premises) -> list[Premise]`: keeps the first of each (normalised claim, chunk_index).
- `grounded_score(premises) -> float`: the mean credit; 0.0 when there are none.
- `set_aside_reason(premises, *, relevance: float, relevance_floor: float, bridge_document: str | None, min_accepted: int) -> str | None`. Returns the first that applies:
  1. `"contradicted"`
  2. `"ungrounded"`: accepted + partial premises < `min_accepted`
  3. `"bridge_ungrounded"`: there is a bridge, and no accepted or partial premise comes from its document
  4. `"off_topic"`: relevance < floor
- `composite(grounded, novelty, testable, weights: dict[str, float]) -> float`: weights normalised to sum to 1.
- `score_candidate(candidate, mechanism: Mechanism | None, *, conjecture_vec, query_vec, index, relevance_floor: float, weights: dict, min_accepted: int) -> Scores`
  - Premises: `unique_premises((mechanism.premises if mechanism else []) + candidate.premises)`.
  - `bridge_document` comes from `candidate.bridge`.
  - Composite is 0.0 when the candidate is set aside.
- `rank(candidates) -> list[Candidate]`: composite descending, set-aside candidates last, stable.

Config:
- `IDEATION_PRESETS = {"balanced": {"grounded": 0.4, "novelty": 0.3, "testable": 0.3}, "grounded": {"grounded": 0.6, "novelty": 0.15, "testable": 0.25}, "radical": {"grounded": 0.3, "novelty": 0.5, "testable": 0.2}}`
- `IDEATION_RELEVANCE_RATIO = float(os.environ.get("CITATION_IDEATION_RELEVANCE_RATIO", "0.9"))`
- `IDEATION_MIN_ACCEPTED_PREMISES = _env_int("CITATION_IDEATION_MIN_ACCEPTED_PREMISES", 2)`

- [ ] **Step 1: Write the failing tests**

```python
FULL = models.Protocol(system="two-dimensional MoS2 flakes on SiO2", observable="sheet conductance at 300 K",
                       prediction="conductance will increase by an order of magnitude after amorphization",
                       null_result="no change in sheet conductance beyond the run-to-run scatter",
                       method="Ga+ FIB amorphization", method_key="C3")
def _p(j, key="C1", doc="a.pdf", idx=1):
    return models.Premise(f"{j} {key}", key, doc, idx, judgement=j, span_verified=True)

class TestTestability(unittest.TestCase):
    def test_a_complete_protocol_passes_all_five_checks(self):
        self.assertEqual(scoring.testability_checks(FULL, {"C3"}), dict.fromkeys(scoring.TESTABILITY_CHECKS, True))

    def test_each_failure_fails_only_its_own_check(self):
        cases = {"system": replace(FULL, system="MoS2"), "observable": replace(FULL, observable=""),
                 "prediction": replace(FULL, prediction="the conductance will be quite different then"),
                 "null_result": replace(FULL, null_result=FULL.prediction.upper() + "!")}
        for name, proto in cases.items():
            failed = [k for k, ok in scoring.testability_checks(proto, {"C3"}).items() if not ok]
            self.assertEqual(failed, [name], name)
        self.assertFalse(scoring.testability_checks(FULL, {"C1"})["method"])

class TestSetAside(unittest.TestCase):
    KW = dict(relevance=0.9, relevance_floor=0.1, min_accepted=2)
    def test_a_contradicted_premise_sets_the_candidate_aside_whatever_else_it_scores(self):
        ps = [_p("Supports"), _p("Supports", "C2", "b.pdf", 2), _p("Contradicts", "C3")]
        self.assertEqual(scoring.set_aside_reason(ps, bridge_document="b.pdf", **self.KW), "contradicted")
    def test_too_few_accepted_premises_is_ungrounded(self):
        ps = [_p("Supports"), _p("Unclear / insufficient evidence", "C2", "b.pdf", 2)]
        self.assertEqual(scoring.set_aside_reason(ps, bridge_document=None, **self.KW), "ungrounded")
    def test_a_bridge_with_no_accepted_premise_from_its_paper_is_set_aside(self):
        ps = [_p("Supports"), _p("Partially supports", "C2", "a.pdf", 2)]
        self.assertEqual(scoring.set_aside_reason(ps, bridge_document="b.pdf", **self.KW), "bridge_ungrounded")
    def test_a_conjecture_below_the_relevance_floor_is_off_topic(self):
        ps = [_p("Supports"), _p("Supports", "C2", "b.pdf", 2)]
        self.assertEqual(scoring.set_aside_reason(ps, relevance=0.3, relevance_floor=0.5, bridge_document="b.pdf", min_accepted=2), "off_topic")
        self.assertIsNone(scoring.set_aside_reason(ps, relevance=0.6, relevance_floor=0.5, bridge_document="b.pdf", min_accepted=2))

class TestScores(unittest.TestCase):
    def test_grounded_is_the_mean_credit_of_inherited_and_own_premises_each_counted_once(self):
        mech = [_p("Supports"), _p("Does not support", "C2", idx=2)]
        own = [_p("Partially supports", "C3", "b.pdf", 3), _p("Supports")]          # the last repeats mech[0]
        self.assertAlmostEqual(scoring.grounded_score(scoring.unique_premises(mech + own)), 0.5)

    def test_composite_normalises_the_weights(self):
        self.assertAlmostEqual(scoring.composite(1.0, 0.0, 0.0, {"grounded": 2, "novelty": 1, "testable": 1}), 0.5)

    def test_novelty_and_relevance_come_from_the_embeddings(self):
        cand = models.Candidate("S1", "t", "c", [_p("Supports"), _p("Supports", "C2", "b.pdf", 2)], FULL)
        s = scoring.score_candidate(cand, None, conjecture_vec=at(30), query_vec=at(0), index=five_paper_index(),
                                    relevance_floor=0.5, weights=config.IDEATION_PRESETS["balanced"], min_accepted=2)
        self.assertEqual(s.nearest_document, "d2"); self.assertAlmostEqual(s.nearest_similarity, 1.0)
        self.assertEqual(s.novelty, 0.0); self.assertAlmostEqual(s.relevance, math.cos(math.radians(30)))
        self.assertIsNone(s.set_aside)

    def test_rank_puts_set_aside_candidates_last(self):
        mk = lambda i, c, aside=None: models.Candidate(i, i, "c", [], models.Protocol(),
                                                       scores=models.Scores(0, 0, 0, 0, c, "", 0, {}, aside))
        ranked = scoring.rank([mk("a", 0.0, "ungrounded"), mk("b", 0.5), mk("c", 0.7), mk("d", 0.5)])
        self.assertEqual([c.id for c in ranked], ["c", "b", "d", "a"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ideation_scoring.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `scoring.py` and the config lines**

- `nearest_similarity` and `novelty` come from `bands.distances_to(index, conjecture_vec)`.
- `relevance` is `float(np.dot(q̂, ĉ))`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ideation_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ideation/scoring.py research_assistant/config.py tests/test_ideation_scoring.py
git commit -m "feat(ideation): score premises, novelty, relevance and testability in code"
```

---

### Task 5: Prompts and the engine

**Files:**
- Create: `research_assistant/ideation/prompts.py`, `research_assistant/ideation/engine.py`
- Modify: `research_assistant/config.py`, `tests/ideation_fixtures.py`
- Test: `tests/test_ideation_engine.py`

**Interfaces:**

Consumes:
- Tasks 1–4
- `retrieve.gate_documents`, `search.hybrid_search`, `db.load_search_resources`
- `pipeline_status.get_status`, `pipeline_status.is_cancel_requested`, `pipeline_status.PipelineCancelledError`
- `shared.llm.chat`, `get_embeddings`
- `judgement.judge.judge`, `shared.retry.retry`
- config `DOC_SELECT_K`, `EXCLUDED_FROM_EVIDENCE`, `LLM_MODEL`, `JUDGEMENT_MODEL`

Produces, in `prompts.py`:
- `IDEATION_SYSTEM`
- `MECHANISMS_USER`, with fields `{query}`, `{passages}`, `{n}`
- `PROPOSAL_USER`, with fields `{query}`, `{mechanism}`, `{mechanism_premises}`, `{passages}`, `{bridge_citation}`
- `SAMPLED_USER`, with fields `{query}`, `{passages}`
- `format_passages(passages: list[Passage], max_chars: int) -> str`
  - Each block is `[C7] (Paper B · methods) text`. The text is cut from the tail at `max_chars`.
  - Mechanism premises render as `- <claim> (C2)`: parentheses, never brackets. Only passage blocks carry `[C…]`.
  - `PROPOSAL_USER` lists the bridge passages last.

Rules the templates carry, word for word (each template carries those that apply):
1. "A premise is one sentence that the passage it cites states. Cite exactly one key from the passages above, written like C7."
2. "The conjecture is yours. No passage states it, so it carries no key."
3. Mechanisms only: "Give {n} mechanisms, each with 2 or 3 premises. The abstraction restates the mechanism with no material, system, organism or field named, so that a paper from another field could match it."
4. Proposals only: "Give 2 to 4 premises. At least one must cite a passage from {bridge_citation}. The conjecture carries that paper's finding over to the research question."
5. Proposals and sampled: "The protocol's prediction states a direction or a magnitude; null_result names the outcome that would refute the conjecture; method_key is the key of the premise that describes the method."
6. "Reply with the JSON object only."

Reply shapes:
- Mechanisms: `{"mechanisms": [{"statement", "abstraction", "premises": [{"claim", "key"}]}]}`
- Proposal and sampled: `{"title", "conjecture", "premises": [{"claim", "key"}], "protocol": {"system", "observable", "prediction", "null_result", "method", "method_key"}}`

Produces, in `engine.py`:
- `class CorpusCoverageError(RuntimeError)`, constructed as `CorpusCoverageError(query: str, kept: list[dict], shortlist: list[dict])`, with `.kept` and `.shortlist`. Its message names the shortlisted citations and says to build a corpus in the Research idea tab.
- `engine` imports `hybrid_search` by name and `pipeline_status` as a module; the tests patch both there.
- Warnings from resolving a proposal's premises go in `candidate.warnings`. Every other warning goes in `result.warnings`.
- `class BusyError(RuntimeError)` and `ensure_idle() -> None`. When `pipeline_status.get_status()["active"]`, it raises `BusyError(f"Another run is active: {stage_label}. Wait for it to finish.")`.
- `@dataclass Deps`:
  ```python
  chat: Callable[[list[dict], float], str]
  judge: Callable[[str, str], dict]
  embed_query: Callable[[str], list[float]]
  embed_documents: Callable[[list[str]], list[list[float]]]
  search: Callable[[str, int, set], list[dict]]
  gate: Callable[[str, list[dict]], list[dict]]
  texts: list[str]
  metadatas: list[dict]
  index: SummaryIndex
  cancelled: Callable[[], bool] = lambda: False
  progress: Callable[[int, int, str], None] = lambda done, total, label: None
  ```
- `make_search(resources) -> Callable[[str, int, set], list[dict]]`: calls `hybrid_search(q, collection, bm25, texts, metadatas, top_k=k, doc_filter=set(docs), exclude_types=EXCLUDED_FROM_EVIDENCE)`.
- `live_deps(search_resources=None) -> Deps`
  - `chat` passes `options=IDEATION_OLLAMA_OPTIONS`.
  - `judge` is `retry(max_retries=2, backoff=1.0)(judge)`.
  - `gate` is `retrieve.gate_documents`.
  - `index` is `bands.load_summary_index()`.
  - `cancelled` is `pipeline_status.is_cancel_requested`.
- `run_best_of_n(query: str, deps: Deps, *, mechanisms: int = IDEATION_MECHANISMS, bridges: int = IDEATION_BRIDGES, preset: str = "balanced") -> IdeationResult`
- `run_sampled(query: str, deps: Deps, *, n: int, preset: str = "balanced") -> IdeationResult`
- `describe_bands(query: str, deps: Deps, limit: int = 10) -> dict`: returns `{"kept", "rejected", "relevance_floor", "band_size", "band"}`.
  - `kept` and `rejected` are `as_ranked` rows.
  - `band` holds the first `limit` band papers, nearest first, each `{document, citation, percentile, similarity}`.
  - It makes embedding and gate calls only.
- `estimate_calls(mechanisms: int, bridges: int) -> dict`: `{"llm": 2 + m*b, "judge_low": 2*m + 2*m*b, "judge_high": 3*m + 4*m*b}`.

Config (each env name is `CITATION_` plus the setting name):
- `IDEATION_MECHANISMS = 3`
- `IDEATION_BRIDGES = 2`
- `IDEATION_MIN_NEAR_PAPERS = 3`
- `IDEATION_NEAR_PASSAGES = 8`
- `IDEATION_BRIDGE_PASSAGES = 3`
- `IDEATION_PASSAGE_MAX_CHARS = 7000`
- `IDEATION_TEMPERATURE = 0.7`
- `IDEATION_MECHANISM_TEMPERATURE = 0.3`
- `IDEATION_OLLAMA_OPTIONS = {"num_ctx": _env_int("CITATION_IDEATION_NUM_CTX", 8192)}`

`run_best_of_n` does the following, in the order the tests assume:

1. **Coverage.** `q = embed_query(query)`; `d = distances_to(index, q)`; `shortlist = as_ranked(index, nearest(d, DOC_SELECT_K), d)`; `kept = gate(query, shortlist)`. Fewer than `IDEATION_MIN_NEAR_PAPERS` kept raises `CorpusCoverageError`, and nothing else is called.
2. **Near passages.** `search(query, IDEATION_NEAR_PASSAGES, {kept documents})` gives Passages keyed `C1…` in hit order, with role `"near"`, deduplicated by chunk index.
3. **Mechanisms.**
   - Call `ask_json` with `MECHANISMS_USER` at `IDEATION_MECHANISM_TEMPERATURE`.
   - Each item becomes a `Mechanism` (`M1…`), with premises from `resolve_premises(limit=3)` against the passages shown in that prompt.
   - Drop a mechanism with no premise or an empty abstraction, with a warning.
   - If the reply never parses, return the result with no candidates and a warning.
   - Judge every mechanism premise.
4. **Bridges.**
   - `pool = band(index, d, IDEATION_BAND_LO_PCT, IDEATION_BAND_HI_PCT, exclude_documents={kept documents})`.
   - For each mechanism in order: `picks = rank_bridges(index, pool, embed_query(abstraction), bridges, exclude_documents=used)`, then add their documents to `used`.
   - Fewer than `bridges` picks gives the warning `band held {k} unused papers for M{i}`.
5. **Proposals.** For each (mechanism, pick), in order:
   - Check `cancelled()` first. If it is set, raise `PipelineCancelledError("Hypothesis search stopped by user")`.
   - Bridge passages: `search(abstraction, IDEATION_BRIDGE_PASSAGES, {doc})`, with role `"bridge"`; keys continue the run's numbering.
   - Call `ask_json` with `PROPOSAL_USER` at `IDEATION_TEMPERATURE`. The prompt shows the mechanism's premise passages, then the bridge passages.
   - If the reply fails, skip the proposal with the warning `"{id}: the model's reply was not usable JSON; skipped"`.
   - Otherwise, resolve the premises with `resolve_premises(limit=4)` against **that prompt's passages only**, and judge them.
   - Embed the conjecture: `embed_documents([conjecture])[0]`.
   - `relevance_floor = IDEATION_RELEVANCE_RATIO × min(similarity of kept papers)`.
   - Score the candidate with `score_candidate`.
   - Call `progress(done, total, title)`, where `total = 2 + mechanisms × bridges`. The gate counts as step 1 and the mechanisms as step 2. Skipped proposals advance progress too.
6. **Rank.** Rank the candidates and fill `calls` from counting wrappers around `deps`. The gate counts as one `llm` call.

`run_sampled` runs steps 1–2, then `n` proposals from `SAMPLED_USER` over the near passages: ids `S1…`, no bridge.

Fixture, in `tests/ideation_fixtures.py`: `world(gate_keeps=3, replies=None, cancel_after=None) -> (deps, log)`.
- Twenty papers `p0…p19` sit on the unit circle at 0°, 9°, …, 171°.
- `embed_query` maps `"abs-1"` to 60°, `"abs-2"` to 80°, and anything else to 0°. `embed_documents` maps everything to 5°.
- The gate keeps the first `gate_keeps` rows.
- `search` returns one hit per document in `doc_filter`, with chunk text `text of <doc>`, and logs `doc_filter`.
- `chat`:
  - answers a prompt containing `"mechanisms"` with M1 (abstraction `abs-1`; premises `near fact one`/C1 and `near fact two`/C2) and M2 (abstraction `abs-2`; premises `near fact two`/C2 and `near fact three`/C3);
  - answers every other prompt with a proposal whose premises are `bridge fact <K>` and `method fact <K>`, where K is the **last** `[C…]` key in the prompt, and whose `method_key` is K;
  - `replies` maps a proposal's 1-based ordinal to the replies it receives;
  - logs `chat_calls` and `proposals`.
- `judge` returns Supports with `span_verified=True` and logs each claim in `judged`.
- `cancelled()` becomes true once `cancel_after` proposals have been answered.
- `progress` is logged.

- [ ] **Step 1: Write the failing tests**

```python
class TestBestOfN(unittest.TestCase):
    def test_a_question_the_corpus_does_not_cover_is_refused_before_any_generation(self):
        deps, log = world(gate_keeps=1)
        with self.assertRaises(engine.CorpusCoverageError) as cm:
            engine.run_best_of_n("lithium dendrites with MXenes", deps, mechanisms=2, bridges=2)
        self.assertEqual((log.chat_calls, log.judged), (0, [])); self.assertIn("Research idea", str(cm.exception))

    def test_one_candidate_per_mechanism_and_bridge_each_from_its_own_paper(self):
        r = engine.run_best_of_n("q", world()[0], mechanisms=2, bridges=2)
        self.assertEqual(sorted(c.id for c in r.candidates), ["M1.B1", "M1.B2", "M2.B1", "M2.B2"])
        self.assertEqual({c.bridge["document"] for c in r.candidates}, {"p6", "p7", "p8", "p9"})

    def test_searches_are_limited_to_the_kept_papers_then_to_one_bridge_paper_each(self):
        deps, log = world(); engine.run_best_of_n("q", deps, mechanisms=2, bridges=2)
        self.assertEqual(log.doc_filters, [{"p0", "p1", "p2"}, {"p7"}, {"p6"}, {"p9"}, {"p8"}])

    def test_mechanism_premises_are_judged_once_however_many_candidates_inherit_them(self):
        deps, log = world(); r = engine.run_best_of_n("q", deps, mechanisms=2, bridges=2)
        self.assertEqual([log.judged.count(c) for c in ("near fact one", "near fact two", "near fact three")], [1, 1, 1])
        self.assertEqual((len(log.judged), r.calls["judge"], r.calls["llm"]), (11, 11, 6))
        self.assertTrue(all(c.scores.set_aside is None for c in r.candidates))

    def test_a_proposal_that_never_parses_is_skipped_and_the_rest_are_kept(self):
        r = engine.run_best_of_n("q", world(replies={2: ["nope", "still nope"]})[0], mechanisms=2, bridges=2)
        self.assertEqual(len(r.candidates), 3); self.assertTrue(any(w.startswith("M1.B2:") for w in r.warnings))

    def test_a_premise_citing_a_passage_from_another_prompt_is_dropped(self):
        # C3 is M2's premise passage; M1.B1's prompt shows only C1, C2 and its bridge passage C4
        reply = json.dumps({"title": "t", "conjecture": "c", "protocol": {},
                            "premises": [{"claim": "x", "key": "C3"}, {"claim": "bridge fact", "key": "C4"}]})
        r = engine.run_best_of_n("q", world(replies={1: [reply]})[0], mechanisms=2, bridges=2)
        first = next(c for c in r.candidates if c.id == "M1.B1")
        self.assertEqual([p.key for p in first.premises], ["C4"]); self.assertTrue(any("C3" in w for w in first.warnings))

    def test_cancelling_stops_before_the_next_proposal(self):
        deps, log = world(cancel_after=1)
        with self.assertRaises(pipeline_status.PipelineCancelledError):
            engine.run_best_of_n("q", deps, mechanisms=2, bridges=2)
        self.assertEqual(log.proposals, 1)

    def test_progress_reaches_its_total(self):
        deps, log = world(); engine.run_best_of_n("q", deps, mechanisms=2, bridges=2)
        self.assertEqual(log.progress[-1][:2], (6, 6))

    def test_a_band_with_too_few_papers_gives_fewer_candidates_and_says_so(self):
        r = engine.run_best_of_n("q", world()[0], mechanisms=2, bridges=4)
        self.assertEqual(len(r.candidates), 7); self.assertIn("band held 3 unused papers for M2", r.warnings)

class TestSampled(unittest.TestCase):
    def test_sampled_makes_n_candidates_from_the_near_passages_only(self):
        deps, log = world(); r = engine.run_sampled("q", deps, n=3)
        self.assertEqual([c.id for c in r.candidates], ["S1", "S2", "S3"])
        self.assertEqual(log.doc_filters, [{"p0", "p1", "p2"}]); self.assertTrue(all(c.bridge is None for c in r.candidates))

class TestGuards(unittest.TestCase):
    def test_ensure_idle_refuses_while_another_run_is_active(self):
        busy = {"active": True, "stage_label": "Ingesting and summarizing papers"}
        with patch.object(engine.pipeline_status, "get_status", return_value=busy):
            with self.assertRaisesRegex(engine.BusyError, "Ingesting and summarizing papers"):
                engine.ensure_idle()
        with patch.object(engine.pipeline_status, "get_status", return_value={"active": False}):
            engine.ensure_idle()

    def test_every_search_excludes_figures(self):
        with patch.object(engine, "hybrid_search", return_value=[]) as hs:
            engine.make_search(("col", "bm25", ["t"], [{}]))("q", 3, {"a.pdf"})
        self.assertEqual(hs.call_args.kwargs["exclude_types"], config.EXCLUDED_FROM_EVIDENCE)
        self.assertEqual(hs.call_args.kwargs["doc_filter"], {"a.pdf"})

    def test_describe_bands_reports_kept_rejected_and_the_band_without_generating(self):
        deps, log = world(); out = engine.describe_bands("q", deps)
        self.assertEqual([r["document"] for r in out["kept"]], ["p0", "p1", "p2"])
        self.assertEqual([r["document"] for r in out["rejected"]], ["p3", "p4", "p5"])
        self.assertEqual((out["band_size"], out["band"][0]["document"], log.chat_calls), (7, "p3", 0))

    def test_estimate_calls_for_three_by_two(self):
        self.assertEqual(engine.estimate_calls(3, 2), {"llm": 8, "judge_low": 18, "judge_high": 33})
```

Where the numbers come from:
- **Band.** With 20 papers, the band is ranks 2–9. Removing the kept p0–p2 leaves p3–p9.
- **Bridges.** M1 at 60° picks p7 then p6; M2 at 80° picks p9 then p8.
- **Judge calls.** 3 unique mechanism premises, plus 2 new premises for each of 4 proposals, makes 11.
- **Model calls.** The gate, the mechanisms call and 4 proposals make 6.
- **Too few bridges.** With 4 bridges each, M1 takes p7, p6, p8 and p5, which leaves only p9, p4 and p3 for M2.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ideation_engine.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `prompts.py`, `engine.py`, the fixture and the config lines**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ideation_engine.py tests/test_ideation_*.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ideation/prompts.py research_assistant/ideation/engine.py research_assistant/config.py \
        tests/ideation_fixtures.py tests/test_ideation_engine.py
git commit -m "feat(ideation): best-of-N over mechanisms x bridge papers, with coverage and busy guards"
```

---

### Task 6: Rendering and storage

**Files:**
- Create: `research_assistant/ideation/render.py`, `research_assistant/ideation/store.py`
- Modify: `research_assistant/config.py`
- Test: `tests/test_ideation_render.py`, `tests/test_ideation_store.py`

**Interfaces:**

Produces, in `render.py`:
- `premise_badge(premise) -> str`: first match wins.
  1. accepted → `"✅ Supports"`
  2. partial → `"🟡 Partially supports"`
  3. contradicts → `"❌ Contradicts"`
  4. error → `"⛔ Judge error"`
  5. otherwise → `"⚠️ Not accepted"`
- `mermaid_label(text: str, limit: int = 60) -> str`: whitespace collapsed, `"` turned into `'`, every one of ``#<>&`{}[]|`` removed, cut to `limit` characters ending in `…`.
- `render_tree(result) -> str`
  - Starts `flowchart TD`.
  - Root node: `R["<query>"]`.
  - Mechanism nodes: `M1["M1 · <statement>"]`.
  - Candidate nodes: `C_M1_B2["<title> · <composite:.2f>"]`; the id is `"C_" + id.replace(".", "_")`.
  - Edges run `R --> M1 --> C_…`; in sampled runs, `R --> C_…`.
  - Defines `classDef best` and `classDef aside`. The line `class <id> best` goes on the top candidate that is not set aside, and `class <id> aside` on each set-aside one.
- `render_markdown(result) -> str`, in this order:
  1. `# Hypothesis search: <query>`
  2. a line with the arm, preset, model, judge model and calls
  3. the spec's caveat sentence
  4. for each ranked candidate: `## <rank>. <title>`, then `**Conjecture (not stated by any source):** …`; premises as `- <badge> <claim> — <citation> (<key>)`, each with its verified span quoted beneath; the bridge paper with its percentile; the protocol; the scores
  5. `## Set aside`, with each candidate's reason
  6. `## Search tree`, with a ```` ```mermaid ```` block holding `render_tree`

Produces, in `store.py`:
- `save_result(result, directory: str | None = None) -> str`
  - `directory` defaults to `config.IDEATION_DIR`, read at call time.
  - Writes `<run_id>.json` (`atomic_write_json`) and `<run_id>.md`.
  - Returns the JSON path.
- `load_result(path: str) -> IdeationResult`

Config: `IDEATION_DIR = os.path.join(DATA_DIR, "ideation")`.

- [ ] **Step 1: Write the failing tests**

```python
class TestMermaid(unittest.TestCase):
    def test_hostile_text_cannot_break_a_label(self):
        label = render.mermaid_label('He said "x" [y] {z} #1 <b> & `code` |a|\nnext ' + "w" * 80, limit=60)
        for ch in '"[]{}#<>&`|\n':
            self.assertNotIn(ch, label)
        self.assertLessEqual(len(label), 60); self.assertTrue(label.endswith("…"))

    def test_the_tree_has_a_node_per_mechanism_and_candidate_and_marks_best_and_aside(self):
        tree = render.render_tree(sample_result())
        for line in ("R --> M1", "M1 --> C_M1_B1", "M1 --> C_M1_B2", "class C_M1_B1 best", "class C_M1_B2 aside"):
            self.assertIn(line, tree)

class TestMarkdown(unittest.TestCase):
    def test_the_report_labels_the_conjecture_badges_premises_and_sets_aside_with_the_reason(self):
        md = render.render_markdown(sample_result())
        for text in ("**Conjecture (not stated by any source):**", "✅ Supports", "## Set aside", "contradicted", "```mermaid"):
            self.assertIn(text, md)

class TestStore(unittest.TestCase):
    def test_save_writes_json_and_markdown_and_loads_back(self):
        with tempfile.TemporaryDirectory() as d:
            path = store.save_result(sample_result(), directory=d)
            self.assertEqual(path, os.path.join(d, "20261021-120000-abcdef12.json"))
            self.assertTrue(os.path.exists(path[:-5] + ".md"))
            self.assertEqual(store.load_result(path), sample_result())

    def test_the_default_directory_is_under_data_dir(self):
        self.assertTrue(os.path.realpath(config.IDEATION_DIR).startswith(os.path.realpath(config.DATA_DIR)))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ideation_render.py tests/test_ideation_store.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `render.py`, `store.py` and the config line**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ideation_render.py tests/test_ideation_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ideation/render.py research_assistant/ideation/store.py research_assistant/config.py \
        tests/test_ideation_render.py tests/test_ideation_store.py
git commit -m "feat(ideation): Markdown report, escaped Mermaid tree, results under data/ideation"
```

---

### Task 7: Command line

**Files:**
- Create: `scripts/ideate.py`
- Test: `tests/test_ideate_cli.py`

**Interfaces:**

Consumes: `engine.ensure_idle`, `live_deps`, `run_best_of_n`, `run_sampled`, `describe_bands`, `CorpusCoverageError` and `BusyError`; `store.save_result`.

Produces: `main(argv: list[str] | None = None) -> int`, with arguments:
- `--query/-q`: required
- `--arm {best_of_n,sampled}`: default `best_of_n`
- `--preset {balanced,grounded,radical}`
- `--mechanisms`, `--bridges`
- `--n`: sampled only; default mechanisms × bridges
- `--show-bands`
- `--out`: default `config.IDEATION_DIR`

Exit codes:

| code | when | output |
|---|---|---|
| 0 | the run finished | — |
| 2 | another run is busy | the `BusyError` message on stderr, before any loading |
| 3 | the corpus doesn't cover the question | the `CorpusCoverageError` message on stderr |

Output:
- On success, prints the JSON and Markdown paths, then one line per candidate: `rank  composite  G  N  T  set-aside  title`.
- Writes progress lines, `[done/total] label`, to stderr.
- `--show-bands` prints the kept and rejected shortlist with similarities, the relevance floor, the band size, and the 10 band papers nearest the query, each with its percentile and citation.

Docstring usage lines:
- `PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/ideate.py --query "..."`
- the same with `--show-bands`

- [ ] **Step 1: Write the failing tests** (imports: `from scripts import ideate`, `from research_assistant.ideation import engine`)

```python
class TestIdeateCli(unittest.TestCase):
    def test_a_busy_pipeline_exits_2_before_loading_anything(self):
        with patch.object(ideate.engine, "ensure_idle", side_effect=engine.BusyError("Another run is active: Ingesting")), \
             patch.object(ideate.engine, "live_deps") as deps, redirect_stderr(io.StringIO()) as err:
            self.assertEqual(ideate.main(["-q", "x"]), 2)
        deps.assert_not_called(); self.assertIn("Ingesting", err.getvalue())

    def test_an_uncovered_question_exits_3_with_the_message(self):
        refusal = engine.CorpusCoverageError("x", [], [{"document": "a.pdf", "citation": "Paper A", "summary": "", "score": 0.4}])
        with patch.object(ideate.engine, "ensure_idle"), patch.object(ideate.engine, "live_deps"), \
             patch.object(ideate.engine, "run_best_of_n", side_effect=refusal), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(ideate.main(["-q", "x"]), 3)
        self.assertIn("Paper A", err.getvalue())

    def test_a_run_is_saved_where_asked_and_its_path_printed(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ideate.engine, "ensure_idle"), \
             patch.object(ideate.engine, "live_deps"), patch.object(ideate.engine, "run_best_of_n", return_value=sample_result()), \
             redirect_stdout(io.StringIO()) as out:
            self.assertEqual(ideate.main(["-q", "x", "--out", d]), 0)
            self.assertIn(os.path.join(d, "20261021-120000-abcdef12.json"), out.getvalue())

    def test_show_bands_makes_no_generation_call(self):
        deps, log = world()
        with patch.object(ideate.engine, "ensure_idle"), patch.object(ideate.engine, "live_deps", return_value=deps), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(ideate.main(["-q", "q", "--show-bands"]), 0)
        self.assertEqual((log.chat_calls, log.judged), (0, []))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ideate_cli.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `scripts/ideate.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ideate_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Live check, once the migration has finished**

Run: `PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/ideate.py -q "Suppression of lithium dendrite growth using 2D MXenes"`
Expected: exit 3, with a message naming the nearest papers.

Run: `PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/ideate.py -q "Inferring the disorder configuration of a two-dimensional material from its conductance" --show-bands`
Expected: at least 3 kept papers, a band size above 0, and band papers that look like "different but not unrelated". If they don't, adjust `CITATION_IDEATION_BAND_LO_PCT`/`_HI_PCT` and record the values in the spec, §6.

Run: the same command without `--show-bands`.
Expected: JSON and Markdown paths under `data/ideation/`, and at least one candidate that is not set aside.

- [ ] **Step 6: Commit**

```bash
git add scripts/ideate.py tests/test_ideate_cli.py
git commit -m "feat(ideation): command line, with --show-bands for calibrating the band"
```

---

### Task 8: Evaluation harness

**Files:**
- Create: `research_assistant/eval/cases/ideation_queries.jsonl`, `research_assistant/eval/ideation_metrics.py`, `scripts/evaluate_ideation.py`
- Test: `tests/test_ideation_metrics.py`, `tests/test_evaluate_ideation.py`

**Interfaces:**

Cases: six lines of `{"id", "query", "cluster"}`, with this exact text (the maintainer may edit it later):

| id | query | cluster |
|---|---|---|
| q01 | Inferring the disorder configuration of a two-dimensional material from its conductance | quantum transport inverse problems |
| q02 | Tuning the electrical conductivity of MoS2 flakes through controlled amorphization | 2D materials |
| q03 | How edge disorder limits charge transport in graphene nanoribbons | graphene nanoribbons |
| q04 | Inverse design of molecules with targeted electronic properties | molecular design |
| q05 | Measuring the creativity of hypotheses proposed by large language models | LLMs for science |
| q06 | Forecasting hourly electricity demand where historical data are scarce | time series |

Record: `{"case", "arm", "run", "result", "answer", "seconds", "error"}`
- `arm` is `"chat"`, `"sampled"` or `"best_of_n"`.
- `result` is `result_to_dict(...)`, or `None` for the chat arm.
- `answer` is set for the chat arm only.
- A coverage refusal is recorded as `error="coverage: <message>"`.

Produces, in `ideation_metrics.py` (pure; rebuilds results with `models.result_from_dict`):
- `CAVEAT_NOTE = "Groundedness is the judge's verdict on each premise, never on the conjecture. Novelty is distance from the nearest paper in this corpus, not from the literature. Compare arms with each other on one corpus; an absolute number is not idea quality."`
- `HEADLINE = ("premise_acceptance", "grounded", "novelty", "testable", "composite", "set_aside_share", "llm_calls", "judge_calls", "seconds")`
- `top_candidate(result: dict) -> dict | None`: the first candidate that is not set aside.
- `score(records) -> dict`: returns `{"arms": {arm: {<each HEADLINE key>, "set_aside_by_reason": {...}, "spread", "scored", "errors"}}, "note": CAVEAT_NOTE}`.
  - `grounded`, `novelty`, `testable` and `composite` average the top candidate per (case, run). A record with no top candidate counts as composite 0.
  - `premise_acceptance` is accepted plus partial premises over all premises, counting each run's mechanism premises once.
  - `spread` is the mean over cases of |run 1 − run 2| in top composite; `None` with one run.
  - Chat records count only in `scored` and `errors`.
- `render(summary) -> str`
- `compare(a, b) -> str`: per arm and per headline, prints a, b and b − a, and marks `within spread` any difference smaller than either run's spread.
- `blind_sheet(records, seed: int = 0) -> tuple[str, dict]`
  - Uses run 1 only. Each case gets one item per arm: the chat arm's answer, or the top candidate (title, conjecture, protocol, premises with badges).
  - Items are shuffled per case with `random.Random(seed)` and labelled `q01-A`, `q01-B`, …
  - The Markdown names no arm. The key is `{label: arm}`.
- `merge_ratings(summary, rows: list[dict], key: dict) -> dict`
  - `rows` are `{label, plausible, novel, testable, pursue}`, rated 1–5; blank cells are skipped.
  - Adds `summary["arms"][arm]["ratings"] = {field: mean}`, creating the arm's entry when the summary has none.

Produces, in `scripts/evaluate_ideation.py`:
- `RESULTS_DIR = os.path.join(DATA_DIR, "eval", "ideation", "results")`
- `run_case(case, arm, run, deps, resources) -> dict`. Any exception becomes `error`; `run_case` never raises.

  | arm | how it runs |
  |---|---|
  | chat | `agent = ResearchChat(search_resources=resources, mode="hypotheses")`, then `"".join(agent.stream_turn(query, mode="hypotheses"))`; the answer is `agent.turns[-1]["resolved_answer"]` |
  | sampled | `engine.run_sampled(query, deps, n=IDEATION_MECHANISMS * IDEATION_BRIDGES)` |
  | best_of_n | `engine.run_best_of_n(query, deps)` |

- `main(argv=None) -> int`
  - `--score --runs N [--arms chat,sampled,best_of_n] [--cases ids…] [--out DIR]`:
    1. Call `ensure_idle()`.
    2. Call `live_deps()` and `load_search_resources()` once.
    3. Run every case × run × arm.
    4. Write `<stamp>.json` (`{"records", "summary"}`), `<stamp>-sheet.md`, `<stamp>-key.json` and `<stamp>-ratings.csv` (header `label,plausible,novel,testable,pursue`, one row per label).
  - `--ratings CSV RESULT_JSON`: merges the ratings and rewrites the summary.
  - `--compare A B`: prints `compare`.
  - Usage lines go in the docstring, as in `scripts/evaluate_citer.py`.

- [ ] **Step 1: Write the failing tests**

Records are built from `result_to_dict(sample_result())`, with edited scores and premises (helper `_rec(case_id, arm, run, top=0.5, aside=None)`).

```python
class TestIdeationMetrics(unittest.TestCase):
    def test_the_top_candidate_is_the_first_not_set_aside(self):
        r = models.result_to_dict(sample_result())
        self.assertEqual(m.top_candidate(r)["id"], "M1.B1")
        r["candidates"] = r["candidates"][1:]
        self.assertIsNone(m.top_candidate(r))

    def test_premise_acceptance_counts_each_runs_mechanism_premises_once(self):
        # sample_result: M1 accepted; M1.B1 accepted; M1.B2 contradicted → 2 of 3
        self.assertAlmostEqual(m.score([_rec("q01", "best_of_n", 1)])["arms"]["best_of_n"]["premise_acceptance"], 2 / 3)

    def test_the_spread_is_the_mean_gap_between_two_runs_and_absent_with_one(self):
        recs = [_rec("q01", "best_of_n", 1, top=0.6), _rec("q01", "best_of_n", 2, top=0.4),
                _rec("q02", "best_of_n", 1, top=0.5), _rec("q02", "best_of_n", 2, top=0.5)]
        self.assertAlmostEqual(m.score(recs)["arms"]["best_of_n"]["spread"], 0.1)
        self.assertIsNone(m.score(recs[::2])["arms"]["best_of_n"]["spread"])

    def test_a_failed_record_is_counted_not_scored(self):
        s = m.score([_rec("q01", "sampled", 1), {**_rec("q02", "sampled", 1), "result": None, "error": "coverage: …"}])
        self.assertEqual((s["arms"]["sampled"]["scored"], s["arms"]["sampled"]["errors"]), (1, 1))

    def test_the_blind_sheet_hides_the_arm_and_the_key_restores_it(self):
        recs = [_rec("q01", "best_of_n", 1), _rec("q01", "sampled", 1),
                {"case": {"id": "q01", "query": "q"}, "arm": "chat", "run": 1, "result": None, "answer": "An idea.", "seconds": 1, "error": None}]
        sheet, key = m.blind_sheet(recs)
        for arm in ("chat", "sampled", "best_of_n"):
            self.assertNotIn(arm, sheet)
        self.assertEqual(set(key), {"q01-A", "q01-B", "q01-C"}); self.assertEqual(sorted(key.values()), ["best_of_n", "chat", "sampled"])

    def test_ratings_are_averaged_per_arm(self):
        s = m.merge_ratings(m.score([_rec("q01", "best_of_n", 1)]),
                            [{"label": "q01-A", "pursue": "2"}, {"label": "q01-B", "pursue": "4"}],
                            {"q01-A": "chat", "q01-B": "best_of_n"})
        self.assertEqual((s["arms"]["chat"]["ratings"]["pursue"], s["arms"]["best_of_n"]["ratings"]["pursue"]), (2.0, 4.0))
```

```python
class TestEvaluateIdeation(unittest.TestCase):
    def test_a_coverage_refusal_is_recorded_not_raised(self):
        with patch.object(ev.engine, "run_best_of_n", side_effect=engine.CorpusCoverageError("q", [], [])):
            rec = ev.run_case({"id": "q01", "query": "q"}, "best_of_n", 1, deps=None, resources=None)
        self.assertTrue(rec["error"].startswith("coverage:")); self.assertIsNone(rec["result"])

    def test_any_other_failure_becomes_an_error_record(self):
        with patch.object(ev.engine, "run_sampled", side_effect=RuntimeError("boom")):
            self.assertEqual(ev.run_case({"id": "q01", "query": "q"}, "sampled", 1, None, None)["error"], "RuntimeError: boom")

    def test_the_chat_arm_uses_the_hypotheses_lens(self):
        fake = MagicMock(); fake.return_value.stream_turn.return_value = iter(["An ", "idea."])
        fake.return_value.turns = [{"resolved_answer": "An idea."}]
        with patch.object(ev, "ResearchChat", fake):
            rec = ev.run_case({"id": "q01", "query": "q"}, "chat", 1, None, ("c", "b", [], []))
        self.assertEqual(rec["answer"], "An idea."); self.assertEqual(fake.call_args.kwargs["mode"], "hypotheses")

    def test_score_writes_results_sheet_key_and_ratings_under_out(self):
        with tempfile.TemporaryDirectory() as d, patch.object(ev.engine, "ensure_idle"), \
             patch.object(ev.engine, "live_deps"), patch.object(ev, "load_search_resources"), \
             patch.object(ev, "run_case", side_effect=lambda case, arm, run, *rest, **kw: _rec(case["id"], arm, run)):
            self.assertEqual(ev.main(["--score", "--runs", "2", "--cases", "q01", "--out", d]), 0)
            names = os.listdir(d)
            for suffix in ("-sheet.md", "-key.json", "-ratings.csv"):
                self.assertEqual(sum(n.endswith(suffix) for n in names), 1, suffix)
            result = [n for n in names if n.endswith(".json") and not n.endswith("-key.json")][0]
            self.assertEqual(len(json.load(open(os.path.join(d, result)))["records"]), 6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_ideation_metrics.py tests/test_evaluate_ideation.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the cases file, `ideation_metrics.py` and `scripts/evaluate_ideation.py`**

The cases file is already covered by `pyproject.toml`'s `"research_assistant.eval" = ["cases/*.jsonl"]`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ideation_metrics.py tests/test_evaluate_ideation.py tests/test_packaging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/eval/cases/ideation_queries.jsonl research_assistant/eval/ideation_metrics.py \
        scripts/evaluate_ideation.py tests/test_ideation_metrics.py tests/test_evaluate_ideation.py
git commit -m "feat(eval): hypothesis-search evaluation: chat lens vs sampled vs best-of-N, blind sheet"
```

- [ ] **Step 6: Live run, maintainer, once the migration has finished**

Run: `PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/evaluate_ideation.py --score --runs 2`
Expected: four files under `data/eval/ideation/results/`.

Fill in `<stamp>-ratings.csv` from `<stamp>-sheet.md` without opening the key, then run:
`PYTHONPATH=. CITATION_LOG_FILE=0 python scripts/evaluate_ideation.py --ratings <stamp>-ratings.csv <stamp>.json`

Copy the rendered summary into the spec, §6, and apply the rule in the spec, §2.H.

---

> **Checkpoint.** Tasks 9–11 don't depend on the eval's outcome, but the eval says whether this feature deserves a tab. If `best_of_n` does not beat `sampled` by more than the spread (spec §2.H), stop here and keep it CLI-only.

---

### Task 9: Background-job plumbing for a third origin

**Files:**
- Modify: `research_assistant/shared/pipeline_status.py:60-82`
- Modify: `research_assistant/ui/jobs.py` (`_adopt_job_result`, `_handle_job_state`)
- Test: `tests/test_pipeline_status.py`, and `tests/test_app_render.py` in class `TestJobResolution` (it already patches `seed_audit.AUDIT_DIR` and resets the registry)

**Interfaces:**
- `pipeline_status`:
  - `"ideate"` joins `VALID_STAGES`.
  - `STAGE_METADATA["ideate"] = (1, "Searching for hypotheses")`.
  - Without this, `set_status` coerces the stage to `"idle"`.
- `ui/jobs.py`:
  - `_RESULT_KEYS = {"idea": ("idea_result", "idea_query"), "ideation": ("ideation_result", "ideation_query")}`.
  - `_adopt_job_result(final, origin)` writes `_RESULT_KEYS[origin]` for those two origins. Every other origin keeps today's audit branch (`audit_result`, `audit_query`, `build_result`, `build_query`).
  - `_handle_job_state(origin)` reads `already` from `_RESULT_KEYS.get(origin, ("audit_result",))[0]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_status.py — class TestPipelineStatus (its setUp points the status file at a temp path)
def test_the_ideate_stage_is_kept_and_labelled(self):
    s = pipeline_status.set_status(active=True, stage="ideate")
    self.assertEqual((s["stage"], s["stage_label"], s["current_step"]), ("ideate", "Searching for hypotheses", 1))
```

```python
# tests/test_app_render.py — class TestJobResolution
def test_an_ideation_result_is_adopted_into_its_own_keys(self):
    with patch.object(ui_jobs.st, "session_state", {"active_job": "j", "active_job_origin": "ideation"}) as ss:
        ui_jobs._adopt_job_result({"query": "q", "candidates": []}, "ideation")
    self.assertEqual((ss["ideation_result"]["query"], ss["ideation_query"]), ("q", "q"))
    self.assertFalse({"audit_result", "idea_result", "build_result", "active_job"} & set(ss))

def test_audit_and_idea_adoption_are_unchanged(self):
    with patch.object(ui_jobs.st, "session_state", {}) as ss:
        ui_jobs._adopt_job_result({"seed_path": "p.pdf"}, "audit")
    self.assertIn("build_result", ss)
    with patch.object(ui_jobs.st, "session_state", {}) as ss:
        ui_jobs._adopt_job_result({"query": "i"}, "idea")
    self.assertEqual(ss["idea_query"], "i")

def test_a_finished_ideation_run_belongs_to_the_ideation_tab_only(self):
    self.run_jobs.save_run("k", {"query": "q", "candidates": []}, label="q", origin="ideation")
    with patch.object(ui_jobs.st, "session_state", {"active_job": "k"}) as ss, patch.object(ui_jobs.st, "query_params", {}):
        self.assertFalse(ui_jobs._handle_job_state("idea")); self.assertNotIn("idea_result", ss)
        self.assertFalse(ui_jobs._handle_job_state("ideation")); self.assertEqual(ss["ideation_result"]["query"], "q")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_status.py tests/test_app_render.py -v`
Expected: FAIL on the new tests only.

- [ ] **Step 3: Implement the two changes**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_status.py tests/test_app_render.py tests/test_app_audit_in_two_tabs.py -v`
Expected: PASS, including the existing audit and idea tests.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/shared/pipeline_status.py research_assistant/ui/jobs.py tests/test_pipeline_status.py tests/test_app_render.py
git commit -m "feat(ui): an 'ideation' job origin and pipeline stage beside audit and idea"
```

---

### Task 10: The Hypothesis search tab

**Files:**
- Create: `research_assistant/ui/ideation_tab.py`
- Modify: `research_assistant/ui/common.py`, `research_assistant/tasks/ui.py:59-101`, `app.py`
- Test: `tests/test_app_smoke.py:23-27`, `tests/test_app_ideation_tab.py`

**Interfaces:**

Consumes: Tasks 5, 6 and 9; `run_jobs.start_job`, `run_jobs.job_id_for`, `JobBusy`; `ui.jobs._handle_job_state` and `_render_recent_runs`; `ui.status._render_live_pipeline_status`.

Produces:
- `ui/common.py`: `_render_mermaid(mermaid_code: str, height: int = 420) -> None`, moved verbatim from `tasks/ui.py::_render_mermaid_interactive`.
- `tasks/ui.py`: `from research_assistant.ui.common import _render_mermaid as _render_mermaid_interactive`, which leaves its call sites and `tests/test_task_ui.py` untouched.
- `ui/ideation_tab.py`:
  - `render_ideation_tab() -> None`
  - `_start_ideation_job(query: str, preset: str, mechanisms: int, bridges: int) -> str | None`. It mirrors `ui/jobs._start_pipeline_job`:
    - `job_id = run_jobs.job_id_for(f"ideation|{query}|{preset}|{mechanisms}|{bridges}")`
    - `run_jobs.start_job(job_id, query, _ideation_runner(...), origin="ideation")`
    - `JobBusy` → `st.warning`, return `None`.
    - Sets `st.query_params["job"]`, `active_job` and `active_job_origin`.
  - `_ideation_runner(query, preset, mechanisms, bridges) -> Callable[[], dict]`. It touches no Streamlit and takes no `on_snapshot`, so the live view shows only `pipeline_status`. Its steps:
    1. `pipeline_status.set_status(active=True, stage="ideate", detail=query[:60], item_current=0, item_total=0)`
    2. `deps = engine.live_deps()`, with `deps.progress = lambda d, t, label: pipeline_status.update_progress(item_current=d, item_total=t, detail=label)`
    3. `run_best_of_n`, then `store.save_result`
    4. Return `models.result_to_dict(result)`.
    5. On `CorpusCoverageError`, return `{"query": query, "coverage_error": str(e)}` instead.
    6. In `finally`, call `pipeline_status.set_status(active=False, stage="idle", detail="Hypothesis search finished")`.
  - `_render_result(result: IdeationResult) -> None`
- `app.py`:
  - tabs `["Agent task planner", "Citation auditor", "Research idea", "Cite a draft", "Research chat", "Hypothesis search", "How to use"]`
  - `with tab_ideation: render_ideation_tab()`
  - the docstring says "seven tabs" and lists `ideation_tab`

`render_ideation_tab` lays out the tab in this order:
1. `_render_live_pipeline_status("tab_ideation")`.
2. Subheader `🧪 Hypothesis search`, then the caption: "Experimental. Proposes conjectures that carry a mechanism from the papers nearest your question onto a finding from a paper further away. Each premise is checked by the citation judge; the conjecture itself is not stated by any source and is marked so."
3. `st.form("ideation_form")` containing:
   - a text input "Research question" (`key="ideation_query_input"`);
   - a select box "Preset" over the `IDEATION_PRESETS` keys;
   - "Mechanisms" (1–3) and "Bridge papers per mechanism" (1–4), defaulting to the config values;
   - the caption "About {llm} model calls and {judge_low}–{judge_high} judge calls.", from `estimate_calls`;
   - the submit button "Search for hypotheses" (`key="ideation_submit"`).
4. On submit:
   - An empty question → `st.warning("Enter a research question first.")`.
   - `pipeline_status.get_status()["active"]` → `st.warning(f"Another run is active ({stage_label}). Wait for it to finish, then search again.", icon="⏳")`.
   - Otherwise → `_start_ideation_job`.
5. If `_handle_job_state("ideation")` returns true, stop.
6. Else, if `ideation_result` is in the session:
   - with `coverage_error`: `st.warning(...)`;
   - otherwise: `_render_result(models.result_from_dict(...))`.
7. Else, `_render_recent_runs("ideation")`.

`_render_result` shows:
- the query and a settings caption;
- one `st.expander(f"{rank}. {title} — {composite:.2f}")` per candidate, the top 3 expanded, each holding:
  - four `st.metric`s: Grounded, Novelty, Testable, Relevance;
  - `st.info("**Conjecture (not stated by any source):** …")`;
  - the premises, each as `render.premise_badge` plus the claim and citation, with the verified span in a blockquote;
  - the bridge paper and its percentile;
  - the protocol, with ✓ or ✗ per check;
- the set-aside candidates in a collapsed `st.expander(f"Set aside ({n})")`, each with its reason;
- `_render_mermaid(render.render_tree(result))`;
- `st.download_button("Download report (.md)", …)` and `st.download_button("Download run (.json)", …)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app_smoke.py — replaces test_the_six_tabs_are_there_in_order
def test_the_seven_tabs_are_there_in_order(self):
    self.assertEqual(self.page["tabs"], ["Agent task planner", "Citation auditor", "Research idea", "Cite a draft",
                                         "Research chat", "Hypothesis search", "How to use"])
```

```python
# tests/test_app_ideation_tab.py — skips without streamlit, as tests/test_app_smoke.py does
class TestIdeationTab(unittest.TestCase):
    def test_a_finished_search_renders_its_candidates_and_both_downloads(self):
        payload = json.dumps(models.result_to_dict(sample_result()))
        page = run_page(before=f'import json\nat.session_state["ideation_result"] = json.loads({payload!r})',
                        after='out["expanders"] = [e.label for e in at.expander]\n'
                              'out["downloads"] = [e.proto.label for e in at.get("download_button")]')
        self.assertEqual(page["exceptions"], [])
        self.assertTrue(any("Disorder maps from noise spectra" in label for label in page["expanders"]))
        self.assertIn("Download report (.md)", page["downloads"]); self.assertIn("Download run (.json)", page["downloads"])

    def test_a_search_is_refused_while_another_run_is_active(self):
        before = ('import os\nfrom research_assistant.shared import pipeline_status\n'
                  'pipeline_status.set_status(active=True, stage="ingest_refs", pid=os.getpid())')
        after = ('at.text_input(key="ideation_query_input").input("graphene disorder")\n'
                 'at.button(key="ideation_submit").click().run()\n'
                 'out["warnings"] = [w.value for w in at.warning]')
        page = run_page(before=before, after=after)
        self.assertTrue(any("Another run is active (Ingesting and summarizing papers)" in w for w in page["warnings"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_app_smoke.py tests/test_app_ideation_tab.py -v`
Expected: FAIL. There is no seventh tab yet, and `ideation_tab` does not exist.

- [ ] **Step 3: Implement the move, the tab and the `app.py` change**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_app_smoke.py tests/test_app_ideation_tab.py tests/test_task_ui.py tests/test_ui_modules.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add research_assistant/ui/ideation_tab.py research_assistant/ui/common.py research_assistant/tasks/ui.py app.py \
        tests/test_app_smoke.py tests/test_app_ideation_tab.py
git commit -m "feat(ui): Hypothesis search tab, run as a background job, after Research chat"
```

---

### Task 11: Documentation and the CI import sweep

**Files:**
- Modify: `README.md:56-64`, `README.md:239` and its path table
- Modify: `HOW_TO_USE.md:9`, plus a new section before `### Tab 5 — How to use` (line 297)
- Modify: `ARCHITECTURE.md`, adding a new §5.6 after §5.5
- Modify: `.github/workflows/tests.yml`, the import sweep list

**Interfaces:** None. The doc copy is below.

- [ ] **Step 1: Make the doc edits**

`README.md`:
- Line 56 becomes: "Seven tabs. The seventh is the instructions, which nobody reads, which he has noticed."
- Add this row after Research chat: `| **Hypothesis search** *(experimental)* | a research question your corpus already covers | conjectures that carry a mechanism from the nearest papers onto a finding from a paper further away, ranked; every premise checked by the citation judge, the conjecture itself marked as stated by no source; a Markdown report and the search tree |`
- Line 239 becomes "the Streamlit page: sidebar and seven tabs".
- Add this row after `research_assistant/tasks/`: `| research_assistant/ideation/ | Hypothesis search: paper-level bands, premise checking, best-of-N; scripts/ideate.py runs it, scripts/evaluate_ideation.py evaluates it |`

`HOW_TO_USE.md`:
- The heading becomes `## 🧭 Interface Overview (The 7 Tabs)`, and its list gains the new tab.
- New section `### Tab 5 — Hypothesis search (experimental)`, covering:
  - what to type;
  - the coverage refusal, and building a corpus in Research idea first;
  - what the premise badges mean, and what "not stated by any source" means;
  - the call estimate;
  - that the run continues in the background and the sidebar can stop it.
- `### Tab 5 — How to use` becomes `### Tab 6 — How to use`.

`ARCHITECTURE.md`: new §5.6, "Hypothesis search reads summary vectors and judges premises". One short paragraph each on spec decisions A, D, E, F and H, linking the spec.

`tests.yml`: add `research_assistant.ideation.engine research_assistant.ideation.render research_assistant.ideation.store research_assistant.eval.ideation_metrics` before `app orchestrate watch`.

- [ ] **Step 2: Verify**

Run: `for m in research_assistant.ideation.engine research_assistant.ideation.render research_assistant.ideation.store research_assistant.eval.ideation_metrics; do CITATION_LOG_FILE=0 python -c "import $m" || echo "FAIL $m"; done`
Expected: no output.

Run: `CITATION_LOG_FILE=0 python -m pytest tests/ -q`
Expected: all pass. The help tab renders the new HOW_TO_USE without an exception (`tests/test_app_smoke.py`).

- [ ] **Step 3: Commit**

```bash
git add README.md HOW_TO_USE.md ARCHITECTURE.md .github/workflows/tests.yml
git commit -m "docs: the Hypothesis search tab, its design, and the CI import sweep"
```
