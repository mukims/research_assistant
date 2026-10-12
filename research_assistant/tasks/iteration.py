"""
Multi-Turn Proposer-Critic Iteration Engine.

Executes a hierarchical refinement loop between Proposer and Critic (up to 3 turns),
terminating early as soon as zero blocking flaws remain, and attaching the critique
trace and final rubric scorecard to the task graph metadata.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from research_assistant.config import LLM_MODEL
from research_assistant.shared.llm import chat
from research_assistant.shared.log import get_logger
from research_assistant.tasks.critic import (
    FlawSeverity,
    MacroCritic,
    MicroCritic,
    RubricScorecard,
    TaskCritic,
)
from research_assistant.tasks.harness import generate_task_test_script
from research_assistant.tasks.models import (
    AgentRole,
    ArtifactContract,
    ArtifactType,
    Task,
    TaskCategory,
    TaskGraph,
)
from research_assistant.tasks.prompts import (
    PROTOCOL_DECONSTRUCTION_SYSTEM,
    PROTOCOL_DECONSTRUCTION_USER,
)

logger = get_logger("task_iteration")

REFINEMENT_PROMPT_TEMPLATE = """You are refining a scientific task decomposition DAG following an adversarial critique.
The previous iteration of the task graph contained the following flaws that MUST be resolved:

### Critique Flaws Identified (Turn {turn}):
{flaws_report}

### Target Scientific Paper:
# Title: {paper_title}

{paper_text}

### Remediation Requirements:
1. Resolve ALL blocking flaws and address warnings.
2. Maintain strict DAG topology: NO cycles, self-dependencies, or missing dependencies.
3. Ensure all 6 mandatory scientific phases are present:
   - Environment Setup (`environment_setup`, `devops_agent`)
   - Data Splits & Acquisition (`data_preprocessing`, `data_engineer`)
   - Sanity Smoke-Tests (`sanity_smoke_test`, `model_architect`)
   - Training / Simulation (`training_or_simulation`, `experiment_runner`)
   - Metric Evaluation (`metric_evaluation`, `evaluation_agent`)
   - Verification Audit (`verification_audit`, `audit_agent`)
4. Explicit artifact deliverables with file extensions (.pt, .h5, .json, .csv, .lock).
5. Concrete, deterministic acceptance criteria (pass/fail thresholds, exit codes; no subjective phrasing).
6. CPU-safe bounds: single-threaded execution, torch.set_num_threads(1), OMP_NUM_THREADS=1.

Return ONLY a valid JSON object matching the standard task graph schema with 'paper_title' and 'tasks' list.
"""


def _build_heuristic_7phase_fallback(paper_id: str, paper_title: str, context: Optional[Dict[str, Any]] = None) -> TaskGraph:
    """Builds a verified, high-scoring 7-task DAG covering all 6 mandatory phases."""
    graph = TaskGraph(paper_id=paper_id, paper_title=paper_title)
    ctx = context or {}

    abstract_excerpt = [ctx["abstract"][:300]] if ctx.get("abstract") else []
    section_excerpt = (ctx.get("sections") or ["Dataset splits and preprocessing pipeline."])[:1]
    model_excerpt = (ctx.get("sections") or ["Model architecture and mathematical formulation."])[:1]
    exp_excerpt = (ctx.get("sections") or ["Experimental setup and training protocol."])[:1]
    eval_excerpt = (ctx.get("tables") or ctx.get("sections") or ["Benchmark metric evaluation."])[:1]

    # 1. Environment Setup
    t1 = Task(
        task_id="task_01_env",
        title="Provision Environment & Lock Dependencies",
        category=TaskCategory.ENVIRONMENT_SETUP,
        description="Configure Python runtime, install required computational packages, verify CPU/CUDA drivers, and lock thread limits (OMP_NUM_THREADS=1).",
        assigned_role=AgentRole.DEVOPS_AGENT,
        dependencies=[],
        inputs=["requirements.txt", "runtime_config.yaml"],
        outputs=["env_ready.lock"],
        acceptance_criteria=["Environment builds cleanly with exit code 0", "Core scientific packages import without error"],
        paper_context_excerpts=abstract_excerpt,
        artifact_contracts=[
            ArtifactContract(
                path="env_ready.lock",
                artifact_type=ArtifactType.LOCKFILE,
                description="Locked Python runtime environment and package dependencies",
            ),
        ],
    )

    # 2. Data Splits & Acquisition
    t2 = Task(
        task_id="task_02_data",
        title="Acquire Raw Data & Generate Deterministic Splits",
        category=TaskCategory.DATA_PREPROCESSING,
        description="Download raw scientific datasets described in paper, apply standard normalization, and partition into reproducible train, validation, and test splits with fixed random seed 42.",
        assigned_role=AgentRole.DATA_ENGINEER,
        dependencies=["task_01_env"],
        inputs=["env_ready.lock", "data_spec.yaml"],
        outputs=["data/train.pt", "data/val.pt", "data/test.pt"],
        acceptance_criteria=["Train/Val/Test splits verified non-empty", "Zero missing values or NaN entries detected"],
        paper_context_excerpts=section_excerpt,
        artifact_contracts=[
            ArtifactContract(
                path="data/train.pt",
                artifact_type=ArtifactType.TENSOR,
                shape=["N_train", "features"],
                dtype="float32",
                description="Deterministic training split tensor",
            ),
            ArtifactContract(
                path="data/val.pt",
                artifact_type=ArtifactType.TENSOR,
                shape=["N_val", "features"],
                dtype="float32",
                description="Validation split tensor",
            ),
            ArtifactContract(
                path="data/test.pt",
                artifact_type=ArtifactType.TENSOR,
                shape=["N_test", "features"],
                dtype="float32",
                description="Held-out evaluation split tensor",
            ),
        ],
    )

    # 3. Model Architecture
    t3 = Task(
        task_id="task_03_model",
        title="Implement Core Neural Architecture & Mathematical Operators",
        category=TaskCategory.MODEL_IMPLEMENTATION,
        description="Implement the mathematical model architecture, layer dimensions, custom operators, and differentiable loss functions described in the methodology.",
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_01_env"],
        inputs=["env_ready.lock", "model_config.yaml"],
        outputs=["src/model.py", "src/loss.py"],
        acceptance_criteria=["Forward pass completes with correct tensor dimensions", "Loss function is differentiable and gradient norm is finite"],
        paper_context_excerpts=model_excerpt,
        artifact_contracts=[
            ArtifactContract(
                path="src/model.py",
                artifact_type=ArtifactType.CODE,
                description="Core neural network architecture implementation",
            ),
            ArtifactContract(
                path="src/loss.py",
                artifact_type=ArtifactType.CODE,
                description="Mathematical loss formulation and differentiable objective",
            ),
        ],
    )

    # 4. Sanity Smoke-Tests
    t4 = Task(
        task_id="task_04_smoke",
        title="Sanity Smoke-Test Forward Pass & Gradient Flow",
        category=TaskCategory.SANITY_SMOKE_TEST,
        description="Execute a single forward pass and backward step on a dummy mini-batch to verify gradient propagation, shape compatibility, and CPU thread safety before full training.",
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_02_data", "task_03_model"],
        inputs=["data/train.pt", "src/model.py", "src/loss.py"],
        outputs=["logs/smoke_test_passed.log"],
        acceptance_criteria=["Single mini-batch forward pass completes with exit code 0", "Gradients non-zero and no NaN values produced"],
        paper_context_excerpts=model_excerpt,
        artifact_contracts=[
            ArtifactContract(
                path="logs/smoke_test_passed.log",
                artifact_type=ArtifactType.LOCKFILE,
                description="Smoke test verification log certifying finite gradients and shape consistency",
            ),
        ],
    )

    # 5. Training / Simulation
    t5 = Task(
        task_id="task_05_train",
        title="Execute Training Loop & Optimization",
        category=TaskCategory.TRAINING_OR_SIMULATION,
        description="Train the model using specified optimizer, learning rate schedule, batch size, and epochs while enforcing CPU-safe thread bounds (torch.set_num_threads(1), OMP_NUM_THREADS=1).",
        assigned_role=AgentRole.EXPERIMENT_RUNNER,
        dependencies=["task_04_smoke"],
        inputs=["data/train.pt", "data/val.pt", "src/model.py", "logs/smoke_test_passed.log"],
        outputs=["checkpoints/best_model.pt", "logs/train_metrics.csv"],
        acceptance_criteria=["Training loss converges over epochs", "Model checkpoint saved successfully"],
        paper_context_excerpts=exp_excerpt,
        artifact_contracts=[
            ArtifactContract(
                path="checkpoints/best_model.pt",
                artifact_type=ArtifactType.TENSOR,
                shape=["weights"],
                dtype="float32",
                description="Trained model checkpoint weights",
            ),
            ArtifactContract(
                path="logs/train_metrics.csv",
                artifact_type=ArtifactType.TABULAR,
                columns={"epoch": "int", "loss": "float", "val_loss": "float", "learning_rate": "float"},
                description="Training loss and validation metrics per epoch",
            ),
        ],
    )

    # 6. Metric Evaluation
    t6 = Task(
        task_id="task_06_eval",
        title="Compute Evaluation Metrics on Held-Out Test Set",
        category=TaskCategory.METRIC_EVALUATION,
        description="Evaluate the trained model checkpoint on the held-out test split, compute reported metrics, and generate comparison tables against paper baseline targets.",
        assigned_role=AgentRole.EVALUATION_AGENT,
        dependencies=["task_05_train"],
        inputs=["checkpoints/best_model.pt", "data/test.pt"],
        outputs=["metrics/evaluation_report.json"],
        acceptance_criteria=["Primary performance metric computed on test set", "Comparison table generated against paper targets"],
        paper_context_excerpts=eval_excerpt,
        artifact_contracts=[
            ArtifactContract(
                path="metrics/evaluation_report.json",
                artifact_type=ArtifactType.METRICS_JSON,
                required_keys=["test_accuracy", "f1_score", "paper_target_comparison"],
                description="Final evaluation metrics on held-out test split compared against paper targets",
            ),
        ],
    )

    # 7. Verification Audit
    t7 = Task(
        task_id="task_07_audit",
        title="Audit Deliverables & Compile Replication Scorecard",
        category=TaskCategory.VERIFICATION_AUDIT,
        description="Verify checksums of all output deliverables, cross-reference replicated metrics against paper claims, and produce formal replication scorecard.",
        assigned_role=AgentRole.AUDIT_AGENT,
        dependencies=["task_06_eval"],
        inputs=["metrics/evaluation_report.json", "logs/train_metrics.csv"],
        outputs=["results/replication_scorecard.md"],
        acceptance_criteria=["Replication scorecard generated", "All artifact checksums certified non-empty"],
        paper_context_excerpts=eval_excerpt,
        artifact_contracts=[
            ArtifactContract(
                path="results/replication_scorecard.md",
                artifact_type=ArtifactType.CODE,
                description="Certified replication audit scorecard",
            ),
        ],
    )

    for t in [t1, t2, t3, t4, t5, t6, t7]:
        graph.add_task(t)

    return graph



def _resolve_category(t_dict: Dict[str, Any], title_hint: str = "", desc_hint: str = "") -> TaskCategory:
    """Robustly resolves TaskCategory from LLM outputs handling aliases, case, and keyword context."""
    raw = (
        t_dict.get("category")
        or t_dict.get("phase")
        or t_dict.get("task_category")
        or t_dict.get("type")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")

    for c in TaskCategory:
        if c.value == raw:
            return c

    CATEGORY_ALIASES = {
        "env": TaskCategory.ENVIRONMENT_SETUP,
        "environment": TaskCategory.ENVIRONMENT_SETUP,
        "environment_setup": TaskCategory.ENVIRONMENT_SETUP,
        "setup": TaskCategory.ENVIRONMENT_SETUP,
        "data": TaskCategory.DATA_PREPROCESSING,
        "data_acquisition": TaskCategory.DATA_ACQUISITION,
        "data_download": TaskCategory.DATA_ACQUISITION,
        "data_preprocessing": TaskCategory.DATA_PREPROCESSING,
        "preprocessing": TaskCategory.DATA_PREPROCESSING,
        "dataset": TaskCategory.DATA_PREPROCESSING,
        "model": TaskCategory.MODEL_IMPLEMENTATION,
        "model_implementation": TaskCategory.MODEL_IMPLEMENTATION,
        "architecture": TaskCategory.MODEL_IMPLEMENTATION,
        "algorithm": TaskCategory.MODEL_IMPLEMENTATION,
        "smoke": TaskCategory.SANITY_SMOKE_TEST,
        "smoke_test": TaskCategory.SANITY_SMOKE_TEST,
        "sanity": TaskCategory.SANITY_SMOKE_TEST,
        "sanity_smoke_test": TaskCategory.SANITY_SMOKE_TEST,
        "sanity_smoke_tests": TaskCategory.SANITY_SMOKE_TEST,
        "train": TaskCategory.TRAINING_OR_SIMULATION,
        "training": TaskCategory.TRAINING_OR_SIMULATION,
        "simulation": TaskCategory.TRAINING_OR_SIMULATION,
        "training_or_simulation": TaskCategory.TRAINING_OR_SIMULATION,
        "experiment": TaskCategory.TRAINING_OR_SIMULATION,
        "ablation": TaskCategory.ABLATION_STUDY,
        "ablation_study": TaskCategory.ABLATION_STUDY,
        "eval": TaskCategory.METRIC_EVALUATION,
        "evaluation": TaskCategory.METRIC_EVALUATION,
        "metric_evaluation": TaskCategory.METRIC_EVALUATION,
        "metrics": TaskCategory.METRIC_EVALUATION,
        "benchmark": TaskCategory.METRIC_EVALUATION,
        "audit": TaskCategory.VERIFICATION_AUDIT,
        "verification": TaskCategory.VERIFICATION_AUDIT,
        "verification_audit": TaskCategory.VERIFICATION_AUDIT,
    }
    if raw in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[raw]

    combined = f"{t_dict.get('task_id', '')} {title_hint} {desc_hint}".lower()
    if any(k in combined for k in ["environment", "devops", "docker", "requirements", "cuda driver", "lockfile", "env_setup", "hardware"]):
        return TaskCategory.ENVIRONMENT_SETUP
    if any(k in combined for k in ["download dataset", "acquire data", "raw data", "data acquisition"]):
        return TaskCategory.DATA_ACQUISITION
    if any(k in combined for k in ["preprocess", "split", "tokeniz", "clean data", "normalize", "dataset", "data"]):
        return TaskCategory.DATA_PREPROCESSING
    if any(k in combined for k in ["architecture", "loss function", "forward pass", "neural", "kernel", "layer", "model"]):
        return TaskCategory.MODEL_IMPLEMENTATION
    if any(k in combined for k in ["smoke", "sanity", "gradient check"]):
        return TaskCategory.SANITY_SMOKE_TEST
    if any(k in combined for k in ["train", "simulat", "optimizer", "epoch", "hyperparameter", "sweep", "experiment"]):
        return TaskCategory.TRAINING_OR_SIMULATION
    if any(k in combined for k in ["ablation", "sensitivity"]):
        return TaskCategory.ABLATION_STUDY
    if any(k in combined for k in ["metric", "evaluat", "benchmark", "accuracy", "f1", "test set", "table", "results"]):
        return TaskCategory.METRIC_EVALUATION
    if any(k in combined for k in ["audit", "verif", "reproduc", "replicat", "scorecard", "claim", "score"]):
        return TaskCategory.VERIFICATION_AUDIT

    return TaskCategory.MODEL_IMPLEMENTATION


def _resolve_role(t_dict: Dict[str, Any], category: TaskCategory, title_hint: str = "", desc_hint: str = "") -> AgentRole:
    """Robustly resolves AgentRole from LLM outputs handling aliases, case, and category defaults."""
    raw = (
        t_dict.get("assigned_role")
        or t_dict.get("role")
        or t_dict.get("agent")
        or t_dict.get("agent_role")
        or t_dict.get("assigned_agent")
        or ""
    ).strip().lower().replace("-", "_").replace(" ", "_")

    for r in AgentRole:
        if r.value == raw:
            return r

    ROLE_ALIASES = {
        "devops": AgentRole.DEVOPS_AGENT,
        "devops_agent": AgentRole.DEVOPS_AGENT,
        "sysadmin": AgentRole.DEVOPS_AGENT,
        "infrastructure": AgentRole.DEVOPS_AGENT,
        "data": AgentRole.DATA_ENGINEER,
        "data_engineer": AgentRole.DATA_ENGINEER,
        "data_engineer_agent": AgentRole.DATA_ENGINEER,
        "model": AgentRole.MODEL_ARCHITECT,
        "model_architect": AgentRole.MODEL_ARCHITECT,
        "model_architect_agent": AgentRole.MODEL_ARCHITECT,
        "architect": AgentRole.MODEL_ARCHITECT,
        "experiment": AgentRole.EXPERIMENT_RUNNER,
        "experiment_runner": AgentRole.EXPERIMENT_RUNNER,
        "experiment_runner_agent": AgentRole.EXPERIMENT_RUNNER,
        "trainer": AgentRole.EXPERIMENT_RUNNER,
        "eval": AgentRole.EVALUATION_AGENT,
        "evaluation": AgentRole.EVALUATION_AGENT,
        "evaluation_agent": AgentRole.EVALUATION_AGENT,
        "evaluator": AgentRole.EVALUATION_AGENT,
        "audit": AgentRole.AUDIT_AGENT,
        "audit_agent": AgentRole.AUDIT_AGENT,
        "auditor": AgentRole.AUDIT_AGENT,
        "verifier": AgentRole.AUDIT_AGENT,
    }
    if raw in ROLE_ALIASES:
        return ROLE_ALIASES[raw]

    CATEGORY_ROLE_MAP = {
        TaskCategory.ENVIRONMENT_SETUP: AgentRole.DEVOPS_AGENT,
        TaskCategory.DATA_ACQUISITION: AgentRole.DATA_ENGINEER,
        TaskCategory.DATA_PREPROCESSING: AgentRole.DATA_ENGINEER,
        TaskCategory.MODEL_IMPLEMENTATION: AgentRole.MODEL_ARCHITECT,
        TaskCategory.SANITY_SMOKE_TEST: AgentRole.MODEL_ARCHITECT,
        TaskCategory.TRAINING_OR_SIMULATION: AgentRole.EXPERIMENT_RUNNER,
        TaskCategory.ABLATION_STUDY: AgentRole.EXPERIMENT_RUNNER,
        TaskCategory.METRIC_EVALUATION: AgentRole.EVALUATION_AGENT,
        TaskCategory.VERIFICATION_AUDIT: AgentRole.AUDIT_AGENT,
    }
    return CATEGORY_ROLE_MAP.get(category, AgentRole.MODEL_ARCHITECT)


def _resolve_title(t_dict: Dict[str, Any], category: TaskCategory, desc: str = "") -> str:
    """Robustly extracts or synthesizes a meaningful, non-generic title."""
    raw_title = (
        t_dict.get("title")
        or t_dict.get("name")
        or t_dict.get("task_name")
        or t_dict.get("task")
        or t_dict.get("summary")
        or t_dict.get("action")
        or t_dict.get("step")
        or ""
    ).strip()

    if not raw_title or raw_title.lower() in ("execution task", "task", "step", "todo", "unnamed task"):
        CATEGORY_CANONICAL_TITLES = {
            TaskCategory.ENVIRONMENT_SETUP: "Provision Environment & Lock Dependencies",
            TaskCategory.DATA_ACQUISITION: "Acquire Raw Scientific Datasets & Benchmarks",
            TaskCategory.DATA_PREPROCESSING: "Preprocess Data & Generate Canonical Splits",
            TaskCategory.MODEL_IMPLEMENTATION: "Implement Core Model Architecture & Algorithms",
            TaskCategory.SANITY_SMOKE_TEST: "Sanity Smoke-Test Forward Pass & Gradient Flow",
            TaskCategory.TRAINING_OR_SIMULATION: "Execute Training & Simulation Protocol",
            TaskCategory.ABLATION_STUDY: "Execute Ablation Studies & Hyperparameter Sweeps",
            TaskCategory.METRIC_EVALUATION: "Compute Benchmark Metrics & Evaluate Results",
            TaskCategory.VERIFICATION_AUDIT: "Replication Verification & Target Audit",
        }
        if category in CATEGORY_CANONICAL_TITLES:
            return CATEGORY_CANONICAL_TITLES[category]
        if desc:
            first_sentence = desc.split(".")[0].strip()
            if len(first_sentence) > 10:
                return first_sentence[:60]
        return "Implement Scientific Replication Step"

    return raw_title


class ProposerCriticEngine:
    """
    Orchestrates the multi-turn Proposer <-> Critic refinement loop.
    Iterates up to max_turns (default 3), terminating early when 0 blocking flaws exist.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        chat_fn: Optional[Any] = None,
        full_phases: bool = True,
    ):
        self.model = model or LLM_MODEL
        self.chat_fn = chat_fn or chat
        self.full_phases = full_phases
        self.macro_critic = MacroCritic()
        self.micro_critic = MicroCritic()
        self.critic = TaskCritic()

    def _parse_llm_json(self, raw_text: str, paper_id: str, paper_title: str) -> Optional[TaskGraph]:
        """Safely parses LLM JSON string into a TaskGraph."""
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
        except Exception as e:
            logger.debug("Failed to parse JSON from LLM: %s", e)
            return None

        tasks_data = parsed.get("tasks", [])
        if not tasks_data or not isinstance(tasks_data, list):
            return None

        graph = TaskGraph(paper_id=paper_id, paper_title=parsed.get("paper_title") or paper_title)
        for t_dict in tasks_data:
            try:
                title_hint = t_dict.get("title") or t_dict.get("name") or t_dict.get("task_name") or ""
                desc_hint = t_dict.get("description", "")
                category = _resolve_category(t_dict, title_hint=title_hint, desc_hint=desc_hint)
                role = _resolve_role(t_dict, category=category, title_hint=title_hint, desc_hint=desc_hint)
                title = _resolve_title(t_dict, category=category, desc=desc_hint)

                contracts = []
                if t_dict.get("artifact_contracts"):
                    for c_raw in t_dict["artifact_contracts"]:
                        if isinstance(c_raw, dict):
                            contracts.append(ArtifactContract.from_dict(c_raw))
                        elif isinstance(c_raw, ArtifactContract):
                            contracts.append(c_raw)
                else:
                    for out in t_dict.get("outputs", []):
                        if out.endswith(".json"):
                            contracts.append(ArtifactContract(path=out, artifact_type=ArtifactType.METRICS_JSON, required_keys=["accuracy", "metric"]))
                        elif out.endswith(".csv"):
                            contracts.append(ArtifactContract(path=out, artifact_type=ArtifactType.TABULAR, columns={"id": "str"}))
                        elif out.endswith(".py"):
                            contracts.append(ArtifactContract(path=out, artifact_type=ArtifactType.CODE))
                        elif out.endswith((".npy", ".npz", ".pt", ".h5")):
                            contracts.append(ArtifactContract(path=out, artifact_type=ArtifactType.TENSOR, shape=["N", "D"], dtype="float32"))
                        elif out.endswith(".lock"):
                            contracts.append(ArtifactContract(path=out, artifact_type=ArtifactType.LOCKFILE))

                task = Task(
                    task_id=t_dict.get("task_id", f"task_{len(graph.tasks)+1}"),
                    title=title,
                    category=category,
                    assigned_role=role,
                    dependencies=t_dict.get("dependencies", []),
                    description=t_dict.get("description", ""),
                    inputs=t_dict.get("inputs", []),
                    outputs=t_dict.get("outputs", []),
                    acceptance_criteria=t_dict.get("acceptance_criteria", []),
                    paper_context_excerpts=t_dict.get("paper_context_excerpts", []),
                    tools_required=t_dict.get("tools_required", []),
                    artifact_contracts=contracts,
                )
                task.test_harness_script = generate_task_test_script(task)
                graph.add_task(task)
            except Exception as e:
                logger.warning("Error parsing task dictionary %s: %s", t_dict, e)

        return graph

    def _format_context_text(self, context: Dict[str, Any]) -> str:
        """Formats planning context including un-truncated faceted context if present."""
        if context.get("formatted_context"):
            return context["formatted_context"]

        parts = []
        if context.get("abstract"):
            parts.append(f"Abstract:\n{context['abstract']}\n")
        if context.get("sections"):
            parts.append("\n".join(context["sections"]))
        if context.get("tables"):
            parts.append("\nKey Figures / Tables:\n" + "\n".join(context["tables"]))
        return "\n\n".join(parts) if parts else "Scientific methodology and replication details."

    def _synthesize_turn(
        self,
        paper_id: str,
        paper_title: str,
        context: Dict[str, Any],
        turn: int,
        previous_scorecard: Optional[RubricScorecard] = None,
    ) -> TaskGraph:
        """
        Synthesizes a proposed TaskGraph for the current turn.
        On turn 1, uses initial deconstruction prompt.
        On turn >= 2, feeds previous scorecard flaws back for targeted refinement.
        """
        paper_text = self._format_context_text(context)

        if turn == 1 or previous_scorecard is None:
            prompt = PROTOCOL_DECONSTRUCTION_USER.format(
                paper_title=paper_title,
                paper_text=paper_text,
            )
            messages = [
                {"role": "system", "content": PROTOCOL_DECONSTRUCTION_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        else:
            # Format critique flaws for feedback
            flaw_lines = []
            for f in previous_scorecard.flaws:
                prefix = "🔴 BLOCKING" if f.severity == FlawSeverity.BLOCKING.value else "🟡 WARNING"
                target = f" [Task: {f.task_id}]" if f.task_id else ""
                flaw_lines.append(f"- {prefix}{target} {f.category}: {f.description}\n  -> Fix: {f.suggestion}")
            flaws_report = "\n".join(flaw_lines) if flaw_lines else "No specific flaws logged."

            user_prompt = REFINEMENT_PROMPT_TEMPLATE.format(
                turn=turn - 1,
                flaws_report=flaws_report,
                paper_title=paper_title,
                paper_text=paper_text,
            )
            messages = [
                {"role": "system", "content": PROTOCOL_DECONSTRUCTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]

        try:
            res = self.chat_fn(messages, model=self.model, temperature=0.1, options={"num_ctx": 4096})
            raw = res.content.strip() if hasattr(res, "content") else str(res).strip()
            graph = self._parse_llm_json(raw, paper_id, paper_title)
            if graph and len(graph.tasks) > 0:
                return graph
        except Exception as exc:
            logger.warning("LLM synthesis turn %d failed: %s", turn, exc)

        # If LLM failed, fallback to heuristic graph according to self.full_phases
        from research_assistant.tasks.extractor import _heuristic_task_graph
        return _heuristic_task_graph(paper_id, paper_title, context, full_phases=self.full_phases)

    def _heal_remaining_flaws(self, graph: TaskGraph, scorecard: RubricScorecard, context: Dict[str, Any]) -> TaskGraph:
        """
        Applies deterministic DAG healing if blocking flaws persist after max_turns:
        1. Severs circular dependency feedback edges.
        2. Heals generic titles and assigns distinct specialized agent roles.
        3. Decouples completely serialized chains to enable parallel execution waves.
        4. Injects missing mandatory scientific phases when full_phases is True.
        """
        if not self.full_phases:
            return graph
        # 1. Sever circular dependencies if present
        errors = graph.validate()
        for err in errors:
            if "Cycle detected:" in err:
                parts = err.split("Cycle detected: ")[-1].split(" -> ")
                if len(parts) >= 2:
                    child = parts[-2]
                    parent = parts[-1]
                    if child in graph.tasks and parent in graph.tasks[child].dependencies:
                        graph.tasks[child].dependencies.remove(parent)
                        logger.info("Severed cyclic dependency edge %s -> %s", child, parent)

        # 2. Heal generic titles and uniform single-role assignments
        for task in graph.tasks.values():
            if not task.title or task.title.strip() == "Execution Task":
                task.title = _resolve_title({"title": ""}, task.category, task.description)

        roles = {t.assigned_role for t in graph.tasks.values()}
        if len(graph.tasks) >= 3 and len(roles) <= 1:
            for task in graph.tasks.values():
                task.assigned_role = _resolve_role({}, task.category, task.title, task.description)

        # 3. Decouple completely serialized linear chains if model and data tasks are chained
        if len(graph.tasks) >= 4:
            try:
                waves = graph.topological_waves()
                if len(waves) == len(graph.tasks):
                    env_tasks = [tid for tid, t in graph.tasks.items() if t.category == TaskCategory.ENVIRONMENT_SETUP]
                    data_tasks = [tid for tid, t in graph.tasks.items() if t.category in (TaskCategory.DATA_ACQUISITION, TaskCategory.DATA_PREPROCESSING)]
                    if env_tasks and data_tasks:
                        for task in graph.tasks.values():
                            if task.category == TaskCategory.MODEL_IMPLEMENTATION:
                                for dt in data_tasks:
                                    if dt in task.dependencies:
                                        task.dependencies.remove(dt)
                                        if env_tasks[0] not in task.dependencies:
                                            task.dependencies.append(env_tasks[0])
            except Exception as e:
                logger.debug("Error decoupling serial waves during healing: %s", e)

        # 4. Fill missing mandatory phases if any (when full_phases is True)
        if self.full_phases:
            missing = graph.get_missing_phases()
            if missing:
                canonical = _build_heuristic_7phase_fallback(graph.paper_id, graph.paper_title, context)
                phase_to_task_id = {
                    "environment_setup": "task_01_env",
                    "data_splits": "task_02_data",
                    "sanity_smoke_test": "task_04_smoke",
                    "training_or_simulation": "task_05_train",
                    "metric_evaluation": "task_06_eval",
                    "verification_audit": "task_07_audit",
                }
                for m_phase in missing:
                    t_id = phase_to_task_id.get(m_phase)
                    if t_id and t_id in canonical.tasks:
                        c_task = canonical.tasks[t_id]
                        valid_deps = [d for d in c_task.dependencies if d in graph.tasks]
                        c_task.dependencies = valid_deps
                        graph.add_task(c_task)
                        logger.info("Injected missing mandatory phase task: %s (%s)", t_id, m_phase)

        return graph

    def run_refinement_loop(
        self,
        paper_id: str,
        paper_title: str,
        context: Dict[str, Any],
        max_turns: int = 3,
    ) -> Tuple[TaskGraph, List[RubricScorecard]]:
        """
        Executes Proposer <-> Critic refinement loop.
        Terminates early on 0 blocking flaws. Capped at max_turns.
        Returns the finalized TaskGraph and list of RubricScorecards per turn.
        """
        scorecards: List[RubricScorecard] = []
        critique_trace: List[Dict[str, Any]] = []
        current_graph: Optional[TaskGraph] = None
        last_scorecard: Optional[RubricScorecard] = None

        effective_max_turns = max(1, min(max_turns, 3))

        for turn in range(1, effective_max_turns + 1):
            logger.info("Executing Proposer-Critic Iteration Turn %d / %d for '%s'", turn, effective_max_turns, paper_title)

            current_graph = self._synthesize_turn(
                paper_id=paper_id,
                paper_title=paper_title,
                context=context,
                turn=turn,
                previous_scorecard=last_scorecard,
            )

            # Evaluate proposed graph with Critic
            scorecard = self.critic.evaluate(current_graph)
            scorecards.append(scorecard)
            last_scorecard = scorecard

            trace_entry = {
                "turn": turn,
                "topology_score": scorecard.topology_score,
                "contract_score": scorecard.contract_score,
                "testability_score": scorecard.testability_score,
                "overall_score": scorecard.overall_score,
                "blocking_flaws_count": scorecard.blocking_flaws_count,
                "warning_flaws_count": scorecard.warning_flaws_count,
                "flaws": [f.to_dict() for f in scorecard.flaws],
            }
            critique_trace.append(trace_entry)

            logger.info(
                "Turn %d Scorecard: Overall=%.1f, Blocking=%d, Warnings=%d (Passing=%s)",
                turn,
                scorecard.overall_score,
                scorecard.blocking_flaws_count,
                scorecard.warning_flaws_count,
                scorecard.is_passing,
            )

            # Early Exit: 0 blocking flaws found
            if scorecard.is_passing:
                logger.info("Proposer-Critic converged on Turn %d with 0 blocking flaws.", turn)
                break

        # If blocking flaws remain after max_turns, heal remaining flaws
        if current_graph and last_scorecard and not last_scorecard.is_passing:
            logger.info("Applying deterministic graph healing after turn %d.", len(scorecards))
            current_graph = self._heal_remaining_flaws(current_graph, last_scorecard, context)
            final_scorecard = self.critic.evaluate(current_graph)
            if scorecards:
                scorecards[-1] = final_scorecard
            else:
                scorecards.append(final_scorecard)
            critique_trace.append({
                "turn": len(scorecards),
                "action": "deterministic_healing",
                "topology_score": final_scorecard.topology_score,
                "contract_score": final_scorecard.contract_score,
                "testability_score": final_scorecard.testability_score,
                "overall_score": final_scorecard.overall_score,
                "blocking_flaws_count": final_scorecard.blocking_flaws_count,
                "warning_flaws_count": final_scorecard.warning_flaws_count,
                "flaws": [f.to_dict() for f in final_scorecard.flaws],
            })
            last_scorecard = final_scorecard

        if current_graph:
            current_graph.metadata["critique_trace"] = critique_trace
            current_graph.metadata["scorecard"] = last_scorecard.to_dict() if last_scorecard else {}

        return current_graph or _build_heuristic_7phase_fallback(paper_id, paper_title, context), scorecards
