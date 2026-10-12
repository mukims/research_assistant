"""
Two-Level Hierarchical Critic Engine and Quantitative Rubric Scorecards.

Implements MacroCritic (graph topology, 6 mandatory scientific phases, wave parallelizability)
and MicroCritic (task instruction specificity, artifact contracts, testable criteria, CPU thread safety)
to rigorously evaluate scientific replication workflows against quantitative rubrics (0-100).
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import os
import re
from typing import Any, Dict, List, Optional

from research_assistant.shared.log import get_logger
from research_assistant.tasks.models import (
    AgentRole,
    ArtifactType,
    Task,
    TaskCategory,
    TaskGraph,
)

logger = get_logger("task_critic")


class FlawSeverity(str, Enum):
    """Severity classification of critique flaws."""
    BLOCKING = "blocking"
    WARNING = "warning"


@dataclass
class CritiqueFlaw:
    """A concrete flaw identified during macro or micro critique."""
    flaw_id: str
    task_id: Optional[str]
    critic_level: str       # "macro" | "micro"
    severity: str           # "blocking" | "warning"
    category: str           # "topology" | "missing_phase" | "vague_instruction" | "contract" | "testability" | "cpu_oversubscription" | "wave"
    description: str
    suggestion: str

    @property
    def message(self) -> str:
        """Alias for description."""
        return self.description

    @property
    def remediation_instruction(self) -> str:
        """Alias for suggestion."""
        return self.suggestion

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RubricScorecard:
    """Quantitative evaluation scorecard across topology, contracts, and testability."""
    topology_score: float      # 0-100
    contract_score: float      # 0-100
    testability_score: float   # 0-100
    overall_score: float       # 0-100
    blocking_flaws_count: int
    warning_flaws_count: int
    flaws: List[CritiqueFlaw] = field(default_factory=list)

    @property
    def is_passing(self) -> bool:
        """A scorecard passes if and only if there are zero blocking flaws."""
        return self.blocking_flaws_count == 0

    @property
    def blocking_count(self) -> int:
        return self.blocking_flaws_count

    @property
    def warning_count(self) -> int:
        return self.warning_flaws_count

    @property
    def warnings(self) -> List[CritiqueFlaw]:
        return [f for f in self.flaws if f.severity == FlawSeverity.WARNING.value]

    @property
    def blocking_flaws(self) -> List[CritiqueFlaw]:
        return [f for f in self.flaws if f.severity == FlawSeverity.BLOCKING.value]

    def render_markdown(self) -> str:
        """Renders scorecard into clean Markdown table and summary."""
        lines = [
            "### Replication Workflow Rubric Scorecard",
            f"**Overall Composite Score**: `{self.overall_score:.1f} / 100` | **Status**: `{'PASSING (0 blocking flaws)' if self.is_passing else 'FAILING (blocking flaws detected)'}`",
            "",
            "| Rubric Dimension | Score | Status |",
            "|---|---|---|",
            f"| **DAG Topology & 6 Scientific Phases** | `{self.topology_score:.1f} / 100` | {'✅ Pass' if self.topology_score >= 80 else '⚠️ Review'} |",
            f"| **Artifact Deliverables & Contracts** | `{self.contract_score:.1f} / 100` | {'✅ Pass' if self.contract_score >= 80 else '⚠️ Review'} |",
            f"| **Deterministic Testability & CPU Bounds** | `{self.testability_score:.1f} / 100` | {'✅ Pass' if self.testability_score >= 80 else '⚠️ Review'} |",
            "",
            f"**Blocking Flaws**: `{self.blocking_flaws_count}` | **Warnings**: `{self.warning_flaws_count}`",
        ]
        if self.flaws:
            lines.append("\n#### Flaws Identified & Actionable Remediation")
            for f in self.flaws:
                badge = "🔴 [BLOCKING]" if f.severity == FlawSeverity.BLOCKING.value else "🟡 [WARNING]"
                task_str = f" `{f.task_id}`" if f.task_id else ""
                lines.append(f"- {badge} **{f.category}**{task_str}: {f.description}")
                lines.append(f"  *Remediation*: {f.suggestion}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topology_score": self.topology_score,
            "contract_score": self.contract_score,
            "testability_score": self.testability_score,
            "overall_score": self.overall_score,
            "blocking_flaws_count": self.blocking_flaws_count,
            "warning_flaws_count": self.warning_flaws_count,
            "is_passing": self.is_passing,
            "flaws": [f.to_dict() for f in self.flaws],
        }


# 6 Mandatory Scientific Phases definitions
MANDATORY_PHASES = {
    "environment_setup": "Environment Setup & Hardware Dependency Locking",
    "data_splits": "Data Acquisition, Preprocessing & Splits",
    "sanity_smoke_test": "Sanity Smoke-Test Forward Pass & Gradient Check",
    "training_or_simulation": "Core Training, Optimization, or Simulation",
    "metric_evaluation": "Held-Out Metric Evaluation & Target Comparison",
    "verification_audit": "Artifact Certification & Replication Scorecard Audit",
}

# Subjective words that invalidate acceptance criteria testability
SUBJECTIVE_WORDS = [
    "look good", "looks good", "perform nicely", "performs nicely",
    "as expected", "good performance", "nice", "looks nice",
    "should work", "decent results", "reasonable", "pretty good",
]

# Unsafe thread patterns indicating oversubscription risks
UNSAFE_THREAD_PATTERNS = [
    "without thread limits", "unbounded thread", "num_thread: 16",
    "threads: 16", "threads=16", "all cores", "maximum threads",
    "no thread limit", "unlimited threads",
]

# Positive thread safety indicators
SAFE_THREAD_INDICATORS = [
    "thread", "omp_num_threads", "torch.set_num_threads",
    "single-thread", "single thread", "workers=1", "num_workers",
    "cpu-safe", "thread cap", "thread limit",
]


class MacroCritic:
    """
    Evaluates global workflow structure:
    1. DAG topological ordering and absence of cycles (blocking).
    2. Presence of all 6 mandatory scientific phases (blocking).
    3. Execution wave parallelizability and structure (warning).
    """

    def evaluate(self, graph: TaskGraph) -> List[CritiqueFlaw]:
        flaws: List[CritiqueFlaw] = []

        if not graph or not graph.tasks:
            flaws.append(
                CritiqueFlaw(
                    flaw_id="macro_empty_graph",
                    task_id=None,
                    critic_level="macro",
                    severity=FlawSeverity.BLOCKING.value,
                    category="topology",
                    description="Task graph contains zero tasks.",
                    suggestion="Synthesize tasks covering the 6 mandatory scientific phases.",
                )
            )
            return flaws

        # 1. Topological ordering and cycle detection
        errors = graph.validate()
        for idx, err in enumerate(errors, 1):
            if "Cycle detected" in err:
                flaws.append(
                    CritiqueFlaw(
                        flaw_id=f"macro_cycle_{idx}",
                        task_id=None,
                        critic_level="macro",
                        severity=FlawSeverity.BLOCKING.value,
                        category="topology",
                        description=f"Circular dependency detected in task graph: {err}",
                        suggestion="Remove the feedback dependency edge to ensure a strictly acyclic DAG.",
                    )
                )
            elif "cannot depend on itself" in err:
                flaws.append(
                    CritiqueFlaw(
                        flaw_id=f"macro_self_dep_{idx}",
                        task_id=None,
                        critic_level="macro",
                        severity=FlawSeverity.BLOCKING.value,
                        category="topology",
                        description=err,
                        suggestion="Remove self-referential dependency edge.",
                    )
                )
            elif "depends on missing task" in err:
                flaws.append(
                    CritiqueFlaw(
                        flaw_id=f"macro_missing_dep_{idx}",
                        task_id=None,
                        critic_level="macro",
                        severity=FlawSeverity.BLOCKING.value,
                        category="topology",
                        description=err,
                        suggestion="Define the referenced dependency task or update the dependency list.",
                    )
                )

        # 2. Check 6 mandatory scientific phases
        missing_phases = graph.get_missing_phases()
        for phase_key in missing_phases:
            label = MANDATORY_PHASES.get(phase_key, phase_key)
            flaws.append(
                CritiqueFlaw(
                    flaw_id=f"macro_phase_{phase_key}",
                    task_id=None,
                    critic_level="macro",
                    severity=FlawSeverity.BLOCKING.value,
                    category="missing_phase",
                    description=f"Missing mandatory scientific phase: {label}.",
                    suggestion=f"Add a dedicated task representing the {label} phase with concrete inputs and outputs.",
                )
            )

        # 3. Wave parallelizability check (only if graph is acyclic)
        if not any(f.category == "topology" and f.severity == FlawSeverity.BLOCKING.value for f in flaws):
            try:
                waves = graph.topological_waves()
                total_tasks = len(graph.tasks)
                # If there are >= 4 tasks and depth equals total tasks, everything is completely serialized
                if total_tasks >= 4 and len(waves) == total_tasks:
                    flaws.append(
                        CritiqueFlaw(
                            flaw_id="macro_wave_serialization",
                            task_id=None,
                            critic_level="macro",
                            severity=FlawSeverity.WARNING.value,
                            category="wave",
                            description="All tasks are strictly serialized; independent steps could run concurrently.",
                            suggestion="Check if data preprocessing and model architecture tasks can execute in parallel in Wave 1.",
                        )
                    )
            except Exception as e:
                logger.debug("Wave calculation in MacroCritic encountered: %s", e)

        return flaws

    def compute_topology_score(self, graph: TaskGraph, flaws: List[CritiqueFlaw]) -> float:
        """Computes 0-100 topology score."""
        if not graph or not graph.tasks:
            return 0.0

        score = 100.0
        # Deduct heavily for cycles and missing dependencies
        cycle_flaws = [f for f in flaws if f.category == "topology" and "Cycle" in f.description]
        other_topo_flaws = [f for f in flaws if f.category == "topology" and "Cycle" not in f.description]
        missing_phase_flaws = [f for f in flaws if f.category == "missing_phase"]

        score -= len(cycle_flaws) * 40.0
        score -= len(other_topo_flaws) * 20.0
        score -= len(missing_phase_flaws) * 15.0

        # Small deduction for wave serialization warning
        wave_flaws = [f for f in flaws if f.category == "wave"]
        score -= len(wave_flaws) * 5.0

        return max(0.0, min(100.0, score))


class MicroCritic:
    """
    Evaluates individual tasks:
    1. Task instruction specificity (flags descriptions under 60 chars or lacking detail).
    2. Input/output formal artifact deliverables (flags missing deliverables or extensions).
    3. Deterministic acceptance criteria (flags missing or subjective criteria).
    4. CPU thread oversubscription risks (flags training tasks without thread bounds).
    """

    def evaluate(self, graph: TaskGraph) -> List[CritiqueFlaw]:
        flaws: List[CritiqueFlaw] = []
        if not graph or not graph.tasks:
            return flaws

        for task in graph.tasks.values():
            flaws.extend(self.scrutinize_task(task))
        return flaws

    def scrutinize_task(self, task: Task) -> List[CritiqueFlaw]:
        flaws: List[CritiqueFlaw] = []
        t_id = task.task_id

        # 0. Title Quality
        title = (task.title or "").strip()
        if not title or title.lower() in ("execution task", "unnamed task", "placeholder task"):
            flaws.append(
                CritiqueFlaw(
                    flaw_id=f"micro_title_generic_{t_id}",
                    task_id=t_id,
                    critic_level="micro",
                    severity=FlawSeverity.BLOCKING.value,
                    category="generic_title",
                    description=f"Task '{t_id}' has a generic placeholder title ('{task.title}').",
                    suggestion="Provide a concrete, imperative title representing the task's scientific objective.",
                )
            )

        # 1. Instruction Specificity
        desc = (task.description or "").strip()
        if not desc:
            flaws.append(
                CritiqueFlaw(
                    flaw_id=f"micro_desc_empty_{t_id}",
                    task_id=t_id,
                    critic_level="micro",
                    severity=FlawSeverity.BLOCKING.value,
                    category="vague_instruction",
                    description=f"Task '{t_id}' has an empty description lacking operational instructions.",
                    suggestion="Add concrete, step-by-step technical instructions (minimum 60 characters).",
                )
            )
        elif len(desc) < 60:
            flaws.append(
                CritiqueFlaw(
                    flaw_id=f"micro_desc_vague_{t_id}",
                    task_id=t_id,
                    critic_level="micro",
                    severity=FlawSeverity.WARNING.value,
                    category="vague_instruction",
                    description=f"Task '{t_id}' description is vague or too brief ({len(desc)} characters; minimum 60 recommended).",
                    suggestion="Elaborate on operational parameters, mathematical formulations, or tools required.",
                )
            )

        # 2. Output Deliverables & Artifact Contracts
        contracts = getattr(task, "artifact_contracts", []) or []
        has_outputs = bool(task.outputs) or bool(contracts)
        if not has_outputs:
            flaws.append(
                CritiqueFlaw(
                    flaw_id=f"micro_contract_empty_{t_id}",
                    task_id=t_id,
                    critic_level="micro",
                    severity=FlawSeverity.BLOCKING.value,
                    category="contract",
                    description=f"Task '{t_id}' produces no output deliverables or artifact contracts.",
                    suggestion="Define explicit output file deliverables (e.g. data/processed.pt, src/model.py, metrics/eval.json).",
                )
            )
        else:
            # Check for file extension completeness on raw outputs
            for out in task.outputs:
                if out and "." not in out.split("/")[-1]:
                    flaws.append(
                        CritiqueFlaw(
                            flaw_id=f"micro_contract_no_ext_{t_id}",
                            task_id=t_id,
                            critic_level="micro",
                            severity=FlawSeverity.WARNING.value,
                            category="contract",
                            description=f"Task '{t_id}' output deliverable '{out}' lacks an explicit file extension (.pt, .h5, .json, .csv, .lock).",
                            suggestion=f"Provide a concrete file extension for '{out}' to enable deterministic schema verification.",
                        )
                    )

            # Check if task defines outputs but completely lacks formal artifact contracts
            if task.outputs and not contracts:
                flaws.append(
                    CritiqueFlaw(
                        flaw_id=f"micro_contract_missing_schema_{t_id}",
                        task_id=t_id,
                        critic_level="micro",
                        severity=FlawSeverity.WARNING.value,
                        category="contract",
                        description=f"Task '{t_id}' defines output deliverables {task.outputs} without formal ArtifactContract schemas.",
                        suggestion="Define formal ArtifactContract specifications (artifact_type, tensor shapes, dtypes, or columns).",
                    )
                )

            # Deep scrutiny of individual ArtifactContract instances
            for idx, c in enumerate(contracts, 1):
                c_path = getattr(c, "path", "")
                if not c_path or not str(c_path).strip():
                    flaws.append(
                        CritiqueFlaw(
                            flaw_id=f"micro_contract_empty_path_{t_id}_{idx}",
                            task_id=t_id,
                            critic_level="micro",
                            severity=FlawSeverity.BLOCKING.value,
                            category="contract",
                            description=f"Artifact contract #{idx} in task '{t_id}' specifies an empty file path.",
                            suggestion="Provide a non-empty relative path for the deliverable.",
                        )
                    )
                    continue

                if "." not in os.path.basename(str(c_path)):
                    flaws.append(
                        CritiqueFlaw(
                            flaw_id=f"micro_contract_path_no_ext_{t_id}_{idx}",
                            task_id=t_id,
                            critic_level="micro",
                            severity=FlawSeverity.WARNING.value,
                            category="contract",
                            description=f"Artifact contract deliverable '{c_path}' in task '{t_id}' lacks a file extension.",
                            suggestion=f"Specify an explicit extension (.pt, .npy, .csv, .json) for contract path '{c_path}'.",
                        )
                    )

                c_type = getattr(c, "artifact_type", None)
                # Tensor contract checks: shape and dtype
                if c_type == ArtifactType.TENSOR or str(c_path).endswith((".pt", ".pth", ".npy", ".npz", ".h5", ".safetensors")):
                    if c.shape is None:
                        flaws.append(
                            CritiqueFlaw(
                                flaw_id=f"micro_contract_tensor_no_shape_{t_id}_{idx}",
                                task_id=t_id,
                                critic_level="micro",
                                severity=FlawSeverity.WARNING.value,
                                category="contract",
                                description=f"Tensor contract '{c_path}' in task '{t_id}' lacks explicit shape specification.",
                                suggestion=f"Specify expected tensor dimensions in contract.shape (e.g. ['batch_size', 'channels', 224, 224]).",
                            )
                        )
                    if c.dtype is None:
                        flaws.append(
                            CritiqueFlaw(
                                flaw_id=f"micro_contract_tensor_no_dtype_{t_id}_{idx}",
                                task_id=t_id,
                                critic_level="micro",
                                severity=FlawSeverity.WARNING.value,
                                category="contract",
                                description=f"Tensor contract '{c_path}' in task '{t_id}' lacks explicit dtype specification.",
                                suggestion=f"Specify numerical data type in contract.dtype (e.g. 'float32', 'int64').",
                            )
                        )

                # Tabular contract checks: columns
                elif c_type == ArtifactType.TABULAR or str(c_path).endswith((".csv", ".tsv", ".parquet")):
                    if not c.columns:
                        flaws.append(
                            CritiqueFlaw(
                                flaw_id=f"micro_contract_tabular_no_columns_{t_id}_{idx}",
                                task_id=t_id,
                                critic_level="micro",
                                severity=FlawSeverity.WARNING.value,
                                category="contract",
                                description=f"Tabular contract '{c_path}' in task '{t_id}' lacks column dictionary schema specification.",
                                suggestion="Define expected column names and types in contract.columns (e.g. {'epoch': 'int', 'loss': 'float'}).",
                            )
                        )

                # Metrics JSON checks: required_keys
                elif c_type == ArtifactType.METRICS_JSON or (str(c_path).endswith(".json") and c_type != ArtifactType.CODE):
                    if not c.required_keys:
                        flaws.append(
                            CritiqueFlaw(
                                flaw_id=f"micro_contract_json_no_keys_{t_id}_{idx}",
                                task_id=t_id,
                                critic_level="micro",
                                severity=FlawSeverity.WARNING.value,
                                category="contract",
                                description=f"Metrics JSON contract '{c_path}' in task '{t_id}' lacks required_keys specification.",
                                suggestion="Define required metric keys in contract.required_keys (e.g. ['accuracy', 'loss', 'f1']).",
                            )
                        )

                # CLI interface checks: cli_args
                elif c_type == ArtifactType.CLI_INTERFACE:
                    if not c.cli_args:
                        flaws.append(
                            CritiqueFlaw(
                                flaw_id=f"micro_contract_cli_no_args_{t_id}_{idx}",
                                task_id=t_id,
                                critic_level="micro",
                                severity=FlawSeverity.WARNING.value,
                                category="contract",
                                description=f"CLI interface contract '{c_path}' in task '{t_id}' lacks expected cli_args specification.",
                                suggestion="Define required argument flags in contract.cli_args (e.g. ['--batch-size', '--lr']).",
                            )
                        )

        # 3. Deterministic Acceptance Criteria
        criteria = task.acceptance_criteria or []
        if not criteria:
            flaws.append(
                CritiqueFlaw(
                    flaw_id=f"micro_testability_empty_{t_id}",
                    task_id=t_id,
                    critic_level="micro",
                    severity=FlawSeverity.BLOCKING.value,
                    category="testability",
                    description=f"Task '{t_id}' has no acceptance criteria defined.",
                    suggestion="Define automated, pass/fail acceptance criteria (e.g. 'Exit code 0', 'Accuracy >= 0.85').",
                )
            )
        else:
            for idx, c in enumerate(criteria, 1):
                c_lower = c.lower()
                if any(w in c_lower for w in SUBJECTIVE_WORDS):
                    flaws.append(
                        CritiqueFlaw(
                            flaw_id=f"micro_testability_subjective_{t_id}_{idx}",
                            task_id=t_id,
                            critic_level="micro",
                            severity=FlawSeverity.WARNING.value,
                            category="testability",
                            description=f"Task '{t_id}' contains subjective or untestable acceptance criterion: '{c}'.",
                            suggestion="Replace subjective phrasing with measurable thresholds or programmatic assertions.",
                        )
                    )

        # 4. CPU Thread Oversubscription / Resource Limits
        is_training = (
            task.category in (TaskCategory.TRAINING_OR_SIMULATION, TaskCategory.ABLATION_STUDY)
            or task.assigned_role == AgentRole.EXPERIMENT_RUNNER
            or "train" in t_id.lower()
            or "train" in task.title.lower()
        )
        if is_training:
            desc_lower = desc.lower()
            if any(p in desc_lower for p in UNSAFE_THREAD_PATTERNS):
                flaws.append(
                    CritiqueFlaw(
                        flaw_id=f"micro_thread_unsafe_{t_id}",
                        task_id=t_id,
                        critic_level="micro",
                        severity=FlawSeverity.BLOCKING.value,
                        category="cpu_oversubscription",
                        description=f"Task '{t_id}' specifies unbounded CPU threads, risking thread oversubscription and CPU lockup.",
                        suggestion="Enforce single-threaded execution (OMP_NUM_THREADS=1, torch.set_num_threads(1), num_workers <= 2).",
                    )
                )
            elif not any(s in desc_lower for s in SAFE_THREAD_INDICATORS):
                # If it mentions training or torch without explicit thread caps, emit warning
                if "torch" in desc_lower or "train" in desc_lower or "fit" in desc_lower or "epoch" in desc_lower:
                    flaws.append(
                        CritiqueFlaw(
                            flaw_id=f"micro_thread_unspecified_{t_id}",
                            task_id=t_id,
                            critic_level="micro",
                            severity=FlawSeverity.WARNING.value,
                            category="cpu_oversubscription",
                            description=f"Task '{t_id}' performs training/simulation without documenting CPU thread caps.",
                            suggestion="Document CPU thread limits (e.g. torch.set_num_threads(1), OMP_NUM_THREADS=1) to prevent CPU starvation.",
                        )
                    )

        return flaws

    def compute_contract_score(self, graph: TaskGraph, flaws: List[CritiqueFlaw]) -> float:
        """Computes 0-100 contract score."""
        if not graph or not graph.tasks:
            return 0.0

        total_tasks = len(graph.tasks)
        score = 100.0
        contract_flaws = [f for f in flaws if f.category == "contract"]

        for f in contract_flaws:
            if f.severity == FlawSeverity.BLOCKING.value:
                score -= (100.0 / total_tasks)
            else:
                score -= (15.0 / total_tasks)

        return max(0.0, min(100.0, score))

    def compute_testability_score(self, graph: TaskGraph, flaws: List[CritiqueFlaw]) -> float:
        """Computes 0-100 testability and hardware guardrail score."""
        if not graph or not graph.tasks:
            return 0.0

        total_tasks = len(graph.tasks)
        score = 100.0

        for f in flaws:
            if f.category == "testability":
                if f.severity == FlawSeverity.BLOCKING.value:
                    score -= (80.0 / total_tasks)
                else:
                    score -= (25.0 / total_tasks)
            elif f.category == "cpu_oversubscription":
                if f.severity == FlawSeverity.BLOCKING.value:
                    score -= (50.0 / total_tasks)
                else:
                    score -= (15.0 / total_tasks)
            elif f.category == "vague_instruction":
                if f.severity == FlawSeverity.BLOCKING.value:
                    score -= (30.0 / total_tasks)
                else:
                    score -= (10.0 / total_tasks)

        return max(0.0, min(100.0, score))


class TaskCritic:
    """
    Combined critic orchestrator that runs MacroCritic and MicroCritic,
    computes quantitative rubric scores, and produces a RubricScorecard.
    """

    def __init__(self):
        self.macro_critic = MacroCritic()
        self.micro_critic = MicroCritic()

    def evaluate(self, graph: TaskGraph) -> RubricScorecard:
        """Evaluates graph against Macro and Micro rubrics, returning RubricScorecard."""
        macro_flaws = self.macro_critic.evaluate(graph)
        micro_flaws = self.micro_critic.evaluate(graph)
        all_flaws = macro_flaws + micro_flaws

        topo_score = self.macro_critic.compute_topology_score(graph, macro_flaws)
        contract_score = self.micro_critic.compute_contract_score(graph, micro_flaws)
        testability_score = self.micro_critic.compute_testability_score(graph, micro_flaws)

        overall_score = round((topo_score + contract_score + testability_score) / 3.0, 1)
        blocking_count = sum(1 for f in all_flaws if f.severity == FlawSeverity.BLOCKING.value)
        warning_count = sum(1 for f in all_flaws if f.severity == FlawSeverity.WARNING.value)

        return RubricScorecard(
            topology_score=round(topo_score, 1),
            contract_score=round(contract_score, 1),
            testability_score=round(testability_score, 1),
            overall_score=overall_score,
            blocking_flaws_count=blocking_count,
            warning_flaws_count=warning_count,
            flaws=all_flaws,
        )

    def evaluate_with_llm(self, graph: TaskGraph, model: Optional[str] = None) -> RubricScorecard:
        """
        Evaluates graph using LLM against the structured rubrics.
        Falls back to rule-based evaluate(graph) if LLM call fails or returns non-JSON.
        """
        import json
        from research_assistant.config import LLM_MODEL
        from research_assistant.shared.llm import chat

        prompt = (
            f"Evaluate this task decomposition graph for paper '{graph.paper_title}'.\n"
            f"Tasks: {[t.task_id for t in graph.tasks.values()]}.\n"
            "Return JSON with topology_score, contract_score, testability_score, overall_score, "
            "blocking_flaws_count, warning_flaws_count."
        )
        try:
            res = chat([{"role": "user", "content": prompt}], model=model or LLM_MODEL, options={"num_ctx": 4096})
            content = res.content.strip() if hasattr(res, "content") else str(res).strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            data = json.loads(content)
            return RubricScorecard(
                topology_score=float(data.get("topology_score", 100.0)),
                contract_score=float(data.get("contract_score", 100.0)),
                testability_score=float(data.get("testability_score", 100.0)),
                overall_score=float(data.get("overall_score", 100.0)),
                blocking_flaws_count=int(data.get("blocking_flaws_count", 0)),
                warning_flaws_count=int(data.get("warning_flaws_count", 0)),
                flaws=[],
            )
        except Exception as e:
            logger.debug("evaluate_with_llm failed or malformed JSON (%s); using rule-based fallback.", e)
            return self.evaluate(graph)


def evaluate_workflow(graph: TaskGraph) -> RubricScorecard:
    """Convenience helper function to evaluate a TaskGraph."""
    critic = TaskCritic()
    return critic.evaluate(graph)
