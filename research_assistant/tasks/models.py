"""
Data models and schemas for the Paper-to-Agent Task Decomposition Pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class TaskCategory(str, Enum):
    """Categorical classification of scientific execution tasks."""
    ENVIRONMENT_SETUP = "environment_setup"
    DATA_ACQUISITION = "data_acquisition"
    DATA_PREPROCESSING = "data_preprocessing"
    MODEL_IMPLEMENTATION = "model_implementation"
    SANITY_SMOKE_TEST = "sanity_smoke_test"
    SANITY_SMOKE_TESTS = "sanity_smoke_test"  # alias for plural form
    TRAINING_OR_SIMULATION = "training_or_simulation"
    ABLATION_STUDY = "ablation_study"
    METRIC_EVALUATION = "metric_evaluation"
    VERIFICATION_AUDIT = "verification_audit"


class TaskStatus(str, Enum):
    """Execution lifecycle status of an agent task."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


# Backward compatibility / semantic alias
TaskStatus.SUCCESS = TaskStatus.COMPLETED


class AgentRole(str, Enum):
    """Specialized agent personas for task execution."""
    DEVOPS_AGENT = "devops_agent"           # Environment, packages, runtime dependencies
    DATA_ENGINEER = "data_engineer"         # Datasets, downloads, preprocessing, splits
    MODEL_ARCHITECT = "model_architect"     # Models, algorithms, loss functions, math
    EXPERIMENT_RUNNER = "experiment_runner" # Training runs, simulations, sweeps
    EVALUATION_AGENT = "evaluation_agent"   # Metrics, benchmark evaluation, comparison tables
    AUDIT_AGENT = "audit_agent"             # Claim verification against paper targets


class ArtifactType(str, Enum):
    """Types of concrete scientific artifacts produced by tasks."""
    TENSOR = "tensor"
    TABULAR = "tabular"
    CODE = "code"
    METRICS_JSON = "json"
    CLI_INTERFACE = "cli"
    LOCKFILE = "lockfile"
    OTHER = "other"


@dataclass
class ArtifactContract:
    """Formal schema and contract specification for an artifact produced by a task."""
    path: str
    artifact_type: ArtifactType = ArtifactType.OTHER
    shape: Optional[List[Union[int, str]]] = None
    dtype: Optional[str] = None
    columns: Optional[Dict[str, str]] = None
    required_keys: Optional[List[str]] = None
    cli_args: Optional[List[str]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "artifact_type": self.artifact_type.value if hasattr(self.artifact_type, "value") else str(self.artifact_type),
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "columns": dict(self.columns) if self.columns is not None else None,
            "required_keys": list(self.required_keys) if self.required_keys is not None else None,
            "cli_args": list(self.cli_args) if self.cli_args is not None else None,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactContract":
        if not isinstance(data, dict):
            return cls(path=str(data))
        d = dict(data)
        raw_type = d.get("artifact_type", ArtifactType.OTHER.value)
        atype = ArtifactType.OTHER
        if isinstance(raw_type, ArtifactType):
            atype = raw_type
        elif isinstance(raw_type, str):
            clean_type = raw_type.strip().lower()
            if clean_type in ("json", "metrics_json", "metric_json"):
                atype = ArtifactType.METRICS_JSON
            elif clean_type in ("cli", "cli_interface", "cli_args"):
                atype = ArtifactType.CLI_INTERFACE
            elif clean_type in ("tensor", "tensors", "ndarray"):
                atype = ArtifactType.TENSOR
            elif clean_type in ("tabular", "table", "csv", "dataframe"):
                atype = ArtifactType.TABULAR
            elif clean_type in ("code", "source", "script", "python"):
                atype = ArtifactType.CODE
            elif clean_type in ("lockfile", "lock", "env_lock"):
                atype = ArtifactType.LOCKFILE
            else:
                try:
                    atype = ArtifactType(clean_type)
                except (ValueError, TypeError):
                    atype = ArtifactType.OTHER

        return cls(
            path=str(d.get("path", "")),
            artifact_type=atype,
            shape=list(d["shape"]) if d.get("shape") is not None else None,
            dtype=str(d["dtype"]) if d.get("dtype") is not None else None,
            columns=dict(d["columns"]) if d.get("columns") is not None else None,
            required_keys=list(d["required_keys"]) if d.get("required_keys") is not None else None,
            cli_args=list(d["cli_args"]) if d.get("cli_args") is not None else None,
            description=str(d.get("description", "")),
        )


@dataclass
class Task:
    """An executable task derived from a scientific paper."""
    task_id: str
    title: str
    category: TaskCategory = TaskCategory.MODEL_IMPLEMENTATION
    description: str = ""
    assigned_role: AgentRole = AgentRole.MODEL_ARCHITECT
    dependencies: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    paper_context_excerpts: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    artifact_contracts: List[ArtifactContract] = field(default_factory=list)
    test_harness_script: Optional[str] = None
    execution_log: Optional[str] = None
    metrics_produced: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "category": self.category.value if hasattr(self.category, "value") else str(self.category),
            "description": self.description,
            "assigned_role": self.assigned_role.value if hasattr(self.assigned_role, "value") else str(self.assigned_role),
            "dependencies": list(self.dependencies),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "acceptance_criteria": list(self.acceptance_criteria),
            "paper_context_excerpts": list(self.paper_context_excerpts),
            "tools_required": list(self.tools_required),
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "artifact_contracts": [
                c.to_dict() if hasattr(c, "to_dict") else (
                    ArtifactContract.from_dict(c).to_dict() if isinstance(c, dict) else dict(c)
                )
                for c in (self.artifact_contracts or [])
            ],
            "test_harness_script": self.test_harness_script,
            "execution_log": self.execution_log,
            "metrics_produced": dict(self.metrics_produced),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        d = dict(data)
        cat_val = d.get("category", TaskCategory.MODEL_IMPLEMENTATION.value)
        try:
            category = TaskCategory(cat_val)
        except (ValueError, TypeError):
            category = TaskCategory.MODEL_IMPLEMENTATION

        role_val = d.get("assigned_role", AgentRole.MODEL_ARCHITECT.value)
        try:
            assigned_role = AgentRole(role_val)
        except (ValueError, TypeError):
            assigned_role = AgentRole.MODEL_ARCHITECT

        status_val = d.get("status", TaskStatus.PENDING.value)
        try:
            status = TaskStatus(status_val)
        except (ValueError, TypeError):
            status = TaskStatus.PENDING

        raw_contracts = d.get("artifact_contracts") or []
        contracts: List[ArtifactContract] = []
        for c in raw_contracts:
            if isinstance(c, ArtifactContract):
                contracts.append(c)
            elif isinstance(c, dict):
                contracts.append(ArtifactContract.from_dict(c))

        return cls(
            task_id=str(d.get("task_id", "")),
            title=str(d.get("title", "")),
            category=category,
            description=str(d.get("description", "")),
            assigned_role=assigned_role,
            dependencies=list(d.get("dependencies") or []),
            inputs=list(d.get("inputs") or []),
            outputs=list(d.get("outputs") or []),
            acceptance_criteria=list(d.get("acceptance_criteria") or []),
            paper_context_excerpts=list(d.get("paper_context_excerpts") or []),
            tools_required=list(d.get("tools_required") or []),
            status=status,
            artifact_contracts=contracts,
            test_harness_script=d.get("test_harness_script"),
            execution_log=d.get("execution_log"),
            metrics_produced=dict(d.get("metrics_produced") or {}),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class TaskGraph:
    """Directed Acyclic Graph (DAG) representing the complete decomposition of a paper."""
    paper_id: str
    paper_title: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_dependencies(self, task_id: str) -> List[Task]:
        task = self.tasks.get(task_id)
        if not task:
            return []
        return [self.tasks[dep] for dep in task.dependencies if dep in self.tasks]

    def get_dependents(self, task_id: str) -> List[Task]:
        return [t for t in self.tasks.values() if task_id in t.dependencies]

    def validate(self) -> List[str]:
        """Validates graph integrity. Returns list of errors (empty if valid)."""
        errors = []
        # Check missing dependencies
        for t_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task '{t_id}' depends on missing task '{dep}'")
                elif dep == t_id:
                    errors.append(f"Task '{t_id}' cannot depend on itself")

        # Cycle detection via DFS
        visited = {}  # 0: unvisited, 1: visiting, 2: visited
        for t_id in self.tasks:
            visited[t_id] = 0

        def dfs(node: str, path: List[str]) -> bool:
            visited[node] = 1
            for dep in self.tasks[node].dependencies:
                if dep not in self.tasks:
                    continue
                if visited[dep] == 1:
                    cycle = " -> ".join(path + [dep])
                    errors.append(f"Cycle detected: {cycle}")
                    return True
                if visited[dep] == 0:
                    if dfs(dep, path + [dep]):
                        return True
            visited[node] = 2
            return False

        for t_id in self.tasks:
            if visited[t_id] == 0:
                dfs(t_id, [t_id])

        return errors

    def topological_waves(self) -> List[List[Task]]:
        """
        Partition tasks into execution waves (levels) using topological sort.
        Wave 0 contains root tasks with 0 dependencies.
        Wave k contains tasks whose dependencies are all in waves < k.
        """
        errs = self.validate()
        if errs:
            raise ValueError(f"Cannot sort invalid TaskGraph: {errs}")

        in_degree = {t_id: len(task.dependencies) for t_id, task in self.tasks.items()}
        # Adjacency: parent -> list of children
        dependents = {t_id: [] for t_id in self.tasks}
        for t_id, task in self.tasks.items():
            for dep in task.dependencies:
                dependents[dep].append(t_id)

        waves = []
        current_wave = [t_id for t_id, deg in in_degree.items() if deg == 0]

        while current_wave:
            waves.append([self.tasks[t_id] for t_id in current_wave])
            next_wave = []
            for t_id in current_wave:
                for child in dependents[t_id]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_wave.append(child)
            current_wave = next_wave

        return waves

    def ready_tasks(self) -> List[Task]:
        """Returns tasks that are ready to run (all dependencies COMPLETED)."""
        ready = []
        for task in self.tasks.values():
            if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                deps = self.get_dependencies(task.task_id)
                if all(d.status == TaskStatus.COMPLETED for d in deps):
                    ready.append(task)
        return ready

    def get_missing_phases(self) -> List[str]:
        """
        Validates presence of all 6 mandatory scientific phases:
        1. Environment Setup
        2. Data Splits & Acquisition
        3. Sanity Smoke-Tests
        4. Training / Simulation
        5. Metric Evaluation
        6. Verification Audit
        Returns list of missing phase names.
        """
        missing = []
        # Phase 1: Environment Setup
        has_env = any(
            t.category == TaskCategory.ENVIRONMENT_SETUP
            or t.assigned_role == AgentRole.DEVOPS_AGENT
            or "env" in t.task_id.lower()
            or "environment" in t.title.lower()
            for t in self.tasks.values()
        )
        if not has_env:
            missing.append("environment_setup")

        # Phase 2: Data Splits & Acquisition
        has_data = any(
            t.category in (TaskCategory.DATA_ACQUISITION, TaskCategory.DATA_PREPROCESSING)
            or t.assigned_role == AgentRole.DATA_ENGINEER
            or "data" in t.task_id.lower()
            or "dataset" in t.title.lower()
            or "split" in t.title.lower()
            for t in self.tasks.values()
        )
        if not has_data:
            missing.append("data_splits")

        # Phase 3: Sanity Smoke-Tests
        smoke_cats = {
            TaskCategory.SANITY_SMOKE_TEST,
            getattr(TaskCategory, "SANITY_SMOKE_TESTS", TaskCategory.SANITY_SMOKE_TEST),
        }
        has_smoke = any(
            t.category in smoke_cats
            or "smoke" in t.task_id.lower()
            or "smoke" in t.title.lower()
            or "smoke" in t.description.lower()
            or "sanity" in t.title.lower()
            or "sanity" in t.description.lower()
            for t in self.tasks.values()
        )
        if not has_smoke:
            missing.append("sanity_smoke_test")

        # Phase 4: Training / Simulation
        has_train = any(
            t.category in (TaskCategory.TRAINING_OR_SIMULATION, TaskCategory.ABLATION_STUDY)
            or t.assigned_role == AgentRole.EXPERIMENT_RUNNER
            or "train" in t.task_id.lower()
            or "train" in t.title.lower()
            or "simulation" in t.title.lower()
            or "sweep" in t.title.lower()
            or "experiment" in t.task_id.lower()
            for t in self.tasks.values()
        )
        if not has_train:
            missing.append("training_or_simulation")

        # Phase 5: Metric Evaluation
        has_eval = any(
            t.category == TaskCategory.METRIC_EVALUATION
            or t.assigned_role == AgentRole.EVALUATION_AGENT
            or "eval" in t.task_id.lower()
            or "metric" in t.title.lower()
            or "evaluation" in t.title.lower()
            for t in self.tasks.values()
        )
        if not has_eval:
            missing.append("metric_evaluation")

        # Phase 6: Verification Audit
        has_audit = any(
            t.category == TaskCategory.VERIFICATION_AUDIT
            or t.assigned_role == AgentRole.AUDIT_AGENT
            or "audit" in t.task_id.lower()
            or "scorecard" in t.title.lower()
            or "verification" in t.title.lower()
            for t in self.tasks.values()
        )
        if not has_audit:
            missing.append("verification_audit")

        return missing

    def has_all_mandatory_phases(self) -> bool:
        """Returns True if the workflow covers all 6 mandatory scientific phases."""
        return len(self.get_missing_phases()) == 0

    def wave_count(self) -> int:
        """Returns the total number of execution waves (depth of DAG)."""
        return len(self.topological_waves())

    def max_concurrency(self) -> int:
        """Returns maximum number of tasks scheduled in parallel in any single wave."""
        waves = self.topological_waves()
        return max((len(w) for w in waves), default=0)

    def is_parallelizable(self) -> bool:
        """Returns True if any execution wave contains 2 or more concurrent tasks."""
        waves = self.topological_waves()
        return any(len(w) > 1 for w in waves)

    def wave_summary(self) -> List[Dict[str, Any]]:
        """Returns a structured summary of execution waves and concurrency."""
        waves = self.topological_waves()
        return [
            {
                "wave_index": idx,
                "task_count": len(wave),
                "task_ids": [t.task_id for t in wave],
                "roles": [t.assigned_role.value for t in wave],
            }
            for idx, wave in enumerate(waves)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "tasks": {t_id: task.to_dict() for t_id, task in self.tasks.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskGraph":
        tasks = {
            t_id: Task.from_dict(t_data)
            for t_id, t_data in (data.get("tasks") or {}).items()
        }
        return cls(
            paper_id=data.get("paper_id", ""),
            paper_title=data.get("paper_title", ""),
            tasks=tasks,
            metadata=data.get("metadata") or {},
        )
