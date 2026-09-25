"""
Data models and schemas for the Paper-to-Agent Task Decomposition Pipeline.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskCategory(str, Enum):
    """Categorical classification of scientific execution tasks."""
    ENVIRONMENT_SETUP = "environment_setup"
    DATA_ACQUISITION = "data_acquisition"
    DATA_PREPROCESSING = "data_preprocessing"
    MODEL_IMPLEMENTATION = "model_implementation"
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


class AgentRole(str, Enum):
    """Specialized agent personas for task execution."""
    DEVOPS_AGENT = "devops_agent"           # Environment, packages, runtime dependencies
    DATA_ENGINEER = "data_engineer"         # Datasets, downloads, preprocessing, splits
    MODEL_ARCHITECT = "model_architect"     # Models, algorithms, loss functions, math
    EXPERIMENT_RUNNER = "experiment_runner" # Training runs, simulations, sweeps
    EVALUATION_AGENT = "evaluation_agent"   # Metrics, benchmark evaluation, comparison tables
    AUDIT_AGENT = "audit_agent"             # Claim verification against paper targets


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
    execution_log: Optional[str] = None
    metrics_produced: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        data["assigned_role"] = self.assigned_role.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        d = dict(data)
        d["category"] = TaskCategory(d.get("category", TaskCategory.MODEL_IMPLEMENTATION.value))
        d["assigned_role"] = AgentRole(d.get("assigned_role", AgentRole.MODEL_ARCHITECT.value))
        d["status"] = TaskStatus(d.get("status", TaskStatus.PENDING.value))
        return cls(**d)


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
