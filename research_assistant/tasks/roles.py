"""
Agent Roles & Persona Specifications for Scientific Task Execution.

Defines the system instructions, capabilities, tool access, and prompt templates
for each specialized subagent executing tasks in the research pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from research_assistant.tasks.models import AgentRole, Task


@dataclass
class RoleSpec:
    """Specification of an agent role persona."""
    role: AgentRole
    name: str
    description: str
    system_prompt: str
    capabilities: List[str] = field(default_factory=list)
    default_tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "default_tools": self.default_tools,
        }


DEVOPS_AGENT_PROMPT = """You are the DevOps & Environment Specialist for scientific code execution.
Your responsibility is to provision reproducible computational environments for scientific paper replication.
Key rules:
1. Parse dependency manifests, requirements.txt, environment.yml, or Dockerfile.
2. Verify Python version, PyTorch/CUDA availability, or CPU-safe limits (e.g. num_ctx: 4096, thread caps).
3. Ensure reproducible lockfiles and environment state flags (e.g., env_ready.lock).
4. Never assume external GPU access unless explicitly confirmed. Default to portable, CPU-safe execution fallbacks.
"""

DATA_ENGINEER_PROMPT = """You are the Scientific Data Engineer for research replication.
Your responsibility is to acquire, inspect, clean, and preprocess scientific datasets described in papers.
Key rules:
1. Verify raw data integrity, checksums, and download sources.
2. Implement deterministic data splits (train/validation/test) with explicit random seeds.
3. Validate data dimensions, missing/NaN values, and feature scaling according to the paper's methodology.
4. Output structured artifacts (e.g. HDF5, NumPy arrays, PyTorch datasets, or Parquet tables) ready for model consumption.
"""

MODEL_ARCHITECT_PROMPT = """You are the Model Architect & Mathematical Implementation Specialist.
Your responsibility is to translate theoretical equations, algorithms, and architectures from the paper into clean code.
Key rules:
1. Translate mathematical equations from the paper accurately into code.
2. Verify layer dimensions, parameter counts, and tensor shapes against paper descriptions.
3. Implement loss functions, regularizers, and forward pass routines strictly as formulated in the text.
4. Include smoke-test assertions and unit tests to verify gradients flow and loss is differentiable.
"""

EXPERIMENT_RUNNER_PROMPT = """You are the Experiment Runner & Computational Simulation Specialist.
Your responsibility is to execute baseline runs, ablation sweeps, and training routines.
Key rules:
1. Apply the exact hyperparameters specified in the paper (learning rate, batch size, epochs, optimizers).
2. Record metrics at each epoch or simulation step into structured logs (CSV, JSON, TensorBoard).
3. Save model checkpoints at key milestones (best validation loss, final step).
4. Monitor convergence, detect exploding/vanishing gradients, and log anomalies.
"""

EVALUATION_AGENT_PROMPT = """You are the Evaluation & Benchmark Specialist.
Your responsibility is to measure model performance and generate comparative benchmark tables.
Key rules:
1. Evaluate models strictly on held-out test sets without data leakage.
2. Compute all primary and secondary metrics reported in the paper (e.g., Accuracy, F1, MSE, BLEU, State Fidelity).
3. Generate side-by-side comparison tables contrasting paper-reported values vs replicated values.
4. Flag any statistical discrepancies outside accepted variance thresholds.
"""

AUDIT_AGENT_PROMPT = """You are the Scientific Replication Auditor.
Your responsibility is to independently inspect, audit, and certify replication artifacts.
Key rules:
1. Review all produced artifacts (environment, data, code, logs, results) against task acceptance criteria.
2. Verify claim fidelity: compare replicated metrics against specific claims made in the original paper text.
3. Check for signs of overfitting, data contamination, or undocumented hyperparameter tuning.
4. Synthesize a structured Replication Scorecard grading the reproducibility of the paper.
"""


ROLE_SPECS: Dict[AgentRole, RoleSpec] = {
    AgentRole.DEVOPS_AGENT: RoleSpec(
        role=AgentRole.DEVOPS_AGENT,
        name="DevOps & Environment Specialist",
        description="Provisions runtime environments, locks dependencies, and verifies hardware constraints.",
        system_prompt=DEVOPS_AGENT_PROMPT,
        capabilities=["env_setup", "dependency_resolution", "cuda_verification", "containerization"],
        default_tools=["run_command", "view_file", "write_to_file"],
    ),
    AgentRole.DATA_ENGINEER: RoleSpec(
        role=AgentRole.DATA_ENGINEER,
        name="Scientific Data Engineer",
        description="Acquires raw datasets, executes preprocessing pipelines, and creates canonical splits.",
        system_prompt=DATA_ENGINEER_PROMPT,
        capabilities=["data_download", "normalization", "splitting", "integrity_validation"],
        default_tools=["run_command", "view_file", "write_to_file"],
    ),
    AgentRole.MODEL_ARCHITECT: RoleSpec(
        role=AgentRole.MODEL_ARCHITECT,
        name="Model Architect & Algorithm Engineer",
        description="Translates equations, loss functions, and network architectures into executable code.",
        system_prompt=MODEL_ARCHITECT_PROMPT,
        capabilities=["equation_transcription", "model_coding", "loss_implementation", "smoke_testing"],
        default_tools=["view_file", "write_to_file", "replace_file_content", "run_command"],
    ),
    AgentRole.EXPERIMENT_RUNNER: RoleSpec(
        role=AgentRole.EXPERIMENT_RUNNER,
        name="Experiment & Simulation Runner",
        description="Executes training loops, hyperparameter schedules, and saves model checkpoints.",
        system_prompt=EXPERIMENT_RUNNER_PROMPT,
        capabilities=["training_execution", "hyperparameter_tuning", "metric_logging", "checkpointing"],
        default_tools=["run_command", "view_file"],
    ),
    AgentRole.EVALUATION_AGENT: RoleSpec(
        role=AgentRole.EVALUATION_AGENT,
        name="Evaluation & Benchmark Specialist",
        description="Computes benchmark metrics on test sets and compares results with paper claims.",
        system_prompt=EVALUATION_AGENT_PROMPT,
        capabilities=["metric_calculation", "benchmark_comparison", "statistical_analysis", "table_generation"],
        default_tools=["run_command", "view_file", "write_to_file"],
    ),
    AgentRole.AUDIT_AGENT: RoleSpec(
        role=AgentRole.AUDIT_AGENT,
        name="Scientific Replication Auditor",
        description="Audits deliverables, inspects code integrity, and writes final replication scorecard.",
        system_prompt=AUDIT_AGENT_PROMPT,
        capabilities=["artifact_verification", "discrepancy_detection", "claim_audit", "scorecard_synthesis"],
        default_tools=["view_file", "run_command", "write_to_file"],
    ),
}


def get_role_spec(role: Union[AgentRole, str]) -> RoleSpec:
    """Returns the RoleSpec for a given role enum or string identifier."""
    if isinstance(role, str):
        try:
            role = AgentRole(role.lower())
        except ValueError:
            role = AgentRole.MODEL_ARCHITECT
    return ROLE_SPECS.get(role, ROLE_SPECS[AgentRole.MODEL_ARCHITECT])


def get_all_roles() -> List[RoleSpec]:
    """Returns a list of all defined role specifications."""
    return list(ROLE_SPECS.values())


def format_context_pack_section(
    task: Task,
    plan_path: Optional[str] = None,
) -> str:
    """
    Renders the grounded `TaskContextPack` (seed passages, dereferenced cited protocols,
    technique cards, unresolved citations) and live database query instructions.
    """
    pack = getattr(task, "context_pack", None)
    lines: List[str] = []

    if pack is not None and not pack.is_empty():
        lines.append("## Grounded Reference Database Context Pack")

        if pack.seed_passages:
            lines.append("### Target Paper Grounded Passages")
            for idx, p in enumerate(pack.seed_passages[:4], 1):
                doc = p.get("document") or "target_paper"
                sec = p.get("section") or "text"
                page = p.get("page", "?")
                txt = (p.get("text") or "").strip()
                if txt:
                    lines.append(f"- **[Passage {idx} | {doc} | p.{page} | {sec}]**: {txt}")

        if pack.cited_protocols:
            lines.append("\n### Dereferenced Cited Protocols & External Baselines")
            for idx, c in enumerate(pack.cited_protocols[:4], 1):
                cite_marker = c.get("cite_text") or f"Ref #{idx}"
                role = c.get("role") or "method"
                c_title = c.get("cited_title") or c.get("cited_doc_key") or "Cited Reference"
                c_doc = c.get("cited_doc_key") or ""
                page = c.get("page", "?")
                sec = c.get("section") or "text"
                citing_sent = (c.get("citing_sentence") or "").strip()
                txt = (c.get("text") or "").strip()
                lines.append(
                    f"- **[{cite_marker} ({role}) -> {c_title} ({c_doc}, p.{page}, {sec})]**:"
                )
                if citing_sent:
                    lines.append(f"  - _Citing Context_: \"{citing_sent}\"")
                if txt:
                    lines.append(f"  - _Protocol Detail_: {txt}")

        if pack.technique_cards:
            lines.append("\n### Verified Technique Cards from Corpus")
            for tc in pack.technique_cards[:3]:
                t_name = tc.get("name") or "Technique"
                t_doc = tc.get("document") or ""
                t_prob = tc.get("problem_solved") or ""
                t_in = tc.get("inputs") or ""
                t_out = tc.get("outputs") or ""
                lines.append(
                    f"- **{t_name}** (`{t_doc}`): {t_prob} | Inputs: {t_in} | Outputs: {t_out}"
                )

        if pack.unresolved_citations:
            lines.append("\n### Unresolved External Citations (Not in Local Corpus)")
            for u in pack.unresolved_citations[:4]:
                u_cite = u.get("cite_text") or "?"
                u_title = u.get("title") or "External Reference"
                u_role = u.get("role") or "method"
                lines.append(f"- `[{u_cite}]` ({u_role}): {u_title}")

    plan_arg = plan_path or "plan.json"
    lines.append("\n## Live Reference Database Query Access")
    lines.append(
        "If you need additional equations, hyperparameters, or cited paper protocols from the "
        "paper's reference subgraph during execution, query the database directly via CLI:"
    )
    lines.append(
        f"`python orchestrate_tasks.py query {plan_arg} --task-id {task.task_id} "
        f"--query \"<your specific methodological or equation question>\" --scope subgraph`"
    )

    return "\n".join(lines).strip() + "\n\n"


def format_subagent_prompt(
    task: Task,
    role_spec: Optional[RoleSpec] = None,
    workspace_dir: str = ".",
    context_notes: Optional[str] = None,
    upstream_artifacts: Optional[Dict[str, Any]] = None,
    plan_path: Optional[str] = None,
) -> str:
    """
    Constructs an execution prompt for a subagent assigned to a specific task,
    embedding formal artifact contracts, schemas, grounded database context packs,
    and executable test harness instructions.
    """
    from research_assistant.tasks.harness import generate_task_test_script

    spec = role_spec or get_role_spec(task.assigned_role)
    inputs_str = ", ".join(task.inputs) if task.inputs else "None (Initial task)"
    outputs_str = ", ".join(task.outputs) if task.outputs else "Standard execution artifacts"
    criteria_str = "\n".join(f"- {c}" for c in task.acceptance_criteria) if task.acceptance_criteria else "- Task finishes with returncode 0"
    paper_ctx = "\n".join(f"> {p}" for p in task.paper_context_excerpts) if task.paper_context_excerpts else "N/A"
    context_pack_section = format_context_pack_section(task, plan_path=plan_path)

    # 1. Format formal artifact contracts & schemas
    contracts_section = ""
    if getattr(task, "artifact_contracts", None):
        c_lines = ["## Formal Deliverable Artifact Contracts & Schemas"]
        for c in task.artifact_contracts:
            c_type = c.artifact_type.value if hasattr(c.artifact_type, "value") else str(c.artifact_type)
            details = [f"type: `{c_type}`"]
            if c.shape:
                details.append(f"shape: `{c.shape}`")
            if c.dtype:
                details.append(f"dtype: `{c.dtype}`")
            if c.required_keys:
                details.append(f"required_keys: `{c.required_keys}`")
            if c.columns:
                details.append(f"columns: `{dict(c.columns)}`")
            if c.cli_args:
                details.append(f"cli_args: `{c.cli_args}`")
            desc = f" — {c.description}" if c.description else ""
            c_lines.append(f"- **`{c.path}`** ({', '.join(details)}){desc}")
        contracts_section = "\n".join(c_lines) + "\n\n"

    # 2. Format verified upstream deliverables
    upstream_section = ""
    if upstream_artifacts:
        u_lines = ["## Verified Upstream Deliverables on Disk"]
        for path, info in upstream_artifacts.items():
            u_lines.append(f"- `{path}`: {info}")
        upstream_section = "\n".join(u_lines) + "\n\n"

    # 3. Format executable test harness
    harness_code = getattr(task, "test_harness_script", None)
    if not harness_code:
        harness_code = generate_task_test_script(task, workspace_dir)

    test_filename = f"test_{task.task_id}.py"
    harness_section = (
        f"## Automated Acceptance Test Harness\n"
        f"Your deliverables will be tested deterministically with the following acceptance test:\n"
        f"```python\n{harness_code}\n```\n\n"
        f"**Verification Command**:\n"
        f"Execute this test from `{workspace_dir}` to verify acceptance:\n"
        f"`python -m unittest {test_filename}`\n\n"
    )

    # 4. Format optional notes (a nested f-string with "\n" is a SyntaxError before Python 3.12)
    notes_section = f"## Additional Notes\n{context_notes}\n\n" if context_notes else ""

    return f"""# Scientific Replication Task: {task.title}
**Task ID**: `{task.task_id}`
**Category**: `{task.category.value}`
**Assigned Role**: {spec.name} ({spec.role.value})
**Workspace Directory**: `{workspace_dir}`

## Objective
{task.description}

## Context Excerpts from Paper
{paper_ctx}

{context_pack_section}## Prerequisites & Input Artifacts
{inputs_str}

{upstream_section}## Required Deliverables
{outputs_str}

{contracts_section}## Acceptance Criteria
{criteria_str}

{harness_section}{notes_section}## Execution Instructions
1. Inspect the workspace and confirm all input prerequisites exist.
2. Review and maintain `rationale.md` in the workspace documenting your Problem Knowledge, Scientific Insight, and Implementation Strategy (AUTOMAT idea.md protocol).
3. Produce all required deliverables strictly conforming to the schemas in Formal Artifact Contracts and complexity bounds.
4. Verify that your implementation satisfies every acceptance criterion above.
5. Run the Automated Acceptance Test Harness before marking completion; exit code must be 0.
"""


format_agent_prompt = format_subagent_prompt
