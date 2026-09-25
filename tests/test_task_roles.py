"""
Unit tests for agent roles, specifications, and prompt synthesis.
"""

import pytest
from research_assistant.tasks.models import AgentRole, Task, TaskCategory
from research_assistant.tasks.roles import (
    RoleSpec,
    get_role_spec,
    get_all_roles,
    format_agent_prompt,
)


def test_get_all_roles():
    roles = get_all_roles()
    assert len(roles) == 6
    role_enums = {r.role for r in roles}
    assert AgentRole.DEVOPS_AGENT in role_enums
    assert AgentRole.MODEL_ARCHITECT in role_enums
    assert AgentRole.AUDIT_AGENT in role_enums


def test_get_role_spec():
    devops = get_role_spec(AgentRole.DEVOPS_AGENT)
    assert devops.name == "DevOps & Environment Specialist"
    assert "dependency_resolution" in devops.capabilities
    assert "run_command" in devops.default_tools
    assert "num_ctx: 4096" in devops.system_prompt or "reproducible" in devops.system_prompt

    # Test string lookup
    data_spec = get_role_spec("data_engineer")
    assert data_spec.role == AgentRole.DATA_ENGINEER

    # Test fallback
    unknown_spec = get_role_spec("non_existent_role")
    assert unknown_spec.role == AgentRole.MODEL_ARCHITECT


def test_role_to_dict():
    spec = get_role_spec(AgentRole.AUDIT_AGENT)
    d = spec.to_dict()
    assert d["role"] == "audit_agent"
    assert d["name"] == "Scientific Replication Auditor"
    assert "scorecard_synthesis" in d["capabilities"]


def test_format_agent_prompt():
    task = Task(
        task_id="task_03_model",
        title="Implement Variational Autoencoder",
        category=TaskCategory.MODEL_IMPLEMENTATION,
        description="Implement the VAE encoder and decoder network.",
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_01_env"],
        inputs=["env.lock", "config.yaml"],
        outputs=["model.py"],
        acceptance_criteria=["Reconstruction loss converges", "KL loss term matches eq 4"],
        paper_context_excerpts=["Section 3: Variational Autoencoder formulation."],
    )

    prompt = format_agent_prompt(task, workspace_dir="/tmp/workspace")
    assert "Scientific Replication Task: Implement Variational Autoencoder" in prompt
    assert "task_03_model" in prompt
    assert "Model Architect & Algorithm Engineer" in prompt
    assert "model.py" in prompt
    assert "KL loss term matches eq 4" in prompt
    assert "Section 3: Variational Autoencoder formulation." in prompt
