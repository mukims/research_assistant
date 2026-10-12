"""
Unit tests for agent roles, specifications, and prompt synthesis.
"""

import unittest

from research_assistant.tasks.models import AgentRole, Task, TaskCategory
from research_assistant.tasks.roles import (
    RoleSpec,
    get_role_spec,
    get_all_roles,
    format_agent_prompt,
)


class TestTaskRoles(unittest.TestCase):
    def test_get_all_roles(self):
        roles = get_all_roles()
        self.assertEqual(len(roles), 6)
        role_enums = {r.role for r in roles}
        self.assertIn(AgentRole.DEVOPS_AGENT, role_enums)
        self.assertIn(AgentRole.MODEL_ARCHITECT, role_enums)
        self.assertIn(AgentRole.AUDIT_AGENT, role_enums)

    def test_get_role_spec(self):
        devops = get_role_spec(AgentRole.DEVOPS_AGENT)
        self.assertEqual(devops.name, "DevOps & Environment Specialist")
        self.assertIn("dependency_resolution", devops.capabilities)
        self.assertIn("run_command", devops.default_tools)
        self.assertTrue("num_ctx: 4096" in devops.system_prompt or "reproducible" in devops.system_prompt)

        # Test string lookup
        data_spec = get_role_spec("data_engineer")
        self.assertEqual(data_spec.role, AgentRole.DATA_ENGINEER)

        # Test fallback
        unknown_spec = get_role_spec("non_existent_role")
        self.assertEqual(unknown_spec.role, AgentRole.MODEL_ARCHITECT)

    def test_role_to_dict(self):
        spec = get_role_spec(AgentRole.AUDIT_AGENT)
        d = spec.to_dict()
        self.assertEqual(d["role"], "audit_agent")
        self.assertEqual(d["name"], "Scientific Replication Auditor")
        self.assertIn("scorecard_synthesis", d["capabilities"])

    def test_format_agent_prompt(self):
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
        self.assertIn("Scientific Replication Task: Implement Variational Autoencoder", prompt)
        self.assertIn("task_03_model", prompt)
        self.assertIn("Model Architect & Algorithm Engineer", prompt)
        self.assertIn("model.py", prompt)
        self.assertIn("KL loss term matches eq 4", prompt)
        self.assertIn("Section 3: Variational Autoencoder formulation.", prompt)


if __name__ == "__main__":
    unittest.main()
