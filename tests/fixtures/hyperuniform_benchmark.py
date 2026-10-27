"""
Domain Benchmark Fixture: 1D Stealthy Hyperuniform Disorder & Inverse Transport Benchmark.

Provides a reference TaskGraph benchmark for reproducing transport in 1D stealthy
hyperuniform systems (Fourier filtering, tight-binding Hamiltonian, RGF solver,
transmittance spectra, and parameter inversion landscape).
"""

from typing import Optional
from research_assistant.tasks.models import (
    AgentRole,
    ArtifactContract,
    ArtifactType,
    Task,
    TaskCategory,
    TaskGraph,
)


def build_hyperuniform_benchmark_graph(
    paper_id: str = "physics_stealthy_1d",
    paper_title: str = "Inverse Determination of Stealthy Hyperuniform Disorder",
) -> TaskGraph:
    """
    Constructs the canonical 5-task benchmark DAG for 1D stealthy hyperuniform disorder inversion.
    """
    graph = TaskGraph(paper_id=paper_id, paper_title=paper_title)

    t1 = Task(
        task_id="task_01_env",
        title="Provision Environment for Quantum Transport & Disorder Physics",
        category=TaskCategory.ENVIRONMENT_SETUP,
        description="Set up Python environment, install required computational packages (numpy, scipy, torch), and verify single-thread CPU limits (OMP_NUM_THREADS=1, torch.set_num_threads(1)).",
        assigned_role=AgentRole.DEVOPS_AGENT,
        dependencies=[],
        inputs=["requirements.txt", "Dockerfile"],
        outputs=["env_ready.lock"],
        acceptance_criteria=["Environment builds cleanly with exit code 0", "Key scientific libraries import without error"],
        artifact_contracts=[
            ArtifactContract(
                path="env_ready.lock",
                artifact_type=ArtifactType.LOCKFILE,
                description="Locked environment and dependencies for quantum transport",
            ),
        ],
    )

    t2 = Task(
        task_id="task_02_data",
        title="Synthesize 1D Stealthy Hyperuniform Disorder Configurations & Structure Factor S(k)",
        category=TaskCategory.DATA_PREPROCESSING,
        description=(
            "Synthesize 1D stealthy hyperuniform disorder configurations w_j via Fourier filtering: "
            "(1) Discretize tight-binding chain of N=10^4 sites with wavevectors k_n = 2*pi*n/N for n in [-N/2, N/2 - 1]; "
            "(2) Impose spectral cutoff K = 2*pi*chi (chi in (0, 0.5)) with structure factor S(k) = Theta(|k| - K); "
            "(3) Generate random complex Gaussian Fourier amplitudes w_k = W * sqrt(S(k)) * eta_k with Hermitian symmetry "
            "w_{-k} = w_k* and w_0 = 0 to enforce real-space potential; "
            "(4) Apply inverse FFT to obtain real-space on-site disorder potential w_j for N_conf disorder realizations."
        ),
        assigned_role=AgentRole.DATA_ENGINEER,
        dependencies=["task_01_env"],
        inputs=["env_ready.lock", "data_spec.yaml"],
        outputs=["data/disorder_configurations.h5", "data/structure_factor.pt"],
        acceptance_criteria=[
            "Structure factor S(k) strictly vanishes (S(k) = 0) for all |k| < 2*pi*chi",
            "Generated on-site potential w_j is strictly real-valued with imaginary component norm < 1e-12",
            "Disorder configurations tensor verified non-empty with zero NaN/Inf entries",
        ],
        artifact_contracts=[
            ArtifactContract(
                path="data/disorder_configurations.h5",
                artifact_type=ArtifactType.TENSOR,
                shape=[1000, 10000],
                dtype="float32",
                description="Synthesized stealthy hyperuniform disorder realizations",
            ),
            ArtifactContract(
                path="data/structure_factor.pt",
                artifact_type=ArtifactType.TENSOR,
                shape=[10000],
                dtype="float32",
                description="Stealthy hyperuniform structure factor S(k)",
            ),
        ],
    )

    t3 = Task(
        task_id="task_03_model",
        title="Implement Tight-Binding Hamiltonian & Recursive Green's Function (RGF) Solver",
        category=TaskCategory.MODEL_IMPLEMENTATION,
        description=(
            "Construct the 1D tight-binding Hamiltonian H = H_C + H_coupling + H_L + H_R with nearest-neighbor "
            "hopping amplitude t=1 and on-site energies w_j. Implement the Recursive Green's Function (RGF) algorithm "
            "and Caroli-Meir-Wingreen formula to compute surface Green's functions and energy-dependent transmittance T(E) with O(N) complexity."
        ),
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_01_env"],
        inputs=["env_ready.lock", "model_config.yaml"],
        outputs=["src/tight_binding.py", "src/rgf_solver.py"],
        acceptance_criteria=[
            "Tight-binding Hamiltonian matrix matches open-system lead boundary conditions",
            "RGF solver forward pass computes transmittance T(E) in [0, 1] without NaN values",
        ],
        artifact_contracts=[
            ArtifactContract(
                path="src/tight_binding.py",
                artifact_type=ArtifactType.CODE,
                description="Tight-binding Hamiltonian and lead coupling operators",
            ),
            ArtifactContract(
                path="src/rgf_solver.py",
                artifact_type=ArtifactType.CODE,
                description="Recursive Green's Function (RGF) transport solver",
            ),
        ],
    )

    t4 = Task(
        task_id="task_04_experiment",
        title="Simulate Ensemble Transmittance Spectra T(E; W, chi) & Map Misfit Cost Landscape F(W, chi)",
        category=TaskCategory.TRAINING_OR_SIMULATION,
        description=(
            "Compute disorder-averaged transmittance spectra T(E; W, chi) = (1/N_conf) sum_i T_i(E) over N=10^4 chains. "
            "Generate target spectrum T_tar(E) at (W_tar=0.10, chi_tar=0.30) with N_conf=1000. Sweep trial parameters (W, chi) "
            "with N_conf=200 and compute misfit cost landscape F(W, chi) = (1/N_E) sum_m [T(E_m; W, chi) - T_tar(E_m)]^2 "
            "over energy window [E-, E+] with single-thread capping (torch.set_num_threads(1), OMP_NUM_THREADS=1)."
        ),
        assigned_role=AgentRole.EXPERIMENT_RUNNER,
        dependencies=["task_02_data", "task_03_model"],
        inputs=["data/disorder_configurations.h5", "src/rgf_solver.py"],
        outputs=["results/transmittance_spectra.h5", "results/misfit_landscape.csv"],
        acceptance_criteria=[
            "Disorder ensemble averaging completed across target and trial parameter grid",
            "Misfit cost landscape F(W, chi) contains non-empty finite float entries across all trial pairs",
        ],
        artifact_contracts=[
            ArtifactContract(
                path="results/transmittance_spectra.h5",
                artifact_type=ArtifactType.TENSOR,
                shape=["N_trials", "N_energies"],
                dtype="float32",
                description="Ensemble-averaged transmittance spectra T(E)",
            ),
            ArtifactContract(
                path="results/misfit_landscape.csv",
                artifact_type=ArtifactType.TABULAR,
                columns={"W": "float", "chi": "float", "misfit": "float"},
                description="Misfit cost function landscape F(W, chi)",
            ),
        ],
    )

    t5 = Task(
        task_id="task_05_eval",
        title="Extract Inversion Parameters (W_opt, chi_opt) & Benchmark Against Paper Spectra",
        category=TaskCategory.METRIC_EVALUATION,
        description=(
            "Locate global minimum of misfit cost landscape F(W, chi) to extract recovered parameters (W_opt, chi_opt). "
            "Compare against reference target parameters (W_tar=0.10, chi_tar=0.30) within 5% error and benchmark "
            "transmittance curves against Fig. 2 and Fig. 3."
        ),
        assigned_role=AgentRole.EVALUATION_AGENT,
        dependencies=["task_04_experiment"],
        inputs=["results/misfit_landscape.csv", "results/transmittance_spectra.h5"],
        outputs=["results/optimal_parameters.json", "results/benchmark_report.json"],
        acceptance_criteria=[
            "Parameter recovery error |W_opt - W_tar|/W_tar < 0.05 and |chi_opt - chi_tar|/chi_tar < 0.05",
            "Benchmark report cross-references Fig. 2 and Fig. 3 transmittance step locations",
        ],
        artifact_contracts=[
            ArtifactContract(
                path="results/optimal_parameters.json",
                artifact_type=ArtifactType.METRICS_JSON,
                required_keys=["W_opt", "chi_opt", "relative_error_W", "relative_error_chi"],
                description="Recovered stealthiness and disorder parameters",
            ),
            ArtifactContract(
                path="results/benchmark_report.json",
                artifact_type=ArtifactType.METRICS_JSON,
                required_keys=["benchmark_status", "energy_window"],
                description="Transmittance curve benchmark evaluation",
            ),
        ],
    )

    for t in [t1, t2, t3, t4, t5]:
        graph.add_task(t)

    return graph
