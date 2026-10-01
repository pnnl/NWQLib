# Examples

The notebooks in [`examples/`](https://github.com/pnnl/NWQLib/tree/main/examples) apply NWQLib to complete scientific problems. Each introductory and scientific notebook solves its problem with a quantum algorithm and shows on its first screen the answer, its comparison with an independent classical reference and its cost.

The resource-estimation notebook plans problems too large to simulate and reports only their cost. The stored notebooks include their outputs, so they can be read on GitHub without running them.

## Start from the problem you have

| You have | Notebook | Methods | Largest simulated circuit | Run time | Extras |
| --- | --- | --- | --- | --- | --- |
| A linear system, A x = b | [`examples/qls_linear_system_intro.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/qls_linear_system_intro.ipynb): steady heat in a plate | QSVT inverse, Dalzell shortcut (arXiv:2406.12086v2) | 12 qubits | about 23 s | `aer,notebook` |
| A linear ODE, du/dt = −A u + b | [`examples/lchs_linear_dynamics_intro.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/lchs_linear_dynamics_intro.ipynb): a pulse carried and spread by a flow | LCHS kernels, quadratures and state preparation | 12 qubits | about 22 s | `aer,notebook,tensor` |
| A Hamiltonian | [`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb): ground-state energy of a stretched H4 chain | ADAPT-GCIM, Chebyshev Lanczos, QCELS | 8 qubits | about 19 s | `aer,notebook,chemistry` |
| An objective, bounds and optional constraints | [`examples/qhd_optimization_intro.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/qhd_optimization_intro.ipynb): a two-valley function of two variables and a quadratic on the unit disk | Quantum Hamiltonian descent, augmented Lagrangian for constraints | 12 qubits | about 30 s | `aer,notebook` |
| A problem from a paper: heat flow with a boundary penalty | [`examples/lchs_scientific.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/lchs_scientific.ipynb) | LCHS with a constant source | 10 qubits | about 20 s | `aer,notebook` |
| A problem from a paper: collision relaxation in a fluid | [`examples/qls_scientific.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/qls_scientific.ipynb) | Dalzell QLS | 10 qubits | about 7 s | `aer,notebook` |
| A problem from a paper: precise and constrained optimization on a coarse grid | [`examples/qhd_scientific.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/qhd_scientific.ipynb) | Quantum Hamiltonian descent with box refinement, augmented Lagrangian with refinement in every round | 4 qubits, and 10-qubit grid models evolved classically | about 54 s | `aer,notebook` |
| A problem at 80 to 100 system qubits, and the question of what it would cost | [`examples/resource_estimation_at_scale.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/resource_estimation_at_scale.ipynb): a linear system, heat flow, two spin chains, a parity Hamiltonian and a three-variable quadratic, planned only | QLS, LCHS, QCELS, FixedGCIM, QHD in one-hot and binary encodings | none. Planned circuits reach 109 qubits in total, for LCHS at 100 system qubits | about 22 s | `aer,notebook` |

Run times were measured on an Apple M3 Max with 36 GiB of memory (Python 3.12.14, Qiskit 2.5.2, Aer 0.17.2). `python -m nwqlib algorithms` lists every method with the outputs it supports, and `python -m nwqlib card qls` prints one method's declared inputs, references and limitations. [Why NWQLib](why_nwqlib.md) compares this problem-level workflow with other quantum software packages.

To connect a notebook step to its paper, open the guide of the method it calls. The guide's source map lists each implemented step with its paper equation and the function that computes it. The [code tour](CODE_TOUR.md#find-the-source-and-reason-for-a-line-of-code) explains how to follow a line of code back to its source and to the reason for any departure.

## Run the notebooks {#running-the-notebooks}

Download a notebook from the [`examples/`](https://github.com/pnnl/NWQLib/tree/main/examples) folder on GitHub, or clone the repository to get all of them:

```bash
git clone https://github.com/pnnl/NWQLib.git
```

Install NWQLib with the extras that the notebook needs, listed in the table above. The linear-system notebook, for example, needs the simulator and notebook extras:

```bash
python -m pip install "nwqlib[aer,notebook]"
```

The `tensor` extra of the linear-ODE notebook also needs `scikit_tt`, installed separately as [Install and first result](quickstart.md#install) shows. Open the notebook in Jupyter and run all cells. Every circuit that the notebooks execute on the simulator has at most 12 qubits. The parameter cell near the top holds the inputs and method settings that the text refers to.

The notebooks are generated from the percent-format sources in `examples/generators/`. The [maintenance guide](MAINTENANCE.md#notebook-generation) explains how to edit and rebuild them.

## Introductory notebooks

Each introduction solves one nontrivial problem. Its result card gives the answer, the error against a classical reference, the quantum cost (qubits, CX gates, shots) and the classical cost (memory, planning work, seconds on the computer that ran it).

Sections 1 to 5 then ask whether the problem will run, what it costs, how accurate the answer is, how large the problem can grow and what the method assumes. Section 1 runs a refusal that names the limit to change, followed by its remedy.

Section 6 shows how to substitute your own input. What NWQLib adds and the method in detail, with the parameters that control accuracy and cost, follow under Go deeper. An appendix holds the full resource estimate, or the circuit counts for the eigenvalue notebook, and a saved result that is reloaded and recomputed.

The notebook names start with the algorithms they cover and end with the mathematical problem, so a reader with an equivalent problem from another field can find them.

| Notebook | Problem | Algorithms and choices shown |
| --- | --- | --- |
| [`examples/qls_linear_system_intro.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/qls_linear_system_intro.ipynb) | Steady heat in a square plate, a 16-unknown linear system | The answer with its quantum and classical cost, a refusal by `max_degree` and its remedy, predicted against compiled CX count, the error against the polynomial's allowance, QSVT inverse polynomial, condition number and polynomial degree, a finite-shot quadratic form, Dalzell's unit-direction shortcut and its norm guess, accuracy against CX count |
| [`examples/lchs_linear_dynamics_intro.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/lchs_linear_dynamics_intro.ipynb) | A pulse carried and spread by a flow, a non-unitary linear ODE | The answer with its quantum and classical cost, a refusal by `max_quadrature_work` and its remedy, predicted against compiled CX count, the error beside the Plan's kernel and quadrature bounds, LCHS with the An–Childs–Lin (arXiv:2312.03916v2), Cauchy and Low–Somma (arXiv:2508.19238v2) kernels, Gauss and uniform quadratures, Trotter error against quadrature error, exact and matrix-product-state loading of the coefficient and initial states, a driven oscillator with a constant source |
| [`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb) | Ground-state energy of a stretched H4 chain, 8 qubits. Hamiltonian from a molecular geometry or a Pauli sum. | The answer with its quantum and classical cost, a refusal of QCELS by `max_work` and its remedy, ADAPT-GCIM (Zheng et al., npj Quantum Information 2024, arXiv:2312.07691v3), convergence against HF, MP2, CCSD and FCI, reanalysis of the stored matrix elements with another overlap cutoff, the same algorithm run as quantum circuits on H2, Chebyshev Lanczos and QCELS on the H4 Hamiltonian |
| [`examples/qhd_optimization_intro.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/qhd_optimization_intro.ipynb) | A two-valley function of two variables, and a quadratic on the unit disk | The answer with its quantum and classical cost and the budget of the gradient-descent baseline, a refusal of a 22-qubit circuit and the classical-execution remedy, Quantum Hamiltonian descent on a one-hot grid, the most probable point and the candidate, mechanism comparison with gradient descent, predicted against compiled CX count, finite-shot sampling, the effect of the schedule, an inequality constraint by an augmented-Lagrangian sequence of QHD solves with its evaluated finite-grid reference, the circuit of each round compared with its classical evolution at the same step count, slack variables for inequality constraints |

## Scientific notebooks

Each scientific notebook develops a problem from a published paper. Its first screen answers the paper's question, compares the quantum result with the references needed to interpret it and gives the quantum and classical cost.

The main path then asks whether the problem will run, what it costs, how accurate the answer is, how large the problem can grow and what the method assumes. Appendices hold the construction details and further checks.

| Notebook | Source paper | Question |
| --- | --- | --- |
| [`examples/lchs_scientific.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/lchs_scientific.ipynb) | Schleich, Kharazi, Li et al., *Arbitrary boundary conditions and constraints in quantum algorithms for differential equations via penalty projections*, [arXiv:2506.21751v1](https://arxiv.org/abs/2506.21751v1) | How much of the error in a heated rod comes from an imaginary boundary penalty, and how much from solving the penalized equation with LCHS? |
| [`examples/qls_scientific.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/qls_scientific.ipynb) | Li et al., *Potential quantum advantage for simulation of fluid dynamics*, Physical Review Research 7, 013036 (2025), [arXiv:2303.16550v3](https://arxiv.org/abs/2303.16550v3) | Does a Dalzell QLS solve of a small collision history resolve its relaxation better than a no-evolution baseline? |
| [`examples/qhd_scientific.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/qhd_scientific.ipynb) | Wu et al., *Benchmarking and Resource Analysis for Augmented-Lagrangian Quantum Hamiltonian Descent*, [arXiv:2605.12066v1](https://arxiv.org/abs/2605.12066v1) | How close do box refinement on the shifted Ackley function, and an augmented Lagrangian with refinement on a constrained Rastrigin function, come to the known optima on small binary grids, how many shots find such a point, and what separates a passed stopping test from a met accuracy target? |

## Resource estimation at scale

[`examples/resource_estimation_at_scale.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/resource_estimation_at_scale.ipynb) plans one problem per algorithm family at 80, 90 and 100 system qubits, two for QPE and one fixed GCiM trial basis. The largest of these circuits, LCHS for heat flow at 100 system qubits, has 109 qubits in total. The notebook also prices one QHD step for a three-variable quadratic on 8 to 32 grid points per variable, in the one-hot and binary encodings. At 32 points per variable, the one-hot encoding uses 96 logical qubits and the binary encoding 15. The notebook simulates no circuit.

Its opening cells give the quantum and classical cost of each 100-qubit plan. They compare the CX count of the formulas with compiled circuits of periodic QLS and periodic LCHS at 4 to 16 system qubits and of QCELS on the Ising and Heisenberg chains at 4 and 8 spins, 14 checks in all, and the counts are equal in each. The 80- to 100-qubit counts rest on the same formulas.

The notebook reports qubits, encoding queries, branches, product-formula steps, shots and CX gates with their labels (exact count, upper bound, estimate or unavailable), the error bounds that the counts buy, and the state-vector memory these sizes would need. Its summary states what the plans do not establish.

Appendix A rederives the CX counts, the QPE logical operations, the LCHS bounds, the QHD counts and evolution bounds, and the LCHS and QPE step counts independently of NWQLib. Appendix E lists every compiled comparison with its compiler settings, including three small QHD circuits and a three-qubit GCiM plan.

| Notebook | Problems | Algorithms and choices shown |
| --- | --- | --- |
| [`examples/resource_estimation_at_scale.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/resource_estimation_at_scale.ipynb) | A screened Poisson system on a ring, heat flow on a ring, periodic Ising and Heisenberg chains, and a parity Hamiltonian, at 80 to 100 system qubits, and a coupled quadratic in three variables on 8 to 32 grid points per variable | Each plan's quantum and classical cost, CX counts of the formulas against compiled circuits at small sizes, a refusal with its remedy, QLS degree and CX against the requested tolerance, an LCHS error budget and its Strang step count, QCELS step counts from the Hamiltonian's commutators with a commuting counterexample, the settings, shots and CX bound of a two-state GCiM basis, the largest state vector that fits on this machine, the qubit, CX and rotation tradeoff of one-hot and binary QHD with its ideal-evolution bound |
