# Example notebooks

Each notebook solves one scientific problem with NWQLib, compares the answer with a classical reference and reports what the quantum circuits cost, except the resource-estimation notebook, which plans problems too large to simulate and reports only their cost. The stored notebooks include their outputs, so they can be read without running them.

## Start from the problem you have

| You have | Notebook | Methods | Largest simulated circuit | Run time | Extras |
| --- | --- | --- | --- | --- | --- |
| A linear system, A x = b | [qls_linear_system_intro](qls_linear_system_intro.ipynb): steady heat in a plate | QSVT inverse, Dalzell shortcut (arXiv:2406.12086v2) | 12 qubits | about 23 s | `aer,notebook` |
| A linear ODE, du/dt = −A u + b | [lchs_linear_dynamics_intro](lchs_linear_dynamics_intro.ipynb): a pulse carried and spread by a flow | LCHS kernels, quadratures and state preparation | 12 qubits | about 22 s | `aer,notebook,tensor` |
| A Hamiltonian | [gcim_lanczos_qpe_eigenvalue_intro](gcim_lanczos_qpe_eigenvalue_intro.ipynb): ground-state energy of a stretched H4 chain | ADAPT-GCIM, Chebyshev Lanczos, QCELS | 8 qubits | about 19 s | `aer,notebook,chemistry` |
| An objective, bounds and optional constraints | [qhd_optimization_intro](qhd_optimization_intro.ipynb): a two-valley function of two variables and a quadratic on the unit disk | Quantum Hamiltonian descent, augmented Lagrangian for constraints | 12 qubits | about 30 s | `aer,notebook` |
| A problem from a paper: heat flow with a boundary penalty | [lchs_scientific](lchs_scientific.ipynb): Schleich, Kharazi, Li et al., arXiv:2506.21751v1 | LCHS with a constant source | 10 qubits | about 20 s | `aer,notebook` |
| A problem from a paper: collision relaxation in a fluid | [qls_scientific](qls_scientific.ipynb): Li et al., Phys. Rev. Research 7, 013036 (2025), arXiv:2303.16550v3 | Dalzell QLS | 10 qubits | about 7 s | `aer,notebook` |
| A problem from a paper: precise and constrained optimization on a coarse grid | [qhd_scientific](qhd_scientific.ipynb): Wu et al., arXiv:2605.12066v1, shifted Ackley and constrained Rastrigin | Quantum Hamiltonian descent with box refinement, augmented Lagrangian with refinement in every round | 4 qubits, and 10-qubit grid models evolved classically | about 54 s | `aer,notebook` |
| A problem at 80 to 100 system qubits, and the question of what it would cost | [resource_estimation_at_scale](resource_estimation_at_scale.ipynb): a linear system, heat flow, two spin chains, a parity Hamiltonian and a three-variable quadratic, planned only | QLS, LCHS, QCELS, FixedGCIM, QHD in one-hot and binary encodings | none. Planned circuits reach 109 qubits in total, for LCHS at 100 system qubits | about 22 s | `aer,notebook` |

The first four are introductions, and each shows how to put in your own problem. The next three develop problems from published papers. The last plans QLS, LCHS, QPE and GCiM at 80 to 100 system qubits, beyond state-vector simulation, and prices one three-variable QHD step whose one-hot encoding takes up to 96 qubits. It labels each count as exact, an upper bound, an estimate or unavailable. Run times were measured on an Apple M3 Max with 36 GiB of memory (Python 3.12.14, Qiskit 2.5.2, Aer 0.17.2).

## Run a notebook

Download a notebook from this folder, or clone the repository to get all of them:

```bash
git clone https://github.com/pnnl/NWQLib.git
```

Install NWQLib with the extras in the notebook's row, for example for the linear-system notebook:

```bash
python -m pip install "nwqlib[aer,notebook]"
```

The linear-ODE notebook compares matrix-product-state loading, which needs the `tensor` extra and `scikit_tt`, installed separately as [Install and first result](https://pnnl.github.io/NWQLib/quickstart/#install) shows. The `chemistry` extra supplies the molecular Hamiltonian of the eigenvalue notebook. Open the notebook in Jupyter and run all cells.

`python -m nwqlib algorithms` lists every method with the outputs it supports, and `python -m nwqlib card qls` prints one method's declared inputs, references and limitations. [Why NWQLib](../docs/why_nwqlib.md) compares this problem-level workflow with other quantum software packages.

## Edit a notebook

The notebooks are generated from the percent-format sources in `generators/`. After editing a source, rebuild and execute its notebook from the repository root:

```bash
python examples/generators/build_notebooks.py qls_linear_system_intro --execute
```
