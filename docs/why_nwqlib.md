# Why NWQLib

NWQLib differs from the quantum software packages under [Packages checked](#packages-checked) in five ways:

1. You state a scientific problem and the accuracy you need, and get the answer, for example x of `Ax=b` with its scale and phase, in one Result that you can verify, save and reanalyze.
2. NWQLib builds Qiskit circuits for block encodings, QSVT linear solvers and LCHS, which Qiskit 2.x does not provide.
3. It implements ADAPT-GCIM, QCELS, and the augmented-Lagrangian QHD and QHD box refinement of Wu et al., which none of the other packages checked implements as a library.
4. It estimates the resources of the construction chosen for your input before any circuit is built, and the example notebooks compare these estimates with compiled circuits.
5. Each Plan states its quantum and classical cost. Each stage checks its work and memory limits before starting the work they govern, with a refusal that identifies the stage and its constraints.

## What NWQLib adds

### A workflow at the level of the scientific problem

You state the problem, for example `LinearSystem(A=A, b=b)`, and choose the accuracy of the inverse polynomial, `QLS(epsilon_inv=0.01)`. For a dense or periodic A, `plan` computes the encoded condition number `alpha/sigma_min(A)`, with alpha the normalization of the chosen block encoding, chooses the inverse polynomial and computes its phase angles. `solve` then returns x in the original units, including its scale and phase, from an exact simulator readout, or quadratic forms and norms of x from finite shots. `verify`, `save` and reanalysis act on that same Result.

In other packages you do some of these steps yourself. The QSVT linear solvers of PennyLane, Classiq and CUDA-Q Algorithms are example code built on their QSVT building blocks, namely the PennyLane demo [QSVT in Practice](https://pennylane.ai/qml/demos/tutorial_apply_qsvt), Classiq's [QSVT matrix inversion](https://docs.classiq.io/explore/algorithms/quantum_linear_solvers/qsvt_matrix_inversion/qsvt_matrix_inversion) notebook and the [linear-system example](https://nvidia.github.io/cudaq-algorithms/examples_rst/getting_started.html#solve-a-linear-system-with-qsvt) of CUDA-Q Algorithms 0.1. They restore the scale of x in the example code. Classiq's example recovers x only up to sign, and the PennyLane demo sets the bound on the encoded condition number by hand and compares only normalized vectors. Qrisp's linear solvers, such as [`inversion`](https://qrisp.eu/reference/Algorithms/generated/qrisp.gqsp.inversion.html) and [`dalzell_inversion`](https://qrisp.eu/reference/Algorithms/generated/qrisp.gqsp.dalzell_inversion.html), are library functions, but they take a bound on the encoded condition number from the user and prepare the normalized solution. NWQLib also asks for such inputs in some cases. A Pauli-sum A needs a bound on the encoded condition number, `QLS(kappa=K)`, and the Dalzell shortcut (arXiv:2406.12086v2), like Qrisp's, needs an estimate of the solution norm.

### The algorithm layer that Qiskit 2.x does not provide

Qiskit removed its HHL linear solver (arXiv:0811.3171v3) in release 0.43 ([release notes](https://quantum.cloud.ibm.com/docs/en/api/qiskit/release-notes/0.43)), and the [circuit library](https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit_library) has no block encoding, QSVT or LCHS. The community package [qiskit-algorithms](https://github.com/qiskit-community/qiskit-algorithms) 0.4.0 has QPE, VQE and AdaptVQE but no linear solver, and the main branch of the community package [quantum_linear_solvers](https://github.com/anedumla/quantum_linear_solvers) requires `qiskit<2.0`. NWQLib builds Qiskit circuits for block encodings, QSVT linear solvers and LCHS. A construction with a finite-shot output runs on the local Aer simulator, with a noise model if you supply one, and [Choose a backend](backends.md) lists the other targets and what each has been checked for.

### Methods not implemented in the other packages checked

Within the packages checked, ADAPT-GCIM, QCELS, and the augmented-Lagrangian QHD and QHD box refinement of Wu et al., arXiv:2605.12066v1, have no library implementation. QCELS is available as [research scripts](https://github.com/zhiyanding/QCELS), and ADAPT-GCIM as the PNNL paper code [QuGCM](https://github.com/pnnl/QuGCM).

### Resource estimates of the construction chosen for your input

`estimate` adds up the resource formulas of the construction that `plan` chose, each block's formula for its gate count, before any circuit is built, and labels each count as exact, an upper bound, an estimate or unavailable. The example notebooks compare this prediction with the compiled circuit, using Qiskit 2.5.2 at optimization level 1 unless the row states otherwise. All rows come from the stored outputs of the example notebooks in this repository.

| Notebook | CX gates, predicted | CX gates, compiled |
| --- | --- | --- |
| `qls_linear_system_intro` | 7,846, an estimate | 7,710 |
| `qhd_optimization_intro` | 13,790, an upper bound | 13,790 |
| `lchs_linear_dynamics_intro` | 79,712, an estimate | 79,680 |
| `lchs_scientific` | 1,322,108, an upper bound | 622,076 |
| `qls_scientific` | not predicted, because three blocks have no CX formula | 68,276 |
| `qhd_scientific`, its 4-qubit binary circuit at optimization level 0 | 320, an upper bound | 320 |
| `resource_estimation_at_scale`, QLS on 12 and 16 system qubits at optimization level 0 | 4,062 and 6,746, estimates | 4,062 and 6,746 |
| `resource_estimation_at_scale`, LCHS on 12 and 16 system qubits at optimization level 0 | 114,738 and 138,906, upper bounds | 114,738 and 138,906 |
| `resource_estimation_at_scale`, QCELS on the Ising chain of 4 and 8 spins, over all shots, at optimization level 0 | 15,040,512 and 41,680,896, exact counts | 15,040,512 and 41,680,896 |
| `resource_estimation_at_scale`, QCELS on the Heisenberg chain of 4 and 8 spins, over all shots, at optimization level 0 | 38,535,168 and 103,415,808, exact counts | 38,535,168 and 103,415,808 |

The resource estimation notebook runs 14 such checks, periodic QLS and periodic LCHS at 4, 6, 8, 12 and 16 system qubits and QCELS on both chains at 4 and 8 spins, and finds the formula equal to the compiled count in all of them. Its Appendix E also compiles three small QHD circuits, whose CX counts equal the formula, and a three-qubit GCiM plan, whose compiled CX count is half the formula's upper bound. Its counts at 80 to 100 system qubits rest on the same formulas.

The prediction is unavailable for two notebooks. The eigenvalue notebook `gcim_lanczos_qpe_eigenvalue_intro` reports 136 circuits of 9 qubits for a quantum run but no CX count, because these circuits have no gate-count formula. The Dalzell shortcut of `qls_scientific` has three blocks without a CX formula, so that notebook reports only the compiled count.

Planning and estimation also work at sizes that cannot be simulated when A and b are given in compact form, such as a periodic stencil and a product state. The [QLS guide](algorithms/qls.md#plan-and-estimate-beyond-simulation) plans and estimates a periodic problem with `2**40` unknowns without solving or simulating it. The [resource estimation notebook](examples.md#resource-estimation-at-scale) plans QLS, LCHS, QPE and `FixedGCIM` at 80 to 100 system qubits, where the LCHS circuit has up to 109 qubits in total. It also prices one QHD step in the one-hot and binary encodings on a three-variable grid of up to 32 points per variable, which takes 96 qubits in the one-hot encoding.

### The quantum and classical cost of a Plan

A Plan carries a quantum cost and a classical cost. The quantum cost counts qubits, CX gates or circuits, and shots. The classical cost counts peak memory in bytes, planning work in work units (a count of operations computed from the input sizes before the work starts, not seconds) and stored bytes. Before circuit construction, `plan` and `estimate` state the quantum cost, the planning work and the memory that the resource formulas cover. Planning, preparation and execution check their own limits before starting the work they govern. A refusal identifies the stage and its constraints, and some stages report a lower bound on the required budget because later requirements are not yet known. The result card of every example notebook shows both costs with the measured seconds.

## What other packages also provide

These capabilities are also available in other packages. For some of them, the next section describes what NWQLib's implementation adds.

| Capability | Also available in |
| --- | --- |
| QSVT and phase-angle solvers | PennyLane [`qp.qsvt`](https://docs.pennylane.ai/en/stable/code/api/pennylane.qsvt.html) with its own angle solver, Qrisp, Classiq, [pyqsp](https://github.com/ichuang/pyqsp), [QSPPACK](https://github.com/qsppack/pyqsppack) |
| Block encoding | PennyLane (see `qp.qsvt`), Qrisp [`BlockEncoding`](https://qrisp.eu/reference/Block%20Encodings/BlockEncoding.html), [CUDA-Q Algorithms 0.1](https://nvidia.github.io/cuda-quantum/blogs/blog/2026/08/18/cudaq-algorithms-0.1/), [Qualtran](https://github.com/quantumlib/Qualtran) |
| Fault-tolerant resource estimation | [Qualtran](https://github.com/quantumlib/Qualtran), PennyLane [`qp.estimator`](https://docs.pennylane.ai/en/stable/code/qp_estimator.html) |
| QSVT linear-system solvers | PennyLane [QSVT in Practice](https://pennylane.ai/qml/demos/tutorial_apply_qsvt), Classiq [QSVT matrix inversion](https://docs.classiq.io/explore/algorithms/quantum_linear_solvers/qsvt_matrix_inversion/qsvt_matrix_inversion), the [linear-system example](https://nvidia.github.io/cudaq-algorithms/examples_rst/getting_started.html#solve-a-linear-system-with-qsvt) of CUDA-Q Algorithms 0.1, Qrisp [`inversion`](https://qrisp.eu/reference/Algorithms/generated/qrisp.gqsp.inversion.html) |
| LCHS | Classiq [LCHS notebook](https://docs.classiq.io/explore/algorithms/quantum_differential_equations_solvers/lchs/lchs) |
| Dalzell's kernel-reflection shortcut | Qrisp [`dalzell_inversion`](https://qrisp.eu/reference/Algorithms/generated/qrisp.gqsp.dalzell_inversion.html) |
| Chebyshev Lanczos and Krylov methods | Qrisp [Lanczos](https://qrisp.eu/reference/Algorithms/Lanczos.html), IBM's [Krylov quantum diagonalization tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/krylov-quantum-diagonalization), the [Krylov example](https://nvidia.github.io/cudaq-algorithms/examples_rst/getting_started.html#from-a-molecule-to-its-ground-state-energy) of CUDA-Q Algorithms 0.1 |
| Quantum Hamiltonian descent | [QHDOPT](https://github.com/PhysOpt/QHDOPT), the [code published with Leng et al.](https://github.com/jiaqileng/quantum-hamiltonian-descent) for arXiv:2303.01471v1 |

## Where NWQLib's implementations go further

**Quantum Hamiltonian descent.** The code published with Leng et al. simulates the Schrödinger equation classically and runs quadratic programs on D-Wave annealers. QHDOPT passes its Hamiltonian to [SimuQ](https://github.com/PicksPeng/SimuQ), which maps it onto D-Wave annealers, simulates it with QuTiP or compiles the evolution into IonQ native gates. NWQLib builds the whole QHD circuit itself in Qiskit and estimates its CX count before building it. QHDOPT accepts objective terms in at most two variables, and a two-variable term only as a product of one-variable functions, so its decomposition fails on `sin(x*y)` and rejects `x*y*z`. NWQLib tabulates each term on the grid of its own variables, so a term may involve any number of variables within the Method's work and byte limits.

QHDOPT accepts box bounds only. The QP notebook of the code published with Leng et al. can add linear equality constraints to its D-Wave problem as a quadratic penalty of fixed weight, which the committed notebook sets to zero. NWQLib's `solve_augmented_lagrangian` accepts equality and inequality constraints and solves a sequence of QHD problems, updating the multipliers and the penalty after each round. NWQLib's `refine_box` repeats QHD on shrinking boxes, which neither code does. QHDOPT's `optimize(refine=True)`, the default, is a different operation. It polishes each decoded sample with a classical local optimizer, SciPy's TNC by default, within the original box. NWQLib has no such classical step.

QHDOPT offers unary, one-hot and Hamming-weight encodings, and its IonQ and QuTiP routes require the one-hot encoding. The code published with Leng et al. runs QHD on D-Wave in the Hamming-weight encoding. NWQLib offers the one-hot encoding and, on a periodic grid whose number K of points per variable is a power of two, a binary encoding that stores each variable's grid index in log2(K) qubits and applies the kinetic term between quantum Fourier transforms. Neither code estimates gate counts or fault-tolerant resources. Besides the CX estimate, `circuit_resources` reports, without building the circuit, the number of arbitrary rotations of a QHD circuit and its separate error sources and, for a stated synthesis budget, a T estimate. `run_resources` totals the rotations and T estimates over an augmented-Lagrangian run or a box refinement ([QHD guide](algorithms/qhd.md#fault-tolerant-resources)).

**LCHS.** Classiq's LCHS notebook is a tutorial, and its code solves `du/dt = -A u` without a source term. It returns a normalized state, checks it only by fidelity against the normalized reference and does not check the method's requirement that the Hermitian part of A be positive semidefinite. NWQLib's `LCHS` also takes a constant source, shifts a Hermitian part that is not positive semidefinite and restores the resulting growth, and returns the solution with its norm and phase from an exact simulator readout or classical execution.

**Chebyshev Lanczos and Krylov.** Qrisp's `lanczos_alg` and IBM's tutorial discard overlap directions below a threshold that you fix in advance. With finite shots, NWQLib's default threshold follows the measured shot noise of the moments, and NWQLib reports a bound on the sampling error of the overlap matrix separately.

**Block encoding.** In PennyLane, Qrisp, CUDA-Q Algorithms and Qualtran, the construction follows the function you call or the type of your input. NWQLib's `plan` gives an exactly circulant matrix a banded encoding, otherwise chooses between the Pauli and dense encodings by their CX count per query, and records the choice with its normalization.

## Packages checked

The packages below were checked, at the versions and commits listed, in their official documentation, GitHub repositories and PyPI releases. QHDOPT and the code published with Leng et al. were checked again, at the same commits, for constraints, box refinement, encodings and resource estimates. Each statement on this page holds within this set. A statement that a package lacks a method means that a targeted search and a look through its documentation found nothing, not that an exhaustive audit was made.

| Package | Version checked |
| --- | --- |
| Qiskit, with qiskit-algorithms and quantum_linear_solvers | 2.5.2, 0.4.0, main branch |
| IBM Krylov quantum diagonalization tutorial | Qiskit/documentation commit cda70fa |
| PennyLane | 0.45.1 |
| Classiq | 1.29.1, notebooks at classiq-library commit 3c609cd |
| Qrisp | 0.9.9 |
| CUDA-Q Algorithms | 0.1.0 |
| QHDOPT, with SimuQ | 0.0.1 at commit 0f1aec6, SimuQ commit eb82d67 |
| Code published with Leng et al. | commit 250bc18 |
| Qualtran | 0.7.0 |
| pyqsp, QSPPACK (Python package qsppack) | 0.2.0, 0.4.0 |

A later release of any of these packages can change the comparison. Performance, total cost and ease of migration were not compared.
