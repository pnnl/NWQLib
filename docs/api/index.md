# Overview {#api-reference}

Each entry gives the arguments, defaults, return value and errors of a call, or the fields of a record with their meaning and unit. The [quickstart](../quickstart.md) and the guides show the same calls in context.

Import the Problems and the workflow calls from `nwqlib`, and the Methods from `nwqlib.algorithms`. This example finds the smaller eigenvalue of `[[2, 1], [1, 2]]`, whose eigenvalues are 1 and 3:

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms import Lanczos

problem = Eigenproblem(A=[[2.0, 1.0], [1.0, 2.0]])
method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
result = solve(problem, method=method, seed=7)
print(round(result.eigenvalue, 10))  # 1.0
```

Entry headings show the object's name. The anchor of an entry names a module path, usually the module that defines it, which can differ from the public import path. The import lines and the opening paragraphs of each page give the public import path of the objects you construct or call.

## Find an entry by task

| Task | Reference | Guide |
| --- | --- | --- |
| Solve a problem and read the answer | [Solve, plan and read results](workflow.md) | [Plan, compare and solve](../scientist.md), [Choose a problem and output](../problems.md) |
| Build inputs from a matrix, Pauli sum, stencil or state | [Inputs and input types](inputs.md) | [Supply inputs](../inputs.md) |
| Build a molecular Hamiltonian and its reference energies from a geometry | [`build_gcim_chemistry_problem`](algorithms/gcim.md#nwqlib.algorithms.gcim.chemistry.build_gcim_chemistry_problem) | [Chemistry inputs and reference energies](../algorithms/gcim.md#chemistry-inputs-and-reference-energies) |
| Plan, compare or estimate before running | [Plan, compare and estimate](workflow.md#planning-execution-and-reports) | [Plan, compare and solve](../scientist.md) |
| Count qubits, gates and shots of a Plan | [Resource estimates](resources.md) | [Estimate resources](../resources.md) |
| Rank candidate Plans by shots, width or predicted time | [Rank candidate plans](workflow.md#finite-search) | [Rank candidate plans](../search.md) |
| Check accuracy or verify a result | [Accuracy and verification](evidence.md) | [Check accuracy and verify a result](../verification.md) |
| Choose a backend, check device fit or export OpenQASM | [Backends, profiles and export](backends.md) | [Choose a backend](../backends.md), [Check device fit and run time](../profiles.md) |
| Run step by step, continue an interrupted run, or reopen saved work | [Run step by step](workflow.md#actual-execution-state), [Save and reopen](workflow.md#save-and-reopen) | [Run on a backend](../prepared_execution.md), [Continue an interrupted run](../run_archives.md), [Save, load and reanalyze results](../saved_evidence.md) |
| Configure a Method and read its Result | [Algorithms](#algorithms) | The guide in each row of that table |
| Build circuits from subroutines | [Subroutines](#subroutines) | [Compose blocks](../blocks.md) |
| Pass your own circuit or write a Method | [Extending NWQLib](extending.md) | [Run your own circuit](../own_circuit.md), [Add a method](../algorithm_protocol.md) |

## Algorithms

| Computes | Methods | Reference and guide |
| --- | --- | --- |
| Expectation of a Hermitian observable in a state | `ExpectationMethod` | [Pauli expectation](algorithms/expectation.md) and the guide [Finite Pauli expectation](../algorithms/expectation.md) |
| Estimate of the smallest eigenvalue from Chebyshev moments | `Lanczos` | [Lanczos](algorithms/lanczos.md) and the guide [Chebyshev Lanczos](../algorithms/lanczos.md) |
| Eigenphase or energy by phase estimation | `QCELS`, `SPE`, `RFE`, `RWPE` | [QPE](algorithms/qpe.md) and the guide [QPE](../algorithms/qpe.md) |
| Estimate of the smallest eigenvalue in a fixed or adaptively grown trial basis | `FixedGCIM`, `ADAPT` | [GCiM and ADAPT](algorithms/gcim.md) and the guide [GCiM](../algorithms/gcim.md) |
| Solution of `du/dt = -A u + b` | `LCHS` | [LCHS](algorithms/lchs.md) and the guide [LCHS](../algorithms/lchs.md) |
| Solution of `A x = b` | `QLS` | [QLS](algorithms/qls.md) and the guide [QLS](../algorithms/qls.md) |
| Candidate minimizer of an objective over a box | `QHD` | [QHD](algorithms/qhd.md) and the guide [QHD](../algorithms/qhd.md) |
| Candidate minimizer subject to constraints, and box refinement | `solve_augmented_lagrangian`, `refine_box` | [QHD constrained problems and box refinement](algorithms/qhd_constrained.md) and the guide [Constrained problems](../algorithms/qhd.md#constrained-problems) |

## Subroutines

| Builds | Reference |
| --- | --- |
| Block encodings of matrices | [Block encoding](subroutines/block_encoding.md) |
| Linear combinations of unitaries (PREP and SELECT) | [LCU](subroutines/lcu.md) |
| Coherent phase-estimation circuits | [Coherent QPE](subroutines/qpe.md) |
| QSP and QSVT phases and circuits | [QSP and QSVT](subroutines/qsp.md) |
| State preparation circuits, direct or from a matrix product state | [State preparation](subroutines/state_preparation.md) |
| Evolution under a Pauli sum | [Hamiltonian evolution](subroutines/hamiltonian_evolution.md) |
| Fermionic operator pools and their generator circuits | [Fermionic pools](subroutines/fermionic_pool.md) |
| Pauli decomposition of a matrix | [Pauli decomposition](subroutines/pauli_decomposition.md) |
| Trotter step counts with error bounds, and Trotter circuits | [Trotterization](subroutines/trotterization.md) |
