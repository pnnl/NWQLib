# Plan, compare and solve

Compare several Methods on one scientific problem, first by their costs before anything runs and then by their answers. A Problem holds the physical input, and a configured Method chooses the numerical approximation and the circuits. `plan` returns that choice as a [`Plan`](glossary.md#plan), and `solve` executes it and returns the Method's Result. To solve a single problem, start with the [quickstart](quickstart.md). [Choose a problem and output](problems.md) lists the Problems and what each output means.

## Compare configured methods

`compare` plans each Method for the same Problem, whose inputs are accepted once, and estimates its circuit without running it. The Hamiltonian below is a two-qubit transverse-field Ising model, given as a Qiskit `SparsePauliOp`:

```python
import nwqlib
from nwqlib.algorithms import FixedGCIM, Lanczos
from qiskit.quantum_info import SparsePauliOp

H = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)])
problem = nwqlib.Eigenproblem(A=H)
comparison = nwqlib.compare(
    problem,
    methods=(
        Lanczos(initial_state=[1, 0, 0, 0], krylov_dimension=2),
        FixedGCIM(basis=([1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1])),
    ),
    seed=7,
)
for row in comparison.rows:
    print(row.method.descriptor.method, row.reason or "planned")
    if row.plan is not None:
        print(row.estimate.quantity("logical_width", location="logical_device"))
```

```text
chebyshev_lanczos planned
logical_width at logical_device: 4 count [exact]
  basis=selected_logical; lifecycle=planned; population=simultaneous live footprint
fixed_gcim planned
logical_width at logical_device: 3 count [exact]
  basis=selected_logical; lifecycle=planned; population=simultaneous live footprint
```

`logical_width` is the peak number of logical qubits in use at once. The indented line gives the gate basis of the count, the workflow stage and what exactly is counted, as [Read the labels](resources.md#meaning-of-each-quantity) explains. `row.estimate.quantities` lists every count of the row. An unavailable count carries its reason and is distinct from zero ([Estimate resources](resources.md)).

A Method that does not support the problem gives a row with a `reason` and no `Plan`. An invalid input raises an error. `compare` keeps every row, in the order given, and ranks none of them. [Rank candidate plans](search.md) ranks rows by shots, width or predicted time.

A Method you write goes into `methods` beside the built-in ones and is compared in the same way ([Run your own circuit](own_circuit.md), [Add a method](algorithm_protocol.md)).

## Solve the compared rows {#execute-the-selected-row}

Solve each planned row and print its energy beside the exact smallest eigenvalue from NumPy:

```python
import numpy as np

exact = np.linalg.eigvalsh(H.to_matrix())[0]
for row in comparison.rows:
    if row.plan is not None:
        result = nwqlib.solve(row)
        name = row.method.descriptor.method
        print(f"{name:18} {result.eigenvalue: .6f}  exact {exact: .6f}")
```

```text
chebyshev_lanczos  -1.224745  exact -1.414214
fixed_gcim         -1.414214  exact -1.414214
```

Both rows ran with exact readout on the local Aer statevector simulator, the default described in [exact and sampled execution](#exact-and-sampled-execution). The Lanczos result is a projected estimate from the Krylov space of `krylov_dimension=2`, and it carries no claim to be the ground energy ([Chebyshev Lanczos](algorithms/lanczos.md)). Lanczos with `krylov_dimension=3` returns -1.414214 for this Hamiltonian.

When the supplied preparations miss the ground state, a projected energy from Lanczos or GCIM, or an eigencomponent energy estimated by QPE, can differ from the smallest eigenvalue. A QPE estimate refers to the eigenstates that its preparation overlaps, so it is not automatically the ground energy. Equal construction tolerances or exact readouts do not make two methods equally accurate, and they do not rank the methods.

The eigenvalue notebook (`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`, see [Examples](examples.md)) applies ADAPT-GCIM, Lanczos and QCELS to the Hamiltonian of a stretched H4 chain and compares their energies with the exact FCI reference.

To solve one row, select it by index, as in `nwqlib.solve(comparison.select(1))`. `select` returns the row unchanged. Preparation and execution use the row's `Plan` and, when `compare` was given a device profile or an Allocation, the row's `PlanEstimate` (`row.estimate`) and `Allocation`, so they neither plan again nor replace the device prediction. An `Allocation` describes the devices granted to the run ([Check device fit and run time](profiles.md)) and does not configure a backend. The backend must support the instructions and readout that the Plan uses.

`solve` also accepts a Problem with a Method, or an existing `Plan`. For direct control of execution, use `prepare`, `submit` and `Run.wait`. [Run on a backend](prepared_execution.md) explains asynchronous execution, pending work and repeated submissions. Adaptive methods make their later decisions and measurements within the same Run.

## Exact and sampled execution {#exact-and-sampled-execution}

Two choices fix how a Plan runs. Pass them to `plan` or `compare`, or to `solve` with a Problem. A `Plan` or comparison row already fixes both, so changing them needs a new Plan.

- `execution="quantum"`, the default, runs the Method's circuit. `execution="classical"` evaluates the Method's numerical model on the classical computer, without circuits.
- `shots=None`, the default, selects exact readout, with no finite sampling. `shots=N` requests N shots per readout setting. For a grouped Pauli readout, one setting is one qubit-wise commuting group.

A provider estimate, such as the result of a provider's Estimator primitive, has its own precision, and the number of raw shots behind it is unknown.

With `shots=None`, an exact simulation evaluates every requested observable that shares one coherent circuit prefix from a single simulation. The Plan declares where each observation is taken, its labels or marginals, and any readout views, which are basis changes for an intermediate readout that are undone afterward. Several non-destructive observations can belong to one experiment. After each intermediate readout view, the simulator restores the ideal state before the evolution continues. Sampled settings use fresh preparations for the shots they measure.

The [preparation record](glossary.md#preparation-record) counts every executed operation, including readout basis rotations and their inverses, and the Run checks the size of the complete readout data before submission. Readout values share the circuit execution that produced them. An exact probability or Pauli expectation may exceed its unit bound only within a binary64 roundoff window, which follows the record's operation count, the full register width and the stated exclusions ([Readout tolerance branches](error_evidence.md#readout-tolerance-branches)).

To execute a Plan, a backend must support every readout that the Plan schedules. Aer's exact statevector target supports Pauli, probability and amplitude points, registered reductions and readout views in one trajectory. NWQ-Sim CPU statevector, including its Slurm adapter, supports Pauli and probability points and registered reductions, with views at Pauli and probability points. Its intermediate reductions must be phase invariant, and amplitude output uses a separate single-endpoint experiment ([NWQ-Sim readouts](nwqsim.md#routes-and-readouts)). If a target lacks a declared readout capability required by the trajectory, NWQLib refuses that trajectory before preparing or executing its circuit.

## Interpret and save the Result

A Result holds the Method's scientific quantities, its `Plan` and the `RunData` of its execution. The observations, preparation records, saved arrays and trace events in `RunData` record what ran. A scalar output does not mean that a full state was computed or saved. `result.report()` reads only what the Result stores, including the error bounds attached to it, and runs no reference solve, check or measurement. A Result can hold an accurate projected value and still lack evidence that the value is the ground energy, or evidence of its total accuracy.

`print(result)` reads only records the Result already holds. Its first lines give the scientific value and what it means, for example that a Ritz value is not a ground-state proof and that a QHD candidate is not a global optimum. A QPE Result also describes its estimator and prepared samples there. Array outputs show at most eight elements that are already in memory, and larger or unloaded arrays show their shape, their normalization and phase convention, and their units. The next lines give the Method and the execution mode, the backend target of the first prepared circuit, how many of the collected observation records the Result used (printed as `data: k/n chunks used`), and the number of attempts, with any whose outcome is uncertain and any failed classical computations (printed as `failed host invocations`). The last line always says `accuracy not assessed`, because a Result stores no assessment and only an explicit `assess` call compares it with an accuracy criterion. [`result.report()`](api/workflow.md#nwqlib.core.analysis.Result.report) returns the same summary with the Plan, the Result and its run records as a dictionary.

```python
result.save("compared-gcim-result")
restored = nwqlib.load_result("compared-gcim-result")
print(restored.eigenvalue)  # -1.414213562373095
```

[Save, load and reanalyze results](saved_evidence.md) covers saved Results and reanalysis. [Continue an interrupted run](run_archives.md) covers continuing a Run with its inputs and iteration state. To load the saved data of a Method you wrote, pass that Method, as in `nwqlib.load_result(path, method=MyMethod)`.

## Estimate, rank or verify

| Goal | Operation | Guide |
| --- | --- | --- |
| Estimate the cost of an existing `Plan` | `nwqlib.estimate(plan)` | [Estimate resources](resources.md) |
| Evaluate a device profile and Allocation | `nwqlib.estimate(plan, profile=..., allocation=...)` | [Check device fit and run time](profiles.md) |
| Rank a finite set of existing `Plan` objects, or rescore a Comparison | `nwqlib.scan(...)` | [Rank candidate plans](search.md) |
| Check a named scientific relation using the stored data it needs | `result.verify(...)` | [Check accuracy and verify a result](verification.md) |
| List the built-in Methods | `nwqlib.methods()` | [Use the command line](cli.md#python-discovery), [Add a method](algorithm_protocol.md) |

A bound or check that no operation has computed stays unavailable. A resource ranking establishes no claim about scientific accuracy.
