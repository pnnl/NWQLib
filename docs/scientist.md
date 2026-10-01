# Plan, compare and solve

A scientific problem contains the physical inputs. A configured Method selects its numerical approximation and experiments. `plan` returns that selection; `solve` executes it and returns the method's Result. The [quickstart](quickstart.md) starts with a single problem. This guide compares methods for one target.

## Compare configured methods

```python
import nwqlib
from nwqlib.algorithms import Lanczos, FixedGCIM
from nwqlib.operators import ingest_pauli
from nwqlib.problems.inputs import ingest_vector
from nwqlib.resources import ResourceContext

operator = ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)
plus = ingest_vector([1.0, 1.0])
plus_i = ingest_vector([1.0, 1j])
problem = nwqlib.Eigenproblem(A=operator, unit="Hartree")
comparison = nwqlib.compare(
    problem,
    methods=(
        Lanczos(initial_state=plus, krylov_dimension=1),
        FixedGCIM(basis=(plus, plus_i)),
    ),
    context=ResourceContext(batch_schedule="serial"),
    seed=7,
)
for row in comparison.rows:
    print(row.method.descriptor.method, row.reason or "selected")
    if row.plan is not None:
        print(row.estimate.quantity("logical_width", location="logical_device"))
        print(row.estimate.quantity("operations"))
```

The inputs are ingested once. `compare` selects each configured Method and estimates its construction without measuring anything. An unsupported scientific domain produces a row with a reason and no `Plan` (the selected construction and its costs, computed before any circuit exists). Malformed inputs raise at their informed owner. Every row stays visible, and comparison itself chooses no winner.

The Lanczos column is the normalized state |+>, and its one-dimensional projected energy is 1 Hartree. The GCIM columns |+>, |+i> span this two-dimensional space, whose eigenvalues are 0 and 2 Hartree. These independent relations describe the example's subspaces. Selecting a `Plan` does not perform either projected solve.

The eigenvalue notebook (`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`) applies ADAPT-GCIM, Lanczos and QCELS to the Hamiltonian of a stretched H4 chain and compares their energies with the exact FCI reference. When the supplied preparations omit the ground state, a valid projected or component energy can differ from the requested smallest eigenvalue. A QPE preparation determines its spectral population, so its estimate is not automatically a ground-state identification. Equal construction tolerances or exact readouts do not establish matched algorithm accuracy or a SOTA ranking.

## Execute the selected row

```python
row = comparison.select(0)
result = nwqlib.solve(row)
print(result.eigenvalue)
report = result.report()
```

Selection returns the original row. Preparation and execution use its original `Plan` and, when supplied, its original forecast and Allocation. They do not rerun selection or replace the device prediction. Allocation describes the supplied resource grant; it does not silently configure a backend. The backend must support the actual selected instructions and readout.

`solve` also accepts a problem plus Method, or an existing `Plan`. For direct lifecycle control use `prepare`, `submit` and `Run.wait`; [prepared execution](prepared_execution.md) explains asynchronous execution, pending work and repeated submissions. Adaptive methods use the same Run owner for later decisions and measurements.

## Interpret and save the Result

A Result exposes its method-specific scientific quantities, its `Plan` and actual `RunData`. Observations, preparation records, artifacts and trace events identify what ran. A scalar output does not imply that a full state was obtained or saved. `result.report()` reads existing evidence without a reference solve or another measurement. An accurate projected value can coexist with missing ground-identification or total-accuracy evidence.

```python
result.save("compared-lanczos-result")
restored = nwqlib.load_result("compared-lanczos-result")
print(restored.eigenvalue)
```

[Saved evidence](saved_evidence.md) describes current-format Result archives and explicit reanalysis. [Run archives](run_archives.md) describe continuation with the selected inputs and controller state. An external Method must be supplied explicitly when loading its executable data.

## Estimate, rank or verify

| Goal | Operation and guide |
| --- | --- |
| Estimate an existing selection | `nwqlib.estimate(plan)`, [resources](resources.md) |
| Evaluate a supplied profile and Allocation | `nwqlib.estimate(plan, profile=..., allocation=...)`, [profiles](profiles.md) |
| Rank a finite set of existing `Plan` objects or rescore a Comparison | `scan`, [finite search](search.md) |
| Check a named scientific relation using the required stored data | `result.verify(...)`, [verification](verification.md) |
| Inspect available built-in Methods | `nwqlib.methods()`, [method extension](algorithm_protocol.md) |

Each operation reports its own scope and actual work. Missing evidence remains unavailable until an explicit operation supplies it. Resource ranking alone establishes no scientific accuracy claim.
