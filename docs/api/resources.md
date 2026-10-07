# Resource estimates

Count the qubits, gates, depth, measurements and memory of a planned computation before running it, and read each count with its label. The [Estimate resources](../resources.md) guide explains the counting rules, and [Estimate fault-tolerant resources](../fault-tolerant-resources.md) covers compiled Clifford+T counts and physical projections.

```python
import nwqlib
from nwqlib.resources import ResourceContext
```

[`nwqlib.estimate(plan)`][nwqlib.scientist.estimate] returns a [`WorkloadEstimate`][nwqlib.resources.records.WorkloadEstimate] with one quantity per metric and location. Read one with `quantity`:

```python
import nwqlib
from nwqlib import Eigenproblem
from nwqlib.algorithms import Lanczos
from nwqlib.resources import ResourceContext

problem = Eigenproblem(A=[[1.5, -1], [-1, 0.5]])
method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
selected = nwqlib.plan(problem, method=method, seed=7)
workload = nwqlib.estimate(selected, context=ResourceContext(basis="cx"))

cx = workload.quantity("cx")
width = workload.quantity("logical_width", location="logical_device")
operations = workload.quantity("operations")
print(cx.interpretation, cx.fact.value.numerator, cx.fact.unit.symbol)
print(width.interpretation, width.fact.value.numerator)
print(operations.interpretation)
```

```text
upper_bound 3 count
exact 2
unavailable
```

The Plan uses at most 3 CX gates and exactly 2 qubits. No rule of its circuit blocks gives a total operation count in this basis, so that count is unavailable rather than zero, and `operations.fact.reason` says why. `print(workload)` lists every quantity with its label and conditions. A quantity with a value also keeps its evidence in `fact.evidence`, whose kind is the weakest among the sources that determined the value, for example a proof, a numerical estimate, an observation or an assertion.

An estimate describes the planned logical circuit. It is not a count of compiled gates, which `Prepared.inspect_resources` gives for a prepared circuit, and it includes no routing, error correction or provider billing.

## Estimate a plan

The entry below is `nwqlib.resources.estimate`, the lower-level function that [`nwqlib.estimate(plan)`][nwqlib.scientist.estimate] calls on `plan.construction`. It takes the Plan's `SelectedConstruction`, not the Plan. Every entry on this page imports from `nwqlib.resources`.

::: nwqlib.resources.fold.estimate
    options:
      heading_level: 3

::: nwqlib.resources.records.WorkloadEstimate
    options:
      heading_level: 3

::: nwqlib.resources.records.ResourceContext
    options:
      heading_level: 3

## Read a quantity

Each [`ResourceQuantity`][nwqlib.resources.records.ResourceQuantity] holds its value in `fact.value`: an exact `Rational` (read `numerator` and `denominator`) or a `Float64` (read `value`). Its `interpretation` says how to read that value:

| `interpretation` | Meaning |
| --- | --- |
| `exact` | The exact count |
| `upper_bound` | The count is at most this value, for example a CX bound before gate cancellation and routing, or the maximum over the arms of a branch |
| `estimate` | Derived from observed, empirically predicted or numerically estimated evidence |
| `conditional` | Holds only under a stated condition, such as a memory peak that needs independent circuit batches to run one at a time (`required_schedule`) |
| `unavailable` | No applicable rule. `fact.reason` says why, and the value is never read as zero |

An exact count or upper bound is only as strong as its evidence, `fact.evidence.kind`, so a count derived from an asserted rule stays an assertion. The conditions of a value are in `fact.assumptions`.

Totals over the whole run have `location=None`. Widths and memory are peaks at a location, so read them with `quantity(metric, location=...)`:

| Metrics | Value | Unit |
| --- | --- | --- |
| Gates: `operations`, `single_qubit`, `two_qubit`, `controlled`, `global_phases`, `clifford`, `t`, `toffoli`, `ccz`, `arbitrary_rotations`, `cx` | Whole run | count |
| Depths: `logical_depth`, `t_depth`, `non_clifford_depth` | Whole run | count |
| Calls and measurement: `calls`, `measurements`, `resets`, `settings`, `shots`, `exact_evaluations`, `unique_settings`, `root_setting_declarations`, `root_repetitions`, `adaptive_rounds`, `expected_operations` | Whole run | count |
| Work: `classical_work`, `construction_work`, `preparation_components` | Whole run | count |
| Qubits: `logical_width`, `system`, `clean_ancilla`, `dirty_ancilla` at a register's location | Peak | count |
| Classical registers: `classical_registers`, `classical_bits` at `"host"` | Peak | count |
| Memory: `memory`, `known_memory`, `input_bytes`, `analysis_bytes`, `io_bytes`, `materialization_bytes`, `stored_bytes` at a workspace's location | Peak | byte |

The [Estimate resources](../resources.md) guide defines what `shots`, `settings`, `exact_evaluations` and the other measurement counts include.

::: nwqlib.resources.records.ResourceQuantity
    options:
      heading_level: 3

## Declare costs and memory

A circuit block declares its own costs and memory with these records. The estimate reads them and creates no allocation.

::: nwqlib.resources.records.ResourceLaw
    options:
      heading_level: 3

::: nwqlib.resources.records.Workspace
    options:
      heading_level: 3
