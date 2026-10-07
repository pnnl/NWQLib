# Estimate resources

`nwqlib.estimate` counts the qubits, gates, shots and memory of a planned quantum algorithm before any circuit is built. Each count carries a label that says whether it is exact, an upper bound, an estimate, conditional on a stated assumption, or unavailable.

Use it to size a problem before running it. [Check device fit and run time](profiles.md) compares these counts with a machine, [Estimate fault-tolerant resources](fault-tolerant-resources.md) gives T counts and physical qubits of small compiled circuits, and the notebook [`examples/resource_estimation_at_scale.ipynb`](https://github.com/pnnl/NWQLib/blob/main/examples/resource_estimation_at_scale.ipynb) plans QLS, LCHS, QPE and GCiM at 100 system qubits. [Defaults that change your results](ENGINEERING_CONSTANTS.md#defaults-that-change-your-results) lists the defaults that decide the size, accuracy or stopping point of a computation.

## Count qubits, CX gates and shots {#count-qubits-cx-gates-and-shots}

This example plans a [QLS](algorithms/qls.md) solve of `Ax = b` that estimates `<Z>` of the normalized solution from 1000 shots, then counts its resources:

```python
from fractions import Fraction

from nwqlib import LinearSystem, NormalizedExpectation, estimate, plan
from nwqlib.algorithms import QLS
from nwqlib.operators import ingest_pauli
from nwqlib.resources import ResourceContext

problem = LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1., .25])
z = ingest_pauli([("Z", 1.0)], num_qubits=1)
selected = plan(problem, method=QLS(), seed=7, shots=1000,
                output=NormalizedExpectation(observable=z))
workload = estimate(selected, context=ResourceContext(basis="cx"))


def count(quantity):
    """The value of one count: a number, a formula name, or None."""
    fact = quantity.fact
    if fact.symbol is not None:
        return fact.symbol.name  # a formula in workload.expressions
    if fact.value is None:
        return None  # unavailable, see fact.reason
    if fact.value.kind == "rational":
        return Fraction(fact.value.numerator, fact.value.denominator)
    return fact.value.value


for metric, location in [("logical_width", "logical_device"), ("cx", None),
                         ("shots", None), ("t", None)]:
    quantity = workload.quantity(metric, location=location)
    print(f"{metric:14}{count(quantity)!s:>9}  {quantity.interpretation}")
```

Output:

```text
logical_width         4  conditional
cx              39000.0  estimate
shots              1000  exact
t                  None  unavailable
```

- **Qubits.** The circuit uses 4 logical qubits, 1 for the solution and 3 ancillas. The count is conditional because it is the peak width when the 1000 shots run one after another. [Read the labels](#meaning-of-each-quantity) explains how to state that schedule.
- **CX gates.** The estimate is 39 CX gates per circuit times 1000 shots. Gate counts are totals over the whole workload, so a plan with `shots=1` reports 39. The value is an estimate because part of it comes from a synthesis model, the per-query CX model of the encoding of A ([QLS guide](algorithms/qls.md#actual-construction-and-resource-meaning)). It counts gates before cancellation and hardware routing.
- **Shots.** Exactly the 1000 requested shots.
- **T gates.** Unavailable, because no block of this circuit has a T-count rule. Compiling a small circuit to Clifford+T gives its T count ([Estimate fault-tolerant resources](fault-tolerant-resources.md)).
- **Depth.** `logical_depth` is also unavailable for this plan. The depth of the built circuit comes from [inspecting a prepared circuit](#explicit-native-inspection).

Replace `A`, `b`, the observable and `shots` with your own. The [QLS guide](algorithms/qls.md#plan-and-estimate-beyond-simulation) plans and estimates a structured system with `2**40` unknowns in the same way.

`estimate` reads the counting rules of the planned construction. It builds no circuit, transpiles nothing, calls no backend and does not choose another construction. A logical estimate is therefore distinct from an inventory of a prepared circuit ([Inspect a prepared circuit](#explicit-native-inspection)), from execution and from provider billing. LCHS can also count representative circuits of its construction ([LCHS guide](algorithms/lchs.md#resource-inspection)).

## Read the labels {#meaning-of-each-quantity}

Each count is a `ResourceQuantity`. It holds the metric, its value with unit, conditions and the kind of support behind it (`fact`), the label (`interpretation`), the gate basis, the location, what exactly it counts (`population`) and the workflow stage (`lifecycle`), which is `planned` for every count of an estimate.

| Label | Meaning |
| --- | --- |
| `exact` | An exact count. It is only as strong as the rules it came from, so an exact count from an asserted rule remains an assertion. |
| `upper_bound` | The true count is at most this value, with the same caveat as `exact`. |
| `estimate` | Derived from observed data, an empirical prediction or a numerical estimate, such as the synthesis model in the example above. |
| `conditional` | Holds only under a stated assumption, such as shots run one after another. |
| `unavailable` | No applicable counting rule exists. `fact.reason` names what is missing. |

An unavailable count is never replaced by zero, because a zero would present an incomplete estimate as a cheap and complete one. A missing gate, synthesis, workspace or probability rule each keeps its own reason.

`print(quantity)` shows the value, unit, label, basis, what is counted and the conditions the value depends on, and `print(workload)` shows every count. Neither recomputes anything:

```python
print(workload.quantity("logical_width", location="logical_device"))
```

```text
logical_width at logical_device: 4 count [conditional]
  basis=cx; lifecycle=planned; population=whole-workload peak under serial independent acquisitions
  required schedule: serial_acquisitions
  conditions: independent acquisitions share workspace only under the required serial schedule
```

A reported peak width or memory assumes that the circuit executions of a measurement batch, every shot of every circuit, run one after another and reuse the same qubits and memory. Provider jobs are a separate count, and one job can run many shots. With the default `ResourceContext(batch_schedule="unspecified")`, such a peak is `conditional` and has `required_schedule="serial_acquisitions"`, so it is not an unconditional fit. `ResourceContext(basis="cx", batch_schedule="serial")` declares that schedule. The width above then reads `4 count [exact]`, and the required schedule stays recorded for execution to follow. Peaks of data that stay resident for the whole workload, and a single circuit run once, carry no schedule condition.

For exact arithmetic, read `fact.value`, a `Rational` (`numerator`, `denominator`) or a `Float64` (`value`). A symbolic count names a formula in `workload.expressions` (`fact.symbol`), and an unavailable count has no value. Counts are real and nonnegative. Exact counts are integers, except `expected_operations`, and estimates and upper bounds may be fractional.

Labels arise in these common cases:

- A branch contributes the larger of its alternatives, so the count becomes an upper bound. NWQLib infers no branch probability and no expected number of rounds.
- An adaptive method's per-round cost stays unavailable unless an explicit `resource_envelope` of its adaptive loop (`nwqlib.ir.AdaptiveLoop`) asserts a uniform bound per round over the history. The work and workspace of its decision rule stay separate missing costs.
- `expected_operations` needs a probability model and stays unavailable without one.
- In a batch with a parameter range, a cost that changes along the range stays unavailable, and a cost that does not change is multiplied by the number of points. NWQLib does not enumerate the range.

## Choose the location of a count {#choose-the-location-of-a-count}

Gate, shot and setting counts are totals over the workload and have no location. Width and memory counts are peaks at one location, and `quantity` needs that location:

- `logical_device`, the default location of quantum registers: `logical_width` (all qubits), `system`, `clean_ancilla`, `dirty_ancilla` and `memory`.
- `host`, the classical computer: `classical_registers`, `classical_bits` and `memory`.

`workload.quantity("logical_width")` without `location=` raises `ValueError` and lists the locations available for that metric. `workload.quantities` lists every metric and location.

Each register of a `Program`, NWQLib's description of a circuit as named steps, declares its location and role (system, clean ancilla or dirty ancilla). Allocating, releasing and reusing registers determine the peaks. Declared extra quantum workspace is included, and unknown clean or dirty ancilla requirements stay unknown. A classical register stays live from its definition until the experiment ends. The width in qubits is not converted into state-vector bytes. [Will it fit in memory?](profiles.md#will-it-fit-in-memory) gives that conversion.

## Choose the gate basis {#selected-laws-and-logical-bases}

`ResourceContext(basis=...)` chooses which gate counts the estimate answers. The bases are `selected_logical` (the default, which counts the planned construction's own operations), `cx`, `clifford_t` and `toffoli`. They are alternative questions about one construction, and counts in different bases are never converted into or added to each other. In the example above, CX is unavailable in the default basis and is an estimate in `basis="cx"`.

A CX count is a formula or an estimate from the rules of the planned blocks. It is not a count of gates in a built circuit and does not include hardware routing. A block without a CX rule leaves the total unavailable, and a CX bound supplies no T, rotation, depth or workspace counts. The [source map](#source-map) lists the published counts behind these rules.

A Toffoli count and a T decomposition are alternative bases, and neither is converted into the other. An arbitrary rotation gets no fixed T count without its synthesis precision (`precision=`) and synthesis rule (`synthesis=`). A fault-tolerant counting rule can give a conditional estimate for a block without an executable circuit.

A one-qubit phase is classified by its stored angle. An angle of 0, ±pi/2, ±pi, ±3pi/2 or ±2pi is a Clifford gate, and ±pi/4 is a T or T-inverse gate. The comparison is exact, so an angle near these values stays an arbitrary rotation. For example, a controlled preparation of `[i, 0]` is an S gate on its control, with zero non-Clifford depth.

## Count shots, settings and exact evaluations {#count-shots-settings-and-exact-evaluations}

Methods measure through measurement batches (`nwqlib.ir.MeasurementBatch`). A batch runs its body, the circuit it repeats, under one or more settings. A setting is one experiment point with a label and parameter values, such as one measurement basis. Settings are independent experiments, and the batch's `repetitions` sets how many times each one runs. A terminal batch measures its settings. Its readout kind (`observation_kind`) is `counts` for sampled shots, `pauli_expectation` or `probabilities` for exact statistics, `trajectory` for one exact evaluation of the body read at its observation points, `estimated_observable` for a provider's estimate, whose shots stay unavailable, or `None` when unknown. An outer batch repeats the batches inside it. Repetitions multiply the work of the batch body, and unknown repetitions leave the work unknown.

| Count | What it counts |
| --- | --- |
| `shots` | Sampled shots of terminal settings with `counts` readout. Exact readout adds none. |
| `exact_evaluations` | Exact statistics of terminal settings with `pauli_expectation`, `probabilities` or `trajectory` readout. One grouped request is one evaluation, however many labels or bins it returns. A nonempty trajectory counts one evaluation of its body, and its readout points are counted separately. An empty request has no readout. Counts readout adds none. |
| `settings` | Visits to terminal settings. A terminal batch's own repetitions do not multiply this count, while outer repetitions and `Repeat` do. A visited setting with zero repetitions counts one visit and no readout work. An outer batch with zero repetitions visits nothing. |
| `unique_settings` | Distinct terminal settings that may be reached, identified by definition, setting and parameter values. Repeated references add nothing. Two different circuit definitions count twice even if they are equivalent. |
| `root_setting_declarations` | Settings declared by a `Program` whose root is a measurement batch. Absent for other roots. |
| `root_repetitions` | Repetitions of that root batch's body, before the readout counts of the batches inside it. |

A QHD plan with `execution="quantum"` uses one measurement setting named `qhd`. A counts measurement with `S` repetitions counts `S` shots, one setting and zero exact evaluations. An exact probability or amplitude readout counts zero shots, one setting and one exact evaluation. A kept-amplitude readout takes its size from the amplitude declaration of the experiment.

For example, an outer batch with one setting repeats, three times, an inner batch with two settings of two shots each. The estimate reports 12 shots, 6 setting visits, 2 distinct settings, 1 root setting declaration and 3 root repetitions. Steps in sequence and `Repeat` pass these readout counts on to the enclosing `Program`. With exact readout and one repetition per inner setting instead, it reports 0 shots and 6 exact evaluations. Unknown values propagate as follows:

- Unknown inner repetitions never become exactly counted outer readouts.
- A known readout kind proves the other count exactly zero, even when the repetitions are unknown.
- An unknown readout kind with positive repetitions keeps the body work but leaves both readout counts unknown.
- Zero repetitions give zero work.
- Distinct settings across a parameter range of an outer batch stay unknown, and no range is enumerated.
- An unresolved control or repetition can make `unique_settings` an upper bound.

These are the declared readout counts. They are not simulator trajectories, hardware-independent state-evolution counts or accuracy guarantees. A plan with no measurement batch, such as the default QLS or LCHS plan that reads exact amplitudes, reports zero shots, settings and exact evaluations because it declares no terminal measurement, not because running it is free. Building the circuit makes one body per batch, so `counts` with four repetitions is one body submitted with four shots, not four bodies submitted four times each. A `Repeat` step still repeats operations inside that body. Execution checks the measurement requirements and does not substitute 1 for an unknown count.

A gate count or depth of a circuit describes one prepared circuit, while the estimate multiplies gate counts by the repetitions. Depths over several repetitions or settings are summed as if every circuit ran after the previous one, which gives an upper bound on the serial workload. It is not the depth of one circuit and does not claim that the independent jobs depend on each other. An exact evaluation has zero sampled shots and does not fix how many repetitions hardware would need, so a hardware cost that needs that number can stay unknown with a reason.

## Count memory and resident data {#count-memory-and-resident-data}

The `memory` count at a location is the peak of declared bytes there. A block's declared workspace lasts for each of its calls. Calls in parallel add their workspace at a location, and calls in sequence reuse it. `ResourceContext(resident=...)` declares `Workspace` entries that stay in memory for the whole workload, each with a `purpose` such as `"input"`, `"analysis"`, `"io"`, `"materialization"` or `"stored"`, and they add to every peak. The estimate allocates none of these bytes.

Missing block or classical workspace leaves `memory` unavailable. When some declared workspace is known, `known_memory` gives the known part, and `input_bytes`, `analysis_bytes`, `io_bytes`, `materialization_bytes` and `stored_bytes` give the known parts by purpose.

`ResourceContext(capacities=...)` stores device limits for a later assessment. `estimate` compares no count with a device and claims no fit. [Check device fit and run time](profiles.md) makes that comparison, location by location.

## Save and reload an estimate {#save-and-reload-an-estimate}

An estimate is a record that saves to JSON and loads through its class:

```python
from nwqlib.resources import WorkloadEstimate

restored = WorkloadEstimate.model_validate_json(workload.model_dump_json())
assert restored == workload
```

The JSON keeps every count with its conditions and sources. The schema version and `construction_id`, the content hash of the construction, identify what the counts describe.

## Inspect a prepared circuit {#explicit-native-inspection}

`inspect_resources` counts the operations and depth of a circuit that `prepare` has built. Preparing uses Aer (`nwqlib[aer]`) and runs nothing. Continuing the example above:

```python
from nwqlib import prepare

prepared = prepare(selected)
native = prepared.inspect_resources(index=0)
print(native["num_qubits"], native["depth"])
print(native["operations"])
compiled = prepared.inspect_resources(
    index=0, transpile_options={"basis_gates": ["cx", "rz", "sx", "x"],
                                "optimization_level": 1, "seed_transpiler": 7})
print(compiled["depth"], compiled["operations"]["cx"])
prepared.run.close()
```

```text
4 48
{'h': 2, 'z': 1, 'x': 24, 'cx': 24, 'rz': 6, 'ry': 1, 'unitary': 5, 'measure': 4}
75 34
```

The prepared circuit has 4 qubits and depth 48 (Qiskit 2.5.2, Aer 0.17.2). Its 24 `cx` entries leave out the CX gates inside the 5 `unitary` gates, because inspection counts each top-level operation once under its own name. The copy transpiled with Qiskit 2.5.2 to the basis `cx, rz, sx, x` at optimization level 1 with seed 7 has depth 75 and 34 CX, compared with the estimate of 39 CX per circuit.

Inspection reads the existing circuit and names it by its content hash. It submits nothing, simulates nothing and does not copy the circuit to count it. It needs an open Run and a prepared quantum circuit, so a classical computation step of the Plan cannot be inspected. The result is a plain dictionary:

- `operations` maps each top-level operation name to its count. Measurements, resets, barriers, simulator saves, `Clifford` objects and user-defined gates are listed under their own names. A composite gate or control-flow operation counts once, and its definition or body is not expanded, so `cx` counts only top-level entries with that name.
- `total_operations`, `num_qubits`, `num_clbits` and `depth` describe the same circuit. `depth` is Qiskit's default circuit depth, which skips directives such as barriers and simulator saves.
- `circuit` names what was counted, `basis` the gate basis and `compiler` the Qiskit version and the resolved options of a transpiled copy, or `None`.

Nonempty `transpile_options` compiles one separate copy and counts that. The original circuit and readout stay unchanged. Simulator saves are removed from the copy before compiling, so its counts have no save entries. These observed counts are distinct from the formulas of the estimate, from representative samples and from physical models.

`prepared.circuits` creates detached copies of every circuit, and `prepared.circuit(index)` copies one. Use the indexed inspection above to count without those copies.

`max_operations=100000` bounds the inspected and compiled operations, and `max_bytes=10_000_000_000` bounds the known option, inventory and copy data. Options must be finite scalars, lists or dictionaries without cycles, nested at most 64 deep. The output of a compilation that expands beyond `max_operations` is refused after compiling. These limits do not bound compiler workspace, existing SDK data or process memory, which stay unknown.

## How the estimator counts {#how-the-estimator-counts}

<a id="fold-a-selected-construction"></a>
<a id="composition-stored-structure-and-lifetimes"></a>

The estimator's bookkeeping, meaning how a counting rule is matched to a block, how sequences, parallel steps, repetitions and register lifetimes combine, how readout bytes are set aside and how its own work is limited, is described in [Resource estimate bookkeeping](development/execution.md#resource-estimate-bookkeeping).

## Source map

NWQLib's own counting rules (how counts combine, how unknowns propagate, readout counts and footprints) are implemented in `resources.fold._Fold`. The registered CX rules of the blocks combine published synthesis counts as follows. Page numbers refer to the listed arXiv versions.

| Relation | Source | Location | Code |
| --- | --- | --- | --- |
| A uniformly controlled rotation with k >= 1 controls uses 2^k CX and the k = 0 rotation uses none, so a direct preparation tree uses 2^n-2 CX | Mottonen et al., quant-ph/0407010v1 | Sec. II, p. 2, and Sec. III, Eq. (7), p. 3 | `_preparation_laws.direct_preparation_cx_bound` |
| A diagonal on m qubits uses 2^m-2 CX and 2^m-1 Rz rotations | Shende, Bullock and Markov, quant-ph/0406176v5 | Theorems 7 and 8, pp. 10-11 | `_preparation_laws.direct_preparation_cx_bound`, `blocks.selection.signed_pauli_cx_bound` |
| One added control costs at most 2 CX per one-qubit gate | Shende, Bullock and Markov, quant-ph/0406176v5 | Sec. 3.1, p. 9 | `_preparation_laws.direct_preparation_controlled_cx_bound`, `blocks.selection.signed_pauli_cx_bound` |
| One added control turns a CX into a Toffoli, which costs 6 CX and one-qubit gates. No circuit of CX and one-qubit gates implements the Toffoli with fewer CX, even with ancillas | Shende and Markov, arXiv:0803.2316v1, reproducing the textbook circuit of Nielsen and Chuang (2000, ISBN 978-0-521-63503-5) | Fig. 1 and Theorem 1, p. 3 | Same code |
| A uniformly controlled one-qubit gate with a controls and its completion diagonal use 3(2^a-1) CX. The core of 2^a one-qubit gates is exact up to a diagonal and uses 2^a-1 CX, and that diagonal on a+1 qubits uses 2^(a+1)-2 | Bergholm, Vartiainen, Mottonen and Salomaa, quant-ph/0410066v2, for the core, and the diagonal row above. Qiskit `UCGate` with `up_to_diagonal=False` implements both and gives exactly this count in Qiskit 2.5.2 for a = 1 to 4 | Sec. III, pp. 3-4, and Fig. 6(a), p. 5 | `blocks.selection.signed_pauli_cx_bound` |
| Signed Pauli SELECT uses at most 3q(L-1)+max(0,L-2) CX, with L = 2^a labels | NWQLib derivation from the rows above | Docstring of the code | `blocks.selection.signed_pauli_cx_bound` |
| Pauli readout basis uses the SELECT bound without the sign diagonal | NWQLib derivation | Docstring of the code | `blocks.selection.pauli_readout_cx_bound` |
| Controlled direct PREP bound | NWQLib derivation | ENGINEERING_CONSTANTS.md, "Controlled direct PREP slot bound" | `_preparation_laws.direct_preparation_controlled_cx_bound` |
| Contiguous-uniform fast-path gate positions. With M = sum of 2^(l_j) over j = 0..k, the preparation uses one RY, k X, l_0 H, l_k-l_0 open controlled H and k-1 open controlled RY, where a controlled H needs one CX and a controlled RY two | Shukla and Vedula, arXiv:2306.11747v2, which Qiskit `UniformSuperpositionGate` implements. The per-gate expansions (two H and four phase gates around the CX of a controlled H, two RY with the two CX of a controlled RY, two X per open control) are Qiskit 2.5.2's gate definitions | Algorithm 1 (p. 4), gate counts in Sec. 2.5 (p. 14) and CX constructions in Fig. 6 (pp. 15-16) | `_preparation_laws._uniform_superposition_gate_slots` |
| Work limit of an estimate | NWQLib engineering limit | ENGINEERING_CONSTANTS.md, "Shared Program admission limits" | `resources.records.FOLD_WORK_PER_ADMISSION_STEP` |
