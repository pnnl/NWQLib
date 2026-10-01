# Compile and project fault-tolerant resources

Use these explicit auxiliary operations after selecting a scientific construction. Ordinary `plan`, `compare`, `estimate`, execution, report and load operations do not invoke NWQEC or QDK. Compilation produces a separate `LogicalCompilation`; physical projection produces a separate `PhysicalProjection`. Neither replaces the `Plan` (the selected construction and its costs, computed before any circuit exists), its selected blocks or its measured observations.

Install the optional released dependencies. The example below prepares its circuit with Aer, so it needs the `aer` extra as well.

```bash
python -m pip install --only-binary=:all: "nwqlib[aer,nwqec,qre]"
```

H followed by RZ(0.1234) prepares a state whose X expectation is cos(0.1234). The `Plan` below selects eight counts shots. Preparing it builds the circuit on Aer and submits no shots:

```python
from qiskit import QuantumCircuit
from nwqlib import Expectation, plan, prepare
from nwqlib.algorithms import ExpectationMethod
from nwqlib.operators import ingest_pauli

circuit = QuantumCircuit(1)
circuit.h(0)
circuit.rz(0.1234, 0)
selected = plan(
    Expectation(state=circuit, observable=ingest_pauli([("X", 1)], num_qubits=1)),
    method=ExpectationMethod(), shots=8, seed=7,
)
prepared = prepare(selected)
```

The sections below compile this prepared body and then apply the physical model to the resulting compilation. Close the Run with `prepared.run.close()` after compiling. NWQEC is pinned to 0.1.2 and QDK to 1.32.3. Native integration has been exercised with Python 3.12 on macOS arm64 and Ubuntu Linux aarch64 with released wheels, real compiler/model calls and saved-record checks. Python 3.14 native integration remains outside this qualification. NWQEC wheel availability is separate from execution qualification, and there is no automatic source-build fallback.

## Compile an existing selected body

```python
from nwqlib.backends.nwqec import compile_logical, LogicalCompilation

compiled = compile_logical(prepared, index=0, epsilon=1e-3)
print(compiled)
compiled.save("compiled.json")
stored = LogicalCompilation.load("compiled.json")
```

`source` is a live `Prepared` or a `LogicalCircuit` returned by `nwqlib.blocks.lower_qiskit`. The latter permits block-only compilation without an Aer Run, and its index must be 0. The prepared index follows the order of `prepared.circuits` and `prepared.setting_names`. A circuit that `run.release_native()` released is reloaded from its saved QPY snapshot, as for `prepared.inspect_resources`, without lowering or transpiling it again. Compilation records the source identity, lowering context, actual layout, native basis, terminal measurement mapping, original readout and phase, input-QASM digest, dependency version, requested/effective options, counts, output artifact and elapsed seconds.

Supported targets are `clifford_t`, `pbc`, `pbc_tfuse` and `clifford_reduction`. `count_only=True` is legal only with `clifford_t`: it records synthesis-stage counts **before final fusion**, without an output circuit or known depth. Full compilation inventories the actual final artifact. For example, H;H has two H gates before fusion and none after fusion. These stages answer different questions.

`rz_err` accepts `per-gate`, `total` and `relative`. Omitted epsilon resolves to 1e-10 for per-gate and 1e-2 otherwise, and both the requested and the effective value are recorded. `keep_ccx` and `keep_cx` are fixed false. NWQEC precision behavior remains the external dependency's responsibility. The requested epsilon is not a verified achieved fidelity or a total algorithm error. Changing epsilon can change synthesis counts, and optimizer rewrites need not give a monotone count curve.

The input must have a static unitary body. Terminal computational measurements and recognized terminal Aer saves are removed only from compiler input; their original settings remain source metadata. Reset, intermediate measurement/save, classical control and unsupported nonunitary instructions reject. Existing bounded native intake checks stored nested gate definitions. QASM2 export/parser limitations report an unsupported input; choose an explicitly exportable selected gate realization. In particular, some nested gates with array parameters in the lowered signed-Pauli encoding of the [block guide](blocks.md#compose-and-lower-a-pauli-encoding) do not export directly through the current Qiskit QASM2 writer.

QASM2 and NWQEC transformations can discard global phase. Recording the source phase does not recover the compiler's missing phase; the output is auxiliary standalone/modulo-phase data, **not a phase-faithful reusable block**. Original NWQLib coherent/control paths continue to use their original selection. PBC retains raw names such as `t_pauli`, `s_pauli`, `z_pauli`, `m_pauli` and their angles in output QASM. Its all-qubit terminal Pauli frame is not the original requested observation population, and those operations are not automatically native T counts. PBC depth stays unknown.

Each compilation uses one child process with defaults of 100000 top-level input/output operations, 10 GB transport/publication bytes and 60 seconds. Nested stored input data are checked against the existing snapshot byte limit. These limits do not bound compiler internal RSS or synthesis work. Timeout/capture failures reap the child and publish no successful result, and there is no retry or automatic cap increase. Stored record display/save/load performs no compilation. Saving uses a new file and refuses to overwrite an existing archive artifact.

## Compile every setting of a Plan

A `Plan` can measure several settings with one circuit each, for example the queries of a sampled QCELS or RFE `Plan` (`shots=N`) or the measurement bases of an Expectation. An exact QCELS or RFE `Plan` reads every query from one trajectory, so it has one setting. For a static `Plan`, `prepare` prepares only the first setting by default, so only index 0 has a circuit. `prepare(..., settings="all")` prepares every static setting on local Aer without submitting, and `setting_names[i]` names the setting of circuit i. ZZ and XX need different measurement bases, so the `Plan` below has two settings and two circuits:

```python
two_bases = plan(
    Expectation(state=[1.0, 0.0, 0.0, 1.0],
                observable=ingest_pauli([("ZZ", 1), ("XX", 0.5)], num_qubits=2)),
    method=ExpectationMethod(), shots=8, seed=7,
)
prepared_all = prepare(two_bases, settings="all")
compiled_by_setting = {
    name: compile_logical(prepared_all, index=index, epsilon=1e-3)
    for index, name in enumerate(prepared_all.setting_names)
}
prepared_all.run.close()
```

Each compilation counts one execution of one circuit, and the `Plan`'s shots are not multiplied into it. Aer checks the width of each circuit against `ExecutionLimits.max_simulation_qubits` (20 by default) when it prepares the circuit, so a wider `Plan` needs `prepare(..., limits=ExecutionLimits(max_simulation_qubits=...))` even though preparation submits nothing. RWPE, Lanczos with `SensitivitySampling` and ADAPT set later queries from earlier outcomes and refuse `settings="all"`. See [prepared execution](prepared_execution.md) for the preparation rules.

## Apply one explicit physical model

```python
from nwqlib.backends.qre import estimate_physical, PhysicalProjection

projection = estimate_physical(compiled)
print(projection)
projection.save("projection.json")
stored_projection = PhysicalProjection.load("projection.json")
```

This accepts only full Clifford+T compilation with an accepted residual gate set. T count is T+TDG, width is the actual body width, and measurement count comes from the original terminal mapping. A body without terminal measurements requires an explicit `measurement_count=...` model assumption; a simulator save does not become hardware readout. An existing measurement mapping cannot be overridden. Rotation/CCZ/CCiX counts are zero because of the accepted gate basis, not because missing quantities are treated as zero.

QDK's `LogicalCounts` entry uses serial T/measurement work and omits the source Clifford schedule and T depth. Results are **conditional counts-model projections**, not runtime estimates of the original optimized schedule.

The projection concerns one compiled body. An original `Plan`'s shots, repetitions and future adaptive measurements are not automatically multiplied into these counts.

The fixed model is:

| Choice | Effective setting |
| --- | --- |
| Physical gates | `GateBased`, error rate 1e-4, one- and two-qubit gate times 100 ns and measurement time 500 ns |
| Code | `SurfaceCode`, distances 3, 5 and 7 only, with its recorded default coefficients |
| Factories | `Litinski19Factory` finite published table (Litinski, arXiv:1905.06903v3) |
| Trace transforms | `PSSPC(num_ts_per_rotation=20, ccx_magic_states=False)` and `LatticeSurgery(slow_down_factor=1.0)` |
| Error composition | UnionBound, `max_error=.01` |
| Execution controls | cache/trace backend/graph/post-processing disabled, telemetry off, one local child, 60 s timeout |

`gate_time_ns` changes both gate times, and `measurement_time_ns` changes measurement time. The code-cycle and patch formulas are those of the `SurfaceCode` model in QDK 1.32.3 (`qdk/qre/models/qec/_surface_code.py`). One syndrome cycle takes `gate_time + 4*two_qubit_gate_time + measurement_time` (1000 ns at the defaults), and a logical cycle repeats it d times. A distance-d patch uses `2*d*d-1` physical qubits (`d*d` data and `d*d-1` syndrome qubits). The error rate of one lattice-surgery step on one patch is `0.03 * (p/0.01)**((d+1)//2)`, with p the largest of the H, CNOT and measurement error rates. QDK cites three papers for these formulas, and the [backend source map](backends.md#source-map) states which parts of the model each paper supports. Reported runtime converts QDK nanoseconds to seconds. Each entry keeps the source of its actual code and factory, the factory populations and the model properties. No price or measured hardware calibration is inferred.

An empty feasible set is a valid result. For sufficiently short workloads, this model can find no factory that completes within its modeled runtime; the bounded diagnostic is retained. Distances, factories and error budget do not expand automatically. The modeled execution error excludes unverified NWQEC synthesis error and other algorithm errors, so it is not total scientific accuracy.

## Estimate QHD rotations and T gates without compiling

Compilation is limited to small bodies. A QHD `Plan` with native one-hot execution publishes the arbitrary-rotation count of its circuit when it is planned, a binary `Plan` an upper bound on it, and `circuit_resources` adds a T estimate for a stated synthesis budget together with the circuit's separate error sources, without building the circuit:

```python
import sympy as sp
from nwqlib import plan
from nwqlib.algorithms.qhd import QHD, QuadraticSchedule, UniformState, circuit_resources
from nwqlib.problems import Optimization

x = sp.Symbol("x", real=True)
selected = plan(
    Optimization(objective=(x - sp.Rational(1, 5))**2, variables=(x,), bounds=((-1.0, 1.0),)),
    method=QHD(num_grid_points=3, num_steps=1, total_time=0.17, trotter_order=1,
               schedule=QuadraticSchedule(gamma=0.0), initial_state=UniformState()),
    shots=8, seed=7,
)
record = circuit_resources(selected, synthesis_epsilon=1e-4)
print(record.arbitrary_rotations, record.exact_t, record.synthesis.t.value.value)
```

This circuit has 8 arbitrary rotations and 2 exact T gates, and the leading-order estimate at a total budget of `1e-4` is 392.9 T. Preparing the same `Plan` and compiling it with `compile_logical(prepared, rz_err="total", epsilon=1e-4)` counted 348 T and T-inverse gates in the final artifact (NWQEC 0.1.2 and Qiskit 2.5.2). The [QHD guide](algorithms/qhd.md#fault-tolerant-resources) states the counting convention, the budgeted Clifford replacement, the assumptions of the estimate and the totals over an augmented-Lagrangian run or a box refinement.
