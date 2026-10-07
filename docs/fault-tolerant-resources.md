# Estimate fault-tolerant resources

<a id="compile-and-project-fault-tolerant-resources"></a>Compile a small prepared circuit to Clifford+T gates, then project its cost on a surface-code machine. Compilation gives the T count, logical qubits and depth of the compiled circuit. The projection gives physical qubits and runtime under a fixed surface-code model with stated defaults ([Models and sources](#models-and-sources)).

Compilation uses NWQEC ([github.com/pnnl/nwqec](https://github.com/pnnl/nwqec)), a toolkit for fault-tolerant circuit transpilation that synthesizes rotations with gridsynth (Ross and Selinger, arXiv:1403.2975v3). NWQEC can also produce a Pauli-based circuit (PBC). The projection uses the resource estimator of the Microsoft Quantum Development Kit (QDK, [github.com/microsoft/qdk](https://github.com/microsoft/qdk)). For a QHD plan, `circuit_resources` estimates rotations and T gates without compiling or building the circuit ([below](#estimate-qhd-rotations-and-t-gates-without-compiling)).

Compilation returns a separate `LogicalCompilation` and the projection a separate `PhysicalProjection`. Neither changes the `Plan` (the method's chosen construction and its costs, computed before any circuit exists), its blocks or its measured observations. `plan`, `compare`, `estimate`, execution, reports and loading never call NWQEC or QDK.

## Install {#install}

Install the optional released dependencies. The examples prepare their circuits with Aer, so they also need the `aer` extra.

```bash
python -m pip install --only-binary=:all: "nwqlib[aer,nwqec,qre]"
```

## Compile a prepared circuit {#compile-an-existing-selected-body}

H followed by RZ(0.1234) prepares a state whose X expectation is cos(0.1234). The `Plan` below measures it with eight shots. Preparing builds the circuit on Aer and submits no shots, and `compile_logical` compiles that circuit:

```python
from qiskit import QuantumCircuit
from nwqlib import Expectation, plan, prepare
from nwqlib.algorithms import ExpectationMethod
from nwqlib.backends.nwqec import LogicalCompilation, compile_logical
from nwqlib.operators import ingest_pauli

circuit = QuantumCircuit(1)
circuit.h(0)
circuit.rz(0.1234, 0)
observable = ingest_pauli([("X", 1)], num_qubits=1)
selected = plan(Expectation(state=circuit, observable=observable),
                method=ExpectationMethod(), shots=8, seed=7)
prepared = prepare(selected)
compiled = compile_logical(prepared, index=0, epsilon=1e-3)
prepared.run.close()
print(compiled)
compiled.save("compiled.json")
stored = LogicalCompilation.load("compiled.json")
```

```text
NWQEC 0.1.2: clifford_t, final_artifact
source=sha256:5e090080db5555983c30daa0def55546c4f0dfd274367f54c28a619ab42a65cb; requested RZ epsilon=0.001
operations: 71 count [exact]
  basis=nwqec:clifford_t; lifecycle=prepared; population=one static compiler body; final_artifact
logical_width: 1 count [exact]
  basis=nwqec:clifford_t; lifecycle=prepared; population=one static compiler body; final_artifact
logical_depth: 71 count [exact]
  basis=nwqec:clifford_t; lifecycle=prepared; population=one static compiler body; final_artifact
t: 26 count [exact]
  basis=nwqec:clifford_t; lifecycle=prepared; population=one static compiler body; final_artifact
cx: 0 count [exact]
  basis=nwqec:clifford_t; lifecycle=prepared; population=one static compiler body; final_artifact
Requested RZ epsilon is not verified achieved precision or total algorithm error.
QASM2/compiler phase loss is not recovered; output is not a phase-faithful reusable block.
Original terminal readout is source metadata; no observations were acquired.
Operation/transport caps and timeout do not bound native internal RSS or synthesis work.
```

The compiled circuit has 26 T gates on 1 logical qubit, with 71 operations and depth 71 (NWQEC 0.1.2, Qiskit 2.5.2, macOS arm64). The counts are exact for this compiled circuit and describe one execution of it, not the plan's eight shots. The `source` content hash identifies this preparation and differs from run to run. Repeated compiles of one circuit can give different Clifford counts, and hence different operation counts and depths. In the repeated compiles recorded under [NWQEC 0.1.2](dependency_issues.md#nwqec-012), the T count stayed the same. Saving writes a new file and refuses to overwrite an existing one. Displaying, saving and loading a stored record compile nothing.

The input of `compile_logical` is a live prepared circuit (`Prepared`), or a `LogicalCircuit` returned by `nwqlib.blocks.lower_qiskit`, which compiles blocks without an Aer Run and needs `index=0`. The index of a prepared circuit follows `prepared.circuits` and `prepared.setting_names`. A circuit released with `run.release_native()` is reloaded from its saved QPY copy, as for `prepared.inspect_resources`, without building or transpiling it again. The record keeps the source content hash, how the circuit was built, the qubit layout, the gate basis, the mapping of terminal measurements, the original readout and phase, the digest of the input QASM, the NWQEC version, the requested and effective options, the counts, the output circuit and the elapsed seconds.

### Choose the target and precision

- `target` is `clifford_t`, `pbc`, `pbc_tfuse` or `clifford_reduction`. `count_only=True` works only with `clifford_t`. It records the synthesis-stage counts **before final fusion**, without an output circuit or a known depth, while a full compilation counts the final output circuit. These stages answer different questions. For example, H followed by H has two H gates before fusion and none after it.
- `rz_err` is `per-gate`, `total` or `relative`. An omitted `epsilon` becomes 1e-10 for `per-gate` and 1e-2 otherwise, and both the requested and the effective value are recorded. `keep_ccx` and `keep_cx` are fixed to `False`.
- The requested epsilon is not a verified achieved precision or a total algorithm error, and NWQEC is responsible for its precision behavior ([NWQEC 0.1.2](dependency_issues.md#nwqec-012)). Changing epsilon can change the synthesis counts, and optimizer rewrites need not make the count monotone in epsilon.

### What the input must be

The circuit must have a static unitary body. Terminal computational-basis measurements and recognized terminal Aer saves are removed from the compiler input only, and their original settings stay in the record. Reset, intermediate measurements or saves, classical control and other non-unitary instructions are refused. The existing size checks on stored nested gate definitions apply. An input that the QASM2 exporter or parser cannot handle is reported as unsupported, so choose a construction whose gates export to QASM2. In particular, some nested gates with array parameters in the signed-Pauli encoding of the [block guide](blocks.md#compose-and-lower-a-pauli-encoding), once built as a Qiskit circuit, do not export directly through the current Qiskit QASM2 writer.

QASM2 and NWQEC transformations can discard global phase, and recording the source phase does not recover the phase the compiler dropped. The output is standalone data that is valid up to phase, **not a phase-faithful reusable block**. NWQLib's own coherent and controlled paths keep using the original construction. PBC output keeps NWQEC's names `t_pauli`, `s_pauli`, `z_pauli` and `m_pauli` with their angles in the output QASM. Its terminal Pauli frame on all qubits is not the originally requested measurement, its operations are not automatically T counts, and its depth stays unknown.

### Limits

Each compilation runs in one child process with defaults of 100000 top-level input or output operations (`max_operations`), 10 GB of transport and saved bytes (`max_bytes`) and 60 seconds (`timeout_seconds`). The stored data of the input circuit, including nested gate definitions and their arrays, are checked separately against the same `max_bytes`. These limits do not bound the compiler's internal memory or synthesis work. A timeout or capture failure stops the child and saves no successful result. Nothing is retried, and no limit is raised automatically.

## Project physical qubits and runtime {#apply-one-explicit-physical-model}

`estimate_physical` projects one full Clifford+T compilation through the fixed QDK model. Continuing the example above:

```python
from nwqlib.backends.qre import PhysicalProjection, estimate_physical

projection = estimate_physical(compiled)
print(projection)
projection.save("projection.json")
stored_projection = PhysicalProjection.load("projection.json")
```

```text
Physical projection under QDK's serial logical-counts model:
3012 physical qubits; 0.000189 s; modeled error=1.1926e-06
3534 physical qubits; 0.000135 s; modeled error=6.004e-06
5772 physical qubits; 8.1e-05 s; modeled error=0.000487144
Conditional serial-T/measurement counts model; original Clifford schedule and T depth are omitted.
GateBased error 1e-4 and supplied timings are model assumptions, not measured calibration.
Only distances 3, 5 and 7 and the finite published factory table were considered.
Modeled execution error excludes unverified NWQEC synthesis error and total scientific error.
```

Each line is one feasible configuration that QDK 1.32.3 returns, with its physical qubits, runtime in seconds and modeled error. `entry.configuration_json` records its code distance, factories and qubit counts:

```python
import json

for entry in projection.entries:
    model = json.loads(entry.configuration_json)
    distance = next(node["parameters"]["distance"] for node in model["nodes"]
                    if node["transform"] == "SurfaceCode")
    counts = model["properties"]
    print(distance, counts["LOGICAL_COMPUTE_QUBITS"],
          counts["PHYSICAL_COMPUTE_QUBITS"], counts["PHYSICAL_FACTORY_QUBITS"])
```

```text
7 6 582 2430
5 6 294 3240
3 6 102 5670
```

The columns are code distance, logical qubits after QDK's layout, physical qubits of those logical qubits and physical qubits of the T factories. At distance 7, for example, 6 patches of `2*7*7 - 1 = 97` physical qubits give 582, and with the 2430 factory qubits the total is 3012. QDK's layout uses 6 logical qubits for the 1 logical qubit of this compiled circuit.

The projection accepts only a full Clifford+T compilation whose remaining gates are in the accepted set. The T count is T plus T-inverse, the width is the compiled width, and the measurement count comes from the original terminal measurements. A circuit without terminal measurements needs an explicit `measurement_count=...` as a model assumption, a simulator save does not become a hardware measurement, and an existing measurement mapping cannot be overridden. Rotation, CCZ and CCiX counts are zero because the accepted gate set has none of those gates, not because missing counts are taken as zero.

QDK's `LogicalCounts` input runs T gates and measurements one after another and leaves out the Clifford schedule and T depth of the source circuit. The results are therefore **conditional projections from counts**, not runtime estimates of the original optimized schedule. A projection concerns one compiled circuit, and the plan's shots, repetitions and future adaptive measurements are not multiplied into it.

An empty feasible set is a valid result. For a short enough workload, this model can find no factory that finishes within its modeled runtime, and the projection keeps the bounded diagnostic. Distances, factories and the error budget are not widened automatically. The modeled execution error excludes unverified NWQEC synthesis error and other algorithm errors, so it is not total scientific accuracy.

## Compile every setting of a Plan {#compile-every-setting-of-a-plan}

A `Plan` can measure several settings with one circuit each, for example the queries of a sampled QCELS or RFE `Plan` (`shots=N`) or the measurement bases of an `Expectation`. An exact QCELS or RFE `Plan` reads every query from one exact evaluation of its circuit (a `trajectory` readout), so it has one setting. When a Plan's settings are fixed in advance, `prepare` prepares only the first setting by default, so only index 0 has a circuit. `prepare(..., settings="all")` prepares every fixed setting on local Aer without submitting, and `setting_names[i]` names the setting of circuit i. ZZ and XX need different measurement bases, so the `Plan` below has two settings and two circuits. Continuing the example above:

```python
zz_xx = ingest_pauli([("ZZ", 1), ("XX", 0.5)], num_qubits=2)
two_bases = plan(
    Expectation(state=[1.0, 0.0, 0.0, 1.0], observable=zz_xx),
    method=ExpectationMethod(), shots=8, seed=7,
)
prepared_all = prepare(two_bases, settings="all")
compiled_by_setting = {
    name: compile_logical(prepared_all, index=index, epsilon=1e-3)
    for index, name in enumerate(prepared_all.setting_names)
}
prepared_all.run.close()
for name, setting in compiled_by_setting.items():
    counts = {q.metric: q.fact.value.numerator for q in setting.quantities}
    print(name, counts["logical_width"], counts["t"], counts["cx"])
```

```text
group_0 2 0 2
group_1 2 0 2
```

Both settings compile to circuits on 2 logical qubits with no T gates and 2 CX gates.

Each compilation counts one execution of one circuit, and the `Plan`'s shots are not multiplied into it. Aer checks the width of each circuit against `ExecutionLimits.max_simulation_qubits` (20 by default) when it prepares the circuit, so a wider `Plan` needs `prepare(..., limits=ExecutionLimits(max_simulation_qubits=...))` even though preparing submits nothing. RWPE, Lanczos with `SensitivitySampling` and ADAPT choose later queries from earlier outcomes and refuse `settings="all"`. See [Run on a backend](prepared_execution.md) for the preparation rules.

## Estimate QHD rotations and T gates without compiling {#estimate-qhd-rotations-and-t-gates-without-compiling}

Compilation is limited to small circuits. A QHD `Plan` with `execution="quantum"` and the one-hot encoding records the arbitrary-rotation count of its circuit when it is planned, and a binary `Plan` an upper bound on it. `circuit_resources` adds a T estimate for a stated synthesis budget together with the circuit's separate error sources, without building the circuit:

```python
import sympy as sp
from nwqlib import plan
from nwqlib.algorithms.qhd import (
    QHD, QuadraticSchedule, UniformState, circuit_resources)
from nwqlib.problems import Optimization

x = sp.Symbol("x", real=True)
selected = plan(
    Optimization(objective=(x - sp.Rational(1, 5))**2, variables=(x,),
                 bounds=((-1.0, 1.0),)),
    method=QHD(num_grid_points=3, num_steps=1, total_time=0.17,
               trotter_order=1, schedule=QuadraticSchedule(gamma=0.0),
               initial_state=UniformState()),
    shots=8, seed=7,
)
record = circuit_resources(selected, synthesis_epsilon=1e-4)
print(record.arbitrary_rotations, record.exact_t,
      record.synthesis.t.value.value)
```

```text
8 2 392.90509710918684
```

This circuit has 8 arbitrary rotations and 2 exact T gates, and the leading-order estimate at a total budget of `1e-4` is 392.9 T. Preparing the same `Plan` and compiling it with `compile_logical(prepared, rz_err="total", epsilon=1e-4)` counted 348 T and T-inverse gates in the final circuit (NWQEC 0.1.2 and Qiskit 2.5.2). The [QHD guide](algorithms/qhd.md#fault-tolerant-resources) states the counting convention, the budgeted Clifford replacement, the assumptions of the estimate and the totals over an augmented-Lagrangian run or a box refinement.

## Models and sources {#models-and-sources}

The fixed QDK model is:

| Choice | Setting |
| --- | --- |
| Physical gates | `GateBased`, error rate 1e-4, one- and two-qubit gate times 100 ns and measurement time 500 ns |
| Code | `SurfaceCode`, distances 3, 5 and 7 only, with its recorded default coefficients |
| Factories | `Litinski19Factory` finite published table (Litinski, arXiv:1905.06903v3) |
| Trace transforms | `PSSPC(num_ts_per_rotation=20, ccx_magic_states=False)` and `LatticeSurgery(slow_down_factor=1.0)` |
| Error composition | UnionBound, `max_error=.01` |
| Execution controls | cache, trace backend, graph and post-processing disabled, telemetry off, one local child process, 60 s timeout |

`gate_time_ns` changes both gate times, and `measurement_time_ns` changes the measurement time. The code-cycle and patch formulas are those of the `SurfaceCode` model in QDK 1.32.3 (`qdk/qre/models/qec/_surface_code.py`). One syndrome cycle takes `gate_time + 4*two_qubit_gate_time + measurement_time` (1000 ns at the defaults), and a logical cycle repeats it d times. A distance-d patch uses `2*d*d-1` physical qubits (`d*d` data and `d*d-1` syndrome qubits). The error rate of one lattice-surgery step on one patch is `0.03 * (p/0.01)**((d+1)//2)`, with p the largest of the H, CNOT and measurement error rates. Reported runtime converts QDK's nanoseconds to seconds. Each entry keeps the source of its code and factory, the factory counts and the model properties. No price or measured hardware calibration is inferred.

QDK cites three papers for these formulas, without versions. The table below states, against the versions given, which parts of the model each paper supports. The rest is QDK's own modeling choice.

| Step | Source | Location | Code |
| --- | --- | --- | --- |
| Surface-code patch size `2*d*d - 1`, with d*d data and d*d - 1 syndrome qubits | QDK 1.32.3 `SurfaceCode.provided_isa` (`qdk/qre/models/qec/_surface_code.py`), whose comment cites Horsman et al., arXiv:1111.4022, without a version (checked against [v3](https://arxiv.org/abs/1111.4022v3)) | Sec. 7.1, pp. 18–20 (same section in v1 and v2). The rotated lattice has d*d data qubits, and its d = 5 and d = 3 examples have d*d - 1 independent stabilizers. One syndrome qubit per stabilizer gives `2*d*d - 1`. The paper also notes that at d = 3 reusing the four central syndrome qubits reduces the patch to 13 qubits, which QDK does not model | `backends.qre._estimate_qdk` |
| Syndrome cycle of one one-qubit gate time, four two-qubit gate times and one measurement time, repeated d times per logical cycle | QDK 1.32.3 `SurfaceCode.provided_isa`, whose comment cites Wang, Fowler and Hollenberg, arXiv:1009.3686, Fig. 2, without a version (checked against [v1](https://arxiv.org/abs/1009.3686v1), the only version) | Figs. 1(b) and 2, p. 1. Fig. 1(b) orders the four CNOT layers, and Fig. 2 measures one stabilizer with four CNOTs between two syndrome measurements, without initialization gates. The one-qubit gate time, which QDK calls ancilla preparation, is QDK's addition. The d cycles per logical step match arXiv:1111.4022v3, Sec. 6, which requires d rounds of error correction after each operation | `backends.qre._estimate_qdk` |
| Logical error rate `0.03 * (p/0.01)**((d+1)//2)` per patch and lattice-surgery step, p the largest of the H, CNOT and measurement error rates | QDK 1.32.3 `SurfaceCode.provided_isa`, whose comments cite Fowler et al., arXiv:1208.0928, Eqs. (10) and (11), and arXiv:1009.3686 for the threshold, without versions (checked against [v2](https://arxiv.org/abs/1208.0928v2)) | arXiv:1208.0928v2, Sec. VII, Eqs. (10) and (11), p. 11 (same numbers in v1), give the empirical approximation P_L ~ 0.03 (p/p_th)**d_e with d_e = (d+1)/2 for odd d and d/2 for even d. There P_L is the rate of logical X errors per surface-code cycle, fitted with p_th = 0.57% for that paper's circuits. The paper's footnote 14 says logical Z errors occur at about the same rate. QDK keeps 0.03, uses p_th = 0.01 and applies the rate per lattice-surgery step of d cycles. arXiv:1009.3686v1 reports thresholds of 1.1% to 1.4% (abstract, pp. 3–4), so 0.01 lies below them and is QDK's choice | `backends.qre._estimate_qdk` |
| Magic-state factory table | QDK 1.32.3, `qdk/qre/models/factories/_litinski.py`, the published table of Litinski, arXiv:1905.06903v3 | `Litinski19Factory` | `backends.qre._estimate_qdk` |

## Versions and platforms {#versions-and-platforms}

NWQEC is pinned to 0.1.2 and QDK to 1.32.3. These integrations have been run with Python 3.12 on macOS arm64 and Ubuntu Linux aarch64, with released wheels, real compiler and model calls and checks of the saved records. Python 3.14 has not been tested with them. A wheel being available for a platform does not mean the integration was tested there, and there is no automatic fallback to building NWQEC from source.
