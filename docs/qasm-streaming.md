# Export OpenQASM

<a id="direct-qasm3-artifacts"></a>NWQLib writes OpenQASM text in two ways. `export_qasm` exports a Qiskit circuit that is already built, such as a circuit NWQLib prepared. `write_qasm3_file` writes the construction of a `Plan` directly as OpenQASM 3, without Qiskit or any other vendor SDK. Neither executes the circuit.

## Export a built circuit

`export_qasm` needs Qiskit, for example from `pip install "nwqlib[qiskit]"`.

```python
from qiskit import QuantumCircuit
from nwqlib.io import export_qasm

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
print(export_qasm(circuit))
```

This prints:

```text
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
```

`export_qasm` returns the complete text, or with `path=` writes the file atomically and returns its `Path`. `format="qasm2"` selects OpenQASM 2. The circuits that NWQLib prepared for a backend are available as `prepared.circuit(index)` while their run is open, and `export_qasm` exports them the same way. This is the export for use with NWQ-Sim outside NWQLib or with another simulator, and NWQEC compilation reads its OpenQASM 2 output ([Choose a backend](backends.md#export-circuits)).

`max_operations=100000` checks the number of top-level instructions of the circuit before export, and `max_text_bytes=10_000_000_000` bounds the emitted UTF-8 bytes. OpenQASM 3 export streams the text but still builds a complete syntax tree in the Qiskit exporter. Qiskit's OpenQASM 2 exporter first builds the complete string, so its text limit bounds what is written, not that temporary string. Neither limit bounds expanded gate definitions or SDK workspace. Parent directories must already exist, and a failed export keeps any earlier file at the destination. Omit `path` to receive the text, for example to pass it to a compiler.

## Write a Plan's construction directly

`write_qasm3_file` writes the construction of a `Plan` as an OpenQASM 3 file, and `write_qasm3` writes it to a synchronous binary stream. Importing `nwqlib.io` and writing directly require no vendor SDK. The budget bounds every quantity the writer checks before output:

```python
from nwqlib import Expectation, plan
from nwqlib.algorithms import ExpectationMethod
from nwqlib.io import QasmWriteBudget, write_qasm3_file
from nwqlib.operators import ingest_pauli

observable = ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)
expectation_plan = plan(Expectation(state=[2.0, 2.0], observable=observable),
                        method=ExpectationMethod(), seed=7)
writer_budget = QasmWriteBudget(
    max_metadata_bytes=1000000, max_walk_steps=100000,
    max_qubits=1, max_clbits=0, max_bytes=4096,
    max_instructions=100, max_chunk_bytes=256,
)
record = write_qasm3_file(expectation_plan.construction, "preparation.qasm",
                          budget=writer_budget)
print(open("preparation.qasm").read())
```

This prints:

```text
OPENQASM 3.0;
include "stdgates.inc";
gate g0 a0 {
h a0;
}
qubit[1] q0;
g0 q0[0];
```

This Plan prepares the normalized vector proportional to (2, 2) with one H gate, and the file contains only that state preparation. The physical norm and the requested I+Z observable stay in the Plan, and writing does not measure anything. The returned write record, a `QasmWriteReceipt`, describes what was written. `write_qasm3_file` replaces the file only after the complete text and the write record have been checked.

### Writer subset {#writer-subset}

The writer version `nwqlib.direct-qasm3.v1` emits OpenQASM 3.0 with `stdgates.inc`. Identifiers are generated ASCII names, and the write record's layouts keep the original names, widths and declaration order. The order of a gate's formal parameters decides the order of its call operands, and qubit index zero is the least significant bit. Stored angles use the shortest binary64 decimal representation that round-trips, including signed adjoint phases. The phase primitive is written as `gphase`, a global phase in OpenQASM 3. Under `ctrl` it becomes a relative phase on the branch where all controls are one, so a phase in the construction is always emitted. An adjoint recipe reverses its order, turns `sdg` into `s` and negates phase angles.

| Construction | Writer behavior | Checked import with Qiskit |
| --- | --- | --- |
| Direct native occupation, exact basis-vector or full-uniform PREP, HZH, positive-zero reflection | Emits its ordered x/h/z/mc_z/phase recipe once per reachable definition | Imported as logical user gates |
| Pauli parity network (`pauli.parity`) | Emits its ordered sdg/h/cx recipe once per reachable definition | Imported as logical user gates |
| Pauli group readout basis (`pauli.group_basis`) | Emits its ordered sdg/h recipe once per reachable definition | Imported as logical user gates |
| Control and/or adjoint | Prepends the control, reverses the adjoint recipe, negates the phase and keeps a controlled phase | Supported, except Z with more than two controls in total |
| Bound Repeat | One signed int[64] loop for each positive count up to 2**63-1, and no body for a zero count | Total visits and operations, derived from the source, are checked before import and loop unrolling |
| Sequence, CoherentRegion | Keeps the order | Supported |
| Allocate/Release | Declaration in the zero state, no emitted discard, and reallocation rejected | Same file meaning |
| Computational Measure/Reset | Explicit per-bit statements at their original positions | Supported |
| One root batch setting, no axes | One template, with repetitions, setting label and observation kind kept in the write record | No measurement or shot loop |
| Zero-width identity | No zero-width register or empty-operand gate call | Supported |
| Opaque recipes, including generic or prefix PREP, SELECT and label readout | Rejected before output | Unavailable |
| Parallel, nested batches, dynamic, host or timing nodes, non-bit classical values, unbound or parameterized calls | Rejected before output | Unavailable |

Direct writing does not translate the circuit to a backend's gates, synthesize any block's circuit, call a backend or call Qiskit's `dumps`. The subset of `write_qasm3` implies no compatibility with Lanczos, GCiM, full LCHS or QLS constructions, or with NWQ-Sim.

### Bounds and completion {#bounds-and-completion}

`max_metadata_bytes` bounds a conservative estimate of the JSON size of the construction graph and of the final write record, each separately. `max_walk_steps` bounds each walk of the graph and the static counting and emission passes. Both checks precede any output, and the input graph is checked before hashing, JSON work and layout allocation. Stored data is bounded by the checked graph, recipes, layouts and output chunk, independent of Repeat multiplicity and of the total output bytes. Static repeated references still use checked traversal and emission work. Integer products also obey the limits of NWQLib's circuit description.

`bytes_written` counts the accepted ASCII and UTF-8 bytes, punctuation included. `emitted_instructions` counts textual gate applications, measurements and resets, including definition bodies once per text occurrence. `expanded_operations` counts direct primitive applications plus per-bit measurements and resets after loops and user calls are expanded. `dynamic_visits` separately counts visits of the circuit description, including empty calls and loops. Neither expansion counter counts shots or lower-level controlled-gate synthesis. These are syntactic counts, not hardware estimates. Native bytes and peak memory remain unknown.

A blocking binary stream must return a positive integer no larger than the offered buffer. Partial prefixes are completed before more text is produced. Cancellation is checked between emission steps and before every retry of a partial write, and a blocking write cannot be interrupted. `QasmWriteError.prefix` gives the accepted-byte count and completed-instruction count, with the original failure chained. Errors of the checks before output occur before any output. A failed stream may expose its prefix but never returns a completed write record.

Files use a temporary file in the same directory, flush and close it, then replace the destination atomically only after the write record has been built. Failure or cancellation tries to remove the temporary file and keeps an existing destination. The original `QasmWriteError` keeps its prefix and cause if closing or removal also fails. `secondary` records these failures in order, and `temporary` names a remaining temporary file when removal fails. Control-flow exceptions keep their original type and expose the secondary failures and remaining paths through exception notes. No fsync or durability after power loss is promised. A completed write record binds the source ID, the construction IDs, the subset and angle convention, layouts, batch context and counts.

## Import a written file with Qiskit {#explicit-offline-materialization}

```python
from nwqlib.io import QasmMaterializationBudget, materialize_qasm3_file

materialized = materialize_qasm3_file(
    expectation_plan.construction, "preparation.qasm", record,
    writer_budget=writer_budget,
    budget=QasmMaterializationBudget(
        max_bytes=4096, max_qubits=1, max_clbits=0,
        max_dynamic_visits=100, max_operations=100,
    ),
)
print(materialized.output_nodes)  # 1
```

Use the unchanged file, construction and write record from the same writer call. Before opening the file, this optional step derives dynamic work from the construction, compares the write record’s `construction_id` with the construction’s `content_id` and refuses a mismatch, then checks its qubit, bit, operation and node-visit counts. The comparison pairs the write record with the construction and does not compare either with the file’s text. The step checks the file’s actual size against `max_bytes`, reads the text and passes it to Qiskit’s OpenQASM 3 importer and `UnrollForLoops`. The importer supplies no bound on native bytes or peak memory.

Install `nwqlib[qasm]` to enable it, for example with `pip install "nwqlib[qasm]"`. Select `dev,qasm` together to run its tests. Importing `nwqlib.io` and writing directly remain SDK-free. The `qasm` extra includes Qiskit and both the parser and the importer, and the base install does not include Qiskit.

The checked dependency combination is the OpenQASM parser 1.0.1, qiskit-qasm3-import 0.6.0 and Qiskit 2.5.2. Small text, syntax-tree, import and unroll checks cover user gates, `ctrl`, `inv`, global phase, controlled phase and nested fixed loops. The parser accepts a larger grammar than this writer subset. The importer can eagerly synthesize Z with more than two controls, so that construct is rejected before SDK allocation while remaining valid writer output. User definitions stay logical, and no explicit gate decomposition, device transpilation, simulation or backend execution is requested. `materialized.output_nodes` is the number of top-level instructions after loops are removed, which can differ from the source's expanded primitive count. Dependency failures propagate. These checks do not prove that physical execution is equivalent or that native memory is bounded in general.

A check in a fresh Python process writes OpenQASM directly while imports of Qiskit, Aer, the OpenQASM importer and NWQLib's backends are blocked.
