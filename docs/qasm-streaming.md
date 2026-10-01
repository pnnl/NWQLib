# Direct QASM3 artifacts

`nwqlib.io.write_qasm3` writes an admitted SelectedConstruction to a synchronous binary sink. `nwqlib.io.write_qasm3_file` atomically replaces a file after complete writing and receipt validation. Public IO import and direct writing require no vendor SDK. `nwqlib.io.export_qasm` separately converts an already-built Qiskit circuit: without `path` it returns complete text; with `path` it atomically writes the destination and returns a `Path`. Its `max_operations=100000` admits the supplied top-level native instruction count before export, and `max_text_bytes=10_000_000_000` bounds emitted UTF-8 bytes. QASM3 streams the text but still builds a complete dependency AST. QASM2's dependency first constructs a complete string, so its text limit is a publication bound, not a bound on that temporary allocation. Neither control bounds expanded definitions or SDK workspace. Export failure preserves any prior destination file. Explicit compiler text consumers continue to omit `path`.

```python
from nwqlib import Expectation, plan
from nwqlib.algorithms import ExpectationMethod
from nwqlib.io import QasmWriteBudget, write_qasm3_file
from nwqlib.operators import ingest_pauli

selected = plan(
    Expectation(state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)),
    method=ExpectationMethod(), seed=7,
)
writer_budget = QasmWriteBudget(
    max_metadata_bytes=1000000, max_walk_steps=100000,
    max_qubits=1, max_clbits=0, max_bytes=4096,
    max_instructions=100, max_chunk_bytes=256,
)
receipt = write_qasm3_file(selected.construction, "preparation.qasm", budget=writer_budget)
```

This actual expectation Plan selects H for the normalized vector proportional to (2, 2). The file prepares that state; the physical norm and requested I+Z observation remain in the Plan. Writing does not collect the observation.

## Writer subset

The version `nwqlib.direct-qasm3.v1` emits OpenQASM 3.0 with `stdgates.inc`. Identifiers are generated ASCII names; receipt layouts keep original names, widths and declaration order. Formal signature order owns call operands, and qubit index zero is the least-significant bit. Stored angles use shortest round-trip binary64 decimal representation, including signed adjoint phases. The phase primitive is written as `gphase`, a global phase in OpenQASM 3. Under `ctrl` it becomes a relative phase on the branch where all controls are one, so a selected phase is always emitted. An adjoint recipe reverses its order, turns `sdg` into `s` and negates phase angles.

| Construction | Writer behavior | Offline materializer |
| --- | --- | --- |
| Direct native occupation, exact basis-vector/full-uniform PREP; HZH; positive-zero reflection | Emit selected ordered x/h/z/mc_z/phase recipe once per reachable definition | Admitted logical user gates |
| Pauli parity network (`pauli.parity`) | Emit selected ordered sdg/h/cx recipe once per reachable definition | Admitted logical user gates |
| Pauli group readout basis (`pauli.group_basis`) | Emit selected ordered sdg/h recipe once per reachable definition | Admitted logical user gates |
| Selected control and/or adjoint | Prepend control, reverse adjoint recipe, negate phase; keep controlled phase | Supported, except Z with more than two total controls |
| Bound Repeat | One signed int[64] loop for each positive count through 2**63-1; zero emits no body | Source-derived total visits/operations admitted before import and loop unrolling |
| Sequence, CoherentRegion | Preserve order | Supported |
| Allocate/Release | Initial zero-state declaration/no emitted discard; reallocation rejected | Same file semantics |
| Computational Measure/Reset | Explicit per-bit statements at their original positions | Supported |
| One root batch setting, no axes | One template; repetitions, setting label and observation kind stay in receipt | No acquisition or shot loop |
| Zero-width identity | No zero-width register or empty-operand gate call | Supported |
| Opaque recipes, including generic/prefix PREP, SELECT and label readout | Reject before output | Unavailable |
| Parallel, nested batches, dynamic/host/timing nodes, non-bit classical values, unbound/parameterized calls | Reject before output | Unavailable |

No outer native lowering, native leaf synthesis, backend call or Qiskit dumps occurs during direct writing. No Lanczos, GCiM, full LCHS/QLS or NWQ-Sim export compatibility is implied by this subset.

## Bounds and completion

`max_metadata_bytes` bounds the conservative JSON envelope of the selected graph and final receipt separately. `max_walk_steps` bounds each graph walk and the static accounting/emission traversals. Both checks precede artifact output; the input graph is checked before identity/JSON work and layout allocation. Stored data is bounded by the admitted graph, recipes, layouts and output chunk, independent of Repeat multiplicity and total output bytes. Static repeated references still consume admitted traversal/emission work. Integer products also obey IR limits.

`bytes_written` counts actual accepted ASCII/UTF-8 bytes including punctuation. `emitted_instructions` counts textual gate applications, measurement and reset, including definition bodies once per text occurrence. `expanded_operations` counts direct primitive applications plus per-bit measurement/reset after expanding loops and user calls. `dynamic_visits` separately counts IR visits, including empty calls/loops. Neither expansion counter counts acquisition shots or lower-level controlled-gate synthesis. These are syntactic counts, not hardware estimates. Native bytes and peak RSS remain unknown.

A blocking binary sink must return a positive integer no larger than the offered buffer. Partial prefixes are completed before producing more text. Cancellation is checked between emission steps and every partial-write retry; an arbitrary blocking write cannot be preempted. `QasmWriteError.prefix` provides accepted bytes, fully accepted instruction statements and their `sha256:` digest, with the original failure chained. Admission errors occur before output. Failed streams may expose their prefix but never return a completed receipt.

Files use a same-directory temporary, flush/close, then atomic replacement only after receipt construction succeeds. Failure or cancellation attempts to remove the owned temporary and preserves an existing destination. The original `QasmWriteError` keeps its prefix and cause if close or removal also fails; `secondary` records these failures in order, and `temporary` names a remaining owned file when removal fails. Control-flow exceptions keep their original type and expose secondary failures and remaining paths through exception notes. No fsync or power-loss durability is promised. Completed receipts bind source ID, selected IDs, subset/angle convention, layouts, batch context, counts and digest.

## Explicit offline materialization

```python
from nwqlib.io import QasmMaterializationBudget, materialize_qasm3_file

materialized = materialize_qasm3_file(
    selected.construction, "preparation.qasm", receipt, writer_budget=writer_budget,
    budget=QasmMaterializationBudget(
        max_bytes=4096, max_qubits=1, max_clbits=0,
        max_dynamic_visits=100, max_operations=100,
    ),
)
```

This optional operation rederives capacity from the supplied construction, re-emits bounded text into a digest sink to verify the complete receipt, reads one bounded file snapshot, and verifies its bytes before passing those same bytes to Qiskit's QASM3 importer and UnrollForLoops. A persisted receipt alone cannot authorize unrelated text. Huge dynamic workloads reject before opening the file or importing it. Required hard native-byte or peak-RSS bounds also reject before import because this consumer cannot establish them.

Install `nwqlib[qasm]` to enable this explicit consumer, for example with `pip install "nwqlib[qasm]"`. Select `dev,qasm` together for the consumer tests. Public IO import and direct writing remain SDK-free. The `qasm` extra includes Qiskit and both parser/importer requirements; base does not install Qiskit.

The checked dependency combination is OpenQASM parser 1.0.1, qiskit-qasm3-import 0.6.0 and Qiskit 2.5.2. Tiny text/AST and actual import/unroll checks cover user gates, ctrl, inv, global phase, controlled phase and nested fixed loops. Parser grammar acceptance is broader than this writer subset. The importer can eagerly synthesize Z with more than two controls, so that consumer construct is rejected before SDK allocation while remaining valid writer output. User definitions remain logical; no explicit gate decomposition, device transpilation, simulation or backend execution is requested. `materialized.output_nodes` is the actual loop-free top-level instruction count, which can differ from the source primitive expansion envelope. Dependency failures propagate; these checks do not prove physical execution equivalence or universal native memory bounds.

The fresh-process audit is `docs/scripts/check_qasm_records.py`; it also requires a swallowed forbidden-import negative control to fail. Focused tests live in `tests/test_qasm_streaming.py` and use only bounded text/AST/logical import work.
