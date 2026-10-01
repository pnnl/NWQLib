# Estimate resources

For an existing selected `plan`, use the public workflow:

```python
import nwqlib
from nwqlib.resources import ResourceContext

workload = nwqlib.estimate(plan, context=ResourceContext())
for quantity in workload.quantities:
    print(quantity)
```

This describes the selected construction without executing it. A logical estimate is distinct from native inspection, actual execution and provider billing. See [prepared execution](prepared_execution.md) for `handle.inspect_resources` and the method guide for explicit representative sampling.

To ask for the selected CX laws, choose their basis explicitly:

```python
cx_workload = nwqlib.estimate(plan, context=ResourceContext(basis="cx"))
print(cx_workload.quantity("cx"))
```

This is a selected formula/estimate where a law exists, not measured native gates or hardware routing. A missing law remains unavailable. The default `selected_logical` basis is a different cost question; it does not automatically convert arbitrary native operations to CX. A metric at a location must be read with that exact location; inspect `.quantities` for the available pairs.

For explicit Clifford+T/PBC compilation and one conditional physical projection, see [fault-tolerant resources](fault-tolerant-resources.md). These operate on a selected native body and leave this compact fold unchanged.

## Fold a selected construction

`nwqlib.resources.estimate` consumes a `SelectedConstruction`, exactly the portable Program and selected definitions accepted by the explicit logical lowering seam. It returns a `WorkloadEstimate`. It never reads native inputs, builds a circuit, transpiles, invokes a backend, or chooses a replacement recipe. The following example estimates the HZH preparation that `plan` selects for `|1>` with a configured `ExpectationMethod`. The default X recipe has one operation. The [block guide](blocks.md) describes both recipes.

```python
from nwqlib import Expectation, plan
from nwqlib.algorithms import ExpectationMethod
from nwqlib.operators import ingest_pauli
from nwqlib.problems import ingest_occupation
from nwqlib.resources import WorkloadEstimate, estimate

problem = Expectation(
    state=ingest_occupation("1", num_qubits=1),
    observable=ingest_pauli((("Z", 1.0),), num_qubits=1),
)
selected = plan(problem, method=ExpectationMethod(preparation_choice="hzh"), seed=7)
workload = estimate(selected.construction)
operations = workload.quantity("operations")
assert operations.fact.value.numerator == 3
assert operations.lifecycle == "planned"
restored = WorkloadEstimate.model_validate_json(workload.model_dump_json())
```

Load resource records through their current concrete classes, as in the last line. Their serialized schema version and identifier describe the supplied construction and quantities. The compact JSON keeps every fact and identity.

## Meaning of each quantity

`print(workload)` displays stored values and conditions; `print(workload.quantity("operations"))` selects one metric. Both are read-only formatting operations. For exact arithmetic use the quantity's `fact.value` numerator/denominator; a symbolic or unavailable fact has no numerical value to substitute. Structured records remain the source of the displayed information.

Each `ResourceQuantity` keeps its metric, unit, population, lifecycle, basis, location, interpretation and `Fact`. Availability and evidence remain separate. Exact integer inventories use `Rational` with denominator one. Numerical laws can keep `Float64`. An upper bound is distinguished from an exact count, a model estimate and an unavailable quantity. Missing gate, synthesis, workspace and probability facts have their own reasons. An unavailable quantity is never replaced by zero, because a zero would present an incomplete estimate as a cheap and complete one. The known-memory subtotal remains available when additional workspace is unknown.

Original law evidence is kept in the estimate's shared source table. Arithmetic creates a declared derived fact, never a witnessed preparation record or an execution record. User assertions remain assertions. Extrapolating observed source data becomes an empirical prediction even when another summand is asserted. The separate `derivation_sources` references identify evidence actually used for the value. `sources` also keeps ancillary conformance evidence. A literal one-X count therefore stays exact when an overlapping upper-bound law is also checked against it. An estimated zero cost remains an estimate after positive repetition, while an exact zero multiplicity removes even unavailable body work. All quantities emitted by this operation are planned. Core `Stage` names workflow phases for Limits. Resource representation is given by metric/basis/source: a native CX envelope remains planned, with its original source. Resource quantities do not infer a planning/preparation/execution phase. Compiled inventories, actual submitted/observed consumption and physical QEC, factory or routing projections require their later owners.

## Selected laws and logical bases

A `SelectedDefinition` may contain per-metric `ResourceLaw` declarations. The law must match the selected control/adjoint flags, concrete cost and call bindings, metric basis, precision and synthesis choice. A law can explicitly cover the whole accepted domain of named formal parameters through unbound_parameters. Every other actual argument must still match its fixed bindings. Empty coverage keeps exact-point matching. Covered names must be declared, distinct and disjoint from fixed bindings. Coverage alone supplies no scientific proof. Duplicate applicable laws are rejected. Applicable summaries supply their metrics. Missing metrics use the kept ordered primitives. If that fallback also determines the same metric exactly, it is kept and checked against the declared exact value or bound. Kept literal recipes are checked once per bounded selected context even when all metrics have summary laws. Law-only/native summary definitions require no decomposition traversal. Scalar conformance does not prove the selected quantum action or turn source evidence into a witnessed preparation record.

The finite native CX adapters recognize the exact registered identities of the selected block implementations for direct PREP, signed Pauli SELECT and Pauli readout. They reuse those law owners, including outer-control rotations, sign diagonals and phases, and reject excessive integer growth before a shift or multiplication. Product PREP keeps its q single-qubit component invocations through `preparation_components`. It does not receive the generic q-qubit vector bound. A CX envelope supplies no missing T, rotation, depth or workspace facts.

Bases are `selected_logical`, `cx`, `clifford_t` and `toffoli`. T, Toffoli, CCZ, arbitrary-rotation, logical-depth, T-depth and non-Clifford-depth metrics are canonical. A Toffoli inventory and a T decomposition are alternative bases. There is no automatic conversion or addition between them. Unknown rotation precision or missing synthesis cannot imply a fixed T count. FT laws can describe a conditional estimate without an executable block. The currently executable primitive vocabulary remains X/H/Z, S-adjoint, CX, multi-controlled Z and global phase. Control acts on each primitive, including global phase. Primitive arity and concrete selected target widths are checked before accounting. An effective one-qubit phase with a stored conventional value of 0, ±pi/2, ±pi, ±3pi/2 or ±2pi is classified as Clifford. ±pi/4 is the named T/T-inverse phase. These exact comparisons do not snap nearby arbitrary angles. In particular, controlled PREP of [i,0] is S on its control, with zero non-Clifford depth.

## Composition, stored structure and lifetimes

Sequence and ordered primitive recipes are serial. `Parallel` is an explicit join of direct declared unitary/preserve calls on disjoint whole registers. It sums work and takes the maximum depth. Shared-wire calls and nonunitary effects are rejected by the Program check. Logical lowering currently rejects this new node. Resource estimation does not imply a parallel compiler implementation.

Repeat multiplies work without expansion. The fold keeps one result per selected definition/call-binding context and one per graph/lifetime context, within Program's existing work and integer limits. Context fixes the basis, precision and synthesis. Selected identity fixes control, adjoint and recipe. There is no persistent global cache. Expression definitions reuse the bounded Constant/ParameterRef/add/multiply/ceildiv/min/max/comparison AST with local references. JSON does not expand shared expressions. Arbitrary parsing, symbolic expansion, general range summation and automatic simplification are unsupported. This subset needs exact scalar arithmetic rather than a separate SymPy engine. Both Program and selected/context metadata entry paths set aside room for a collection's children before pushing them on the traversal stack. A rejected collection cannot first create an over-budget copy of its references. This is a finite graph traversal limit, not an RSS or CPU guarantee.

Branches produce per-metric maxima with upper-bound scope. No branch probability or expected-round model is inferred. Adaptive body costs need an explicit `resource_envelope` asserting a uniform per-round bound over history. Otherwise they remain unknown. Policy work/workspace remain separate missing costs. MeasurementBatch settings are independent experiments. Explicit `repetitions` multiplies body work. Absent repetitions remain unknown. A terminal batch's `observation_kind` is `counts`, `pauli_expectation`, `probabilities`, or unknown (`None`). An outer batch repeats its descendants and must have kind `None`. Its summed depths are a serial workload upper envelope, not a claim that independent jobs have that causal depth or that this is a per-circuit depth. Axis-dependent costs/repetitions stay unknown because aggregate range laws are outside this scalar-law subset. Invariant costs use the compact axis cardinality.

Readout quantities have distinct stable populations:

| Metric | Population |
| --- | --- |
| `shots` | Terminal MeasurementBatch sampled shots for kind `counts`. Exact readout contributes zero. |
| `exact_evaluations` | Planned terminal exact-statistic evaluations for kind `pauli_expectation`, `probabilities` or `trajectory`. Counts contributes zero. One grouped statistic request is one evaluation, not one per returned label/bin. A nonempty trajectory folds one exact evaluation of its selected body and reports its readout items separately. An empty observation request has zero readouts. |
| `settings` | Dynamic visits to terminal setting points. A terminal batch's own repetitions do not multiply this count. Outer repetitions and Repeat do. A visited zero-repetition terminal setting contributes one visit but no readout work. A zero outer repetition visits no descendants. |
| `unique_settings` | Distinct potentially reached terminal definition/setting/effective-binding points. Repeated references do not add unique points. This is identity-based declaration inventory, not a claim of semantic equivalence between different circuit definitions. |
| `root_setting_declarations` | Setting/axis declarations of a Program whose root is a MeasurementBatch. This metric is absent for other root kinds. |
| `root_repetitions` | Body repetitions of that root batch, before descendant readout populations are counted. |

Native QHD uses one measurement setting named `qhd`. A counts measurement with `S` repetitions is charged `S` shots, one setting and zero exact evaluations. An exact probability or amplitude readout is charged zero sampled shots, one setting and one exact evaluation. A kept-amplitude readout is resolved from the selected experiment's amplitude declaration.

For one declared trajectory, `N_items = sum_k L_k`. A Pauli point contributes its number of distinct requested labels. A probability marginal on `q_k` selected wires contributes `2**q_k` logical outcomes regardless of its eventual sparse stored-entry count. An amplitude point contributes the dimension of its selected output declaration. A reduction contributes the output cardinality computed by its registered shape function from the validated parameters. Two distinct views at one boundary contribute both payloads. Positions are neither repetitions nor shots. An empty request is algebraic work with no measurement. `core/planning.py::point_items` owns this count, and assessment, `readout_shape` and `readout_bytes` use the same declared point population. The stored data set aside for a trajectory is the sum of its point declarations, fixed chunk headers and value payloads, plus one completion-metadata budget, with the complete trajectory declaration charged once in its preparation record (`_prepared_execution.trajectory_reservation`). The Run sums the representation-specific amounts set aside for the points before native preparation (`_prepared_execution._admit_readout`). A dense or adaptively sparse probability array sets aside `8 * 2**q_k` bytes, Pauli values use their JSON prototype envelope, amplitude artifacts their binary payload, and a reduction the JSON envelope of its registered output components (`ReducedValues`). A probability point's fixed header funds its array manifests, scalar summaries, publication-row growth and payload-reference rows at their largest legal encoding. The completion-metadata budget funds the job envelope and any bytes beyond that set-aside bound. With Aer's 36-character ASCII job identifier, probability and Pauli trajectories without a forecast require `58*K + 1113` metadata bytes for K points. The default 65,536-byte limit accepts 1,110 such points when their declarations, headers, binary payloads and execution workspace fit the other limits. An unsupported reduction or a reduction with unresolved output shape fails before native preparation. Stored data set aside and execution working memory have separate checks.

Per-circuit resources describe one prepared circuit. Shot-weighted resource totals also depend on the requested repetitions. Zero sampled shots for an exact evaluation does not specify a hardware repetition budget, so a hardware cost that needs that budget can remain unknown with a reason.

An outer one-setting batch repeating an inner two-setting/two-shot batch three times has 12 terminal shots, 6 terminal setting visits, 2 distinct terminal points, 1 root declaration and 3 root body repetitions. Serial subgraphs and Repeat keep these descendants. Changing the terminal declarations to exact readout with one repetition gives zero shots and six exact evaluations. Unknown inner repetitions never become exactly counted outer readouts. Known kind proves the opposite population exactly zero even with unknown terminal/outer repetitions. Unknown terminal kind with positive repetitions preserves body work but leaves both readout populations unknown. Zero repetitions still give zero work. Distinct point unions across nonterminal compact axes are outside the supported subset and stay unknown. No range is enumerated. Unresolved control/repetition can make the potentially reached unique inventory an upper bound.

These are declared readout populations, not actual simulator trajectories, hardware-independent state-evolution counts or physical accuracy guarantees. Logical lowering materializes one selected batch body: counts repetitions four means one body submitted with four shots, not four bodies each submitted four times. IR Repeat still repeats operations inside that body. Native execution must check measurement requirements. Unknown metadata is not defaulted to one. A direct non-batch Program is program-scoped: zero encoded shots/settings/exact evaluations means no terminal measurement was declared, not that a native run is free. The later assessment/execution consumer must keep this distinction.

Register `location` and `role` specify per-location logical system/clean/dirty width. Allocate, Release and reuse determine peaks. Quantum width is not converted into statevector bytes or kept compiled wires. Declared additional quantum workspace is included, and unknown clean/dirty obligations stay unknown. Classical registers remain live after definition until the experiment ends. May-live branch/zero-iteration outputs contribute a conservative storage bound. A guaranteed positive direct symbolic Repeat exports the same availability as IR. An empty Reset has no events or depth layer. A nonempty Reset has one layer. For a legal unresolved width that may be zero, its depth remains the symbolic min(1,width). Measure keeps its existing positive-width result check.

Selected `Workspace` entries are simultaneous per-invocation bytes. Parallel calls sum them per location. Serial calls reuse them. `ResourceContext.resident` holds declared input, analysis, I/O, materialization or kept payloads for the whole workload. No entry is allocated by the fold. Missing block workspace or classical-stage workspace stays unknown. `capacities` keeps `Limit` declarations for a later assessment. This operation does not combine device capacities or claim feasibility. An 8 GiB device cannot host a declared 10 GiB peak merely because a second 8 GiB device exists.

Unresolved branches keep upper peak meaning and the exclusive-path assumption. For independent batches, variable per-job peaks require a schedule: with the default `ResourceContext.batch_schedule="unspecified"`, such a known value is `conditional` and has the machine-readable `ResourceQuantity.required_schedule="serial_acquisitions"`. It is not an unconditional capacity-fit fact. An explicit `batch_schedule="serial"` discharges that planning condition and keeps the underlying exact/upper/estimate meaning. The required schedule is still recorded for the execution consumer to honor. Resident-only invariant subtotals do not inherit this condition. A single one-repetition setting introduces no additional concurrency condition.

`defined_selections` lists potentially reachable selected IDs (excluding a known zero Repeat/measurement population). Its construction-work upper subtotal covers those unique declarations. It is distinct from the actually built native definitions in the lowering's preparation record, dynamic construction replay and hidden base-gate cache. `work_units` and `evaluated_contexts` expose bounded fold bookkeeping, not CPU time or observed memory. The fold's work limit grows in proportion to the Program's own checking work, as described in the engineering constants.

The metadata-only q=60, M=10^6 witness declares the current raw Pauli mask storage as two uint64 words and a complex128 coefficient per term: 32,000,000 bytes. It accesses no payload. This excludes coalescing workspace, indices, copies and Python object overhead, and establishes no actual ingestion cost or throughput.

Concrete resource quantities are real and nonnegative. Exact cardinalities are integral. Exact `expected_operations` and estimated or upper-bound resource values may be fractional. These restrictions belong to resources, while generic `Fact` values remain usable for signed and complex quantities.

The quantum lifetime fold incrementally stores width sums by role and location. Unresolved widths keep expression multiplicities so Release removes the actual formal contribution without inventing symbolic subtraction. Wire membership and branch/loop lifetime equality belong to the Program check. The fold does not copy the complete live-wire set at every Allocate/Release. One hundred one-bit registers therefore require a compact running subtotal; symbolic peaks still preserve the actual binding context.

## Source map

The composition rules, unknown propagation, readout populations and footprints above are NWQLib's own resource contract, owned by `resources.fold._Fold`. The registered native CX envelopes combine published synthesis counts as follows. Page numbers refer to the listed arXiv versions.

| Relation | Source | Location | Code owner |
| --- | --- | --- | --- |
| A uniformly controlled rotation with k >= 1 controls uses 2^k CX and the k = 0 rotation uses none, so a direct preparation tree uses 2^n-2 CX | Mottonen et al., quant-ph/0407010v1 | Sec. II, p. 2, and Sec. III, Eq. (7), p. 3 | `_preparation_laws.direct_preparation_cx_bound` |
| A diagonal on m qubits uses 2^m-2 CX and 2^m-1 Rz rotations | Shende, Bullock and Markov, quant-ph/0406176v5 | Theorems 7 and 8, pp. 10-11 | `_preparation_laws.direct_preparation_cx_bound`, `blocks.selection.signed_pauli_cx_bound` |
| One added control costs at most 2 CX per one-qubit gate | Shende, Bullock and Markov, quant-ph/0406176v5 | Sec. 3.1, p. 9 | `_preparation_laws.direct_preparation_controlled_cx_bound`, `blocks.selection.signed_pauli_cx_bound` |
| One added control turns a CX into a Toffoli, which costs 6 CX and one-qubit gates. No circuit of CX and one-qubit gates implements the Toffoli with fewer CX, even with ancillas | Shende and Markov, arXiv:0803.2316v1, reproducing the textbook circuit of Nielsen and Chuang (2000, ISBN 978-0-521-63503-5) | Fig. 1 and Theorem 1, p. 3 | Same owners |
| A uniformly controlled one-qubit gate with a controls and its completion diagonal use 3(2^a-1) CX. The core of 2^a one-qubit gates is exact up to a diagonal and uses 2^a-1 CX, and that diagonal on a+1 qubits uses 2^(a+1)-2 | Bergholm, Vartiainen, Mottonen and Salomaa, quant-ph/0410066v2, for the core, and the diagonal row above. Qiskit `UCGate` with `up_to_diagonal=False` implements both and gives exactly this count in Qiskit 2.5.2 for a = 1 to 4 | Sec. III, pp. 3-4, and Fig. 6(a), p. 5 | `blocks.selection.signed_pauli_cx_bound` |
| Signed Pauli SELECT uses at most 3q(L-1)+max(0,L-2) CX, with L = 2^a labels | NWQLib derivation from the rows above | Owner docstring | `blocks.selection.signed_pauli_cx_bound` |
| Pauli readout basis uses the SELECT bound without the sign diagonal | NWQLib derivation | Owner docstring | `blocks.selection.pauli_readout_cx_bound` |
| Controlled direct PREP envelope | NWQLib derivation | ENGINEERING_CONSTANTS.md, "Controlled direct PREP slot bound" | `_preparation_laws.direct_preparation_controlled_cx_bound` |
| Contiguous-uniform fast-path slots. With M = sum of 2^(l_j) over j = 0..k, the preparation uses one RY, k X, l_0 H, l_k-l_0 open controlled H and k-1 open controlled RY, where a controlled H needs one CX and a controlled RY two | Shukla and Vedula, arXiv:2306.11747v2, which Qiskit `UniformSuperpositionGate` implements. The per-gate expansions (two H and four phase gates around the CX of a controlled H, two RY with the two CX of a controlled RY, two X per open control) are Qiskit 2.5.2's gate definitions | Algorithm 1 (p. 4), gate counts in Sec. 2.5 (p. 14) and CX constructions in Fig. 6 (pp. 15-16) | `_preparation_laws._uniform_superposition_gate_slots` |
| Fold work ceiling | NWQLib engineering limit | ENGINEERING_CONSTANTS.md, "Shared Program admission limits" | `resources.records.FOLD_WORK_PER_ADMISSION_STEP` |

## Explicit native inspection

`prepared.inspect_resources(index=0)` counts the operations of the actual existing native circuit and names the prepared artifact by its identifier. It performs no submission or simulation and does not copy the circuit merely to count it. Inspection requires an open Run and a live quantum artifact. A host kernel is not a circuit.

The result is a plain mapping. `operations` maps each top-level operation name in the circuit to its count, and `total_operations`, `num_qubits`, `num_clbits` and `depth` describe the same circuit. Every entry is listed under its own name, including measurements, resets, barriers, simulator saves, `Clifford` objects and user-defined gates. A composite gate or control-flow operation counts once under its own name, and its definition or body is not expanded. The count under `cx` therefore covers only top-level entries with that name, whatever their definition. `depth` is Qiskit's default circuit depth, which skips directives such as barriers and simulator saves.

`prepared.circuits` explicitly creates detached mutable circuit copies. Using that property merely to reach one inspection first copies every circuit. `prepared.circuit(index)` copies only the circuit at that index. Use the indexed inspection above for an inventory without those copies.

Supplying nonempty `transpile_options` explicitly selects one auxiliary compile. The original circuit and readout remain unchanged. Simulator saves are removed from the auxiliary copy before compilation, so its inventory has no save entries. Its `basis` names the compiled gate basis, and its `compiler` entry records the Qiskit version, the resolved optimization level and seed, and the exact supplied/default compiler configuration. This observed inventory is distinct from pre-optimization formula laws, representative samples and subsequent physical models.

`max_operations=100000` bounds inspected input/output operations. Known option, inventory and auxiliary-copy metadata are checked against `max_bytes=10_000_000_000`. Options have finite scalar/list/dict form, no cycles and depth at most 64. Compiler workspace, existing SDK payloads and process RSS remain unknown. An auxiliary compilation may refuse its output after compilation if the compiler expands beyond the inspected-operation limit; these are not compiler RSS limits.
