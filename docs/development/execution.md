# Execution and storage {#execution-ownership}

This page describes how a Run prepares, submits, collects and saves work, for contributors who change that code. `_prepared_execution.Run` implements the single prepare, submit and collect lifecycle, and the public `scientist.prepare` and `scientist.submit` functions use that same object. Method controllers choose their next experiment through `Plan.resolve`. Backend adapters implement native preparation, submission, retrieval and cancellation. No other object reconstructs a request or refreshes spent limits.

## Checks before native preparation

`_prepared_execution.prepare_experiment` validates the chosen realization and filters the native block bindings to its exact construction. It then runs its checks from cheap to expensive, so an unsupported request fails before any circuit exists:

1. **Program readiness** of the chosen construction.
2. **Simulator width.** On a local simulator, the circuit width must not exceed `max_simulation_qubits`.
3. **Trajectory target support** (trajectories only). A backend that does not execute the trajectory's schedule, view or reduction rejects it, naming the backend and the unsupported feature, before the trajectory's readout shape and metadata allowance are checked and before native construction or submission.
4. **Readout items and payload bytes** (`_prepared_execution._admit_readout`). The Run subtracts the readout declaration and the completion-metadata allowance from `max_data_bytes` and divides the remaining bytes by the readout's statistic stride before it forms an item count. A trajectory reserves the sum of its point declarations, fixed chunk headers and representation-specific value payloads, plus one completion-metadata allowance for the acquisition. Each point chunk's job text and empty values envelope, together with the backend's maximum established growth of the acquisition's completion rows, must fit within that allowance. For a trajectory, the readout shape check also refuses a chosen Program with measurement registers, which a measurement needs. [Readout reservation](#readout-reservation) gives the reservation of each readout kind.
5. **Target readout support** for readouts other than trajectories.
6. **One coherent evolution** (trajectories only). The chosen IR must be one coherent evolution along the executed prefix. An explicit reset of a nonempty register, or an outcome-dependent branch or adaptive loop that starts before the last point's boundary or appears in the tail or inverse of a readout view, is refused with a message that names the selected IR.
7. **Reduction work and simulator memory** (trajectories only). The trajectory's reductions are checked against the calling Method's remaining work allowance. Its live state, every kept save (probability marginals, expectations and saved states) and the transient marginal workspace are checked against `simulator_memory_mb`, with the thread cap that the simulator is given. On the inherited static path (`Method.prepare` and `Method.execute`) the allowance comes from the Method's `reduction_allowance(plan, point, *, observation, width, run)` hook, which first applies the Method's own limit on the private workspace of its reducers.
8. **Direct state preparations.** The largest direct state preparation of the Plan must not exceed `max_direct_amplitudes`.
9. **Amplitude materialization** must fit `max_data_bytes`.
10. **Cumulative circuit cap.** A new circuit preparation must fit `max_total_circuits`.

A host-kernel readout skips steps 2 and 5 to 8.

Then, unless the caller supplies `runtime`, the runtime seed is drawn, and the preparation charge commits with the RNG position (step 2 of [Write order and crash windows](#write-order-and-crash-windows)). Lowering starts only after that commit.

## Lowering and native preparation

The strict lowering boundary rejects missing or extra identities. Runtime draws come from the saved Run RNG. Compiler options come from the chosen backend configuration, and resource contexts cannot modify lowering or execution. Native receipts keep the readout shape, the layout, the compiler and configuration identity, the seed and the acquisition provenance.

Each chosen block lowers to a Qiskit gate, which cannot hold a reset, a measurement or an `Initialize`. After lowering and before submission, the adapter checks the top-level instructions of the lowered circuit. A trajectory receipt records the body length and each point's resolved boundary.

The exact statevector target of Qiskit Aer executes a trajectory in one simulation. It inserts every point's saves at the point's boundary in the bound body before lowering, shares one save between identical requests at one boundary, and executes nothing after the last save. Its receipt counts the native evolution operations executed through the final observation, counts no save instruction, and does not multiply that count by the number of labels or points. At a readout view's boundary, Aer applies the view's tail, the view's saves and, when a later observation follows, the exact inverse of that tail. A view executes for Pauli and probability points.

## Submission

Before a new original acquisition, the shared submission code calls `Method.before_submit(plan, prepared_tuple, run=run)` once, before the intent and any backend work. This applies to a single synchronous handle and to a detached batch. Retrieval never repeats this hook. A Method uses it to keep each sample set valid under the Method's definition. For example, QPE and finite-Pauli sampling whose shot count was chosen from an `Accuracy` reject known reused fixed streams, including completed unused data and uncertain attempted work. Distinct UUIDs, circuits, seeds or equal histograms do not prove independence.

The inherited static path asks the `reduction_allowance` hook again before it submits an existing preparation, including one that a reopened Run submits, so the acquisition is checked against the allowance and limits in force at submission.

The complete fixed workload is checked before its first preparation. Each physical intent then records the circuits, shots, jobs and provider-managed sampling before execution. A failed call updates the corresponding event and physical submission together. A lost acknowledgement keeps the uncertainty and the original locators. Refresh retrieves the original job, publishes each successful observation atomically and once, and keeps earlier successful items if a later item fails. Repeated collection does not duplicate observations or refund exposure.

## Execution limits

`ExecutionLimits` controls the represented work and the stored data. The data check covers array publication, pending native output, saved native payloads and journal output, before those operations. The simulator memory field configures the simulator and is not whole-process RSS. Provider estimates keep managed sampling separate from raw shot counts. Native execution packets keep the number of evolutions they executed, independent of the shot number or the MPI rank count.

`Run.extend_limits` writes only the new current limits and one fixed-field `LimitAmendment` in the existing transaction. It snapshots the known counters before publication and publishes the live limits and history only after success. Equal caps do not write. Each amendment excludes previous history, trace and arrays, and its encoded metadata counts against `max_data_bytes`. The original journal header supplies the initial caps, and restore validates the ordered chain of old and new limits before continuation. A saved completed Result records its captured history length and byte counter, so it keeps its earlier snapshot when the Run's caps later increase.

## Readout reservation

Acquisition checks and completion use the same JSON size code. One count, probability or Pauli statistic supplies the per-item envelope, multiplied by the declared cardinality. Host scalar labels use their escaped UTF-8 lengths, and arrays use their declared byte counts. The size therefore needs no enumeration of outcomes. Strings have their escaped UTF-8 size, integers their decimal size, and finite binary64 values a conservative 32-byte envelope. A replaced row is credited with its existing encoded bytes, so a repeated batch-header update counts only its growth.

`ExecutionLimits.max_completion_metadata_bytes` reserves 65,536 additional bytes per acquisition for fields of variable size, apart from declarations, statistic bodies and binary arrays. The table gives what each readout or field reserves and when it is checked.

| Readout or field | What is reserved | When it is checked |
| --- | --- | --- |
| Declared observations, bindings and register layouts | Their known encoded sizes, exactly, in the fixed reservation | At reservation |
| Host application receipts whose JSON size bound the kernel declares in `SelectedKernel.application_bytes` | That bound, exactly, in the fixed reservation, because the Method fixes these receipts at selection | At completion, the smaller of that declaration and the returned receipts' bound is subtracted before the variable metadata is checked |
| Provider annotations and undeclared host application receipts | The completion-metadata allowance | At completion |
| Single-endpoint probability readout | Eight bytes per possible outcome, its declared chunk fields and one completion-metadata allowance | Before lowering and again before submission, the remaining chunk fields, the manifests and summaries of the largest legal probability encoding, and the known net publication and completion-row costs must fit that allowance |
| Counts, exact Pauli scalar and amplitude readouts | The per-item envelope times the declared cardinality, plus the completion-metadata allowance | On Aer, their known completion-metadata requirement is checked against the allowance before native preparation. Another backend checks them when it publishes them |
| Trajectory | Each point chunk's one-point declaration, fixed header fields and values, separately, plus one completion-metadata allowance. A probability point's fixed header funds its array manifests, scalar summaries, publication-row growth and payload-reference rows at their largest legal encoding, so the allowance funds each point chunk's job text and empty values envelope and the acquisition's other completion rows | Before native work, the Run refuses a trajectory whose required envelopes exceed the allowance, including the completion-row growth below |

Trajectory admission also reserves the backend's maximum established growth of the acquisition's completion rows ([Engineering constants](../ENGINEERING_CONSTANTS.md#budgets-and-mechanical-bounds)):

| Backend | Completion rows | Reserved growth |
| --- | --- | --- |
| Aer | Synchronous completion rows, including revision parents, job locator, timing evidence and microsecond timestamps | 1,113 bytes without a forecast, 1,044 bytes when the event already has a revision parent |
| NWQ-Sim and its Slurm backend | Detached completion | 168 bytes, or 99 bytes with a forecast |
| Any other backend | Its completion rows | Temporarily zero, until a derivation of its completion lifecycle establishes the growth of its completion rows |

A backend with unbounded job text or other unbounded completion fields still checks their actual bytes when it publishes the result. Replaced rows are credited once at publication. Completion checks a conservative JSON bound on the variable metadata delta and names `max_completion_metadata_bytes` when it is exceeded. The allowance can be increased with the other execution limits. It reserves storage without allocating an output buffer and does not restrict the number of declared Pauli terms.

## Saved Run storage

A Run with a saved folder stores its chosen inputs before any work and records controller checkpoints in SQLite. Checkpoints reference existing observations and immutable arrays instead of copying a growing history. Numerical and native caches of a Method use explicit archive hooks. Loading restores those values without rebuilding eigensystems, references, circuits or chosen inputs. Unchanged cache objects reuse their files after reopening. A changed cache file becomes current only when the frontier commits, and superseded files are then removed.

### Cache files

New cache payloads use the reserved `.nwqlib-cache-` filename prefix. An interruption can leave a completed or partial file without a committed SQLite pointer. Reopening acquires the exclusive journal lock, loads the committed cache objects and their file dependencies, and then removes unused files in that namespace before it counts the stored file bytes and writes any recovery revision. This order allows recovery even when unpublished files fill the allowance. Plan inputs, referenced cache files and files outside the reserved namespace are kept. Recovery scans filenames once and uses the references gathered by ordinary cache restoration. It does not parse orphan arrays or circuits.

### Controller checkpoints

A controller checkpoint is one header row, holding its sequence number and field names, and one row per top-level state field. The header, the changed field rows and the RNG position commit in one transaction. `Run.checkpoint(state, changed=names)` encodes and writes only the named fields, and every other field keeps its committed row. The caller names each field that changed since its previous checkpoint, including an added or removed field. The Run rejects an unnamed added or removed field but does not compare the values of the other fields. Omitting `changed` writes every field. `run.checkpoint_state` and `RunData.controller` join the committed rows into one state.

ADAPT names all of its small fields at every checkpoint. It adds the projected pencil when an analysis replaces it, and the decision history when a record is appended or the newest record is updated. A checkpoint made while acquiring one observation therefore writes that observation's pending entry and the small fields, and does not encode the pencil or the decision history again.

### Cache rows and the active frontier

Each immutable native definition, lowered Aer source and prepared native handle has its own stable cache row. The code that changes their dictionaries records the changed keys, so checkpointing does not walk prepared-handle or definition history.

The active frontier has two rows, each saved only when it changes:

| Row | Holds | Saved when |
| --- | --- | --- |
| `frontier` | The current parameter specializations and backend data | Each native preparation replaces its specializations and rewrites only this small row |
| `method` | The live method context | Each access to a method context that is not a dictionary marks the row for saving at the next journal write, unless the code that owns the context marks its own changes |

ADAPT marks its context after each projected analysis and when its controller returns, both for host acquisitions and for native preparations. That context holds keyed caches such as state vectors, Hamiltonian actions and compiled gates. If the process ends before the controller returns, a resumed Run recomputes at most the cache entries created after the last saved context.

File references are indexed per row. A successful replacement or deletion updates only those references and then removes newly unreferenced cache files. A failed publication restores the changed dictionary entries, the provisional file identities and their byte charges. Active-frontier traversal, explicit snapshots and once-only restoration still visit every item they cover, so not every controller operation has constant cost.

### Collection order

Collection has a separate positive ordinal in SQLite, and zero means completed but uncollected. This keeps the collection order independent of the completion and receipt order, including when completed observations are collected in reverse order. Repeated collection keeps its original ordinal. The saved caches of ADAPT use the Run's canonical observations and rebuild their lookup and order once on restore, without acquisition or analysis. A standalone Result keeps its own explicit snapshot of observations and context.

### Journal writes

`Run._write` encodes each changed row once and passes that checked encoding to SQLite. The journal independently checks the transaction's byte total and binary declarations. Upserts and deletions of cache rows and checkpoint fields can share one transaction, and conflicting keys are rejected before mutation. Deletion is limited to cache rows and to checkpoint fields that a controller removed from its state. Scientific observations, receipts and payload associations cannot be pruned. Live metadata byte counters change only after commit. Logical row and binary bytes and allocated database-file bytes stay distinct. Deleting rows does not promise that SQLite files shrink, and checkpoints do not run VACUUM.

### Locking and durability

SQLite uses local filesystem durability and a POSIX advisory lock for one active controller. Close a Run before reopening it, and keep the archive folder available for mapped arrays. A lock file left by a crash is reused after its process exits, and its presence alone does not mean a live process. Sharing a live controller across fork, replacing files externally and controllers on distributed or network filesystems are unsupported. Failed persistence blocks further mutation instead of inventing a completion. Current extended limits load before larger checkpoint data, and accumulated work stays spent.

## Cancellation and recovery

Cancellation prevents new work and targets the original pending preparation and submission locators. Only a returned provider status establishes cancellation. `resume(reanalyze=True)` is a separate, explicit recovery, implemented by the Method, of interrupted classical controller analysis. Ordinary resume and wait do not replay it. Recovery must keep the scientific settings, the saved samples, the RNG and the exposure, and cannot replace unresolved provider work.

RunData snapshots share immutable observations, receipts and arrays. Result context is optional and belongs to the specific verification that uses it. A Method must supply its immutable snapshot and archive hooks. The runtime never computes extra references for persistence. Loading a completed Run returns its saved Result without reanalysis. [Run on a backend](../prepared_execution.md) and [Save, load and reanalyze results](../saved_evidence.md) describe the public workflow.

## Write order and crash windows

Every write that reserves work happens before that work starts, and every write that publishes an outcome includes all of that outcome. In-memory state changes only after the journal commit succeeds, or is restored when it fails.

A static Plan has a fixed list of experiments, each prepared and acquired once by `prepare_static` and `execute_static`. `prepare_static` prepares the first experiment, or all of them for `prepare(plan, settings="all")`, and `execute_static` prepares each remaining one when it reaches it. Before its first preparation, `prepare_static` checks the readout of every experiment against the backend target and the whole Plan against the Run's cumulative limits. On Aer it also checks, before that preparation, the largest known completion-metadata requirement of all the Plan's counts, exact Pauli scalar and amplitude settings. A target refusal there names the experiment and, for a counts readout, the shots per program that it requests, or for any other readout its readout kind.

One static acquisition passes through these commits, and a process can stop between any two of them:

1. **Workflow row.** The experiment's runtime seed and the RNG position commit before anything is prepared, so a restart before the preparation charge prepares the experiment with that same seed.
2. **Preparation charge and RNG position.** If the process stops after this commit and before the receipt, reopening counts a circuit preparation's charge against `max_total_circuits`. The static workflow does not rebuild a failed or interrupted preparation. The Run never repeats a preparation attempt on its own, since that would be new charged work the caller did not request. Such a Run cannot complete, so `execute_static` raises `PreparationNotRebuilt` before it submits any further experiment. After `prepare(plan, settings="all")` this row can precede experiments that were never submitted. Before lowering or the backend makes an exact dense synthesis, a revision of this charge reserves its work against `max_synthesis_work` in its own commit, so a stop during the synthesis leaves that work charged. When the limit refuses the work, the revision records the refused amount instead and, in the same commit, removes the preparation from its static workflow item or controller checkpoint. The refused preparation stays charged, and because the commit removed it from the workflow item or checkpoint, the next call of the workflow or controller prepares the experiment again, unlike after a failed or interrupted preparation.
3. **Receipt and native payload**, with the Aer handle's QPY cache row. If the process stops here, the preparation is complete on disk. A later `resume` submits the saved payload without lowering or transpiling again.
4. **Submission intent, reserved events and output reservation.** The backend is contacted only after this commit. If the process stops before an acknowledgement or outcome commits, reopening makes its reserved events `uncertain` with their shots and circuits still charged. An unacknowledged detached submission becomes `uncertain`. Recovering its original job requires a backend reconciliation path using the original submission identity. An interrupted synchronous Aer submission with no locator becomes `failed` with `results_consumed=True`, because its in-process output is unavailable, and its pending output-byte reservation is released. If the uncancelled Run has no retrievable outcome and the Method cannot return a valid Result, `resume()` and `wait()` raise `RunFailed` at the recovery stage with status `uncertain` and the original attempt identity. The attempt stays uncertain and its exposure stays charged. A new Run is required to execute the Plan again. A synchronous backend has no acknowledgement step, so this window lasts until its outcome commits.
5. **Acknowledged locator** (detached backends). If the process stops after this commit, the submission stays `acknowledged` and its events `reserved`, and the next refresh retrieves the original job by its locator.
6. **Outcome.** For a synchronous acquisition, the completed event, the observation chunk, the submission status and any published arrays commit in one transaction, so reopening never finds one without the others. A quantum chunk was checked against its receipt when it was decoded and against its submission's job when it was published, so reopening indexes it without checking it again.
7. **Collection ordinal.** The Method's consumption of the observation is a separate commit, so the order in which a controller consumed observations survives reopening even when completion order differed.

A refresh of a detached job splits step 6. It first commits the provider's status and item associations on their own, so they survive a later decoding error. The observations then commit in a second transaction, which also sets the submission's `results_consumed` flag once the job is terminal. If the process stops between these two commits, the flag is still unset, so the next refresh reads the same job again and skips items already published.

Remote compilation follows the same intent pattern. Each upload or compile intent commits before the SDK call. The backend adapter passes the new identity to an acknowledgement callback as soon as the SDK returns it, and the callback commits it before any further work that could fail. An intent without an acknowledged identity is resolved only by reconciliation with the original preparation identity.

Cache checkpoints write new files under unused names, commit their references, and then delete files no row references. Interruption can leave unpublished or superseded cache files. Reopening removes them from the reserved cache namespace before it counts the stored file bytes. A failed commit removes provisional files, refunds their charge and restores the previous dictionary entries. This recovery contract covers process interruption and does not cover power loss.

## Lifecycle rules {#lifecycle-rules-and-their-owners}

The lifecycle rules of a Run, with the failure each prevents, the code that keeps it and the test that witnesses it, are indexed in [Design rationale](design_rationale.md#runs-journals-and-archives). Two further rules belong to this page:

| Rule | Failure it prevents | Code |
| --- | --- | --- |
| Collection order is its own ordinal | A reopened controller seeing observations in another order | `Run.collect`, `LocalJournal.commit` |
| Idle native handles can be released | Memory held by fully collected native circuits in a long Run | `Run.release_native`, `PreparedHandle._restore_native` |

## Resource estimate bookkeeping {#resource-estimate-bookkeeping}

### Estimate a construction directly

`nwqlib.resources.estimate` consumes a `SelectedConstruction`, exactly the portable Program and selected definitions accepted by the explicit logical lowering seam. It returns a `WorkloadEstimate`. It never reads native inputs, builds a circuit, transpiles, invokes a backend, or chooses a replacement recipe. The following example estimates the HZH preparation that `plan` selects for `|1>` with a configured `ExpectationMethod`. The default X recipe has one operation. The [block guide](../blocks.md) describes both recipes.

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
selected = plan(
    problem, method=ExpectationMethod(preparation_choice="hzh"), seed=7
)
workload = estimate(selected.construction)
operations = workload.quantity("operations")
assert operations.fact.value.numerator == 3
assert operations.lifecycle == "planned"
restored = WorkloadEstimate.model_validate_json(workload.model_dump_json())
```

Load resource records through their current concrete classes, as in the last line. Their serialized schema version and identifier describe the supplied construction and quantities. The compact JSON keeps every fact and identity.

### Evidence of each count

Original law evidence is kept in the estimate's shared source table. Arithmetic creates a declared derived fact, never a witnessed preparation record or an execution record. User assertions remain assertions. Extrapolating observed source data becomes an empirical prediction even when another summand is asserted. The separate `derivation_sources` references identify evidence actually used for the value. `sources` also keeps ancillary conformance evidence. A literal one-X count therefore stays exact when an overlapping upper-bound law is also checked against it. An estimated zero cost remains an estimate after positive repetition, while an exact zero multiplicity removes even unavailable body work. All quantities emitted by this operation are planned. Core `Stage` names workflow phases for Limits. Resource representation is given by metric/basis/source: a native CX envelope remains planned, with its original source. Resource quantities do not infer a planning/preparation/execution phase. Compiled inventories, actual submitted/observed consumption and physical QEC, factory or routing projections require their later owners.

### Counting rules for selected blocks

A `SelectedDefinition` may contain per-metric `ResourceLaw` declarations. The law must match the selected control/adjoint flags, concrete cost and call bindings, metric basis, precision and synthesis choice. A law can explicitly cover the whole accepted domain of named formal parameters through unbound_parameters. Every other actual argument must still match its fixed bindings. Empty coverage keeps exact-point matching. Covered names must be declared, distinct and disjoint from fixed bindings. Coverage alone supplies no scientific proof. Duplicate applicable laws are rejected. Applicable summaries supply their metrics. Missing metrics use the kept ordered primitives. If that fallback also determines the same metric exactly, it is kept and checked against the declared exact value or bound. Kept literal recipes are checked once per bounded selected context even when all metrics have summary laws. Law-only/native summary definitions require no decomposition traversal. Scalar conformance does not prove the selected quantum action or turn source evidence into a witnessed preparation record.

The finite native CX adapters recognize the exact registered identities of the selected block implementations for direct PREP, signed Pauli SELECT and Pauli readout. They reuse those law owners, including outer-control rotations, sign diagonals and phases, and reject excessive integer growth before a shift or multiplication. Product PREP keeps its q single-qubit component invocations through `preparation_components`. It does not receive the generic q-qubit vector bound. A CX envelope supplies no missing T, rotation, depth or workspace facts.

Bases are `selected_logical`, `cx`, `clifford_t` and `toffoli`. T, Toffoli, CCZ, arbitrary-rotation, logical-depth, T-depth and non-Clifford-depth metrics are canonical. A Toffoli inventory and a T decomposition are alternative bases. There is no automatic conversion or addition between them. Unknown rotation precision or missing synthesis cannot imply a fixed T count. FT laws can describe a conditional estimate without an executable block. The currently executable primitive vocabulary remains X/H/Z, S-adjoint, CX, multi-controlled Z and global phase. Control acts on each primitive, including global phase. Primitive arity and concrete selected target widths are checked before accounting. An effective one-qubit phase with a stored conventional value of 0, ±pi/2, ±pi, ±3pi/2 or ±2pi is classified as Clifford. ±pi/4 is the named T/T-inverse phase. These exact comparisons do not snap nearby arbitrary angles. In particular, controlled PREP of [i,0] is S on its control, with zero non-Clifford depth.

### Composition, repetition and branches

Sequence and ordered primitive recipes are serial. `Parallel` is an explicit join of direct declared unitary/preserve calls on disjoint whole registers. It sums work and takes the maximum depth. Shared-wire calls and nonunitary effects are rejected by the Program check. Logical lowering currently rejects this new node. Resource estimation does not imply a parallel compiler implementation.

Repeat multiplies work without expansion. The fold keeps one result per selected definition/call-binding context and one per graph/lifetime context, within Program's existing work and integer limits. Context fixes the basis, precision and synthesis. Selected identity fixes control, adjoint and recipe. There is no persistent global cache. Expression definitions reuse the bounded Constant/ParameterRef/add/multiply/ceildiv/min/max/comparison AST with local references. JSON does not expand shared expressions. Arbitrary parsing, symbolic expansion, general range summation and automatic simplification are unsupported. This subset needs exact scalar arithmetic rather than a separate SymPy engine. Both Program and selected/context metadata entry paths set aside room for a collection's children before pushing them on the traversal stack. A rejected collection cannot first create an over-budget copy of its references. This is a finite graph traversal limit, not an RSS or CPU guarantee.

Branches produce per-metric maxima with upper-bound scope. No branch probability or expected-round model is inferred. Adaptive body costs need an explicit `resource_envelope` asserting a uniform per-round bound over history. Otherwise they remain unknown. Policy work/workspace remain separate missing costs. MeasurementBatch settings are independent experiments. Explicit `repetitions` multiplies body work. Absent repetitions remain unknown. A terminal batch's `observation_kind` is `counts`, `pauli_expectation`, `probabilities`, or unknown (`None`). An outer batch repeats its descendants and must have kind `None`. Its summed depths are a serial workload upper envelope, not a claim that independent jobs have that causal depth or that this is a per-circuit depth. Axis-dependent costs/repetitions stay unknown because aggregate range laws are outside this scalar-law subset. Invariant costs use the compact axis cardinality.

### Trajectory readout items and bytes

For one declared trajectory, `N_items = sum_k L_k`. A Pauli point contributes its number of distinct requested labels. A probability marginal on `q_k` selected wires contributes `2**q_k` logical outcomes regardless of its eventual sparse stored-entry count. An amplitude point contributes the dimension of its selected output declaration. A reduction contributes the output cardinality computed by its registered shape function from the validated parameters. Two distinct views at one boundary contribute both payloads. Positions are neither repetitions nor shots. An empty request is algebraic work with no measurement. `core/planning.py::point_items` owns this count, and assessment, `readout_shape` and `readout_bytes` use the same declared point population. The stored data set aside for a trajectory is the sum of its point declarations, fixed chunk headers and value payloads, plus one completion-metadata budget, with the complete trajectory declaration charged once in its preparation record (`_prepared_execution.trajectory_reservation`). The Run sums the representation-specific amounts set aside for the points before native preparation (`_prepared_execution._admit_readout`). A dense or adaptively sparse probability array sets aside `8 * 2**q_k` bytes, Pauli values use their JSON prototype envelope, amplitude artifacts their binary payload, and a reduction the JSON envelope of its registered output components (`ReducedValues`). A probability point's fixed header funds its array manifests, scalar summaries, publication-row growth and payload-reference rows at their largest legal encoding. The completion-metadata budget funds the job envelope and any bytes beyond that set-aside bound. With Aer's 36-character ASCII job identifier, probability and Pauli trajectories without a forecast require `58*K + 1113` metadata bytes for K points. The default 65,536-byte limit accepts 1,110 such points when their declarations, headers, binary payloads and execution workspace fit the other limits. An unsupported reduction or a reduction with unresolved output shape fails before native preparation. Stored data set aside and execution working memory have separate checks.

### Register lifetimes and workspace

Register `location` and `role` specify per-location logical system/clean/dirty width. Allocate, Release and reuse determine peaks. Quantum width is not converted into statevector bytes or kept compiled wires. Declared additional quantum workspace is included, and unknown clean/dirty obligations stay unknown. Classical registers remain live after definition until the experiment ends. May-live branch/zero-iteration outputs contribute a conservative storage bound. A guaranteed positive direct symbolic Repeat exports the same availability as IR. An empty Reset has no events or depth layer. A nonempty Reset has one layer. For a legal unresolved width that may be zero, its depth remains the symbolic min(1,width). Measure keeps its existing positive-width result check.

The quantum lifetime fold incrementally stores width sums by role and location. Unresolved widths keep expression multiplicities so Release removes the actual formal contribution without inventing symbolic subtraction. Wire membership and branch/loop lifetime equality belong to the Program check. The fold does not copy the complete live-wire set at every Allocate/Release. One hundred one-bit registers therefore require a compact running subtotal; symbolic peaks still preserve the actual binding context.

Selected `Workspace` entries are simultaneous per-invocation bytes. Parallel calls sum them per location. Serial calls reuse them. `ResourceContext.resident` holds declared input, analysis, I/O, materialization or kept payloads for the whole workload. No entry is allocated by the fold. Missing block workspace or classical-stage workspace stays unknown. `capacities` keeps `Limit` declarations for a later assessment. This operation does not combine device capacities or claim feasibility. An 8 GiB device cannot host a declared 10 GiB peak merely because a second 8 GiB device exists.

### Reachable definitions and the estimator's own work

`defined_selections` lists potentially reachable selected IDs (excluding a known zero Repeat/measurement population). Its construction-work upper subtotal covers those unique declarations. It is distinct from the actually built native definitions in the lowering's preparation record, dynamic construction replay and hidden base-gate cache. `work_units` and `evaluated_contexts` expose bounded fold bookkeeping, not CPU time or observed memory. The fold's work limit grows in proportion to the Program's own checking work, as described in the engineering constants.

### Pauli-mask storage example

The metadata-only q=60, M=10^6 witness declares the current raw Pauli mask storage as two uint64 words and a complex128 coefficient per term: 32,000,000 bytes. It accesses no payload. This excludes coalescing workspace, indices, copies and Python object overhead, and establishes no actual ingestion cost or throughput.

### Value domains

Concrete resource quantities are real and nonnegative. Exact cardinalities are integral. Exact `expected_operations` and estimated or upper-bound resource values may be fractional. These restrictions belong to resources, while generic `Fact` values remain usable for signed and complex quantities.

## Execution accounting {#execution-accounting}

### Count-likelihood index

The count-likelihood check keeps a Run-owned index derived from actual preparation records and measurement identities. Work already committed and set aside keeps its fixed streams unavailable for another independent sample, even if its result is uncertain or never collected. Completed observations update that index once, and reopening rebuilds it once when the check is next needed. The index is not another serialized observation history and supplies no new independence proof.

### Limit amendments

Every actual increase appends an immutable `LimitAmendment` to `run.limit_amendments`. It records the complete old/new limits, sequence, timezone-aware timestamp and counters immediately before publication: circuit preparations, circuit measurement attempts, explicitly prescribed raw shots, stored data bytes, pending output bytes and the dense synthesis work set aside against `max_synthesis_work`. The record refuses counters that exceed the old limits, counting stored and pending bytes together against `max_data_bytes`. Failed and unused preparations count, and so do set-aside and uncertain measurements. Host work and unknown provider-managed sampling do not become circuit/shot counts. Equal values are a no-op. The new cap and its single history row commit together, and metadata must fit the new data cap. A failed limit check or write publishes neither change. Recording a limit amendment consumes storage, and later writes can require a larger cap. `run.trace` and each Result's data preserve their captured cap/history prefix, so increasing the Run later does not alter an earlier Result.

`run.extend_limits(...)` sets cumulative caps without refunding previous use. Each actual increase stores the old/new limits and pre-publication counters in one immutable journal row, atomically with the new current cap. Save/reopen preserves the ordered history for both initially in-memory and durable Runs. A completed Result keeps the cap/history prefix it captured, even when the enclosing Run was extended later. This needs only a prefix marker and captured byte counter beside the saved Result, without another full trace or array copy. A reopened Result keeps its own identity and observations and is attached to the Run's trace as saved, apart from that prefix and byte counter. Events and submissions that the Run revised after the Result, for example through a later `run.cancel()` and its refresh, appear there with their later status, and so does that cancellation request.

### Completion-metadata budget

The output limit check includes the encoded readout declaration, declared numeric values and arrays, the host application records a kernel declares, plus `ExecutionLimits.max_completion_metadata_bytes`, which defaults to 65,536 bytes per completed attempt. This extra budget covers variable provider annotations and undeclared host application records, excluding the already-funded declaration and declared records. A single-endpoint probability readout also spends it on its remaining chunk fields, the manifests and summaries of its largest legal encoding and its publication rows, and is refused before lowering when those known costs exceed it. A trajectory funds each probability point's manifests, summaries and publication rows in the point's set-aside header, so the budget funds each point's job envelope and the completion rows. Increase it explicitly for providers that return larger metadata. A response beyond its selected limit can be rejected after measurement, with its work still charged. Before native preparation, a static Plan on Aer checks the largest known completion-metadata requirement of all its counts, exact Pauli scalar and amplitude settings. Counts reserve the maximum decimal width of the returned-shot count in both the chunk and its completion event. Exact Pauli scalar readouts use the same declaration/value/metadata split. Amplitude endpoints also fund their scalar summaries, physical scale and the larger of the array-publication and declared unavailability branches. These laws are established for Aer. Another backend checks such readouts at publication. A publication failure that has already made a synchronous acquisition terminal cannot be cleared by extending a limit on that Run. Setting bytes aside allocates no output buffer.

### Recovery of interrupted attempts

If an interrupted attempt has no retrievable outcome, `resume()` and `wait()` raise `RunFailed` with `stage="recovery"` and `status="uncertain"`. This applies to interrupted host kernels and local Aer calls without a recoverable backend job. The error names the original attempt and saved Run. It preserves recorded failure text, observations and charged work. For a reopened synchronous Aer attempt without a job locator (next paragraph), the recorded failure text is the one written on reopening, which states that the in-process result is unavailable. It does not assert that the underlying execution failed. A Method may still return a valid partial Result before this check. Remote work with a refreshable locator, or a supported reconciliation path to that locator, keeps its pending behavior. The order of the journal writes behind these recovery rules, and what a crash between any two of them leaves behind, is described in [Write order and crash windows](#write-order-and-crash-windows).

When an uncancelled Run reopens an interrupted synchronous Aer attempt with no saved job locator, its in-process result is unavailable. If the Method cannot return a valid Result from the saved data, both `resume()` and `wait()` raise `RunFailed` with `stage="recovery"`, `status="uncertain"` and the original attempt identity in `attempt`. The exception includes the attempt record in `trace`, the saved Run directory and the charged exposure. Reopening marks the submission `failed` with `results_consumed=True` to record that its output cannot be retrieved, and releases its pending output-byte reservation. The attempt stays `uncertain`, and its job, circuit and shot exposure stays charged. Inspect the saved evidence or cancel this Run. To execute the Plan again, start a new Run. Nothing is resubmitted in the original Run. Detached submissions keep that reservation while their original job can still supply output. NWQ-Sim can recover a lost launch acknowledgement from the original submission UUID even when no job locator was saved.

## Report reader limits {#report-reader-limits}

### Method discovery in the CLI and in Python

Missing or ambiguous registrations reject without fallback. Builtin discovery reads the same configuration owners as Python exports. In Python, discovery keeps the actual immutable registration records. A compact display does not replace them with strings.

### Saved report reader

`report PATH` reads `result.json` with the archive owner's JSON reader, which refuses nesting deeper than the JSON parser's recursion limit, duplicate keys and nonfinite numbers. The report also refuses a file that parses but is nested too deeply for the indented JSON encoder that writes the report. The joins of observations to their receipts and attempts were checked when the Result was saved and are not repeated. No array, native circuit or controller cache is loaded.

## Saved run mechanics {#saved-run-mechanics}

### QHD constrained-run directories

With `solve_augmented_lagrangian(..., directory=path)`, every inner Run of the augmented-Lagrangian layer is created with `prepare(..., directory=...)` under `iterations/<k>/run/`, or `iterations/<k>/levels/<z>/run/` with refinement, and keeps its own journal ([Continue an interrupted run](../run_archives.md)). The directory also holds `problem.pickle` and the outer record `controller.json`, which stores the arguments of the run, the configuration of the backend of every inner Run and the rounds and levels that have completed. A noisy Aer backend's model is saved once as `noise-model.json`, in the form that `run.save` uses ([Aer](../aer.md#noise)). After each completed round or level, and once more when the run ends, the layer writes the new outer record to a temporary file and renames it over the previous one, so an interruption leaves one of the two complete records.

The outer record repeats no observation of the Runs, and its size grows linearly with the completed rounds and levels. It is rewritten after each of them, so the bytes written over a run grow with the square of their number, while the directory keeps one copy. The per-level `tables.json` files contain generated table-stage data. Their total stored size is the sum of their UTF-8 file lengths and grows with the retained levels and their table sizes. They are outer files and are excluded from the inner Runs' `max_data_bytes`. Their save and load memory is checked against the level Method's `max_bytes`.

In a refinement run directory of `refine_box(..., directory=path)`, each level also writes its table-stage data once as `levels/<z>/tables.json`, which holds the search model's unscaled support tables, their evaluation count and, when it is rational, the refinement's constant C. A resumed level in progress reads these tables instead of evaluating them again, and the refinement takes C from the first level's file. Saving and loading a level file are checked against the level's `QHD.max_bytes`, with the earlier levels' Results counted as live data.

`resume_augmented_lagrangian(path, backend=...)` reads the constraint preprocessing and the check counts that the first call saved in `controller.json` and does not evaluate the setup's support tables and summand scans again. Resume then binds the model saved in the directory, with its errors and basis gates, to its own copy of that backend for the Runs it creates, as `load_run` binds a Run's saved model for the reopened Run, so every round is lowered to the Aer target of the original Runs. Resume reads the files that the layer wrote as they are, including the saved preprocessing and check counts, which it does not recompute, and does not detect an edit.

### Ignored LCHS `trotter_steps`

With `hamiltonian_evolution_backend` set to `dense_exact` or `qsp_block_encoding`, LCHS does not use `trotter_steps`. Method validation sets that field to one before the content hash is computed and keeps a nondefault supplied value in the private attribute `LCHS._ignored_trotter_steps` (`src/nwqlib/algorithms/lchs/method.py`), which planning uses for its warning. The private record of that supplied value is not reconstructed by `model_copy()` or by loading the canonical Method.

### ADAPT reanalysis of saved chunks

During reanalysis, ADAPT validates each saved chunk against its query. The `Plan` keeps only its most recent resolution, so a forward pass over Q distinct queries normally resolves Q points again, even when a query supplies several adjacent observation chunks. Saving a quantum Result also derives the query constructions needed to validate its saved data. These passes use the obtained data and do not run the quantum circuits again. The matrix stage validates each original query before assembly and passes one current matrix to projection. That temporary matrix is released after use and is not saved as another history, and restoring an interrupted projection rebuilds it from the original observations. Mutable Qiskit `Gate` definitions stay in the private Run cache for continuation and are excluded from public Result data.

## Saved formats {#saved-formats}

### Run archive format {#run-archive-format}

Current loaders support Run format 18 (`nwqlib.run/18`) and Result format 11 (`nwqlib.result/11`). In both, a host kernel declaration records `application_bytes`, the application record bytes that each invocation sets aside. Both formats preserve cap-amendment history. Run format 18 reserves the cache filename prefix for the Run. Each journal row stores the byte size of its JSON text in a column written with the text, and the size caps and stored-data totals read that column without reading the text. With each prepared Aer handle, the format stores the prefix phases of its saved trajectory states and its thread cap. The journal keeps the initial limits and every later amendment.

`run.json` contains `{format, selection, limits}`: the format, the name of the selection file and the execution limits in force when the folder was written. The selection file, normally `selection.json`, holds the Method name, the `Plan` identity and the record returned by the Method's archive hook. The built-in hooks put the `Plan` record in that record and hold or name the selected data there. Keeping the `Plan` and its data out of `run.json` keeps that manifest small, and `load_run` refuses a manifest larger than one MiB before parsing it. A Run writes the manifest as `run.json.partial` and then renames it to `run.json`, which replaces the file atomically within one filesystem, before it commits its journal header. A creation that a `max_data_bytes` refusal or an interruption stops before or during that write therefore leaves no `run.json`, and a folder with `run.json` holds a complete one. For a folder with `run.sqlite` and no `run.json`, `load_run` adds a note that the creation of the Run stopped before any preparation or measurement. The partial file stays and counts among the folder's stored file bytes. `run.sqlite` contains the evolving journal, including current limits, preparation and observation records, cache references and the completed Result when present. An observation row leaves out its readout declaration, which its preparation record stores, and reopening restores the declaration from that record, so a readout with many labels is stored once per preparation. The adjacent `.lock` file supports exclusive controller ownership. These files and their referenced payloads form one Run, and `run.json` alone is not sufficient for continuation.

### Arrays and trajectory observations

Probability readouts store their numerical values in binary arrays. The observation chunk contains the array manifests, outcome layout and scalar population summaries. Dense probabilities use outcome positions as indices. Sparse probabilities use sorted unique integer indices. The first selected probability qubit is least significant in its outcome index. Saved reports describe these arrays without loading their values. Counts keep one JSON count record per stored outcome. A trajectory readout saves one observation per point. Each names its point ID and resolved boundary, so a saved value maps to its body, point and preparation record after reload, and the point observations share their measurement's completion, preparation, shot and native-work population. A point observation stores only its own point's readout declaration and the identity of the trajectory declaration, which the preparation record stores once.

A published numerical array has one saved payload in SQLite, committed with its manifest and completed observation. An in-memory Run saved later writes those same payload bytes and associations in one snapshot transaction, and no second NPY file is written for the array. Run loading registers each published array without reading it. Its first use reads the stored blocks once per payload digest into one read-only buffer, checking block order and byte count without recomputing the digest or scanning the values, and each manifest gets a view of its own dtype and shape. An array first used after the Run is closed is read through a read-only journal connection, and an array already read stays readable. Run loading also registers each prepared handle's saved circuit file without decoding it, and the circuit is decoded at the handle's first submission or inspection.

A reopened Run reads an array's already-stored journal bytes under the Run's data limit the first time the array is used, for example by `run.hydrate(manifest)` or a probability chunk's `histogram()`, and keeps one copy of each payload for the Run's lifetime.

The shape, little-endian encoding (`<c16` for complex128, `<f8` for float64 or `<u8` for uint64 outcome indices, as its manifest declares), byte count and contiguity of every published array are checked on every load against its manifest, and a mismatch rejects.

Original Plan inputs and unpublished method-cache arrays use their existing single-file storage. An explicit Run copy or standalone Result save is a separate requested output and may contain its own payload. Copying a saved Run whose cache rows are all committed copies the files that those rows name under the same names, without decoding or encoding the cached objects again.

### Result archive format

The new directory contains `result.json` plus its supporting payload files. The current JSON envelope is `{format, selection, data, result}`, with format `nwqlib.result/11`. It is distinct from the human-reading dictionary returned by `result.report()`. That report is not a loadable archive. The envelope's `selection` entry holds the Method name, the `Plan` identity and the record returned by the Method's archive hook. The built-in hooks put the `Plan` record in that record and hold or name the selected data there. LCHS, for example, saves its node tables, host actions and SELECT angle tables in their own JSON file with NPY arrays. A Run folder uses its separate `run.json` and journal format, described in [Run archive format](#run-archive-format).

### Circuit entries and Qiskit UCGate compatibility {#qiskit-ucgate-compatibility}

Ordinary native entries use QPY; entries containing UCGate use the versioned NWQLIB-QPY-UC1 envelope around a QPY storage representation.

In Qiskit 2.5.2, a native `UCGate` can be written by QPY but cannot be loaded: QPY passes the gate's matrix table as separate positional constructor arguments, whereas `UCGate` expects one list. This is a constructor reconstruction defect; it is not evidence of a reversed qubit order.

NWQLib stores an encountered UCGate as a compact QPY instruction carrying its matrix table and explicit reconstruction fields. A file prefix `NWQLIB-QPY-UC1` identifies this versioned container. The reader restores the nominal number of qubits, existing simplified table, active controls, `up_to_diagonal`, label and any already materialized definition. Circuit phase and wire associations remain attached to their original circuit. Under this SDK's convention, the target is wire 0 and simplified control indices are one-based within the gate, from 1 through `num_qubits - 1`. Loading does not rerun multiplexor simplification or expand the table to all nominal controls. Saving does not synthesize a UCGate or construct its full matrix.

Use NWQLib's archive loaders for these `.qpy` entries: the prefix makes them containers, rather than standalone input for `qiskit.qpy.load`. Circuit files without this adaptation remain ordinary QPY. An old raw QPY entry affected by the upstream bug still reports its original SDK failure, with the entry path; the loader cannot infer missing reconstruction fields from it.

The small `test_native_ucgate_qpy_limitation_canary` test exercises raw QPY on one two-qubit UCGate. It runs no simulation or synthesis. If native loading starts succeeding, or its failure changes, the test requests a compatibility review. A future Qiskit fix does not by itself remove support for existing `UC1` files: keep that decoder and verify simplified controls, diagonal flags and executable round-trips before changing the writer. The UC1 storage is not switched off automatically based on an SDK version number.

### Method record formats

QPE archives use `qpe/<estimator>/7`, which also store the selected polar base of a unitary input. QCELS keeps analysis source `qpe.qcels.analysis` version 3, and the other estimators use `qpe.analysis` version 3. Earlier development formats require their original source/environment.

The QHD constrained controller uses `qhd.constrained_run/5`, and a standalone refinement controller uses `qhd.refinement_run/6`. A saved constrained-result archive uses `qhd.constrained/4`, and a saved standalone refinement archive uses `qhd.refinement/4`, both with inner QHD Results in `qhd/7`. These format names belong to the layer and Method records, separately from the common Run and Result formats.

## Backend adapter contract {#backend-adapter-contract}

### Adapter rules

Every connection implements one adapter contract, stated in the module docstring of `nwqlib.backends.connection`. The common execution owner in `nwqlib._prepared_execution` saves the intent, locator and observations around each adapter call and relies on these rules.

| Rule | Failure it prevents | Owner |
| --- | --- | --- |
| Readout, shots and batch shape are checked before preparation, upload or submission | Provider work spent on a request that cannot produce the selected observation | each adapter's `target_for` and `admit_batch` |
| Intent and the submissions set aside are saved before one create request, and NWQLib never retries a create request. The IBM SDK's own safe retries are described in [IBM Runtime](../ibm.md) | A duplicate job after a lost acknowledgement | `submit_detached` and each adapter's `launch` |
| A lost acknowledgement is matched only by the label written at launch. No match or several matches leave the intent uncertain and charged | Binding another job, resubmitting, or refunding work whose outcome is unknown | `refresh_submissions` and each `reconcile` |
| Refresh reads the original job, publishes each completed item once and keeps earlier items when a later item fails | Duplicate or lost observations | `refresh_submissions` and each `refresh` |
| Cancellation is one request to the original job, confirmed only by later status, and the submissions stay charged | Treating a request as a confirmed cancellation or a refund | `Run.cancel` and each `cancel` |
| Measurement maps and bit order are translated in one decoder per adapter | Reordered bits or mixed registers | `_decode` for IBM, IonQ and Nexus, `_submit_aer_execution` for Aer, and `NWQSimBackend._result` with the runner for NWQ-Sim and Slurm |
| Gate-angle units are fixed at preparation | IonQ native gates read in the wrong unit | `IonQBackend.prepare` keeps native turns and lowers QIS gates in radians |
| Bytes that NWQLib reads or parses are checked against the byte limit before the read or parse | Unbounded provider responses or saved payloads | `run.check_data` calls in each adapter |
| The qualification notice states offline versus live evidence once per Run | Reading an offline transport check as live device or site qualification | each adapter's `qualification_notice` |

### Nexus preparation stages

The individual upload/compile methods of `NexusBackend` are not a substitute for the durable execution check. The common owner must record these steps independently. Upload and compilation are preparation work, not quantum executions or measured shots.

| Stage | Persist before the outbound operation | Acknowledgement / continuation |
| --- | --- | --- |
| Local selection | Preparation ID, snapshot, original observation and register maps, and `NexusPreparationData` with pytket `Circuit.to_dict()` JSON | `prepare_local` replaces each dense `UnitaryGate` by its exact synthesis from the Run's synthesis cache (each distinct matrix synthesized once per Run object, its work set aside against the Run's cumulative `max_synthesis_work`), uses public Qiskit `transpile` to `u`/`cx` at optimization level 0, then invokes the public Qiskit-to-pytket converter once, without simulation |
| Upload | Upload intent, exact local data, project, unique `nwqlib:upload:<id>` name | `upload` returns public `CircuitRef` JSON; persist it before compilation |
| Compile | Compile intent, acknowledged input reference, target/configuration, optimization level, `nwqlib:compile:<id>` name | `start_compile` returns a `JobLocator`; persist it before refreshing |
| Compile pending | Original locator and input reference | `refresh_compile` returns `BackendRefresh`; no repeated upload or compilation |
| Compile complete | The completed result's actual input/output association | `refresh_compile` returns `NativePreparation` holding the compiled `CircuitRef` in `provider_options_json.compiled_program` |
| Execute | The accepted submission and ordered program/shot inventory | `launch` returns an execution locator immediately; the common owner persists it |

`reconcile_preparation` queries the original name and project. Compile-stage reconciliation also requires `input_ref_json`. `reconcile` checks the execution association. Both inspect at most two matches through the public paginated iterator. No match leaves the intent unresolved; multiple matches raise. Neither case permits an automatic replacement upload, compile, or execution.

`upload` and `start_compile` require an `acknowledge` callback from the common owner. Upload calls it with the circuit ID and compile calls it with the `JobLocator` immediately after the SDK returns. This precedes potentially failing serialization/limit-check work. After an upload acknowledgement followed by serialization failure, `restore_upload` reads that same public circuit ID; it never uploads again. `restore_preparation` parses the saved provider model. `refresh_compile` takes a `load_data` callback and invokes it only after successful terminal compilation and output association, so queued refreshes do not load the saved circuit JSON.

### NWQ-Sim runner protocol

The runner protocol is `nwqlib.nwqsim/3`. Amplitude output uses a separate interleaved binary64 file, and a probability marginal a separate dense binary64 file of `2**k` values, one per outcome of the k observed qubits. Bit zero of a marginal index is the first observed wire. For observed wires `(0,...,k-1)`, a native basis index contributes to `x & (2**k-1)`. Unobserved high bits are summed into that bin. The native index itself is the bin only for an identity-ordered full-register observation. The result names each file with its dtype, element count and bytes, the adapter checks them before reading it, and the runner removes the files of an attempt that ends without its terminal result. The U/CX request writer checks each compact JSON gate against the byte limit before encoding or writing it. A U gate costs at most `134+digits(q)` bytes and a CX gate at most `38+digits(q0)+digits(q1)`, plus its list separator. The bytes of fixed request fields and the closing suffix are set aside separately. The writer validates finite binary64 parameters and verifies each actual encoded length against the bound.

With MPI, rank zero alone forms a histogram and publishes it: it projects each outcome onto its classical bits in the sampler's outcome array, sorts that array and counts runs, so each distinct outcome is formatted once.

After a local launch intent loses its acknowledgement, the adapter checks file metadata and an existing child handle through `can_reconcile`, without a replacement launch or refund.

### NWQ-Sim trajectories

For a CPU/SV trajectory, the adapter marks each point boundary of the bound body with a labeled barrier, lowers the body through its last point to U/CX once, and reads each boundary's native gate position from its barrier; nothing after the last point runs. A view's tail is lowered once, and its inverse is the reversed, adjointed tail of that lowered sequence. The runner evolves the gates between consecutive boundaries with one NWQ-Sim call each, without resetting or renormalizing the state, and applies a view's inverse only when a later point follows. Pauli values are keyed by point and label, each marginal is a dense binary64 file and each saved state a complex128 file.

The trajectory's metadata check includes each point's variable envelope and the net growth of the detached outcome rows. The provider-status row is committed first and credited at outcome publication. A fresh outcome adds 161–168 JSON bytes, or 92–99 when a forecast has already revised the event, and the check sets the maximum aside. Local job identifiers are 36-character UUIDs, so each non-amplitude point's empty-value envelope costs 58 bytes. Acknowledgement and status writes use the Run's cumulative data capacity while the output bytes remain set aside. Restored readouts are checked against their actual stored rows when published. A single-endpoint probability readout, locally or through Slurm, adds the same detached outcome bytes, 168 or 99 with a forecast, to its metadata check, which refuses the readout before the runner starts when `max_completion_metadata_bytes` cannot hold it.
