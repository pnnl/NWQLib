# Run on a backend

<a id="prepared-execution"></a>Use `prepare` and `submit` instead of `solve` when you want to cap circuits, shots and stored data before anything runs, look at the circuits before they are measured, or run on a provider whose jobs can stay pending. `solve` creates a Run of its own and closes it before returning.

A `Plan` holds the construction that `plan(...)` chose and its costs, computed before any circuit exists, together with the Problem, the configured Method, the output, the scientific settings and the randomness. `prepare` creates one `Prepared` object over a live `Run`, and `submit` advances that Run. A pending provider job stays reachable through the returned Run.

## Run locally with limits

```python
import nwqlib
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.execution import ExecutionLimits
from nwqlib.operators.inputs import ingest_pauli
from nwqlib.problems import Expectation

problem = Expectation(state=[1., 1.],
    observable=ingest_pauli((("I", 1.), ("Z", 1.)), num_qubits=1))
plan = nwqlib.plan(problem, method=ExpectationMethod(), shots=64, seed=7)
limits = ExecutionLimits(max_total_circuits=1, max_total_shots=64)
prepared = nwqlib.prepare(plan, limits=limits)
with prepared.run as run:
    nwqlib.submit(prepared)
    result = run.wait()
    print(result.value)
    print(run.exposure["completed"])
```

```text
1.0625
{'jobs': 1, 'circuits': 1, 'shots': 64, 'provider_managed_sampling': 0}
```

The exact expectation of `I + Z` in the normalized state `(1, 1)/sqrt(2)` is 1, and 64 shots estimate it as 1.0625. The Run checked the whole workload, one circuit and 64 shots, against `limits` before it prepared anything.

`run.exposure` is the work submitted so far. It counts jobs, circuits, shots and provider-managed sampling in three categories: `completed`, `reserved` (set aside for submissions that have not finished) and `uncertain` (submissions whose outcome is unknown after an interruption). Its `failed` category applies to confirmed host failures and adds no quantum work.

Work is counted against the Run's limits ("charged" in messages and records) when it is set aside, before it runs. It stays counted when it fails, is cancelled, is never used or has an uncertain outcome. Host work and unknown provider-managed sampling do not become circuit or shot counts. A provider estimate does not become a known count of raw shots.

`run.data` is an immutable snapshot of the observations, preparation records, execution trace and kept array handles. Taking a snapshot shares the immutable arrays. Completed observations are saved once, and collecting them again does not duplicate them. Failed and unused work, host invocations and work set aside remain in the trace.

## Run on a provider

For a provider, pass a backend object and a folder for the Run's saved state to `prepare`. This IBM template needs the `ibm` extra and an account authorized for the named device and instance. It uses the `plan` above:

```python
from nwqlib.backends import IBMRuntimeBackend

backend = IBMRuntimeBackend(
    device="YOUR-IBM-DEVICE", instance="YOUR-IBM-INSTANCE",
    max_input_bytes=65_536,
)
prepared = nwqlib.prepare(plan, backend=backend, directory="my-run")
with prepared.run as run:
    nwqlib.submit(prepared)
    print(run.directory)
    print(run.exposure)
```

Constructing the backend reads no credentials and does not contact the service. Remote execution requires a saved folder, and when `directory` is omitted the Run chooses a local folder, which `run.directory` reports. The `Plan` and the prepared backend circuits are saved before submission. Preparation may itself wait for remote compilation.

`run.resume()` retrieves outstanding work from the provider and advances the Method. `run.wait()` repeats that operation until a Result is available or its explicit timeout expires. A retrieval error keeps the original provider job locator and propagates to the caller. It does not authorize another submission. To collect the jobs later, close the Run and reopen its folder with `load_run`, as [Continue an interrupted run](run_archives.md#save-and-reopen-pending-work) shows.

The [IBM](ibm.md), [IonQ](ionq.md), [Nexus](nexus.md), [Slurm](slurm.md) and [NWQ-Sim](nwqsim.md) pages give each backend's constructor, supported readouts and qualification scope.

## Limits on circuits, shots and data {#limits-on-circuits-shots-and-data}

`ExecutionLimits` caps the work of one Run. The complete fixed workload is checked against the limits before preparation, and each backend call sets its circuits and shots aside before it starts.

| Field | Default | What it caps |
| --- | --- | --- |
| `max_total_circuits` | 512 | Quantum preparations, and separately circuit measurement attempts, over the Run |
| `max_total_shots` | 1,000,000 | Raw shots set aside, including uncertain attempts |
| `max_data_bytes` | 10 GB | Stored Run data and numerical arrays: saved arrays, serialized backend circuits and run-log output. Not process memory |
| `max_completion_metadata_bytes` | 65,536 bytes | Extra metadata per completed attempt, for variable provider annotations and undeclared host application records |
| `max_simulation_qubits` | 20 | Circuit width accepted before local simulation on Aer or local NWQ-Sim |
| `simulator_memory_mb` | 1024 | Simulator memory allowance in MiB, which Aer receives as its cap on quantum-state storage |
| `max_direct_amplitudes` | 65,536 | Amplitude count of each declared direct state preparation that the Run synthesizes, checked for the whole `Plan` before the first preparation |
| `max_synthesis_work` | 1,000,000,000 | Total work of the exact dense-unitary syntheses that the Run makes while it prepares circuits |

`max_synthesis_work` counts the syntheses made when a backend translates a circuit to a gate basis or when circuit construction controls a transformed block. It is a total over all preparations, and each amount is set aside before its synthesis starts ([exact synthesis limits](development/dense_synthesis.md#admission-of-the-exact-synthesis)). Basis translation synthesizes each distinct matrix once per Run object, and a reopened Run synthesizes it and sets its work aside again.

Increase `max_completion_metadata_bytes` for providers that return larger metadata. A response beyond it can be rejected after measurement, with its work still counted.

These limits control represented work. They are not provider billing, process memory (RSS) or proven CPU-time bounds.

### Raise a limit during a run

`run.extend_limits(max_total_shots=128)` sets a new cumulative cap of 128. It does not add 128 shots, reset usage or change scientific settings. Each actual increase appends one `LimitAmendment` record to `run.limit_amendments`, with the old and new limits, a timestamp and the counters immediately before the change. Equal values change nothing. The new cap and its record are saved together, and a failed limit check or write saves neither. Recording an increase uses stored data, so later writes can need a larger `max_data_bytes`. When an error message suggests a `max_data_bytes` value, that value covers the transition or gate the message names. A Result keeps the limits and history it captured, so increasing the Run's limits later does not alter an earlier Result. [Execution accounting](development/execution.md#execution-accounting) lists the recorded counters.

## Cancel, close and handle failures

`run.cancel()` prevents new work, requests cancellation of the original remote jobs and keeps all counted work. A provider status establishes cancellation, and a request alone does not. A backend without a cancellation operation, such as local [NWQ-Sim](nwqsim.md), receives no request. The Run then adds a notice to `run.warnings`, and a job already launched can still finish and be read by a later `run.resume()` of the same Run. Cancellation cannot interrupt a synchronous backend call that is already running.

Closing a Run releases its private caches and its lock on the folder. It keeps the Result, the scientific data, the preparation records and all counted work. Inspect `prepared.circuits` before closing. Those returned circuit copies cannot change the prepared input. `solve` closes the Run it creates before returning its Result or raising an error, and a failed `prepare` also releases its Run. Their saved files and original jobs remain available for recovery, and closing does not cancel a job. A successful `prepare` and `submit` leave you a live Run to close yourself, for example with `with prepared.run as run:`.

`run.wait()` raises `nwqlib.execution.RunFailed` when an original compilation or execution is confirmed failed or cancelled and the Method has not returned a valid Result. The exception provides `stage`, `status`, `locator`, `failure`, `directory`, and immutable `trace` and `exposure` snapshots. Pending work still raises `TimeoutError` when the requested timeout expires. A retrieval error keeps its original exception and locator and is not reclassified as a failed job. Partial results are up to the Method, and no failure causes an automatic replacement submission.

A synchronous host exception that finishes and is saved has event status `failed`. Its original exception propagates from `submit`, and a later `wait`, or reopening the Run, exposes `RunFailed` with the recorded failure. The invocation and its work stay counted. Unused output capacity can be released without refunding that work. Process interruption, or failure to save completion, leaves the attempt `uncertain`. It is not treated as a completed host failure or retried implicitly.

If an interrupted attempt has no retrievable outcome, `resume()` and `wait()` raise `RunFailed` with `stage="recovery"` and `status="uncertain"`, and nothing is resubmitted. [Interrupted attempts](run_archives.md#interrupted-attempts) describes what to do next.

Ordinary `resume` does not retry an interrupted classical analysis. A Method with a recovery hook accepts `run.resume(reanalyze=True)` to retry that saved analysis explicitly. Completed Results use `result.analyze(...)` instead.

### Files in a Run folder

The `.nwqlib-cache-` filename prefix is reserved for disposable cache files. After an interruption, reopening deletes files with that prefix that the run log does not reference, while it holds the Run's lock, before it counts the stored data. Current inputs and referenced cache files are kept. Keep files you create outside this prefix. Recovery covers an interrupted process (an exception, a keyboard interrupt or a killed process). The run log (`run.sqlite`) commits are synchronized, but external input and cache files are written without `fsync`, so the folder does not guarantee recovery after power loss.

[Execution accounting](development/execution.md#execution-accounting) describes how the Run counts work and stored bytes, including the completion-metadata budget and the recovery of interrupted attempts.

## Prepare every setting before measuring

By default `prepare` prepares the setting that the Method submits first, which for a static `Plan` is its first experiment, and `submit` prepares each later setting when it reaches it. `prepare(plan, settings="all")` prepares every static setting in `Plan` order and submits nothing, so `prepared.circuits`, `prepared.inspect_resources(index=...)` and [logical compilation](fault-tolerant-resources.md#compile-every-setting-of-a-plan) reach every circuit before any measurement. `prepared.setting_names[i]` is the name of the `Plan` experiment (an entry of `plan.experiments`) that circuit i implements. ADAPT, Lanczos with `SensitivitySampling` and RWPE refuse `settings="all"`, because their later settings depend on earlier outcomes. ADAPT's later queries act on generators selected from earlier outcomes, the main stage of `SensitivitySampling` allocates its shots from the pilot, and each RWPE feedback angle depends on the outcomes before it. For them, `prepare(plan)` prepares the next setting.

Each setting is prepared and counted once, and a Run with a folder saves it. `submit`, and `resume` of a saved Run reopened after `prepare`, use the saved preparations without building them again. A preparation that fails or is interrupted after it was counted, other than a refused synthesis, stays counted without a preparation record. The Run never repeats it on its own, so `resume` raises before it submits any further setting, and a new Run prepares the `Plan` again. Both modes draw each setting's runtime seed in `Plan` order, so choosing `"all"` does not change the seeds. The Run checks the whole static workload against its limits before the first preparation in both modes, so `"all"` has no limit of its own.

`"all"` needs local Aer, local NWQ-Sim or classical host execution, and remote backends are [open work](ROADMAP.md#preparing-every-setting-on-remote-backends). Another backend, such as IBM Runtime, IonQ, Nexus or Slurm, is refused before a Run is created, because its preparation can be a remote compilation that the provider charges for. The construction is the same on every backend, so for counts and exact readouts, preparing on Aer shows the circuit of every setting within `ExecutionLimits.max_simulation_qubits`. RWPE, Lanczos with `SensitivitySampling` and ADAPT set later queries from earlier outcomes (feedback angles, main-stage shots and chosen generators), so they refuse `"all"` before a Run is created, and the default prepares the query that the Method submits next.

## Inspect prepared circuits

`prepared.inspect_resources(index=0)` counts the top-level operations of one prepared backend circuit by name, without copying it. The index follows the order of `prepared.circuits` and `prepared.setting_names`. Its finite `max_operations` and `max_bytes` controls apply to that inspection. Supplying `transpile_options` requests one extra compilation of that circuit. The resulting count describes the compiled copy, records the effective options and does not replace the circuit that runs. Inspection requires an open Run.

Compilation follows the configured backend and the preparation record. Alternative resource contexts describe estimates and cannot change the circuit being executed. Counts of prepared circuits refer to the effective compiler configuration and layout of the backend.

A bound Aer noise model can select a narrower gate basis. Counts preparation translates canonical intermediate gates to that basis with Qiskit's BasisTranslator, so noise applies to the gates that are executed. The preparation record identifies this translation and the effective gate counts. Exact readouts still require a noiseless backend, and raw noisy counts do not establish a bound on physical bias.

### Release prepared circuits in a long run

For a long open Run with a folder, `run.release_native()` releases fully collected, inactive prepared backend circuits that already have saved QPY copies. It returns the identifiers it `released` and `kept` pairs of identifier and reason. Only Aer circuits are released. Other backends, unsaved objects, unsubmitted work and items that the Method or a pending job still needs are kept. Lightweight handles and inspection order remain. Inspecting a released item loads its saved QPY once, without rebuilding, transpiling or submitting it, and an explicit repeated submission can use that same restored input. Shared gate definitions needed for future preparations remain cached. These counts describe released objects, not measured bytes or RSS, and references you hold can keep objects alive. Keep the Run folder available for later inspection.

## Forecasts and allocations

When supplied, `run.forecast` and `run.allocation` keep the `PlanEstimate` and `Allocation` chosen for the Run. The same immutable values appear in `run.data` and the Result's data and survive save and reopen. The Run checks their association with its original `Plan` and their sources. Before each new submission or host invocation, it records the matching point assessment in `event.assessment_id`. A future adaptive point without a forecast keeps `None`. Neither continuation nor opening a saved Result reevaluates models, changes shots or configures a backend from an Allocation.

## Progress display

In a live notebook, `solve` and `prepare` update one compact progress display. It shows input acceptance, preparation, submission, collected observations and analysis. Preparation and collection counts come from saved arrays and observations, and unknown totals stay unknown. Scripts are quiet by default. Use `progress=False` to turn the display off, or supply a synchronous `progress(stage, done, total)` callable. If the callable fails, the Run turns it off and records a process-local notice in `run.warnings`. It never retries scientific work, draws another seed or resubmits a job. Finishing analysis is an execution stage, not a scientific accuracy claim.

## Submit the same Prepared again

The first `submit(prepared)` starts its Run. Calling `submit` again on the same `Prepared` creates a new Run with new cumulative limits and the original randomness of the `Plan`. It does not continue the first Run. The original folder is not reused. Synchronous local work gets no new run log unless it is prepared with a directory, and a remote Run chooses its own folder. The same `Plan` seed can reproduce the same sampled stream, so a new Run does not establish statistical independence. To continue existing work, use the returned Run's `resume()` or `wait()`.
