# Run on a backend

<a id="prepared-execution"></a>Use `prepare` and `submit` instead of `solve` when you want to cap circuits, shots and saved data before anything runs, look at the circuits before they are measured, or run on a provider whose jobs can stay pending. `solve` creates a Run of its own and closes it before returning.

A `Plan` holds the construction that `plan(...)` chose and its costs, computed before any circuit exists, together with the Problem, the configured Method, the output, the scientific settings and the randomness. `prepare` creates one `Prepared` object attached to an open `Run`, and `submit` advances that Run. A pending provider job stays reachable through the returned Run.

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

`run.exposure` counts jobs per submission, counts circuits, reserved shots and provider-managed sampling per attempt, and reports these counts under `completed`, `reserved` (including jobs awaiting acknowledgement or completion), `uncertain` (including failed or cancelled provider jobs whose consumed work is unknown) and `failed`. Its `failed` category applies to confirmed failures of classical computations and adds no quantum work.

Work is counted against the Run's limits ("charged" in messages and records) when it is set aside, before it runs. It stays counted when it fails, is cancelled, is never used or has an uncertain outcome. Classical computation and unknown provider-managed sampling do not become circuit or shot counts. A provider estimate does not become a known count of raw shots.

`run.data` is a read-only record of the current observations, preparation records, execution trace and saved arrays. It shares the arrays rather than copying them. Completed observations are saved once, and collecting them again does not duplicate them. Failed and unused work, classical computations and work set aside remain in the trace.

## Run on a provider

For a provider, pass a backend object and a folder for the Run's saved state to `prepare`. This IBM template needs the `ibm` extra and an account authorized for the named device and instance. It uses the `plan` above:

```python
from nwqlib.backends import IBMRuntimeBackend

backend = IBMRuntimeBackend(device="YOUR-IBM-DEVICE", instance="YOUR-IBM-INSTANCE")
prepared = nwqlib.prepare(plan, backend=backend, directory="my-run")
with prepared.run as run:
    nwqlib.submit(prepared)
    print(run.directory)
    print(run.exposure)
```

Constructing the backend reads no credentials and does not contact the service. Remote execution requires a Run folder, and when `directory` is omitted the Run creates `~/.nwqlib/runs/<run_id>`, which `run.directory` reports. The `Plan` and the prepared backend circuits are saved before submission. Preparation may itself wait for remote compilation.

`run.resume()` retrieves the outcomes of outstanding jobs from the provider and lets the Method take its next step. `run.wait()` repeats that operation until a Result is available or its explicit timeout expires. A retrieval error is raised to the caller, the Run keeps the locator of the original provider job (the saved reference to the job, such as its job ID), and nothing is submitted again. To collect the jobs later, close the Run and reopen its folder with `load_run`, as [Continue an interrupted run](run_archives.md#save-and-reopen-pending-work) shows.

The [IBM](ibm.md), [IonQ](ionq.md), [Nexus](nexus.md), [Slurm](slurm.md) and [NWQ-Sim](nwqsim.md) pages give each backend's constructor, supported readouts and the scope of its testing.

## Limits on circuits, shots and data {#limits-on-circuits-shots-and-data}

`ExecutionLimits` caps the work of one Run. Before the first preparation of a `Plan` with fixed settings, the Run compares the circuits and shots of the whole `Plan` with their limits, and each backend call sets its circuits and shots aside before it starts.

| Field | Default | What it caps |
| --- | --- | --- |
| `max_total_circuits` | `None` (no limit) on Aer and local NWQ-Sim, 512 on other backends | Quantum preparations, and separately circuit measurement attempts, over the Run |
| `max_total_shots` | `None` (no limit) on Aer and local NWQ-Sim, 1,000,000 on other backends | Raw shots set aside, including uncertain attempts |
| `max_data_bytes` | `None` (no limit) | Saved Run data (arrays, serialized backend circuits and run-log output) beyond which the Run starts no new work. Not the memory of the computer |
| `simulator_memory_mb` | `None` (physical memory of the computer) | Simulator memory allowance in MiB. Aer receives it as its cap on quantum-state storage, and on Aer the Run checks the statevector simulation of a readout other than counts against it before preparation |

Before a circuit is prepared, the Run also checks that the direct state preparations of vectors that the circuit calls fit in the physical memory of the computer, and for a `Plan` with fixed settings it checks every setting this way before the first preparation. On q qubits each call builds a circuit of about `4 * 2**q` gates, counted at 256 bytes each, once for every time the circuit applies it. Noiseless Aer instead sets an uncontrolled, non-inverted preparation that opens a statevector simulation as the initial state, which the Run counts at `48 * 2**n` bytes for the n qubits of the circuit. A preparation that the `Plan` holds but the circuit does not call counts nothing, and on Aer and the local NWQ-Sim runner neither does an uncontrolled, non-inverted call after the position of an exact Pauli expectation or the last point of a trajectory, because they run only the instructions before it ([engineering constants](ENGINEERING_CONSTANTS.md#selected-block-lowering-limits)).

`run.trace.synthesis_work` counts the work of the exact dense-unitary syntheses that the Run's saved preparations made when a backend translated a circuit to a gate basis or when circuit construction added controls to a circuit block, in the work units of their size formula. A preparation that fails before it is saved counts none. Basis translation synthesizes each distinct matrix once per Run object, and a reopened Run synthesizes it and counts its work again. These syntheses have no time limit, and their work grows about eightfold with each further qubit of the matrix. Before they start, the Run checks their memory against the physical memory of the computer, emptying its synthesis cache first when that makes room ([exact synthesis](development/dense_synthesis.md#admission-of-the-exact-synthesis)).

Saved data are checked stage by stage, because what results will save is not known in advance and a sum of upper bounds could refuse a `Plan` that fits. Before a circuit or classical computation runs, the Run sets aside room for its declared results under `max_data_bytes`, so a `Plan` that passed the circuit and shot comparison can still stop partway at this limit. A stage refused before it starts counts nothing, and the circuits, shots and results already recorded stay as they are. `run.extend_limits(max_data_bytes=...)` followed by `run.resume()` continues a stage refused before it starts. If the limit interrupts saving an already counted preparation, that preparation stays counted without a preparation record. In a `Plan` with fixed settings, after raising the limit, `run.resume(rebuild_preparations=True)` prepares it again in the same Run without repeating any job ([rebuild an interrupted preparation](run_archives.md#rebuild-an-interrupted-preparation)). A Method whose later settings depend on earlier outcomes cannot prepare it again. A new Run with a larger limit prepares the `Plan` again and can repeat earlier jobs, and the original saved observations stay readable. The records of a started execution, such as its status, timing, provider messages and the descriptive data saved with its results, are always saved at their actual size, so the Run's saved data can exceed `max_data_bytes` by an amount that depends on those records. The Run then refuses new work, a new Result and `run.save` to a new folder until `run.extend_limits(max_data_bytes=...)` raises the limit above the saved total. The saved observations stay readable, and `load_run` reopens such a folder.

These limits count the work and data that the Run records. They are not provider billing, process memory (RSS) or proven CPU-time bounds.

### Memory of the prepared circuits {#memory-of-the-prepared-circuits}

Before it builds circuits, the Run compares a coarse estimate of their memory with the `max_bytes` of the Method, or with the physical memory of the computer when that is `None`. The estimate adds what the Run already holds, what the new circuits keep, what building them needs for a time and their new gate definitions. It counts each circuit's top-level instructions, or uses the law of a Method that has measured its own circuits ([engineering constants](ENGINEERING_CONSTANTS.md#prepared-circuit-memory-estimate)). `prepare(plan, settings="all")` compares the whole `Plan` before its first preparation. `prepare(plan)` compares the first circuit and leaves the rest of the `Plan` to the first execution, so `submit` or `run.resume()` can refuse a `Plan` that `prepare` accepted.

A refusal states the estimate and what it left out for lack of a formula, and nothing is built or counted for that request. To continue, reduce the size or accuracy of the problem, use a computer with more memory, or plan again with a larger `max_bytes`, since a Run keeps the limit of its Method. When one later circuit is refused on Aer, `run.release_native()` can also make room by freeing circuits whose results have been collected, as described under [Release prepared circuits in a long run](#release-prepared-circuits-in-a-long-run). A comparison of the whole `Plan` comes before its circuits run, so that call frees nothing for it.

The estimate is calibrated on Aer, where the measured growth of preparation at the calibrated sizes was usually within a factor of two of it. Other backends have no calibration of their own and use the same count. It is not a bound on the memory of the process. Input preparation bodies, IBM's folded circuit classes, an eager sampled QSP template and a template circuit kept from an earlier request have no estimate here, and a block you supply has one only if you pass it.

On Aer, a reopened Run compares each saved circuit with the limit before reading it. A reopened Run on another backend starts its held total at zero and adds nothing for the circuits it restores, so the total that a refusal prints is low by what was prepared and not yet submitted when the Run stopped. Those circuits were compared with the limit when they were prepared. With `max_bytes=None` that limit was the physical memory of the computer at that time, so reopening such a Run on a computer with less memory has no comparison for them.

### Raise a limit during a run

`run.extend_limits(max_total_shots=128)` sets a new cumulative cap of 128. It does not add 128 shots, reset usage or change scientific settings. Each increase appends one `LimitAmendment` record to `run.limit_amendments`, with the old and new limits, a timestamp and the counters immediately before the change. Equal values change nothing. The new cap and its record are saved together, and a failed limit check or write saves neither. Recording an increase adds saved data, so later writes can need a larger `max_data_bytes`. A Result keeps the limits and their history as they were when the Result was created, so increasing the Run's limits later does not alter an earlier Result. [Execution accounting](development/execution.md#execution-accounting) lists the recorded counters.

### Checks just before submission {#checks-just-before-submission}

The circuit and shot limits above are checked for the whole fixed workload before preparation. A Method can also check each prepared circuit just before its submission. FixedGCIM, ADAPT and QFD check their `max_2q_depth` there, because the native two-qubit depth, the two-qubit gate depth of the circuit compiled for the device, is known only after the backend has compiled the circuit ([IBM Runtime](ibm.md#native-two-qubit-depth)). A refused circuit has been prepared, and its preparation stays counted, but the Run records no submission and spends no shots on it. When one submission sends several prepared circuits, a single refused circuit refuses all of them.

Only IBM Runtime computes this depth, and on IBM Runtime `prepare(plan, settings="all")` is refused ([Prepare every setting before measuring](#prepare-every-setting-before-measuring)). `prepare` therefore builds the first setting and `submit` each later setting when it reaches it, so a deeper setting is found and refused only after the earlier settings have run. Their counts and shots stay in the Run. For FixedGCIM and QFD, `plan.method.analyze(plan, run.data, settings={})` then returns a Result that lists the settings without data in `missing` and has no projected matrices.

## Cancel, close and handle failures

`run.cancel()` prevents new work, requests cancellation of the original remote jobs and compilations and keeps all counted work. A provider status establishes cancellation, and a request alone does not. Each pending job receives one request and is then read.

A request can fail, for example because the provider refuses to cancel a job that has already finished. It is then recorded as not sent. It does not stop the other jobs from receiving theirs, and it does not stop its own job from being read. The Run saves what the read found, so the result of a finished job is collected and a finished compilation keeps its record, and then raises the first such error from `cancel`, `resume` or `wait`. If the read of that job fails too, the raised error is the request's, with a note naming the read's error. While the job has not ended, a later `run.cancel()` or `run.resume()` sends the request again and reads the job. A job recorded as ended receives no further request, and `run.wait()` returns the Result when the saved outcomes complete it. Otherwise, a cancelled Run starts no new work and `run.wait()` raises `RuntimeError` naming its folder. A request interrupted while it was being sent stays recorded as sent, because it may have arrived, and it is not sent again. A backend without a cancellation operation, such as local [NWQ-Sim](nwqsim.md), receives no request. The Run then adds a notice to `run.warnings`, and a job already launched can still finish and be read by a later `run.resume()` of the same Run. Cancellation cannot interrupt a synchronous backend call that is already running.

Closing a Run releases its private caches and its lock on the folder. It keeps the Result, the scientific data, the preparation records and all counted work. Inspect `prepared.circuits` before closing. Those returned circuit copies cannot change the prepared input. `solve` closes the Run it creates before returning its Result or raising an error, and a failed `prepare` also releases its Run. Their saved files and original jobs remain available for recovery, and closing does not cancel a job. A successful `prepare` and `submit` leave you an open Run to close yourself, for example with `with prepared.run as run:`.

`run.wait()` raises `nwqlib.execution.RunFailed` when an original compilation or execution is confirmed failed or cancelled and the Method has not returned a valid Result. The exception provides `stage`, `status`, `locator`, `failure`, `directory`, and unchangeable copies of `trace` and `exposure` taken when the error is raised. Pending work still raises `TimeoutError` when the requested timeout expires. A retrieval error keeps its original exception and locator and is not reclassified as a failed job. `run.trace.submissions[i].refresh_failure` holds the text of the last failed read or cancellation request of that job. The next successful read clears it, so it stays on a completed submission only when the request failed in the same refresh whose read found the job finished. Whether a partial result is returned depends on the Method, and no failure causes an automatic replacement submission.

When a provider or the local system confirms that a job ended with no result so far, the submission stays `uncertain` and `run.wait()` raises `RunFailed` with `stage="recovery"` instead of polling. A later `run.resume()` can still collect a result that arrives afterwards, without a replacement job and without returning the counted circuits and shots.

A classical computation that raises an exception, once its failure is saved, has event status `failed`. Its original exception propagates from `submit`, and a later `wait`, or reopening the Run, exposes `RunFailed` with the recorded failure. The invocation and its work stay counted. Unused output capacity can be released without refunding that work. Process interruption, or failure to save completion, leaves the attempt `uncertain`. It is not treated as a completed failure of a classical computation or retried implicitly.

If an interrupted attempt has no retrievable outcome, `resume()` and `wait()` raise `RunFailed` with `stage="recovery"` and `status="uncertain"`, and nothing is resubmitted. [Interrupted attempts](run_archives.md#interrupted-attempts) describes what to do next.

Ordinary `resume` does not retry an interrupted classical analysis. A Method that supports retrying its analysis accepts `run.resume(reanalyze=True)` to retry that saved analysis explicitly. Completed Results use `result.analyze(...)` instead.

### Files in a Run folder

The `.nwqlib-cache-` filename prefix is reserved for disposable cache files. After an interruption, reopening deletes files with that prefix that the run log does not reference, while it holds the Run's lock, before it counts the saved data. Current inputs and referenced cache files are kept. Keep files you create outside this prefix. Recovery covers an interrupted process (an exception, a keyboard interrupt or a killed process). Writes to the run log (`run.sqlite`) are synchronized to disk, but external input and cache files are written without `fsync`, so the folder does not guarantee recovery after power loss.

[Readout reservation](development/execution.md#readout-reservation) gives the bytes set aside for each readout, and [Execution accounting](development/execution.md#execution-accounting) describes how the Run counts work and recovers interrupted attempts.

## Prepare every setting before measuring

By default `prepare` prepares the setting that the Method submits first, which for a `Plan` whose settings are fixed in advance is its first experiment, and `submit` prepares each later setting when it reaches it. `prepare(plan, settings="all")` prepares every fixed setting in `Plan` order and submits nothing, so `prepared.circuits`, `prepared.inspect_resources(index=...)` and [logical compilation](fault-tolerant-resources.md#compile-every-setting-of-a-plan) reach every circuit before any measurement. `prepared.setting_names[i]` is the name of the `Plan` experiment (an entry of `plan.experiments`) that circuit i implements. ADAPT, Lanczos with `SensitivitySampling` and RWPE refuse `settings="all"` before a Run is created, because their later settings depend on earlier outcomes. ADAPT's later queries act on generators chosen from earlier outcomes, the main stage of `SensitivitySampling` allocates its shots from the estimates of its first, pilot stage, and each RWPE feedback angle depends on the outcomes before it. For them, `prepare(plan)` prepares the next setting.

Each setting is prepared and counted once, and a Run with a folder saves it. `submit`, and `resume` of a saved Run reopened after `prepare`, use the saved preparations without building them again. A preparation that fails or is interrupted after it was counted stays counted without a preparation record. The Run never repeats it on its own, because that would be new counted work, so `resume` raises before it submits any further setting. `run.resume(rebuild_preparations=True)` asks for that work ([rebuild an interrupted preparation](run_archives.md#rebuild-an-interrupted-preparation)). Both modes draw each setting's runtime seed in `Plan` order, so choosing `"all"` does not change the seeds. The Run checks the circuits and shots of the whole fixed workload against its limits before the first preparation in both modes, so `"all"` has no limit of its own.

`"all"` needs local Aer, local NWQ-Sim or classical execution, and remote backends are [open work](ROADMAP.md#preparing-every-setting-on-remote-backends). Another backend, such as IBM Runtime, IonQ, Nexus or Slurm, is refused before a Run is created, because its preparation can be a remote compilation that the provider charges for. The construction is the same on every backend, so for counts and exact readouts (probabilities, Pauli expectation values or amplitudes that a simulator computes with `shots=None` without shot sampling, subject to numerical rounding), preparing on Aer shows the circuit of every setting whose simulation fits `simulator_memory_mb`.

## Inspect prepared circuits

`prepared.inspect_resources(index=0)` counts the top-level operations of one prepared backend circuit by name, without copying it. The index follows the order of `prepared.circuits` and `prepared.setting_names`. Its `max_bytes`, by default the physical memory of the computer, bounds the data of that inspection, not the memory of the compiler. Supplying `transpile_options` requests one extra compilation of that circuit. The resulting count describes the compiled copy, records the effective options and does not replace the circuit that runs. Inspection requires an open Run.

Compilation follows the configured backend and the preparation record. Alternative resource contexts describe estimates and cannot change the circuit being executed. Counts of prepared circuits refer to the effective compiler configuration and layout of the backend.

An Aer noise model attached to the backend can select a narrower gate basis. When the Plan reads sampled counts, preparation translates canonical intermediate gates to that basis with Qiskit's BasisTranslator, so noise applies to the gates that are executed. The preparation record identifies this translation and the effective gate counts. Exact readouts still require a noiseless backend, and raw noisy counts do not establish a bound on physical bias.

### Release prepared circuits in a long run {#release-prepared-circuits-in-a-long-run}

A Run keeps the prepared circuit of each point while it is open, so a `Plan` with one circuit per sampled row grows with its rows. For LCHS with `estimator="sampled_pairs"` on a two-qubit input, with 198 instructions per circuit on Aer and every circuit still held, 20,000 rows raised the resident memory by 611 MB and 100,000 rows by 3135 MB, or 31 to 32 kB per row (Python 3.12.14, Qiskit 2.6.0, Aer 0.18.0, macOS, one thread). One million rows would be about 32 GB by extrapolation, which was not run. These values describe this fixture and are not a limit that the Run enforces.

For a long open Run with a folder, `run.release_native()` frees the in-memory Qiskit circuits whose results the Method has fully collected, that are not in use and that already have saved QPY copies (QPY is Qiskit's binary circuit format). It returns `released`, the content hashes of the freed preparations, and `kept`, pairs of a content hash and the reason the circuit was kept. Only Aer circuits are released, so this call makes no room on another backend. Circuits of other backends, unsaved circuits, unsubmitted work and circuits that the Method or a pending job still needs are kept. The preparation records and the order of the prepared circuits remain. Inspecting a released circuit loads its saved QPY once, without rebuilding, transpiling or submitting it, and an explicit repeated submission can use that same restored input. Shared gate definitions needed for future preparations remain cached. These counts describe released circuits, not measured bytes or RSS, and references you hold can keep circuits in memory. Keep the Run folder available for later inspection.

## Forecasts and allocations

When supplied, `run.forecast` and `run.allocation` keep the `PlanEstimate` and `Allocation` chosen for the Run. The same immutable values appear in `run.data` and the Result's data and survive save and reopen. When the Run is created, it checks that they belong to its original `Plan` and checks their sources. Before each new submission or classical computation, it records the matching assessment in `ConsumptionEvent.assessment_id`. For a circuit setting chosen from earlier results, `ConsumptionEvent.assessment_id` is `None` when the forecast contains no matching assessment. Neither continuation nor opening a saved Result reevaluates models, changes shots or configures a backend from an Allocation.

## Progress display

In an interactive notebook, `solve` and `prepare` update one compact progress display. It shows input checks, preparation, submission, collected observations and analysis. Preparation and collection counts come from saved arrays and observations, and unknown totals stay unknown. Scripts are quiet by default. Use `progress=False` to turn the display off, or supply a synchronous `progress(stage, done, total)` callable. If the callable fails, the Run turns it off and records a notice for the current process in `run.warnings`. It never retries scientific work, draws another seed or resubmits a job. A finished analysis stage reports the progress of execution, not the accuracy of the result.

## Submit the same Prepared again

The first `submit(prepared)` starts its Run. Calling `submit` again on the same `Prepared` creates a new Run with new cumulative limits and the original randomness of the `Plan`. It does not continue the first Run. The original folder is not reused. Synchronous local work gets no new run log unless it is prepared with a directory, and a remote Run chooses its own folder. The same `Plan` seed can reproduce the same sampled stream, so a new Run does not establish statistical independence. To continue existing work, use the returned Run's `resume()` or `wait()`.
