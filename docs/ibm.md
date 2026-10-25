# IBM Runtime

<a id="ibm-runtime-execution"></a>`IBMRuntimeBackend` runs NWQLib's circuits on an IBM Quantum device through Qiskit IBM Runtime. Install the `ibm` extra, construct a connection for your device and instance, and run with the steps in [Run on any backend](backends.md#run-on-any-backend). The example below is a template and is not run here, because it needs an IBM account:

```python
from nwqlib.backends import IBMRuntimeBackend

backend = IBMRuntimeBackend(
    device="YOUR-IBM-DEVICE", instance="YOUR-IBM-INSTANCE",
    max_input_bytes=65_536,
)
```

The configuration cannot change after construction. Offline SDK and transport checks do not qualify a live account, queue or QPU.

The backend returns sampled counts through SamplerV2, and a provider expectation estimate through EstimatorV2 when the Method requests one, as `ExpectationMethod(estimate_precision=...)` does. It rejects exact readouts, which a simulator computes from the state with `shots=None`, without shot sampling and subject to numerical rounding, and trajectory readout, which reads several points of one simulated evolution, before contacting the service.

## Credentials

The constructor reads no credentials and does not contact the service. Preparing or refreshing a run loads the SDK and its credentials. With `account_name`, the backend uses that saved SDK account. Otherwise it reads the environment variable named by `token_env`, by default `NWQLIB_IBM_RUNTIME_TOKEN`. Secrets are not saved in the run's records.

## Submit and wait {#ibm-runtime-prepared-execution}

Pass the connection to `nwqlib.prepare(plan, backend=backend, directory=...)`, call `nwqlib.submit(prepared)` and continue through the returned run. [Save and reopen pending work](run_archives.md#save-and-reopen-pending-work) shows how to save that run and reopen it later. Job mode is the default. `mode="batch"` or `mode="session"` attaches every job to the existing Batch or Session whose ID you pass as `container_id`.

Preparation transpiles to the device's instruction set and keeps the logical-to-physical layout, the effective primitive options, the SDK versions and the [native two-qubit depth](#native-two-qubit-depth) of the compiled circuit. The effective options are the requested `options_json` with the defaults that preparation writes in, resilience level 0 for an estimate and an explicit `false` for each unset dynamical-decoupling or twirling option of counts, and every job of the preparation is sent with them. The default `optimization_level` is 1, and the [rules shared by all backends](backends.md#optimization-level) apply.

The run folder keeps each circuit as QPY (Qiskit's binary circuit format) before submission, so a prepared but unsubmitted circuit survives an interruption. Later retrieval uses the original job and does not serialize the circuit again, and resuming a known job uses its original job ID without recompiling or submitting another job. The saved QPY circuits have ordinary local IDs and keep their saved bytes. They are not content-hashed or authenticated, and editing them does not trigger a check of the archive's mathematical assumptions. Jobs already submitted and their returned results still describe their original executions. A failed retrieval records the original job state and raises the original exception with its traceback, and a later explicit refresh can retrieve the same job. Errors in decoding a result are raised and are not treated as a pending job.

## Counts

SamplerV2 returns joint classified counts. Each PUB (primitive unified bloc, the circuit with its parameter values and shot count that the Sampler receives) keeps its original item position, also when one prepared circuit appears more than once. Named classical registers are joined shot by shot before the logical bit order is restored. Each physical circuit execution produces its own set of shots, which by itself establishes neither independence nor stationarity. Fixed simulator seeds can repeat known data and do not establish independent observations.

The bytes of the NumPy array elements held in memory while one scalar Sampler PUB is decoded, including the packed register inputs, are computed by `sampler_decode_bytes` in `nwqlib.backends.ibm_runtime` and checked against the run's `ExecutionLimits.max_data_bytes` before joining. Up to 64 classical bits, this bound covers the packed join, uint64 shot arrays and the sorted unique and count workspace, and the backend returns one uint64 index and one int64 count per distinct outcome. Above 64 classical bits, it checks the padded join array bound listed in [Engineering constants](ENGINEERING_CONSTANTS.md) against that limit and returns bit-string count keys. Converting the packed rows to Python bytes, integers, strings and count dictionaries adds no ndarray element buffers. The shared decoder checks either representation against the cardinality of the prepared readout and builds the public count records. This check applies `max_data_bytes` to one PUB at a time, separately from the run's cumulative total of saved data, and does not bound Python object memory, Runtime transport or whole-process memory.

## Native two-qubit depth {#native-two-qubit-depth}

The native two-qubit depth of a circuit is its two-qubit gate depth after compilation to the device's instruction set, including the routing that the compilation adds: the largest number of two-qubit gates on any chain of operations that depend on one another. Preparation records it for each compiled circuit in [`PreparedArtifact.native_depth`][nwqlib.execution.PreparedArtifact]. FixedGCIM, ADAPT and QFD read it when you pass `max_native_two_qubit_depth=D`. Each circuit is then checked just before its submission, after its preparation, and a circuit whose depth exceeds D, or whose depth is not qualified in the sense below, is not submitted. A FixedGCIM or QFD run then raises `nwqlib.backends.inspection.NativeDepthCapRefusal`, an ADAPT run ends with the stop reason `native_depth_refused` and the last pencil it solved, and in both cases the circuits submitted earlier keep their counts and shots. IBM Runtime is the only backend that computes this depth, so on every other backend an integer limit refuses the first circuit.

A recorded depth is qualified, that is, it describes the circuit that the device executes, only when all of these hold:

- The circuit is a counts (SamplerV2) preparation. An Estimator preparation never has a qualified depth, because the provider measures the observable with circuits that it builds from the submitted one.
- The effective options keep `dynamical_decoupling.enable`, `twirling.enable_gates` and `twirling.enable_measure` false, so that the provider runs the submitted circuit without changing its gates. An option set to `true` stays on and leaves the depth unqualified.
- The installed qiskit-ibm-runtime is 0.49.0, the one version whose Sampler options were checked for this. Under another version the depth is unqualified even when all three options are false.
- Every operation of the compiled circuit has a known meaning on the device target, and the circuit has at most 100,000 operations (`nwqlib.backends.inspection.inspect_native_depth` lists the operations that qualify).

An unqualified depth is unknown, never zero, and its reason is recorded in its place. With an integer limit, an unqualified depth refuses the circuit, so with a qiskit-ibm-runtime version other than 0.49.0 every IBM circuit is refused before submission. Without a limit, the default `None`, preparation and execution continue as usual. The `ibm` extra accepts later qiskit-ibm-runtime versions, so to use a depth limit, install qiskit-ibm-runtime 0.49.0. Accepting another version needs the check in [Maintenance](MAINTENANCE.md#dependency-follow-ups) first.

The limit also refuses a circuit whose preparation recorded a version of nwqlib, Qiskit or qiskit-ibm-runtime other than the installed one, as after a Run is reopened in another environment. This refusal raises `NativeDepthCapRefusal` with its `provenance` set for every method, ADAPT included, and the Run stays resumable. Reopened where the recorded versions are installed, the Run checks the same circuit against the limit again. A new Run prepared in the installed environment records depths of its own.

## Expectation estimates

For finite-Pauli expectation, `ExpectationMethod.estimate_precision` selects EstimatorV2. The full weighted observable, including identity terms and small coefficients, follows the circuit's physical layout. The returned signed estimate is kept without clipping. Provider `stds`, an optional ensemble standard error, the requested settings and the returned metadata are kept separately. Without ZNE, the documented `stds` can supply the sampling standard error. With ZNE, `stds` describes the fit uncertainty and is not presented as a sampling standard error or a bound on physical bias. Missing or ambiguous uncertainty stays unavailable. The provider manages the shots, which the raw-shot limit does not count, so that limit does not cap Estimator spending. The default resilience level is zero. Other supported options require explicit selection, and experimental overrides are rejected before preparation.

## Retries and cancellation

Every job carries the tag `nwqlib:<submission UUID>`. If its job ID never reaches the run folder, because the process stopped or the launch raised after the request was sent, the submission stays uncertain and its work stays counted, and the next refresh searches for the job. It asks the service for at most two jobs with that tag on the configured device, instance and primitive. One match whose instance, primitive, device and tag agree with the submission becomes the saved reference to the job. No match leaves the submission unresolved, and two matches raise, because neither case identifies the original job and neither permits a new submission.

A run cancellation stops new work. The next explicit refresh requests cancellation once, and later provider status confirms the outcome. Cancellation does not return the jobs or shots already counted. A failed or interrupted cancellation remains a request and is not repeated automatically.

The SDK retries certain 5xx POST responses under IBM's documented safe-retry rules, with at most five retries per HTTP request and at most three connection retries. NWQLib itself never resends a job request. The Runtime 0.49 transport was checked through a local HTTP server with the real SDK client, primitive, job and decoder. A lost POST acknowledgement was not resubmitted, and the tag lookup recovered the original job. A rejected 503 followed by success, and a retried GET, were checked. These tests establish the behavior of the client, not server billing or hardware accuracy. Internal SDK buffers, the read timeout, compilation memory and total network traffic are not bounded by the backend's metadata limit. Later SDK versions need the same transport check before these statements apply to them.
