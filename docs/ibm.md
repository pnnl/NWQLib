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

Preparation transpiles to the device's instruction set and keeps the logical-to-physical layout, the requested options and the SDK versions. The default `optimization_level` is 1, and the [rules shared by all backends](backends.md#optimization-level) apply.

The run folder keeps each circuit as QPY (Qiskit's binary circuit format) before submission, so a prepared but unsubmitted circuit survives an interruption. Later retrieval uses the original job and does not serialize the circuit again, and resuming a known job uses its original job ID without recompiling or submitting another job. The saved QPY circuits have ordinary local IDs and keep their saved bytes. They are not content-hashed or authenticated, and editing them does not trigger a check of the archive's mathematical assumptions. Jobs already submitted and their returned results still describe their original executions. A failed retrieval records the original job state and raises the original exception with its traceback, and a later explicit refresh can retrieve the same job. Errors in decoding a result are raised and are not treated as a pending job.

## Counts

SamplerV2 returns joint classified counts. Each PUB (primitive unified bloc, the circuit with its parameter values and shot count that the Sampler receives) keeps its original item position, also when one prepared circuit appears more than once. Named classical registers are joined shot by shot before the logical bit order is restored. Each physical circuit execution produces its own set of shots, which by itself establishes neither independence nor stationarity. Fixed simulator seeds can repeat known data and do not establish independent observations.

The bytes of the NumPy array elements held in memory while one scalar Sampler PUB is decoded, including the packed register inputs, are computed by `sampler_decode_bytes` in `nwqlib.backends.ibm_runtime` and checked against the run's `ExecutionLimits.max_data_bytes` before joining. Up to 64 classical bits, this bound covers the packed join, uint64 shot arrays and the sorted unique and count workspace, and the backend returns one uint64 index and one int64 count per distinct outcome. Above 64 classical bits, it checks the padded join array bound listed in [Engineering constants](ENGINEERING_CONSTANTS.md) against that limit and returns bit-string count keys. Converting the packed rows to Python bytes, integers, strings and count dictionaries adds no ndarray element buffers. The shared decoder checks either representation against the cardinality of the prepared readout and builds the public count records. This check applies `max_data_bytes` to one PUB at a time, separately from the run's cumulative total of saved data, and does not bound Python object memory, Runtime transport or whole-process memory.

## Expectation estimates

For finite-Pauli expectation, `ExpectationMethod.estimate_precision` selects EstimatorV2. The full weighted observable, including identity terms and small coefficients, follows the circuit's physical layout. The returned signed estimate is kept without clipping. Provider `stds`, an optional ensemble standard error, the requested settings and the returned metadata are kept separately. Without ZNE, the documented `stds` can supply the sampling standard error. With ZNE, `stds` describes the fit uncertainty and is not presented as a sampling standard error or a bound on physical bias. Missing or ambiguous uncertainty stays unavailable. The provider manages the shots, which the raw-shot limit does not count, so that limit does not cap Estimator spending. The default resilience level is zero. Other supported options require explicit selection, and experimental overrides are rejected before preparation.

## Retries and cancellation

Every job carries the tag `nwqlib:<submission UUID>`. If its job ID never reaches the run folder, because the process stopped or the launch raised after the request was sent, the submission stays uncertain and its work stays counted, and the next refresh searches for the job. It asks the service for at most two jobs with that tag on the configured device, instance and primitive. One match whose instance, primitive, device and tag agree with the submission becomes the saved reference to the job. No match leaves the submission unresolved, and two matches raise, because neither case identifies the original job and neither permits a new submission.

A run cancellation stops new work. The next explicit refresh requests cancellation once, and later provider status confirms the outcome. Cancellation does not return the jobs or shots already counted. A failed or interrupted cancellation remains a request and is not repeated automatically.

The SDK retries certain 5xx POST responses under IBM's documented safe-retry rules, with at most five retries per HTTP request and at most three connection retries. NWQLib itself never resends a job request. The Runtime 0.49 transport was checked through a local HTTP server with the real SDK client, primitive, job and decoder. A lost POST acknowledgement was not resubmitted, and the tag lookup recovered the original job. A rejected 503 followed by success, and a retried GET, were checked. These tests establish the behavior of the client, not server billing or hardware accuracy. Internal SDK buffers, the read timeout, compilation memory and total network traffic are not bounded by the backend's metadata limit. Later SDK versions need the same transport check before these statements apply to them.
