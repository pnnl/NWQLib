# IonQ

<a id="ionq-execution"></a>`IonQBackend` runs NWQLib's circuits on an IonQ QPU through the IonQ v0.4 API and returns raw sampled counts. Install `nwqlib[ionq]`, construct a connection for one `qpu.*` device and run with the steps in [Run on any backend](backends.md#run-on-any-backend). The example below is a template and is not run here, because it needs an IonQ API key:

```python
from nwqlib.backends.ionq import IonQBackend

backend = IonQBackend(
    device="qpu.forte-1",
    gateset="qis",
    max_input_bytes=1_000_000,
    max_response_bytes=1_000_000,
)
```

The configuration cannot change after construction, and constructing it neither imports the provider SDK nor contacts IonQ. The byte values in the example are limits, not provider cost estimates. Choose `ExecutionLimits` for the intended workload.

Before any conversion or request, the backend rejects readouts other than counts, the ideal simulator (which ignores the requested shots) and shot counts outside 1 to 1,000,000.

## Credentials

Set `NWQLIB_IONQ_API_KEY` only when you execute or retrieve a job. Credentials are not stored in the configuration, the preparation record or the run folder.

## Circuits and gates

The backend converts circuits with the public `qiskit_ionq.helpers.qiskit_circ_to_ionq_circ` converter.

- **QIS gates** (`gateset="qis"`): Qiskit's public transpiler translates the circuit to Rx, Ry, Rz and CX at `optimization_level`, 0 by default, under the [rules shared by all backends](backends.md#optimization-level). Rotations stay in radians.
- **Native gates** (`gateset="native"`): the circuit is converted without Qiskit translation, and only level 0 is accepted. Native SDK gates reach the converter with their turns-valued parameters unchanged.

The flat circuit route requires final measurements and keeps each measured qubit's classical-bit destination. Unmeasured classical bits stay zero. It does not offer OpenQASM 3, mid-circuit measurement, reset or control flow.

## Submit and wait

`nwqlib.prepare(plan, backend=backend, directory=...)` followed by `nwqlib.submit(prepared)` returns a live run. The job can stay pending, and `run.resume()` and `run.wait()` retrieve the original jobs and advance the method. The run saves each circuit as QPY before the original submission. To continue later, close the run, then reopen its folder with `nwqlib.load_run(directory, backend=backend)`. A retrieval exception keeps the job for a later explicit refresh.

## Batches

The lower-level `submit_detached((handle,), run=run)` submits several circuits as one job when their requested shots are equal. A batch holds at most 5,000 circuits and 150,000 gates in total, the multi-circuit limits of the [create-job reference](https://docs.ionq.com/api-reference/v0.4/jobs/create-job). NWQLib counts one gate per entry of a circuit's IonQ gate list and applies the same 150,000-gate total to a single-circuit job, for which the reference gives no gate limit.

Each circuit receives a name made of its original submission and item. Each returned child job must have that name and the correct parent job, and its position in the returned list does not decide which measurement it is. The run folder saves the child job IDs before their individual observations, and keeps the raw result ID of each child before downloading it. Children are read once the parent job is no longer pending, and completed children can then be used while others are still pending. If a later status, download or decode fails, the run first saves the child and result IDs and the completed observations it has, then raises the original exception with its cause. Those completed observations stay available from their original attempts. Later refreshes skip results already saved and refuse to replace a known result ID. One batch remains one submitted job, with its own set of shots per child. Duplicate names, duplicate child IDs and changed associations are rejected before any result is downloaded.

## Read the counts

The readout requires a [raw histogram result](https://docs.ionq.com/api-reference/v0.4/schemas/results-formats). Both the decimal-state histogram v1 and the joint `output_all` histogram v2 are accepted. The decoder keeps unequal integer counts and applies the original measurement map. A v1 decimal key is read as an integer whose bit q is qubit q, the convention of qiskit-ionq's own decoder. A v2 binary key is read most significant bit first to give the same integer. That v2 reading is checked only against offline fixtures. Classical bit b takes the value of the qubit that the converter's measurement map assigns to b, and count keys put classical bit 0 rightmost. The decoder checks the shots it read against both the histogram sum and the accepted shot count. Separate register marginals cannot establish a joint histogram and are rejected. Probability results are never rounded, resampled, sharpened, or reported as binomial counts or an exact readout. Counts from a new physical measurement do not establish independent identically distributed samples, stationarity or an error bound, and the method's own statistical assumptions still apply.

For a histogram with `m` entries on `q` native qubits and `c` classical bits, decoding at `q,c<=64` checks `57*m + 8*min(m, 2**min(q,c))` bytes for live ndarray elements, including the returned index and count pair. The bytes of the count keys, `m*(c+16)`, are checked separately. Wider inputs use Python integers and bit-string keys within those key bytes. Public count records count against the run's stored-data limit. These per-decode checks do not bound Python object memory, transport buffers or the total memory of a batch.

Every job, single-circuit or batch, is sent with debiasing disabled, and unexpected nested child jobs or debiasing variants are rejected before the results are saved. The [job schema](https://docs.ionq.com/api-reference/v0.4/jobs/get-job) gives variant IDs, shots and a nullable `qubit_map`, but does not define that map's direction or the coordinate convention of each variant result. Supporting variants would require keeping their separate sets of shots and establishing those conventions, and they cannot be pooled by assumption. These are limits of NWQLib's support, not a claim that IonQ lacks those service features.

## Retries and cancellation

One submission makes one documented [v0.4 create-job request](https://docs.ionq.com/api-reference/v0.4/jobs/create-job), not the SDK's retrying job-creation method. A lost acknowledgement leaves the submission uncertain, with its shots and jobs still counted, and no replacement job is submitted. The public API has no qualified idempotency key or exact lookup by metadata or name, so an unknown job ID cannot be recovered automatically, and an empty or partial job listing is never taken as proof of rejection. Without an acknowledged job ID, `run.wait()` raises `RunFailed` with an uncertain recovery status.

Cancellation requests cancellation of the known job, and only later status establishes the outcome.

## Byte limits and timeouts

`max_input_bytes` limits serialized requests and the optional QPY output. `max_response_bytes` limits the HTTP body bytes that the backend reads from one response. The reader never holds more than one byte beyond it, and that extra byte is what reveals an oversized body, which is then rejected. Network buffering inside requests and urllib3, JSON object memory, provider compilation, runtime, process memory and billing are not bounded by these fields. The request timeout is a connect and read timeout, not a total wall-clock deadline.

Each refresh reads the known parent job once and, when the parent is no longer pending, each outstanding child's status once. It downloads a completed child's raw result once, skipping child measurements already saved. A failed download or decode may need a later explicit download of the same missing item, and children decoded before the failure are kept and not downloaded again. An interruption before the results are saved remains distinct from that handled error. No retrieval failure causes another submission. The run's stored-data limits cover the represented results and saved output and do not bound provider CPU time, network latency or money spent.

## What has been checked

Offline checks used Qiskit 2.5.2 and qiskit-ionq 1.1.1 with injected HTTP responses, including the common finite-Pauli analysis and reopening a saved run. No live QPU, account, queue, result availability or hardware accuracy was qualified. Preparation warns about that limitation once per run. Submission and retrieval go only through the common run interface described above.
