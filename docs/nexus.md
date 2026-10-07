# Quantinuum Nexus

<a id="quantinuum-nexus-h2"></a>`NexusBackend` runs NWQLib's circuits on Quantinuum H2 hardware and emulators through Quantinuum Nexus. Construct a connection for an existing Nexus project and run with the steps in [Run on any backend](backends.md#run-on-any-backend). The example below is a template and is not run here, because it needs a Nexus login:

```python
from nwqlib.backends.nexus import NexusBackend

backend = NexusBackend(
    project="YOUR-EXISTING-PROJECT-UUID",
    device="H2-1",
    target_region="us",    # execution region, "us" (the default) or "sg"
    credential_name=None,  # optional name of a Nexus-linked credential
    optimization_level=1,
    max_cost_hqc=100.0,    # per submitted program, in Hardware Quantum Credits
    max_input_bytes=1_000_000,
)
```

Construction imports no provider SDK and does not log in. `max_cost_hqc` is a cap per program, [not a total](#cost-cap-and-timeouts). Without it, the default `None` sets no cap.

## Install {#costs-timeout-and-qualification}

The `nexus` extra, `pip install "nwqlib[nexus]"`, gives the normal installation with dependency resolution. The qualified environment instead keeps pandas 3.0.5, which conflicts with the `pandas>=2,<3` requirement that qnexus 0.49.0 declares. To install that environment:

1. Install NWQLib with Qiskit, the remaining public qnexus requirements and `selene-core==0.3.2` with normal dependency resolution. Run the command first with `--dry-run` to check the proposed changes against the environment:

    ```bash
    python -m pip install "nwqlib[qiskit]" \
        'click>=8.1,<9' 'colorama>=0.4,<1' 'httpx>=0,<1' 'hugr==0.18.6' \
        'nest-asyncio2>=1.6,<2' 'pandas==3.0.5' 'pydantic-settings>=2,<3' \
        'pydantic>=2.4,<3' 'pyjwt>=2.10.1,<3' 'pytket==2.18.1' \
        'quantinuum-schemas==7.8.2' 'rich>=13.6,<14' 'websockets>11,<16' \
        'pytket-qiskit==0.78.0' 'selene-core==0.3.2'
    ```

2. Only after those dependencies are present, install qnexus alone without dependencies: `python -m pip install --no-deps qnexus==0.49.0`.
3. Run `python -m pip check`. It must report exactly one conflict, `qnexus 0.49.0 has requirement pandas<3,>=2, but you have pandas 3.0.5.`

Do not apply `--no-deps` to the whole NWQLib extra or its other packages. The published qnexus wheel imports `selene_core.trace` without declaring that dependency, which is why step 1 installs selene-core explicitly. This exception does not make `pip check` succeed, and it must not downgrade pandas or patch the SDK automatically.

The checked environment has qnexus 0.49.0, pytket 2.18.1, pytket-qiskit 0.78.0, quantinuum-schemas 7.8.2, hugr 0.18.6, selene-core 0.3.2 and pandas 3.0.5, with the existing numerical and Qiskit packages unchanged. `pip check` reports the pandas conflict and no other. No pandas 3 runtime failure was observed in the offline checks, and there is no private SDK patch or compatibility shim. A separate offline CI job installs exactly this combination with these steps, from a source checkout with `-e ".[dev,aer]"` in place of `"nwqlib[qiskit]"` and `docs/ENVIRONMENT_LOCK.txt` as a constraint, and fails on any additional dependency conflict ([Maintenance](MAINTENANCE.md)). The first use in a run warns about the offline qualification, this metadata conflict and the [HTTP wait without a timeout](#cost-cap-and-timeouts). A run reopened after it has saved its state does not repeat the warning.

## Log in and choose a project

Use an existing qnexus login, configuration and project. The backend never creates a project, logs in, changes environments, estimates cost by running a syntax checker, or retries a submission. `credential_name` names an account or key reference in Nexus and is not a token. The execution region is `us` or `sg`. The login or home region of the Nexus connection stays in the SDK configuration.

Before the first remote request of each operation, the backend checks that local authentication is present. It keeps the SDK's singleton client and its in-memory cookies, including authentication that the SDK loaded from disk, and accepts the SDK-managed token environment. It does not reload the client, inspect token contents, or call the login and expiry checks that use the network. Empty local authentication raises an error before any request. Presence does not prove that authentication is valid, so expiry and service errors still come from the SDK and propagate unchanged during the remote work you request.

## Upload and compilation {#durable-preparation-boundary}

Preparation first builds the circuit locally. Each dense `UnitaryGate` is replaced by its [exact synthesis](backends.md#exact-synthesis-of-dense-matrices), public Qiskit `transpile` translates the circuit to `u` and `cx` at optimization level 0 with the original runtime seed, and the public Qiskit-to-pytket converter runs once, without simulation. The converter receives its documented `UGate` and `CXGate` representation, and the original named classical bits are kept. Qiskit carries the global phase when translating rotations to this basis.

Preparation then uploads the circuit and starts a Nexus compile job. Upload and compilation are preparation work, not quantum executions or measured shots. The run saves each step before and after its request and never repeats an upload or compilation after an interruption. A step whose acknowledgement was lost is looked up by its name, `nwqlib:upload:<id>`, `nwqlib:compile:<id>` or `nwqlib:execute:<submission>`, in the original project, among at most two matches. No match leaves the step unresolved and several matches raise, and neither case permits an automatic replacement upload, compilation or execution. The step-by-step rules are in [Backend adapter contract](development/execution.md#backend-adapter-contract).

`optimization_level` is the level of the Nexus compile job, 1 by default, and the [rules shared by all backends](backends.md#optimization-level) apply. Each target identifies a stable project, device and region, and the compiler record keeps the compile-job ID, so that independently compiled programs can form one compatible execute request.

A compiled program is opaque. Its native operation count, native gate inventory and logical-to-physical qubit mapping remain unavailable. A hard limit that needs that inventory rejects the run before execution. Restoring results needs only the saved compiled reference and the original bit map, not a local circuit or a reconstruction of the payload. The backend does not download a compiled circuit only to report its operations.

The submitted job description contains the ordered program IDs, requested shots, project, target configuration, execution region, optimization level, credential name and requested HQC cap. NWQLib supplies this description to match jobs, and Nexus returns it. The job ID, project, job type, backend configuration, input and output program references and result statuses are data reported by the provider. The description is not a shot count or authentication confirmed separately by the provider. Restoring a run reads SDK formats without a circuit hash, replanning, numerical reconstruction or mathematical revalidation.

## Counts and job status

The circuit route accepts the named H2 hardware and emulator devices (`H2-N`, `H2-NE`, `H2-NLE`) and rejects syntax-checker targets, readouts other than counts, and Helios before submission. Each program may run 1 to 10,000 shots. The range applies to all shots of the program, such as a sampled query's shots times its multiplicity, and every experiment is checked against it before the first preparation or submission. A static `Plan` also checks the readout of every experiment against the target before its first preparation. Such a refusal names the experiment and, for a counts readout, the shots per program that it requests, or for any other readout its readout kind. Batch items may have unequal shot counts, but each must refer to a distinct compiled program. Repeated use of one compiled program requires separate Nexus submissions, because the backend does not assume an undocumented ordering convention for job items.

The decoder joins each returned result to its original input `CircuitRef`, checks the original project and classical bits, and calls pytket's public `BackendResult.get_counts(cbits=...)` with the full original global bit order. Joint outcomes over several registers stay joint, and registers are never marginalized separately and multiplied. NWQLib writes global classical bits from high to low. Raw integer counts stay unnormalized, and the returned shots may be fewer than requested. Result tuples keep the provider's return order, while their keys identify the original submitted items.

A completed item can be saved while another item is still pending. Growing `RUNNING` histograms are not downloaded or saved early. A provider failure or cancellation can carry final partial counts. A transport or decoder error raises its original exception, and the run keeps the known job locator and the observations already saved. If a later item fails within one refresh, the run first saves the earlier completed results and their associations, then raises the original exception with its cause. The unfinished result stays pending, and no error is turned into a successful completion. Repeated fetches do not create new measurements, and the run saves each result once, bound to its original submitted item. After reopening, an already completed item is skipped before any input or result is downloaded. There is no temporary association or raw-array cache. A cancellation request does not claim immediate confirmation or return the work already counted.

## Cost cap and timeouts

`max_cost_hqc` is passed through `start_execute_job(max_cost=[...])` once for each submitted program. It is an HQC limit, not currency, an estimate of the required cost or a total across the batch. The backend does not read the provider's actual cost or its child hardware jobs, and one Nexus job is not evidence of one physical hardware invocation. The run's stored-data limits cover the represented JSON, bit and count data and the saved output. SDK HTTP buffering, internal serialization, local Qiskit transpilation and remote compilation cost, process memory and server billing are outside those limits. `max_input_bytes` bounds a conservative estimate of the JSON size before the pytket circuit is encoded.

**qnexus 0.49.0 uses an HTTP client with `timeout=None`.** Its public create, get and result methods offer no per-request timeout. A refresh performs a bounded number of SDK operations and never waits through the queue, but one request can block indefinitely. NWQLib installs no patch of the private HTTP client or of global sockets. The SDK can refresh authentication after an HTTP 401 and resend the unauthenticated request, and the backend never calls `retry_submission`.

## What has been checked

Real SDK imports, public reference and configuration models, a small Qiskit-to-pytket conversion, reference serialization, and pytket result decoding backed by shots and by counts pass offline tests. Public service boundaries are replaced by test doubles whose signatures are checked, and no live service, credentials, compilation or execution was used. These checks establish a bounded offline qualification, not general SDK or server compatibility.

An offline test of an Expectation run goes through a pending upload and compilation, a reopening while queued that reads no payload, compile completion with execution pending, count saving and a final reopening with zero SDK calls. A two-program interruption check confirms that the first result survives a later download error and is not downloaded again. These are small offline fixtures, not cloud execution.

## Helios and sources {#helios-boundary-and-sources}

Helios is a separate program and result route. The official workflow uploads HUGR and executes it with `HeliosConfig`, and the generic `start_execute_job` also accepts QIR references. Its QSYS/HUGR and QIR result formats are not pytket register counts, and the H2 circuit converter is not evidence that those programs are valid for Helios. Supporting Helios would need an NWQLib route that produces such programs and an explicit mapping from their result records to the requested observations. The documented Helios maximum-cost behavior also has a multi-program caveat.

The documented pytket bridge is `guppy.load_pytket(name, input_circuit, use_arrays=True)`, called by a Guppy entry point that is compiled to HUGR. The register arguments and results use lexicographic order, measured qubits require explicit disposal, and only gates supported by the Guppy quantum library may be loaded. These are additional program and result conventions, so H2 conversion tests do not qualify that route. Nexus does not offer server-side tket compilation for HUGR programs.

- [Public compile API](https://docs.quantinuum.com/nexus/nexus_api/compile.html)
- [Public execute API](https://docs.quantinuum.com/nexus/nexus_api/execute.html)
- [Job retrieval, results and cancellation](https://docs.quantinuum.com/nexus/nexus_api/jobs.html)
- [Public reference types](https://docs.quantinuum.com/nexus/nexus_api/references.html)
- [H2 workflow](https://docs.quantinuum.com/systems/trainings/h2/getting_started/)
- [Helios workflow](https://docs.quantinuum.com/systems/trainings/helios/getting_started/)
- [Systems program and compilation pathways](https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/workflow.html)
- [Public Guppy pytket-loading API](https://docs.quantinuum.com/guppy/api/decorator.html)
- [Helios HQC costing and multi-program caveat](https://docs.quantinuum.com/systems/trainings/helios/getting_started/costing.html)
- [Backend configurations and HQC units](https://docs.quantinuum.com/nexus/nexus_api/backend_configs.html)
- [qnexus source at the inspected revision](https://github.com/Quantinuum/qnexus/tree/7c5b696744224af5b9fa4c6bfc78a016f0a7c04b)

The execute-job interface carries one shot count and cost cap per program item, and NWQLib applies its 10,000-shot H2 limit to each item. This reading follows the [per-program Nexus API](https://docs.quantinuum.com/nexus/_modules/qnexus/client/jobs/_execute.html) and the [multi-program batching guide](https://docs.quantinuum.com/systems/trainings/h2/getting_started/batch_jobs.html), and does not treat the Nexus container as one H2 program. It does not qualify the provider's internal scheduling or aggregate batch limits. The [Nexus job-size limit](https://docs.quantinuum.com/nexus/user_guide/concepts/jobs.html#job-size) allows at most 300 programs per execute job, which NWQLib checks before submission.
