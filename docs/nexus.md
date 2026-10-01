# Quantinuum Nexus H2

`nwqlib.backends.nexus.NexusBackend` implements the public Nexus H2 provider operations. It is used by the common preparation and measurement owner. Its individual upload/compile methods are not a substitute for the durable execution check. Construction imports no provider SDK and does not log in.

```python
from nwqlib.backends.nexus import NexusBackend

backend = NexusBackend(
    project="YOUR-EXISTING-PROJECT-UUID",
    device="H2-1",
    target_region="us",  # explicit execution region; never ambient context
    credential_name=None,  # optional name of a Nexus-linked credential
    optimization_level=1,
    max_cost_hqc=100.0,  # per submitted program, in Hardware Quantum Credits
    max_input_bytes=1_000_000,
)
```

Use an existing qnexus login/configuration and project. This adapter never creates a project, logs in, changes environments, estimates cost by executing a syntax checker, or retries a submission. `credential_name` is an account/key reference in Nexus, not a token. The execution region is `us` or `sg`; the Nexus connection's login/home region remains owned by the SDK configuration.

Before the first remote request in each operation, the adapter checks local authentication presence. It preserves the SDK's selected singleton client and its in-memory cookies, including authentication loaded by the SDK from disk, and accepts the SDK-managed-token environment. It does not reload the client, inspect token contents, or call the network-backed login/expiry checks. Empty ordinary local authentication raises an error before any request. Presence is not proof that authentication is valid: expiry and service errors still belong to the SDK and propagate unchanged during explicitly requested remote work.

## Durable preparation boundary

The common owner must record these steps independently. Upload and compilation are preparation work, not quantum executions or measured shots.

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

`optimization_level` is the level of the Nexus compile job, 1 by default. A level other than 1 is recorded as the exclusion `optimization_level` in the preparation record's `probability_window_exclusions`, which marks a circuit compiled at a non-default level. The Nexus target has no derived roundoff constant, so `state_error()` gives no bound at any level.

The submitted description contains the actual ordered program IDs, requested shots, project, target configuration, execution region, optimization level, credential name and requested HQC cap. This is NWQLib-supplied correlation metadata echoed by Nexus. The job ID, project, job type, backend configuration, input/output program references and result statuses are provider-reported data. The correlation description is not a separately provider-confirmed shot count or authentication scheme. Restore reads SDK formats without a circuit hash, replanning, numerical reconstruction or mathematical revalidation.

A compiled reference is opaque: native operation count, native gate inventory, and logical-to-physical quantum mapping remain unavailable. A hard limit needing that inventory must reject before execution. Result-only restoration needs the saved compiled reference and original bit map, not a local circuit or payload reconstruction. The adapter does not download a compiled circuit merely to produce an inventory report.

Local basis lowering uses the original runtime seed. It supplies the converter's documented `UGate`/`CXGate` representation and preserves original named classical bits. Qiskit carries global phase when translating rotations to this basis. Each target source identifies stable project/device/region; the compiler source keeps the actual compile-job ID so independently compiled programs can form one compatible execute request.

## Counts and job status

The circuit path accepts named H2 hardware/emulator devices (`H2-N`, `H2-NE`, `H2-NLE`) and rejects syntax-checker targets, non-count readouts and Helios before submission. Each program may run 1–10,000 shots. The range applies to the program's pooled population, such as a sampled query's shots times its multiplicity, and every Experiment is checked against it before the first preparation or submission. A static `Plan`, the selected construction and its costs, computed before any circuit exists, also checks the readout of every experiment against the target before its first preparation. Such a refusal names the experiment and, for a counts readout, the shots per program that it requests, or for any other readout its readout kind. Batch items may have unequal shot counts, but each must refer to a distinct compiled program. Repeated use of one compiled program requires separate Nexus submissions because this adapter does not assume an undocumented job-item ordinal convention.

The decoder joins each returned result to its actual input `CircuitRef`, checks the original project and classical bit inventory, and calls pytket's public `BackendResult.get_counts(cbits=...)` with the full original global bit order. Joint multi-register outcomes remain joint; registers are never independently marginalized and multiplied. NWQLib renders global classical bits from high to low. Raw integer counts remain unnormalized, and returned shots may be fewer than requested. Result tuples keep the actual provider return order while their keys identify original submitted items.

A terminal completed sibling can be published while another item is pending. Growing `RUNNING` histograms are not downloaded or prematurely published. Provider terminal failure/cancellation can carry final partial shot populations. A transport or decoder error propagates its original exception; the common owner keeps the known job locator and previously published observations. If a later item fails within one refresh, `BackendRefresh.error` carries that same exception alongside earlier completed results. The common owner saves their associations and observations first, then raises the original exception with its cause preserved. The unfinished result stays pending; no error is converted into successful completion. Repeated fetches do not create new measurements. The common owner controls exactly-once publication/consumption. The adapter returns the actual result reference ID through `BackendRefresh.associations`, bound to the original `SubmissionItem.provider_result_id`. The common owner persists that association before observation publication. After reopening, an already completed item's exact saved result ID is skipped before any input/result download. There is no transient association or raw-array cache. Cancellation requests do not claim immediate confirmation or refund charged work.

## Costs, timeout and qualification

`max_cost_hqc` is passed through `start_execute_job(max_cost=[...])` once for each submitted program. It is an HQC limit, not currency, an estimate of required cost, or a total across the batch. The adapter does not read actual provider cost or child hardware-job populations, and one Nexus job is not evidence of one physical hardware invocation. Run stored-data limits cover represented JSON/bit/count payloads and saved output. SDK HTTP buffering, internal serialization, local Qiskit transpilation and remote compilation cost, process RSS and server billing remain outside those limits. `max_input_bytes` bounds the conservative JSON envelope before encoding the selected pytket circuit.

**qnexus 0.49.0 uses an HTTP client with `timeout=None`.** Public create/get/result methods expose no per-request timeout option. A detached refresh performs a bounded number of SDK operations and never waits through the queue, but one request can block indefinitely. No private HTTP-client or global-socket patch is installed. The SDK can refresh authentication after an HTTP 401 and resend the unauthenticated request; the adapter never calls `retry_submission`.

The checked environment has qnexus 0.49.0, pytket 2.18.1, pytket-qiskit 0.78.0, quantinuum-schemas 7.8.2, hugr 0.18.6, selene-core 0.3.2, and pandas 3.0.5. Authorized installation kept the existing numerical and Qiskit stack. Dependencies were resolved normally, followed by qnexus `--no-deps` solely to keep pandas 3 despite qnexus' declared `pandas>=2,<3` cap. The published qnexus wheel imports `selene_core.trace` without declaring that dependency, so it was installed explicitly with its normal dependencies.

The `nexus` extra describes the normal dependency-resolved installation. To keep pandas 3.0.5 in this specifically qualified environment, first install the remaining public qnexus requirements and `selene-core==0.3.2` using normal dependency resolution, checking the proposed changes against that environment. Only after those dependencies are present, install `qnexus==0.49.0` with `python -m pip install --no-deps qnexus==0.49.0`. Do not apply `--no-deps` to the whole NWQLib extra or its other packages. This exception does not establish `pip check` success and must not automatically downgrade pandas or patch the SDK. The `Nexus Offline Qualification` workflow in `.github/workflows/nexus-qualification.yml` records the complete install command and runs separately from the stable/latest Full CI environments. It requires exactly this one metadata conflict, fails on any additional conflict, imports the real SDK before testing, and exercises the offline adapter, decoding and archive workflow. It does not contact the service. The first actual use in a run warns about offline qualification, this metadata conflict and the unbounded HTTP wait; the notice survives journal reopen.

Real SDK imports, public reference/configuration models, a small actual Qiskit-to-pytket conversion, reference serialization, and both shots-backed and counts-backed pytket result decoding pass offline tests. Public service boundaries are replaced by signature-checked test doubles; no live service, credentials, compilation or execution was exercised. These checks establish bounded offline qualification, not universal SDK or server compatibility. `pip check` reports the kept pandas metadata conflict and no other conflicts. No pandas 3 runtime failure was observed in this scope; there is no private SDK patch or runtime compatibility shim.

The public Expectation archive test traverses upload/compile pending, queued reopen without payload reads, compile completion/execution pending, count publication and a final reopen with zero SDK calls. A two-program interruption test verifies that the first result survives a later download error and is not downloaded again. These are small offline fixtures, not cloud execution.

## Helios boundary and sources

Helios is a separate program/result route. The official workflow uploads HUGR and executes it with `HeliosConfig`; generic `start_execute_job` also accepts QIR references. Its QSYS/HUGR and QIR result formats are not pytket register counts. The H2 circuit converter is not evidence that those programs are valid for Helios. No unrelated language frontend is added here. Helios support requires a concrete NWQLib program-producing consumer and an explicit mapping from its actual result records to the requested observations. The documented Helios maximum-cost behavior also has a multi-program caveat.

The documented pytket bridge is `guppy.load_pytket(name, input_circuit, use_arrays=True)`, called by a Guppy entrypoint that is compiled to HUGR. The register arguments/results use lexicographic order; measured qubits require explicit disposal, and only gates supported by the Guppy quantum library may be loaded. These are additional program/result semantics, so H2 conversion tests do not qualify that route. Nexus does not offer server-side tket compilation for HUGR programs.

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

The execute-job interface carries one shot count and cost cap per program item. NWQLib applies its 10,000-shot H2 policy to each item. This interpretation follows the [per-program Nexus API](https://docs.quantinuum.com/nexus/_modules/qnexus/client/jobs/_execute.html) and the [multi-program batching guide](https://docs.quantinuum.com/systems/trainings/h2/getting_started/batch_jobs.html), rather than treating the Nexus container as one H2 program. It does not qualify the provider's internal scheduling or aggregate batch limits. The [Nexus job-size limit](https://docs.quantinuum.com/nexus/user_guide/concepts/jobs.html#job-size) allows at most 300 programs per execute job, which NWQLib checks before submission.
