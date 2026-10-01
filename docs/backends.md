# Choose a backend

A backend connection owns preparation, submission and result retrieval for a selected method/readout. Pass it to `nwqlib.prepare(plan, backend=backend)`. The [execution guide](prepared_execution.md) explains the common workflow and [run archives](run_archives.md) explain continuation after process exit.

| Backend | Connection | Current evidence and limits |
| --- | --- | --- |
| [Aer](aer.md) | `AerBackend` | Local execution, with explicit optional noise for counts. Exact simulation remains bounded by its state storage. Its noiseless exact target also executes a multi-point trajectory schedule, with readout views of Pauli and probability points and reductions whose registered reducers supply both work and execution functions, in one simulation. |
| [NWQ-Sim](nwqsim.md) | `NWQSimBackend` | Qualified CPU paths and local process-exit continuation. Each native backend/method pair has its own build and readout limits. CPU/SV also executes a multi-point trajectory schedule, with readout views of Pauli and probability points and reductions whose registered reducers supply both work and execution functions, in one evolution. Its trajectory path has native checks on macOS arm64 only. |
| [IBM Runtime](ibm.md) | `IBMRuntimeBackend` | Offline SDK, transport and result checks. Live credentials, queues and QPUs remain unqualified. |
| [IonQ](ionq.md) | `IonQBackend` | Offline conversion and v0.4 histogram checks. No live QPU qualification. |
| [Nexus H2](nexus.md) | `NexusBackend` | Offline conversion, remote-job protocol and result checks. See its dependency exception and timeout limitation. |
| [Slurm](slurm.md) | `NWQSimSlurmBackend` | Offline scheduler/site-profile checks. Site allocation, MPI and GPU execution require the selected build and live qualification. |

## Match the method's readout

Methods declare the exact readout they require, such as counts, Pauli statistics, probability marginals, a provider estimate, amplitudes or a host output. A backend label alone does not establish support for every instruction, control, width or readout. The selected target's capability checks reject unsupported combinations before the affected work.

Host execution invokes the selected numerical method. Statevector execution simulates the selected circuit and returns its requested readout. Counts describe finite-shot execution. Resource estimation is a separate operation that does not submit a job. See the method's guide for its actual option names and supported modes.

## Estimate and inspect resources

`nwqlib.estimate(plan, context=...)` folds the selected construction. `handle.inspect_resources(max_operations=100000, max_bytes=10_000_000_000)` inventories an already prepared artifact. Nonempty transpilation options request an additional compilation for that inspection. Formula laws, representative samples and prepared inventories describe different objects and cannot be treated as interchangeable hardware costs. See [resource estimates](resources.md).

## Continue and cancel

Durable backends use the original `Run.resume()` and `Run.wait()`. Refresh uses the original job locator. A failed retrieval does not authorize a replacement submission. Run cancellation stops new work, and an explicit refresh requests remote cancellation before later status confirms the outcome.

Cloud and Slurm paths report their qualification caveat once per run, including across journal reopen. Local detached execution identifies its host and spool before launch. That computer or VM must remain running after Python exits. Provider pages document their actual limits.

QASM export remains separately available for NWQ-Sim and NWQEC. [QASM streaming](qasm-streaming.md) describes bounded export without execution.

[Fault-tolerant resources](fault-tolerant-resources.md) describes explicit NWQEC compilation and QDK physical projection. They are auxiliary operations, not execution backend connections.

## Provider rules

Every connection implements one adapter contract, stated in the module docstring of `nwqlib.backends.connection`. The common execution owner in `nwqlib._prepared_execution` saves the intent, locator and observations around each adapter call and relies on these rules.

| Rule | Failure it prevents | Owner |
| --- | --- | --- |
| Readout, shots and batch shape are checked before preparation, upload or submission | Provider work spent on a request that cannot produce the selected observation | each adapter's `target_for` and `admit_batch` |
| Intent and the submissions set aside are saved before one create request, and NWQLib never retries a create request. The IBM SDK's own safe retries are described in [IBM Runtime execution](ibm.md) | A duplicate job after a lost acknowledgement | `submit_detached` and each adapter's `launch` |
| A lost acknowledgement is matched only by the label written at launch. No match or several matches leave the intent uncertain and charged | Binding another job, resubmitting, or refunding work whose outcome is unknown | `refresh_submissions` and each `reconcile` |
| Refresh reads the original job, publishes each completed item once and keeps earlier items when a later item fails | Duplicate or lost observations | `refresh_submissions` and each `refresh` |
| Cancellation is one request to the original job, confirmed only by later status, and the submissions stay charged | Treating a request as a confirmed cancellation or a refund | `Run.cancel` and each `cancel` |
| Measurement maps and bit order are translated in one decoder per adapter | Reordered bits or mixed registers | `_decode` for IBM, IonQ and Nexus, `_submit_aer_execution` for Aer, and `NWQSimBackend._result` with the runner for NWQ-Sim and Slurm |
| Gate-angle units are fixed at preparation | IonQ native gates read in the wrong unit | `IonQBackend.prepare` keeps native turns and lowers QIS gates in radians |
| Bytes that NWQLib reads or parses are checked against the byte limit before the read or parse | Unbounded provider responses or saved payloads | `run.check_data` calls in each adapter |
| The qualification notice states offline versus live evidence once per Run | Reading an offline transport check as live device or site qualification | each adapter's `qualification_notice` |

The labels used for reconciliation are the IBM job tag `nwqlib:<submission>`, the Nexus job name `nwqlib:execute:<submission>` with its program and shot description, the Slurm job name `nwqlib-<submission>` and the NWQ-Sim spool directory `<spool>/<submission>`. No exact job lookup is qualified for the IonQ v0.4 API, so a lost IonQ acknowledgement stays uncertain.

## Source map

| Step | Source | Location | Code owner |
| --- | --- | --- | --- |
| Dense-unitary CX count `(23/48) 4**n - (3/2) 2**n + 4/3` | Shende, Bullock and Markov, [arXiv:quant-ph/0406176v5](https://arxiv.org/abs/quant-ph/0406176v5) | Eq. (19) in Appendix A (printed page 16), and Table 1 (printed page 14), row "QSD (l = 2, optimized)", which lists 0, 3, 20 and 100 CX for n = 1 to 4. The integer formula reproduces the entries for n >= 2, and the code returns the n = 1 entry 0 as an explicit base case | `backends.resources.dense_unitary_cx_qsd_upper_bound` |
| Banded block-encoding CX per query | NWQLib census of one QFT/inverse-QFT pair plus the recorded multiplexor, diagonal and PREP counts | comment in the code owner | `backends.resources.block_encoding_per_query_cx` |
| Surface-code patch size `2*d*d - 1`, with d*d data and d*d - 1 syndrome qubits | QDK 1.32.3 `SurfaceCode.provided_isa` (`qdk/qre/models/qec/_surface_code.py`), whose comment cites Horsman et al., arXiv:1111.4022, without a version (checked here against [v3](https://arxiv.org/abs/1111.4022v3)) | Sec. 7.1, pp. 18–20 (same section in v1 and v2). The rotated lattice has d*d data qubits, and its d = 5 and d = 3 examples have d*d - 1 independent stabilizers. One syndrome qubit per stabilizer gives `2*d*d - 1`. The paper also notes that at d = 3 reusing the four central syndrome qubits reduces the patch to 13 qubits, which QDK does not model | `backends.qre._estimate_qdk` |
| Syndrome cycle of one one-qubit gate time, four two-qubit gate times and one measurement time, repeated d times per logical cycle | QDK 1.32.3 `SurfaceCode.provided_isa`, whose comment cites Wang, Fowler and Hollenberg, arXiv:1009.3686, Fig. 2, without a version (checked here against [v1](https://arxiv.org/abs/1009.3686v1), the only version) | Figs. 1(b) and 2, p. 1. Fig. 1(b) orders the four CNOT layers, and Fig. 2 measures one stabilizer with four CNOTs between two syndrome measurements, without initialization gates. The one-qubit gate time, which QDK calls ancilla preparation, is QDK's addition. The d cycles per logical step match arXiv:1111.4022v3, Sec. 6, which requires d rounds of error correction after each operation | `backends.qre._estimate_qdk` |
| Logical error rate `0.03 * (p/0.01)**((d+1)//2)` per patch and lattice-surgery step, p the largest of the H, CNOT and measurement error rates | QDK 1.32.3 `SurfaceCode.provided_isa`, whose comments cite Fowler et al., arXiv:1208.0928, Eqs. (10) and (11), and arXiv:1009.3686 for the threshold, without versions (checked here against [v2](https://arxiv.org/abs/1208.0928v2)) | arXiv:1208.0928v2, Sec. VII, Eqs. (10) and (11), p. 11 (same numbers in v1), give the empirical approximation P_L ~ 0.03 (p/p_th)**d_e with d_e = (d+1)/2 for odd d and d/2 for even d. There P_L is the rate of logical X errors per surface-code cycle, fitted with p_th = 0.57% for that paper's circuits. The paper's footnote 14 says logical Z errors occur at about the same rate. QDK keeps 0.03, uses p_th = 0.01 and charges the rate per lattice-surgery step of d cycles. arXiv:1009.3686v1 reports thresholds of 1.1% to 1.4% (abstract, pp. 3–4), so 0.01 lies below them and is QDK's choice | `backends.qre._estimate_qdk` |
| Magic-state factory table | QDK 1.32.3, `qdk/qre/models/factories/_litinski.py` | `Litinski19Factory` | `backends.qre._estimate_qdk` |
| Exact Pauli expectation read directly from native CPU/SV amplitudes, without a dense Pauli matrix | NWQLib derivation from the action of a Pauli string on a basis state, with `Y = iXZ` | comment in the code owner | `pauli_value` in `backends/_native/nwqsim_runner.cpp` |
| NWQ-Sim known-array bytes per process | Inspected NWQ-Sim constructor and sampler allocations | [NWQ-Sim execution](nwqsim.md) | `backends.nwqsim._buffer_bytes` and the runner |
| State-body memory lower requirement `b * 2**q` (statevector) or `b * 4**q` (density matrix) | NWQLib, from the stored representation and precision | [Device and physical-model domains](profiles.md#device-and-physical-model-domains) | `backends.assessment._state_body_detail` |
| Linear time model `acquisition_linear/1` | NWQLib | [Finite time models](profiles.md#finite-time-models) | `backends.assessment._predict_time` |
| Estimator `stds` meaning with and without ZNE | IBM Estimator input and output guide, qiskit-ibm-runtime 0.49 | URL recorded in the uncertainty Source | `backends.ibm_runtime.IBMRuntimeBackend._estimate_result` |
