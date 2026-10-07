# Choose a backend

A backend is where NWQLib runs the circuits of a `Plan` (the method and construction that `nwqlib.plan` fixes before any circuit exists): a local simulator, an HPC simulator or a cloud quantum computer. Pick one by task from the table below, build its connection object and pass it to `nwqlib.prepare` or `nwqlib.solve`.

## Choose by task

| Task | Backend and connection | What has been checked |
| --- | --- | --- |
| Small exact simulation on your machine | [Aer](aer.md), `AerBackend()`, the default. Install `nwqlib[aer]` | Runs locally. Exact simulation stores the full state, so memory limits its width. The noiseless exact simulator can also read several points of one evolution in one simulation ([trajectory readout](#trajectory-readout)): Pauli values, probabilities, and reductions whose registered functions supply both their work count and their execution. |
| Noisy counts | [Aer](aer.md) with a noise model, `AerBackend.from_noise_model(noise)`. Install `nwqlib[aer]` | A noise model applies to sampled counts only. |
| Larger simulation, on one machine or with MPI or a GPU | [NWQ-Sim](nwqsim.md), `NWQSimBackend`. Install `nwqlib[qiskit]` | CPU/SV runners built from a revision that [Qualified NWQ-Sim revisions](nwqsim.md#qualified-nwq-sim-revisions) lists as qualified or checked are supported, and a local run can be continued after the Python process exits. Each backend/method pair has its own build and readouts. CPU/SV also supports [trajectory readout](#trajectory-readout) of Pauli values, probabilities and registered reductions in one evolution. Its trajectory readout has been checked with a built runner on macOS arm64 only. |
| NWQ-Sim on an HPC cluster | [Slurm](slurm.md), `NWQSimSlurmBackend`. Install `nwqlib[qiskit]` | Checked offline against scheduler and site-profile responses. Site allocation, MPI and GPU execution need a [qualified build](nwqsim.md#qualified-nwq-sim-revisions) and a live check at the site. |
| IBM quantum hardware | [IBM Runtime](ibm.md), `IBMRuntimeBackend`. Install `nwqlib[ibm]` | Checked offline: SDK, transport and result decoding. Live credentials, queues and QPUs are not checked. |
| IonQ quantum hardware | [IonQ](ionq.md), `IonQBackend`. Install `nwqlib[ionq]` | Checked offline: circuit conversion and v0.4 histograms. No live QPU is checked. |
| Quantinuum H2 hardware | [Quantinuum Nexus](nexus.md), `NexusBackend`. Install `nwqlib[nexus]` | Checked offline: conversion, remote-job protocol and results. Its page describes a dependency exception and an HTTP wait without a timeout. |
| An OpenQASM file for another tool | [Export OpenQASM](qasm-streaming.md), `export_qasm` or `write_qasm3_file`. Install `nwqlib[qiskit]` for `export_qasm` | Writes text only. Nothing is executed. |

Two operations are not backends: [resource estimation](resources.md), which submits no job, and [fault-tolerant compilation and physical projection](fault-tolerant-resources.md) with NWQEC and QDK.

## Run on any backend

Every backend uses the same four steps: prepare the Plan for the backend, submit it, wait for the result, and reopen the saved run folder later if needed. This example runs on Aer and prints `1.0` twice, because the state $\lvert 0\rangle$ is an eigenstate of $Z$ with eigenvalue 1:

```python
import nwqlib
from nwqlib.algorithms import ExpectationMethod
from nwqlib.backends import AerBackend

problem = nwqlib.Expectation(state=[1., 0.],
                             observable=[[1., 0.], [0., -1.]])
plan = nwqlib.plan(problem, method=ExpectationMethod(), shots=128, seed=7)
backend = AerBackend()

prepared = nwqlib.prepare(plan, backend=backend, directory="my-run")
with prepared.run as run:
    nwqlib.submit(prepared)
    print(run.wait().value)

with nwqlib.load_run("my-run", backend=backend) as run:
    print(run.wait().value)
```

`prepare` builds the circuit of the first setting for the backend, or the circuits of every setting with `settings="all"` (local Aer, local NWQ-Sim and classical runs only), and saves them with the Plan in the new folder `directory`. `submit` sends the circuits and builds each later one when it reaches it. `run.wait()` returns the Result. `load_run` reopens the folder. Here the run has finished, so `wait()` returns the saved Result without measuring again.

For a provider, replace the connection and expect the job to stay pending. The template below is not run here. It needs the provider's credentials:

```python
backend = ...  # IBMRuntimeBackend, IonQBackend, NexusBackend or
               # NWQSimSlurmBackend, configured as on its page
prepared = nwqlib.prepare(plan, backend=backend, directory="provider-run")
with prepared.run as run:
    nwqlib.submit(prepared)
    print(run.exposure)  # circuits and shots submitted so far

# Later, in this or another process, with the same backend configuration:
with nwqlib.load_run("provider-run", backend=backend) as run:
    result = run.wait(timeout=3600)
    print(result.value)
```

For a provider, `submit` returns the run when the method next has to wait for a job, and `run.result` is `None` while the job is pending. `run.wait(timeout=...)` checks the provider until the Result is ready and raises `TimeoutError` when the timeout passes, leaving the run available for a later `wait()` or `resume()`. `run.resume()` refreshes each pending job once and returns. [Run on a backend](prepared_execution.md) covers the limits of a run, and [Continue an interrupted run](run_archives.md) covers saved folders.

## What every backend does

<a id="match-the-methods-readout"></a>

### Readouts

Each method declares the readout it needs: counts, Pauli statistics, probability marginals, a provider estimate, amplitudes or a host output. A backend name alone does not establish support for every instruction, control, width or readout. Each backend checks the request against its capabilities and rejects an unsupported combination before the affected work starts.

Host execution calls the method's numerical routine. Statevector execution simulates the circuit and returns the requested readout. Counts come from finite-shot execution. The method's guide gives its option names and supported modes.

### Exact synthesis of dense matrices

In the cases that the table under [Optimization level](#optimization-level) lists, a backend replaces each dense `UnitaryGate` by its exact synthesis before it translates the circuit to its gates, because Qiskit's own synthesis of such a gate is not exact to rounding ([dense unitaries](development/dense_synthesis.md#controlled-dense-unitaries)). A run synthesizes each distinct matrix once and reuses the circuit in its later preparations. Before each new synthesis it counts the work against `ExecutionLimits.max_synthesis_work`, a total over all preparations of the run. A reopened run synthesizes a matrix, and counts its work, again.

### Optimization level

`optimization_level` is the Qiskit transpiler level (on Nexus, the level of the Nexus compile job). On NWQ-Sim, IBM Runtime and IonQ, levels 2 and 3 resynthesize two-qubit blocks with Qiskit's own synthesis, so these levels receive the circuit as given, without exact synthesis. The preparation record is what NWQLib saves for each prepared circuit: how it was built, for which backend, and the error assessment of its readout. A level other than the backend's default is recorded as the exclusion `optimization_level` in the preparation record's `probability_window_exclusions`, which marks a circuit compiled at a non-default level, and the record's operation count describes the compiled circuit.

| Backend | Default level | Exact synthesis of dense matrices | Roundoff constant and `state_error()` |
| --- | --- | --- | --- |
| Aer | No transpiler level | Only with a noise model whose gate basis omits `UnitaryGate`, outside control-flow blocks. Otherwise Aer applies the matrix directly | Derived for qiskit-aer 0.17.2, see [Aer](aer.md#readout-and-population) |
| NWQ-Sim and Slurm | 0 | At levels 0 and 1 | Derived for CPU/SV. At a level other than 0, `state_error()` gives no bound, see [NWQ-Sim](nwqsim.md#optimization-level) |
| IBM Runtime | 1 | At levels 0 and 1, outside control-flow blocks | None, so `state_error()` gives no bound at any level |
| IonQ, QIS gates | 0 | At levels 0 and 1 | None, so `state_error()` gives no bound at any level |
| IonQ, native gates | 0, the only accepted level | No Qiskit translation | None, so `state_error()` gives no bound |
| Nexus | 1, of the Nexus compile job | Always, before the local translation to `u`/`cx` at level 0 | None, so `state_error()` gives no bound at any level |

### Trajectory readout

A trajectory reads several points of one circuit's evolution in one simulation, on Aer and on NWQ-Sim CPU/SV. A point is a position in the circuit at which NWQLib reads a Pauli expectation, a probability marginal or a reduction of the state, and on Aer also amplitudes. A point's readout can first apply a short circuit, its tail, such as a basis change, which is undone before the evolution continues.

Trajectory readout evaluates the declared points of one deterministic, noiseless coherent evolution. Probabilities and Pauli expectations are functions of the pure state at each point. The circuit up to the last point and the readout tails must not introduce measurement-conditioned evolution, resets, postselection or non-unitary channels. Both backends refuse, before building their circuit, a tail or the part of the circuit up to the last point that contains a measurement, a reset, control flow or an opaque instruction without a definition, also inside the definition of a composite instruction such as `Initialize`. Operations after the last point are neither run nor checked. A reduction uses a registered reducer, which must support the state representation and the backend.

<a id="continue-and-cancel"></a>

### Continue, cancel and recover {#provider-rules}

A saved run continues with its original `Run.resume()` and `Run.wait()`, which read the original job. A failed retrieval keeps the job's locator and does not cause a replacement submission. `run.cancel()` stops new work in the run, and the next refresh requests cancellation of the remote job. IBM Runtime, IonQ, Nexus and Slurm runs warn once per run that the backend has been checked only offline. A run reopened after it has saved its state does not repeat the warning. A local detached NWQ-Sim run names its host and result directory before launch, and that computer or VM must keep running after Python exits.

Interrupted submissions follow these rules on every provider:

- NWQLib sends each job request once and never resubmits it, also after an error or a lost acknowledgement. The IBM SDK's own safe retries of single HTTP requests are described on the [IBM Runtime](ibm.md#retries-and-cancellation) page.
- If the provider's acknowledgement (the job ID) is lost, the submission stays uncertain and its circuits and shots stay counted against the run's limits. The next refresh looks for the job by the label written at submission, listed below. No match or several matches leave the submission uncertain.
- A refresh reads the original job, saves each completed item once and keeps earlier items when a later item fails.
- Cancellation is one request to the original job. Only later provider status confirms it, and the submitted work stays counted.

| Backend | Label that identifies a submission |
| --- | --- |
| IBM Runtime | Job tag `nwqlib:<submission>` |
| Quantinuum Nexus | Job name `nwqlib:execute:<submission>`, with its program and shot description |
| Slurm | Job name `nwqlib-<submission>` |
| NWQ-Sim, local | Result directory `<spool>/<submission>`, inside the `spool` directory of `NWQSimBackend` |
| IonQ | None. No exact job lookup has been checked for the IonQ v0.4 API, so a lost IonQ acknowledgement stays uncertain |

The rules that each backend connection implements are listed in [Backend adapter contract](development/execution.md#backend-adapter-contract).

## Estimate and inspect resources

`nwqlib.estimate(plan, context=...)` adds up the resource formulas of the planned construction without building a circuit. `prepared.inspect_resources(max_operations=100000, max_bytes=10_000_000_000)` counts the operations of a circuit that is already prepared. Nonempty `transpile_options` request an additional compilation for that inspection. Formulas, representative samples and counts of prepared circuits describe different objects and are not interchangeable hardware costs. See [Resource estimates](resources.md).

## Export circuits

`export_qasm` writes a built Qiskit circuit as OpenQASM 2 or 3 text, for use with NWQ-Sim outside NWQLib or with another simulator, and NWQEC compilation reads its OpenQASM 2 output. This is separate from the direct writer `write_qasm3_file`, whose subset makes no NWQ-Sim compatibility claim. See [Export OpenQASM](qasm-streaming.md).

## Source map

| Step | Source | Location | Code |
| --- | --- | --- | --- |
| Dense-unitary CX count `(23/48) 4**n - (3/2) 2**n + 4/3` | Shende, Bullock and Markov, [arXiv:quant-ph/0406176v5](https://arxiv.org/abs/quant-ph/0406176v5) | Eq. (19) in Appendix A (printed page 16), and Table 1 (printed page 14), row "QSD (l = 2, optimized)", which lists 0, 3, 20 and 100 CX for n = 1 to 4. The integer formula reproduces the entries for n >= 2, and the code returns the n = 1 entry 0 as an explicit base case | `backends.resources.dense_unitary_cx_qsd_upper_bound` |
| Banded block-encoding CX per query | NWQLib count of one QFT/inverse-QFT pair plus the recorded multiplexor, diagonal and PREP counts | comment in the code | `backends.resources.block_encoding_per_query_cx` |
| Exact Pauli expectation read directly from native CPU/SV amplitudes, without a dense Pauli matrix | NWQLib derivation from the action of a Pauli string on a basis state, with `Y = iXZ` | comment in the code | `pauli_value` in `backends/_native/nwqsim_runner.cpp` |
| NWQ-Sim known-array bytes per process | Inspected NWQ-Sim constructor and sampler allocations | [NWQ-Sim](nwqsim.md#memory-per-process) | `backends.nwqsim._buffer_bytes` and the runner |
| Estimator `stds` meaning with and without ZNE | IBM Estimator input and output guide, qiskit-ibm-runtime 0.49 | URL recorded in the uncertainty Source | `backends.ibm_runtime.IBMRuntimeBackend._estimate_result` |

The surface-code patch, syndrome-cycle, logical-error-rate and magic-state factory models are listed under [Models and sources](fault-tolerant-resources.md#models-and-sources). The state-memory and linear time models are listed under [Finite time models](profiles.md#finite-time-models).
