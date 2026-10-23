# NWQ-Sim

<a id="nwq-sim-execution"></a>NWQ-Sim simulates NWQLib's circuits on CPUs, with MPI or on one GPU, through a runner, an executable that you build for one backend/method pair of NWQ-Sim: its backend (CPU, MPI, NVGPU or AMDGPU) and its simulation method (SV for statevector, DM for density matrix). Build the pair you intend to use, then pass the connection to `prepare` as in [Run on any backend](backends.md#run-on-any-backend). Preparation translates circuits with Qiskit, so install `nwqlib[qiskit]` on the host that prepares the run.

Constructing the connection does not launch or inspect the runner:

```python
from nwqlib.backends import NWQSimBackend

backend = NWQSimBackend(backend="CPU", method="SV", ranks=1,
    executable="/path/to/nwqlib-nwqsim", spool="/path/to/results",
    max_input_bytes=1_000_000, max_output_bytes=1_000_000,
    max_buffer_bytes=10_000_000)
```

Replace both paths with absolute paths on the execution host. The executable must be a [qualified build](#qualified-nwq-sim-revisions) of the chosen pair, and `spool` is the directory that holds its result files. The example byte limits are explicit limits, not a measured capacity or an allocation that suits every problem. Pass this connection to `prepare(plan, backend=backend, directory=...)` only when you intend to run that simulation on the execution host. This page's examples are templates and are not run here, because they need a built runner.

## Build the runner {#building-a-selected-nwq-sim-runner}

The build command compiles one public backend/method pair from the headers of a clean NWQ-Sim checkout. It uses the compiler already installed for the target: `c++` for CPU, `mpic++` for MPI, `nvcc` for NVIDIA or `hipcc` for AMD HIP. It installs and downloads nothing and builds no separate VQE or front-end targets:

```bash
python -m nwqlib.backends.nwqsim /path/to/NWQ-Sim /path/to/new/nwqlib-nwqsim \
    --compiler c++
/path/to/new/nwqlib-nwqsim --describe
# Each route needs its own new executable and installed toolchain.
python -m nwqlib.backends.nwqsim /path/to/NWQ-Sim /path/to/new/nwqlib-mpi \
    --backend MPI --method SV
python -m nwqlib.backends.nwqsim /path/to/NWQ-Sim /path/to/new/nwqlib-cuda \
    --backend NVGPU --method SV
python -m nwqlib.backends.nwqsim /path/to/NWQ-Sim /path/to/new/nwqlib-hip \
    --backend AMDGPU --method SV
```

The output path must not exist yet. `--describe` reports the runner's NWQ-Sim revision, compiler, executable architecture, backend/method pair, collective support and readouts. Preparation checks that description against the `NWQSimBackend(backend=..., method=..., ranks=..., ...)` configuration, which cannot change after construction. A cross-compiled executable can have a different architecture from the Python host. The build command compiles with `-O3` and no fast-math option. A runner compiled with fast-math arithmetic is outside the roundoff derivation of exact readouts (probabilities, Pauli expectations or amplitudes computed from the simulated state with `shots=None`, without shot sampling and subject to numerical rounding), although its description does not report compiler options. The command records the exact NWQ-Sim revision and leaves the checkout unchanged. It does not establish qualification, and a direct caller of the runner's command line is responsible for choosing a qualified revision.

## Routes and readouts

| Backend/method | Readouts | Execution |
| --- | --- | --- |
| CPU/SV | Counts, Pauli expectations, probabilities, amplitudes, trajectories | One process, with direct access to the state |
| CPU/DM | Counts | One process. The decoding of statevector results does not apply to density matrices |
| MPI/SV | Counts | Cooperating ranks, a power of two, with at least two local qubits each |
| NVGPU/SV or DM | Counts | One process and one visible NVIDIA GPU |
| AMDGPU/SV or DM | Counts | One process and one visible AMD GPU through HIP |
| MPI/DM | Rejected | NWQ-Sim's state-creation function would silently substitute SV |
| NVGPU_MPI | Blocked | The inspected SV and DM constructors increment the uninitialized `gpu_mem` at b35763d and at efd0226 |
| AMD GPU/MPI | No route | NWQ-Sim's public state-creation function has no AMD GPU/MPI combination |

The runner accepts Qiskit-convention U and CX gates, and preparation translates the circuit to that basis with Qiskit at the chosen [optimization level](#optimization-level), after the [exact synthesis of dense matrices](backends.md#exact-synthesis-of-dense-matrices) at levels 0 and 1. Global phase is kept for amplitude output. Counts use NWQ-Sim's own sampling, while exact CPU readouts read the simulated state directly. Unsupported dynamic operations are not simulated by rerunning the circuit per shot. The per-gate print-only `PURITY_CHECK` diagnostic is disabled. The byte limits on input, output and arrays of known size (`max_input_bytes`, `max_output_bytes` and `max_buffer_bytes`) do not bound the whole process memory or additional SDK workspace. The file formats of the runner are described in [Backend adapter contract](development/execution.md#backend-adapter-contract).

### Exact readouts and roundoff

An exact CPU/SV readout is checked against the probability window of its preparation record, the record that NWQLib saves for each prepared circuit with how it was built and the error assessment of its readout. The probability window is the roundoff window of exact execution. It bounds to first order how far binary64 roundoff can move the total probability from one, with a per-instruction constant derived for NWQ-Sim CPU/SV ([derivation](ENGINEERING_CONSTANTS.md#numerical-guards-and-tolerances)). The derivation covers the [qualified revisions](#qualified-nwq-sim-revisions) and assumes IEEE binary64 arithmetic. The preparation record of a runner outside that coverage lists "unchecked NWQ-Sim revision" in `probability_window_exclusions` when its revision does not descend from efd0226 or its description lacks the roundoff fields, and "changed NWQ-Sim roundoff sources" when it descends from efd0226 with a changed source file.

An efd0226 runner built with Apple clang 21 on macOS arm64 ran the default QCELS (quantum complex exponential least-squares) schedule of the eight-qubit H4 problem in the eigenvalue notebook (`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`) on nine-qubit circuits containing up to 3.25 million U and CX gates sent to the runner. The probability totals differed from one by at most 5.5e-12, and no difference exceeded 1.6e-4 of the probability window of its circuit. The Qiskit version that translated these circuits to U and CX was not recorded.

Before execution, preparation asks the runner for the binary64 phase factor of its output and sends the real and imaginary parts of that factor with the request. The runner multiplies by that factor the amplitudes read at the end of the circuit and the states saved at the final point for a reduction, a statistic that a registered function (its reducer) computes from the saved state. The preparation record of these outputs includes an upper bound on the roundoff of the multiplication by the finite factor and on the factor's deviation from unit modulus, which `saved_state_error` and `saved_state_probability_window` carry into every use of those arrays. This bound assumes round-to-nearest binary64 arithmetic, gradual underflow, and finite intermediate products and sums. The resulting state error estimate stays conditional on the [qualified first-order model of the runner](ENGINEERING_CONSTANTS.md#probability-windows-of-exact-execution) and measures distance up to one common phase. Asking for the factor creates no simulator state.

### Trajectories on CPU/SV

A CPU/SV [trajectory](backends.md#trajectory-readout) reads all its points while advancing one simulator state. Each point lies before its readout tail. Saving a Pauli expectation leaves the state unchanged. A probability readout applies its tail, saves the marginal and, when a later point follows, applies the exact inverse of the tail before the evolution continues. The runner evolves the state between consecutive points without resetting or renormalizing it, and nothing after the last point runs. The preparation record counts the U and CX gates the runner executes through the final point, including each readout tail and every inverse applied before a later point, and the run checks the summed outputs of all points against its limits before the trajectory runs.

An intermediate output that depends on the global phase requires the phase accumulated up to its point, so the translation to U and CX must account for that point and its phase.

Scalar reducers declared invariant under one common input phase can use the raw saved amplitudes. Relative branch phases remain part of the coherent preparation. Qiskit reports one global phase for the whole translated prefix and no phase per segment. A reduction before the last point therefore requires a reducer registered as phase invariant, and a state saved at the last point is multiplied, in its output copy, by the phase of the translated prefix.

Amplitude points are refused, because the result that NWQLib reads from the runner process, which keeps running after Python exits, contains no arrays for intermediate points. The backend refuses them before the preparation is counted against the run's limits, and at that stage it also refuses a reduction before the last point when both positions are explicit. When the last point's position is left as `None`, which means the end of the circuit, that reduction is refused after translation. Like any other NWQ-Sim job, a trajectory runs in a runner process that continues after Python exits, with one evolution per execution.

For a trajectory, the CPU/SV roundoff model counts the U/CX gates executed through each point, including earlier reversible readout tails. Fusion within a segment performs no more matrix folds and applications than the model counts. The model remains first order and subject to the record's exclusions.

A trajectory needs room in `ExecutionLimits.max_completion_metadata_bytes` for 58 bytes per point and up to 168 bytes for the record of its completed job, or 99 bytes when a forecast has already revised that record. A single-endpoint probability readout, locally or through Slurm, needs the same 168 or 99 bytes. A readout that does not fit is refused before the runner starts.

### Counts on MPI and GPU

Every counts route uses the outcomes that NWQ-Sim's public `measure_all` returns. MPI's collective operation returns the same global set of shots on each rank, and rank zero alone forms and saves the histogram. MPI local arrays and GPU device pointers are never read into a full state in host memory. Counts need no added state gather, state copy, per-term basis fallback or second circuit evolution. The GPU sampler calls its measurement-only kernel internally, and the recorded evolution still names the original circuit.

Exact readouts outside CPU/SV are rejected before preparation work. In particular, upstream AMD SV's public Z expectation covers only a finite thread prefix, and AMD DM's public Z expectation throws, so neither is used as a substitute.

GPU builds enable the dependency's `GPU_ERROR_CHECK` option, and their description must confirm it. CUDA or HIP allocation, copy and kernel errors then stop the runner process before a completed histogram can be saved. The dependency exits instead of throwing a C++ exception, so missing output stays uncertain after the process is lost and does not cause an automatic replacement job. On the successful counts route this option adds three synchronization API calls and three last-error reads, one call and one read after each of construction, the simulation and the measurement-only kernel. Their latency is unmeasured. No per-gate check, state copy or evolution is added.

### MPI launch

Local MPI launch uses the configured `mpi_launcher` (default `mpirun`) with `-n ranks`. MPI is initialized before NWQ-Sim's public state-creation function is called, all ranks take part, and the state is destroyed before MPI finalizes. A rank-local error aborts the communicator, and no rank is retried on its own. The execution host or VM must keep running until completion. The run's notice names that host and the result directory. A missing result stays uncertain. MPI shot counts must fit a signed 32-bit count. A U/CX gate involving a global wire also requires `2**num_qubits / ranks <= 2**31 - 1`, because the current kernel sends the entire local partition with that count type. The runner rejects a transfer it cannot represent before allocating the state. Local-only gates, global measured bits and global phase do not need that state transfer.

## Memory per process

Known state, work and sample arrays are checked against `max_buffer_bytes` **per process**. For q qubits, S shots and R MPI ranks, the inspected allocations are:

| Route | Resident state/work arrays per process | Sample arrays per process |
| --- | --- | --- |
| CPU/SV | `24 * 2**q + 8` bytes | `8 * S` bytes |
| CPU/DM | `16 * 4**q + 8 * (2**q + 1)` bytes | `8 * S` bytes |
| MPI/SV | `32 * 2**q / R` bytes | `24 * S` bytes |
| GPU/SV | `48 * 2**q + 16` bytes, host plus device | `32 * S` bytes, host plus device |
| GPU/DM | `48 * 4**q + 16` bytes, host plus device | `32 * S` bytes, host plus device |

CPU probability readout also uses `16 * 2**k` bytes for its k-bit marginal, and its binary result file holds `8 * 2**k` bytes, which count against the output limit. The runner releases the parsed gate array once NWQ-Sim's circuit object holds the gates, before it allocates the state. Fused gates, the parsed input's other fields, copies of that circuit object, runtime allocations, MPI communication buffers and extra SDK workspace are outside these arrays, and their size is unknown. Each rank holds its own input and circuit, and the file check counts one input write plus R reads. Totals of MPI arrays sum over the R processes, while shots and evolutions are counted once. NWQ-Sim's MPI sampler scans global indices and compares each local state's probability interval with S samples. These costs come from the source and are not a measured scaling guarantee.

For a 20-qubit example, the known SV state and work arrays are about 24 MiB on CPU, 8 MiB per rank with four MPI ranks, or 48 MiB across host and device on one GPU, before samples and the excluded workspace. Twenty-qubit DM instead requires about 16 TiB on CPU or 48 TiB on GPU. These numbers are evaluated from the formulas above. They are not allocations, and they do not qualify the workload. A small passing simulation does not establish that capacity or runtime.

## Qualified NWQ-Sim revisions

| NWQ-Sim revision | Status | Reason |
| --- | --- | --- |
| `efd02262ff9c5def4f2eac1df6416c1ee2023d5b` (efd0226), and a descendant whose seven roundoff source files below are unchanged | Qualified for CPU/SV | It initializes the CPU memory counters and composes finite U phases without overflowing their sum. Its CPU/SV runner, built with g++ 14.2.0 on Linux aarch64, passed NWQLib's CPU/SV runner tests and an exact Pauli expectation through the public route |
| `202f5cab03477c78ab9f0b4dbbbf8f135778ac8f` | Checked for CPU/SV | Its `include/` tree equals that of efd0226. Its CPU/SV runner passed the same checks with g++ 14.2.0 on Linux aarch64, and the preparation record of its exact Pauli expectation listed no exclusion |
| `b35763d846e6512ed817d3f88ac8ce79a7e82a7e` (b35763d) | Rejected | Its CPU constructors have undefined behavior, because each increments an uninitialized memory counter. Its U matrix factory can also overflow the sum of finite phases. Python preparation rejects it before translation or execution |

The efd0226 rule qualifies CPU/SV builds and no other backend or method. Any other revision needs its own review and numerical checks, and a revision string that NWQLib accepts is not by itself qualified. Restrictions on circuit angles cannot repair the constructor defect of b35763d. Revision efd0226 also stores the CU gate's gamma phase, which the NWQLib runner does not use because it sends only U and CX gates. NWQLib neither distributes nor applies a patch to NWQ-Sim.

The roundoff derivation of exact readouts rests on these seven files of efd0226:

- the fusion pass, `include/circuit_pass/fusion.hpp`
- the CPU/SV kernels, `include/svsim/sv_cpu.hpp`
- the U and CX matrices, `include/private/gate_factory/sv_gate.hpp`
- the binary64 value type (`ValType = double`) and the fusion switch (`ENABLE_FUSION`), `include/config.hpp`
- the binary64 value type, `include/nwq_util.hpp`
- the gate records that the circuit builds and stores, `include/circuit.hpp` and `include/private/sim_gate.hpp`

The file condition catches a later upstream change to any of these assumptions, which the revision string alone would not show. The build command decides both conditions from the NWQ-Sim checkout, and the runner's `--describe` output reports the outcome as `roundoff_base_revision` (efd0226 when the revision descends from it, otherwise null) and `roundoff_sources_identical`.

### MPI and GPU checks

The local MPI evidence consists of two four-qubit, two-rank, 32-shot checks with an x86_64 executable translated on an arm64 macOS host. They cover a public parity workflow run in a separate runner process and cross-rank Bell sampling with an unequal classical-bit map. The GPU/DM routes have been checked only by source inspection and offline tests of their connection to NWQLib. No GPU toolchain, hardware or live site was qualified by those checks. A qualified runner and its low-level checks do not by themselves establish a complete scientific workflow or another backend/method combination.

## Optimization level

`optimization_level` selects the Qiskit transpiler level of the translation to U and CX, 0 by default. The rules shared with other backends are in [Optimization level](backends.md#optimization-level). Without a coupling map, levels 2 and 3 can also remove SWAP and permutation gates and relabel the wires after them. Counts follow the relabeled measurements, and their preparation record keeps the final layout. The backend refuses an exact readout or a trajectory whose translation relabeled wires, because the runner reads the state in logical wire order. At levels 2 and 3, a measured circuit can supply computational-basis probabilities when the translation reports no layout. Amplitude readouts require level 0 or 1, or a circuit without measurements, because removing terminal diagonal gates can change relative phases. A level other than 0 adds the exclusion `optimization_level`, so `state_error()` gives no bound, and each method that uses that bound takes the fallback its guide documents. The error numbers in the record, such as the tolerance of its probability check, are reference values, and its count of U and CX gates describes the compiled circuit.

## Cancel and clean up

For local execution, `run.cancel()` records the cancellation request and refuses new preparation or measurement in that run. `NWQSimBackend` cannot stop a process, so a local NWQ-Sim process that has already started can keep running until it finishes. The run adds a notice to `run.warnings` that the original job can still be checked, and a later `run.resume()` of the same run can still read the process's result file if one is written.

If a local launch loses its acknowledgement, automatic waiting requires either the original child process, still running and visible to NWQLib, or a final result file. A result directory alone does not show that the process started. Without either, `resume()` and `wait()` raise `RunFailed` with status `uncertain`, and the run keeps the original submission ID and `run.exposure`. A later explicit `resume()` can still collect output that arrives at the same path, without launching a replacement and without returning the counted circuits and shots to the run's limits.

Before execution, the runner marks its output directory as taken by an attempt, and it writes files without overwriting existing results. A process crash can leave this mark, a `.claim` directory. Refresh the run with `run.resume()`, which looks up that attempt, before deciding whether another execution is needed. A normal failure removes the attempt's partial files and keeps a size-limited failure result when the directory is writable. If execution started and no final result can be written, the `.claim` directory stays for that lookup.

Result directories are named by the original submission and are separate from the run folder. They can accumulate across runs, and the run's data limit does not cap their total size. NWQLib does not delete them. Delete one only when its job is confirmed finished, all its observations are saved in the run and no retrieval or lookup still needs it. Keep the data of pending, uncertain or incomplete jobs, of cancellations not yet confirmed and of paths that this run did not create.

## Run on a Slurm cluster

`NWQSimSlurmBackend` uses the same preparation and result decoding with Slurm. See [Slurm](slurm.md) for its site profile, accounting window, reopening rules and GPU/MPI limits.
