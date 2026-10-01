# Slurm execution and site handoff

`NWQSimSlurmBackend` connects the selected native preparation and result decoder to `Run`, `prepare`, `submit_detached` and `refresh_submissions`. The route is asynchronous. Public `nwqlib.submit(prepared)` returns a durable Run, and `run.resume()` and `run.wait()` advance the same original work. If a scheduler command such as `sbatch` is unavailable, its original `FileNotFoundError` includes the command name and setup guidance. No replacement submission or scheduler retry follows that error. Offline checks use supplied native result packets and scheduler responses. No NERSC or OLCF job or GPU binding was tested. The two small local MPI checks described in [NWQ-Sim execution](nwqsim.md) establish only that local scope and do not qualify Slurm or another platform.

`SlurmProfile` is inert and serializable. Its fields keep account, cluster, partition/QoS, time, nodes/ranks/threads, GPU binding, backend/method, modules, build identity, executable and shared spool. Supply the actual verified executable build and site module names; a profile is not evidence that its binary supports a device. `profile.script(submission_id, max_input_bytes=...)` renders a reviewable batch script without submission. The runner receives the existing three arguments: input JSON, result JSON and maximum input bytes. It owns the `nwqlib.nwqsim/3` result and its binary amplitude or probability files.

Construct `SlurmLauncher(profile, max_response_bytes=..., timeout_seconds=...)` with explicit finite limits. Its `SlurmTransport` executes the local CLI argv without shell evaluation, bounds combined stdout/stderr, and kills/reaps the local process on deadline or overflow. CLI errors preserve their return code and bounded streams. It removes inherited `SBATCH_*` and `SRUN_*` variables, because `SBATCH_*` values would override the reviewed script's `#SBATCH` directives and `SRUN_*` values would add `srun` options that the script leaves unset. Killing the local client does not undo a possibly accepted scheduler request. Offline tests can inject `transport(argv, max_output_bytes=...)` instead. `launch`, `refresh`, `reconcile` and `cancel` take the common owner's Run as `run` and check their file writes and each scheduler response (`max_response_bytes`) through `run.check_data` before doing the work. `launcher.admit(payload, max_input_bytes=...)` reads no Run and writes nothing. It checks `max_response_bytes >= 12 + len(profile.cluster.encode("ascii"))`, the rank and GPU request, the build identity of a non-CPU runner, the script rendering and the payload's input byte allowance, and `launch` runs it before any spool write or scheduler call. The response minimum covers a ten-digit supported job ID, the separator, the known cluster and one LF in canonical sbatch parsable output. The normal response cap and Run data check still apply to every command. Status and accounting responses can require a larger cap. This primitive neither owns a journal nor publishes an observation. The common owner must persist submission intent and charged work before calling `launch`, then save the returned locator.

The locator's `instance` is the submission UUID, `job_id` is the Slurm job number, and `cluster`/`account` keep scheduler routing. Save the profile with it so resume uses the same shared spool. `launch` creates `<spool>/<submission UUID>` exactly once, writes `input.json` and `job.sh`, and calls `sbatch --parsable`. An acknowledgement failure leaves these files intact and propagates the original error. Repeating launch with that UUID fails before a second scheduler call. `reconcile` makes one `sacct` query with the original UUID job name and an explicit saved time window. No match or multiple matches remain unresolved; there is no automatic retry or resubmission. Slurm correlation is not an exactly-once service.

For a fresh successful Slurm trajectory of K Pauli or probability points without a forecast, `max_completion_metadata_bytes >= 32*K + 168` covers the point envelopes and detached outcome growth because the launcher accepts only positive uint32 job IDs of at most ten ASCII digits. The metadata check of a single-endpoint probability readout and the completion credit of each probability point's header also count the job ID at ten characters.

`refresh` makes one `squeue` call and, only when absent, one `sacct` call. Missing accounting remains unknown. Arrays/partial or mismatched rows remain uncertain; this launcher submits one job, not arrays. Scheduler state, exit code and raw result visibility are separate fields. A completed allocation without output is not scientific completion, nor is an output file proof of successful execution. The native result owner must decode and validate the packet and artifact, apply nonzero exits, and publish each measurement once through the common owner. A subsequent refresh reads the same job; it never launches or publishes work. Cancellation targets only the known cluster/job and still requires refresh to confirm its outcome. It does not refund charged work.

The profile's selected backend/method must match the actual binary description. CPU SV/DM requires one rank and no GPU allocation. MPI/SV permits a power-of-two total rank count with at least two local qubits per rank; every rank cooperates in one simulation and rank zero alone claims/publishes the result. NVGPU and AMDGPU SV/DM require one process and exactly one GPU per task. Multi-process single-GPU copies are rejected. MPI/DM and the blocked NVGPU_MPI route reject. The NWQ-Sim public factory at the inspected revisions `b35763d` and `efd0226` has no AMD GPU/MPI route, so `SlurmProfile` has no backend value for it. The lower-level launcher separately checks the actual distributed/GPU binary before `sbatch`. No rank writes a controller journal, and counts require no extra state gather.

## Common detached execution

Construct one immutable backend with the actual profile and explicit limits:

```python
from nwqlib.backends import NWQSimSlurmBackend, SlurmProfile
import nwqlib
from nwqlib.execution import ExecutionLimits

backend = NWQSimSlurmBackend(
    profile=SlurmProfile(
        cluster="YOUR_CLUSTER", account="YOUR_ALLOCATION", walltime="00:05:00",
        nodes=1, ranks_per_node=1, threads=1,
        executable="/YOUR_SHARED_PATH/nwqlib-nwqsim",
        build_identity="YOUR_QUALIFIED_NWQSIM_SOURCE_COMMIT",
        spool="/YOUR_SHARED_PATH/nwqlib-spool",
    ),
    accounting_since="2026-09-10T00:00:00", accounting_until="2026-09-11T00:00:00",
    max_input_bytes=65536, max_output_bytes=1048576, max_buffer_bytes=1048576,
    max_response_bytes=4096, timeout_seconds=30,
)
prepared = nwqlib.prepare(plan, backend=backend, directory="slurm-run",
    limits=ExecutionLimits(max_total_circuits=1, max_total_shots=32))
with prepared.run as run:
    nwqlib.submit(prepared)
    print(run.exposure)

# A later process uses the original same folder and exact backend configuration.
with nwqlib.load_run("slurm-run", backend=backend) as run:
    run.resume()

```

Supply the `Plan` (the selected construction and its costs, computed before any circuit exists), finite ExecutionLimits and native I/O limits. Choose site-local accounting dates covering the intended submission period; the dates above are examples. Preserve the entire backend configuration for resume. Its profile is the sole backend/method/executable/build/spool owner; `build_identity` must equal the actual runner's described NWQ-Sim source commit. Preparation checks that equality before lowering. It never calls the scheduler. Prepare from a host where the binary and its required shared libraries are available. Profile modules are loaded by the batch script; they do not modify the environment of the process preparing the job.

The backend's own `optimization_level`, 0 by default, is passed to the NWQ-Sim adapter, so the lowering, the refusals and the preparation-record exclusion `optimization_level` are those of [NWQ-Sim](nwqsim.md).

The common owner persists intent and charged work before `sbatch`. The backend's `admit_batch` runs the launcher's `admit` before that intent, so a configuration or input-allowance refusal leaves no submission, event, exposure or output reservation in the Run. The Run's data headroom checks of the output, payload and response bytes still run after the intent. A lost acknowledgement reconciles the original UUID with one bounded `sacct` query; known job IDs bypass that lookup. It never queues a replacement. Native decoding uses the original prepared snapshot, revision, readout and shot population, while the returned observation's job is the numeric scheduler ID. Each successful packet records one circuit evolution, independent of shots, ranks and allocated threads. Repeated resume/collection keeps one observation and one job.

The file check at launch counts one input write and one native read per rank, the script write and one shared `max_output_bytes` envelope for the native output. Every refresh checks the `max_output_bytes` envelope again before it reads a result file. `max_buffer_bytes` bounds the known arrays of each process, by the host and device formulas of each method in [NWQ-Sim execution](nwqsim.md). It excludes additional SDK workspace and whole-process RSS. Plan the aggregate allocation across ranks as well as the per-process limit.

Each refresh reads the scheduler once (plus accounting when absent from the queue) and the bounded native output if present. Malformed or foreign completed packets propagate an error and remain refreshable without an accepted evolution count. Native failure packets keep their reported population. A known nonzero exit value or signal prevents successful publication; `provider_status` and failure keep the original Slurm `ExitCode` pair. A valid native completion can publish while accounting is delayed, with the provider status still `unknown`. That publication establishes native readout, not later allocation or epilogue success. Completed or cancelled allocations without a visible native packet remain refreshable, with no fabricated execution count. Cancellation addresses the original cluster/job once. An interrupted acknowledgement never triggers an automatic second cancellation or a refund of charged work.

## Perlmutter NVIDIA profile

Replace all capitalized placeholders with user-selected values before use: this profile selects a single-process NVIDIA runner and remains unqualified until the actual site/toolchain/hardware combination has been checked.

```python
from nwqlib.backends.slurm import SlurmProfile

perlmutter = SlurmProfile(
    cluster="perlmutter", account="YOUR_ALLOCATION", qos="regular",
    constraint="gpu", walltime="00:05:00", nodes=1, ranks_per_node=1,
    threads=1, gpus_per_task=1, gpu_bind="closest", cpu_bind="cores",
    modules=("YOUR_VERIFIED_COMPILER_MODULE", "YOUR_VERIFIED_GPU_MODULE"),
    executable="/YOUR_SHARED_PATH/nwqlib-nwqsim",
    backend="NVGPU", method="SV",
    build_identity="YOUR_QUALIFIED_NWQSIM_SOURCE_COMMIT",
    spool="/YOUR_SHARED_PATH/nwqlib-spool",
)
```

The site recommends explicit account, node constraint and QoS. Its GPU resource and binding options are documented in [running jobs on Perlmutter](https://docs.nersc.gov/systems/perlmutter/running-jobs/) and [process affinity](https://docs.nersc.gov/jobs/affinity/). CPU execution uses a CPU constraint and no GPU fields. For CPU/MPI, select an MPI/SV binary with `backend="MPI", method="SV"`, remove GPU fields, and choose power-of-two `nodes * ranks_per_node`; q qubits require at most `2**(q-2)` ranks. The CUDA/NVSHMEM route remains blocked by its upstream constructor defect; no working installation or dependency correction is assumed.

## Frontier AMD HIP profile

```python
frontier = SlurmProfile(
    cluster="frontier", account="YOUR_ALLOCATION", partition="batch",
    walltime="00:05:00", nodes=1, ranks_per_node=1, threads=7,
    gpus_per_task=1, gpu_bind="closest", cpu_bind="cores",
    modules=("YOUR_VERIFIED_COMPILER_MODULE", "YOUR_VERIFIED_ROCM_MODULE"),
    executable="/YOUR_SHARED_PATH/nwqlib-nwqsim",
    backend="AMDGPU", method="SV",
    build_identity="YOUR_QUALIFIED_NWQSIM_SOURCE_COMMIT",
    spool="/YOUR_SHARED_PATH/nwqlib-spool",
)
```

The [Frontier user guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html) describes `srun` CPU/GPU placement and site allocation requirements. Recheck the selected account, partition and modules before a later site run. HIP exposes counts only through this runner; unsupported exact readouts reject. CPU/MPI uses the MPI/SV binary and CPU-only allocation described above. The current factory does not route AMD GPU/MPI. That route is outside the documented qualification. These examples do not authorize GPU execution or fix that gap.

## Bounded later live check

After explicit human authorization, use one accepted small existing runner input, a site-verified single-rank binary and finite input/output/buffer limits. Inspect the generated script, then submit it once with `sbatch --parsable job.sh` from its shared spool directory. Save job and cluster, refresh that exact job once, and compare its requested readout with an independently known small reference. The allocation walltime bounds scheduled execution, not queue delay. Live submission, building and numerical qualification are separate work from these offline checks. See [sbatch](https://slurm.schedmd.com/sbatch.html) and [sacct](https://slurm.schedmd.com/sacct.html) for the CLI contracts.
