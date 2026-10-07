# Slurm

<a id="slurm-execution-and-site-handoff"></a>`NWQSimSlurmBackend` runs an [NWQ-Sim](nwqsim.md) runner as a Slurm batch job on an HPC cluster and collects its result. The run is asynchronous. `nwqlib.submit(prepared)` returns a saved run, and `run.resume()` and `run.wait()` continue the same job, also from a later process.

## Submit a job {#common-detached-execution}

You need a runner built for the profile's backend/method pair, as [Build the runner](nwqsim.md#building-a-selected-nwq-sim-runner) describes, on the shared path that the profile names, and `nwqlib[qiskit]` on the host that prepares the run.

Construct one backend with the site profile and explicit limits. This example is a template and is not run here, because it needs a Slurm site and a built runner:

```python
import nwqlib
from nwqlib.backends import NWQSimSlurmBackend, SlurmProfile
from nwqlib.execution import ExecutionLimits

backend = NWQSimSlurmBackend(
    profile=SlurmProfile(
        cluster="YOUR_CLUSTER", account="YOUR_ALLOCATION",
        walltime="00:05:00", nodes=1, ranks_per_node=1, threads=1,
        executable="/YOUR_SHARED_PATH/nwqlib-nwqsim",
        build_identity="YOUR_QUALIFIED_NWQSIM_SOURCE_COMMIT",
        spool="/YOUR_SHARED_PATH/nwqlib-spool",
    ),
    accounting_since="YYYY-MM-DDT00:00:00",
    accounting_until="YYYY-MM-DDT00:00:00",
    max_input_bytes=65536, max_output_bytes=1048576,
    max_buffer_bytes=1048576, max_response_bytes=4096, timeout_seconds=30,
)
prepared = nwqlib.prepare(plan, backend=backend, directory="slurm-run",
    limits=ExecutionLimits(max_total_circuits=1, max_total_shots=32))
with prepared.run as run:
    nwqlib.submit(prepared)
    print(run.exposure)  # circuits and shots submitted so far

# A later process reopens the same folder with the same backend.
with nwqlib.load_run("slurm-run", backend=backend) as run:
    run.resume()
```

Supply the `Plan`, finite `ExecutionLimits` and the runner's byte limits. `build_identity` must equal the NWQ-Sim source commit that the runner reports, and preparation checks that before it builds the circuit. The profile alone decides the backend/method pair, executable, build and shared result directory (`spool`). Keep the whole backend configuration to reopen the run.

`accounting_since` and `accounting_until` bound the one `sacct` query that finds the job by its name if the acknowledgement of the submission is lost. Choose site-local times, with the start before the end, that cover the period in which you will submit. A job whose ID was saved needs no lookup.

Preparation never calls the scheduler. Prepare on a host where the runner and its shared libraries are available. The batch script loads the profile's modules, and they do not change the environment of the process that prepares the job.

`SlurmProfile` describes the job and runs nothing, and it can be saved. Its fields hold the account, cluster, partition or QoS, walltime, nodes, ranks and threads, GPU binding, backend/method, modules, build identity, executable and shared result directory. Supply the verified executable and the site's module names. A profile is not evidence that its runner supports a device.

## Routes and optimization level

The profile's backend/method must match the runner's description.

- CPU SV and DM require one rank and no GPU allocation.
- MPI/SV allows a power-of-two total rank count with at least two local qubits per rank. Every rank takes part in one simulation, and rank zero alone writes the result.
- NVGPU and AMDGPU SV and DM require one process and exactly one GPU per task. Several processes sharing one GPU are rejected.
- MPI/DM and the blocked NVGPU_MPI route are rejected.
- The NWQ-Sim public factory at the inspected revisions `b35763d` and `efd0226` has no AMD GPU/MPI route, so `SlurmProfile` has no backend value for it.

The launcher also checks the distributed or GPU runner before `sbatch`. No rank writes to the run folder, and counts need no extra state gather.

The backend's own `optimization_level`, 0 by default, is passed to the NWQ-Sim backend, so the translation, the refusals and the preparation-record exclusion `optimization_level` are those of [NWQ-Sim](nwqsim.md#optimization-level).

## What the backend checks

The run saves the submission and counts its work before `sbatch`. The launcher's checks of the configuration and of the input size run before that, so a refusal there leaves no submission, event, submitted work or reserved output in the run. The run's checks of the output, payload and response bytes against its remaining data allowance still run after the submission is saved. A lost acknowledgement is resolved with one bounded `sacct` query for the original submission UUID, and the backend never queues a replacement. Decoding uses the original prepared snapshot, revision, readout and shots, and the returned observation's job is the numeric scheduler ID. Each successful result packet records one circuit evolution, independent of shots, ranks and allocated threads. Repeated resumes keep one observation and one job. If a scheduler command such as `sbatch` is missing, the original `FileNotFoundError` names the command and gives setup guidance, and no replacement submission or scheduler retry follows.

At launch, the file check counts one input write and one native read per rank, the script write and one shared `max_output_bytes` allowance for the native output. Every refresh checks the `max_output_bytes` allowance again before it reads a result file. `max_buffer_bytes` bounds the known arrays of each process, by the host and device formulas of each method in [Memory per process](nwqsim.md#memory-per-process). It excludes additional SDK workspace and whole-process memory. Plan the total allocation across ranks as well as the per-process limit.

For a successful Slurm trajectory of K Pauli or probability points without a forecast, `max_completion_metadata_bytes >= 32*K + 168` covers the point records and the growth of the completed-job record, because the launcher accepts only positive uint32 job IDs of at most ten ASCII digits. The metadata check of a single-endpoint probability readout, and the bytes credited back at completion for each probability point's header, also count the job ID at ten characters.

## Refresh and cancel

Each refresh makes one `squeue` call, makes one `sacct` call only when the job has left the queue, and reads the native output, within its byte limit, if it exists. Missing accounting stays unknown. Job arrays and partial or mismatched rows stay uncertain, because the launcher submits one job and not an array. Scheduler state, exit code and visibility of the raw result are separate fields. A completed allocation without output is not a scientific completion, and an output file is not proof of successful execution. NWQLib decodes and validates the result packet and its files, applies nonzero exit codes and saves each measurement once. A refresh reads the same job and never launches another one.

A malformed completed packet, or one from another job, raises an error and can still be refreshed, without counting an evolution. A native failure packet keeps the work it reports. A known nonzero exit value or signal prevents a successful result, and `provider_status` and the failure keep the original Slurm `ExitCode` pair. A valid native result can be saved while accounting is delayed, with the provider status still `unknown`. It establishes the native readout, not the later success of the allocation or its epilogue. A completed or cancelled allocation without a visible result packet can still be refreshed, and no execution count is invented.

Cancellation addresses the original cluster and job once, and only a later refresh confirms its outcome. It does not return the work already counted. An interrupted acknowledgement of the cancellation never causes a second cancellation.

## Perlmutter NVIDIA profile

Replace every capitalized placeholder with your own values. This profile selects a single-process NVIDIA runner and stays unqualified until the combination of site, toolchain and hardware has been checked.

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

The site recommends an explicit account, node constraint and QoS. Its GPU resource and binding options are documented in [running jobs on Perlmutter](https://docs.nersc.gov/systems/perlmutter/running-jobs/) and [process affinity](https://docs.nersc.gov/jobs/affinity/). CPU execution uses a CPU constraint and no GPU fields. For CPU/MPI, select an MPI/SV runner with `backend="MPI", method="SV"`, remove the GPU fields and choose a power-of-two `nodes * ranks_per_node`. q qubits allow at most `2**(q-2)` ranks. The CUDA/NVSHMEM route stays blocked by its upstream constructor defect, and no working installation or dependency correction is assumed.

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

The [Frontier user guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html) describes `srun` CPU and GPU placement and the site's allocation requirements. Check the account, partition and modules again before each new site run. HIP returns only counts through this runner, and exact readouts are rejected. CPU/MPI uses the MPI/SV runner and the CPU-only allocation described above. The current factory has no AMD GPU/MPI route, which is outside the documented qualification. These examples do not qualify GPU execution or close that gap.

## What has been checked

The offline checks use supplied native result packets and scheduler responses. No NERSC or OLCF job or GPU binding was tested. The two small local MPI checks described in [NWQ-Sim](nwqsim.md#qualified-nwq-sim-revisions) cover only that local scope and do not qualify Slurm or another platform. Live submission, building the runner and numerical qualification are separate from these offline checks.

## Check a new site with one small job {#bounded-later-live-check}

Before larger work on a new site, run one small job by hand:

1. Use a single-rank runner verified on the site, a small runner input that NWQLib has already checked, and finite input, output and buffer limits.
2. Read the generated batch script, then submit it once with `sbatch --parsable job.sh` from its shared result directory.
3. Save the job ID and cluster, refresh that job once, and compare its readout with a small reference whose answer you know independently.

The allocation's walltime bounds the scheduled execution, not the time in the queue. See [sbatch](https://slurm.schedmd.com/sbatch.html) and [sacct](https://slurm.schedmd.com/sacct.html) for the command-line behavior.

## Advanced: launcher and transport {#advanced-launcher-and-transport}

`NWQSimSlurmBackend` builds a `SlurmLauncher` from its profile and limits, and the run uses it to call the scheduler. A direct caller of these objects takes over the run's duties described at the end of this section.

`profile.script(submission_id, max_input_bytes=...)` renders a batch script for review without submitting it. The runner receives three arguments: the input JSON, the result JSON and the maximum input bytes. It writes the `nwqlib.nwqsim/3` result and its binary amplitude or probability files.

Construct `SlurmLauncher(profile, max_response_bytes=..., timeout_seconds=...)` with explicit finite limits. Its `SlurmTransport` runs the local command-line arguments without a shell, bounds the combined stdout and stderr, and kills and reaps the local process at the deadline or on overflow. Command errors keep their return code and bounded output. The transport removes inherited `SBATCH_*` and `SRUN_*` variables, because `SBATCH_*` values would override the reviewed script's `#SBATCH` directives and `SRUN_*` values would add `srun` options that the script leaves unset. Killing the local client does not undo a scheduler request that may have been accepted. Offline tests can supply `transport(argv, max_output_bytes=...)` instead.

`launch`, `refresh`, `reconcile` and `cancel` take the run as `run` and check their file writes and each scheduler response (`max_response_bytes`) through `run.check_data` before doing the work. `launcher.admit(payload, max_input_bytes=...)` reads no run and writes nothing. It checks `max_response_bytes >= 12 + len(profile.cluster.encode("ascii"))`, the rank and GPU request, the build identity of a non-CPU runner, the script rendering and the payload's input byte allowance, and `launch` runs it before any write to the result directory or scheduler call. The response minimum covers a ten-digit job ID, the separator, the known cluster and one LF in the canonical `sbatch --parsable` output. The normal response limit and the run's data check still apply to every command. Status and accounting responses can need a larger limit.

The locator's `instance` is the submission UUID, `job_id` is the Slurm job number, and `cluster` and `account` keep the scheduler routing. Save the profile with it so that a reopened run uses the same shared result directory. `launch` creates `<spool>/<submission UUID>` exactly once, writes `input.json` and `job.sh`, and calls `sbatch --parsable`. An acknowledgement failure leaves these files in place and raises the original error. Repeating a launch with that UUID fails before a second scheduler call. `reconcile` makes one `sacct` query with the original UUID job name and the saved time window. No match or several matches stay unresolved, and there is no automatic retry or resubmission. Slurm job matching is not an exactly-once service.

The launcher neither keeps a run log nor saves an observation. A direct caller must save the submission and its counted work before calling `launch`, then save the returned locator.
