# Continue an interrupted run

<a id="save-results-and-continue-runs"></a>Reopen a saved Run folder to collect provider jobs that are still pending, to continue after an interruption or to get back a completed run's Result. To save, load and reanalyze a completed Result, see [Save, load and reanalyze results](saved_evidence.md).

A Result folder supports inspection and the reanalysis a Method supports. A Run folder supports continuing the same execution, including pending jobs. Both save the original `Plan` (the construction and its costs, computed by `plan` before any circuit exists) through its Method's archive hooks.

| Next action | Save | Open |
| --- | --- | --- |
| Inspect or reanalyze stored scientific output | `result.save(path)` | `nwqlib.load_result(path, method=None)` |
| Continue the same execution | `run.save(path)` or the live `run.directory` | `nwqlib.load_run(path, backend=connection, method=None)` |

Save destinations must be new directories whose parents already exist. A Run created with `prepare(..., directory=...)` creates its working directory and missing parents itself.

## Save a completed local run

```python
import nwqlib
from nwqlib.algorithms import ExpectationMethod
from nwqlib.backends import AerBackend

problem = nwqlib.Expectation(state=[1., 0.], observable=[[1., 0.], [0., -1.]])
selected = nwqlib.plan(problem, method=ExpectationMethod(), shots=64, seed=7)
prepared = nwqlib.prepare(selected)
with prepared.run as run:
    nwqlib.submit(prepared)
    result = run.wait()
    result.save("expectation-result")
    run.save("expectation-run")

restored = nwqlib.load_result("expectation-result")
print(restored.value)
with nwqlib.load_run("expectation-run", backend=AerBackend()) as completed:
    same_result = completed.wait()
    print(same_result.value)
```

```text
1.0
1.0
```

The state `(1, 0)` is an eigenstate of Z with eigenvalue 1, so every shot gives +1 and the estimate is exactly 1. Opening the completed Run restores its saved Result, and `wait()` returns that Result without another measurement or analysis. `load_result` needs no backend and cannot continue a pending provider job ([Save, load and reanalyze results](saved_evidence.md)).

An in-memory local Run can save a copy even though no folder was chosen at preparation. The copy gets its own run log, and the live Run keeps its existing storage. Call `run.save` while the Run is open, at a point where `submit`, `resume` or `wait` has returned. Closing releases private execution caches and the Run's lock, and the Result and its immutable scientific data remain available.

## Save and reopen pending work

For a remote backend, choose the backend and a folder before preparation. The following IBM template requires the `ibm` extra and an account authorized for the named device and instance. It uses the `selected` counts `Plan` above.

```python
from nwqlib.backends import IBMRuntimeBackend

backend = IBMRuntimeBackend(
    device="YOUR-IBM-DEVICE", instance="YOUR-IBM-INSTANCE",
    max_input_bytes=65_536,
)
prepared = nwqlib.prepare(selected, backend=backend, directory="provider-run")
with prepared.run as run:
    nwqlib.submit(prepared)
    snapshot = run.save("provider-snapshot")
    print(run.exposure)
```

`max_input_bytes` limits the serialized QPY bytes of each prepared circuit that this saved Run stores before submission. Choose a limit that fits the prepared circuits. Constructing the backend reads no credentials and does not contact the service.

Preparation can itself wait for remote compilation. `submit` returns the Run when the Method reaches its next stopping point. `run.result` is `None` while scientific output is pending. `run.save(path)` copies the Run's current state at that point into a new folder, including the original preparation IDs, provider job locators, completed observations and the Method's iteration state. Remote execution chooses a folder automatically if `directory` is omitted. Read its path from `run.directory`.

After closing the original Run, reopen either its working folder or the saved copy with an instance of the same backend configuration:

```python
with nwqlib.load_run(snapshot, backend=backend) as continued:
    continued.resume()
    if continued.result is None:
        print(continued.exposure)
    else:
        continued.result.save("provider-result")
```

`load_run` requires the matching backend configuration. It restores the saved `Plan`, the random-number generator positions, the Method's iteration state, the prepared backend circuits, existing numerical caches, completed observations and pending submissions. It performs no planning, analysis, circuit building or numerical reference reconstruction, and it does not contact the provider.

`load_run` opens the folder's existing writable run log (`run.sqlite`) and takes the Run's exclusive lock. It does not copy the run log to a new working location, and recovery can record an interrupted attempt's status. Continue only one copy of a run, the original or a snapshot, since their provider job IDs identify the same work. Continue the original working folder if the saved snapshot must stay unchanged. Opening a snapshot makes that directory the new working state.

`run.resume()` retrieves outstanding original work and advances the Method. Known jobs and uncertain submission attempts do not authorize a replacement submission. Retrieval errors keep the original locator and propagate to the caller. `run.wait(timeout=..., poll_interval=...)` repeatedly resumes until a Result is available or its timeout is reached. A timeout leaves the same Run available for later continuation.

## Interrupted attempts

An interrupted local attempt, such as a host kernel or a local Aer call without a recoverable backend job, has no external job to retrieve. If the Method cannot return a valid Result from the data already collected, both `resume()` and `wait()` raise `RunFailed` with `stage="recovery"`, `status="uncertain"` and the original attempt's identifier in `attempt`. The exception includes the attempt record in `trace`, the saved Run directory and the Run's exposure, and it keeps the recorded failure text, the observations and the counted work. The attempt stays `uncertain`, and its job, circuit and shot counts stay counted against the Run's limits. The error does not authorize resubmission or promise that the same local attempt can continue.

When an uncancelled Run reopens an interrupted synchronous Aer attempt with no saved job locator, its in-process result is unavailable. The recorded failure text is then the one written on reopening, which states that the in-process result is unavailable and does not assert that the underlying execution failed.

Inspect the saved observations and counted work, then close or cancel the Run. To execute the `Plan` again, start a new Run. Nothing is resubmitted in the original Run. A remote job with a job locator, or a supported way to recover that locator, keeps its pending behavior and can still be retrieved. NWQ-Sim can recover a lost launch acknowledgement from the original submission UUID even when no job locator was saved. A Method that can finish from the data already collected may return its Result before this error is raised.

## Methods that iterate between rounds

RWPE, Lanczos sensitivity sampling and ADAPT continue their own iteration on this same Run. There is no separate public driver to select. Ordinary `resume` does not retry an interrupted classical analysis. Where the Method supports explicit recovery, `run.resume(reanalyze=True)` retries that saved analysis step. This can change a later optimizer trajectory and is separate from retrieving a pending quantum job.

## QHD constrained and refinement runs

Each unrefined QHD augmented-Lagrangian round and each box-refinement level executes its `Plan` in its own Run. Automatic inequality selection can plan additional candidates before that execution. The constrained run saves the round's resolved inequality representation before its first inner Run and reuses it on resume. With `directory=...` both layers create these Runs on disk under one directory and keep an outer record of the completed rounds and levels. `resume_augmented_lagrangian` and `resume_box_refinement` continue an unfinished round's or level's Run with `load_run` under the rules above and never give it a second Run ([QHD guide](algorithms/qhd.md#constrained-problems)).

Resume reopens the recorded inner Run with its recorded `Plan` and run log. Completed work stays in the outer record and is counted once. A recoverable Run can continue, including a remote result that becomes available after interruption. Some interrupted local preparation, measurement or classical evolution cannot be completed from the saved Run.

An unfinishable Run normally raises with a recovery note. With `end_at_unfinishable=True`, a handled unfinishable case after a completed round or level can return `termination="inner_failed"` with the completed results and counted work. If no round or level has completed, the error still propagates. A folder without a committed run-log header fails during reopen, including with `end_at_unfinishable=True`. For such a folder, follow the error's instruction about removing that named inner folder before retrying. Resume does not remove or recreate it automatically. When an existing run log cannot be read, keep the folder and retry when it is readable. Its resource counts can be unknown.

A saved result archive and a run's working directory serve different purposes:

| You have | Use |
| --- | --- |
| A saved constrained-result archive | `load_augmented_lagrangian` |
| The working directory of an unfinished constrained run | `resume_augmented_lagrangian`, then save its returned constrained result to an archive if needed |
| A saved standalone refinement archive | `load_box_refinement` |
| The working directory of an unfinished refinement run | `resume_box_refinement`, then call `save` on its returned result to create a separate result archive |

## What stays associated

The Run folder keeps the Problem, Method, output, reconstruction and randomness together with the preparation records, submission attempts, observations, used limits and the Method's iteration decisions. A row chosen with `comparison.select` ([compare methods](scientist.md)) also keeps its original `PlanEstimate` and `Allocation`. Continuation does not reevaluate the profile, replace shots or configure the backend from an Allocation. Changed runtime or compiler evidence remains distinct from the original forecast, as described in [profiles and telemetry](profiles.md#original-forecasts-and-observed-telemetry).

Method archive hooks store the data the Method chose and the caches it already produced. Examples include LCHS quadrature, source times, PF schedules, QSP phases and MPS cores, QLS encoding, inverse phases and the original factors that a classical QLS `Plan` reuses, and QPE's reached numerical context. Empty or unavailable caches stay empty or unavailable. Saving does not run a phase search, eigensolver, TT-SVD, objective-grid evaluation or backend compilation to recreate missing data. Shared array and circuit objects are written once per save, by Python object identity.

A bound Aer noise model stores its error circuits, their application sites and its basis gates, which select the Aer target that later preparations build circuits for. Loading starts from the saved basis gates and reconstructs quantum and readout errors with their probabilities, parameterized gates and all-qubit or local qubit associations. It uses the saved parameters rather than replacing a gate name with a fixed matrix. Restoring this data does not simulate the error channel or run another measurement, and ordinary SDK error checks still apply. The loaded model is bound to the reopened Run's own copy of the backend (`run.backend`), so the backend passed to `load_run` keeps the model it holds. Later checkpoints of the Run reuse the loaded model and the unchanged backend caches.

Built-in Method implementations are resolved from the supported registry. An external Method must be supplied explicitly as `method=YourMethod` when loading, and must match the saved implementation. A saved `Source` or arbitrary module path does not choose executable imports. Method-specific formats may require their own dependencies, but do not restore arbitrary Python sessions, clients, sockets or credentials. Reconnect through the backend's normal credential configuration. See the [IBM](ibm.md), [IonQ](ionq.md), [Nexus](nexus.md) and [Slurm](slurm.md) guides.

## Circuit files {#qiskit-ucgate-compatibility}

Load saved `.qpy` circuit entries through NWQLib's loaders (`load_run`, `load_result`), not `qiskit.qpy.load`, because entries containing a Qiskit `UCGate` use a versioned container that works around a Qiskit QPY defect ([dependency issues](dependency_issues.md#qpy-and-ucgate)). Other entries remain ordinary QPY.

## Storage and folder contents {#files-arrays-and-recovery-limits}

<a id="saved-folders-are-read-only"></a>Treat a saved Run folder as read-only, as for a saved Result. [Saved folders are read-only](saved_evidence.md#saved-folders-are-read-only) lists the loading checks.

Run storage uses `ExecutionLimits.max_data_bytes`, which defaults to 10 GB. This is a control on stored data and file bytes, not a bound on transient SDK memory, process RSS or serialization time. Data count against it when they are written, and loading reads a saved folder without counting its files again.

A Run folder holds `run.json`, the selection file (normally `selection.json`), the run log `run.sqlite` with its adjacent `.lock` file, and the files these reference. They form one Run, and `run.json` alone is not sufficient for continuation. A folder with `run.sqlite` and no `run.json` comes from a Run whose creation stopped before any preparation or measurement, and `load_run` adds a note saying so. [Saved formats](development/execution.md#saved-formats) describes the files.

Keep the whole folder available while using a restored `Plan` or Result. Numerical arrays load as read-only NumPy mappings, and their bytes are not normalized or regenerated. Run folders must also stay writable for run-log updates, the lock and new cache files. Available output arrays are restored with their manifests. Missing bytes remain unavailable and do not trigger another measurement.

`run.save(path)` copies the current data, reached caches and a consistent snapshot of the run log. It does not move the live Run or reset used work. Unchanged arrays and backend circuits can be reused after reopening. Superseded working cache files are removed only after their replacements are saved. An output array can also be present in the run log's binary storage, so file usage can include both copies.

`run.extend_limits(...)` sets cumulative caps without refunding previous use. Save and reopen keep the ordered history of increases, and a completed Result keeps the caps and history it captured, even when the Run was extended later ([raise a limit during a run](prepared_execution.md#raise-a-limit-during-a-run)).

The `.nwqlib-cache-` filename prefix is reserved for cache files that reopening can remove ([files in a Run folder](prepared_execution.md#files-in-a-run-folder)). Loaders support one current Run format and one current Result format. Other versions raise an explicit format error and are not converted or overwritten. Changing an input, preparation, readout or scientific setting requires a new Problem or Method and a new `Plan`.
