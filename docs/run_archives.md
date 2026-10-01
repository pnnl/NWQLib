# Save results and continue runs

A Result archive supports inspection and a Method's supported reanalysis. A Run folder supports continuation of the same execution, including pending jobs. Both save the original `Plan` (the selected construction and its costs, computed by `plan` before any circuit exists) through its Method's archive hooks.

| Next action | Save | Open |
| --- | --- | --- |
| Inspect or reanalyze stored scientific output | `result.save(path)` | `nwqlib.load_result(path, method=None)` |
| Continue the same execution | `run.save(path)` or the live `run.directory` | `nwqlib.load_run(path, backend=connection, method=None)` |

The standalone Result archive has a 10 GB limit. Run storage uses `ExecutionLimits.max_data_bytes`, which defaults to 10 GB. These are stored-data and file-byte controls, not bounds on transient SDK memory, process RSS or serialization time. They are charged when data are written, and loading reads a saved folder without charging its files again. Save destinations must be new directories whose parents already exist. A Run created with `prepare(..., directory=...)` creates its working directory and missing parents itself.

## Saved folders are read-only

Treat a saved Result or Run folder as read-only. Loading checks format versions, the identity of the selected `Plan` and of every record saved together with its identity, the header of every native input array and published Result array against its declaration (shape, encoding, byte count and contiguity), and what the Method's archive hook checks. The LCHS hook compares the recomputed identities of its saved `LCHSData` parameters and PREP tensors with the selected records, and it uses the periodic Strang payload as saved. Reopening a Run registers each published array without reading it. An array is read from the journal when it is first used, with its block order and byte count checked, and its digest and values are not recomputed. A loaded Result is also checked by its Method's `validate_plan`. QHD checks its observed masses against the roundoff window of their executions (`validate_analysis_masses`) when it analyzes them, not when a Result is loaded. The joins of observations to their preparation records and attempts, including the unit bound of exact probabilities, are checked when an observation is created and before a Result is saved, and loading does not repeat them. A counts observation requests at most `2**63 - 1` shots, the int64 maximum of the weights that `ObservationChunk.histogram()` returns, and each of its counts and their exact total are at most its requested shots. An observation outside that domain is rejected when it is created and when it is loaded. Probability readouts store their numerical values in binary arrays. The observation chunk contains the array manifests, outcome layout and scalar population summaries. Dense probabilities use outcome positions as indices. Sparse probabilities use sorted unique integer indices. The first selected probability qubit is least significant in its outcome index. Saved reports describe these arrays without loading their values. Counts keep one JSON count record per stored outcome. A trajectory readout saves one observation per point. Each names its point ID and resolved boundary, so a saved value maps to its body, point and preparation record after reload, and the point observations share their measurement's completion, preparation, shot and native-work population. A point observation stores only its own point's readout declaration and the identity of the trajectory declaration, which the preparation record stores once. A missing point makes its dependent quantity incomplete. Loading and analysis never replay the shared prefix to supply it. Loading does not detect edits to other saved data, such as the values of native input arrays and of Result arrays, QPY circuit files, Method caches such as QPE spectra or ADAPT vectors, other saved construction data of the Methods, and the rows of `run.sqlite`. An edited value there can change a later solve, analysis, verification or continuation while the `Plan` identity stays the same. ADAPT's saved compiler rows carry an identity of the accepted generator, route and blocks, which is checked when the `Plan` archive is loaded and when verification or continuation adopts a Result's or Run's rows, and which is a consistency digest rather than an authentication, so an edit made together with a recomputed identity is not detected, as for the identifiers of the Records. To change an input or a scientific setting, build a new Problem or Method and a new `Plan`.

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

Opening the completed Run restores its saved Result. `wait()` returns that result without another measurement or analysis. An in-memory local Run can save a snapshot without having selected a journal at preparation time. The snapshot receives a journal, while the original live Run keeps its existing storage arrangement. Call `run.save` while the Run is open, at a returned execution boundary. Closing releases private execution caches and the controller lock, while the Result and its immutable scientific data remain available.

`load_result` needs no backend. It loads the saved concrete Result and sufficient data without calling `plan`, `analyze`, circuit lowering or a numerical reference. `restored.analyze(...)` is a separate explicit operation and uses only the data and posthoc settings supported by that Method. It cannot continue a pending provider job. See [saved results](saved_evidence.md).

## Save and reopen pending work

For a remote backend, select its actual connection and a durable folder before preparation. The following IBM example requires an account authorized for the named device and instance. It uses the `selected` counts `Plan` above.

```python
from nwqlib.backends import IBMRuntimeBackend

backend = IBMRuntimeBackend(
    device="YOUR-IBM-DEVICE", instance="YOUR-IBM-INSTANCE",
    max_input_bytes=65_536,
)
prepared = nwqlib.prepare(selected, backend=backend, directory="provider-run")
with prepared.run as run:
    nwqlib.submit(prepared)
    snapshot = run.save("provider-frontier")
    print(run.exposure)
```

`max_input_bytes` limits the serialized QPY bytes of each prepared circuit that this durable Run stores before submission. Choose a limit that fits the prepared circuits. Constructing the backend reads no credentials and does not contact the service.

Preparation can itself wait for remote compilation. `submit` returns the Run after the Method reaches its next available boundary. `run.result` is `None` while scientific output is pending. Saving at that boundary keeps the original preparation IDs, job locators, completed observations and controller state. Remote execution selects a durable folder automatically if `directory` is omitted. Read its actual path from `run.directory`.

After closing the original controller, reopen either its working folder or the saved snapshot with an instance of the same backend configuration:

```python
with nwqlib.load_run(snapshot, backend=backend) as continued:
    continued.resume()
    if continued.result is None:
        print(continued.exposure)
    else:
        continued.result.save("provider-result")
```

`load_run` opens that folder's existing writable SQLite journal and takes its exclusive controller lock. It does not copy the journal into a new working location, and recovery can record an interrupted attempt's status. Continue only one original or snapshot copy of a run, since their provider job IDs identify the same work. Continue the original working folder if the saved snapshot must stay unchanged. Opening a snapshot makes that directory the new working state.

Loading restores local state without contacting the provider. `run.resume()` retrieves outstanding original work and advances the Method. Known jobs and uncertain submission attempts do not authorize a replacement submission. Retrieval errors preserve the original locator and propagate to the caller. `run.wait(timeout=..., poll_interval=...)` repeatedly resumes until a Result is available or its timeout is reached. A timeout leaves the same Run available for later continuation.

An interrupted local attempt has no external job to retrieve. If the Method cannot finish from its already collected prefix, `resume()` and `wait()` raise `RunFailed` at the recovery stage with the original uncertain attempt attached. Its status stays `uncertain`, and consumed work stays charged. This error does not authorize resubmission or promise that the same local attempt can continue. Close the Run after inspecting its observations and charged work. Use a new Run for new work. A recoverable remote job can still be retrieved through its original locator, and a Method that can finish from an existing prefix may return its Result before this recovery error is needed.

When an uncancelled Run reopens an interrupted synchronous Aer attempt with no saved job locator, its in-process result is unavailable. If the Method cannot return a lawful Result from the saved data, both `resume()` and `wait()` raise `RunFailed` with `stage="recovery"`, `status="uncertain"` and the original attempt identity in `attempt`. The exception includes the attempt record in `trace`, the saved Run directory and the charged exposure. Reopening marks the submission `failed` with `results_consumed=True` to record that its output cannot be retrieved, and releases its pending output-byte reservation. The attempt stays `uncertain`, and its job, circuit and shot exposure stays charged. Inspect the saved evidence or cancel this Run. To execute the Plan again, start a new Run. Nothing is resubmitted in the original Run. Detached submissions keep that reservation while their original job can still supply output. NWQ-Sim can recover a lost launch acknowledgement from the original submission UUID even when no job locator was saved.

RWPE, Lanczos sensitivity sampling and ADAPT continue through their Method-owned controller on this same Run. There is no separate public driver to select. An interrupted classical analysis is not retried by ordinary resume. Where the Method supports explicit recovery, `run.resume(reanalyze=True)` retries that saved analysis boundary. This can change a later optimizer trajectory and is separate from retrieving a pending quantum job.

An unrefined QHD augmented-Lagrangian round or a box-refinement level executes its selected `Plan` in its own Run. Automatic inequality selection can plan additional candidates before that execution. The constrained controller commits the round's resolved inequality representation before its first inner Run and reuses it on resume. With `directory=...` the two layers create these Runs durably under one directory and keep an outer record of the completed rounds and levels. `resume_augmented_lagrangian` and `resume_box_refinement` continue an unfinished round's or level's Run with `load_run` under the rules above and never give it a second Run ([QHD guide](algorithms/qhd.md#constrained-problems)).

The QHD constrained controller uses `qhd.constrained_run/4`, and a standalone refinement controller uses `qhd.refinement_run/5`. A saved constrained-result archive uses `qhd.constrained/3`, and a saved standalone refinement archive uses `qhd.refinement/3`, both with inner QHD Results in `qhd/7`. These format names belong to the layer and Method records, separately from the common Run and Result formats.

Durable resume reopens the recorded inner Run with its recorded `Plan` and journal. Completed work stays in the outer record and is counted once. A recoverable Run can continue, including a remote result that becomes available after interruption. Some interrupted local preparation, measurement or classical evolution cannot be completed from the saved Run.

An unfinishable Run normally raises with a recovery note. With `end_at_unfinishable=True`, a handled unfinishable case after a completed round or level can return `termination="inner_failed"` with the completed results and charged work. If no round or level has completed, the error still propagates. A folder without a committed Run journal header fails during reopen, and this option does not convert that failure to a terminal result.

For a headerless Run folder, follow the error's instruction about removing that named inner folder before retrying. Resume does not remove it or recreate it automatically. When an existing journal cannot be read, keep the folder and retry when it is readable. Its resource counts can be unknown.

A constrained-result archive and a durable controller directory serve different purposes. Load a saved constrained-result archive with `load_augmented_lagrangian`. To continue a durable constrained run, use `resume_augmented_lagrangian`, then save its returned constrained result to an archive if needed.

Open a saved standalone refinement archive with `load_box_refinement`. Continue a durable refinement controller directory with `resume_box_refinement`, then call `save` on its returned result to create a separate result archive. A headerless inner Run folder fails during reopen, including with `end_at_unfinishable=True`. Follow the recovery note for that specific folder.

## What stays associated

The Run folder keeps the selected Problem, Method, output, reconstruction and randomness together with actual preparation records, submission attempts, observations, consumed limits and controller decisions. A selected comparison row also keeps its original `PlanEstimate` and `Allocation`. Continuation does not reevaluate the profile, replace shots or configure the backend from an Allocation. Changed runtime/compiler evidence remains distinct from the original forecast, as described in [profiles and telemetry](profiles.md#original-forecasts-and-observed-telemetry).

Method archive hooks store actual selected data and already-produced caches. Examples include LCHS quadrature, source times, PF schedules, QSP phases and MPS cores, QLS encoding, inverse phases and the original factors that a classical QLS `Plan` reuses, and QPE's reached numerical context. Empty or unavailable caches stay empty or unavailable. Saving does not run a phase search, eigensolver, TT-SVD, objective-grid evaluation or native compilation to recreate missing data. Shared array and circuit objects are written once per save by object identity.

A bound Aer noise model stores its error circuits, their application sites and its basis gates, which select the Aer target that later preparations lower circuits to. Loading starts from the saved basis gates and reconstructs quantum and readout errors with their probabilities, parameterized gates and all-qubit or local qubit associations. It uses the saved parameters rather than replacing a gate name with a fixed matrix. Unsupported saved instruction fields reject instead of being silently discarded. Restoring this data does not simulate the error channel or run another measurement, and ordinary SDK error checks still apply. The loaded model is bound to the reopened Run's own copy of the backend (`run.backend`), so the backend passed to `load_run` keeps the model it holds. Subsequent checkpoints reuse the loaded model and the unchanged native caches. A model saved without basis gates is rebuilt from Aer's default basis (`id`, `rz`, `sx`, `cx`) plus the gates its errors name. When the original model had other basis gates, settings that the reopened Run prepares are lowered to that different target, and an analysis that requires one native context across settings, such as Expectation, rejects the mixed Run. Complete such a Run in its original process, or plan it again.

Built-in Method implementations are resolved from the supported registry. An external Method must be supplied explicitly as `method=YourMethod` when loading, and must match the saved implementation. A saved `Source` or arbitrary module path does not choose executable imports. Native circuits use QPY with the UCGate storage adaptation described below. Method-specific formats may require their selected dependencies, but do not restore arbitrary Python sessions, clients, sockets or credentials. Reconnect through the backend's normal credential configuration. See the [IBM](ibm.md), [IonQ](ionq.md), [Nexus](nexus.md) and [Slurm](slurm.md) guides.

## Qiskit UCGate compatibility

In Qiskit 2.5.2, a native `UCGate` can be written by QPY but cannot be loaded: QPY passes the gate's matrix table as separate positional constructor arguments, whereas `UCGate` expects one list. This is a constructor reconstruction defect; it is not evidence of a reversed qubit order.

NWQLib stores an encountered UCGate as a compact QPY instruction carrying its matrix table and explicit reconstruction fields. A file prefix `NWQLIB-QPY-UC1` identifies this versioned container. The reader restores the nominal number of qubits, existing simplified table, active controls, `up_to_diagonal`, label and any already materialized definition. Circuit phase and wire associations remain attached to their original circuit. Under this SDK's convention, the target is wire 0 and simplified control indices are one-based within the gate, from 1 through `num_qubits - 1`. Loading does not rerun multiplexor simplification or expand the table to all nominal controls. Saving does not synthesize a UCGate or construct its full matrix.

Use NWQLib's archive loaders for these `.qpy` entries: the prefix makes them containers, rather than standalone input for `qiskit.qpy.load`. Circuit files without this adaptation remain ordinary QPY. An old raw QPY entry affected by the upstream bug still reports its original SDK failure, with the entry path; the loader cannot infer missing reconstruction fields from it.

The small `test_native_ucgate_qpy_limitation_canary` test exercises raw QPY on one two-qubit UCGate. It runs no simulation or synthesis. If native loading starts succeeding, or its failure changes, the test requests a compatibility review. A future Qiskit fix does not by itself remove support for existing `UC1` files: keep that decoder and verify simplified controls, diagonal flags and executable round-trips before changing the writer. The UC1 storage is not switched off automatically based on an SDK version number.

## Files, arrays and recovery limits

`run.json` contains `{format, selection, limits}`: the format, the name of the selection file and the execution limits in force when the folder was written. The selection file, normally `selection.json`, holds the Method name, the `Plan` identity and the record returned by the Method's archive hook. The built-in hooks put the `Plan` record in that record and hold or name the selected data there. Keeping the `Plan` and its data out of `run.json` keeps that manifest small, and `load_run` refuses a manifest larger than one MiB before parsing it. A Run writes the manifest as `run.json.partial` and then renames it to `run.json`, which replaces the file atomically within one filesystem, before it commits its journal header. A creation that a `max_data_bytes` refusal or an interruption stops before or during that write therefore leaves no `run.json`, and a folder with `run.json` holds a complete one. For a folder with `run.sqlite` and no `run.json`, `load_run` adds a note that the creation of the Run stopped before any preparation or measurement. The partial file stays and counts among the folder's stored file bytes. `run.sqlite` contains the evolving journal, including current limits, preparation and observation records, cache references and the completed Result when present. An observation row leaves out its readout declaration, which its preparation record stores, and reopening restores the declaration from that record, so a readout with many labels is stored once per preparation. The adjacent `.lock` file supports exclusive controller ownership. These files and their referenced payloads form one Run, and `run.json` alone is not sufficient for continuation.

Keep the whole folder available while using a restored `Plan` or Result. Numerical arrays load as read-only NumPy mappings, and their bytes are not normalized or regenerated. Run folders must also stay writable for journal transitions, the lock and new cache files. Available output arrays are restored with their manifests. A reopened Run reads an array's already-stored journal bytes under the Run's data limit the first time the array is used, for example by `run.hydrate(manifest)` or a probability chunk's `histogram()`, and keeps one copy of each payload for the Run's lifetime. Missing bytes remain unavailable and do not trigger another measurement.

`run.save(path)` copies the current selected data, reached caches and a consistent journal snapshot. It does not move the live Run or reset consumed work. Unchanged arrays and native payloads can be reused after reopening. Superseded working cache files are removed only after the replacement references commit. An output array can also be present in the journal's binary storage, so file usage can include both representations.

`run.extend_limits(...)` sets cumulative caps without refunding previous use. Each actual increase stores the old/new limits and pre-publication counters in one immutable journal row, atomically with the new current cap. Save/reopen preserves the ordered history for both initially in-memory and durable Runs. The loader validates sequence, old/new continuity and the final cap against the original journal header. A completed Result keeps the cap/history prefix it captured, even when the enclosing Run was extended later. This needs only a prefix marker and captured byte counter beside the saved Result, without another full trace or array copy. A reopened Result keeps its own identity and observations and is attached to the Run's trace as saved, apart from that prefix and byte counter. Events and submissions that the Run revised after the Result, for example through a later `run.cancel()` and its refresh, appear there with their later status, and so does that cancellation request.

Current loaders support Run format 18 (`nwqlib.run/18`) and Result format 11 (`nwqlib.result/11`). In both, a host kernel declaration records `application_bytes`, the application record bytes that each invocation sets aside. The Run format reserves the `.nwqlib-cache-` namespace so reopening can remove unpublished cache files while protecting input files. Both formats preserve cap-amendment history. Other versions raise an explicit format error and are not converted or overwritten. These files are not authenticated scientific records, and loading does not detect every edit ([Saved folders are read-only](#saved-folders-are-read-only)). Changing an input, preparation, readout or scientific setting requires a new Problem or Method and a new `Plan`.
