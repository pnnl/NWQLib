# Saved results and continuation

A scientific `Result` contains its original `Plan` (the selected construction and its costs, computed by `plan` before any circuit exists) and actual RunData. Save those together:

```python
import nwqlib

path = result.save("completed-result")
restored = nwqlib.load_result(path)
revised = restored.analyze()
```

Here `result` is an already completed scientific Result. The new directory contains `result.json` plus its supporting payload files. The current JSON envelope is `{format, selection, data, result}`, with format `nwqlib.result/11`. It is distinct from the human-reading dictionary returned by `result.report()`. That report is not a loadable archive. The envelope's `selection` entry holds the Method name, the `Plan` identity and the record returned by the Method's archive hook. The built-in hooks put the `Plan` record in that record and hold or name the selected data there. LCHS, for example, saves its node tables, host actions and SELECT angle tables in their own JSON file with NPY arrays. A Run folder uses its separate `run.json` and journal format, described in [Run archives](run_archives.md).

For a metadata-only read without loading Method code or numerical payloads:

```python
from nwqlib.saved_evidence import read_report

metadata = read_report("completed-result")
```

`load_result` reconstructs the configured Method and selected `Plan` through their explicit archive hooks, then loads the original concrete Result and its observations, preparation records, trace and kept arrays. Loading does not call `plan`, `analyze`, native compilation, an eigensolver or a reference computation. The saved input and numerical arrays use read-only NumPy mappings; keep the folder available while using them. Ordinary native entries use QPY; entries containing UCGate use the versioned NWQLIB-QPY-UC1 envelope around a QPY storage representation. See [UCGate compatibility](run_archives.md#qiskit-ucgate-compatibility). The archive is local execution data, not an authenticated scientific record.

Saving a Result validates the association between the `Plan`, selected construction, observations, preparation records and physical submissions, including the unit bound of exact probabilities. A trajectory readout is saved with the observation of every declared point, and their collection in the preparation record's point order must be the one its completed attempt names. Loading checks the preparation records against the selected construction and the Result against its observations, but it does not repeat the joins of observations to their preparation records and attempts ([Saved folders are read-only](run_archives.md#saved-folders-are-read-only)). Validity checks do not replay inference or produce missing numerical evidence. Runtime dependency loading is explicit, and a saved `Source` never chooses arbitrary executable imports.

The execution trace keeps the cumulative caps and immutable amendment history captured by that Result. Later Run extensions do not change this snapshot, including after either standalone Result or completed Run save/reopen. See [cap amendments](prepared_execution.md) for the counters and atomic publication rules.

`restored.analyze(...)` explicitly computes a new analysis from existing data. It preserves the original measurements and settings unless the Method supports a named posthoc analysis setting, such as binary inference. It makes no new measurements. `restored.assess(...)` applies a new stated accuracy criterion to available scoped evidence. Neither action turns missing assumptions, numerical estimates or conditional intervals into a proven total error bound.

Method-owned Result context is saved only when an actual second action consumes it. For example, QPE verification may reuse its already-produced eigensystem and spectrum references. The Method provides an immutable mapping over those existing read-only arrays. Storage never computes additional references for possible future use. Full adaptive Run context remains a separate controller state.

For pending jobs or interrupted controllers use `run.save(path)` and `nwqlib.load_run(path, backend=...)`; see [prepared execution](prepared_execution.md). A Run folder contains its selected `Plan`, current numerical/native caches and the SQLite execution journal. Reopening it uses one exclusive controller lock and restores the same RNG, limits, accumulated usage and provider locators. A completed Run returns its already saved Result. Cache checkpoints share unchanged arrays and native payloads, including after reopen; replaced cache files are removed after the new frontier commits.

The standalone Result archive limit is 10 GB. Run storage follows `ExecutionLimits.max_data_bytes`. These bounds cover represented data and saved output, not all transient SDK/NumPy memory. They are charged when data are written, and loading a Result or Run does not charge its files again. Failed storage prevents further mutation until recovery. Closing a Run releases the controller lock but does not erase saved data or refund work.

Both `load_result` and the metadata-only `read_report` reject an invalid current-format envelope, a saved record whose recomputed identity differs, and a Result that does not match its observations or its forecast, before loading Method code or binary payloads. The dictionary that `read_report` returns lists these checks under `metadata_validation`. Native state/operator headers are checked against their actual representation, dimensions and encoding. This preserves real/complex, compact and sparse storage without normalization, densification or re-ingestion. Header checks do not check every value, sparse index or native-input digest.

Standalone published arrays load as lazy read-only mmaps, so opening a large archive costs its metadata and array headers rather than a full read. The shape, little-endian encoding (`<c16` for complex128, `<f8` for float64 or `<u8` for uint64 outcome indices, as its manifest declares), byte count and contiguity of every published array are checked on every load against its manifest, and a mismatch rejects. That check reads no payload bytes, and it keeps a mismatched file from being read as data of the wrong shape. The array values are not read, and their digests are not recomputed. Native input arrays, Method caches and QPY files are also used as saved ([Saved folders are read-only](run_archives.md#saved-folders-are-read-only)). The metadata-only `read_report` never reads numerical payloads.
