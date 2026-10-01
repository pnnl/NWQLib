# Save, load and reanalyze results

<a id="saved-results-and-continuation"></a>Save a completed `Result` to a folder, load it later without a backend, and reanalyze its saved measurements with new settings or assess them against a new tolerance, all without new measurements. To continue a run whose provider jobs are still pending, or one that was interrupted, see [Continue an interrupted run](run_archives.md).

## Save, load and reanalyze

```python
import nwqlib
from nwqlib.algorithms import ExpectationMethod
from nwqlib.evidence.binary import BinaryInferenceOptions

problem = nwqlib.Expectation(state=[1., 1.],
                             observable=[[1., 0.], [0., -1.]])
result = nwqlib.solve(problem, method=ExpectationMethod(), shots=256, seed=3)
path = result.save("expectation-result")

restored = nwqlib.load_result(path)
hoeffding = BinaryInferenceOptions(
    method="hoeffding", failure_probability=0.05,
    sampling_model="iid_bernoulli", independent_populations=True,
)
revised = restored.analyze(inference=hoeffding)
print(restored.value, revised.value)
print(revised.facts[0].fact.value.value)
print(revised.assess(absolute_tolerance=0.2, component="sampling").status)
```

```text
-0.03125 -0.03125
0.16976268946757744
INCONCLUSIVE
```

The exact expectation of Z in the state `(1, 1)/sqrt(2)` is 0, and 256 shots estimate it as -0.03125. The reanalysis applies Hoeffding's inequality (doi:10.1080/01621459.1963.10500830) to the same 256 shots and records a sampling radius of about 0.170 at failure probability 0.05 in `revised.facts`. The radius is below 0.2, but the assessment stays INCONCLUSIVE, because the radius is a numerical estimate under the declared sampling model and its binary64 evaluation is not a proven enclosure ([sampling error by method](verification.md#sampling-error-by-method)).

- `result.save(path)` writes a new directory with `result.json` and its supporting files: the `Plan`, the Result, its observations, preparation records, execution trace and kept arrays. The destination must be a new directory whose parent already exists. Saving validates the association between the `Plan`, the construction, the observations, the preparation records and the physical submissions, including the unit bound of exact probabilities. A trajectory readout is saved with the observation of every declared point, and their collection in the preparation record's point order must be the one its completed attempt names.
- `nwqlib.load_result(path)` reconstructs the configured Method and the `Plan` through their archive hooks, then loads the Result and its saved data. It needs no backend and does not call `plan`, `analyze`, backend compilation, an eigensolver or a reference computation.
- `restored.analyze(...)` explicitly computes a new analysis from the saved data. It keeps the original measurements and settings unless the Method supports a named setting that can be changed after measurement, such as binary inference. It makes no new measurements.
- `restored.assess(...)` applies a new stated accuracy criterion to the available evidence ([check against a tolerance](verification.md#check-against-a-tolerance)). Neither `analyze` nor `assess` turns missing assumptions, numerical estimates or conditional intervals into a proven total error bound.

`result.report()` returns a dictionary for reading, which is not a loadable archive.

## Read a saved result without loading it

`read_report` reads the metadata of a saved Result without loading Method code or numerical payloads:

```python
from nwqlib.saved_evidence import read_report

metadata = read_report("expectation-result")
print(sorted(metadata))
```

```text
['data', 'format', 'metadata_validation', 'result', 'selection']
```

Both `read_report` and `load_result` reject an invalid current-format file, a saved record whose recomputed content hash differs, and a Result that does not match its observations or its forecast, before loading Method code or binary payloads. The dictionary that `read_report` returns lists these checks under `metadata_validation`.

## Saved folders are read-only

Treat a saved Result or Run folder as read-only. These checks run on saved data:

| Check | When |
| --- | --- |
| The format version, and the content hash of the `Plan` and of every record saved with its hash | Every load |
| The header of every backend input array and every saved Result array against its declaration (shape, encoding, byte count and contiguity). The check reads no values. A mismatch is rejected, so a mismatched file is never read as data of the wrong shape | Every load |
| Backend state and operator headers against their representation, dimensions and encoding. Real, complex, compact and sparse storage is kept without normalization, densification or re-ingestion | Every load |
| The Method's archive hook. For example, the LCHS hook compares the recomputed content hashes of its saved `LCHSData` parameters and PREP tensors with the records the `Plan` chose, and it uses the periodic Strang payload as saved ([LCHS guide](algorithms/lchs.md#explicit-checks-and-saved-results)) | Every load |
| The Result against its Method's `validate_plan`, the preparation records against the construction, and the Result against its observations | Every load |
| A counts observation requests at most `2**63 - 1` shots, the int64 maximum of the weights that `ObservationChunk.histogram()` returns, and each of its counts and their exact total are at most its requested shots | When the observation is created and when it is loaded |
| The joins of observations to their preparation records and attempts, including the unit bound of exact probabilities | When an observation is created and before a Result is saved. Loading does not repeat them |
| QHD's observed masses against the roundoff window of their executions (`validate_analysis_masses`) | When QHD analyzes them, not when a Result is loaded |
| The block order and byte count of a reopened Run's saved array | When the array is first read from the run log. Reopening registers each saved array without reading it |

Loading does not detect:

- Changed array values, sparse indices or digests of arrays and backend inputs. Values are not read and digests are not recomputed.
- Edits to other saved data, such as the values of backend input arrays and of Result arrays, QPY circuit files, Method caches such as QPE spectra or ADAPT vectors, other saved construction data of the Methods, and the rows of `run.sqlite`. An edited value there can change a later solve, analysis, verification or continuation while the `Plan`'s content hash stays the same.
- An edit made together with a recomputed hash. ADAPT's saved compiler rows carry a content hash of the accepted generator, route and blocks, checked when the `Plan` archive is loaded and when verification or continuation adopts a Result's or Run's rows. It is a consistency digest rather than an authentication, as are the content hashes of the records.

The folder is local execution data, not an authenticated scientific record. A missing trajectory point makes its dependent quantity incomplete, and loading and analysis never replay the shared prefix to supply it. Validity checks do not replay inference or produce missing numerical evidence. To change an input or a scientific setting, build a new Problem or Method and a new `Plan`.

## Storage and loading cost

A saved Result has a 10 GB limit. It is a control on stored data, not on transient SDK or NumPy memory. Data count against it when they are written, and loading a Result does not count its files again. A failed write prevents further changes until recovery.

Saved arrays load lazily as read-only memory maps, so opening a large archive costs its metadata and array headers rather than a full read. Keep the folder available while you use a loaded Result. Backend input arrays, Method caches and QPY circuit files are also used as saved ([circuit files](run_archives.md#qiskit-ucgate-compatibility)). Runtime dependency loading is explicit, and a saved `Source` never chooses arbitrary executable imports.

The execution trace keeps the cumulative limits and the history of limit increases that the Result captured. Later increases on the Run do not change it, including after the Result or the completed Run is saved and reopened ([raise a limit during a run](prepared_execution.md#raise-a-limit-during-a-run)).

A Method saves extra Result context only when a later action uses it. For example, QPE verification may reuse the eigensystem and spectrum references it already produced, which the Method provides as an immutable mapping over the saved read-only arrays. Saving never computes additional references for possible future use. The full iteration state of an adaptive Run stays in the Run folder ([Continue an interrupted run](run_archives.md)).

[Saved formats](development/execution.md#saved-formats) describes the files of a saved Result.
