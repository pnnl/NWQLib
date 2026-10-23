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

- `result.save(path)` writes a new directory with `result.json` and its supporting files: the `Plan`, the Result, its observations, preparation records, execution trace and kept arrays. The destination must be a new directory whose parent already exists. Saving checks that the `Plan`, the construction, the observations, the preparation records and the physical submissions belong together, and that exact probabilities respect their unit bound. Saving also refuses a `trajectory` readout unless it has one saved observation for every declared point and those observations are the ones its completed attempt names ([Saved formats](development/execution.md#saved-formats)).
- `nwqlib.load_result(path)` reconstructs the configured Method and the `Plan`, then loads the Result and its saved data. It needs no backend and does not call `plan`, `analyze`, backend compilation, an eigensolver or a reference computation.
- `restored.analyze(...)` explicitly computes a new analysis from the saved data. It keeps the original measurements and settings unless the Method supports a named setting that can be changed after measurement, such as binary inference. It makes no new measurements.
- `restored.assess(...)` applies a new stated accuracy criterion to the available evidence ([check against a tolerance](verification.md#check-against-a-tolerance)). Neither `analyze` nor `assess` turns missing assumptions, numerical estimates or conditional intervals into a proven total error bound.

`result.report()` returns a dictionary for reading, which is not a loadable archive.

## Read a saved result without loading it

`read_report` reads the metadata of a saved Result without loading Method code or numerical arrays:

```python
from nwqlib.saved_evidence import read_report

metadata = read_report("expectation-result")
print(sorted(metadata))
```

```text
['data', 'format', 'result', 'selection']
```

`read_report` checks the format version and returns the saved metadata. `load_result` reconstructs the records through their Method and attaches the Result to its Plan and observations. Arrays are read when requested.

## Saved folders are read-only

Treat a saved Result or Run folder as read-only. The checks below support reconstruction of folders written by NWQLib. They do not certify that the files are unchanged.

| Check | When |
| --- | --- |
| The shared format version | Every load |
| The header of every backend input array and every saved Result array against its declaration (shape, encoding, byte count and contiguity). The check reads no values. A mismatch is rejected, so a mismatched file is never read as data of the wrong shape | Every load |
| Backend state and operator headers against their representation, dimensions and encoding. Real, complex, compact and sparse representations are kept without normalization, conversion to dense form or another input conversion | Every load |
| The Result against its Method's `validate_plan` and its observations | Every load |

Every load also reconstructs the Method's saved inputs and computation data, and checks the shot counts of each counts observation. Other checks run at other times. Observations are checked against their preparation records and attempts when they are created and before a Result is saved, and loading does not repeat these checks. QHD checks its observed masses when it analyzes them. A reopened Run checks a saved array when the array is first read. [Saved formats](development/execution.md#saved-formats) gives each of these checks.

A missing trajectory point makes its dependent quantity incomplete, and loading and analysis never simulate the circuit again to supply it. These checks do not repeat the statistical inference or produce missing error evidence. To change an input or a scientific setting, build a new Problem or Method and a new `Plan`.

## Storage and loading cost

The data of a saved Result are limited to 10 GB. The limit applies to saved data, not to the temporary memory of Qiskit, other SDKs or NumPy. Data count against it when they are written, and loading a Result does not count its files again. If saving fails, the new directory is removed.

Saved arrays are loaded on first access as read-only memory maps, so opening a large saved Result reads its metadata and array headers, not all of its data. Keep the folder available while you use a loaded Result. Backend input arrays, Method caches and QPY circuit files (Qiskit's binary circuit format) are also used as saved ([circuit files](run_archives.md#qiskit-ucgate-compatibility)). NWQLib imports the code it needs explicitly, and a saved `Source` never determines which code is imported.

The execution trace keeps the cumulative limits and the history of limit increases as they were when the Result was created. Later increases on the Run do not change it, including after the Result or the completed Run is saved and reopened ([raise a limit during a run](prepared_execution.md#raise-a-limit-during-a-run)).

A Method saves extra data with a Result only when a later action uses them. For example, QPE verification may reuse the eigensystem and spectrum references it has already computed. Saving never computes additional references for possible future use. The full iteration state of an adaptive Run stays in the Run folder ([Continue an interrupted run](run_archives.md)).

[Saved formats](development/execution.md#saved-formats) describes the files of a saved Result.
