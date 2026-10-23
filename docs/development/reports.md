# Result reporting {#scientific-results-and-comparisons}

This page gives the rules that a Result follows when it is attached to its data, displayed, reported, compared and saved, for contributors who write a Method or change the reporting code. [Plan, compare and solve](../scientist.md#interpret-and-save-the-result) and [Save, load and reanalyze results](../saved_evidence.md) describe these operations for users.

## What a Result is attached to

The public objects are the original Problem, the immutable configured Method, the plain `core.planning.Plan`, `core.analysis.Result` and `_prepared_execution.Run`. A Plan contains the chosen construction, its experiments, the requested output and the data needed to rebuild the Method. Resolving an experiment produces a `Realization` that refers to the Plan. Method archive hooks restore the saved bindings and numerical data without planning again.

A Result identifies its Plan, construction, observation view and contributing observation chunks. Its attached RunData keeps every execution event, preparation record and physical submission, including successful unused data and failed or uncertain work. The data that contribute to the scientific result may be a subset of the collected data, while the [exposure](../glossary.md#exposure) still counts every data collection. An analyzer keeps the immutable copy of the data it used. When the Method returns a Result that it has already attached, `Run.resume` saves it as the Run's Result only if it belongs to this Plan and its observations, receipts, forecast, allocation and trace equal the Run's current data. The counter of saved bytes is excluded from that comparison because it keeps growing after the Result is captured.

`Result.analyze` computes a new explicit reduction from existing data. `Result.assess` checks a stated criterion against the scoped facts. Neither action starts a measurement or a retrospective method choice. An absent criterion means no accuracy verdict. The original confidence, samples and dependence assumptions belong to their scientific evidence, and a later requested confidence cannot overwrite them. Conditional evidence stays conditional, and a component bound does not establish total physical accuracy.

## Display

`print(result)` and notebook display use `Result.__str__`, which reads only records the Result already holds. Each concrete Result supplies its first lines through the `_summary_lines` hook, which must not measure, reanalyze, build arrays or recompute scientific facts. These lines lead with the scientific value and its meaning. A Ritz value is not a ground-state proof, a QHD candidate is not a global optimum, and QPE describes its estimator and prepared samples.

- Array outputs preview at most eight elements held in memory. Larger or unavailable arrays show their shape and physical or unit frame, with the output's units.
- An initial condition can display the supplied physical or unit vector held in memory, keeping its scale and phase convention. Compact or descriptive inputs show only their shape until an array exists. Display never creates that array.
- The labels of how data were collected come from the original observation, so an output of a [classical routine](../glossary.md#host-kernel) (a host kernel in the code) stays identified as classical work even when the Plan's execution preference is quantum.
- LCHS names its chosen construction, including classically exponentiated dense branches.

The common lines that follow name the Method and execution route, the first prepared target, how many observation chunks the Result used out of those collected, and the attempt count with uncertain attempts and failed classical-routine calls, which the line prints as "failed host invocations". The last line always says `accuracy not assessed`, because a Result stores no assessment and only an explicit `assess` call compares it with an accuracy criterion. Conditional component facts do not change that line.

## Report dictionary

`result.report()` returns a dictionary for reading:

| Key | Contents |
| --- | --- |
| `summary` | The display text of the Result |
| `plan` | The original chosen construction, output, assumptions and error model |
| `result` | The scientific result with its original analysis origin, environment, facts and uncertainty |
| `trace`, `observations`, `receipts` | Every collected chunk and preparation record, including unused and uncertain exposure |
| `artifacts` | Array manifests and whether each array is held in memory, without the array data |
| `forecast` | The original forecast, not refreshed or reassessed |
| `allocation` | The supplied Allocation |
| `controller` | The existing portable text of the saved iteration state. Private numerical caches and caches of backend objects are excluded |

Unknown values stay `None`, distinct from zero or from an empty set of collected data. The report is not the saved archive. [Save, load and reanalyze results](../saved_evidence.md) describes the `result.json` file and the metadata-only `read_report`, and [Continue an interrupted run](../run_archives.md) describes Run folders.

Reading a summary or a report performs no measurement, reanalysis, numerical validation, array loading or copying. A detached Result still reports its scientific record, with the unattached Plan and data-collection fields explicitly absent. `Result.save` writes through the archive code independently of display.

## Comparison

Comparison keeps the Methods and Plans compared in their supplied order. Missing applicability or resource information stays explicit. Resource context and device profile affect estimates, not scientific inputs or backend compilation. An execution uses the effective backend configuration, and its preparation record and the operation counts of the backend circuit describe that configuration. Resource inspection never silently escalates from a formula to a representative backend compilation or a full preparation.

## Saving and loading

Saving and loading check the content hashes that tie the Result to its Plan and observations, and the concrete Method associations, without repeating numerical routines, per-gate content hashing or physical simulations. They keep the immutable arrays and the Method context through the existing archive code. See [Save, load and reanalyze results](../saved_evidence.md) and [Execution and storage](execution.md).
