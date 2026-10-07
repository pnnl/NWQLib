# Glossary

Guides, the API reference and docstrings use the words below in the narrow senses given here. Terms that only the QHD guide uses are in [QHD terms](algorithms/qhd.md#terms), and the sign, order, unit and normalization rules are on the [Conventions](conventions.md) page. The Code column names the module or object that defines each term.

## Workflow objects

[How NWQLib works](how_it_works.md) shows how these objects connect.

| Term | Meaning | Code |
| --- | --- | --- |
| <a id="problem"></a>Problem | The scientific question with its input data and requested output, independent of any algorithm. `LinearSystem`, for example, holds A and b. | `problems/records.py` |
| <a id="method"></a>Method | A configured algorithm, such as `LCHS` or `QLS`, with its options. It plans, executes and analyzes one Problem. | `algorithms/protocol.py` |
| <a id="plan"></a>Plan | The choices a Method makes once for one Problem: the construction, its parameters, the snapshot of the random-number stream and the data needed to reconstruct the output. Execution, analysis and saved runs reuse these choices, and a different choice needs a new Plan. | `core/planning.py` `Plan` |
| <a id="setting"></a>setting | One experiment of a Plan, carried out by a prepared circuit or a classical computation. `prepare(plan, settings="all")` prepares every setting that is known before running, and `Prepared.setting_names` names the settings of the prepared circuits. | `_prepared_execution.py` `Prepared` |
| <a id="run"></a>Run | One execution of a Plan. It records the work reserved against its limits, its submissions, the returned observations and its [exposure](#exposure), and it can be saved and reopened. | `_prepared_execution.py` `Run` |
| <a id="preparation-record"></a><a id="receipt"></a>preparation record | The saved record of one preparation of a circuit or host kernel as it was built, with its operation counts on the target, its width, the target and the compiler. Its class is `PreparedArtifact`. Some docstrings and field names call it a receipt. The `receipt` that `result.verify` returns is a different record, a `VerificationReceipt` with the raw facts and the numerical calls of the check. | `execution.py` `PreparedArtifact` |
| <a id="observation"></a>observation | The data one completed execution returned, such as counts, probabilities or Pauli values, joined to its preparation record. | `execution.py` `ObservationChunk` |
| <a id="result"></a><a id="rundata"></a>Result, RunData | The analyzed scientific output and the immutable observations it was computed from. `result.analyze(...)` produces a new Result from the same RunData with new analysis settings and runs no circuit. | `core/analysis.py` |

## Accuracy and error evidence

| Term | Meaning | Code |
| --- | --- | --- |
| <a id="error-evidence"></a>error evidence | Error bounds, reference comparisons and check results, each stated as a fact about one output. A Result carries its own facts in `result.facts`, `result.verify` returns further facts, and `Result.assess` combines the available facts against a tolerance ([Check accuracy and verify a result](verification.md)). | `evidence/error_model.py` |
| <a id="fact"></a><a id="error-model"></a>fact, error model | A fact is a scoped statement about one error source, such as a bound with its confidence. A Plan's error model lists the error sources its Result needs, and `Result.assess` combines the available facts. | `evidence/error_model.py` |
| <a id="unknown"></a><a id="conditional"></a>unknown, conditional | Unknown means that no evidence exists for a quantity, and it never counts as zero or as a pass. Conditional means that a statement holds only under an assumption the library did not check. | `evidence/error_model.py`, `backends/assessment.py` |
| <a id="error-frame"></a>error frame | The quantity, metric, unit, scope, conditioning and domain in which evidence about an output is stated. Two frames are compatible only when all six agree, and `Result.assess` combines only facts whose frame is compatible with the output's frame ([Error frames](conventions.md#error-frames)). | `evidence/error_model.py` `ErrorFrame` |
| <a id="source"></a>Source | A declaration of the paper, version and domain behind a construction or cost formula. It records where the construction comes from and verifies nothing. | `core/records.py` `Source` |

## Limits and cost

| Term | Meaning | Code |
| --- | --- | --- |
| <a id="work"></a><a id="work-units"></a>work, work units | A count of scalar operations, derived from sizes by a stated formula and compared with a limit, such as `max_work`, before the computation starts. It is not measured CPU time. | [Byte and work budgets](CODE_TOUR.md#byte-and-work-budgets) |
| <a id="peak-live-arrays"></a><a id="known-bytes"></a><a id="data-frontier"></a>peak live arrays, known bytes, data frontier | The bytes of the arrays that one operation holds at the same time at its peak, computed from sizes before allocation. Each byte formula is an allowance, a sum of per-array terms meant to cover that peak, so it can exceed the bytes actually allocated. | `operators/access.py`, [Byte and work budgets](CODE_TOUR.md#byte-and-work-budgets) |
| <a id="admission"></a><a id="admit"></a>admission, admit | The check, made before expensive work, that an input, Program or request lies in its mathematical domain and within its byte and work limits. An admitted request proceeds, and a rejected one raises an error. | `operators/inputs.py`, `ir/validation.py` |
| <a id="planning-work-limit"></a><a id="admission-ceiling"></a>planning work limit (admission ceiling) | The `max_steps` of `AdmissionLimits` when a Method option sets it for its Program, such as `QLS.max_admission_steps`. | `ir/expressions.py` `AdmissionLimits` |
| <a id="charge"></a>charge | To count an amount against one named budget. The budgets are work units against `max_work`, bytes against `max_bytes`, circuits, shots and synthesis work against the limits of a Run, and an error bound against a state budget, the phase-error budget of the QHD running phase total or one of the error sources (`error_sources`) of a QHD circuit. The text names the budget, and a charge counts only against the budget it names. | [Byte and work budgets](CODE_TOUR.md#byte-and-work-budgets), `_prepared_execution.py`, `algorithms/qhd/resources.py` |
| <a id="exposure"></a>exposure | Exposure reports a Run's counts of jobs, circuits, shots and provider-managed sampling under `completed`, `reserved`, `uncertain` and `failed`, with failed or cancelled provider jobs counted as `uncertain` because their consumed work is unknown. | `_prepared_execution.py` |
| <a id="uncertain"></a>uncertain | The state of an attempt or submission whose outcome cannot be established, for example after an interruption or a lost acknowledgement. Its reserved work stays counted, and the library neither resubmits it nor returns it to the budget. When the backend can still refresh the submission or reconcile its progress, a later explicit resume can collect the late original result. Among attempts, only a host kernel that raised an ordinary exception is recorded as failed instead. | `_prepared_execution.py` `Run.exposure`, `Run.resume` |
| <a id="fold"></a>fold | Computation of resource counts from a Program's calls and their analytical cost formulas, without building circuits. An unknown cost stays unknown. | `resources/fold.py` |
| <a id="slot"></a>slot | A gate position of a decomposition that a CX or rotation count formula counts before cancellation or angle classification, so a built circuit can use fewer gates than its slots. | `_preparation_laws.py`, `algorithms/qhd/resources.py` |

## Circuits and numerical construction

| Term | Meaning | Code |
| --- | --- | --- |
| <a id="program"></a>Program | NWQLib's description of the construction a Plan chose, as named block calls with their arguments and qubit ports, without a circuit. Resource estimates and circuit building both read it. | `ir/records.py` `Program` |
| <a id="block"></a><a id="binding"></a>block, binding | A block is a subroutine definition called by a Program. Binding attaches trusted backend code, such as a Qiskit circuit builder, to a block, matched by the exact [content hash](#content-identity) of the block's record. | `blocks/selection.py` `SelectedBlock` |
| <a id="lowering"></a>lowering | Building a Qiskit circuit from a Program and its bound blocks. | `blocks/lowering.py` |
| <a id="host-kernel"></a>host kernel | A classical computation that a Program runs on the host computer in place of a circuit, for example the noiseless Hadamard-test means of QPE's classical execution. The declaration is part of the Plan, and `BoundKernel` attaches the callable from the Method's own code. | `blocks/records.py` `SelectedKernel` |
| <a id="reconstruction"></a>reconstruction | The data of a Plan, defined by its Method and fixed at planning, that analysis needs to interpret the observations. QHD's `QHDReconstruction`, for example, holds the support tables, the initial amplitudes, the compiled step blocks, the physical phase sources and the pruning and AQFT error bounds (`pruning_error_bound`, `aqft_error_bound`), so execution, analysis and loading do not evaluate the objective again. | `core/planning.py` `Plan`, `algorithms/qhd/records.py` `QHDReconstruction` |
| <a id="quadrature"></a>quadrature | Two unrelated senses. In LCHS it is a numerical integration rule, over the kernel variable k, whose nodes and weights define the SELECT branches, or over the source time of the Duhamel integral. In the Hadamard tests of QPE and of ADAPT's sampled queries it is one of the two settings of an amplitude, with ancilla phase 0 for the real part and -pi/2 (an S† gate) for the imaginary part. | `algorithms/lchs/providers.py`, `algorithms/qpe/numerical.py`, `algorithms/gcim/pencil.py` |
| <a id="recovery-scale"></a>recovery scale | The positive factor that turns success-projected amplitudes or probabilities back into physical values, such as the normalization and polynomial rescale of QLS or the LCU normalization of LCHS. It is stored as a binary mantissa and exponent so that products outside the binary64 range stay representable. | `problems/inputs.py` `PhysicalScale`, `compose_recovery` |
| <a id="norming-bound"></a>norming bound | An upper bound on the maximum of a degree-d polynomial over [-1, 1], computed from its largest absolute value on N > d Chebyshev zeros divided by cos(pi d / (2N)) (Ehlich and Zeller 1964, doi:10.1007/BF01111276, Satz 2). QSP phase solving, the QLS polynomial certificates and QSP evolution use it. | `subroutines/qsp/phases.py` `chebyshev_norming_sup_bound` |
| <a id="physical-phase"></a>physical phase | The global phase of an operator or state as the physics defines it, kept rather than dropped modulo a global phase. QHD's compiled one-hot blocks omit the identity part of their exponents, and the circuit restores that part once as the physical phase, while binary diagonal blocks keep theirs. | `artifacts.py`, `algorithms/qhd/compiler.py` |

## Rounding error of computed states

| Term | Meaning | Code |
| --- | --- | --- |
| <a id="state-budget"></a>state budget | A first-order bound delta on the 2-norm distance, up to a global phase, of a computed state from the exact state of the model that the path evaluates, formed by charging each operation its roundoff. Mass and tie windows are derived from it. | `_validation.py` `expm_multiply_state_error`, `native_state_error`, `execution.py` `PreparedArtifact` |
| <a id="mass-window"></a>mass window | The allowed deviation of a computed probability mass from its model value, with the source of the allowance stated where it is used. A host kernel derives its window for the gap between one and its total probability from its state budget. | `_validation.py` `state_mass_window` |
| <a id="roundoff-exclusion"></a>roundoff exclusion | A label in a preparation record's `probability_window_exclusions` that names an operation or condition the roundoff derivation does not cover, such as an excluded `unitary` instruction or an unchecked simulator version. The preparation record then gives no state budget unless the caller resolves the entry. | `execution.py` `PreparedArtifact` |

## Saved runs

| Term | Meaning | Code |
| --- | --- | --- |
| <a id="run-log"></a><a id="journal"></a>run log (journal) | The file `run.sqlite` of a saved Run. It holds the current limits, the preparation and observation records, cache references and the completed Result when present ([Continue an interrupted run](run_archives.md)). | `_prepared_execution.py`, `_run_archive.py` |
| <a id="crash-window"></a>crash window | The interval between two consecutive commits to the [run log](#run-log) during one acquisition, the step in which a Run runs its circuits and collects their data. The module docstring of `_prepared_execution.py` numbers the commits and states what reopening a Run finds after a stop in each window. | `_prepared_execution.py`, [Execution and storage](development/execution.md#write-order-and-crash-windows) |
| <a id="execution-frontier"></a>execution frontier | The state from which a Run resumes: its pending submissions and the next step of its Method's iteration. Saving a Run writes it. This sense differs from the data frontier under [Limits and cost](#limits-and-cost). QHD box refinement keeps the same kind of state in `Frontier`, which holds the levels saved so far and the stop that the last of them decided. | `_run_archive.py`, `algorithms/qhd/_durable.py` |
| <a id="content-identity"></a><a id="content-hash"></a>content identity, content hash | A hash over a record's type, schema version and declared fields. Loading recomputes it for every record saved together with its hash, so an edit to such a record is rejected. Other saved files are trusted, as [Saved folders are read-only](saved_evidence.md#saved-folders-are-read-only) lists. Docstrings do not enter it. | `core/records.py` `Record` |

## Words in docstrings and contributor pages

| Term | Meaning | Code |
| --- | --- | --- |
| <a id="selected"></a>selected | Chosen at planning and fixed in the Plan. Execution, analysis and saved runs reuse the selected construction, and a different choice needs a new Plan. | `core/planning.py` |
| <a id="actual"></a>actual | What was really used or executed, as opposed to a declared, nominal or recomputed value. The actual shots, for example, are the shots the backend returned. | |
| <a id="owner"></a>owner | The one module or function that defines a quantity or enforces a rule. Other code calls the owner instead of repeating the rule. | |
