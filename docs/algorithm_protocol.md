# Add a method

A Method turns a Problem into a Plan (the Program and the experiments to run) and turns the observations into a Result. This page states what a Method must provide, its optional hooks, how to register a Method and how to check it with `check-method`. [Run your own circuit](own_circuit.md) shows the smallest Method, which needs only `descriptor`, `plan` and `analyze`, and [Compose blocks](blocks.md) shows how to build the subroutines a Program calls, such as PREP and SELECT.

## The reference Method {#the-reference-method}

The examples on this page use NWQLib's reference external Method, which estimates the normalized expectation of one Pauli term for an `Expectation` Problem with a Hadamard test. It uses the same public operations as the built-in Methods. Its complete implementation is [`tests/_hadamard_method.py`](https://github.com/pnnl/NWQLib/blob/main/tests/_hadamard_method.py), and its registration, which does not import the implementation, is [`tests/_hadamard_method_metadata.py`](https://github.com/pnnl/NWQLib/blob/main/tests/_hadamard_method_metadata.py). Both files are in the source repository and are not installed with the package, and the test suite uses them as its reference external Method. Put your own Method in an importable module of your package, here called `my_methods`:

```python
from nwqlib import Expectation, plan, prepare, submit
from nwqlib.operators import ingest_pauli
from my_methods import HadamardPauliExpectation

problem = Expectation(
    state=(1., 1j),
    observable=ingest_pauli((("Y", 1.),), num_qubits=1),
)
selected = plan(problem, method=HadamardPauliExpectation(), seed=7)
prepared = prepare(selected)
with prepared.run as run:
    submit(prepared)
    result = run.wait()
    print(result.value)  # approximately one
```

Define a `case()` factory next to the Method, as the reference implementation does, and run the complete author check on your module:

```sh
python -m nwqlib check-method my_methods:case
```

The Hadamard Method supports one nonzero real Pauli term `A=cP` and a normalized preparation of the matching dimension. It rejects several terms, unsupported representations and another output before building anything. Its Program calls system PREP, ancilla H, controlled signed Pauli SELECT and inverse H. With the ancilla first in register order, its final Pauli label is `I...IZ`. For `A=cP`, the ancilla mean is `sign(c) <psi|P|psi>`, and multiplication by `abs(c)` recovers the original normalized expectation. The block functions of `nwqlib.blocks` build the circuits, their control, global phase and restoration, and the Method supplies their ordered ports and the scientific interpretation.

Its tests check three answers known independently of the Method: `|0>, Z -> 1`, `|+>, Z -> 0`, and `|+i>, Y -> 1`. The complex case catches a sign error in Y. A negative coefficient tests the controlled phase, and explicit counts with an HZH preparation show a valid alternative choice of block. These checks use at most two qubits, and the Method builds no reference state and solves no eigenproblem. The Method shows how to write a Method and is not a new algorithm.

The Hadamard Method returns `NormalizedExpectation` through a small Result subclass whose `.value` has the same output frame (quantity, metric, unit and scope) as that built-in output. Its Result's `validate_plan` checks the inputs, alpha (the factor `abs(c)` that the Plan records), the control, the ancilla wire and the Plan's `reconstruction` field, which holds the data the Method needs to turn the ancilla mean back into the expectation. Its `validate_data` ties the reported mean to the completed observation. Another scalar with the same Plan is valid on its own but fails `validate_plan`, and the original observation remains available for reanalysis. These checks neither reconstruct a state nor recompute the quantum experiment.

## Configure the Method {#configuration-and-selected-science}

Subclass `nwqlib.algorithms.Method`, an immutable Record. Declare the constant `descriptor: ClassVar[AlgorithmDescriptor]` and the Method's scientific configuration fields on that class. Required inputs stay required, and schema inspection does not invent values for them. A descriptor states scope and references, and it neither proves scientific applicability nor qualifies a backend.

The Method's content hash is part of every Plan it makes, so changing a configuration field gives a new Plan instead of altering an existing one. Execution, analysis and saved archives then use that one Plan without running `plan` again, which keeps each Result and observation tied to the construction it came from.

### What every Method provides {#required-hooks}

| Part | What your Method must do |
| --- | --- |
| Selection: `plan(problem, *, output, execution, shots, rng, accuracy=None)` | Return the common Plan, keeping the original Problem, the configured Method, the output and the final RNG state. |
| Construction | Put the Program and its block records in the Plan's `construction`, and bind the live `SelectedBlock` objects and any named live inputs with `Plan._bind`. Resource estimates and circuit building read the same blocks. |
| Analysis: `analyze(plan, data, *, settings)` | Return the scientific Result computed from that Plan and the data it uses. It does not collect observations. NWQLib attaches the Result to the Plan and the data with `Result._attach`, which calls `validate_plan`, and also accepts a Result already attached to the same Plan and data. |

### Optional hooks {#optional-hooks}

Leave out the hooks your Method does not support. Do not add placeholder archive or sampling methods to make capability inspection succeed.

| Hook | What your Method must do |
| --- | --- |
| Error information: `error_model(plan)` | Return the Method's known and unavailable error sources, each with its `ErrorFrame`. The default returns `plan.error_model`. |
| Explicit verification: `verify` | Override it only for supported checks that the caller requests explicitly. Each check's cost is counted where its work runs. |
| Result checks: `validate_plan(self, plan)` and `validate_data(self, data)` on your Result class | Raise `ValueError` when the Result's fields do not follow from its Plan or its data. `Result._attach` calls `validate_plan`, and saving a Result calls `validate_data` before writing. In `validate_plan`, call `_validate_common_plan` first, as the [protected hooks](#supported-protected-extension-hooks) table describes. |
| Persistence: `result_type`, `save_archive(plan, files)`, `load_archive(saved, files)` | `result_type` names the Result class. The two hooks save and restore the input data and the live blocks without planning again. [Run your own circuit](own_circuit.md#when-the-archive-hooks-are-required) says when they are needed. |
| Shots from an accuracy request: `sampling_shots(self, problem, *, output, accuracy, execution, shots)` | Return the shot count. It runs before planning. Without it, automatic shot selection is unavailable and `plan(..., accuracy=...)` raises `ApplicabilityError`. Explicitly supplied shots and a Method's own adaptive iteration keep their own rules for the shots they request. |
| Adaptive execution: `prepare_all_refusal()` | Override it when your Method chooses later settings from earlier outcomes, as [Adaptive Methods](#adaptive-methods) describes. |
| Points and submissions: `validate_point(plan, experiment, values)`, `specialize_experiment(plan, experiment, values)`, `selected_kernels(plan, experiment, program)`, `before_submit(plan, prepared, *, run)` | Override them when your experiments take parameter values per point (one assignment of values to the Program's parameters, such as one grid value of a `MeasurementBatch` axis) or your Method must refuse a submission. When one point of an experiment is resolved, `validate_point` raises for values that break your Method's own relations, `specialize_experiment` returns the experiment with its readout fixed from those values and keeps its `name`, `setting`, `batch` and `setting_index`, and `selected_kernels` returns the point's tuple of `SelectedKernel` records, each the Plan's entry of the same name or a revision of it with the same implementation, inputs and dependencies. `before_submit` runs before a submission is recorded and raises to refuse it. The defaults accept every point and submission and change nothing. |
| Continuing an interrupted analysis: `recover_analysis` | A separate, explicitly called hook that continues a saved interrupted analysis. It does not resubmit earlier circuits. |

### Adaptive Methods {#adaptive-methods}

Static Methods inherit the common preparation and execution. Adaptive Methods advance their iteration state through the same Run. `prepare(plan, settings="all")` first calls `method.prepare_all_refusal()`, before a Run exists, and raises ValueError with the returned reason when it is not None, so a refusal leaves no Run folder. The `prepare(plan, *, run)` hook receives the `settings` keyword only for `"all"`. An override without that keyword therefore keeps working for the default, and the inherited `prepare_all_refusal` refuses `"all"` for it. A Method whose iteration chooses later settings from earlier outcomes overrides `prepare_all_refusal` to return why, and what can be prepared instead. Otherwise the inherited static preparation would prepare every experiment of the Plan and count it against the Run's limits, including experiments the iteration never submits.

### Only for reductions during data collection {#reduction-hooks}

A Plan can declare a reduction that a backend applies to readout data, such as a saved state, while the circuits run. Only such a Method needs these two parts.

| Part | What your Method must do |
| --- | --- |
| Reduction allowance: `reduction_allowance(plan, point, *, observation, width, run)` | Run the family's workspace check, raising ValueError to refuse, and return the Method's remaining work allowance as a Python `int`. Preparation calls it before any backend work and may call it again on submission and after a reopen, so it reads its record of work already used without changing it. With `prepare(plan, settings="all")` each preparation is checked separately, so a cumulative limit subtracts the work of the reductions the Run has already completed. |
| Reducer registration | Register the reducer by name, as [Reducer registration](#reducer-registration) describes. |

### Reducer registration {#reducer-registration}

A reducer is registered by name in `core.planning.READOUT_REDUCERS` with its shape function and, when it executes, its work and execution functions. A reducer that ships with NWQLib registers itself when its module is imported. That module is listed in `core.planning.BUILTIN_REDUCER_MODULES`, so that a saved record naming the reducer validates in a process that has not imported the module (`core.planning.registered_reducer`).

A reducer registered with `receives_context=True` also receives a read-only `ReductionContext` of the preparation record that produced the data: its probability window, its exclusions and its qualified state errors. The record's state-error budget bounds the computed state only up to a common global phase. A saved-state reduction receives this estimate as `state_error` only when its declared outputs are invariant under that phase. The estimate is conditional on a [first-order native roundoff model](ENGINEERING_CONSTANTS.md#probability-windows-of-exact-execution) that covers the target. The operation count must be known. Every exclusion must have been assessed and resolved by the reducer's own arithmetic bounds. The host phase correction must also have been assessed. Phase-defined state uncertainty is unavailable unless a separate phase-error model supplies it, and the context reports that reason.

A host phase correction multiplies the saved amplitudes by one finite complex phase factor. Its multiplication error and the factor's deviation from unit modulus enter the state-error estimate and probability window used by all code that uses the corrected array. Projected mass checks can use the modulo-phase estimate even when other diagnostics from the same reducer depend on phase. If native assumptions remain unavailable, those checks use the propagated probability-window convention and report the original exclusions. The propagated window is also unavailable when the host correction has not been assessed.

The context gives three values:

- `state_error`: the modulo-phase estimate for a reducer registered with `phase_invariant=True`, and None with a `state_error_reason` for any other reducer.
- `modulo_phase_state_error`: the modulo-phase estimate, for every reducer.
- `probability_window`: the preparation record's window propagated through the host phase correction, or None when the backend did not assess that correction.

The registration field `state_error_resolutions`, empty by default, names the exclusion labels of the preparation record whose effects the reducer's own arithmetic bounds, and the context's state errors are computed with those labels resolved. A Method that stores the reducer's statistics in its results, or validates them again, must pass the same labels to the preparation record's `saved_state_error` and use its `saved_state_probability_window`, so that data collection, storage and any later validation use the same branch.

### What the shared code keeps {#what-the-shared-code-keeps}

An external Method needs no change to NWQLib's shared code, and it keeps NWQLib's backend accounting, Run logs and transport. Input conversion and circuit building keep their own size limits, and the Run's limits bound the total data collected and stored. An unknown SDK cost stays unknown.

## Register a Method {#explicit-trusted-registration}

`AlgorithmRegistry()` starts empty. `builtin_registrations()` reads the built-in Method types and descriptors from the same map as the public exports. It can import built-in configuration modules, but it constructs no Method, does no planning and imports no quantum SDK. `options_schema(registration)` reads the configured Method's `model_json_schema()`. Listing the built-in Methods can therefore load NumPy, SciPy and configuration dependencies, which has an import cost.

A `Registration` contains an exact Source, a trusted `module:attribute` factory, and optionally an already known descriptor and Method type. Listing registrations does not run a factory. Declare the registration in a module that does not import the Method implementation, as `tests/_hadamard_method_metadata.py` does. For example, with a module `my_methods_metadata` whose factory is `my_methods:HadamardPauliExpectation`:

```python
from nwqlib.algorithms import AlgorithmRegistry
from my_methods_metadata import REGISTRATION

registry = AlgorithmRegistry((REGISTRATION,))
print(registry.discover())  # my_methods remains unloaded

method = registry.resolve(REGISTRATION.source, preparation_choice="native")
```

`resolve` calls only that factory, with the supplied configuration. `expected_type=MyMethod` optionally checks that the Method is an instance of the caller's type and returns it with that type. `direct_method(instance)` checks a Method you already constructed, without a factory. No cache reuses an earlier configuration.

`third_party_registrations()` lists installed `nwqlib.algorithms` entry points named `method@version`. It reads metadata without loading entry-point code. Unknown external descriptors and schemas remain unknown. Duplicate versions are rejected, and a missing version does not fall back to another. Discovery is explicit, and listing the built-in Methods does not discover installed extensions.

A saved Source is lookup data, never an import instruction. Resolution requires an exact registration in the current process, including where the Method came from. Factories and Method hooks are trusted Python code and run without a security sandbox. Loading the saved Result of an external Method requires `load_result(path, method=MyMethod)`, and strings in an archive cannot cause an arbitrary module to be loaded.

## Check a Method with an author case {#author-cases-and-their-limits}

`MethodCase` holds the Method and Problem, an optional output, the execution, shots and seed, and three callbacks:

- `evaluate(plan)` runs the declared, bounded data collection through `prepare` and `submit`, or analyzes data you supply, and returns the Result attached to that Plan.
- `accepts(result)` checks a scientific relation that is expected independently of the Method.
- `invalid_result(result)` changes a scientific field while keeping the Result's concrete type and Plan content hash. The changed Result must remain valid on its own. `check_method` then requires your Result's `validate_plan` to reject it, so a Result class that keeps the base `validate_plan`, which compares only content hashes, fails the check.

`check_method` runs planning and the supplied evaluation, checks the expected relation and the Method's error information, and saves and reopens one temporary Result archive, loading the external Method explicitly. The usual archive byte limits apply. The check adds no reference solve, array reconstruction or verification campaign of its own. The round-trip comparison uses the saved scientific records and their content hashes, not the Python identity of reopened circuit objects. The `accepts` callback alone checks the independent relation, including the meaning of arrays when relevant.

A test-only identity case uses the built-in Expectation Method and the relation `<psi|2I|psi>/<psi|psi>=2`, without running circuits. A false expected relation, an `invalid_result` that does not make the check fail, or disabled pair validation makes the checker fail. `CONFORMANT` means this one case passed. It does not certify other callbacks, other inputs, physical accuracy, provider support or general execution cost. Use a case whose cost and independent expected relation you understand.

## API reference

<a id="api-owners"></a>The API entries for `MethodCase`, `check_method`, `AlgorithmRegistry` and `Registration` are in [Extending NWQLib](api/extending.md):

- <a id="nwqlib.algorithms.authoring.MethodCase"></a>[`MethodCase`][nwqlib.algorithms.authoring.MethodCase]
- <a id="nwqlib.algorithms.authoring.check_method"></a>[`check_method`][nwqlib.algorithms.authoring.check_method]
- <a id="nwqlib.algorithms.registry.AlgorithmRegistry"></a>[`AlgorithmRegistry`][nwqlib.algorithms.registry.AlgorithmRegistry]
- <a id="nwqlib.algorithms.registry.AlgorithmRegistry.discover"></a>[`AlgorithmRegistry.discover`][nwqlib.algorithms.registry.AlgorithmRegistry.discover]
- <a id="nwqlib.algorithms.registry.AlgorithmRegistry.resolve"></a>[`AlgorithmRegistry.resolve`][nwqlib.algorithms.registry.AlgorithmRegistry.resolve]
- <a id="nwqlib.algorithms.registry.Registration"></a>[`Registration`][nwqlib.algorithms.registry.Registration]

## Supported protected extension hooks

These names start with an underscore but are supported for Method and Result implementations. They bind and check the Plan's construction and Result. Calling them does not run a second computation and does not permit a different archive format.

| Hook | What your implementation must do |
| --- | --- |
| `Plan._bind(self, *, blocks=(), **native)` | Bind the live block handles and named live inputs once, and return that Plan. On load, restore the same bindings instead of planning again. |
| `Result._attach(self, plan, data)` | Attach the original Plan and RunData after checking content hashes, observation membership and `validate_plan`. It returns the same Result and refuses to attach it to another Plan or other data. |
| `Result._validate_common_plan(self, plan, plan_type)` | Call it from a concrete `validate_plan` to check the exact Plan type, the Plan's content hash and the recorded Method, then check the Method-specific scientific relation. |
| `Result._summary_lines(self)` | Return a finite sequence of readable lines from existing scalar fields. It must not collect data, solve a reference problem, write arrays or reanalyze. |
| `nwqlib.blocks._archive.write_blocks(blocks, files)` | Call it in `save_archive` to write the Plan's blocks into the archive. It supports the library's trusted block factories, not arbitrary Python callables. |
| `nwqlib.blocks._archive.read_blocks(data, records, files)` | Call it in `load_archive` to restore the saved blocks. It binds only the library's known constructors, so loading never imports or calls code named in the archive. |

A Method that can be saved defines `result_type`, `save_archive(plan, files)` and `load_archive(saved, files)`. The base Method has no callable archive stubs. The `files` argument is an `ArchiveFiles` object, not a filename. Save the Plan description with `plan.to_record()`. `files.read_plan` rebuilds the Plan from that record, the input, output and block readers restore the live objects, and `load_archive` binds them with `plan._bind`. Each file write counts against the archive's byte limits. Use the `ArchiveFiles` object you receive rather than writing a parallel serializer. The reference Hadamard Method writes and restores its blocks with `write_blocks` and `read_blocks`, as the [GHZ example](own_circuit.md#when-the-archive-hooks-are-required) does.

A loader that caches data must register every file it depends on with `files.read_path(name)` during restore, including files whose contents are read later. Registration reads only file metadata, raises for a missing file and does not scan the contents. Reopening a Run protects the registered files before it removes uncommitted files from the reserved cache namespace.
