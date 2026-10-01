# Add a Method

For block composition first, follow the [selected blocks guide](blocks.md): operator input, PREP–SELECT–unPREP wiring, the projected A/alpha relation and explicit native lowering. A Method consumes the same selected blocks and Program. Continue below to connect a new scientific operation to the shared lifecycle.

A Method that builds one circuit and returns its counts needs only `descriptor`, `plan` and `analyze`. [Run your own circuit](own_circuit.md) shows that smallest form and when the archive hooks below become necessary.

A configured `Method` selects one scientific `Problem`, returns the common `Plan` and interprets its actual `RunData` into a `Result`. The external Hadamard Method described here uses `Expectation(state=..., observable=...)` and the same public operations as builtin methods. Its complete implementation is `tests/_hadamard_method.py`, with its inert registration in `tests/_hadamard_method_metadata.py`. The test suite uses it as its reference external Method. Put your own Method in an importable module of your package, here called `my_methods`:

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

The Hadamard Method supports one nonzero real Pauli term `A=cP` and a normalized preparation of the matching dimension. It rejects multiple terms, unsupported representations and another output target before construction. Its Program contains actual system PREP, ancilla H, controlled signed Pauli SELECT and inverse H calls. With ancilla first in register order, its final Pauli label is `I...IZ`. For `A=cP`, the ancilla mean is `sign(c) <psi|P|psi>`; multiplication by `abs(c)` recovers the original normalized expectation. The existing selected-subroutine factories own native construction, control, global phase and restoration. The Method supplies their ordered ports and scientific interpretation.

The independent witnesses are `|0>, Z -> 1`, `|+>, Z -> 0`, and `|+i>, Y -> 1`. The complex case rejects a Y-sign error. A negative coefficient tests controlled phase; explicit counts and an HZH preparation demonstrate a lawful alternative selection. This Method illustrates the extension contract and is not a claim of a new SOTA algorithm. The witnesses use at most two qubits, and the Method performs no reference state construction or eigensolve.

## Configuration and selected science

Subclass `nwqlib.algorithms.Method`, the immutable Record owner. Declare constant `descriptor: ClassVar[AlgorithmDescriptor]` metadata and actual scientific configuration fields on that class. Required inputs remain required; schema inspection does not fabricate them. A descriptor states scope and references; it does not prove scientific applicability or qualify a backend.

The Method's content identity is part of every Plan it selects, so changing a configuration field produces a new Plan rather than altering an existing one. Execution, analysis and archives then consume that one selection without running `plan` again, which keeps each Result and observation tied to the construction it came from.

| Responsibility | Actual contract |
| --- | --- |
| Selection | `plan(problem, *, output, execution, shots, rng, accuracy=None)` returns the common Plan, preserving the original scientific input, configured Method, output and final RNG state. |
| Construction | Plan binds the selected Program and actual `SelectedBlock` or host-kernel capabilities. Resource estimates and native lowering consume the same selections. |
| Execution | Static methods inherit common prepare/execute behavior. Adaptive methods advance their actual controller through the same Run. `prepare(plan, settings="all")` first calls `method.prepare_all_refusal()`, before a Run exists, and raises ValueError with the returned reason when it is not None, so a refusal leaves no Run folder. The `prepare(plan, *, run)` hook receives the `settings` keyword only for `"all"`, so an override without that keyword keeps working for the default, and the inherited `prepare_all_refusal` refuses `"all"` for it. A Method whose controller chooses later settings from earlier outcomes overrides `prepare_all_refusal` to return why and what can be prepared instead. Otherwise the inherited static preparation would prepare and charge every Plan experiment, including ones its controller never submits. |
| Reduction allowance | A Method whose Plan declares an acquisition-time reduction defines `reduction_allowance(plan, point, *, observation, width, run)`. It runs the family's workspace check, raising ValueError to refuse, and returns the Method's remaining work allowance as a Python `int`. Preparation calls it before native work and may call it again on submission and after a reopen, so it reads its ledger without changing it. With `prepare(plan, settings="all")` each preparation is admitted separately, so a cumulative limit subtracts the work of the reductions the Run has already completed. |
| Reducer registration | A reducer is registered by name in `core.planning.READOUT_REDUCERS` with its shape function and, when it executes, its work and execution functions. A reducer that ships with NWQLib registers itself when its module is imported, and that module is listed in `core.planning.BUILTIN_REDUCER_MODULES`, so that a saved record naming the reducer validates in a process that has not imported the module (`core.planning.registered_reducer`). A reducer registered with `receives_context=True` also receives a read-only `ReductionContext` of the producing receipt: its probability window, its exclusions and its qualified state errors. The receipt's state-error budget bounds the computed state only up to a common global phase. A saved-state reduction receives a qualified native estimate modulo global phase when its declared outputs are invariant under that phase. Phase-defined state uncertainty is unavailable unless a separate phase-error model supplies it. The context reports that reason. A host phase correction adds its finite multiplication and modulus error to every consumer of the corrected array. Projected mass checks can use the modulo-phase estimate even when other diagnostics from the same reducer depend on phase. If native premises remain unavailable, they use the propagated probability-window convention and report the original exclusions. In the context, `state_error` is the modulo-phase estimate for a reducer registered with `phase_invariant=True`, and it is None with a `state_error_reason` for any other reducer. `modulo_phase_state_error` gives every reducer the modulo-phase estimate, and `probability_window` is the receipt's window propagated through the host phase correction, or None when the backend did not assess that correction. `state_error_resolutions`, empty by default, names the receipt exclusion labels whose effects the reducer's own arithmetic bounds, and the context's state errors are computed with those labels resolved. A Method that publishes or validates the reducer's statistics again must pass the same labels to the receipt's `saved_state_error` and use its `saved_state_probability_window`, so that acquisition, publication and any later validation use the same branch. |
| Analysis | `analyze(plan, data, *, settings)` returns the scientific Result attached to that Plan and sufficient data. It does not acquire observations. |
| Error information | `error_model(plan)` exposes the selected method's framed known and unavailable sources; the default returns `plan.error_model`. |
| Explicit verification | Override `verify` only for supported explicitly selected checks at their actual cost owners. |
| Persistence | `result_type` names the actual Result class. `save_archive(plan, files)` and `load_archive(data, files)` save/restore selected input data and native bindings without replanning. |

An external method adds no common family switch. It also does not replace backend accounting, durable journals or transport. Input conversion and selected native work keep their concrete size controls; Run limits govern cumulative acquisition and stored data. An unknown SDK cost remains unknown.

The Hadamard Method selects `NormalizedExpectation`, with a small Result subclass whose `.value` has that existing output frame. Its Plan/result validator checks actual inputs, alpha, control, ancilla wire and reconstruction. Its data validator joins the reported mean to the actual completed observation. A different same-Plan scalar is intrinsically valid metadata but fails that scientific pair. The original observation remains usable for reanalysis. These validators do not reconstruct a state or recompute the quantum experiment.

## Explicit trusted registration

`AlgorithmRegistry()` starts empty. `builtin_registrations()` reads the actual builtin Method types and descriptors from the same owner map as public exports. It can import builtin configuration modules, but constructs no Method, performs no planning and imports no quantum SDK. `options_schema(registration)` reads the actual configured Method's `model_json_schema()`. Explicit builtin inventory can therefore load NumPy, SciPy and configuration dependencies. Its promise of no SDK import and no planning is not a promise of zero import cost.

A `Registration` contains an exact Source, a trusted `module:attribute` factory, and optionally an already known descriptor and Method type. Inventory does not execute a factory. Declare the registration in a module that does not import the Method implementation, as `tests/_hadamard_method_metadata.py` does. For example, with a module `my_methods_metadata` whose factory is `my_methods:HadamardPauliExpectation`:

```python
from nwqlib.algorithms import AlgorithmRegistry
from my_methods_metadata import REGISTRATION

registry = AlgorithmRegistry((REGISTRATION,))
print(registry.discover())  # my_methods remains unloaded

method = registry.resolve(REGISTRATION.source, preparation_choice="native")
```

Resolution invokes only that selected factory with the supplied configuration. `expected_type=MyMethod` optionally preserves a caller's concrete type at this dynamic boundary. `direct_method(instance)` checks the supplied Method without a factory. No cache silently reuses a prior configuration.

`third_party_registrations()` enumerates installed `nwqlib.algorithms` entry points named `method@version`. It reads metadata without loading entry-point code. Unknown external descriptors and schemas remain unknown. Duplicate versions are rejected, and an absent version does not fall back. Discovery is explicit; builtin inventory does not automatically discover installed extensions.

A saved Source is lookup data, never an import instruction. Resolution requires an exact in-process registration, including provenance. Selected factories and Method hooks are trusted Python code, not a security sandbox. Full external Result loading requires `load_result(path, method=MyMethod)`; archive strings cannot authorize loading arbitrary modules.

## Author cases and their limits

`MethodCase` contains the actual method/problem, optional selected output, execution/shots/seed, and three explicit callbacks:

- `evaluate(plan)` executes the bounded declared acquisition through `prepare`/`submit`, or analyzes explicitly supplied data, and returns the actual selected Plan's attached Result.
- `accepts(result)` checks an independently expected scientific relation.
- `invalid_result(result)` changes a scientific field while preserving the concrete Result type and Plan identity. It must remain intrinsically valid.

`check_method` exercises selection and the supplied evaluation, checks the oracle and method-owned error information, and writes/reopens one temporary Result archive with explicit external Method loading. It tests that altered scientific result metadata fails against the same saved selection/data, then restores and rechecks the legal result. Normal archive byte limits still apply. No generic reference solve, array reconstruction or verification campaign is added. Round-trip comparison uses persistent scientific records and identities, not the object identity of reopened native handles. The explicit `accepts` callback still owns the independent relation, including array meaning when relevant.

The test-only identity case uses the existing Expectation Method and the relation `<psi|2I|psi>/<psi|psi>=2`, with no acquisition. A false oracle, ineffective falsifier, or disabled pair validation fails the checker. `CONFORMANT` means this one explicit case passed; it does not certify arbitrary callbacks, all inputs, physical accuracy, provider support or general execution cost. Use an explicit case whose cost and independent oracle you understand.

## API owners

::: nwqlib.algorithms.authoring.MethodCase

::: nwqlib.algorithms.authoring.check_method

::: nwqlib.algorithms.registry.AlgorithmRegistry

::: nwqlib.algorithms.registry.Registration

## Supported protected extension hooks

The following existing underscore-prefixed hooks are supported for Method and Result implementations. Their role is to bind and validate selected science; calling them does not authorize a second computation or a different archive format.

| Owner and exact hook | Implementation obligation |
| --- | --- |
| `Plan._bind(self, *, blocks=(), **native)` | Bind the actual selected block handles and named live inputs once; return that Plan. Restore the same bindings on load instead of replanning. |
| `Result._attach(self, plan, data)` | Attach the original Plan and RunData after checking identity, observation membership and `validate_plan`. It returns the same Result and refuses rebinding. |
| `Result._validate_common_plan(self, plan, plan_type)` | Call from a concrete `validate_plan` to check exact Plan type, Plan identity and recorded Method lineage, then check the Method-specific scientific relation. |
| `Result._summary_lines(self)` | Return a finite sequence of readable lines from existing scalar records. No acquisition, reference solve, artifact materialization or reanalysis is permitted. |

`result_type`, `save_archive(plan, files)`, and `load_archive(saved, files)` are provided by a persistable Method. They are not callable archive stubs on the base Method. The `files` argument is the supplied archive capability, not a filename: its `write_plan`/`read_plan`, input/output and selected-block operations keep actual native bindings, and writes are charged to their normal byte caps. The reference Hadamard Method shows how its selected blocks are written and restored through `nwqlib.blocks._archive.write_blocks` and `read_blocks`. This protected serializer supports the existing trusted block factories, not arbitrary Python callables. Use the passed `ArchiveFiles` object rather than constructing a parallel serializer. Cache loaders must register every file dependency through `files.read_path(name)` during restore, including files whose contents stay lazy. Registration reads only file metadata, and a missing file raises there. It does not scan the payload. Reopening protects registered dependencies before reclaiming unpublished files in the reserved cache namespace.

A Method selecting shots from an Accuracy request additionally supplies `sampling_shots(self, problem, *, output, accuracy, execution, shots)` and returns the selected shot count. This optional hook runs before planning. Omission means automatic shot selection is unavailable; explicitly supplied shots and a method-specific adaptive controller keep their own population contracts. Do not add placeholder archive or sampling methods just to make capability inspection succeed. `recover_analysis` is a separate explicit continuation hook for a saved interrupted analysis, not permission to resubmit earlier acquisitions.
