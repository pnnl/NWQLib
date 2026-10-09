# Release notes

## v1.0.2.post1 (2026-10-08)

Relative to 1.0.2, this maintenance release tightens sufficient error bounds, improves scalar range handling and raises five planning limits. Saved format labels are unchanged. Replanning can select different step counts, polynomial degrees, census variants and content identities. Loading a saved selection preserves its stored choices.

- Periodic LCHS uses the matching-operator Strang coefficient for its emitted factor order. Pure diffusion on one or two system qubits has zero ideal product-formula synthesis error and selects one step. For larger grids its coefficient is no larger than the previous norm-cubed bound. It bounds the weighted finite-branch synthesis component. Kernel, quadrature, angle and execution components need their own error evidence.
- The pair-only `relaxed_prefix` Pauli bound uses weighted anticommutation degrees to limit nested mass. It encloses the full Pauli-triangle expression and is no larger than the previous suffix relaxation. Its work and live-storage charges include the added degree passes and pair contraction. Because the relaxed count includes its degree passes, a configuration near `max_work` or `max_bytes` that 1.0.2 planned with `relaxed_prefix` can be refused. The full triangle expression can still exceed the actual evolution error.
- The normalized-expectation error bridge uses the sharp factor `2*w*delta/r`, for a complete supplied physical-vector budget `delta < r` and an observable norm bound `w`. Scale composition prevents avoidable intermediate overflow and false zero from underflow. A positive result outside the representable range has unavailable error evidence. This scalar propagation uses ordinary floating-point rounding, and unknown input components remain unknown. A new analysis of saved LCHS observations uses the factor-2 normalized-expectation bridge.
- Jacobi–Anger selection uses a real-argument Bessel remainder and chooses the smallest usable degree within its existing finite horizon, then the smallest computed complete tail and the earliest terminal. The search examines at most 202 terminals. A lower degree can have a larger reported tail. Tail evaluation has ordinary floating-point premises. If a selected polynomial's tail certificate is unrepresentable, the polynomial remains available with unknown tail evidence. Phase fitting and its convergence scope are unchanged.
- `AdmissionLimits.max_definitions` increases from 16,384 to 1,000,000, `FixedGCIM.max_admission_steps` from 1,000,000 to 10,000,000, `LCHS.max_select_work` from 100,000,000 to 1,000,000,000, `LCHS.max_dense_select_slots` from 256 to 4,096, and `LCHSRefinement.max_structural_work` from 100,000,000 to 1,000,000,000. Corresponding helper defaults agree. These limits allow more declarations, checking work, SELECT slots or counted construction work. Independent byte, native and execution limits still apply, and a planning count does not establish feasible execution.
- QHD's pre-expansion estimate is described as a nominal polynomial-skeleton charge. Its term-count proof covers constants, symbols, addition, multiplication and nonnegative integer powers. General SymPy rewriting, coefficient storage, symbolic temporaries and `Sum`/`Product` range work remain outside that proof. The symbolic decomposition and its scientific semantics are unchanged.
- Mathematical and algorithm documentation distinguishes quadrature-component looseness from total finite-action error, a hypothetical shot-based Gram cutoff from sampled data, and Low–Somma node count from coefficient mass and postselection probability.

Package and citation source versions are `1.0.2.post1`, with the project concept DOI.

## v1.0.2 (2026-10-07)

Relative to 1.0.1.post1, this release simplifies the library's internals, rewrites its documentation, and changes ten saved formats and the public names, arguments, fields and keys listed below.

**Saved data**

- The saved formats are `qls/7`, `qpe/<estimator>/8`, `lchs/10`, `expectation/6`, `fixed_gcim/6`, `adapt/7`, `lanczos/7`, `qhd/8`, `nwqlib.run/20` and `nwqlib.result/13`. This release cannot load archives written under the earlier labels of these formats. The format labels of QHD refinement and augmented-Lagrangian directories are unchanged, but loading or resuming such a directory written by an earlier release is refused where it opens a nested Run or Result saved under an earlier label. A Program, evidence options, a `QasmWriteReceipt`, an `IonQBackend` configuration, a profile or a forecast saved as standalone JSON by an earlier release is refused by record validation. Keep the writer's NWQLib version available to read old results or finish pending runs, or replan and rerun with this release to create new archives.

**Workflow and execution**

- `load_run` no longer requires `backend`. When it is omitted or `None`, `load_run` reopens a classical Run and a Run on the default `AerBackend()`. A Run saved with any other backend still needs that backend, and a backend that does not match the saved configuration is refused with a message that names the saved configuration and what to pass.
- `ObservationChunk` has no `readout()`. Read `chunk.observation`.
- `nwqlib.execution.prepare_experiment` no longer takes `reduction_allowance`. An extension sets the work allowance of its reductions through its Method's `reduction_allowance` hook.
- `python -m nwqlib.cli` no longer runs the command line. Use `python -m nwqlib` or the `nwqlib` console script.
- `check-method` and `check_method` no longer write a wrong Result into a temporary saved archive to show that it is refused. They still run the in-memory wrong-answer check and the save-and-load round trip.
- `third_party_registrations()` no longer takes `entries=`. It discovers the installed entry points as before.
- `ReducerOutput` no longer takes `index_width` and has no `payload_bytes`, and `nwqlib.core.planning.point_array_reservation` is removed. These have no replacement. Registered reducers publish JSON values sized by dtype and shape.
- `Measure` has no `basis` field. Measurements are in the computational basis, and a basis change is a preceding block. A Plan whose Program contains a `Measure` therefore has a different `content_id` from the same Plan built by 1.0.1.post1.
- `ProjectedVerificationOptions`, `EnergyShiftOptions` and `NumberSectorOptions` no longer take `source`. Each class fixes its own source, so these options have different `content_id` values from those of 1.0.1.post1.

**QASM export and import**

- `materialize_qasm3_file` refuses a `QasmWriteReceipt` whose `construction_id` differs from the construction's `content_id`, before it opens the file. The message names both content hashes and asks for the `QasmWriteReceipt` that the writer returned for this construction.
- `QasmMaterializationBudget` no longer takes `required_max_native_bytes` or `required_max_peak_rss`, whose only accepted value was `None`. `QasmMaterialization` and `QasmWriteReceipt` no longer have `native_bytes` or `peak_rss`, which were always `None`. The importer reports no memory bound.

**Eigenvalues and phases (QPE, Lanczos, GCiM)**

- Classical QPE Plans no longer repeat the work and byte allowance of their classical routine as `numerical_work` and `numerical_payload` in `plan.reconstruction`. Read them from `plan.construction.kernels[0].invocation_work` and `plan.construction.kernels[0].workspace[0].bytes`. A quantum QPE Plan has no entry in `plan.construction.kernels` and recorded 0 in both fields.
- A supplied initial state that `ADAPT`, `Lanczos` or `FixedGCIM` pads to a power-of-two dimension now needs `max_bytes` of at least 64 bytes per padded entry, plus the supplied state's stored size, plus 65,536 bytes. Previously it needed 32 bytes per padded entry. At the default 10 GB limit, only supplied states with more than 2^26 entries are affected, and only `ADAPT` in classical execution pads a supplied state at that size.
- The refusal of a saved or reloaded `FixedGCIM` pencil whose overlap filter does not match its selected observations now reads "pencil overlap filter differs from its selected observations", in place of "pencil sampling policy differs from its selected observations".

**Linear systems (QLS)**

- QLS Plans no longer store `encoding_id`, `preparation_id` and `encoding_operator` in `plan.reconstruction`. A quantum Plan names its selected encoding and preparation in `plan.construction.selections`, and a classical Plan's encoding follows from `plan.problem` and `plan.method`.
- QLS work records (`plan.reconstruction.work`) no longer list `matrix_products`, which was always 0, in classical and quantum Plans alike. A classical QLS Result's `applications[0].arguments` no longer include it either.
- A Plan whose block encoding the Method selects, as quantum QLS does by default, now has the same `content_id` whenever it is built from the same inputs, in any process. An encoding that you wrap with `select_block_encoding` gets a new random label at each call, so Plans built from separate calls have different `content_id` values.

**Linear dynamics (LCHS)**

- `nwqlib.algorithms.lchs` no longer exports `resolve_hamiltonian_evolution_backend` or `LCHS_HAMILTONIAN_EVOLUTION_IMPLEMENTATIONS`. The `LCHS.hamiltonian_evolution_backend` field chooses the backend of each branch (`"dense_exact"`, `"trotter"`, `"trotter_error_budgeted"` or `"qsp_block_encoding"`), as its description states.
- `LCHSAnalysis` no longer has the `references` field, which was always `"not_run"`. Reference checks are explicit verification calls.
- `LCHSCoefficientPlan.record()` no longer returns `lchs_kernel_quadrature_compatibility` or `lchs_kernel_quadrature_certificate_status`. Read the certificate status from `solution_error_certificate_status`. The compatibility key recorded only that an allowed provider combination was selected and has no replacement.
- `LCHSQuadraturePlan` and `LCHSQuadratureData` no longer have `effective_range_k`, which equaled `range_k` for every built-in quadrature rule, and `lchs_quadrature_summary` no longer returns the matching `effective_K` key. Read `range_k`, or the summary's `K` key.
- `LCHSProductFormulaSelectPlan` no longer has `max_step_count`. Read `repetitions`, which holds the same value.
- `generate_lchs_product_formula_select_plan` takes the quadrature data as `quadrature=` in place of `matrix=` and `_quadrature_plan=`. Build that data with `generate_lchs_quadrature(matrix=..., final_time=..., method=...)`. The function still refuses a dimension that is not a power of two.
- `nwqlib.algorithms.lchs.time_independent_common` no longer exports `complex_to_dict` or `is_power_of_two`. They have no public replacement.
- The LCHS exact-readout work refusal gives the requested work, the limit and the minimum sufficient `LCHS.max_readout_work`, without the clauses about planning units and completed reduction units.

**Optimization (QHD)**

- QHD archives preserve unevaluated SymPy `Sum` expressions through Result loading and refinement or augmented-Lagrangian continuation.
- `solve_augmented_lagrangian`, `plan_augmented_lagrangian`, `refine_box`, `constrained_grid_minimum` and the `ConstrainedQHDResult` constructor no longer raise a dedicated `TypeError` for an argument of the wrong type. Such an argument now fails where the layer first uses it, typically with an `AttributeError` that names the missing attribute. `refine_box` still raises `TypeError` for a problem that is not an `Optimization`.
- The description of `QHDVerification.max_bytes` states that, for observed counts or probabilities without a kept state, the `grid_minimum` charge omits histogram construction or loading, decoding and passing-weight storage at every readout width. This limit therefore does not bound the `minimum_success_mass` step.

**Subroutines**

- The undocumented functions `nwqlib.subroutines.lcu.lcu_preparation_implementation_metadata` and `nwqlib.subroutines.block_encoding.block_encoding_implementation_metadata` are removed. Every block encoding records its selected implementation under `BlockEncoding.metadata["implementation_metadata"]`, and every LCU circuit under `LCUCircuit.preparation_metadata["implementation_metadata"]`. The registries `BLOCK_ENCODING_IMPLEMENTATIONS` and `LCU_PREPARATION_IMPLEMENTATIONS` list each selectable implementation's record.
- `coefficient_arithmetic` is removed from `TrotterStepSelection` (its attribute and `to_dict()` key), from QPE reconstructions and from LCHS Trotter node records. NWQLib evaluates the coefficient one way, which the records' descriptions now state. `bound_variant` identifies the enclosed expression, and QPE reconstructions keep the exact coefficient in `bound_coefficient`.
- `nwqlib.subroutines.trotterization.trotter_error_bound` is removed. Call `evaluate_trotter_bound(hamiltonian, time=t, steps=1, order=o)`, which returns the same bound under the same limits.
- Banded block encodings no longer repeat the band offsets as `metadata["band_offsets"]` or set `metadata["toeplitz"]`, which was always `True` and has no replacement. Read the offsets from `metadata["band_specification"]["offsets"]`.
- `resolve_number_projector_lowering` is removed from `nwqlib.subroutines.hamiltonian_evolution.pauli_evolution`. Call `structured_number_projector_provider(support_size)` from the same module, which returns the provider record that the removed function wrapped, with its CX count under `"cx"`.
- `nwqlib.subroutines.qsp.evolution` no longer exports `_reflection_about_ancilla_zero`, the reflection `I - 2|0><0|` on the ancilla register. It has no public replacement. `nwqlib.blocks.select_zero_reflection` builds the reflection of the opposite sign, `2|0><0| - I`.

**Backends, device profiles and forecasts**

- Resource forecasts compare by their declared fields, so saving and loading a `PlanEstimate` preserves equality regardless of cached lookups. `PlanEstimate` and `AssessmentContext` also allow the other object's equality comparison when its type differs.
- `IonQBackend` no longer takes `debiasing`. Requests are sent with debiasing disabled, as before.
- `TimeModel` and `TimePrediction` no longer have `form`. The model is the linear model that its coefficients, scope, domain and uncertainty describe.

**Documentation and examples**

- The guides, the API reference and the docstrings are rewritten to describe the behavior of this release, and the API reference renders the records it names.
- The `resource_estimation_at_scale` notebook is regenerated and executed with this release.

Package and citation versions are synchronized at `1.0.2`.

## v1.0.1.post1 (2026-10-04)

This release simplifies validation and saved-data loading, and removes forwarding APIs and their supporting code.

- Removes saved-record hash verification, unused ADAPT compiler hashes, QASM file digests and separate QPE/QLS Program checks. QHD loading compares the restored expressions directly with their saved descriptions.
- QASM import applies construction work limits to unchanged output from the same writer call and checks the file's actual size before reading.
- Use `plan.to_record()` and `run.artifacts.get(manifest)` in place of `ArchiveFiles.write_plan` and `Run.hydrate`. `read_report` returns the saved metadata without the `metadata_validation` entry.
- `nwqlib.backends.capability_set`, `require_backend_capabilities` and `BackendTarget.missing_capabilities` are removed. Use a tuple of capabilities when constructing a target and `frozenset(required) - frozenset(target.capabilities or ())` to find missing capabilities. The adapter still checks the instructions and readout it can execute.
- `QasmWriteReceipt.digest` and `QasmPrefix.digest` are removed. `bytes_written` measures size, not content identity. Remove the `digest` key from separately stored receipt JSON before calling `QasmWriteReceipt.model_validate`.
- The shared archive formats are `nwqlib.run/19` and `nwqlib.result/12`. This release cannot load earlier shared formats, including those written by v1.0.1. Keep the writer's NWQLib version available to read old results or finish pending runs, or replan and rerun with this release to create new archives. This also applies to Result and Run subfolders of QHD refinement and augmented-Lagrangian archives. Package and citation versions are synchronized at `1.0.1.post1`.

## v1.0.1 (2026-10-03)

This release simplifies tests and the validation code they support. It removes 107 collected test cases while keeping independent checks of numerical results, resource limits and normal save, load and resume workflows.

- QLS uses the shared Program validation. Preparation and native lowering check their own work limits before constructing circuits, without a separate planning-time simulation of those checks.
- Several entry points stop explicitly rejecting incorrect shots types, unknown LCHS provider parameters, unknown analysis settings and unsupported chemistry reference names. Use the argument names and types listed in each Method's guide.
- GCiM analysis, Pauli grouping, QPE census, Trotter bounds and periodic LCHS stop computing additional bounds solely to recommend exact retry limits. Their actual work and memory checks remain in place.
- Result and Run loaders restore the current saved data with fewer repeated consistency checks. Save-time scientific associations, current format and array-layout checks, explicit Method selection and path restrictions continue to apply. Metadata reports expose the Result's association with its observations as `metadata_validation["result_observation_association"]`.
- The standalone analytical specimen, its fixtures and its CI checks are removed. Its independent Lanczos moments and projected pencil are checked through the production readout and reconstruction functions.
- QPE sampling tests compare directly with an independently known eigenvalue. Controlled-gate cost tests share circuit construction while checking the distinct gate prices. Other tests drop fixed strategy choices, duplicate cases, exact SDK-version assertions and diagnostics for obvious input misuse. Helpers and documentation used only by those tests are removed with them.

## v1.0.0.post3 (2026-10-02)

Counts-based QHD box refinement and augmented-Lagrangian runs with refinement now include selected region mass lower bounds in `print(result)` and `result.report()`. The default 95% confidence covers the configured run, and the mass is conditional on valid decoding of the level's backend-sampled distribution. The report compares Hoeffding and one-sided Clopper–Pearson bounds after allocating half the failure budget to each method.

`BoxRefinementResult.report()` now defaults to `failure_probability=0.05`, and `ConstrainedQHDResult.report()` accepts the same parameter. Pass `None` to omit the statistical report. Each completed level records `region_axis_counts` and `region_count`. The outer archive formats are `qhd.refinement/4` and `qhd.constrained/4`, and the controller formats are `qhd.refinement_run/6` and `qhd.constrained_run/5`. Earlier outer formats cannot be loaded or resumed by these readers.

## v1.0.0.post2 (2026-10-01)

This release changes documentation, one error message and repository settings. Code that runs with v1.0.0.post1 runs unchanged, with the same results.

- The documentation is reorganized by reader, and old page addresses still work.
- The API reference is rewritten, with a summary, parameters and examples for its entries.
- The descriptions in the configuration schemas that `nwqlib options` prints have changed.
- The example notebooks use the documentation's terms, and their outputs come from a new run.
- The import error of `lcu_state_preparation="mps_circuit"` gives the install command of `scikit_tt`, which is not on PyPI.
- The CI, Full CI, Mutation Probes and Nexus Offline Qualification workflows give their `GITHUB_TOKEN` read-only access to the repository contents.
- Two test assertions that failed on Python 3.14 are removed. Both checked deeply nested JSON in a hand-edited `result.json`.

## v1.0.0.post1 (2026-09-30)

The README is rewritten for the PyPI page. Nothing else changed.

## v1.0.0 (2026-09-30)

NWQLib applies quantum algorithms to scientific inputs and returns the computed quantities with their methods, resources and evidence. This is the first public release. The pages linked below hold the details.

**Algorithms**

- Linear systems: [QLS](algorithms/qls.md) with the QSVT inverse polynomial and the Dalzell kernel shortcut.
- Linear dynamics: [LCHS](algorithms/lchs.md) with the An–Childs–Lin, Cauchy and Low–Somma kernels on product-formula and QSP block-encoding routes.
- Eigenvalues and phases: [Chebyshev Lanczos](algorithms/lanczos.md), [fixed and adaptive GCiM](algorithms/gcim.md), and [QPE](algorithms/qpe.md) with QCELS, RWPE, SPE and RFE.
- Expectation values of finite Pauli sums: [Expectation](algorithms/expectation.md).
- Optimization over a box: [QHD](algorithms/qhd.md) in one-hot and binary encodings, with box refinement and an augmented-Lagrangian route for equality and inequality constraints.

**Subroutines** ([API](api/index.md)): block encodings, LCU, QSP and QSVT phase selection, coherent QPE, Hamiltonian evolution, Trotterization with error budgets and a Pauli census, state preparation including matrix product states, fermionic pools, and Pauli decomposition.

**End to end**

- One workflow for every method: `plan` selects the construction, `estimate` and `compare` price it before any circuit is built, `solve` runs it, and `prepare` and `submit` run it as a Run that can be saved, reopened and continued ([Plan, compare and solve](scientist.md), [Run on a backend](prepared_execution.md), [Continue an interrupted run](run_archives.md)).
- Resource estimates come from the gate-count laws of the selected construction, are checked against compiled circuits at small size, and extend to sizes no simulator holds ([Estimate resources](resources.md)).
- Every limit on work, memory and stored data is checked before the work starts, and a refusal names the parameter and the amount ([Supply inputs](inputs.md)).
- Results carry their error evidence and references. Saved Results and Runs reload without planning or measuring again, a reopened Run can continue its unfinished work, and `result.analyze` reanalyzes the saved observations with new settings without new measurements ([Save, load and reanalyze results](saved_evidence.md), [Check accuracy and verify a result](verification.md)).
- Backends: local Aer, NWQ-Sim with Slurm submission, IBM Runtime, IonQ, Quantinuum Nexus, and QASM export ([Choose a backend](backends.md)). A command line lists the methods and prints their cards ([Use the command line](cli.md)).

**Documentation**: a [quickstart](quickstart.md), the user guides, one guide per algorithm with its source map from code to paper equations, the [mathematics page](mathematics.md) with the proofs and resource laws, the backend guides, the [API reference](api/index.md), and the development pages from the [contribution policy](FRAMEWORK.md) to the [code tour](CODE_TOUR.md) and the [engineering constants](ENGINEERING_CONSTANTS.md).

**Examples** ([overview](examples.md)): eight executed notebooks. Four introductions cover a linear system, linear dynamics, eigenvalues and optimization. Three paper-derived studies cover collision relaxation with QLS, heat flow with a boundary penalty with LCHS, and constrained optimization with QHD. One notebook plans five problems at 80 to 100 qubits and prices them without simulating a circuit.
