# Release notes

## v1.0.3.post2 (unreleased)

Relative to v1.0.3.post1, this release adds QFD, an eigenvalue method in a basis of time-evolved states, and new measurement, shot-allocation and basis-growth options for GCiM. It changes the default measurement of sampled ADAPT, the default pair preparation and quadratures of FixedGCIM and ADAPT, the default memory limit of the local simulator, the shared Run and Result archive formats and the saving of fermionic operators. It cannot load Run or Result archives written by v1.0.3.post1.

### QFD

`QFD` (`nwqlib.algorithms.QFD`) estimates the smallest eigenvalue of a Hermitian operator $A$ in the basis $V^{p}|\psi\rangle$ of integer powers $p$ of one fixed product-formula step $V$ applied to a reference state $|\psi\rangle$. It solves the Hamiltonian pencil of Parrish and McMahon, arXiv:1909.08925v1, Eqs. (5)–(8), with the measurement and projected solve of FixedGCIM, by exact readout or with grouped shots. You replace $A$, the reference, the time step and the powers. By default $V$ is built from the Pauli terms of $A$, and `evolution="fermionic_blocks"` builds it from complete number-conserving fermionic blocks of a separate generator. `chemistry_evolution_terms` (in `nwqlib.algorithms.qfd`) builds such a generator from the integrals of `build_gcim_chemistry_problem` data. The [QFD guide](algorithms/qfd.md) describes the workflow, its limits and what the result does not establish.

### GCiM measurement and its defaults

`PencilAcquisition`, passed as `acquisition=` to FixedGCIM, ADAPT and QFD, chooses how $H$ and $S$ are measured ([How matrix elements are obtained](algorithms/gcim.md#how-matrix-elements-are-obtained)). Its defaults change what an existing sampled script measures:

- Sampled ADAPT measures grouped joint states by default (`matrix_measurement="grouped"`), with one setting per qubit-wise commuting group and part, as FixedGCIM does. v1.0.3.post1 measured one Hadamard test per Pauli term and part. `PencilAcquisition(matrix_measurement="termwise", pair_preparation="controlled")` keeps that measurement as a comparison.
- `quadratures="auto"` measures only the real parts of the off-diagonal pairs when a proof from the stored inputs shows that $A$ and every state the method can prepare are real, and records the imaginary parts as exact zeros. Otherwise it measures both parts, which v1.0.3.post1 always did. Real-only measurement does not show that the imaginary parts are zero on a noisy device.
- `pair_preparation="prefix"` runs the blocks that both preparations of a pair begin with once without control and controls only the rest, with the same joint state. v1.0.3.post1 controlled both complete preparations of every pair, which `"controlled"` still does.

For ADAPT, `pair_preparation="vacuum_reference"` and `"vacuum_restore"` control only the reference preparation, when planning can show from the Pauli terms of every pool generator that its exponential maps the all-zero state to itself up to a phase. FixedGCIM and QFD refuse both values in quantum execution unless the Plan measures nothing.

### Shot allocation from a pilot stage

`sampling=PencilSampling(total_shots=..., pilot_fraction=...)` of FixedGCIM, ADAPT and QFD splits a shot budget into a uniform pilot stage and a main stage whose shots per setting the pilot pencil decides. Only the main-stage measurements form the final $H$ and $S$, the pilot shots count toward the total, and `result.shot_allocation` records the decision ([Pilot and main shot allocation](algorithms/gcim.md#pilot-and-main-shot-allocation)). For ADAPT the two stages measure the basis that growth ended with, with a budget separate from the `shots=` of the growth. Without `sampling`, every setting gets `shots` shots, as before.

### ADAPT basis growth

`ADAPT(basis_growth=BoundedRitzGrowth(max_chain_length=L))` grows a basis of generator chains of at most $L$ generators, one chain per round, chosen by a score from the current Ritz pair, and `basis_depth_limit` bounds the CX count of each chain ([Bounded Ritz growth](algorithms/gcim.md#bounded-ritz-growth)). It changes the trial space, so its energies are those of a different projected problem from the default growth. With shots, an extension is scored only when it passes an empirical eligibility rule on the sampled overlaps, and growth stops with `noise_inconclusive` when none of a round's extensions passes. The default growth, `basis_growth=None`, is unchanged.

### Circuit depth on IBM Runtime

`max_native_two_qubit_depth=D` of FixedGCIM, ADAPT and QFD refuses, before its submission, every circuit whose two-qubit depth after compilation for the device exceeds $D$ or is not qualified ([Native two-qubit depth](ibm.md#native-two-qubit-depth)). Only IBM Runtime computes this depth, and it qualifies it only for counts circuits with dynamical decoupling and twirling off, under qiskit-ibm-runtime 0.49.0. With any other installed version a depth limit refuses every IBM circuit, so after an upgrade of qiskit-ibm-runtime the limit is accepted again only once the re-check in [Maintenance](MAINTENANCE.md#dependency-follow-ups) has qualified the new version. The default `None` sets no limit.

### Other changes

- The default `ExecutionLimits.simulator_memory_mb` is 8192 MiB, raised from 1024. It is an upper limit that the memory needs of a local simulation are checked against.
- `ProjectedPencil.offset_free_eigenvalues` records the lowest one or two Ritz values before the identity coefficient is added back ([Read the result](algorithms/gcim.md#inspect-numerical-diagnostics-and-error-bounds)).
- `nwqlib.estimate(plan)` of an ADAPT Plan, and of a FixedGCIM or QFD Plan with `sampling`, reports the counts that depend on the iteration or on the pilot as unavailable, with the reason that the new `Method.query_catalog_reason()` returns ([ADAPT queries](algorithms/gcim.md#adapt-queries)).
- The shared archive now saves an ordered fermionic operator, such as one from `ingest_fermion`, in its own arrays. v1.0.3.post1 raised `ValueError` when an archive had to save one.
- The `coefficient_cutoff` documentation of `build_gcim_chemistry_problem` now states that OpenFermion's operator addition, which its Jordan–Wigner transform also uses, applies OpenFermion's own tolerance, in OpenFermion 1.8.1 deleting a term whose running coefficient has absolute value below 1e-8. A stored Pauli coefficient can therefore differ from the sum of its integral contributions by those deletions.
- The new example notebook `gcim_qfd_shallow_workflows` compares the measurement and basis options of GCiM and QFD on H2 and a stretched H4 chain and plans a 26-qubit model. The eigenvalue notebook adds QFD and the grouped and termwise finite-shot counts, and the resource notebook gives the real-only GCiM settings, shots and CX bound. The stored outputs of every example notebook come from this version, including the 8192 MiB memory default where a notebook prints it.

### Formats

The shared archive formats are `nwqlib.run/22` and `nwqlib.result/15`, and this release cannot load `nwqlib.run/21` or `nwqlib.result/14` archives written by v1.0.3.post1. Keep v1.0.3.post1 available to read those archives or finish their pending runs, or replan and rerun with this release. The Plan and Result content hashes of existing FixedGCIM and ADAPT configurations change, because their records gain the new fields with their defaults, and no reader for the earlier hashes is kept.

The method archives embedded in these files change from `fixed_gcim/6` to `fixed_gcim/7` for FixedGCIM and from `adapt/7` to `adapt/8` for ADAPT. The new QFD method archive uses `qfd/1`.

Package and citation source versions are `1.0.3.post2`.

## v1.0.3.post1 (2026-10-21)

Relative to v1.0.3, this maintenance release changes the test suite, the documentation, two extension interfaces, the shared Run and Result archive formats, the error for a missing compiler extra, the CX and rotation counts that follow Qiskit's gate definitions, QHD's projector choice at 7 qubits and the minimum Qiskit and qiskit-aer versions. It cannot load Run or Result archives written by v1.0.3. Default option values are unchanged.

The test suite holds 1,501 tests, against 3,259 in v1.0.3. It keeps the checks of computed quantities against independent references, the resource limits that refuse work before it starts, the normal save, load and resume paths, the checks against installed SDKs, the example notebooks and every test that the mutation probes or the documentation name. Tests were removed when they were judged to check only message wording, printed text, internal records and helper calls, the rejection of misuse such as unknown options or edited saved files, or a case that another current test covers.

New tests follow a user's workflow through the public API:

- The Python examples on the README and documentation pages run as written. On 20 of these pages, the values the examples compute are compared with independent computations.
- The `nwqlib` command lists the registered Methods, reports the options each Method requires as its constructor does, and lists installed third-party Methods without importing them.
- LCHS and QLS solutions for a periodic stencil agree with the dense matrix exponential and linear solve within their stated error bounds. On Aer, a QLS solution scales with the magnitude, sign and phase of the right-hand side, and its physical scalars lie within their error bounds. A two-qubit Pauli operator `A` with a supplied `kappa` gives the same solution as its dense matrix, sampled indices follow the solution's distribution, and each accuracy setting in a comparison meets its own tolerance.
- A classical LCHS solution with a constant source lies within its refined error bound of the closed-form solution, and a sampled expectation value lies within its reported sampling radius.
- The `Solution` of a LinearSystem or LinearDynamics carries the Problem's unit, and its `NormSquared` carries the square of that unit. An Expectation's unit labels its normalized expectation, so a `QuadraticForm` output has an unspecified unit unless it declares one.
- QCELS estimates the H2 ground energy from a Pauli Hamiltonian with and without a constant offset. Lanczos keeps only the reachable directions when the requested Krylov dimension exceeds the subspace that the initial state generates. A saved sampled Lanczos result reanalyzes with its shot counts, and `scan` by requested shots and `logical_width` keeps both Plans of a shots-width trade-off in its nondominated set.

The LCHS guide states that `A` must be a dense matrix or a periodic stencil, and how to pass a SciPy sparse matrix or a Qiskit `SparsePauliOp` in dense form. The pages on comparing and ranking Plans state that ranking sampled Plans by width needs `ResourceContext(batch_schedule="serial")`. The Units section states that a unit-normalized StateVector is dimensionless for every Problem. The maintenance guide describes the documentation example check.

The documentation wording has been revised.

Two extension interfaces and the shared archive formats change:

- NWQLib passes `settings` to the Method's `prepare(plan, *, run, settings)` hook on every call, so an override of that hook must accept the keyword. In v1.0.3, NWQLib passed `settings` only for `"all"`, and the inherited `Method.prepare_all_refusal` refused `"all"` for an override without the keyword. The inherited `Method.prepare_all_refusal` now returns `None`, so a Method that cannot prepare every setting in advance overrides it, as [Adaptive Methods](algorithm_protocol.md#adaptive-methods) describes.
- The `prepares_from_templates` attribute of backend adapters and the `template_seed` fields of `NativePreparation` and `PreparedArtifact` are removed. No backend included with NWQLib set that attribute. An adapter that set it no longer receives the `construction_id`, `backend_context` and `template_seed` keyword arguments at `prepare` or `prepare_local`. These arguments have no replacement.
- Because preparation records change, the shared archive formats are `nwqlib.run/21` and `nwqlib.result/14`. This release cannot load earlier shared formats, including those written by v1.0.3. Keep the writer's NWQLib version available to read old results or finish pending runs, or replan and rerun with this release to create new archives. This also applies to Result and Run subfolders of QHD refinement and augmented-Lagrangian archives.

Without the `nwqec` extra, `compile_logical` raises `ModuleNotFoundError` with the install command before it starts its child process, and so does `estimate_physical` without the `qre` extra. In v1.0.3, the child process failed instead, and the error arrived as a `RuntimeError` that contained the child's traceback.

Every extra that installs Qiskit now requires Qiskit 2.6.0 or later, and the `aer` extra also requires qiskit-aer 0.18.0 or later. Upgrade with the extras you use, for example `python -m pip install -U "nwqlib[aer]"`, so that pip also upgrades Qiskit and qiskit-aer. With older Qiskit or qiskit-aer releases installed, the counts described below do not match the installed synthesis, and exact Aer readouts list "unchecked qiskit-aer version". The validated environment in `docs/ENVIRONMENT_LOCK.txt` changes only these two packages.

Without ancillas, Qiskit 2.6.0 synthesizes a multi-controlled X gate with 5 to 32 controls, and a multi-controlled phase gate with 5 or more controls, by the construction of da Silva and Park ([Phys. Rev. A 106, 042602](https://doi.org/10.1103/PhysRevA.106.042602)), with `4k**2 - 4k + 2` CX for k controls. Qiskit 2.5.2 used more CX for these gates, for example 2,328 instead of 1,522 for an X gate with 20 controls. NWQLib's resource counts of the circuits that Qiskit builds follow the new definitions:

- The CX table of multi-controlled X gates, `MCX_CX_BY_CONTROLS`, has the new counts for 5 to 32 controls. The QLS CX estimates of projector flips and of the `Q_b'` and `A_t` predicates read it, and so do the LCHS SELECT bounds.
- The LCHS CX bounds count the CX of controlled phases and controlled U gates by the new definitions. The projector flip of a QSP pass with k block ancillas, controlled on its parity qubit, costs `36k**2 - 34k + 20` CX from k = 5, for example 750 at k = 5 against 746 with Qiskit 2.5.2, and 35,796 against 65,768 at k = 32.
- QHD's structured number projector uses Qiskit's multi-controlled phase from 7 qubits, where it costs 122 CX against 126 for the phase diagonal, instead of from 8 qubits. Its CX and rotation counts follow the new construction. At 8 qubits it costs 170 CX against 220 with Qiskit 2.5.2, and for a generic angle it has 135 arbitrary rotations and 44 exact T gates, against 147 and 116.

Qiskit 2.6.0 also corrects the global phase that the two-qubit peephole optimization of optimization levels 2 and 3 changed by pi for some CX patterns. The example notebooks were executed again with Qiskit 2.6.0 and qiskit-aer 0.18.0.

The probability windows of exact Aer readouts name qiskit-aer 0.18.0 as the checked release. Its statevector, fusion and gate-matrix sources are those of 0.17.2, so the windows themselves are unchanged.

On Linux, qiskit-aer 0.18.0 and PySCF bundle different builds of the OpenMP runtime, and a Python process that loads both can fail when it loads the second, with "cannot allocate memory in static TLS block". On the Ubuntu 24.04 hosts of NWQLib's CI, such a process failed with qiskit-aer 0.18.0 and ran with qiskit-aer 0.17.2. A process that uses the `aer` and `chemistry` extras together on Linux, such as the eigenvalue notebook, should therefore start with the OpenMP runtime of PySCF preloaded, as [Dependency issues](dependency_issues.md#openmp-runtime-with-pyscf-on-linux) describes.

Package and citation source versions are `1.0.3.post1`, with the project concept DOI.

## v1.0.3 (2026-10-19)

Relative to v1.0.2, this release includes the mathematical and planning updates first published in `1.0.2.post1` and the complete BSD-2-Clause license with the PNNL/DOE disclaimer.

### License and distribution

The root `LICENSE` contains the Battelle copyright notice, the two BSD redistribution conditions, and the PNNL/DOE disclaimer. Source and wheel distributions carry this file unchanged. The publishing workflow compares the packaged license with the repository file byte for byte before uploading to PyPI. Zenodo metadata uses the explicit `bsd-2-clause` license identifier.

### Mathematical and planning changes

- Periodic LCHS uses a Strang error coefficient derived for the order in which its circuit applies the operators. When steps are selected from a synthesis tolerance, pure diffusion on one or two system qubits selects one step with zero ideal product-formula synthesis error. Other error components, including quadrature and execution, remain separate.
- The `relaxed_prefix` Pauli bound uses weighted anticommutation degrees to tighten the previous suffix relaxation. The work and memory it counts against the limits include the additional degree passes and contraction work. A plan close to `max_work` or `max_bytes` can therefore be refused where v1.0.2 accepted it.
- The normalized-expectation error bound changes from `4*w*delta/r` to the sharp `2*w*delta/r`, where `r` is the norm of the computed physical vector, `delta < r` is its complete error budget, and `w` bounds the observable norm. Scaled evaluation avoids intermediate overflow and false zero from underflow. An unrepresentable positive bound is reported as unavailable evidence.
- Jacobi–Anger selection uses a tighter real-argument Bessel remainder and chooses the smallest usable polynomial degree within its finite search horizon. Ties select the smaller computed complete tail, then the earlier terminal. The search examines at most 202 terminals. A lower degree can have a larger reported tail. When the only passing candidates have an analytic remainder that underflows to zero, the polynomial remains available with unknown tail evidence. Phase fitting is unchanged.
- Five default planning limits increase as shown below. Independent byte and execution limits still apply, so passing the planning checks does not establish that a plan can be executed.

| Planning limit | v1.0.2 | v1.0.3 |
| --- | ---: | ---: |
| `AdmissionLimits.max_definitions` | 16,384 | 1,000,000 |
| `FixedGCIM.max_admission_steps` | 1,000,000 | 10,000,000 |
| `LCHS.max_select_work` | 100,000,000 | 1,000,000,000 |
| `LCHS.max_dense_select_slots` | 256 | 4,096 |
| `LCHSRefinement.max_structural_work` | 100,000,000 | 1,000,000,000 |

### Documentation and compatibility

The QHD documentation scopes the pre-expansion term-count bound to constants, symbols, addition, multiplication and nonnegative integer powers. General SymPy rewriting work and storage are outside that bound. The mathematics and algorithm guides clarify the distinction between component bounds and total error, and the LCHS introduction and resource-estimation notebooks have updated outputs.

Saved-format labels are unchanged from v1.0.2. Loading saved construction data preserves the choices recorded in it. Replanning can change step counts, polynomial degrees, Pauli-triangle bound variants (`bound_variant`) and content hashes. Reanalyzing saved LCHS observations uses the factor-2 normalized-expectation bound. Scalar error propagation and Bessel-tail evaluation use ordinary floating-point rounding.

## v1.0.2.post1 (2026-10-08)

Relative to 1.0.2, this maintenance release tightens sufficient error bounds, improves scalar range handling and raises five planning limits. Saved format labels are unchanged. Replanning can select different step counts, polynomial degrees, Pauli-triangle bound variants (`bound_variant`) and content hashes. Loading saved construction data preserves the choices recorded in it.

- Periodic LCHS uses the matching-operator Strang coefficient for the factor order that its circuit applies. Pure diffusion on one or two system qubits has zero ideal product-formula synthesis error and selects one step. For larger grids its coefficient is no larger than the previous norm-cubed bound. It bounds the weighted finite-branch synthesis component. Kernel, quadrature, angle and execution components need their own error evidence.
- The pair-only `relaxed_prefix` Pauli bound uses weighted anticommutation degrees to limit nested mass. It encloses the full Pauli-triangle expression and is no larger than the previous suffix relaxation. The work and live storage it counts against the limits include the added degree passes and pair contraction. Because the relaxed count includes its degree passes, a configuration near `max_work` or `max_bytes` that 1.0.2 planned with `relaxed_prefix` can be refused. The full triangle expression can still exceed the actual evolution error.
- The normalized-expectation error bound uses the sharp factor `2*w*delta/r`, for a complete supplied physical-vector budget `delta < r` and an observable norm bound `w`. Scale composition prevents avoidable intermediate overflow and false zero from underflow. A positive result outside the representable range has unavailable error evidence. This scalar propagation uses ordinary floating-point rounding, and unknown input components remain unknown. A new analysis of saved LCHS observations uses the factor-2 normalized-expectation bound.
- Jacobi–Anger selection uses a real-argument Bessel remainder and chooses the smallest usable degree within its existing finite horizon, then the smallest computed complete tail and the earliest terminal. The search examines at most 202 terminals. A lower degree can have a larger reported tail. Tail evaluation assumes ordinary floating-point rounding. If a selected polynomial's tail certificate is unrepresentable, the polynomial remains available with unknown tail evidence. Phase fitting and its convergence scope are unchanged.
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
- The description of `QHDVerification.max_bytes` states that, for observed counts or probabilities without a kept state, the bytes that `grid_minimum` counts against the limit omit histogram construction or loading, decoding and passing-weight storage at every readout width. This limit therefore does not bound the `minimum_success_mass` step.

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
- `QasmWriteReceipt.digest` and `QasmPrefix.digest` are removed. `bytes_written` measures size, not content identity. Remove the `digest` key from a `QasmWriteReceipt` saved separately as JSON before calling `QasmWriteReceipt.model_validate`.
- The shared archive formats are `nwqlib.run/19` and `nwqlib.result/12`. This release cannot load earlier shared formats, including those written by v1.0.1. Keep the writer's NWQLib version available to read old results or finish pending runs, or replan and rerun with this release to create new archives. This also applies to Result and Run subfolders of QHD refinement and augmented-Lagrangian archives. Package and citation versions are synchronized at `1.0.1.post1`.

## v1.0.1 (2026-10-03)

This release simplifies tests and the validation code they support. It removes 107 collected test cases while keeping independent checks of numerical results, resource limits and normal save, load and resume workflows.

- QLS uses the shared Program validation. Preparation and Qiskit circuit construction check their own work limits before any circuit is built, without a separate planning-time simulation of those checks.
- Several entry points stop explicitly rejecting incorrect shots types, unknown LCHS provider parameters, unknown analysis settings and unsupported chemistry reference names. Use the argument names and types listed in each Method's guide.
- GCiM analysis, Pauli grouping, the Pauli commutation tables of QPE, Trotter bounds and periodic LCHS stop computing additional bounds solely to recommend exact retry limits. Their work and memory checks remain in place.
- Result and Run loaders restore the current saved data with fewer repeated consistency checks. Save-time scientific associations, current format and array-layout checks, explicit Method selection and path restrictions continue to apply. Metadata reports expose the Result's association with its observations as `metadata_validation["result_observation_association"]`.
- The standalone analytical reference case, its test data and its CI checks are removed. Its independent Lanczos moments and projected pencil are checked through the library's own readout and reconstruction functions.
- QPE sampling tests compare directly with an independently known eigenvalue. Controlled-gate cost tests share circuit construction while checking the distinct gate costs. Other tests drop fixed strategy choices, duplicate cases, exact SDK-version assertions and diagnostics for obvious input misuse. Helpers and documentation used only by those tests are removed with them.

## v1.0.0.post3 (2026-10-02)

Counts-based QHD box refinement and augmented-Lagrangian runs with refinement now include selected region mass lower bounds in `print(result)` and `result.report()`. The default 95% confidence covers the configured run, and the mass is conditional on valid decoding of the level's backend-sampled distribution. The report compares Hoeffding and one-sided Clopper–Pearson bounds after allocating half the failure budget to each method.

`BoxRefinementResult.report()` now defaults to `failure_probability=0.05`, and `ConstrainedQHDResult.report()` accepts the same parameter. Pass `None` to omit the statistical report. Each completed level records `region_axis_counts` and `region_count`. The outer archive formats are `qhd.refinement/4` and `qhd.constrained/4`, and the formats of the outer record of a run directory (`controller.json`) are `qhd.refinement_run/6` and `qhd.constrained_run/5`. Earlier outer formats cannot be loaded or resumed by these readers.

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

**Subroutines** ([API](api/index.md)): block encodings, LCU, QSP and QSVT phase selection, coherent QPE, Hamiltonian evolution, Trotterization with error budgets and Pauli commutation tables, state preparation including matrix product states, fermionic pools, and Pauli decomposition.

**End to end**

- One workflow for every method: `plan` selects the construction, `estimate` and `compare` give its resource counts before any circuit is built, `solve` runs it, and `prepare` and `submit` run it as a Run that can be saved, reopened and continued ([Plan, compare and solve](scientist.md), [Run on a backend](prepared_execution.md), [Continue an interrupted run](run_archives.md)).
- Resource estimates come from the gate-count formulas of the selected construction, are checked against compiled circuits at small size, and extend to sizes no simulator holds ([Estimate resources](resources.md)).
- Every limit on work, memory and stored data is checked before the work starts, and a refusal names the parameter and the amount ([Supply inputs](inputs.md)).
- Results carry their error evidence and references. Saved Results and Runs reload without planning or measuring again, a reopened Run can continue its unfinished work, and `result.analyze` reanalyzes the saved observations with new settings without new measurements ([Save, load and reanalyze results](saved_evidence.md), [Check accuracy and verify a result](verification.md)).
- Backends: local Aer, NWQ-Sim with Slurm submission, IBM Runtime, IonQ, Quantinuum Nexus, and QASM export ([Choose a backend](backends.md)). A command line lists the methods and prints their cards ([Use the command line](cli.md)).

**Documentation**: a [quickstart](quickstart.md), the user guides, one guide per algorithm with its source map from code to paper equations, the [mathematics page](mathematics.md) with the proofs and resource formulas, the backend guides, the [API reference](api/index.md), and the development pages from the [contribution policy](FRAMEWORK.md) to the [code tour](CODE_TOUR.md) and the [engineering constants](ENGINEERING_CONSTANTS.md).

**Examples** ([overview](examples.md)): eight executed notebooks. Four introductions cover a linear system, linear dynamics, eigenvalues and optimization. Three paper-derived studies cover collision relaxation with QLS, heat flow with a boundary penalty with LCHS, and constrained optimization with QHD. One notebook plans five problems at 80 to 100 qubits and estimates their resources without simulating a circuit.
