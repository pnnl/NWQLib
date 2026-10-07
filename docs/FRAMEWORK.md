# Contribution policy

<a id="nwqlib-framework"></a>Changes to NWQLib must meet the rules on this page. They cover what counts as a working implementation, how inputs and errors are checked, what reports, tests and examples must show, and which dependencies the library may use. [Set up, test and build](development/setup.md) gives the commands, and [Maintenance](MAINTENANCE.md) gives the checks for each kind of change. The [Glossary](glossary.md) defines the terms used here, such as Plan, Run, Program and preparation record.

The rules state scientific, numerical and cost requirements. The API names, record schemas and module layout that implement them may change, provided the scientific goals and the stated cost limits are kept. A change that alters them updates every affected caller and guide in the same pull request. Lists of names and fields in these pages describe the current code and are not compatibility promises. [API stability](#api-stability) states what users can rely on.

[Plan, compare and solve](scientist.md) documents the public workflow of Problem, Method, Plan and Result. [Limitations and open work](ROADMAP.md) describes the current implementation and what it does not yet do.

## Purpose

NWQLib provides Qiskit-based quantum algorithm workflows for scientific applications. It serves two groups:

- Scientists who apply an algorithm to their own data without studying every quantum subroutine.
- Quantum algorithm developers who add algorithms, subroutines, validation data and research examples in one consistent form.

Validated user-facing code belongs in the root package, with tests, documentation and its scientific assumptions stated.

## What end-to-end means {#what-end-to-end-means}

An end-to-end workflow does four things:

1. It accepts the scientific problem, the requested output and, when given, accuracy criteria.
2. It chooses a construction.
3. It executes that construction on a supported path.
4. It analyzes the observations into the requested scientific quantity.

Planning, resource estimation, execution, analysis and explicit verification are separate operations, each with its own cost and evidence. A report keeps the content hashes and the unknown values that these operations produce, and it runs no reference calculation. A circuit builder, theory helper or resource formula alone provides only the step it states.

## Relationship to Qiskit

Qiskit provides general circuit objects, transpilation, simulation interfaces and provider integration. NWQLib composes them into workflows with typed problems and options, algorithm-specific decoding, validation, resource estimation and reports. Reuse Qiskit primitives. Translate to or from an SDK only at the backend or provider boundary, and only where a current scientific workflow needs it. Do not duplicate Qiskit or write wrappers for a hypothetical replacement.

## Circuit-level implementation {#circuit-level-implementation-policy}

In NWQLib, an implementation of a quantum algorithm means a circuit that carries it out, unless the task, the API name and the report say otherwise. A NumPy, SciPy, symbolic or closed-form calculation may serve for validation, resource estimation, a classical reference, preprocessing or a clearly labeled theory path. It is not an implementation of a quantum algorithm by itself.

A completed algorithm implementation meets these requirements:

- The public planning and `prepare` path reaches a circuit for each quantum capability the algorithm claims. A separate algorithm-specific `build_*` function is not required.
- The circuit runs in statevector mode on small deterministic validation cases when the algorithm's readout allows statevector readout.
- The circuit runs in shot mode, with real measurements, on at least one representative example.
- The report names the subroutines built, the execution backend, the qubits, whether the gate counts describe the raw circuit or an explicitly transpiled one, the applicable depths and gate counts, the provider package versions, the transpilation settings when transpilation was applied, and the validation references.
- A theory-only or NumPy-assisted path is named and reported as such, and lists the circuit subroutines it skips.

These requirements also apply to subroutines that are quantum algorithms in their own right, such as QPE, LCU, state preparation, Hamiltonian evolution, block encoding, QSP and QSVT, and the GCiM measurement routines. A subroutine is not complete just because dense linear algebra reproduces its mathematical output. Dense linear algebra is acceptable as the independent reference of a test. The implementation includes the circuit construction unless the issue is scoped explicitly as documentation, classical validation or resource-only planning.

## Classical preprocessing and quantum-advantage boundaries {#classical-preprocessing-and-quantum-advantage-boundaries}

Classical preprocessing is allowed when the workflow records it and the algorithm guide states its cost and how it scales. Dense eigensolves, matrix exponentials and exact statevectors are tools for validation or for building small instances, unless the execution mode says the path is theory. Reports separate problem loading, preprocessing, quantum circuit work, postselection, decoding and classical validation, so that a dense helper is never mistaken for an efficient oracle.

The metadata of a block encoding records in `preprocessing_label` how the scientific input reaches the circuit:

| Label | Meaning |
| --- | --- |
| `scalable_oracle` | A structured oracle or encoding is supplied without hidden dense synthesis, as for a banded encoding built from its band specification. |
| `small_dense_validation` | Dense linear algebra or general circuit synthesis is used, at validation scale only. |
| `stored_pauli_structure` | Circulant structure is detected directly from the kept Pauli terms. |

Reports and resource tables never describe an oracle, a subroutine or a formula-only result as a full algorithm circuit.

## Solver input generality {#solver-input-generality-policy}

Every solver accepts arbitrary inputs within its stated theoretical domain. A solver is not hard-coded to one named equation, benchmark, molecule, grid, boundary condition or paper example unless its API name and documentation say it is an example helper. The scientific problem and its input state:

- The mathematical domain of valid inputs, such as Hermitian matrices, positive-semidefinite dissipative parts, sparse Pauli Hamiltonians, chemistry Hamiltonians in a chosen basis or finite-dimensional ODE matrices.
- The scientific data the user supplies, such as `A`, `u0`, `T`, Hamiltonian terms, basis data, source functions or measurement operators.
- The assumptions and automatic conversions that bring an input into the theoretical domain, such as a PSD shift, normalization, padding, truncation, discretization or a change of basis.

Initialization, encoding or embedding, subroutine and measurement choices belong to the Method and its Plan, unless a reference or a preparation itself defines the requested scientific population. One Method's choices never force another Method to change the same scientific target.

Examples may supply heat-equation matrices, H6 chemistry data, small diagonal spectra or power-grid test systems, but they call the same generic solver APIs that users call with their own valid data. When an algorithm is restricted to a special class of equations, the restriction appears in the class or function name, the option validation, the report and the README status table.

## Mathematical definitions before evidence

Start from the definition of each quantity and its valid domain, including sign, range, units and any relation it must satisfy with other quantities. Check those conditions in the first module that knows what the quantity means, before the value is used.

- A mathematically invalid input raises an error that says what is wrong. It never becomes `INCONCLUSIVE`, `unknown` or a passing criterion. A negative L2 error, for example, is an invalid norm discrepancy, and whether the evidence suffices is a separate question.
- Existing tests, schemas and recorded decisions do not override the definition.
- A generic signed scalar, or an estimator explicitly identified as signed, keeps its own domain.

Numerical roundoff is an exception, under these conditions. An explicitly stated tolerance may map a very small negative approximation of a nonnegative quantity to zero, and a value outside that window raises. Code that applies such a tolerance:

- states the tolerance, its units and scale, and the numerical representation it applies to,
- keeps the raw value, and
- discloses the adjustment where it affects a decision.

Do not widen the window after a failure. Arbitrary clipping, sampling noise and an inconsistent model are not roundoff.

Domain checks use the representation the input already has and cheap checks in the module that defines the quantity. They add no simulation, decomposition or extraction of an operator from its oracle. If a check would need such work, propose it in the pull request with its cost, and a maintainer decides whether to add it.

## Choosing among subroutine implementations {#subroutine-implementation-selection}

A Method's settings resolve, at planning, to named and versioned implementations whose capability and scale are stated. The circuit constructors are bound to exactly those definitions and inputs. A saved `Source` record, which names the paper, version and domain behind a construction, never imports executable code. A Method may keep its own finite catalogue of implementations without adding a branch for its algorithm family to the shared planning, circuit-building or reporting code. An invalid combination fails before expensive work. A dense choice states the scale it is limited to, and a structured choice states its input assumptions and how it accounts for error. Two implementations of the same step need independent evidence for the equivalence they claim, or a statement of how their guarantees differ.

When an implementation choice has its own scalar parameters, its typed option may be a `ProviderConfig` request, which holds an implementation name and only the parameters the user supplied. The provider of that implementation validates the map and returns a separate, immutable resolved configuration with defaults filled in and parameter keys sorted. Reports carry both forms, and the content hashes used for caching and calibration are computed from the resolved form. A new provider-specific parameter never adds a top-level field to every algorithm's options dataclass.

## Research data and references {#research-data-and-reference-provenance}

Keep the research inputs, original citations and independent reference checks that support a current scientific claim or are needed to reproduce one. When code moves to another module, move the relation it keeps and its evidence with it. An old report, output collection or job list is not by itself a requirement for accepting a change. Tests use the current public API and never import an old copy of an implementation. When obsolete evidence is removed, remove every claim that depended on it alone.

Keep paper copies outside the repository and local PDF paths out of the documentation. [References](references.md) is the public citation catalogue, with bibliographic data, DOI or arXiv links and useful equation locations.

## Package layout

Each algorithm package holds its scientific records, planning choices, numerical analysis and optional verification. Reusable circuit primitives go under `subroutines`, and backend execution goes under `backends`. The shared records, resource totals, runtime and reporting follow one set of rules for every algorithm. Imports point from algorithms to the shared layers, and the shared layers contain no branch for a particular algorithm family. [Code tour](CODE_TOUR.md) gives a reading order and a file map.

<a id="block-encoding-and-qsp-conventions"></a>The block-encoding definition, the tensor order of formulas compared with Qiskit's qubit order, the Wx QSP phase convention and the rules for combining error bounds are in [Conventions](conventions.md).

## Public algorithm contract

The Problem holds the mathematical input and the target. The Plan holds the Method's choices of construction, initialization, numerical data, measurement and readout. Execution runs what the Plan fixed, and analysis, resource estimation and explicit verification are separate operations with their own costs and evidence. The [API reference](api/index.md) and the algorithm guides give the signatures and supported operations.

The Plan's `Program`, NWQLib's description of a circuit as named block calls, carries its child calls, arguments, qubit-port connections and operation order unchanged into inspection, resource totals and circuit construction. Matching names, together with a whole-method circuit rebuilt separately, do not show that this relation holds. Optimized, dense or user-supplied circuit pieces are allowed at the leaves of a Program, each within its declared domain. Build on the existing composition code instead of adding a second selection tree, planner or executor.

A valid change of subroutine creates a new Plan and never alters an existing Plan or Result. Estimating the same Plan under another cost model asks a new analysis question, which neither compiles nor changes the Plan. A comparison of built circuits names the gate basis, rotation precision, synthesis settings and version, and whether it counts the represented, compiled or executed circuits. A missing cost formula leaves only the quantities that depend on it unknown, and unknown is not zero.

All Methods, adaptive ones included, share one execution, observation, storage and reporting lifecycle. A new Method supplies its construction, readout and analysis, and it adds no family switch to the shared code and no copy of that lifecycle. A domain check belongs in the record or input type that knows the domain, and a Method-specific result field belongs where the code that reads it is. A supported circuit or host path reaches the code that uses its output, or it rejects the request before doing unsupported work. A dense reference, a weaker output or another execution mode is never substituted silently.

Analysis returns the requested quantity together with the content hashes of the contributions it used, any partial state and the evidence that is still missing. A scalar output does not also return a full state. Reanalysis, reports and loading need only the saved data and do no new backend or reference work. Verification runs only when requested. Resource estimation and inspection of a prepared circuit work without running the scientific computation, and each names what its numbers are based on.

### Planning, realization, execution and validation {#planning-realization-execution-and-validation}

These operations have separate costs and evidence:

- **Planning** validates inputs and options, resolves implementations, applies analytic cost formulas and does only the approximation work that synthesis needs. It may inspect the supplied representation and its structure. It does not execute a backend, simulate a circuit, build candidate circuits in order to measure them, transpile before the construction is chosen, rebuild dense operators from circuits, or run dense reference linear algebra only to enrich a report.
- **Realization** resolves one setting of the checked Plan without running it. **Preparation** checks that setting against its limits and builds only its circuit or host kernel. Inspection, when requested, counts the operations of a prepared circuit without submitting it. Represented workloads, successful preparations and executions are counted separately. On a local backend, `prepare(plan, settings="all")` prepares every setting of a static Plan once, before any submission, so that all of them can be inspected. An adaptive Method whose later settings depend on earlier outcomes cannot prepare them in advance.
- **Execution** runs the requested prepared circuit or kernel according to the algorithm's scientific meaning. Resource estimation never executes a circuit, and totals of built or executed circuits are named as such.
- **Validation** covers independent simulation, dense scientific references, empirical fidelity and diagnostics that only enrich a report. It runs only on request, unless it is itself the requested theory computation. An error or fidelity value that was not requested is saved as null or as a not-evaluated status, never as zero error or unit fidelity.

A computation counts as needed by synthesis only when it changes the provider, tier, repetition count, degree, or the coefficients or phases of the built circuit, or when it decides whether the returned synthesis is accepted. A diagnostic that only enriches a report stays out of planning even when its cost is polynomial. Planning cost is measured against the representation the user supplied, so a dense input of exponential size does not justify a second simulation or a dense reference computation.

Public circuit builders build the circuit the Plan fixed. Resource estimation is a separate operation that adds up the cost formulas registered for the Program's blocks, and inspection, when requested, reports the operation counts of the prepared circuit it inspected. Builders share no table that dispatches resource options. Raw `gate_counts` and counts of gates acting on exactly two qubits are not counts in a device's native gate set. A backend target alone is metadata and never triggers transpilation.

### Package import surface

The package root loads its public names lazily, on first access, through PEP 562. The public API is what each module's `__all__` and the [API reference](api/index.md) list. Importing metadata must not load the full scientific stack or any vendor SDK. A native subpackage that the caller imports explicitly may load the SDK it requires.

`tests/test_shared_contracts.py` checks lazy loading, and `tests/test_foundation_imports.py` checks the currently declared public names. When a public name changes on purpose, these tests change with it.

### API stability {#api-stability}

NWQLib makes no stability promise for its public API yet. A minor release may rename or remove public names, and the [release notes](release_notes.md) list every such change. NWQLib owes no compatibility with the development APIs, exports, aliases or schemas that preceded release 1.0.

Each public name has one defining module. When you rename or replace a public name, update its callers, tests, examples and documentation in the same change, and remove the old name. Do not add aliases, bridges, fallback loaders or deprecated wrappers. Replacing a name does not remove a scientific capability that is still useful, and saved files in the current format must still validate, keep their content hashes and load back unchanged.

## Execution modes

The public `execution` choice (`"quantum"` or `"classical"`) and exact or sampled readout (`shots=None` or a positive shot count) are described in [Exact and sampled execution](scientist.md#exact-and-sampled-execution).

Lower-level circuit helpers take `ExecutionMode.STATEVECTOR` or `ExecutionMode.SHOTS` to choose how they run a circuit and read it out. These values are not a further public execution choice of a Method. A physical vector needs amplitudes with their phases, and amplitude, scalar and counts readouts never silently replace one another. Resource estimation and inspection of a prepared circuit stay separate operations.

Classical execution reaches the code that uses its stated numerical result and discloses the approximation it chose. It is not a fallback when circuit execution fails, and planning metadata is not a classical result. No execution mode silently runs an independent reference for comparison.

Pauli or probability readout does not require returning a full state. When a Qiskit Aer operation explicitly requests the full state, remove only the final measurements before adding `save_statevector`. Keep mid-circuit measurements, because they may be part of the algorithm, and report the saved state as conditioned on their outcomes, not as the deterministic state of a unitary circuit. Inspection of a prepared circuit leaves simulator save instructions out of operation counts and depth. When Aer preparation decomposes a copy of the circuit so that Aer can run it, the backend result metadata records that step, including the decomposition depth or repetition setting and the operation names before and after decomposition.

## Circuit readability options

Options for barriers and final measurements control presentation or execution and are not scientific parameters. Builders keep the shared final-measurement policy, insert subroutine barriers only at documented boundaries, and store these choices in the serialized options, so that diagrams and executed circuits can be reproduced.

## Backend and provider policy

Do not force every algorithm through one generic primitive. Each algorithm declares the execution capabilities it needs, and a backend adapter checks those capabilities before it submits a job.

`BackendCapability` and the target records define the capabilities. Common examples:

- `statevector`: exact simulator state access.
- `counts`: shot-based measured circuit execution.
- `expectation`: observable expectation values.
- `qasm_export`: circuit export for external simulators.
- `hardware_submit`: cloud hardware job submission.
- `noise_model`: simulator or provider noise-model support.
- `native_gate_target`: provider-specific native-gate compilation.

[Choose a backend](backends.md) documents each backend, how to connect it and what has been checked on it.

Algorithm-specific primitive choices:

- QPE normally needs measured sampling (counts). It does not require an expectation primitive unless a specific method is written that way.
- GCiM may use expectation estimation for Hamiltonian and overlap matrix elements. Fixed-basis validation can still use statevector or exact classical paths, and ADAPT-GCiM reuses the same projected-matrix solver after choosing its basis.
- LCHS may require statevector readout for small cases, measured success probabilities for circuit demonstrations, and resource-only or theoretical modes for large circuits.

Keep provider adapters thin. An adapter never hides an algorithmic choice such as QPE precision, Trotter order, postselection convention or the definition of an observable.

## Resource estimation before execution

Checks of resources against their limits, and resource estimates, come before the work they govern, and they also work when nothing is executed. The primary Methods use the shared resource totals, quantities with a stated scope, device assessments and the Run's record of executed work. Capacity, consumption, deadlines and logical work are distinct quantities. An unknown cost is not zero. Preparations, attempts, observations and kept output are each counted separately, with their own budget for each stage. An estimate never starts execution.

A scientific estimate adds up the costs of the Plan's construction. Representative sampling and inspection of a prepared circuit are separate operations that run only on request, and an estimate does not switch to building circuits because a small case happens to be feasible. Estimates on different evidence bases keep separate labels and are not combined into a validated uncertainty interval. A representative sample weights each block by its nonnegative integer multiplicity, and its weighted totals remain estimates, not a full count of the built circuit. Inspection counts the operations of one existing circuit and records whether the counts describe that raw circuit or an explicitly requested extra compilation.

Cloud execution first checks the planned capacity and the work already submitted against their limits, and it records where its results came from. A resource-only operation can show the scale of a problem without submitting anything to NWQ-Sim, IBM, IonQ or Quantinuum. A cost formula that applies to any backend does not show that an adapter for a particular backend works.

## Qiskit and dependency policy

NWQLib is one package and one codebase, tested in two CI environments named `stable` and `latest`. [Support and external-data validation](MAINTENANCE.md#support-and-external-data-validation) describes the CI workflows, what each environment installs and how to recreate the stable environment.

Base dependency requirements:

```text
python>=3.12
numpy>=2.5.2
scipy>=1.18.1
sympy>=1.14.0
pydantic>=2.13.5,<3
```

Without any vendor SDK, the base distribution supports [scientific records](records/README.md), checked inputs, Method metadata, symbolic planning and resource totals, saved results and the host numerical paths that Methods choose. Circuit operations and the power-of-two LCHS Pauli and Trotter kernels require Qiskit, and Aer execution also requires Aer. Pure NumPy MPS compression needs no circuit SDK. Each Method's Plan and descriptor record which of these requirements its operations have.

[Install](quickstart.md#install) lists the optional extras and what each one installs, including the separate `scikit_tt` install that the MPS route needs.

NWQ-Sim export uses the `qiskit` extra. NWQ-Sim execution requires a separately built runner. Nexus dependencies resolve normally unless the caller follows the narrowly scoped [pandas 3 install exception](nexus.md#costs-timeout-and-qualification), which remains a declared metadata conflict, not a compatible resolution.

Provider modules may expose target metadata without importing optional packages. They import an optional dependency only when the user requests that backend, and a missing package raises an error that names the extra to install, such as `pip install 'nwqlib[ionq]'`.

A cloud adapter checks the qubit, shot, readout and basis requirements of the Plan before the work they govern. `prepare` binds the backend, `submit` starts the Run, and resuming a Run first retrieves the jobs that were acknowledged or whose outcome is uncertain, before it considers new work. Provider preparation may include remote compilation when it is explicitly selected. Credentials come from the provider's configured environment and are never stored in records or reports. The saved state of a Run keeps the association of each item with its qubit layout, readout and pending job.

Supported SDK rules:

- Use `qiskit_aer` for Aer imports.
- Prefer circuit, quantum information and transpiler APIs documented as public in the supported Qiskit releases.
- Use Qiskit Runtime primitives only where the algorithm needs the IBM Runtime execution model. Do not require all algorithms to use the same primitive.

## Documentation model

This page holds the rules that apply across the library. [Limitations and open work](ROADMAP.md) describes current limitations and open work, and each algorithm guide states its method's scientific guarantees and maps them to the code. Examples teach the public API. Notebooks are generated from their percent-format sources, pass the freshness check and never become a second home for algorithm logic. Figures are regenerated from documented inputs or marked as illustrative.

### Examples policy

An example shows, through the public workflow, a natural scientific input, the requested quantity and a useful next step. A beginner can change the input within the stated bounds and explain the result. A research example keeps its meaningful structured inputs, method choices, analysis and cost questions. Organize examples by these workflows, not by a fixed number of files or a tour of every option. The API reference documents every field and default.

Introductory examples use simple models and direct explanations. Scientific examples develop a concrete case from a collaborator's or method author's paper, cite the relevant equations, and separate the published model from any reduction, discretization or parameter choice made for the tutorial. Both kinds show the NWQLib capabilities that matter to the problem: the chosen construction and its intermediate parameters, the meaning of the physical or normalized output, resource quantities with what they are based on, and the public Result report. Explain why the displayed parameters affect accuracy, success probability or cost, instead of printing a list of fields.

Connect inputs, chosen parameters, observations, checks and reported quantities through their Plan and Result records. Give each scientific comparison an independent reference or relation, its scope and a readable interpretation. Say which checks the example executed and which it only provides as code. A generation or syntax check does not validate numerical results. Show how readers can inspect and save the relevant report or evidence without starting more computation. Break the workflow into readable notebook cells, and explain the mathematics and purpose of nontrivial code. Keep authoring status and internal review notes out of the tutorial.

A beginner example executes on the default Aer backend at a bounded size. Label resource-only, host or reference, subroutine and full-circuit paths accurately. Provider submission, costly verification and larger studies run only when the reader selects them. Up-to-date code or printed planning metadata does not show that anything executed or that a result is accurate.

The percent-format sources in `examples/generators/` are the originals. `examples/generators/build_notebooks.py` generates each notebook from its source, so the notebook runs the same code with the same deterministic seeds where seeds apply. Edit the source and regenerate the notebook, as [Notebook generation](MAINTENANCE.md#notebook-generation) describes. When a change affects them, check both notebook freshness and notebook execution, because neither replaces the other.

Update an example when the public name it imports changes, and keep no alias for an old import spelling. Keep presentation code in the example and scientific code in the library module it belongs to. Merge two examples only after each of their distinct scientific operations has a working new home. Test workload and output meaning with bounded fixtures. A list of file names does not prove coverage.

## Reporting standard

Reports are built from the stored Plan choices, assessments, preparation records, observations and analyses. Keep formatting separate from computation, with `print()` only in presentation code. Reports, loading and certificates never map, compile, solve eigenproblems, reconstruct states, replay optimizers or execute references. Resource operations return their own quantities with a stated scope and evidence.

Required report sections:

- Report metadata: report schema version and the package versions that generated it.
- Problem summary: equation, matrix or operator dimensions, boundary conditions, units where applicable, and input data source.
- Algorithm summary: algorithm name, subroutines used, approximation order, theoretical equations and relevant paper references.
- Execution summary: backend, execution mode, shots, random seed, Qiskit version, provider package version, and transpilation settings when applied.
- Parameters: tolerances, beta or kernel choices, QPE precision, Trotter steps, state-preparation method, block-encoding convention, QSP degree and phase count, ancilla counts and truncation parameters.
- Resources: the planned workload and what its counts are based on, the preparations, attempts and observations that occurred, applicable gate counts and the target they were counted for, independent capacity and consumption quantities, and unknown costs. Text and machine-readable fields come from the same stored records.
- Evidence: requested accuracy and conditioning, sources of uncertainty, the checks that ran, the preparation records of reference computations, the thresholds, and conclusions that are unavailable or conditional. A reference value that was not requested stays unavailable.
- Interpretation: a short explanation of what the output means for the scientific problem and what limitations remain.

Reports are printable, serialize to JSON, carry a version and are stable enough for tests.

Each report declares its current format and keeps its complete scientific fields, their meanings and the content hashes through loading. The [API reference](api/workflow.md#save-and-reopen) documents the current types and the inputs that loading needs. Saved `Source` text never chooses an executable import or recreates a live binding to code. Validation of the current format remains required, and converters from development schemas are not.

Static and adaptive reports keep the measurement settings and the compatible observation streams, with the iteration history of adaptive Methods (their saved controller state). Seeds alone do not prove independence. Error combinations and checks keep their error frames (the quantity, metric and unit an error refers to), assumptions and sources. A nominal interval or residual does not establish total accuracy or that the ground state was identified. [Plan, compare and solve](scientist.md) and the algorithm guides describe the reports and verification that are supported.

## Documentation and commenting standard

Use Google-style docstrings. Docstrings for scientific methods include:

- What equation or algorithmic step is implemented.
- Paper reference, with the equation number, theorem, algorithm or section.
- Inputs and units or normalization conventions.
- Output meaning and shape.
- Approximation assumptions and known limitations.

Inline comments explain non-obvious scientific or circuit logic and do not repeat the code. Keep research sources and citations, and remove descriptions of superseded implementations.

Every hardcoded engineering value has a comment at its definition that explains the choice and the condition for revisiting it, plus an entry in [Engineering constants](ENGINEERING_CONSTANTS.md). This includes instance-size guards, numerical tolerances, safety factors, accuracy factors, schedule defaults and budgets. Document paper-derived values with their equations and engineering choices with their reasons. A historical value needs a scientific or engineering basis that still applies, and its age alone does not justify keeping it.

Public record docstrings use `Args:` or `Attributes:` entries whose names match the dataclass fields. Literal-valued fields list their accepted spellings. Scientific quantities state units, normalization and shape when the type does not make them obvious.

## Testing and validation standard

Scientific independent verification and validation (Scientific IV&V) is the review framework for substantive scientific, execution and resource changes. Derive the expected relation independently, and assess whether the implementation, the code that uses its results and its evidence support the requested quantity. Resource and lifecycle violations count against acceptance alongside numerical defects. [Scientific IV&V review selection](MAINTENANCE.md#scientific-ivv-review-selection) defines when it applies, how earlier evidence is reused, metamorphic relations and reporting. Every substantive review assesses the relevant check families and closes the evidence gaps that matter for the change. Not every test family or experiment runs for every change.

Use layered tests:

- Unit tests for subroutines: state preparation, LCU, Trotterization, block encoding, QSP phase application, QPE likelihood and objective helpers, measurement post-processing and reporting.
- Circuit tests: small circuits compare `Statevector.from_instruction()` with expected states or unitary action. Small dense matrix extraction is a test oracle, not the runtime definition of a structured encoding.
- Shot tests: measured circuits run with a fixed seed and broad statistical tolerances.
- Scientific validation tests: compare algorithm output with classical references on small, named datasets.
- Cross-implementation equivalence tests: when a subroutine offers more than one implementation, build the same instance through each and check statevector equivalence or the documented trade-offs (see [Choosing among subroutine implementations](#subroutine-implementation-selection)).
- Regression tests: keep small JSON or NPZ expected outputs only when their origin is documented.

<a id="suite-runtime-discipline"></a><a id="platform-dependent-quantities-in-tests"></a><a id="policy-changes-and-their-enforcement-tests"></a>Test size, where external-data validation runs, platform-dependent quantities and keeping a policy in step with its enforcement are covered in [Test runtime and platform variance](MAINTENANCE.md#test-runtime-and-platform-variance).

### Accuracy and tolerance classes {#accuracy-and-tolerance-discipline}

Every numerical test tolerance belongs to one of the following classes. Name the class in the assertion or an adjacent comment:

1. **Machine-precision identities** (tolerance justified by scale, conditioning and numerical representation): exact mathematical identities, including unitary equivalence of two synthesis paths, reconstruction of an operator from its own decomposition, and circuit-versus-theory agreement when both use the same dense arithmetic.
2. **Method-error-bounded** (tolerance derived from the documented approximation model): quadrature truncation, Trotter order, QPE phase resolution. The tolerance follows from the method parameters used in the test, with independent evidence for the claimed approximation relation. Reuse an existing convergence check that applies, or use the smallest refinement check that can tell the cases apart. State the derivation of the tolerance. A new parameter sweep adds run time, so propose it in the pull request with its cost before adding it.
3. **Statistical** (`k * sigma` with a fixed seed): a shot-based assertion states the shot count and uses a bound derived from the estimator variance (for example the binomial `sigma = sqrt(p(1-p)/S)`), not an arbitrary constant.

Two additional rules:

- **Regression anchors.** Keep fixed-instance and fixed-seed reference values that detect a real unintended change, alongside independent scientific relations. Their tolerances follow the numerical problem.
- **Mutation-probe selection.** Mutation probes are reserved for scientific equations, signs, normalizations, error compositions, resource formulas or algorithm-defining constructions that have an independent reference and could otherwise return a believable but scientifically wrong result. Defaults, metadata shape, ordinary validation branches and solver-policy literals are tested directly.

Accepting an algorithm depends on its claimed domain and the public relations the change affects. Use a representative statevector check and a measured-shot check when those capabilities are claimed, a public example, independent scientific quantities and documented equations. When code moves, move the existing tests that can detect its defects with it, instead of adding tests to reach a count. State partial or unknown scope explicitly, and apply the mutation selection rule only where a probe adds evidence.

## Add an algorithm {#how-to-add-a-new-algorithm}

Follow [Add a method](algorithm_protocol.md) for the interface, and the rules on [circuit-level implementation](#circuit-level-implementation-policy), [solver input generality](#solver-input-generality-policy), [documentation](#documentation-and-commenting-standard) and [testing](#testing-and-validation-standard) for what a new algorithm needs before it is accepted.

## Add a subroutine {#how-to-add-a-new-subroutine}

1. Add the implementation under `nwqlib.subroutines`.
2. Define a narrow interface with explicit input and output shapes.
3. Include paper references and equation locations in docstrings.
4. Provide a small standalone circuit test if the subroutine builds circuits. If the subroutine is itself a quantum algorithm or circuit primitive, this circuit test is mandatory.
5. Add a short integration test showing that at least one algorithm can call it.
6. Add report metadata so that downstream algorithms can say the subroutine was used.

## README requirements

The root README states the validation scope accurately:

- State what is validated now and what remains deferred or resource-only work.
- Link to this page and to [Limitations and open work](ROADMAP.md).
- Give installation commands for the package and its extras.
- List dependencies and Qiskit target versions.
- Cite the scientific papers that define the implemented methods.
- Avoid implying that archived outputs are validated library guarantees.
- Point developers to the tests and examples for each algorithm.
