# NWQLib Framework

This document owns enduring scientific, numerical and resource contracts. [ROADMAP.md](ROADMAP.md) and the public API guides describe the checked-out implementation and its current limitations. An authorized redesign may replace its API/schema/owner choices while preserving the scientific goals and explicit cost boundaries. Update affected callers and guides with that implementation; descriptive inventories are not compatibility requirements.

The [scientist workflow](scientist.md) documents the current Problem/Method/Plan/Result API. The [contract specimen](contract-specimen/README.md) supplies fixed analytical and accounting witnesses for that architecture. It is not a general planner or evidence of backend support.

## Purpose

NWQLib provides Qiskit-based quantum algorithm workflows for scientific applications. It serves:

- Scientific users who want to apply an algorithm to their own data without needing to understand every quantum subroutine.
- Quantum algorithm developers who want to add algorithms, subroutines, validation data, and research examples consistently.

Validated user-facing code belongs in the root package with stable interfaces, tests, documentation, and explicit scientific assumptions. Current status is tracked in [ROADMAP.md](ROADMAP.md).

## What "End-to-End" Means

An end-to-end workflow accepts the scientific problem and requested output, with accuracy criteria when selected, selects an actual construction, executes a supported selection, and analyzes its observations into the requested scientific quantity. Planning, resource assessment, execution, analysis and explicit verification have separate costs and evidence. Reports preserve the resulting identities and unknowns without executing a reference calculation. A circuit builder, theory helper or resource formula alone supplies only its stated layer of the workflow.

## Relationship to Qiskit

Qiskit owns general circuit objects, transpilation, simulation interfaces, and provider integration. NWQLib composes those primitives into workflows with typed problems and options, algorithm-specific decoding, validation, resource estimation, and reports. Reuse Qiskit primitives. Keep SDK translation at the actual native/provider boundary when a current scientific workflow needs it. Do not duplicate Qiskit or create wrappers solely for hypothetical replacement.

## Circuit-Level Implementation Policy

In NWQLib, "implementation" means a circuit-level realization unless the task, API name, and report explicitly say otherwise. A NumPy, SciPy, symbolic, or closed-form calculation may be used for validation, resource estimation, classical references, preprocessing, or a clearly labeled theory path, but it is not a completed quantum algorithm implementation by itself.

A completed algorithm implementation must satisfy these requirements:

- The public selected-construction/prepare path reaches an actual native circuit implementation for each claimed quantum capability. A separate algorithm-specific `build_*` facade is not required.
- The circuit path can run in statevector mode for small deterministic validation cases when the algorithm semantics allow statevector readout.
- The circuit path can run in shot mode with actual measurements for at least one representative example.
- The report identifies the realized subroutines, execution backend, qubits, raw or explicitly transpiled inventory basis, applicable depths and gate counts, provider package versions, transpilation settings when applied, and validation references.
- Any theory-only or NumPy-assisted path is named and reported as such, with the skipped circuit subroutines listed explicitly.

This policy applies to algorithms and to subroutines that are themselves quantum algorithms, such as QPE, LCU, state preparation, Hamiltonian evolution, block encoding, QSP/QSVT, and GCiM measurement routines. A subroutine is not complete solely because its mathematical output can be reproduced with dense linear algebra. Dense linear algebra is acceptable as an oracle for tests. The implementation must include the corresponding circuit construction path unless the issue is explicitly scoped as documentation, classical validation, or resource-only planning.

## Classical Preprocessing And Quantum-Advantage Boundaries

Classical preprocessing is permitted when the workflow records it and the algorithm guide states its cost and scalability boundary. Dense eigensolves, matrix exponentials, and exact statevectors are validation or small-instance realization tools unless the execution mode explicitly says theory. Reports must distinguish problem loading, preprocessing, quantum circuit work, postselection, decoding, and classical validation so that a dense helper is never mistaken for an efficient oracle.

Preprocessing labels describe how the scientific input reaches the quantum workflow. `scalable_oracle` means a structured oracle or encoding is supplied without hidden dense synthesis. `small_dense_validation` means dense linear-algebra or general circuit synthesis is used only at validation scale. `theory_reference` means a non-circuit formula or classical reference was evaluated. Reports and resource tables must not describe an oracle, subroutine, or formula-only result as a full algorithm circuit.

## Solver Input Generality Policy

Every solver should expose clear arbitrary inputs within its stated theoretical domain. A solver must not be hard-coded to one named equation, benchmark, molecule, grid, boundary condition, or paper example unless the API name and documentation explicitly say it is an example helper. The scientific problem and its input access should state:

- The mathematical domain of valid inputs, such as Hermitian matrices, positive-semidefinite dissipative parts, sparse Pauli Hamiltonians, chemistry Hamiltonians in a chosen basis, or finite-dimensional ODE matrices.
- The user-supplied scientific data, such as `A`, `u0`, `T`, Hamiltonian terms, basis data, source functions, or measurement operators.
- The assumptions and automatic conversions applied to bring an input into the theoretical domain, such as PSD shifting, normalization, padding, truncation, discretization, or basis transformations.

Method initialization, encoding/embedding, subroutine and acquisition choices belong to the selected method/construction, unless a reference or preparation actually defines the requested scientific population. They must not force another method to change the same scientific target.

Examples may provide heat-equation matrices, H6 chemistry data, small diagonal spectra, or power-grid test systems, but those examples must call the same generic solver APIs that users call with their own valid data. If an algorithm is genuinely restricted to a special equation class, that restriction belongs in the class/function name, option validation, report, and README status table.

## Mathematical Definitions Before Evidence

Start from the definition of the quantity and its admissible domain, including sign, range, units and coupled relations. Validate those conditions at the earliest owner that knows the quantity's meaning, before using the value. A negative L2 error is an invalid norm discrepancy. Insufficient evidence is a separate question. A mathematically invalid input must raise an actionable error, not become `INCONCLUSIVE`, `unknown` or a passing criterion. Existing tests, schemas and recorded decisions do not override the definition. A generic signed scalar or an explicitly identified signed estimator keeps its own domain.

Numerical roundoff is a qualified exception. An explicitly stated tolerance may map a very small negative approximation of a nonnegative quantity to zero. A value outside that window must raise. State the tolerance, units, scale and applicable numerical representation, preserve the raw value and disclose the adjustment where it affects a decision. Do not enlarge the window after a failure or treat arbitrary clipping, sampling noise or an inconsistent model as roundoff. This policy does not authorize extra simulation, decomposition or oracle extraction: use the admitted representation and cheap owner-boundary checks, with additional expensive work subject to the existing approval rules.

## Subroutine Implementation Selection

Selected method settings resolve explicit versioned implementations with stated capability and scale. Native constructors are bound to their actual selected definitions and inputs. Portable Source records never import executable code. A method may maintain its own finite implementation catalogue without adding a scientific family branch to common planning, lowering or reporting. Invalid combinations fail before expensive work. Dense choices keep their explicit scale, and structured choices state their input assumptions and error accounting. Alternative implementations need independent evidence for the equivalence they claim or an explicit statement of their different contracts.

When one implementation slot has provider-specific scalar parameters, its typed option may be a `ProviderConfig` request containing an implementation name and only the parameters explicitly supplied by the user. The provider validates that map and produces a distinct immutable resolved configuration with defaults injected and parameter keys sorted. Reports carry both forms. Cache and calibration identities use the resolved form. Adding a provider-specific parameter must not add a top-level field to every algorithm options dataclass.

## Research Data and Reference Provenance

Keep research inputs, original citations and independent witnesses that support a current scientific claim or reproduction need. When an owner changes, move its applicable relation and evidence with it. An old report, output corpus or job list is not automatically a current acceptance requirement. Tests exercise current public APIs without importing superseded implementation copies; removing obsolete evidence also removes any unsupported claim that depended on it.

Paper copies used for developer cross-checking belong outside the repository. The public citation catalogue is [References](references.md), with bibliographic metadata, DOI or arXiv URLs, and useful equation locations. Keep local PDF paths out of public documentation.

## Package Layout

Algorithm packages own scientific records, selection, numerical analysis and optional verification. Reusable circuit primitives belong under `subroutines`. Backend execution belongs under `backends`. Common records, resource fold, runtime and reporting keep their shared contracts. Dependencies point from algorithms toward shared layers, without common scientific family branches. The maintainer reading order and concrete file map live in [CODE_TOUR.md](CODE_TOUR.md).

## Block-Encoding and QSP Conventions

A unitary `U` on `a + n` qubits is an `(alpha, a, epsilon)` block encoding of `A` when the all-zero ancilla block approximates `A / alpha`, equivalently `||A - alpha (<0^a| (x) I) U (|0^a> (x) I)|| <= epsilon`. Every `BlockEncoding` records `alpha`, ancilla and system widths, its operator-error bound, and the resolved implementation as canonical fields. Polynomial transformations act on `A / alpha`, so simulating `exp(-i A t)` therefore uses effective time `alpha * t`. Products multiply subnormalizations, while an LCU combination has subnormalization `sum_j |c_j| alpha_j`.

Tensor formulas in these docs and in docstrings write the ancilla, label or control register first, as the papers do, for example `(<0^a| (x) I) U (|0^a> (x) I)` above and `SELECT = sum_j |j><j| (x) sign(c_j) P_j`. The circuits place that register on the low-order qubits, starting at qubit 0. A dense matrix of such a circuit in Qiskit's Kronecker order, whose qubit 0 is the least significant index, therefore reads `(I (x) <0^a|) U (I (x) |0^a>)` and `sum_j sign(c_j) P_j (x) |j><j|`. The written order is notation, and the stored register order decides the qubit positions. Saved relation strings keep the paper order.

The project uses the Wx QSP convention, `W(x) = [[x, i sqrt(1-x^2)], [i sqrt(1-x^2), x]]`, with phases applied as `exp(i phi Z)`. Since Qiskit uses `RZ(theta) = exp(-i theta Z / 2)`, a direct `exp(i phi Z)` is `RZ(-2 phi)`. The projector-phase circuit of `build_qsvt_circuit` applies reflection-convention phases `phi'` instead, which `wx_phases_to_reflection` computes from the Wx phases as `phi'_0 = phi_0 - pi/4`, `phi'_d = phi_d - pi/4` and `phi'_j = phi_j - pi/2` for `0 < j < d` (Martyn et al., arXiv:2105.02859v5, Eq. (14) and App. A.2, Eq. (A5)). Each phase step flips the signal qubit onto the projector subspace `Pi`, applies `RZ(+2 phi'_j) = exp(-i phi'_j Z)` and flips back, which realizes `exp(i phi'_j (2 Pi - 1))`. The circuit's global phase `d pi / 2` multiplies the reflection-form product by `i^d`, which makes the encoded block the Wx polynomial. Phase ordering and convention are part of the contract and may not be silently reversed, conjugated, or renormalized. The all-zero phase vector of length `d + 1` realizes `T_d(x)` and remains the convention known-answer test.

Composition combines only errors with compatible mathematical metrics, domains and premises. A sum of multiplicities times unitary operator-norm bounds needs the applicable telescoping relation; it is not a general total-error rule. State-preparation agreement on the all-zero input does not establish full-unitary equivalence under control, adjoint or workspace reuse. Solver residual criteria, physical operator error and model/cost uncertainty remain distinct. Small dense matrix extraction is a test oracle, not the runtime definition of a structured encoding.

## Public Algorithm Contract

The scientific problem owns its actual mathematical input and target. A selected construction owns the method's actual choices, initialization, numerical data and acquisition/readout. Execution consumes that selection; analysis, resource assessment and explicit verification have their own costs and evidence. Concrete signatures and supported operations follow the live [API reference](api/workflow.md) and method guides.

The selected Program must carry actual child calls, arguments, wire/port associations and operation order through inspection, resource folding and native construction. Shared names plus an independently rebuilt whole-method circuit do not establish that relation. Keep legitimate optimized/dense/supplied native leaves with their declared domains. Reuse the current composition owner rather than create a second selection tree, planner or executor.

A legal subroutine change creates a new selection; it does not mutate an existing Plan/result. An alternative cost context asks a new analysis question about that same selection and does not silently compile or change it. Native comparisons identify the effective basis, rotation precision, synthesis settings/version and actual represented/compiled/executed population. A missing law affects its dependent quantities only; unknown is not zero.

Methods share the common execution, observation, storage and reporting lifecycle, including adaptive controllers. A new method supplies its scientific construction/readout/analysis without adding a common family switch or duplicating that lifecycle. Scientific domains belong at the informed record/input owner; method-specific result associations follow the actual consumer. Supported native/host paths must reach their declared consumer or reject before unsupported work. A dense reference, weaker output or another execution mode is not a silent substitute.

Analysis returns the requested quantity with actual contribution identities, partial state and unresolved evidence. A scalar output does not implicitly publish a full state. Reanalysis and report/load operations consume sufficient stored evidence without new backend or reference work. Optional verification remains explicit. Resource fold and admitted prepared inspection work independently of scientific execution and identify their evidence basis.

### Planning, Realization, Execution, and Validation

These operations have separate cost and evidence contracts:

- **Planning** validates inputs and options, resolves implementations, applies analytic resource laws, and performs only synthesis-essential approximation work. It may inspect the supplied representation and structural metadata, but it does not execute a backend, simulate a circuit, build candidate circuits merely to measure them, transpile before selecting a realization, reconstruct dense operators from circuits, or run dense reference linear algebra solely to enrich a report.
- **Realization** resolves an admitted point of the selected Plan without executing it. **Preparation** admits and materializes only that point's selected native or host artifact. Explicit inspection inventories the prepared artifact without submitting it. Represented workloads, successful preparations and actual executions are different populations. On a local backend, `prepare(plan, settings="all")` prepares every point of a static Plan before any submission, each once, so that all of its artifacts can be inspected. A controller whose later points depend on earlier outcomes cannot prepare them in advance.
- **Execution** runs the requested selected artifact according to the algorithm's scientific semantics. Resource estimation never executes a circuit, and realized or executed circuit totals use explicitly named terms.
- **Validation** owns independent simulation, dense scientific references, empirical fidelity, and report-only diagnostics. Such work is explicit opt-in unless it is the requested THEORY computation itself. Unrequested error and fidelity values serialize as null or a canonical not-evaluated status, never as zero error or unit fidelity.

A computation is synthesis-essential only when it changes the provider, tier, repetitions, degree, realized coefficients or phases, or guards acceptance of the returned synthesis. Polynomial cost alone does not move a report-only diagnostic into planning. Planning complexity is measured against the supplied representation: receiving a dense exponential-size input does not authorize a second simulation or dense reference computation.

Public circuit builders construct their selected artifact. Scientific resource estimation separately folds the selected Program and its registered laws; explicitly requested prepared-artifact inspection reports the actual inspected inventory. There is no shared builder resource-option dispatch matrix. Raw `gate_counts` and exact-arity two-qubit counts are not native-gate measurements. A backend target alone is metadata and does not authorize transpilation.

### Package Import Surface

The root resolves public entries lazily through PEP 562. Current module `__all__` lists and the [API reference](api/workflow.md) describe the actual advertised surface. Importing metadata must not eagerly load the full scientific or vendor stack. An explicitly selected native subpackage may load its required dependency.

This unreleased redesign owes no legacy API, export, alias or development-schema compatibility. Select one scientific owner and update its actual callers, tests and documentation together. Replaced paths leave with their replacement; do not add bridges, fallback loaders or deprecated facades. Useful scientific capabilities and current-format validation, identity and round-trip remain required.

`tests/test_shared_contracts.py` checks lazy loading and unknown-name rejection; `tests/test_foundation_imports.py` follows the currently declared surface. These checks follow intentional changes rather than preserving historical spelling.

## Execution Modes

The public `execution` choice is `quantum` or `classical`. Quantum execution consumes the selected native circuit; classical execution performs the method's declared host operation. Readout is selected separately by the requested output and `shots`: exact readout uses no finite samples, while positive shots requests counts per selected readout setting, which for a grouped Pauli readout is one qubit-wise commuting group. A provider-managed estimate has its own precision and cannot claim a known raw-shot population.

With `shots=None`, deterministic exact-simulation routes evaluate every requested observable of a shared coherent prefix from one simulation. A Plan declares the observation positions, labels or marginals, and any reversible readout views. Several non-destructive observations may belong to one experiment. Each intermediate view restores the ideal continuation state before evolution resumes. Sampled settings use fresh preparations for their actual measurement populations. The prepared receipt counts all executed native operations, including readout rotations and their inverses, and the Run admits the complete readout payload before submission. Readout values share the acquisition that produced them. Their numerical windows follow the receipt's operation count, full register width and stated exclusions. A backend must support the selected observation schedule to execute that Plan. The exact statevector target of Qiskit Aer executes a schedule of Pauli, probability and amplitude points, reductions whose registered reducers supply both work and execution functions, and readout views of Pauli and probability points in one simulation. Every other adapter refuses a schedule before native work.

Lower-level native helpers use `ExecutionMode.STATEVECTOR` or `SHOTS` for their actual transport/readout. Those values are not competing public Method execution choices. A physical vector needs phase-faithful amplitudes; scalar and counts paths do not silently substitute one for the other. Resource estimation and explicit prepared inspection remain separate operations.

Classical execution must reach its stated numerical consumer and disclose the selected approximation. It is not a fallback when native execution fails, and planning metadata is not a host result. No selected mode silently executes an independent reference for comparison.

Selected Pauli or probability readout does not require publishing a full state. For Qiskit Aer operations that explicitly request full-state output, remove only final measurements before adding `save_statevector`. Preserve mid-circuit measurements because they may be part of the algorithm, but report the saved state as measurement-conditioned rather than a deterministic exact unitary state. Prepared inspection excludes simulator save instructions from operation counts and depth. When Aer preparation decomposes a copied circuit for execution compatibility, the backend result metadata must record that preprocessing, including the decomposition depth/repetition setting and the logical versus execution operation names.

## Circuit Readability Options

Options controlling barriers and final measurements are presentation or execution-policy choices, not scientific parameters. Builders preserve the shared final-measurement policy, insert subroutine barriers only at documented boundaries, and keep those choices in serialized options so diagrams and executed circuits are reproducible.

## Backend and Provider Policy

Do not force every algorithm through one generic primitive abstraction. Each algorithm must declare the execution capabilities it needs, and backend adapters must check those capabilities before submitting a job.

Capabilities are defined by `BackendCapability` and the selected target records. Common examples include:

- `statevector`: exact simulator state access.
- `counts`: shot-based measured circuit execution.
- `expectation`: observable expectation values.
- `qasm_export`: circuit export for external simulators.
- `hardware_submit`: cloud hardware job submission.
- `noise_model`: simulator or provider noise-model support.
- `native_gate_target`: provider-specific native-gate compilation.

Backend connections and their current qualification are documented in the [backend guide](backends.md):

| Target | Role in NWQLib | Integration path |
| --- | --- | --- |
| Qiskit Aer | Local small-to-medium validation and CI. | `qiskit_aer` plus Qiskit circuits. |
| NWQ-Sim | Selected native simulation. | Separately built runner, typed file protocol and explicit durable workflow. See [NWQ-Sim](nwqsim.md). |
| IBM Quantum hardware | Real-device execution for IBM systems. | Qiskit Runtime `SamplerV2` or `EstimatorV2`, chosen per algorithm. |
| IonQ | Trapped-ion simulator and QPU execution. | `qiskit-ionq` provider/backend path. Measured circuits required. |
| Quantinuum Nexus H2 | Explicit cloud compile and execution workflow. | Optional qnexus and pytket conversion dependencies. Helios is unsupported. See [Nexus](nexus.md). |

Algorithm-specific primitive choices:

- QPE normally needs measured sampling/counts. It should not require an expectation primitive unless a specific method is written that way.
- GCiM may use expectation estimation for Hamiltonian/overlap matrix elements. Fixed-basis validation can still use statevector or exact classical paths, and Adapt-GCiM reuses the same projected-matrix solver after selecting its basis.
- LCHS may require statevector readout for small cases, measured success probabilities for circuit demonstrations, and resource-only/theoretical modes for large circuits.

Provider adapters should be thin. They should not hide algorithmic choices such as QPE precision, Trotter order, post-selection convention, or observable definition.

## Resource Estimation Before Execution

Resource admission and assessment precede the work they govern. They also work when execution is not selected. Primary methods use the common selected resource fold, scoped quantities, device assessments and actual execution ledger. Capacity, consumption, deadlines and logical work are distinct. An unknown cost is not zero. Actual preparations, attempts, observations and kept output have their own populations and stage budgets. Estimates do not authorize execution.

Scientific estimation folds the selected construction. Representative sampling and prepared inspection are explicit operations; an estimate does not switch into circuit construction because a small case happens to be feasible. Multiple evidence bases keep separate labels and do not become validated uncertainty intervals. A representative sample weights each block by its nonnegative integer multiplicity, and its weighted totals remain estimates rather than a full native inventory. Prepared inspection counts the operations of one existing native circuit and records whether they describe that raw circuit or an explicitly requested auxiliary compilation.

Cloud execution requires its selected capacity/exposure admission and actual provenance. A resource-only operation can expose scale without submitting to NWQ-Sim, IBM, IonQ or Quantinuum. A portable law does not establish a live adapter.

## Qiskit and Dependency Policy

Maintain one NWQLib package and codebase. Pull requests targeting `main` and manual full-check dispatches validate it in two CI execution environments named `stable` and `latest`, with the same full test suite, examples, optional dependencies, provider SDK import checks, lint, and documentation build. Ordinary pushes install Ruff, jsonschema and Pydantic for lint, stdlib-only policy checks, the analytical contract specimen, and the SDK-free scientific input/record interchange audit. The package requirements set the minimum versions, while completed full CI runs establish compatibility for the tested combinations.

Base dependency requirements:

```text
python>=3.12
numpy>=2.5.2
scipy>=1.18.1
sympy>=1.14.0
pydantic>=2.13.5,<3
```

The base distribution supports [scientific records](records/README.md), admitted inputs, method metadata, symbolic planning/folding, saved evidence and selected host numerical paths without vendor SDKs. Native circuit operations and power-of-two LCHS Pauli/Trotter kernels require Qiskit. Aer execution additionally requires Aer. Pure NumPy MPS compression has no circuit-SDK dependency. A method's concrete selection and descriptor keep these operation-specific requirements.

Current extras are self-contained with their direct requirements:

- `qiskit`: `qiskit>=2.5.2` for native intake, lowering, export and selected kernels.
- `aer`: Qiskit and `qiskit-aer>=0.17.2` for local execution.
- `tensor`: Qiskit and `scikit_tt` for circuit MPS preparation.
- `qasm`: Qiskit, OpenQASM parser and Qiskit QASM3 importer for explicit materialization.
- `chemistry`: Qiskit, PySCF and OpenFermion for chemistry input preparation.
- `ibm`: Qiskit and `qiskit-ibm-runtime>=0.49.0` for admitted Runtime operations.
- `ionq`: Qiskit, `qiskit-ionq>=1.1.1`, `requests>=2.34.2` and `urllib3>=2.8.0` for admitted IonQ operations. The urllib3 floor carries the streaming-reader behavior and security fixes that the response reader relies on.
- `nexus`: Qiskit, `qnexus>=0.49.0`, `pytket>=2.18.1`, `pytket-qiskit>=0.78.0` and `selene-core>=0.3.2` for Nexus H2 operations.
- `nwqec`: Qiskit and `nwqec==0.1.2` for explicit logical Clifford+T compilation of small circuits.
- `qre`: `qdk[qre]==1.32.3` for the explicit QDK physical resource projection of a compiled body.
- `dev`: pytest, pytest-xdist, Ruff and jsonschema. Select `qasm` separately for parser/importer tests. Notebook tooling lives in `notebook`.
- `docs`: `mkdocs-material>=9.5` and `mkdocstrings[python]>=0.25` for building this documentation with MkDocs.

NWQ-Sim export uses `qiskit`. Native execution requires a separately built runner. Nexus dependencies use normal resolution unless the caller explicitly follows the narrowly scoped [pandas 3 install exception](nexus.md#costs-timeout-and-qualification). That exception remains a declared metadata conflict, not resolver compatibility.

Provider modules may expose target metadata without importing optional packages. They should import optional dependencies lazily only when the user requests that backend, and missing packages should raise an actionable error that names the required extra, such as `pip install -e ".[ionq]"`.

Cloud adapters validate selected qubit, shot, readout and basis requirements before the work they govern. `prepare` binds the selected backend; `submit` starts its actual Run, and resume retrieves acknowledged or uncertain jobs before considering new work. Provider preparation may include explicitly selected remote compilation. Credentials are read from the configured provider environment and are never stored in records or reports. Durable state preserves actual item/layout/readout and pending-job associations.

Supported SDK rules:

- Do not use old `qiskit.opflow`, `QuantumInstance`, `BasicAer`, or old `qiskit.algorithms` APIs in current code.
- Use `qiskit_aer` for Aer imports.
- Prefer circuit, quantum information, and transpiler APIs documented as public in the supported Qiskit releases.
- Use Qiskit Runtime primitives only where the algorithm needs the IBM Runtime execution model. Do not require all algorithms to use the same primitive.
- Run `pip check` in both CI environments and record resolved versions with the validation results.

The stable environment uses the exact Python version in `.python-version`, package and build constraints from `ENVIRONMENT_LOCK.txt`, and its audited `scikit_tt` revision. The latest environment uses the latest stable Python and packages compatible with it, upgrades packages eagerly to final releases, checks the selected direct package versions, and exercises tensor upstream HEAD. Full checks and scientific mutation probes run on pull requests targeting `main`, including updates to those pull requests, or manual dispatches. The push after merge runs only the lightweight checks. There are no scheduled runs. Promote a new stable environment only after the full checks pass, updating the Python version and constraints snapshot together.

Compatibility validation checks the latest stable Python and compatible direct dependencies through the public APIs used by the package. Transitive dependency constraints remain in effect, including SymPy 1.14's `mpmath<1.4` requirement. Compatibility claims apply to the tested combinations and require new evidence for later upstream releases. Both CI environments exercise the same package version. Setup and validation commands are described in [MAINTENANCE.md](MAINTENANCE.md).

## Documentation Model

FRAMEWORK owns enduring cross-library scientific policy, ROADMAP owns descriptive current limitations and open work, and algorithm guides own scientific contracts and implementation maps. Examples teach the public API. Notebooks are generated from their percent-format sources, must pass the freshness check, and may not become a second source of algorithm logic. Figures are regenerated from documented inputs or clearly marked as illustrative.

### Examples Policy

Examples teach natural scientific input, the requested quantity and a useful next action through the actual public workflow. A beginner should be able to change a bounded input and explain the result; a research example keeps its meaningful structured inputs, method choices, analysis and cost questions. Organize examples by those workflows rather than a fixed file count or exhaustive Options tour. The API reference owns complete field/default documentation.

Introductory examples use simple models and direct explanations. Scientific examples develop a concrete example from a collaborator's or method author's paper, citing the relevant equations and distinguishing the published model from any reduction, discretization or parameter choices made for the tutorial. Both levels demonstrate the NWQLib capabilities that matter to the problem: the selected construction and intermediate parameters, physical or normalized output meaning, resource quantities with their evidence basis, and the public Result report. Explain why the displayed parameters affect accuracy, success probability or cost rather than printing an unexplained field inventory.

Connect inputs, selected parameters, observations, checks and reported quantities through their actual Plan and Result records. Give each scientific comparison an independent reference or relation, its scope and a readable interpretation. Distinguish checks that were executed from checks that are merely provided as code; generation and syntax checks do not validate numerical results. Show how readers can inspect and save the relevant report or evidence without silently launching more computation. Break the workflow into readable notebook cells and explain the mathematics and purpose of nontrivial code. Keep authoring status and internal review notes outside the tutorial.

For the scientist-first redesign, the beginner path includes actual bounded default Aer execution. Mark resource-only, host/reference, subroutine and full-circuit paths accurately. Provider submission, costly verification and larger studies require explicit selection and their existing authorization. Code freshness or printed planning metadata does not establish execution or accuracy.

The percent-format sources in `examples/generators/` are canonical. `examples/generators/build_notebooks.py` generates each notebook from its source, so the notebook runs the same code with the same deterministic seeds where applicable. Edit the source and regenerate the notebook. Check both notebook freshness and actual shared-kernel execution when affected, because neither substitutes for the other. No test requires every public field to be named in an example or a fixed intro/scientific pair.

Update examples when their public owner changes; do not keep aliases for old import spellings. Keep presentation logic in the example and scientific implementation in its real package owner. Merge examples only after their distinct scientific operations have working destinations. Test actual workload/output semantics with bounded fixtures; a filename inventory is not proof of coverage.

## Reporting Standard

Reports derive from actual selected records, assessments, receipts, observations and analyses. Keep formatting separate from computation, with `print()` in presentation code. Reports, loading and certificates do not perform mapping, compilation, eigensolves, state reconstruction, optimizer replay or reference execution. Resource operations return their own scoped quantities and evidence.

Required report sections:

- Report metadata: report schema version and generated package versions.
- Problem summary: equation, matrix/operator dimensions, boundary conditions, units where applicable, and input data source.
- Algorithm summary: algorithm name, subroutines used, approximation order, theoretical equations, and relevant paper references.
- Execution summary: backend, execution mode, shots, random seed, Qiskit version, provider package version, and transpilation settings when applied.
- Parameters: tolerances, beta/kernel choices, QPE precision, Trotter steps, state-preparation method, block-encoding convention, QSP degree/phase count, ancilla counts, and truncation parameters.
- Resources: selected workload and evidence basis, actual preparations/attempts/observations, applicable gate inventories and target provenance, independent capacity/consumption quantities, and unknown costs. Text and machine-readable fields derive from the same stored records.
- Evidence: requested accuracy and conditioning, uncertainty sources, actual selected checks/reference receipts and thresholds, and unavailable or conditional conclusions. An unrequested reference value stays unavailable.
- Interpretation: short explanation of what the output means for the scientific problem and what limitations remain.

Reports should be printable, serializable to JSON, versioned, and stable enough for tests.

Each report declares its current format and preserves complete scientific fields, concrete meanings and identities through supported loading. The [API reference](api/workflow.md#planning-execution-and-reports) documents current types and explicit loading inputs. Persisted Source text never selects an executable import or recreates a live binding. Current-format validation remains required; development-schema converters do not.

Static and adaptive reports preserve the actual acquisition settings and compatible observation streams, with adaptive controller history where relevant. Seeds alone do not prove independence. Error composition and checks keep their frames, assumptions and provenance; nominal intervals/residuals do not establish total accuracy or ground identity. See [scientist.md](scientist.md) and method guides for the currently supported report/verification surface.

## Documentation and Commenting Standard

Use Google-style docstrings. Docstrings for scientific methods must include:

- What equation or algorithmic step is implemented.
- Paper reference, equation number, theorem, algorithm, or section location.
- Inputs and units or normalization conventions.
- Output meaning and shape.
- Approximation assumptions and known limitations.

Inline comments should explain non-obvious scientific or circuit logic, not repeat the code. Keep research provenance and citations, and remove descriptions of superseded implementations.

Every hardcoded engineering value must have a comment at its definition explaining the choice and the condition for revisiting it, plus an entry in [ENGINEERING_CONSTANTS.md](ENGINEERING_CONSTANTS.md). This includes instance-size guards, numerical tolerances, safety factors, accuracy factors, schedule defaults, and budgets. Document paper-derived values with their equations and engineering choices with their rationales. A historical value needs a still-applicable scientific or engineering basis; its age alone does not justify keeping it.

Public record docstrings use `Args:` or `Attributes:` entries whose names match the actual dataclass fields. Literal-valued fields enumerate their accepted spellings. Scientific quantities state units, normalization, and shape when those are not obvious from the type.

## Testing and Validation Standard

Scientific independent verification and validation (Scientific IV&V) is the review framework for substantive scientific, execution and resource changes. Independently derive the expected relation and assess whether the implementation, its actual consumers and its evidence support the requested quantity. Resource and lifecycle violations are acceptance concerns alongside numerical defects. The [Maintenance Runbook](MAINTENANCE.md#scientific-ivv-review-selection) defines applicability, evidence reuse, metamorphic relations and reporting. Every substantive review must assess the relevant check families and close necessary evidence gaps within the authorized scope. This does not require every test family or experiment to run for every change.

Use layered tests:

- Unit tests for subroutines: state preparation, LCU, Trotterization, block encoding, QSP phase application, QPE likelihood/objective helpers, measurement post-processing, and reporting.
- Circuit tests: small circuits compare `Statevector.from_instruction()` with expected states or unitary action.
- Shot tests: measured circuits run with fixed seed and broad statistical tolerances.
- Scientific validation tests: compare algorithm output with classical references on small, named datasets.
- Cross-implementation equivalence tests: when a subroutine slot offers more than one implementation, build the same instance through each and check statevector equivalence or documented trade-offs (see Subroutine Implementation Selection).
- Regression tests: keep small JSON/NPZ expected outputs only when their provenance is documented.

### Suite-Runtime Discipline

Suite sizing, placement of external-data validation runs, and platform-variance rules live in [MAINTENANCE.md](MAINTENANCE.md#test-runtime-and-platform-variance).

### Accuracy and Tolerance Discipline

Every numerical test tolerance must belong to one of the following classes. Identify the class in the assertion or an adjacent comment:

1. **Machine-precision identities** (tolerance justified by scale, conditioning and numerical representation): exact mathematical identities, including unitary equivalence of two synthesis paths, reconstruction of an operator from its own decomposition, circuit-vs-theory agreement when both use the same dense arithmetic.
2. **Method-error-bounded** (tolerance derived from the documented approximation model): quadrature truncation, Trotter order, QPE phase resolution. The tolerance must follow from the method parameters used in the test, with independent evidence for the claimed approximation relation. Reuse an applicable existing convergence witness, or use the smallest discriminating refinement check when needed; this does not authorize a new parameter sweep. State the tolerance derivation.
3. **Statistical** (`k * sigma` with fixed seed): shot-based assertions must state the shot count and use a bound derived from the estimator variance (for example binomial `sigma = sqrt(p(1-p)/S)`), not an arbitrary constant.

Two additional rules:

- **Regression anchors.** Keep fixed-instance/seed reference values that detect a real unintended change, alongside independent scientific relations. Their tolerances follow the actual numerical problem. No per-algorithm anchor count or universal decimal threshold replaces that evidence.
- **Mutation-probe selection.** Mutation probes are reserved for scientific equations, signs, normalizations, error compositions, resource laws, or algorithm-defining constructions that have an independent oracle and could otherwise return a believable but scientifically wrong result. Defaults, metadata shape, ordinary validation branches, and solver-policy literals remain direct-test responsibilities. There is no per-algorithm probe quota.

Acceptance for an algorithm follows its claimed domain and affected public relations. Use a representative statevector and measured-shot witness when those capabilities are claimed, an actual public example, independent scientific quantities and documented equations. Move existing discriminating evidence to the new owner rather than add tests merely to satisfy a count. Report partial or unknown scope explicitly and apply the mutation selection rule only where it adds evidence.

### Platform-Dependent Quantities in Tests

Platform-dependent testing rules live in [MAINTENANCE.md](MAINTENANCE.md#test-runtime-and-platform-variance).

### Policy Changes and Their Enforcement Tests

Policy/enforcement synchronization lives in [MAINTENANCE.md](MAINTENANCE.md#test-runtime-and-platform-variance).

## How to Add a New Algorithm

Define the scientific input/output domain and actual access, implement the concrete method protocol at its owner, and bind the real numerical/native construction, readout and analysis. Use existing subroutines and the common execution/storage/report lifecycle. Follow the current [author guide](algorithm_protocol.md) for shipped signatures; an authorized redesign updates the guide and runnable extension together. Do not add family unions, Source-driven executable loaders or an unfinished scaffold as a substitute for a working method.

Provide a bounded public example and the independent scientific/execution evidence needed for its claims, including statevector or shot readouts when supported. Put original equations, sign/normalization conventions and approximation limits beside the corresponding source calculation. Publish capability statements only for validated behavior. Keep example input builders separate from the generic solver.

## How to Add a New Subroutine

1. Add the implementation under `nwqlib.subroutines`.
2. Define a narrow interface with explicit input/output shapes.
3. Include paper references and equation locations in docstrings.
4. Provide a small standalone circuit test if the subroutine builds circuits. If the subroutine is itself a quantum algorithm or circuit primitive, this circuit test is mandatory.
5. Add a short integration test showing at least one algorithm can call it.
6. Add report metadata so downstream algorithms can explain that the subroutine was used.

## README Requirements

The root README states validation scope accurately:

- State what is validated now and what remains deferred or resource-only work.
- Link to this framework document and to [ROADMAP.md](ROADMAP.md).
- Give installation commands for the actual package and selected extras.
- List dependencies and Qiskit target versions.
- Cite the scientific papers that define the implemented methods.
- Avoid implying that archived outputs are validated library guarantees.
- Point developers to tests and examples for each algorithm.
