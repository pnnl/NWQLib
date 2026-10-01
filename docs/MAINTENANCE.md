# Maintenance

<a id="maintenance-runbook"></a>This page gives the checks and procedures for each kind of change. The rules they enforce are in [Contribution policy](FRAMEWORK.md), and [Set up, test and build](development/setup.md) gives the basic commands. The user guides describe the current API and change with it. Read old reports against the current rules.

## Validation commands

Install the scientific dependencies that the chosen tests need first. The `dev` extra supplies the test tools but not Qiskit, which the test suite's `conftest.py` imports during collection. `.[dev,aer]` is enough to collect the tests, but the full suite needs the further extras of the [CI environment setup](#support-and-external-data-validation) below and, where applicable, the separately built native executables. The SDK-free audit scripts run without those SDKs.

The commands below are a menu. Run the ones that check the relation your change affects, not every suite, golden generator or example. Look at how much work a command does before running it, and reuse results that your change does not affect. Use the environment described by `.python-version` and `docs/ENVIRONMENT_LOCK.txt`, and check exit codes. Check that `python` is the interpreter of that environment and that `nwqlib` imports from your checkout, and record the commit with the results. Do not add expensive or poorly scaling checks to the default test run or to a library code path, because every pull request to `main` runs the full suite. Propose such a check in the pull request with its cost, and a maintainer decides whether to add it.

```bash
OMP_NUM_THREADS=1 python -m pytest -n auto            # full suite, parallel workers
ruff check
python docs/scripts/policy_lint.py                    # source documentation contracts
python docs/scripts/contract_specimen.py --check       # analytical development fixture
python docs/scripts/check_core_records.py             # fresh-process SDK-free public records
python docs/scripts/check_prepared_records.py         # real planning/preparation-record import-attempt audit
python docs/scripts/check_algorithm_protocol.py       # inert discovery and explicit Method selection, with poison controls
python docs/scripts/check_ordered_operator_records.py # actual ordered mapping/metadata SDK audit
python docs/scripts/check_ordered_operator_records.py --poison # MUST fail: caught SDK import
python docs/scripts/check_qasm_records.py             # direct writer import-attempt audit
python docs/scripts/check_profile_records.py          # actual five-axis assessment/import-attempt audit
python docs/scripts/check_scientist_records.py        # actual root two-method workflow/import audit
python docs/scripts/check_qls_records.py              # explicit dense-dilation QLS classical models; actual archive/reanalysis import audit
python docs/scripts/check_qls_records.py --poison     # MUST fail: caught SDK import
python docs/scripts/check_qpe_records.py              # four actual QPE choices with injected work, no native acquisitions
python docs/scripts/check_qpe_records.py --poison     # MUST fail: caught SDK import
python docs/scripts/check_qhd_records.py              # actual QHD planning/injected host/readout and AL run/archive import audit
python docs/scripts/check_qhd_records.py --poison     # MUST fail: caught SDK import
python docs/scripts/check_adapt_records.py            # actual ADAPT planning/scalar comparison import audit
python docs/scripts/check_df_records.py               # supplied-factor conversion and FixedGCIM scan import audit
python docs/scripts/check_adapt_records.py --poison   # MUST fail: caught SDK import
python docs/scripts/check_lchs_records.py             # injected primary host workflow/import audit; no native evolution
python docs/scripts/check_lchs_records.py --poison    # MUST fail: caught SDK import negative control
python docs/scripts/check_scientist_records.py --poison # MUST fail: caught SDK import negative control
python examples/generators/build_notebooks.py --check  # read-only notebook freshness
python -m mkdocs build --strict                       # documentation composition
python docs/scripts/policy_lint.py --site-dir site     # source checks plus rendered API/link checks
python docs/scripts/mutation_probes.py --probe NAME   # affected scientific relation only; EXCLUSIVE
python docs/scripts/golden_snapshot.py --generate OUT_A   # before your change
python docs/scripts/golden_snapshot.py --generate OUT_B   # after
python docs/scripts/golden_snapshot.py --compare OUT_A OUT_B
```

- `mutation_probes.py` rewrites source files while it runs. Never run other work against that worktree concurrently, and check `git status` afterwards. Its internal per-probe pytest runs stay serial by design. All selected source replacements must resolve before the baseline runs. Before the first write, the deduplicated selected tests must pass on unmodified source with no skips or expected failures. Missing selected extras, unavailable tests, and pytest infrastructure errors fail the campaign. The hosted `Mutation Probes` workflow must install every module that a probe `requires`, which `tests/test_code_review_fixes.py` checks against `.github/workflows/mutation-probes.yml`. A pytest run that takes longer than `PYTEST_TIMEOUT_SECONDS` (600 s) counts as such an error, so a mutant that makes a test loop stops the campaign, and the original source is restored first. Only an actual test failure after that baseline counts as a kill. Strict XPASS is unavailable evidence because the test body passed, even though pytest includes it in its failed count. `--probe` limits both the baseline and the mutations, so unrelated optional tests are not required.
- The `-n auto` workers come from the `pytest-xdist` dev extra. The `OMP_NUM_THREADS=1` guard stops each worker from spawning a full BLAS thread pool. Single-test invocations skip `-n` (worker startup would dominate), and pytest's configured `addopts` already applies `-q`. Passing it a second time hides the summary line.
- Golden artifacts are not committed. When an affected recorded relation needs a comparison, use the actual pre-change source in an isolated checkout and the same applicable workload. Verify both checkouts and their package import locations before comparing results. Do not generate a whole historical corpus for an unrelated change.
- There is no maintained golden corpus or continuously passing golden CI job. The snapshot tool supports explicit before/after comparisons of its selected LCHS, QLS and QHD fixtures. It does not establish all-family, shot-path or provider qualification.
- Quick CI directly runs `contract_specimen.py --check` and the core/prepared record audits. Full CI directly checks the specimen and runs the core, prepared and QASM record audits through `test_core_records.py` in the full suite. The other SDK-isolation audits and poison controls above are explicitly selected checks, not an assertion that every CI job runs them. The existing Mutation Probes workflow runs the whole current probe collection and its deduplicated baseline on main-targeting pull requests or manual dispatch, with a 45-minute timeout. Local `--probe` usage is the bounded alternative.
- Native NWQ-Sim tests require `NWQLIB_TEST_NWQSIM_EXECUTABLE` to identify a compatible executable. Record that condition and the actual passed/skipped counts, because a default environment with no executable does not establish those native relations.
- Golden generation fixes run/snapshot/attempt UUIDs and fixture timestamps for comparison while keeping the complete provenance graph. These timestamps are not timing evidence. Production identifiers and clocks remain unchanged.
- The strict MkDocs build checks documentation composition. Then run policy lint with `--site-dir site` to check actual API entries and links in the generated pages. The notebook freshness command separately checks notebooks against their percent-format sources.
- Default policy lint checks documented dataclass fields, implementation maps, the code locations on the mathematics page and registered constant definitions, internal Markdown file links at every documentation depth, and ```` ```math ```` fences, which must not contain `<` before a letter because MkDocs passes their content to the page unescaped. It uses only the standard library. The optional rendered check reads an existing build, verifies directive targets and class/function documentation on API pages, and checks document links and HTML anchors. It performs no build, SDK import or network request. These checks do not determine whether explanatory prose accurately describes the code, which still needs review.
- When a change affects an existing example output, compare its scientific/default/decision semantics using the appropriate current expectation or a bounded before/after capture. A documentation-only edit does not require running every scientific example.

Each script accepts `--help` before audit or generation work and rejects unknown flags. The core, prepared, QASM and profile audits normally run both a clean child and a poison child. Use `--child --poison` to run their failing control directly. The protocol audit's corresponding control is `--child --poison MODULE`. For scripts whose help lists a standalone `--poison`, that option directly runs the failing audit. A poison-control failure is expected only when it reports the intended forbidden import. An argument or environment error is not evidence that the control worked.

`qsp_phase_solver_sweep.py --help` describes an optional numerical characterization tool. Its default selects 8 degrees × 4 coefficient classes × 3 seeds, or 96 phase solves. With `--evolution` it instead runs 1000 Hamiltonian-evolution preparations over its default `--taus` and `--epsilons`. On an Apple M3 Max with Python 3.12.14 and SciPy 1.18.1, a degree-128 `near_sup` solve took up to 21 seconds and the evolution default about 95 seconds. Run it only to answer a specific characterization question, with explicit `--degrees`, `--classes`, `--seeds`, `--taus` and `--epsilons`. It is not part of the ordinary validation menu or a default test.

## Change recipes

### Fixing a bug

Reuse or strengthen an existing test for the affected production accuracy, consistency, scientific logic, public behavior, or material resource invariant. Add a permanent test only for a distinct current failure mode without a suitable test owner. Demonstrate pre-fix failure and post-fix success with proportional evidence, using golden comparisons when recorded outputs are affected. Temporary audit reproducers do not automatically become permanent tests.

### Deliberately changing recorded behavior

Mark the golden comparison EXPECTED-DIFF and list the artifacts and fields expected to change before editing. Explain and verify anything outside that list before acceptance. Update the affected example expectations in the same change. Removing a comparison cannot turn unknown evidence into a stronger validation verdict. Establish the intended default, output and stopping behavior before adapting a failing assertion. An observed new value is not its own justification.

### Accepting scientific and resource changes

Acceptance connects a specific claim to evidence that could falsify it. Passing test totals, internal consistency, and a recorded implementation choice do not establish all public behavior. Scale the following checks to the changed relation. They are development checks, not extra work in the production algorithm.

- Establish the quantity's mathematical domain before assessing evidence, following [Mathematical definitions before evidence](FRAMEWORK.md#mathematical-definitions-before-evidence). Check one invalid-domain case and a legal preservation case when that boundary changes. A documented numerical-roundoff exception is distinct from accepting an impossible value as uncertain evidence.

- Follow the changed result to its actual consumer. A public circuit must preserve its intended action under the supported native backend/transpilation path as well as the library's execution helper. Include control, inverse, global phase, or instruction-position cases when those transformations affect the claim. A helper's own matrix method is not an independent numerical reference.
- Compare actual expensive events at the same workload, accuracy, mode and environment. Count repeated preparation evolution, operator observations, compilation, shots and data growth across iterations or parameter updates when affected. A forbidden-call test alone cannot detect repeated allowed work. Distinguish physical measurement settings, represented workloads, simulator trajectories and executed events in resource records.
- Derive resource projections from all operation classes in the selected construction, including the cost of adding controls to single-qubit gates and the projector/reflection work. A multiplicative calibration factor does not replace missing operations. A representative sample needs a structural relation to the represented population. Zero terms, angle-specific simplification and compilation context can break apparent equivalence. Unknown cost remains distinct from zero and from an exact estimate.
- Compare against an independent scientific quantity or observed event, and include a legal case that could expose overcorrection. Expected widths, ranks, outcomes and costs must not be derived from the same possibly incorrect output. When changing cache behavior, exercise an actual parameter update and its later consumer, not only a manually populated cache.
- Before deleting or renaming a test or mutation target, identify its current protected relation and the evidence that will cover it. Retire obsolete implementation mechanisms without reviving unused production work. Update source replacements and selected test nodes together.
- Validate the final commit of the pull request, and record that commit and the environment with the results. Run the checks that apply, including the affected mutation probes when the change touches a probed relation. Listing a command here does not ask you to dispatch a hosted workflow or run a long campaign. Report skipped or unavailable checks separately from those that passed. A reviewer other than the author inspects every consequential change to scientific results, to behavior that users or other library code rely on, and to resource use. Reuse valid evidence that the change does not affect, and repeat an expensive check only when the change raises a new question about it.

Keep default execution, scientific validation and error-controlled claims distinct. Optional reference work stays explicit. A repair should correct the numerical or execution owner before considering an auxiliary check. It must not introduce reference solves, full states, automatic retries or expanding output capture merely to justify a status or report.

#### Scientific IV&V review selection

For a substantive NWQLib change, acceptance must connect the selected Problem and Method to the actual execution, requested output, saved result and downstream consumer where applicable. Assess independent-reference, metamorphic, resource, lifecycle, detector-strength and convergence evidence according to the affected relation. Classical and algebraic paths do not require native execution. Complete the checks the change needs, and cite earlier evidence that still applies with its commit, environment and scope. Explain material exclusions or unresolved gaps.

In particular, changes to operator conversion, physical recovery or composition need the applicable transformation laws below. Changes to Run, collection, retry or save/load need relevant failure and recovery cases. Changes to loops, cache or preparation need actual event and allocation/lifetime evidence, including growth with circuits and history. A load/report that solves a reference problem or a checkpoint that repeatedly scans all history can violate its cost contract despite correct numerical output. Resource severity follows the violated contract and impact.

Changes to approximation controls or stopping rules need their method's control-to-error or uncertainty relation at the requested quantity. Finite-shot error need not decrease for each seed as shots increase, and lower iterative energy alone does not prove correctness or ground-state identification. Targeted mutation follows FRAMEWORK's selection rule when detector strength is uncertain. An applicable pre-fix failure or existing mutation may already establish detection.

Claims that an allocation, fallback or optimization improves outcomes need valid component controls at matched workload, accuracy and total cost. Empirical contribution claims can require ablation, while an independently justified operation-count reduction can use event evidence. Ordinary correctness repairs do not automatically require a study. Lint, type and dependency checks provide engineering evidence. Formal verification is selective for precisely stated core properties.

For each material claim, report its independent expected relation, a falsifying case, a legal preservation case when relevant, executed results or reusable evidence, and uncovered scope. These are review requirements, not additional runtime checks or a new result schema. The review and cost rules above still apply.

#### Metamorphic relations and their premises

Derive each relation from the actual contract before choosing test values. Include domain-invalid inputs and a legal preservation case where the changed boundary could reject valid inputs. Choose tolerances before observing the new output, using the relevant approximation, conditioning and statistical model. Cross-representation agreement alone can miss a shared error, so connect the relation to an independent anchor when necessary.

| Transformation | Expected relation and limits |
| --- | --- |
| QLS simultaneous scaling | For invertible `A`, `(A,b) → (cA,cb)` with nonzero `c` preserves the exact physical solution. Both instances must satisfy the selected Method's domain and numerical representation. Complex scaling need not preserve a Hermitian-only domain. Approximate results are compared under their physical error budgets, not by bitwise equality. Circuits, success probabilities and resource costs need not stay equal. |
| RHS / output scaling | For fixed invertible `A`, `b → cb` gives `x → cx`. For a fixed Hermitian observable `O`, `x†Ox` scales by `abs(c)**2` and `x†Ox/(x†x)` is unchanged when both states are nonzero. At `c=0` or `x=0`, the physical quadratic form can be zero but the normalized quantity is undefined. If solution errors are bounded by `e0` and `e1`, the corresponding covariance discrepancy is bounded by `e1 + abs(c)*e0`, plus justified numerical error. Scalar output budgets need their own propagation. |
| Representation equivalence | Dense, sparse and Pauli inputs representing the same operator must agree on the requested quantity within their justified errors on the common supported domain. Match physical coordinates and observable meaning. Different spectral enclosures, approximation choices or sparse access costs need not produce identical moments, ranks or resource records. |
| Wire permutation | Transform input, circuit and readout together. For a permutation `P`, use `P U P†`, `P psi` and `P O P†` in the corresponding spaces. Compare the mapped ordered amplitudes, marginals or measured quantity. Total probability alone cannot detect a misplaced wire. |
| Circuit composition | For supported control, inverse and workspace reuse, check the required operator action on discriminating inputs, including superpositions and relevant workspace sectors. All-zero state preparation alone is insufficient. A global phase that is irrelevant to an isolated state can become a relative phase under control. Do not require an action outside the circuit's documented workspace domain. |

Existing starting points include `test_original_matrix_scaling_preserves_selected_inverse_and_mass` and `test_inverse_preserves_physical_complex_rhs_scale` in `tests/test_qls_primary.py`, and `test_quadratic_form_preserves_magnitude_phase_and_zero` in `tests/test_expectation_current.py`. They cover specific selected inputs and paths, not every transformation above. Extend their actual protected relations when affected instead of adding a duplicate suite.

MorphQ of Paltenghi and Pradel, [arXiv:2206.01111v2](https://arxiv.org/abs/2206.01111v2), studies quantum-specific metamorphic transformations for testing Qiskit. It motivates the testing approach, while NWQLib's relations and acceptance limits follow from its own contracts.

### Adding an engineering constant

Register its value, owning site, rationale, and revisit condition in [ENGINEERING_CONSTANTS.md](ENGINEERING_CONSTANTS.md). Add a mutation probe only when an independent test witnesses a real scientific, numerical, or resource defect.

### Adding a scientific configuration field

Document its meaning, domain and effective default at the actual owner and API reference. Add it to an example when it helps the example's scientific decision, not to complete an exhaustive field tour. Changing a default changes behavior, so verify the affected independent relation and any applicable recorded output. Update obsolete field/tour enforcement at the owner.

### Removing or renaming a public parameter

Remove the old spelling in the same change, as [API stability](FRAMEWORK.md#api-stability) describes. Update the algorithm guide with the replacement and check call sites in src, tests, docs, and examples. Verify supported replacement behavior through an existing suitable test owner. Do not add exact-signature snapshots or ordinary unknown-keyword `TypeError` tests solely to record the removal. Add evidence only for a distinct production obligation that remains uncovered. Renames of entry points follow [Public surface](#public-surface) and need no compatibility shim.

### Adding an algorithm

Follow [Add a method](algorithm_protocol.md) for the interface and [Add an algorithm](FRAMEWORK.md#how-to-add-a-new-algorithm) for what a new algorithm needs before it is accepted, beyond passing the policy lint and the tests.

## Common maintenance errors

1. **Derive inventories from their definitions.** Never hand-copy a name list, gate list, or constant that exists elsewhere. Derive it from the owning definition. If a projection must be duplicated, give that projection a direct test that fails on divergence. No repository-wide inventory gate provides this guarantee.
2. **Check documentation against the source.** Verify each named file, function, and behavior in the code it describes.
3. **Demonstrate that checks detect their target defects.** Run each new test or check against the defect it is intended to catch.
4. **Use the right source for expected values.** Derive relocatable paths and configuration limits from their owner. Derive expected scientific quantities and structural counts independently. Copying a computed output or its implementation formula can hide a shared error.
5. **Test the claimed behavior.** Add a regression check that fails if the property a change promises stops holding.
6. **Keep SDK translation at its boundary.** Reuse an existing adapter when callers need the same translation, and do not create a generic wrapper for a hypothetical future dependency.
7. **Check rejection at the responsible module.** Owner-module tests exercise the exact planning or builder path so indirect simulation, dense-reference, resource-replay, and transpilation work is caught.

## Public surface

[Package import surface](FRAMEWORK.md#package-import-surface) and [API stability](FRAMEWORK.md#api-stability) state the rules for public names, how to change them and the tests that check them. The API reference and the source document the current operations.

## Tolerance classes in tests

Tolerance classes and their numerical meaning are defined in [Accuracy and tolerance classes](FRAMEWORK.md#accuracy-and-tolerance-discipline). Choose instances with enough headroom that expected platform variation cannot cross the assertion boundary. A tolerance is justified by the independent oracle, approximation bound, or sampling variance. Copying an implementation literal into a test is not an independent justification.

### Numeric assertion forms

| Quantity | Required assertion form |
| --- | --- |
| libm or special-function floats | Use `pytest.approx(..., rel=..., abs=...)`. Derive `rel=` in an adjacent error-budget comment. |
| Identities between recorded fields | Compute once, share that same-run value, and compare exactly. |
| Synthesis or transpile counts | On seeded instances, assert bounds, ordering, or structural laws, never platform-specific literals. |
| Genuinely exact values | Use literal equality with a short comment naming the exactness argument. |

## Test runtime and platform variance

- Keep ordinary tests on the smallest instance that exercises the scientific contract. Run long external-data reproductions separately with the local checks below.
- Measure a slow test before weakening it. Preserve the independent oracle and reduce redundant setup, dimensions, or repeated transpilation first.
- Hosted-runner layouts and gate counts may vary with transpiler and platform. Pin exact counts only when the target and transpiler configuration are part of the contract. Otherwise assert invariant structure or a documented band.
- Hosted-runner floating point can differ from a development machine. Prefer seeded generic-spectrum instances for transpilation, and do not require bitwise equality between independently evaluated floating-point reductions.
- When a derived integer is exact by construction, do not obtain it by flooring a platform-varying floating-point ratio. Otherwise document enough distance from the nearest rounding boundary that platform variation cannot flip it.
- Do not assert ordinary wall-clock performance. An elapsed-time assertion is reserved for an order-of-magnitude regression tripwire and must name the failure class rather than claim a portable runtime.
- Policy changes update their owning document and enforcement together. Direct tests own local behavior. The policy lint is reserved for cheap cross-file invariants.

## Support and external-data validation

NWQLib has one package version and codebase. Its `stable` and `latest` CI jobs name two execution environments. `pyproject.toml` sets the minimum versions of the base requirements and of each extra, and [Qiskit and dependency policy](FRAMEWORK.md#qiskit-and-dependency-policy) lists the base requirements. Scientific extras declare their direct SDK requirements. The developer and notebook extras do not bring in every algorithm dependency.

The lightweight `CI` workflow runs on ordinary pushes and on pull requests targeting branches other than `main`. It installs Ruff, jsonschema, Pydantic, NumPy, SciPy and SymPy under the lock-file constraints, runs Ruff and the stdlib-only policy lint, checks the analytical development fixture, and runs both the public core-record and real prepared-record import-attempt/interchange audits with swallowed-import negative controls. The prepared-record audit uses numerical inputs but performs no native preparation or submission. This workflow does not install NWQLib or vendor SDKs, execute pytest, or build the documentation. The full test suite additionally checks the fixture's independent scientific relations and bounded public-builder bridge, plus production record domain and revision invariants.

`Full CI` runs for pull requests targeting `main` or manual dispatches. Both environment jobs explicitly install the development, Aer, QASM, documentation, notebook, chemistry, IBM, IonQ, and tensor dependencies. They run `pip check`, import the provider SDK symbols without credentials, run Ruff, build the documentation with strict MkDocs checks, run source and rendered-site policy lint, and execute the same full test suite including examples. Each job publishes report-only coverage and its resolved environment. `Mutation Probes` runs separately in the stable environment for the same pull-request target or a manual dispatch. Opening, reopening, or updating a `main` pull request runs the full checks before merge. The push after merge runs only the lightweight checks. There are no scheduled runs.

The stable environment uses the exact Python version in `.python-version`, currently 3.12.14. `docs/ENVIRONMENT_LOCK.txt` is a constraints snapshot of the validated stable environment. Use it with `-c` and `--build-constraint`, and select the desired NWQLib extras explicitly. The recorded `scikit_tt` URL supplies the audited tensor revision. To recreate the full stable CI environment from the repository root:

```bash
conda create -n nwqlib "python=$(cat .python-version)"
conda activate nwqlib
python -m pip install --upgrade -c docs/ENVIRONMENT_LOCK.txt pip
python -m pip install -c docs/ENVIRONMENT_LOCK.txt --build-constraint docs/ENVIRONMENT_LOCK.txt -e ".[dev,aer,qasm,docs,notebook,chemistry,ibm,ionq]" "$(sed -n '/^scikit_tt @ /p' docs/ENVIRONMENT_LOCK.txt)" pytest-cov
python -m pip check
```

The latest environment selects the latest stable Python and packages compatible with it, using `--upgrade --upgrade-strategy eager` with final dependency releases. It installs the tensor extra from upstream HEAD and checks that selected direct index dependencies have not been downgraded below their latest stable releases compatible with that Python. Its resolved environment and test results provide the evidence for that combination. Promote a newly validated environment to stable by updating `.python-version` and `ENVIRONMENT_LOCK.txt` together after the full checks pass.

Compatibility validation checks the latest stable Python and compatible direct dependencies and records the resolved environment. `pip check` establishes consistency between installed package requirements, while passing tests establish the exercised behavior. Transitive requirements still apply, including SymPy 1.14's `mpmath<1.4` constraint. Compatibility evidence applies to the tested combinations. Later upstream releases require a new check.

The separate `Nexus Offline Qualification` workflow runs on the same main-target pull requests and manual dispatches. It pins the locally qualified Nexus SDK combination, keeps pandas 3.0.5, resolves all other dependencies normally and installs only qnexus 0.49.0 with `--no-deps`. Its dependency check requires exactly the documented pandas upper-cap conflict and fails on any different or additional conflict. Real SDK imports precede offline adapter, result and durable workflow tests. This job has no credentials or service calls. It neither establishes a clean `pip check` nor changes the stable/latest Full CI environment claims. See [Nexus qualification](nexus.md#costs-timeout-and-qualification) for its scope.

## When to split a file

Split a file when evidence shows that a responsibility boundary obscures ownership, couples independently changing behavior, or impedes debugging. Split along that boundary while preserving the public workflow and scientific meaning. A fixed number of debugging sessions or a line count does not justify or prevent a split. Algorithm packages follow a common layout without cross-algorithm base classes.

## Notebook generation

The notebooks in `examples/` are generated from percent-format sources in `examples/generators/`. A `# %% [markdown]` line starts a markdown cell whose `# `-prefixed lines become text, and `# %%` starts a code cell. Images live in `examples/generators/assets/`. Edit a source, never the notebook JSON.

A `# %% jupyter={"source_hidden": true}` line starts a code cell whose source JupyterLab shows collapsed. An introduction can use it for its display helpers, so that the visible code is the problem, the calls to NWQLib and the analysis. The builder stores the marker as the nbformat cell metadata `jupyter.source_hidden`, and `--check` compares that metadata with the source. nbconvert HTML and GitHub ignore it and show the source, so a notebook that uses it places its helper cell after the cell with the minimal call, and a static reader still sees that call first.

`examples/generators/build_notebooks.py` rebuilds the notebooks without running them. A notebook whose cells already match its source keeps its stored outputs, and a changed source replaces it with an output-free version. `--execute` then runs the notebook in a Jupyter kernel started from the running interpreter and stores the outputs, which is how the committed notebooks are produced. `--check` reports every notebook whose cells differ from its source, and positional names restrict the build to those notebooks:

```bash
python examples/generators/build_notebooks.py qls_linear_system_intro --execute
python examples/generators/build_notebooks.py --check
```

`tests/test_examples.py` checks that every notebook matches its source. It then executes each notebook in a kernel bound to the checked interpreter, verifies the interpreter and checkout origin inside the kernel, and appends a cell that compares the notebook's main result with a reference the notebook computes independently of the quantum method. A new notebook needs such a check in that file. The execution test keeps no output file.
