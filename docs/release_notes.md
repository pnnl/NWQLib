# Release notes

## v1.0.1 (2026-10-03)

This release simplifies tests and the validation code they support. It removes 107 collected test cases while keeping independent checks of numerical results, resource limits and normal save, load and resume workflows.

- QLS uses the shared Program validation. Preparation and native lowering check their own work limits before constructing circuits, without a separate planning-time simulation of those checks.
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
