# Release notes

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

- One workflow for every method: `plan` selects the construction, `estimate` and `compare` price it before any circuit is built, `solve` runs it, and `prepare` and `submit` run it as a Run that can be saved, reopened and continued ([scientist workflow](scientist.md), [prepared execution](prepared_execution.md), [run archives](run_archives.md)).
- Resource estimates come from the gate-count laws of the selected construction, are checked against compiled circuits at small size, and extend to sizes no simulator holds ([resources](resources.md)).
- Every limit on work, memory and stored data is admitted before the work starts, and a refusal names the parameter and the amount ([inputs](inputs.md)).
- Results carry their error evidence and references, and saved Results and Runs reload and recompute ([saved evidence](saved_evidence.md), [verification](verification.md)).
- Backends: local Aer, NWQ-Sim with Slurm submission, IBM Runtime, IonQ, Quantinuum Nexus, and QASM export ([backends](backends.md)). A command line lists the methods and prints their cards ([CLI](cli.md)).

**Documentation**: a [quickstart](quickstart.md), the user guides, one guide per algorithm with its source map from code to paper equations, the [mathematics page](mathematics.md) with the proofs and resource laws, the backend guides, the [API reference](api/index.md), and the development pages from the [framework](FRAMEWORK.md) to the [code tour](CODE_TOUR.md) and the [engineering constants](ENGINEERING_CONSTANTS.md).

**Examples** ([overview](examples.md)): eight executed notebooks. Four introductions cover a linear system, linear dynamics, eigenvalues and optimization. Three paper-derived studies cover collision relaxation with QLS, heat flow with a boundary penalty with LCHS, and constrained optimization with QHD. One notebook plans five problems at 80 to 100 qubits and prices them without simulating a circuit.
