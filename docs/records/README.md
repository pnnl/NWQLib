# Scientific records

Start with a scientific input such as `Eigenproblem(A=...)`, `LinearDynamics(A=..., initial_state=..., time=...)`, or `Expectation(state=..., observable=...)`, imported from `nwqlib`. The [scientist workflow](../scientist.md) selects a Method, creates its Plan, and produces an attached Result. [Input access](../inputs.md) describes the actual operator and state handles.

Importing the scientific input classes does not load vendor SDKs or execute algorithms. Core metadata uses Pydantic and standard-library code; admitting and using numerical or symbolic inputs can use the base NumPy, SciPy and SymPy dependencies. Native execution dependencies are loaded at their selected operation boundaries.

## Ownership and scientific meaning

Core owns immutable records, exact scalar encodings, units, references and limits. Evidence owns facts, frames and assessment. Problems owns scientific questions and output definitions; the selected Method owns its applicability, numerical choices, execution and analysis.

| Problem | Scientific question | Default output |
| --- | --- | --- |
| Eigenproblem | Smallest eigenvalue of a Hermitian operator, optionally in an explicit subspace or sector | Eigenvalue |
| LinearDynamics | Time-independent `du/dt = -A u + source` from the given initial time and state | Solution |
| LinearSystem | Physical solution of `A x = b` | Solution |
| Expectation | Normalized expectation of the given Hermitian observable in the given state | NormalizedExpectation |
| SpectralEstimation | Phase information for a Hamiltonian or unitary, with a given initial state | Eigenphase |
| Optimization | A SymPy objective on ordered real variables with finite box bounds | OptimizationCandidate |
| ConstrainedOptimization | The Optimization question subject to at least one equality `h_i(x) = 0` or inequality `g_j(x) <= 0` | OptimizationCandidate |

ConstrainedOptimization stores each constraint as its residual. `Eq(a, b)` is stored as the equality residual `a - b`, `Le(a, b)` as the inequality residual `a - b` and `Ge(a, b)` as `b - a`. A plain SymPy expression is taken as the residual itself. Strict relations are rejected because `g(x) <= 0` cannot express strictness, so state a margin explicitly. `Ne`, relations that SymPy has already evaluated to True or False, matrices and constraints with symbols outside `variables` are also rejected. ConstrainedOptimization is not an Optimization, so a box-only Method such as QHD refuses it at planning instead of ignoring its constraints. The QHD family solves it with `solve_augmented_lagrangian`, a sequence of QHD box problems ([QHD guide](../algorithms/qhd.md#constrained-problems)).

A valid record is not a promise that every Method supports it. Hermitian admission, dimensions and coordinate order are checked by the informed input owners. Method-specific premises remain at the selected Method; for example, a descriptive unitary input record does not itself prove unitarity. JSON stores symbolic descriptions without evaluating source strings.

Input admission preserves physical magnitude, complex phase and original coordinates. A full-space eigenvalue and a value in an explicitly supplied subspace keep distinct scientific identities. Output scope comes from the Problem, while output-specific conditioning identifies such relations as a nonzero norm or a selected subspace.

`NormalizedExpectation` means `u† O u / (u† u)` for nonzero `u`; `QuadraticForm` means `u† O u`; `NormSquared` means `u† u`. These are separate quantities. On an Expectation input, `problem.unit` labels the normalized observable expectation. It does not supply the missing amplitude unit for a physical quadratic form, norm or vector. Their default frame unit therefore remains unspecified unless the output declares it explicitly.

For LinearDynamics and LinearSystem, `problem.unit` labels the physical solution amplitude: Solution and physical StateVector inherit it, and NormSquared uses its squared unit. A unit-normalized StateVector has a dimensionless error frame. There is no automatic conversion between unit systems. An explicit output unit records the caller's convention rather than proving it.

Solution uses physical coordinates and `l2` error. StateVector explicitly chooses physical or unit normalization and physical phase or equivalence modulo global phase; the latter uses `phase_aligned_l2`. Samples uses total variation between distributions. Eigenphase uses circular distance in dimensionless turns, with `U v = exp(2 pi i phase) v`. OptimizationCandidate uses objective gap and does not imply an optimality certificate. Declaring an output creates neither acquired data nor a live quantum state handle.

## Accuracy and facts

Accuracy specifies exactly one positive, finite absolute or relative tolerance, confidence strictly between zero and one, and either the total or sampling component. It contains no achieved-accuracy status. Its criterion is separate from the quantity, unit, scope and conditioning supplied by the output's error frame. Relative assessment needs supported reference-scale evidence; zero or unknown reference norm cannot become an implicit denominator. See [error evidence](../error_evidence.md) for explicit reference and absolute-fallback assessment.

A Fact separates availability, evidence kind, optional composition role, quantity, unit, scope and assumptions. Concrete zero is distinct from unknown. Unknown and not-applicable facts carry a reason without an invented value or evidence basis. Symbolic facts use inert Symbol and Source references. Concrete values use Float64, Complex128, Rational or an explicit boolean.

Evidence records its source, version, domain and acquisition-work provenance where applicable. A witnessed receipt also requires an artifact, subject identity and scope. Evidence witnessed by a verification receipt also names the content identity of the options record that produced the value. Receipt of a user assertion preserves `user_assertion`; neither a receipt nor schema validation proves the assertion. FramedFact can carry the failure probability actually supported by its evidence. This does not turn an unknown fact into a bound or borrow a confidence level from a later Accuracy criterion.

## Immutability, identity and payloads

Nested Record fields are frozen, typed containers detach caller-owned collections, and public JSON/dict descriptions are detached. `revise(...)` and `model_copy(update=...)` validate changes and leave the original untouched. Core and evidence records record revision ancestry through `parent_id`. Scientific Problem, output and Accuracy inputs omit revision and schema bookkeeping fields from their public field set. Unchecked `model_construct` and Pydantic `.copy(...)` are disabled; public admission rejects undeclared fields even with an extra-field override.

Record identity is the SHA-256 digest of a compact, sorted-key UTF-8 envelope containing `nwqlib.record/1`, the fully qualified record type and declared fields. Computed IDs are excluded from the digest input. Ordered collections remain ordered. Loading checks all supplied IDs, including nested ones, after domain validation and scalar normalization. Omitting an ID does not bypass admission. An identity is a content association, not source authentication or proof of an external payload. Unit compatibility compares symbol and dimension independently of revision ancestry.

Float64 uses finite binary64 values; Complex128 stores two such components. Signed zero canonicalizes to positive zero. Rational uses exact integers in reduced form with a positive denominator: `1/3` remains distinct from its binary64 approximation. Nonfinite values, boolean numbers and coercion from numeric strings are rejected by these scalar encodings.

Actual operator/state ingestion owns read-only numerical snapshots. Their JSON descriptions preserve identities and preparation metadata but do not recreate executable numerical access. Use the existing Method archive hooks through [saved evidence](../saved_evidence.md) to save and reopen Plans, Runs and Results with their required payloads.

InputRef declares a representation and identity without creating data access.

Limit identifies stage, metric, unit, kind, scope and nonnegative finite value. Memory and stored bytes are capacity stocks; transfer/materialization bytes and CPU seconds are consumption; wall seconds are elapsed-duration deadlines. Byte and count limits require exact integers, including beyond binary64's exact-integer range. The record does not execute, authorize or enforce work. Actual operation controls and ExecutionLimits are enforced by their respective execution owners.

## Current validation

Run `docs/scripts/check_core_records.py` using the checked-command procedure in [maintenance](../MAINTENANCE.md). Its fresh subprocess installs an import-attempt blocker before NWQLib imports, then performs real scientific-input admission, a two-entry operator action, schema and JSON checks, Program binding, selected-resource estimation, selected-block construction JSON round trips and Pauli readout selection. A deliberately caught forbidden import must still fail the audit. This checks dependency isolation and current record behavior without vendor SDKs or algorithm execution; it does not validate scientific method accuracy or authenticate external payloads.
