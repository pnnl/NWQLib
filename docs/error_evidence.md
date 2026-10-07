# Error evidence internals

<a id="error-evidence"></a>Each Method's `ErrorModel` lists the error contributions that the Method has already propagated into the error frame of its output. An error frame, `ErrorFrame`, names the quantity, metric, units, scope, conditioning and mathematical domain that an error value refers to. `ErrorFrame.from_output(problem, output)` delegates to the output's definition, so the frame follows from the scientific question. `Accuracy` contains only a tolerance, a confidence and a component.

This page describes how error evidence is represented and checked inside NWQLib. [Check accuracy and verify a result](verification.md) describes `result.assess`, how it combines bounds, what `PASS`, `INCONCLUSIVE` and `NOT_APPLICABLE` mean, and the sampling error of each method.

## Error frames and premises {#frames-premises-and-sufficient-criteria}

A `FramedFact` keeps its complete error frame and its parameter restrictions. A projected, zero-input or conditional quantity therefore cannot silently become an unconditional bound for the full operator. A bare `Fact` lacks this association. Unit and scope compatibility concern scientific meaning, independently of record ancestry.

A witnessed proof or bound needs a witnessing receipt, as [Record contracts](records/README.md#accuracy-and-facts) defines it, with a matching subject and scope. A numerical estimate, an observation or a user assertion does not become a certified bound by passing record validation. Unresolved premises stay attached when an assessed value is exported and reused.

`AssessmentContext` identifies the original Problem, the construction, the parameter point that ran, the Plan, the observations, the contributing chunks and the Result. `result_context(result)` obtains these from the attached Plan and RunData. Conflicting point restrictions raise an error. Evidence restricted to a parameter point applies only in an admitted context (`admitted=True`), which names the Plan and the construction that ran. Without one, such evidence stays inconclusive.

How `ErrorModel.assess` combines the bounds of several error sources, and what a `PASS` requires, is described in [Check against a tolerance](verification.md#check-against-a-tolerance). The sampling rule of each method is in [Sampling error by method](verification.md#sampling-error-by-method).

<a id="relative-criteria"></a>
Assessment against a relative tolerance, with an explicit reference and an absolute fallback, is described in [Relative criteria](verification.md#relative-criteria).

## Exact scalar arithmetic

`ExactArithmetic` checks the bit lengths of the prospective numerator and denominator before each exact `Fraction` operation. Its default `max_integer_bits=4096` limits the size of the exact integer representation. It is not a CPU, process-memory or accumulated-work guarantee. It covers every individual finite binary64 input. Composed calculations may exceed it, and they are then rejected before the operation.

The guard conservatively counts unreduced cross products, so a rejected calculation may have a smaller reduced result. Raising the explicit limit enlarges the supported exact-arithmetic domain. It does not change or approximate the formula. Assessing scalar evidence requires no generic policy for items, metadata, phases or cumulative allocation.

## Covariance and binary inference

`linear_variance` groups identical data IDs before it applies the covariance identity. Thus `Var(2X)=4 Var(X)` for a random variable `X`. The variance of `X-X` is zero, so this difference needs no supplied variance of `X`. Distinct variables need a supplied cross-covariance or an explicit `IndependenceLaw`, and a missing covariance never means independence. The law keeps its joint identity, frame, restrictions and premises.

Every supplied variance must be nonnegative, even when its coefficient cancels. A supplied covariance must obey `c² <= vx*vy` when both variances are known. This pairwise condition does not establish that an entire covariance matrix is positive semidefinite. A negative complete exact sum is rejected, and an incomplete subtotal stays unknown. Separately rounded impossible moments are not repaired without a justified model of the code that produced them.

Binary inference uses the sufficient counts that were actually acquired and their source identities. Reusing the same frozen source does not increase the sample size. Empirical variance, fixed-time coverage, time-uniform coverage and Bayesian credibility keep distinct meanings. A stopped acquisition cannot reuse a fixed-time guarantee. A prior or sampling premise is always stated explicitly.

Affine binary calibration keeps its signed estimator, its original calibration samples and its local Jacobian. A usable negative contrast is valid. A corrected estimate outside `[-1,1]` is not clipped into a physical expectation. Parameter intervals keep, separately, the image of the calibration box and its justified intersection with the physical domain. Missing or incompatible data does not trigger hidden calibration or new measurements.

## Explicit checks

`CheckSpec` declares a threshold, a frame, a mathematical domain, prerequisites, the required access and a description of the concrete computation and data. `assemble_check` uses supplied facts only. A fact witnessed by a verification receipt answers only the exact `CheckSpec` whose content hash it records, so a revised threshold needs a new verification. Besides `PASS` and `FAIL`, which compare the value with the threshold, a check reports these statuses:

| Situation | Status |
| --- | --- |
| No evidence was supplied | `NOT_RUN` |
| The supporting evidence is unavailable | `INCONCLUSIVE` |
| The fact states that the check does not apply | `NOT_APPLICABLE` |

A value outside its definition domain is rejected before any status is decided. A declared numerical-roundoff window is distinct from a tolerance for accepting scientific accuracy. The code that declares the window must justify its scale, and the raw fact and the disclosed comparison adjustment stay available. The exact-probability window of a quantum preparation record, `PreparedArtifact.probability_window`, is one such window. The comment above the per-simulator constants in `_validation.py` derives it for Aer and NWQ-Sim, and [Engineering constants](ENGINEERING_CONSTANTS.md#numerical-guards-and-tolerances) summarizes that derivation. Signed metrics keep valid signed values. [Check accuracy and verify a result](verification.md) describes checks of saved Result data.

## Readout tolerance branches {#readout-tolerance-branches}

### LCHS and QLS amplitude-derived masses

At readout and publication, LCHS and QLS resolve the amplitude-derived masses label of the producing preparation record, because their projected kernels bound their own scaled squared-norm reductions. Each uses the resulting qualified saved-state budget (`saved_state_error`, the native-state budget propagated through the envelope of the host phase correction of the saved state) when available and the preparation record's probability-check tolerance convention, propagated through the same correction, otherwise. A preparation record whose host phase correction was not assessed supplies neither, and the mass check refuses the point. Both branches include the host mass error budget, preserve the raw masses and check their nested populations. The `probability_window_exclusions` of the preparation record, in `result.data.receipts`, shows which branch applies. The budget applies when `amplitude-derived masses` is its only label. Any other label, such as `unitary` or `multiplexer` for a supplied matrix instruction, `optimization_level` for a non-default compiler level or `unchecked qiskit-aer version`, keeps the probability-check tolerance convention, and so does a preparation record whose exclusions were not assessed. For example, the preparation record of exact QLS on Aer with `A = diag(1, -2, 1.5)` lists `unitary`, so its masses are checked under the probability-check tolerance convention.

### QPE trajectory expectations

Exact trajectory readout saves the ancilla X and Y expectations at every point. A saved expectation may exceed one in magnitude only within the binary64 roundoff window of the executed trajectory, `exact_probability_window(G, n) = max(1e-12, (c*G + 2**(n+1) + 10)*u)`. Here G is the native operation count recorded in the preparation record (on Aer, without save instructions), n the circuit width and u = 2**-53. c is the executing simulator's derived first-order bound on the change of the squared state norm per instruction, about 6957 for Aer and 164 for NWQ-Sim CPU/SV. Any other target uses Aer's constant, the larger one. [Engineering constants](ENGINEERING_CONSTANTS.md#probability-windows-of-exact-execution) derives both and names what they exclude. A preparation record lists the exclusions present in its own execution in `probability_window_exclusions`. An unknown instruction count keeps the 1e-12 floor. Deep controlled-evolution trajectories need this scaling. One whole-trajectory G gives a conservative window for every point, and neither G nor the window is multiplied by the number of saved labels or positions. The observation is checked against its preparation record when it is created and again before a Result is saved. Raw saved expectations remain unchanged. If one exceeds ±1 within the window, QPESample stores the original value in `raw_mean` and uses the corresponding exact endpoint as `mean` for bounded likelihoods. The adjustment is `mean - raw_mean`, and the sample's contribution ID identifies the point chunk. The classical scalar-signal route uses the fixed `NUMERICAL_RELATION_RTOL=1e-12` window of a host kernel, preserving its original scalar in `raw_mean` and its source contribution ID. In-range means are unchanged and counts never use this exception. The window bounds roundoff in the saved expectation, apart from the preparation record's listed exclusions. It is not an accuracy bound for the estimate. Comparing a trajectory value with a separately prepared value needs the sum of their two justified error bounds, not one preparation record's window.

## Independent scalar-bound inversion

`nwqlib.evidence.resolve_scalar_bound(coefficient, tolerance, maximum=..., ...)` solves a supplied `c/n` or `c/sqrt(n)` bound for the smallest sufficient integer on a finite positive integer domain. With `fixed_error=1/8` and `tolerance=3/8`, `c=1` needs 4 or 16 units respectively. The default is `decay="inverse"`, and `decay="inverse_sqrt"` selects the second law. Changing the tolerance computes a new scalar answer. It does not select a Plan or change a method parameter. The caller uses the answer in a new method configuration, and only when the supplied law describes that parameter.

The result keeps premises, same-unit assumptions, a conservative rational covered bound and an optional declared linear work. Missing quantities stay unresolved, false premises make the result inapplicable and unknown premises make it conditional. No result is a total-accuracy `PASS`. Framed facts must agree in metric, unit, scope and conditioning, and plain numerical inputs declare that agreement. Exact integer and rational operations use the 4096-bit guard of [Exact scalar arithmetic](#exact-scalar-arithmetic). No registry, backend or general optimizer is used.

## Code map

Apart from the Hoeffding inequality in binary inference (Hoeffding 1963, doi:10.1080/01621459.1963.10500830, Theorem 1, Eq. (2.3), p. 15, and Theorem 2, Eq. (2.6), p. 16, as listed in [References](references.md#expectation-and-sampling)) and the Krylov-span identity behind the energy-shift check, the evidence layer has no paper source. Each mechanism below enforces one relation that keeps a stored claim from becoming stronger than its evidence. The code is in `src/nwqlib/evidence/`.

| Mechanism | Relation it keeps | Code |
| --- | --- | --- |
| `Fact`, `Evidence` | Availability and evidence kind are separate. Validation never turns a declared kind into witnessed support. | `records.py` |
| `ErrorFrame`, `FramedFact` | A value is used only in its own quantity, metric, unit, scope, conditioning, domain and parameter point. A replacement bound carries its own failure probability. | `error_model.py::applicable` |
| `ErrorModel.assess` | Triangle inequality and union bound over the required sources. A component criterion never covers total error. PASS needs witnessed evidence of kind `proved_relation` or `certified_bound` without open assumptions. | `error_model.py::ErrorModel.assess`, `error_model.py::supported` |
| `CheckSpec`, `CheckDomain` | Domain checks precede status. A roundoff window is explicit, and the raw value stays in the fact. | `error_model.py::_assemble_check` |
| `Certificate.with_verification`, `assemble_check` | A verification fact answers only the options record and `CheckSpec` that produced it. | `error_model.py::Certificate`, `verification.py::_publish` |
| `linear_variance` | One data identity is one random variable. A missing covariance is unknown, never zero. | `statistics.py::linear_variance` |
| `ExactArithmetic` | Every exact rational operation is checked against the integer-bit guard before it runs. | `_work.py::ExactArithmetic` |
| Binary inference | Sample size counts distinct counted sources. Fixed-time, time-uniform and Bayesian intervals keep distinct meanings. | `binary.py` |
