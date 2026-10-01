# Error evidence

A method's `ErrorModel` describes already propagated error contributions in the frame of its selected scientific output. `ErrorFrame.from_output(problem, output)` delegates to that output's definition. Quantity, metric, units, scope, conditioning and mathematical domain are derived from the scientific question; `Accuracy` contains only a tolerance, confidence and component.

A Result does not store an accuracy assessment. An explicit later assessment consumes the saved experiment and data:

```python
assessment = result.assess(absolute_tolerance=0.01)
sampling = result.assess(absolute_tolerance=0.01, component="sampling")
```

This does not acquire data, change shots or rerun the method. Each `ClaimAssessment` stores its own `accuracy`, original experiment/data context and the evidence used. A later stricter criterion is a new assessment of the same experiment.

`Result.facts` stores the method's existing error evidence even when no criterion was selected. An assessment consumes those facts without changing the Result. A replacement `FramedFact` carries its own `failure_probability`; omitting it leaves confidence unknown, even if the original model term had a known probability. Saved assessment facts keep the probability actually used, so changing the requested confidence cannot relabel older evidence.

## Frames, premises and sufficient criteria

`FramedFact` keeps its complete scientific frame and parameter restrictions. A projected, zero-input or conditional quantity cannot silently become an unconditional full-operator bound. A bare Fact lacks this association. Unit and scope compatibility concern scientific meaning, independently of record ancestry.

A witnessed proof or bound needs a matching subject and scope receipt. A numerical estimate, observation or user assertion does not become a certified bound when it passes record validation. Unresolved premises remain attached when an assessed value is exported and reused.

`AssessmentContext` identifies the original Problem, construction, selected point, Plan, observations, contributing chunks and Result. `result_context(result)` obtains these from the attached Plan and RunData. Conflicting point restrictions raise an error. Missing admission leaves restricted evidence inconclusive.

The model combines already propagated additive bounds by the triangle inequality and declared failure probabilities by the union bound. The triangle inequality rests on a premise that the library does not check, namely that the Method has propagated every source into the output frame, so that the output error is the sum of the propagated source errors. Where each source bound then holds, the absolute output error is at most the sum of the bounds. The union bound needs no independence. If bound k fails with probability at most `delta_k`, all bounds hold together with probability at least `1 - sum_k delta_k`, so PASS at a required confidence `c` needs `sum_k delta_k <= 1 - c`. The Method that builds the model must list every error source its output needs. The library does not check that list for completeness. For example, a zero residual for the excited eigenpair of `diag(0,1)` does not establish ground identification. A missing required source stays unknown. `component="sampling"` assesses only the sampling contribution and does not establish total physical accuracy.

`ClaimAssessment.status` is `PASS`, `INCONCLUSIVE` or `NOT_APPLICABLE`. `PASS` means the complete supported sufficient criterion meets the selected tolerance and confidence. An upper bound exceeding the tolerance is inconclusive because it does not prove that the actual error exceeds it. Numerical or asserted terms remain visible in the subtotal and `unverified` list. `NOT_APPLICABLE` answers only a `component="sampling"` criterion on a model that declares no sampling source.

Every family follows one sampling rule. The `sampling` source is the statistical error of an estimate that averages or emulates the outcomes of random draws. Measurement outcomes drawn by a backend are such draws, and so are the draws a Method makes itself. Classical RWPE emulates one measurement bit from each exact signal. RFE and SPE draw their powers or frequencies at random, and their finite average over the draws estimates a sum over the draw distribution. A draw that only selects an input, such as the reference state that Lanczos draws when `initial_state` is omitted, fixes the state that the Plan records. The estimate averages nothing over it, so its effect belongs to `ground_identification` and `projected_solve`. Exact readout means that every observation is an exact probability, Pauli value, amplitude or host scalar, as with `shots=None`, or that the Result needed no observation. With exact readout and an estimate that averages or emulates no random draw, sampling contributes zero. A family whose error model lists a sampling source publishes that zero in `Result.facts` as a proved relation witnessed by the observation population (`evidence.error_model.exact_readout_sampling`), so `component="sampling"` can PASS. The observations cannot show the draws a Method makes, so the Method states them when it requests the fact. The estimates of QCELS, Expectation, Lanczos, FixedGCIM, ADAPT and LCHS average or emulate no random draw, so with exact readout they receive the zero. SPE, RFE and an RWPE Result with at least one update keep sampling unknown, because no fact bounds their draws. QLS and QHD list a sampling source only when shots are drawn, so with exact readout their `component="sampling"` assessment is `NOT_APPLICABLE` rather than missing. Counts and provider estimates draw outcomes and never receive this zero.

For example, ExpectationMethod's conditional Hoeffding radius (doi:10.1080/01621459.1963.10500830) can satisfy the numerical threshold while `component="sampling"` remains INCONCLUSIVE. Its sampling-model premise and binary64 evaluation are not witnessed certified support. The interval remains useful under its stated assumptions. A supplied replacement bound needs genuine evidence for the same frame, subject, point and failure coverage; changing a record's kind/status or deleting assumptions does not establish those facts.

## Relative criteria

A relative assessment requires a supported nonzero target scale:

```python
assessment = result.assess(relative_tolerance=0.05, reference=reference)
```

`TargetReference` distinguishes an exact target, a lower bound on its magnitude and an upper bound. An exact nonzero target or positive magnitude lower bound supplies a sufficient relative threshold. An observed estimate or magnitude upper bound does not. Reference failure coverage participates in the union bound.

If that relative scale is unavailable or zero, the threshold stays unavailable. An explicit post-run `absolute_fallback=...` selects a separate positive absolute criterion for that case. It does not modify the initial planning Accuracy interface or pretend that the relative criterion was established. Supplying both absolute and relative tolerances is invalid.

## Exact scalar arithmetic

`ExactArithmetic` checks prospective numerator and denominator bit lengths before exact Fraction operations. Its default `max_integer_bits=4096` is a finite arithmetic representation guard, not a CPU, process-memory or accumulated-work guarantee. It covers every individual finite binary64 input. Composed calculations may exceed it and reject before the operation.

The guard conservatively counts unreduced cross products. A rejected calculation may have a smaller reduced result. Raising the explicit limit enlarges the supported exact-arithmetic domain; it does not change or approximate the formula. No generic items, metadata, phase or cumulative-allocation policy is required to assess scalar evidence.

## Covariance and binary inference

`linear_variance` groups identical data IDs before applying the covariance identity. `Var(2X)=4 Var(X)`, and `X-X=0` needs no unavailable unused variance. Distinct variables need supplied cross-covariance or an explicit `IndependenceLaw`; absence never means independence. The law keeps its joint identity, frame, restrictions and premises.

Every supplied variance must be nonnegative, even when its coefficient cancels. A supplied covariance must obey `c² <= vx*vy` when both variances are known. This pairwise condition does not establish that an entire covariance matrix is positive semidefinite. A negative complete exact sum rejects; an incomplete subtotal stays unknown. Separately rounded impossible moments are not repaired without a justified producer model.

Binary inference consumes actual sufficient counts and source identities. Frozen-source reuse does not increase the sample size. Empirical variance, fixed-time coverage, time-uniform coverage and Bayesian credibility remain distinct. A stopped acquisition cannot reuse a fixed-time guarantee. A selected prior or sampling premise is explicit.

Affine binary calibration keeps its signed estimator, original calibration populations and local Jacobian. A usable negative contrast is legal. A corrected estimate outside `[-1,1]` is not clipped into a physical expectation. Parameter intervals separately preserve the calibration-box image and its justified intersection with the physical domain. Missing or incompatible data does not cause hidden calibration or reacquisition.

## Explicit checks

`CheckSpec` declares a threshold, frame, mathematical domain, prerequisites, required access and the concrete computation/data description. `assemble_check` consumes supplied facts only. A fact witnessed by a verification receipt answers only the exact `CheckSpec` whose identity it records, so a revised threshold needs a new verification. Missing evidence is `NOT_RUN`; unavailable support is `INCONCLUSIVE`; explicit inapplicability stays `NOT_APPLICABLE`.

Definition-invalid values reject before status decisions. A declared numerical-roundoff window is distinct from a tolerance for accepting scientific accuracy. Its producer must justify the scale, and the raw fact plus disclosed comparison adjustment remain available. The exact-probability window of a quantum preparation receipt, `PreparedArtifact.probability_window`, is one such window. The comment above the per-simulator constants in `_validation.py` derives it for Aer and NWQ-Sim, and [Engineering constants](ENGINEERING_CONSTANTS.md#numerical-guards-and-tolerances) summarizes that derivation. Signed metrics keep legal signed values. See [Verification](verification.md) for checks of actual saved Result data.

## Independent scalar-bound inversion

`nwqlib.evidence.resolve_scalar_bound(coefficient, tolerance, maximum=..., ...)` solves the supplied `c/n` or `c/sqrt(n)` bound on a finite positive integer domain. With `fixed_error=1/8` and `tolerance=3/8`, `c=1` needs 4 or 16 units respectively. Select the second law explicitly with `decay="inverse_sqrt"`; the default is `decay="inverse"`. Changing the tolerance computes a new scalar answer; it does not select a Plan or change a method parameter. The caller explicitly uses the answer in a new method configuration when the supplied law actually describes that parameter.

The result keeps premises, same-unit assumptions, a conservative rational covered bound and optional declared linear work. Missing quantities remain unresolved, false premises inapplicable and unknown premises conditional. No result is a total-accuracy PASS. Framed facts must agree in metric, unit, scope and conditioning; plain numerical inputs declare that agreement. Exact integer/rational operations use the existing 4096-bit guard; no registry, backend or general optimizer is used.

## Code map

Apart from the Hoeffding inequality in binary inference (Hoeffding 1963, doi:10.1080/01621459.1963.10500830, Theorem 1, Eq. (2.3), p. 15, and Theorem 2, Eq. (2.6), p. 16, as listed in [References](references.md#expectation-and-sampling)) and the Krylov-span identity behind the energy-shift check, the evidence layer has no paper source. Each mechanism below enforces one relation that keeps a stored claim from becoming stronger than its evidence. Owners are in `src/nwqlib/evidence/`.

| Mechanism | Relation it keeps | Owner |
| --- | --- | --- |
| `Fact`, `Evidence` | Availability and evidence kind are separate. Validation never turns a declared kind into witnessed support. | `records.py` |
| `ErrorFrame`, `FramedFact` | A value is used only in its own quantity, metric, unit, scope, conditioning, domain and parameter point. A replacement bound carries its own failure probability. | `error_model.py::applicable` |
| `ErrorModel.assess` | Triangle inequality and union bound over the required sources. A component criterion never covers total error. PASS needs witnessed proved or certified support without open assumptions. | `error_model.py::ErrorModel.assess`, `error_model.py::supported` |
| `CheckSpec`, `CheckDomain` | Domain admission precedes status. A roundoff window is explicit, and the raw value stays in the fact. | `error_model.py::_assemble_check` |
| `Certificate.with_verification`, `assemble_check` | A verification fact answers only the options record and `CheckSpec` that produced it. | `error_model.py::Certificate`, `verification.py::_publish` |
| `linear_variance` | One data identity is one random variable. A missing covariance is unknown, never zero. | `statistics.py::linear_variance` |
| `ExactArithmetic` | Every exact rational operation is checked against the integer-bit guard before it runs. | `_work.py::ExactArithmetic` |
| Binary inference | Sample size counts distinct counted sources. Fixed-time, time-uniform and Bayesian intervals keep distinct meanings. | `binary.py` |
