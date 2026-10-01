# Finite Pauli expectation

<a id="expectation-and-quadratic-form"></a>`ExpectationMethod` computes the expectation of a Hermitian observable $O$ in a state $\psi$. By default it returns the normalized expectation $\psi^\dagger O\psi/\psi^\dagger\psi$. With the `QuadraticForm` output it returns $\psi^\dagger O\psi$, which also depends on the norm of $\psi$. The method writes $O$ as a Pauli sum $O=c_0I+\sum_j c_jP_j$ and evaluates it from exact simulator probabilities, from finite shots, from a provider's Estimator, or classically by matrix-vector products. The fixed-time and time-uniform intervals and the shot count for a target accuracy rest on Hoeffding's inequality (Hoeffding 1963, doi:10.1080/01621459.1963.10500830). The other steps are standard identities or NWQLib's own derivations, listed in the [source and code map](#source-and-code-map).

Use this method when you have a state and want the value of an observable in it, its sampling uncertainty, or the number of shots a target accuracy needs. To find the lowest eigenvalue of an operator instead, use [Chebyshev Lanczos](lanczos.md), [GCiM](gcim.md) or [QPE](qpe.md).

```python
from nwqlib import Expectation, QuadraticForm, solve
from nwqlib.algorithms import ExpectationMethod

problem = Expectation(state=[2., 2.], observable=[[2., 0.], [0., 0.]])
result = solve(problem, method=ExpectationMethod())
print(result.value)
physical = solve(problem, method=ExpectationMethod(),
                 output=QuadraticForm(observable=problem.observable),
                 execution="classical")
print(physical.value)
```

```text
1.0000000000000002
7.999999999999999
```

The exact answers are 1 and 8, because $\psi=(2,2)$ and $O=\operatorname{diag}(2,0)$ give $\psi^\dagger O\psi=8$ and $\psi^\dagger\psi=8$. The first `solve` runs this one-qubit example on Aer through the default quantum path, and the second evaluates the quadratic form classically. Replace the state and the Hermitian observable to evaluate your own expectation. Exact simulator readout does not establish an accuracy claim for hardware.

## Inputs

The quantum path needs a power-of-two dimension. It accepts a compact Pauli observable, or an explicit dense Hermitian observable of dimension at most 16, which it converts to Pauli coefficients $c_P=\operatorname{tr}(PO)/D$ without an eigensolve, symmetrization or pruning of nonzero coefficients. Quantum dimension 3 is not supported. With `execution="classical"`, the method uses the dense, CSR, CSC or Pauli matrix-vector product in the original dimension, including dimension 3, with no quantum state preparation or Pauli conversion.

Let $s=\lVert\psi\rVert^2$. The quadratic form multiplies the normalized point and interval by $s$, and its estimator variance by $s^2$. A negative observable can give a negative quadratic form. The quadratic form of the zero state is zero and is returned without measurement. The normalized expectation of the zero state is undefined.

## Exact, sampled, provider and classical readout {#readout}

The default `ExpectationMethod()` reads the observable exactly. It prepares the normalized state direction once and returns every requested nonidentity Pauli expectation from that one experiment. Identity coefficients are added algebraically. The weighted sum gives the normalized expectation, and $s$ gives the quadratic form. Every readout label refers to that shared experiment.

A positive `shots` selects measured counts. Without readout mitigation, the method measures qubit-wise commuting groups of Pauli labels. Each group uses one measurement basis and its own fresh shots, from which the parities of its labels are decoded, and the weighted uncertainties include the covariance between labels of the same group. With [binary readout mitigation](#binary-readout-calibration), each nonidentity term has its own parity circuit, and calibration circuits are added. `shots` counts the shots of each group or parity circuit. An identity-only observable needs no preparation, measurement or calibration, even when shot options are given.

`ExpectationMethod(estimate_precision=...)` instead asks a provider's Estimator for the whole weighted Pauli sum ([provider estimates](#provider-estimates)). `execution="classical"` computes the value by matrix-vector products, and it cannot be combined with shots, binary inference, mitigation or a provider precision.

### How measurement groups are formed {#measurement-groups}

The groups form by first fit in term order over the nonzero nonidentity labels. A label joins the first group whose accumulated basis has the same single-qubit Pauli on every qubit where both act, and `max_classical_products` bounds the number of these comparisons. A group's circuit prepares the state on one register that holds every qubit of the state and applies the group's basis change, which gives each X site H and each Y site S† then H to map the Pauli eigenbasis to the computational basis. It then measures the whole register into one outcome with one bit per qubit, bit i holding qubit i. The parity of label j is `(-1)**popcount(outcome & support_j)`, where `support_j` marks the qubits on which the label is not I. Measuring the qubits outside the group's support adds measurements but no basis rotation or CX gate, and summing over their outcomes gives the distribution that measuring only the support would give.

With mitigation, each nonidentity term has its own parity circuit. After the same site rotations, a CX network collects the parity of all active sites on the highest-index active qubit, called the pivot. The measured pivot bit is 0 for the +1 eigenvalue of the Pauli term and 1 for the -1 eigenvalue.

## Read the result {#reading-the-result}

Values are in the units of the requested output unless the table says otherwise. These are $v^\dagger Ov$ for `NormalizedExpectation` and $s\,v^\dagger Ov$ for `QuadraticForm`, where $v$ is the normalized direction of $\psi$. Per-setting quantities are in the units of one parity mean, before the coefficients and $s$ are applied. Raw means, `BinaryEstimate` points and per-setting intervals lie in $[-1,1]$, while a corrected point and its corner image can lie outside.

| Field | Meaning |
| --- | --- |
| `value` | $c_0+\sum_j c_j\,\mathrm{estimate}_j$, times $s$ for a quadratic form, where $\mathrm{estimate}_j$ is the exact Pauli mean, the binary point of the chosen inference, or the corrected point with mitigation. The provider path uses the mean of the provider estimates of the whole weighted sum, and classical execution the matrix-vector value. None when a label has no data, when a per-term point is unavailable (a Beta posterior whose assumptions fail, or a correction below `minimum_contrast`, without preparation records or without calibration counts), or when the value is not representable. |
| `missing`, `unavailable` | Nonidentity labels without data, and the reason for an unavailable value. `value` is None exactly when `missing` is nonempty or `unavailable` is set. |
| `physical_scale` | $\lVert\psi\rVert$ as a mantissa and binary exponent, the source of $s$. |
| `statistics.populations` | Without mitigation, one `BinaryEstimate` per nonzero nonidentity label, named `<group setting>:<label>` in group and member order. Its counts are the label's marginal parities in its group's histogram, and the labels of one group share that group's source, observation and preparation-record identifiers, which count as one measurement. With mitigation, one `BinaryEstimate` per science and calibration setting, in `plan.reconstruction.settings` order. Its `population` holds the counts `zeros` ($n_0$) and `ones` ($n_1$) of the distinct counted sources, with their identifiers. `raw_mean` is $z$, `empirical_variance` the exact rational sample-mean variance (None for $n\le1$), `point` the Beta posterior mean for `beta` inference (None when its assumptions fail) and $z$ otherwise, `posterior_variance` the Beta posterior variance of $2p-1$, and `interval` the per-setting interval evaluated at $\alpha$ (None for point inference), whose `probability` records the family level $1-\delta$. |
| `statistics.corrections` | With mitigation, one `BinaryCorrection` per science term. `contrast` is $a$, `offset` $b$, `point` the corrected $\mu$, `derivatives` the Jacobian in $(z,z_0,z_1)$, `interval_image` the corner image before intersection with $[-1,1]$, and `interval` that intersection. |
| `statistics.raw_value` | The same weighted sum over the uncorrected raw means. |
| `statistics.variance`, `variance_kind` | A `VarianceAssessment` of the estimator variance in squared output units, composed by `linear_variance`. Without mitigation it has one contribution per group, the empirical variance of the group's weighted score mean, which includes the covariance of labels in the group. With mitigation it composes the per-setting variances with the output coefficients times the calibration Jacobian. `variance_kind` is `empirical` for sample-mean variances, `delta_method` for the linearized correction, or `posterior` for Beta variances. The assessed value is unknown when a needed variance is missing ($n\le1$, or no posterior), or when several groups or settings enter and `independent_populations=True` has not declared their covariances zero. `variance` is None, with `variance_unavailable`, when preparation records are missing, a reason concerning a setting's shots or sampling stream exists, a coefficient, fitted point or Jacobian is unavailable, or Beta inference meets a group of several labels. |
| `statistics.interval` | Linear image, through the signed coefficients, of the simultaneous per-term intervals (the corrected ones with mitigation). None for point inference. Otherwise its `status` is `conditional`, `unavailable` or `empty`, and a conditional interval has coverage at least $1-\delta$ under the sampling model and, with mitigation, the calibration channel model, or for Beta posterior credibility under the product-prior model. When labels share a group's shots, its assumptions record that a joint credibility statement lacks its model. |
| `statistics.fixed_time`, `applicability_reason`, `independence_reason` | Whether the data are one complete original measurement per setting, why missing preparation records leave the target, compiler and layout unverified, and why a shared fixed sampling stream leaves independence across settings unverified. |
| `estimates`, `provider_output_estimates` | Provider path only. The original `EstimateValue` records in the normalized units, and their values and standard errors in the units of the requested output. |
| `facts` | The evaluated sampling radius with its own failure probability, described under [Choose shots for a target accuracy](#selecting-absolute-sampling-accuracy). It is empty with mitigation, with Beta inference, when `value` is None, when a `hoeffding` or `anytime_hoeffding` interval is not conditional, and for point inference unless the shots were chosen for accuracy, the data qualify as fixed-time and no fixed sampling stream is shared across settings. |

## Choose the inference for measured counts {#selecting-measured-inference}

Set the inference and the optional readout mitigation on the method, and the shot count on `plan` or `solve`:

```python
from nwqlib import Expectation, plan
from nwqlib.algorithms import ExpectationMethod
from nwqlib.evidence.binary import (
    BinaryInferenceOptions,
    BinaryReadoutMitigation,
)

problem = Expectation(state=[2., 2.], observable=[[2., 0.], [0., 0.]])
method = ExpectationMethod(
    preparation_choice="native",  # "hzh" also supports occupation inputs
    inference=BinaryInferenceOptions(
        method="hoeffding",
        failure_probability=0.05,
        sampling_model="iid_bernoulli",
        independent_populations=True,
    ),
    mitigation=BinaryReadoutMitigation(
        calibration_shots=512,    # shots of each zero or one calibration
        minimum_contrast=0.05,
    ),
)
selected = plan(problem, method=method, shots=256)
print([experiment.name for experiment in selected.experiments])
```

```text
['science_0', 'zero_0', 'one_0']
```

`plan(problem, method=method, shots=256)` returns a [`Plan`](../glossary.md#plan), the construction and its costs computed before any circuit exists. Its `experiments` list the science and calibration experiments, and nothing is measured. Here $O=I+Z$ has one nonidentity term and one pivot, so mitigation plans one science experiment and two calibration experiments. `mitigation=None` adds no calibration work. The exact default has `shots=None`, the default point-only inference options and no mitigation. Statistical inference or mitigation without positive shots is rejected during planning.

Without mitigation, G qubit-wise commuting groups use G preparations and measurements and `G*shots` shots. With mitigation, for K nonidentity terms and P distinct parity pivots, one completed workload uses K+2P preparations and measurements and `K*shots + 2*P*calibration_shots` shots. The science and calibration experiments belong to one circuit description, and the resource estimate counts their preparation, basis-change, parity and measurement operations. `solve` checks the whole workload against `ExecutionLimits` before the first preparation. The cost of failed and repeated attempts stays counted. Scalar requests keep counts and inference records, without a saved statevector.

## Choose shots for a target accuracy {#selecting-absolute-sampling-accuracy}

For $O=c_0I+\sum_j c_jP_j$, let $L$ be the number of nonzero nonidentity labels, $G$ the number of nonempty qubit-wise commuting groups and $C=\sum_j\lvert c_j\rvert$. An absolute sampling request with tolerance $\epsilon$ and confidence $1-\delta$ selects

```math
n=\left\lceil 2\left(\frac{C}{\epsilon}\right)^2\log\frac{2L}{\delta}\right\rceil
```

shots per group. One group shot supplies every parity in that group, so the science circuits use $Gn$ shots. For a quadratic form, use $sC$ in place of $C$, where $s$ is the input's squared norm. An identity-only observable needs no measurement. Request this choice with an `Accuracy` for the sampling component in place of `shots`:

```python
from nwqlib import Accuracy, Expectation, plan
from nwqlib.algorithms import ExpectationMethod

problem = Expectation(state=[2., 2.], observable=[[2., 0.], [0., 0.]])
target = Accuracy(absolute_tolerance=0.05, confidence=0.95,
                  component="sampling")
selected = plan(problem, method=ExpectationMethod(), accuracy=target)
print(selected.shots)
```

```text
2952
```

Here $O=I+Z$ gives $C=1$, $L=1$ and $G=1$, so $\epsilon=\delta=0.05$ gives $n=\lceil800\log40\rceil=2952$.

[Proposition 1](../mathematics.md#r1) derives this choice from Hoeffding's inequality for $n$ independent outcomes in $[-1,1]$ with a fixed mean, $\Pr(\lvert z_j-\mu_j\rvert>r)\le2e^{-nr^2/2}$ (Hoeffding 1963, doi:10.1080/01621459.1963.10500830, Theorem 2), applied to each label at failure probability $\delta/L$, a union bound over the labels and the triangle inequality, which bounds the weighted error by $Cr$. Correlation between labels measured in the same shot does not affect this argument. The sampling assumption concerns successive shots, and successful execution does not establish it.

After measurement, the method evaluates the fixed-time radius of the normalized expectation from the returned group sizes $n_{g(j)}$,

```math
\sum_j\lvert c_j\rvert\sqrt{\frac{2\log(2L/\delta)}{n_{g(j)}}},
```

when every setting returned one complete set of fresh shots. Multiply it by $s$ for a quadratic form. Partial returns, data-dependent stopping and reused measurements follow the eligibility rules of the chosen inference ([fixed-time and anytime intervals](#fixed-time-and-anytime-intervals)). The explicitly selected anytime rule has its own time-uniform radius. A group's measurement counts once even though several labels use it.

`Plan.selection_accuracy` records the criterion that chose the shots, and `Result.facts` records an eligible evaluated radius with its own failure probability. Accuracy-chosen shots are supported for raw counts. Mitigation, Beta posterior inference, provider estimates and a competing explicit shot count are rejected with them.

The radius is a numerical estimate under the declared sampling assumption, and its binary64 evaluation is not a proven numerical enclosure. Choosing `iid_bernoulli` does not prove that assumption for measured data. Preparation, simulation and physical-model errors have separate evidence. So even when the radius is below the requested tolerance, `result.assess(component="sampling", absolute_tolerance=...)` can remain `INCONCLUSIVE` with `unverified=("sampling",)`. Read the conditional interval or radius and its original confidence alongside `unverified`, and do not read execution success as an unconditional accuracy PASS ([sampling error by method](../verification.md#sampling-error-by-method)).

Before each original submission, the backend connection rejects a fixed sampling stream that is known to have been used already, including by work that succeeded but was not used and by attempted work whose outcome is uncertain. PCG64 draws alone do not establish independence or stationarity.

For an identity-only observable there are no measurement settings. `plan.shots` can still hold a positive requested value, which is not an executed shot count. The Result's observations and trace show that no measurement ran.

## What the statistical fields mean

For $n_0$ zero outcomes and $n_1$ one outcomes, $n=n_0+n_1$, the empirical mean is $z=(n_0-n_1)/n$. Outcome 0 of a pivot bit, or even parity of a label's support bits in a group outcome, is the +1 eigenvalue, so under ideal readout $z$ estimates $\langle P_j\rangle$. For $n>1$, the empirical sample-mean variance is $4n_0n_1/[n^2(n-1)]$, under a model of independent identically distributed shots within a setting. A zero empirical variance after identical outcomes does not establish a known physical mean.

For a qubit-wise commuting group $g$ with the weighted score $Y_g=\sum_{j\in g}c_jX_j$, where $X_j$ is the decoded $\pm1$ parity of label $j$, the empirical variance of its mean over $n_g>1$ shots is $\bigl(\operatorname{mean}(Y_g^2)-\operatorname{mean}(Y_g)^2\bigr)/(n_g-1)$. Within a group the covariance is essential, and per-label means alone cannot recover it. For the Bell-state labels ZI and IZ with coefficients one, the weighted score has variance 4, while the sum of the individual variances is 2. The output variance adds one such contribution per group, so the variance sum assumes independent shots across groups.

The chosen inference stays visible in `result.statistics`. Each entry keeps its original counts, source and observation identifiers, empirical mean and variance, and any posterior or interval. `result.value` is the requested sum of the estimates, and `statistics.raw_value` separately keeps the uncorrected empirical sum. In the table, $\delta$ is `failure_probability` and $\alpha=\delta/F$ is its share for each of the $F$ predeclared settings, which are the $L$ labels without mitigation or the science and calibration settings with mitigation.

| `inference.method` | Returned statement |
| --- | --- |
| `point` | Empirical point and available empirical variance, no interval. |
| `hoeffding` | Conditional fixed-time interval with radius $\sqrt{2\log(2/\alpha)/n}$ for a binary mean. |
| `anytime_hoeffding` | Conditional time-uniform interval using $\alpha_n=\alpha/[n(n+1)]$, with radius $\sqrt{2\log(2n(n+1)/\alpha)/n}$. |
| `beta` | Positive-prior Beta posterior for $p=P(\text{bit}=0)$, with shapes `prior_alpha` $+\,n_0$ and `prior_beta` $+\,n_1$. Posterior mean, variance and equal-tail interval are mapped to $2p-1$. Credibility is not frequentist coverage. When the shape total exceeds 5e11 the interval is unavailable and the mean and variance remain, because SciPy's quantile endpoints were validated to 1e-3 posterior standard deviations only up to that total (`MAX_BETA_SHAPE_TOTAL` in [Engineering constants](../ENGINEERING_CONSTANTS.md#binary-inference-representation)). |

The inferences rest on these model assumptions, which no option, seed or successful execution establishes:

- The family is the `Plan`'s predeclared labels, or with mitigation its science and calibration settings. The failure probability is divided across that family before the intervals are evaluated.
- Frequentist linear (rectangle) propagation does not require independence across labels or settings, including labels that share a group's shots.
- A joint Beta family requires the independent-setting and product-prior model, and `independent_populations=True` declares independent groups, not independent parities that share shots. The joint posterior of parities that share a group's shots is unspecified. Each such label keeps its marginal Beta point and interval, and the aggregate posterior variance is unavailable with that reason. Without a compatible joint posterior, a weighted sum of marginal Beta points is still the chosen point estimate, but a joint credibility statement lacks its model.
- With `sampling_model="unknown"`, raw empirical values remain useful and the corresponding interval is unavailable. Bayesian posterior output requires its likelihood model.

### Fixed-time and anytime intervals {#fixed-time-and-anytime-intervals}

Fixed-time coverage also requires one complete original measurement for every setting, with exactly the prescribed returned shots and the preparation records of the run. Partial returns, extra or reused measurements and recorded stopping leave the fixed-time interval unavailable, and they never switch the method to the anytime option. The anytime option uses the stated union bound at every $n$ under a fixed-target, constant-conditional-mean model. It does not claim a sharper confidence-sequence rate or coverage of the winner across several `Plan`s.

### Accumulated and imported data {#accumulated-and-imported-data}

Accumulated statistical inference requires, for each setting, a chronological prefix of complete sets of fresh shots. Omitting an earlier set, using an incomplete or uncertain set, or selecting a favorable subset cannot inherit the interval or posterior. Raw empirical points remain available, and a fitted empirical correction still depends on whether its channel applies. A stop after a complete prefix is supported by the explicitly selected anytime model, and unused attempts after that prefix do not alter its kept data. Fixed-seed shots imported from another run have no validated fresh history in the current run. Reusing them keeps the raw data and the execution cost, but they cannot supply an anytime interval or Beta posterior from the new trace alone. A fresh hardware measurement of an imported circuit uses the current job's history. Existing reports keep their original history and remain readable.

Preparation records keep `CountsSampling`, which states whether the shots came from a fixed seed, a fresh measurement or an unknown source. Fixed-seed repetitions in the same context do not add sample size, even when a preparation UUID or RuntimeOptions ancestry changes, and conflicting counts from such a fixed source are rejected. Different queries that use the same fixed stream remain distinct raw data. Across settings, a shared fixed stream withholds independent covariance and product-Beta inference, while marginal Hoeffding intervals and their union bound remain valid under their own model assumptions. Within a setting, incompatible accumulation from the same stream cannot supply a fresh marginal sample. Hardware sources use the original provider job, PUB, child job and result coordinates, so one preparation can supply several fresh sets of shots. `BinaryPopulation.preparation_ids` links every source to its preparation records, including those of replays. Equal histograms alone never merge data. Unknown cross-covariance remains unknown, and `independent_populations=True` is a disclosed assumption for the covariance sum, not a deduction from different identifiers.

## Readout calibration {#binary-readout-calibration}

Each science pivot has two separate calibration experiments, which prepare computational zero and one on that pivot and measure it in the same logical layout. Analysis requires matching backend configuration, target, compiler and readout context, and the same physical mapping of that parity pivot. Matching metadata alone does not establish that the physical channel is stationary. Missing preparation records keep the raw values but cannot establish that the fitted correction or the interval applies.

The model is a stationary single-bit readout channel ([Proposition 2](../mathematics.md#r2)). It maps the true parity mean $\mu$ to the observed pivot mean $z=a\mu+b$, with one contrast $a$ and offset $b$ shared by a pivot's science and calibration settings. Preparing zero ($\mu=+1$) and one ($\mu=-1$) gives the calibration means $z_0=a+b$ and $z_1=b-a$, so $a=(z_0-z_1)/2$, $b=(z_0+z_1)/2$, and the corrected point is $\mu=(z-b)/a$. Negative nonzero contrast is legal. If $\lvert a\rvert$ is below `minimum_contrast`, the correction and therefore `value` are unavailable, and the uncorrected sum stays in `statistics.raw_value`. Finite-sample corrected points may lie outside $[-1,1]$ and remain signed in records and JSON.

The optional variance is a first-order delta-method estimate. With $d=z_0-z_1$, its derivatives in $(z,z_0,z_1)$ are $(2/d,\,-(1+\mu)/d,\,(\mu-1)/d)$. Terms that share calibration use the same data, so their coefficient-weighted derivatives add before squaring. `linear_variance` performs that covariance composition, and missing joint evidence is not filled with zeros.

The joint confidence rectangle propagates only when its denominator interval excludes zero. Its image is kept before a justified intersection with the physical mean domain $[-1,1]$. An empty intersection, an unavailable interval and a signed out-of-range point remain distinct. Beta inference with fitted readout mitigation is rejected before measurement, because this recipe does not define that posterior model. The scalar correction helper also requires matching inference options, interval kinds, probabilities and family declarations. It cannot relabel fixed-time intervals as time-uniform or mix posterior credibility with coverage.

Calibration preparation, channel stationarity, transfer to spectator qubits and fit bias remain unverified unless supplied independently. Aer can use an explicit noise model for counts. Calibration alone does not demonstrate noise reduction. Nonidentity correction is tested separately with supplied or injected affine-channel data, and no provider or hardware effectiveness claim follows from those checks.

## Provider estimates

`ExpectationMethod(estimate_precision=0.1)` explicitly asks for an estimate of the entire physical weighted Pauli sum. It excludes positive `shots`, binary inference and binary calibration. The precision has the units of the original coefficients. Identity-only problems remain exact scalar reductions without a job, even when this option is set.

`IBMRuntimeBackend` compiles the original state preparation to the device's instruction set (ISA) and applies its kept layout to the complete observable, including identity coefficients. Qiskit's default observable coercion can drop small nonzero terms. This path instead uses exact-zero sparse simplification and Qiskit's public prevalidated-array API after the common term and domain checks, which keeps all nonzero coefficients without constructing a dense operator.

The result keeps each `EstimateValue` in `result.estimates`, with the original observable and precision identifiers, the signed finite value, the requested primitive options and the returned provider metadata. For one measurement, `result.value` is that value for normalized output, and a quadratic form scales that point by $s$. `result.provider_output_estimates` gives the values and standard errors in the units of the requested output, each linked to the identifier of its raw estimate's saved record, and stores no second provider payload. The raw `result.estimates` values remain in the units of the normalized-state observable. Several supplied measurements have an arithmetic-mean point. Their uncertainties remain separate, because no covariance rule was chosen.

`standard_error` keeps a reported standard error of the mean when its meaning is established. For IBM ZNE, `stds` describes extrapolation-fit uncertainty, so that raw value remains in `provider_metadata_json` with an explicit reason why the sampling standard error is unavailable. The requested precision, the reported uncertainty and the physical bias are separate. Estimates can lie outside the physical observable's spectrum after finite sampling or mitigation. They never become bounded `PauliValue` statistics, exact trajectories or evidence of zero sampling error. IBM describes its statistical meanings in its [Estimator input/output documentation](https://quantum.cloud.ibm.com/docs/en/guides/estimator-input-output).

Estimator resilience defaults to zero unless set in `IBMRuntimeBackend.options_json`. Effective provider defaults, measurement-basis counts, extra variants and billing remain unknown unless returned with a stated scope. `ConsumptionEvent.provider_managed_sampling` reports that unknown cost, and the run's direct-shot cap is not a limit on provider spend.

## Save and reanalyze {#saved-data-reanalysis-and-numerical-scope}

`result.save(path)` writes the `Plan`, Result, observations, preparation records and kept arrays. `nwqlib.load_result(path)` restores them without replanning, preparing circuits or reanalyzing. `result.analyze(inference=...)` explicitly applies another supported inference to the same data and records a new `AnalysisOrigin`. `result.assess(...)` applies a stated criterion without measuring anything. A Result without a criterion has no accuracy verdict.

## Limits

- The quantum path needs a power-of-two dimension, and automatic dense conversion stops at dimension 16 ([Inputs](#inputs)).
- Exact simulator readout does not establish hardware accuracy.
- Statistical intervals are conditional numerical evaluations, not proven floating-point enclosures or total physical accuracy. The common `ErrorModel` keeps unresolved preparation, simulation, physical-model, sampling and reconstruction errors, plus calibration-transfer error when it applies. A useful statistical interval does not make that physical claim PASS.
- Sampling intervals hold under the declared sampling model, which execution does not establish ([Choose shots for a target accuracy](#selecting-absolute-sampling-accuracy)).
- A joint Beta posterior for labels that share a group's shots is unspecified ([What the statistical fields mean](#what-the-statistical-fields-mean)).
- Readout calibration assumes a stationary channel and gives no hardware effectiveness claim ([Readout calibration](#binary-readout-calibration)).
- Provider defaults and billing stay unknown unless the provider returns them ([Provider estimates](#provider-estimates)).

[Limitations and open work](../ROADMAP.md) lists the library-wide limits.

## Numerical scope and planning limits

`PhysicalScale` keeps the norm as mantissa and exponent, so squaring the norm need not overflow when the final value is representable. Its scale is a numerical value, not a proven norm bound. A result that is not representable remains unavailable and does not become zero.

Classical work checks the matrix-vector products, reductions and known preparation products against the limits before it computes the state of a supplied circuit or product state. `max_classical_products` defaults to 100,000,000, and `max_bytes` controls known arrays. Standard circuit gates have a formula for their state-action cost, while the work of an opaque or custom preparation remains unknown. A classical evaluation runs synchronously and is not interrupted by `Run.wait(timeout=...)`.

`max_admission_steps` (default 1,000,000) limits the planning work for each circuit description the method builds and does not change the quantum operations. When planning refuses, the message names the field, the refused stage and its count. Raise the field to the reported count, and further when the count is marked as a lower bound, because later planning stages can need more ([planning work limit](../development/program_checks.md#planning-work-limit)).

Classical planning recognizes an observable equal to $cI$ by testing its stored nonzero support and constant diagonal, or its identity-only Pauli table. Such an observable has normalized expectation $c$ for every nonzero state, so it needs no measurement. A general observable's mean diagonal does not establish this property.

A classical nonconstant observable is scaled by a binary exponent in one local copy before the matrix-vector product. The stored `scaled_observable` is $v^\dagger(O\cdot2^{-e})v$, where $e$ is `operator_exponent` and $v$ the normalized direction. Analysis combines that exponent with the norm scale before converting the requested value to binary64, so an overflowing normalized intermediate can still give a finite quadratic form. Loading a saved result checks the exponent against the original operator storage without repeating the product. CSR and CSC inputs remain sparse, and Pauli masks stay compact. If a single binary64 scaling would erase a nonzero component, the scalar is unavailable with that reason and is never silently dropped or clipped.

The binary64 count domain is at most `2**53-1` per accumulated setting, which keeps integer and $n+1$ inputs exact. Positive Beta priors, posterior shapes and tail probabilities must remain representable before the scalar SciPy kernel is called. A posterior or correction quantity that is not representable remains unavailable, and the raw empirical data are kept. Method inference uses finite scalar and record limits, while Run storage follows `ExecutionLimits.max_data_bytes`. These do not bound all CPU time, memory or opaque vendor workspace.

## Source and code map

Only the Hoeffding inequality comes from a paper. [References](../references.md#expectation-and-sampling) gives its full citation, and the page numbers below are the journal pages of that paper. The other steps are standard identities or derivations stated in the sections above. In the table, $\delta$ is the family failure probability (`failure_probability`, or one minus the chosen confidence) and $\alpha=\delta/F$ is its share for each of the $F$ predeclared settings. The implementation is in `src/nwqlib/algorithms/expectation.py`, with its capability descriptor in `src/nwqlib/algorithms/expectation_metadata.py`, count inference in `src/nwqlib/evidence/binary.py`, variance composition in `src/nwqlib/evidence/statistics.py` and the preparation and parity blocks in `src/nwqlib/blocks/selection.py`. Code paths below are relative to `src/nwqlib/`.

| Step | Source | Code |
| --- | --- | --- |
| Normalized value `c0 + sum(cj <Pj>)`, quadratic form `s` times it | Output definitions | `algorithms/expectation.py::_weighted_sum`, `algorithms/expectation.py::_output_fraction` |
| Default exact path: mean of the returned values of each label, then the weighted sum | Output definition | `algorithms/expectation.py::_exact_value`, `algorithms/expectation.py::_mean` |
| Dense observable of dimension at most `AUTO_DENSE_PAULI_DIMENSION` (16) to Pauli coefficients `c_P = tr(P O)/D` | Pauli-basis expansion | `algorithms/expectation.py::_terms`, `operators/_pauli.py::pauli_coefficients` |
| Shots `ceil(2 (C/epsilon)^2 log(2L/delta))` per QWC group, with L counting labels | Hoeffding (1963), DOI 10.1080/01621459.1963.10500830, Theorem 2, followed by a union bound over labels and the weighted triangle inequality ([Proposition 1](../mathematics.md#r1)) | `algorithms/expectation.py::ExpectationMethod.sampling_shots` |
| Fixed-time radius `sqrt(2 log(2/alpha) / n)` | Hoeffding (1963), doi:10.1080/01621459.1963.10500830, Theorem 1, Eq. (2.3), p. 15, rescaled as above, two tails as in Eq. (1.4), p. 13, solved for t at failure probability alpha. For `sampling_model="constant_conditional_mean"`, p. 18 of the same paper keeps Theorem 1 when independence is replaced by the martingale condition of Eq. (2.18), which a constant conditional mean implies | `evidence/binary.py::infer_binary` |
| Time-uniform radius with `alpha_n = alpha/[n(n+1)]` | Union bound over n, which holds because these `alpha_n` sum to alpha | `evidence/binary.py::infer_binary` |
| Beta posterior mean, variance and equal-tail interval | Conjugate Beta-Bernoulli update | `evidence/binary.py::infer_binary` |
| Sample-mean variance `4 n0 n1 / [n^2 (n-1)]` | Unbiased sample variance divided by n | `evidence/binary.py::infer_binary` |
| Family split `alpha = delta/F` and linear image of the joint rectangle | Union bound over the predeclared family | `evidence/binary.py::infer_binary`, `algorithms/expectation.py::_combined_interval` |
| Accepting measured counts: one joint histogram per group or one set of shots per setting, each fixed-seed or provider source counted once, chronological prefix and fixed-time eligibility | NWQLib design, stated under [What the statistical fields mean](#what-the-statistical-fields-mean) | `algorithms/expectation.py::_binary_data` with `_parity_counts`, `_ReceiptChecks`, `_note_source_limits`, `_count_observation`, `_accumulated_populations`, `_prefix_reasons` and `_fixed_time`. Source identifiers come from `_counts.py::CountsSources`. |
| Affine correction, its Jacobian and corner image | Standard single-bit affine readout model, derived under [Readout calibration](#binary-readout-calibration) ([Proposition 2](../mathematics.md#r2)) | `evidence/binary.py::correct_binary` |
| Output variance with shared calibration data | Variance of a linear combination | `evidence/statistics.py::linear_variance`, `algorithms/expectation.py::_binary_variance` |
| Sampling fact from the returned counts | Hoeffding, evaluated at the returned sizes | `algorithms/expectation.py::_sampling_facts` |
| One experiment per qubit-wise commuting group without mitigation, first fit in term order, with one whole-register basis block and measurement. With mitigation, one parity experiment per nonidentity term and calibration per pivot | NWQLib design | `algorithms/expectation.py::_measured_program`, `operators/_pauli.py::PauliTerms.group`, `blocks/selection.py::select_pauli_group_basis` (groups), `blocks/selection.py::select_pauli_parity` (mitigation) |
| Group score variance `(mean(Y_g^2) - mean(Y_g)^2) / (n_g - 1)` | Sample variance of the group's weighted score, divided by n_g | `algorithms/expectation.py::_group_variance` |
| Provider estimate of the whole weighted sum, averaged over measurements | Provider Estimator output, see [Provider estimates](#provider-estimates) | `algorithms/expectation.py::_analyze_estimated`, `algorithms/expectation.py::_provider_mean` |
| Classical matvec on `O 2^(-e)` with exact recovery of `2^e` | NWQLib design | `algorithms/expectation.py::ExpectationMethod._host_blocks`, `algorithms/expectation.py::_classical_value` |
| Stored Result checked against its `Plan` without new inference | NWQLib design | `algorithms/expectation.py::ExpectationAnalysis.validate_plan` with `_validate_measured_family`, `_validate_interval_meaning` and `_validate_corrections_and_value` |

With mitigation, each measured term has its own parity experiment, so its single measured bit is exactly the $\pm1$ sample that binary inference and pivot calibration use. Without mitigation, each label's marginal parity comes from its group's histogram, and the labels of one group refer to that one measurement. The derivations above assume the stated sampling model and calibration channel. They do not establish those assumptions for measured data.
