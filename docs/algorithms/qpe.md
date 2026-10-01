# QPE

<a id="quantum-phase-and-energy-estimation"></a>The QPE methods estimate an eigenvalue of a Hamiltonian $H$, or an eigenphase of a unitary $U$, from the signal $z_p=\langle\psi|U^p|\psi\rangle$ of a prepared state $\psi$, with $U=e^{-i\tau H}$ for a Hamiltonian. Real and imaginary Hadamard tests give $z_p$ (Ding and Lin, arXiv:2211.11973v2, Fig. 1 and Eqs. (4)–(5)). Each method estimates a different quantity.

- `QCELS` fits one complex exponential $z_p\approx ae^{-ip\theta}$ at integer powers $p$ and returns the energy $\theta/\tau$ (Ding and Lin, arXiv:2211.11973v2, Eqs. (2) and (9)–(12), [Result 3](../mathematics.md#r3)).
- `SPE`, statistical phase estimation, returns the lowest energy in the prepared spectral support, at the first point where a filtered estimate of the spectral cumulative distribution crosses half a declared lower bound $\eta$ on the target component's probability (Wan, Berta and Campbell, arXiv:2110.12071v2, Eqs. (2) and (6) and Algorithm 1, [Result 4](../mathematics.md#r4)).
- `RFE`, randomized Fourier estimation, draws powers uniformly, forms sampled Fourier coefficients of the signal and returns the phase of the largest one (Kshirsagar, Katabarwa and Johnson, arXiv:2209.11322v3, Eqs. (5)–(7) and Algorithm 1, [Result 4](../mathematics.md#r4)).
- `RWPE`, random-walk phase estimation, updates a Gaussian estimate of the phase from one measured bit per step and reports its mean with a Gaussian interval (Granade and Wiebe, arXiv:2208.04526v1, Eq. (7) and Algorithm 1, [Result 5](../mathematics.md#r5)).

The preparation determines which part of the spectrum is observed. A single-mode estimate establishes neither that the value is the ground energy nor that it covers the dominant component. The four methods share the controlled-power construction and the execution steps. For the lowest eigenvalue in a trial space built from one initial state, use [Chebyshev Lanczos](lanczos.md), and for supplied or adaptively grown trial states, use [GCiM](gcim.md).

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.qpe import QCELS, SPE, RFE, RWPE

problem = Eigenproblem(A=[[0.2, 0.0], [0.0, 0.7]])
result = solve(problem, method=QCELS(initial_state=[1, 0]), seed=7)
print(result.eigenvalue, result.phase)
```

```text
0.20000000000000012 0.8714285714285713
```

The prepared state $(1,0)$ is the eigenvector of eigenvalue 0.2, and `result.phase` is the same eigenvalue as an eigenphase of $U=e^{-i\tau H}$ in turns ([Read the result](#meaning-of-the-result)). This default QCELS example performs a real two-qubit Aer execution. Other choices include `SPE(initial_state=[1, 0], overlap_lower_bound=1.)`, `RFE(initial_state=[1, 0])` and `RWPE(initial_state=[1, 0])`. Replace `A` and `initial_state` to treat your own problem. The eigenvalue notebook (`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`) applies QCELS to a stretched H4 chain together with ADAPT-GCIM and Lanczos.

## Choose an estimator

| Method | Returns | What its accuracy statement assumes | Interval | Default measurement work |
| --- | --- | --- | --- | --- |
| `QCELS` | Energy of a one-frequency least-squares fit to $z_p$ | Ding–Lin's accuracy results, Theorems 1–2 (arXiv:2211.11973v2, p. 14), assume $p_0>0.71$, the squared overlap of the preparation with the target eigenvector, and their data set uses consecutive powers (Eq. (6)). NWQLib does not check $p_0$ ([When the accuracy theorems apply](#qcels-validity)). | None | 22 elementary settings, two quadratures at each of 11 powers |
| `SPE` | Lowest energy in the prepared support | A declared lower bound $\eta$ on the target component's probability, which planning does not verify. The search guarantee of Wan, Berta and Campbell (arXiv:2110.12071v2, Theorem 1 with Eq. (11)) does not transfer to NWQLib's grid scan ([SPE](#spe)). | None | 32 paired draws, 64 quadrature draws |
| `RFE` | Phase of the largest sampled Fourier coefficient | Theorem 2.1 of arXiv:2209.11322v3 (p. 6), for an eigenstate, needs more than 1100 single-shot draws for any $\epsilon<\pi/2$ and $\delta<1$. The default $M=97$ is a finite workload without that guarantee ([RFE](#rfe)). | None | 97 paired draws, 194 quadrature draws |
| `RWPE` | Mean of a Gaussian phase estimate | The basic Gaussian walk of Granade and Wiebe (arXiv:2208.04526v1, Eq. (7)), whose mean can move at most about 2.96 prior widths ([RWPE](#rwpe)). | Nominal 95-percent interval of the moment-matched Gaussian model | 14 one-shot steps |

None of these intervals is frequentist coverage or a total physical-error bound ([Intervals and sampling error](#intervals-and-verification)).

## Inputs

The natural input is an `Eigenproblem` with the preparation on the method, or a `SpectralEstimation` with its explicit Hamiltonian or unitary and its preparation. `initial_state` is required on each method for an `Eigenproblem`. A `SpectralEstimation` already holds the preparation, so a second initialization on the method is rejected. A supplied `Eigenproblem` remains the same target when the method's preparation changes. Preparing an excited eigenstate can return its energy while the problem still requests the smallest eigenvalue, and the result states that limitation.

Small dense non-power-of-two input uses a mean-trace Hermitian dummy block and a zero-padded preparation, and the original problem and preparation identifiers remain recorded. A unitary uses an identity dummy block. Scalar targets use one control qubit and no system register or preparation block.

Dense input uses `controlled_power_backend="dense_exact"` by default. A Hamiltonian power is the exact spectral power, formed from the shared eigendecomposition, so its block records algorithmic error 0. For a dense unitary input A, the base is its unitary polar factor V. Every nominal power and trajectory gap uses this same base, so the represented target at power p is $V^p$. A supplied unitary is accepted when every entry of $U^\dagger U-I$ is at most 1e-8 in magnitude. Polar selection, floating-point power formation and controlled synthesis contribute numerical error. Each controlled matrix is synthesized as its unitary polar factor to binary64 rounding ([Controlled dense unitaries](../development/dense_synthesis.md#controlled-dense-unitaries)). Under the default `max_work`, each dense block fits on up to seven system qubits ([Numerical scope and planning limits](#numerical-scope-and-planning-limits)).

## Read the result {#meaning-of-the-result}

`result.phase` is in turns, in $[0,1)$, with $Uv=e^{2\pi i\,\mathrm{phase}}v$. For a Hamiltonian, $U=e^{-i\tau H}$, so `result.eigenvalue` is in the original matrix units and includes the minus sign of the conversion. Unitary-only input has no Hamiltonian eigenvalue without a chosen logarithm branch, and accessing that field raises `AttributeError`. `result.value` follows the requested output.

SPE returns the lowest value in the prepared spectral support. For a Hamiltonian that is the lowest energy. For a unitary it is the lowest principal eigenphase angle $\theta$, with $Uv=e^{i\theta}v$ and $\theta$ scanned on $[-\pi/2,\pi/2)$. When $\tau\lvert E\rvert<\pi/2$ for every prepared energy $E$ of a unitary $U=e^{-i\tau H}$, $\theta=-\tau E$, so SPE on $U$ returns the phase of the highest energy of the prepared support. Supply $H$ itself, or $U^\dagger=e^{i\tau H}$, to target the lowest energy.

The stored `estimator_value` is the reference value of the estimator, energy for Hamiltonians and principal radians for unitary input. Phase and interval are converted from it once. Hamiltonian energy bounds are multiplied by $-\tau/(2\pi)$, their endpoints are reversed, and they are recentered around the displayed phase. The interval stays unwrapped, because wrapping each endpoint separately would change its meaning. It is a method interval, not composed total-error or ground-identification evidence.

Each exact ancilla readout must lie within $\pm1$ up to a binary64 roundoff window of the executed trajectory, or the observation is refused. The window bounds roundoff in the saved expectation, apart from the exclusions that the preparation record lists, and is not an accuracy bound for the estimate ([Readout tolerance branches](../error_evidence.md#readout-tolerance-branches)).

| Field | Meaning |
| --- | --- |
| `eigenvalue`, `phase`, `value` | The energy in the original units, the eigenphase of $U$ in turns, and the requested output, as described above. |
| `estimator_value` | Energy in operator units for a Hamiltonian, or principal eigenphase in radians for a unitary. None when no estimate was identified. |
| `interval` | RWPE's nominal model interval, None for the other estimators ([Intervals and sampling error](#intervals-and-verification)). |
| `fit` | QCELS search record: effective grid, objective evaluations, residual and interval method. |
| `gaussian` | RWPE's Gaussian phase mean and standard deviation. |
| `exposure` | One `QPEExposure` per count-mode SPE or RFE query, with its draw `multiplicity` and its `requested` and `received` shots. |
| `requested_exposure_complete` | Whether every count query returned its requested shots. |
| `minimum_effective_shots_per_draw` | The smallest `received/multiplicity` over the queries, as an exact (numerator, denominator) pair. |
| `rfe_coordinate_variance_upper` | An upper bound, in units of squared ancilla expectation, on the received-count variance proxy of each real and imaginary coordinate of the sampled Fourier coefficients. It concerns the conditional shot contribution under independent stationary shots, not the random-draw term, and gives no energy confidence interval. |
| `stop_reason`, `missing`, `complete` | Why measurement stopped or no estimate is identified (for QCELS, also that the fit is aliased), the planned queries without data, and whether all planned data and an estimate are present (for RWPE, whether every planned step was processed). |

Each SPE or RFE query contributes its mean over the shots actually returned, weighted by its original random-power draw multiplicity, and every count query needs at least one valid shot. Under outcome-independent sampling with stationary independent shots, the expected weighted contribution is the multiplicity times the query signal, and its shot variance scales as the square of the multiplicity divided by the received shots. Sampling bounds use these shot numbers and keep the uncertainty from drawing Fourier powers separate. Complete requested shots preserve the separate equal-shot draw distribution, while a short return uses a different shot-noise distribution. The finite cumulative-distribution crossing and the Fourier peak do not provide an energy confidence interval. Measurements and seeds describe the pooled shots, and draw-level batch records require separate measurements. The public two-qubit checks count the calls, verify multiplicities and check dense synthesis reuse without automatic reference projection. They do not establish mixed-spectrum accuracy or interval coverage.

## Intervals and sampling error {#intervals-and-verification}

QCELS returns `interval=None`, because the finite complex fit has no matched uncertainty model for the measurement. `result.fit` keeps its residual, effective grid and objective-evaluation count, and a flat objective has no identified estimate. SPE and RFE also return `interval=None`, because a cumulative-distribution crossing or a Fourier peak does not supply a confidence interval. RWPE reports the nominal 95-percent interval of its moment-matched Gaussian model, with its phase mean and standard deviation in `result.gaussian`. The Gaussian approximation, the prepared spectral weights and controlled-evolution error limit its interpretation. It is not a frequentist coverage or total physical-error guarantee.

The `Plan`'s error model lists every QPE error source as unknown ([How a PASS is reached](../verification.md#how-a-pass-is-reached)). With exact readout, `result.facts` states a zero `sampling` error for QCELS, whose schedule is fixed, and for an RWPE Result that reports its prior without an update. SPE and RFE average over randomly drawn frequencies or powers, and each RWPE update uses one random bit, which classical execution emulates from the exact signal. Their estimates can therefore change with the seed even with exact readout. The sample-count conditions of Wan, Berta and Campbell, arXiv:2110.12071v2, Eq. (11), and of Kshirsagar, Katabarwa and Johnson, arXiv:2209.11322v3, Theorem 2.1, control this error. NWQLib attaches no bound derived from them, so `result.assess(component="sampling", ...)` stays INCONCLUSIVE with `sampling` among the remaining sources.

## Settings and measurement work {#scientific-choices-and-actual-cost}

| Method | Estimator controls and defaults | Meaning |
| --- | --- | --- |
| `QCELS` | `num_times=16`, `max_time=None`, `grid_size=4096` | Integer powers plus zero and paired quadratures. `None` selects maximum power 10. A supplied `max_time` selects `floor(max_time/tau)` (unitary: `floor(max_time)`). Complex least squares uses a finite grid with bounded bracket refinement. |
| `SPE` | Required `overlap_lower_bound`, `num_samples=32`, `fourier_degree=11`, `filter_beta=6`, `grid_size=4096` | Sample positive odd Fourier frequencies proportional to their coefficient magnitudes, measure paired quadratures, and locate the first crossing of $\eta/2$ by the approximate cumulative distribution. |
| `RFE` | `num_samples=97`, `num_frequencies=49` | Draw M independent uniform powers from 0 through K-1, measure both quadratures for each draw, and select the largest sampled Fourier coefficient. |
| `RWPE` | `max_steps=14`, `prior_mean=0`, `prior_std=pi` | Basic Gaussian random walk with one datum per continuous-time experiment. Zero steps reports the prior without measurement. |

Each method also accepts the shared `tau`, `controlled_power_backend`, `controlled_power_error_budget` and `pauli_pruning_rtol` controls. The estimator follows the concrete method and cannot be changed by an `estimator` keyword. SPE's maximum Fourier frequency is `2*fourier_degree+1`. The common `max_power` separately caps the absolute evolution time in units of tau, including RWPE's noninteger times.

The static methods default to exact readout of one controlled trajectory. A positive `shots` sets the shots per elementary QCELS setting and per random SPE or RFE draw, so an SPE or RFE setting drawn m times receives m times `shots`, and backend limits on shots per circuit apply to that combined number. RWPE uses one shot per feedback step and defaults to 14 steps. It rejects a different requested shot count, because its update is defined for one Bernoulli datum.

The default shot workload is 22 elementary settings for QCELS and 14 one-shot measurements for RWPE. SPE and RFE draw 32 and 97 paired frequencies respectively, so their 64 and 194 quadrature draws request 64 and 194 times `shots` coherent shots. SPE and RFE make one query per distinct drawn (power, quadrature), so U distinct drawn powers give 2U queries, and at RFE's defaults the expected number of distinct settings is about 84.7. Exact readout observes every query once, on one controlled trajectory. A repeated sampled query receives the combined number of fresh shots.

Quantum RWPE plans `max_steps` circuit preparations, measurements and one-shot observations, and it checks all three totals before preparing the first feedback query. A caller cancellation can end the run earlier. The feedback angle depends on the preceding outcomes, so only the next circuit is prepared at a time.

For a trajectory built from repeated controlled applications of one base unitary, the number of applications is the largest requested power. A dense gap-power construction has its own matrix-construction and gate costs. Resource reports identify the construction and count each executed increment once. An exact QCELS, SPE or RFE [`Plan`](../glossary.md#plan), the construction and its costs computed before any circuit exists, therefore makes one Aer submission, and the operation count in its preparation record counts the executed circuit operations through the last point once.

### Classical execution {#classical-execution}

`execution="classical"` evaluates the estimator signal from an exact eigendecomposition and the prepared spectral weights. Obtaining static signals takes one spectrum and projection and $O(D)$ scalar work per unique power. Classical RWPE then emulates one Bernoulli datum from each obtained exact signal using the saved point seed. Its raw exact signal remains in the observation, and the emulated bit does not count as a quantum shot. RWPE and the dense powers built for circuits share matching existing spectra within the Run. Setup and projection counts stay separate from conservative work declarations. Supplied circuit preparations are not implicitly expanded by classical execution, and structured Pauli input has no hidden dense classical route.

### Inspect the circuits and their resources {#inspect-the-circuits-and-their-resources}

`prepare(plan).circuits` exposes the circuit of the setting that the method submits first. `prepare(plan, settings="all")` exposes the circuits of every setting of a QCELS, SPE or RFE `Plan` on the local backends that [Run on a backend](../prepared_execution.md) names. RWPE refuses it, because each feedback angle depends on the outcomes before it. Printing or estimating does not build another circuit.

Each Suzuki step carries operation and CX counts for its bound terms and represented `step_time`. Common-grid power positions refer to that shared step and record cumulative step counts. In the `selected_logical` basis a step stores 2L operations for L terms, one controlled Pauli rotation per term in the forward sweep of the symmetric formula and one in the reverse sweep. In the `cx` basis it costs $4\sum_jw_j$ CX, where $w_j$ is the number of nonidentity symbols of term $j$ (`powers.suzuki_step_cx`, [Proposition 7](../mathematics.md#r7)). A controlled rotation of support w becomes a parity chain and its inverse, 2(w-1) CX, around a controlled RZ of 2 CX. The CX count is exact for Qiskit 2.5.2 decomposition into `cx` and one-qubit gates at optimization level 0 without a coupling map, before cancellation, rotation merging or routing. The identity phase P, the feedback RZ and the Hadamard gates are one operation each and cost no CX. With a preparation that has no CX, such as an occupation state, a Hamiltonian made only of the identity term therefore has an exact zero CX total. A repeated step is multiplied by the number of applications actually executed. A common-grid exact trajectory counts its increments through the final position once, while sampled experiments count their own preparations and shots. Thus `estimate(plan)` returns the whole-workload operation count and `estimate(plan, context=ResourceContext(basis="cx"))` its CX count. The whole-workload counts stay unavailable when the preparation has no applicable formula. A dense power supplies neither formula, so its operation and CX counts are unavailable.

## Time step and energy range {#time-step-and-energy-range}

For static Hamiltonian queries, integer powers identify energy only modulo $2\pi/\tau$. Automatic selection of $\tau$ uses the existing spectral enclosure, without another eigensolve. Without a supplied `tau`, planning chooses $\tau=0.9\,r/R$, where $r$ is the principal radius of the estimator, stated below, and $R$ is the dense row-sum bound or the Pauli L1 norm. A unitary has no $\tau$. QCELS and RFE use a principal radius of $\pi$. SPE also needs a filter transition margin $\delta$, since its scan evaluates $F(x-\tau E)$ near the periodic wrap boundary. A sufficient condition is $\tau R+\delta\le\pi/2$, where $R$ bounds the spectrum, and SPE's automatic radius is $\min(\pi/3,\pi/2-\delta)$, followed by the common 0.9 interior margin. Theorem 3 of Wan, Berta and Campbell, arXiv:2110.12071v2, supplies $\delta$ from beta and the declared overlap, and its degree assumption is checked separately. An unavailable assumption or a supplied time outside that domain appears in the Result. A crossing at the scan boundary is unidentified and is not reported as an endpoint energy. Raw-unitary SPE has no established support enclosure. RWPE uses continuous times and a Gaussian prior, whose finite reach is checked against the existing enclosure.

## How the signal is measured {#how-the-signal-is-measured}

Static exact QPE prepares the control and system once and reads ancilla X and Y at the declared power positions of a controlled trajectory. These expectations give the real and imaginary parts of $z_p$, with $\langle Y_a\rangle=\operatorname{Im}z_p$. Sampled X and Y settings use separate preparations, and RWPE follows its feedback-dependent measurement schedule. SPE and RFE keep their original random-draw multiplicities. Each query records the experiment that measures it and, on the trajectory, its point, where the real and imaginary queries of one power share one measurement. A missing point leaves its queries missing, and analysis and loading never replay a prefix to replace it.

For a Pauli-sum Hamiltonian the controlled powers are product formulas. Exact static QPE trajectories use one ordered common Suzuki step on an integer time grid whose bounds meet every requested power's error budget, and the `Plan` records cumulative step counts and per-power errors. Sampled QCELS, SPE and RFE settings prepare separate circuits and select a step count for each power independently, and the reconstruction records each power's count and block, with no common-step record. Both routes reuse one count of Pauli commutators. Every power must fit its controlled-evolution allowance after adding pruning, product-formula error at the represented step time, represented-time displacement, leaf-angle formation and identity-phase discrepancy. The exact trajectory counts its step increments once, while resource estimates for sampled `Plan`s multiply each setting's own count by its requested shots.

On the exact trajectory, the nonnegative integer power p targets the exact time $t_p=p\cdot\mathrm{val}(\tau)$, where $\mathrm{val}(\tau)$ is the real value of the stored binary64 time unit. A displayed evolution time may round this product. Each power records its cumulative step count and a bound that includes pruning, the product formula at the represented step time, represented-time displacement, and applicable phase and rotation-angle formation errors. Planning checks that subtotal at every queried power. The identity term acts on the control through phase increments targeting $-c_I(p_k-p_{k-1})\,\mathrm{val}(\tau)$, because removing it would change the measured interference signal ([Proposition 7](../mathematics.md#r7)). The circuit description contains the shared controlled evolution and its X and Y observation points. Sampled Hadamard settings include their own phase and final Hadamard readout tails, and circuit construction, inspection and resource totals use those same choices.

RWPE independently selects a step count for each continuous power and checks the complete bound at the exact signed target time $\mathrm{val}(p)\,\mathrm{val}(\tau)$. Its `total_error` is the upward rounding of the exact sum of pruning, product-formula error at the represented step time, represented-time displacement, leaf-angle formation and the emitted identity-phase discrepancy. The step block's epsilon includes its own product-formula and leaf-angle terms. The complete subtotal must fit the controlled-evolution error budget, and a failed recheck refuses planning without retry. Circuit synthesis and execution roundoff remain outside this structural bound.

## QCELS

### How the fit works {#qcels-fit}

QCELS fits $z_p\approx ae^{-ip\theta}$ with $\theta=E\tau$, and the raw-unitary phase is $-\theta$. Eliminating $a=\operatorname{mean}_p(z_pe^{ip\theta})$ (Ding–Lin arXiv:2211.11973v2, Eqs. (11)–(12)) reduces the complex least-squares objective in [Ding–Lin Eq. (2)](https://arxiv.org/html/2211.11973v2#S1.E2) to one variable ([Result 3](../mathematics.md#r3)). `grid_size` is a requested minimum. The effective grid `G=max(grid_size, 8*(max(p)-min(p))+1)` covers one phase period with at least eight points per period of the fastest objective term. Every discrete local minimum receives `QCELS_BRACKET_EVALUATIONS=64` objective evaluations of golden-section search inside its adjacent periodic bracket, and all original grid candidates are kept. This finite search is not a global-optimum theorem.

### When the accuracy theorems apply {#qcels-validity}

Ding–Lin's accuracy results, Theorems 1 and 2 (arXiv:2211.11973v2, p. 14), assume $p_0>0.71$, where $p_0$ is the squared overlap of the preparation with the target eigenvector. NWQLib does not check $p_0$. Smaller overlaps need their Fourier-filtered variant (Sec. IV, p. 15), which this method does not implement. NWQLib fits one schedule rather than the multi-level schedule of their Algorithm 1, whose times double as in their Eq. (35). Their data set, Eq. (6), uses the consecutive powers $0,\ldots,N-1$. When the maximum power exceeds `num_times`, NWQLib spreads `num_times` integer powers over 1 through that maximum, a schedule their theorems do not describe.

### Choose num_times and max_time {#qcels-schedule}

With `max_time=None` the maximum power is 10. A supplied `max_time` $T$ gives the maximum power $P=\lfloor T/\tau\rfloor$, or $\lfloor T\rfloor$ for a unitary, and a `max_time` below one positive power is rejected. When `num_times` is at least $P$, the schedule holds the consecutive powers 0 through $P$. Otherwise it keeps power 0 and spreads `num_times` integer powers over 1 through $P$.

The point estimator is dimensionless, so changing energy and time units together preserves its physical meaning for the same powers. The default integer schedule is independent of units. With a supplied `max_time`, the schedule is preserved when the floored dimensionless ratios agree. The strict floor has discontinuities, and binary64 unit conversion at an integer boundary can change the final power by one. Inputs immediately below that boundary must remain below it, since no near-integer snapping or decimal reinterpretation is applied.

!!! warning "A spread schedule can fit a replica of the energy"

    A spread schedule can make the objective nearly periodic, and the Result does not report that. For maximum power $P$, its positive powers are $p_j=\lfloor1+j\bar g\rfloor$ with $\bar g=(P-1)/(N_t-1)$, where $N_t$ is `num_times`. At $\theta+2\pi/\bar g$ each term $e^{ip_j\theta}$ turns by $2\pi(1-f_j)/\bar g$, where $f_j$ is the fractional part of $1+j\bar g$, and the power-0 term does not turn. Every turn lies in $[0,2\pi/\bar g]$, about one radian for $\bar g$ near 6, so the rotated terms stay nearly aligned and the objective there keeps most of the depth of the minimum at $\theta$. With a mixed spectrum, finite shots or evolution error, such a replica minimum can win the fit and move the energy by $2\pi/(\bar g\tau)$. On the H4 problem of the eigenvalue example notebook (Hartree–Fock reference, $\tau=0.2$/Ha, `num_times=32`), `max_time=38`/Ha gives $P=190$ and $\bar g=6.10$, and the fit lands 5.156 Ha above the ground energy, next to the replica spacing $2\pi/(\bar g\tau)=5.153$ Ha, while 36 and 40/Ha stay within 7 mHa ([Measured evidence](#measured-evidence)). Consecutive powers 0 through $P$, which the schedule uses when `num_times` $\ge P$, have no such near period.

### Aliasing report {#qcels-aliasing}

If the powers with a nonzero summed signal share a common spacing $g>1$, the objective has period $2\pi/g$ and the Result reports the aliasing. The planned schedule always contains powers 0 and 1, so this happens only when the signal vanishes at some powers.

### When a signal counts as zero {#qcels-zero-signal}

A power's summed signal counts as zero when it lies within the error that its accepted exact means and the summation allow, `sqrt(2)*mean_window*W` plus the rounding of the sum for $W$ summed samples. `mean_window` is the largest `probability_window` of the preparation records for exact probabilities and `NUMERICAL_RELATION_RTOL` for classical execution, and count data use an exact-zero test ([Engineering constants](../ENGINEERING_CONSTANTS.md)). Exact data whose signal vanishes in exact arithmetic therefore report aliasing or a flat objective despite their roundoff, and RFE applies the same test before it calls a sampled Fourier spectrum flat.

### Reanalyze the fit {#qcels-reanalysis}

```python
import numpy as np
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.qpe import QCELS

result = solve(
    Eigenproblem(A=np.diag([.4, -.7])),
    method=QCELS(initial_state=[0, 1], tau=.2, num_times=16,
                 max_time=100),
    execution="classical", seed=7,
)
print(result.eigenvalue, result.fit.effective_grid, result.fit.evaluations)
refined = result.analyze(grid_size=4096)  # same observations
print(refined.eigenvalue)
```

```text
-0.7000000000000006 4096 31168
-0.7000000000000006
```

The prepared state $(0,1)$ is the eigenvector of eigenvalue $-0.7$. Post-hoc `result.analyze(grid_size=...)` changes the QCELS or SPE analysis grid using the existing observations, and no reanalysis measures new samples. Increasing `grid_size` is checked against the original method's limits. If the intended later analysis exceeds them, construct a new `Plan` with the needed limits before measurement. `result.analyze(grid_size=4096)` reuses the current observations within this example's `max_work`. RFE's K defines both the power distribution and the Fourier frequencies, so changing it requires a new `Plan`, as do sample counts, the preparation and every RWPE setting. `seed` records separate PCG64 streams for the method and the backend, and reanalysis does not draw from them.

## SPE

SPE uses the Fourier filter of [Wan, Berta and Campbell, arXiv:2110.12071v2](https://arxiv.org/html/2110.12071v2), PDF Eqs. (A1)–(A3), corresponding to HTML Eqs. (16)–(18). Write $F_j=-ia_j$ for positive odd frequencies and $S=\sum_ja_j$. Drawing $j$ with probability $a_j/S$ and measuring the complex overlap $z_j$ gives $1/2+(2S/M)\sum\operatorname{Im}(e^{ijx}z_j)$. Negative frequencies follow by conjugation. Grouping repeated draws multiplies each exact or pooled contribution by its draw count. The terminal coefficient contains $I_d$ alone, as specified by PDF Eq. (A2), HTML Eq. (17). Algorithm 1 instead samples $j$ over all of $\{0,\pm1,\pm3,\ldots\}$ in proportion to $\lvert F_j\rvert$. NWQLib keeps $F_0=1/2$ exactly and pairs each positive draw with its conjugate, which also gives an unbiased estimate of the approximate cumulative distribution.

The declared overlap lower bound $\eta$ is an assumption supplied by the caller. Planning does not diagonalize an operator or expand a preparation to verify it. For the filter domain, Theorem 3 of Wan, Berta and Campbell, arXiv:2110.12071v2, uses $\epsilon_1=\epsilon_2=\eta/8$ and $\epsilon_3=\eta/2$ in PDF Eqs. (A6)–(A7) and (A12), HTML Eqs. (21)–(22) and (27). This gives a conditional filter error of at most $3\eta/8$ away from a transition half-width $\delta$, leaving $\eta/8$ below the $\eta/2$ decision threshold for sampling error. The Result reports $\delta$ separately from the grid spacing. These analytic bounds are evaluated in binary64 and do not include coefficient roundoff or controlled-evolution error. Degree and beta are tested against Theorem 3, and a failed assumption appears in the Result.

NWQLib scans a fixed grid using one shared sample set, which permits reanalysis without measurement. The paper also reuses one sample set, but it makes only $s=O(\log(1/\delta))$ decisions chosen as in binary search (Sec. III, p. 4) and gives each decision failure probability $\xi/s$, so a union bound limits the total failure probability of Theorem 1 to $\xi$. A scan over $G$ grid points would need failure probability $\xi/G$ per decision in Eq. (11). The default sample count is not derived from Eq. (11), so the search guarantee does not transfer. The paper's randomized LCU compiler and gate-complexity claim are also outside this controlled-evolution implementation.

## RFE

RFE uses the randomized Fourier estimator of [Kshirsagar, Katabarwa and Johnson, Sec. 2, Eqs. (5)–(7)](https://arxiv.org/html/2209.11322v3#S2). A draw $k$ contributes $z_ke^{-2\pi ikj/K}/M$ to frequency $j$. Grouping the identical draws of one power into one exact or pooled observation replaces their terms by the draw multiplicity times that observation's term. Uniform draws make the result an unbiased estimate of the finite Fourier coefficient. The measured quadrature convention fixes $z_k=e^{ik\,\mathrm{phase}}$, so postprocessing returns $2\pi j/K$ and the Hamiltonian energy $-\mathrm{phase}/\tau$. The arXiv:2209.11322v3 note on p. 1 records that Eq. (4) and the output line of Algorithm 1 were corrected for a missing sign. With the paper's S gate before the final Hadamard, outcome 0 has probability $(1-\sin(k\theta))/2$, so the raw imaginary outcome has mean $-\sin(k\theta)$ and the printed output is $2\pi-2\pi j/K$. NWQLib's imaginary quadrature uses ancilla phase $-\pi/2$ and measures $+\operatorname{Im}z_k$ directly, so its output needs no reflection. Work is $O(UK)$ for the U distinct drawn powers, with $O(K)$ workspace. These sizes depend on the estimator budget, not the Hilbert-space dimension.

The default K gives phase spacing $2\pi/49$ and maximum power 48. M sets the random-draw count. Its 194 quadrature draws become one query per distinct (power, quadrature) in both modes, and shot mode requests 194 times `shots` coherent shots. The paper's precision theorem is Theorem 2.1 of arXiv:2209.11322v3, p. 6. For an eigenstate it gives an $\epsilon$-accurate estimate with probability above $1-\delta$ when $K\ge\lceil2\pi/\epsilon\rceil$ and $M\ge\lceil(81\pi^2/2)\ln(8\pi/(\delta\epsilon))\rceil$ single-shot draws. For any $\epsilon<\pi/2$ and $\delta<1$ that bound exceeds 1100 draws, so the default $M=97$ is a finite workload without the theorem's guarantee. A successful finite Fourier fit establishes neither the eigenstate assumption nor the sample budget and supplies no RFE uncertainty interval. A dominant peak in a mixed spectrum need not represent its ground energy.

## RWPE

RWPE implements the basic Gaussian update of [Granade and Wiebe, arXiv:2208.04526v1, Eq. (7) and Algorithm 1](https://arxiv.org/pdf/2208.04526v1). Its coordinate is the unwrapped phase $\phi=-\tau E$. At step $k$, $\sigma_k=\sigma_0((e-1)/e)^{k/2}$, where $\sigma_0$ is `prior_std`, the relative evolution time is $1/\sigma_k$, and one bit $d$ changes the mean by $(1-2d)\sigma_k/\sqrt e$. The physical evolution is $e^{-i\tau H/\sigma_k}$, so a bare unitary with integer-power access is insufficient. All times are known from the prior and the step count, while the feedback depends on previous bits. The update uses constant scalar workspace at every Hilbert-space dimension.

The Hadamard circuit uses the feedback $-\mathrm{mean}/\sigma-\pi/2$, giving $P(0)=(1+\sin((\phi-\mathrm{mean})/\sigma))/2$. Integrating this likelihood against the Gaussian prior gives the positive $d=0$ mean shift in Eq. (7a) of arXiv:2208.04526v1, because $\mathbb E[x\sin x]=e^{-1/2}$ for a standard normal $x$. In the paper's likelihood, Eq. (2), $P(d|\omega)=\cos^2(t(\omega-\omega_{\mathrm{inv}})/2+d\pi/2)$, this feedback is $\omega_{\mathrm{inv}}=\mathrm{mean}+\pi\sigma/2$. Algorithm 1 prints $\omega_{\mathrm{inv}}=\mathrm{mean}-\pi\sigma/2$ alongside the same update, which gives $P(0)=(1-\sin((\omega-\mathrm{mean})/\sigma))/2$ and would lower the mean after $d=0$. At $t=1/\sigma$ the paper's own exact update, Eq. (6a), p. 4, reproduces Eq. (7a) with $\omega_{\mathrm{inv}}=\mathrm{mean}+\pi\sigma/2$ and gives the opposite sign with $\omega_{\mathrm{inv}}=\mathrm{mean}-\pi\sigma/2$. The sentence that introduces Eq. (7) on p. 5 states $\omega_{\mathrm{inv}}=\mathrm{mean}$, for which Eq. (6a) leaves the mean unchanged. An independent Gaussian-integral check fixes the sign from the circuit probability.

This implementation omits Algorithm 2's consistency checks and unwinding. As discussed after Eq. (8), the basic walk has a finite exploration range. The mean can move at most $\sigma_0/(\sqrt e-\sqrt{e-1})$, about 2.96 prior widths (the paper prints 2.95), so a shrinking Gaussian width alone does not establish convergence. The Result reports when the scaled spectral radius and the prior mean reach that range.

RWPE's planned times do not specify its future feedback points. Profile estimates and comparisons leave those points unpredicted, and a shot-count scan keeps the unresolved number unavailable. Forecasting does not choose feedback or run the update.

## Verify a result {#verify-a-result}

`result.verify(checks=QPEVerification(tolerance=...))` explicitly computes a nominal comparison of the spectrum and its eigenvalue clusters with a QR projector. SPE compares the lowest cluster and uses its declared overlap bound unless `minimum_overlap` is supplied. The other estimators compare the largest prepared cluster, with default minimum overlap 0.9. The comparison reports discrepancy, overlap, numerical adjustment, resolution and the reference calls made. It returns `(receipt, facts)`, a verification record and the two checked facts, `component_error` against `tolerance` in the output unit and `overlap_deficit = max(0, minimum_overlap - weight)` against zero, for `Certificate.with_verification` ([Check accuracy and verify a result](../verification.md#evidence-and-receipts)). `group_atol` clusters in the unit of the compared eigenvalues, the Problem's energy unit for a Hamiltonian and radians of eigenphase angle for a unitary, whatever the output. `reuse_only=True` requires existing spectral and reference data whose target and preparation identifiers match, including after save and load. The saved arrays themselves are not rechecked ([Saved folders are read-only](../saved_evidence.md#saved-folders-are-read-only)). A reference for a supplied circuit is computed only with `materialize_preparation=True`. These are explicit, bounded dense comparisons, with no automatic Pauli expansion.

For the static estimators, `method_resolution` is the Result's grid spacing in output units. It includes a later supported `analyze(grid_size=...)` call and is not a fit-error bound. RWPE's method resolution is its current Gaussian standard deviation in output units. Neither quantity is a proven total-error bound, and verification leaves the original Result and `Plan` unchanged.

## Continue and save {#continuation-and-saved-evidence}

`prepare` binds only RWPE's next query. Before new work, RWPE saves its Gaussian mean, processed-step count, evolution time, feedback, backend seed and the link to the pending checkpoint. `submit` and `Run.wait` advance the same RWPE state through the shared backend. Completed pending measurements update the Gaussian once. Unresolved submitted work is not resubmitted, and cancellation leaves its recorded cost visible.

The boundary before submission rejects known repeated fixed-seed streams from completed, uncollected, uncertain and same-batch measurements. Returned provider coordinates remain checked after completion. Reading the same completed measurement again during recovery is legal, but it does not authorize another posterior update.

`result.save(path)` and `load_result(path)` keep the concrete method, the evolutions it chose (including the polar base of a unitary input), the observations, the Gaussian moments and the existing numerical caches. Loading rejects a conflicting method before it loads state data. [Saved formats](../development/execution.md#saved-formats) lists the archive format names and versions. `run.save(path)` and `load_run(path, backend=...)` keep the Run's progress, the random-number state and the circuit caches. Loading performs no schedule search, eigensolve, estimator replay or circuit construction, and reanalysis shares unchanged data.

`result.analyze(...)` measures nothing new. For RWPE it reads the Gaussian mean and the number of processed steps from the saved run state, and its width follows Granade and Wiebe, arXiv:2208.04526v1, Eq. (7b), without replaying observations. Without saved run state, the RWPE estimate is reported as missing. Its stop reason is the one the run recorded when it ended.

## Measured evidence

| Measurement and section | Settings and environment |
| --- | --- |
| QCELS replica on H4 ([Choose num_times and max_time](#qcels-schedule)) | The H4 chain of the eigenvalue notebook: four hydrogen atoms 2.0 Å apart, STO-3G, all eight spin orbitals active, its dense 256-by-256 Hamiltonian and the Hartree–Fock basis state. `QCELS(tau=0.2, num_times=32, max_time=36, 38 or 40)` with the default `grid_size`, `execution="classical"` and seed 7. The errors are relative to the CASCI (full CI) energy, −1.8977806 Ha: −6.4 mHa at 36/Ha, 5.156 Ha at 38/Ha and −0.073 mHa at 40/Ha. Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1, PySCF 2.14.0 and Qiskit 2.5.2 on macOS arm64. |

## Limits

- A single-mode estimate establishes neither the ground energy nor coverage of the dominant component, and the preparation determines which part of the spectrum is observed.
- QCELS, SPE and RFE return no interval, and RWPE's interval is a nominal Gaussian model interval ([Intervals and sampling error](#intervals-and-verification)).
- The papers' accuracy theorems need conditions that NWQLib does not check or does not meet by default ([Choose an estimator](#choose-an-estimator)). QCELS's theorems assume $p_0>0.71$, which is not checked, and consecutive powers. SPE's search guarantee assumes the paper's binary-search decisions, not a grid scan. RFE's theorem needs more than 1100 draws, against a default of 97.
- A spread QCELS schedule can fit a replica of the energy, and the Result does not report it ([Choose num_times and max_time](#qcels-schedule)).
- The basic RWPE walk has a finite exploration range and omits Algorithm 2's consistency checks and unwinding ([RWPE](#rwpe)).
- Dense powers are explicit finite constructions, not scalable oracles. Under the default `max_work`, each dense block fits on up to seven system qubits.
- The default sampling assessment stays INCONCLUSIVE, because NWQLib attaches no bound from the papers' sample-count conditions.

[Limitations and open work](../ROADMAP.md#qpe) lists the open items for these methods.

## Numerical scope and planning limits

`max_power=100000` and `max_trotter_steps=1000000` cap finite evolution work. `max_work=1000000000` bounds known numerical, schedule and iteration-state size units, and `max_bytes=10_000_000_000` bounds known arrays. They do not bound LAPACK or SDK CPU time or memory. RWPE's update counts 2 units when it starts and 32 for each choice and each update, and its last update follows the last measurement, so planning checks this whole count, `2+64*max_steps` units, against `max_work` before the first query. On its product-formula route, RWPE also checks the independent exact recheck of all `max(1, max_steps)` relative times during planning. That recheck, the commutator count and the RWPE update each have their own comparison against `max_work`, so the update total alone does not fund every planning stage. On resume, an interrupted choice or update is counted again. The shared `ExecutionLimits` cover preparation, measurement and data. Symbolic evolutions and repeated steps remain compact until the Qiskit circuits are built explicitly.

The QCELS analysis limit check counts at most `(1 + QCELS_BRACKET_EVALUATIONS)*G = 65*G` objective evaluations, the $G$ grid points plus one bracket for each possible local minimum. Each evaluation counts 32 scalar size units per complex point, so the QCELS formula is $32\cdot65\cdot GN$ units plus 64 per sample for the reduction. Here $N$ counts distinct complex points after quadrature reduction. These are conservative operation-size units, not measured CPU instructions. Grid and scores use $O(G)$ storage and row temporaries $O(N)$, without a $G\times N$ matrix. Limits are checked before the schedule is built or measured and again before reanalysis. The default has $N=11$, $G=4096$ and 22 quadrature queries. A limit of 100 million would reject this schedule before measurement, and the default `max_work=1000000000` accepts it.

Limit-check errors report the prospective work and bytes and the supplied `max_work` and `max_bytes`. The shared `NUMERICAL_RELATION_RTOL` is the internal `nwqlib._validation` input convention, not a tunable method accuracy option.

### Commutator count and common step {#commutator-count-and-common-step}

Before counting commutators, exact-trajectory product-formula planning sets aside an upper bound on the common-step work and logical bytes from the powers, term count and input exponents. The commutator count uses the full Pauli-triangle expression when its work plus that bound fits `max_work` and each stage's bytes fit `max_bytes`. Otherwise it uses the second-order suffix relaxation, which needs quadratic pair work and can select more steps. The recorded choice names the expression and the outward coefficient arithmetic. For $p$ terms on $q$ qubits with $q\le64$, define $P=p(p-1)/2$ and $J=p(p-1)(2p-1)/6$. Counting each packed test and each magnitude-contribution visit once gives uniform upper bounds $2pq+2(P+J)$ for the full path and $2pq+3P+p$ for the relaxed path. With `max_work=100000000` and 20 system qubits, the commutator count alone allows initial full and relaxed bounds of at most 531 and 8151 terms. Sampled settings and RWPE set aside no common-step bound. Their independent exact recheck is checked against `max_work` as its own stage before its exact arithmetic runs, with the work formula in [Engineering constants](../ENGINEERING_CONSTANTS.md).

On the product-formula route, default sampled schedules select 11 distinct powers for QCELS, at most 12 for SPE, and at most 49 for RFE. For $K$ powers, including zero, and $L$ kept nonidentity terms, the independent recheck accepts `32L(K+1)C_T + (320K+128)C_S + 128K_a C_I <= max_work`, where $K_a$ counts nonzero powers when $L$ is positive and $C_T$, $C_S$ and $C_I$ are the term, scalar and candidate-inversion width costs defined in [Engineering constants](../ENGINEERING_CONSTANTS.md). With costs bounded by $(1,27,100)$, a `max_work` of 100000000, a tenth of the default, accepts this recheck throughout the default commutator-count term range, including schedules through 64 powers. At costs exactly $(8,27,100)$ and that limit, an RFE schedule containing all 49 allowed powers accepts at most 7,731 terms at this stage. Wider arithmetic can reduce that limit further. The coefficients, times and pruning mass determine the widths, so $K$ and $L$ alone do not guarantee acceptance. The commutator count and the other planning requirements remain separate checks.

An exact-trajectory `Plan` must also fit the common-step bound determined by its power schedule, coefficient exponents, pruning loss, identity coefficient, time scale, per-power error budgets and step cap, so these commutator-count figures are not its term limits. Above the full-path threshold, an accepted relaxed initial bound can still lead to a full count after the pair structure is known. Other planning work, storage and step limits also apply.

The exact trajectory's common-step selection checks exact arithmetic against the limits before constructing candidate counts. Its work count uses the input exponent ranges and includes candidates later refused by `max_trotter_steps`. It computes the kept coefficient mass and the one-step rotation-angle discrepancy once, then adds each emitted identity phase once to the exact running prefix. The arithmetic reductions take $O(L+K)$ visits for $L$ kept terms and $K$ requested powers. The limit check also includes sorting, a conservative $K^2$ count for integer-key collisions, and an integer-width factor. Both commutator-count checks include the same common-step bound in `max_work`, whose default is `1_000_000_000`, so an accepted commutator-count variant also fits the later exact-arithmetic work limit. The full expression is preferred when the combined count fits, and otherwise planning uses the accepted relaxation. Each prefix is accepted only when its complete exact subtotal fits its error budget. `max_bytes` separately checks the logical live-object bound of the exact stage.

### Dense powers {#dense-powers}

Planning forms the polar factor V once per `Plan` from one SVD, counted as $9D^3+8D^2$ units of planning work, and the classical route and verification diagonalize V, not A. The exact trajectory constructs one block per distinct positive gap between consecutive powers, and a sampled `Plan` one block per distinct nonzero power. A gap of k powers is constructed by repeated squaring of V and built as a controlled dense block. Each gap's limit check covers its matrix work, synthesis and live cache storage. Planning counts each block's matrix arithmetic in $D^3$ units per product of two $D$-square matrices, where $D$ is the dimension of the system register. A unitary gap or power k counts the $\mathrm{bit\_length}(k)+\mathrm{popcount}(k)-2$ products of an independent repeated squaring, although the squares of V are shared within the `Plan` and a later block forms only the squares it lacks. Its bytes include every cached square, $16JD^2$ for the largest square index $J$ of the `Plan`'s exponents. A Hamiltonian power takes one product and also counts the eigendecomposition, $8D^3+32D^2$ units, because the first power that is constructed computes it and the others reuse it. Planning adds $11\cdot8^m+(m^2+5m+256)\cdot4^m$ units for the synthesis on $m=n+1$ qubits with $n$ system qubits. Each block is checked against `max_work` on its own, so the default `max_work` accepts dense blocks on up to seven system qubits. The `Plan`'s total over its blocks is not compared with `max_work`.

<a id="implementation-map"></a>

## Source and code map

The code is in `src/nwqlib/algorithms/qpe/`:

- `method.py`: method settings, original inputs, the circuit description, readout and static reconstruction.
- `records.py`: the meaning of queries, powers and results, and exact validation of the stored evolution.
- `powers.py`: the signed-power numerical and circuit constructors.
- `numerical.py`: the QCELS, SPE, RFE and RWPE scalar kernels and their source conventions.
- `controller.py`: RWPE's Gaussian mean and pending point over the common Run.
- `archive.py`, `verification.py`: storage of the data and caches, and explicit nominal references.
- `subroutines/trotterization/error_budget.py`: commutator bounds and exact integer selection for a supplied evolution time. Common-grid QPE uses those bounds for sufficient shared-step prefix counts and for the check of represented steps against them.
- `subroutines/qpe/coherent.py`: the separate coherent phase-register circuit.

The QPE methods build their controlled product-formula steps in `powers.py` from Qiskit Pauli evolution gates. They use only the bounds of the Trotterization subroutine, not its circuit builders.

Each row below names the scientific step, paper version, equation and code in `algorithms/qpe` unless a subroutine path is given. Here `z_p` is the Hadamard-test signal at evolution multiplier `p`, and `sigma` is RWPE's Gaussian phase width. The docstrings of the listed code derive implementation-specific formulas and state their assumptions.

| Step | Source | Location | Code |
| --- | --- | --- | --- |
| Real and imaginary Hadamard tests (W = I or S†) give `z_p`. The exact trajectory reads the same quadratures as ancilla X and Y at each power position | Ding–Lin, arXiv:2211.11973v2 | Fig. 1, p. 3. Eqs. (4)–(5), p. 8 | `method._quantum_program`, `method._trajectory_samples`, `method._static_estimate` |
| QCELS objective and eliminated amplitude | Ding–Lin, arXiv:2211.11973v2 | Eq. (2), p. 4. Eqs. (9)–(12), p. 9 | `numerical.qcels` |
| QCELS search | Ding–Lin, arXiv:2211.11973v2, with NWQLib grid | Sec. II.2, p. 10, suggests up to ceil(T_max) grid points with gradient ascent. NWQLib uses eight cells per unit of power span and golden-section brackets of `QCELS_BRACKET_EVALUATIONS` evaluations | `numerical.qcels`, `numerical.qcels_grid_size` |
| QCELS schedule and validity | Ding–Lin, arXiv:2211.11973v2 | Eq. (6), p. 8. Algorithm 1, p. 13, and its doubling times, Eq. (35), p. 14, not implemented. Theorems 1–2, p. 14, need `p0 > 0.71`. Sec. IV, p. 15, not implemented | `numerical.planned_power_schedule`, `method.QCELS` |
| SPE random schedule and filter | Wan–Berta–Campbell, arXiv:2110.12071v2 | PDF Eqs. (A1)–(A2), p. 7, HTML Eqs. (16)–(17), Algorithm 1. Positive odd frequencies sampled by coefficient magnitude, with paired quadratures and conjugate negative-frequency contributions | `numerical.spe_fourier_weights`, `numerical.planned_power_schedule` |
| SPE estimator | Wan–Berta–Campbell, arXiv:2110.12071v2 | Eq. (2), p. 2. Eq. (6), p. 3. Algorithm 1, p. 4. First CDF crossing of eta/2 over a finite grid of decisions, in place of the O(log(1/delta)) binary-search decisions of Sec. III, p. 4. Theorem 3, p. 10, gives the filter domain. Controlled-evolution synthesis error is accounted for separately | `numerical.spe_cdf`, `numerical.spe`, `numerical.spe_filter_domain` |
| RFE signal | Kshirsagar–Katabarwa–Johnson, arXiv:2209.11322v3 | Eqs. (1)–(6), p. 4 | `numerical.planned_power_schedule`, `method._static_estimate` |
| RFE estimator | Kshirsagar–Katabarwa–Johnson, arXiv:2209.11322v3 | Algorithm 1 and Eq. (7), p. 5. Uniform powers in [0,K), sampled Fourier coefficients, largest magnitude. Sign note, p. 1. Sample bound of Theorem 2.1, p. 6 | `numerical.rfe` |
| RWPE likelihood | Granade–Wiebe, arXiv:2208.04526v1 | Eq. (2), p. 2 | `numerical.rwpe_choose`, `numerical.rwpe_update` |
| RWPE experiment choice | Granade–Wiebe, arXiv:2208.04526v1 | Algorithm 1, p. 3. Time 1/sigma and one Bernoulli datum. Feedback sign follows Eq. (2) and the positive d=0 shift in Eq. (7a) | `numerical.rwpe_choose`, `controller.run_rwpe` |
| RWPE posterior and estimate | Granade–Wiebe, arXiv:2208.04526v1 | Eqs. (5)–(7), pp. 4–5. Gaussian moment updates and mean estimate. Algorithm 2 unwinding is absent | `numerical.rwpe_scale`, `numerical.rwpe_update`, `numerical.rwpe_estimate` |
| Second-order Suzuki step and term order | Childs et al., Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020 | Prop. 10, p. 011020-22 (arXiv:1912.08854v3 Prop. 16, p. 39) | `powers.rotation_schedule` |
| Commutator bounds | Childs et al., doi:10.1103/PhysRevX.11.011020 (Phys. Rev. X version) | Prop. 9, Eq. (120). Prop. 10, Eq. (121). p. 011020-22 (arXiv:1912.08854v3 Sec. 5.1, Prop. 15, Eq. (145), p. 38, and Prop. 16, Eq. (152), p. 39) | `subroutines/trotterization/error_budget.py` |
| Step-count selection | Childs et al., doi:10.1103/PhysRevX.11.011020, with NWQLib's common-grid specialization | Sec. V B for the independent minimum-count rule. The common-grid derivation gives sufficient prefix counts, each checked against the represented-step bounds | `powers.select_powers`, `subroutines/trotterization/error_budget.py` |
| CX of a controlled Suzuki step | NWQLib, from Qiskit 2.5.2 `qiskit/circuit/library/pauli_evolution.py::PauliEvolutionGate.control` and `crates/synthesis/src/pauli_evolution.rs` ([Proposition 7](../mathematics.md#r7)) | Basis change, parity chain of w-1 CX and its inverse around a controlled RZ of 2 CX, derivation in the docstring | `powers.suzuki_step_cx`, `powers._make` |
| Pruning budget, controlled identity phase, automatic tau, non-power-of-two padding | NWQLib | Docstrings of the listed code | `method._split_pauli_terms`, `powers.select_powers`, `method._quantum_program`, `method._safe_time`, `method._QPEMethod.plan` (Pauli input), `method._QPEMethod._inputs` |
| Classical nominal signal | NWQLib | One dense eigendecomposition and the prepared spectral weights give the noiseless ancilla means | `method._bind_host`, `method._host_construction`, `numerical.overlap` |
| Coherent phase register and inverse QFT | Cleve–Ekert–Macchiavello–Mosca, arXiv:quant-ph/9708016v1 | Sec. 5. Fig. 6 and Eq. (5.1), p. 10. QFT of Eq. (4.1), p. 8. Readout and its 4/pi**2 bound, Eqs. (5.2)–(5.4), p. 11. The builder docstring maps the paper's bit order to Qiskit's | `subroutines/qpe/coherent.py` |

Independent checks cover two-level signed quadratures, the controlled global phase, mutations of readout wires and arguments, physical-unit conversion, finite likelihoods, non-power-of-two coordinates, rejection of a reused sampling stream and recovery of an interrupted RWPE run. Circuit-level checks of the QPE methods use at most four qubits. The separate coherent QPE subroutine check uses seven, six phase qubits and one system qubit.
