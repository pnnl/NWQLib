# Mathematics {#mathematical-foundations-and-derived-bounds-in-nwqlib}

## 1. Introduction

NWQLib computes expectations, projected eigenvalues, phase and energy estimates, linear-system solutions, linear differential-equation solutions, and finite-grid quantum Hamiltonian descent trajectories. Its quantum constructions and classical models share explicit input operators, normalization conventions, reconstruction maps, and resource descriptions. This makes the distinction between a circuit's output and a physical answer part of the mathematical specification. An inverse polynomial produces an unnormalized success branch. A subspace eigensolver produces a Ritz value. A position distribution produces an optimization candidate. Each requires additional relations before its error can be interpreted in the user's requested units.

This distinction also determines the meaning of a resource estimate. A bound on the number of calls to a block encoding does not determine its gate count, and a gate count does not determine a physical runtime. Similarly, a sampling interval does not cover polynomial approximation, circuit bias, loss of a trial direction, or failure to prepare a target eigenstate. NWQLib carries these obligations separately and combines them when the required premises are available.

The results below describe the mathematical constructions that NWQLib implements. They cover all seven algorithm families, their shared subroutines, circuit synthesis, resource composition, and verification. The central contributions are finite and composable statements: polynomial residuals in physical units, quadrature bounds with explicit domains, derivatives through a truncated overlap subspace, error budgets for the actual emitted evolution, and resource laws evaluated without expanding the computation.

Each numbered result has one provenance label. **From a reference** identifies a published construction or theorem and its precise source. **Derived in NWQLib** identifies an implementation-linked derivation whose contribution can lie in a new combination of established arguments. **Improved from a reference** identifies a correction, additional case, sharper bound, or computationally usable form of a named source result. These labels describe provenance relative to the cited sources. They do not assert worldwide priority. For each result, the Contribution column of the implementation and evidence index in Section 13 states whether it is standard material and what it adds to its sources.

Throughout, vector norms are Euclidean and matrix norms are spectral unless a subscript states otherwise. The unit roundoff of binary64 arithmetic is $u=2^{-53}$, and $\gamma_n=nu/(1-nu)$ when $nu<1$. An analytic inequality and its floating-point evaluation are different objects. An ordinary binary64 evaluation of an upper-bound formula is not an interval certificate. Conditional roundoff results state their arithmetic or backend assumptions. The implementation and evidence index in Section 13 identifies the code and repository checks for every result. The analytic results are established by the proofs below or by the cited source theorems.

### Find a result

Result and proposition numbers are stable identifiers, cited in the documentation and in the source code, with anchors `#r1` to `#r58`. They are not in reading order. Proposition 58, for example, sits in Section 4 with the other Lanczos results.

| Result | What it gives |
| --- | --- |
| **Expectation estimation** ([Section 2](#2-expectation-estimation-and-binary-inference)) | |
| [Proposition 1. Simultaneous sampling bounds for a weighted expectation](#r1) | A confidence radius for every setting mean at once, with probability at least $1-\delta$, the resulting bound on the weighted expectation, an equal shot count for an absolute tolerance $\epsilon$, and a time-uniform version |
| [Proposition 2. Affine readout correction with shared calibration](#r2) | For a stationary affine readout channel, the corrected mean and its gradient, the vertex rule for a confidence rectangle, and the variance when terms share calibration |
| **Phase and energy estimation** ([Section 3](#3-phase-and-energy-estimation)) | |
| [Result 3. The QCELS scalar objective](#r3) | The QCELS least-squares objective with its amplitude eliminated. The paper's accuracy theorems, which include target-overlap assumptions, do not automatically describe NWQLib's finite grid and bracket search |
| [Result 4. Fourier estimators and their target populations](#r4) | The SPE grid crossing and the RFE Fourier peak as implemented. The SPE grid has more decision points than the paper's binary search, so the paper's failure allocation must change before its coverage theorem applies. An RFE peak of a mixed spectral measure need not identify the lowest energy |
| [Result 5. Gaussian random-walk phase updates](#r5) | The Gaussian moment update of random-walk phase estimation (RWPE). The reported width alone is not a proved confidence interval for the whole procedure |
| [Result 6. Coherent phase-register readout](#r6) | For an eigenstate and ideal controlled powers, the nearest $m$-bit phase outcome has probability at least $4/\pi^2$ |
| [Proposition 7. Controlled Pauli powers, pruning, and CX cost](#r7) | The CX count $4\sum_jw_j$ of a controlled second-order Suzuki step, the pruning error $\lvert t\rvert m$, and the identity coefficient as a phase on the control |
| **Chebyshev Lanczos** ([Section 4](#4-chebyshev-lanczos)) | |
| [Result 8. The Chebyshev projected pencil](#r8) | The overlap matrix and projected Hamiltonian from Chebyshev moments $\mu_k$. A full requested trial dimension does not establish full trial rank or overlap with the ground state |
| [Proposition 9. Two moments per recurrence step](#r9) | All moments $\mu_0,\ldots,\mu_{2m-1}$ from $m$ applications of $K$ and $O(D+m)$ storage, as a classical computation |
| [Proposition 10. Differentiation through the kept overlap subspace](#r10) | The derivative of a selected Ritz value, including the motion of the truncated trial space, for fixed rank and a simple Ritz value |
| [Proposition 11. Variance-weighted pilot allocation](#r11) | The main-stage shot allocation $n_k\propto\lvert g_k\rvert\sqrt{v_k}$ that minimizes the linearized variance for continuous allocations, conditional on a pilot stage |
| [Proposition 12. Linear-work Gram noise estimates and confidence cutoff](#r12) | The expected Frobenius error of the sampled Gram matrix and the confidence radius $\epsilon_S$ that motivates the overlap cutoff. Neither alone bounds the Ritz-energy error |
| [Proposition 58. Non-destructive Lanczos continuation and its resource counts](#r58) | One trajectory with restored readout views supplies all moments on the exact simulated route (`shots=None`), with exact decoding identities that keep total-mass and padding terms, coherent operation counts, saves and saved bytes |
| **GCiM and ADAPT** ([Section 5](#5-gcim-and-adapt)) | |
| [Result 13. Generator-coordinate projection and thresholded diagonalization](#r13) | The projected pencil $H_pf=ESf$, canonical orthogonalization above an overlap threshold, and ADAPT basis growth and stopping. Thresholding alone does not identify the ground state |
| [Proposition 14. Removing identity offsets before a projected solve](#r14) | Solving without the identity offset $cI$ and adding $c$ is exactly equivalent, and avoids amplifying the offset by the inverse overlap scale |
| [Proposition 15. Pauli insertion after a complete generator exponential](#r15) | The parameter-shift derivative after a complete Pauli-sum generator exponential, with no commutation needed among its Pauli terms |
| [Proposition 16. Exact active-block synthesis for shared-index fermionic generators](#r16) | Exact circuits for shared-index double-excitation generators from their occupation blocks, for at most six active modes in the current construction |
| [Result 17. What a full-space residual can establish](#r17) | The residual interval $[\rho-r,\rho+r]$ and the Temple lower bound, which needs an independently valid excited-state lower bound $\beta$ |
| **Quantum linear systems** ([Section 6](#6-quantum-linear-systems)) | |
| [Proposition 18. Relative polynomial residual and physical solution recovery](#r18) | Relative solution error at most $\epsilon_{\rm inv}$ from an inverse-polynomial residual bound when every singular value of $A/\alpha$ is at least $1/\kappa$, and the physical recovery factor $\lVert b\rVert\kappa s/\alpha$ |
| [Proposition 19. Correct signed shortcut reflection and norm-search direction](#r19) | The corrected sign in Dalzell's shortcut-reflection bound and the direction of the norm search |
| [Proposition 20. Outward coverage of a rounded periodic encoding gap](#r20) | Error and smallest-eigenvalue bounds for the stored periodic encoding of $mI+d(2I-S-S^\dagger)$, and a polynomial domain covering both gaps |
| **QSP and QSVT** ([Section 7](#7-quantum-signal-processing-and-singular-value-transformation)) | |
| [Result 21. QSP conventions and the real-polynomial branch](#r21) | The $W_x$ and reflection phase conventions, and the real polynomial obtained by combining the $+\Phi$ and $-\Phi$ passes |
| [Proposition 22. A valid polynomial norming grid and a counterexample to a published extension](#r22) | The Chebyshev-node norming factor $\sec(\pi d/(2N))$, also for complex coefficients and affine intervals, and an exact counterexample to its published extension to an equispaced grid |
| [Proposition 23. Symmetric-phase Jacobians and an oversampled Newton step](#r23) | Phase Jacobians in $O(Md)$ for symmetric phases, and equality of the oversampled Newton step with the coefficient Newton step when the coefficient Jacobian is invertible |
| [Proposition 24. A finite Bessel-tail budget and explicit amplification residual](#r24) | A finite bound $R_T$ on the Jacobi–Anger Bessel tail, and the error $r_{\rm OAA}$ after oblivious amplitude amplification |
| **LCHS** ([Section 8](#8-linear-combinations-of-hamiltonian-simulation)) | |
| [Result 25. LCHS representation and the dissipativity premise](#r25) | The LCHS integral representation of $e^{-AT}$ for $L\succeq0$, its kernels, and recovery after a shift $s$ by the factor $e^{sT}$ |
| [Proposition 26. An integrated near-optimal-kernel tail bound](#r26) | The tail bound $2E_1(x)/(\beta C_\beta)$ of the near-optimal kernel, for every cutoff $K>0$ |
| [Proposition 27. Composite Gauss quadrature without a lower bound on dissipative time](#r27) | A composite Gauss–Legendre error bound for all $T\ge0$, and a finite stopping rule for the panel search |
| [Proposition 28. Constant-source Duhamel remainder in physical units](#r28) | The Gauss–Legendre error of the constant-source Duhamel integral in physical units, without a dimension factor |
| [Proposition 29. Affine SELECT angles and source-time scaling](#r29) | Affine SELECT angles with one unconditional and at most $a$ singly controlled rotations, and the elapsed-time weights of each branch |
| [Proposition 30. One physical vector budget for several output frames](#r30) | Bounds on the squared norm, an observable's quadratic form, the normalized vector and the normalized expectation, from one physical vector error $\delta$ that already includes every stage |
| **Block encodings and state preparation** ([Section 9](#9-block-encodings-and-state-preparation)) | |
| [Proposition 31. Singular-frame dilation and LCU normalization](#r31) | A one-ancilla dilation that block-encodes $A$ with normalization $\alpha$, padding without a second singular-value decomposition, and the LCU normalization $\alpha=\sum_j\lvert c_j\rvert$ |
| [Proposition 32. Circulant normalization and a Pauli-domain certificate](#r32) | The circulant LCU normalization, and an exact Frobenius-distance identity used to certify that a Pauli table is circulant, including absent Pauli labels |
| [Proposition 33. TT-SVD discarded weight and normalized fidelity](#r33) | For a tensor-train SVD sweep with discarded weight $D$, the error $\lVert G-T\rVert^2=D$ and the normalized fidelity $1-D$, in exact arithmetic |
| **Circuit synthesis and product formulas** ([Section 10](#10-circuit-synthesis-and-product-formula-budgets)) | |
| [Result 34. Multiplexed rotations, phase diagonals, and preparation trees](#r34) | CX counts of uniformly controlled rotations ($2^a$), phase diagonals and state-preparation trees ($2^n-2$) |
| [Proposition 35. Gate-count bounds for exact dense synthesis and their control cost](#r35) | Gate-count upper bounds $C_n$ for exact synthesis of an $n$-qubit unitary and for block-diagonal inputs, and the classical factorization work |
| [Proposition 36. Low-order product bounds with exact integer step selection](#r36) | First- and second-order product-formula bounds with exact integer step counts, Pauli-triangle coefficients, step selection on a common QPE time grid, and the weighted Strang bound for periodic LCHS |
| **QHD: model, schedules and evolution** ([Section 11](#11-quantum-hamiltonian-descent)) | |
| [Result 37. Grid Hamiltonians and boundary-dependent spectra](#r37) | Kinetic spectra on Dirichlet and periodic grids, and the one-hot encodings of hopping and potential |
| [Proposition 38. Stable interval integrals for the three schedules](#r38) | Stable step integrals of the quadratic, shifted-cubic and cubic schedules, with their floating-point allowances |
| [Proposition 39. Finite time-ordering and coefficient-quadrature bounds](#r39) | A per-step time-ordering bound (the smaller of an integral form and a derivative form) and the midpoint-quadrature error |
| [Derivation: QHD schedule values, integrals and derivative bounds](#qhd-schedule-arithmetic) | The rounding of each schedule's point values and step integrals, the exact rational values that the coefficient residual uses, and the derivative bounds $a_*,b_*,A_1,A_2,B_1,B_2$ that Proposition 39 needs |
| [Proposition 40. Product bounds for the actual one-hot and binary steps](#r40) | Product-formula errors of the emitted one-hot and binary steps, and commutator bounds computed from support tables |
| [Proposition 41. Split-step roundoff and work at a fixed grid](#r41) | Roundoff propagation in the classical split-step kernel, its observed first-order budget, and its work and storage |
| **QHD: circuit construction errors** | |
| [Proposition 42. Compensated running phase total and the phase condition for kept states](#r42) | The error bound of the compensated running phase total, the native phase-assignment allowances, and the requirement that a kept state's phase allowance be finite and below $\pi$ |
| [Proposition 43. Signed binary momentum and sparse kinetic polynomials](#r43) | The signed Walsh expansion of $q^2$ for the binary spectral kinetic, the Walsh structure of the finite-difference kinetic, and the difference between the two kinetic models |
| [Proposition 44. Binary phase synthesis, bit reversal, and truncation budgets](#r44) | CX counts and error terms of binary phase synthesis: Walsh coefficients, exponents, pruning, bit reversal and the approximate QFT |
| [Proposition 53. Nonnegative one-hot preparation and a finite product-state error](#r53) | One-hot chain preparation of nonnegative amplitudes with $3L$ CX gates, the error of a chain stopped early, and product-state errors |
| **QHD: readout** | |
| [Proposition 45. Probability-difference windows for mode selection](#r45) | The error of a computed probability difference given a state-error bound $\delta$, and the probability deficit of a mode selected within that window |
| **QHD: constrained problems** | |
| [Result 46. Normalized augmented-Lagrangian equations](#r46) | The normalized augmented-Lagrangian round objective with the PHR inequality term, and its multiplier and penalty updates |
| [Proposition 47. The finite-grid meaning of feasibility and complementarity stopping](#r47) | What feasibility and complementarity stopping imply on a finite grid: an objective-gap bound, an exact counterexample, and the stationarity diagnostic |
| Proposition 47, [Evaluated grid comparisons](#evaluated-grid-comparisons) | What `grid_minimum` and `constrained_grid_minimum` report, and a bound on the exact gap given an evaluation-error bound $E$ |
| [Proposition 51. A distance bound for the augmented-Lagrangian and refinement composition](#r51) | The distance from the returned point to the KKT point in a quadratic example with refinement |
| [Proposition 54. Slack variables for inequality constraints](#r54) | Explicit slack coordinates for inequality constraints, in four parts |
| Proposition 54, [Partial minimization over a nonnegative slack](#partial-minimization-over-a-nonnegative-slack) | Minimizing over a slack reproduces the PHR term. The smallest sufficient cap $U_j^*$, and the multiplier update at the projected point |
| Proposition 54, [Exact slack-grid excess and endpoint errors](#exact-slack-grid-excess-and-endpoint-errors) | The error a finite slack grid adds to the round objective for each grid convention, and how it enters the approximate-minimizer premise of Proposition 47 |
| Proposition 54, [A lower bound from support tables](#a-lower-bound-from-support-tables) | Slack caps from support-table minima, exact for disjoint supports and valid only on the domain that the bound covers |
| Proposition 54, [A periodic slack grid defines a different search kinetic](#a-periodic-slack-grid-defines-a-different-search-kinetic) | A periodic slack grid couples the two ends of the slack interval. The slack-elimination identity supplies no equivalence between a joint QHD evolution and one under $L$ |
| **QHD: box refinement** | |
| [Proposition 48. Search-model normalization and affine invariance](#r48) | Range normalization of the search model and its invariance under positive affine maps of box and objective, in exact table arithmetic |
| Proposition 48, [Selection across refinement levels](#selection-across-refinement-levels) | A conditional bound on the round objective of the point chosen across completed refinement levels |
| [Proposition 49. Marginal refinement and guaranteed joint mass](#r49) | Joint mass from marginals and simultaneous selected region coverage using Hoeffding and Clopper–Pearson bounds |
| Proposition 49, [Finite-shot confidence](#finite-shot-confidence) | A population joint-mass bound from counts, simultaneous over levels, axes and intervals, and sufficient shot counts |
| [Proposition 50. A sufficient stall condition and the limitation of a valley split](#r50) | The stall condition, what a valley split keeps, and the count test for a resolved valley |
| **QHD: fault-tolerant resources** | |
| [Proposition 52. Budgeted Clifford replacement and the logarithmic T estimate](#r52) | Clifford replacement within a synthesis budget, the per-rotation tolerance, and the logarithmic T estimate, which is an estimate and not a bound |
| **Resources, error composition and verification** ([Section 12](#12-resource-composition-error-evidence-and-verification)) | |
| [Proposition 55. Sound resource composition on a compact program](#r55) | Rules for composing work, depth and storage over a compact program without expanding repetitions |
| [Proposition 56. Error composition with units, provenance, and failure probabilities](#r56) | A total error bound and failure probability from components in one frame. A missing component leaves a subtotal |
| [Proposition 57. Scale-aware Gram and fidelity verification windows](#r57) | Floating-point bounds for a computed Gram matrix and the upper excursion window of a normalized fidelity |
| [Derivation: error budget of the `expm_multiply` reference](#expm-multiply-error-budget) | The first-order 2-norm error budget $\delta$ of a host kernel that evolves a unit state with SciPy's `expm_multiply`, the start terms of the QHD initial states, the mass window built from $\delta$, and a measured comparison |
| Derivation of the `expm_multiply` budget, [QHD initial-state vectors](#qhd-initial-state-vectors) | The per-variable entries of the uniform, kinetic ground and Gaussian initial states and the first-order error of each vector |
| [Derivation: QHD operation sizes](#qhd-operation-sizes) | The work units and bytes that QHD planning counts against `QHD.max_work` and `QHD.max_bytes` for support tables, step rows, the initial state, the augmented-Lagrangian layer and the classical kernels, with measured examples |

[Section 13](#13-implementation-and-evidence-index) gives the code and the repository checks for each result, and [Section 14](#14-open-mathematical-problems) lists open problems.

## 2. Expectation estimation and binary inference

<a id="r1"></a>
### Proposition 1. Simultaneous sampling bounds for a weighted expectation

**Provenance: Improved from a reference. Evidence: proved under the stated sampling model.**

Let $O=c_0I+\sum_{j=1}^m c_jP_j$ be Hermitian, with real coefficients and Pauli strings $P_j$. Let $X_{j,k}\in[-1,1]$ have conditional mean $\mu_j$ at every acquisition in setting $j$. Independent identically distributed shots are one sufficient model. For fixed counts $n_j\ge1$, $0<\delta<1$, and a predeclared family of $F\ge m\ge1$ settings, define

```math
r_j=\sqrt{\frac{2\log(2F/\delta)}{n_j}}.
```

With probability at least $1-\delta$, all setting means satisfy $|\widehat\mu_j-\mu_j|\le r_j$. Consequently,

```math
|\widehat E-E|\le\sum_{j=1}^m|c_j|r_j.
```

For a quadratic form with an unnormalized input $b$, multiply this radius by $\|b\|^2$. When the family comprises the $m$ science settings, equal shot counts can meet an absolute tolerance $\epsilon>0$ by taking

```math
n=\left\lceil 2\left(\frac{C}{\epsilon}\right)^2\log\frac{2m}{\delta}\right\rceil,\qquad C=\sum_{j=1}^m|c_j|,
```

with the same physical scaling included in $C$ when appropriate. Identity terms require no samples.

**Derivation.** Hoeffding's Theorem 1, Eq. (2.3), rescaled as stated after Eq. (2.4), and the two-tail relation in Eq. (1.4), give $\Pr(|\widehat\mu_j-\mu_j|>r)\le2e^{-n_jr^2/2}$. His Eq. (2.18) gives the relevant conditional-mean extension. Assign failure probability $\delta/F$ to each setting and apply the union bound, followed by the triangle inequality for the weighted sum. No independence between settings is needed for this propagation. For a time-uniform version, assign setting $j$ and sample count $n$ the failure probability $\delta/[F n(n+1)]$. Since $\sum_{n\ge1}1/[n(n+1)]=1$, the simultaneous radius is

```math
r_{j,n}=\sqrt{\frac{2\log(2F n(n+1)/\delta)}{n}}.
```

This permits stopping at a data-dependent sample count under the fixed-target conditional-mean premise. It does not permit choosing a favorable subset of observations. The extension consists of the family and time composition, not a sharper concentration inequality. [Hoeffding, Theorem 1 and Eqs. (1.4), (2.3), (2.18)][Hoeffding].

<a id="r2"></a>
### Proposition 2. Affine readout correction with shared calibration

**Provenance: Derived in NWQLib. Evidence: proved conditional on a stationary affine channel.**

Suppose an observed binary mean satisfies $z=a\mu+b$. Calibration at true means $+1$ and $-1$ gives means $z_0,z_1$. If $d=z_0-z_1\ne0$, then

```math
a=d/2,\qquad b=(z_0+z_1)/2,\qquad \mu=\frac{2z-z_0-z_1}{d},\qquad \nabla\mu=\left(\frac2d,-\frac{1+\mu}{d},\frac{\mu-1}{d}\right).
```

For a joint rectangular confidence set in $(z,z_0,z_1)$, if the denominator has constant nonzero sign throughout the rectangle, the extreme corrected means occur at its vertices. Its image can then be intersected with the physical parameter interval $[-1,1]$. If several science terms share calibration, their contributions to a first-order variance must be combined by calibration population before squaring.

**Proof.** Solving the two calibration equations gives $a,b$, and differentiating the rational expression gives the displayed Jacobian. Holding two coordinates fixed leaves a ratio of affine functions of the remaining coordinate. Its derivative has constant sign or is identically zero as long as the denominator does not vanish. Its extrema are therefore at endpoints. Applying this argument successively to the three coordinates proves the vertex rule. A weighted output $Y=\sum_jc_j\mu_j$ has differential $dY=\sum_s g_s\,dz_s$, where $g_s$ is the sum of all coefficient-weighted derivatives using population $s$. Hence its linearized variance is $g^T\Sigma g$. Replacing this by a sum of per-term variances would omit cross terms created by shared calibration. The variance is a delta-method approximation, whereas rectangle propagation is a finite conditional coverage statement. The channel's stationarity and transfer from calibration to science remain physical premises.

## 3. Phase and energy estimation

For $U=e^{-i\tau H}$ and a normalized preparation $|\psi\rangle$, the signal is $z_p=\langle\psi|U^p|\psi\rangle$. At a controlled-power boundary, ancilla expectations satisfy $\langle X_a\rangle=\operatorname{Re}z_p$ and $\langle Y_a\rangle=\operatorname{Im}z_p$. Exact non-destructive readout can acquire both along one controlled trajectory. Sampled Hadamard tests acquire them from separate sets of shots. Hamiltonian energy is the negative eigenphase of $U$, divided by $\tau$. An accepted near-unitary dense input $A$ defines the single unitary base $V=\operatorname{polar}(A)$ before any power is selected. Static QPE signals are evaluated against powers of that base, so controlled gap products obey the unitary semigroup relation in exact arithmetic.

<a id="r3"></a>
### Result 3. The QCELS scalar objective

**Provenance: From a reference. Evidence: the cited analytic identity.**

The least-squares fit $z_p\approx a e^{-ip\theta}$ eliminates its complex amplitude as

```math
a(\theta)=\frac1N\sum_p z_pe^{ip\theta},\qquad \min_a\frac1N\sum_p|z_p-ae^{-ip\theta}|^2=\frac1N\sum_p|z_p|^2-|a(\theta)|^2.
```

The energy estimate is $\theta/\tau$. This is [Ding and Lin, Eqs. (2), (9)–(12)][QCELS]. NWQLib performs a finite grid and bracket search for this one-level objective. The paper's multilevel Algorithm 1 and its accuracy theorems, which include target-overlap assumptions, do not describe that finite search automatically.

<a id="r4"></a>
### Result 4. Fourier estimators and their target populations

**Provenance: From a reference. Evidence: the cited estimator identities.**

Statistical phase estimation (SPE) approximates the spectral cumulative distribution and compares its filtered estimate with half a supplied lower bound $\eta$ on the target spectral weight. NWQLib keeps the zero-frequency contribution $1/2$ exactly, samples positive odd frequencies by coefficient magnitude, and includes their conjugate contributions. The implemented grid crossing follows [Wan, Berta and Campbell, Eqs. (2), (6), Appendix A Eqs. (A1)–(A2), and Algorithm 1][SPE]. Its grid has more decision points than the paper's binary search, so the paper's failure allocation must be changed before its coverage theorem can be applied.

Randomized Fourier estimation (RFE) draws powers $k_i\in\{0,\ldots,K-1\}$ uniformly and forms

```math
\widehat f_j=\frac1M\sum_{i=1}^M z_{k_i}e^{-2\pi i k_i j/K}.
```

The largest magnitude selects phase $2\pi j/K$, with energy $-2\pi j/(K\tau)$ after choosing the principal phase. The sign corresponds to NWQLib's direct $+\operatorname{Im}z_k$ quadrature. [Kshirsagar, Katabarwa and Johnson, Eqs. (5)–(7) and Algorithm 1][RFE]. A dominant Fourier peak of a mixed spectral measure need not identify its lowest energy. Theorem 2.1 is an eigenstate result with explicit sample requirements.

<a id="r5"></a>
### Result 5. Gaussian random-walk phase updates

**Provenance: From a reference. Evidence: the cited Gaussian moment update.**

For the experiment time $1/\sigma_k$ and feedback convention of [Granade and Wiebe, Algorithm 1 and Eqs. (5)–(7)][RWPE], one Bernoulli datum $d_k$ updates

```math
\mu_{k+1}=\mu_k+(-1)^{d_k}\sigma_k/\sqrt e,\qquad \sigma_{k+1}=\sigma_k\sqrt{(e-1)/e}.
```

Thus $\sigma_k=\sigma_0((e-1)/e)^{k/2}$ is known before the outcomes. These are moments used to replace the posterior by a Gaussian at each step. The reported width alone is not a proved confidence interval for the entire implemented procedure.

<a id="r6"></a>
### Result 6. Coherent phase-register readout

**Provenance: From a reference. Evidence: the cited theorem.**

For an eigenstate of $U$ with phase $\phi\in[0,1)$, controlled powers $U^{2^j}$, followed by the inverse quantum Fourier transform on an $m$-qubit phase register, give amplitudes

```math
a_y=2^{-m}\sum_{k=0}^{2^m-1}e^{2\pi i k(\phi-y/2^m)}.
```

The nearest $m$-bit phase outcome has probability at least $4/\pi^2$. [Cleve, Ekert, Macchiavello and Mosca, Section 5, Eqs. (5.1)–(5.4)][CoherentQPE]. The bound concerns an eigenstate and the ideal controlled powers. Approximation of those powers requires its own error bound.

<a id="r7"></a>
### Proposition 7. Controlled Pauli powers, pruning, and CX cost

**Provenance: Derived in NWQLib. Evidence: proved for the specified circuit decomposition.**

Let $H=c_I I+\sum_{j=1}^L c_jP_j$, where $P_j$ has nonidentity support $w_j\ge1$. A controlled second-order Suzuki step that applies every Pauli term twice has

```math
N_{\rm CX}=4\sum_{j=1}^Lw_j.
```

If a discarded set has coefficient absolute sum $m$, its contribution to a power at signed physical time $t$ is at most $|t|m$ in operator norm. The identity coefficient must appear as a phase on the control.

**Proof.** A Pauli rotation needs a parity ladder and its inverse, with $2(w_j-1)$ CX gates. Controlling only its central $R_z$ adds two CX gates, for $2w_j$ per occurrence. The Suzuki step has two occurrences per term. Basis changes are one-qubit gates. This is the controlled Pauli-evolution decomposition used by Qiskit 2.5.2, before routing and optimization. For pruning, $\|H-H'\|\le\sum_{j\in D}|c_j|=m$. Duhamel's identity for Hermitian generators gives $\|e^{-itH}-e^{-itH'}\|\le|t|m$. The controlled version has the same difference norm because its other block is zero. Finally, $\operatorname{ctrl}(e^{-itc_I}U)$ has the phase $e^{-itc_I}$ on its control-one branch. Removing it changes the measured interference signal. This connects the Pauli coefficient budget, the actual controlled construction, and its circuit count.

## 4. Chebyshev Lanczos

<a id="r8"></a>
### Result 8. The Chebyshev projected pencil

**Provenance: From a reference. Evidence: the cited polynomial identities.**

Write $H=cI+\alpha K$, with $K=K^\dagger$, $\|K\|\le1$, and $\mu_j=\langle\psi|T_j(K)|\psi\rangle$. On the trial vectors $T_i(K)|\psi\rangle$, $0\le i<m$,

```math
S_{ij}=\tfrac12(\mu_{i+j}+\mu_{|i-j|}),
```

```math
K_{ij}^{\rm proj}=\tfrac14(\mu_{i+j+1}+\mu_{|i+j-1|}+\mu_{|i-j|+1}+\mu_{||i-j|-1|}),\qquad H^{\rm proj}=cS+\alpha K^{\rm proj}.
```

These follow [Kirby, Motta and Mezzacapo, Eqs. (13)–(19)][Kirby]. Sampled acquisition uses separately prepared moment settings and the readout identities of [Kirby, Lemma 1 and Eqs. (23)–(27)][Kirby]. The exact simulated route with `shots=None` uses the single trajectory of [Proposition 58](#r58). The physical shift multiplies $S$, not the coordinate identity. A full requested trial dimension does not establish full trial rank or overlap with the ground state.

<a id="r9"></a>
### Proposition 9. Two moments per recurrence step

**Provenance: Improved from a reference. Evidence: proved in exact arithmetic.**

All moments $\mu_0,\ldots,\mu_{2m-1}$ can be computed classically with $m$ applications of $K$, $O(mD)$ vector arithmetic, and $O(D+m)$ storage for a $D$-dimensional operator accessed by matrix-vector products.

**Proof.** Let $v_k=T_k(K)\psi$. The three-term recurrence needs only $v_{k-1},v_k$. The product identity in [Kirby, Eq. (16)][Kirby] gives

```math
\mu_{2k}=2\langle v_k,v_k\rangle-\mu_0,\qquad \mu_{2k+1}=2\langle v_k,v_{k+1}\rangle-\mu_1.
```

The last odd moment uses $v_m$, requiring exactly $m$ applications from $v_0$. Each step contributes constant-many inner products. Storing the recurrence frontier and the $2m$ scalar moments proves the stated memory law. This reduces the straightforward recurrence through degree $2m-1$ without changing the moment definition. It is a classical computation and supplies no claim about quantum access cost.

<a id="r10"></a>
### Proposition 10. Differentiation through the kept overlap subspace

**Provenance: Improved from a reference. Evidence: proved for fixed rank and a simple selected Ritz value.**

Let $S=V_Rs_RV_R^\dagger+V_Ds_DV_D^\dagger$ be Hermitian, with $s_R$ above a cutoff and $s_D$ below it. Suppose every kept-discarded gap is nonzero and no eigenvalue crosses the cutoff under the perturbation. Let $c=V_Ry$ solve the reduced pencil, with $c^\dagger Sc=1$ and selected simple Ritz value $E$. Then

```math
dE=c^\dagger(dH-E\,dS)c+2\operatorname{Re}\bigl[(dV_Ry)^\dagger(H-ES)c\bigr].
```

Writing $r_D=V_D^\dagger(H-ES)c$, the second term is $\operatorname{Tr}((W+W^\dagger)dS)$, where

```math
W=V_D\left[\frac{r_{D,d}\overline y_r}{s_r-s_d}\right]_{dr}V_R^\dagger.
```

This includes the motion of the truncated trial space, which is needed when differentiating the regularized energy used in [Oumarou et al., Section 3.3.2, Eqs. (59)–(60)][Oumarou].

**Proof.** Differentiate the reduced generalized eigenproblem and its normalization. The usual stationary variation within the kept space gives $c^\dagger(dH-E\,dS)c$. Variation of the embedding contributes the two conjugate terms involving $dV_R$. The residual has zero kept projection, $V_R^\dagger(H-ES)c=0$. Differentiating $SV_R=V_Rs_R$ and projecting onto $V_D$ gives

```math
(V_D^\dagger dV_R)_{dr}=\frac{(V_D^\dagger dS V_R)_{dr}}{s_r-s_d}.
```

Substitution yields $W$ and its adjoint. Rotations within either subspace cancel, so internal degeneracies require no division. The moment-to-pencil maps of Result 8 are linear, which gives every $dE/d\mu_k$ by contracting the two matrix gradients with their coefficient arrays. The matrix products cost $O(m^3)$. At a cutoff crossing or a degenerate selected Ritz value, this differentiability statement does not apply.

<a id="r11"></a>
### Proposition 11. Variance-weighted pilot allocation

**Provenance: Improved from a reference. Evidence: proved for a local linearization and continuous allocations.**

For sampled acquisition with independent shots within and across separately prepared moment settings, with per-shot variances $v_k$, local energy derivatives $g_k$, and $n_k$ main-stage shots, the linearized variance is

```math
V=\sum_k\frac{g_k^2v_k}{n_k}\ge\frac{\left(\sum_k|g_k|\sqrt{v_k}\right)^2}{N},\qquad \sum_kn_k=N.
```

Equality holds on nonzero-weight settings when $n_k\propto|g_k|\sqrt{v_k}$. The $|g_k|$ allocation of [Oumarou et al., Eqs. (59)–(60)][Oumarou] is the equal-variance case.

**Proof.** Apply Cauchy–Schwarz to $|g_k|\sqrt{v_k/n_k}$ and $\sqrt{n_k}$. Equality gives the stated proportions. If a pilot chooses the main counts, condition on the pilot. Fresh main observations then have fixed counts under that conditioning, so the variance calculation remains valid with the chosen allocation. NWQLib uses only those main observations in the final moments. Estimated sensitivities, integer counts, minimum allocations, and nonlinear Ritz reconstruction limit the conclusion to the local allocation problem. The pilot's cost is part of the total workload.

<a id="r12"></a>
### Proposition 12. Linear-work Gram noise estimates and confidence cutoff

**Provenance: Derived in NWQLib. Evidence: proved under independent moment acquisition.**

For sampled acquisition from separate preparations, assume independent and unbiased moment estimates, with $\sigma_k^2=\operatorname{Var}(\widehat\mu_k)$. For the $m\times m$ Gram matrix of Result 8,

```math
\mathbb E\|\widehat S-S\|_F^2=\sum_{k=0}^{2m-2}w_k\sigma_k^2,\qquad w_k=(h_k+t_k+c_k)/4,
```

where $h_k=m-|k-(m-1)|$, $t_0=m$, $t_k=2(m-k)$ for $0<k<m$, and $t_k=0$ otherwise. Here $c_0=2$, $c_k=4$ for $0<k<m$, and $c_k=0$ otherwise. If $r$ sampled moments enter the Gram matrix and every setting has at least $n_{\min}$ shots satisfying the fixed-target conditional-mean model of Result 1, then with probability at least $1-\delta$,

```math
\|\widehat S-S\|_2\le m\sqrt{\frac{2\log(2r/\delta)}{n_{\min}}}=:\epsilon_S.
```

**Proof.** An entry uses moment indices $a=i+j$, $b=|i-j|$. Its variance is $(\sigma_a^2+\sigma_b^2+2\mathbf1_{a=b}\sigma_a^2)/4$. Counting antidiagonals, absolute-difference diagonals, and their coincidences gives $h,t,c$. The expected squared Frobenius norm is the sum of entry variances even though entries share moments. These weights take $O(m)$ scalar work. For the confidence statement, Hoeffding's Theorem 2 and a union bound give $\max_k|\widehat\mu_k-\mu_k|\le e=\sqrt{2\log(2r/\delta)/n_{\min}}$. Every entry error is at most $e$, hence the Frobenius norm is at most $me$. Weyl's inequality implies that an observed overlap eigenvalue above $2\epsilon_S$ has a corresponding population eigenvalue above $\epsilon_S$. This motivates the explicit confidence cutoff. The empirical variance formula, used with sample variances, supplies an exploratory noise scale rather than that coverage guarantee. Neither quantity alone bounds the Ritz-energy error.

<a id="r58"></a>
### Proposition 58. Non-destructive Lanczos continuation and its resource counts {#proposition-58-non-destructive-lanczos-continuation-and-its-resource-population}

**Provenance: Derived in NWQLib from [Kirby, Motta and Mezzacapo, arXiv:2208.00567v4, Eqs. (23)–(26)][Kirby]. Evidence: proved for coherent unitary evolution and non-destructive saves, with exact raw-mass and padding identities.**

**Statement.** On the exact simulated route with `shots=None`, one initial preparation and $m-1$ walks supply all moments $\mu_0,\ldots,\mu_{2m-1}$ of Result 8. At boundary $j$ the even view reads $\mu_{2j}$ and the odd view reads $\mu_{2j+1}$, and each view is undone before the next view or walk. This holds under the four premises below: deterministic coherent simulation with non-destructive probability saves, restored views, the active-subspace and padding convention, and the distinction between ideal and rounded states. For an arbitrary, possibly unnormalized vector entering an ideal view, the decoded values satisfy exact relations that keep its total-mass and padding contributions. The proposition also gives the coherent operation counts of the executed, paired and separate-circuit schedules, the number of saves and the saved bytes. Hardware and sampled routes use separate preparations for their moment settings, and non-destructive simulator access supplies no hardware acquisition saving or independence claim for numerical errors.

Let $|\phi\rangle$ be the normalized reference $|\psi\rangle$ of Result 8. For the Pauli encoding, write $H=cI+\sum_{l=0}^{L-1}c_lP_l$, where the $c_l$ are real and nonzero and the $P_l$ are nonidentity Pauli operators. Put $\alpha=\sum_l|c_l|>0$, $K=\sum_l(c_l/\alpha)P_l$, $a=\lceil\log_2 L\rceil$, and

```math
|G\rangle=\sum_{l=0}^{L-1}\sqrt{|c_l|/\alpha}\,|l\rangle,\qquad \mathrm{PREP}|0\rangle=|G\rangle,\qquad R=(2|G\rangle\langle G|-I)\otimes I.
```

The index register is written first in tensor products. In outcome integers it occupies the $a$ least significant bits. Define the padding projector, the readout observable, and signed SELECT by

```math
\Pi_{\rm pad}=\sum_{l=L}^{2^a-1}|l\rangle\langle l|\otimes I,\qquad U_{\rm read}=\sum_{l=0}^{L-1}|l\rangle\langle l|\otimes\operatorname{sign}(c_l)P_l,\qquad U=U_{\rm read}+\Pi_{\rm pad}.
```

Thus SELECT acts as identity on padding. Set $W=RU$ and $|\psi_j\rangle=W^j(|G\rangle\otimes|\phi\rangle)$. On the exact simulated route with `shots=None`, one initial preparation and $m-1$ walks supply all moments $\mu_0,\ldots,\mu_{2m-1}$. At boundary $j$, the even view reads $\mu_{2j}$ and the odd view reads $\mu_{2j+1}$. The Plan supplies $\mu_0=1$ algebraically, so it needs no view at degree zero. Hardware and sampled routes use separate preparations for their moment settings. Non-destructive simulator access supplies no hardware acquisition saving or independence claim for numerical errors.

The premises are as follows.

1. **Deterministic coherent simulation.** The selected preparation and walk describe coherent evolution, and the backend supports non-destructive probability saves. There is no mid-circuit measurement, reset, postselection, stochastic noise trajectory, or outcome-dependent continuation. The setting `shots=None` alone does not establish these premises.
2. **Restored views.** The even view applies $T_e=\mathrm{PREP}^\dagger\otimes I$, saves the index marginal, and applies $T_e^{-1}=\mathrm{PREP}\otimes I$ before continuation. The odd view applies $T_o=B=\sum_{l=0}^{2^a-1}|l\rangle\langle l|\otimes B_l$, where $B_lP_lB_l^\dagger$ is the computational-basis Pauli parity on active address $l$, and $B_l=I$ on padding. It saves the index and system marginal and applies $B^\dagger$ before continuation. At a shared boundary the next view also starts from the restored state. The last save needs no inverse when nothing follows it.
3. **Active subspace and padding convention.** The ideal walk stays in the span of addresses $0,\ldots,L-1$, where $U_{\rm read}=U$. An odd outcome at an active address has score $\operatorname{sign}(c_l)$ times the Pauli-support parity. A padded address has score zero. Exact probability reduction keeps every returned weight with denominator one, including any mass on padding. It performs no normalization or postselection. For sampled counts the denominator is the total returned count, including padding.
4. **Rounded states.** The ideal identities concern exact arithmetic. A computed state can have nonunit total mass, padding mass, and errors from preceding view excursions. These populations stay in the decoder. Rounded representations of a view and its adjoint can also have a nonzero restoration residual, which must be distinguished from non-destructive saving.

**Proof.** Since $R^2=U^2=I$, $RWR=W^\dagger$ and $R|\psi_0\rangle=|\psi_0\rangle$. The block-encoding identity $\langle\psi_0|W^k|\psi_0\rangle=\mu_k$ therefore gives

```math
\langle\psi_j|R|\psi_j\rangle=\langle\psi_0|R W^{2j}|\psi_0\rangle=\mu_{2j},\qquad \langle\psi_j|U|\psi_j\rangle=\langle\psi_0|R W^{2j+1}|\psi_0\rangle=\mu_{2j+1}.
```

The first equality in each chain follows by moving $R$ to the left through $(W^\dagger)^j$. The second also uses $U=RW$. This is the moment identity of Kirby et al. Since $|G\rangle$ has no padding amplitude and both $R$ and $U$ preserve the active subspace, $\Pi_{\rm pad}|\psi_j\rangle=0$, so $U_{\rm read}$ gives the same odd moment. The simulator continuation follows because a probability save leaves its input vector unchanged and $T_e^{-1}T_e=B^\dagger B=I$ for the ideal views. Applying either view, saving, and undoing it consequently returns exactly $|\psi_j\rangle$ before the next view or walk.

For an arbitrary, possibly unnormalized vector $x$ entering an ideal view, let $t=\langle x,x\rangle$, $q=\langle x|\Pi_{\rm pad}|x\rangle$, and $p_0$ be the index-zero probability after $T_e$. If $p_{l,s}$ denotes the joint probability after $B$, and $\chi_l(s)$ is the Pauli-support parity, the decoded values satisfy the exact relations

```math
d_e(x)=2p_0-t=\langle x|R|x\rangle=(2p_0-1)-(t-1),
```

```math
d_o(x)=\sum_{l=0}^{L-1}\sum_s\operatorname{sign}(c_l)\chi_l(s)p_{l,s}=\langle x|U_{\rm read}|x\rangle=\langle x|U|x\rangle-q.
```

In particular, setting $t$ to one would change the even value by $t-1$, and including identity SELECT on padding would change the odd value by $q$. The exact second moments of the scores are $t$ for the even view and $t-q$ for the odd view. For $x=\psi_j+e$, the corresponding deviations from the ideal moments are

```math
d_e(x)-\mu_{2j}=2\left[p_0-\frac{\mu_{2j}+1}{2}\right]-(t-1)=2\operatorname{Re}\langle\psi_j|R|e\rangle+\langle e|R|e\rangle,
```

```math
d_o(x)-\mu_{2j+1}=2\operatorname{Re}\langle\psi_j|U|e\rangle+\langle e|U|e\rangle-q.
```

These equalities keep both the total-mass and padding contributions, without assuming that they are small. They apply to the exact probabilities of the specified vector and ideal view. To include numerical view and readout errors, let $T$ be that ideal view, $D=\sum_b s_b\Pi_b$ its diagonal score operator, and $\widehat y=Tx+f$ the computed state at probability formation. Here $\Pi_b$ projects onto marginal bin $b$, $s_b\in\{-1,0,1\}$ is its score, $\epsilon_b=\widehat p_b-\langle\widehat y|\Pi_b|\widehat y\rangle$ is probability-evaluation error, and $\epsilon_{\rm sum}$ is the error in the final signed sum. Then

```math
\widehat d=d(x)+2\operatorname{Re}\langle Tx|D|f\rangle+\langle f|D|f\rangle+\sum_b s_b\epsilon_b+\epsilon_{\rm sum}.
```

The vector $f$ includes the current view's construction and application errors. Earlier excursions enter $e$. More explicitly, for represented forward and inverse operators $F$ and $F_{\rm inv}$, put $E=F_{\rm inv}F-I$. With application errors $e_f,e_b$, restoration obeys

```math
\widehat x_{\rm after}=(I+E)x+F_{\rm inv}e_f+e_b,\qquad \|\widehat x_{\rm after}-x\|_2\le\|E\|_2\|x\|_2+\|F_{\rm inv}\|_2\|e_f\|_2+\|e_b\|_2.
```

Saving itself contributes no state disturbance. An adjoint of a rounded matrix need not be its exact inverse, so exact-arithmetic restoration does not imply bitwise equality of computed trajectories or a certified numerical bound when a preparation record excludes native operations from its error model.

**Resource counts.** For $L\ge2$, write $g_{\rm init}$ for the coherent cost of preparing both registers, $g_W$ for one complete walk, and $g_e,g_{e^{-1}},g_o,g_{o^{-1}}$ for the forward and inverse readout-tail costs in the selected native realization. The walk cost already includes SELECT, coefficient unpreparation, the positive zero reflection, and coefficient preparation. A schedule that explicitly reads both degrees at each of the $m$ boundaries and restores every view has

```math
G_{\rm paired}=g_{\rm init}+(m-1)g_W+m(g_e+g_{e^{-1}}+g_o+g_{o^{-1}})
```

coherent operations and $2m$ saves. The executed full NWQLib schedule supplies degree zero algebraically and stops after the last odd save. It consists of one initial preparation, $m-1$ walks, $m-1$ even views with their inverses, $m$ odd views with $m-1$ inverses, and $2m-1$ saves. Thus, for additive costs of fixed lowered blocks,

```math
G_{\rm exact}=g_{\rm init}+(m-1)g_W+(m-1)(g_e+g_{e^{-1}})+m g_o+(m-1)g_{o^{-1}}.
```

Separate circuits for acquired degrees $1,\ldots,2m-1$ use $2m-1$ initial preparations and $\sum_{k=1}^{2m-1}\lfloor k/2\rfloor=m(m-1)$ walks. They need the forward tail for each degree and no restoration tail, giving the coherent cost

```math
G_{\rm separate}=(2m-1)g_{\rm init}+m(m-1)g_W+(m-1)g_e+m g_o.
```

These compare one evaluation per acquired degree. Sampled work also counts its shot repetitions and measurements. The walk ratio is $m$ for $m>1$, while a total-gate comparison must include the displayed readout-tail and inverse-tail terms. A preparation record counts the actual emitted native operations, including any executed final inverse or suffix. Saves count toward readout work and output storage, but are excluded from the coherent-operation count.

For a nonempty set $A$ of acquired degrees, let $J=\max_{k\in A}\lfloor k/2\rfloor$ and $k_*=\max A$. The trajectory stops after $J$ walks, uses $|A|$ saves, and executes an inverse for every requested view except the final one. With $g_k$ and $g_{k^{-1}}$ the costs of the view for degree $k$ and its inverse,

```math
G_A=g_{\rm init}+Jg_W+\sum_{k\in A}g_k+\sum_{k\in A\setminus\{k_*\}}g_{k^{-1}},\qquad N_{W,\rm separate}(A)=\sum_{k\in A}\lfloor k/2\rfloor.
```

Algebraically known degrees contribute no save. For $L=1$, $a=0$ and $K^2=I$, so every even moment is algebraic and only the selected odd settings remain. If $\alpha=0$, the scalar-operator route supplies all moments algebraically. An empty acquired set needs no preparation, walk, or save.

The simulator state has width $w=n+a$, including the index register. Each even save contains $2^a$ real probabilities and each odd save contains $2^w$ real probabilities. With $N_e$ even and $N_o$ odd acquired degrees, the saved float64 numerical payload is $8(N_e2^a+N_o2^w)$ bytes. For the full $L\ge2$ schedule this is $8[(m-1)2^a+m2^w]$ bytes, in addition to the live complex128 state of $16\,2^w$ bytes and implementation workspace, metadata, and transport copies. Aer accumulates the selected saves until completion. Proposition 9's $O(D+m)$ storage law concerns the classical recurrence and does not bound this simulator storage.


## 5. GCiM and ADAPT

<a id="r13"></a>
### Result 13. Generator-coordinate projection and thresholded diagonalization

**Provenance: From a reference. Evidence: the cited projected equations.**

For normalized preparation columns $V=(\phi_1,\ldots,\phi_m)$, generator-coordinate-inspired methods (GCiM) form $H_p=V^\dagger HV$, $S=V^\dagger V$, and solve $H_pf=ESf$. [Zheng et al. (2023), Eqs. (13)–(15), (33)–(34)][GCIM2023]. Canonical orthogonalization keeps overlap eigenvectors above a selected positive threshold and solves $X^\dagger H_pX$, where $X=V_RS_R^{-1/2}$. [Epperly, Lin and Nakatsukasa, Algorithm 1.1][Epperly], and [Zheng et al. (2024), Appendix F, Eqs. (F1)–(F3)][GCIM2024]. The exact pencil has Rayleigh–Ritz bounds within its trial space. A noisy acquired pencil need not obey the full operator's spectral enclosure, and thresholding alone does not identify its ground state.

The adaptive variant screens a surrogate product state using $\langle[H,A_j]\rangle$, selects a largest-magnitude unselected generator, and expands the basis with its individual excitation and the cumulative product state. Its default fixed angle is $\pi/4$. The basis-growth construction and flat-energy stopping rule follow [Zheng et al. (2024), main text p. 4 and Methods p. 9][GCIM2024]. The published stopping threshold is the smaller of a user cap and $20\%$ of the remaining pool size, applied to a counter of consecutive small energy changes. This is a search heuristic. Optional product-state optimization does not replace the projected pencil by a variational energy of the surrogate.

<a id="r14"></a>
### Proposition 14. Removing identity offsets before a projected solve

**Provenance: Derived in NWQLib. Evidence: proved in exact arithmetic, with a first-order conditioning analysis.**

If $H=cI+H_0$, then $H_p=cS+H_{0,p}$. Solving $(H_{0,p},S)$ and adding $c$ to each Ritz value is exactly equivalent to solving $(H_p,S)$ with the same overlap subspace. It avoids amplification of the identity offset by the inverse overlap scale during numerical orthogonalization.

**Proof.** Since $X^\dagger SX=I$, $X^\dagger H_pX=cI+X^\dagger H_{0,p}X$. Their eigenvectors coincide and their eigenvalues differ by $c$. A perturbation $\Delta H_p$ produces an orthogonalized perturbation bounded by $\|X\|^2\|\Delta H_p\|=\|\Delta H_p\|/s_{\min}$. Thus an assembly or multiplication error on the scale $u|c|$ can enter on the scale $u|c|/s_{\min}$, with additional norm and dimension factors determined by the multiplication. Removing the offset before products removes that source of amplification. It cannot recover low-order information already lost when the original entries of $H$ were supplied.

The same identity explains why overlap observations should also supply the identity contribution of the measured Hamiltonian pencil. Independent identity acquisitions would replace $cS$ by a different noisy matrix and break exact shift covariance.

<a id="r15"></a>
### Proposition 15. Pauli insertion after a complete generator exponential

**Provenance: Improved from a reference. Evidence: proved for an anti-Hermitian Pauli-sum generator.**

Let $A=i\sum_k a_kP_k$, with real $a_k$, and let $\chi=e^{\theta A}\psi$. For the observable $O$ seen after this factor in an ansatz, define $E_{k,\pm}=\langle\chi|e^{\mp i\pi P_k/4}Oe^{\pm i\pi P_k/4}|\chi\rangle$. Then

```math
\frac{dE}{d\theta}=\sum_ka_k(E_{k,+}-E_{k,-}).
```

No commutation among the $P_k$ is required. This specializes the derivative decomposition of [Schuld et al., Section III.B, Eqs. (16)–(17)][Schuld] using their two-eigenvalue identity, Theorem 1 and Eqs. (13)–(14), at a location valid for the complete spin-adapted exponential.

**Proof.** Since $A$ commutes with $e^{\theta A}$, differentiation gives $E'=\langle\chi|[O,A]|\chi\rangle$. Expanding $e^{\pm i\pi P/4}=(I\pm iP)/\sqrt2$ shows $E_{P,+}-E_{P,-}=\langle\chi|[O,iP]|\chi\rangle$. Summing proves the formula. Applying primitive parameter shifts separately inside a noncommuting product that merely approximates $e^{\theta A}$ would differentiate a different ansatz. NWQLib also supports finite-frequency shifts from the active generator spectrum, following [Wierichs et al., Sections 3.2–3.4 and Appendix B][Wierichs].

<a id="r16"></a>
### Proposition 16. Exact active-block synthesis for shared-index fermionic generators

**Provenance: Derived in NWQLib. Evidence: proved for the supported real anti-Hermitian generators.**

The shared-index double-excitation generators can be synthesized from their finite occupation blocks by fermionic permutations and two-level rotations. This supplies the cases outside the compact pair/split and four-distinct formulas of [Magoulas and Evangelista, Sections V–VI and Tables I–II][SpinAdapted].

**Proof.** A fermionic swap is SWAP followed by CZ. It exchanges mode occupations and gives $-1$ to a pair of occupied modes, exactly the fermionic exchange sign. Moving active modes to a contiguous set therefore preserves spectator signs. In that ordering, the generator decomposes into invariant occupation blocks $B$. Real anti-Hermiticity gives $B^T=-B$, so $Q=e^{\theta B}\in SO(k)$. Givens elimination reduces $Q$ to an upper triangular orthogonal matrix. Choosing the first $k-1$ diagonal entries positive makes them one, and determinant one forces the last entry to one. Reversing the elimination expresses $Q$ as two-level real rotations, each implemented as a two-level $R_y$. Undo the fermionic permutation. This proves the ideal circuit identity. The current construction accepts at most six active modes and blocks of dimension five. Numerical exponentials and angle formation introduce separate floating-point errors.

<a id="r17"></a>
### Result 17. What a full-space residual can establish

**Provenance: From a reference. Evidence: standard spectral residual and Temple bounds.**

For a normalized vector $v$, Hermitian $H$, Rayleigh quotient $\rho=v^\dagger Hv$, and residual norm $r=\|Hv-\rho v\|$, the interval $[\rho-r,\rho+r]$ contains some eigenvalue. If an independently valid excited-state lower bound $\beta>\rho$ places every eigenvalue except the ground eigenvalue above $\beta$, then

```math
\rho-\frac{r^2}{\beta-\rho}\le\lambda_0\le\rho.
```

The inclusion and Temple inequalities are [Zhu, Argentati and Knyazev, Section 2, Eqs. (2.6) and (2.3)][ResidualBounds]. The upper endpoint is Rayleigh–Ritz. A second projected Ritz value does not generally supply the required lower bound $\beta$ on the next eigenvalue. Thus the implementation's Kato–Temple-style quantity remains heuristic when that spectral separation is unverified.

## 6. Quantum linear systems

<a id="r18"></a>
### Proposition 18. Relative polynomial residual and physical solution recovery

**Provenance: Derived in NWQLib. Evidence: proved conditional on the spectral domain and polynomial bound.**

Let $A$ be invertible and $\alpha\ge\|A\|$. Suppose an odd polynomial $P$ satisfies

```math
\sup_{1/\kappa\le|x|\le1}|\kappa xP(x)-1|\le\epsilon_{\rm inv},
```

and every singular value of $A/\alpha$ is at least $1/\kappa$. Define the inverse-oriented odd transform using the Hermitian dilation

```math
\mathcal H=\begin{pmatrix}0&A/\alpha\\A^\dagger/\alpha&0\end{pmatrix},\qquad P(\mathcal H)(b,0)=(0,y).
```

Then $x_P=(\kappa/\alpha)y$ obeys

```math
\|x_P-A^{-1}b\|\le\epsilon_{\rm inv}\|A^{-1}b\|.
```

For a circuit encoding $P/s$ on a normalized right-hand side, the physical recovery factor is $\|b\|\kappa s/\alpha$.

**Proof.** Let $A=U\Sigma V^\dagger$, $s_j=\sigma_j/\alpha$, and $b=\sum_jb_ju_j$. Odd functional calculus on the dilation gives $y=\sum_jb_jP(s_j)v_j$. The residual inequality gives $|(\kappa/\alpha)P(s_j)-1/\sigma_j|\le\epsilon_{\rm inv}/\sigma_j$. Squaring, weighting by $|b_j|^2$, and summing proves the relative vector bound. The circuit's normalization divides $P$ by $s$ and $b$ by $\|b\|$, which yields the recovery factor. The Hermitian case is the same argument in the signed eigenbasis.

The classical model evaluates the lower block $y=VP(\Sigma/\alpha)U^\dagger\widehat b$ directly from one original factorization, with $\widehat b=b/\|b\|$. Conjugating $\mathcal H$ by ${\rm diag}(U,V)$ gives $\begin{pmatrix}0&S\\S&0\end{pmatrix}$ with $S=\Sigma/\alpha$, whose even powers are diagonal and whose odd powers carry the odd powers of $S$ in the off-diagonal blocks, so linear combination gives the odd polynomial identity. Starting from $(0,\widehat b)$ instead would produce $UP(\Sigma/\alpha)V^\dagger\widehat b$ in the upper block. For Hermitian $A$, one eigendecomposition supplies signed eigenvalues $\lambda$ and the action $VP(\lambda/\alpha)V^\dagger\widehat b$. Using $P(|\lambda|)$ would lose the sign of negative eigenvalues. For the positive padding $A\oplus\alpha I$ with zero-padded $b$, the factors extend as $U\oplus I$, $V\oplus I$ and $\Sigma\oplus\alpha I$. The dummy coordinates have zero weight and no inverse output, so they need no decomposition.

NWQLib fits the relative residual in odd Chebyshev coefficients and checks it on an affine Chebyshev grid using Result 22 with degree $d+1$. The fit's accepted numerical bound concerns this polynomial stage. Spectral estimates, block-encoding error, fitted phases, and readout remain separate. The degree form $O(\kappa\log(\kappa/\epsilon))$ comes from [Childs, Kothari and Somma, Lemmas 17–19, Eqs. (74), (77), (88)][CKS]. The finite search limit inspired by that form is not a theorem that this least-squares fit must succeed below it.

<a id="r19"></a>
### Proposition 19. Correct signed shortcut reflection and norm-search direction

**Provenance: Improved from a reference. Evidence: proved from the source formulas.**

Let $0<\Delta<1$, $0<\eta<1$, $\ell=\lceil\log(2/\eta)/(2\Delta)\rceil$, and

```math
z(x)=\frac{1+\Delta^2-2x^2}{1-\Delta^2},\qquad D=T_\ell(z(0)),\qquad K(x)=\frac{2T_\ell(z(x))+2}{D+1}-1.
```

Then $K(0)=1$, $|K(x)|\le1$ on $[-1,1]$, and

```math
-1\le K(x)\le-1+\frac{4\eta}{1+\eta},\qquad \Delta\le|x|\le1.
```

This corrects the absolute-value sign in [Dalzell, Appendix B.3, Lemma 3, item 2][Dalzell].

**Proof.** The map $z$ sends the off-kernel domain to $[-1,1]$, so $-1\le T_\ell(z(x))\le1$ there. Moreover,

```math
\operatorname{arccosh}z(0)=2\operatorname{artanh}\Delta\ge2\Delta,\qquad D=\cosh(\ell\operatorname{arccosh}z(0))\ge1/\eta.
```

Substituting these bounds into $K$ proves the off-kernel inequality. On the intervening interval, $T_\ell$ increases from one to $D$, so the full range stays within $[-1,1]$, and substitution at zero gives one. An upper bound of $-1+4\eta/(1+\eta)$ on $|K|$ would be negative for $\eta<1/3$, which is impossible. Equations (62)–(63) and their proof support the signed statement.

The same source's Eq. (44) gives ideal kernel-projection success $q(t)=t^2/(t^2+\|x\|^2)$. It is increasing in $t$, and $q(t)>1/2$ implies $t>\|x\|$. A noiseless search must therefore discard candidates above this probe, contrary to the direction printed in Section 5.2. NWQLib uses the direction of Eq. (44). Noisy decisions still require the source's finite-sample treatment. For a supplied norm estimate $t\ne\|x\|$, the shortcut's trace-distance bound remains $\eta/\cos\theta_t$, with $\theta_t=\arctan(\|x\|/t)$, rather than the known-norm simplification $\sqrt2\eta$ of Eq. (20).

<a id="r20"></a>
### Proposition 20. Outward coverage of a rounded periodic encoding gap

**Provenance: Derived in NWQLib. Evidence: proved from the stored coefficients.**

For $A=mI+d(2I-S-S^\dagger)$ on $2^q\ge2$ periodic sites, with $m>0$, $d\ge0$, the exact spectral endpoints are $m$ and $m+4d$. Let the implementation store $c_0=\operatorname{fl}(m+2d)$ and normalization $\widehat\alpha$. Set

```math
e_0=|c_0-m-2d|,\quad e_\alpha=|\widehat\alpha-c_0-2d|,\quad g=c_0-2d-e_\alpha.
```

The ideal block of the stored coefficient construction, before native gate-synthesis roundoff, has physical operator error at most $e_0+e_\alpha$, and its smallest eigenvalue is at least $g$. If $g>0$, choosing the polynomial domain parameter to cover $\widehat\alpha/\min(m,g)$ covers both the original and encoded gaps.

**Proof.** The Fourier eigenvalues of $S+S^\dagger$ are $2\cos(2\pi k/2^q)$. Changing the diagonal contributes norm $e_0$. The normalization residual contributes at most $e_\alpha$, by the construction's coefficient sum. The triangle inequality gives the error, and Weyl's inequality gives the gap. Exact rational evaluation on the stored binary64 numbers followed by upward conversion makes the error and domain coverage outward. For $q=1$, the two wrapped shifts coincide and their coefficients combine to $-2dX$, giving the same endpoints. The encoded gap parameter, physical condition number, and polynomial-domain floor are distinct quantities.

## 7. Quantum signal processing and singular-value transformation

<a id="r21"></a>
### Result 21. QSP conventions and the real-polynomial branch

**Provenance: From a reference. Evidence: the cited polynomial calculus.**

In the $W_x$ convention,

```math
W(x)=\begin{pmatrix}x&i\sqrt{1-x^2}\\i\sqrt{1-x^2}&x\end{pmatrix},\qquad U_\Phi(x)=e^{i\phi_0Z}\prod_{j=1}^d W(x)e^{i\phi_jZ}.
```

Its upper-left entry has the degree and parity constraints of [Martyn et al., Theorem 1 and Eqs. (1)–(3)][Martyn]. Zero phases give $T_d(x)$. The reflection convention uses the shifts in their Eq. (14) and Appendix A.2, Eq. (A5), with the compensating factor $i^d$ kept explicitly. Quantum singular-value transformation (QSVT) applies this calculus to the selected block encoding, subject to its input and output projectors. [Gilyén et al., Theorem 17 and Lemma 19][GSLW]. A coherent combination of the $+\Phi$ and $-\Phi$ passes obtains the real polynomial as in Corollary 18, Eq. (33). The phase convention and the $(b,0)$ dilation orientation of Proposition 18 jointly determine the physical output.

<a id="r22"></a>
### Proposition 22. A valid polynomial norming grid and a counterexample to a published extension

**Provenance: Improved from a reference. Evidence: proved using the cited norming theorem.**

For a polynomial of degree at most $d<N$, sampled at $x_j=\cos((2j-1)\pi/(2N))$, $1\le j\le N$,

```math
\|p\|_{\infty,[-1,1]}\le\sec\!\left(\frac{\pi d}{2N}\right)\max_j|p(x_j)|.
```

This is [Ehlich and Zeller, Satz 2, Eqs. (12)–(14), pp. 42–43][EZ]. It also holds for complex coefficients and after an affine change of interval. NWQLib applies this theorem to inverse residuals and fitted QSP polynomials.

**Derivation and correction.** At a point $x_*$ attaining the complex modulus maximum, rotate $p$ by the phase of $p(x_*)$ and take its real part. The resulting real polynomial has value $|p(x_*)|$ there and sample magnitudes bounded by those of $p$. Apply the real theorem. An affine coordinate change preserves polynomial degree.

[Sünderhauf et al., Eq. (25)][InversePolynomials] states the same factor for $D_N=\{-1+2i/N:0\le i<N\}$. On that grid, the degree-one polynomial $p(x)=(x+1)/(2-2/N)$ has sample maximum one but $p(1)=N/(N-1)$. At $N=25$, this is $25/24$, greater than $\sec(\pi/50)$. Indeed $\cos(\pi/50)\ge1-\pi^2/5000>24/25$. This exact counterexample disproves the extension on its stated grid. The correction is to use the Chebyshev nodes. The library evaluates the corrected bound in binary64, so formal outward certification of the polynomial evaluations remains a separate question.

<a id="r23"></a>
### Proposition 23. Symmetric-phase Jacobians and an oversampled Newton step

**Provenance: Improved from a reference. Evidence: proved in exact arithmetic where the coefficient Jacobian is invertible.**

For symmetric phases $\phi_j=\theta_{\min(j,d-j)}$, all objective-node derivatives can be formed in $O(Md)$ arithmetic on $2\times2$ matrices. An oversampled least-squares Newton step on the node residual equals the Newton step on the definite-parity Chebyshev coefficients when the latter Jacobian is invertible. This extends the objective construction of [Dong, Meng, Whaley and Lin, Eqs. (23)–(24), (27)–(30)][DMWL] and the coefficient Newton iteration of [Dong, Lin, Ni and Wang, Eq. (3.1), Algorithm 3.1][DLNW].

**Proof.** Write $U=F_0\cdots F_d$, where $F_0=e^{i\phi_0Z}$ and $F_j=W(x)e^{i\phi_jZ}$. Then

```math
\frac{\partial U}{\partial\phi_j}=(F_0\cdots F_{j-1})\,W(x)iZe^{i\phi_jZ}(F_{j+1}\cdots F_d)
```

for $j>0$, and $\partial U/\partial\phi_0=iZU$. Prefix and suffix products give every column at constant work per phase and node. Columns $j$ and $d-j$ add into one free column, with the center counted once.

Let $\alpha$ be the target coefficient vector, $\beta(\theta)$ the realized one, and $V$ the parity-restricted Chebyshev evaluation matrix at the distinct positive nodes $x_j=\cos((2j-1)\pi/(4M))$, $1\le j\le M$. The residual and Jacobian are $r=V(\beta-\alpha)$, $J=V\beta'$. The positive nodes are accompanied by their negatives through parity. Since $2M>d$, a parity polynomial vanishing at all these points is zero, so $V$ has full column rank. If $\beta'$ is invertible, $s=(\beta')^{-1}(\beta-\alpha)$ solves $Js=r$ exactly and is the unique least-squares step. Step halving until the sampled maximum residual decreases preserves improvement relative to the current iterate. It supplies no global convergence theorem near the QSP amplitude boundary.

<a id="r24"></a>
### Proposition 24. A finite Bessel-tail budget and explicit amplification residual

**Provenance: Derived in NWQLib. Evidence: proved conditional on the child and phase residual bounds.**

The Jacobi–Anger expansions of [Gilyén et al., Lemma 57, Eqs. (53)–(54)][GSLW] separate cosine and sine into even and odd Chebyshev polynomials. For $\tau>0$ and a terminal $T$ with $q=\tau/[2(T+2)]<1$, the uncomputed Bessel suffix satisfies

```math
\sum_{k>T}|J_k(\tau)|\le R_T:=\frac{(\tau/2)^{T+1}e^{\tau^2/[4(T+2)]}}{(T+1)!(1-q)}.
```

If the pre-amplification block differs from $aV$ by at most $e_B$, where $V$ is unitary and $0<a\le1/2$, the three-step oblivious amplitude-amplification block $3B-4BB^\dagger B$ differs from $(3a-4a^3)V$ by at most

```math
r_{\rm OAA}=3e_B(1+4a^2)+12ae_B^2+4e_B^3\le6e_B+6e_B^2+4e_B^3.
```

**Proof.** The Bessel power series and $(k+m)!/k!\ge(k+1)^m$ give $|J_k(\tau)|\le(\tau/2)^ke^{\tau^2/[4(k+1)]}/k!$. The ratio of consecutive majorants for $k>T$ is at most $q$, so their sum is bounded by $R_T$. This is conservative relative to the sharper real-argument inequality in Gilyén et al., Eq. (55). A finite suffix sum must include a bound on this infinite remainder, even when its floating-point value underflows.

For amplification, set $B=aV+D$, $\|D\|\le e_B$. Expanding $BB^\dagger B$ gives three linear terms of norm at most $a^2e_B$, three quadratic terms of norm at most $ae_B^2$, and one cubic term of norm at most $e_B^3$. Adding $3D$ proves the bound. For QSP evolution, one may take

```math
e_B=a\bigl(\epsilon_{\cos}+\epsilon_{\sin}+s(r_{\cos}+r_{\sin})\bigr)+|t|\epsilon_{\rm child},\qquad a=1/(2s),
```

provided target and encoded generators are Hermitian. Here $\epsilon_{\cos},\epsilon_{\sin}$ are truncation bounds and $r_{\cos},r_{\sin}$ are phase-fit residual bounds. The child term follows Gilyén et al., Lemma 61, and is conservatively unscaled by $a$. The raw block error relative to $V$ is at most $1-(3a-4a^3)+r_{\rm OAA}$. Recovery by division through $3a-4a^3$ has residual $r_{\rm OAA}/(3a-4a^3)$. These are different output comparisons.

## 8. Linear combinations of Hamiltonian simulation

<a id="r25"></a>
### Result 25. LCHS representation and the dissipativity premise

**Provenance: From a reference. Evidence: the cited representation theorem.**

Let $A=L+iH$, with $L,H$ Hermitian and $L\succeq0$. Linear combination of Hamiltonian simulation (LCHS) represents $e^{-AT}$ as an integral of unitary evolutions generated by $kL+H$. For the near-optimal kernel,

```math
e^{-AT}=\int_{\mathbb R}g_\beta(k)e^{-iT(kL+H)}\,dk,\qquad g_\beta(k)=\frac{e^{-(1+ik)^\beta}}{C_\beta(1-ik)},\qquad C_\beta=2\pi e^{-2^\beta},\quad0<\beta<1.
```

See [An, Childs and Lin (ACL), Eqs. (6)–(7), Theorem 6, Eq. (12), and Eq. (60)][ACL]. The Cauchy-kernel construction is their Eq. (5), and the implemented Low–Somma kernel is [Low and Somma, Eq. (6), with the parameter and trapezoid rules of Theorems 2–4][LowSomma]. Bounds belong to the selected kernel and quadrature pair.

If a shift $s\ge0$ makes $L+sI\succeq0$, the original solution is recovered by $e^{-AT}=e^{sT}e^{-(A+sI)T}$. A norm-relative floating-point PSD window is a numerical decision, not a proof of positive semidefiniteness. Error propagation must include the recovery factor.

<a id="r26"></a>
### Proposition 26. An integrated near-optimal-kernel tail bound

**Provenance: Improved from a reference. Evidence: proved from the pointwise kernel estimate.**

Let $c_\beta=\cos(\beta\pi/2)$ and $x=c_\beta K^\beta$. For every $K>0$,

```math
\left\|\int_{|k|>K}g_\beta(k)e^{-iT(kL+H)}\,dk\right\|\le\frac{2E_1(x)}{\beta C_\beta}\le\frac{2e^{-x}}{\beta C_\beta x}.
```

**Proof.** ACL Eqs. (185)–(186) bound the integrand norm by $e^{-c_\beta|k|^\beta}/(C_\beta|k|)$. The substitution $v=c_\beta k^\beta$ gives $dk/k=dv/(\beta v)$, hence the two-sided $E_1$ expression. In its integral, $1/v\le1/x$, so $E_1(x)\le e^{-x}/x$. For $K\ge1$, NWQLib takes the smaller of this bound and ACL Lemma 10, Eq. (62). The new expression is available also for $K<1$, where the latter proof's domain does not apply. Scalar bisection can select a cutoff from the bound. Logarithmic evaluation prevents premature overflow of factors, but ordinary binary64 evaluation does not certify outward rounding.

<a id="r27"></a>
### Proposition 27. Composite Gauss quadrature without a lower bound on dissipative time

**Provenance: Improved from a reference. Evidence: proved using an analytic-strip bound.**

Divide $[-K,K]$ into $2m$ panels of width $h=K/m$, each with $Q\ge2$ Gauss–Legendre nodes. Choose $0<d_0<1$, $b_0=1-d_0$, and $\rho=\exp(\operatorname{arsinh}(2b_0/h))$. The quadrature error for the integral in Result 25 is at most

```math
\epsilon_{\rm G}\le\frac{64K\,e^{b_0T\|L\|}}{15C_\beta d_0(\rho^2-1)\rho^{2(Q-1)}}.
```

This applies for all $T\ge0$. It replaces the panel prescription in ACL Lemma 11, Eqs. (64)–(65), whose stated domain includes $T\max\|L\|\ge32/e$.

**Proof.** In the strip $|\operatorname{Im}k|\le b_0$, both $1\pm ik$ have real part at least $d_0$. Thus $|1-ik|\ge d_0$ and $\operatorname{Re}(1+ik)^\beta\ge0$, giving $|g_\beta(k)|\le1/(C_\beta d_0)$. The Hermitian part of $-iT(kL+H)$ is $T\operatorname{Im}(k)L$, so its exponential norm is at most $e^{b_0T\|L\|}$, including when $L$ and $H$ do not commute. The Bernstein ellipse of each affine-mapped panel has semiminor axis $b_0$ when $\log\rho=\operatorname{arsinh}(2b_0/h)$. Apply [Trefethen, Theorem 19.3, Eq. (19.8)][ATAP] with $n=Q-1$, multiply by $h/2$, and sum the $2m$ panels. For the operator-valued integral, apply the scalar theorem to $v^\dagger F(k)u$, rotating its final error to the real axis if necessary, and take the supremum over unit $u,v$. This introduces no dimension factor.

For each $m$, solve the scalar inequality for the least $Q$ that satisfies it. Once $4m$ exceeds the best total node count, no later candidate can improve it because $2mQ\ge4m$. This proves the finite stopping rule for the selector's search over this panel family. It is not an optimality theorem over all quadratures.

<a id="r28"></a>
### Proposition 28. Constant-source Duhamel remainder in physical units

**Provenance: Improved from a reference. Evidence: proved for constant $A$ and source $b$.**

For $\dot x=-Ax+b$, write the source contribution as $\int_0^T e^{-A(T-s)}b\,ds$. An $m$-point Gauss–Legendre rule on $[0,T]$ has vector error at most

```math
\epsilon_{\rm D}\le\frac{T^{2m+1}(m!)^4}{(2m+1)((2m)!)^3}\|A\|^{2m}e^{T\max(0,-\lambda_{\min}(L))}\|b\|,\qquad L=(A+A^\dagger)/2.
```

**Proof.** Combine [DLMF Eqs. 3.5.19 and 3.5.21][DLMF] and the affine map from $[-1,1]$ to $[0,T]$. The derivative of order $2m$ has norm at most $\|A\|^{2m}\sup_{t\in[0,T]}\|e^{-At}\|\|b\|$. ACL Lemma 21, Eq. (162), bounds this semigroup by $e^{-t\lambda_{\min}(L)}$. For vector quadrature error $E\ne0$, apply the real scalar remainder theorem to $\operatorname{Re}[(E/\|E\|)^\dagger e^{-A(T-s)}b]$. Its quadrature error is $\|E\|$, proving the formula without a dimension factor or a shared complex mean-value point. The spectral endpoint is that of the original unshifted $L$, so the bound includes physical growth.

<a id="r29"></a>
### Proposition 29. Affine SELECT angles and source-time scaling

**Provenance: Derived in NWQLib. Evidence: proved for a common product schedule and affine address law.**

Suppose the quadrature address $b\in\{0,1\}^a$ has $k(b)=k_0+\sum_qk_qb_q$. For a Pauli term $P$, any product-formula slot with angle $\theta(b)=\theta_0+\sum_q\theta_qb_q$ is realized by one unconditional rotation and at most $a$ singly controlled rotations. For a joint QSP generator using reference time $T>0$, a branch with elapsed time $t_j$ must use diagonal weights $D_L(j)=(t_j/T)k_j$, $D_H(j)=t_j/T$.

**Proof.** All rotations about the same Pauli axis commute, so

```math
e^{-i\theta(b)P/2}=e^{-i\theta_0P/2}\prod_qe^{-ib_q\theta_qP/2}.
```

This is an exact affine-table factorization. It applies to a slot shared across the selected branches. Varying repetition counts require the construction's common repetition blocks, and arbitrary nonlinear angle tables do not meet the premise. For the QSP branch, $T(D_L(j)L+D_H(j)H)=t_j(k_jL+H)$. Omitting either elapsed-time factor simulates a different Duhamel branch. Complex quadrature phases remain in SELECT, with PREP amplitudes $\sqrt{|c_j|/\alpha}$, so the selected block is $\sum_jc_jU_j/\alpha$, $\alpha=\sum_j|c_j|$. This is a structured specialization of ACL Appendix A.3, Lemma 24, Eq. (178), and the effective-generator construction of [Pocrnic et al., Section IV, Eqs. (61)–(65)][Pocrnic].

<a id="r30"></a>
### Proposition 30. One physical vector budget for several output frames

**Provenance: Derived in NWQLib. Evidence: proved from norm inequalities.**

Suppose all required stage bounds have been converted into one physical vector error $\|x-y\|\le\delta$. Put $r=\|y\|$ and $w\ge\|O\|$. Then

```math
\bigl|\|x\|^2-\|y\|^2\bigr|\le\delta(2r+\delta),\qquad |x^\dagger Ox-y^\dagger Oy|\le w\delta(2r+\delta).
```

If $r>\delta$, then $x\ne0$, and

```math
\left\|\frac{x}{\|x\|}-\frac{y}{r}\right\|\le\frac{2\delta}{r},\qquad \left|\frac{x^\dagger Ox}{\|x\|^2}-\frac{y^\dagger Oy}{r^2}\right|\le\frac{4w\delta}{r}.
```

**Proof.** The reverse triangle inequality gives $|\|x\|-r|\le\delta$ and $\|x\|\le r+\delta$. Factor the difference of squared norms. Expanding the quadratic-form difference as $(x-y)^\dagger Ox+y^\dagger O(x-y)$ proves its bound. For normalization, add and subtract $x/r$. The two resulting distances are at most $\delta/r$ each. For unit vectors $u,v$, the same quadratic-form expansion bounds their expectation difference by $2w\|u-v\|$.

Every component must have the same physical frame before summation. A missing component prevents a complete bound. Applying these nonlinear frame conversions separately to each component and then adding them need not reproduce a bound for the combined error.

## 9. Block encodings and state preparation

<a id="r31"></a>
### Proposition 31. Singular-frame dilation and LCU normalization

**Provenance: Derived in NWQLib. Evidence: proved in exact arithmetic.**

Let $B=A/\alpha=W\Sigma V^\dagger$, $\|\Sigma\|\le1$, and $K=W\sqrt{I-\Sigma^2}V^\dagger$. The one-ancilla unitary

```math
\mathcal U=\begin{pmatrix}B&K\\K&-B\end{pmatrix}
```

block-encodes $A$ with normalization $\alpha$. The same singular frames support a positive dummy extension without a second singular-value decomposition.

**Proof.** Factor $\mathcal U=(I_2\otimes W)\begin{pmatrix}\Sigma&\sqrt{I-\Sigma^2}\\\sqrt{I-\Sigma^2}&-\Sigma\end{pmatrix}(I_2\otimes V^\dagger)$. The middle matrix is a direct sum of real orthogonal $2\times2$ matrices. Hence $\mathcal U$ is unitary. Appending dummy singular value one and identity frames gives a block $A\oplus\alpha I$. A right-hand side with zero dummy entries stays in its original coordinates. The smallest possible zero-error normalization is $\|A\|$, by the contraction property of a unitary block, as stated after [Gilyén et al., Definition 43][GSLW].

For $A=\sum_jc_jU_j$, prepare $\sum_j\sqrt{|c_j|/\alpha}|j\rangle$, apply SELECT with branch $e^{i\arg c_j}U_j$, and unprepare. Projection gives $\sum_jc_jU_j/\alpha$, where $\alpha=\sum_j|c_j|$. This standard LCU step is [Childs and Wiebe, Lemma 2 and Theorem 3][LCU], and [Low and Chuang, Lemma 5, Eq. (10)][Qubitization]. The derived part here is its explicit compatibility with the shared singular-frame completion and physical padding.

<a id="r32"></a>
### Proposition 32. Circulant normalization and a Pauli-domain certificate

**Provenance: Improved from a reference. Evidence: proved from Fourier and Pauli orthogonality.**

For a circulant $A=\sum_f\beta_fS^f$, the shift LCU has $\alpha=\sum_f|\beta_f|\le s\max_f|\beta_f|$, where $s$ is a padded band count. The right-hand side is the normalization obtained by scaling the sparse-access construction of [Camps et al., Theorem 4.1 and Section 4.2][Camps] to unit-magnitude entries. The comparison concerns normalization, not the total costs of the two access models.

Given only a finite Pauli table for $A$, define candidate bands from its first column, $\beta_f=A_{f0}$, and let $C=\sum_f\beta_fS^f$. If $a_P$ are the supplied coefficients and $c_P=2^{-n}\operatorname{Tr}(P^\dagger C)$, then

```math
\frac{\|A-C\|_F^2}{2^n}=\sum_{P\ {\rm supplied}}|a_P-c_P|^2+\sum_f|\beta_f|^2-\sum_{P\ {\rm supplied}}|c_P|^2.
```

**Proof.** The normalization inequality follows by bounding each band magnitude by their maximum. For $P=i^{\#Y}X^fZ^z$, $P|0\rangle=i^{\#Y}|f\rangle$, so the first-column bands are obtained without a dense matrix. Pauli orthogonality gives $\|M\|_F^2=2^n\sum_P|m_P|^2$. Distinct cyclic shifts are Frobenius-orthogonal, giving $\|C\|_F^2/2^n=\sum_f|\beta_f|^2$. Split the Pauli sum into supplied and absent labels to obtain the identity. Multiplying its right-hand side by $2^n$ and taking the square root gives $\|A-C\|_F$, which bounds the spectral error. NWQLib evaluates the shift traces with exact scalar arithmetic and uses zero certified mismatch for automatic circulant selection. The certificate covers absent Pauli labels, which a comparison limited to supplied coefficients would miss.

<a id="r33"></a>
### Proposition 33. TT-SVD discarded weight and normalized fidelity

**Provenance: Improved from a reference. Evidence: proved for exact successive truncated SVDs.**

Let $G$ be a normalized tensor and $T$ the raw tensor produced by a tensor-train SVD sweep. Let $D$ be the sum of squared singular values actually discarded at all stages. Then

```math
\|G-T\|^2=D,\qquad \langle G,T\rangle=\|T\|^2=1-D,\qquad \left|\left\langle G,\frac{T}{\|T\|}\right\rangle\right|^2=1-D\quad(D<1).
```

**Proof.** At a stage $A=UB+E$, the kept left singular vectors satisfy $U^\dagger U=I$, $U^\dagger E=0$. Every later approximation has form $UB'$. Therefore $\|A-UB'\|^2=\|E\|^2+\|B-B'\|^2$, and the earlier discarded component is orthogonal to the final tensor. Iterating proves both the total discarded norm and $\langle G-T,T\rangle=0$. The remaining equalities follow from $\|G\|=1$. This adds the norm and fidelity identities to the orthogonal-error argument of [Oseledets, proof of Theorem 2.2, Eq. (2.5), and Algorithm 1][TT].

The implemented rank rule uses a singular-value threshold and a maximum bond dimension, so $D$ is computed from actual truncations rather than inferred from the requested threshold. Floating-point SVD turns these equalities into estimates. A subsequent layered preparation circuit, based on [Ran, Section III, Eqs. (6)–(9)][Ran], has its own approximation error. The fidelity formula does not certify that circuit merely because it starts from these cores.

## 10. Circuit synthesis and product-formula budgets

<a id="r34"></a>
### Result 34. Multiplexed rotations, phase diagonals, and preparation trees

**Provenance: From a reference. Evidence: the cited exact gate decompositions.**

A uniformly controlled rotation with $a\ge1$ address qubits uses $2^a$ CX gates in its Gray-code decomposition. A phase diagonal on $n$ qubits uses $2^n-2$ CX gates and $2^n-1$ $R_z$ rotations, apart from its global phase. [Shende, Bullock and Markov, Theorems 7–8][SBM], and [Möttönen et al., Section II, Fig. 2, Eq. (3)][Mottonen]. The magnitude tree for state preparation follows Möttönen et al., Section III, Eq. (8). It also sums to $2^n-2$ CX gates, before adding a phase diagonal for complex amplitudes. Removing an address bit on which the whole unitary table is exactly independent is a multiplexor identity. Affine angle tables have the stronger linear-size factorization of Result 29.

<a id="r35"></a>
### Proposition 35. Gate-count bounds for exact dense synthesis and their control cost {#proposition-35-exact-dense-synthesis-envelopes-and-their-control-cost}

**Provenance: Improved from a reference. Evidence: proved for the recursive gate template, conditional on accurate numerical factorizations.**

The implementation combines two-qubit KAK synthesis with the block-ZXZ recursion of [Krol and Al-Ars, Sections 4–5, Eqs. (5)–(11)][ZXZ]. It uses the diagonal transfer of [Shende, Bullock and Markov, Appendix A.2][SBM] when that transfer is numerically resolved. An arbitrary $n\ge2$ qubit input therefore has the conservative gate-count bounds

```math
C_n=\frac{25}{48}4^n-\frac32\,2^n+\frac23,\quad U_n=7\,4^{n-2},\quad Z_n=\frac38\,4^n-\frac32\,2^n,\quad H_n=\frac{4^n-16}{24}.
```

A matrix block diagonal in a top qubit, including a controlled dense unitary, has for total width $m\ge3$

```math
C_m^{\rm bd}\le\frac{25}{96}4^m-2^m+\frac43.
```

**Proof.** The base two-qubit template uses at most three CX gates and seven one-qubit $U$ gates. A generic $n$-qubit level has four children, three multiplexed $R_z$ blocks, and two Hadamards. Two closing CX gates are absorbed into the central block. Thus

```math
C_n=4C_{n-1}+3\,2^{n-1}-2,\quad Z_n=4Z_{n-1}+3\,2^{n-1},\quad H_n=4H_{n-1}+2,
```

with $C_2=3$, $Z_2=H_2=0$. Solving the recurrences gives the formulas. A top demultiplexing has two $(m-1)$-qubit children and one $2^{m-1}$-CX multiplexor, so $C_m^{\rm bd}\le2C_{m-1}+2^{m-1}$. The smaller generic count $22\,4^n/48-3\,2^n/2+5/3$ of Krol and Al-Ars assumes all applicable diagonal transfers succeed. The conservative bounds remain valid when a nearly singular transfer is skipped.

The same recursion gives cubic classical factorization work and quadratic array storage in matrix dimension $M$. At depth $j$, there are $4^j$ blocks of side $M/2^j$, so cubic work sums as $M^3\sum_j2^{-j}$, and the number of entries at any one depth is $M^2$. Additional $k$ controls handled by whole-matrix synthesis replace $M$ by $2^kM$, multiplying these leading work and storage terms by $8^k$ and $4^k$. Gate-wise control has a different cost derived from the gate count. These are representation-dependent laws, not a proof of CPU time or process memory.

The algebraic synthesis is exact. Its implementation uses floating-point factorizations and small rounding-window replacements, so a uniform certified matrix-to-circuit error bound remains a separate obligation.

<a id="r36"></a>
### Proposition 36. Low-order product bounds with exact integer step selection

**Provenance: Improved from a reference. Evidence: proved from the cited bounds and exact arithmetic.**

Let $H=\sum_jH_j$ be Hermitian, ordered so $H_1$ acts first, and $R_j=\sum_{k>j}H_k$. Define

```math
B_1=\frac{|t|^2}{2}\sum_j\|[R_j,H_j]\|,\qquad B_2=|t|^3\left(\frac1{12}\sum_j\|[R_j,[R_j,H_j]]\|+\frac1{24}\sum_j\|[H_j,[H_j,R_j]]\|\right).
```

The $r$-step first- and second-order errors are bounded by $B_1/r$ and $B_2/r^2$, respectively. These are [Childs et al., Propositions 9–10, Eqs. (120)–(121)][Trotter] in the journal numbering, or arXiv:1912.08854v3 Propositions 15–16, Eqs. (145), (152). The smallest step count allowed by these bounds is

```math
r_1=\max(1,\lceil B_1/\epsilon\rceil),\qquad r_2=\max(1,\lceil\sqrt{\lceil B_2/\epsilon\rceil}\rceil).
```

**Derivation.** Telescoping $r$ unitary steps gives $r$ times the bound at duration $t/r$. For an integer $r$, $r^2\ge B_2/\epsilon$ is equivalent to $r^2\ge\lceil B_2/\epsilon\rceil$. Exact rational division followed by integer square-root and ceiling operations therefore avoids a downward-rounded step count. The optimum is relative to this sufficient bound, not to the unknown true error.

**Pauli-triangle bounds and a quadratic relaxation.** Let $H=\sum_{i=0}^{p-1}c_iP_i$ be an ordered sum of Hermitian Pauli strings with real coefficients, and let $a_i=|c_i|$. Define $A=\{(i,j):i<j,\ P_iP_j=-P_jP_i\}$ and $T=\{(i,j,k):(i,j)\in A,\ k>i,\ P_k(P_iP_j)=-(P_iP_j)P_k\}$. The range of T includes $k=j$. Write $S_i=\sum_{k>i}a_k$. The first-order Pauli-triangle coefficient is $W_1=\sum_Aa_i a_j$. At order two,

```math
W_2=\frac13\sum_Ta_i a_j a_k+\frac16\sum_Aa_i^2a_j
\le W_{2,\mathrm{rel}}=\frac13\sum_Aa_i a_jS_i+\frac16\sum_Aa_i^2a_j.
```

Either second-order coefficient bounds the error of r symmetric steps by $W|t|^3/r^2$, under the ordering and Hermiticity premises of [Childs et al., Eq. (121)][Trotter]. The relaxed expression uses pair tests and suffix sums, so a fresh evaluation takes $O(p^2\lceil q/64\rceil)$ work on packed q-qubit labels. The full triple structure can take cubic work.

**Proof.** A nonzero Pauli commutator has norm two and a surviving nested commutator has norm four. Expansion of the tail sums gives $C_{12}^{\triangle}=4\sum_Ta_i a_j a_k$ and $C_{24}^{\triangle}=4\sum_Aa_i^2a_j$. Replacing the nested anticommutation indicator by one increases a sum of nonnegative terms and yields the stated relaxation. Unitary telescoping gives the factor $r^{-2}$. For the same positive time and allowance, define $r_{\mathrm{exact}}$ and $r_{\mathrm{rel}}$ by exact inversion of $W_2$ and $W_{2,\mathrm{rel}}$, respectively. If $W_2>0$ and $\alpha=\sqrt{W_{2,\mathrm{rel}}/W_2}$, these integers satisfy $r_{\mathrm{exact}}\le r_{\mathrm{rel}}\le\lceil\alpha r_{\mathrm{exact}}\rceil$. When the coefficient is zero the selected count is one. This comparison concerns the mathematical coefficients before outward numerical evaluation.

The production coefficient is evaluated with scaled binary64 upper products and sums, followed by rational rescaling. Its value $W_{\mathrm{up}}$ bounds the selected triangle expression. For formula order $o\in\{1,2\}$, time multiplication, division by $r^o$ and integer step inversion use exact rationals formed from $W_{\mathrm{up}}$ and the stored binary64 time. Published bounds are rounded upward. This preserves a positive bound below the binary64 range and reports a range error above it. The selected count is minimal for $W_{\mathrm{up}}$, which can exceed the exact triangle coefficient. The record names the full or relaxed expression and its coefficient arithmetic. Pruning consumes its allowance before step selection, and a zero remaining allowance permits only a zero coefficient.

For a common-grid QPE trajectory, nonnegative integer power $k$ targets $t_k=k\,\operatorname{val}(\tau)$, where $\operatorname{val}(\tau)$ is the exact real value of the stored binary64 time unit. Write $W$ for an upper bound on the second-order coefficient of the kept nonidentity generator, $C$ for its coefficient L1 mass, and $d_{\mathrm{drop}}$ for the dropped coefficient L1 mass. If the emitted binary64 step has exact real value $h$ and is repeated $r_k$ times, its product-formula bound is $W r_k|h|^3$. The time-displacement bound is $C|r_kh-t_k|$, and pruning contributes $d_{\mathrm{drop}}t_k$. These terms and the applicable identity-phase and rotation-angle formation bounds must fit the allowance at every power position. The shared grid gives sufficient prefix counts and does not assert an independently smallest count at each time.

Let $S_2(h)$ be one fixed ordered second-order step with one-step operator bound $W|h|^3$. At an integer step boundary $r_p=t_p/h$, telescoping gives $\|S_2(h)^{r_p}-e^{-it_pH}\|\le Wh^2t_p$. If every power has the same absolute allowance, satisfying that allowance at the largest nonnegative time suffices for every earlier boundary. For allowances $\epsilon_p$, the step must satisfy $Wh^2\le\min_{p>0}\epsilon_p/t_p$. Pruning consumes $t_p$ times the dropped coefficient L1 mass before this comparison. A wholly zero-power schedule needs no evolution step selection. Every queried time must be represented by an integer step count, and finite representation of the step and rotation angles has separate numerical error.

For periodic LCHS, write the diffusion coefficient as $\zeta\ge0$ and the coefficient of $Z_0$ as $v$. At quadrature node $k_j$, the two matching terms have norm $|k_j|\zeta$ each, including the coincident-neighbor case on two sites, and the potential term has norm $|v|$. The identity term is applied separately. Set

```math
\Lambda_j=2|k_j|\zeta+|v|,\qquad B_{\rm per}=\frac{T^3}{3}\sum_j\overline c_j\Lambda_j^3,\qquad \overline c_j\ge|c_j|.
```

Then the weighted $r$-step Strang error is at most $B_{\rm per}/r^2$. To derive this, let $a_i$ be the norms of the ordered terms of a branch. Replacing each nested commutator by four times the product of its term norms gives the coefficient

```math
\frac13\sum_{i\lt j}a_i a_j^2+\frac16\sum_{i\lt j}a_i^2a_j+\frac23\sum_{i\lt j\lt k}a_i a_j a_k\le\frac13\left(\sum_i a_i\right)^3.
```

Unitary telescoping gives the $r^{-2}$ factor, and the triangle inequality for $\sum_jc_jU_j$ gives the weighted sum. The implementation constructs rational upper magnitudes $\overline c_j$ by outward scaled square-root evaluation, then evaluates $B_{\rm per}$ exactly from these values and the stored inputs. A tolerance selects $r=\max(1,\lceil\sqrt{\lceil B_{\rm per}/\epsilon\rceil}\rceil)$ before construction limits are applied. With a supplied fixed step count, a bound exceeding binary64 is reported as unavailable. These bounds concern the selected product formulas, with coefficient, gate-formation, and execution errors accounted for separately.

## 11. Quantum Hamiltonian descent

Quantum Hamiltonian descent (QHD) evolves a finite-grid state under $H(t)=a(t)T+b(t)V$, where $T$ is a kinetic operator and $V$ is a tabulated objective. A time-evolution approximation, a probability readout, and an optimization statement are distinct mathematical objects. The following results connect these objects only where their premises justify it.

For the range-checked arithmetic in QHD planning, put $\nu_{\rm fp}=2^{-1022}$, $\lambda=2^{-1074}$, and let $\Omega$ be the largest finite binary64 number. A nonzero multiplicative result must have exact magnitude in $[\nu_{\rm fp},\Omega]$, reading its stored operands as exact reals. A rounded result equal to $\nu_{\rm fp}$ or $\Omega$ needs an exact boundary check. A zero is accepted as exact only when its operands or the mathematical construction establish it. Under round-to-nearest arithmetic these conditions justify the relative-error model, and accepted power-of-two scalings are exact. Finite addition and subtraction may produce exact subnormal scratch values under gradual underflow. Positive error terms are evaluated exactly or outward, so a positive term below $\lambda$ is reported as $\lambda$.

Stored objective tables need only be finite and may contain subnormal entries. Compiled potential and binary phase constructions can omit a contribution whose required arithmetic falls below the normal range, with the error terms of Propositions 42 and 44. Structured one-hot preparation can stop at a lower-range link, with the state-error term of Proposition 53. Kinetic coefficients, energies, and one-hot kinetic terms outside the accepted range are refused, while phase products of accepted binary kinetic tables can be omitted and counted as error terms. Required nonfinite or overflowing arithmetic is refused. The direct classical kernels use the original stored tables and complete initial vectors. These construction checks supply range premises for the error budgets, while elementary-function and transform accuracy remain separate assumptions.

<a id="r37"></a>
### Result 37. Grid Hamiltonians and boundary-dependent spectra

**Provenance: From a reference. Evidence: the cited discretization and elementary stencil spectra.**

The time-dependent model is [Leng et al., Eq. (1)][Leng]. Their Eqs. (F.7), (F.9), and the text after Eq. (F.4) specify the central-difference kinetic and grid potential with vanishing boundary values. For $K$ represented sites and spacing $h$, the Dirichlet kinetic eigenvalues and vectors are

```math
E_r=\frac{2}{h^2}\sin^2\!\frac{\pi r}{2(K+1)},\qquad v_r(j)\propto\sin\frac{\pi r(j+1)}{K+1},\qquad1\le r\le K.
```

The periodic endpoint-exclusive grid has $E_k=2h^{-2}\sin^2(\pi k/K)$. Its one-hot hopping includes the wrap link of [Liu et al., Eq. (12)][Liu]. On the one-excitation sector, a link $-(XX+YY)/(4h^2)$ gives matrix element $-1/(2h^2)$, and the scalar stencil diagonal is $1/h^2$ per variable. One-hot occupation operators $n_j=(I-Z_j)/2$ encode a potential table as in [Wu et al., Eqs. (9)–(10)][Wu]. Products of these projectors encode support-local tables.

The periodic uniform state is a zero-momentum eigenstate. The Dirichlet ground state is the $r=1$ sine vector, so a uniform Dirichlet start has excited kinetic components. These spectra follow by substitution into the tridiagonal or circulant stencil. An even periodic cycle admits two disjoint matching layers. The current one-hot two-layer implementation requires even $K\ge4$, while the binary periodic implementation also covers $K=2$ with coincident neighbors combined.

<a id="r38"></a>
### Proposition 38. Stable interval integrals for the three schedules

**Provenance: Derived in NWQLib. Evidence: proved identities and an analytic quadrature remainder.**

For $0\le t_0<t_1$, let $h=t_1-t_0$. The quadratic schedule is $a(t)=(1+\gamma t^2)^{-1}$, $b(t)=1+\gamma t^2$, as recommended in [QHDOPT, Section 2.1][QHDOPT]. If $\gamma>0$, set $d=1+\gamma t_0t_1$, $z=\sqrt\gamma h/d$. Then

```math
\int_{t_0}^{t_1}a(t)\,dt=\frac hd\,\frac{\arctan z}{z},\qquad \int_{t_0}^{t_1}b(t)\,dt=h+\frac{\gamma h}{3}(t_1^2+t_0t_1+t_0^2).
```

The first expression has continuous value $h/d$ at $z=0$, and both integrals equal $h$ when $\gamma=0$. Both cubic schedules below require $s>0$. The shifted-cubic schedule of Wu et al., Eq. (15), has $a(t)=8/(s+t)^3$, $b(t)=2t^3$, giving

```math
\int a=\frac{4h(1+x_0/x_1)}{x_0^2x_1},\quad x_i=s+t_i,\qquad \int b=\frac h2(t_1+t_0)(t_1^2+t_0^2).
```

**Proof.** Integrate the rational functions directly. For the quadratic kinetic term, use $\arctan x-\arctan y=\arctan((x-y)/(1+xy))$ for $x\ge y\ge0$. Factor the differences of cubes, inverse squares, and fourth powers to obtain the displayed positive expressions. Keeping $h$ as a factor avoids subtracting nearly equal primitive values.

For Leng et al.'s cubic schedule $a(t)=2/(s+t^3)$, Eq. (C.4), let $r=s^{1/3}$. Below $r$, substitute $y=t/r$, giving $2r^{-2}\,dy/(1+y^3)$. Above $r$, substitute $x=r/t$, giving $-2r^{-2}x\,dx/(1+x^3)$. Both integrations are positive and lie in $[0,1]$. The implementation uses four 16-point Gauss panels for each transformed interval. To bound truncation, the poles of $f(z)=1/(1+z^3)$ have distance at least $R=\sqrt3/2$ from the real interval. For a panel of half-width $L\le1/8$, put $q=L/R$. Expanding the three reciprocal factors about its midpoint bounds the $k$-th Taylor coefficient by $f(m)\binom{k+2}{2}R^{-k}$. Define

```math
T_n(q)=\frac{q^n}{2}\left(\frac{(n+1)(n+2)}{1-q}+\frac{(2n+3)q}{(1-q)^2}+\frac{q(1+q)}{(1-q)^3}\right).
```

Exactness through degree 31 and positive quadrature weights bound the relative panel error for $f$ and $zf$ by $2(1+q)^3(T_{32}(q)+T_{31}(q))<1.9\times10^{-23}$, using $q\le1/(4\sqrt3)$. For $zf$, its midpoint satisfies $m\ge L$, which bounds the additional degree-31 tail. Positive panel sums preserve the relative bound.

Floating-point evaluation dominates this analytic truncation. The implementation's first-order kinetic-integral allowances are $32u$ for the quadratic schedule and $48u$ for the cubic schedule, under its elementary-function accuracy and scaled-arithmetic assumptions. For the quadratic formula, formation errors in $d$ and $z$ are at most $3u$ and $7u$. The logarithmic derivative of $\arctan z/z$ lies in $[-1,0]$. Its evaluation and division add $3u$, giving $10u$ for that factor and below $20u$ for the ordinary expression. The exponent-scaled branch and an underflowed intermediate in $d$ raise the allowance to $32u$. For the cubic formula, the computed $r$ defines $s'=r^3$ within $6u$ of $s$, and the logarithmic sensitivity to $s$ is at most one. Transformed-coordinate errors are at most $10u$ or $12u$ times the interval midpoint. The bounds $f\ge1/2$, $|f'|\le1$, and $|(zf)'|\le1$ give at most $24u$ relative error in either mean. Cubing, division, weights, and compensated summation add $7u$. The outer scaled products, segment formation, and the change in $s$ add at most $14u$, for $45u<48u$. These are first-order allowances, with a separate absolute subnormal error term. The cubic qualification is restricted to $s\in[10^{-8},10^8]$, $t\le10^4$.

<a id="r39"></a>
### Proposition 39. Finite time-ordering and coefficient-quadrature bounds

**Provenance: Derived in NWQLib. Evidence: proved for finite Hermitian generators, nonnegative scalar schedules for the integral bound, and differentiable scalar schedules for the derivative bound.**

On a step $[\ell,r]$ of length $\Delta$, let $A=\int a$, $B=\int b$, $M=e^{-i(AT+BV)}$, and $U$ be the exact time-ordered propagator.

For each time step the time-ordering allowance uses the smaller of an available derivative bound and an available integral bound. For $H(t)=a(t)T+b(t)V$ with finite Hermitian $T,V$, nonnegative $a,b$, step width $\Delta$ and a commutator bound $C\ge\|[T,V]\|$, write $A=\int a$ and $B=\int b$ over that step. The integral allowance is

```math
 w_{\rm int}=\min\{2,CAB/2\}.
```

If $a_*$ and $b_*$ bound the coefficients and $A_1$ and $B_1$ bound their first derivatives on the step, the derivative allowance is

```math
 w_{\rm der}=\Delta^3(a_*B_1+b_*A_1)C/12.
```

The code takes the per-step minimum of the available bounds, accumulates it with outward rounding and applies the unitary-distance cap of 2. Exact step integrals are used for the shifted cubic schedule and for the quadratic schedule with $\gamma=0$. Other schedules use the existing derivative bound. An unavailable commutator bound remains unavailable.

If $\mu_T\ge\|T\|$, $\mu_V\ge\|V\|$, replacing the integrals by midpoint values adds at most

```math
\frac{\Delta^3}{24}(A_2\mu_T+B_2\mu_V),\qquad A_2=\sup|a''|,\quad B_2=\sup|b''|.
```

**Proof.** Put $Z(t)=-i\int_\ell^tH(v)\,dv$, $W(t)=e^{Z(t)}$. Its right logarithmic derivative is $\int_0^1e^{sZ}Z'e^{-sZ}\,ds$. Subtracting $Z'$ and integrating the commutator derivative bounds the difference by $\|[Z,Z']\|/2$. Duhamel's identity then gives

```math
\|U-M\|\le\frac12\int_\ell^rdu\int_\ell^u dv\,\|[H(u),H(v)]\|.
```

Now $[H(u),H(v)]=(a(u)b(v)-b(u)a(v))[T,V]$. The scalar difference is bounded by $(a_*B_1+b_*A_1)|u-v|$, and its triangular integral is $\Delta^3/6$. The midpoint result follows from the scalar midpoint remainder and $\|e^{-iX}-e^{-iY}\|\le\|X-Y\|$ for Hermitian $X,Y$.

The commutator is $(a(u)b(v)-b(u)a(v))[T,V]$. Nonnegativity implies that its norm is at most $C(a(u)b(v)+b(u)a(v))$. This majorant is symmetric in $u,v$. Its integral over the whole square is $2AB$, and over the triangular half is $AB$. The factor $1/2$ gives $CAB/2$. Two unitaries are at distance at most 2.

Each is an upper bound for the same local error, so their minimum is also an upper bound. Telescoping products of unitaries adds local errors, which justifies the sum of per-step minima.

Stored exponents $\widehat A,\widehat B$ add a separate term $|\widehat A-A|\mu_T+|\widehat B-B|\mu_V$. Raw norms are required here because the identity parts carry physical phase. Exact integrated coefficients remove midpoint quadrature error, but generally leave time-ordering error.

### Derivation: QHD schedule values, integrals and derivative bounds {#qhd-schedule-arithmetic}

The schedule records in `src/nwqlib/algorithms/qhd/schedules.py` give the point values, step integrals and derivative bounds that planning and `evolution_bound` use, with their rounding in units of $u=2^{-53}$. [Proposition 38](#r38) states the step integrals and [Proposition 39](#r39) the bounds that use them. The statements below are those of the methods' docstrings, with the notation of these propositions.

**Point values.** Each point-value allowance bounds the relative error $|\hat f-f|/f$ of a computed weight $\hat f$ against its exact value $f$, to first order in $u$. For the quadratic schedule, $b(t)=1+\gamma t^2$ is formed as $(\gamma t)t$ with two roundings, and adding the positive 1 makes a third, so the relative error of $b$ is at most $3u$. It is at least 1 and fails the range only by overflowing, for which the refusal names a shorter `total_time`. $a(t)=1/(1+\gamma t^2)$ is the reciprocal of $b$, one more rounding, so its relative error is at most $4u$, and it must be a positive normal binary64 number. That fails once $b(t)$ exceeds about $4.5\times10^{307}$, and the refusal names a shorter `total_time` as the remedy (`QuadraticSchedule.kinetic_weight`, `potential_weight`).

For the cubic schedule, $a(t)=2/(s+t^3)$ forms $t^3$ as $(tt)t$ with two roundings, and the positive sum and the division add one each, so the relative error of $a$ is at most $4u$. For the shifted cubic schedule, $a(t)=(2/(s+t))^3$ forms $x=2/(s+t)$ with two roundings, which cubing triples, and the two products add two more, so the relative error of $a$ is at most $8u$. In both, $a$ is positive for every $t$, so a result that is zero or subnormal lies outside the normal binary64 range and raises, for example the cubic $a(10^{103})$ with $s=1$, whose exact value is about $2\times10^{-309}$. A lower-range refusal recommends reducing $s$ and/or `total_time`, since both may need to change. An upper-range refusal recommends a larger $s$. Both cubic schedules have $b(t)=2t^3$. Doubling is exact, and $(2tt)t$ has two roundings, so the relative error of $b$ is at most $2u$. It is zero only at $t=0$, so another zero or a subnormal value raises. The refusal names fewer steps or a longer `total_time` below the range, and a shorter `total_time` above it (`kinetic_weight`, `potential_weight` of `CubicSchedule` and `ShiftedCubicSchedule`).

**Step integrals.** The computed value $\hat I$ of each step integral $I$ satisfies $|\hat I-I|\le c\,u\,I+2\lambda$ to first order in $u$, with $\lambda=2^{-1074}$ the smallest subnormal. The term $2\lambda$ covers the rounding of the positive parts and of their sum when $I$ is subnormal, where no binary64 result can have a small relative error, and below $\lambda/2$ zero is the correctly rounded result. On $\gamma,s\in[10^{-8},10^8]$ and $0\le t\le10^4$, the constant $c$ is 20 for the quadratic $A$ with $\gamma>0$ (32 over the whole finite binary64 range), 1 for the quadratic $A$ and $B$ with $\gamma=0$, 16 for the quadratic $B$, 48 for the cubic $A$, and 16 for the cubic $B$ and both shifted-cubic integrals. These constants assume round-to-nearest binary64 with gradual underflow, correctly rounded basic operations and square root, at most one ulp of error in $\arctan$, $\operatorname{atan2}$ and the cube root, and a relative error of at most $2u$ for `fsum` of nonnegative terms. Python's `math` module takes these functions from the platform C library, so they are platform assumptions (module docstring of `schedules.py`).

On $[t_0,t_1]$ with $h=t_1-t_0$, the quadratic kinetic integral of Proposition 38 is $(h/d)\operatorname{atanc}(z)$ with $\operatorname{atanc}(z)=\arctan(z)/z$. For $z\le2^{-27}$ the computation replaces $\operatorname{atanc}(z)$ by 1, an approximation with relative error at most $z^2/3\le u/6$. Keeping $h$ as a factor preserves a short interval whose two angles round to the same value. When $d$ or $\sqrt\gamma\,h$ overflows, the exponent-scaled branch carries both as mantissa and exponent and forms the angle as $\operatorname{atan2}(\sqrt\gamma\,h,d)$ on operands scaled by one common power of two, and it returns $h/d$ when the exponents show $z\le2^{-27}$. This branch has $c=32$, which also covers a product $\gamma t_0$ that underflows before its multiplication by a very large $t_1$, because the lost term changes $d$ by less than $4u$ (`QuadraticSchedule.kinetic_integral`).

With $q=t_0/t_1$, the quadratic potential integral's cubic difference factors as $\gamma ht_1^2(1+q+q^2)/3$, positive like $h$. The sensitivity of $1+q+q^2$ to $q$ is at most 2, so the polynomial carries at most $5u$, and with the mantissa operations and the positive `fsum` the constant is $c=16$. A $q$ or $q^2$ that underflows is negligible against the constant 1 of the polynomial (`QuadraticSchedule.potential_integral`). Both cubic schedules form $B=(t_1^4-t_0^4)/2$ as $ht_1^3(1+q)(1+q^2)/2$, a product of positive factors, so no difference of fourth powers is formed, and $c=16$ (`potential_integral` of both cubic schedules).

For the shifted cubic schedule, $x_0=s+t_0$ and $x_1=s+t_1$ have the exact difference $h$ even when the rounded sums are equal, and $A=4h(1+x_0/x_1)/(x_0^2x_1)$ forms no difference of inverse squares. The logarithmic sensitivities to $x_0$ and $x_1$ are at most 2 and $3/2$, so the two rounded sums contribute $3.5u$, and $c=16$. On $[0,s]$, $A$ is exactly $3/s^2$. When $s+t_1$ overflows, $s$ exceeds $2^{969}$ and the whole integral is at most $4/s^2<2^{-1936}$, below the subnormal range, so the result is zero (`ShiftedCubicSchedule.kinetic_integral`).

For the cubic schedule, the partial-fraction primitive in $y=t/s^{1/3}$,

```math
\frac{\log(1+y)}{3}-\frac{\log(y^2-y+1)}{6}+\frac{1}{\sqrt3}\arctan\frac{2y-1}{\sqrt3},
```

cancels catastrophically for large $y$, where its terms are much larger than the integral, of order $h/y^3$. The quadrature of Proposition 38 therefore works on positive integrands only. With $r=s^{1/3}$, on a segment below $r$ of length $h$ starting at $t_0$, $y=t_0/r+(h/r)q$ gives $A_{\rm low}=(2h/r^3)\operatorname{mean}_q 1/(1+y^3)$. On a segment $[p,v]$ above $r$ of length $h$, $x=r/t$ maps it to $[r/v,r/p]$, whose length $(r/p)(h/v)$ is formed without subtracting $r/p-r/v$, so $A_{\rm high}=(2h/(rpv))\operatorname{mean}_q x/(1+x^3)$ with $x=r/v+(r/p)(h/v)q$. Both means are over $q\in[0,1]$, each by the 16-point Gauss–Legendre rule on four equal panels, and carry at most $31u$. Each interval costs 64 or 128 integrand evaluations, however close its endpoints or small $s$. Outside $s\in[10^{-8},10^8]$ and $t\le10^4$, a transformed coordinate can underflow only where the tail it describes is itself below the smallest subnormal $2^{-1074}$ or negligible in a bounded mean. For example $r/p<2^{-1022}$ needs $p>2^{663}$ even for the smallest $s$, and the whole remaining tail is then at most $\int_p^\infty 2t^{-3}\,dt=p^{-2}<2^{-1326}$ (`CubicSchedule.kinetic_integral`).

**Exact rationals.** The coefficient residual of `evolution_bound` compares the stored midpoint weights with the exact rational values $(a(t),b(t))$ at an exact rational $t\ge0$, so under the midpoint rule it needs no roundoff estimate. Under the integrated rule it compares the stored exponents $\widehat A=\Delta\hat a$ and $\widehat B=\Delta\hat b$, the exact products of the stored step duration $\Delta$ and the stored step averages $\hat a,\hat b$, with the nominal integrals $A$ and $B$ over the step $[\ell,r]$ of Proposition 39, exactly where those integrals are rational. From here on, $r$ is the right end of the step, not the cube root above. $B$ is rational for every schedule, $(r-\ell)+\gamma(r^3-\ell^3)/3$ for the quadratic schedule and $(r^4-\ell^4)/2$ for both cubic schedules. $A$ is rational for the quadratic schedule with $\gamma=0$, where $A=r-\ell$, and for the shifted cubic schedule, where $A=4((s+\ell)^{-2}-(s+r)^{-2})$ and the exact subtraction has no cancellation error. For $\gamma>0$, $A$ is an arctangent difference, and for the cubic schedule it mixes a logarithm and an arctangent. There the residual uses the first-order relative error constant of the kinetic integral, 32 over the whole finite binary64 range for the quadratic schedule and 48 on $s\in[10^{-8},10^8]$ and $0\le t\le10^4$ for the cubic schedule (`exact_weights`, `exact_integrals`).

**Derivative bounds.** On a step $[\ell,r]$, with exact rationals $0\le\ell<r$ and with $\gamma$ or $s$ read as the exact rational of its binary64 value, each schedule returns exact bounds $(a_*,b_*,A_1,A_2,B_1,B_2)$ on $|a|$, $|b|$ and their first two derivatives. These are analytic bounds over the whole interval, not sampled maxima, and `evolution_bound` uses them in the time-ordering and midpoint-quadrature bounds of Proposition 39 (`derivative_bounds`).

For the quadratic schedule, with $D=1+\gamma\ell^2$, $a$ decreases and $b$ increases on $t\ge0$, so $a_*=a(\ell)=1/D$ and $b_*=b(r)$. From

```math
a'=-\frac{2\gamma t}{(1+\gamma t^2)^2},\qquad a''=\frac{2\gamma(3\gamma t^2-1)}{(1+\gamma t^2)^3},
```

bounding each numerator at $r$ and each denominator at $\ell$ gives $A_1=2\gamma r/D^2$ and $|a''|\le2\gamma(1+3\gamma r^2)/D^3$, which $A_2=2\gamma/D^2+8\gamma^2r^2/D^3$ covers because $D\ge1$. $b'=2\gamma t$ and $b''=2\gamma$ give $B_1=2\gamma r$ and $B_2=2\gamma$.

For the cubic schedule, with $D=s+\ell^3$, $a=2/(s+t^3)$ decreases and $b=2t^3$ increases, so $a_*=2/D$ and $b_*=2r^3$. $a'=-6t^2/(s+t^3)^2$ gives $A_1=6r^2/D^2$. Since $|2t^3-s|\le(s+t^3)+t^3$,

```math
a''=\frac{12t(2t^3-s)}{(s+t^3)^3},\qquad |a''|\le\frac{12t}{(s+t^3)^2}+\frac{12t^4}{(s+t^3)^3},
```

which $A_2=12r/D^2+36r^4/D^3$ covers. $b'=6t^2$ and $b''=12t$ give $B_1=6r^2$ and $B_2=12r$. Numerators are bounded at $r$ and denominators at $\ell$.

For the shifted cubic schedule, $a=8/(s+t)^3$ decreases with $a'=-24/(s+t)^4$ and $a''=96/(s+t)^5$, so $a_*=8/(s+\ell)^3$, $A_1=24/(s+\ell)^4$ and $A_2=96/(s+\ell)^5$. $b=2t^3$ gives $b_*=2r^3$, $B_1=6r^2$ and $B_2=12r$.

<a id="r40"></a>
### Proposition 40. Product bounds for the actual one-hot and binary steps

**Provenance: Improved from a reference. Evidence: proved by applying Result 36 to the emitted groups.**

Let $C,D_T,D_V$ bound $\|[T,V]\|$, $\|[T,[T,V]]\|$, and $\|[V,[V,T]]\|$. For stored step exponents $\alpha,\beta$, the binary first-order and symmetric second-order steps have errors at most

```math
s_{\rm bin,1}=\tfrac12|\alpha\beta|C,\qquad s_{\rm bin,2}=\frac{\alpha^2|\beta|D_T}{12}+\frac{|\alpha|\beta^2D_V}{24}.
```

For one-hot kinetic link weights $w_j=1/(2h_j^2)$, first-order sequential links add $\alpha^2\Gamma/2$, where

```math
\Gamma=(K-2)\sum_jw_j^2\quad\hbox{on a chain},\qquad \Gamma=(K-1)\sum_jw_j^2\quad\hbox{on an even cycle}.
```

The symmetric one-hot step is $e^{-i\beta V/2}e^{-i\alpha O/2}e^{-i\alpha E}e^{-i\alpha O/2}e^{-i\beta V/2}$, where $O,E$ are the odd and even matching sums. It adds

```math
|\alpha|^3(J_E/12+J_O/24),\quad J_E=r_E\sum_jw_j^3,\quad J_O=r_O\sum_jw_j^3,
```

with $r_E,r_O\le4$ the absolute-row-sum bounds of the corresponding nested commutators of the unit matching adjacencies.

**Proof.** Apply the ordered Lie and Strang bounds of Result 36 first to $V$ and the full kinetic, and then to the kinetic factors. Different variables commute. On a chain, each nonfinal link has one later neighboring link, and its commutator has norm $w_j^2$. On an even cycle, the first link has two later neighbors whose commutator occupies two disjoint skew-symmetric blocks, together still of norm $w_j^2$. Counting the remaining links gives $\Gamma$. For matching layers, each unit adjacency has norm at most one, so the nested commutator norm is at most four. Computing its sparse row sums can improve that bound and preserves graph-specific cancellations. The binary kinetic is one whole Fourier-conjugated factor per variable, so it has no internal link split.

The commutator bounds can be obtained from support tables without a $K^d$-dimensional matrix. Let $V_S$ be the table on support $S$, $v_S$ half its range, $\kappa_j\ge\|T_j-\theta_jI\|$, $\tau_S=\sum_{j\in S}\kappa_j$, $\tau=\sum_j\kappa_j$, and $\nu=\sum_Sv_S$. Centering leaves commutators unchanged, giving

```math
\|[T,V]\|\le C_0:=2\sum_Sv_S\tau_S,\qquad \|[T,[T,V]]\|\le D_{T,0}:=4\sum_Sv_S\tau_S^2.
```

For a finite-difference stencil, define the potential-difference bound on an axis-$j$ link by

```math
\Delta_{j,pq}:=\sum_{S\ni j}\max_{\mathbf y}|V_S(q,\mathbf y)-V_S(p,\mathbf y)|,
```

where $\mathbf y$ ranges over the other coordinates of $S$. Let $\mathcal N_j(p)$ be the neighbor multiset, including periodic wrap edges and repeated neighbors, and $w_j=1/(2h_j^2)$. The edge bounds are

```math
C_{\rm edge}:=\sum_jw_j\max_p\sum_{q\in\mathcal N_j(p)}\Delta_{j,pq},\qquad D_{V,\rm edge}:=\sum_jw_j\max_p\sum_{q\in\mathcal N_j(p)}\Delta_{j,pq}^2.
```

The identities $[T,V]_{xy}=T_{xy}(V_y-V_x)$ and $[V,[V,T]]_{xy}=T_{xy}(V_x-V_y)^2$ give the corresponding absolute row sums. Since the commutators are anti-Hermitian or Hermitian, these row sums bound their operator norms. Support contributions are added before squaring, keeping cross terms. Once $C$ bounds $\|[T,V]\|$, the outer-commutator inequality gives $\|[T,[T,V]]\|\le2\tau C$ and $\|[V,[V,T]]\|\le2\nu C$. Thus the chosen bounds are defined by

```math
\|[T,V]\|\le\min(C_0,C_{\rm edge})=:C,
```

```math
\|[T,[T,V]]\|\le\min(D_{T,0},2\tau C)=:D_T,\qquad \|[V,[V,T]]\|\le\min(2\nu C,D_{V,\rm edge})=:D_V.
```

For the dense spectral kinetic, or when edge bounds are unavailable, take $C_{\rm edge}=D_{V,\rm edge}=+\infty$ in these selections, so the range bounds stand alone. Summing step errors uses unitarity and can be capped by two. A bound based only on $[T,V]$ would miss the one-hot link error even for constant $V$.

<a id="r41"></a>
### Proposition 41. Split-step roundoff and work at a fixed grid

**Provenance: Derived in NWQLib. Evidence: a proved propagation formula conditional on local numerical-error bounds.**

The split-step kernel applies $e^{-iB_kV/2}e^{-iA_kT}e^{-iB_kV/2}$, with the kinetic factor diagonal in an orthonormal DST-I or Fourier basis. This is the symmetric form of Leng et al., Eq. (C.3), using [Strang's splitting][Strang]. Suppose the initial state error is at most $\delta_0$, and step $k$ has relative numerical implementation error at most $\rho_k$. A finite propagation bound is

```math
\delta_{\rm finite}\le(1+\delta_0)\prod_k(1+\rho_k)-1.
```

For a fixed execution with $\delta_0=\delta_0^{(1)}+O(u^2)$ and $\rho_k=\rho_k^{(1)}+O(u^2)$, its first-order part is $\delta_0^{(1)}+\sum_k\rho_k^{(1)}$. The Plan records this first-order budget in `split_step.state_error`.

**Derivation.** Applying an approximate unitary step to an already perturbed state gives $\delta_{k+1}\le(1+\rho_k)\delta_k+\rho_k$. Induction proves the finite formula. For $d$ variables, two transforms per axis and $d+2$ phase applications give the Plan's first-order step allowance

```math
\rho_{k,\max}^{(1)}=2dC_TL(K)u+5(d+2)u+(C_E+2)u|A_k|\sum_j\max E_j+2(M+1)u|B_k/2|W,
```

where $M$ is the number of support tables, $W=\sum_S\max|V_S|$, $C_E=13$, and $C_T=5$. Here $L(K)=\lceil\log_2K\rceil$ for FFT and $\lceil\log_2(2(K+1))\rceil$ for DST-I. Each normalized transform is assumed to satisfy $\|\operatorname{fl}(\mathcal Fx)-\mathcal Fx\|\le C_TL(K)u\|x\|$. Squared errors on orthogonal tensor fibers add, so the same factor applies to the whole tensor. One-ulp sine and cosine evaluations contribute at most $2u$ to a phase, and complex multiplication contributes $\sqrt2\gamma_2$. The implementation rounds their first-order coefficient $2+2\sqrt2$ upward to 5. Forming $A_k$ and its eigenvalue product gives the two additional units beside $C_E$. Summing $M$ tables, forming $B_k/2$, and multiplying the potential give the $(M+1)W$ term, zero when there are no tables.

The kernel observes smaller angle-error terms on its computed trajectory. If entrywise angle errors are $q_m$, their action on a unit input $z$ is at most $(\sum_m|z_m|^2q_m^2)^{1/2}$. The kinetic error term for axis $j$ is therefore $(C_E+2)u|A_k|\|E_jz\|$, with $z$ observed after that axis's forward transform. Each potential half has error term

```math
\eta_{V,k}(z)=u|B_k/2|\left(2\|Vz\|+mW\right),\qquad m=\max(M-1,0).
```

The two halves use their respective inputs. A diagonal unitary preserves their entry magnitudes, so the input moment of step $k$'s second half can also serve step $k+1$'s first half. The implementation takes one initial potential moment and one after each step's kinetic factors, for $N+1$ moments over $N$ steps. Reusing a moment across the computed phase multiplication omits a product of local roundoff terms of second order, consistently with the budget's first-order scope. Absolute half-exponents are counted separately, so signed weights do not cancel their allowances.

The moments are evaluated after scaling. For kinetic angles $\widehat\theta_m$, let $s_m$ be their computed mode populations and let $a$ be the computed cap $|A_k|\max E_j$. The kernel forms

```math
a\left(\frac{\sum_m(\widehat\theta_m/a)^2s_m}{\sum_ms_m}\right)^{1/2}.
```

Monotonic rounding makes $a$ at least every $|\widehat\theta_m|$, so these squared ratios cannot overflow. For a potential moment on computed input $y$, let $w_x=\widehat V_x|y_x|$, $S=\sum_x|y_x|^2$, and $b=\max_x|w_x|$, with $\widehat V$ the stored sum of the support tables. The kernel forms

```math
\mu=b\left(\frac{\sum_x(w_x/b)^2}{S}\right)^{1/2}.
```

The scaled potential squares lie in $[0,1]$, and the largest is one when $b>0$. Thus squaring cannot overflow, and all scaled squares below the normal range together affect this sum by less than $D\nu_{\rm fp}$, where $D=K^d$. A zero input or zero weighted potential gives zero. The implementation caps each observed moment at its Plan maximum and replaces a nonfinite moment by that maximum. Roundoff of the moment reductions and the use of $\widehat V$ enter the error term at second order under the relative-error model. Subnormal weighted products have an additional absolute rounding scale, so scaling alone does not establish an all-orders relative-error bound.

The returned state budget is the smaller of the observed first-order sum and the Plan's first-order sum. The former determines the executed tie window through Proposition 45 with $e=5u$. The Plan budget determines the mass window and the upper limit of the tie window. A state budget and a probability window have different units and are related by that conversion.

The transform constant and elementary-function accuracy are qualification assumptions. Finite checks cannot prove a uniform transform bound for all lengths. The stored budgets omit higher-order products and local errors, and their relative terms require the stated normal-arithmetic premises. All-orders local bounds would be needed to put implementation values into the finite product bound. The numerical reference uses the stored duration and weights, analytic kinetic eigenvalues at stored spacings, and the exact sum of the stored potential tables. Schedule, grid, potential-evaluation, and splitting errors belong elsewhere.

The evolution has $O(NdD\log K)$ transform work. Including moment observations, `split_step.sizes` uses the work proxy

```math
W_{\rm start}+D(d+M+7)+dK+N\left(2dL(K)D+(2d+13)D+7dK+d+4\right).
```

Here $W_{\rm start}$ is the start-vector work. Apart from the transform term, which is a nominal level count, each per-step term counts one unit per element of an array pass that the kernel runs, while the once-per-evolution terms are nominal counts. Per axis and step these are $K$ angle products, two passes of $K$ for the phases (the product with $-i$ and the exponential), and four passes of $K$ that weight the kinetic moment (the sum of the mode populations, the division of the angles by their cap, the square and the reduction against the populations), $7dK$ in all. The $D$-length terms include one population pass per axis and seven passes for each potential moment. The potential angles are formed once for both halves of a step while the formula counts $2D$ for them. The potential moment reuses the angle array, the kinetic moment keeps $K$ mode populations, and the kernel stores $N$ scalar step error terms, giving $O(D+dK+N)$ main numerical arrays besides its stored input tables. The work proxy estimates the selected array operations and transform levels, with backend-internal work and workspace subject to their separate qualifications. At fixed $N$ and grid it depends on neither the kinetic coefficient nor the potential range. The number of steps needed for a desired physical accuracy can depend strongly on both.

<a id="r42"></a>
### Proposition 42. Compensated running phase total and the phase condition for kept states {#proposition-42-a-compensated-phase-ledger-and-phase-sensitive-admission}

**Provenance: Improved from a reference. Evidence: proved under the stated floating-point model.**

Consider the exact identity phase of the stored one-hot discrete model,

```math
\Phi=-\sum_k\Delta t\left(a_k\sum_jh_j^{-2}+b_kc+b_k\sum_{S,r}2^{-|S|}v_{S,r}\right).
```

The implementation accumulates its contributions by compensated summation in a running phase total. Let $S_C,S_P,S_K$ be the absolute sums of its constant, projector, and kinetic contributions. Under round-to-nearest arithmetic without overflow or inexact underflow, define

```math
\beta_n=u+(1+u)\gamma_{n-1}^2,\quad \eta_2=2u+u^2,\quad \eta_K=\frac{3u+u^2+(1+2u+u^2)\beta_d}{1-u},
```

```math
D_{\rm form}=\eta_2(S_C+S_P)+\eta_KS_K,\qquad E_{\rm ledger}=D_{\rm form}+\beta_M(S_C+S_P+S_K+D_{\rm form}),
```

where $M$ counts recorded contributions, including zero events for omitted identities, and an empty running total has allowance zero. The displayed expression bounds the arithmetic of accepted contributions using the original absolute sums. A constant or projector identity whose required products fall below the normal binary64 range is recorded as a zero event, and its exact omitted phase magnitude is added once to $E_{\rm ledger}$.

For a potential projector $P$ on support size $s\ge1$, let $Z=tbv$ be its exact intended exponent from the stored duration, weight, and table value. A lower-range failure omits the occurrence and its identity contribution together. Since $P=(P-2^{-s}I)+2^{-s}I$ and $\|P-2^{-s}I\|=1-2^{-s}$, sufficient error terms are

```math
E_{\rm projector}=(1-2^{-s})|Z|,\qquad E_{\rm identity}=2^{-s}|Z|.
```

Their sum is $|Z|$, the bound for dropping the full projector. The first term is already included in `rotation_pruning`, and the second enters `identity_phase` through the running phase total. Both second-order halves count as occurrences. An omitted scalar constant contributes $|\Delta t\,b_kc|$ only to the identity term. The construction keeps the original table values, so a subnormal value amplified into accepted products need not be omitted.

**Proof.** Each constant or projector contribution has two rounded products, giving $\eta_2$. A kinetic coefficient uses a rounded square, division, compensated sum over $d$ coordinates, and duration product. Multiplying their relative factors gives $\eta_K$. Compensated accumulation has error bounded by $u|X|+\gamma_{M-1}^2A$, with the slightly enlarged $\beta_MA$ covering its final rounding. This is the summation bound of [Ogita, Rump and Oishi, Proposition 4.5][CompSum]. The absolute sum of formed contributions is at most $S_C+S_P+S_K+D_{\rm form}$, which proves the displayed bound on the running total. An omitted contribution adds zero to the accumulator, so the same bound holds for the accepted contributions, and the triangle inequality adds each omitted phase $|a|$. Cancellation does not reduce the formation term. Every operation used to evaluate the positive bound is rounded outward in the implementation.

Native phase assignments need an additional bound because phases are repeatedly reduced modulo the stored binary64 approximation $\widehat\tau$ to $2\pi$. Let $\lambda=2^{-1074}$, $\Delta_\tau=2^{-51}\ge|\widehat\tau-2\pi|$. For an intended increment of magnitude $v$ with formation error $e$, an existing stored phase in $[0,\widehat\tau]$ gives

```math
H=\widehat\tau+v+e,\quad A=uH+\lambda,\quad K=(H+A)/\widehat\tau+1,\quad C(v,e)=e+A+\Delta_\tau K+\Delta_\tau.
```

The terms respectively account for increment error, addition, use of the approximate period at a bounded reduction quotient, and rounding of the remainder. Since $6<\widehat\tau<7$, $\Delta_\tau=4u$, and $e\le\lambda$, the coefficient of $v$ is $u+4u(1+u)/\widehat\tau<2u$. The constant contribution is below $19u$ plus terms smaller than $u$, giving $C(v,e)\le2uv+20u$. Projector scaling uses $e=\lambda$, and the final stored phase is an exact input to its assignment, with $e=0$. Its prior formation error is covered separately by $E_{\rm ledger}$. Circular phase errors add. For $B$ projector assignments with absolute sum $Y$, and the final physical-phase assignment, this yields the implementation's conservative bound

```math
E_{\rm native}=E_{\rm ledger}+2u(Y+|\widehat\Phi|)+20u(B+1).
```

This last form uses the specified Qiskit 2.5.2 phase setter and gradual underflow. In binary encoding the scalar running phase total contains only the objective constant. Walsh blocks need a separate formation allowance even when that total is empty.

The objective constant enters the running phase total as a global phase, so a large phase-sensitive bound can include error that leaves the ideal readout probabilities unchanged and does not by itself establish poor readout accuracy.

For one selected Walsh block, let $X=tw$ be its exact intended exponent, $x=\operatorname{fl}(tw)$, $\epsilon_x\ge|x-X|$, and $\overline c_0$ the accepted identity coefficient. Let $c_0^*$ be the corresponding exact coefficient and $\rho_0\ge|\overline c_0-c_0^*|$. Write $\overline\phi$ for the emitted identity phase, zero when its product is omitted. Then

```math
e_W=|X|\rho_0+\epsilon_x|\overline c_0|+|\overline\phi+x\overline c_0|,\qquad F_W=\sum_{\text{Walsh blocks}}e_W.
```

Indeed $\overline\phi+Xc_0^*=(\overline\phi+x\overline c_0)+(X-x)\overline c_0+X(c_0^*-\overline c_0)$. The final absolute term is an exact scalar discrepancy covering either product rounding or omission. For a potential table of length $2^n$, $\rho_0=\gamma_n a+d_0$, where $a$ is its mean absolute value and $d_0$ its identity normalization omission, as in Proposition 44. An analytic kinetic table uses its coefficient enclosure instead. The implementation evaluates the allowance upward, using $|X|\le|x|+\epsilon_x$ and $\epsilon_x=u|tw|+\lambda$, the latter conservatively counting the half-ulp underflow term as a full $\lambda$.

For reconstruction from the emitted identity phase and kept rotation angles, put $T_W=\frac12\sum_{\rm kept}|\theta_m|$. An $n$-level inverse Walsh butterfly, halving, and final subtraction give the per-entry allowance

```math
R_{W,\mathrm{block}}=u|\overline\phi|+\gamma_{n+1}T_W+2\lambda,\qquad R_W=\sum_{\text{Walsh blocks}}R_{W,\mathrm{block}}.
```

The butterfly half-sum error is at most $\gamma_nT_W$. Combining the final relative rounding with it uses $\gamma_n+u(1+\gamma_n)\le\gamma_{n+1}$, and the absolute halving and subtraction terms fit within $2\lambda$. Reconstruction uses the actual emitted identity and kept angles. Omitted coefficients and virtual pruned angles do not enter $T_W$.

For native binary phase assignments, let $Y_W$ be the sum of emitted identity-phase magnitudes and $B_W$ the Walsh-block count. The allowance is

```math
E_{\rm native,bin}=E_{\rm ledger}+F_W+2u(Y_W+|\widehat\Phi|)+20u(B_W+1).
```

The `ir_product` phase comparison instead includes $F_W+R_W$ in addition to the allowance of the scalar running phase total. $R_W$ is not a native circuit error. Dense-diagonal wrapping and synthesis have their separate error-budget entry, whose unavailable status is not resolved by these Walsh formulas.

The binary IR floating-point budget describes state operations on computed phase arrays and selected stored QFT angles. Its first-order arithmetic model excludes phase-array formation and the discrepancy between a reconstructed Walsh array and the exact action of the stored rotations. The phase allowance checked for a kept state includes $R_W$, but $R_W$ is not propagated into IR readout accuracy, the mode tie window or verification uncertainty.

Finally, a phase uncertainty $E$ changes a unit vector by at most $2\sin(\min(E,\pi)/2)$. This follows from $|e^{i\theta}-1|=2|\sin(\theta/2)|$. Once $E\ge\pi$, the bound is the full diameter two. A kept state is therefore accepted only with a finite phase allowance below $\pi$. Probabilities are invariant under a common phase, so this criterion concerns phase-sensitive states.

<a id="r43"></a>
### Proposition 43. Signed binary momentum and sparse kinetic polynomials

**Provenance: Improved from a reference. Evidence: proved from the periodic spectrum and bit expansion.**

For $K=2^b$, the Fourier index is signed: $q=k$ for $k<K/2$, and $q=k-K$ otherwise. The spectral kinetic is $E_q^{\rm sp}=2\pi^2q^2/L^2$, with $L=Kh$. Let $w_\ell=2^\ell$ below the sign bit and $w_{b-1}=-2^{b-1}$. Then

$$
q^2=\frac{1+\sum_\ell w_\ell^2}{4}I+\frac12\sum_\ell w_\ell Z_\ell+\frac12\sum_{\ell<j}w_\ell w_j Z_\ell Z_j.
$$

This is the signed correction to the unsigned $k^2$ expression of [Liu et al., Section IV.E, Eqs. (89)–(91)][Liu]. It has $b(b-1)/2$ two-qubit strings and hence $b(b-1)$ CX gates in the parity-ladder synthesis.

**Proof.** Write $q=\sum_\ell w_\ell n_\ell$, $n_\ell=(1-Z_\ell)/2$. Since $\sum_\ell w_\ell=-1$, expanding the square and using $Z_\ell^2=I$ gives the result. The unsigned formula gives $(K-1)^2$ at index $K-1$, which represents momentum $-1$ and requires value one. Each weight-two string needs two CX gates.

The exact finite-difference kinetic has a different Walsh structure. Factor

```math
e^{2\pi ik/K}=e^{i\pi(K-1)/K}\prod_\ell\left(\cos a_\ell-iZ_\ell\sin a_\ell\right),\qquad a_\ell=\pi2^\ell/K.
```

The sign-bit cosine is zero. Therefore every nonconstant nonzero coefficient of $E_k^{\rm FD}=(1-\cos(2\pi k/K))/h^2$ contains the sign bit. There are $2^{b-1}$ such masks. Summing $2(w-1)$ over them gives $(b-1)2^{b-1}$ CX gates. Computing these structural zeros analytically avoids treating Fourier-rounding residue as a physical interaction.

The model difference remains substantive. For $|q|\le K/2$,

```math
0\le E_q^{\rm sp}-E_q^{\rm FD}\le\frac{2\pi^4q^4h^2}{3L^4}.
```

Indeed $x^2-\sin^2x\le x^4/3$ for $0\le x\le\pi/2$, by $\sin x\ge x-x^3/6\ge0$, and substitute $x=\pi|q|h/L$. Fixed low momenta converge quadratically in $h$, while the Nyquist ratio is $\pi^2/4$. A trajectory comparison requires $\int|a(t)|\|(T_{\rm sp}-T_{\rm FD})\psi_{\rm sp}(t)\|dt$, obtained by Duhamel, or a valid upper bound on it. Initial low momentum alone does not control that integral in a nonconstant potential.

The circuit error budget therefore reports `kinetic_model` as unavailable for a spectral kinetic, with no numerical value, and as not applicable for the finite-difference kinetic. The later circuit entries compare with the selected finite model. Including `kinetic_model` changes the reference to the finite-difference model, and its missing bound prevents that comparison from becoming complete. The objective-gap error model also declares this source for a spectral choice, without promoting a state-norm allowance into an optimization bound.

<a id="r44"></a>
### Proposition 44. Binary phase synthesis, bit reversal, and truncation budgets

**Provenance: Derived in NWQLib. Evidence: proved for the selected diagonal and Fourier products.**

For a diagonal table $V=\sum_m c_mZ_m$,

```math
e^{-ixV}=e^{-ixc_0}\prod_{m\ne0}e^{-ixc_mZ_m}.
```

A mask of weight $w$ needs $2(w-1)$ CX gates and one $R_z(2xc_m)$. A nonzero dense diagonal on $n$ qubits has the structural count $2^n-2$ CX gates. `min_cx` chooses the smaller count of the accepted constructions at that exponent, selecting Walsh rotations on a tie. If all dense phases are omitted, its construction costs zero. This comparison concerns these two decompositions and does not imply minimum T count.

**Proof and errors.** Commutativity gives the factorization. Computing the parity onto a pivot and uncomputing uses two ladders of $w-1$ CX gates. Threshold-pruning a computed rotation angle $\theta$ changes its factor by $2|\sin(\theta/4)|\le|\theta|/2$, and these error terms add by unitary telescoping.

For a potential table of length $N=2^n$, let $\widehat s_m$ be its finite unnormalized Walsh butterfly outputs, $z_m=\widehat s_m/N$, and $c_m^*$ the exact Walsh coefficients of the stored table. Define $a=N^{-1}\sum_z|v_z|$. Every summand passes through $n$ rounded additions or subtractions, so $|z_m-c_m^*|\le\gamma_n a$ for $n u<1$. Exact subnormal sums and differences preserve that model. The normalization accepts $\overline c_m=z_m$ when $z_m=0$ or $|z_m|\ge\nu_{\rm fp}$, and otherwise sets $\overline c_m=0$ before an underflowing division. Thus

```math
d_m=|\overline c_m-z_m|,\qquad |\overline c_m-c_m^*|\le\rho_m:=\gamma_n a+d_m.
```

The accepted normal division is an exact power-of-two scaling. A computed zero still carries the $\gamma_n a$ term. In particular, the table $(\nu_{\rm fp},\nu_{\rm fp}+\lambda)$ has exact nonidentity coefficient $-\lambda/2$, whose positive omission error term must survive conversion to binary64. A dense-only table uses its values directly and needs no Walsh transform or Walsh range check.

For one block let $X=tw$ be its exact intended exponent, $x=\operatorname{fl}(tw)$ its accepted exponent, and $\epsilon_x=|x-X|$. With

```math
C=\sum_{m\ne0}|\overline c_m|,\qquad B_c=\sum_{m\ne0}\rho_m=(N-1)\gamma_n a+\sum_{m\ne0}d_m,
```

the nonidentity Walsh formation error is at most

```math
\min\left(2,\ |X|B_c+(\epsilon_x+u|x|)C\right).
```

To obtain it, split each phase discrepancy into coefficient error $X(\overline c_m-c_m^*)$, exponent error $(x-X)\overline c_m$, and rotation-product error. The doubled exponent is accepted and exact. A normal computed angle has half-angle error at most $u|x\overline c_m|$. A lower-range rotation is compared at this stage through its exact virtual angle $2x\overline c_m$, with zero product error, then counted for its removal in `rotation_pruning`. Summing all nonidentity masks, including computed zeros and spurious nonzeros, proves the bound. Analytic kinetic Walsh coefficients use their individual enclosure radii in $B_c$. The identity formation $F_W$ is counted separately by Proposition 42.

A range-omitted Walsh rotation contributes $|x\overline c_m|$ to `rotation_pruning`. For dense phases, range omission contributes $\max_{z\,\mathrm{omitted}}|xv_z|$ per block, with an empty maximum equal to zero, because the omitted generator is diagonal. The dense potential formation allowance is $(\epsilon_x+u|x|)\max_z|v_z|$. For accepted kinetic energies $\overline E_q$ with maximum magnitude $M_E$ and error radius $H_E\ge\max_q|\overline E_q-E_q|$, it is $(\epsilon_x+u|x|)M_E+|X|H_E$. These formation allowances compare range-omitted products through their exact virtual values. Coefficient omissions, rotation or dense-entry omissions, and omitted identity products therefore enter `angle_formation`, `rotation_pruning`, and $F_W$, respectively, without counting the same removal twice. Only the selected construction contributes omission counts and error terms. Dense wrapping and its later synthesis remain separate.

Let $W=RF$ be the swap-free Fourier transform, where $R^2=I$ reverses the address bits. Then $W^\dagger(RDR)W=F^\dagger DF$. Relabeling the phase table therefore removes both swap layers exactly. The positive and negative Fourier conventions also give the same kinetic conjugation because both selected energy tables satisfy $E_{-k\bmod K}=E_k$.

If controlled phases at separations $r>m$ are omitted from a $b$-qubit Fourier transform, then

```math
e_F\le\min\left(2,\sum_{r=m+1}^{b-1}(b-r)\,2\sin\frac{\pi}{2^{r+1}}\right).
```

Each omitted controlled phase has difference norm $2|\sin(\theta/2)|$. Counting the $b-r$ occurrences and telescoping proves the bound. Replacing $F^\dagger DF$ by $\widetilde F^\dagger D\widetilde F$ costs at most $2e_F$, hence at most $2dNe_F$ over $N$ time steps and $d$ variables. This is an explicit norm budget for the approximate-QFT construction of [Coppersmith][Coppersmith].

The stored QFT angles also differ from those using mathematical $\pi$. With $\epsilon_\pi\ge|\widehat\pi-\pi|$, exact accepted power-of-two scaling and the controlled-phase distance bound give

```math
E_{Q,\pi}\le\min\left(2,\ 2dN\epsilon_\pi\left(b-2+2^{1-b}\right)\right).
```

Here $\sum_{r=1}^{b-1}(b-r)2^{-r}=b-2+2^{1-b}$, zero for $b=1$, and the factor two counts each kinetic conjugation's forward and inverse QFT. `angle_formation` includes this full-QFT error term. The `aqft` entry compares the full stored-angle transform with its truncation. Kinetic trigonometric enclosures and evaluation of the AQFT bound remain conditional on their stated elementary-function accuracy assumptions. The implementation evaluates $a$, $C$ and the finite-difference kinetic radii as upper binary64 values, correctly rounded sums moved upward and interval products propagated outward, and converts them exactly to rationals for the remaining chain, so the published $a$ and $C$ may exceed their exact values by a few units in the last place and the kinetic radii in $B_c$ by more, and the allowance cannot be smaller than its exact-rational value.

<a id="r45"></a>
### Proposition 45. Probability-difference windows for mode selection

**Provenance: Derived in NWQLib. Evidence: proved conditional on a state-error bound.**

If $\|\widehat\psi-e^{i\phi}\psi\|\le\delta$, $\|\psi\|=1$, and each computed probability has relative evaluation error at most $e$, then every computed difference obeys

```math
|(\widehat p_i-\widehat p_j)-(p_i-p_j)|\le2\delta+\delta^2+e(1+\delta)^2.
```

**Proof.** Let $D_{ij}=|i\rangle\langle i|-|j\rangle\langle j|$, which has norm one. Expanding its expectation in $\widehat\psi=e^{i\phi}\psi+\epsilon$ bounds the state term by $2\delta\sqrt{p_i+p_j}+\delta^2\le2\delta+\delta^2$. The two evaluation errors add at most $e(|\widehat\psi_i|^2+|\widehat\psi_j|^2)\le e(1+\delta)^2$. This avoids adding two independent full-state probability bounds.

The same expansion bounds mass error by $2\delta+\delta^2$. Evaluation of $D$ absolute squares and their sum contributes $(D+4)u(1+\delta)^2$ to first order for the host kernel. Its pairwise tie conversion instead uses $e=5u$ in the displayed difference bound. The split-step route uses the observed first-order state budget of Proposition 41 for the tie window and its Plan budget for the mass window and the upper limit of the tie window. The other host flavors use their Plan state budget for both conversions. These conversions inherit the premises and approximation order of their input budget. The direct one-hot product kernel applies the analytic projector and hopping blocks at their stored angles, and its first-order state-operation budget is $u(\mathrm{start}+10B_P+6B_H+5)$ for $B_P$ projector and $B_H$ hopping occurrences (`theory.onehot_product_state_error`).

The displayed inequality concerns the exact difference of the computed probabilities. The implemented comparison forms `peak - probability` in binary64 and does not add a separate subtraction-rounding term, so the formula is not an outward certificate for that complete comparison. Exact count populations are compared as integers and use no floating-point tie window. A mode selected within a nonzero window represents an unresolved probability ordering. The window is unrelated to objective optimality.

Suppose $q_i$ are computed probabilities and $p_i$ are their model values on the observed valid set. Assume that every pairwise difference has error at most $W$. If the selected point $s$ satisfies $\operatorname{RN}(q_m-q_s)\le W$ for a computed maximizer $m$, then its model-probability deficit from any model maximizer in that set is at most $W+W/(1-u)$, where $u=2^{-53}$, provided the subtraction is finite. If that subtraction is exact, the bound is $2W$. For normalized model probabilities, either bound can be capped at 1.

**Derivation of the selected-point deficit.** Let $t$ maximize $p$ within the observed valid set. The pairwise-error premise is

```math
 |(q_i-q_j)-(p_i-p_j)|\le W.
```

Since $q_t\le q_m$,

```math
 p_t-p_s\le q_t-q_s+W\le q_m-q_s+W.
```

Put $d=q_m-q_s\ge0$. When subtraction is normal and finite, $\operatorname{RN}(d)=d(1+\delta)$ with $|\delta|\le u$. The actual acceptance test gives $d\le W/(1-u)$. A subnormal difference of binary64 inputs is exact. In either case the advertised bound follows. If $q_s\ge q_m/2$, Sterbenz's exact-subtraction condition applies and gives $d\le W$, hence $2W$. The premises say nothing about an unobserved outcome. Counts use $W=0$ for their empirical frequencies, not for the unknown sampling distribution. With an unavailable window there is no supplied numerical $W$ premise.

<a id="r46"></a>
### Result 46. Normalized augmented-Lagrangian equations

**Provenance: From a reference. Evidence: the cited PHR and augmented-Lagrangian identities.**

Use normalized objective $f/s_f$, equalities $h_i/s_{h_i}$, and inequalities $g_j/s_{g_j}$, with positive scales, and suppress tildes below. Equality multipliers $\bar\lambda$ are unrestricted in sign, while inequality multipliers satisfy $\bar\mu\ge0$. A round with penalty $\rho>0$ uses

```math
L(x)=f(x)+\bar\lambda^Th(x)+\frac\rho2\|h(x)\|^2+\sum_j\frac{[\bar\mu_j+\rho g_j(x)]_+^2-\bar\mu_j^2}{2\rho}.
```

The equality part and update occur in [Wu et al., Eqs. (6)–(8)][Wu]. The inequality term is the Powell–Hestenes–Rockafellar (PHR) form, equal to $\min_{z\ge0}\{\bar\mu_j(g_j+z)+\rho(g_j+z)^2/2\}$, whose minimizer is $z_*=\max(0,-g_j-\bar\mu_j/\rho)$. [Rockafellar][PHR], and [Birgin and Martínez, Eq. (4.3)][BM], which omits the constant term. The default PHR representation needs no slack register, while an inequality converted to explicit slack form adds an inner coordinate. [Proposition 54](#r54) states what changes when an inequality is instead represented by an explicit inner slack coordinate.

The tentative updates and penalty measure are

```math
\lambda^+=\bar\lambda+\rho h,\quad \mu^+=[\bar\mu+\rho g]_+,\quad V_j=\min(-g_j,\bar\mu_j/\rho),\quad m=\max(\|h\|_\infty,\|V\|_\infty).
```

These follow Birgin and Martínez, Algorithm 4.1 and Eqs. (4.7)–(4.9). In particular $\mu^+-\bar\mu=-\rho V$. The default penalty stays unchanged after the first round and whenever the current $m$ is at most a selected fraction of its preceding value. Otherwise it increases by a selected factor, subject to a finite cap. Safeguards, when selected, act on the multipliers for the next round. In original units, equality multipliers are $(s_f/s_{h_i})\lambda_i$, and inequalities have the analogous formula, from the scaled KKT equations and the book's Eq. (10.4).

Rescaling a constraint and its positive scale by the same factor leaves its normalized expression, residual tests, and updates unchanged in exact arithmetic. This does not guarantee identical binary64 runs, even for a power of two. The symbolic normalization divides by a binary64 SymPy scale before expansion, while a scale of one leaves exact symbolic coefficients unchanged. Different coefficient rounding can change a table tie, the selected point, or a stopping decision. Normalized multipliers remain the layer's state even when their original-unit conversion is not representable in the accepted range. Requesting such a conversion raises a range error.

<a id="r47"></a>
### Proposition 47. The finite-grid meaning of feasibility and complementarity stopping

**Provenance: Derived in NWQLib. Evidence: proved bound and exact counterexample.**

Suppose $x$ is a $\delta$-approximate global minimizer of the round's augmented Lagrangian over a finite grid, and $x$ is feasible. Then for every feasible grid point $y$,

```math
f(x)-f(y)\le\delta-\sum_jP_j(g_j(x))\le\delta+\sum_j\frac{\bar\mu_j^2}{2\rho}.
```

**Proof.** At feasible points the equality terms vanish, and $P_j(g_j(y))\le0$. The inequality $L(x)\le L(y)+\delta$ gives the first bound. Since $P_j(t)\ge-\bar\mu_j^2/(2\rho)$, the second follows. If $x$ is complementary with the entering multipliers, its PHR terms vanish and the gap is at most $\delta$. Complementarity with the updated multipliers need not have that consequence.

For an exact counterexample, take grid $\{0,1\}$, $f(x)=(x-1)^2/10$, $g(x)=x-1\le0$, $\bar\mu=1/2$, $\rho=1$. Then $L(0)=-1/40<L(1)=0$, so exact inner minimization selects zero. Its updated multiplier is zero, and feasibility and updated complementarity both hold exactly, although the feasible grid optimum at one has objective smaller by $1/10$.

NWQLib's default stopping conditions are the normalized feasibility and complementarity quantities of Birgin and Martínez, Eqs. (10.7)–(10.8). Their stationarity condition, Eq. (10.6), is a separate diagnostic.

Request the diagnostic with `options=AugmentedLagrangian(stationarity=True)`. Read `result.last.evaluation.stationarity` when `result.last` is present, or `result.last.evaluation.stationarity_unavailable` for its unavailability reason.

Let $f_n=f/s_f$, $h_n=h/s_h$ and $g_n=g/s_g$ denote the normalized objective and constraints, with positive scales. The stationarity diagnostic uses the tentative multipliers at the selected point,

```math
 \widetilde\lambda^+=\bar\lambda+\rho h_n(x),
 \qquad
 \widetilde\mu^+=[\bar\mu+\rho g_n(x)]_+,
```

and the gradient

```math
 \nabla_x L
 =\nabla f_n+J_{h_n}^{T}\widetilde\lambda^+
              +J_{g_n}^{T}\widetilde\mu^+.
```

For box coordinates $x=a+Du$, where $D$ is the diagonal matrix of positive box widths, the reported residual is

```math
 r_{\rm stat}
 =\left\|u-\Pi_{[0,1]^d}(u-D\nabla_xL)\right\|_\infty.
```

It is a computed first-order diagnostic for the last selected point and its tentative update. Minimizing a finite grid does not imply that this continuous projected-gradient residual vanishes.

The gradient identity follows by differentiating the PHR term, whose derivative is $2[z]_+$ for $[z]_+^2$, including at $z=0$. It uses tentative multipliers, not safeguarded-for-next-round multipliers. The chain rule gives $\nabla_uL=D\nabla_xL$.

A grid-point rule can satisfy an equality tolerance only if its grid contains a point whose computed normalized equality residual passes that tolerance. Increasing the penalty or updating multipliers does not add grid points. Failure of this condition prevents feasibility at that resolution, but does not imply that selected points oscillate or that the penalty grows without bound. Safeguards, stopping tests, work limits or an inner error can end the run.

A finer grid or a changed box can make a passing point available. A larger feasibility tolerance can permit a nonzero residual, which changes the acceptance criterion. A mean point rule can return an off-grid point, so the fixed-grid obstruction does not apply to it in the same way.

For a feasible inequality, let $s_j=-g_j(x)/s_{g,j}\ge0$ be its normalized slack and $\mu_j^+\ge0$ its tentative normalized multiplier. The complementarity test compares $\min(s_j,\mu_j^+)$ with the tolerance. A strictly positive slack alone does not force a small multiplier. The test forces $\mu_j^+\le\epsilon_{\rm comp}$ only when $s_j>\epsilon_{\rm comp}$. When the slack itself is within tolerance, a larger multiplier can still pass that component.

#### Evaluated grid comparisons

`result.verify(checks=QHDVerification(comparisons=("grid_minimum",)))` evaluates the objective on the finite grid, including its constant term, as one array table from which every reported reference value is read. The returned gap compares freshly evaluated binary64 values. It is a diagnostic of that evaluated grid and does not by itself bound the gap for the exact mathematical objective. The comparison can also report `minimum_success_mass`, the observed weight of grid points whose evaluated objective differs from the evaluated minimum by at most `minimum_tolerance`. With zero tolerance, this counts ties in the evaluated values. The mass is computed from a kept state (`keep_state=True`), native exact probabilities or counts, and a classical Result without a kept state reports it as unavailable.

`constrained_grid_minimum(result)` checks its work and memory against `max_work` and `max_bytes`, evaluates $f$, $h$ and $g$ on all $K^d$ grid points, and finds the least evaluated objective among grid points that pass the computed feasibility test. Its signed difference subtracts this freshly evaluated minimum from the result's stored best objective. That difference can be negative, including when the stored point belongs to the evaluated feasible grid, because the two values can have different rounding errors. An off-grid or infeasible stored point gives no grid-optimality conclusion. If no grid point passes computed feasibility, `indices`, `point`, `objective` and `gap` are `None`. The gap is also `None` when the result has no best point. The helper does not support a result that uses box refinement.

Let $G$ be finite and nonempty, $a\in G$, $F$ the mathematical objective, $s$ the stored objective at $a$, and $t_x$ the fresh evaluation at $x\in G$. Assume finiteness and a supplied bound $E\ge0$ with

```math
 |s-F(a)|\le E,\qquad |t_x-F(x)|\le E\quad(x\in G).
```

Set $g=F(a)-\min_G F\ge0$ and $d=s-\min_G t$ in real arithmetic. From $s\ge F(a)-E$ and $\min_G t\le\min_G F+E$, obtain $d\ge g-2E$, hence $g\le d+2E$. The recorded difference is $r=\operatorname{RN}(d)$. For a normal finite binary64 result, $r=d(1+\delta)$ with $|\delta|\le u=2^{-53}$. For $r\ge0$ this implies $d\le r/(1-u)$. For $r<0$ the upper bound instead uses the largest positive denominator, $d\le r/(1+u)$. Subnormal subtraction of finite binary64 operands is exact and satisfies the relevant inequality. Therefore

```math
0\le g\le
\begin{cases}
r/(1-u)+2E,&r\ge0,\\
r/(1+u)+2E,&r<0.
\end{cases}
```

These are real bounds. A machine enclosure of the right side would require outward arithmetic. The code supplies neither $E$ nor this enclosure. For a constrained claim, $G$ must be the intended feasible comparison set, or the conclusion must explicitly concern the computed feasible set. No continuous-domain conclusion follows.

<a id="r48"></a>
### Proposition 48. Search-model normalization and affine invariance

**Provenance: Derived in NWQLib. Evidence: proved in exact table arithmetic, with a conditional evaluation bound.**

On box $B=a+D[0,1]^d$, decompose the stored objective into constant $c_0$ and support tables $T_S$. Set

```math
c=c_0+\sum_S\min T_S,\qquad E\ge\sum_S(\max T_S-\min T_S)>0,\qquad V(u)=\kappa\frac{F(a+Du)-c}{E}.
```

At the represented grid points, $0\le V\le\kappa$. The search uses the fixed dimensionless kinetic $-\frac12\sum_j\partial_{u_j}^2$. Under positive diagonal coordinate scaling $x'=sx+t$ and objective scaling $F'(x')=rF((x'-t)/s)+c_1$, $r>0$, the ideal normalized potential is unchanged provided that the unit grids and support tables correspond and $E$ uses the homogeneous rule $E=\sum_S(\max T_S-\min T_S)$ in exact arithmetic. The same gain, schedule, and initial state in unit coordinates then give the same level dynamics.

Table correspondence can be ensured by transforming the tables one by one without regrouping, or by applying the same support decomposition after expressing both objectives in unit coordinates, before forming their tables. A different support decomposition can change both $c$ and $E$. NWQLib uses the second construction in `src/nwqlib/algorithms/qhd/refinement.py::_level_problem`, mapping the objective to the unit box before decomposition and table formation. Its binary64 implementation uses an outward allowance for the sum of table ranges.

**Proof.** Every centered table lies between zero and its range. Their sum is between zero and $E$, proving the enclosure. The transformed box has $a'=sa+t$, $D'=sD$, and $F'(a'+D'u)=rF(a+Du)+c_1$. Under the table-correspondence premise, $T'_S(u_S)=rT_S(u_S)$ and $c'_0=rc_0+c_1$. Each table's minimum and range therefore scale by $r$. The homogeneous range rule gives $E'=rE$, while the definition of the shift gives $c'=rc+c_1$, so $V'=V$. The fixed kinetic and matching schedule and initial state give equal level dynamics. The gain $\kappa$ must be applied after range normalization, since multiplying the objective before forming its range cancels out.

If original table evaluations have errors $e_S$, replanned scaled tables have errors $e'_S$, and $f=\kappa/E$, then each difference from the scaled stored table is at most $e'_S+fe_S$. Rounding the centered constant adds at most $uf\sum_S|\min T_S|$ in normal arithmetic. Thus the solved table sum lies in $[-\delta,\kappa+\delta]$, with

```math
\delta=\sum_S(e'_S+fe_S)+uf\sum_S|\min T_S|.
```

This is conditional on the actual expression-evaluation errors. A range near a few units in the last place cannot certify the objective's variation. Binary64 threshold and tie decisions need not be affine invariant. Finally, the physical kinetic under the coordinate change is $-\frac12\sum_jD_j^{-2}\partial_{u_j}^2$, so the search model is a different Hamiltonian on each box, not a coordinate rewrite of a fixed physical evolution.

#### Selection across refinement levels

The following comparison assumes `inequality_form="phr"`. A refined round compares the recorded relative $L_k$ values of its completed levels and chooses the least one, with the earlier level winning an exact tie. Within a level, `best_observed` chooses the least evaluated value of the objective that level solved. A search-scaled level solves its normalized search objective, while a physical level solves $L_k$. Comparisons between levels use recorded relative $L_k$ values.

With exact evaluations and full observation of the first level's grid, `best_observed` makes the selected round point no worse in $L_k$ than that grid's minimum. In floating point, the corresponding statement requires bounds for both the solved-table error and the error in comparing relative $L_k$ values. Partial observation, another point rule, or a numerically flat level does not supply the full-grid premise.

For the rounded version fix $L=L_k$ and a common constant $C_0$. Write the recorded relative value as $R(z)=L(z)-C_0+n(z)$ with $|n(z)|\le\eta$. If $p$ wins across completed levels, then for every reported $z$,

```math
 L(p)-C_0+n(p)\le L(z)-C_0+n(z),
 \qquad L(p)\le L(z)+2\eta.
```

Suppose the first completed level observes all of $G_0$ and uses `best_observed`. After undoing a positive affine search scaling, assume its solved table is $T(x)=L(x)+e(x)$ with $|e(x)|\le\epsilon$. This premise includes errors from forming and evaluating the table. If $x_1$ minimizes $T$ and $x_*$ minimizes $L$ on $G_0$,

```math
L(x_1)\le L(x_*)+e(x_*)-e(x_1)
       \le\min_{G_0}L+2\epsilon,
\qquad
L(p)\le\min_{G_0}L+2\epsilon+2\eta.
```

With one level the $2\eta$ term is unnecessary. Partial observation replaces $G_0$ by the observed subset. A mode or mean rule need not minimize the table. Exact positive affine scaling $\kappa(L-c)/E$ preserves minimizers for $\kappa,E>0$, but separately rounded tables need not preserve their order. A flat or unresolved first level can fail to supply $x_1$. The API supplies neither a general $\epsilon$ nor $\eta$, so the conditional inequality is not an automatic certificate.

<a id="r49"></a>
### Proposition 49. Marginal refinement and guaranteed joint mass

**Provenance: Improved from a reference. Evidence: proved for any joint distribution on valid grid points, and for finite shots under the stated sampling model.**

**Statement.** A product box whose axis intervals have conditional marginal masses $m_j$ has conditional joint mass at least $\max(0,1-\sum_j(1-m_j))$. For sampled refinement, let $P_z$ be the level's backend-sampled distribution conditioned on valid decoding and the earlier history, and let $B_z$ be its selected old-grid region. Under the premises below, with probability at least $1-\alpha$, all covered levels simultaneously satisfy

```math
P_z(B_z)\ge L_z:=\max\{L_{H,z},L_{CP,z}\}.
```

The two lower bounds use Hoeffding and one-sided Clopper–Pearson (CP), each with half the total failure budget. They cover selection of both the region and the larger bound from the same counts ([Wu et al., arXiv:2605.12066][WuCoverage]). The derivation here specifies the statistical events and the formulas evaluated by `report()`.

For the marginal relation, the complement of a product event is the union of the coordinate complements. Therefore

```math
p_{\rm box}\ge\max\left(0,1-\sum_j(1-m_j)\right).
```

Population marginal masses at least $\eta=1-\delta/d$ imply joint mass at least $1-\delta$. The distribution $\begin{pmatrix}0.8&0.1\\0.1&0\end{pmatrix}$, with index zero selected on both axes, attains the bound with marginals $0.9$ and joint mass $0.8$. Thus neither the threshold $0.9$ nor the product $0.81$ is a joint-mass lower bound. Applied to exact empirical frequencies, the same relation bounds the empirical distribution.

Centered cells put internal box faces at midpoints of neighboring grid coordinates, differing from the left-endpoint cells of [Wu et al., Eq. (14)][Wu]. Ordered resolved coordinates give ordered internal faces, so the next box is nonempty, contains the selected grid points and stays inside the parent box. The statistical event is the selected set of old-grid indices. Periodic refinement uses a nonwrapping interval at the seam fixed by the level's parent box.

#### Finite-shot confidence

Fix $0<\alpha<1$ and a maximum number $H\ge1$ of covered levels before inspecting any counts. Number levels chronologically by $z$, with earlier history $\mathcal F_{z-1}$. The level has $d_z$ coordinates and $K$ grid points per coordinate. Its dimension includes every slack coordinate of an augmented-Lagrangian inner problem. The sampling premises are as follows.

1. Given $\mathcal F_{z-1}$, the grid, coordinate ordering, periodic seam, raw shot count and sampling distribution are fixed before this batch's positions are inspected. They may depend on earlier levels.
2. The raw draws are conditionally independent and identically distributed. Every valid decoded draw gives all $d_z$ coordinates. For one-hot encoding, validity means exactly one excitation in each variable register. For binary encoding, every complete register outcome is valid. Thus $P_z(B)=\Pr(X\in B\mid\text{valid decoding},\mathcal F_{z-1})$.
3. All valid draws in the batch are used. The guarantee does not cover stopping shots according to the positions already seen, selectively discarding valid draws, pooling different sampling populations, correlated shots or drift within the sampled batch.
4. $H$ covers the whole claimed run. Standalone refinement uses `max_levels`. An augmented-Lagrangian run uses `max_iterations * max_levels`, including early termination. A new round's local level numbering does not reset this budget.

Let $S_z$ be the realized number of valid draws. For fixed raw shot count, the joint law factors over raw draws. Conditioning each factor on its validity indicator leaves the valid positions independent with law $P_z$. The formulas can therefore condition on the full validity pattern and use its realized $S_z$, then average over patterns. A level with $S_z=0$, or one never reached, supplies no bound. The simultaneous statement covers levels that occur with valid draws without conditioning on the later event that the whole run finishes successfully.

Each axis has the family of nonempty, nonwrapping intervals

```math
\mathcal I_K=\{\{a,a+1,\ldots,b\}:0\le a\le b\le K-1\},\qquad
I_K=|\mathcal I_K|=\sum_{a=0}^{K-1}(K-a)=\frac{K(K+1)}2.
```

The marginal family has $J_z=d_z I_K$ events and the joint-box family has $N_z=I_K^{d_z}$ events. The greedy interval starts at a largest marginal entry and grows by frontier neighbors, so all its outputs belong to this family. The same families cover the selected mode, growth path, endpoints, split axis, valley and split side. The implementation uses their counts or logarithms without enumerating the boxes. Choosing a seam from the current batch or allowing wrapping intervals would require a different candidate family.

For an ordinary level, $B_z$ is the product of `level.intervals`. At a stall split, its selected axis keeps $[0,v]$ or $[v,K-1]$, including the valley index $v$, and all other axes keep $[0,K-1]$. The stored `joint_mass` still concerns the ordinary intervals, which cover the whole box at a split. The new integer counts describe the selected region itself.

Write $C_{z,j}$ for valid draws whose coordinate $j$ lies in the selected interval, and $M_z$ for draws in their intersection. They come from the same joint observations, so

```math
\max\left\{0,S_z-\sum_j(S_z-C_{z,j})\right\}\le M_z\le\min_j C_{z,j}\le S_z.
```

`RefinementLevel` stores $S_z$ as `valid_count`, $C_{z,j}$ as `region_axis_counts` and $M_z$ as `region_count`. `returned_shots` is the raw returned count. The confidence calculation uses these integers, while `axis_masses`, `joint_mass` and `joint_mass_bound` keep their existing rounded empirical meanings.

#### Hoeffding bound

For a fixed level, axis and candidate interval, its hit indicators are Bernoulli variables under the conditional sampling model. [Hoeffding, Theorem 1 and Eqs. (1.4), (2.3)][Hoeffding] gives

```math
\Pr(|\widehat p-p|>e\mid\mathcal F_{z-1},\text{validity indicators})\le2e^{-2S_z e^2}.
```

For completeness, the centered Bernoulli log moment-generating function $g(t)=\log(1-p+pe^t)-pt$ has $g(0)=g'(0)=0$ and $g''(t)=q_t(1-q_t)\le1/4$, where $q_t=pe^t/(1-p+pe^t)$. Hence $g(t)\le t^2/8$. Independence and exponential Markov give an upper-tail bound $\exp(-tS_z e+S_z t^2/8)$, minimized at $t=4e$. Repeating for the lower tail gives the displayed two-sided bound. The cases $p=0$ and $p=1$ are deterministic.

Allocate $\alpha_H=\alpha/2$ to this method over the whole run. A union bound over all $J_z$ marginal events gives per-level failure probability at most $\alpha/(2H)$ with

```math
\epsilon_z=\sqrt{\frac{\log(4HJ_z/\alpha)}{2S_z}},\qquad
2J_z e^{-2S_z\epsilon_z^2}=\frac{\alpha}{2H}.
```

On this event every selected marginal mass is at least $C_{z,j}/S_z-\epsilon_z$. A whole axis has population mass one exactly. Let $A_z$ contain the axes whose selected interval is not whole and put $q_z=|A_z|$. Applying the complement union bound only to these axes gives

```math
L_{H,z}=\max\left\{0,1-\sum_{j\in A_z}\left(1-\frac{C_{z,j}}{S_z}\right)-q_z\epsilon_z\right\}.
```

The implementation first forms the integer misses $D_z=\sum_{j\in A_z}(S_z-C_{z,j})$, then evaluates $\max(0,\max(0,S_z-D_z)/S_z-q_z\epsilon_z)$. This is the same nonnegative bound. If an individual marginal lower bound is negative, the final joint bound is also zero, so clipping each marginal first gives the same result. A split has $q_z=1$ and pays one radius. The correction still uses $J_z=d_z I_K$ because the active axes were selected from the data.

#### Clopper–Pearson bound and joint selection

For any fixed candidate box $B$, its hit count $M_B$ has distribution $\operatorname{Bin}(S_z,p_B)$ under the same conditioning. The [Clopper–Pearson construction][ClopperPearson] inverts the one-sided binomial test. With $I_x(a,b)$ the regularized incomplete beta function, define

```math
\beta_z=\frac{\alpha}{2HN_z},\qquad
L_{CP}(M,S,\beta)=
\begin{cases}
0,&M=0,\\
I^{-1}_{\beta}(M,S-M+1),&1\le M\le S.
\end{cases}
```

For $M\ge1$, the endpoint solves $\Pr_{p=L_{CP}}[\operatorname{Bin}(S,p)\ge M]=\beta$. The right tail increases with $p$, so test inversion gives $\Pr_p(L_{CP}(M,S,\beta)>p)\le\beta$. A union bound over all $N_z$ candidate boxes makes their CP bounds simultaneous with conditional failure probability at most $N_z\beta_z=\alpha/(2H)$. The selected box is one of them, so $L_{CP,z}=L_{CP}(M_z,S_z,\beta_z)$ is covered even though the same counts selected its coordinates. This is a frequentist lower bound without a prior.

When $M_z=S_z$, the right tail is $p^{S_z}$ and the endpoint has the closed form

```math
L_{CP,z}=\exp\!\left(\frac{\log\beta_z}{S_z}\right),\qquad
\log\beta_z=\log\alpha-\log2-\log H-d_z\log I_K.
```

All observed samples hitting a smaller region still gives a mathematical CP bound strictly below one. If the selected region contains the entire valid grid support, its conditional mass is one independently of sampling, and the report returns one directly.

Let $E_z$ be failure of either method's simultaneous event at level $z$, and take $E_z$ empty when the level is absent or has no valid sample. Each method has conditional failure probability at most $\alpha/(2H)$, so $\Pr(E_z\mid\mathcal F_{z-1})\le\alpha/H$ after averaging over validity patterns. The methods need not be independent. Conditional averaging and the union bound over levels give

```math
\Pr\!\left(\bigcup_{z=1}^{H}E_z\right)
\le\sum_{z=1}^{H}\mathbb E\!\left[\Pr(E_z\mid\mathcal F_{z-1})\right]
\le\alpha.
```

Outside that union, both lower bounds hold at every covered level, and therefore their maximum does too. The dimension and candidate family may vary with earlier history because they are fixed under each level's conditioning. The implementation uses the actual $d_z$, including slack coordinates. It neither reduces $H$ after early stopping nor substitutes a data-selected active-axis count for $d_z$ in either family correction.

#### Sufficient valid-shot budgets

For a common sufficient budget across levels, fix a dimension upper bound $d$ before sampling and use $J=dI_K$ and $\epsilon_S=\sqrt{\log(4HJ/\alpha)/(2S)}$. Let $0<\eta\le1$ and $0<\delta<1$. For an ordinary box whose exact empirical marginal frequencies are all at least $\eta$, Hoeffding gives population mass at least $1-\delta$ whenever $\eta\ge1-\delta/d+\epsilon_S$. For $e>0$ and $\tau=\eta-1+\delta/d>0$, sufficient integer budgets are

```math
S\ge\left\lceil\frac{\log(4HJ/\alpha)}{2e^2}\right\rceil
\quad\text{for }\epsilon_S\le e,\qquad
S\ge\left\lceil\frac{\log(4HJ/\alpha)}{2\tau^2}\right\rceil
\quad\text{for mass at least }1-\delta.
```

The empirical threshold can be at most one when $S\ge\lceil d^2\log(4HJ/\alpha)/(2\delta^2)\rceil$. These are sufficient budgets for this argument, not necessary sample sizes for a particular distribution. When $\tau\le0$, the threshold alone gives no positive-radius guarantee of the requested mass, although the actual counts can give a stronger bound.

For a level of dimension $d_z$ where the selected region receives all valid draws, $M=S$, a target mass $0<c<1$ has the CP sufficient budget

```math
S\ge\left\lceil\frac{\log(\alpha/(2H I_K^{d_z}))}{\log c}\right\rceil.
```

With $K=64$, $d_z=2$, $H=60$, $c=.99$ and $\alpha=.05$, the budget is 2295 valid draws. This result assumes the observed all-hit event and does not promise that a future batch will satisfy it. One-hot validity filtering can also make the valid count smaller than the raw shot count. At $\eta=.9999$ and $S=2295$, the integer threshold $C_j\ge\lceil\eta S\rceil$ still requires every marginal to receive all valid draws.

For general $\eta$, actual integer counts satisfying $C_j\ge\lceil\eta S\rceil$ imply $M\ge\max(0,S-d(S-\lceil\eta S\rceil))$. The ceiling can make this threshold-derived bound decrease locally as $S$ grows. It is not a monotone curve suitable for unqualified binary search. The existing interval rule uses rounded marginal accumulation, so the report always uses the observed integers rather than replacing them by the nominal threshold. The library does not adjust shots from these formulas.

#### Numerical evaluation and interpretation

`_coverage.py` evaluates the formulas in binary64. Logarithms avoid forming $I_K^d$ or a possibly underflowing $\alpha/2$. For $0<M<S$, the inverse uses `scipy.special.betaincinv(M, S-M+1, exp(log_beta))` only when $S+1\le2^{53}$, so both integer shape parameters are exactly representable, and the target beta is at least the smallest normal binary64 value. These representation checks do not supply an inverse error bound. Large shapes and extreme tails can lose relative tail accuracy, and a returned quantile can be subnormal or round to zero.

The $M=0$, all-hit and full-support cases use their separate formulas. Nonfinite inverse values, values outside $[0,1]$, and CP values rounded to one are unavailable. The report then uses the Hoeffding bound with its original half-budget. It does not reallocate the unused CP share after seeing the counts. Ordinary logarithm, square root, exponential and inverse-beta evaluation give numerical approximations to the mathematical bounds, not directed-rounding enclosures. The report marks `numerical_evaluation="binary64"`, uses approximate wording and preserves near-one values in its display. A reported deterministic one denotes the entire valid grid support of the level's conditional distribution.

The bounds concern each level's backend-sampled distribution conditioned on valid decoding. For noisy one-hot readout, a conditional mass of one says nothing about the fraction of raw outcomes that decode validly. The bounds do not cover systematic differences from ideal quantum evolution, global-minimizer preservation or augmented-Lagrangian convergence. Different levels generally sample different populations, so their bounds cannot be multiplied into the final box's mass under the initial distribution. The valley-resolution screen of Proposition 50 has its own failure allowance. A claim combining that screen with region coverage must account for both.


<a id="r50"></a>
### Proposition 50. A sufficient stall condition and the limitation of a valley split

**Provenance: Derived in NWQLib. Evidence: proved for a fixed realized refinement trajectory.**

If both endpoint cells of an axis have marginal mass greater than $1-\eta$, every contiguous interval with mass at least $\eta$ covers the entire axis. A valley split can continue such a stalled search only by relaxing its ordinary mass-preservation condition. If splitting is triggered only when the unsplit search would stop, and the smallest value of one fixed objective comparison is kept over all completed levels, that value cannot be larger than the unsplit trajectory's with the same earlier observations.

**Proof.** An interval omitting the first endpoint has mass below $\eta$, and the same is true if it omits the last. A contiguous interval containing both covers the whole axis. At a full-box stall, the split divides one axis into two regions and leaves every other axis whole. Its kept joint mass is therefore exactly the chosen axis marginal mass, independent of coordinate correlations. Including the valley cell on the chosen side reduces the discarded mass by precisely that cell's marginal mass. It does not make the discarded region harmless.

Every level before the trigger agrees with the unsplit trajectory. The unsplit result is therefore among the candidates from which the split trajectory chooses its best comparison value. Taking a minimum over more candidates proves the inequality. NWQLib compares the relative objective $F-c_*$, where $c_*$ is the exact constant of the first level's support decomposition, fixed throughout the refinement. It sums the same evaluated support terms with the level constant minus $c_*$ for comparison, and with the level constant for reporting. Split-side scores use the same offset. In exact arithmetic this preserves the ordering of $F$. Rounded reported values can conceal differences resolved by the relative comparison. The result concerns a common realized prefix, not independent reruns with new shots. A narrow minimum between evaluated points or in the discarded region can escape the two side scores, so this continuation has no global-optimum preservation guarantee. The library makes it optional and reports its discarded mass.

There is a stronger exact-arithmetic consequence for the implemented greedy interval. It starts at a largest cell and repeatedly adds the larger frontier neighbor until reaching mass $\eta$. Suppose it reaches $\eta$ only after adding the final endpoint, of mass $h$, and let $T$ be the axis's total mass. The selected valley maximizes the smaller-flanking-peak to valley ratio, breaking equal ratios by the larger smaller peak. Among valleys passing an acceptance rule that becomes no harder when peaks increase or the valley decreases, each strict side of this selected valley has mass at least $h$. Hence either side with the valley included has mass at most $T-h<\eta$.

To prove the side bound, let $m$ be the greedy start and $e$ the last endpoint. Both have mass at least $h$. A valley between them has one on each strict side. Otherwise they lie on the same side. If the opposite strict side had total mass below $h$, its maximum would be below $h$. At the step when the greedy interval added that valley, the frontier cell $w$ toward $e$ had mass at most the valley mass. It lay between $m$ and $e$, so both its flanking maxima were at least $h$. It therefore had a strictly larger ratio, or a larger smaller peak when both valley masses were zero, and also passed the monotone acceptance rule. This contradicts the selected valley. The conclusion concerns exact sums. A rounded running sum can trigger a stall after the exact threshold was already reached, so its strict $<\eta$ conclusion does not automatically hold for a binary64 trajectory.

For counts, the implementation also screens whether the valley is statistically resolved. With at most $H$ adaptive levels, $d$ axes, $K$ cells, $N$ valid independent shots in the current conditional population, and failure allowance $0<\alpha<1$, the analytic screen is

```math
\min(C_L,C_R)-c_v>\sqrt{2N\log(2HdK/\alpha)}.
```

Here $C_L,C_R$ are flanking-peak counts and $c_v$ the valley count. Hoeffding for cell indicators and a union bound give simultaneous cell error at most $\epsilon=\sqrt{\log(2HdK/\alpha)/(2N)}$. A selected peak-to-valley population difference is therefore positive when its empirical difference exceeds $2\epsilon$, giving the count rule. The implementation enlarges the computed logarithm by $1+2^{-40}$ and compares squared quantities as exact rationals, a conservative numerical realization under the elementary-function accuracy assumption. The union bound covers selection after inspecting the cells and, conditional on each previous history, selection of later levels. It requires fresh shots from the level's conditional population. A resolved valley is evidence of population shape, not of objective quality in the discarded region.

<a id="r51"></a>
### Proposition 51. A distance bound for the augmented-Lagrangian and refinement composition

**Provenance: Derived in NWQLib. Evidence: proved for the stated quadratic problem and inner-selection premise.**

Consider $f(x,y)=(x-11/16)^2+(y-7/16)^2$, equality $4(x+y-1/8)=0$, and box $[-1,1]^2$. With scales $s_f=2$, $s_h=4$, the unique KKT point is $x_*=(3/16,-1/16)$, with normalized multiplier $1/2$. For a round entering with $\lambda\in[-1,1]$, $\rho\in[1,4]$, put

```math
v=(1,1)^T,\quad M=1+2\rho,\quad H=I+\rho vv^T,\quad m=x_*+\frac{1/2-\lambda}{M}v.
```

Let $G_0=\{-3/4,-1/2,\ldots,3/4\}^2$ be the first grid, and define

```math
\Delta=\min_{q\in G_0}\tfrac12(q-m)^TH(q-m).
```

If the first level selects a minimum of the analytic round objective over $G_0$, and the round selects its best completed level by that same objective, its returned point $p$ satisfies

```math
\|p-x_*\|\le\frac{\sqrt2\,|1/2-\lambda|}{M}+\sqrt{2\Delta},\qquad \Delta\le M/64.
```

**Proof.** The normalized round objective is $L=f/2+\lambda(x+y-1/8)+\rho(x+y-1/8)^2/2$. Differentiation gives its Hessian $H$ and minimizer $m$. The eigenvalues of $H$ are one and $M$. The multiplier and penalty ranges place $m$ in $[1/48,11/16]\times[-11/48,7/16]$, within half a grid spacing of some point in $G_0$ on each axis. The spacing is $1/4$, so

```math
\min_{q\in G_0}(L(q)-L(m))\le\tfrac12M\,2(1/8)^2=M/64.
```

The exact quadratic identity is $L(z)-L(m)=\tfrac12(z-m)^TH(z-m)$. Best-level selection gives $L(p)\le\min_{q\in G_0}L(q)$, irrespective of whether later boxes contain $m$. Thus $\|p-m\|\le\sqrt{2\Delta}$, and the triangle inequality proves the result.

Writing $e=p-m$, the equality update also gives

```math
\lambda^+-\tfrac12=\frac{\lambda-1/2}{M}+\rho v^Te,\qquad |v^Te|\le2\sqrt{\Delta/M}.
```

The last bound is Cauchy–Schwarz in the $H$ inner product, since $v^TH^{-1}v=2/M$. Projection back to $[-1,1]$ cannot increase distance to $1/2$. Therefore the next multiplier error is at most $|\lambda-1/2|/M+2\rho\sqrt{\Delta/M}$, also capped by $3/2$. The final best round uses its own entering multipliers and penalty in the distance bound. Exact readout alone does not establish the premise that every grid point is observed or that rounded table minimizers minimize the analytic quadratic. The repository check verifies these premises independently by rational enumeration.

<a id="r52"></a>
### Proposition 52. Budgeted Clifford replacement and the logarithmic T estimate

**Provenance: Derived in NWQLib. Evidence: proved replacement and allocation bounds, followed by an explicitly conditional cost estimate.**

Let $\delta=\theta-k\pi/2$ be the distance to a nearest Clifford rotation angle, with $|\delta|\le\pi/4$. The phase-minimized operator distance is

```math
d_C=2\sin(|\delta|/4).
```

Let $E$ be the positive finite binary64 synthesis budget and $N$ the initial arbitrary-rotation count. If $N=0$, the replacement threshold and per-rotation tolerance are `None` and the replacement error term is zero. For $N>0$, candidate Clifford replacements are tested against the exact rational share $q=E/N$. If $B$ replacements have exact accumulated distance bound $C$ and $M=N-B$ arbitrary rotations remain, publish

```math
 E_C=\operatorname{up64}(C),\qquad
 R=\operatorname{down64}(E-E_C).
```

For $M>0$, the per-rotation synthesis tolerance is

```math
 \epsilon_{\rm rot}=\operatorname{down64}(R/M).
```

The subtraction and division are formed from exact rational values before their directed conversion. The synthesis allowance in the error budget is the same stored $R$. Consequently the stored fields satisfy

```math
 E_C+M\epsilon_{\rm rot}\le E_C+R\le E.
```

With no remaining arbitrary rotations, no per-rotation tolerance is needed. If $M>0$ but the downward-rounded tolerance is zero, `circuit_resources` and `run_resources` refuse that budget, and the Plan is unaffected. The stored per-rotation tolerance can be below the initial exact share $q$ because both directed roundings preserve the total budget.

**Proof.** Multiplying by the inverse Clifford leaves eigenvalues $e^{\pm i\delta/2}$. The optimal common phase bisects them, giving $d_C$. A replacement is therefore possible whenever $|\delta|\le4\arcsin(\epsilon/2)$.

Each replacement is accepted with distance at most $q$, so $C\le Bq\le E$. Because $E$ itself is representable, the least representable value at least $C$ also satisfies $E_C\le E$. Thus $E-E_C\ge0$. Downward rounding gives $0\le R\le E-E_C$ and, for positive $M$, $0\le\epsilon_{\rm rot}\le R/M$. Multiplication by the positive integer $M$ and addition of $E_C$ prove both inequalities.

For ideal unrounded bookkeeping one has

```math
 \frac{E-C}{N-B}\ge\frac EN
 \quad\Longleftrightarrow\quad
 N(E-C)\ge E(N-B)
 \quad\Longleftrightarrow\quad
 C\le B E/N.
```

This proves only the exact remaining share, not the stored one. Replacing $C$ by its upward-rounded value $E_C$ and applying two downward conversions can reduce the published share. If $N=0$, neither $q$ nor a per-rotation tolerance is required. If $M=0$, the replacement error term alone is at most $E$. A unitary-distance cap on the replacement-error entry cannot increase that term. A synthesis law evaluated at the stored tolerance is still an estimation model, not proof that an arbitrary external compiler attained it.

For a fixed number $M>0$ of remaining arbitrary rotations and a real available budget $R>0$ with $R/M<1$, the model $r(\epsilon)=a\log_2(1/\epsilon)+b_0$, $a>0$, obeys

```math
 \sum_{j=1}^{M}r(\epsilon_j)
 \ge M\!\left[a\log_2\!\left(\frac{M}{\sum_j\epsilon_j}\right)+b_0\right]
 \ge M[a\log_2(M/R)+b_0],
```

provided $0<\epsilon_j<1$ and $\sum_j\epsilon_j\le R$. The first inequality is Jensen's inequality for the convex function $-\log$, and the second uses the decreasing cost as the total allowance grows. Equality in the real allocation problem uses $\epsilon_j=R/M$. Stored downward rounding can leave unused margin. When $R=0$ with rotations remaining, the logarithmic allocation formula is inapplicable, and `circuit_resources` and `run_resources` refuse that budget, with the Plan unaffected.

NWQLib uses $a=3$, $b_0=0$, the leading typical-rotation term of [Ross and Selinger][RossSelinger], and adds exact T gates already in the decomposition. The resulting number is an estimate, not a finite synthesis bound. Repeated structured angles do not become typical random instances merely because there are many of them. Standalone phase minimization also does not justify a controlled reuse without consistent phase representatives.

<a id="r53"></a>
### Proposition 53. Nonnegative one-hot preparation and a finite product-state error

**Provenance: Derived in NWQLib. Evidence: proved circuit identity and conditional numerical bounds.**

For nonnegative normalized amplitudes $\alpha_0,\ldots,\alpha_{K-1}$, let $r_m=(\sum_{j\ge m}\alpha_j^2)^{1/2}$. A one-hot chain with

```math
\theta_m=2\operatorname{atan2}(r_m,\alpha_{m-1})
```

prepares $\sum_j\alpha_j|e_j\rangle$. If the last positive amplitude has index $L$, the complete chain uses $3L$ CX gates. The implemented lower-range cutoff can emit a shorter prefix, with the error term below.

**Proof.** Initially excite qubit zero. Before link $m$, the unassigned amplitude is $r_{m-1}|e_{m-1}\rangle$. A controlled $R_y(\theta_m)$, followed by CX from qubit $m$ back to $m-1$, replaces it by $\alpha_{m-1}|e_{m-1}\rangle+r_m|e_m\rangle$. Induction proves the state. Each controlled rotation uses two CX gates, followed by the transfer CX. When $r_m=0$, the target has already been reached and all later controlled operations are unnecessary. This generalizes the uniform start of Leng et al., Algorithm 1, to kinetic-ground and Gaussian product starts.

For the cutoff, read the stored nonnegative vector $v$ exactly, put $N_v=\|v\|>0$ and $r_m=(\sum_{i\ge m}v_i^2)^{1/2}$. At the first link $m$ whose computed controlled-$R_y$ half-angle is below $\nu_{\rm fp}$, including a rounded zero from a positive tail, the implementation stops before the link. The ideal prefix prepares $v_i/N_v$ for $i<m-1$, $r_{m-1}/N_v$ at $m-1$, and zero later. Its distance from the full target is

```math
D_m=\frac{\sqrt{(r_{m-1}-v_{m-1})^2+r_m^2}}{N_v}=\frac{r_m}{N_v}\sqrt{\frac{2r_{m-1}}{r_{m-1}+v_{m-1}}}\le\frac{\sqrt2\,r_m}{N_v}.
```

The second equality uses $r_{m-1}^2-v_{m-1}^2=r_m^2$. Let $L_m=\sum_{i\ge m}v_i\ge r_m$ be summed exactly. The normalization bound gives $N_v\ge1-\eta_K$, where $\eta_K=3u/(1-2u)+\lceil\sqrt K\rceil\lambda$. With an upper rational $\sqrt2_{\rm up}\ge\sqrt2$, an outward error term is

```math
\Delta_m=\min\left(\sqrt2_{\rm up},\frac{\sqrt2_{\rm up}L_m}{1-\eta_K}\right)\quad\text{when }\eta_K<1.
```

If a positive denominator is unavailable, $\sqrt2_{\rm up}$ bounds the distance between the two nonnegative unit vectors. A prefix ending before link $m$ has $m-1$ links and $3(m-1)$ CX gates. The same prefix determines the builder and rotation count. Per-register error terms add by tensor-product telescoping and enter `state_preparation` once, before its cap at two. This cutoff error term is a bound, while the combined preparation entry remains a first-order estimate because of its chain-roundoff terms. The stored amplitudes and the direct classical start continue to describe the complete vector.

For a shifted Gaussian, let exact amplitudes be $t_i=e^{-y_i-\Delta_i}$, with $|\Delta_i|\le x_i$, and computed normal amplitudes $v_i=\operatorname{fl}(e^{-y_i})$. Under a one-ulp exponential assumption,

```math
|t_i-v_i|\le v_i\frac{\operatorname{expm1}(x_i)+2u}{1-2u}.
```

This follows from $|e^{-\Delta_i}-1|\le e^{x_i}-1$ and $e^{-y_i}\le v_i/(1-2u)$. Subnormal entries need an absolute-ulp version. If these entry bounds imply $\|t-v\|\le\beta\|v\|$, normalization changes direction by at most $2\beta$, by Result 30's normalization argument. A common exponent shift prevents all entries from underflowing without changing the exact direction.

If each computed axis vector differs from its exact unit vector by at most $e_j$, telescoping tensor products gives error at most $\prod_j(1+e_j)-1$. Let $N_j$ bound the norm of computed axis $j$. Each product entry has $d-1$ multiplications, adding at most $\gamma_{d-1}\prod_jN_j$ and an absolute gradual-underflow term. For $D=K^d$, the implementation counts the latter as

```math
U_D=(d-1)\lceil\sqrt D\rceil\lambda,\qquad \lceil\sqrt D\rceil=\operatorname{isqrt}(D-1)+1.
```

This dominates the $\sqrt D(d-1)\lambda/2$ absolute product allowance. It is formed by one correctly rounded quotient of the integers $(d-1)\lceil\sqrt D\rceil$ and $2^{1074}$, without converting $D$ to a float. The surrounding outward factor covers that rounding. The nonnegative-vector cap uses a computed-norm bound $N$ to give distance at most $\sqrt{N^2+1}$. A dimension beyond binary64, such as $4^{520}$, therefore need not prevent evaluation of this bound. The function raises a named range error, rather than publishing an infinite record, whenever $U_D$ or the returned bound in units of $u$ is not finite. When $(d-1)\lceil\sqrt D\rceil$ dominates, the returned bound leaves the binary64 range before $U_D$ does, so at $K=4$ the error starts at $d=2035$. This is a metadata calculation and does not allocate the $D$ amplitudes. The uniform-start first-order allowance is separately $2u$. The elementary-function assumptions remain part of each bound's applicability.

<a id="r54"></a>
### Proposition 54. Slack variables for inequality constraints

**Provenance: Derived in NWQLib. Evidence: proved identities and bounds with exact counterexamples, and one computed comparison of two evolutions.**

**Statement.** Result 46 writes each PHR term as a minimum over a slack $z\ge0$. When that slack becomes an explicit coordinate $s_j$ of the inner objective, minimizing over it reproduces the PHR term exactly, also under an upper cap that contains the minimizer. The proposition has four parts.

1. [Partial minimization over a nonnegative slack](#partial-minimization-over-a-nonnegative-slack) gives this identity, the smallest common cap $U_j^*$ on a domain, and the outer update. The PHR update at the projected point keeps the outer map of [Birgin and Martínez, Algorithm 4.1 and Eqs. (4.7)–(4.9)][BM], without making that algorithm's convergence hypotheses automatic. An update from the sampled slack differs from it by $\rho(s_j-s_j^*(x))$.
2. [Exact slack-grid excess and endpoint errors](#exact-slack-grid-excess-and-endpoint-errors) shows that a finite slack grid adds a nonnegative error to the round objective $L$ of Result 46, bounds it for each grid convention, and adds it to the inner-minimization error in the approximate-minimizer premise of Proposition 47.
3. [A lower bound from support tables](#a-lower-bound-from-support-tables) shows that support tables supply a sufficient cap on a grid, exact for disjoint supports on their Cartesian product. A lower bound from a finite grid need not hold on the continuous box or on later refinement grids.
4. [A periodic slack grid defines a different search kinetic](#a-periodic-slack-grid-defines-a-different-search-kinetic) shows that a periodic kinetic couples the two ends of the slack interval. For any slack axis, the identity gives no equivalence between a QHD evolution in the joint coordinates and an evolution under $L$.

#### Partial minimization over a nonnegative slack

Let $F=f/s_f$, $H_i=h_i/s_{h_i}$ and $G_j=g_j/s_{g_j}$, with positive scales. Let $\rho>0$, $\bar\mu_j\ge0$ and $\bar\lambda_i\in\mathbb R$. Equality multipliers may have either sign, as in Result 46. All functions are finite on the domain under consideration. Write

```math
A(x)=F(x)+\sum_i\bar\lambda_iH_i(x)+\frac\rho2\sum_iH_i(x)^2,
\qquad
P_j(t)=\frac{[\bar\mu_j+\rho t]_+^2-\bar\mu_j^2}{2\rho},
```

so that the round objective of Result 46 is $L(x)=A(x)+\sum_jP_j(G_j(x))$. For an inequality converted to slack form, introduce a separate normalized slack coordinate $s_j\ge0$. Its contribution is

```math
\phi_j(x,s_j)=\bar\mu_j(G_j(x)+s_j)+\frac\rho2(G_j(x)+s_j)^2.
```

Then

```math
s_j^*(x)=\left[-G_j(x)-\frac{\bar\mu_j}{\rho}\right]_+,
\qquad \min_{s_j\ge0}\phi_j(x,s_j)=P_j(G_j(x)),
\qquad
\bar\mu_j+\rho(G_j(x)+s_j^*(x))=[\bar\mu_j+\rho G_j(x)]_+.
```

For a nonempty domain $\mathcal D$, let $\gamma_j=\inf_{x\in\mathcal D}G_j(x)>-\infty$. The smallest common upper cap that includes every $s_j^*(x)$ is

```math
U_j^*(\bar\mu_j,\rho)=\left[-\gamma_j-\frac{\bar\mu_j}{\rho}\right]_+.
```

Thus $\min_{0\le s_j\le U_j}\phi_j=P_j(G_j)$ everywhere on $\mathcal D$ if and only if $U_j\ge U_j^*$. The domain can be the finite grid or the continuous box. For a finite grid the infimum is a minimum. For a continuous function on a compact box it is also a minimum. The multiplier-independent choice $U_j=[-\gamma_j]_+$ suffices for every nonnegative entering multiplier and every positive penalty.

**Proof.** Completing the square gives

```math
\phi_j(x,s)=\frac\rho2\left(s+G_j(x)+\frac{\bar\mu_j}{\rho}\right)^2
-\frac{\bar\mu_j^2}{2\rho}.
```

The unconstrained minimizer is $-G_j-\bar\mu_j/\rho$. Projection onto $[0,\infty)$ gives $s_j^*$ and substitution gives the PHR expression. If $G_j+\bar\mu_j/\rho\ge0$, then $s_j^*=0$ and the last identity is immediate. Otherwise $s_j^*=-G_j-\bar\mu_j/\rho$ and both sides of that identity vanish. Strict convexity in $s$ makes this minimizer unique. A bounded interval reproduces the minimum precisely when it contains the minimizer. Taking the supremum over $x$ yields the cap formula. Nonnegativity of $\bar\mu_j/\rho$ proves the multiplier-independent upper bound.

For a set $\mathcal C$ of converted inequalities, define the inner objective

```math
\mathcal L(x,s)=A(x)+\sum_{j\notin\mathcal C}P_j(G_j(x))
+\sum_{j\in\mathcal C}\phi_j(x,s_j).
```

The slack coordinates separate at fixed $x$, so their individual minima add and $\min_s\mathcal L(x,s)=L(x)$. This permits a mixture of PHR and slack terms within a round. Every admissible $s\ge0$ also satisfies $\mathcal L(x,s)\ge L(x)$, even when an upper cap is too small to attain equality.

A selected inner point $(x,s)$ can update the outer state in two ways. The first projects it to its original coordinates $x$, evaluates the original $f$, $h$ and $g$ there, and applies the tentative updates, safeguard and penalty rule of Result 46 and the residual tests of Proposition 47. Neither $G_j(x)+s_j$ nor the sampled slack then replaces an original constraint residual. The second would update each multiplier from the sampled slack. This update and its difference from the PHR update $\mu_j^+=[\bar\mu_j+\rho G_j(x)]_+$ are

```math
\widetilde\mu_j^+=\bar\mu_j+\rho(G_j(x)+s_j),\qquad
\widetilde\mu_j^+-\mu_j^+=\rho(s_j-s_j^*(x)).
```

The two agree only when the selected slack is its continuous conditional minimizer. With $\bar\mu=0$, $\rho=1$ and $G(x)=-1$, the PHR update is zero. Sampling $s=0$ gives $\widetilde\mu^+=-1$, whereas sampling $s=2$ gives $\widetilde\mu^+=1$. Clipping a negative result does not correct the second discrepancy. The second update would instead require an equality-constrained augmented-Lagrangian formulation in $(x,s)$, with its own multipliers, lower-level slack bounds and inner accuracy conditions. It would not be the inequality algorithm of Result 46.

The PHR update at the projected point keeps the outer map of [Birgin and Martínez, Algorithm 4.1 and Eqs. (4.7)–(4.9), pp. 33–34][BM]. It preserves access to that algorithm's conditional convergence results when their inner-solve and other hypotheses are met. It does not make those hypotheses automatic. For example, in [Birgin and Martínez, Assumption 5.1 and Theorem 5.2, pp. 41–42][BM], the assumption requires an approximate global minimizer of the PHR subproblem on the hard domain, and the theorem requires its error to tend to zero, feasibility of the original problem, the algorithm's bounded multiplier safeguards, and the specified penalty behavior. The book's Chapter 6 uses approximate first-order conditions instead, in its Assumption 6.1, p. 48. Finite QHD evolution, a fixed grid, a sampled mode, optional safeguards and a capped penalty do not generally establish these premises. The slack-grid error of the next part must be included in any inner-error claim.

#### Exact slack-grid excess and endpoint errors

Fix $x$, one converted inequality and a cap $U>0$ containing its continuous minimizer $t=s^*(x)\in[0,U]$. Put $r=[\bar\mu+\rho G(x)]_+$. Let $\mathcal S\subset[0,U]$ be a nonempty slack grid. All expressions in this part use exact arithmetic. Then

```math
\phi(x,s)-P(G(x))=\frac\rho2(s-t)^2+r s,
\qquad
e(x):=\min_{s\in\mathcal S}\phi(x,s)-P(G(x))\ge0.
```

If $t>0$, then $r=0$ and $e(x)=\rho\operatorname{dist}(t,\mathcal S)^2/2$. If $t=0$, the smallest grid value $a=\min\mathcal S$ is optimal and

```math
e(x)=r a+\frac\rho2a^2.
```

For $K_s\ge2$ equally spaced points, the three grid conventions give the following bounds.

| Slack grid | Spacing and points | Bound for all $t\in[0,U]$ | Error at $t=0$ | Error at $t=U>0$ |
| --- | --- | --- | --- | --- |
| Dirichlet interior | $h_s=U/(K_s+1)$, $h_s,\ldots,K_sh_s$ | $e(x)\le r h_s+\rho h_s^2/2$ | $r h_s+\rho h_s^2/2$ | $\rho h_s^2/2$ |
| Dirichlet with boundary points, endpoint inclusive | $h_s=U/(K_s-1)$, $0,h_s,\ldots,U$ | $e(x)\le\rho h_s^2/8$ | $0$ | $0$ |
| Periodic, endpoint exclusive | $h_s=U/K_s$, $0,h_s,\ldots,U-h_s$ | $e(x)\le\rho h_s^2/2$ | $0$ | $\rho h_s^2/2$ |

The endpoint-inclusive bound is a worst-case bound. Its zero endpoint errors are exact. For either grid that omits $U$, the upper-endpoint error is quadratic, since at $t=U>0$ the derivative of $\phi$ with respect to the slack is zero. Omitting zero is different. When $t=0$, that derivative can be positive, and the linear term $r h_s$ cannot be dropped.

**Proof.** Expanding the quadratic about its constrained minimizer gives

```math
\phi(x,s)-\phi(x,t)=\frac\rho2(s-t)^2+
\bigl(\bar\mu+\rho(G(x)+t)\bigr)(s-t).
```

The coefficient in parentheses is $r$, and $rt=0$, proving the identity. For $t>0$, choose a nearest grid node. An endpoint-inclusive mesh has covering radius $h_s/2$ on $[0,U]$. An interior mesh has distance at most $h_s/2$ between its first and last nodes and at most $h_s$ near either omitted endpoint. A periodic endpoint-exclusive mesh has distance at most $h_s/2$ on $[0,U-h_s]$ and at most $h_s$ on the final interval. These are distances on the numerical interval, not circular distances. For $t=0$, the derivative of $rs+\rho s^2/2$ is nonnegative for $s\ge0$, so the first node minimizes it. Substituting the endpoint distances proves every entry of the table.

For example, take $U=1$, $K_s=4$, $\rho=2$, $\bar\mu=1$ and $G(x)=0$. The interior grid starts at $1/5$, so the error is $1/5+1/25=6/25$, whereas both grids containing zero have error zero. At the other end, take $\bar\mu=0$ and $G(x)=-1$, so $t=U=1$. The errors are respectively $1/25$, zero and $1/16$.

This error enters the approximate-minimizer premise of Proposition 47 as follows. Let $X$ be a finite grid of original variables, and give each converted inequality $j\in\mathcal C$ a slack grid $\mathcal S_j$ with error $e_j$. At fixed $x$, minimization separates over the slack coordinates, so

```math
\min_{s\in\prod_j\mathcal S_j}\mathcal L(x,s)
=L(x)+E(x),\qquad E(x)=\sum_{j\in\mathcal C}e_j(x).
```

Define $E_{\max}=\max_{x\in X}E(x)$. The bounds below can replace this maximum without enumerating $X$. Suppose the selected inner point obeys

```math
\mathcal L(\widehat x,\widehat s)
\le\min_{x\in X,\ s\in\prod_j\mathcal S_j}\mathcal L(x,s)+\varepsilon_{\rm in},
\qquad E(x)\le E_{\max}\quad(x\in X).
```

Then its projection is an approximate minimizer of the round objective with

```math
L(\widehat x)\le\min_{x\in X}L(x)+\delta,
\qquad \delta=\varepsilon_{\rm in}+E_{\max}.
```

**Proof.** Let $x_*$ minimize $L$ on $X$. The chain

```math
L(\widehat x)
\le\mathcal L(\widehat x,\widehat s)
\le\min_{x\in X}(L(x)+E(x))+\varepsilon_{\rm in}
\le L(x_*)+E(x_*)+\varepsilon_{\rm in}
```

proves the result and also gives the sharper error $\varepsilon_{\rm in}+E(x_*)$ when that quantity is known. For a uniform interior-grid bound, an upper bound $M_j\ge\sup_XG_j$ gives

```math
E_{\max}\le\sum_{j\in\mathcal C}
\left(h_{s,j}[\bar\mu_j+\rho M_j]_++\frac\rho2h_{s,j}^2\right).
```

The other two grid types use the sums of their quadratic bounds in the table. The entering multipliers and penalty are those of this round.

At a feasible projected point, the proof of Proposition 47 therefore gives, for every feasible $y\in X$,

```math
F(\widehat x)-F(y)
\le\delta-\sum_jP_j(G_j(\widehat x))
\le\delta+\sum_j\frac{\bar\mu_j^2}{2\rho}.
```

Multiply by $s_f$ for original objective units. Feasibility and the inner-selection inequality are still premises. Neither a QHD mode nor exact probability readout alone supplies the latter. With a table error bounded by $\varepsilon_T$ on the relevant grid, selecting a table minimum contributes $2\varepsilon_T$ to $\varepsilon_{\rm in}$. Comparing reported relative values with errors bounded by $\varepsilon_R$ contributes another $2\varepsilon_R$. This follows by adding the error at the winner and at the comparison point. Search-model table errors must first be converted into $\mathcal L$ units by the positive factor $E/\kappa$ of Proposition 48, with transformation errors included.

Proposition 51 is an equality-only quadratic example. It has no inequality slack, so its own $E_{\max}$ is zero. Its inner-selection premise nevertheless has a useful error version. If, for the quadratic and the $x_*,m,H,M,\Delta$ of that proposition, the returned point $p$ satisfies

```math
L(p)\le\min_{q\in G_0}L(q)+\delta,
```

then the quadratic identity gives $\tfrac12(p-m)^TH(p-m)\le\Delta+\delta$, and the proof of Proposition 51 gives

```math
\|p-x_*\|\le\frac{\sqrt2\,|1/2-\lambda|}{M}+\sqrt{2(\Delta+\delta)},
\qquad
|\lambda^+-1/2|\le\frac{|\lambda-1/2|}{M}
+2\rho\sqrt{\frac{\Delta+\delta}{M}}.
```

The first inequality uses $H\succeq I$ and the triangle inequality. The second uses $v^TH^{-1}v=2/M$ and Cauchy–Schwarz in the $H$ inner product. This shows where an established inner error enters. It does not extend that quadratic's Hessian or KKT-distance theorem to arbitrary inequality problems.

The grid adds the nonnegative function $E(x)$ to the round objective. It does not always move the selected $x$ toward the feasible interior. For $t>0$, the error oscillates as the desired slack passes between nodes. Near the omitted upper endpoint it penalizes points requiring a large slack, which can favor points closer to the inequality boundary. The endpoint-inclusive and periodic grids have no lower-endpoint error at all.

The missing zero of an interior grid has a precise interpretation. If only the lower bound $s\ge a>0$ is imposed and the upper cap does not bind, substitute $s=a+u$ to obtain

```math
\min_{s\ge a}\phi(x,s)=P(G(x)+a).
```

The PHR transition moves from $G=-\bar\mu/\rho$ to $G=-\bar\mu/\rho-a$. This is an inward shift of $a$ in the normalized constraint coordinate, or $s_ga$ in the original residual. It is not a universal displacement bound on the optimizer. For the one-dimensional example $G(x)=x$, $F(x)=-cx$, $c>0$, $\bar\mu=0$, with bounds containing the minimizers, the minimizer of $L$ is $x=c/\rho$. If the smallest available slack is $a$, the inner objective is minimized at $s=a$, $x=c/\rho-a$. Differentiating $-cx+\rho(x+s)^2/2$ proves that exact shift. For general problems, near-tied minima can exchange order under an arbitrarily small error. A spatial bound requires an additional growth condition. If $L(x)-L(x_c)\ge\sigma\|x-x_c\|^2/2$ on $X$ for a point $x_c$ and some $\sigma>0$, then

```math
\|\widehat x-x_c\|\le\sqrt{\frac{2(\Delta_x+\delta)}{\sigma}},
\qquad \Delta_x=\min_X L-L(x_c).
```

This follows by combining the growth inequality with the established inner-value bound.

#### A lower bound from support tables

On the grid $X$, suppose the normalized inequality has an exact additive representation

```math
G_j(x)=c_{0,j}+\sum_{A\in\mathcal A_j}T_{j,A}(x_A).
```

The set $A$ is a variable support, $S_j=\bigcup_{A\in\mathcal A_j}A$ is the inequality's full support, and each table lists all assignments of its variables on their grid. Define

```math
\ell_j=c_{0,j}+\sum_A\min T_{j,A},\qquad
M_j=c_{0,j}+\sum_A\max T_{j,A}.
```

Then $\ell_j\le G_j(x)\le M_j$ on $X$, and either of the caps

```math
U_j=[-\ell_j]_+,
\qquad
U_{j,k}=[-\ell_j-\bar\mu_j^k/\rho_k]_+
```

is a sufficient cap on that grid, that is, at least $U_j^*$ there, within its stated multiplier scope. The first works for all nonnegative multipliers. The second, with the entering multiplier $\bar\mu_j^k$ and penalty $\rho_k$ of round $k$, is a potentially smaller cap for that round.

**Proof.** Each table entry is at least its table's minimum and at most its maximum. Adding these inequalities and the constant gives the bounds. Hence

```math
s_j^*(x)=[-G_j(x)-\bar\mu_j/\rho]_+
\le[-\ell_j-\bar\mu_j/\rho]_+\le[-\ell_j]_+.
```

The clipping function is increasing, which justifies both inequalities. The minima of overlapping support terms need not be attained at the same point, so the first bound can be strict. For example, on $\{0,1\}^2$, $G(x,y)=(x-y)^2=x^2+y^2-2xy$ has minimum zero, but its three support minima sum to $-2$.

If the supports are disjoint and the domain is their Cartesian product, their minimizing assignments can be combined. In that case the bound is exact. In particular, for an additively separable $G_j=c_0+\sum_iq_i(x_i)$, group all terms of the same variable first and minimize each one-dimensional table. This gives the exact grid minimum in $K|S_j|$ table entries rather than $K^{|S_j|}$ entries. It does not require that each $q_i$ be linear. An inequality represented by one irreducible support still needs all $K^{|S_j|}$ entries of that support's table for this bound, so this construction does not make every high-support constraint cheap to bound.

A lower bound from a finite grid is not a lower bound on the continuous box or on every later refinement grid. On $[0,1]$, take $G(x)=(x-1/2)^2-1/100$. Its two interior-grid values at $1/3,2/3$ are both $4/225>0$, but $G(1/2)=-1/100$. The grid bound would give $U=0$ even though a continuous or later-grid point can need positive slack. To claim equivalence throughout the continuous box, replace table extrema with certified bounds on the support functions throughout their continuous domains. Affine constraints admit such bounds directly from endpoint values. General nonlinear continuous bounds require additional mathematics or a user-supplied valid bound.

Stored binary64 tables also require an arithmetic qualification. If $|T_{j,A}-\widetilde T_{j,A}|\le e_{j,A}$ and $c_{0,j}\ge\widetilde c_{0,j}-e_{0,j}$ on the relevant domain, then

```math
\ell_j^{\rm safe}=\widetilde c_{0,j}-e_{0,j}
+\sum_A\left(\min\widetilde T_{j,A}-e_{j,A}\right)
```

is a valid mathematical lower bound. A binary64 evaluation must round this sum downward and the resulting cap upward. An outward final sum alone does not bound the errors in symbolic normalization or table evaluation. Without those error bounds, the result is a bound for the tabulated representation only. This is the same distinction between mathematical and evaluated objectives made in Propositions 47–48.

An oversized cap enlarges the slack spacing at fixed $K_s$. If $U$ is multiplied by $c\ge1$, each quadratic bound in the slack-grid table grows by $c^2$. The interior-grid linear term grows by $c$. Thus an easily computed but loose lower bound can erase the accuracy of the slack grid at a fixed grid size.

A zero cap $U=\max(0,-\min G)=0$ means that $G\ge0$ on the stated domain, not that the inequality is never active. The constant $G\equiv1$ is infeasible there, whereas $G\equiv0$ is active everywhere. In either case $s^*=0$, and the PHR term on that domain is exactly

```math
P(G)=\bar\mu G+\frac\rho2G^2.
```

A strictly positive lower bound establishes infeasibility only on the domain that the bound covers. A sign test on a grid alone justifies neither a continuous-domain infeasibility conclusion nor this quadratic form on later grids.

For comparison, an upper bound $M_j<0$ certifies strict inactivity of the original inequality everywhere on the stated domain. A sufficient condition for its PHR term to be constant there is the round-dependent test $M_j+\bar\mu_j/\rho\le0$, in which case the term equals $-\bar\mu_j^2/(2\rho)$. A strictly inactive inequality can still have a nonconstant PHR term when the entering multiplier is positive. Likewise, the PHR term equals the quadratic form above if $\ell_j+\bar\mu_j/\rho\ge0$. Both tests are sufficient conditions, each valid on the domain that its bound covers.

#### A periodic slack grid defines a different search kinetic

Let the slack grid points be $s_i=ih_s$, $i=0,\ldots,K_s-1$, with $h_s=U/K_s$. Use the periodic finite-difference kinetic of the binary encoding, Result 37. The potential values are $\phi(x,s_i)$, without imposing periodicity on the quadratic function itself. Then this is a well-defined finite Hermitian Hamiltonian, but its kinetic couples the two ends of the slack interval. The continuous slack-elimination identity supplies no equivalence between its QHD dynamics and an evolution with potential $L$ on $x$ alone.

**Proof.** A real diagonal potential plus a real symmetric kinetic matrix is Hermitian for any finite potential values. On this grid the kinetic is

```math
T_{\rm per}=\frac{2I-S-S^\dagger}{2h_s^2},
```

where $S$ is the cyclic shift. It couples $s=0$ directly to $s=U-h_s$. The potential difference between these two points is

```math
\phi(x,U-h_s)-\phi(x,0)
=(\bar\mu+\rho G(x))(U-h_s)+\frac\rho2(U-h_s)^2,
```

which is generally nonzero. For comparison, delete this wrap coupling while keeping the same grid spacing and the Dirichlet stencil's diagonal. Then

```math
T_{\rm per}-T_{\rm D}
=-\frac{|0\rangle\langle K_s-1|+|K_s-1\rangle\langle0|}{2h_s^2},
\qquad
\|T_{\rm per}-T_{\rm D}\|_2=\frac1{2h_s^2}.
```

The difference has eigenvalues $\pm1/(2h_s^2)$ on the span of the two endpoint basis vectors and zero elsewhere. Thus it is not a grid-error term that vanishes as $h_s\to0$. This comparison holds at the same coordinates and spacing. The actual endpoint-inclusive mesh also changes its spacing to $U/(K_s-1)$.

In a joint evolution, $T_x+T_s+\operatorname{diag}\mathcal L$ acts on amplitudes for all represented $(x,s)$, not just on the minimizing slack at each $x$. Kinetic transport in $s$ and changes of $s^*(x)$ couple those amplitudes. Taking a pointwise minimum of potential values is neither taking a partial trace of the evolution nor eliminating the slack kinetic. Even a fine slack grid with a small error $E(x)$ therefore need not reproduce the distribution of an evolution under $L$ or its mode. This difference exists for Dirichlet slack axes as well. The periodic wrap is an additional choice of kinetic, not the sole source of changed dynamics. The spectral binary kinetic uses the same periodic Fourier representation and also does not impose a hard boundary at zero.

A small computation illustrates the distinction. For $F=(x-1/2)^2+(y-1/2)^2$ and $G=x+y-1/2$ on $[0,1]^2$, with $\bar\mu=0$, $\rho=1$ and slack range $[0,1/2]$, the PHR form and the slack form were solved as ordinary QHD problems, using the classical split-step flavor with exact readout, binary periodic $K=4$, 32 steps, total time 1, the default initial state and seed 7. The reported most-probable points projected to $(0.25,0.5)$ for the PHR form and $(0.25,0.25)$ for the slack form. The largest absolute difference between corresponding marginal probabilities, over both $x$ and $y$, was approximately $0.00335197$. This compares those two computed 16- and 64-component evolutions. It is not an augmented-Lagrangian convergence experiment or a claim of temporal convergence.

Suppose $U_0$ bounds every required continuous slack. On a periodic grid choose

```math
U\ge\frac{K_s}{K_s-1}U_0.
```

Then $U-h_s=U(1-1/K_s)\ge U_0$. Every required $t\in[0,U_0]$ lies between represented endpoints, so its distance to a node is at most $h_s/2$. Zero is present, and the slack-grid bound improves to $e(x)\le\rho h_s^2/8$. At equality, $h_s=U_0/(K_s-1)$, the same spacing as an endpoint-inclusive $K_s$-point mesh over $[0,U_0]$.

This margin is a proved remedy for upper-endpoint coverage. It does not remove the wrap coupling. A larger margin can put the last node at higher potential for some $x$, but neither that height nor the slack-elimination identity proves that its population or its influence on the low-energy dynamics is negligible. At fixed $K_s$, increasing the physical range changes both resolution and the kinetic coefficient. Under the default refinement search model of Proposition 48 the kinetic is instead fixed in unit coordinates, while the whole potential is range-normalized, so a physical-range increase does not simply weaken the wrap kinetic. There is no universal margin that reproduces a Dirichlet axis or certifies improved search.

## 12. Resource composition, error evidence, and verification

<a id="r55"></a>
### Proposition 55. Sound resource composition on a compact program

**Provenance: Derived in NWQLib. Evidence: proved conditional on valid leaf laws and declared execution semantics.**

For a finite selected program with exact or upper-bound leaf costs, the following rules produce valid total work and depth quantities. Serial composition adds work and depth. Explicit parallel composition on disjoint registers adds work and takes maximum depth. Repetition by $r$ multiplies work and serial depth by $r$. A branch uses the maximum of its alternatives, yielding an upper bound. An adaptive loop can use a per-round upper bound only if it holds uniformly over its allowed histories.

**Proof.** Structural induction on the program proves each rule. The serial case concatenates executions. Disjoint parallel calls have summed operations and maximum elapsed logical depth under the declared concurrency. Repetition concatenates $r$ copies. Every realized branch is bounded by the maximum alternative. An adaptive loop with at most $r$ rounds is bounded by $r$ times a history-uniform upper bound. Without that bound, a finite round limit alone gives no numerical per-round cost.

Storage follows lifetime rather than total work. At a location, allocation adds its live width, release subtracts that same allocation, and the peak is the maximum live sum. Serial temporary workspace is reusable, while simultaneous parallel workspace adds. Input or output arrays resident for the entire workload add to every affected peak. Independent experiments need a scheduling premise before their temporary peaks can be reused. These rules also follow by tracking the realized live objects at each execution point.

The evaluator can apply the induction without expanding repetitions. With memoization per distinct binding and lifetime context, its cost depends on the compact graph and the number of distinct contexts visited, rather than the expanded gate count. It does not claim a bound linear in the graph when the number of contexts grows. A missing leaf law remains unknown. An estimated leaf produces an estimate, and a conditional leaf carries its assumptions. Neither can become a proved bound by algebraic composition.

For QHD run-resource projections, per-circuit arbitrary-rotation counts and T estimates remain distinct from their shot-weighted totals. Under exact readout, `shot_arbitrary_rotations` and `shot_t_estimate` are unavailable (`None`), because no intended hardware shot count was selected. They are not zero-work hardware estimates. A chosen shot multiplicity is needed to form them. This does not change the recorded acquisition count of zero shots for an exact-readout execution.

<a id="r56"></a>
### Proposition 56. Error composition with units, provenance, and failure probabilities

**Provenance: Derived in NWQLib. Evidence: proved conditional on a complete error decomposition.**

Suppose a scientific output error has a valid decomposition into components with nonnegative bounds $\epsilon_i$, all expressed in the same norm, units, normalization, and target frame. If component $i$'s bound fails with probability at most $\delta_i$, then

```math
\|y-y_*\|\le\sum_i\epsilon_i\quad\text{with probability at least }1-\min\left(1,\sum_i\delta_i\right).
```

**Proof.** On the intersection of all component events, the triangle inequality gives the magnitude bound. The union bound controls the complement without requiring independence. A relative criterion needs an independently justified positive scale, and the failure event of a probabilistic scale bound joins the same union.

The implementation associates each bound with its output, operator, construction, parameter point, and premises. These associations establish which mathematical statement is being applied. They do not prove an unverified premise. If any required component is unknown, the sum of known terms is a subtotal rather than a total upper bound. If a sufficient upper bound exceeds the requested tolerance, the conclusion is inconclusive because the true error may be smaller. A passed numerical consistency check supplies only the relation it actually checked. The soundness statement assumes the method's declared component list is complete, an obligation that the generic combiner cannot infer from types.

<a id="r57"></a>
### Proposition 57. Scale-aware Gram and fidelity verification windows

**Provenance: Derived in NWQLib. Evidence: proved under the standard arithmetic model and stated elementary-function assumptions.**

For a matrix $V$ with complex columns of length $D$, let $\widehat S$ be the Gram matrix formed by floating-point inner products. Put $g=\sqrt2\gamma_{2D}<1$. An entrywise inner-product model gives

```math
\|\widehat S-V^\dagger V\|_2\le g\sum_j\|v_j\|^2\le\frac{g\,\operatorname{tr}\widehat S}{1-g}.
```

**Proof.** The complex dot-product error is at most $g\sum_k|v_{ki}||v_{kj}|$. Cauchy–Schwarz bounds this by $ga_ia_j$, $a_j=\|v_j\|$. Entrywise domination by $gaa^T$ bounds the spectral norm by $g\|a\|^2$. Each computed diagonal is at least $(1-g)\|v_j\|^2$, proving the final expression. This uses the dot-product model of [Higham, Lemma 3.1 and Eq. (3.5)][Higham]. A separate eigensolver rounding allowance is needed before rejecting a computed negative overlap eigenvalue. For measured entries with a symmetric matrix of error bounds $e_{ij}$, the alternative bound is $\|\Delta S\|_2\le\max_i\sum_je_{ij}$.

For nonzero complex vectors $x,y$, compute the normalized fidelity

```math
F=\frac{|\langle x,y\rangle|^2}{\langle x,x\rangle\langle y,y\rangle}
```

after scaling each by its largest modulus. This keeps the sums in range and does not assume that supplied vectors were normalized accurately. For $g_F=\gamma_{4D+4}<1/2$, the implementation's upper excursion window is

```math
\widehat F\le1+w_F,\qquad w_F=\left(\frac{1+g_F}{1-g_F}\right)^2\frac{1+u}{1-u}-1.
```

To derive it, the overlap error is at most $\sqrt2\gamma_{2D}\|x\|\|y\|$, and each computed squared norm is at least $(1-\gamma_{2D})$ times its exact value. One-ulp modulus and square evaluations add factors bounded by $1+\gamma_2$, while the final multiplication and division give $(1+u)/(1-u)$. Combining these factors and using $(1+\gamma_j)(1+\gamma_k)\le1+\gamma_{j+k}$ gives the displayed conservative $g_F$ expression. Its extra $D$ rounding units cover tiny-product underflow after scaling. This bounds an excess above one caused by evaluating the supplied vectors. It does not bound the physical error of either vector.

### Derivation: error budget of the `expm_multiply` reference {#expm-multiply-error-budget}

`expm_multiply_state_error(calls, start=s)` in `src/nwqlib/_validation.py` is the first-order 2-norm error budget $\delta$ of a host kernel that evolves a unit initial state of length $D$ by SciPy's `expm_multiply`, which is Algorithm 3.2 of Al-Mohy and Higham (SIAM J. Sci. Comput. 33 (2011) 488–511, doi:10.1137/100788860). For each call with a Hermitian generator $G$ it takes a bound $N$ on $\lVert G-(\operatorname{tr}(G)/D)I\rVert_1$, the shifted matrix whose Taylor sum SciPy evaluates, and a bound $r$ on the nonzeros in a row. With $\bar\theta=\min(N,9.9)$ and $S=55\max(1,\lceil N/9.9\rceil)$, one call charges (`expm_multiply_call_roundoff`)

```math
c=N+2\sqrt2\,rNe^{\bar\theta}+S\left(2e^{\bar\theta}+7\right),
```

and

```math
\delta=\Bigl(s+\sum_{\text{calls}}(c+N)+5P\Bigr)u.
```

The extra $N$ covers the rounded subtraction of the trace shift, and $5$ each of the $P$ final phase products (`GLOBAL_PHASE_STATE_ROUNDOFF`).

The start term $s$ bounds the construction error of the initial vector (`src/nwqlib/algorithms/qhd/initial_state.py::restricted_state_error`). It is $2$ for the uniform state's $1/\sqrt D$, one correctly rounded square root and one division, and for the kinetic ground state on the periodic grid, which is the uniform state. It is about $9d-1$ for the kinetic ground state of $d$ variables on the Dirichlet grids, from $5u$ per sine entry after reducing its angle to $(0,\pi/2]$, $3u$ to normalize each variable and $d-1$ product roundings, to first order. For a Gaussian the bound is finite (`initial_state.GaussianState._variables` and `_gaussian_beta`). Its exponents $z$ are shifted by their minimum $z_{\min}$ before exponentiation, and each exponent carries the enclosure $\lvert z-\hat z\rvert\le5u\,\hat z/(1-10u)+2^{-1073}$. Each amplitude then errs by at most $v\,(\operatorname{expm1}(x)+2u)/(1-2u)$ for the exponent-error enclosure $x$, with an explicit allowance for subnormal and zero entries, which gives $\beta=\lVert t-v\rVert/\lVert v\rVert$ against the exact Gaussian vector $t$ and the direction bound $2\beta$ from the exact inequality $\bigl\lVert t/\lVert t\rVert-v/\lVert v\rVert\bigr\rVert\le2\beta$. Two common factors give $x$, $(5z+(z-z_{\min}))u$ against the computed minimum and $(5(z+z_m)+(z-z_m))u$ against the exact exponent of the minimizing entry, which is then exact, and the smaller direction bound is used. Normalization adds $3u/(1-2u)+\sqrt K\,2^{-1075}$, and the tensor product composes the factors finitely as $\prod_j(1+e_j)-1$ plus $\gamma_{d-1}$ for the product roundings. The result is capped at $\sqrt{N_\psi^2+1}$, the largest distance between the computed nonnegative state of norm at most $N_\psi$ and the exact unit state, about $\sqrt2$.

Per variable, the initial-state records of `src/nwqlib/algorithms/qhd/initial_state.py` form these vectors and errors. The uniform state has entries $1/\sqrt K$, each one correctly rounded square root and one division, so each errs by at most $2u$. The classical start vector does not use these vectors and fills the equal product entry $1/\sqrt{K^d}$ directly (`UniformState.variable_amplitudes`). The kinetic ground state is the uniform vector on the periodic grid and $\sin(\pi(i+1)/(K+1))$ on the Dirichlet grids. There $\sin(\pi-\theta)=\sin\theta$ lets entry $i$ use the angle $\theta_i=\pi m_i/(K+1)$ with $m_i=\min(i+1,K-i)$, so $0<\theta_i\le\pi/2$. The computed angle has relative error at most $3u$ (`math.pi`, the product with the integer $m_i$ and the division by the integer $K+1$). A relative angle error $\eta$ changes $\sin\theta$ by the relative amount $\theta\cot\theta\,\eta$, and $0\le\theta\cot\theta<1$ on $(0,\pi/2]$, so with the one-ulp sine each entry errs by at most $5u$. The first-order 2-norm error of each vector is therefore $8u$ on the Dirichlet grids, $5u$ per entry and $3u$ for normalizing, and $2u$ on the periodic grid (`KineticGroundState.variable_amplitudes`, `variable_errors`). A Gaussian factor's exponents are shifted by their minimum before exponentiation, so its largest entry is exactly 1 before normalization (`GaussianState.variable_amplitudes`). Its 2-norm error bound against its exact direction is the direction bound of the computed amplitudes plus the rounding of the normalization, $3u/(1-2u)+\sqrt K\,2^{-1074}$, which charges the subnormal half-ulp $2^{-1075}$ above as the representable $2^{-1074}$, evaluated upward with the factor $1+2^{-40}$. The subnormal allowance is representable, and its addition to the far larger terms rounds like every other operation that the factor $1+2^{-40}$ counts. The bound is infinite where the direction bound is, namely when some exponents overflow and the smallest exponent exceeds $2^{1020}$, or when the smallest exponent itself overflows and the grid points nearest the center are tied or nearly tied (`GaussianState.variable_errors`).
{: #qhd-initial-state-vectors }

`state_mass_window(delta, D)` turns $\delta$ into the mass window

```math
\max\left(\tau,\ 2\delta+\delta^2+(D+4)u(1+\delta)^2\right),
```

where $\tau$ is `NUMERICAL_RELATION_RTOL`, $(D+4)u$ covers the evaluations of $\lvert z\rvert^2$ with `np.abs` and their $D-1$ additions, and the QHD tie window of [QHD most probable point and tie window](ENGINEERING_CONSTANTS.md#qhd-most-probable-point-and-tie-window) uses the same $\delta$.

The docstrings derive each term to first order in $u$ under two assumptions that SciPy's own error control makes. Its norm estimates are not below the spectral radius, and its two-term early-exit test (the paper's Eq. (3.15)) bounds the omitted Taylor tail in 2-norm. The truncated Taylor sum has backward error at most $uN$ (Eqs. (3.5)–(3.9)). An error made in Taylor term $j$ reaches the result through the rest of the finite sum, whose norm is at most $\sum_k\theta^kj!/(j+k)!$, and over all terms these factors total $\sum_{v=1}^mv\,\theta^{v-1}/v!\le e^\theta$, which gives the sparse-product coefficient. The remaining roundings of a substep total at most $(2\theta+1)e^\theta+m+6$, which is at most $m(2e^\theta+7)$ because SciPy's $\theta$ table has $\theta_m\le m/4$. The factor $e^{\bar\theta}$ is the per-substep $e^{\lVert A\rVert}$ of the paper's Lemma 4.1, and its Eq. (4.7) shows that for a normal matrix with a unitary exponential this factor exceeds what the problem's conditioning requires. $S$ bounds the number of matrix-vector products, because SciPy's cost minimization never chooses more products than degree 55 with $\lceil N/9.9\rceil$ substeps. Under the two assumptions the bound holds for every unit state and is correspondingly loose. The centering charge is first order and assumes $N$ also bounds the shifted matrix SciPy forms, which a huge scalar offset can violate.

On the one-variable $K=39$ grid of the QHD mass-window test, $(x-1/5)^2$ on $[-1,1]$ with two steps to $T=1$ from the default kinetic ground state, the Schrodinger kernel's mass exceeds one by $3.0\times10^{-12}$ against a window of $3.2\times10^{-8}$, and the computed state differs from a 40-digit evaluation of the same two steps by $2.9\times10^{-12}$ in the 2-norm up to a global phase. From the uniform initial state the two figures are $3.4\times10^{-12}$ and $2.8\times10^{-12}$. Both runs were made with Python 3.12.14, SciPy 1.18.1 and mpmath 1.3.0 on macOS arm64.

The QHD classical kernel computes $N$ from its stored schedule and support tables (`method._host_generator_norms`) and records the mass window as the `probability_window` argument of its `KernelApplication`, the counterpart of the instruction count that a circuit's preparation record holds.

### Derivation: QHD operation sizes {#qhd-operation-sizes}

`QHD.max_work=1_000_000_000` bounds known symbolic/table/action work, including every `expm_multiply` call of a classical host evolution charged with `_linalg_laws.expm_multiply_requirements`, and `QHD.max_bytes=10_000_000_000` bounds known arrays before selection or explicitly chosen construction. `QHDVerification` uses the same defaults for its separate explicit reference invocation. They exclude unknown SymPy/SciPy/Qiskit internal work and process RSS.

Planning charges each support table $K^{\lvert S\rvert}(N_S+\lvert S\rvert)$ units, one per node of its expanded expression and per coordinate at every grid tuple (`QHD._admit_symbolic_work`). Its bytes are $8E+8dK+H_{\rm tables}+\max(W_{\rm eval},I)$ for $E$ stored float64 entries, the centered coordinate vectors and the qualified table metadata $H_{\rm tables}$ (row "Support-table metadata and serialization allowances" of [Budgets and mechanical bounds](ENGINEERING_CONSTANTS.md#budgets-and-mechanical-bounds)), since table construction and identity serialization are successive phases. The construction workspace $W_{\rm eval}$ is the largest of the coordinate staging $40K$, one table's freezing snapshot $9K^{\lvert S\rvert}$ and its chunk charge

```math
B_{{\rm slab},S}(b)=8b(N_S+2)+(8\lvert S\rvert+64)b+256(N_S+2)+H_0,
```

with the chunk $b_S$ chosen from the bytes that remain after every other admitted population and the final tables (`compiler.support_chunk_size`). The serialization workspace $I=H_{\rm json,arrays}+Q+q_{\max}+12J$ covers the first identity computation of the tables and the initial vectors, with $Q$ the sum and $q_{\max}$ the largest of their padded-base64 lengths and $J$ their JSON length (`method._support_table_bytes`). The evaluator term $8b(N_S+2)$ and the header allowances are engineering allowances ([Budgets and mechanical bounds](ENGINEERING_CONSTANTS.md#budgets-and-mechanical-bounds)). For $(x+y+z+w-1)^{10}$ at $K=16$ this is $1.7\times10^8$ units, and degree 9 needs $9.9\times10^7$.

The `max_work` default is a fuse against runaway planning work, sized so that no example or test workload in the repository reaches it. On $[-1,1]^4$ the degree-10 tables took 1.6 to 1.7 s to evaluate (Python 3.12.14 and SymPy 1.14.0 on an Apple M3 Max). The monomial bound admitted before the expansion counts monomials, not the digits of their coefficients. The default therefore admits $(x+y)^{1000000}$, whose expansion forms about a million binomial coefficients of up to about 301,000 digits.

Every coefficient rule charges the step rows before any row exists, 2 units per midpoint step for its two point values or the schedule's `interval_work` per integrated step for its interval integrals (129 for the cubic schedule's quadrature and 2 for the closed forms of the other two), and `_STEP_BYTES` per step, and each compiled block occurrence is charged `_BLOCK_BYTES` (`QHD._admit_symbolic_work`, [Budgets and mechanical bounds](ENGINEERING_CONSTANTS.md#budgets-and-mechanical-bounds)).

Before any table work planning admits the evaluation of the initial state and its stored payload, one unit per grid point of each variable and

```math
320dK+(8dK+120d+40)+(8dK+120d+64+204d)+4096(d+1)
```

bytes (`src/nwqlib/algorithms/qhd/method.py::_initial_state_bytes`, `QHD.plan`). It is the only evaluation, because the reconstruction stores its vectors and error bound for the kernel, its windows and the native builder (`initial_state.evaluate`). This allowance is measured and specific to the implementation and runtime, Python 3.12.14, NumPy 2.5.2 and 64-bit CPython on macOS arm64, not a derived bound. The coefficient 320 per grid point (`_INITIAL_STATE_BYTES`) covers the evaluator's Python float lists and float64 vectors. The Gaussian state, which also forms its entry-error bounds, had a traced peak of about 277 bytes per point for one variable with $K$ from 4096 to $10^6$, and the kinetic ground state 73. That coefficient alone does not cover the fixed costs of small grids, where the Gaussian stage traced 1,505 bytes at $d=1$, $K=2$ against 640. The second term is the normalized float64 vectors and their tuple, the third the stored payload of $d$ separately owning float64 arrays of $K$ entries, counting data $8dK$, ndarray headers $112d$, their tuple $8d+40$ and one 24-byte error float, plus 204 bytes per array for its `FrozenArray` wrapper (`_FROZEN_ARRAY_BYTES`, [Budgets and mechanical bounds](ENGINEERING_CONSTANTS.md#budgets-and-mechanical-bounds)), and $4096(d+1)$ (`_INITIAL_STATE_OBJECT_BYTES`) an untuned allowance for Python and NumPy bookkeeping. The stage traced at most 0.80 of the allowance over Gaussian states and the kinetic ground state with $d$ from 1 to 3 and $K$ from 2 to 4096, up to $K=256$ at $d=3$ (`tests/test_qhd_workflow.py` keeps the small cases). It does not cover the complete `QHDReconstruction` constructor, and the table admission charges the identity serialization of the stored vectors. Requalify it when the evaluator, the stored representation, the object lifetimes or the runtime changes. The same units open the running total of the table admission (`QHD._admit_symbolic_work`), whose final admission includes these bytes, so the per-Plan admissions that the augmented-Lagrangian and refinement bounds count already contain the initial state.

Revisit the defaults when a concrete larger workload is selected. Shared ExecutionLimits separately govern acquisition and stored results.

The augmented-Lagrangian layer admits its support-table checks and required original-expression numerical scans, together with its reserved round evaluations of $f$, $h$ and $g$, against `QHD.max_work` before the corresponding work. Each original-summand scan costs $K^{\lvert S_j\rvert}(N_j+\lvert S_j\rvert)$ units for its support $S_j$ and supplied tree size $N_j$, with one charge when structural coverage and the range check share it. An inconclusive addition bound can require a further admitted whole-term scan costing $K^{\lvert S\rvert}(N+\lvert S\rvert)$ units (`src/nwqlib/algorithms/qhd/constrained.py::_admit_layer_work`). The layer evaluates each support table on arrays in chunks that `QHD.max_bytes` admits beside a reservation for the term's tables $8\sum E$, the centered coordinates $8dK$, their metadata $H_0$ and the largest table copy $8\max E$, of which it keeps one table and the extrema at a time, with the chunk charge $B_{{\rm slab},S}(b)$ above (`src/nwqlib/algorithms/qhd/constrained.py::_support_tables`). Each original-summand scan is evaluated on arrays in chunks admitted the same way beside the coordinate vectors $8dK$ and $H_0$, streams its maximum and keeps no table, and a chunk whose array arithmetic signals a division by zero, an overflow or an invalid operation is evaluated as scalars (`src/nwqlib/algorithms/qhd/constrained.py::_domain_check`).

Each classical kernel law, `method.restricted_sizes` for the Schrodinger and one-hot IR-product flavors and `split_step.sizes`, charges forming the start vector from the stored amplitudes once per evolution, $W_{\rm start}=D$ for the uniform fill and $dK+\sum_{j=2}^dK^j+D$ for a product of the stored factors (`src/nwqlib/algorithms/qhd/initial_state.py::start_vector_work`). Its explicit arrays, at most $24D+8dK$ bytes, precede the evolution arrays and fit within both byte laws. The classical split-step flavor charges

```math
W_{\rm start}+D(d+M+7)+dK+N_s\bigl(2dLD+(2d+13)D+7dK+d+4\bigr)
```

units for $D=K^d$, the start vector's construction $W_{\rm start}$, $M$ support tables, $N_s$ steps and the transform level count $L$, and

```math
64D+32(d+1)K+8T_{\max}+8N_s+1024K+4096(d+1)
```

bytes for the largest table size $T_{\max}$ (`src/nwqlib/algorithms/qhd/split_step.py::sizes`), including the reductions and per-step charges of the state budget that the evolution observes, whose transform term $2dLD$ is a nominal proxy, not a count of the backend's operations. Neither depends on the schedule weights or the objective's range.

For the two-dimensional Ackley function on $[-5,5]^2$ mapped to the unit square, with $K=32$ on the default Dirichlet interior grid, 1000 midpoint steps and total time 10 (planned with Python 3.12.14 and SymPy 1.14.0 on macOS arm64, the quadratic schedule with $\gamma=0.3$ unless another schedule is named, and `max_work = 1e20` and `max_bytes = 1e12` so that planning reports each charge), the split-step charge is $4.7\times10^7$ units and the Schrodinger flavor is charged $9.0\times10^8$ under the quadratic schedule and $8.9\times10^{13}$ under `ShiftedCubicSchedule(s=2e-4)`, or $1.5\times10^{16}$ with the integrated rule.

## 13. Implementation and evidence index

Code locations below are paths in the NWQLib repository. A name after a double colon is a function, class, or method defined in that file. The provenance abbreviations are R, **From a reference**, D, **Derived in NWQLib**, and I, **Improved from a reference**. The evidence column identifies source-defined validation checks in the repository. Proofs establish the analytic statements, while these checks test their implementation and premises on finite cases.

| Result and statement | Label | Code | Source-defined check or evidence | Contribution |
| --- | --- | --- | --- | --- |
| [1. Weighted and time-uniform sampling](#r1) | I | `src/nwqlib/algorithms/expectation.py::ExpectationMethod.sampling_shots`, `src/nwqlib/evidence/binary.py::infer_binary` | `tests/test_expectation_inference.py`, fixed-time eligibility and family intervals | Standard inequality composed across settings, time, and physical scale |
| [2. Shared readout calibration](#r2) | D | `src/nwqlib/evidence/binary.py::correct_binary`, `src/nwqlib/evidence/statistics.py::linear_variance` | `tests/test_expectation_inference.py`, affine-channel and shared-data cases | Useful combination of exact interval geometry and covariance accounting |
| [3. QCELS](#r3) | R | `src/nwqlib/algorithms/qpe/numerical.py::qcels` | `tests/test_qpe_workflow.py`, `tests/test_qpe_correction.py` | Standard objective with explicit finite-search scope |
| [4. SPE and RFE](#r4) | R | `src/nwqlib/algorithms/qpe/numerical.py::spe`, `src/nwqlib/algorithms/qpe/numerical.py::rfe` | `tests/test_qpe_domain_evidence.py`, `tests/test_qpe_correction.py` | Standard estimators with sign and target qualifications |
| [5. RWPE](#r5) | R | `src/nwqlib/algorithms/qpe/numerical.py::rwpe_update`, `src/nwqlib/algorithms/qpe/numerical.py::rwpe_scale` | `tests/test_qpe_campaigns.py` | Standard Gaussian moment update |
| [6. Coherent QPE](#r6) | R | `src/nwqlib/subroutines/qpe/coherent.py::build_coherent_qpe_circuit` | `tests/test_qpe_subroutines.py` | Standard circuit and readout theorem |
| [7. Controlled powers](#r7) | D | `src/nwqlib/algorithms/qpe/powers.py::suzuki_step_cx`, `src/nwqlib/algorithms/qpe/powers.py::select_powers` | `tests/test_qpe_efficiency_contract.py`, controlled phase and CX checks | Constructor-specific exact law and pruning composition |
| [8. Chebyshev pencil](#r8) | R | `src/nwqlib/algorithms/lanczos/numerical.py::_projected_matrices` | `tests/test_lanczos.py::test_signed_centered_pencil_from_independent_moments` | Standard projected identities |
| [9. Paired moments](#r9) | I | `src/nwqlib/algorithms/lanczos/numerical.py::chebyshev_moments` | `tests/test_lanczos.py`, independent moments and offset checks | Reduced operator applications from a known product identity |
| [10. Moving-subspace derivative](#r10) | I | `src/nwqlib/algorithms/lanczos/numerical.py::_moment_energy_derivatives` | `tests/test_lanczos.py::test_regularized_sensitivity_rotates_the_kept_projector` | Explicit analytic derivative of the thresholded solve |
| [11. Pilot allocation](#r11) | I | `src/nwqlib/algorithms/lanczos/numerical.py::_sensitivity_weights`, `src/nwqlib/algorithms/lanczos/workflow.py` | `tests/test_lanczos.py::test_sensitivity_weights_match_independent_quadratic_root` | Variance-weighted extension with conditional main-stage statistics |
| [12. Gram noise](#r12) | D | `src/nwqlib/algorithms/lanczos/numerical.py::_resolve_overlap_cutoff` | `tests/test_lanczos.py::test_shared_moment_covariance_and_rank_policy` | Linear-work covariance contraction and explicit coverage scope |
| [58. Lanczos continuation](#r58) | D | `src/nwqlib/algorithms/lanczos/method.py::_trajectory_program`, `src/nwqlib/backends/qiskit_aer.py::_prepare_aer_execution`, `src/nwqlib/algorithms/lanczos/readout.py::LanczosReadout`, `src/nwqlib/algorithms/lanczos/readout.py::outcome`, `src/nwqlib/algorithms/lanczos/readout.py::reduce_moments`, `src/nwqlib/algorithms/lanczos/readout.py::decode_histogram`, `src/nwqlib/algorithms/lanczos/records.py::MomentSetting` | `tests/test_lanczos_circuits.py::test_shared_walk_metadata_counts_and_padding_readout`, `tests/test_lanczos_circuits.py::test_native_signed_centered_two_qubit_public_chain` | Coherent view restoration, raw-mass and padding identities, and the resource counts of the executed trajectory |
| [13. GCiM pencil](#r13) | R | `src/nwqlib/algorithms/gcim/pencil.py::assemble_pencil`, `src/nwqlib/_projected_eigensolver.py::_solve_projected_pencil` | `tests/test_gcim_pencil.py`, `tests/test_projected_eigensolver.py` | Standard projected eigensolver |
| [14. Identity offsets](#r14) | D | `src/nwqlib/algorithms/gcim/fixed_basis.py::solve_pencil`, `src/nwqlib/algorithms/lanczos/numerical.py::reconstruct` | `tests/test_gcim_production.py::test_identity_offset_moves_the_ritz_value_by_its_coefficient` | Stable implementation of shift covariance |
| [15. Pauli insertion](#r15) | I | `src/nwqlib/algorithms/gcim/optimization.py::_generator_shift_rule`, `src/nwqlib/algorithms/gcim/adapt_acquisition.py::construct_adapt` | `tests/test_adapt_gcim.py`, `tests/test_adapt_gcim_efficiency_contract.py` | Valid derivative placement for noncommuting Pauli sums |
| [16. Shared-index fermionic blocks](#r16) | D | `src/nwqlib/subroutines/fermionic_circuits.py::_append_shared_index` | `tests/test_gcim_native_chain.py`, `tests/test_fermionic_pool.py` | Coverage of a missing compact-generator case using standard rotations |
| [17. Residual conclusions](#r17) | R | `src/nwqlib/algorithms/gcim/adapt_verification.py::_residual_metrics` | `tests/test_adapt_verification.py` | Standard inequalities with explicit identification premises |
| [18. Relative inverse residual](#r18) | D | `src/nwqlib/subroutines/qsp/inverse.py::_fit_candidate`, `src/nwqlib/algorithms/qls/host_planning.py::select_polynomial`, `src/nwqlib/algorithms/qls/numerical.py::inverse_polynomial_action` | `tests/test_qls_primary.py`, `tests/test_qls_native.py` | A fit criterion that directly controls relative physical-vector error |
| [19. Shortcut corrections](#r19) | I | `src/nwqlib/subroutines/qsp/shortcut.py::_realize_kernel_reflection`, `src/nwqlib/algorithms/qls/norm_search.py` | `tests/test_qls_primary.py`, `tests/test_qls_quantum.py` | Corrections to signed bound and search direction |
| [20. Periodic gap](#r20) | D | `src/nwqlib/algorithms/qls/periodic.py::select_periodic_encoding` | `tests/test_qls_periodic.py::test_periodic_downward_coefficient_rounding_covers_actual_gap` | Outward coverage of the implemented encoding |
| [21. QSP conventions](#r21) | R | `src/nwqlib/subroutines/qsp/phases.py::wx_phases_to_reflection`, `src/nwqlib/subroutines/qsp/evolution.py::build_qsvt_circuit` | `tests/test_qsp_evolution.py`, zero-phase and reflection identities | Standard polynomial calculus |
| [22. Norming correction](#r22) | I | `src/nwqlib/subroutines/qsp/phases.py::chebyshev_norming_sup_bound` | `tests/test_qsp_evolution.py::test_chebyshev_grid_norming_bound_covers_polynomial_suprema`, exact counterexample above | Correction with a minimal analytic counterexample |
| [23. Phase Newton step](#r23) | I | `src/nwqlib/subroutines/qsp/phases.py::_symmetric_problem`, `src/nwqlib/subroutines/qsp/phases.py::_damped_newton` | `tests/test_qsp_evolution.py`, fitted coefficients and boundary targets | Explicit Jacobian and equivalence on an oversampled grid |
| [24. QSP error budget](#r24) | D | `src/nwqlib/subroutines/qsp/evolution.py::_analytic_jacobi_anger_remainder`, `src/nwqlib/subroutines/qsp/evolution.py::qsp_evolution_error_terms` | `tests/test_qsp_evolution.py`, complete tail, child error, and exponential comparisons | Finite stage composition, not a sharper Bessel bound |
| [25. LCHS identity](#r25) | R | `src/nwqlib/algorithms/lchs/providers.py::eq7_coefficient`, `src/nwqlib/algorithms/lchs/providers.py::low_somma_fhat_2` | `tests/test_lchs_providers.py`, kernel profiles | Standard representations |
| [26. LCHS tail](#r26) | I | `src/nwqlib/algorithms/lchs/providers.py::eq7_tail_bound` | `tests/test_lchs_providers.py::test_closed_tail_bound_keeps_small_cutoff_domain_and_positive_underflow` | Finite closed form and extended cutoff domain |
| [27. Gauss selector](#r27) | I | `src/nwqlib/algorithms/lchs/providers.py::_ellipse_rule` | `tests/test_lchs_providers.py::test_quadrature_work_refusal_names_a_value_that_admits_on_the_first_retry` | Analytic selector applicable at short dissipative times |
| [28. Source remainder](#r28) | I | `src/nwqlib/algorithms/lchs/inhomogeneous_theory.py::_duhamel_quadrature_error_bound` | `tests/test_lchs_numerics.py::test_duhamel_remainder_uses_physical_growth_and_exact_zero_routes` | Dimension-free vector remainder in original physical units |
| [29. Structured SELECT](#r29) | D | `src/nwqlib/algorithms/lchs/time_independent_terms.py::_attach_affine_pauli_structure`, `src/nwqlib/algorithms/lchs/parameters.py::select_parameters` | `tests/test_lchs_qsp_source.py`, `tests/test_lchs_compiled_select.py` | Structure-dependent circuit reduction and source-time consistency |
| [30. Output frames](#r30) | D | `src/nwqlib/algorithms/lchs/solution_error_budget.py::propagate_physical_error` | `tests/test_lchs_verification.py::test_refinement_propagates_all_physical_components_once_without_vector` | Standard norm inequalities assembled into one physical budget |
| [31. Dilation](#r31) | D | `src/nwqlib/subroutines/block_encoding/core.py::_dense_dilation_encoding`, `src/nwqlib/subroutines/lcu/core.py::build_lcu_circuit` | `tests/test_block_encoding.py`, `tests/test_lcu_subroutines.py` | Singular-frame construction and compatible padding, based on standard LCU |
| [32. Circulant certificate](#r32) | I | `src/nwqlib/subroutines/block_encoding/core.py::_detect_banded_pauli_structure`, `src/nwqlib/subroutines/block_encoding/banded.py::build_banded_block_encoding` | `tests/test_block_encoding.py::test_circulant_frobenius_certificate_rounds_outward` | Coefficient-domain certification including absent labels |
| [33. TT-SVD fidelity](#r33) | I | `src/nwqlib/subroutines/state_preparation/mps.py::analyze_mps_state_compression` | `tests/test_mps_state_preparation.py` | Exact-arithmetic fidelity consequence of the discarded-weight argument |
| [34. Multiplexors](#r34) | R | `src/nwqlib/subroutines/_multiplexors.py::append_control_diagonal_phases`, `src/nwqlib/subroutines/state_preparation/direct.py::_build_normalized_state_preparation` | `tests/test_multiplexors.py` | Standard exact synthesis identities |
| [35. Dense-synthesis laws](#r35) | I | `src/nwqlib/subroutines/_dense_synthesis.py::dense_synthesis_gate_census`, `src/nwqlib/subroutines/_dense_synthesis.py::controlled_synthesis_size` | `tests/test_dense_synthesis.py`, `tests/test_synthesis_admission.py` | Upper bounds that cover skipped numerical optimizations and control routes |
| [36. Product-formula steps](#r36) | I | `src/nwqlib/subroutines/trotterization/error_budget.py::_pauli_bound_coefficient_from_terms`, `src/nwqlib/subroutines/trotterization/error_budget.py::_smallest_step_count`, `src/nwqlib/algorithms/lchs/periodic.py::select_periodic_parameters` | `tests/test_trotterization.py`, `tests/test_lchs_periodic.py`, `tests/test_trotter_efficiency_contract.py`, `tests/test_qpe_efficiency_contract.py` | Full or relaxed Pauli-triangle coefficient with outward evaluation, exact integer selection for that coefficient, and a weighted periodic Strang specialization |
| [37. QHD grids](#r37) | R | `src/nwqlib/algorithms/qhd/kinetic.py::KineticCompiler`, `src/nwqlib/algorithms/qhd/split_step.py::kinetic_eigenvalues` | `tests/test_qhd_periodic.py`, `tests/test_qhd_workflow.py` | Standard finite-grid physics with explicit boundaries |
| [38. Schedule integrals](#r38) | D | `src/nwqlib/algorithms/qhd/schedules.py::QuadraticSchedule.kinetic_integral`, `src/nwqlib/algorithms/qhd/schedules.py::CubicSchedule.kinetic_integral` | `tests/test_qhd_schedules.py` | Stable forms and a bounded positive quadrature |
| [39. Time ordering](#r39) | D | `src/nwqlib/algorithms/qhd/evolution_bounds.py::_schedule_terms`, `src/nwqlib/algorithms/qhd/evolution_bounds.py::_coefficient_term` | `tests/test_qhd_resources.py` | Finite bound separating time ordering, quadrature, and coefficient error |
| [40. Emitted QHD products](#r40) | I | `src/nwqlib/algorithms/qhd/evolution_bounds.py::evolution_bound`, `src/nwqlib/algorithms/qhd/evolution_bounds.py::_norm_inputs` | `tests/test_qhd_resources.py`, native/restricted product checks | Encoding-dependent specialization with internal link terms |
| [41. Split-step error](#r41) | D | `src/nwqlib/algorithms/qhd/split_step.py::state_error`, `src/nwqlib/algorithms/qhd/split_step.py::evolve`, `src/nwqlib/algorithms/qhd/split_step.py::_kinetic_axis`, `src/nwqlib/algorithms/qhd/split_step.py::_potential_moment`, `src/nwqlib/algorithms/qhd/split_step.py::sizes` | `tests/test_qhd_split_step.py` | Conditional observed budget, scaled moments, and fixed-step work and storage |
| [42. Running phase total](#r42) | I | `src/nwqlib/algorithms/qhd/validation.py::_normal_range`, `src/nwqlib/algorithms/qhd/potential.py::PotentialCompiler.select_occurrences`, `src/nwqlib/algorithms/qhd/method.py::_phase_ledger`, `src/nwqlib/algorithms/qhd/method.py::_native_phase_allowance`, `src/nwqlib/algorithms/qhd/binary.py::compile_binary_steps`, `src/nwqlib/algorithms/qhd/circuit_errors.py::binary_phase_allowance` | `tests/test_qhd_resources.py`, `tests/test_qhd_binary.py`, `tests/test_qhd_workflow.py` | Formation, omission, reconstruction, and native modular-phase allowances |
| [43. Signed binary kinetic](#r43) | I | `src/nwqlib/algorithms/qhd/binary.py::signed_square_walsh`, `src/nwqlib/algorithms/qhd/binary.py::kinetic_walsh`, `src/nwqlib/algorithms/qhd/resources.py::circuit_resources`, `src/nwqlib/algorithms/qhd/method.py::_error_model` | `tests/test_qhd_binary.py`, `tests/test_qhd_resources.py`, `tests/test_qhd_split_step.py::test_spectral_and_finite_difference_energies_at_nyquist_and_low_momentum` | Signed-index correction, structural sparsity, and a distinct model-error obligation |
| [44. Binary synthesis errors](#r44) | D | `src/nwqlib/algorithms/qhd/binary.py::walsh_admission`, `src/nwqlib/algorithms/qhd/binary.py::PhaseTable.synthesize`, `src/nwqlib/algorithms/qhd/circuit_errors.py::binary_angle_formation`, `src/nwqlib/algorithms/qhd/circuit_errors.py::kinetic_fd_summary_up`, `src/nwqlib/algorithms/qhd/binary.py::qft_error_bound` | `tests/test_qhd_binary.py`, `tests/test_qhd_resources.py` | Distinct coefficient, angle, omission, permutation, and transform error terms |
| [45. Mode windows](#r45) | D | `src/nwqlib/_validation.py::probability_difference_window`, `src/nwqlib/algorithms/qhd/method.py::_host_window`, `src/nwqlib/algorithms/qhd/method.py::_readout_window`, `src/nwqlib/algorithms/qhd/theory.py::onehot_product_state_error` | `tests/test_qhd_readout.py`, `tests/test_qhd_split_step.py` | Direct probability-difference bound with explicit host-budget and subtraction scope |
| [46. AL equations](#r46) | R | `src/nwqlib/algorithms/qhd/constrained.py::_effective_objective`, `src/nwqlib/algorithms/qhd/constrained.py::_normalized`, `src/nwqlib/algorithms/qhd/constrained.py::_update`, `src/nwqlib/algorithms/qhd/constrained.py::_original_multiplier` | `tests/test_qhd_augmented_lagrangian.py`, independent grid updates | Standard normalized PHR construction with exact-arithmetic scaling scope |
| [47. AL stopping](#r47) | D | `src/nwqlib/algorithms/qhd/constrained.py::_stationarity`, `src/nwqlib/algorithms/qhd/constrained.py::solve_augmented_lagrangian` | `tests/test_qhd_augmented_lagrangian.py::test_feasible_complementary_can_stop_away_from_the_feasible_grid_minimum` | Finite-grid limitation and objective-gap bound |
| [48. Conditional search invariance](#r48) | D | `src/nwqlib/algorithms/qhd/refinement.py::_level_problem`, `src/nwqlib/algorithms/qhd/_outer.py::range_bound` | `tests/test_qhd_refinement.py::test_search_model_levels_are_invariant_under_affine_maps_of_box_and_objective` | Affine invariance for corresponding tables and a homogeneous range rule, with a conditional numerical bound |
| [49. Joint refinement mass](#r49) | I | `src/nwqlib/algorithms/qhd/refinement.py::_joint_mass_bound`, `src/nwqlib/algorithms/qhd/refinement.py::_next_box`, `src/nwqlib/algorithms/qhd/refinement.py::_axis_interval`, `src/nwqlib/algorithms/qhd/refinement.py::_joint_mass`, `src/nwqlib/algorithms/qhd/_coverage.py::bounds_from_counts`, `src/nwqlib/algorithms/qhd/_coverage.py::coverage` | `tests/test_qhd_refinement.py::test_joint_mass_bound_is_attained_and_is_neither_eta_nor_the_product`, `tests/test_qhd_refinement.py::test_selected_region_coverage_uses_both_bounds_and_preserves_numerical_limits` | Sharp marginal union bound and simultaneous finite-shot selected region coverage, including data selection and all configured levels |
| [50. Stall split](#r50) | D | `src/nwqlib/algorithms/qhd/refinement.py::_stall_split`, `src/nwqlib/algorithms/qhd/refinement.py::_tabulated_objective`, `src/nwqlib/algorithms/qhd/refinement.py::_refine` | `tests/test_qhd_refinement.py::test_stall_split_follows_the_end_mass_lemma_and_the_clearest_valley` | Stall condition and realized-prefix comparison using a fixed relative objective |
| [51. AL/refinement composition](#r51) | D | `src/nwqlib/algorithms/qhd/constrained.py::_refined_choice`, `src/nwqlib/algorithms/qhd/refinement.py::refine_box` | `tests/test_qhd_augmented_lagrangian.py::test_refined_rounds_stay_within_the_distance_bound_to_the_kkt_point` | New combination bound with independently checkable inner premises |
| [52. Clifford/T allocation](#r52) | D | `src/nwqlib/algorithms/qhd/resources.py::clifford_distance`, `src/nwqlib/algorithms/qhd/resources.py::synthesis_projection` | `tests/test_qhd_resources.py` | Finite replacement budget and optimal continuous allocation for a fixed cost model |
| [53. One-hot preparation](#r53) | D | `src/nwqlib/algorithms/qhd/initial_state.py::append_amplitude_chain`, `src/nwqlib/algorithms/qhd/initial_state.py::chain_selection`, `src/nwqlib/algorithms/qhd/initial_state.py::_gaussian_beta`, `src/nwqlib/algorithms/qhd/initial_state.py::restricted_state_error`, `src/nwqlib/algorithms/qhd/circuit_errors.py::preparation_error` | `tests/test_qhd_workflow.py`, `tests/test_qhd_resources.py` | Structured preparation, error terms for shortened chains, and input-state bounds with integer dimension factors |
| [54. Slack variables](#r54) | D | `src/nwqlib/algorithms/qhd/constrained.py::_inner_objective`, `src/nwqlib/algorithms/qhd/constrained.py::_inequality_range`, `src/nwqlib/algorithms/qhd/constrained.py::_slack_axis`, `src/nwqlib/algorithms/qhd/constrained.py::_branch`, `src/nwqlib/algorithms/qhd/constrained.py::_round_problem`, `src/nwqlib/algorithms/qhd/constrained.py::_choose`, `src/nwqlib/algorithms/qhd/constrained.py::_refined_choice`, `src/nwqlib/algorithms/qhd/constrained.py::plan_augmented_lagrangian`, with `src/nwqlib/algorithms/qhd/constrained_records.py::InnerRepresentation`, `src/nwqlib/algorithms/qhd/constrained_records.py::SlackAxis` and `src/nwqlib/algorithms/qhd/constrained_records.py::FormTrial` | `tests/test_qhd_augmented_lagrangian.py::test_minimizing_the_slack_grid_reproduces_the_phr_term_within_the_recorded_error_bound`, `tests/test_qhd_augmented_lagrangian.py::test_support_table_caps_are_exact_for_separable_constraints_and_bound_overlapping_supports`, `tests/test_qhd_augmented_lagrangian.py::test_forced_slack_rounds_keep_the_phr_update_at_the_projected_point`, the policy, joint-readout, refinement, preprocessing and archive checks in the same file, and `tests/test_qhd_resume.py::test_a_slack_round_resumes_with_its_committed_representation` | Standard slack elimination with cap and arithmetic conditions, finite-grid excess, support-table caps, projected PHR updates and the limits of joint slack evolution |
| [55. Resource composition](#r55) | D | `src/nwqlib/resources/fold.py::_Fold`, `src/nwqlib/algorithms/qhd/resources.py::run_resources` | `tests/test_resource_fold.py`, `tests/test_resource_estimation_tiers.py`, `tests/test_qhd_resources.py` | Compositional semantics with explicit unknowns, shot multiplicities, and lifetimes |
| [56. Error evidence](#r56) | D | `src/nwqlib/evidence/error_model.py::ErrorModel.assess` | `tests/test_error_model.py` | Mathematical obligations and provenance kept with their composition |
| [57. Verification windows](#r57) | D | `src/nwqlib/_projected_eigensolver.py::gram_formation_allowance`, `src/nwqlib/_numerics.py::normalized_fidelity_with_window` | `tests/test_projected_eigensolver.py`, `tests/test_mps_state_preparation.py`, `tests/test_qhd_split_step.py` | Scale-aware checks with a specified numerical reference |

## 14. Open mathematical problems

**Complete physical error budgets.** A successful scalar or circuit check does not close all components of a scientific error model. QLS still needs certified spectral premises and a complete treatment of approximate encodings and numerical phase synthesis for each route. LCHS needs all coefficient-preparation, source-preparation, quadrature, and simulation terms in the recovered physical frame. GCiM and Lanczos need bounds linking acquisition errors and overlap truncation to the desired full-space eigenvalue, with an independently established target-identification premise. These are proposals for further analysis, not consequences of a small projected residual.

**QHD continuum and optimization error.** The finite-grid evolution bounds do not establish convergence to a continuous minimizer. A useful total budget must relate spatial discretization, boundary choice, finite evolution time, initial preparation, sampling, and the objective quality of the readout. A spectral kinetic substituted for finite differences needs a trajectory-dependent momentum bound or a suitable uniform model bound. Feasibility and complementarity stopping require additional inner-minimization and stationarity control before augmented-Lagrangian convergence results apply. Refinement can discard a global basin, even when its probability-mass calculation is exact.

**Unavailable entries of the QHD circuit error budget.** The QHD circuit error budget separates kinetic-model replacement, evolution, coefficient formation, phase formation, pruning, approximate Fourier transforms, preparation, rotation synthesis, and physical execution. Missing terms cannot be interpreted as zero. Some classical-kernel roundoff budgets are first order and conditional on transform or elementary-function accuracy. A finite all-orders evaluation and a fully qualified transform error model would strengthen these entries. Exact dense synthesis also needs a global matrix-to-circuit rounding theorem covering numerical factorizations and every local replacement.

**Certified polynomial evaluation.** The corrected norming theorem supplies an analytic continuum bound from exact samples. Directed evaluation of the polynomial samples and of spectral enclosures would turn more numerical certificates into finite outward certificates. The degree-search limit and damped Newton fallback currently have finite stopping behavior, but no theorem that all accepted targets converge within the selected limits.

**Adaptive statistical guarantees.** Pilot/main separation supports conditional moment statistics. It does not settle nonlinear Ritz bias, cutoff selection near rank changes, adaptive operator-pool screening, or selection among many compared plans. ADAPT's flat-energy stopping rule and QHD's valley score are finite search decisions without general optimality guarantees. Further results should name the target, selection history, and acquisition population they cover.

**Resource laws beyond selected decompositions.** Logical gate counts, rotation estimates, and physical resource models have different assumptions. The leading Ross–Selinger term does not provide a finite T-count upper bound for the structured angles of a large QHD computation. A useful extension would jointly choose decompositions, precision allocation, and compilation while maintaining an explicit error budget. Compact work laws also need representation-specific guarantees whenever a backend's hidden workspace or control synthesis determines feasibility.

## References

The equation and theorem locators in the text refer to the versions below.

- [Hoeffding][Hoeffding]. W. Hoeffding, “Probability Inequalities for Sums of Bounded Random Variables,” JASA 58, 13–30 (1963). DOI 10.1080/01621459.1963.10500830.
- [Clopper and Pearson][ClopperPearson]. “The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial.” Biometrika 26(4), 404–413 (1934). DOI 10.1093/biomet/26.4.404.
- [Ding and Lin][QCELS]. “Even Shorter Quantum Circuit for Phase Estimation on Early Fault-Tolerant Quantum Computers with Applications to Ground-State Energy Estimation.” arXiv:2211.11973v2.
- [Wan, Berta and Campbell][SPE]. “Randomized Quantum Algorithm for Statistical Phase Estimation.” arXiv:2110.12071v2.
- [Kshirsagar, Katabarwa and Johnson][RFE]. “On proving the robustness of algorithms for early fault-tolerant quantum computers.” arXiv:2209.11322v3.
- [Granade and Wiebe][RWPE]. “Using Random Walks for Iterative Phase Estimation.” arXiv:2208.04526v1.
- [Cleve, Ekert, Macchiavello and Mosca][CoherentQPE]. “Quantum Algorithms Revisited.” arXiv:quant-ph/9708016v1.
- [Kirby, Motta and Mezzacapo][Kirby]. “Exact and efficient Lanczos method on a quantum computer.” Quantum 7, 1018 (2023), arXiv:2208.00567v4.
- [Oumarou, Ollitrault, Polla and Gogolin][Oumarou]. “Optimizing and Comparing Quantum Resources of Statistical Phase Estimation and Krylov Subspace Diagonalization.” arXiv:2603.15552v1.
- [Zheng et al.][GCIM2023]. “Quantum algorithms for generator coordinate methods.” Physical Review Research 5, 023200 (2023), arXiv:2212.09205v1.
- [Zheng et al.][GCIM2024]. “Unleashed from constrained optimization: quantum computing for quantum chemistry employing generator coordinate inspired method.” npj Quantum Information 10, 127 (2024), arXiv:2312.07691v3.
- [Epperly, Lin and Nakatsukasa][Epperly]. “A theory of quantum subspace diagonalization.” SIAM Journal on Matrix Analysis and Applications 43, 1263–1290 (2022), arXiv:2110.07492v2.
- [Schuld et al.][Schuld]. “Evaluating analytic gradients on quantum hardware.” Physical Review A 99, 032331 (2019), arXiv:1811.11184v1.
- [Wierichs, Izaac, Wang and Lin][Wierichs]. “General parameter-shift rules for quantum gradients.” Quantum 6, 677 (2022), arXiv:2107.12390v3.
- [Magoulas and Evangelista][SpinAdapted]. “Spin-Adapted Fermionic Unitaries: From Lie Algebras to Compact Quantum Circuits.” arXiv:2511.13485v2.
- [Childs, Kothari and Somma][CKS]. “Quantum algorithm for systems of linear equations with exponentially improved dependence on precision.” SIAM Journal on Computing 46, 1920–1950 (2017), arXiv:1511.02306v2.
- [Dalzell][Dalzell]. “A shortcut to an optimal quantum linear system solver.” arXiv:2406.12086v2.
- [Martyn, Rossi, Tan and Chuang][Martyn]. “A Grand Unification of Quantum Algorithms.” arXiv:2105.02859v5.
- [Gilyén, Su, Low and Wiebe][GSLW]. “Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics.” arXiv:1806.01838v1.
- [Ehlich and Zeller][EZ]. “Schwankung von Polynomen zwischen Gitterpunkten.” Mathematische Zeitschrift 86, 41–44 (1964). DOI 10.1007/BF01111276.
- [Sünderhauf, Nemeth, Walayat, Patterson and Berntson][InversePolynomials]. “Matrix inversion polynomials for the quantum singular value transformation.” arXiv:2507.15537v1.
- [Dong, Meng, Whaley and Lin][DMWL]. “Efficient phase-factor evaluation in quantum signal processing.” arXiv:2002.11649v2.
- [Dong, Lin, Ni and Wang][DLNW]. “Robust iterative method for symmetric quantum signal processing in all parameter regimes.” arXiv:2307.12468v1.
- [An, Childs and Lin][ACL]. “Quantum algorithm for linear non-unitary dynamics with near-optimal dependence on all parameters.” arXiv:2312.03916v2.
- [Low and Somma][LowSomma]. “Optimal quantum simulation of linear non-unitary dynamics.” arXiv:2508.19238v2.
- [Trefethen][ATAP]. Approximation Theory and Approximation Practice, SIAM (2013), Chapter 19. ISBN 978-1-61197-239-9.
- [NIST DLMF][DLMF]. Section 3.5(v), Gauss quadrature, Eqs. 3.5.19 and 3.5.21.
- [Pocrnic, Johnson, Katabarwa and Wiebe][Pocrnic]. “Constant-Factor Improvements in Quantum Algorithms for Linear Differential Equations.” arXiv:2506.20760v2.
- [Childs and Wiebe][LCU]. “Hamiltonian simulation using linear combinations of unitary operations.” arXiv:1202.5822v1.
- [Low and Chuang][Qubitization]. “Hamiltonian Simulation by Qubitization.” arXiv:1610.06546v3.
- [Camps, Lin, Van Beeumen and Yang][Camps]. “Explicit quantum circuits for block encodings of certain sparse matrices.” arXiv:2203.10236v4.
- [Oseledets][TT]. “Tensor-Train Decomposition.” SIAM Journal on Scientific Computing 33, 2295–2317 (2011). DOI 10.1137/090752286.
- [Ran][Ran]. “Encoding of matrix product states into quantum circuits of one- and two-qubit gates.” arXiv:1908.07958v2.
- [Möttönen, Vartiainen, Bergholm and Salomaa][Mottonen]. “Transformation of quantum states using uniformly controlled rotations.” arXiv:quant-ph/0407010v1.
- [Shende, Bullock and Markov][SBM]. “Synthesis of Quantum Logic Circuits.” arXiv:quant-ph/0406176v5.
- [Krol and Al-Ars][ZXZ]. “Beyond Quantum Shannon Decomposition.” arXiv:2403.13692v2.
- [Childs, Su, Tran, Wiebe and Zhu][Trotter]. “Theory of Trotter Error with Commutator Scaling.” Physical Review X 11, 011020 (2021). DOI 10.1103/PhysRevX.11.011020. Preprint arXiv:1912.08854v3 has different proposition and equation numbers.
- [Leng, Hickman, Li and Wu][Leng]. “Quantum Hamiltonian Descent.” arXiv:2303.01471v1.
- [Kushnir, Leng, Peng, Fan and Wu][QHDOPT]. “QHDOPT: A Software for Nonlinear Optimization with Quantum Hamiltonian Descent.” arXiv:2409.03121v1.
- [Wu et al.][WuCoverage]. “Benchmarking and Resource Analysis for Augmented-Lagrangian Quantum Hamiltonian Descent.” arXiv:2605.12066. Algorithm equation locators use [arXiv:2605.12066v1][Wu].
- [Liu et al.][Liu]. “Encoding Choices and Fault-Tolerant Resource Estimates for Digital Quantum Hamiltonian Descent.” arXiv:2607.16996v1.
- [Strang][Strang]. “On the Construction and Comparison of Difference Schemes.” SIAM Journal on Numerical Analysis 5, 506–517 (1968). DOI 10.1137/0705041.
- [Ogita, Rump and Oishi][CompSum]. “Accurate Sum and Dot Product.” SIAM Journal on Scientific Computing 26, 1955–1988 (2005). DOI 10.1137/030601818.
- [Coppersmith][Coppersmith]. “An approximate Fourier transform useful in quantum factoring.” arXiv:quant-ph/0201067v1.
- [Rockafellar][PHR]. “A dual approach to solving nonlinear programming problems by unconstrained optimization.” Mathematical Programming 5, 354–373 (1973). DOI 10.1007/BF01580138.
- [Birgin and Martínez][BM]. Practical Augmented Lagrangian Methods for Constrained Optimization, SIAM (2014). DOI 10.1137/1.9781611973365.
- [Ross and Selinger][RossSelinger]. “Optimal ancilla-free Clifford+T approximation of z-rotations.” arXiv:1403.2975v3.
- [Higham][Higham]. Accuracy and Stability of Numerical Algorithms, second edition, SIAM (2002). DOI 10.1137/1.9780898718027.
- [Zhu, Argentati and Knyazev][ResidualBounds]. “Bounds for the Rayleigh quotient and the spectrum of self-adjoint operators.” SIAM Journal on Matrix Analysis and Applications 34, 244–256 (2013), arXiv:1207.3240v2.

[Hoeffding]: https://doi.org/10.1080/01621459.1963.10500830
[ClopperPearson]: https://doi.org/10.1093/biomet/26.4.404
[QCELS]: https://arxiv.org/abs/2211.11973v2
[SPE]: https://arxiv.org/abs/2110.12071v2
[RFE]: https://arxiv.org/abs/2209.11322v3
[RWPE]: https://arxiv.org/abs/2208.04526v1
[CoherentQPE]: https://arxiv.org/abs/quant-ph/9708016v1
[Kirby]: https://arxiv.org/abs/2208.00567v4
[Oumarou]: https://arxiv.org/abs/2603.15552v1
[GCIM2023]: https://arxiv.org/abs/2212.09205v1
[GCIM2024]: https://arxiv.org/abs/2312.07691v3
[Epperly]: https://arxiv.org/abs/2110.07492v2
[Schuld]: https://arxiv.org/abs/1811.11184v1
[Wierichs]: https://arxiv.org/abs/2107.12390v3
[SpinAdapted]: https://arxiv.org/abs/2511.13485v2
[CKS]: https://arxiv.org/abs/1511.02306v2
[Dalzell]: https://arxiv.org/abs/2406.12086v2
[Martyn]: https://arxiv.org/abs/2105.02859v5
[GSLW]: https://arxiv.org/abs/1806.01838v1
[EZ]: https://doi.org/10.1007/BF01111276
[InversePolynomials]: https://arxiv.org/abs/2507.15537v1
[DMWL]: https://arxiv.org/abs/2002.11649v2
[DLNW]: https://arxiv.org/abs/2307.12468v1
[ACL]: https://arxiv.org/abs/2312.03916v2
[LowSomma]: https://arxiv.org/abs/2508.19238v2
[ATAP]: https://www.chebfun.org/ATAP/chap19.m
[DLMF]: https://dlmf.nist.gov/3.5#v
[Pocrnic]: https://arxiv.org/abs/2506.20760v2
[LCU]: https://arxiv.org/abs/1202.5822v1
[Qubitization]: https://arxiv.org/abs/1610.06546v3
[Camps]: https://arxiv.org/abs/2203.10236v4
[TT]: https://doi.org/10.1137/090752286
[Ran]: https://arxiv.org/abs/1908.07958v2
[Mottonen]: https://arxiv.org/abs/quant-ph/0407010v1
[SBM]: https://arxiv.org/abs/quant-ph/0406176v5
[ZXZ]: https://arxiv.org/abs/2403.13692v2
[Trotter]: https://doi.org/10.1103/PhysRevX.11.011020
[Leng]: https://arxiv.org/abs/2303.01471v1
[QHDOPT]: https://arxiv.org/abs/2409.03121v1
[Wu]: https://arxiv.org/abs/2605.12066v1
[WuCoverage]: https://arxiv.org/abs/2605.12066
[Liu]: https://arxiv.org/abs/2607.16996v1
[Strang]: https://doi.org/10.1137/0705041
[CompSum]: https://doi.org/10.1137/030601818
[Coppersmith]: https://arxiv.org/abs/quant-ph/0201067v1
[PHR]: https://doi.org/10.1007/BF01580138
[BM]: https://doi.org/10.1137/1.9781611973365
[RossSelinger]: https://arxiv.org/abs/1403.2975v3
[Higham]: https://doi.org/10.1137/1.9780898718027
[ResidualBounds]: https://arxiv.org/abs/1207.3240v2
