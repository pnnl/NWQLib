# Chebyshev Lanczos

Supply the Hermitian target and configure its trial space separately:

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.lanczos import Lanczos

problem = Eigenproblem(A=[[1.5, -1], [-1, .5]])
result = solve(problem, method=Lanczos(initial_state=[1, 0], krylov_dimension=2), seed=7)
print(result.eigenvalue)
```

The default quantum path obtains signed Chebyshev moments using the actual selected PREP, signed SELECT, reflection and parity readout. `shots=N` means N shots per elementary setting; omitted shots select exact probabilities on a compatible backend. Exact probability readout is limited to 20 measured qubits per marginal (`MAX_READOUT_ENTRY_BITS`). Planning refuses a wider marginal, and `shots=N` selects sampled readout instead. `execution="classical"` explicitly evaluates the selected Chebyshev recurrence through the same Run and analysis. Neither path substitutes a reference eigensolve.

On a backend with exact trajectory observations, Lanczos prepares the selected initial walk state once and records the requested moments while advancing the walk. All degrees through 2m-1 require m-1 walk steps. Sampled execution prepares each moment setting independently. The classical recurrence computes the same ideal Chebyshev moments through a different numerical execution and remains an explicitly selected classical mode.

For exact trajectories, each temporary readout view is followed by its selected inverse before continuation, and comparisons with separately prepared moments require bounds for both routes' state and scalar readout errors, including the accumulated restoration residuals of earlier views. The inverse of an odd-degree view is the adjoint of the forward basis-change table, target by target, as the multiplexer of the adjoint matrices or, for a target without controls, the adjoint matrix unitary, so the Gram defect of the stored table is the represented residual of each odd-degree view and its inverse. For degree k, the trajectory moment and a separately prepared moment satisfy `|mu_k,traj - mu_k,prefix| <= D(Delta_traj,k) + D(Delta_prefix,k) + a_traj,k + a_prefix,k` with `D(x) = 2x + x^2`, where the `Delta` are the two routes' state-error bounds to the same ideal readout and the `a` are their scalar readout-error bounds (`readout.py::reduce_moments`). A preparation record with an unresolved exclusion, such as Aer's native `multiplexer`, has no state bound, and the relation then supplies no numerical acceptance tolerance.

An exact-probability Lanczos `Plan` needs a target that declares the `trajectory` readout with the `multi_position` and `views` features. Qiskit Aer's statevector target and the local NWQ-Sim adapter's CPU/SV route declare them and run it. Targets without them, such as Aer's counts target, the other NWQ-Sim routes (native counts only) and the IBM Runtime, IonQ and Nexus targets, reject the `Plan` before any work.

For `H=cI+alpha K`, the moments are `mu_j=<psi|T_j(K)|psi>`. The raw Gram is `S_ij=(mu_(i+j)+mu_|i-j|)/2`; its diagonal generally differs from one. The projected Hamiltonian follows the Chebyshev product identity, and physical Ritz values are restored as `c+alpha*x`. Identity terms therefore shift by `cS`, not a coordinate identity. The input state is normalized for this trial space; its original physical scale remains recorded.

`krylov_dimension` defaults to `min(8, original_dimension)`. A requested larger value is invalid. Known algebraic moments do not become fictitious measurements. If initialization is omitted, a power-of-two domain uses the recorded random product state; a small non-power-of-two domain uses a normalized complex Gaussian vector in original coordinates. Explicit sectors require supplied compatible preparation. Dense padding uses `trace(A)/d` on dummy coordinates and zeros in the supplied initial state.

The result is a projected estimate, without a ground-identification claim. For example, `diag(0,1)` with initial state `|1>` returns 1 even if the requested Krylov dimension is 2. Full requested dimension does not imply full trial rank or ground overlap.

When comparing sampling strategies, match total shots including the pilot and inspect pilot/main allocation, fallback status and the resulting Gram rank. A pilot can reduce the main-stage budget enough to change the trial-space rank. Improved allocation at fixed main shots therefore does not establish improved eigenvalue accuracy at fixed total shots. Report this comparison for the selected workload and distinguish its projected eigenvalue from independently established ground-state evidence.

## Continue from the same observations

```python
revised = result.analyze(overlap_cutoff=1e-9)
result.save("lanczos-result")
```

Reanalysis can change the cutoff, `overlap_cutoff_policy`, `overlap_noise_multiplier` and `overlap_failure_probability` using the same moments and observations. Exact moments use the numerical floor `1e-12`. Sampled default regularization uses `max(1e-12, overlap_noise_multiplier*RMS)`, where RMS is the empirical estimate of the Gram Frobenius noise. Multiplier 1 is a tunable exploratory choice. It has no sampling-coverage or Ritz-energy guarantee, and zero empirical variance does not prove zero noise. The Result reports this policy and the independent conditional sampling bound separately.

Even with exact readout, the moments and the projected solve are evaluated in finite precision, and the numerical Gram cutoff can remove nonzero directions of the algebraic Krylov space. Reaching its mathematical dimension therefore need not reproduce the exact ground energy. Increasing `krylov_dimension` can also increase the reported energy, because thresholding each Gram matrix can produce trial spaces that are not nested. Inspect `kept_rank`, `overlap_spectrum`, `cutoff` and `cutoff_source` alongside the energy. `projected_backward_error` measures the normalized projected pencil, not the residual of a state in the full Hilbert space. Exact Rayleigh–Ritz is monotone on nested spaces and recovers the lowest reachable eigenvalue at Krylov exhaustion when the full space is used.

The RMS estimates the square root of `E||Delta S||_F^2 = sum_ij Var(S_ij)` from the sample-mean variances `sigma_k**2` of independent moment estimates, each a per-shot variance divided by its shot count. Since `S_ij=(mu_(i+j)+mu_|i-j|)/2`, `Var(S_ij)=(sigma_a**2+sigma_b**2+2[a=b]sigma_a**2)/4` with `a=i+j` and `b=|i-j|`, and collecting by degree gives `RMS**2 = sum_(k<2m-1) w_k sigma_k**2`. Here `w_k=(h_k+t_k+c_k)/4`, where `h_k=m-|k-(m-1)|` counts the entries with `i+j=k` and `t_k` counts those with `|i-j|=k` (`m` for `k=0`, `2(m-k)` for `0<k<m`, zero otherwise). `c_k` is twice the number of entries with `i+j=|i-j|=k`, which are `(0,0)` for `k=0` and `(0,k)`, `(k,0)` for `0<k<m`, so `c_k` is 2, 4 or zero.

For r sampled moments entering the Gram matrix, independent shots in [-1,1], and tail allocation delta, Hoeffding Theorem 2, Eq. (2.6), plus a two-sided union bound gives `e=sqrt(2*log(2*r/delta)/n_min)`. The identity `S_ij=(mu_(i+j)+mu_|i-j|)/2` then gives `||Delta S||_2 <= ||Delta S||_F <= m*e`. The code evaluates this bound and the empirical RMS in O(m) scalar work from actual populations, without constructing a matrix to validate them. [Hoeffding (1963), p. 16](https://doi.org/10.1080/01621459.1963.10500830) gives the bounded-variable inequality. Its ideal-Gram interpretation also requires unbiased measurements. The bound is evaluated in binary64 and excludes circuit bias, lost subspace information and Ritz-energy error.

Explicit `overlap_cutoff_policy="confidence"` uses `max(1e-12, 2*m*e)`. Weyl then places each kept population eigenvalue above m*e. This can remove useful trial directions at modest shot budgets. Keeping a Gram eigenvalue lambda requires `n_min > 8*m**2*log(2*r/delta)/lambda**2`. At m=8, r=14 and delta=.05, this is about `3240/lambda**2` shots per setting. The Result prints the kept rank and reports an unavailable estimate if every mode is removed. `result.analyze(overlap_cutoff_policy="confidence", overlap_failure_probability=.01)` makes this choice explicitly, without measurement. The default exploratory policy supports the smaller illustrative budget below. Neither policy establishes the additional hypotheses of Kirby's (arXiv:2208.00567v4) or Epperly's (arXiv:2110.07492v2) energy-error theorems.

`Lanczos(sampling=SensitivitySampling(total_shots=1400, pilot_fraction=.2))` explicitly selects pilot/main allocation. Both fields are required; scalar `shots` conflicts with this choice. Every setting needs at least two shots per stage. The derivative includes rotation of the kept Gram subspace. A failed premise chooses the documented uniform allocation without another pilot. Pilot/main observations and analysis attempts remain in the controller checkpoint. `run.resume(reanalyze=True)` explicitly retries an interrupted controller analysis; it never resubmits physical observations. Completed results use `result.analyze(...)`.

For independent moment estimates with per-shot variances `v_k`, local energy derivatives `g_k` and shot counts `n_k`, the linearized energy variance is `sum(g_k**2 * v_k / n_k)`. At fixed total shots `N`, Cauchy–Schwarz bounds this below by `(sum(abs(g_k) * sqrt(v_k)))**2 / N`, with equality when `n_k` is proportional to `abs(g_k) * sqrt(v_k)`. NWQLib uses pilot estimates of these weights. [Oumarou et al., arXiv:2603.15552v1, Eqs. (59)–(60)](https://arxiv.org/html/2603.15552v1) instead allocate by `abs(g_k)`. The variance-weighted rule solves the continuous local allocation problem. Estimated derivatives, integer allocations, pilot floors and nonlinear energy error prevent a claim of optimal final accuracy or optimal circuit-time cost.

Only main-stage moments enter the final estimate. Conditional on the pilot, their chosen sample counts are fixed, so fresh independent main observations support the usual fixed-population moment analysis. Combining the pilot with the data-dependent main population would require a statistical treatment of those adaptive weights. The pilot stays available with its allocation evidence. This separation does not make the nonlinear Ritz estimate unbiased or provide a physical-error bound.

For stored projected checks, pass `ProjectedVerificationOptions` to `result.verify(checks=...)`; these checks do not identify the full-space ground state. Input storage, classical recurrence products and projected analysis have separate finite method controls. The projected analysis runs after the moments are obtained, but its size depends only on `krylov_dimension`. Planning therefore checks it against `max_bytes` and `max_analysis_work`, and under `SensitivitySampling` also checks the pilot's solve and derivative, before any moment is obtained. Logical work is not measured CPU time or process RSS.

See [the method API](../api/algorithms/lanczos.md) and the eigenvalue notebook (`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`), which applies Lanczos, ADAPT-GCIM and QCELS to a stretched H4 chain.

Reference: Kirby, Motta and Mezzacapo, *Exact and efficient Lanczos method on a quantum computer*, Quantum **7**, 1018 (2023), Eqs. (24)–(27), [DOI](https://doi.org/10.22331/q-2023-05-23-1018). The [source and code map](#source-and-code-map) gives the equation behind each step, including the allocation and thresholding sources.

## Choose input access and understand its cost

Classical Lanczos keeps dense, CSR, CSC and Pauli access in the original dimension, including a three-dimensional problem. It does not convert sparse matrices to dense or pad classical trial vectors. Classical matrix input uses scaled Gershgorin row intervals, or the equal column intervals for Hermitian CSC storage, to select the affine frame `B=(A-cI)/alpha`. Each radius excludes the diagonal before summation. A conservative binary64 roundoff budget covers the reduction independently of its sequential or pairwise order, under the stated arithmetic and hypot assumptions (round-to-nearest binary64 arithmetic, gradual underflow and a one-ulp hypot premise). The final half-width covers both outward-rounded endpoints about the stored center. A different valid frame can change finite-order moments and the overlap rank selected by a cutoff. The roundoff budget is not a formally proven interval bound. Pauli inputs keep their centered coefficient L1 enclosure. For Pauli input, Lanczos uses the accepted packed Pauli table to define its centered operator and SELECT address order. The reconstruction stores the scalar frame and table identity. Readout signs, support masks and normalized coefficients are derived from that same table, so the classical action and quantum decoder use a common term order. The reconstruction records `spectral_lower`, `spectral_upper`, `center`, `alpha`, `enclosure_source` and the actual operator association. A wide enclosure can compress the effective spectrum and change the rank kept at a finite Gram cutoff. Equal full-rank trial spans recover the same Ritz spectrum. No full spectrum is computed to describe enclosure quality.

Automatic quantum conversion accepts explicit dense matrices of original dimension at most 16. Compact Pauli inputs keep their supported widths. Sparse quantum conversion and larger explicit dense transforms require the method option:

```python
from scipy.sparse import csr_matrix

problem = Eigenproblem(A=csr_matrix([[1., .2], [.2, -1.]]))
classical = solve(problem, method=Lanczos(initial_state=[1, 0], krylov_dimension=2),
                  execution="classical", seed=7)
quantum = solve(problem, method=Lanczos(initial_state=[1, 0], krylov_dimension=2,
                                      input_conversion="dense_pauli",
                                      max_conversion_work=100_000_000), seed=7)
```

`dense_pauli` realizes this original input under `max_bytes` and a `q*D²` transform-work cap, where `D` includes the quantum embedding. Dummy diagonal entries use `trace(A)/d`; the supplied state is zero-padded. This option does not change the original Problem. Only exact-zero Pauli branches are omitted.

The classical product cap is checked before materializing the seed and counts all recurrence actions, reductions and known preparation work. Product-state expansion and supported standard circuit gates have amplitude-work laws. Custom circuit simulation work is explicitly unknown, not zero; the recorded known-work cap does not promise a total SDK CPU/RSS bound. Supplied circuit materialization runs synchronously; `Run.wait(timeout=...)` does not preempt an active SDK call. Scalar `cI`, including `c=0`, returns its algebraic value without measurements or stage bindings, even with positive requested shots. The shot argument itself must be None or positive, and `shots=0` is not an accepted request.

`EnergyShiftOptions.for_result` chooses an exact Pauli-table or original matrix-entry relation from the existing input. Dense/CSR/CSC endpoints store only existing nonzero entries when explicitly requested, under the method byte cap; ordinary solve and reanalysis do not create or cache this table. The check verifies the complete complex relation `A_target-A_baseline=shift*I`, including implicit diagonal zeros, before comparing physical Ritz values. Different affine normalizations do not change this original-input relation. Pauli/matrix cross-format relations are not proven automatically. An explicitly asserted relation remains an unresolved premise.

## Source and code map

Equation numbers refer to these versions: Kirby, Motta and Mezzacapo, [arXiv 2208.00567v4](https://arxiv.org/abs/2208.00567v4), Oumarou et al., [arXiv 2603.15552v1](https://arxiv.org/abs/2603.15552v1), and Epperly, Lin and Nakatsukasa, [arXiv 2110.07492v2](https://arxiv.org/abs/2110.07492v2). Page numbers are those of these arXiv PDFs. The Oumarou HTML version uses the same equation numbers. A bare file name in the Owner column is in `src/nwqlib/algorithms/lanczos/`, and other paths are relative to `src/nwqlib/`.

| Step | Source | Location | Owner |
| --- | --- | --- | --- |
| Chebyshev trial space, equal to the power Krylov space of `H` | Kirby | Sec. 3.1, Eqs. (13)–(14), p. 5 | `method.py::Lanczos.plan` |
| Shift and rescaling `H = cI + alpha K` with the spectrum of `K` in [-1,1], and physical eigenvalues `c + alpha x` | Oumarou | Sec. 2, Eqs. (1) and (4), pp. 8–9 | `method.py::Lanczos.plan`, `numerical.py::restore_ritz_values` |
| Signed Pauli encoding of `K` with unit L1 norm, after the identity term is removed | Kirby | Sec. 2.1, Eqs. (2)–(4), p. 3 | `readout.py::packed_lanczos_census`, `blocks/selection.py::select_signed_pauli` |
| Walk `RU`, whose powers block-encode `T_k(K)` | Kirby | Lemma 1, Eqs. (6)–(7), p. 4 | `method.py::_selected_blocks`, `method.py::_moment_program` |
| Moment `mu_k` after `floor(k/2)` walk steps, reading `R` for even `k` and `U` for odd `k` | Kirby, restated by Oumarou | Kirby Eqs. (24)–(25), p. 5, Eq. (26) and Sec. 3.1, steps 1–2, p. 6. Oumarou Eqs. (16)–(18), p. 11 | `method.py::_trajectory_program` (exact, one walk with a reversible view per moment), `method.py::_moment_program` (sampled), `readout.py::signed_outcomes`, `readout.py::outcome` |
| Coherent SELECT-basis readout, one sampled setting or one exact trajectory point per odd degree | NWQLib choice. It estimates the same `<U>` of Kirby Eq. (27) as Kirby's per-term step 3 | Kirby Sec. 3.1, step 3, p. 6 | `blocks/selection.py::select_pauli_readout`, `readout.py::LanczosReadout`, `readout.py::signed_outcomes` |
| Gram `S` and projected `K` from moments, with weights 1/2 and 1/4 | Kirby, restated by Oumarou | Kirby Eqs. (16), (17), (19), p. 5. Oumarou Eqs. (12), (14), (15), pp. 10–11 | `numerical.py::_projected_matrices` |
| Physical pencil `c S + alpha K` | Same shift and rescaling applied to the projected matrix | Oumarou Sec. 2, Eq. (1), p. 8 | `numerical.py::reconstruct` |
| All `2m` classical moments from `m` operator applications | Chebyshev product identity | Kirby Eq. (16), p. 5 | `numerical.py::chebyshev_moments` |
| Thresholded solve of the pencil `H c = E S c` | Epperly, also Kirby Sec. 4 | Epperly Algorithm 1.1, p. 5. Kirby Eq. (10), p. 4 | `_solve_projected_pencil` in `src/nwqlib/_projected_eigensolver.py` |
| Exact numerical floor, exploratory Gram RMS and explicit confidence cutoff | Empirical noise filtering and the independent Hoeffding bound, derived above | Hoeffding (1963), doi:10.1080/01621459.1963.10500830, Theorem 2, Eq. (2.6), p. 16, and Eq. (1.4), p. 13 | `numerical.py::_resolve_overlap_cutoff` |
| Shot allocation from energy sensitivity to each moment | Oumarou | Sec. 3.3.2, Eqs. (59)–(60), p. 29 | `numerical.py::_sensitivity_weights`, `numerical.py::_allocate` |
| Measured pilot, weights `abs(g_k) sqrt(v_k)`, analytic derivative with kept-subspace rotation, pilot floor and main-stage-only estimate | NWQLib choices, and a first-order perturbation derivation for the derivative | Notes below and the docstrings of `_sensitivity_weights`, `_moment_energy_derivatives` and `_apply_pilot_floor` | `numerical.py::_moment_energy_derivatives`, `numerical.py::_apply_pilot_floor`, `workflow.py` |

The NWQLib rows differ from the papers as follows.

- The coherent readout needs one sampled setting, or one exact trajectory point, per odd degree, and each shot returns one signed outcome for `<U>`. Kirby's step 3 needs one setting per term or per compatible Pauli-basis group. The price is a label-controlled basis change, whose table-size contract is in [Engineering constants](../ENGINEERING_CONSTANTS.md#chebyshev-lanczos-defaults). A Lanczos count outcome stores the SELECT index in its least significant bits. Its used row supplies a sign and the system-support mask. The signed value is the sign times the support parity, with zero for an unused address. Reflection readout returns +1 only for index zero. Vectorized decoding uses the same first and second moments and the returned count population. Exact probability reduction preserves the supplied mass without renormalization.
- Oumarou et al. allocate by `abs(g_k)` alone and evaluate `g_k` by automatic differentiation at noiseless moments. Here `g_k` is the analytic derivative at measured pilot moments. The continuous allocation proof and its limits are given above. Their rule coincides with the variance-weighted rule when every per-shot variance is equal.
- Only main-stage moments enter the estimate. The pilot outcomes chose the main sample sizes, so leaving them out keeps each main population a fixed-size sample given the frozen allocation.
- The separately reported Gram sampling bound controls sampling perturbations under the stated bounded-shot model. The default empirical cutoff is exploratory. It does not establish the additional conditions of Epperly's Theorem 2.7 or Kirby's Theorem 1 for energy error. Those also concern the projected Hamiltonian, trial-space overlap and spectral structure. No default reference solve is performed to test them.
