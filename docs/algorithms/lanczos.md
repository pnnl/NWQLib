# Chebyshev Lanczos

`Lanczos` estimates the lowest eigenvalue of a Hermitian matrix $H$ projected onto a Chebyshev-Krylov trial space. It writes $H=cI+\alpha K$ with the spectrum of $K$ in $[-1,1]$, obtains the Chebyshev moments $\mu_k=\langle\psi|T_k(K)|\psi\rangle$ of an initial state $\psi$, and forms the Gram matrix $S$ and the projected Hamiltonian on the trial vectors $T_i(K)\psi$, $0\le i<m$, where $m$ is `krylov_dimension`. It returns the lowest eigenvalue of the thresholded pencil $Hc=ESc$. The trial space and the projected matrices follow Kirby, Motta and Mezzacapo (arXiv:2208.00567v4, Sec. 3.1, Eqs. (13)–(19), published as Quantum 7, 1018 (2023), doi:10.22331/q-2023-05-23-1018). The shift and rescaling follow Oumarou et al. (arXiv:2603.15552v1, Sec. 2, Eqs. (1) and (4)), and the thresholded solve follows Epperly, Lin and Nakatsukasa (arXiv:2110.07492v2, Algorithm 1.1). The result is a projected estimate, without a claim that it identifies the ground state. A full requested dimension does not imply full trial rank or overlap with the ground state.

Use Lanczos when you can prepare one initial state and want the lowest eigenvalue in the Krylov space it generates. Use [GCiM](gcim.md) when you supply the trial states yourself or grow them from a pool of generators, and [QPE](qpe.md) when you want an eigenphase or energy estimated from controlled time evolution of a prepared state.

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.lanczos import Lanczos

problem = Eigenproblem(A=[[1.5, -1], [-1, .5]])
method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
result = solve(problem, method=method, seed=7)
print(result.eigenvalue, result.kept_rank)
```

```text
-0.11803398874989468 2
```

The eigenvalues of this matrix are $1\pm\sqrt5/2$. With two kept trial directions the trial space is the whole two-dimensional space, so the projected value equals the lowest eigenvalue $1-\sqrt5/2\approx-0.1180$. A trial space need not reach the ground state. For `diag(0,1)` with initial state $|1\rangle$, the method returns 1 even if the requested Krylov dimension is 2. Replace `A` and `initial_state` to treat your own operator. The eigenvalue notebook (`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`) applies Lanczos, ADAPT-GCIM and QCELS to a stretched H4 chain, and the [method API](../api/algorithms/lanczos.md) lists every option.

## Execution and readout

By default the moments come from a quantum circuit built from a coefficient preparation (PREP), a signed SELECT of the Pauli terms, a reflection and parity readout. `shots=N` means N shots per elementary setting. Without `shots`, the method uses exact probabilities on a compatible backend. Exact probability readout is limited to 20 measured qubits per marginal (`MAX_READOUT_ENTRY_BITS`). Planning refuses a wider marginal, and `shots=N` selects sampled readout instead. `execution="classical"` explicitly evaluates the Chebyshev recurrence through the same Run and analysis. No path substitutes a reference eigensolve.

With exact probabilities on a backend that records trajectory observations, Lanczos prepares the initial walk state once and records the requested moments while it advances the walk, so all degrees through $2m-1$ take $m-1$ walk steps ([Proposition 58](../mathematics.md#r58)). Readout snapshots leave the state that the walk continues from unchanged. Sampled execution prepares each moment setting independently. With counts, one measurement batch per degree parity holds one setting per degree. A setting prepares the reference and the coefficient state, applies $\lfloor k/2\rfloor$ walk steps, then applies `PREP^dagger` and reads the index register (even $k$), or applies the coherent SELECT readout and reads the index and system registers (odd $k$). The classical recurrence computes the same ideal Chebyshev moments by a different numerical route and runs only when selected explicitly.

An exact-probability Lanczos [`Plan`](../glossary.md#plan), the construction and its costs computed before any circuit exists, needs a target that declares the `trajectory` readout with the `multi_position` and `views` features. Qiskit Aer's statevector target and the local NWQ-Sim adapter's CPU/SV route declare them and run it. Targets without them, such as Aer's counts target, the other NWQ-Sim routes (counts only) and the IBM Runtime, IonQ and Nexus targets, reject the `Plan` before any work.

## Inputs and their cost {#choose-input-access-and-understand-its-cost}

`Eigenproblem` holds the Hermitian matrix, and `Lanczos` holds the initial state and the trial space. `krylov_dimension` defaults to `min(8, original_dimension)`, and a larger requested value is invalid. If `initial_state` is omitted, a power-of-two domain uses the recorded random product state, and a small non-power-of-two domain uses a normalized complex Gaussian vector in the original coordinates. Explicit sectors require a compatible supplied preparation. Planning refuses, with `ApplicabilityError`, a Problem whose `subspace` differs from the Chebyshev trial subspace that it selects. Dense padding uses `trace(A)/d` on dummy coordinates and zeros in the supplied initial state. The input state is normalized for the trial space, and its original norm is recorded. Moments that are known algebraically are not recorded as measurements.

A scalar operator $cI$, including $c=0$, returns its algebraic value without measurements or circuit stages, even with positive requested shots. The shot argument must be None or positive, and `shots=0` is not an accepted request.

Classical Lanczos keeps dense, CSR, CSC and Pauli access in the original dimension, including a three-dimensional problem. It does not convert sparse matrices to dense or pad classical trial vectors. Automatic quantum conversion accepts explicit dense matrices of original dimension at most 16, and compact Pauli inputs keep their supported widths. Sparse quantum conversion and larger explicit dense transforms require the method option `input_conversion="dense_pauli"`:

```python
from scipy.sparse import csr_matrix
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.lanczos import Lanczos

problem = Eigenproblem(A=csr_matrix([[1., .2], [.2, -1.]]))
classical = solve(
    problem,
    method=Lanczos(initial_state=[1, 0], krylov_dimension=2),
    execution="classical", seed=7,
)
quantum = solve(
    problem,
    method=Lanczos(initial_state=[1, 0], krylov_dimension=2,
                   input_conversion="dense_pauli",
                   max_conversion_work=100_000_000),
    seed=7,
)
print(classical.eigenvalue, quantum.eigenvalue)
```

```text
-1.019803902718558 -1.0198039027185546
```

Both values approximate the lowest eigenvalue $-\sqrt{1.04}$. `dense_pauli` converts the original input under `max_bytes` and a `q*D²` transform-work cap, where `D` includes the padding of the quantum embedding. Dummy diagonal entries use `trace(A)/d`, and the supplied state is zero-padded. This option does not change the original Problem. Only exact-zero Pauli branches are omitted.

## Read the result

| Field | Meaning |
| --- | --- |
| `eigenvalue` | Lowest physical Ritz value $c+\alpha x$ of the kept pencil, or None when unavailable. A projected estimate, not an identified ground state. |
| `eigenvalues` | Physical Ritz values of the kept pencil in ascending order. |
| `kept_rank` | Number of Gram (overlap) eigenvalues above the cutoff. |
| `overlap_spectrum` | Eigenvalues of the Gram matrix $S$ before thresholding. |
| `cutoff`, `cutoff_source` | The overlap-eigenvalue cutoff and the rule that produced it, for example `deterministic_default` for exact moments, `empirical_gram_rms` for the default sampled policy or `hoeffding_gram_bound` for the confidence policy. |
| `gram_sampling_bound` | Conditional Hoeffding bound on the Gram perturbation, independent of the cutoff policy ([Gram noise and the confidence cutoff](#gram-noise-and-the-confidence-cutoff)). |
| `empirical_gram_noise_frobenius_rms` | Sample-variance estimate of the Gram noise scale, without a coverage claim. |
| `projected_backward_error` | Dimensionless backward error of the normalized pencil that the solve uses. It is not the residual of a state in the full Hilbert space, and it is not comparable with the GCiM value, which is stated on the physical pencil ([Projected quantities](../verification.md#projected-quantities)). |
| `moments`, `moment_variances` | The moments $\mu_0,\ldots,\mu_{2m-1}$ used by the analysis, None where missing, and their empirical sample-mean variances. |
| `failure`, `missing` | The reason no Ritz value is available, and the moment degrees without data. |
| `target_identification` | The statement that the value is a projected estimate, not an identified ground state. |

The raw Gram matrix is $S_{ij}=(\mu_{i+j}+\mu_{\lvert i-j\rvert})/2$, and its diagonal generally differs from one. The projected Hamiltonian follows the Chebyshev product identity, and physical Ritz values are restored as $c+\alpha x$. Identity terms therefore shift the projected matrix by $cS$, not by a coordinate identity ([Result 8](../mathematics.md#r8)).

## Reanalyze with another cutoff {#continue-from-the-same-observations}

```python
revised = result.analyze(overlap_cutoff=1e-9)
result.save("lanczos-result")
```

Reanalysis can change the cutoff, `overlap_cutoff_policy`, `overlap_noise_multiplier` and `overlap_failure_probability`, using the same moments and observations. Exact moments use the numerical floor `1e-12`. Sampled moments use the default regularization `max(1e-12, overlap_noise_multiplier*RMS)`, where RMS is the empirical estimate of the Gram Frobenius noise. The multiplier 1 is a tunable exploratory choice. It has no sampling-coverage or Ritz-energy guarantee, and zero empirical variance does not prove zero noise. The Result reports this policy and the independent conditional sampling bound separately.

Even with exact readout, the moments and the projected solve are evaluated in finite precision, and the numerical Gram cutoff can remove nonzero directions of the algebraic Krylov space. Reaching its mathematical dimension therefore need not reproduce the exact ground energy. Increasing `krylov_dimension` can also increase the reported energy, because thresholding each Gram matrix can produce trial spaces that are not nested. Inspect `kept_rank`, `overlap_spectrum`, `cutoff` and `cutoff_source` alongside the energy. Exact Rayleigh–Ritz is monotone on nested spaces and recovers the lowest reachable eigenvalue at Krylov exhaustion when the full space is used.

## If the energy looks wrong

Each row points to one possible source of error. A comparison isolates that source only when the compared solves use the same trial state, frame and backend assumptions. Pass `initial_state` explicitly to every solve you compare, because an omitted state is drawn from the planning random generator. Compare `center` and `alpha` in `result.plan.reconstruction`, because classical and quantum planning can select different frames.

| Check | What a difference shows |
| --- | --- |
| Solve again without `shots` on the default backend | Exact readout has zero sampling error ([sampling error by method](../verification.md#sampling-error-by-method)). A change from the sampled value comes from sampling and from the sampled cutoff, which can keep fewer directions ([Reanalyze with another cutoff](#continue-from-the-same-observations)). If the sampled run used a noisy simulator or hardware, the difference also contains that backend's error. |
| Solve with `execution="classical"` | The classical recurrence computes ideal Chebyshev moments without a circuit, in the frame that classical planning selects ([Execution and readout](#execution-and-readout)). When `center` and `alpha` match those of the circuit run, a difference from the exact-readout circuit value comes from the circuit route or from roundoff, not from sampling. |
| For a small input, compare the classical value with the smallest eigenvalue from `numpy.linalg.eigvalsh` | The value is a projected estimate. A difference larger than floating-point roundoff means the trial space of `initial_state` and `krylov_dimension` does not reach the ground state, or the Gram cutoff removed directions. |
| Read `kept_rank`, `overlap_spectrum`, `cutoff` and `cutoff_source` | A `kept_rank` below `krylov_dimension` means the cutoff removed trial directions. `result.analyze` can apply another cutoff to the same moments without new measurements. |
| Compare `gram_sampling_bound` with `overlap_spectrum` | The confidence policy keeps a direction only when its Gram eigenvalue exceeds `max(1e-12, 2*m*e)`, at least twice the bound, and `result.analyze(overlap_cutoff_policy="confidence")` applies it without new measurements. The bound covers Gram sampling noise only and excludes circuit bias, lost subspace information and Ritz-energy error ([Gram noise](#gram-noise-and-the-confidence-cutoff)). |
| Raise `krylov_dimension` or change `initial_state` | This changes the trial space. A larger dimension can also raise the reported energy, because thresholded trial spaces need not be nested. |

## Gram noise and the confidence cutoff

The RMS estimates the square root of $\mathbb E\lVert\Delta S\rVert_F^2=\sum_{ij}\operatorname{Var}(S_{ij})$ from the sample-mean variances $\sigma_k^2$ of independent moment estimates, each a per-shot variance divided by its shot count. Collecting entries by degree gives $\mathrm{RMS}^2=\sum_{k<2m-1}w_k\sigma_k^2$, with the weights $w_k$ of [Proposition 12](../mathematics.md#r12).

For $r$ sampled moments entering the Gram matrix, independent shots in $[-1,1]$ and tail allocation $\delta$, Hoeffding's Theorem 2, Eq. (2.6), with a two-sided union bound gives $e=\sqrt{2\log(2r/\delta)/n_{\min}}$ ([Hoeffding (1963), p. 16](https://doi.org/10.1080/01621459.1963.10500830)). The identity $S_{ij}=(\mu_{i+j}+\mu_{\lvert i-j\rvert})/2$ then gives $\lVert\Delta S\rVert_2\le\lVert\Delta S\rVert_F\le me$ ([Proposition 12](../mathematics.md#r12)). The code evaluates this bound and the empirical RMS in $O(m)$ scalar work from the returned shots, without constructing a matrix to validate them. Reading the bound as a bound on the ideal Gram matrix also requires unbiased measurements. The bound is evaluated in binary64 and excludes circuit bias, lost subspace information and Ritz-energy error.

The explicit `overlap_cutoff_policy="confidence"` uses `max(1e-12, 2*m*e)`. Weyl's inequality then places the corresponding eigenvalue of the noise-free Gram matrix above $me$ for each kept direction. This can remove useful trial directions at modest shot budgets. Keeping a Gram eigenvalue $\lambda$ requires $n_{\min}>8m^2\log(2r/\delta)/\lambda^2$. At $m=8$, $r=14$ and $\delta=0.05$, this is about $3240/\lambda^2$ shots per setting. The Result prints the kept rank and reports an unavailable estimate if every direction is removed. `result.analyze(overlap_cutoff_policy="confidence", overlap_failure_probability=.01)` makes this choice explicitly, without measurement. The default exploratory policy supports smaller budgets, such as the one in the next section. Neither policy establishes the additional hypotheses of the energy-error theorems of Kirby et al. (arXiv:2208.00567v4) or Epperly et al. (arXiv:2110.07492v2).

## Allocate shots by sensitivity

`Lanczos(sampling=SensitivitySampling(total_shots=1400, pilot_fraction=.2))` explicitly selects a pilot stage followed by a main stage. Both fields are required, and a scalar `shots` conflicts with this choice. Every setting needs at least two shots per stage. The energy derivative includes the rotation of the kept Gram subspace ([Proposition 10](../mathematics.md#r10)). If an assumption of the allocation fails, the method falls back to uniform allocation without another pilot. The pilot and main observations and the analysis attempts are saved with the Run's iteration state. `run.resume(reanalyze=True)` explicitly retries an interrupted analysis and never resubmits measurements. Completed results use `result.analyze(...)`.

For independent moment estimates with per-shot variances $v_k$, local energy derivatives $g_k$ and shot counts $n_k$, the linearized energy variance at fixed total shots $N$ satisfies

```math
\sum_k\frac{g_k^2v_k}{n_k}\ge\frac{\left(\sum_k\lvert g_k\rvert\sqrt{v_k}\right)^2}{N},
```

with equality when $n_k$ is proportional to $\lvert g_k\rvert\sqrt{v_k}$ ([Proposition 11](../mathematics.md#r11), by Cauchy–Schwarz). NWQLib uses pilot estimates of these weights. [Oumarou et al., arXiv:2603.15552v1, Eqs. (59)–(60)](https://arxiv.org/html/2603.15552v1) instead allocate by $\lvert g_k\rvert$. The variance-weighted rule solves the continuous local allocation problem. Estimated derivatives, integer allocations, pilot floors and nonlinear energy error prevent a claim of optimal final accuracy or optimal circuit-time cost.

Only main-stage moments enter the final estimate. Given the pilot, the main-stage sample counts are fixed, so fresh independent main observations support the usual fixed-sample moment analysis. Combining the pilot with the data-dependent main observations would require a statistical treatment of those adaptive weights. The pilot stays available with its allocation record. This separation does not make the nonlinear Ritz estimate unbiased or provide a physical-error bound.

When comparing sampling strategies, match total shots including the pilot, and inspect the pilot and main allocation, the fallback status and the resulting Gram rank. A pilot can reduce the main-stage budget enough to change the trial-space rank, so improved allocation at fixed main shots does not establish improved eigenvalue accuracy at fixed total shots. Report such a comparison for its workload, and distinguish its projected eigenvalue from independently established ground-state evidence.

## Check a result

For stored projected checks, pass `ProjectedVerificationOptions` to `result.verify(checks=...)`. These checks do not identify the full-space ground state.

`EnergyShiftOptions.for_result` checks that two results differ by a constant energy shift. It chooses an exact Pauli-table or original matrix-entry relation from the existing input. Dense, CSR and CSC endpoints store only existing nonzero entries when explicitly requested, under the method byte limit, and ordinary solve and reanalysis do not create or cache this table. The check verifies the complete complex relation `A_target-A_baseline=shift*I`, including implicit diagonal zeros, before comparing physical Ritz values. Different affine normalizations do not change this original-input relation. Relations across Pauli and matrix formats are not proven automatically, and an explicitly asserted relation remains an unresolved assumption.

### Compare trajectory and separately prepared moments {#comparing-trajectory-and-separately-prepared-moments}

On the exact trajectory, each temporary readout view is followed by its inverse before the walk continues. The inverse of an odd-degree view is the adjoint of the forward basis-change table, target by target. It is the multiplexer of the adjoint matrices or, for a target without controls, the adjoint matrix unitary, so the Gram defect of the stored table is the represented residual of each odd-degree view and its inverse. A comparison of a trajectory moment with a separately prepared moment needs bounds for both routes' state and scalar readout errors, including the accumulated restoration residuals of earlier views. For degree $k$,

```math
\lvert\mu_{k,\mathrm{traj}}-\mu_{k,\mathrm{prefix}}\rvert\le D(\Delta_{\mathrm{traj},k})+D(\Delta_{\mathrm{prefix},k})+a_{\mathrm{traj},k}+a_{\mathrm{prefix},k},\qquad D(x)=2x+x^2,
```

where the $\Delta$ are the two routes' state-error bounds to the same ideal readout and the $a$ are their scalar readout-error bounds (`readout.py::reduce_moments`). A preparation record with an unresolved exclusion, such as Aer's `multiplexer` instruction, has no state bound, and the relation then supplies no numerical acceptance tolerance.

## Limits

- The value is a projected Ritz estimate without a ground-identification claim, and a full requested dimension does not imply full trial rank or ground-state overlap.
- Exact probability readout is limited to 20 measured qubits per marginal and runs on Aer's statevector target and NWQ-Sim's CPU/SV route ([Execution and readout](#execution-and-readout)).
- The default sampled cutoff is exploratory, and neither cutoff policy establishes the hypotheses of the energy-error theorems ([Gram noise and the confidence cutoff](#gram-noise-and-the-confidence-cutoff)).
- The Gram sampling bound excludes circuit bias, lost subspace information and Ritz-energy error.
- A larger `krylov_dimension` can raise the reported energy, because thresholded trial spaces need not be nested.
- Sensitivity allocation gives no optimal-accuracy, unbiasedness or physical-error claim ([Allocate shots by sensitivity](#allocate-shots-by-sensitivity)).

[Limitations and open work](../ROADMAP.md#chebyshev-lanczos) lists the open items for this method.

## Numerical scope and planning limits

Classical matrix input uses scaled Gershgorin row intervals, or the equal column intervals for Hermitian CSC storage (`enclosure_source="scaled_gershgorin"`), to select the affine frame $B=(A-cI)/\alpha$. Each radius excludes the diagonal before summation. A conservative binary64 roundoff budget covers the reduction independently of its sequential or pairwise order, under round-to-nearest binary64 arithmetic, gradual underflow and a one-ulp hypot assumption. The final half-width covers both outward-rounded endpoints about the stored center. A different valid frame can change finite-order moments and the overlap rank selected by a cutoff. The roundoff budget is not a formally proven interval bound. Pauli inputs keep their centered coefficient L1 enclosure (`"pauli_l1"`), which gives $B$ the unit L1 norm of the encoding of Kirby et al., arXiv:2208.00567v4, Section 2.1, after the identity term is removed. For Pauli input, Lanczos uses the accepted packed Pauli table to define its centered operator and SELECT address order. The reconstruction stores the scalar frame and the table's identifier. Readout signs, support masks and normalized coefficients are derived from that same table, so the classical action and the quantum decoder use a common term order. The reconstruction records `spectral_lower`, `spectral_upper`, `center`, `alpha`, `enclosure_source` and the operator association. The frame determines every moment, so later analysis never changes it. A wide enclosure can compress the effective spectrum and change the rank kept at a finite Gram cutoff. Equal full-rank trial spans recover the same Ritz spectrum. No full spectrum is computed to describe enclosure quality.

The classical product limit is checked before the initial vector is computed and counts all recurrence actions, reductions and known preparation work. Product-state expansion and supported standard circuit gates have formulas for their amplitude work. Custom circuit simulation work is explicitly unknown, not zero, and the recorded known-work limit does not promise a bound on total SDK CPU time or memory. Building a supplied circuit's state runs synchronously, and `Run.wait(timeout=...)` does not interrupt an active SDK call.

Input storage, classical recurrence products and projected analysis have separate finite method limits. The projected analysis runs after the moments are obtained, but its size depends only on `krylov_dimension`. Planning therefore checks it against `max_bytes` and `max_analysis_work`, and under `SensitivitySampling` also checks the pilot's solve and derivative, before any moment is obtained. Logical work is not measured CPU time or process memory.

## Source and code map

The method follows Kirby, Motta and Mezzacapo, *Exact and efficient Lanczos method on a quantum computer*, Quantum **7**, 1018 (2023), Eqs. (24)–(27), [DOI](https://doi.org/10.22331/q-2023-05-23-1018). Equation numbers in the table refer to these versions: Kirby, Motta and Mezzacapo, [arXiv 2208.00567v4](https://arxiv.org/abs/2208.00567v4), Oumarou et al., [arXiv 2603.15552v1](https://arxiv.org/abs/2603.15552v1), and Epperly, Lin and Nakatsukasa, [arXiv 2110.07492v2](https://arxiv.org/abs/2110.07492v2). Page numbers are those of these arXiv PDFs. The Oumarou HTML version uses the same equation numbers. A bare file name in the Code column is in `src/nwqlib/algorithms/lanczos/`, and other paths are relative to `src/nwqlib/`.

| Step | Source | Location | Code |
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
| Exact numerical floor, exploratory Gram RMS and explicit confidence cutoff | Empirical noise filtering and the independent Hoeffding bound, stated above and in [Proposition 12](../mathematics.md#r12) | Hoeffding (1963), doi:10.1080/01621459.1963.10500830, Theorem 2, Eq. (2.6), p. 16, and Eq. (1.4), p. 13 | `numerical.py::_resolve_overlap_cutoff` |
| Shot allocation from energy sensitivity to each moment | Oumarou | Sec. 3.3.2, Eqs. (59)–(60), p. 29 | `numerical.py::_sensitivity_weights`, `numerical.py::_allocate` |
| Measured pilot, weights `abs(g_k) sqrt(v_k)`, analytic derivative with kept-subspace rotation, pilot floor and main-stage-only estimate | NWQLib choices, and a first-order perturbation derivation for the derivative ([Propositions 10 and 11](../mathematics.md#r10)) | Notes below and the docstrings of `_sensitivity_weights`, `_moment_energy_derivatives` and `_apply_pilot_floor` | `numerical.py::_moment_energy_derivatives`, `numerical.py::_apply_pilot_floor`, `workflow.py` |

The NWQLib rows differ from the papers as follows.

- The coherent readout needs one sampled setting, or one exact trajectory point, per odd degree, and each shot returns one signed outcome for `<U>`. Kirby's step 3 needs one setting per term or per compatible Pauli-basis group. The price is a label-controlled basis change, whose table-size limit is in [Engineering constants](../ENGINEERING_CONSTANTS.md#chebyshev-lanczos-defaults). A Lanczos count outcome stores the SELECT index in its least significant bits. Its used row supplies a sign and the system-support mask. The signed value is the sign times the support parity, with zero for an unused address. Reflection readout returns +1 only for index zero. Vectorized decoding uses the same first and second moments and the returned shot counts. Exact probability reduction preserves the supplied mass without renormalization.
- Oumarou et al. allocate by `abs(g_k)` alone and evaluate `g_k` by automatic differentiation at noiseless moments. Here `g_k` is the analytic derivative at measured pilot moments. The continuous allocation proof and its limits are given above. Their rule coincides with the variance-weighted rule when every per-shot variance is equal.
- Only main-stage moments enter the estimate. The pilot outcomes chose the main sample sizes, so leaving them out keeps each main-stage sample a fixed-size sample given the frozen allocation.
- The separately reported Gram sampling bound controls sampling perturbations under the stated bounded-shot model. The default empirical cutoff is exploratory. It does not establish the additional conditions of Epperly's Theorem 2.7 or Kirby's Theorem 1 for energy error. Those also concern the projected Hamiltonian, trial-space overlap and spectral structure. No default reference solve is performed to test them.
