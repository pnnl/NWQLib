# GCiM

<a id="gcim-and-adapt"></a>GCiM, the generator-coordinate-inspired method, computes the smallest eigenvalue of the projected generalized eigenproblem $Hf=ESf$, where $H_{ij}=\langle\phi_i|A|\phi_j\rangle$ and $S_{ij}=\langle\phi_i|\phi_j\rangle$ for normalized basis states $\phi_i$ and the Hermitian operator $A$ (Zheng et al., Phys. Rev. Research 5, 023200 (2023), doi:10.1103/PhysRevResearch.5.023200, the projected Hill–Wheeler equation, Eq. (13) of arXiv:2212.09205v1). `result.eigenvalue` is the smallest projected value after overlap directions below a cutoff are removed, and it does not establish the full-space ground energy. `FixedGCIM` uses basis states you supply. `ADAPT` grows the basis from a reference state and a pool of generators, adding in each round the unselected generator $A_j$ with the largest absolute gradient $\langle[H,A_j]\rangle$ on the current product state (Zheng et al., npj Quantum Information 10, 127 (2024), arXiv:2312.07691v3, main text p. 4 and Methods p. 9).

Use `FixedGCIM` when you have trial states and want the lowest energy in their span, and `ADAPT` when you have a reference state and a generator pool, such as the spin-adapted fermionic pool for [chemistry](#chemistry-inputs-and-reference-energies), and want the basis chosen for you. [Chebyshev Lanczos](lanczos.md) instead builds its trial space from one initial state and powers of the operator, and [QPE](qpe.md) estimates an eigenphase or energy from controlled time evolution of a prepared state.

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.gcim import FixedGCIM

problem = Eigenproblem(A=[[1, 0], [0, -1]])
result = solve(problem, method=FixedGCIM(basis=([1, 1], [1, 1j])), seed=7)
print(result.eigenvalue)
```

```text
-0.9999999999999983
```

The two basis states span the whole two-dimensional space, so the projected value is the lowest eigenvalue $-1$ of $A=\operatorname{diag}(1,-1)$, up to roundoff. Replace `A` and the basis columns to treat your own operator. The eigenvalue notebook (`examples/gcim_lanczos_qpe_eigenvalue_intro.ipynb`) applies ADAPT-GCIM to a stretched H4 chain and compares Lanczos and QCELS on the same problem, and the [API](../api/algorithms/gcim.md) lists every option.

## Inputs and matrix access {#matrix-access-and-repeated-basis-inputs}

`Eigenproblem` holds the Hermitian operator. `FixedGCIM` holds the basis columns and uses each one normalized, and `ADAPT` holds the reference state, the generator pool and the stopping controls. Every supplied column stays defined in original coordinates. Classical FixedGCIM keeps dense, CSR and CSC inputs and all basis columns in their original dimension, and quantum dense and sparse conversions pad the dimension with the mean trace.

Quantum FixedGCIM and ADAPT use the same `input_conversion="auto"|"dense_pauli"` and `max_conversion_work=100_000_000` options as [Lanczos](lanczos.md#choose-input-access-and-understand-its-cost). Automatic conversion accepts only explicit dense matrices of original dimension at most 16, and compact Pauli inputs do not receive that limit. Choose `dense_pauli` explicitly for a sparse quantum matrix or a sparse custom ADAPT generator. For example, `ADAPT(initial_state=[1, 1], pool=(csr_matrix([[0., 1.], [-1., 0.]]),), input_conversion="dense_pauli")` accepts that finite custom generator under the same transform-work and byte limits.

Classical ADAPT keeps the sparse Hamiltonian action and does not convert a sparse Hamiltonian to dense. Its qubit generator domain uses mean-diagonal sparse block padding for non-power-of-two targets and zero-padded original reference states. A quantum ADAPT [`Plan`](../glossary.md#plan), the construction and its costs computed before any circuit exists, counts the exact synthesis of the dense unitaries of a supplied reference circuit, and Qiskit's control of them in the Hadamard-test and joint-state queries, against `max_products` and `max_bytes` at planning ([exact synthesis](../development/dense_synthesis.md#admission-of-the-exact-synthesis)).

Within one classical evaluation, repeated uses of the same immutable state input share its computed vector and `(A-cI)@vector`. Distinct state inputs are not merged by shape, declared name or circuit equivalence, and saving and loading keep this sharing. Known preparation products enter the projected-work limit check before any state is computed. The simulation work of an opaque custom circuit remains unknown.

## Read the result {#inspect-numerical-diagnostics-and-error-bounds}

`result.eigenvalue` is the smallest obtained projected value and does not establish the full-space ground energy. `result.analyze(overlap_cutoff=...)` uses the same obtained H and S with a new absolute cutoff, whose default is `1e-12`. `result.pencil` exposes the raw complex Gram matrix, the projected Hamiltonian, the kept spectrum and the numerical diagnostics below. These fields describe the projected problem and can be examined after loading a saved Result, without obtaining its matrix elements again.

| Field | Meaning |
| --- | --- |
| `overlap_eigenvalues`, `overlap_cutoff`, `kept_rank` | Raw Gram spectrum, absolute cutoff and number of positive directions used in the solve. |
| `overlap_condition_number` | Largest divided by smallest Gram eigenvalue among the directions above the cutoff. |
| `projected_residual` | Computed $\lVert Hc-ESc\rVert_2$ for the lowest returned projected eigenpair, evaluated as $\lVert H_0c-(E-c_I)Sc\rVert_2$ without the identity coefficient $c_I$, which gives the same value in exact arithmetic. |
| `projected_backward_error` | Computed residual divided by $(\lVert H\rVert_F+\lvert E\rvert\,\lVert S\rVert_F)\lVert c\rVert_2$, with the physical $H=H_0+c_IS$ and $E$. This is dimensionless and unchanged by a rescaled energy unit, but adding an identity term to the operator changes it. Lanczos states its value on a normalized pencil instead ([Projected quantities](../verification.md#projected-quantities)). The reported value is zero when the computed residual is zero. |
| `overlap_normalization_error` | Computed $\lvert c^\dagger Sc-1\rvert$ for the lowest returned coefficient vector. |

Here $H$ and $S$ are the projected matrices and $c$ is the corresponding coefficient vector, distinct from the identity coefficient, which the table writes $c_I$. $\lVert\cdot\rVert_2$ is the Euclidean vector norm and $\lVert\cdot\rVert_F$ the Frobenius matrix norm. The diagnostics are floating-point evaluations. Their sizes help explain the solve and its conditioning, but they do not by themselves bound trial-space approximation or full-space ground-energy error. A zero Hermiticity defect in a matrix assembled by conjugate completion describes that representation, not an independent validation of its obtained entries. [Projected verification](../verification.md#projected-quantities) exposes selected stored diagnostics as numerical facts without obtaining the entries again or running another eigensolve.

For sampled quantum FixedGCIM and ADAPT, the current pencil also records the processed Pauli L1 spectral enclosure, the Ritz estimate's distance outside it, and the relative numerical comparison window. A noisy pencil need not obey the exact operator's Rayleigh–Ritz enclosure even when its own eigensolve has small backward error. This is a physical plausibility diagnostic, not a joint sampling bound, a proven accuracy bound or a ground-state test. The raw estimate is kept, and no clipping, resampling or additional full-space eigensolve is performed. An enclosure that is not representable stays unavailable. The summary describes the current pencil's enclosure diagnostic, including after cutoff reanalysis, and does not attach an earlier round's warning to a later estimate. Solver failure and ADAPT's stop reason remain separate, and an enclosure excursion does not change generator selection, stopping or previous-energy updates.

A numerical error bound must identify both the quantity it bounds and its assumptions. For example, if $S$ is exactly the identity and two Hermitian projected Hamiltonians have a justified difference bound $\lVert\hat H-H_\star\rVert_F\le\delta$, their minimum eigenvalues differ by at most $\delta$ by the [Hermitian eigenvalue perturbation bound](https://nhigham.com/2021/03/09/eigenvalue-inequalities-for-hermitian-matrices/). This bounds the matrix-perturbation contribution to the projected energy. It does not include error in the subsequent eigensolve or error from the chosen trial space. With a perturbed or ill-conditioned $S$, a bound on $H$ alone is insufficient.

FixedGCIM's default total-error model declares six unresolved sources, and its analysis does not automatically produce the corresponding physical-output error bounds. The default total `result.assess(absolute_tolerance=...)` therefore remains `INCONCLUSIVE` even for a full trial basis. Additional compatible `facts=` can supply established bounds, but a reference energy alone does not fill the missing contributions. The absolute-error criterion does not require a known ground energy. [Limitations and open work](../ROADMAP.md#gcim-physical-output-error-bounds-and-automatic-assessment) tracks the work on these bounds.

## How matrix elements are obtained

Exact quantum FixedGCIM and ADAPT obtain $H_{ij}$ and $S_{ij}$ once per basis pair, and by default the quantum path reduces each pair's exact saved state when it is obtained. Each exact pair returns the offset-free Hamiltonian entry and the overlap entry. Sampled FixedGCIM measures them with grouped joint-state counts, and sampled ADAPT with complex Hadamard transition amplitudes. Every path then uses the shared generalized Hermitian eigensolver.

- **Off-diagonal pair, exact.** One joint preparation holds $\phi_i$ and $\phi_j$ on the two branches of an ancilla, with their relative phase. The pair computation applies the offset-free Hamiltonian $H_0$ once to the right branch with the grouped Pauli kernel and contracts with the left branch, and it computes the overlap directly from the two branches. The Hamiltonian entry's error combines preparation error, grouped-action rounding and complex-dot rounding. The overlap entry has only preparation and dot error.
- **Diagonal pair, exact.** One system preparation returns its raw squared norm as the obtained overlap. Pencil assembly uses unit diagonal overlap under the normalized-basis convention, while the raw obtained scalars and the padded coordinates stay linked to the readout that produced them.
- **Sampled FixedGCIM.** `shots=N` gives N shots per qubit-wise commuting system group, with separate ancilla X and Y settings for an off-diagonal pair and one system setting per group on a diagonal pair. A group that supplies the overlap also records its overlap–Hamiltonian cross-moment, so the variance of the physical Hamiltonian includes the covariance of their shared shots.
- **Sampled ADAPT.** Pair queries use N shots per elementary Hadamard setting.
- **Classical.** `execution="classical"` obtains $V^\dagger(A-cI)V$ and $V^\dagger V$ directly from the input, with $c=\operatorname{tr}(A)/d$, or the identity coefficient of a Pauli input. It removes $c$ from the Pauli table or from the stored diagonal before any product, and it performs no reference eigensolve or unnecessary Pauli conversion.

Exact quantum FixedGCIM and ADAPT therefore need a target that runs these pair computations, such as Aer or NWQ-Sim's CPU/SV target. On other targets, use `shots=N` or `execution="classical"`. All paths share the projected analysis. Missing real or imaginary data leaves the pencil incomplete.

For $A=cI+A_0$, the same overlap observations supply $H=cS+H_0$, with no separately measured identity term and no coordinate-identity substitution. The solve uses $H_0$ and adds $c$ to the Ritz values afterwards, and `result.pencil` records $H$ ([Proposition 14](../mathematics.md#r14)). Inside the solve, $cS$ would enter the product $X^\dagger HX$ of the canonical orthogonalization, whose roundoff grows like $u\lvert c\rvert/s_{\min}$ for the smallest kept overlap eigenvalue $s_{\min}$, so a large offset on a nearly dependent basis would shift the Ritz values by far more than roundoff. Like classical FixedGCIM, a classical ADAPT pair query removes the identity component $c=\operatorname{tr}(H)/d$ before it projects, from the Pauli table or from the stored diagonal of a dense or sparse matrix, and the solve adds $c$ back, so an identity offset of $H$ does not enter the rounding of the projected entries.

### Number of circuits and shots {#number-of-circuits-and-shots}

A complete pencil on $b$ basis states has $b(b+1)/2$ pairs $i\le j$, of which $b$ are diagonal. Exact quantum execution uses one preparation per pair. Sampled FixedGCIM uses $b^2G$ settings for $G$ nonempty qubit-wise commuting groups, each with `shots` shots (`fixed_basis.py::sampled_settings`). `pencil.matrix_element_count` counts the scalar matrix elements before any pair or setting expansion. For $M$ pairs, $D$ of them diagonal, and $L$ nonidentity Pauli terms, it is $2(M-D)(1+L)+DL$, because an off-diagonal pair needs the real and imaginary parts of the overlap and of every term, while a diagonal pair needs only the real part of each term (`pencil.py::matrix_element_count`).

### When the overlap matrix is refused {#when-the-overlap-matrix-is-refused}

A legal rank-deficient trial space is distinguished from an invalid deterministic Gram matrix. The solve refuses a deterministic Gram matrix only when an overlap eigenvalue lies below minus the sum of the derived error of its entries and the eigensolver roundoff. A sampled overlap uses the positive-subspace policy instead. The entry error is as follows, with unit roundoff $u=2^{-53}$ and $\gamma_n=nu/(1-nu)$ (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., doi:10.1137/1.9780898718027, Lemma 3.1).

- **Classical entries.** The error comes from the inner products of length $d$, $\sqrt2\,\gamma_{2d}\operatorname{tr}(S)$ to first order.
- **Exact quantum FixedGCIM and ADAPT.** The error is the largest row sum of the off-diagonal overlap entry bounds, each from the saved-state error (`saved_state_error`) of its own pair's preparation record ([Engineering constants](../ENGINEERING_CONSTANTS.md)). Only off-diagonal overlap bounds enter this Gram error budget. An unresolved preparation record leaves the bound unavailable, and the solve then allows no input error beyond its own roundoff.

Exact FixedGCIM pairs use X gates in the circuit for the ancilla flips. Their preparation records supply a first-order state-error budget when the executed circuit operations and parameters satisfy the backend's roundoff model, and the pair computation accounts for the host contractions. Supplied matrix gates are synthesized when controlled for off-diagonal pairs, whose finite bounds assume the library's exact-to-rounding synthesis convention. An uncontrolled matrix in a diagonal pair can leave that pair's bound unavailable, and unresolved exclusions of gate parameters, such as an excessive U-gate phase sum, can also leave off-diagonal bounds unavailable.

## Adaptive generator coordinates

```python
from nwqlib import Eigenproblem, solve
from nwqlib.algorithms.gcim import ADAPT
from nwqlib.operators import ingest_pauli

problem = Eigenproblem(A=[[1, 0], [0, -1]])
pool = (ingest_pauli((("Y", 1j),), num_qubits=1),)
method = ADAPT(initial_state=[1, 1], pool=pool)
result = solve(problem, method=method, seed=7)
print(result.selected, result.stop_reason, result.eigenvalue)
```

```text
(0,) pool_exhausted -0.999999999999999
```

ADAPT screens the current product state with $\langle[H,A_j]\rangle$, chooses the largest absolute gradient, then expands the generator-coordinate basis. Here $H=Z$ and $A=iY$ give $[Z,iY]=2X$, so the initial $|+\rangle$ gradient is 2, and the expanded basis reaches the eigenvalue $-1$ before the one-generator pool is exhausted. A projected Ritz solution does not replace this product-state screening rule. The basis order is the reference, the individual selected generators, then the current full product and earlier prefixes without duplicate first products.

Pool order decides only between absolute gradients that are equal as binary64 numbers, and the lowest index wins. Gradients that are equal in exact arithmetic, for example because a symmetry of H, the reference and the pool maps one generator to another, usually differ in binary64 by roundoff, so the summation order, such as the order of the Hamiltonian's Pauli terms, can decide which of them wins. When a symmetry relates the tied generators, each choice gives the same energies in exact arithmetic. For example, take six qubits on a ring, $H=\sum_j(h_1Z_j+h_2X_jX_{j+1}+h_3X_j)$ with $h_1=\sqrt2/3$, $h_2=\pi/7$ and $h_3=e/5$ on every site, the product reference $\cos(0.3)|0\rangle+\sin(0.3)|1\rangle$ on every qubit and the pool $iY_j$. The six gradients agree to 50 digits. With the Pauli terms passed through `ingest_pauli` in the order of this sum for $j=0,\ldots,5$, and qubit 0 as the rightmost character of each label, classical screening gives five different binary64 values and selects $iY_5$. Reversing the order of the terms gives four values and selects $iY_3$ ([Measured evidence](#measured-evidence)).

### Defaults and stopping {#defaults-and-stopping}

Defaults are `theta=pi/4`, gradient floor `1e-8`, energy-change tolerance `1e-6`, overlap cutoff `1e-12`, flat-counter stopping and eight iterations. These are operational settings, not convergence or ground-coverage guarantees. With a fixed angle, the selected generators and the energy after a given iteration depend on the sign of each generator. Replacing a generator $A$ by $-A$ changes the product states $\exp(\theta A)$, so the spanned subspace and the later screening differ.

The default flat-counter stopping rule follows [Zheng et al., arXiv:2312.07691v3, Methods](https://arxiv.org/html/2312.07691v3). It compares consecutive projected-energy changes with `energy_change_tolerance` and stops after `min(t_auto_fraction * unselected_count, t_user)` flat rounds. The defaults are `t_auto_fraction=.2` and `t_user=10`. An integer counter reaches a fractional threshold at its ceiling. Counting starts with the reference-to-first-basis energy change and resets after a large or unavailable change. This heuristic does not prove a bound on the physical energy error. [Replacing it with better-supported stopping evidence](../ROADMAP.md#adapt-gcim-stopping-beyond-the-flat-counter) is planned research work.

With both default limits, `max_iterations=8` and `t_user=10`, the flat counter can fire only if the original pool has at most 42 generators. After k selections a pool of P generators has P-k unselected members and a flat count of at most k, so the stop needs k >= .2(P-k). The iteration cap is checked first and ends the run at step 8, which makes step 7 the last step for this stop and gives P <= 42. Larger pools reach the iteration cap before this particular stop can fire. This does not affect the other stopping conditions.

The paper's accompanying [QuGCM implementation, `adapt_gcim.py`, lines 303–398](https://github.com/pnnl/QuGCM/blob/c5efcb0171eaed8a51b75b2dff78440a820f903c/ADAPT-GCIM/adapt_gcim.py#L303-L398), selects each pool index at most once, and NWQLib follows this ADAPT-GCIM selection convention. Reusing a fixed-angle generator would repeat its single-generator reference basis vector, although its cumulative product vector could be new, so allowing reuse would change the trial-space construction and the pool-exhaustion rule. Grimsley's ADAPT-VQE step 7 (arXiv:1812.11173v2) addresses a different ansatz with optimized parameters. The same QuGCM source uses `max(10, floor(.1*unselected))` and an initial warmup for stopping, which differs from Zheng's published Methods rule (arXiv:2312.07691v3). NWQLib's flat counter follows the published rule above.

### Generator pools {#generator-pools}

The named spin-adapted, UCCSD-SD, QEB-SD and OVP-CEO pools keep their established ordering and preparation restrictions. Processing cutoffs and any accepted pool symmetry correction are recorded with their removed mass and the original input's identifier.

The default spin-adapted pool uses the signed operators of [Zheng et al., arXiv:2312.07691v3, Appendix E.1, Eqs. (E1)–(E3)](https://arxiv.org/html/2312.07691v3). It enumerates spatial pairs with `p <= q`, `r <= s` and the first pair no later than the second in lexicographic order. The double builders write interleaved ladder operators, so their signs follow from anticommutation, adjoint subtraction and positive coefficient normalization together. An independent scalar check compares every operator in Table V (Table 5 in the npj version) with the pool, including signs and repeated indices. Eq. (E4)'s `q < p` convention describes elementary circuit decomposition and does not prescribe the spin-adapted pool's enumeration.

### Optional product-state optimization {#optional-product-state-optimization}

Optional optimization requires a period, an iteration count and a finite limit on total energy evaluations, which counts distinct evaluated parameter points. Quantum optimization uses the bounded generalized or Pauli-insertion shift rules ([Proposition 15](../mathematics.md#r15)). Classical product optimization evaluates energy and gradient together with a forward and backward sweep, sharing one Hamiltonian action per parameter point. The derivative follows the selected generator's action and normalization. BFGS iteration limits, query accounting and the distinction between the surrogate product energy and the projected GCiM energy remain explicit. Completed query points are reused without new seeds. A restarted SciPy BFGS begins from its completed best point and records that its search trajectory may change.

### Continue and reanalyze {#continue-and-reanalyze}

`result.history` keeps the settings, choices and observations that drove each decision, and post-hoc cutoff analysis cannot rewrite that history. `run.resume()` retrieves known jobs before new work, and `run.resume(reanalyze=True)` explicitly recovers an interrupted projected analysis from complete saved data. Unknown jobs are not replaced.

Before a quantum matrix stage runs its first query, the Run checks the stage's new pencil queries against its circuit and shot limits, counted by `pencil.matrix_element_count` from the basis size and the pairs of basis states that earlier stages obtained. A stage that those limits cannot finish is refused before any of its work, and the message gives the counted, requested and needed totals. After `run.extend_limits(...)`, `run.resume()` runs the stage with the same seeds as a Run that was never refused. The stage in progress when a Run resumes may be partly run, so its queries are checked one at a time. `result.save(...)` and current-format loading keep the selected pool, the compiler data of the selected generators and sufficient observations without replanning.

## Verify an ADAPT result {#verify-an-adapt-result}

Explicit `AdaptVerificationOptions` selects residual, sector or supplied-reference-energy checks through `result.verify(checks=...)`. `ProjectedVerificationOptions` performs only checks on stored projected data. `EnergyShiftOptions.for_result` also supports FixedGCIM's original dense, CSR and CSC classical inputs. Its explicit matrix-entry evidence uses the same [energy-shift check](lanczos.md#check-a-result) as Lanczos, keeps complex off-diagonals and implicit zeros, and performs no Pauli conversion or solve. A zero residual can belong to an excited eigenstate ([Result 17](../mathematics.md#r17)). Sector expectations are not membership proofs, and signed spin contamination stays signed. Sampled results cannot inherit joint-coverage or ground claims that were not supplied.

Circuit-level verification constructs the Qiskit gate suffixes of the selected generators and records that construction work. Classical verification reuses immutable prefix vectors from the Result's saved data, including after loading an archive. Both accumulate the Ritz state in basis order and release each basis or prefix state after its last contribution and successor evolution. Their check against `max_bytes` covers these states, for circuit-level verification the largest gate matrix applied at once, and the arrays the saved Result already holds. Circuit-level residual and sector checks rebuild each basis state as a full statevector of $2^n$ amplitudes on $n$ qubits.

Before any state exists, verification builds and decomposes each circuit it will simulate exactly once. Each dense unitary in a supplied reference circuit is first replaced by its exact synthesis, whose work (`_dense_synthesis.dense_synthesis_size`) is counted against `max_products` and `max_bytes` before it starts and adds to the total. The limit check then counts each decomposed k-qubit gate as $1+4^k+2^n(2^k+2)$ operations, for its instruction copy, its matrix, its application and two state copies, and adds $2^n$ for the reference's zero state and for each nonzero global phase. The decomposition itself and Qiskit's workspace are outside this count. With an occupation reference, the count gives these sizes:

| Check | Qubits | Operations |
| --- | --- | --- |
| Circuit-level check on the spin-adapted pool, any selection of up to eight generators | 8 | at most 91.1 million, plus the shared Pauli action of the Hamiltonian for a residual |
| One shared-index double | 12 | up to 104 million |
| One four-distinct double | 12 | up to 56.7 million |
| One pair-split double | 12 | up to 10.3 million |
| A direct-vector reference alone | 12 | about 335 million |

These counts depend on Qiskit's decomposition of the circuits. The Qiskit version and the circuits of each row, including the generator selection that gives the first row's maximum, were not recorded. The counts indicate the size of a check, and another Qiskit version can decompose the circuits into other gates and give other counts.

The default `AdaptVerificationOptions.max_products=1_000_000_000` guards against runaway planning work and is sized so that no shipped example or test workload reaches it. A check above it needs an explicitly raised `max_products`.

## Chemistry inputs and reference energies {#chemistry-inputs-and-reference-energies}

`build_gcim_chemistry_problem(...)` returns the molecular Hamiltonian, the occupation ordering and an explicit reference panel. Its `eigenproblem()` keeps Hartree units and the recorded constant offset, and `adapt_method(...)` uses its occupation preparation. MP2, CCSD and CASCI run only when requested. Molecular-orbital signs from the eigensolver are arbitrary and can differ between LAPACK builds, and each orbital's sign fixes the sign of the generators that contain it. The builder therefore makes the first AO coefficient of each orbital above `ORBITAL_PHASE_RELATIVE_THRESHOLD` of its largest magnitude positive. This removes the dependence on the eigensolver's sign choice and the generator sign flips that choice could cause. It fixes signs only. Degenerate orbitals can still be returned in another rotation of their subspace, and binary64 rounding can still decide between generators whose gradients tie in exact arithmetic ([Adaptive generator coordinates](#adaptive-generator-coordinates)).

To show the stored reference panel next to an obtained energy, build a Chemistry References `ReportSection` from the chemistry input's `metadata` and the Result's `eigenvalue`:

```python
from nwqlib import solve
from nwqlib.algorithms.gcim import (
    build_gcim_chemistry_problem,
    chemistry_reference_diagnostic,
    chemistry_reference_report_section,
)

data = build_gcim_chemistry_problem("H 0 0 0; H 0 0 0.74", basis="sto-3g",
                                    reference_methods=("mp2", "ccsd"))
result = solve(data.eigenproblem(), method=data.adapt_method(),
               execution="classical", seed=7)
print(result.eigenvalue)
diagnostic = chemistry_reference_diagnostic(data.metadata,
                                            energy=result.eigenvalue)
section = chemistry_reference_report_section(diagnostic)
print("\n".join(section.lines))
```

```text
-1.1372838344885012
label: application context, not a validation gate
E_HF: -1.1167593074
E_MP2: -1.12989738099
E_CCSD: -1.13728399861
full-space CCSD correlation fraction: 0.999992003683
```

The first line is the ADAPT energy estimate `result.eigenvalue` in Hartree. The diagnostic reads only the stored reference panel and the supplied energy E, and the section only formats that diagnostic. Neither starts an RHF, MP2, CCSD or CASCI calculation or a simulation. The section lists E_HF, E_MP2 and E_CCSD, plus E_CASCI for an active space. A missing value shows its recorded status and reason, such as `not_requested`, `failed` or `non_converged`, or `unavailable` when no status is stored. When the needed energies are present, separate lines give the full-space CCSD correlation fraction $(E-E_{HF})/(E_{CCSD}-E_{HF})$ and the active-space CASCI fraction $(E-E_{HF})/(E_{CASCI}-E_{HF})$. A zero denominator is reported as `undefined: E_CCSD == E_HF`, or `undefined: E_CASCI == E_HF` for the CASCI fraction. `section.data` holds the same diagnostic as a mapping, and `correlation_fraction(E, E_HF, E_CCSD)` evaluates one ratio directly, raising `ValueError` for a zero denominator. These references give application context. They neither prove a bound on the total error nor identify the ground state. `result.report()` does not include this section, so print or store it next to the report.

## Measured evidence

| Example and section | Settings and environment |
| --- | --- |
| Six-qubit ring tie ([Adaptive generator coordinates](#adaptive-generator-coordinates)) | Classical ADAPT with `max_iterations=1` and seed 7, the Hamiltonian passed through `ingest_pauli` as stated, read from the first round's gradients in `result.history`. Python 3.12.14, NumPy 2.5.2, SciPy 1.18.1, Qiskit 2.5.2 and Aer 0.17.2 on macOS arm64. |
| Chemistry reference panel ([Chemistry inputs and reference energies](#chemistry-inputs-and-reference-energies)) | The example as shown, with PySCF 2.14.0 and the versions of the row above. |

## Limits

- The eigenvalue is a projected value and does not establish the full-space ground energy ([Read the result](#inspect-numerical-diagnostics-and-error-bounds)).
- The default total-error assessment remains `INCONCLUSIVE`, because six error sources have no automatic bound.
- The sampled enclosure diagnostic is not a sampling or accuracy bound.
- ADAPT's defaults and flat-counter stopping are operational settings, not convergence guarantees, and fixed-angle results depend on the sign of each generator ([Defaults and stopping](#defaults-and-stopping)).
- A zero residual can belong to an excited state, and sector expectations are not membership proofs ([Verify an ADAPT result](#verify-an-adapt-result)).
- Exact quantum execution needs a target that runs the pair computations, such as Aer or NWQ-Sim's CPU/SV target.
- Chemistry reference energies give context and no error bound.

[Limitations and open work](../ROADMAP.md#gcim) lists the open items for this method.

## Numerical scope and planning limits

`FixedGCIM(max_admission_steps=1_000_000)` limits the planning work for each circuit description the method builds and does not change the quantum operations. When planning refuses, the message names the field, the refused stage and its count. Raise the field to the reported count, and further when the count is marked as a lower bound, because later planning stages can need more ([planning work limit](../development/program_checks.md#planning-work-limit)).

Classical GCiM fills normalized trial columns $V$ and shifted actions $W=(A-cI)V$ in two preallocated complex arrays, whose payload is $32bN$ bytes for $b$ columns of length $N$. Conjugating inner products form $H_0=V^\dagger W$ and $S=V^\dagger V$ without an additional state block. The projection's byte count adds the $b$-by-$b$ matrices and the preparation, operator and action workspace. The stored physical Hamiltonian is $H=H_0+cS$. Pauli input acts on all columns with one grouped Pauli action. Dense input is shifted in row blocks refilled in one buffer. CSR and CSC input is shifted once per Run into one read-only sparse payload, which holds a full copy of the stored values and shares the index pattern only when every diagonal entry is stored. Its bytes are given in [Engineering constants](../ENGINEERING_CONSTANTS.md). The ordered columns and their complete pencil $(V^\dagger(A-cI)V,\,V^\dagger V)$ remain present, including legal rank deficiency.

ADAPT planning checks the size of the growing basis before any query. After k selections the basis holds 2k states, so `2*min(pool size, max_iterations)` bounds the projected dimension, which must not exceed `max_basis_size`. For an explicit pool this bound is checked before conversion and commutator formation. A named pool is checked after it is enumerated, under that enumeration's own byte and product limits. Planning refuses, with `ApplicabilityError`, a Problem that names a fixed subspace, which the changing basis of ADAPT does not implement.

### ADAPT queries {#adapt-queries}

Classical ADAPT screening reuses the cached action of H minus its identity coefficient on the current normalized product state. The exact quantum full-chain query supplies both its Hamiltonian diagonal values and the active commutator expectations. The matrix stage obtains this query before projected analysis, and later screening reuses its observation. The diagonal overlap is one under the normalized-preparation convention. Shared values refer to one readout, whose labels, preparation record and contribution identifier survive continuation and loading. A round that stops after its matrix stage has still paid for the screening values of that query.

Quantum queries use the generator compiler data, chain parameters, controlled phase and readout. ADAPT checks an upper bound on the logical construction size for the generator family against the limits before querying. That bound uses the system width, the Pauli term count and the supported compiler family. A generator's angle-independent compiler data is constructed when that generator is selected and reused across its later angles. Logical construction positions and later circuit synthesis costs are reported separately. Classical queries use the same processed input and iteration state, with explicit state and action accounting. Only a quantum `Plan` forms the commutators $[H,A_j]$ whose Pauli labels its screening queries read. Only a quantum `Plan` with shots also groups those labels and the Hamiltonian's into qubit-wise commuting readout groups, whose comparisons are counted against `max_products`.

Reanalysis and saving validate the saved chunks of each query from the obtained data and do not run the quantum circuits again ([saved run mechanics](../development/execution.md#adapt-reanalysis-of-saved-chunks)).

## Source and code map

The projected Hill–Wheeler equation is from Zheng et al., Physical Review Research **5**, 023200 (2023), Eq. (13) in the numbering of arXiv:2212.09205v1, [DOI](https://doi.org/10.1103/PhysRevResearch.5.023200). The adaptive method is from Zheng et al., npj Quantum Information **10**, 127 (2024), arXiv:2312.07691v3. Equation, section and page numbers refer to the arXiv version named in each row. Rows marked NWQLib have no paper source, and the docstring of the code in the last column states the derivation or rule.

| Step | Source | Location | Code |
| --- | --- | --- | --- |
| Projected Hill–Wheeler problem `H f = E S f` | Zheng et al. (2023), arXiv:2212.09205v1 | Eq. (13), p. 2 | `_projected_eigensolver.py` |
| Matrix elements and their Pauli-sum form | Zheng et al. (2023), arXiv:2212.09205v1 | Eqs. (14)–(15), p. 3<br>Eqs. (33)–(34), p. 5 | `pencil.py`: `matrix_elements`, `assemble_pencil` |
| Hadamard-test amplitudes and per-shot variance | Zheng et al. (2024), arXiv:2312.07691v3 | App. G, Eqs. (G4)–(G5) and (G8), pp. 14–15 | `adapt_acquisition.py`: `construct_adapt` |
| Grouped sampled FixedGCIM: joint-state ancilla X/Y quadratures, qubit-wise-commuting system groups, weighted group moments and their pooled variances and covariance | NWQLib | Derivation in the `sampled_settings`, `_sampled_construction`, `reduce_group`, `pooled_statistics` and `read_groups` docstrings | `fixed_basis.py`: `FixedGCIM.plan`, `_analyze_groups` |
| Joint-state pair identity and weighted-pencil reduction | NWQLib | One joint preparation per pair, the zero branch of the ancilla at native bit 0 holding phi_i and the one branch phi_j. Entry bounds and pooling are in the `pair_reducer` docstrings | `pair_reducer.py`: `entry_bounds`, `pooled_bound`, `pair_work`, `pair_bytes`<br>`fixed_basis.py`: `_exact_pair_construction`, `_exact_overlap_allowance` |
| Positive-overlap truncation (canonical orthogonalization) | Epperly, Lin and Nakatsukasa, arXiv:2110.07492v2<br>Zheng et al. (2024), arXiv:2312.07691v3 | Algorithm 1.1, Sec. 1.2, p. 5, and Theorem 2.7, p. 14<br>App. F, Eqs. (F1)–(F3), p. 14 | `_projected_eigensolver.py` |
| Identity offset removed before the projected solve | NWQLib ([Proposition 14](../mathematics.md#r14)) | Exact shift covariance of the pencil | `fixed_basis.py`: `solve_pencil` |
| Sampled Ritz enclosure diagnostic | NWQLib | Pauli L1 bound and Rayleigh–Ritz interlacing | `fixed_basis.py`: `processed_pauli_enclosure` |
| Gradient screening with `<[H, A_i]>` | Grimsley et al. (2019), arXiv:1812.11173v2<br>Anastasiou et al., arXiv:2306.03227v3<br>NWQLib screens only unselected generators, while Grimsley et al. step 7 does not drain the pool. | Sec. II.B, steps 5–7, pp. 3–4<br>Eq. (2), p. 3 | `adapt_inputs.py`: `prepare_adapt_inputs`<br>`adapt_acquisition.py`: `drive_adapt`, `_screen_gradients`, `_offset_free_column` |
| Surrogate product state, fixed angle and two new basis states per selection | Zheng et al. (2024), arXiv:2312.07691v3 | Main text, p. 4<br>METHODS, p. 9 | `adapt_acquisition.py`: `basis_chains`<br>`adapt.py`: `ADAPT` |
| Flat-counter stop | Zheng et al. (2024), arXiv:2312.07691v3 | METHODS, p. 9, T = min(T_auto, T_usr), with T_auto equal to 20% of the unselected operators | `adapt.py`: `ADAPT`<br>`adapt_acquisition.py`: `drive_adapt` |
| Intermittent, truncated BFGS optimization | Zheng et al. (2024), arXiv:2312.07691v3 | App. H, p. 17<br>Table III caption, p. 8 | `optimization.py`: `optimize_product` |
| Classical product energy and its Taylor-model gradient | NWQLib | Normalized adjoint of the fixed-step Taylor factors, derivation in the `normalized_taylor_energy_gradient` docstring | `optimization.py`: `normalized_taylor_energy_gradient`, `taylor_action_derivative`<br>`adapt_acquisition.py`: `_classical_energy_gradient` |
| Generalized parameter-shift rule | Wierichs et al., arXiv:2107.12390v3 | Sec. 2.1, Eqs. (1)–(3)<br>Sec. 3.2, Eqs. (16)–(18)<br>Sec. 3.4, Eq. (24)<br>App. B.1 | `optimization.py`: `_generator_shift_rule` |
| Pauli insertion derivative | Schuld et al., arXiv:1811.11184v1, for the two-eigenvalue identity<br>NWQLib for inserting after the complete exponential ([Proposition 15](../mathematics.md#r15)) | Sec. III.A, Theorem 1, Eq. (8), and Eqs. (13)–(14), pp. 3–4<br>Sec. III.B, Eqs. (16)–(17), p. 5 | `optimization.py`: `_generator_shift_rule`<br>`adapt_acquisition.py`: `construct_adapt` |
| Spin-orbital order and default spin-adapted GSD pool | Zheng et al. (2024), arXiv:2312.07691v3<br>Magoulas and Evangelista, arXiv:2511.13485v2 | App. E, Eqs. (E1)–(E3), p. 13<br>Sec. III, Eqs. (3), (6), (7) | `fermionic_pool.py`: `enumerate_spin_adapted_gsd_pool` |
| UCCSD-SD, QEB-SD and OVP-CEO pools | Romero et al., arXiv:1701.02691v2<br>Zheng et al. (2024), arXiv:2312.07691v3<br>Yordanov et al., arXiv:2011.10540v2<br>Ramôa et al., arXiv:2407.08696v3 | Eqs. (8)–(9), p. 3, and Eq. (17), p. 4<br>App. E, Eqs. (E4)–(E5), p. 13<br>Eqs. (17)–(18), p. 6<br>Sec. II.B.3, Eqs. (23)–(24), p. 8 | `fermionic_pool.py`: `enumerate_uccsd_sd_pool`, `enumerate_qeb_sd_pool`, `enumerate_ceo_ovp_pool` |
| Orientation, `sqrt(2)` normalization, factor order and signs of the compact circuits | Magoulas and Evangelista, arXiv:2511.13485v2<br>Zheng et al. (2024), arXiv:2312.07691v3 | Sec. III, Eqs. (2), (5)–(7), p. 4, and Sec. V, text after Eq. (15), p. 6<br>App. E, Eqs. (E2), (E3) and (E8), p. 13 | `fermionic_circuits.py`: module docstring, `_append_excitation` |
| Pair/split double circuit | Magoulas and Evangelista, arXiv:2511.13485v2 | Sec. V, Eqs. (14)–(15), (21), (22), (25)–(27), pp. 6–7 | `fermionic_circuits.py`: `build_generator_circuit` |
| Four-distinct double circuits | Magoulas and Evangelista, arXiv:2511.13485v2 | Sec. VI, Table I, p. 13, and Table II, p. 14<br>Supplemental Tables SI–SII, pp. S16–S18 | `fermionic_circuits.py`: `build_generator_circuit` |
| Shared-index double circuits | NWQLib ([Proposition 16](../mathematics.md#r16)) | Exact block synthesis with fermionic swaps and Givens rotations | `fermionic_circuits.py`: `_append_shared_index` |
| Classical generator exponential `exp(theta A)` | NWQLib | Scaled degree-18 Taylor steps with `||step A|| <= 1/2`, derivation beside `GENERATOR_TAYLOR_DEGREE` | `fermionic_pool.py`: `apply_generator_exponential`, `_taylor_steps`<br>`adapt_actions.py`: `packed_exponential`, the same steps on the classical Plan's packed tables |
| Residual nearness interval and Kato–Temple heuristic | NWQLib, from standard Hermitian residual bounds ([Result 17](../mathematics.md#r17)) | Derivation in the residual-metrics docstring | `adapt_verification.py`: `_residual_metrics`, `verify` |
| Sector expectations with `S^2 = S_- S_+ + S_z (S_z + 1)` | NWQLib | Interleaved spin-orbital order of the default pool | `sector.py`: `sector_expectations`<br>`fermionic_pool.py`: `apply_spin_squared` |
| Byte and work formulas | NWQLib | Units in [Byte and work budgets](../CODE_TOUR.md#byte-and-work-budgets). Each function maps every term of its formula to the arrays or passes it counts and names any untuned limit term | `pencil.py`: `_projected_solve_bytes`, `_projected_solve_work`<br>`fixed_basis.py`: `FixedGCIM.plan`, `_plan_classical`, `sampled_analysis_requirements`, `group_requirements`, `read_groups`<br>`adapt.py`: `_construction`, `selected_kernels`, `generator_pool_slot_bound`<br>`adapt_records.py`: `_exponential_action_work`, `_energy_gradient_work`<br>`adapt_actions.py`: `action_requirements`, `build_action_tables`, `action_sizes`, `query_work`<br>`adapt_inputs.py`: `_pool_enumeration_requirements`, `array_record_bytes`<br>`adapt_acquisition.py`: `bind_classical`<br>`adapt_verification.py`: `verify`, `_state_frontier`<br>`sector.py`: `_sector_work`<br>`fermionic_pool.py`: `_reference_pool_size` |
| Molecular Hamiltonian and RHF orbital signs | NWQLib, with PySCF RHF and integrals | `ORBITAL_PHASE_RELATIVE_THRESHOLD` convention | `chemistry.py`: `build_gcim_chemistry_problem` |
