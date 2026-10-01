# QSP and QSVT API

The [framework conventions](../../FRAMEWORK.md#block-encoding-and-qsp-conventions) define the normalization and phase signs used here. Import the following objects from their defining modules. Each module docstring names the paper, arXiv version and equation behind its steps. The [QLS source map](../../algorithms/qls.md#sources-and-code-map) collects the phase-convention, phase-solving, inverse-polynomial and kernel-reflection rows.

## Phase fitting

`solve_symmetric_qsp_phases(chebyshev_coefficients, ...)` accepts finite, real, definite-parity targets. It runs L-BFGS and a damped Gauss-Newton polish from the point `(pi/4, 0, ..., 0, pi/4)` of Dong, Meng, Whaley and Lin, arXiv:2002.11649v2. If the maximum residual on the objective nodes stays at or above the tolerance, or is not finite, Newton's method runs from the same point ([Dong, Lin, Ni and Wang, arXiv:2307.12468v1](https://arxiv.org/abs/2307.12468v1), Sec. 3, Eq. (3.1) and Algorithm 3.1), and the better of the two results goes to the final verification. Near `max|f| = 1`, where the evolution and QLS targets lie, L-BFGS from that point has no convergence guarantee (Sec. 2.2 of that paper). The `phases.py` module docstring explains this order, the `_damped_newton` docstring derives the Newton step, and the engineering constants record the sweeps of the [evolution targets](../../ENGINEERING_CONSTANTS.md#qsp-phase-solver-and-evolution-synthesis) and the [QLS targets](../../ENGINEERING_CONSTANTS.md#qls-1x-fit-and-phase-pipeline). Its `evaluations` result counts actual objective/Jacobian and final verification calls across both starts, polish steps and line searches. Degree and known simultaneous numerical arrays are checked before optimization. The byte limit describes those arrays without measuring process RSS or vendor workspaces.

::: nwqlib.subroutines.qsp.phases

## Hamiltonian evolution and circuit construction

`jacobi_anger_expansion(tau, epsilon, ...)` bounds the selected degree and finite Bessel search while preserving the analytic infinite-tail term. `prepare_qsp_evolution(tau=..., epsilon=..., expansion=...)` applies one cumulative phase-evaluation allowance across both parities, both starts of each phase solve and all attempted target margins. A supplied expansion is used without another Bessel search. The pure expansion and phase-preparation functions do not import Qiskit. Native circuit assembly consumes the selected phase tuples.

`build_qsp_evolution_encoding` requires Hermitian target and encoded generators. These two premises support the QSVT polynomial calculus and the perturbation bound in [GSLW, arXiv:1806.01838v1, Lemma 61, p. 53](https://arxiv.org/pdf/1806.01838v1). Admission reads existing construction metadata. Real Pauli coefficients and conjugate-paired periodic shifts establish the premise from their stored terms. For external encodings with missing evidence, the default assumes the caller has checked both operators, issues a warning, and records `hermitian_premise="caller_assumption"`. The numerical error bound is conditional on that premise. `require_hermitian_evidence=True` instead requires both metadata entries. An explicit non-Hermitian entry is rejected in either mode.

For an already-supplied dense matrix, exact entry equality establishes Hermitian evidence. A nonzero antisymmetric assembly error is rejected even when it is tiny. If the intended input is Hermitian, explicitly choosing `(A + A.conj().T)/2` before encoding removes that component. This changes the supplied target and must reflect the caller's model. Admission never performs this projection implicitly.

This admission does not expand a circuit or apply an operator. A dense complex128 matrix needs `16 * 4**n` bytes, which is 16 TiB at 20 system qubits and 16 PiB at 25, before ancillas or temporary arrays. An optional numerical check of an external operator therefore belongs to the caller's representation and resource budget. A declaration for the target alone does not establish that an approximate encoded block is Hermitian.

Controlling a parity pass of `build_qsp_evolution_encoding`, or a branch of `build_control_diagonal_generator_encoding`, synthesizes each dense `UnitaryGate` that it holds exactly, such as the unitary of a `dense_dilation` child. On the gate-wise route, which the parity control of a pass always takes, Qiskit then controls the synthesized gates at every occurrence. Both builders take `max_work=1_000_000_000` and `max_bytes`, and they admit the work and bytes of those syntheses and of Qiskit's control of them (`_dense_synthesis.dense_synthesis_size`, `controlled_synthesis_size` and `gatewise_control_size`) before the first synthesis ([selected blocks](../../blocks.md#admission-of-the-exact-synthesis)). `build_control_diagonal_generator_encoding` takes `dense_control_route`, whose default `"auto"` synthesizes a `dense_dilation` child together with the combine qubit's control and controls the branch's diagonal rotation gate-wise ([selected blocks](../../blocks.md#dense-control-route)). The parity control of a pass is gate-wise on every route. A gate that an earlier control already synthesized or unrolled, such as a branch of a two-child joint generator, is controlled again by each pass outside this charge, and LCHS planning charges that control.

When a mathematically nonzero analytic tail is unrepresentable in binary64, the expansion's tail bounds and slack are `None`. Circuit construction remains available. Its `BlockEncoding.error_bound` and dependent OAA error claims are `None`, while known normalization, recovery scale and query counts remain available. A missing child error bound propagates only to claims that depend on that child. NaN, infinity and negative physical error inputs are invalid rather than missing certificates.

The evolution steps follow Gilyén, Su, Low and Wiebe, arXiv:1806.01838v1:

| Step | Location in arXiv:1806.01838v1 | Owner |
| --- | --- | --- |
| Jacobi-Anger expansion of `cos(tau x)` and `sin(tau x)`, parity tails | Lemma 57, Eqs. (53)-(54) | `jacobi_anger_expansion` |
| Bessel remainder beyond the analysis terminal | NWQLib power-series bound. Eq. (55) is the sharper real-argument form. | `jacobi_anger_expansion` |
| Parity combination with coefficients `(1/2, -i/2)` and 3-step OAA | Proof of Theorem 58, and Theorem 28 with `n = 3` | `build_qsp_evolution_encoding` |
| Child encoding error in physical time | Lemma 61 | `qsp_evolution_error_terms` |
| OAA residual and amplitude-deficit bound | NWQLib derivation in the docstring | `qsp_evolution_error_terms` |
| Reference query scaling | Corollary 60 | `jacobi_anger_expansion` |

::: nwqlib.subroutines.qsp.evolution

## Inverse polynomials

::: nwqlib.subroutines.qsp.inverse

## Dalzell kernel reflection

::: nwqlib.subroutines.qsp.shortcut
