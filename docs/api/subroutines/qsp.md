# QSP and QSVT {#qsp-and-qsvt-api}

Compute quantum signal processing (QSP) phase factors for a real Chebyshev target or a complex Laurent polynomial, and build circuits that apply the polynomial to a block-encoded matrix: a real Chebyshev transform, the Hamiltonian evolution `exp(-i t A)` by shared real QSVT or by a qubitized walk, the imaginary-time filter `exp(-beta(1 - x))`, the `1/x` polynomial of the quantum linear-system solver (QLS) and Dalzell's kernel reflection. The [conventions](../../conventions.md#block-encoding-and-qsp-conventions) define the normalization and phase signs used here. Import the functions and records from `nwqlib.subroutines.qsp`. Six entries import only from their submodule: `chebyshev_norming_sup_bound` from `nwqlib.subroutines.qsp.phases`, `QSPPreparedEvolution`, `prepare_qsp_evolution`, `qsp_evolution_error_terms` and `plan_laurent_target` from `nwqlib.subroutines.qsp.evolution`, and `plan_kernel_reflection` from `nwqlib.subroutines.qsp.shortcut`.

```python
import numpy as np
from qiskit.quantum_info import Operator
from nwqlib.subroutines.block_encoding import build_block_encoding
from nwqlib.subroutines.qsp import (
    build_real_chebyshev_encoding,
    solve_wx_qsp_phases,
)

solution = solve_wx_qsp_phases([0.0, 0.0, 0.0, 0.5])  # f = T_3 / 2
encoding = build_block_encoding(np.diag([0.6, -0.2]))  # alpha = 0.6
circuit = build_real_chebyshev_encoding(encoding, solution.phases)
stride = 2 ** (circuit.num_qubits - encoding.system_qubits)
block = Operator(circuit).data[::stride, ::stride]
print(np.round(block.real, 6))
```

```text
[[0.5      0.      ]
 [0.       0.425926]]
```

`A / alpha` is `diag(1, -1/3)`, and `T_3(x)/2` is `0.5` at `1` and `23/54 = 0.425926...` at `-1/3`. The ancilla qubits come first, so the block is read at the stride of their dimension.

## Phase synthesis

Two solvers return phase factors. `solve_wx_qsp_phases` synthesizes a real definite-parity Chebyshev target in the Wx convention of [Martyn, Rossi, Tan and Chuang, arXiv:2105.02859v5](https://arxiv.org/abs/2105.02859v5), and `solve_laurent_qsp_phases` a complex Laurent polynomial on the unit circle with a supplied strict cap below one. Both use a canonical inverse nonlinear Fourier transform and check the returned angles by reconstructing the polynomial coefficients, and [Canonical phase synthesis](../../algorithms/qsp_phase_derivation.md) derives the construction and its conventions. The transform length doubles from the smallest power of two above twice the degree until the reconstructed coefficient residual meets `residual_tolerance`. The solve fails when a length that an ideal-arithmetic bound calls sufficient still misses the tolerance, when the next length would exceed an explicit `max_fft`, or when its arrays would exceed `max_bytes`, which defaults to the physical memory. The reconstructed residual is numerical evidence rather than a floating-point certificate.

::: nwqlib.subroutines.qsp.phases
    options:
      show_root_heading: false
      heading_level: 3

## Hamiltonian evolution and circuit construction

`build_qsvt_circuit` and `build_real_chebyshev_encoding` apply given Wx phases to a block encoding. `build_qsp_evolution_encoding` builds a block encoding of `exp(-i t A)` from one of a Hermitian `A`. `jacobi_anger_expansion` computes its polynomial without Qiskit, for finite real `tau>0` and `0<epsilon<1`. It first finds the smallest analysis terminal `T` whose analytic Bessel remainders beyond `T` meet `epsilon`, then returns the smallest degree whose complete tail, the omitted finite orders through `T` plus those remainders, meets `epsilon` ([Proposition 24](../../mathematics.md#r24)). The Bessel values are computed once through order `T`, and their `96*(T+2)` bytes of arrays are checked against `max_bytes` before allocation. A positive bound below the binary64 range is reported as `None` with its logarithm kept. The bounds are evaluated in ordinary binary64 arithmetic. `prepare_qsp_evolution` prepares the two real parity targets of the shared QSVT route under one common scale, and `qsp_evolution_error_terms` gives the error terms without a circuit. `residual_tolerance` limits each parity's coefficient residual, relative to the normalized target. [Phase targets of the shared routes](../../algorithms/lchs.md#shared-route-phase-target) gives the choice from a known total allowance, which differs with and without amplitude amplification. `plan_jacobi_anger_laurent` and `plan_jacobi_anger_gqet` describe the two-sided and one-sided Laurent targets of a qubitized walk before any phase is synthesized, `with_phase_allowance` attaches the residual allowance and memory limit of their phase synthesis, `prepare_qubitized_target` synthesizes the phases, and `build_qubitized_evolution` applies the result to a block encoding. The [assumptions and limits](#assumptions-and-limits-of-the-evolution-builders) below apply to the builders.

::: nwqlib.subroutines.qsp.evolution
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_", "!^compiled_select_resource_law$"]

## Imaginary-time filter

`imaginary_time_expansion` and `select_imaginary_time_degree` choose the modified-Bessel polynomial of `exp(-beta(1 - x))` on `[-1, 1]` from a tail bound, and `prepare_imaginary_time_filter` fits its phases. LCHS uses this filter for `qsp_route="imaginary_gqet"` when H is a multiple of the identity, so the solution needs no k quadrature.

::: nwqlib.subroutines.qsp.imaginary_time
    options:
      show_root_heading: false
      heading_level: 3

## Inverse polynomials

::: nwqlib.subroutines.qsp.inverse
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Dalzell kernel reflection

::: nwqlib.subroutines.qsp.shortcut
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]

## Assumptions and limits of the evolution builders

Hermitian assumption of `build_qsp_evolution_encoding`:

- It requires Hermitian target and encoded generators. These two assumptions support the QSVT polynomial calculus and the perturbation bound of [Gilyén, Su, Low and Wiebe (GSLW), arXiv:1806.01838v1, Lemma 61, p. 53](https://arxiv.org/pdf/1806.01838v1).
- The check reads existing construction metadata. Real Pauli coefficients and conjugate-paired periodic shifts establish the assumption from their terms.
- For an external encoding without this metadata, the default mode assumes that the caller has checked both operators, issues a warning and records `hermitian_premise="caller_assumption"`. The numerical error bound is then conditional on that assumption. `require_hermitian_evidence=True` instead requires both metadata entries. An explicit non-Hermitian entry is rejected in either mode.
- The check does not expand a circuit or apply an operator. A dense complex128 matrix needs `16 * 4**n` bytes, which is 16 TiB at 20 system qubits and 16 PiB at 25, before ancillas or temporary arrays, so an optional numerical check of an external operator is left to the caller, in the caller's representation and within the caller's resource budget. A declaration for the target alone does not establish that an approximate encoded block is Hermitian.

Synthesis limits of the controlled passes and branches:

- Controlling a parity pass of `build_qsp_evolution_encoding`, or a branch of `build_control_diagonal_generator_encoding`, synthesizes each dense `UnitaryGate` that it holds exactly, such as the unitary of a `dense_dilation` child. On the gate-wise route, which the parity control of a pass always takes, Qiskit then controls the synthesized gates at every occurrence.
- Both builders take `max_work` (default `None`, no limit) and `max_bytes` (default `None`, the physical memory), and they check the work and bytes of those syntheses and of Qiskit's control of them before the first synthesis ([checks of the exact synthesis](../../development/dense_synthesis.md#admission-of-the-exact-synthesis)).
- `build_control_diagonal_generator_encoding` takes `dense_control_route`, whose default `"auto"` synthesizes a `dense_dilation` child together with the combine qubit's control and controls the branch's diagonal rotation gate-wise ([dense control route](../../development/dense_synthesis.md#dense-control-route)). The parity control of a pass is gate-wise on every route.
- A gate that an earlier control already synthesized or unrolled, such as a branch of a two-child joint generator, is controlled again by each pass outside this count. Neither builder counts that control, which is left to the caller that composes them.

Missing error bounds:

- When a mathematically nonzero analytic tail is unrepresentable in binary64, the expansion's tail bounds and slack are `None`. Circuit construction remains available. Its `BlockEncoding.error_bound` and dependent OAA error claims are `None`, while the known normalization, recovery scale and query counts remain available.
- A missing child error bound propagates only to claims that depend on that child.
- NaN, infinity and negative physical error inputs are invalid, not missing bounds.

## Source map

The evolution steps follow Gilyén, Su, Low and Wiebe, arXiv:1806.01838v1. The [QLS source map](../../algorithms/qls.md#sources-and-code-map) collects the phase-convention, phase-solving, inverse-polynomial and kernel-reflection rows. [Canonical phase synthesis](../../algorithms/qsp_phase_derivation.md) derives the inverse nonlinear Fourier transform of the Wx and Laurent solvers, and the engineering constants record the solver sweeps of the [evolution targets](../../ENGINEERING_CONSTANTS.md#qsp-phase-synthesis-and-evolution-targets) and the [QLS targets](../../ENGINEERING_CONSTANTS.md#qls-1x-polynomial-and-phase-pipeline).

| Step | Location in GSLW | Code |
| --- | --- | --- |
| Jacobi-Anger expansion of `cos(tau x)` and `sin(tau x)`, parity tails | Lemma 57, Eqs. (53)-(54) | `jacobi_anger_expansion` |
| Bessel remainder beyond the analysis terminal | Real-argument inequality in Eq. (55), derived from DLMF 10.9.4, and the Kapteyn and Watson bounds collected in arXiv:2608.02734v1, each summed as a geometric suffix in [Proposition 24](../../mathematics.md#r24) | `jacobi_anger_expansion` |
| Parity combination with coefficients `(1/2, -i/2)` and 3-step OAA | Proof of Theorem 58, and Theorem 28 with `n = 3` | `build_qsp_evolution_encoding` |
| Child encoding error in physical time | Lemma 61 | `qsp_evolution_error_terms` |
| OAA residual and amplitude-deficit bound | NWQLib derivation in the docstring | `qsp_evolution_error_terms` |
| Reference query scaling | Corollary 60 | `jacobi_anger_expansion` |
