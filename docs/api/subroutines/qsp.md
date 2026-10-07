# QSP and QSVT {#qsp-and-qsvt-api}

Compute symmetric quantum signal processing (QSP) phase factors for a real Chebyshev target, and build QSVT circuits that apply the polynomial to a block-encoded matrix: a real Chebyshev transform, the Hamiltonian evolution `exp(-i t A)`, the `1/x` polynomial of QLS and Dalzell's kernel reflection. The [conventions](../../conventions.md#block-encoding-and-qsp-conventions) define the normalization and phase signs used here. Import the functions and records from `nwqlib.subroutines.qsp`. Five entries import only from their submodule: `chebyshev_norming_sup_bound` from `nwqlib.subroutines.qsp.phases`, `QSPPreparedEvolution`, `prepare_qsp_evolution` and `qsp_evolution_error_terms` from `nwqlib.subroutines.qsp.evolution`, and `plan_kernel_reflection` from `nwqlib.subroutines.qsp.shortcut`.

```python
import numpy as np
from qiskit.quantum_info import Operator
from nwqlib.subroutines.block_encoding import build_block_encoding
from nwqlib.subroutines.qsp import (
    build_real_chebyshev_encoding,
    solve_symmetric_qsp_phases,
)

solution = solve_symmetric_qsp_phases([0.0, 0.0, 0.0, 0.5])  # f = T_3 / 2
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

## Phase fitting

`solve_symmetric_qsp_phases` accepts finite, real, definite-parity Chebyshev targets with `max |f| <= 1`. The module text below states the Wx convention, the solver and their sources ([Martyn, Rossi, Tan and Chuang, arXiv:2105.02859v5](https://arxiv.org/abs/2105.02859v5), [Dong, Meng, Whaley and Lin, arXiv:2002.11649v2](https://arxiv.org/abs/2002.11649v2), and [Dong, Lin, Ni and Wang, arXiv:2307.12468v1](https://arxiv.org/abs/2307.12468v1)).

::: nwqlib.subroutines.qsp.phases
    options:
      show_root_heading: false
      heading_level: 3

## Hamiltonian evolution and circuit construction

`build_qsvt_circuit` and `build_real_chebyshev_encoding` apply given Wx phases to a block encoding. `build_qsp_evolution_encoding` builds a block encoding of `exp(-i t A)` from one of a Hermitian `A`. Without Qiskit, `jacobi_anger_expansion` computes its polynomial with degree at most `max_degree` and evaluates Bessel functions only through order `max_degree + 200`. Its reported tail bound includes an analytic bound on the infinite suffix beyond the selected finite sum. If that positive analytic term underflows, the tail bound is `None`. `prepare_qsp_evolution` solves its phases within one evaluation limit. `qsp_evolution_error_terms` gives its error terms without a circuit. The [assumptions and limits](#assumptions-and-limits-of-the-evolution-builders) below apply to the evolution builders.

::: nwqlib.subroutines.qsp.evolution
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_", "!^compiled_select_resource_law$"]

## Inverse polynomials

::: nwqlib.subroutines.qsp.inverse
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_", "!^inverse_degree_law_bound$", "!^inverse_candidate_cost$"]

## Dalzell kernel reflection

::: nwqlib.subroutines.qsp.shortcut
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_", "!^kernel_reflection_cost$"]

## Assumptions and limits of the evolution builders

Hermitian assumption of `build_qsp_evolution_encoding`:

- It requires Hermitian target and encoded generators. These two assumptions support the QSVT polynomial calculus and the perturbation bound of [Gilyén, Su, Low and Wiebe (GSLW), arXiv:1806.01838v1, Lemma 61, p. 53](https://arxiv.org/pdf/1806.01838v1).
- The check reads existing construction metadata. Real Pauli coefficients and conjugate-paired periodic shifts establish the assumption from their stored terms.
- For an external encoding with missing evidence, the default assumes that the caller has checked both operators, issues a warning and records `hermitian_premise="caller_assumption"`. The numerical error bound is then conditional on that assumption. `require_hermitian_evidence=True` instead requires both metadata entries. An explicit non-Hermitian entry is rejected in either mode.
- The check does not expand a circuit or apply an operator. A dense complex128 matrix needs `16 * 4**n` bytes, which is 16 TiB at 20 system qubits and 16 PiB at 25, before ancillas or temporary arrays, so an optional numerical check of an external operator belongs to the caller's representation and resource budget. A declaration for the target alone does not establish that an approximate encoded block is Hermitian.

Synthesis limits of the controlled passes and branches:

- Controlling a parity pass of `build_qsp_evolution_encoding`, or a branch of `build_control_diagonal_generator_encoding`, synthesizes each dense `UnitaryGate` that it holds exactly, such as the unitary of a `dense_dilation` child. On the gate-wise route, which the parity control of a pass always takes, Qiskit then controls the synthesized gates at every occurrence.
- Both builders take `max_work` (default 1,000,000,000) and `max_bytes` (default 10,000,000,000), and they check the work and bytes of those syntheses and of Qiskit's control of them before the first synthesis ([checks of the exact synthesis](../../development/dense_synthesis.md#admission-of-the-exact-synthesis)).
- `build_control_diagonal_generator_encoding` takes `dense_control_route`, whose default `"auto"` synthesizes a `dense_dilation` child together with the combine qubit's control and controls the branch's diagonal rotation gate-wise ([dense control route](../../development/dense_synthesis.md#dense-control-route)). The parity control of a pass is gate-wise on every route.
- A gate that an earlier control already synthesized or unrolled, such as a branch of a two-child joint generator, is controlled again by each pass outside this count, and LCHS planning counts that control.

Missing error bounds:

- When a mathematically nonzero analytic tail is unrepresentable in binary64, the expansion's tail bounds and slack are `None`. Circuit construction remains available. Its `BlockEncoding.error_bound` and dependent OAA error claims are `None`, while the known normalization, recovery scale and query counts remain available.
- A missing child error bound propagates only to claims that depend on that child.
- NaN, infinity and negative physical error inputs are invalid, not missing bounds.

## Source map

The evolution steps follow Gilyén, Su, Low and Wiebe, arXiv:1806.01838v1. The [QLS source map](../../algorithms/qls.md#sources-and-code-map) collects the phase-convention, phase-solving, inverse-polynomial and kernel-reflection rows. The Newton step of the phase solver is derived in the docstring of `phases._damped_newton`, and the engineering constants record the solver sweeps of the [evolution targets](../../ENGINEERING_CONSTANTS.md#qsp-phase-solver-and-evolution-synthesis) and the [QLS targets](../../ENGINEERING_CONSTANTS.md#qls-1x-fit-and-phase-pipeline).

| Step | Location in GSLW | Code |
| --- | --- | --- |
| Jacobi-Anger expansion of `cos(tau x)` and `sin(tau x)`, parity tails | Lemma 57, Eqs. (53)-(54) | `jacobi_anger_expansion` |
| Bessel remainder beyond the analysis terminal | NWQLib power-series bound. Eq. (55) is the sharper real-argument form. | `jacobi_anger_expansion` |
| Parity combination with coefficients `(1/2, -i/2)` and 3-step OAA | Proof of Theorem 58, and Theorem 28 with `n = 3` | `build_qsp_evolution_encoding` |
| Child encoding error in physical time | Lemma 61 | `qsp_evolution_error_terms` |
| OAA residual and amplitude-deficit bound | NWQLib derivation in the docstring | `qsp_evolution_error_terms` |
| Reference query scaling | Corollary 60 | `jacobi_anger_expansion` |
