# LCU {#lcu-api}

Build the circuit PREP, SELECT, PREP† whose all-zero control block is `sum_j c_j U_j / alpha`, with `alpha = sum_j |c_j|`, from complex coefficients `c_j` and dense unitaries `U_j` (Childs and Wiebe, arXiv:1202.5822v1, Lemma 2, Fig. 1 and Theorem 3). Import the functions and records from `nwqlib.subroutines.lcu`, except `prepare_lcu_gate_data`, which imports from `nwqlib.subroutines.lcu.core`. `prepare_lcu_data` checks the coefficients and unitaries, and `build_lcu_prepare`, `build_lcu_select` and `build_lcu_circuit` build PREP, SELECT or their composition from that data.

```python
import numpy as np
from qiskit.quantum_info import Operator
from nwqlib.subroutines.lcu import build_lcu_circuit

I = np.eye(2)
Z = np.diag([1.0, -1.0])
lcu = build_lcu_circuit([0.75, -0.25], [I, Z])
alpha = lcu.data.coefficient_l1_norm
block = Operator(lcu.circuit).data[::2, ::2]  # control qubit is bit 0
print(alpha, lcu.preparation_l2_error)
print(np.round(alpha * block.real, 12))
```

```text
1.0 0.0
[[0.5 0. ]
 [0.  1. ]]
```

`0.75 I - 0.25 Z` is `diag(0.5, 1)`, and direct PREP has zero preparation error.

## What the circuit encodes

- PREP prepares `sum_j sqrt(|c_j| / alpha) |j>` on the control register, with `alpha = sum_j |c_j|`. SELECT applies `(c_j / |c_j|) U_j` on address `j`, so the coefficient phases sit in SELECT and PREP encodes only nonnegative magnitudes (Childs and Wiebe, Sec. II, after Eq. (4)). This is the single-register standard form of Low and Chuang, arXiv:1610.06546v3, Lemma 5 and Eq. (10), and the case `P_L = P_R` of the subnormalization of a linear combination of block encodings in Gilyén, Su, Low and Wiebe, arXiv:1806.01838v1, Lemma 52, with each phase-adjusted `U_j` a `(1, 0, 0)` block encoding of itself (Definition 44).
- The control register precedes the system register, so it occupies the low-order bits of Qiskit's little-endian index, and the block entry for system basis index `s` sits at index `s * 2**num_control_qubits`.
- The control register has `P` addresses, the smallest power of two at least the number `N` of terms. The `P - N` padding addresses have zero PREP amplitude and act as the identity, without stored matrices or controlled gates.
- A nonzero global scale of the coefficients changes `alpha` and keeps the normalized coefficient direction. The default `coefficient_atol=0` therefore rejects only zero total weight. A supplied positive threshold applies to that total weight and never removes individual coefficients.

## Build the circuit

::: nwqlib.subroutines.lcu.core.prepare_lcu_data
    options:
      heading_level: 3

::: nwqlib.subroutines.lcu.core.build_lcu_circuit
    options:
      heading_level: 3

::: nwqlib.subroutines.lcu.core.build_lcu_prepare
    options:
      heading_level: 3

::: nwqlib.subroutines.lcu.core.build_lcu_select
    options:
      heading_level: 3

::: nwqlib.subroutines.lcu.core.prepare_lcu_gate_data
    options:
      heading_level: 3

## Results

::: nwqlib.subroutines.lcu.core.LCUData
    options:
      heading_level: 3

::: nwqlib.subroutines.lcu.core.LCUCircuit
    options:
      heading_level: 3

## Accuracy and limits

- A supplied zero-weight unitary that is not the identity stays a real SELECT branch and goes through the normal unitarity check.
- Every function takes `max_bytes` (default 10,000,000,000) and `max_work` (default 1,000,000,000). Before any matrix conversion or synthesis they check the coefficient and PREP tables, the supplied matrices and, for a dense SELECT with a control register, the exact synthesis of each controlled branch, for `N` terms, `P` addresses and system dimension `D`. A branch is synthesized from its `PD`-square controlled matrix. Work counts scalar operations, not time. With `M = PD` and `m = log2(M)`, each branch counts `11 M³ + (m² + 5m + 256) M²` work units, `256 M² + 65536` bytes of working arrays, which one branch at a time uses, and `176 M² + 16384` bytes for the circuit it keeps.
- The default `max_work` is a guard against runaway planning work, and a dense SELECT above it needs an explicitly raised `max_work`. The `DEFAULT_MAX_LCU_WORK` row of [Engineering constants](../../ENGINEERING_CONSTANTS.md#budgets-and-mechanical-bounds) gives the derivation and the measured times and memory behind these terms.
- Pass the limits to each stage you call. `build_lcu_circuit` forwards the same values to its stages. The workspace of the tensor library used by layered MPS PREP is not counted.

## Source map

Code paths are relative to `nwqlib.subroutines`. Equation and section numbers refer to the listed arXiv versions.

| Scientific step | Source and location | Code |
| --- | --- | --- |
| Two-term PREP-SELECT-PREP† circuit and its extension to general combinations | Childs and Wiebe, arXiv:1202.5822v1, Lemma 2, Fig. 1 and Theorem 3 | `lcu.core.build_lcu_circuit` |
| Complex weights absorbed as phases of the unitaries | Childs and Wiebe, arXiv:1202.5822v1, Sec. II, after Eq. (4) | `lcu.core.prepare_lcu_data` |
| Single-register PREP state and `alpha` as the coefficient 1-norm | Low and Chuang, arXiv:1610.06546v3, Lemma 5 and Eq. (10) | `lcu.data._coefficient_bookkeeping` |
| Subnormalization of a combination of block encodings | Gilyén et al., arXiv:1806.01838v1, Lemma 52 | Composition rule in the [conventions](../../conventions.md#block-encoding-and-qsp-conventions) |
| Branch `j` fires on control value `j` | Qiskit little-endian `ctrl_state` convention | `lcu.core._controlled_branch_gate` |
| Exact synthesis of each controlled branch, demultiplexed at the top because the controlled matrix is block diagonal in each control qubit | Shende, Bullock and Markov, arXiv:quant-ph/0406176v5, Theorem 12, with the block-ZXZ steps of Krol and Al-Ars, arXiv:2403.13692v2. [Controlled dense unitaries](../../development/dense_synthesis.md#controlled-dense-unitaries) lists the other locators | `qiskit_compat.controlled`, `_dense_synthesis.controlled_unitary_circuit` |
| Input and construction limit checks | NWQLib, registered as `DEFAULT_MAX_LCU_WORK` in [Engineering constants](../../ENGINEERING_CONSTANTS.md). The synthesis terms follow the recursion of the exact synthesis | `lcu.core._admit_lcu`, `_dense_synthesis.controlled_synthesis_size` |
