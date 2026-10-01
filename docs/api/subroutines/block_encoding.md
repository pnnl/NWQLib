# Block encoding {#block-encoding-api}

Build a unitary circuit `U` whose all-zero ancilla block is `A / alpha` for a square matrix, a Pauli sum or a circulant `A`, and get `alpha`, the ancilla count and an error bound with it. `U` on `a + n` qubits is an `(alpha, a, eps)` block encoding of the `n`-qubit operator `A` when `|| A - alpha (<0^a| ⊗ I) U (|0^a> ⊗ I) || <= eps` (Gilyén, Su, Low and Wiebe, arXiv:1806.01838v1, Definition 43).

Import the functions and records below from `nwqlib.subroutines.block_encoding`. Plan the two general constructions of `A = Z + 0.5 X`, then build the second:

```python
import numpy as np
from nwqlib.subroutines.block_encoding import (
    build_block_encoding_from_plan,
    plan_block_encoding,
)

A = np.array([[1.0, 0.5], [0.5, -1.0]])
for name in ("pauli_lcu", "dense_dilation"):
    plan = plan_block_encoding(A, implementation=name)
    print(plan.implementation, round(plan.alpha, 10), plan.num_ancillas)
encoding = build_block_encoding_from_plan(plan)
print(encoding.circuit.num_qubits)
```

```text
multiplexed_pauli 1.5 1
dense_dilation 1.1180339887 1
2
```

The Pauli construction's `alpha` is the coefficient 1-norm `1 + 0.5`, and the dense dilation's is the spectral norm `sqrt(5)/2`. `build_block_encoding(A)` plans and builds in one call, and its default `implementation="auto"` chooses the construction. Inside a Program, [`select_block_encoding`](../extending.md#nwqlib.blocks.encoding.select_block_encoding) chooses the encoding of a block, and [Compose blocks](../../blocks.md) explains composition, control, adjoints and circuit construction.

## Constructions

Every implementation places `A / alpha` in the all-zero ancilla block. Ancilla registers precede the system register, so they occupy the low-order bits of Qiskit's little-endian index, and `block_encoding_top_left` reads the block at stride `2**num_ancillas`.

| Implementation | Encoded operator and `alpha` | Ancillas | Reason for the construction |
| --- | --- | --- | --- |
| `multiplexed_pauli`, requested as `pauli_lcu` | `A = sum_j c_j P_j` from a Pauli decomposition, with `alpha` the coefficient 1-norm | `ceil(log2 L)` address qubits for `L` terms | Exact for the supplied terms and applicable to any square power-of-two matrix. Each system qubit's multiplexor reads only the address bits on which its Pauli factor depends. The cost grows with the number of terms. |
| `banded` | Circulant `A = sum_b beta_b S^b`, with `alpha` the band-coefficient 1-norm | `ceil(log2 B)` address qubits for `B` bands | Exact without forming a dense matrix. Its `alpha` never exceeds `s * max_b abs(beta_b)`, the normalization of the sparse-access circulant circuit of Camps et al. (arXiv:2203.10236v4) with `s` padded bands. |
| `dense_dilation` | Any square power-of-two matrix, with `alpha` equal to the spectral norm unless a normalization is supplied | One ancilla, circuit qubit 0 | Exact with the smallest `alpha` that admits zero error, at the cost of synthesizing one dense unitary on `n + 1` qubits. It serves small validation instances. |

## Build an encoding

::: nwqlib.subroutines.block_encoding.build_block_encoding
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.plan_block_encoding
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.build_block_encoding_from_plan
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.build_banded_block_encoding
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.block_encoding_top_left
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.projector_complement_matrix
    options:
      heading_level: 3

## Inputs and results

::: nwqlib.subroutines.block_encoding.BandSpecification
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.BlockEncoding
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.BlockEncodingPlan
    options:
      heading_level: 3

## Choice, error bound and limits

How `"auto"` chooses:

- An exact circulant goes to `banded`: a supplied `BandSpecification`, or a matrix or Pauli table whose band structure is detected with zero error.
- Otherwise planning compares the per-query CX costs of the Pauli and dense constructions before it evaluates the dense normalization. A strict Pauli winner leaves the dense normalization uncomputed. A dense winner and an equal-cost comparison evaluate it.
- Automatic banded detection requires exact stored structure, and automatic Pauli decomposition removes only exact zeros. Nonzero roundoff coefficients therefore stay in the term count and may change the choice. A supplied Pauli table whose exact-rational circulant residual is nonzero keeps its Pauli or dense route.

What `error_bound` covers, in the units of the supplied operator:

- For dense dilation it bounds the normalization loss with an entrywise rounding bound and a row and column norm bound. The dense complement is formed by column scaling and one matrix product, followed by one explicit unitarity check. The residual of that check is a separate numerical check and does not bound transpilation or hardware error.
- An explicit `implementation="banded"` keeps its detector tolerance and reports a bound on the difference between the supplied and the banded operator: from row and column sums for dense input, or the exact-rational coefficient residual for Pauli input. Planning and circuit construction count that bound once.
- An exact `BandSpecification` has zero algebraic error. Gate-synthesis roundoff is outside this algebraic statement.

SVDs and limits:

- `build_block_encoding(A)` computes one dense SVD and uses it for the normalization, an equal-cost comparison and the completion. `plan_block_encoding(A)` computes the normalization without keeping the singular vectors, so building that saved plan later computes its own SVD for the synthesis.
- The plan and build functions take keyword-only `max_bytes` (default 10,000,000,000) and `max_work` (default 1,000,000,000). Dense conversion, Pauli classification and table construction, and the dense completion are checked against them before allocation or decomposition. Pass the intended limits to both a standalone plan and its later build. A larger explicit value is forwarded to the nested construction.
- The work count is an operation estimate, and the counted array bytes do not bound undocumented SDK synthesis workspace or process memory. The `DEFAULT_MAX_BLOCK_WORK` row of [Engineering constants](../../ENGINEERING_CONSTANTS.md#budgets-and-mechanical-bounds) gives the counted terms.

## Source map

Code paths are relative to `nwqlib.subroutines`. Equation and section numbers refer to the listed arXiv versions.

| Scientific step | Source | Location | Code |
| --- | --- | --- | --- |
| Block-encoding definition and `alpha` bookkeeping | Gilyén et al., arXiv:1806.01838v1 | Definition 43 | `block_encoding.core.BlockEncoding` |
| Spectral norm as the smallest zero-error `alpha` | Gilyén et al., arXiv:1806.01838v1 | Remark after Definition 43 | `block_encoding.core._dense_dilation_encoding` |
| One-ancilla dilation `[[B, K], [K, -B]]` with `K = W sqrt(I - S^2) V^dagger` | NWQLib derivation, stated in its docstring | | `block_encoding.core._dense_dilation_encoding` |
| LCU block with `alpha` the coefficient 1-norm | Low and Chuang, arXiv:1610.06546v3, and Gilyén et al., arXiv:1806.01838v1 | Lemma 5 and Eq. (10), and Lemma 52 | `block_encoding.core._pauli_lcu_encoding` |
| Removing select bits on which a multiplexor does not depend | Shende, Bullock and Markov, quant-ph/0406176v5 | Sec. 3 | `_multiplexors.project_unitary_table_dependencies`; Pauli SELECT tables use the equivalent packed letter-code test `_multiplexors.local_pauli_dependencies` |
| Uniformly controlled unitary CX count | Qiskit `UCGate` synthesis, pinned by tests | | `_multiplexors.projected_unitary_resource_law` |
| Coefficient-phase diagonal | Shende, Bullock and Markov, quant-ph/0406176v5 | Theorem 7 | `_multiplexors.append_control_diagonal_phases` |
| Cyclic shifts diagonalized by the QFT | NWQLib derivation, stated in the module docstring | | `block_encoding.banded.build_banded_block_encoding` |
| Circulant normalization comparison | Camps, Lin, Van Beeumen and Yang, arXiv:2203.10236v4 | Theorem 4.1 and Sec. 4.2 | `block_encoding.banded` module docstring |
| Circulant certificate from Pauli terms | NWQLib derivation from Pauli orthogonality, stated in its docstring | | `block_encoding.core._detect_banded_pauli_structure` |
| Dense routing CX count | Shende, Bullock and Markov, quant-ph/0406176v5 | Eq. (19), p. 16, and Table 1, row QSD (l = 2, optimized), p. 14 | `block_encoding.core._dense_dilation_predicted_cx` |
| Banded UCRZ tables at `2**a` CX each, and for both Pauli and banded routes an address diagonal and a positive PREP tree at `2**a - 2` CX each | Shende, Bullock and Markov, quant-ph/0406176v5, and Mottonen et al., quant-ph/0407010v1 | Theorems 7 and 8, pp. 10-11, and Sec. II, p. 2, and Sec. III, Eq. (7), p. 3 | `block_encoding.core._pauli_plan_detail`, `block_encoding.core._banded_plan` |
| Choice of construction, limit checks and plan reuse | NWQLib, described above and in `plan_block_encoding` | | `block_encoding.core._plan_block_encoding` |

## Entries on other pages

<a id="nwqlib.blocks.encoding.select_block_encoding"></a>[`select_block_encoding`][nwqlib.blocks.encoding.select_block_encoding] is documented on [Extending NWQLib](../extending.md).
