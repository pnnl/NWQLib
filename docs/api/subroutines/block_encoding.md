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

The Pauli construction's `alpha` is the coefficient 1-norm `1 + 0.5`, and the dense dilation's is the spectral norm `sqrt(5)/2`. `build_block_encoding(A)` plans and builds in one call, and its default `implementation="auto"` chooses the construction. Inside a `Program` (NWQLib's description of a circuit as named steps), [`select_block_encoding`](../extending.md#nwqlib.blocks.encoding.select_block_encoding) chooses the encoding of a block, and [Compose blocks](../../blocks.md) explains composition, control, adjoints and circuit construction.

## Constructions

Every implementation places `A / alpha` in the all-zero ancilla block. Ancilla registers precede the system register, so they occupy the low-order bits of Qiskit's little-endian index, and `block_encoding_top_left` reads the block at stride `2**num_ancillas`.

| Implementation | Encoded operator, `alpha` and ancillas | Reason for the construction |
| --- | --- | --- |
| `multiplexed_pauli`, requested as `pauli_lcu` | `A = sum_j c_j P_j` from a Pauli decomposition, with `alpha` the coefficient 1-norm. Ancillas: `ceil(log2 L)` address qubits for `L` terms | Exact for the supplied terms and applicable to any square power-of-two matrix. Each system qubit's multiplexor reads only the address bits on which its Pauli factor depends. The cost grows with the number of terms. |
| `banded` | Circulant `A = sum_b beta_b S^b`, with `alpha` the band-coefficient 1-norm. Ancillas: `ceil(log2 B)` address qubits for `B` bands | Exact without forming a dense matrix. Its `alpha` never exceeds `s * max_b abs(beta_b)`, the normalization of the sparse-access circulant circuit of Camps et al. (arXiv:2203.10236v4) with `s` padded bands. When the stored coefficients satisfy `beta_{-b} = conj(beta_b)` exactly, the encoding pairs the labels of `b` and `-b` and exchanges them, which makes the whole unitary a Hermitian involution (Sünderhauf, Campbell and Camps, arXiv:2302.10949v2, Sec. 2.4). The exchange adds at most `a - 1` multi-controlled X gates for `a` address qubits, and the half shift and the signs of negative self-inverse coefficients at most three more. |
| `dense_dilation` | Any square power-of-two matrix, with `alpha` equal to the spectral norm unless a normalization is supplied. Ancillas: one ancilla, circuit qubit 0 | Exact with the smallest `alpha` that admits zero error, at the cost of synthesizing one dense unitary on `n + 1` qubits. It serves small validation instances. |

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

## Queries under external control

A block encoding used inside a larger algorithm is called as a query, possibly controlled by outside qubits or as its adjoint. `build_block_encoding_query` builds that call from a plan, and `block_encoding_query_cx` counts its CX gates without building it.

::: nwqlib.subroutines.block_encoding.build_block_encoding_query
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.block_encoding_query_cx
    options:
      heading_level: 3

## Node registers and diagonal tables

These encodings act on a register of quadrature nodes or on a table of diagonal values, for example the k nodes of LCHS, without loading a dense matrix. [Node encodings](../../algorithms/lchs_derivations.md#node-encodings) gives the address maps, the completions on padding labels, the Pauli forms and why the normalization `alpha` can exceed the physical radius.

::: nwqlib.subroutines.block_encoding.UniformNodeRecipe
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.uniform_node_recipes
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.uniform_node_labels
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.uniform_node_values
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.build_uniform_node_encoding
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.build_uniform_node_query
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.build_diagonal_table_encoding
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.build_diagonal_table_query
    options:
      heading_level: 3

## Joint generators and access facts

A joint generator combines child encodings, such as encodings of the Hermitian parts L and H of a matrix, into one encoding of a weighted sum. `access_summary` reports the normalization, error bound, widths and structural facts of any supported source without building a circuit, and `build_access_query` builds its call. [Joint generators](../../algorithms/lchs_derivations.md#joint-generators) derives the normalizations and error relations of `plan_joint_generator`, `plan_single_a_generator`, `plan_convex_contraction` and `plan_block_encoding_product`.

::: nwqlib.subroutines.block_encoding.AccessSummary
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.access_summary
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.build_access_query
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.JointEncodingPlan
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.plan_joint_generator
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.plan_single_a_generator
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.plan_convex_contraction
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.plan_block_encoding_product
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.build_joint_encoding
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.DiagonalTable
    options:
      heading_level: 3

::: nwqlib.subroutines.block_encoding.ComplexDiagonalTable
    options:
      heading_level: 3


## Choice, error bound and limits

How `"auto"` chooses:

- A supplied `BandSpecification` or a matrix or Pauli table whose circulant structure is detected within the roundoff window goes to `banded`. Any nonzero detection residual enters `error_bound`.
- Otherwise planning compares the per-query CX costs of the Pauli and dense constructions before it evaluates the dense normalization. When the Pauli construction is strictly cheaper, the dense normalization is not computed. When the dense construction is cheaper or the costs are equal, it is computed. For a dense matrix with more than `d**2/2` nonzero Pauli coefficients, the PREP pair of the Pauli construction alone, `2*(2**a - 2)` CX for `a = ceil(log2 L)`, costs more than the dense dilation, so planning selects the dense dilation from the count of nonzero coefficients without forming the Pauli terms. Its `detail` then records `term_count` and `prep_pair_cx` in place of the Pauli gate counts.
- Automatic banded detection compares dense entries within `64 eps max(1,n) max|A|`, or compares a Pauli table's exact-rational residual with `64 eps max(1,n,terms,bands) ||A||_F`. The residual is bounded separately. Automatic Pauli decomposition removes only exact zeros, so nonzero coefficients stay in the term count and may affect the choice.

A proved Hermitian input with a roundoff circulant structure:

- When exact Hermiticity of a dense matrix, or all-real coefficients of a Pauli table, proves the target Hermitian, the detected band coefficients of each pair `(b, -b)` are replaced by one rounded conjugate average before the labels are paired. A band that this makes exactly zero is removed. A supplied `BandSpecification` is used as given.
- The three structure facts stay separate. `target_operator_is_hermitian` comes from the original input, while `encoded_operator_is_hermitian` and `full_unitary_involution` come from the conjugate-pair layout. A matrix that is Hermitian only up to rounding does not have the proved target fact, so a route that needs it refuses the input and an exactly symmetrized matrix passes.

What `error_bound` covers, in the units of the supplied operator:

- For dense dilation it bounds the normalization loss with an entrywise rounding bound and a row and column norm bound. The dense complement is formed by column scaling and one matrix product. The dilation is unitary in exact arithmetic, and its deviation with computed singular vectors comes from their rounding. The error bound does not include transpilation or hardware error.
- An explicit `implementation="banded"` compares dense entries within `1e-12 max|A|` and refuses a matrix when the entries of one wrapped diagonal differ by more than that window. In exact arithmetic a nonzero finite factor on `A` does not change which bands it finds, although a complex factor can make a Hermitian `A` non-Hermitian, which a route that needs a Hermitian operator does not accept. The reported bound on the difference between the supplied and the banded operator includes any dropped band. It comes from row and column sums for dense input, or the exact-rational coefficient residual for Pauli input. Planning and circuit construction count that bound once. The conjugate average above adds at most `2**-52 * sum_b |beta_b|` for the pairs it changes, a numerical allowance in the normal binary64 range.
- An exact `BandSpecification` has zero algebraic error. Gate-synthesis roundoff is outside this algebraic statement.

SVDs and limits:

- `build_block_encoding(A)` computes one dense SVD only when it needs the dense normalization, that is, when the dense dilation is requested or, under `"auto"`, costs no more than the Pauli construction. It uses that SVD for the normalization, the equal-cost comparison and the completion. A banded plan and a strictly cheaper Pauli plan need no dense SVD. A dense-dilation plan from `plan_block_encoding(A)` keeps no singular vectors, so building it later computes its own SVD for the completion. Planning and then building without a `normalization` therefore costs one values-only SVD more than `build_block_encoding(A)`, about 4d³/3 counted units over the 9d³ of the completion. Use `build_block_encoding(A)` when the circuit will be built. For a dense dilation, a supplied `normalization` avoids the norm computation and becomes `alpha` as given.
- The plan and build functions take keyword-only `max_bytes` (default `None`, the physical memory) and `max_work` (default `None`, no limit). Dense conversion, Pauli classification and table construction, and the dense completion are checked against them before allocation or decomposition. Pass the intended limits to both a standalone plan and its later build. A larger explicit value is passed on to the nested construction.
- The work count is an operation estimate, and the counted bytes are a coarse estimate of the dominant arrays, not of Qiskit's synthesis workspace or process memory. The "Dense block-encoding arrays and work" row of [Engineering constants](../../ENGINEERING_CONSTANTS.md#budgets-and-mechanical-bounds) gives the counted terms.

## Source map

Code paths are relative to `nwqlib.subroutines`. Equation and section numbers refer to the listed arXiv versions.

| Scientific step | Source and location | Code |
| --- | --- | --- |
| Block-encoding definition and `alpha` bookkeeping | Gilyén et al., arXiv:1806.01838v1, Definition 43 | `block_encoding.core.BlockEncoding` |
| Spectral norm as the smallest zero-error `alpha` | Gilyén et al., arXiv:1806.01838v1, Remark after Definition 43 | `block_encoding.core._explicit_dense_dilation_plan`, `block_encoding.core._plan_block_encoding` |
| One-ancilla dilation `[[B, K], [K, -B]]` with `K = W sqrt(I - S^2) V^dagger` | NWQLib derivation, stated in its docstring | `block_encoding.core._dense_dilation_encoding` |
| LCU block with `alpha` the coefficient 1-norm | Low and Chuang, arXiv:1610.06546v3, Lemma 5 and Eq. (10), and Gilyén et al., arXiv:1806.01838v1, Lemma 52 | `block_encoding.core._pauli_lcu_encoding` |
| Removing select bits on which a multiplexor does not depend | Shende, Bullock and Markov, quant-ph/0406176v5, Sec. 3 | `_multiplexors.project_unitary_table_dependencies`. Pauli SELECT tables use the equivalent packed letter-code test `_multiplexors.local_pauli_dependencies` |
| Uniformly controlled unitary CX count | Qiskit `UCGate` synthesis, pinned by tests | `_multiplexors.projected_unitary_resource_law` |
| Coefficient-phase diagonal | Shende, Bullock and Markov, quant-ph/0406176v5, Theorem 7 | `_multiplexors.append_control_diagonal_phases` |
| Cyclic shifts diagonalized by the QFT | NWQLib derivation, stated in the module docstring | `block_encoding.banded.build_banded_block_encoding` |
| Circulant normalization comparison | Camps, Lin, Van Beeumen and Yang, arXiv:2203.10236v4, Theorem 4.1 and Sec. 4.2 | `block_encoding.banded` module docstring |
| Circulant certificate from Pauli terms | NWQLib derivation from Pauli orthogonality, stated in its docstring | `block_encoding.core._detect_banded_pauli_structure` |
| CX count of the dense dilation in the automatic choice | Shende, Bullock and Markov, quant-ph/0406176v5, Eq. (19), p. 16, and Table 1, row QSD (l = 2, optimized), p. 14 | `block_encoding.core._dense_dilation_predicted_cx` |
| Banded UCRZ tables at `2**a` CX each, and for both Pauli and banded routes an address diagonal and a positive PREP tree at `2**a - 2` CX each | Shende, Bullock and Markov, quant-ph/0406176v5, Theorems 7 and 8, pp. 10-11, and Mottonen et al., quant-ph/0407010v1, Sec. II, p. 2, and Sec. III, Eq. (7), p. 3 | `block_encoding.core._pauli_plan_detail`, `block_encoding.core._banded_plan` |
| Choice of construction, limit checks and plan reuse | NWQLib, described above and in `plan_block_encoding` | `block_encoding.core._plan_block_encoding` |
| Node maps and joint-generator relations | NWQLib, with Low and Somma, arXiv:2508.19238v2, Lemma 18, and Pocrnic et al., arXiv:2506.20760v2, Sec. IV. [LCHS derivations](../../algorithms/lchs_derivations.md#node-encodings) | `block_encoding.nodes`, `block_encoding.joint` |

## Entries on other pages

<a id="nwqlib.blocks.encoding.select_block_encoding"></a>[`select_block_encoding`][nwqlib.blocks.encoding.select_block_encoding] is documented on [Extending NWQLib](../extending.md).
