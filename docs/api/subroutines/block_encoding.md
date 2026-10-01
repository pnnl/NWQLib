# Block encoding API

`select_block_encoding` binds the chosen encoding to its actual input and Program body. Explicit native builders consume the selected structural plan. The [selected-block guide](../../blocks.md) explains composition, control, adjoints and lowering.

::: nwqlib.blocks.encoding.select_block_encoding

::: nwqlib.subroutines.block_encoding

Automatic planning resolves structure and compares per-query CX costs before evaluating a dense candidate's normalization. A strict Pauli winner omits the unused dense normalization record. A selected dense candidate and an equal-cost comparison still evaluate the required normalization.

`build_block_encoding(A)` uses one selected dense SVD for normalization, any equal-cost comparison, and completion. Standalone `plan_block_encoding(A)` computes scalar normalization without keeping unused singular frames; building that saved plan subsequently performs its own synthesis SVD. The dense complement uses column scaling followed by one matrix product and one explicit unitarity check. The `error_bound` is in the supplied operator's units and bounds normalization loss with an entrywise rounding envelope and a row/column norm bound. The unitarity-check residual is a separate numerical check and does not establish a bound on transpilation or hardware error.

Automatic banded detection requires exact stored structure, and automatic Pauli decomposition removes only exact zeros. Nonzero roundoff coefficients therefore remain visible in the term count and may change auto selection. Supplied Pauli tables with nonzero exact-rational circulant residuals keep their Pauli or dense route. Explicit `implementation="banded"` keeps its detector tolerance and reports a bound for the supplied-to-structured difference: row/column sums for dense input, or the exact-rational coefficient residual for Pauli input. Planning and realization use that bound once. An exact `BandSpecification` has zero algebraic approximation error. Gate-synthesis roundoff is outside this algebraic statement.

The plan and build APIs accept keyword-only `max_bytes` (default 10,000,000,000) and `max_work` (default 1,000,000,000). Known dense conversion, Pauli classification/table and native completion work is admitted before allocation or decomposition. Pass the intended limits to both a standalone plan and its later build. A larger explicit allowance is forwarded to the selected nested construction. The work law is an operation estimate, and known array bytes do not bound undocumented SDK synthesis workspace or process RSS.

Scalar metadata requires finite positive `alpha`, a finite nonnegative `error_bound` when available, and exact nonnegative integer widths. Admission does not inspect or prove the circuit's operator relation. A structural `BlockEncodingPlan` with `implementation="exact_zero"` represents an omitted LCHS child with zero alpha, error and ancilla count and no source or decomposition. It cannot be realized as a block encoding and introduces no division by zero.

## Constructions

Every implementation places `A / alpha` in the all-zero ancilla block. Ancilla registers precede the system register, so they occupy the low-order bits of Qiskit's little-endian index, and `block_encoding_top_left` reads the block at stride `2**num_ancillas`.

| Implementation | Encoded operator and `alpha` | Ancillas | Reason for the construction |
| --- | --- | --- | --- |
| `multiplexed_pauli`, requested as `pauli_lcu` | `A = sum_j c_j P_j` from a Pauli decomposition, with `alpha` the coefficient 1-norm | `ceil(log2 L)` address qubits for `L` terms | Exact for the supplied terms and applicable to any square power-of-two matrix. Each system qubit's multiplexor reads only the address bits on which its Pauli factor depends. The cost grows with the number of terms. |
| `banded` | Circulant `A = sum_b beta_b S^b`, with `alpha` the band-coefficient 1-norm | `ceil(log2 B)` address qubits for `B` bands | Exact without forming a dense matrix. Its `alpha` never exceeds `s * max_b abs(beta_b)`, the normalization of the sparse-access circulant circuit of Camps et al. (arXiv:2203.10236v4) with `s` padded bands. |
| `dense_dilation` | Any square power-of-two matrix, with `alpha` equal to the spectral norm unless a normalization is supplied | One ancilla, circuit qubit 0 | Exact with the smallest `alpha` that admits zero error, at the cost of synthesizing one dense unitary on `n + 1` qubits. It serves small validation instances. |

## Source map

Owners are relative to `nwqlib.subroutines`. Equation and section numbers refer to the listed arXiv versions.

| Scientific step | Source | Location | Code owner |
| --- | --- | --- | --- |
| Block-encoding definition and `alpha` bookkeeping | Gilyén et al., arXiv:1806.01838v1 | Definition 43 | `block_encoding.core.BlockEncoding` |
| Spectral norm as the smallest zero-error `alpha` | Gilyén et al., arXiv:1806.01838v1 | Remark after Definition 43 | `block_encoding.core._dense_dilation_encoding` |
| One-ancilla dilation `[[B, K], [K, -B]]` with `K = W sqrt(I - S^2) V^dagger` | NWQLib derivation, stated in the owner's docstring | | `block_encoding.core._dense_dilation_encoding` |
| LCU block with `alpha` the coefficient 1-norm | Low and Chuang, arXiv:1610.06546v3, and Gilyén et al., arXiv:1806.01838v1 | Lemma 5 and Eq. (10), and Lemma 52 | `block_encoding.core._pauli_lcu_encoding` |
| Removing select bits on which a multiplexor does not depend | Shende, Bullock and Markov, quant-ph/0406176v5 | Sec. 3 | `_multiplexors.project_unitary_table_dependencies`; Pauli SELECT tables use the equivalent packed letter-code test `_multiplexors.local_pauli_dependencies` |
| Uniformly controlled unitary CX census | Qiskit `UCGate` synthesis, pinned by tests | | `_multiplexors.projected_unitary_resource_law` |
| Coefficient-phase diagonal | Shende, Bullock and Markov, quant-ph/0406176v5 | Theorem 7 | `_multiplexors.append_control_diagonal_phases` |
| Cyclic shifts diagonalized by the QFT | NWQLib derivation, stated in the module docstring | | `block_encoding.banded.build_banded_block_encoding` |
| Circulant normalization comparison | Camps, Lin, Van Beeumen and Yang, arXiv:2203.10236v4 | Theorem 4.1 and Sec. 4.2 | `block_encoding.banded` module docstring |
| Circulant certificate from Pauli terms | NWQLib derivation from Pauli orthogonality, stated in the owner's docstring | | `block_encoding.core._detect_banded_pauli_structure` |
| Dense routing CX count | Shende, Bullock and Markov, quant-ph/0406176v5 | Eq. (19), p. 16, and Table 1, row QSD (l = 2, optimized), p. 14 | `block_encoding.core._dense_dilation_predicted_cx` |
| Banded UCRZ tables at `2**a` CX each, and for both Pauli and banded routes an address diagonal and a positive PREP tree at `2**a - 2` CX each | Shende, Bullock and Markov, quant-ph/0406176v5, and Mottonen et al., quant-ph/0407010v1 | Theorems 7 and 8, pp. 10-11, and Sec. II, p. 2, and Sec. III, Eq. (7), p. 3 | `block_encoding.core._pauli_plan_detail`, `block_encoding.core._banded_plan` |
| Routing, admission and plan reuse | NWQLib contract, described above and in `plan_block_encoding` | | `block_encoding.core._plan_block_encoding` |
