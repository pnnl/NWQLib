# State preparation API

`select_preparation` selects a reusable Program body from an admitted StateInput. It does not build a native circuit. The direct and MPS constructors below perform explicit native construction when called.

::: nwqlib.blocks.selection.select_preparation

::: nwqlib.subroutines.state_preparation.direct

::: nwqlib.subroutines.state_preparation.mps

::: nwqlib.subroutines.state_preparation.mps_circuit

The direct constructor accepts dense complex vectors and preserves exact basis, full-uniform, and prefix-uniform fast paths. General vectors use a binary tree of conditional RY rotations followed by a phase diagonal. Pairwise `hypot` norms and `atan2` angles avoid squared-magnitude underflow, and native Walsh/Gray synthesis skips exact zero angles. No synthesis cutoff discards a requested small rotation.

For an n-qubit generic vector, each magnitude tree or phase diagonal uses at most `max(0, 2**n - 2)` CX gates. General complex preparation uses two such trees, while positive LCU coefficient preparation needs the magnitude tree. Mottonen et al. (arXiv:quant-ph/0407010v1) interleave the phase multiplexors with the magnitude levels and cancel one CX per multiplexor, two per level, reaching `2**(n+1) - 2n - 2` CX from a basis state. The separate diagonal used here costs `2n - 2` more CX for a complex state and lets a nonnegative vector omit the phase stage entirely. Construction takes `O(n * 2**n)` classical arithmetic and `O(2**n)` live numerical storage. It does not construct a full unitary or simulate the prepared state.

`preparation_l2_error = 0` describes the ideal-gate construction with no algorithmic approximation. It is not a measured bound on floating-point synthesis, backend execution, or hardware error. The direct preparation record reports `fidelity_to_target = None` because ordinary construction does not evaluate fidelity. Binary64 roundoff can limit relative accuracy for components near or below machine precision, especially when a tiny component results from cancellation among rotations.

MPS circuit construction runs one TT-SVD sweep or consumes an explicitly selected matching decomposition. The returned object stores `decomposition` and the normalized `target_state` used by its explicit circuit validator. It does not store a compression-analysis object or additional reconstructed vectors. Core metadata includes ranks, original normalized-input identity and discarded singular-value weight. The full input and economy SVD still have their dimension-dependent storage and computation costs.

`decompose_state_to_mps(vector, ...)` returns cores without reconstruction. `analyze_mps_state_compression(decomposition)` reads their scalar truncation data without another SVD or contraction. Its raw L2, norm-squared and normalized-fidelity estimates follow the orthogonal sweep relation in Oseledets (2011), doi:10.1137/090752286, Theorem 2.2 proof. They are not measured circuit fidelity or floating-point certificates. Passing `reference=vector` explicitly requests one reconstruction and numerical comparison, naming whether the normalized reference matches the original input. `decomposition.to_statevector(max_bytes=...)` explicitly exports the raw compressed tensor. Scalar display and serialization do not expand arrays.

TT-SVD admits the whole prospective economy-SVD work before its first decomposition, including factors computed before truncation. `max_svd_work` bounds the declared dense-work proxy, and `max_bytes` covers known arrays without measuring process RSS. The circuit builder also admits its layered construction against both limits before scikit_tt starts, with the law `mps.layered_construction_size` of the sweeps that scikit_tt repeats after every extracted gate. Contraction has its separate scalar-product and byte limits. `validate_mps_circuit_state_preparation` evaluates the completed circuit once against the stored target only when requested. MPS one-qubit synthesis includes its global phase, so controlled preparations preserve relative phase as well as state populations. Each two-qubit gate of a layer is synthesized exactly to rounding with at most three CX (`_dense_synthesis.dense_unitary_circuit`), so the synthesis adds only rounding to the error of the layered construction.

## Source map

Owners are relative to `nwqlib.subroutines`. Equation and section numbers refer to the listed arXiv or journal versions.

| Scientific step | Source | Location | Code owner |
| --- | --- | --- | --- |
| Magnitude-tree RY angles | Mottonen et al., quant-ph/0407010v1 | Sec. III, Eq. (8) | `state_preparation.direct._build_normalized_state_preparation` |
| Uniformly controlled RY and RZ lowering by Walsh transform and Gray code | Mottonen et al., quant-ph/0407010v1, and Shende, Bullock and Markov, quant-ph/0406176v5 | Sec. II, Fig. 2 and Eq. (3), and Theorem 8 | `_multiplexors._rotation_multiplexor` |
| Phase diagonal as one RZ multiplexor per qubit | Shende, Bullock and Markov, quant-ph/0406176v5 | Theorem 7 | `_multiplexors.append_control_diagonal_phases` |
| CX counts `2**k` per multiplexor and `2**n - 2` per tree or diagonal | Shende, Bullock and Markov, quant-ph/0406176v5, summed over the tree by NWQLib | Theorem 8 and Fig. 2 | `_multiplexors.multiplexor_resource_law` |
| TT-SVD sweep and truncation estimates | Oseledets, SIAM J. Sci. Comput. 33(5) (2011), doi:10.1137/090752286 | Algorithm 1 (p. 2301) for the sweep. Proof of Theorem 2.2 and Eq. (2.5) (p. 2299) for the error estimate, with NWQLib's inner-product step for the norm and fidelity estimates | `state_preparation.mps.decompose_state_to_mps`, `state_preparation.mps.analyze_mps_state_compression` |
| Layered disentangling circuit | Ran, arXiv:1908.07958v2 | Sec. III steps 1-4, Eqs. (6)-(9) | `state_preparation.mps_circuit.mps_to_circuit` |
