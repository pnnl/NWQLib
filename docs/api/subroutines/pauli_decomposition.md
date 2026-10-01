# Pauli decomposition API

`decompose_matrix_to_pauli` converts an explicit square matrix with the shared I/X/Y/Z block transform in `operators._pauli.pauli_coefficients`. It uses vectorized componentwise averages at each level and omits only computed exact-zero coefficients. For dimension D and width q, the arithmetic is O(qD²) with O(D²) array workspace. Default `atol=0` and omitted `rtol` preserve computed nonzero coefficients. This includes tiny terms caused by rounding when the supplied matrix was assembled. Explicit pruning uses `max(atol, rtol*max_coefficient_magnitude)`, in the original operator's units, and the returned `atol` records that resolved cutoff. The transform avoids the tiny-magnitude loss of Qiskit's zero-tolerance decomposition without adding a matrix norm or reference solve. An unrepresentable coefficient magnitude raises `ValueError`.

The decomposition does not report removed mass. To obtain that bound, decompose without pruning and pass the terms to `prune_pauli_terms_relative`. Its default `PAULI_COEFFICIENT_RTOL=1e-12` is relative to the largest coefficient magnitude, and it returns the removed coefficient 1-norm. Because each Pauli string has operator norm one, this sum bounds the norm of the removed operator. Structured Pauli inputs do not use this dense conversion.

::: nwqlib.subroutines.pauli_decomposition

## Source map

| Scientific step | Source | Code owner |
| --- | --- | --- |
| `c_P = 2**-n Tr(P A)` for Hermitian, trace-orthogonal Pauli strings | Pauli-basis orthogonality, stated in the module docstring | `nwqlib.subroutines.pauli_decomposition.decompose_matrix_to_pauli` |
| Removed operator norm at most the pruned coefficient 1-norm | Triangle inequality with unit-norm Pauli strings | `nwqlib.subroutines.pauli_decomposition.prune_pauli_terms_relative` |
