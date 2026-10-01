# Pauli decomposition {#pauli-decomposition-api}

Write an explicit square matrix as a sum of Pauli strings, `A = sum_P c_P P` with `c_P = 2**-n Tr(P A)`, and remove small terms with a bound on the removed operator. Import the functions from `nwqlib.subroutines.pauli_decomposition`. Structured Pauli inputs do not use this dense conversion.

```python
import numpy as np
from nwqlib.subroutines.pauli_decomposition import decompose_matrix_to_pauli

decomposition = decompose_matrix_to_pauli(np.eye(3), pad_to_power_of_two=True)
for term in decomposition.terms:
    print(term.label, term.coefficient.real)
print(decomposition.input_dimension, decomposition.operator_dimension)
```

```text
II 0.75
IZ 0.25
ZI 0.25
ZZ -0.25
3 4
```

The 3 by 3 identity, zero-padded to 4 by 4, is `diag(1, 1, 1, 0) = (3 II + IZ + ZI - ZZ) / 4`, with qubit 0 the last character of each label.

## Decompose and prune

::: nwqlib.subroutines.pauli_decomposition
    options:
      show_root_heading: false
      heading_level: 3

## Source map

| Scientific step | Source | Code |
| --- | --- | --- |
| `c_P = 2**-n Tr(P A)` for Hermitian, trace-orthogonal Pauli strings | Pauli-basis orthogonality, stated in the module docstring | `nwqlib.subroutines.pauli_decomposition.decompose_matrix_to_pauli` |
| I/X/Y/Z block transform with componentwise averages | NWQLib, described in its docstring | `nwqlib.operators._pauli.pauli_coefficients` |
| Removed operator norm at most the pruned coefficient 1-norm | Triangle inequality with unit-norm Pauli strings | `nwqlib.subroutines.pauli_decomposition.prune_pauli_terms_relative` |
