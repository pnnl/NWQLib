# State preparation {#state-preparation-api}

Build a circuit that prepares a given state vector from `|0...0>`: exactly, with the magnitude tree and phase diagonal of Mottonen et al. (quant-ph/0407010v1), or approximately, with the layered MPS disentangling circuit of Ran (arXiv:1908.07958v2) after a TT-SVD compression (Oseledets, doi:10.1137/090752286). Import the functions from `nwqlib.subroutines.state_preparation`, except `mps_to_circuit`, which imports from `nwqlib.subroutines.state_preparation.mps_circuit`. Inside a Program, [`select_preparation`](../extending.md#nwqlib.blocks.selection.select_preparation) chooses the preparation of a state input.

```python
import numpy as np
from nwqlib.subroutines.state_preparation import (
    analyze_mps_state_compression,
    build_qiskit_state_preparation,
    decompose_state_to_mps,
)

w = np.zeros(8)
w[[1, 2, 4]] = 1  # W state on 3 qubits, unnormalized
exact = build_qiskit_state_preparation(w)
print(round(exact.input_norm**2, 10), exact.preparation_l2_error)
cores = decompose_state_to_mps(w, max_bond_dim=1)
estimate = analyze_mps_state_compression(cores)
print(cores.bond_dimensions, round(estimate.estimated_normalized_fidelity, 10))
```

```text
3.0 0.0
(1, 1, 1, 1) 0.3333333333
```

The input has squared norm 3, and the direct circuit prepares the normalized W state exactly. Capping the bond dimension at 1 discards a total squared singular-value weight of `2/3`, so the estimated fidelity of the truncated tensor is `1 - 2/3 = 1/3`.

## Exact preparation

::: nwqlib.subroutines.state_preparation.direct
    options:
      show_root_heading: false
      heading_level: 3

## MPS compression

::: nwqlib.subroutines.state_preparation.mps
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_", "!^admit_layered_construction$", "!^layered_construction_size$"]

## Layered MPS circuits

::: nwqlib.subroutines.state_preparation.mps_circuit
    options:
      show_root_heading: false
      heading_level: 3

## Accuracy, cost and limits

Exact preparation:

- Basis states, full-uniform states and prefix-uniform states take exact fast paths. General vectors use a binary tree of conditional RY rotations followed by a phase diagonal. Pairwise `hypot` norms and `atan2` angles avoid squared-magnitude underflow, and the Walsh and Gray-code synthesis skips exact zero angles. No synthesis cutoff discards a requested small rotation.
- For an n-qubit generic vector, each magnitude tree or phase diagonal uses at most `max(0, 2**n - 2)` CX gates. General complex preparation uses both, and positive LCU coefficient preparation needs the magnitude tree only. Mottonen et al. interleave the phase multiplexors with the magnitude levels and cancel one CX per multiplexor, two per level, reaching `2**(n+1) - 2n - 2` CX from a basis state. The separate diagonal used here costs `2n - 2` more CX for a complex state and lets a nonnegative vector omit the phase stage.
- `preparation_l2_error = 0` describes the ideal-gate construction with no algorithmic approximation. It is not a measured bound on floating-point synthesis, backend execution or hardware error. The direct preparation record reports `fidelity_to_target = None` in `to_dict()` because ordinary construction does not evaluate fidelity. Binary64 roundoff can limit relative accuracy for components near or below machine precision, especially when a tiny component results from cancellation among rotations.

MPS compression and layered circuits:

- `build_mps_circuit_state_preparation` runs one TT-SVD sweep, or reuses a matching decomposition that NWQLib's planning passes to it. The core metadata includes the ranks, the content hash of the normalized input and the discarded singular-value weight. The full input and the economy SVDs still have their dimension-dependent storage and computation costs.
- The layered-construction check of `build_mps_circuit_state_preparation` uses the [layered MPS construction formula](../../ENGINEERING_CONSTANTS.md#layered-mps-construction-law), which follows the sweeps that scikit_tt repeats after every extracted gate. Contraction of the cores has its separate scalar-product and byte limits.
- The one-qubit gates of the MPS circuit include their global phase, so controlled preparations keep the relative phase as well as the state populations.
- The `DEFAULT_MAX_SVD_WORK` paragraph of the [constants registry](../../ENGINEERING_CONSTANTS.md#state-preparation) gives the default of 100,000,000 and the measurements behind the cost formula of the layered construction.

## Source map

Code paths are relative to `nwqlib.subroutines`. Equation and section numbers refer to the listed arXiv or journal versions.

| Scientific step | Source and location | Code |
| --- | --- | --- |
| Magnitude-tree RY angles | Mottonen et al., quant-ph/0407010v1, Sec. III, Eq. (8) | `state_preparation.direct._build_normalized_state_preparation` |
| Uniformly controlled RY and RZ lowering by Walsh transform and Gray code | Mottonen et al., quant-ph/0407010v1, Sec. II, Fig. 2 and Eq. (3), and Shende, Bullock and Markov, quant-ph/0406176v5, Theorem 8 | `_multiplexors._rotation_multiplexor` |
| Phase diagonal as one RZ multiplexor per qubit | Shende, Bullock and Markov, quant-ph/0406176v5, Theorem 7 | `_multiplexors.append_control_diagonal_phases` |
| CX counts `2**k` per multiplexor and `2**n - 2` per tree or diagonal | Shende, Bullock and Markov, quant-ph/0406176v5, Theorem 8 and Fig. 2, summed over the tree by NWQLib | `_multiplexors.multiplexor_resource_law` |
| TT-SVD sweep and truncation estimates | Oseledets, SIAM J. Sci. Comput. 33(5) (2011), doi:10.1137/090752286, Algorithm 1 (p. 2301) for the sweep. Proof of Theorem 2.2 and Eq. (2.5) (p. 2299) for the error estimate, with NWQLib's inner-product step for the norm and fidelity estimates | `state_preparation.mps.decompose_state_to_mps`, `state_preparation.mps.analyze_mps_state_compression` |
| Layered disentangling circuit | Ran, arXiv:1908.07958v2, Sec. III steps 1-4, Eqs. (6)-(9) | `state_preparation.mps_circuit.mps_to_circuit` |

## Entries on other pages

<a id="nwqlib.blocks.selection.select_preparation"></a>[`select_preparation`][nwqlib.blocks.selection.select_preparation] is documented on [Extending NWQLib](../extending.md).
