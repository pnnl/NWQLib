# Hamiltonian evolution {#hamiltonian-evolution-api}

Build Qiskit circuits for Hamiltonian-evolution steps of Pauli sums, `exp(-i * time_step * sum_j c_j P_j)`, and convert between NWQLib's compact Pauli labels and Qiskit's Pauli strings. Import the records and functions from `nwqlib.subroutines.hamiltonian_evolution`.

```python
import numpy as np
from scipy.linalg import expm
from qiskit.quantum_info import Operator, SparsePauliOp
from nwqlib.subroutines.hamiltonian_evolution import (
    PauliEvolutionBlock,
    PauliEvolutionTerm,
    build_pauli_evolution_circuit,
)

hopping = tuple(
    PauliEvolutionTerm(pauli=label, coefficient=0.4)
    for label in ("x0x1", "y0y1")
)
block = PauliEvolutionBlock(terms=hopping, time_step=0.5, kind="kinetic")
circuit = build_pauli_evolution_circuit([block], num_qubits=2)
print(dict(circuit.count_ops()))
H = Operator(SparsePauliOp(["XX", "YY"], [0.4, 0.4])).data
print(np.allclose(Operator(circuit).data, expm(-0.5j * H)))
```

```text
{'xx_plus_yy': 1}
True
```

The block `0.4 (XX + YY)` becomes one `XXPlusYYGate`, which is exact because `XX` and `YY` commute.

## Conventions

- A default-kind block with terms `c_j P_j` and duration `time_step` means `exp(-i * time_step * sum_j c_j P_j)`, approximated by one application of the chosen product formula unless the terms commute. The `pauli_evolution` module text below describes the exact number-projector and `XX + YY` forms.
- Compact labels such as `x0z2` index qubits explicitly, in strictly increasing order, while Qiskit dense labels put qubit 0 in the rightmost character.

## Pauli records and circuits

::: nwqlib.subroutines.hamiltonian_evolution.pauli_ir
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_", "!^coerce_pauli_evolution_"]

::: nwqlib.subroutines.hamiltonian_evolution.pauli_evolution
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_", "!^make_evolution_synthesis$", "!^resolve_number_projector_lowering$", "!^wrapped_phase_workspace_bytes$", "!^structured_number_projector_provider$", "!^wrapped_projector_phase$"]

## Pauli labels

::: nwqlib.subroutines.hamiltonian_evolution.pauli_labels
    options:
      show_root_heading: false
      heading_level: 3

## Source map

Code paths are relative to `nwqlib.subroutines.hamiltonian_evolution`. Most of these forms have no paper equation, so their rows name the identity or dependency behind them.

| Scientific step | Source | Code |
| --- | --- | --- |
| `exp(-i dt c (XX + YY))` as one `XXPlusYYGate(4 dt c)` | XX and YY commute, and Qiskit defines `XXPlusYYGate(theta) = exp(-i theta (XX + YY)/4)`. The parameter is formed as `(2 dt c) * 2`, so a finite parameter is exactly twice the stored angle of a QHD hopping block, because forming `4 dt` first can overflow where theta is finite | `pauli_evolution.append_pauli_evolution_block`, `pauli_evolution._hopping_parameter` |
| `exp(-i angle (prod_q n_q - I/2**s))` with `n_q = (I - Z_q)/2` | A phase `-angle` on the all-ones basis state plus the global phase `angle/2**s` | `pauli_evolution.append_number_projector_phase` |
| Number-projector CX count, phase-diagonal provider | Shende, Bullock and Markov, quant-ph/0406176v5, Theorem 7, p. 10, with the `2**k` CX of a multiplexed Rz on k select bits stated after Theorem 8, p. 11. The [QHD guide](../../algorithms/qhd.md#cost-and-existing-evidence) states which provider each support size selects | `pauli_evolution.structured_number_projector_provider` |
| Number-projector CX count, multi-controlled-phase provider | Qiskit's `MCPhaseGate` definition and its multi-controlled X synthesis, checked against transpiled circuits | `pauli_evolution._mcphase_cx` |
| Pauli rotation by basis change and a CX parity ladder | Standard identity `exp(-i angle P/2)` stated in its docstring | `pauli_evolution.apply_pauli_rotation` |
