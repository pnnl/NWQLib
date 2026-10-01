# Hamiltonian evolution API

Public exports are available from `nwqlib.subroutines.hamiltonian_evolution`. Pauli evolution builds the selected circuit from its admitted terms. A default-kind block with terms `c_j P_j` and duration `time_step` means `exp(-i * time_step * sum_j c_j P_j)`, approximated by one application of the selected product formula unless the terms commute. The `pauli_evolution` module describes the exact number-projector and XX+YY forms. Compact labels such as `x0z2` index qubits explicitly, while Qiskit dense labels put qubit 0 in the rightmost character.

## Pauli records and circuits

::: nwqlib.subroutines.hamiltonian_evolution.pauli_ir

::: nwqlib.subroutines.hamiltonian_evolution.pauli_evolution

## Source map

Owners are relative to `nwqlib.subroutines.hamiltonian_evolution`. Most of these forms have no paper equation, so their rows name the identity or dependency behind them.

| Scientific step | Source | Code owner |
| --- | --- | --- |
| `exp(-i dt c (XX + YY))` as one `XXPlusYYGate(4 dt c)` | XX and YY commute, and Qiskit defines `XXPlusYYGate(theta) = exp(-i theta (XX + YY)/4)`. The parameter is formed as `(2 dt c) * 2`, so a finite parameter is exactly twice the stored angle of a QHD hopping block, because forming `4 dt` first can overflow where theta is finite | `pauli_evolution.append_pauli_evolution_block`, `pauli_evolution._hopping_parameter` |
| `exp(-i angle (prod_q n_q - I/2**s))` with `n_q = (I - Z_q)/2` | A phase `-angle` on the all-ones basis state plus the global phase `angle/2**s` | `pauli_evolution.append_number_projector_phase` |
| Number-projector CX count, phase-diagonal provider | Shende, Bullock and Markov, quant-ph/0406176v5, Theorem 7, p. 10, with the `2**k` CX of a multiplexed Rz on k select bits stated after Theorem 8, p. 11 | `pauli_evolution.structured_number_projector_provider` |
| Number-projector CX count, multi-controlled-phase provider | Qiskit's `MCPhaseGate` definition and its multi-controlled X synthesis, checked against transpiled circuits | `pauli_evolution._mcphase_cx` |
| Pauli rotation by basis change and a CX parity ladder | Standard identity `exp(-i angle P/2)` stated in the owner's docstring | `pauli_evolution.apply_pauli_rotation` |
