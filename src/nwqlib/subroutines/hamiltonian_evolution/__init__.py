"""Lazy Hamiltonian evolution records and native construction exports."""

from importlib import import_module

_EXPORTS = {
    "PauliEvolutionSynthesis": "nwqlib.subroutines.hamiltonian_evolution.pauli_evolution",
    "append_pauli_evolution_block": "nwqlib.subroutines.hamiltonian_evolution.pauli_evolution",
    "apply_pauli_rotation": "nwqlib.subroutines.hamiltonian_evolution.pauli_evolution",
    "build_pauli_evolution_circuit": "nwqlib.subroutines.hamiltonian_evolution.pauli_evolution",
    "make_evolution_synthesis": "nwqlib.subroutines.hamiltonian_evolution.pauli_evolution",
    "PauliEvolutionBlock": "nwqlib.subroutines.hamiltonian_evolution.pauli_ir",
    "PauliEvolutionTerm": "nwqlib.subroutines.hamiltonian_evolution.pauli_ir",
    "coerce_pauli_evolution_block": "nwqlib.subroutines.hamiltonian_evolution.pauli_ir",
    "coerce_pauli_evolution_term": "nwqlib.subroutines.hamiltonian_evolution.pauli_ir",
    "pauli_terms_to_dicts": "nwqlib.subroutines.hamiltonian_evolution.pauli_ir",
    "make_pauli_label": "nwqlib.subroutines.hamiltonian_evolution.pauli_labels",
    "parse_pauli_label": "nwqlib.subroutines.hamiltonian_evolution.pauli_labels",
    "pauli_label_to_qiskit_string": "nwqlib.subroutines.hamiltonian_evolution.pauli_labels",
    "sparse_pauli_op_from_terms": "nwqlib.subroutines.hamiltonian_evolution.pauli_labels",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
