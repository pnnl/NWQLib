"""SDK-free selected blocks with an explicit bounded Qiskit lowering seam."""

from .records import BlockSemantics, PauliEncoding, Primitive, SelectedConstruction, SelectedDefinition, SelectedKernel
from .selection import (
    SelectedBlock, pauli_readout_cx_bound, select_pauli_preparation, select_pauli_readout,
    select_preparation, select_signed_pauli, select_zero_reflection, signed_pauli_cx_bound,
    transform_block,
)
from .lowering import LogicalCircuit, lower_qiskit
from .encoding import select_block_encoding

__all__ = [
    "BlockSemantics", "PauliEncoding", "Primitive", "SelectedConstruction", "SelectedDefinition", "SelectedKernel",
    "SelectedBlock", "select_pauli_preparation", "select_preparation", "select_signed_pauli",
    "select_zero_reflection", "signed_pauli_cx_bound", "transform_block", "LogicalCircuit", "lower_qiskit",
    "select_pauli_readout", "pauli_readout_cx_bound",
    "select_block_encoding",
]
