"""State preparation with NumPy compression and selected optional circuits."""

from importlib import import_module

from nwqlib.subroutines.state_preparation.mps import (
    MPSCompressionAnalysis,
    MPSDecomposition,
    analyze_mps_state_compression,
    decompose_state_to_mps,
)

__all__ = [
    "DirectStatePreparation",
    "MPSCompressionAnalysis",
    "MPSCircuitStatePreparation",
    "MPSDecomposition",
    "analyze_mps_state_compression",
    "build_mps_circuit_state_preparation",
    "build_qiskit_state_preparation",
    "decompose_state_to_mps",
    "validate_mps_circuit_state_preparation",
]

_CIRCUITS = {
    "DirectStatePreparation": "direct",
    "build_qiskit_state_preparation": "direct",
    "MPSCircuitStatePreparation": "mps_circuit",
    "build_mps_circuit_state_preparation": "mps_circuit",
    "validate_mps_circuit_state_preparation": "mps_circuit",
}


def __getattr__(name):
    if name not in _CIRCUITS:
        raise AttributeError(name)
    from nwqlib._optional import optional_import

    optional_import("qiskit", extra="qiskit")
    value = getattr(import_module(f"{__name__}.{_CIRCUITS[name]}"), name)
    globals()[name] = value
    return value
