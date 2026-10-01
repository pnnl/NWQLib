"""Stored builtin target declarations, independent of all provider SDKs.

These are versioned package declarations, not refreshed device specifications.
A DeviceProfile supplies their runtime/build provenance and freshness. Missing
instruction/readout inventories remain unknown; coarse capabilities alone are
not a complete execution certification.
"""

from nwqlib.backends.capabilities import BackendCapability, BackendTarget, capability_set

AER_STATEVECTOR_TARGET = BackendTarget(
    name="aer_statevector",
    provider="qiskit_aer",
    capabilities=capability_set(BackendCapability.STATEVECTOR),
    description="Qiskit Aer exact statevector simulator.",
    readouts=("pauli_expectation", "probabilities", "amplitudes", "trajectory"),
    readout_features=("multi_position", "views"),
    reducers="registered",
)
AER_COUNTS_TARGET = BackendTarget(
    name="aer_counts",
    provider="qiskit_aer",
    capabilities=capability_set(BackendCapability.COUNTS, BackendCapability.NOISE_MODEL),
    description="Qiskit Aer finite-shot measured circuit simulator.",
    readouts=("counts",),
)
