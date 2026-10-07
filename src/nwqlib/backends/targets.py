"""Stored builtin target declarations, independent of all provider SDKs.

These are versioned package declarations, not refreshed device specifications.
A DeviceProfile supplies their runtime/build provenance and freshness. Missing
instruction and readout inventories remain unknown. Coarse capabilities alone
are not a complete execution certification.
"""

from nwqlib.backends.capabilities import BackendCapability, BackendTarget

AER_STATEVECTOR_TARGET = BackendTarget(
    name="aer_statevector",
    provider="qiskit_aer",
    capabilities=(BackendCapability.STATEVECTOR,),
    description="Qiskit Aer exact statevector simulator.",
    readouts=("pauli_expectation", "probabilities", "amplitudes", "trajectory"),
    readout_features=("multi_position", "views"),
    reducers="registered",
)
"""Built-in target of the Qiskit Aer statevector simulator, for exact readouts.

It declares the `statevector` capability and the readouts `pauli_expectation`,
`probabilities`, `amplitudes` and `trajectory`, with the trajectory features
`multi_position` and `views` and every registered reducer. It leaves
`artifacts`, `instructions` and `program_nodes` unknown. The capability
checks that read these fields report `"unknown"`, so the capability axis of
an assessment with this target cannot be `"feasible"` until `revise` declares
them. Use it as `DeviceConfiguration(target=...)` for exact readouts, and use
[`AER_COUNTS_TARGET`][nwqlib.backends.targets.AER_COUNTS_TARGET] for sampled
shots. It is a stored declaration that imports no SDK, not a live device
inventory.
"""
AER_COUNTS_TARGET = BackendTarget(
    name="aer_counts",
    provider="qiskit_aer",
    capabilities=(BackendCapability.COUNTS, BackendCapability.NOISE_MODEL),
    description="Qiskit Aer finite-shot measured circuit simulator.",
    readouts=("counts",),
)
"""Built-in target of the Qiskit Aer simulator with measured shots.

It declares the `counts` and `noise_model` capabilities and the `counts`
readout. It leaves `readout_features`, `reducers`, `artifacts`, `instructions`
and `program_nodes` unknown. The capability checks that read these fields
report `"unknown"`, so the capability axis of an assessment with this target
cannot be `"feasible"` until `revise` declares them. Use it as
`DeviceConfiguration(target=...)` for a Plan with sampled shots. It is a
stored declaration that imports no SDK, not a live device inventory.
"""
