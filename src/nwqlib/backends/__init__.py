"""Portable target declarations; native adapters load only when selected."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nwqlib.io.qasm import export_qasm

from nwqlib.backends.capabilities import (
    BackendCapability,
    BackendTarget,
    InstructionSupport,
)

# Each native operation has one actual owner. Merely reading a stored target
# must not import an SDK, even if that SDK happens to be installed.
_OPERATIONS = {
    "AerBackend": "connection",
    "NWQSimBackend": "nwqsim",
    "NWQSimSlurmBackend": "slurm",
    "SlurmProfile": "slurm",
    "IBMRuntimeBackend": "ibm_runtime",
    "IonQBackend": "ionq",
    "NexusBackend": "nexus",
    "assess": "assessment",
    "ProfileAssessment": "assessment",
    "PlanEstimate": "assessment",
    "Allocation": "profiles",
    "DeviceConfiguration": "profiles",
    "DeviceProfile": "profiles",
    "ModelDomain": "profiles",
    "TimeCoefficient": "profiles",
    "ModelUncertainty": "profiles",
    "CalibrationReference": "profiles",
    "TimeModel": "profiles",
    "align_telemetry": "telemetry",
    "PredictionLedger": "telemetry",
    "AttemptTiming": "telemetry",
    "AER_COUNTS_TARGET": "targets",
    "AER_STATEVECTOR_TARGET": "targets",
    "SampledBlock": "resources",
    "BackendRunResult": "results",
}

__all__ = [
    "BackendCapability", "BackendTarget", "InstructionSupport",
    "export_qasm", *_OPERATIONS,
]


def __getattr__(name):
    """Import the owning module of ``name`` on first access (PEP 562), as ``_OPERATIONS`` maps it."""
    if name == "export_qasm":
        return import_module("nwqlib.io.qasm").export_qasm
    if name in _OPERATIONS:
        return getattr(import_module(f"nwqlib.backends.{_OPERATIONS[name]}"), name)
    raise AttributeError(name)
