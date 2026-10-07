"""Target of host kernels, the classical routines a Method selects.

They run synchronously in this Python process and return the ``host_scalars`` readout.
"""

from .capabilities import BackendCapability, BackendTarget

HOST_TARGET = BackendTarget(
    name="local_host", provider="nwqlib_host", capabilities=(BackendCapability.HOST_KERNEL,),
    description="synchronous selected local numerical kernel; native CPU/RSS and floating error unbounded",
    artifacts=("selected_construction",), readouts=("host_scalars",),
    program_nodes=("classical_stage",), instructions=(),
    host_dependencies=("numpy", "scipy", "qiskit"),
)
