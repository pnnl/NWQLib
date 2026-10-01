"""Implementation registry for LCU PREP subroutines."""

from __future__ import annotations

from typing import Any

from nwqlib.subroutines._registry_utils import registry_metadata_function


LCU_PREPARATION_IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "direct": {
        "provider": "nwqlib_native",
        "method": "conditional magnitude rotations on the LCU coefficient register",
        "package_names": ("qiskit",),
        "capability_notes": (
            "Exact dense coefficient-register PREP for small validation LCU blocks."
        ),
    },
    "mps_circuit": {
        "provider": "nwqlib_native",
        "method": "MPS disentangling circuit for the LCU coefficient register",
        "package_names": ("qiskit", "scikit-tt"),
        "capability_notes": (
            "Uses NWQLib's circuit-level MPS state-preparation backend for PREP; "
            "finite layers approximate the target amplitudes with unevaluated circuit error. "
            "TT-SVD discarded weight is a separate compression quantity. "
            "Entangled coefficient states require the tensor optional dependency."
        ),
    },
}


lcu_preparation_implementation_metadata = registry_metadata_function(
    LCU_PREPARATION_IMPLEMENTATIONS, slot="LCU PREP"
)


__all__ = [
    "LCU_PREPARATION_IMPLEMENTATIONS",
    "lcu_preparation_implementation_metadata",
]
