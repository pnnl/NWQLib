"""Implementation registry for block-encoding subroutines."""

from __future__ import annotations

from typing import Any


BLOCK_ENCODING_IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "multiplexed_pauli": {
        "provider": "nwqlib_native",
        "method": (
            "PREP-SELECT-PREP^dagger LCU with one dependency-projected exact "
            "UCGate per system qubit"
        ),
        "package_names": ("qiskit",),
        "capability_notes": (
            "General square power-of-two-dimension inputs; alpha is the Pauli "
            "coefficient 1-norm. SELECT preserves the Pauli strings and projects "
            "each local I/X/Y/Z table onto its exact address-bit dependency "
            "support. Decomposing a dense matrix remains small-dense validation "
            "preprocessing, not a scalable oracle."
        ),
    },
    "banded": {
        "provider": "nwqlib_native",
        "method": (
            "exact LCU of cyclic shifts for periodic banded Toeplitz (circulant) "
            "operators; shifts realized as Draper-style modular adders "
            "(arXiv:quant-ph/0008033v1) sharing one "
            "QFT/IQFT pair (Fourier-diagonal controlled phases)"
        ),
        "package_names": ("qiskit",),
        "capability_notes": (
            "Scalable-oracle path: accepts a band specification without ever "
            "materializing a dense matrix; alpha is the band-coefficient 1-norm. "
            "The supported specification is periodic-only; see Camps et al. "
            "(arXiv:2203.10236v4) for structured banded block encodings."
        ),
    },
    "dense_dilation": {
        "provider": "qiskit",
        "method": (
            "exact one-ancilla unitary dilation completed via SVD and synthesized "
            "as one dense UnitaryGate"
        ),
        "package_names": ("qiskit", "numpy"),
        "capability_notes": (
            "Small dense validation fallback with the optimal alpha = ||A||_2; "
            "eager O(4^(n+1)) synthesis, no quantum-advantage claim."
        ),
    },
}


__all__ = [
    "BLOCK_ENCODING_IMPLEMENTATIONS",
]
