"""Shared JSON-serialization helpers for record payloads.

Leaf module: imports nothing from ``nwqlib``. It is safe for algorithms and
subroutines; backend modules should avoid importing through
``nwqlib.subroutines`` during package initialization. These shared array and
complex-scalar encodings keep the same JSON shapes across record consumers.
Changing those shapes changes the record's serialization contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["array_to_json", "complex_to_dict"]


def array_to_json(value: Any) -> Any:
    """Return a compact JSON-like array representation."""

    if value is None:
        return None
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {
            "real": np.real(array).tolist(),
            "imag": np.imag(array).tolist(),
        }
    return array.tolist()


def complex_to_dict(value: complex) -> dict[str, float]:
    """Return a JSON-like complex-number record."""

    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}
