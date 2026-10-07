"""Leaf record-serialization helpers shared by reporting and LCHS records.

This module imports no other NWQLib module, so ``reporting.records`` and the
LCHS provider and application records (``algorithms/lchs/provider_config.py``,
``algorithms/lchs/solution_error_budget.py``) use its recursive conversion
without an import cycle.
"""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
from typing import Any, Mapping

import numpy as np


def _record_value(value: Any) -> Any:
    """Return the JSON-like form of one record field value.

    Enums serialize to their value, NumPy real scalars to native floats, other
    NumPy scalars through ``item``, and nested records through their own
    ``to_dict``. Containers recurse into their values. Array formats remain the
    responsibility of their record owners.
    """

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return _record_value(value.to_dict())
    if isinstance(value, Mapping):
        return {key: _record_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_record_value(item) for item in value]
    return value


class FieldSerializedRecord:
    """``dataclasses.fields()``-driven ``to_dict`` base for contract records.

    Keys follow dataclass field declaration order. Subclasses override
    ``to_dict`` only for genuine custom serialization (numpy arrays, derived
    keys). Such overrides call this base and patch its result instead of
    listing the fields again.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like representation with keys in declaration order."""

        return {
            record_field.name: _record_value(getattr(self, record_field.name))
            for record_field in fields(self)
        }
