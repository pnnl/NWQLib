"""Small text/data sections kept by scientific diagnostic formatters."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from nwqlib.serialization import FieldSerializedRecord


@dataclass(frozen=True, kw_only=True)
class ReportSection(FieldSerializedRecord):
    """A named section in a human-readable and machine-readable report.

    Args:
        title: Section title.
        lines: Human-readable lines for text rendering.
        data: Machine-readable section data.
    """

    title: str
    lines: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)
