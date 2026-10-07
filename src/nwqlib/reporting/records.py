"""Small text/data sections kept by scientific diagnostic formatters."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from nwqlib.serialization import FieldSerializedRecord


@dataclass(frozen=True, kw_only=True)
class ReportSection(FieldSerializedRecord):
    """A titled section of a report, with lines of text and machine-readable data.

    Build it with keyword arguments, for example
    `ReportSection(title="Reference", lines=("E_ref = -1.0",), data={"E_ref": -1.0})`.
    `title` is required. In the [GCiM guide](../algorithms/gcim.md),
    `chemistry_reference_report_section` returns one to show a stored
    reference panel next to an obtained energy. `Result.report()` does not
    include such a section, so print or store it beside the report.

    Args:
        title: Section title.
        lines: Lines of text for a printed report.
        data: Machine-readable data of the section.
    """

    title: str
    lines: tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)
