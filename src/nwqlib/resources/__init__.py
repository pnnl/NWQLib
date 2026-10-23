"""Resource estimates of a Plan's circuit construction (`SelectedConstruction`), with the evidence of each metric.

No quantum SDK is imported.
"""

from typing import TYPE_CHECKING

from .records import ResourceContext, ResourceLaw, ResourceQuantity, WorkloadEstimate, Workspace

if TYPE_CHECKING:
    from .fold import estimate

__all__ = ["ResourceContext", "ResourceLaw", "ResourceQuantity", "WorkloadEstimate", "Workspace", "estimate"]


def __getattr__(name):
    if name == "estimate":
        from .fold import estimate
        return estimate
    raise AttributeError(name)
