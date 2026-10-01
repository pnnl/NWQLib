"""NWQLib package root.

Root modules resolve lazily (PEP 562), so importing the root does not load
the whole library. ``__all__`` lists the current advertised entries; each
scientific owner defines its supported operations.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("nwqlib")
except PackageNotFoundError:
    __version__ = "0.0.0"

_SCIENTIST_OPERATIONS = ("plan", "compare", "estimate", "prepare", "submit", "solve",
                         "load_result", "load_run", "methods")
_SEARCH_OPERATIONS = ("scan",)
_SCIENTIFIC_INPUTS = (
    "Accuracy", "ConstrainedOptimization", "Eigenphase", "Eigenproblem", "Eigenvalue", "Expectation",
    "LinearDynamics", "LinearSystem", "NormalizedExpectation", "NormSquared",
    "Optimization", "OptimizationCandidate", "QuadraticForm", "Samples",
    "Solution", "SpectralEstimation", "StateVector",
)

__all__ = [
    "__version__",
    "algorithms",
    "backends",
    "execution",
    "reporting",
    "subroutines",
    *_SCIENTIST_OPERATIONS,
    *_SEARCH_OPERATIONS,
    *_SCIENTIFIC_INPUTS,
]

def __getattr__(name: str) -> Any:
    """Resolve a root name lazily (PEP 562), importing only its owner.

    A missing first-level module becomes AttributeError. An ImportError raised
    inside an existing module, such as a missing optional dependency, propagates
    unchanged so it is not mistaken for a missing attribute.
    """
    if name in _SCIENTIFIC_INPUTS:
        return getattr(import_module("nwqlib.problems.records"), name)
    if name in _SCIENTIST_OPERATIONS:
        return getattr(import_module("nwqlib.scientist"), name)
    if name in _SEARCH_OPERATIONS:
        return getattr(import_module("nwqlib.search"), name)
    try:
        return import_module(f"nwqlib.{name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"nwqlib.{name}":
            raise AttributeError(
                f"module 'nwqlib' has no attribute {name!r}"
            ) from None
        raise


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
