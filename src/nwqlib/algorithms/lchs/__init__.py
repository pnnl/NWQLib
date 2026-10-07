"""LCHS Method, scientific Result, explicit checks and coefficient analysis."""

from importlib import import_module

_EXPORT_GROUPS = {
    "method": ("LCHS",),
    "primary_records": ("LCHSAnalysis",),
    "verification": ("LCHSVerification",),
    "refinement": ("LCHSRefinement",),
    "time_independent_terms": ("cartesian_decomposition",),
    "provider_config": ("ProviderConfig", "ProviderParameter", "ResolvedProviderConfig"),
    "providers": (
        "LCHS_KERNEL_IMPLEMENTATIONS", "LCHS_K_QUADRATURE_IMPLEMENTATIONS", "LCHSCoefficientPlan",
        "LCHSKernelProvider", "LCHSQuadratureProvider", "resolve_lchs_coefficient_plan",
    ),
    "inhomogeneous_theory": ("duhamel_quadrature",),
}
_OWNERS = {name: owner for owner, names in _EXPORT_GROUPS.items() for name in names}
__all__ = list(_OWNERS)


def __getattr__(name):
    owner = _OWNERS.get(name)
    if owner is None:
        raise AttributeError(name)
    return getattr(import_module(f".{owner}", __name__), name)
