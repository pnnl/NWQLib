"""QSP subroutines, each imported from its module on first access.

The numerical fits import no quantum SDK.
"""

from importlib import import_module

_EXPORT_GROUPS = {
    "evolution": (
        "JacobiAngerExpansion",
        "QSP_OAA_QUERY_MULTIPLIER",
        "build_control_diagonal_generator_encoding",
        "build_qsp_evolution_encoding",
        "build_qsvt_circuit",
        "build_real_chebyshev_encoding",
        "compiled_select_resource_law",
        "jacobi_anger_expansion",
    ),
    "inverse": (
        "InverseChebyshevFit",
        "inverse_degree_law_bound",
    ),
    "phases": (
        "QSP_SOLVER_RESIDUAL_TOLERANCE",
        "SymmetricQSPPhases",
        "chebyshev_grid",
        "chebyshev_polynomial_sup_bound",
        "evaluate_qsp_polynomial",
        "solve_symmetric_qsp_phases",
        "wx_phases_to_reflection",
    ),
    "shortcut": (
        "KernelReflectionPolynomial",
        "dalzell_eta_from_precision",
    ),
}
_OWNERS = {name: owner for owner, names in _EXPORT_GROUPS.items() for name in names}
__all__ = list(_OWNERS)


def __getattr__(name):
    owner = _OWNERS.get(name)
    if owner is None:
        raise AttributeError(name)
    return getattr(import_module(f".{owner}", __name__), name)
