"""QSP subroutines, each imported from its module on first access.

The numerical fits import no quantum SDK.
"""

from importlib import import_module

_EXPORT_GROUPS = {
    "evolution": (
        "JacobiAngerExpansion",
        "QSP_OAA_QUERY_MULTIPLIER",
        "QubitizedEvolutionPlan",
        "QubitizedTarget",
        "QubitizedTargetPlan",
        "SharedQSVTEvolutionPlan",
        "build_control_diagonal_generator_encoding",
        "build_qsp_evolution_encoding",
        "build_qsvt_circuit",
        "build_qubitized_evolution",
        "build_qubitized_evolution_query",
        "build_real_chebyshev_encoding",
        "build_shared_qsvt_evolution",
        "build_shared_qsvt_evolution_query",
        "compiled_select_resource_law",
        "imaginary_time_filter_error_terms",
        "imaginary_time_target",
        "jacobi_anger_expansion",
        "plan_jacobi_anger_gqet",
        "plan_jacobi_anger_laurent",
        "plan_qubitized_evolution",
        "plan_shared_qsvt_evolution",
        "prepare_jacobi_anger_gqet",
        "prepare_jacobi_anger_laurent",
        "prepare_qubitized_target",
        "with_phase_allowance",
    ),
    "inverse": (
        "InverseChebyshevPolynomial",
    ),
    "imaginary_time": (
        "ImaginaryTimeDegreeSelection",
        "ImaginaryTimeExpansion",
        "PreparedImaginaryTimeFilter",
        "select_imaginary_time_degree",
        "imaginary_time_expansion",
        "prepare_imaginary_time_filter",
    ),
    "phases": (
        "QSP_SOLVER_RESIDUAL_TOLERANCE",
        "WxQSPPhases",
        "LaurentQSPPhases",
        "PhaseSynthesisEvidence",
        "chebyshev_grid",
        "chebyshev_polynomial_sup_bound",
        "evaluate_qsp_polynomial",
        "solve_wx_qsp_phases",
        "solve_laurent_qsp_phases",
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
