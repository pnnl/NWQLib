"""Trotterization subroutines.

Product-formula step selection under the commutator-scaling bounds of
Childs, Su, Tran, Wiebe and Zhu, Phys. Rev. X 11, 011020 (2021),
doi:10.1103/PhysRevX.11.011020. Unless marked as arXiv:1912.08854v3,
proposition and equation numbers in this package follow that Phys. Rev. X
version. The module ``error_budget`` lists the proposition,
equation and page for each bound, their arXiv:1912.08854v3 equivalents and
the term-ordering convention the bounds assume.
"""

from nwqlib.subroutines.trotterization.error_budget import (
    TROTTER_BOUND_REFERENCE,
    TrotterStepSelection,
    build_trotter_evolution_circuit,
    dense_trotter_bound_coefficient,
    evaluate_trotter_bound,
    select_trotter_step_count,
    trotter_bound_coefficient,
    trotter_error_bound,
)

__all__ = [
    "TROTTER_BOUND_REFERENCE",
    "TrotterStepSelection",
    "build_trotter_evolution_circuit",
    "dense_trotter_bound_coefficient",
    "evaluate_trotter_bound",
    "select_trotter_step_count",
    "trotter_bound_coefficient",
    "trotter_error_bound",
]
