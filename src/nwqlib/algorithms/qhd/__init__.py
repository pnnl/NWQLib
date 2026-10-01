"""QHD Method on a one-hot Dirichlet or periodic grid or a binary periodic grid, its schedules, initial
states, result records, the augmented-Lagrangian layer, box refinement and fault-tolerant resource laws.

Each public name resolves on first use from the module that owns it (``_OWNERS``, PEP 562), so importing
this package loads no SymPy, SciPy or Qiskit code, as ``docs/FRAMEWORK.md``, "Package Import Surface",
requires. The QHD guide, ``docs/algorithms/qhd.md``, is the reading entry, and the ``method`` module
docstring maps the Method to Leng et al., arXiv:2303.01471v1.
"""

from importlib import import_module

_OWNERS = {
    "QHD": "method",
    "BinarySynthesis": "binary",
    **dict.fromkeys(("QuadraticSchedule", "CubicSchedule", "ShiftedCubicSchedule"), "schedules"),
    **dict.fromkeys(("UniformState", "KineticGroundState", "GaussianState"), "initial_state"),
    **dict.fromkeys(("QHDAnalysis", "QHDVerification"), "records"),
    **dict.fromkeys(("solve_augmented_lagrangian", "plan_augmented_lagrangian", "resume_augmented_lagrangian",
                     "load_augmented_lagrangian",
                     "constrained_grid_minimum", "ConstrainedQHDResult"), "constrained"),
    **dict.fromkeys(("AugmentedLagrangian", "MultiplierBounds", "ConstraintPreprocessing", "AbsorbedBound",
                     "ALIteration", "ALEvaluation", "ALResources", "AugmentedLagrangianRecord",
                     "InnerRepresentation", "SlackAxis", "FormTrial",
                     "ConstrainedGridMinimum"), "constrained_records"),
    **dict.fromkeys(("refine_box", "resume_box_refinement", "load_box_refinement"), "refinement"),
    **dict.fromkeys(("circuit_resources", "run_resources", "QHDCircuitResources", "QHDSynthesisProjection",
                     "QHDErrorSource", "QHDRunResources", "QHDRunEntry"), "resources"),
    **dict.fromkeys(("evolution_bound", "QHDEvolutionBound"), "evolution_bounds"),
    **dict.fromkeys(
        ("BoxRefinement", "RefinementLevel", "RefinementResources", "BoxRefinementResult", "StallSplit"),
        "refinement_records",
    ),
}
__all__ = list(_OWNERS)


def __getattr__(name):
    if name not in _OWNERS:
        raise AttributeError(name)
    return getattr(import_module(f".{_OWNERS[name]}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
