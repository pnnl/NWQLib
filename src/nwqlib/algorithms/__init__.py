"""Scientific Method configurations and their current result operations."""

from importlib import import_module

# The same configured-type owners supply exports and builtin discovery. Required
# scientific configuration is never guessed to construct an inventory instance.
_METHOD_OWNERS = {
    "QHD": "qhd.method",
    "ADAPT": "gcim.adapt",
    "FixedGCIM": "gcim.fixed_basis",
    "Lanczos": "lanczos.method",
    "QLS": "qls.method",
    "QCELS": "qpe.method",
    "SPE": "qpe.method",
    "RFE": "qpe.method",
    "RWPE": "qpe.method",
    "LCHS": "lchs.method",
    "ExpectationMethod": "expectation",
}
_EXPORT_GROUPS = {
    "qhd": (
        "QHDAnalysis", "QHDVerification", "QuadraticSchedule", "CubicSchedule", "ShiftedCubicSchedule",
        "UniformState", "KineticGroundState", "GaussianState",
        "solve_augmented_lagrangian", "plan_augmented_lagrangian", "resume_augmented_lagrangian",
        "load_augmented_lagrangian",
        "constrained_grid_minimum", "ConstrainedQHDResult", "AugmentedLagrangian", "MultiplierBounds",
        "ConstraintPreprocessing", "AbsorbedBound", "ALIteration", "ALEvaluation", "ALResources",
        "AugmentedLagrangianRecord", "InnerRepresentation", "SlackAxis", "FormTrial", "ConstrainedGridMinimum",
        "refine_box", "resume_box_refinement", "load_box_refinement",
        "BoxRefinement", "RefinementLevel", "RefinementResources",
        "BoxRefinementResult", "StallSplit", "BinarySynthesis",
        "circuit_resources", "run_resources", "QHDCircuitResources", "QHDSynthesisProjection",
        "QHDErrorSource", "QHDRunResources", "QHDRunEntry", "evolution_bound", "QHDEvolutionBound",
    ),
    "gcim": ("ADAPTResult", "AdaptVerificationOptions"),
    "gcim.fixed_basis": ("FixedGCIMBasis", "FixedGCIMResult", "ProjectedPencil"),
    "lanczos": ("LanczosResult", "SensitivitySampling"),
    "qpe": ("QPEAnalysis", "QPEVerification"),
    "lchs": ("LCHSAnalysis",),
    "expectation": ("ExpectationAnalysis",),
    "protocol": ("Method", "AlgorithmDescriptor", "ApplicabilityError"),
    "registry": (
        "Registration",
        "AlgorithmRegistry",
        "third_party_registrations",
        "builtin_registrations",
        "algorithm_card",
        "options_schema",
        "direct_method",
    ),
}
_OWNERS = {
    **_METHOD_OWNERS,
    **{name: owner for owner, names in _EXPORT_GROUPS.items() for name in names},
}
__all__ = list(_OWNERS)


def __getattr__(name):
    if name not in _OWNERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_OWNERS[name]}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
