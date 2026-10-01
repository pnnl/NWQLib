"""Scientific questions, output quantities, and optional accuracy requests."""

from .records import (
    Accuracy, ConstrainedOptimization, Eigenphase, Eigenproblem, Eigenvalue, Expectation, LinearDynamics,
    LinearSystem, NormalizedExpectation, NormSquared, Optimization,
    OptimizationCandidate, OutputRecord, ProblemRecord, QuadraticForm, Samples,
    Solution, SpectralEstimation, StateVector,
)

__all__ = [
    "Accuracy", "ConstrainedOptimization", "Eigenphase", "Eigenproblem", "Eigenvalue", "Expectation",
    "LinearDynamics", "LinearSystem", "NormalizedExpectation", "NormSquared",
    "Optimization", "OptimizationCandidate", "OutputRecord", "ProblemRecord",
    "QuadraticForm", "Samples", "Solution", "SpectralEstimation", "StateVector",
]

_INPUT_EXPORTS = (
    "PhysicalScale", "QiskitPreparation", "StateInput", "StatePreparationSpec",
    "ingest_occupation", "ingest_product", "ingest_vector",
    "prepare_qiskit", "bind_preparation_circuit", "state_input",
)
__all__ += list(_INPUT_EXPORTS)


def __getattr__(name):
    if name in _INPUT_EXPORTS:
        from importlib import import_module
        return getattr(import_module(".inputs", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
