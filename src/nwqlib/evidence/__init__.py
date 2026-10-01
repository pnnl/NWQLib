"""SDK-free scientific facts and evidence provenance.

This package records what is known about a Result's quantity and on what
basis. It never acquires data, reruns a Method or computes a reference. A
Fact keeps its availability separate from its evidence kind, and a
FramedFact binds it to one scientific frame and parameter point. A Method's
ErrorModel lists the error sources its output must cover. ClaimAssessment
applies a post-run criterion by the triangle inequality over those sources
and a union bound over their failure probabilities. CheckSpec selects one
explicit scalar check, and Certificate attaches completed checks to an
assessment.

Three rules keep a claim from becoming stronger than its evidence. A
component bound, such as the sampling radius, never covers total error,
which needs every required source. An accuracy claim passes only on
witnessed proved or certified evidence without open assumptions, so a
numerical estimate or an assertion stays listed as unverified. A
verification fact answers only the options record and CheckSpec that
produced it. docs/error_evidence.md and docs/verification.md define these
contracts.
"""

from typing import TYPE_CHECKING

from .records import Evidence, Fact, WorkProvenance

if TYPE_CHECKING:
    from .scalar_bound import ScalarBoundResult, resolve_scalar_bound
    from .error_model import (
        AssessmentContext, Certificate, CheckAssessment, CheckDomain, CheckSpec, ClaimAssessment, ErrorFrame, ErrorModel,
        ErrorTerm, FramedFact, TargetReference, assemble_check, assess_result, result_context,
        predicate_frame, predicate_value,
    )
    from .statistics import CrossCovariance, EstimatorContribution, IndependenceLaw, VarianceAssessment, linear_variance

__all__ = [
    "ScalarBoundResult", "resolve_scalar_bound",
    "Evidence", "Fact", "WorkProvenance", "ErrorFrame", "ErrorTerm", "ErrorModel",
    "ClaimAssessment", "CheckDomain", "CheckSpec", "CheckAssessment", "Certificate",
    "assemble_check", "assess_result", "result_context", "EstimatorContribution", "CrossCovariance",
    "VarianceAssessment", "linear_variance", "TargetReference", "IndependenceLaw",
    "FramedFact", "AssessmentContext", "predicate_frame", "predicate_value",
]


def __getattr__(name):
    from importlib import import_module
    if name in {"ScalarBoundResult", "resolve_scalar_bound"}:
        return getattr(import_module(".scalar_bound", __name__), name)
    if name in {"EstimatorContribution", "CrossCovariance", "VarianceAssessment", "linear_variance", "IndependenceLaw"}:
        return getattr(import_module(".statistics", __name__), name)
    if name in __all__:
        return getattr(import_module(".error_model", __name__), name)
    raise AttributeError(name)
