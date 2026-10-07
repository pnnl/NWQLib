"""Internal solution-level error accounting for time-independent LCHS.

Two classes of error are kept apart. Discretization error separates the
exact propagator from the ideal finite sum sum_j c_j U_j. Its stages are the
kernel-integral error ("kernel_approximation"), the k-quadrature ("k_quadrature") and,
with a constant source, the source-time quadrature ("duhamel_quadrature").
Realization error separates that finite sum from the realized block. Its
stages are product-formula or QSP branch synthesis and coefficient or
input-state preparation.

Each stage becomes a physical-vector bound by its own rule. The kernel_approximation,
k_quadrature and trotter_synthesis stages are sum_a w_a*r_a*e_a over the
physical applications a. Here w_a is the input norm times the magnitude of
the Duhamel weight, r_a = exp(shift*elapsed_a) and e_a is the operator-level
bound (sum_j |c_j|*e_j over nodes for trotter_synthesis). qsp_synthesis is
gamma times the compensated QSP recovery bound, and
lcu_coefficient_preparation is 2*gamma*delta_lcu, with gamma the physical
recovery scale of the prepared coefficient state. duhamel_quadrature and
initial_state_preparation are supplied physical bounds. Each stage is
reported with its own value or reason. propagate_physical_error sums the
stage values once into the algorithmic-approximation component of the
requested output. Floating-point and sampling errors remain separate terms,
and one missing stage makes that sum unknown.

The sum bounds the distance between the exact solution of du/dt = -A u + b
and the output that the selected construction would give in exact
arithmetic without noise, through the stages listed above and only under
their premises (in particular the numerical PSD decision). It does not bound
binary64 rounding in classical exponentials, phases, angles and gate
synthesis, transpiler approximations, hardware noise or shot noise. The
controlled dense branches of dense_exact carry only rounding, because
qiskit_compat.controlled synthesizes each branch unitary with
subroutines/_dense_synthesis.py before adding the address controls.
Unrolling the branch through Qiskit 2.5.2's own synthesis instead would
snap a two-qubit unitary close to the identity to a local gate, with entry
errors measured up to 4.2e-5 for a branch on two system qubits.
Layered MPS preparation has no bound until an explicit circuit validation,
so its stage stays unknown. The construction tolerance of the LCHS Method
supplies the kernel and k-quadrature allowances of a provider pair with
bounds (half each for the default pair, a third each for Low-Somma
arXiv:2508.19238v2), and 0.1 times its value is the QSP synthesis
allowance or the per-node allowance of trotter_error_budgeted. Fixed-step
trotter, the Duhamel stage and the preparation stages receive no allowance
from it, and it is not a certificate for the total physical error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, floor, fsum, isfinite, log
from typing import Any

from nwqlib.serialization import FieldSerializedRecord

_LN2 = log(2.0)


def psd_recovery(shift: float, elapsed: float) -> float | None:
    """Return the PSD recovery exp(shift*elapsed), or None when binary64 cannot represent it.

    exp(-A*t) = exp(shift*t)*exp(-(A + shift*I)*t), so every application of
    a shifted LCHS construction is multiplied back by this factor. It
    overflows once shift*elapsed exceeds log(max float), about 709.78, even
    when the recovered physical vector itself is representable.
    """
    try:
        return exp(shift*elapsed)
    except OverflowError:
        return None


def psd_recovery_exponent(shift: float, final_time: float) -> int:
    """Return the power of two e that a recovery carries outside every array, or 0.

    While exp(shift*final_time) is representable, e = 0 and the arrays hold
    the full recovery. Otherwise e = floor(shift*final_time/ln 2), so
    exp(shift*final_time)*2**-e lies in [1, 2) up to rounding, and the
    factor 2**e joins the output's binary PhysicalScale recovery instead
    (problems.inputs.compose_recovery). final_time is the longest elapsed
    time of the Problem, so every application's part fits the same e.
    """
    if psd_recovery(shift, final_time) is not None:
        return 0
    return int(floor(shift*final_time/_LN2))


def psd_recovery_part(shift: float, elapsed: float, exponent: int) -> float:
    """Return exp(shift*elapsed)*2**-exponent, the part of the recovery applied to arrays.

    With exponent = 0 this is exactly exp(shift*elapsed). Otherwise the
    power of two is removed from the exponent before exp is evaluated. That
    reduction adds rounding of order u*exponent*ln 2, about
    u*shift*final_time, the order of the rounding that exp(shift*final_time)
    itself would carry from its argument. A part below the smallest normal
    binary64 number, about exp(-708), loses relative precision, and a part
    below the smallest subnormal one, about exp(-745), underflows to zero.
    Both happen only when this application's recovery is that much smaller
    than the recovery exp(shift*final_time) of the longest application.
    """
    if exponent == 0:
        return exp(shift*elapsed)
    return exp(shift*elapsed - exponent*_LN2)


def psd_recovery_scale(exponent: int):
    """Return 2**exponent as a PhysicalScale factor for compose_recovery, or None for exponent 0."""
    from nwqlib.problems.inputs import PhysicalScale

    if exponent == 0:
        return None
    return PhysicalScale(mantissa=0.5, exponent=exponent + 1, evidence="composed_floating_point_recovery")


def propagate_physical_error(output, components, *, radius, observable_norm=None):
    """Sum the complete physical L2 error once, then change output frame once.

    With delta the summed physical-vector error, r the observed physical norm
    and w the observable's Pauli coefficient 1-norm (at least ||O||), a
    Solution or physical StateVector keeps delta unchanged. The triangle
    inequality gives delta*(2*r + delta) for NormSquared,
    w*delta*(2*r + delta) for QuadraticForm, 2*delta/r for a unit StateVector
    and 4*w*delta/r for NormalizedExpectation. For the last two the code
    requires r > delta, which keeps the exact vector nonzero. NWQLib derives
    these relations (see Derivation below). Any missing, negative or
    nonfinite component yields None, never a partial sum.

    Derivation. Let x be the exact vector and y the computed one, with
    ||x - y|| <= delta and ||y|| = r, so ||x|| <= r + delta. Then
    abs(||x||**2 - ||y||**2) = abs(||x|| - ||y||)*(||x|| + ||y||) <= delta*(2r + delta)
    and abs(x†Ox - y†Oy) <= ||O||*(||x|| + ||y||)*||x - y||. For the unit
    outputs, the distance between x/||x|| and y/||y|| is at most
    2*||x - y||/||y||, and abs(u†Ou - v†Ov) <= 2*||O||*||u - v|| for unit
    vectors u and v.

    Args:
        output: Requested output record, which selects the frame change.
        components: Physical L2 stage bounds of one Solution, in its units.
        radius: Observed physical norm r of the computed vector, or None.
        observable_norm: Pauli coefficient 1-norm w of the observable, or None.

    Returns:
        The bound in the output's own frame and units, or None when any input
        needed for that frame is missing or the result is not finite.
    """
    from nwqlib.problems.records import (Solution, StateVector, NormSquared,
                                        QuadraticForm, NormalizedExpectation)

    values = tuple(components)
    if not values or any(value is None or not isfinite(value) or value < 0 for value in values):
        return None
    try:
        delta = fsum(values)
        if isinstance(output,Solution) or isinstance(output,StateVector) and output.normalization == "physical":
            return delta
        if radius is None or not isfinite(radius) or radius < 0:
            return None
        bound = None
        if isinstance(output,NormSquared):
            bound = delta*(2*radius+delta)
        elif isinstance(output,QuadraticForm):
            bound = None if observable_norm is None else observable_norm*delta*(2*radius+delta)
        elif radius > delta:
            if isinstance(output,StateVector):
                bound = 2*delta/radius
            elif isinstance(output,NormalizedExpectation) and observable_norm is not None:
                bound = 4*observable_norm*delta/radius
        return bound if bound is None or (isfinite(bound) and bound >= 0) else None
    except OverflowError:
        return None

_STAGE_ORDER = (
    "kernel_approximation",
    "k_quadrature",
    "trotter_synthesis",
    "duhamel_quadrature",
    "qsp_synthesis",
    "lcu_coefficient_preparation",
    "initial_state_preparation",
)

# Registered under ``psd_tolerance`` in docs/ENGINEERING_CONSTANTS.md.
# _numerical_psd_decision explains the value.
DEFAULT_PSD_TOLERANCE = 1.0e-12


@dataclass(frozen=True, kw_only=True)
class _NumericalPSDDecision:
    """One scale-aware numerical ruling for the LCHS PSD premise.

    Attributes:
        shift: s >= 0 added to L. The caller multiplies the propagator back by
            exp(s*t), since exp(-A*t) = exp(s*t)*exp(-(A + s*I)*t).
        tolerance_window: psd_tolerance*max(|lambda_min|, |lambda_max|), that
            is psd_tolerance*||L||_2, in the units of L (inverse time).
        premise_satisfied: Whether the shifted minimum eigenvalue lies at or
            above minus the window.
    """

    shift: float
    tolerance_window: float
    premise_satisfied: bool


def _numerical_psd_decision(
    *,
    lambda_min: float,
    lambda_max: float,
    psd_tolerance: float,
    make_l_psd: bool,
) -> _NumericalPSDDecision:
    """Return the shared pass/shift/raise decision for an LCHS Hermitian part.

    The window is psd_tolerance*max(|lambda_min|, |lambda_max|), which is
    psd_tolerance*||L||_2 for Hermitian L. A value inside the window is
    accepted unshifted, because roundoff can make an exactly PSD L look
    slightly negative. Below the window the shift s = -lambda_min + window
    leaves the shifted minimum at +window. Every consumer (decomposition,
    host and quantum selection) uses this one rule, so they cannot disagree
    about the LCHS premise.

    The window is relative to ||L||_2 because that is the scale of the
    rounding it absorbs. A backward-stable Hermitian eigensolver returns the
    eigenvalues of L + E with ||E||_2 <= p(n)*u*||L||_2, so each computed
    eigenvalue lies within that amount of an exact one (LAPACK Users' Guide,
    3rd ed., doi:10.1137/1.9780898719604, Sec. 4.7). The default 1e-12
    covers p(n) up to about 9000. The decision is then independent of the
    time unit, because replacing A by r*A and T by T/r, for r > 0, scales
    lambda_min, the window and the shift by r. An admitted unshifted violation
    lets ||exp(-A*T)||_2 reach at most
    exp(window*T) = exp(psd_tolerance*||L||_2*T), since ||exp(-A*T)|| <=
    ||exp(-L*T)|| (ACL arXiv:2312.03916v2, Lemma 21, Eq. (162)). An
    absolute term in the window would have units of inverse time, and for
    ||L||_2 much smaller than 1/T it would admit a genuinely negative
    lambda_min with |lambda_min|*T of order one. An exactly zero L has a
    zero window and lambda_min = 0, so it needs no shift.
    """

    if not isfinite(float(psd_tolerance)) or psd_tolerance <= 0.0:
        raise ValueError("psd_tolerance must be finite and positive")
    tolerance_window = float(
        psd_tolerance * max(abs(lambda_min), abs(lambda_max))
    )
    shift = 0.0
    if lambda_min < -tolerance_window:
        if not make_l_psd:
            raise ValueError(
                "L from the Cartesian decomposition has a negative eigenvalue "
                f"({lambda_min}) below the numerical PSD tolerance window "
                f"({tolerance_window})"
            )
        shift = float(-lambda_min + tolerance_window)
    return _NumericalPSDDecision(
        shift=shift,
        tolerance_window=tolerance_window,
        premise_satisfied=bool(lambda_min + shift >= -tolerance_window),
    )


@dataclass(frozen=True, kw_only=True)
class LCHSApplicationRecord(FieldSerializedRecord):
    """One physical solution-operator application in an LCHS solve.

    The homogeneous term is one application from time zero. With a constant
    source, each Duhamel node s_q adds one application from s_q to T.

    Attributes:
        start_time: Time at which this application's input enters.
        elapsed_time: Evolution time T - start_time of this application.
        weight: Physical input norm times the magnitude of its Duhamel weight
            (one for the initial state).
        recovery_scale: exp(shift*elapsed_time), the PSD compensation this
            application's propagator carries, or None when binary64 cannot
            represent it. The weighted stages are then unknown.
        synthesis_error_bound: Coefficient-weighted product-formula bound for
            this application before weight and recovery_scale, None when
            unavailable. Backends without product-formula synthesis record zero
            here and report their synthesis through other stages.
    """

    start_time: float
    elapsed_time: float
    weight: float
    recovery_scale: float | None
    synthesis_error_bound: float | None

def _build_lchs_application_record(
    quadrature: Mapping[str, Any],
    *,
    final_time: float,
    start_time: float,
    weight: float,
    input_norm: float,
    backend: str,
    synthesis_error_bound: float | None,
) -> LCHSApplicationRecord:
    """Build one application record from the shared quadrature frame."""

    elapsed_time = float(final_time - start_time)
    recorded_synthesis = 0.0
    if backend in ("trotter", "trotter_error_budgeted"):
        recorded_synthesis = (
            None if synthesis_error_bound is None else float(synthesis_error_bound)
        )
    return LCHSApplicationRecord(
        start_time=float(start_time),
        elapsed_time=elapsed_time,
        weight=float(weight * input_norm),
        recovery_scale=psd_recovery(float(quadrature["psd_shift"]), elapsed_time),
        synthesis_error_bound=recorded_synthesis,
    )


class _StageFailure(Exception):
    """A stage has no usable value. The message is the reported reason."""


def _stage_scalar(name: str, value: Any) -> float:
    """Return a finite nonnegative stage input, or raise with the reason code.

    None is ``missing_stage``, a nonfinite value ``nonfinite_stage``, and a
    non-numeric or negative value ``unusable_stage``.
    """
    if value is None:
        raise _StageFailure(f"missing_stage:{name}")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _StageFailure(f"unusable_stage:{name}") from None
    if not isfinite(scalar):
        raise _StageFailure(f"nonfinite_stage:{name}")
    if scalar < 0.0:
        raise _StageFailure(f"unusable_stage:{name}")
    return scalar


def _stage_product(name: str, *factors: Any) -> float:
    """Multiply validated factors, refusing overflow and underflow to zero.

    A product of positive factors that rounds to zero would report a
    mathematically positive bound as exact zero, so it is ``unusable_stage``.
    """
    values = tuple(_stage_scalar(name, value) for value in factors)
    result = 1.0
    for value in values:
        result *= value
        if not isfinite(result):
            raise _StageFailure(f"nonfinite_stage:{name}")
    if result == 0.0 and all(value > 0.0 for value in values):
        raise _StageFailure(f"unusable_stage:{name}")
    return float(result)


def _validate_applications(applications: Sequence[LCHSApplicationRecord]) -> None:
    """Reject an empty, nonfinite or sign-violating application population.

    Weighted stages multiply every application's weight and recovery scale,
    so one invalid application would make every weighted stage meaningless.
    """
    if not applications:
        raise _StageFailure("missing_stage:applications")
    for application in applications:
        if not isinstance(application, LCHSApplicationRecord):
            raise TypeError("applications must contain LCHSApplicationRecord values")
        if application.recovery_scale is None:
            raise _StageFailure("unrepresentable_psd_recovery:applications")
        values = (
            application.start_time,
            application.elapsed_time,
            application.weight,
            application.recovery_scale,
        )
        try:
            finite = all(isfinite(float(value)) for value in values)
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise _StageFailure("nonfinite_stage:applications")
        if (
            application.start_time < 0.0
            or application.elapsed_time < 0.0
            or application.weight < 0.0
            or application.recovery_scale <= 0.0
        ):
            raise _StageFailure("unusable_stage:applications")


def _weighted_raw_stage(
    name: str,
    raw_value: Any,
    applications: Sequence[LCHSApplicationRecord],
) -> float:
    """Return sum_a w_a*r_a*e for one operator-level kernel bound e.

    The grid is selected for the full time T. The tail bounds do not depend
    on time, and the quadrature conditions (the ATAP ISBN 978-1-61197-239-9
    ellipse bound and the Low-Somma arXiv:2508.19238v2 step condition) only
    relax as T*||L|| decreases, so the same grid keeps its recorded bounds
    for every shorter Duhamel elapsed time T-s.
    """
    value = _stage_scalar(name, raw_value)
    terms = [
        _stage_product(name, application.weight, application.recovery_scale, value)
        for application in applications
    ]
    total = float(fsum(terms))
    if not isfinite(total):
        raise _StageFailure(f"nonfinite_stage:{name}")
    return total


def _weighted_synthesis_stage(applications: Sequence[LCHSApplicationRecord]) -> float:
    """Return sum_a w_a*r_a*s_a with each application's own synthesis bound s_a.

    Unlike the kernel stages, s_a differs between applications because each
    Duhamel application evolves for its own elapsed time.
    """
    name = "trotter_synthesis"
    if any(application.synthesis_error_bound is None for application in applications):
        raise _StageFailure("missing_stage:trotter_synthesis")
    terms = [
        _stage_product(
            name,
            application.weight,
            application.recovery_scale,
            application.synthesis_error_bound,
        )
        for application in applications
    ]
    total = float(fsum(terms))
    if not isfinite(total):
        raise _StageFailure(f"nonfinite_stage:{name}")
    return total


def _stage_manifest(*, backend: str, inhomogeneous: bool, circuit: bool) -> tuple[str, ...]:
    """Return the error stages present on one realization path, in _STAGE_ORDER.

    The kernel stages are always present. Product-formula backends add
    trotter_synthesis, a constant source adds duhamel_quadrature, and a
    circuit adds both preparation stages and, for QSP, qsp_synthesis.
    dense_exact has no synthesis stage because its branches are exact
    exponentials and their controlled circuits equal them to rounding
    (module docstring).
    """
    present = {"kernel_approximation", "k_quadrature"}
    if backend in ("trotter", "trotter_error_budgeted"):
        present.add("trotter_synthesis")
    if inhomogeneous:
        present.add("duhamel_quadrature")
    if circuit:
        present.update(("lcu_coefficient_preparation", "initial_state_preparation"))
        if backend == "qsp_block_encoding":
            present.add("qsp_synthesis")
    return tuple(name for name in _STAGE_ORDER if name in present)


def _unbudgeted_stages(raw_record):
    """Map each stage outside an ``incomplete`` pair's budget to its reason.

    The Low-Somma pair bounds its kernel and quadrature by Theorems 2-3
    (arXiv:2508.19238v2) but does not budget the stages it lists. Each of
    those stages is reported with the reason
    ``unbudgeted_stage:PROFILE:STAGE`` instead of a value.
    Other pairs return an empty mapping.
    """
    if raw_record.get("solution_error_certificate_status") != "incomplete":
        return {}
    stages = raw_record.get("unbudgeted_error_stages")
    if not isinstance(stages, list) or not stages or not all(
        isinstance(stage, str) and stage in _STAGE_ORDER for stage in stages
    ):
        raise ValueError("incomplete solution error certificate requires registered unbudgeted_error_stages")
    selection = raw_record.get("kernel_parameter_selection")
    profile = selection.get("profile") if isinstance(selection, Mapping) else None
    if not isinstance(profile, str) or not profile:
        raise ValueError("incomplete solution error certificate requires a named parameter profile")
    return {name: f"unbudgeted_stage:{profile}:{name}" for name in stages}


_APPLICATION_RAW_KEYS = {"kernel_approximation": "approximate_lchs_error_bound", "k_quadrature": "quadrature_error_bound"}


def application_stage_bounds(raw_record, *, applications, backend, psd_premise_satisfied, unusable_stages=()):
    """Return the application-weighted physical L2 bound of each classical stage.

    Application a has weight w_a, its physical input norm times the magnitude
    of its Duhamel time-quadrature weight (one for the initial state), and PSD
    recovery scale r_a=exp(shift*elapsed_a). The kernel-approximation and k-quadrature stages are
    sum_a w_a*r_a*e with the recorded operator-level bound e, and
    trotter_synthesis is sum_a w_a*r_a*s_a with the application's recorded
    synthesis bound s_a. A failed PSD or application premise suppresses every
    stage. Otherwise a missing, nonfinite, unusable, underflowed or unbudgeted
    stage suppresses only its own value. Each stage maps to (value, None) or
    (None, reason). The stages are not summed into a total bound.
    """
    manifest = _stage_manifest(backend=backend, inhomogeneous=False, circuit=False)
    unusable = set(unusable_stages)
    if unusable.difference(manifest):
        raise ValueError(f"unusable_stages contains off-path stages: {sorted(unusable.difference(manifest))!r}")
    unbudgeted = _unbudgeted_stages(raw_record)
    try:
        if not psd_premise_satisfied:
            raise _StageFailure("uncertified_psd_premise")
        _validate_applications(applications)
    except _StageFailure as failure:
        return {name: (None, str(failure)) for name in manifest}
    stages = {}
    for name in manifest:
        try:
            if name in unbudgeted:
                raise _StageFailure(unbudgeted[name])
            if name in unusable:
                raise _StageFailure(f"unusable_stage:{name}")
            value = (_weighted_raw_stage(name, raw_record.get(_APPLICATION_RAW_KEYS[name]), applications)
                     if name in _APPLICATION_RAW_KEYS else _weighted_synthesis_stage(applications))
            stages[name] = value, None
        except _StageFailure as failure:
            stages[name] = None, str(failure)
    return stages


def _solution_stage_value(name, raw_record, *, applications, duhamel_quadrature_error_bound,
                          gamma, delta_lcu, input_preparation_output_error_bound,
                          compensated_recovery_error_bound):
    """Return the physical L2 bound of one circuit-path stage. Every stage value is computed here."""
    if name in _APPLICATION_RAW_KEYS:
        return _weighted_raw_stage(name, raw_record.get(_APPLICATION_RAW_KEYS[name]), applications)
    if name == "trotter_synthesis":
        return _weighted_synthesis_stage(applications)
    if name == "duhamel_quadrature":
        return _stage_scalar(name, duhamel_quadrature_error_bound)
    if name == "qsp_synthesis":
        return _stage_product(name, gamma, compensated_recovery_error_bound)
    if name == "lcu_coefficient_preparation":
        return _stage_product(name, 2.0, gamma, delta_lcu)
    return _stage_scalar(name, input_preparation_output_error_bound)


def circuit_stage_bounds(raw_record, *, applications, backend, inhomogeneous,
                         psd_premise_satisfied, gamma, delta_lcu,
                         input_preparation_output_error_bound, compensated_recovery_error_bound=None,
                         duhamel_quadrature_error_bound=None, unusable_stages=()):
    """Return the physical L2 bound of each stage on a selected circuit path.

    The kernel-approximation, k-quadrature and trotter_synthesis stages are weighted as
    in application_stage_bounds. With gamma, the physical recovery scale of the
    prepared coefficient state, lcu_coefficient_preparation is
    2*gamma*delta_lcu (PREP and inverse PREP each contribute one state error
    delta_lcu) and qsp_synthesis is gamma times the compensated QSP
    recovery bound. The initial-state preparation and Duhamel stages are the
    supplied bounds. A failed PSD or application premise suppresses only the
    application-weighted stages. This does not publish a combined certificate
    or perform a reference solve.
    """
    manifest = _stage_manifest(backend=backend, inhomogeneous=inhomogeneous, circuit=True)
    unusable = set(unusable_stages)
    if unusable.difference(manifest):
        raise ValueError("unusable circuit stages must belong to the selected path")
    unbudgeted = _unbudgeted_stages(raw_record)
    application_failure = None
    try:
        if not psd_premise_satisfied:
            raise _StageFailure("uncertified_psd_premise")
        _validate_applications(applications)
    except _StageFailure as failure:
        application_failure = str(failure)
    result = {}
    for name in manifest:
        try:
            if application_failure is not None and name in {"kernel_approximation", "k_quadrature", "trotter_synthesis"}:
                raise _StageFailure(application_failure)
            if name in unbudgeted:
                raise _StageFailure(unbudgeted[name])
            if name in unusable:
                raise _StageFailure(f"unusable_stage:{name}")
            value = _solution_stage_value(name, raw_record, applications=applications,
                duhamel_quadrature_error_bound=duhamel_quadrature_error_bound, gamma=gamma, delta_lcu=delta_lcu,
                input_preparation_output_error_bound=input_preparation_output_error_bound,
                compensated_recovery_error_bound=compensated_recovery_error_bound)
            result[name] = value,None
        except _StageFailure as failure:
            result[name] = None,str(failure)
    return result


__all__ = ["LCHSApplicationRecord"]
