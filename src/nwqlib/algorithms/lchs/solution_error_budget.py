"""Internal solution-level error accounting for time-independent LCHS.

Two classes of error are kept apart. Discretization error separates the
exact propagator from the ideal finite sum sum_j c_j U_j. Its stages are the
kernel-integral error ("kernel_approximation"), the k-quadrature ("k_quadrature") and,
with a constant source, the source-time quadrature ("duhamel_quadrature").
Construction error separates that finite sum from the block the circuit
applies. Its stages are product-formula or QSP branch synthesis and coefficient or
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

Weights, recoveries, gamma and supplied bounds may be PhysicalScale values,
whose integer exponents can lie outside the binary64 range. Every factor
and term of a stage is combined with those exponents, and the completed
stage is converted to binary64 once. A positive stage above the binary64
range is unknown with the reason ``nonfinite_stage:<stage>``. A positive
stage below the normal range is rounded up to a binary64 number, at least
2**-1074, so it stays a positive upper bound. Only an exactly zero factor
gives an exact zero stage.

The sum bounds the distance between the exact solution of du/dt = -A u + b
and the output that the planned construction would give in exact
arithmetic without noise, through the stages listed above and only under
their assumptions (in particular the numerical PSD decision). It does not bound
binary64 rounding in classical exponentials, phases, angles and gate
synthesis, transpiler approximations, hardware noise or shot noise. The
controlled dense branches of dense_exact carry only rounding, because
qiskit_compat.controlled synthesizes each branch unitary with
subroutines/_dense_synthesis.py before adding the address controls.
Qiskit's own synthesis, used instead, can replace a two-qubit unitary close
to the identity by a local gate. Entry errors up to 4.2e-5 were measured
with Qiskit 2.5.2. The same behavior was reproduced with Qiskit 2.6.0,
including an entry error of about 4.2e-5 for exp(i*a*XX) at a = 4.2e-5.
Layered MPS preparation has no bound until an explicit circuit validation,
so its stage stays unknown. The construction tolerance of the LCHS Method
supplies the kernel and k-quadrature allowances of a kernel-quadrature pair
with bounds (half each for the default pair, a third each for Low-Somma
arXiv:2508.19238v2), and 0.1 times its value is the QSP synthesis
allowance or the per-node allowance of trotter_error_budgeted. Fixed-step
trotter, the Duhamel stage and the preparation stages receive no allowance
from it, and it is not a certificate for the total physical error. Fixed-step
trotter leaves trotter_synthesis unknown and reports the known part it can,
its Pauli pruning, as the separate listed component trotter_pruning
(pruning_stage), which enters no sum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, floor, frexp, fsum, inf, isfinite, ldexp, log, nextafter
from typing import Any

from nwqlib.serialization import FieldSerializedRecord

_LN2 = log(2.0)
_SMALLEST_NORMAL = 2.2250738585072014e-308


def signed_shift(conversion: Mapping[str, Any]) -> float:
    """Return the signed scalar shift s = psd_shift - scalar_decay of an LCHS conversion record.

    For a lower bound ell <= lambda_min(L),
    psd_shift = max(0, -ell) is the shift that makes L + s*I positive
    semidefinite and scalar_decay = max(0, ell) is known physical decay, so
    s = -ell and exp(-A*t) = exp(s*t)*exp(-(A + s*I)*t). Recovery and
    compensation use this signed s. Component error bounds may use the
    nonnegative psd_shift instead, since exp(s*t) <= exp(psd_shift*t).
    """
    return float(conversion["psd_shift"]) - float(conversion["scalar_decay"])


def psd_recovery(shift: float, elapsed: float) -> float | None:
    """Return the PSD recovery exp(shift*elapsed), or None when binary64 cannot represent it.

    exp(-A*t) = exp(shift*t)*exp(-(A + shift*I)*t), so every application of
    a shifted LCHS construction is multiplied back by this factor. It
    overflows once shift*elapsed exceeds log(max float), about 709.78, even
    when the recovered physical vector itself is representable. A negative
    signed shift (known decay) can instead make it smaller than the smallest
    normal binary64 number, which also returns None, because the value would
    have lost relative precision or become zero.
    """
    try:
        value = exp(shift*elapsed)
    except OverflowError:
        return None
    return value if isfinite(value) and value >= _SMALLEST_NORMAL else None


def binary_exponent(value: float) -> int:
    """Return floor(value/ln 2), the power of two that carries exp(value), or refuse a nonfinite value.

    A value or quotient outside binary64 has no representable recovery, so
    it raises ApplicabilityError, the range refusal of a legal input, before
    any floor or exp conversion.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError

    quotient = value/_LN2 if isfinite(value) else value
    if not isfinite(quotient):
        raise ApplicabilityError(f"the LCHS scalar compensation exp({value!r}) is outside the range LCHS represents: "
                                 "its exponent s*t/ln 2 is not a finite binary64 number")
    return floor(quotient)


def scaled_exp(value: float):
    """Return exp(value) as a PhysicalScale, or None when value or value/ln 2 is not finite.

    When exp(value) is a normal binary64 number, the scale holds exactly
    that number. Otherwise q = value/ln 2 is split into floor(q) and
    q - floor(q), which binary64 subtracts exactly, so exp(value) =
    2**(q - floor(q))*2**floor(q) with an integer exponent and no overflow
    or underflow. The quotient rounds with absolute error of order u*|q| in
    the exponent, a numerical estimate without an enclosure.
    """
    from nwqlib.problems.inputs import PhysicalScale

    if not isfinite(value):
        return None
    direct = psd_recovery(value, 1.0)
    if direct is not None:
        mantissa, exponent = frexp(direct)
    else:
        quotient = value/_LN2
        if not isfinite(quotient):
            return None
        whole = floor(quotient)
        mantissa, exponent = frexp(2.0**(quotient - whole))
        exponent += whole
    return PhysicalScale(mantissa=mantissa, exponent=exponent, evidence="composed_floating_point_recovery")


def admitted_exp(value: float):
    """Return exp(value) as a PhysicalScale (``scaled_exp``), or raise the range refusal of ``binary_exponent``."""
    scale = scaled_exp(value)
    if scale is None:
        binary_exponent(value)
    return scale


def extracted_phase(time: float, k: float, scalar_phase) -> complex:
    """Return exp(-i*time*(k*a + b)), the scalar phase a recipe takes out of its evolved children, or refuse.

    With children L - c_L*I and H - c_H*I, a = c_L + s and b = c_H
    (``selection.LCHSGenerator``). A finite phase argument is evaluated in
    binary64, with rounding that grows with |time*(k*a + b)| and no separate
    enclosure. A nonfinite argument raises ApplicabilityError before any
    complex exponential is formed.
    """
    from cmath import exp as complex_exp
    from nwqlib.algorithms.protocol import ApplicabilityError

    dissipative, hamiltonian = scalar_phase
    argument = time*(float(k)*dissipative + hamiltonian)
    if not isfinite(argument):
        raise ApplicabilityError(f"the extracted LCHS scalar phase argument t*(k*a + b) = {argument!r} at t={time!r}, "
                                 f"k={float(k)!r} is not a finite binary64 number")
    return complex_exp(-1j*argument)


def unit_angle(angle: float) -> float:
    """Return the angle of exp(i*angle) in [-pi, pi], reduced through its unit phase.

    An angle already in [-pi, pi] is returned unchanged. A larger finite
    angle is reduced as atan2(sin(angle), cos(angle)): libm reduces the
    argument against pi itself, while ``angle % (2*pi)`` reduces it against
    the binary64 value of 2*pi, which is 2.4e-16 too small, and so is off
    by about 3.9e-17*|angle| radians, several turns near 2**60. A bounded
    phase added afterwards, such as a coefficient argument, keeps its value,
    whereas added to the unreduced angle it would round at that angle's
    spacing. The result carries the ordinary rounding of the reduction,
    and the finite input angle carries its own rounding; neither is an
    enclosure.
    """
    from math import atan2, cos, pi, sin

    return angle if -pi <= angle <= pi else atan2(sin(angle), cos(angle))


def scaled_sum(scales):
    """Return the sum of nonnegative PhysicalScale values as a PhysicalScale, without a binary64 conversion.

    Each term is ldexp(mantissa, exponent - top) relative to the largest
    exponent top, and the binary64 sum of those relative values is composed
    with 2**top. A term more than 1100 binary orders below the largest is
    below the rounding of that sum and is added as zero. A zero total is the
    zero scale.
    """
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery

    terms = [scale for scale in scales if scale.mantissa != 0]
    if not terms:
        return PhysicalScale(mantissa=0.0, exponent=0)
    top = max(scale.exponent for scale in terms)
    relative = fsum(ldexp(scale.mantissa, max(scale.exponent - top, -1100)) for scale in terms)
    return compose_recovery(relative, PhysicalScale(mantissa=0.5, exponent=top + 1))


def psd_recovery_exponent(shift: float, final_time: float) -> int:
    """Return the power of two e that a recovery carries outside every array, or 0.

    While exp(shift*final_time) is a normal binary64 number, e = 0 and the
    arrays hold the full recovery. Otherwise e = floor(shift*final_time/ln 2),
    so exp(shift*final_time)*2**-e lies in [1, 2) up to rounding, and the
    factor 2**e joins the output's binary PhysicalScale recovery instead
    (problems.inputs.compose_recovery). For a nonnegative shift, final_time
    is the longest elapsed time of the Problem, so every application's part
    fits the same e. A constant-source layout with known decay chooses its
    common scale from all applications instead
    (source_selection.select_source_branches). A nonfinite
    shift*final_time raises ApplicabilityError (``binary_exponent``).
    """
    if psd_recovery(shift, final_time) is not None:
        return 0
    return binary_exponent(shift*final_time)


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
    and 2*w*delta/r for NormalizedExpectation. For the last two the code
    requires finite r > delta >= 0, which keeps both vectors nonzero. The
    observable norm w must be finite and nonnegative. NWQLib derives these
    relations below. Any missing, negative or nonfinite component, an empty
    component set or an overflowing sum yields None, never a partial sum.

    Derivation. Let x be the exact vector and y the computed one, with
    ||x - y|| <= delta and ||y|| = r, so ||x|| <= r + delta. Then
    abs(||x||**2 - ||y||**2) = abs(||x|| - ||y||)*(||x|| + ||y||) <= delta*(2r + delta)
    and abs(x†Ox - y†Oy) <= ||O||*(||x|| + ||y||)*||x - y||. For the unit
    StateVector output, the phase-sensitive distance between x/||x|| and
    y/||y|| is at most 2*||x - y||/||y||.

    For NormalizedExpectation, put u = x/||x||, v = y/r and c = u†v.
    Projection onto the complex line spanned by u gives
    ||x-y||**2 = abs(||x||-r*c)**2 + r**2*(1-abs(c)**2), so
    s = sqrt(1-abs(c)**2) <= delta/r. The Hermitian matrix D = uu†-vv†
    has rank at most two, trace zero and tr(D**2) = 2*s**2. Its possibly
    nonzero eigenvalues are +s and -s, hence its trace norm is 2*s.
    Trace duality then gives abs(u†Ou-v†Ov) = abs(tr(O*D)) <=
    ||O||*||D||_1 <= 2*w*delta/r. The public observable is Hermitian, and
    the vectors may be complex. The factor 2 is sharp: for 0<t<1, take
    y=(1,0), x=(1-t**2, i*t*sqrt(1-t**2)) and O=(uu†-yy†)/t. Then
    r=w=1, ||x-y||=t and the expectation difference is 2*t.

    These are exact-arithmetic relations evaluated with ordinary scalar
    rounding. NormalizedExpectation reuses compose_recovery to avoid
    intermediate overflow or underflow in the positive product. A true zero
    delta or w gives zero, and a positive final bound outside binary64 gives None.
    This evaluation is not a fully outward-rounded certificate. No vector,
    observable norm solve or spectral enclosure is computed here.

    Args:
        output: Requested output record, which selects the frame change.
        components: Physical L2 stage bounds of one Solution, in its units.
        radius: Observed physical norm r of the computed vector, or None.
        observable_norm: Pauli coefficient 1-norm w of the observable, or None.

    Returns:
        The bound in the output's own frame and units, or None when any input
        needed for that frame is missing or the result is not finite. For
        NormalizedExpectation, a positive result underflowing to zero is None.
    """
    from nwqlib.problems.records import (Solution, StateVector, NormSquared,
                                        QuadraticForm, NormalizedExpectation)
    from nwqlib.problems.inputs import compose_recovery

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
                if isfinite(observable_norm) and observable_norm >= 0:
                    bound = (0.0 if delta == 0 or observable_norm == 0 else
                             compose_recovery(2.0, observable_norm, (delta, radius)).as_float())
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
    """One scale-aware numerical decision on the LCHS PSD assumption.

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
    leaves the shifted minimum at +window. Every caller (decomposition,
    classical and quantum planning) uses this one rule, so they cannot
    disagree about the LCHS PSD assumption.

    The window is relative to ||L||_2 because that is the scale of the
    rounding it absorbs. A backward-stable Hermitian eigensolver returns the
    eigenvalues of L + E with ||E||_2 <= p(n)*u*||L||_2, so each computed
    eigenvalue lies within that amount of an exact one (LAPACK Users' Guide,
    3rd ed., doi:10.1137/1.9780898719604, Sec. 4.7). The default 1e-12
    covers p(n) up to about 9000. The decision is then independent of the
    time unit, because replacing A by r*A and T by T/r, for r > 0, scales
    lambda_min, the window and the shift by r. An accepted unshifted violation
    lets ||exp(-A*T)||_2 reach at most
    exp(window*T) = exp(psd_tolerance*||L||_2*T), since ||exp(-A*T)|| <=
    ||exp(-L*T)|| (ACL arXiv:2312.03916v2, Lemma 21, Eq. (162)). An
    absolute term in the window would have units of inverse time, and for
    ||L||_2 much smaller than 1/T it would accept a genuinely negative
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
            from nwqlib.algorithms.protocol import ApplicabilityError

            raise ApplicabilityError(
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
            (one for the initial state), a float or a PhysicalScale.
        recovery_scale: exp(shift*elapsed_time), the PSD compensation this
            application's propagator carries, as a float or a PhysicalScale,
            or None when shift*elapsed_time is not finite. The weighted
            stages are then unknown.
        synthesis_error_bound: Coefficient-weighted product-formula bound
            sum_j |c_j|*e_j for this application before weight and
            recovery_scale, a float or a PhysicalScale, None when
            unavailable. Backends without product-formula synthesis record zero
            here and report their synthesis through other stages.
    """

    start_time: float
    elapsed_time: float
    weight: Any
    recovery_scale: Any
    synthesis_error_bound: Any

def _build_lchs_application_record(
    quadrature: Mapping[str, Any],
    *,
    final_time: float,
    start_time: float,
    weight: Any,
    input_norm: Any,
    backend: str,
    synthesis_error_bound: Any,
) -> LCHSApplicationRecord:
    """Build one application record from the shared quadrature frame.

    ``weight``, the magnitude of the quadrature weight, and ``input_norm``
    are nonnegative floats or PhysicalScale values. Their product is
    composed as a PhysicalScale (``_stage_product``), so a product outside
    binary64 still reaches the stage products intact. The recovery
    exp(psd_shift*elapsed) is a PhysicalScale (``scaled_exp``), equal to
    the binary64 value whenever that value is normal. A product-formula
    ``synthesis_error_bound`` is kept as given, a float or the PhysicalScale
    of ``_trotter_budget_quadrature_terms``, so its positive value reaches
    ``_weighted_synthesis_stage`` without a binary64 conversion.
    """
    elapsed_time = float(final_time - start_time)
    recorded_synthesis = 0.0
    if backend in ("trotter", "trotter_error_budgeted"):
        recorded_synthesis = synthesis_error_bound
    recorded_weight = _stage_product("applications", weight, input_norm)
    return LCHSApplicationRecord(
        start_time=float(start_time),
        elapsed_time=elapsed_time,
        weight=recorded_weight,
        recovery_scale=scaled_exp(float(quadrature["psd_shift"]) * elapsed_time),
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


def _stage_factor(name: str, value: Any):
    """Return a nonnegative stage factor as a PhysicalScale, or raise with the reason code.

    A PhysicalScale is used as it is. A real value passes the checks of
    ``_stage_scalar`` and becomes its exact binary mantissa and exponent.
    """
    from nwqlib.problems.inputs import PhysicalScale

    if isinstance(value, PhysicalScale):
        return value
    mantissa, exponent = frexp(_stage_scalar(name, value))
    return PhysicalScale(mantissa=mantissa, exponent=exponent if mantissa else 0)


def _stage_product(name: str, *factors: Any):
    """Multiply validated factors with integer exponents and return the product as a PhysicalScale.

    No binary64 overflow or underflow can occur here. While every partial
    product is a normal number, the mantissa products round exactly as the
    binary64 products of the factors would. A zero factor gives the zero
    scale, an exact zero.
    """
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery

    scales = tuple(_stage_factor(name, value) for value in factors)
    if any(scale.mantissa == 0 for scale in scales):
        return PhysicalScale(mantissa=0.0, exponent=0)
    return compose_recovery(*scales)


def application_weight(weight: float, input_norm) -> float | None:
    """Return the descriptive application weight |weight|*input_norm in binary64, or None outside binary64.

    The product is composed with integer exponents (``_stage_product``), so
    it is exactly zero only for a zero factor, a positive subnormal value is
    returned as it is, and a positive product below or above the binary64
    range is None, never 0.0 or inf. The host's ``weight`` binding records
    this value only when it is not None, and
    selected_grid.host_applications forms the same value from the Plan's
    saved weight and input PhysicalScale. Physical stages use the two
    factors themselves, not this value.
    """
    return _stage_product("applications", abs(weight), input_norm).as_float()


def _stage_value(name: str, scale) -> float:
    """Convert a completed physical stage bound to binary64, once.

    Only an exactly zero scale gives 0.0. A positive bound above the
    binary64 range raises ``nonfinite_stage:<name>``. Below the normal range
    ldexp rounds to the nearest subnormal number or to zero, so a rounded
    value is replaced by the next binary64 number above it. That keeps an
    upper bound that is never zero, at least 2**-1074.
    """
    if scale.mantissa == 0:
        return 0.0
    value = scale.as_float()
    if value is None and scale.exponent > 0:
        raise _StageFailure(f"nonfinite_stage:{name}")
    value = value or 0.0
    # Scaling back by 2**-exponent is exact, so it recovers the mantissa
    # exactly when ldexp did not round.
    if ldexp(value, -scale.exponent) != scale.mantissa:
        value = nextafter(value, inf)
    return value


def _validate_applications(applications: Sequence[LCHSApplicationRecord]) -> None:
    """Reject a missing application set or a PSD recovery outside binary64.

    Weighted stages multiply every application's weight and recovery scale.
    _build_lchs_application_record forms the weight as a PhysicalScale and
    the recovery exp(shift*elapsed) with scaled_exp, which is None when the
    exponent itself is not finite, so that is the one recovery a stage
    cannot use.
    """
    if not applications:
        raise _StageFailure("missing_stage:applications")
    if any(application.recovery_scale is None for application in applications):
        raise _StageFailure("unrepresentable_psd_recovery:applications")


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
    for every shorter Duhamel elapsed time T-s. The terms are added with a
    common exponent (``scaled_sum``) and the sum is converted once, so a
    sum of representable terms that exceeds binary64 is ``nonfinite_stage``.
    """
    value = _stage_scalar(name, raw_value)
    return _stage_value(name, scaled_sum([
        _stage_product(name, application.weight, application.recovery_scale, value)
        for application in applications
    ]))


def _weighted_synthesis_stage(applications: Sequence[LCHSApplicationRecord]) -> float:
    """Return sum_a w_a*r_a*s_a with each application's own synthesis bound s_a.

    Unlike the kernel stages, s_a differs between applications because each
    Duhamel application evolves for its own elapsed time. The sum is formed
    as in ``_weighted_raw_stage``.
    """
    name = "trotter_synthesis"
    if any(application.synthesis_error_bound is None for application in applications):
        raise _StageFailure("missing_stage:trotter_synthesis")
    return _stage_value(name, scaled_sum([
        _stage_product(
            name,
            application.weight,
            application.recovery_scale,
            application.synthesis_error_bound,
        )
        for application in applications
    ]))


def pruning_stage(applications: Sequence[LCHSApplicationRecord], pruning_bounds, *, psd_premise_satisfied):
    """Return (value, None) or (None, reason) for sum_a w_a*r_a*p_a, the fixed-step pruning component.

    p_a is the application's operator bound
    (time_independent_terms._trotter_pruning_bound), weighted once like the
    synthesis stage. It is a known part of the error that a fixed step count
    leaves unknown, so it is outside the stage manifest. A later fixed-step
    certificate contains it inside trotter_synthesis, so analysis and
    refinement never add it to the total (analysis.adds_to_total). The
    premise and application failures suppress it as they do the other stages.
    """
    name = "trotter_pruning"
    try:
        if not psd_premise_satisfied:
            raise _StageFailure("uncertified_psd_premise")
        _validate_applications(applications)
        return _stage_value(name, scaled_sum([
            _stage_product(name, application.weight, application.recovery_scale, bound)
            for application, bound in zip(applications, pruning_bounds, strict=True)
        ])), None
    except _StageFailure as failure:
        return None, str(failure)


def _stage_manifest(*, backend: str, inhomogeneous: bool, circuit: bool) -> tuple[str, ...]:
    """Return the error stages present on one construction path, in _STAGE_ORDER.

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


_APPLICATION_RAW_KEYS = {"kernel_approximation": "approximate_lchs_error_bound", "k_quadrature": "quadrature_error_bound"}


def application_stage_bounds(raw_record, *, applications, backend, psd_premise_satisfied, unusable_stages=()):
    """Return the application-weighted physical L2 bound of each classical stage.

    Application a has weight w_a, its physical input norm times the magnitude
    of its Duhamel time-quadrature weight (one for the initial state), and PSD
    recovery scale r_a=exp(shift*elapsed_a). The kernel-approximation and k-quadrature stages are
    sum_a w_a*r_a*e with the recorded operator-level bound e, and
    trotter_synthesis is sum_a w_a*r_a*s_a with the application's recorded
    synthesis bound s_a. Each sum is formed with integer exponents and
    converted to binary64 once (module docstring), so a large recovery and a
    small weight combine before either is converted. A failed PSD or
    application assumption suppresses every stage. Otherwise a missing,
    nonfinite or unusable stage suppresses only its own value. A pair's
    ``incomplete`` certificate status says what its split of the tolerance
    between kernel and k quadrature covers, and changes no stage
    (``circuit_stage_bounds``). Each stage maps to
    (value, None) or (None, reason). The stages are not summed into a total
    bound.
    """
    manifest = _stage_manifest(backend=backend, inhomogeneous=False, circuit=False)
    unusable = set(unusable_stages)
    if unusable.difference(manifest):
        raise ValueError(f"unusable_stages contains off-path stages: {sorted(unusable.difference(manifest))!r}")
    try:
        if not psd_premise_satisfied:
            raise _StageFailure("uncertified_psd_premise")
        _validate_applications(applications)
    except _StageFailure as failure:
        return {name: (None, str(failure)) for name in manifest}
    stages = {}
    for name in manifest:
        try:
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
        return _stage_value(name, _stage_factor(name, duhamel_quadrature_error_bound))
    if name == "qsp_synthesis":
        return _stage_value(name, _stage_product(name, gamma, compensated_recovery_error_bound))
    if name == "lcu_coefficient_preparation":
        return _stage_value(name, _stage_product(name, 2.0, gamma, delta_lcu))
    return _stage_value(name, _stage_factor(name, input_preparation_output_error_bound))


def circuit_stage_bounds(raw_record, *, applications, backend, inhomogeneous,
                         psd_premise_satisfied, gamma, delta_lcu,
                         input_preparation_output_error_bound, compensated_recovery_error_bound=None,
                         duhamel_quadrature_error_bound=None, unusable_stages=()):
    """Return the physical L2 bound of each stage on a planned circuit path.

    The kernel-approximation, k-quadrature and trotter_synthesis stages are weighted as
    in application_stage_bounds. With gamma, the physical recovery scale of the
    prepared coefficient state, lcu_coefficient_preparation is
    2*gamma*delta_lcu (PREP and inverse PREP each contribute one state error
    delta_lcu) and qsp_synthesis is gamma times the compensated QSP
    recovery bound. The initial-state preparation and Duhamel stages are the
    supplied bounds. gamma and the supplied bounds may be PhysicalScale
    values, combined before the one binary64 conversion of each stage (module
    docstring). A failed PSD or application assumption suppresses only the
    application-weighted stages. A missing, nonfinite or unusable stage
    suppresses only its own value. This does not report a combined certificate
    or perform a reference solve.

    A pair's ``incomplete`` certificate status and its
    ``unbudgeted_error_stages`` say that its split of the tolerance between
    kernel and k quadrature does not cover those stages, as for the Low-Somma
    pair (arXiv:2508.19238v2, Theorems 2-3). A route can still give such a stage
    its own allowance (module docstring). They do not bear on a bound that the
    planned route supplies for such a stage, so they suppress no value here. An
    empty tuple does not certify that every stage has an allowance: the default
    pair's is empty, yet its Duhamel stage receives none (module docstring).

    The relations above are the triangle inequality on the finite sum for any
    kernel: with a branch error at most e_aj, the coefficient-weighted synthesis
    bound of application a is s_a = sum_j |c_j|*e_aj, already weighted by the raw
    coefficients. A zero preparation stage is the ideal direct construction, which
    claims nothing about native floating-point or execution error.
    """
    manifest = _stage_manifest(backend=backend, inhomogeneous=inhomogeneous, circuit=True)
    unusable = set(unusable_stages)
    if unusable.difference(manifest):
        raise ValueError("unusable circuit stages must belong to the selected path")
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
