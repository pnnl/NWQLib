"""Physical-vector sufficient statistics reduced into requested scalar quantities."""

from math import isfinite


def physical_scalar(kind, *, norm_squared, scale, numerator=None, numerator_frame="physical", numerator_unavailable=None, norm_unavailable=None):
    """Return a physical scalar and a precise unavailable reason when undefined.

    norm_squared is u†u, without postselection or quantum success probability.
    numerator is u†Ou in the same physical frame. The method supplies both
    values and owns approximation/scale evidence. No clipping or epsilon divide.
    """
    if norm_squared is not None and (not isfinite(norm_squared) or norm_squared < 0):
        raise ValueError("physical norm squared must be finite and nonnegative")
    if kind == "norm_squared":
        return norm_squared, None if norm_squared is not None else (norm_unavailable or "physical norm squared is not representable in binary64")
    if kind not in {"quadratic_form", "normalized_expectation"}:
        raise ValueError("requested quantity is not a supported physical scalar")
    if numerator_frame not in {"physical", "unit"}:
        raise ValueError("observable numerator frame must be physical or unit")
    if kind == "normalized_expectation" and (norm_squared == 0 or scale is not None and scale.mantissa == 0):
        return None, "normalized expectation is undefined for a zero physical vector"
    if numerator is None:
        return None, numerator_unavailable or "observable moment is unavailable in the selected vector frame"
    if not isfinite(numerator):
        raise ValueError("observable moment must be finite or explicitly unavailable")
    if kind == "quadratic_form":
        if numerator_frame != "physical":
            raise ValueError("quadratic form requires its physical numerator")
        return numerator, None
    if numerator_frame == "unit":
        return numerator, None
    if norm_squared is None:
        return None, "normalized expectation requires an available physical denominator or unit-frame moment"
    value = numerator / norm_squared
    if not isfinite(value):
        return None, "normalized expectation is not representable in binary64"
    return value, None


def _observable_moment(vector, observable, *, vector_exponent=0, recovery=None):
    """Reduce once, combining operator/vector scales before the requested float."""
    from math import frexp, ldexp
    from nwqlib.operators.inputs import _scaled_observable_moment

    moment, exponent = _scaled_observable_moment(observable, vector)
    if moment is None:
        return None, "scaled observable lost a nonzero component or reduction is unrepresentable"
    if moment == 0:
        return moment, None
    mantissa, shift = frexp(moment)
    exponent += shift + 2 * vector_exponent
    if recovery is not None:
        mantissa = (mantissa * recovery.mantissa) * recovery.mantissa
        exponent += 2 * recovery.exponent
    try:
        value = ldexp(mantissa, exponent)
    except OverflowError:
        value = None
    if value is None or not isfinite(value) or value == 0:
        return None, "nonzero observable moment is not representable in its requested frame"
    return value, None


def physical_vector_statistics(solution, *, observable=None, numerator_frame="physical", need_direction=False, recovery=None):
    """Statistics of solution or recovery*solution, with one observable action.

    Optional positive binary recovery never participates in unit normalization.
    Its physical norm/moment may be known when a recovered array is unavailable.
    Physical moments keep the input's power-of-two frame, not a unit-vector
    frame. Observable scaling is recovered only after that one scaled action.
    Dividing the vector by its power-of-two norm frame is exact unless a
    quotient falls below the smallest normal binary64 value 2**-1022, so
    entries smaller than the norm by more than about that factor can lose
    precision. Scaling the observable never silently removes stored entries.
    """
    from math import frexp
    import numpy as np
    from nwqlib._numerics import stable_vector_norm, _normalized_vector_with_scale
    from nwqlib.execution import ScalarValue
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery

    if numerator_frame not in {"physical", "unit"}:
        raise ValueError("observable numerator frame must be physical or unit")
    if recovery is not None and (not isinstance(recovery, PhysicalScale) or recovery.mantissa == 0):
        raise ValueError("physical statistics require a positive PhysicalScale recovery")

    need_direction = need_direction or (observable is not None and numerator_frame == "unit")
    norm = stable_vector_norm(solution)
    if not isfinite(norm):
        raise ValueError("realized vector norm is not finite in the established normalization kernel")
    direction = None
    if norm > 0 and (need_direction or norm < np.finfo(float).tiny):
        direction, binary_scale = _normalized_vector_with_scale(solution, norm)
    else:
        binary_scale = frexp(norm)
    scale = PhysicalScale(mantissa=binary_scale[0], exponent=binary_scale[1])
    vector_exponent = scale.exponent
    if recovery is not None and scale.mantissa != 0:
        combined = compose_recovery(scale, recovery)
        scale = PhysicalScale(mantissa=combined.mantissa, exponent=combined.exponent)
    norm_squared = scale.squared_as_float()
    statistics = [ScalarValue(label="norm_squared", value=norm_squared,
        unavailable=None if norm_squared is not None else "physical norm squared is not representable in binary64")]
    if observable is not None:
        # Exactly one selected observable application, part of this acquisition.
        frame = numerator_frame
        reason = "observable moment is unavailable for a zero unit vector"
        if frame == "physical":
            # Keep cancellation in the original power-of-two coordinate frame.
            vector = np.empty(solution.shape, dtype=complex)
            np.ldexp(solution.real, -vector_exponent, out=vector.real)
            np.ldexp(solution.imag, -vector_exponent, out=vector.imag)
        else:
            vector = direction
        numerator = None
        if norm == 0 and frame == "physical":
            numerator = 0.0
        elif vector is not None:
            numerator, reason = _observable_moment(
                vector, observable,
                vector_exponent=vector_exponent if frame == "physical" else 0,
                recovery=recovery if frame == "physical" else None,
            )
        statistics.append(ScalarValue(label="numerator", value=numerator, frame=frame,
            unavailable=None if numerator is not None else reason))
    return scale, direction, tuple(statistics)
