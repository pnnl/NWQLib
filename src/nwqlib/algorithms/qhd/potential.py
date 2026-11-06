"""Finite real values of the QHD objective: one scalar, and the objective at one grid point."""

from __future__ import annotations

import math


def coerce_real_scalar(value: object, *, context: str) -> float:
    """Return ``value`` as a finite real scalar.

    Any nonzero imaginary part, however small, is rejected. A complex objective
    value would make the diagonal potential non-Hermitian, and dropping the
    imaginary part would change the objective.
    """

    scalar = complex(value)
    if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
        raise ValueError(f"{context} evaluated to a non-finite value: {value!r}")
    if scalar.imag != 0.0:
        raise ValueError(f"{context} evaluated to a complex value: {value!r}")
    return float(scalar.real)


def _evaluate_objective(evaluator, point, *, expression, support, name="QHD objective"):
    """Return the value of ``evaluator`` at one grid point as a finite real float.

    An ArithmeticError, ValueError, TypeError or NameError from the call or
    from the finite-real check is raised as a ValueError naming ``name``, the
    expression, the support and the point.
    """
    try:
        return coerce_real_scalar(evaluator(*point), context=name)
    except (ArithmeticError, ValueError, TypeError, NameError) as error:
        context = f"{name} {expression} on support {support} at grid point {point}"
        raise ValueError(f"{context} failed finite-real evaluation: {error}") from error
