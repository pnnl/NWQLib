"""Small numerical predicates shared by scientific subroutines."""

from __future__ import annotations

from math import expm1, frexp, fsum, isfinite, ldexp, log1p
from numbers import Real
from typing import Any, Iterable

import numpy as np
from nwqlib._validation import finite_real
from scipy.linalg.blas import dnrm2, dznrm2


def normalized_matrix(matrix, alpha):
    """Divide A by positive real alpha without overflowing a complex reciprocal."""
    alpha = finite_real(alpha, "matrix normalization")
    if alpha <= 0:
        raise ValueError("matrix normalization must be positive")
    array = np.asarray(matrix)
    if alpha == 1.0:
        return array
    if np.iscomplexobj(array):
        result = np.empty(array.shape, dtype=np.complex128)
        np.divide(array.real, alpha, out=result.real)
        np.divide(array.imag, alpha, out=result.imag)
        return result
    return np.divide(array, alpha)


def binary_scaled_matrix(matrix):
    """Return (A / 2**e, e), using a representable binary scale even at extremes.

    Real component maxima avoid an overflowing complex modulus. Scaling
    improves eigensolver range; it does not certify numerical singular values.
    """
    array = np.asarray(matrix)
    largest = max(float(np.max(np.abs(array.real))), float(np.max(np.abs(array.imag))))
    if not np.isfinite(largest) or largest == 0:
        raise ValueError("spectral input must be finite and nonzero")
    exponent = frexp(largest)[1] - 1
    return normalized_matrix(array, ldexp(1.0, exponent)), exponent


def stable_complex_sum(real_parts: Iterable[float], imag_parts: Iterable[float]) -> complex:
    """Sum binary64 components with fsum, rejecting overflow and nonfinite sums.

    Separate iterables permit packed coefficient buckets without copying them.
    This preserves fsum's cancellation/subnormal behavior and its intermediate
    overflow rejection; it is not arbitrary-precision real arithmetic.
    """
    try:
        result = complex(fsum(real_parts), fsum(imag_parts))
    except (OverflowError, ValueError):
        raise ValueError("coefficient accumulation overflowed or is nonfinite; coefficients must be finite") from None
    if not (isfinite(result.real) and isfinite(result.imag)):
        raise ValueError("coefficient accumulation overflowed or is nonfinite; coefficients must be finite")
    return result


def componentwise_exact_zero(value: Any) -> bool:
    """Return whether every stored real and imaginary component is exactly zero."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        return bool(np.all(array.real == 0.0) and np.all(array.imag == 0.0))
    return bool(np.all(array == 0.0))


def stable_vector_norm(value: Any) -> float:
    """Return a scaling-safe Euclidean norm for a real or complex vector."""

    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    # BLAS NRM2 scales its sum of squares, including partial squared underflow.
    kernel = dznrm2 if np.iscomplexobj(array) else dnrm2
    return float(kernel(array.reshape(-1)))


def _normalized_vector_with_scale(
    vector: np.ndarray, norm: float,
) -> tuple[np.ndarray, tuple[float, int]]:
    """Couple a normalized direction to its norm before subnormal rounding.

    A power-of-two change of frame is exact for the small input components.
    The scaled BLAS norm therefore keeps information lost when a subnormal
    norm is rounded to float64. Both direction and scale use that norm.
    """
    state = np.empty(vector.shape, dtype=complex)
    exponent = 0
    if norm < np.finfo(float).tiny:
        exponent = frexp(norm)[1]
        np.ldexp(vector.real, -exponent, out=state.real)
        np.ldexp(vector.imag, -exponent, out=state.imag)
        norm = float(dznrm2(state))
        np.divide(state.real, norm, out=state.real)
        np.divide(state.imag, norm, out=state.imag)
    else:
        np.divide(vector.real, norm, out=state.real)
        np.divide(vector.imag, norm, out=state.imag)
    mantissa, adjustment = frexp(norm)
    return state, (mantissa, exponent + adjustment)


def normalized_vector(vector: np.ndarray, norm: float) -> np.ndarray:
    """Normalize a nonzero vector using its already computed float norm."""
    return _normalized_vector_with_scale(vector, norm)[0]


def normalize_state_vector(vector: Any) -> tuple[np.ndarray, float]:
    """Return direction and rounded float norm for scalar numerical consumers."""
    normalized, norm, _ = normalize_state_vector_with_scale(vector)
    return normalized, norm


def normalize_state_vector_with_scale(
    vector: Any,
) -> tuple[np.ndarray, float, tuple[float, int]]:
    """Return direction, rounded float norm, and scaled physical norm.

    Args:
        vector: Input state amplitudes.

    Returns:
        ``(normalized_vector, rounded_float_norm, (mantissa, exponent))``.
        The last pair keeps the numerical norm as mantissa * 2**exponent,
        including subnormal inputs before rounding to a float loses scale.

    Raises:
        ValueError: If the vector is not one-dimensional, has non-power-of-two
            length, or has zero norm.
    """

    array = np.asarray(vector, dtype=complex)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError("state vector must be one-dimensional")
    if not (array.shape[0] > 0 and (array.shape[0] & (array.shape[0] - 1)) == 0):
        raise ValueError("state vector length must be a power of two")
    return normalize_physical_vector_with_scale(array)


def normalize_physical_vector_with_scale(vector):
    """Apply the established norm/direction kernel without a quantum width rule."""
    array = np.asarray(vector, dtype=complex)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("physical vector must have a positive dimension")
    if componentwise_exact_zero(array):
        raise ValueError("state vector must have nonzero norm")
    norm = stable_vector_norm(array)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("state vector must have a finite nonzero norm")
    normalized, scale = _normalized_vector_with_scale(array, norm)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("state-vector normalization produced nonfinite amplitudes")
    return normalized, norm, scale


def _divided_by_largest_modulus(vector: np.ndarray) -> np.ndarray:
    """Return a finite complex vector divided by its largest entry modulus.

    A largest modulus in the normal binary64 range is divided directly.
    Otherwise the vector is first rescaled by an exact power of two. A
    subnormal largest modulus would overflow the reciprocal that NumPy's
    complex division by a real uses, so the vector is scaled up by 2**64,
    which is exact. A modulus above the binary64 maximum, possible although
    both parts are finite (for example ``1.5e308 + 1.5e308j``), is scaled
    down by 4. That is exact for every part of at least 2**-1020, and
    smaller parts are below 2**-2044 of the largest modulus, so the division
    that follows makes them zero or subnormal in either case.

    Raises:
        ValueError: If every entry is zero.
    """
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError("fidelity requires nonzero vectors")
    if not np.finfo(float).tiny <= scale <= np.finfo(float).max:
        exponent = 64 if scale < np.finfo(float).tiny else -2
        scaled = np.empty_like(vector)
        np.ldexp(vector.real, exponent, out=scaled.real)
        np.ldexp(vector.imag, exponent, out=scaled.imag)
        vector = scaled
        scale = float(np.max(np.abs(vector)))
    return vector / scale


def normalized_fidelity_with_window(left: Any, right: Any) -> tuple[float, float]:
    """Return the fidelity of two complex vectors and its roundoff window.

    The fidelity ``|<x,y>|**2 / (<x,x> <y,y>)`` is evaluated in binary64
    after each vector is divided by its largest entry modulus
    (``_divided_by_largest_modulus``), so each squared norm lies between
    about 1 and D and cannot underflow or overflow. The result does not
    depend on how accurately the inputs were normalized. The exact value for
    the two scaled vectors lies in [0, 1] by the Cauchy-Schwarz inequality,
    which also holds for the rounded scaled entries, so only the rounding of
    the remaining evaluation can move the computed value outside [0, 1]. The
    computed value is never negative, and ``window`` bounds how far that
    rounding can push it above one.

    The window is NWQLib's derivation in the standard model
    ``fl(a op b) = (a op b)(1 + delta)``, ``|delta| <= u = 2**-53``, with
    ``gamma_n = n u / (1 - n u)``. Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed. (SIAM, 2002, doi:10.1137/1.9780898718027),
    treats this model and
    ``gamma_n`` in Secs. 2.2 and 3.4, inner products in Sec. 3.1 and complex
    arithmetic in Sec. 3.6.

    1. The real and imaginary parts of ``<x,y>`` are real sums of 2D
       products. In any summation order, with or without fused multiply-add,
       each product passes through at most 2D roundings, so each part errs
       by at most ``gamma_2D`` times the sum of its product magnitudes. Each
       of those sums is at most ``sum_i |x_i| |y_i| <= ||x|| ||y||``, so the
       computed overlap differs from ``<x,y>`` by at most
       ``sqrt(2) gamma_2D ||x|| ||y||``. Each squared norm is a sum of 2D
       nonnegative squares and is computed no smaller than
       ``1 - gamma_2D`` times its exact value.
    2. ``abs()`` of the computed overlap calls the C ``hypot``, and ``** 2``
       on the resulting NumPy float64 calls the C ``pow``, which is not
       always correctly rounded. Each adds a factor of at most
       ``1 + gamma_2`` when it errs by at most one ulp, as the rounded
       ``sqrt(a**2 + b**2)`` and ``a * a`` also do. The product of the two
       norms and the quotient add ``1/(1 - u)`` and ``1 + u``. With
       ``|<x,y>| <= ||x|| ||y||`` the computed fidelity is at most
       ``B**2 (1 + u) / ((1 - gamma_2D)**2 (1 - u))`` with
       ``B = (1 + sqrt(2) gamma_2D)(1 + gamma_2)(1 + gamma_2)**(1/2)``.
    3. ``c gamma_n <= gamma_(cn)`` for ``c >= 1`` and
       ``(1 + gamma_j)(1 + gamma_k) <= 1 + gamma_(j+k)`` give
       ``B <= 1 + gamma_(3D+4)``. The code uses ``g = gamma_(4D+4)``, so
       ``window = ((1 + g)/(1 - g))**2 (1 + u)/(1 - u) - 1`` bounds the
       excess of the computed fidelity over one.

    In ``4D+4``, the ``+4`` pays for the two ``gamma_2`` factors of abs()
    and squaring. Relative to ``||x|| ||y||``, which is at least about 1,
    the remaining D units of u leave a margin of about ``D u``, far above
    the error of at most about ``D * 2**-1074`` that gradual underflow of
    products of tiny scaled entries can add. The window covers evaluation
    roundoff of the supplied vectors only, not the accuracy of whatever
    produced them.

    Args:
        left: Complex vector x with D entries, finite and not all zero.
        right: Complex vector y with the same shape, finite and not all zero.

    Returns:
        The computed fidelity, a float in ``[0, 1 + window]``, and
        ``window``, both dimensionless.

    Raises:
        ValueError: If the vectors differ in shape or have a nonfinite
            entry, if either is zero, or if ``(4D+4) u >= 1/2``, where the
            model gives no bound. The final check, a fidelity that is not
            finite or exceeds one by more than ``window``, fires only if the
            arithmetic breaks the rounding model above.
    """
    x = np.asarray(left, dtype=complex)
    y = np.asarray(right, dtype=complex)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("fidelity requires matching finite vectors")
    x = _divided_by_largest_modulus(x)
    y = _divided_by_largest_modulus(y)
    fidelity = float(abs(np.vdot(x, y)) ** 2 / (np.vdot(x, x).real * np.vdot(y, y).real))
    u = np.finfo(float).eps / 2
    count = 4 * x.size + 4
    if count * u >= 0.5:
        raise ValueError("fidelity reduction exceeds the admitted roundoff model")
    g = count * u / (1 - count * u)
    window = expm1(2 * (log1p(g) - log1p(-g)) + log1p(u) - log1p(-u))
    if not isfinite(fidelity) or 1.0 - fidelity < -window:
        raise ValueError(
            "normalized fidelity is not finite or exceeds one beyond its reduction roundoff window"
        )
    return fidelity, window


def _occupation_bits(occupations, name: str = "reference_occupations") -> tuple[int, ...]:
    """Read exact 0/1 scalars or a bitstring without truncating occupations."""

    if isinstance(occupations, str):
        if any(bit not in "01" for bit in occupations):
            raise ValueError(f"{name} must contain only 0/1 values")
    else:
        occupations = tuple(occupations)
        if any(not isinstance(bit, (Real, np.bool_)) or bit not in (0, 1) for bit in occupations):
            raise ValueError(f"{name} must contain only 0/1 values")
    return tuple(int(bit) for bit in occupations)
