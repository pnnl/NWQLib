"""Constant-source LCHS numerical kernels and explicit remainder bounds."""

from __future__ import annotations

from nwqlib._limits import DEFAULT_MAX_BYTES

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log

import numpy as np

from nwqlib._linalg_laws import singular_values_work
from nwqlib._validation import integer
from nwqlib.algorithms.lchs.time_independent_common import validate_final_time
from nwqlib._numerics import componentwise_exact_zero, stable_vector_norm


@dataclass(frozen=True, kw_only=True)
class _DuhamelBoundEvaluation:
    """Duhamel remainder bound and whether it may be used.

    Attributes:
        value: Physical L2 bound on the source-time quadrature error. It is
            nan when the norm of a nonzero A or b underflows to zero, and inf
            when a norm or the bound overflows.
        unusable_nonzero: True when the true remainder may be nonzero but the
            evaluated value is nan, inf or rounded to zero.
    """

    value: float
    unusable_nonzero: bool


def duhamel_quadrature(final_time: float, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss–Legendre nodes and weights on [0, T] for the source time integral.

    One Gauss–Legendre panel on [0, T] discretizes the Duhamel integral of
    An, Childs and Lin, arXiv:2312.03916v2, Eq. (2), the single-panel case
    of their Eq. (72). `LCHS` uses `duhamel_nodes` of these nodes for a
    constant source. The rule is exact for polynomials of degree
    `2*node_count - 1`. Its remainder bound follows DLMF Eqs. 3.5.19 and
    3.5.21, and `LCHSRefinement(components=("duhamel",))` evaluates it for
    a Result.

    Args:
        final_time (float): Positive, finite elapsed time T.
        node_count (int): Number of nodes, at least 1.

    Returns:
        rule (tuple[numpy.ndarray, numpy.ndarray]): The nodes `T*(x_i + 1)/2`
            and weights `T*w_i/2`, from the Gauss–Legendre rule `(x_i, w_i)`
            on [-1, 1].

    Raises:
        ValueError: If `final_time` is not positive and finite or
            `node_count` is below 1.
    """

    final_time = validate_final_time(final_time)
    node_count = integer(node_count, "node_count", 1)
    nodes, weights = np.polynomial.legendre.leggauss(node_count)
    mapped_nodes = 0.5 * final_time * (nodes + 1.0)
    mapped_weights = 0.5 * final_time * weights
    return mapped_nodes, mapped_weights


def _scaling_safe_spectral_norm(matrix: np.ndarray) -> float:
    """Evaluate a matrix spectral norm without squaring at the input scale."""

    array = np.asarray(matrix, dtype=complex)
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scaled_norm = float(np.linalg.norm(array / maximum, ord=2))
        return float(maximum * scaled_norm)


def _duhamel_quadrature_error_bound(
    *,
    matrix: np.ndarray,
    source_term: np.ndarray,
    final_time: float,
    node_count: int,
    lambda_min_before_psd_conversion: float,
    matrix_norm: float | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_dense_work: int = 100_000_000,
) -> _DuhamelBoundEvaluation:
    """Evaluate the constant-source Gauss-Legendre remainder in log form.

    Write ``f_2m`` for the derivative of order 2m of f. DLMF 3.5.19 gives
    the m-point Gauss remainder ``gamma_m*f_2m(xi)/(2m)!``, with xi in the
    integration interval, and DLMF 3.5.21 gives
    ``gamma_m = 2**(2m+1)*(m!)**4/((2m+1)*((2m)!)**2)`` for Legendre on
    [-1,1]. The map ``s = T*(x+1)/2`` multiplies the remainder by
    ``(T/2)**(2m+1)``. Together they give
    ``T**(2m+1)*(m!)**4/((2m+1)*((2m)!)**3)`` times ``sup||f_2m||``.
    Here m is node_count, T is final_time, A is matrix, b is source_term,
    and ``L=(A+A†)/2``. For ACL arXiv:2312.03916v2 Eq.(2),
    ``f(s)=exp(-A*(T-s))*b`` and
    ``||f_2m|| <= ||A||**(2m)*exp(max(0,-lambda_min(L))*T)*||b||``, because
    ``||exp(-A*t)|| <= ||exp(-L*t)||`` (ACL Lemma 21, Eq. (162)). The
    smallest eigenvalue lambda_min is taken before the PSD shift, so the
    bound describes the physical growth. For nonzero complex vector error E,
    choose ``v=E/||E||`` and the real scalar function ``Re(v†f)``. Its scalar
    remainder equals ``||E||``, with derivatives bounded by ``||f_2m||``.
    No dimension factor or common complex mean-value point is assumed.
    This is a numerical evaluation of the ideal bound, not an
    outward-rounded interval.
    """

    from nwqlib.operators.access import _check_bytes
    final_time = validate_final_time(final_time)
    node_count = integer(node_count, "node_count", 1)
    max_dense_work = integer(max_dense_work, "max_dense_work", 0)
    d = len(source_term)
    # Bytes: 64*d*d, four complex128 d-by-d arrays, covers the complex copy,
    # its absolute values, the scaled copy and the SVD input of
    # _scaling_safe_spectral_norm. 64*d covers four complex128 d-vectors.
    _check_bytes(64*d*d+64*d, max_bytes, "Duhamel remainder arrays")
    # Work: 8*d for the stable source norm, 32 for the scalar log formula,
    # and, when ||A|| is not supplied, the singular values of
    # _scaling_safe_spectral_norm (_linalg_laws.singular_values_work) plus
    # 4*d**2 entrywise.
    work = 8*d+32+(singular_values_work(d)+4*d*d if matrix_norm is None else 0)
    if work > max_dense_work:
        raise ValueError("Duhamel remainder exceeds max_dense_work before norm work")

    if final_time == 0 or componentwise_exact_zero(matrix) or componentwise_exact_zero(source_term):
        return _DuhamelBoundEvaluation(value=0.0, unusable_nonzero=False)

    if matrix_norm is None:
        matrix_norm = _scaling_safe_spectral_norm(matrix)
    source_norm = stable_vector_norm(source_term)
    if matrix_norm <= 0.0 or source_norm <= 0.0:
        return _DuhamelBoundEvaluation(value=float("nan"), unusable_nonzero=True)
    if not isfinite(matrix_norm) or not isfinite(source_norm):
        return _DuhamelBoundEvaluation(value=float("inf"), unusable_nonzero=True)

    rho_plus = max(0.0, -float(lambda_min_before_psd_conversion))
    m = int(node_count)
    log_bound = (
        (2 * m + 1) * log(float(final_time))
        + 4.0 * lgamma(m + 1)
        - log(2 * m + 1)
        - 3.0 * lgamma(2 * m + 1)
        + 2.0 * m * log(matrix_norm)
        + rho_plus * float(final_time)
        + log(source_norm)
    )
    try:
        value = float(exp(log_bound))
    except OverflowError:
        value = float("inf")
    return _DuhamelBoundEvaluation(
        value=value,
        unusable_nonzero=not isfinite(value) or value == 0.0,
    )


__all__ = ["duhamel_quadrature"]
