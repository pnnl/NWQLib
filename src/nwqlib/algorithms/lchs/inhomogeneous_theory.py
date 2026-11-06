"""Constant-source LCHS numerical routines and explicit remainder bounds."""

from __future__ import annotations

from math import lgamma, log

import numpy as np
from scipy.special import roots_legendre

from nwqlib._validation import integer
from nwqlib.algorithms.lchs.time_independent_common import validate_final_time


def duhamel_quadrature(final_time: float, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss–Legendre nodes and weights on [0, T] for the source time integral.

    One Gauss–Legendre panel on [0, T] discretizes the Duhamel integral of
    An, Childs and Lin, arXiv:2312.03916v2, Eq. (2), the single-panel case
    of their Eq. (72). `LCHS` uses `duhamel_nodes` of these nodes for a
    constant source. The rule is exact for polynomials of degree
    `2*node_count - 1`. Its remainder bound follows DLMF Eqs. 3.5.19 and
    3.5.21, and `LCHSRefinement(components=("duhamel",))` evaluates it for
    a Result. The nodes and weights are SciPy's binary64 Gauss–Legendre
    values. The remainder bound is for the exact Gauss rule, and the
    error of these binary64 values belongs to `native_floating_point`, which
    LCHS reports as unknown.

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
    nodes, weights = roots_legendre(node_count)
    mapped_nodes = 0.5 * final_time * (nodes + 1.0)
    mapped_weights = 0.5 * final_time * weights
    return mapped_nodes, mapped_weights


def _duhamel_quadrature_error_bound(*, generator_norm, lower_bound, source_scale, final_time, node_count):
    """Return the constant-source Gauss-Legendre remainder bound as a PhysicalScale.

    Write ``f_2m`` for the derivative of order 2m of f. DLMF 3.5.19 gives
    the m-point Gauss remainder ``gamma_m*f_2m(xi)/(2m)!``, with xi in the
    integration interval, and DLMF 3.5.21 gives
    ``gamma_m = 2**(2m+1)*(m!)**4/((2m+1)*((2m)!)**2)`` for Legendre on
    [-1,1]. The map ``s = T*(x+1)/2`` multiplies the remainder by
    ``(T/2)**(2m+1)``. Together they give
    ``T**(2m+1)*(m!)**4/((2m+1)*((2m)!)**3)`` times ``sup||f_2m||``.
    Here m is node_count, T is final_time, b is the source and
    ``L=(A+A†)/2``. For ACL arXiv:2312.03916v2 Eq.(2),
    ``f(s)=exp(-A*(T-s))*b`` and
    ``||f_2m|| <= ||A||**(2m)*exp(max(0,-lambda_min(L))*T)*||b||``, because
    ``||exp(-A*t)|| <= ||exp(-L*t)||`` (ACL Lemma 21, Eq. (162)). The
    smallest eigenvalue lambda_min is taken before the PSD shift, so the
    bound describes the physical growth. For nonzero complex vector error E,
    choose ``v=E/||E||`` and the real scalar function ``Re(v†f)``. Its scalar
    remainder equals ``||E||``, with derivatives bounded by ``||f_2m||``.
    No dimension factor or common complex mean-value point is assumed.
    The bound is for the exact Gauss rule, with exact nodes and weights.
    This is a numerical evaluation of the ideal bound, not an
    outward-rounded interval.

    The bound grows as B_A**(2m) in the bound B_A on ||A||, so a B_A looser
    by a factor c enlarges it by c**(2m), about 1e30 for m = 50 and c = 2.

    Args:
        generator_norm: B_A, a finite bound on ||A||_2 of the original
            generator, not of the shifted L + s*I or a centered child.
        lower_bound: Any lower bound on lambda_min(L) of the original L.
        source_scale: The PhysicalScale of ||b||.
        final_time: Elapsed time T.
        node_count: Number m of Gauss-Legendre nodes.

    Returns:
        The bound as a PhysicalScale, whose integer exponent may lie outside
        the binary64 range, exactly zero when T, B_A or ||b|| is zero, or
        ``float("inf")`` when its logarithm is not finite.
    """
    from nwqlib.problems.inputs import PhysicalScale
    from .solution_error_budget import scaled_exp

    if final_time == 0 or generator_norm == 0 or source_scale.mantissa == 0:
        return PhysicalScale(mantissa=0.0, exponent=0)
    log_bound = duhamel_log_bound(final_time=final_time, node_count=node_count, generator_norm=generator_norm,
        lower_bound=lower_bound, log_source_norm=log(source_scale.mantissa)+source_scale.exponent*log(2.0))
    scale = scaled_exp(log_bound)
    return float("inf") if scale is None else scale


def duhamel_log_bound(*, final_time, node_count, generator_norm, lower_bound, log_source_norm):
    """Return the natural logarithm of the Gauss-Legendre remainder bound of the constant-source term.

    ``_duhamel_quadrature_error_bound`` derives, for m = node_count
    Gauss-Legendre points on [0, T] (DLMF 3.5.19 and 3.5.21) and
    f(r) = exp(-A (T - r)) b,

        delta <= T**(2m+1) (m!)**4 / ((2m+1) ((2m)!)**3) * B_A**(2m) * exp(omega*T) * ||b||,

    with omega = max(0, -lower_bound). ``lower_bound`` is any lower bound on
    lambda_min(L) of the original L, since exp(omega*T) only grows as the
    bound decreases. B_A = ``generator_norm`` must bound the original
    physical generator A, not the shifted L + s*I or a centered child,
    because the derivatives of f are powers of A. When only a bound B_Ls on
    ||L + s*I|| is known, L = (L + s*I) - s*I gives ||L|| <= B_Ls + |s| for
    the signed shift s, which is negative under known decay.
    ``log_source_norm`` is log ||b||. All factors are combined as
    logarithms, so a bound outside the binary64 range is still evaluated;
    the caller decides how to report it. This is a binary64 evaluation of
    the ideal bound, not an outward-rounded interval.
    """
    m = int(node_count)
    return ((2 * m + 1) * log(float(final_time)) + 4.0 * lgamma(m + 1) - log(2 * m + 1)
            - 3.0 * lgamma(2 * m + 1) + 2.0 * m * log(generator_norm)
            + max(0.0, -float(lower_bound)) * float(final_time) + log_source_norm)


__all__ = ["duhamel_quadrature"]
