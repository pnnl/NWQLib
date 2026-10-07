"""Kernel-reflection polynomial of Dalzell's shortcut quantum linear solver.

Dalzell, "A shortcut to an optimal quantum linear system solver",
arXiv:2406.12086v2. Equation and page numbers refer to that version. Algorithm 1
(p. 5) applies kernel reflection (KR) to ``|e_n>`` through QSVT on the
augmented matrix ``G_t`` of Eq. (11). KR uses the even degree-``2 ell``
polynomial ``K`` of App. B.3, Eq. (62), built from the filter ``F`` of
App. B.2, Eq. (52), with ``Delta = 1/kappa`` and ``ell`` from Eq. (6).
Lemma 3 gives ``K(0) = 1`` and ``|K| <= 1`` on ``[-1, 1]``, and p. 4
states ``-1 <= K(x) <= -1 + 4 eta/(1 + eta)`` for ``1/kappa <= x <= 1``.
Lemma 3, item 2, prints this upper bound for ``|K(x)|``, which cannot hold
for ``eta < 1/3`` because the bound is then negative. The proof, through
Eq. (63), bounds ``K(x)`` itself. That signed bound is the statement of
p. 4 and the one behind Eqs. (5) and (17). The QLS Method's circuit and its
classical numerical model use this polynomial.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log, sqrt

import numpy as np

from nwqlib._linalg_laws import least_squares_work
from nwqlib._validation import integer
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes
from nwqlib.subroutines.qsp.inverse import _FitCost
from nwqlib.subroutines.qsp.phases import chebyshev_grid


# K is exactly an even polynomial of degree 2*ell in x, so least squares on
# more than 2*ell + 1 Chebyshev nodes recovers its coefficients up to
# rounding. The fit uses max(2001, 8(d+1)) nodes. The floor 2001 is an
# untuned sample count, registered in docs/ENGINEERING_CONSTANTS.md. Revisit
# with a changed polynomial family, degree range or fit workload.
_KR_CONSTRUCTION_GRID_POINTS = 2001


@dataclass(frozen=True, kw_only=True)
class KernelReflectionPolynomial:
    """Kernel-reflection polynomial `K` of Dalzell arXiv:2406.12086v2, App. B.3, Eq. (62).

    [`plan_kernel_reflection`][nwqlib.subroutines.qsp.shortcut.plan_kernel_reflection]
    returns a coefficient-free plan, and the QLS Method's planning fills in
    the coefficients. The domain gap is `Delta = 1/kappa_be`. The fields below are read-only.

    Attributes:
        coefficients: Chebyshev coefficients of the even kernel-reflection
            polynomial, or empty for a coefficient-free sizing plan.
        kr_ell: Integer half-degree from Dalzell arXiv:2406.12086v2, Eq. 6.
        kappa_be: Reciprocal of the gap `Delta = 1/kappa_be`, a lower
            bound on the nonzero singular values of the normalized input.
            It is not a matrix condition number. QLS passes its polynomial
            domain parameter, which is at least its encoded gap parameter,
            `alpha/sigma_min(A)` or the caller's bound on it (the `kappa`
            setting of [`QLS`][nwqlib.algorithms.qls.method.QLS]). That
            parameter can exceed `sigma_max(A)/sigma_min(A)`
            ([QLS guide](../../algorithms/qls.md#inputs-scale-and-padding)).
        kr_eta: Kernel-reflection approximation parameter in `(0, 1]`.
    """

    coefficients: tuple[float, ...]
    kr_ell: int
    kappa_be: float
    kr_eta: float

    @property
    def degree(self) -> int:
        """Even KR degree, structurally ``2 * kr_ell`` (Dalzell arXiv:2406.12086v2, Eq. 6)."""

        return 2 * self.kr_ell


def dalzell_eta_from_precision(epsilon_inv: float) -> float:
    """Return the known-norm shortcut choice ``eta = eps / sqrt(2)``.

    Dalzell (arXiv:2406.12086v2, pp. 5-6, before Eq. (22)) chooses this
    kernel-reflection
    parameter for the norm estimate equal to the solution norm,
    ``t = ||x||``. Eq. (20) bounds the output trace distance by
    ``eta / cos(theta_t)``, and ``cos(theta_t) = 1/sqrt(2)`` at that ``t``,
    so the bound equals ``eps``. The shortcut solvers use this ``eta`` for
    every numeric ``t``. For ``t != ||x||`` the same bound is
    ``eta / cos(theta_t)`` with ``theta_t = arctan(||x||/t)`` (Eq. (7)).
    """

    epsilon_inv = float(epsilon_inv)
    if not 0.0 < epsilon_inv < 1.0:
        raise ValueError("epsilon_inv must be in the open interval (0, 1)")
    return epsilon_inv / sqrt(2.0)


def plan_kernel_reflection(kappa_be: float, eta: float) -> KernelReflectionPolynomial:
    """Return the coefficient-free sizing plan with the half-degree of Dalzell arXiv:2406.12086v2, Eq. 6.

    ``kr_ell = ceil(kappa_be * ln(2 / eta) / 2)``. This is the upper bound
    in Dalzell arXiv:2406.12086v2, Eq. (51), with ``Delta = 1/kappa_be``, so
    it satisfies the
    degree condition under which Lemma 3 holds with the filter value
    ``F(Delta) <= eta``.

    Args:
        kappa_be (float): Reciprocal of the gap `Delta`, greater than 1.
        eta (float): Kernel-reflection approximation parameter, in
            `(0, 1]`.

    Returns:
        plan (KernelReflectionPolynomial): Empty `coefficients`, the
            half-degree `kr_ell` above, and the arguments as `kappa_be` and
            `kr_eta`. The polynomial degree is `2*kr_ell`.

    Raises:
        ValueError: If `kappa_be <= 1`, or if `eta` is not in `(0, 1]`.
    """

    kappa_be = float(kappa_be)
    eta = float(eta)
    if kappa_be <= 1.0:
        raise ValueError("kappa_be must exceed 1")
    if not 0.0 < eta <= 1.0:
        raise ValueError("eta must lie in (0, 1]")
    return KernelReflectionPolynomial(
        coefficients=(),
        kr_ell=int(ceil(kappa_be * log(2.0 / eta) / 2.0)),
        kappa_be=kappa_be,
        kr_eta=eta,
    )


def kernel_reflection_cost(plan) -> _FitCost:
    """Return the bytes and work counted for computing the kernel-reflection polynomial of one plan.

    With degree ``d = 2 ell`` and ``n = max(2001, 8 (d + 1))`` fit nodes,
    ``peak_bytes`` counts 8-byte slots:

    - ``3 n (d + 1)``: the Chebyshev-Vandermonde matrix built inside
      ``chebfit``, its column-scaled copy and the copy that ``lstsq``
      factorizes, which are alive together.
    - ``12 n``: node, filter and target vectors, including the Clenshaw
      recurrence vectors of ``chebval``.
    - ``4 (d + 1)``: coefficient vectors.

    With d and n as defined above and ``ell = plan.kr_ell``, ``work`` adds
    ``least_squares_work(n, d + 1)``, NWQLib's work formula for
    ``numpy.linalg.lstsq`` on the ``n x (d + 1)`` design matrix, and
    ``8 n (d + 1)`` for the Vandermonde matrix, the degree-``ell`` filter
    evaluation and the scaling. LAPACK workspace is excluded.
    """
    d = plan.degree
    n = max(_KR_CONSTRUCTION_GRID_POINTS, 8 * (d + 1))
    return _FitCost(
        coefficients=d + 1,
        peak_bytes=8*(3*n*(d+1)+12*n+4*(d+1)),
        work=least_squares_work(n, d + 1) + 8*n*(d+1),
    )


def _realize_kernel_reflection(
    plan: KernelReflectionPolynomial,
    *, max_degree=256, max_work=1_000_000_000, max_bytes=DEFAULT_INPUT_BYTES,
) -> KernelReflectionPolynomial:
    """Realize the synthesis coefficients of a sizing plan.

    The selected polynomial's integer half-degree is
    ``ell = ceil(kappa_be * ln(2/eta) / 2)``, where
    ``kappa_be = plan.kappa_be`` is the reciprocal of the polynomial-domain
    gap ``Delta = 1/kappa_be`` and ``eta = plan.kr_eta`` is the
    kernel-reflection approximation parameter.
    The values are Dalzell arXiv:2406.12086v2, App. B.3:
    start from the Chebyshev filter
    ``F(x) = T_ell((1 + Delta^2 - 2x^2)/(1 - Delta^2)) / T_ell((1 + Delta^2)/(1 - Delta^2))``
    of Eq. (52) and map it affinely into
    ``K(x) = (2F(x) - 1 + F(Delta)) / (1 + F(Delta))`` of Eq. (62), with
    ``K(0)=1`` and approximately ``-1`` off the kernel. The Chebyshev
    coefficients of ``K`` come from a least-squares fit of these exact
    values. Odd coefficients are then set to zero and the even ones divided
    by the fitted ``K(0)``, so the stored table has exact even parity and
    ``K(0) = 1`` up to rounding.

    The argument ``z(x) = (1 + Delta^2 - 2x^2)/(1 - Delta^2)`` maps
    ``Delta <= |x| <= 1`` onto ``[-1, 1]`` and ``|x| < Delta`` onto
    ``(1, z0]`` with ``z0 = z(0) = (1 + Delta^2)/(1 - Delta^2)``. The
    denominator is ``T_ell(z0)``, and its argument ``z0`` exceeds 1. For
    ``z > 1`` Dalzell arXiv:2406.12086v2, Eq. (53), gives
    ``T_ell(z) = cosh(ell arccosh z)``, and
    Eq. (54) applies it to this denominator. The code evaluates that closed
    form directly instead of a Chebyshev recurrence. Since
    ``arccosh z0 = ln((1 + Delta)/(1 - Delta)) = 2 artanh(Delta) >= 2 Delta``
    and ``2 ell Delta >= ln(2/eta)`` by the choice of ``ell``,
    ``T_ell(z0) >= cosh(ln(2/eta)) >= 1/eta``. Hence
    ``F(Delta) = 1/T_ell(z0) <= eta``, the filter bound of Lemma 1, item 2.
    The proof of Lemma 3 applies the same denominator bound to ``K``
    through Eq. (63).
    """

    if plan.degree > integer(max_degree, "max_degree", 1):
        raise ValueError(f"kernel reflection degree exceeds max_degree={max_degree}")
    cost = kernel_reflection_cost(plan)
    _check_bytes(cost.peak_bytes, max_bytes, "kernel reflection coefficient fit")
    if cost.work > integer(max_work, "max_work", 1):
        raise ValueError(f"kernel reflection coefficient fit exceeds max_work={max_work}")
    kappa_be = plan.kappa_be
    eta = plan.kr_eta
    ell = plan.kr_ell
    degree = 2 * ell
    delta = 1.0 / kappa_be

    def _filter(values: np.ndarray) -> np.ndarray:
        """Evaluate the filter ``F`` of Dalzell arXiv:2406.12086v2, Eq. (52), at the points ``values``."""
        z_values = (1.0 + delta**2 - 2.0 * values**2) / (1.0 - delta**2)
        # Denominator T_ell(z0) for z0 > 1, by the closed form of Eq. (54).
        denominator = np.cosh(
            ell * np.arccosh((1.0 + delta**2) / (1.0 - delta**2))
        )
        # Numerator T_ell(z(x)), evaluated as the Chebyshev series with one unit
        # coefficient.
        basis = np.zeros(ell + 1)
        basis[ell] = 1.0
        return np.polynomial.chebyshev.chebval(z_values, basis) / denominator

    fit_points = max(_KR_CONSTRUCTION_GRID_POINTS, 8 * (degree + 1))
    x_values = chebyshev_grid(fit_points)
    filter_values = _filter(x_values)
    filter_delta = float(_filter(np.asarray([delta]))[0])
    # Eq. (62): K = (2F - 1 + F(Delta)) / (1 + F(Delta)).
    target_values = (2.0 * filter_values - 1.0 + filter_delta) / (1.0 + filter_delta)
    coefficients = np.polynomial.chebyshev.chebfit(x_values, target_values, degree)
    # K is even in x, so the fitted odd coefficients are rounding residue.
    coefficients[1::2] = 0.0
    # Rescale the even coefficients so that K(0) = 1 after the least-squares fit.
    coefficients[0::2] /= np.polynomial.chebyshev.chebval(0.0, coefficients)

    return KernelReflectionPolynomial(
        coefficients=tuple(float(value) for value in coefficients),
        kr_ell=ell,
        kappa_be=kappa_be,
        kr_eta=eta,
    )


__all__ = [
    "KernelReflectionPolynomial",
    "dalzell_eta_from_precision",
    "plan_kernel_reflection",
]
