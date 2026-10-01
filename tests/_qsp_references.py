"""Independent ``1/x`` polynomial references for the QLS inverse fit.

The fitted polynomial approximates ``1/(kappa_be x)`` on
``D(a) = {x : a <= |x| <= 1}`` with ``a = 1/kappa_be``. These references
approximate ``1/x``; divide them by ``kappa_be`` to compare with the fit.

[CKS] is Childs, Kothari, and Somma, arXiv:1511.02306v2, whose Lemmas 17-19
give the ``1/x`` expansion and the ``d = O(kappa log(kappa/eps))`` degree
law. [SNWPB] is Sunderhauf, Nemeth, Walayat, Patterson, and Berntson,
arXiv:2507.15537v1, whose Theorem 1 and Prop. 5 give the optimal polynomial
family ``L_n``, Eqs. (7)-(8) its closed-form error and minimum degree, and
Eqs. (9)-(13) its stable recurrence evaluation. Numbers refer to these
arXiv versions.
"""

from __future__ import annotations

from math import ceil, log

import numpy as np
import scipy.special


def cks_inverse_coefficients(kappa_be: float, epsilon_inv: float) -> np.ndarray:
    """Return the [CKS] (arXiv:1511.02306v2) closed-form Chebyshev coefficients of ``p(x) ~ 1/x``.

    Implements the truncated ``(1 - (1 - x^2)^b)/x`` expansion of [CKS]
    arXiv:1511.02306v2, Lemmas 17-19, in the summary parameterization of
    [SNWPB] arXiv:2507.15537v1, Eqs. (17)-(19), with the halved-error
    convention of its footnote (both the smoothing and the truncation get
    ``epsilon/2``):

    ``b = ceil(kappa^2 log(kappa/(eps/2)))``,
    ``D = ceil(sqrt(b log(4 b/(eps/2))))``, degree ``d = 2D + 1``, and

    ``p(x) = 4 sum_{j=0}^{D} (-1)^j [2^{-2b} sum_{i=j+1}^{b} C(2b, b+i)]
    T_{2j+1}(x)``.

    The binomial sums are evaluated in log-space (``gammaln`` +
    ``logsumexp``): at the ``b ~ kappa^2 log(kappa/eps)`` scale the direct
    ``2^{-2b} C(2b, b+i)`` products underflow float64. The result is the full
    Chebyshev coefficient vector ``(c_0..c_d)`` with zero even entries.
    """

    half_epsilon = 0.5 * epsilon_inv
    b_smoothing = int(ceil(kappa_be**2 * log(kappa_be / half_epsilon)))
    truncation = int(ceil(np.sqrt(b_smoothing * log(4.0 * b_smoothing / half_epsilon))))
    truncation = min(truncation, b_smoothing - 1)

    log_two_b_choose = scipy.special.gammaln(2 * b_smoothing + 1)
    coefficients = np.zeros(2 * truncation + 2)
    for j in range(truncation + 1):
        i_values = np.arange(j + 1, b_smoothing + 1)
        log_binomials = (
            log_two_b_choose
            - scipy.special.gammaln(b_smoothing + i_values + 1)
            - scipy.special.gammaln(b_smoothing - i_values + 1)
        )
        log_sum = scipy.special.logsumexp(log_binomials)
        magnitude = np.exp(log_sum - 2.0 * b_smoothing * log(2.0))
        coefficients[2 * j + 1] = 4.0 * ((-1.0) ** j) * magnitude
    return coefficients


def _snwpb_lfrac(y_values: np.ndarray, order: int, a: float) -> np.ndarray:
    """Evaluate the normalized ``L_n`` family via the stable recurrence.

    ``Lfrac_n(y; a) = L_n(y; a) / alpha(a)^n`` with
    ``alpha(a) = (1 + a) / (2 (1 - a))``, following [SNWPB]
    arXiv:2507.15537v1, Eqs. (9)-(12):
    ``Lfrac_1 = (y + (1-a)/(1+a)) / alpha``,
    ``Lfrac_2 = (y^2 + ((1-a)/(2(1+a))) y - 1/2) / alpha^2``, and
    ``Lfrac_n = y Lfrac_{n-1} / alpha - Lfrac_{n-2} / (4 alpha^2)``.
    """

    alpha_a = (1.0 + a) / (2.0 * (1.0 - a))
    first = (y_values + (1.0 - a) / (1.0 + a)) / alpha_a
    if order == 1:
        return first
    second = (y_values**2 + (1.0 - a) / (2.0 * (1.0 + a)) * y_values - 0.5) / alpha_a**2
    previous, current = first, second
    for _ in range(3, order + 1):
        previous, current = current, y_values * current / alpha_a - previous / (
            4.0 * alpha_a**2
        )
    return current


def snwpb_inverse_values(x_values, *, degree: int, kappa_be: float) -> np.ndarray:
    """Evaluate the [SNWPB] (arXiv:2507.15537v1) optimal ``1/x`` polynomial ``P_{2n-1}(x; a)``.

    Implements Theorem 1 through the numerically stable form of Eq. (10)
    (with the Prop. 6 denominator identity already folded in):

    ``P_{2n-1}(x; a) = (1 - (-1)^n ((1+a)^2/(4a))
    Lfrac_n((2x^2 - (1+a^2))/(1 - a^2); a)) / x``,  ``a = 1/kappa_be``.

    ``P`` is the odd degree-``2n-1`` polynomial minimizing the absolute
    error ``||p - 1/x||`` on ``D(a)`` ([SNWPB] arXiv:2507.15537v1, Eq. (4));
    this differs from
    the fit certificate's ``|kappa_be x p(x) - 1|`` form, so comparisons
    measure both on the same grid. ``degree`` is odd and ``x_values``
    exclude 0, where the quotient form is 0/0 although ``P(0) = 0``.
    """

    grid = np.asarray(x_values, dtype=float)
    a = 1.0 / float(kappa_be)
    order = (degree + 1) // 2
    y_values = (2.0 * grid**2 - (1.0 + a**2)) / (1.0 - a**2)
    lfrac = _snwpb_lfrac(y_values, order, a)
    sign = -1.0 if order % 2 else 1.0
    return (1.0 - sign * (1.0 + a) ** 2 / (4.0 * a) * lfrac) / grid


def snwpb_error_for_degree(degree: int, kappa_be: float) -> float:
    """Return the closed-form optimal absolute error ``eps_{2n-1}(a)``.

    [SNWPB] arXiv:2507.15537v1, Eq. (7):
    ``eps = (1-a)^n / (a (1+a)^(n-1))`` with
    ``a = 1/kappa_be`` and ``n = (degree+1)/2``; evaluated in log-space so
    large degrees do not underflow.
    """

    a = 1.0 / float(kappa_be)
    order = (degree + 1) // 2
    return float(np.exp(order * np.log1p(-a) - np.log(a) - (order - 1) * np.log1p(a)))


def snwpb_mindegree_for_error(epsilon: float, kappa_be: float) -> int:
    """Return the closed-form minimum odd degree for absolute error ``epsilon``.

    [SNWPB] arXiv:2507.15537v1, Eq. (8):
    ``n = ceil((log(1/eps) + log(1/a) + log(1+a)) / (log(1+a) - log(1-a)))``
    and ``d = 2n - 1``, with ``a = 1/kappa_be``.
    """

    a = 1.0 / float(kappa_be)
    order = int(
        ceil((log(1.0 / epsilon) + log(1.0 / a) + np.log1p(a)) / (np.log1p(a) - np.log1p(-a)))
    )
    return 2 * max(order, 1) - 1
