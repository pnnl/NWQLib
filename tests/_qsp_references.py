"""Independent references for the QLS inverse polynomial.

The inverse polynomial approximates ``1/(kappa x)`` on
``D(a) = {x : a <= |x| <= 1}`` with ``a = 1/kappa``. The [CKS] reference
approximates ``1/x``, so divide it by ``kappa`` to compare. The restricted
residual bound evaluates a stored table in exact rational arithmetic.

[CKS] is Childs, Kothari, and Somma, arXiv:1511.02306v2, whose Lemmas 17-19
give the ``1/x`` expansion and the ``d = O(kappa log(kappa/eps))`` degree
law. [SNWPB] is Sunderhauf, Nemeth, Walayat, Patterson, and Berntson,
arXiv:2507.15537v1, whose Eqs. (17)-(19) supply the summary
parameterization used below. Numbers refer to these arXiv versions.
"""

from __future__ import annotations

from fractions import Fraction
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


def restricted_inverse_residual_bound(odd: list[Fraction], kappa: Fraction) -> Fraction:
    """Exact upper bound on ``max |kappa x P(x) - 1|`` over ``1/kappa <= |x| <= 1``.

    ``odd`` holds the coefficients ``c_1, c_3, ..., c_{2m-1}`` of
    ``P = sum_j c_{2j-1} T_{2j-1}`` as exact rationals. With ``s = 2x^2 - 1``,
    ``x T_{2j-1}(x) = (T_j(s) + T_{j-1}(s))/2``, so the residual is a degree-m
    Chebyshev series in ``s``. On the domain ``s = alpha + beta z`` with
    ``alpha = 1/kappa^2``, ``beta = 1 - alpha`` and ``|z| <= 1``. Each
    ``T_j(alpha + beta z)`` is expanded in Chebyshev polynomials of ``z`` by
    the three-term recurrence, and the sum of the absolute coefficients bounds
    the residual. The work is ``O(m^2)`` rational operations.
    """
    m = len(odd)
    residual = [Fraction(0)] * (m + 1)
    residual[0] = Fraction(-1)
    for j, value in enumerate(odd, 1):
        residual[j] += kappa * value / 2
        residual[j - 1] += kappa * value / 2
    alpha = 1 / kappa**2
    beta = 1 - alpha
    in_z = [Fraction(0)] * (m + 1)
    previous, current = [], [Fraction(1)]
    for j in range(m + 1):
        for i, value in enumerate(current):
            in_z[i] += residual[j] * value
        if j == m:
            break
        following = [Fraction(0)] * (j + 2)
        for i, value in enumerate(current):
            following[i] += 2 * alpha * value
            # 2 z T_0 = 2 T_1 and 2 z T_i = T_{i-1} + T_{i+1} for i > 0.
            following[i + 1] += beta * value if i else 2 * beta * value
            if i:
                following[i - 1] += beta * value
        for i, value in enumerate(previous):
            following[i] -= value
        # T_1(s) = s T_0, half of the recurrence's 2 s T_0.
        previous, current = current, following if j else [value / 2 for value in following]
    return sum(abs(value) for value in in_z)
