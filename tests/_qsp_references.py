"""Independent ``1/x`` polynomial references for the QLS inverse fit.

The fitted polynomial approximates ``1/(kappa_be x)`` on
``D(a) = {x : a <= |x| <= 1}`` with ``a = 1/kappa_be``. These references
approximate ``1/x``; divide them by ``kappa_be`` to compare with the fit.

[CKS] is Childs, Kothari, and Somma, arXiv:1511.02306v2, whose Lemmas 17-19
give the ``1/x`` expansion and the ``d = O(kappa log(kappa/eps))`` degree
law. [SNWPB] is Sunderhauf, Nemeth, Walayat, Patterson, and Berntson,
arXiv:2507.15537v1, whose Eqs. (17)-(19) supply the summary
parameterization used below. Numbers refer to these arXiv versions.
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
