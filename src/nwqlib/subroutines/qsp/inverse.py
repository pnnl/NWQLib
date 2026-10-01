"""Certified ``1/x`` polynomial construction for QSVT matrix inversion.

QLS polynomial selection calls ``_fit_inverse_chebyshev``, which builds an
odd polynomial ``P`` approximating ``1/(kappa x)`` on the domain
``D(a) = {x : a <= |x| <= 1}`` with ``a = 1/kappa``: odd-Chebyshev least
squares on domain-restricted Chebyshev nodes, degree found by
double-then-bisect against an affine Chebyshev-grid norming bound on
``max_D |kappa * x * P(x) - 1| <= epsilon_inv``. The fit procedure is
NWQLib's own construction.

The domain parameter ``kappa`` must exceed 1. QLS passes its
``polynomial_kappa``, the encoding's ``kappa_be`` raised to at least
``QLS_POLYNOMIAL_KAPPA_FLOOR = 1.01`` so that a perfectly conditioned A
still gives a domain with ``a < 1``. ``kappa_be`` is
``max(1, alpha / sigma_min(A))`` for ``kappa="auto"`` and otherwise the
supplied ``kappa``, which must cover that value. With ``kappa="auto"``, the
analytic periodic encoding passes ``alpha / min(sigma_min, encoded gap)``,
rounded outward (``algorithms/qls/periodic.py``). Every ``kappa`` below is
that domain parameter.

Two choices make the polynomial usable by QSVT and by physical recovery.
The target ``1/(kappa x)`` has magnitude at most one on ``D``, so ``P``
needs only a small rescale into the QSP amplitude domain. The criterion is
a relative residual. For each eigenvalue (Hermitian ``A``) or singular value
(general ``A``, through its odd singular-value transform) ``s`` of
``A/alpha`` in ``D``, ``|P(s) - 1/(kappa s)| <= epsilon_inv / (kappa |s|)``.
Summing over components, the transformed vector ``y`` of ``b`` satisfies
``||y - (kappa A/alpha)^{-1} b|| <= epsilon_inv ||(kappa A/alpha)^{-1} b||``.
Childs, Kothari, and Somma [CKS], arXiv:1511.02306v2, and Sunderhauf,
Nemeth, Walayat, Patterson, and Berntson [SNWPB], arXiv:2507.15537v1,
instead bound the absolute error ``|p(x) - 1/x|``. Equation, lemma and
section numbers refer to these arXiv versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log
from typing import NamedTuple

import numpy as np

from nwqlib._linalg_laws import least_squares_work
from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes
from nwqlib.subroutines.qsp.phases import chebyshev_grid, chebyshev_norming_sup_bound


# Engineering constants (registered in docs/ENGINEERING_CONSTANTS.md):
# - LSQ_NODE_MULTIPLIER 4: the odd-coefficient LSQ is exactly determined on
#   (d+1)/2 domain nodes; a 4x overdetermined grid removes fit-the-grid
#   minima, mirroring the phase solver's grid-multiplier rationale.
# - CERTIFICATE_GRID_PER_DEGREE 25: the sampling density N = 25d of
#   [SNWPB] arXiv:2507.15537v1, Sec. III, Eq. (26), placed on affine
#   Chebyshev-zero nodes of [1/kappa, 1]
#   so that the shared norming inequality bounds the degree-(d+1) residual.
#   That inequality is Ehlich and Zeller (1964), doi:10.1007/BF01111276,
#   Satz 2, Eqs. (12)-(14), on Chebyshev-zero nodes. SNWPB's Eq. (25) is not
#   valid on the equidistant x grid stated there, which Ehlich and Zeller,
#   doi:10.1007/BF01111276, Satz 1, treat separately. The QLS guide gives a
#   rational counterexample.
QLS_FIT_LSQ_NODE_MULTIPLIER = 4
QLS_FIT_CERTIFICATE_GRID_PER_DEGREE = 25

# Degree-law tripwire factor over the [CKS] (arXiv:1511.02306v2) reference form
# d = O(kappa log(kappa/eps)). In [CKS], Lemma 17, Eq. (74), approximates
# 1/x by (1-(1-x^2)^b)/x with b >= kappa^2 log(kappa/eps), Lemma 18,
# Eq. (77), expands it in odd Chebyshev polynomials, and Lemma 19, Eq. (88),
# truncates at j_0 = sqrt(b log(4b/eps)), which together give that degree
# law. C_law scales the reference degree
# ceil(C_law * kappa * log(kappa / epsilon_inv)). It is an engineering
# value motivated by that law, not a finite guarantee that a passing fit
# exists below the bound. In this module it only sets the runaway stop of
# the doubling search. Revisit when the fit construction or certificate
# grid changes.
QLS_INVERSE_DEGREE_LAW_FACTOR = 1.25

# Runaway stop for the doubling search, far above the tripwire. If no tested
# degree passes within this limit, fail with the finite search scope; this
# does not prove that every larger degree or different fit would fail.
# Registered in docs/ENGINEERING_CONSTANTS.md. Revisit when the fit
# construction or the certificate grid changes.
_DOUBLING_FACTOR_OVER_TRIPWIRE = 8


class _FitCost(NamedTuple):
    """Admission cost of one least-squares Chebyshev fit, computed from its degree before it runs.

    ``inverse_candidate_cost`` and ``shortcut.kernel_reflection_cost``
    return this record. Their docstrings map each term to its arrays.

    Attributes:
        coefficients: Length ``d + 1`` of the Chebyshev coefficient table.
        peak_bytes: Byte allowance for the NumPy arrays alive at the fit's
            peak, summed from per-array terms. LAPACK workspace and Python
            object overhead are excluded.
        work: Scalar operation count compared with ``max_work``. It is a
            size law, not measured time.
    """

    coefficients: int
    peak_bytes: int
    work: int


@dataclass(frozen=True, kw_only=True)
class InverseChebyshevFit:
    """Certified odd-Chebyshev fit of ``1/(kappa * x)`` on ``D(1/kappa)``.

    Args:
        coefficients: Full Chebyshev coefficient vector ``(c_0..c_d)`` of the
            fitted ``P``; even entries are exactly zero.
        degree: Selected odd degree ``d`` from the bounded search.
        kappa: Domain parameter, greater than 1, defining the domain
            ``D = {1/kappa <= |x| <= 1}``. QLS passes its ``polynomial_kappa``.
        epsilon_inv: Requested certificate tolerance.
        certificate: Chebyshev-grid norming upper bound on
            ``max_D |kappa * x * P(x) - 1|``, evaluated in binary64, not
            interval arithmetic.
        certificate_grid_points: Affine Chebyshev-grid size behind the bound.
        lsq_node_count: Least-squares node count of the accepted fit.
    """

    coefficients: tuple[float, ...]
    degree: int
    kappa: float
    epsilon_inv: float
    certificate: float
    certificate_grid_points: int
    lsq_node_count: int


def inverse_degree_law_bound(kappa: float, epsilon_inv: float) -> int:
    """Return the tripwire ``ceil(C_law * kappa * log(kappa/eps))`` in the form of [CKS] arXiv:1511.02306v2."""

    return int(ceil(QLS_INVERSE_DEGREE_LAW_FACTOR * kappa * log(kappa / epsilon_inv)))


def _odd_chebyshev_values(x_values: np.ndarray, degree: int) -> np.ndarray:
    """Return the matrix ``T_{2l+1}(x_i)`` for ``l = 0..(degree-1)//2``."""

    # One three-term-recurrence pass builds every column at once; a
    # per-column chebval loop repays O(degree) extra work per column on
    # every candidate of the doubling/bisection search.
    return np.polynomial.chebyshev.chebvander(x_values, degree)[:, 1::2]


def _fit_candidate(
    degree: int,
    *,
    kappa: float,
    certificate_grid: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Fit one odd degree and return ``(coefficients, certificate, num_nodes)``.

    The odd coefficients minimize ``sum_i (kappa x_i P(x_i) - 1)^2`` over
    ``num_nodes = 4 (d + 1)/2`` affine Chebyshev nodes on
    ``[1/kappa, 1]``. ``coefficients`` is the full table ``c_0..c_d``
    with zero even entries. ``certificate`` is the Ehlich-Zeller bound on
    ``max |kappa x P(x) - 1|`` over ``1/kappa <= |x| <= 1``, computed
    from the samples on ``certificate_grid``.
    """

    num_unknowns = (degree + 1) // 2
    num_nodes = QLS_FIT_LSQ_NODE_MULTIPLIER * num_unknowns
    # Chebyshev points affine-mapped onto the positive domain [1/kappa, 1].
    # Odd symmetry makes the negative side redundant for both fit and
    # certificate.
    reference = chebyshev_grid(num_nodes)
    lower = 1.0 / kappa
    nodes = 0.5 * (1.0 - lower) * reference + 0.5 * (1.0 + lower)
    # Residual is the certificate integrand itself: kappa * x * P(x) - 1.
    design = kappa * nodes[:, None] * _odd_chebyshev_values(nodes, degree)
    solution = np.linalg.lstsq(design, np.ones(num_nodes), rcond=None)[0]
    coefficients = np.zeros(degree + 1)
    coefficients[1::2] = solution
    certified_values = np.polynomial.chebyshev.chebval(certificate_grid, coefficients)
    sampled = float(np.max(np.abs(kappa * certificate_grid * certified_values - 1.0)))
    # Affine interval restriction preserves polynomial degree. x*P(x)-1
    # has degree d+1 and is even, so its positive-domain bound covers D(a).
    certificate = chebyshev_norming_sup_bound(sampled, degree=degree+1,
                                             num_points=len(certificate_grid))
    return coefficients, certificate, num_nodes


def inverse_candidate_cost(degree) -> _FitCost:
    """Return the admission cost of ``_fit_candidate`` at one odd degree ``d``.

    With ``u = (d + 1)/2`` odd unknowns, ``n = 4u`` least-squares nodes and
    ``q = 25 d`` certificate nodes, ``peak_bytes`` counts 8-byte slots:

    - ``n (d + 1)``: the Chebyshev-Vandermonde matrix from ``chebvander``.
      Its odd columns are a view, which the scaling turns into the design
      matrix.
    - ``3 n u``: three ``n x u`` slots, two for the design matrix and the
      copy that ``lstsq`` factorizes and one to spare.
    - ``10 n``: length-``n`` vectors (reference and affine nodes, their
      temporaries, the right-hand side and its copy inside ``lstsq``).
    - ``u + 3 (d + 1)``: the ``lstsq`` solution, the coefficient table and
      its copies inside ``chebval``.
    - ``8 q``: certificate-grid vectors (the nodes, the Clenshaw recurrence
      of ``chebval`` and the residual temporaries).

    The terms are summed although the Vandermonde matrix is released before
    ``lstsq`` runs, so ``peak_bytes`` is an allowance above the peak of these
    arrays. ``work`` is ``_linalg_laws.least_squares_work(n, u)`` for the
    dense least squares, ``4 n (d + 1)`` for the Vandermonde recurrence and
    design scaling, and ``8 q (d + 1)`` for the Clenshaw evaluation and
    residual on the certificate grid. The caller checks ``peak_bytes`` per
    candidate and sums ``work`` over all candidates.
    """
    u = (degree + 1) // 2
    n = QLS_FIT_LSQ_NODE_MULTIPLIER * u
    q = QLS_FIT_CERTIFICATE_GRID_PER_DEGREE * degree
    slots = n * (degree + 1) + 3*n*u + 10*n + u + 3*(degree+1) + 8*q
    return _FitCost(
        coefficients=degree + 1,
        peak_bytes=8 * slots,
        work=least_squares_work(n, u) + 4*n*(degree+1) + 8*q*(degree+1),
    )


def _fit_inverse_chebyshev(
    kappa: float,
    epsilon_inv: float,
    *, max_degree=256, max_work=1_000_000_000, max_bytes=DEFAULT_INPUT_BYTES,
    candidate_degrees=None,
) -> InverseChebyshevFit:
    """Select an odd-Chebyshev approximation of ``1/(kappa x)``.

    Implements NWQLib's inverse-fit construction (module docstring):
    odd-Chebyshev least squares on domain-restricted Chebyshev nodes,
    candidate degree doubled until the affine Chebyshev-grid residual norming
    bound ``max_D |kappa * x * P(x) - 1| <= epsilon_inv`` holds, then
    bisected within that bracket. This search does not establish global
    degree optimality or monotonicity of floating-point least-squares fits.
    The reference degree law is
    ``d = O(kappa log(kappa/eps))`` [CKS arXiv:1511.02306v2, Lemmas 17-19];
    the doubling search
    aborts loudly at ``8x`` the registered tripwire bound.

    Args:
        kappa: Domain parameter, greater than 1. QLS passes its
            ``polynomial_kappa``, described in the module docstring.
        epsilon_inv: Certificate tolerance in ``(0, 1)`` on the relative
            residual ``|kappa * x * P(x) - 1|``.
        max_degree: Largest admitted odd degree, checked before every fit.
        max_work: Cumulative known scalar-work envelope over actual candidates.
        max_bytes: Known simultaneous numerical arrays, excluding vendor workspace.
        candidate_degrees: Optional actual attempted-degree ledger, including failed candidates.
    Returns:
        InverseChebyshevFit with the selected coefficients, residual norming
        bound and grid density.

    Raises:
        ValueError: For out-of-range arguments, or when no degree below the
            runaway stop passes the certificate.
    """

    kappa = finite_real(kappa, "kappa")
    epsilon_inv = finite_real(epsilon_inv, "epsilon_inv")
    max_degree = integer(max_degree, "max_degree", 1)
    max_work = integer(max_work, "max_work", 1)
    if kappa <= 1.0:
        raise ValueError("kappa must exceed 1, so that the domain 1/kappa <= |x| <= 1 has positive width")
    if not 0.0 < epsilon_inv < 1.0:
        raise ValueError("epsilon_inv must be in the open interval (0, 1)")

    used_work = 0

    def _certificate_grid(degree: int) -> np.ndarray:
        """Admit one candidate, record its degree and return its certificate nodes.

        The ``25 d`` nodes are the Chebyshev zeros of ``[-1, 1]`` mapped
        affinely onto ``[1/kappa, 1]``.
        """
        nonlocal used_work
        cost = inverse_candidate_cost(degree)
        _check_bytes(cost.peak_bytes, max_bytes, "inverse polynomial fit")
        if used_work + cost.work > max_work:
            raise ValueError(
                f"inverse polynomial fit exceeds max_work={max_work}: candidate degree {degree} needs "
                f"{used_work + cost.work} work units including {used_work} for earlier candidates. "
                f"Raise max_work to at least {used_work + cost.work} to admit this candidate.")
        used_work += cost.work
        if candidate_degrees is not None:
            candidate_degrees.append(degree)
        points = QLS_FIT_CERTIFICATE_GRID_PER_DEGREE * degree
        lower = 1.0 / kappa
        return .5 * (1-lower) * chebyshev_grid(points) + .5 * (1+lower)

    runaway_stop = min(max_degree, _DOUBLING_FACTOR_OVER_TRIPWIRE * inverse_degree_law_bound(kappa, epsilon_inv))
    # Round down to an odd degree, because P approximates the odd function 1/x.
    runaway_stop -= (runaway_stop + 1) % 2
    coefficients, certificate, node_count = _fit_candidate(
        1, kappa=kappa, certificate_grid=_certificate_grid(1)
    )
    passing: tuple[int, np.ndarray, float, int] | None = (
        (1, coefficients, certificate, node_count) if certificate <= epsilon_inv else None
    )
    degree = 3
    last_failure = 1
    while passing is None and degree <= runaway_stop:
        coefficients, certificate, node_count = _fit_candidate(
            degree, kappa=kappa, certificate_grid=_certificate_grid(degree)
        )
        if certificate <= epsilon_inv:
            passing = (degree, coefficients, certificate, node_count)
            break
        last_failure = degree
        if degree == runaway_stop:
            break
        degree = min(2 * degree + 1, runaway_stop)
    if passing is None:
        raise ValueError(
            f"no tested odd degree through {runaway_stop} meets the residual norming bound target "
            f"|kappa*x*P(x) - 1| <= {epsilon_inv:.3g} at kappa = {kappa:.6g}; "
            f"max_degree={max_degree}. Inspect conditioning and intended work before increasing "
            "the cap; the requested tolerance may also exceed float64 fit resolution"
        )

    # Bisect odd degrees in (last failing, first passing].
    low = last_failure
    high = passing[0]
    while high - low > 2:
        middle = low + 2 * ((high - low) // 4)  # odd midpoint
        coefficients, certificate, node_count = _fit_candidate(
            middle, kappa=kappa, certificate_grid=_certificate_grid(middle)
        )
        if certificate <= epsilon_inv:
            passing = (middle, coefficients, certificate, node_count)
            high = middle
        else:
            low = middle

    degree, coefficients, certificate, node_count = passing
    return InverseChebyshevFit(
        coefficients=tuple(float(value) for value in coefficients),
        degree=int(degree),
        kappa=kappa,
        epsilon_inv=epsilon_inv,
        certificate=certificate,
        certificate_grid_points=int(QLS_FIT_CERTIFICATE_GRID_PER_DEGREE * degree),
        lsq_node_count=int(node_count),
    )


__all__ = [
    "InverseChebyshevFit",
    "QLS_FIT_CERTIFICATE_GRID_PER_DEGREE",
    "QLS_FIT_LSQ_NODE_MULTIPLIER",
    "QLS_INVERSE_DEGREE_LAW_FACTOR",
    "inverse_degree_law_bound",
]
