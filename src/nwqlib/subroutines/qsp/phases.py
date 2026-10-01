"""Wx-convention quantum signal processing: evaluation and phase factors.

[MRTC] is Martyn, Rossi, Tan, and Chuang, arXiv:2105.02859v5, [DMWL] is
Dong, Meng, Whaley, and Lin, arXiv:2002.11649v2, and [DLNW] is Dong, Lin,
Ni, and Wang, arXiv:2307.12468v1. Equation and section numbers refer to
these arXiv versions, those of [DLNW] to the PDF of that version.

Convention (project standard, [MRTC] arXiv:2105.02859v5, Sec. II.A,
Eqs. (1)-(3) and Theorem 1; App. A.1, Eq. (A2)): the signal operator is
``W(x) = [[x, i sqrt(1-x^2)], [i sqrt(1-x^2), x]]``, phases apply as
``e^{i phi Z}``, and a phase vector ``(phi_0, ..., phi_d)`` realizes

``P(x) = <0| e^{i phi_0 Z} prod_{j=1..d} [ W(x) e^{i phi_j Z} ] |0>``.

Conversion identities implemented here (each guarded by a unit test):

- Reflection convention: with
  ``R(x) = [[x, sqrt(1-x^2)], [sqrt(1-x^2), -x]]`` ([MRTC]
  arXiv:2105.02859v5, Eq. (A4)) and
  ``W(x) = i e^{-i pi/4 Z} R(x) e^{-i pi/4 Z}`` (the inverse of [MRTC]
  arXiv:2105.02859v5, Eq. (14)), the Wx product equals ``i^d`` times the
  reflection product with phases ``phi'_0 = phi_0 - pi/4``,
  ``phi'_d = phi_d - pi/4``, and interior ``phi'_j = phi_j - pi/2``.
  ``wx_phases_to_reflection`` returns those shifted phases plus the
  compensating global phase ``d * pi/2``. [MRTC] arXiv:2105.02859v5,
  App. A.2, Eq. (A5), uses the same interior and final shifts but sets
  ``phi'_0 = phi_0 + (2d-1) pi/4``, folding the ``i^d`` factor into
  the first phase. That form reproduces the ``<0|.|0>`` entry. Keeping the
  factor as a global phase makes the identity hold for the whole 2 x 2
  product.
- Qiskit sign: ``RZGate(theta) = exp(-i theta Z / 2)`` exactly, so applying
  ``e^{i phi Z}`` is ``RZGate(-2 phi)`` with no residual global phase. The
  circuit layer (``subroutines/qsp/evolution.py``) applies projector phases
  ``e^{i phi (2 Pi - 1)}`` as ``RZGate(+2 phi)`` on a signal qubit that is
  flipped onto the projector subspace, which is the same identity.

Phase factors follow the symmetric-QSP optimization of [DMWL]
arXiv:2002.11649v2: symmetric phase vectors ``phi_j = phi_{d-j}``
(Sec. III.1, Theorem 2, and Eq. (24)),
the mean-square residual of ``Re P(x)`` against the target polynomial on
positive Chebyshev nodes (Sec. III.2, Eq. (23)), the initialization
``(pi/4, 0, ..., 0, pi/4)`` of Eq. (27), which realizes ``Re P = 0``
and whose free phases ``(pi/4, 0, ..., 0)`` are Eq. (28) (Sec. III.4),
and L-BFGS descent (Sec. III.5, Algorithm 1). This module
departs from the paper in four places. The objective uses four times as
many nodes as free phases (``QSP_SOLVER_GRID_MULTIPLIER``). A damped
Gauss-Newton polish supplies the terminal digits after L-BFGS. If L-BFGS
from the Eq. (27) point leaves an objective-node residual at or above the
tolerance, the same damped iteration runs from that point, which is the
Newton method of [DLNW] arXiv:2307.12468v1, Sec. 3, Eq. (3.1) and
Algorithm 3.1. Their zero reduced phases are the [DMWL] start written in
an imaginary-part convention ([DLNW] Sec. 2.2 and Eq. (2.3)). Acceptance is decided on a
separate dense verification grid with the Chebyshev norming bound rather
than on the objective.

The Newton start exists for targets whose sup norm lies near 1. The
larger parity target of the Hamiltonian evolution in
``subroutines/qsp/evolution.py`` has a sup norm near ``1/(1 + margin)``
with a margin of ``1e-3``, and every QLS phase target has the norming
bound ``1/(1 + QLS_TARGET_MARGIN)``, with the same margin. [DMWL]
arXiv:2002.11649v2, Sec. IV.1, divides the Jacobi-Anger series by 2, so
their Hamiltonian-simulation tests of the fixed start stay at
``max|f| <= 1/2``, and Sec. IV.5 reports a Hessian condition number that
grows as ``max|f|`` approaches 1. [DLNW] arXiv:2307.12468v1, Sec. 2.2,
states that optimization methods such as L-BFGS lose their
convergence guarantee near ``max|f| = 1``, and Sec. 4.3 (Fig. 8) reports
Newton's method from the same start converging for ``1 - max|f|`` down to
``1e-9``. ``docs/ENGINEERING_CONSTANTS.md`` records the NWQLib sweeps
behind this order, for the evolution targets in its QSP phase-solver
section and for the QLS targets in the ``QLS_TARGET_MARGIN`` row. In both
families L-BFGS missed the tolerance on some sampled targets, and the
Newton start converged on each of them.

L-BFGS runs in ``scipy.optimize`` and each Newton step is one
``numpy.linalg.lstsq`` solve. Phase-factor computation
is classical preprocessing of ``poly(d)`` cost, independent of any
Hilbert-space dimension
(``classical_preprocessing_scope: "qsp_phase_factor_optimization"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
import scipy.optimize

from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes

# Engineering constants for the phase solver (registered in
# docs/ENGINEERING_CONSTANTS.md). Rationale:
# - GRID_MULTIPLIER 4: the symmetric problem is exactly determined on
#   ceil((d+1)/2) positive Chebyshev nodes; a 4x overdetermined grid removes
#   spurious fit-the-grid minima observed at 1x. Revisit if solver cost binds.
# - LBFGS ftol 1e-30 / gtol 1e-18: the objective is a mean SQUARE residual,
#   so hitting a 1e-12 residual needs objective ~1e-24; scipy's defaults stop
#   at ~1e-18 objective. maxiter 2000 covers the worst converged case (~1500).
#   Revisit when the residual tolerance or the objective changes.
# - POLISH_MAX_STEPS 30 with 15 halvings: damped Gauss-Newton supplies the
#   terminal quadratic digits after L-BFGS, and the same limits bound the
#   Newton start of [DLNW] arXiv:2307.12468v1. Steps that do not reduce the
#   max residual are
#   rejected, so each run can only improve its starting point. On the
#   default grid of the registered tau x epsilon sweep the Newton start
#   converged within 10 residual evaluations, including the evaluation at
#   the starting point.
#   Revisit if the polish or the Newton start stops above the residual
#   tolerance on a target that passes the max|f| <= 1 check.
# - RESIDUAL_TOLERANCE 1e-12: the stopping tolerance of [DMWL]
#   arXiv:2002.11649v2, Eq. (30),
#   applied here to the dense verification grid. It is a solver acceptance
#   control, not a bound on the error of any transformed operator. Revisit
#   for a different floating-point precision.
# - TARGET_BOUNDARY_ULPS 256: admission band above max|f| = 1 for the
#   binary64 evaluation of the target's norming bound. A target whose bound
#   exceeds one by more than this band is rejected before optimization.
#   Revisit if the norming implementation or the supported precision changes.
QSP_SOLVER_GRID_MULTIPLIER = 4
QSP_SOLVER_LBFGS_OPTIONS: dict[str, Any] = {"maxiter": 2000, "ftol": 1e-30, "gtol": 1e-18}
QSP_SOLVER_POLISH_MAX_STEPS = 30
QSP_SOLVER_POLISH_MAX_HALVINGS = 15
QSP_SOLVER_RESIDUAL_TOLERANCE = 1.0e-12
QSP_TARGET_BOUNDARY_ULPS = 256.0


class _QSPPhaseConvergenceError(ValueError):
    """A valid QSP target failed the final phase-residual acceptance check."""

    def __init__(self, message: str, *, evaluations: int):
        super().__init__(message)
        self.evaluations = evaluations


class _QSPEvaluationsExhausted(ValueError):
    """The phase solver used up the evaluation allowance it was given."""


def _real_coefficients(values: Any, *, max_degree: int, max_bytes: int) -> np.ndarray:
    """Admit the finite real vector before any polynomial or optimizer work."""
    max_degree = integer(max_degree, "max_degree", 0)
    try:
        count = len(values)
    except TypeError:
        raise ValueError("QSP targets require a coefficient vector") from None
    if count == 0 or count - 1 > max_degree:
        raise ValueError(f"QSP target degree must be between zero and max_degree={max_degree}")
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "iufc":
            raise ValueError("QSP targets require a numeric coefficient vector")
    elif not all(isinstance(value, Number) for value in values):
        raise ValueError("QSP targets require a numeric coefficient vector")
    # Conversion can hold the complex source and its real floating copy.
    _check_bytes(24 * count, max_bytes, "QSP coefficient conversion")
    source = np.asarray(values)
    if source.ndim != 1 or source.dtype.kind not in "iufc":
        raise ValueError("QSP targets require a numeric coefficient vector")
    if np.any(source.imag != 0):
        raise ValueError("QSP targets must have real Chebyshev coefficients")
    coefficients = np.asarray(source.real, dtype=float)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("QSP targets must have finite Chebyshev coefficients")
    return coefficients


class _PhaseEvaluations:
    """One solver's actual objective/Jacobian and verification evaluations."""

    def __init__(self, maximum: int, limit_name: str = "max_evaluations"):
        self.maximum = integer(maximum, limit_name, 1)
        self.limit_name = limit_name
        self.used = 0

    def consume(self) -> None:
        if self.used >= self.maximum:
            raise _QSPEvaluationsExhausted(f"QSP phase solving exhausted {self.limit_name}={self.maximum}")
        self.used += 1


def chebyshev_grid(num_points: int) -> np.ndarray:
    """Return the Chebyshev grid ``x_j = cos(pi (j + 1/2) / M)``."""

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    return np.cos(np.pi * (np.arange(num_points) + 0.5) / num_points)


def chebyshev_norming_sup_bound(
    grid_max: float,
    *,
    degree: int,
    num_points: int,
) -> float:
    """Bound a degree-``degree`` polynomial's sup norm from a Chebyshev grid.

    Returns ``grid_max / cos(pi d / (2 N))``, an upper bound on
    ``sup_{x in [-1, 1]} |p(x)|`` when ``grid_max`` is the largest ``|p|``
    on the ``N`` nodes of ``chebyshev_grid(N)``. This is the Chebyshev-node
    norming bound of Ehlich and Zeller (1964), doi:10.1007/BF01111276,
    Satz 2, Eqs. (12)-(14), pp. 42-43, which use exactly these
    Chebyshev-zero nodes.
    The extension to complex coefficients follows by rotating a maximum
    value to the real axis and applying the real-polynomial inequality.
    Sunderhauf et al., arXiv:2507.15537v1 Eq. (25), state this factor for
    equidistant x points, where it is not valid. Equispaced angles give
    the required nodes x_j=cos(theta_j), not an equispaced x grid.
    Ehlich--Zeller, doi:10.1007/BF01111276, Satz 1, treats equidistant x
    nodes separately. The QLS
    guide gives an exact-rational counterexample to the Eq. (25) extension.

    ``num_points`` is the number ``N`` of nodes on the whole interval
    ``[-1, 1]``, as opposed to the positive-half grid of the phase solver.
    The inequality requires strictly more nodes than the polynomial degree.
    The result is an exact-arithmetic bound evaluated in binary64, not an
    interval-arithmetic enclosure.
    """

    value = float(grid_max)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("grid_max must be a finite nonnegative number")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if num_points <= degree:
        raise ValueError("Chebyshev norming requires num_points > degree")
    denominator = float(np.cos(degree * np.pi / (2.0 * num_points)))
    return value / denominator


def _chebyshev_polynomial_norming_data(
    chebyshev_coefficients: Any,
    *, max_degree: int = 256, max_bytes: int = DEFAULT_INPUT_BYTES,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(grid, values, bound)`` for a real Chebyshev series ``f``.

    ``grid`` holds the ``N = 64 (d + 1)`` Chebyshev zeros of
    ``chebyshev_grid(N)`` on ``[-1, 1]``, ``values`` the target
    ``f(x_j) = sum_k c_k T_k(x_j)`` at those nodes, and ``bound`` the
    Ehlich-Zeller upper bound on ``sup_{[-1, 1]} |f|`` from
    ``chebyshev_norming_sup_bound``. The phase solver keeps ``grid`` and
    ``values`` as its final verification grid, so its admission bound and
    its acceptance residual are evaluated on the same nodes.
    """

    coefficients = _real_coefficients(chebyshev_coefficients, max_degree=max_degree, max_bytes=max_bytes)
    degree = coefficients.size - 1
    # At N = 64(d+1), the norming-factor inflation is at most
    # sec(pi/128)-1 ~= 3.01e-4: rigorous, but small compared with the
    # registered 1e-3 target margin used by the evolution and QLS callers.
    points = 64 * (degree + 1)
    # 8*(6N + d + 1) bytes allow six float64 vectors of N entries, for the
    # nodes and the vectors of NumPy's Clenshaw recurrence in chebval (2x, the
    # two running coefficient vectors, the previous one and a product
    # temporary), and chebval's copy of the d + 1 coefficients.
    _check_bytes(8 * (6 * points + degree + 1), max_bytes, "QSP norming grid")
    grid = chebyshev_grid(points)
    values = np.polynomial.chebyshev.chebval(grid, coefficients)
    bound = chebyshev_norming_sup_bound(
        float(np.max(np.abs(values))),
        degree=degree,
        num_points=grid.size,
    )
    return grid, values, bound


def chebyshev_polynomial_sup_bound(chebyshev_coefficients: Any, *, max_degree: int = 256,
                                  max_bytes: int = DEFAULT_INPUT_BYTES) -> float:
    """Return an upper bound on ``sup_{x in [-1, 1]} |f(x)|`` for ``f = sum_k c_k T_k``.

    The series is sampled on ``64 (d + 1)`` Chebyshev zeros, and the sampled
    maximum is divided by ``cos(pi d / (2N))`` as in
    ``chebyshev_norming_sup_bound``. The QLS rescale, the evolution target
    scale and the admission check of ``solve_symmetric_qsp_phases`` all use
    this bound. ``max_degree`` and ``max_bytes`` are checked before the grid
    is formed.
    """

    return _chebyshev_polynomial_norming_data(chebyshev_coefficients, max_degree=max_degree, max_bytes=max_bytes)[2]


def _signal_matrices(x_values: np.ndarray) -> np.ndarray:
    """Return the stacked Wx signal matrices ``W(x)`` for a grid.

    The off-diagonal entry uses ``sqrt((1 - x)(1 + x))``. For binary64 ``x``
    in ``[1/2, 1]``, ``1 - x`` is exact (Sterbenz lemma), so the product has a
    relative error of a few ulps of ``1 - x^2``. Forming ``1 - x**2`` instead
    leaves an absolute error of about ``u = 2**-53``, the binary64 unit
    roundoff, in ``s^2 = 1 - x^2``, which acts like a shift of ``x`` by
    ``u/2``. The ``<0|.|0>`` entry of a degree-``d`` product depends on ``s``
    only through ``s^2``, and Markov's polynomial inequality
    ``|P'| <= d^2`` for ``|P| <= 1`` bounds the resulting endpoint error by
    about ``d^2 u / 2``. That is ``3.6e-12`` at ``d = 255``, above the
    solver's ``1e-12`` acceptance tolerance. Near ``x = -1`` the factor
    ``1 + x`` is the exact one.
    """

    signal = np.zeros((x_values.size, 2, 2), dtype=complex)
    root = np.sqrt(np.clip((1.0 - x_values) * (1.0 + x_values), 0.0, None))
    signal[:, 0, 0] = x_values
    signal[:, 1, 1] = x_values
    signal[:, 0, 1] = 1j * root
    signal[:, 1, 0] = 1j * root
    return signal


def _phase_matrices(phi: float, count: int) -> np.ndarray:
    """Return stacked ``e^{i phi Z}`` matrices."""

    phase = np.zeros((count, 2, 2), dtype=complex)
    phase[:, 0, 0] = np.exp(1j * phi)
    phase[:, 1, 1] = np.exp(-1j * phi)
    return phase


def evaluate_qsp_polynomial(phases: Any, x_values: Any) -> np.ndarray:
    """Evaluate the Wx-convention QSP polynomial ``P(x)`` on a grid.

    Args:
        phases: Phase vector ``(phi_0, ..., phi_d)``.
        x_values: Signal values in ``[-1, 1]``.

    Returns:
        Complex ``P(x)`` values, one per grid point.
    """

    phase_array = np.asarray(phases, dtype=float).reshape(-1)
    if phase_array.size == 0:
        raise ValueError("QSP needs at least one phase")
    grid = np.asarray(x_values, dtype=float).reshape(-1)
    signal = _signal_matrices(grid)
    product = _phase_matrices(float(phase_array[0]), grid.size)
    for phi in phase_array[1:]:
        product = product @ signal @ _phase_matrices(float(phi), grid.size)
    return product[:, 0, 0]


def wx_phases_to_reflection(phases: Any) -> tuple[np.ndarray, float]:
    """Convert Wx phases to reflection-convention phases ([MRTC] arXiv:2105.02859v5, Eq. (14)).

    The shifts are those of [MRTC] arXiv:2105.02859v5, App. A.2, Eq. (A5),
    except that the
    ``i^d`` factor is returned as a global phase (see the module docstring).

    Returns:
        ``(reflection_phases, global_phase)`` such that the reflection-form
        product times ``e^{i global_phase}`` equals the Wx-form product;
        ``global_phase = d * pi / 2`` compensates the ``i^d`` factor.
    """

    converted = np.asarray(phases, dtype=float).copy().reshape(-1)
    if converted.size < 2:
        raise ValueError("the reflection conversion needs degree >= 1")
    degree = converted.size - 1
    converted[0] -= np.pi / 4.0
    converted[-1] -= np.pi / 4.0
    if degree >= 2:
        converted[1:-1] -= np.pi / 2.0
    return converted, degree * np.pi / 2.0


@dataclass(frozen=True, kw_only=True)
class SymmetricQSPPhases:
    """Solved symmetric-QSP phase factors and solver diagnostics.

    Args:
        phases: Full symmetric Wx phase vector ``(phi_0, ..., phi_d)``.
        degree: Polynomial degree ``d``.
        parity: Target parity (``0`` even, ``1`` odd).
        target_chebyshev_coefficients: The solved-for coefficient vector.
        max_residual: Max ``|Re P - f|`` on the dense verification grid.
        residual_sup_bound: Chebyshev-grid norming bound on
            ``sup_{x in [-1,1]} |Re P(x) - f(x)|``.
        evaluations: Actual objective/Jacobian and final verification calls.
    """

    phases: tuple[float, ...]
    degree: int
    parity: int
    target_chebyshev_coefficients: tuple[float, ...]
    max_residual: float
    residual_sup_bound: float
    evaluations: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like phase-solution record."""

        return {
            "phases": list(self.phases),
            "degree": self.degree,
            "parity": self.parity,
            "target_chebyshev_coefficients": list(self.target_chebyshev_coefficients),
            "max_residual": self.max_residual,
            "residual_sup_bound": self.residual_sup_bound,
            "evaluations": self.evaluations,
        }


def _symmetric_problem(coefficients: np.ndarray, *, evaluations: _PhaseEvaluations):
    """Return ``(residual_and_jacobian, num_free, full_from_free)`` for [DMWL] arXiv:2002.11649v2, Eq. (23).

    For degree ``d``, the ``num_free = ceil((d+1)/2)`` free phases
    ``theta_k`` expand to the symmetric vector ``phi_j = theta_{min(j, d-j)}``
    of [DMWL] arXiv:2002.11649v2, Eq. (24) (``full_from_free``). The
    residual nodes are the
    ``M`` positive zeros ``x_m = cos((2m - 1) pi / (4M))``, ``m = 1..M``, of
    ``T_{2M}``. [DMWL] arXiv:2002.11649v2, Eq. (23), uses ``M = num_free``.
    This module uses
    ``M = QSP_SOLVER_GRID_MULTIPLIER * num_free``. Parity makes the negative
    nodes redundant.

    ``residual_and_jacobian(theta)`` returns ``r_m = Re P(x_m) - f(x_m)``,
    shape ``(M,)``, and the exact Jacobian ``dr_m/dtheta_k``, shape
    ``(M, num_free)``. Write ``U = F_0 F_1 ... F_d`` with
    ``F_0 = e^{i phi_0 Z}`` and ``F_j = W(x) e^{i phi_j Z}``, so
    ``P = <0|U|0>``. Since ``Z`` commutes with ``e^{i phi_j Z}``,
    ``dF_j/dphi_j = W(x) (iZ) e^{i phi_j Z}``, and
    ``dU/dphi_j = (F_0 ... F_{j-1}) W(x) (iZ) e^{i phi_j Z} (F_{j+1} ... F_d)``.
    Stored prefix and suffix products give all ``d + 1`` columns with a
    constant number of 2 x 2 products each, ``O(d)`` per node. For
    ``j = 0``, ``dU/dphi_0 = iZ U`` and ``<0| iZ = i <0|`` give
    ``dP/dphi_0 = iP``. By the chain rule through
    ``phi_j = theta_{min(j, d-j)}``, columns ``j`` and ``d - j`` of the full
    Jacobian add into free column ``min(j, d-j)``, and the middle column of
    an even degree ``d`` enters once.
    """

    degree = coefficients.size - 1
    num_free = (degree + 2) // 2
    num_points = QSP_SOLVER_GRID_MULTIPLIER * num_free
    # Positive-half Chebyshev nodes of [DMWL] arXiv:2002.11649v2; parity makes
    # x < 0 redundant.
    grid = np.cos((2.0 * np.arange(1, num_points + 1) - 1.0) * np.pi / (4.0 * num_points))
    target = np.polynomial.chebyshev.chebval(grid, coefficients)
    signal = _signal_matrices(grid)
    i_z = np.array([[1j, 0.0], [0.0, -1j]])
    identity = np.broadcast_to(np.eye(2, dtype=complex), (num_points, 2, 2))

    def full_from_free(free: np.ndarray) -> np.ndarray:
        """Expand free phases to the symmetric vector of [DMWL] arXiv:2002.11649v2, Eq. (24)."""
        return np.array([free[min(j, degree - j)] for j in range(degree + 1)])

    def residual_and_jacobian(free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(r, dr/dtheta)`` at free phases ``free``, consuming one evaluation."""
        evaluations.consume()
        full = full_from_free(free)
        phase_stack = [_phase_matrices(float(phi), num_points) for phi in full]
        factors = [phase_stack[0]] + [signal @ phase_stack[j] for j in range(1, degree + 1)]
        # prefix[j] = F_0 ... F_{j-1} and suffix[j] = F_j ... F_d, so
        # prefix[d+1] is U.
        prefix = [identity]
        for factor in factors:
            prefix.append(prefix[-1] @ factor)
        suffix = [identity] * (degree + 2)
        for j in range(degree, -1, -1):
            suffix[j] = factors[j] @ suffix[j + 1]
        realized = prefix[degree + 1][:, 0, 0]
        residual = realized.real - target
        jacobian_full = np.empty((num_points, degree + 1))
        # dP/dphi_0 = i P (the leading e^{i phi Z} differentiates in place).
        jacobian_full[:, 0] = (1j * realized).real
        for j in range(1, degree + 1):
            derivative = prefix[j] @ signal @ i_z @ (phase_stack[j] @ suffix[j + 1])
            jacobian_full[:, j] = derivative[:, 0, 0].real
        # Chain rule for phi_j = theta_{min(j, d-j)}.
        jacobian = np.zeros((num_points, num_free))
        for j in range(degree + 1):
            jacobian[:, min(j, degree - j)] += jacobian_full[:, j]
        return residual, jacobian

    return residual_and_jacobian, num_free, full_from_free


def _max_abs_residual(residual: np.ndarray) -> float:
    """Return ``max_m |r_m|``, or ``inf`` when some ``r_m`` is not finite.

    ``np.max`` propagates NaN, and every comparison with NaN is False. A
    NaN maximum would make ``best >= residual_tolerance`` False, which the
    solver reads as converged, and ``candidate_best < best`` could never
    replace it. As ``inf``, a nonfinite residual counts as not converged and
    ranks above every finite one.
    """

    value = float(np.max(np.abs(residual)))
    return value if np.isfinite(value) else np.inf


def _solve_from(residual_and_jacobian, start: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimize from one start and return ``(theta, max_m |r_m|)`` on the objective nodes.

    L-BFGS minimizes the [DMWL] arXiv:2002.11649v2, Eq. (23), mean square
    ``L = mean(r^2)``,
    whose gradient is ``(2/M) J^T r``. ``_damped_newton`` then polishes the
    L-BFGS point, so the result is never worse than that point. The returned
    maximum is taken on the ``M`` objective nodes, and it is ``inf`` when
    L-BFGS ends at a nonfinite point. The caller decides acceptance on the
    separate verification grid.
    """

    def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
        residual, jacobian = residual_and_jacobian(free)
        return float(np.mean(residual**2)), (2.0 / residual.size) * (jacobian.T @ residual)

    result = scipy.optimize.minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options=dict(QSP_SOLVER_LBFGS_OPTIONS),
    )
    return _damped_newton(residual_and_jacobian, result.x)


def _damped_newton(residual_and_jacobian, free: np.ndarray) -> tuple[np.ndarray, float]:
    """Run damped Newton steps from ``free`` and return ``(theta, max_m |r_m|)``.

    Each step solves ``J s = r`` in the least-squares sense and halves ``s``
    until the maximum residual on the objective nodes decreases, so the
    result is never worse than ``free``. The residual is
    ``r = V (beta(theta) - alpha)``, where ``alpha`` and ``beta`` hold the
    ``num_free`` Chebyshev coefficients of the target parity of ``f`` and
    ``Re P``, and ``V`` holds those Chebyshev polynomials at the ``M``
    positive nodes. ``V`` has full column rank. A coefficient vector in its
    null space gives a polynomial of that parity that vanishes at the ``M``
    positive nodes and, by parity, at their negatives. These ``2 M > d``
    roots force the polynomial, and hence the vector, to be zero. The
    least-squares step therefore solves
    ``(d beta / d theta) s = beta - alpha`` exactly whenever that square
    Jacobian is invertible, which is the Newton update of [DLNW]
    arXiv:2307.12468v1, Eq. (3.1). [DLNW] arXiv:2307.12468v1, Algorithm 3.1,
    takes full steps. The halving here only rejects a step that would not
    decrease the maximum residual.

    At finite phases ``P`` is an entry of a product of unitary matrices and
    the target is finite, so the residual is nonfinite only at nonfinite
    phases, and every step from such a point is nonfinite as well. A
    nonfinite start is therefore returned unchanged with maximum ``inf``.
    Its nonfinite Jacobian is not passed to ``np.linalg.lstsq``, where
    LAPACK prints a DLASCL argument error (NumPy 2.5.2 on macOS arm64)
    before NumPy raises ``LinAlgError``.
    """

    residual, jacobian = residual_and_jacobian(free)
    best = _max_abs_residual(residual)
    for _ in range(QSP_SOLVER_POLISH_MAX_STEPS):
        # Registered stop below the 1e-12 acceptance tolerance, for the
        # polish and the Newton start alike (docs/ENGINEERING_CONSTANTS.md,
        # "Numerical choices"). A nonfinite start has no finite step (see
        # the docstring).
        if best < 1.0e-14 or not np.isfinite(best):
            break
        try:
            step = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        damping, improved = 1.0, False
        for _ in range(QSP_SOLVER_POLISH_MAX_HALVINGS):
            candidate = free - damping * step
            candidate_residual, candidate_jacobian = residual_and_jacobian(candidate)
            candidate_best = _max_abs_residual(candidate_residual)
            if candidate_best < best:
                free, residual, jacobian = candidate, candidate_residual, candidate_jacobian
                best, improved = candidate_best, True
                break
            damping *= 0.5
        if not improved:
            break
    return free, best


def solve_symmetric_qsp_phases(
    chebyshev_coefficients: Any,
    *,
    residual_tolerance: float = QSP_SOLVER_RESIDUAL_TOLERANCE,
    max_degree: int = 256,
    max_evaluations: int = 20000,
    max_bytes: int = DEFAULT_INPUT_BYTES,
    limit_name: str = "max_evaluations",
) -> SymmetricQSPPhases:
    """Solve symmetric Wx phase factors for a real definite-parity target.

    Implements the [DMWL] optimization (arXiv:2002.11649v2, Sec. III):
    symmetric phases, positive
    Chebyshev node objective, ``(pi/4, 0, ..., 0, pi/4)`` initialization and
    L-BFGS, followed by a damped Gauss-Newton polish. If the objective-node
    residual from that start is at or above ``residual_tolerance`` or
    nonfinite, the damped Newton iteration of [DLNW] arXiv:2307.12468v1 runs
    from the same
    initialization, and the better of the two results is verified. The
    returned phases satisfy ``Re P(x) ~= f(x)`` in the Wx convention.
    ``residual_sup_bound`` bounds
    ``sup |Re P - f|`` over ``[-1, 1]`` by the Chebyshev norming inequality.

    Args:
        chebyshev_coefficients: Real Chebyshev coefficients ``(c_0..c_d)`` of
            the target ``f``; entries of the wrong parity must be zero and
            ``max |f|`` must not exceed 1.
        residual_tolerance: Max acceptable ``|Re P - f|`` on the dense
            verification grid.
        max_degree: Largest admitted target degree.
        max_evaluations: Combined objective/Jacobian and final verification
            calls across both starts, polish steps and line searches.
        max_bytes: Bound on known simultaneous numerical arrays. This does
            not measure process RSS, vendor workspaces or elapsed time.
        limit_name: Name of the caller's option that sets ``max_evaluations``,
            used in the exhaustion and validation messages.
    Returns:
        SymmetricQSPPhases with the full phase vector and diagnostics.

    Raises:
        ValueError: For empty/mixed-parity/out-of-range targets, or when the
            solver cannot reach ``residual_tolerance`` (with diagnostics).
    """

    coefficients = _real_coefficients(chebyshev_coefficients, max_degree=max_degree, max_bytes=max_bytes)
    residual_tolerance = finite_real(residual_tolerance, "residual_tolerance")
    if residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be positive")
    evaluations = _PhaseEvaluations(max_evaluations, limit_name)
    if coefficients.size < 2:
        raise ValueError("the phase solver needs a target of degree >= 1")
    degree = coefficients.size - 1
    parity = degree % 2
    off_parity = coefficients[(1 - parity) :: 2]
    if off_parity.size and float(np.max(np.abs(off_parity))) > 0.0:
        raise ValueError(
            "QSP targets must have definite parity: coefficients of parity "
            f"{1 - parity} must be exactly zero for a degree-{degree} target"
        )
    num_free = (degree + 2) // 2
    points = QSP_SOLVER_GRID_MULTIPLIER * num_free
    # Peak live arrays of one solve, checked before the optimization problem
    # is allocated, with M = points objective nodes, N = 64(d+1)
    # verification nodes and u = num_free. A stack of 2 x 2 complex128
    # matrices costs 64 bytes per node.
    # - kept_bytes, alive throughout: 16N for the verification nodes and
    #   their target values. 160M for the objective nodes, their targets and
    #   the W(x) stack (80 bytes per node together) plus the per-call P and
    #   residual vectors (24 more), which leaves 56 bytes per node to spare.
    #   128(d+1) for the coefficient, phase and optimizer vectors.
    # - phase_bytes, one residual_and_jacobian call: 640(d+2)M allows ten
    #   64-byte stack entries per node for each of the d + 2 prefix
    #   positions. The phase, factor, prefix and suffix stacks take four of
    #   them, and the products of one derivative column and the unfolded
    #   Jacobian fit in the rest.
    #   16Mu holds the current and the candidate folded Jacobian.
    # - polish_bytes, one Gauss-Newton least-squares step: 32Mu for Jacobian
    #   copies, 32u^2 for u x u factors and 16M for two residual vectors.
    #   LAPACK workspace is not counted.
    # - verification_bytes: 320N for the five stacks of
    #   evaluate_qsp_polynomial (W(x), running product, one phase stack and
    #   two product temporaries).
    # Optimization and final verification do not overlap, hence the max.
    kept_bytes = 16 * 64 * (degree + 1) + 160 * points + 128 * (degree + 1)
    phase_bytes = 640 * (degree + 2) * points + 16 * points * num_free
    polish_bytes = 32 * points * num_free + 32 * num_free**2 + 16 * points
    verification_bytes = 320 * 64 * (degree + 1)
    _check_bytes(kept_bytes + max(phase_bytes + polish_bytes, verification_bytes),
                 max_bytes, "QSP phase solving")
    verification_grid, wanted, target_sup_bound = _chebyshev_polynomial_norming_data(
        coefficients, max_degree=max_degree, max_bytes=max_bytes
    )
    # Registered under QSP target-boundary ULP policy in
    # docs/ENGINEERING_CONSTANTS.md.
    target_boundary_width = (
        QSP_TARGET_BOUNDARY_ULPS
        * np.finfo(float).eps
        * max(1.0, target_sup_bound)
    )
    if target_sup_bound > 1.0 + target_boundary_width:
        raise ValueError(
            "QSP targets must satisfy max|f| <= 1; rescale the target and "
            "record the factor (see subroutines/qsp/evolution.py)"
        )

    residual_and_jacobian, num_free, full_from_free = _symmetric_problem(coefficients, evaluations=evaluations)
    start = np.zeros(num_free)
    # [DMWL] arXiv:2002.11649v2, Eq. (28): the free phases (pi/4, 0, ..., 0)
    # of the full start (pi/4, 0, ..., 0, pi/4) of Eq. (27), which realizes
    # Re P = 0.
    start[0] = np.pi / 4.0
    free, best = _solve_from(residual_and_jacobian, start)
    # _max_abs_residual reports a nonfinite L-BFGS point as best = inf, so
    # the Newton start below also runs for it, and any finite Newton result
    # replaces it.
    if best >= residual_tolerance:
        # Near max|f| = 1, L-BFGS from this start has no convergence
        # guarantee ([DLNW] arXiv:2307.12468v1, Sec. 2.2). Newton's method from
        # the same start ([DLNW] arXiv:2307.12468v1, Algorithm 3.1) is the
        # second attempt.
        candidate, candidate_best = _damped_newton(residual_and_jacobian, start)
        if candidate_best < best:
            free, best = candidate, candidate_best
        attempts = "the L-BFGS and Newton starts"
    else:
        attempts = "the L-BFGS start"

    full = full_from_free(free)
    evaluations.consume()
    realized = evaluate_qsp_polynomial(full, verification_grid).real
    verification_residual = realized - wanted
    max_residual = float(np.max(np.abs(verification_residual)))
    # Written as `not (residual <= tol)` so that a NaN residual, for which
    # every comparison is False, is rejected rather than accepted.
    if not (max_residual <= residual_tolerance):
        if not np.isfinite(max_residual):
            raise _QSPPhaseConvergenceError(
                "symmetric-QSP phase optimization broke down: the verification "
                f"residual is nonfinite at degree {degree} after {attempts}, "
                "indicating optimizer overflow on this target. "
                "Rescale the target coefficients below max|f| = 1 with a "
                "larger margin, or reduce the degree.", evaluations=evaluations.used,
            )
        raise _QSPPhaseConvergenceError(
            "symmetric-QSP phase optimization did not converge: max residual "
            f"{max_residual:.3e} > {residual_tolerance:.1e} at degree {degree} "
            f"after {attempts}. Near-unit-sup targets are the hard regime. "
            "Rescale the target with a larger recorded margin.", evaluations=evaluations.used,
        )
    residual_sup_bound = chebyshev_norming_sup_bound(
        max_residual,
        degree=degree,
        num_points=verification_grid.size,
    )
    return SymmetricQSPPhases(
        phases=tuple(float(value) for value in full),
        degree=degree,
        parity=parity,
        target_chebyshev_coefficients=tuple(float(value) for value in coefficients),
        max_residual=max_residual,
        residual_sup_bound=residual_sup_bound,
        evaluations=evaluations.used,
    )
__all__ = [
    "QSP_SOLVER_RESIDUAL_TOLERANCE",
    "SymmetricQSPPhases",
    "chebyshev_grid",
    "chebyshev_norming_sup_bound",
    "chebyshev_polynomial_sup_bound",
    "evaluate_qsp_polynomial",
    "solve_symmetric_qsp_phases",
    "wx_phases_to_reflection",
]
