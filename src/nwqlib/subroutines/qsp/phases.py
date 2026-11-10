"""Real Wx and complex Laurent phase synthesis with canonical inverse NLFT.

The common solver uses Weiss ratio FFT, causal-prefix divide-and-conquer,
small peeling leaves, and a fresh balanced reconstruction from the returned
angles. The reconstructed coefficient one-norm is numerical evidence for
uniform error. FFT, transcendental and reconstruction arithmetic have no
complete enclosure, so the result is not a floating-point certificate.

Wx uses W(x)=[[x,i sqrt(1-x²)],[i sqrt(1-x²),x]] and phases exp(i phi Z).
General Laurent synthesis uses A0 D ... D Ad iX, D=diag(z,1), with the
first ell signals replaced by diag(1,z^-1). The adapters differ at their
endpoints. See docs/algorithms/qsp_phase_derivation.md for the derivation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Number
import operator
from typing import Any

import numpy as np
from scipy.fft import dct

from nwqlib._limits import ResourceLimitError, memory_limit
from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from . import _nlft

QSP_SOLVER_RESIDUAL_TOLERANCE = 1.0e-12


class _QSPPhaseConvergenceError(ValueError):
    """The fixed target failed its residual acceptance before the FFT doubling stopped."""

    def __init__(self, message: str, *, evidence=None):
        super().__init__(message)
        self.evidence = evidence


class _QSPPhaseLimitError(_QSPPhaseConvergenceError, ResourceLimitError):
    """The doubling stopped because the next FFT exceeds the memory limit or an explicit ``max_fft``.

    The attempts failed their residual acceptance, and ``evidence`` records them.
    Whether a longer FFT would reach the target stays unknown. A stop by an
    ideal-arithmetic rule is the plain ``_QSPPhaseConvergenceError``, even when a
    limit would also block the next attempt, because that attempt would not run.
    """


def _real_coefficients(values: Any, *, max_bytes=None) -> np.ndarray:
    try:
        count = len(values)
    except TypeError:
        raise ValueError("QSP targets require a coefficient vector") from None
    if count == 0:
        raise ValueError("QSP targets require a nonempty coefficient vector")
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "iufc":
            raise ValueError("QSP targets require a numeric coefficient vector")
    elif not all(isinstance(value, Number) for value in values):
        raise ValueError("QSP targets require a numeric coefficient vector")
    _check_bytes(24 * count, max_bytes, "QSP coefficient conversion")
    source = np.asarray(values)
    if source.ndim != 1 or source.dtype.kind not in "iufc" or np.any(source.imag != 0):
        raise ValueError("QSP targets must have real Chebyshev coefficients")
    coefficients = np.asarray(source.real, dtype=float)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("QSP targets must have finite Chebyshev coefficients")
    return coefficients


def chebyshev_grid(num_points: int) -> np.ndarray:
    """Return x_j=cos(pi(j+1/2)/N), j=0,...,N-1, for positive N=num_points."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    return np.cos(np.pi * (np.arange(num_points) + 0.5) / num_points)


def chebyshev_norming_sup_bound(grid_max: float, *, degree: int, num_points: int) -> float:
    """Evaluate the Chebyshev-zero norming bound max|p(x_j)|/cos(pi*d/(2N)).

    The polynomial degree d must be nonnegative and N>d. Ehlich and Zeller
    (1964), doi:10.1007/BF01111276, Satz 2, give this exact-arithmetic
    inequality. Its binary64 evaluation here is not an arithmetic enclosure.
    The complex-polynomial extension rotates a maximum to the real axis.
    """
    value = float(grid_max)
    if not math.isfinite(value) or value < 0:
        raise ValueError("grid_max must be finite and nonnegative")
    if degree < 0 or num_points <= degree:
        raise ValueError("Chebyshev norming requires 0 <= degree < num_points")
    return value / math.cos(degree * math.pi / (2 * num_points))


def _values_at_zeros(coefficients, points):
    """Evaluate the Chebyshev series ``coefficients`` at the ``points`` zeros of `chebyshev_grid`.

    DCT-III is x0+2 sum_k xk cos(pi*k*(j+1/2)/N), so the input c_0, c_k/2
    gives sum_k c_k T_k(x_j) in O(N log N) work, not Clenshaw's O(Nd).
    Requires len(coefficients) <= points.
    """
    padded = np.zeros(points)
    padded[:len(coefficients)] = coefficients / 2
    padded[0] = coefficients[0]
    return dct(padded, type=3)


def _chebyshev_norm_bound(coefficients, max_bytes):
    degree = len(coefficients) - 1
    if degree == 0:
        return abs(float(coefficients[0]))
    points = 64 * (degree + 1)
    _check_bytes(8 * (6 * points + degree + 1), max_bytes, "QSP norming grid")
    values = _values_at_zeros(coefficients, points)
    return chebyshev_norming_sup_bound(float(np.max(abs(values))), degree=degree, num_points=points)


def chebyshev_polynomial_sup_bound(chebyshev_coefficients: Any, *, max_bytes=None) -> float:
    """Estimate a real Chebyshev polynomial's norming upper bound on [-1,1].

    ``chebyshev_coefficients`` contains c_0,...,c_d in ascending order.
    The Chebyshev-zero samples use a DCT with N=64(d+1), followed by the
    Ehlich-Zeller factor. The inequality is exact in real arithmetic, but
    this floating-point evaluation has no enclosure. ``max_bytes=None``
    checks the grid arrays against the physical memory.
    """
    return _chebyshev_norm_bound(_real_coefficients(chebyshev_coefficients, max_bytes=max_bytes), max_bytes)


def _signal_matrices(x_values):
    signal = np.zeros((x_values.size, 2, 2), complex)
    root = np.sqrt(np.clip((1.0 - x_values) * (1.0 + x_values), 0.0, None))
    signal[:, 0, 0] = signal[:, 1, 1] = x_values
    signal[:, 0, 1] = signal[:, 1, 0] = 1j * root
    return signal


def _phase_matrices(phi, count):
    phase = np.zeros((count, 2, 2), complex)
    phase[:, 0, 0] = np.exp(1j * phi)
    phase[:, 1, 1] = np.exp(-1j * phi)
    return phase


def evaluate_qsp_polynomial(phases: Any, x_values: Any) -> np.ndarray:
    """Evaluate P(x)=<0|exp(i phi0 Z) W(x)...W(x)exp(i phid Z)|0>.

    ``phases`` is the nonempty Wx phase vector and ``x_values`` are points
    in [-1,1]. Returns complex values, whose real part is the real target.
    Pointwise evaluation costs O(number of points times degree).
    """
    phases = np.asarray(phases, float).reshape(-1)
    if not len(phases):
        raise ValueError("QSP needs at least one phase")
    x = np.asarray(x_values, float).reshape(-1)
    signal = _signal_matrices(x)
    product = _phase_matrices(phases[0], len(x))
    for phi in phases[1:]:
        product = product @ signal @ _phase_matrices(phi, len(x))
    return product[:, 0, 0]


def wx_phases_to_reflection(phases: Any) -> tuple[np.ndarray, float]:
    """Convert Wx phases to reflection phases and the compensating global phase.

    For R=[[x,sqrt(1-x²)],[sqrt(1-x²),-x]], W=i exp(-i pi Z/4)
    R exp(-i pi Z/4). Subtract pi/4 at both endpoints and pi/2 inside,
    returning global phase d*pi/2. This preserves the full 2x2 product,
    including controlled use. A constant sequence needs no conversion.
    """
    converted = np.asarray(phases, float).copy().reshape(-1)
    if not len(converted):
        raise ValueError("reflection conversion needs at least one phase")
    degree = len(converted) - 1
    if degree:
        converted[0] -= np.pi / 4
        converted[-1] -= np.pi / 4
        converted[1:-1] -= np.pi / 2
    return converted, degree * np.pi / 2


@dataclass(frozen=True, kw_only=True)
class PhaseSynthesisEvidence:
    """Numerical synthesis evidence of one fixed target and its FFT doubling.

    The attempts start at N0=next_power_of_two(2d+1) and double.
    ``residuals[i]`` is the coefficient one-norm of the polynomial rebuilt
    from the angles of attempt ``fft_lengths[i]``; a residual that stops
    decreasing is recorded here as a diagnostic, not used to stop.
    ``coefficient_residual`` is the last one, which accepted the result when
    it is at most the requested tolerance. Its exact-arithmetic
    uniform-error relation excludes the unknown arithmetic error of the FFT,
    transcendental and reconstruction steps.

    ``ideal_rule`` names the ideal-arithmetic rule (``"generic_sufficient"``,
    ``"real_wx_annulus"``, ``"full_ja_sufficient"`` or ``"constant"``) that
    calls the last attempted length sufficient for half the tolerance, and
    ``ideal_aliasing_bound`` is its conditional ideal bound there. Both are
    None when no rule holds at that length. The rules are evaluated only at
    attempted lengths, so this evidence does not depend on the memory limit.
    A failed attempt at a length where a rule holds ends the doubling, as do
    the explicit ``max_fft`` and an attempt whose arrays would exceed the
    memory limit. ``known_array_bytes`` estimates the dominant arrays of the
    longest attempt, excluding FFT library workspace and process memory.
    """
    norm_bound: float
    norm_bound_qualification: str
    fft_lengths: tuple[int, ...]
    residuals: tuple[float, ...]
    coefficient_residual: float | None
    ideal_rule: str | None
    ideal_aliasing_bound: float | None
    known_array_bytes: int
    qualification: str = "numerical_uniform_error_estimate"
    ideal_aliasing_qualification: str = "conditional_ideal_arithmetic"

    def to_dict(self):
        record = asdict(self)
        record["fft_lengths"] = list(self.fft_lengths)
        record["residuals"] = list(self.residuals)
        return record


@dataclass(frozen=True, kw_only=True)
class WxQSPPhases:
    """Real definite-parity Wx phases and their numerical residual evidence.

    ``phases`` realizes Re P(x) for the ascending Chebyshev coefficients.
    ``residual_sup_bound`` carries the reconstructed coefficient one-norm
    estimate, not a sampled maximum or a certified arithmetic bound.
    ``evidence`` reports the actual attempts.
    """
    phases: tuple[float, ...]
    degree: int
    parity: int
    target_chebyshev_coefficients: tuple[float, ...]
    residual_sup_bound: float
    evidence: PhaseSynthesisEvidence

    def to_dict(self):
        return {
            "phases": list(self.phases),
            "degree": self.degree,
            "parity": self.parity,
            "target_chebyshev_coefficients": list(self.target_chebyshev_coefficients),
            "residual_sup_bound": self.residual_sup_bound,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True, kw_only=True)
class LaurentQSPPhases:
    """General complex Laurent target synthesized as A0 D ... D Ad iX.

    ``rotation_angles`` gives matrix-order (RZ, RX, RZ) angles for each A.
    A circuit appends the matrices in reverse order, beginning with
    RX(``terminal_rx``), and uses the first ``negative_signals`` mathematical
    signals as diag(1,z^-1). ``target_coefficients`` is indexed from
    ``minimum_power``. The numerical residual is qualified by ``evidence``.
    """
    rotation_angles: tuple[tuple[float, float, float], ...]
    terminal_rx: float
    negative_signals: int
    target_coefficients: tuple[complex, ...]
    minimum_power: int
    degree: int
    evidence: PhaseSynthesisEvidence


def _wx_target(coefficients):
    # c_k/2 is inexact only when c_k is a subnormal with an odd significand.
    # It is then off by 2**-1075 and can round to zero. The degree and parity
    # come from the input, and the residual is measured against this target,
    # an estimate without an arithmetic envelope.
    degree = len(coefficients) - 1
    target = np.zeros(degree + 1, complex)
    for k in range(degree % 2, degree + 1, 2):
        if k == 0:
            target[degree // 2] += coefficients[0]
        else:
            target[(degree + k) // 2] += coefficients[k] / 2
            target[(degree - k) // 2] += coefficients[k] / 2
    return target


@dataclass(frozen=True)
class _WxProblem:
    coefficients: np.ndarray
    target: np.ndarray
    rho: float


def _prepare_wx_problem(values, max_bytes=None):
    coefficients = _real_coefficients(values, max_bytes=max_bytes)
    support = np.flatnonzero(coefficients)
    # Only exact zeros can change support. An underflowed annulus moment
    # must never be used as this constant predicate.
    degree = int(support[-1]) if support.size else 0
    if degree + 1 < len(coefficients):
        coefficients = coefficients[:degree + 1].copy()
    if np.any(coefficients[1 - degree % 2::2] != 0):
        raise ValueError("QSP targets require exact definite parity")
    rho = _chebyshev_norm_bound(coefficients, max_bytes)
    if not 0 <= rho < 1:
        raise ValueError("QSP requires a strict norm bound below one; normalize the target with a recorded scale")
    return _WxProblem(coefficients, _wx_target(coefficients), rho)


def _wx_reconstruct(phases):
    rotations = np.empty((len(phases), 2, 2), complex)
    rotations[:, 0, 0] = rotations[:, 1, 1] = np.cos(phases)
    rotations[:, 0, 1] = rotations[:, 1, 0] = 1j * np.sin(phases)
    matrix = _nlft.matrix_product(rotations)
    # H W(x) H=w^-1 D(w²). Thus U00=w^-d sum(M_ab(w²))/2.
    # Real part on the circle pairs coefficient j with conjugate(d-j).
    coefficients = np.sum(matrix, axis=(0, 1)) / 2
    return (coefficients + coefficients[::-1].conj()) / 2


def _angles_from_factors(factors):
    phi = np.arctan(abs(factors))
    psi = (np.angle(factors) - np.pi / 2) / 2
    psi[factors == 0] = 0
    return tuple((-2 * float(p), -2 * float(a), 2 * float(p)) for a, p in zip(phi, psi))


def _rotation_matrices(angles):
    rotations = np.empty((len(angles), 2, 2), complex)
    for k, (left, rx, right) in enumerate(angles):
        c, s = math.cos(rx / 2), -1j * math.sin(rx / 2)
        rz_l = np.exp(-0.5j * left * np.array([1, -1]))
        rz_r = np.exp(-0.5j * right * np.array([1, -1]))
        rotations[k] = rz_l[:, None] * np.array([[c, s], [s, c]]) * rz_r[None, :]
    return rotations


def _laurent_reconstruct(angles, terminal_rx):
    matrix = _nlft.matrix_product(_rotation_matrices(angles))
    # Rebuild the terminal RX too: sin/cos conversion is part of the result.
    return math.cos(terminal_rx / 2) * matrix[0, 0] - 1j * math.sin(terminal_rx / 2) * matrix[0, 1]


def _check_synthesis_start(degree: int, *, max_fft=None, max_bytes=None) -> int:
    """Check the first FFT attempt of a phase synthesis of ``degree`` and return its length N0.

    N0 is the least power of two above ``2*degree`` (0 for degree 0). It is
    refused above an explicit ``max_fft``, and its dominant arrays
    (``_nlft.known_array_bytes``) against ``max_bytes``, None being the
    physical memory. Both grow with the degree, so under the same limits a
    degree that passes admits the first attempt of every lower degree. The
    later doublings and the residual test are not checked here.
    """
    n = _nlft.power_two(2 * degree + 1) if degree else 0
    if max_fft is not None and n > max_fft:
        raise ResourceLimitError(f"phase synthesis of degree {degree} needs FFT length {n}, above max_fft={max_fft}")
    _check_bytes(_nlft.known_array_bytes(degree, n), max_bytes, f"QSP phase synthesis at FFT={n}")
    return n


def _solve(b, target, rho, tolerance, adapter, norm_qualification, *, max_fft, max_bytes,
           real_coefficients=None, near_flat=None):
    """Double the FFT length from N0 until the actual coefficient residual is at most ``tolerance``.

    Doubling stops at the first accepted attempt, after a failed attempt at
    a length that an ideal-arithmetic rule calls sufficient or at the
    explicit ``max_fft``, or before an attempt whose dominant arrays exceed
    the memory limit (`_check_synthesis_start` checks N0). A residual that
    stagnates is recorded but does not stop the doubling. The target never
    changes.

    The refusal states the observed residual at each attempted length together
    with the actual stopping cause, and the target and the controls. When the
    residual has stagnated it says so as an observation at those lengths. A
    finite sequence of similar measured residuals proves only the absence of
    improvement at those sizes. It cannot distinguish an arithmetic floor, an
    estimator limitation, conditioning or delayed convergence without more
    evidence, so it establishes nothing about further memory and does not
    authorize loosening the target.
    """
    degree = len(b) - 1
    memory = memory_limit(max_bytes)
    log_r = None if real_coefficients is None else _nlft.real_log_radius(real_coefficients, rho)
    n = _check_synthesis_start(degree, max_fft=max_fft, max_bytes=max_bytes)
    lengths, residuals = [], []
    rule = rule_bound = None

    def evidence():
        return PhaseSynthesisEvidence(
            norm_bound=rho, norm_bound_qualification=norm_qualification, fft_lengths=tuple(lengths),
            residuals=tuple(residuals), coefficient_residual=residuals[-1] if residuals else None,
            ideal_rule=rule, ideal_aliasing_bound=rule_bound,
            known_array_bytes=_nlft.known_array_bytes(degree, max(lengths, default=n)))

    while True:
        lengths.append(n)
        rule, rule_bound = _nlft.ideal_rule(degree, rho, tolerance, n, log_r=log_r, near_flat=near_flat)
        try:
            factors = _nlft.factors_from_target(b, n)
            if not np.all(np.isfinite(factors)):
                raise FloatingPointError("phase synthesis produced nonfinite factors")
            if adapter == "wx":
                output = np.arctan(factors.imag)
                output[0] -= np.pi / 4
                output[-1] -= np.pi / 4
                realized = _wx_reconstruct(output)
            else:
                output = _angles_from_factors(factors)
                realized = _laurent_reconstruct(output, -math.pi)
            residual = float(np.sum(abs(realized - target)))
            if not math.isfinite(residual):
                raise FloatingPointError("phase reconstruction is nonfinite")
        except FloatingPointError as error:
            raise _QSPPhaseConvergenceError(f"phase synthesis stopped at FFT={n}: {error}",
                                            evidence=evidence()) from error
        residuals.append(residual)
        if residual <= tolerance:
            return output, evidence()
        del factors, output, realized
        limits, failure = "bound the FFT length", _QSPPhaseLimitError
        if rule is not None:
            reason = (f"the ideal-arithmetic rule {rule} calls FFT={n} sufficient, bounding its aliasing "
                      f"contribution by {rule_bound:.3g}")
            limits, failure = "do not extend a stop by this rule", _QSPPhaseConvergenceError
        elif max_fft is not None and 2 * n > max_fft:
            reason = f"FFT={2 * n} would exceed max_fft={max_fft}"
        elif memory is not None and _nlft.known_array_bytes(degree, 2 * n) > memory:
            reason = (f"FFT={2 * n} needs about {_nlft.known_array_bytes(degree, 2 * n)} bytes, above the "
                      f"memory limit of {memory} bytes")
        else:
            n *= 2
            continue
        raise failure(
            f"numerical phase synthesis residual {residual:.6g} exceeds its target {tolerance:.6g}. "
            f"FFT lengths {tuple(lengths)} gave residuals {tuple(float(f'{r:.3g}') for r in residuals)}. "
            f"{_stagnation_note(lengths, residuals)}The doubling stopped because {reason}. Controls: residual_tolerance "
            "sets the target (LCHS sets it from approx_tol and QLS from phase_tol), and a "
            f"larger target accepts a larger synthesis error. The limits max_fft and max_bytes {limits}",
            evidence=evidence())


def _stagnation_note(lengths, residuals):
    """Describe the last attempts where no doubling reduced the residual by more than half, or return "" when the last one did.

    This wording is the only use of the factor of two, and the stopping rule
    never reads it. The factor covers a residual that drifts at its floor, such as
    1.41e-12 then 1.39e-12, which a strict comparison would call an improvement.
    """
    first = len(residuals) - 1
    while first and residuals[first] >= residuals[first - 1] / 2:
        first -= 1
    if first == len(residuals) - 1:
        return ""
    return (f"No doubling from FFT={lengths[first]} to FFT={lengths[-1]} reduced the residual by more than half, and these attempts do "
            "not show whether a longer FFT would reach the target. ")


def _controls(residual_tolerance, max_fft):
    tolerance = finite_real(residual_tolerance, "residual_tolerance")
    if tolerance <= 0:
        raise ValueError("residual_tolerance must be positive")
    return tolerance, None if max_fft is None else integer(max_fft, "max_fft", 1)


def solve_wx_qsp_phases(chebyshev_coefficients: Any, *, residual_tolerance=QSP_SOLVER_RESIDUAL_TOLERANCE,
                        max_fft=None, max_bytes=None) -> WxQSPPhases:
    """Synthesize real definite-parity Wx phases with a strict norm margin.

    ``chebyshev_coefficients`` is a finite ascending real vector. Wrong-parity
    coefficients must be exactly zero and its numerical norming bound must
    be strictly below one. Constants and zero targets are supported. The
    target, degree and normalization never change between attempts.

    ``residual_tolerance`` defaults to 1e-12 in the normalized target frame.
    The FFT length doubles from N0=next_power_of_two(2d+1) until the actual
    coefficient residual is at most that tolerance. Doubling stops after a
    failed attempt at a length that the real-Wx annulus rule or the generic
    analytic rule calls sufficient (PhaseSynthesisEvidence), at the explicit
    ``max_fft`` (default None, no cap), or before an attempt whose dominant
    arrays exceed ``max_bytes`` (default None, the physical memory).

    Returns Wx phases and numerical uniform-error evidence. A residual above
    the tolerance when doubling stops raises ValueError with every attempted
    length and residual, the last attempted lengths at which no doubling halved
    the residual, and the available controls. Those attempts do not show
    whether a longer FFT would reach the tolerance, and a larger
    ``residual_tolerance`` accepts a larger synthesis error. The coefficient
    one-norm relation is rigorous in exact arithmetic, but the numerical result
    has no complete floating-point error enclosure.
    """
    tolerance, max_fft = _controls(residual_tolerance, max_fft)
    return _solve_wx_problem(_prepare_wx_problem(chebyshev_coefficients, max_bytes), tolerance,
                             max_fft=max_fft, max_bytes=max_bytes)


def _solve_wx_problem(problem, tolerance, *, max_fft=None, max_bytes=None):
    phases, evidence = _solve(
        1j * problem.target, problem.target, problem.rho, tolerance, "wx", "numerical_chebyshev_norming",
        max_fft=max_fft, max_bytes=max_bytes, real_coefficients=problem.coefficients)
    degree = len(problem.coefficients) - 1
    return WxQSPPhases(
        phases=tuple(float(value) for value in phases), degree=degree, parity=degree % 2,
        target_chebyshev_coefficients=tuple(float(value) for value in problem.coefficients),
        residual_sup_bound=evidence.coefficient_residual, evidence=evidence)


def solve_laurent_qsp_phases(
    coefficients: Any, *, minimum_power=0, norm_bound: float,
    norm_bound_qualification="supplied_bound", residual_tolerance=QSP_SOLVER_RESIDUAL_TOLERANCE,
    max_fft=None, max_bytes=None, full_ja_scale: float | None = None, full_ja_error: float | None = None,
) -> LaurentQSPPhases:
    """Synthesize a general complex Laurent polynomial with a known strict cap.

    ``coefficients[j]`` multiplies z**(minimum_power+j), with integer
    ``minimum_power`` (default zero). ``norm_bound`` must satisfy
    sup_{|z|=1}|f(z)| <= norm_bound < 1. Its qualification is recorded verbatim
    by ``norm_bound_qualification``. Degree counts the interval containing
    both the supplied support and zero, including negative signal powers.

    The normalized residual tolerance defaults to 1e-12. The FFT doubling,
    ``max_fft`` and ``max_bytes`` act as in `solve_wx_qsp_phases`; the ideal
    rules are the generic analytic rule and, when supplied, the near-flat
    rule. The target never changes between attempts.

    ``full_ja_scale`` and ``full_ja_error`` may be supplied together only for
    a full complex approximation p to a unit-modulus target, f=p/scale,
    with uniform error <= full_ja_error, including coefficient uncertainty.
    They enable the separate near-flat ideal rule. They do not apply to
    individual cosine or sine targets. Failure raises ValueError.
    """
    tolerance, max_fft = _controls(residual_tolerance, max_fft)
    try:
        count = len(coefficients)
    except TypeError:
        raise ValueError("Laurent coefficients must be a nonempty numeric vector") from None
    if not count:
        raise ValueError("Laurent coefficients must be a nonempty numeric vector")
    if isinstance(minimum_power, (bool, np.bool_)):
        raise ValueError("minimum_power must be an integer")
    try:
        lower = operator.index(minimum_power)
    except TypeError:
        raise ValueError("minimum_power must be an integer") from None
    upper = lower + count - 1
    negative = max(0, -lower)
    degree = negative + max(0, upper)
    _check_bytes(48 * (degree + 1), max_bytes, "Laurent target conversion")
    source = np.asarray(coefficients)
    if source.ndim != 1 or source.dtype.kind not in "iufc":
        raise ValueError("Laurent coefficients must be a nonempty numeric vector")
    source = np.asarray(source, complex)
    if not np.all(np.isfinite(source)):
        raise ValueError("Laurent coefficients must be finite")
    rho = finite_real(norm_bound, "norm_bound")
    if not 0 <= rho < 1:
        raise ValueError("norm_bound must lie in [0,1)")
    target = np.zeros(degree + 1, complex)
    target[lower + negative:lower + negative + len(source)] = source
    near_flat = None
    if (full_ja_scale is None) != (full_ja_error is None):
        raise ValueError("full_ja_scale and full_ja_error must be supplied together")
    if full_ja_scale is not None:
        scale = finite_real(full_ja_scale, "full_ja_scale")
        error = finite_real(full_ja_error, "full_ja_error")
        if scale <= 1 or error < 0:
            raise ValueError("full-JA evidence requires scale > 1 and error >= 0")
        near_flat = (float(np.sum(abs(target))), scale, error)
    angles, evidence = _solve(-1j * target, target, rho, tolerance, "laurent", str(norm_bound_qualification),
                              max_fft=max_fft, max_bytes=max_bytes, near_flat=near_flat)
    return LaurentQSPPhases(rotation_angles=angles, terminal_rx=-math.pi,
                            negative_signals=negative, target_coefficients=tuple(source),
                            minimum_power=lower, degree=degree, evidence=evidence)


__all__ = [
    "QSP_SOLVER_RESIDUAL_TOLERANCE", "WxQSPPhases", "LaurentQSPPhases",
    "PhaseSynthesisEvidence", "chebyshev_grid", "chebyshev_norming_sup_bound",
    "chebyshev_polynomial_sup_bound", "evaluate_qsp_polynomial",
    "solve_wx_qsp_phases", "solve_laurent_qsp_phases", "wx_phases_to_reflection",
]
