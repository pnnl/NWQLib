"""Finite modified-Bessel filters for exp[-beta(1-x)] on [-1, 1].

The Chebyshev expansion is the modified Jacobi-Anger identity of Martyn
et al., arXiv:2110.11327v6, Appendix B Eq. (61), with x replaced by -x.
Its nonnegative coefficients also define the one-sided analytic Q(z).
For the integer-valued distribution exp(-beta) I_|k|(beta), the moment
generating function is exp[beta(cosh(u)-1)]. Applying the two Chernoff
inequalities at r=m+1 and optimizing at beta*sinh(u)=r gives
tail <= min(1, 2 exp[-A]), A=r*asinh(r/beta)-(hypot(beta,r)-beta).
The stable forms below implement this finite bound without a Bessel suffix.

The target uses the supplied numerical beta, not a physical alpha*time
product. A caller must distinguish structural zero from a positive physical
product rounded to zero, and separately propagate parameter error. For a
Hermitian contraction X that discrepancy costs at most 2*abs(beta*-beta).
Coefficients, tail evaluation and reconstructed phases have separate
arithmetic qualifications. No complete floating-point enclosure is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy
from scipy.special import ive

from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from .phases import (
    LaurentQSPPhases,
    _QSPPhaseConvergenceError,
    solve_laurent_qsp_phases,
)


@dataclass(frozen=True, kw_only=True)
class ImaginaryTimeDegreeSelection:
    """Scalar selection for a finite modified-Bessel truncation.

    ``degree`` is a numerically selected sufficient degree, not a certified
    minimum. A positive-beta ``tail_bound=None`` means its positive bound
    underflowed, while ``tail_log_bound`` keeps the logarithm. Both are
    conditional ideal bounds with unenclosed libm evaluation. The
    predecessor log belongs to degree-1, or is None at degree zero.
    ``ratio_underflow_evaluations`` counts tail evaluations in which an
    underflowed ratio beta/r was replaced by zero, which only enlarges the
    bound.
    """

    beta: float
    epsilon_tail: float
    degree: int
    tail_bound: float | None
    tail_log_bound: float | None
    predecessor_tail_log_bound: float | None
    ratio_underflow_evaluations: int
    qualification: str


@dataclass(frozen=True, kw_only=True)
class ImaginaryTimeExpansion:
    """One finite expansion, its normalized analytic Q and its phase allowance.

    ``coefficients[k]`` multiplies T_k(x) before normalization, while
    ``target_coefficients[k]`` multiplies z**k after division by ``scale``.
    Their nominal degree remains ``selection.degree``, including numerical
    endpoint zeros. ``actual_support`` gives the first/last nonzero Q powers,
    or None for an all-zero numerical Q. It does not change the signal count.
    ``coefficient_error_bound`` concerns sum|scale*Q_k-a_k^ideal|, after
    division, so it includes normalization error relative to the ideal
    coefficients. It is None (unknown) for every positive beta, small or
    large. ``coefficient_method`` names the numerical source, SciPy's
    scaled modified Bessel function ``ive`` and its version, which supplies
    no error enclosure. Finite returned values therefore do not imply
    binary64-level relative accuracy. ``stored_norm_bound`` encloses
    only the exact sum of the numerical Q coefficients under the stated
    elementary rounding model. It does not enclose their ideal error.
    ``max_fft`` and ``max_bytes`` are the controls of the phase solve.
    """

    selection: ImaginaryTimeDegreeSelection
    coefficients: tuple[float, ...]
    target_coefficients: tuple[complex, ...]
    actual_support: tuple[int, int] | None
    coefficient_method: str
    coefficient_error_bound: float | None
    arithmetic_error_bound: float | None
    scale: float
    margin: float
    stored_norm_bound: float
    norm_bound_qualification: str
    epsilon_phase: float
    max_fft: int | None
    max_bytes: int | None


@dataclass(frozen=True, kw_only=True)
class PreparedImaginaryTimeFilter:
    """The selected filter and its canonical Laurent phase solution.

    ``solution=None`` denotes only the exact beta-zero identity. Otherwise
    its phase residual is in the normalized Q frame and is multiplied by
    ``expansion.scale`` once when composing filter error. The tail and
    coefficient error already use the unnormalized filter frame. Arithmetic
    enclosure remains separate in the expansion and phase evidence.
    """

    expansion: ImaginaryTimeExpansion
    solution: LaurentQSPPhases | None


def _tail_log(beta, degree):
    """Stable Chernoff exponent, with the conservative ratio-zero limit.

    Rationalizing sqrt(beta²+r²)-beta yields r*x/(hypot(1,x)+1)
    for x=r/beta<=1. For r>beta, writing y=beta/r avoids the reciprocal
    overflow. Both y-dependent terms increase with y, so a y that
    underflows to zero lowers A and raises the ideal tail bound, and the
    second value reports it. Exact integer r<=2**53 keeps the order
    conversion exact.
    """
    r = degree + 1
    underflow = False
    if r <= beta:
        x = r / beta
        exponent = r * (math.asinh(x) - x / (math.hypot(1.0, x) + 1.0))
    else:
        y = beta / r
        underflow = y == 0
        h = math.hypot(1.0, y)
        exponent = r * (math.log(r) - math.log(beta) + math.log1p(h) - 1.0 / (h + y))
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError("imaginary-time Chernoff exponent is outside its numerical range")
    return min(0.0, math.log(2.0) - exponent), underflow


def select_imaginary_time_degree(beta, epsilon_tail) -> ImaginaryTimeDegreeSelection:
    """Select a finite sufficient degree by scalar bracketing and bisection.

    In exact arithmetic the Chernoff exponent
    A(r)=r*asinh(r/beta)-(hypot(beta,r)-beta), r=degree+1, increases with r
    (A'(r)=asinh(r/beta)>0), and a degree passes when 2 exp(-A) <=
    epsilon_tail. Since asinh(x) >= log(2x) and hypot(beta,r)-beta <= r,
    A(r) >= r(log(2r/beta)-1), so A(r) >= r once r >= e²beta/2, and every
    r >= max(e²beta/2, log(2/epsilon_tail)) passes. The search starts its
    bracket there and doubles only if the binary64 evaluation disagrees.

    Args:
        beta (float): Finite nonnegative dimensionless numerical target parameter.
            Zero must be structural, never a positive physical product lost
            to underflow. Physical product error belongs to the caller.
        epsilon_tail (float): Finite nonnegative absolute allowance for the ideal
            truncated filter on [-1,1]. Positive beta requires a positive
            allowance. An allowance of at least one admits degree zero.

    Returns:
        selection (ImaginaryTimeDegreeSelection): Passing degree, adjacent
            failing predecessor and conditional tail.

    Raises:
        ValueError: For an invalid domain.
    """
    beta = finite_real(beta, "beta")
    epsilon_tail = finite_real(epsilon_tail, "epsilon_tail")
    if beta < 0 or epsilon_tail < 0:
        raise ValueError("beta and epsilon_tail must be nonnegative")
    if beta > 0 and epsilon_tail == 0:
        raise ValueError("positive beta has a positive omitted Bessel tail at every finite degree")
    underflows = 0

    def visit(degree):
        nonlocal underflows
        value, underflow = _tail_log(beta, degree)
        underflows += int(underflow)
        return value

    predecessor = None
    if beta == 0:
        degree, tail, log_tail = 0, 0.0, None
    else:
        log_allowance = math.log(epsilon_tail)
        degree = 0
        log_tail = visit(0)
        if log_tail > log_allowance:
            lower, lower_log = 0, log_tail
            upper = max(1, math.ceil(max(math.e**2 * beta / 2, math.log(2) - math.log(epsilon_tail))) - 1)
            upper_log = visit(upper)
            while upper_log > log_allowance:
                lower, lower_log = upper, upper_log
                upper *= 2
                upper_log = visit(upper)
            while upper - lower > 1:
                middle = (lower + upper) // 2
                middle_log = visit(middle)
                if middle_log <= log_allowance:
                    upper, upper_log = middle, middle_log
                else:
                    lower, lower_log = middle, middle_log
            degree, log_tail, predecessor = upper, upper_log, lower_log
        tail = math.exp(log_tail)
        if tail == 0:
            tail = None
    return ImaginaryTimeDegreeSelection(
        beta=beta, epsilon_tail=epsilon_tail, degree=degree, tail_bound=tail, tail_log_bound=log_tail,
        predecessor_tail_log_bound=predecessor, ratio_underflow_evaluations=underflows,
        qualification="exact_identity" if beta == 0 else "numerical_conditional_ideal_bound",
    )


def _upward_sum(values):
    """Enclose the exact sum of a nonempty float64 array of nonnegative values.

    With round-to-nearest and gradual underflow, math.fsum returns the exact
    sum rounded to nearest, so one nextafter step up bounds it. Python
    documents that on some builds whose C library adds in extended precision
    the result can be off by one unit in the last place, so two steps up
    bound the sum on every build. A sum of at most one nonzero value is
    exact and is returned unchanged. This does not enclose the special
    function that produced the values. ``values.min() >= 0`` also refuses
    NaN, and an infinite value gives an infinite sum, which is refused.
    """
    if not values.min() >= 0:
        raise ValueError("imaginary-time coefficients must be finite and nonnegative")
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ValueError("imaginary-time nonnegative coefficient sum overflows") from error
    if not math.isfinite(total):
        raise ValueError("imaginary-time coefficients must be finite and nonnegative")
    if np.count_nonzero(values) > 1:
        total = math.nextafter(math.nextafter(total, math.inf), math.inf)
        if not math.isfinite(total):
            raise ValueError("imaginary-time nonnegative coefficient sum overflows")
    return total


def imaginary_time_expansion(
    selection: ImaginaryTimeDegreeSelection, *, epsilon_phase,
    margin=1 / 16, max_fft=None, max_bytes=None,
) -> ImaginaryTimeExpansion:
    """Acquire the selected coefficients once and fix their phase allowance.

    Args:
        selection: Scalar result from select_imaginary_time_degree.
        epsilon_phase (float): Required positive finite allowance for the phase
            residual of the normalized Q, chosen from the caller's own error
            allocation and separate from the selection's tail allowance. It
            bounds e, the one-norm coefficient residual of the Q realized by
            the returned angles. In exact arithmetic that residual changes
            the unnormalized filter on [-1, 1] by at most ``scale``*e, so a
            filter allowance B needs epsilon_phase <= B/``scale``. A further
            factor of the caller's composition, such as a vector norm,
            multiplies this term as well. ``scale`` is known only from the
            returned expansion, so check the allocation there, before
            prepare_imaginary_time_filter, rather than repeating this call.
            The allowance is a numerical target: synthesis may still fail,
            and success does not certify the floating-point synthesis error.
        margin (float): Selected norm margin, one of 1/16 (default), 1e-2 or 1e-3.
            It never changes following a synthesis failure.
        max_fft (int | None): Optional positive FFT-length cap of the phase
            solve, default None.
        max_bytes (int | None): Default None, the physical memory. Memory
            limit of the coefficient arrays and of each phase attempt.

    Returns:
        expansion (ImaginaryTimeExpansion): Immutable coefficient tuples,
            scale, actual numerical circle cap and phase controls.

    Raises:
        ValueError: Invalid arguments; coefficient arrays above the memory
            limit; no usable scaled-Bessel coefficients from the installed
            SciPy for this beta and order range, which a finite beta does
            not guarantee; or a stored normalized cap not below one. An
            error raised under a SciPy special-function error policy that
            the caller set, for example with scipy.special.errstate,
            propagates unchanged.
    """
    epsilon_phase = finite_real(epsilon_phase, "epsilon_phase")
    margin = finite_real(margin, "margin")
    if epsilon_phase <= 0:
        raise ValueError("epsilon_phase must be positive")
    if margin not in (1 / 16, 1e-2, 1e-3):
        raise ValueError("margin must be selected from 1/16, 1e-2 or 1e-3")
    max_fft = None if max_fft is None else integer(max_fft, "max_fft", 1)
    controls = dict(epsilon_phase=epsilon_phase, max_fft=max_fft, max_bytes=max_bytes)
    if selection.beta == 0:
        return ImaginaryTimeExpansion(
            selection=selection, coefficients=(1.0,), target_coefficients=(1 + 0j,),
            actual_support=(0, 0), coefficient_method="exact_identity",
            coefficient_error_bound=0.0, arithmetic_error_bound=0.0,
            scale=1.0, margin=margin, stored_norm_bound=1.0,
            norm_bound_qualification="exact_identity", **controls)
    h = selection.degree + 1
    # The int64 orders, the float coefficients and the complex Q.
    _check_bytes(32 * h, max_bytes, "imaginary-time coefficient preparation")
    orders = np.arange(h, dtype=np.int64)
    coefficients = ive(orders, selection.beta)
    del orders
    coefficients[1:] *= 2
    try:
        stored_sum = _upward_sum(coefficients)
    except ValueError as error:
        # The beta domain is valid; this numerical method has no usable
        # result here. The finite range is version- and order-dependent.
        raise ValueError(
            f"scipy.special.ive {scipy.__version__} returned no usable scaled-Bessel coefficients "
            f"for beta={selection.beta!r} and orders 0..{selection.degree}; no phase synthesis "
            "started. A finite beta does not guarantee a usable evaluation") from error
    rho = 1.0 - margin
    scale = math.nextafter(max(1.0, stored_sum) / rho, math.inf)
    if not math.isfinite(scale):
        raise ValueError("imaginary-time normalization scale is not finite")
    target = coefficients.astype(complex)
    target.real /= scale
    cap = _upward_sum(target.real)
    if not cap < 1:
        raise ValueError("imaginary-time normalized coefficient cap is not strictly below one")
    support = np.flatnonzero(target.real)
    return ImaginaryTimeExpansion(
        selection=selection, coefficients=tuple(float(value) for value in coefficients),
        target_coefficients=tuple(complex(value) for value in target),
        actual_support=None if not support.size else (int(support[0]), int(support[-1])),
        coefficient_method=f"scipy.special.ive:{scipy.__version__}",
        coefficient_error_bound=None, arithmetic_error_bound=None, scale=scale, margin=margin,
        stored_norm_bound=cap,
        norm_bound_qualification="stored_nonnegative_sum_fsum_two_steps_up", **controls)


def prepare_imaginary_time_filter(expansion: ImaginaryTimeExpansion) -> PreparedImaginaryTimeFilter:
    """Prepare canonical phases for the fixed expansion.

    Args:
        expansion: The selected imaginary_time_expansion result, including
            its immutable Q coefficients, phase allowance and solve controls.

    Returns:
        prepared (PreparedImaginaryTimeFilter): The same expansion and its
            canonical Laurent solution. Exact beta zero returns the identity
            without a numerical fit. Only a degree-zero selection uses no
            FFT. A positive selected degree is fitted at that degree even
            when its numerical Q is constant.

    Raises:
        ValueError: The unchanged target fails its residual allowance, its
            memory limit or a numerical completion premise. Convergence
            errors include phase evidence.
    """
    if expansion.selection.beta == 0:
        return PreparedImaginaryTimeFilter(expansion=expansion, solution=None)
    try:
        # Q is the physical analytic target. The public adapter inserts -i.
        solution = solve_laurent_qsp_phases(
            expansion.target_coefficients, minimum_power=0,
            norm_bound=expansion.stored_norm_bound,
            norm_bound_qualification=expansion.norm_bound_qualification,
            residual_tolerance=expansion.epsilon_phase, max_fft=expansion.max_fft, max_bytes=expansion.max_bytes)
    except _QSPPhaseConvergenceError as error:
        raise type(error)(
            f"imaginary-time selected filter failed: {error}. imaginary_time_expansion(epsilon_phase=...) chooses "
            "another target, and its documentation gives the effect on the error",
            evidence=error.evidence) from error
    return PreparedImaginaryTimeFilter(expansion=expansion, solution=solution)


__all__ = [
    "ImaginaryTimeDegreeSelection",
    "ImaginaryTimeExpansion",
    "PreparedImaginaryTimeFilter",
    "select_imaginary_time_degree",
    "imaginary_time_expansion",
    "prepare_imaginary_time_filter",
]
