"""Finite integer inversion of a supplied scalar error bound, without planning."""

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Literal

from nwqlib.core.records import Float64, Rational

from ._work import DEFAULT_MAX_INTEGER_BITS, ExactArithmetic
from .error_model import FramedFact, predicate_frame, scalar


@dataclass(frozen=True)
class ScalarBoundResult:
    """What [`resolve_scalar_bound`][nwqlib.evidence.scalar_bound.resolve_scalar_bound] returns: a sufficient integer for the supplied error model, with its conditions and assumptions.

    The answer is `integer`. `covered_bound` is a rational upper bound on
    the modeled error, not its exact value and not a total-accuracy
    certificate. The fields below are read-only.

    Attributes:
        integer: The smallest sufficient positive integer n in the supplied
            domain, the checked `fixed` value, or `None` when none is
            established.
        status: Outcome of the bound and its conditions, not of an execution:
            `"resolved"`, `"conditional"` (it rests on assumptions or
            unspecified conditions), `"unresolved"` (a quantity or condition is
            unknown), `"inapplicable"` (a condition is false or does not
            apply) or `"infeasible"` (no permitted n meets the tolerance).
        reason: Explanation of the choice, or why no sufficient choice was
            established.
        covered_bound: Rational upper bound on `e(n)` in the supplied error
            metric, or `None` without a choice.
        conditions: The supplied conditions the bound depends on.
        assumptions: Every stated assumption, including unverified ones.
        declared_work: The supplied linear work model at the chosen n, or
            `None` without a model or a choice.
    """

    integer: int | None
    status: str
    reason: str
    covered_bound: Fraction | None
    conditions: tuple
    assumptions: tuple[str, ...]
    declared_work: Fraction | None = None

    def to_dict(self):
        """Return the fields as a JSON-ready dictionary, with rationals as numerator and denominator."""
        def ratio(value):
            return None if value is None else dict(numerator=value.numerator, denominator=value.denominator)
        return dict(integer=self.integer, status=self.status, reason=self.reason,
                    covered_bound=ratio(self.covered_bound), declared_work=ratio(self.declared_work),
                    conditions=[item.model_dump(mode="json") if isinstance(item, FramedFact) else item
                                for item in self.conditions], assumptions=list(self.assumptions))


def resolve_scalar_bound(
    coefficient: int | float | Fraction | Rational | Float64 | FramedFact | None,
    tolerance: int | float | Fraction | Rational | Float64 | FramedFact | None,
    *,
    maximum: int,
    minimum: int = 1,
    fixed_error: int | float | Fraction | Rational | Float64 | FramedFact | None = 0,
    decay: Literal["inverse", "inverse_sqrt"] = "inverse",
    fixed: int | None = None,
    conditions: Iterable[FramedFact | bool | None] = (),
    assumptions: Iterable[str] = (),
    work_per_unit: int | float | Fraction | Rational | Float64 | None = None,
    max_integer_bits: int = DEFAULT_MAX_INTEGER_BITS,
) -> ScalarBoundResult:
    """Find the smallest integer n with `e_0 + c/n <= epsilon` (or `c/sqrt(n)`) in a finite range.

    The supplied model is `e(n) = e_0 + c/n` or `e(n) = e_0 + c/sqrt(n)`.
    For `epsilon > e_0` and `c > 0`, `e(n) <= epsilon` holds exactly when
    `n >= c/(epsilon - e_0)` for `c/n` and when `n >= (c/(epsilon - e_0))**2`
    for `c/sqrt(n)`, so the smallest permitted integer is an exact rational
    ceiling, with no search over n. For example, with `e_0 = 1/8`,
    `epsilon = 3/8` and `c = 1`, the answer is 4 for `c/n` and 16 for
    `c/sqrt(n)`.

    All error quantities must share an absolute metric, unit and scope.
    `FramedFact` inputs are checked for that, and plain numbers assume it.
    Unknown quantities do not become zero. A false condition prevents a
    choice, and an unspecified condition makes the result conditional. The
    supplied bounds
    need not cover every error of a method, so the result is not a
    total-accuracy claim. It also sets no Method parameter. Use the answer
    in a new Method configuration yourself when the model describes that
    parameter.

    Args:
        coefficient (number | FramedFact): c, nonnegative.
        tolerance (number | FramedFact): epsilon, the allowed error,
            nonnegative.
        maximum (int): Largest permitted n, positive.
        minimum (int): Smallest permitted n, positive. Default 1.
        fixed_error (number | FramedFact): e_0, the part of the error that
            does not depend on n, nonnegative. Default 0.
        decay (str): `"inverse"` for `c/n`, the default, or `"inverse_sqrt"`
            for `c/sqrt(n)`.
        fixed (int | None): An n to check instead of choosing the smallest
            one.
        conditions (Iterable): Conditions of the model, as framed boolean
            predicates, booleans, or `None` for an unspecified condition.
        assumptions (Iterable[str]): Further assumptions stated as text.
        work_per_unit (number | None): Declared work per unit of n,
            multiplied by the chosen n.
        max_integer_bits (int): Bit limit of the exact arithmetic. Default
            4096.

    Returns:
        bound (ScalarBoundResult): The chosen n, its status, a rational upper
            bound on `e(n)` and the conditions and assumptions it depends on.

    Raises:
        ValueError: If `minimum`, `maximum` or `fixed` is not a positive
            integer in order, `decay` is another value, a quantity is
            negative, the framed quantities disagree in frame or parameter
            point, or a condition is not a boolean predicate.
    """
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    for name, value in (("minimum", minimum), ("maximum", maximum), ("fixed", fixed)):
        if name == "fixed" and value is None:
            continue
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        arithmetic.fraction(value)
    if minimum > maximum or fixed is not None and not minimum <= fixed <= maximum:
        raise ValueError("finite integer domain and fixed value must agree")
    if decay not in {"inverse", "inverse_sqrt"}:
        raise ValueError("decay must be inverse or inverse_sqrt")
    conditions, assumptions = tuple(conditions), tuple(assumptions)
    if any(not isinstance(item, str) or not item.strip() for item in assumptions):
        raise ValueError("assumptions must be nonempty strings")

    frame = None
    restrictions = {}
    extra = []
    values = []
    for name, value in (("coefficient", coefficient), ("tolerance", tolerance), ("fixed_error", fixed_error)):
        if isinstance(value, FramedFact):
            if frame is not None and not frame.compatible(value.frame):
                raise ValueError("bound quantities must share metric, unit, scope and conditioning")
            frame = value.frame
            extra.extend(value.fact.assumptions)
            for binding in value.bindings:
                prior = restrictions.setdefault(binding.parameter, binding.value)
                if arithmetic.fraction(prior) != arithmetic.fraction(binding.value):
                    raise ValueError("bound quantities have conflicting parameter restrictions")
                extra.append(f"supplied bound requires {binding.parameter}={binding.value}")
            number = scalar(value, arithmetic)
        else:
            number = None if value is None else arithmetic.fraction(value)
        if number is not None and number < 0:
            raise ValueError(f"{name} must be nonnegative")
        values.append(number)
    work = None if work_per_unit is None else arithmetic.fraction(work_per_unit)
    if work is not None and work < 0:
        raise ValueError("work_per_unit must be nonnegative")
    conditional = bool(assumptions) or bool(extra)
    assumptions = tuple(dict.fromkeys((*assumptions, *extra,
        "Supplied bounds share an absolute error metric, unit and scope; they need not cover every method error.")))
    blocked = None
    for condition in conditions:
        if isinstance(condition, FramedFact):
            if not predicate_frame(condition.frame).compatible(condition.frame):
                raise ValueError("condition must use a dimensionless predicate frame")
            if frame is not None and not predicate_frame(frame).compatible(condition.frame):
                raise ValueError("condition must have the bound's predicate frame")
            if condition.fact.availability != "concrete":
                if blocked is None or blocked[0] != "inapplicable":
                    blocked = ("inapplicable" if condition.fact.availability == "not_applicable" else "unresolved",
                               condition.fact.reason or "premise unavailable")
                continue
            value = condition.fact.value
            if type(value) is not bool:
                raise ValueError("a bound premise must be a boolean predicate")
            # Received assertions and bindings remain visible; no receipt type is
            # promoted to proof or matched to a made-up Plan context here.
            conditional = True
        elif condition is None or type(condition) is bool:
            value = condition
            conditional |= condition is None
        else:
            raise ValueError("conditions must be framed predicates, booleans or None")
        if value is False:
            blocked = ("inapplicable", "a supplied premise is false")

    def result(n, status, reason, bound=None):
        declared = None if n is None or work is None else arithmetic.multiply(work, Fraction(n))
        return ScalarBoundResult(n, status, reason, bound, conditions, assumptions, declared)

    if blocked is not None:
        return result(None, *blocked)
    if any(value is None for value in values):
        return result(None, "unresolved", "a supplied coefficient, tolerance or fixed contribution is unavailable")
    c, epsilon, fixed_part = values
    allowance = arithmetic.subtract(epsilon, fixed_part)
    if allowance < 0 or allowance == 0 and c > 0:
        return result(None, "infeasible", "fixed error exhausts the available tolerance")
    if c == 0:
        required = minimum
    else:
        # From c/n <= a obtain n >= c/a; from c/sqrt(n) <= a obtain
        # n >= (c/a)^2 for c,a>0. Exact rational ceiling has no search grid.
        ratio = arithmetic.divide(c, allowance)
        threshold = ratio if decay == "inverse" else arithmetic.multiply(ratio, ratio)
        quotient, remainder = divmod(threshold.numerator, threshold.denominator)
        required = max(minimum, quotient + bool(remainder))
        arithmetic.fraction(required)
    n = required if fixed is None else fixed
    if n < required or n > maximum:
        return result(None, "infeasible", "no permitted integer meets the supplied bound")
    if decay == "inverse" or c == 0:
        bound = arithmetic.add(fixed_part, arithmetic.divide(c, Fraction(n)))
    else:
        # floor(sqrt(n)) <= sqrt(n), so c/floor(sqrt(n)) bounds the
        # irrational term above. The exact inversion also proves <= epsilon;
        # the minimum of these two rational upper bounds remains conservative.
        coarse = arithmetic.add(fixed_part, arithmetic.divide(c, Fraction(isqrt(n))))
        bound = coarse if arithmetic.le(coarse, epsilon) else epsilon
    return result(n, "conditional" if conditional else "resolved",
                  "sufficient for the supplied scalar model only; no total-accuracy claim", bound)


__all__ = ["ScalarBoundResult", "resolve_scalar_bound"]
