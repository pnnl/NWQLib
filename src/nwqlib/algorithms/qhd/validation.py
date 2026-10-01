"""QHD symbolic admission, the binary64 range rule of planning and the exact dropped-angle ledger.

Concrete scalar admission belongs to potential.
"""

from fractions import Fraction
from math import isfinite
from sys import float_info
from typing import Any, Callable, Sequence
import sympy as sp

# Keep every computed nonzero rotation at every objective and time scale
# unless the caller explicitly selects angle pruning, whose removed angles are
# then recorded. Registered in docs/ENGINEERING_CONSTANTS.md. Revisit only
# with a declared physical error allocation.
DEFAULT_ROTATION_THRESHOLD = 0.0

# Every finite binary64 number is an integer multiple of 2**-1074, the
# smallest subnormal, so a sum of binary64 angles is an exact integer count of
# that unit. Exact, not tuned. Revisit only with another floating-point format.
_ANGLE_UNIT_BITS = 1074

# nu = 2**-1022 and Omega = (2 - 2**-52) 2**1023, the ends of the normal
# binary64 range (_normal_range). Exact format constants, not tuned.
_NORMAL_MIN, _NORMAL_MAX = float_info.min, float_info.max


def _normal_range(result, name, exact=None, *, zero=False):
    """Admit one operation used by a normal binary64 relative-error proof.

    The phase and angle ledgers of QHD planning
    (``method._compiled_phase_allowance``, ``circuit_errors``) bound each
    product with a relative rounding error, which an underflowed product
    does not have, so every product they charge must pass this rule.

    Read the stored operands as exact real numbers. A multiplication,
    division or power-of-two scaling must have exact result zero or magnitude
    between ``nu = 2**-1022`` and ``Omega = sys.float_info.max``, inclusive.
    Then round-to-nearest gives ``fl(z) = z (1 + delta)`` with
    ``|delta| <= u = 2**-53`` and no inexact underflow (Higham, *Accuracy and
    Stability of Numerical Algorithms*, 2nd ed., doi:10.1137/1.9780898718027,
    Theorem 2.2 and the standard model, Eq. (2.4)), and a power-of-two
    scaling is exact. A normal rounded result alone does not establish this.
    For example, ``fl(nu nextafter(1, 0)) = nu`` although the exact product is
    ``nu - 2**-1075``. A result strictly between nu and Omega in magnitude
    has its exact value in that interval, so only a result at either end
    consults ``exact``, the callable that returns the exact rational result.
    ``exact`` is None when the result is itself the stored quantity, such as
    a table value or a step weight, whose normality is the requirement.

    A zero result is legal only when its operands or the producer's
    mathematical structure establish zero (``zero``). A zero from nonzero
    operands is a rejection, even if a later operation would bring the exact
    value back into range, so each producer checks an operation before
    another operation, pruning or zero omission can hide its range loss.

    Finite additions and subtractions may produce exact subnormal scratch
    under gradual underflow, since every binary64 number is an integer
    multiple of ``2**-1074`` (Higham's underflow model, Eq. (2.8), has no
    absolute term for them). A scratch value used as a model coefficient or
    gate parameter must pass this check at that boundary. Error bounds use
    outward arithmetic and may be subnormal. They are not governed by this
    rule. The augmented-Lagrangian layer applies the same rule to its exact
    multiplier conversion (``constrained._original_multiplier``).

    Returns:
        ``result``, unchanged.

    Raises:
        ValueError: The named quantity or operation violates this rule.
    """
    if _underflows(result, name, exact, zero=zero):
        raise ValueError(f"{name} is {result!r}, below the normal binary64 range [2**-1022, {_NORMAL_MAX!r}] "
                         "that the error bounds of QHD planning assume")
    return result


def _underflows(result, name, exact=None, *, zero=False):
    """Return whether one operation fails the rule of ``_normal_range`` at the lower end of the range.

    A lower-range failure is a nonzero exact result below ``2**-1022`` in
    magnitude, a zero from nonzero operands included, the one failure that
    the omission policy of planning turns into an omitted contribution with
    a charge (``potential.PotentialCompiler.select_occurrences``,
    ``binary.PhaseTable.synthesize``, ``initial_state.chain_selection``). A
    nonfinite result or an exact result above the largest binary64 number
    still raises ValueError here, since no omission can charge it.
    """
    if not isfinite(result):
        raise ValueError(f"{name} is {result!r}, not a finite binary64 number")
    magnitude = abs(result)
    if magnitude == 0.0:
        return not zero
    if magnitude == _NORMAL_MAX and exact is not None and abs(exact()) > _NORMAL_MAX:
        raise ValueError(f"{name} is {result!r} from an exact value above the largest binary64 number")
    return magnitude < _NORMAL_MIN or (magnitude == _NORMAL_MIN and exact is not None
                                       and abs(exact()) < _NORMAL_MIN)


def _normal_product(x, y, name):
    """Return ``fl(x y)`` after ``_normal_range`` admits it, with an exact zero only for a zero factor."""
    return _normal_range(x * y, name, lambda: Fraction(x) * Fraction(y), zero=x == 0 or y == 0)


def _normal_quotient(x, y, name):
    """Return ``fl(x/y)`` for a finite nonzero y after ``_normal_range`` admits it, zero only for x = 0."""
    if y == 0 or not isfinite(y):
        raise ValueError(f"{name} divides by {y!r}, not a finite nonzero binary64 number")
    return _normal_range(x / y, name, lambda: Fraction(x) / Fraction(y), zero=x == 0)


def _normal_products(factor, values, name):
    """Return the elementwise ``fl(factor v)`` of an array after ``_normal_range`` admits every entry.

    An entry is an exact zero only where its value or the factor is zero
    (``_admitted_products`` classifies the entries).
    """
    import numpy as np

    products, omitted = _admitted_products(factor, values, name)
    if omitted.any():
        value = float(np.asarray(values, dtype=float)[omitted][0])
        # A lower-range failure, which the rule refuses here.
        _normal_range(factor * value, name, lambda: Fraction(factor) * Fraction(value))
    return products


def _admitted_products(factor, values, name):
    """Return ``(products, omitted)``: the elementwise ``fl(factor v)`` with the lower-range failures set to zero.

    ``omitted`` marks the entries whose exact product is nonzero and below
    ``2**-1022`` (``_underflows``), where the product is replaced by exact
    zero. The array pass checks finiteness and the open normal range, and
    only the entries at either end of the range need the exact product. A
    nonfinite product or an exact product above the largest binary64 number
    raises ValueError.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    products = factor * values
    free = (values != 0) & (factor != 0)
    finite = np.isfinite(products)
    if not finite.all():
        _normal_range(float(products[~finite][0]), name)
    magnitudes = np.abs(products)
    omitted = free & (magnitudes < _NORMAL_MIN)
    for index in np.flatnonzero(free & ((magnitudes == _NORMAL_MIN) | (magnitudes == _NORMAL_MAX))).tolist():
        value = float(values[index])
        omitted[index] = _underflows(float(products[index]), name, lambda: Fraction(factor) * Fraction(value))
    return np.where(omitted, 0.0, products), omitted


def angle_units(value):
    """Return ``|value|`` as an exact integer count of the unit ``2**-1074``.

    ``float.as_integer_ratio`` gives ``|value| = p/q`` with q a power of two
    no larger than ``2**1074``, so ``|value| = p (2**1074/q) 2**-1074``
    exactly. Summing these integers adds dropped angles with no rounding, so
    ``records.QHDReconstruction.pruning_error_bound`` is formed from the
    exact sum and rounded upward once, never below the exact bound and zero
    exactly when nothing was removed.
    """
    numerator, denominator = abs(float(value)).as_integer_ratio()
    return numerator * ((1 << _ANGLE_UNIT_BITS) // denominator)


def _validate_qhd_problem_fields(problem):
    """Reject objectives and variables that cannot define a real diagonal potential.

    ``Optimization`` already admits the ordered SymPy symbols, unique names,
    declared free symbols and one finite ``lower < upper`` interval per
    variable. QHD additionally needs a live commutative SymPy expression and
    commutative variables not declared non-real, because each grid value
    becomes a real diagonal entry of the Hamiltonian.
    """
    if not isinstance(problem.objective, sp.Expr):
        raise TypeError("objective must be a live SymPy expression")
    if problem.objective.is_commutative is not True:
        raise ValueError("optimization objective must be a commutative scalar expression")
    if any(v.is_commutative is not True or v.is_real is False for v in problem.variables):
        raise ValueError("optimization variables must admit commutative real scalar values")


def _kinetic_range_error(error):
    """Return the refusal of a kinetic coefficient, energy or angle outside the admitted range, with its remedy.

    The lower-range omission policy covers contributions formed from
    objective data and the initial state, and the phase products of binary
    kinetic blocks (``binary.PhaseTable.synthesize``). Kinetic coefficients,
    energies and one-hot kinetic terms are set by the grid spacing h and the
    kinetic schedule weight a alone, and fall outside the normal binary64
    range only for extreme boxes, such as a spacing near 1e153, so planning
    refuses them rather than approximating them.
    """
    return ValueError(f"{error}. The grid spacing h and the kinetic schedule weight a put this kinetic term, of "
                      "scale a/h**2, outside the range that planning admits. Choose a box and grid whose spacing "
                      "keeps it in range, or another total_time or schedule")


def _lambdify_objective(variables: Sequence[sp.Symbol], objective: sp.Expr) -> Callable[..., Any]:
    """Return a NumPy function of ``variables`` in order, or raise ValueError when SymPy cannot.

    SymPy's NumPy printer raises KeyError for a constant it cannot print, such as ``zoo``.
    """
    try:
        return sp.lambdify(variables, objective, modules=["numpy"])
    except (SyntaxError, TypeError, ValueError, NotImplementedError, KeyError) as exc:
        raise ValueError(f"objective cannot be lambdified: {exc}") from exc
