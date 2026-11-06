"""Checks of the QHD symbolic input, the binary64 range rule of planning and the exact sum of dropped angles.

The checks of concrete scalar values are in ``potential``.
"""

from fractions import Fraction
from math import isfinite
from sys import float_info
from typing import Any, Callable, Sequence
import mpmath
import sympy as sp
from sympy.core.evalf import PrecisionExhausted

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

# The exponential of a combined exponent (_combined_product), printed as NumPy's exp. An undefined function, because
# SymPy pulls a number out of the exponent of its own exp, also of an unevaluated one when a product holding it is
# rebuilt, and that number is the constant the rewrite exists to remove.
_HELD_EXP = sp.Function("_held_exp")

# Floor of the decimal digits of the scalar logarithms of a combined exponential (_working_digits). It is well above the
# 17 digits of the binary64 literal formed from them. Revisit only if a logarithm enters other than through that literal.
_LOG_DIGITS = 40


def _normal_range(result, name, exact=None, *, zero=False):
    """Check one operation used by a normal binary64 relative-error proof.

    The phase and angle error sums of QHD planning
    (``method._compiled_phase_allowance``, ``circuit_errors``) bound each
    product with a relative rounding error, which an underflowed product
    does not have, so every product they count must pass this rule.

    The stored operands are read as exact real numbers. A multiplication,
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

    A zero result is legal only when its operands or the mathematical
    structure of the computation establish zero (``zero``). A zero from nonzero
    operands is a rejection, even if a later operation would bring the exact
    value back into range, so each computation checks an operation before
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
    the omission policy of planning turns into an omitted contribution whose
    magnitude enters the error bound (``onehot_schedule`` module docstring,
    ``binary.PhaseTable.synthesize``, ``initial_state.chain_selection``). A
    nonfinite result or an exact result above the largest binary64 number
    still raises ValueError here, since no omission can account for it.
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
    """Return ``fl(x y)`` after ``_normal_range`` accepts it, with an exact zero only for a zero factor."""
    return _normal_range(x * y, name, lambda: Fraction(x) * Fraction(y), zero=x == 0 or y == 0)


def _normal_quotient(x, y, name):
    """Return ``fl(x/y)`` for a finite nonzero y after ``_normal_range`` accepts it, zero only for x = 0."""
    if y == 0 or not isfinite(y):
        raise ValueError(f"{name} divides by {y!r}, not a finite nonzero binary64 number")
    return _normal_range(x / y, name, lambda: Fraction(x) / Fraction(y), zero=x == 0)


def _normal_products(factor, values, name):
    """Return the elementwise ``fl(factor v)`` of an array after ``_normal_range`` accepts every entry.

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
    """Return ``(products, omitted)``, the elementwise ``fl(factor v)`` with the lower-range failures set to zero.

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

    ``Optimization`` already checks the ordered SymPy symbols, unique names,
    declared free symbols and one finite ``lower < upper`` interval per
    variable. QHD additionally needs the objective as a commutative SymPy
    expression object and commutative variables not declared non-real, because each grid value
    becomes a real diagonal entry of the Hamiltonian.
    """
    if not isinstance(problem.objective, sp.Expr):
        raise TypeError("objective must be a live SymPy expression")
    if problem.objective.is_commutative is not True:
        raise ValueError("optimization objective must be a commutative scalar expression")
    if any(v.is_commutative is not True or v.is_real is False for v in problem.variables):
        raise ValueError("optimization variables must admit commutative real scalar values")


def _kinetic_range_error(error):
    """Return the error for a kinetic coefficient, energy or angle outside the accepted range, with its remedy.

    The lower-range omission policy covers contributions formed from
    objective data and the initial state, and the phase products of binary
    kinetic blocks (``binary.PhaseTable.synthesize``).
    Kinetic coefficient and energy scales depend on the grid spacing h and,
    when weighted, the kinetic schedule weight a. One-hot kinetic angles also
    depend on the layer duration t, and the binary kinetic exponent is dt*a.
    Extreme spacings, weights or durations can put these quantities outside
    the accepted binary64 range, so planning rejects them.
    """
    return ValueError(f"{error}. The grid spacing h and the kinetic schedule weight a put this kinetic term, of "
                      "scale a/h**2, outside the range that planning admits. Choose a box and grid whose spacing "
                      "keeps it in range, or another total_time or schedule")


def _constant_power(node):
    """Return whether ``node`` is ``b**g`` with b a known finite positive real constant and g not a number.

    ``b**g = exp(g log(b))`` with the real logarithm of b and no branch choice, also for complex g. A zero, negative or
    variable base, or a symbol declared positive, is not such a b.
    """
    return (node.is_Pow and not node.exp.is_number and node.base.is_number and node.base.is_real is True
            and node.base.is_positive is True)


def _exponential_factor(node):
    """Return whether ``node`` is an ``exp(g)`` or a ``_constant_power`` whose exponent is not a number."""
    return (isinstance(node, sp.exp) and not node.exp.is_number) or _constant_power(node)


def _multiplicative_factors(node):
    """Return the factors of ``node``, read through nested ``Mul`` nodes, none of them rebuilt."""
    return [factor for argument in node.args for factor in _multiplicative_factors(argument)] if node.is_Mul else [node]


def _working_digits(numbers):
    """Return the decimal digits at which the logarithms of ``numbers`` are taken from their received values.

    At least ``_LOG_DIGITS``, and enough to hold every Float mantissa and every Rational in them, since 3 bits hold less
    than one decimal digit and a Rational gets ``_LOG_DIGITS`` more. Then a Float or Rational base near 1 keeps its distance
    from 1 and its logarithm is relatively accurate. ``log(Rational)`` under ``evalf`` is not used for this, because it
    returns 0 for ``1 + 10**-60``. ``_base_logarithm`` covers an exact compound base, whose atoms do not show its distance.
    """
    atoms = set().union(*(number.atoms(sp.Float, sp.Rational) for number in numbers))
    return max([_LOG_DIGITS, *(atom._prec // 3 if atom.is_Float
                               else _LOG_DIGITS + max(abs(atom.p), atom.q).bit_length() // 3 for atom in atoms)])


def _base_logarithm(base, digits):
    """Return ``log(base)`` of a known positive constant, taken from its distance from 1 when that distance is small.

    ``log(N(base, digits))`` keeps only the digits of the distance that follow the leading 1, which an exact compound base
    such as ``1 + exp(-100)`` can lose entirely. A base within ``10**(20 - digits)`` of 1 therefore uses
    ``log1p(N(base - 1, digits))``, and every other base keeps the direct form. SymPy's strict evaluation raises when it
    cannot resolve the distance, and that refuses here, with the remedy, because reading the base as 1 could be wrong by
    an exponent of any size.
    """
    near = sp.N(base, digits)
    if abs(near - 1) >= sp.Float(10) ** (20 - digits):
        return sp.log(near)
    try:
        distance = sp.N(base - 1, digits, strict=True)
    except PrecisionExhausted as error:
        raise ValueError(f"QHD cannot resolve the distance from 1 of the constant base {base} at {digits} digits. "
                         "Supply the base as a Float with enough digits, such as sp.N(base, 1000).") from error
    with mpmath.workdps(digits + 5):
        return sp.Float(mpmath.log1p(mpmath.mpf(str(distance))), digits)


def _combined_product(node):
    """Return a product or power with its exponential factors held as one exponential, or None when it stays as written.

    The group of ``node`` is its factors, read through nested ``Mul`` nodes. An eligible factor is ``exp(g)`` or ``b**g``
    with b a known finite positive real constant (``_constant_power``) and g not a number. A number-valued factor such as
    ``exp(10)``, ``sqrt(2)`` or ``pi**2`` is a constant. A sum, a power of a nonconstant product or a function is one other
    factor of the group, whatever it holds. ``_combine_exponentials`` visits it for its own groups and nothing moves
    across it. The group is combined when

    - it has two or more eligible factors, whatever its constants, or
    - it has one, and the product of its constant magnitudes below 1, the product of those above 1, or a constant base
      lies outside ``[2**-1022, Omega]``, the normal binary64 range. Every subset product of the constants lies between
      the two extreme products, so a literal that another constant compensates is caught, as in
      ``1e-324*exp(40)*exp(x)``, which reads as zero although the product of its constants is normal.

    A group with no eligible factor has nothing to repair, so ``x + 1e-310*y``, ``c*(exp(g) + y)`` and ``c*cosh(x)*cosh(y)``
    stay as written. So does a lone exponential with a normal constant. For ``T = C exp(g)`` with no other factor and no
    enclosing operation, ``|C| <= Omega``, and a correctly rounded exponential of an exact g, a zero result has
    ``|T| <= Omega 2**-1075``, about 4.4e-16, and an absolute error eps_g of g multiplies this by ``exp(eps_g)``. This
    bounds neither ``v*C*exp(g)`` nor an enclosing function, and it concerns the absolute value only.

    For ``T(x) = v(x) prod_k c_k prod_j exp(g_j(x)) prod_l b_l**h_l(x)``, with the c_k known finite nonzero real constants
    and the b_l known finite positive real constants,

        ``s = prod_k sign(c_k),  q(x) = sum_k log|c_k| + sum_j g_j(x) + sum_l h_l(x) log(b_l),  T(x) = v(x) s exp(q(x))``

    holds wherever v and the exponents have defined finite complex values. ``exp(a) exp(b) = exp(a + b)`` for finite complex
    a and b, with no logarithm of a or b and no branch to choose, ``c = sign(c) exp(log|c|)`` for real c, and
    ``b**h = exp(h log(b))`` for positive real b. So no exponent and no factor of v needs a proof that it is real, finite
    or polynomial. The one refusal here is of a constant base whose distance from 1 stays unresolved (``_base_logarithm``).
    The finite-real check of the caller reads the computed value.

    A SymPy Float has a larger exponent range than binary64, so each constant logarithm comes from the received value
    before any binary64 conversion, whose result can be zero or infinity, at the working precision of
    ``_working_digits``. A zero or non-real number is a factor of v, so no logarithm of zero is taken.

    Each exponent stays a numerical operand. It is one term of an unevaluated ``Add``, which the printer emits as one sum,
    so it is evaluated as the unrewritten expression evaluates it and its value is then added, with one partial sum and
    one operand live at a time. An evaluating ``Add`` would combine the exponent trees: ``exp(1/x)*exp(-1/x)`` would become
    ``exp(0)``, which is 1 at ``x = 0``, where the received expression is undefined. Here ``inf - inf`` gives NaN, which
    the check of the caller refuses. A real-array ``log`` or ``sqrt`` keeps its NaN, because no coordinate is cast to
    complex. The factors of v are not rebuilt either, so a held ``0/x`` keeps its NaN. This keeps the operations and their
    error behavior and is no domain validator: ``exp(-1/x)`` at ``x = 0`` is 0 with or without this function.

    The identity is exact and its binary64 evaluation is not. With an absolute error eps_q of the computed q, possibly
    complex, and a relative error eps_exp of the exponential,

        ``|E_hat - exp(q)| / |exp(q)| <= expm1(eps_q) + eps_exp exp(eps_q)``

    for a normal finite ``E_hat``. eps_q counts the constant logarithms, each g_j, the sum, and each computed ``h_l L_l``,
    which adds at most ``|L_l| eps_h + |h_l| eps_L + eps_L eps_h + eps_mul``. Normal finite factors with relative errors
    eps_v and eps_E and a multiplication error of at most u give ``(1 + eps_v)(1 + eps_E)(1 + u) - 1`` for ``v exp(q)``.
    Zeros and subnormal values have absolute accuracy only.

    The error that matters in an exponent is absolute and includes its phase, so regrouping can lose accuracy as well as
    repair range. ``exp(I*x)*exp(I*y)`` at ``x = 2**53``, ``y = 1`` rounds the phase sum by one radian where separate
    exponentials are accurate, and exponents near 1e308 can overflow while they add, which gives a finite zero where the
    exact product is 1. Cancellation between exponents stays in q. For ``2**(-1030)*exp(10)*exp(x)`` at ``x = 700`` the
    relative error moves from 2.2e-16 to 1.3e-14.

    v stays outside the exponential, so a v that itself compensates the range is not covered. ``q = 800`` with
    ``v = 1e-300`` overflows and ``q = 800`` with ``v = 0`` gives NaN, and the finite-real check of the caller stops both.
    ``q = -800`` with ``v = 1e300`` gives 0 against a true value near 3.7e-48.

    Args:
        node: A SymPy ``Mul`` or a ``_constant_power``.
    """
    factors = _multiplicative_factors(node)
    eligible = [factor for factor in factors if _exponential_factor(factor)]
    if not eligible:
        return None
    exponents = [factor.exp for factor in eligible if isinstance(factor, sp.exp)]
    powers = [factor for factor in eligible if not isinstance(factor, sp.exp)]
    constants, values, others = [], [], []
    for factor in factors:
        value = sp.N(factor, _LOG_DIGITS) if factor.is_number and factor.is_real else None
        if value is not None and value.is_Float and value != 0:
            constants.append(factor)
            values.append(value)
        elif not _exponential_factor(factor):
            others.append(factor)
    magnitudes = [abs(value) for value in values]
    extremes = [sp.Mul(*(magnitude for magnitude in magnitudes if magnitude > 1)),
                sp.Mul(*(magnitude for magnitude in magnitudes if magnitude < 1)),
                *(sp.N(power.base, _LOG_DIGITS) for power in powers)]
    if len(eligible) < 2 and all(_NORMAL_MIN <= extreme <= _NORMAL_MAX for extreme in extremes):
        return None
    digits = _working_digits([*constants, *(power.base for power in powers)])
    constant = sp.Add(*(sp.log(abs(sp.N(factor, digits))) for factor in constants))
    terms = [*([constant] if constant else []), *map(_combine_exponentials, exponents),
             *(sp.Mul(_combine_exponentials(power.exp), _base_logarithm(power.base, digits), evaluate=False)
               for power in powers)]
    sign = sp.Mul(*(sp.sign(value) for value in values))
    return sp.Mul(*([] if sign == 1 else [sign]), *map(_combine_exponentials, others),
                  _HELD_EXP(sp.Add(*terms, evaluate=False)), evaluate=False)


def _combine_exponentials(expression):
    """Return ``expression`` with each group that ``_combined_product`` combines replaced.

    This is scalar work on the expression: no grid pass, no expansion and no ``simplify``. A sum is treated term by term,
    and a function or a power is visited for its own groups. A combined group visits its exponents and the factors of v
    through this function. A nested ``Mul`` is read as part of the outer group, so ``(exp(400*y)*exp(400*z))*exp(-800*x)``
    becomes one exponential, where combining the inner pair first would evaluate ``exp(800)`` before the decaying factor.
    A subgroup of a group that stays as written has no more eligible factors and no wider extremes, so it stays too.

    The enclosing nodes of a combined group are rebuilt unevaluated, so Float arithmetic cannot change the rest of the
    expression. ``sp.evaluate`` clears SymPy's cache when it changes the setting, so it runs only when a group is combined.
    """
    combined = {}
    nodes = sp.preorder_traversal(expression)
    for node in nodes:
        if (node.is_Mul or _constant_power(node)) and (held := _combined_product(node)) is not None:
            combined[node] = held
            nodes.skip()
    if not combined:
        return expression
    with sp.evaluate(False):
        return expression.xreplace(combined)


def _lambdify_objective(variables: Sequence[sp.Symbol], objective: sp.Expr) -> Callable[..., Any]:
    """Return a NumPy function of ``variables`` in order, or raise ValueError when SymPy cannot.

    SymPy's NumPy printer raises KeyError for a constant it cannot print, such as ``zoo``.

    Every generated evaluator passes here after the transformations that form new constants, such as centering,
    refinement and the assembly of an augmented Lagrangian. Multiplicative groups of exponential factors, including
    powers of positive finite numerical bases, are regrouped before numerical exponentiation, the groups with at least two
    such factors and a single factor when a constant factor, constant subproduct or base leaves the normal binary64 range
    (``_combined_product``). This change is local to the evaluator, and the stored objective stays as it was received. It
    changes rounding and can change table values and Plan ids.

    It is no complete range repair. Cancellation, large phase sums, overflow in exponent arithmetic, final underflow and
    surrounding operations can still give inaccurate finite results or a nonfinite evaluation, also at ordinary values
    outside a group. ``exp(-800*x)*sinh(400*y)*sinh(400*z)`` gives 0 against 0.25 at ``(1, 1, 1)``, ``c*cosh(x)*cosh(y)``
    with c near 1e-400 gives zero, and a ``sqrt`` of a combined product can lose its value before the square root. The
    finite-real check of the caller rejects computed nonfinite values and nonzero imaginary parts, and a finite zero or a
    zero ``grid_minimum`` gap is no certificate against these errors.

    The table names the SciPy function behind each special function that an
    objective can use. A function that returns a tuple is read at the component
    that SymPy's function names. An entry keeps SymPy's real value under these
    premises, and ``tests/test_qhd_workflow.py`` compares each with mpmath at
    finite grid points of its box, which is evidence at those points and not a
    statement for every order, branch or singular point:

    - ``besselj(0, x)`` and ``besseli(2, x)``: integer order, real x.
    - ``besselj(1/2, x)``, ``bessely(1, x)``, ``besselk(0, x)``: x > 0. Zero is
      singular for the last two, and a negative argument needs branch treatment.
    - ``airyai``, ``airyaiprime``, ``airybi``, ``airybiprime``: real x, with no
      claim of a finite binary64 value at arbitrarily large arguments.
    - ``LambertW(x)``: principal branch, whose finite real domain starts at
      ``-1/e``. ``LambertW(x, -1)``: finite real on ``[-1/e, 0)``, with zero
      singular. The compared boxes stay away from ``-1/e``.
    - ``Si``, ``erfi``: real x. ``Ei``: real ``x != 0``, the real principal-value
      convention, so a grid that holds 0 is refused.
    - ``Ci``, ``loggamma``: positive real arguments.
    - ``erfinv``: the principal real inverse on ``(-1, 1)``. ``erfcinv``: on
      ``(0, 2)``. The ends of each are infinite.
    - ``polygamma(n, x)``: nonnegative integer order, compared at ``n = 1`` for
      x > 0. ``digamma``, which SymPy writes as ``polygamma(0, x)``, compared on
      ``(-0.9, -0.1)``, excludes the nonpositive integers. The name does not
      qualify a fractional or negative order.

    ``special.gammaln`` and ``special.sici`` are real at a negative argument, where
    ``Ci(x) = Ci(-x) + i pi`` and ``loggamma(x) = log|Gamma(x)| - n i pi`` on
    ``(-n, -n + 1)`` are not, so those two entries return NaN at a nonpositive
    argument and the finite-real check of the caller refuses the point. This is a
    restricted function contract, not proof that the whole objective is nonreal:
    ``re(loggamma(x))`` at ``x = -0.5`` is real and is refused too, because the NaN
    comes before the real part. An entry's NaN or complex value elsewhere, such as
    ``bessely`` at a negative argument or ``LambertW`` below ``-1/e``, is refused
    by the same check, and a NaN from a real SciPy routine need not mean that the
    complex function is undefined.

    A function outside the table keeps the printer's meaning, Python's ``math`` for
    ``erf``, ``erfc`` and ``gamma``, or is refused as an undefined name when it is
    evaluated. SymPy's own ``modules=["scipy", "numpy"]`` is not used because it
    brings in names whose values differ from SymPy's at real arguments. Its
    ``factorial`` is 0 at a negative argument where SymPy's is ``gamma(x + 1)``,
    and its ``jn`` and ``yn`` truncate a non-integer order to an integer.
    """
    import numpy as np
    from scipy import special

    functions = {
        "besselj": special.jv,
        "bessely": special.yv,
        "besseli": special.iv,
        "besselk": special.kv,
        "airyai": lambda x: special.airy(x)[0],
        "airyaiprime": lambda x: special.airy(x)[1],
        "airybi": lambda x: special.airy(x)[2],
        "airybiprime": lambda x: special.airy(x)[3],
        "LambertW": special.lambertw,
        "Si": lambda x: special.sici(x)[0],
        "Ci": lambda x: np.where(np.greater(x, 0), special.sici(x)[1], np.nan),
        "Ei": special.expi,
        "erfi": special.erfi,
        "erfinv": special.erfinv,
        "erfcinv": special.erfcinv,
        "polygamma": special.polygamma,
        "loggamma": lambda x: np.where(np.greater(x, 0), special.gammaln(x), np.nan),
        "_held_exp": np.exp,
    }
    objective = _combine_exponentials(objective)
    try:
        return sp.lambdify(variables, objective, modules=[functions, "numpy"])
    except (SyntaxError, TypeError, ValueError, NotImplementedError, KeyError) as exc:
        raise ValueError(f"objective cannot be lambdified: {exc}") from exc
