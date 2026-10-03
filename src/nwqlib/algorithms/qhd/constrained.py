"""Augmented-Lagrangian layer of the QHD family.

``solve_augmented_lagrangian`` solves a ``ConstrainedOptimization``

    minimize f(x) subject to h_i(x) = 0, g_j(x) <= 0, x in the box,

by a sequence of box problems. Round k searches the normalized effective
objective L_k (``_effective_objective``) with one new QHD Plan and the ordinary
Plan, Run and Result workflow, reads a point from the inner result
(``_choose``), and updates the multipliers and the penalty (``_update``). The
layer keeps only this outer state. Every inner solve keeps its own Plan, Run,
Result, archive and report. It is a function of the QHD family rather than a
Method, because each round changes the objective itself, so each round is a
new QHD Problem (``nwqlib.scientist.compare`` also organizes several Plans
outside a Method). With ``refinement=BoxRefinement(...)`` each round runs box
refinement (``refinement.refine_box``) on its inner objective instead of one solve
and takes the projection of the point of the level with the least recorded
relative value of that objective (``_refined_choice``). ``_outer.refinement_stop`` states
how the refinement's stop ends a round and the run. With ``directory`` every
inner Run is durable, and ``resume_augmented_lagrangian`` continues an
interrupted run (``_durable``).

The equality terms are Wu et al. arXiv:2605.12066v1, Eqs. (6)-(8). The
inequality terms are the Powell-Hestenes-Rockafellar (PHR) form of
Rockafellar, doi:10.1007/BF01580138. The multiplier, penalty and stopping
rules follow Birgin and Martinez, *Practical Augmented Lagrangian Methods for
Constrained Optimization*, SIAM 2014, doi:10.1137/1.9781611973365 (below "the
book"): Algorithm 4.1 (Sec. 4.1, pp. 33-34, Eqs. (4.6)-(4.9)) and the stopping
test of Algencan (Sec. 10.2.2, Eqs. (10.6)-(10.8), pp. 116-117). Term by
term, L_k is the book's Eq. (10.3) (Sec. 10.1, p. 114) on the normalized f, h
and g (``_effective_objective``).

In the book's notation the problem is its (4.1). The box is the lower-level
set Omega, whose constraints the book calls hard or nonrelaxable (p. 34), and
h and g are the constraints that the multipliers and the penalty relax. Each
inner QHD Plan searches a finite grid in the box, so QHD takes the place of
the subproblem solver of Step 1 of Algorithm 4.1 on that finite subset of
Omega. Step 1 asks only for an approximate minimizer of L_k on Omega and
leaves its accuracy deliberately open (p. 34). The book's convergence results
add a condition on that point: an approximate global minimizer on Omega
(Chapter 5, Assumption 5.1, p. 41) or an approximate KKT point of the
subproblem in the sense of Assumption 6.1 (Chapter 6, p. 48). No point rule
of this layer (``_choose``) certifies either condition for the box. Two
options also depart from Algorithm 4.1 as those results assume it.
``max_penalty`` caps rho, where Step 3 multiplies it by gamma without limit,
and the default ``multiplier_bounds=None`` omits the safeguard of Step 4.
These results are therefore not claimed for a run.

The layer searches a finite grid. A stopping status states which numerical
tests held and does not certify optimality, and finite-grid multipliers are
dual iterates of the discrete problem, not the continuous KKT multipliers.
``constrained_grid_minimum`` is an explicit comparison with freshly
evaluated, tolerance-feasible grid values. It supplies neither an
objective-evaluation error bound nor a continuous optimality certificate.
The stationarity diagnostic is evidence about the continuous first-order
condition. docs/algorithms/qhd.md, "Constrained problems", explains the
choices for readers.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import fsum, inf, isfinite, nextafter
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any, Callable

import sympy as sp

from nwqlib.problems.records import ConstrainedOptimization
from ._coverage import DEFAULT_FAILURE_PROBABILITY, coverage, coverage_lines, validate_alpha
from .constrained_records import (
    ALEvaluation,
    ALIteration,
    ALResources,
    AbsorbedBound,
    AugmentedLagrangian,
    AugmentedLagrangianRecord,
    ConstrainedGridMinimum,
    ConstraintPreprocessing,
    DEFAULT_INEQUALITY_FEASIBILITY_TOLERANCE,
    FormTrial,
    GRID_MINIMUM_MAX_BYTES,
    GRID_MINIMUM_MAX_WORK,
    InnerRepresentation,
    grid_convention,
    SlackAxis,
)
from ._outer import (
    RUN_COUNTS,
    USED_COUNTS,
    HeaderlessRun,
    check_arguments,
    law_count,
    grid_point,
    inner_failure,
    mean_point,
    plan_round,
    range_bound,
    refinement_stop,
    round_limits,
    run_counts,
    total_counts,
)
from ._durable import (
    FORMATS,
    RECORD,
    RESUME,
    Directory,
    Frontier,
    closed_trace,
    live_problem,
    reopen,
    require_new,
    saved_result,
    unrecoverable,
)
from .grid import OneHotGrid
from .objective import ObjectiveDecomposer, centered_objective, expansion_centers, monomial_bound, node_count
from .potential import coerce_real_scalar
from .refinement import ARCHIVE_RECORD as REFINEMENT_RECORD, _refine, check_refinement_options
from .refinement_records import BoxRefinement, RefinementLevel

ARCHIVE_FORMAT = "qhd.constrained/4"

# Functions whose derivative is undefined or discontinuous somewhere. The
# stationarity diagnostic is not applicable to a problem that contains one,
# because the layer does not locate their kinks. sqrt and similar functions,
# such as other fractional powers and asin, are differentiable on the interior
# of their real domain and can fail to be differentiable at its boundary, where
# they can be finite while their derivative is not (sqrt at 0). The evaluated
# derivative is then not finite, which makes the diagnostic unavailable at that
# point.
_NONSMOOTH = (sp.Abs, sp.sign, sp.Max, sp.Min, sp.Heaviside, sp.floor, sp.ceiling, sp.frac, sp.Mod,
              sp.Piecewise, sp.re, sp.im, sp.arg, sp.DiracDelta)


@dataclass(frozen=True)
class _Term:
    """One of f, h_i or g_j: its expression, its scale and the function the layer evaluates.

    ``function`` is the expression lambdified once over all variables in
    order. ``support`` holds the positions of the variables it contains, and
    ``nodes`` its tree size (``objective.node_count``), the work of one
    evaluation apart from its d coordinates. ``gradient`` holds one
    ``(function, nodes)`` pair per variable, or None per variable absent from
    the expression, when the stationarity diagnostic applies.
    """

    name: str
    expression: Any
    scale: float
    function: Callable[..., Any]
    support: tuple[int, ...]
    nodes: int
    gradient: tuple | None


@dataclass(frozen=True)
class _Setup:
    """Everything the rounds share, derived once from the problem, the QHD configuration and the options.

    ``stationarity_unavailable`` is None when the stationarity diagnostic
    applies and otherwise the reason it does not (``_derivatives``).
    ``slack_symbols`` holds the slack variable that each kept inequality
    receives when a round converts it (``_slack_symbols``).
    """

    options: AugmentedLagrangian
    grid: OneHotGrid
    objective: _Term
    equalities: tuple[_Term, ...]
    inequalities: tuple[_Term, ...]
    preprocessing: ConstraintPreprocessing
    feasibility: float
    complementarity: float
    stationarity_unavailable: str | None
    check_evaluations: int
    check_work: int
    slack_symbols: tuple


def _live_constraints(problem):
    """Require live commutative SymPy constraints, naming the first that is not."""
    for field in ("equalities", "inequalities"):
        for position, expression in enumerate(getattr(problem, field)):
            if not isinstance(expression, sp.Expr):
                raise TypeError(f"{field}[{position}] must be a live SymPy expression. A saved problem "
                                "description cannot be evaluated")
            if expression.is_commutative is not True:
                raise ValueError(f"{field}[{position}] must be a commutative scalar expression")


def _decided_less(p, q):
    """Return whether ``p < q`` for real SymPy numbers, or None when SymPy cannot decide it."""
    relation = sp.Lt(p, q)
    return True if relation is sp.true else False if relation is sp.false else None


def _binary64(value):
    """Return the binary64 number equal to the exact real ``value``, or None when there is none."""
    if not value.is_Rational:
        return None
    exact = Fraction(int(value.p), int(value.q))
    try:
        number = float(exact)
    except OverflowError:
        return None
    return number if Fraction(number) == exact else None


def _absorb(problem, positions):
    """Tighten the box by one-variable linear inequalities, the bound absorption of the options.

    Such an inequality bounds one coordinate, so instead of staying among the
    constraints that the rounds relax, it can tighten the box, the book's
    lower-level set Omega. The book poses the comparison of the two
    placements as its Problem 4.6 (p. 38). Every point of the tightened box
    satisfies the absorbed constraint in exact arithmetic, and the rounds
    carry no penalty term and no multiplier for it. Coordinate rounding can
    place the last point of a Dirichlet endpoint grid slightly above an
    absorbed upper bound (``grid.OneHotGrid.__post_init__``). The grid and
    the dynamics change with the box, so
    ``AugmentedLagrangian.absorb_bounds`` is off by default.

    An inequality ``g(x) = a x_j + b <= 0`` that contains only x_j, with real
    coefficients and a nonzero a whose sign SymPy decides, is the cut
    ``x_j <= q`` for ``a > 0`` and ``x_j >= q`` for ``a < 0``, with the exact
    quotient ``q = -b/a``. Every binary64 ``Float`` of the inequality is
    replaced by its exact rational before the coefficients are read, so a
    polynomial over the rationals, not over binary64, gives a and b, and q
    is exact whenever a and b are rational.

    Each variable's feasible interval is classified before any rounding: the
    original box side, whose bounds are binary64 numbers and hence exact
    rationals, intersected with every cut on that variable is empty, a single
    point, or of positive width. Empty means the problem has no feasible
    point in its box. A single point fixes the variable, which the grid
    cannot represent, since it needs ``lower < upper``, so the remedy is to
    substitute the value. Rounding the cuts first would merge these cases.
    On the box ``[1/2, 1]`` the upper cuts ``1/2 - 2**-56``, ``1/2`` and
    ``1/2 + 2**-56`` give an empty interval, a point and a positive width,
    and all three round to 0.5.

    With positive width, a cut whose exact value is a binary64 number
    (``_binary64``) becomes the box bound, and the constraint then holds on
    the whole tightened box, so it is absorbed. A cut with no binary64 value
    stays a constraint of the rounds, with the box bound unchanged, because
    rounding it outward would admit infeasible points and rounding it inward
    would remove feasible ones. The tightened box contains the exact
    interval, so it keeps a positive width. A cut whose sign or comparisons
    SymPy leaves undecided also stays a constraint, and so does every cut of
    a variable whose classification is undecided.

    Returns:
        ``(bounds, absorbed, kept)``: the tightened box, the AbsorbedBound
        records and the problem positions of the inequalities not absorbed,
        both in problem order.

    Raises:
        ValueError: A variable's exact interval is empty or a single point.
    """
    cuts, kept = {}, []
    for position in positions:
        expression = problem.inequalities[position]
        expression = expression.xreplace({f: sp.Rational(f) for f in expression.atoms(sp.Float)})
        support = [j for j, v in enumerate(problem.variables) if v in expression.free_symbols]
        linear = None
        if len(support) == 1:
            try:
                poly = sp.Poly(expression, problem.variables[support[0]])
            except sp.polys.polyerrors.BasePolynomialError:
                poly = None
            if poly is not None and poly.degree() == 1:
                a, b = poly.all_coeffs()
                if a.is_extended_real and b.is_extended_real and (a.is_positive or a.is_negative):
                    linear = a, b
        if linear is None:
            kept.append(position)
            continue
        a, b = linear
        cuts.setdefault(support[0], []).append((position, "upper" if a.is_positive else "lower", -b / a))
    bounds = [list(pair) for pair in problem.bounds]
    absorbed = []
    for j, items in sorted(cuts.items()):
        # The exact interval [low, high]: the box side and every cut whose
        # comparison with the running bound SymPy decides.
        low, high = sp.Rational(bounds[j][0]), sp.Rational(bounds[j][1])
        classified = []
        for position, side, q in items:
            tighter = _decided_less(low, q) if side == "lower" else _decided_less(q, high)
            if tighter is None:
                kept.append(position)
                continue
            classified.append((position, side, q))
            if tighter:
                low, high = (q, high) if side == "lower" else (low, q)
        names = ", ".join(f"inequalities[{p}]" for p, _, _ in classified)
        name = problem.variable_names[j]
        below, above = _decided_less(low, high), _decided_less(high, low)
        if above:
            raise ValueError(f"absorbing {names} leaves {name} no value in the box ({low} > {high}), so the "
                             "problem is infeasible in its box")
        if below is False and above is False:
            value = _binary64(low)
            raise ValueError(f"absorbing {names} fixes {name} = {low if value is None else value}. The grid "
                             "needs lower < upper, so this representation does not support a fixed variable. "
                             "Substitute the value and restate the problem, or keep the constraint with "
                             "absorb_bounds=False")
        for position, side, q in classified:
            value = _binary64(q) if below else None
            if value is None:
                kept.append(position)
                continue
            k = 0 if side == "lower" else 1
            bounds[j][k] = max(bounds[j][k], value) if side == "lower" else min(bounds[j][k], value)
            absorbed.append(AbsorbedBound(inequality=position, variable=j, side=side, value=value))
    absorbed.sort(key=lambda item: item.inequality)
    return tuple(tuple(pair) for pair in bounds), tuple(absorbed), tuple(sorted(kept))


def _counted(given, count, name, default):
    """Return one option value per constraint, ``(default,) * count`` when ``given`` is None."""
    if given is None:
        return (default,) * count
    if len(given) != count:
        raise ValueError(f"{name} needs one value per constraint ({count}), got {len(given)}")
    return given


def _original_multiplier(value, objective_scale, scale, name):
    """Return the scaled problem's multiplier ``value`` in original units, ``s_f value / s``, or raise.

    Multiplying the scaled problem's KKT condition ``grad f/s_f + sum_i
    lambda~_i grad h_i/s_{h_i} + sum_j mu~_j grad g_j/s_{g_j} = 0`` by s_f
    gives the original condition with ``lambda_i = (s_f/s_{h_i}) lambda~_i``
    and ``mu_j = (s_f/s_{g_j}) mu~_j``, where ``scale`` is the constraint's
    s. The complete product of the three binary64 operands is formed exactly
    as a ``Fraction`` and rounded once to nearest, so a ratio ``s_f/s``
    outside the binary64 range cannot hide a representable product. With
    s_f = 1e200, s = 1e-200 and value 1e-200, forming the ratio first gives
    inf, and inf times a zero multiplier gives nan, while the product is
    1e200 and a zero multiplier stays 0. A zero multiplier converts to 0
    exactly. A nonzero product z must be a normal binary64 number,
    ``2**-1022 <= |z| <= sys.float_info.max``, so that its rounding is
    relative, at most ``2**-53 |z|``, and otherwise ValueError names the
    multiplier. This is the range rule of QHD planning
    (``validation._normal_range``) applied to one exact three-operand
    product. It stays inline because that helper checks a rounded binary64
    result, which a product above the largest float does not have, and
    because this message reports the product's binary exponent.
    ``ConstrainedQHDResult.multipliers`` converts every multiplier it
    publishes this way. The rounds use only normalized multipliers, so no
    decision needs this conversion, and a multiplier without a normal
    original-unit value, an initial one included, stops no run.
    """
    if value == 0.0:
        return 0.0
    exact = Fraction(objective_scale) * Fraction(value) / Fraction(scale)
    if not Fraction(sys.float_info.min) <= abs(exact) <= Fraction(sys.float_info.max):
        exponent = abs(exact).numerator.bit_length() - abs(exact).denominator.bit_length()
        raise ValueError(f"{name} = {value!r} in normalized units is about 2**{exponent} in original units "
                         f"(objective scale {objective_scale!r}, constraint scale {scale!r}), outside the normal "
                         "binary64 range, so its original-unit value cannot be given")
    return float(exact)


def _value(term, point):
    """Evaluate ``term`` at ``point`` as a finite real float, or raise ValueError naming the term and point."""
    try:
        return coerce_real_scalar(term.function(*point), context=term.name)
    except (ArithmeticError, ValueError, TypeError, NameError) as error:
        raise ValueError(f"{term.name} = {term.expression} at {tuple(point)} is not a finite real "
                         f"value: {error}") from error


def _setup(problem, qhd, options, refinement=None, stored=None):
    """Check the problem, preprocess its constraints and check f, h and g on the grid, before any round.

    ``ConstrainedOptimization`` admits constant residuals such as ``nan`` or
    ``zoo`` and complex expressions such as ``I*x``. The layer checks every
    term on the grid of the preprocessed box by its support decomposition,
    the decomposition that QHD planning tabulates for an objective
    (``objective.ObjectiveDecomposer``, expanded about
    ``objective.expansion_centers`` of the box): ``term = c0 + sum_A T_A(x_A)``
    with one table per variable support A over its ``K**|A|`` grid tuples
    (``_support_tables``). Each table value and c0 must be finite and real.
    The range check of their represented sum (``_range_certificate``)
    requests an original-evaluation check when inconclusive. That check
    scans the supplied additive summands and bounds their additions, with
    an admitted whole-term scan when the bound is inconclusive. The layer
    raises with the term's position on a failed numerical evaluation.
    A one-variable support table or original summand costs K evaluations.
    An irreducible summand on r variables requires K**r evaluations and can
    be refused by work admission.

    The initial-grid check inventories possible restrictions in the
    supplied expression before centering, separately for its top-level
    additive summands. An identical restriction is covered only by an
    unconditional occurrence in the represented expression written in
    original coordinates. An uncovered restriction requests scans of all
    distinct supplied summands of that term, each on its own support grid
    at the original binary64 coordinates. Piecewise, Mul, Pow and function
    applications stay intact. A failed call or a nonfinite or nonreal
    selected value refuses before planning. Completed scans bound each
    summand's magnitude, and ``_addition_certificate`` includes rounding
    in its bound on the additions. An inconclusive represented range uses
    the same scans without charging or running one twice. An inconclusive
    addition certificate requests an admitted whole-term scan.

    Finite rounded values do not certify the exact real domain. Terms
    checked only by their represented tables can still fail in their
    original arithmetic. The initial check covers the preprocessed grid.
    Rounds evaluate the original f, h and kept g at their selected point
    and any attempted mean. A failed mean records mean_unavailable and
    falls back to the grid point. Refinement checks each compared point
    at every level, including the two points that a stall split scores. A
    failure at a selected grid point or at a scored point follows the
    existing failure path (_outer.inner_failure).

    The same tables give, for every kept inequality, the bounds
    ``ell_j = c0 + sum_A min T_A`` and ``M_j = c0 + sum_A max T_A`` of
    ``g_j/s_{g_j}`` on the grid (``_inequality_range``, Proposition 54, "A
    lower bound from support tables"), which ``ConstraintPreprocessing``
    records and the slack representation reads (``_round_problem``).

    The monomial count of each expansion (``objective.monomial_bound``), the
    table work ``K**|A| (N_A + |A|)`` of each support with N_A tree nodes, as
    QHD planning counts it (``QHD._admit_symbolic_work``), the
    work ``K**|S_j| (N_j + |S_j|)`` for each distinct supplied summand
    scanned on its support S_j, any additional whole-term fallback work
    ``K**|S| (N + |S|)``, and every later evaluation by the layer are admitted
    against ``qhd.max_work`` before the work they count
    (``_admit_layer_work``). Write C for the work of evaluating f and every
    kept constraint at one point, ``C = sum_t (N_t + d)`` over those terms t
    with ``N_t`` tree nodes, and ``a = 2`` for ``mode_or_mean`` and 1
    otherwise. Without box refinement (``refinement`` None) a round evaluates
    f, h and g at its point and, for ``mode_or_mean``, at the mean, and
    reserves ``a C``. With box refinement each attempted level checks f, h
    and g at the projection of its grid point and, for ``mode_or_mean``, of
    its mean, each attempted stall split checks them at the projections of
    its two scored points (``_level_check``), and the round evaluates them
    once more at its point. At most L = ``refinement.max_levels`` levels are
    attempted. A split is attempted only below the last allowed level and
    while fewer than ``max_splits`` splits have been made, and a failed side
    check ends the refinement, so at most
    ``B = min(max_splits, max(L - 1, 0))`` splits are attempted with
    ``stall_split="best_region"``, and B = 0 otherwise. A round therefore
    reserves ``(a L + 2 B + 1) C``, and the stationarity diagnostic adds its
    derivative evaluations. The check covers the grid of the preprocessed
    box, the first level's grid, and not the grids of later levels.

    ``stored``, for a resumed durable run, holds the preprocessing and the
    check counts that the first call saved in the controller's settings
    (``ConstraintPreprocessing`` JSON, ``check_work`` and
    ``check_evaluations``). The support tables, summand scans and range
    checks are then not evaluated again. Only the parts that need no
    function evaluation are rebuilt: the checks of the arguments, the bound
    absorption, whose box, absorbed bounds and kept inequalities must equal
    the saved ones, the lambdified terms and derivatives, the grid and the
    admission of the reserved round work with the saved check work. The
    saved preprocessing and check counts are read as the layer wrote them,
    as resume reads the directory's other files, and are not recomputed to
    detect an edit.

    Before any evaluation the initial multipliers are checked as the rounds
    will use them. Algorithm 4.1 of Birgin and Martinez starts from
    multipliers inside the safeguard box. ``AugmentedLagrangian._domains``
    checks the given values, and omitted equality multipliers start at 0,
    which ``multiplier_bounds`` can exclude, so a problem with equalities
    then needs explicit ones. Omitted inequality multipliers start at 0,
    which ``[0, inequality_upper]`` always contains. Every decision uses the
    normalized initial multipliers, so their original-unit values need not
    lie in the normal binary64 range. ``ConstrainedQHDResult.multipliers``
    with ``tentative=False`` converts them when round 0 is the last round
    and raises for such a value (``_original_multiplier``).
    """
    from .validation import _lambdify_objective, _validate_qhd_problem_fields

    if not isinstance(problem, ConstrainedOptimization):
        raise TypeError("solve_augmented_lagrangian requires a ConstrainedOptimization")
    _validate_qhd_problem_fields(problem)
    _live_constraints(problem)
    equality_scales = _counted(options.equality_scales, len(problem.equalities), "equality_scales", 1.0)
    inequality_scales = _counted(options.inequality_scales, len(problem.inequalities), "inequality_scales", 1.0)
    # Only their counts are checked here, before any work. Round 0 reads the values.
    _counted(options.equality_multipliers, len(problem.equalities), "equality_multipliers", 0.0)
    _counted(options.inequality_multipliers, len(problem.inequalities), "inequality_multipliers", 0.0)
    bounds = options.multiplier_bounds
    if (bounds is not None and problem.equalities and options.equality_multipliers is None
            and not bounds.equality_lower <= 0.0 <= bounds.equality_upper):
        raise ValueError("initial multipliers must lie inside multiplier_bounds, where Birgin-Martinez Algorithm "
                         "4.1 starts them, and equality_multipliers=None starts every equality multiplier at 0, "
                         f"outside [{bounds.equality_lower}, {bounds.equality_upper}]. Give initial "
                         "equality_multipliers within the bounds")
    positions = tuple(range(len(problem.inequalities)))
    box, absorbed, kept = (_absorb(problem, positions) if options.absorb_bounds
                           else (problem.bounds, (), positions))
    from .initial_state import GaussianState

    if options.inequality_form == "slack" and kept and isinstance(qhd.initial_state, GaussianState):
        raise ValueError("inequality_form='slack' adds a slack variable per kept inequality, and the QHD "
                         "initial_state GaussianState(center, widths) has one center and width per original "
                         "variable and no extension to the slack axes, which no QHD option configures. Use "
                         "UniformState or KineticGroundState, whose factors cover every axis, or "
                         "inequality_form='phr' or 'auto', which keeps the PHR form with this state")
    feasibility = options.feasibility_tolerance
    if feasibility is None:
        if problem.equalities:
            raise ValueError("feasibility_tolerance is required for a problem with equality constraints: "
                             "the residual a finite grid can reach depends on the grid and the constraint "
                             "(docs/algorithms/qhd.md, 'Constrained problems')")
        feasibility = DEFAULT_INEQUALITY_FEASIBILITY_TOLERANCE
    complementarity = (feasibility if options.complementarity_tolerance is None
                       else options.complementarity_tolerance)
    variables = problem.variables
    d, k = len(variables), qhd.num_grid_points
    named = [("objective", problem.objective, options.objective_scale)]
    named += [(f"equalities[{i}]", problem.equalities[i], equality_scales[i])
              for i in range(len(problem.equalities))]
    named += [(f"inequalities[{j}]", problem.inequalities[j], inequality_scales[j]) for j in kept]
    supports = [tuple(j for j, v in enumerate(variables) if v in expression.free_symbols)
                for _, expression, _ in named]
    # Tree sizes are counted up to max_work + 1 nodes, so each count stays
    # within the work it admits.
    nodes = [node_count(expression, qhd.max_work) for _, expression, _ in named]
    # One evaluation of f and every kept constraint at a point costs C units. A round without refinement
    # evaluates at its point and, for mode_or_mean, at the mean, a C. With refinement each of at most L levels
    # checks its grid point and, for mode_or_mean, its mean, each of at most B attempted stall splits checks its
    # two scored points (_level_check), and the round evaluates its point once more, (a L + 2 B + 1) C. No
    # split is attempted at the last allowed level, and a failed side check ends the refinement, so
    # B = min(max_splits, max(L - 1, 0)) with splitting and 0 without (the docstring above).
    per_point, a = sum(n + d for n in nodes), 2 if options.inner_point == "mode_or_mean" else 1
    if refinement is None:
        round_work = a * per_point
    else:
        levels = refinement.max_levels
        splits = (min(refinement.max_splits, max(levels - 1, 0)) if refinement.stall_split == "best_region"
                  else 0)
        round_work = (a * levels + 2 * splits + 1) * per_point
    _admit_layer_work(qhd, options, 0, round_work)
    unavailable, derivatives = _derivatives(named, supports, variables, options.stationarity)
    if unavailable is None:
        derivative_nodes = [[None if e is None else node_count(e, qhd.max_work) for e in row]
                            for row in derivatives]
        round_work += sum(n + d for row in derivative_nodes for n in row if n is not None)
        _admit_layer_work(qhd, options, 0, round_work)
    # Lambdify each term, and each needed derivative, once for the whole run.
    terms = []
    for position, (name, expression, scale) in enumerate(named):
        try:
            function = _lambdify_objective(variables, expression)
            gradient = None if unavailable is not None else tuple(
                None if e is None else (_lambdify_objective(variables, e), n)
                for e, n in zip(derivatives[position], derivative_nodes[position], strict=True))
        except ValueError as error:
            raise ValueError(f"{name} = {expression}: {error}") from error
        terms.append(_Term(name, expression, scale, function, supports[position], nodes[position], gradient))
    # The grid of every inner QHD Plan on the preprocessed box, with the
    # Method's boundary (grid.OneHotGrid defines the three grids).
    grid = OneHotGrid(problem.variable_names, box, k, qhd.include_boundary_points, qhd.boundary)
    if stored is not None:
        # A resumed run reads the preprocessing and the check counts that its first call saved, rebuilding
        # only what needs no function evaluation: the grid, the lambdified terms and the reserved round work.
        preprocessing = ConstraintPreprocessing.model_validate(stored["preprocessing"])
        if (preprocessing.original_bounds != tuple(tuple(side) for side in problem.bounds)
                or preprocessing.bounds != tuple(tuple(side) for side in box)
                or preprocessing.absorbed != tuple(absorbed) or preprocessing.inequalities != tuple(kept)
                or preprocessing.equalities != tuple(range(len(problem.equalities)))):
            raise ValueError("the saved constraint preprocessing differs from the problem's bound absorption")
        check_work, check_evaluations = stored["check_work"], stored["check_evaluations"]
        if any(type(count) is not int or count < 0 for count in (check_work, check_evaluations)):
            raise ValueError("the saved setup check counts are not nonnegative integers")
        _admit_layer_work(qhd, options, check_work, round_work)
        count = len(preprocessing.equalities)
        symbols = _slack_symbols(problem)
        return _Setup(options, grid, terms[0], tuple(terms[1:1 + count]), tuple(terms[1 + count:]), preprocessing,
                      feasibility, complementarity, unavailable, check_evaluations, check_work,
                      tuple(symbols[j] for j in kept))
    centers = expansion_centers(box)
    # Admit every expansion, support table and structurally requested summand scan with the reserved
    # round work before evaluation. The later range check admits any further scans before they run.
    check_work = check_evaluations = 0
    decompositions = []
    for term in terms:
        centered = centered_objective(term.expression, variables, centers)
        try:
            check_work += monomial_bound(centered,
                                         lambda count: _admit_layer_work(qhd, options, check_work + count, round_work,
                                                                         lower_bound=True))
        except ValueError as error:
            raise ValueError(f"expanding {term.name}: {error}") from error
        decomposer = ObjectiveDecomposer(term.expression, variables, centers)
        expressions, records = _summand_records(term, variables, qhd.max_work)
        represented = (*decomposer.support_expressions.values(), decomposer.constant)
        if any(decomposer.centers):
            unshift = {v: v - sp.Rational(m)
                       for v, m in zip(variables, decomposer.centers, strict=True) if m}
            represented = tuple(e.xreplace(unshift) for e in represented)
        uncovered = tuple(expression for expression in records if any(
            not any(_contains_unconditional(e, node) for e in represented)
            for node in _restrictions(expression)))
        for expression in decomposer.support_expressions.values():
            width = sum(1 for v in variables if v in expression.free_symbols)
            tuples = k**width
            limit = max(0, qhd.max_work - options.max_iterations * round_work - check_work) // tuples
            check_work += tuples * (node_count(expression, limit) + width)
            check_evaluations += tuples
            _admit_layer_work(qhd, options, check_work, round_work, lower_bound=True)
        # Coverage requests a proof of this original term. Until a companion
        # has its own original-evaluation certificate, scan that companion too.
        requested = tuple(records.values()) if uncovered else ()
        for summand in requested:
            tuples = k ** len(summand.support)
            check_work += tuples * (summand.nodes + len(summand.support))
            check_evaluations += tuples
            _admit_layer_work(qhd, options, check_work, round_work)
        # The bytes of this term's support tables and requested scans, before any term is evaluated.
        _admit_layer_work(qhd, options, check_work, round_work, grid=grid,
                          scans=((term, decomposer), *((summand, None) for summand in requested)))
        decompositions.append((decomposer, expressions, records, requested))
    lower, upper = [], []
    for position, (term, (decomposer, expressions, records, requested)) in enumerate(
            zip(terms, decompositions, strict=True)):
        maxima = {summand.expression: _domain_check(summand, variables, grid, qhd) for summand in requested}
        constant, tables = _support_tables(term, decomposer, grid, qhd)
        needs_range_check = not _range_certificate(constant, tables)
        if requested or needs_range_check:
            missing = tuple(summand for expression, summand in records.items() if expression not in maxima)
            # Admit the entire missing batch before any scan in that batch.
            for summand in missing:
                tuples = k ** len(summand.support)
                check_work += tuples * (summand.nodes + len(summand.support))
                check_evaluations += tuples
                _admit_layer_work(qhd, options, check_work, round_work, grid=grid, scans=((summand, None),))
            for summand in missing:
                maxima[summand.expression] = _domain_check(summand, variables, grid, qhd)
            # With one supplied summand this is already the whole-term scan,
            # independent of the raw scalar's eligibility for composition.
            checked_original = len(expressions) == 1 or _addition_certificate(
                tuple(maxima[expression] for expression in expressions))
            if not checked_original:
                tuples = k ** len(term.support)
                check_work += tuples * (term.nodes + len(term.support))
                check_evaluations += tuples
                _admit_layer_work(qhd, options, check_work, round_work, grid=grid, scans=((term, None),))
                _domain_check(term, variables, grid, qhd)
        if position > len(problem.equalities):
            low, high = _inequality_range(constant, tables, term.scale)
            lower.append(low)
            upper.append(high)
    preprocessing = ConstraintPreprocessing(original_bounds=problem.bounds, bounds=box, absorbed=absorbed,
                                            equalities=tuple(range(len(problem.equalities))), inequalities=kept,
                                            inequality_lower=tuple(lower), inequality_upper=tuple(upper))
    count = len(preprocessing.equalities)
    symbols = _slack_symbols(problem)
    return _Setup(options, grid, terms[0], tuple(terms[1:1 + count]), tuple(terms[1 + count:]), preprocessing,
                  feasibility, complementarity, unavailable, check_evaluations, check_work,
                  tuple(symbols[j] for j in kept))


def _slack_symbols(problem):
    """Return one real slack Symbol per inequality of ``problem``, by problem position, with collision-free names.

    The name of inequality j is ``s_g<j>``, followed by as many underscores
    as make it differ from every variable of the problem, every symbol of
    its objective and constraints and every earlier slack, so a problem that
    already uses ``s_g0`` gets ``s_g0_``. The names depend on the problem
    alone, so a resumed or loaded run rebuilds the same symbols
    (``InnerRepresentation.variables`` records them).
    """
    used = {str(v) for v in problem.variables}
    for expression in (problem.objective, *problem.equalities, *problem.inequalities):
        used |= {str(symbol) for symbol in expression.free_symbols}
    symbols = []
    for j in range(len(problem.inequalities)):
        name = f"s_g{j}"
        while name in used:
            name += "_"
        used.add(name)
        symbols.append(sp.Symbol(name, real=True))
    return tuple(symbols)


def _support_tables(term, decomposer, grid, qhd):
    """Return ``(c0, extrema)``: the constant and the ``(minimum, maximum, magnitude)`` of each support table of ``term``.

    ``decomposer`` is the term's expansion about the centers of the box
    (``_setup``). Each support expression is lambdified over its own
    variables and evaluated on arrays at the centered coordinates ``x - m``
    of its support grid (``compiler.evaluate_support``), the evaluator of an
    objective table in ``compiler.QHDCompiler``, whose outputs must be
    finite and real, so the table is the selected data of this check. A
    warning or nonfinite intermediate from an inactive Piecewise branch
    does not invalidate a finite real selected table value. Nonfinite or
    nonreal selected outputs are refused. Original-expression scans and
    checks at reported points use their own evaluation and may still fail.
    Array evaluations may differ from scalar evaluations by an ulp or more, and no
    universal evaluation-error bound exists for arbitrary lambdified
    expressions. The extrema are selected from each table
    (``records.table_extrema``), and the range of their sums is checked
    separately (``_range_certificate``, ``_inequality_range``). The constant
    c0 is converted to binary64 as ``QHD.plan`` converts an objective's
    constant.

    ``_admit_layer_work`` admits the support-table payloads of every term
    against the QHD byte allowance before any table is evaluated, and this
    function evaluates in the chunks of that admission
    (``_evaluation_chunks``). For a support of s variables,
    ``E = K**s`` entries and N expression nodes, the evaluation of a chunk of
    b entries is charged ``B_scan(b) = W_eval(b) + (8s + 64) b + H`` bytes
    (``compiler.support_chunk_size``), with the engineering allowance
    ``W_eval(b) = 8 b (N + 2)`` (``compiler.support_workspace``), not a
    derived bound, whose premise is that the printer's evaluation holds at
    most one slab-sized temporary per expression node, plus the output and
    one broadcast input. Beside it the admission reserves the term's tables
    ``8 sum E``, the centered coordinates ``8 d K``, their metadata ``H0``
    and the largest current table copy ``8 max E``, and b is the largest
    chunk that ``QHD.max_bytes`` then admits. Only one table and the
    extrema are kept at a time, so the reserved tables are conservative.
    """
    from .compiler import evaluate_support, user_point
    from .records import table_extrema
    from .validation import _lambdify_objective

    supports = decomposer.support_expressions
    chunks = _evaluation_chunks(term, grid, qhd, decomposer)
    extrema = []
    try:
        constant = coerce_real_scalar(decomposer.constant, context=f"constant term of {term.name}")
        for (support, expression), chunk in zip(supports.items(), chunks, strict=True):
            evaluator = _lambdify_objective([grid.variables[j] for j in support], expression)
            centered = [grid.coordinate_array(j) - decomposer.centers[j] for j in support]
            extrema.append(table_extrema(evaluate_support(
                evaluator, centered, chunk, expression=expression, support=support, name=term.name,
                locate=user_point(grid, decomposer.centers, support, expression))))
    except ValueError as error:
        raise ValueError(f"{term.name} = {term.expression} is not finite and real on the grid of the preprocessed "
                         f"box: {error}") from error
    return constant, tuple(extrema)


def _range_certificate(constant, tables):
    """Whether the stored support tables certify that ``c0 + sum_A T_A`` lies in the binary64 range on the grid.

    ``tables`` holds the ``(minimum, maximum, magnitude)`` of each support
    table (``_support_tables``), selected from its entries, so the exact
    sums below read the same binary64 values as the entries.

    The sum of absolute support magnitudes is a sufficient range
    certificate. When it exceeds binary64, the layer requires a further
    admitted range or original-expression check before deciding whether the
    input is numerically valid.

    The first certificate is ``B = |c0| + sum_A max |T_A|``: by the triangle
    inequality ``|c0 + sum_A T_A(x_A)| <= B`` at every grid point, and B also
    bounds every partial sum of c0 and one entry per table, in any order.
    The second, which keeps the signs, is the exact interval
    ``[c0 + sum_A min T_A, c0 + sum_A max T_A]``, which contains the sum at
    every grid point because every entry lies between its table's minimum
    and maximum. For ``1e308 (1 - x**2)`` on the endpoint grid of
    ``[-1, 1]``, c0 = 1e308 and the one table is ``(-1e308, -1e308)``, so B
    exceeds binary64 while the interval is ``[0, 0]``. Both sums are formed
    exactly from the stored binary64 values. When neither certifies the
    range, ``_setup`` completes the admitted scans of the supplied
    summands, sharing scans with the structural trigger. Their maxima enter
    ``_addition_certificate``. An inconclusive addition bound requests an
    admitted whole-term scan, and a failed scalar evaluation raises.
    """
    # abs, min and max of binary64 values are exact, so only their results become exact rationals.
    largest, c0 = Fraction(sys.float_info.max), Fraction(constant)
    magnitude = abs(c0) + sum((Fraction(mag) for _, _, mag in tables), Fraction(0))
    if magnitude <= largest:
        return True
    low = c0 + sum((Fraction(lo) for lo, _, _ in tables), Fraction(0))
    high = c0 + sum((Fraction(hi) for _, hi, _ in tables), Fraction(0))
    return -largest <= low and high <= largest


# Functions defined at every real argument, so an application of one restricts no domain (_restrictions).
_TOTAL = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh, sp.tanh, sp.atan, sp.asinh, sp.erf, sp.Abs, sp.sign, sp.Max,
          sp.Min, sp.floor, sp.ceiling)


def _restrictions(expression):
    """Return the subexpressions of ``expression`` whose value can be undefined at a real point, each once.

    These are the powers whose exponent is not a nonnegative integer,
    ``1/u`` and ``u**(1/2)`` for example, apart from a power of a positive
    number such as ``2**x``, the applications of functions outside
    ``_TOTAL``, such as ``log(u)``, and each outermost ``Piecewise``. Every
    other node of a SymPy expression tree, a sum, a product or a
    nonnegative integer power, is defined wherever its arguments are.

    Each outermost Piecewise is one structural unit, and its branch
    descendants are not inventoried separately. When the unit triggers a
    scan, its containing supplied summand is evaluated as a unit,
    including its branches and conditions before NumPy's ``select``,
    so an inactive branch or condition that raises still refuses. A kept
    occurrence covers an original restriction only when the kept occurrence
    contributes its value wherever the original restriction does. The
    structural coverage check therefore excludes occurrences below a
    Piecewise branch, because selection can discard a branch's value
    (``_contains_unconditional``).

    The role of these helpers is to identify when the support representation
    does not provide unconditional syntactic coverage of a possible
    restriction. An uncovered restriction requests original summand
    scans through ``_domain_check`` and an addition check. These helpers
    are not an exact test of domain equivalence.
    """
    found = {}

    def visit(node):
        if isinstance(node, sp.Piecewise):
            found.setdefault(node, None)
            return
        if isinstance(node, sp.Pow):
            restricted = not (node.exp.is_Integer and node.exp >= 0) and not (
                node.base.is_number and node.base.is_positive)
        else:
            restricted = node.is_Function and not isinstance(node, _TOTAL)
        if restricted:
            found.setdefault(node, None)
        for child in node.args:
            visit(child)

    visit(expression)
    return tuple(found)


def _contains_unconditional(expression, node):
    """Whether ``expression`` contains ``node`` outside every Piecewise branch, so its value enters the result."""
    if expression == node:
        return True
    if isinstance(expression, sp.Piecewise):
        return False
    return any(_contains_unconditional(child, node) for child in expression.args)


def _summands(expression):
    """Flatten Add nodes only, preserving every other supplied subtree."""
    if isinstance(expression, sp.Add):
        return tuple(q for child in sp.Add.make_args(expression) for q in _summands(child))
    return (expression,)


def _summand_records(term, variables, limit):
    expressions = _summands(term.expression)
    records = {}
    for expression in expressions:
        if expression not in records:
            support = tuple(j for j, variable in enumerate(variables) if variable in expression.free_symbols)
            records[expression] = _Term(f"{term.name} summand", expression, term.scale, None, support,
                                          node_count(expression, limit), None)
    return expressions, records


def _domain_check(term, variables, grid, qhd):
    """Scan one supplied expression and return its binary64 magnitude bound, or None.

    The scan evaluates the expression at the original coordinates of its
    support grid in C-order chunks (``compiler.chunk_coordinates``) with the
    existing finite-real rule, and streams the maximum absolute value; no
    table is kept. A failed call or a nonfinite or nonreal value raises
    ValueError naming the first failing grid point. None means that the
    values are finite and real but the raw result type does not support the
    binary64 addition certificate: raw float64 and complex128 outputs with
    zero imaginary part have the addition semantics used in the certificate,
    and a Python integer is composable only when exactly representable by
    the coerced float. Other raw types, including object arrays, and a
    callable that cannot be evaluated or converted on arrays are evaluated
    entry by entry as scalars with that same rule, and so is a chunk whose
    array evaluation signals a division by zero, an overflow or an invalid
    operation, where scalar Python arithmetic can raise, so an admitted
    scalar expression keeps its behavior. A nonfinite or nonreal array value
    that signals nothing is refused even when a scalar evaluation of that
    point succeeds: scalar re-evaluation may explain a failure, but it does
    not turn a nonfinite array result into an accepted finite entry. Array evaluations may differ from scalar
    evaluations by an ulp or more, and the scanned maximum is that of the
    selected evaluation. Empty support is one evaluation.

    The caller admits ``K**len(term.support) * (term.nodes + len(term.support))``
    units before scanning. The AL layer admits each original-expression
    scan's slab and evaluator workspace against the QHD byte allowance before
    the scan's batch runs (``_admit_layer_work``, ``_evaluation_chunks``): a
    chunk of b entries on s support variables is charged
    ``B_scan(b) = W_eval(b) + (8s + 64) b + H`` bytes
    (``compiler.support_chunk_size``) with the engineering allowance
    ``W_eval(b) = 8 b (N + 2)`` for N expression nodes
    (``compiler.support_workspace``), not a derived bound, whose premise is
    that the printer's evaluation holds at most one slab-sized temporary per
    expression node, plus the output and one broadcast input. The grid's
    coordinate vectors ``8 d K`` and their metadata H0 are held, and b is
    the largest chunk that ``QHD.max_bytes`` then admits.
    """
    import numpy as np

    from .compiler import chunk_coordinates
    from .validation import _lambdify_objective

    support = term.support
    entries = grid.num_grid_points ** len(support)
    (chunk,) = _evaluation_chunks(term, grid, qhd)
    axes = [grid.coordinate_array(j) for j in support]
    evaluator = _lambdify_objective([variables[j] for j in support], term.expression)
    maximum, composable = 0.0, True

    def scalar(point):
        nonlocal maximum, composable
        try:
            with np.errstate(all="ignore"):
                raw = evaluator(*point)
            value = coerce_real_scalar(raw, context=term.name)
            maximum = max(maximum, abs(value))
            # An exactly representable integer is unchanged by a later float
            # conversion. Other accepted scalar types use the whole-term scan.
            if type(raw) is int:
                composable = composable and raw == value
            else:
                array = np.asarray(raw)
                composable = composable and array.ndim == 0 and array.dtype in (
                    np.dtype("float64"), np.dtype("complex128"))
        except (ArithmeticError, ValueError, TypeError, NameError) as error:
            shown = {str(variables[j]): value for j, value in zip(support, point, strict=True)}
            raise ValueError(f"{term.name} = {term.expression} at {shown} is not a finite real "
                             f"value on the grid of the preprocessed box: {error}") from error

    for start in range(0, entries, chunk):
        stop = min(entries, start + chunk)
        arguments = chunk_coordinates(axes, start, stop)
        try:
            # A floating-point exception in the chunk, where scalar Python arithmetic can raise (a division
            # by zero, an overflow or an invalid operation), sends the chunk to the scalar rule.
            with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
                raw = evaluator(*arguments)
            values = np.asarray(raw)
            array = values.dtype in (np.dtype("float64"), np.dtype("complex128"))
            if array:
                values = np.broadcast_to(values, (stop - start,))
        except (ArithmeticError, ValueError, TypeError, NameError):
            array = False
        if not array:
            # Other raw types and signalling chunks keep the scalar rule, entry by entry.
            for position in range(stop - start):
                scalar(tuple(float(argument[position]) for argument in arguments))
            continue
        good = np.isfinite(values.real) & np.isfinite(values.imag) & (values.imag == 0)
        if not good.all():
            local = int(np.argmin(good))
            scalar(tuple(float(argument[local]) for argument in arguments))
            point = {str(variables[j]): float(argument[local]) for j, argument in zip(support, arguments, strict=True)}
            raise ValueError(f"{term.name} = {term.expression} at {point} is not a finite real value on the grid of "
                             f"the preprocessed box: {values[local]!r} is not finite and real")
        maximum = max(maximum, float(np.max(np.abs(values.real))))
    return maximum if composable else None


def _evaluation_chunks(term, grid, qhd, decomposer=None):
    """Return the chunk sizes of the array evaluations of ``term`` that the AL layer's grid check makes.

    With ``decomposer`` there is one chunk per support table of the term's
    expansion (``_support_tables``); without it, one chunk for the scan of
    ``term``'s supplied expression on its support (``_domain_check``). Each
    chunk of b entries on s support variables is charged
    ``B_scan(b) = W_eval(b) + (8s + 64) b + H`` bytes
    (``compiler.support_chunk_size``) with the engineering allowance
    ``W_eval(b) = 8 b (N + 2)`` for N expression nodes
    (``compiler.support_workspace``), beside the held bytes: for the tables
    the term's tables ``8 sum E``, the centered coordinates ``8 d K``, their
    metadata H0 and the largest table copy ``8 max E``, and for a scan the
    grid's coordinate vectors ``8 d K`` and H0. b is the largest chunk that
    ``QHD.max_bytes`` then admits.

    Raises:
        ValueError: one entry does not fit ``QHD.max_bytes``, naming the
            smallest admitting limit.
    """
    from .compiler import H0, support_chunk_size, support_workspace

    k = grid.num_grid_points
    if decomposer is None:
        support = term.support
        per_entry, fixed = support_workspace(term.nodes)
        held = 8 * grid.num_variables * k + H0
        smallest = held + fixed + H0 + per_entry + 8 * len(support) + 64
        if smallest > qhd.max_bytes:
            raise ValueError(f"scanning {term.name} on the grid requires at least {smallest} bytes for one entry, "
                             f"with QHD(max_bytes={qhd.max_bytes}); use QHD(max_bytes>={smallest})")
        return (support_chunk_size(k ** len(support), len(support), qhd.max_bytes - held, per_entry, fixed, H0),)
    supports = decomposer.support_expressions
    sizes = [k ** len(support) for support in supports]
    held = 8 * sum(sizes) + 8 * grid.num_variables * k + H0 + 8 * max(sizes, default=0)
    chunks = []
    for (support, expression), entries in zip(supports.items(), sizes, strict=True):
        per_entry, fixed = support_workspace(node_count(expression, qhd.max_work))
        smallest = held + fixed + H0 + per_entry + 8 * len(support) + 64
        if smallest > qhd.max_bytes:
            raise ValueError(f"checking {term.name} by its support tables requires at least {smallest} bytes for "
                             f"one entry of the table on support {support}, with QHD(max_bytes={qhd.max_bytes}); "
                             f"use QHD(max_bytes>={smallest})")
        chunks.append(support_chunk_size(entries, len(support), qhd.max_bytes - held, per_entry, fixed, H0))
    return tuple(chunks)


def _addition_certificate(maxima):
    """Certify any association/order of binary64 additions of the bounded leaves.

    For m finite real leaf values bounded by M_j, put n = max(m - 1, 0)
    and u = 2**-53. Their rounded additive tree has magnitude at most
    (1 + u)**n * sum(M_j). Since (1 + u)**n <= 1/(1 - n*u) when n*u < 1,
    sum(M_j) <= float_max * (1 - n*u) suffices to keep every addition finite.
    Form the bound exactly from the binary64 maxima. None marks a raw
    scalar type whose addition semantics have not been certified.
    """
    if any(value is None for value in maxima):
        return False
    n = max(0, len(maxima) - 1)
    u = Fraction(1, 2**53)
    if n * u >= 1:
        return False
    magnitude = sum((Fraction(value) for value in maxima), Fraction(0))
    return magnitude <= Fraction(sys.float_info.max) * (1 - n * u)


def _directed(value, upward):
    """Return the exact rational ``value`` rounded to binary64 upward or downward, or None outside its range."""
    try:
        number = float(value)
    except OverflowError:
        return None
    if upward and Fraction(number) < value:
        number = nextafter(number, inf)
    elif not upward and Fraction(number) > value:
        number = nextafter(number, -inf)
    return number if isfinite(number) else None


def _inequality_range(constant, tables, scale):
    """Return ``(ell, M)``, the normalized lower and upper bounds of one inequality on the grid.

    ``tables`` holds the ``(minimum, maximum, magnitude)`` of each support
    table (``_support_tables``).

    Proposition 54 of docs/mathematics.md, "A lower bound from support
    tables": with ``g = c0 + sum_A T_A(x_A)`` on the grid, every table entry
    lies between its table's minimum and maximum, so adding these
    inequalities gives

        ell = (c0 + sum_A min T_A)/s_g  <=  g(x)/s_g  <=  (c0 + sum_A max T_A)/s_g = M

    at every grid point. Grouping the terms of one variable into one table
    makes ell and M the exact grid extrema of an additively separable g. For
    overlapping supports ell can lie below the minimum: on ``{0, 1}**2``,
    ``(x - y)**2 = x**2 + y**2 - 2 x y`` has minimum 0, while its three table
    minima sum to -2. The sums and the division by s_g are formed exactly
    from the stored binary64 values, and ell is rounded downward and M
    upward, so both bound the tabulated representation of ``g/s_g``. The
    evaluation errors of the stored tables relative to the mathematical g
    are not bounded here (Proposition 54, ``ell^safe``), so the bounds
    describe the tabulated representation, on the grid of the preprocessed
    box only. A value outside the binary64 range is None.
    """
    s = Fraction(scale)
    low = (Fraction(constant) + sum((Fraction(lo) for lo, _, _ in tables), Fraction(0))) / s
    high = (Fraction(constant) + sum((Fraction(hi) for _, hi, _ in tables), Fraction(0))) / s
    return _directed(low, False), _directed(high, True)


def _admit_layer_work(qhd, options, check_work, round_work, *, lower_bound=False, grid=None, scans=()):
    """Raise before the layer's own work would exceed ``qhd.max_work``, or a requested evaluation ``qhd.max_bytes``.

    The check of _setup counts one unit per monomial of each term's
    expansion (objective.monomial_bound) and K**|A| (N_A + |A|) for each
    support table with N_A tree nodes (objective.node_count), as QHD
    planning counts its tables (QHD._admit_symbolic_work). Each
    original-summand scan costs K**|S_j| (N_j + |S_j|), using its supplied
    tree and support. A scan shared by structural coverage and the range
    check runs and is counted once. All scans in each requested batch are
    admitted before any of that batch runs. An inconclusive addition bound
    can request a further whole-term scan, charged K**|S| (N + |S|) before
    it starts. A one-summand term has already received that scan.

    One evaluation of a term with N tree nodes at a point of d
    coordinates counts ``N + d`` units, and C is the sum over f and the kept
    constraints. Without refinement each round evaluates every term once at
    its chosen point and once more at the valid-mass mean for
    ``mode_or_mean``, ``a C`` with ``a = 2`` for ``mode_or_mean`` and 1
    otherwise. With refinement over at most L levels each level checks the
    projection of its grid point and, for ``mode_or_mean``, of its mean,
    each of at most B attempted stall splits checks the projections of its
    two scored points (``_level_check``), and the round evaluates its point
    once more, ``(a L + 2 B + 1) C``, with
    ``B = min(max_splits, max(L - 1, 0))`` for
    ``stall_split="best_region"`` and B = 0 otherwise (``_setup`` derives
    it). For the stationarity diagnostic every nonzero partial
    derivative is evaluated once per round. The admitted total is the check plus
    ``max_iterations`` rounds, the layer's work over the whole run, charged
    as one operation as QHD charges its table and compilation work
    (``QHD._admit_symbolic_work``). QHD planning of each inner Plan and the
    trials of automatic representation selection (``_round_problem``) admit
    their own work separately. With ``lower_bound`` a count may have stopped
    early, so the reported work is a lower bound.

    ``scans`` holds ``(term, decomposer)`` pairs on ``grid``: the support
    tables of a term's expansion about the box centers with its decomposer,
    or with None the scan of a supplied expression. Each one's chunked array
    evaluation must admit at least one entry within ``qhd.max_bytes``
    (``_evaluation_chunks``). ``_setup`` passes the support tables of every
    term and the scans that the structure requests before any term is
    evaluated, and each later scan with the work admission of its batch.
    The tables of one term are held one term at a time, so each term is
    admitted against the whole allowance.
    """
    total = check_work + options.max_iterations * round_work
    if total > qhd.max_work:
        amount = f"at least {total}" if lower_bound else str(total)
        raise ValueError(f"the augmented-Lagrangian layer's own work needs {amount} work units, above "
                         f"qhd.max_work={qhd.max_work}: {check_work} to check f, h and g on the grid by their "
                         f"support tables and original-expression scans and {round_work} per round for "
                         f"{options.max_iterations} rounds")
    for term, decomposer in scans:
        _evaluation_chunks(term, grid, qhd, decomposer)


def _admitted_bounds(qhd, options, refinement, execution, kept):
    """Return ``(work, bytes)``, the host work and array bytes that the run can admit, fixed before its first round.

    QHD planning admits its work in categories, each at most W =
    ``qhd.max_work`` units and B = ``qhd.max_bytes`` bytes. An ordinary Plan
    has at most four category envelopes: the symbolic expansion, whose
    monomial count is admitted with 16 bytes per monomial
    (``QHD._admit_symbolic_work``), one running total of the initial
    state's evaluation and the table, compiled-block and schedule-integral
    work with its byte allowance (the same owner and ``QHD.plan``), the
    optional kept state with 16 bytes per amplitude (``QHD.plan``), and the
    classical evolution or the native construction
    (``QHD._host_construction``, ``QHD._select_native``). "Four" counts
    category envelopes, not calls of ``QHD._admit``: the recursive monomial
    checks and the checks of successive running totals call it many times,
    and their intermediate arguments are not separate work populations. The
    initial-state check that ``QHD.plan`` makes before evaluating the state
    admits ``d K`` units and ``_INITIAL_STATE_BYTES d K`` bytes, and the later
    running-total admission includes exactly these, so it is a preliminary
    check of the running-total category.

    Without refinement a round plans once, so a = 4 and L = 1. With
    refinement a round attempts at most L = ``refinement.max_levels``
    levels, and a level that stops or fails occupies one of them. A physical
    level plans once, a = 4. A search-model level first runs the table stage
    of its unscaled objective (``refinement._support_tables``), which calls
    ``QHD._admit_symbolic_work`` without compiled blocks and so adds the
    symbolic and running-total envelopes, the latter still with the ``d K``
    initial-state reservation and, under the integrated coefficient rule,
    the schedule integrals, and then plans the solved objective, a = 6. The
    layer's own work, the support-table check of ``_setup``, which also
    gives the inequality ranges, and the evaluations of every round, share
    one work admission for the whole run (``_admit_layer_work``), which has
    no byte part.

    Automatic selection (``inequality_form="auto"``) on the quantum route
    without a ``GaussianState`` plans, in each round, the current
    representation and at most one trial per kept inequality
    (``_round_problem``). Each of these at most ``m + 1`` candidates, with m
    kept inequalities, calls ``QHD._admit_symbolic_work`` once for its
    ledger, two envelopes, and plans once, four envelopes
    (``_planned_costs``). Without refinement the round's Plan is the kept
    candidate's Plan, or its refusal, so its four envelopes are among
    these, and the trials add ``t = 6 (m + 1) - 4`` envelopes per round.
    With refinement the levels plan separately and ``t = 6 (m + 1)``. The
    other policies make no trial, t = 0. With M = ``options.max_iterations``
    rounds,

        work <= (a M L + t M + 1) W,    bytes <= (a M L + t M) B,

    sums of the selected category allowances over attempts, not a peak
    memory. Nothing else in a level admits against W or B. Refinement
    geometry, the marginals, interval and next box, and the joint-mass scan
    of the observations or kept state, with its decoding of a native state,
    lie outside the bounds, as do ``refinement._tabulated_objective``'s
    repeated objective decomposition and its point evaluations, the
    symbolic construction of the inner objectives, the identity JSON of the
    records and library internals. Acquisition and synthesis have their own
    limits (``_outer.round_limits``).
    """
    from .initial_state import GaussianState

    levels = 1 if refinement is None else refinement.max_levels
    admissions = 6 if refinement is not None and refinement.scaling == "search_model" else 4
    operations = admissions * levels * options.max_iterations
    if (options.inequality_form == "auto" and execution == "quantum"
            and not isinstance(qhd.initial_state, GaussianState)):
        operations += (6 * (kept + 1) - (4 if refinement is None else 0)) * options.max_iterations
    return (operations + 1) * qhd.max_work, operations * qhd.max_bytes


def _derivatives(named, supports, variables, requested):
    """Return ``(reason, derivatives)`` for the stationarity diagnostic.

    ``derivatives`` holds, per term, the SymPy partial derivative for each
    variable in its support and None elsewhere. ``reason`` is None when the
    diagnostic applies, and otherwise says why not: it was not requested, a
    term contains a function from ``_NONSMOOTH``, or SymPy left a derivative
    unevaluated.
    """
    if not requested:
        return "not requested (AugmentedLagrangian.stationarity=False)", None
    kinds = sorted({type(atom).__name__ for _, expression, _ in named
                    for atom in expression.atoms(*_NONSMOOTH)})
    if kinds:
        return (f"f, h or g contains {', '.join(kinds)}, which is not differentiable everywhere, and the "
                "layer does not locate its kinks"), None
    derivatives = [[sp.diff(expression, variables[j]) if j in support else None
                    for j in range(len(variables))]
                   for (_, expression, _), support in zip(named, supports, strict=True)]
    for (name, _, _), row in zip(named, derivatives, strict=True):
        if any(e is not None and e.has(sp.Derivative, sp.Subs) for e in row):
            return f"{name} has a derivative that SymPy leaves unevaluated", None
    return None, derivatives


def _normalized(term):
    """The expression of ``term`` divided by its scale, unchanged for scale 1.

    SymPy writes the division as multiplication by the binary64 reciprocal
    and distributes it over a sum, so the tables hold ``term * fl(1/s)``
    while the layer computes ``term/s``. The two agree up to the rounding of
    the reciprocal and of the coefficient products, for which the layer
    states no bound. A comparison of a table value with a value that the
    layer evaluates, as ``mode_or_mean`` makes (``_outer.mean_point``),
    therefore may not give the exact order when the two nearly tie. For a
    scale other than 1 the product also rounds exact SymPy numbers of the
    expression to 53 bits before QHD planning expands L_k, so rescaling a
    term and its scale by a power of two can change the tables by rounding
    (``_update``).
    """
    return term.expression if term.scale == 1.0 else term.expression / sp.Float(term.scale)


def _effective_objective(setup, equality_multipliers, inequality_multipliers, penalty):
    """Return the normalized effective objective L_k of one round as a SymPy expression.

    With normalized ``f = f/s_f``, ``h_i = h_i/s_{h_i}`` and
    ``g_j = g_j/s_{g_j}``, multipliers lambda-bar and mu-bar and penalty rho,

        L_k(x) = f(x) + sum_i lambda-bar_i h_i(x) + (rho/2) sum_i h_i(x)**2
                 + sum_j P_j(g_j(x)),
        P_j(t) = (rho/2) (max(0, t + mu-bar_j/rho)**2 - (mu-bar_j/rho)**2)
               = ([mu-bar_j + rho t]_+**2 - mu-bar_j**2) / (2 rho).

    The equality part is Wu et al. arXiv:2605.12066v1, Eq. (6). The
    inequality part is the PHR term (Rockafellar, doi:10.1007/BF01580138). It
    is the minimum over a slack z >= 0 of ``mu (t + z) + (rho/2)(t + z)**2``,
    whose minimizer ``z* = max(0, -t - mu/rho)`` gives P(t) above, so the
    PHR form needs no slack register. A round adds a slack variable, as a
    register of the inner Plan, only for an inequality that its
    ``AugmentedLagrangian.inequality_form`` converts (``_round_problem``,
    ``_inner_objective``, Proposition 54 of docs/mathematics.md), and its
    outer update still uses this L_k at the projected point. The book's
    Problem 4.8 (p. 38) asks the reader to derive the multiplier update
    Eq. (4.8) from the equivalent squared slack ``g_j + z_j**2 = 0``. The
    derivative ``dP/dt = [mu + rho t]_+`` is continuous, and P is C1 but not
    C2 where ``mu + rho t = 0``.

    Term by term, L_k is the book's Eq. (10.3) (Sec. 10.1, p. 114). The
    book's Eq. (4.3) completes the squares instead, with
    ``(rho/2) (h_i + lambda-bar_i/rho)**2`` and
    ``(rho/2) max(0, g_j + mu-bar_j/rho)**2``, which exceed the terms above
    by the constants ``lambda-bar_i**2/(2 rho)`` and ``mu-bar_j**2/(2 rho)``.
    A constant moves neither the minimizer nor the multiplier update and
    only adds a global phase to the QHD evolution. Without these constants,
    ``L_k(x) = f(x)`` at a feasible point x where every
    ``mu-bar_j g_j(x) = 0``, complementarity with the multipliers of the
    round, so recorded L values compare with f directly.

    Every number enters as a ``sp.Float`` equal to the binary64 value the
    layer uses: lambda-bar_i, ``rho/2`` and ``mu-bar_j/rho``. SymPy forms
    ``(mu-bar_j/rho)**2`` at 53-bit precision with round-to-nearest and an
    unbounded exponent range. This agrees with the binary64 square when
    the rounded result is finite and normal. Outside that range the SymPy
    square can remain nonzero below the binary64 range or finite above it.
    QHD planning expands this expression and tabulates it by variable
    support like any objective. ``_effective_value`` evaluates the same
    mathematical formula using binary64 arithmetic.
    """
    return _inner_objective(setup, ("phr",) * len(setup.inequalities), equality_multipliers,
                            inequality_multipliers, penalty)


def _inner_objective(setup, forms, equality_multipliers, inequality_multipliers, penalty):
    """Return the inner objective L(x, s) of a round whose kept inequalities take ``forms``.

    Proposition 54 of docs/mathematics.md. With the notation of
    ``_effective_objective``, ``G_j = g_j/s_{g_j}`` and the slack variable
    s_j of ``_Setup.slack_symbols``, inequality j contributes

    - ``phr``: the PHR term ``(rho/2) (max(0, G_j + mu_j/rho)**2 - (mu_j/rho)**2)``,
    - ``slack``: ``phi_j = mu_j (G_j + s_j) + (rho/2) (G_j + s_j)**2``,
    - ``quadratic``: ``mu_j G_j + (rho/2) G_j**2``, and
    - ``constant``: ``-(rho/2) (mu_j/rho)**2 = -mu_j**2/(2 rho)``.

    Completing the square, ``phi_j = (rho/2) (s + G_j + mu_j/rho)**2 -
    mu_j**2/(2 rho)``, so its minimum over ``s >= 0`` is the PHR term, at
    ``s* = [-G_j - mu_j/rho]_+``, and ``phi_j`` exceeds that minimum at every
    other ``s >= 0``. The quadratic equals the PHR term where
    ``G_j + mu_j/rho >= 0``, which ``ell_j + mu_j/rho >= 0`` gives on the
    grid, and the constant equals it where ``G_j + mu_j/rho <= 0``, which
    ``M_j + mu_j/rho <= 0`` gives (``_round_problem``). The equality part and
    the objective are those of L_k, and each number enters as a ``sp.Float``
    of its binary64 value, as there. With every form ``phr`` the expression
    is ``_effective_objective``'s.
    """
    half = sp.Float(penalty / 2)
    terms = [_normalized(setup.objective)]
    for multiplier, term in zip(equality_multipliers, setup.equalities, strict=True):
        h = _normalized(term)
        terms.append(sp.Float(multiplier) * h + half * h**2)
    for form, multiplier, term, symbol in zip(forms, inequality_multipliers, setup.inequalities,
                                              setup.slack_symbols, strict=True):
        g = _normalized(term)
        shift = sp.Float(multiplier / penalty)
        if form == "phr":
            terms.append(half * (sp.Max(0, g + shift) ** 2 - shift**2))
        elif form == "slack":
            terms.append(sp.Float(multiplier) * (g + symbol) + half * (g + symbol) ** 2)
        elif form == "quadratic":
            terms.append(sp.Float(multiplier) * g + half * g**2)
        else:
            terms.append(-half * shift**2)
    return sp.Add(*terms)


def _effective_value(equality_multipliers, inequality_multipliers, penalty, f, h, g):
    """Return L_k from normalized values f, h and g at one point, with ``_effective_objective``'s formula.

    The layer uses it at the off-grid valid-mass mean of ``mode_or_mean``,
    where no table value exists, and at the projected point of a round with
    an ``InnerRepresentation`` (``ALEvaluation.effective_value``). The terms
    are summed with ``fsum``.
    """
    return _inner_value(("phr",) * len(g), (), equality_multipliers, inequality_multipliers, penalty, f, h, g)


def _inner_value(forms, slacks, equality_multipliers, inequality_multipliers, penalty, f, h, g):
    """Return L(x, s) of ``_inner_objective`` from normalized f, h and g at x and the slack values ``slacks``.

    The same formula in binary64 arithmetic, with one slack value per
    ``slack`` form in order, the terms summed with ``fsum``. The layer uses
    it at a ``mode_or_mean`` mean of a round with a representation, where no
    table value exists (``_choose``).
    """
    half = penalty / 2
    terms = [f]
    terms += [multiplier * value + half * value * value
              for multiplier, value in zip(equality_multipliers, h, strict=True)]
    values = iter(slacks)
    for form, multiplier, value in zip(forms, inequality_multipliers, g, strict=True):
        shift = multiplier / penalty
        if form == "phr":
            terms.append(half * (max(0.0, value + shift) ** 2 - shift * shift))
        elif form == "slack":
            total = value + next(values)
            terms.append(multiplier * total + half * total * total)
        elif form == "quadratic":
            terms.append(multiplier * value + half * value * value)
        else:
            terms.append(-half * shift * shift)
    return fsum(terms)


def _slack_axis(setup, qhd, t, multiplier, penalty):
    """Return ``(SlackAxis, None)`` for kept inequality t, or ``(None, reason)`` when it has no admissible slack box.

    Proposition 54 of docs/mathematics.md. The cap is
    ``U_0 = [-ell_t]_+`` with the recorded lower bound ell_t of the
    normalized inequality on the grid (``_inequality_range``). Since
    ``s*(x) = [-G(x) - mu/rho]_+ <= [-ell_t - mu/rho]_+ <= [-ell_t]_+``, the box
    ``[0, U_0]`` contains the minimizer at every grid point for every
    nonnegative multiplier and positive penalty. A zero cap adds no axis, and
    the caller keeps the inequality's other form. The slack grid has the
    Method's K points and grid convention on ``[0, U]``, the grid that
    ``grid.OneHotGrid`` admits, with ``U = U_0`` on a Dirichlet grid. A
    periodic grid ``0, h, ..., U - h`` omits U, and the margin
    ``U >= K/(K-1) U_0`` gives ``U - h = U (1 - 1/K) >= U_0``, so zero and
    every required slack in ``[0, U_0]`` lie within ``h/2`` of a node.

    The error bound of the slack grid, for the round's multiplier mu and
    penalty rho, with ``r = [mu + rho M_t]_+`` and the recorded upper bound
    M_t, is the bound of Proposition 54's table:

    - Dirichlet interior, nodes ``h, ..., K h`` with ``h = U/(K+1)``:
      ``e <= r h + rho h**2/2``, where ``r >= [mu + rho G(x)]_+`` at every
      grid point x;
    - Dirichlet with boundary points, ``h = U/(K-1)``: ``e <= rho h**2/8``;
    - periodic with the margin, ``h = U/K``: ``e <= rho h**2/8``, since zero
      is a node and ``[0, U_0]`` has covering radius ``h/2``.

    The bound is formed exactly from the binary64 h, mu, rho and M_t and
    rounded upward once. It assumes the exact mesh of spacing h, the
    premises of the cap and the tabulated representation of G on the grid of
    the preprocessed box. Interpreting the formula as a slack-grid excess
    bound requires the cap and table-arithmetic premises and the node
    coverage used in Proposition 54. The interior formula requires a first
    node no larger than h and distance at most h from every required slack
    to the grid. The endpoint and covered periodic formulas require zero and
    distance at most h/2. Rounding the spacing and coordinates can affect
    these conditions. This bound supplies no separate allowance for that
    effect.
    """
    pre = setup.preprocessing
    low, high = pre.inequality_lower[t], pre.inequality_upper[t]
    if low is None or high is None:
        return None, "its grid range lies outside the binary64 range"
    if low >= 0.0:
        return None, "its cap [-ell]_+ is zero"
    cap, k = -low, qhd.num_grid_points
    upper = cap if qhd.boundary != "periodic" else _directed(Fraction(k, k - 1) * Fraction(cap), True)
    if upper is None:
        return None, "its periodic slack margin exceeds binary64"
    name = setup.slack_symbols[t].name
    try:
        grid = OneHotGrid((name,), ((0.0, upper),), k, qhd.include_boundary_points, qhd.boundary)
    except ValueError as error:
        return None, f"its slack grid on [0, {upper!r}] is not admissible: {error}"
    h, rho = Fraction(grid.spacing(0)), Fraction(penalty)
    if grid_convention(qhd) == "dirichlet_interior":
        r = max(Fraction(0), Fraction(multiplier) + rho * Fraction(high))
        bound = r * h + rho * h * h / 2
    else:
        bound = rho * h * h / 8
    error_bound = _directed(bound, True)
    if error_bound is None:
        return None, "its slack-grid error bound exceeds binary64"
    return SlackAxis(inequality=pre.inequalities[t], variable=name, cap=cap, upper=upper,
                     spacing=grid.spacing(0), error_bound=error_bound), None


def _branch(setup, t, multiplier, penalty):
    """Return ``quadratic``, ``constant`` or None: the exact form of kept inequality t's PHR term on the grid.

    Proposition 54 of docs/mathematics.md, "A lower bound from support
    tables": with the recorded bounds ``ell_t <= G <= M_t`` on the grid
    (``_inequality_range``), ``ell_t + mu/rho >= 0`` gives ``G + mu/rho >= 0``
    at every grid point, where the PHR term is ``mu G + (rho/2) G**2``, and
    ``M_t + mu/rho <= 0`` gives ``G + mu/rho <= 0``, where it is the constant
    ``-mu**2/(2 rho)``. The first test is applied first. Both comparisons are
    exact in rational arithmetic on the binary64 values. They are sufficient
    conditions on the grid of the preprocessed box only.
    """
    pre = setup.preprocessing
    shift = Fraction(multiplier) / Fraction(penalty)
    low, high = pre.inequality_lower[t], pre.inequality_upper[t]
    if low is not None and Fraction(low) + shift >= 0:
        return "quadratic"
    if high is not None and Fraction(high) + shift <= 0:
        return "constant"
    return None


def _inner_problem(setup, problem, forms, slacks, lambda_bar, mu_bar, penalty):
    """Return ``(Optimization, identity)``: a round's inner problem and the identity of its objective.

    The inner variables are the problem's variables followed by the slack
    variables of the ``slack`` forms, in constraint order, and the inner box
    is the preprocessed box followed by each slack box ``[0, U]``
    (``SlackAxis.upper``). The identity is ``records.objective_reference`` of
    the objective's ``srepr``, the inner variables' ``srepr`` and the inner
    box, the identity of ``ALIteration.effective_objective`` for PHR forms.
    """
    from nwqlib.problems.records import Optimization
    from .records import objective_reference

    objective = _inner_objective(setup, forms, lambda_bar, mu_bar, penalty)
    symbols = tuple(symbol for symbol, form in zip(setup.slack_symbols, forms, strict=True) if form == "slack")
    variables = tuple(problem.variables) + symbols
    bounds = tuple(setup.preprocessing.bounds) + tuple((0.0, axis.upper) for axis in slacks)
    identity = objective_reference(sp.srepr(objective), tuple(sp.srepr(v) for v in variables), bounds).identity
    return Optimization(objective=objective, variables=variables, bounds=bounds, unit=problem.unit), identity


def _planned_costs(problem, qhd, shots, child):
    """Return ``(work, cx, planned, evaluated)``: the costs of a quantum inner Plan and its evaluated tables.

    The host work is the admitted work of QHD's symbolic and table stage
    (``QHD._admit_symbolic_work``: the monomial bound, the initial state,
    the step rows, the support tables and the compilation) plus the selected
    native construction's work (``SelectedDefinition.construction_work``),
    and the CX count is the selected construction's CX law
    (``_outer.law_count``), None without one. The table stage is admitted
    before any table is evaluated, so a refused inequality table, such as a
    PHR term on many variables, ranks the candidate as refused without
    being formed. ``planned`` is the Plan or the exception that refused or
    ended its planning, and ``work`` is None after one. The round treats that
    exception as the failure of its own planning (``_outer.inner_failure``),
    as it treats any exception of ``_outer.plan_round``. The Plan is planned
    from a copy of the round's random-stream child, so it is the Plan that
    ``_outer.plan_round`` gives the round.
    The Plan's own planning reuses that admission, its decomposer and
    chunks through a private context (``method._ADMITTED_SYMBOLIC``), so
    each candidate's objective is expanded and admitted once.

    ``evaluated`` counts the objective-table grid tuples that the attempt
    evaluated: the Plan's ``support_evaluations``, zero when the table
    admission refused, since that admission precedes every table
    evaluation, and None when the planning raised after it, since the
    tables that it evaluated before raising are unknown.
    """
    import numpy as np

    streams = np.random.SeedSequence(child.entropy, spawn_key=child.spawn_key, pool_size=child.pool_size,
                                     n_children_spawned=child.n_children_spawned)
    from .method import _ADMITTED_SYMBOLIC

    try:
        admitted = qhd._admit_symbolic_work(problem, True)
    except Exception as error:
        return None, None, error, 0
    symbolic = admitted[2]
    # The candidate's planning reuses this admission and its decomposer (QHD._admit_symbolic_work).
    token = _ADMITTED_SYMBOLIC.set(((qhd.content_id, problem.content_id, True), admitted))
    try:
        plan = plan_round(problem, qhd, "quantum", shots, streams)
    except Exception as error:
        return None, None, error, None
    finally:
        _ADMITTED_SYMBOLIC.reset(token)
    work = symbolic + sum(selection.construction_work for selection in plan.construction.selections)
    return work, law_count(plan, "cx", 1)[0], plan, plan.reconstruction.support_evaluations


def _accepts(current, trial):
    """Return ``(accepted, reason)``: whether automatic selection keeps a trial conversion (``FormTrial``).

    The rule of the quantum route: when both candidates are admitted, the
    trial is kept when its planned host work and its CX count are both no
    larger than the current representation's and one is strictly smaller. A
    tie, or one larger and one smaller, keeps the current representation. A
    trial whose planning is admitted and whose construction has a CX law
    replaces a current one whose planning was refused, which enables an
    otherwise refused inner Plan. A missing CX law of either admitted
    candidate supplies no value for the CX comparison and keeps the current
    representation, also when the current planning was refused.
    """
    (work, cx), (new_work, new_cx) = current, trial
    if new_work is None:
        return False, "the trial's planning was refused"
    if new_cx is None:
        return False, "a CX law is unavailable, so the costs do not compare"
    if work is None:
        return True, "the current planning was refused and the trial's is admitted"
    if cx is None:
        return False, "a CX law is unavailable, so the costs do not compare"
    if new_work <= work and new_cx <= cx and (new_work < work or new_cx < cx):
        return True, "no larger planned host work and CX count, one strictly smaller"
    return False, "the planned host work and CX count are not both no larger with one strictly smaller"


def _round_problem(setup, problem, qhd, execution, shots, child, lambda_bar, mu_bar, penalty, refined, saved=None):
    """Return ``(representation, inner problem, planned)`` of one round under ``options.inequality_form``.

    ``phr`` returns no representation and the problem of L_k
    (``_effective_objective``). Otherwise, Proposition 54 of
    docs/mathematics.md, by this selection rule:

    1. When every point that the round compares lies on the grid of the
       preprocessed box, that is without refinement and with a grid-point
       rule, the branch tests (``_branch``) replace a kept inequality's PHR
       term by its exact quadratic or constant form on that grid. A refined
       level's grid or a ``mode_or_mean`` mean lies outside the tests'
       domain, so such a round keeps the PHR term there.
    2. ``slack`` gives every remaining kept inequality a slack variable
       (``_slack_axis``). A zero cap adds none, and the inequality keeps its
       form, and a cap without an admissible slack grid raises ValueError.
    3. ``auto`` converts on the quantum route only. Under
       ``execution="classical"`` it keeps the PHR form of every inequality,
       since each slack multiplies the restricted state by K, and so it does
       with a ``GaussianState``, which defines no slack axis, both without
       the branch tests. Otherwise it plans the current representation and then,
       in constraint order, a trial with one more slack
       (``_planned_costs``), keeps each trial that ``_accepts`` and compares
       the next trial with the representation kept so far, one pass without
       a search over subsets. A refused PHR table ranks the current
       representation as refused before any of its tables is formed.

    The representation is fixed for the whole round, its refinement levels
    included, and the caller commits it before the first inner Run. A
    resumed round passes its committed representation as ``saved``, which
    is reused after checking that it rebuilds the saved inner objective.

    Returns:
        ``(representation, problem, planned)``: the ``InnerRepresentation``
        or None, the inner Optimization, and the Plan of the kept
        representation or the ValueError that refused its planning when
        automatic selection planned it without refinement, else None, so
        the round plans it once.
    """
    from nwqlib.problems.records import Optimization
    from .initial_state import GaussianState

    options = setup.options
    policy = options.inequality_form
    if policy == "phr":
        effective = _effective_objective(setup, lambda_bar, mu_bar, penalty)
        return None, Optimization(objective=effective, variables=problem.variables, bounds=setup.preprocessing.bounds,
                                  unit=problem.unit), None
    if saved is not None:
        inner, identity = _inner_problem(setup, problem, saved.forms, saved.slacks, lambda_bar, mu_bar, penalty)
        if identity != saved.inner_objective or saved.policy != policy:
            raise ValueError("the committed representation of the round in progress does not rebuild its inner "
                             "objective from the run's problem, multipliers and penalty")
        return saved, inner, None
    m = len(setup.inequalities)
    forms, axes, trials = ["phr"] * m, {}, []
    tables, unavailable = 0, None
    classical = execution != "quantum"
    gaussian = isinstance(qhd.initial_state, GaussianState)
    keep = policy == "auto" and (classical or gaussian)
    if not (refined or options.inner_point == "mode_or_mean" or keep):
        for t in range(m):
            forms[t] = _branch(setup, t, mu_bar[t], penalty) or "phr"
    planned = None
    if keep and classical:
        selection = ("execution='classical' keeps the PHR form of every inequality, since each slack variable "
                     "multiplies the restricted state by K")
    elif keep:
        selection = ("the GaussianState initial state defines no slack axis, so automatic selection keeps the PHR "
                     "form of every inequality")
    elif policy == "slack":
        for t in range(m):
            if forms[t] != "phr" or setup.preprocessing.inequality_lower[t] is not None and (
                    setup.preprocessing.inequality_lower[t] >= 0.0):
                continue
            axis, reason = _slack_axis(setup, qhd, t, mu_bar[t], penalty)
            if axis is None:
                raise ValueError(f"inequality_form='slack' cannot give {setup.inequalities[t].name} a slack "
                                 f"variable: {reason}")
            forms[t], axes[t] = "slack", axis
        selection = "every kept inequality without an exact branch form and with a positive cap has a slack variable"
    else:
        inner = _inner_problem(setup, problem, forms, (), lambda_bar, mu_bar, penalty)[0]
        work, cx, planned, evaluated = _planned_costs(inner, qhd, shots, child)
        counts = [evaluated]
        for t in range(m):
            if forms[t] != "phr":
                continue
            axis, reason = _slack_axis(setup, qhd, t, mu_bar[t], penalty)
            if axis is None:
                trials.append(FormTrial(inequality=setup.preprocessing.inequalities[t], current_work=work,
                                        current_cx=cx, trial_work=None, trial_cx=None, accepted=False,
                                        reason=f"no slack variable: {reason}"))
                continue
            trial_forms = forms[:t] + ["slack"] + forms[t + 1:]
            extended = {**axes, t: axis}
            trial_axes = tuple(extended[s] for s in sorted(extended))
            candidate = _inner_problem(setup, problem, trial_forms, trial_axes, lambda_bar, mu_bar, penalty)[0]
            new_work, new_cx, new_planned, evaluated = _planned_costs(candidate, qhd, shots, child)
            counts.append(evaluated)
            accepted, reason = _accepts((work, cx), (new_work, new_cx))
            trials.append(FormTrial(inequality=setup.preprocessing.inequalities[t], current_work=work,
                                    current_cx=cx, trial_work=new_work, trial_cx=new_cx, accepted=accepted,
                                    reason=reason))
            if accepted:
                forms[t], axes[t] = "slack", axis
                work, cx, planned = new_work, new_cx, new_planned
        selection = "the quantum-route comparison of planned host work and CX count"
        # Every candidate planned once, the current representation first (ALResources.table_evaluations).
        tables = None if None in counts else sum(counts)
        unavailable = None if tables is not None else (
            "the planning of an automatic-selection candidate raised after its table admission, so the tables "
            "that it evaluated are unknown")
    slacks = tuple(axes[t] for t in sorted(axes))
    inner, identity = _inner_problem(setup, problem, forms, slacks, lambda_bar, mu_bar, penalty)
    error = _directed(sum((Fraction(axis.error_bound) for axis in slacks), Fraction(0)), True)
    if error is None:
        raise ValueError("the sum of the slack-grid error bounds exceeds binary64")
    representation = InnerRepresentation(
        policy=policy, forms=tuple(forms), slacks=slacks, variables=tuple(str(v) for v in inner.variables),
        grid=grid_convention(qhd), grid_points=qhd.num_grid_points, error_bound=error, trials=tuple(trials),
        selection=selection, table_evaluations=tables, table_evaluations_unavailable=unavailable,
        inner_objective=identity)
    return representation, inner, None if refined else planned


def _safeguard(bounds, lambda_plus, mu_plus):
    """Return the multipliers of the next round, the book's Algorithm 4.1 Step 4.

    Without bounds the next round uses the tentative multipliers. With
    ``MultiplierBounds`` it uses ``clip(lambda+, lambda_min, lambda_max)`` and
    ``min(mu+, mu_max)``. mu+ is already nonnegative, so the lower bound 0 of
    the book's ``[0, mu_max]`` needs no clip. Clipping rounds nothing.

    Step 4 only asks for multipliers inside the bounds. Clipping keeps
    lambda+ and mu+ unchanged whenever they lie inside, the choice that the
    book's Assumption 7.6 (Sec. 7.5, p. 64) prescribes. The book bounds the
    multipliers so that the shifts lambda-bar/rho and mu-bar/rho tend to zero
    when rho grows (p. 35), and ``MultiplierBounds`` states what the book's
    convergence results assume of the bounds.
    """
    if bounds is None:
        return lambda_plus, mu_plus
    return (tuple(min(max(value, bounds.equality_lower), bounds.equality_upper) for value in lambda_plus),
            tuple(min(value, bounds.inequality_upper) for value in mu_plus))


def _next_penalty(options, penalty, measure, previous):
    """Return rho_{k+1} from rho_k, the measure m_{k+1} of this round and m_k of the previous one.

    ``on_insufficient_decrease`` is the book's Algorithm 4.1 Step 3 with
    Eq. (4.9): after the first round (``previous`` None), or when
    ``m_{k+1} <= tau m_k``, rho stays, the choice ``rho_{k+1} = rho_k`` among
    the book's admissible ``rho_{k+1} >= rho_k``. Otherwise
    ``rho_{k+1} = min(gamma rho_k, rho_max)``. The book's Theorem 5.2
    (p. 42) and its boundedness theorem for the penalty, Theorem 7.2 with
    Assumption 7.9 (Sec. 7.7, p. 70), assume that rho stays when the test
    holds. ``every_iteration`` multiplies
    by gamma after every round, the rule of Wu et al.'s code. rho_max is Wu
    et al.'s cap (arXiv:2605.12066v1, Sec. VI.B), and the book has none.
    """
    grown = min(options.penalty_growth * penalty, options.max_penalty)
    if options.penalty_update == "every_iteration":
        return grown
    if previous is None or measure <= options.reduction_ratio * previous:
        return penalty
    return grown


def _infeasibility(h, g):
    """Return ``max(||h||_inf, ||g_+||_inf)`` of normalized residuals, the left side of Algencan's Eq. (10.8).

    Zero without constraints. The rounds (``_update``) and the explicit grid
    reference (``constrained_grid_minimum``) apply the same test.
    """
    return max(max(map(abs, h), default=0.0), max((max(value, 0.0) for value in g), default=0.0))


def _update(options, penalty, lambda_bar, mu_bar, h_raw, g_raw, h_scales, g_scales, previous):
    """Return the normalized residual tests, the multiplier update and the next penalty of one round.

    With normalized residuals ``h_i = h_raw_i/s_{h_i}`` and
    ``g_j = g_raw_j/s_{g_j}`` at the chosen point x_{k+1}:

    - infeasibility ``max(||h||_inf, ||g_+||_inf)`` (``_infeasibility``), the
      left side of Algencan's Eq. (10.8).
    - ``V_j = min(-g_j, mu-bar_j/rho)`` and the penalty measure
      ``m_{k+1} = max(||h||_inf, ||V||_inf)``, Eq. (4.9) (book pp. 33-34). By
      Eq. (4.8), ``mu+_j - mu-bar_j = -rho V_j``, so V tracks both the
      multiplier change and complementarity.
    - tentative multipliers ``lambda+ = lambda-bar + rho h`` (Eq. (4.7), Wu et
      al. arXiv:2605.12066v1 Eq. (8)) and ``mu+ = max(0, mu-bar + rho g)``
      (Eq. (4.8)), which make ``grad_x L_k = grad f + J_h^T lambda+ +
      J_g^T mu+`` (``_stationarity``).
    - complementarity ``max(||h||_inf, max_j |min(-g_j, mu+_j)|)``, the left
      side of Eq. (10.7). Its inequality part is the alternative V of the
      book's Problem 4.9 (p. 39), with mu+_j in place of mu-bar_j/rho. The
      layer uses it in this stopping test only, and the penalty test keeps
      the V of Eq. (4.9).

    Algencan applies Eq. (10.8) to the unscaled constraints. Here both tests
    act on normalized residuals, and original-unit residuals are reported
    next to them. Writing a constraint as ``c g <= 0`` with scale ``c s_g``,
    c > 0, leaves every normalized quantity, and so every decision,
    unchanged in exact arithmetic, since ``c g/(c s_g) = g/s_g``. The layer
    computes in binary64, where the tables and residuals of the two runs
    agree only up to rounding, even when c is a power of two, and rounding
    can change a decision. The inner Plans tabulate L_k from the symbolic
    ``_normalized``, which divides ``c g`` by ``sp.Float(c s_g)`` before QHD
    planning expands it, so SymPy rounds the coefficients to 53 bits and
    rounds their products in the expansion, while ``_normalized`` returns a
    term of scale 1 unchanged, with exact SymPy numbers such as ``1/5``. The
    tables of L_k can therefore differ in their last bits, and a round's
    point can differ where table values nearly tie. For the constraint
    ``g <= 0`` with ``g = x**2 + x**4/5`` and the objective ``-g**2/2`` on
    ``[-1, 1]``, with K = 3, one round, penalty 1 and zero multipliers, L_k
    is exactly zero, and c = 2 leaves a table residue of 1.36e-20, so
    ``best_observed`` reads another point than c = 1 does. For other c the
    normalized residuals also differ by the rounding of the evaluated
    ``c g`` and of the scale division. So a comparison near a tie or
    exactly at its threshold, of
    the penalty measure or of a stopping test, can go either way. The next
    multipliers and penalty come from ``_safeguard`` and ``_next_penalty``.
    The stopping test that compares these quantities with the tolerances is
    in ``solve_augmented_lagrangian``.
    """
    h = tuple(value / scale for value, scale in zip(h_raw, h_scales, strict=True))
    g = tuple(value / scale for value, scale in zip(g_raw, g_scales, strict=True))
    h_norm = max(map(abs, h), default=0.0)
    infeasibility = _infeasibility(h, g)
    shifts = tuple(min(-value, multiplier / penalty) for value, multiplier in zip(g, mu_bar, strict=True))
    measure = max(h_norm, max(map(abs, shifts), default=0.0))
    lambda_plus = tuple(multiplier + penalty * value for multiplier, value in zip(lambda_bar, h, strict=True))
    mu_plus = tuple(max(0.0, multiplier + penalty * value) for multiplier, value in zip(mu_bar, g, strict=True))
    complementarity = max(h_norm, max((abs(min(-value, multiplier)) for value, multiplier in zip(g, mu_plus)),
                                      default=0.0))
    lambda_next, mu_next = _safeguard(options.multiplier_bounds, lambda_plus, mu_plus)
    return dict(
        h=h, g=g, infeasibility=infeasibility, complementarity=complementarity, measure=measure,
        lambda_plus=lambda_plus, mu_plus=mu_plus, lambda_next=lambda_next, mu_next=mu_next,
        lambda_truncated=tuple(a != b for a, b in zip(lambda_plus, lambda_next, strict=True)),
        mu_truncated=tuple(a != b for a, b in zip(mu_plus, mu_next, strict=True)),
        next_penalty=_next_penalty(options, penalty, measure, previous),
    )


def _stationarity(setup, point, lambda_plus, mu_plus):
    """Return ``(r_stat, reason, evaluations, work)`` of the projected-gradient diagnostic at ``point``.

    Map the preprocessed box ``[a, b]`` to ``u in [0, 1]^d`` by
    ``x = a + D u``, ``D = diag(b - a)``. Let l(u) be the normalized
    augmented Lagrangian with this round's multipliers. By the chain rule and
    ``dP/dt = [mu + rho t]_+`` (``_effective_objective``),

        grad_u l = D grad_x L,  grad_x L = grad f + J_h^T lambda+ + J_g^T mu+,

    with the tentative multipliers lambda+ and mu+ of ``_update`` and
    normalized f, h and g. The diagnostic is

        r_stat = || u - P_[0,1]^d (u - grad_u l) ||_inf,

    the book's Eq. (10.6) in box-normalized coordinates, so every component
    is dimensionless whatever the box widths. Safeguarded multipliers do not
    satisfy the gradient identity above and are not used. A finite-grid
    minimizer of L_k generally has ``r_stat > 0``, so r_stat is reported, never
    used to stop, and a small value only indicates approximate stationarity.
    PHR is C1, so the first derivative exists where ``mu + rho g = 0``. The
    diagnostic is unavailable for a problem with a nondifferentiable function
    and at a point where a partial derivative is not finite.
    """
    if setup.stationarity_unavailable is not None:
        return None, setup.stationarity_unavailable, 0, 0
    d = setup.grid.num_variables
    weighted = ((setup.objective, 1.0),
                *zip(setup.equalities, lambda_plus, strict=True),
                *zip(setup.inequalities, mu_plus, strict=True))
    components = [[] for _ in range(d)]
    evaluations = work = 0
    for term, weight in weighted:
        for j, item in enumerate(term.gradient):
            if item is None:
                continue
            function, nodes = item
            # An attempt costs nodes + d units (_admit_layer_work) whether or
            # not it succeeds, so it is counted before the call.
            evaluations, work = evaluations + 1, work + nodes + d
            try:
                value = coerce_real_scalar(function(*point), context=f"derivative of {term.name}")
            except (ArithmeticError, ValueError, TypeError, NameError):
                reason = f"a partial derivative of {term.name} is not finite at this point"
                return None, reason, evaluations, work
            # d(t/s)/dx = (dt/dx)/s, weighted by 1 for f and by lambda+ or mu+.
            components[j].append(weight * (value / term.scale))
    residual = 0.0
    for j, ((lower, upper), x) in enumerate(zip(setup.preprocessing.bounds, point, strict=True)):
        width = upper - lower
        u = (x - lower) / width
        try:
            gradient = width * fsum(components[j])
        except (OverflowError, ValueError):
            gradient = float("inf")
        if not isfinite(gradient):
            reason = "the gradient of the augmented Lagrangian is not finite at this point"
            return None, reason, evaluations, work
        residual = max(residual, abs(u - min(max(u - gradient, 0.0), 1.0)))
    return residual, None, evaluations, work


def _point_cost(setup):
    """Return ``(evaluations, work)`` of evaluating f, h and g at one point, as ``_admit_layer_work`` counts it."""
    terms = (setup.objective, *setup.equalities, *setup.inequalities)
    return len(terms), sum(term.nodes + setup.grid.num_variables for term in terms)


def _evaluate(setup, point):
    """Return ``(f, h, g, evaluations, work)`` in original units at ``point``, each value finite and real."""
    terms = (setup.objective, *setup.equalities, *setup.inequalities)
    values = [_value(term, point) for term in terms]
    count = len(setup.equalities)
    return (values[0], tuple(values[1:1 + count]), tuple(values[1 + count:]), *_point_cost(setup))


def _level_check(setup, attempts):
    """Return the check that each refinement level of a round makes at every point it compares (``refinement._refine``).

    The check projects the reported point of the round's inner problem onto
    the original variables, its first d coordinates, and evaluates f, h and
    g there (``_evaluate``), raising ValueError when one is not finite and
    real. Each refinement level checks the original objective and
    constraints at the original-coordinate projection of every compared
    point: its grid point, for ``mode_or_mean`` its mean, and at a stall
    split the two scored points, each at its reported image, which for the
    search model is the rounded image of the unit point. If an off-grid mean
    makes an original function nonfinite or undefined, the level records
    mean_unavailable and uses its valid grid point. A failure at the grid
    point or at a scored point ends the refinement. The level's inner-value
    comparison keeps its existing table and coordinate-transformation
    conventions. Each call appends to ``attempts``, whether it succeeds or
    not, and the round charges every attempt as ``_point_cost`` counts one
    evaluation of f, h and g, within the reservation ``(a L + 2 B + 1) C``
    of ``_setup`` (``_admit_layer_work``).
    """
    d = setup.grid.num_variables

    def check(point):
        attempts.append(point)
        _evaluate(setup, point[:d])

    return check


class _PointFailure(Exception):
    """f, h or g is not finite and real at a round's grid point, with the layer's attempted evaluations.

    ``__cause__`` is the ValueError of ``_evaluate``. ``evaluations`` and
    ``work`` count every evaluation that the round attempted, as
    ``_admit_layer_work`` counts them, so the round charges them.
    """

    def __init__(self, evaluations, work):
        super().__init__("f, h or g is not finite and real at the round's grid point")
        self.evaluations, self.work = evaluations, work


def _evaluate_grid_point(setup, point, evaluations=0, work=0):
    """Return ``_evaluate`` at a grid point with ``evaluations`` and ``work`` added, or raise ``_PointFailure``.

    The initial check evaluates the support representation and performs any
    required summand or whole-term scans. Every round point is checked in the
    original numerical expression. On the support-table path that evaluation
    can still fail, for example through intermediate overflow in the
    original expression. A failure becomes _PointFailure with its evaluation
    and work counts.
    """
    try:
        f, h, g, more, extra = _evaluate(setup, point)
    except ValueError as error:
        count, cost = _point_cost(setup)
        raise _PointFailure(evaluations + count, work + cost) from error
    return f, h, g, evaluations + more, work + extra


def _normalized_values(setup, f, h, g):
    """Return ``(f/s_f, h/s_h, g/s_g)`` from original-unit values, in the order of the setup's terms."""
    return (f / setup.objective.scale,
            tuple(value / term.scale for value, term in zip(h, setup.equalities, strict=True)),
            tuple(value / term.scale for value, term in zip(g, setup.inequalities, strict=True)))


def _projected(setup, fields, representation):
    """Return the ALEvaluation fields of a joint inner point with the point projected onto the original variables.

    Without a representation the inner point is the round's point and the
    fields are unchanged. With one, the first d coordinates and indices are
    the original variables and the rest the slack variables in constraint
    order (``_inner_problem``). The inner objective's value moves to
    ``inner_value`` with its source, and ``effective_value``, L_k at the
    projected point, is evaluated after f, h and g (``_effective_value``).
    The probability, the tie deficit and the mode status stay those of the
    joint point (``ALEvaluation``).
    """
    if representation is None:
        return fields
    d = setup.grid.num_variables
    fields = dict(fields)
    point, indices = fields["point"], fields["indices"]
    fields.update(point=tuple(point[:d]), indices=None if indices is None else tuple(indices[:d]),
                  inner_value=fields.pop("effective_value"), inner_value_source=fields.pop("effective_value_source"),
                  effective_value_source="evaluated")
    if representation.slacks:
        fields.update(slack_point=tuple(point[d:]), slack_indices=None if indices is None else tuple(indices[d:]))
    return fields


def _choose(setup, inner, lambda_bar, mu_bar, penalty, representation=None):
    """Read this round's point from the inner QHD result under ``options.inner_point``.

    ``_outer.grid_point`` reads the grid point of the rule, whose inner
    objective value is the inner result's table value. For ``mode_or_mean``,
    ``_outer.mean_point`` compares that table value with the inner objective
    at the valid-mass mean position, which the layer evaluates from f, h and
    g, a value that is not a table value and that can resolve finer than
    the grid spacing. The mean replaces the grid point only when its value
    is strictly smaller. The grid check of ``_setup`` does not cover the
    mean, so when f, h or g is not finite and real there the round takes the
    grid point and records why. When they are not finite and real at the
    grid point, the round raises ``_PointFailure``
    (``_evaluate_grid_point``). No rule proves that the inner problem was
    minimized.

    Without a representation the inner objective is L_k
    (``_effective_value`` at the mean). With one it is L(x, s) of
    ``_inner_objective``, and every rule acts on the joint point
    (Proposition 54 of docs/mathematics.md): ``most_probable`` reads the most
    probable joint grid point, ``best_observed`` the least observed table
    value of L(x, s), and ``mode_or_mean`` compares L(x, s) at the joint mode
    with L(x, s) at the full conditional mean (E[x], E[s]) with the same tie
    rule (``_inner_value``). The round's point is the projection x of the
    chosen joint point (``_projected``), where f, h and g are evaluated and
    L_k is evaluated for ``effective_value``.

    Returns:
        ``(fields, values)``: the ALEvaluation fields that describe the
        choice, and ``(f, h, g, evaluations, work)`` of the layer's
        evaluations, including an evaluation at the mean that was attempted.
    """
    rule = setup.options.inner_point
    fields = _projected(setup, grid_point(inner, rule), representation)
    d = setup.grid.num_variables

    def effective(values):
        return _effective_value(lambda_bar, mu_bar, penalty, *_normalized_values(setup, *values[:3]))

    if rule != "mode_or_mean":
        values = _evaluate_grid_point(setup, fields["point"])
        if representation is not None:
            fields["effective_value"] = effective(values)
        return fields, values

    def at_mean(mean):
        values = _evaluate(setup, mean[:d])
        if representation is None:
            return effective(values), (None, values)
        normalized = _normalized_values(setup, *values[:3])
        inner_value = _inner_value(representation.forms, mean[d:], lambda_bar, mu_bar, penalty, *normalized)
        return inner_value, (effective(values), values)

    compared = fields["effective_value" if representation is None else "inner_value"]
    mean, evaluated, unavailable = mean_point(inner, compared, at_mean)
    if mean is not None:
        value, (projected, values) = evaluated
        chosen = dict(kind="valid_mean", indices=None, point=tuple(mean[:d]), probability=None,
                      effective_value=value, effective_value_source="evaluated")
        if representation is not None:
            chosen.update(effective_value=projected, inner_value=value, inner_value_source="evaluated")
            if representation.slacks:
                chosen["slack_point"] = tuple(mean[d:])
        return chosen, values
    if unavailable is not None:
        # The attempted evaluations at the mean are charged.
        fields["mean_unavailable"] = unavailable
        evaluations, work = _point_cost(setup)
    else:
        evaluations, work = evaluated[1][1][3:]
    values = _evaluate_grid_point(setup, fields["point"], evaluations, work)
    if representation is not None:
        fields["effective_value"] = effective(values)
    return fields, values


def _refined_choice(refined, best, rule, setup, representation=None):
    """Read the point of the level with the least recorded relative inner objective.

    The refinement solves Optimization(objective=L, ...), with L the round's
    inner objective, L_k without a representation and L(x, s) with one. It
    compares each completed level's reported point using its recorded
    evaluation of L minus one common constant, with ties to the earlier
    level. A search-model level selects its grid candidate using the
    binary64 table of V_k = kappa*(L-c)/E. A physical level selects using L.
    The point, grid indices and probability belong to that selected level.
    For a most-probable grid point, its tie deficit, window and mode status
    come from that level's QHD result, not from the last level or a
    combination of levels. A mode_or_mean point can be off-grid. With a
    representation the level's point is joint, and the round's point is its
    projection onto the original variables (``_projected``), whose L_k the
    caller evaluates after f, h and g.

    Let epsilon bound the first level's table error in L units and eta
    bound the errors of the recorded relative objectives at completed
    level points. If the first level observes every point of G0 and reads
    best_observed, its point x1 obeys
    L(x1) <= min_G0 L + 2*epsilon.
    For a nonflat search-model level epsilon includes the positive
    conversion E/kappa and errors of coordinate mapping and evaluation.
    Since the selected point p has R(p) <= R(x1) for recorded relative
    values R(z) = L(z)-C0+n(z), |n(z)| <= eta,
    L(p) <= L(x1)+2*eta <= min_G0 L+2*epsilon+2*eta.
    Also L(p) <= L(z)+2*eta for every completed level z. With only
    one level p=x1 and the eta term is unnecessary. Exact probability
    readout alone does not make either evaluation error vanish. These
    error bounds are premises, not quantities certified by this function.
    With slack variables, ``L_k(x) <= L(x, s)`` for every slack value, and
    Proposition 54 of docs/mathematics.md relates the first level's grid
    minimum of L(x, s) to that of L_k by the slack-grid error.

    Returns:
        The ALEvaluation fields describing that level's point.
    """
    grid = best.point_indices is not None
    fields = dict(kind="grid_point" if grid else "valid_mean", indices=best.point_indices, point=best.point,
                  probability=best.point_probability, effective_value=best.objective,
                  effective_value_source="table" if grid else "evaluated", mean_unavailable=best.mean_unavailable)
    if grid and rule != "best_observed":
        read = grid_point(refined.results[best.level - 1], "most_probable")
        fields.update(tie_deficit=read["tie_deficit"], tie_window=read["tie_window"], mode_status=read["mode_status"])
    return _projected(setup, fields, representation)


def _refined_resources(refined, evaluations, work, form=None):
    """Read one refined round's resources from its refinement record (ALResources states the rules).

    The Run counts, the CX bound and the rotation count are the refinement's totals over its
    completed and stopping levels. The table evaluations add both of its
    counts, the unscaled table stage of each search-model level
    (``RefinementResources.table_evaluations``) and the solved Plans
    (``support_evaluations``), all of them grid tuples that QHD planning
    evaluated, and every candidate that automatic selection planned for the
    round's representation ``form`` (``InnerRepresentation.table_evaluations``),
    since the levels plan their own problems and reuse none of them. Width
    and restricted dimension are those of the completed levels, which share
    the round's grid, encoding and variables.
    """
    total = refined.resources
    reasons = dict(total.unavailable)
    counts = dict(layer_evaluations=evaluations, layer_work=work,
                  refinement_evaluations=total.objective_evaluations, joint_mass_reads=total.joint_mass_reads)
    unavailable = []
    for name in ("cx", "arbitrary_rotations", *RUN_COUNTS):
        counts[name] = getattr(total, name)
        if counts[name] is None:
            unavailable.append((name, reasons[name]))
    tables = {name: getattr(total, name) for name in ("table_evaluations", "support_evaluations")}
    unknown = [reasons[name] for name, value in tables.items() if value is None]
    if form is not None:
        tables["selection"] = form.table_evaluations
        if form.table_evaluations is None:
            unknown.append(form.table_evaluations_unavailable)
    counts["table_evaluations"] = None if unknown else sum(tables.values())
    if unknown:
        unavailable.append(("table_evaluations", "; ".join(unknown)))
    for name, field in (("width", "logical_width"), ("restricted_dimension", "restricted_dimension")):
        counts[name] = max((getattr(level, field) for level in refined.levels), default=None)
        if counts[name] is None:
            unavailable.append((name, "the round's refinement completed no level"))
    return ALResources(**counts, unavailable=_ordered(unavailable))


def _ordered(unavailable):
    """Sort the ``(field, reason)`` pairs of unknown ALResources counts into one fixed field order."""
    names = ("width", "restricted_dimension", "cx", "arbitrary_rotations", "table_evaluations", *RUN_COUNTS)
    order = {name: position for position, name in enumerate(names)}
    return tuple(sorted(unavailable, key=lambda item: order[item[0]]))


def _round_resources(plan, trace, evaluations, work, *, planning_reason=None, run_reason=None, form=None,
                     selected=False):
    """Read one round's resources from its Plan and Run trace (``_outer.run_counts``, ``_outer.law_count``).

    Without a Plan (planning raised, ``planning_reason``) there was no Run,
    so acquisition and host execution are zero while the planning work
    spent before the error is unknown. A preparation that raised closed its
    Run before the layer could read it, and ``_durable.closed_trace`` gives
    ``run_reason`` and, for a durable Run, the trace read from its folder,
    or an ``_outer.HeaderlessRun`` when the Run's creation raised before its
    journal header, while without a durable Run, or when its journal cannot
    be read, the counts stay unknown.
    The CX bound and the rotation count are those of the circuits the Run
    prepared (``_outer.law_count``), unknown after a preparation that raised
    once the header was committed, and zero before it.

    The table evaluations are the Plan's, and with a representation ``form``
    whose automatic selection planned candidates, the grid tuples of every
    candidate once (``InnerRepresentation.table_evaluations``), which
    include those of the round's Plan, the kept candidate's Plan that the
    round reuses: ``sum_c E_c = E_kept + sum_(c != kept) E_c``. When the
    kept candidate's planning raised during the selection (``selected``),
    the round planned nothing more, and its table evaluations are the
    candidates' count, known or unknown with its reason.
    """
    counts = dict(layer_evaluations=evaluations, layer_work=work)
    reasons = []
    if plan is None:
        for name in ("width", "restricted_dimension", "table_evaluations"):
            counts[name] = None
            reasons.append((name, planning_reason))
        if selected:
            counts["table_evaluations"] = form.table_evaluations
            reasons = [item for item in reasons if item[0] != "table_evaluations"]
            if form.table_evaluations is None:
                reasons.append(("table_evaluations", form.table_evaluations_unavailable))
    else:
        r = plan.reconstruction
        counts.update(width=r.width, restricted_dimension=r.restricted_dimension,
                      table_evaluations=r.support_evaluations)
        if form is not None and form.table_evaluations != 0:
            counts["table_evaluations"] = form.table_evaluations
            if form.table_evaluations is None:
                reasons.append(("table_evaluations", form.table_evaluations_unavailable))
    run, unknown = run_counts(trace, run_reason)
    circuits = None if run_reason is not None else run["circuit_preparations"]
    for metric in ("cx", "arbitrary_rotations"):
        counts[metric], law_reason = law_count(plan, metric, circuits, run_reason)
        if law_reason is not None:
            reasons.append((metric, law_reason))
    return ALResources(**counts, **run, unavailable=tuple(reasons) + unknown)


def _total_resources(rounds, evaluations, work):
    """Sum per-round resources plus the layer's preprocessing evaluations (ALResources states the rules)."""
    counts = dict(layer_evaluations=evaluations + sum(item.layer_evaluations for item in rounds),
                  layer_work=work + sum(item.layer_work for item in rounds),
                  refinement_evaluations=sum(item.refinement_evaluations for item in rounds),
                  joint_mass_reads=sum(item.joint_mass_reads for item in rounds))
    entries = [(f"round {k}", item) for k, item in enumerate(rounds)]
    summed, reasons = total_counts(entries, ("cx", "arbitrary_rotations", "table_evaluations", *RUN_COUNTS))
    reasons = list(reasons)
    for name in ("width", "restricted_dimension"):
        largest, unknown = total_counts(entries, (name,))
        if unknown:
            counts[name] = None
            reasons += unknown
        else:
            counts[name] = max((getattr(item, name) for item in rounds), default=None)
            if counts[name] is None:
                reasons.append((name, "no round selected a Plan"))
    return ALResources(**counts, **summed, unavailable=_ordered(reasons))


def _best_and_last(iterations, tolerance):
    """Return the rounds of the reported best and last points (AugmentedLagrangianRecord states the rule).

    Rounds are compared by f and not by L_k, because each round's L_k has its
    own multipliers and penalty, so values of different rounds measure
    different functions. A feasible round comes before every infeasible one,
    and among infeasible rounds the smaller normalized infeasibility comes
    first. The experiment code behind the results of Wu et al.,
    arXiv:2605.12066v1, also prioritizes feasibility, then f among feasible
    rounds and violation among infeasible rounds. For equal infeasible
    violations it keeps the earlier round regardless of f, whereas this
    layer compares f before the round number. These ordering rules are
    properties of that code, not statements in the paper. That code uses
    unscaled violation and a strict feasibility test, while this layer uses
    normalized infeasibility and accepts equality at the tolerance.
    """
    chosen = [item for item in iterations if item.evaluation is not None]
    if not chosen:
        return None, None
    feasible = [item for item in chosen if item.evaluation.infeasibility <= tolerance]
    if feasible:
        best = min(feasible, key=lambda item: (item.evaluation.objective, item.iteration))
    else:
        best = min(chosen, key=lambda item: (item.evaluation.infeasibility, item.evaluation.objective,
                                             item.iteration))
    return best.iteration, chosen[-1].iteration


def solve_augmented_lagrangian(problem, *, qhd, options=AugmentedLagrangian(), refinement=None, backend=None,
                               execution=None, shots=None, seed=None, limits=None, progress=None, directory=None):
    """Minimize a constrained objective by an augmented-Lagrangian sequence of QHD solves.

    `solve_augmented_lagrangian(problem, qhd=QHD(...))` solves a
    `ConstrainedOptimization`, the box problem with equalities `h_i(x) = 0` and
    inequalities `g_j(x) <= 0`. Round k builds the normalized effective objective
    `L_k(x) = F + sum_i lambda_bar_i H_i + (rho/2) sum_i H_i**2 + sum_j ([mu_bar_j + rho G_j]_+**2 - mu_bar_j**2)/(2 rho)`
    with `F = f/s_f`, `H_i = h_i/s_{h_i}`, `G_j = g_j/s_{g_j}`, the round's multipliers
    lambda_bar = lambda-bar^k, mu_bar = mu-bar^k and penalty rho = rho_k. The equality
    part is Eq. (6) of Wu et al., arXiv:2605.12066v1, and the inequality part the
    Powell-Hestenes-Rockafellar (PHR) term (Rockafellar, doi:10.1007/BF01580138). The
    round plans
    `Optimization(objective=L_k, variables, bounds=the preprocessed box, unit=problem.unit)`
    with `qhd`, runs it with `prepare` and `submit`, reads a point x_{k+1} by
    `options.inner_point`, evaluates f, h and g there with numerical functions formed
    once for the run (SymPy `lambdify`), and updates the multipliers to
    `lambda+ = lambda + rho h` and `mu+ = max(0, mu + rho g)` and the penalty (Birgin
    and Martinez, doi:10.1137/1.9781611973365, Algorithm 4.1 and Eqs. (4.7)-(4.9)). The
    guide's [constrained problems](../../algorithms/qhd.md#constrained-problems) section
    explains each step.

    The run stops when the normalized tests of `options.termination` hold. With
    `feasibility_and_complementarity` these are complementarity <= epsilon_c and
    infeasibility <= epsilon_f (Algencan's Eqs. (10.7)-(10.8)), and the status is
    `feasible_complementary`. With `feasibility` the test is infeasibility <= epsilon_f,
    and the status is `feasible`. Otherwise it stops after `options.max_iterations`
    rounds, when an inner result has no valid point, when the cumulative `limits` cannot
    fund a round, or when an inner planning, preparation or execution raises after the
    first round
    ([`AugmentedLagrangianRecord`][nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangianRecord]
    lists the statuses). No round is retried. A stopping status does not assess
    optimality. Like QHD itself, the layer is not a global optimizer. The rounds read
    points of a finite grid of the box, except an off-grid mean under `mode_or_mean`,
    and no status is a statement about the continuous problem. Read feasibility,
    complementarity and `termination` alongside the objective. In the first round an
    error propagates unchanged, because no round has completed and a configuration that
    QHD rejects should surface at once. That includes a first round whose Run refuses
    the limits, while limits that the layer's own test finds unable to fund the first
    round's circuit or shots end the run with `budget_exhausted` after zero rounds.

    Args:
        problem (ConstrainedOptimization): The problem, with live SymPy expressions.
        qhd (QHD): The QHD configuration of every round.
        options (AugmentedLagrangian): The layer's options. Default
            `AugmentedLagrangian()`.
        refinement (BoxRefinement | None): Box refinement of every round's inner
            objective, L_k or L(x, s), or None (the default) for one QHD solve per
            round, which leaves every result as without this argument.
        backend (object | None): Passed to every round's `prepare`, as in
            `nwqlib.solve`.
        execution (str | None): `"quantum"` (the default) or `"classical"`, as in
            `nwqlib.solve`.
        shots (int | None): Shots per round, or None for exact readout.
        seed (int | None): Nonnegative root seed of the run.
        limits (ExecutionLimits | None): Cumulative limits of the whole run, the
            defaults when None. Each round's Run, or each level's Run with refinement,
            gets the remaining circuits, shots, data bytes and synthesis work and the
            per-Run caps unchanged.
        progress (object | None): Passed to every round's Run, as in `nwqlib.solve`.
        directory (str | Path | None): A new directory for a saved run that
            `resume_augmented_lagrangian` can continue, or None to keep every inner Run
            in memory. It must not exist, and missing parents are created.

    Returns:
        result (ConstrainedQHDResult): The run. `candidate` and `objective` give the
            best point and f there in original units, `termination` the stopping status,
            `multipliers()` the last round's multiplier estimates in original units, and
            `record` the full record. The function never calls
            `constrained_grid_minimum`.

    Raises:
        TypeError: If `options` is not an `AugmentedLagrangian` or `refinement` is
            neither a `BoxRefinement` nor None.
        ValueError: If `refinement.point_rule` differs from `options.inner_point`, if
            the refinement options ask for what the QHD configuration cannot carry out,
            or if the first round fails as described above.
        FileExistsError: If `directory` exists. The message names
            `resume_augmented_lagrangian`, which continues it.

    Inequality representation:
        Under `options.inequality_form="slack"` or `"auto"` the round first fixes how
        its kept inequalities enter (Proposition 54 of
        [the mathematics page](../../mathematics.md#r54)). A converted inequality adds a
        slack variable, after the problem's variables and with the box `[0, U]` of its
        cap, and the round plans its inner objective L(x, s) instead, reads a joint
        point and projects it to x, where the update above is unchanged
        (`ALIteration.representation`).

    Point selection with refinement:
        With `refinement`, round k instead runs box refinement
        ([`refine_box`][nwqlib.algorithms.qhd.refinement.refine_box]) on its inner
        problem, from the preprocessed box and the round's slack boxes, with `qhd`. This
        is the order of the reproduction scripts of Wu et al., arXiv:2605.12066v1, which
        refine inside each multiplier round. `refinement.point_rule` reads every level
        and must equal `options.inner_point`, which has no second role, and two
        different rules are rejected before any work, as are stall-split and level
        initial-state options that the QHD configuration cannot carry out. A stall
        split, when the options ask for one, acts within each round's refinement with
        the options' budget.

        The round takes as x_{k+1} the projection of the point of the level with the
        least recorded relative value of the inner objective, the earlier level on ties.
        The round compares its inner objective, L_k under the PHR policy and L(x, s)
        with an inequality representation, by the point rules that the Point rules note
        of [`AugmentedLagrangian`][nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangian]
        describes. The outer point is the projection onto the original variables, and
        effective_value is the PHR value L_k evaluated at that projection when a
        representation is present. The best and last points are chosen among rounds by
        the rules above, not among levels.

        f, h and g are then evaluated at the outer point in the original coordinates,
        and the update is unchanged. The point is the level's reported point
        (`RefinementLevel.point`), for the search model the image `a + D u` of the
        level's unit point u rounded once to binary64, while the level evaluated its
        objective at the exact image. f, h and g are evaluated at the rounded point
        because it is the point that the round and the run return to the user, and the
        recorded residuals and objective must describe that point.

        Each refinement level checks the original objective and constraints at the
        original-coordinate projection of every compared point. If an off-grid mean
        makes an original function nonfinite or undefined, the level records
        `mean_unavailable` and uses its valid grid point. The level's inner-value
        comparison keeps its existing table and coordinate-transformation conventions.

    Failures and stopping with refinement:
        When an original function is not finite and real at a level's grid point, or at
        a point that a stall split scores, that level fails as a level whose Run raised
        does, before a split or a next box is decided. In a round after the first, or
        after a completed level, the refinement stops with `inner_failed` and the run
        ends after the round, and in the first level of round 0 the error propagates.
        Without refinement the round keeps the grid point when the mean fails, and a
        failure at the grid point ends a round after the first with `inner_failed` and
        propagates in round 0.

        A refinement that completed a level always gives the round its point. After
        `budget_exhausted`, `inner_failed` or `no_valid_point` the run records the round
        and stops, with the success status when the round's point met the stopping test
        and with the refinement's reason otherwise, and a refinement that completed no
        level ends the run with its own reason.

        An error of the first level of round 0 propagates. Any later error of a level
        ends the run after keeping the completed rounds and levels, with `inner_failed`
        or, when its round kept a point that met the stopping test, with the success
        status.

    Limits with refinement:
        `limits` stays cumulative over all rounds and levels, and each level's Run gets
        what the earlier rounds and levels left. The configuration fixes before the
        first round at most `max_iterations * refinement.max_levels` inner solves and the
        host-work bound `AugmentedLagrangianRecord.admitted_work_bound`.

    Random streams:
        Every round's random streams come from `SeedSequence(seed).spawn`, as
        [`compare`][nwqlib.scientist.compare] seeds its candidates, and the round
        records its child's entropy and spawn key. With refinement, level z of round k
        plans from the z-th child of the round's child, whose spawn key `(k, z - 1)` its
        level record keeps.

    Saved runs:
        With `directory` every inner Run is saved under `iterations/<k>/run/`, or
        `iterations/<k>/levels/<z>/run/` with refinement, and the outer record is
        rewritten after each completed round or level and once more at the end, so that
        `resume_augmented_lagrangian` can continue an interrupted run. The outer record
        stores the configuration of the backend of every inner Run, and the model of a
        noisy Aer backend is saved once in the directory. The rounds are the same as
        without a directory, except that a saved Run also stores its run log and inputs,
        which count against `max_data_bytes`, so a run whose data limit binds can stop
        earlier, and that a round whose preparation raised reads its counts from its
        closed Run folder.

    Examples:
        This is the unit-disk example of the guide. On the 4-point grid of each axis the
        first two rounds read the infeasible point (0.8, 0.8), raise the multiplier to
        0.56 and the penalty to 2, and the third round reads (0.6, 0.8) on the circle,
        the least evaluated objective among the feasible grid points, so the gap to the
        explicit grid reference is 0.

        >>> import sympy as sp
        >>> from nwqlib.problems import ConstrainedOptimization
        >>> from nwqlib.algorithms.qhd import (
        ...     QHD, AugmentedLagrangian, constrained_grid_minimum,
        ...     solve_augmented_lagrangian)
        >>> x, y = sp.symbols("x y", real=True)
        >>> problem = ConstrainedOptimization(
        ...     objective=(x - 1)**2 + (y - 1)**2, variables=(x, y),
        ...     bounds=((0.0, 1.0), (0.0, 1.0)), inequalities=(x**2 + y**2 - 1,))
        >>> result = solve_augmented_lagrangian(
        ...     problem, qhd=QHD(num_grid_points=4, num_steps=80, total_time=10.0),
        ...     options=AugmentedLagrangian(stationarity=True),
        ...     execution="classical", seed=7)
        >>> print(result.termination, [round(v, 6) for v in result.candidate],
        ...       round(result.objective, 6))
        feasible_complementary [0.6, 0.8] 0.2
        >>> print(len(result.iterations), round(result.penalty, 6))
        3 2.0
        >>> print(constrained_grid_minimum(result).gap)
        0.0
    """
    import numpy as np

    if type(options) is not AugmentedLagrangian:
        raise TypeError("options must be AugmentedLagrangian")
    if refinement is not None:
        if type(refinement) is not BoxRefinement:
            raise TypeError("refinement must be BoxRefinement or None")
        if refinement.point_rule != options.inner_point:
            raise ValueError(f"refinement.point_rule={refinement.point_rule!r} differs from options.inner_point="
                             f"{options.inner_point!r}. With refinement one rule reads every level, so set both "
                             "to the same rule")
    execution, limits = check_arguments(qhd, execution, shots, seed, limits)
    if refinement is not None:
        check_refinement_options(refinement, qhd, execution)
    if directory is not None:
        require_new(directory, "constrained")
    setup = _setup(problem, qhd, options, refinement)
    root = np.random.SeedSequence(seed)
    if directory is None:
        return _rounds(problem, qhd, setup, refinement, backend, execution, shots, root, limits, progress)
    settings = dict(problem=problem.model_dump(mode="json"), qhd=qhd.model_dump(mode="json"),
                    options=options.model_dump(mode="json"),
                    refinement=None if refinement is None else refinement.model_dump(mode="json"),
                    execution=execution, shots=shots, entropy=root.entropy, limits=limits.model_dump(mode="json"),
                    setup=dict(preprocessing=setup.preprocessing.model_dump(mode="json"),
                               check_work=setup.check_work, check_evaluations=setup.check_evaluations))
    objects = (problem.objective, problem.variables, problem.equalities, problem.inequalities)
    with Directory.create(directory, "constrained", objects, settings, backend) as outer:
        return _rounds(problem, qhd, setup, refinement, outer.backend, execution, shots, root, limits, progress,
                       outer=outer)


def plan_augmented_lagrangian(problem, *, qhd, options=AugmentedLagrangian(), execution=None, shots=None, seed=None):
    """Plan round 0 of `solve_augmented_lagrangian` without a Run, to inspect its preprocessing and cost.

    The call does what `solve_augmented_lagrangian` does before its first inner Run,
    with the same arguments: the checks and the support-wise preprocessing of the
    problem, the representation of the kept inequalities in round 0 under
    `options.inequality_form`, with its automatic selection when chosen, and the Plan of
    round 0's inner problem, planned from the first child of `SeedSequence(seed)` as the
    solve plans it. The returned Plan is therefore the Plan that the solve's round 0
    would prepare, and [`nwqlib.estimate(plan)`][nwqlib.scientist.estimate] and
    [`circuit_resources(plan)`][nwqlib.algorithms.qhd.resources.circuit_resources] apply
    to it. Its width, restricted dimension and CX count describe the inner problem with
    its slack variables. Nothing is prepared, executed or measured, no state is evolved,
    and neither a classical reference nor `constrained_grid_minimum` runs, so the call
    also serves problems whose evolution cannot be simulated. A resource-only QHD with
    `initial_state_preparation="none"` is accepted, since no round runs.

    A planning-only call reports the first round only: the preprocessing, with the
    inequality ranges on the grid, round 0's representation, with its slack caps and its
    slack-grid error bound for the initial multipliers and penalty, and round 0's inner
    Plan. Later rounds' multipliers and penalties depend on the points that executed
    rounds read, so their representations, Plans and costs are not known here. The
    structural bounds of a whole run, at most `options.max_iterations` rounds within the
    work and byte bounds of `AugmentedLagrangianRecord.admitted_work_bound`, hold for
    every round. They are upper bounds, not an executed trajectory or the exact costs of
    later rounds. Box refinement is not planned here. A refined round keeps the PHR form
    where the branch tests do not apply, and its first level plans the inner problem on
    the same box with the physical scaling, or its unit-box normalization with the
    search model.

    Args:
        problem (ConstrainedOptimization): The problem, with live SymPy expressions.
        qhd (QHD): The QHD configuration of every round.
        options (AugmentedLagrangian): The layer's options. Default
            `AugmentedLagrangian()`.
        execution (str | None): `"quantum"` (the default) or `"classical"`, as in
            `solve_augmented_lagrangian`. Automatic selection converts inequalities on
            the quantum route only.
        shots (int | None): Shots per round, or None for exact readout.
        seed (int | None): Nonnegative root seed of the run.

    Returns:
        planned (tuple): `(preprocessing, representation, plan)`, the
            [`ConstraintPreprocessing`][nwqlib.algorithms.qhd.constrained_records.ConstraintPreprocessing]
            of the run, round 0's
            [`InnerRepresentation`][nwqlib.algorithms.qhd.constrained_records.InnerRepresentation]
            or None under `inequality_form="phr"`, and round 0's inner QHD Plan.

    Raises:
        ValueError: As `solve_augmented_lagrangian` before its first Run, including a
            round-0 planning that QHD refuses, which a solve would raise from its first
            round.
    """
    import numpy as np

    if type(options) is not AugmentedLagrangian:
        raise TypeError("options must be AugmentedLagrangian")
    execution, _ = check_arguments(qhd, execution, shots, seed, None, executed=False)
    setup = _setup(problem, qhd, options)
    child = np.random.SeedSequence(seed).spawn(1)[0]
    lambda_bar, mu_bar, penalty, _, _ = _continuation(problem, setup, ())
    representation, inner, planned = _round_problem(setup, problem, qhd, execution, shots, child, lambda_bar,
                                                    mu_bar, penalty, False)
    if isinstance(planned, Exception):
        raise planned
    plan = planned if planned is not None else plan_round(inner, qhd, execution, shots, child)
    return setup.preprocessing, representation, plan


def _continuation(problem, setup, iterations):
    """Return the multipliers, penalty, previous measure and spent counts with which round ``len(iterations)`` starts.

    Before the first round they are the options' initial multipliers and
    penalty, no previous measure and zero counts. After completed rounds,
    every one of which went on to a next round, they are the next
    multipliers, next penalty and measure of the last round's update
    (``_update``) and the sums of the rounds' ``_outer.USED_COUNTS``, which
    the next round's remainder of the cumulative limits subtracts. A round
    that goes on completed its Runs, so its counts are known, while a
    refinement level whose preparation raised leaves them unknown only in a
    round that ends the run. The rounds of a new run and of a resumed one
    start from this one rule.
    """
    options = setup.options
    used = dict.fromkeys(USED_COUNTS, 0)
    for item in iterations:
        for name in used:
            used[name] += getattr(item.resources, name)
    if iterations:
        last = iterations[-1].evaluation
        return (last.next_equality_multipliers, last.next_inequality_multipliers, last.next_penalty, last.measure,
                used)
    lambda_bar = _counted(options.equality_multipliers, len(setup.equalities), "equality_multipliers", 0.0)
    mu_bar = tuple(_counted(options.inequality_multipliers, len(problem.inequalities), "inequality_multipliers",
                            0.0)[j] for j in setup.preprocessing.inequalities)
    return lambda_bar, mu_bar, options.initial_penalty, None, used


def _solves(problem, plan, identity, representation):
    """Whether the inner ``plan`` solves the recorded inner objective ``identity`` of a round of ``problem``.

    The identity is recomputed as the round formed it (``_inner_problem``):
    the ``srepr`` of the Plan's objective, the ``srepr`` of the problem's
    variables followed by the slack variables of ``representation`` as
    ``_slack_symbols`` names them, and the Plan's box. A Plan over other
    variables, slack symbols or box, or of another objective, gives another
    identity. The archive loader and resume use it to bind a saved inner
    Result to its round.
    """
    from .records import objective_reference

    symbols = _slack_symbols(problem)
    slacks = () if representation is None else representation.slacks
    variables = tuple(problem.variables) + tuple(symbols[axis.inequality] for axis in slacks)
    found = objective_reference(sp.srepr(plan.problem.objective), tuple(sp.srepr(v) for v in variables),
                                plan.problem.bounds).identity
    return found == identity and tuple(sp.srepr(v) for v in plan.problem.variables) == tuple(
        sp.srepr(v) for v in variables)


def _rounds(problem, qhd, setup, refinement, backend, execution, shots, root, limits, progress, *, outer=None,
            iterations=(), results=(), pending=((), (), None), representation=None):
    """Run the rounds of ``solve_augmented_lagrangian`` from the completed ``iterations``, arguments already checked.

    ``root`` is the run's ``numpy.random.SeedSequence``. A new run passes no
    iterations. ``resume_augmented_lagrangian`` passes the completed rounds,
    their inner results and, in ``pending``, the completed levels of the
    round in progress with their results and the stop that the last of them
    decided. The random-stream children of the completed rounds are spawned
    and discarded, and ``_continuation`` gives the state of the next round,
    so it plans from the child and with the multipliers, penalty and spent
    limits of the uninterrupted run. With ``outer``, a ``_durable.Directory``,
    every inner Run is durable in its round's or level's folder, the outer
    record is committed after every round that goes on and after every
    completed level, and the final record is committed at the end. A round
    or level whose Run exists continues that Run (``_durable.reopen``), and
    an error of that reopening, or an outcome of the reopened Run that only
    new work could resolve (``_durable.unrecoverable``), propagates instead
    of ending the run, because the round would otherwise need a second Run.
    """
    from nwqlib.scientist import prepare, submit
    from .records import objective_reference

    options = setup.options
    work_bound, bytes_bound = _admitted_bounds(qhd, options, refinement, execution, len(setup.inequalities))
    variables = tuple(sp.srepr(v) for v in problem.variables)
    box = setup.preprocessing.bounds
    equality_scales = tuple(term.scale for term in setup.equalities)
    inequality_scales = tuple(term.scale for term in setup.inequalities)
    iterations, results, failure = list(iterations), list(results), None
    termination = "iteration_limit"
    resumed = len(iterations)
    root.spawn(resumed)
    for k in range(resumed, options.max_iterations):
        lambda_bar, mu_bar, penalty, previous, used = _continuation(problem, setup, iterations)
        funded, reason = round_limits(limits, used, execution, shots)
        if funded is None:
            termination, failure = "budget_exhausted", reason
            break
        child = root.spawn(1)[0]
        # The round's representation of its kept inequalities, fixed for the whole round, its refinement
        # levels included, and committed before its first inner Run, so that a resumed round reuses it
        # instead of selecting again (_round_problem). Its inner problem is L_k under the PHR policy.
        saved = representation if k == resumed else None
        form, inner_problem, planned = _round_problem(setup, problem, qhd, execution, shots, child, lambda_bar,
                                                      mu_bar, penalty, refinement is not None, saved)
        effective = (inner_problem.objective if form is None
                     else _effective_objective(setup, lambda_bar, mu_bar, penalty))
        base = dict(iteration=k, penalty=penalty, equality_multipliers=lambda_bar,
                    inequality_multipliers=mu_bar,
                    effective_objective=objective_reference(sp.srepr(effective), variables, box).identity,
                    entropy=child.entropy, spawn_key=tuple(child.spawn_key))
        if form is not None:
            base["representation"] = form
            if outer is not None and saved is None:
                outer.commit(representation=form)
        stop = None
        if refinement is not None:
            # The refinement of the inner objective from the preprocessed box spawns its levels
            # from this round's child, draws on the remainder of the whole
            # run's limits, and records a first-level error as inner_failed
            # after round 0 (refinement._refine). A durable round commits the
            # completed rounds with its completed levels after each level.
            frontier = None
            if outer is not None:
                levels, level_results, blocked = pending if k == resumed else ((), (), None)
                frontier = Frontier(outer=outer, base=outer.path / "iterations" / str(k), levels=levels,
                                    results=level_results, blocked=blocked)
            attempts = []
            refined = _refine(inner_problem, qhd, refinement, execution, shots, backend, child, limits, progress,
                              used=used, completed=k, frontier=frontier, check=_level_check(setup, attempts))
            base["refinement"] = refined
            results.append(refined.results)
            # The level checks of this round: those made in this call, and for each level that a resumed round
            # restored the checks that it made when it ran, one per compared point (_level_check): a, 2 for
            # mode_or_mean, whose mean counts even when unavailable, and 1 otherwise, and two more for its
            # scored points when it split. A restored level completed, so both scored points passed.
            a = 2 if options.inner_point == "mode_or_mean" else 1
            restored = () if frontier is None else frontier.levels
            checks = sum(a + 2 * (level.split is not None) for level in restored) + len(attempts)
            count, cost = _point_cost(setup)
            checked = (checks * count, checks * cost)
            if refined.best is None:
                iterations.append(ALIteration(**base, resources=_refined_resources(refined, *checked, form)))
                termination = refinement_stop(refined.termination, 0)
                failure = refined.failure if termination in ("inner_failed", "budget_exhausted") else None
                break
            chosen = _refined_choice(refined, refined.best, options.inner_point, setup, form)
            try:
                f, h_raw, g_raw, evaluations, work = _evaluate(setup, chosen["point"])
                if form is not None:
                    chosen["effective_value"] = _effective_value(lambda_bar, mu_bar, penalty,
                                                                 *_normalized_values(setup, f, h_raw, g_raw))
            except ValueError as error:
                # f, h or g is not finite and real at the point: a later failure of the
                # round, recorded after round 0 (_outer.inner_failure). The attempted
                # evaluations are charged as _admit_layer_work counts them.
                failure = inner_failure(error, k)
                evaluations, work = _point_cost(setup)
                iterations.append(ALIteration(**base, resources=_refined_resources(refined, evaluations + checked[0],
                                                                                   work + checked[1], form)))
                termination = "inner_failed"
                break
            stop = refinement_stop(refined.termination, len(refined.levels))
        else:
            # Only the inner planning, preparation and execution count as an
            # inner failure, and a first-round failure propagates
            # (_outer.inner_failure). An error in the layer's own code
            # propagates too, and so does an error of reopening a durable
            # round's Run.
            folder = None if outer is None else outer.path / "iterations" / str(k) / "run"
            plan = prepared = trace = run_reason = None
            run = reopen(folder, backend=backend, progress=progress)
            reopened = run is not None
            if reopened:
                plan = run.plan
                if not _solves(problem, plan, base["effective_objective"] if form is None else form.inner_objective,
                               form):
                    run.close()
                    raise ValueError(f"the Run in {folder} solves another inner objective than round {k} of the "
                                     "durable record")
            elif isinstance(planned, Exception):
                # Automatic selection already planned this representation, and its refusal is the round's.
                failure = inner_failure(planned, k)
            elif planned is not None:
                plan = planned
            else:
                try:
                    plan = plan_round(inner_problem, qhd, execution, shots, child)
                except Exception as error:
                    failure = inner_failure(error, k)
            if plan is not None:
                bound = range_bound((t.minimum, t.maximum) for t in plan.reconstruction.support_values)
                base.update(plan_id=plan.content_id,
                            **({"effective_range_bound": bound} if form is None else {"inner_range_bound": bound}))
            if plan is not None and not reopened:
                try:
                    prepared = prepare(plan, backend=backend, limits=funded, progress=progress, directory=folder)
                except Exception as error:
                    failure = inner_failure(error, k)
                    # The work that the failed preparation spent, from the durable Run's journal.
                    trace, run_reason = closed_trace(folder, backend)
                if prepared is not None:
                    run = prepared.run
            if run is None:
                # A Run without a journal header has no recorded identity (_outer.HeaderlessRun).
                if trace is not None and not isinstance(trace, HeaderlessRun):
                    base["run_id"] = trace.run_id
                resources = _round_resources(plan, trace, 0, 0, run_reason=run_reason, form=form,
                                             planning_reason="planning raised before selecting a Plan",
                                             selected=isinstance(planned, Exception))
                iterations.append(ALIteration(**base, resources=resources))
                results.append(None)
                termination = "inner_failed"
                break
            error_text = None
            with run:
                base["run_id"] = run.run_id
                try:
                    inner = run.wait() if reopened else submit(prepared).wait()
                except Exception as error:
                    if reopened and unrecoverable(error, outer.path, end=outer.end_at_unfinishable):
                        raise
                    error_text = inner_failure(error, k)
            # Read after the Run closed, the trace holds every charge, including
            # data recorded after the Result's own snapshot (_outer.run_counts).
            trace = run.trace
            if error_text is not None:
                iterations.append(ALIteration(**base, resources=_round_resources(plan, trace, 0, 0, form=form)))
                results.append(None)
                termination, failure = "inner_failed", error_text
                break
            base.update(result_id=inner.content_id, valid_mass=inner.valid_mass, invalid_mass=inner.invalid_mass,
                        missing=inner.missing)
            results.append(inner)
            if inner.value is None:
                iterations.append(ALIteration(**base, resources=_round_resources(plan, trace, 0, 0, form=form)))
                termination = "no_valid_point"
                break
            try:
                chosen, (f, h_raw, g_raw, evaluations, work) = _choose(setup, inner, lambda_bar, mu_bar, penalty,
                                                                       form)
            except _PointFailure as error:
                # f, h or g is not finite and real at the grid point in its written binary64 evaluation
                # (_evaluate_grid_point): a later failure of the round, recorded after round 0
                # (_outer.inner_failure).
                failure = inner_failure(error.__cause__, k)
                iterations.append(ALIteration(**base, resources=_round_resources(plan, trace, error.evaluations,
                                                                                 error.work, form=form)))
                termination = "inner_failed"
                break
        update = _update(options, penalty, lambda_bar, mu_bar, h_raw, g_raw, equality_scales, inequality_scales,
                         previous)
        stationarity, unavailable, more, extra = _stationarity(setup, chosen["point"], update["lambda_plus"],
                                                               update["mu_plus"])
        evaluation = ALEvaluation(
            rule=options.inner_point, **chosen,
            objective=f, objective_normalized=f / setup.objective.scale,
            equality_residuals=h_raw, equality_residuals_normalized=update["h"],
            inequality_residuals=g_raw, inequality_residuals_normalized=update["g"],
            infeasibility=update["infeasibility"], complementarity=update["complementarity"],
            measure=update["measure"],
            tentative_equality_multipliers=update["lambda_plus"],
            tentative_inequality_multipliers=update["mu_plus"],
            next_equality_multipliers=update["lambda_next"], next_inequality_multipliers=update["mu_next"],
            equality_truncated=update["lambda_truncated"], inequality_truncated=update["mu_truncated"],
            next_penalty=update["next_penalty"], stationarity=stationarity,
            stationarity_unavailable=unavailable,
        )
        if refinement is None:
            resources = _round_resources(plan, trace, evaluations + more, work + extra, form=form)
        else:
            resources = _refined_resources(refined, evaluations + more + checked[0], work + extra + checked[1], form)
        iterations.append(ALIteration(**base, evaluation=evaluation, resources=resources))
        feasible = update["infeasibility"] <= setup.feasibility
        if options.termination == "feasibility" and feasible:
            termination = "feasible"
            break
        if options.termination == "feasibility_and_complementarity" and feasible and (
            update["complementarity"] <= setup.complementarity
        ):
            termination = "feasible_complementary"
            break
        # A refinement cut short after a completed level ends the run once the
        # round's own stopping test has failed (_outer.refinement_stop).
        if stop is not None:
            termination = stop
            failure = refined.failure if stop in ("inner_failed", "budget_exhausted") else None
            break
        if outer is not None:
            outer.commit(iterations=tuple(iterations))
    best, last = _best_and_last(iterations, setup.feasibility)
    record = AugmentedLagrangianRecord(
        problem=problem, qhd=qhd, options=options, refinement=refinement, execution=execution, shots=shots,
        entropy=root.entropy,
        limits=limits, preprocessing=setup.preprocessing, equality_scales=equality_scales,
        inequality_scales=inequality_scales, feasibility_tolerance=setup.feasibility,
        complementarity_tolerance=setup.complementarity,
        admitted_work_bound=work_bound, admitted_bytes_bound=bytes_bound,
        iterations=tuple(iterations), termination=termination, failure=failure, best=best, last=last,
        resources=_total_resources(tuple(item.resources for item in iterations), setup.check_evaluations,
                                   setup.check_work),
    )
    if outer is not None:
        outer.commit(record=record)
    return ConstrainedQHDResult(record, problem, tuple(results))


_STATUS_TEXT = {
    "feasible_complementary": (
        "The last round meets the normalized complementarity and feasibility tests (Birgin and Martinez, "
        "doi:10.1137/1.9781611973365, Eqs. (10.7)-(10.8)). Optimality was not assessed: the projected "
        "stationarity of Eq. (10.6) is not a stopping condition, and a finite grid can stop away from its "
        "feasible grid minimum, which constrained_grid_minimum compares in evaluated binary64 values, "
        "without bounding evaluation error."),
    "feasible": ("The last round meets the normalized feasibility test alone (termination='feasibility'). "
                 "Complementarity and optimality were not assessed."),
    "iteration_limit": "No round met the stopping test within max_iterations rounds.",
    "no_valid_point": ("The last inner result, or a level of the last round's refinement, had no valid grid "
                       "point, so the run stopped."),
    "budget_exhausted": ("The remaining cumulative limits could not fund another round, or another level of "
                         "the last round's refinement."),
    "inner_failed": ("The last round's inner planning or execution, or that of a level of its refinement, "
                     "raised, or f, h or g was not finite and real at its point or at a point that a level of its "
                     "refinement compared, its grid point or a point that a stall split scored, so the run "
                     "stopped."),
    "flat_objective": ("The last round's refinement found no variation of the round's inner objective on the "
                       "preprocessed box's grid, so it had nothing to normalize, and the round has no point and no "
                       "update."),
    "unresolved_objective": ("The last round's refinement could not tell the variation of the round's inner "
                             "objective on the preprocessed box's grid from evaluation error, so the round has no "
                             "point and no update."),
    "width_floor": ("A side of the preprocessed box is at the refinement's width floor, so the last round's "
                    "refinement solved no level, and the round has no point and no update."),
}

_RULE_TEXT = {
    "best_observed": (
        "Inner points are QHD candidates. Each candidate has the least evaluated binary64 value of the "
        "objective solved at its level among observed positive-weight grid points, with the smallest grid "
        "index on ties. With complete exact readout and positive probability everywhere this covers the "
        "whole grid. The solved objective is the round's inner objective for an unrefined or physical-model "
        "solve, and its positive normalization V_k for a search-model level."
    ),
    "most_probable": ("Inner points are the most probable valid grid points. No bound relates them to the "
                      "grid minimizer of the round's inner objective."),
    "mode_or_mean": ("Inner points are the most probable point or the off-grid valid-mass mean, whichever "
                     "has the smaller value of the round's inner objective, with a tie going to the grid point. "
                     "The value at a mean is evaluated, not a table value."),
}

# Every run's statements name the objective that its rounds compared (AugmentedLagrangian.inner_point).
_COMPARED_TEXT = (
    "The round compares its inner objective, L_k under the PHR policy and L(x, s) with an inequality "
    "representation. A physical level solves this objective, and a search-model level solves its positive "
    "normalization. The point rules act in the inner coordinates. best_observed chooses the least observed table "
    "value. most_probable chooses a joint mode. mode_or_mean compares the joint mode with the joint conditional "
    "mean using the inner objective, with a tie going to the grid point. Refinement compares recorded relative "
    "values of the inner objective across levels and takes the earlier level on a tie. The outer point is the "
    "projection onto the original variables, and effective_value is the PHR value L_k evaluated at that "
    "projection when a representation is present."
)


@dataclass(frozen=True)
class ConstrainedQHDResult:
    """Result of an augmented-Lagrangian run: its record, live problem and inner QHD results.

    `solve_augmented_lagrangian`, `resume_augmented_lagrangian` and
    `load_augmented_lagrangian` return it. The answer is `candidate`, the best point in
    original coordinates, with `objective`, f there in original units, and
    `termination`, the stopping status. Read the feasibility and complementarity of that
    point in `best.evaluation`, and the multiplier estimates with `multipliers()`. A
    stopping status does not assess optimality. `best` and `last` are separate rounds.
    `candidate` and `objective` describe `best`. The multipliers and the penalty belong
    to `last`. They are the tentative multipliers lambda+ and mu+ of the last round,
    which are the run's multiplier estimates, next to the multipliers that round used,
    and they do not form a KKT pair with `best`. `print(result)` summarizes the run,
    `report()` returns it as JSON-ready data, and `save(path)` writes it to a new
    directory.

    Attributes:
        record: The portable
            [`AugmentedLagrangianRecord`][nwqlib.algorithms.qhd.constrained_records.AugmentedLagrangianRecord].
        problem: The live ConstrainedOptimization.
        results: The inner QHDAnalysis of each round in `record.iterations`, or None for
            a round whose inner solve raised. With refinement, each round's entry is the
            tuple of the QHDAnalysis of its completed levels, in level order.
    """

    record: AugmentedLagrangianRecord
    problem: ConstrainedOptimization
    results: tuple

    def __post_init__(self):
        """Bind the record to its live problem and to the inner results it names, one per round or level.

        The mode status that a level or a round's evaluation copied must be
        its named inner Result's, the best level's for a refined round, so a
        copy that disagrees with its Result is refused without re-execution.
        """
        from .records import QHDAnalysis

        def named(result, identity):
            return isinstance(result, QHDAnalysis) and result.content_id == identity

        def copied(evaluation, result):
            return evaluation is None or evaluation.mode_status in (None, result.mode_status)

        def matches(result, item):
            if item.refinement is not None:
                levels, best = item.refinement.levels, item.refinement.best_level
                return type(result) is tuple and len(result) == len(levels) and all(
                    named(inner, level.result_id) and level.mode_status == inner.mode_status
                    for inner, level in zip(result, levels)) and (
                    best is None or copied(item.evaluation, result[best - 1]))
            return (result is None) == (item.result_id is None) and (
                result is None or named(result, item.result_id) and copied(item.evaluation, result))

        if type(self.record) is not AugmentedLagrangianRecord or not isinstance(
            self.problem, ConstrainedOptimization
        ):
            raise TypeError("ConstrainedQHDResult needs its AugmentedLagrangianRecord and live problem")
        if self.problem.content_id != self.record.problem.content_id:
            raise ValueError("the live problem differs from the record's problem")
        if type(self.results) is not tuple or len(self.results) != len(self.record.iterations) or not all(
            matches(result, item) for result, item in zip(self.results, self.record.iterations)
        ):
            raise ValueError("inner results must be the ones the record names, one per round or level, with the "
                             "mode status that the record copied from them")

    @property
    def termination(self):
        """The run's stopping status, such as ``feasible_complementary``.

        ``AugmentedLagrangianRecord`` lists the statuses. A status does not assess
        optimality.
        """
        return self.record.termination

    @property
    def iterations(self):
        """Every started round, as ALIteration records."""
        return self.record.iterations

    @property
    def resources(self):
        """Totals of the rounds and the layer's preprocessing, as ALResources."""
        return self.record.resources

    @property
    def best(self):
        """The ALIteration of the reported best point, or None when no round chose a point.

        It is the feasible round with the least f, or the round with the least violation
        when no round is feasible.
        """
        return None if self.record.best is None else self.record.iterations[self.record.best]

    @property
    def last(self):
        """The ALIteration of the last round that chose a point, or None."""
        return None if self.record.last is None else self.record.iterations[self.record.last]

    @property
    def candidate(self):
        """The best point in the original coordinates, or None."""
        return None if self.best is None else self.best.evaluation.point

    @property
    def objective(self):
        """f at the best point in original units, or None."""
        return None if self.best is None else self.best.evaluation.objective

    @property
    def penalty(self):
        """rho of the last round that chose a point, or None."""
        return None if self.last is None else self.last.penalty

    def multipliers(self, *, tentative=True):
        """Return the last round's `(equality, inequality)` multipliers in original units, or None.

        The layer's multipliers belong to the scaled problem, and each is converted to
        original units, `lambda_i = (s_f/s_{h_i}) lambda~_i` and
        `mu_j = (s_f/s_{g_j}) mu~_j`, formed exactly and rounded once. A value outside the
        normal binary64 range raises ValueError. `tentative=True`, the default, converts
        lambda+ and mu+ of the last round, the run's estimates, and False the multipliers
        that round used, the initial multipliers when it is round 0. The equality tuple
        follows the problem's equalities, and the inequality tuple follows
        `record.preprocessing.inequalities`, the problem positions of the inequalities that
        the rounds use, so an inequality absorbed into the box has no entry. None when no
        round chose a point.
        """
        if self.last is None:
            return None
        item, record = self.last, self.record
        scale = record.options.objective_scale
        if tentative:
            equality = item.evaluation.tentative_equality_multipliers
            inequality = item.evaluation.tentative_inequality_multipliers
        else:
            equality, inequality = item.equality_multipliers, item.inequality_multipliers
        pre = record.preprocessing
        return (tuple(_original_multiplier(value, scale, s, f"equalities[{i}] multiplier")
                      for i, s, value in zip(pre.equalities, record.equality_scales, equality, strict=True)),
                tuple(_original_multiplier(value, scale, s, f"inequalities[{j}] multiplier")
                      for j, s, value in zip(pre.inequalities, record.inequality_scales, inequality, strict=True)))

    def _positioned_multipliers(self):
        """Return ``(multipliers, reason)``: ``multipliers()`` keyed by problem position, or why it gives none.

        ``multipliers`` is None, with ``reason`` None when no round chose a
        point and with the conversion error's text when a value lies outside
        the normal binary64 range.
        """
        try:
            values = self.multipliers()
        except ValueError as error:
            return None, str(error)
        if values is None:
            return None, None
        pre = self.record.preprocessing
        return (dict(zip(pre.equalities, values[0], strict=True)),
                dict(zip(pre.inequalities, values[1], strict=True))), None

    def _statements(self):
        """Interpretation sentences that follow from the stored record only."""
        record = self.record
        lines = [_STATUS_TEXT[record.termination]]
        if record.failure is not None:
            lines.append("Reason: " + record.failure)
        lines += self._absorbed_statements()
        bounds = record.options.multiplier_bounds
        if bounds is None:
            lines.append("No multiplier safeguard was used, so the convergence results of Birgin and Martinez "
                         "that assume bounded safeguarded multipliers do not apply.")
        else:
            lines.append(f"Multiplier safeguard used: lambda in [{bounds.equality_lower}, "
                         f"{bounds.equality_upper}], mu in [0, {bounds.inequality_upper}] (normalized units).")
        lines.append(_RULE_TEXT[record.options.inner_point])
        forms = [item.representation for item in record.iterations if item.representation is not None]
        if forms:
            lines.append(_COMPARED_TEXT)
        refinement = record.refinement
        if refinement is not None:
            gain = "" if refinement.gain is None else f", potential gain {refinement.gain:g}"
            lines.append(f"Each round refined the box of its inner objective from the preprocessed box "
                         f"({refinement.scaling}{gain}, at most {refinement.max_levels} levels) and took the level "
                         "with the least recorded relative value of that objective, the earlier level on ties. The "
                         "rule above reads every level. Each level checks the original objective and constraints "
                         "at the original-coordinate projection of every compared point. Level boxes and masses "
                         "describe each level's distribution and give no proof of a continuous or global optimum.")
            if record.termination in ("feasible_complementary", "feasible") and (
                record.iterations[-1].refinement.termination in ("budget_exhausted", "inner_failed",
                                                                 "no_valid_point")
            ):
                lines.append("The last round's refinement stopped early with "
                             f"{record.iterations[-1].refinement.termination}, and its point met the stopping test.")
        if any(form.slacks for form in forms):
            lines.append("Rounds with slack variables solve the inner objective L(x, s) of Proposition 54 of "
                         "docs/mathematics.md, whose minimum over the slacks is L_k in exact arithmetic when the "
                         "caps contain the minimizers. Each such round's point is the projection of the joint point "
                         "that the rule selected, with the joint point's probability and tie deficit, and the "
                         "multiplier update, the stopping tests and the stationarity diagnostic are those of L_k "
                         "there. The slack caps and the recorded slack-grid error bounds hold on the grid of the "
                         "preprocessed box.")
        if any(kind in ("quadratic", "constant") for form in forms for kind in form.forms):
            lines.append("A quadratic or constant inequality form equals the PHR term on the grid of the "
                         "preprocessed box, where the recorded bounds of the inequality establish it.")
        lines.append("Multipliers are finite-grid dual iterates, not continuous KKT multipliers.")
        reason = self._positioned_multipliers()[1]
        if reason is not None:
            lines.append(f"The multiplier estimates have no original-unit values, because {reason}.")
        return lines

    def _absorbed_statements(self):
        """Sentences on bound absorption: the model change and any initial multiplier that it left unused.

        Absorption (``_absorb``) replaces an inequality by a box bound, so the
        rounds solve a problem on another box, and its multiplier is gone.
        Where a grid can place a point on an absorbed bound follows from the
        grids that ``grid.OneHotGrid`` defines.
        """
        pre, record = self.record.preprocessing, self.record
        if not pre.absorbed:
            return []
        names = record.problem.variable_names
        cuts = ", ".join(f"inequalities[{item.inequality}] by {names[item.variable]} "
                         f"{'<=' if item.side == 'upper' else '>='} {item.value!r}" for item in pre.absorbed)
        model = f"Bound absorption (absorb_bounds=True) replaced inequalities by box bounds: {cuts}."
        if pre.bounds == pre.original_bounds:
            model += (f" These bounds hold everywhere on the problem's box {pre.bounds}, which the rounds searched "
                      "unchanged.")
        else:
            model += (f" The rounds searched the box {pre.bounds} instead of the problem's box "
                      f"{pre.original_bounds}, and the absorbed constraints hold everywhere on it.")
        qhd = record.qhd
        if qhd.boundary == "periodic":
            model += (" A periodic grid contains the lower edge of each side of its box and not the upper edge, so "
                      "a grid point can lie on an absorbed lower bound and none lies on an absorbed upper bound.")
        elif qhd.include_boundary_points:
            model += " The endpoint grid contains the edges of its box, so a grid point can lie on an absorbed bound."
        else:
            model += (" The Dirichlet interior grid contains no point on the edges of its box, so no grid point lies "
                      "on an absorbed bound, where its constraint is active.")
        lines = [model]
        given = record.options.inequality_multipliers
        if given is not None:
            unused = ", ".join(f"inequalities[{item.inequality}] {given[item.inequality]!r}" for item in pre.absorbed)
            lines.append(f"The initial multipliers given for the absorbed constraints were not used: {unused}.")
        return lines

    def __str__(self):
        """Summarize the run and its default coverage bounds from stored records.

        Multipliers are listed by the problem position of their constraint.
        When the best round chose a most probable grid point, its copied mode
        status (``ALEvaluation.mode_status``) follows the point. It is the
        status of the selected joint inner point before projection when that
        round has slack axes, and of its inner point otherwise. For a run with
        shots it carries the empirical wording of the inner Result
        (``records.QHDAnalysis``), since a resolved mode of observed counts
        gives no proof of the mode of the sampled population.
        """
        return "\n".join(self._summary_lines(self._confidence(DEFAULT_FAILURE_PROBABILITY)))

    def _confidence(self, failure_probability):
        """Evaluate refinement coverage with one horizon for the whole AL run."""
        alpha = validate_alpha(failure_probability)
        record = self.record
        if alpha is None or record.refinement is None:
            return None
        return coverage(((item.iteration, level) for item in record.iterations for level in item.refinement.levels),
                        k=record.qhd.num_grid_points,
                        horizon=record.options.max_iterations * record.refinement.max_levels,
                        failure_probability=alpha)

    def _summary_lines(self, confidence):
        """Build the display from stored fields and the already evaluated coverage bounds."""
        record, best, last = self.record, self.best, self.last
        lines = [f"Constrained QHD (augmented Lagrangian): {record.termination} after "
                 f"{len(record.iterations)} rounds; execution: {record.execution}"]
        if record.refinement is not None:
            lines.append("Box refinement per round (levels, stop): " + "; ".join(
                f"round {item.iteration}: {len(item.refinement.levels)}, {item.refinement.termination}"
                for item in record.iterations))
        if record.options.inequality_form != "phr":
            lines.append(f"Inequality forms ({record.options.inequality_form}) per round, by problem position: "
                         + "; ".join(f"round {item.iteration}: {self._forms(item)}" for item in record.iterations))
        if best is None:
            lines.append("No round chose a point")
        else:
            label = ("Best feasible point" if best.evaluation.infeasibility <= record.feasibility_tolerance
                     else "Least-violation point, no feasible round")
            lines.append(f"{label} {list(best.evaluation.point)} (round {best.iteration}): objective "
                         f"{best.evaluation.objective:.8g}; normalized infeasibility "
                         f"{best.evaluation.infeasibility:.3g} (tolerance {record.feasibility_tolerance:.3g})")
            status = best.evaluation.mode_status
            if status is not None:
                # A joint point only with slack axes, and the inner Result's wording and qualifier for counts
                # (records.QHDAnalysis._mode_lines), from the stored representation and shots.
                point = ("selected inner joint point" if best.representation is not None
                         and best.representation.slacks else "selected inner point")
                if status != "unavailable" and record.shots is not None:
                    lines.append(f"Empirical mode status of the {point}: {status}. Counts give no proof of the "
                                 "population mode.")
                else:
                    lines.append(f"Mode status of the {point}: {status}.")
            multipliers = self._positioned_multipliers()[0]
            estimates = ("unavailable" if multipliers is None else
                         f"equalities {multipliers[0]}, inequalities {multipliers[1]}")
            lines.append(f"Last round {last.iteration}: penalty {last.penalty:.6g}; multiplier estimates in "
                         f"original units by problem position: {estimates}")
        totals = record.resources
        reasons = dict(totals.unavailable)
        counts = {}
        for name in ("circuit_attempts", "shots", "data_bytes", "table_evaluations", "evolution_work", "layer_work"):
            value = getattr(totals, name)
            counts[name] = str(value) if value is not None else f"unknown ({reasons[name]})"
        lines.append(f"Acquisition: {counts['circuit_attempts']} circuit attempts, {counts['shots']} shots, "
                     f"{counts['data_bytes']} data bytes. Host work: {counts['table_evaluations']} table "
                     f"evaluations, {counts['evolution_work']} evolution units, {counts['layer_work']} layer units; "
                     f"the configuration admits at most {record.admitted_work_bound} host work units over the run")
        lines += self._statements()
        lines.extend(coverage_lines(confidence))
        return lines

    def _forms(self, item):
        """Name a round's inequality forms by problem position, with its slack-grid error bound when it has slacks."""
        form = item.representation
        groups = {}
        for j, kind in zip(self.record.preprocessing.inequalities, form.forms, strict=True):
            groups.setdefault(kind, []).append(j)
        text = ", ".join(f"{kind} {groups[kind]}" for kind in ("slack", "quadratic", "constant", "phr")
                         if kind in groups) or "no kept inequality"
        return text + (f" (slack-grid error bound {form.error_bound:.3g})" if form.slacks else "")

    def __repr__(self):
        return str(self)

    def report(self, *, failure_probability=DEFAULT_FAILURE_PROBABILITY):
        """Return a JSON-ready report built only from the stored record and content hashes.

        It performs no objective evaluation, planning or measurement. `multipliers` maps each constraint's
        problem position, as a string, to its original-unit multiplier estimate
        (`multipliers()`), separately for equalities and inequalities, and is None when no
        round chose a point or a value lies outside the normal binary64 range, which a
        statement then names. `inner_results` holds one content hash per round, or with
        refinement one list of level hashes per round, and `refinement` the number of levels
        and the stopping reason of each round's refinement, None without refinement.
        `inequality_forms` holds, per round, the form of each kept inequality by problem
        position, the round's slack-grid error bound and why the policy chose the forms
        (`InnerRepresentation`), and is None under `inequality_form="phr"`.

        With counts and box refinement, `confidence` contains selected region mass lower
        bounds for each level's backend-sampled distribution conditioned on valid decoding,
        with simultaneous 95% confidence by default. The horizon is
        `max_iterations * max_levels` for the whole run, including early termination.
        Hoeffding and one-sided Clopper–Pearson bounds each use half the failure budget;
        each level reports the larger available bound. The dimension includes its slack
        coordinates. The sampling assumptions and binary64 evaluation are those of
        [Proposition 49](../../mathematics.md#r49). Exact readout, no refinement, no completed
        counts levels, or explicit `None` gives `confidence=None`.

        Args:
            failure_probability (float | None): Default `0.05`. Total failure probability
                alpha, strictly between 0 and 1 and fixed before inspecting the counts.
                Set to `None` to omit the statistical report.

        Returns:
            report (dict): JSON-ready summary, interpretation statements, stored record,
                multipliers, inner Result hashes, refinement details, coverage bounds and
                inequality forms.

        Raises:
            ValueError: If `failure_probability` is neither `None` nor a number strictly
                between 0 and 1.

        Examples:
            `result.report()` includes the default bounds.
            `result.report(failure_probability=0.01)` uses a total failure probability of 0.01.
        """
        confidence = self._confidence(failure_probability)
        multipliers = self._positioned_multipliers()[0]
        refined = self.record.refinement is not None
        statements = self._statements()
        if confidence is not None:
            statements.append(confidence["meaning"])
        return dict(summary="\n".join(self._summary_lines(confidence)), statements=statements, confidence=confidence,
                    record=self.record.model_dump(mode="json"),
                    multipliers=None if multipliers is None else dict(
                        equalities={str(i): value for i, value in multipliers[0].items()},
                        inequalities={str(j): value for j, value in multipliers[1].items()}),
                    inner_results=[[inner.content_id for inner in result] if refined
                                   else None if result is None else result.content_id for result in self.results],
                    refinement=None if not refined else [
                        dict(levels=len(item.refinement.levels), termination=item.refinement.termination)
                        for item in self.record.iterations],
                    inequality_forms=None if self.record.options.inequality_form == "phr" else [
                        dict(forms={str(j): kind for j, kind in zip(self.record.preprocessing.inequalities,
                                                                    item.representation.forms, strict=True)},
                             slack_error_bound=item.representation.error_bound,
                             selection=item.representation.selection)
                        for item in self.record.iterations])

    def save(self, path):
        """Write the run to a new directory and return its path.

        The directory holds `constrained.json` (format `qhd.constrained/4`) with the record,
        `problem.pickle` with the live SymPy objective, variables and constraints, and
        `iterations/<k>/result/`, each inner Result saved by its own archive. With
        refinement each completed level z of round k has its Result under
        `iterations/<k>/levels/<z>/result/` instead, and the record nests the refinement
        records. One format, `qhd.constrained/4`, covers both shapes, because the record
        itself says which shape a folder has. `load_augmented_lagrangian` reads it back. The
        archive stores the problem itself, because no Plan holds a ConstrainedOptimization.
        A failed save removes the directory.

        Args:
            path (str | Path): A directory that does not exist yet. Its parent must exist.

        Returns:
            path (Path): The written directory.
        """
        from nwqlib._choice_archive import ArchiveFiles
        from nwqlib._limits import DEFAULT_MAX_BYTES

        files = ArchiveFiles(Path(path), DEFAULT_MAX_BYTES)
        files.path.mkdir()
        try:
            problem = self.problem
            with files.writer("problem.pickle") as stream:
                pickle.dump((problem.objective, problem.variables, problem.equalities, problem.inequalities),
                            stream, protocol=5)
            files.write_json("constrained.json", dict(format=ARCHIVE_FORMAT, problem="problem.pickle",
                                                      record=self.record.model_dump(mode="json")))
            for item, result in zip(self.record.iterations, self.results, strict=True):
                folder = files.path / "iterations" / str(item.iteration)
                if item.refinement is not None:
                    for level, inner in zip(item.refinement.levels, result, strict=True):
                        (folder / "levels" / str(level.level)).mkdir(parents=True)
                        inner.save(folder / "levels" / str(level.level) / "result")
                elif result is not None:
                    folder.mkdir(parents=True)
                    result.save(folder / "result")
        except BaseException:
            shutil.rmtree(files.path)
            raise
        return files.path


def load_augmented_lagrangian(path):
    """Load a saved augmented-Lagrangian run without planning, evaluating or measuring.

    `load_augmented_lagrangian(path)` reads the directory that
    `ConstrainedQHDResult.save` wrote and returns the `ConstrainedQHDResult`. The record
    is validated with its content hashes. The SymPy objects come back through a reader
    that resolves SymPy classes only, and the rebuilt problem must have the record's
    problem hash. Each inner Result is loaded with `load_result` and must have the
    recorded result and Plan hashes, and its Plan's objective the recorded hash of L_k,
    or of the inner objective of the round's representation over the problem's variables
    and the slack variables. With refinement each level's Result must have the result
    and Plan hashes of its `RefinementLevel`. A level Plan solves the level's own
    objective, for the search model the normalized one on the unit box, so it is checked
    by these hashes and not against the hash of L_k. A saved standalone refinement is
    refused with the name of `load_box_refinement`, and a saved-run directory of
    `solve_augmented_lagrangian(..., directory=...)` with the name of
    `resume_augmented_lagrangian`, which continues it.

    Args:
        path (str | Path): The directory that `ConstrainedQHDResult.save` wrote.

    Returns:
        result (ConstrainedQHDResult): The saved run, with its problem and inner Results
            attached.
    """
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.scientist import load_result
    from .archive import _SymbolicReader

    files = ArchiveFiles(Path(path), None)
    try:
        saved = files.read_json("constrained.json")
    except FileNotFoundError as error:
        if files.file(REFINEMENT_RECORD).is_file():
            raise FileNotFoundError(
                f"{path} is a saved box refinement containing {REFINEMENT_RECORD}, not a saved augmented-Lagrangian "
                f"result containing constrained.json. Open it with load_box_refinement({str(path)!r})"
            ) from error
        if files.file(RECORD).is_file():
            # Name the resume function of the directory's layer.
            if files.read_json(RECORD).get("format") == FORMATS["refinement"]:
                raise FileNotFoundError(
                    f"{path} is a durable box-refinement directory containing controller.json, not a saved "
                    f"augmented-Lagrangian result containing constrained.json. Use "
                    f"{RESUME['refinement']}({str(path)!r}, backend=...) to resume it. An interrupted inner Run "
                    "may be unable to finish from its recorded work. Save the returned BoxRefinementResult to a "
                    "new result archive and open that archive with load_box_refinement."
                ) from error
            raise FileNotFoundError(
                f"{path} is a durable run directory containing controller.json, not a saved result containing "
                f"constrained.json. Use resume_augmented_lagrangian({str(path)!r}, backend=...) to resume it, "
                "then call save on a returned result to create a saved archive. An interrupted inner Run "
                "may be unable to finish from its recorded work"
            ) from error
        raise
    if saved.get("format") != ARCHIVE_FORMAT:
        raise ValueError(f"unsupported constrained QHD archive format {saved.get('format')!r}, expected "
                         f"{ARCHIVE_FORMAT!r}")
    record = AugmentedLagrangianRecord.model_validate(saved["record"])
    with files.read_path(saved["problem"]).open("rb") as stream:
        problem = live_problem(record.problem, _SymbolicReader(stream).load())
    results = []
    for item in record.iterations:
        folder = files.path / "iterations" / str(item.iteration)
        if item.refinement is not None:
            levels = []
            for level in item.refinement.levels:
                result = load_result(folder / "levels" / str(level.level) / "result")
                if (result.content_id, result.plan.content_id) != (level.result_id, level.plan_id):
                    raise ValueError(f"saved result of level {level.level} of round {item.iteration} differs "
                                     "from the record")
                levels.append(result)
            results.append(tuple(levels))
            continue
        if item.result_id is None:
            results.append(None)
            continue
        result = load_result(folder / "result")
        if not _belongs(problem, item, result):
            raise ValueError(f"saved inner result of round {item.iteration} differs from the record")
        results.append(result)
    return ConstrainedQHDResult(record, problem, tuple(results))


def _belongs(problem, item, result):
    """Whether the inner ``result`` of an unrefined round is the one that its ALIteration ``item`` records.

    The Result and its Plan must have the recorded identities, and the Plan
    must solve the recorded inner objective, L_k under the PHR policy and
    the representation's inner objective otherwise, over the problem's
    variables and the representation's slack variables (``_solves``).
    """
    form = item.representation
    return (result.content_id, result.plan.content_id) == (item.result_id, item.plan_id) and _solves(
        problem, result.plan, item.effective_objective if form is None else form.inner_objective, form)


def _run_results(path, iterations, backend, problem):
    """Read the inner Results of the recorded rounds of a durable run from their Runs (``_durable.saved_result``).

    One Result per round, None for a round without one, or with refinement
    the tuple of the Results of its completed levels, as
    ``ConstrainedQHDResult.results`` holds them. The Result of an unrefined
    round must belong to its recorded Plan and inner objective
    (``_belongs``), and ``ConstrainedQHDResult`` checks the identities of
    level Results.
    """
    results = []
    for item in iterations:
        folder = Path(path) / "iterations" / str(item.iteration)
        if item.refinement is not None:
            results.append(tuple(saved_result(folder / "levels" / str(level.level) / "run", backend=backend)
                                 for level in item.refinement.levels))
        elif item.result_id is None:
            results.append(None)
        else:
            result = saved_result(folder / "run", backend=backend)
            if not _belongs(problem, item, result):
                raise ValueError(f"the Run of round {item.iteration} in {path} holds another inner result than "
                                 "the durable record")
            results.append(result)
    return tuple(results)


def resume_augmented_lagrangian(directory, *, backend, progress=None, end_at_unfinishable=False):
    """Continue an interrupted saved run of `solve_augmented_lagrangian`, or return it when it has ended.

    Pass the directory and the backend of the original call. The directory's outer
    record holds the run's problem, QHD configuration, options, refinement options,
    execution, shots, root entropy, cumulative limits and backend configuration, and its
    completed rounds and levels. The layer wrote every file of the directory, and resume
    reads them as they are. A directory is read-only like a saved Run folder
    ([Saved folders are read-only](../../saved_evidence.md#saved-folders-are-read-only)),
    and resume does not detect edits. Before any new work, `backend` must have the stored
    configuration,
    and for a noisy Aer run resume binds the noise model saved in the directory to its
    own copy of `backend`, so that the Runs it creates run the original model in a new
    process too. Another backend raises ValueError before anything is reopened, planned
    or committed. The problem rebuilt from `problem.pickle` must have the content hash
    of the stored problem record, or ValueError is raised before any completed Result is
    read or the run advances, since an expression that the reader cannot reproduce would
    otherwise continue the run as another problem. The completed rounds' and levels'
    Results are read from their Runs with `load_run`, and the layer's preprocessing and
    check counts are read from the setup that the first call saved, without recomputing
    them to detect an edit.

    The rounds then continue from the last completed round, with the state and the
    random stream of the uninterrupted run. A round in progress reuses the inequality
    representation that it committed before its first inner Run, after checking that it
    rebuilds the committed inner objective, so automatic selection is not repeated, and
    a completed round's Result must belong to its recorded inner Plan. A round or level
    whose Run exists continues that Run with `load_run(...).wait()`, and none gets a
    second Run, so completed work is read from the records and a continued Run is
    counted once, by its own run log. When that Run cannot finish without new work,
    because the interruption stopped a local preparation, measurement or classical
    evolution whose outcome nothing can retrieve, its error propagates with a note, and
    the outer record keeps its committed rounds and levels unchanged. With
    `end_at_unfinishable=True` the run instead ends there with `inner_failed`, keeps its
    completed rounds and records the Run's error as the failure. This option handles an
    unfinishable continuation after a Run has reopened. A Run folder without a committed
    run-log header fails during reopen and is not handled by `end_at_unfinishable`.
    Reopen does not remove or recreate that folder automatically. Before a completed
    round or level exists, an inner failure still propagates. Resume covers an
    interrupted process, not a power loss.

    A directory whose run has ended returns its result, with the inner Results read from
    their Runs, and plans, evaluates and measures nothing.

    Resuming gives the records of an uninterrupted saved run with the same root entropy,
    apart from the fields that differ between any two saved executions, namely the
    identifiers of the Runs and Results, which every Run draws anew, the stored-data
    byte counts, whose text of wall times and measured timings varies in length from run
    to run, and the content hashes that include these fields. Those byte counts enter
    the remainder of `max_data_bytes`, so a run whose data limit is almost spent can
    stop at another round, as two uninterrupted saved runs can.

    Args:
        directory (str | Path): The directory of
            `solve_augmented_lagrangian(..., directory=...)`.
        backend (object): The backend of the original call, whose configuration the
            directory stores. None stands for `AerBackend()` under quantum execution, as
            in the original call. For a noisy Aer run in a new process,
            `AerBackend(noise_model_id=...)` with the identifier of the original
            binding, which the error for another backend names.
        progress (object | None): Progress callback of the inner Runs that this call
            creates or continues, as in `nwqlib.solve`.
        end_at_unfinishable (bool): Default `False`, which raises the Run's error and
            leaves the directory as it is for a later resume. True ends the run with
            `inner_failed` at a round or level whose Run cannot finish without new work.
            It handles an unfinishable continuation after a Run has reopened, not a Run
            folder without a committed run-log header, which fails during reopen. Before
            a completed round or level exists, an inner failure still propagates.

    Returns:
        result (ConstrainedQHDResult): The run, as `solve_augmented_lagrangian` returns
            it.

    Raises:
        TypeError: If `end_at_unfinishable` is not a bool.
        ValueError: If `backend` differs from the stored configuration, or the rebuilt
            problem differs from the stored record.
    """
    import numpy as np
    from nwqlib.execution import ExecutionLimits
    from .method import QHD

    if type(end_at_unfinishable) is not bool:
        raise TypeError("end_at_unfinishable must be a bool")
    with Directory.open(directory, "constrained") as outer:
        outer.end_at_unfinishable = end_at_unfinishable
        settings, saved = outer.settings, outer.saved
        backend = outer.bind(backend)
        stored = ConstrainedOptimization.model_validate(settings["problem"])
        problem = live_problem(stored, outer.problem())
        if problem.content_id != stored.content_id:
            raise ValueError("the live problem differs from the record's problem")
        if saved["record"] is not None:
            record = AugmentedLagrangianRecord.model_validate(saved["record"])
            return ConstrainedQHDResult(record, problem,
                                        _run_results(outer.path, record.iterations, backend, problem))
        qhd, options = QHD.model_validate(settings["qhd"]), AugmentedLagrangian.model_validate(settings["options"])
        refinement = None if settings["refinement"] is None else BoxRefinement.model_validate(settings["refinement"])
        iterations = tuple(ALIteration.model_validate(item) for item in saved["iterations"])
        setup = _setup(problem, qhd, options, refinement, stored=settings["setup"])
        results = _run_results(outer.path, iterations, backend, problem)
        k = len(iterations)
        levels = tuple(RefinementLevel.model_validate(level) for level in saved["levels"])
        level_results = tuple(
            saved_result(outer.path / "iterations" / str(k) / "levels" / str(level.level) / "run", backend=backend)
            for level in levels)
        blocked = None if saved["blocked"] is None else tuple(saved["blocked"])
        # The representation that the round in progress committed before its first inner Run.
        form = None if saved["representation"] is None else InnerRepresentation.model_validate(saved["representation"])
        return _rounds(problem, qhd, setup, refinement, backend, settings["execution"], settings["shots"],
                       np.random.SeedSequence(settings["entropy"]), ExecutionLimits.model_validate(settings["limits"]),
                       progress, outer=outer, iterations=iterations, results=results,
                       pending=(levels, level_results, blocked), representation=form)


def constrained_grid_minimum(result, *, max_work=GRID_MINIMUM_MAX_WORK, max_bytes=GRID_MINIMUM_MAX_BYTES):
    """Find the least evaluated feasible objective on the run's grid and its gap to the best point.

    `constrained_grid_minimum(result)` evaluates f, h and g in binary64 at every point
    of the preprocessed box's grid and returns a
    [`ConstrainedGridMinimum`][nwqlib.algorithms.qhd.constrained_records.ConstrainedGridMinimum].
    Its `gap` is the run's stored best objective minus the least evaluated feasible
    objective, in original units. The solver never calls it.

    The explicit grid reference evaluates the original objective and required
    constraints on the declared original-variable grid. It chooses the lexicographically
    first point attaining the least computed feasible objective. Feasibility uses the
    normalized residual test at the stored tolerance. All reported reference values come
    from the same evaluated table. The result describes this finite numerical grid and
    supplies neither a continuous optimum nor an expression-evaluation error
    certificate.

    The reference evaluates the original functions in C-order slabs, streams constraint
    values into the feasibility test and keeps the first feasible minimum. Its work
    counts every evaluated function and its byte allowance includes expression
    intermediates, coordinate arrays and masked-reduction buffers.

    Selection. Given a finite reference objective array f and a Boolean feasible mask,
    ``argmin(where(feasible, f, inf))`` returns the first feasible minimum in C order,
    which is the lexicographically first tuple. For each original equality value h and
    inequality value g, the stored positive scale is applied before comparing with the
    tolerance: the violation is ``max(max(abs(h/s_h)), max(max(g/s_g, 0)))``, with an
    empty maximum zero. All raw values must be finite and real, even at infeasible
    points. Division may overflow if a scale is tiny, which gives infinite infeasibility
    and rejects that point as infeasible. Slabs are processed in increasing global C
    order and the running minimum is updated only on a strictly smaller value, which
    keeps the first tie across slab boundaries. Counts of feasible points use Python
    integers across slabs. Vectorized expression evaluation may change the reference
    table's entries relative to scalar evaluation and can change a minimum or
    feasibility decision near a threshold, so this is one consistent array-evaluated
    reference table, and re-evaluating only the chosen point scalarly would not restore
    a scalar ranking.

    The signed gap compares a stored objective with a fresh grid minimum and can be
    negative. Let G be the nonempty grid subset passing this check's computed
    feasibility test, and let the chosen point a lie in G. Suppose a finite E >= 0
    bounds the absolute error of the stored objective at a and of every fresh objective
    evaluation on G, relative to exact values at those same coordinates. For finite
    reported gap r and u = 2**-53, the exact gap `F(a) - min_G F` is at most
    `r/(1-u) + 2*E` when r >= 0, and `r/(1+u) + 2*E` when r < 0. These are
    real-arithmetic inequalities under round-to-nearest binary64 with gradual underflow.
    This function establishes neither E nor equality of G with the mathematically
    feasible grid set.

    The reference enumerates the original variables only, also for a run whose rounds
    added slack variables, and applies the original feasibility test. It evaluates f, h
    and g in their written form. This can raise ValueError at a grid point where the
    original binary64 evaluation fails, for example through intermediate overflow on a
    term whose initial check used only its support tables. Neither this reference nor
    the initial check certifies the exact real domain or bounds expression-evaluation
    error.

    A run with box refinement is refused. Its rounds search nested grids that each run
    chooses from its own distributions, and the preprocessed box's grid is only the
    first of them, so a feasible minimum of that grid is not the finite-grid reference
    of the points the run compared, and the gap would mix points of different grids.

    Before the first evaluation, the reference checks its work and slab bytes against
    `max_work` and `max_bytes` by the laws in
    [Engineering constants](../../ENGINEERING_CONSTANTS.md#qhd-augmented-lagrangian-defaults).
    It keeps no reference table of all `K**d` grid points.

    Args:
        result (ConstrainedQHDResult): A run from ``solve_augmented_lagrangian`` or
            ``load_augmented_lagrangian``.
        max_work (int): Work limit of this reference. Default `1_000_000_000`.
        max_bytes (int): Byte limit of this reference. Default 10 GB (decimal,
            `10_000_000_000` bytes).

    Returns:
        reference (ConstrainedGridMinimum): The least evaluated feasible value in
            `objective`, its point, the signed gap in `gap` and the checked work.


    Raises:
        TypeError: If `result` is not a `ConstrainedQHDResult`.
        ValueError: If the run used box refinement, if the work or the bytes of a
            one-point slab exceed `max_work` or `max_bytes`, or if the original binary64
            evaluation fails at a grid point.
    """
    import numpy as np

    from .compiler import H0, evaluate_chunk
    from .validation import _lambdify_objective

    if type(result) is not ConstrainedQHDResult:
        raise TypeError("constrained_grid_minimum requires a ConstrainedQHDResult")
    record, problem = result.record, result.problem
    if record.refinement is not None:
        raise ValueError("constrained_grid_minimum is the reference of the preprocessed box's grid, and a run with "
                         "box refinement searches the nested grids of its levels")
    grid = OneHotGrid(problem.variable_names, record.preprocessing.bounds, record.qhd.num_grid_points,
                      record.qhd.include_boundary_points, record.qhd.boundary)
    d, k = grid.num_variables, grid.num_grid_points
    named = [("objective", problem.objective, record.options.objective_scale)]
    named += [(f"equalities[{i}]", problem.equalities[i], s)
              for i, s in zip(record.preprocessing.equalities, record.equality_scales, strict=True)]
    named += [(f"inequalities[{j}]", problem.inequalities[j], s)
              for j, s in zip(record.preprocessing.inequalities, record.inequality_scales, strict=True)]
    nodes = [node_count(expression, max_work) for _, expression, _ in named]
    work_is_lower_bound = any(n > max_work for n in nodes)
    points = k**d
    # With M = max_work and complete tree sizes N_i, node_count returns
    # n_i = min(N_i, M+1). n_i <= M proves completion. Otherwise D*n_i > M
    # already proves refusal, and the displayed charge is only a lower bound.
    work = points * (d + sum(nodes) + 8 * (len(named) - 1) + 12)
    fixed = 8 * d * k + H0
    rate = max(8 * (n + 2) for n in nodes) + 8 * d + 96
    if work > max_work or fixed + rate > max_bytes:
        work_requirement = f"at least {work}" if work_is_lower_bound else str(work)
        message = (
            f"constrained_grid_minimum needs {work_requirement} work units and at least {fixed + rate} bytes "
            f"for a one-point slab, above max_work={max_work} or max_bytes={max_bytes}"
        )
        if work_is_lower_bound:
            message += (
                ". At least one expression node count stopped at max_work. "
                "The displayed work is a lower bound on the full admission charge. "
                f"Raising max_work to {work} can still lead to another work refusal because "
                "counting further can increase the required work"
            )
        raise ValueError(message)
    slab = min(points, (max_bytes - fixed) // rate)
    terms = [_Term(name, expression, scale, _lambdify_objective(problem.variables, expression), (), n, None)
             for (name, expression, scale), n in zip(named, nodes, strict=True)]
    count = len(record.equality_scales)
    tolerance = record.feasibility_tolerance
    axes = [grid.coordinate_array(j) for j in range(d)]
    support = tuple(range(d))
    objective, values, violation = np.empty(slab), np.empty(slab), np.empty(slab)
    feasible, minimum = 0, None
    for start in range(0, points, slab):
        stop = min(points, start + slab)
        n = stop - start
        f = evaluate_chunk(terms[0].function, axes, start, stop, objective[:n], expression=terms[0].expression,
                           support=support, name=terms[0].name)
        worst = violation[:n]
        worst.fill(0.0)
        for position, term in enumerate(terms[1:]):
            value = evaluate_chunk(term.function, axes, start, stop, values[:n], expression=term.expression,
                                   support=support, name=term.name)
            value /= term.scale
            if position < count:
                np.abs(value, out=value)
            else:
                np.maximum(value, 0.0, out=value)
            np.maximum(worst, value, out=worst)
        passing = worst <= tolerance
        found = int(np.count_nonzero(passing))
        if found:
            feasible += found
            local = int(np.argmin(np.where(passing, f, np.inf)))
            if minimum is None or f[local] < minimum[0]:
                minimum = (float(f[local]), start + local)
    if minimum is not None:
        indices = tuple(int(i) for i in np.unravel_index(minimum[1], (k,) * d))
        minimum = (minimum[0], indices, tuple(float(axes[j][i]) for j, i in enumerate(indices)))
    best = result.best
    best_objective = None if best is None else best.evaluation.objective
    return ConstrainedGridMinimum(
        record_id=record.content_id, grid_points=points, feasible_points=feasible,
        indices=None if minimum is None else minimum[1], point=None if minimum is None else minimum[2],
        objective=None if minimum is None else minimum[0], best_objective=best_objective,
        best_feasible=None if best is None else best.evaluation.infeasibility <= tolerance,
        gap=None if minimum is None or best is None else best_objective - minimum[0],
        feasibility_tolerance=tolerance, work=work,
    )
