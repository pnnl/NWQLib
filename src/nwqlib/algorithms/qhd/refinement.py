"""Box refinement: repeated QHD solves on shrinking boxes, Wu et al. arXiv:2605.12066v1, Sec. V.

Each level runs one ordinary QHD Plan with ``prepare`` and ``submit`` and keeps,
on every axis, the index interval that holds most of the level's marginal
probability (Eqs. (12)–(13) of that section). The cells of the kept grid
points form the next level's box. The search model of ``_level_problem``
defines what each level solves. The distribution and interval boxes use
the level's own grid coordinates, with the Method's boundary. A grid-point
rule selects one of those coordinates. ``mode_or_mean`` can instead select
an off-grid conditional mean. A search-model level evaluates the transformed
original objective at its unit-coordinate point and reports the affine image
rounded to binary64. Evaluating F again at that reported point can give a
different value or fail where the unrounded image was admissible. When the
box stops changing, the optional stall split of ``_stall_split`` divides it between two separated
peaks of one axis and continues in one part. With ``directory`` every level
Run is durable, and ``resume_box_refinement`` continues an interrupted
refinement (``_durable``).

The docstrings below quote measured refinements as examples of what an
option does. They were measured on 2026-09-26 at revision
aede9fd119563d63562c9afed4b00021920560c6 with Python 3.12.14, NumPy 2.5.2,
SciPy 1.18.1 and SymPy 1.14.0 on macOS arm64, and the quantum example with
Qiskit 2.5.2 and Aer 0.17.2. Each example runs ``refine_box`` with
``execution="classical"`` and ``seed=7``, except the one with exact
quantum readout, which runs with ``seed=3``, and keeps every QHD and
``BoxRefinement`` setting it does not name at its default. The defaults
that matter here are the Dirichlet interior grid,
``QuadraticSchedule(gamma=0.3)`` with the midpoint coefficient rule and,
for classical execution, the ``schrodinger`` flavor. An error or distance
is the infinity-norm distance of the best point from the continuous
minimizer.
"""

from fractions import Fraction
import itertools
import math
from math import fsum, inf, ulp
from pathlib import Path
import pickle
import shutil

import sympy as sp

from nwqlib.problems.records import Optimization
from ._outer import (
    USED_COUNTS,
    check_arguments,
    law_count,
    grid_point,
    inner_failure,
    mean_point,
    plan_round,
    range_bound,
    round_limits,
    run_counts,
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
    saved_result,
    unrecoverable,
)
from .binary import _integer_storage, _require_bytes
from .grid import OneHotGrid
from .refinement_records import (
    BoxRefinement,
    BoxRefinementResult,
    RefinementLevel,
    RefinementResources,
    StallSplit,
    total_resources,
)

# The saved standalone refinement (BoxRefinementResult.save). The format
# changes with the fields of the record, and another format is refused rather
# than converted, since this unreleased package keeps no development-schema
# compatibility (docs/FRAMEWORK.md, "Package Import Surface").
ARCHIVE_FORMAT = "qhd.refinement/3"
ARCHIVE_RECORD = "refinement.json"


def _level_grid(problem, box, qhd, search):
    """Return the one-hot grid of a level's QHD Plan, the grid that ``method._grid`` rebuilds from it.

    A physical level plans the original variables on ``box``. A search-model
    level plans the unit variables ``u_<name>`` on ``[0, 1]^d``
    (``_level_problem``). Both use the Method's number of points, boundary
    points and boundary. ``grid.OneHotGrid`` admits the box or raises
    ValueError. The unit box always passes, and a physical box fails when
    its spacing or kinetic coefficient leaves the normal binary64 range.
    """
    names = tuple(f"u_{name}" for name in problem.variable_names) if search else problem.variable_names
    bounds = ((0.0, 1.0),) * len(box) if search else box
    return OneHotGrid(names, bounds, qhd.num_grid_points, qhd.include_boundary_points, qhd.boundary)


def _to_box(side, u):
    """Return ``a + (b - a) u`` for the box side ``(a, b)``, rounded once from the exact binary64 values."""
    a = Fraction(side[0])
    return float(a + (Fraction(side[1]) - a) * Fraction(u))


def _unit_symbols(problem):
    """Return the unit variables ``u_<name>`` of a search-model level of ``problem``."""
    return tuple(sp.Symbol(f"u_{name}", real=True) for name in problem.variable_names)


def _unit_objective(problem, box):
    """Return ``(unit, F(a + D u))``: the unit variables and the objective of ``problem`` over the unit box of ``box``.

    ``x_j = a_j + (b_j - a_j) u_j`` maps ``u in [0, 1]^d`` onto the box. The
    substitution uses the exact rationals of the binary64 bounds, so when
    the objective's coefficients are exact, the expression's exact value at
    a binary64 unit point u is F at the exact point ``a + D u``, whether or
    not that point is a binary64 number. A binary64 ``Float`` coefficient
    times a rational becomes a ``Float`` again, rounded by SymPy to 53 bits
    during the substitution, so for such an objective the expression is F
    up to that rounding. The search model tabulates this expression
    (``_level_problem``), and ``refine_box`` reads its tables at a reported
    grid point and evaluates it at an off-grid unit point
    (``_tabulated_objective``).
    """
    unit = _unit_symbols(problem)
    shift = {
        x: sp.Rational(a) + (sp.Rational(b) - sp.Rational(a)) * u
        for x, (a, b), u in zip(problem.variables, box, unit, strict=True)
    }
    return unit, problem.objective.xreplace(shift)


def _problem_coordinates(problem, box):
    """Return ``show(expression, support, point)``: a unit-box expression and point in the problem's coordinates.

    ``show`` inverts the substitution of ``_unit_objective`` for a refusal
    message. It replaces each unit variable ``u_j`` of ``expression`` by
    ``(x_j - a_j)/(b_j - a_j)`` with the exact rationals of the bounds, and
    maps the unit coordinates ``point``, those of the variables in
    ``support``, to ``a_j + (b_j - a_j) u_j`` (``_to_box``). Both are formed
    only when an entry or a point is refused.
    """
    unit = _unit_symbols(problem)
    inverse = {
        u: (x - sp.Rational(a)) / (sp.Rational(b) - sp.Rational(a))
        for x, (a, b), u in zip(problem.variables, box, unit, strict=True)
    }

    def show(expression, support, point):
        return (expression.xreplace(inverse),
                tuple(_to_box(box[j], u) for j, u in zip(support, point, strict=True)))

    return show


def _exact_constant(constant):
    """Return the SymPy constant ``constant`` with each binary64 ``Float`` replaced by its exact rational value.

    SymPy adds a ``Float`` to a rational in 53-bit arithmetic, which rounds,
    so a sum or difference of constants is exact only after this conversion.
    """
    return constant.xreplace({f: sp.Rational(f) for f in constant.atoms(sp.Float)})


def _level_decomposition(problem, box, search):
    """Return the ``objective.ObjectiveDecomposer`` of the objective that a level on ``box`` tabulates.

    A physical level tabulates F in the original variables on ``box``, and a
    search-model level ``F(a + D u)`` in the unit variables on ``[0, 1]^d``
    (``_unit_objective``). Either is expanded about the centers
    ``objective.expansion_centers`` of its bounds, as
    ``QHD._admit_symbolic_work`` builds it.
    """
    from .objective import ObjectiveDecomposer, expansion_centers

    if search:
        variables, objective = _unit_objective(problem, box)
        bounds = ((0.0, 1.0),) * len(variables)
    else:
        variables, objective, bounds = problem.variables, problem.objective, box
    return ObjectiveDecomposer(objective, variables, expansion_centers(bounds))


def _tabulated_objective(decomposer, offset, tables, k, show=None):
    """Return a function that evaluates a level's objective at a point as QHD tabulates it, and minus ``offset``.

    ``decomposer`` is the level's decomposition (``_level_decomposition``,
    or for the search model the one that ``_support_tables`` admitted),
    ``tables`` the level's unscaled ``SupportValues``, those of
    ``_support_tables`` for the search model and of the level Plan for the
    physical model, paired with the decomposition's support expressions by
    their supports, and ``k`` the level grid's K. QHD writes the objective as
    ``c0 + sum_S p_S`` expanded about the centers m of that decomposition.
    Planning evaluates each support expression p_S at the centered
    coordinates ``x - m`` of every grid tuple into a stored float64 table
    (``compiler.QHDCompiler._objective_grid_values``), ``QHD.plan`` stores
    the constant as ``coerce_real_scalar(c0)``, and ``method.objective_at``
    combines the constant and the table entries with ``fsum``. The returned
    function ``evaluate(point, indices=None)`` returns ``(value, relative)``,
    with ``value`` the objective and ``relative`` the objective minus
    ``offset`` (below), both checked to be finite and real.

    At a known level-grid index, read the stored entries of that level's
    unscaled objective and combine them with its constant using ``fsum``.
    The relative value uses the same entries with the exactly formed
    constant difference ``c0-C``. At an off-grid point, evaluate the level's
    support expressions at its centered coordinates. Array and scalar
    expression evaluation need not produce identical binary64 entries, so
    only an indexed read establishes agreement with the selected grid table.

    With ``indices``, the grid-index tuple of the level point, the value is
    therefore ``method.objective_at`` of the stored tables. Without it, as
    at a ``mode_or_mean`` mean position, each p_S is lambdified over its own
    variables and evaluated separately at ``x - m`` for the point x
    (``potential._evaluate_objective``), and the terms, which sum to F
    exactly, are combined with the separately converted constant, so the
    value is F up to the evaluation of each term and the rounding of the
    sum. Combining the support expressions into one expression before
    evaluating it would round differently, and can overflow in a partial
    sum when every term and the total are finite, as for
    ``10**308 (x - x**2 - x y) - y`` at ``(1, 1)``. Evaluating F in another
    written form, such as unexpanded, can also meet a pole that the tables
    do not. An evaluation error of a term raises ValueError, and an ``fsum``
    whose exact sum exceeds binary64 raises OverflowError. The decomposition
    expands the objective once more, the symbolic work that QHD admitted
    for the level's planning.

    ``relative`` is the objective minus the exact constant C = ``offset``.
    It sums the same term values with ``fsum``, together with the constant
    term ``c0 - C`` in place of c0, formed exactly from the exact rationals
    of both (``_exact_constant``) and converted to binary64 once.
    ``_refine`` passes one C for the whole refinement and compares points by
    ``relative``. When C holds a large constant of the objective, ``c0 - C``
    is of the size of the objective's variation, which then survives the
    sum, while ``value`` rounds that variation to the spacing of binary64
    numbers near the constant. ``(x - 1/3)**2 + 2**54`` lies between
    ``2**54`` and ``2**54 + 4/9`` on ``[0, 1]``, where binary64 numbers are
    4 apart, so every ``value`` is ``2**54``. This needs exact constant
    terms. A binary64 ``Float`` coefficient anywhere in the objective makes
    SymPy form c0 in 53-bit arithmetic during the expansion, so next to a
    large constant c0 has lost the variation before the subtraction, and
    ``relative`` loses it as ``value`` does. Writing such coefficients as
    exact rationals, for example with ``sympy.Rational``, which keeps the
    exact binary64 value, avoids this.

    A refused off-grid evaluation names the problem's own expression and
    point. For the search model ``show`` (``_problem_coordinates``) maps the
    unit expression and point back to the problem's coordinates.

    Raises:
        ValueError: A table does not have the ``k**|S|`` entries of its
            support.
    """
    from .objective import uncentered_objective
    from .potential import _evaluate_objective, coerce_real_scalar
    from .validation import _lambdify_objective

    variables, centers = decomposer.variables, decomposer.centers
    constant = coerce_real_scalar(decomposer.constant, context="constant objective")
    difference = coerce_real_scalar(_exact_constant(decomposer.constant) - offset,
                                    context="constant objective relative to the refinement's offset")
    # The stored tables are paired with the decomposition by support, not by position.
    by_support = {tuple(table.support): table.values.array for table in tables}
    if set(by_support) != set(decomposer.support_expressions) or len(by_support) != len(tables):
        raise ValueError("a refinement level's support tables differ from its decomposition and grid")
    stored = [(support, by_support[support]) for support in decomposer.support_expressions]
    if any(table.shape != (k ** len(support),) for support, table in stored):
        raise ValueError("a refinement level's support tables differ from its decomposition and grid")
    terms = [
        (support, expression, _lambdify_objective([variables[j] for j in support], expression))
        for support, expression in decomposer.support_expressions.items()
    ]

    def evaluate(point, indices=None):
        if indices is None:
            centered = tuple(x - m for x, m in zip(point, centers, strict=True))
            values = []
            for support, expression, function in terms:
                try:
                    values.append(_evaluate_objective(function, tuple(centered[j] for j in support),
                                                      expression=expression, support=support))
                except ValueError as error:
                    # Report the problem's own expression and point, not the centered ones.
                    shown = uncentered_objective(expression, [variables[j] for j in support],
                                                 [centers[j] for j in support])
                    at = tuple(point[j] for j in support)
                    if show is not None:
                        shown, at = show(shown, support, at)
                    raise ValueError(f"QHD objective {shown} on support {support} at point "
                                     f"{at} failed finite-real evaluation: "
                                     f"{error.__cause__ or error}") from error
        else:
            # The C-order position of the support's grid tuple, first support variable most significant.
            values = []
            for support, table in stored:
                position = 0
                for j in support:
                    position = position * k + indices[j]
                values.append(table.item(position))
        return (coerce_real_scalar(fsum([constant, *values]), context="objective at a refinement point"),
                coerce_real_scalar(fsum([difference, *values]), context="objective at a refinement point"))

    return evaluate


def _coordinates(grid, box, search):
    """Return, for each variable, the original coordinates of the K points of the level grid.

    The coordinates come from ``grid.grid_value`` of the level's grid
    (``_level_grid``), whose points the level's QHD Plan tabulates and whose
    marginals its result reports. A physical level solves in the original
    variables, so these are the grid values themselves. A search-model level
    solves at the unit points ``u_i``. Its table value there is the level
    objective ``F(a + D u)`` at ``u = u_i`` (``_unit_objective``), so it is F
    at the exact point ``a + D u_i`` up to the evaluation's rounding.
    ``_to_box`` rounds that point once to binary64. The rounded coordinates
    place the cell faces and serve as the displayed point. The point can
    move by up to half a unit in the last place, onto a value where F
    differs or is undefined, so ``refine_box`` evaluates F through the unit
    point instead.
    """
    k = grid.num_grid_points
    return tuple(
        tuple(_to_box(side, grid.grid_value(j, i)) if search else grid.grid_value(j, i) for i in range(k))
        for j, side in enumerate(box)
    )


def _face(p, q):
    """Return the midpoint of the coordinates p and q, rounded once to binary64.

    It is the face between the centered cells of two adjacent grid points,
    ``x_i + h/2`` on a uniform grid of spacing h.

    Return the midpoint of two finite binary64 coordinates, rounded once to
    binary64. A finite rounded sum can be halved without changing that
    result, including in the subnormal range. If the sum overflows, halving
    both same-sign inputs is exact and their sum rounds the midpoint
    directly. Let ``t = p + q`` be the exact real sum
    and ``s = RN(t)`` finite. If ``abs(s) >= 2*nu`` (``nu = 2**-1022``),
    halving s is exact normal scaling and the rounding lattice around t
    scales to the lattice around t/2, so ``s/2 = RN(t/2)``. If
    ``abs(s) < 2*nu``, every possible t is an integer multiple of
    ``2**-1074`` in a range where those sums are representable, the first
    addition is exact, and the single rounding of ``s/2`` is ``RN(t/2)``,
    including a subnormal result or signed underflow to zero. If the sum
    overflows, p and q have the same sign, both halves are normal and
    exactly representable, and ``RN(p/2 + q/2)`` rounds the exact midpoint
    once. An exactly zero sum returns positive zero, as the exact rational
    midpoint does.
    """
    p, q = float(p), float(q)
    s = p + q
    if math.isfinite(s):
        # Match Fraction's positive zero for an exactly zero sum.
        return 0.0 if s == 0 else s * 0.5
    return p * 0.5 + q * 0.5


def _resolved(coordinates, box):
    """Return whether every side of the level box is above its width floor.

    The width floor is a geometric resolution condition. Let
    ``f_i = _face(x_i, x_(i+1))`` be the face between grid points i and i+1
    of a side ``[a, b]``, with the original coordinates of ``_coordinates``.
    A side is resolved when ``x_i < f_i < x_(i+1)`` for every i, ``a < f_0``
    and ``f_(K-2) < b``. Its faces then increase,
    ``a < f_0 < f_1 < ... < f_(K-2) < b``, because
    ``f_i < x_(i+1) < f_(i+1)``, and its grid points are distinct and
    increasing. ``_next_box`` builds a nonempty box inside the level box from
    these faces. A box with an unresolved side stops refinement with
    ``width_floor``. The test reads the actual coordinates, so it needs no
    bound on the rounding inside the grid owner's arithmetic.

    With M the largest magnitude among the side's coordinates and faces, a
    face lies strictly between two adjacent coordinates at least ``2 ulp(M)``
    apart: their exact midpoint lies at least ``ulp(M)`` from both, and
    rounding to nearest moves it by at most ``ulp(M)/2``. The grid points lie
    in the box up to rounding, so a side of n intervals reaches the floor at
    a width of the order of ``n ulp(M)``, below which distinct grid points can
    share a coordinate. The floor keeps coordinates
    distinct, not objective values, which near a smooth minimum stop
    differing much earlier.
    """
    for (a, b), points in zip(box, coordinates, strict=True):
        faces = [_face(p, q) for p, q in zip(points, points[1:])]
        if not (a < faces[0] and faces[-1] < b and all(p < f < q for p, f, q in zip(points, faces, points[1:]))):
            return False
    return True


def _level_geometry(problem, box, qhd, search, first):
    """Return ``(grid, coordinates, resolved)`` of a level box (``_resolved``).

    A box that the grid owner rejects (``_level_grid``) raises its
    ValueError at the first level, as QHD's own planning of that box would.
    At a later level the shrinking box has passed the range where its
    spacing or kinetic coefficient is a normal binary64 number, which is the
    physical model's width floor, and the result is ``(None, None, False)``.
    """
    try:
        grid = _level_grid(problem, box, qhd, search)
    except ValueError:
        if first:
            raise
        return None, None, False
    coordinates = _coordinates(grid, box, search)
    return grid, coordinates, _resolved(coordinates, box)


def _axis_interval(masses, eta):
    """Return ``(first, last, mass)``, the index interval of one axis that holds conditional mass eta.

    ``masses`` is the normalized marginal ``p(i) = marginal(i)/valid_mass``
    of Wu et al. arXiv:2605.12066v1, Eq. (12), p. 7. The paper normalizes by
    the total probability of the grid points, which is the valid mass in
    both encodings, the total for the binary encoding and the mass of the
    valid one-hot outcomes for the one-hot encoding. Following their
    Eq. (13), the interval starts at the largest p and adds one neighbor at
    a time, the one with the larger p, or the only one left at an end of the
    axis. The paper does not break ties.
    Starting at the smallest index among equal largest values and adding the
    lower neighbor on equal values make the interval a deterministic
    function of the marginal. It stops once the running sum, accumulated in
    the order of addition, is at least eta, or once it covers the whole
    axis. ``mass`` is that running sum, the value the stopping test
    compared. On the periodic grid the interval does not wrap from index
    K-1 to 0, so mass on both sides of the identified faces ``a = b`` makes
    it grow across the axis (``_next_box``).
    """
    k = len(masses)
    first = last = max(range(k), key=lambda i: (masses[i], -i))
    mass = masses[first]
    while mass < eta and (first > 0 or last < k - 1):
        left = masses[first - 1] if first > 0 else None
        right = masses[last + 1] if last < k - 1 else None
        if right is None or (left is not None and left >= right):
            first -= 1
            mass += masses[first]
        else:
            last += 1
            mass += masses[last]
    return first, last, mass


def _joint_mass_bound(axis_masses):
    """Return ``max(0, 1 - sum_j (1 - m_j))``, a lower bound on the conditional mass of the joint box.

    With ``A_j`` the event that coordinate j lies in its interval, the
    complement of the joint box is the union of the complements of the
    ``A_j``. Boole's inequality (the union bound) gives
    ``P(not all A_j) <= sum_j (1 - m_j)``, so
    ``P(all A_j) >= 1 - sum_j (1 - m_j)``, whatever the joint distribution
    with these marginals. Wu et al. arXiv:2605.12066v1, Sec. V, state only
    the per-axis threshold eta. Neither eta nor the product of the ``m_j``
    is a bound on the joint mass. The distribution ``[[0.8, 0.1], [0.1, 0]]`` has ``m_j = 0.9``
    on its first index for both axes (product 0.81) and joint mass 0.8,
    which the bound attains. A caller who needs joint mass ``1 - delta``
    sets ``eta = 1 - delta/d``.
    With counts these are empirical masses conditional on a valid outcome,
    not a confidence bound for the underlying population.
    """
    return max(0.0, 1.0 - fsum(1.0 - m for m in axis_masses))


def _next_box(box, coordinates, intervals):
    """Return the next level's box under the centered rule of ``BoxRefinement.box_rule``.

    Kept grid point i represents its centered cell, from the face
    ``f_(i-1) = _face(x_(i-1), x_i)`` to the face ``f_i = _face(x_i, x_(i+1))``
    between its original coordinate and those of its neighbors
    (``_coordinates``). For the interval ``[first, last]`` of a side
    ``[a, b]`` the next side is ``[f_(first-1), f_last]``, except that an
    interval that reaches an end index keeps that face of the box, so the
    lower bound stays a when ``first = 0`` and the upper bound stays b when
    ``last = K - 1``. On a uniform grid of spacing h the faces are
    ``x_i - h/2`` and ``x_i + h/2``, so the rule is the centered counterpart
    of Wu et al. arXiv:2605.12066v1, Eq. (14), p. 7, whose left-endpoint
    cells ``[x_l, x_(u+1)]`` differ from these by half a cell. The paper
    clips those endpoints to the current domain, which the rule here does by
    keeping the box face at an end index. On the Dirichlet interior grid the end
    cells extend to the box faces, and on the grid with boundary points they
    are cut at them. On the periodic grid (``x_0 = a``, ``x_(K-1) = b - h``)
    the next side is an ordinary sub-interval of ``[a, b]`` that does not
    wrap across the identified faces, and its end cells are not centered.
    The centered cell of x_0 is ``[a - h/2, a + h/2]`` on the circle, that
    is ``[b - h/2, b]`` together with ``[a, a + h/2]``. An interval that
    starts at index 0 keeps only ``[a, a + h/2]``, and one that ends at
    index K - 1 keeps the centered cell of ``x_(K-1)`` together with
    ``[b - h/2, b]``, the other half of the cell of x_0. Every kept grid
    point still lies in the next side. When the level's mass lies on both
    sides of them, the interval grows across the axis and keeps the whole
    side, so this rule depends on where the period is cut. The next level
    keeps the periodic kinetic on its sub-box, a choice of the search rather
    than a property of the problem.

    The bounds satisfy ``a <= lower < upper <= b`` for every resolved box.
    All faces lie strictly inside ``(a, b)`` in increasing order
    (``_resolved``). With ``0 < first`` and ``last < K - 1``,
    ``f_(first-1) < x_first <= x_last < f_last``. With ``first = 0`` the
    lower bound a is below ``f_0``, which is at most the upper bound, and
    with ``last = K - 1`` the upper bound b is above ``f_(K-2)``, which is at
    least the lower bound. Every kept grid point that is not an end point of
    its axis lies strictly inside the next side. The next box equals the
    level box exactly when every interval covers its whole axis.
    """
    sides = []
    for (a, b), x, (first, last) in zip(box, coordinates, intervals, strict=True):
        lower = a if first == 0 else _face(x[first - 1], x[first])
        upper = b if last == len(x) - 1 else _face(x[last], x[last + 1])
        sides.append((lower, upper))
    return tuple(sides)


def _best_point_gaussian(best, box, width, search):
    """Return the ``GaussianState`` of ``BoxRefinement.level_initial_state="best_point_gaussian"`` for a level on ``box``.

    ``best`` is the refinement's best completed level so far
    (``RefinementLevel``, the least relative objective) and ``width`` the
    fraction w of ``BoxRefinement.gaussian_width``. The Gaussian is
    ``prod_j exp(-(x_j - p_j)**2/(2 (w L_j)**2))`` in the original
    coordinates, centered at the best level's location p with width w times
    the side ``L_j = b_j - a_j`` of the level box ``[a, b]``, so it keeps its
    size relative to the box as the box shrinks. The level problem solves in
    its own coordinates (``_level_problem``), and the record is written in
    them. The search model solves in ``u = (x - a)/L``, where the factor
    becomes ``exp(-(u_j - (p_j - a_j)/L_j)**2/(2 w**2))``, so the center is
    ``(p_j - a_j)/L_j`` and the width w on every axis. There p is the exact
    image ``a' + D' u'`` of the best level's unit point u' in its box
    ``[a', b']`` (``RefinementLevel.unit_point``), whose rounded value is the
    displayed point, and the center is formed from the exact rationals of
    these binary64 numbers and rounded once. The physical model solves in x,
    with the recorded point p as the center and the widths ``w L_j``
    rounded once.

    The center is not clipped to the box. A best point outside the level
    box, for example in the region a stall split gave up, gives a Gaussian
    centered outside it, which ``GaussianState`` defines and normalizes on
    the grid for every finite center, because it shifts the exponents by
    their minimum. The state's construction error and native preparation are
    those of any ``GaussianState`` (``initial_state``).

    The Gaussian carries the location of the best point into the next box,
    at the cost of momentum components that the kinetic ground state does
    not have, and neither start is better after every finite-time evolution.
    On ``(x - 3/10)**2`` over ``[0, 1]`` with K = 16, the split-step flavor,
    T = 1, the quadratic schedule with gamma = 0.3, gain 8, eta = 0.99 and
    ``max_levels = max_no_improve = 6``, a kinetic first level followed by
    Gaussians of width 1/6 ended at best-point error 0.00668, against 0.0534
    with the kinetic ground state at every level, with six solves each and
    the same results at 128, 512 and 2048 steps. With the default
    ``max_no_improve = 2`` the Gaussians stopped after three levels at
    0.112, since their second and third levels did not improve on the
    first. On the double well of
    ``_level_problem`` at gain 1, with K = 12, T = 10, 200 steps, eta = 0.99,
    the most probable point, ``max_levels = max_no_improve = 10`` and
    ``max_work = max_bytes = 10**10``, a
    uniform first level followed by these Gaussians stopped after six levels
    at error 0.116, against 0.0062 with the uniform state and 0.00022 with
    the kinetic ground state at every level, and after a kinetic first level
    it reached 0.00137. In that comparison, which also covered the
    anisotropic quadratic and the Ackley function of ``BoxRefinement`` at
    gains 1 and 8, the kinetic ground state at every level gave the smallest
    error in all six cases, after either first level. The module docstring
    gives the revision and environment of these measurements.
    """
    from .initial_state import GaussianState

    centers, widths = [], []
    for j, (a, b) in enumerate(box):
        side = Fraction(b) - Fraction(a)
        if search:
            # (p - a)/L with p = a' + (b' - a') u', exact until the one rounding.
            (a0, b0), u = best.box[j], best.unit_point[j]
            location = Fraction(a0) + (Fraction(b0) - Fraction(a0)) * Fraction(u)
            centers.append(float((location - Fraction(a)) / side))
            widths.append(float(width))
        else:
            centers.append(float(best.point[j]))
            widths.append(float(Fraction(width) * side))
    return GaussianState(center=tuple(centers), widths=tuple(widths))


def _support_tables(qhd, problem, show=None):
    """Return ``(tables, decomposer, evaluations)`` of ``problem`` as ``QHD.plan`` would select them.

    This is the table stage of ``QHD.plan``: the objective admission, the
    admitted symbolic work (``QHD._admit_symbolic_work``) and one evaluation
    of each support table. It selects no schedule blocks and no kernel, so
    the admission of a classical evolution of the unscaled objective, which
    is never run, is not charged. Its admission still opens with the initial
    state's ``d K`` units, as in ``QHD.plan``, although the stage evaluates
    no initial state, so it admits exactly what the table admission of a
    Plan without compiled blocks, a classical ``schrodinger`` Plan, admits.

    The tables are evaluated by ``compiler.evaluate_support``, the module
    function of the planning table producer
    (``compiler.QHDCompiler._objective_grid_values``), with the centered
    coordinate vectors formed as that producer forms them,
    ``grid.grid_value(j, i) - decomposer.centers[j]`` in binary64, and the
    admitted chunk of each support, so the entries are the ones planning
    would store. No ``QHDCompiler`` is built, so no step weights are formed
    for this stage. ``tables`` holds one ``records.SupportValues`` per
    support, in the order of the decomposition's support expressions, with
    its cached exact extrema. ``decomposer`` is the admitted
    ``objective.ObjectiveDecomposer``, whose constant c0 is checked to be
    finite and real here and whose support expressions the tables evaluate.
    The decomposition groups the terms of the expanded objective, so
    ``c0 + sum_S p_S`` is the objective exactly. On the unit box the
    expansion center is the origin (``objective.expansion_centers``), so the
    support expressions are in the problem's own variables. The refinement
    reuses this decomposition and these unscaled tables for F and F - C
    (``_tabulated_objective``) and never reconstructs F from the rounded
    scaled tables.

    A refused table entry is reported at its expression and grid point
    (``compiler.user_point``). For a refinement level's unit problem,
    ``show`` (``_problem_coordinates``) maps both back to the coordinates of
    the problem being refined.
    """
    import numpy as np

    from nwqlib.core.records import FrozenArray
    from .compiler import evaluate_support, user_point
    from .potential import coerce_real_scalar
    from .records import SupportValues
    from .validation import _lambdify_objective, _validate_qhd_problem_fields

    _validate_qhd_problem_fields(problem)
    evaluations, decomposer, _, chunks = qhd._admit_symbolic_work(problem, False)
    grid = OneHotGrid(problem.variables, problem.bounds, qhd.num_grid_points, qhd.include_boundary_points,
                      qhd.boundary)
    coerce_real_scalar(decomposer.constant, context="constant objective")
    k = grid.num_grid_points
    tables = []
    for support, expression in decomposer.support_expressions.items():
        evaluator = _lambdify_objective([problem.variables[j] for j in support], expression)
        centered_axes = [np.array([grid.grid_value(j, i) - decomposer.centers[j] for i in range(k)],
                                  dtype=np.float64) for j in support]
        locate = user_point(grid, decomposer.centers, support, expression)
        if show is not None:

            def locate(indices, found=locate, support=support):
                shown, point = found(indices)
                return show(shown, support, point)

        values = evaluate_support(evaluator, centered_axes, chunks[support], expression=expression,
                                  support=tuple(support), locate=locate)
        tables.append(SupportValues.tabulate(tuple(support), FrozenArray(values)))
    return tuple(tables), decomposer, evaluations


def _table_magnitude(tables):
    """Return ``sum_S max |T_S|`` over the grid, the size of the table values that E is a difference of.

    Each table value carries its own evaluation rounding, a few units of
    ``u |T_S|`` for a simple expression and more for an ill-conditioned one,
    with ``u = 2**-53``. The search model divides the tables by E, so that
    rounding reaches its potential multiplied by ``table_magnitude/E``, the
    conditioning that each level records (``RefinementLevel.conditioning``,
    derived in ``_level_problem``). No rounding bound that holds for every
    objective expression is known here, so the level reports the factor
    rather than a certified error.
    """
    # The cached exact extremum max |T_S| of each stored table (records.SupportValues.magnitude).
    return fsum(t.magnitude for t in tables)


def _resolution_stop(tables):
    """Return ``"flat_objective"``, ``"unresolved_objective"`` or None for a level's stored support tables.

    With the exact range sum ``R = sum_S (max T_S - min T_S)`` of the stored
    binary64 values, ``R = 0`` means E = 0 (``_outer.range_bound``). The
    tables then show no variation and there is no potential to normalize,
    so the level stops with ``flat_objective``.

    A positive R at most ``sum_S ulp(max_i |T_S(i)|)`` stops the level with
    ``unresolved_objective``. This is a resolution rule, not a proof that
    the objective is constant. ``math.ulp`` of a table's largest magnitude
    is the spacing of binary64 numbers at the top of that table, subnormal
    spacing included, so such a range is at most one such spacing per
    table, the size that an evaluation error of about one unit in the last
    place at each grid point can produce. The tables cannot then tell the
    objective's variation from its evaluation error, and normalizing by E
    would turn that error into a potential of order one. A varying function
    can fall under the rule. ``exp(2**-52 x)`` on the grid ``{0, 1}`` is
    strictly increasing, its correctly rounded values 1.0 and
    ``1 + 2**-52`` differ by one spacing, and it stops. The evaluation
    error can also exceed the threshold, as for ``sin(x)**2 + cos(x)**2``
    evaluated through rounded intermediates, so a level that passes can
    still vary mostly by rounding. Its ``RefinementLevel.conditioning``
    reports this, and the rule does not bound it. For normal values the
    threshold is at most ``2 u sum_S max |T_S|`` with ``u = 2**-53``. Both
    sums are formed and compared exactly.
    """
    # The cached exact extrema of the stored tables (records.SupportValues).
    variation = sum((Fraction(t.maximum) - Fraction(t.minimum) for t in tables), Fraction(0))
    if variation == 0:
        return "flat_objective"
    rounding = sum((Fraction(ulp(t.magnitude)) for t in tables), Fraction(0))
    return "unresolved_objective" if variation <= rounding else None


def _require_scale(tables):
    """Return E = ``_outer.range_bound(tables)``, raising when the bound exceeds binary64.

    A level whose E is not finite cannot be normalized, so it stops the
    refinement as ``inner_failed`` with this message.
    """
    # range_bound reads each table's max and min, here its cached exact extrema.
    scale = range_bound((t.minimum, t.maximum) for t in tables)
    if scale is None:
        raise ValueError("the level objective's table ranges sum beyond binary64, so E is not finite")
    return scale


def _level_problem(problem, box, qhd, scaling, gain, stored=None):
    """Return ``(level problem, tables, c, table evaluations, decomposer)`` of one level box.

    ``"physical"`` returns the original objective on ``box``, whose kinetic
    ``-1/2 sum_j d^2/dx_j^2`` on grid spacing ``h_j = L_j/n`` grows as the
    box shrinks. Its tables come from its own planning, so this function
    returns ``tables = c = decomposer = None`` and no table evaluations.

    ``stored`` is ``(tables, evaluations, decomposer)``: the tables and
    evaluations that a durable level persisted when its table stage
    completed (``_LevelTables``) and the symbolic decomposition of its
    objective (``_level_decomposition``), formed before they were loaded. A
    resumed level then reads those tables instead of evaluating them again,
    since the support expressions and the constant cannot be rebuilt from
    the stored entries.

    ``"search_model"`` returns, on the unit box ``u in [0, 1]^d``, the
    normalized objective ``V(u) = kappa (F(a + D u) - c)/E``, with a the
    lower corner of ``box``, ``D = diag(L_j)`` its side lengths and kappa the
    gain. The unit grid
    has the same spacing at every level, so QHD applies the same
    dimensionless kinetic ``-1/2 sum_j d^2/du_j^2``. The substitution uses
    the exact rationals of the binary64 bounds, so a polynomial objective
    keeps its variation over a small box without cancellation against its
    value. The unscaled objective ``F(a + D u)`` is planned first
    (``_support_tables``). Its stored tables T_S and constant term c0 give
    ``E = _outer.range_bound(tables)``, at least ``sum_S (max T_S - min T_S)``,
    and ``c = c0 + sum_S min T_S``. With these tables in exact arithmetic,
    ``F - c = sum_S (T_S - min T_S)`` lies in ``[0, E]`` at every grid point,
    so ``(F - c)/E`` lies in ``[0, 1]``. A level that ``_resolution_stop``
    stops returns no level problem.

    ``QHD.plan`` then plans the solved objective ``V = kappa (F - c)/E``, with
    kappa the dimensionless ``gain`` (``BoxRefinement.gain``), a
    second planning whose support tables
    the resources count separately. V is formed without rounding c. Since
    ``F(a + D u) - c0`` is the sum of the support expressions
    (``_support_tables``), V is that sum minus the exact rationals of the
    stored minima, times the exact rational ``f = kappa/E`` of the binary64
    numbers kappa and E. The constant term c0 never enters, whether SymPy
    holds it as an integer, a rational or a binary64 ``Float``. V's constant
    term is ``-f sum_S min T_S``, and its support expressions are those of F
    times f. Subtracting a rounded c would leave a constant of the size of
    the lost digits, and SymPy adds the minima to a ``Float`` c0 in 53-bit
    arithmetic, which rounds c in the same way. Dividing by E alone would
    leave the constant ``c/E``, which ``method.objective_at`` adds into
    every entry of the classical potential diagonal. For ``1e16 + x`` on
    ``[0, 1]`` with K = 6 that constant is about ``1.4e16``, and the
    diagonal rounds to two distinct values. The recorded c is the exact sum
    ``c0 + sum_S min T_S`` rounded once to binary64, from its exact rational
    value or, for an irrational c0, from a 60-digit evaluation. It serves
    the record only.

    The second planning evaluates V's tables T'_S anew, so they differ from
    ``f T_S`` by rounding. If every stored value lies within ``e_S`` of the
    exact support value and every value of T'_S within ``e'_S`` of f times
    that value, with the evaluation that uses the actual factor f, then
    ``|T'_S - f T_S| <= e'_S + f e_S``. The constant term is converted to
    binary64 once, which in normal arithmetic moves it by at most
    ``u f sum_S |min T_S|``, with ``u = 2**-53``. At every grid point the
    exact sum of the solved constant and table values therefore lies within
    ``delta = sum_S (e'_S + f e_S) + u f sum_S |min T_S|`` of kappa times
    the value in ``[0, 1]`` that the stored tables give, so in
    ``[-delta, kappa + delta]``. The classical kernel's ``method.objective_at``
    rounds that sum once more with ``fsum``, by at most u times its
    magnitude. For values rounded once, ``e_S`` is about ``u max |T_S|`` and
    ``e'_S`` about ``2 u f max |T_S|``, one more rounding for the factor f,
    so delta is about ``4 u kappa`` times ``table_magnitude/E``
    (``RefinementLevel.conditioning``). This estimate illustrates the scale
    for simple expressions. It does not bound the evaluation error of an
    arbitrary expression, whose e_S and e'_S can be much larger. The
    resolution stop does not bound the conditioning either.
    ``exp(2**-51 x)`` on the grid ``{0, 1}`` passes it with conditioning
    about ``2.25e15``, where ``4 u`` times the conditioning is about one.
    Each level reports its conditioning. A strict enclosure would need
    outward rounding of every table evaluation.

    Kappa acts after E is formed. Multiplying F by kappa before forming E
    would cancel, because E and c scale with F. A fixed kappa depends on
    neither the box, the coordinates nor the units of F, so the invariances
    below hold for every kappa. A larger kappa strengthens the potential
    against the kinetic term. For a local quadratic
    ``-(a/2) d^2/du^2 + (kappa b lambda/2) u^2`` the ground-state amplitude
    ``exp(-u**2/(2 sigma**2))`` has ``sigma**4 = a/(kappa b lambda)``, so it
    narrows as kappa grows. A finite-time run need not follow the ground
    state, however, and a larger potential difference also limits
    transitions between basins. For two sites with constant hopping v and
    constant detuning ``kappa Delta``, a state that starts on one site moves
    at most ``4 v**2/(4 v**2 + kappa**2 Delta**2)`` of its probability to the
    other. This bound needs both assumptions. A coherent superposition of
    the two sites can change a site's population by more, and
    time-dependent coefficients are not covered. So kappa has no general
    monotone effect and no value is best for every problem. Through
    ``refine_box`` on the double well
    ``(2 x**2 - 1)**2 + 3 x/5 + 2 (y - 3/10)**2 + 6 x y/5`` on
    ``[-1.2, 1.2]**2``, with K = 6, T = 10, 80 steps, the quadratic schedule
    with gamma = 0.3, the uniform initial state, eta = 0.99, the most
    probable point, ``max_levels = 10``, ``max_no_improve = 10`` and
    classical execution, kappa = 1 stopped after five levels with
    best-point error 0.0826 in the infinity norm. At its last level both end cells of each axis held more than
    ``1 - eta`` of the probability, so every contiguous interval of mass eta
    covered the whole axis. kappa = 8
    kept shrinking for ten levels to error 0.00325, and kappa = 64 stopped
    after seven levels at 0.0791. With 160 and 320 steps kappa = 1 and 8 gave
    the same numbers of levels and best errors, and kappa = 64 stopped after
    six levels at the same error, so that run is not converged in the step
    count. With the default ``max_no_improve = 2`` all three gains stop
    after three levels at 0.0826. The module docstring gives the revision
    and environment of these measurements.

    Under ``x -> s x + t`` with positive diagonal s (anisotropic scaling
    included), the box ``s B + t`` and the objective
    ``F'(x) = r F((x - t)/s) + c1`` with ``r > 0`` give
    ``F'(a' + D' u) = r F(a + D u) + c1``, so E scales by r, c becomes
    ``r c + c1``, and V, the kinetic and hence the whole level are
    unchanged. The next box then maps as ``s B + t`` too. In binary64 two
    such runs agree bit for bit when SymPy's exact rational arithmetic
    reduces both level objectives to the same expression, which needs the
    transformed bounds to be the exact images of the level bounds, when r is
    a power of two, so that tables, E and c scale exactly, and when every
    rounded coordinate and face of each level maps exactly, as with dyadic
    data on a grid whose unit coordinates are dyadic. Otherwise the levels
    agree to rounding, and rounding can change a discrete choice near a tie
    or at the threshold. The decisions between levels (the best level, the
    no-improvement count, the ``mode_or_mean`` choice and the split scores)
    compare ``F - C`` with one exact constant C for the whole refinement
    (``_tabulated_objective``). C becomes ``r C + c1`` exactly, so these
    values scale by r, exactly when r is a power of two, and c1 cancels
    from them exactly when c1 and every coefficient of F are exact SymPy
    numbers, such as integers and rationals. A binary64 ``Float`` anywhere
    in F or c1 makes SymPy form each constant term in 53-bit arithmetic at
    the size of c1, so next to a large c1 the compared values can lose the
    objective's variation as F does (``_tabulated_objective``). The
    reported objective F, in any case, keeps c1 and rounds the objective's
    variation to the spacing of binary64 numbers near it, which for
    ``c1 = 2**54`` is 4. The physical model lacks the scaling
    invariances, since its kinetic scales by ``1/s**2`` relative to the
    potential and not at all with r. The search model is a different
    Hamiltonian at each level, not a change of variables of the original
    one. In u the original kinetic is ``-1/2 sum_j L_j**-2 d^2/du_j^2``, so
    for side lengths (1, 2) the physical weights are (1, 1/4) while the model
    uses (1, 1).
    """
    if scaling == "physical":
        level = Optimization(
            objective=problem.objective, variables=problem.variables, bounds=box, unit=problem.unit
        )
        return level, None, None, 0, None
    unit, objective = _unit_objective(problem, box)
    unscaled = Optimization(objective=objective, variables=unit, bounds=((0.0, 1.0),) * len(unit))
    if stored is None:
        tables, decomposer, evaluations = _support_tables(qhd, unscaled, _problem_coordinates(problem, box))
    else:
        tables, evaluations, decomposer = stored
    constant = decomposer.constant
    variation = sp.Add(*decomposer.support_expressions.values())
    if _resolution_stop(tables) is not None:
        return None, tables, None, evaluations, decomposer
    scale = _require_scale(tables)
    # V = kappa (F - c)/E = (kappa/E) (variation - sum_S min T_S), with c0
    # left out and the exact rationals of kappa, E and the stored minima,
    # the cached exact extrema of the tables (records.SupportValues).
    minima = sum((sp.Rational(t.minimum) for t in tables), sp.Integer(0))
    factor = sp.Rational(gain) / sp.Rational(scale)
    level = unscaled.revise(objective=(variation - minima) * factor)
    # The recorded c = c0 + sum_S min T_S, summed exactly and rounded once. A
    # binary64 Float in c0 enters as its exact rational, since SymPy would add
    # it to the minima in 53-bit arithmetic.
    shift = _exact_constant(constant) + minima
    energy_shift = (float(Fraction(int(shift.p), int(shift.q))) if shift.is_Rational
                    else float(shift.evalf(60)))
    return level, tables, energy_shift, evaluations, decomposer


def _level_resources(plan, trace, *, run_reason=None, planning_reason=None, table_evaluations=0,
                     table_reason=None, objective_evaluations=0, joint_mass_reads=0):
    """Count one level's work from its Plan, its Run trace and the refinement's own evaluations.

    The Run counts come from ``_outer.run_counts``: from the trace read after
    the level Run finished or failed, or from the closed folder of a durable
    level whose preparation raised (``_durable.closed_trace``, which also
    gives ``run_reason``), zero when no Run was started, and unknown with
    ``run_reason`` when the preparation raised without a durable Run to read
    or when that Run's journal cannot be read.
    A durable level whose Run's creation raised before its journal header
    has zero counts apart from its stored data bytes (``_outer.HeaderlessRun``).
    Support evaluations come from the Plan. Without a Plan they are unknown
    with ``planning_reason`` when the planning of the solved objective
    raised, and zero when it was not attempted. The CX bound and the
    arbitrary-rotation count are those of the circuits the level's Run
    prepared (``_outer.law_count``), unknown after a preparation that
    raised once the header was committed, and zero before it.
    ``table_evaluations`` of the search model's first planning is None, with
    ``table_reason``, when that planning raised.
    """
    counts, unavailable = run_counts(trace, run_reason)
    reasons = list(unavailable)
    counts.update(table_evaluations=table_evaluations, objective_evaluations=objective_evaluations,
                  joint_mass_reads=joint_mass_reads)
    if table_evaluations is None:
        reasons.append(("table_evaluations", table_reason))
    laws = ("cx", "arbitrary_rotations")
    if plan is not None:
        counts["support_evaluations"] = plan.reconstruction.support_evaluations
        for metric in laws:
            counts[metric], law_reason = law_count(
                plan, metric, None if run_reason is not None else counts["circuit_preparations"], run_reason)
            if law_reason is not None:
                reasons.append((metric, law_reason))
    elif planning_reason is not None:
        # No Plan, so no Run prepared a circuit.
        counts.update(support_evaluations=None, **dict.fromkeys(laws, 0))
        reasons.append(("support_evaluations", planning_reason))
    else:
        counts.update(support_evaluations=0, **dict.fromkeys(laws, 0))
    return RefinementResources(**counts, unavailable=tuple(reasons))



def _state_probabilities(state):
    """Return ``numpy.abs(state)**2``, the probabilities of a classical kept state as the host kernel forms them.

    ``method._execute_theory`` computes its marginals, masses and most
    probable point from this expression, and the kernel's tie window
    (``method._host_window``) charges its rounding
    (``ABSOLUTE_SQUARE_ROUNDOFF``). ``abs(z)**2`` taken
    per amplitude can differ from it in the last bit, so the joint box mass
    (``_joint_mass``) and the split's point weights (``_point_weights``)
    both use this expression.
    """
    import numpy as np

    return np.abs(state) ** 2


def _kept_readout_choice(d, k, entries, native, held, max_bytes, max_work):
    """Choose a complete dense allowance, or an admitted streamed chunk."""
    from .decoding import QHD_DECODING_PROBABILITY_CHUNK_SIZE

    dimension = k**d
    dense_bytes = (48 if native else 32) * dimension + 24 * k + 65536
    fixed = 65536 + 1024 * d
    rate = 16 * d + 128
    dense_work = (d + 4) * dimension + (4 * d + 8) * entries
    if held + dense_bytes <= max_bytes and dense_work <= max_work:
        chunk, workspace = 0, dense_bytes
        work = dense_work
    else:
        work = (8 * d + 32) * dimension
        minimum = min(dense_bytes, fixed + rate) if dense_work <= max_work else fixed + rate
        _require_bytes(max_bytes, held=held, local=minimum,
                       stage='refinement readout (dense or one-point stream)')
        chunk = min(dimension, QHD_DECODING_PROBABILITY_CHUNK_SIZE,
                    (max_bytes - held - fixed) // rate)
        workspace = fixed + rate * chunk
    if work > max_work:
        raise ValueError(f'QHD refinement readout requires {work} work units, '
                         f'with QHD(max_work={max_work}). Use QHD(max_work>={work})')
    return chunk, workspace, work


class _LevelReadout:
    """The valid grid population of one level, decoded once from its joint observations and reused within the level.

    Decode each level's joint observations once into its valid grid
    population. One-hot indices set the bit ``j*K+i_j`` for every variable,
    and binary indices use the variable-axis permutation. Valid, invalid and
    total masses describe their respective observed populations. Joint box
    mass sums the selected grid slice and conditions it on valid mass. A
    result containing only marginals has no directly observed joint mass.

    Forms. A kept state, classical or native amplitudes, gives the dense
    ``(K,)*d`` grid of probabilities: ``_state_probabilities`` of the
    classical state, or ``np.square(np.absolute(z))`` of the native valid
    amplitudes gathered by ``decoding.onehot_register_indices`` or permuted
    by ``binary.lexicographic_register_array``, the evaluation of
    ``decoding.decode_statevector_probabilities``. Counts and exact bins
    give the observed valid entries of every chunk, decoded by array
    operations on the histogram indices (``decoding.onehot_valid_words``,
    ``decoding.onehot_local_indices``, the base-K digits of a binary index),
    with no ``2**(d*K)`` array: ``flat`` holds each entry's lexicographic
    grid index in observation order and ``values`` its integer count or its
    value ``v/C`` over C chunks. A readout wider than one 64-bit index is
    decoded entry by entry (``decoding.decode_histogram``). Integer counts stay exact
    integers until a probability is formed (int64 sums only while the
    returned shots fit ``MAX_COUNT``, Python integers otherwise).

    A kept-state readout first admits its dense probability grid and later
    mode workspace, 48D+24K+65536 bytes for native amplitudes and
    32D+24K+65536 bytes for classical amplitudes, beyond the caller's live
    buffers. D is the valid grid size and the caller includes every live
    kept state. If that route does not fit, streaming reserves
    65536+1024d+(16d+128)c bytes and chooses the largest c no greater than
    D or 4096 that fits the remainder. If no point fits, the readout refuses
    with QHD.max_bytes before constructing a grid or chunk. The stream
    allows old and new chunk coordinates to coexist and forms the same
    vectorized probabilities as the dense route. Each joint request uses one
    streamed pass. Both sides of a split share two passes, first their
    maxima and then their first qualifying points in C order. The read
    counter charges one full kept-state length per started pass, including
    a second pass that stops early. Dense-grid uses share one such charge.
    Subsequent streamed operations are admitted against the level's
    remaining readout-work allowance before their first pass
    (``_kept_readout_choice``: dense work ``(d+4)D+(4d+8)F`` for F kept
    amplitudes, ``(8d+32)D`` per streamed pass). The streamed chunk gathers
    each chunk's amplitudes at their register indices,
    ``sum_j 2**(j K + i_j)`` for one-hot registers and ``sum_j i_j K**j``
    for binary ones, and squares their moduli entry by entry as the dense
    grid does, so every probability is the same binary64 number.
    """

    def __init__(self, result, qhd, held):
        import numpy as np
        from nwqlib.execution import MAX_COUNT
        from .decoding import decode_histogram
        from .method import _bits, _grid

        plan = result.plan
        grid = _grid(plan)
        d, k = grid.num_variables, grid.num_grid_points
        self.shape, self.valid_mass = (k,) * d, result.valid_mass
        observation = plan.resolve("qhd").resolved_observation(plan)[1]
        self.kind, self.state, self.grid, self.flat, self.values = "exact", None, None, None, None
        self.available, self.counts, self._formed = True, False, False
        if plan.execution == "classical" or observation.kind == "amplitudes":
            if result.artifact is None:
                self.available, self.reads = False, 0
                return
            self.state = result.data.artifact(result.artifact).array
            self.native = plan.execution != "classical"
            self.bits = _bits(plan.method) if self.native else None
            self.reads = len(self.state)
            entries = len(self.state)
            chunk, workspace, work = _kept_readout_choice(
                d, k, entries, self.native, held, qhd.max_bytes, qhd.max_work)
            self._chunk_size = chunk
            self._pass_work, self._max_work = work, qhd.max_work
            self._readout_work = 0
            if chunk == 0:
                self.grid = self._population()
                self._readout_work = work
            return
        chunks = result.data.observations.chunks
        self.counts = observation.kind == "counts"
        self.kind = "empirical" if self.counts else "exact"
        shots = sum(c.returned_shots or 0 for c in chunks)
        self.denominator = shots if self.counts else len(chunks)
        exact = self.counts and shots > MAX_COUNT
        bits = _bits(plan.method)
        flats, values, self.reads = [], [], 0
        for chunk in chunks:
            histogram = chunk.histogram()
            self.reads += histogram.entries
            weights = histogram.weights
            valid, points = decode_histogram(histogram, grid, bits)
            flat = np.ravel_multi_index(tuple(points.T), self.shape).astype(np.int64)
            flats.append(flat)
            kept = weights[valid]
            values.append(kept.tolist() if exact else kept if self.counts else kept / len(chunks))
        self.flat = np.concatenate(flats) if flats else np.zeros(0, dtype=np.int64)
        if exact:
            self.values = [value for chunk_values in values for value in chunk_values]
        else:
            self.values = (np.concatenate(values) if values
                           else np.zeros(0, dtype=np.int64 if self.counts else np.float64))

    def _population(self):
        """Return the dense ``(K,)*d`` probability grid of the kept state."""
        import numpy as np

        if not self.native:
            return _state_probabilities(self.state).reshape(self.shape)
        d, k = len(self.shape), self.shape[0]
        if self.bits is None:
            from .decoding import onehot_register_indices

            amplitudes = self.state[onehot_register_indices(k, d)]
        else:
            from .binary import lexicographic_register_array

            amplitudes = lexicographic_register_array(self.state, d, self.bits).reshape(self.shape)
        return np.square(np.absolute(amplitudes))

    def _stream(self):
        """Yield ``(points, probabilities)`` of the grid in C order, one bounded chunk at a time, without the grid.

        ``points`` holds the d index arrays of the chunk's grid points. A pass
        after the first reads the kept state once more, which ``reads`` counts.
        """
        import numpy as np

        if self._formed:
            self.reads += len(self.state)
        self._formed = True
        d, k = len(self.shape), self.shape[0]
        total = k**d
        for start in range(0, total, self._chunk_size):
            flat = np.arange(start, min(start + self._chunk_size, total), dtype=np.int64)
            points = np.unravel_index(flat, self.shape)
            if not self.native:
                yield points, _state_probabilities(self.state[flat])
                continue
            index = np.zeros(flat.shape, dtype=np.int64)
            for j in range(d):
                index |= (np.int64(1) << (j * k + points[j])) if self.bits is None else points[j] << (j * self.bits)
            yield points, np.square(np.absolute(self.state[index]))

    def mode(self, axis, first, last, window):
        """Return ``method._summarize``'s point among the positive probabilities with axis index in ``[first, last]``.

        The lexicographically smallest point within ``window`` of their
        largest probability: the first such entry in C order, read from the
        kept grid or, for a grid the limits did not admit, from two streamed
        passes (the largest probability, then the first entry within the
        window).
        """
        import numpy as np

        if self.grid is not None:
            # Dense grid: the axis slice keeps C order, whose first true entry is the
            # lexicographically smallest point.
            cut = [slice(None)] * self.grid.ndim
            cut[axis] = slice(first, last + 1)
            block = self.grid[tuple(cut)]
            top = block[block > 0].max()
            flat = int(np.flatnonzero((block > 0) & (top - block <= window))[0])
            point = [int(i) for i in np.unravel_index(flat, block.shape)]
            point[axis] += first
            return tuple(point)
        return self.modes(axis, ((first, last),), window)[0]

    def modes(self, axis, ranges, window):
        """Find one or both split-side modes with one common pair of passes."""
        import numpy as np

        ranges = tuple(ranges)
        if not 1 <= len(ranges) <= 2:
            raise ValueError('readout modes requires one or two index intervals')
        if self.grid is not None:
            return tuple(self.mode(axis, first, last, window)
                         for first, last in ranges)
        self._admit_passes(2)
        tops = [None] * len(ranges)
        for points, values in self._stream():
            for side, (first, last) in enumerate(ranges):
                inside = (points[axis] >= first) & (points[axis] <= last) & (values > 0)
                if inside.any():
                    largest = values[inside].max()
                    tops[side] = largest if tops[side] is None else max(tops[side], largest)
        if any(top is None for top in tops):
            raise ValueError('the side holds no positive probability')
        found = [None] * len(ranges)
        for points, values in self._stream():
            for side, (first, last) in enumerate(ranges):
                if found[side] is not None:
                    continue
                chosen = np.flatnonzero((points[axis] >= first) & (points[axis] <= last) & (values > 0)
                                        & (tops[side] - values <= window))
                if chosen.size:
                    found[side] = tuple(int(point[chosen[0]]) for point in points)
            if all(point is not None for point in found):
                return tuple(found)
        raise ValueError('the side holds no positive probability')

    def _admit_passes(self, count):
        """Admit the next complete streamed operation before its first pass."""
        required = self._readout_work + count * self._pass_work
        if required > self._max_work:
            raise ValueError(
                f"QHD refinement streamed readout requires {required} work units "
                f"({self._readout_work} used + {count * self._pass_work} requested), "
                f"with QHD(max_work={self._max_work}). Use QHD(max_work>={required})")
        self._readout_work = required

    def pooled(self):
        """Return ``(indices, weights)`` of each observed valid point once, in first-observation order.

        Integer counts are summed exactly over chunks and values ``v/C`` are
        added in observation order, as ``method.QHD.analyze`` pools them.
        """
        import numpy as np

        unique, first, inverse = np.unique(self.flat, return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")
        if isinstance(self.values, list):
            sums = [0] * unique.size
            for position, value in zip(inverse.tolist(), self.values, strict=True):
                sums[position] += value
            weights = np.array([sums[i] for i in order.tolist()], dtype=object)
        else:
            sums = np.zeros(unique.size, dtype=self.values.dtype)
            np.add.at(sums, inverse, self.values)
            weights = sums[order]
        indices = np.stack(np.unravel_index(unique[order], self.shape), axis=-1).astype(np.int64)
        return indices.reshape(-1, len(self.shape)), weights

    def joint(self, intervals):
        """Return the joint box mass conditioned on the valid mass (``_joint_mass``)."""
        import numpy as np

        if self.grid is not None:
            slices = tuple(slice(first, last + 1) for first, last in intervals)
            return fsum(map(float, self.grid[slices].flat)) / self.valid_mass
        if self.state is not None:
            self._admit_passes(1)
            # The same terms streamed from the kept state; fsum is correctly rounded in any order.
            def terms():
                for points, values in self._stream():
                    inside = np.ones(values.shape, dtype=bool)
                    for j, (first, last) in enumerate(intervals):
                        inside &= (points[j] >= first) & (points[j] <= last)
                    yield from map(float, values[inside])

            return fsum(terms()) / self.valid_mass
        points = np.stack(np.unravel_index(self.flat, self.shape), axis=-1).reshape(-1, len(self.shape))
        inside = np.ones(self.flat.shape, dtype=bool)
        for j, (first, last) in enumerate(intervals):
            inside &= (points[:, j] >= first) & (points[:, j] <= last)
        if self.counts:
            # The exact integer count in the box, divided once by the returned shots.
            numerator = (sum(value for value, keep in zip(self.values, inside.tolist(), strict=True) if keep)
                         if isinstance(self.values, list) else int(np.sum(self.values[inside], dtype=np.int64)))
            return numerator / self.denominator / self.valid_mass
        return fsum(self.values[inside].tolist()) / self.valid_mass


def _joint_mass(readout, intervals):
    """Return ``(mass, kind)``: the joint box's conditional mass from the level's decoded joint observations.

    The box is the product of the index intervals. A kept state sums the
    selected slice of its dense grid with ``fsum``, the classical C-order
    terms of the direct box sum, and divides by the result's valid mass,
    ``joint = fsum(valid_population[slices].flat) / valid_mass``. A strided slice's ``.flat`` iterates without a D-length
    copy, and an empty intersection gives zero. Counts sum their exact
    integer counts in the box and divide once by the total returned shots
    (``"empirical"``), and exact bins sum their observed values ``v/C`` in
    the box with ``fsum``. Pooling exact counts before one division is an
    explicit arithmetic change from the former sum of per-entry quotients
    ``count/total`` and can change the last bits and a threshold decision.
    A classical result without a kept state has only marginals and returns
    ``(None, None)``, since different joint distributions can have the same
    marginals.
    """
    if not readout.available:
        return None, None
    return readout.joint(intervals), readout.kind


def _point_weights(readout):
    """Return ``(indices, weights)``: the observed valid grid points and their weights as ``analyze`` gives them.

    A kept state gives ``indices`` None and ``weights`` the level's
    ``_LevelReadout``, whose ``mode`` reads its ``(K,)*d`` grid of
    probabilities in C order, lexicographic grid order, kept or streamed, so
    no index array of the ``K**d`` points is formed; the zero entries take
    no part in ``_stall_split``. Counts and
    exact bins list each observed valid point once, with its integer count
    summed over chunks or its values ``value/C`` over C chunks added in
    observation order, so these arrays grow with the observations and not
    with ``K**d``. ``method._summarize`` selects the level's most probable
    point from these weights, so ``_stall_split`` selects a side's most
    probable point from the same numbers. A kept dense grid is reused
    directly. A streamed readout supplies the admitted passes used by the
    split's paired mode selection, and its read counter records those
    passes. The result has joint
    observations, since ``refine_box`` requires ``keep_state`` for a
    classical split.
    """
    if readout.state is not None:
        return None, readout
    return readout.pooled()


# The probability, within one refinement, that the count screen of _valley_admission admits a
# valley that sampling cannot resolve. It is a confidence policy, not a derived number. A split
# discards probability, so a sampled valley must be resolved at this level before it can cause
# one. R refinements, one per augmented-Lagrangian round, compose by the union bound to R alpha.
# Registered in docs/ENGINEERING_CONSTANTS.md ("Box refinement split confidence").
_FALSE_VALLEY_ALLOWANCE = Fraction(1, 100)


def _valley_admission(result, readout, rows, levels):
    """Return ``(admits, note)``: the resolution test a candidate valley must pass, or None when none can.

    ``admits(axis, v)`` says whether the valley at index v of an axis is
    resolved by the level's readout, and ``note`` names the screen, or says
    why no valley can pass when ``admits`` is None. It runs before ``_clearest_valley``
    ranks the valleys, since a split discards probability and a valley that
    rounding or sampling cannot resolve must not cause that. Write ``b`` for
    the smaller of the two flanking peaks (the maxima strictly below and
    above v) and ``p_v`` for the valley. Checking the smaller peak suffices
    for both flanks, since each test below only gets easier for a larger
    peak. The count screen reads the level's decoded population
    (``_LevelReadout``), so it reads no observed value again.

    Exact readout (a classical kept state, native exact probabilities or
    amplitudes). Let W be the level's point-difference window
    (``QHDAnalysis.most_probable_tie_window``, derived by
    ``method._readout_window``), V its stored valid mass and
    ``n = K**(d-1)`` the number of points in a marginal entry. For a
    classical kernel W is ``method._host_window`` of the state budget that
    the kernel recorded, which for the ``split_step`` flavor is the smaller
    budget that ``split_step.evolve`` observed on its trajectory and for the
    other flavors the Plan's budget, and it is at most the Plan's ceiling
    ``method._host_tie_window``. For native exact data W comes from the
    preparation receipts.
    The state-error derivation of W bounds the change of ``<psi, D psi>`` for
    any diagonal D of norm one, so it also bounds the difference of two
    disjoint marginal populations, whose signed projector ``P_A - P_B`` has
    norm one. For a computed state ``psi + z`` with ``||z|| <= delta``, up
    to a global phase, that change is at most ``2 delta + delta**2``, and
    evaluating the probabilities with relative error e adds at most
    ``e (1 + delta)**2``, since the two disjoint populations sum to at most
    the computed state's squared norm. This is the derivation of W for two
    points (``method._host_window``, ``method._readout_window``), so no
    factor n enters. It rests on W coming from a state-error budget. A
    producer that supplied only a bound for each pair of points would give
    ``n W`` for a marginal difference. A marginal entry adds at most n
    nonnegative stored point probabilities, and dividing by the same stored
    V gives
    ``q = (t/V)(1 + theta)`` with ``|theta| <= gamma_n``, so
    ``|q - t/V| <= rho q`` with ``rho = gamma_n/(1 - gamma_n) = n u/(1 - 2 n u)``
    and ``u = 2**-53``. The valley is admitted when
    ``b - p_v > W/V + rho (b + p_v)``, compared exactly as rationals of the
    stored numbers, so that the threshold is not rounded inward. The
    reference difference is divided by the same V, which keeps its sign. When
    W is None, or ``2 n u >= 1``, no valley is admitted. The test inherits
    the readout owner's error model, a first-order roundoff budget against
    the reference of the kernel's state budget (``method._host_state_error``),
    with no bound on spatial
    discretization, time-model error or hardware noise. The relative bound
    of the reduction assumes that the marginal sums and the division
    neither overflow nor underflow.

    Counts. Let N be the number of valid shots and ``C_L``, ``C_R`` and
    ``c_v`` the integer marginal counts of the two peaks and the valley,
    summed over the valid outcomes of every chunk without a joint table. For
    an allowance alpha per refinement (``_FALSE_VALLEY_ALLOWANCE``) over at
    most H levels (``max_levels``), d axes and K cells, the valley is admitted when
    ``min(C_L, C_R) - c_v > sqrt(2 N log(2 H d K/alpha))``. Hoeffding's
    inequality (W. Hoeffding, J. Amer. Statist. Assoc. 58 (1963) 13-30,
    doi:10.1080/01621459.1963.10500830, Theorem 1, Eq. (2.3), p. 15, for
    the mean of N independent variables in [0, 1], and Eq. (1.4), p. 13, for
    both tails) gives ``Pr(|q - p| >= eps) <= 2 exp(-2 N eps**2)`` for each
    empirical cell mass ``q = c/N``, the mean of the N indicators that a
    valid shot lands in the cell. It follows from the exponential Markov
    inequality and the bound ``E exp(t (X - E X)) <= exp(t**2/8)`` for a
    variable in [0, 1], minimized over t (Hoeffding's Sec. 4, Eqs. (4.11)
    and (4.16), p. 22). The union bound over the d K cells
    of a level and over at most H levels makes every cell simultaneously
    within ``eps = sqrt(log(2 H d K/alpha)/(2 N))`` of its population mass
    with probability at least ``1 - alpha``. On that event every true
    peak-to-valley difference is at least the empirical one minus
    ``2 eps``, although the valley was chosen after seeing the data, and the
    union over adaptively chosen levels holds when each level's shots are
    independent draws from one conditional population given the earlier
    levels. No independence between axes is needed. The statement concerns
    the population that the backend samples, so correlated shots, drift
    between pooled batches or a systematic device error against the ideal
    model need a different claim. The augmented-Lagrangian
    layer runs one refinement per round, each with its own allowance, so
    the union bound over R rounds gives probability at most R alpha of
    admitting such a valley anywhere in the run. The screen adds no
    shots and can decline a real valley when N is small. The zero tie window
    that counts use for the most probable point describes arithmetic
    equality of integers only, not a sampling error bound. The squared
    threshold is compared exactly, after enlarging ``log`` by a relative
    ``2**-40``, far above the rounding of ``math.log``, so the rounded test
    admits no valley that the exact test rejects.
    """
    import math
    import numpy as np
    from nwqlib._validation import UNIT_ROUNDOFF

    d, k = len(rows), len(rows[0])
    if readout.counts:
        # The exact integer marginal counts and valid shots of the level's decoded population.
        points = np.unravel_index(readout.flat, readout.shape)
        if isinstance(readout.values, list):
            counts = [[0] * k for _ in range(d)]
            for j in range(d):
                for i, count in zip(points[j].tolist(), readout.values, strict=True):
                    counts[j][i] += count
            valid = sum(readout.values)
        else:
            counts = []
            for j in range(d):
                axis = np.zeros(k, dtype=np.int64)
                np.add.at(axis, points[j], readout.values)
                counts.append(axis.tolist())
            valid = int(np.sum(readout.values, dtype=np.int64))
        # 2 N log(2 H d K/alpha), the squared threshold, as an exact rational upper bound.
        log_term = math.log(float(Fraction(2 * levels * d * k) / _FALSE_VALLEY_ALLOWANCE))
        bound = Fraction(2 * valid) * Fraction(log_term) * (1 + Fraction(1, 2**40))
        # The smaller flanking count peak of every interior index, from prefix/suffix maxima.
        count_peaks = [_flank_peaks(c) for c in counts]

        def admits(axis, v):
            gap = count_peaks[axis][v - 1] - counts[axis][v]
            return gap > 0 and gap * gap > bound

        return admits, "the count screen"
    window = result.most_probable_tie_window
    n, u = k ** (d - 1), Fraction(UNIT_ROUNDOFF)
    if window is None:
        return None, (f"the readout has no tie window ({result.most_probable_tie_window_unavailable}), "
                         "so no valley can be resolved")
    if 2 * n * u >= 1:
        return None, "a marginal entry sums too many points for the rounding bound (2 n u >= 1)"
    # W/V and rho = n u/(1 - 2 n u), exact rationals of the stored numbers.
    allowance, rho = Fraction(window) / Fraction(result.valid_mass), n * u / (1 - 2 * n * u)
    peaks = [_flank_peaks(p) for p in rows]

    def admits(axis, v):
        peak, dip = Fraction(peaks[axis][v - 1]), Fraction(rows[axis][v])
        return peak - dip > allowance + rho * (peak + dip)

    return admits, "the exact-readout screen"


def _flank_peaks(p):
    """Return ``min(max(p[:v]), max(p[v+1:]))`` for each interior index v = 1..len(p)-2, in order.

    For finite nonnegative entries p, ``prefix[v] = max(p[:v+1])`` and
    ``suffix[v] = max(p[v:])``, so the two peaks of an interior v are
    ``prefix[v-1]`` and ``suffix[v+1]``. Max selects an existing number
    exactly, so this changes no ratios or ties. The
    maxima are Python ``max`` over the given entries, exact for integer
    counts and float64 probabilities alike. Fewer than three entries have
    no interior valley.
    """
    prefix = list(itertools.accumulate(p, max))
    suffix = list(itertools.accumulate(reversed(p), max))[::-1]
    return [min(prefix[v - 1], suffix[v + 1]) for v in range(1, len(p) - 1)]


def _clearest_valley(rows, admits):
    """Return ``((axis, valley) or None, unscreened)``: the resolved valley with the largest smaller-peak/valley ratio.

    ``unscreened`` says whether any axis has a valley before the resolution
    test, found in the same pass, so that a level without a resolved valley
    can name its reason. The flanking peaks come from prefix and suffix
    maxima (``_flank_peaks``), O(K) per axis, and the exact Fraction ratio
    comparison and the tie rules below are unchanged.

    ``rows`` holds the conditional marginal p of each axis and ``admits`` the
    resolution test of ``_valley_admission``. An interior index v is a
    valley when ``p(v) < min(L, R)``, with ``L = max_{i<v} p(i)`` and
    ``R = max_{i>v} p(i)``. This holds exactly when some indices
    ``l < v < r`` have ``p(v) < p(l)`` and ``p(v) < p(r)``. The largest index
    attaining L is then the edge of a possibly flat local maximum facing v
    (its neighbor toward v is smaller, and nothing on its other side is
    larger), and so is the smallest index attaining R, so the axis has two
    local maxima separated by the lower index v. Boundary peaks and flat
    peak tops count.

    Among the valleys that ``admits`` passes, the chosen one has the largest
    ratio ``s(v) = min(L, R)/p(v)``, infinite for ``p(v) = 0``, on equal
    ratios the larger ``min(L, R)``, then the smaller axis and index. The
    ratio measures the multiplicative rise from the valley to the smaller
    flanking peak and ranks relative contrast. It is unchanged when a
    marginal is multiplied by a positive constant, and ``1 - p(v)/min(L, R)``
    ranks the same way. It bounds neither the discarded mass nor the
    objective in either region, and it can favor a small but sharply
    separated tail. Ratios are compared exactly as rationals of the stored
    binary64 masses. The tie rules give priority to the larger flanking
    peaks at equal contrast, including several zero valleys, and are
    otherwise only deterministic. The valley mass alone and the difference
    ``min(L, R) - p(v)`` measure other quantities, the first ranking a
    shallow dip of 0.010 between 0.011 and 0.012 (s = 1.1) above 0.10
    between 0.45 and 0.45 (s = 4.5).

    The chosen v minimizes p between peaks attaining its L and R. Any lower
    cell w between them has both peaks available, so ``min(L(w), R(w)) >=
    min(L(v), R(v))``, and it would have a larger ratio. The resolution test
    only gets easier for a larger smaller peak and a lower valley, so w
    would also pass it.
    """
    best, unscreened = None, False
    for axis, p in enumerate(rows):
        for v, minor in enumerate(_flank_peaks(p), start=1):
            unscreened = unscreened or p[v] < minor
            if p[v] < minor and admits(axis, v):
                separation = inf if p[v] == 0 else Fraction(minor) / Fraction(p[v])
                key = (separation, Fraction(minor))
                if best is None or key > best[0]:
                    best = (key, axis, v)
    return (None if best is None else best[1:]), unscreened


def _kept_regions(box, coordinates, axis, valley):
    """Return the two regions that a split at ``valley`` can keep, each side of it together with the valley cell.

    The lower one ends at the face ``_face(x_v, x_(v+1))`` on ``axis`` and
    the upper one starts at ``_face(x_(v-1), x_v)``. Every other side is the
    level box's (``_stall_split``). ``refine_box`` checks both before a split,
    since the one kept is known only after the scores.
    """
    x, (a, b) = coordinates[axis], box[axis]

    def region(side):
        return tuple(side if j == axis else other for j, other in enumerate(box))

    return region((a, _face(x[valley], x[valley + 1]))), region((_face(x[valley - 1], x[valley]), b))


def _stall_split(result, observed, rows, axis, valley, box, coordinates, score):
    """Return the ``StallSplit`` of a stalled level at the valley of ``_clearest_valley``.

    ``result`` is the level's QHD result, ``observed`` its observed valid
    grid points and their weights ``(indices, weights)`` from
    ``_point_weights``, which the caller reads and counts before any score,
    and ``rows`` its conditional marginals, ``axis`` and ``valley`` the
    valley that ``_clearest_valley`` found, ``coordinates`` the original
    coordinates of the level grid (``_coordinates``), and ``score`` maps the
    grid indices of a point to ``(F, F - C)`` there, the original objective
    and the relative objective that ``refine_box`` gives a reported grid
    point (``_tabulated_objective``). An exception of ``score`` propagates,
    and no split is made.

    A level stalls when every axis interval covers its whole axis, since
    ``_next_box`` then returns the level box and the refinement would stop
    with ``box_unchanged``. One sufficient condition is that both end cells
    of an axis hold more than ``1 - eta`` of its conditional marginal p, with
    eta the mass threshold. A contiguous interval that leaves out index 0
    has mass at most ``1 - p(0) < eta``, and one that leaves out index
    ``K - 1`` at most ``1 - p(K - 1) < eta``, so every interval of mass at
    least eta contains both ends and hence the whole axis, whatever rule
    chooses it. No choice of interval can then shrink that axis. On the
    double well of the example below, both axes met this condition at the
    level where the run without a split stopped. A greedy stall can also
    occur without it.

    For the valley of ``_clearest_valley`` an exact greedy stall gives more.
    Let T be the axis's total mass and h the mass of the end cell that the
    greedy interval of ``_axis_interval`` added last. Each strict side of the
    selected valley, the indices below it and those above it, holds at least
    h. The greedy start m is a largest cell, so when the valley lies between
    m and the last end cell e, one side holds m and the other e. Otherwise
    one side holds both, and if the other strict side held less than h, its
    peak would be
    below h, and the frontier cell w toward the last end cell, at the step
    that added the valley, would have ``p(w) <= p(v)`` while both its side
    maxima are at least h, which makes w a valley with a larger ratio, or,
    when both are zero, a larger smaller peak, and the screen of
    ``_valley_admission`` passes w whenever it passes v. So either region the
    split can keep holds at most ``T - h``, which is below eta when exact
    greedy growth reached eta only after adding the last end cell. For
    example ``(6, 1, 9, 2, 3)/21`` with eta = 4/5 stalls although one end
    cell holds only 3/21, its valley is index 1, and the kept regions hold
    1/3 or 5/7. This is an exact-arithmetic statement. The greedy rule
    compares a running binary64 sum with eta and can stall at a threshold
    that the exact sum already reached, so a kept region can hold slightly
    more than eta. The binary64 row ``(1/2, six entries 2**-55, 0,
    1/2 - 3*2**-54)`` with ``eta = 1/2 + 2**-53`` stalls, splits at index 7,
    and its lower region holds ``1/2 + 3*2**-54``. The excess is bounded for
    the stored row. The running sum before the last end cell is added takes
    at most K - 2 inexact additions, so when it is below eta, the exact sum
    of those entries, ``T - h``, and with it either exact kept sum, is below
    ``eta/(1 - gamma_(K-2))`` with ``gamma_r = r u/(1 - r u)`` and
    ``u = 2**-53``. The ``fsum`` of a kept region adds at most a factor
    ``1 + u``. These relations concern sums of the stored row, not the error
    against the exact evolution. Splitting gives up the ordinary
    mass-threshold requirement.

    The split divides the level box on one axis. ``_clearest_valley``
    chooses the axis and its valley minimum v, and states why. Each strict
    side of the valley is scored by ``score`` at its most probable joint
    grid point, two evaluations in all. That point is the lexicographically
    smallest grid point of the side with positive weight
    (``_point_weights``) within the level's tie window of the side's largest
    weight, the rule by which ``method._summarize`` selects the level's most
    probable point, with counts compared as integers. In the
    augmented-Lagrangian layer ``score`` first checks the original objective
    and constraints at the point's reported image (``_refine``), so a score
    takes part in the choice only where they are finite and real. The refinement
    continues on the side with the lower relative score F - C, on equal
    scores on the side with more probability, and then on the lower side.
    The valley cell joins the chosen side only after this choice, so the cut
    is the face of v's centered cell toward the other side,
    ``_face(x_v, x_(v+1))`` when the lower side is chosen and
    ``_face(x_(v-1), x_v)`` otherwise. Of the two
    faces of v's cell, this one discards less, exactly ``p(v)`` less, since
    the other would discard the valley cell too. It does not minimize the
    discard over all cuts or choose the side by its discard. A scored point
    need not be the most probable point of the kept region, which can gain a
    more probable joint point in the valley column. The cut is a face of
    ``_next_box``, so both regions are unions of centered cells of the level
    grid, like every box of the ordinary rule. The lower region keeps the
    level box below the cut on that axis, the upper region keeps it above
    the cut, and both keep every other side of the level box, which at a
    stall is also that side's next box. A region covers every index of the
    other axes, so its conditional probability is the sum of p over its
    indices on the split axis, whatever the correlation between axes. Only
    the chosen region continues, so each level still solves one box and
    every solve counts against ``max_levels`` and the cumulative limits.
    ``refine_box`` calls this function only when the next level can run.

    The cost of a split is the discarded region's probability, which no
    later level returns to. The split replaces only a ``box_unchanged``
    stop, so for the same realized observations every level before it is
    the level of the run without it, and since the best point is taken over
    all levels, its relative objective F - C is never larger than that run's.
    With shots this is a statement about the realized observations, not
    about a rerun with new outcomes. The two scores are selection evidence
    and do not enter the best point.

    The risk is in the continuation. The score is one objective value per
    side, at the point where the evolution put the most probability, and it
    is weak evidence about a side's minimum whenever the distribution is not
    concentrated near the minima. A narrow, deep minimum away from that
    point, between grid points or in the discarded region, can make the
    discarded region the better one without changing any value the rule
    reads. A broad basin can do the same when the distribution is spread
    out, as on a coarse grid or at a small gain, because the most probable
    point of each side can then lie far from that side's minimum. On the
    double well below with K = 4, 20 steps, T = 10, gain 1, the uniform
    initial state, three levels and exact quantum readout, the first level's
    sides scored 1.441 and 1.487, and the split kept ``y <= 0.48``, whose
    best grid value is -0.631. It gave up the region that holds the grid
    point (-0.72, 0.72), of value -0.700, and the continuous minimizer. So a
    peak score cannot justify discarding a region in general. For these reasons
    ``BoxRefinement.stall_split`` defaults to ``"none"``. A periodic grid
    is refused (``refine_box``). A periodic axis is a circle, one cut opens
    it but does not divide it into two regions, and the linear valley rule
    would use the period's seam as the second boundary, so that one peak
    across the seam would look like two peaks at the ends of the axis.

    The double well ``(2 x**2 - 1)**2 + 3 x/5 + 2 (y - 3/10)**2 + 6 x y/5``
    on ``[-1.2, 1.2]**2`` shows a stall that the split resolves, with the
    search model at potential gain 1, K = 6, T = 10, 80 steps, the quadratic
    schedule with gamma = 0.3, the uniform initial state, eta = 0.99, the
    most probable point, ``max_levels = 10``, ``max_no_improve = 10``,
    ``keep_state=True`` and classical execution. Without the split the run
    stops with ``box_unchanged`` after five levels at best-point error
    0.0826 in the infinity norm. With ``max_splits = 1`` level 5 splits x at its valley
    index 3 (conditional x marginal 0.0897, 0.870, 0.0135, 0.00277, 0.00333,
    0.0211, ratio 7.6), scores the lower side -0.690 and the upper side
    0.795, cuts above the valley at x = 0.0122, and discards the upper
    region's 0.0244 of that level's distribution. The five levels in the
    lower region reach best-point error 0.0108 after ten solves, against
    0.00325 with potential gain 8 and no split. With 160 and 320 steps the
    intervals, the split and the best point were the same. The module
    docstring gives the revision and environment of the measurements in
    this docstring.

    A different trigger, which this function does not offer, would split
    as soon as one axis is kept whole. On the double well it would split x
    at level 1, where the y side still shrank. The same valley and cut rule
    there cuts x at 0.343 and discards 0.0583 of the level's x marginal,
    0.0570 of its distribution inside the ordinary y interval, and nine
    further solves in the lower region, with the ordinary y interval,
    reached best-point error 0.000857. That trigger changes the run before
    any stall, so nothing bounds its best relative objective by the default
    run's, while splitting only at a stall keeps it never larger than the
    default run's. The measurement applied ``_stall_split`` to the level-1
    result of a one-level refinement and continued with ``refine_box`` on
    the chosen region, so it is evidence for such a variant, not a result of
    this option.
    """

    p, x, (a, b) = rows[axis], coordinates[axis], box[axis]
    indices, weights = observed
    # The tie window of the level's most probable point, or exact comparison without one (method._summarize).
    window = 0.0 if result.most_probable_tie_window is None else result.most_probable_tie_window

    def mode(first, last):
        # method._summarize among the positive weights whose index on the axis lies in [first, last]:
        # the lexicographically smallest point within the tie window of their largest weight.
        if indices is None:
            # A kept state: the readout's grid in C order (_LevelReadout.mode).
            return weights.mode(axis, first, last, window)
        side = (indices[:, axis] >= first) & (indices[:, axis] <= last) & (weights > 0)
        top = weights[side].max()
        return min(tuple(int(i) for i in row) for row in indices[side & (top - weights <= window)])

    modes = (weights.modes(axis, ((0, valley - 1), (valley + 1, len(x) - 1)), window)
             if indices is None else (mode(0, valley - 1), mode(valley + 1, len(x) - 1)))
    scores, relative = zip(*(score(indices) for indices in modes))
    sides = (fsum(p[:valley]), fsum(p[valley + 1:]))
    # The lower relative score wins, then the side with more probability, then the lower side.
    chosen = min((0, 1), key=lambda r: (relative[r], -sides[r], r))
    # The last index of the lower region. The valley cell joins the chosen side.
    last = valley if chosen == 0 else valley - 1
    coordinate = _face(x[last], x[last + 1])
    masses = (fsum(p[:last + 1]), fsum(p[last + 1:]))
    lower = tuple((a, coordinate) if j == axis else side for j, side in enumerate(box))
    upper = tuple((coordinate, b) if j == axis else side for j, side in enumerate(box))
    points = tuple(tuple(coordinates[j][i] for j, i in enumerate(indices)) for indices in modes)
    split = StallSplit(axis=axis, valley=valley, coordinate=coordinate, regions=(lower, upper),
                       point_indices=modes, points=points, scores=scores, relative_scores=relative,
                       region_masses=masses, chosen=chosen, discarded_mass=masses[1 - chosen])
    return split


def refine_box(
    problem,
    *,
    qhd,
    options,
    execution=None,
    shots=None,
    backend=None,
    seed=None,
    limits=None,
    progress=None,
    directory=None,
):
    """Refine the box of ``problem`` by repeated QHD solves, following Wu et al. arXiv:2605.12066v1, Sec. V.

    Level z solves one QHD Plan on its box ``B_z`` (``_level_problem``, by
    ``options.scaling``). Each level reports a point by ``options.point_rule``
    and records the original objective evaluated in the level's coordinates.
    For the search model this is the transformed expression ``F(a + D u)``
    evaluated at the selected unit point, before rounding its affine image
    for display. From the level's marginals it keeps an index interval on
    every axis (``_axis_interval``), records the joint mass bound
    (``_joint_mass_bound``) and, when the result has joint observations, the
    directly computed joint mass (``_joint_mass``), and forms ``B_{z+1}``
    (``_next_box``). The best point is the reported point with the least
    recorded relative objective F - C, the earlier level on ties, where C is one
    exact constant for the whole refinement (``_tabulated_objective``).
    Before each level, refinement stops at ``max_levels`` levels, an
    unchanged box, a box at its width floor (``_resolved``, and for the
    physical model the grid owner's admission, ``_level_geometry``),
    ``max_no_improve`` levels without a strict decrease of the best relative
    objective, or remaining limits that cannot fund a level
    (``_outer.round_limits``), checked in this order. With
    ``options.stall_split="best_region"`` a level whose next box equals its
    box, and which is not the last of ``max_levels``, is split instead of
    stopping the refinement while fewer than ``max_splits`` splits have been
    made (``_stall_split``), and the next level solves the chosen region.
    A split is made only when the next level could run, decided before any
    score or read for it. When the no-improvement count after this level
    reaches ``max_no_improve``, or the limits left after this level cannot
    fund one, the level records no split and the refinement stops with
    ``no_improvement`` or ``budget_exhausted``, without a valley search.
    Otherwise the level looks for a resolved valley (``_valley_admission``,
    ``_clearest_valley``). Without one it stops with ``box_unchanged``, and
    when neither region a split at the valley can keep (``_kept_regions``)
    passes the width floor it stops with ``width_floor``. When only one of
    the two regions fails the width floor and the split keeps it, the
    refinement stops with ``width_floor`` after the split. The
    no-improvement count continues across a split. An unchanged box after
    the last split stops with ``split_limit``. A stalled level that does not
    split records why in ``RefinementLevel.split_declined``. A split budget
    on a periodic
    grid raises ValueError before the first level, since one cut does not
    divide a periodic axis into two regions. A level whose tables show no
    variation or no resolvable variation, whose planning or Run raises, or
    whose result has no valid point also stops it, and completed levels are
    kept for every one of these stops (``_resolution_stop``). These stops
    end the refinement and never lead to a split. When the first
    box is one that the grid owner rejects, or the planning or Run of the
    first level raises, nothing has completed, and the original exception
    propagates (``_outer.inner_failure``). An exception from the refinement's own
    evaluation of a solved level, such as a nonreal objective value at the
    reported grid point, propagates.

    Point rules: ``_outer.grid_point`` reads the rule's grid point. For
    ``mode_or_mean``, ``_outer.mean_point`` compares F at that point with F at
    the valid-mass mean position of the level's own coordinates. Refinement
    compares its points by F - C and reports F, both of which each level
    reads from its stored tables at a grid point and evaluates term by term
    at an off-grid mean (``_tabulated_objective``). C is the
    constant term of the first level's decomposition, so an additive
    constant of F leaves every comparison unchanged when the constant and
    every coefficient of F are exact SymPy numbers, while F itself can round
    the objective's variation away next to a large constant. A physical level takes F at its grid
    coordinate or mean. A search-model level takes ``F(a + D u)`` at the
    unit point u (``_unit_objective``), for exact coefficients F at the exact
    affine image of u, and records u as ``RefinementLevel.unit_point``. Its reported point
    is that image rounded once (``_coordinates``), a display value.

    Initial state: every level plans its level problem with ``qhd``, so
    ``QHD.initial_state`` is read in the coordinates of the level problem.
    For the search model these are the unit coordinates u, so the center and
    widths of a ``GaussianState`` there are unit coordinates, fixed relative
    to every level box, and not original ones. With
    ``options.level_initial_state="best_point_gaussian"`` every level after
    the first instead starts from a Gaussian at the best point so far
    (``_best_point_gaussian``), written in the same coordinates.

    Each level plans from its own child of ``SeedSequence(seed)``
    (``_outer.plan_round``) and runs its Plan with ``prepare`` and
    ``submit``. This is an orchestration of QHD Plans, not a Method. It never
    retries a level. The best point is the best of the reported finite-grid
    points, not a continuous or global optimum.

    With ``directory`` every level Run is durable, under
    ``levels/<z>/run/``, each level writes its table-stage data once to
    ``levels/<z>/tables.json`` (``_LevelTables``), and the outer record is
    rewritten after each completed level and once more at the end
    (``_durable``), so that ``resume_box_refinement`` can continue an
    interrupted refinement. The
    outer record stores the configuration of the backend of every level
    Run, and the model of a noisy Aer backend is saved once in the
    directory.

    Args:
        problem (Optimization): Objective, ordered variables and the initial box.
        qhd (QHD): QHD configuration of every level. With
            ``options.level_initial_state="best_point_gaussian"`` a
            best-point Gaussian replaces its initial state after the first
            level.
        options (BoxRefinement): Refinement options, including the required scaling.
        execution (str | None): ``"quantum"`` (the default) or ``"classical"``, as for ``nwqlib.solve``.
        shots (int | None): Shots per level, or None for exact readout.
        backend (object | None): Backend of every level Run, as for ``nwqlib.solve``.
        seed (int | None): Nonnegative seed of a ``numpy.random.SeedSequence``.
            Each level plans with its own spawned child, as ``nwqlib.compare``
            seeds its candidates.
        limits (ExecutionLimits | None): Cumulative limits of the whole
            refinement. Each level Run receives what the earlier levels left
            (``_outer.round_limits``). None uses the ExecutionLimits defaults.
        progress (object | None): Progress callback of every level Run, as for ``nwqlib.solve``.
        directory (str | Path | None): A new directory for a durable
            refinement, or None to keep every level Run in memory. It must
            not exist, and missing parents are created.

    Returns:
        result (BoxRefinementResult): Completed levels, best point, stopping
            reason and resources, with the level QHD results in ``results``
            and the original problem in ``problem``, which ``save`` stores.
    """
    import numpy as np

    if not isinstance(problem, Optimization):
        raise TypeError("refine_box requires an Optimization; box refinement has no constraint handling")
    if not isinstance(options, BoxRefinement):
        raise TypeError("refine_box requires BoxRefinement options")
    execution, limits = check_arguments(qhd, execution, shots, seed, limits)
    check_refinement_options(options, qhd, execution)
    root = np.random.SeedSequence(seed)
    if directory is None:
        return _refine(problem, qhd, options, execution, shots, backend, root, limits, progress)
    settings = dict(problem=problem.model_dump(mode="json"), qhd=qhd.model_dump(mode="json"),
                    options=options.model_dump(mode="json"), execution=execution, shots=shots,
                    entropy=root.entropy, limits=limits.model_dump(mode="json"))
    with Directory.create(directory, "refinement", (problem.objective, problem.variables), settings,
                          backend) as outer:
        result = _refine(problem, qhd, options, execution, shots, outer.backend, root, limits, progress,
                         frontier=Frontier(outer=outer, base=outer.path))
        outer.commit(record=result)
        return result


def resume_box_refinement(directory, *, backend, progress=None, end_at_unfinishable=False):
    """Continue a durable refinement of ``refine_box(..., directory=...)``, or return it when it has ended.

    The outer record of the directory (``_durable``) names the problem, the
    QHD configuration, the options, execution, shots, root entropy,
    cumulative limits and backend configuration of the refinement, and holds
    its completed levels. As in ``constrained.resume_augmented_lagrangian``,
    ``backend`` must have the stored configuration, a noisy Aer run gets the
    noise model saved in the directory bound to a copy of ``backend``
    (``_durable.Directory.bind``), and another backend raises ValueError
    before any work. The problem comes back from ``problem.pickle`` and must
    have the content identity of the stored problem record, as in
    ``constrained.resume_augmented_lagrangian``, or ValueError is raised
    before any level Result is read or the refinement advances. Each
    completed level's Result is read from its Run with ``load_run``. The
    directory is read as the layer wrote it, and edits are not detected
    (docs/run_archives.md, "Saved folders are read-only"). The refinement
    then continues from its last completed level with
    the state that the uninterrupted run had there (``_refine``). The level
    in progress reads its persisted unscaled tables and their evaluation
    count instead of evaluating its tables again, and the refinement's
    constant C comes from the first level's persisted data when it is
    rational (``_LevelTables``). A level
    whose Run exists continues that Run with ``load_run(...).wait()``, and
    no level gets a second Run. When that Run cannot finish without new work,
    because the interruption stopped a local preparation, acquisition or
    classical evolution whose outcome nothing can retrieve, its error
    propagates with a note (``_durable.unrecoverable``), and the outer record
    keeps its committed levels unchanged. With ``end_at_unfinishable=True``
    the refinement instead ends there with ``inner_failed``, keeps its
    completed levels and records the Run's error as the failure. This option
    handles an unfinishable continuation after a Run has reopened. A
    headerless folder fails during reopen and is not handled by
    ``end_at_unfinishable``. Reopen does not remove or recreate that folder
    automatically. Before a completed round or level exists, an inner
    failure still propagates.

    A directory whose refinement has ended returns its result, with the
    level Results read from their Runs and the problem from
    ``problem.pickle`` attached, and plans, evaluates and acquires nothing.
    Either returned result can be saved to a separate result archive
    (``BoxRefinementResult.save``).

    Resuming gives the records of an uninterrupted durable refinement with
    the same root entropy, apart from the fields that differ between any two
    durable executions, which ``constrained.resume_augmented_lagrangian``
    lists with their reasons.

    Args:
        directory (str | Path): The directory of ``refine_box(..., directory=...)``.
        backend (object): The backend of the original call, whose
            configuration the directory stores. None stands for
            ``AerBackend()`` under quantum execution, as in the original call.
            For a noisy Aer refinement in a new process,
            ``AerBackend(noise_model_id=...)`` with the identity of the
            original binding, which the error for another backend names.
        progress (object | None): Progress callback of the level Runs that
            this call creates or continues, as for ``nwqlib.solve``.
        end_at_unfinishable (bool): Whether to end the refinement with
            ``inner_failed`` at a level whose Run cannot finish without new
            work, instead of raising the Run's error. False by default, which
            leaves the directory as it is for a later resume.
            This option handles an unfinishable continuation after a Run has
            reopened. A headerless folder fails during reopen and is not
            handled by ``end_at_unfinishable``. Reopen does not remove or
            recreate that folder automatically. Before a completed round or
            level exists, an inner failure still propagates.

    Returns:
        result (BoxRefinementResult): The refinement, as ``refine_box`` returns it.
    """
    import numpy as np
    from nwqlib.execution import ExecutionLimits
    from .method import QHD

    if type(end_at_unfinishable) is not bool:
        raise TypeError("end_at_unfinishable must be a bool")
    with Directory.open(directory, "refinement") as outer:
        outer.end_at_unfinishable = end_at_unfinishable
        settings = outer.settings
        backend = outer.bind(backend)
        stored = Optimization.model_validate(settings["problem"])
        problem = live_problem(stored, outer.problem())
        if problem.content_id != stored.content_id:
            raise ValueError("the saved SymPy objective and variables differ from the stored problem record")
        qhd, options = QHD.model_validate(settings["qhd"]), BoxRefinement.model_validate(settings["options"])
        execution, shots = settings["execution"], settings["shots"]
        record = outer.saved["record"]
        levels = tuple(RefinementLevel.model_validate(level) for level in
                       (outer.saved["levels"] if record is None else record["levels"]))
        results = tuple(saved_result(outer.path / "levels" / str(level.level) / "run", backend=backend)
                        for level in levels)
        if record is not None:
            result = BoxRefinementResult.model_validate(record)
            result._problem, result._results = problem, results
            return result
        frontier = Frontier(outer=outer, base=outer.path, levels=levels, results=results,
                            blocked=None if outer.saved["blocked"] is None else tuple(outer.saved["blocked"]))
        result = _refine(problem, qhd, options, execution, shots, backend,
                         np.random.SeedSequence(settings["entropy"]),
                         ExecutionLimits.model_validate(settings["limits"]), progress, frontier=frontier)
        outer.commit(record=result)
        return result


def _check_attachments(result, problem, results):
    """Check a standalone refinement against its original problem and completed level Results, without re-execution.

    ``save_archive`` and ``load_box_refinement`` apply the same checks. The
    problem must have the record's problem identity, whose content includes
    the ordered variables, bounds and units. There must be exactly one
    ``QHDAnalysis`` per completed level, in order, with the level's Result
    and Plan identities, and a trace of the level's Run. Each level Plan is
    checked as that Plan by its own identity: a physical level solves the
    original objective on its smaller box, while a search-model level
    solves its normalized transformed expression on the unit box, so a
    level Plan's objective identity need not equal the original problem's.
    The level's copied valid count, returned shots, valid mass and mode
    status must be its Result's, and a grid point must be the Result's
    point of the refinement's point rule, the candidate for
    ``best_observed`` and the most probable point otherwise. A physical
    level's point is that point's coordinates in the Result, and its
    objective the Result's table objective, since at its grid index ``_tabulated_objective``
    reads the same stored table entries and the same
    separately converted constant and ``fsum`` as ``method.objective_at``. A
    search-model level's unit point is those coordinates, and its displayed
    point their correctly rounded affine image ``fl(a_j + (b_j - a_j) u_j)``
    (``_to_box``). The search-model Result holds the normalized objective,
    so its table value is not compared with the level's original-objective
    field. No objective is evaluated again.
    """
    from .records import QHDAnalysis

    if not isinstance(problem, Optimization) or problem.content_id != result.problem_id:
        raise ValueError("the refinement's original problem differs from the problem identity of its record")
    if type(results) is not tuple or len(results) != len(result.levels):
        raise ValueError(f"the refinement needs one level Result per completed level, {len(result.levels)} in "
                         f"level order, and has {len(results) if type(results) is tuple else 'none'}")
    source = "candidate" if result.options.point_rule == "best_observed" else "most_probable"
    for level, inner in zip(result.levels, results, strict=True):
        if (not isinstance(inner, QHDAnalysis)
                or (inner.content_id, inner.plan.content_id, inner.data.trace.run_id)
                != (level.result_id, level.plan_id, level.run_id)):
            raise ValueError(f"the Result of level {level.level} differs from the Result, Plan and Run that the "
                             "record names")
        if ((level.valid_count, level.returned_shots, level.valid_mass, level.mode_status)
                != (inner.valid_count, inner.returned_shots, inner.valid_mass, inner.mode_status)):
            raise ValueError(f"level {level.level} records counts, a valid mass or a mode status that differ from "
                             "its Result")
        if level.point_indices is not None and (level.point_indices, level.point_probability) != (
                getattr(inner, f"{source}_indices"), getattr(inner, f"{source}_probability")):
            raise ValueError(f"the grid point of level {level.level} differs from its Result's "
                             f"{source.replace('_', ' ')} point")
        if level.point_indices is not None:
            coordinates = getattr(inner, f"{source}_coordinates")
            if result.options.scaling == "physical":
                objective = inner.value if source == "candidate" else inner.most_probable_objective
                if level.point != coordinates or level.objective != objective:
                    raise ValueError(f"the physical grid point or objective of level {level.level} differs "
                                     "from its Result")
            elif level.unit_point != coordinates or level.point != tuple(
                _to_box(side, u) for side, u in zip(level.box, coordinates, strict=True)
            ):
                raise ValueError(f"the search-model grid coordinates of level {level.level} differ "
                                 "from its Result or their affine image")


def save_archive(result, path):
    """Write ``result`` to the new directory ``path`` and return its path (``BoxRefinementResult.save``).

    The directory holds ``refinement.json`` (format ``qhd.refinement/3``)
    with the portable refinement record and the problem record, which keeps
    the bounds, units and identity of the original problem, and
    ``problem.pickle`` with its live SymPy objective and variables, read
    back by ``archive._SymbolicReader``. Each completed level z keeps its
    Result, saved by its own archive with its own Plan, under
    ``levels/<z>/result/``. A refinement without a completed level has no
    ``levels/`` folder. The checks of ``_check_attachments`` run first, so
    a record without its live problem or level Results is refused before
    the directory is created. ``path`` must not exist, and a failed save
    removes the directory. This is a terminal result archive, not a
    controller directory that ``resume_box_refinement`` continues.
    """
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib._limits import DEFAULT_MAX_BYTES

    problem = result.problem
    if problem is None:
        raise ValueError("saving a refinement needs its original problem, which refine_box, "
                         "resume_box_refinement and load_box_refinement attach. A record validated from JSON alone "
                         "has none, and neither has a refinement nested in an augmented-Lagrangian record, which "
                         "ConstrainedQHDResult.save stores with its run")
    _check_attachments(result, problem, result.results)
    files = ArchiveFiles(Path(path), DEFAULT_MAX_BYTES)
    files.path.mkdir()
    try:
        with files.writer("problem.pickle") as stream:
            pickle.dump((problem.objective, problem.variables), stream, protocol=5)
        files.write_json(ARCHIVE_RECORD, dict(format=ARCHIVE_FORMAT, problem="problem.pickle",
                                              problem_record=problem.model_dump(mode="json"),
                                              record=result.model_dump(mode="json")))
        for level, inner in zip(result.levels, result.results, strict=True):
            folder = files.path / "levels" / str(level.level)
            folder.mkdir(parents=True)
            inner.save(folder / "result")
    except BaseException:
        shutil.rmtree(files.path)
        raise
    return files.path


def load_box_refinement(path):
    """Load a saved standalone refinement without planning, objective evaluation or acquisition.

    Validate the original problem identity and each completed level's Result
    and Plan identities. Each Plan describes that level's own objective and
    coordinate box. Return a BoxRefinementResult with its original problem
    and level Results attached.

    The archive is the one ``BoxRefinementResult.save`` writes
    (``save_archive``). The record is validated with its content identities
    and its own validators, the SymPy objects come back through
    ``archive._SymbolicReader``, which resolves SymPy classes only, and the
    rebuilt problem must have the identity of the stored problem record and
    of the refinement record. Each level Result is loaded with
    ``load_result``, which validates it with its own Plan and archive, and
    the checks of ``_check_attachments`` follow. No backend is needed. A
    durable controller directory is refused with the function that
    continues it, and a saved augmented-Lagrangian result with
    ``load_augmented_lagrangian``.

    Args:
        path (str | Path): The directory that ``BoxRefinementResult.save`` wrote.

    Returns:
        result (BoxRefinementResult): The saved refinement, with ``problem`` and ``results`` attached.
    """
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.scientist import load_result
    from .archive import _SymbolicReader

    files = ArchiveFiles(Path(path), None)
    try:
        saved = files.read_json(ARCHIVE_RECORD)
    except FileNotFoundError as error:
        if files.file(RECORD).is_file():
            stored = files.read_json(RECORD).get("format")
            kind = next((name for name, fmt in FORMATS.items() if fmt == stored), None)
            if kind is None:
                raise FileNotFoundError(f"{path} is a durable run directory of format {stored!r} containing {RECORD}, "
                                        f"not a saved box refinement containing {ARCHIVE_RECORD}") from error
            layer = {"refinement": "box-refinement", "constrained": "augmented-Lagrangian"}[kind]
            raise FileNotFoundError(
                f"{path} is a durable {layer} directory containing {RECORD}, not a saved box refinement containing "
                f"{ARCHIVE_RECORD}. Use {RESUME[kind]}({str(path)!r}, backend=...) to continue it, then call save "
                "on its returned result to create a saved archive. An interrupted inner Run may be unable to finish "
                "from its recorded work"
            ) from error
        if files.file("constrained.json").is_file():
            raise FileNotFoundError(
                f"{path} is a saved augmented-Lagrangian result containing constrained.json, not a saved box "
                f"refinement containing {ARCHIVE_RECORD}. Open it with load_augmented_lagrangian({str(path)!r})"
            ) from error
        raise
    if saved.get("format") != ARCHIVE_FORMAT:
        raise ValueError(f"unsupported box-refinement archive format {saved.get('format')!r}, expected "
                         f"{ARCHIVE_FORMAT!r}")
    result = BoxRefinementResult.model_validate(saved["record"])
    stored = Optimization.model_validate(saved["problem_record"])
    with files.read_path(saved["problem"]).open("rb") as stream:
        problem = live_problem(stored, _SymbolicReader(stream).load())
    if problem.content_id != stored.content_id:
        raise ValueError("the saved SymPy objective and variables differ from the stored problem record")
    results = tuple(load_result(files.path / "levels" / str(level.level) / "result") for level in result.levels)
    _check_attachments(result, problem, results)
    result._problem, result._results = problem, results
    return result


def check_refinement_options(options, qhd, execution):
    """Refuse a stall split or a level initial state that the QHD configuration cannot carry out, before any level runs.

    ``refine_box`` and the augmented-Lagrangian layer
    (``constrained.solve_augmented_lagrangian``) both call it. The binary
    encoding's structured preparation prepares the uniform state only
    (``method.QHD``), so a quantum refinement with best-point Gaussians on
    that encoding needs ``initial_state_preparation="qiskit_state_preparation"``.
    """
    if (options.level_initial_state == "best_point_gaussian" and execution == "quantum"
            and qhd.encoding == "binary" and qhd.initial_state_preparation == "structured"):
        raise ValueError("level_initial_state='best_point_gaussian' starts later levels from a GaussianState, "
                         "which the binary encoding's structured preparation cannot prepare. Use "
                         "initial_state_preparation='qiskit_state_preparation'")
    splitting = options.stall_split == "best_region"
    if splitting and execution == "classical" and not qhd.keep_state:
        raise ValueError("stall_split='best_region' scores each region at its most probable joint grid point, "
                         "and a classical level keeps its joint distribution only with QHD(keep_state=True)")
    # One cut opens a periodic axis but does not divide it into two regions (_stall_split).
    if splitting and options.max_splits > 0 and qhd.boundary == "periodic":
        raise ValueError("stall_split='best_region' divides an axis with one cut, which does not divide a "
                         "periodic axis into two regions, so a periodic grid needs stall_split='none'")


def _decimal_length_bound(value):
    """Return an upper bound on the decimal characters of ``str(value)`` for an integer, sign included.

    ``30103/100000 > log10(2)``; the bound includes zero and a possible minus sign.
    """
    value = int(value)
    # 30103/100000 > log10(2); includes zero and a possible minus sign.
    return max(1, (abs(value).bit_length() * 30103) // 100000 + 1) + (value < 0)


def _rational_json_bound(value):
    """Return the JSON bytes of a level file's rational ``[str(p), str(q)]``, ``7 + digits(p) + digits(q)``, or 4 for null."""
    if value is None or not value.is_Rational:
        return 4
    return 7 + _decimal_length_bound(value.p) + _decimal_length_bound(value.q)


def _table_json_bound(specs):
    """Return an upper bound on the JSON bytes of a level file's table list from ``(support, entries)`` metadata.

    For table i of length ``n_i``, ``p_i = 8 n_i`` raw bytes and
    ``q_i = 4*((p_i + 2)//3)`` base64 characters. The FrozenArray object has
    exact compact-JSON length ``36 + sum digits(shape_j) + max(0, rank-1) + q_i``,
    counting the existing dtype/shape/data representation without reading
    or encoding entries; base64 characters are ASCII and need no JSON
    escaping. The rank-one ``SupportValues`` bound is
    ``J_fields,i <= 227 + digits(n_i) + sum_(j in support_i) digits(j)
    + max(0, len(support_i) - 1) + q_i`` for ``parent_id=None``, plus 69 if a
    parent identity is present, which this bound always allows. The 227
    includes the record framing and 32 bytes for each of three finite JSON
    floats. The default ``model_dump(mode="json")`` also includes
    ``content_id``, whose comma, key, colon and quoted 71-character identity
    add 87 bytes, so ``J_file,i = J_fields,i + 87`` (383 = 227 + 69 + 87)
    and the list has ``2 + max(0, T-1) + sum J_file,i`` bytes; ``tables=None``
    costs four.
    """
    if specs is None:
        return 4
    total = 2 + max(0, len(specs) - 1)
    for support, entries in specs:
        q = 4 * ((8 * int(entries) + 2) // 3)
        total += (383 + len(str(entries)) + sum(len(str(j)) for j in support)
                  + max(0, len(support) - 1) + q)
    return total


def _level_json_bound(specs, evaluations, offset):
    """Bound UTF-8 bytes of a qhd.refinement_level_tables/3 file from metadata.

    With ``T(specs) = _table_json_bound(specs)``, ``D(evaluations) =
    _decimal_length_bound(evaluations)`` and ``R(offset) =
    _rational_json_bound(offset)``: the compact ASCII skeleton
    ``{"format":"qhd.refinement_level_tables/3","tables":null,"evaluations":0,"offset":null}``
    has 86 bytes. Replacing its three data placeholders gives

    ``J_v3 = 86+[T(specs)-4]+[D(evaluations)-1]+[R(offset)-4]
    = 77+T(specs)+D(evaluations)+R(offset)``.

    T still includes every nested support-table record, content identity
    and base64 array. D and R keep their existing integer-width-dependent
    decimal bounds. For null tables or offset, the corresponding value is
    four bytes. No array is encoded to compute this envelope.
    """
    import json

    skeleton = dict(
        format="qhd.refinement_level_tables/3",
        tables=None, evaluations=0, offset=None,
    )
    size = len(json.dumps(
        skeleton, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode("ascii"))
    size += _table_json_bound(specs) - 4
    size += _decimal_length_bound(evaluations) - 1
    size += _rational_json_bound(offset) - 4
    return size


def _level_table_phase_bytes(d, specs, json_bytes):
    """Return ``(B_save, B_load)`` beyond other live data, the save and load peaks of a level file of J bytes.

    With ``Q = sum q_i``, ``q_max = max q_i``, ``n_max = max n_i``,
    ``P_data = sum 8 n_i`` and ``M = sum_i len(support_i)`` (empty maxima
    zero), one metadata reservation covers the final records and the
    portable/parsed tree,
    ``H_level = H0 + 10240T + [48 + 2L(bit_length(max(0, d-1)))] M``:
    6144T record/array-wrapper allowance, 2048T for portable/parsed
    dictionaries, shape/support lists and scalar/string headers, 2048T for a
    simultaneous identity-serialization tree and validator references, 48M
    for six support-reference populations, two integer-referent allowances
    for source and reconstructed supports, and H0 for fixed
    file/encoder/record bookkeeping. With ``S = max(0, J - Q)`` non-base64
    characters, ``R_scalar(S) = 4S + 64 L(4 max(1, S) + 1)`` covers those
    strings, decimal conversions and the reconstruction of at most four
    rational numerator/denominator integers (``log2(10) < 4``, so each
    decimal integer of at most S characters has at most 4S magnitude bits);
    the factor 64 is a conservative engineering allowance for
    simultaneously live integer/conversion/GCD storage under the current
    scalar codec, ``sp.Rational(int(p), int(q))``, and also covers the
    evaluation-count integer. The identity payload of one ``SupportValues``
    is bounded with a 192-byte wrapper allowance, ``J_id,i = J_file,i + 192``
    and ``I_max = max_i [2q_i + 3J_id,i + 512]`` (the entirely ASCII
    identity, the encoder law ``3wJ`` with w = 1, other scalar tokens within
    512 bytes). Then

    ``W_save = max(Q + q_max + 12J, Q + I_max) + R_scalar(S)``,
    ``W_load = max(4J, Q + max(q_max, 9 n_max), Q + I_max) + R_scalar(S)``,
    ``B_save = P_data + H_level + W_save``, ``B_load = P_data + H_level + W_load``.

    Save: the frozen tables contribute ``P_data``; the portable tree keeps
    all Q base64 characters; serializing the current array overlaps raw
    and base64 bytes, or encoded bytes and the ASCII string, within
    ``Q + q_max``; the default dump's computed ``content_id`` forms a
    separate identity serialization, ``I_max`` beyond Q; the streamed
    ``json.dump`` chunks, escaped current token and UTF-8 writing fit 12J.
    Load: ``read_text`` and parsing overlap input bytes/text and the parsed
    strings within 4J, and the complete input text dies when
    ``json.loads(path.read_text(...))`` returns; each array is decoded one
    at a time (ASCII encoding and base64 decoding overlap ``q_i + p_i``, the
    byte copy dies before ``FrozenArray`` makes its owner, and freezing
    overlaps the decoded raw ``8n_i``, the final owner and the ``n_i``
    finite mask), so with all final ``P_data`` reserved first the extra is
    ``max(q_i, 9n_i)``; ``Record._check_identity`` reserializes the current
    table and needs ``I_max`` while the parsed file and final arrays stay
    live, after the decode phase has ended. These are qualified engineering
    bounds for 64-bit CPython 3.12.14, NumPy 2.5.2, SymPy 1.14.0 and
    Pydantic 2.13.5, for files produced by this writer with the existing
    format check, not a general hostile-JSON parser bound
    or a process-RSS bound.
    """
    specs = () if specs is None else specs
    entries = sum(int(n) for _, n in specs)
    indices = sum(len(s) for s, _ in specs)
    largest = max((int(n) for _, n in specs), default=0)
    q_total = sum(4 * ((8 * int(n) + 2) // 3) for _, n in specs)
    q_max = 4 * ((8 * largest + 2) // 3)
    j = int(json_bytes)
    if j < q_total:
        raise ValueError("level file is shorter than its expected array encoding")
    referent = _integer_storage(max(0, d - 1).bit_length())
    headers = (
        65536 + 10240 * len(specs)
        + (48 + 2 * referent) * indices
    )
    scalar_chars = j - q_total
    scalar = 4 * scalar_chars + 64 * _integer_storage(4 * max(1, scalar_chars) + 1)
    resident = 8 * entries + headers
    identity = 0
    for support, n in specs:
        q = 4 * ((8 * int(n) + 2) // 3)
        record_json = (383 + len(str(n)) + sum(len(str(i)) for i in support)
                       + max(0, len(support) - 1) + q)
        identity = max(identity, 2 * q + 3 * (record_json + 192) + 512)
    save = resident + max(q_total + q_max + 12 * j, q_total + identity) + scalar
    load = resident + max(4 * j, q_total + max(q_max, 9 * largest),
                          q_total + identity) + scalar
    return save, load


def _admit_level_save(qhd, *, d, tables, evaluations, offset, held_bytes):
    """Admit a level file's save phase against ``qhd.max_bytes`` before any serialization; return its bound J."""
    specs = None if tables is None else tuple(
        (tuple(t.support), int(t.values.array.size)) for t in tables)
    j = _level_json_bound(specs, evaluations, offset)
    save, _ = _level_table_phase_bytes(d, specs, j)
    _require_bytes(qhd.max_bytes, held=held_bytes, local=save, stage="level tables save")
    return j


def _admit_level_load(qhd, *, path, d, specs, held_bytes):
    """Admit a level file's load phase against ``qhd.max_bytes`` from its size, before reading it; return that size."""
    j = path.stat().st_size
    _, load = _level_table_phase_bytes(d, specs, j)
    _require_bytes(qhd.max_bytes, held=held_bytes, local=load, stage="level tables load")
    return j


def _retained_bytes(results):
    """Return the bytes of the retained level Results' kept states, 16 per amplitude, and their Plans' tables, 8 per entry."""
    return sum(16 * inner.data.artifact(inner.artifact).array.size if inner.artifact is not None else 0
               for inner in results) + sum(8 * t.values.array.size for inner in results
                                           for t in inner.plan.reconstruction.support_values)


class _LevelTables:
    """The unscaled support tables of one durable level, persisted when its table stage completes.

    Resume needs the actual unscaled tables, not just E, c and the extrema,
    since those cannot reconstruct arbitrary unscaled entries without
    evaluation. A durable level therefore writes, once, the file
    ``tables.json`` in its level folder, next to its ``run`` folder
    (``_durable.Frontier.folder``). A version-3 refinement level file stores
    its format, unscaled support tables, table-evaluation count and rational
    reference offset. Resume reuses the tables and count, and obtains the
    reference offset from the first level's file. A null offset follows the
    symbolic recomputation path. ``_level_json_bound`` prices the fixed ASCII
    skeleton, table-record/base64 envelopes, evaluation-count digits and
    rational-offset digits without serializing array entries. The file-byte
    bound is ``77+T(specs)+D(evaluations)+R(offset)``. The support tables
    are the search model's unscaled ``records.SupportValues`` of the table
    stage (None for the physical model, whose tables live in the level
    Plan). The data stay in the level folder, not in the outer record,
    which every commit rewrites whole. The file is written to a temporary
    name and moved over with ``os.replace``, the commit rule of
    ``_durable``.

    ``load`` checks the format, validates each table as a ``SupportValues``
    record (dtype, shape and cached extrema derived from its entries) and
    reads the stored table evaluations and C as saved; ``stored_offset``
    reads C from the first level's file. The expected population of a
    resumed search-model level, which sizes the load admission, comes from
    its symbolic decomposition, formed before the load and reused by the level (``_level_problem``), with ``K**|S|`` entries per
    support S; that of the first level, read for C, from the first level's
    Plan. The decomposition itself, whose support expressions and constant
    cannot be rebuilt from the stored entries, is formed again symbolically.
    An irrational C is not stored and is obtained from a decomposition as
    before.

    Saving admits the live frozen tables, portable base64 strings, encoder
    buffers, computed record identities and scalar metadata before
    serialization. Loading admits text parsing and, one table at a time,
    base64 decoding, the new float64 owner, its finite-value mask and the
    reserialization needed to validate the supplied record identity before
    reading the file. Both phases use the level QHD Method's max_bytes with
    other live level and retained-result data included. The file's UTF-8
    length describes outer-directory storage; under the outer-directory
    contract it is outside the inner Runs' max_data_bytes.

    The save phase is ``_admit_level_save`` with the upper file length of
    ``_level_json_bound``, and the load phase ``_admit_level_load`` with the
    file's actual length, both by ``_level_table_phase_bytes``. The held
    data are the retained level Results' kept states and Plan tables
    (``_retained_bytes``), and for the first level's file read for C also
    the current level's loaded tables.
    """

    NAME = "tables.json"
    FORMAT = "qhd.refinement_level_tables/3"

    def __init__(self, tables, evaluations, offset):
        self.tables, self.evaluations, self.offset = tables, evaluations, offset

    @staticmethod
    def _rational(value):
        value = sp.sympify(value)
        return [str(value.p), str(value.q)] if value.is_Rational else None

    @classmethod
    def path(cls, folder):
        """Return the level's ``tables.json`` beside its Run folder."""
        return folder.parent / cls.NAME

    @classmethod
    def save(cls, folder, box, tables, evaluations, offset, *, qhd, held_bytes):
        """Write the level's table-stage data once, before its Run is reopened or created, after admitting it."""
        import json
        import os

        _admit_level_save(qhd, d=len(box), tables=tables, evaluations=evaluations,
                          offset=None if offset is None else sp.sympify(offset), held_bytes=held_bytes)
        path = cls.path(folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(
            format=cls.FORMAT,
            tables=None if tables is None else [t.model_dump(mode="json") for t in tables],
            evaluations=evaluations,
            offset=cls._rational(offset),
        )
        temporary = path.with_name(cls.NAME + ".partial")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        os.replace(temporary, path)

    @classmethod
    def load(cls, folder, box, *, qhd, specs, held_bytes):
        """Return the persisted data of the level in ``folder``, or None when its table stage has not completed.

        ``specs``, the expected ``(support, entries)`` of every table (None for
        a physical-model level, whose file holds no table), sizes the load
        admission.
        """
        import json

        from .records import SupportValues

        path = cls.path(folder)
        if not path.is_file():
            return None
        _admit_level_load(qhd, path=path, d=len(box), specs=specs, held_bytes=held_bytes)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != cls.FORMAT:
            raise ValueError(
                f"unsupported QHD refinement level file format {data.get('format')!r} "
                f"in {path}. This NWQLib reads only {cls.FORMAT!r}. "
                "Start a new refinement directory with this NWQLib."
            )
        items = data["tables"]
        tables = None if items is None else tuple(SupportValues.model_validate(item) for item in items)

        def rational(pair):
            return None if pair is None else sp.Rational(int(pair[0]), int(pair[1]))

        return cls(tables, data["evaluations"], rational(data["offset"]))

    @classmethod
    def stored_offset(cls, frontier, levels, persisted, initial, *, search, qhd, held_bytes):
        """Return the refinement's persisted C, from the first level's data, or None.

        C is read from the first level's file as saved. A later level reads
        that file, whose expected tables are those of the first level's Plan,
        with the current level's loaded tables held.
        """
        first = persisted if not levels else None
        if levels and frontier is not None:
            plan = frontier.results[0].plan
            specs = None if not search else tuple(
                (tuple(t.support), int(t.values.array.size)) for t in plan.reconstruction.support_values)
            held = held_bytes + (0 if persisted is None or persisted.tables is None
                                 else sum(8 * t.values.array.size for t in persisted.tables))
            first = cls.load(frontier.folder(1), initial, qhd=qhd, specs=specs, held_bytes=held)
        if first is None or first.offset is None:
            return None
        return first.offset


def _refine(problem, qhd, options, execution, shots, backend, root, limits, progress, used=None, completed=0,
            frontier=None, check=None):
    """Run the levels of ``refine_box`` from the ``numpy.random.SeedSequence`` ``root``, arguments already checked.

    Level z plans from ``root.spawn(1)[0]``, the z-th child of ``root``.
    ``refine_box`` passes ``SeedSequence(seed)``, and the augmented-Lagrangian
    layer passes the child of its round (``constrained.solve_augmented_lagrangian``),
    so a level's spawn key is ``(z - 1,)`` alone and ``(k, z - 1)`` in round k.
    ``limits`` caps the whole run, and ``used`` holds the ``_outer.USED_COUNTS``
    that the run spent before this refinement, zeros when None, so every
    level's remainder (``_outer.round_limits``) is that of the whole run.
    ``completed`` counts the rounds of the run completed before this
    refinement. An exception of a level's planning or Run ends the refinement
    as ``inner_failed`` when anything has completed, earlier rounds or
    levels, and propagates otherwise (``_outer.inner_failure``). The caller
    has checked the options (``check_refinement_options``).

    ``check``, when given, is called with the reported point of every point
    that a level compares, in the coordinates of ``RefinementLevel.point``,
    and raises ValueError when the caller's functions are not finite and
    real there: the level's grid point before the level evaluates it, for
    ``mode_or_mean`` its mean before the mean is evaluated, and at a stall
    split each side's scored grid point before its score is used
    (``_stall_split``). For the search model the reported point is the
    rounded image of ``_coordinates``, while the level's objective is
    evaluated at the unit point. The augmented-Lagrangian layer passes the
    check of its original objective and constraints
    (``constrained._level_check``) and reserves the work of these checks
    (``constrained._setup``). A failed mean check keeps the grid point and
    records ``mean_unavailable`` (``_outer.mean_point``). A failed grid-point
    or side-point check ends the refinement as an exception of the level's
    Run does, before the level's point takes part in the selection and
    before a split or a next box is decided. The stopped level's resources
    then count its Run, the objective evaluations it completed and the
    observed values it read. The level's comparison of its objective keeps
    its table and coordinate conventions. Without ``check``, as in
    ``refine_box``, no point is checked.

    With a ``_durable.Frontier`` every level Run is durable in the
    frontier's folder of its level, each level persists its table-stage
    data once beside that folder (``_LevelTables``), and the frontier commits
    the completed levels after each of them. A resumed level in progress
    reads those tables and their evaluation count instead of evaluating them again.
    A resumed refinement passes the levels it
    completed before, with their results and the stop that the last of them
    decided (``blocked``). They enter the loop state through ``advance``,
    the same update that a newly completed level makes, and their
    random-stream children are spawned and discarded, so the next level
    starts from the box, best level, no-improvement count, split count,
    spent limits and stream of the uninterrupted run. A level whose Run
    exists continues it (``_durable.reopen``). An error of that reopening,
    an error of reading or writing the level's persisted table
    data (``_LevelTables``), or an outcome of the reopened Run that only new
    work could resolve
    (``_durable.unrecoverable``), propagates instead of ending the
    refinement, because the level would otherwise need a second Run.
    """
    from nwqlib.scientist import prepare, submit

    splitting = options.stall_split == "best_region"
    search = options.scaling == "search_model"
    box = initial = tuple((float(a), float(b)) for a, b in problem.bounds)
    used = dict.fromkeys(USED_COUNTS, 0) if used is None else dict(used)
    levels, results = [], []
    best = offset = None
    stale = splits = 0
    termination = failure = stopped = blocked = None

    def advance(level, result):
        """Add a completed level to the loop state, with its counts, the best level and the no-improvement and split counts."""
        nonlocal best, stale, splits, box
        levels.append(level)
        results.append(result)
        for name in USED_COUNTS:
            used[name] += getattr(level.resources, name)
        # Strict decrease of the relative objective F - C is the improvement
        # test (_tabulated_objective). Wu et al.'s scripts count
        # (best - current)/max(|best|, 1e-30) > 1e-16 of F instead, which is a
        # strict decrease of F for |best| >= 1e-30, since adjacent binary64
        # numbers differ by a relative 2**-52/(2 - 2**-52), about 1.11e-16, or
        # more. Both tests are blind to a variation of F below the spacing of
        # binary64 numbers near a large constant of F, which F - C removes.
        if best is None or level.relative_objective < best.relative_objective:
            best, stale = level, 0
        else:
            stale += 1
        splits += level.split is not None
        box = level.next_box

    if frontier is not None:
        for level, result in zip(frontier.levels, frontier.results, strict=True):
            advance(level, result)
        blocked = frontier.blocked
        root.spawn(len(levels))
    while termination is None:
        if len(levels) == options.max_levels:
            termination = "level_limit"
        elif levels and levels[-1].next_box == levels[-1].box:
            # A stalled level with a valley whose next level could not run records that stop.
            if blocked is not None:
                termination, failure = blocked
            else:
                termination = "split_limit" if splitting and splits == options.max_splits else "box_unchanged"
        else:
            grid, coordinates, resolved = _level_geometry(problem, box, qhd, search, first=not levels)
            if not resolved:
                termination = "width_floor"
            elif stale >= options.max_no_improve:
                termination = "no_improvement"
        if termination is not None:
            break
        funded, reason = round_limits(limits, used, execution, shots)
        if funded is None:
            termination, failure = "budget_exhausted", reason
            break
        child = root.spawn(1)[0]
        folder = None if frontier is None else frontier.folder(len(levels) + 1)
        plan = prepared = run = closed = None
        reopened = False
        table_evaluations, table_reason, planning_reason, run_reason = 0, None, None, None
        # The QHD configuration of this level is the configured one, or after the first
        # level of a best-point Gaussian refinement the same with that Gaussian as its start.
        stage, level_qhd, gaussian = "initial_state", qhd, None
        try:
            if options.level_initial_state == "best_point_gaussian" and best is not None:
                gaussian = _best_point_gaussian(best, box, options.gaussian_width, search)
                level_qhd = qhd.revise(initial_state=gaussian)
            # A durable level whose table stage completed before an interruption persisted its unscaled
            # tables and their evaluation count, which resume reads instead of evaluating them again (_LevelTables).
            # A fault of that file is a fault of the durable directory, which propagates as reopen does.
            stage = "level_tables"
            # The retained level Results stay live through the level-file phases (_LevelTables).
            retained = _retained_bytes(results)
            persisted = decomposition = None
            if folder is not None and _LevelTables.path(folder).is_file():
                # The resumed search-model level's decomposition gives the expected tables before the
                # load, and the level reuses it (_level_problem).
                specs = None
                if search:
                    decomposition = _level_decomposition(problem, box, True)
                    specs = tuple((tuple(support), grid.num_grid_points ** len(support))
                                  for support in decomposition.support_expressions)
                persisted = _LevelTables.load(folder, box, qhd=level_qhd, specs=specs, held_bytes=retained)
            stored_offset = None if offset is not None else _LevelTables.stored_offset(
                frontier, levels, persisted, initial, search=search, qhd=level_qhd, held_bytes=retained)
            stage = "table"
            level_problem, tables, energy_shift, table_evaluations, decomposition = _level_problem(
                problem, box, level_qhd, options.scaling, options.gain,
                None if persisted is None or persisted.tables is None
                else (persisted.tables, persisted.evaluations, decomposition))
            if decomposition is None:
                decomposition = _level_decomposition(problem, box, search)
            if offset is None:
                # C, the exact constant term of the first level's decomposition, persisted by the
                # first level of a durable refinement (_LevelTables.offset).
                offset = stored_offset if stored_offset is not None else _exact_constant(
                    (decomposition if not levels else _level_decomposition(problem, initial, search)).constant)
            stage = "level_tables"
            if persisted is None and folder is not None:
                _LevelTables.save(
                    folder, box, tables, table_evaluations, offset,
                    qhd=level_qhd, held_bytes=retained,
                )
            if level_problem is not None:
                stage = "reopen"
                run = reopen(folder, backend=backend, progress=progress)
                stage = "planning"
                plan = plan_round(level_problem, level_qhd, execution, shots, child) if run is None else run.plan
                if tables is None:
                    # The level Plan's stored SupportValues, references to their immutable arrays.
                    tables = plan.reconstruction.support_values
            stage = "scale"
            termination = _resolution_stop(tables)
            if termination is None:
                scale = _require_scale(tables)
                reopened = run is not None
                if not reopened:
                    stage = "preparation"
                    prepared = prepare(plan, backend=backend, limits=funded, progress=progress, directory=folder)
                    run = prepared.run
                stage = "run"
                with run:
                    result = run.wait() if reopened else submit(prepared).wait()
        except Exception as error:
            outer = None if frontier is None else frontier.outer
            if stage in ("reopen", "level_tables") or stage == "run" and reopened and unrecoverable(
                    error, outer.path, end=outer.end_at_unfinishable):
                raise
            failure, termination = inner_failure(error, completed + len(levels)), "inner_failed"
            if stage == "table" and search:
                table_evaluations, table_reason = None, "the planning of the unscaled level objective raised"
            elif stage == "planning":
                planning_reason = "the planning of the level objective raised"
            elif stage == "preparation":
                # The work that the failed preparation spent, from the durable Run's journal.
                closed, run_reason = closed_trace(folder, backend)
        finally:
            # A reopened Run is closed here too when the level stopped before it ran. Closing is idempotent.
            if run is not None:
                run.close()
        # Read after the Run closed, the trace holds every charge, including
        # data recorded after the Result's own snapshot (_outer.run_counts).
        trace = closed if run is None else run.trace
        if termination is None and result.value is None:
            termination, failure = "no_valid_point", "; ".join(result.missing)
        if termination is not None:
            stopped = _level_resources(plan, trace, run_reason=run_reason, planning_reason=planning_reason,
                                       table_evaluations=table_evaluations, table_reason=table_reason)
            break
        # F at a point in the level's own coordinates, as the level tabulates it:
        # F(a + D u) at a unit point for the search model, which is F at the
        # exact affine image of the solved point when the symbolic coefficients
        # and coordinate transformation are exact, apart from subsequent
        # numerical evaluation (_unit_objective), and F(x) for the physical
        # model, together with F - C (_tabulated_objective), from the level's
        # decomposition and unscaled tables of the table stage. C is one exact
        # constant for the whole refinement, the constant term of the first
        # level's decomposition. A durable refinement resumed after its first
        # level reads it from the first level's persisted data when it is
        # rational, and otherwise decomposes the first box again to obtain it.
        level_value = _tabulated_objective(decomposition, offset, tables, grid.num_grid_points,
                                           _problem_coordinates(problem, box) if search else None)

        fields = grid_point(result, options.point_rule)
        indices, probability = fields["indices"], fields["probability"]
        own = tuple(grid.grid_value(j, i) for j, i in enumerate(indices))
        # The displayed point, the grid coordinate or the rounded image of u.
        point = tuple(coordinates[j][i] for j, i in enumerate(indices))
        if check is not None:
            try:
                check(point)
            except ValueError as error:
                # The caller's functions are not finite and real at the level's point, which therefore takes
                # no part in the selection: the level fails as one whose Run raised does.
                failure, termination = inner_failure(error, completed + len(levels)), "inner_failed"
                stopped = _level_resources(plan, trace, table_evaluations=table_evaluations,
                                           table_reason=table_reason)
                break
        value, relative = level_value(own, indices)
        evaluations, mean_unavailable = 1, None
        if options.point_rule == "mode_or_mean":

            def original(mean):
                # E[x] conditional on a valid outcome, from the marginals of the
                # level's own coordinates, evaluated there and displayed as
                # x = a + D u, rounded once, for the search model. The mean
                # competes by F - C and reports F.
                shown = tuple(_to_box(side, m) for side, m in zip(box, mean, strict=True)) if search else mean
                if check is not None:
                    # A ValueError keeps the grid point and records why (_outer.mean_point).
                    check(shown)
                at_mean, relative_at_mean = level_value(mean)
                return relative_at_mean, (at_mean, shown, mean)

            mean, evaluated, mean_unavailable = mean_point(result, relative, original)
            evaluations = 2
            if mean is not None:
                (relative, (value, point, own)), indices, probability = evaluated, None, None
        rows = [[m / result.valid_mass for m in row] for row in result.marginals.array.tolist()]
        intervals = [_axis_interval(row, options.mass_threshold) for row in rows]
        masses = tuple(mass for _, _, mass in intervals)
        intervals = tuple((first, last) for first, last, _ in intervals)
        # The level's joint observations, decoded once and reused by the joint mass, the count screen
        # and the split's point weights, with every live kept state, source table, stored initial
        # vector, marginal array and the current coordinate and marginal rows held.
        readout_results = (*results, result)
        readout_held = _retained_bytes(readout_results)
        readout_held += sum(
            vector.array.nbytes
            for inner in readout_results
            for vector in inner.plan.reconstruction.initial_amplitudes)
        readout_held += sum(inner.marginals.array.nbytes for inner in readout_results)
        if tables is not plan.reconstruction.support_values:
            readout_held += sum(t.values.array.nbytes for t in tables)
        readout_held += (80 * grid.num_variables * grid.num_grid_points
                         + 512 * grid.num_variables + 32 * len(readout_results))
        readout = _LevelReadout(result, level_qhd, readout_held)
        joint, kind = _joint_mass(readout, intervals)
        next_box, split = _next_box(box, coordinates, intervals), None
        # A split takes the place of the box_unchanged stop, which level_limit precedes. It is made
        # only when the next level could run, decided before any score or read for the split. The
        # no-improvement count after this level and the limits left after its Run need no valley,
        # and the width floor needs the two regions that a split at the valley can keep. A stalled
        # level that does not split records why in split_declined.
        declined = None
        if next_box == box and splitting:
            after = dict(used)
            for name, count in run_counts(trace)[0].items():
                if name in after:
                    after[name] += count
            funded, reason = round_limits(limits, after, execution, shots)
            if len(levels) + 1 == options.max_levels:
                declined = f"the level is the last of max_levels={options.max_levels}"
            elif splits == options.max_splits:
                declined = f"the {options.max_splits} allowed stall splits have been made"
            elif (0 if best is None or relative < best.relative_objective else stale + 1) >= options.max_no_improve:
                blocked = ("no_improvement", None)
                declined = f"the next level cannot run: no improvement for max_no_improve={options.max_no_improve} levels"
            elif funded is None:
                blocked = ("budget_exhausted", reason)
                declined = f"the next level cannot run: {reason}"
            else:
                admits, note = _valley_admission(result, readout, rows, options.max_levels)
                valley, unscreened = (None, False) if admits is None else _clearest_valley(rows, admits)
                if admits is None:
                    declined = note
                elif valley is None:
                    declined = ("no axis marginal has a valley" if not unscreened
                                else f"no valley passes {note}")
                elif not any(_level_geometry(problem, region, qhd, search, first=False)[2]
                             for region in _kept_regions(box, coordinates, *valley)):
                    blocked = ("width_floor", None)
                    declined = "the next level cannot run: neither region of the valley passes the width floor"
                else:
                    # The point weights are read, and counted, before any score (_stall_split).
                    scored, refused = 0, None

                    def grid_value(indices):
                        # The caller's check at the scored point's reported image, in the coordinates of
                        # RefinementLevel.point, before its score is used. Then F and F - C read from
                        # the level's stored tables at the point's grid index, as for the reported grid point
                        # (_tabulated_objective).
                        nonlocal scored, refused
                        if check is not None:
                            try:
                                check(tuple(coordinates[j][i] for j, i in enumerate(indices)))
                            except ValueError as error:
                                refused = error
                                raise
                        value = level_value(tuple(grid.grid_value(j, i) for j, i in enumerate(indices)), indices)
                        scored += 1
                        return value

                    try:
                        split = _stall_split(result, _point_weights(readout), rows, *valley, box, coordinates,
                                             grid_value)
                    except ValueError as error:
                        if error is not refused:
                            raise
                        # The caller's functions are not finite and real at a scored point, so no score
                        # decides a region: the level fails as one whose own point fails the check does,
                        # and its resources keep the scores completed and the values read.
                        failure, termination = inner_failure(error, completed + len(levels)), "inner_failed"
                        stopped = _level_resources(plan, trace, table_evaluations=table_evaluations,
                                                   table_reason=table_reason,
                                                   objective_evaluations=evaluations + scored, joint_mass_reads=readout.reads)
                        del readout
                        break
                    next_box = split.regions[split.chosen]
                    evaluations += 2
        level = RefinementLevel(
            level=len(levels) + 1,
            box=box,
            # The level grid's spacing, times the side length for the search model.
            spacing=tuple(float((Fraction(b) - Fraction(a)) * Fraction(grid.spacing(j))) if search
                          else grid.spacing(j) for j, (a, b) in enumerate(box)),
            energy_scale=scale,
            potential_gain=options.gain,
            energy_shift=energy_shift,
            table_magnitude=_table_magnitude(tables),
            spawn_key=tuple(child.spawn_key),
            plan_id=plan.content_id,
            result_id=result.content_id,
            run_id=trace.run_id,
            logical_width=plan.reconstruction.width,
            restricted_dimension=plan.reconstruction.restricted_dimension,
            valid_mass=result.valid_mass,
            valid_count=result.valid_count,
            returned_shots=result.returned_shots,
            # The readout's own status under every point rule. No refinement rule reads it.
            mode_status=result.mode_status,
            point_indices=indices,
            point=point,
            unit_point=own if search else None,
            point_probability=probability,
            objective=value,
            relative_objective=relative,
            mean_unavailable=mean_unavailable,
            intervals=intervals,
            axis_masses=masses,
            joint_mass_bound=_joint_mass_bound(masses),
            joint_mass=joint,
            joint_mass_kind=kind,
            next_box=next_box,
            split=split,
            split_declined=declined,
            initial_state=gaussian,
            resources=_level_resources(
                plan, trace, table_evaluations=table_evaluations,
                objective_evaluations=evaluations, joint_mass_reads=readout.reads,
            ),
        )
        del readout
        advance(level, result)
        if frontier is not None:
            frontier.outer.commit_levels(tuple(levels), blocked)
    record = BoxRefinementResult(
        problem_id=problem.content_id,
        qhd=qhd,
        options=options,
        execution=execution,
        shots=shots,
        seed_entropy=root.entropy,
        levels=tuple(levels),
        best_level=None if best is None else best.level,
        termination=termination,
        failure=failure,
        stopped_resources=stopped,
        resources=total_resources(levels, stopped),
        max_plannings=options.max_levels * (2 if search else 1),
    )
    record._problem, record._results = problem, tuple(results)
    return record
