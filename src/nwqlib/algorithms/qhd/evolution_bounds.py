"""Operator-norm splitting, schedule and coefficient bounds of one compiled QHD product.

The finite model is ``H(t) = a(t) T + b(t) V`` on the ``K**d`` grid states,
with the exact schedule functions a and b of the Method's schedule record, T
the Plan's kinetic operator on its grid (the binary64 spacings h_j read as
exact reals) and ``V = c I + sum_S diag(f_S)`` from the stored support tables
f_S and constant c. T is the finite-difference ``-Delta/2`` of the one-hot
grid or of the binary periodic grid, or with the binary encoding's
``kinetic_model="spectral"`` the periodic spectral operator with energies
``2 pi**2 q**2/L_j**2`` in its Fourier basis. The bounds say nothing about the
symbolic objective between or at the grid points, the continuum problem or an
optimization gap. For the one-hot embedding J of the grid space into the
``d K`` qubits as one excitation per register, a bound here is on
``||U_emitted J - J U_exact||``, not on invalid one-hot inputs. The binary
encoding uses every state of its ``d b`` qubits, and a bound is on the whole
register after the permutation between circuit and lexicographic variable
order (``binary.lexicographic_register_indices``). Times use the library's
convention with hbar and mass equal to one.

Step k covers the nominal interval ``[x_k, x_(k+1)]``, ``x_k = k delta``, with
the binary64 ``delta = fl(total_time/num_steps)`` and exact real products, so
the reference evolution ends at ``N delta``. It differs from ``total_time``
by the exact ``|N delta - total_time|``, at most ``u total_time`` when the
division result is normal and at most ``u total_time + N lambda/2``,
``lambda = 2**-1074``, with gradual underflow. The compiled product uses the
exponents ``alpha^_k = delta a^_k`` and ``beta^_k = delta b^_k``, exact
products of delta and the stored step weights
``QHDReconstruction.step_weights``. Four adjacent stages are bounded here,
the coefficient residual in part by a first-order estimate, each against
the next, with the physical identity phase included throughout
(Childs, Su, Tran, Wiebe and Zhu, arXiv:1912.08854v3, Sec. 5.1,
Propositions 15-16, Eqs. (145) and (152), pp. 38-39, which are
Propositions 9-10 of Phys. Rev. X 11, 011020,
doi:10.1103/PhysRevX.11.011020, for the splitting):

- time ordering: the exact time-ordered step against
  ``M_k = exp(-i (A_k T + B_k V))`` with the interval integrals A_k, B_k
  (``_schedule_terms``);
- midpoint quadrature, under the ``"midpoint"`` coefficient rule: ``M_k``
  against ``F_k = exp(-i delta (a(m_k) T + b(m_k) V))`` at the midpoint
  ``m_k = (k + 1/2) delta`` (``_schedule_terms``);
- coefficient residual: ``M_k`` (integrated) or ``F_k`` (midpoint) against
  ``exp(-i (alpha^_k T + beta^_k V))`` (``_coefficient_term``);
- splitting: that exponential against the compiled product of its potential
  and link factors (one-hot) or its potential and whole-kinetic factors with
  exact QFTs (binary), with the same exponents (``_splitting``).

The rounding of the emitted gate angles, the omission of pruned blocks, an
approximate QFT, the native phase bookkeeping and the preparation are
separate entries of the circuit's error ledger
(``resources.circuit_resources``). The operator differences telescope with
unit-norm factors, so the step bounds add over steps and stages without
any independence assumption, and a distance between unitaries is at most 2.

Every formula is evaluated outward: each quantity is formed exactly as a
rational number from the binary64 inputs and converted to binary64 upward
(``_upward``), each step's terms are rounded upward, summed exactly and
rounded upward once more, and the neighbor differences of the tables round
each binary64 difference to the next number above its magnitude, which under
round-to-nearest exceeds the exact difference. Upward conversion of each
nonnegative term followed by the exact sum of those binary64 numbers, which
are dyadic, and one final upward conversion keeps the result at or above the
exact sum, and the dyadic denominators stay below ``2**1074`` whatever the
step count. The reported numbers are therefore never below the
real-arithmetic formulas. A first-order estimate evaluated this way remains
an estimate. The work is ``O(sum_S |S| K**|S| + d K + N)`` scalar
operations, one pass over the stored tables and the steps, with no state
vector, matrix or Pauli expansion.
"""

from fractions import Fraction
from math import inf, nextafter, pi
from typing import Literal

import numpy as np

from nwqlib.core.records import Nonnegative, Record, Text
from nwqlib.operators.access import Count

# u = 2**-53 and tau = 2**-1074, the unit roundoff and the smallest subnormal
# of binary64, exact rationals for the coefficient residual.
_U = Fraction(1, 2**53)
_TAU = Fraction(1, 2**1074)


class QHDEvolutionBound(Record):
    """Operator-norm bounds on the compiled product of one QHD Plan against its finite model.

    The norm inputs and the splitting, time-ordering and midpoint-quadrature
    values are upper bounds in the spectral norm on ``domain``, evaluated
    outward (module docstring). The coefficient residual is a bound or a
    first-order estimate, as ``coefficient_status`` says. A norm input is
    None when it exceeds the binary64 range, and the bounds it enters are
    then the cap 2.

    Attributes:
        domain: ``"valid_one_hot_subspace"``, the states with one excitation
            per variable register, or ``"full_binary_register"``, every state
            of the ``d b`` binary qubits after the permutation between circuit
            and lexicographic variable order, with the physical identity
            phase included.
        reference: The finite model: encoding, kinetic model, boundary, grid
            points, spacings, step count and time step, schedule, coefficient
            rule and order.
        formula: ``"one_hot_first_order"`` for the potential factor followed
            by every link in emitted order, ``"one_hot_second_order"`` for the
            symmetric potential, odd-link, even-link product, and
            ``"binary_first_order"``, ``"binary_second_order"`` for the
            potential and whole-kinetic factors of the binary encoding with
            exact QFTs.
        commutator: C, a bound on ``||[T, V]||``.
        kinetic_nested: D_T, a bound on ``||[T, [T, V]]||``.
        potential_nested: D_V, a bound on ``||[V, [V, T]]||``.
        kinetic_norm: mu_T, a bound on ``||T||``.
        potential_norm: mu_V, a bound on ``||V||`` including the constant.
        hopping: Gamma, the sequential link commutator sum of one-hot first
            order, None for second order and for the binary encoding.
        even_nested: J_E, a bound on ``||[E, [E, O]]||`` of the even and odd
            link layers of one-hot second order, None otherwise.
        odd_nested: J_O, a bound on ``||[O, [O, E]]||``, None as J_E is.
        norm_methods: ``(input, method)`` pairs naming which bound attained
            each minimum: ``range`` for the table-range bounds C_0 and
            D_(T,0), ``neighbor`` for the neighbor-difference bounds C_edge and
            D_(V,edge), and ``commutator`` for ``2 tau C`` (D_T) and
            ``2 nu C`` (D_V). mu_T and mu_V come from the kinetic diagonal and
            the table extremes, Gamma from its closed form and J_E, J_O from
            the K-point graph.
        splitting: ``min(2, sum_k s_k)``, the product-formula bound.
        time_ordering: ``min(2, sum_k w_k)``, the time-ordering bound.
        midpoint_quadrature: ``min(2, sum_k q_k)`` under the midpoint rule,
            None under the integrated rule.
        schedule: ``min(2, sum_k (w_k + q_k))``.
        coefficient_residual: ``min(2, sum_k e_coeff,k)``, or None when a
            premise of its estimate is missing.
        coefficient_status: ``"bound"`` when every step's residual is an
            exact rational discrepancy, ``"estimate"`` when some step uses the
            first-order integral estimate, ``"unavailable"`` when the
            estimate's premise fails.
        coefficient_unavailable: Why the coefficient residual is None, or None.
        evolution: ``min(2, splitting + schedule + coefficient_residual)``,
            for the compiled product with the stored exponents against the
            exact time-ordered evolution of the finite model: a bound when
            ``evolution_status`` is ``"bound"``, conditional on the
            coefficient estimate when it is ``"conditional"``, and None
            without the coefficient residual.
        evolution_status: ``"bound"``, ``"conditional"`` when it rests on the
            coefficient estimate, or ``"unavailable"``.
        work: Scalar work: table entries reduced, support-axis comparisons,
            graph rows and steps.
    """

    domain: Literal["valid_one_hot_subspace", "full_binary_register"]
    reference: Text
    formula: Literal["one_hot_first_order", "one_hot_second_order", "binary_first_order", "binary_second_order"]
    commutator: Nonnegative | None
    kinetic_nested: Nonnegative | None
    potential_nested: Nonnegative | None
    kinetic_norm: Nonnegative | None
    potential_norm: Nonnegative | None
    hopping: Nonnegative | None
    even_nested: Nonnegative | None
    odd_nested: Nonnegative | None
    norm_methods: tuple[tuple[Text, Text], ...]
    splitting: Nonnegative
    time_ordering: Nonnegative
    midpoint_quadrature: Nonnegative | None
    schedule: Nonnegative
    coefficient_residual: Nonnegative | None
    coefficient_status: Literal["bound", "estimate", "unavailable"]
    coefficient_unavailable: Text | None
    evolution: Nonnegative | None
    evolution_status: Literal["bound", "conditional", "unavailable"]
    work: Count


def _upward(value):
    """Return the least binary64 number not below the exact rational ``value``, or inf beyond the range.

    ``float`` of a Fraction rounds to nearest, so the result moves one step
    up when that conversion landed below the exact value.
    """
    try:
        result = float(value)
    except OverflowError:
        return inf
    if Fraction(result) < value:
        result = nextafter(result, inf)
    return result


def _exact(value):
    """Return the exact rational of a binary64 bound, or None for inf, which marks an input beyond the range."""
    return None if value == inf else Fraction(value)


def _neighbor_differences(grid, tables):
    """Return, per variable j and link l, ``Delta_(j,l)``, a bound on the potential change across the link, or None.

    ``Delta_(j,pq) = sum_(S containing j) max_x |f_S(x_j = q, x) - f_S(x_j = p, x)|``
    over the other coordinates x of S bounds the change of the complete
    potential when only coordinate j moves between the linked points p and q.
    The periodic wrap link is included even if the objective jumps across it.
    Each table is read once per axis, ``O(|S| K**|S|)`` work. Every binary64
    difference is replaced by the next number above its magnitude, which
    under round-to-nearest exceeds the exact difference, and the maxima are
    added exactly. None when a difference exceeds the binary64 range, and
    the range bounds of ``_norm_inputs`` then stand alone.
    """
    k, links = grid.num_grid_points, grid.links()
    differences = [[Fraction(0)] * len(links) for _ in range(grid.num_variables)]
    for table in tables:
        values = np.asarray(table.values, dtype=float).reshape((k,) * len(table.support))
        for axis, variable in enumerate(table.support):
            moved = np.moveaxis(values, axis, 0).reshape(k, -1)
            for link, (p, q) in enumerate(links):
                with np.errstate(over="ignore"):
                    largest = float(np.max(np.nextafter(np.abs(moved[q] - moved[p]), np.inf)))
                if largest == inf:
                    return None
                differences[variable][link] += Fraction(largest)
    return differences


def _norm_inputs(grid, tables, constant, spectral=False):
    """Return the exact bounds C, D_T, D_V, mu_T, mu_V and the method that attained each minimum.

    Per variable ``||T_j - theta_j I|| <= kappa_j``. The finite-difference
    stencil of either encoding has ``theta_j = kappa_j = 1/h_j**2``, because
    ``T_j = (I - A_j/2)/h_j**2`` with the chain or cycle adjacency A_j of norm
    at most 2 (at binary K = 2 both neighbors coincide and
    ``T_j = (I - X)/h_j**2``). The binary spectral model (``spectral``) has
    the energies ``2 pi**2 q**2/L_j**2`` with the signed index q in
    ``[-K/2, K/2 - 1]`` and ``L_j = K h_j``, which lie in
    ``[0, pi**2/(2 h_j**2)]``, so ``theta_j = kappa_j = pi**2/(4 h_j**2)``.
    Mathematical pi lies below the binary64 number after ``math.pi``, which
    replaces it here, so kappa_j and every bound it enters are enclosed from
    above. The spectral operator is dense in the position basis, so it takes
    the range bounds alone. Per table, center ``m_S`` and radius
    ``v_S = (max f_S - min f_S)/2``, ``nu = sum_S v_S``, ``tau = sum_j kappa_j``
    and ``tau_S = sum_(j in S) kappa_j``. Subtracting the scalar centers
    leaves every commutator unchanged, only the T_j with j in S fail to
    commute with ``V_S``, and ``||[X, Y]|| <= 2 ||X|| ||Y||``, so

    ``C_0 = 2 sum_S v_S tau_S``, ``D_(T,0) = 4 sum_S v_S tau_S**2``

    bound ``||[T, V]||`` and ``||[T, [T, V]]||``, and applying the same
    inequality to the outer commutator gives ``D_T <= 2 tau C`` and
    ``D_V <= 2 nu C``. The finite-difference stencil adds a sharper bound
    from the neighbor differences (``_neighbor_differences``). With
    ``w_j = 1/(2 h_j**2)`` and the neighbor multiset ``N_j(p)``,

    ``C_edge = sum_j w_j max_p sum_(q in N_j(p)) Delta_(j,pq)``,
    ``D_(V,edge) = sum_j w_j max_p sum_(q in N_j(p)) Delta_(j,pq)**2``,

    because the entries of ``[T, V]`` are ``T_xy (V_y - V_x)`` and those of
    ``[V, [V, T]]`` are ``T_xy (V_x - V_y)**2``, whose absolute row sums these
    expressions bound, and the largest absolute row sum of an anti-Hermitian
    or Hermitian matrix bounds its spectral norm, because
    ``||A||_2 <= (||A||_1 ||A||_inf)**(1/2)`` (Higham,
    doi:10.1137/1.9780898718027, Sec. 6.3, Eq. (6.19), p. 113) and the
    largest absolute column and row sums of such a matrix are equal. The
    support differences are summed before squaring, which keeps the cross
    terms. The results are
    ``C = min(C_0, C_edge)``, ``D_T = min(D_(T,0), 2 tau C)`` and
    ``D_V = min(2 nu C, D_(V,edge))``. The raw norms, which the phase-sensitive
    midpoint quadrature needs, are ``mu_T = sum_j (theta_j + kappa_j)`` and
    ``mu_V = |c| + sum_S max |f_S|``.
    """
    d = grid.num_variables
    if spectral:
        # kappa_j = pi**2/(4 h_j**2) with pi enclosed from above.
        pi_up = Fraction(nextafter(pi, inf))
        kappa = [pi_up ** 2 / (4 * Fraction(grid.spacing(j)) ** 2) for j in range(d)]
    else:
        kappa = [1 / Fraction(grid.spacing(j)) ** 2 for j in range(d)]
    # w_j = 1/(2 h_j**2), the stencil's link weight, used by the finite-difference bounds only.
    weight = [1 / (2 * Fraction(grid.spacing(j)) ** 2) for j in range(d)]
    tau = sum(kappa)
    radius = {}
    for table in tables:
        # The cached exact extrema of the stored table (records.SupportValues).
        radius[table.support] = (Fraction(table.maximum) - Fraction(table.minimum)) / 2
    nu = sum(radius.values(), Fraction(0))
    c0 = 2 * sum((v * sum(kappa[j] for j in s) for s, v in radius.items()), Fraction(0))
    dt0 = 4 * sum((v * sum(kappa[j] for j in s) ** 2 for s, v in radius.items()), Fraction(0))
    differences = None if spectral else _neighbor_differences(grid, tables)
    commutator, methods = c0, [("commutator", "range")]
    potential_nested = 2 * nu * c0
    if differences is not None:
        k, links = grid.num_grid_points, grid.links()
        c_edge = d_edge = Fraction(0)
        for j in range(d):
            # The links incident to point p are its neighbor multiset N_j(p).
            first, second = [Fraction(0)] * k, [Fraction(0)] * k
            for link, (p, q) in enumerate(links):
                delta = differences[j][link]
                for point in (p, q):
                    first[point] += delta
                    second[point] += delta * delta
            c_edge += weight[j] * max(first)
            d_edge += weight[j] * max(second)
        if c_edge < c0:
            commutator, methods = c_edge, [("commutator", "neighbor")]
        potential_nested = min(2 * nu * commutator, d_edge)
    kinetic_nested = min(dt0, 2 * tau * commutator)
    methods.append(("kinetic_nested", "range" if dt0 <= 2 * tau * commutator else "commutator"))
    methods.append(("potential_nested", "commutator" if potential_nested == 2 * nu * commutator else "neighbor"))
    mu_t = 2 * tau
    # The cached exact extremum max |T_S| of each stored table (records.SupportValues.magnitude).
    mu_v = abs(Fraction(constant)) + sum((Fraction(t.magnitude) for t in tables), Fraction(0))
    return dict(commutator=commutator, kinetic_nested=kinetic_nested, potential_nested=potential_nested,
                kinetic_norm=mu_t, potential_norm=mu_v, weight=weight), tuple(methods)


def _graph(pairs):
    """Return the symmetric 0/1 adjacency of ``pairs`` as sparse rows ``{p: {q: 1}}``."""
    rows = {}
    for p, q in pairs:
        rows.setdefault(p, {})[q] = 1
        rows.setdefault(q, {})[p] = 1
    return rows


def _product(left, right):
    """Return the sparse product of two sparse integer matrices given as rows."""
    result = {}
    for p, row in left.items():
        accumulated = {}
        for m, x in row.items():
            for q, y in right.get(m, {}).items():
                accumulated[q] = accumulated.get(q, 0) + x * y
        result[p] = accumulated
    return result


def _combine(left, right, sign):
    """Return ``left + sign * right`` for sparse integer matrices given as rows."""
    result = {p: dict(row) for p, row in left.items()}
    for p, row in right.items():
        target = result.setdefault(p, {})
        for q, y in row.items():
            target[q] = target.get(q, 0) + sign * y
    return result


def _nested_row_sum(outer, inner):
    """Return ``max_p sum_q |[F, [F, G]]_pq|`` for the adjacencies F = outer and G = inner.

    ``[F, [F, G]] = F F G - 2 F G F + G F F``, formed from sparse rows with
    equal destinations combined before absolute values, so exact
    cancellations such as the commuting layers of the periodic K = 4 cycle
    stay exact. A layer of disjoint links has at most one entry per row, so
    each row has a constant number of paths and the work is O(K).
    """
    ffg = _product(outer, _product(outer, inner))
    fgf = _product(outer, _product(inner, outer))
    gff = _product(inner, _product(outer, outer))
    nested = _combine(_combine(ffg, fgf, -2), gff, 1)
    return max((sum(abs(x) for x in row.values()) for row in nested.values()), default=0)


def _graph_constants(grid):
    """Return ``(r_E, r_O)``, the row-sum bounds of the nested layer commutators on the K-point graph.

    The second-order compiler applies the odd-indexed links for dt/2, the
    even-indexed links for dt and the odd ones again (``kinetic.KineticCompiler``),
    so the odd layer O is the outer factor. With the unit adjacencies
    ``E_0``, ``O_0`` of the two layers, ``r_E = max_p sum_q |[E_0, [E_0, O_0]]_pq|``
    and ``r_O`` likewise with the layers swapped. The layers of variable j
    are ``-w_j E_0`` and ``-w_j O_0``, different variables commute, and the
    nested commutators are real symmetric, so ``J_E = r_E sum_j w_j**3`` and
    ``J_O = r_O sum_j w_j**3`` bound their spectral norms. The sign of the
    layers cancels in the norm, and ``r_E, r_O <= 4`` because each nonempty
    matching has norm at most one.
    """
    links = grid.links()
    odd, even = _graph(links[1::2]), _graph(links[0::2])
    return _nested_row_sum(even, odd), _nested_row_sum(odd, even)


def evolution_bound(plan):
    """Return the ``QHDEvolutionBound`` of a Plan whose step blocks were compiled, in either encoding.

    Splitting (Childs et al., arXiv:1912.08854v3, Eqs. (145) and (152)). For
    dimensionless Hermitian generators G_j in application order and
    ``R_j = sum_(q > j) G_q``, the first-order product errs by at most
    ``(1/2) sum_j ||[R_j, G_j]||`` and the symmetric second-order product by
    at most ``(1/12) sum_j ||[R_j, [R_j, G_j]]|| + (1/24) sum_j ||[G_j, [G_j, R_j]]||``.
    Unitarity makes these valid with no exponential growth factor.

    First order applies ``beta_k V`` and then every link generator
    ``alpha_k L_(j,e)`` in emitted order, with ``L_(j,e) = -w_j (|p><q| + |q><p|)``
    of norm ``w_j = 1/(2 h_j**2)``, the restriction of
    ``-(XX + YY)/(4 h_j**2)``. XX and YY on one link commute, so the fused
    hopping gate adds no split, and all projector blocks commute. The
    potential factor against all links gives ``|alpha_k beta_k| C/2``, and
    the links among themselves ``alpha_k**2 Gamma/2`` with
    ``Gamma = sum_j sum_e ||[sum_(f > e) L_(j,f), L_(j,e)]||``. On a chain
    each link but the last has one later adjacent link, whose commutator has
    norm ``w_j**2``. On an even cycle the first link has two later adjacent
    links, which give two disjoint skew-symmetric 2-by-2 blocks of norm
    ``w_j**2`` together, and every other nonfinal link has one. Disjoint links
    and different variables commute, so ``Gamma = (K - 2) sum_j w_j**2`` on
    the Dirichlet chain and ``(K - 1) sum_j w_j**2`` on the periodic cycle
    (zero for K = 2), and

    ``s_k = |alpha_k beta_k| C/2 + alpha_k**2 Gamma/2``.

    Second order applies ``exp(-i beta_k V/2) exp(-i alpha_k O/2)
    exp(-i alpha_k E) exp(-i alpha_k O/2) exp(-i beta_k V/2)`` after regrouping
    the per-variable layers, which commute across variables, for the odd and
    even layer sums O and E. The symmetric bound for the three generators
    ``beta_k V``, ``alpha_k O`` and ``alpha_k E`` gives

    ``s_k = alpha_k**2 |beta_k| D_T/12 + |alpha_k| beta_k**2 D_V/24 + |alpha_k|**3 (J_E/12 + J_O/24)``,

    where the first two terms hold every kinetic/potential commutator,
    since ``O + E`` is T up to an identity, and the last the internal link
    split, which a bound from ``[T, V]`` alone would miss (it is nonzero for a
    constant V). ``_norm_inputs`` gives C, D_T, D_V and ``_graph_constants``
    the layer constants.

    Binary encoding. An exact Fourier conjugation applies each variable's
    kinetic factor ``F^dagger exp(-i alpha_k diag(E_j)) F`` whole
    (``binary.compile_binary_steps``), and the bit-reversal relabeling
    conjugates the phase table as well, so it leaves ``T_j`` unchanged.
    Kinetic factors on different variables commute, and so do the potential
    tables. First order therefore has the two generators ``beta_k V`` and
    ``alpha_k T``, and second order the symmetric ``beta_k V/2``,
    ``alpha_k T``, ``beta_k V/2``, which give

    ``s_k = |alpha_k beta_k| C/2`` and
    ``s_k = alpha_k**2 |beta_k| D_T/12 + |alpha_k| beta_k**2 D_V/24``,

    with no link term, K = 2 included. The dense and unpruned Walsh
    diagonals implement the same factors. An approximate QFT, pruning and
    the rounding of the computed angles are later ledger entries.

    Schedule and coefficients. ``_schedule_terms`` gives ``w_k`` and ``q_k``
    and ``_coefficient_term`` the residual ``e_coeff,k``. The totals are
    ``splitting = min(2, sum_k s_k)``, ``schedule = min(2, sum_k (w_k + q_k))``,
    ``coefficient_residual = min(2, sum_k e_coeff,k)`` and
    ``evolution = min(2, splitting + schedule + coefficient_residual)``, which
    is conditional on the coefficient estimate where one is used. For the
    fixed smooth schedules the schedule bound falls as ``N**-2`` at fixed
    final time, first-order splitting as ``N**-1`` and second-order splitting
    as ``N**-2``. A small schedule parameter s or a fine grid can make the
    bounds large, and the cap 2 is then valid but uninformative.

    Raises:
        ValueError: The Plan has no compiled step blocks, so there is no
            product to bound.
    """
    from .method import _grid

    method, r = plan.method, plan.reconstruction
    if not r.compact_schedule_selected:
        raise ValueError("the evolution bound needs a Plan with compiled step blocks: quantum execution "
                         "or theory_flavor='ir_product'")
    grid = _grid(plan)
    d, k = grid.num_variables, grid.num_grid_points
    binary = method.encoding == "binary"
    inputs, methods = _norm_inputs(grid, r.support_values, r.constant, spectral=method.kinetic_model == "spectral")
    weight = inputs.pop("weight")
    first_order = method.trotter_order == 1
    if binary:
        hopping = even_nested = odd_nested = None
    elif first_order:
        count = k - 2 if method.boundary == "dirichlet" else k - 1
        hopping, even_nested, odd_nested = count * sum(w * w for w in weight), None, None
    else:
        r_e, r_o = _graph_constants(grid)
        cubes = sum(w ** 3 for w in weight)
        hopping, even_nested, odd_nested = None, r_e * cubes, r_o * cubes
    bounds = dict(inputs, hopping=hopping, even_nested=even_nested, odd_nested=odd_nested)
    floats = {name: None if value is None else _upward(value) for name, value in bounds.items()}
    exact = {name: None if value is None else _exact(value) for name, value in floats.items()}
    delta = method.total_time / method.num_steps
    dt = Fraction(delta)
    midpoint = method.coefficient_rule == "midpoint"
    # Upward-rounded binary64 step terms summed exactly, with None marking a term beyond the range.
    sums = {"split": Fraction(0), "time": Fraction(0), "quadrature": Fraction(0), "coefficient": Fraction(0)}
    estimated, missing = False, None
    for step, (_time, kinetic_weight, potential_weight) in enumerate(r.step_weights):
        # The stored exponents alpha^_k = dt a^_k and beta^_k = dt b^_k, exact products.
        alpha, beta = abs(dt * Fraction(kinetic_weight)), abs(dt * Fraction(potential_weight))
        timing, quadrature = _schedule_terms(method.schedule, step * dt, (step + 1) * dt, exact, midpoint)
        coefficient, note = _coefficient_term(method.schedule, midpoint, step, delta, kinetic_weight,
                                              potential_weight, exact)
        if coefficient is None:
            missing = missing or note
        estimated = estimated or note is True
        terms = {"split": _splitting(alpha, beta, exact, first_order, binary), "time": timing,
                 "quadrature": quadrature, "coefficient": coefficient}
        for name, term in terms.items():
            if sums[name] is not None and term is not None:
                up = _upward(term)
                sums[name] = None if up == inf else sums[name] + Fraction(up)
            elif name != "quadrature" or midpoint:
                sums[name] = None
    capped = {name: 2.0 if total is None else min(2.0, _upward(total)) for name, total in sums.items()}
    schedule_sum = None if sums["time"] is None or sums["quadrature"] is None else sums["time"] + sums["quadrature"]
    schedule = 2.0 if schedule_sum is None else min(2.0, _upward(schedule_sum))
    coefficient = None if missing else capped["coefficient"]
    evolution = None if missing else min(2.0, _upward(Fraction(capped["split"]) + Fraction(schedule)
                                                      + Fraction(coefficient)))
    spacings = ", ".join(format(grid.spacing(j), ".17g") for j in range(d))
    model = (f"binary periodic {method.kinetic_model.replace('_', '-')} kinetic model with exact QFTs" if binary
             else f"one-hot {method.boundary} finite-difference kinetic model")
    reference = (f"{model}, K = {k} per variable, spacings "
                 f"{spacings}, {method.num_steps} steps of dt = {delta!r} ending at the exact num_steps*dt, "
                 f"{method.schedule.kind} schedule, {method.coefficient_rule} coefficient rule, order "
                 f"{method.trotter_order}, stored support tables, constant and step weights")
    work = (sum(len(t.values) * (1 + len(t.support)) for t in r.support_values) + d * k + len(r.step_weights))
    return QHDEvolutionBound(
        reference=reference,
        domain="full_binary_register" if binary else "valid_one_hot_subspace",
        formula=f"{'binary' if binary else 'one_hot'}_{'first' if first_order else 'second'}_order",
        **{name: None if value == inf else value for name, value in floats.items()},
        norm_methods=methods,
        splitting=capped["split"],
        time_ordering=capped["time"],
        midpoint_quadrature=capped["quadrature"] if midpoint else None,
        schedule=schedule,
        coefficient_residual=coefficient,
        coefficient_status="unavailable" if missing else "estimate" if estimated else "bound",
        coefficient_unavailable=missing or None,
        evolution=evolution,
        evolution_status="unavailable" if missing else "conditional" if estimated else "bound",
        work=work,
    )


def _splitting(alpha, beta, bounds, first_order, binary=False):
    """Return the exact splitting bound s_k of one step (``evolution_bound``), or None when an input is beyond the range.

    ``binary`` selects the potential and whole-kinetic factors of the binary
    encoding, which have no internal link term.
    """
    if binary:
        c, d_t, d_v = bounds["commutator"], bounds["kinetic_nested"], bounds["potential_nested"]
        if first_order:
            # s_k = |alpha_k beta_k| C/2
            return None if c is None else alpha * beta * c / 2
        # s_k = alpha_k**2 |beta_k| D_T/12 + |alpha_k| beta_k**2 D_V/24
        return None if d_t is None or d_v is None else alpha ** 2 * beta * d_t / 12 + alpha * beta ** 2 * d_v / 24
    if first_order:
        c, gamma = bounds["commutator"], bounds["hopping"]
        if c is None or gamma is None:
            return None
        # s_k = |alpha beta| C/2 + alpha**2 Gamma/2
        return alpha * beta * c / 2 + alpha * alpha * gamma / 2
    d_t, d_v, j_e, j_o = (bounds[name] for name in ("kinetic_nested", "potential_nested", "even_nested", "odd_nested"))
    if None in (d_t, d_v, j_e, j_o):
        return None
    # s_k = alpha**2 |beta| D_T/12 + |alpha| beta**2 D_V/24 + |alpha|**3 (J_E/12 + J_O/24)
    return alpha * alpha * beta * d_t / 12 + alpha * beta * beta * d_v / 24 + alpha ** 3 * (j_e / 12 + j_o / 24)


def _schedule_terms(schedule, lower, upper, bounds, midpoint):
    """Return the exact ``(w_k, q_k)`` of one step, q_k None under the integrated rule.

    Time ordering. For the exact propagator ``U_k`` of ``H(t)`` over the step
    and ``M_k = exp(-i (A_k T + B_k V))``, differentiating
    ``W(t) = exp(-i int_l^t H(u) du)`` and applying Duhamel's identity gives
    ``||U_k - M_k|| <= (1/2) int_l^r du int_l^u dv ||[H(u), H(v)]||``: with
    ``Z(t) = -i int_l^t H``, the right logarithmic derivative
    ``W' W^dagger = int_0^1 exp(s Z) Z' exp(-s Z) ds`` differs from Z' by at
    most ``||[Z, Z']||/2``, and bounding ``[Z, Z']`` by its time integral
    gives the double integral. It is a finite remainder bound, not a
    truncated Magnus series. Since
    ``[H(u), H(v)] = (a(u) b(v) - b(u) a(v)) [T, V]`` and
    ``|a(u) b(v) - b(u) a(v)| <= (a* B1 + b* A1) |u - v|`` for the bounds of
    ``schedule.derivative_bounds``, the double integral of ``|u - v|`` over
    the triangle, ``delta**3/6``, gives

    ``w_k = delta**3 (a* B1 + b* A1) C/12``,

    whose local order agrees with the leading Magnus term
    ``delta**3 (a b' - a' b) [T, V]/12``.

    For finite-dimensional Hermitian T,V and nonnegative integrable a,b,
    ``|a(u)b(v) - b(u)a(v)| <= a(u)b(v) + b(u)a(v)``. This latter function
    is symmetric under u,v. Its square-domain integral is ``2 A_k B_k``,
    so its triangular integral is ``A_k B_k``. Thus the alternative local
    bound is ``min(2, C A_k B_k/2)``. Both bounds concern the same exact
    interval and finite model. Take their minimum before summing steps,
    then combine with the existing quadrature, coefficient and splitting
    terms. A minimum of upper bounds remains an upper bound. The
    common-phase terms commute and do not alter C.

    Certified exact integrals are available for both weights of
    ShiftedCubicSchedule and for QuadraticSchedule(gamma=0). Keep the
    derivative formula on the other branches: an estimated integral is
    not an upper endpoint. The minimum applies to both coefficient rules,
    because time ordering compares the exact interval evolution with its
    exact first integral before midpoint quadrature is considered.

    Midpoint quadrature. ``||M_k - F_k|| <= |A_k - delta a(m_k)| mu_T +
    |B_k - delta b(m_k)| mu_V`` by the exponential perturbation inequality
    ``||exp(-i X) - exp(-i Y)|| <= ||X - Y||`` for Hermitian X and Y, which
    follows from the identity ``exp(-i X) - exp(-i Y) = -i int_0^1
    exp(-i (1 - s) X) (X - Y) exp(-i s Y) ds`` with unitary factors. The
    midpoint rule's remainder ``|int f - delta f(m)| <= delta**3
    max |f''|/24`` then gives ``q_k = delta**3 (A2 mu_T + B2 mu_V)/24``. It uses
    the raw norms because the identity parts carry phase. A constant
    objective still contributes a phase error when its schedule integral is
    replaced by a midpoint value. The quadratic schedule with gamma = 0 gives
    ``w_k = q_k = 0`` exactly.
    """
    a_max, b_max, a1, a2, b1, b2 = schedule.derivative_bounds(lower, upper)
    delta = upper - lower
    c = bounds["commutator"]
    timing = None if c is None else delta ** 3 * (a_max * b1 + b_max * a1) * c / 12
    kinetic_integral, potential_integral = schedule.exact_integrals(lower, upper)
    if c is not None and kinetic_integral is not None and potential_integral is not None:
        # The nonnegative-weight triangular integral is at most A_k B_k.
        integral_bound = min(Fraction(2), c * kinetic_integral * potential_integral / 2)
        timing = integral_bound if timing is None else min(timing, integral_bound)
    if not midpoint:
        return timing, None
    mu_t, mu_v = bounds["kinetic_norm"], bounds["potential_norm"]
    quadrature = None if mu_t is None or mu_v is None else delta ** 3 * (a2 * mu_t + b2 * mu_v) / 24
    return timing, quadrature


def _coefficient_term(schedule, midpoint, step, delta, kinetic_weight, potential_weight, bounds):
    """Return ``(e_coeff,k, note)`` for one step: the residual as an exact rational, and whether it is estimated.

    ``e_coeff,k = eps_a mu_T + eps_b mu_V`` bounds the change from the
    reference exponential of the step, ``M_k`` (integrated) or ``F_k``
    (midpoint), to ``exp(-i (alpha^_k T + beta^_k V))`` by the exponential
    perturbation inequality. The raw norms enter because a coefficient error
    also changes the identity phase. The residual compares schedule stages
    only, so it does not overlap the identity-phase ledger, which compares
    the stored-weight model with its rounded phase. With ``f`` standing for a
    or b and ``c^ = delta f^`` the exact product with the stored weight:

    Midpoint rule. ``eps_f = |delta f^ - delta f(m_k)|`` at the exact
    nominal midpoint ``m_k = (k + 1/2) delta``. The admitted point formulas
    are rational (``exact_weights``), so the discrepancy is exact and covers
    the stored evaluation time, the point-value arithmetic and any underflow
    of the stored weight.

    Integrated rule. Where the nominal integral ``I`` over
    ``[k delta, (k + 1) delta]`` is rational (``exact_integrals``: every
    potential integral, the shifted-cubic kinetic integral and the quadratic
    one at gamma = 0), ``eps_f = |delta f^ - I|`` exactly, which covers the
    rounded endpoints, the integral evaluation and the division into a
    weight. Otherwise, for the quadratic kinetic integral at gamma > 0 and
    the cubic one, the routine integrates over the rounded endpoints
    ``xbar_k = fl(k delta)`` (``schedules.step_weights``), its documented
    error is ``|I^ - I_e| <= C u I_e + 2 tau`` to first order, the division
    into the stored weight adds ``u |I^| + delta tau/2``, and the endpoint
    shift adds ``R = f* (|xbar_k - x_k| + |xbar_(k+1) - x_(k+1)|)``, with
    f* = a at the left end of the hull of both intervals because a is
    positive and nonincreasing. So

    ``eps_a ~ (C + 1) u I* + 2 tau + delta tau/2 + R``, ``I* = (xbar_(k+1) - xbar_k) f*``,

    with C = ``schedule.kinetic_integral_roundoff``, a first-order estimate
    under the platform assumptions of ``schedules.py``. The cubic constant
    holds on ``s in [1e-8, 1e8]`` and ``t <= 1e4``, and outside that range the
    residual is unavailable.

    Returns:
        ``(value, False)`` for an exact residual, ``(value, True)`` when the
        kinetic integral uses the estimate, and ``(None, reason)`` when a
        premise is missing.
    """
    mu_t, mu_v = bounds["kinetic_norm"], bounds["potential_norm"]
    if mu_t is None or mu_v is None:
        # A distance between unitaries is at most 2, the value an unbounded norm leaves.
        return Fraction(2), False
    dt = Fraction(delta)
    lower, upper = step * dt, (step + 1) * dt
    alpha, beta = dt * Fraction(kinetic_weight), dt * Fraction(potential_weight)
    if midpoint:
        a, b = schedule.exact_weights((lower + upper) / 2)
        # e = |alpha^ - dt a(m_k)| mu_T + |beta^ - dt b(m_k)| mu_V, exact.
        return abs(alpha - dt * a) * mu_t + abs(beta - dt * b) * mu_v, False
    kinetic, potential = schedule.exact_integrals(lower, upper)
    potential_term = abs(beta - potential) * mu_v
    if kinetic is not None:
        return abs(alpha - kinetic) * mu_t + potential_term, False
    rounded = (Fraction(step * delta), Fraction((step + 1) * delta))
    span = getattr(schedule, "kinetic_integral_range", None)
    if span is not None and not (span[0] <= schedule.s <= span[1] and rounded[1] <= span[2]):
        return None, (f"the {schedule.kind} kinetic integral's roundoff constant holds for s in [{span[0]:g}, "
                      f"{span[1]:g}] and t <= {span[2]:g}")
    largest = schedule.exact_weights(min(lower, rounded[0]))[0]
    # eps_a ~ (C + 1) u I* + 2 tau + dt tau/2 + R, with I* = (xbar_(k+1) - xbar_k) f*.
    integral = (rounded[1] - rounded[0]) * largest
    shift = largest * (abs(rounded[0] - lower) + abs(rounded[1] - upper))
    kinetic_error = (schedule.kinetic_integral_roundoff + 1) * _U * integral + 2 * _TAU + dt * _TAU / 2 + shift
    return kinetic_error * mu_t + potential_term, True
