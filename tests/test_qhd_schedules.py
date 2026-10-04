"""QHD schedule records against 50-digit formulas and the coefficient rule against independent propagators."""

import math

import mpmath as mp
import pytest
import sympy as sp

from nwqlib._validation import UNIT_ROUNDOFF as U
from nwqlib.algorithms.qhd import (
    QHD, CubicSchedule, QuadraticSchedule, ShiftedCubicSchedule, UniformState,
)
from nwqlib.algorithms.qhd import method as owner
from nwqlib.problems import Optimization
from nwqlib.scientist import plan, solve

TAU = math.ulp(0.0)


@pytest.fixture(autouse=True)
def fifty_digits():
    """Evaluate every mpmath expression of these tests at 50 digits."""
    with mp.workdps(50):
        yield


def _formulas(schedule):
    """Return a(t), b(t) and the time scale of a's early variation, written from each formula."""
    if schedule.kind == "quadratic":
        gamma = mp.mpf(schedule.gamma)
        return (lambda t: 1 / (1 + gamma * t**2), lambda t: 1 + gamma * t**2,
                1 / mp.sqrt(gamma) if gamma else None)
    s = mp.mpf(schedule.s)
    if schedule.kind == "cubic":
        return lambda t: 2 / (s + t**3), lambda t: 2 * t**3, mp.cbrt(s)
    return lambda t: (2 / (s + t)) ** 3, lambda t: 2 * t**3, s


def _integral(f, t0, t1, scale=None):
    """Return ∫ f over [t0, t1] by mpmath.quad, with the binary64 endpoints as exact reals.

    Knots at ``scale * 8**j``, j >= -2, resolve the early peak of a(t). Each
    piece is divided by its midpoint value so that quad's absolute stopping
    rule cannot drop a tiny integral, and quad's own error estimate must stay
    far below u.
    """
    a, b = mp.mpf(t0), mp.mpf(t1)
    knots = [a]
    if scale is not None:
        knot = scale / 64
        while knot < b:
            if knot > a:
                knots.append(knot)
            knot *= 8
    knots.append(b)
    total = mp.mpf(0)
    for left, right in zip(knots, knots[1:]):
        h, center = right - left, f((left + right) / 2)
        value, error = mp.quad(lambda q: f(left + h * q) / center, [0, 1], error=True)
        assert error <= mp.mpf(10) ** -30 * abs(value)
        total += h * center * value
    return total


def _label(schedule):
    """Short test id, for example cubic-0.5."""
    return f"{schedule.kind}-{schedule.gamma if schedule.kind == 'quadratic' else schedule.s:g}"


def _assert_within(value, exact, bound):
    """Assert the mixed bound ``|value - exact| <= C u exact + 2 tau`` of schedules.py."""
    assert abs(mp.mpf(value) - exact) <= bound * U * exact + 2 * TAU, (value, exact)


SCHEDULES = [QuadraticSchedule(gamma=0.3), QuadraticSchedule(gamma=1e4), CubicSchedule(s=1e-3),
             CubicSchedule(s=1.0), ShiftedCubicSchedule(s=2e-4), ShiftedCubicSchedule(s=0.5)]


@pytest.mark.parametrize("schedule", [QuadraticSchedule(gamma=0.0), *SCHEDULES], ids=_label)
def test_point_values_equal_their_formulas(schedule):
    """a(t) and b(t) against each formula evaluated at 50 digits.

    The relative bounds (4u and 3u quadratic, 4u and 2u cubic, 8u and 2u
    shifted cubic) are the roundings counted in the docstrings of the
    ``kinetic_weight`` and ``potential_weight`` methods in schedules.py.
    """
    a, b, _scale = _formulas(schedule)
    bounds = {"quadratic": (4, 3), "cubic": (4, 2), "shifted_cubic": (8, 2)}[schedule.kind]
    for t in (0.0, 1e-7, 0.05, 0.37, 1.0, 2.5, 9.75):
        for value, exact, bound in ((schedule.kinetic_weight(t), a(mp.mpf(t)), bounds[0]),
                                    (schedule.potential_weight(t), b(mp.mpf(t)), bounds[1])):
            assert abs(mp.mpf(value) - exact) <= bound * U * exact


INTERVALS = [
    (0.0, 1.0), (0.3, 2.7), (1.0, 10.0), (0.0, 10.0),
    # Nearly coincident endpoints, including t0 near 0 where a is largest for the cubic forms.
    *((t0, t0 * (1 + 1e-12)) for t0 in (1e-6, 1e-3, 0.7, 9.0)),
    (9.0, math.nextafter(9.0, math.inf)), (0.0, 1e-300),
]
# First-order relative bounds C of (A, B) in units of u, derived in the
# integral docstrings of schedules.py for gamma and s in [1e-8, 1e8] and t in
# [0, 1e4].
BOUNDS = {"quadratic": (20, 16), "cubic": (48, 16), "shifted_cubic": (16, 16)}


@pytest.mark.parametrize("schedule", SCHEDULES, ids=_label)
def test_interval_integrals_match_high_precision_quadrature(schedule):
    a, b, scale = _formulas(schedule)
    kinetic, potential = BOUNDS[schedule.kind]
    for t0, t1 in INTERVALS:
        _assert_within(schedule.kinetic_integral(t0, t1), _integral(a, t0, t1, scale), kinetic)
        _assert_within(schedule.potential_integral(t0, t1), _integral(b, t0, t1), potential)
    if schedule.kind == "shifted_cubic":
        # The first interval [0, s] has A = 4 (1/s**2 - 1/(4 s**2)) = 3/s**2 exactly.
        _assert_within(schedule.kinetic_integral(0.0, schedule.s), 3 / mp.mpf(schedule.s) ** 2, 16)


def test_integrals_at_zero_rate_and_beyond_the_ordinary_range():
    """gamma = 0 gives exactly ``t1 - t0`` (bound 1u). With gamma = 1e300, ``1 + gamma t0 t1``
    or ``sqrt(gamma) (t1 - t0)`` overflows, so quadratic A takes its exponent-scaled branch,
    whose bound is 32u (``QuadraticSchedule.kinetic_integral``)."""
    flat = QuadraticSchedule(gamma=0.0)
    for t0, t1 in INTERVALS:
        exact = mp.mpf(t1) - mp.mpf(t0)
        _assert_within(flat.kinetic_integral(t0, t1), exact, 1)
        _assert_within(flat.potential_integral(t0, t1), exact, 1)
    steep = QuadraticSchedule(gamma=1e300)
    a, _b, scale = _formulas(steep)
    _assert_within(steep.kinetic_integral(1.5e4, 1.5e4 * (1 + 1e-12)),
                   _integral(a, 1.5e4, 1.5e4 * (1 + 1e-12), scale), 32)
    # From t0 = 0 the 50-digit primitive atan(k t1)/k has no cancellation.
    k = mp.sqrt(mp.mpf(1e300))
    _assert_within(steep.kinetic_integral(0.0, 1e160), mp.atan(k * mp.mpf(1e160)) / k, 32)


def test_schedule_kind_and_parameter_define_the_method():
    """Equal parameters of the two cubic forms still select different Methods, and the
    Method's JSON restores each schedule's kind and parameter."""
    cubic, shifted = QHD(schedule=CubicSchedule(s=0.5)), QHD(schedule=ShiftedCubicSchedule(s=0.5))
    assert cubic.content_id != shifted.content_id
    for method in (QHD(), cubic, shifted, shifted.revise(coefficient_rule="integrated")):
        restored = QHD.model_validate_json(method.model_dump_json())
        assert restored == method and type(restored.schedule) is type(method.schedule)
    for record in (CubicSchedule, ShiftedCubicSchedule):
        for invalid in (0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError):
                record(s=invalid)
    # 1 + 1e-300 rounds to 1, so this is not a positive interval.
    with pytest.raises(ValueError, match="0 <= t0 < t1"):
        CubicSchedule(s=1.0).kinetic_integral(1.0, 1.0 + 1e-300)


def _distance(state, expected):
    """Return the 2-norm distance of a binary64 state from a 50-digit vector."""
    return mp.sqrt(sum(abs(mp.mpc(state[i]) - expected[i]) ** 2 for i in range(len(state))))


def _window(result):
    """Return the probability window that the Result's execution recorded."""
    if result.applications:
        arguments = {v.parameter: v.value for v in result.applications[0].arguments}
        return arguments["probability_window"].value
    return max(receipt.probability_window for receipt in result.data.receipts)


@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("schedule", [QuadraticSchedule(gamma=0.3), CubicSchedule(s=0.5),
                                      ShiftedCubicSchedule(s=0.5)], ids=_label)
def test_integrated_rule_is_exact_when_the_hamiltonians_commute(schedule, steps):
    """A constant objective c makes V = c I, which commutes with K at all times, so the exact
    propagator over [0, T] is ``exp(-i (A K + B c))`` with A and B the integrals of a and b.

    The expected state applies it to the uniform state through the 50-digit
    eigendecomposition of the 3 x 3 restricted kinetic matrix (h = 1). The
    integrated rule's classical Schrodinger state equals it at any step count,
    and the midpoint rule's does not.

    Tolerance (machine precision). ``_validation.state_mass_window`` doubles
    the host state budget, the sum of the error norms of the ``expm_multiply``
    calls, for the mass, so half the kernel's recorded window, which the fixed floor
    ``NUMERICAL_RELATION_RTOL`` can only raise, bounds its state error. The
    step integrals carry at most 48u (the largest constant, derived in
    ``schedules.CubicSchedule.kinetic_integral``), ``dt (A/dt)`` two more
    roundings, and forming the generator ``dt (a K + b V)`` at most 4u per
    entry, so the exponents move the state by at most
    ``54u (A ||K|| + B |c|)`` with ``||K|| <= 2``. Rounding the unit-norm
    state to binary64 adds at most u. ``64u (2 A + B |c| + 1)`` covers both.
    """
    x = sp.Symbol("x")
    c, total_time = mp.mpf(3) / 2, 2.0
    problem = Optimization(objective=sp.Rational(3, 2), variables=(x,), bounds=((0.0, 4.0),))
    a, b, scale = _formulas(schedule)
    energies, vectors = mp.eigsy(mp.matrix([[1, -0.5, 0], [-0.5, 1, -0.5], [0, -0.5, 1]]))
    area_a, area_b = _integral(a, 0.0, total_time, scale), _integral(b, 0.0, total_time)
    phases = mp.diag([mp.exp(-1j * area_a * energies[i]) for i in range(3)])
    expected = mp.exp(-1j * area_b * c) * (vectors * phases * vectors.T * (mp.matrix([1, 1, 1]) / mp.sqrt(3)))
    tolerance = None
    for rule in ("integrated", "midpoint"):
        # The expected state starts from the uniform state, which has components on two kinetic
        # eigenvectors of different energy, so the comparison sees a relative kinetic phase. The
        # default kinetic ground state is a single eigenvector of the commuting K and V = c I and
        # would see only a global phase.
        method = QHD(num_grid_points=3, num_steps=steps, total_time=total_time, schedule=schedule,
                     coefficient_rule=rule, keep_state=True, initial_state=UniformState())
        result = solve(problem, method=method, execution="classical", seed=7)
        error = _distance(result.data.artifact(result.artifact).array, expected)
        if rule == "integrated":
            tolerance = _window(result) / 2 + 64 * U * (2 * area_a + area_b * c + 1)
            assert error <= tolerance
            # Each step's generator norm dt a_k H + 2 Q_k, with the kinetic column bound H = sum 1/h**2 of two
            # neighbors per axis and no potential spread, is A_k within 64U, which covers the small assembly
            # charge 2 Q_k (method._schrodinger_step_bounds).
            dt = total_time / steps
            r = result.plan.reconstruction
            norms = [norm for norm, _rows in owner._generator_norms(
                method.theory_flavor, method, owner._grid(result.plan), r.support_values, r.steps, r.step_weights)]
            for k, norm in enumerate(norms):
                part = _integral(a, k * dt, (k + 1) * dt, scale)
                assert abs(norm - part) <= 64 * U * part
        else:
            assert error > 1e6 * tolerance


def _product_state(areas, values, h, order):
    """Apply the one-variable K = 3 product formula with the given per-step integrals (A_k, B_k).

    Link e couples grid points e and e + 1 with ``-1/(2 h**2)``. First order
    applies V(B), then link 0 and link 1 for A. Second order applies V(B/2),
    link 1 for A/2, link 0 for A, link 1 for A/2 and V(B/2). The stencil
    diagonal ``1/h**2`` contributes the phase ``exp(-i A/h**2)``.
    """
    links = []
    for e in (0, 1):
        link = mp.zeros(3, 3)
        link[e, e + 1] = link[e + 1, e] = -1 / (2 * h**2)
        links.append(link)

    def potential(weight, state):
        return mp.matrix([mp.exp(-1j * weight * v) * state[i] for i, v in enumerate(values)])

    def hop(e, weight, state):
        return mp.expm(-1j * weight * links[e]) * state

    state = mp.matrix([1, 1, 1]) / mp.sqrt(3)
    for area_a, area_b in areas:
        if order == 1:
            state = hop(1, area_a, hop(0, area_a, potential(area_b, state)))
        else:
            state = potential(area_b / 2, state)
            state = hop(1, area_a / 2, hop(0, area_a, hop(1, area_a / 2, state)))
            state = potential(area_b / 2, state)
        state = mp.exp(-1j * area_a / h**2) * state
    return state


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("schedule", [CubicSchedule(s=0.5), ShiftedCubicSchedule(s=0.5)], ids=_label)
def test_integrated_product_applies_the_whole_step_integrals(schedule, order):
    """Native and ir_product amplitudes, with their physical phase, equal the product formula
    built from independent step integrals, and the stored block angles follow them.

    The objective ``(x - 1/5)**2 + 1/2`` on the interior grid x = -1/2, 0, 1/2
    (h = 1/2) keeps a nonzero constant, 27/50 after expansion about x = 0,
    which enters only through the physical phase. Its support table holds
    ``f(x) - f(0)`` = 0.45, 0, 0.05.

    Tolerance (machine precision). ``_validation.exact_probability_window``
    for Aer and ``state_mass_window`` for the host kernel double
    per-instruction or per-call error norms for the mass, so half the
    recorded window bounds each execution's state error. Each exponent carries
    at most 48u from its integral (``schedules.CubicSchedule.kinetic_integral``)
    and a few roundings of the weight and angle, below 64u relative, so the
    exponents move the state by at most 64u times the sum of their generator
    norms, ``sum_k (2 A_k/h**2 + B_k max f)``. Rounding the unit-norm state to
    binary64 adds at most u.
    """
    x = sp.Symbol("x")
    objective = (x - sp.Rational(1, 5)) ** 2 + sp.Rational(1, 2)
    problem = Optimization(objective=objective, variables=(x,), bounds=((-1.0, 1.0),))
    total_time, steps, h = 1.2, 2, mp.mpf(1) / 2
    values = [(g - mp.mpf(1) / 5) ** 2 + mp.mpf(1) / 2 for g in (-h, 0, h)]
    a, b, scale = _formulas(schedule)
    dt = total_time / steps
    areas = [(_integral(a, k * dt, (k + 1) * dt, scale), _integral(b, k * dt, (k + 1) * dt))
             for k in range(steps)]
    expected = _product_state(areas, values, h, order)
    exponents = sum(2 * area_a / h**2 + area_b * max(values) for area_a, area_b in areas)
    # The independent product (_product_state) starts from the uniform state, which the Method
    # names explicitly, since its default is the kinetic ground state.
    method = QHD(num_grid_points=3, num_steps=steps, total_time=total_time, schedule=schedule,
                 coefficient_rule="integrated", trotter_order=order, keep_state=True,
                 initial_state=UniformState())
    native = solve(plan(problem, method=method, seed=7))
    host = solve(plan(problem, method=method.revise(theory_flavor="ir_product"), execution="classical", seed=7))
    # Grid point i of the one variable is qubit i, so the native state holds the
    # three grid amplitudes at indices 1, 2 and 4 and zeros elsewhere.
    full = mp.zeros(8, 1)
    for i, index in enumerate((1, 2, 4)):
        full[index] = expected[i]
    for result, reference in ((native, full), (host, expected)):
        state = result.data.artifact(result.artifact).array
        assert _distance(state, reference) <= _window(result) / 2 + 64 * U * (exponents + 1)

    # One stored block per action, in block order, first the projectors of
    # nonzero table values, then the hopping links. A projector's angle is
    # B_part (f(x) - f(0)) and a link's is A_part/(2 h**2), each applied at
    # its stored angle (theory._run_ir_product).
    table = [values[0] - values[1], values[2] - values[1]]
    expected_norms = []
    for area_a, area_b in areas:
        if order == 1:
            expected_norms += [area_b * v for v in table] + [area_a / (2 * h**2)] * 2
        else:
            expected_norms += ([area_b / 2 * v for v in table]
                               + [area_a / (4 * h**2), area_a / (2 * h**2), area_a / (4 * h**2)]
                               + [area_b / 2 * v for v in table])
    norms = [abs(block.angle) for step in host.plan.reconstruction.steps for block in step]
    assert len(norms) == len(expected_norms)
    for norm, expected_norm in zip(norms, expected_norms, strict=True):
        assert abs(norm - expected_norm) <= 64 * U * expected_norm

    # The error model names the rule's time-discretization source, beside the
    # floating-point, product-formula and pruning sources.
    midpoint = plan(problem, method=method.revise(coefficient_rule="midpoint"), seed=7)
    for selected, time, other in ((native.plan, "time_ordering", "midpoint_time"),
                                  (midpoint, "midpoint_time", "time_ordering")):
        sources = selected.error_model.required_sources
        assert time in sources and other not in sources
        assert {"floating_point", "product_formula", "rotation_pruning"} <= set(sources)


def test_potential_integral_requires_a_finite_value():
    """The integrated cubic potential T**4/2 must be finite before a Plan can use it."""
    x = sp.Symbol("x")
    problem = Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 1.0),))
    method = QHD(num_grid_points=4, num_steps=1, total_time=2e103, schedule=CubicSchedule(s=1.0),
                 coefficient_rule="integrated", initial_state=UniformState(), theory_flavor="split_step")
    with pytest.raises(ValueError, match="potential integral over step 0"):
        plan(problem, method=method, execution="classical", seed=7)
    assert plan(problem, method=method.revise(total_time=1e70), execution="classical", seed=7)
