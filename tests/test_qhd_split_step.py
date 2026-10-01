"""Split-step classical QHD: independent Strang products, kinetic eigenbases, time order and the transform canary."""

import itertools
from math import fsum

import mpmath
import numpy as np
import pytest
import scipy.integrate
import sympy as sp

from nwqlib._validation import GLOBAL_PHASE_STATE_ROUNDOFF, UNIT_ROUNDOFF as u
from nwqlib.algorithms.qhd import (
    QHD, CubicSchedule, GaussianState, KineticGroundState, QuadraticSchedule, ShiftedCubicSchedule,
    UniformState,
)
from nwqlib.algorithms.qhd import method as owner
from nwqlib.algorithms.qhd import split_step
from nwqlib.algorithms.qhd.grid import OneHotGrid
from nwqlib.problems import Optimization
from nwqlib.scientist import plan, solve

x, y = sp.symbols("x y", real=True)


def _coordinates(bounds, k, boundary, endpoints):
    """Return ``(points, h)`` per variable from the three grid definitions of the QHD guide.

    Dirichlet interior: ``lower + (i+1) h`` with ``h = (upper-lower)/(K+1)``.
    Dirichlet endpoints: ``lower + i h`` with ``h = (upper-lower)/(K-1)``.
    Periodic: ``lower + i h`` with ``h = (upper-lower)/K``, upper excluded.
    """
    grids = []
    for lower, upper in bounds:
        if boundary == "periodic":
            h, offset = (upper - lower) / k, 0
        elif endpoints:
            h, offset = (upper - lower) / (k - 1), 0
        else:
            h, offset = (upper - lower) / (k + 1), 1
        grids.append(([lower + (i + offset) * h for i in range(k)], h))
    return grids


def _signed_alias(m, k):
    """Return the integer of least magnitude congruent to m modulo K, +K/2 at the Nyquist mode."""
    return m if 2 * m <= k else m - k


def _kinetic_matrix(k, h, boundary, model):
    """Return one variable's kinetic operator as an mpmath matrix, built from its definition.

    The finite-difference operator is ``-1/2`` times the three-point second
    difference: diagonal ``1/h**2`` and ``-1/(2 h**2)`` between neighbors, a
    chain on the Dirichlet grids and a cycle on the periodic grid. The
    spectral operator acts on the Fourier mode ``exp(2 pi i m j/K)`` as
    ``(1/2)(2 pi q/L)**2`` for its least-magnitude alias q and ``L = K h``.
    That energy depends only on ``|q|``, so the matrix is the real cosine sum
    ``(1/K) sum_m E_m cos(2 pi m (i - j)/K)``.
    """
    h = mpmath.mpf(h)
    if model == "spectral":
        length = k * h
        energies = [2 * mpmath.pi**2 * _signed_alias(m, k) ** 2 / length**2 for m in range(k)]
        return mpmath.matrix([[mpmath.fsum(energies[m] * mpmath.cos(2 * mpmath.pi * m * (i - j) / k)
                                           for m in range(k)) / k for j in range(k)] for i in range(k)])
    matrix = mpmath.zeros(k, k)
    for i in range(k):
        matrix[i, i] = 1 / h**2
        for j in (i - 1, i + 1):
            # Indices modulo K on the cycle, where K = 2 counts the one neighbor twice.
            j = j % k if boundary == "periodic" else j
            if 0 <= j < k:
                matrix[i, j] -= 1 / (2 * h**2)
    return matrix


def _exact(value):
    """Return a SymPy rational as an mpmath number at the working precision."""
    value = sp.Rational(value)
    return mpmath.mpf(value.p) / value.q


def _along_axis(state, matrix, axis, k, d):
    """Apply a K-by-K mpmath matrix to axis ``axis`` of a lexicographic d-variable state.

    Position ``sum_j n_j K**(d-1-j)`` holds the grid tuple n.
    """
    result = []
    for indices in itertools.product(range(k), repeat=d):
        total = []
        for m in range(k):
            source = list(indices)
            source[axis] = m
            total.append(matrix[indices[axis], m] * state[int(np.ravel_multi_index(source, (k,) * d))])
        result.append(mpmath.fsum(total))
    return result


def _distance(computed, reference, *, align=False):
    """Return the 2-norm distance of a binary64 state from an mpmath state, optionally up to a global phase."""
    computed = [mpmath.mpc(complex(value)) for value in computed]
    if align:
        overlap = mpmath.fsum(mpmath.conj(r) * c for r, c in zip(reference, computed))
        computed = [c * mpmath.conj(overlap) / abs(overlap) for c in computed]
    return mpmath.sqrt(mpmath.fsum(abs(c - r) ** 2 for c, r in zip(computed, reference)))


def _initial_reference(initial, grids, boundary):
    """Return the exact unit initial state of the Method at the working precision, from the record definitions.

    Each variable's amplitudes follow the QHD guide's table ("Initial states
    and preparation"): 1 for the uniform state, ``sin(pi (i+1)/(K+1))`` for
    the kinetic ground state on the Dirichlet grids and 1 on the periodic
    grid, and ``exp(-(x_i - c)**2/(2 sigma**2))`` at the grid coordinate x_i
    for a Gaussian, with the binary64 center and width read exactly. Each
    vector is normalized, and the state is their product in lexicographic
    order, variable 0 most significant.
    """
    vectors = []
    for j, (points, _h) in enumerate(grids):
        k = len(points)
        if isinstance(initial, KineticGroundState) and boundary != "periodic":
            values = [mpmath.sin(mpmath.pi * (i + 1) / (k + 1)) for i in range(k)]
        elif isinstance(initial, GaussianState):
            center, width = mpmath.mpf(initial.center[j]), mpmath.mpf(initial.widths[j])
            values = [mpmath.exp(-((_exact(p) - center) ** 2) / (2 * width**2)) for p in points]
        else:
            values = [mpmath.mpf(1)] * k
        norm = mpmath.sqrt(mpmath.fsum(v**2 for v in values))
        vectors.append([v / norm for v in values])
    return [mpmath.fprod(factors) for factors in itertools.product(*vectors)]


def _strang_distance(problem, method, digits=30):
    """Solve a split-step Plan and return the distance of its kept state from a Strang oracle and the tolerance.

    The oracle builds each variable's kinetic matrix from its definition
    (``_kinetic_matrix``), not from SciPy's DST or FFT, forms
    ``exp(-i alpha K)`` from an eigendecomposition at ``digits`` significant
    digits, 30 unless the kinetic angles need more, applies it along
    that variable's axis, and applies the potential phases of the whole
    objective, constant included, directly at the grid points, where the
    dyadic coordinates substituted as exact rationals give exact objective
    values. It uses the Plan's stored step weights ``(t, a, b)``, so
    ``alpha = dt a`` and ``beta = dt b/2`` with the binary64 dt, and starts
    from the exact initial state of the Method (``_initial_reference``). The
    distance is the plain 2-norm, so the kept state's physical phase counts.

    Tolerance (derived). The state budget that the kernel observed on its
    trajectory and published as ``readout_state_error``
    (``split_step.evolve``) bounds the kernel's state without the objective
    constant c against the exact product without it, applied to the exact
    initial state, and its start term bounds the construction of the start
    vector. It is at most the Plan's budget ``split_step.state_error``
    (``method._host_state_error``), which the helper also checks. The kept
    state is that state times the constant's phase ``exp(i phi)``
    (``method._constant_phase``), whose angle is within
    ``(3u + 3u**2 + u**3) |c| sum_k |dt b_k|`` of ``-c sum_k dt b_k`` and whose
    multiplication adds at most ``GLOBAL_PHASE_STATE_ROUNDOFF`` u.
    """
    selected = plan(problem, method=method, execution="classical", seed=7)
    result = solve(selected)
    (chunk,) = result.data.observations.chunks
    observed = next(v.value for v in chunk.values if v.label == "readout_state_error")
    assert observed <= selected.reconstruction.host_state_error
    state = result.data.artifact(result.artifact).array
    k, d = method.num_grid_points, len(problem.variables)
    grids = _coordinates(problem.bounds, k, method.boundary, method.include_boundary_points)
    with mpmath.workdps(digits):
        potential = [
            _exact(problem.objective.subs({v: sp.Rational(p) for v, p in zip(problem.variables, point)}))
            for point in itertools.product(*(points for points, _ in grids))
        ]
        eigen = [mpmath.eigsy(_kinetic_matrix(k, h, method.boundary, method.kinetic_model)) for _, h in grids]
        reference = _initial_reference(method.initial_state, grids, method.boundary)
        dt = mpmath.mpf(method.total_time / method.num_steps)
        for _time, a, b in selected.reconstruction.step_weights:
            alpha, beta = dt * mpmath.mpf(a), dt * mpmath.mpf(b) / 2
            half = [mpmath.expj(-beta * value) for value in potential]
            reference = [p * s for p, s in zip(half, reference)]
            for axis, (energies, vectors) in enumerate(eigen):
                propagator = vectors * mpmath.diag([mpmath.expj(-alpha * energies[i]) for i in range(k)]) * vectors.T
                reference = _along_axis(reference, propagator, axis, k, d)
            reference = [p * s for p, s in zip(half, reference)]
        distance = _distance(state, reference)
    weights = fsum(abs(method.total_time / method.num_steps * b) for _t, _a, b in selected.reconstruction.step_weights)
    phase = (GLOBAL_PHASE_STATE_ROUNDOFF + (3 + 3 * u + u * u) * abs(selected.reconstruction.constant) * weights) * u
    return distance, observed + phase


# Dyadic objective terms, exact at the dyadic grid points of every case below.
# The constant 3/8 enters the kept state only through its global phase.
_OBJECTIVE = x**3 / 8 - x**2 / 2 + x / 4 + sp.Rational(3, 8)
_COUPLING = x * y / 4 + y**2 / 8 - y / 2


@pytest.mark.parametrize("rule", ["midpoint", "integrated"])
@pytest.mark.parametrize(
    "d,k,boundary,endpoints,model",
    [
        (1, 5, "dirichlet", False, "finite_difference"),
        (1, 5, "dirichlet", True, "finite_difference"),
        (2, 4, "dirichlet", False, "finite_difference"),
        (1, 8, "periodic", False, "finite_difference"),
        (1, 8, "periodic", False, "spectral"),
        (2, 4, "periodic", False, "finite_difference"),
        (2, 4, "periodic", False, "spectral"),
    ],
)
def test_split_step_matches_an_independent_strang_product(d, k, boundary, endpoints, model, rule):
    """Each step applies ``exp(-i dt b V/2) exp(-i dt a K) exp(-i dt b V/2)``, K exact in its eigenbasis.

    The oracle is ``_strang_distance``. Under the integrated rule its
    ``alpha`` and ``beta`` are ``A_k`` and equal halves ``B_k/2``, and the
    schedule records' weights have their own tests. The two variables have
    spacings 1/2 and 1, which catches a swapped axis, the endpoint grid has
    its own spacing ``2/(K - 1)``, and every grid value, spacing and step
    duration is dyadic, so the objective values and the tables are exact in
    binary64.

    Tolerance (derived in ``_strang_distance``). The state budget that the
    kernel observed (``split_step.evolve``), at most ``split_step.state_error``,
    bounds the distance of the kernel's state from this exact product, and
    the objective constant 3/8 adds its global-phase allowance. The
    tolerance is 1.6e-14 to 3.8e-14 here, and the 30-digit oracle adds about
    1e-29. An unnormalized DST, an unsigned
    spectral index, reversed mode phases, swapped axes (d = 2), full instead
    of half potential phases, the potential applied once per step or a
    missing constant phase give distances of 0.13 or more.
    """
    variables = (x, y)[:d]
    objective = _OBJECTIVE + (_COUPLING if d == 2 else 0)
    if boundary == "periodic":
        bounds = ((0.0, k / 2), (-k / 2, k / 2))[:d]
    elif endpoints:
        bounds = ((-1.0, 1.0),)
    else:
        bounds = ((-(k + 1) / 4, (k + 1) / 4), (-(k + 1) / 2, (k + 1) / 2))[:d]
    problem = Optimization(objective=objective, variables=variables, bounds=bounds)
    method = QHD(num_grid_points=k, num_steps=3, total_time=0.75, schedule=QuadraticSchedule(gamma=0.3),
                 coefficient_rule=rule, boundary=boundary, include_boundary_points=endpoints,
                 kinetic_model=model, theory_flavor="split_step", keep_state=True)
    distance, tolerance = _strang_distance(problem, method)
    assert distance <= tolerance


@pytest.mark.parametrize(
    "d,k,boundary,endpoints,model,initial",
    [
        (2, 4, "dirichlet", False, "finite_difference", KineticGroundState()),
        (1, 5, "dirichlet", True, "finite_difference", KineticGroundState()),
        (2, 4, "dirichlet", False, "finite_difference", GaussianState(center=(0.3, -0.7), widths=(0.6, 1.1))),
        (2, 4, "periodic", False, "spectral", GaussianState(center=(0.3, -0.7), widths=(0.6, 1.1))),
    ],
    ids=["kinetic_ground-interior", "kinetic_ground-endpoints", "gaussian-dirichlet", "gaussian-periodic"],
)
def test_split_step_starts_from_the_selected_initial_state(d, k, boundary, endpoints, model, initial):
    """The split-step flavor evolves the Method's initial state, and its budget charges that state's construction.

    Same oracle and cases as ``test_split_step_matches_an_independent_strang_product``,
    with ``_initial_reference`` writing the kinetic ground state and an
    off-center Gaussian from their definitions. Their start terms
    (``initial_state.restricted_state_error``) replace the uniform state's 2u
    in the tolerance, which is 2.1e-14 to 3.9e-14 here. Starting the kernel
    from the uniform state instead gives distances of 0.27 to 0.61, and
    swapping the Gaussian's two factors, whose centers and widths differ,
    gives 0.52 or more.
    """
    variables = (x, y)[:d]
    objective = _OBJECTIVE + (_COUPLING if d == 2 else 0)
    if boundary == "periodic":
        bounds = ((0.0, k / 2), (-k / 2, k / 2))[:d]
    elif endpoints:
        bounds = ((-1.0, 1.0),)
    else:
        bounds = ((-(k + 1) / 4, (k + 1) / 4), (-(k + 1) / 2, (k + 1) / 2))[:d]
    problem = Optimization(objective=objective, variables=variables, bounds=bounds)
    method = QHD(num_grid_points=k, num_steps=3, total_time=0.75, schedule=QuadraticSchedule(gamma=0.3),
                 boundary=boundary, include_boundary_points=endpoints, kinetic_model=model,
                 theory_flavor="split_step", keep_state=True, initial_state=initial)
    distance, tolerance = _strang_distance(problem, method)
    assert distance <= tolerance


@pytest.mark.parametrize("d,k,boundary,model", [
    (1, 16, "periodic", "spectral"),
    (1, 16, "dirichlet", "finite_difference"),
    (2, 8, "periodic", "finite_difference"),
    (2, 64, "dirichlet", "finite_difference"),
    (3, 6, "dirichlet", "finite_difference"),
])
def test_split_step_byte_law_bounds_the_traced_evolution(d, k, boundary, model):
    """The byte law of ``split_step.sizes`` bounds the traced allocation of one evolution, and its work ignores the schedule.

    ``split_step.sizes`` charges ``64 D + 32 (d + 1) K + 8 T_max + 8 N_s`` for
    the arrays of ``evolve``, ``SCRATCH_BYTES_PER_POINT * K`` for the native
    transform scratch of ducc0.fft, which tracemalloc does not see, and
    ``SCRATCH_BYTES_PER_AXIS * (d + 1)`` for Python and NumPy objects. After a
    warm-up evolution, which loads SciPy's transform modules once, the traced
    peak of the host evolution, start vector and observed state budget
    included (``method._evolve_restricted``), must fit within the law less its
    native term. The cases cover the FFT and DST-I, one to three axes and a
    grid whose 64 D term dominates. The work law is a nominal count that does
    not depend on the schedule weights, so a cubic schedule must give the same
    units as the quadratic one.
    """
    import tracemalloc

    variables = (x, y, sp.Symbol("z", real=True))[:d]
    objective = sum(v**2 for v in variables) + variables[0] * variables[-1] / 4
    problem = Optimization(objective=objective, variables=variables, bounds=((-1.0, 1.0),) * d)
    method = QHD(num_grid_points=k, num_steps=3, total_time=0.5, boundary=boundary, kinetic_model=model,
                 theory_flavor="split_step")
    selected = plan(problem, method=method, execution="classical", seed=7)
    owner._evolve_restricted(selected, "split_step")
    tracemalloc.start()
    try:
        owner._evolve_restricted(selected, "split_step")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= selected.reconstruction.workspace_bytes - split_step.SCRATCH_BYTES_PER_POINT * k
    other = plan(problem, method=method.revise(schedule=CubicSchedule(s=0.1)), execution="classical", seed=7)
    assert other.reconstruction.size_units == selected.reconstruction.size_units


def test_split_step_tie_window_follows_the_state():
    """The observed state budget resolves two wells whose probabilities the Plan's ceiling would tie.

    One variable on the periodic spectral grid of [-1, 1), K = 16, objective
    ``(x**2 - 1/2)**2 + x/20000000``, ``ShiftedCubicSchedule(s=2e-4)``, the
    integrated rule, 100 steps and T = 10. The first kinetic integral is
    ``4 (s**-2 - (s + dt)**-2)``, about 1.0e8, so the Plan's ceiling, which charges the angle
    error of the Nyquist mode to the whole state (``split_step.state_error``),
    exceeds the gap between the two wells' most probable points, 2 and 14.
    The state holds almost no weight in the high modes, and the budget
    observed on the trajectory (``split_step.evolve``) is far smaller.

    Reference: the same Strang product at 40 digits with the stored step
    weights, the stored support table and the binary64 dt and spacing read
    exactly, an exact DFT and the energies ``2 (pi |q|/(K h))**2`` of the
    signed indices q. The kernel's tie window must bound the error of every
    computed pair difference, and the reference gap exceeds twice the window,
    so the window cannot tie the two wells and the selection must be the
    reference maximum.
    """
    k = 16
    problem = Optimization(objective=(x**2 - sp.Rational(1, 2)) ** 2 + x / 20000000, variables=(x,),
                           bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=k, boundary="periodic", kinetic_model="spectral", theory_flavor="split_step",
                 schedule=ShiftedCubicSchedule(s=2e-4), coefficient_rule="integrated", num_steps=100,
                 total_time=10.0, keep_state=True)
    result = solve(problem, method=method, execution="classical", seed=7)
    selected = result.plan
    computed = np.abs(result.data.artifact(result.artifact).array) ** 2
    (table,) = selected.reconstruction.support_values
    with mpmath.workdps(40):
        h = mpmath.mpf(owner._grid(selected).spacing(0))
        potential = [mpmath.mpf(v) for v in table.values.array.tolist()]
        energies = [2 * (mpmath.pi * abs(_signed_alias(m, k)) / (k * h)) ** 2 for m in range(k)]
        roots = [mpmath.expj(-2 * mpmath.pi * m / k) for m in range(k)]
        state = [mpmath.mpc(1) / mpmath.sqrt(k)] * k
        dt = mpmath.mpf(method.total_time / method.num_steps)
        for _time, a, b in selected.reconstruction.step_weights:
            alpha, beta = dt * mpmath.mpf(a), dt * mpmath.mpf(b) / 2
            half = [mpmath.expj(-beta * v) for v in potential]
            state = [s * p for s, p in zip(state, half)]
            modes = [mpmath.fsum(state[j] * roots[(m * j) % k] for j in range(k)) / mpmath.sqrt(k) for m in range(k)]
            modes = [c * mpmath.expj(-alpha * e) for c, e in zip(modes, energies)]
            state = [mpmath.fsum(modes[m] * mpmath.conj(roots[(m * j) % k]) for m in range(k)) / mpmath.sqrt(k)
                     for j in range(k)]
            state = [s * p for s, p in zip(state, half)]
        reference = [abs(s) ** 2 for s in state]
        errors = [mpmath.mpf(float(p)) - q for p, q in zip(computed, reference)]
        pair_error = float(max(errors) - min(errors))
        top, second = sorted(range(k), key=lambda i: reference[i], reverse=True)[:2]
        gap = float(reference[top] - reference[second])
    window = result.most_probable_tie_window
    assert {top, second} == {2, 14}
    assert pair_error <= window < gap / 2
    assert gap < owner._host_tie_window(selected)
    assert result.most_probable_indices == (top,)


@pytest.mark.parametrize("case", ["kinetic", "potential", "tiny_potential", "huge_kinetic_weight",
                                  "overflowing_window"])
def test_split_step_budget_covers_the_formation_of_large_phase_angles(case):
    """Large phase angles lose accuracy when they are formed, and the state budgets charge it.

    Kinetic case. The shifted cubic schedule with ``s = 1e-3`` has the
    kinetic integral ``A_0 = 4 (s**-2 - (s + dt)**-2)``, about 4e6, over the
    first step ``[0, 0.25]``, so the first kinetic angles ``A_0 E_r`` reach
    about 3e7 rad on the h = 1/2 Dirichlet grid. Rounding an angle of that
    size moves it by about 2e-9 rad, which no later operation can restore.
    Against the 30-digit Strang product with the stored weights
    (``_strang_distance``) the kernel's state from the kinetic ground state
    is 4.3e-10 away. The budget observed on the trajectory
    (``split_step.evolve``), which weights each kinetic angle error by the
    state's mode populations, is 3.6e-9, and the Plan's budget
    ``split_step.state_error``, which charges the largest eigenvalue, is
    5.0e-8. Both are dominated by the phase-formation term
    ``(EIGENVALUE_ROUNDOFF + 2) u |alpha| E``. Without it the transform and
    multiplication charges give about 2e-14, which the measured distance
    exceeds by a factor of about 2e4.

    Potential case. The objective ``2**20 (x**2/2 - x/4)`` on the same grid,
    with the quadratic schedule, has values up to 786,432, so each potential
    half's angles ``beta_k V`` reach about 1e5 rad and their rounding about
    1e-11 rad. The kernel's state is 2.1e-12 from the 30-digit product, and
    the observed budget is 5.6e-11, almost all of it the potential-angle
    charge ``u |beta_k| (2 ||V z|| + m W)`` of ``split_step.evolve``. With
    that charge set to zero the budget is 2.5e-14, below the distance, so
    this case catches a dropped potential-angle charge. The budget exceeds
    the distance about 27 times, so the case cannot detect an under-charge
    within that slack, for example charging only one potential half per
    step or never refreshing the cached potential moment.

    Tiny-potential case. Four periodic points on ``[0, 4e150)`` carry the
    tables ``(1, 2, 3, 3) 1e-200``, and one step of ``T = 1e210`` gives the
    potential angles up to about 1.5e10 rad, whose rounding the observed
    budget charges through the moment ``||V z||``, about 2.4e-200. The
    terms ``q V**2`` of that moment, about 1e-400, lie below the binary64
    range, so an unscaled evaluation would give the moment 0 and a budget of
    about 4e-15, while the state is about 1.3e-6 from the 30-digit product.
    The observed budget, which scales the moment by its largest weighted
    entry (``split_step._potential_moment``), is 5.3e-6.

    Huge-kinetic-weight case. ``ShiftedCubicSchedule(s=1e-80)`` with the
    integrated rule gives the one step's kinetic integral 4e160 and
    angles up to 8e160 on four periodic points of spacing 1, whose
    squares overflow. The uniform start is the zero mode of every periodic
    kinetic operator and the objective is zero, so the exact state stays
    uniform. Planning admits the Plan, and an unscaled moment would be NaN
    and make the budget and the tie window NaN (``split_step._kinetic_axis``
    scales by the largest angle). The oracle works at 200 digits here,
    because its eigenvalues are exact only to its working precision, and an
    error of 1e-30 in the zero eigenvalue would turn that mode's phase at
    the angle scale 1e160 into noise.

    Overflowing-window case. ``ShiftedCubicSchedule(s=1e-150)`` with the
    integrated rule, three steps and T = 1 on eight periodic spectral points
    of [-1, 1), with ``(x - 1/3)**2`` and a Gaussian start, has the Plan
    budget 5.3e287, whose square in the mass and tie windows
    (``_validation.probability_difference_window``) overflows, so no Result
    could record its windows. Planning refuses it before the kernel is bound
    (``method._admit_host_windows``), where the evolution would otherwise
    finish and the window conversion fail.
    """
    digits = 30
    if case == "overflowing_window":
        problem = Optimization(objective=(x - sp.Rational(1, 3)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
        method = QHD(num_grid_points=8, boundary="periodic", kinetic_model="spectral", num_steps=3,
                     total_time=1.0, schedule=ShiftedCubicSchedule(s=1e-150), coefficient_rule="integrated",
                     initial_state=GaussianState(center=(0.2,), widths=(0.3,)), theory_flavor="split_step")
        with pytest.raises(ValueError, match="mass window of the classical QHD split_step kernel is inf"):
            plan(problem, method=method, execution="classical", seed=7)
        return
    if case == "kinetic":
        problem = Optimization(objective=_OBJECTIVE, variables=(x,), bounds=((-1.5, 1.5),))
        method = QHD(num_grid_points=5, num_steps=3, total_time=0.75, schedule=ShiftedCubicSchedule(s=1e-3),
                     coefficient_rule="integrated", theory_flavor="split_step", keep_state=True)
    elif case == "tiny_potential":
        tiny = sp.Float(1e-200)
        objective = sp.Piecewise((tiny, x < sp.Float(1e150)), (2 * tiny, x < sp.Float(2e150)),
                                 (3 * tiny, x < sp.Float(3e150)), (4 * tiny, True))
        problem = Optimization(objective=objective, variables=(x,), bounds=((0.0, 4e150),))
        method = QHD(num_grid_points=4, boundary="periodic", num_steps=1, total_time=1e210,
                     schedule=QuadraticSchedule(gamma=0.0), theory_flavor="split_step", keep_state=True)
    elif case == "huge_kinetic_weight":
        problem = Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 4.0),))
        method = QHD(num_grid_points=4, boundary="periodic", num_steps=1, total_time=1.0,
                     schedule=ShiftedCubicSchedule(s=1e-80), coefficient_rule="integrated",
                     theory_flavor="split_step", keep_state=True)
        digits = 200
    else:
        problem = Optimization(objective=sp.Integer(2) ** 20 * (x**2 / 2 - x / 4), variables=(x,),
                               bounds=((-1.5, 1.5),))
        method = QHD(num_grid_points=5, num_steps=3, total_time=0.75, schedule=QuadraticSchedule(gamma=0.3),
                     theory_flavor="split_step", keep_state=True)
    distance, tolerance = _strang_distance(problem, method, digits)
    assert distance <= tolerance


@pytest.mark.parametrize("k", [2, 3, 5, 8, 16])
@pytest.mark.parametrize("boundary,endpoints,model", [
    ("dirichlet", False, "finite_difference"),
    ("dirichlet", True, "finite_difference"),
    ("periodic", False, "finite_difference"),
    ("periodic", False, "spectral"),
])
def test_kinetic_transforms_are_orthonormal_eigenbases_of_the_stencils(k, boundary, endpoints, model):
    """The kernel's transform is the orthonormal eigenbasis of the kinetic matrix, with the stated eigenvalues.

    The test builds, at 30 digits, the sine matrix
    ``S[r, j] = sqrt(2/(K+1)) sin(pi (j+1)(r+1)/(K+1))`` on the Dirichlet grids
    and the unitary DFT ``F[m, j] = K**(-1/2) exp(-2 pi i m j/K)`` on the
    periodic grid, and the kinetic matrix from its definition
    (``_kinetic_matrix``). ``S S = I`` and ``F F* = I`` (orthonormality),
    ``split_step._transform`` applied to each basis vector reproduces the
    column of S or F, and the diagonal of ``S K S`` or ``F K F*`` is the
    eigenvalue that ``split_step.kinetic_eigenvalues`` reports at the same
    output index, with zero off-diagonal entries.

    Tolerances (derived). A column of the kernel's transform lies within the
    qualification bound ``TRANSFORM_ROUNDOFF * L * u`` of the exact column
    (``split_step.TRANSFORM_ROUNDOFF``, ``transform_levels``), each reported
    eigenvalue within ``EIGENVALUE_ROUNDOFF * u`` relative of the exact one
    (``split_step.EIGENVALUE_ROUNDOFF``), and the 30-digit identities within
    1e-25, which also covers the 30-digit zero eigenvalue of the constant
    mode (entries up to about 300 here, so about 1e-28). A DST without the
    orthonormal factor misses by a factor ``sqrt(2 (K + 1))``, and reversed
    or unsigned modes by far more than 13u.
    """
    periodic = boundary == "periodic"
    (points, h), = _coordinates(((-1.0, 1.0),), k, boundary, endpoints)
    grid = OneHotGrid(("x",), ((-1.0, 1.0),), k, include_boundary_points=endpoints, boundary=boundary)
    assert grid.spacing(0) == h
    reported = split_step.kinetic_eigenvalues(grid, 0, model)
    bound = split_step.TRANSFORM_ROUNDOFF * split_step.transform_levels(grid) * u
    with mpmath.workdps(30):
        if periodic:
            basis = mpmath.matrix([[mpmath.expj(-2 * mpmath.pi * m * j / k) / mpmath.sqrt(k)
                                    for j in range(k)] for m in range(k)])
            adjoint = basis.transpose_conj()
        else:
            basis = mpmath.matrix([[mpmath.sqrt(mpmath.mpf(2) / (k + 1)) * mpmath.sin(mpmath.pi * (j + 1) * (r + 1) / (k + 1))
                                    for j in range(k)] for r in range(k)])
            adjoint = basis.T
        assert mpmath.mnorm(basis * adjoint - mpmath.eye(k), 1) < 1e-25
        diagonal = basis * _kinetic_matrix(k, h, boundary, model) * adjoint
        for m in range(k):
            column = split_step._transform(np.eye(k, dtype=complex)[:, m].copy(), 0, periodic)
            assert _distance(column, [basis[r, m] for r in range(k)]) <= bound
            for r in range(k):
                if r != m:
                    assert abs(diagonal[r, m]) < 1e-25
            exact = mpmath.re(diagonal[m, m])
            assert abs(reported[m] - exact) <= split_step.EIGENVALUE_ROUNDOFF * u * abs(exact) + 1e-25


@pytest.mark.parametrize("k", [4, 8, 16, 64])
def test_spectral_and_finite_difference_energies_at_nyquist_and_low_momentum(k):
    """On the periodic grid, E_sp/E_FD = pi**2/4 at the Nyquist mode and 0 <= E_sp - E_FD <= 2 pi**4 q**4 h**2/(3 L**4).

    With ``x = pi q/K``, ``E_FD = (2/h**2) sin(x)**2`` and ``E_sp = (2/h**2) x**2``
    (``split_step.kinetic_eigenvalues``), and ``x**2 - x**4/3 <= sin(x)**2 <= x**2``
    gives the bound for ``|q| <= K/2``. At ``q = K/2``, ``x = pi/2``. The
    signed index of each mode is the alias of least magnitude, built here
    from its definition.

    Tolerance (derived). Each reported energy carries at most
    ``EIGENVALUE_ROUNDOFF * u`` relative error (``split_step.EIGENVALUE_ROUNDOFF``),
    so the computed difference lies within ``2 * 13 u E_sp`` of the exact
    one. The ratio carries ``(2 * 13 + 1) u`` relative, and the reference
    ``np.pi**2/4`` another 2u (np.pi within u/2 of pi, one rounding of the
    square and an exact division). An unsigned index
    gives mode K - 1 the energy of momentum K - 1, far above the bound.
    """
    grid = OneHotGrid(("x",), ((-1.5, 2.5),), k, boundary="periodic")
    h, length = grid.spacing(0), 4.0
    finite = split_step.kinetic_eigenvalues(grid, 0, "finite_difference")
    spectral = split_step.kinetic_eigenvalues(grid, 0, "spectral")
    roundoff = split_step.EIGENVALUE_ROUNDOFF * u
    ratio = spectral[k // 2] / finite[k // 2]
    assert abs(ratio - np.pi**2 / 4) <= (2 * split_step.EIGENVALUE_ROUNDOFF + 3) * u * np.pi**2 / 4
    for m in range(k):
        q = _signed_alias(m, k)
        difference = spectral[m] - finite[m]
        assert -2 * roundoff * spectral[m] <= difference
        assert difference <= 2 * np.pi**4 * q**4 * h**2 / (3 * length**4) + 2 * roundoff * spectral[m]


def _restricted_hamiltonian(problem, k, boundary, model):
    """Return the kinetic matrix and the potential diagonal of a one-variable problem in binary64."""
    (points, h), = _coordinates(problem.bounds, k, boundary, False)
    with mpmath.workdps(30):
        kinetic = np.array(_kinetic_matrix(k, h, boundary, model).tolist(), dtype=float)
    evaluate = sp.lambdify(problem.variables, problem.objective)
    return kinetic, np.diag([float(evaluate(point)) for point in points])


def _state(problem, **options):
    """Return the kept final state of one classical solve."""
    result = solve(plan(problem, method=QHD(keep_state=True, **options), execution="classical", seed=7))
    return result.data.artifact(result.artifact).array


def _assert_second_order(errors):
    """Require ratios of successive errors that approach 4 as the step halves.

    A time-symmetric step (``S(dt) S(-dt) = I``) has only even powers in its
    global error at fixed T, ``e(dt) = c2 dt**2 + c4 dt**4 + O(dt**6)``, so
    ``e(dt)/e(dt/2) = 4 + 3 (c4/c2) dt**2 + O(dt**4)``, whose distance from 4
    shrinks by about 4 each time dt halves. The check requires a factor of
    at least 2. A first-order method has ratios near 2 and fails.
    """
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    deviations = [abs(r - 4) for r in ratios]
    assert all(deviations[i + 1] <= deviations[i] / 2 for i in range(len(deviations) - 1)), ratios


@pytest.mark.parametrize("rule", ["midpoint", "integrated"])
@pytest.mark.parametrize("boundary,model", [("dirichlet", "finite_difference"), ("periodic", "spectral")])
def test_split_step_converges_at_second_order_in_time(boundary, model, rule):
    """At fixed grid, objective and gamma, the error against the time-ordered evolution falls as dt**2.

    The reference integrates ``psi' = -i (a(t) K + b(t) V) psi`` for the
    quadratic schedule ``a = 1/(1 + gamma t**2)``, ``b = 1 + gamma t**2``
    with SciPy's DOP853 at relative tolerance 1e-13, with K and V built
    independently (``_restricted_hamiltonian``). Kinetic and potential do not
    commute, so the step has splitting and time-ordering error of local order
    three under both rules, and both steps are time symmetric. From 16 to 128
    steps the ratios are about 4.02, 4.004 and 4.001 on the Dirichlet grid and
    4.15, 4.03 and 4.008 with the spectral kinetic, whose largest eigenvalue
    ``2 pi**2 (K/2)**2/L**2`` is 2.5 times the finite-difference one, so it
    enters the asymptotic range at a smaller step (ratios 5.8 and 5.6 from 4
    to 16 steps). The reference error of about 1e-12 moves a ratio by less
    than 1e-6.
    """
    k = 3 if boundary == "dirichlet" else 4
    problem = Optimization(objective=(x - 0.3) ** 2 + x**3 / 2, variables=(x,), bounds=((-1.0, 1.0),))
    kinetic, potential = _restricted_hamiltonian(problem, k, boundary, model)

    def rhs(t, value):
        psi = value[:k] + 1j * value[k:]
        weight = 1 + 0.3 * t * t
        change = -1j * ((kinetic / weight + weight * potential) @ psi)
        return np.concatenate([change.real, change.imag])

    start = np.concatenate([np.full(k, 1 / np.sqrt(k)), np.zeros(k)])
    solution = scipy.integrate.solve_ivp(rhs, (0.0, 2.0), start, method="DOP853", rtol=1e-13, atol=1e-14)
    reference = solution.y[:k, -1] + 1j * solution.y[k:, -1]
    # The reference starts from the uniform state, so the Method names it instead of its default
    # kinetic ground state. The ratios above were measured from this start.
    errors = [np.linalg.norm(_state(problem, num_grid_points=k, num_steps=n, total_time=2.0,
                                     schedule=QuadraticSchedule(gamma=0.3), coefficient_rule=rule,
                                     boundary=boundary, kinetic_model=model, theory_flavor="split_step",
                                     initial_state=UniformState())
                             - reference) for n in (16, 32, 64, 128)]
    _assert_second_order(errors)


def test_integrated_split_step_is_exact_when_the_hamiltonians_commute():
    """A constant objective c makes V = c I, so the exact evolution is ``exp(-i c B) exp(-i A K)`` at any step count.

    The kernel leaves c out of its potential (``method._constant_phase``), so
    its steps apply the kinetic factors alone, which commute and telescope to
    the exponential of the summed exponents. Under the integrated rule these
    are the exact integral ``A = ∫ a`` over [0, T] up to the step integrals'
    roundoff. The cubic schedule ``a = 2/(s + t**3)`` with ``s = 0.05``
    varies fast near t = 0. The reference takes A from a 30-digit mpmath
    quadrature and applies ``exp(-i A K)`` to the uniform state through a
    30-digit eigendecomposition of the Dirichlet stencil (h = 1). The phase
    ``exp(-i c B)`` is global, so the comparison removes a global phase and
    checks the telescoping of the kinetic factors (the Strang test covers
    the restored phase). The midpoint rule's frozen weights miss the peak of
    a and lie 0.03 to 0.34 away.

    Tolerance (derived). ``split_step.state_error`` bounds the kernel's
    distance from the exact product with its stored weights. Each stored
    kinetic exponent ``dt (A_k/dt)`` is within 48u of the exact step integral
    (``schedules.CubicSchedule.kinetic_integral``) plus two roundings, so the
    summed exponent is within ``50 u A`` of A and moves the state by at most
    ``50 u A max E``. The tolerance is about 2.2e-13 and the measured
    distances below 1e-15.
    """
    k, s, total = 3, 0.05, 1.5
    problem = Optimization(objective=sp.Rational(1, 2) + 0 * x, variables=(x,), bounds=((-2.0, 2.0),))
    with mpmath.workdps(30):
        integral = mpmath.quad(lambda t: 2 / (s + t**3), [0, mpmath.cbrt(s), total])
        energies, vectors = mpmath.eigsy(_kinetic_matrix(k, 1.0, "dirichlet", "finite_difference"))
        start = mpmath.matrix([1 / mpmath.sqrt(k)] * k)
        propagated = vectors * mpmath.diag([mpmath.expj(-integral * e) for e in energies]) * vectors.T * start
        reference = [propagated[i] for i in range(k)]
        largest = max(energies)
        for steps in (1, 2, 5):
            # The uniform start has components on two kinetic eigenvectors of different energy.
            # The default kinetic ground state is one eigenvector, on which the midpoint rule errs only by the global
            # phase that the comparison removes.
            options = dict(num_grid_points=k, num_steps=steps, total_time=total, schedule=CubicSchedule(s=s),
                           theory_flavor="split_step", initial_state=UniformState())
            selected = plan(problem, method=QHD(coefficient_rule="integrated", keep_state=True, **options),
                            execution="classical", seed=7)
            result = solve(selected)
            tolerance = selected.reconstruction.host_state_error + 50 * u * integral * largest
            assert _distance(result.data.artifact(result.artifact).array, reference, align=True) <= tolerance
            midpoint = _state(problem, coefficient_rule="midpoint", **options)
            assert _distance(midpoint, reference, align=True) > 1e6 * tolerance


@pytest.mark.parametrize("boundary", ["dirichlet", "periodic"])
def test_finite_difference_split_step_approaches_the_schrodinger_flavor(boundary):
    """With gamma = 0 the schrodinger flavor is the exact evolution, and split-step differs from it by O(dt**2).

    A time-independent Hamiltonian makes every ``schrodinger`` step the exact
    exponential of ``dt (K + V)``, so its state is the exact restricted
    evolution at any step count up to its roundoff budget
    (``_validation.expm_multiply_state_error``, below 1e-12 here). The
    finite-difference split-step state has only the Strang splitting error,
    which is time symmetric, so the distance between the two falls as dt**2
    (``_assert_second_order``). For these problems the ratios are 4.5 and 4.8,
    4.1 and 4.13, and 4.02 and 4.03 on the Dirichlet and periodic grids.
    """
    k = 3 if boundary == "dirichlet" else 4
    problem = Optimization(objective=(x - 0.3) ** 2 + x**3 / 2, variables=(x,), bounds=((-1.0, 1.0),))
    options = dict(num_grid_points=k, total_time=2.0, schedule=QuadraticSchedule(gamma=0.0), boundary=boundary)
    distances = [np.linalg.norm(_state(problem, num_steps=n, theory_flavor="split_step", **options)
                                - _state(problem, num_steps=n, theory_flavor="schrodinger", **options))
                 for n in (4, 8, 16, 32)]
    _assert_second_order(distances)


def _canary_inputs(k):
    """Three unit inputs: a basis vector, ``(-1)**j + i cos(0.7 j)`` and ``cos(0.33 j) + i sin(0.91 j)``."""
    j = np.arange(k)
    basis = np.zeros(k, dtype=complex)
    basis[min(1, k - 1)] = 1
    for vector in (basis, (-1.0) ** j + 1j * np.cos(0.7 * j), np.cos(0.33 * j) + 1j * np.sin(0.91 * j)):
        yield vector / np.linalg.norm(vector)


@pytest.mark.parametrize("periodic,lengths", [
    (True, (2, 3, 5, 8, 16, 17, 31, 64, 101, 127)),
    (False, (2, 3, 5, 8, 16, 31, 60, 100)),
])
def test_scipy_transforms_meet_the_qualification_constant(periodic, lengths):
    """Dependency canary: SciPy's orthonormal FFT, inverse FFT and DST-I against 80-digit sums.

    ``split_step.TRANSFORM_ROUNDOFF`` states the assumption
    ``||fl(T x) - T x|| <= 5 u L ||x||`` with ``L = ceil(log2 K)`` for the FFT
    and ``ceil(log2(2 (K + 1)))`` for DST-I (``split_step.transform_levels``),
    which the split-step state budget uses. The lengths include powers of
    two, small primes and the primes 101 and 127 and the DST logical length
    ``2 (100 + 1) = 202 = 2 * 101``, which has a large prime factor. Each
    exact value is an 80-digit sum over the binary64 input read exactly.
    SciPy 1.18.1, whose backend is ducc0.fft, measured at most 0.8 of ``u L``. A failure means the installed
    SciPy no longer meets the stated constant, and the budget must be
    requalified before it is trusted.
    """
    for k in lengths:
        grid = OneHotGrid(("x",), ((0.0, 1.0),), k, boundary="periodic" if periodic else "dirichlet")
        bound = split_step.TRANSFORM_ROUNDOFF * split_step.transform_levels(grid) * u
        with mpmath.workdps(80):
            if periodic:
                twiddles = [mpmath.expj(-2 * mpmath.pi * m / k) for m in range(k)]
                entries = {False: lambda r, c: twiddles[(r * c) % k],
                           True: lambda r, c: mpmath.conj(twiddles[(r * c) % k])}
                scale = 1 / mpmath.sqrt(k)
            else:
                sines = [mpmath.sin(mpmath.pi * m / (k + 1)) for m in range(2 * (k + 1))]
                entries = {False: lambda r, c: sines[((r + 1) * (c + 1)) % (2 * (k + 1))]}
                scale = mpmath.sqrt(mpmath.mpf(2) / (k + 1))
            for inverse, entry in entries.items():
                for vector in _canary_inputs(k):
                    values = [mpmath.mpc(complex(v)) for v in vector]
                    exact = [scale * mpmath.fsum(entry(r, c) * values[c] for c in range(k)) for r in range(k)]
                    computed = split_step._transform(vector.copy(), 0, periodic, inverse)
                    assert _distance(computed, exact) <= bound, (k, inverse)


def test_spectral_kinetic_model_requires_the_periodic_split_step_route():
    """The spectral model is the periodic Fourier operator, which only the split-step flavor applies.

    Its Fourier modes do not vanish at the box ends, so the Dirichlet
    boundary rejects it when the Method is built. Native execution and the
    ``schrodinger`` and ``ir_product`` flavors apply the finite-difference
    stencil of the one-hot product, so planning them with the spectral model
    raises before any table is evaluated. The periodic split-step plan is the
    legal case.
    """
    problem = Optimization(objective=x**2, variables=(x,), bounds=((-1.0, 1.0),))
    with pytest.raises(ValueError, match="requires boundary='periodic'"):
        QHD(kinetic_model="spectral")
    method = QHD(num_grid_points=4, boundary="periodic", kinetic_model="spectral")
    for execution, flavor in (("quantum", "split_step"), ("classical", "schrodinger"), ("classical", "ir_product")):
        with pytest.raises(ValueError, match="theory_flavor='split_step'"):
            plan(problem, method=method.revise(theory_flavor=flavor), execution=execution, seed=7)
    selected = plan(problem, method=method.revise(theory_flavor="split_step"), execution="classical", seed=7)
    assert "spectral kinetic model" in selected.construction.kernels[0].implementation.domain
    assert "kinetic_model" in selected.error_model.required_sources


@pytest.mark.parametrize("boundary", ["dirichlet", "periodic"])
def test_split_step_transforms_run_with_one_worker(boundary):
    """Every kernel transform passes ``workers=1``, the setting of the scratch measurement.

    ``split_step.SCRATCH_BYTES_PER_POINT`` was measured with one transform
    worker, and SciPy otherwise takes the worker count from
    ``scipy.fft.set_workers``, so the kernel fixes it. Under
    ``set_workers(2)`` each FFT, inverse FFT or DST call of a two-variable,
    two-step evolution must still receive ``workers=1``, 2d = 4 transforms
    per step.
    """
    import scipy.fft
    from unittest.mock import Mock, patch

    problem = Optimization(objective=x**2 + x * y / 4, variables=(x, y), bounds=((0.0, 2.0), (-2.0, 2.0)))
    method = QHD(num_grid_points=4, num_steps=2, total_time=0.5, boundary=boundary, theory_flavor="split_step")
    selected = plan(problem, method=method, execution="classical", seed=7)
    names = ("fft", "ifft") if boundary == "periodic" else ("dst",)
    wrapped = {name: Mock(wraps=getattr(scipy.fft, name)) for name in names}
    with scipy.fft.set_workers(2), patch.multiple(scipy.fft, **wrapped):
        owner._evolve_restricted(selected, "split_step")
    calls = [c for call in wrapped.values() for c in call.call_args_list]
    assert len(calls) == 2 * 4
    assert all(c.kwargs.get("workers") == 1 for c in calls)
