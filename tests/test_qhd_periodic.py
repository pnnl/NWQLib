"""Periodic one-hot QHD: the circulant restricted model, the wrap link and native convergence."""

import itertools

import mpmath
import numpy as np
import pytest
import sympy as sp

from nwqlib._validation import expm_multiply_state_error
from nwqlib.algorithms.qhd import QHD, QuadraticSchedule
from nwqlib.algorithms.qhd import method as owner
from nwqlib.problems import Optimization
from nwqlib.scientist import plan, solve

# gamma = 0 makes the Hamiltonian time independent, with a = b = 1 at every step.
STATIC = QuadraticSchedule(gamma=0.0)


def _problem(d, k, *, objective=True):
    """Return a d-variable problem on periodic boxes whose grid data are exact in binary64.

    Variable 0 has the period k/2 on [0, k/2) and variable 1 the period k on
    [-k/2, k/2), so their spacings are 1/2 and 1 on the periodic grid of K = k
    points and the two variables have different h. Both boxes contain the
    origin, so planning expands the objective about zero
    (``objective.expansion_centers``). Its dyadic monomials evaluated at the
    dyadic grid points need only a few bits, so every table value is exact.
    """
    x, y = sp.symbols("x y", real=True)
    expression = x**3 / 8 - x**2 / 2 + x / 4
    if d == 2:
        expression += x * y / 4 + y**2 / 8 - y / 2
    return Optimization(
        objective=expression if objective else sp.Integer(0),
        variables=(x, y)[:d],
        bounds=((0.0, k / 2), (-k / 2, k / 2))[:d],
    )


def _periodic_coordinates(problem, k):
    """Return ``(x_i, h)`` per variable, with ``x_i = lower + i h`` and ``h = (upper - lower)/K``, i < K."""
    grids = []
    for lower, upper in problem.bounds:
        h = (upper - lower) / k
        grids.append(([lower + i * h for i in range(k)], h))
    return grids


def _circulant(k, h):
    """Return -1/2 times the periodic second difference on K points.

    ``(psi_(i+1) - 2 psi_i + psi_(i-1))/h**2`` with indices modulo K gives the
    diagonal ``1/h**2`` and ``-1/(2 h**2)`` at ``(i, i +- 1 mod K)``.
    """
    matrix = np.zeros((k, k))
    for i in range(k):
        matrix[i, i] = 1 / h**2
        matrix[i, (i + 1) % k] = matrix[(i + 1) % k, i] = -1 / (2 * h**2)
    return matrix


def _kronecker_sum(matrices):
    """Return ``sum_j I x ... x M_j x ... x I`` with variable 0 the most significant factor.

    This is the lexicographic grid-index order of the restricted basis, whose
    position is ``sum_j n_j K**(d-1-j)``.
    """
    total = np.zeros((1, 1))
    for matrix in matrices:
        total = np.kron(total, np.eye(len(matrix))) + np.kron(np.eye(len(total)), matrix)
    return total


@pytest.mark.parametrize("k", [4, 6])
@pytest.mark.parametrize("d", [1, 2])
def test_periodic_classical_model_matches_circulant_eigendecomposition(d, k):
    """The periodic Schrodinger kernel evolves ``-Delta_h/2 + diag(f)`` with the circulant stencil.

    With gamma = 0 each step applies the exact exponential of one time
    independent restricted Hamiltonian H, so the final state is
    ``exp(-i T H) psi0`` for the uniform psi0. The test builds H from the
    endpoint-exclusive coordinates and a circulant matrix per variable,
    written here independently of the library, and forms the exponential by
    a dense 30-digit eigendecomposition. Every grid value, spacing and step
    duration is dyadic, so the kernel's floating-point generators
    ``dt (K + V)`` equal the exact ones. The kernel's state therefore lies
    within the first-order 2-norm budget delta of its ``expm_multiply`` calls
    of the oracle, up to a global phase (``_validation.expm_multiply_state_error``
    with the per-call norms of ``method._generator_norms``), which is
    at most about 1e-12 here. The 30-digit oracle adds about 1e-29. Without
    the wrap link the phase-aligned distance is about 0.5, and with the
    endpoint-inclusive spacing 0.06 to 0.28.
    """
    problem = _problem(d, k)
    method = QHD(num_grid_points=k, num_steps=3, total_time=0.75, schedule=STATIC, boundary="periodic",
                 keep_state=True)
    selected = plan(problem, method=method, execution="classical", seed=7)
    result = solve(selected)
    state = result.data.artifact(result.artifact).array

    grids = _periodic_coordinates(problem, k)
    # The Result reports its points on the endpoint-exclusive grid.
    assert result.candidate_coordinates == tuple(
        grids[j][0][i] for j, i in enumerate(result.candidate_indices))
    evaluate = sp.lambdify(problem.variables, problem.objective)
    potential = [evaluate(*(grids[j][0][i] for j, i in enumerate(indices)))
                 for indices in itertools.product(range(k), repeat=d)]
    hamiltonian = _kronecker_sum([_circulant(k, h) for _, h in grids]) + np.diag(potential)
    # The uniform start vector errs by 2u (initial_state.restricted_state_error).
    r = selected.reconstruction
    norms = owner._generator_norms(method.theory_flavor, method, owner._grid(selected),
                                   r.support_values, r.steps, r.step_weights)
    delta = expm_multiply_state_error(norms, start=2.0)
    with mpmath.workdps(30):
        # The dyadic float entries convert to mpmath exactly.
        energies, vectors = mpmath.eigsy(mpmath.matrix(hamiltonian.tolist()))
        dimension = k**d
        start = vectors.T * mpmath.matrix([1 / mpmath.sqrt(dimension)] * dimension)
        reference = vectors * mpmath.matrix(
            [mpmath.expj(-mpmath.mpf("0.75") * energies[i]) * start[i] for i in range(dimension)])
        computed = mpmath.matrix([complex(value) for value in state])
        overlap = sum(mpmath.conj(reference[i]) * computed[i] for i in range(dimension))
        # Distance after removing the global phase that the budget leaves free.
        aligned = computed * (mpmath.conj(overlap) / abs(overlap))
        distance = mpmath.norm(aligned - reference)
    assert distance <= delta


@pytest.mark.parametrize("d", [1, 2])
def test_periodic_kinetic_ground_state_is_the_uniform_zero_energy_mode(d):
    """On the periodic grid ``KineticGroundState`` is the uniform state, the zero mode of the circulant stencil.

    The Kronecker sum of the circulant matrices written above has the Fourier
    eigenvalues ``sum_j 2 sin(pi r_j/K)**2/h_j**2``, zero only for r = 0, whose
    eigenvector is constant (``initial_state.KineticGroundState``). The test
    checks with NumPy's eigenvalues that zero is the smallest and simple, up
    to ``10 D u ||H||`` for the backward-stable solver (LAPACK Users' Guide,
    3rd ed., Sec. 4.7), and that the library's start vector is the uniform
    one entry by entry with the uniform state's 2u start term. The dyadic
    grid makes ``H psi`` exactly zero. Through the public workflow, a zero
    objective with gamma = 0 leaves the kinetic term alone, so the evolved
    state stays at the start within the kernel's budget delta
    (``_validation.expm_multiply_state_error`` with the per-call norms of
    ``method._generator_norms``) after its global phase is removed.
    """
    from nwqlib.algorithms.qhd import KineticGroundState, UniformState
    from nwqlib.algorithms.qhd.initial_state import restricted_state, restricted_state_error

    u, k = 2.0**-53, 4
    problem = _problem(d, k, objective=False)
    kinetic = _kronecker_sum([_circulant(k, h) for _, h in _periodic_coordinates(problem, k)])
    energies = np.linalg.eigvalsh(kinetic)
    tolerance = 10 * k**d * u * np.max(np.abs(energies))
    assert abs(energies[0]) <= tolerance < energies[1]
    method = QHD(num_grid_points=k, num_steps=2, total_time=0.75, schedule=STATIC, boundary="periodic",
                 keep_state=True, initial_state=KineticGroundState())
    selected = plan(problem, method=method, execution="classical", seed=7)
    grid = owner._grid(selected)
    start = restricted_state(KineticGroundState(), grid)
    np.testing.assert_array_equal(start, restricted_state(UniformState(), grid))
    assert restricted_state_error(KineticGroundState(), grid) == 2.0
    assert not np.any(kinetic @ start)
    result = solve(selected)
    state = result.data.artifact(result.artifact).array
    overlap = np.vdot(start, state)
    r = selected.reconstruction
    norms = owner._generator_norms(method.theory_flavor, method, owner._grid(selected),
                                   r.support_values, r.steps, r.step_weights)
    delta = expm_multiply_state_error(norms, start=2.0)
    assert np.linalg.norm(state * (np.conj(overlap) / abs(overlap)) - start) <= delta


@pytest.mark.parametrize("d,k", [(1, 4), (1, 6), (2, 4)])
def test_periodic_hopping_blocks_restrict_to_the_circulant_stencil(d, k):
    """The emitted XX+YY blocks, wrap link included, give dt times the circulant off-diagonal.

    A block ``exp(-i tau c (X_p X_q + Y_p Y_q))`` has the generator
    ``tau c (XX + YY)``, and ``XX + YY`` maps ``|10>`` to ``2|01>``. The test
    builds each block's generator from dense Pauli matrices on the ``d K``
    qubits, sums one step and restricts it to the one-hot states, whose
    qubits are ``j K + i``. Second order applies the odd links twice for
    dt/2 and the even links once for dt, so every link, the wrap link
    ``(0, K-1)`` included, must total ``dt (-1/(2 h**2))``, which the
    coefficient ``-1/(4 h**2)`` gives. Every value is a dyadic product or
    sum, so the comparison is exact. Each run of blocks with one duration
    must act on disjoint qubit pairs, the condition for exact layers that
    only even K meets on the cycle. The step's K + K/2 hopping blocks per
    variable are also the count that planning admits
    (``QHD._admit_symbolic_work``) and that the CX law prices at two each.
    """
    problem = _problem(d, k, objective=False)
    method = QHD(num_grid_points=k, num_steps=2, total_time=0.5, schedule=STATIC, boundary="periodic")
    selected = plan(problem, method=method, execution="quantum", seed=7)
    dt, width = 0.25, d * k
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    def two_qubit(pauli, p, q):
        # Qubit 0 is the least significant index bit, the rightmost Kronecker factor.
        factors = [pauli if qubit in (p, q) else np.eye(2) for qubit in reversed(range(width))]
        matrix = factors[0]
        for factor in factors[1:]:
            matrix = np.kron(matrix, factor)
        return matrix

    onehot = [sum(1 << (j * k + i) for j, i in enumerate(indices))
              for indices in itertools.product(range(k), repeat=d)]
    for step in selected.reconstruction.steps:
        assert {b.kind for b in step} == {"kinetic"}
        assert len(step) == d * (k + k // 2)
        generator = sum(b.time_step * b.coefficient * (two_qubit(pauli_x, *b.support)
                                                       + two_qubit(pauli_y, *b.support)) for b in step)
        restricted = generator[np.ix_(onehot, onehot)]
        stencils = [_circulant(k, h) for _, h in _periodic_coordinates(problem, k)]
        expected = dt * _kronecker_sum([s - np.diag(np.diag(s)) for s in stencils])
        assert np.array_equal(restricted, expected)
        for _, run in itertools.groupby(step, key=lambda b: b.time_step):
            qubits = [q for b in run for q in b.support]
            assert len(qubits) == len(set(qubits))
    # Hopping blocks of the two steps, the only blocks of the zero objective.
    blocks = 2 * d * (k + k // 2)
    (cx,) = [law for law in selected.construction.selections[0].resource_laws if law.metric == "cx"]
    # The chain for the uniform state costs 3 d (K - 1) CX (initial_state.append_amplitude_chain).
    assert cx.value == 3 * d * (k - 1) + 2 * blocks
    # The same admission charges the initial state's d K units, the two point values of each
    # midpoint step's weight row and one unit per block (QHD._admit_symbolic_work).
    admitted = width + 2 * 2 + blocks
    with pytest.raises(ValueError, match="max_work"):
        method.revise(max_work=admitted - 1)._admit_symbolic_work(problem, compact=True)
    method.revise(max_work=admitted)._admit_symbolic_work(problem, compact=True)


def test_periodic_native_probabilities_converge_at_second_order():
    """Exact native probabilities approach the restricted model as dt**2.

    With gamma = 0 the classical Schrodinger kernel is exact for the
    restricted Hamiltonian, and the native circuit applies the symmetric
    second-order product of the same one-hot Hamiltonian. A symmetric step
    ``S(dt)`` satisfies ``S(dt) S(-dt) = I``, so its error expansion has only
    even powers, and the 2-norm of the probability difference at fixed T is
    ``e(dt) = c2 dt**2 + c4 dt**4 + O(dt**6)``. The ratio
    ``e(dt)/e(dt/2) = 4 + 3 (c4/c2) dt**2 + O(dt**4)`` therefore approaches 4
    and its distance from 4 shrinks by a factor approaching 4 when dt halves.
    The test requires at least a factor 2 at each of the two refinements.
    A first-order product (ratios near 1.95) fails it, and so does a native
    kinetic term without the wrap link while the classical matrix keeps it,
    a model error that does not shrink (ratios near 1). For this problem the
    ratios are about 3.993, 3.998 and 3.9996.
    """
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=(x - sp.Rational(13, 10)) ** 2 / 4 + x**3 / 20, variables=(x,),
                           bounds=((0.0, 4.0),))
    method = QHD(num_grid_points=4, total_time=2.0, schedule=STATIC, boundary="periodic")
    errors = []
    for steps in (8, 16, 32, 64):
        chosen = method.revise(num_steps=steps)
        native = solve(problem, method=chosen, seed=7)
        classical = solve(problem, method=chosen, execution="classical", seed=7)
        # With one variable the marginal of grid point i is its probability.
        errors.append(np.linalg.norm(np.subtract(native.marginals.array[0], classical.marginals.array[0])))
    ratios = [coarse / fine for coarse, fine in itertools.pairwise(errors)]
    deviations = [abs(ratio - 4) for ratio in ratios]
    assert all(fine <= coarse / 2 for coarse, fine in itertools.pairwise(deviations)), ratios


@pytest.mark.parametrize("fields,message", [
    (dict(num_grid_points=4, include_boundary_points=True), "has no boundary points"),
    (dict(num_grid_points=5), "even num_grid_points of at least 4, got 5"),
    (dict(num_grid_points=2), "even num_grid_points of at least 4, got 2"),
])
def test_periodic_grid_rejects_boundary_points_and_odd_or_two_points(fields, message):
    """The periodic grid has no endpoints, its two-layer link split needs an even K, and K = 2 is degenerate.

    The same fields remain legal on the Dirichlet grid, and a legal periodic
    Method refuses the same values when revised.
    """
    with pytest.raises(ValueError, match=message):
        QHD(boundary="periodic", **fields)
    with pytest.raises(ValueError, match=message):
        QHD(boundary="periodic", num_grid_points=6).revise(**fields)
    assert QHD(**fields).boundary == "dirichlet"


def test_augmented_lagrangian_layer_uses_the_periodic_grid():
    """The AL layer's grid check, inner Plans and grid reference use the Method's periodic grid.

    ``(x-1)**2 + (y-1)**2`` subject to ``x**2 + y**2 <= 1`` on the periodic box
    [0, 2)² with K = 4 has the points 0, 1/2, 1 and 3/2 per variable. These
    points and the values of f and g on them are dyadic, so the enumeration
    below is exact, and the feasibility test ``g <= 0`` selects the same points
    as the run's tolerance 1e-9, because the nearest infeasible point has
    g = 1/4. The enumeration finds six feasible points and the feasible grid
    minimum 1/2 at (1/2, 1/2). The Dirichlet interior grid of the same box,
    2/5, 4/5, 6/5 and 8/5, would give 2/5 at (2/5, 4/5) instead. The added term
    ``1/(5x - 2)`` is singular at the Dirichlet point x = 2/5 and finite at the
    periodic points, so the layer's check of f on the grid (``constrained._setup``)
    passes only on the periodic grid.
    """
    from nwqlib.algorithms.qhd import AugmentedLagrangian, constrained_grid_minimum, solve_augmented_lagrangian
    from nwqlib.problems import ConstrainedOptimization

    x, y = sp.symbols("x y")
    objective, disk = (x - 1) ** 2 + (y - 1) ** 2, x**2 + y**2 - 1
    box = ((0.0, 2.0), (0.0, 2.0))
    qhd = QHD(num_grid_points=4, num_steps=1, total_time=0.5, boundary="periodic")
    options = AugmentedLagrangian(inner_point="best_observed")

    def run(expression):
        problem = ConstrainedOptimization(objective=expression, variables=(x, y), bounds=box,
                                          inequalities=(disk,))
        return solve_augmented_lagrangian(problem, qhd=qhd, options=options, execution="classical", seed=11,
                                          progress=False)

    result = run(objective)
    axis = [0.0, 0.5, 1.0, 1.5]
    points = list(itertools.product(axis, axis))
    feasible = [p for p in points if p[0] ** 2 + p[1] ** 2 - 1 <= 0]
    value, point = min(((p[0] - 1) ** 2 + (p[1] - 1) ** 2, p) for p in feasible)
    assert (value, point, len(feasible)) == (0.5, (0.5, 0.5), 6)
    assert all(inner.plan.method.boundary == "periodic" for inner in result.results)
    assert all(item.evaluation.point in points for item in result.iterations)
    reference = constrained_grid_minimum(result)
    assert (reference.objective, reference.point, reference.feasible_points) == (value, point, len(feasible))
    assert reference.grid_points == 16
    assert run(objective + 1 / (5 * x - 2)).iterations
