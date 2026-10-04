"""Natural QHD, independent finite differences, actual 2–4 qubit readouts."""

import itertools
from unittest.mock import Mock
import numpy as np
import pytest
import scipy.linalg
import sympy as sp
from nwqlib.algorithms.qhd import (
    QHD, CubicSchedule, GaussianState, KineticGroundState, QHDVerification, QuadraticSchedule,
    ShiftedCubicSchedule, UniformState,
)
from nwqlib.algorithms.qhd.compiler import QHDCompiler
from nwqlib.algorithms.qhd import method as owner
from nwqlib.algorithms.qhd.records import SupportValues, table_extrema
from nwqlib._validation import NUMERICAL_RELATION_RTOL
from nwqlib.problems import Optimization
from nwqlib.scientist import plan, solve
from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan


def make_plan(problem=None, method=None, *, execution="quantum", shots=None):
    x = sp.Symbol("x")
    problem = problem or Optimization(
        objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),)
    )
    return plan(
        problem,
        method=method
        or QHD(num_grid_points=3, num_steps=2, total_time=0.17, schedule=QuadraticSchedule(gamma=0.3)),
        execution=execution,
        shots=shots,
        seed=7,
    )


def _independent_start(initial, x):
    """One-variable initial amplitudes written from their definitions, normalized with NumPy."""
    k = len(x)
    if isinstance(initial, KineticGroundState):
        values = np.sin(np.pi * np.arange(1, k + 1) / (k + 1))
    elif isinstance(initial, GaussianState):
        (center,), (width,) = initial.center, initial.widths
        values = np.exp(-((x - center) ** 2) / (2 * width**2))
    else:
        values = np.ones(k)
    return values.astype(complex) / np.linalg.norm(values)


def _independent_state(binding, options):
    """Small test-only finite-difference/product reference; no production builder."""
    assert len(binding.variables) == 1
    k = options.num_grid_points
    lo, hi = binding.bounds[0]
    if options.include_boundary_points:
        # k equally spaced points including both box endpoints.
        x, h = np.linspace(lo, hi, k), (hi - lo) / (k - 1)
    else:
        h = (hi - lo) / (k + 1)
        x = np.array([lo + (i + 1) * h for i in range(k)])
    potential = np.array([float(binding.objective.subs(binding.variables[0], v)) for v in x])
    kinetic = (
        np.diag(np.full(k, 1 / h**2))
        + np.diag(np.full(k - 1, -1 / (2 * h * h)), 1)
        + np.diag(np.full(k - 1, -1 / (2 * h * h)), -1)
    )
    state = _independent_start(options.initial_state, x)
    dt = options.total_time / options.num_steps
    for step in range(options.num_steps):
        t = (step + 0.5) * dt
        v = 1 + options.schedule.gamma * t * t
        a = 1 / v
        if options.theory_flavor == "schrodinger":
            state = scipy.linalg.expm(-1j * dt * (a * kinetic + v * np.diag(potential))) @ state
        else:
            duration = dt / (2 if options.trotter_order == 2 else 1)
            state = np.exp(-1j * duration * v * potential) * state
            groups = (
                (
                    (range(1, k - 1, 2), dt / 2),
                    (range(0, k - 1, 2), dt),
                    (range(1, k - 1, 2), dt / 2),
                )
                if options.trotter_order == 2
                else ((range(k - 1), dt),)
            )
            for edges, kinetic_duration in groups:
                for edge in edges:
                    matrix = np.zeros((k, k))
                    matrix[edge, edge + 1] = matrix[edge + 1, edge] = -a / (2 * h * h)
                    state = scipy.linalg.expm(-1j * kinetic_duration * matrix) @ state
            state *= np.exp(-1j * dt * a / h**2)
            if options.trotter_order == 2:
                state = np.exp(-1j * duration * v * potential) * state
    return state


@pytest.mark.parametrize(
    "execution,flavor,boundary,initial",
    [
        ("classical", "schrodinger", False, UniformState()),
        ("classical", "ir_product", False, UniformState()),
        ("quantum", "ir_product", False, UniformState()),
        ("classical", "schrodinger", True, UniformState()),
        ("classical", "ir_product", True, UniformState()),
        # A Gaussian off the box center has three distinct amplitudes, so a
        # route that started elsewhere or reversed the grid would differ.
        ("classical", "schrodinger", False, GaussianState(center=(0.3,), widths=(0.4,))),
        ("classical", "ir_product", False, GaussianState(center=(0.3,), widths=(0.4,))),
        ("quantum", "ir_product", False, GaussianState(center=(0.3,), widths=(0.4,))),
        ("classical", "schrodinger", True, KineticGroundState()),
    ],
    ids=lambda value: value.kind if hasattr(value, "kind") else None,
)
def test_qhd_statevector_matches_ir_product_theory_distribution(execution, flavor, boundary, initial):
    """Every route evolves the selected initial state like an independent 3-point model.

    ``_independent_state`` writes the start amplitudes from their definitions
    and applies dense exponentials, so the comparison covers the classical
    start vector of both flavors and the native chain.
    """
    selected = make_plan(
        method=QHD(
            num_grid_points=3,
            num_steps=2,
            total_time=0.17,
            schedule=QuadraticSchedule(gamma=0.3),
            theory_flavor=flavor,
            keep_state=True,
            include_boundary_points=boundary,
            initial_state=initial,
        ),
        execution=execution,
    )
    assert selected.reconstruction.width == 3
    result = solve(selected)
    state = result.data.artifact(result.artifact).array
    if execution == "quantum":
        state = state[[1, 2, 4]]
    # Independent 3x3 finite-difference dynamics, including absolute phase.
    expected = _independent_state(selected.problem, selected.method)
    np.testing.assert_allclose(state, expected, rtol=0, atol=1e-12)
    assert result.objective == pytest.approx(
        float(selected.problem.objective.subs(selected.problem.variables[0], result.candidate[0])),
        abs=1e-14,
    )


def test_onehot_ir_product_matches_dense_exponentials_of_its_stored_blocks():
    """The one-hot ir_product state equals the ordered dense exponentials of the stored blocks at d = 2, K = 4.

    The reference applies ``exp(-i G_b)`` of every stored block, in stored
    order, at its stored angle, ``G = angle (Pi_S - 2**-s I)`` for a projector
    and ``angle X`` on the two linked slices for a hopping, to the exact
    uniform start state ``1/4``, then the stored physical phase. The allowed
    difference is the direct kernel's state budget
    (``theory.onehot_product_state_error``) plus the dense oracle's own
    numerical allowance, the state-error law of exponential actions
    ``_validation.expm_multiply_state_error`` for one call per block with
    shifted 1-norm ``|angle|`` and one stored entry per row. Admission: the
    direct workflow is charged ``theory.onehot_product_sizes`` and the oracle
    separately ``_linalg_laws.expm_requirements`` per block plus its
    matrix formation, matrix-vector product and comparison; both routes' work
    adds against this test's cap, and the byte peak is the larger sequential
    workspace plus the retained states.
    """
    from nwqlib._linalg_laws import expm_requirements
    from nwqlib._validation import expm_multiply_state_error
    from nwqlib.algorithms.qhd.theory import onehot_product_sizes, onehot_product_state_error

    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2 + x * y + (y + sp.Rational(1, 3)) ** 2,
                           variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    selected = make_plan(problem, QHD(num_grid_points=4, num_steps=2, total_time=1.3, theory_flavor="ir_product",
                                      initial_state=UniformState(), keep_state=True), execution="classical")
    r = selected.reconstruction
    blocks = [block for group in r.steps for block in group]
    k, d = 4, 2
    dimension = k**d
    # The test's stated caps for both routes together.
    work_cap, byte_cap = 10**8, 10**8
    direct_work, direct_bytes = onehot_product_sizes(dimension, k, r.steps, start_work=dimension,
                                                     start_bytes=16 * dimension, held_bytes=0)
    oracle_work = oracle_bytes = 0
    for block in blocks:
        work, data = expm_requirements(dimension, abs(block.angle))
        # Dense generator formation, one matrix-vector product and the final comparison.
        oracle_work += work + dimension**2 + dimension**2
        oracle_bytes = max(oracle_bytes, data + 16 * dimension**2)
    oracle_work += dimension
    # Sequential routes: the larger workspace, plus the two retained states.
    assert direct_work + oracle_work <= work_cap
    assert max(direct_bytes, oracle_bytes) + 2 * 16 * dimension <= byte_cap

    result = solve(selected)
    state = result.data.artifact(result.artifact).array
    reference = np.full(dimension, 0.25, dtype=complex)
    points = np.indices((k,) * d).reshape(d, -1)
    for block in blocks:
        if block.kind == "number_projector":
            active = np.ones(dimension, dtype=bool)
            for qubit in block.support:
                j, i = divmod(qubit, k)
                active &= points[j] == i
            generator = block.angle * (np.diag(active.astype(float)) - np.eye(dimension) / 2 ** len(block.support))
        else:
            (j, p), (_, q) = (divmod(qubit, k) for qubit in block.support)
            generator = np.zeros((dimension, dimension))
            for column in range(dimension):
                if points[j][column] in (p, q):
                    other = points[:, column].copy()
                    other[j] = q if points[j][column] == p else p
                    generator[np.ravel_multi_index(tuple(other), (k,) * d), column] = block.angle
        reference = scipy.linalg.expm(-1j * generator) @ reference
    reference *= np.exp(1j * r.physical_phase)
    allowance = (onehot_product_state_error(r.steps, start=r.initial_state_error)
                 + expm_multiply_state_error([(abs(block.angle), 1) for block in blocks],
                                             start=r.initial_state_error, phase_multiplications=1))
    assert np.linalg.norm(state - reference) <= allowance


@pytest.mark.parametrize("recipe", ["structured", "qiskit_state_preparation"])
def test_actual_native_selected_product_phase_and_variable_order(recipe):
    """An independently constructed four-state Strang model checks physical phase and the
    nontrivial one-hot index embedding for both native preparations of the uniform state.
    """
    x, y = sp.symbols("x y")
    problem = Optimization(
        objective=x * y + 2 * x - y + sp.Rational(1, 5),
        variables=(x, y),
        bounds=((-1.0, 2.0), (0.0, 6.0)),
    )
    selected = make_plan(problem, QHD(num_steps=2, total_time=0.17, schedule=QuadraticSchedule(gamma=0.3),
                                      keep_state=True, initial_state_preparation=recipe))
    assert selected.reconstruction.width == 4
    result = solve(selected)
    state = result.data.artifact(result.artifact).array
    kinetic = np.array(
        [
            [1.25, -0.125, -0.5, 0],
            [-0.125, 1.25, 0, -0.5],
            [-0.5, 0, 1.25, -0.125],
            [0, -0.5, -0.125, 1.25],
        ]
    )
    potential = np.array([-1.8, -3.8, 2.2, 2.2])
    expected = np.ones(4, dtype=complex) / 2
    dt = 0.17 / 2
    for step in range(2):
        t = (step + 0.5) * dt
        weight = 1 + 0.3 * t * t
        expected = np.exp(-1j * dt * weight * potential / 2) * expected
        expected = scipy.linalg.expm(-1j * dt * kinetic / weight) @ expected
        expected = np.exp(-1j * dt * weight * potential / 2) * expected
    full = np.zeros(16, dtype=complex)
    full[[5, 9, 6, 10]] = expected
    np.testing.assert_allclose(state, full, rtol=0, atol=1e-12)
    assert result.candidate_indices == (0, 1)
    assert result.valid_probability == pytest.approx(1.0, abs=1e-12)
    assert result.data.trace.jobs == 1
    assert result.analyze().candidate == result.candidate


@pytest.mark.parametrize("variables", [1, 2])
def test_initial_state_amplitudes_match_their_definitions(variables):
    """The classical start vector of each initial state is its defining product, within its derived bound.

    For each state and both Dirichlet grids the test writes the per-variable
    amplitudes from their definitions in 50-digit mpmath at the binary64
    grid coordinates, normalizes them and forms the product in lexicographic
    grid order, variable 0 most significant. The 2-norm distance of
    ``initial_state.restricted_state`` must not exceed the bound
    ``initial_state.restricted_state_error`` times u, whose owner derives it
    (first order for the kinetic ground state, whose second-order terms are
    below 1e-9 of the bound, and finite for the Gaussian). The mpmath
    evaluation errs by about 1e-50. Besides a Gaussian inside the box, the
    cases cover a center 38.7 widths beyond the last interior grid point,
    whose unnormalized amplitudes on that grid all underflow before the
    shift, a far and wide Gaussian whose bound grows with ``10 z`` (about
    1e5 here), a width of 1e-200 whose off-center exponents overflow, and,
    for one variable, extreme cases on [0, 1] (``GaussianState._variables``).
    With center -1.7e308 and width 1e308 the exponents stay finite. With
    the same center and width 2**-1074 every exponent overflows, the rounded
    distances tie, and the uniform vector, about 1.05 from the exact first
    basis vector, carries the cap of ``restricted_state_error``, about
    sqrt(2). With center 10 and width 2**-1074 every exponent overflows too,
    but the nearest point is unique and well separated, and it gets the whole
    state with a bound near zero. A symmetric Gaussian of width 8.8e-10 on
    the endpoint grid {-1, 1} has entry-error bounds near 1.1e154, whose
    norm the owner forms without overflow before the cap applies.

    The kinetic ground state must be an eigenvector of an independently built
    kinetic matrix H, ``(1/h_j**2)(I - (S + S^T)/2)`` on factor j summed over
    the variables, for its smallest eigenvalue. H is positive semidefinite,
    so ``||H - lambda I|| <= ||H||`` and the residual
    ``||H psi - lambda psi||`` of the computed psi is at most
    ``||H|| delta`` with delta the owner's bound. The dense product adds at
    most ``(2d + 1) u || |H| ||`` for rows of 2d + 1 nonzeros, and
    ``|H|`` has the spectrum of H, because flipping the sign of every other
    basis vector of each factor maps one to the other. The backward-stable
    ``eigvalsh`` errs by a small multiple of ``u ||H||`` (LAPACK Users'
    Guide, 3rd ed., Sec. 4.7), taken as ``10 D u ||H||`` for dimension D, and
    so does the formula ``sum_j 2 sin(pi/(2 (K + 1)))**2/h_j**2`` it is
    compared with. The grid with boundary points has the same matrix with
    another spacing, so its ground state is the same vector.
    """
    import mpmath
    from nwqlib.algorithms.qhd.grid import OneHotGrid
    from nwqlib.algorithms.qhd.initial_state import restricted_state, restricted_state_error

    mpmath.mp.dps = 50
    u = 2.0**-53
    k = 5
    symbols = sp.symbols("x y")[:variables]
    bounds = ((-1.0, 2.0), (3.0, 7.0))[:variables]
    gaussians = tuple(
        GaussianState(center=center[:variables], widths=widths[:variables])
        for center, widths in (
            ((0.4, 5.5), (1.1, 0.8)),
            ((1.5 + np.sqrt(1500.0), 5.5), (1.0, 0.8)),
            ((1.0e6, 5.5), (1.0e4, 0.8)),
            ((0.5, 5.0), (1e-200, 0.8)),
        )
    )

    def check(state, grid):
        """Compare restricted_state with the mpmath product within the owner's bound."""
        n = grid.num_grid_points
        factors = []
        for j in range(grid.num_variables):
            if isinstance(state, UniformState):
                values = [mpmath.mpf(1)] * n
            elif isinstance(state, KineticGroundState):
                values = [mpmath.sin(mpmath.pi * (i + 1) / (n + 1)) for i in range(n)]
            else:
                c, s = mpmath.mpf(state.center[j]), mpmath.mpf(state.widths[j])
                exponents = [(mpmath.mpf(grid.grid_value(j, i)) - c) ** 2 / (2 * s**2) for i in range(n)]
                values = [mpmath.exp(min(exponents) - z) for z in exponents]
            norm = mpmath.sqrt(mpmath.fsum(v**2 for v in values))
            factors.append([v / norm for v in values])
        exact = [mpmath.fprod(factors[j][i] for j, i in enumerate(indices))
                 for indices in itertools.product(range(n), repeat=grid.num_variables)]
        computed = restricted_state(state, grid)
        assert computed.shape == (n**grid.num_variables,) and not np.any(computed.imag)
        distance = mpmath.sqrt(mpmath.fsum((mpmath.mpf(float(z.real)) - e) ** 2
                                           for z, e in zip(computed, exact, strict=True)))
        assert distance <= restricted_state_error(state, grid) * u, (state, grid.include_boundary_points)

    for boundary in (False, True):
        grid = OneHotGrid(symbols, bounds, k, boundary)
        for state in (UniformState(), KineticGroundState(), *gaussians):
            check(state, grid)
        if variables == 1:
            unit = OneHotGrid(symbols, ((0.0, 1.0),), k, boundary)
            check(GaussianState(center=(-1.7e308,), widths=(1e308,)), unit)
            # Every exponent overflows, and every halved distance rounds to 0.85e308.
            tied = GaussianState(center=(-1.7e308,), widths=(2.0**-1074,))
            check(tied, unit)
            np.testing.assert_array_equal(restricted_state(tied, unit), restricted_state(UniformState(), unit))
            assert np.sqrt(2) < restricted_state_error(tied, unit) * u < np.sqrt(2) * (1 + 2e-12)
            # Every exponent overflows. The nearest point is unique and well separated.
            saturated = GaussianState(center=(10.0,), widths=(2.0**-1074,))
            check(saturated, unit)
            np.testing.assert_array_equal(restricted_state(saturated, unit), [0, 0, 0, 0, 1])
            assert restricted_state_error(saturated, unit) < 4
            # Endpoint grid {-1, 1}, center 0: the exact state is uniform. Both
            # exponents are about 6.4e17, so the exponent-error enclosures give
            # entry errors near 1.1e154, whose squares sum beyond the binary64
            # range while their norm does not (initial_state._gaussian_beta).
            # The bound reaches the cap, just above sqrt(2).
            symmetric = GaussianState(center=(0.0,), widths=(8.845950911055478e-10,))
            pair = OneHotGrid(symbols, ((-1.0, 1.0),), 2, True)
            check(symmetric, pair)
            np.testing.assert_array_equal(restricted_state(symmetric, pair), restricted_state(UniformState(), pair))
            assert np.sqrt(2) < restricted_state_error(symmetric, pair) * u < np.sqrt(2) * (1 + 2e-12)
            make_plan(Optimization(objective=symbols[0], variables=symbols, bounds=((-1.0, 1.0),)),
                      QHD(num_grid_points=2, include_boundary_points=True, initial_state=symmetric))

        # Independent dense kinetic matrix on the lexicographic grid.
        kinetic = np.zeros((k**variables, k**variables))
        for j in range(variables):
            h = grid.spacing(j)
            one = (np.eye(k) - 0.5 * (np.eye(k, k=1) + np.eye(k, k=-1))) / h**2
            factors = [np.eye(k)] * variables
            factors[j] = one
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            kinetic += term
        scale = np.linalg.norm(kinetic, 2)
        eigenvalue = 10 * k**variables * u * scale
        lowest = np.linalg.eigvalsh(kinetic)[0]
        expected_energy = sum(2 * np.sin(np.pi / (2 * (k + 1))) ** 2 / grid.spacing(j) ** 2 for j in range(variables))
        assert abs(lowest - expected_energy) <= 2 * eigenvalue
        state = KineticGroundState()
        psi = restricted_state(state, grid).real
        residual = np.linalg.norm(kinetic @ psi - lowest * psi)
        delta = restricted_state_error(state, grid) * u
        assert residual <= scale * (delta + (2 * variables + 1) * u) + eigenvalue

    # A center 1e400 widths from the grid overflows every exponent, and the
    # distances round to one value. The tie leaves the uniform vector and the
    # cap sqrt(N**2 + 1) with N the computed norm bound, just above sqrt(2).
    x = symbols[0]
    line = Optimization(objective=x, variables=(x,), bounds=((-1.0, 2.0),))
    extreme = GaussianState(center=(1e300,), widths=(1e-100,))
    grid = OneHotGrid((x,), line.bounds, k)
    np.testing.assert_allclose(restricted_state(extreme, grid), np.full(k, 1 / np.sqrt(k)), rtol=4 * u, atol=0)
    # The upward factor _OUTWARD = 1 + 2**-40 (about 9.1e-13) dominates the excess.
    assert np.sqrt(2) < restricted_state_error(extreme, grid) * u < np.sqrt(2) * (1 + 2e-12)
    # The public workflow accepts it, and its host windows stay finite.
    result = solve(make_plan(line, QHD(num_grid_points=k, initial_state=extreme), execution="classical"))
    assert np.isfinite(result.most_probable_tie_window)
    with pytest.raises(ValueError, match="one entry per Problem variable, 1 here"):
        make_plan(line, QHD(initial_state=GaussianState(center=(0.0, 0.0), widths=(1.0, 1.0))))
    if variables == 1:
        # The default kinetic ground state's bound grows with sqrt(K**d), and its owner forms it without
        # converting K**d to a float, so a resource-only Plan on 520 variables keeps the exact dimension
        # 4**520 beside a finite bound.
        many = sp.symbols("x:520")
        wide = make_plan(Optimization(objective=many[0] ** 2, variables=many, bounds=((-1.0, 1.0),) * 520),
                         QHD(num_grid_points=4, num_steps=1))
        assert (wide.reconstruction.width, wide.reconstruction.restricted_dimension) == (2080, 4**520)
        assert np.isfinite(wide.reconstruction.initial_state_error)


@pytest.mark.parametrize("recipe", ["structured", "qiskit_state_preparation"])
@pytest.mark.parametrize(
    "initial",
    [UniformState(), KineticGroundState(), GaussianState(center=(0.4, 5.5), widths=(1.1, 0.8))],
    ids=["uniform", "kinetic_ground", "gaussian"],
)
def test_native_preparation_equals_the_classical_initial_vector(initial, recipe):
    """The prepared native register, restricted to one-hot states, is the classical start vector.

    The objective is zero and ``rotation_threshold=1e6`` omits every hopping
    rotation, so the native circuit of this public solve is the preparation
    with the restored physical phase. With two variables and K = 4, grid
    tuple (i, j) is the basis index ``2**i + 2**(4 + j)``, listed in
    lexicographic order, and every other amplitude must vanish.

    Tolerance, a first-order bound in u = 2**-53. The simulator's final state
    is within ``receipt.state_error`` of the native circuit
    (``_validation.native_state_error``). For Qiskit's ``StatePreparation``
    the receipt excludes its synthesized ``unitary`` instructions because
    their unitarity is an input property. The test resolves that exclusion by
    assuming that the synthesized one- and two-qubit matrices are unitary to
    roundoff, so that the per-instruction Aer constant applies to them. The
    chain's angles move each register by at most
    ``(K (K - 1)/2 + pi (K - 1)) u`` (``initial_state.append_amplitude_chain``),
    and the test assumes, without a derivation, that the angles of Qiskit's
    synthesis move it by no more. The per-variable
    vectors that the circuit receives lie within ``sum_j E_j u`` of the exact
    factors, where E_j is ``variable_errors`` (at most
    ``restricted_state_error``) or 2 for each uniform factor, and the
    classical vector lies within ``restricted_state_error`` u of the exact
    state (``initial_state.restricted_state_error``), so
    ``(2 restricted_state_error + 2d) u`` covers both.
    """
    from nwqlib.algorithms.qhd.initial_state import restricted_state, restricted_state_error

    x, y = sp.symbols("x y")
    problem = Optimization(objective=sp.Integer(0), variables=(x, y), bounds=((-1.0, 2.0), (3.0, 7.0)))
    k = 4
    selected = make_plan(problem, QHD(
        num_grid_points=k, num_steps=1, total_time=1.0, schedule=QuadraticSchedule(gamma=0.0),
        rotation_threshold=1e6, keep_state=True, initial_state=initial, initial_state_preparation=recipe,
    ))
    assert selected.reconstruction.steps == ((),)
    result = solve(selected)
    native = result.data.artifact(result.artifact).array * np.exp(-1j * selected.reconstruction.physical_phase)
    onehot = [(1 << i) | (1 << (k + j)) for i, j in itertools.product(range(k), repeat=2)]
    classical = restricted_state(initial, owner._grid(selected))
    (receipt,) = result.data.receipts
    delta, reason = receipt.state_error(("amplitude-derived masses", "unitary"))
    assert reason is None
    u = 2.0**-53
    construction = 2 * restricted_state_error(initial, owner._grid(selected)) + 2 * 2
    angles = 2 * (k * (k - 1) / 2 + np.pi * (k - 1))
    tolerance = delta + (construction + angles) * u
    assert np.linalg.norm(native[onehot] - classical) <= tolerance
    assert np.linalg.norm(np.delete(native, onehot)) <= tolerance


@pytest.mark.parametrize(
    "execution,flavor",
    [("classical", "schrodinger"), ("classical", "ir_product"), ("classical", "split_step"),
     ("quantum", "ir_product")],
)
@pytest.mark.parametrize(
    "initial",
    [KineticGroundState(), GaussianState(center=(0.1, -0.2), widths=(0.5, 0.7))],
    ids=["kinetic_ground", "gaussian"],
)
def test_initial_state_is_evaluated_once_at_planning(execution, flavor, initial, monkeypatch):
    """Planning evaluates the initial state once and stores it, and solve and analyze only read it.

    ``QHD.plan`` admits one evaluation of the per-variable amplitudes and of
    the start vector's error bound, and stores both in the reconstruction
    (``initial_amplitudes``, ``initial_state_error``). Planning must evaluate
    the amplitudes exactly once, counted at ``GaussianState._variables`` or
    ``KineticGroundState.variable_amplitudes``, and the stored values must
    equal a fresh evaluation bitwise. A classical Plan also stores the
    kernel's state budget delta, which must equal a fresh
    ``method._host_state_error`` of the Plan's own stored tables, steps,
    step rows and start error, so a planning call that passes other data
    fails. After planning every evaluation entry point raises, so a later
    evaluation, work outside that admission, fails the solve or the analysis.
    """
    from nwqlib.algorithms.qhd.initial_state import restricted_state_error

    x, y = sp.symbols("x y")
    problem = Optimization(objective=x**2 + x * y / 4, variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    name = "_variables" if isinstance(initial, GaussianState) else "variable_amplitudes"
    original, calls = getattr(type(initial), name), []

    def counting(self, grid):
        calls.append(grid)
        return original(self, grid)

    with monkeypatch.context() as count:
        count.setattr(type(initial), name, counting)
        selected = make_plan(problem, QHD(
            num_grid_points=4, num_steps=2, total_time=0.5, schedule=QuadraticSchedule(gamma=0.3),
            theory_flavor=flavor, keep_state=True, initial_state=initial,
        ), execution=execution)
    assert len(calls) == 1
    grid = owner._grid(selected)
    stored = selected.reconstruction
    assert len(stored.initial_amplitudes) == 2 and all(
        np.array_equal(a.array, np.asarray(v, dtype=np.float64))
        for a, v in zip(stored.initial_amplitudes, initial.variable_amplitudes(grid), strict=True))
    assert stored.initial_state_error == restricted_state_error(initial, grid)
    if execution == "classical":
        assert stored.host_state_error == owner._host_state_error(
            selected.method, grid, stored.support_values, stored.steps, stored.step_weights,
            stored.initial_state_error)
    for record in (UniformState, KineticGroundState, GaussianState):
        for name in ("variable_amplitudes", "variable_errors", "_variables"):
            if hasattr(record, name):
                monkeypatch.setattr(record, name, Mock(side_effect=AssertionError("initial state evaluated")))
    result = solve(selected)
    assert result.analyze().candidate == result.candidate


@pytest.mark.parametrize("d,k", [(1, 5), (2, 4), (3, 3)])
def test_start_vector_work_counts_the_entries_its_construction_forms(d, k, monkeypatch):
    """Each classical kernel law charges at least the entries that forming the start vector touches, once.

    ``restricted_state`` forms the tensor product of the stored factors of the
    kinetic ground state on the Dirichlet grid with ``np.kron``. The test
    counts the entries of every Kronecker product it forms and requires
    ``initial_state.start_vector_work`` to cover them with the d converted
    vectors and the complex cast of the D entries, and the uniform fill to be
    charged at least its D entries. The Schrodinger and split-step Plans of
    the kinetic ground state must charge at least that difference more than
    those of the uniform state, so each law includes the start vector.
    """
    from nwqlib.algorithms.qhd.initial_state import restricted_state, start_vector_work

    variables = sp.symbols("x y z")[:d]
    problem = Optimization(objective=sum(v**2 for v in variables), variables=variables,
                           bounds=((-1.0, 1.0),) * d)
    method = QHD(num_grid_points=k, num_steps=2, total_time=0.5, initial_state=KineticGroundState())
    grid = owner._grid(make_plan(problem, method, execution="classical"))
    formed = []
    kron = np.kron

    def counting(a, b):
        product = kron(a, b)
        formed.append(product.size)
        return product

    monkeypatch.setattr(np, "kron", counting)
    restricted_state(KineticGroundState(), grid)
    monkeypatch.undo()
    touched = d * k + sum(formed) + k**d
    assert start_vector_work(KineticGroundState(), grid) >= touched
    assert start_vector_work(UniformState(), grid) >= k**d
    for flavor in ("schrodinger", "split_step"):
        ground, uniform = (
            make_plan(problem, method.revise(theory_flavor=flavor, initial_state=state), execution="classical")
            for state in (KineticGroundState(), UniformState())
        )
        assert ground.reconstruction.size_units - uniform.reconstruction.size_units >= touched - k**d


@pytest.mark.parametrize(
    "d,k,state",
    [
        (1, 2, GaussianState(center=(0.4,), widths=(0.15,))),
        (1, 4, GaussianState(center=(0.4,), widths=(0.15,))),
        (2, 4, GaussianState(center=(0.4, 0.4), widths=(0.15, 0.15))),
        (1, 64, GaussianState(center=(0.4,), widths=(0.15,))),
        (2, 4, KineticGroundState()),
    ],
    ids=["gaussian-1-2", "gaussian-1-4", "gaussian-2-4", "gaussian-1-64", "kinetic_ground-2-4"],
)
def test_initial_state_planning_stage_fits_its_measured_allowance(d, k, state):
    """Planning's initial-state stage, its evaluation and its stored payload, fits ``method._initial_state_bytes``.

    The allowance is measured and specific to the implementation and runtime
    that ``method._initial_state_bytes`` names. The test repeats the stage
    that ``QHD.plan`` runs, ``initial_state.evaluate`` and then
    ``initial_state.stored_amplitudes`` with the normalized arrays still
    alive, under tracemalloc after one warm-up call, and requires the traced
    peak to fit. At d = 1, K = 2 the per-point coefficient alone, 640 bytes,
    is below the traced evaluation.
    """
    import tracemalloc
    from nwqlib.algorithms.qhd.initial_state import evaluate, stored_amplitudes

    variables = sp.symbols("x y")[:d]
    problem = Optimization(objective=sum(v**2 for v in variables), variables=variables, bounds=((0.0, 1.0),) * d)
    selected = make_plan(problem, QHD(num_grid_points=k, initial_state=state), execution="classical")
    grid = owner._grid(selected)

    def stage():
        amplitudes, error = evaluate(state, grid)
        return amplitudes, stored_amplitudes(amplitudes), error

    stage()
    tracemalloc.start()
    try:
        kept = stage()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del kept
    assert peak <= owner._initial_state_bytes(d, k)


def test_chain_stops_at_a_zero_tail():
    """A register whose amplitudes end in zeros gets no link past its last positive amplitude.

    The vectors (0, 0.6, 0.8, 0) and (1, 0, 0, 0) have their last positive
    entries at local indices 2 and 0, so the chain emits two links (X, then
    CRY and CX twice) and one bare X. Without the stop the next angle would
    be the undefined 0/0. The one-hot amplitudes must equal the product of
    the two vectors, which here is exact in binary64, within the chain's
    angle bound ``(K (K - 1)/2 + pi (K - 1)) u`` per register
    (``initial_state.append_amplitude_chain``) plus the dense simulation of
    six small gates, a few u each, covered by 100u in total.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from nwqlib.algorithms.qhd.grid import OneHotGrid
    from nwqlib.algorithms.qhd.initial_state import append_amplitude_chain, chain_links

    x, y = sp.symbols("x y")
    grid = OneHotGrid((x, y), ((0.0, 5.0), (0.0, 5.0)), 4)
    vectors = (np.array([0.0, 0.6, 0.8, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert [chain_links(v) for v in vectors] == [2, 0]
    circuit = QuantumCircuit(grid.num_qubits)
    append_amplitude_chain(circuit, grid, vectors)
    assert dict(circuit.count_ops()) == {"x": 2, "cry": 2, "cx": 2}
    state = Statevector(circuit).data
    onehot = [(1 << i) | (1 << (4 + j)) for i, j in itertools.product(range(4), repeat=2)]
    u = 2.0**-53
    tolerance = (2 * (4 * 3 / 2 + np.pi * 3) + 100) * u
    assert np.linalg.norm(state[onehot] - np.kron(*vectors)) <= tolerance
    assert np.linalg.norm(np.delete(state, onehot)) <= tolerance


def test_resource_only_initial_state_counts_the_evolution_blocks_alone():
    """initial_state_preparation='none' keeps the evolution cost and drops the preparation.

    The hand count uses two variables, K = 3, one first-order step and a zero
    objective. The step has two XX+YY hopping blocks per variable and no
    projector block. Each hopping block costs 1 + s + s**2 = 7 construction
    units on s = 2 qubits and two CX, so the evolution alone costs 28 units
    and 8 CX. The chain for the uniform state adds d + 2 d (K - 1) = 10
    emitted gates and 3 d (K - 1) = 12 CX, which the gates of the actual
    circuit confirm. A Gaussian whose first factor underflows past its first
    point has one positive amplitude there and three in the second register,
    so the chain emits 0 + 2 links, d + 2*2 = 6 gates and 3*2 = 6 CX. Qiskit's
    StatePreparation adds d K 2**K = 48 units, and without a CX bound for it
    the Plan has no CX law.
    """
    from nwqlib.algorithms.qhd.native import construct_qhd

    x, y = sp.symbols("x y")
    problem = Optimization(objective=sp.Integer(0), variables=(x, y), bounds=((0.0, 4.0), (0.0, 4.0)))
    options = QHD(num_grid_points=3, num_steps=1, total_time=1.0, schedule=QuadraticSchedule(gamma=0.0),
                  trotter_order=1)
    # Points 1, 2, 3. exp(-(2 - 1)**2/(2 * 0.02**2)) = exp(-1250) underflows.
    tail = GaussianState(center=(1.0, 2.0), widths=(0.02, 10.0))
    selections = {}
    for name, method in (
        ("none", options.revise(initial_state_preparation="none")),
        ("structured", options),
        ("tail", options.revise(initial_state=tail)),
        ("qiskit_state_preparation", options.revise(initial_state_preparation="qiskit_state_preparation")),
    ):
        selected = make_plan(problem, method)
        selections[name] = (selected, selected.construction.selections[0])
    resource_only, record = selections["none"]
    assert [(b.kind, b.support) for group in resource_only.reconstruction.steps for b in group] == [
        ("kinetic", (0, 1)), ("kinetic", (1, 2)), ("kinetic", (3, 4)), ("kinetic", (4, 5))
    ]
    assert record.construction_work == 28
    # Two CX and two Ry(-angle) rotations per fused hopping block (resources.rotation_population).
    assert [(law.metric, law.value) for law in record.resource_laws] == [("cx", 8), ("arbitrary_rotations", 8)]
    assert record.blocker == "initial_state_preparation='none' is resource-only"
    assert resource_only.experiments == ()
    with pytest.raises(ValueError, match="resource-only"):
        solve(resource_only)
    with pytest.raises(ValueError, match="resource-only"):
        construct_qhd(resource_only.blocks[0], (), None)
    with pytest.raises(ValueError, match="resource-only"):
        make_plan(problem, options.revise(initial_state_preparation="none"), execution="classical")

    for name, emitted, cx in (("structured", 10, 12), ("tail", 6, 6)):
        selected, chain_record = selections[name]
        gates = construct_qhd(selected.blocks[0], (), None).count_ops()
        preparation = {gate: gates.get(gate, 0) for gate in ("x", "cry", "cx")}
        assert chain_record.construction_work - record.construction_work == sum(preparation.values()) == emitted
        # A controlled RY lowers to two CX.
        chain_cx = 2 * preparation["cry"] + preparation["cx"]
        assert chain_record.resource_laws[0].value - record.resource_laws[0].value == chain_cx == cx
    _, sdk_record = selections["qiskit_state_preparation"]
    assert sdk_record.construction_work - record.construction_work == 2 * 3 * 2**3
    assert sdk_record.resource_laws == ()


@pytest.mark.parametrize("flavor", ["schrodinger", "ir_product"])
def test_classical_mass_window_follows_the_expm_multiply_generators(flavor):
    """K = 39 classical QHD admits its computed masses within the recorded window.

    With h = 1/20 one step's generator has a shifted 1-norm near 200, and the
    Schrodinger mass exceeds one by about 3e-12, although exact evolution is
    unitary. The fixed 1e-12 host floor rejects these correct states. The
    recorded window must cover the observed mass, each Schrodinger generator
    bound must cover the shifted 1-norm of its step generator (the direct
    ir_product kernel makes no expm_multiply call and records none), and a
    mass beyond the window must be rejected.
    """
    import scipy.sparse
    from nwqlib.algorithms.qhd.split_step import potential_diagonal
    from nwqlib.algorithms.qhd.theory import restricted_kinetic_sparse
    from nwqlib.core.records import Float64
    from nwqlib.execution import ScalarValue

    x = sp.Symbol("x", real=True)
    problem = Optimization(
        objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),)
    )
    method = QHD(num_grid_points=39, num_steps=2, total_time=1.0, theory_flavor=flavor)
    selected = make_plan(problem, method, execution="classical")
    result = solve(selected)
    (application,) = result.applications
    window = {b.parameter: b.value for b in application.arguments}["probability_window"]
    assert window == Float64(value=owner._host_probability_window(selected))
    assert abs(result.observed_mass - 1.0) <= window.value
    if flavor == "schrodinger":
        assert abs(result.observed_mass - 1.0) > 1e-12

    def shifted_one_norm(generator):
        generator = scipy.sparse.csr_matrix(generator)
        shift = generator.diagonal().sum() / generator.shape[0]
        shifted = generator - shift * scipy.sparse.identity(generator.shape[0], format="csr")
        return float(abs(shifted).sum(axis=0).max())

    grid = owner._grid(selected)
    if flavor == "schrodinger":
        kinetic = restricted_kinetic_sparse(grid)
        potential = scipy.sparse.diags(potential_diagonal(selected.reconstruction, grid))
        generators = [0.5 * (a * kinetic + v * potential)
                      for _t, a, v in selected.reconstruction.step_weights]
    else:
        # The direct one-hot kernel makes no expm_multiply call and records no generator bound.
        generators = []
    r = selected.reconstruction
    bounds = tuple(owner._generator_norms(flavor, selected.method, grid, r.support_values, r.steps, r.step_weights))
    assert len(bounds) == len(generators)
    for (bound, rows), generator in zip(bounds, generators, strict=True):
        assert shifted_one_norm(generator) <= bound * (1 + 1e-12)
        assert np.max(np.diff(scipy.sparse.csr_matrix(generator).indptr)) <= rows

    values = result.data.observations.chunks[0].values
    beyond = tuple(
        ScalarValue(label=v.label, value=1.0 + 2.0 * window.value, frame=v.frame)
        if v.label == "observed_mass" else v
        for v in values
    )
    with pytest.raises(ValueError, match="roundoff window"):
        owner._host_summary(selected, beyond)
    # The two-step K = 3 Schrodinger schedule of make_plan stays at the floor.
    small = make_plan(execution="classical")
    assert owner._host_probability_window(small) == NUMERICAL_RELATION_RTOL


def test_objective_far_from_the_origin_keeps_its_grid_values():
    """(x - c)**2 and (x - y)**2 on width-2 boxes near 1e7 keep their grid values.

    Expanded about the origin, (x - c)**2 becomes the support table
    x**2 - 2 c x plus the constant c**2, both near 1e14, so their binary64
    sum keeps only about one ulp of 1e14, 2**-6 ~= 0.016. Tabulated that
    way, the one-variable case selects a grid point whose exact objective
    is 0.0102 instead of the grid minimum 0.0098, and reports 0.0 there.
    Expanded about the box point nearest the origin, every term is at most
    the squared box width, about 4, so a few roundings leave an error near
    1e-15. The allowance 1e-13 is 225 u times that scale, u = 2**-53. The
    reference is exact rational arithmetic at the binary64 grid coordinates.
    """
    from fractions import Fraction

    x, y = sp.symbols("x y", real=True)
    c = 10**7
    box = (c - 0.899, c + 1.101)
    single = make_plan(
        Optimization(objective=(x - c) ** 2, variables=(x,), bounds=(box,)),
        QHD(num_grid_points=9, num_steps=2, total_time=1.0),
        execution="classical",
    )
    grid = owner._grid(single)
    exact = [(Fraction(grid.grid_value(0, i)) - c) ** 2 for i in range(9)]
    stored = [owner.objective_at(single.reconstruction, (i,), 9) for i in range(9)]
    assert max(abs(Fraction(value) - reference) for value, reference in zip(stored, exact)) <= 1e-13
    result = solve(single)
    best = min(range(9), key=lambda i: (exact[i], i))
    assert result.candidate_indices == (best,)
    assert result.objective == pytest.approx(float(exact[best]), rel=0, abs=1e-13)

    # With two variables, the cross term -2 x y also needs the centered expansion.
    pair = make_plan(
        Optimization(objective=(x - y) ** 2, variables=(x, y), bounds=(box, box)),
        QHD(num_grid_points=3, num_steps=1, total_time=1.0),
        execution="classical",
    )
    grid = owner._grid(pair)
    for i in range(3):
        for j in range(3):
            reference = (Fraction(grid.grid_value(0, i)) - Fraction(grid.grid_value(1, j))) ** 2
            value = owner.objective_at(pair.reconstruction, (i, j), 3)
            assert abs(Fraction(value) - reference) <= 1e-13


@pytest.mark.parametrize("flavor", ["schrodinger", "split_step"])
def test_classical_probabilities_do_not_depend_on_an_additive_constant(flavor):
    """An objective ``c + g(x)`` has the probabilities of ``g(x)`` for any c, bitwise.

    The constant adds ``b(t) c I`` to the Hamiltonian, which commutes with
    every operator, so it changes the final state by a global phase only
    (``method._constant_phase``). Near ``c = 1e16`` adjacent binary64 numbers
    are 2 apart, so a potential diagonal that held ``c + g`` would round the
    support-table values of g, which span 3.2 on this grid, to two values and
    change the dynamics. The Schrodinger and split-step flavors evolve the
    support tables alone and read the probabilities before the constant's
    phase is restored, and ``c + g`` and g share their tables, so the probabilities
    must be equal bitwise.
    They are far from uniform, so the equality compares a real evolution.
    The readout objective keeps the constant (``method.objective_at``).

    A kept state carries the phase ``phi = -c sum_k dt b_k``, which planning
    admits only when it is finite and its allowance
    ``(3u + 3u**2 + u**3) |c| sum_k |dt b_k|``, evaluated upward, is below pi
    (``method._admit_constant_phase``). With gamma = 0 and two unit steps,
    ``sum_k dt b_k = 2``. For ``c = 1e308`` phi overflows, so a kept state is
    rejected at planning while the probabilities equal those of g bitwise.
    For ``c = 1e17`` phi is finite but the allowance is 67 rad, and for
    ``c = 1e15`` it is 0.67 rad and the kept state is finite. At the boundary,
    with ``c = 2414255523835869.5``, seven steps of the stored
    ``dt = 0.5581322458313378`` and every weight 1, the exact first-order
    value ``3u |c| sum_k |dt b_k|`` from the stored inputs exceeds pi by about
    5e-17 while round-to-nearest evaluation gives 3.1415926535897927, below
    ``math.pi``. The test checks the exact value against a 40-digit pi and
    the rounded one against ``math.pi``. The upward allowance rejects that
    kept state, and ``c = 1e15`` with the same steps is admitted.
    """
    import math
    from fractions import Fraction

    import mpmath

    x = sp.Symbol("x", real=True)
    g = 4 * (x - sp.Rational(3, 10)) ** 2
    method = QHD(num_grid_points=4, num_steps=4, total_time=2.0, theory_flavor=flavor)

    def problem(constant):
        return Optimization(objective=g + constant, variables=(x,), bounds=((-1.0, 1.0),))

    results = [solve(plan(problem(c), method=method, execution="classical", seed=7))
               for c in (0, sp.Integer(10) ** 16)]
    base, shifted = (result.marginals.array[0].tolist() for result in results)
    assert shifted == base
    assert max(base) - min(base) > 0.1
    assert results[1].objective == owner.objective_at(results[1].plan.reconstruction,
                                                      results[1].candidate_indices, 4) >= 1e16

    edge = QHD(num_grid_points=4, num_steps=2, total_time=2.0, schedule=QuadraticSchedule(gamma=0.0),
               theory_flavor=flavor)
    base = solve(plan(problem(0), method=edge, execution="classical", seed=7)).marginals
    huge = solve(plan(problem(sp.Integer(10) ** 308), method=edge, execution="classical", seed=7))
    assert huge.marginals == base
    kept = edge.revise(keep_state=True)
    with pytest.raises(ValueError, match="exceeds the binary64 range"):
        plan(problem(sp.Integer(10) ** 308), method=kept, execution="classical", seed=7)
    with pytest.raises(ValueError, match="reaches pi"):
        plan(problem(sp.Integer(10) ** 17), method=kept, execution="classical", seed=7)
    result = solve(plan(problem(sp.Integer(10) ** 15), method=kept, execution="classical", seed=7))
    assert np.isfinite(result.data.artifact(result.artifact).array).all()
    assert result.marginals == base

    boundary = QHD(num_grid_points=3, num_steps=7, total_time=3.906925720819365,
                   schedule=QuadraticSchedule(gamma=0.0), theory_flavor=flavor, keep_state=True)

    def square(constant):
        return Optimization(objective=x**2 + constant, variables=(x,), bounds=((-1.0, 1.0),))

    edge_c = sp.Rational("2414255523835869.5")
    r = plan(square(edge_c), method=boundary.revise(keep_state=False), execution="classical",
             seed=7).reconstruction
    dt = boundary.total_time / boundary.num_steps
    assert r.constant == 2414255523835869.5 and {b for _t, _a, b in r.step_weights} == {1.0}
    nearest = 3 * 2.0**-53 * abs(r.constant) * math.fsum(abs(dt * b) for _t, _a, b in r.step_weights)
    first_order = 3 * Fraction(1, 2**53) * Fraction(r.constant) * sum(abs(Fraction(dt) * Fraction(b))
                                                                    for _t, _a, b in r.step_weights)
    assert nearest < math.pi
    with mpmath.workdps(40):
        assert mpmath.mpf(first_order.numerator) / first_order.denominator > mpmath.pi
    with pytest.raises(ValueError, match="reaches pi"):
        plan(square(edge_c), method=boundary, execution="classical", seed=7)
    legal = plan(square(sp.Integer(10) ** 15), method=boundary, execution="classical", seed=7)
    assert legal.reconstruction.constant == 1e15


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_circuit_route_rejects_a_physical_phase_beyond_the_binary64_range(execution):
    """The compiled identity phase must be finite on the native and ir_product routes, for every readout.

    The compiler adds ``-dt b_k c`` per step to the physical phase, which the
    native circuit adds to its global phase and the ir_product reference
    multiplies into its state before the probabilities are read. For
    ``c = 1e308`` over two unit steps the sum overflows, and planning names
    that cause instead of failing on the record's finiteness check.
    """
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x**2 + sp.Integer(10) ** 308, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=4, num_steps=2, total_time=2.0, schedule=QuadraticSchedule(gamma=0.0),
                 theory_flavor="ir_product")
    with pytest.raises(ValueError, match="compiled physical phase .* objective_constant part"):
        plan(problem, method=method, execution=execution, seed=7)


def _exact_native_components(problem, method):
    """Return the four additive native allowance components in rational arithmetic, the Plan and the rejection.

    The components follow ``method._native_components`` for one variable,
    ``L_s = [eta_s + beta_M (1 + eta_s)] S_s`` for the constant, projector and
    kinetic sources, the projector blocks' ``2u Y + 20u B`` and the final
    assignment ``2u |physical_phase| + 20u``, from the stored inputs of the
    Plan without ``keep_state``. The rejection is that of the same Plan with
    ``keep_state``, and its reported largest component is returned too.
    """
    import re
    from fractions import Fraction

    probabilities = plan(problem, method=method, execution="quantum", seed=7)
    r, grid = probabilities.reconstruction, owner._grid(probabilities)
    blocks = [b for step in r.steps for b in step if b.kind == "number_projector"]
    rows = sum(len(t.values) for t in r.support_values)
    contributions = method.num_steps * (1 + (r.constant != 0) + 2 * rows)
    u = Fraction(1, 2**53)

    def gamma(n):
        return n * u / (1 - n * u)

    dt = Fraction(method.total_time / method.num_steps)
    potential_time = sum(abs(dt * Fraction(b)) for _t, _a, b in r.step_weights)
    s_c = abs(Fraction(r.constant)) * potential_time
    s_p = potential_time * sum(sum(abs(Fraction(v)) for v in t.values.array.tolist()) / 2 ** len(t.support)
                               for t in r.support_values)
    s_k = sum(abs(dt * Fraction(a)) for _t, a, _b in r.step_weights) / Fraction(grid.spacing(0)) ** 2
    eta_2 = 2 * u + u**2
    eta_k = (3 * u + u**2 + (1 + 2 * u + u**2) * u) / (1 - u)
    beta_m = u + (1 + u) * gamma(contributions - 1) ** 2
    scaled = sum(abs(Fraction(b.angle)) / 2 ** len(b.support) for b in blocks)
    components = {
        "constant": (eta_2 + beta_m * (1 + eta_2)) * s_c,
        "kinetic": (eta_k + beta_m * (1 + eta_k)) * s_k,
        "projector": (eta_2 + beta_m * (1 + eta_2)) * s_p + 2 * u * scaled + 20 * u * len(blocks),
        "final": 2 * u * abs(Fraction(r.physical_phase)) + 20 * u,
    }
    with pytest.raises(ValueError, match="cannot establish the requested physical phase") as rejected:
        plan(problem, method=method.revise(keep_state=True), execution="quantum", seed=7)
    message = str(rejected.value)
    reported = Fraction(float(re.search(r"largest component, (\S+) rad", message).group(1)))
    return components, r, message, reported


def _check_native_rejection_components(method):
    """The native witnesses of ``test_circuit_route_kept_state_needs_a_phase_allowance_below_pi``."""
    x = sp.Symbol("x", real=True)
    witness = Optimization(objective=-sp.Integer(10) ** 16 * x**2 + 3 * sp.Integer(10) ** 15, variables=(x,),
                           bounds=((-1.0, 1.0),))
    components, r, message, reported = _exact_native_components(witness, method)
    blocks = sum(b.kind == "number_projector" for step in r.steps for b in step)
    assert (blocks, r.physical_phase) == (8, -1000000000000008.0)
    assert max(components, key=components.get) == "projector"
    assert components["projector"] > components["constant"] + components["final"]
    assert ("comes from the projector-identity contributions with the projector blocks' own phase "
            "assignments") in message
    assert "removing it from the objective" not in message
    assert components["projector"] <= reported < components["projector"] + components["final"]

    wide = Optimization(objective=sp.Integer(10) ** 32 * x**2 + sp.Integer(10) ** 16, variables=(x,),
                        bounds=((-2e-8, 2e-8),))
    components, r, message, reported = _exact_native_components(wide, method.revise(total_time=0.5))
    assert max(components, key=components.get) == "final"
    assert "comes from the final assignment of the physical phase to the circuit" in message
    assert "removing it from the objective" not in message
    assert components["final"] <= reported < components["final"] + components["constant"]


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_circuit_route_kept_state_needs_a_phase_allowance_below_pi(execution):
    """A kept native or ir_product state needs the compiled phase within an allowance below pi.

    ``method._admit_compiled_phase`` applies ``method._compiled_phase_allowance``,
    the formation and compensated-accumulation bound of the compiler's phase
    ledger, and on the native route ``method._native_phase_allowance`` for
    the circuit's phase assignments. With ``c = 1e17`` and two unit steps the
    constant contributes ``S_C = 2e17``, and the ledger of its 16
    contributions gets 66.6 rad (111 rad on the native route), so a kept
    state is rejected while the same Plan without ``keep_state`` solves. The
    allowance in the rejection message must equal the owners' value for the
    16 contributions that planning records, so the test also pins the
    contribution count and the route's rule, and the message must name the
    constant's component and advise removing the constant. With ``c = 3/8``
    the allowance is far below pi and the kept state is admitted. The
    objective ``1e17 x + 1e10`` has projector contributions whose signed
    total cancels but whose absolute sum drives the allowance, so its
    rejection names the projector component and gives no advice about the
    constant.

    Natively, ``-1e16 x**2 + 3e15`` over two unit steps has M = 16
    contributions, B = 8 projector blocks and ``physical_phase = -1e15 - 8``,
    and its allowance of about 5.0 rad splits into the additive components
    ``L_C``, ``L_K``, ``L_P + 2u Y + 20u B`` and the final assignment
    ``2u |physical_phase| + 20u`` (``method._native_components``), with
    ``L_s = [eta_s + beta_M (1 + eta_s)] S_s``. The test evaluates them in
    rational arithmetic from the Plan's stored inputs. The projector
    component, about 2.78 rad, is the largest, ahead of the constant's 2.00
    rad, so the message must name it, report an upper value of it below the
    projector and final components together, and give no advice about the
    constant. For ``1e32 x**2 + 1e16`` on (-2e-8, 2e-8) with total time 0.5
    the final assignment of the physical phase, about 3.33 rad, is the
    largest component, and the message must name it rather than a
    Hamiltonian source.
    """
    import re

    x = sp.Symbol("x", real=True)
    method = QHD(num_grid_points=3, num_steps=2, total_time=2.0, schedule=QuadraticSchedule(gamma=0.0),
                 theory_flavor="ir_product")
    route = "native circuit" if execution == "quantum" else "ir_product reference"

    def problem(constant):
        return Optimization(objective=x**2 + constant, variables=(x,), bounds=((-1.0, 1.0),))

    with pytest.raises(ValueError, match=f"of the {route}, whose rounding allowance") as rejected:
        plan(problem(sp.Integer(10) ** 17), method=method.revise(keep_state=True), execution=execution, seed=7)
    assert "comes from the objective constant's contributions" in str(rejected.value)
    assert "removing it from the objective" in str(rejected.value)
    probabilities = plan(problem(sp.Integer(10) ** 17), method=method, execution=execution, seed=7)
    grid, r = owner._grid(probabilities), probabilities.reconstruction
    contributions = sum(len(t.values) for t in r.support_values) * 2 * 2 + 2 * 2
    assert contributions == 16
    expected = owner._compiled_phase_allowance(r, probabilities.method, grid, contributions)
    if execution == "quantum":
        expected = owner._native_phase_allowance(r, expected)
    reported = float(re.search(r"rounding allowance (\S+) rad", str(rejected.value)).group(1))
    assert reported == expected > 66.6
    assert solve(probabilities).valid_probability == pytest.approx(1.0, abs=1e-9)
    kept = solve(plan(problem(sp.Rational(3, 8)), method=method.revise(keep_state=True), execution=execution,
                      seed=7))
    assert np.isfinite(kept.data.artifact(kept.artifact).array).all()
    # A steep table cancels in its signed total, but its absolute sum drives the
    # allowance, so the message names the projector component and gives no advice
    # about the small constant.
    steep = Optimization(objective=sp.Integer(10) ** 17 * x + sp.Integer(10) ** 10, variables=(x,),
                         bounds=((-1.0, 1.0),))
    with pytest.raises(ValueError, match="comes from the projector-identity contributions") as rejected:
        plan(steep, method=method.revise(keep_state=True), execution=execution, seed=7)
    assert "removing it from the objective" not in str(rejected.value)
    if execution == "quantum":
        _check_native_rejection_components(method)


def test_compiled_phase_allowance_bounds_the_ledger_error():
    """The ledger allowance covers the error of the stored physical phase against its exact value.

    The exact identity angle of the discrete model reads the stored binary64
    dt, step weights, spacing, table values and constant exactly,
    ``-sum_k dt (a_k/h**2 + b_k c + b_k sum_r v_r/2)`` for one variable, and
    is evaluated in rational arithmetic. With ``c = 1e16`` and 16 steps the
    formation and the compensated sum of the ledger round, and the distance
    of the stored phase from the exact angle must not exceed
    ``method._compiled_phase_allowance``.
    """
    from fractions import Fraction

    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x**2 / 3 + x / 7 + sp.Integer(10) ** 16, variables=(x,),
                           bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=4, num_steps=16, total_time=1.0, schedule=QuadraticSchedule(gamma=0.3),
                 theory_flavor="ir_product")
    selected = plan(problem, method=method, execution="classical", seed=7)
    r, grid = selected.reconstruction, owner._grid(selected)
    dt = Fraction(method.total_time / method.num_steps)
    inverse_square = 1 / Fraction(grid.spacing(0)) ** 2
    tables = sum(Fraction(v) / 2 ** len(t.support) for t in r.support_values for v in t.values.array.tolist())
    exact = -sum(dt * (Fraction(a) * inverse_square + Fraction(b) * Fraction(r.constant) + Fraction(b) * tables)
                 for _t, a, b in r.step_weights)
    contributions = 16 * (2 + 2 * sum(len(t.values) for t in r.support_values))
    error = abs(Fraction(r.physical_phase) - exact)
    assert error <= Fraction(owner._compiled_phase_allowance(r, method, grid, contributions))


def test_compensated_ledger_sum_meets_its_bound_where_a_plain_sum_does_not(monkeypatch):
    """The ledger's Neumaier sums lie within their bounds of the exact sums of the recorded contributions.

    The test records every contribution ``x_j = -fl(t_j q_j)`` that
    ``compiler.QHDCompiler._record_global_phase`` receives while planning a
    c = 1e16, 16-step Plan and sums them exactly. The stored total must lie
    within ``beta_M A`` of that sum, ``beta_M = u + (1 + u) gamma_(M-1)**2``
    with ``A = sum_j |x_j|`` (Neumaier's bound in the recorder's docstring),
    and the constant's own total within ``[eta_2 + (1 + eta_2) beta_N] S_C``
    of its exact angle ``-sum_k dt b_k c`` over the stored inputs. A plain
    running sum of the same contributions misses the exact total by more
    than ``beta_M A``, so the bound separates the two accumulations.
    """
    from fractions import Fraction

    from nwqlib.algorithms.qhd.compiler import QHDCompiler

    recorded = []
    original = QHDCompiler._record_global_phase

    def recording(self, *, coefficient, evolution_time, time, source, support=None):
        recorded.append(-evolution_time * coefficient)
        return original(self, coefficient=coefficient, evolution_time=evolution_time, time=time, source=source,
                        support=support)

    monkeypatch.setattr(QHDCompiler, "_record_global_phase", recording)
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x**2 / 3 + x / 7 + sp.Integer(10) ** 16, variables=(x,),
                           bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=4, num_steps=16, total_time=1.0, schedule=QuadraticSchedule(gamma=0.3),
                 theory_flavor="ir_product")
    r = plan(problem, method=method, execution="classical", seed=7).reconstruction
    u, m, n = Fraction(1, 2**53), len(recorded), method.num_steps
    # The ledger's count M, which the allowances take from the reconstruction, is the number of recorded events.
    assert owner._phase_contributions(r, method) == m

    def gamma(k):
        return k * u / (1 - k * u)

    def beta(k):
        return u + (1 + u) * gamma(k - 1) ** 2

    exact = sum(Fraction(v) for v in recorded)
    absolute = sum(abs(Fraction(v)) for v in recorded)
    assert abs(Fraction(r.physical_phase) - exact) <= beta(m) * absolute
    plain = 0.0
    for value in recorded:
        plain += value
    assert abs(Fraction(plain) - exact) > beta(m) * absolute
    dt = Fraction(method.total_time / method.num_steps)
    s_c = abs(Fraction(r.constant)) * sum(abs(dt * Fraction(b)) for _t, _a, b in r.step_weights)
    constant = -sum(dt * Fraction(b) * Fraction(r.constant) for _t, _a, b in r.step_weights)
    eta_2 = 2 * u + u**2
    stored = Fraction(dict(r.phase_sources)["objective_constant"])
    assert abs(stored - constant) <= (eta_2 + (1 + eta_2) * beta(n)) * s_c


def test_phase_ledger_compensation_recovers_what_a_plain_sum_loses():
    """Each angle total of the phase ledger uses Neumaier's branch on the larger magnitude.

    The contributions 1, 1e100, 1 and -1e100 sum to 2. A plain running sum
    gives 0, the compensated sum without the branch on ``|s| >= |x|``
    (Kahan-Babuska) gives 1, and swapped branches give 0, while Neumaier's
    residuals recover both units. The recorder must return 2 for the total
    and for the source total of the same contributions.
    """
    from types import SimpleNamespace

    from nwqlib.algorithms.qhd.compiler import QHDCompiler

    ledger = SimpleNamespace(_phase_sums={}, _metadata={"dropped_global_phase": {
        "phase_angle": 0.0, "per_source_total_angles": {},
        "per_source_coefficient_totals": {}}})
    ledger._compensated_add = lambda name, value: QHDCompiler._compensated_add(ledger, name, value)
    for angle in (1.0, 1e100, 1.0, -1e100):
        QHDCompiler._record_global_phase(ledger, coefficient=-angle, evolution_time=1.0, time=0.0,
                                         source="projector_identity")
    phases = ledger._metadata["dropped_global_phase"]
    assert phases["phase_angle"] == phases["per_source_total_angles"]["projector_identity"] == 2.0


def test_compiled_phase_allowances_are_evaluated_upward():
    """The ledger and native allowances are upper values of their exact expressions.

    The test evaluates the expressions of ``method._compiled_phase_allowance``
    and ``method._native_phase_allowance`` in rational arithmetic from the
    Plan's stored binary64 inputs, ``gamma_r = r u/(1 - r u)``,
    ``eta_2 = 2u + u**2``, ``beta_d = u + (1 + u) gamma_(d-1)**2``,
    ``eta_K = (3u + u**2 + (1 + 2u + u**2) beta_d)/(1 - u)``,
    ``E_ledger = D + beta_M (S_C + S_P + S_K + D)`` with
    ``beta_M = u + (1 + u) gamma_(M-1)**2``, and
    ``E_native = E_ledger + 2u (Y + |physical_phase|) + 20u (B + 1)``, and
    requires each binary64 allowance to be at least its exact value. Two
    variables make ``gamma_(d-1)`` nonzero, and the y spacing 0.75 is not a
    power of two. On this Plan round-to-nearest evaluation happens to stay
    above the exact values, so the helper assertions carry the
    discrimination. The upward helpers must bound single operations that
    round-to-nearest rounds down, keep exact zeros, lift an underflowed
    product to 2**-1074 and give infinity when a gamma denominator has no
    positive lower bound or a product overflows.
    """
    from fractions import Fraction

    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=x**2 / 3 + x * y / 7 + y / 5 + sp.Rational(3, 8), variables=(x, y),
                           bounds=((-1.0, 1.0), (0.0, 3.0)))
    method = QHD(num_grid_points=3, num_steps=3, total_time=0.7, schedule=QuadraticSchedule(gamma=0.3),
                 theory_flavor="ir_product")
    selected = plan(problem, method=method, execution="classical", seed=7)
    r, grid = selected.reconstruction, owner._grid(selected)
    u, d, steps = Fraction(1, 2**53), 2, method.num_steps

    def gamma(n):
        return n * u / (1 - n * u)

    dt = Fraction(method.total_time / method.num_steps)
    potential_time = sum(abs(dt * Fraction(b)) for _t, _a, b in r.step_weights)
    kinetic_time = sum(abs(dt * Fraction(a)) for _t, a, _b in r.step_weights)
    s_c = abs(Fraction(r.constant)) * potential_time
    s_p = potential_time * sum(sum(abs(Fraction(v)) for v in t.values.array.tolist()) / 2 ** len(t.support)
                               for t in r.support_values)
    s_k = kinetic_time * sum(1 / Fraction(grid.spacing(j)) ** 2 for j in range(d))
    beta = u + (1 + u) * gamma(d - 1) ** 2
    eta_k = (3 * u + u**2 + (1 + 2 * u + u**2) * beta) / (1 - u)
    formation = (2 * u + u**2) * (s_c + s_p) + eta_k * s_k
    contributions = steps * (2 + 2 * sum(len(t.values) for t in r.support_values))
    beta_m = u + (1 + u) * gamma(contributions - 1) ** 2
    exact_ledger = formation + beta_m * (s_c + s_p + s_k + formation)
    projectors = [b for step in r.steps for b in step if b.kind == "number_projector"]
    scaled = sum(abs(Fraction(b.angle)) / 2 ** len(b.support) for b in projectors)
    exact_native = exact_ledger + 2 * u * (scaled + abs(Fraction(r.physical_phase))) + 20 * u * (len(projectors) + 1)
    ledger = owner._compiled_phase_allowance(r, method, grid, contributions)
    assert Fraction(ledger) >= exact_ledger > 0
    assert Fraction(owner._native_phase_allowance(r, ledger)) >= exact_native
    # Each upward operation bounds a case that round-to-nearest rounds down,
    # including the running sum of 1 and 2**15 copies of 2**-54.
    tiny, above_one = 2.0**-54, 1.0 + 2.0**-52
    assert Fraction(owner._add_up(1.0, tiny)) >= 1 + Fraction(tiny) > Fraction(1.0 + tiny)
    assert Fraction(owner._mul_up(above_one, above_one)) >= Fraction(above_one) ** 2 > Fraction(above_one * above_one)
    assert Fraction(owner._div_up(1.0, 3.0)) >= Fraction(1, 3) > Fraction(1.0 / 3.0)
    assert Fraction(owner._sum_up([1.0] + [tiny] * 2**15)) >= 1 + Fraction(1, 2**39)
    assert Fraction(owner._gamma_up(3)) >= gamma(3)
    assert (owner._add_up(0.0, 0.0), owner._mul_up(0.0, 5.0), owner._gamma_up(0)) == (0.0, 0.0, 0.0)
    assert owner._mul_up(2.0**-600, 2.0**-600) == 2.0**-1074
    assert owner._gamma_up(2**53) == owner._gamma_up(2**53 - 1) == owner._mul_up(1e200, 1e200) == float("inf")


def test_compiler_forms_the_constant_coefficient_from_the_stored_binary64_constant():
    """Each step's constant coefficient is ``fl(b_k c)`` with the constant the reconstruction stores.

    The 80-digit constant 9007199254740993 rounds to the stored binary64
    9007199254740992. With one unit step and b = 1.5 the coefficient is
    ``fl(1.5 * 9007199254740992) = 13510798882111488``, where multiplying the
    unrounded constant first gives 13510798882111490.
    """
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x**2 + sp.Float(9007199254740993, 80), variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=3, num_steps=1, total_time=1.0, schedule=QuadraticSchedule(gamma=2.0),
                 theory_flavor="ir_product")
    r = plan(problem, method=method, execution="classical", seed=7).reconstruction
    assert r.constant == 9007199254740992.0
    assert r.step_weights[0][2] == 1.5
    assert dict(r.phase_sources)["objective_constant"] == -13510798882111488.0


def test_objective_expansion_is_plain_when_the_box_contains_the_origin():
    """A box that contains the origin expands about zero, the plain monomial expansion."""
    from nwqlib.algorithms.qhd.objective import ObjectiveDecomposer, expansion_centers

    x, y = sp.symbols("x y")
    objective = x * y + 2 * x - y + sp.Rational(1, 5)
    bounds = ((-1.0, 2.0), (0.0, 6.0))
    assert expansion_centers(bounds) == (0.0, 0.0)
    decomposer = ObjectiveDecomposer(objective, (x, y), expansion_centers(bounds))
    assert decomposer.objective == sp.expand(objective)
    assert expansion_centers(((1.0, 3.0), (-4.0, -2.0))) == (1.0, -2.0)


@pytest.mark.parametrize("execution,schedule,rule", [
    ("quantum", QuadraticSchedule(), "midpoint"),
    ("classical", QuadraticSchedule(gamma=0.3), "integrated"),
    ("classical", CubicSchedule(s=0.5), "midpoint"),
    ("quantum", CubicSchedule(s=0.5), "integrated"),
    ("classical", ShiftedCubicSchedule(s=0.5), "integrated"),
    ("quantum", ShiftedCubicSchedule(s=0.5), "integrated"),
])
def test_archive_keeps_support_tables_and_schedule(tmp_path, monkeypatch, execution, schedule, rule):
    """A saved Plan restores its schedule kind, parameter, rule, step weights and initial state without recomputing them."""
    from nwqlib.algorithms.qhd import compiler

    initial = GaussianState(center=(0.1,), widths=(0.7,))
    method = QHD(num_grid_points=3, num_steps=2, total_time=0.17, schedule=schedule, coefficient_rule=rule,
                 initial_state=initial)
    selected = make_plan(method=method, execution=execution)
    files = ArchiveFiles(tmp_path, 4_000_000)
    saved = save_plan(selected, files)

    def forbid(*args, **kwargs):
        raise AssertionError("load replanned or recomputed a table, schedule or evolution")

    monkeypatch.setattr(QHD, "plan", forbid)
    monkeypatch.setattr(QHDCompiler, "_potential_grid_values", forbid)
    monkeypatch.setattr(QHDCompiler, "build_step_pauli_ir", forbid)
    monkeypatch.setattr(compiler, "step_weights", forbid)
    for name in ("kinetic_weight", "potential_weight", "kinetic_integral", "potential_integral"):
        monkeypatch.setattr(type(schedule), name, forbid)
    monkeypatch.setattr(owner, "_evolve_restricted", forbid)
    monkeypatch.setattr(QHD, "_select_native", forbid)
    restored = load_plan(saved, ArchiveFiles(tmp_path, 4_000_000))
    assert restored == selected and restored.content_id == selected.content_id
    assert type(restored.method.schedule) is type(schedule) and restored.method.schedule == schedule
    assert restored.method.coefficient_rule == rule
    assert type(restored.method.initial_state) is GaussianState and restored.method.initial_state == initial
    assert restored.reconstruction.step_weights == selected.reconstruction.step_weights
    (table,) = restored.reconstruction.support_values
    assert (table.minimum, table.maximum, table.magnitude) == table_extrema(table.values.array)
    assert restored.problem == selected.problem
    assert tuple(b.record for b in restored.blocks) == tuple(b.record for b in selected.blocks)


@pytest.mark.parametrize("value", [sp.I * sp.Rational(1, 10**15), sp.I * sp.Symbol("x") / 10**15, sp.oo, sp.nan])
def test_objective_domain_precedes_pruning(value):
    x = sp.Symbol("x")
    problem = Optimization(objective=x + value, variables=(x,), bounds=((0.0, 1.0),))
    with pytest.raises(ValueError, match="complex|non-finite"):
        make_plan(problem, QHD(rotation_threshold=1e6))
    zero = make_plan(
        Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 1.0),)),
        QHD(rotation_threshold=0.0),
    )
    assert zero.reconstruction.constant == 0 and zero.reconstruction.support_values == ()


def test_objective_pole_has_owned_context_and_preserves_legal_grid(monkeypatch):
    x = sp.Symbol("x")
    problem = Optimization(objective=1/x, variables=(x,), bounds=((-1., 1.),))
    with monkeypatch.context() as guard:
        guard.setattr(QHDCompiler, "build_step_pauli_ir", Mock(side_effect=AssertionError("built before admission")))
        with pytest.raises(ValueError, match=r"QHD objective 1/x.*support \(0,\).*grid point \(0.0,\)") as caught:
            make_plan(problem, QHD(num_grid_points=3))
    assert "non-finite" in str(caught.value)
    selected = make_plan(problem, QHD(num_grid_points=2))
    np.testing.assert_allclose(selected.reconstruction.support_values[0].values, [-3., 3.],
                               rtol=0, atol=2e-15)


@pytest.mark.parametrize("failure", [MemoryError("allocation"), KeyboardInterrupt()])
def test_objective_evaluation_does_not_translate_system_failures(failure):
    from nwqlib.algorithms.qhd.potential import _evaluate_objective

    def fail(*point):
        raise failure
    with pytest.raises(type(failure)) as caught:
        _evaluate_objective(fail, (0.,), expression="1/x", support=(0,))
    assert caught.value is failure


@pytest.mark.parametrize("execution,flavor,comparisons,relations", [
    ("classical", "schrodinger", ("schrodinger_fidelity",), ("same-producer replay",)),
    ("classical", "ir_product", ("schrodinger_fidelity", "ir_product_fidelity"),
        ("cross-model comparison", "same-producer replay")),
    ("quantum", "schrodinger", ("schrodinger_fidelity", "ir_product_fidelity"),
        ("circuit-versus-restricted evolution", "circuit-versus-IR construction consistency")),
])
def test_explicit_fidelity_keeps_producer_scope_and_real_replay(execution, flavor, comparisons, relations, monkeypatch):
    from nwqlib.execution import VerificationReceipt

    selected = make_plan(method=QHD(total_time=.17, schedule=QuadraticSchedule(gamma=.3), theory_flavor=flavor,
                                    keep_state=True), execution=execution)
    result = solve(selected)
    stored = result.data.artifact(result.artifact).array
    calls = []
    evolve = owner._evolve_restricted
    def count(plan, reference_flavor):
        reference = evolve(plan, reference_flavor)
        assert not np.shares_memory(stored, reference)
        calls.append(reference_flavor)
        return reference
    monkeypatch.setattr(owner, "_evolve_restricted", count)
    receipt, _ = result.verify(checks=QHDVerification(comparisons=comparisons))
    assert calls == [c.removesuffix("_fidelity") for c in comparisons]
    facts = {f.fact.quantity: f for f in receipt.applications[0].facts}
    for comparison, relation in zip(comparisons, relations, strict=True):
        assert relation in facts[comparison + ".within_tolerance"].frame.conditioning
        assert relation in receipt.reference.reference
    arguments = {b.parameter: b.value for b in receipt.applications[0].arguments}
    assert arguments["schrodinger_evolutions"] == int("schrodinger_fidelity" in comparisons)
    assert arguments["ir_product_evolutions"] == int("ir_product_fidelity" in comparisons)
    monkeypatch.setattr(owner, "_evolve_restricted", Mock(side_effect=AssertionError("receipt loading replayed")))
    restored = VerificationReceipt.model_validate_json(receipt.model_dump_json())
    assert restored == receipt
    if execution == "classical" and flavor == "schrodinger":
        # A real replay result must be consumed, not merely computed and then
        # ignored in favor of comparing the stored array with itself.
        orthogonal = np.array([-stored[1].conjugate(), stored[0].conjugate()])
        monkeypatch.setattr(owner, "_evolve_restricted", lambda *a: orthogonal)
        changed, _ = result.verify(checks=QHDVerification(comparisons=comparisons))
        changed_facts = {f.fact.quantity: f.fact.value.value for f in changed.applications[0].facts}
        assert changed_facts["schrodinger_infidelity"] == pytest.approx(1., rel=0, abs=2e-14)
        assert changed_facts["schrodinger_fidelity.within_tolerance"] == 0.


def test_observed_candidate_partial_and_invalid_population_without_reference(monkeypatch):
    selected = make_plan()
    monkeypatch.setattr(
        owner, "_evolve_restricted", Mock(side_effect=AssertionError("unexpected reference"))
    )
    empty = owner._summarize(selected, np.zeros(0), 1.0, 1.0, 0.0, points=np.zeros((0, 1), dtype=np.int64))
    assert empty["candidate_coordinates"] is None and empty["value"] is None
    partial = owner._summarize(selected, np.array([0.2, 0.3]), 0.5, 1.0, 0.0, points=np.array([[0], [2]]))
    assert partial["valid_mass"] == 0.5 and partial["invalid_mass"] == 0.5
    assert partial["candidate_coordinates"] == (0.5,)
    # Coordinates 0 and 2 carry different mass: the sum cannot detect a swap.
    assert partial["marginals"].tolist() == [[0.2, 0.0, 0.3]]
    result = solve(selected)
    chunk = result.data.observations.chunks[0]
    from nwqlib.execution import ObservationView, RegisterMap

    wrong = chunk.revise(quantum_layout=(RegisterMap(name="qhd", bits=(2, 1, 0)),))
    with pytest.raises(ValueError, match="layout"):
        owner._admit_observations(selected, ObservationView(chunks=(wrong,)))


@pytest.mark.parametrize("offset", (0, -1))
def test_candidate_objective_association_survives_public_save_load(tmp_path, monkeypatch, offset):
    """An observed grid coordinate determines its objective, including legal negative values.

    The injected state puts all probability on index 0, so the candidate and
    the most probable point are both x = -1/2, and each coordinate and
    objective must stay bound to its indices through save and load.
    """
    from nwqlib.scientist import load_result

    x = sp.Symbol("x")
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2 + offset,
                           variables=(x,), bounds=((-1., 1.),))
    monkeypatch.setattr(owner, "_evolve_restricted", lambda *args: np.array([1., 0., 0.], complex))
    result = solve(make_plan(problem, execution="classical"))
    assert result.candidate == result.most_probable_coordinates == (-0.5,)
    # Direct scalar arithmetic at x=-1/2: (-1/2-1/5)^2 = 49/100.
    assert result.objective == pytest.approx(0.49 + offset, abs=1e-14)
    assert result.most_probable_objective == result.objective
    path = result.save(tmp_path / "result")
    monkeypatch.setattr(owner, "_evolve_restricted", Mock(side_effect=AssertionError("load evolved")))
    restored = load_result(path)
    assert restored.objective == result.objective
    assert (restored.most_probable_indices, restored.most_probable_objective) == ((0,), result.objective)
    for forged, message in (
        (dict(value=-2.), "candidate objective"),
        (dict(most_probable_objective=result.objective + 1.), "most-probable objective"),
        (dict(most_probable_coordinates=(0.,)), "most-probable coordinates"),
    ):
        with pytest.raises(ValueError, match=message):
            result.revise(**forged).validate_plan(result.plan)
    # The candidate minimizes the objective over the observed population that holds the most probable
    # point, so the record alone refuses a most-probable objective below the candidate's.
    with pytest.raises(ValueError, match="lies below the candidate's"):
        result.revise(most_probable_objective=result.objective - 1.0)


def test_an_unevaluated_objective_survives_public_save_load(tmp_path, monkeypatch):
    """``load_result`` reopens an objective built with ``evaluate=False`` as it was supplied.

    f = x + (x + 11/16)/(x + 11/16) keeps its quotient, 0/0 at x = -11/16 and 1 on the grid, which SymPy's
    evaluation would turn into x + 1. The loader compares the Plan’s stored ``srepr``
    with the objective that ``archive._SymbolicReader`` rebuilds.
    """
    import pickle
    from nwqlib.algorithms.qhd import archive
    from nwqlib.scientist import load_result

    x = sp.Symbol("x")
    quotient = sp.Mul(x + sp.Rational(11, 16), sp.Pow(x + sp.Rational(11, 16), -1, evaluate=False), evaluate=False)
    problem = Optimization(objective=sp.Add(x, quotient, evaluate=False), variables=(x,), bounds=((-1., 1.),))
    result = solve(make_plan(problem, execution="classical"))
    path = result.save(tmp_path / "result")
    with monkeypatch.context() as patch:
        patch.setattr(archive, "_SymbolicReader", pickle.Unpickler)
        with pytest.raises(ValueError, match="expressions differ"):
            load_result(path)
    restored = load_result(path)
    assert sp.srepr(restored.plan.problem.objective) == sp.srepr(problem.objective) != sp.srepr(x + 1)
    assert restored.content_id == result.content_id


@pytest.mark.parametrize("rows", [((0.25, 0.0, 1.0),), ((0.25, 1.0, 2.0**-1023),), ((0.0, 1.0, 1.0),),
                                  ((2.0**-1023, 1.0, 1.0),)],
                         ids=["zero_weight", "subnormal_weight", "zero_midpoint", "subnormal_midpoint"])
def test_reconstruction_admits_only_the_step_rows_that_planning_gives(rows):
    """A stored step row outside planning's range admission is refused, and rows at the normal floor stay legal.

    ``schedules.step_weights`` admits only positive normal weights and a
    positive normal first midpoint ``dt/2``, and binary64 rounding keeps
    the midpoints nondecreasing, but it can repeat one at step counts near
    2**52 (``records.QHDReconstruction.step_weights``). The record must
    refuse a zero or subnormal weight or first midpoint and keep two equal
    rows at the smallest normal number 2**-1022.
    """
    selected = make_plan(method=QHD(num_grid_points=2), execution="classical")
    with pytest.raises(ValueError, match="positive normal"):
        selected.reconstruction.revise(step_weights=rows)
    floor = 2.0**-1022
    kept = selected.reconstruction.revise(step_weights=((floor, floor, floor), (floor, floor, floor)))
    assert kept.step_weights[0] == kept.step_weights[1]


def test_explicit_grid_verification_uses_original_objective_without_evolution(monkeypatch):
    # The interior grid of [-1, 1] with K = 3 is x_i = (i - 1)/2. There (x - 1/5)**2 takes 49/100, 1/25 and
    # 9/100, and (x - 1/2)**2 takes 1, 1/4 and 0, whose minimum at i = 2 is not where the short evolution
    # from the kinetic ground state keeps most of its probability, i = 1.
    x = sp.Symbol("x")
    result = solve(make_plan())
    shifted = solve(make_plan(Optimization(objective=(x - sp.Rational(1, 2)) ** 2, variables=(x,),
                                           bounds=((-1.0, 1.0),))))
    monkeypatch.setattr(
        owner, "_evolve_restricted", Mock(side_effect=AssertionError("unexpected reference"))
    )
    for solved, values in ((result, (0.49, 0.04, 0.09)), (shifted, (1.0, 0.25, 0.0))):
        receipt, _ = solved.verify(checks=QHDVerification(comparisons=("grid_minimum",)))
        facts = {f.fact.quantity: f.fact.value.value for f in receipt.applications[0].facts}
        (mode,) = solved.most_probable_indices
        assert facts["grid_minimum"] == 0.0
        assert facts["reference_minimum"] == pytest.approx(min(values), abs=1e-14)
        # The gap of the point that the outer layers read, at the mode's own indices.
        assert facts["most_probable_gap"] == pytest.approx(values[mode] - min(values), abs=1e-14)
        assert facts["minimum_success_mass"] > 0.0
    assert shifted.candidate_indices == (2,) and shifted.most_probable_indices == (1,)

    # Five nodes in (x - 1/5)**2, D=3, K=3 and d=1 give the complete work charge
    # W_grid = D (d + N_f + 12) = 3*(1+5+12) = 54 (verification._grid_minimum_bytes). At
    # max_work=1 only one node is visited, giving the lower bound 3*(1+1+12) = 42, and with that
    # count the one-entry byte minimum 8dK + 8D + 65536 + 256 (N + 2d + 16) + 8 (N + 2) + 8d + 96
    # = 24 + 24 + 65536 + 4864 + 128 = 70576.
    checks = QHDVerification(comparisons=("grid_minimum",), max_work=1, max_bytes=216)
    with pytest.raises(ValueError, match="requires at least 42 work units and at least 70576 bytes") as caught:
        result.verify(checks=checks)
    message = str(caught.value)
    assert "lower bounds on the full admission charges" in message
    assert "Set QHDVerification(max_bytes=" not in message
    assert "Using the displayed lower bound can still refuse" in message
    assert "These limits belong to QHDVerification, not QHD" in message


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_grid_minimum_beside_the_onehot_ir_product_reference_holds_the_restricted_state_once(execution):
    # The one-hot ir_product reference law holds the restricted stored state, 16D bytes
    # (verification._reference_sizes), which the grid-minimum phase also holds with the stored artifact
    # (verification._held_state_bytes). Adding the reference to grid_minimum adds its bytes less that state.
    import re

    from nwqlib.algorithms.qhd.verification import _reference_sizes

    x, y = sp.symbols("x y")
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2 + (y + sp.Rational(1, 3)) ** 2,
                           variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    result = solve(problem, method=QHD(num_grid_points=6, num_steps=4, total_time=1.0, keep_state=True,
                                       theory_flavor="ir_product"), execution=execution, seed=1)
    charges = []
    for comparisons in (("grid_minimum",), ("ir_product_fidelity", "grid_minimum")):
        with pytest.raises(ValueError, match="bytes") as caught:
            result.verify(checks=QHDVerification(comparisons=comparisons, max_bytes=1))
        charges.append(int(re.search(r"(\d+) bytes", str(caught.value)).group(1)))
    dimension = result.plan.reconstruction.restricted_dimension
    reference = _reference_sizes(result.plan, owner._grid(result.plan), "ir_product")[1]
    assert charges[1] == charges[0] + reference - 16 * dimension


def test_symbolic_work_is_admitted_before_expansion_and_table_evaluation(monkeypatch):
    import nwqlib.algorithms.qhd.compiler as compiler

    x, y = sp.symbols("x y")
    unit = ((0.0, 1.0), (0.0, 1.0))
    # sp.expand writes the first four powers through (x + y)**1_000_000, which
    # has 1_000_001 monomials. It multiplies them by sqrt(x + y) for the
    # half-integer exponent and by (x + y)**sqrt(2) for the exponent sum, and
    # it puts them in one denominator for the negative exponent. In the last
    # power the even powers of sqrt(x + y) become integer powers of x + y,
    # about 250_000 monomials in all. Each is refused before the expansion.
    b = x + y
    objectives = (b**1_000_000, b ** sp.Rational(2_000_001, 2), b ** (sp.sqrt(2) + 1_000_000),
                  b**-1_000_000, (sp.sqrt(b) + 1) ** 999)
    with monkeypatch.context() as guard:
        guard.setattr(owner, "ObjectiveDecomposer", Mock(side_effect=AssertionError("expanded")))
        for objective in objectives:
            p = Optimization(objective=objective, variables=(x, y), bounds=unit)
            with pytest.raises(ValueError, match="max_work|max_bytes"):
                make_plan(p, QHD(max_work=1000))

    # The initial amplitudes cost one unit per grid point of each variable,
    # d K = 1000 here, and 320 d K + (8 d K + 120 d + 40) + (8 d K + 120 d + 64 + 204 d)
    # + 4096 (d + 1) = 349,280 bytes. The one default midpoint step row adds
    # 2 units and 768 bytes, and both are admitted before the amplitudes are
    # evaluated (QHD.plan, method._initial_state_bytes).
    gaussian = GaussianState(center=(0.5, 0.5), widths=(0.2, 0.2))
    with monkeypatch.context() as guard:
        guard.setattr(GaussianState, "_variables", Mock(side_effect=AssertionError("evaluated")))
        for limits in (dict(max_work=1001), dict(max_bytes=350_048 - 1)):
            with pytest.raises(ValueError, match="max_work=.*max_bytes="):
                make_plan(Optimization(objective=x + y, variables=(x, y), bounds=unit),
                          QHD(num_grid_points=500, initial_state=gaussian, **limits))
        # At exactly the allowance the early check passes and the evaluation starts.
        with pytest.raises(AssertionError, match="evaluated"):
            make_plan(Optimization(objective=x + y, variables=(x, y), bounds=unit),
                      QHD(num_grid_points=500, initial_state=gaussian, max_bytes=350_048))

    # Each support table evaluates its lambdified expression at K**|S| grid
    # tuples, and the lambdified code evaluates every node of the expression
    # tree. The expansion of (x + y)**(7/2) repeats sqrt(x + y) in each of its
    # four terms, so its tree is several times larger than the objective's.
    lambdified = []
    original = compiler._lambdify_objective

    def spy(variables, expression):
        lambdified.append((len(variables), expression))
        return original(variables, expression)

    monkeypatch.setattr(compiler, "_lambdify_objective", spy)
    p = Optimization(objective=(x + y) ** sp.Rational(7, 2), variables=(x, y), bounds=unit)
    method = QHD(num_grid_points=2, num_steps=1, initial_state_preparation="none")
    make_plan(p, method)
    # The admission opens with the initial state's d K = 4 units and the step
    # rows, 2 point values for the one midpoint step (QHD._admit_symbolic_work).
    tables = sum(
        2**width * (sum(1 for _ in sp.preorder_traversal(expression)) + width)
        for width, expression in lambdified
    )
    evaluated = 4 + 2 + tables
    # The table charge of the owner equals that count, without the compiled
    # blocks that a quantum Plan adds.
    with pytest.raises(ValueError, match="max_work"):
        method.revise(max_work=evaluated - 1)._admit_symbolic_work(p, compact=False)
    method.revise(max_work=evaluated)._admit_symbolic_work(p, compact=False)
    # The integrated rule's rows are the interval integrals of every step, for
    # the cubic schedule at most 128 Gauss-Legendre integrand evaluations for A
    # and one closed form for B (schedules.CubicSchedule.kinetic_integral).
    integrated = method.revise(schedule=CubicSchedule(s=1.0), coefficient_rule="integrated", num_steps=3)
    with pytest.raises(ValueError, match="max_work"):
        integrated.revise(max_work=4 + 3 * 129 + tables - 1)._admit_symbolic_work(p, compact=False)
    integrated.revise(max_work=4 + 3 * 129 + tables)._admit_symbolic_work(p, compact=False)
    # The quantum Plan, with the blocks of its one step, fits in twice that.
    make_plan(p, method.revise(max_work=2 * evaluated))
    # The rows' work and their measured bytes per step are admitted before any
    # row is formed, under either limit. The classical flavors compile no
    # blocks, whose charge would refuse a quantum Plan first, so only the row
    # charge stands between these limits and the rows.
    with monkeypatch.context() as guard:
        guard.setattr(compiler, "step_weights", Mock(side_effect=AssertionError("rows formed")))
        for flavor in ("schrodinger", "split_step"):
            for limits in (dict(max_work=10**6), dict(max_bytes=10**8)):
                with pytest.raises(ValueError, match="max_work=.*max_bytes="):
                    make_plan(p, QHD(num_grid_points=2, num_steps=10**6, theory_flavor=flavor, **limits),
                              execution="classical")


def test_compact_support_cache_survives_step_reuse(monkeypatch):
    import nwqlib.algorithms.qhd.compiler as compiler

    calls = []
    original = compiler._lambdify_objective

    def compile(*args):
        evaluator = original(*args)

        def evaluate(*point):
            calls.append(point)
            return evaluator(*point)

        return evaluate

    monkeypatch.setattr(compiler, "_lambdify_objective", compile)
    selected = make_plan(method=QHD(num_grid_points=3, num_steps=5, theory_flavor="ir_product"))
    assert len(calls) == 1 and np.array_equal(calls[0][0], [-0.5, 0.0, 0.5])
    assert selected.reconstruction.support_evaluations == 3
    assert len(selected.reconstruction.steps) == 5


def test_chunked_support_tables_keep_the_c_order_of_their_grid_tuples(monkeypatch):
    """A support table evaluated in several chunks stores the exact values of its grid tuples in C order.

    ``compiler.evaluate_support`` evaluates a table in flattened C-order chunks
    of the size that the planning admission chooses (``method._support_table_bytes``),
    forming the coordinates of entry e from its mixed-radix digits
    (``compiler.chunk_coordinates``). A callable that cannot be evaluated on
    arrays keeps its scalar functionality: its chunks are evaluated entry by
    entry. Here the callable of the support that holds ``Max`` refuses arrays.
    The boxes [-1, 3], [-2, 6] and [-4, 4] with K = 3 have the grid
    coordinates (0, 1, 2), (0, 2, 4) and (-2, 0, 2) and the expansion center
    0, and every coefficient is dyadic, so each table entry is exact in
    binary64. With every chunk forced to 7 entries, which splits the 9- and
    27-entry tables unevenly, each table must equal the exact values at its
    grid tuples, first support variable most significant.
    """
    import nwqlib.algorithms.qhd.compiler as compiler

    x, y, z = sp.symbols("x y z")
    problem = Optimization(objective=x * y * z**2 + 2 * x * y**2 + y * sp.Max(x - 1, 0) - z / 4,
                           variables=(x, y, z), bounds=((-1.0, 3.0), (-2.0, 6.0), (-4.0, 4.0)))
    admitted, lambdify = owner._support_table_bytes, compiler._lambdify_objective

    def sevens(*args):
        reserved, workspace, identity, chunks = admitted(*args)
        return reserved, workspace, identity, dict.fromkeys(chunks, 7)

    def points_only(variables, expression):
        evaluator = lambdify(variables, expression)
        if not expression.has(sp.Max):
            return evaluator

        def evaluate(*coordinates):
            if any(np.ndim(c) for c in coordinates):
                raise TypeError("this callable evaluates one point at a time")
            return evaluator(*coordinates)

        return evaluate

    monkeypatch.setattr(owner, "_support_table_bytes", sevens)
    monkeypatch.setattr(compiler, "_lambdify_objective", points_only)
    selected = make_plan(problem, QHD(num_grid_points=3, num_steps=1))
    grid = owner._grid(selected)
    axes = [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [-2.0, 0.0, 2.0]]
    assert [[grid.grid_value(j, i) for i in range(3)] for j in range(3)] == axes
    exact = {(2,): lambda c: -c / 4, (0, 1): lambda a, b: 2 * a * b * b + b * max(a - 1, 0),
             (0, 1, 2): lambda a, b, c: a * b * c * c}
    tables = {t.support: t.values.array.tolist() for t in selected.reconstruction.support_values}
    assert tables == {support: [f(*point) for point in itertools.product(*(axes[j] for j in support))]
                      for support, f in exact.items()}


def test_numpy_integer_controls_preserve_grid_and_step_settings():
    expected = QHD(num_grid_points=3, num_steps=2, trotter_order=1)
    assert QHD(num_grid_points=np.int64(3), num_steps=np.int32(2), trotter_order=np.int64(1)) == expected


def test_classical_host_admits_the_norm_estimation_of_expm_multiply():
    # With gamma 0 each step's generator G = dt (a K + v diag(V)) has shifted
    # 1-norm N = 75.4, above 63.36, the bound of condition (3.13) of Al-Mohy
    # and Higham, SIAM J. Sci. Comput. 33, 488 (2011), doi:10.1137/100788860,
    # for one vector. SciPy's
    # expm_multiply then also estimates the 1-norms of G**p for p = 2 to 9
    # with onenormest, so a real call makes more products with a vector than
    # its at most 55 ceil(N/9.9) Taylor terms (checked below). Each product
    # costs the stored entries of G plus 7 D units for the vector operations
    # around it, the unit of _linalg_laws.expm_multiply_requirements. A
    # max_work one unit below the counted work of the calls must refuse the
    # Plan. The default limit admits it.
    import scipy.sparse
    import scipy.sparse.linalg
    from nwqlib._linalg_laws import seeded_norm_estimates
    from nwqlib.algorithms.qhd.theory import restricted_kinetic_sparse

    class Counted(scipy.sparse.csr_matrix):
        products = 0

        def _matmul_dispatch(self, other):
            Counted.products += other.shape[1] if getattr(other, "ndim", 1) == 2 else 1
            return super()._matmul_dispatch(other)

    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x**2, variables=(x,), bounds=((0.0, 1.0),))
    method = QHD(num_grid_points=4, num_steps=2, total_time=6.0, schedule=QuadraticSchedule(gamma=0.0))
    selected = plan(problem, method=method, execution="classical")
    grid = owner._grid(selected)
    kinetic = restricted_kinetic_sparse(grid)
    potential = scipy.sparse.diags([owner.objective_at(selected.reconstruction, (i,), 4) for i in range(4)])
    dt = method.total_time / method.num_steps
    work = 0
    for _t, a, v in selected.reconstruction.step_weights:
        generator = scipy.sparse.csr_matrix(dt * (a * kinetic + v * potential))
        shifted = generator - generator.diagonal().mean() * scipy.sparse.identity(4, format="csr")
        norm = abs(shifted).sum(axis=0).max()
        assert 63.36 < norm < 200
        Counted.products = 0
        with seeded_norm_estimates():
            scipy.sparse.linalg.expm_multiply(Counted(-1j * generator), np.ones(4, complex))
        assert Counted.products > 55 * np.ceil(norm / 9.9)
        work += Counted.products * (generator.nnz + 7 * 4)
    with pytest.raises(ValueError, match="max_work"):
        plan(problem, method=method.revise(max_work=work - 1), execution="classical")


@pytest.mark.parametrize("d, k, boundary", [
    (1, 2, "periodic"), (2, 2, "periodic"), (2, 4, "periodic"), (4, 2, "periodic"),
    (2, 2, "dirichlet"), (2, 4, "dirichlet"),
])
def test_the_stencil_work_law_covers_the_raw_writes_that_the_schrodinger_admission_charges(d, k, boundary):
    """The stencil's counted raw writes lie within its work law, which the Schrodinger admission charges.

    The raw row, column and value arrays of ``theory.restricted_kinetic_sparse``
    are counted as they are written, one visit per written element. Their
    3S writes are part of ``W_stencil = 28S + 4M + (37d + 12)D + 40(d + 1)``
    (``theory.restricted_kinetic_work``). ``method.restricted_sizes``
    supplies the boundary-free upper counts ``S = 3dD`` and ``M = (1 + 2d)D``
    to the monotone law, so its work includes the stencil's.
    """
    from unittest.mock import patch
    from nwqlib.algorithms.qhd import theory
    from nwqlib.algorithms.qhd.grid import OneHotGrid

    dimension = k**d
    grid = OneHotGrid(tuple(f"x{j}" for j in range(d)), ((0.0, 1.0),) * d, k, boundary=boundary)
    raw = 3 * d * dimension if boundary == "periodic" else d * dimension + 2 * d * (k - 1) * (dimension // k)
    counts = {"row": 0, "column": 0, "value": 0}
    labels = iter(counts)

    class Counted(np.ndarray):
        def __array_finalize__(self, source):
            self.label = getattr(source, "label", None)

        def __setitem__(self, key, value):
            if self.label is not None:
                counts[self.label] += np.asarray(self[key]).size
            super().__setitem__(key, value)

    def empty(shape, *args, **kwargs):
        array = np.empty(shape, *args, **kwargs)
        if shape == raw:
            array = array.view(Counted)
            array.label = next(labels)
        return array

    class Proxy:
        def __getattr__(self, key):
            return empty if key == "empty" else getattr(np, key)

    with patch.object(theory, "np", Proxy()):
        matrix = theory.restricted_kinetic_sparse(grid)
    assert counts == dict.fromkeys(counts, raw)
    writes = sum(counts.values())
    assert writes <= theory.restricted_kinetic_work(dimension, d, raw, matrix.nnz)
    upper = theory.restricted_kinetic_work(dimension, d, 3 * d * dimension, (1 + 2 * d) * dimension)
    assert theory.restricted_kinetic_work(dimension, d, raw, matrix.nnz) <= upper
    size, _ = owner.restricted_sizes(dimension, d, 0, ((), ()), "schrodinger", iter(()), 0)
    assert size >= upper


@pytest.mark.parametrize("objective, bounds, fields, message", [
    # b = 2.5e303 at the one midpoint times the table range 6.7e5 overflows before dt = 0.01 applies. Every last
    # midpoint lies at T/2 or later, so no step count makes this unscaled product finite.
    (sp.Integer(10) ** 6 * sp.Symbol("x", real=True), ((-1.0, 1.0),),
     dict(num_grid_points=2, total_time=0.01, schedule=QuadraticSchedule(gamma=1e308)),
     "\\|b\\| \\* R = inf,.*Scale the objective down"),
    # H = 4.0e307 from the tiny x box and R = 1.5e308 from 1.25e308 y are finite, and so are their products with
    # dt = 0.1, but their sum overflows before dt applies.
    (sp.Float(1.25e308) * sp.Symbol("y", real=True), ((0.0, 7.9e-154), (-1.0, 1.0)),
     dict(num_grid_points=4, total_time=0.1, schedule=QuadraticSchedule(gamma=0.0)),
     "The sum \\|a\\| \\* H \\+ \\|b\\| \\* R overflows before multiplication by dt"),
    # a = b = 1, h = 0.5 and the Dirichlet K = 2 column bound H = 0.5/h**2 = 2 (one neighbor), with R = 0.5, give
    # the finite sum 2.5, which the final multiplication by dt = 1e308 overflows.
    (sp.Symbol("x", real=True), ((0.0, 1.5),),
     dict(num_grid_points=2, total_time=1e308, schedule=QuadraticSchedule(gamma=0.0)),
     "are finite, but the final multiplication by dt overflows"),
    # The base norm dt*(|a|*H + |b|*R) is finite at about 1.5e308.
    # With J about 1.5 and W_vec about 2, the intermediate
    # dt*(|a|*J + |b|*W_vec) overflows before multiplication by gamma_3.
    # The real gamma_3-scaled bound is finite (about 1.17e293).
    (sp.Symbol("x", real=True), ((0.0, 3.0),),
     dict(num_grid_points=2, total_time=1e308, schedule=QuadraticSchedule(gamma=0.0)),
     r"N=inf, Q=inf.*before gamma_3.*dt = total_time/num_steps.*"
     r"Lower total_time or increase num_steps.*when a and b are fixed"),
    (sp.Symbol("x", real=True), ((-1.0, 1.0),),
     dict(num_grid_points=4, total_time=8e-103, schedule=ShiftedCubicSchedule(s=1e-300)),
     r"unscaled kinetic product overflows: .*a larger s"),
], ids=["unscaled_product", "unscaled_sum", "final_scaling", "assembly_bound", "unscaled_kinetic"])
def test_schrodinger_generator_overflow_names_the_failing_operation(objective, bounds, fields, message):
    """A nonfinite Schrodinger step generator norm is refused with the operation that overflowed.

    The norm ``dt (|a| H + |b| R)``, with H the stored kinetic column bound
    and R the table-range bound, forms the unscaled products and their
    sum before the multiplication by dt (``method._step_norm_refusal``), so
    an overflow in an earlier operation is not repaired by a smaller dt. Each
    case overflows at a different operation of one step, and the refusal
    must name that operation with the remedy of its factors. When that norm
    is finite and the table/assembly bound of
    ``method._schrodinger_step_bounds`` is not, the refusal names that bound.
    """
    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=objective, variables=(x, y)[:len(bounds)], bounds=bounds)
    with pytest.raises(ValueError, match=message):
        plan(problem, method=QHD(num_steps=1, initial_state=UniformState(), **fields), execution="classical")
    if isinstance(fields["schedule"], ShiftedCubicSchedule):
        # Increasing s clears the kinetic overflow; the later work admission may still refuse.
        control = fields | {"schedule": ShiftedCubicSchedule(s=1e-100)}
        try:
            plan(problem, method=QHD(num_steps=1, initial_state=UniformState(), **control), execution="classical")
        except ValueError as error:
            assert "max_work" in str(error)


def test_schrodinger_assembly_bound_admits_adjacent_finite_step():
    """The last finite outward assembly intermediate still yields finite N and Q."""
    from math import isfinite

    from nwqlib.algorithms.qhd.schedules import step_weights

    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x, variables=(x,), bounds=((0.0, 3.0),))
    method = QHD(num_grid_points=2, num_steps=1, total_time=1.0,
                 schedule=QuadraticSchedule(gamma=0.0), initial_state=UniformState())
    selected = plan(problem, method=method, execution="classical")
    # Here h=1, J=3/2, H=1/2, W=2 and R=1 before outward rounding.
    # The outward unscaled sum is 0x1.c000000000005p+1. Exact rational
    # comparison with the midpoint of the two largest finite binary64
    # numbers puts this dt immediately below _mul_up's overflow boundary.
    method = method.revise(total_time=float.fromhex("0x1.249249249248ep+1022"))
    weights = step_weights(method.schedule, method.coefficient_rule,
                           method.total_time, method.num_steps)
    rows = list(owner._schrodinger_step_bounds(
        method, owner._grid(selected), selected.reconstruction.support_values, weights))
    assert len(rows) == 1
    norm, population, perturbation = rows[0]
    assert population == 3 and isfinite(norm) and isfinite(perturbation)


@pytest.mark.parametrize("num_steps, total_time", [
    (1, float.fromhex("0x1.249249249248ep+1022")), (1, 2e307), (2, 5e302)])
def test_a_plan_at_the_binary64_edge_refuses_its_nonfinite_state_budget(num_steps, total_time):
    """Plans at the binary64 edge refuse their nonfinite state error budget.

    The first duration is the last one that method._schrodinger_step_bounds
    admits on this grid. Its generator norm N is near the binary64 maximum,
    so the integer product count S of expm_multiply_call_roundoff does not
    convert to a float. At the second duration S converts and the charge
    overflows in float arithmetic. In the two-step case each call's charge
    is finite, but their sum in expm_multiply_state_error exceeds binary64.
    The work and byte limits are raised far above the charge so that the
    budget, not a resource limit, is the refusal.
    """
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x, variables=(x,), bounds=((0.0, 3.0),))
    method = QHD(num_grid_points=2, num_steps=num_steps, total_time=total_time,
                 max_work=10**400, max_bytes=10**400,
                 schedule=QuadraticSchedule(gamma=0.0), initial_state=UniformState())
    with pytest.raises(ValueError, match=r"state error budget .* is inf, not a finite binary64 number"):
        plan(problem, method=method, execution="classical")


def test_the_stored_kinetic_diagonal_is_the_ordered_sum_in_every_row():
    """The stored Schrodinger kinetic diagonal is one ordered binary64 sum, so its range is zero.

    ``method._schrodinger_kinetic_bounds`` prices the stored matrix with a
    zero diagonal range, which rests on ``theory.restricted_kinetic_sparse``'s
    column-major COO fill and SciPy's ordered duplicate reduction. With
    d = 4, K = 2 and axis contributions ``(1, 2**-54, 2**-54, 2**-54)`` the
    left-to-right sum is ``0x1.0000000000000p+0`` and ``fsum`` is
    ``0x1.0000000000000p+0`` plus one ulp, so a reordered or accurate
    reduction shows in the stored bits.
    """
    from math import fsum
    from nwqlib.algorithms.qhd.grid import OneHotGrid
    from nwqlib.algorithms.qhd.theory import restricted_kinetic_sparse

    grid = OneHotGrid(tuple(f"x{j}" for j in range(4)), ((0.0, 2.0),) + ((0.0, 2.0**28),) * 3, 2,
                      boundary="periodic")
    contributions = [1.0 / (grid.spacing(j) * grid.spacing(j)) for j in range(4)]
    assert contributions == [1.0, 2.0**-54, 2.0**-54, 2.0**-54]
    diagonal = restricted_kinetic_sparse(grid).diagonal()
    assert np.all(diagonal.imag == 0)
    assert {float(v).hex() for v in diagonal.real} == {"0x1.0000000000000p+0"}
    assert fsum(contributions).hex() == "0x1.0000000000001p+0"


def test_the_schrodinger_step_bounds_cover_the_assembled_generator_exactly():
    """N and Q of each step bound the exact shifted norm and assembly error of the kernel's generator.

    ``method._schrodinger_step_bounds`` bounds, in column 1-norm, the error
    of the assembled generator ``RN(dt*RN(RN(a*Khat)+RN(b*Vvec)))`` against
    ``dt (a Khat + b diag(V*))`` with V* the exact sum of the stored tables
    (Q), and the assembled generator after an exact trace shift (N). The
    oracle forms both quantities in exact rationals from the generator that
    ``method._fill_generator`` actually writes, on a d = 2, K = 8 Dirichlet
    grid with three stored tables and two steps.
    """
    from fractions import Fraction
    from types import SimpleNamespace
    from nwqlib.algorithms.qhd.grid import OneHotGrid
    from nwqlib.algorithms.qhd.split_step import potential_diagonal
    from nwqlib.algorithms.qhd.theory import restricted_kinetic_sparse

    grid = OneHotGrid(("x0", "x1"), ((0.0, 9 * 0.13), (0.0, 9 * 0.37)), 8)
    tables = (
        SupportValues.tabulate((0,), np.array([0.1, 0.2, -0.3, 1.0, -1.0, 0.6, 1.2, -0.1])),
        SupportValues.tabulate((1,), np.array([0.3, -0.2, 0.1, -0.5, 0.7, 0.9, -1.1, 0.4])),
        SupportValues.tabulate((0, 1), np.arange(64, dtype=float) / 1000),
    )
    method = SimpleNamespace(total_time=0.017, num_steps=2)
    weights = ((0.0, 0.37, 0.81), (0.1, 0.22, 1.13))
    kinetic = restricted_kinetic_sparse(grid)
    generator, slots, diagonal = owner._generator_workspace(kinetic)
    assert np.shares_memory(generator.indices, kinetic.indices) and np.shares_memory(generator.indptr, kinetic.indptr)
    # SciPy chooses int64 indices from the raw slot count of a large stencil although the final pattern fits
    # int32. The generator must still share that pattern rather than a narrowed copy.
    wide = kinetic.copy()
    wide.indices, wide.indptr = wide.indices.astype(np.int64), wide.indptr.astype(np.int64)
    shared = owner._generator_workspace(wide)[0]
    assert shared.indices is wide.indices and shared.indptr is wide.indptr
    potential = potential_diagonal(SimpleNamespace(support_values=tables), grid).astype(np.complex128).reshape(-1)
    exact = [sum((Fraction(float(t.values.array[sum(index[j] * 8 ** (len(t.support) - p - 1)
                                                        for p, j in enumerate(t.support))]))
                  for t in tables), Fraction(0)) for index in np.ndindex(8, 8)]
    dense = kinetic.toarray()
    dt = method.total_time / method.num_steps
    bounds = list(owner._schrodinger_step_bounds(method, grid, tables, weights))
    assert len(bounds) == 2
    for (_, a, b), (norm, rows, perturbation) in zip(weights, bounds, strict=True):
        assert rows == 5
        formed = owner._fill_generator(generator, kinetic, slots, diagonal, potential, a, b, dt).toarray()
        # -i RN(dt*h): the real parts are zero and the imaginary parts are -RN(dt*h).
        assert np.all(formed.real == 0)
        entries = [[-Fraction(float(formed[row, col].imag)) for col in range(64)] for row in range(64)]
        shift = sum((entries[i][i] for i in range(64)), Fraction(0)) / 64
        exact_norm = exact_error = Fraction(0)
        for col in range(64):
            column_norm = column_error = Fraction(0)
            for row in range(64):
                target = Fraction(dt) * Fraction(a) * Fraction(float(dense[row, col].real))
                if row == col:
                    target += Fraction(dt) * Fraction(b) * exact[col]
                column_error += abs(entries[row][col] - target)
                column_norm += abs(entries[row][col] - (shift if row == col else 0))
            exact_norm, exact_error = max(exact_norm, column_norm), max(exact_error, column_error)
        assert exact_error > 0
        assert Fraction(norm) >= exact_norm and Fraction(perturbation) >= exact_error


def test_schrodinger_assembly_bound_covers_gradual_underflow():
    """The assembly bound covers a legal subnormal generator without rejecting its Plan."""
    from fractions import Fraction
    import numpy as np
    import sympy as sp
    from nwqlib import Optimization, plan
    from nwqlib.algorithms.qhd import QHD, UniformState
    from nwqlib.algorithms.qhd import method as owner, theory

    variables = sp.symbols("x0:7", real=True)
    selected = plan(
        Optimization(objective=sp.Integer(0), variables=variables,
                     bounds=((0.0, 2e150),) * 7),
        method=QHD(encoding="binary", boundary="periodic", num_grid_points=2,
                   num_steps=1, total_time=3e-24, initial_state=UniformState()),
        execution="classical", seed=7,
    )
    grid = owner._grid(selected)
    r = selected.reconstruction
    kinetic = theory.restricted_kinetic_sparse(grid)
    generator, slots, diagonal = owner._generator_workspace(kinetic)
    _, a, b = r.step_weights[0]
    dt = selected.method.total_time / selected.method.num_steps
    owner._fill_generator(generator, kinetic, slots, diagonal,
                          np.zeros(128, dtype=complex), a, b, dt)
    stored = kinetic.toarray().real
    actual = -generator.toarray().imag
    error = max(
        sum((abs(Fraction(float(actual[row, col]))
                 - Fraction(dt) * Fraction(a) * Fraction(float(stored[row, col])))
             for row in range(128)), Fraction(0))
        for col in range(128)
    )
    _, _, bound = next(owner._schrodinger_step_bounds(
        selected.method, grid, r.support_values, r.step_weights))
    assert error > 0
    assert error <= Fraction(bound)


def test_the_broadcast_fixed_pattern_state_lies_within_the_computed_state_comparison_of_the_sparse_kernel():
    """The Schrodinger kernel's state lies within the computed-state comparison bound of the former kernel.

    The former kernel summed each point's tables with ``fsum`` and formed
    ``-1j * dt * (a K + b diag(V))`` by sparse arithmetic. For two computed
    states whose exact products use assembled generators within ``Q_cmp,k``
    of each other, Duhamel's identity and unitarity give
    ``||psi_new - psi_old||_2 <= delta_new_scipy + delta_old_scipy
    + sum_k Q_cmp,k``, where each delta is the numerical charge
    ``expm_multiply_state_error`` of that kernel's own shifted norms, and
    ``Q_cmp,k = |dt b_k| E_cmp + gamma_3 |dt| (2 |a_k| J + |b_k| (W_vec + W_point))``
    with ``E_cmp = (gamma_m + rho u) W``, ``W_point = (1 + rho u) W``, m = T-1
    and ``rho = 2``, the one-ulp premise for ``fsum``'s final rounding. The
    state distance is exact in rationals. The objective has a nonzero
    constant, whose phase neither kernel's state carries.
    """
    from fractions import Fraction
    from math import fsum, nextafter, inf
    import scipy.sparse
    import scipy.sparse.linalg
    from nwqlib._linalg_laws import seeded_norm_estimates
    from nwqlib._validation import expm_multiply_state_error
    from nwqlib.algorithms.qhd.initial_state import restricted_state
    from nwqlib.algorithms.qhd.theory import restricted_kinetic_sparse

    x, y = sp.symbols("x y", real=True)
    # The factor 40 makes the potential comparable to the kinetic diagonal, so the broadcast and pointwise
    # tables, which differ in the last bit at some points, give different generators and states.
    problem = Optimization(objective=40 * ((x - sp.Rational(1, 5)) ** 2 + x * y + (y + sp.Rational(1, 3)) ** 2 + 3),
                           variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    selected = plan(problem, method=QHD(num_grid_points=8, num_steps=2, total_time=0.7, theory_flavor="schrodinger",
                                        initial_state=UniformState()), execution="classical")
    r = selected.reconstruction
    assert r.constant != 0 and len(r.support_values) >= 2
    new = owner._evolve_restricted(selected, "schrodinger")

    grid = owner._grid(selected)
    d, k = 2, 8
    dt = selected.method.total_time / selected.method.num_steps
    kinetic = restricted_kinetic_sparse(grid)
    point = [fsum(owner._table_values(r, i, k)) for i in itertools.product(range(k), repeat=d)]
    potential = scipy.sparse.diags(point, format="csr", dtype=complex)
    old = restricted_state(selected.method.initial_state, grid, r.initial_amplitudes)
    old_norms = []
    with seeded_norm_estimates():
        for _t, a, b in r.step_weights:
            generator = -1j * dt * (a * kinetic + b * potential)
            dense = (1j * generator).toarray().real
            shift = sum((Fraction(float(dense[i, i])) for i in range(k**d)), Fraction(0)) / k**d
            exact = max(sum((abs(Fraction(float(dense[row, col])) - (shift if row == col else 0))
                             for row in range(k**d)), Fraction(0)) for col in range(k**d))
            old_norms.append((nextafter(float(exact), inf), 1 + 2 * d))
            old = scipy.sparse.linalg.expm_multiply(generator, old)

    start = r.initial_state_error
    delta_new = expm_multiply_state_error(
        ((n, rows) for n, rows, _ in owner._schrodinger_step_bounds(selected.method, grid, r.support_values,
                                                                    r.step_weights)), start=start)
    delta_old = expm_multiply_state_error(old_norms, start=start)
    up = owner._mul_up
    add = owner._add_up
    j_bound, _ = owner._schrodinger_kinetic_bounds(grid)
    magnitude = owner._sum_up(max(abs(t.minimum), abs(t.maximum)) for t in r.support_values)
    m = len(r.support_values) - 1
    rho_u = 2 * owner.UNIT_ROUNDOFF
    vector = add(magnitude, up(owner._gamma_up(m), magnitude))
    pointwise = up(add(1.0, rho_u), magnitude)
    compared = up(add(owner._gamma_up(m), rho_u), magnitude)
    gamma3 = owner._gamma_up(3)
    comparison = [add(up(up(abs(dt), abs(b)), compared),
                      up(gamma3, up(abs(dt), add(up(2 * abs(a), j_bound), up(abs(b), add(vector, pointwise))))))
                  for _t, a, b in r.step_weights]
    bound = Fraction(delta_new) + Fraction(delta_old) + sum((Fraction(q) for q in comparison), Fraction(0))
    squared = Fraction(0)
    for p, q in zip(new.flat, old.flat, strict=True):
        re = Fraction(float(p.real)) - Fraction(float(q.real))
        im = Fraction(float(p.imag)) - Fraction(float(q.imag))
        squared += re * re + im * im
    assert 0 < squared <= bound**2


@pytest.mark.parametrize("d, k", [(2, 1024), (1, 4096)])
def test_the_dense_summary_fits_its_decoding_and_mean_phase_law(d, k):
    """One warmed ``_summarize`` call on dense probabilities stays within its incremental byte law.

    ``method.summary_sizes`` admits ``H0 + max(17D + H_decode, 25D + 8dK)``
    incremental bytes beside the borrowed 8D probability array, with
    ``H_decode = 64C(d+1) + 512(d+1)`` for ``C = min(P, 4096)`` positive
    positions decoded at a time. At d = 2, K = 1024 the mean phase binds and
    the exhausted decoding iterator of the last chunk must be released
    before it. At d = 1, K = 4096 the decoding phase binds, above the
    ``25D + 8dK + H0`` form. The Plan and weights are live before tracing.
    """
    import gc
    import tracemalloc

    names = sp.symbols(f"x0:{d}")
    objective = sum((v - sp.Rational(1, 5)) ** 2 for v in names) + (names[0] * names[-1] / 3 if d > 1 else 0)
    selected = plan(Optimization(objective=objective, variables=names, bounds=((-1.0, 1.0),) * d),
                    method=QHD(num_grid_points=k, num_steps=1, total_time=0.3, theory_flavor="split_step",
                               max_work=10**13, max_bytes=10**11), execution="classical", seed=7)
    weights = np.random.default_rng(3).random((k,) * d)
    weights /= weights.sum()
    weights.flat[:3] = 0
    expected = owner._summarize(selected, weights, 0.0, None, 1e-12)
    gc.collect()
    tracemalloc.start()
    try:
        actual = owner._summarize(selected, weights, 0.0, None, 1e-12)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert actual.keys() == expected.keys()
    admitted = owner.summary_sizes(selected, d, k, int(np.count_nonzero(weights)))[1]
    assert peak <= admitted - weights.nbytes


@pytest.mark.parametrize("flavor, steps, d, k", [("schrodinger", 64, 1, 2), ("ir_product", 256, 1, 2),
                                                  ("ir_product", 1, 4, 30)])
def test_classical_kernel_call_fits_its_admitted_workspace(flavor, steps, d, k, monkeypatch):
    """The traced peak of one warmed classical kernel call fits the workspace that its Plan admits.

    Besides the size laws, the kernel's admission adds the measured per-call
    allowance ``method._KERNEL_CALL_BYTES`` for Python objects of bounded
    size, and the kernel reads its generator norms and compiled blocks one
    at a time (``method._generator_norms``, ``native.raw_blocks``). With
    d = 1 and K = 2 the size laws admit about 18 kB, so both peaks exceed
    the admission without the allowance, and the 256-step ir_product peak
    exceeds it with the allowance when all blocks are held at once (about
    500 kB against 280 kB with the materializing functions of efe52699,
    traced on 2026-09-27 at develop 21d75e3c). A warm-up solve of the same Plan
    loads the modules and fills CPython's free lists, and garbage is
    collected before tracing, so the traced peak excludes one-time growth.
    With d = 4 and K = 30, D = 810,000 grid points, the arrays proportional to
    D dominate the admission, and the call fits only while the summary keeps
    its storage within ``H0 + max(17D + H_decode, 25D + 8dK)``
    (``method._summarize``, ``method.summary_sizes``).
    """
    import gc
    import tracemalloc

    names = sp.symbols(f"x0:{d}")
    problem = Optimization(objective=sum((v - sp.Rational(1, 5)) ** 2 for v in names), variables=names,
                           bounds=((-1.0, 1.0),) * d)
    # The large grid needs limits above the defaults.
    limits = dict(max_work=10**13, max_bytes=10**11) if d > 1 else {}
    selected = plan(problem, method=QHD(num_grid_points=k, num_steps=steps, total_time=0.25, theory_flavor=flavor,
                                        initial_state=UniformState(), **limits), execution="classical", seed=7)
    solve(selected)
    peaks = []
    original = owner._execute_theory

    def traced(*args):
        gc.collect()
        tracemalloc.start()
        try:
            output = original(*args)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        return output

    monkeypatch.setattr(owner, "_execute_theory", traced)
    solve(selected)
    admitted = selected.construction.kernels[0].workspace[0].bytes
    assert len(peaks) == 1 and peaks[0] <= admitted


def _global_random_state_equal(first, second):
    return (first[0] == second[0] and np.array_equal(first[1], second[1])
            and tuple(first[2:]) == tuple(second[2:]))


@pytest.mark.parametrize("flavor", ["schrodinger"])
def test_classical_host_evolution_keeps_the_global_random_state(flavor):
    # This schedule has generators with shifted 1-norm above 63.36, where
    # SciPy's expm_multiply estimates norms with onenormest, which draws from
    # NumPy's global generator. The evolution must leave the caller's state
    # unchanged and give the same state bitwise for different caller states.
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x**2, variables=(x,), bounds=((0.0, 1.0),))
    method = QHD(num_grid_points=4, num_steps=2, total_time=30.0, schedule=QuadraticSchedule(gamma=0.0),
                 theory_flavor=flavor)
    selected = plan(problem, method=method, execution="classical")
    r = selected.reconstruction
    norms = owner._generator_norms(flavor, selected.method, owner._grid(selected),
                                   r.support_values, r.steps, r.step_weights)
    assert max(norm for norm, _ in norms) > 63.36
    np.random.seed(1)
    before = np.random.get_state()
    first = owner._evolve_restricted(selected, flavor)
    assert _global_random_state_equal(np.random.get_state(), before)
    np.random.seed(2)
    second = owner._evolve_restricted(selected, flavor)
    assert np.array_equal(first, second)


def test_initial_state_and_schedule_work_refuse_insufficient_limits():
    """The initial state and schedule reject insufficient planning work and storage."""
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=2, num_steps=1, max_work=1, max_bytes=1)
    with pytest.raises(ValueError, match="initial state and schedule rows requires"):
        plan(problem, method=method, execution="classical")


def test_native_acquisition_batch_preserves_selected_readout_after_reopen(tmp_path, monkeypatch):
    """One exact amplitude acquisition survives saved preparation with its selected identity.

    Kept amplitudes use one evaluation. The aggregate belongs to the base Plan,
    while an amplitude artifact belongs to its selected child construction.
    Reopening must reuse the prepared circuit and preserve that association.
    """
    import nwqlib
    from nwqlib.backends import AerBackend
    from nwqlib.blocks import lowering
    from nwqlib.ir import MeasurementBatch

    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=2,
                 num_steps=1, total_time=0.1, keep_state=True)
    selected = plan(problem, method=method, execution="quantum", seed=7)
    batches = [d.node for d in selected.construction.program.definitions if isinstance(d.node, MeasurementBatch)]
    assert len(batches) == 1
    batch = batches[0]
    assert tuple(s.label for s in batch.settings) == ("qhd",)
    assert batch.repetitions == 1 and batch.observation_kind == "probabilities"
    point = selected.resolve("qhd")
    child = point.selected_construction(selected)
    assert child.content_id != selected.construction.content_id
    setting, observation = point.resolved_observation(selected)
    assert setting == "qhd" and observation.kind == "amplitudes"
    assert observation.amplitudes.construction_id == child.content_id
    assert not observation.qubits
    directory = tmp_path / "run"
    prepared = nwqlib.prepare(selected, backend=AerBackend(), settings="all", directory=directory)
    assert prepared.setting_names == ("qhd",) and len(prepared.circuits) == 1
    assert prepared.run.trace.jobs == 0
    prepared_ids = prepared.run.trace.local_prepared_ids
    prepared.run.close()

    def forbidden(*args, **kwargs):
        pytest.fail("reopened QHD preparation rebuilt the circuit")

    monkeypatch.setattr(lowering, "_lower_qiskit", forbidden)
    with nwqlib.load_run(directory, backend=AerBackend()) as run:
        result = run.wait()
        assert run.trace.local_prepared_ids == prepared_ids
        assert run.trace.preparations == run.trace.jobs == 1
        assert result.construction_id == selected.construction.content_id
        chunk = result.data.observations.chunks[0]
        assert chunk.observation == observation
        assert result.artifact.construction_id == child.content_id
        result.validate_plan(selected)
        saved = result.save(tmp_path / "result")
    loaded = nwqlib.load_result(saved)
    assert loaded.candidate_indices == result.candidate_indices
    assert loaded.most_probable_probability == result.most_probable_probability


def test_a_support_table_refusal_names_the_users_expression_and_coordinates():
    """A refused support-table entry is reported in the coordinates and expression the user wrote.

    Tables are evaluated in the centered coordinates ``x - m`` (m = 2 on
    [2, 4]), where ``log(x - 3)`` reads ``log(x - 1)`` and the first interior
    point of K = 5 is 1/3. The refusal must name ``log(x - 3)`` at
    x = 2 + 1/3 (``compiler.user_point``), for the objective, for an
    augmented-Lagrangian constraint table and for a box refinement's default
    search model, which tabulates ``log(2 u_x - 1)`` at the unit point 1/6
    (``refinement._problem_coordinates``).
    """
    from nwqlib.algorithms.qhd import AugmentedLagrangian, BoxRefinement, plan_augmented_lagrangian, refine_box
    from nwqlib.problems import ConstrainedOptimization

    x = sp.Symbol("x", real=True)
    at = r"log\(x - 3\) on support \(0,\) at grid point \(2\.3333333333333335,\)"
    with pytest.raises(ValueError, match="QHD objective " + at):
        plan(Optimization(objective=sp.log(x - 3), variables=(x,), bounds=((2.0, 4.0),)),
             method=QHD(num_grid_points=5), execution="classical")
    with pytest.raises(ValueError, match=r"inequalities\[0\] " + at):
        plan_augmented_lagrangian(
            ConstrainedOptimization(objective=x**2, variables=(x,), bounds=((2.0, 4.0),),
                                    inequalities=(sp.log(x - 3),)),
            qhd=QHD(num_grid_points=5), options=AugmentedLagrangian(max_iterations=1), execution="classical")
    with pytest.raises(ValueError, match="QHD objective " + at):
        refine_box(Optimization(objective=sp.log(x - 3), variables=(x,), bounds=((2.0, 4.0),)),
                   qhd=QHD(num_grid_points=5), options=BoxRefinement(max_levels=2), execution="classical",
                   seed=7, progress=False)
    # A search-model level's off-grid evaluation, as of a mode_or_mean mean, reports the problem's point
    # a + D u too: at u = 1/2 on [2, 4] the expression 1/(2 u_x - 1) of the unit problem is 1/(x - 3) at 3.
    from nwqlib.algorithms.qhd import refinement

    pole, box = Optimization(objective=1 / (x - 3), variables=(x,), bounds=((2.0, 4.0),)), ((2.0, 4.0),)
    unit, objective = refinement._unit_objective(pole, box)
    tables, decomposer, _ = refinement._support_tables(
        QHD(num_grid_points=4), Optimization(objective=objective, variables=unit, bounds=((0.0, 1.0),)))
    evaluate = refinement._tabulated_objective(decomposer, sp.Integer(0), tables, 4,
                                               refinement._problem_coordinates(pole, box))
    with pytest.raises(ValueError, match=r"QHD objective 1/\(x - 3\) on support \(0,\) at point \(3\.0,\)"):
        evaluate((0.5,))
