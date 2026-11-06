"""Natural QHD, independent finite differences, actual 2–4 qubit readouts."""

import itertools
from unittest.mock import Mock
import mpmath
import numpy as np
import pytest
import scipy.linalg
import sympy as sp
from nwqlib.algorithms.qhd import (
    QHD, BoxRefinement, CubicSchedule, GaussianState, KineticGroundState, QHDVerification, QuadraticSchedule,
    ShiftedCubicSchedule, UniformState, refine_box, step_blocks,
)
from nwqlib.algorithms.qhd.compiler import QHDCompiler
from nwqlib.algorithms.qhd.onehot_schedule import ENTRY_TILE, OneHotSchedule
from nwqlib.algorithms.qhd import method as owner
from nwqlib.algorithms.qhd.records import table_extrema
from nwqlib.algorithms.qhd.validation import _lambdify_objective
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


def _tabulated_values(selected):
    """Return the Plan's constant and support tables added on its grid, and the meshed grid coordinates."""
    grid = owner._grid(selected)
    variables, k = len(selected.problem.variables), selected.method.num_grid_points
    mesh = np.meshgrid(*(np.asarray(grid.coordinate_array(j)) for j in range(variables)), indexing="ij")
    values = np.full(mesh[0].shape, float(selected.reconstruction.constant))
    for table in selected.reconstruction.support_values:
        shape = [k if j in table.support else 1 for j in range(variables)]
        values = values + np.asarray(table.values).reshape(shape)
    return values, mesh


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
    """The one-hot ir_product state equals the ordered dense exponentials of the Plan's blocks at d = 2, K = 4.

    The reference applies ``exp(-i G_b)`` of every block that ``step_blocks``
    returns, in circuit order, at its angle, ``G = angle (Pi_S - 2**-s I)`` for a projector
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
    blocks = [block for step in range(len(r.step_weights)) for block in step_blocks(selected, step)]
    k, d = 4, 2
    dimension = k**d
    # The test's stated caps for both routes together.
    work_cap, byte_cap = 10**8, 10**8
    direct_work, direct_bytes = onehot_product_sizes(dimension, k, d, r.onehot, start_work=dimension,
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
    allowance = (onehot_product_state_error(r.onehot, start=r.initial_state_error)
                 + expm_multiply_state_error([(abs(block.angle), 1) for block in blocks],
                                             start=r.initial_state_error, phase_multiplications=1))
    assert np.linalg.norm(state - reference) <= allowance


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
    assert step_blocks(selected, 0) == ()
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
def test_planning_stores_the_initial_state_and_its_error_bound(execution, flavor, initial):
    """Planning stores the initial state and its error bound, and both equal a fresh evaluation.

    ``QHD.plan`` stores the per-variable amplitudes and the start vector's
    error bound in the reconstruction (``initial_amplitudes``,
    ``initial_state_error``). The stored values must
    equal a fresh evaluation bitwise. A classical Plan also stores the
    kernel's state budget delta, which must equal a fresh
    ``method._host_state_error`` of the Plan's own stored tables, steps,
    step rows and start error, so a planning call that passes other data
    fails.
    """
    from nwqlib.algorithms.qhd.initial_state import restricted_state_error

    x, y = sp.symbols("x y")
    problem = Optimization(objective=x**2 + x * y / 4, variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    selected = make_plan(problem, QHD(
        num_grid_points=4, num_steps=2, total_time=0.5, schedule=QuadraticSchedule(gamma=0.3),
        theory_flavor=flavor, keep_state=True, initial_state=initial,
    ), execution=execution)
    grid = owner._grid(selected)
    stored = selected.reconstruction
    assert len(stored.initial_amplitudes) == 2 and all(
        np.array_equal(a.array, np.asarray(v, dtype=np.float64))
        for a, v in zip(stored.initial_amplitudes, initial.variable_amplitudes(grid), strict=True))
    assert stored.initial_state_error == restricted_state_error(initial, grid)
    if execution == "classical":
        assert stored.host_state_error == owner._host_state_error(
            selected.method, grid, stored.support_values,
            stored.steps if stored.onehot is None else stored.onehot, stored.step_weights,
            stored.initial_state_error)
    result = solve(selected)
    assert result.analyze().candidate == result.candidate


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
    assert [(b.kind, b.support) for b in step_blocks(resource_only, 0)] == [
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
    bounds = tuple(owner._generator_norms(flavor, selected.method, grid, r.support_values, r.step_weights))
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
    blocks = [b for step in range(len(r.step_weights)) for b in step_blocks(probabilities, step)
              if b.kind == "number_projector"]
    tiles = sum(-(-len(t.values) // ENTRY_TILE) for t in r.support_values)
    contributions = method.num_steps * (1 + (r.constant != 0) + 2 * tiles)
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
    # beta' = u + (1 + u) beta_M of the tiled one-hot sum (onehot_schedule._Neumaier).
    beta_m = u + (1 + u) * (u + (1 + u) * gamma(contributions - 1) ** 2)
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
    blocks = sum(count for _s, count in r.onehot.projector_blocks)
    assert (blocks, r.physical_phase) == (8, -1000000000000008.0)
    assert max(components, key=components.get) == "projector"
    assert components["projector"] > components["constant"] + components["final"]
    assert ("comes from the projector-identity contributions with the projector blocks' own phase "
            "assignments") in message
    assert "removing it from the objective" not in message
    assert components["projector"] <= reported < components["projector"] + components["final"]

    wide = Optimization(objective=sp.Integer(10) ** 32 * x**2 + 15 * sp.Integer(10) ** 15, variables=(x,),
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
    constant contributes ``S_C = 2e17``, and the ledger of its 8 compensated
    additions gets 88.8 rad (133 rad on the native route), so a kept
    state is rejected while the same Plan without ``keep_state`` solves. The
    allowance in the rejection message must equal the owners' value for the
    8 additions that planning records, so the test also pins the
    contribution count and the route's rule, and the message must name the
    constant's component and advise removing the constant. With ``c = 3/8``
    the allowance is far below pi and the kept state is admitted. The
    objective ``1e17 x + 1e10`` has projector contributions whose signed
    total cancels but whose absolute sum drives the allowance, so its
    rejection names the projector component and gives no advice about the
    constant.

    Natively, ``-1e16 x**2 + 3e15`` over two unit steps has M = 8
    compensated additions, B = 8 projector blocks and
    ``physical_phase = -1e15 - 8``, and its allowance of about 6.2 rad splits
    into the additive components ``L_C``, ``L_K``, ``L_P + 2u Y + 20u B``
    and the final assignment ``2u |physical_phase| + 20u``
    (``method._native_components``), with
    ``L_s = [eta_s + beta' (1 + eta_s)] S_s`` and
    ``beta' = u + (1 + u) beta_M``. The test evaluates them in rational
    arithmetic from the Plan's stored inputs. The projector component, about
    3.33 rad, is the largest, ahead of the constant's 2.66 rad, so the
    message must name it, report an upper value of it below the projector
    and final components together, and give no advice about the constant.
    For ``1e32 x**2 + 1.5e16`` on (-2e-8, 2e-8) with total time 0.5 the
    final assignment of the physical phase, about 3.89 rad, is the largest
    component, ahead of the others at about 3.33 rad, and the message must
    name it rather than a Hamiltonian source.
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
    # N (1 + [c != 0] + F tiles): two steps, one tile of the single table in each of two potential factors.
    contributions = 2 * (1 + 1 + 2 * len(r.support_values))
    assert contributions == owner._phase_contributions(r, probabilities.method) == 8
    expected = owner._compiled_phase_allowance(r, probabilities.method, grid, contributions)
    if execution == "quantum":
        expected = owner._native_phase_allowance(r, expected)
    reported = float(re.search(r"rounding allowance (\S+) rad", str(rejected.value)).group(1))
    assert reported == expected > 88.8
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


def test_compensated_ledger_sum_lies_within_its_bound_of_the_exact_sum(monkeypatch):
    """The ledger's sums lie within their bounds of the exact sums of the phase events.

    The test records every value that the running total adds with Neumaier's
    compensation while planning a c = 1e16, 16-step Plan, the kinetic and
    constant events and one ``math.fsum`` per tile of projector identities
    (``onehot_schedule._record_partial``). Their count is the ledger's M,
    and the stored total lies within ``beta_M A`` of their exact sum,
    ``beta_M = u + (1 + u) gamma_(M-1)**2``. The stored total also lies within
    ``beta' A`` of the exact sum of the individual events
    ``-fl(t fl(fl(b v) 2**-s))``, ``-fl(dt fl(b c))`` and the kinetic
    events, formed here independently, with ``beta' = u + (1 + u) beta_M``.
    The constant's own total lies within ``[eta_2 + (1 + eta_2) beta_N] S_C``
    of its exact angle ``-sum_k dt b_k c`` over the stored inputs.
    """
    from fractions import Fraction

    from nwqlib.algorithms.qhd import onehot_schedule

    recorded = []
    original = onehot_schedule._record_partial

    def recording(sources, totals, name, value):
        recorded.append(value)
        return original(sources, totals, name, value)

    monkeypatch.setattr(onehot_schedule, "_record_partial", recording)
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=x**2 / 3 + x / 7 + sp.Integer(10) ** 16, variables=(x,),
                           bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=4, num_steps=16, total_time=1.0, schedule=QuadraticSchedule(gamma=0.3),
                 theory_flavor="ir_product")
    selected = plan(problem, method=method, execution="classical", seed=7)
    r, grid = selected.reconstruction, owner._grid(selected)
    u, m, n = Fraction(1, 2**53), len(recorded), method.num_steps
    # The ledger's count M, which the allowances take from the reconstruction, is the number of added values.
    assert owner._phase_contributions(r, method) == m

    def gamma(k):
        return k * u / (1 - k * u)

    def beta(k):
        return u + (1 + u) * gamma(k - 1) ** 2

    exact = sum(Fraction(v) for v in recorded)
    absolute = sum(abs(Fraction(v)) for v in recorded)
    assert abs(Fraction(r.physical_phase) - exact) <= beta(m) * absolute
    # The individual events, in binary64 as the schedule forms them.
    dt = method.total_time / method.num_steps
    (table,) = r.support_values
    events = []
    for _t, a, b in r.step_weights:
        events.append(-dt * (a / grid.spacing(0) ** 2))
        events.append(-dt * (b * r.constant))
        for duration in (dt / 2, dt / 2):
            events.extend(-duration * ((b * v) * 0.5) for v in table.values.array.tolist())
    exact_events = sum(Fraction(v) for v in events)
    assert abs(Fraction(r.physical_phase) - exact_events) <= (u + (1 + u) * beta(m)) * sum(
        abs(Fraction(v)) for v in events)
    step = Fraction(dt)
    s_c = abs(Fraction(r.constant)) * sum(abs(step * Fraction(b)) for _t, _a, b in r.step_weights)
    constant = -sum(step * Fraction(b) * Fraction(r.constant) for _t, _a, b in r.step_weights)
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
    from nwqlib.algorithms.qhd.onehot_schedule import _Neumaier, _record

    totals, sources = _Neumaier(), {}
    for angle in (1.0, 1e100, 1.0, -1e100):
        _record(sources, totals, "projector_identity", -angle, 1.0)
    assert totals.value("total") == sources["projector_identity"] == 2.0


def test_compiled_phase_allowances_are_evaluated_upward():
    """The ledger and native allowances are upper values of their exact expressions.

    The test evaluates the expressions of ``method._compiled_phase_allowance``
    and ``method._native_phase_allowance`` in rational arithmetic from the
    Plan's stored binary64 inputs, ``gamma_r = r u/(1 - r u)``,
    ``eta_2 = 2u + u**2``, ``beta_d = u + (1 + u) gamma_(d-1)**2``,
    ``eta_K = (3u + u**2 + (1 + 2u + u**2) beta_d)/(1 - u)``,
    ``E_ledger = D + beta' (S_C + S_P + S_K + D)`` with
    ``beta' = u + (1 + u) beta_M``, ``beta_M = u + (1 + u) gamma_(M-1)**2``, and
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
    contributions = steps * (2 + 2 * sum(-(-len(t.values) // ENTRY_TILE) for t in r.support_values))
    beta_m = u + (1 + u) * (u + (1 + u) * gamma(contributions - 1) ** 2)
    exact_ledger = formation + beta_m * (s_c + s_p + s_k + formation)
    projectors = [b for step in range(steps) for b in step_blocks(selected, step) if b.kind == "number_projector"]
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


@pytest.mark.parametrize("execution,schedule,rule", [
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
    monkeypatch.setattr(OneHotSchedule, "summary", forbid)
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
    # The loaded one-hot Plan derives the same circuit from its recorded data without evaluating the objective.
    from nwqlib.algorithms.qhd.native import construct_qhd

    def gates(circuit):
        return [(item.operation.name, [circuit.find_bit(q).index for q in item.qubits], item.operation.params)
                for item in circuit.data], circuit.global_phase

    assert gates(construct_qhd(restored.blocks[0], (), None)) == gates(construct_qhd(selected.blocks[0], (), None))


def test_content_hashes_read_support_tables_by_digest(tmp_path, monkeypatch):
    """Planning and the first content hashes encode no support table.

    The tables enter their records' content hashes by digest
    (``core.records.FrozenArray``).
    """
    from nwqlib.core import records

    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=x * y, variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    method = QHD(num_grid_points=16, num_steps=1, total_time=0.25)
    with monkeypatch.context() as patch:
        patch.setattr(records.base64, "b64encode", lambda *a: pytest.fail("a content hash encoded a table"))
        selected = plan(problem, method=method, execution="classical", seed=31)
        (table,) = selected.reconstruction.support_values
        identities = (table.content_id, selected.reconstruction.content_id, selected.content_id)
    restored = load_plan(save_plan(selected, ArchiveFiles(tmp_path, 4_000_000)), ArchiveFiles(tmp_path, 4_000_000))
    (loaded,) = restored.reconstruction.support_values
    assert (loaded.content_id, restored.reconstruction.content_id, restored.content_id) == identities
    np.testing.assert_array_equal(loaded.values.array, table.values.array)
    changed = plan(problem, method=QHD(num_grid_points=16, num_steps=2, total_time=0.25), execution="classical",
                   seed=31)
    assert changed.content_id != selected.content_id


@pytest.mark.parametrize("order, threshold, objective", [
    (1, 0.0, "x**2/3 + x*y/7 + y/5"),
    (2, 2e-3, "x**2/3 + x*y/7 + y/5000"),
    (2, 0.0, "x + 1e-310*y + 2*x*y"),
], ids=["first_order", "second_order_pruned", "range_omission"])
def test_onehot_step_blocks_match_an_independent_scalar_derivation(order, threshold, objective):
    """Every block of every step equals the block of the scalar formation rules, bit for bit.

    Oracle, per step k with row ``(t_k, a_k, b_k)`` and ``dt = fl(T/N)``:
    the potential factors are ``(dt,)`` or ``(dt/2, dt/2)``, and each table
    entry v, in table order and C order, gives ``p = fl(b v)`` and
    ``zeta = fl(t p)``. It is kept when v is nonzero, ``p``, ``zeta``,
    ``fl(p/2**s)`` and ``fl(t fl(p/2**s))`` are at least ``2**-1022`` in
    magnitude and ``|zeta|`` is at least the threshold. Each link of
    variable j, in the layer order odd, even, odd for second order, gives
    ``c = fl(-a/fl(fl(4 h) h))`` and ``angle = fl(fl(2 t) c)``, kept when
    ``|angle|`` is at least the threshold. These supports of at most two
    variables use MCPhase, which has no wrapped-phase rule. The Plan's
    dropped and omitted counts, its kept counts and its CX law must agree
    with the oracle's blocks.
    """
    from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import structured_number_projector_provider

    x, y = sp.symbols("x y", real=True)
    k, steps, total = 3, 3, 0.7
    problem = Optimization(objective=sp.sympify(objective, locals=dict(x=x, y=y)), variables=(x, y),
                           bounds=((0.0, 1.0), (0.0, 1.0)))
    selected = make_plan(problem, QHD(num_grid_points=k, num_steps=steps, total_time=total, trotter_order=order,
                                      rotation_threshold=threshold, schedule=QuadraticSchedule(gamma=0.3),
                                      initial_state_preparation="none"))
    r, grid = selected.reconstruction, owner._grid(selected)
    dt, tiny = total / steps, 2.0**-1022
    links = grid.links()
    layers = (((links[1::2], dt / 2), (links[0::2], dt), (links[1::2], dt / 2)) if order == 2 else ((links, dt),))
    dropped = omitted = cx = 0
    for step, (time, a, b) in enumerate(r.step_weights):
        expected = []

        def potential(duration):
            nonlocal dropped, omitted, cx
            for table in r.support_values:
                s = len(table.support)
                for position, v in enumerate(table.values.array.tolist()):
                    if v == 0.0:
                        continue
                    p = b * v
                    zeta = duration * p
                    q = p * (1.0 / 2**s)
                    if min(abs(p), abs(zeta), abs(q), abs(duration * q)) < tiny:
                        omitted += 1
                    elif abs(zeta) < threshold:
                        dropped += 1
                    else:
                        indices = np.unravel_index(position, (k,) * s)
                        expected.append(("number_projector", tuple(j * k + int(i) for j, i in zip(
                            table.support, indices, strict=True)), p, zeta, duration, time))
                        cx += structured_number_projector_provider(s)["cx"]

        potential(dt if order == 1 else dt / 2)
        for j in range(grid.num_variables):
            h = grid.spacing(j)
            c = -a / (4.0 * h * h)
            for layer, duration in layers:
                angle = (2.0 * duration) * c
                for left, right in layer:
                    if abs(angle) < threshold:
                        dropped += 1
                        continue
                    expected.append(("kinetic", (j * k + left, j * k + right), c, angle, duration, time))
                    cx += 2
        if order == 2:
            potential(dt / 2)
        derived = [(block.kind, block.support, block.coefficient, block.angle, block.time_step, block.time)
                   for block in step_blocks(selected, step)]
        assert derived == expected
    assert (r.dropped_count, r.range_omissions.projectors) == (dropped, omitted)
    assert (dropped > 0) == (threshold > 0) and (omitted > 0) == ("1e-310" in objective)
    assert {law.metric: law.value for law in selected.construction.selections[0].resource_laws}["cx"] == cx
    assert [p.support_size for p in r.onehot.providers] == sorted({len(t.support) for t in r.support_values})
    classical = make_plan(problem, QHD(num_grid_points=k, num_steps=steps, total_time=total), execution="classical")
    with pytest.raises(ValueError, match="compiled schedule"):
        step_blocks(classical, 0)


def test_onehot_plan_records_do_not_grow_with_steps_or_entries():
    """A one-hot Plan holds a number of records that does not depend on its step count or table sizes.

    The Plan stores its tables, step rows and the schedule record, not one
    record per block, so planning the same problem with 2 steps on 3 points
    and with 40 steps on 6 points gives Plans with the same number of
    ``Record`` instances, counted through every field.
    """
    from nwqlib.core.records import Record

    def records(value):
        if isinstance(value, Record):
            return 1 + sum(records(getattr(value, name)) for name in type(value).model_fields)
        if isinstance(value, (tuple, list)):
            return sum(records(item) for item in value)
        return 0

    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=x**2 / 3 + x * y / 7 + y / 5, variables=(x, y),
                           bounds=((-1.0, 1.0), (-1.0, 1.0)))
    counts = {records(make_plan(problem, QHD(num_grid_points=k, num_steps=n, total_time=0.7)))
              for k, n in ((3, 2), (6, 40))}
    assert len(counts) == 1


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
    # count the bytes of the fixed slab of b = min(D, 2**16) = 3 points,
    # 8dK + 8D + 65536 + 256 (N + 2d + 16) + b (8 (N + 2) + 8d + 96)
    # = 24 + 24 + 65536 + 4864 + 3*128 = 70832.
    checks = QHDVerification(comparisons=("grid_minimum",), max_work=1, max_bytes=216)
    with pytest.raises(ValueError, match="requires at least 42 work units and at least 70832 bytes"):
        result.verify(checks=checks)


def test_support_tables_are_evaluated_in_fixed_blocks_that_the_byte_limit_only_admits(monkeypatch):
    """A 32 x 32 support table is evaluated in one block of 1024 entries, at the default limit and at the least limit that admits it.

    NumPy's array and scalar evaluations can differ in their last bits, so the
    block is fixed at min(n, 2**16) entries (compiler.support_chunk_size). The
    thirty cosines give the table's evaluator the largest byte term of the
    one-hot table stage. A limit one byte below that stage's stated
    requirement refuses before any evaluation, instead of evaluating smaller
    blocks, and the stated limit gives the same table as the default. With
    the block lowered to 256 entries, the table is evaluated in four blocks at
    both limits and its stated requirement is smaller.
    """
    import re
    from nwqlib.algorithms.qhd import compiler

    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=sum(sp.cos(j * x * y) for j in range(1, 31)), variables=(x, y),
                           bounds=((-1.0, 1.0), (-1.0, 1.0)))
    base = QHD(num_grid_points=32, num_steps=1, total_time=0.5)
    seen, original = [], compiler.chunk_coordinates

    def observe(axes, start, stop):
        seen.append(stop - start)
        return original(axes, start, stop)

    monkeypatch.setattr(compiler, "chunk_coordinates", observe)

    def tables(**limit):
        chosen = plan(problem, method=base.revise(**limit), execution="quantum", seed=17)
        return [t.values.array for t in chosen.reconstruction.support_values]

    with pytest.raises(ValueError, match="symbolic tables and one-hot compilation") as refused:
        tables(max_bytes=10**5)
    need = int(re.search(r"and (\d+) bytes", str(refused.value)).group(1))
    with pytest.raises(ValueError, match=f"and {need} bytes"):
        tables(max_bytes=need - 1)
    assert seen == []
    admitted = tables(max_bytes=need)
    assert seen == [1024]
    assert all(np.array_equal(a, b) for a, b in zip(admitted, tables(), strict=True)) and seen == [1024, 1024]

    # A block below the table size splits the table into fixed blocks, and the
    # stated requirement falls with the block's evaluator term.
    monkeypatch.setattr(compiler, "SUPPORT_BLOCK_ENTRIES", 256)
    seen.clear()
    with pytest.raises(ValueError, match="symbolic tables and one-hot compilation") as refused:
        tables(max_bytes=10**5)
    smaller = int(re.search(r"and (\d+) bytes", str(refused.value)).group(1))
    assert smaller < need
    blocks = tables(max_bytes=smaller)
    assert seen == [256] * 4
    assert all(np.array_equal(a, b) for a, b in zip(blocks, tables(), strict=True)) and seen == [256] * 8


def test_a_native_register_is_refused_when_a_saved_record_cannot_hold_its_basis():
    """2**14284 < 10**4300 < 2**14285, so a register of 14,284 qubits passes and 14,285 is refused.

    A width of 10**12 is refused from its bit length, before 2**width, about 125 GB, is formed.
    """
    from nwqlib.algorithms.qhd.method import _admit_register_basis

    _admit_register_basis(14284)
    for width in (14285, 10**12):
        with pytest.raises(ValueError, match="a saved record holds integers below 10\\*\\*4300"):
            _admit_register_basis(width)


def test_known_state_sizes_are_refused_before_the_tables(monkeypatch):
    """K, d and the step count fix the classical arrays and a kept state, so no table is evaluated before they fit.

    On two variables with ``max_bytes`` = 20 MiB, K = 512 needs at least 26087440 bytes for the split-step arrays
    and the readout (``method._known_host_bytes``, ``64 D + 33 D`` and the smaller terms), and K = 128 needs 2860560.
    A kept state of the 64-qubit one-hot register, K = 32, needs ``16 * 2**64`` bytes.
    """
    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=sp.cos(x) * y + x**2, variables=(x, y), bounds=((-1.0, 1.0),) * 2)
    tables = []
    original = QHDCompiler._potential_grid_values
    monkeypatch.setattr(QHDCompiler, "_potential_grid_values",
                        lambda self: tables.append(1) or original(self))
    limited = QHD(num_grid_points=512, num_steps=2, total_time=0.1, theory_flavor="split_step", max_bytes=20 * 2**20)
    with pytest.raises(ValueError, match="classical host evolution and readout requires at least"):
        plan(problem, method=limited, execution="classical", seed=1)
    assert tables == []
    plan(problem, method=limited.revise(num_grid_points=128), execution="classical", seed=1)
    assert tables
    tables.clear()
    with pytest.raises(ValueError, match="kept state requires"):
        plan(problem, method=QHD(num_grid_points=32, num_steps=1, keep_state=True), execution="quantum", seed=1)
    assert tables == []


@pytest.mark.parametrize("variables, objective, bounds, analytic, route", [
    # The split factors exp(-320) exp(80 x) exp(-5 x**2) overflow to inf at a grid point near 8.9.
    (1, lambda x: -sp.exp(-5 * (x - 8) ** 2), ((0.0, 10.0),), lambda x: -np.exp(-5 * (x - 8) ** 2), "plan"),
    # A box away from the origin evaluates the tables in the centered coordinate x - 8.
    (1, lambda x: -sp.exp(-5 * (x - 9) ** 2), ((8.0, 10.0),), lambda x: -np.exp(-5 * (x - 9) ** 2), "plan"),
    # The monomials keep their own expansion beside the Gaussian, in one support group.
    (1, lambda x: x**2 / 50 - sp.exp(-5 * (x - 8) ** 2), ((0.0, 10.0),),
     lambda x: x**2 / 50 - np.exp(-5 * (x - 8) ** 2), "plan"),
    # One exponential of two variables is one table on their joint support.
    (2, lambda x, y: -sp.exp(-5 * (x - 8) ** 2 - 5 * (y - 2) ** 2), ((0.0, 10.0),) * 2,
     lambda x, y: -np.exp(-5 * (x - 8) ** 2 - 5 * (y - 2) ** 2), "plan"),
    # Float centers. An expanded exponent pulls out the constant factor exp(-900.0), below the binary64 range.
    (2, lambda x, y: -sp.exp(-50 * (x - 3.0) ** 2) * sp.exp(-50 * (y + 3.0) ** 2), ((-5.0, 5.0),) * 2,
     lambda x, y: -np.exp(-50 * (x - 3.0) ** 2) * np.exp(-50 * (y + 3.0) ** 2), "plan"),
    # The constant factor would be exp(-744.8), a subnormal number that keeps few bits.
    (2, lambda x, y: -sp.exp(-46.5 * (x - 2.83) ** 2) * sp.exp(-46.5 * (y + 2.83) ** 2), ((-5.0, 5.0),) * 2,
     lambda x, y: -np.exp(-46.5 * (x - 2.83) ** 2) * np.exp(-46.5 * (y + 2.83) ** 2), "plan"),
    # One variable with c a**2 = 800, where the split exponential overflows beside a zero factor.
    (1, lambda x: -sp.exp(-50 * (x - 4.0) ** 2), ((-5.0, 5.0),), lambda x: -np.exp(-50 * (x - 4.0) ** 2), "plan"),
    # A polynomial next to the well, whose monomials expand while the well stays whole.
    (2, lambda x, y: 0.01 * (x**2 + y**2) - sp.exp(-50 * (x - 3.0) ** 2) * sp.exp(-50 * (y + 3.0) ** 2),
     ((-5.0, 5.0),) * 2,
     lambda x, y: 0.01 * (x**2 + y**2) - np.exp(-50 * (x - 3.0) ** 2) * np.exp(-50 * (y + 3.0) ** 2), "plan"),
    # The default search model of refine_box, whose exponent -12 (10 u - 8.0)**2 on the unit box has the expanded
    # constant term -768.0, past the normal binary64 range although the function is finite.
    (1, lambda x: -sp.exp(-12 * (x - 3.0) ** 2), ((-5.0, 5.0),), lambda x: -np.exp(-12 * (x - 3.0) ** 2),
     "search_model"),
    # Exponents written out with a Float constant. SymPy pulls the constants out and holds their product, about
    # -1.36e-391, a value that reads as zero in binary64.
    (2, lambda x, y: -sp.exp(-50 * x**2 + 300.0 * x - 450.0) * sp.exp(-50 * y**2 - 300.0 * y - 450.0), ((-5.0, 5.0),) * 2,
     lambda x, y: -np.exp(-50 * (x - 3.0) ** 2) * np.exp(-50 * (y + 3.0) ** 2), "plan"),
    # The same with c = 46.5, a constant about -3.35e-324 that reads as a subnormal changed by 47%.
    (2, lambda x, y: -sp.exp(-46.5 * x**2 + 263.19 * x - 372.413850) * sp.exp(-46.5 * y**2 - 263.19 * y - 372.413850),
     ((-5.0, 5.0),) * 2, lambda x, y: -np.exp(-46.5 * (x - 2.83) ** 2) * np.exp(-46.5 * (y + 2.83) ** 2), "plan"),
    # A prefactor about 7.33e390 that reads as infinity beside decaying factors, on a box where the values stay ordinary.
    (2, lambda x, y: sp.exp(50 * x**2 - 300.0 * x + 450.0) * sp.exp(50 * y**2 + 300.0 * y + 450.0),
     ((2.9, 3.1), (-3.1, -2.9)), lambda x, y: np.exp(50 * (x - 3.0) ** 2 + 50 * (y + 3.0) ** 2), "plan"),
    # A variable factor stays outside the combined exponential, and on a sign-changing coordinate it carries its sign.
    (2, lambda x, y: y * sp.exp(-50 * x**2 + 300.0 * x - 450.0) * sp.exp(-50 * y**2 - 300.0 * y - 450.0),
     ((-5.0, 5.0),) * 2, lambda x, y: y * np.exp(-50 * (x - 3.0) ** 2) * np.exp(-50 * (y + 3.0) ** 2), "plan"),
    # Two constants in range whose product is not, with no atom that a scan of the literals would flag.
    (2, lambda x, y: sp.Mul(sp.Float("1e-200"), sp.Float("1e-200"), sp.exp(x + 450, evaluate=False),
                            sp.exp(y + 450, evaluate=False), evaluate=False), ((-1.0, 1.0),) * 2,
     lambda x, y: (1e-200 * np.exp(x + 450)) * (1e-200 * np.exp(y + 450)), "plan"),
    # A literal that reads as zero, with a second constant that brings the combined value about 2.35e-307 back into the
    # normal range. The values are ordinary, and the oracle squares a form whose factors stay in range.
    (1, lambda x: sp.Float("1e-324", 15) * sp.exp(40) * sp.exp(x), ((700.0, 705.0),),
     lambda x: (1e-162 * np.exp(20 + x / 2)) ** 2, "plan"),
    # Positive constant powers beside a constant that reads as zero. The oracle is one power, with no such constant.
    (2, lambda x, y: sp.Float("1e-400", 15) * 10 ** (x**2) * 10 ** (y**2), ((-15.0, 15.0),) * 2,
     lambda x, y: 10.0 ** (x**2 + y**2 - 400.0), "plan"),
    # A lone power whose base reads as zero, with ordinary values.
    (1, lambda x: sp.Float("1e-400", 50) ** (x / 400), ((-1.0, 1.0),), lambda x: 10.0 ** (-x), "plan"),
    # An exact base 1 + 1e-60 in a combined product. Its logarithm has to keep the distance from 1, and the expansion has to
    # keep the power whole, since p**g*q**(-g) cancels only in binary64.
    (1, lambda x: sp.Float("1e-400", 15) * sp.Rational(10**60 + 1, 10**60) ** (10**60 * x) * sp.exp(921), ((-1.0, 1.0),),
     lambda x: np.exp(x) * (1e-200 * np.exp(460.5)) ** 2, "plan"),
    # An exact compound base closer to 1 than the working digits, on either side. With t = exp(-100), the exponent of the
    # first row is x (log(1 + t)/t - 1/2), and log(1 + t)/t lies in (1 - t/2, 1), so its table is exp(x/2) in binary64.
    (1, lambda x: sp.Mul(sp.Pow(1 + sp.exp(-100), sp.exp(100) * x, evaluate=False), sp.exp(-x / 2), evaluate=False),
     ((-1.0, 1.0),), lambda x: np.exp(x / 2), "plan"),
    (1, lambda x: sp.Mul(sp.Pow(1 - sp.exp(-100), sp.exp(100) * x, evaluate=False), sp.exp(3 * x / 2), evaluate=False),
     ((-1.0, 1.0),), lambda x: np.exp(x / 2), "plan"),
    # A Rational base stays whole, and its exponent in monomials still expands about the far box, as with base 1.5. The
    # table route stops before grid_minimum, which evaluates those monomials as written.
    (1, lambda x: sp.Rational(3, 2) ** (x**2 - 200000 * x + 10**10), ((99999.0, 100001.0),),
     lambda x: 1.5 ** ((x - 100000.0) ** 2), "table"),
    # Centering forms exp(-800.0) beside sin, whose factor stays outside the exponential. The term is negligible on this box.
    (1, lambda x: 0.001 * x**2 + sp.sin(x) * sp.exp(-2.0 * x), ((400.0, 410.0),),
     lambda x: 0.001 * x**2 + np.sin(x) * np.exp(-2.0 * x), "plan"),
    # The same term in a refinement whose box reaches, from a level where the constant is normal, a level where it is not.
    (1, lambda x: 0.01 * (x - 405.0) ** 2 + sp.sin(x) * sp.exp(-2.0 * x), ((350.0, 500.0),),
     lambda x: 0.01 * (x - 405.0) ** 2 + np.sin(x) * np.exp(-2.0 * x), "moving_box"),
    # A written-out product with sin(x) in an exponent, and the same objective with squares, which holds no such constant.
    (2, lambda x, y: -sp.exp(-50 * x**2 + 300.0 * x - 450.0) * sp.exp(sp.sin(x) - 50 * y**2 - 300.0 * y - 450.0), ((-5.0, 5.0),) * 2,
     lambda x, y: -np.exp(-50 * (x - 3.0) ** 2) * np.exp(np.sin(x) - 50 * (y + 3.0) ** 2), "plan"),
    (2, lambda x, y: -sp.exp(-50 * (x - 3.0) ** 2) * sp.exp(sp.sin(x) - 50 * (y + 3.0) ** 2), ((-5.0, 5.0),) * 2,
     lambda x, y: -np.exp(-50 * (x - 3.0) ** 2) * np.exp(np.sin(x) - 50 * (y + 3.0) ** 2), "plan"),
    # No constant at all. The decaying factor underflows before its partners compensate, with true values from 0.3 to 3.
    (3, lambda x, y, z: sp.exp(-800 * x) * sp.exp(400 * y) * sp.exp(400 * z), ((0.999, 1.001),) * 3,
     lambda x, y, z: np.exp(-800 * x + 400 * y + 400 * z), "table"),
    # Two factors beside a normal constant. The first underflows, and the values are ordinary.
    (1, lambda x: 1e20 * sp.exp(-746 * x**2) * sp.exp(700 * x**2 + x), ((0.99, 1.0),),
     lambda x: 1e20 * np.exp(-46 * x**2 + x), "table"),
    # Square-root exponents, real at every coordinate, with values near 1.
    (3, lambda x, y, z: sp.exp(-800 * sp.sqrt(1 + x**2)) * sp.exp(400 * sp.sqrt(1 + y**2)) * sp.exp(400 * sp.sqrt(1 + z**2)),
     ((-0.01, 0.01),) * 3, lambda x, y, z: np.exp(-800 * np.sqrt(1 + x**2) + 400 * np.sqrt(1 + y**2) + 400 * np.sqrt(1 + z**2)),
     "table"),
    # Log densities in real variables. The exponential is positive, so SymPy writes the logarithm of the product as
    # a sum, the supports separate and no exponential underflows before its logarithm.
    (1, lambda x: sp.log(x**2 * sp.exp(-50 * x**2)), ((0.1, 5.0),), lambda x: np.log(x**2) - 50 * x**2, "log_density"),
    (2, lambda x, y: sp.log(y * sp.exp(-50 * (x - 3.0) ** 2)), ((-5.0, 5.0), (0.5, 3.0)),
     lambda x, y: -50 * (x - 3.0) ** 2 + np.log(y), "log_density"),
    (2, lambda x, y: sp.log(sp.exp(-50 * (x - 3.0) ** 2) / y), ((-5.0, 5.0), (0.5, 3.0)),
     lambda x, y: -50 * (x - 3.0) ** 2 - np.log(y), "log_density"),
])
def test_a_resolved_gaussian_well_is_tabulated_at_its_analytic_values(variables, objective, bounds, analytic, route):
    """The stored tables, added on the actual grid, equal the original function, and the solve finds its grid minimum.

    The oracle is NumPy's evaluation of the analytic function at the grid coordinates, on log densities in the log
    form that cannot underflow. The tolerance 1e-11 is a witness-specific empirical regression allowance that predates
    these rows. The arithmetic here is a retrospective rationale, not a preselected error derivation or a bound. An
    absolute error of an exponent q becomes a first-order relative error of ``exp(-q)``. Direct squaring of
    ``q = a (x - b)**2`` gives at most ``gamma4 q``, with ``gamma_n = n u/(1 - n u)`` in the ordinary rounding model,
    and the errors of the two compared paths add. For raw values within delta of each other, normalization gives at
    most ``4 |kappa| delta / min(RF, RG)`` with their grid ranges RF and RG, so the gain kappa alone misses the offset
    and the range. The log-density rows reach values near 3000 and have the least room under this allowance.

    On the rows with a constant combined into one exponential E, ``|E_hat - E|/|E| <= expm1(eps_q) + eps_exp exp(eps_q)``.
    With exponent terms near 900 and about ten rounded operations, eps_q is about ``10 * 900 * u``, or 1e-12, and the
    values are at most about 3, which puts this model's error near 3e-12, under the allowance.

    The result's objective is the best observed candidate, which exact probabilities place at the grid minimum. A
    search-model level stores ``kappa (F(a + D u) - c)/E`` in its own unit coordinates. With one support table c is the
    grid minimum of F and E its grid range, up to one outward rounding of E, so the oracle normalizes the NumPy values
    of F at the mapped coordinates the same way, with the gain kappa that the level records. The "moving_box" route does
    so at each level with the shift and scale it records, and the "table" route stops at the table, since the classical
    evolution of so narrow a box is slow, 27 s at K = 6 and width 0.01.
    """
    symbols = sp.symbols(f"x0:{variables}", real=True)
    problem = Optimization(objective=objective(*symbols), variables=symbols, bounds=bounds)
    method = QHD(num_grid_points={1: 128, 2: 32, 3: 6}[variables], num_steps=2, total_time=0.5,
                 initial_state=UniformState())
    if route == "moving_box":
        run = refine_box(problem, qhd=method, options=BoxRefinement(max_levels=3), execution="classical", seed=3,
                         progress=False)
        assert run.levels[-1].box[0][0] > 354.5  # exp(-2.0*354.5) is below the normal binary64 range.
        for level, inner in zip(run.levels, run.results, strict=True):
            values, (unit,) = _tabulated_values(inner.plan)
            (lower, upper), = level.box
            expected = level.potential_gain * (analytic(lower + (upper - lower) * unit) - level.energy_shift) / level.energy_scale
            np.testing.assert_allclose(values, expected, rtol=0, atol=1e-11)
        return
    if route == "search_model":
        run = refine_box(problem, qhd=method, options=BoxRefinement(max_levels=1), execution="classical", seed=3,
                         progress=False)
        values, (unit,) = _tabulated_values(run.results[0].plan)
        (lower, upper), = bounds
        original = analytic(lower + (upper - lower) * unit)
        expected = run.levels[0].potential_gain * (original - original.min()) / (original.max() - original.min())
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-11)
        return
    selected = plan(problem, method=method, execution="classical", seed=3)
    values, mesh = _tabulated_values(selected)
    expected = analytic(*mesh)
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-11)
    if route == "table":
        return
    result = solve(selected)
    assert abs(result.objective - expected.min()) <= 1e-11
    if route == "log_density":
        # The identity splits the supports, K entries per variable. grid_minimum evaluates the expression as the user
        # wrote it, whose exponential underflows before its logarithm, so verification refuses this objective.
        assert all(len(table.support) == 1 for table in selected.reconstruction.support_values)
        return
    _, facts = result.verify(checks=QHDVerification(comparisons=("grid_minimum",)))
    assert next(item.fact.value.value for item in facts if item.fact.quantity == "grid_minimum") == 0.0


@pytest.mark.parametrize("factors, variables, point, expected", [
    (lambda x, y: (sp.exp(1 / x, evaluate=False), sp.exp(-1 / x, evaluate=False)), 1, (0.0,), np.nan),
    (lambda x, y: (sp.exp(sp.log(x), evaluate=False), sp.exp(-sp.log(x), evaluate=False)), 1, (0.0,), np.nan),
    (lambda x, y: (sp.exp(sp.log(x), evaluate=False), sp.exp(-sp.log(x), evaluate=False)), 1, (-1.0,), np.nan),
    (lambda x, y: (sp.exp(sp.sqrt(x), evaluate=False), sp.exp(y, evaluate=False)), 2, (-1.0, 0.0), np.nan),
    (lambda x, y: (sp.exp(sp.sign(x), evaluate=False), sp.exp(sp.floor(y), evaluate=False)), 2, (0.0, 1.0), np.e),
    (lambda x, y: (sp.exp(sp.I * x, evaluate=False), sp.exp(y, evaluate=False)), 2, (1.0, 0.0), np.cos(1.0) + 1j * np.sin(1.0)),
    (lambda x, y: (sp.Mul(0, sp.Pow(x, -1, evaluate=False), evaluate=False), sp.exp(x), sp.exp(y)), 2, (0.0, 1.0), np.nan),
    (lambda x, y: (sp.Mul(sp.exp(400 * y), sp.exp(400 * y), evaluate=False), sp.exp(-800 * x)), 2, (1.0, 1.0), 1.0),
    (lambda x, y: (sp.exp(sp.exp(-800*x)*sp.exp(800*y)), sp.sin(sp.exp(-800*x)*sp.exp(800*y)), 2**(sp.exp(-800*x)*sp.exp(800*y))), 2, (1.0, 1.0), 2*np.e*np.sin(1.0)),
    (lambda x, y: (sp.sqrt(2), sp.exp(x), sp.exp(y)), 2, (0.0, 0.0), np.sqrt(2.0)),
    (lambda x, y: ((-2.0)**x, sp.exp(x), sp.exp(y)), 2, (0.5, 0.0), np.nan),
    (lambda x, y: (sp.Add(sp.Pow(x, -1, evaluate=False), -sp.Pow(x, -1, evaluate=False), sp.exp(x)*sp.exp(y), evaluate=False),), 2, (0.0, 1.0), np.nan),
])
def test_a_combined_exponent_keeps_the_domain_and_branch_of_its_operands(factors, variables, point, expected):
    """Each exponent is evaluated as written and its value then added, and the other factors stay as they are.

    The product is held so that SymPy cannot simplify it first. At these points the exact answer is known by hand: an
    undefined received expression stays NaN, for which the finite-real check of the caller refuses the point, a real-array
    ``log`` or ``sqrt`` stays NaN with no cast to complex, and the discontinuous and imaginary exponents keep their values.
    A symbolic sum of the exponents would cancel ``1/x - 1/x`` to 0 and return 1, which every table row would still pass.
    The eighth row is a nested product, which planning flattens and the evaluator meets as supplied in ``grid_minimum``. It
    is one group, with value 1, where combining the inner pair first would overflow before the outer factor decays.

    The last four rows hold the rest of the rule. A product inside an exponent, inside ``v`` or inside a power's exponent
    forms its own group, each with value 1 at this point. ``sqrt(2)`` is a constant, whose logarithm enters once. A negative
    base is a factor of ``v``, so ``(-2.0)**0.5`` stays NaN where a logarithm of the base would give a complex value. A sum
    around a combined product keeps its unevaluated ``1/x - 1/x``, so the pole stays NaN.
    """
    symbols = sp.symbols("x y")
    product = sp.Mul(*factors(*symbols), evaluate=False)
    with np.errstate(all="ignore"):
        value = _lambdify_objective(symbols[:variables], product)(*map(np.array, point))
    np.testing.assert_allclose(value, expected, rtol=4e-16, atol=0)


# Value and branch of each special function that an objective can name, against mpmath at 30 digits, a separate
# implementation. Each row is the SymPy function, its mpmath reference and a box on which SymPy's value is real.
# Rows for both signs of the argument where the function is real there.
SPECIAL_FUNCTIONS = {
    "besselj": (lambda x: sp.besselj(0, x), lambda v: mpmath.besselj(0, v), (-5.0, 5.0)),
    "besselj half order": (lambda x: sp.besselj(sp.Rational(1, 2), x), lambda v: mpmath.besselj(0.5, v), (0.5, 5.0)),
    "bessely": (lambda x: sp.bessely(1, x), lambda v: mpmath.bessely(1, v), (0.5, 5.0)),
    "besseli": (lambda x: sp.besseli(2, x), lambda v: mpmath.besseli(2, v), (-3.0, 3.0)),
    "besselk": (lambda x: sp.besselk(0, x), lambda v: mpmath.besselk(0, v), (0.5, 5.0)),
    "airyai": (sp.airyai, mpmath.airyai, (-5.0, 5.0)),
    "airybi": (sp.airybi, mpmath.airybi, (-5.0, 3.0)),
    "airyaiprime": (sp.airyaiprime, lambda v: mpmath.airyai(v, derivative=1), (-5.0, 5.0)),
    "airybiprime": (sp.airybiprime, lambda v: mpmath.airybi(v, derivative=1), (-5.0, 3.0)),
    "LambertW": (sp.LambertW, lambda v: mpmath.lambertw(v).real, (-0.3, 3.0)),
    "LambertW lower branch": (lambda x: sp.LambertW(x, -1), lambda v: mpmath.lambertw(v, -1).real, (-0.3, -0.05)),
    "Si": (sp.Si, mpmath.si, (-3.0, 3.0)),
    "Ci": (sp.Ci, mpmath.ci, (0.5, 5.0)),
    "Ei": (sp.Ei, mpmath.ei, (-3.0, 3.0)),
    "erfi": (sp.erfi, mpmath.erfi, (-1.0, 1.0)),
    "erfinv": (sp.erfinv, mpmath.erfinv, (-0.9, 0.9)),
    "erfcinv": (sp.erfcinv, lambda v: mpmath.erfinv(1 - v), (0.1, 1.9)),
    # SymPy writes digamma(x) as polygamma(0, x).
    "digamma": (sp.digamma, mpmath.digamma, (-0.9, -0.1)),
    "polygamma": (lambda x: sp.polygamma(1, x), lambda v: mpmath.polygamma(1, v), (0.5, 5.0)),
    "loggamma": (sp.loggamma, mpmath.loggamma, (0.5, 50.0)),
}


@pytest.mark.parametrize("name", list(SPECIAL_FUNCTIONS))
def test_a_special_function_is_tabulated_at_its_independent_value(name):
    """The table and the grid_minimum verification of an objective with a special function follow mpmath.

    The tolerance 1e-11, relative with the same absolute floor for values near a zero of the function, is empirical.
    No error bound of these SciPy functions is derived here, and with SciPy 1.18.1 the rows differ from mpmath by at
    most about 1e-14 in absolute value. The comparison covers these orders and boxes at finite grid points.
    """
    build, reference, bounds = SPECIAL_FUNCTIONS[name]
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=build(x), variables=(x,), bounds=(bounds,))
    selected = plan(problem, method=QHD(num_grid_points=16, num_steps=1, total_time=0.5, initial_state=UniformState()),
                    execution="classical", seed=3)
    values, (axis,) = _tabulated_values(selected)
    with mpmath.workdps(30):
        expected = np.array([float(reference(mpmath.mpf(float(v)))) for v in axis])
    assert values == pytest.approx(expected, rel=1e-11, abs=1e-11)
    _, facts = solve(selected).verify(checks=QHDVerification(comparisons=("grid_minimum",)))
    assert next(item.fact.value.value for item in facts if item.fact.quantity == "grid_minimum") == 0.0


@pytest.mark.parametrize("objective, bounds", [
    # SymPy's value is gamma(x + 1), while SciPy's factorial is 0 at a negative argument.
    (sp.factorial, (-0.9, -0.1)),
    # SymPy's value is complex at a negative argument, where SciPy's and Python's functions return a real number.
    (sp.loggamma, (-2.5, -0.5)),
    (sp.Ci, (-5.0, -0.5)),
])
def test_a_special_function_whose_double_precision_namesake_differs_is_refused(objective, bounds):
    # A table of the namesake's values would be a silently different objective.
    x = sp.Symbol("x", real=True)
    problem = Optimization(objective=objective(x), variables=(x,), bounds=(bounds,))
    with pytest.raises(ValueError):
        plan(problem, method=QHD(num_grid_points=16, num_steps=1, total_time=0.5), execution="classical", seed=3)


def test_the_default_byte_limit_is_the_physical_memory(monkeypatch):
    """With max_bytes=None, a stage that needs more than the physical memory is refused, naming that memory.

    The two-variable binary split-step Plan on D = 1024 points has known classical arrays of 501256 bytes:
    ``64 D + 32 (d + 1) K + 8 N_s + 1024 K + 4096 (d + 1)`` for ``split_step.sizes`` with no tables and
    ``8 D + H0 + 17 D + H_decode`` for the readout (``method.summary_sizes``). A computer that reports 400000
    bytes refuses it before any table is evaluated, and one that reports 3000000 bytes admits it.
    """
    from nwqlib import _limits

    x, y = sp.symbols("x y", real=True)
    problem = Optimization(objective=sp.cos(x * y) + x * y, variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    method = QHD(num_grid_points=32, num_steps=1, total_time=0.5, encoding="binary", boundary="periodic",
                 theory_flavor="split_step")
    assert method.max_bytes is None
    monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: 400000)
    with pytest.raises(ValueError, match="max_bytes=None, the physical memory of 400000 bytes. a computer with at "
                                         "least 501256 bytes of physical memory admits this byte charge"):
        plan(problem, method=method, execution="classical", seed=17)
    monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: 3000000)
    assert plan(problem, method=method, execution="classical", seed=17).method.max_bytes is None


def test_verification_rejects_a_plan_the_result_does_not_belong_to():
    # verify and verification_checks both reject a Plan the Result does not belong to.
    from nwqlib.algorithms.qhd import verification

    x = sp.Symbol("x")
    result = solve(make_plan())
    other = make_plan(Optimization(objective=(x - sp.Rational(1, 2)) ** 2, variables=(x,), bounds=((-1.0, 1.0),)))
    checks = QHDVerification(comparisons=("grid_minimum",))
    receipt, _ = result.verify(checks=checks)
    assert {f.fact.quantity for f in receipt.applications[0].facts} >= {"grid_minimum", "reference_minimum"}
    assert [spec.name for spec in checks.verification_checks(result)] == ["grid_minimum"]
    with pytest.raises(ValueError):
        verification.verify(other, result, checks=checks)
    with pytest.raises(ValueError):
        checks.verification_checks(result.model_copy(update={"plan": other}))


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
    # d K = 1000 here, and 320 d K + (8 d K + 120 d + 40) + (8 d K + 120 d + 64 + 237 d)
    # + 4096 (d + 1) = 349,346 bytes. The one default midpoint step row adds
    # 2 units and 768 bytes, and both are admitted before the amplitudes are
    # evaluated (QHD.plan, method._initial_state_bytes).
    gaussian = GaussianState(center=(0.5, 0.5), widths=(0.2, 0.2))
    with monkeypatch.context() as guard:
        guard.setattr(GaussianState, "_variables", Mock(side_effect=AssertionError("evaluated")))
        for limits in (dict(max_work=1001), dict(max_bytes=350_114 - 1)):
            with pytest.raises(ValueError, match="max_work=.*max_bytes="):
                make_plan(Optimization(objective=x + y, variables=(x, y), bounds=unit),
                          QHD(num_grid_points=500, initial_state=gaussian, **limits))
        # At exactly the allowance the early check passes and the evaluation starts.
        with pytest.raises(AssertionError, match="evaluated"):
            make_plan(Optimization(objective=x + y, variables=(x, y), bounds=unit),
                      QHD(num_grid_points=500, initial_state=gaussian, max_bytes=350_114))

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
    assert len(selected.reconstruction.step_weights) == 5 and selected.reconstruction.onehot is not None


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


def test_classical_kernel_call_fits_its_admitted_workspace(monkeypatch):
    """The traced peak of one warmed classical kernel call fits the workspace that its Plan admits.

    Besides the size laws, the kernel's admission adds the measured per-call
    allowance ``method._KERNEL_CALL_BYTES`` for Python objects of bounded
    size, and the kernel reads its generator norms and compiled blocks one
    at a time (``method._generator_norms``, ``native.raw_blocks``). A warm-up
    solve of the same Plan loads the modules and fills CPython's free lists,
    and garbage is collected before tracing, so the traced peak excludes
    one-time growth. With d = 4 and K = 30, D = 810,000 grid points, the
    arrays proportional to D dominate the admission, and the call fits only
    while the summary keeps its storage within
    ``H0 + max(17D + H_decode, 25D + 8dK)`` (``method._summarize``,
    ``method.summary_sizes``).
    """
    import gc
    import tracemalloc

    d = 4
    names = sp.symbols(f"x0:{d}")
    problem = Optimization(objective=sum((v - sp.Rational(1, 5)) ** 2 for v in names), variables=names,
                           bounds=((-1.0, 1.0),) * d)
    # The large grid needs limits above the defaults.
    selected = plan(problem, method=QHD(num_grid_points=30, num_steps=1, total_time=0.25, theory_flavor="ir_product",
                                        initial_state=UniformState(), max_work=10**13, max_bytes=10**11),
                    execution="classical", seed=7)
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
                                   r.support_values, r.step_weights)
    assert max(norm for norm, _ in norms) > 63.36
    np.random.seed(1)
    before = np.random.get_state()
    first = owner._evolve_restricted(selected, flavor)
    assert _global_random_state_equal(np.random.get_state(), before)
    np.random.seed(2)
    second = owner._evolve_restricted(selected, flavor)
    assert np.array_equal(first, second)


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


def test_a_constant_base_whose_distance_from_one_is_unresolved_is_refused_with_its_remedy():
    """SymPy cannot tell this base from 1 at its working precision, and a Float with enough digits leads past the refusal."""
    x = sp.Symbol("x", real=True)
    base = sp.cos(sp.exp(-200))
    method = QHD(num_grid_points=16, num_steps=1, total_time=0.5, initial_state=UniformState())

    def problem(constant):
        return Optimization(objective=sp.Mul(sp.Pow(constant, x, evaluate=False), sp.exp(-x / 2), evaluate=False),
                            variables=(x,), bounds=((-1.0, 1.0),))

    with pytest.raises(ValueError, match="cannot resolve the distance from 1"):
        plan(problem(base), method=method, execution="classical", seed=3)
    values, (grid,) = _tabulated_values(plan(problem(sp.N(base, 1000)), method=method, execution="classical", seed=3))
    np.testing.assert_allclose(values, np.exp(-grid / 2), rtol=0, atol=1e-14)
