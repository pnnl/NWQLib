"""One-hot QHD rotation laws, T estimate, splitting and schedule bounds and run totals.

The rotation laws are checked against the emitted circuits expanded by Qiskit's own gate definitions, so
the check is also a canary for a Qiskit upgrade that changes a definition. The operator-norm bounds are
checked against an independently integrated time-ordered propagator and against dense commutators.
"""

import itertools
import math
import re
import tracemalloc
from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest
import scipy.integrate
import scipy.linalg
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from nwqlib.algorithms.qhd import (
    QHD,
    AugmentedLagrangian,
    BinarySynthesis,
    BoxRefinement,
    CubicSchedule,
    GaussianState,
    QuadraticSchedule,
    ShiftedCubicSchedule,
    UniformState,
    circuit_resources,
    evolution_bound,
    refine_box,
    run_resources,
    solve_augmented_lagrangian,
    step_blocks,
)
from nwqlib._choice_archive import ArchiveFiles, load_plan, save_plan
from nwqlib.algorithms.qhd.grid import OneHotGrid
from nwqlib.algorithms.qhd.initial_state import append_amplitude_chain
from nwqlib.algorithms.qhd.method import _binary_source_reservation, _grid
from nwqlib.algorithms.qhd.native import append_binary_steps, construct_qhd, raw_blocks
from nwqlib.algorithms.qhd.onehot_schedule import OneHotSchedule, derivation_bytes
from nwqlib.algorithms.qhd.records import QHDBlock
from nwqlib.algorithms.qhd.refinement_records import BoxRefinementResult, RefinementResources
from nwqlib.algorithms.qhd.resources import (
    QHDRunEntry,
    RotationPopulation,
    _total,
    _plan_population,
    _projector_slots,
    clifford_distance,
    rotation_population,
    synthesis_projection,
)
from nwqlib.core.records import FrozenArray
from nwqlib.execution import ExecutionLimits
from nwqlib.problems import ConstrainedOptimization, Optimization
from nwqlib.resources.records import CLIFFORD_ANGLES, T_ANGLES
from nwqlib.scientist import plan, prepare
from nwqlib.subroutines.hamiltonian_evolution import append_pauli_evolution_block
from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import (
    append_number_projector_phase,
    structured_number_projector_provider,
)

x, y, z, w = sp.symbols("x y z w")
ROTATIONS = {"rz", "ry", "rx", "p"}
CLIFFORDS = {"cx", "h", "s", "sdg", "x", "y", "z", "sx", "sxdg", "measure"}


def _census(circuit, magnitudes=None):
    """Return ``(arbitrary, exact_t)`` of a circuit expanded by its gate definitions down to standard gates.

    Rotations are classified by the library's stored conventional angles, the classification the law
    states. Any other leaf without a definition fails the check, so a new gate is not silently skipped.
    ``magnitudes``, a Counter, collects the magnitude of every arbitrary rotation.
    """
    arbitrary = exact_t = 0
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in ROTATIONS:
            angle = abs(float(operation.params[0]))
            if angle in T_ANGLES:
                exact_t += 1
            elif angle not in CLIFFORD_ANGLES:
                arbitrary += 1
                if magnitudes is not None:
                    magnitudes[angle] += 1
        elif operation.name in ("t", "tdg"):
            exact_t += 1
        elif operation.name not in CLIFFORDS:
            assert operation.definition is not None, operation.name
            more, t = _census(operation.definition, magnitudes)
            arbitrary, exact_t = arbitrary + more, exact_t + t
    return arbitrary, exact_t


def _plan(problem, method, **options):
    return plan(problem, method=method, execution="quantum", seed=7, **options)


@pytest.mark.parametrize("support", range(1, 10))
@pytest.mark.parametrize("angle", [0.371, 8 * math.pi])
def test_projector_rotations_match_the_qiskit_definitions(support, angle):
    # 8 pi puts the MCPhase slots at 2 pi, pi and pi/2 (Clifford), pi/4 (T), pi/8 and below, while the
    # multiplexor (4 <= s <= 6) wraps the phase first. From s = 7 the MCPhase is Qiskit's da Silva-Park
    # construction, whose CRX gates add fixed T gates and rotations at pi/8 and below (135 arbitrary
    # rotations and 44 T at s = 8 for a generic angle, against 255 rotations from a 2**s - 1 law).
    provider = structured_number_projector_provider(support)["provider"]
    block = QHDBlock(kind="number_projector", time_step=1.0, time=0.5, support=tuple(range(support)),
                     coefficient=angle, angle=angle, **structured_number_projector_provider(support))
    circuit = QuantumCircuit(support)
    append_number_projector_phase(circuit, range(support), angle)
    population = rotation_population(SimpleNamespace(initial_amplitudes=()), "none", [block])
    magnitudes = Counter()
    assert (population.arbitrary_rotations, population.exact_t) == _census(circuit, magnitudes)
    assert population.magnitudes == tuple(sorted(magnitudes.items()))
    if angle == 0.371:
        assert population.arbitrary_rotations + population.exact_t == (1, 3, 7, 15, 31, 63, 129, 179, 237)[support - 1]
    slots, _, _ = _projector_slots(block)
    mcphase_slots = max(1, 4 * support - 5) if support <= 5 else 4 * support**2 - 10 * support + 3
    assert sum(slots.values()) == (2**support - 1 if provider == "diagonal_synthesis" else mcphase_slots)


def test_rotation_law_bounds_the_emitted_circuit_and_a_budget_classifies_it():
    # Supports of size 1, 2, 3 (MCPhase) and 4 (phase diagonal), the uniform chain, second order. The
    # 1e-3 y term gives projector angles near 3e-5 that rotation_threshold=1e-4 removes. Planning counts
    # every rotation gate without classifying an angle, an upper bound on the arbitrary rotations and exact
    # T gates of the built circuit. Without a synthesis budget no angle is classified, and with one the
    # record holds the classified counts of the emitted circuit.
    problem = Optimization(objective=x + x * y + x * y * z + 3 * x * y * z * w + y / 1000,
                           variables=(x, y, z, w), bounds=((-1.0, 1.0),) * 4)
    method = QHD(num_grid_points=2, num_steps=2, total_time=0.3, rotation_threshold=1e-4)
    selected = _plan(problem, method)
    assert selected.reconstruction.dropped_count > 0
    unpruned = _plan(problem, method.revise(rotation_threshold=0.0))
    periodic = _plan(Optimization(objective=sp.cos(3 * x), variables=(x,), bounds=((0.0, 1.0),)),
                     QHD(num_grid_points=4, num_steps=3, total_time=0.4, boundary="periodic", trotter_order=1))
    # A narrow Gaussian leaves a zero suffix in the first register (qhd workflow test of the same state).
    tail = _plan(Optimization(objective=x * y, variables=(x, y), bounds=((0.0, 4.0), (0.0, 4.0))),
                 QHD(num_grid_points=3, num_steps=1, initial_state=GaussianState(center=(1.0, 2.0), widths=(0.02, 10.0))))
    counts = {}
    for name, selected_plan in (("pruned", selected), ("unpruned", unpruned), ("periodic", periodic), ("tail", tail)):
        (law,) = [law for law in selected_plan.construction.selections[0].resource_laws
                  if law.metric == "arbitrary_rotations"]
        emitted = _census(construct_qhd(selected_plan.blocks[0], (), None))
        assert law.interpretation == "upper_bound" and law.value >= sum(emitted)
        unclassified = circuit_resources(selected_plan)
        assert (unclassified.arbitrary_rotations, unclassified.exact_t, unclassified.synthesis) == (None, None, None)
        record = circuit_resources(selected_plan, synthesis_epsilon=1e-3)
        assert (record.arbitrary_rotations, record.exact_t) == emitted
        counts[name] = emitted[0]
    assert counts["pruned"] < counts["unpruned"]
    # The SDK preparation has no rotation law, like its CX law.
    sdk = _plan(problem, method.revise(init_prep="qiskit_state_preparation"))
    assert sdk.construction.selections[0].resource_laws == ()
    with pytest.raises(ValueError, match="no rotation law"):
        circuit_resources(sdk)


def test_clifford_distance_and_budgeted_replacement():
    def distance(theta):
        # Phase-minimized distance to the nearest of the eight Clifford rotations Rz(k pi/2), by a dense
        # scan of the common phase refined near its minimum, independent of the closed form.
        rz = np.array([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
        best = np.inf
        for k in range(-4, 5):
            target = np.array([np.exp(-0.25j * k * np.pi), np.exp(0.25j * k * np.pi)])
            phases = np.linspace(-np.pi, np.pi, 20001)
            values = np.max(np.abs(rz[None, :] - np.exp(1j * phases)[:, None] * target[None, :]), axis=1)
            center = phases[np.argmin(values)]
            fine = np.linspace(center - 1e-3, center + 1e-3, 20001)
            best = min(best, np.min(np.max(np.abs(rz[None, :] - np.exp(1j * fine)[:, None] * target[None, :]),
                                           axis=1)))
        return best

    def exact(theta):
        # 2 sin(|delta|/4) with the mathematical pi at 60 digits.
        with mpmath.workdps(60):
            delta = mpmath.mpf(theta) - mpmath.nint(mpmath.mpf(theta) / (mpmath.pi / 2)) * mpmath.pi / 2
            return 2 * mpmath.sin(abs(delta) / 4)

    # The scan's phase step 1e-7 moves the distance by at most 1e-7 (a first-order change of |exp(i phi) - c|).
    # The value is an outward bound on the distance.
    for theta in (0.03, -0.03, np.pi / 2 + 0.01, -np.pi + 0.2, 2 * np.pi - 0.01, np.pi / 4, 5.0):
        assert abs(clifford_distance(theta) - distance(theta)) < 2e-7
        assert clifford_distance(theta) >= exact(theta)
    # 2**52 binary64 quarter turns lie 0.28 rad from 2**52 mathematical ones, a distance of 0.138 that a
    # reduction by the binary64 pi/2 misses. At E = 1e-4 the rotation stays, and so do the projector
    # rotations of a public Plan with this slope.
    large = math.ldexp(math.pi / 2, 52)
    assert clifford_distance(large) >= exact(large) > 0.13
    assert synthesis_projection(RotationPopulation(magnitudes=((large, 1),), exact_t=0), 1e-4).replaced == 0
    steep = _plan(Optimization(objective=sp.Float(large) * x, variables=(x,), bounds=((0.0, 3.0),)),
                  QHD(num_grid_points=2, num_steps=1, total_time=1.0, trotter_order=1, rotation_threshold=0.0,
                      initial_state=UniformState(), schedule=QuadraticSchedule(gamma=0.0)))
    assert circuit_resources(steep, synthesis_epsilon=1e-4).synthesis.replaced == 0
    # Three rotations near Cliffords and five generic ones, E = 8e-3, so q = 1e-3. d_C(1e-3) = 5e-4 and
    # d_C(pi/2 + 3e-3) = 1.5e-3 about, so only the first is replaced. The recorded charge is outward and the
    # allowance downward, so the replaced distance plus 7 allowances stays within E.
    population = RotationPopulation(magnitudes=((1e-3, 1), (0.3, 5), (np.pi / 2 + 3e-3, 2)), exact_t=4)
    projection = synthesis_projection(population, 8e-3)
    charge = exact(1e-3)
    assert (projection.candidates, projection.replaced, projection.rotations) == (8, 1, 7)
    assert charge <= projection.replacement_error <= charge * (1 + 2**-51)
    assert projection.rotation_epsilon == pytest.approx(float((8e-3 - charge) / 7), rel=2**-51)
    with mpmath.workdps(60):
        assert charge + 7 * mpmath.mpf(projection.rotation_epsilon) <= mpmath.mpf(8e-3)
    assert projection.rotation_epsilon >= projection.threshold
    expected = 4 + 3 * 7 * math.log2(7 / (8e-3 - float(charge)))
    assert projection.t.value.value == pytest.approx(expected, rel=1e-15)  # machine precision: one log2
    # Eleven rotations at 0.0047 and a budget one step below eleven binary64 distances: the replacement
    # charge stays within E, which a rounded sum of the binary64 distances would exceed.
    angle, budget = 0.004728018137566003, 0.026004093701408337
    tight = synthesis_projection(RotationPopulation(magnitudes=((angle, 11),), exact_t=0), budget)
    assert tight.replacement_error <= budget
    edge = math.nextafter(11 * clifford_distance(angle), math.inf)
    every = synthesis_projection(RotationPopulation(magnitudes=((angle, 11),), exact_t=0), edge)
    assert every.replaced == 11 and 11 * exact(angle) <= every.replacement_error <= edge
    # A budget above every d_C replaces all rotations: the count is the exact T of the decomposition.
    everything = synthesis_projection(population, 10.0)
    assert (everything.rotations, everything.t.value, everything.t.interpretation) == (0, 4, "exact")
    # An allowance of at least 1 is outside the logarithmic model: q = 0.012 replaces the 99 rotations at
    # 1e-3 and leaves 0.7 (d_C = 0.35), which receives 1.2 - 99 d_C(1e-3), about 1.15.
    large = synthesis_projection(RotationPopulation(magnitudes=((1e-3, 99), (0.7, 1)), exact_t=0), 1.2)
    assert large.t is None and "not below 1" in large.t_unavailable

    # The stored fields and public ledger must compose in exact arithmetic, as derived in
    # synthesis_projection. A binary64 sum can hide an excess over the supplied budget.
    objective = sp.Piecewise((0, sp.Eq(x, 0)), (sp.Rational(1, 2), x < 7), (sp.Rational(742, 9), True))
    selected = _plan(Optimization(objective=objective, variables=(x,), bounds=((-1.0, 8.0),)),
                     QHD(num_grid_points=8, num_steps=1, total_time=0.0045, trotter_order=1,
                         initial_state=UniformState(), schedule=QuadraticSchedule(gamma=0.0)))
    no_candidates = _plan(Optimization(objective=sp.Integer(0), variables=(x,), bounds=((-1.5, 1.5),)),
                          QHD(num_grid_points=2, num_steps=1, total_time=math.pi, trotter_order=1,
                              initial_state=UniformState(), schedule=QuadraticSchedule(gamma=0.0)))
    for chosen, budget in ((selected, 0.037155), (selected, 100.0), (no_candidates, 0.037155)):
        resources = circuit_resources(chosen, synthesis_epsilon=budget)
        stored = resources.synthesis
        entries = {entry.name: entry for entry in resources.error_sources}
        replacement = Fraction(stored.replacement_error)
        remaining = Fraction(entries["synthesis"].value)
        allocated = stored.rotations * Fraction(stored.rotation_epsilon or 0)
        assert entries["synthesis"].status == "requested"
        # The ledger caps operator-distance bounds at 2, while requested allowances stay uncapped.
        assert Fraction(entries["clifford_replacement"].value) == min(2, replacement)
        assert 0 <= allocated <= remaining
        assert replacement + remaining <= Fraction(budget)
        assert replacement + allocated <= Fraction(budget)
        if chosen is no_candidates:
            assert stored.candidates == 0 and stored.threshold is None
        elif budget == 100.0:
            assert stored.candidates > 0 and stored.rotations == 0
        else:
            assert 0 < stored.replaced < stored.candidates
        assert (stored.rotation_epsilon is None) == (stored.rotations == 0)


def test_t_estimate_of_a_small_circuit():
    # (x - 1/5)**2 on three interior points from the uniform start, one first-order step, time-independent
    # schedule: four chain rotations (two exact T at the last link), four hopping and two potential
    # rotations. With E = 1e-4 and no replacement the estimate is 2 + 8 * 3 log2(8/1e-4). NWQEC 0.1.2
    # compiled this circuit at the same total budget to 348 T (docs/algorithms/qhd.md, "Fault-tolerant resources").
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    selected = _plan(problem, QHD(num_grid_points=3, num_steps=1, total_time=0.17, trotter_order=1,
                                  schedule=QuadraticSchedule(gamma=0.0), initial_state=UniformState()), shots=8)
    record = circuit_resources(selected, synthesis_epsilon=1e-4)
    assert (record.arbitrary_rotations, record.exact_t) == (8, 2)
    assert record.synthesis.t.value.value == pytest.approx(2 + 24 * math.log2(8e4), rel=4e-16)
    statuses = {source.name: source.status for source in record.error_sources}
    assert statuses["synthesis"] == "requested" and statuses["compiler"] == "unavailable"
    # A subnormal budget still has a finite estimate, and one whose share of 8 rotations lies below 2**-1074
    # is refused.
    for budget in (1e-309, 1e-320):
        tiny = circuit_resources(selected, synthesis_epsilon=budget).synthesis
        assert tiny.t.value.value == pytest.approx(2 + 24 * (3 - math.log2(budget)), rel=1e-12)
    with pytest.raises(ValueError, match="below the smallest positive binary64 number"):
        circuit_resources(selected, synthesis_epsilon=2.0**-1074)


def _kinetic(k, h, periodic):
    """Restricted finite-difference -Delta/2 on K points: 1/h**2 on the diagonal, -1/(2 h**2) per link."""
    matrix = np.diag(np.full(k, 1 / h**2))
    for i in range(k - 1 + int(periodic)):
        j = (i + 1) % k
        matrix[i, j] = matrix[j, i] = -1 / (2 * h**2)
    return matrix


def _weights(schedule):
    if isinstance(schedule, QuadraticSchedule):
        return lambda t: 1 / (1 + schedule.gamma * t * t), lambda t: 1 + schedule.gamma * t * t
    if isinstance(schedule, CubicSchedule):
        return lambda t: 2 / (schedule.s + t**3), lambda t: 2 * t**3
    return lambda t: 8 / (schedule.s + t) ** 3, lambda t: 2 * t**3


WELL = x**2 / 4 + x**3 / 16 + sp.Rational(7, 8)
# (K, box, objective, order, schedule, rule, steps, total time, boundary). The first six use six points
# of spacing 1 over four steps to time 0.5. The two zero-objective cases isolate the link splitting.
# The K = 2 second-order integrated case with gamma = 5 fails without the time-ordering term (bound
# 3.2e-4, 8.0e-5 without it, measured 1.9e-4), and the K = 2 constant-objective midpoint case without
# the quadrature term (bound 2.7e-3, 4e-17 without it, measured 8.2e-4), since their splitting terms are
# small or zero.
CASES = [
    (6, (-3.5, 3.5), WELL, 1, QuadraticSchedule(gamma=0.3), "midpoint", 4, 0.5, "dirichlet"),
    (6, (-3.5, 3.5), WELL, 2, CubicSchedule(s=1.0), "integrated", 4, 0.5, "dirichlet"),
    (6, (-2.5, 3.5), WELL, 1, ShiftedCubicSchedule(s=2.0), "integrated", 4, 0.5, "periodic"),
    (6, (-2.5, 3.5), WELL, 2, QuadraticSchedule(gamma=0.3), "midpoint", 4, 0.5, "periodic"),
    (6, (-3.5, 3.5), sp.Integer(0), 1, QuadraticSchedule(gamma=0.0), "midpoint", 4, 0.5, "dirichlet"),
    (6, (-2.5, 3.5), sp.Integer(0), 2, QuadraticSchedule(gamma=0.0), "midpoint", 4, 0.5, "periodic"),
    (2, (-1.5, 1.5), x**2 / 2 + x / 3, 2, QuadraticSchedule(gamma=5.0), "integrated", 16, 1.0, "dirichlet"),
    (2, (-1.5, 1.5), sp.Rational(7, 8), 1, QuadraticSchedule(gamma=3.0), "midpoint", 16, 1.0, "dirichlet"),
]


@pytest.mark.parametrize("k, box, objective, order, schedule, rule, steps, total, boundary", CASES)
def test_evolution_bound_dominates_the_time_ordered_error(k, box, objective, order, schedule, rule, steps, total,
                                                          boundary):
    # The emitted product of the stored blocks, with the physical phase restored, is compared with
    # U' = -i H(t) U integrated by DOP853 at rtol 1e-12, atol 1e-14 over [0, steps*dt] on the stored
    # tables, constant and grid, the finite model that evolution_bound targets. Its integration error,
    # below 1e-10, is negligible against the margins (the bound exceeds the measured error by at least a
    # factor 1.5 in these cases), so the check allows only that.
    method = QHD(num_grid_points=k, num_steps=steps, total_time=total, trotter_order=order, schedule=schedule,
                 coefficient_rule=rule, boundary=boundary, init_prep="none")
    selected = _plan(Optimization(objective=objective, variables=(x,), bounds=(box,)), method)
    r = selected.reconstruction
    circuit = QuantumCircuit(k)
    for block in raw_blocks(r, method, _grid(selected)):
        append_pauli_evolution_block(circuit, block)
    circuit.global_phase += r.physical_phase
    onehot = [1 << i for i in range(k)]
    emitted = Operator(circuit).data[np.ix_(onehot, onehot)]
    table = np.zeros(k) if not r.support_values else np.array(r.support_values[0].values)
    periodic = boundary == "periodic"
    spacing = (box[1] - box[0]) / (k if periodic else k + 1)
    kinetic, potential_matrix = _kinetic(k, spacing, periodic), np.diag(table + r.constant)
    a, b = _weights(schedule)

    def rhs(t, u):
        return (-1j * (a(t) * kinetic + b(t) * potential_matrix) @ u.reshape(k, k)).ravel()

    solution = scipy.integrate.solve_ivp(rhs, (0.0, steps * (total / steps)), np.eye(k, dtype=complex).ravel(),
                                         method="DOP853", rtol=1e-12, atol=1e-14)
    exact = solution.y[:, -1].reshape(k, k)
    error = np.linalg.norm(emitted - exact, 2)
    bound = evolution_bound(selected)
    assert error > 1e-6
    assert bound.evolution_status == "conditional" if rule == "integrated" and schedule.kind != "shifted_cubic" \
        else bound.evolution_status == "bound"
    assert error <= bound.evolution + 1e-10
    if objective == 0:
        assert bound.commutator == 0 and bound.time_ordering == 0


@pytest.mark.parametrize("schedule", [QuadraticSchedule(gamma=0.7), CubicSchedule(s=0.3), ShiftedCubicSchedule(s=0.4)])
def test_schedule_derivative_bounds_cover_the_weights(schedule):
    # |a|, |b| and their first two derivatives, differentiated symbolically and sampled densely on each
    # interval, lie below the analytic interval bounds of derivative_bounds (a maximum over samples can
    # only be smaller than the true maximum, which the bound must exceed).
    t = sp.Symbol("t")
    a, b = _weights(schedule)
    functions = [sp.lambdify(t, expression) for f in (a(t), b(t)) for expression in (f, f.diff(t), f.diff(t, 2))]
    for lower, upper in ((0.0, 0.25), (0.5, 0.75), (1.0, 3.0)):
        bounds = schedule.derivative_bounds(Fraction(lower), Fraction(upper))
        a_max, b_max, a1, a2, b1, b2 = (float(value) for value in bounds)
        samples = np.linspace(lower, upper, 2001)
        observed = [np.max(np.abs(np.vectorize(f)(samples))) for f in functions]
        for value, limit in zip(observed, (a_max, a1, a2, b_max, b1, b2), strict=True):
            assert value <= limit * (1 + 1e-12)


def _dense_layers(k, h, periodic):
    links = [(i, (i + 1) % k) for i in range(k - 1 + int(periodic))]
    layers = []
    for parity in (1, 0):
        matrix = np.zeros((k, k))
        for i, j in links[parity::2]:
            matrix[i, j] = matrix[j, i] = -1 / (2 * h**2)
        layers.append(matrix)
    return links, layers


def _commutator(a, b):
    return a @ b - b @ a


def test_norm_inputs_bound_the_dense_commutators():
    # Two variables on four points of spacing 1/2 with a coupled table, where the range and neighbor
    # bounds differ. The dense 16-by-16 restricted operators are an independent check of C, D_T, D_V,
    # mu_T and mu_V, of the closed form of Gamma and of the layer constants J_E and J_O.
    problem = Optimization(objective=x**2 / 4 + y**2 / 8 + x * y / 3 + sp.Rational(7, 8), variables=(x, y),
                           bounds=((-1.0, 1.0),) * 2)
    method = QHD(num_grid_points=4, num_steps=1, boundary="periodic", init_prep="none")
    first, second = _plan(problem, method.revise(trotter_order=1)), _plan(problem, method)
    r = first.reconstruction
    k = 4
    one = _kinetic(k, 0.5, True)
    kinetic = np.kron(one, np.eye(k)) + np.kron(np.eye(k), one)
    values = np.full((k, k), r.constant)
    for table in r.support_values:
        entries = np.array(table.values).reshape((k,) * len(table.support))
        values += entries if table.support == (0, 1) else (
            entries[:, None] if table.support == (0,) else entries[None, :])
    potential = np.diag(values.ravel())
    record, rtol = evolution_bound(first), 1e-12  # dense double-precision norms against exact bounds
    tv = _commutator(kinetic, potential)
    for dense, bound in ((tv, record.commutator), (_commutator(kinetic, tv), record.kinetic_nested),
                         (_commutator(potential, -tv), record.potential_nested), (kinetic, record.kinetic_norm),
                         (potential, record.potential_norm)):
        assert np.linalg.norm(dense, 2) <= bound * (1 + rtol)
    links, (odd, even) = _dense_layers(k, 0.5, True)
    gamma = 0.0
    for e in range(len(links)):
        link = np.zeros((k, k))
        later = np.zeros((k, k))
        for f, (i, j) in enumerate(links):
            target = link if f == e else later if f > e else None
            if target is not None:
                target[i, j] = target[j, i] = -1 / (2 * 0.5**2)
        gamma += np.linalg.norm(_commutator(later, link), 2)
    assert record.hopping == pytest.approx(2 * gamma, rel=rtol)  # two identical variables
    nested = evolution_bound(second)
    both = [np.kron(layer, np.eye(k)) + np.kron(np.eye(k), layer) for layer in (odd, even)]
    assert np.linalg.norm(_commutator(both[1], _commutator(both[1], both[0])), 2) <= nested.even_nested * (1 + rtol)
    assert np.linalg.norm(_commutator(both[0], _commutator(both[0], both[1])), 2) <= nested.odd_nested * (1 + rtol)
    # The layer constants are the integer row sums of the unit-weight nested commutators of one variable
    # times sum_j w_j**3 = 2 * 2**3, all exact in binary64. The odd layer (links[1::2]) is the outer one.
    unit_odd, unit_even = (np.round(layer / (-1 / (2 * 0.5**2))) for layer in (odd, even))
    rows = [np.max(np.sum(np.abs(_commutator(f, _commutator(f, g))), axis=1))
            for f, g in ((unit_even, unit_odd), (unit_odd, unit_even))]
    assert (nested.even_nested, nested.odd_nested) == (16 * rows[0], 16 * rows[1])


def test_run_totals_are_sums_over_the_recorded_circuits():
    # Three augmented-Lagrangian rounds, two refinement levels, and a run with refinement in each round
    # (two levels in round 0, one in round 1 before three circuits are spent), each with its own circuit
    # and 64 shots. The records sum the per-circuit laws, and run_resources multiplies each circuit by
    # its preparations and shots. A binary refinement records its upper-bound rotation law. The uniform start
    # and gain 1 are named because the asserted level counts follow from the refinement path they set.
    f = (x - 1) ** 2 + (y - 1) ** 2
    equality = ConstrainedOptimization(objective=f, variables=(x, y), bounds=((-1.0, 1.0),) * 2,
                                       equalities=(x + y,))
    options = AugmentedLagrangian(inner_point="best_observed", feasibility_tolerance=1e-9)
    constrained = solve_augmented_lagrangian(equality,
                                             qhd=QHD(num_grid_points=3, num_steps=1, total_time=0.5,
                                                     initial_state=UniformState()),
                                             options=options, execution="quantum", shots=64, seed=11,
                                             progress=False)
    levels = BoxRefinement(scaling="search_model", potential_gain=1.0, point_rule="best_observed", max_levels=2,
                           mass_threshold=0.6)
    qhd = QHD(num_grid_points=3, num_steps=4, total_time=3.0, initial_state=UniformState())
    refined = refine_box(Optimization(objective=(x - sp.Rational(3, 10)) ** 2, variables=(x,), bounds=((-1.0, 1.0),)),
                         qhd=qhd, options=levels, execution="quantum", shots=64, seed=3, progress=False)
    nested = solve_augmented_lagrangian(equality, qhd=qhd, refinement=levels, options=options, execution="quantum",
                                        shots=64, seed=5, limits=ExecutionLimits(max_total_circuits=3),
                                        progress=False)
    binary = refine_box(Optimization(objective=(x - sp.Rational(3, 10)) ** 2, variables=(x,), bounds=((-1.0, 1.0),)),
                        qhd=qhd.revise(num_grid_points=4, encoding="binary", boundary="periodic"), options=levels,
                        execution="quantum", shots=64, seed=3, progress=False)
    cases = (
        (constrained, [item.resources for item in constrained.iterations], [r.plan for r in constrained.results]),
        (refined, [level.resources for level in refined.levels], [r.plan for r in refined.results]),
        (nested, [level.resources for item in nested.iterations for level in item.refinement.levels],
         [r.plan for round_results in nested.results for r in round_results]),
        (binary, [level.resources for level in binary.levels], [r.plan for r in binary.results]),
    )
    assert [len(item.refinement.levels) for item in nested.iterations] == [2, 1]
    for result, rows, plans in cases:
        assert len(plans) >= 2 and len(rows) == len(plans)
        laws = [circuit_resources(p, synthesis_epsilon=1e-3) for p in plans]
        assert result.resources.arbitrary_rotations == sum(item.rotation_law.value for item in laws)
        totals = run_resources(result, synthesis_epsilon=1e-3)
        assert totals.unavailable == ()
        assert totals.rotation_interpretation == "upper_bound"
        assert totals.arbitrary_rotations == sum(row.circuit_preparations * item.rotation_law.value
                                                 for row, item in zip(rows, laws))
        assert totals.shot_arbitrary_rotations == sum(row.shots * item.rotation_law.value
                                                      for row, item in zip(rows, laws))
        assert totals.t_estimate == pytest.approx(
            sum(row.circuit_preparations * item.synthesis.t.value.value for row, item in zip(rows, laws)), rel=1e-15)
        assert totals.shot_t_estimate == pytest.approx(
            sum(row.shots * item.synthesis.t.value.value for row, item in zip(rows, laws)), rel=1e-15)
        assert [entry.shots for entry in totals.entries] == [64] * len(plans)
    # An entry with zero multiplicity contributes zero even without a value, and one with shots but no value
    # makes the total unavailable with its reason (QHDRunResources states the rule).
    entries = (QHDRunEntry(label="round 0", plan_id=None, circuits=0, shots=0, arbitrary_rotations=None,
                           t_estimate=None, unavailable=(("arbitrary_rotations", "planning raised"),
                                                         ("t_estimate", "planning raised"))),
               QHDRunEntry(label="round 1", plan_id=None, circuits=1, shots=64, arbitrary_rotations=10, t_estimate=None,
                           unavailable=(("t_estimate", "its QHD result is not available"),)))
    assert _total(entries, "shots", "arbitrary_rotations") == (640, None)
    assert _total(entries, "shots", "t_estimate") == (None, "round 1: its QHD result is not available")
    # A level that stopped a refinement keeps its recorded count without a Plan, and the count stays an
    # upper bound.
    single = QHD(num_grid_points=2, num_steps=1, total_time=0.3, trotter_order=1, rotation_threshold=0.0,
                 initial_state=UniformState(), schedule=QuadraticSchedule(gamma=0.0))
    stopping = _plan(Optimization(objective=x / 3, variables=(x,), bounds=((0.0, 1.0),)), single)
    law = next(item for selection in stopping.construction.selections for item in selection.resource_laws
               if item.metric == "arbitrary_rotations")
    spent = RefinementResources(circuit_preparations=1, circuit_attempts=1, shots=8, arbitrary_rotations=law.value)
    stopped = BoxRefinementResult(problem_id=stopping.problem.content_id, qhd=single, options=BoxRefinement(),
                                  execution="quantum", shots=8, seed_entropy=7, levels=(), best_level=None,
                                  termination="no_valid_point", failure="no valid one-hot samples",
                                  stopped_resources=spent, resources=spent, max_plannings=2)
    totals = run_resources(stopped)
    assert law.interpretation == "upper_bound"
    assert (totals.rotation_interpretation, totals.arbitrary_rotations, totals.shot_arbitrary_rotations) == (
        "upper_bound", law.value, 8 * law.value)
    # Exact readout reserves no shots, and its shot totals are unknown rather than zero work, while the
    # compiled-circuit totals stay available.
    exact = solve_augmented_lagrangian(equality, qhd=QHD(num_grid_points=3, num_steps=1, total_time=0.5,
                                                         initial_state=UniformState()),
                                       options=AugmentedLagrangian(inner_point="best_observed", max_iterations=1,
                                                                   feasibility_tolerance=1e-9),
                                       execution="quantum", seed=11, progress=False)
    totals = run_resources(exact, synthesis_epsilon=1e-3)
    assert [entry.shots for entry in totals.entries] == [0]
    assert totals.arbitrary_rotations > 0 and totals.t_estimate > 0
    assert totals.shot_arbitrary_rotations is None and totals.shot_t_estimate is None
    assert [name for name, reason in totals.unavailable if "exact readout" in reason] == [
        "shot_arbitrary_rotations", "shot_t_estimate"]


def test_error_ledger_names_every_stage():
    # Second order with a pruned table and supports of size 1 and 4, so both projector providers, the
    # omitted blocks and the phase-diagonal wrap occur. Without pruning nothing is omitted. With it the
    # omitted weight is the dropped blocks' intended exponents times their traceless norm, which the
    # rounded running sum of their stored angles times that norm matches to within a few units of roundoff.
    problem = Optimization(objective=x + 3 * x * y * z * w + y / 1000, variables=(x, y, z, w),
                           bounds=((-1.0, 1.0),) * 4)
    method = QHD(num_grid_points=2, num_steps=2, total_time=0.3, rotation_threshold=1e-4)
    pruned, full = _plan(problem, method), _plan(problem, method.revise(rotation_threshold=0.0))
    records = [circuit_resources(p, synthesis_epsilon=1e-3) for p in (pruned, full)]
    sources = [{item.name: item for item in record.error_sources} for record in records]
    assert [item.name for item in records[0].error_sources] == [
        "kinetic_model", "time_ordering", "midpoint_quadrature", "coefficient_rounding", "product_formula", "angle_formation",
        "aqft", "rotation_pruning", "gate_parameters", "diagonal_wrap", "identity_phase",
        "state_preparation", "clifford_replacement", "synthesis", "compiler"]
    assert sources[1]["rotation_pruning"].value == 0.0
    # The dropped blocks are the single-qubit y projectors, whose traceless generator has norm 1/2.
    assert sources[0]["rotation_pruning"].value == pytest.approx(pruned.reconstruction.dropped_angle_sum / 2, rel=1e-12)
    assert sources[0]["diagonal_wrap"].status == "unavailable"
    assert sources[0]["coefficient_rounding"].status == "bound"
    for name in ("angle_formation", "identity_phase", "state_preparation"):
        # Roundoff-sized entries: a few units of u times the exponents, phases and chain lengths involved.
        assert 0 < sources[0][name].value < 1e-12


def test_one_hot_angle_formation_bounds_the_exact_discrepancies():
    # Oracle: the exact sum of min(2, |lambda^ - lambda*|) over the hopping blocks and of
    # min(2, (1 - 2**-s) |zeta^ - zeta*|) over the projectors, at the nominal durations, which the a priori
    # bound 3u/(1 - 6u) |lambda^| + 2u/(1 - 4u) (1 - 2**-s) |zeta^| must reach without exceeding about
    # 3u sum |angle|. The blocks are those the circuit emits (step_blocks), and the entry is the Plan's record.
    problem = Optimization(objective=x + 3 * x * y * z * w + y / 1000 + x * y / 7, variables=(x, y, z, w),
                           bounds=((-1.0, 1.0), (-0.7, 1.3), (-1.0, 1.0), (-1.0, 1.0)))
    method = QHD(num_grid_points=3, num_steps=3, total_time=0.7, schedule=QuadraticSchedule(gamma=0.3),
                 init_prep="none")
    selected = _plan(problem, method)
    r, grid = selected.reconstruction, _grid(selected)
    k, dt = grid.num_grid_points, Fraction(method.total_time / method.num_steps)
    tables = {table.support: table.values.array for table in r.support_values}
    odd = {link for index, link in enumerate(grid.links()) if index % 2}

    def oracle(steps):
        total = magnitude = Fraction(0)
        for group, (_t, a, b) in zip(steps, r.step_weights, strict=True):
            for block in group:
                stored = Fraction(block.angle)
                if block.kind == "kinetic":
                    left, right = block.support
                    t = dt / 2 if (left % k, right % k) in odd else dt
                    intended = -t * Fraction(a) / (2 * Fraction(grid.spacing(left // k)) ** 2)
                    norm = Fraction(1)
                else:
                    index = sum((q % k) * k ** (len(block.support) - 1 - i) for i, q in enumerate(block.support))
                    v = tables[tuple(q // k for q in block.support)][index]
                    intended, norm = dt / 2 * Fraction(b) * Fraction(v), 1 - Fraction(1, 2 ** len(block.support))
                total += min(2, norm * abs(stored - intended))
                magnitude += norm * abs(stored)
        return total, magnitude

    steps = tuple(step_blocks(selected, step) for step in range(len(r.step_weights)))
    exact, magnitude = oracle(steps)
    formation = {item.name: item for item in circuit_resources(selected).error_sources}["angle_formation"]
    assert {block.provider for step in steps for block in step} >= {"XXPlusYYGate", "diagonal_synthesis"}
    assert 0 < exact <= formation.value <= Fraction(3, 2**53) * (1 + 2.0**-40) * magnitude


COUPLED = x**2 / 3 + x * y - y / 50 + sp.Rational(1, 7)


@pytest.mark.parametrize("objective, lower, schedule, order, threshold", [
    (COUPLED, -1.0, QuadraticSchedule(gamma=0.3), 1, 2e-3), (COUPLED, -1.0, QuadraticSchedule(gamma=0.3), 2, 3e-2)])
def test_later_ledger_stages_bound_the_native_circuit_against_the_split_product(objective, lower, schedule, order,
                                                                               threshold):
    # The split product applies, per step, exp(-i beta V) and every intended hopping exponential with the
    # stored exponents alpha = dt a_k and beta = dt b_k in the compiler's order (first order: V, then each
    # variable's links in order; second order: V/2, odd links for alpha/2, even links, odd links for alpha/2,
    # V/2), with the kinetic diagonal as a phase. It is built densely on the valid one-hot subspace from the
    # stored tables and constant, independently of the compiler. The native circuit with its physical phase
    # differs from it by at most the sum of the ledger's later stages (circuit_errors), which the omitted
    # blocks dominate here.
    # The float64 reference errs by about 1e-15, far below the margins.
    k, box = 3, ((lower, 1.2), (-0.8, 1.0))
    problem = Optimization(objective=objective, variables=(x, y), bounds=box)
    method = QHD(num_grid_points=k, num_steps=3, total_time=0.6, trotter_order=order, schedule=schedule,
                 rotation_threshold=threshold, init_prep="none")
    selected = _plan(problem, method)
    r = selected.reconstruction
    circuit = QuantumCircuit(2 * k)
    for block in raw_blocks(r, method, _grid(selected)):
        append_pauli_evolution_block(circuit, block)
    circuit.global_phase += r.physical_phase
    states = list(itertools.product(range(k), repeat=2))
    onehot = [(1 << s[0]) | (1 << (k + s[1])) for s in states]  # point i of variable j on qubit j*K + i
    emitted = Operator(circuit).data[np.ix_(onehot, onehot)]
    potential = np.full(len(states), float(r.constant))
    for table in r.support_values:
        values = np.array(table.values).reshape((k,) * len(table.support))
        potential += np.array([values[tuple(s[j] for j in table.support)] for s in states])
    spacings = [(upper - lower) / (k + 1) for lower, upper in box]
    dt = method.total_time / method.num_steps
    split = np.eye(len(states), dtype=complex)
    for _t, a, b in r.step_weights:
        alpha, beta = dt * a, dt * b
        segments = [((0, 1), 1.0)] if order == 1 else [((1,), 0.5), ((0,), 1.0), ((1,), 0.5)]
        kinetic = np.eye(len(states), dtype=complex)
        for j, h in enumerate(spacings):
            for links, share in segments:
                for link in links:
                    pair = np.zeros((k, k))
                    pair[link, link + 1] = pair[link + 1, link] = 1.0
                    generator = np.kron(pair, np.eye(k)) if j == 0 else np.kron(np.eye(k), pair)
                    kinetic = scipy.linalg.expm(1j * share * alpha / (2 * h * h) * generator) @ kinetic
        diagonal = np.exp(-1j * alpha * sum(1 / h**2 for h in spacings))
        fraction = 1.0 if order == 1 else 0.5
        field = np.diag(np.exp(-1j * fraction * beta * potential))
        step = kinetic @ field if order == 1 else field @ kinetic @ field
        split = diagonal * step @ split
    distance = np.linalg.norm(emitted - split, 2)
    sources = {item.name: item for item in circuit_resources(selected).error_sources}
    later = ("angle_formation", "rotation_pruning", "gate_parameters", "identity_phase")
    assert all(sources[name].status == "bound" for name in later)
    assert sources["rotation_pruning"].value > 0
    hopping = r.onehot.hopping_blocks
    assert (hopping < 4 * method.num_steps) == (lower < -1.0)
    assert 1e-3 < distance <= sum(sources[name].value for name in later)


LAMBDA = 2.0**-1074


@pytest.mark.parametrize("objective, box, fields, message, admitted", [
    # A step of 3 * 2**-1074 would store the inexact half fl(delta/2) = 2 * 2**-1074 and make every projector
    # angle a third too large.
    (sp.Float(1e308) * x, (0.0, 1.0), dict(total_time=3 * LAMBDA, trotter_order=2), "step duration", None),
    # The butterfly of the binary table 1e308 x overflows. Only a Walsh construction forms it, so the explicit
    # dense diagonal, whose phases -x v stay below 8.75e307, plans (binary.PhaseTable.dense).
    (sp.Float(1e308) * x, (0.0, 1.0), dict(encoding="binary", boundary="periodic", num_grid_points=8),
     "Walsh butterfly sum", dict(binary_synthesis=BinarySynthesis(potential="dense_diagonal"))),
], ids=["odd_subnormal_step", "walsh_overflow"])
def test_planning_refuses_arithmetic_outside_the_normal_range(objective, box, fields, message, admitted):
    # Planning admits only normal nonzero durations, weights and kinetic terms, refuses every overflow and keeps
    # grid coordinates distinct (validation._normal_range, OneHotGrid). Objective data and the initial state are
    # the exception: their contributions below the range are omitted and charged. A step whose half is the
    # smallest normal number, and so exact, still plans when its products stay normal, here with table values
    # near 2**59 whose projector angles and identity phases are about 2**-963. Where ``admitted`` is given, the same
    # problem plans with a Method that does not form the refused quantity.
    method = QHD(**{**dict(num_grid_points=2, num_steps=1, trotter_order=1, rotation_threshold=0.0,
                           initial_state=UniformState(), schedule=QuadraticSchedule(gamma=0.0)), **fields})
    problem = Optimization(objective=objective, variables=(x,), bounds=(box,))
    with pytest.raises(ValueError, match=message):
        _plan(problem, method)
    if admitted is not None:
        assert _plan(problem, method.revise(**admitted))
    boundary = _plan(Optimization(objective=sp.Integer(2) ** 60 * x, variables=(x,), bounds=((0.0, 1.0),)),
                     method.revise(total_time=2.0**-1021, trotter_order=2, encoding="one_hot", boundary="dirichlet",
                                   num_grid_points=2, schedule=QuadraticSchedule(gamma=0.0)))
    assert {block.time_step for block in step_blocks(boundary, 0)} >= {2.0**-1022}


def test_a_chain_link_below_the_range_ends_the_structured_preparation():
    # A Gaussian start of width h/38 at grid index 3 of the eight interior points of [0, 1] stores the amplitude
    # exp(-722), about 2.8e-314, at index 4, so the fourth link's parameter theta/2 would fall below 2**-1022 and
    # the chain stops after three links. Oracle: the prefix prepares the tail norm r_3 at index 3 and nothing at
    # index 4, at distance at least alpha_4 from the target and at most sqrt(2) alpha_4/||alpha||, and the circuit
    # has three CRY gates.
    h = 1.0 / 9.0
    selected = _plan(Optimization(objective=(x - sp.Rational(1, 2)) ** 2, variables=(x,), bounds=((0.0, 1.0),)),
                     QHD(num_grid_points=8, num_steps=1, trotter_order=2, total_time=1.0, rotation_threshold=0.0,
                         initial_state=GaussianState(center=(4 * h,), widths=(h / 38,)),
                         schedule=QuadraticSchedule(gamma=0.0)))
    r = selected.reconstruction
    (alpha,) = r.initial_amplitudes
    assert [i for i, a in enumerate(alpha.array.tolist()) if a > 0][-1] == 4
    ((links, emitted, charge),) = r.range_omissions.chains
    assert (links, emitted) == (4, 3)
    assert alpha.array[4] <= charge <= 1.5 * alpha.array[4]
    assert construct_qhd(selected.blocks[0], (), None).count_ops()["cry"] == 3


def test_hopping_parameter_is_twice_the_stored_angle():
    # A zero objective on a box of width 3e153 with K = 2 stores the hopping coefficient -1/(4 h**2) =
    # -2.5e-307, and one first-order step of 5e307 the angle 2 dt c = -25, so the gate parameter 4 dt c is
    # -50 although 4 dt alone overflows. A parameter that is not finite is refused.
    selected = _plan(Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 3e153),)),
                     QHD(num_grid_points=2, num_steps=1, total_time=5e307, trotter_order=1, rotation_threshold=0.0,
                         initial_state=UniformState(), schedule=QuadraticSchedule(gamma=0.0)))
    circuit = QuantumCircuit(2)
    for block in raw_blocks(selected.reconstruction, selected.method, _grid(selected)):
        append_pauli_evolution_block(circuit, block)
    assert [item.operation.params[0] for item in circuit.data if item.operation.name == "xx_plus_yy"] == [-50.0]
    sources = {item.name: item for item in circuit_resources(selected).error_sources}
    assert (sources["gate_parameters"].status, sources["gate_parameters"].value) == ("bound", 0.0)
    overflow = {"terms": [{"pauli": "x0x1", "coefficient": 1.0}, {"pauli": "y0y1", "coefficient": 1.0}],
                "time_step": 1e308, "kind": "kinetic"}
    with pytest.raises(ValueError, match="not a finite binary64 number"):
        append_pauli_evolution_block(QuantumCircuit(2), overflow)


@pytest.mark.parametrize("center, widths", [((0.2, 0.7), (0.3, 0.4))])
def test_gaussian_preparation_entry_bounds_the_prepared_chain(center, widths):
    # The chain's statevector against the normalized product of exp(-(x - c)**2/(2 sigma**2)) on the binary64
    # grid points, at 50 digits. The entry adds the stored aggregate initial-state error, a norm allowance per
    # register and the chain's rounding, about 5e-15 here, far above the float64 simulation's own 1e-16.
    k = 3
    selected = _plan(Optimization(objective=x * y, variables=(x, y), bounds=((0.0, 1.0),) * 2),
                     QHD(num_grid_points=k, num_steps=1, initial_state=GaussianState(center=center, widths=widths)))
    r, grid = selected.reconstruction, _grid(selected)
    entry = {item.name: item for item in circuit_resources(selected).error_sources}["state_preparation"]
    circuit = QuantumCircuit(2 * k)
    append_amplitude_chain(circuit, grid, r.initial_amplitudes)
    state = Operator(circuit).data[:, 0]
    with mpmath.workdps(50):
        axes = []
        for j in range(2):
            values = [mpmath.exp(-(mpmath.mpf(grid.grid_value(j, i)) - center[j]) ** 2 / (2 * mpmath.mpf(widths[j]) ** 2))
                      for i in range(k)]
            axes.append([v / mpmath.sqrt(sum(u * u for u in values)) for v in values])
        target = {(1 << i) | (1 << (k + j)): axes[0][i] * axes[1][j] for i in range(k) for j in range(k)}
        distance = mpmath.sqrt(sum(abs(mpmath.mpc(a) - target.get(index, 0)) ** 2 for index, a in enumerate(state)))
    assert entry.status == "estimate" and distance <= entry.value
    assert entry.value < 1e-14


@pytest.mark.parametrize("schedule, rule", [(ShiftedCubicSchedule(s=0.5), "integrated"),
                                            (QuadraticSchedule(gamma=0.7), "midpoint")])
def test_coefficient_residual_matches_exact_integrals_and_midpoints(schedule, rule):
    # The stored exponents dt*weight against the exact nominal integrals (SymPy, over [k dt, (k + 1) dt]) or
    # midpoint values, weighted by the raw norms of the record. The record rounds each step upward and sums
    # exactly, so it lies at or above the exact sum and within one unit in the last place per step of it.
    problem = Optimization(objective=x**2 / 3 + x / 5 + sp.Rational(1, 7), variables=(x,), bounds=((-1.0, 1.0),))
    selected = _plan(problem, QHD(num_grid_points=3, num_steps=5, total_time=0.9, schedule=schedule,
                                  coefficient_rule=rule, init_prep="none"))
    bound = evolution_bound(selected)
    t = sp.Symbol("t")
    # The schedule parameter is read as the exact rational of its binary64 value, as in the finite model.
    if isinstance(schedule, ShiftedCubicSchedule):
        s = sp.Rational(schedule.s)
        a, b = (lambda u: 8 / (s + u) ** 3), (lambda u: 2 * u**3)
    else:
        g = sp.Rational(schedule.gamma)
        a, b = (lambda u: 1 / (1 + g * u**2)), (lambda u: 1 + g * u**2)
    dt = Fraction(0.9 / 5)
    mu_t, mu_v = Fraction(bound.kinetic_norm), Fraction(bound.potential_norm)
    exact = Fraction(0)
    for k, (_time, a_hat, b_hat) in enumerate(selected.reconstruction.step_weights):
        lower, upper = sp.Rational(k * dt), sp.Rational((k + 1) * dt)
        references = []
        for f in (a, b):
            if rule == "midpoint":
                value = sp.Rational(dt) * f(sp.Rational(k * dt + dt / 2))
            else:
                value = sp.integrate(f(t), (t, lower, upper))
            references.append(Fraction(int(sp.numer(value)), int(sp.denom(value))))
        exact += abs(dt * Fraction(a_hat) - references[0]) * mu_t + abs(dt * Fraction(b_hat) - references[1]) * mu_v
    assert bound.coefficient_status == "bound" and bound.evolution_status == "bound"
    assert exact <= Fraction(bound.coefficient_residual) <= exact * (1 + Fraction(1, 10**12)) + Fraction(1, 2**1060)


BINARY = x**2 / 3 - x * y / 5 + y / 7 + sp.Rational(1, 9)


@pytest.mark.parametrize("k, fields", [
    (2, dict()),
    (4, dict(binary_synthesis=BinarySynthesis(potential="walsh_rotations", kinetic_phase="walsh_rotations"))),
    (8, dict(binary_synthesis=BinarySynthesis(potential="dense_diagonal", kinetic_phase="dense_diagonal",
                                              aqft_cutoff=1, qft_bit_reversal="swap"), trotter_order=1)),
    (16, dict(binary_synthesis=BinarySynthesis(potential="walsh_rotations", kinetic_phase="walsh_rotations"))),
    (8, dict(kinetic_model="spectral", rotation_threshold=0.05)),
])
def test_binary_rotations_match_the_emitted_circuit(k, fields):
    # circuit_resources classifies the computed angles of every binary block. The oracle is the emitted
    # circuit, H-layer preparation included, expanded by Qiskit's gate definitions. The Plan's binary law
    # counts rotation gates, so with Walsh diagonals only it equals the arbitrary rotations plus the exact T
    # gates of the QFTs' r = 1 phases, which pins the b (b - 1)/2 controlled phases per QFT at b = 2 and 4.
    # A dense diagonal omits exactly zero multiplexor angles, so there the law is an upper bound. The emitted
    # magnitudes also give the same budgeted replacement and T estimate.
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=k, num_steps=2, total_time=0.45,
                 schedule=QuadraticSchedule(gamma=0.2), initial_state=UniformState(), **fields)
    selected = _plan(Optimization(objective=BINARY, variables=(x, y), bounds=((0.0, 1.7), (-1.0, 1.2))), method)
    record = circuit_resources(selected, synthesis_epsilon=1e-4)
    magnitudes = Counter()
    census = _census(construct_qhd(selected.blocks[0], (), None), magnitudes)
    assert (record.arbitrary_rotations, record.exact_t) == census
    emitted = RotationPopulation(magnitudes=tuple(sorted(magnitudes.items())), exact_t=census[1])
    assert record.synthesis == synthesis_projection(emitted, 1e-4)
    total = record.arbitrary_rotations + record.exact_t
    if all(b.synthesis == "walsh_rotations" for step in selected.reconstruction.steps for b in step):
        assert record.rotation_law.value == total
    assert record.rotation_law.value >= total and record.rotation_law.interpretation == "upper_bound"


def test_binary_census_and_evolution_states_are_admitted_with_the_model_before_synthesis(monkeypatch):
    # Inspection with a synthesis budget, which runs the census, and the binary product share QHD.max_bytes
    # between the caller's live data and the model's baseline (binary.BinaryModel), checked before any
    # synthesis. The binary product holds three D-entry
    # state owners and the phase buffers 16 (sum_potential E_t + K + E_max) (theory.run_binary_product), and
    # both refusals name the smallest admitting max_bytes, which then admits with the unchanged result. The
    # census reservation holds the Plan's tables, 8 bytes per entry, and at least the grouping and Python
    # population allowances H0 each, 1024 bytes per distinct magnitude, and the dense angle formation of the
    # 16-entry potential table, D_dense(16) = H0 + 112*16 + (16 + L(4))*8 = 67744 with L(4) = 36
    # (resources._census_sizes). The binary product also holds the Plan's source records
    # (method._binary_source_reservation).
    from nwqlib.algorithms.qhd import binary
    from nwqlib.algorithms.qhd.theory import run_binary_product

    method = QHD(encoding="binary", boundary="periodic", num_grid_points=4, num_steps=2, total_time=0.45,
                 binary_synthesis=BinarySynthesis(potential="dense_diagonal"))
    selected = _plan(Optimization(objective=BINARY, variables=(x, y), bounds=((0.0, 1.7), (-1.0, 1.2))), method)
    reference = circuit_resources(selected, synthesis_epsilon=1e-4)
    grid, r = _grid(selected), selected.reconstruction
    start = np.full(16, 0.25, dtype=np.complex128)
    evolved = run_binary_product(grid, selected.method, r, start.copy())
    synthesized = []
    original = binary.PhaseTable.synthesize
    monkeypatch.setattr(binary.PhaseTable, "synthesize",
                        lambda self, *a, **k: synthesized.append(1) or original(self, *a, **k))

    def limited(limit):
        return selected.model_copy(update={"method": selected.method.model_copy(update={"max_bytes": limit})})

    def smallest(call):
        limit = 1
        while True:
            try:
                return limit, call(limit)
            except ValueError as error:
                limit = int(re.search(r"max_bytes>=(\d+)", str(error)).group(1))

    limit, record = smallest(lambda b: circuit_resources(limited(b), synthesis_epsilon=1e-4))
    ignored = {"content_id", "plan_id"}
    assert record.model_dump(exclude=ignored) == reference.model_dump(exclude=ignored)
    synthesized.clear()
    with pytest.raises(ValueError, match=f"resource inspection.*requires {limit} bytes") as caught:
        circuit_resources(limited(limit - 1), synthesis_epsilon=1e-4)
    assert not synthesized
    entries = [t.values.array.size for t in r.support_values]
    census = int(re.search(r"\((\d+) held", str(caught.value)).group(1)) - 8 * sum(entries)
    magnitudes = len(_plan_population(selected).magnitudes)
    assert census >= 2 * 65536 + 1024 * magnitudes + 67744
    specs = [(e, len(t.support)) for e, t in zip(entries, r.support_values, strict=True)]
    source, _ = _binary_source_reservation(2, 4, specs, method.num_steps, method.trotter_order)
    held = source + 48 * 16 + 16 * (sum(entries) + 4 + max([4, *entries]))
    with pytest.raises(ValueError, match=rf"\({held} held \+ \d+ phase\)"):
        run_binary_product(grid, limited(1).method, r, start.copy())
    limit, state = smallest(lambda b: run_binary_product(grid, limited(b).method, r, start.copy()))
    assert np.array_equal(state, evolved)
    synthesized.clear()
    with pytest.raises(ValueError, match=f"requires {limit} bytes"):
        run_binary_product(grid, limited(limit - 1).method, r, start.copy())
    assert not synthesized


@pytest.mark.parametrize("k, model, rule, order", [(4, "finite_difference", "midpoint", 1),
                                                   (4, "spectral", "integrated", 2),
                                                   (4, "spectral", "midpoint", 1),
                                                   (2, "finite_difference", "midpoint", 2)])
def test_binary_evolution_bound_dominates_the_time_ordered_error(k, model, rule, order):
    # The binary steps with the physical phase, permuted from Qiskit's register order (variable j on qubits
    # j b to j b + b - 1) to lexicographic order, against U' = -i H(t) U integrated by DOP853 at rtol 1e-12,
    # atol 1e-14 on the stored tables and constant, with T_j = F^dagger diag(E_q) F from the analytic energies
    # 2 sin(pi q/K)**2/h**2 or 2 pi**2 q**2/L**2 (signed q). On these four-qubit cases the bound exceeds the
    # measured distance 2.1 to 12 times. K = 2 has both periodic neighbors at the same point.
    box = ((-1.0, 1.0), (-1.0, 1.0))
    method = QHD(encoding="binary", boundary="periodic", kinetic_model=model, num_grid_points=k, num_steps=4,
                 total_time=0.25, trotter_order=order, schedule=QuadraticSchedule(gamma=0.3), coefficient_rule=rule,
                 init_prep="none")
    objective = x**2 / 4 + y**2 / 8 + x * y / 3 + sp.Rational(7, 8)
    selected = _plan(Optimization(objective=objective, variables=(x, y), bounds=box), method)
    r = selected.reconstruction
    circuit = QuantumCircuit(r.width)
    append_binary_steps(circuit, r, method, _grid(selected), held_bytes=0)
    circuit.global_phase += r.physical_phase
    bits, length = k.bit_length() - 1, 2.0
    states = list(itertools.product(range(k), repeat=2))
    order_map = [s[0] | (s[1] << bits) for s in states]
    emitted = Operator(circuit).data[np.ix_(order_map, order_map)]
    q = np.arange(k)
    energies = (2 * np.sin(np.pi * q / k) ** 2 / (length / k) ** 2 if model == "finite_difference"
                else 2 * np.pi**2 * np.where(q < k / 2, q, q - k) ** 2 / length**2)
    fourier = np.exp(-2j * np.pi * np.outer(q, q) / k) / np.sqrt(k)
    one = fourier.conj().T @ np.diag(energies) @ fourier
    kinetic = np.kron(one, np.eye(k)) + np.kron(np.eye(k), one)
    values = np.full((k, k), r.constant)
    for table in r.support_values:
        entries = np.array(table.values).reshape((k,) * len(table.support))
        values += entries if table.support == (0, 1) else (
            entries[:, None] if table.support == (0,) else entries[None, :])
    potential = np.diag(values.ravel())

    def rhs(t, u):
        weight = 1 + 0.3 * t * t
        return (-1j * (kinetic / weight + weight * potential) @ u.reshape(k * k, k * k)).ravel()

    solution = scipy.integrate.solve_ivp(rhs, (0.0, 4 * (0.25 / 4)), np.eye(k * k, dtype=complex).ravel(),
                                         method="DOP853", rtol=1e-12, atol=1e-14)
    error = np.linalg.norm(emitted - solution.y[:, -1].reshape(k * k, k * k), 2)
    bound = evolution_bound(selected)
    assert (bound.domain, bound.formula) == ("full_binary_register", f"binary_{'first' if order == 1 else 'second'}_order")
    assert bound.hopping is None and bound.even_nested is None
    assert 1e-5 < error <= bound.evolution + 1e-10


def test_binary_error_ledger_takes_the_reconstruction_bounds():
    # A truncated QFT, a pruning threshold and dense kinetic diagonals, against exact QFTs with Walsh diagonals
    # and no threshold. The AQFT and pruning entries are the reconstruction's bounds, the dense diagonals'
    # wrap is unavailable, the H layer prepares the uniform start exactly and every angle reaches its gate
    # unchanged or halved exactly. The formation of the computed angles is a roundoff-sized bound. The same
    # exact circuit on the spectral model names its replacement of the finite-difference stencil as unavailable.
    problem = Optimization(objective=BINARY, variables=(x, y), bounds=((0.0, 1.7), (-1.0, 1.2)))
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=8, num_steps=2, total_time=0.45,
                 schedule=QuadraticSchedule(gamma=0.2), initial_state=UniformState())
    approximate = _plan(problem, method.revise(rotation_threshold=0.02, binary_synthesis=BinarySynthesis(
        potential="walsh_rotations", kinetic_phase="dense_diagonal", aqft_cutoff=1)))
    exact = _plan(problem, method.revise(binary_synthesis=BinarySynthesis(potential="walsh_rotations",
                                                                          kinetic_phase="walsh_rotations")))
    spectral = _plan(problem, exact.method.revise(kinetic_model="spectral"))
    sources = [{item.name: item for item in circuit_resources(p).error_sources}
               for p in (approximate, exact, spectral)]
    assert [entries["kinetic_model"].status for entries in sources] == [
        "not_applicable", "not_applicable", "unavailable"]
    r = approximate.reconstruction
    assert r.dropped_count > 0 and r.aqft_error_bound > 0
    assert (sources[0]["aqft"].status, sources[0]["aqft"].value) == ("bound", r.aqft_error_bound)
    assert (sources[0]["rotation_pruning"].status, sources[0]["rotation_pruning"].value) == (
        "bound", r.pruning_error_bound)
    assert sources[0]["diagonal_wrap"].status == "unavailable"
    assert sources[1]["aqft"].status == sources[1]["diagonal_wrap"].status == "not_applicable"
    assert sources[1]["rotation_pruning"].value == 0.0
    for entries in sources:
        assert entries["angle_formation"].status == "bound" and 0 < entries["angle_formation"].value < 1e-12
        assert (entries["state_preparation"].status, entries["state_preparation"].value) == ("bound", 0.0)
        assert (entries["gate_parameters"].status, entries["gate_parameters"].value) == ("bound", 0.0)
        assert entries["identity_phase"].status == "bound" and 0 < entries["identity_phase"].value < 1e-12


def _binary_formation_oracle(selected):
    """Evaluate independently what the binary angle_formation entry bounds, in exact rationals and 60-digit mpmath.

    Per block occurrence the largest diagonal phase error of the formed block against the exact stored-weight
    target, with the Walsh identity excluded and every mask counted before pruning, and per kinetic conjugation
    twice ||F^ - F|| of the QFT built from binary64 pi against the exact one. Walsh targets are the exact
    transforms of the stored tables and of the analytic energies 2 sin(pi |q|/K)**2/h**2 or 2 pi**2 q**2/L**2.
    """
    from nwqlib.algorithms.qhd.binary import kinetic_walsh, qft_gates, walsh_coefficients
    from nwqlib.algorithms.qhd.split_step import kinetic_eigenvalues

    method, r, grid = selected.method, selected.reconstruction, _grid(selected)
    k, model = grid.num_grid_points, method.kinetic_model
    bits = k.bit_length() - 1
    dt = Fraction(method.total_time / method.num_steps)
    tables = {tuple(t.support): [Fraction(v) for v in t.values.array.tolist()] for t in r.support_values}

    def address(values, size):
        # Lexicographic table (first variable most significant) to the register address sum_t n_t K**t.
        if size == 1:
            return list(values)
        return [values[(z % k) * k + z // k] for z in range(k * k)]

    def energies(j):
        h = mpmath.mpf(Fraction(grid.spacing(j)).numerator) / Fraction(grid.spacing(j)).denominator
        signed = [q if q < k // 2 else q - k for q in range(k)]
        if model == "spectral":
            return [2 * mpmath.pi**2 * q * q / (k * h) ** 2 for q in signed]
        return [2 * mpmath.sin(mpmath.pi * abs(q) / k) ** 2 / h**2 for q in signed]

    def walsh(values):
        size = len(values)
        return [sum((-1) ** bin(m & z).count("1") * values[z] for z in range(size)) / size for m in range(size)]

    def max_phase_error(target, computed):
        # max_z |sum_(m != 0) chi_m(z) (computed_m - target_m)| over the masks of one diagonal
        size = len(target)
        return max(abs(sum((-1) ** bin(m & z).count("1") * (computed[m] - target[m]) for m in range(1, size)))
                   for z in range(size))

    with mpmath.workdps(60):
        oracle = mpmath.mpf(0)
        for (_t, a, b), group in zip(r.step_weights, r.steps, strict=True):
            for block in group:
                kinetic = block.kind == "binary_kinetic"
                half = not kinetic and method.trotter_order == 2
                exact = (dt / 2 if half else dt) * Fraction(a if kinetic else b)
                stored_exponent = block.exponent
                if kinetic:
                    j = block.variables[0]
                    exact_values = energies(j)
                    stored = kinetic_eigenvalues(grid, j, model)
                    coefficients = kinetic_walsh(model, bits, grid.spacing(j))
                else:
                    support = tuple(block.variables)
                    exact_values = address(tables[support], len(support))
                    stored = np.array([float(v) for v in exact_values])
                    coefficients = walsh_coefficients(stored)
                exact_mp = mpmath.mpf(exact.numerator) / exact.denominator
                if block.synthesis == "walsh_rotations":
                    # Phase error of each string: theta^_m/2 against X c_m, with theta^_m = fl((2 x) c^_m).
                    angles = (2.0 * stored_exponent) * coefficients
                    target = [exact_mp * (mpmath.mpf(c.numerator) / c.denominator if isinstance(c, Fraction) else c)
                              for c in walsh(exact_values)]
                    computed = [mpmath.mpf(float(t)) / 2 for t in angles]
                    error = max_phase_error(target, computed)
                else:
                    phases = -stored_exponent * np.asarray(stored, dtype=float)
                    error = max(abs(mpmath.mpf(float(p)) + exact_mp * (mpmath.mpf(v.numerator) / v.denominator
                                                                       if isinstance(v, Fraction) else v))
                                for p, v in zip(phases.tolist(), exact_values, strict=True))
                oracle += min(2, error)

        # 2 ||F^ - F|| per conjugation, F^ with the binary64 angles ldexp(math.pi, -r) of the swap-free QFT.
        def qft_matrix(angle):
            matrix = mpmath.eye(k)
            for gate in qft_gates(bits, swaps=False):
                if gate[0] == "h":
                    q, step = gate[1], mpmath.eye(k)
                    for z in range(k):
                        for w in range(k):
                            if (z ^ w) & ~(1 << q) == 0:
                                step[z, w] = (-1 if (z >> q) & (w >> q) & 1 else 1) / mpmath.sqrt(2)
                else:
                    _name, value, c, t = gate
                    step = mpmath.diag([mpmath.expj(angle(value)) if (z >> c) & (z >> t) & 1 else 1 for z in range(k)])
                matrix = step * matrix
            return matrix

        exact_angle = lambda value: mpmath.pi * mpmath.mpf(value) / mpmath.mpf(math.pi)  # noqa: E731
        difference = qft_matrix(lambda value: mpmath.mpf(value)) - qft_matrix(exact_angle)
        norm = max(mpmath.svd_c(difference, compute_uv=False))
        return oracle + 2 * method.num_steps * grid.num_variables * norm


# One variable, one step. In each case one term of the bound carries the exact discrepancy (the terms of e_W
# and of the dense bound in the circuit_errors.binary_angle_formation docstring), so dropping that term puts the
# bound below the oracle. In the first case the K = 4 table (2**52, 2**52, 2**52, 2**52 + 1) on a box of width
# 2**100, which keeps the kinetic energies near 2**-196, has the butterfly output 0 for the mask 2, whose exact
# coefficient is -1/4, so at the exponent 1/8 the phase error 1/32 comes from the coefficient term alone. The
# other three take the one-bit kinetic coefficient -1/h**2 or energy 2/h**2 at h = 3 on its own, as Walsh
# rotations or as a dense phase diagonal, with gamma = 0, where the stored exponent x = fl(dt a) is exact. In
# the last case the midpoint weight a = 1/(1 + gamma t**2) makes x differ from the exact X = dt a, and the
# one-bit kinetic coefficient on an ordinary box carries an exact discrepancy of 6.7e-15, which the exponent
# term eps_x C covers: without it the Walsh bound is 5.5e-15, with it 7.9e-15. The floor is a round number
# below the exact discrepancy each case carries, so a case cannot pass on a vanishing oracle.
@pytest.mark.parametrize("objective, box, k, total_time, gamma, synthesis, floor", [
    (sp.Piecewise((sp.Integer(2) ** 52 + 1, x > 5 * sp.Integer(2) ** 97), (sp.Integer(2) ** 52, True)),
     (0.0, 2.0**100), 4, 0.125, 0.0, "walsh_rotations", 0.03),
    (sp.Integer(0), (0.0, 6.0), 2, 0.7, 0.0, "walsh_rotations", 7e-18),
    (sp.Integer(0), (0.0, 6.0), 2, 0.37, 0.0, "dense_diagonal", 1e-17),
    (sp.Integer(0), (0.0, 6.0), 2, 0.7, 0.0, "dense_diagonal", 1.5e-17),
    (sp.Integer(0), (0.0, 0.5508514739871617), 2, 2.271013125799355, 0.09950095546360811, "walsh_rotations",
     6e-15),
], ids=["walsh_coefficients", "walsh_angle_rounding", "dense_kinetic_energy", "dense_kinetic_rounding",
        "walsh_exponent"])
def test_binary_angle_formation_charges_each_component(objective, box, k, total_time, gamma, synthesis, floor):
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=k, num_steps=1, total_time=total_time,
                 trotter_order=1, kinetic_model="finite_difference", schedule=QuadraticSchedule(gamma=gamma),
                 initial_state=UniformState(), rotation_threshold=0.0,
                 binary_synthesis=BinarySynthesis(potential=synthesis, kinetic_phase=synthesis))
    selected = _plan(Optimization(objective=objective, variables=(x,), bounds=(box,)), method)
    formation = {item.name: item for item in circuit_resources(selected).error_sources}["angle_formation"]
    oracle = _binary_formation_oracle(selected)
    assert formation.status == "bound"
    assert floor < oracle <= formation.value


@pytest.mark.parametrize("k, objective", [(2, x**2 / 3 + x / 7), (4, BINARY)])
def test_binary_norm_inputs_bound_the_dense_commutators(k, objective):
    # Dense restricted operators on the K**d grid: T_j = F^dagger diag(E_q) F from the analytic
    # finite-difference energies 2 sin(pi |q|/K)**2/h**2, and V from the stored tables and constant. mu_T is
    # the largest energy, attained at |q| = K/2, so it is tight. On one variable at K = 2,
    # T = (I - X)/h**2 and the neighbor bounds of the doubled neighbor are tight as well.
    variables = (x,) if objective.free_symbols == {x} else (x, y)
    bounds = ((0.0, 1.7), (-1.0, 1.2))[: len(variables)]
    method = QHD(encoding="binary", boundary="periodic", kinetic_model="finite_difference", num_grid_points=k,
                 num_steps=1, total_time=0.3, init_prep="none")
    selected = _plan(Optimization(objective=objective, variables=variables, bounds=bounds), method)
    r, d = selected.reconstruction, len(variables)
    q = np.arange(k)
    signed = np.where(q < k / 2, q, q - k)
    fourier = np.exp(-2j * np.pi * np.outer(q, q) / k) / np.sqrt(k)
    kinetic = np.zeros((k**d, k**d), dtype=complex)
    for j, (lower, upper) in enumerate(bounds):
        length = upper - lower
        energies = 2 * np.sin(np.pi * np.abs(signed) / k) ** 2 / (length / k) ** 2
        one = fourier.conj().T @ np.diag(energies) @ fourier
        kinetic += np.kron(np.kron(np.eye(k**j), one), np.eye(k ** (d - 1 - j)))
    values = np.full((k,) * d, r.constant)
    for table in r.support_values:
        entries = np.array(table.values).reshape((k,) * len(table.support))
        values += entries if len(table.support) == d else (
            entries[:, None] if table.support == (0,) else entries[None, :])
    potential = np.diag(values.ravel())
    record, rtol = evolution_bound(selected), 1e-12
    tv = _commutator(kinetic, potential)
    dense = {"commutator": tv, "kinetic_nested": _commutator(kinetic, tv),
             "potential_nested": _commutator(potential, -tv), "kinetic_norm": kinetic, "potential_norm": potential}
    norms = {name: np.linalg.norm(matrix, 2) for name, matrix in dense.items()}
    for name, norm in norms.items():
        assert norm <= getattr(record, name) * (1 + rtol), name
    assert record.kinetic_norm == pytest.approx(norms["kinetic_norm"], rel=rtol)
    if k == 2:
        for name in ("commutator", "kinetic_nested", "potential_nested"):
            assert getattr(record, name) == pytest.approx(norms[name], rel=rtol), name


def test_default_estimate_reports_the_rotation_counts_of_both_encodings():
    # Both encodings publish their rotation law in the selected_logical basis, so the default resource estimate
    # reports one-hot and binary rotation counts side by side, each the count of rotation gates before angle
    # classification and so an upper bound.
    from nwqlib.resources import estimate

    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    for method, interpretation in ((QHD(num_grid_points=4, num_steps=2, initial_state=UniformState()), "upper_bound"),
                                   (QHD(num_grid_points=4, num_steps=2, initial_state=UniformState(),
                                        encoding="binary", boundary="periodic"), "upper_bound")):
        selected = _plan(problem, method)
        (law,) = [law for selection in selected.construction.selections for law in selection.resource_laws
                  if law.metric == "arbitrary_rotations"]
        quantity = estimate(selected.construction).quantity("arbitrary_rotations")
        assert (law.basis, quantity.interpretation) == ("selected_logical", interpretation)
        assert (quantity.fact.value.numerator, quantity.fact.value.denominator) == (law.value, 1)


def test_a_classical_binary_product_plan_runs_at_its_smallest_planning_limit():
    # Planning admits the classical binary product's model beside the run's source records, state buffers and
    # phase buffers (theory.run_binary_product), so a Plan admitted at its smallest max_bytes also runs.
    from nwqlib.scientist import solve

    method = QHD(encoding="binary", boundary="periodic", num_grid_points=4, num_steps=60, total_time=2.0,
                 theory_flavor="ir_product")
    problem = Optimization(objective=BINARY, variables=(x, y), bounds=((0.0, 1.0),) * 2)
    limit = 1
    while True:
        try:
            selected = plan(problem, method=method.revise(max_bytes=limit), execution="classical", seed=7)
            break
        except ValueError as error:
            required = max(int(n) for n in re.findall(r"QHD\(max_bytes>?=(\d+)\)", str(error)))
            assert required > limit
            limit = required
    solve(selected)


def test_binary_circuit_bytes_are_recorded_at_planning_and_checked_at_construction():
    # Planning records the coarse construction estimate W as the preparation workspace without refusing on
    # it, so a Plan with max_bytes = W - 1 is made. construct_qhd checks W before it allocates, refuses one
    # byte below with that amount, and builds at W.
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=16, num_steps=2, total_time=0.45,
                 binary_synthesis=BinarySynthesis(potential="dense_diagonal"))
    problem = Optimization(objective=BINARY, variables=(x, y), bounds=((0.0, 1.7), (-1.0, 1.2)))
    workspace = max(w.bytes for block in _plan(problem, method).blocks
                    for w in block.record.workspace if w.purpose == "preparation")
    below = _plan(problem, method.model_copy(update={"max_bytes": workspace - 1}))
    with pytest.raises(ValueError, match=f"construction and use requires {workspace} bytes"):
        construct_qhd(below.blocks[0], (), None)
    circuit = construct_qhd(_plan(problem, method.model_copy(update={"max_bytes": workspace})).blocks[0], (), None)
    assert circuit.num_qubits == 8


def test_onehot_circuit_bytes_are_recorded_at_planning_and_checked_at_construction(tmp_path):
    # Planning admits the construction work and records the byte estimate W, so a Plan with max_bytes = W - 1 is
    # made. construct_qhd checks W before it allocates, for a loaded Plan too, refuses one byte below and builds at
    # W. Eighty second-order steps on K = 8 make W exceed the bytes that planning itself checks.
    problem = Optimization(objective=x, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=8, num_steps=80, total_time=0.3, trotter_order=2)
    selected = _plan(problem, method)
    (workspace,) = [w.bytes for w in selected.blocks[0].record.workspace if w.purpose == "preparation"]
    below = _plan(problem, method.revise(max_bytes=workspace - 1))
    loaded = load_plan(save_plan(below, ArchiveFiles(tmp_path, 4_000_000)), ArchiveFiles(tmp_path, 4_000_000))
    for candidate in (below, loaded):
        with pytest.raises(ValueError, match=f"needs about {workspace} bytes, exceeding max_bytes={workspace - 1}"):
            construct_qhd(candidate.blocks[0], (), None)
    assert construct_qhd(_plan(problem, method.revise(max_bytes=workspace)).blocks[0], (), None).num_qubits == 8
    # More steps on two variables: the Plan records its larger W and is made at W - 1 without building a circuit.
    pair = Optimization(objective=x * y + x + y, variables=(x, y), bounds=((-1.0, 1.0),) * 2)
    longer = QHD(num_grid_points=8, num_steps=32, total_time=0.3, trotter_order=2)
    (estimate,) = [w.bytes for w in _plan(pair, longer).blocks[0].record.workspace if w.purpose == "preparation"]
    made = _plan(pair, longer.revise(max_bytes=estimate - 1))
    assert [w.bytes for w in made.blocks[0].record.workspace if w.purpose == "preparation"] == [estimate]


def test_one_hot_stage_bounds_cap_an_overflowing_angle_sum_at_two():
    # Zero objective, 64 Dirichlet points in (0, 6.5e-152) and one first-order step of length 8: each of the 63
    # hopping angles is finite and their absolute sum exceeds the largest binary64 number. From the last grid point
    # only the last hop acts, and its stored angle differs from the intended exponent by about 3.8e290 rad, so the
    # step differs from the intended one by more than 1 even up to a global phase. Formation takes the stage cap 2.
    method = QHD(num_grid_points=64, num_steps=1, total_time=8.0, trotter_order=1, schedule=QuadraticSchedule(gamma=0))
    zero = Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 6.5e-152),))
    overflow = _plan(zero, method)
    blocks = step_blocks(overflow, 0)
    assert sum(abs(Fraction(b.angle)) for b in blocks) > Fraction(np.finfo(float).max)
    h = Fraction(_grid(overflow).spacing(0))
    delta = abs(Fraction(blocks[-1].angle) + Fraction(8) / (2 * h * h))
    with mpmath.workdps(355):
        gap = mpmath.sqrt(2 - 2 * abs(mpmath.cos(mpmath.mpf(delta.numerator) / delta.denominator)))
    entries = {e.name: e for e in circuit_resources(overflow, synthesis_epsilon=1e-3).error_sources}
    assert 1 < gap <= overflow.reconstruction.onehot.angle_formation == entries["angle_formation"].value == 2.0
    # A threshold that prunes every hop: the dropped angles' sum exceeds the binary64 range and is not recorded,
    # and both pruning totals take the cap.
    pruned = _plan(zero, method.revise(rotation_threshold=1e307)).reconstruction
    assert (pruned.dropped_count, pruned.dropped_angle_sum, pruned.pruning_error_bound,
            pruned.onehot.rotation_pruning) == (63, None, 2.0, 2.0)
    # Control on (0, 1): the angles sum within range and keep the a priori bound 3u/(1 - 6u) sum |lambda|.
    control = _plan(Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 1.0),)), method)
    u = Fraction(1, 2**53)
    exact = 3 * u / (1 - 6 * u) * sum(abs(Fraction(b.angle)) for b in step_blocks(control, 0))
    assert exact <= Fraction(control.reconstruction.onehot.angle_formation) <= exact * (1 + Fraction(1, 2**40))


def test_one_hot_projector_assignments_beyond_range_give_an_infinite_phase_allowance():
    # 8e307 x on 16 points: the scaled projector angles sum beyond the binary64 range while their signed identity
    # phases cancel. The assignment sum is recorded as beyond range, the identity-phase and formation stages take
    # the cap 2, and a kept state, whose phase allowance is then infinite, is refused at planning.
    problem = Optimization(objective=sp.Float(8e307) * x, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=16, num_steps=1, total_time=1.0)
    selected = _plan(problem, method)
    assert selected.reconstruction.onehot.phase_assignments is None
    entries = {e.name: e for e in circuit_resources(selected, synthesis_epsilon=1e-3).error_sources}
    assert entries["identity_phase"].value == entries["angle_formation"].value == 2.0
    with pytest.raises(ValueError, match="keep_state needs the physical phase"):
        _plan(problem, method.revise(keep_state=True))


def test_one_hot_derivation_storage_follows_the_largest_table():
    # A two-entry table is charged its two entries, not a full tile, so the two-qubit, two-step Plan plans, lists its
    # step, counts its rotations and builds its 13 gates within 1 MiB, with the default Plan's error sources.
    problem = Optimization(objective=x, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=2, num_steps=2, total_time=0.3)
    regular = _plan(problem, method)
    bounded = _plan(problem, method.revise(max_bytes=1 << 20))
    assert step_blocks(bounded, 0) == step_blocks(regular, 0)
    reference, limited = (circuit_resources(p, synthesis_epsilon=1e-3) for p in (regular, bounded))
    assert (limited.arbitrary_rotations, limited.exact_t, limited.error_sources) == (
        reference.arbitrary_rotations, reference.exact_t, reference.error_sources)
    assert len(construct_qhd(bounded.blocks[0], (), None).data) == 13


@pytest.mark.parametrize("k, d, provider, order, threshold", [
    (64, 2, None, 2, 0.0),
    (32, 2, ("mcphase", 2), 2, 0.0),
    (6, 4, ("diagonal_synthesis", 0), 1, 0.0),
    (6, 4, ("diagonal_synthesis", 0), 2, 10.0),
    (256, 2, ("mcphase", 2), 1, 0.0),
    (256, 2, ("mcphase", 2), 1, 10.0),
])
def test_one_hot_derivation_peak_stays_within_its_storage(k, d, provider, order, threshold):
    # The traced peak of deriving one step's blocks and of the planning summary, without a table, with a 1,024- or
    # 1,296-entry table and with a full 65,536-entry tile, all kept or all pruned, against derivation_bytes.
    grid = OneHotGrid(tuple(f"v{j}" for j in range(d)), ((-1.0, 1.0),) * d, k, True, "dirichlet")
    tables = () if provider is None else (
        SimpleNamespace(support=tuple(range(d)), values=FrozenArray(np.linspace(1.0, 2.0, k**d))),)
    schedule = OneHotSchedule(grid, tables=tables, providers={} if provider is None else {d: provider},
                              step_weights=((0.1, 1.0, 1.0),), constant=0.0, total_time=0.2, num_steps=1,
                              trotter_order=order, rotation_threshold=threshold)
    storage = derivation_bytes(k, d, (len(t.support) for t in tables))

    def blocks():
        for _ in schedule.blocks(0):
            pass

    for run in (blocks, schedule.summary):
        run()
        tracemalloc.start()
        try:
            run()
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak <= storage


def test_summary_objectives_equal_pointwise_objective_at(monkeypatch):
    # The summary gathers each point's constant and table entries and sums them with fsum, so every value,
    # the candidate, its objective, the mean and the marginals equal an evaluation by objective_at at each
    # observed point, for dense probabilities and for sparse counts, without calling objective_at.
    from math import fsum
    from nwqlib.algorithms.qhd import method as qhd_method

    method = QHD(encoding="binary", boundary="periodic", num_grid_points=8, num_steps=2, total_time=0.45)
    selected = _plan(Optimization(objective=BINARY, variables=(x, y), bounds=((0.0, 1.7), (-1.0, 1.2))), method)
    r = selected.reconstruction
    rng = np.random.default_rng(3)
    probabilities = rng.random((8, 8)) * (rng.random((8, 8)) < 0.6)
    probabilities /= probabilities.sum()
    points = np.array(sorted({(int(i), int(j)) for i, j in rng.integers(0, 8, size=(20, 2))}))
    counts = rng.integers(1, 50, size=len(points)).astype(np.int64)
    pointwise = qhd_method.objective_at
    for weights, shots, kept in ((probabilities, None, None), (counts, int(counts.sum()), points)):
        cells = ([tuple(p) for p in np.argwhere(weights > 0)] if kept is None
                 else [tuple(map(int, p)) for p in kept])
        flat = weights[weights > 0] if kept is None else weights
        expected = {cell: pointwise(r, cell, 8) for cell in cells}
        with monkeypatch.context() as patch:
            patch.setattr(qhd_method, "objective_at", lambda *a: pytest.fail("pointwise objective_at"))
            summary = qhd_method._summarize(selected, weights, 0.0, None, 0.0, shots=shots, points=kept)
        best = min(cells, key=lambda cell: (expected[cell], cell))
        assert summary["candidate_indices"] == best and summary["value"] == expected[best]
        valid = int(flat.sum()) / shots if shots else fsum(flat.tolist())
        probability = (flat / shots) if shots else flat
        assert summary["expected_objective"] == qhd_method._array_mean(
            [expected[cell] for cell in cells], probability, valid)
        for axis in range(2):
            for i in range(8):
                block = [weight for cell, weight in zip(cells, flat.tolist(), strict=True) if cell[axis] == i]
                assert summary["marginals"][axis][i] == (sum(block) / shots if shots else fsum(block))


def test_binary_rotation_census_is_admitted_only_when_a_synthesis_budget_runs_it(monkeypatch):
    # The default Plan and resource query classify no angle, so a work limit one unit below the census
    # still plans and reports the rotation law and the error sources, while a synthesis budget, which runs
    # the census, is refused with its work.
    from nwqlib.algorithms.qhd import resources

    problem = Optimization(objective=x * y + x**2 + y, variables=(x, y), bounds=((-1.0, 1.0),) * 2)
    method = QHD(encoding="binary", boundary="periodic", num_grid_points=16, num_steps=3, total_time=0.4,
                 binary_synthesis=BinarySynthesis(potential="dense_diagonal"))
    selected = plan(problem, method=method)
    sizes, original = [], resources._census_sizes
    monkeypatch.setattr(resources, "_census_sizes", lambda *a, **k: sizes.append(original(*a, **k)) or sizes[-1])
    default = circuit_resources(selected)
    assert sizes == [] and (default.arbitrary_rotations, default.exact_t) == (None, None)
    circuit_resources(selected, synthesis_epsilon=1e-3)
    (work, _), = sizes
    bounded = plan(problem, method=method.revise(max_work=work - 1))
    record = circuit_resources(bounded)
    assert (record.rotation_law.value, record.rotation_law.interpretation) == (default.rotation_law.value, "upper_bound")
    assert record.error_sources == default.error_sources
    with pytest.raises(ValueError, match=f"binary rotation census requires {work} work units"):
        circuit_resources(bounded, synthesis_epsilon=1e-3)


def _aer_positions(circuit):
    from qiskit_aer import AerSimulator
    from nwqlib.backends.qiskit_aer import _AerDecompose, _lower_aer_circuit

    lowered, _ = _lower_aer_circuit(circuit, _AerDecompose(AerSimulator().target))
    return len(lowered.data)


@pytest.mark.parametrize("fields", [
    dict(num_grid_points=20, num_steps=10),
    dict(num_grid_points=4, num_steps=2, init_prep="qiskit_state_preparation"),
    dict(num_grid_points=16, num_steps=4, encoding="binary", boundary="periodic"),
    dict(num_grid_points=16, num_steps=2, encoding="binary", boundary="periodic",
         binary_synthesis=BinarySynthesis(qft_bit_reversal="swap")),
    dict(num_grid_points=32, num_steps=2, encoding="binary", boundary="periodic",
         binary_synthesis=BinarySynthesis(aqft_cutoff=2)),
])
def test_construction_memory_counts_the_native_positions_of_the_built_circuit(fields):
    # Aer's lowering of the built circuit is the independent count of what the Run's estimate multiplies. Hopping
    # blocks, projector providers, the preparation recipes, QFT swaps and an approximate QFT all change that count.
    from nwqlib.blocks.selection import NATIVE_POSITION_BYTES

    selected = _plan(Optimization(objective=sp.cos(x), variables=(x,), bounds=((-1.0, 1.0),)),
                     QHD(total_time=.5, **fields))
    block = selected.blocks[0]
    native = _aer_positions(construct_qhd(block, (), None))
    assert native / 2 <= block._native_bytes / NATIVE_POSITION_BYTES <= 2 * native
    (workspace,) = [w.bytes for w in block.record.workspace if w.purpose == "preparation"]
    assert block._native_bytes + block._definition_bytes == workspace


def test_binary_memory_counts_a_walsh_diagonal_at_most_once_per_gate():
    # A weight-two parity rotation is one native RZZ, which the recorded CX and rotation counts overstate.
    from nwqlib.algorithms.qhd.native import _binary_positions

    selected = _plan(Optimization(objective=sp.cos(x) + sp.cos(y) + x * y, variables=(x, y), bounds=((-1.0, 1.0),) * 2),
                     QHD(num_grid_points=8, num_steps=4, total_time=.5, encoding="binary", boundary="periodic"))
    native = _aer_positions(construct_qhd(selected.blocks[0], (), None))
    assert native <= _binary_positions(selected.reconstruction, selected.method, 2) <= 2 * native


def test_prepare_refuses_a_long_circuit_before_deriving_a_block_and_after_loading(tmp_path, monkeypatch):
    # K = 20 and N = 8000 flatten to about 3.5 million native positions, and a fresh process grew by 3.5 GB to build
    # them. With a limit of 60 MiB the Run refuses before the first block is derived, for a loaded Plan as well.
    from nwqlib.algorithms.qhd import native

    def forbidden(*args, **kwargs):
        raise AssertionError("a block was derived before the memory comparison")

    monkeypatch.setattr(native, "raw_blocks", forbidden)
    selected = _plan(Optimization(objective=sp.cos(x), variables=(x,), bounds=((-1.0, 1.0),)),
                     QHD(num_grid_points=20, num_steps=8000, max_bytes=60 * 2**20))
    loaded = load_plan(save_plan(selected, ArchiveFiles(tmp_path, 4_000_000)), ArchiveFiles(tmp_path, 4_000_000))
    for candidate in (selected, loaded):
        with pytest.raises(ValueError, match=rf"preparing this circuit needs about \d+ bytes.*max_bytes={60 * 2**20}"):
            prepare(candidate, settings="first", progress=False)
