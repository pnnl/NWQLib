"""Independent small algebra/scalar and injected population correction witnesses."""

from math import exp, fsum, nextafter, sqrt
from unittest.mock import Mock

import numpy as np
import pytest
import sympy as sp

from nwqlib.algorithms.qhd import QHD, QuadraticSchedule
from nwqlib.problems import Optimization
from nwqlib.algorithms.qhd import method as owner
from test_qhd_workflow import make_plan


def test_default_schedule_has_decay_without_expanding_the_selected_work():
    options = QHD()
    schedule = options.schedule
    assert type(schedule) is QuadraticSchedule and schedule.gamma > 0
    assert schedule.potential_weight(1.) == pytest.approx(1.3, rel=2e-15, abs=0.)
    assert schedule.potential_weight(2.) == pytest.approx(2.2, rel=2e-15, abs=0.)
    # gamma = 0 is the time-independent model, whose weights and step averages
    # are exactly one at every time.
    undamped = QuadraticSchedule(gamma=0.)
    assert undamped.potential_weight(0.) == undamped.potential_weight(100.) == 1.
    assert undamped.kinetic_weight(0.) == undamped.kinetic_weight(100.) == 1.
    assert undamped.kinetic_integral(3., 5.) == undamped.potential_integral(3., 5.) == 2.
    damped_plan = make_plan(method=options)
    constant_plan = make_plan(method=options.revise(schedule=undamped))
    def shape(chosen):
        return tuple(tuple((block.kind, block.support) for block in step)
                     for step in chosen.reconstruction.steps)
    assert shape(damped_plan) == shape(constant_plan)
    assert sum(reg.width for reg in damped_plan.construction.program.registers) == 2
    for invalid in (-.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            QuadraticSchedule(gamma=invalid)


def test_actual_second_order_taylor_and_forward_falsifier():
    """Exact 3x3 Taylor coefficient; no evolution or reference state."""
    x = sp.Symbol("x")
    binding = Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 4.0),))
    options = QHD(
        num_grid_points=3,
        num_steps=1,
        total_time=1.0,
        schedule=QuadraticSchedule(gamma=0.0),
        theory_flavor="ir_product",
        rotation_threshold=0.0,
    )
    expected_h = sp.Matrix(
        [
            [0, -sp.Rational(1, 2), 0],
            [-sp.Rational(1, 2), 0, -sp.Rational(1, 2)],
            [0, -sp.Rational(1, 2), 0],
        ]
    )

    def coefficients(order):
        plan = make_plan(binding, options.revise(trotter_order=order))
        first, second = sp.zeros(3), sp.zeros(3)
        for block in plan.reconstruction.steps[0]:
            assert block.kind == "kinetic"
            a, b = block.support
            term = sp.zeros(3)
            term[a, b] = term[b, a] = (
                2 * sp.Rational(block.coefficient) * sp.Rational(block.time_step)
            )
            second += -term * term / 2 - term * first
            first += term
        return first, second

    first, second = coefficients(2)
    assert first == expected_h
    assert second == -(expected_h**2) / 2
    old_first, old_second = coefficients(1)
    assert old_first == expected_h and old_second != -(expected_h**2) / 2


def test_half_angle_pruning_strict_threshold_and_zero_work():
    from nwqlib.algorithms.qhd.grid import OneHotGrid
    from nwqlib.algorithms.qhd.kinetic import KineticCompiler

    grid = OneHotGrid(("x",), ((0.0, 4.0),), 3)
    compiler = KineticCompiler(rotation_threshold=0.25)
    blocks = compiler.compile(grid, dt=1.0, kinetic_weight=1.0, trotter_order=2)
    assert [b.angle for b in blocks] == [-0.25, -0.5, -0.25]
    assert compiler._dropped_nonzero_contribution_count == 0
    compiler = KineticCompiler(rotation_threshold=nextafter(0.25, 1.0))
    blocks = compiler.compile(grid, dt=1.0, kinetic_weight=1.0, trotter_order=2)
    assert [b.angle for b in blocks] == [-0.5]
    assert compiler._dropped_nonzero_contribution_count == 2
    assert compiler._dropped_absolute_angle_sum == 0.5
    compiler = KineticCompiler(rotation_threshold=0.0)
    assert compiler.compile(grid, dt=1.0, kinetic_weight=0.0, trotter_order=2) == ()
    assert compiler._dropped_nonzero_contribution_count == compiler._dropped_absolute_angle_sum == 0
    # Small physical time changes angle magnitude, not the existence of the
    # three selected Strang occurrences. Default pruning must keep them.
    compiler = KineticCompiler()
    small = compiler.compile(grid, dt=1e-20, kinetic_weight=1.0, trotter_order=2)
    assert len(small) == 3 and all(block.angle != 0.0 for block in small)
    assert compiler._dropped_nonzero_contribution_count == 0


def test_actual_second_order_occurrence_resource_increment():
    x = sp.Symbol("x")
    binding = Optimization(objective=sp.Integer(0), variables=(x,), bounds=((0.0, 4.0),))
    options = QHD(
        num_grid_points=3,
        num_steps=2,
        total_time=1.0,
        schedule=QuadraticSchedule(gamma=0.0),
        rotation_threshold=0.0,
    )
    first = make_plan(binding, options.revise(trotter_order=1))
    second = make_plan(binding, options)
    delta = 2  # N*d*floor((K-1)/2), independent of stored blocks.
    a, b = first.construction.selections[0], second.construction.selections[0]
    assert b.construction_work - a.construction_work == 7 * delta
    assert b.resource_laws[0].value - a.resource_laws[0].value == 2 * delta
    assert (
        sum(map(len, second.reconstruction.steps)) - sum(map(len, first.reconstruction.steps))
        == delta
    )
    # Each stored block occurrence is one direct action of the one-hot ir_product host law
    # (theory.onehot_product_sizes), so the added occurrences add work and not workspace.
    shape_a = owner.restricted_sizes(
        3, 1, 0, (first.reconstruction.step_weights, first.reconstruction.steps), "ir_product", (), 3
    )
    shape_b = owner.restricted_sizes(
        3, 1, 0, (second.reconstruction.step_weights, second.reconstruction.steps), "ir_product", (), 3
    )
    assert shape_b[0] > shape_a[0] and shape_b[1] == shape_a[1]


@pytest.mark.parametrize(
    "values,probabilities,expected",
    [
        ((-2.0, 3.0), (0.2, 0.3), fsum((-2.0 * 0.2, 3.0 * 0.3)) / 0.5),
        ((-3.0, -3.0), (0.2, 0.3), -3.0),
        (
            (
                float.fromhex("0x1.fffffffffffffp+1023"),
                nextafter(float.fromhex("0x1.fffffffffffffp+1023"), 0.0),
            ),
            (0.5 + 1e-13,) * 2,
            nextafter(float.fromhex("0x1.fffffffffffffp+1023"), 0.0),
        ),
        ((-1e308, nextafter(1e308, 0.0)), (0.5, 0.5), -float(2**970)),
        ((1e-200, 3e-200), (nextafter(0.0, 1.0),) * 2, 2e-200),
        ((nextafter(3.0, 0.0), 3.0), (0.1, 0.1), 3.0),
        ((-1e300, 3.0, 3.0), (nextafter(0.0, 1.0), 0.1, 0.1), 3.0),
        (
            (-float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023")),
            (nextafter(0.0, 1.0), 1.0 + 1e-13),
            float.fromhex("0x1.fffffffffffffp+1023"),
        ),
        (
            (-float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023")),
            (1.0 + 1e-13, nextafter(0.0, 1.0)),
            -float.fromhex("0x1.fffffffffffffp+1023"),
        ),
    ],
)
def test_weighted_mean_analytic_witnesses(values, probabilities, expected, monkeypatch):
    plan = make_plan(method=QHD(num_grid_points=len(values)))
    objective = Mock(side_effect=values)
    monkeypatch.setattr(owner, "objective_at", objective)
    actual = owner._summarize(
        plan, np.array(probabilities), 0.0, fsum(probabilities), 0.0, points=np.arange(len(probabilities)).reshape(-1, 1)
    )
    assert actual["expected_objective"] == expected
    assert objective.call_count == len(values)
    assert min(values) <= actual["expected_objective"] <= max(values)
    if values[0] < 0 < values[1] and abs(values[0]) > 1e300:
        from fractions import Fraction

        exact = sum(
            Fraction(v) * Fraction(p) for v, p in zip(values, probabilities, strict=True)
        ) / sum(map(Fraction, probabilities))
        assert actual["expected_objective"] == float(exact)
        if probabilities == (0.5, 0.5):
            assert actual["expected_objective"] == fsum(
                v * p for v, p in zip(values, probabilities, strict=True)
            )


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_analysis_masses_are_bounded_by_their_execution_window(execution):
    """Observed masses may exceed one only within the roundoff window of the executions that
    produced them. That is the circuit receipt's window for quantum data and, for the host
    kernel, its recorded expm_multiply window, which is the fixed floor on this one-step K = 3
    grid.
    """
    import nwqlib
    from nwqlib._validation import NUMERICAL_RELATION_RTOL

    x = sp.Symbol("x")
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    result = nwqlib.solve(problem, method=QHD(num_grid_points=3, num_steps=1, total_time=0.1),
                          execution=execution, seed=7)
    window = max(receipt.probability_window for receipt in result.data.receipts)
    assert (window == NUMERICAL_RELATION_RTOL) == (execution == "classical")
    result.validate_analysis_masses(result.data)
    for excess, admitted in ((window / 2, True), (2 * window, False)):
        scale = (1 + excess) / result.observed_mass
        forged = result.revise(valid_mass=result.valid_mass * scale, invalid_mass=result.invalid_mass * scale,
                               observed_mass=1 + excess, candidate_probability=result.candidate_probability * scale,
                               most_probable_probability=result.most_probable_probability * scale,
                               probability_maximizer_probability=result.probability_maximizer_probability * scale,
                               marginals=result.marginals.array * scale)
        forged = forged._attach(result.plan, result.data)
        if admitted:
            forged.validate_analysis_masses(result.data)
        else:
            with pytest.raises(ValueError, match="roundoff window of its execution"):
                forged.validate_analysis_masses(result.data)


def test_expm_multiply_window_follows_its_stated_relation():
    """Hand evaluation of the host state budget and the mass and tie windows derived from it.

    The owners are ``_validation.expm_multiply_state_error`` (delta),
    ``state_mass_window`` (mass) and
    ``method._host_tie_window`` (ties). Two calls of norm N = 19.8 = 2 theta_55
    on rows of three nonzeros have theta = 9.9 and S = 55 * 2 = 110 each, so
    each call charges ``c = N + sqrt(2)*2*3*N*e**9.9 + 110*(2*e**9.9 + 7)``
    and the centering ``N``. With the uniform state's 2u start vector
    (``initial_state.restricted_state_error``) and one phase product,
    ``delta = (2 + 2*(c + N) + 5)*u``. The mass window for D = 39 adds the
    evaluation ``(D + 4)*u``, the tie window ``5u``. Each expected value
    repeats a handful of roundings of terms of the same sign, so rel=1e-14
    with abs=0 separates a missing centering or phase term (relative 2.6e-6
    and 3.2e-7 of delta). A zero generator keeps only the start vector and
    the per-product charges, which stay below the mass floor. The tie window
    has no floor.
    """
    from nwqlib._validation import (
        NUMERICAL_RELATION_RTOL,
        UNIT_ROUNDOFF as u,
        expm_multiply_state_error,
        probability_difference_window,
        state_mass_window,
    )

    calls = [(19.8, 3)] * 2
    c = 19.8 + sqrt(2) * 2 * 3 * 19.8 * exp(9.9) + 110 * (2 * exp(9.9) + 7)
    delta = (2 + 2 * (c + 19.8) + 5) * u
    assert expm_multiply_state_error(calls, start=2.0, phase_multiplications=1) == pytest.approx(
        delta, rel=1e-14, abs=0)
    mass = 2 * delta + delta**2 + (39 + 4) * u * (1 + delta) ** 2
    assert state_mass_window(expm_multiply_state_error(calls, start=2.0, phase_multiplications=1), 39) == pytest.approx(
        mass, rel=1e-14, abs=0)
    tie = 2 * delta + delta**2 + 5 * u * (1 + delta) ** 2
    assert probability_difference_window(delta, 5 * u) == pytest.approx(tie, rel=1e-14, abs=0)
    # Nonnegative state error cannot cancel; zero evaluation contributes exactly zero.
    assert probability_difference_window(1e155, 0.0) == float("inf")
    assert probability_difference_window(0.5, 0.0) == 1.25
    zero = (2 + 55 * 9) * u
    assert expm_multiply_state_error([(0.0, 3)], start=2.0) == pytest.approx(zero, rel=1e-14, abs=0)
    assert state_mass_window(expm_multiply_state_error([(0.0, 3)], start=2.0), 39) == NUMERICAL_RELATION_RTOL
    with pytest.raises(ValueError, match="finite and nonnegative"):
        expm_multiply_state_error([(-1.0, 1)], start=2.0)


def test_pooled_masses_add_the_running_sum_roundoff_of_their_bins():
    """QHD pools exact bins with running sums, which can add (M + 2)*u beyond the largest chunk
    window for M bin values. A mass inside that allowance is admitted and one beyond it is not."""
    import nwqlib
    from nwqlib._validation import UNIT_ROUNDOFF
    from nwqlib.algorithms.qhd.records import QHDAnalysis

    x = sp.Symbol("x")
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    result = nwqlib.solve(problem, method=QHD(num_grid_points=3, num_steps=1, total_time=0.1),
                          execution="quantum", seed=7)
    chunks = result.data.observations.chunks
    pooled = QHDAnalysis.pooled_summation_roundoff(chunks)
    assert pooled == (sum(chunk.histogram().entries for chunk in chunks) + 2) * UNIT_ROUNDOFF
    window = max(receipt.probability_window for receipt in result.data.receipts)
    for excess, admitted in ((window + pooled / 2, True), (window + 2 * pooled, False)):
        scale = (1 + excess) / result.observed_mass
        forged = result.revise(valid_mass=result.valid_mass * scale, invalid_mass=result.invalid_mass * scale,
                               observed_mass=1 + excess, candidate_probability=result.candidate_probability * scale,
                               most_probable_probability=result.most_probable_probability * scale,
                               probability_maximizer_probability=result.probability_maximizer_probability * scale,
                               marginals=result.marginals.array * scale)
        forged = forged._attach(result.plan, result.data)
        if admitted:
            forged.validate_analysis_masses(result.data)
        else:
            with pytest.raises(ValueError, match="roundoff window of its execution"):
                forged.validate_analysis_masses(result.data)


def test_live_amplitude_analysis_bounds_masses_before_returning(monkeypatch):
    """A stored QHD state carries no mass scalars for the chunk-receipt check, so the live
    analysis must apply the receipt window itself."""
    import numpy as np
    import nwqlib
    from nwqlib.algorithms.qhd import decoding

    x = sp.Symbol("x")
    problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    result = nwqlib.solve(problem, method=QHD(num_grid_points=3, num_steps=1, total_time=0.1, keep_state=True),
                          execution="quantum", seed=7)
    window = max(receipt.probability_window for receipt in result.data.receipts)
    decode = decoding.decode_statevector_arrays
    # Rescale the executed state to a chosen squared norm, then decode it as usual.
    for squared_norm, admitted in ((1 + window / 2, True), (1 + 2 * window, False), (4.0, False)):
        def rescaled(state, grid, bits=None, target=squared_norm):
            state = np.asarray(state)
            return decode(np.sqrt(target / np.vdot(state, state).real) * state, grid, bits)

        monkeypatch.setattr(decoding, "decode_statevector_arrays", rescaled)
        if admitted:
            analysis = result.plan.method.analyze(result.plan, result.data, settings={})
            assert analysis.observed_mass == pytest.approx(squared_norm, rel=0, abs=window / 4)
        else:
            with pytest.raises(ValueError, match="roundoff window of its execution"):
                result.plan.method.analyze(result.plan, result.data, settings={})
