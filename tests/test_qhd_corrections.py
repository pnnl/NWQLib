"""Independent small algebra/scalar and injected population correction witnesses."""

from math import exp, sqrt

import pytest
import sympy as sp

from nwqlib.algorithms.qhd import QHD, QuadraticSchedule
from nwqlib.problems import Optimization
from test_qhd_workflow import make_plan


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
