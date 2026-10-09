"""QHD augmented-Lagrangian layer on 2-, 3-by-3 and 4-by-4 grids against independent grid oracles.

The trajectory oracles minimize the effective objective exactly over the grid, with ties to the
lexicographically smallest index tuple. That is what the QHD candidate (``best_observed``) returns
under exact readout when every grid point has nonzero probability, so every such test first checks
that premise on the kept states. The recorded trajectories of the plan's grid oracle (Birgin and
Martinez doi:10.1137/1.9781611973365, Algorithm 4.1 and Eqs. (10.7)-(10.8)) have binary-exact points,
multipliers, penalties and table values, so those assertions are exact equalities. The default rule,
``most_probable``, is compared with an independently written restricted evolution instead. The layer
with box refinement in each round is checked on a 7-by-7 grid against a distance bound to the
analytic KKT point, evaluated in exact rational arithmetic from the recorded rounds.
"""

from fractions import Fraction
import itertools
from math import fsum, sqrt
from unittest.mock import patch

import numpy as np
import pytest
import scipy.linalg
import sympy as sp

from nwqlib.algorithms.qhd import (
    QHD,
    AbsorbedBound,
    AugmentedLagrangian,
    BoxRefinement,
    KineticGroundState,
    MultiplierBounds,
    QuadraticSchedule,
    UniformState,
    constrained_grid_minimum,
    load_augmented_lagrangian,
    plan_augmented_lagrangian,
    resume_augmented_lagrangian,
    run_resources,
    solve_augmented_lagrangian,
)
from nwqlib.algorithms.qhd.objective import node_count
from nwqlib.execution import ExecutionLimits
from nwqlib.problems import ConstrainedOptimization

UNIT_ROUNDOFF = 2.0**-53
x, y, z = sp.symbols("x y z")
F = (x - 1) ** 2 + (y - 1) ** 2
BOX = ((-1.0, 1.0), (-1.0, 1.0))
AXIS = (-0.5, 0.0, 0.5)  # interior grid of [-1, 1] with K = 3


def _qhd(**fields):
    return QHD(**{"num_grid_points": 3, "num_steps": 1, "total_time": 0.5, "keep_state": True, **fields})


def _solve(problem, *, execution="classical", qhd=None, shots=None, limits=None, **options):
    options.setdefault("inner_point", "best_observed")
    return solve_augmented_lagrangian(problem, qhd=qhd or _qhd(), options=AugmentedLagrangian(**options),
                                      execution=execution, shots=shots, limits=limits, seed=11,
                                      progress=False)


def _onehot_index(indices, k):
    """Full-register basis index of a grid-index tuple: variable j's point i is qubit j*K + i."""
    return sum(1 << (j * k + i) for j, i in enumerate(indices))


def _assert_every_grid_point_observed(result):
    """The premise under which best_observed reads the grid point with the least binary64 table value of each L_k."""
    for inner in result.results:
        state = inner.data.artifact(inner.artifact).array
        k, d = inner.plan.method.num_grid_points, len(inner.plan.problem.variables)
        if inner.plan.execution == "quantum":
            state = state[[_onehot_index(i, k) for i in itertools.product(range(k), repeat=d)]]
        assert np.min(np.abs(state) ** 2) > 0


def _feasible_grid_minimum(objective, residuals, axes, tolerance):
    """Least objective among the grid points whose residuals pass, ties to the smallest index tuple."""
    best = None
    for indices in itertools.product(*(range(len(axis)) for axis in axes)):
        point = tuple(axis[i] for axis, i in zip(axes, indices))
        violation = max((abs(r(*point)) if kind == "eq" else max(r(*point), 0.0) for kind, r in residuals),
                        default=0.0)
        if violation <= tolerance and (best is None or objective(*point) < best[0]):
            best = (objective(*point), point)
    return best


# Plan evidence al_grid_oracle.out.txt (exact grid minimization, penalty test Eq. (4.9), stopping
# Eqs. (10.7)-(10.8)): chosen points, multipliers used, penalties and the last tentative multiplier.
# The oracle stops before its last penalty test, which gives next_penalty by hand: the last measure
# is 0 <= tau * 0.5 in the first two cases and 0.125 = tau * 0.5 in the third, so rho stays.
ORACLE = {
    "equality": dict(constraints=dict(equalities=(x + y,)), options=dict(feasibility_tolerance=1e-9),
                     points=[(0.5, 0.5), (0.0, 0.5), (0.0, 0.0)], multipliers=[0.0, 1.0, 1.5],
                     penalties=[1.0, 1.0, 2.0], estimate=1.5, next_penalty=2.0),
    "phr": dict(constraints=dict(inequalities=(x + y,)), options={},
                points=[(0.5, 0.5), (0.0, 0.5), (0.0, 0.0)], multipliers=[0.0, 1.0, 1.5],
                penalties=[1.0, 1.0, 2.0], estimate=1.5, next_penalty=2.0),
    "complementarity": dict(constraints=dict(inequalities=(x + y - sp.Rational(3, 2),)),
                            options=dict(inequality_multipliers=(4.0,)),
                            points=[(0.0, 0.0), (0.0, 0.5), (0.5, 0.5), (0.5, 0.5)],
                            multipliers=[4.0, 2.5, 1.5, 0.5], penalties=[1.0, 1.0, 2.0, 4.0], estimate=0.0,
                            next_penalty=4.0),
}


@pytest.mark.parametrize("case", sorted(ORACLE))
def test_best_observed_trajectories_match_the_grid_oracle(case):
    oracle = ORACLE[case]
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, **oracle["constraints"])
    result = _solve(problem, **oracle["options"])
    _assert_every_grid_point_observed(result)
    equality = "equalities" in oracle["constraints"]
    assert result.termination == "feasible_complementary"
    assert [item.evaluation.point for item in result.iterations] == oracle["points"]
    assert [item.penalty for item in result.iterations] == oracle["penalties"]
    used = [item.equality_multipliers if equality else item.inequality_multipliers
            for item in result.iterations]
    assert used == [(value,) for value in oracle["multipliers"]]
    last = result.last.evaluation
    estimate = last.tentative_equality_multipliers if equality else last.tentative_inequality_multipliers
    assert estimate == (oracle["estimate"],) and last.next_penalty == oracle["next_penalty"]
    # The reported best point is the feasible grid minimum by independent enumeration, and with
    # equal objectives the earliest round.
    residual = ("eq", lambda a, b: a + b) if equality else (
        "ineq", sp.lambdify((x, y), oracle["constraints"]["inequalities"][0]))
    value, point = _feasible_grid_minimum(sp.lambdify((x, y), F), [residual], (AXIS, AXIS), 1e-9)
    assert result.candidate == point and result.objective == value
    assert result.best.iteration == oracle["points"].index(point)
    # Record linkage: the table value of L_k at the chosen point equals f, h, g, the multipliers and
    # rho recombined as
    # L = f + lambda h + (rho/2) h**2 + (rho/2) max(0, g + mu/rho)**2 - (rho/2) (mu/rho)**2,
    # whose last part is the PHR constant. The table sums the expanded terms of L_k (QHD planning
    # expands about the origin here) and the recombination sums the parts above. In these three
    # fixtures every operand is a dyadic rational with few significant bits: the coordinates
    # -1/2, 0, 1/2, the objective (x - 1)**2 + (y - 1)**2 with unit scales, the residuals x + y and
    # x + y - 3/2, and the multipliers, penalties and PHR shifts of the oracle trajectories above.
    # Every product, quotient and partial sum of both evaluation orders is then exact in binary64,
    # so the two sides agree exactly. This argument covers these fixtures only. A linkage check
    # for non-dyadic data would need the error of each residual, scale division and coefficient,
    # not only a count of the final additions.
    for item, inner in zip(result.iterations, result.results):
        e, rho = item.evaluation, item.penalty
        parts = [e.objective_normalized]
        for lam, h in zip(item.equality_multipliers, e.equality_residuals_normalized):
            parts += [lam * h, rho / 2 * h * h]
        for mu, g in zip(item.inequality_multipliers, e.inequality_residuals_normalized):
            parts += [rho / 2 * max(0.0, g + mu / rho) ** 2, -rho / 2 * (mu / rho) ** 2]
        assert e.effective_value_source == "table"
        assert Fraction(e.effective_value) == sum(map(Fraction, parts)) and e.effective_value == fsum(parts)


def _restricted_probabilities(values, total_time, steps, gamma):
    """Independent 9-by-9 restricted QHD evolution on AXIS x AXIS, returning the grid probabilities.

    Each step applies ``exp(-i dt H(t))`` at its midpoint t with ``H = a(t) K + b(t) diag(values)``,
    ``b = 1 + gamma t**2`` and ``a = 1/b`` (the quadratic schedule with the midpoint rule, QHDOPT
    arXiv:2409.03121v1 Sec. 2.1), K the central-difference ``-Delta/2`` with spacing 1/2 (Leng et al.
    arXiv:2303.01471v1 Eqs. (F.7), (F.9)), from the default initial state, the ground state of the
    kinetic matrix K. On this Dirichlet interior grid of 3 points the one-variable ground state is
    proportional to ``sin(pi (i + 1)/4)``, which normalizes to (1/2, 1/sqrt(2), 1/2), and the
    two-variable state is its tensor product. Index tuples are lexicographic with x most significant.
    """
    spacing = 0.5
    one = (np.diag(np.full(3, 1 / spacing**2)) + np.diag(np.full(2, -0.5 / spacing**2), 1)
           + np.diag(np.full(2, -0.5 / spacing**2), -1))
    kinetic = np.kron(one, np.eye(3)) + np.kron(np.eye(3), one)
    ground = np.array([0.5, 1 / sqrt(2), 0.5])
    state = np.kron(ground, ground).astype(complex)
    dt = total_time / steps
    for step in range(steps):
        weight = 1 + gamma * ((step + 0.5) * dt) ** 2
        state = scipy.linalg.expm(-1j * dt * (kinetic / weight + weight * np.diag(values))) @ state
    return np.abs(state) ** 2


def test_default_most_probable_rounds_follow_an_independent_restricted_evolution():
    """The default rule on f + x/4 subject to x + y = 0, round by round against the evolution above.

    The x/4 term breaks the x <-> y symmetry, whose tied probabilities roundoff would decide. Each
    round first checks that the top two probabilities of the independent evolution differ by more
    than the recorded tie window (``method._host_tie_window``), so the chosen point is the unique
    mode, then follows the multiplier and penalty rules independently.
    """
    objective = F + x / 4
    problem = ConstrainedOptimization(objective=objective, variables=(x, y), bounds=BOX, equalities=(x + y,))
    qhd = QHD(num_grid_points=3, num_steps=4, total_time=2.0)
    result = solve_augmented_lagrangian(problem, qhd=qhd, options=AugmentedLagrangian(feasibility_tolerance=1e-9),
                                        execution="classical", seed=11, progress=False)
    assert result.record.options.inner_point == "most_probable"
    points = list(itertools.product(range(3), repeat=2))
    f = sp.lambdify((x, y), objective)
    lam, rho, previous = 0.0, 1.0, None
    for item in result.iterations:
        # The multipliers and penalties are dyadic, so the comparison is exact.
        assert (item.equality_multipliers, item.penalty) == ((lam,), rho)
        values = []
        for i, j in points:
            h = AXIS[i] + AXIS[j]
            values.append(f(AXIS[i], AXIS[j]) + lam * h + rho / 2 * h * h)
        p = _restricted_probabilities(np.array(values), 2.0, 4, qhd.schedule.gamma)
        first, second = np.argsort(-p, kind="stable")[:2]
        e = item.evaluation
        assert e.tie_window is not None and p[first] - p[second] > e.tie_window
        assert e.indices == points[first] and e.tie_deficit == 0.0
        h = AXIS[e.indices[0]] + AXIS[e.indices[1]]
        rho_next = rho if previous is None or abs(h) <= 0.25 * previous else min(2 * rho, 1e9)
        lam, rho, previous = lam + rho * h, rho_next, abs(h)
    assert result.termination == "feasible_complementary" and result.iterations[-1].evaluation.infeasibility == 0
    # The mode of round 0 is not the grid minimizer of L_0, the point best_observed would take.
    assert result.iterations[0].evaluation.indices != result.results[0].candidate_indices


def test_mode_or_mean_takes_the_off_grid_mean_or_falls_back_to_the_grid_point():
    """The replication rule compares L_k at the most probable point with L_k at the valid-mass mean.

    With f = (z - 1/5)**2 and the inactive z <= 9/10 on the grid {-1/2, 0, 1/2}, the mean lies closer
    to 1/5 than any grid point, so the round takes it, evaluates L_k there and can beat the feasible grid
    minimum. With f containing sqrt(z (z - 1)) on the grid {0, 1}, f is real on the grid but not at the
    mean, which lies strictly between 0 and 1, so the round takes the grid point and says why.
    """
    options = AugmentedLagrangian(inner_point="mode_or_mean")
    problem = ConstrainedOptimization(objective=(z - sp.Rational(1, 5)) ** 2, variables=(z,),
                                      bounds=((-1.0, 1.0),), inequalities=(z - sp.Rational(9, 10),))
    result = solve_augmented_lagrangian(problem, qhd=QHD(num_grid_points=3, total_time=1.0), options=options,
                                        execution="classical", seed=11, progress=False)
    e, inner = result.iterations[0].evaluation, result.results[0]
    assert (e.kind, e.effective_value_source, e.indices, e.probability) == ("valid_mean", "evaluated", None, None)
    assert e.point == inner.position_mean and e.effective_value < inner.most_probable_objective
    # Machine-precision window: at the mean m the inequality is inactive with mu = 0, so its PHR part is
    # exactly 0 and L_k = (m - 1/5)**2. With d = m - 1/5, the computed m - fl(1/5) errs by at most
    # u (1/5 + |d|) and the square adds u d**2, so the value lies within u (2 |d| (1/5 + |d|) + d**2) of
    # d**2 to first order. Twice that covers the second-order terms.
    d = Fraction(e.point[0]) - Fraction(1, 5)
    window = 2 * UNIT_ROUNDOFF * float(2 * abs(d) * (Fraction(1, 5) + abs(d)) + d * d)
    assert abs(e.effective_value - float(d * d)) <= window
    reference = constrained_grid_minimum(result)
    assert reference.best_feasible and reference.gap < 0
    fallback = ConstrainedOptimization(objective=sp.sqrt(z * (z - 1)) + (z - 1) ** 2, variables=(z,),
                                       bounds=((-1.0, 2.0),), inequalities=(z - 2,))
    result = solve_augmented_lagrangian(fallback, qhd=QHD(num_grid_points=2), options=options,
                                        execution="classical", seed=11, progress=False)
    e, inner = result.iterations[0].evaluation, result.results[0]
    assert 0 < inner.position_mean[0] < 1
    assert e.kind == "grid_point" and e.point == inner.most_probable_coordinates
    assert "is not a finite real value" in e.mean_unavailable
    # Both terms were attempted at the mean and evaluated again at the grid point.
    assert result.iterations[0].resources.layer_evaluations == 4


def _equality_oracle(objective, residual, rounds, tolerance):
    """Brute-force AL with one equality on AXIS x AXIS (the plan's al_grid_oracle.py rules).

    Returns the chosen index tuples and penalties of each round.
    """
    lam, rho, previous, history = 0.0, 1.0, None, []
    for _ in range(rounds):
        def effective(i):
            point = (AXIS[i[0]], AXIS[i[1]])
            h = residual(*point)
            return objective(*point) + lam * h + rho / 2 * h**2
        index = min(itertools.product(range(3), repeat=2), key=lambda i: (effective(i), i))
        h = residual(AXIS[index[0]], AXIS[index[1]])
        history.append((index, rho))
        if abs(h) <= tolerance:
            break
        measure = abs(h)
        rho_next = rho if previous is None or measure <= 0.25 * previous else min(2 * rho, 1e9)
        lam, rho, previous = lam + rho * h, rho_next, measure
    return history


def test_off_grid_equality_ends_at_the_iteration_limit_with_the_least_violation_as_best():
    """x + y = 3/10 meets no point of the 3-by-3 grid, so the multiplier oscillates and rho doubles.

    Sixteen rounds reach rho = 2**14, whose classical planning passes the default admission with
    these settings. lambda accumulates rounded residuals, so it is compared with its exact rational
    value within a derived window.
    """
    residual = x + y - sp.Rational(3, 10)
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, equalities=(residual,))
    result = _solve(problem, qhd=_qhd(total_time=0.1), feasibility_tolerance=1e-3, max_iterations=16)
    _assert_every_grid_point_observed(result)
    assert result.termination == "iteration_limit" and len(result.iterations) == 16
    oracle = _equality_oracle(sp.lambdify((x, y), F), sp.lambdify((x, y), residual), 16, 1e-3)
    assert [(item.evaluation.indices, item.penalty) for item in result.iterations] == oracle
    assert result.iterations[-1].penalty == 2.0**14
    # Machine-precision window around the exact lambda*_{k+1} = lambda*_k + rho_k (x_k + y_k - 3/10)
    # of the recorded points. The
    # recorded h_k = fl(fl(x + y) - fl(3/10)) errs by at most u (3/10 + |h*_k|), rho_k is a power of
    # two, and each addition rounds once more, so to first order
    # |lambda_k - lambda*_k| <= u sum_{i<k} (rho_i (3/10 + |h*_i|) + |lambda*_{i+1}|). Twice that
    # covers the second-order terms.
    exact, allowance = Fraction(0), 0.0
    for item in result.iterations:
        assert abs(item.equality_multipliers[0] - float(exact)) <= 2 * UNIT_ROUNDOFF * allowance
        a, b = item.evaluation.point
        h = Fraction(a) + Fraction(b) - Fraction(3, 10)
        exact += Fraction(item.penalty) * h
        allowance += item.penalty * (0.3 + abs(float(h))) + abs(float(exact))
    violations = [item.evaluation.infeasibility for item in result.iterations]
    assert min(violations) > 1e-3
    best = min(result.iterations, key=lambda item: (item.evaluation.infeasibility, item.evaluation.objective,
                                                    item.iteration))
    assert result.best is best and result.last is result.iterations[-1]
    assert "Least-violation point, no feasible round" in str(result)


def test_cumulative_limits_and_inner_failure_keep_the_completed_rounds():
    equality = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, equalities=(x + y,))
    # One circuit per quantum round: two circuits, or 128 shots at 64 per round, fund two rounds, and
    # the third, which the unlimited run needs, is refused before it starts.
    cases = ((None, ExecutionLimits(max_total_circuits=2), "max_total_circuits has 0 left of 2"),
             (64, ExecutionLimits(max_total_shots=128), "max_total_shots has 0 left of 128"))
    for shots, limits, message in cases:
        limited = _solve(equality, execution="quantum", qhd=_qhd(keep_state=False),
                         feasibility_tolerance=1e-9, shots=shots, limits=limits)
        assert limited.termination == "budget_exhausted" and len(limited.iterations) == 2
        assert message in limited.record.failure
        assert limited.resources.circuit_preparations == limited.resources.circuit_attempts == 2
        assert limited.resources.shots == 2 * (shots or 0)
    # QHD's classical planning admits the evolution work that grows with rho. A max_work between the
    # largest admitted work of the first k rounds and the work of round k rejects round k's planning.
    offgrid = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX,
                                      equalities=(x + y - sp.Rational(3, 10),))
    options = dict(qhd=_qhd(total_time=0.1, keep_state=False), feasibility_tolerance=1e-3, max_iterations=16)
    reference = _solve(offgrid, **options)
    work = [inner.plan.construction.kernels[0].invocation_work for inner in reference.results]
    k = next(k for k in range(2, len(work)) if work[k] > max(work[:k]))
    failing = _solve(offgrid, **{**options, "qhd": _qhd(total_time=0.1, keep_state=False,
                                                         max_work=(max(work[:k]) + work[k]) // 2)})
    assert failing.termination == "inner_failed"
    assert failing.record.failure.startswith("ValueError") and "set QHD(max_work=" in failing.record.failure
    assert len(failing.iterations) == k + 1 and failing.results[k] is None
    stopped = failing.iterations[k]
    assert stopped.evaluation is None and stopped.plan_id is None

    def trajectory(run):
        return [(item.evaluation.point, item.penalty, item.equality_multipliers) for item in run.iterations[:k]]

    assert trajectory(failing) == trajectory(reference)
    # The failed planning's table work is unknown, so the total is unknown rather than a partial sum.
    assert failing.resources.table_evaluations is None
    assert dict(failing.resources.unavailable)["table_evaluations"] \
        == f"round {k}: planning raised before selecting a Plan"
    reason = dict(failing.resources.unavailable)["table_evaluations"]
    assert f"unknown ({reason}) table evaluations" in str(failing)
    # In the first round no round has completed, so QHD's own error propagates: a configuration it rejects
    # and a max_work below the admitted work of L_0.
    with pytest.raises(ValueError, match="keep_state requires exact amplitude acquisition"):
        _solve(equality, execution="quantum", shots=64, qhd=_qhd(keep_state=True), feasibility_tolerance=1e-9)
    with pytest.raises(ValueError, match=r"set QHD\(max_work="):
        _solve(offgrid, **{**options, "qhd": _qhd(total_time=0.1, keep_state=False, max_work=work[0] - 1)})


def test_quantum_rounds_follow_the_classical_trajectory_and_count_their_shots():
    equality = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, equalities=(x + y,))
    exact = _solve(equality, execution="quantum", feasibility_tolerance=1e-9)
    _assert_every_grid_point_observed(exact)
    assert exact.results[0].plan.reconstruction.width == 6
    assert [item.evaluation.point for item in exact.iterations] == ORACLE["equality"]["points"]
    assert all(item.resources.cx is not None for item in exact.iterations)
    sampled = _solve(equality, execution="quantum", shots=64, qhd=_qhd(keep_state=False),
                     feasibility_tolerance=1e-9)
    for item, inner in zip(sampled.iterations, sampled.results):
        observed = set()
        for chunk in inner.data.observations.chunks:
            assert chunk.observation.kind == "counts"
            for index in chunk.histogram().index_list():
                cells = [[i for i in range(3) if index >> (3 * j + i) & 1] for j in range(2)]
                if all(len(cell) == 1 for cell in cells):
                    observed.add(tuple(cell[0] for cell in cells))
        assert item.evaluation.indices in observed
        assert item.resources.shots == item.resources.completed_shots == 64
    assert sampled.resources.shots == 64 * len(sampled.iterations)
    # With counts the summary qualifies the copied status as the inner Result does.
    counts = _solve(equality, execution="quantum", shots=64, qhd=_qhd(keep_state=False), feasibility_tolerance=1e-9,
                    inner_point="most_probable", max_iterations=1)
    status = counts.best.evaluation.mode_status
    assert status == counts.results[0].mode_status != "unavailable"
    qualifier = "Counts give no proof of the population mode."
    assert f"Empirical mode status: {status}. {qualifier}" in str(counts.results[0]).splitlines()
    assert f"Empirical mode status of the selected inner point: {status}. {qualifier}" in str(counts).splitlines()


def test_feasible_complementary_can_stop_away_from_the_feasible_grid_minimum():
    """Grid {0, 1}, f = (z - 1)**2/10, g = z - 1 <= 0, mu = 1/2, rho = 1 (plan 3.4).

    L(0) = 1/10 - 1/8 < L(1) = 0, so the exact inner minimizer is 0, where mu+ = 0 and both tests hold,
    while the feasible grid minimum is 1. At 0 the box-normalized coordinate is u = 1/3 on [-1, 2] and
    the gradient 3 * (f'(0) + mu+ g'(0)) = -0.6, so r_stat = |1/3 - P(1/3 + 0.6)| = 0.6.
    """
    problem = ConstrainedOptimization(objective=sp.Rational(1, 10) * (z - 1) ** 2, variables=(z,),
                                      bounds=((-1.0, 2.0),), inequalities=(z - 1,))
    result = _solve(problem, qhd=_qhd(num_grid_points=2), inequality_multipliers=(0.5,), stationarity=True)
    _assert_every_grid_point_observed(result)
    assert result.termination == "feasible_complementary" and result.candidate == (0.0,)
    reference = constrained_grid_minimum(result)
    assert reference.point == (1.0,) and reference.feasible_points == 2 and reference.best_feasible
    # f(0) is fl(1/10) * 1 and f(1) = 0 exactly.
    assert reference.gap == 0.1
    stationarity = result.best.evaluation.stationarity
    # Five roundings on values below one (the quotient, the product, the sum and the two differences).
    assert stationarity == pytest.approx(0.6, rel=0, abs=8 * UNIT_ROUNDOFF)
    assert "Optimality was not assessed" in str(result)


def test_stationarity_uses_the_tentative_multipliers_when_the_safeguard_truncates():
    """grad_x L_k = grad f + J_h^T lambda+ + J_g^T mu+ at each chosen point, differentiated independently.

    The inner Plan's own objective L_k is differentiated with SymPy and compared with the formula in
    the recorded tentative multipliers, while the safeguard clips the multipliers of the next round.
    """
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, equalities=(x + y,),
                                      inequalities=(x**2 + y**2 - sp.Rational(1, 4),))
    bounds = MultiplierBounds(equality_lower=-0.5, equality_upper=0.5, inequality_upper=0.125)
    result = _solve(problem, feasibility_tolerance=1e-9, multiplier_bounds=bounds, stationarity=True,
                    max_iterations=3)
    truncated = [False, False]
    for item, inner in zip(result.iterations, result.results):
        e = item.evaluation
        a, b = e.point
        (lam,), (mu,) = e.tentative_equality_multipliers, e.tentative_inequality_multipliers
        terms = [(2 * (a - 1), lam, 2 * mu * a), (2 * (b - 1), lam, 2 * mu * b)]
        objective = inner.plan.problem.objective
        # Machine-precision windows. Each side of a gradient component sums three terms that take
        # at most two roundings each, so the two sides differ by at most 16 u times the sum of the
        # terms' magnitudes S_j. The box width 2 doubles that, and the maps to u and the clipped
        # difference, which are 1-Lipschitz, add at most 8 u on values below 2.
        for variable, parts in zip((x, y), terms):
            actual = float(sp.diff(objective, variable).subs({x: a, y: b}))
            assert abs(actual - fsum(parts)) <= 16 * UNIT_ROUNDOFF * fsum(map(abs, parts))
        gradient = [2 * fsum(parts) for parts in terms]  # D = diag(2, 2) on [-1, 1]**2
        u = [(a + 1) / 2, (b + 1) / 2]
        expected = max(abs(ui - min(max(ui - gi, 0.0), 1.0)) for ui, gi in zip(u, gradient))
        window = 32 * UNIT_ROUNDOFF * max(fsum(map(abs, parts)) for parts in terms) + 8 * UNIT_ROUNDOFF
        assert abs(e.stationarity - expected) <= window
        assert e.next_equality_multipliers == (min(max(lam, -0.5), 0.5),)
        assert e.next_inequality_multipliers == (min(mu, 0.125),)
        assert e.equality_truncated == (lam != e.next_equality_multipliers[0],)
        truncated = [truncated[0] or e.equality_truncated[0], truncated[1] or e.inequality_truncated[0]]
    assert truncated == [True, True]
    assert result.record.safeguard_used


@pytest.mark.parametrize("scaled_term", ["constraint", "objective", "both_out_of_range"])
def test_rescaling_a_term_leaves_the_normalized_run_and_rescales_the_multipliers(scaled_term):
    """Writing a term t as c t with scale c s_t changes no decision of the normalized run.

    For g <= 0 written as c g <= 0 with s_g -> c s_g, mu~ is unchanged and the original-unit
    multiplier mu = (s_f/s_g) mu~ scales by 1/c. For f written as c f with s_f -> c s_f, every
    original-unit multiplier scales by c. The factors c = 1/4 and c = 4 are powers of two, and the
    coefficients of F and g are integers and dyadic rationals, which stay exact binary64 numbers when
    ``constrained._normalized`` divides by the binary64 scale, so the normalized values are bitwise
    equal and the comparisons are exact. A power of two alone does not give this
    (``constrained._update``). With tolerance 1/2 the normalized run stops at round 2 on
    complementarity 1/2, while the unnormalized complementarity of the rewritten constraint would
    already pass at round 0.

    ``both_out_of_range`` writes f as 2**600 f and g as 2**-600 g with the matching scales, so
    s_f/s_g = 2**1200 exceeds the binary64 range. The nonzero initial multiplier then has no normal
    original-unit value, which stops no run because every decision uses the normalized multipliers
    (``constrained._setup``). Explicit ``multipliers`` conversion raises, while the summary and report
    show the original-unit estimates as unavailable.
    """
    g = x + y - sp.Rational(3, 2)
    common = dict(inequality_multipliers=(4.0,), feasibility_tolerance=0.5, stationarity=True)
    base = _solve(ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(g,)),
                  **common)
    assert "Best feasible point" in str(base)
    if scaled_term == "constraint":
        problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(g / 4,))
        scaled, factor = _solve(problem, inequality_scales=(0.25,), **common), 4.0
    elif scaled_term == "objective":
        problem = ConstrainedOptimization(objective=4 * F, variables=(x, y), bounds=BOX, inequalities=(g,))
        scaled, factor = _solve(problem, objective_scale=4.0, **common), 4.0
    else:
        c = sp.Integer(2) ** 600
        problem = ConstrainedOptimization(objective=c * F, variables=(x, y), bounds=BOX, inequalities=(g / c,))
        scaled, factor = _solve(problem, objective_scale=float(c), inequality_scales=(float(1 / c),), **common), None
    assert base.termination == scaled.termination == "feasible_complementary"
    normalized = ("point", "objective_normalized", "effective_value", "inequality_residuals_normalized",
                  "infeasibility", "complementarity", "measure", "tentative_inequality_multipliers",
                  "next_inequality_multipliers", "next_penalty", "stationarity")
    assert len(base.iterations) == len(scaled.iterations) == 3
    for one, other in zip(base.iterations, scaled.iterations):
        assert (one.penalty, one.inequality_multipliers) == (other.penalty, other.inequality_multipliers)
        assert [getattr(one.evaluation, n) for n in normalized] \
            == [getattr(other.evaluation, n) for n in normalized]
        if scaled_term == "constraint":
            assert other.evaluation.inequality_residuals == (one.evaluation.inequality_residuals[0] / 4,)
        elif scaled_term == "objective":
            assert other.evaluation.objective == 4 * one.evaluation.objective
        else:
            assert other.evaluation.objective == 2.0**600 * one.evaluation.objective
            assert other.evaluation.inequality_residuals == (one.evaluation.inequality_residuals[0] * 2.0**-600,)
    for tentative in (True, False):
        (mu,) = base.multipliers(tentative=tentative)[1]
        if factor is None:
            with pytest.raises(ValueError, match="normal binary64 range"):
                scaled.multipliers(tentative=tentative)
            assert scaled.report()["multipliers"] is None
            assert "unavailable" in next(line for line in str(scaled).splitlines()
                                         if line.startswith("Last round"))
        else:
            (scaled_mu,) = scaled.multipliers(tentative=tentative)[1]
            assert mu > 0 and scaled_mu == factor * mu


def test_active_disk_constraint_reaches_the_feasible_grid_minimum():
    """Minimize (x-1)**2 + (y-1)**2 on the unit disk: x* = (1, 1)/sqrt 2 with multiplier sqrt 2 - 1.

    For any y, f(y) - f* = (1 - sqrt 2) g(y) + sqrt 2 |y - x*|**2 with g = |y|**2 - 1, because both
    sides are quadratics with the same value, gradient and Hessian at x*. The finite-grid multiplier
    is a dual iterate of the grid problem and is not compared with sqrt 2 - 1.
    """
    disk = x**2 + y**2 - 1
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=((0.0, 1.0), (0.0, 1.0)),
                                      inequalities=(disk,))
    result = _solve(problem, qhd=_qhd(num_grid_points=4))
    _assert_every_grid_point_observed(result)
    assert result.termination == "feasible_complementary"
    reference = constrained_grid_minimum(result)
    axis = [0.0 + (i + 1) * (1.0 / 5) for i in range(4)]
    value, point = _feasible_grid_minimum(sp.lambdify((x, y), F), [("ineq", sp.lambdify((x, y), disk))],
                                          (axis, axis), 1e-9)
    assert result.candidate == reference.point == point and result.objective == reference.objective == value
    assert reference.gap == 0.0
    corner = 1 / sqrt(2)
    distance2 = (point[0] - corner) ** 2 + (point[1] - corner) ** 2
    residual = result.best.evaluation.inequality_residuals[0]
    gap = value - (3 - 2 * sqrt(2))
    # Machine-precision identity: its terms are below 1 in magnitude and each takes at most eight
    # roundings, so the two sides differ by at most 32 u.
    assert gap == pytest.approx((1 - sqrt(2)) * residual + sqrt(2) * distance2, rel=0, abs=32 * UNIT_ROUNDOFF)
    # A feasible point cannot beat the constrained optimum of this convex problem.
    assert gap > 0


def test_default_options_are_the_documented_search_defaults_and_a_refined_run_records_them():
    """The defaults of docs/ENGINEERING_CONSTANTS.md, through the public constructors and one run.

    The kinetic initial state, the search model with gain 8, the most probable point and the
    normalized feasibility and complementarity stop. A refined run on the unit-disk example of the
    QHD notebook with every option at its default records the values it used: each level solved the
    search model with gain 8, each level Plan started from the kinetic ground state, and every
    round read its most probable point.
    """
    options, refinement = AugmentedLagrangian(), BoxRefinement()
    assert QHD().initial_state == KineticGroundState()
    assert (refinement.scaling, refinement.potential_gain, refinement.gain) == ("search_model", None, 8.0)
    assert refinement.point_rule == options.inner_point == "most_probable"
    assert options.termination == "feasibility_and_complementarity"
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=((0.0, 1.0), (0.0, 1.0)),
                                      inequalities=(x**2 + y**2 - 1,))
    qhd = QHD(num_grid_points=4, num_steps=80, total_time=10.0)
    result = solve_augmented_lagrangian(problem, qhd=qhd, refinement=BoxRefinement(), execution="classical",
                                        seed=7, progress=False)
    record = result.record
    assert record.qhd.initial_state == KineticGroundState()
    assert (record.refinement.scaling, record.refinement.point_rule) == ("search_model", "most_probable")
    assert record.options.inner_point == "most_probable" and result.termination == "feasible_complementary"
    for item, levels in zip(result.iterations, result.results, strict=True):
        assert item.refinement.levels and item.evaluation.rule == "most_probable"
        assert all(level.potential_gain == 8.0 for level in item.refinement.levels)
        assert all(inner.plan.method.initial_state == KineticGroundState() for inner in levels)


def test_bound_absorption_moves_a_one_variable_constraint_into_the_box():
    problem = ConstrainedOptimization(objective=(z - 1) ** 2, variables=(z,), bounds=((-1.0, 1.0),),
                                      inequalities=(z - sp.Rational(1, 2),))
    # The endpoint grid of the tightened box [-1, 1/2] contains the constrained optimum 1/2.
    boundary = _solve(problem, qhd=_qhd(num_grid_points=4, include_boundary_points=True), absorb_bounds=True)
    preprocessing = boundary.record.preprocessing
    assert preprocessing.bounds == ((-1.0, 0.5),) and preprocessing.inequalities == ()
    assert preprocessing.absorbed == (AbsorbedBound(inequality=0, variable=0, side="upper", value=0.5),)
    assert boundary.termination == "feasible_complementary" and boundary.candidate == (0.5,)
    # The Dirichlet interior grid keeps the absorbed boundary out of the register.
    interior = _solve(problem, qhd=_qhd(num_grid_points=4), absorb_bounds=True)
    assert interior.candidate == (-1.0 + 4 * (1.5 / 5),)
    # The result states the model change, lists the kept multiplier by its problem position and names the
    # initial multiplier that the absorbed constraint did not use (PLAN 3.3, item 2).
    both = ConstrainedOptimization(objective=(x - 1) ** 2 + (y - 1) ** 2, variables=(x, y), bounds=BOX,
                                   inequalities=(x - sp.Rational(1, 2), x**2 + y**2 - sp.Rational(1, 2)))
    report = _solve(both, qhd=_qhd(num_grid_points=4), absorb_bounds=True, inequality_multipliers=(0.5, 0.0)).report()
    assert report["multipliers"]["inequalities"].keys() == {"1"} and report["multipliers"]["equalities"] == {}
    statements = " ".join(report["statements"])
    assert "inequalities[0] by x <= 0.5" in statements and "((-1.0, 0.5), (-1.0, 1.0))" in statements
    assert "were not used: inequalities[0] 0.5" in statements
    for constraint, message in ((sp.Ge(z, 2), "infeasible in its box"), (sp.Le(z, -1), "fixes z = -1.0")):
        empty = ConstrainedOptimization(objective=z, variables=(z,), bounds=((-1.0, 1.0),),
                                        inequalities=(constraint,))
        with pytest.raises(ValueError, match=message):
            _solve(empty, absorb_bounds=True)


def test_bound_absorption_classifies_the_exact_cut_before_rounding():
    # On the box [1/2, 1] the upper cuts 1/2 - 2**-56, 1/2 and 1/2 + 2**-56 all round to 0.5, while
    # their exact intersections with the box are empty, the point 1/2 and [1/2, 1/2 + 2**-56]
    # (constrained._absorb).
    tiny = sp.Rational(1, 2**56)

    def problem(cut):
        return ConstrainedOptimization(objective=(z - 1) ** 2, variables=(z,), bounds=((0.5, 1.0),),
                                       inequalities=(z - cut,))

    with pytest.raises(ValueError, match="infeasible in its box"):
        _solve(problem(sp.Rational(1, 2) - tiny), absorb_bounds=True)
    with pytest.raises(ValueError, match="fixes z = 0.5"):
        _solve(problem(sp.Rational(1, 2)), absorb_bounds=True)
    # The cut with positive width has no binary64 value, so it stays a constraint of the rounds and
    # the box keeps its bound.
    record = _solve(problem(sp.Rational(1, 2) + tiny), absorb_bounds=True, max_iterations=1).record
    assert record.preprocessing.bounds == ((0.5, 1.0),) and record.preprocessing.absorbed == ()
    assert record.preprocessing.inequalities == (0,)
    # A binary64 coefficient enters as its exact rational, also next to a rational one. With
    # b = fl(1/6), the cut z <= 3 b of z/3 - b <= 0 lies just below 1/2 and has no binary64 value, so
    # it stays a constraint on [0, 1], and the lower cut of b - z/3 <= 0 on [0, 1/2] leaves the positive
    # width [3 b, 1/2] rather than fixing z. Reading the coefficients in binary64 would give 1/2 for both.
    for cut, box in ((z / 3 - 1 / 6, (0.0, 1.0)), (1 / 6 - z / 3, (0.0, 0.5))):
        mixed = ConstrainedOptimization(objective=(z - 1) ** 2, variables=(z,), bounds=(box,), inequalities=(cut,))
        kept = _solve(mixed, absorb_bounds=True, max_iterations=1).record.preprocessing
        assert kept.bounds == (box,) and kept.absorbed == () and kept.inequalities == (0,)
    # Covering every constraint does not show that each is kept once, so a record that keeps the
    # inequality twice is rejected.
    pre = record.preprocessing
    with pytest.raises(ValueError, match="each of the problem's constraints once"):
        record.revise(preprocessing=pre.revise(inequalities=(0, 0), inequality_lower=pre.inequality_lower * 2,
                                               inequality_upper=pre.inequality_upper * 2),
                      inequality_scales=record.inequality_scales * 2, iterations=(),
                      termination="budget_exhausted", failure="no round started", best=None, last=None)


def test_a_failed_derivative_attempt_is_charged_to_its_round():
    # f = sqrt(z) on the endpoint grid {0, 1} of [0, 1] with the inactive z - 2 <= 0. The round chooses
    # z = 0, where sqrt is finite and its derivative 1/(2 sqrt(z)) is not, so r_stat is unavailable.
    # The layer still evaluated f and g at the point and attempted that derivative, each costing its
    # tree nodes plus d = 1 units (constrained._admit_layer_work), and the round counts all three.
    problem = ConstrainedOptimization(objective=sp.sqrt(z), variables=(z,), bounds=((0.0, 1.0),),
                                      inequalities=(z - 2,))
    result = _solve(problem, qhd=_qhd(num_grid_points=2, include_boundary_points=True), stationarity=True)
    item = result.iterations[0]
    assert item.evaluation.point == (0.0,) and item.evaluation.stationarity is None
    assert "partial derivative of" in item.evaluation.stationarity_unavailable
    expressions = (sp.sqrt(z), z - 2, sp.diff(sp.sqrt(z), z))
    assert item.resources.layer_evaluations == 3
    assert item.resources.layer_work == sum(node_count(e, 10**8) + 1 for e in expressions)


@pytest.mark.parametrize("field,constraints,name", [
    ("inequalities", (x - 1, sp.nan), "inequalities\\[1\\]"),
    ("inequalities", (sp.zoo,), "inequalities\\[0\\]"),
    ("equalities", (sp.I * x,), "equalities\\[0\\]"),
])
def test_nonfinite_or_complex_constraints_reject_before_the_first_round(field, constraints, name):
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, **{field: constraints})
    with patch("nwqlib.scientist._plan_with_streams", side_effect=AssertionError("a round was planned")):
        with pytest.raises(ValueError, match=name):
            _solve(problem, feasibility_tolerance=1e-9)


def test_equality_constraints_need_an_explicit_feasibility_tolerance():
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, equalities=(x + y,))
    with pytest.raises(ValueError, match="feasibility_tolerance is required"):
        _solve(problem)
    # Omitted equality multipliers start at 0, which these bounds exclude, and Algorithm 4.1 of Birgin and
    # Martinez starts inside them, so the run is refused before it plans a round.
    bounds = MultiplierBounds(equality_lower=1.0, equality_upper=2.0, inequality_upper=1.0)
    with patch("nwqlib.scientist._plan_with_streams", side_effect=AssertionError("a round was planned")):
        with pytest.raises(ValueError, match="equality_multipliers=None starts every equality multiplier at 0"):
            _solve(problem, feasibility_tolerance=1e-9, multiplier_bounds=bounds)


def test_saved_run_reloads_without_planning_evaluating_or_acquiring(tmp_path):
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX,
                                      inequalities=(x + y - sp.Rational(3, 2),))
    result = _solve(problem, inequality_multipliers=(4.0,))
    path = result.save(tmp_path / "run")
    with patch("nwqlib.algorithms.qhd.method.QHD.plan", side_effect=AssertionError("planned")), \
            patch("sympy.lambdify", side_effect=AssertionError("evaluated")), \
            patch("nwqlib.scientist.prepare", side_effect=AssertionError("acquired")):
        loaded = load_augmented_lagrangian(path)
    assert loaded.record.content_id == result.record.content_id and loaded.problem == result.problem
    assert [inner.content_id for inner in loaded.results] == [inner.content_id for inner in result.results]
    assert loaded.report()["record"] == result.report()["record"]
    assert constrained_grid_minimum(loaded) == constrained_grid_minimum(result)


# The combination problem: f = (x - 11/16)**2 + (y - 7/16)**2 subject to h = 4 (x + y - 1/8) = 0 on
# [-1, 1]**2, with scales s_f = 2 and s_h = 4. Its KKT point and normalized multiplier are below.
KKT_POINT = (Fraction(3, 16), Fraction(-1, 16))
KKT_MULTIPLIER = Fraction(1, 2)
FIRST_AXIS = tuple(Fraction(i - 3, 4) for i in range(7))  # the 7-point interior grid of [-1, 1]


def _combination_run():
    """The augmented Lagrangian with box refinement on the combination problem, seed 11.

    The uniform start and gain 1 are named because the defaults are the kinetic ground state and
    gain 8. The observed run and the detector-strength checks of the test below were recorded with
    them.
    """
    problem = ConstrainedOptimization(
        objective=(x - sp.Rational(11, 16)) ** 2 + (y - sp.Rational(7, 16)) ** 2, variables=(x, y), bounds=BOX,
        equalities=(4 * (x + y - sp.Rational(1, 8)),))
    qhd = QHD(num_grid_points=7, num_steps=20, total_time=6.0, schedule=QuadraticSchedule(gamma=0.3),
              theory_flavor="schrodinger", keep_state=True, initial_state=UniformState())
    options = AugmentedLagrangian(
        initial_penalty=1.0, penalty_growth=2.0, reduction_ratio=0.25, max_penalty=4.0, max_iterations=4,
        objective_scale=2.0, equality_scales=(4.0,), equality_multipliers=(0.0,),
        multiplier_bounds=MultiplierBounds(equality_lower=-1.0, equality_upper=1.0, inequality_upper=1.0),
        feasibility_tolerance=5 / 32, complementarity_tolerance=5 / 32, inner_point="best_observed")
    refinement = BoxRefinement(scaling="search_model", potential_gain=1.0, max_levels=3, mass_threshold=0.5,
                               max_no_improve=2, point_rule="best_observed")
    return solve_augmented_lagrangian(problem, qhd=qhd, options=options, refinement=refinement,
                                      execution="classical", seed=11, progress=False)


def _effective(point, lam, rho):
    """Exact normalized L = f/2 + lam r + rho r**2/2 with r = h/4 = x + y - 1/8."""
    r = point[0] + point[1] - Fraction(1, 8)
    return ((point[0] - Fraction(11, 16)) ** 2 + (point[1] - Fraction(7, 16)) ** 2) / 2 + lam * r + rho * r * r / 2


def _hessian_form(e, rho):
    """e^T H e with H = I + rho v v^T and v = (1, 1)."""
    return e[0] * e[0] + e[1] * e[1] + rho * (e[0] + e[1]) ** 2


def _within_bound(point, lam, rho):
    """Whether ||point - x*|| <= B = sqrt(2) |1/2 - lam|/M + sqrt(2 Delta) holds, decided exactly.

    With A = 2 d**2/M**2, C = 2 Delta and R = ||point - x*||**2 as fractions, sqrt(R) <= sqrt(A) + sqrt(C)
    is R <= A + C or, when R > A + C, (R - A - C)**2 <= 4 A C, since both sides are then nonnegative.
    Returns the decision, the minimizer m of L and Delta, as the combination test derives them.
    """
    d, big_m = KKT_MULTIPLIER - lam, 1 + 2 * rho
    m = (KKT_POINT[0] + d / big_m, KKT_POINT[1] + d / big_m)
    delta = min(_hessian_form((q[0] - m[0], q[1] - m[1]), rho) for q in itertools.product(FIRST_AXIS, repeat=2)) / 2
    a, c = 2 * d * d / big_m**2, 2 * delta
    r2 = (point[0] - KKT_POINT[0]) ** 2 + (point[1] - KKT_POINT[1]) ** 2
    return r2 <= a + c or (r2 - a - c) ** 2 <= 4 * a * c, m, delta


def test_refined_rounds_stay_within_the_distance_bound_to_the_kkt_point():
    """Each round refines the box of L_k and returns a point within a derived distance of the KKT point.

    Problem. f = (x - 11/16)**2 + (y - 7/16)**2 and h = 4 (x + y - 1/8) = 0 on [-1, 1]**2, with scales
    s_f = 2 and s_h = 4, so the normalized terms are F = f/2 and r = h/4 = x + y - 1/8. At
    x* = (3/16, -1/16), h = 0, grad f = (-1, -1) and grad h = (4, 4), so grad f + (1/4) grad h = 0. x* is
    interior, so the box multipliers vanish, and f is strictly convex, so x* is the unique KKT point.
    Its normalized multiplier is 1/2 (1/4 in original units).

    Round k. L = F + lambda r + (rho/2) r**2 with the multiplier lambda and penalty rho entering the
    round. With v = (1, 1), its Hessian is H = I + rho v v^T, with eigenvalues 1 and M = 1 + 2 rho, and
    its exact minimizer is m = x* + d v/M with d = 1/2 - lambda. For a quadratic,
    (1) L(z) - L(m) = (z - m)^T H (z - m)/2.
    The safeguard lambda in [-1, 1] and rho in [1, 4] (max_penalty) put m in
    [1/48, 11/16] x [-11/48, 7/16], inside [-7/8, 7/8]**2. The first grid G0 = {-3/4, ..., 3/4}**2 has
    spacing h0 = 1/4, and every coordinate in [-7/8, 7/8] lies within h0/2 of a grid coordinate, so the
    grid point nearest m is within h0/2 per coordinate, and with the largest eigenvalue M,
    (2) Delta = min_{q in G0} (q - m)^T H (q - m)/2 <= M h0**2/4 = M/64.
    Premise: every completed level reports a minimizer of L over its own grid. best_observed under exact
    readout returns the least tabulated value among observed points, so every grid point must have
    positive probability, checked on the kept states. The search model tabulates a rescaled potential
    with rounded divisions, so each reported point is also compared, in Fraction arithmetic, with the
    minimum of the analytic L over the 49 points of its level grid. The round keeps the best level by
    its recorded relative objective, the earlier on ties (constrained._refined_choice). This test also
    checks that the selected level minimizes the analytic L over the reported level points. With
    these independently checked premises and the first level solving G0, its point
    p satisfies
    (3) L(p) <= min_{q in G0} L(q),
    whatever boxes the later levels chose. By (1), (2) and (3), e = p - m has
    (4) e^T H e <= 2 Delta, so ||e|| <= sqrt(2 Delta) because the smallest eigenvalue is 1, and
    (5) ||p - x*|| <= ||m - x*|| + ||e|| <= sqrt(2) |d|/M + sqrt(2 Delta) = B.
    No later box is assumed to contain m. Since r(m) = 2 d/M, the update lambda+ = lambda + rho r(p)
    satisfies
    (6) lambda+ - 1/2 = (lambda - 1/2)/M + rho v^T e,
    and the next multiplier clips lambda+ to [-1, 1]. With one equality and no inequality, the normalized
    infeasibility, complementarity and penalty measure all equal a_k = |r(p_k)|, and
    (7) rho_{k+1} = rho_k when k = 0 or a_k <= a_{k-1}/4, and min(2 rho_k, 4) otherwise.
    The run stops when a_k <= 5/32. The best round b is the one with the least original f among
    rounds with a_k <= 5/32, the earliest on ties (the rounds' own rule), so by (5)
    (8) ||x_best - x*|| <= B_b with the multiplier and penalty entering round b.
    With A = 2 d**2/M**2, C = 2 Delta and R = ||p - x*||**2, (5) is R <= A + C or
    (R - A - C)**2 <= 4 A C when R > A + C, decided exactly with fractions (_within_bound). The
    coordinates, scales, grids and updates are dyadic here, so the recorded binary64 values convert to
    fractions exactly and the relations above are compared exactly.

    Observed run (seed 11, classical, 20 steps of the quadratic schedule with gamma = 0.3 over T = 6):
    two rounds of three levels. Round 1 enters with lambda = 87/256 and rho = 1, so Delta = 49/196608
    and B = (41 sqrt(2) + 7 sqrt(6))/768 = 0.0978244587, and returns x_best = (1/4, 0) at distance
    sqrt(2)/16 = 0.0883883476. In that round the last level and the level with the least f both report
    (29/128, 5/128), with a = 9/64 <= 5/32, which would stop the run there at distance
    sqrt(194)/128 = 0.1088155334 > B, so a run that took either of them would fail (8).
    """
    result = _combination_run()
    record = result.record
    assert record.preprocessing.bounds == BOX and record.equality_scales == (4.0,)
    assert (record.execution, record.shots, record.options.objective_scale) == ("classical", None, 2.0)
    assert record.refinement.point_rule == record.options.inner_point == "best_observed"
    rounds, previous = [], None
    for item, levels in zip(result.iterations, result.results, strict=True):
        refined = item.refinement
        lam, rho = Fraction(item.equality_multipliers[0]), Fraction(item.penalty)
        assert -1 <= lam <= 1 and 1 <= rho <= 4
        values, points = [], []
        for z, (level, inner) in enumerate(zip(refined.levels, levels, strict=True)):
            probabilities = np.abs(inner.data.artifact(inner.artifact).array) ** 2
            assert probabilities.size == 49 and np.all(np.isfinite(probabilities)) and np.all(probabilities > 0)
            assert 0 < level.energy_scale < np.inf and level.potential_gain == 1.0
            # Every round restarts from the preprocessed box, and each later box lies inside the one before.
            if z == 0:
                assert level.box == BOX
            else:
                outer = refined.levels[z - 1].box
                assert level.box == refined.levels[z - 1].next_box
                assert all(a <= c < e <= b for (a, b), (c, e) in zip(outer, level.box))
            axes = [[Fraction(a) + (Fraction(b) - Fraction(a)) * Fraction(i + 1, 8) for i in range(7)]
                    for a, b in level.box]
            point = tuple(Fraction(value) for value in level.point)
            assert point == tuple(axis[i] for axis, i in zip(axes, level.point_indices))
            assert _effective(point, lam, rho) == min(_effective(q, lam, rho) for q in itertools.product(*axes))
            values.append(_effective(point, lam, rho))
            points.append(point)
        # Check that the recorded choice minimizes analytic L here; rounded relative-objective
        # comparisons need not have this property for a general objective.
        best = min(range(len(values)), key=lambda z: (values[z], z))
        assert refined.best_level == best + 1 and item.evaluation.point == refined.levels[best].point
        e = item.evaluation
        p = points[best]
        r = p[0] + p[1] - Fraction(1, 8)
        assert Fraction(e.equality_residuals_normalized[0]) == r
        assert Fraction(e.infeasibility) == Fraction(e.complementarity) == Fraction(e.measure) == abs(r)
        lam_plus = lam + rho * r
        big_m = 1 + 2 * rho
        within, m, delta = _within_bound(p, lam, rho)
        v_e = (p[0] - m[0]) + (p[1] - m[1])
        assert lam_plus - KKT_MULTIPLIER == (lam - KKT_MULTIPLIER) / big_m + rho * v_e  # (6)
        assert Fraction(e.tentative_equality_multipliers[0]) == lam_plus
        assert Fraction(e.next_equality_multipliers[0]) == min(max(lam_plus, -1), 1)
        grown = rho if previous is None or abs(r) <= previous / 4 else min(2 * rho, 4)
        assert Fraction(e.next_penalty) == grown  # (7)
        assert all(-Fraction(7, 8) <= c <= Fraction(7, 8) for c in m) and delta <= big_m / 64  # (2)
        assert _hessian_form((p[0] - m[0], p[1] - m[1]), rho) <= 2 * delta  # (4)
        assert within  # (5)
        f = (p[0] - Fraction(11, 16)) ** 2 + (p[1] - Fraction(7, 16)) ** 2
        assert Fraction(e.objective) == f
        last = points[-1]
        by_f = min(range(len(points)), key=lambda z: ((points[z][0] - Fraction(11, 16)) ** 2
                                                      + (points[z][1] - Fraction(7, 16)) ** 2, z))
        rounds.append(dict(a=abs(r), f=f, lam=lam, rho=rho, point=p, alternatives=(last, points[by_f])))
        previous = abs(r)
    stops = [entry["a"] <= Fraction(5, 32) for entry in rounds]
    assert result.termination == "feasible_complementary" and stops[-1] and not any(stops[:-1])
    # The best round by the rounds' own rule, and (8) with the multiplier and penalty entering it.
    feasible = [k for k, entry in enumerate(rounds) if stops[k]]
    b = min(feasible, key=lambda k: (rounds[k]["f"], k))
    assert result.best.iteration == b and tuple(map(Fraction, result.candidate)) == rounds[b]["point"]
    assert _within_bound(tuple(map(Fraction, result.candidate)), rounds[b]["lam"], rounds[b]["rho"])[0]
    # Detector strength. The run has more than one round and level. Before b the last level and the
    # least-f level coincide with the chosen point, so a run that took either would reach round b with
    # the same lambda and rho. In round b both differ from it, pass the stopping test, and lie outside B_b,
    # so either choice would end the run there and fail (8).
    assert len(rounds) > 1 and all(len(item.refinement.levels) > 1 for item in result.iterations)
    for k, entry in enumerate(rounds):
        if k < b:
            assert all(alternative == entry["point"] for alternative in entry["alternatives"])
    for alternative in rounds[b]["alternatives"]:
        assert alternative != rounds[b]["point"]
        assert abs(alternative[0] + alternative[1] - Fraction(1, 8)) <= Fraction(5, 32)
        assert not _within_bound(alternative, rounds[b]["lam"], rounds[b]["rho"])[0]


def test_a_refinement_that_stops_early_ends_the_run_by_the_rule_of_both_layers():
    """_outer.refinement_stop: how a round's refinement ends the run, and the point-rule contract.

    Two circuits fund two native levels of round 0 and refuse the third. On x + y = 0 the round's point
    misses the stopping test, so the run records the round with its update and stops with the
    refinement's budget_exhausted. On an inactive inequality the round's point passes the test, so the
    run ends feasible_complementary and the nested record keeps budget_exhausted. Three circuits fund
    two levels of round 0 and one of round 1, so the limits hold across rounds and levels. A failed
    second level keeps the round's first-level point and ends the run with inner_failed. The physical
    model's classical evolution work grows with rho, so a max_work between the largest work of rounds
    0..k-1 and that of round k's first level refuses that level: a later round's first-level failure
    ends the run with inner_failed and keeps the earlier rounds, while in round 0 QHD's error
    propagates. A constant objective gives L_0 no variation on the first grid, so round 0 has no point
    and the run ends with the refinement's flat_objective.
    """
    search = BoxRefinement(scaling="search_model", point_rule="best_observed", max_levels=3, mass_threshold=0.6)
    qhd = QHD(num_grid_points=3, num_steps=4, total_time=3.0)
    two = ExecutionLimits(max_total_circuits=2)
    equality = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, equalities=(x + y,))
    cut = solve_augmented_lagrangian(equality, qhd=qhd, refinement=search, execution="quantum", limits=two, seed=5,
                                     options=AugmentedLagrangian(inner_point="best_observed",
                                                                 feasibility_tolerance=1e-9), progress=False)
    (item,) = cut.iterations
    assert cut.termination == "budget_exhausted" and item.refinement.termination == "budget_exhausted"
    assert len(item.refinement.levels) == 2 and len(cut.results[0]) == 2
    assert item.evaluation.infeasibility > 1e-9 and item.evaluation.next_penalty == 1.0
    assert cut.record.failure == item.refinement.failure
    assert "max_total_circuits has 0 left of 2" in cut.record.failure
    assert cut.resources.circuit_preparations == cut.resources.circuit_attempts == 2
    inactive = ConstrainedOptimization(objective=(x - sp.Rational(1, 4)) ** 2 + (y - sp.Rational(1, 4)) ** 2,
                                       variables=(x, y), bounds=BOX, inequalities=(x + y - sp.Rational(3, 2),))
    met = solve_augmented_lagrangian(inactive, qhd=qhd, refinement=search, execution="quantum", limits=two,
                                     seed=5, options=AugmentedLagrangian(inner_point="best_observed"),
                                     progress=False)
    (item,) = met.iterations
    assert met.termination == "feasible_complementary" and item.refinement.termination == "budget_exhausted"
    assert item.evaluation.infeasibility == item.evaluation.complementarity == 0 and met.record.failure is None
    # The limits stay cumulative over rounds and levels: three circuits fund both levels of round 0 and
    # the first level of round 1, and refuse its second.
    two_levels = BoxRefinement(scaling="search_model", point_rule="best_observed", max_levels=2, mass_threshold=0.6)
    spread = solve_augmented_lagrangian(equality, qhd=qhd, refinement=two_levels, execution="quantum", seed=5,
                                        limits=ExecutionLimits(max_total_circuits=3), progress=False,
                                        options=AugmentedLagrangian(inner_point="best_observed",
                                                                    feasibility_tolerance=1e-9))
    assert [(len(item.refinement.levels), item.refinement.termination) for item in spread.iterations] == [
        (2, "level_limit"), (1, "budget_exhausted")]
    assert spread.termination == "budget_exhausted" and spread.iterations[1].evaluation.infeasibility > 1e-9
    assert spread.resources.circuit_attempts == 3
    # A failure after a completed level: the physical model's second level of round 0 needs more kernel work
    # than its first, so a max_work between the two refuses it. The round keeps the first level's point and
    # update, which misses the stopping test, and the run ends with the refinement's inner_failed.
    three = BoxRefinement(scaling="physical", point_rule="best_observed", max_levels=3, mass_threshold=0.6)

    def one_round(max_work):
        return solve_augmented_lagrangian(
            ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX,
                                    equalities=(x + y - sp.Rational(3, 10),)),
            qhd=QHD(num_grid_points=3, num_steps=8, total_time=3.0, max_work=max_work), refinement=three,
            options=AugmentedLagrangian(inner_point="best_observed", feasibility_tolerance=1e-3, max_iterations=1),
            execution="classical", seed=5, progress=False)

    level_work = [inner.plan.construction.kernels[0].invocation_work for inner in one_round(10**8).results[0]]
    assert level_work[1] > level_work[0]
    stopped = one_round((level_work[0] + level_work[1]) // 2)
    (item,) = stopped.iterations
    assert stopped.termination == item.refinement.termination == "inner_failed" and len(item.refinement.levels) == 1
    assert item.evaluation.point == item.refinement.levels[0].point and item.evaluation.infeasibility > 1e-3
    assert "set QHD(max_work=" in stopped.record.failure
    # A later round's first-level failure, with the classical kernel work of every level of a reference run.
    offgrid = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX,
                                      equalities=(x + y - sp.Rational(3, 10),))
    physical = BoxRefinement(scaling="physical", point_rule="best_observed", max_levels=2, mass_threshold=0.6)
    options = AugmentedLagrangian(inner_point="best_observed", feasibility_tolerance=1e-3, max_iterations=9)

    def physical_run(max_work=100_000_000):
        return solve_augmented_lagrangian(offgrid, qhd=QHD(num_grid_points=3, total_time=0.1, max_work=max_work),
                                          options=options, refinement=physical, execution="classical", seed=5,
                                          progress=False)

    reference = physical_run()
    work = [[inner.plan.construction.kernels[0].invocation_work for inner in levels] for levels in reference.results]
    k = next(k for k in range(1, len(work)) if work[k][0] > max(max(levels) for levels in work[:k]))
    failing = physical_run((max(max(levels) for levels in work[:k]) + work[k][0]) // 2)
    assert failing.termination == "inner_failed" and len(failing.iterations) == k + 1
    stopped = failing.iterations[k]
    assert stopped.refinement.levels == () and stopped.refinement.termination == "inner_failed"
    assert stopped.evaluation is None and failing.results[k] == ()
    assert failing.record.failure.startswith("ValueError") and "set QHD(max_work=" in failing.record.failure

    def trajectory(run):
        return [(item.evaluation.point, item.penalty, item.equality_multipliers) for item in run.iterations[:k]]

    assert trajectory(failing) == trajectory(reference) and failing.last.iteration == k - 1
    with pytest.raises(ValueError, match=r"set QHD\(max_work="):
        physical_run(work[0][0] - 1)
    flat = ConstrainedOptimization(objective=3 + 0 * z, variables=(z,), bounds=((-1.0, 1.0),), inequalities=(z - 2,))
    empty = solve_augmented_lagrangian(flat, qhd=QHD(num_grid_points=3), refinement=BoxRefinement(
        scaling="search_model"), execution="classical", seed=5, progress=False)
    assert empty.termination == "flat_objective" and empty.iterations[0].evaluation is None
    assert empty.best is None and empty.candidate is None
    # A physical level plans before its resolution check, so on quantum execution the flat level has a Plan with
    # a CX and a rotation law but prepares no circuit. It contributes neither (_outer.law_count), and the
    # record's rotation total equals the body total of run_resources, which weights each round's law by the
    # circuits it prepared.
    quantum = solve_augmented_lagrangian(flat, qhd=QHD(num_grid_points=3), refinement=BoxRefinement(
        scaling="physical"), execution="quantum", seed=5, progress=False)
    assert quantum.termination == "flat_objective"
    counts = quantum.resources
    assert (counts.circuit_preparations, counts.cx, counts.arbitrary_rotations) == (0, 0, 0)
    assert run_resources(quantum).arbitrary_rotations == 0
    # A stall with no split budget left stops a refinement with split_limit after its level, and the
    # round goes on under the layer's own tests. With the whole axis kept (threshold 1) every round stalls
    # at its first level, the search-model grid minimizer of L_k, so the run follows the grid oracle.
    stalled = BoxRefinement(scaling="search_model", point_rule="best_observed", max_levels=2, mass_threshold=1.0,
                            stall_split="best_region", max_splits=0)
    oracle = ORACLE["equality"]
    split = solve_augmented_lagrangian(equality, qhd=_qhd(), refinement=stalled, execution="classical", seed=11,
                                       options=AugmentedLagrangian(inner_point="best_observed", **oracle["options"]),
                                       progress=False)
    assert [(len(item.refinement.levels), item.refinement.termination) for item in split.iterations] == [
        (1, "split_limit")] * len(oracle["points"])
    assert [item.evaluation.point for item in split.iterations] == oracle["points"]
    assert split.termination == "feasible_complementary"
    # One rule reads every level, so two different rules are refused before any round is planned, as is a
    # stall split that a classical level without its kept state cannot score.
    with patch("nwqlib.scientist._plan_with_streams", side_effect=AssertionError("a round was planned")):
        with pytest.raises(ValueError, match="differs from options.inner_point"):
            solve_augmented_lagrangian(equality, qhd=qhd, refinement=BoxRefinement(scaling="search_model"),
                                       options=AugmentedLagrangian(inner_point="best_observed",
                                                                   feasibility_tolerance=1e-9),
                                       execution="classical", seed=5, progress=False)
        with pytest.raises(ValueError, match="keep_state=True"):
            solve_augmented_lagrangian(equality, qhd=qhd, refinement=BoxRefinement(
                scaling="search_model", stall_split="best_region"), options=AugmentedLagrangian(
                    feasibility_tolerance=1e-9), execution="classical", seed=5, progress=False)


def test_saved_refined_run_reloads_its_levels_and_a_plain_run_keeps_its_layout(tmp_path):
    result = _combination_run()
    path = result.save(tmp_path / "refined")
    with patch("nwqlib.algorithms.qhd.method.QHD.plan", side_effect=AssertionError("planned")), \
            patch("sympy.lambdify", side_effect=AssertionError("evaluated")), \
            patch("nwqlib.scientist.prepare", side_effect=AssertionError("acquired")):
        loaded = load_augmented_lagrangian(path)
    assert loaded.record.content_id == result.record.content_id and loaded.problem == result.problem
    assert [[inner.content_id for inner in levels] for levels in loaded.results] \
        == [[inner.content_id for inner in levels] for levels in result.results]
    assert loaded.report() == result.report()
    assert sorted(p.relative_to(path).as_posix() for p in (path / "iterations").glob("*/*/*")) == [
        f"iterations/{item.iteration}/levels/{level.level}" for item in result.iterations
        for level in item.refinement.levels]
    with pytest.raises(ValueError, match="nested grids"):
        constrained_grid_minimum(loaded)
    # Without refinement the archive keeps one result folder per round and no level folders.
    plain = _solve(ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, equalities=(x + y,)),
                   feasibility_tolerance=1e-9)
    path = plain.save(tmp_path / "plain")
    assert sorted(p.relative_to(path).as_posix() for p in path.glob("iterations/*/*")) == [
        f"iterations/{item.iteration}/result" for item in plain.iterations]
    assert load_augmented_lagrangian(path).record.content_id == plain.record.content_id
    # Neither exact refinement nor a plain AL run has a sampled selected-box
    # event. Neither report may call the beta inverse for that absent event.
    with patch("scipy.special.betaincinv", side_effect=AssertionError("exact/plain report evaluated CP")):
        assert result.report()["confidence"] is None
        assert plain.report()["confidence"] is None
        with pytest.raises(ValueError, match="failure_probability"):
            plain.report(failure_probability=0)


def test_a_nonfinite_objective_at_a_later_refined_point_ends_the_run_and_keeps_the_earlier_rounds():
    """_outer.inner_failure also covers f, h or g that is not finite at a refinement level's reported point.

    On [0.7, 1.9] with K = 2 the layer checks f on the interior grid {1.1, 1.5} before any round
    (constrained._setup). A search-model level solves at the unit points fl(1/3) and 2 fl(1/3) and
    reports the image 0.7 + 1.2 u rounded once, which for the second point is c = 1.5 - 2**-52 and not
    the checked grid value 1.5. f = 1/(x - c) is finite on the checked grid (2**52 at 1.5) and in the
    level tables, which evaluate the substituted objective at the unit points, and divides by zero at
    the reported point c. Each level checks f and h at its reported point before the point takes part
    in the selection (constrained._level_check). The injected evolution puts all mass on the first point
    in round 0, whose point misses x = 1, and on the second point afterwards, so the level of round 1 fails
    as a level whose Run raised does, the round has no point, and the run ends with inner_failed and keeps
    round 0. With all mass on the second point in round 0 the error propagates, as a first failure does.
    """
    c = 1.5 - 2.0**-52
    problem = ConstrainedOptimization(objective=1 / (x - sp.Rational(c)), variables=(x,), bounds=((0.7, 1.9),),
                                      equalities=(x - 1,))

    def run(states):
        with patch("nwqlib.algorithms.qhd.method._evolve_restricted", side_effect=lambda *args: next(states)):
            return solve_augmented_lagrangian(problem, qhd=QHD(num_grid_points=2),
                                              options=AugmentedLagrangian(feasibility_tolerance=1e-9),
                                              refinement=BoxRefinement(scaling="search_model", max_levels=1),
                                              execution="classical", seed=3, progress=False)

    first, second = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
    result = run(itertools.chain([first], itertools.repeat(second)))
    assert result.termination == "inner_failed" and len(result.iterations) == 2
    kept, failed = result.iterations
    assert kept.evaluation.point == kept.refinement.levels[0].point != (c,)
    assert failed.evaluation is None and failed.refinement.levels == ()
    assert failed.refinement.termination == "inner_failed" and "at (1.4999999999999998,)" in failed.refinement.failure
    assert "is not a finite real value" in result.record.failure
    assert result.best is result.last is kept and result.results[1] == ()
    # The attempted check of f and h at the level's point is charged to the failed round.
    assert failed.resources.layer_evaluations == 2
    with pytest.raises(ValueError, match="is not a finite real value"):
        run(itertools.repeat(second))


# Slack variables for inequality constraints, Proposition 54 of docs/mathematics.md. Every exact check
# evaluates the inner objective that a round's Plan solves with its binary64 coefficients as exact
# rationals, at the Plan's binary64 grid coordinates as exact rationals, against an independently
# written PHR term.

def _exact(expression, values):
    """Evaluate a SymPy expression exactly at rational values, every binary64 Float as its exact rational."""
    exact = expression.xreplace({f: sp.Rational(f) for f in expression.atoms(sp.Float)})
    return sp.Rational(exact.xreplace({symbol: sp.Rational(value) for symbol, value in values.items()}))


def _phr(g, mu, rho):
    """The PHR term ([mu + rho g]_+**2 - mu**2)/(2 rho) of Result 46, in exact arithmetic."""
    mu, rho = sp.Rational(mu), sp.Rational(rho)
    return (max(mu + rho * g, sp.Integer(0)) ** 2 - mu**2) / (2 * rho)


def _grid_axes(plan):
    """The exact rational coordinates of every axis of a Plan's grid."""
    from nwqlib.algorithms.qhd.grid import OneHotGrid

    qhd = plan.method
    grid = OneHotGrid(plan.problem.variable_names, plan.problem.bounds, qhd.num_grid_points,
                      qhd.include_boundary_points, qhd.boundary)
    return [[sp.Rational(grid.grid_value(j, i)) for i in range(qhd.num_grid_points)]
            for j in range(grid.num_variables)]


@pytest.mark.parametrize("grid", [dict(), dict(include_boundary_points=True), dict(boundary="periodic")])
def test_minimizing_the_slack_grid_reproduces_the_phr_term_within_the_recorded_error_bound(grid):
    # f = (x - 1/3)**2 and g = x - 1/4 <= 0 on [0, 1] with mu = 1/8 and rho = 2, so mu/rho = 1/16 and the
    # PHR term is active on part of the grid and zero on the rest, with r = [mu + rho g]_+ > 0 there. With
    # t = [-g - mu/rho]_+ the slack term exceeds the PHR term by rho (s - t)**2/2 + r s exactly, and the
    # least excess over the slack grid is at most the recorded bound of its grid convention.
    problem = ConstrainedOptimization(objective=(z - sp.Rational(1, 3)) ** 2, variables=(z,), bounds=((0.0, 1.0),),
                                      inequalities=(z - sp.Rational(1, 4),))
    options = AugmentedLagrangian(inequality_form="slack", inequality_multipliers=(0.125,), initial_penalty=2.0)
    qhd = QHD(num_grid_points=16, theory_flavor="split_step", **grid)
    pre, form, plan = plan_augmented_lagrangian(problem, qhd=qhd, options=options,
                                                execution="classical", seed=1)
    assert form.forms == ("slack",) and len(plan.problem.variables) == 2
    axis = form.slacks[0]
    assert axis.cap == -pre.inequality_lower[0] and form.error_bound == axis.error_bound
    xs, ss = _grid_axes(plan)
    assert ss[0] == 0 or grid == {}
    assert (ss[-1] == sp.Rational(axis.upper)) == bool(grid.get("include_boundary_points"))
    mu, rho = sp.Rational(1, 8), sp.Integer(2)
    excesses = []
    for value in xs:
        g = value - sp.Rational(1, 4)
        phr = (value - sp.Rational(1, 3)) ** 2 + _phr(g, mu, rho)
        inner = min(_exact(plan.problem.objective, dict(zip(plan.problem.variables, (value, s)))) for s in ss)
        t, r = max(-g - mu / rho, 0), max(mu + rho * g, 0)
        assert t <= sp.Rational(axis.cap)
        assert inner - phr == min(rho * (s - t) ** 2 / 2 + r * s for s in ss)
        excesses.append(inner - phr)
    assert min(excesses) >= 0 and max(excesses) <= sp.Rational(axis.error_bound)
    # The bound is that of the grid convention for this round's multiplier and penalty.
    h = sp.Rational(axis.spacing)
    expected = ((mu + rho * sp.Rational(pre.inequality_upper[0])) * h + rho * h**2 / 2 if grid == {}
                else rho * h**2 / 8)
    assert sp.Rational(axis.error_bound) >= expected > sp.Rational(axis.error_bound) * (1 - 2 * sp.Rational(2) ** -52)


def test_support_table_caps_are_exact_for_separable_constraints_and_bound_overlapping_supports():
    # A separable nonlinear g takes its exact grid minimum from one table per variable. The overlapping
    # (x - y)**2 = x**2 + y**2 - 2 x y on {0, 1}**2 has minimum 0, while its three table minima sum to -2,
    # so its cap 2 exceeds the exact cap 0. g = x**2 + 1 >= 1 has the cap 0, adds no slack variable and
    # takes the exact quadratic form of its PHR term.
    endpoints = _qhd(num_grid_points=2, include_boundary_points=True)
    separable = x**2 + y - sp.Rational(1, 2)
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=((0.0, 1.0), (0.0, 1.0)),
                                      inequalities=(separable, (x - y) ** 2, x**2 + 1))
    pre, form, plan = plan_augmented_lagrangian(problem, qhd=endpoints, options=AugmentedLagrangian(
        inequality_form="slack"), execution="classical", seed=1)
    grid = list(itertools.product((0, 1), repeat=2))
    assert pre.inequality_lower[0] == float(min(separable.subs({x: a, y: b}) for a, b in grid)) == -0.5
    assert pre.inequality_upper[0] == float(max(separable.subs({x: a, y: b}) for a, b in grid)) == 1.5
    assert min((a - b) ** 2 for a, b in grid) == 0 and pre.inequality_lower[1] == -2.0
    assert pre.inequality_lower[2] == 1.0
    assert form.forms == ("slack", "slack", "quadratic")
    assert [(axis.inequality, axis.cap) for axis in form.slacks] == [(0, 0.5), (1, 2.0)]
    assert [str(v) for v in plan.problem.variables] == ["x", "y", "s_g0", "s_g1"] == list(form.variables)


def test_branch_tests_give_the_exact_quadratic_or_constant_phr_term_on_the_grid():
    # With mu = 1 and rho = 1, g1 = x + 2 has ell + mu/rho = 2.5 >= 0, so its PHR term is mu g + rho g**2/2
    # on the grid, and g2 = x - 3 has M + mu/rho = -1.5 <= 0, so its PHR term is -mu**2/(2 rho). Both inner
    # tables therefore equal the PHR objective at every grid point, and neither adds a slack variable.
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(x + 2, x - 3))
    options = AugmentedLagrangian(inequality_form="slack", inequality_multipliers=(1.0, 1.0))
    pre, form, plan = plan_augmented_lagrangian(problem, qhd=_qhd(), options=options, execution="classical", seed=1)
    assert form.forms == ("quadratic", "constant") and form.slacks == () and form.error_bound == 0.0
    for a, b in itertools.product(AXIS, repeat=2):
        a, b = sp.Rational(a), sp.Rational(b)
        phr = (a - 1) ** 2 + (b - 1) ** 2 + _phr(a + 2, 1, 1) + _phr(a - 3, 1, 1)
        assert _exact(plan.problem.objective, {x: a, y: b}) == phr
    # A round that reads an off-grid mean applies neither test, whose premises hold on the grid only: g1,
    # whose cap is 0, keeps its PHR term, and g2 gets a slack variable.
    mean = AugmentedLagrangian(inequality_form="slack", inequality_multipliers=(1.0, 1.0), inner_point="mode_or_mean")
    assert plan_augmented_lagrangian(problem, qhd=_qhd(), options=mean, execution="classical")[1].forms == (
        "phr", "slack")


def _affine_family(n):
    """f = sum_i (x_i - 1/2)**2 and g = sum_i x_i - 3 on [0, 1]**n, a many-variable affine family."""
    xs = sp.symbols(f"x0:{n}", real=True)
    return ConstrainedOptimization(objective=sum((v - sp.Rational(1, 2)) ** 2 for v in xs), variables=xs,
                                   bounds=((0.0, 1.0),) * n, inequalities=(sum(xs) - 3,))


BINARY = QHD(num_grid_points=4, encoding="binary", boundary="periodic", num_steps=1, total_time=0.001)


def test_the_three_inequality_forms_and_automatic_selection_by_execution_route():
    family = _affine_family(8)
    phr = plan_augmented_lagrangian(family, qhd=BINARY, execution="quantum", seed=7)
    assert phr[1] is None and len(phr[2].problem.variables) == 8
    # Automatic selection keeps the PHR form under classical execution, where a slack variable multiplies
    # the restricted state by K.
    classical = plan_augmented_lagrangian(family, qhd=BINARY.revise(theory_flavor="split_step"), execution="classical",
                                          options=AugmentedLagrangian(inequality_form="auto"), seed=7)
    assert classical[1].forms == ("phr",) and classical[1].trials == () and "classical" in classical[1].selection
    assert classical[2].reconstruction.restricted_dimension == 4**8
    # On the quantum route it plans the PHR form and the conversion and keeps the conversion, whose planned
    # host work and CX count are both smaller. The kept Plan is the forced conversion's Plan.
    _, auto, plan = plan_augmented_lagrangian(family, qhd=BINARY, execution="quantum",
                                              options=AugmentedLagrangian(inequality_form="auto"), seed=7)
    forced = plan_augmented_lagrangian(family, qhd=BINARY, execution="quantum",
                                       options=AugmentedLagrangian(inequality_form="slack"), seed=7)[2]
    trial, = auto.trials
    assert auto.forms == ("slack",) and trial.accepted
    assert trial.trial_work < trial.current_work and trial.trial_cx < trial.current_cx
    cx = [law.value for law in phr[2].construction.selections[0].resource_laws if law.metric == "cx"]
    assert cx == [trial.current_cx] and plan.content_id == forced.content_id
    assert plan.reconstruction.width == 2 * 9 and max(len(t.support) for t in plan.reconstruction.support_values) == 2
    # A two-variable affine constraint next to a convex objective gains nothing by a slack variable, so the
    # comparison keeps the PHR form, whose table work and CX count are both smaller.
    u, v = sp.symbols("u v", real=True)
    small = ConstrainedOptimization(objective=(u - sp.Rational(1, 2)) ** 2 + (v - sp.Rational(1, 2)) ** 2,
                                    variables=(u, v), bounds=((0.0, 1.0),) * 2, inequalities=(u + v - sp.Rational(1, 2),))
    kept = plan_augmented_lagrangian(small, qhd=BINARY.revise(num_grid_points=8), execution="quantum",
                                     options=AugmentedLagrangian(inequality_form="auto"), seed=7)[1]
    assert kept.forms == ("phr",) and not kept.trials[0].accepted
    assert kept.trials[0].trial_cx > kept.trials[0].current_cx and kept.trials[0].trial_work > kept.trials[0].current_work
    # With Qiskit's state preparation the constructions record no CX law. Under max_work = 3866, the slack form's
    # resource-inspection census, which quantum binary planning admits, the PHR form of
    # G = sum_(a=1..4) (u**a + v**a) - 4 next to F = -G**2/2 is refused and the slack form, whose expansion
    # cancels G**2, is admitted. A trial without a CX law does not replace a refused PHR form, so automatic
    # selection keeps it and the round raises its refusal.
    g = sum(u**a + v**a for a in range(1, 5)) - 4
    law_free = ConstrainedOptimization(objective=-g**2 / 2, variables=(u, v), bounds=((0.0, 1.0),) * 2,
                                       inequalities=(g,))
    tight = BINARY.revise(initial_state_preparation="qiskit_state_preparation", max_work=3866)
    forced = plan_augmented_lagrangian(law_free, qhd=tight, options=AugmentedLagrangian(inequality_form="slack"),
                                       execution="quantum", seed=7)[2]
    assert [law for selection in forced.construction.selections for law in selection.resource_laws] == []
    with pytest.raises(ValueError, match="requires"):
        plan_augmented_lagrangian(law_free, qhd=tight, options=AugmentedLagrangian(inequality_form="auto"),
                                  execution="quantum", seed=7)


def test_fourteen_variables_plan_with_a_slack_variable_where_the_phr_table_is_refused():
    # Planning only: the preprocessing checks the separable objective and the affine constraint by their
    # 14 one-variable tables, and nothing is prepared or evolved. The PHR table on all 14 variables needs
    # at least 4**14 (N + 14) work units and is refused before it is formed, while the slack form's tables
    # have at most two variables.
    from nwqlib.algorithms.qhd import constrained

    family = _affine_family(14)
    widths, tables = [], constrained._support_tables

    def counted(term, decomposer, grid, qhd):
        widths.extend(len(support) for support in decomposer.support_expressions)
        return tables(term, decomposer, grid, qhd)

    # No term has a possible restriction, so the grid check scans no original support.
    with patch("nwqlib.scientist.prepare", side_effect=AssertionError("prepared")), \
            patch("nwqlib.blocks.kernels.BoundKernel._invoke", side_effect=AssertionError("evolved")), \
            patch.object(constrained, "_domain_check", side_effect=AssertionError("original support scanned")), \
            patch.object(constrained, "_support_tables", side_effect=counted):
        with pytest.raises(ValueError):
            plan_augmented_lagrangian(family, qhd=BINARY, execution="quantum", seed=7)
        pre, form, plan = plan_augmented_lagrangian(family, qhd=BINARY, execution="quantum", seed=7,
                                                    options=AugmentedLagrangian(inequality_form="auto"))
    assert widths and set(widths) == {1}
    assert pre.inequality_lower == (-3.0,) and pre.inequality_upper == (14 * 0.75 - 3,)
    trial, = form.trials
    assert trial.accepted and trial.current_work is None and trial.trial_work <= BINARY.max_work
    assert form.slacks[0].cap == 3.0 and form.slacks[0].upper == 4.0
    assert plan.reconstruction.width == 2 * 15 and plan.reconstruction.restricted_dimension == 4**15
    assert max(len(t.support) for t in plan.reconstruction.support_values) == 2
    assert [law.value for law in plan.construction.selections[0].resource_laws if law.metric == "cx"] == [
        trial.trial_cx]


def test_forced_slack_rounds_keep_the_phr_update_at_the_projected_point():
    # f = (x - 1)**2 + (y - 1)**2 and g = x + y <= 0 on the interior 3-by-3 grid. Every converted round
    # records the projection of its joint point, and its residuals, multiplier update and penalty follow
    # from that point by the PHR rule alone (constrained._update). The effective value is L_k evaluated
    # there, and the inner value is the inner Plan's table value at the joint grid point, at least L_k.
    from nwqlib.algorithms.qhd.constrained import _effective_value, _update
    from nwqlib.algorithms.qhd.method import objective_at

    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(x + y,))
    result = _solve(problem, inequality_form="slack")
    options, previous, converted = result.record.options, None, 0
    for item, inner in zip(result.iterations, result.results):
        e = item.evaluation
        converted += bool(item.representation.slacks)
        f, g = float(F.subs({x: e.point[0], y: e.point[1]})), e.point[0] + e.point[1]
        assert (e.objective, e.inequality_residuals) == (f, (g,))
        update = _update(options, item.penalty, item.equality_multipliers, item.inequality_multipliers, (), (g,),
                         (), (1.0,), previous)
        assert (e.tentative_inequality_multipliers, e.next_inequality_multipliers, e.next_penalty, e.measure,
                e.complementarity, e.infeasibility) == (update["mu_plus"], update["mu_next"],
                                                         update["next_penalty"], update["measure"],
                                                         update["complementarity"], update["infeasibility"])
        previous = e.measure
        assert e.effective_value_source == "evaluated" and e.effective_value == _effective_value(
            item.equality_multipliers, item.inequality_multipliers, item.penalty, f, (), (g,))
        joint = e.indices + (e.slack_indices or ())
        assert inner.plan.problem.variables[:2] == (x, y) and len(joint) == len(inner.plan.problem.variables)
        assert e.inner_value_source == "table" and e.inner_value == objective_at(inner.plan.reconstruction, joint, 3)
        assert e.inner_value >= e.effective_value
        if e.slack_point is not None:
            assert e.slack_point == tuple(inner.candidate_coordinates[2:])
        assert e.probability == inner.candidate_probability
    assert converted and result.termination == "feasible_complementary"


def test_a_mixed_round_reads_its_joint_mean_and_keeps_its_zero_cap_inequality_in_phr_form():
    # mode_or_mean compares the inner objective at the joint mode and at the joint conditional mean, off the
    # grid, so the branch tests do not apply and g2 = x**2 + 1, whose cap is 0, keeps its PHR term, while
    # g1 = x + y gets a slack variable. The round's point is the projected mean when it wins, with its
    # slack coordinate recorded and L_k evaluated at the projection.
    from nwqlib.algorithms.qhd.constrained import _effective_value

    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(x + y, x**2 + 1))
    result = _solve(problem, inequality_form="slack", inner_point="mode_or_mean", max_iterations=2)
    for item, inner in zip(result.iterations, result.results):
        assert item.representation.forms == ("slack", "phr") and inner.plan.reconstruction.width == 9
        e = item.evaluation
        g = (e.point[0] + e.point[1], e.point[0] ** 2 + 1)
        assert e.inequality_residuals == g
        assert e.effective_value == _effective_value((), item.inequality_multipliers, item.penalty, e.objective, (),
                                                      g)
        if e.kind == "valid_mean":
            assert e.point + e.slack_point == tuple(inner.position_mean) and e.inner_value_source == "evaluated"
            # The joint objective at the mean, written independently: F + mu1 (g1 + s) + rho (g1 + s)**2/2
            # + ([mu2 + rho g2]_+**2 - mu2**2)/(2 rho), with unit scales. The tolerance allows for the rounding
            # of the two binary64 evaluations, which form the same terms in different orders.
            (mu1, mu2), rho, (s,) = item.inequality_multipliers, item.penalty, e.slack_point
            joint = (e.objective + mu1 * (g[0] + s) + rho * (g[0] + s) ** 2 / 2
                     + (max(mu2 + rho * g[1], 0.0) ** 2 - mu2**2) / (2 * rho))
            np.testing.assert_allclose(e.inner_value, joint, rtol=2e-14, atol=2e-14)
    assert any(item.evaluation.kind == "valid_mean" for item in result.iterations)
    # One round of F = (x**2 - 1/4)**2/8 and g = x on the interior 4-point grid of [-1, 1] with a slack variable:
    # L_k at the projected mean is below the joint objective at the joint mode, which is below the joint
    # objective at the joint mean, so comparing the mean by L_k would take the mean, and the joint comparison
    # keeps the joint mode. The expectations use the closed forms J(x, s) = (x*x - 1/4)**2/8 + (x + s)**2/2
    # and L(x) = (x*x - 1/4)**2/8 + max(x, 0)**2/2 of the round's zero multiplier and unit penalty.
    discriminator = ConstrainedOptimization(objective=sp.Rational(1, 8) * (x * x - sp.Rational(1, 4)) ** 2,
                                            variables=(x,), bounds=((-1.0, 1.0),), inequalities=(x,))
    result = solve_augmented_lagrangian(
        discriminator, qhd=QHD(num_grid_points=4, num_steps=8, total_time=1.0, keep_state=True),
        options=AugmentedLagrangian(inequality_form="slack", inner_point="mode_or_mean", max_iterations=1),
        execution="classical", seed=7, progress=False)
    inner = result.results[0]
    mode = tuple(inner.most_probable_coordinates)
    mean = tuple(inner.position_mean)

    def joint(point):
        x, s = point
        return (x*x - 0.25)**2 / 8 + (x + s)**2 / 2

    def phr(x):
        return (x*x - 0.25)**2 / 8 + max(x, 0.0)**2 / 2

    assert phr(mean[0]) < joint(mode) < joint(mean)
    chosen = result.iterations[0].evaluation
    assert chosen.kind == "grid_point"
    assert chosen.point + chosen.slack_point == mode


def test_refined_slack_rounds_keep_augmented_levels_and_start_from_the_original_box():
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(x + y,))
    result = solve_augmented_lagrangian(
        problem, qhd=QHD(num_grid_points=4, num_steps=4, total_time=1.0),
        options=AugmentedLagrangian(inequality_form="slack", max_iterations=3, inner_point="best_observed"),
        refinement=BoxRefinement(point_rule="best_observed", max_levels=3, mass_threshold=0.6),
        execution="classical", seed=5, progress=False)
    shrunk = False
    for item in result.iterations:
        axis, = item.representation.slacks
        levels = item.refinement.levels
        # Every level lives in the augmented coordinates, and each round starts from the preprocessed box
        # and its own slack box.
        assert levels[0].box == BOX + ((0.0, axis.upper),)
        assert all(len(level.box) == len(level.point) == 3 for level in levels)
        shrunk |= any(level.box[2] != (0.0, axis.upper) for level in levels[1:])
        best = item.refinement.best
        e = item.evaluation
        assert e.point + e.slack_point == best.point and e.inner_value == best.objective
    assert shrunk and len(result.iterations) >= 2
    with pytest.raises(ValueError, match="nested grids"):
        constrained_grid_minimum(result)


def test_a_converted_run_saves_loads_and_keeps_the_original_grid_reference(tmp_path):
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(x + y,))
    converted = _solve(problem, inequality_form="slack")
    plain = _solve(problem)
    assert any(item.representation.slacks for item in converted.iterations)
    loaded = load_augmented_lagrangian(converted.save(tmp_path / "saved"))
    assert loaded.record.content_id == converted.record.content_id and loaded.report() == converted.report()
    assert loaded.report()["inequality_forms"][0]["forms"] == {"0": "slack"}
    # The reference enumerates the original grid with the original feasibility test, as for PHR rounds.
    reference, expected = constrained_grid_minimum(converted), constrained_grid_minimum(plain)
    assert reference.grid_points == 9 and (reference.indices, reference.objective, reference.feasible_points) == (
        expected.indices, expected.objective, expected.feasible_points)


def test_a_gaussian_initial_state_refuses_forced_slack_and_keeps_automatic_selection_in_phr_form():
    from nwqlib.algorithms.qhd import GaussianState

    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(x + y,))
    gaussian = _qhd(initial_state=GaussianState(center=(0.0, 0.0), widths=(0.5, 0.5)))
    with pytest.raises(ValueError, match="GaussianState.*no extension to the slack axes"):
        _solve(problem, qhd=gaussian, inequality_form="slack")
    form = plan_augmented_lagrangian(problem, qhd=gaussian.revise(initial_state_preparation="qiskit_state_preparation"),
                                     options=AugmentedLagrangian(inequality_form="auto"), execution="quantum")[1]
    assert form.forms == ("phr",) and form.trials == () and "GaussianState" in form.selection


def _refused_before_planning(problem, qhd, options=AugmentedLagrangian()):
    """Assert that the layer refuses ``problem`` with ValueError before it plans a round."""
    with patch("nwqlib.scientist._plan_with_streams", side_effect=AssertionError("a round was planned")):
        with pytest.raises(ValueError):
            plan_augmented_lagrangian(problem, qhd=qhd, options=options, execution="classical", seed=7)


def test_the_grid_check_scans_a_term_whose_expansion_drops_a_restriction_and_certifies_a_large_objective():
    """constrained._setup scans the supplied summands of a term whose expansion drops a possible restriction.

    f = (x - x**2)/x expands to 1 - x, which is finite on the endpoint grid {0, 1} of [0, 1], while the original
    division fails at x = 0, so every form refuses before a round is planned. On [1/2, 1] the grid avoids the
    division by zero and the round plans. The unevaluated (x - 1)/(x - 1) on [1, 2] simplifies to 1 once it is
    centered at 1, so the restrictions are taken from the supplied expression and its division at x = 1 is
    refused, while log(x) on [1, 2], which the expansion keeps, is not scanned. The structural trigger scans
    every distinct summand of a term, an inconclusive range check reuses those scans, and two identical summands
    share one scan and enter the addition bound twice (constrained._addition_certificate). For 1e308 (1 - x**2)
    on {-1, 1} the support magnitudes sum beyond binary64, while the signed range of the tabulated sum is
    exactly [0, 0], so the objective normalized by objective_scale = 1e308 plans (constrained._range_certificate).
    When neither bound certifies the range, the summands are scanned, and the whole term is scanned when their
    addition bound fails: 1e308 (x + y + z - x y z)/2 plans, while 1e308 (x + y + z) and 1e308 x + 1e308 y,
    whose finite summands overflow when added at (-1, ..., -1), are refused.
    """
    from nwqlib.algorithms.qhd import constrained

    qhd = QHD(num_grid_points=2, include_boundary_points=True, num_steps=1, total_time=0.001)

    def problem(objective, box):
        return ConstrainedOptimization(objective=objective, variables=(z,), bounds=(box,),
                                       inequalities=(sp.Integer(-1),))

    for form in ("phr", "slack", "auto"):
        _refused_before_planning(problem((z - z**2) / z, (0.0, 1.0)), qhd, AugmentedLagrangian(inequality_form=form))
    assert plan_augmented_lagrangian(problem((z - z**2) / z, (0.5, 1.0)), qhd=qhd, execution="classical",
                                     seed=7)[2].problem.variables == (z,)
    unevaluated = sp.Mul(z - 1, sp.Pow(z - 1, -1, evaluate=False), evaluate=False)
    _refused_before_planning(problem(unevaluated, (1.0, 2.0)), qhd)
    with patch.object(constrained, "_domain_check", side_effect=AssertionError("an unneeded scan")):
        plan_augmented_lagrangian(problem(sp.log(z), (1.0, 2.0)), qhd=qhd, execution="classical", seed=7)
    # On the 3-point grid the structural trigger scans the distinct summands of each term, eight scans in all,
    # and forcing both range checks to be inconclusive requests no further scan and no further charge.
    small = QHD(num_grid_points=3, include_boundary_points=True, num_steps=1, total_time=0.001)
    canceled = (z + sp.log(z)) ** 2 - sp.expand((z + sp.log(z)) ** 2)
    both = ConstrainedOptimization(objective=canceled + z * z, variables=(z,), bounds=((1.0, 2.0),),
                                   inequalities=(canceled - 1,))
    scan, runs = constrained._domain_check, []
    for certified in (constrained._range_certificate, lambda constant, tables: False):
        with patch.object(constrained, "_range_certificate", side_effect=certified), \
                patch.object(constrained, "_domain_check", wraps=scan) as scans:
            setup = constrained._setup(both, small, AugmentedLagrangian())
        runs.append(([call.args[0].expression for call in scans.call_args_list], setup.check_evaluations,
                     setup.check_work))
    assert runs[0] == runs[1] and len(runs[0][0]) == 8 and runs[0][1] == 25
    certify = constrained._addition_certificate
    with patch.object(constrained, "_domain_check", wraps=scan) as scans, \
            patch.object(constrained, "_addition_certificate", wraps=certify) as bounds:
        setup = constrained._setup(problem(sp.Add(z**-2, z**-2, evaluate=False), (1.0, 2.0)), small,
                                   AugmentedLagrangian())
    maxima, = bounds.call_args.args
    assert scans.call_count == 1 and setup.check_evaluations == 6
    assert bounds.call_count == 1 and len(maxima) == 2 and maxima[0] == maxima[1]
    # With F the largest binary64 value and h = 2**971 the spacing below it, F - 2h, 3h/4, 3h/4 and h/2 sum
    # exactly to F, but their sequential rounded sum overflows, so the addition bound, which includes the
    # rounding of every addition, does not certify them.
    largest, h = float(np.finfo(np.float64).max), 2.0**971
    leaves = (largest - 2 * h, 0.75 * h, 0.75 * h, 0.5 * h)
    total = leaves[0]
    for leaf in leaves[1:]:
        total += leaf
    assert sum(map(Fraction, leaves)) == Fraction(largest) and total == float("inf")
    assert not constrained._addition_certificate(leaves)
    large = problem(sp.Float(1e308) * (1 - z**2), (-1.0, 1.0))
    plan = plan_augmented_lagrangian(large, qhd=qhd, options=AugmentedLagrangian(objective_scale=1e308),
                                     execution="classical", seed=7)[2]
    assert [t.values.array.tolist() for t in plan.reconstruction.support_values] == [[-1.0, -1.0]]
    for objective, admitted in ((sp.Float(1e308) / 2 * (x + y + z - x * y * z), True),
                                (sp.Float(1e308) * (x + y + z), False)):
        cube = ConstrainedOptimization(objective=objective, variables=(x, y, z), bounds=((-1.0, 1.0),) * 3,
                                       inequalities=(sp.Integer(-1),))
        if admitted:
            plan_augmented_lagrangian(cube, qhd=qhd, options=AugmentedLagrangian(objective_scale=1e308),
                                      execution="classical", seed=7)
        else:
            _refused_before_planning(cube, qhd, AugmentedLagrangian(objective_scale=1e308))
    # Both summands are finite, their addition bound fails, and the scan of the whole term refuses.
    pair = ConstrainedOptimization(objective=sp.Float(1e308) * x + sp.Float(1e308) * y, variables=(x, y),
                                   bounds=((-1.0, 1.0),) * 2, inequalities=(sp.Integer(-1),))
    with patch.object(constrained, "_domain_check", wraps=scan) as scans:
        _refused_before_planning(pair, qhd)
    assert [len(call.args[0].support) for call in scans.call_args_list] == [1, 1, 2]


def test_the_summand_scans_follow_numerical_piecewise_selection_and_evaluation_failures():
    """The scans evaluate the supplied summands in binary64, selecting Piecewise branches as NumPy does.

    On the endpoint grid {-1, 0, 1}, the guarded Piecewise((x + sqrt(x))**2, x >= 0), (x**2, True)) selects
    x**2 at -1 and takes 1, 0 and 4, and the run finds objective 0 at x = 0. With u = Piecewise((1, x < 0),
    (2, sqrt(x) > 1/2), (1, True)), the canceled (x + log(u))**2 minus its expansion plus x**2 selects u = 1,
    1 and 2, so it is admitted as well. With v = log(x), (x + v)**2 - v**2 - 2 x v evaluates log(x) at every
    point, and its expansion keeps only the log(x) of a branch active for x > 0, so the summands fail at -1 and 0
    and the term is refused before a round is planned. On {0, 1/2, 1}, the summand (x + sec(pi x))**20 overflows
    at 1/2, where the rounded secant is about 1.6e16, so the term is refused. The Piecewise condition 1/x > 2
    raises a division by zero at 0. The branch 1/x of Piecewise((1/x, x >= 0), (x**2, True)) is selected at 0, so
    the support table's selected infinity is refused, whereas Piecewise((1/x, x > 1/2), (x**2, True)) selects 1, 0
    and 1: its unselected 1/x gives a nonfinite intermediate at 0 but a finite real table, which setup accepts
    (constrained._support_tables). A Piecewise stays one summand, so its branch (x - 1)/(x - 1), unevaluated,
    raises at x = 1 of [1, 2]. The polynomial summand of 10**-308 (x + 10**160)**2 + x**-2 overflows in its
    original arithmetic although its expansion is small, and it is scanned with its companion x**-2, so the term
    is refused. A power written with the integer-valued binary64 exponent 2.0 is evaluated like its integer power
    and is admitted.
    """
    qhd = QHD(num_grid_points=3, include_boundary_points=True, num_steps=1, total_time=0.001, keep_state=True)
    options = AugmentedLagrangian(inner_point="best_observed", max_iterations=1)

    def problem(objective, box=(-1.0, 1.0)):
        return ConstrainedOptimization(objective=objective, variables=(z,), bounds=(box,),
                                       inequalities=(sp.Integer(-1),))

    u = sp.Piecewise((1, z < 0), (2, sp.sqrt(z) > sp.Rational(1, 2)), (1, True))
    for admitted in (sp.Piecewise(((z + sp.sqrt(z)) ** 2, z >= 0), (z * z, True)),
                     (z + sp.log(u)) ** 2 - sp.expand((z + sp.log(u)) ** 2) + z * z):
        result = solve_augmented_lagrangian(problem(admitted), qhd=qhd, options=options, execution="classical",
                                            seed=7, progress=False)
        assert result.candidate == (0.0,) and result.objective == 0.0
        assert constrained_grid_minimum(result).objective == 0.0
    plan_augmented_lagrangian(problem((z + z**2.0) ** 2 - sp.expand((z + z**2.0) ** 2) + z**2), qhd=qhd,
                              execution="classical")
    plan_augmented_lagrangian(problem(sp.Piecewise((1 / z, z > sp.Rational(1, 2)), (z**2, True))), qhd=qhd,
                              execution="classical")
    v = sp.log(z)
    secant = sp.sec(sp.pi * z)
    quotient = sp.Mul(z - 1, sp.Pow(z - 1, -1, evaluate=False), evaluate=False)
    for refused, box in (((v + z) ** 2 - v**2 - 2 * z * v + sp.Piecewise((v, z > 0), (0, True)) + 1 - 2 * z,
                          (-1.0, 1.0)),
                         ((z + secant) ** 20 - sp.expand((z + secant) ** 20) + 1 - z, (0.0, 1.0)),
                         (sp.Piecewise(((z + 1) ** 2, 1 / z > 2), (z**2, True)), (-1.0, 1.0)),
                         (sp.Piecewise((1 / z, z >= 0), (z**2, True)), (-1.0, 1.0)),
                         (sp.Piecewise((quotient, z < sp.Rational(3, 2)), (2, True), evaluate=False), (1.0, 2.0)),
                         (sp.Rational(1, 10**308) * (z + sp.Integer(10) ** 160) ** 2 + z**-2, (1.0, 2.0))):
        _refused_before_planning(problem(refused, box), qhd, options)


def test_the_domain_scan_refuses_an_array_value_with_an_imaginary_residue():
    # An original summand must be finite and real at every grid point. The array scan evaluates
    # z**2 + 1e-13 i on the grid of [-1, 1] as complex128 with imaginary part 1e-13, which is not real however
    # small, so the scan refuses it at the first grid point (constrained._domain_check).
    from types import SimpleNamespace

    from nwqlib.algorithms.qhd import constrained
    from nwqlib.algorithms.qhd.grid import OneHotGrid

    expression = z**2 + sp.Float(1e-13) * sp.I
    term = SimpleNamespace(name="inequalities[0]", expression=expression, support=(0,),
                           nodes=node_count(expression, 10**6))
    grid = OneHotGrid(("z",), ((-1.0, 1.0),), 4, True, "dirichlet")
    with pytest.raises(ValueError, match=r"inequalities\[0\] = .* at \{'z': -1\.0\} is not a finite real value"):
        constrained._domain_check(term, (z,), grid, QHD())


def test_fourteen_variable_terms_scan_one_variable_summands_or_are_refused_before_any_evaluation():
    # f = sum_i (x_i - 3/2)**2 + x_i**-2 on [1, 2]**14 is separable, but its expansion rewrites the restriction
    # of each x_i**-2, so the grid check scans the 28 one-variable summands of f instead of the 4**14 tuples of
    # the whole term: 112 table entries of f and g and 112 scan evaluations. The forced slack form then plans,
    # with nothing prepared or evolved.
    from nwqlib.algorithms.qhd import constrained

    xs = sp.symbols("x0:14", real=True)
    family = ConstrainedOptimization(objective=sum((t - sp.Rational(3, 2)) ** 2 + t**-2 for t in xs), variables=xs,
                                     bounds=((1.0, 2.0),) * 14,
                                     inequalities=(sum(xs) - sp.Rational(3, 2) * 14 + sp.Rational(1, 4),))
    options = AugmentedLagrangian(inequality_form="slack", max_iterations=1)
    scan = constrained._domain_check

    def one_variable(term, variables, grid, qhd):
        if len(term.support) > 1:
            raise AssertionError("a whole-support scan")
        return scan(term, variables, grid, qhd)

    with patch("nwqlib.scientist.prepare", side_effect=AssertionError("prepared")), \
            patch("nwqlib.blocks.kernels.BoundKernel._invoke", side_effect=AssertionError("evolved")), \
            patch.object(constrained, "_domain_check", side_effect=one_variable) as scans:
        assert constrained._setup(family, BINARY, options).check_evaluations == 224 and scans.call_count == 28
        _, form, plan = plan_augmented_lagrangian(family, qhd=BINARY, options=options, execution="quantum", seed=7)
    assert form.forms == ("slack",) and len(plan.problem.variables) == 15
    # With v = log(sum_i x_i), the expansion keeps the log, and the admission refuses its 14-variable support
    # table. (x_0 + v)**2 - v**2 - 2 x_0 v + sum_(i>0) x_i**2 is defined on the box and expands to a separable
    # sum, but the canceled log requests the scans of its summands, which the admission refuses because the
    # summand (x_0 + v)**2 has all 14 variables. Both are refused before any table or scan is evaluated. Each is
    # the last kept term, the inequality, so no later term's admission can refuse in place of its own.
    v = sp.log(sum(xs))
    for g in (v, (xs[0] + v) ** 2 - v * v - 2 * xs[0] * v + sum(t * t for t in xs[1:])):
        expensive = ConstrainedOptimization(objective=sum(xs), variables=xs, bounds=((1.0, 2.0),) * 14,
                                            inequalities=(g - 100,))
        with patch.object(constrained, "_domain_check", side_effect=AssertionError("scanned")), \
                patch.object(constrained, "_support_tables", side_effect=AssertionError("tabulated")):
            _refused_before_planning(expensive, BINARY)
    # The byte admission of every term's support tables also precedes the first evaluation. The inequality
    # g = sum_i sin(i x) cos(i y) - 1 has over a thousand expression nodes, so one entry of its table evaluation
    # needs more than max_bytes, and the objective's tables are not evaluated either.
    x, y = xs[:2]
    g = sum(sp.sin(i * x) * sp.cos(i * y) for i in range(1, 120)) - 1
    heavy = ConstrainedOptimization(objective=(x - sp.Rational(1, 5)) ** 2 + (y + sp.Rational(1, 10)) ** 2,
                                    variables=(x, y), bounds=((-1.0, 1.0),) * 2, inequalities=(g,))
    with patch.object(constrained, "_domain_check", side_effect=AssertionError("scanned")), \
            patch.object(constrained, "_support_tables", side_effect=AssertionError("tabulated")), \
            pytest.raises(ValueError, match=r"inequalities\[0\] by its support tables requires at least \d+ bytes"):
        constrained._setup(heavy, QHD(num_grid_points=8, num_steps=1, total_time=0.1, max_bytes=250_000),
                           AugmentedLagrangian(inequality_form="phr"))


def test_automatic_selection_charges_its_trial_plannings_in_the_admitted_bound():
    # One round on the quantum route with one kept inequality: the current representation and one trial
    # each make at most two ledger and four planning admissions, and the round's Plan is one of them, so
    # the bound is (4 M + (6 (m + 1) - 4) M + 1) max_work with M = 1 round and m = 1 (constrained._admitted_bounds).
    # The PHR policy makes no trial.
    problem = ConstrainedOptimization(objective=F, variables=(x, y), bounds=BOX, inequalities=(x + y,))
    qhd = _qhd(keep_state=False)
    bounds = {}
    for form in ("phr", "auto"):
        result = solve_augmented_lagrangian(problem, qhd=qhd, options=AugmentedLagrangian(inequality_form=form,
                                                                                         max_iterations=1),
                                            execution="quantum", seed=11, progress=False)
        bounds[form] = result.record.admitted_work_bound, result.record.admitted_bytes_bound
        if form == "auto":
            item = result.iterations[0]
            trial, = item.representation.trials
            assert item.plan_id is not None and trial.current_work is not None
    assert bounds["phr"] == (5 * qhd.max_work, 4 * qhd.max_bytes)
    assert bounds["auto"] == (13 * qhd.max_work, 12 * qhd.max_bytes)
    # The round's table evaluations count both candidates once, the PHR form and the slack trial, whichever the
    # round kept and reused (ALResources.table_evaluations).
    candidates = [plan_augmented_lagrangian(problem, qhd=qhd, options=AugmentedLagrangian(inequality_form=form),
                                            execution="quantum", seed=11)[2] for form in ("phr", "slack")]
    counts = [plan.reconstruction.support_evaluations for plan in candidates]
    assert item.resources.table_evaluations == item.representation.table_evaluations == sum(counts)
    assert result.resources.table_evaluations == sum(counts)
    # A refined round reuses no candidate, so it adds the candidates' count to its levels' tables, the unscaled
    # table stage and the solved Plan of its one search-model level.
    refined = solve_augmented_lagrangian(problem, qhd=qhd, options=AugmentedLagrangian(inequality_form="auto",
                                                                                      max_iterations=1),
                                         refinement=BoxRefinement(max_levels=1), execution="quantum", seed=11,
                                         progress=False).iterations[0]
    levels = refined.refinement.resources
    assert refined.representation.table_evaluations == sum(counts)
    assert refined.resources.table_evaluations == levels.table_evaluations + levels.support_evaluations + sum(counts)


def test_a_refined_level_checks_the_original_functions_before_its_mean_can_win():
    """Each refinement level checks f, h and g at the projection of every point it compares (constrained._level_check).

    f = (x - x**2)/x on the interior grid {-1/4, 1/4} of [-3/4, 3/4] is defined on the grid, and its expansion
    1 - x is finite at the joint mean (0, 1/8) of a round with a slack variable, where f itself is 0/0. The
    level records why the mean is unavailable and keeps its valid grid point, as the unrefined round does. The
    round charges the checks of f and g at the level's grid point and mean and its own evaluation at its point.
    """
    problem = ConstrainedOptimization(objective=(x - x * x) / x, variables=(x,), bounds=((-0.75, 0.75),),
                                      inequalities=(x,))
    qhd = QHD(num_grid_points=2, num_steps=1, total_time=1e-20, keep_state=True, initial_state=UniformState())
    options = AugmentedLagrangian(inequality_form="slack", inner_point="mode_or_mean", max_iterations=1)
    result = solve_augmented_lagrangian(problem, qhd=qhd, options=options, execution="classical", seed=7,
                                        refinement=BoxRefinement(scaling="physical", point_rule="mode_or_mean",
                                                                 max_levels=1), progress=False)
    item = result.iterations[0]
    level, = item.refinement.levels
    assert item.evaluation.kind == "grid_point" and item.evaluation.point == (-0.25,)
    assert "at (0.0,) is not a finite real value" in level.mean_unavailable
    assert item.resources.layer_evaluations == 6


# A refined slack round on f = x + (x - p)/(x - p), kept as supplied so that f itself is 0/0 at x = p while QHD
# tabulates its expansion x + 1, with the constant inequality -1 <= 0, on [-1, 1] with K = 5 endpoint grids,
# physical scaling, three levels, eta = 0.95 and one allowed stall split. Injected x marginals times a uniform
# slack factor make level 1 keep [-1, 1/4], whose grid is (-1, -11/16, -3/8, -1/16, 1/4), make level 2 stall
# with its clearest valley at index 2, and put level 3's mass on its last x point.
SPLIT_FIRST, SPLIT_LAST = (0.3, 0.5, 0.19, 0.005, 0.005), (0.0, 0.0, 0.0, 0.0, 1.0)
SPLIT_REFINEMENT = BoxRefinement(scaling="physical", max_levels=3, max_no_improve=3, mass_threshold=0.95,
                                 stall_split="best_region")


def _split_problem(pole):
    quotient = sp.Mul(x - pole, sp.Pow(x - pole, -1, evaluate=False), evaluate=False)
    return ConstrainedOptimization(objective=sp.Add(x, quotient, evaluate=False), variables=(x,),
                                   bounds=((-1.0, 1.0),), inequalities=(sp.Integer(-1),))


def _split_run(problem, stalled, monkeypatch, refinement=SPLIT_REFINEMENT, inner_point="most_probable",
               max_work=100_000_000):
    """Solve one refined round with the injected level marginals, returning the result and every point checked."""
    from nwqlib.algorithms.qhd import constrained

    rows = iter((SPLIT_FIRST, stalled, SPLIT_LAST))
    monkeypatch.setattr("nwqlib.algorithms.qhd.method._evolve_restricted",
                        lambda *args: np.kron(np.sqrt(next(rows)), np.full(5, 1 / np.sqrt(5))).astype(complex))
    checked, evaluate = [], constrained._evaluate

    def recorded(setup, point):
        checked.append(point)
        return evaluate(setup, point)

    monkeypatch.setattr(constrained, "_evaluate", recorded)
    qhd = QHD(num_grid_points=5, include_boundary_points=True, keep_state=True, max_work=max_work)
    options = AugmentedLagrangian(inequality_form="slack", inner_point=inner_point, max_iterations=1)
    result = solve_augmented_lagrangian(problem, qhd=qhd, options=options, refinement=refinement,
                                        execution="classical", seed=541, progress=False)
    return result, checked


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_a_stall_split_checks_the_original_functions_before_a_scored_point_decides(side, monkeypatch):
    """A split's scored points are compared points, checked before their scores are used (constrained._level_check).

    At the level-2 marginal (0.15, 0.3, 0.01, 0.19, 0.35) the lower side scores x = -11/16 and the upper side
    x = 1/4, and at (0.15, 0.35, 0.01, 0.3, 0.19) they score -11/16 and -1/16. With the pole at one scored
    point, the expansion x + 1 alone would score the lower side 13/16 against 7/4 and discard 0.54 of the
    level's distribution. The check fails first, so level 2 fails as one whose own point fails: no split and
    no next box is decided, the refinement keeps level 1 and stops with inner_failed, and the round takes
    level 1's point. The failed level keeps its Run's work, its grid-point score, the lower side's score when
    the upper side fails, and its observed-value reads, K**2 = 25 read once for its joint box mass and
    reused by the point weights. The round charges every attempted check of f and g, the failed one included.
    """
    pole, stalled, level_point, scored = {
        "lower": (sp.Rational(-11, 16), (0.15, 0.3, 0.01, 0.19, 0.35), (0.25,), [(-0.6875,)]),
        "upper": (sp.Rational(-1, 16), (0.15, 0.35, 0.01, 0.3, 0.19), (-0.6875,), [(-0.6875,), (-0.0625,)]),
    }[side]
    result, checked = _split_run(_split_problem(pole), stalled, monkeypatch)
    item = result.iterations[0]
    refined, stopped = item.refinement, item.refinement.stopped_resources
    assert refined.termination == "inner_failed"
    assert f"at ({float(pole)!r},) is not a finite real value" in refined.failure
    level, = refined.levels
    assert level.split is None and level.next_box == ((-1.0, 0.25), (0.0, 1.0))
    assert item.evaluation.point == level.point[:1] == (-0.5,)
    assert checked == [(-0.5,), level_point, *scored, (-0.5,)]
    # The level scored its grid point and each side scored before the failing check.
    completed = len(scored) - 1
    assert stopped.objective_evaluations == 1 + completed and stopped.joint_mass_reads == 25
    assert stopped.evolution_work > 0 and stopped.support_evaluations > 0
    assert item.resources.refinement_evaluations == 2 + completed and item.resources.joint_mass_reads == 50
    assert item.resources.layer_evaluations == 2 * len(checked)


def test_a_stall_split_at_finite_scored_points_keeps_its_choice_and_the_reserved_checks(monkeypatch):
    """With the pole at -7/16, off every checked point, the split keeps the scores and region of the rule.

    The scores are the joint inner objective L(x, s) at s = 0: x + 1 plus the slack term (G + s)**2/2 = 1/2
    of G = -1 with mu = 0 and rho = 1, so 13/16 at x = -11/16 and 7/4 at x = 1/4, and relative to the first
    level's constant 3/2 they are x itself. The lower side wins and keeps the valley cell, 0.15 + 0.3 + 0.01
    of the level's x marginal. Both scored points pass the check at their reported coordinates.

    The layer reserves (a L + 2 B + 1) C per round before any round, with C one evaluation of f and g, a = 2
    for mode_or_mean and 1 otherwise, L = max_levels and B = min(max_splits, max(L - 1, 0)) attempted splits
    with stall_split="best_region" and 0 otherwise (constrained._setup). One unit less refuses a one-round
    run with that reservation. This run's checks, three level points and two scored points, and the round's
    own evaluation use exactly its reservation of 6 C.
    """
    problem = _split_problem(sp.Rational(-7, 16))
    result, checked = _split_run(problem, (0.15, 0.3, 0.01, 0.19, 0.35), monkeypatch)
    item = result.iterations[0]
    split = item.refinement.levels[1].split
    assert [level.level for level in item.refinement.levels] == [1, 2, 3]
    assert split.points == ((-0.6875, 0.0), (0.25, 0.0)) and split.scores == (0.8125, 1.75)
    assert split.relative_scores == (-0.6875, 0.25) and split.chosen == 0
    assert item.refinement.levels[2].box == split.regions[0]
    assert checked == [(-0.5,), (0.25,), (-0.6875,), (0.25,), (-0.21875,), (-0.5,)]
    cost = sum(node_count(term, 100) + 1 for term in (problem.objective, *problem.inequalities))
    assert item.resources.layer_work == (1 * 3 + 2 * 1 + 1) * cost
    for reserved, refinement, rule in (
        ((1 * 3 + 2 * 1 + 1) * cost, SPLIT_REFINEMENT, "most_probable"),
        ((1 * 3 + 2 * 2 + 1) * cost, SPLIT_REFINEMENT.revise(max_splits=5), "most_probable"),
        ((2 * 3 + 2 * 1 + 1) * cost, SPLIT_REFINEMENT.revise(point_rule="mode_or_mean"), "mode_or_mean"),
        ((1 * 3 + 1) * cost, SPLIT_REFINEMENT.revise(stall_split="none"), "most_probable"),
        ((1 * 1 + 1) * cost, SPLIT_REFINEMENT.revise(max_levels=1), "most_probable"),
        (1 * cost, None, "most_probable"),
    ):
        with pytest.raises(ValueError, match=rf"needs {reserved} work units.*: 0 to check .* and {reserved} per "
                                             "round for 1 rounds"):
            _split_run(problem, (0.15, 0.3, 0.01, 0.19, 0.35), monkeypatch, refinement=refinement,
                       inner_point=rule, max_work=reserved - 1)


def test_mode_choices_copy_their_readout_status_and_other_choices_copy_none(tmp_path, monkeypatch):
    """The status of a chosen most probable grid point is its inner readout's, or the best level's with refinement.

    f = (z - 1/5)**2 with the inactive z <= 9/10 on the interior grid {-1/2, 0, 1/2}. The injected
    amplitudes (1, 1, 0)/sqrt(2) give indices 0 and 1 the same computed probability, so the readout is
    unresolved with representative 0 at z = -1/2. With refinement the second level gets all mass on its
    first point, z = -11/16 on the box [-1, 1/4], whose readout is resolved and whose objective is
    larger, so the best level is the first and not the last: the round copies the first level's status.
    best_observed takes the candidate, and mode_or_mean takes the level's mean -1/4, below the
    representative in objective, so neither copies a status while every level keeps its own. The copies
    survive the archive.
    """
    tie = np.array([1.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
    first = np.array([1.0, 0.0, 0.0], dtype=complex)
    problem = ConstrainedOptimization(objective=(z - sp.Rational(1, 5)) ** 2, variables=(z,), bounds=((-1.0, 1.0),),
                                      inequalities=(z - sp.Rational(9, 10),))

    def run(rule, refined):
        calls = itertools.count()
        monkeypatch.setattr("nwqlib.algorithms.qhd.method._evolve_restricted",
                            lambda *args: (tie, first)[next(calls) % 2] if refined else tie)
        return solve_augmented_lagrangian(
            problem, qhd=QHD(num_grid_points=3), options=AugmentedLagrangian(inner_point=rule, max_iterations=1),
            refinement=BoxRefinement(point_rule=rule, max_levels=2) if refined else None,
            execution="classical", seed=11, progress=False)

    plain = run("most_probable", False)
    item, inner = plain.iterations[0], plain.results[0]
    assert inner.most_probable_indices == (0,) and inner.mode_status == "unresolved"
    assert item.evaluation.indices == (0,) and item.evaluation.mode_status == "unresolved"
    with pytest.raises(ValueError, match="mode status belongs to a most probable grid point"):
        item.evaluation.revise(mode_status=None)
    # Without slack axes the summary names the inner point, not a joint point.
    assert "Mode status of the selected inner point: unresolved." in str(plain).splitlines()
    refined = run("most_probable", True)
    item = refined.iterations[0]
    levels, results = item.refinement.levels, refined.results[0]
    assert [level.mode_status for level in levels] == [r.mode_status for r in results] == ["unresolved", "resolved"]
    assert item.refinement.best_level == 1 < len(levels)
    assert item.evaluation.mode_status == "unresolved" and item.evaluation.tie_deficit == 0.0
    for rule in ("best_observed", "mode_or_mean"):
        other = run(rule, True).iterations[0]
        assert other.evaluation.mode_status is None
        with pytest.raises(ValueError, match="mode status belongs to a most probable grid point"):
            other.evaluation.revise(mode_status="resolved")
        assert other.evaluation.kind == ("valid_mean" if rule == "mode_or_mean" else "grid_point")
        assert [level.mode_status for level in other.refinement.levels] == ["unresolved", "resolved"]
    path = refined.save(tmp_path / "refined")
    loaded = load_augmented_lagrangian(path)
    assert loaded.iterations[0].evaluation.mode_status == "unresolved"
    assert [level.mode_status for level in loaded.iterations[0].refinement.levels] == ["unresolved", "resolved"]


def test_a_slack_round_copies_the_joint_status_whose_projection_is_not_the_marginal_mode(monkeypatch):
    """With a slack variable the round's point is the projection of the joint mode, with the joint status.

    f = (z - 1)**2 with g = z <= 0, forced into slack form on the two-point grids of z and its slack. The
    injected joint probabilities (0.30, 0.29; 0.31, 0.10), rows z, have their unique maximum at the joint
    point (1, 0), 0.01 above the next, far beyond the host window, so the readout is resolved. Its
    projection z index 1 is not the mode of the z marginal (0.59, 0.41). The round copies the joint
    status, which does not describe that marginal, and the run summary names it the joint point's status.
    """
    joint = np.sqrt(np.array([0.30, 0.29, 0.31, 0.10])).astype(complex)
    monkeypatch.setattr("nwqlib.algorithms.qhd.method._evolve_restricted", lambda *args: joint)
    problem = ConstrainedOptimization(objective=(z - 1) ** 2, variables=(z,), bounds=((-1.0, 1.0),),
                                      inequalities=(z,))
    result = solve_augmented_lagrangian(
        problem, qhd=QHD(num_grid_points=2), options=AugmentedLagrangian(inequality_form="slack", max_iterations=1),
        execution="classical", seed=11, progress=False)
    item, inner = result.iterations[0], result.results[0]
    assert item.representation.slacks and len(inner.plan.problem.variables) == 2
    assert inner.most_probable_indices == inner.probability_maximizer_indices == (1, 0)
    assert item.evaluation.indices == (1,) and item.evaluation.slack_indices == (0,)
    assert item.evaluation.mode_status == inner.mode_status == "resolved"
    marginal = inner.marginals.array[0].tolist()
    assert marginal[0] > marginal[1]
    assert "Mode status of the selected inner joint point: resolved." in str(result).splitlines()


def _reopened_problem(form):
    """A one-variable problem on [-1, 1] in the form ``form``, for the reopening tests below.

    q = (x + 11/16)/(x + 11/16) is kept as supplied, so it is 0/0 at x = -11/16 and 1 elsewhere, while
    SymPy's evaluation turns it into 1. "objective" is f = x + q with the constant inequality -1 <= 0.
    "constraints" is f = (x - 1/4)**2 with h = x + q - 1 = 0 and g = 2 x + q - 2 <= 0, and "evaluated" is
    the same problem with h = x and g = 2 x - 1, the evaluated forms of those constraints. "sum" is the
    evaluated problem with f = sum_{n=0}^{3} x**n/n!, whose ``Sum`` constructor multiplies the summand by
    1. The pole is no point of the 5-point endpoint grid, so the pre-round domain check admits every form.
    """
    if form == "objective":
        return _split_problem(sp.Rational(-11, 16))
    q = sp.Mul(x + sp.Rational(11, 16), sp.Pow(x + sp.Rational(11, 16), -1, evaluate=False), evaluate=False)
    h, g = ((sp.Add(x, q, -1, evaluate=False), sp.Add(2 * x, q, -2, evaluate=False)) if form == "constraints"
            else (x, 2 * x - 1))
    n = sp.Symbol("n", integer=True)
    f = sp.Sum(x**n / sp.factorial(n), (n, 0, 3)) if form == "sum" else (x - sp.Rational(1, 4)) ** 2
    return ConstrainedOptimization(objective=f, variables=(x,), bounds=((-1.0, 1.0),), equalities=(h,),
                                   inequalities=(g,))


def _expressions(problem):
    """The ``srepr`` of the SymPy objects that the archives pickle, in their order."""
    return sp.srepr((problem.objective, problem.variables, problem.equalities, problem.inequalities))


def _contents(path):
    """Each file of a saved archive: its bytes, or for a pickle the ``srepr`` of the SymPy objects it holds.

    Pickle bytes also record which equal SymPy objects are one object, which SymPy's construction cache
    decides, so a pickle is compared by its expressions.
    """
    from nwqlib.algorithms.qhd.archive import _SymbolicReader

    def read(file):
        if file.suffix != ".pickle":
            return file.read_bytes()
        with file.open("rb") as stream:
            return sp.srepr(_SymbolicReader(stream).load())
    return {p.relative_to(path).as_posix(): read(p) for p in sorted(path.rglob("*")) if p.is_file()}


def _reopened_run(problem, refinement, directory):
    return solve_augmented_lagrangian(problem, qhd=QHD(num_grid_points=5, include_boundary_points=True),
                                      options=AugmentedLagrangian(feasibility_tolerance=1e-9, max_iterations=1),
                                      refinement=refinement, execution="classical", seed=1, directory=directory,
                                      progress=False)


@pytest.mark.parametrize("refinement", [None, BoxRefinement(max_levels=1)], ids=["plain", "refined"])
@pytest.mark.parametrize("form", ["objective", "constraints", "evaluated", "sum"])
def test_a_reopened_problem_keeps_its_supplied_expressions(tmp_path, form, refinement):
    """A saved archive and an ended durable directory reopen the objective and constraints as they were supplied.

    The archives pickle the live SymPy objects, and ``archive._SymbolicReader`` builds each node as plain
    unpickling does, and again with evaluation disabled when that changes the node. Plain unpickling
    evaluates q to 1, and disabling evaluation for the whole load wraps the Sum's summand in Mul(1, ...),
    so either reader would give a problem, and inner Plan objectives, that differ from the recorded
    expressions and identities. Both reopenings keep the ``srepr`` of every expression, the problem's
    content identity and the run's record, and an archive saved from either holds the records and
    expressions of the first archive, for the evaluated forms too.
    """
    import pickle

    problem = _reopened_problem(form)
    # The premise: each of the two other readers changes at least one form, and neither changes "evaluated".
    saved = pickle.dumps((problem.objective, problem.variables, problem.equalities, problem.inequalities), protocol=5)
    plain = pickle.loads(saved)
    with sp.evaluate(False):
        unevaluated = pickle.loads(saved)
    assert (sp.srepr(plain) != _expressions(problem)) == (form in ("objective", "constraints"))
    assert (sp.srepr(unevaluated) != _expressions(problem)) == (form == "sum")
    result = _reopened_run(problem, refinement, tmp_path / "run")
    first = result.save(tmp_path / "saved")
    for number, reopen in enumerate((lambda: load_augmented_lagrangian(first),
                                     lambda: resume_augmented_lagrangian(tmp_path / "run", backend=None))):
        reopened = reopen()
        assert _expressions(reopened.problem) == _expressions(problem)
        assert reopened.problem.content_id == problem.content_id
        assert reopened.record.content_id == result.record.content_id
        assert _contents(reopened.save(tmp_path / f"again-{number}")) == _contents(first)


def test_the_grid_reference_keeps_the_first_feasible_minimum_across_admitted_slabs():
    # constrained_grid_minimum streams C-order slabs sized from max_bytes and replaces its running minimum only
    # on a strictly smaller value. On the interior grid x, y in {-1/2, 0, 1/2}, f = x**2 is least at x = 0 for
    # every y, the flat indices 3, 4 and 5, and a four-point slab puts index 3 in the first slab and 4 and 5
    # in the second. The lexicographically first tie (1, 0) is the reference with any slab size.
    from nwqlib.algorithms.qhd.compiler import H0

    problem = ConstrainedOptimization(objective=x**2, variables=(x, y), bounds=BOX, inequalities=(x - 5,))
    result = solve_augmented_lagrangian(problem, qhd=QHD(num_grid_points=3), options=AugmentedLagrangian(),
                                        execution="classical", seed=11, progress=False)
    rate = 8 * (max(node_count(problem.objective, 10**8), node_count(problem.inequalities[0], 10**8)) + 2) + 8 * 2 + 96
    fixed = 8 * 2 * 3 + H0
    whole = constrained_grid_minimum(result)
    for slab in (1, 2, 4):
        reference = constrained_grid_minimum(result, max_bytes=fixed + slab * rate)
        assert reference.indices == whole.indices == (1, 0) and reference.objective == whole.objective == 0.0
        assert reference.feasible_points == whole.feasible_points == 9
    with pytest.raises(ValueError, match="one-point slab"):
        constrained_grid_minimum(result, max_bytes=fixed + rate - 1)


def test_automatic_selection_expands_each_candidate_objective_once():
    # The cost comparison of automatic selection admits each candidate's symbolic work, and the candidate's
    # planning reuses that admission and its decomposer (constrained._planned_costs), so the current PHR form
    # and the one slack trial each expand and charge their objective once.
    from nwqlib.algorithms.qhd import method as owner

    with patch.object(owner, "expansion_term_charge", side_effect=owner.expansion_term_charge) as bounded:
        _, auto, _ = plan_augmented_lagrangian(_affine_family(8), qhd=BINARY, execution="quantum",
                                               options=AugmentedLagrangian(inequality_form="auto"), seed=7)
    assert len(auto.trials) == 1 and bounded.call_count == 2
