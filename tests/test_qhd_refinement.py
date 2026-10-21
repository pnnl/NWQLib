"""QHD box refinement: centered box rule, finite-grid error, joint mass, search-model invariance and stops."""

from decimal import Decimal, localcontext
from fractions import Fraction
from math import fsum, isfinite, log, sqrt

import mpmath
import numpy as np
import pytest
import sympy as sp

from nwqlib._validation import UNIT_ROUNDOFF
from nwqlib.algorithms.qhd import QHD, BoxRefinement, UniformState, refine_box
from nwqlib.algorithms.qhd import method as owner
from nwqlib.algorithms.qhd import refinement
from nwqlib.execution import ExecutionLimits
from nwqlib.problems import Optimization

x, y = sp.symbols("x y", real=True)


def _independent_interval(masses, eta):
    """Wu et al. arXiv:2605.12066v1, Eq. (13), written apart from the library.

    Grow a set of kept indices from the first maximum. Each step adds the
    neighbor with the larger mass, the lower index on ties. Returns the
    interval, the mass summed in the order of addition, and how many steps
    chose between two present neighbors of unequal mass (the steps where
    growing toward the smaller side would differ).
    """
    peak = int(np.argmax(masses))
    kept, mass, deciding = [peak], masses[peak], 0
    while mass < eta and len(kept) < len(masses):
        candidates = [i for i in (min(kept) - 1, max(kept) + 1) if 0 <= i < len(masses)]
        if len(candidates) == 2 and masses[candidates[0]] != masses[candidates[1]]:
            deciding += 1
        chosen = sorted(candidates, key=lambda i: (-masses[i], i))[0]
        kept.append(chosen)
        mass += masses[chosen]
    return (min(kept), max(kept)), mass, deciding


def test_next_box_follows_the_recorded_marginals_under_the_centered_rule():
    # The oracle recomputes Eq. (13) and the centered cells from each level's
    # recorded QHD marginals and the coordinates of the grid its Plan solved.
    # A search-model level solves at the unit points u_i = grid_value(i), whose
    # original coordinates are a + (b - a) u_i rounded once. The face between
    # two kept cells is the midpoint of adjacent coordinates, rounded once, which
    # on a uniform grid is x_i + h/2. Both sides round the same exact values,
    # so exact equality holds between two correct implementations.
    problem = Optimization(objective=(x - sp.Rational(3, 10)) ** 2 + (y + sp.Rational(1, 5)) ** 2 + x * y / 2,
                           variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 2.0)))
    k, eta = 5, 0.9
    result = refine_box(problem, qhd=QHD(num_grid_points=k, num_steps=20, total_time=6.0),
                        options=BoxRefinement(scaling="search_model", max_levels=4, mass_threshold=eta),
                        execution="classical", seed=3, progress=False)
    assert result.termination == "level_limit" and len(result.levels) == 4
    deciding = inner_faces = 0
    for level, inner in zip(result.levels, result.results, strict=True):
        sides, masses, intervals = [], [], []
        grid = owner._grid(inner.plan)
        points = []
        for j, ((a, b), row) in enumerate(zip(level.box, inner.marginals.array.tolist(), strict=True)):
            (first, last), mass, steps = _independent_interval([m / inner.valid_mass for m in row], eta)
            deciding += steps
            xs = [float(Fraction(a) + (Fraction(b) - Fraction(a)) * Fraction(grid.grid_value(j, i)))
                  for i in range(k)]
            faces = [float((Fraction(p) + Fraction(q)) / 2) for p, q in zip(xs, xs[1:])]
            lower = a if first == 0 else faces[first - 1]
            upper = b if last == k - 1 else faces[last]
            points.append(xs[inner.most_probable_indices[j]])
            inner_faces += (first > 0) + (last < k - 1)
            sides.append((lower, upper))
            masses.append(mass)
            intervals.append((first, last))
        assert level.intervals == tuple(intervals)
        assert level.axis_masses == tuple(masses)
        assert level.next_box == tuple(sides)
        # The default point rule reports the level result's most probable grid
        # point at the original coordinate of that grid point.
        assert level.point_indices == inner.most_probable_indices
        assert level.point_probability == inner.most_probable_probability
        assert level.point == tuple(points)
    # The data must be able to expose the two probed defects: a step that
    # chose between unequal neighbors, and a bound that moved inside a face.
    assert deciding > 0 and inner_faces > 0


def test_best_observed_point_stays_within_the_level_grid_spacing():
    # For a separable convex quadratic the grid minimizer is, per axis, the
    # grid point nearest the minimizer. The interior points of [a, b] lie at
    # a + h, ..., b - h with spacing h, so the nearest one is within h of a
    # minimizer in the box and within h/2 of one at least h/2 from both faces.
    # best_observed with exact readout is that grid minimizer when every grid
    # point has positive probability, which the kept state checks.
    minimizer = (-0.8, 0.3)
    problem = Optimization(objective=(x + sp.Rational(4, 5)) ** 2 + 3 * (y - sp.Rational(3, 10)) ** 2,
                           variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    result = refine_box(problem,
                        qhd=QHD(num_grid_points=3, num_steps=16, total_time=8.0, keep_state=True),
                        options=BoxRefinement(scaling="search_model", point_rule="best_observed",
                                              max_levels=6, mass_threshold=0.9),
                        execution="classical", seed=3, progress=False)
    checked = near_face = 0
    for level, inner in zip(result.levels, result.results, strict=True):
        if not all(a <= m <= b for m, (a, b) in zip(minimizer, level.box, strict=True)):
            break
        assert np.all(np.abs(inner.data.artifact(inner.artifact).array) ** 2 > 0)
        checked += 1
        for coordinate, m, (a, b), h in zip(level.point, minimizer, level.box, level.spacing, strict=True):
            # The unit grid points 1/4, 1/2, 3/4 and spacing 1/4 are exact, so the
            # recorded point a + (b - a) u and h are correctly rounded values of the
            # exact grid point and spacing (at most ulp(M)/2 and ulp(h)/2 off,
            # M = max(|a|, |b|)), the literal m is within ulp(m)/2 of the exact
            # rational minimizer, and the subtraction rounds once more, so the
            # computed error exceeds the exact one by at most this slack.
            slack = np.spacing(max(abs(a), abs(b))) + np.spacing(abs(m)) + np.spacing(h)
            error = abs(coordinate - m)
            if min(m - a, b - m) >= h / 2:
                assert error <= h / 2 + slack
            else:
                near_face += 1
                assert error <= h + slack
        # The union bound holds exactly for the exact masses. The computed
        # joint mass carries at most about 6u (squared amplitudes, their sum,
        # the division by the valid mass), and each axis mass about 9u
        # (marginal sums over K = 3 points, normalization and the running sum),
        # so the bound carries at most about 20u. 64u covers both.
        assert level.joint_mass >= level.joint_mass_bound - 64 * UNIT_ROUNDOFF
    # At level 1 the minimizer x = -0.8 is 0.2 from the face -1, less than
    # h/2 = 0.25, so the h bound is exercised, and the minimizer stays in
    # several boxes.
    assert checked >= 4 and near_face > 0


def test_joint_mass_bound_is_attained_and_is_neither_eta_nor_the_product(monkeypatch):
    # The injected distribution [[0.8, 0.1], [0.1, 0]] has marginal mass 0.9
    # on index 0 of both axes. With eta = 0.85 each interval is {0}, the union
    # bound is 1 - 0.1 - 0.1 = 0.8, and the joint box {0} x {0} holds exactly
    # 0.8, while eta is 0.85 and the product of the axis masses is 0.81.
    state = np.sqrt(np.array([0.8, 0.1, 0.1, 0.0])).astype(complex)
    monkeypatch.setattr(owner, "_evolve_restricted", lambda *args: state)
    problem = Optimization(objective=x + y, variables=(x, y), bounds=((0.0, 1.0), (0.0, 1.0)))
    result = refine_box(problem, qhd=QHD(num_grid_points=2, keep_state=True),
                        options=BoxRefinement(scaling="search_model", point_rule="best_observed",
                                              max_levels=1, mass_threshold=0.85),
                        execution="classical", seed=3, progress=False)
    level = result.levels[0]
    assert level.intervals == ((0, 0), (0, 0))
    # Each probability is a squared rounded square root, within about 4u. A
    # marginal adds one sum (5u), the valid mass is a sum of three (5u), and an
    # axis mass divides the two, within about 11u. The bound 1 - sum(1 - m_j)
    # then carries at most 24u, and the joint mass p/valid about 10u. 32u
    # covers both, far below the 0.01 gap to 0.81 and 0.05 gap to 0.85.
    window = 32 * UNIT_ROUNDOFF
    assert level.axis_masses == pytest.approx((0.9, 0.9), rel=0, abs=window)
    assert level.joint_mass_bound == pytest.approx(0.8, rel=0, abs=window)
    assert level.joint_mass == pytest.approx(0.8, rel=0, abs=window)
    assert level.joint_mass_kind == "exact"
    # From the recorded axis masses the bound rounds only in its sum and final
    # subtraction (each 1 - m_j is exact by Sterbenz's lemma for m_j in [0.5, 2]).
    exact = max(Fraction(0), 1 - sum(1 - Fraction(m) for m in level.axis_masses))
    assert abs(Fraction(level.joint_mass_bound) - exact) <= 4 * Fraction(UNIT_ROUNDOFF)
    # The report gives both stored masses with the event they describe, the product of the kept intervals,
    # and the nonsampled readout, and states no confidence level for exact readout.
    report = result.report(failure_probability=0.05)
    entry = report["levels"][0]
    assert (entry["joint_mass"], entry["joint_mass_bound"], entry["joint_mass_kind"], entry["readout"]) == (
        level.joint_mass, level.joint_mass_bound, "exact", "exact")
    assert entry["interval_event"] == "product of the kept intervals" and entry["split_region_mass"] is None
    assert (f"Level 1 interval mass: {level.joint_mass:.8g} (exact-readout). Marginal union-bound formula: "
            f"{level.joint_mass_bound:.8g}.") in report["summary"].splitlines()
    assert report["confidence"] is None and "simultaneous confidence" not in report["summary"]


@pytest.mark.parametrize("shots,keep_state", [(256, False), (None, False), (None, True)])
def test_native_binary_joint_mass_decodes_the_binary_register(shots, keep_state):
    # The binary encoding places variable j on qubits j b .. j b + b - 1, little-endian (binary module
    # docstring), so a basis index z is the grid point n_j = (z >> j b) mod K, decoded here apart from
    # the library. Every outcome is a grid point. The level's joint box mass must be the mass of the
    # outcomes inside the product of its intervals, for counts, exact probabilities and a kept state.
    # Both sides pool the same binary64 terms (count/256, exact since 256 is a power of two, the single
    # chunk's probabilities, or numpy.abs(state)**2 as decoding.decode_statevector_probabilities forms
    # them) and divide by the same stored valid mass, and math.fsum rounds the exact sum once whatever
    # the order, so the two agree exactly. Both intervals leave out part of their axis, so a one-hot
    # reading (no valid outcome) or swapped variables give another mass.
    k, b = 4, 2
    problem = Optimization(objective=-sp.cos(2 * sp.pi * (x - sp.Rational(3, 10)))
                           - sp.cos(2 * sp.pi * (y + sp.Rational(1, 10))),
                           variables=(x, y), bounds=((0.0, 1.0), (0.0, 1.0)))
    qhd = QHD(num_grid_points=k, num_steps=4, total_time=2.0, boundary="periodic", encoding="binary",
              keep_state=keep_state)
    result = refine_box(problem, qhd=qhd, options=BoxRefinement(max_levels=1, mass_threshold=0.75),
                        execution="quantum", shots=shots, seed=3, progress=False)
    level, inner = result.levels[0], result.results[0]
    assert all(0 < first or last < k - 1 for first, last in level.intervals)
    if keep_state:
        pairs = enumerate(np.abs(inner.data.artifact(inner.artifact).array) ** 2)
    else:
        (chunk,) = inner.data.observations.chunks
        histogram = chunk.histogram()
        pairs = ((index, weight / shots if shots else weight)
                 for index, weight in zip(histogram.index_list(), histogram.weights.tolist()))
    inside = [p for z, p in pairs
              if all(first <= (z >> (j * b)) % k <= last for j, (first, last) in enumerate(level.intervals))]
    assert level.joint_mass == fsum(inside) / inner.valid_mass
    assert level.joint_mass_kind == ("empirical" if shots else "exact")
    if shots:
        observed = list(zip(histogram.index_list(), histogram.weights.tolist()))
        axes = tuple(sum(count for z, count in observed if first <= (z >> (j * b)) % k <= last)
                     for j, (first, last) in enumerate(level.intervals))
        joint = sum(count for z, count in observed if all(
            first <= (z >> (j * b)) % k <= last for j, (first, last) in enumerate(level.intervals)))
        assert (level.region_axis_counts, level.region_count) == (axes, joint)
    else:
        assert level.region_axis_counts is None and level.region_count is None


def test_a_sampled_level_keeps_the_integer_totals_of_its_result(tmp_path, monkeypatch):
    """A counts level copies ``valid_count`` and ``returned_shots`` from its QHD result, and both survive its directory.

    They are the S_z and the raw draw count of the finite-shot statement of
    Proposition 49 (docs/mathematics.md). Depolarizing noise moves one-hot
    outcomes out of the one-excitation sector, so a level's valid count
    falls below its 64 returned shots, and a level that stored one total in
    the other's field would differ. Both totals are recounted from each
    result's observations. The refinement has ended, so
    ``resume_box_refinement`` reads the level records from the directory's
    outer record and the level results from their Runs without planning or
    acquiring, and both must carry the same integers. The report's finite-shot radius uses the
    valid count and stays finite for the smallest allowed failure
    probabilities, and the saved result archive loads without a backend.
    """
    import json
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from nwqlib.algorithms.qhd import resume_box_refinement
    from nwqlib.backends.connection import AerBackend

    model = NoiseModel(basis_gates=["u", "cx"])
    model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["u"])
    model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    backend = AerBackend.from_noise_model(model)
    problem = Optimization(objective=(x - sp.Rational(3, 10)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    options = BoxRefinement(scaling="search_model", point_rule="best_observed", max_levels=2, mass_threshold=0.6)
    directory = tmp_path / "refinement"
    result = refine_box(problem, qhd=QHD(num_grid_points=3, num_steps=2, total_time=0.5), options=options,
                        backend=backend, execution="quantum", shots=64, seed=3, progress=False, directory=directory)
    loaded = resume_box_refinement(directory, backend=backend, progress=False)
    totals = []
    for run in (result, loaded):
        assert len(run.levels) == 2
        for level, inner in zip(run.levels, run.results, strict=True):
            chunks = inner.data.observations.chunks
            valid = sum(count for chunk in chunks
                        for index, count in zip(chunk.histogram().index_list(), chunk.histogram().weights.tolist())
                        if index.bit_count() == 1)
            returned = sum(chunk.returned_shots for chunk in chunks)
            assert (level.valid_count, level.returned_shots) == (inner.valid_count, inner.returned_shots)
            assert (level.valid_count, level.returned_shots) == (valid, returned)
            # A valid one-hot word 1 << i decodes to grid index i. This
            # one-dimensional intersection is also its single axis event.
            first, last = level.intervals[0]
            kept = sum(count for chunk in chunks
                       for index, count in zip(chunk.histogram().index_list(), chunk.histogram().weights.tolist())
                       if index.bit_count() == 1 and first <= index.bit_length() - 1 <= last)
            assert (level.region_axis_counts, level.region_count) == ((kept,), kept)
        totals.append([(level.valid_count, level.returned_shots) for level in run.levels])
    assert totals[0] == totals[1]
    assert all(returned == 64 for _, returned in totals[0]) and any(valid < 64 for valid, _ in totals[0])
    # qhd._coverage.interval_radius uses each level's valid draws, not its 64
    # returned shots. H = 2 and J = 6; alpha/2 goes to each method. The
    # independent expression has a different order, allowing eight roundings.
    alpha = 0.05
    report = result.report()
    assert report == result.report(failure_probability=alpha)
    assert report["summary"] == str(result)
    assert report["confidence"]["horizon"] == 2
    assert report["confidence"]["budget_fractions"] == dict(hoeffding=0.5, clopper_pearson=0.5)
    for entry, level in zip(report["confidence"]["levels"], result.levels, strict=True):
        radius = sqrt(log(4 * 2 * 6 / alpha) / (2 * level.valid_count))
        assert (entry["level"], entry["valid_count"], entry["marginal_events"]) == (level.level, level.valid_count, 6)
        assert (entry["axis_counts"], entry["joint_count"]) == (list(level.region_axis_counts), level.region_count)
        assert entry["radius"] == pytest.approx(radius, rel=8 * UNIT_ROUNDOFF, abs=0)
        assert entry["lower_bound"] == max(entry["hoeffding_lower"], entry["cp_lower"])
        assert f"Level {level.level} selected region mass lower bound:" in report["summary"]
        # Independent binomial right-tail inversion of qhd._coverage.cp_lower.
        # The 1e-9 relative tail residual is the advisor's development check,
        # not a production inverse error bound or directed-rounding claim.
        with localcontext() as context:
            context.prec = 75
            n, m, p = level.valid_count, level.region_count, Decimal.from_float(entry["cp_lower"])
            term = tail = p**n
            # Successively add P(X=n-f) using its exact ratio to P(X=n-f+1).
            for failures in range(1, n - m + 1):
                term *= Decimal(n - failures + 1) / failures * (1 - p) / p
                tail += term
            beta = Decimal.from_float(alpha) / (2 * 2 * 6)
            assert abs(tail / beta - 1) <= Decimal("1e-9")
    assert any(entry["radius"] > sqrt(log(4 * 2 * 6 / alpha) / (2 * 64)) for entry in report["confidence"]["levels"])
    # Tiny alpha preserves the original Hoeffding half-budget when the CP
    # tail is outside its normal binary64 range. The radius oracle uses 50 digits.
    for tiny in (1e-308, 5e-324):
        small = result.report(failure_probability=tiny)
        json.dumps(small, allow_nan=False)
        for entry, level in zip(small["confidence"]["levels"], result.levels, strict=True):
            with mpmath.workdps(50):
                reference = float(mpmath.sqrt(mpmath.log(mpmath.mpf(48) / mpmath.mpf(tiny))
                                              / (2 * level.valid_count)))
            assert isfinite(entry["radius"])
            assert entry["radius"] == pytest.approx(reference, rel=8 * UNIT_ROUNDOFF, abs=0)
            assert entry["cp_lower"] is None and entry["cp_unavailable"] == "beta_tail_below_normal_range"
            assert entry["lower_bound"] == entry["hoeffding_lower"]
    # The counts refinement saves and loads without a backend, with its counts and level Results.
    from nwqlib.algorithms.qhd import load_box_refinement

    loaded = load_box_refinement(result.save(tmp_path / "saved"))
    assert loaded.content_id == result.content_id and loaded.problem.content_id == problem.content_id
    assert [(inner.content_id, inner.valid_count) for inner in loaded.results] == [
        (level.result_id, level.valid_count) for level in result.levels]
    _forbid_record_work(monkeypatch)
    assert loaded.report() == report
    with monkeypatch.context() as patch:
        patch.setattr("scipy.special.betaincinv", lambda *args: pytest.fail("disabled coverage evaluated CP"))
        assert result.report(failure_probability=None)["confidence"] is None


def test_selected_region_coverage_uses_both_bounds_and_preserves_numerical_limits():
    """Independent scalar relations of qhd._coverage, without another QHD trajectory."""
    from nwqlib.algorithms.qhd._coverage import bounds_from_counts, coverage_lines

    # For M=S, the binomial right tail is p**S, so the lower endpoint is
    # beta**(1/S). Evaluate that defining relation with 75-digit powers,
    # independently of the implementation's exp(log_beta/S).
    full_hits = bounds_from_counts(n=2000, axis_counts=(2000, 2000), joint_count=2000,
                                   intervals=((0, 62), (0, 62)), k=64, horizon=60, alpha=0.05)
    with localcontext() as context:
        context.prec = 75
        beta = Decimal.from_float(0.05) / (2 * 60 * (64 * 65 // 2)**2)
        p = Decimal.from_float(full_hits["cp_lower"])
        # The same advisor-specified development residual as the tail sum below.
        assert abs(p**2000 / beta - 1) <= Decimal("1e-9")
    assert full_hits["lower_bound"] == full_hits["cp_lower"] < 1
    assert full_hits["selected_method"] == "clopper_pearson"

    # Three full axes need no marginal error subtraction, so Hoeffding
    # wins here. Its event family still includes all four axes. The CP
    # oracle independently sums the binomial right tail at 75 digits.
    mixed = bounds_from_counts(n=2000, axis_counts=(1600, 2000, 2000, 2000), joint_count=1600,
                               intervals=((0, 0), (0, 31), (0, 31), (0, 31)),
                               k=32, horizon=19, alpha=0.05)
    expected_h = 0.8 - sqrt(log(4 * 19 * 4 * (32 * 33 // 2) / 0.05) / 4000)
    assert mixed["hoeffding_lower"] == pytest.approx(expected_h, rel=8 * UNIT_ROUNDOFF, abs=0)
    assert mixed["lower_bound"] == mixed["hoeffding_lower"] > mixed["cp_lower"]
    assert mixed["selected_method"] == "hoeffding"
    with localcontext() as context:
        context.prec = 75
        p = Decimal.from_float(mixed["cp_lower"])
        term = tail = p**2000
        for failures in range(1, 401):
            term *= Decimal(2001 - failures) / failures * (1 - p) / p
            tail += term
        beta = Decimal.from_float(0.05) / (2 * 19 * (32 * 33 // 2)**4)
        # Advisor's development residual, not an inverse error certificate.
        assert abs(tail / beta - 1) <= Decimal("1e-9")

    kw = dict(intervals=((0, 0),), k=2, horizon=3, alpha=0.05)
    zero = bounds_from_counts(n=10, axis_counts=(0,), joint_count=0, **kw)
    assert zero["lower_bound"] == zero["cp_lower"] == 0
    whole = bounds_from_counts(n=10, axis_counts=(10,), joint_count=10,
                               intervals=((0, 1),), k=2, horizon=3, alpha=0.05)
    assert whole["lower_bound"] == 1 and whole["selected_method"] == "full_support"
    huge = 2**63 - 1
    rounded = bounds_from_counts(n=huge, axis_counts=(huge,), joint_count=huge, **kw)
    # The exact all-hit lower bound is below one. A rounded CP endpoint
    # is unavailable, retaining Hoeffding's original half-budget.
    assert rounded["cp_lower"] is None and rounded["cp_unavailable"] == "beta_lower_rounded_to_one"
    assert rounded["lower_bound"] == rounded["hoeffding_lower"] < 1
    row = dict(round=None, level=1, valid_count=huge, **rounded)
    text = "\n".join(coverage_lines(dict(failure_probability=0.05, horizon=3, levels=[row])))
    assert f"approximately {rounded['lower_bound']!r}" in text
    shapes = bounds_from_counts(n=huge, axis_counts=(huge - 1,), joint_count=huge - 1, **kw)
    assert shapes["cp_lower"] is None and shapes["cp_unavailable"] == "beta_shapes_not_exactly_representable"
    assert shapes["lower_bound"] == shapes["hoeffding_lower"]


def test_mode_or_mean_reports_the_mean_position_in_original_coordinates(monkeypatch):
    # With boundary points, the grid of [-1, 1] is -1, -1/2, 0, 1/2, 1 (u = 0, 1/4, ..., 1
    # for the search model). The injected probabilities (1, 9, 4, 1, 1)/16 are squares of
    # dyadic amplitudes, so every value below is exact. The most probable point is
    # index 1 at x = -1/2, and the conditional mean is x = -4/16 = -1/4 (u = 6/16).
    # For (x + 1/4)**2 the mean's objective 0 is strictly below the mode's 1/16, so
    # the level reports the mean, off the grid. A mean left in u (x = 3/8) or mapped
    # from the wrong face (x = 1 - 2 u = 1/4) would not be smaller.
    state = np.array([0.25, 0.75, 0.5, 0.25, 0.25], dtype=complex)
    monkeypatch.setattr(owner, "_evolve_restricted", lambda *args: state)
    problem = Optimization(objective=(x + sp.Rational(1, 4)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    result = refine_box(problem, qhd=QHD(num_grid_points=5, include_boundary_points=True),
                        options=BoxRefinement(scaling="search_model", point_rule="mode_or_mean", max_levels=1),
                        execution="classical", seed=3, progress=False)
    level = result.levels[0]
    assert result.results[0].most_probable_indices == (1,)
    assert level.point == (-0.25,) and level.objective == 0.0
    assert level.point_indices is None and level.point_probability is None
    # The search model records the unit point it evaluated, u = 6/16 for the mean.
    assert level.unit_point == (0.375,)
    assert level.resources.objective_evaluations == 2
    # The level copies its readout's status whatever it reports.
    assert level.mode_status == result.results[0].mode_status == "resolved"
    # Only mode_or_mean reports a mean position.
    with pytest.raises(ValueError, match="only mode_or_mean reports a mean position"):
        result.revise(options=result.options.revise(point_rule="most_probable"))


@pytest.mark.parametrize("scaling", ["search_model"])
@pytest.mark.parametrize("objective,bounds,points,termination", [
    (sp.Integer(3) + 0 * x, (-1.0, 1.0), 6, "flat_objective"),
    (sp.sin(x) ** 2 + sp.cos(x) ** 2, (-1.0, 1.0), 6, "unresolved_objective"),
])
def test_flat_or_unresolved_objective_stops_before_any_solve(scaling, objective, bounds, points, termination):
    # refinement._resolution_stop. A constant has no support table, so its table ranges sum to zero,
    # E = 0 and the level stops as flat_objective. sin**2 + cos**2, which is 1, stops by a resolution
    # rule, not because it is constant. Its K = 6 table values differ only by the rounding of its
    # evaluation, about 1.1e-16 in all. That range is at most the unit in the last place of the largest
    # table value, where the tables cannot tell the variation from evaluation error, and dividing by E
    # would turn it into a potential of order one, so it stops as unresolved_objective.
    problem = Optimization(objective=objective, variables=(x,), bounds=(bounds,))
    result = refine_box(problem, qhd=QHD(num_grid_points=points, num_steps=2, include_boundary_points=False),
                        options=BoxRefinement(scaling=scaling), execution="classical", seed=3,
                        progress=False)
    assert result.termination == termination
    assert result.levels == () and result.best_level is None and result.candidate is None
    assert result.stopped_resources.evolution_work == 0 and result.resources.shots == 0


@pytest.mark.parametrize("gain", [1.0, 8.0])
def test_search_model_levels_are_invariant_under_affine_maps_of_box_and_objective(gain):
    # x -> s x + t (s = (3, 1/2), t = (5/4, -3/8)) with F'(x) = F((x - t)/s),
    # F + 7/2, F + 2**54 and 4 F change the level objective F(a + D u) only by a constant
    # and a positive factor, which c and E absorb (refinement._level_problem),
    # so every level solves the same normalized tables, for the plain model
    # and with a fixed potential gain alike. The level objective is formed with
    # the exact values of c and kappa/E, so SymPy reduces the five level
    # objectives to one expression. Dyadic data on K = 3 grids keep every box,
    # table value, E and c exact in binary64, so the equality is bit for bit.
    # The levels are compared by F - C with one exact constant C, the constant
    # term of the first level's decomposition (refinement._tabulated_objective),
    # which the added constant enters exactly and the factor 4 scales exactly,
    # so the relative objectives and the best level follow too. Next to 2**54,
    # where binary64 numbers are 4 apart, every reported objective is 2**54,
    # and comparing those would stop the run early without improvement.
    F = ((x - sp.Rational(1, 4)) ** 2 + (x - sp.Rational(1, 4)) * (y + sp.Rational(1, 2)) / 2
         + (y + sp.Rational(1, 2)) ** 2 / 4)
    box = ((0.0, 1.0), (-1.0, 1.0))
    s, t = (3, Fraction(1, 2)), (Fraction(5, 4), Fraction(-3, 8))
    inverse = {x: (x - sp.Rational(t[0])) / s[0], y: (y - sp.Rational(t[1])) / sp.Rational(s[1])}

    def move(side, j):
        return tuple(float(s[j] * Fraction(v) + t[j]) for v in side)

    # name: (problem, factor r and constant c1 of F' relative to F)
    problems = {
        "original": (Optimization(objective=F, variables=(x, y), bounds=box), 1, 0),
        "constant": (Optimization(objective=F + sp.Rational(7, 2), variables=(x, y), bounds=box), 1, 3.5),
        "large constant": (Optimization(objective=F + 2**54, variables=(x, y), bounds=box), 1, 2.0**54),
        "multiple": (Optimization(objective=4 * F, variables=(x, y), bounds=box), 4, 0),
        "affine": (Optimization(objective=F.xreplace(inverse), variables=(x, y),
                                bounds=tuple(move(side, j) for j, side in enumerate(box))), 1, 0),
    }
    runs = {name: refine_box(problem, qhd=QHD(num_grid_points=3, num_steps=8, total_time=4.0),
                             options=BoxRefinement(scaling="search_model", max_levels=4, mass_threshold=0.8,
                                                   potential_gain=gain),
                             execution="classical", seed=3, progress=False)
            for name, (problem, _, _) in problems.items()}
    reference = runs["original"]
    assert len(reference.levels) >= 3
    for inner in reference.results:
        # E and c make the normalized tables' ranges sum to kappa and their minima
        # cancel the normalized constant. Each table entry evaluates at most n
        # monomials c u**p of the solved objective with u in [0, 1], so each
        # term is at most |c|. Its rounded coefficient and at most two products
        # give 3u per term, and the sum adds n - 1 more, so an entry is within
        # (n + 2) u C of its exact value, with C the sum of all |c|. Each
        # relation adds two entries per table plus 2T roundings of sums of at
        # most kappa, hence the window below. A level solved without the division by E
        # would give its E instead, 0.5 or more here.
        tables = inner.plan.reconstruction.support_values
        terms = [t for t in sp.Add.make_args(sp.expand(inner.plan.problem.objective)) if t.free_symbols]
        magnitude = sum(abs(float(t.as_coeff_Mul()[0])) for t in terms)
        window = (2 * len(tables) * (len(terms) + 2) * magnitude + (2 * len(tables) + 2) * gain) * UNIT_ROUNDOFF
        assert sum(float(t.values.array.max()) - float(t.values.array.min()) for t in tables) == pytest.approx(gain, rel=0, abs=window)
        constant = inner.plan.reconstruction.constant
        assert constant + sum(float(t.values.array.min()) for t in tables) == pytest.approx(0, rel=0, abs=window)
    for name, run in runs.items():
        _, factor, shift = problems[name]
        assert run.termination == reference.termination
        assert len(run.levels) == len(reference.levels) and run.best_level == reference.best_level
        for level, base, inner, base_inner in zip(run.levels, reference.levels, run.results,
                                                  reference.results, strict=True):
            assert inner.plan.reconstruction.support_values == base_inner.plan.reconstruction.support_values
            assert inner.plan.reconstruction.constant == base_inner.plan.reconstruction.constant
            assert level.energy_scale == factor * base.energy_scale and level.potential_gain == gain
            assert level.energy_shift == factor * base.energy_shift + shift
            assert level.relative_objective == factor * base.relative_objective
            assert level.intervals == base.intervals and level.axis_masses == base.axis_masses
            if name == "affine":
                assert level.next_box == tuple(move(side, j) for j, side in enumerate(base.next_box))
            else:
                assert level.next_box == base.next_box


def test_stall_split_follows_the_end_mass_lemma_and_the_clearest_valley(monkeypatch):
    # A product state with dyadic amplitudes makes every probability, marginal and region mass exact.
    # The x marginal is (36, 81, 81, 9, 49)/256 and the y marginal (16, 4, 100, 36, 100)/256. With
    # eta = 31/32 all four end cells exceed 1 - eta = 8/256, so no contiguous interval of mass eta can
    # leave out an end (its mass would be at most 1 - p(end) < eta), and both axes are kept whole
    # (refinement._stall_split). The valleys (refinement._clearest_valley) are x at 3, separation
    # min(81, 49)/9 = 5.4, y at 1, 16/4 = 4, and y at 3, 100/36 = 2.8. The clearest is x at 3, while
    # the difference min(L, R) - p(v) would take y at 3 (64 against 40) and the valley mass y at 1.
    # The sides of the x valley are indices 0..2 and 4. Their most probable points are (1, 2), the
    # lexicographically smallest of four ties, and (4, 2), at (1/4, 1) and (1, 1), where -x + y is 3/4
    # and 0. So the refinement continues on the upper side, which holds less probability and not the
    # level's most probable point. The valley cell joins it, the cut is the face toward the lower side,
    # between x = 1/2 and 3/4 at 5/8, and the lower side's 198/256 is discarded.
    a = np.array([6, 9, 9, 3, 7]) / 16
    b = np.array([4, 2, 10, 6, 10]) / 16
    state = np.outer(a, b).reshape(-1).astype(complex)
    monkeypatch.setattr(owner, "_evolve_restricted", lambda *args: state)
    eta = 0.96875
    qhd = QHD(num_grid_points=5, include_boundary_points=True, keep_state=True)
    options = BoxRefinement(scaling="search_model", mass_threshold=eta, max_levels=3, stall_split="best_region")

    def run(objective, bounds=((0.0, 1.0), (-1.0, 3.0)), settings=options):
        return refine_box(Optimization(objective=objective, variables=(x, y), bounds=bounds), qhd=qhd,
                          options=settings, execution="classical", seed=3, progress=False)

    result = run(y - x)
    first, second = result.levels
    inner = result.results[0]
    for row in inner.marginals.array.tolist():
        p = [Fraction(m) / Fraction(inner.valid_mass) for m in row]
        assert p[0] > 1 - Fraction(eta) and p[-1] > 1 - Fraction(eta)
        # Every contiguous interval of mass at least eta is the whole axis.
        assert [(i, j) for i in range(5) for j in range(i, 5) if sum(p[i:j + 1]) >= eta] == [(0, 4)]
    assert first.intervals == ((0, 4), (0, 4))
    split = first.split
    assert (split.axis, split.valley, split.coordinate) == (0, 3, 0.625)
    assert split.regions == (((0.0, 0.625), (-1.0, 3.0)), ((0.625, 1.0), (-1.0, 3.0)))
    assert split.point_indices == ((1, 2), (4, 2)) and split.points == ((0.25, 1.0), (1.0, 1.0))
    assert split.scores == (0.75, 0.0) and split.chosen == 1
    assert split.region_masses == (198 / 256, 58 / 256) and split.discarded_mass == 198 / 256
    assert first.point_indices == (1, 2) and first.next_box == second.box == split.regions[1]
    # The reported point and the two scores.
    assert first.resources.objective_evaluations == 3
    # The same injected state keeps the second box whole too, after the only split of the budget.
    assert second.intervals == ((0, 4), (0, 4)) and second.split is None
    # A stalled level that does not split records why, and a level that splits records nothing.
    assert first.split_declined is None and "stall splits have been made" in second.split_declined
    assert result.termination == "split_limit"
    # With equal scores, y at both points, the side with more probability continues, here the lower
    # one with 198/256, and the valley cell joins it, so the cut lies between 3/4 and 1 at 7/8.
    tie = run(y).levels[0].split
    assert tie.scores == (1.0, 1.0) and tie.chosen == 0 and tie.coordinate == 0.875
    # A stall at the last allowed level stops the run, since no level would solve a region.
    last = run(y - x, settings=options.model_copy(update=dict(max_levels=1)))
    assert last.levels[0].split is None and last.termination == "level_limit"
    assert "last of max_levels" in last.levels[0].split_declined
    assert last.levels[0].resources.objective_evaluations == 1
    # The physical model's last boundary coordinate of [-3, -0.9] is computed as -0.8999999999999999,
    # outside the box (grid.OneHotGrid.grid_value). The split still runs on that coordinate.
    physical = run(y - x, bounds=((-3.0, -0.9), (-1.0, 3.0)),
                   settings=options.model_copy(update=dict(scaling="physical"))).levels[0].split
    assert physical.point_indices == ((1, 2), (4, 2)) and physical.chosen == 1
    assert physical.points[0][0] < physical.coordinate < physical.points[1][0]
    # Table ownership (refinement._tabulated_objective, method.objective_at). Array evaluation of a
    # support expression need not reproduce its scalar evaluation, and the stored tables are the
    # selected data, so the reported grid point and both split scores read the stored entries at their
    # grid indices. Shifting every array output of the compiler's evaluator by 2**-10 makes the stored
    # tables differ from a scalar evaluation of the same expressions at the grid points.
    import nwqlib.algorithms.qhd.compiler as compiler

    def shifted(variables, expression, lambdify=compiler._lambdify_objective):
        evaluator = lambdify(variables, expression)
        return lambda *coordinates: evaluator(*coordinates) + 2.0**-10

    with monkeypatch.context() as patched:
        patched.setattr(compiler, "_lambdify_objective", shifted)
        owned = run(y - x, bounds=((-3.0, -0.9), (-1.0, 3.0)),
                    settings=options.model_copy(update=dict(scaling="physical")))
    level, stored = owned.levels[0], owned.results[0].plan.reconstruction
    assert level.objective == owner.objective_at(stored, level.point_indices, 5)
    assert level.split.scores == tuple(owner.objective_at(stored, i, 5) for i in level.split.point_indices)
    # A stall whose next level could not run is not split. With max_no_improve = 1, the second level,
    # whose point has the first level's objective y = 1, would end the run, so it stops with
    # no_improvement and spends no evaluation on a split. Each level reads its 25 kept amplitudes once,
    # and the first level's split reuses them.
    stop = run(y, settings=options.model_copy(update=dict(max_splits=2, max_no_improve=1)))
    assert [level.split is not None for level in stop.levels] == [True, False]
    assert stop.termination == "no_improvement" and stop.levels[1].resources.objective_evaluations == 1
    assert "no improvement" in stop.levels[1].split_declined
    assert (stop.levels[0].resources.joint_mass_reads, stop.levels[1].resources.joint_mass_reads) == (25, 25)
    # A side's point follows the tie window of the level's most probable point. Raising (2, 2) above
    # its tied neighbors by a tenth of the recorded window leaves it within the window, so both the
    # level's most probable point and the lower side's point stay the lexicographically smallest (1, 2).
    window = inner.most_probable_tie_window
    raised = state.copy()
    raised[2 * 5 + 2] = np.sqrt(abs(state[2 * 5 + 2]) ** 2 + window / 10)
    monkeypatch.setattr(owner, "_evolve_restricted", lambda *args: raised)
    near = run(y - x).levels[0]
    assert near.point_indices == (1, 2) and near.split.point_indices[0] == (1, 2)
    # A dip within the readout's resolution is no valley (refinement._valley_admission). The x marginal
    # (1/5, 1/5, 1/5 - g, 1/5, 1/5 + g) has its only valley at 2, below its smaller peak by g, and the
    # uniform y marginal has none. With g a tenth of the level's window W the level stops as without
    # the option, and with g ten times W it splits. The valid mass is 1 up to rounding, and the
    # summation term rho (peak + valley) is of order 1e-15, far below W here.
    assert 1e-12 < window < 1e-3
    for gap, splits in ((window / 10, False), (10 * window, True)):
        px = np.array([0.2, 0.2, 0.2 - gap, 0.2, 0.2 + gap])
        dip = np.outer(np.sqrt(px), np.sqrt(np.full(5, 0.2))).reshape(-1).astype(complex)
        monkeypatch.setattr(owner, "_evolve_restricted", lambda *args, dip=dip: dip)
        lone = run(y - x, settings=options.model_copy(update=dict(max_levels=2))).levels[0]
        assert lone.intervals == ((0, 4), (0, 4)) and (lone.split is not None) == splits
        assert lone.split_declined == (None if splits else "no valley passes the exact-readout screen")
    # A side's scored point leaves out the valley cell, which stays with the chosen side. In the
    # correlated state with amplitudes (1, 5, 8; 0, 9, 0; 6, 0, 7)/16, x first, the x marginal
    # (90, 81, 85)/256 has its valley at 1, whose column holds the level's most probable point (1, 1)
    # with 81/256. The sides' points are (0, 2) and (2, 2), where x + y is 1 and 2, so the lower side
    # continues with the valley cell, the cut is the face at 3/4, and the upper side's 85/256 is discarded.
    joint = np.array([1, 5, 8, 0, 9, 0, 6, 0, 7], dtype=complex) / 16
    monkeypatch.setattr(owner, "_evolve_restricted", lambda *args: joint)
    valley = refine_box(Optimization(objective=x + y, variables=(x, y), bounds=((0.0, 1.0), (0.0, 1.0))),
                        qhd=QHD(num_grid_points=3, include_boundary_points=True, keep_state=True),
                        options=options.model_copy(update=dict(mass_threshold=0.9)), execution="classical",
                        seed=3, progress=False).levels[0]
    assert valley.point_indices == (1, 1)
    assert (valley.split.valley, valley.split.point_indices, valley.split.chosen) == (1, ((0, 2), (2, 2)), 0)
    assert valley.split.coordinate == 0.75 and valley.split.discarded_mass == 85 / 256
    # A classical level keeps its joint distribution only in a kept state, and the budget belongs to the split.
    with pytest.raises(ValueError, match="keep_state=True"):
        refine_box(Optimization(objective=y - x, variables=(x, y), bounds=((0.0, 1.0), (-1.0, 3.0))),
                   qhd=QHD(num_grid_points=5), options=options, execution="classical", seed=3, progress=False)
    with pytest.raises(ValueError, match="needs max_splits=1"):
        BoxRefinement(scaling="search_model", max_splits=2)
    # One cut does not divide a periodic axis into two regions, so a split budget on a periodic grid
    # is refused before any level, while the same grid without splits refines as usual.
    periodic = QHD(num_grid_points=4, boundary="periodic", keep_state=True)
    monkeypatch.setattr(owner, "_evolve_restricted", lambda *args: np.full(16, 0.25, dtype=complex))
    with pytest.raises(ValueError, match="periodic"):
        refine_box(Optimization(objective=y - x, variables=(x, y), bounds=((0.0, 1.0), (-1.0, 3.0))),
                   qhd=periodic, options=options, execution="classical", seed=3, progress=False)
    plain = refine_box(Optimization(objective=y - x, variables=(x, y), bounds=((0.0, 1.0), (-1.0, 3.0))),
                       qhd=periodic, options=options.model_copy(update=dict(stall_split="none")),
                       execution="classical", seed=3, progress=False)
    assert plain.levels and all(level.split is None for level in plain.levels)


def test_stall_split_continues_the_double_well_past_its_stall():
    # The double well of refinement._stall_split, whose default run stops with box_unchanged because
    # both end cells of each axis keep more than 1 - eta. The split run repeats the default run up to
    # that level, splits it, and solves the chosen region next. Its discarded probability is the
    # conditional mass of the other region. The example uses gain 1 and the uniform start, which the
    # test names because the defaults are gain 8 and the kinetic ground state.
    well = (2 * x**2 - 1) ** 2 + 3 * x / 5 + 2 * (y - sp.Rational(3, 10)) ** 2 + 6 * x * y / 5
    problem = Optimization(objective=well, variables=(x, y), bounds=((-1.2, 1.2), (-1.2, 1.2)))
    qhd = QHD(num_grid_points=6, num_steps=80, total_time=10.0, keep_state=True, initial_state=UniformState())
    settings = dict(scaling="search_model", potential_gain=1.0, max_levels=10, max_no_improve=10)
    default = refine_box(problem, qhd=qhd, options=BoxRefinement(**settings), execution="classical", seed=7,
                         progress=False)
    run = refine_box(problem, qhd=qhd, options=BoxRefinement(**settings, stall_split="best_region"),
                     execution="classical", seed=7, progress=False)
    stall = len(default.levels)
    assert default.termination == "box_unchanged" and len(run.levels) > stall
    for base, level in zip(default.levels, run.levels):
        assert (level.box, level.intervals, level.point) == (base.box, base.intervals, base.point)
    assert [level.level for level in run.levels if level.split is not None] == [stall]
    level, inner = run.levels[stall - 1], run.results[stall - 1]
    split = level.split
    other = 1 - split.chosen
    assert run.levels[stall].box == split.regions[split.chosen]
    # Grid indices of each region on the split axis, from the level grid's own coordinates.
    grid = owner._grid(inner.plan)
    a, b = level.box[split.axis]
    coordinates = [float(Fraction(a) + (Fraction(b) - Fraction(a)) * Fraction(grid.grid_value(split.axis, i)))
                   for i in range(6)]
    regions = ([i for i, c in enumerate(coordinates) if c < split.coordinate],
               [i for i, c in enumerate(coordinates) if c > split.coordinate])
    # The marginal reconstruction repeats the stored mass computation and must agree exactly. The
    # independent reference sums the stored amplitudes' squared real and imaginary parts as exact
    # rationals and divides by the same stored valid mass before one final rounding. The probability
    # evaluation model of _validation.ABSOLUTE_SQUARE_ROUNDOFF contributes five rounding units, the
    # six-term marginal contributes five, and division and fsum contribute two. Against the
    # reference's one rounding, (gamma_12 + gamma_1)/(1 - gamma_12) < 14u. Even allowing one extra
    # fsum rounding unit gives a bound below 15u, so 16u times discarded_mass covers the normal-range
    # readout comparison without an evolution-error allowance.
    row = [m / inner.valid_mass for m in inner.marginals.array[split.axis].tolist()]
    assert split.discarded_mass == fsum(row[i] for i in regions[other])
    state = inner.data.artifact(inner.artifact).array
    indices = list(np.ndindex(6, 6))
    squares = sum((Fraction(float(z.real)) ** 2 + Fraction(float(z.imag)) ** 2
                   for z, i in zip(state, indices) if i[split.axis] in regions[other]), Fraction(0))
    reference = float(squares / Fraction(inner.valid_mass))
    assert abs(split.discarded_mass - reference) <= 16 * UNIT_ROUNDOFF * split.discarded_mass
    # The discarded region is the other strict side of the valley. Each side's scored point is the
    # lexicographically smallest positive-weight point within the level's tie window of the side's
    # largest weight, the kernel's weights being numpy.abs(state)**2.
    sides = (range(split.valley), range(split.valley + 1, 6))
    assert regions[other] == list(sides[other])
    weights = dict(zip(indices, map(float, np.abs(state) ** 2)))
    for r, side in enumerate(sides):
        inside = {i: w for i, w in weights.items() if i[split.axis] in side and w > 0}
        top = max(inside.values())
        tied = [i for i, w in inside.items() if top - w <= inner.most_probable_tie_window]
        assert split.point_indices[r] == min(tied)
    assert split.scores[split.chosen] < split.scores[other]
    # Every level before the split is the default run's, so the best objective cannot be larger.
    assert run.objective <= default.objective


def test_stall_split_needs_a_valley_that_the_shots_resolve():
    # With counts, a valley splits only when the smaller flanking peak's count exceeds the valley's by
    # more than sqrt(2 N log(2 H d K/alpha)), N valid shots, H max_levels, d = 2 variables, K = 4 grid
    # points and the per-refinement allowance alpha of refinement._FALSE_VALLEY_ALLOWANCE
    # (refinement._valley_admission, which derives the threshold). The double well at K = 4 (8 qubits)
    # stalls at level 1 with a y valley at index 2 holding about 3% of the shots against peaks near 30%
    # and 40%, so the count gap grows as N while the threshold grows as sqrt(N). The refinement seed
    # fixes the planning streams, and the backend stream of each level's Plan fixes the simulator seed.
    # The stall and the valley masses above were measured with gain 1 and the uniform start, which the
    # test names because the defaults are gain 8 and the kinetic ground state.
    well = (2 * x**2 - 1) ** 2 + 3 * x / 5 + 2 * (y - sp.Rational(3, 10)) ** 2 + 6 * x * y / 5
    problem = Optimization(objective=well, variables=(x, y), bounds=((-1.2, 1.2), (-1.2, 1.2)))
    options = BoxRefinement(scaling="search_model", potential_gain=1.0, max_levels=2, stall_split="best_region")
    qhd = QHD(num_grid_points=4, num_steps=20, total_time=10.0, initial_state=UniformState())
    for shots, resolved in ((40, False), (1000, True)):
        result = refine_box(problem, qhd=qhd, options=options,
                            execution="quantum", shots=shots, seed=3, progress=False)
        level, inner = result.levels[0], result.results[0]
        assert level.intervals == ((0, 3), (0, 3))
        # Integer marginal counts of the valid outcomes, decoded independently: qubit j*K + i is
        # bit j*K + i of the outcome index, and a valid outcome has one 1 per variable.
        counts, valid = [[0] * 4, [0] * 4], 0
        for chunk in inner.data.observations.chunks:
            histogram = chunk.histogram()
            for index, count in zip(histogram.index_list(), histogram.weights.tolist()):
                ones = [q for q in range(histogram.width) if index >> q & 1]
                if sorted(q // 4 for q in ones) == [0, 1]:
                    valid += count
                    for q in ones:
                        counts[q // 4][q % 4] += count
        c = counts[1]
        gap = min(max(c[:2]), max(c[3:])) - c[2]
        bound = 2 * valid * np.log(2 * 2 * 2 * 4 / float(refinement._FALSE_VALLEY_ALLOWANCE))
        # The valley exists in the counts at both shot numbers and lies far from the threshold.
        assert gap > 0 and (gap * gap > 2 * bound if resolved else gap * gap < bound / 2)
        if resolved:
            assert (level.split.axis, level.split.valley) == (1, 2)
            # The actual retained side includes the valley. The pre-split
            # interval box is whole, so reusing its count S would be wrong.
            split = level.split
            selected = c[:split.valley + 1] if split.chosen == 0 else c[split.valley:]
            kept = sum(selected)
            assert 0 < kept < valid
            assert (level.region_axis_counts, level.region_count) == ((valid, kept), kept)
        else:
            assert level.split is None and result.termination == "box_unchanged"
            assert (level.region_axis_counts, level.region_count) == ((valid, valid), valid)


def test_best_point_gaussian_starts_a_later_level_at_the_best_point():
    # With level_initial_state="best_point_gaussian" the first level starts from QHD.initial_state and
    # each later level from exp(-(x - p)**2/(2 (w L)**2)) per axis, p the location of the best level so
    # far (least relative objective, earlier on ties) and L the side of the level's box, written in the
    # level problem's coordinates (refinement._best_point_gaussian). For the search model these are
    # u = (x - a)/L, so the center is (p - a)/L, with p = a' + L' u' the exact image of the best level's
    # unit point, rounded once, and the width w. The second level does not improve on the first, so the third
    # level's Gaussian must sit at the first level's point, not at the last level's. The oracle builds
    # each Gaussian at 50 digits on the level's grid coordinates and compares it with the start vector
    # that the level's Plan stored and evolved. The difference of the normalized vectors is at most the
    # construction bound of GaussianState.variable_errors (in units of u) plus the 50-digit evaluation
    # error, far below 1e-40.
    problem = Optimization(objective=(x - sp.Rational(3, 10)) ** 2 + (y + sp.Rational(1, 5)) ** 2 + x * y / 2,
                           variables=(x, y), bounds=((0.0, 1.0), (-1.0, 2.0)))
    qhd = QHD(num_grid_points=5, num_steps=10, total_time=4.0)
    width = 0.25
    result = refine_box(problem, qhd=qhd,
                        options=BoxRefinement(scaling="search_model", potential_gain=1.0, max_levels=3, max_no_improve=3,
                                              mass_threshold=0.6, level_initial_state="best_point_gaussian",
                                              level_gaussian_width=width),
                        execution="classical", seed=3, progress=False)
    levels = result.levels
    assert len(levels) == 3 and levels[1].relative_objective >= levels[0].relative_objective
    assert levels[0].initial_state is None and result.results[0].plan.method.initial_state == qhd.initial_state
    for level, inner in zip(levels[1:], result.results[1:], strict=True):
        best = min(levels[:level.level - 1], key=lambda item: (item.relative_objective, item.level))
        assert inner.plan.method.initial_state == level.initial_state
        center, widths = [], []
        for j, (a, b) in enumerate(level.box):
            side = Fraction(b) - Fraction(a)
            (a0, b0), u = best.box[j], best.unit_point[j]
            center.append(float((Fraction(a0) + (Fraction(b0) - Fraction(a0)) * Fraction(u) - Fraction(a)) / side))
            widths.append(width)
        assert level.initial_state.center == tuple(center) and level.initial_state.widths == tuple(widths)
        grid = owner._grid(inner.plan)
        bounds = level.initial_state.variable_errors(grid)
        for j, stored in enumerate(inner.plan.reconstruction.initial_amplitudes):
            with mpmath.workdps(50):
                c, w = mpmath.mpf(center[j]), mpmath.mpf(widths[j])
                exact = [mpmath.exp(-(mpmath.mpf(grid.grid_value(j, i)) - c) ** 2 / (2 * w**2)) for i in range(5)]
                norm = mpmath.sqrt(mpmath.fsum(t**2 for t in exact))
                distance = mpmath.sqrt(mpmath.fsum((mpmath.mpf(v) - t / norm) ** 2 for v, t in zip(stored.array.tolist(), exact)))
                assert distance <= bounds[j] * UNIT_ROUNDOFF + mpmath.mpf(10) ** -40


def test_budget_and_inner_failure_stop_after_the_completed_levels():
    problem = Optimization(objective=(x - sp.Rational(3, 10)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    options = BoxRefinement(scaling="search_model", potential_gain=1.0, point_rule="best_observed", max_levels=5,
                            mass_threshold=0.6)
    # Two levels of 64 shots spend a 128-shot budget, so the third level
    # cannot start. Counts make the joint mass empirical. With gain 1 each of
    # these sampled levels keeps part of its three-point axis. With the
    # default gain 8 the first level keeps the whole axis and the run stops with
    # box_unchanged before the budget runs out.
    shots = 64
    budget = refine_box(problem, qhd=QHD(num_grid_points=3, num_steps=4, total_time=3.0),
                        options=options, execution="quantum", shots=shots, seed=3, progress=False,
                        limits=ExecutionLimits(max_total_shots=2 * shots))
    assert budget.termination == "budget_exhausted" and len(budget.levels) == 2
    assert budget.resources.shots == 2 * shots and budget.stopped_resources is None
    for level, inner in zip(budget.levels, budget.results, strict=True):
        # Independent one-hot decoding of the counts. Qubit q is bit q of the
        # outcome index, and an outcome is valid with exactly one 1. Counts
        # over 64 shots are exact in binary64, so both sides round only the
        # final quotient and agree exactly.
        inside = valid = 0
        for chunk in inner.data.observations.chunks:
            histogram = chunk.histogram()
            for index, count in zip(histogram.index_list(), histogram.weights.tolist()):
                ones = [q for q in range(histogram.width) if index >> q & 1]
                if len(ones) == 1:
                    valid += count
                    first, last = level.intervals[0]
                    inside += count if first <= ones[0] <= last else 0
        assert level.joint_mass_kind == "empirical" and level.joint_mass == inside / valid
    # The physical model's kinetic 1/h**2 grows as the box shrinks, and with it
    # the admitted classical evolution work. A max_work equal to the first
    # level's kernel work admits that level and refuses the smaller second box.
    settings = dict(num_grid_points=4, num_steps=10, total_time=4.0)
    physical = dict(scaling="physical", point_rule="best_observed", mass_threshold=0.8)
    first = refine_box(problem, qhd=QHD(**settings), options=BoxRefinement(**physical, max_levels=1),
                       execution="classical", seed=3, progress=False)
    work = first.results[0].plan.construction.kernels[0].invocation_work
    failed = refine_box(problem, qhd=QHD(**settings, max_work=work),
                        options=BoxRefinement(**physical, max_levels=3),
                        execution="classical", seed=3, progress=False)
    assert failed.termination == "inner_failed" and "max_work" in failed.failure
    assert len(failed.levels) == 1 and failed.levels[0].point == first.levels[0].point
    assert failed.levels[0].next_box == first.levels[0].next_box
    # In the first level nothing has completed, so QHD's own error propagates
    # (_outer.inner_failure): a max_work below the first level's kernel work,
    # and a configuration that QHD rejects.
    with pytest.raises(ValueError, match="max_work="):
        refine_box(problem, qhd=QHD(**settings, max_work=work - 1), options=BoxRefinement(**physical),
                   execution="classical", seed=3, progress=False)
    with pytest.raises(ValueError, match="keep_state requires exact amplitude acquisition"):
        refine_box(problem, qhd=QHD(num_grid_points=3, keep_state=True), options=options,
                   execution="quantum", shots=shots, seed=3, progress=False)


def _forbid_record_work(monkeypatch):
    """Make planning, objective evaluation, evolution, acquisition, loading, array reads and resource estimates raise."""
    from nwqlib.artifacts import ArtifactHandle

    def forbid(*args, **kwargs):
        raise AssertionError("a record-only summary planned, evaluated, acquired, loaded or estimated")

    monkeypatch.setattr(QHD, "plan", forbid)
    for name in ("_level_problem", "_tabulated_objective", "_level_decomposition", "saved_result"):
        monkeypatch.setattr(refinement, name, forbid)
    for name in ("objective_at", "_evolve_restricted", "_summarize"):
        monkeypatch.setattr(owner, name, forbid)
    for name in ("prepare", "submit", "load_result", "load_run"):
        monkeypatch.setattr(f"nwqlib.scientist.{name}", forbid)
    monkeypatch.setattr("nwqlib.algorithms.qhd.resources.run_resources", forbid)
    monkeypatch.setattr(ArtifactHandle, "array", property(forbid))
    monkeypatch.setattr("nwqlib.execution.ObservationChunk.histogram", forbid)


def test_an_early_stop_and_a_zero_level_record_report_stored_counts_and_unavailable_resources(monkeypatch):
    """The finite-shot radius uses the configured horizon after an early stop, and unknown counts stay unknown.

    A counts refinement with ``max_levels=4`` stops with box_unchanged after one level, so the horizon of
    Proposition 49 (docs/mathematics.md) stays H = 4, not the one completed level: the radius is
    ``sqrt(log(4 H J/alpha)/(2 S))`` with J = d K (K + 1)/2 = 6 and the level's S valid draws,
    because qhd._coverage allocates half of alpha to each method. The default report includes coverage;
    explicit None suppresses it. A zero-level record that stopped
    with budget_exhausted and a stopped level whose CX and shot counts are unknown reports both as
    unavailable with their stored reasons, never as zero, and has no selected point.
    """
    from nwqlib.algorithms.qhd import BoxRefinementResult, RefinementResources
    from nwqlib.algorithms.qhd.refinement_records import total_resources

    problem = Optimization(objective=(x - sp.Rational(3, 10)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
    options = BoxRefinement(max_levels=4, mass_threshold=0.99)
    early = refine_box(problem, qhd=QHD(num_grid_points=3, num_steps=2, total_time=0.5), options=options,
                       execution="quantum", shots=64, seed=3, progress=False)
    (level,) = early.levels
    assert early.termination == "box_unchanged" and level.valid_count >= 1
    stopped = RefinementResources(cx=None, shots=None, unavailable=(("cx", "its preparation raised"),
                                                                    ("shots", "its journal could not be read")))
    empty = BoxRefinementResult(problem_id=problem.content_id, qhd=QHD(num_grid_points=3), options=options,
                                execution="quantum", shots=64, seed_entropy=3, levels=(), best_level=None,
                                termination="budget_exhausted", failure="max_total_circuits",
                                stopped_resources=stopped, resources=total_resources((), stopped),
                                max_plannings=8)
    _forbid_record_work(monkeypatch)
    alpha = 0.1
    report = early.report(failure_probability=alpha)
    radius = sqrt(log(4 * 4 * 6 / alpha) / (2 * level.valid_count))
    (entry,) = report["confidence"]["levels"]
    assert report["confidence"]["horizon"] == 4 and entry["radius"] == pytest.approx(radius, rel=8 * UNIT_ROUNDOFF, abs=0)
    assert "horizon 4, failure probability 0.1" in report["summary"]
    assert entry["lower_bound"] == 1.0 and entry["selected_method"] == "full_support"
    lines = str(early).splitlines()
    assert f"Level 1 interval mass: {level.joint_mass:.8g} (empirical). Marginal union-bound formula: " \
           f"{level.joint_mass_bound:.8g}." in lines
    assert f"Raw-shot reservations: {early.resources.shots}. Returned shots at completed levels: 64." in lines
    assert early.report()["confidence"]["failure_probability"] == 0.05
    assert any("simultaneous confidence 95%" in line for line in lines)
    omitted = early.report(failure_probability=None)
    assert omitted["confidence"] is None and "simultaneous confidence" not in omitted["summary"]
    # A counts level keeps 0 < S <= R with valid_mass S/R and an empirical joint mass, and every level of a
    # refinement with shots is a counts level.
    for update, message in ((dict(valid_count=None), "counts level records"),
                            (dict(returned_shots=level.valid_count - 1), "counts level records"),
                            (dict(valid_mass=level.valid_mass / 2), "counts level records"),
                            (dict(joint_mass_kind="exact"), "empirical exactly for counts"),
                            (dict(region_axis_counts=(level.valid_count + 1,)), "region_axis_counts"),
                            (dict(region_count=level.valid_count - 1), "region_count")):
        with pytest.raises(ValueError, match=message):
            level.revise(**update)
    with pytest.raises(ValueError, match="with shots records counts"):
        early.revise(shots=None)
    for invalid in (0, 1, float("nan")):
        with pytest.raises(ValueError, match="failure_probability"):
            early.report(failure_probability=invalid)
    resources = report["resources"]["total"]
    assert resources["shots"]["meaning"].startswith("raw-shot reservations")
    assert resources["cx"]["meaning"].startswith("upper bound") and resources["circuit_attempts"]["meaning"].startswith(
        "exact count")
    lines = str(empty).splitlines()
    for line in ("Box refinement stopped: budget_exhausted. Completed levels: 0 of at most 4.",
                 "The remaining cumulative limits could not fund a level.",
                 "Stop detail: max_total_circuits.",
                 "Selected point: unavailable (no completed level).",
                 "CX upper bound across prepared circuit bodies: unavailable (level 1: its preparation raised).",
                 "Raw-shot reservations: unavailable (level 1: its journal could not be read). Returned shots at "
                 "completed levels: unavailable (no completed level)."):
        assert line in lines
    counts = empty.report(failure_probability=alpha)["resources"]
    assert counts["total"]["cx"]["value"] is None and counts["stopped"]["shots"]["value"] is None
    assert counts["total"]["cx"]["unavailable"] == "level 1: its preparation raised"
    assert counts["t_cost"]["value"] is None and counts["max_plannings"]["value"] == 8
    assert empty.report()["confidence"] is None


def test_an_unevaluated_objective_reopens_as_supplied_from_a_refinement_archive_and_directory(tmp_path):
    """A standalone refinement's archive and ended durable directory reopen the objective as it was supplied.

    f = x + (x + 11/16)/(x + 11/16) keeps its quotient, 0/0 at the pole x = -11/16 and 1 elsewhere, which
    SymPy's evaluation would turn into x + 1. ``archive._SymbolicReader`` rebuilds each pickled node that
    evaluation changes with evaluation disabled, so ``resume_box_refinement`` attaches the supplied
    objective and ``load_box_refinement`` accepts it with the problem's original ``srepr``.
    """
    from nwqlib.algorithms.qhd import load_box_refinement, resume_box_refinement

    quotient = sp.Mul(x + sp.Rational(11, 16), sp.Pow(x + sp.Rational(11, 16), -1, evaluate=False), evaluate=False)
    problem = Optimization(objective=sp.Add(x, quotient, evaluate=False), variables=(x,), bounds=((-1.0, 1.0),))
    assert problem.objective.doit() == x + 1 != problem.objective
    result = refine_box(problem, qhd=QHD(num_grid_points=5, include_boundary_points=True),
                        options=BoxRefinement(max_levels=1), execution="classical", seed=1,
                        directory=tmp_path / "run", progress=False)
    path = result.save(tmp_path / "saved")
    for reopen in (lambda: resume_box_refinement(tmp_path / "run", backend=None),
                   lambda: load_box_refinement(path)):
        reopened = reopen()
        assert sp.srepr(reopened.problem.objective) == sp.srepr(problem.objective)
        assert reopened.problem.content_id == problem.content_id and reopened.content_id == result.content_id
