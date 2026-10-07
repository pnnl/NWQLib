"""QHD most-probable readout, its tie windows and the conditional position moments."""

from dataclasses import replace
from math import sqrt
from unittest.mock import Mock

import numpy as np
import pytest
import sympy as sp

from nwqlib._validation import (
    AER_STATEVECTOR_ROUNDOFF_PER_OPERATION,
    UNIT_ROUNDOFF as u,
    probability_difference_window,
)
from nwqlib.algorithms.qhd import QHD, QuadraticSchedule, UniformState
from nwqlib.algorithms.qhd import method as owner
from nwqlib.execution import ObservationChunk, ObservationView
from nwqlib.problems import Optimization
from nwqlib.scientist import load_result, plan, solve
from test_qhd_workflow import _independent_state

x = sp.Symbol("x", real=True)
LINE = Optimization(objective=x, variables=(x,), bounds=((-1.0, 1.0),))
# gamma = 0 makes the Hamiltonian time independent.
STATIC = QuadraticSchedule(gamma=0.0)


def _supplied(data, *chunk_values, reverse=False):
    """Replace the values of the run's own chunks and relink its trace, keeping every association.

    ``chunk_values`` gives one ``{bits: value}`` mapping per chunk, in chunk
    order. Integer values become counts, floats exact probabilities.
    ``reverse`` lists the changed chunks in the opposite order.
    """
    chunks = data.observations.chunks
    changed = []
    for chunk, values in zip(chunks, chunk_values, strict=True):
        # The same fields with the supplied outcomes, as a revision of the run's chunk.
        fields = {name: getattr(chunk, name) for name in type(chunk).model_fields if name != "values"}
        fields["parent_id"] = chunk.content_id
        if all(isinstance(v, int) for v in values.values()):
            fields["returned_shots"] = sum(values.values())
        changed.append(ObservationChunk.from_histogram(values, **fields))
    ids = {old.content_id: new.content_id for old, new in zip(chunks, changed)}
    trace = data.trace.revise(events=tuple(
        event.revise(observation_id=ids.get(event.observation_id, event.observation_id))
        for event in data.trace.events))
    return replace(data, observations=ObservationView(chunks=tuple(changed[::-1] if reverse else changed)),
                   trace=trace)


def _analyze(selected, data):
    return selected.method.analyze(selected, data, settings={})


def _native_window(receipt, evaluation):
    """Tie window of one exact native chunk, written out from its receipt.

    delta = (G*C_A/2 + 5)*u for G native Aer instructions (``_validation.native_state_error``),
    and the window is ``2*delta + delta**2 + e*(1 + delta)**2`` with the probability
    evaluation e (``_validation.probability_difference_window``).
    """
    delta = (receipt.native_operations * AER_STATEVECTOR_ROUNDOFF_PER_OPERATION / 2 + 5) * u
    return 2 * delta + delta * delta + evaluation * (1 + delta) ** 2


def test_most_probable_point_matches_restricted_eigendecomposition():
    """With gamma = 0 each Schrodinger step is exact, so the kernel's state is exp(-i T H) psi0.

    H is the 3-by-3 restricted Hamiltonian ``-Delta_h/2 + diag(f)`` on the
    interior Dirichlet grid of [-1, 1] (h = 1/2, points -1/2, 0, 1/2) with
    f = x, and psi0 is uniform. At T = 0.4 the oracle probabilities are about
    (0.248, 0.599, 0.153). The candidate is index 0 (least f), the most
    probable point index 1, and the top two differ by 0.35, far beyond the
    window. The kernel's probability may differ from the ordered exact
    exponentials of ``dt (a Khat + b diag(V*))`` by its per-probability bound
    ``2 delta + delta**2 + 5u (1 + delta)**2``, the tie window derived by
    ``method._host_tie_window`` from the table-defined budget
    ``method._schrodinger_state_error``, which includes generator formation.
    Two further errors lie outside that bound. The binary64 step parameters
    and tables that this reference takes as exact differ from the oracle's
    by a few u at most, which moves the state by a few u over T = 0.4, and ``numpy.linalg.eigh`` is backward
    stable, ``||E|| <= p(K) u ||H||`` (LAPACK Users' Guide, 3rd ed.,
    Sec. 4.7), so with ||H|| < 7 and two orthogonal products the oracle adds
    a few tens of u. 1e-14 (about 90u) covers both. K = 2 cannot serve,
    because a two-level system started uniform always has its larger
    probability at the lower objective. The Method names the uniform start,
    since its default is the kinetic ground state.
    """
    selected = plan(LINE, method=QHD(num_grid_points=3, num_steps=2, total_time=0.4, schedule=STATIC,
                                     initial_state=UniformState()),
                    execution="classical", seed=7)
    h, grid = 0.5, np.array([-0.5, 0.0, 0.5])
    hamiltonian = (np.diag(np.full(3, 1 / h**2)) + np.diag(np.full(2, -1 / (2 * h * h)), 1)
                   + np.diag(np.full(2, -1 / (2 * h * h)), -1) + np.diag(grid))
    energies, vectors = np.linalg.eigh(hamiltonian)
    state = vectors @ (np.exp(-0.4j * energies) * (vectors.T @ np.full(3, 1 / sqrt(3))))
    expected = np.abs(state) ** 2
    result = solve(selected)
    # The uniform start vector 1/sqrt(3) errs by 2u (initial_state.restricted_state_error).
    r = selected.reconstruction
    delta = owner._schrodinger_state_error(selected.method, owner._grid(selected), r.support_values, r.step_weights,
                                           start=2.0)
    window = probability_difference_window(delta, 5 * u)
    tolerance = window + 1e-14
    top, second = np.sort(expected)[::-1][:2]
    assert top - second > window + 2 * tolerance
    assert result.candidate_indices == (0,)
    assert result.most_probable_indices == (int(np.argmax(expected)),) == (1,)
    assert result.most_probable_probability == pytest.approx(expected[1], rel=0, abs=tolerance)
    assert result.most_probable_coordinates == (0.0,) and result.most_probable_objective == 0.0
    assert result.most_probable_deficit == 0.0
    assert result.most_probable_tie_window == window and result.most_probable_tie_window_unavailable is None
    assert result._summary_lines()[0].startswith("Computed probability maximizer:")
    # The gap exceeds the window, so the kernel's published maximizer is the representative and no
    # other point passes the tie test.
    assert result.probability_maximizer_indices == (1,) and result.mode_status == "resolved"
    assert result.probability_maximizer_probability == result.most_probable_probability
    assert "Mode status: resolved among positive observed valid points." in str(result)


@pytest.mark.parametrize("execution,keep_state", [("quantum", False), ("quantum", True), ("classical", False)])
def test_most_probable_point_matches_independent_product_formula(execution, keep_state):
    """The smallest native case, K = 3 with one first-order step, against a product formula written in the test.

    ``_independent_state`` applies the potential phase, each XX+YY link and
    the stencil-diagonal phase as dense 3-by-3 exponentials. Native exact
    probabilities and stored amplitudes go through ``_summarize``, the
    classical ``ir_product`` kernel through ``_host_summary``. The product
    gives about (0.400, 0.567, 0.033), so the most probable point (1) differs
    from the candidate (0). Each path's window, from
    ``_validation.native_state_error`` with the receipt's instruction count
    (2u component squares for Aer probabilities, 5u for ``np.abs(z)**2``) or
    from ``method._host_tie_window`` of the direct one-hot budget
    (``theory.onehot_product_state_error``), also bounds one probability. The
    dense oracle makes about ten small exponentials and phase products, each
    within a few u, which 1e-14 covers. The probabilities above and the
    classical window's 2u start error are those of the uniform start, which
    the Method names, since its default is the kinetic ground state.
    """
    method = QHD(num_grid_points=3, num_steps=1, total_time=0.4, schedule=STATIC, trotter_order=1,
                 theory_flavor="ir_product", keep_state=keep_state, initial_state=UniformState())
    result = solve(LINE, method=method, execution=execution, seed=7)
    expected = np.abs(_independent_state(LINE, method)) ** 2
    if execution == "classical":
        from nwqlib.algorithms.qhd.theory import onehot_product_state_error

        delta = onehot_product_state_error(result.plan.reconstruction.steps, start=2.0)
        window = probability_difference_window(delta, 5 * u)
    else:
        (receipt,) = result.data.receipts
        window = _native_window(receipt, (5 if keep_state else 2) * u)
    tolerance = window + 1e-14
    assert expected[1] - max(expected[0], expected[2]) > window + 2 * tolerance
    assert result.candidate_indices == (0,) and result.most_probable_indices == (1,)
    assert result.most_probable_probability == pytest.approx(expected[1], rel=0, abs=tolerance)
    assert result.most_probable_coordinates == (0.0,) and result.most_probable_objective == 0.0
    assert result.most_probable_deficit == 0.0
    assert result.most_probable_tie_window == pytest.approx(window, rel=1e-14, abs=0)
    assert result.most_probable_tie_window_unavailable is None


def test_tie_window_is_relative_to_the_computed_maximum():
    """Supplied exact probabilities place gaps below and above the derived native window.

    With w from the receipt (``_native_window``, 2u evaluation, one chunk),
    points within w of the computed maximum M are tied and the
    lexicographically smallest wins; the recorded deficit is the computed
    ``M - p``. Case (a) puts index 0 at M - w/2 below index 1 at M, so index 0
    wins with a positive deficit, which strict equality or a zero window
    would miss. Case (b) moves the gap to 2w, so index 1 wins. Case (c) has
    adjacent gaps of 0.6w but an end-to-end gap of 1.2w. Comparing with M
    ties only indices 1 and 2, while a pairwise chain would reach index 0.
    Bin order does not matter. A receipt outside the roundoff derivation (the
    SDK state preparation lowers to an excluded ``unitary``) records no
    window, with its reason, and compares exactly. The
    summary shows the mode status, unresolved when a point other than the
    representative passes the tie test, and the stored reason when no window
    was derived. The computed probability maximizer is the point of largest
    supplied probability in every case, index 1 in (a) and (b) and index 2 in
    (c), whatever the representative. Without a window it is the
    representative, and the status is unavailable although the raw maximum
    is unique.
    """
    source = solve(LINE, method=QHD(num_grid_points=3, num_steps=1, total_time=0.4, schedule=STATIC, trotter_order=1),
                   seed=7)
    (receipt,) = source.data.receipts
    w = _native_window(receipt, 2 * u)
    assert w > 1e-12
    cases = {
        "a": ({"001": 0.3 - w / 2, "010": 0.3, "100": 0.2, "000": 0.2 + w / 2}, (0,)),
        "b": ({"001": 0.3 - 2 * w, "010": 0.3, "100": 0.2, "000": 0.2 + 2 * w}, (1,)),
        "c": ({"001": 0.3 - 1.2 * w, "010": 0.3 - 0.6 * w, "100": 0.3, "000": 0.1 + 1.8 * w}, (1,)),
    }
    maximizers = {"a": ((1,), "unresolved"), "b": ((1,), "resolved"), "c": ((2,), "unresolved")}
    for name, (values, selected) in cases.items():
        for ordered in (values, dict(reversed(values.items()))):
            data = _supplied(source.data, ordered)
            result = _analyze(source.plan, data)
            peak = max(p for bits, p in values.items() if bits != "000")
            index = selected[0]
            chosen = values[format(1 << index, "03b")]
            top, status = maximizers[name]
            assert result.probability_maximizer_indices == top and result.mode_status == status, name
            assert result.probability_maximizer_probability == peak
            assert result.probability_maximizer_coordinates == (-0.5 + 0.5 * top[0],)
            assert result.probability_maximizer_objective == result.probability_maximizer_coordinates[0]
            assert result.most_probable_indices == selected, name
            assert result.most_probable_probability == chosen
            assert result.most_probable_deficit == peak - chosen <= w
            assert result.most_probable_tie_window == pytest.approx(w, rel=1e-14, abs=0)
            assert all(peak - values[format(1 << i, "03b")] > w for i in range(index))
        assert (result.most_probable_deficit > 0) == (name != "b")
        # Cases (a) and (c) admit a second point to the tie test, case (b) none.
        assert ("Mode status: unresolved." in str(result)) == (name != "b")

    excluded = solve(LINE, method=QHD(num_grid_points=3, num_steps=1, total_time=0.4, schedule=STATIC, trotter_order=1,
                                      initial_state_preparation="qiskit_state_preparation"), seed=7)
    width = excluded.plan.reconstruction.width
    values = {format(1 << i, f"0{width}b"): p for i, p in enumerate((0.3 - w / 2, 0.3, 0.2))}
    values["0" * width] = 0.2 + w / 2
    result = _analyze(excluded.plan, _supplied(excluded.data, values))
    assert result.most_probable_indices == (1,) and result.most_probable_deficit == 0.0
    assert result.most_probable_tie_window is None
    assert result.probability_maximizer_indices == (1,) and result.probability_maximizer_probability == 0.3
    assert result.most_probable_tie_window_unavailable == "outside the roundoff derivation: unitary"
    assert "Mode status: unavailable (outside the roundoff derivation: unitary)." in str(result)
    assert result.mode_status == "unavailable"


def test_mode_status_counts_the_points_that_pass_the_tie_test_not_the_deficit():
    """The status needs exactly one positive observed valid point in the maximum-to-point tie test.

    Supplied exact probabilities on the three-point line, with the receipt's window w far below every
    gap that should separate points (``QHDAnalysis``, "Mode status"). ``(1/8, 3/4, 1/8)`` has one
    point within w of the maximum, so the maximizer is the representative and the mode is resolved.
    ``(0.3, 0.3 - w/2, 0.2)`` with ``0.2 + w/2`` invalid has a unique raw maximum at index 0, which is
    also the lexicographically first point of the test, so the deficit is zero, but index 1 passes the
    test too, so the mode is unresolved: the deficit alone cannot tell the two cases apart. An
    all-invalid population has neither point and an unavailable status with the missing-population
    reason. The population is the positive observed points: a singleton population is resolved, also
    when the observation holds half the probability, whose unlisted half is not a known zero.
    """
    source = solve(LINE, method=QHD(num_grid_points=3, num_steps=1, total_time=0.4, schedule=STATIC, trotter_order=1),
                   seed=7)
    (receipt,) = source.data.receipts
    w = _native_window(receipt, 2 * u)
    assert 1e-12 < w < 1e-6

    def analyzed(values):
        return _analyze(source.plan, _supplied(source.data, values))

    separated = analyzed({"001": 0.125, "010": 0.75, "100": 0.125})
    assert separated.most_probable_indices == separated.probability_maximizer_indices == (1,)
    assert (separated.most_probable_deficit, separated.mode_status) == (0.0, "resolved")
    tied = analyzed({"001": 0.3, "010": 0.3 - w / 2, "100": 0.2, "000": 0.2 + w / 2})
    assert tied.most_probable_indices == tied.probability_maximizer_indices == (0,)
    assert (tied.most_probable_deficit, tied.mode_status) == (0.0, "unresolved")
    assert "Mode status: unresolved. Another positive observed valid point passes the tie test." in str(tied)
    empty = analyzed({"000": 1.0})
    assert empty.valid_mass == 0 and empty.mode_status == "unavailable"
    assert (empty.probability_maximizer_indices, empty.probability_maximizer_coordinates,
            empty.probability_maximizer_probability, empty.probability_maximizer_objective) == (None,) * 4
    assert "Computed probability maximizer and tie representative: unavailable (no positive valid outcome)." in str(
        empty)
    assert "Mode status: unavailable (no valid observed candidate)." in str(empty)
    for values, missing in (({"010": 1.0}, ()), ({"010": 0.5}, ("incomplete probability mass",))):
        single = analyzed(values)
        assert single.probability_maximizer_indices == single.most_probable_indices == (1,)
        assert single.mode_status == "resolved" and single.missing == missing


@pytest.mark.parametrize("rho,epsilon,representative,maximizer", [
    (10.0, -1e-11, (2,), (5,)),
    (100.0, 1e-10, (1,), (6,)),
])
def test_unresolved_double_well_modes_keep_their_representative(rho, epsilon, representative, maximizer):
    """Two nearly mirror-symmetric double wells whose top two grid probabilities lie within the tie window.

    ``rho (x**2 - 1/2)**2 + epsilon x`` on [-1, 1] with K = 8, four Schrodinger steps, T = 0.5 and a kept
    state. The two wells put mirror grid points within the host window of each other, so two points
    pass the tie test, recomputed here from the kept state: the lexicographically smaller one stays the
    representative with its positive deficit, the larger is the computed maximizer, and the mode is
    unresolved. Without the kept state the kernel publishes the same maximizer and status as scalars.
    """
    problem = Optimization(objective=rho * (x * x - sp.Rational(1, 2)) ** 2 + epsilon * x, variables=(x,),
                           bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=8, num_steps=4, total_time=0.5, theory_flavor="schrodinger", keep_state=True)
    result = solve(plan(problem, method=method, execution="classical", seed=11), progress=False)
    probabilities = np.abs(result.data.artifact(result.artifact).array) ** 2
    top = int(np.argmax(probabilities))
    window = result.most_probable_tie_window
    tied = [i for i, p in enumerate(probabilities) if p > 0 and probabilities[top] - p <= window]
    assert len(tied) == 2 and (min(tied),) == representative and (top,) == maximizer
    assert result.most_probable_indices == representative and result.probability_maximizer_indices == maximizer
    assert result.mode_status == "unresolved" and 0 < result.most_probable_deficit <= window
    assert result.most_probable_deficit == result.probability_maximizer_probability - result.most_probable_probability
    assert "Mode status: unresolved." in str(result)
    scalars = solve(plan(problem, method=method.revise(keep_state=False), execution="classical", seed=11),
                    progress=False)
    assert (scalars.most_probable_indices, scalars.probability_maximizer_indices, scalars.mode_status) == (
        representative, maximizer, "unresolved")


def test_exact_chunks_of_one_run_are_averaged():
    """Two exact chunks of one Run, collected through the public ``nwqlib.execution`` exports, are averaged.

    A static QHD Plan has one experiment, and ``Run``, ``prepare_experiment``
    and ``submit_experiment`` can still collect it twice in one Run. The first
    supplied chunk favors index 0 and the second index 1, with dyadic
    probabilities, so halving and adding them is exact and the analysis must
    report the exact mean of the two chunks per point, and the most probable
    point of that mean, index 1. Each pooled probability passes roundings of
    its own, so the tie window must exceed the mean of the two chunks'
    single-chunk windows (``method._readout_window``).
    """
    from nwqlib.execution import Run, prepare_experiment, submit_experiment

    selected = plan(LINE, method=QHD(num_grid_points=3, num_steps=1, total_time=0.4, schedule=STATIC, trotter_order=1),
                    seed=7)
    with Run(selected, progress=False) as run:
        for _ in range(2):
            run.collect(submit_experiment(prepare_experiment(selected.resolve("qhd"), run=run), run=run))
        data = run.data
    first = {"001": 0.5, "010": 0.25, "100": 0.125, "000": 0.125}
    second = {"001": 0.125, "010": 0.625, "100": 0.125, "000": 0.125}
    result = _analyze(selected, _supplied(data, first, second))
    mean = {bits: (first[bits] + second[bits]) / 2 for bits in first}
    assert result.marginals.array.tolist() == [[mean["001"], mean["010"], mean["100"]]]
    assert result.most_probable_indices == (1,) and result.most_probable_probability == mean["010"]
    assert (result.invalid_mass, result.observed_mass) == (mean["000"], 1.0)
    receipts = {receipt.content_id: receipt for receipt in data.receipts}
    single = sum(_native_window(receipts[c.prepared_id], 2 * u) for c in data.observations.chunks) / 2
    assert len(data.observations.chunks) == 2 and result.most_probable_tie_window > single


@pytest.mark.parametrize("flavor,steps,total_time", [("schrodinger", 2, 0.7), ("split_step", 3, 0.5)])
def test_symmetric_problem_ties_to_the_smallest_index(flavor, steps, total_time):
    """``x**2 + y**2 + x y`` on the square [-1, 1]² at K = 4 is swap symmetric, so (1, 2) and (2, 1) tie.

    The objective, the two identical interior grids, the kinetic operator and
    the kinetic ground state are unchanged when x and y are exchanged, so the
    model gives the two points equal probabilities. Their computed
    probabilities agree within the host tie window and exceed every other
    point by more than it, and the selection is the smaller index (1, 2). In
    these two cases, on macOS arm64 with NumPy 2.5.2 and SciPy 1.18.1, the
    computed maximum lies at (2, 1), so exact comparison would select (2, 1).
    """
    y = sp.Symbol("y", real=True)
    problem = Optimization(objective=x**2 + y**2 + x * y, variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    method = QHD(num_grid_points=4, num_steps=steps, total_time=total_time, theory_flavor=flavor, keep_state=True)
    result = solve(problem, method=method, execution="classical", seed=7)
    probabilities = (np.abs(result.data.artifact(result.artifact).array) ** 2).reshape(4, 4)
    window = result.most_probable_tie_window
    pair = (probabilities[1, 2], probabilities[2, 1])
    others = max(p for (i, j), p in np.ndenumerate(probabilities) if (i, j) not in ((1, 2), (2, 1)))
    assert abs(pair[0] - pair[1]) <= window and min(pair) > others + window
    assert result.most_probable_indices == (1, 2)
    assert result.most_probable_deficit == max(pair) - pair[0] <= window


@pytest.mark.parametrize("swap", [False, True])
def test_count_ties_compare_pooled_integers(swap):
    """Ten shots in two chunks of one Run: index 0 at counts (3, 0), index 1 at (1, 2), four invalid outcomes.

    The chunks are collected through the public ``nwqlib.execution`` exports.
    Pooled integers tie at 3 against 3, so the lexicographically smaller
    index 0 is selected, with probability 3/10, window 0 and deficit 0.
    Pooling floats instead would give 3/10 + 0/10 = 0.3 against
    1/10 + 2/10 = 0.30000000000000004 and pick index 1. The Result does not
    depend on chunk order. The objective -x puts the candidate at index 1.
    """
    from nwqlib.execution import Run, prepare_experiment, submit_experiment

    problem = Optimization(objective=-x, variables=(x,), bounds=((-1.0, 1.0),))
    selected = plan(problem, method=QHD(num_grid_points=3), shots=5, seed=7)
    with Run(selected, progress=False) as run:
        for _ in range(2):
            run.collect(submit_experiment(prepare_experiment(selected.resolve("qhd"), run=run), run=run))
        data = run.data
    supplied = _supplied(data, {"001": 3, "010": 1, "000": 1}, {"010": 2, "011": 3}, reverse=swap)
    result = _analyze(selected, supplied)._attach(selected, supplied)
    assert result.most_probable_indices == (0,) and result.most_probable_coordinates == (-0.5,)
    assert result.most_probable_probability == 0.3 and result.most_probable_objective == 0.5
    assert result.most_probable_deficit == 0.0 and result.most_probable_tie_window == 0.0
    assert result.candidate_indices == (1,)
    assert result.marginals.array.tolist() == [[0.3, 0.3, 0.0]]
    assert (result.valid_mass, result.invalid_mass, result.observed_mass) == (0.6, 0.4, 1.0)
    assert result._summary_lines()[0].startswith("Candidate (")
    # The pooled integers tie, so both points pass the W = 0 test: the maximizer is the same
    # lexicographically first point and the empirical mode is unresolved.
    assert result.probability_maximizer_indices == (0,) and result.probability_maximizer_probability == 0.3
    assert result.mode_status == "unresolved"
    assert "Empirical mode status: unresolved. Counts give no proof of the population mode." in str(result)


def test_counts_compare_integers_whose_frequencies_display_equal():
    """Counts 2**54 - 1 and 2**54 of R = 2**55 - 1 returned shots both display the frequency 0.5.

    Pooled integers decide the maximizer and the status before division by R (``method._summarize``),
    so index 1 with the larger count is the unique maximizer and the resolved representative,
    although its frequency and index 0's round to the same binary64 value, 0.5, where a comparison of
    those values would tie and take index 0. A Run cannot return that many shots of a five-shot Plan,
    so the pooled counts go to the summary that ``QHD.analyze`` calls for counts, with the window 0
    that it passes.
    """
    selected = plan(LINE, method=QHD(num_grid_points=3), shots=5, seed=7)
    n = 2**54
    assert (n - 1) / (2 * n - 1) == n / (2 * n - 1) == 0.5
    summary = owner._summarize(selected, np.array([n - 1, n], dtype=np.int64), 0.0, 1.0, 0.0, shots=2 * n - 1,
                               points=np.array([[0], [1]]))
    assert summary["marginals"].tolist() == [[0.5, 0.5, 0.0]]
    assert summary["probability_maximizer_indices"] == summary["most_probable_indices"] == (1,)
    assert summary["mode_status"] == "resolved" and summary["most_probable_deficit"] == 0.0
    assert summary["probability_maximizer_probability"] == summary["most_probable_probability"] == 0.5


def test_counts_record_the_integer_valid_and_returned_totals(tmp_path):
    """Counts analysis records how many outcomes decoded to a grid point and how many shots returned.

    Two chunks of one Run hold five shots each. Seven outcomes have one
    excitation in the three-qubit register and decode to a grid point, and
    three do not. A total over every outcome (10) or over one chunk differs
    from ``valid_count`` (7), and a total of one chunk (5) differs from
    ``returned_shots`` (10). Both are recounted here from the Result's own
    observations. ``valid_mass`` is their quotient rounded once, which
    Python's integer division also rounds correctly, and both integers
    survive save and load. Classical, exact native and kept-state readout
    have no shots, so both fields are None.
    """
    from nwqlib.execution import Run, prepare_experiment, submit_experiment

    selected = plan(LINE, method=QHD(num_grid_points=3), shots=5, seed=7)
    with Run(selected, progress=False) as run:
        for _ in range(2):
            run.collect(submit_experiment(prepare_experiment(selected.resolve("qhd"), run=run), run=run))
        data = run.data
    supplied = _supplied(data, {"001": 3, "100": 1, "000": 1}, {"010": 3, "011": 2})
    result = _analyze(selected, supplied)._attach(selected, supplied)
    chunks = result.data.observations.chunks
    valid = sum(count for chunk in chunks
                for index, count in zip(chunk.histogram().index_list(), chunk.histogram().weights.tolist())
                if index.bit_count() == 1)
    returned = sum(chunk.returned_shots for chunk in chunks)
    assert (result.valid_count, result.returned_shots) == (valid, returned) == (7, 10)
    assert result.valid_count <= result.returned_shots
    assert result.valid_mass == result.valid_count / result.returned_shots
    restored = load_result(result.save(tmp_path / "counts"))
    assert (restored.valid_count, restored.returned_shots) == (7, 10)
    for execution, keep_state in (("classical", False), ("quantum", False), ("quantum", True)):
        exact = solve(LINE, method=QHD(num_grid_points=3, keep_state=keep_state), execution=execution, seed=7)
        assert exact.valid_count is None and exact.returned_shots is None


def test_maximizer_and_mode_status_survive_save_and_load_and_contradictions_are_refused(tmp_path, monkeypatch):
    """Native exact probabilities and counts, host scalars and a host kept state keep their stored readout.

    The exact case is the supplied gap of half a window below the maximum, whose representative (0,)
    differs from the maximizer (1,). The counts tie at 3 against 3 shares one point between them. The
    host cases are the first unresolved double well (representative (2,), maximizer (5,)) with and
    without a kept state, the second carrying the status as a kernel scalar. Loading runs with analysis,
    decoding, the readout summary and evolution poisoned, so every field comes back from the saved
    record. A record whose new fields contradict each other is refused when it is built, before any
    execution.
    """
    from nwqlib.execution import Run, prepare_experiment, submit_experiment

    source = solve(LINE, method=QHD(num_grid_points=3, num_steps=1, total_time=0.4, schedule=STATIC, trotter_order=1),
                   seed=7)
    (receipt,) = source.data.receipts
    w = _native_window(receipt, 2 * u)
    supplied = _supplied(source.data, {"001": 0.3 - w / 2, "010": 0.3, "100": 0.2, "000": 0.2 + w / 2})
    exact = _analyze(source.plan, supplied)._attach(source.plan, supplied)
    sampled = plan(LINE, method=QHD(num_grid_points=3), shots=10, seed=7)
    with Run(sampled, progress=False) as run:
        run.collect(submit_experiment(prepare_experiment(sampled.resolve("qhd"), run=run), run=run))
        data = _supplied(run.data, {"001": 3, "010": 3, "000": 4})
    counts = _analyze(sampled, data)._attach(sampled, data)
    well = Optimization(objective=10.0 * (x * x - sp.Rational(1, 2)) ** 2 - 1e-11 * x, variables=(x,),
                        bounds=((-1.0, 1.0),))
    host = [solve(plan(well, method=QHD(num_grid_points=8, num_steps=4, total_time=0.5, keep_state=keep),
                       execution="classical", seed=11), progress=False) for keep in (False, True)]
    results = [exact, counts, *host]
    expected = [((0,), (1,), "unresolved"), ((0,), (0,), "unresolved"), ((2,), (5,), "unresolved"),
                ((2,), (5,), "unresolved")]
    assert [(r.most_probable_indices, r.probability_maximizer_indices, r.mode_status) for r in results] == expected
    paths = [result.save(tmp_path / str(i)) for i, result in enumerate(results)]
    forbid = Mock(side_effect=AssertionError("load analyzed, decoded, summarized or evolved"))
    for name in ("_evolve_restricted", "_summarize"):
        monkeypatch.setattr(owner, name, forbid)
    monkeypatch.setattr(QHD, "analyze", forbid)
    monkeypatch.setattr("nwqlib.algorithms.qhd.decoding.decode_statevector_probabilities", forbid)
    fields = ("most_probable_indices", "most_probable_deficit", "probability_maximizer_indices",
              "probability_maximizer_coordinates", "probability_maximizer_probability",
              "probability_maximizer_objective", "mode_status")
    for path, result in zip(paths, results, strict=True):
        restored = load_result(path)
        assert restored.content_id == result.content_id
        assert [getattr(restored, name) for name in fields] == [getattr(result, name) for name in fields]
    kept = host[1]
    for update, message in (
        (dict(mode_status="resolved"), "requires an unresolved status"),
        (dict(mode_status="unavailable"), "unavailable exactly without"),
        (dict(probability_maximizer_indices=None), "joint availability"),
        (dict(probability_maximizer_probability=kept.most_probable_probability / 2), "largest observed weight"),
    ):
        with pytest.raises(ValueError, match=message):
            kept.revise(**update)
    # For counts the representative is the maximizer whatever the status, since W = 0.
    with pytest.raises(ValueError, match="requires an unresolved status"):
        counts.revise(probability_maximizer_indices=(1,), probability_maximizer_coordinates=(0.0,),
                      probability_maximizer_objective=0.0)
    # Each further relation is broken alone: a maximizer outside the grid, a deficit that is not the
    # maximizer's probability minus the representative's, and a maximizer objective below the candidate's,
    # which minimizes over the same population.
    for update, message in (
        (dict(probability_maximizer_indices=(8,)), "must belong to the observed valid population"),
        (dict(most_probable_deficit=kept.most_probable_deficit / 2), "largest observed weight"),
        (dict(probability_maximizer_objective=kept.value - 1.0), "largest observed weight"),
    ):
        with pytest.raises(ValueError, match=message):
            kept.revise(**update)
    # The Plan's grid and support tables fix the maximizer's coordinates and objective.
    for update, message in (
        (dict(probability_maximizer_coordinates=(0.0,)), "probability-maximizer coordinates differ"),
        (dict(probability_maximizer_objective=kept.probability_maximizer_objective + 1.0),
         "probability-maximizer objective differs"),
    ):
        with pytest.raises(ValueError, match=message):
            kept.revise(**update).validate_plan(kept.plan)
    # The kernel's status scalar reads back only as 0 or 1.
    (chunk,) = host[0].data.observations.chunks
    values = tuple(v.revise(value=0.5) if v.label == "mode_unresolved" else v for v in chunk.values)
    with pytest.raises(ValueError, match="mode_unresolved must be 0 or 1"):
        owner._host_summary(host[0].plan, values)


def test_position_moments_come_from_saved_marginals(tmp_path, monkeypatch):
    """Conditional means and standard deviations from marginals and grid only, after save and load.

    x has interior grid points 10 and 14 on (6, 18), y has -7 and -5 on
    (-9, -3). Supplied joint probabilities 0.05, 0.05, 0.2, 0.1 with 0.6
    invalid give the unconditional marginals (0.1, 0.3) and (0.25, 0.15) and
    valid mass 0.4. The moments of Liu et al. arXiv:2607.16996v1 Eq. (94),
    conditioned on a valid outcome as ``QHDAnalysis.position_mean`` states, give
    <x> = (10*0.1 + 14*0.3)/0.4 = 13, sigma_x = sqrt((9*0.1 + 1*0.3)/0.4) =
    sqrt(3), <y> = -6.25 and sigma_y = sqrt((0.5625*0.25 + 1.5625*0.15)/0.4)
    = sqrt(0.9375). Each mass carries its decimal rounding and one or two
    additions, and the means and variances are ratios of same-sign sums, so
    the relative error stays below about 20u; rel=1e-14 leaves margin. An
    all-invalid population has no moments.
    """
    y = sp.Symbol("y", real=True)
    problem = Optimization(objective=x + y, variables=(x, y), bounds=((6.0, 18.0), (-9.0, -3.0)))
    source = solve(problem, method=QHD(num_grid_points=2), seed=7)

    def bits(i, j):
        return format((1 << i) | (1 << (2 + j)), "04b")

    populations = (
        {bits(0, 0): 0.05, bits(0, 1): 0.05, bits(1, 0): 0.2, bits(1, 1): 0.1, "0000": 0.6},
        {"0000": 1.0},
    )
    restored = []
    for index, values in enumerate(populations):
        data = _supplied(source.data, values)
        result = _analyze(source.plan, data)
        path = result._attach(source.plan, data).save(tmp_path / str(index))
        restored.append(load_result(path))
    forbid = Mock(side_effect=AssertionError("moments evolved, decoded, evaluated or acquired"))
    for name in ("_evolve_restricted", "objective_at"):
        monkeypatch.setattr(owner, name, forbid)
    monkeypatch.setattr(QHD, "analyze", forbid)
    monkeypatch.setattr("nwqlib.algorithms.qhd.decoding.decode_statevector_probabilities", forbid)
    monkeypatch.setattr("nwqlib._prepared_execution.submit_experiment", forbid)
    moments, empty = restored
    assert moments.valid_mass == pytest.approx(0.4, rel=1e-15, abs=0)
    assert moments.position_mean == pytest.approx((13.0, -6.25), rel=1e-14, abs=0)
    assert moments.position_standard_deviation == pytest.approx((sqrt(3), sqrt(0.9375)), rel=1e-14, abs=0)
    assert empty.valid_mass == 0 and empty.position_mean is None and empty.position_standard_deviation is None


@pytest.mark.parametrize("bits", [None, 1])
def test_decoded_invalid_mass_is_summed_directly(bits):
    """The decoder sums the invalid population itself, not ``total - fsum(valid)``.

    A valid weight 1 and an invalid weight ``2**-54`` give the correctly rounded
    total 1, so the subtraction would report no invalid mass, while the direct
    sum is ``2**-54`` (``decoding.decode_statevector_probabilities``). The binary register has no invalid
    outcome, so its invalid mass is zero and its total the direct sum.
    """
    from math import fsum

    from nwqlib.algorithms.qhd.decoding import decode_statevector_probabilities
    from nwqlib.algorithms.qhd.grid import OneHotGrid

    grid = OneHotGrid(("x",), ((-1.0, 1.0),), 2, False, "dirichlet")
    state = np.zeros(4, dtype=complex)
    if bits is None:
        # Index 1 (qubit 0 set) is the valid point 0 and index 3 (both qubits set) is invalid.
        state[1], state[3] = 1.0, 2.0**-27
        decoded = decode_statevector_probabilities(state, grid)
        assert decoded.invalid_probability == 2.0**-54
        assert decoded.total_probability == 1.0
        assert [(p.grid_indices, p.probability) for p in decoded.points] == [((0,), 1.0)]
    else:
        state = np.array([1.0, 2.0**-27], dtype=complex)
        decoded = decode_statevector_probabilities(state, grid, bits)
        assert decoded.invalid_probability == 0.0
        assert decoded.total_probability == fsum([1.0, 2.0**-54]) == 1.0
        assert [p.grid_indices for p in decoded.points] == [(0,), (1,)]


@pytest.mark.parametrize("fields, execution", [
    (dict(num_grid_points=80, theory_flavor="split_step"), "classical"),
    (dict(num_grid_points=16, encoding="binary", boundary="periodic"), "quantum"),
    (dict(num_grid_points=8), "quantum"),
])
def test_a_refused_readout_grid_is_streamed_with_the_same_probabilities(monkeypatch, fields, execution):
    """A kept state whose dense grid the level's limits refuse is streamed in admitted chunks, never formed as a grid.

    The dense route is kept whenever its complete byte law fits, 48D + 24K + 65536 bytes for native amplitudes
    and 32D + 24K + 65536 for classical ones. Otherwise the stream reserves 65536 + 1024d + (16d + 128)c bytes
    for the largest admitted chunk c, and a limit below both refuses before any grid or chunk exists
    (``refinement._kept_readout_choice``). Each streamed chunk squares the moduli of the amplitudes gathered at
    the one-hot or binary register indices of its grid points, entry by entry as the dense grid does, so the box
    masses, a correctly rounded ``fsum`` of the same terms, and the side modes equal the kept grid's. A joint
    request reads the kept state once and both sides of a split share two passes.
    """
    from types import SimpleNamespace

    from nwqlib.algorithms.qhd import refinement

    x, y = sp.symbols("x y")
    problem = Optimization(objective=(2 * x**2 - 1) ** 2 + 3 * x / 5 + 2 * (y - sp.Rational(3, 10)) ** 2
                           + 6 * x * y / 5, variables=(x, y), bounds=((-1.2, 1.2), (-1.2, 1.2)))
    result = solve(problem, method=QHD(num_steps=20, total_time=2.0, keep_state=True, **fields),
                   execution=execution, seed=3)
    k = fields["num_grid_points"]
    dense = (32 if execution == "classical" else 48) * k**2 + 24 * k + 65536
    fixed, rate = 65536 + 1024 * 2, 16 * 2 + 128
    kept = refinement._LevelReadout(result, SimpleNamespace(max_bytes=dense, max_work=10**12), 0)
    assert kept.grid is not None
    with pytest.raises(ValueError, match="max_bytes"):
        refinement._LevelReadout(result, SimpleNamespace(max_bytes=min(dense, fixed + rate) - 1, max_work=10**12), 0)
    monkeypatch.setattr(refinement._LevelReadout, "_population", Mock(side_effect=AssertionError("formed")))
    streamed = refinement._LevelReadout(result, SimpleNamespace(max_bytes=dense - 1, max_work=10**12), 0)
    assert streamed.grid is None
    assert streamed._chunk_size == min(k**2, 4096, (dense - 1 - fixed) // rate) < k**2
    for intervals in (((0, k - 1), (0, k - 1)), ((1, k - 2), (0, k // 2)), ((k - 1, k - 1), (2, 2))):
        assert streamed.joint(intervals) == kept.joint(intervals)
    window = result.most_probable_tie_window or 0.0
    for axis, first, last in ((0, 0, k // 2 - 1), (1, k // 2, k - 1), (0, 0, k - 1)):
        assert streamed.mode(axis, first, last, window) == kept.mode(axis, first, last, window)
    before = streamed.reads
    sides = ((0, k // 2 - 1), (k // 2 + 1, k - 1))
    assert streamed.modes(0, sides, window) == kept.modes(0, sides, window)
    assert streamed.reads - before == 2 * len(streamed.state)


@pytest.mark.parametrize("window", [None, 0.0, 0.02, 1.0])
def test_the_array_summary_matches_a_scalar_reference_with_ties_and_a_leading_zero_cell(monkeypatch, window):
    """``method._summarize`` on a d = 2, K = 8 probability grid equals a scalar reference written here.

    The reference walks the positive points in lexicographic order: the candidate is the first point of least
    objective, the probability maximizer the first point of the largest weight M, and the mode the first positive
    point with ``M - w <= window``, unresolved when a second point passes, unavailable without a window
    (docs/mathematics.md, Proposition 45). The objective and the weights tie at several points. A zero-weight cell
    precedes the positive cells, and the window 1.0 exceeds M, where a tie predicate without the positive mask
    would select that zero cell.
    """
    from math import fsum

    x, y = sp.symbols("x y")
    problem = Optimization(objective=x**2 + y**2, variables=(x, y), bounds=((-1.0, 1.0), (-1.0, 1.0)))
    selected = plan(problem, method=QHD(num_grid_points=8), seed=7)

    def objective(reconstruction, indices, k):
        return float((indices[0] - 3) ** 2 % 5 + (indices[1] % 2))

    monkeypatch.setattr(owner, "objective_at", objective)
    w = np.full((8, 8), 0.01)
    w[0, 0] = 0.0
    w[2, 5] = w[4, 1] = w[6, 6] = 0.05
    w[3, 3] = 0.04
    w[1, 7] = 0.0
    summary = owner._summarize(selected, w, 0.0, None, window, "no window" if window is None else None)
    items = [(index, float(w[index])) for index in np.ndindex(w.shape) if w[index] > 0]
    values = {index: objective(None, index, 8) for index, _ in items}
    peak = max(p for _, p in items)
    tied = [index for index, p in items if peak - p <= (0.0 if window is None else window)]
    assert summary["candidate_indices"] == min(items, key=lambda item: (values[item[0]], item[0]))[0]
    assert summary["probability_maximizer_indices"] == min(index for index, p in items if p == peak)
    assert summary["most_probable_indices"] == tied[0]
    assert summary["mode_status"] == ("unavailable" if window is None
                                      else "unresolved" if len(tied) > 1 else "resolved")
    assert summary["valid_mass"] == summary["observed_mass"] == fsum(w.flat)
    assert summary["expected_objective"] == fsum(p * values[index] for index, p in items) / fsum(w.flat)
    assert summary["marginals"].tolist() == [[fsum(w[i, :]) for i in range(8)], [fsum(w[:, i]) for i in range(8)]]


@pytest.mark.parametrize("encoding, d, k", [("binary", 17, 16), ("one_hot", 2, 40)])
def test_a_readout_wider_than_64_bits_is_pooled_and_summarized_without_grid_flat_indices(encoding, d, k):
    """Counts readouts 68 and 80 bits wide are decoded, pooled and summarized without a flat index of the grid.

    A binary readout of 17 variables with K = 16 is 68 bits wide and its grid has 2**68 points, and a
    one-hot readout of two variables with K = 40 is 80 bits wide, with words that encode no grid point.
    ``QHD.analyze`` pools the counts of its chunks with ``method._pool_counts``, whose grid points, pooled
    counts, invalid count and total must equal a scalar pooling written here from the register layout: bits
    ``j b`` to ``j b + b - 1`` hold the binary index of variable j, and one-hot register j, bits ``j K`` to
    ``j K + K - 1``, encodes grid index i by its single set bit i. The int64 and the exact Python-integer
    pooling agree. ``method._summarize`` then summarizes the sparse rows.
    """
    from nwqlib.execution import Histogram

    names = sp.symbols(f"x0:{d}")
    problem = Optimization(objective=sum((v - sp.Rational(1, 3)) ** 2 for v in names), variables=names,
                           bounds=((0.0, 1.0),) * d)
    fields = dict(encoding="binary", boundary="periodic") if encoding == "binary" else {}
    selected = plan(problem, method=QHD(num_grid_points=k, num_steps=1, total_time=1.0, **fields), seed=7)
    grid = owner._grid(selected)
    bits = 4 if encoding == "binary" else None
    width = d * (bits or k)

    def decoded(word):
        if bits is not None:
            return tuple((word >> (j * bits)) & (k - 1) for j in range(d))
        point = []
        for j in range(d):
            register = (word >> (j * k)) & ((1 << k) - 1)
            if register == 0 or register & (register - 1):
                return None
            point.append(register.bit_length() - 1)
        return tuple(point)

    if bits is not None:
        chunks = (([5, (1 << 67) + 3, 5], [2, 1, 1]), ([(1 << 67) + 3, 9 << 40], [3, 4]))
    else:
        valid = (1 << 3) | (1 << (40 + 5))
        chunks = (([valid, 0, (1 << 79) | (1 << 2), valid], [2, 5, 1, 1]),
                  ([(1 << 79) | (1 << 2), (3 << 40) | 1, 1 << 40], [4, 2, 7]))
    reference, invalid, total = {}, 0, 0
    for words, counts in chunks:
        for word, count in zip(words, counts, strict=True):
            total += count
            point = decoded(word)
            if point is None:
                invalid += count
            else:
                reference[point] = reference.get(point, 0) + count
    for exact in (False, True):
        histograms = (Histogram(width, words, np.array(counts, dtype=np.int64)) for words, counts in chunks)
        unique, pooled, bad, counted = owner._pool_counts(histograms, grid, bits, exact=exact)
        assert [tuple(row) for row in unique.tolist()] == sorted(reference)
        assert [int(c) for c in pooled.tolist()] == [reference[point] for point in sorted(reference)]
        assert (bad, counted) == (invalid, total) and (invalid > 0) == (encoding == "one_hot")
    summary = owner._summarize(selected, pooled, invalid / total, 1.0, 0, "counts", shots=total, points=unique)
    peak = max(reference.values())
    assert summary["probability_maximizer_indices"] == min(p for p, c in reference.items() if c == peak)
    assert summary["probability_maximizer_probability"] == peak / total
    assert summary["valid_mass"] == sum(reference.values()) / total
