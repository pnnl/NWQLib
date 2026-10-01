"""Actual selected power, compactness, preparation and cache relations."""

from math import pi
from unittest.mock import Mock
import numpy as np
import pytest
from fractions import Fraction
from nwqlib.algorithms.qpe import QCELS, numerical
from nwqlib.algorithms.qpe import method as owner, powers
from nwqlib.algorithms.qpe.records import validate_selection
from nwqlib.operators import ingest_pauli
from nwqlib.operators.inputs import OperatorInput
from nwqlib.resources import ResourceContext, estimate
from nwqlib import solve
from test_qpe_selected import make_plan


def forbid(*a, **k):
    raise AssertionError("unselected numerical or native work")


def test_qpe_admission_reports_the_supplied_work_and_byte_limits():
    method = QCELS(max_work=100, max_bytes=200)
    method._admit(100, 200)
    for work, size in ((101, 200), (100, 201)):
        with pytest.raises(ValueError) as caught:
            method._admit(work, size)
        message = str(caught.value)
        assert f"requires work={work}, bytes={size}" in message
        assert "max_work=100, max_bytes=200" in message


def test_safe_tau_conservative_bound_without_eigensystem(monkeypatch):
    monkeypatch.setattr(numerical, "diagonalize_hermitian", forbid)
    h = np.array([[0.2, 0.3j], [-0.3j, -0.4]])
    tau, bound, source = owner._safe_time(h, None)
    assert bound == pytest.approx(0.7, rel=2e-15, abs=0)
    assert tau == pytest.approx(0.9 * pi / 0.7, rel=2e-15, abs=0)
    assert owner._safe_time(np.zeros((2, 2)), None)[:2] == (1.0, 0.0)
    for scale in (1e-280, 1e280):
        resolved, radius, _ = owner._safe_time(h * scale, None)
        assert resolved * scale == pytest.approx(tau, rel=2e-15, abs=0)
        assert radius / scale == pytest.approx(bound, rel=2e-15, abs=0)


@pytest.mark.parametrize("signed_time", [0.14, -0.14])
def test_suzuki_angles_match_independent_two_level_product(signed_time):
    terms = (("Z", 0.3), ("X", 0.4))
    schedule = powers.rotation_schedule(terms, signed_time)
    assert tuple(label for label, _ in schedule) == ("Z", "X", "X", "Z")
    i, x, z = np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]]), np.diag([1.0, -1.0])
    result = np.eye(2, dtype=complex)
    for label, time in schedule:
        pauli = x if label == "X" else z
        result = (np.cos(time) * i - 1j * np.sin(time) * pauli) @ result
    rz = np.diag(np.exp(-1j * np.array([1.0, -1.0]) * 0.3 * signed_time / 2))
    rx = np.cos(0.4 * signed_time) * i - 1j * np.sin(0.4 * signed_time) * x
    np.testing.assert_allclose(result, rz @ rx @ rz, rtol=0.0, atol=2e-16)


@pytest.mark.parametrize("removed", [1e-5, 1e-200])
def test_pruning_and_signed_power_time_owned_once(monkeypatch, removed):
    target = ingest_pauli((("I", 0.2), ("Z", 0.3), ("X", 0.4), ("Y", removed)), num_qubits=1)
    monkeypatch.setattr(OperatorInput, "dense_array", forbid)
    count = Mock(wraps=powers._pauli_bound_coefficient_from_terms)
    monkeypatch.setattr(powers, "_pauli_bound_coefficient_from_terms", count)
    selected = make_plan(
        estimator="rfe",
        target=target,
        execution="quantum",
        pauli_pruning_rtol=1e-3,
        controlled_power_error_budget=0.002,
    )
    r = selected.reconstruction
    assert r.identity_coefficient == 0.2 and r.pruned_mass == removed
    assert r.pauli_labels == ("Z", "X") and r.pauli_coefficients.array.tolist() == [0.3, 0.4]
    assert count.call_count == 1
    for item in r.powers:
        # The display of the exact target time p*val(tau).
        assert item.evolution_time == float(item.power * Fraction(0.2))
        # Even a tiny representable removed term carries nonzero evolution
        # error. pytest's default absolute floor would let zero pass here.
        assert item.pruning_error == pytest.approx(abs(item.power) * 0.2 * removed, rel=1e-15, abs=0)
        assert max(item.pruning_error, item.evolution_error) <= item.total_error <= 0.002
    validate_selection(selected)
    estimate(selected.construction)
    assert count.call_count == 1


def test_qpe_compact_repeat_and_shot_multiplicity():
    target = ingest_pauli((("I", 0.2), ("Z", 0.3), ("X", 0.4)), num_qubits=1)
    selected = make_plan(
        estimator="rfe",
        target=target,
        execution="quantum",
        controlled_power_error_budget=1e-4,
        shots=37,
    )
    folded = estimate(selected.construction, context=ResourceContext(batch_schedule="serial"))
    # Repeated draws share one query, whose batch requests 37 shots per draw,
    # so the 14 draws still request 2 * 7 * 37 shots over fewer settings.
    queries = selected.reconstruction.queries
    assert sum(q.multiplicity for q in queries) == 14 and any(q.multiplicity > 1 for q in queries)
    assert folded.quantity("shots").fact.value.numerator == 2 * 7 * 37
    assert folded.quantity("settings").fact.value.numerator == len(queries) < 14
    # Preparation + two H + feedback, plus, for a positive power p, its one
    # identity phase and the r(p) repetitions of its own selected step.
    powers_by_value = {p.power: p for p in selected.reconstruction.powers}
    phases = {p: int(p != 0) for p in powers_by_value}
    expected = 37 * sum(q.multiplicity * (4 + phases[q.power] + powers_by_value[q.power].steps)
                       for q in queries)
    assert folded.quantity("calls").fact.value.numerator == expected
    assert len(selected.construction.program.definitions) < expected
    # Stored logical instructions per shot: none to prepare |0>, two H, one
    # feedback RZ, one identity phase for a positive power p, and 2L
    # controlled Pauli rotations in each of its r(p) steps (L = 2 terms,
    # Z and X).
    slots = 37 * sum(q.multiplicity * (3 + phases[q.power] + 2 * 2 * powers_by_value[q.power].steps)
                     for q in queries)
    operations = folded.quantity("operations")
    assert operations.interpretation == "exact"
    assert operations.fact.value.numerator == slots
    # The exact trajectory applies the preparation and H once, then before
    # each positive power its identity phase increment and r(p)-r(p_prev)
    # repetitions of the one common step, r(p_max) in total.
    trajectory = make_plan(estimator="rfe", target=target, execution="quantum",
                           controlled_power_error_budget=1e-4)
    positive = [p for p in trajectory.reconstruction.powers if p.power]
    assert trajectory.reconstruction.common_step is not None and positive[-1].steps > len(positive)
    folded = estimate(trajectory.construction, context=ResourceContext(batch_schedule="serial"))
    assert folded.quantity("calls").fact.value.numerator == 2 + len(positive) + positive[-1].steps
    assert folded.quantity("operations").fact.value.numerator == (
        1 + len(positive) + 2 * 2 * positive[-1].steps)
    # A dense power supplies no operation law, so its count stays unknown, not zero.
    dense = make_plan(estimator="rfe", execution="quantum", shots=37)
    assert estimate(dense.construction).quantity("operations").interpretation == "unavailable"


def test_suzuki_step_cx_law_counts_the_built_step():
    from qiskit import transpile
    from nwqlib.problems import ingest_occupation

    def cx(circuit):
        return transpile(circuit, basis_gates=["cx", "u"], optimization_level=0).count_ops().get("cx", 0)

    # Supports 1, 2 and 3: a controlled leaf of support w has 2w CX and each
    # term gives two leaves, so one step has 4*(1+2+3) = 24 (powers.suzuki_step_cx).
    target = ingest_pauli((("III", 0.19), ("IIX", 0.7), ("IYZ", -0.4), ("XYZ", 0.3)), num_qubits=3)
    selected = make_plan(
        target=target,
        execution="quantum",
        initial=ingest_occupation("101", num_qubits=3),
        shots=7,
    )
    steps = [b for b in selected.blocks if b.record.signature.target.name == "qpe.suzuki_step"]
    assert steps
    for block in steps:
        law = next(law for law in block.record.resource_laws if law.metric == "cx")
        assert cx(block._constructor(block, (), None)) == law.value == 24
    # The occupation preparation, H, the identity phase P and the feedback RZ
    # add no CX, and Repeat supplies each power's r_p steps.
    powers_by_value = {p.power: p.steps for p in selected.reconstruction.powers}
    folded = estimate(selected.construction, context=ResourceContext(basis="cx")).quantity("cx")
    assert folded.interpretation == "exact"
    assert folded.fact.value.numerator == 7 * sum(
        24 * powers_by_value[q.power] for q in selected.reconstruction.queries
    )


def test_host_reuses_one_setup_and_report_does_not_redraw(monkeypatch):
    diagonalize = Mock(wraps=numerical.diagonalize_hermitian)
    schedule = Mock(wraps=numerical.planned_power_schedule)
    monkeypatch.setattr(numerical, "diagonalize_hermitian", diagonalize)
    monkeypatch.setattr(numerical, "planned_power_schedule", schedule)
    selected = make_plan(estimator="rwpe")
    assert diagonalize.call_count == 0 and schedule.call_count == 1
    result = solve(selected)
    assert diagonalize.call_count == 1
    result.report()
    result.analyze()
    assert diagonalize.call_count == 1 and schedule.call_count == 1
    events = [
        {b.parameter: b.value for b in c.applications[0].arguments}
        for c in result.data.observations.chunks
    ]
    assert [e["eigensystems"] for e in events] == [1, 0, 0]
    assert [e["projections"] for e in events] == [1, 0, 0]


def test_zero_remaining_pruning_budget_preserves_only_zero_formula_bound():
    # Dyadic inputs make every emitted RWPE parameter exact: pruning
    # 0.25*2**-8 = 2**-10 uses the whole allowance, and the one-term step
    # 0.25 with leaves 0.5*0.25*0.25 has no formula, angle, time or phase
    # error (powers._independent_powers). Two anticommuting terms have a
    # nonzero formula coefficient and cannot fit the zero remainder.
    target = ingest_pauli((("Z", 0.25),), num_qubits=1)
    config = QCELS(controlled_power_error_budget=2.0 ** -10)
    kwargs = dict(
        target=target,
        base=None,
        tau=0.25,
        identity=0.0,
        pruned_mass=2.0 ** -8,
        powers=(1,),
        backend="trotter_error_budgeted",
        config=config,
        route="rwpe",
    )
    rows = powers.select_powers(labels=("Z",), coefficients=(0.25,), **kwargs).powers
    assert rows[0].steps == 1 and rows[0].evolution_error == 0
    assert rows[0].pruning_error == rows[0].total_error == 2.0 ** -10
    with pytest.raises(ValueError, match="exhausted"):
        powers.select_powers(labels=("Z", "X"), coefficients=(0.25, 0.5), **kwargs)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_controlled_global_phase_is_a_real_ancilla_signal(sign):
    selected = make_plan(phase=True, target=sign * np.eye(2), execution="quantum")
    result = solve(selected)
    real = {q.experiment for q in selected.reconstruction.queries if q.phase_shift == 0.0}
    first = next(s for s in result.samples if s.power == 1 and s.experiment in real)
    assert first.mean == pytest.approx(sign, abs=2e-15)
    # U=I and U=-I differ only by phase before control, yet control distinguishes them.


@pytest.mark.parametrize("phase", [False, True])
def test_scalar_native_selected_power_one_control_without_system(phase):
    matrix = np.array([[np.exp(0.31j) if phase else 0.31]])
    selected = make_plan(phase=phase, target=matrix, initial=[1.0], execution="quantum")
    assert [(r.name, r.width) for r in selected.construction.program.registers] == [("ancilla", 1)]
    result = solve(selected)
    assert result.phase == pytest.approx(
        (0.31 * (1 if phase else -0.2) / (2 * pi)) % 1.0, abs=1e-14
    )


def test_scalar_product_formula_empty_word_keeps_identity_only():
    selected = make_plan(
        estimator="rfe",
        execution="quantum",
        target=np.array([[0.4]]),
        initial=[1.0],
        controlled_power_backend="trotter_error_budgeted",
    )
    r = selected.reconstruction
    assert r.identity_coefficient == 0.4 and not r.pauli_terms
    # The empty common step emits no gate; only the identity phase increments
    # remain, with their formation error in the recorded bound.
    assert all(p.selected_definition is None and p.total_error <= 1e-3 for p in r.powers)
    assert r.common_step is not None and not any(
        b.record.signature.name == "common_step" for b in selected.blocks)
    validate_selection(selected)
    # Only one-qubit ancilla gates remain, so the CX count is an exact zero, not unknown.
    cx = estimate(selected.construction, context=ResourceContext(basis="cx")).quantity("cx")
    assert cx.interpretation == "exact"
    assert cx.fact.value.numerator == 0


def test_census_preselection_selects_full_relaxed_or_refuses_before_structure(monkeypatch):
    from types import SimpleNamespace
    from nwqlib.subroutines.trotterization import error_budget

    # These limits price the census alone; the later independent recheck stage
    # (powers.independent_recheck_work) is its own admission and is isolated here.
    monkeypatch.setattr(powers, "independent_recheck_work", lambda *a: (0, (0, 0, 0), (0, 0, 0)))
    terms = (("Z", 0.3), ("X", 0.4), ("Y", 0.2))
    target = SimpleNamespace(manifest=SimpleNamespace(basis=SimpleNamespace(dimension=2)))

    def census(**limits):
        return powers.select_powers(
            target=target, base=None, tau=0.1, labels=tuple(label for label, _ in terms),
            coefficients=tuple(value for _, value in terms), identity=0.0, pruned_mass=0.0,
            powers=(0,), backend="trotter_error_budgeted", config=QCELS(**limits), route="rwpe",
        ).evaluation

    # p = 3 terms on q = 1 qubit, every pair anticommuting: P = 3, E = 3 and N = J = 5. The full
    # census charges 2pq + w(P+N) + E + N = 22 work and the relaxed one 2pq + wP + p + 2E = 18
    # (error_budget.census_work). In bytes (error_budget.census_bytes) the relaxed envelope needs
    # 67912 at its largest fitting block and the full one at least 69768 at every checked block.
    full = census(max_work=22)
    assert (full.bound_variant, full.coefficient_arithmetic) == ("exact_census", "outward_float64_scaled")
    assert (full.pair_commutation_checks, full.nested_commutation_checks) == (3, 5)
    # p = 8 commuting terms on q = 4 qubits: before the pair structure the full envelope (E = P = 28,
    # N = F = J = 140) charges 400 and the relaxed one 156. The built structure has E = N = 0, and the
    # full census then charges 92, so max_work=156 admits the relaxed envelope first and then selects
    # the full expression from the actual counts.
    commuting = ("ZIII", "IZII", "IIZI", "IIIZ", "ZZII", "IZZI", "IIZZ", "ZIIZ")
    upgraded = powers.select_powers(
        target=SimpleNamespace(manifest=SimpleNamespace(basis=SimpleNamespace(dimension=16))),
        base=None, tau=0.1, labels=commuting, coefficients=(0.1,) * 8, identity=0.0,
        pruned_mass=0.0, powers=(0,), backend="trotter_error_budgeted",
        config=QCELS(max_work=156), route="rwpe",
    ).evaluation
    assert (upgraded.bound_variant, upgraded.pair_commutation_checks,
            upgraded.nested_commutation_checks) == ("exact_census", 28, 0)
    monkeypatch.setattr(error_budget, "triple_structure", forbid)
    assert error_budget.choose_census_block(3, 1, 3, 5, 68000, order=2, variant="exact_census") is None
    for limits in (dict(max_work=21), dict(max_bytes=68000)):
        relaxed = census(**limits)
        assert (relaxed.bound_variant, relaxed.nested_commutation_checks) == ("relaxed_prefix", 0)
        assert relaxed.coefficient >= full.coefficient
    monkeypatch.setattr(error_budget, "pack_labels", forbid)
    for limits, named in ((dict(max_work=17), "requested_work=18,.* Raise the Method's max_work to at least 18$"),
                          (dict(max_bytes=67000), "Raise the Method's max_bytes to at least \\d+$")):
        # RWPE reserves no common step, so the header names the census alone.
        with pytest.raises(ValueError, match=f"^QPE census admission: .*{named}"):
            census(**limits)


# The common-grid checker below recomputes, in exact rational arithmetic from
# the stored counts, step, allowances and emitted parameters, the common integer
# and every complete prefix bound, independently of
# error_budget.select_common_step. The resource notebook uses the same checker.
def _upward(value):
    from fractions import Fraction as Q
    from math import inf, nextafter
    value = Q(value)
    result = float(value)
    return nextafter(result, inf) if Q(result) < value else result


def _common_integer(W, loss, tau, allowances):
    from fractions import Fraction as Q
    from functools import reduce
    from math import gcd, isqrt
    ps = sorted(p for p in allowances if p > 0)
    if not ps:
        return 0, 0
    g = reduce(gcd, ps)
    square = Q(0)
    for p in ps:
        t = p*Q(tau)
        remaining = Q(allowances[p])-Q(loss)*t
        if remaining < 0 or (remaining == 0 and W):
            raise ValueError("pruning exhausts a common-grid allowance")
        if W:
            square = max(square, Q(W)*(g*Q(tau))**2*t/remaining)
    ceil_square = -(-square.numerator//square.denominator)
    return g, 1+isqrt(max(0, ceil_square-1))


def _check_common_selection(selection, *, W_used, W_exact, delta_W, coefficients, identity, loss,
                            actual_angles, actual_increments):
    from fractions import Fraction as Q
    from math import inf, isfinite, nextafter
    tau = selection.common_tau
    allowances = dict(selection.common_power_allowances)
    counts = dict(selection.common_prefix_steps)
    published = dict(selection.common_prefix_bounds)
    assert set(allowances) == set(counts) == set(published)
    assert Q(W_exact) <= W_used <= Q(W_exact)*(1+delta_W)
    g, m = _common_integer(W_used, loss, tau, allowances)
    _, m_lo = _common_integer(W_exact, loss, tau, allowances)
    _, m_hi = _common_integer(Q(W_exact)*(1+delta_W), loss, tau, allowances)
    assert (g, m) == (selection.common_g, selection.common_m)
    assert m_lo <= m <= m_hi
    h = selection.common_step_time
    assert h == float(g*Q(tau)/m)
    assert len(actual_angles) == len(coefficients)
    assert tuple(actual_angles) == tuple(0.5*h*c for c in coefficients)
    assert set(actual_increments) == set(allowances)
    C = sum((abs(Q(c)) for c in coefficients), Q(0))
    angle_one = sum((abs(Q(a)-Q(h)*Q(c)/2)
                     for a, c in zip(actual_angles, coefficients, strict=True)), Q(0))
    phase = Q(0)
    previous = 0
    windows = []
    parts_by_power = {}
    for p in sorted(allowances):
        increment = actual_increments[p]
        assert increment == -(p-previous)*tau*identity
        previous = p
        phase += Q(increment)
        r = counts[p]
        assert r == (p//g)*m
        t = p*Q(tau)
        parts = dict(
            trotter=Q(W_used)*r*abs(Q(h))**3,
            pruning=Q(loss)*abs(t),
            time=C*abs(r*Q(h)-t),
            identity=abs(phase+t*Q(identity)),
            angles=2*r*angle_one,
        )
        total = sum(parts.values(), Q(0))
        assert total <= Q(allowances[p])
        assert published[p] == _upward(total)
        other = total-parts["trotter"]
        low = Q(W_exact)*r*abs(Q(h))**3+other
        high = Q(W_exact)*(1+delta_W)*r*abs(Q(h))**3+other
        assert _upward(low) <= published[p] <= _upward(high)
        windows.append((p, r, (p//g)*m_lo, (p//g)*m_hi,
                        published[p], _upward(low), _upward(high)))
        parts_by_power[p] = parts
    p_max = max(allowances)
    assert selection.step_count == counts[p_max]
    assert selection.bound_value == _upward(parts_by_power[p_max]["trotter"])
    residual = min((Q(allowances[p])-(sum(parts_by_power[p].values(), Q(0))-parts_by_power[p]["trotter"]))
                   * counts[p_max]/counts[p] for p in allowances if p > 0)
    stored = Q(selection.error_budget)
    assert stored <= residual
    if isfinite(nextafter(selection.error_budget, inf)):
        assert Q(nextafter(selection.error_budget, inf)) > residual
    return windows


def test_common_grid_powers_record_the_checked_emitted_prefix_bounds():
    """Static product-formula powers share one step on the gcd grid, checked prefix by prefix.

    Powers {2, 4, 8} have g = 2. The independent checker recomputes the
    integer m, every cumulative count r(p) = (p/g)*m, the stored step
    g*val(tau)/m, and every complete prefix subtotal (Trotter, pruning, time
    displacement, identity phase and leaf angles) from the stored inputs and
    the emitted parameters; the recorded coefficient is the census's W_up,
    so its window is collapsed here (W_exact = W_used). A Plan whose emitted
    identity phase or leaf angles exceed a tiny allowance is refused, naming
    the power.
    """
    from fractions import Fraction
    from types import SimpleNamespace
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms.qpe import RFE
    from nwqlib.subroutines.trotterization.error_budget import upper_combined_error

    target = ingest_pauli((("II", .1), ("ZI", .3), ("XX", .2), ("IY", -.25)), num_qubits=2)
    selected = plan(Eigenproblem(A=target), method=RFE(initial_state=[1, 0, 0, 0], num_frequencies=9,
                                                       num_samples=3, tau=0.3), seed=2)
    rec, budget = selected.reconstruction, selected.method.controlled_power_error_budget
    assert sorted(p.power for p in rec.powers if p.power) == [2, 4, 8]
    common = rec.common_step
    step = next(b for b in selected.blocks if b.record.signature.name == "common_step")
    labels_and_coefficients, step_time = step._payload
    expressions = {e.id: e.value for e in selected.construction.program.expressions}
    increments, previous = {}, 0
    for power in rec.powers:
        name = f"identity_{power.power}"
        increments[power.power] = (float(expressions[name].value.value) if name in expressions
                                   else -(power.power - previous) * rec.tau * rec.identity_coefficient)
        previous = power.power
    selection = SimpleNamespace(
        common_tau=common.tau, common_g=common.g, common_m=common.m,
        common_step_time=common.step_time, step_count=common.step_count,
        bound_value=common.bound_value, error_budget=common.error_budget,
        common_power_allowances=tuple((p.power, budget) for p in rec.powers),
        common_prefix_steps=tuple((p.power, p.steps) for p in rec.powers),
        common_prefix_bounds=tuple((p.power, p.total_error) for p in rec.powers))
    W = Fraction(*rec.bound_coefficient)
    coefficients = rec.pauli_coefficients.array.tolist()
    windows = _check_common_selection(
        selection, W_used=W, W_exact=W, delta_W=0, coefficients=coefficients,
        identity=rec.identity_coefficient, loss=rec.pruned_mass,
        actual_angles=tuple(angle for _, angle in powers.rotation_schedule(labels_and_coefficients,
                                                                           step_time)[:len(coefficients)]),
        actual_increments=increments)
    assert len(windows) == len(rec.powers) and common.g == 2
    for power in rec.powers:
        assert power.evolution_time == float(power.power * Fraction(rec.tau))
        assert (max(power.pruning_error, power.evolution_error) <= power.total_error
                <= upper_combined_error(power.pruning_error, power.evolution_error))
    tiny = ingest_pauli((("I", .1), ("Z", .3)), num_qubits=1)
    with pytest.raises(ValueError, match="at power 1"):
        plan(Eigenproblem(A=tiny), method=QCELS(initial_state=[1, 0], tau=0.3, num_times=3,
                                                controlled_power_error_budget=1e-300), seed=7)


def _pauli_sum(num_qubits, count, seed):
    """Distinct nonidentity Pauli labels with full-mantissa coefficients, as a chemistry sum has."""
    rng = np.random.default_rng(seed)
    labels = set()
    while len(labels) < count:
        label = "".join(rng.choice(list("IXYZ"), num_qubits))
        if label != "I" * num_qubits:
            labels.add(label)
    return [(label, float(c)) for label, c in zip(sorted(labels), rng.normal(0, 0.1, count))]


@pytest.mark.parametrize("case", ["rfe_one_term", "qcels_184_terms"])
def test_default_static_plans_admit_their_common_step_at_a_tenth_of_the_default_work_limit(case):
    """Default RFE on one Pauli term and default QCELS on 184 terms plan at max_work = 1e8.

    The common-step admission (``powers._common_recheck_law``) prices the
    actual input exponents, so these Plans fit a tenth of the default
    together with their census charge. The default changes no selected record.
    """
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms.qpe import RFE

    if case == "rfe_one_term":
        problem = Eigenproblem(A=ingest_pauli((("I", -.125), ("Z", .375)), num_qubits=1))
        method = RFE(initial_state=[1, 0])
    else:
        problem = Eigenproblem(A=ingest_pauli(_pauli_sum(8, 184, 11), num_qubits=8))
        method = QCELS(initial_state=np.eye(256)[3])
    assert method.max_work == 1_000_000_000
    selected = plan(problem, method=method.revise(max_work=10**8), seed=17)
    reference = plan(problem, method=method, seed=17)
    rec, ref = selected.reconstruction, reference.reconstruction
    assert rec.common_step is not None and len(rec.powers) > 10
    assert rec.powers == ref.powers and rec.common_step == ref.common_step


def test_common_step_selection_reduces_the_coefficients_once(monkeypatch):
    """One QPE selection forms the kept mass and the leaf-angle discrepancy once, not once per prefix."""
    from nwqlib import Eigenproblem, plan
    from nwqlib.subroutines.trotterization import error_budget

    calls = []
    reduce = error_budget._common_reductions
    monkeypatch.setattr(error_budget, "_common_reductions",
                        lambda *a: calls.append(len(a[0])) or reduce(*a))
    monkeypatch.setattr(error_budget, "emitted_subtotal", forbid)
    selected = plan(Eigenproblem(A=ingest_pauli(_pauli_sum(3, 12, 5), num_qubits=3)),
                    method=QCELS(initial_state=np.eye(8)[0]), seed=7)
    assert len(selected.reconstruction.powers) == 11
    assert calls == [12]


def test_selected_power_accepts_a_finite_total_whose_component_sum_overflows():
    """max(P, E) <= T <= up(P + E) holds with an infinite upper endpoint.

    Exact parts x = F/2 + 2**968 and y = F/2 - 2**968, F the largest finite
    binary64 value, publish P = up(x) = 2**1023, E = up(y) = F/2 and
    T = up(x + y) = F, while P + E exceeds F. A total below a component or
    above the upward component sum stays refused.
    """
    from sys import float_info
    from math import nextafter
    from nwqlib.algorithms.qpe.records import QPEPower

    F = float_info.max
    x, y = Fraction(F) / 2 + Fraction(2) ** 968, Fraction(F) / 2 - Fraction(2) ** 968
    P, E = 2.0 ** 1023, F / 2
    assert Fraction(nextafter(P, 0)) < x <= Fraction(P) and Fraction(E) == y + Fraction(2) ** 968
    assert x + y == Fraction(F)

    def record(pruning, evolution, total):
        return QPEPower(power=1, backend="trotter_error_budgeted", evolution_time=1.0, steps=1,
                        pruning_error=pruning, evolution_error=evolution, total_error=total)

    assert record(P, E, F).total_error == F
    assert record(1.0, 1.0, 2.0).total_error == 2.0
    for pruning, evolution, total in ((P, E, nextafter(P, 0)), (1.0, 1.0, nextafter(2.0, 3))):
        with pytest.raises(ValueError, match="outward pruning plus evolution error"):
            record(pruning, evolution, total)


def test_shared_step_error_includes_its_leaf_angle_formation():
    """The shared step's epsilon bounds the emitted leaves, not only the ideal product formula.

    For H = 0.3 Z at step 0.1, W_up = 0 and the two emitted leaves
    0.5*0.1*0.3 differ from the exact product of the represented 0.1 and 0.3
    by 1080863910568919/649037107316853453566312041152512 in total, so the
    step's operator-norm error is that value rounded upward
    (``powers._step_block``).
    """
    from nwqlib import Eigenproblem, plan

    selected = plan(Eigenproblem(A=ingest_pauli((("Z", .3),), num_qubits=1)),
                    method=QCELS(initial_state=[1.0, 0.0], tau=0.1), execution="quantum", seed=7)
    step = next(b for b in selected.blocks if b.record.signature.name == "common_step")
    assert selected.reconstruction.common_step.step_time == 0.1
    assert selected.reconstruction.bound_coefficient == (0, 1)
    formation = 2 * abs(Fraction(0.5 * 0.1 * 0.3) - Fraction(0.1) * Fraction(0.3) / 2)
    assert formation == Fraction(1080863910568919, 649037107316853453566312041152512)
    assert step.record.semantics.epsilon == 1.6653345369377347e-18
    assert Fraction(np.nextafter(step.record.semantics.epsilon, 0)) < formation <= Fraction(
        step.record.semantics.epsilon)


_TINY = float.fromhex("0x0.0000000000001p-1022")


@pytest.mark.parametrize("W, loss, ps, coefficients, identity, tau, budget, max_steps, outcome", [
    (1, 0, (0,), (0.3,), 0.1, 0.2, 1.0, 1000, "ok"),
    (0, 0, (0, 1, 2), (0.5,), 0.5, _TINY, 1.0, 100, "ok"),
    (0, 0, (0, 1, 3, 17), (0.3,), 0.7, 0.1, 1.0, 1000, "ok"),
    (0, Fraction(1, 4), (0, 1), (), 0.0, 1.0, 0.25, 1000, "ok"),
    (1, 1, (0, 1), (1.0,), 0.0, 1.0, 1.0, 1000, "exhausts"),
    (Fraction(2) ** 2000, 0, (0, 1, 48), (0.3,), 0.0, 1.0, 0.001, 100, "exceeds max_steps"),
    (1, Fraction(1, (1 << 10000) + 1), (0, 1, 3), (0.3,), -0.7, 0.01, 1.0, 1000, "ok"),
    (0, 0, (0, 2), (), 1.7976931348623157e308, 1.0, 1.7976931348623157e308, 100, "must be finite"),
    (0, 0, (2,), (), 0.0, 1.7976931348623157e308, 1.7976931348623157e308, 100, "binary64"),
    (Fraction(2) ** 2000, 0, (1,), (1.0,), 0.0, _TINY, _TINY, 10**6, "does not meet the allowance"),
], ids=["zero_only", "subnormal_step", "commuting_angle_discrepancy", "pruning_equality",
        "pruning_exhaustion", "refused_count", "wide_loss", "phase_overflow", "step_overflow",
        "subnormal_allowance"])
def test_shared_reductions_equal_the_per_prefix_subtotal_at_scalar_boundaries(
        W, loss, ps, coefficients, identity, tau, budget, max_steps, outcome):
    """The selector's prefix subtotals equal the five parts recomputed from each emitted prefix.

    The selector forms the kept mass and angle discrepancy once and a
    running phase prefix. The reference recomputes every part of
    ``error_budget.emitted_subtotal`` inline from the emitted parameters of
    that prefix: W*r*abs(h)**3, loss*abs(t), mass*abs(r*h - t),
    abs(phase + t*identity) and 2*r*sum(abs(a_j - h*c_j/2)). The cases cover zero-only
    powers, subnormal steps, a nonzero angle discrepancy of a commuting
    term, pruning equality and exhaustion, a count refused by max_steps, a
    10001-bit loss denominator, phase and step overflow and a subnormal
    step whose own formula term exceeds a subnormal allowance.
    """
    from nwqlib.subroutines.trotterization import error_budget as eb

    evaluation = eb._BoundCoefficientEvaluation(
        coefficient=Fraction(W), pauli_term_count=len(coefficients), pair_commutation_checks=0,
        nested_commutation_checks=0, bound_variant="exact_census",
        coefficient_arithmetic=eb.COEFFICIENT_ARITHMETIC)
    kw = dict(coefficients=coefficients, identity=identity, tau=tau,
              allowances=dict.fromkeys(ps, budget), dropped_mass=loss, max_steps=max_steps)
    if outcome != "ok":
        with pytest.raises(ValueError, match=outcome):
            eb._select_common_step(evaluation, **kw)
        return
    selection, bounds, increments = eb._select_common_step(evaluation, **kw)
    if selection is None:
        assert bounds == increments == {} and not any(ps)
        return
    h_hat = selection.common_step_time
    half_angles = tuple(0.5 * h_hat * c for c in coefficients)
    steps = dict(selection.common_prefix_steps)
    h, c = Fraction(h_hat), [Fraction(v) for v in coefficients]
    for p in ps:
        t, r = p * Fraction(tau), steps[p]
        phase = sum((Fraction(increments[q]) for q in ps if q <= p), Fraction(0))
        parts = dict(trotter=Fraction(W) * r * abs(h) ** 3, pruning=Fraction(loss) * abs(t),
                     time=sum(map(abs, c), Fraction(0)) * abs(r * h - t),
                     identity=abs(phase + t * Fraction(identity)),
                     angles=2 * r * sum((abs(Fraction(a) - h * v / 2) for a, v in zip(half_angles, c)),
                                        Fraction(0)))
        assert bounds[p] == (sum(parts.values(), Fraction(0)), parts)
        assert bounds[p] == eb.emitted_subtotal(W, loss, coefficients, identity, tau, p, r, h_hat,
                                                half_angles, tuple(increments[q] for q in ps if q <= p))


def test_common_step_refusal_names_the_method_limit_to_pass():
    """A refused common step names the Method field and the total that admits it."""
    from types import SimpleNamespace

    args = (Fraction(1), 0.0, (0, 1, 2), (0.3, -0.2), 0.1, 0.05, dict.fromkeys((0, 1, 2), 1e-3))
    work, peak, _, _ = powers._common_recheck_law(*args, 1000)
    config = SimpleNamespace(max_work=work, max_bytes=peak, max_trotter_steps=1000)
    assert powers._admit_common_recheck(*args, config) == (work, peak)
    with pytest.raises(ValueError, match=f"raise the Method's max_work to at least {work}$"):
        powers._admit_common_recheck(*args, SimpleNamespace(
            max_work=work - 1, max_bytes=peak, max_trotter_steps=1000))
    with pytest.raises(ValueError, match=f"max_work to at least {work} and max_bytes to at least {peak}$"):
        powers._admit_common_recheck(*args, SimpleNamespace(
            max_work=work - 1, max_bytes=peak - 1, max_trotter_steps=1000))


def _rwpe_parts(record, step, tau, identity, dropped, W):
    """Recompute an RWPE power's five contributions from its emitted parameters.

    With t = val(p)*val(tau), represented step h, r steps, emitted leaves
    a_j = 0.5*h*c_j and phase phi = -p*tau*c_I, the pruning part is
    abs(t)*d_up and the evolution part r*W*abs(h)**3 + r*A +
    M*abs(r*h - t) + abs(val(phi) + t*c_I), A = 2*sum abs(val(a_j) - h*c_j/2)
    (``powers._independent_powers``). Returns (pruning, evolution, step bound).
    """
    t = Fraction(record.power) * Fraction(tau)
    terms, h_hat = step._payload if step is not None else ((), 0.0)
    h, c = Fraction(h_hat), [Fraction(value) for _, value in terms]
    A = 2 * sum((abs(Fraction(0.5 * h_hat * value) - h * v / 2)
                 for (_, value), v in zip(terms, c)), Fraction(0))
    phase = Fraction(-record.power * tau * identity)
    evolution = (W * record.steps * abs(h) ** 3 + record.steps * A
                 + sum(map(abs, c), Fraction(0)) * abs(record.steps * h - t)
                 + abs(phase + t * Fraction(identity)))
    return abs(t) * Fraction(dropped), evolution, W * abs(h) ** 3 + A


def test_rwpe_power_bound_includes_its_emitted_parameter_formation():
    """An RWPE power at an exact product-formula step still bounds its rounded leaves.

    For H = 0.3 Z, tau = 0.1 and p = +-1, one step h = val(0.1) has W_up = 0
    and no pruning, time or identity error, and the two leaves 0.5*0.1*0.3
    differ from h*0.3/2 by 1080863910568919/649037107316853453566312041152512
    in total. The step epsilon and total_error are that value rounded
    upward; power zero has no step and zero error; an allowance of 1e-20
    below it refuses (``powers._independent_powers``).
    """
    from nwqlib.algorithms.qpe import RWPE

    target = ingest_pauli((("Z", 0.3),), num_qubits=1)
    config = RWPE(initial_state=[1.0, 0.0], tau=0.1)
    kwargs = dict(target=target, base=None, tau=0.1, labels=("Z",), coefficients=(0.3,),
                  identity=0.0, pruned_mass=0.0, backend="trotter_error_budgeted", route="rwpe")
    formation = 2 * abs(Fraction(0.5 * 0.1 * 0.3) - Fraction(0.1) * Fraction(0.3) / 2)
    assert formation == Fraction(1080863910568919, 649037107316853453566312041152512)
    for power in (1.0, -1.0):
        selection = powers.select_powers(powers=(power,), config=config, **kwargs)
        record, step = selection.powers[0], selection.blocks[0]
        assert step._payload[1] == power * 0.1 and record.steps == 1
        assert record.total_error == step.record.semantics.epsilon == 1.6653345369377347e-18
        assert Fraction(np.nextafter(record.total_error, 0)) < formation <= Fraction(record.total_error)
    zero = powers.select_powers(powers=(0.0,), config=config, **kwargs)
    assert not zero.blocks and zero.powers[0].total_error == 0
    with pytest.raises(ValueError, match="emitted subtotal 1.6653345369377347e-18 exceeds"):
        powers.select_powers(powers=(1.0,), config=config.revise(controlled_power_error_budget=1e-20),
                             **kwargs)


def test_default_rwpe_plan_records_each_power_complete_bound():
    """Every default RWPE power publishes the upward rounding of its exact five-part subtotal.

    The 14 continuous powers of RWPE on H = 0.3 Z at tau = 0.1 each use one
    step at their own represented time, so their totals are positive
    although W_up = 0 (``powers._independent_powers``).
    """
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms.qpe import RWPE
    from nwqlib.subroutines.trotterization.error_budget import _upward_float

    selected = plan(Eigenproblem(A=ingest_pauli((("Z", .3),), num_qubits=1)),
                    method=RWPE(initial_state=[1.0, 0.0], tau=0.1), execution="quantum", seed=7)
    records = selected.reconstruction.powers
    assert len(records) == 14
    published = {}
    for record in records:
        step = next(b for b in selected.blocks if b.record.content_id == record.selected_definition)
        P, E, intrinsic = _rwpe_parts(record, step, 0.1, 0.0, 0.0, Fraction(0))
        assert (record.pruning_error, record.evolution_error, record.total_error) == (
            _upward_float(P), _upward_float(E), _upward_float(P + E))
        assert step.record.semantics.epsilon == _upward_float(intrinsic)
        assert record.total_error > 0
        published[record.power] = (step._payload[1], step.record.semantics.epsilon, record.total_error)
    assert published[0.3183098861837907] == (
        0.03183098861837907, 3.4049442578366542e-19, 1.2869205069286854e-18)
    assert published[1.0019598957081362] == (
        0.10019598957081362, 1.663158615596698e-18, 2.499089766077121e-18)
    assert published[6.275533862992271] == (
        0.6275533862992271, 1.2518357309634654e-17, 2.2969220589397964e-17)


@pytest.mark.parametrize("power, steps", [(2.3, 1), (-2.3, 1), (13.7, 8), (-13.7, 8)])
def test_rwpe_signed_noncommuting_power_bound_covers_the_emitted_unitary(power, steps):
    """The recorded RWPE bound covers the emitted controlled unitary at its exact signed target.

    H = 0.4 X + 0.3 Z + 0.13 I with dropped mass 1e-6 and tau = 0.1. The
    reference forms, at 90 digits, the emitted control-one block
    exp(i*val(phi)) * S(h)**r from the actual leaf angles and phase, and the
    exact exp(-i*t*H) with t = val(p)*val(tau). Their spectral-norm distance
    lies below evolution_error, and every field equals the upward rounding of
    the five parts recomputed from the emitted parameters.
    """
    import mpmath as mp
    from nwqlib.algorithms.qpe import RWPE
    from nwqlib.subroutines.trotterization.error_budget import _upward_float

    terms = (("X", 0.4), ("Z", 0.3))
    selection = powers.select_powers(
        target=ingest_pauli((("X", 0.4), ("Z", 0.3)), num_qubits=1), base=None, tau=0.1,
        labels=("X", "Z"), coefficients=(0.4, 0.3), identity=0.13, pruned_mass=1e-6,
        powers=(power,), backend="trotter_error_budgeted",
        config=RWPE(initial_state=[1.0, 0.0], tau=0.1), route="rwpe")
    record, step = selection.powers[0], selection.blocks[0]
    assert record.steps == steps
    P, E, intrinsic = _rwpe_parts(record, step, 0.1, 0.13, 1e-6, selection.evaluation.coefficient)
    assert (record.pruning_error, record.evolution_error, record.total_error) == (
        _upward_float(P), _upward_float(E), _upward_float(P + E))
    assert step.record.semantics.epsilon == _upward_float(intrinsic)
    with mp.workdps(90):
        def exact(value):
            value = Fraction(value)
            return mp.mpf(value.numerator) / value.denominator
        X, Z, I = mp.matrix([[0, 1], [1, 0]]), mp.matrix([[1, 0], [0, -1]]), mp.eye(2)
        S = mp.eye(2)
        for label, angle in powers.rotation_schedule(terms, step._payload[1]):
            S = mp.expm(-1j * exact(angle) * (X if label == "X" else Z)) * S
        emitted = mp.exp(1j * exact(selection.increments[power])) * S ** steps
        H = exact(0.4) * X + exact(0.3) * Z + exact(0.13) * I
        reference = mp.expm(-1j * exact(Fraction(power) * Fraction(0.1)) * H)
        error = max(mp.svd(emitted - reference, compute_uv=False))
        assert 0 < error <= exact(record.evolution_error)


def test_rwpe_pruned_mass_is_an_upward_bound_of_the_dropped_terms():
    """RWPE charges the outward dropped-term mass, not its nearest-rounded sum.

    The dropped 1e-5, 1e-5 and 1e-22 have an exact sum above their nearest
    ``fsum``, so the recorded mass is ``error_budget.upper_dropped_mass`` and
    each power's pruning error is abs(t) times that mass rounded upward.
    """
    from math import fsum
    from nwqlib.subroutines.trotterization.error_budget import _upward_float

    raw = (("ZZ", 1.0), ("XI", 1e-5), ("IX", 1e-5), ("YY", 1e-22))
    exact = sum((Fraction(x) for x in (1e-5, 1e-5, 1e-22)), Fraction(0))
    assert Fraction(fsum((1e-5, 1e-5, 1e-22))) < exact
    selected = make_plan(estimator="rwpe", target=ingest_pauli(raw, num_qubits=2),
                         initial=np.eye(4)[0], execution="quantum", pauli_pruning_rtol=1e-4)
    rec = selected.reconstruction
    assert rec.pauli_labels == ("ZZ",)
    assert rec.pruned_mass == 2.000000000000001e-05 and Fraction(rec.pruned_mass) >= exact
    for item in rec.powers:
        charged = abs(Fraction(item.power) * Fraction(0.2)) * Fraction(rec.pruned_mass)
        assert item.pruning_error == _upward_float(charged) and charged >= abs(
            Fraction(item.power) * Fraction(0.2)) * exact


def _six_qubit_ring():
    ring = [("".join("Z" if k in (j, (j + 1) % 6) else "I" for k in range(6)), 1.5)
            for j in range(6)]
    return ring + [("".join("X" if k == j else "I" for k in range(6)), 1.1 + 0.01 * j)
                   for j in range(6)]


def test_census_reserves_the_common_step_before_choosing_its_variant(monkeypatch):
    """Each census admission adds the pre-census common-step reservation (``powers._census``).

    For RFE with num_samples=4 on a 6-qubit, 12-term ring, the reservation is
    63,504 work units and 125,872 bytes. The relaxed census envelope is 354
    before pairs and 246 after, the full census 426 after pairs. Hence
    max_work 63,857 refuses before the masks naming 63,858, 63,858 and
    63,929 select the relaxation, and 63,930 the full expression. At
    63,929, max_bytes 125,871 refuses naming 125,872, which admits. Each
    refusal names only the limits below the selected candidate's combined
    allowance, and RWPE's zero reservation gives the header "QPE census
    admission" at the census owner.
    """
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms.qpe import RFE, RWPE
    from nwqlib.subroutines.trotterization import error_budget

    problem = Eigenproblem(A=ingest_pauli(_six_qubit_ring(), num_qubits=6))
    method = RFE(initial_state=np.eye(64)[5], num_samples=4)
    for work, variant in ((63_858, "relaxed_prefix"), (63_929, "relaxed_prefix"),
                          (63_930, "exact_census")):
        rec = plan(problem, method=method.revise(max_work=work), seed=7).reconstruction
        assert rec.bound_variant == variant and rec.common_step is not None
    assert plan(problem, method=method.revise(max_work=63_929, max_bytes=125_872),
                seed=7).reconstruction.bound_variant == "relaxed_prefix"
    monkeypatch.setattr(error_budget, "pack_labels", forbid)
    with pytest.raises(ValueError, match="reserved_common_work=63504, requested_work=63858, "
                                         "requested_bytes=125872.* Raise the Method's max_work to at least 63858$"):
        plan(problem, method=method.revise(max_work=63_857), seed=7)
    with pytest.raises(ValueError, match="max_bytes=125871\\. .* Raise the Method's max_bytes to at least 125872$"):
        plan(problem, method=method.revise(max_work=63_929, max_bytes=125_871), seed=7)
    with pytest.raises(ValueError, match="Raise the Method's max_work to at least 63858 and max_bytes to at least 125872$"):
        plan(problem, method=method.revise(max_work=63_857, max_bytes=125_871), seed=7)
    labels, coefficients = zip(*_six_qubit_ring())
    with pytest.raises(ValueError, match="^QPE census admission: .*reserved_common_work=0, .*"
                                         "Raise the Method's max_work to at least 354$"):
        powers._census(labels, coefficients, 6, RWPE(initial_state=np.eye(64)[5], tau=0.1, max_work=1))


def test_pre_census_reservation_covers_every_census_coefficient_and_block():
    """The W=None common-step law bounds the law of every census coefficient it can meet.

    ``powers._census_coefficient_bits`` bounds the component widths of W_up
    from the term count and the input exponents alone. The tables include
    the smallest subnormal, near-maximum powers of two, mixed extreme
    exponents, zero coefficients and random full-range magnitudes, each
    under both census expressions and contraction blocks 1, 2 and 65,536.
    """
    import random
    from itertools import product
    from math import frexp, ldexp
    from nwqlib.subroutines.trotterization import error_budget as eb

    rng = random.Random(41)
    labels = ["".join(v) for v in product("IXYZ", repeat=2) if v != ("I", "I")]
    tables = [(.3, .4), (ldexp(1.0, -1074), 1.0), (ldexp(1.0, -1074),) * 3,
              (ldexp(1.0, 1023), ldexp(1.0, 1000)), (0.0, 0.0)]
    tables += [tuple(ldexp(rng.uniform(.5, 1.0), rng.randint(-1073, 1024))
                     for _ in range(rng.randint(2, 15))) for _ in range(100)]
    for coefficients in tables:
        exponents = [frexp(abs(c))[1] for c in coefficients if c]
        bound = powers._census_coefficient_bits(
            len(coefficients), min(exponents, default=None), max(exponents, default=None))
        law = (0.0, (0, 1, 3), coefficients, 0.0, .1, dict.fromkeys((0, 1, 3), .001), 10**6)
        reservation = powers._common_recheck_law(None, *law)
        for variant, block in product(("exact_census", "relaxed_prefix"), (1, 2, 65536)):
            W = eb._pauli_bound_coefficient_from_terms(
                tuple(zip(labels, coefficients)), 2, variant=variant, block=block).coefficient
            assert powers._rational_bits(W) <= bound
            assert all(a <= b for a, b in zip(powers._common_recheck_law(W, *law), reservation))


def test_rwpe_identity_only_powers_bound_their_emitted_phase():
    """Without kept Pauli terms, an RWPE power still bounds its emitted identity phase.

    For the 1x1 Hamiltonian 0.4 with the product-formula backend, no step
    block exists, the Program applies phi = -p*tau*c_I from the selection
    and each total is abs(val(phi) + t*c_I) rounded upward
    (``powers._independent_powers``).
    """
    from nwqlib.subroutines.trotterization.error_budget import _upward_float

    selected = make_plan(estimator="rwpe", execution="quantum", target=np.array([[0.4]]),
                         initial=[1.0], controlled_power_backend="trotter_error_budgeted",
                         max_steps=14)
    rec = selected.reconstruction
    assert not rec.pauli_terms and len(rec.powers) == 14
    for record in rec.powers:
        P, E, _ = _rwpe_parts(record, None, rec.tau, 0.4, 0.0, Fraction(0))
        assert record.selected_definition is None and record.steps == 0
        assert record.total_error == _upward_float(P + E) > 0
    validate_selection(selected)


def test_independent_exact_recheck_is_admitted_before_its_fraction_reductions(monkeypatch):
    """A sampled Plan's per-power exact recheck is its own max_work stage (``powers.independent_recheck_work``).

    Sampled RFE on two system qubits, four Pauli terms and nine powers
    0..8 (K=9, L=4) prices W_independent=131,648 work units, and every
    earlier planning admission fits max_work=131,647. The refusal comes at
    the entry of ``_independent_powers``, before the step law, the exact
    inversion or the emitted subtotal runs, and names 131,648; that value
    admits the same power records as the default limit.
    """
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms.qpe import RFE

    problem = Eigenproblem(A=ingest_pauli((("XI", .7), ("ZX", .4), ("YZ", .3), ("ZZ", .2)),
                                          num_qubits=2))
    method = RFE(initial_state=[1, 0, 0, 0], controlled_power_backend="trotter_error_budgeted",
                 tau=.03, num_samples=32, num_frequencies=9, controlled_power_error_budget=.001)
    reference = plan(problem, method=method, shots=64, seed=7).reconstruction
    assert [p.power for p in reference.powers] == list(range(9)) and reference.common_step is None
    with monkeypatch.context() as patch:
        for name in ("suzuki_step_cx", "_selection_from_evaluation", "_emitted_parts"):
            patch.setattr(powers, name, forbid)
        with pytest.raises(ValueError, match="^QPE independent exact recheck needs work=131648, K=9, L=4, "
                                             ".* Raise the Method's max_work to at least 131648 "):
            plan(problem, method=method.revise(max_work=131_647), shots=64, seed=7)
    admitted = plan(problem, method=method.revise(max_work=131_648), shots=64, seed=7)
    assert admitted.reconstruction.powers == reference.powers
