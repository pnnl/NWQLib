"""Actual selected power, compactness, preparation and cache relations."""

from unittest.mock import Mock
import numpy as np
import pytest
from fractions import Fraction
from nwqlib.algorithms.qpe import QCELS
from nwqlib.algorithms.qpe import powers
from nwqlib.algorithms.qpe.records import validate_selection
from nwqlib.operators import ingest_pauli
from nwqlib.operators.inputs import OperatorInput
from nwqlib.resources import ResourceContext, estimate
from nwqlib import solve
from test_qpe_selected import make_plan


def forbid(*a, **k):
    raise AssertionError("unselected numerical or native work")


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


@pytest.mark.parametrize("sign", [-1.0])
def test_controlled_global_phase_is_a_real_ancilla_signal(sign):
    selected = make_plan(phase=True, target=sign * np.eye(2), execution="quantum")
    result = solve(selected)
    real = {q.experiment for q in selected.reconstruction.queries if q.phase_shift == 0.0}
    first = next(s for s in result.samples if s.power == 1 and s.experiment in real)
    assert first.mean == pytest.approx(sign, abs=2e-15)
    # U=-I differs from I only by a global phase before control; control makes it a real signal.


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


def test_default_static_plans_admit_their_common_step_at_a_tenth_of_the_default_work_limit():
    """Default QCELS on 184 terms plans at max_work = 1e8.

    The common-step admission (``powers._common_recheck_law``) prices the
    actual input exponents, so this Plan fits a tenth of the default
    together with its census charge. The default changes no selected record.
    """
    from nwqlib import Eigenproblem, plan

    problem = Eigenproblem(A=ingest_pauli(_pauli_sum(8, 184, 11), num_qubits=8))
    method = QCELS(initial_state=np.eye(256)[3])
    assert method.max_work == 1_000_000_000
    selected = plan(problem, method=method.revise(max_work=10**8), seed=17)
    reference = plan(problem, method=method, seed=17)
    rec, ref = selected.reconstruction, reference.reconstruction
    assert rec.common_step is not None and len(rec.powers) > 10
    assert rec.powers == ref.powers and rec.common_step == ref.common_step


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


@pytest.mark.parametrize("power, steps", [(-2.3, 1), (13.7, 8)])
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


def _six_qubit_ring():
    ring = [("".join("Z" if k in (j, (j + 1) % 6) else "I" for k in range(6)), 1.5)
            for j in range(6)]
    return ring + [("".join("X" if k == j else "I" for k in range(6)), 1.1 + 0.01 * j)
                   for j in range(6)]


def test_census_reserves_the_common_step_before_choosing_its_variant(monkeypatch):
    """The census variant includes reserved common-step work before its masks are built."""
    from nwqlib import Eigenproblem, plan
    from nwqlib.algorithms.qpe import RFE, RWPE
    from nwqlib.subroutines.trotterization import error_budget

    problem = Eigenproblem(A=ingest_pauli(_six_qubit_ring(), num_qubits=6))
    method = RFE(initial_state=np.eye(64)[5], num_samples=4)
    rec = plan(problem, method=method.revise(max_work=64_002), seed=7).reconstruction
    assert rec.bound_variant == "exact_census" and rec.common_step is not None
    assert plan(problem, method=method.revise(max_work=64_002, max_bytes=125_872),
                seed=7).reconstruction.bound_variant == "exact_census"
    monkeypatch.setattr(error_budget, "pack_labels", forbid)
    with pytest.raises(ValueError, match="QPE census and common-step admission.*max_work"):
        plan(problem, method=method.revise(max_work=64_001), seed=7)
    with pytest.raises(ValueError, match="QPE census and common-step admission.*max_bytes"):
        plan(problem, method=method.revise(max_work=64_002, max_bytes=125_871), seed=7)
    with pytest.raises(ValueError, match="QPE census and common-step admission"):
        plan(problem, method=method.revise(max_work=64_001, max_bytes=125_871), seed=7)
    labels, coefficients = zip(*_six_qubit_ring())
    with pytest.raises(ValueError, match="^QPE census admission"):
        powers._census(labels, coefficients, 6, RWPE(initial_state=np.eye(64)[5], tau=0.1, max_work=1),
                       reservation=(0, 0))
