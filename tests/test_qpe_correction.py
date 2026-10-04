"""Selected QPE meaning, legal alternatives, count identity and immutable evidence."""

from math import pi
import numpy as np
import pytest
from nwqlib import Eigenproblem, solve, plan
from nwqlib.algorithms.qpe import QPEVerification
from nwqlib.algorithms.qpe.records import validate_selection, energy_phase_turns
from nwqlib.ir import PortMap
from nwqlib._counts import CountsSources
from test_qpe_selected import make_plan


def _with_outcomes(chunk, outcomes):
    """A revision of ``chunk`` that holds the supplied outcomes and keeps every other field."""
    from nwqlib.execution import ObservationChunk

    fields = {name: getattr(chunk, name) for name in type(chunk).model_fields if name != "values"}
    return ObservationChunk.from_histogram(outcomes, **{**fields, "parent_id": chunk.content_id})


def changed_node(selected, name, **fields):
    p = selected.construction.program
    definitions = tuple(
        d.revise(node=d.node.revise(**fields)) if d.id == name else d for d in p.definitions
    )
    return selected.revise(
        construction=selected.construction.revise(program=p.revise(definitions=definitions))
    )


@pytest.mark.parametrize(
    "name,fields",
    [
        ("h", dict(ports=(PortMap(port="register", wire="system"),))),
        ("feedback_0", dict(ports=(PortMap(port="register", wire="system"),))),
        ("measure_0", dict(wire="system")),
    ],
)
def test_actual_ancilla_wiring_precedes_native_or_analysis(name, fields):
    selected = make_plan(execution="quantum", shots=7)
    with pytest.raises(ValueError, match="ancilla|feedback|query|Hadamard|acquisition"):
        validate_selection(changed_node(selected, name, **fields))
    validate_selection(selected)


def test_zero_power_and_repeat_bind_actual_ir():
    from nwqlib.operators import ingest_pauli

    selected = make_plan(
        execution="quantum",
        estimator="rfe",
        target=ingest_pauli((("Z", 0.3), ("X", 0.4)), num_qubits=1),
        controlled_power_error_budget=1e-5,
    )
    repeated = next(
        d
        for d in selected.construction.program.definitions
        if d.id.startswith("common_step_") and d.node.count > 1
    )
    with pytest.raises(ValueError, match="multiplicity"):
        validate_selection(changed_node(selected, repeated.id, count=1))
    assert any(q.power == 0 for q in selected.reconstruction.queries)
    gap = next(d.id for d in selected.construction.program.definitions if d.id.startswith("gap_"))
    with pytest.raises(ValueError, match="gap IR"):
        validate_selection(changed_node(selected, gap, children=("h",)))


def test_foreign_target_and_tau_relabel_cannot_reuse_selection():
    selected = make_plan(execution="quantum")
    with pytest.raises(ValueError, match="original target"):
        validate_selection(selected.revise(problem=Eigenproblem(A=np.diag([0.5, -0.7]))))
    with pytest.raises(ValueError, match="Method"):
        validate_selection(selected.revise(method=selected.method.revise(tau=0.3)))
    other = plan(selected.problem, method=selected.method.revise(tau=0.3), seed=7)
    validate_selection(other)
    assert other.problem == selected.problem and other.reconstruction.tau == 0.3


@pytest.mark.parametrize("estimator", ["qcels", "rfe"])
def test_automatic_time_uses_the_same_full_phase_domain_for_dense_and_pauli(estimator):
    from nwqlib.operators import ingest_pauli

    # Both access forms have the same tight radius .5. Planning must use the
    # estimator's phase domain without requiring a Pauli matrix expansion.
    targets = (np.diag([.5, -.5]), ingest_pauli((("Z", .5),), num_qubits=1))
    for target in targets:
        options = dict(max_time=None) if estimator == "qcels" else {}
        selected = make_plan(execution="quantum", estimator=estimator, target=target, tau=None, **options)
        # Two eps allows the multiplication and division defining the time.
        assert selected.reconstruction.tau == pytest.approx(.9 * pi / .5, rel=2*np.finfo(float).eps, abs=0)
        assert selected.reconstruction.aliasing_bound_sufficient


def test_spe_verification_compares_the_lowest_cluster_and_declared_overlap(monkeypatch):
    from nwqlib.algorithms.qpe import SPE, numerical

    result = solve(Eigenproblem(A=np.diag([-.65, .55])), execution="classical",
        method=SPE(initial_state=[np.sqrt(.3), np.sqrt(.7)], tau=1.,
                   overlap_lower_bound=.3, num_samples=8, fourier_degree=4, grid_size=64), seed=7)
    monkeypatch.setattr(numerical, "nominal_eigensystem", lambda *a, **k: pytest.fail("recomputed spectrum"))
    receipt, _ = result.verify(checks=QPEVerification(tolerance=.1, reuse_only=True))
    values = {item.fact.quantity: item.fact.value.value for item in receipt.applications[0].facts}
    assert values["reference_value"] == -.65
    assert values["valid_overlap"] == pytest.approx(.3, rel=0, abs=2e-15)
    assert values["overlap_threshold"] == .3


@pytest.mark.parametrize("access", ["dense", "pauli"])
def test_rwpe_noninteger_time_has_the_declared_native_interference_probability(access):
    from qiskit.quantum_info import Statevector
    from nwqlib._prepared_execution import Run, prepare_experiment
    from nwqlib.algorithms.qpe import RWPE, numerical
    from nwqlib.core.records import Float64
    from nwqlib.ir import Binding
    from nwqlib.operators import ingest_pauli

    target = (np.diag([.6, -.2]) if access == "dense" else
              ingest_pauli((("I", .2), ("Z", .4)), num_qubits=1))
    chosen = plan(Eigenproblem(A=target), method=RWPE(initial_state=[1., 0.],
                  prior_std=.8, max_steps=2, tau=.3), seed=7)
    time, feedback = numerical.rwpe_choose(.37, .8*np.sqrt((np.e-1)/np.e))
    assert not float(time).is_integer()
    point = chosen.resolve("query_1", bindings=(Binding(parameter="feedback", value=Float64(value=feedback)),))
    with Run(chosen) as run:
        prepared = prepare_experiment(point, run=run)
        circuit = prepared.inspect_circuit().remove_final_measurements(inplace=False)
        amplitudes = Statevector.from_instruction(circuit).data
    observed = float(np.sum(np.abs(amplitudes[::2])**2))
    expected = abs(1+np.exp(1j*(-.3*.6*time+feedback)))**2/4
    # Two-qubit dense synthesis or commuting Pauli rotations, followed by two
    # Hadamards and one phase, accumulate only a short binary64 arithmetic chain.
    assert observed == pytest.approx(expected, rel=0, abs=2e-14)


def test_rwpe_unitary_only_input_rejects_before_matrix_access(monkeypatch):
    from nwqlib import SpectralEstimation
    from nwqlib.algorithms.qpe import RWPE
    from nwqlib.operators import OperatorInput

    # Admitting a unitary whose dimension is not a power of two pads it, which
    # reads its dense array, so the refusal must come before that.
    problem = SpectralEstimation(unitary=np.diag([1., 1j, -1.]), initial_state=[1., 0., 0.])
    monkeypatch.setattr(OperatorInput, "dense_array", lambda *a, **k: pytest.fail("expanded unitary"))
    with pytest.raises(ValueError, match="continuous Hamiltonian"):
        plan(problem, method=RWPE())


def test_same_seed_counts_source_is_not_an_independent_population():
    result = solve(make_plan(execution="quantum", shots=11))
    data = result.data
    receipts = {r.content_id: r for r in data.receipts}
    sources = CountsSources(receipts.get, data.trace._acquisition)
    sources.require_independent(data.observations.chunks[0])
    first, second = data.receipts[:2]
    copied = second.revise(counts_sampling=first.counts_sampling)
    receipts[copied.content_id] = copied
    with pytest.raises(ValueError, match="sampling stream"):
        sources.admit_preparation(copied.content_id)
    sources.admit_preparation(second.content_id)
    sources.require_independent(data.observations.chunks[1])
    # Rereading exactly the same acquired source is legal; a controller owns once-only updates.
    sources.require_independent(data.observations.chunks[0])


def test_selection_admits_pooled_draws_only_for_fourier_estimators_with_their_shot_population():
    # A sampled QCELS query cannot stand for several draws, even when its
    # batch requests the matching population.
    qcels = make_plan(execution="quantum", shots=7)
    first = qcels.reconstruction.queries[0]
    widened = changed_node(qcels, "batch_0", repetitions=14)
    widened = widened.revise(reconstruction=widened.reconstruction.revise(
        queries=(first.revise(multiplicity=2), *qcels.reconstruction.queries[1:])))
    with pytest.raises(ValueError, match="only SPE and RFE queries can combine random draws"):
        validate_selection(widened)
    # A pooled SPE query must request shots times its multiplicity.
    spe = make_plan(estimator="spe", execution="quantum", shots=3, num_samples=2)
    assert spe.reconstruction.queries[0].multiplicity == 2
    validate_selection(spe)
    with pytest.raises(ValueError, match="acquisition/readout differs"):
        validate_selection(changed_node(spe, "batch_0", repetitions=3))


def test_pooled_draw_query_uses_its_received_population_with_its_draw_weight(monkeypatch):
    """A pooled query of m draws enters the estimator with the mean of the shots it returned.

    Each count query contributes its mean over the shots actually returned,
    weighted by its original draw multiplicity (method.received_fourier_exposure):
    four of six requested shots are accepted and flagged as a short return,
    the complete population keeps its former value, a population split
    unequally across chunks is pooled by counts rather than averaged by chunk
    means, and an empty or over-returned population is refused. The
    populations are supplied as reduced samples of the query, the input of
    the analysis step that owns this rule.
    """
    import math
    from fractions import Fraction
    from nwqlib.algorithms.qpe import method as owner, numerical

    selected = make_plan(estimator="spe", execution="quantum", shots=3, num_samples=2)
    pooled = selected.reconstruction.queries[0]
    assert (pooled.multiplicity, selected.shots) == (2, 3)
    result = solve(selected)
    first = next(s for s in result.samples if s.experiment == pooled.experiment)
    others = tuple(s for s in result.samples if s is not first)
    assert first.shots == 6
    runs = []
    spe = numerical.spe
    monkeypatch.setattr(numerical, "spe",
                        lambda *a, **k: runs.append((k["multiplicities"], a[1][0].real)) or spe(*a, **k))

    def analyzed(*draws):
        ids = (first.contributions, others[0].contributions)
        split = tuple(first.revise(mean=(zeros - ones)/(zeros + ones), zeros=zeros, ones=ones, contributions=c)
                      for (zeros, ones), c in zip(draws, ids, strict=True))
        return owner._analysis(selected, result.data.observations, split + others,
                               receipts=result.data.receipts)

    short = analyzed((1, 0), (0, 3))
    assert runs[-1] == ([2], (1 - 3) / 4) and short.requested_exposure_complete is False
    assert short.exposure[0].received == 4 and short.exposure[0].requested == 6
    assert short.minimum_effective_shots_per_draw == (2, 1)
    complete = analyzed((1, 2), (0, 3))
    assert runs[-1] == ([2], (1 - 5) / 6) and complete.requested_exposure_complete is True
    # Unequal chunk populations pool their counts: (1,0) and (1,4) give -1/3,
    # whereas averaging the two chunk means would give 1/5.
    expected = Fraction(1 + 1 - 0 - 4, 6)
    assert expected == Fraction(-1, 3) and (Fraction(1, 1) + Fraction(-3, 5)) / 2 == Fraction(1, 5)
    analyzed((1, 0), (1, 4))
    assert math.isclose(runs[-1][1], float(expected), rel_tol=0.0, abs_tol=math.ulp(float(expected)))
    for bad in (((2, 2), (2, 3)),):
        with pytest.raises(ValueError, match="returned 9 shots for an admitted request of 6"):
            analyzed(*bad)
    with pytest.raises(ValueError, match="returned 0 shots"):
        owner.received_fourier_exposure(((pooled, pooled, 0, 6),), selected.shots)


def test_phase_scalar_and_interval_frame_match_original_units():
    from nwqlib.algorithms.qpe.records import QPEInterval

    result = solve(make_plan(estimator="spe", phase=True))
    for phase in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError):
            result.revise(phase_turns=phase)
    with pytest.raises(ValueError, match="phase disagree"):
        result.revise(phase_turns=0.2).validate_plan(result.plan)
    with pytest.raises(ValueError, match="interval frame"):
        result.revise(
            interval=QPEInterval(low=0., high=.5, method="explicit test interval",
                frame="energy in requested operator units", interpretation="nominal test interval")
        ).validate_plan(result.plan)
    assert energy_phase_turns(0.7, 0.2) == pytest.approx((-0.14 / (2 * pi)) % 1.0, abs=1e-15)
    with pytest.raises(ValueError, match="finite"):
        energy_phase_turns(1e308, 1e308)


def test_qcels_units_sparse_schedule_and_actual_admission(monkeypatch):
    from nwqlib.algorithms.qpe import QCELS, numerical
    from nwqlib.algorithms.qpe import method as owner

    for maximum, supplied_tau, grid in ((None, None, 64), (None, .2, 4096), (100.1, .2, 4096)):
        schedules, times, estimates = [], [], []
        for scale in (1., 1e12, 1e14):
            method = QCELS(initial_state=[0, 1], tau=None if supplied_tau is None else supplied_tau/scale,
                           max_time=None if maximum is None else maximum/scale,
                           grid_size=grid, max_work=200_000_000)
            selected = plan(Eigenproblem(A=np.diag([.4, -.7])*scale), method=method,
                            execution="classical", seed=7)
            count, _ = owner._schedule_size(method, selected.reconstruction.tau)
            schedules.append(tuple(query.power for query in selected.reconstruction.queries))
            times.append(tuple(power.evolution_time*scale for power in selected.reconstruction.powers))
            result = solve(selected)
            estimates.append(result.eigenvalue/scale)
            assert len(result.samples) == count == len(schedules[-1])
            assert result.fit.effective_grid == max(grid, 81 if maximum is None else 4001)
            assert result.interval is None
        assert schedules[0] == schedules[1] == schedules[2]
        # Compare two rounded paths through unit conversion, the enclosure,
        # tau division and power multiplication. Eight eps covers their
        # accumulated scalar roundoff without an absolute physical-unit floor.
        np.testing.assert_allclose(times, [times[0]]*3, rtol=8*np.finfo(float).eps, atol=0)
        np.testing.assert_allclose(estimates, [-.7]*3, rtol=0, atol=2e-12)
        assert len(schedules[0]) == (22 if maximum is None else 34)
    original = solve(make_plan(initial=[0, 1], num_times=16, max_time=100., grid_size=4096,
                               max_work=200_000_000))
    assert original.eigenvalue == pytest.approx(-.7, abs=2e-12, rel=0)
    assert tuple(power.power for power in original.plan.reconstruction.powers) == (
        0, *np.linspace(1, 500, 16, dtype=int))
    print(f"Sparse QCELS: {len(original.samples)} quadratures, G={original.fit.effective_grid}, "
          f"evaluations={original.fit.evaluations}, residual={original.fit.residual:.3g}, "
          f"E={original.eigenvalue:.15g}")
    # floor is discontinuous: preserve legal input on both sides, rather
    # than snapping a genuine below-boundary request into a longer schedule.
    for maximum, expected in ((np.nextafter(3., 0.), 2), (3., 3), (np.nextafter(3., 4.), 3)):
        config = QCELS(max_time=float(maximum))
        assert numerical.qcels_schedule_size(config, 1.) == (expected, expected)
    # The original sparse default is now honestly rejected before even
    # materializing its schedule. An explicit sufficient cap above permits
    # exactly that schedule and recovers its independently known eigenvalue.
    with monkeypatch.context() as patch:
        patch.setattr(numerical, "planned_power_schedule", lambda *a: pytest.fail("early schedule"))
        patch.setattr(numerical, "nominal_eigensystem", lambda *a, **k: pytest.fail("early spectrum"))
        for changes in (dict(max_work=100_000_000), dict(max_bytes=100_000),
                        dict(max_time=.1), dict(max_time=1e308)):
            method = QCELS(initial_state=[0, 1], tau=.2,
                           **(dict(max_time=100., max_work=200_000_000) | changes))
            with pytest.raises(ValueError):
                plan(Eigenproblem(A=np.diag([.4, -.7])), method=method, execution="classical", seed=7)


def test_qcels_archive_preserves_effective_grid_without_refitting(tmp_path, monkeypatch):
    from nwqlib.algorithms.qpe import method as owner, numerical
    from nwqlib.algorithms.qpe.records import QPEInterval
    from nwqlib import load_result

    result = solve(make_plan(num_times=3, max_time=4., grid_size=64))
    assert result.fit.effective_grid == 161
    assert result.analyze(grid_size=128).fit.effective_grid == 161
    with pytest.raises(ValueError, match="actual finite analysis grid"):
        result.revise(fit=result.fit.revise(effective_grid=64)).validate_plan(result.plan)
    with pytest.raises(ValueError, match="uncertainty interval"):
        result.revise(interval=QPEInterval(low=0., high=1., method="qcels_residual_slope_t",
                     frame="energy in requested operator units", interpretation="obsolete slope interval"))
    with pytest.raises(ValueError, match="analysis revision"):
        result.revise(origin=result.origin.revise(analyzer=owner._ANALYSIS_SOURCE)).validate_plan(result.plan)
    saved = result.save(tmp_path / "result")
    monkeypatch.setattr(numerical, "qcels", lambda *a, **k: pytest.fail("archive refit"))
    monkeypatch.setattr(numerical, "planned_power_schedule", lambda *a: pytest.fail("archive schedule"))
    loaded = load_result(saved)
    assert loaded.fit == result.fit and loaded.interval is None and loaded.samples == result.samples


@pytest.mark.parametrize("estimator", ("qcels", "spe", "rfe"))
def test_changed_grid_reanalyzes_without_reacquisition(monkeypatch, estimator):
    from nwqlib.algorithms.qpe import numerical
    from nwqlib.backends import qiskit_aer

    result = solve(make_plan(estimator=estimator, execution="quantum"))
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution",
                        lambda *a, **k: pytest.fail("analysis acquired a new sample"))
    monkeypatch.setattr(
        numerical,
        "nominal_eigensystem",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("new spectrum")),
    )
    if estimator == "rfe":
        # K specifies both the power distribution and Fourier frequencies.
        # Changing it cannot be treated as analysis of the same acquisition.
        for settings in (dict(num_frequencies=16), dict(num_samples=16), dict(grid_size=128)):
            with pytest.raises(ValueError, match="acquisition"):
                result.analyze(**settings)
        assert result.analyze().estimator_value == result.estimator_value
        return
    settings = dict(grid_size=128)
    revised = result.analyze(**settings)
    assert revised.data is result.data and revised.plan is result.plan
    assert revised.analysis_settings[0].parameter == "grid_size"
    with pytest.raises(ValueError, match="acquisition"):
        result.analyze(num_samples=16)


@pytest.mark.parametrize("phase", [False, True])
def test_reference_limit_precedes_reference_work(monkeypatch, phase):
    from nwqlib.algorithms.qpe import method as owner, numerical

    # A Hamiltonian quantum run already holds its eigensystem, so verification
    # would next project the reference state. A unitary run holds none, so it
    # would first diagonalize the target.
    result = solve(make_plan(execution="quantum", phase=phase))
    for module, name in ((numerical, "nominal_eigensystem"), (owner, "_reference_data")):
        monkeypatch.setattr(
            module, name, lambda *a, name=name, **k: (_ for _ in ()).throw(AssertionError(name))
        )
    with pytest.raises(ValueError, match="max_work"):
        result.verify(checks=QPEVerification(tolerance=0.1, max_work=1))


def test_static_same_seed_rejects_before_second_submission(monkeypatch):
    from nwqlib import prepare, submit
    from nwqlib.core.planning import RandomStreams

    selected = make_plan(execution="quantum", shots=11)
    monkeypatch.setattr(RandomStreams, "next_seed", lambda self: 17)
    prepared = prepare(selected)
    with pytest.raises(ValueError, match="sampling stream"):
        submit(prepared)
    assert prepared.run.trace.jobs == 1
    assert sum(c.returned_shots for c in prepared.run.observations.chunks) == 11


@pytest.mark.parametrize("uncertain", [False, True])
def test_unused_or_uncertain_count_source_rejects_before_another_submit(monkeypatch, uncertain):
    from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
    from nwqlib.core.planning import RuntimeOptions

    selected = make_plan(execution="quantum", shots=11)
    run = Run(selected)
    first = prepare_experiment(
        selected.resolve(selected.experiments[0].name), run=run, runtime=RuntimeOptions(seed=17)
    )
    original = type(run.backend).submit
    if uncertain:

        def fail(*a, **k):
            raise RuntimeError("uncertain native outcome")

        monkeypatch.setattr(type(run.backend), "submit", fail)
        with pytest.raises(RuntimeError, match="uncertain"):
            submit_experiment(first, run=run)
        monkeypatch.setattr(type(run.backend), "submit", original)
    else:
        submit_experiment(first, run=run)
    assert not run.observations.chunks
    second = prepare_experiment(
        selected.resolve(selected.experiments[1].name), run=run, runtime=RuntimeOptions(seed=17)
    )
    with pytest.raises(ValueError, match="sampling stream"):
        submit_experiment(second, run=run)
    assert len(run.trace.events) == 1 and run.trace.jobs == 1


def test_exact_probability_window_follows_the_stated_roundoff_relation():
    from nwqlib._validation import (AER_STATEVECTOR_ROUNDOFF_PER_OPERATION, NUMERICAL_RELATION_RTOL,
                                    NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION, UNIT_ROUNDOFF,
                                    exact_probability_window, exact_readout_roundoff)

    assert exact_probability_window(None, 9) == exact_probability_window(0, 0) == NUMERICAL_RELATION_RTOL
    assert exact_readout_roundoff(9) == 2**10 + 10
    deep = exact_probability_window(529_017, 9)  # the H4 QCELS power-10 circuit
    assert deep == (AER_STATEVECTOR_ROUNDOFF_PER_OPERATION * 529_017 + 2**10 + 10) * UNIT_ROUNDOFF
    assert deep > 1.2e-12  # its observed normalization gap
    assert exact_probability_window(529_018, 9) > deep
    shallow = exact_probability_window(529_017, 9, per_operation=NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION)
    assert shallow == (NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION * 529_017 + 2**10 + 10) * UNIT_ROUNDOFF
    with pytest.raises(ValueError, match="nonnegative"):
        exact_probability_window(-1, 2)


def test_roundoff_constants_cover_the_worst_instruction_of_each_derivation():
    """Each per-instruction constant must be at least the first-order worst case enumerated here
    over instruction widths and fusion cases from gamma_n ~ n*u and the gate-entry model. The
    enumeration shares the derivation's model, so it guards the constants and their arithmetic
    (for example a constant that folds only three-qubit instructions), not the model itself.
    """
    from nwqlib._validation import (AER_STATEVECTOR_ROUNDOFF_PER_OPERATION, GATE_ENTRY_ERROR,
                                    NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION, PHASE_SUM_LIMIT)

    def gamma(n):
        return n  # first order, in units of u

    def block(dimension, nonzeros):
        # dimension-term complex inner products, || |A| ||_2 <= sqrt(nonzeros)
        return np.sqrt(2) * gamma(2 * dimension) * np.sqrt(nonzeros)

    fused = 32  # Aer folds into at most five qubits
    aer_fold = max(block(2**k, 2**k) * np.sqrt(fused) for k in range(1, 6))
    aer_application = max(block(2**k, 2**k) for k in range(1, 6))
    aer_entry = 2 * (GATE_ENTRY_ERROR + 2 * PHASE_SUM_LIMIT)  # || |G| ||_2 <= 2 for parameterized gates
    aer = 2 * (aer_fold + aer_application + aer_entry)
    # NWQ-Sim: 2 x 2 products, a 4 x 4 absorbing an identity-extended 2 x 2, and 4 x 4 products.
    nwqsim_fold = max(block(2, 2) * np.sqrt(2), block(4, 4) * np.sqrt(2), block(4, 4) * 2)
    nwqsim_application = max(block(2, 2), block(4, 4))
    nwqsim_entry = GATE_ENTRY_ERROR * np.sqrt(2)  # U gate, since CX entries are exact
    nwqsim = 2 * (nwqsim_fold + nwqsim_application + nwqsim_entry)

    assert AER_STATEVECTOR_ROUNDOFF_PER_OPERATION >= aer * (1 - 1e-12)
    assert NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION >= nwqsim * (1 - 1e-12)


def test_aer_exact_preparation_lists_operations_outside_the_roundoff_derivation():
    """Supplied matrices and large phase sums are recorded as exclusions, while parameterized
    gates with reduced angles, including a diagonal rotation by a large angle, are inside.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from nwqlib.backends.qiskit_aer import _prepare_aer_execution
    from nwqlib.backends.targets import AER_STATEVECTOR_TARGET

    def exclusions(circuit, probability_qubits=(0,)):
        prepared = _prepare_aer_execution(AER_STATEVECTOR_TARGET, circuit, shots=None, seed=1,
                                          probability_qubits=probability_qubits)
        return prepared.metadata["probability_window_exclusions"]

    gates = QuantumCircuit(2)
    gates.h(0)
    gates.cx(0, 1)
    gates.rz(1.0e3, 1)
    gates.u(0.3, 2.0, -3.0, 0)
    assert exclusions(gates) == ()
    assert exclusions(gates, probability_qubits=None) == ("amplitude-derived masses",)
    supplied = gates.copy()
    supplied.append(UnitaryGate(np.eye(4)), [0, 1])
    assert exclusions(supplied) == ("unitary",)
    unreduced = QuantumCircuit(1)
    unreduced.u(0.1, 7.0, 7.0, 0)  # sum of |parameters| 14.1 > 4*pi
    assert exclusions(unreduced) == ("u phase sum",)


def test_every_aer_statevector_operation_is_classified_for_the_roundoff_window():
    """A native operation that is neither inside the derivation nor listed as an exclusion would
    make an empty exclusion tuple claim coverage it does not have."""
    from qiskit_aer import AerSimulator
    from nwqlib.backends.qiskit_aer import _PHASE_SUM_OPERATIONS, _WINDOW_EXCLUDED_OPERATIONS

    covered = {
        # Parameterized, permutation, diagonal or controlled 2 x 2 matrices on at most five qubits
        # or of any width with that structure.
        "ccx", "ccz", "cp", "crx", "cry", "crz", "cswap", "csx", "cu", "cu1", "cu2", "cu3", "cx", "cy",
        "cz", "ecr", "h", "id", "mcp", "mcphase", "mcr", "mcrx", "mcry", "mcrz", "mcswap", "mcsx",
        "mcu", "mcu1", "mcu2", "mcu3", "mcx", "mcx_gray", "mcy", "mcz", "p", "pauli", "r", "rx",
        "rxx", "ry", "ryy", "rz", "rzx", "rzz", "s", "sdg", "swap", "sx", "sxdg", "t", "tdg", "u",
        "u1", "u2", "u3", "x", "y", "z",
        # No change of the state.
        "delay", "store", "break_loop", "continue_loop", "save_amplitudes", "save_amplitudes_sq",
        "save_density_matrix", "save_expval", "save_expval_var", "save_probabilities",
        "save_probabilities_dict", "save_state", "save_statevector", "save_statevector_dict",
    }
    names = set(AerSimulator(method="statevector").target.operation_names)
    assert covered.isdisjoint(_WINDOW_EXCLUDED_OPERATIONS)
    assert names == covered | _WINDOW_EXCLUDED_OPERATIONS
    assert _PHASE_SUM_OPERATIONS <= covered


def test_receipt_window_uses_its_target_constant_and_exclusions_stay_off_host_receipts():
    from nwqlib._validation import NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION, exact_probability_window

    result = solve(make_plan(execution="quantum"))
    receipt = next(item for item in result.data.receipts if item.observation.kind == "trajectory")
    assert receipt.target.name == "aer_statevector" and receipt.probability_window_exclusions is not None
    nwqsim = receipt.revise(target=receipt.target.revise(name="nwqsim_cpu_sv"))
    assert nwqsim.probability_window == exact_probability_window(
        receipt.native_operations, len(receipt.logical_to_native),
        per_operation=NWQSIM_CPU_SV_ROUNDOFF_PER_OPERATION)
    host = solve(make_plan()).data.receipts[0]  # classical route
    assert host.execution == "host_kernel" and host.probability_window_exclusions is None
    with pytest.raises(ValueError, match="host preparation"):
        host.revise(probability_window_exclusions=())


def _with_expectations(chunk, shift):
    """A revision of trajectory point ``chunk`` whose ancilla X expectation is moved by ``shift``."""
    values = tuple(item.revise(value=item.value + shift) if item.label.endswith("X") else item
                   for item in chunk.values)
    return chunk.revise(values=values)


def test_trajectory_expectation_admits_the_roundoff_of_its_executed_circuit():
    """A deep controlled-evolution trajectory may exceed a unit expectation by binary64 roundoff
    above the fixed floor. An excess within that circuit's receipt window is kept as the raw
    value beside its endpoint, and an excess beyond it, or without the receipt, still rejects.
    """
    from nwqlib._validation import NUMERICAL_RELATION_RTOL
    from nwqlib.algorithms.qpe import QCELS
    from nwqlib.algorithms.qpe.method import _samples
    from nwqlib.execution import ObservationView
    from nwqlib.operators import ingest_pauli

    target = ingest_pauli((("XX", 0.5), ("ZI", 1.0), ("IZ", 0.3), ("YY", 0.2)), num_qubits=2)
    result = solve(Eigenproblem(A=target), method=QCELS(initial_state=[1, 0, 0, 0]), seed=7)
    (receipt,) = result.data.receipts
    window = receipt.probability_window
    assert window > 100 * NUMERICAL_RELATION_RTOL  # deep enough to separate the circuit window from the floor
    # Power 0 reads Re z_0 = 1 up to roundoff, so a shift reaches the endpoint.
    chunk = next(item for item in result.data.observations.chunks if item.point == "power_0")
    value = next(item.value for item in chunk.values if item.label.endswith("X"))
    for excess, supplied, admitted in ((window / 2, result.data.receipts, True),
                                       (2 * window, result.data.receipts, False),
                                       (window / 2, (), False)):
        changed = _with_expectations(chunk, 1 + excess - value)
        data = ObservationView(chunks=(changed,))
        if admitted:
            changed.validate_unit_bound(receipt)
            samples = _samples(result.plan, data, result.data.trace, receipts=supplied)
            real = next(sample for sample in samples if sample.phase_shift is None
                        and next(q for q in result.plan.reconstruction.queries
                                 if q.experiment == sample.experiment).phase_shift == 0)
            assert real.mean == 1.0 and real.raw_mean == 1 + excess
            for fields in (dict(raw_mean=0.0), dict(mean=0.0)):
                with pytest.raises(ValueError, match="endpoint"):
                    real.revise(**fields)
        else:
            with pytest.raises(ValueError, match="endpoint window"):
                _samples(result.plan, data, result.data.trace, receipts=supplied)


def test_excess_expectation_is_checked_against_its_circuit_window_when_created_and_saved(tmp_path, monkeypatch):
    """Trajectory expectations above one are checked against their own receipt's window where the
    observation is created and where a Result is saved. The same observation passes with its
    circuit's window, including through a reopened Run and a loaded Result, and fails with a
    smaller one.
    """
    import nwqlib
    from nwqlib import _prepared_execution as execution
    from nwqlib._validation import NUMERICAL_RELATION_RTOL
    from nwqlib.algorithms.qpe import QCELS
    from nwqlib.execution import PreparedArtifact
    from nwqlib.operators import ingest_pauli

    target = ingest_pauli((("XX", 0.5), ("ZI", 1.0), ("IZ", 0.3), ("YY", 0.2)), num_qubits=2)
    selected = plan(Eigenproblem(A=target), method=QCELS(initial_state=[1, 0, 0, 0]), seed=7)
    decode = execution._trajectory_chunks

    def excess(factor):
        """Move the power-0 ancilla X expectation to one plus factor times the receipt's window."""
        def decoded(prepared, result, **kwargs):
            chunks = decode(prepared, result, **kwargs)
            added = 1 + factor * prepared.record.probability_window
            return tuple(_with_expectations(chunk, added - next(
                item.value for item in chunk.values if item.label.endswith("X")))
                if chunk.point == "power_0" else chunk for chunk in chunks)
        return decoded

    with monkeypatch.context() as patch:
        patch.setattr(execution, "_trajectory_chunks", excess(2.0))
        with pytest.raises(ValueError):
            solve(selected)
    path = tmp_path / "run"
    with monkeypatch.context() as patch:
        patch.setattr(execution, "_trajectory_chunks", excess(0.5))
        with execution.Run(selected, directory=path) as run:
            result = run.wait()
            largest = max(item.value for chunk in run.observations.chunks for item in chunk.values)
    assert largest > 1 + 100 * NUMERICAL_RELATION_RTOL  # beyond the fixed record endpoint
    saved = result.save(tmp_path / "result")
    with nwqlib.load_run(path, backend=run.backend) as restored:
        assert restored.result.content_id == result.content_id
    assert nwqlib.load_result(saved).content_id == result.content_id
    window = PreparedArtifact.probability_window
    monkeypatch.setattr(PreparedArtifact, "probability_window", property(lambda receipt: window.fget(receipt) / 4))
    with pytest.raises(ValueError):
        result.save(tmp_path / "narrow")
    assert not (tmp_path / "narrow").exists()


@pytest.mark.parametrize("readout", ("trajectory", "counts", "host"))
def test_saved_samples_load_without_estimator_work(readout, tmp_path, monkeypatch):
    from nwqlib import load_result
    from nwqlib.algorithms.qpe import numerical
    from nwqlib.backends import qiskit_aer

    result = solve(make_plan(execution="classical" if readout == "host" else "quantum",
                             shots=7 if readout == "counts" else None))

    def forbidden(*args, **kwargs):
        pytest.fail("saving or loading performed estimator or native work")

    for name in ("qcels", "spe", "rfe", "rwpe_estimate", "nominal_eigensystem", "planned_power_schedule"):
        monkeypatch.setattr(numerical, name, forbidden)
    monkeypatch.setattr(qiskit_aer, "_prepare_aer_execution", forbidden)
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", forbidden)
    folder = result.save(tmp_path / "legal")
    loaded = load_result(folder)
    assert loaded.samples == result.samples and loaded.data.observations == result.data.observations


@pytest.mark.parametrize('family', ['expectation', 'qcels'])
def test_run_counts_index_visits_each_completed_source_once(monkeypatch, family):
    import itertools
    from types import SimpleNamespace
    import nwqlib as nw
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import qiskit_aer
    from nwqlib.operators import ingest_pauli

    totals = []
    for count in (4, 8):
        if family == 'expectation':
            # Two distinct labels without I differ on a site where both act,
            # so each forms its own qubit-wise commuting group and setting.
            labels = [''.join(word) for word in itertools.product('XYZ', repeat=2)]
            problem = nw.Expectation(state=[1,0,0,0], observable=ingest_pauli(
                [(label,1/count) for label in labels[:count]], num_qubits=2))
            choice = nw.plan(problem, method=ExpectationMethod(), accuracy=nw.Accuracy(
                component='sampling', absolute_tolerance=1., confidence=.9), seed=1)
        else:
            from nwqlib.algorithms.qpe import QCELS
            choice = nw.plan(nw.Eigenproblem(A=np.diag([-.2,.3])), method=QCELS(
                initial_state=[1,0], tau=1., num_times=count//2-1, max_time=3., grid_size=32),
                shots=10, seed=1)
        visits, calls = [], []
        original = CountsSources.require_independent
        def joined(self, chunk):
            if self is prepared.run._state['counts_sources']:
                visits.append(chunk.acquisition_key)
            return original(self, chunk)
        def acquire(native):
            calls.append(1)
            key = '00' if family == 'expectation' else '0'  # grouped readout measures both qubits
            return SimpleNamespace(raw_output={'counts': {key: choice.shots}},
                                   metadata={'native_job_id': str(len(calls))})
        with monkeypatch.context() as patch:
            patch.setattr(qiskit_aer, '_submit_aer_execution', acquire)
            prepared = nw.prepare(choice)
            patch.setattr(CountsSources, 'require_independent', joined)
            with prepared.run:
                result = nw.submit(prepared).wait(timeout=0)
                assert len(result.data.observations.chunks) == count
                assert len(visits) == len(set(visits)) == count
                assert len(calls) == count
                totals.append(len(visits))
    assert totals == [4,8]  # One source join per actual acquisition, not N(N-1)/2.


@pytest.mark.parametrize('failed_transition', ['reserved', 'completed'])
def test_counts_index_follows_atomic_reservation_and_rebuilds_once(tmp_path, monkeypatch, failed_transition):
    import sqlite3
    from types import SimpleNamespace
    import nwqlib as nw
    from nwqlib.backends import qiskit_aer
    from nwqlib.core.planning import RuntimeOptions
    from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment, restore_prepared

    choice = make_plan(execution='quantum', shots=11)
    calls = []
    def acquire(native):
        calls.append(1)
        return SimpleNamespace(raw_output={'counts': {'0': 11}},
                               metadata={'native_job_id': str(len(calls))})
    monkeypatch.setattr(qiskit_aer, '_submit_aer_execution', acquire)
    run = Run(choice, directory=tmp_path/'atomic')
    backend = run.backend
    first = prepare_experiment(choice.resolve(choice.experiments[0].name), run=run,
                               runtime=RuntimeOptions(seed=17))
    journal = run._state['journal']
    journal.connection.execute("CREATE TRIGGER reject_transition BEFORE INSERT ON records "
        f"WHEN NEW.kind='event' AND json_extract(NEW.payload,'$.status')='{failed_transition}' "
        "BEGIN SELECT RAISE(ABORT, 'count storage failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='count storage failure'):
        submit_experiment(first, run=run)
    sources = run._state['counts_sources']
    assert not sources.counted
    assert bool(sources.streams) == (failed_transition == 'completed')
    assert len(calls) == int(failed_transition == 'completed')
    journal.connection.execute('DROP TRIGGER reject_transition')
    run.close()
    with nw.load_run(tmp_path/'atomic', backend=backend) as restored:
        assert restored._state['counts_sources'] is None
        handle = restored._state['handles'].get(first.record.content_id) or restore_prepared(
            first.record.content_id, run=restored, for_submission=True)
        if failed_transition == 'reserved':
            # No reservation committed, so using the original prepared stream is legal.
            chunk = submit_experiment(handle, run=restored)
            assert len(calls) == 1 and not restored.observations.chunks
            assert restored._state['counts_sources'].counted
            restored._state['counts_sources'].require_independent(chunk)  # Same acquisition reread.
        else:
            assert restored.trace.events[0].status == 'uncertain'
            with pytest.raises(ValueError, match='sampling stream'):
                submit_experiment(handle, run=restored)
            assert len(calls) == 1
    # The successful unused acquisition is reconstructed exactly once, even
    # though two rejected submission attempts both consult the restored index.
    with nw.load_run(tmp_path/'atomic', backend=backend) as restored:
        second = prepare_experiment(restored.plan.resolve(restored.plan.experiments[1].name),
                                    run=restored, runtime=RuntimeOptions(seed=17))
        joins = []
        original = CountsSources.require_independent
        def joined(self, chunk):
            if self is restored._state['counts_sources']:
                joins.append(chunk.acquisition_key)
            return original(self, chunk)
        with monkeypatch.context() as patch:
            patch.setattr(CountsSources, 'require_independent', joined)
            for _ in range(2):
                with pytest.raises(ValueError, match='sampling stream'):
                    submit_experiment(second, run=restored)
        assert len(joins) == int(failed_transition == 'reserved')
        assert len(calls) == 1


def test_unitary_input_selects_and_binds_one_polar_base(tmp_path):
    """A unitary-input Plan names its polar base V and binds every spectral consumer to it.

    The reconstruction records V = polar(A) of the admitted near-unitary A,
    the saved Result binds it on load, and both the classical route and
    verification refuse a spectral context computed for A itself, because a
    Schur decomposition of A can describe a different target.
    """
    from types import SimpleNamespace
    from scipy.linalg import expm, polar
    from nwqlib.algorithms.qpe.verification import verify
    from nwqlib.algorithms.qpe import QPEVerification
    from nwqlib.algorithms.qpe.method import _eigensystem, _new_context, _spectrum
    from nwqlib.saved_evidence import load_result

    rng = np.random.default_rng(31)
    raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    defective = expm(-1j * (raw + raw.conj().T)) @ np.diag([1 + 4e-9, 1 - 4e-9])
    selected = make_plan(phase=True, target=defective)
    rec, data = selected.reconstruction, selected._native["input"]
    assert rec.selected_base == data.base.reference and rec.selected_base != rec.target
    assert np.abs(data.base.dense_array() - polar(defective)[0]).max() <= 2e-12
    foreign = _new_context()
    _eigensystem(rec.target, defective, kind="unitary", context=foreign)
    with pytest.raises(ValueError, match="another target"):
        _spectrum(data, foreign)
    result = solve(selected)

    class Foreign:
        """The Result with a spectral context computed for A instead of its own."""
        data = SimpleNamespace(method_context=foreign)

        def __getattr__(self, name):
            return getattr(result, name)

    with pytest.raises(ValueError, match="another target"):
        verify(selected, Foreign(), checks=QPEVerification(tolerance=1.0))
    result.verify(checks=QPEVerification(tolerance=1.0))
    loaded = load_result(result.save(tmp_path / "base"))
    assert loaded.plan._native["input"].base.reference == rec.selected_base
    assert loaded.plan.reconstruction == rec


def test_saved_exposure_facts_must_stay_within_their_request(tmp_path):
    """Loading refuses stored exposure facts that exceed the request.

    Every count query must have received between one shot and its requested
    shots times multiplicity. The refusal names the query; the untouched
    archive loads.
    """
    import json
    from nwqlib.saved_evidence import load_result

    result = solve(make_plan(estimator="spe", execution="quantum", shots=3, num_samples=2))
    first = result.exposure[0]
    assert (first.received, first.requested, result.requested_exposure_complete) == (6, 6, True)
    folder = result.save(tmp_path / "legal")
    assert load_result(folder).exposure == result.exposure
    path = folder / "result.json"
    original = path.read_text()
    saved = json.loads(original)
    saved["result"] = result.revise(exposure=(first.revise(received=7), *result.exposure[1:])).model_dump(mode="json")
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="7 received shots for an admitted request of 6") as refused:
        load_result(folder)
    assert first.experiment in str(refused.value)
    path.write_text(original)
    assert load_result(folder).exposure == result.exposure
