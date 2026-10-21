"""Selected QPE meaning, legal alternatives, count identity and immutable evidence."""

import numpy as np
import pytest
from nwqlib import Eigenproblem, solve, plan
from nwqlib.algorithms.qpe import QPEVerification
from test_qpe_selected import make_plan


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


def test_changed_grid_reanalyzes_without_reacquisition(monkeypatch):
    from nwqlib.algorithms.qpe import numerical
    from nwqlib.backends import qiskit_aer

    result = solve(make_plan(estimator="qcels", execution="quantum"))
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution",
                        lambda *a, **k: pytest.fail("analysis acquired a new sample"))
    monkeypatch.setattr(
        numerical,
        "nominal_eigensystem",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("new spectrum")),
    )
    settings = dict(grid_size=128)
    revised = result.analyze(**settings)
    assert revised.data is result.data and revised.plan is result.plan
    assert revised.analysis_settings[0].parameter == "grid_size"
    with pytest.raises(ValueError, match="acquisition"):
        result.analyze(num_samples=16)


@pytest.mark.parametrize("phase", [True])
def test_reference_limit_precedes_reference_work(monkeypatch, phase):
    from nwqlib.algorithms.qpe import method as owner, numerical

    # A unitary quantum run holds no eigensystem, so verification would first
    # diagonalize the target.
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


def _with_expectations(chunk, shift):
    """A revision of trajectory point ``chunk`` whose ancilla X expectation is moved by ``shift``."""
    values = tuple(item.revise(value=item.value + shift) if item.label.endswith("X") else item
                   for item in chunk.values)
    return chunk.revise(values=values)


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


@pytest.mark.parametrize("readout", ("counts", "host"))
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
