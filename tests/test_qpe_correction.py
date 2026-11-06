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


def _cluster_facts(energies, **options):
    """Verify one diagonal Hamiltonian at tau = 1/spread and return its record's facts by name."""
    from nwqlib.algorithms.qpe import QCELS

    spread = max(energies) - min(energies) or 1.0
    state = np.full(len(energies), 1 / np.sqrt(len(energies)))
    result = solve(Eigenproblem(A=np.diag(energies)), execution="classical", seed=7,
                   method=QCELS(initial_state=state, tau=1 / spread, num_times=4, max_time=4 / spread, grid_size=64))
    receipt, _ = result.verify(checks=QPEVerification(tolerance=1e30, **options))
    return {item.fact.quantity: item.fact for item in receipt.applications[0].facts}


@pytest.mark.parametrize("scale, offset", [(1.0, 0.0), (1e-13, 0.0), (1e7, 0.0), (1.0, 5.0), (1e-13, 2e-13)])
def test_default_clusters_follow_the_units_of_the_hamiltonian_and_ignore_an_offset(scale, offset):
    """For E' = lambda E + c the span scales with lambda and ignores c, so the partition of a separated spectrum stays.

    The pair differs by 1e-10 lambda, below a/10 with a = 1e-8 span = 1.8e-8 lambda, and the other gaps are above
    10 a, so rounding of a diagonal input cannot move a value across the window (``verification._group_tolerance``).
    A fixed window of 1e-8 merges this whole spectrum at scale 1e-13 and splits the pair at scale 1e7.
    """
    facts = _cluster_facts([scale * e + offset for e in (-1.3, -1.3 + 1e-10, -.4, .5)])
    assert facts["cluster_count"].value.value == 3
    assert facts["cluster_dimension"].value.value == 2
    assert facts["group_tolerance"].value.value == pytest.approx(1.8e-8 * scale, rel=1e-12, abs=0)
    assert facts["estimated_resolved"].value.value == 1
    assert facts["estimated_resolved"].assumptions == ()


def test_explicit_cluster_tolerance_and_the_rounding_estimate_keep_their_own_meaning():
    # An explicit tolerance is used as given: 1e-11 separates the pair that the default keeps together.
    facts = _cluster_facts([-1.3, -1.3 + 1e-10, -.4, .5], group_atol=1e-11)
    assert facts["cluster_count"].value.value == 4 and facts["group_tolerance"].value.value == 1e-11
    # A single-valued spectrum has tolerance 0, one cluster and no gap for rounding to split.
    facts = _cluster_facts([.7, .7])
    assert facts["cluster_count"].value.value == 1 and facts["group_tolerance"].value.value == 0
    assert facts["estimated_resolved"].value.value == 1
    # max|E|/span = 3.3e8 puts a = 1e-8 span below D u max|E| = 4 * 2**-53 * 1e9. The flag warns and names the
    # remedy, and the clusters are still reported as formed.
    facts = _cluster_facts([1e9, 1e9 + 1, 1e9 + 2, 1e9 + 3])
    assert facts["group_tolerance"].value.value == pytest.approx(3e-8, rel=1e-12, abs=0)
    assert facts["cluster_count"].value.value == 4 and facts["estimated_resolved"].value.value == 0
    assert "group_atol" in facts["estimated_resolved"].assumptions[0]


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


def test_classical_eigendecomposition_is_charged_eighty_matrix_bytes():
    """Classical QPE declares 80*D**2 bytes for the eigendecomposition, which verification then reuses.

    For the 2-dimensional Hamiltonian of make_plan the host routine's
    workspace is 80*4 bytes. A verification of its Result reuses that
    eigensystem and counts only the QR of its two one-vector clusters,
    48*2 bytes at a time, so it is refused at half that and runs at it.
    """
    selected = make_plan(execution="classical")
    (kernel,) = selected.construction.kernels
    assert [item.bytes for item in kernel.workspace] == [80 * 4]
    result = solve(selected)
    with pytest.raises(ValueError, match="cluster QR .* needs 96 bytes. Raise QPEVerification.max_bytes to at least 96"):
        result.verify(checks=QPEVerification(tolerance=1.0, max_bytes=48))
    receipt, _ = result.verify(checks=QPEVerification(tolerance=1.0, max_bytes=96))
    assert receipt.applications


def test_verification_counts_the_eigendecomposition_only_when_computed_and_the_qr_by_cluster(monkeypatch):
    """QPEVerification counts the QR of each cluster at its size, and an eigendecomposition only when it runs one.

    For diag(-0.65, -0.65, 0.55, 0.7) the clusters have 2, 1 and 1
    eigenvectors, so reusing the stored spectrum counts
    qr_work(4, 2) + 2*qr_work(4, 1) + 16*4*4 = 432 units and 48*4*2 bytes
    for three QR calls. With the eigensystem removed from the context, the
    eigendecomposition adds hermitian_eigensystem_work(4) = 1024 units,
    refused before it runs. Each refusal names only the exceeded limit.
    """
    from nwqlib._linalg_laws import hermitian_eigensystem_work, qr_work
    from nwqlib.algorithms.qpe import SPE, numerical
    from nwqlib.algorithms.qpe import method as owner, verification

    result = solve(Eigenproblem(A=np.diag([-.65, -.65, .55, .7])), execution="classical",
        method=SPE(initial_state=[np.sqrt(.15), np.sqrt(.15), np.sqrt(.35), np.sqrt(.35)], tau=1.,
                   overlap_lower_bound=.3, num_samples=8, fourier_degree=4, grid_size=64), seed=7)
    law = qr_work(4, 2) + 2 * qr_work(4, 1) + 16 * 4 * 4
    assert law == 432 and hermitian_eigensystem_work(4) == 1024
    calls = []
    qr, eigensystem = np.linalg.qr, numerical.nominal_eigensystem
    monkeypatch.setattr(np.linalg, "qr", lambda *a, **k: calls.append("qr") or qr(*a, **k))
    monkeypatch.setattr(numerical, "nominal_eigensystem",
                        lambda *a, **k: calls.append("eigensystem") or eigensystem(*a, **k))

    def verify(**limits):
        calls.clear()
        receipt, _ = result.verify(checks=QPEVerification(tolerance=.1, **limits))
        return {item.parameter: item.value for item in receipt.applications[0].arguments}

    with pytest.raises(ValueError, match="cluster QR .* 432 work units. Raise QPEVerification.max_work to at least 432"):
        verify(reuse_only=True, max_work=law // 2)
    assert calls == []
    assert verify(reuse_only=True, max_work=law)["known_work"] == law and calls == ["qr"] * 3
    with pytest.raises(ValueError, match="needs 384 bytes. Raise QPEVerification.max_bytes") as refused:
        verify(reuse_only=True, max_bytes=192)
    assert "max_work" not in str(refused.value)
    monkeypatch.setattr(verification, "memory_limit", lambda value: 100 if value is None else value)
    with pytest.raises(ValueError, match="384 bytes, above the physical memory 100 of this computer") as refused:
        verify(reuse_only=True)
    assert "max_work" not in str(refused.value) and "Raise" not in str(refused.value)
    monkeypatch.undo()
    check = owner._check_context

    def without_eigensystem(context, *args):
        check(context, *args)
        context["eigensystem"] = None

    monkeypatch.setattr(owner, "_check_context", without_eigensystem)
    monkeypatch.setattr(numerical, "nominal_eigensystem",
                        lambda *a, **k: calls.append("eigensystem") or eigensystem(*a, **k))
    with pytest.raises(ValueError, match="eigendecomposition .* 1024 work units"):
        verify(max_work=512)
    assert calls == []
    assert verify(max_work=1024)["known_work"] == 1024 + law and calls[0] == "eigensystem"


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


@pytest.mark.parametrize("seed", range(4))
def test_unitary_clusters_follow_the_first_member_rule_across_the_branch_cut(seed):
    """Unitary eigenvalues group as the direct rule groups them, including a cluster across +-pi.

    The direct rule scans the phase-ordered values and gives each unassigned
    value a cluster of every unassigned value within atol of it, in phase
    order. Clusters near pi and -pi, values just outside atol and repeated
    phases are included.
    """
    from nwqlib.algorithms.qpe import numerical
    from nwqlib.algorithms.qpe.verification import _groups

    rng = np.random.default_rng(seed)
    atol = 1e-3
    centers = np.concatenate([[np.pi - 2e-4, -np.pi + 3e-4, 0.5, 0.5 + 1.5 * atol], rng.uniform(-np.pi, np.pi, 6)])
    phases = np.concatenate([center + rng.uniform(-0.6 * atol, 0.6 * atol, rng.integers(1, 4)) for center in centers])
    phases = np.concatenate([phases, phases[:3]])
    values = np.exp(1j * phases)
    angles = np.angle(values)
    ordered = sorted(range(len(angles)), key=lambda index: float(angles[index]))
    assigned, expected = set(), []
    for index in ordered:
        if index not in assigned:
            group = [c for c in ordered if c not in assigned
                     and abs(numerical.angle_principal(float(angles[c] - angles[index]))) <= atol]
            assigned.update(group)
            expected.append(group)
    groups = _groups(values, tau=None, atol=atol)
    assert groups == expected
    assert any(min(angles[g]) < -3 and max(angles[g]) > 3 for g in groups)


def test_unitary_eigendecomposition_is_counted_as_a_complex_schur_decomposition(monkeypatch):
    """A unitary target counts complex_schur_work, 13*8 + 32*4 = 232 units for D = 2, where a Hamiltonian counts 192.

    Classical planning refuses half of the Schur count plus the projection,
    296 = 232 + 16*4. A verification that diagonalizes again is refused at
    116 before the Schur decomposition and passes at 232, 336 in
    all with its cluster QR. Reusing the stored spectrum counts the 104 of
    the QR alone.
    """
    from nwqlib.algorithms.qpe import QCELS, method as owner, numerical
    from nwqlib.operators import ingest_dense
    from nwqlib.problems import Eigenphase, SpectralEstimation

    assert numerical.nominal_eigensystem_work(2, kind="unitary") == 232
    assert numerical.nominal_eigensystem_work(2, kind="hamiltonian") == 192
    theta = np.array([.25, .7])
    basis = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    problem = SpectralEstimation(unitary=ingest_dense((basis * np.exp(1j * theta)) @ basis.conj().T),
                                 initial_state=basis @ np.sqrt([.75, .25]))

    def method(**limits):
        return QCELS(num_times=4, max_time=8, **limits)

    with pytest.raises(ValueError, match="needs 296 work units"):
        plan(problem, method=method(max_work=148), output=Eigenphase(), execution="classical", seed=7)
    result = solve(problem, method=method(), output=Eigenphase(), execution="classical", seed=7)
    schur, decompose = [], numerical.schur
    monkeypatch.setattr(numerical, "schur", lambda *a, **k: schur.append(1) or decompose(*a, **k))

    def verify(**options):
        receipt, _ = result.verify(checks=QPEVerification(tolerance=.2, **options))
        application = receipt.applications[0]
        return ({item.fact.quantity: item.fact.value.value for item in application.facts},
                {item.parameter: item.value for item in application.arguments})

    reused, saved = verify(reuse_only=True)
    assert saved["known_work"] == 104 and schur == []
    # A unitary target clusters by angle at the default 1e-8 radians, whatever its output unit.
    assert reused["group_tolerance"] == 1e-8 and reused["estimated_resolved"] == 1
    check = owner._check_context

    def without_eigensystem(context, *args):
        check(context, *args)
        context["eigensystem"] = None

    monkeypatch.setattr(owner, "_check_context", without_eigensystem)
    with pytest.raises(ValueError, match="needs 232 work units"):
        verify(max_work=116)
    assert schur == []
    fresh, computed = verify(max_work=232)
    assert computed["known_work"] == 336 and schur == [1]
    assert fresh["reference_value"] == pytest.approx(reused["reference_value"], rel=0, abs=1e-14)
    assert fresh["valid_overlap"] == pytest.approx(.75, rel=0, abs=1e-14)
