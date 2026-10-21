"""Natural QPE targets, selected native bodies and saved controller continuation."""

from math import pi
from unittest.mock import Mock
import numpy as np
import pytest
from nwqlib import Eigenproblem, SpectralEstimation, plan, solve, prepare, submit, load_run
from nwqlib.algorithms.qpe import QCELS, SPE, RFE, RWPE
from nwqlib.algorithms.qpe import method as owner, numerical
from nwqlib._choice_archive import ArchiveFiles, save_plan, load_plan
from nwqlib.saved_evidence import load_result


def test_unverified_energy_domain_is_visible_without_changing_observations(tmp_path):
    # H*[3,1]/sqrt(10) = .5*[3,1]/sqrt(10). tau=8 aliases the energy,
    # while the measured modulo-one phase remains meaningful.
    problem = Eigenproblem(A=[[.4, .3], [.3, -.4]])
    fields = dict(num_samples=32, num_frequencies=128)
    method = RFE(initial_state=[3, 1], tau=8, **fields)
    # RFE draws its 32 powers from the seeded Method stream at planning, so
    # its accuracy assertions below check the estimate on these fixed draws,
    # not the estimator's failure probability. Theorem 2.1 of Kshirsagar,
    # Katabarwa and Johnson arXiv:2209.11322v3 (numerical.rfe) needs over
    # 1100 draws for such a bound, and some other draws of 32 powers put the
    # largest coefficient more than one Fourier cell from the true phase.
    result = solve(problem, method=method, execution="classical", seed=7)
    assert not result.plan.reconstruction.aliasing_bound_sufficient
    assert result.interval is None
    expected = (-4 / (2*pi)) % 1
    assert min(abs(result.phase-expected), 1-abs(result.phase-expected)) <= 1/128
    assert "energy branch is unverified" in str(result)
    loaded = load_result(result.save(tmp_path / "RFE"))
    assert loaded.estimator_value == result.estimator_value
    assert loaded.data.observations == result.data.observations
    assert loaded.interval == result.interval
    assert str(loaded) == str(result)
    legal = solve(problem, method=RFE(initial_state=[3, 1], tau=1, **fields),
                  execution="classical", seed=7)
    assert legal.plan.reconstruction.aliasing_bound_sufficient
    assert "domain is unverified" not in str(legal)
    # One Fourier cell 2*pi/128 at tau=1, on the seeded draws above.
    assert legal.eigenvalue == pytest.approx(.5, rel=0, abs=2*pi / 128)


def test_loose_pauli_enclosure_does_not_reject_a_legal_prepared_population():
    from nwqlib.operators import ingest_pauli
    # The available Pauli-L1 enclosure is .7, but the actual prepared
    # eigenvalue is .5. tau=6 gives 4.2>pi for the bound and 3<pi for E.
    target = ingest_pauli((("Z", .4), ("X", .3)), num_qubits=1)
    result = solve(Eigenproblem(A=target),
                   method=QCELS(initial_state=[3, 1], tau=6, num_times=2,
                                max_time=6, grid_size=128), execution="quantum")
    assert not result.plan.reconstruction.aliasing_bound_sufficient
    assert result.complete and "energy branch is unverified" in str(result)
    # A phase perturbation <= power error maps to energy perturbation / tau;
    # the 1e-3 selected power budget leaves ample separation from a branch jump.
    assert result.eigenvalue == pytest.approx(.5, rel=0, abs=1e-3)


def make_plan(
    *,
    estimator="qcels",
    execution="classical",
    phase=False,
    target=None,
    initial=None,
    shots=None,
    seed=7,
    **settings,
):
    initial = [1.0, 0.0] if initial is None else initial
    matrix = np.diag(np.exp(1j * np.array([0.4, -0.7]))) if phase else np.diag([0.4, -0.7])
    matrix = matrix if target is None else target
    p = (
        (SpectralEstimation(hamiltonian=np.diag([.4, -.7]), initial_state=initial)
         if estimator == "rwpe" else SpectralEstimation(unitary=matrix, initial_state=initial))
        if phase
        else Eigenproblem(A=matrix)
    )
    method_type, fields = {
        "qcels": (QCELS, dict(num_times=3, max_time=3.0 if phase else 0.6, grid_size=64)),
        "spe": (SPE, dict(fourier_degree=1, overlap_lower_bound=1., num_samples=8, grid_size=64)),
        "rfe": (RFE, dict(num_samples=7, num_frequencies=8)),
        "rwpe": (RWPE, dict(max_steps=3)),
    }[estimator]
    fields.update(tau=None if phase else 0.2)
    fields.update(settings)
    m = method_type(initial_state=None if phase else initial, **fields)
    return plan(p, method=m, execution=execution, shots=shots, seed=seed)


@pytest.mark.parametrize("estimator", ["qcels", "spe", "rfe", "rwpe"])
def test_selected_estimator_actual_shared_consumer(estimator):
    selected = make_plan(estimator=estimator, execution="quantum")
    assert sum(r.width for r in selected.construction.program.registers) <= 2
    result = solve(selected)
    if estimator == "rwpe":
        shrink = np.sqrt((np.e-1)/np.e)
        data = [sample.ones for sample in result.samples]
        expected = selected.method.prior_mean + selected.method.prior_std/np.sqrt(np.e)*sum(
            (1-2*datum)*shrink**step for step, datum in enumerate(data))
        assert result.gaussian.mean == pytest.approx(expected, rel=0, abs=2e-15)
        assert result.gaussian.standard_deviation == pytest.approx(selected.method.prior_std*shrink**3,
                                                                  rel=2e-15, abs=0)
        assert result.eigenvalue == pytest.approx(-expected/selected.reconstruction.tau, rel=0, abs=2e-14)
        assert len(result.data.observations.chunks) == 3
        assert result.complete and result.analyze().gaussian == result.gaussian
        assert result.analyze().stop_reason == result.stop_reason == "original RWPE step count completed"
        return
    # Single pure eigenstate has E=.4; grid estimators resolve finite cells.
    cells = selected.method.num_frequencies if estimator == "rfe" else 64
    tolerance = 1e-12 if estimator == "qcels" else 2 * pi / (0.2 * cells)
    assert abs(result.eigenvalue - 0.4) <= tolerance
    assert result.phase == pytest.approx((-result.eigenvalue * 0.2 / (2 * pi)) % 1.0, abs=1e-15)
    assert result.complete and result.plan is selected
    assert result.analyze().estimator_value == result.estimator_value


def test_original_nonpower_coordinates_and_hamiltonian_phase_output():
    p = Eigenproblem(A=np.diag([0.2, 0.4, 0.7]))
    selected = plan(p, method=QCELS(initial_state=[0, 1, 0]), seed=7)
    assert selected.reconstruction.preparation.basis.dimension == 4
    assert selected._native["input"].target.dense_array()[3, 3] == pytest.approx(1.3 / 3, abs=1e-15)
    result = solve(selected)
    assert result.eigenvalue == pytest.approx(0.4, abs=2e-14)
    spectral = SpectralEstimation(hamiltonian=np.diag([0.2, 0.7]), initial_state=[1, 0])
    phase = solve(spectral, method=QCELS(), seed=7)
    assert phase.value == phase.phase
    assert phase.eigenvalue == pytest.approx(0.2, abs=2e-14)
    assert phase.interval is None
    assert "uncertainty interval unavailable" in str(phase)


@pytest.mark.parametrize("estimator", ["spe"])
@pytest.mark.parametrize("execution", ["quantum"])
def test_archive_keeps_selected_schedule_without_numerical_reselection(
    tmp_path, monkeypatch, estimator, execution
):
    selected = make_plan(estimator=estimator, execution=execution)
    data = save_plan(selected, ArchiveFiles(tmp_path, 4_000_000))

    def forbidden(*a, **k):
        raise AssertionError("load repeated scientific selection")

    monkeypatch.setattr(type(selected.method), "plan", forbidden)
    monkeypatch.setattr(numerical, "planned_power_schedule", forbidden)
    monkeypatch.setattr(numerical, "nominal_eigensystem", forbidden)
    monkeypatch.setattr(owner, "_safe_time", forbidden)
    restored = load_plan(data, ArchiveFiles(tmp_path, 4_000_000))
    assert restored == selected
    assert type(restored.method) is type(selected.method)
    assert [b.record for b in restored.blocks] == [b.record for b in selected.blocks]
    assert restored.method.initial_state.reference == selected.method.initial_state.reference


@pytest.mark.parametrize("method_type,acquisitions,shots,per_draw", (
    (QCELS, 1, 0, None),
    (SPE, None, 2*32*5, 5)))
def test_concrete_default_methods_use_actual_nonzero_phase_acquisitions(
        method_type, acquisitions, shots, per_draw, monkeypatch, tmp_path):
    from nwqlib.algorithms.registry import AlgorithmRegistry, builtin_registrations
    from nwqlib.backends import qiskit_aer

    registration = next(row for row in builtin_registrations() if row.method_type is method_type)
    premises = dict(overlap_lower_bound=1.) if method_type is SPE else {}
    method = AlgorithmRegistry((registration,)).resolve(registration.source, initial_state=[0, 1], **premises)
    selected = plan(Eigenproblem(A=np.diag([.25, -.5])), method=method, shots=per_draw, seed=17)
    assert sum(register.width for register in selected.construction.program.registers) == 2
    calls, setups = [], []
    original = qiskit_aer._submit_aer_execution
    eigensystem = numerical.nominal_eigensystem

    def observed(prepared, **kwargs):
        calls.append((prepared.circuit.num_qubits, prepared.shots))
        return original(prepared, **kwargs)

    def native_setup(*args, **kwargs):
        setups.append(1)
        return eigensystem(*args, **kwargs)

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", observed)
    monkeypatch.setattr(numerical, "nominal_eigensystem", native_setup)
    monkeypatch.setattr(owner, "_reference_data", lambda *a, **k: pytest.fail("default projected a reference"))
    monkeypatch.setattr(numerical, "spectral_weights", lambda *a, **k: pytest.fail("default used a reference signal"))
    weights = []
    if method_type is SPE:
        estimator = getattr(numerical, method_type.estimator)
        monkeypatch.setattr(numerical, method_type.estimator,
                            lambda *a, **k: weights.append(list(k["multiplicities"])) or estimator(*a, **k))
    result = solve(selected)
    if method_type is SPE:
        # One query per distinct drawn (power, quadrature). The
        # multiplicities keep all 2M draws, and a counts batch requests
        # shots*multiplicity fresh shots, so the shot total stays 2*M*shots.
        queries = selected.reconstruction.queries
        assert len(queries) == len({(q.power, q.phase_shift) for q in queries}) < 2*method.num_samples
        acquisitions = len(queries)
        assert sum(q.multiplicity for q in queries) == 2*method.num_samples
        # The estimator weights each pooled point by its draws.
        assert weights == [[q.multiplicity for q in queries[::2]]]
        nodes = {d.id: d.node for d in selected.construction.program.definitions}
        requested = [nodes[e.batch].repetitions for e in selected.experiments]
        assert requested == [per_draw*q.multiplicity for q in queries]
        assert sorted(event.shots for event in result.data.trace.events) == sorted(requested)
    assert len(calls) == len(result.data.trace.events) == len(selected.experiments) == acquisitions
    if per_draw is None:
        # The exact static route observes every distinct power on one
        # controlled trajectory: one Experiment, one Aer submission, and each
        # power's X and Y queries share its one point chunk.
        points = {q.point for q in selected.reconstruction.queries}
        assert len(points) == len({q.power for q in selected.reconstruction.queries})
        assert {chunk.point for chunk in result.data.observations.chunks} == points
    assert sum(count or 0 for _, count in calls) == sum(event.shots for event in result.data.trace.events) == shots
    assert all(width == 2 for width, _ in calls)
    assert setups == [1]  # One actual dense-power synthesis setup, reused for every power.
    assert type(result.plan.method) is method_type and result.estimator == registration.source.name
    assert result.complete
    if per_draw is None:
        # The Pauli form of the same target also runs one trajectory.
        from nwqlib.operators import ingest_pauli
        pauli = plan(Eigenproblem(A=ingest_pauli((("I", -.125), ("Z", .375)), num_qubits=1)),
                     method=method, seed=17)
        calls.clear()
        pauli_result = solve(pauli)
        assert len(calls) == len(pauli.experiments) == len(pauli_result.data.trace.events) == 1
        assert pauli_result.complete
    tau = .9 * pi / ((3 if method_type is SPE else 1) * .5)
    assert selected.reconstruction.tau == tau
    if per_draw is None:
        # Exact readout only: five shots per draw carry sampling error
        # that this tolerance does not bound.
        # A seeded single eigenstate witness, not a mixed-spectrum or coverage claim.
        assert abs(result.eigenvalue + .5) <= 2e-13
    assert result.phase == pytest.approx((-tau * result.eigenvalue / (2 * pi)) % 1, abs=2e-15)
    if method_type is QCELS:
        assert result.fit.effective_grid == 4096 and result.interval is None
        restored = load_result(result.save(tmp_path / "default"))
        assert restored.samples == result.samples == restored.analyze().samples
        assert restored.data.observations == result.data.observations
        assert restored.report()["result"]["samples"] == result.report()["result"]["samples"]
        assert len(calls) == acquisitions and setups == [1]
    print(f"{method_type.__name__}: {len(calls)} acquisitions, {shots} shots, E={result.eigenvalue:.12g}, phase={result.phase:.12g}")


def test_rwpe_completed_pending_resume_has_same_trajectory(tmp_path, monkeypatch):
    selected = make_plan(estimator="rwpe", execution="quantum")
    whole = solve(selected)
    prepared = prepare(selected)
    update = numerical.rwpe_update
    monkeypatch.setattr(
        numerical,
        "rwpe_update",
        Mock(side_effect=RuntimeError("interrupted before posterior update")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        submit(prepared)
    run = prepared.run
    assert len(run.observations.chunks) == 1
    assert run.checkpoint_state["processed"] == 0
    saved = run.save(tmp_path / "run")
    monkeypatch.setattr(numerical, "rwpe_update", update)
    restored = load_run(saved, backend=run.backend)
    result = restored.resume().wait()
    assert len(result.data.observations.chunks) == 3
    assert [(s.power, s.phase_shift, s.mean) for s in result.samples] == [
        (s.power, s.phase_shift, s.mean) for s in whole.samples
    ]
    assert result.gaussian == whole.gaussian
    assert [r.runtime.seed for r in result.data.receipts] == [
        r.runtime.seed for r in whole.data.receipts
    ]
    assert result.controller_work > whole.controller_work


def test_native_random_pauli_quadratures_share_each_power():
    from nwqlib.operators import ingest_pauli

    target = ingest_pauli((("I", 0.2), ("Z", 0.3), ("X", 0.4)), num_qubits=1)
    selected = make_plan(
        estimator="rfe",
        target=target,
        execution="quantum",
        num_frequencies=3,
        controlled_power_error_budget=1e-4,
    )
    assert sum(r.width for r in selected.construction.program.registers) == 2
    result = solve(selected)
    for sample in result.samples:
        time = sample.power * 0.2
        overlap = np.exp(-1j * 0.2 * time) * (np.cos(0.5 * time) - 0.6j * np.sin(0.5 * time))
        # The trajectory reads each logical quadrature from the ancilla X or Y.
        quadrature = next(q.phase_shift for q in selected.reconstruction.queries
                          if q.experiment == sample.experiment)
        expected = float(np.real(np.exp(1j * quadrature) * overlap))
        # The independent two-level exact unitary is within the selected
        # per-power operator-norm bound; roundoff is separate and tiny here.
        bound = next(
            p.total_error for p in selected.reconstruction.powers if p.power == sample.power
        )
        assert abs(sample.mean - expected) <= bound + 2e-14


def _independent_signal(selected, power):
    """z_p = <phi|U_p|phi> of the construction the trajectory emits, from explicit matrices.

    Dense Hamiltonian: expm(-i*p*tau*H). Unitary: powers of scipy's polar
    factor of the supplied A. Product formula: exp(i*Phi_p) times the r(p)-th
    power of the emitted second-order step, whose leaves exp(-i*a_j*P_j) use
    the stored step time, with Phi_p = -c_I*p*tau the target identity phase
    (its binary64 formation error is far below the threshold). These are
    independent matrix products, not the library's circuits.
    """
    from scipy.linalg import expm, polar
    from qiskit.quantum_info import SparsePauliOp

    rec = selected.reconstruction
    problem = selected.problem
    phi = np.asarray((selected.method.initial_state or problem.initial_state).physical_vector())
    phi = phi / np.linalg.norm(phi)
    if rec.target_kind == "unitary":
        operator = np.linalg.matrix_power(polar(problem.operator.dense_array())[0], power)
    elif rec.common_step is None:
        operator = expm(-1j * power * rec.tau * problem.A.dense_array())
    else:
        h = rec.common_step.step_time
        leaves = [expm(-1j * (0.5 * h * c) * SparsePauliOp(label).to_matrix())
                  for label, c in rec.pauli_terms]
        step = np.eye(len(phi), dtype=complex)
        for leaf in (*leaves, *reversed(leaves)):
            step = leaf @ step
        steps = next(p.steps for p in rec.powers if p.power == power)
        phase = -rec.identity_coefficient * power * rec.tau
        operator = np.exp(1j * phase) * np.linalg.matrix_power(step, steps)
    return np.vdot(phi, operator @ phi)


@pytest.mark.parametrize("case", ["dense", "pauli", "unitary"])
def test_exact_trajectory_signals_equal_independent_powers(case, monkeypatch):
    """Every trajectory point reads Re z_p and Im z_p of its power's independent signal.

    One controlled trajectory prepares |chi_0> once and applies the gap
    constructions between sorted powers, so the ancilla X and Y
    expectations at the point of power p are Re z_p and Im z_p. The
    comparison uses the predeclared regression threshold 2e-12, rtol=0, at
    most three qubits, stated before the run; it is a regression failure
    threshold, not a production bound, and for the synthesized dense gap
    blocks it certifies nothing beyond this unit-scale fixture.
    """
    from nwqlib.backends import qiskit_aer
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import Eigenphase

    tolerance = 2e-12
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hermitian = (raw + raw.conj().T) / 8
    initial = rng.normal(size=4) + 1j * rng.normal(size=4)
    if case == "dense":
        selected = plan(Eigenproblem(A=hermitian), method=QCELS(initial_state=initial), seed=7)
    elif case == "pauli":
        target = ingest_pauli((("II", .1), ("ZI", .3), ("XX", .2), ("IY", -.25)), num_qubits=2)
        selected = plan(Eigenproblem(A=target), method=RFE(initial_state=initial, num_frequencies=6,
                                                            num_samples=9), seed=7)
    else:
        from scipy.linalg import expm
        exact = expm(-1j * hermitian * 4)
        defective = exact @ (np.eye(4) + 4e-9 * np.diag([1.0, -1.0, 1.0, -1.0]))
        selected = plan(SpectralEstimation(unitary=defective, initial_state=initial),
                        method=QCELS(num_times=5, max_time=9), output=Eigenphase(), seed=7)
    calls = []
    submit = qiskit_aer._submit_aer_execution
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", lambda native: calls.append(1) or submit(native))
    result = solve(selected)
    assert len(calls) == 1 and len(selected.experiments) == 1
    queries = {q.experiment: q for q in selected.reconstruction.queries}
    assert len(result.samples) == len(queries)
    for sample in result.samples:
        signal = _independent_signal(selected, sample.power)
        expected = signal.real if queries[sample.experiment].phase_shift == 0 else signal.imag
        assert abs(sample.mean - expected) <= tolerance


@pytest.mark.parametrize("offset", [0.0, 0.7137, -10.0])
def test_qcels_molecular_energy_and_identity_offset_match_exact_diagonalization(offset):
    """QCELS on H2/STO-3G plus offset*I returns E0 + offset on the Pauli and dense routes.

    The prepared state is the exact ground vector, so only the controlled
    evolution separates the signal from z_p = exp(-i p theta*). The offset
    0.7137 Ha is the nuclear repulsion at this bond length, and -10 Ha shrinks
    the automatic tau by a factor of about five. A dropped or sign-flipped
    identity phase moves the estimate by |offset| or 2|offset|, and a tau that
    ignores the identity coefficient wraps it by 2 pi/tau.
    """
    from math import sqrt
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.operators import ingest_pauli
    from _h2_sto3g_reference import H2_STO3G_JW_TERMS

    matrix = SparsePauliOp.from_list(list(H2_STO3G_JW_TERMS)).to_matrix()
    energies, vectors = np.linalg.eigh(matrix)
    ground, exact = vectors[:, 0], energies[0] + offset
    terms = tuple((label, value + offset if label == "IIII" else value)
                  for label, value in H2_STO3G_JW_TERMS)
    selected = plan(Eigenproblem(A=ingest_pauli(terms, num_qubits=4)),
                    method=QCELS(initial_state=ground), seed=7)
    result = solve(selected)
    powers = sorted({sample.power for sample in result.samples})
    assert powers == list(range(11))
    # Method-error-bounded. Each controlled power is within eps = 1e-3 (the
    # default controlled_power_error_budget) of exp(-i p tau H) in operator
    # norm, so |z_p - exp(-i p theta*)| <= eps. QCELS maximizes
    # F = |mean_p z_p exp(i p theta)|. A returned phase with F >= 1 - eps, which
    # a global maximizer has because F(theta*) >= 1 - eps, gives F* >= 1 - 2 eps,
    # where F* is the Dirichlet kernel of the N = 11 consecutive powers. Using
    # 1 - cos y >= 2 y**2/pi**2 on its main lobe gives
    # |theta - theta*| <= R = sqrt(12 pi**2 eps/(N**2 - 1)) modulo 2 pi, and
    # outside that lobe F* <= 1/(N sin(pi/(2(N - 1)))) = 0.58. The energy bound
    # R/tau also needs |theta*| + R < pi, so that the phase interval does not
    # cross the branch at +-pi. The signal error, the objective at the returned
    # phase and the branch condition are checked from the public samples, so the
    # bound does not rely on the search finding the global maximum.
    tau, radius = selected.reconstruction.tau, sqrt(12 * pi**2 * 1e-3 / (11**2 - 1))
    queries = {query.experiment: query for query in selected.reconstruction.queries}
    signal = dict.fromkeys(powers, 0j)
    for sample in result.samples:
        # Phase shift 0 measures the real part of z_p and -pi/2 its imaginary part.
        shift = queries[sample.experiment].phase_shift
        assert shift in (0, -pi / 2)
        signal[sample.power] += sample.mean * (1 if shift == 0 else 1j)
    p, z = np.arange(11), np.array([signal[power] for power in powers])
    theta_true, theta_fit = tau * exact, tau * result.eigenvalue
    assert np.max(np.abs(z - np.exp(-1j * p * theta_true))) <= 1e-3
    assert abs(np.mean(z * np.exp(1j * p * theta_fit))) >= 1 - 1e-3
    assert -pi <= theta_fit < pi and abs(theta_true) + radius < pi
    assert abs(result.eigenvalue - exact) <= radius / tau
    # Machine precision. The dense route forms exact spectral powers, so only
    # binary64 roundoff remains; 1e-11 per unit of offset is a regression
    # threshold well above it.
    dense = solve(Eigenproblem(A=matrix + offset * np.eye(16)), method=QCELS(initial_state=ground), seed=7)
    assert abs(dense.eigenvalue - exact) <= 1e-11 * max(1.0, abs(offset))


def test_sampled_powers_select_their_own_steps_and_the_trajectory_keeps_the_common_grid():
    """Sampled settings select each power's count independently; the exact trajectory keeps one grid.

    A sampled setting starts from its own preparation, so its count at
    t_p=abs(p)*tau (binary64) is the smallest r with
    W_up*t_p**3/r**2 <= epsilon_p-t_p*d_up (``powers._independent_powers``,
    which then rechecks the complete emitted subtotal). The exact trajectory
    shares one propagated state and records the common step. The CX law
    of a sampled workload is sum over settings of shots*multiplicity*r_p
    times 4*sum_j support(P_j) for a preparation without CX
    (``powers.suzuki_step_cx``). Both routes plan, run on Aer and analyze.
    """
    from fractions import Fraction
    from nwqlib import estimate
    from nwqlib.operators import ingest_pauli
    from nwqlib.resources import ResourceContext

    target = ingest_pauli((("XI", .7), ("ZX", .4), ("YZ", .3), ("ZZ", .2)), num_qubits=2)
    method = QCELS(initial_state=[1, 0, 0, 0], controlled_power_backend="trotter_error_budgeted",
                   tau=.03, max_time=.24, num_times=8, grid_size=32,
                   controlled_power_error_budget=.001)
    budget = Fraction(method.controlled_power_error_budget)
    for shots in (None, 64):
        selected = plan(Eigenproblem(A=target), method=method, shots=shots, seed=7)
        rec = selected.reconstruction
        assert [p.power for p in rec.powers] == list(range(9)) and rec.pruned_mass == 0
        assert all(p.total_error <= method.controlled_power_error_budget for p in rec.powers)
        if shots is None:
            assert rec.common_step is not None
        else:
            assert rec.common_step is None
            W = Fraction(*rec.bound_coefficient)
            for power in rec.powers[1:]:
                t, r = Fraction(power.power * rec.tau), power.steps
                assert W * t**3 <= budget * r**2
                assert r == 1 or W * t**3 > budget * (r - 1)**2
            step_cx = 4 * sum(sum(axis != "I" for axis in label) for label, _ in rec.pauli_terms)
            steps = {p.power: p.steps for p in rec.powers}
            cx = estimate(selected, context=ResourceContext(basis="cx", batch_schedule="serial"))
            assert cx.quantity("cx").fact.value.numerator == shots * step_cx * sum(
                q.multiplicity * steps[q.power] for q in rec.queries)
        assert solve(selected).complete


@pytest.mark.parametrize("cap", ["shots"])
def test_rwpe_admits_its_whole_step_totals_before_the_first_query(tmp_path, monkeypatch, cap):
    """Quantum RWPE checks its max_steps one-shot observations before its first query.

    ``QPEMethod.prepare`` calls ``Run.check_capacity`` with the three
    ``max_steps`` totals on a fresh Run, so a cumulative cap below them
    refuses before any backend call and names ``max_steps``. At the cap the
    three steps run. Preparing again on a reopened Run with a committed
    checkpoint does not check ``max_steps`` again, so it finishes its remaining
    steps under the same cap. The classical route is not charged as quantum
    shots.
    """
    from nwqlib.backends import AerBackend
    from nwqlib.execution import ExecutionLimits

    calls = []
    submit_native = AerBackend.submit
    monkeypatch.setattr(AerBackend, "submit", lambda self, *args, **kwargs: calls.append(1) or submit_native(
        self, *args, **kwargs))
    selected = make_plan(estimator="rwpe", execution="quantum")
    steps = selected.method.max_steps

    def limits(value):
        return ExecutionLimits(**{f"max_total_{cap}": value})

    with pytest.raises(ValueError, match=f"at least {steps} needed"):
        solve(selected, limits=limits(steps - 1))
    assert calls == []
    assert solve(selected, limits=limits(steps)).processed_steps == steps and len(calls) == steps
    calls.clear()
    prepared = prepare(selected, limits=limits(steps), directory=tmp_path / "run")
    update = numerical.rwpe_update
    monkeypatch.setattr(numerical, "rwpe_update", Mock(side_effect=RuntimeError("interrupted before update")))
    with prepared.run as run:
        with pytest.raises(RuntimeError, match="interrupted"):
            submit(prepared)
        assert run.checkpoint_state is not None and len(calls) == 1
    monkeypatch.setattr(numerical, "rwpe_update", update)
    with load_run(tmp_path / "run", backend=AerBackend()) as restored:
        restored.plan.method.prepare(restored.plan, run=restored)
        result = restored.wait()
    assert result.processed_steps == steps and len(calls) == steps
    classical = make_plan(estimator="rwpe", execution="classical")
    assert solve(classical, limits=limits(steps - 1)).processed_steps == steps
