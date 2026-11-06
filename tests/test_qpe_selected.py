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
    # tr(.9 I)/3 is .9, and the rounded sum of the three quotients .3 is one unit below it, so the padded
    # form of this H keeps a nonidentity word of rounding unless the padding reads the constant diagonal.
    constant = plan(Eigenproblem(A=.9 * np.eye(3)), seed=7, method=QCELS(
        initial_state=[1., 0., 0.], controlled_power_backend="trotter_error_budgeted")).reconstruction
    assert constant.identity_coefficient == .9 and not constant.pauli_labels
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


def test_a_first_step_count_lost_to_rounding_is_corrected_and_confirmed_exactly():
    """A count that inverts only the formula term can miss B(r) <= epsilon, and the next candidate must meet it.

    On a three-qubit Ising chain at tau = .005 and epsilon = 1e-4 the first count r0 of power 216550 is about 6.5e6.
    Its margin in ``W t**3/r**2`` is below the time-displacement and angle terms, so the emitted subtotal exceeds
    epsilon by about 1.6e-13. The oracle recomputes the subtotal of the correct binary64 parameters with exact
    fractions here, independently of ``powers._emitted_parts``. This relation does not depend on the platform,
    because IEEE 754 fixes these divisions and products. Power 1 passes first and keeps its count, and an explicit
    ``max_trotter_steps`` still refuses the count the power needs.
    """
    from fractions import Fraction
    from math import ceil, isqrt

    from nwqlib.algorithms.qpe import QCELS
    from nwqlib.operators import ingest_pauli

    terms = (("ZZI", 1.0), ("IZZ", 1.0), ("ZIZ", 1.0), ("XII", .7), ("IXI", .7), ("IIX", .7))
    tau, budget = .005, 1e-4
    method = QCELS(initial_state=[1] + [0] * 7, tau=tau, num_times=2, grid_size=64, max_time=216550.5 * tau,
                   controlled_power_backend="trotter_error_budgeted", controlled_power_error_budget=budget)
    problem = Eigenproblem(A=ingest_pauli(terms, num_qubits=3))
    rec = plan(problem, method=method, shots=1, seed=7).reconstruction
    W = Fraction(*rec.bound_coefficient)
    mass = sum(abs(Fraction(value)) for _, value in terms)
    limit = Fraction(budget)

    def subtotal(power, steps):
        step = power * tau / steps
        angles = 2 * Fraction(1, 2**53) * abs(Fraction(.5 * step)) * mass + Fraction(1, 2**1074) * (len(terms) + mass)
        return (W * steps * abs(Fraction(step))**3 + mass * abs(steps * Fraction(step) - power * Fraction(tau))
                + steps * angles)

    def first_count(power):
        return 1 + isqrt(ceil(W * Fraction(power * tau)**3 / limit) - 1)

    counts = {row.power: row for row in rec.powers}
    assert set(counts) == {0, 1, 216550}
    assert counts[1].steps == first_count(1) and subtotal(1, first_count(1)) <= limit
    assert subtotal(216550, first_count(216550)) > limit
    assert counts[216550].steps > first_count(216550) and subtotal(216550, counts[216550].steps) <= limit
    assert counts[216550].total_error <= budget
    with pytest.raises(ValueError, match=f"Raise max_trotter_steps to at least {first_count(216550)}"):
        plan(problem, method=method.revise(max_trotter_steps=10**6), shots=1, seed=7)


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


def test_each_returned_qpe_result_checks_its_plan_once(tmp_path, monkeypatch):
    """Run completion, reanalysis and loading each check the returned Result against its Plan once.

    The check runs when the Result is attached to its Plan, and a Result
    that names another construction is refused there.
    """
    from nwqlib.algorithms.qpe.records import QPEAnalysis

    selected = make_plan(estimator="qcels")
    calls = []
    check = QPEAnalysis.validate_plan
    monkeypatch.setattr(QPEAnalysis, "validate_plan", lambda self, plan: calls.append(plan) or check(self, plan))
    result = solve(selected)
    assert calls == [selected]
    result.analyze(grid_size=32)
    assert calls == [selected] * 2
    loaded = load_result(result.save(tmp_path / "qcels"))
    assert len(calls) == 3 and calls[-1] is loaded.plan
    with pytest.raises(ValueError, match="method/construction differs"):
        result.revise(construction_id=selected.content_id)._attach(selected, result.data)


def test_rwpe_saves_each_step_in_two_checkpoints_around_its_observation(monkeypatch):
    """Each RWPE step saves its choice before preparation and its update after collection.

    After the prior, a classical run of max_steps steps writes two
    checkpoints per step. The pending choice, with its power, feedback and
    seed, is saved before the step's preparation. The new mean and processed
    count are saved together after the step's observation is collected.
    """
    from nwqlib._prepared_execution import Run
    from nwqlib.core.records import Record
    from nwqlib.algorithms.qpe import controller

    selected = make_plan(estimator="rwpe", execution="classical")
    steps = selected.method.max_steps
    events = []
    checkpoint, prepare_native, collect = controller._checkpoint, controller.prepare_experiment, Run.collect
    monkeypatch.setattr(controller, "_checkpoint", lambda run, state: events.append(
        ("pending", state.processed) if state.pending is not None else ("update", state.processed))
        or checkpoint(run, state))
    monkeypatch.setattr(controller, "prepare_experiment",
                        lambda *args, **kwargs: events.append(("prepare",)) or prepare_native(*args, **kwargs))
    monkeypatch.setattr(Run, "collect", lambda self, chunk: events.append(("collect",)) or collect(self, chunk))
    result = solve(selected)
    assert result.processed_steps == steps
    assert events == [event for k in range(steps)
                      for event in (("pending", k), ("prepare",), ("collect",), ("update", k + 1))]
    assert set(controller.Checkpoint.model_fields) - set(Record.model_fields) == {
        "plan_id", "mean", "processed", "pending", "last_attempt"}


def test_rwpe_max_work_limits_planning_steps_and_not_the_loop(tmp_path, monkeypatch):
    """RWPE's max_work limits each planning step and not its loop of scalar steps.

    For this Pauli Hamiltonian of two one-qubit terms the first step with a
    work limit is the commutation scan, 2*2*1 label reads and one pair, so
    max_work=1 refuses the Plan there with 5 units. The largest step's work
    plans and one unit less refuses. A run that is interrupted before an
    update, saved, loaded and resumed completes all steps, and neither the
    controller nor the final analysis checks a limit.
    """
    from nwqlib.operators import ingest_pauli

    steps = 3
    problem = SpectralEstimation(hamiltonian=ingest_pauli((("Z", .3), ("X", .2)), num_qubits=1),
                                 initial_state=[1.0, 0.0])

    def rwpe_plan(max_work):
        return plan(problem, method=RWPE(max_steps=steps, tau=.2, max_work=max_work), execution="quantum", seed=7)

    with pytest.raises(ValueError, match="needs 5 work units, above RWPE.max_work=1. Raise RWPE.max_work to at least 5"):
        rwpe_plan(1)
    works = []
    with monkeypatch.context() as patch:
        admit = owner._QPEMethod._admit
        patch.setattr(owner._QPEMethod, "_admit",
                      lambda self, work, size, *text, **offer: works.append(work) or admit(self, work, size, *text, **offer))
        rwpe_plan(None)
    largest = max(work for work in works if work is not None)
    with pytest.raises(ValueError, match=f"RWPE.max_work to at least {largest}"):
        rwpe_plan(largest - 1)
    selected = rwpe_plan(largest)
    monkeypatch.setattr(type(selected.method), "_admit", Mock(side_effect=AssertionError("checked after planning")))
    assert solve(selected).processed_steps == steps
    prepared = prepare(selected)
    update = numerical.rwpe_update
    monkeypatch.setattr(numerical, "rwpe_update", Mock(side_effect=RuntimeError("interrupted before update")))
    with pytest.raises(RuntimeError, match="interrupted"):
        submit(prepared)
    saved = prepared.run.save(tmp_path / "run")
    monkeypatch.setattr(numerical, "rwpe_update", update)
    result = load_run(saved, backend=prepared.run.backend).resume().wait()
    assert result.complete and result.processed_steps == steps


def test_method_refusal_separates_the_physical_memory_from_explicit_limits(monkeypatch):
    """A classical routine's refusal offers no limit to raise for the physical memory and keeps it for an explicit limit.

    Classical SPE planning of a 2x2 Hamiltonian first checks its classical
    routine, hermitian_eigensystem_work(2) + 16*4 = 256 work units and 80*4
    bytes. The physical memory is patched to 100 bytes, so nothing large is
    allocated.
    """
    monkeypatch.setattr(owner, "memory_limit", lambda value: 100 if value is None else value)

    def refusal(**limits):
        method = SPE(initial_state=[1, 0], tau=1., overlap_lower_bound=.3, **limits)
        with pytest.raises(ValueError) as refused:
            plan(Eigenproblem(A=np.diag([.1, .2])), method=method, execution="classical", seed=7)
        return str(refused.value)

    both = refusal(max_work=1)
    assert "256 work units, above SPE.max_work=1" in both and both.endswith("Raise SPE.max_work to at least 256.")
    assert "320 bytes, above the physical memory 100 of this computer" in both and "max_bytes" not in both
    assert refusal(max_bytes=100).endswith("320 bytes, above SPE.max_bytes=100. Raise SPE.max_bytes to at least 320.")


def test_an_analysis_refusal_names_its_arrays_and_each_remedy_leads_past_it(monkeypatch):
    """Each static estimator's analysis check states what its size follows, and what the message offers works.

    The sizes are the laws of ``method._analysis_size``, worked by hand, with the physical memory patched to 10**9
    bytes. QCELS at tau = .01 and max_time = 1e7 has largest power 1e9, so G = 8e9 + 1 grid points, and num_times = 8
    gives 18 samples and 9 complex points, for 32 G + 256*18 + 128*9 bytes. Planning allocates none of these arrays, so
    an explicit max_bytes of exactly that size makes the Plan and one byte less refuses. A smaller max_time/tau gives
    a smaller grid, and when grid_size sets G a smaller grid_size does. SPE needs 160 grid_size + 128 samples and RFE
    160 num_frequencies + 128 samples, with two samples per draw. Reanalysis does allocate its arrays, so there the
    message offers no limit, and the grid_size it names passes.

    The schedule check before them is 64 bytes per unit of ``method._schedule_size``'s frontier: 2(1 + min(num_times, P))
    for QCELS, 2 num_samples + 2 fourier_degree + 1 for SPE, 2 num_samples + num_frequencies for RFE and max_steps for
    RWPE. Planning allocates the samples and the SPE filter, so only a large RFE num_frequencies, a bound on its draws,
    is offered an explicit limit. That limit is the analysis amount 160 K + 256 S, so one retry passes both checks. An
    amount set by the samples offers no limit, here S = 5e6 at 1e9 bytes. No refused setting may reach a draw.
    """
    from nwqlib.operators import ingest_pauli

    monkeypatch.setattr(owner, "memory_limit", lambda value: 10**9 if value is None else value)
    problem = Eigenproblem(A=ingest_pauli((("XI", .7), ("ZX", .4), ("YZ", .3), ("ZZ", .2)), num_qubits=2))
    start = [1, 0, 0, 0]

    def planned(method):
        return plan(problem, method=method, shots=8, seed=1)

    def refusal(method):
        with pytest.raises(ValueError) as refused:
            planned(method)
        return str(refused.value)

    qcels = QCELS(initial_state=start, tau=.01, num_times=8, grid_size=64, max_time=1e7)
    needed = 32 * (8 * 10**9 + 1) + 256 * 18 + 128 * 9
    text = refusal(qcels)
    assert text.startswith(f"The QCELS analysis needs {needed} bytes, above the physical memory 1000000000 of this computer.")
    assert "8000000001 points" in text and "max_time/tau" in text and "num_times" in text
    assert f"QCELS.max_bytes to at least {needed}" in text and "solving" in text
    with monkeypatch.context() as patch:
        patch.setattr(numerical, "qcels", Mock(side_effect=AssertionError("planning ran the analysis")))
        assert planned(qcels.revise(max_bytes=needed)).method.max_bytes == needed
    assert refusal(qcels.revise(max_bytes=needed - 1)).endswith(f"Raise QCELS.max_bytes to at least {needed}.")
    planned(qcels.revise(max_time=1e4))

    grid = refusal(qcels.revise(max_time=1., grid_size=10**9))
    assert "1000000000 points" in grid and "grid_size" in grid
    planned(qcels.revise(max_time=1., grid_size=64))

    spe = SPE(initial_state=start, tau=.1, fourier_degree=1, overlap_lower_bound=1., num_samples=8, grid_size=10**8)
    text = refusal(spe)
    assert text.startswith(f"The SPE analysis needs {160 * 10**8 + 128 * 16} bytes,") and "QCELS" not in text
    assert "grid_size" in text and "num_samples" in text
    planned(spe.revise(grid_size=64))

    rfe = RFE(initial_state=start, tau=.1, num_samples=8, num_frequencies=10**7)
    text = refusal(rfe)
    assert text.startswith(f"The RFE analysis needs {160 * 10**7 + 128 * 16} bytes,") and "QCELS" not in text
    assert "num_frequencies" in text and "num_samples" in text
    planned(rfe.revise(num_frequencies=49))

    result = solve(Eigenproblem(A=np.diag([.1, .2])), method=QCELS(initial_state=[1, 0], tau=.1, num_times=4, max_time=.8,
                                                                  grid_size=64), execution="classical", seed=7)
    with pytest.raises(ValueError) as refused:
        result.analyze(grid_size=10**9)
    text = str(refused.value)
    assert text.startswith("The QCELS analysis needs") and "1000000000 points" in text and "max_bytes" not in text
    assert result.analyze(grid_size=128).complete

    wide = rfe.revise(num_frequencies=10**9)
    later = 160 * 10**9 + 256 * 8
    with monkeypatch.context() as patch:
        patch.setattr(numerical, "planned_power_schedule", Mock(side_effect=AssertionError("drew a refused schedule")))
        text = refusal(qcels.revise(num_times=10**9, max_time=1e8))
        assert text.startswith(f"The QCELS schedule needs {64 * 2 * (1 + 10**9)} bytes, above the physical memory 1000000000")
        assert "num_times" in text and "max_time/tau" in text and "max_bytes" not in text
        assert refusal(qcels.revise(max_time=1e8)).startswith("The QCELS analysis needs")
        for change, entries, named in ((dict(fourier_degree=10**9), 2 * 10**9 + 1 + 16, "fourier_degree"),
                                       (dict(num_samples=10**9), 3 + 2 * 10**9, "num_samples")):
            text = refusal(spe.revise(**change))
            assert text.startswith(f"The SPE schedule needs {64 * entries} bytes,") and named in text
            assert "max_bytes" not in text and "QCELS" not in text
        text = refusal(wide)
        assert text.startswith(f"The RFE schedule needs {64 * (10**9 + 16)} bytes,") and "num_frequencies" in text
        assert f"RFE.max_bytes to at least {later}" in text and "solving" in text
        assert refusal(wide.revise(max_bytes=later - 1)).startswith(f"The RFE analysis needs {later} bytes")
        for method, stage, amount in ((rfe.revise(num_samples=10**9, num_frequencies=49), "schedule", 64 * (2 * 10**9 + 49)),
                                      (rfe.revise(num_samples=5 * 10**6, num_frequencies=49), "analysis",
                                       160 * 49 + 256 * 5 * 10**6)):
            text = refusal(method)
            assert text.startswith(f"The RFE {stage} needs {amount} bytes,") and "num_samples" in text
            assert "max_bytes" not in text
        steps = RWPE(initial_state=start, max_steps=1000, max_bytes=10**4)
        with pytest.raises(ValueError) as refused:
            plan(problem, method=steps, seed=1)
        text = str(refused.value)
        assert text.startswith("The RWPE schedule needs 64000 bytes, above RWPE.max_bytes=10000.") and "max_steps" in text
        assert text.endswith("Raise RWPE.max_bytes to at least 64000.")
    planned(wide.revise(max_bytes=later))
    planned(spe.revise(grid_size=64))
    plan(problem, method=steps.revise(max_steps=8), seed=1)


@pytest.mark.parametrize("matrix, finite_radius", [
    (np.array([[1.0, 2.0 - 1j], [2.0 + 1j, -3.0]]), True),
    (np.array([[1.5e308, 1e308], [1e308, -1.5e308]]), False),
    (np.full((2, 2), 5e-324), True),
])
def test_automatic_time_comes_from_the_absolute_row_sum_at_every_scale(matrix, finite_radius):
    """tau = 0.9*pi/R with R the largest absolute row sum, finite when R overflows and clamped when it underflows.

    An ordinary matrix gives R and tau directly. Row sums above the binary64
    maximum give R = None and the finite time of the components scaled by
    the largest one, and a Plan of that matrix records that the aliasing
    bound is not established. Subnormal entries give a time clamped to the
    binary64 maximum.
    """
    tau, radius, label = owner._safe_time(matrix, None)
    assert label == "absolute row-sum bound" and np.isfinite(tau) and tau > 0
    if finite_radius:
        assert radius == float(np.abs(matrix).sum(axis=1).max()) and tau * radius < pi
        assert tau == min(0.9 * pi / radius, np.finfo(float).max) if radius > 1e-300 else tau == np.finfo(float).max
    else:
        assert radius is None
        scale = 1.5e308
        assert tau == pytest.approx(0.9 * pi / (1 + 1e308 / scale) / scale, rel=1e-15, abs=0)
        selected = plan(Eigenproblem(A=matrix), method=RWPE(initial_state=[1.0, 0.0], max_steps=1),
                        execution="classical", seed=7)
        assert selected.reconstruction.spectral_radius_bound is None
        assert not selected.reconstruction.aliasing_bound_sufficient


def test_power_and_step_limits_refuse_only_when_they_are_set():
    # Planning alone reaches power 2e5 and more than 1e6 product-formula steps per power, and each power still
    # meets its allowance. A limit set on the Method refuses the same schedule and names its field.
    from nwqlib.algorithms.qpe import QCELS
    from nwqlib.operators import ingest_pauli

    problem = Eigenproblem(A=ingest_pauli((("ZZI", 1.5), ("IZZ", 1.5), ("XII", 1.5), ("IXI", 1.5), ("IIX", 1.5)),
                                          num_qubits=3))
    method = QCELS(initial_state=[1] + [0] * 7, tau=.02, num_times=8, grid_size=256, max_time=4001.)
    powers = plan(problem, method=method, shots=16, seed=7).reconstruction.powers
    assert max(row.power for row in powers) >= 2 * 10**5 and max(row.steps for row in powers) > 10**6
    assert all(row.total_error <= method.controlled_power_error_budget for row in powers)
    with pytest.raises(ValueError, match="exceeds max_power=1000"):
        plan(problem, method=method.revise(max_power=1000), shots=16, seed=7)
    with pytest.raises(ValueError, match=r"above max_trotter_steps=100\. Raise max_trotter_steps"):
        plan(problem, method=method.revise(max_trotter_steps=100), shots=16, seed=7)


def _aer_positions(circuit):
    from qiskit_aer import AerSimulator
    from nwqlib.backends.qiskit_aer import _AerDecompose, _lower_aer_circuit

    lowered, _ = _lower_aer_circuit(circuit, _AerDecompose(AerSimulator().target))
    return len(lowered.data)


def test_memory_of_a_controlled_step_follows_its_native_positions_and_survives_loading(tmp_path):
    # Aer's lowering of the built step is the independent count, by operation class on this Aer target. One sweep of a
    # term of Pauli weight w with x letters X or Y emits 2(w - 1) CX, one CRZ and 2x basis changes, and a step has two
    # sweeps. These terms separate the native positions from the step's CX count, which decomposes each CRZ into two
    # CX, and from a count without the basis changes. The bytes per position are an empirical constant that this
    # integer relation does not calibrate. No estimate is saved, so each restored step is built and lowered as well.
    from qiskit_aer import AerSimulator
    from nwqlib.algorithms.qpe import powers
    from nwqlib.backends.qiskit_aer import _AerDecompose, _lower_aer_circuit
    from nwqlib.blocks.selection import NATIVE_POSITION_BYTES
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation

    terms = (("XYX", .2), ("ZZI", .3), ("IIX", -.1))
    parity = 2 * sum(2 * (len(label) - label.count("I") - 1) for label, _ in terms)
    central = 2 * len(terms)
    basis = 2 * sum(2 * (label.count("X") + label.count("Y")) for label, _ in terms)
    positions = parity + central + basis
    assert positions != powers.suzuki_step_cx(terms)
    selected = plan(Eigenproblem(A=ingest_pauli(terms, num_qubits=3)),
                    method=QCELS(initial_state=ingest_occupation((1, 0, 0), num_qubits=3), tau=.1, num_times=4,
                                 grid_size=64, max_time=.8),
                    shots=16, seed=7)
    loaded = load_plan(save_plan(selected, ArchiveFiles(tmp_path, 4_000_000)), ArchiveFiles(tmp_path, 4_000_000))
    decompose = _AerDecompose(AerSimulator().target)
    for candidate in (selected, loaded):
        steps = [b for b in candidate.blocks if b._constructor is powers.construct_suzuki_step]
        assert steps
        for block in steps:
            native = _lower_aer_circuit(block._constructor(block, (), None), decompose)[0].count_ops()
            assert (native["cx"], native["crz"], sum(native.values())) == (parity, central, positions)
            assert block._native_bytes == block._definition_bytes == NATIVE_POSITION_BYTES * positions
    assert [(b._native_bytes, b._definition_bytes) for b in loaded.blocks] == [
        (b._native_bytes, b._definition_bytes) for b in selected.blocks]


def test_memory_of_a_dense_power_follows_its_native_positions():
    # The kept-circuit estimate of the controlled synthesis, read per native position at the dominant size.
    from scipy.stats import unitary_group
    from nwqlib.algorithms.qpe import powers
    from nwqlib.blocks.selection import NATIVE_POSITION_BYTES
    from nwqlib.subroutines._dense_synthesis import controlled_unitary_circuit

    native = _aer_positions(controlled_unitary_circuit(unitary_group.rvs(16, random_state=3), 1))
    assert native <= powers.dense_memory_bytes(16) / NATIVE_POSITION_BYTES <= 2 * native


def test_prepare_refuses_the_memory_of_every_circuit_before_building_one(monkeypatch):
    # The circuits of a ten-qubit periodic Ising chain repeat a controlled step of 120 native positions and keep
    # far more than the 10**7 bytes this Method allows. No step is built and the Run records no preparation.
    from nwqlib.execution import Run
    from nwqlib.algorithms.qpe import powers
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation

    def forbidden(*args, **kwargs):
        raise AssertionError("a controlled step was built before the memory comparison")

    monkeypatch.setattr(powers, "construct_suzuki_step", forbidden)
    labels = tuple(("I" * j + "ZZ" + "I" * (8 - j), 1.5) for j in range(9)) + (("Z" + "I" * 8 + "Z", 1.5),)
    labels += tuple(("I" * j + "X" + "I" * (9 - j), 1.5) for j in range(10))
    selected = plan(Eigenproblem(A=ingest_pauli(labels, num_qubits=10)),
                    method=QCELS(initial_state=ingest_occupation((1, 0) * 5, num_qubits=10), tau=.02, num_times=8,
                                 grid_size=256, max_time=4, max_bytes=10**7),
                    shots=1024, seed=7)
    with pytest.raises(ValueError, match=r"preparing every circuit of this Plan needs about \d+ bytes.*max_bytes=10000000"):
        prepare(selected, settings="all", progress=False)
    with Run(selected, progress=False) as run:
        with pytest.raises(ValueError, match=r"needs about \d+ bytes.*max_bytes=10000000"):
            run.resume()
        assert run.trace.preparations == 0 and len(run.trace.events) == 0
