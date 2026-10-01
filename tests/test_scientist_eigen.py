"""Independent two-qubit eigenvalue witnesses on the selected public Method path."""

import numpy as np
import pytest
from nwqlib.core.planning import Plan, RandomStreams
from nwqlib.problems.records import Eigenproblem
from nwqlib.algorithms.lanczos import Lanczos, SensitivitySampling
from nwqlib.algorithms.gcim.fixed_basis import FixedGCIM
from nwqlib._prepared_execution import Run


def execute(problem, method, mode="quantum", shots=None):
    plan = method.plan(
        problem, output=problem.default_output(), execution=mode, shots=shots, rng=RandomStreams(4)
    )
    assert type(plan) is Plan and plan.problem is problem and plan.method is method
    assert sum(int(register.width) for register in plan.construction.program.registers) <= 13
    run = Run(plan)
    method.prepare(plan, run=run)
    result = method.execute(plan, run=run)
    return result, run


@pytest.mark.parametrize("mode", ["quantum", "classical"])
def test_same_target_different_initialization(mode):
    p = Eigenproblem(A=[[1, 1], [1, 1]])
    for method in (
        Lanczos(initial_state=[1, 0], krylov_dimension=2),
        FixedGCIM(basis=([1, 1], [1, -1])),
    ):
        result, run = execute(p, method, mode)
        assert result.eigenvalue == pytest.approx(0, abs=1e-12, rel=0)
        assert (all(e.execution == "host_kernel" for e in run.trace.events)) == (
            mode == "classical"
        )
        again = result.analyze(overlap_cutoff=1e-9)
        assert again.data is result.data and again.plan is result.plan
        assert again.eigenvalue == pytest.approx(result.eigenvalue, abs=1e-12, rel=0)
        assert len(run.trace.events) == len(result.data.trace.events)


def test_complex_gcim_shift_uses_gram_not_coordinate_identity():
    vectors = ([1, 0], [1, 1j])
    h = np.array([[0.7, 0.3 + 0.2j], [0.3 - 0.2j, -0.4]])
    base, _ = execute(Eigenproblem(A=h), FixedGCIM(basis=vectors))
    shifted, _ = execute(Eigenproblem(A=h + 2.5 * np.eye(2)), FixedGCIM(basis=vectors))
    expected = np.linalg.eigvalsh(h)[0]
    assert base.eigenvalue == pytest.approx(expected, abs=2e-12, rel=0)
    assert shifted.eigenvalue - base.eigenvalue == pytest.approx(2.5, abs=2e-12, rel=0)
    s = np.array([[complex(z.real, z.imag) for z in row] for row in base.pencil.overlap])
    assert s[0, 1] == pytest.approx(2**-0.5, abs=1e-12, rel=0)


def test_lanczos_rank_deficiency_does_not_establish_ground():
    result, _ = execute(
        Eigenproblem(A=np.diag([0.0, 1.0])), Lanczos(initial_state=[0, 1], krylov_dimension=2)
    )
    assert result.eigenvalue == pytest.approx(1, abs=1e-12, rel=0)
    assert result.kept_rank == 1


def test_original_dimension_padding_and_raw_chebyshev_gram():
    problem = Eigenproblem(A=np.diag([0.0, 1.0, 2.0]))
    result, _ = execute(problem, Lanczos(initial_state=[1, 1, 1], krylov_dimension=3), "classical")
    assert result.eigenvalue == pytest.approx(0, abs=2e-12, rel=0)
    assert result.overlap[1][1] != pytest.approx(1, abs=1e-8, rel=0)
    assert result.plan.problem.dimension == 3
    with pytest.raises(ValueError, match="original problem dimension"):
        Lanczos(krylov_dimension=4).plan(
            problem,
            output=problem.default_output(),
            execution="quantum",
            shots=None,
            rng=RandomStreams(1),
        )


def test_sensitivity_actual_population_and_no_reacquisition(monkeypatch):
    method = Lanczos(
        initial_state=[1, 0],
        krylov_dimension=2,
        sampling=SensitivitySampling(total_shots=100, pilot_fraction=0.2),
    )
    result, run = execute(Eigenproblem(A=[[1, 1], [1, 1]]), method)
    assert len(run.trace.events) == 4
    assert sum(event.shots for event in run.trace.events) == 100
    assert len(result.contribution_ids) == 2
    old_state = run.rng.snapshot()
    again = method.execute(result.plan, run=run)
    assert run.rng.snapshot() == old_state and len(run.trace.events) == 4
    assert result.eigenvalue is not None
    assert again.eigenvalue == result.eigenvalue
    assert result.cutoff_source == "empirical_gram_rms"
    strict = result.analyze(overlap_cutoff_policy="confidence", overlap_failure_probability=.1)
    rec = result.plan.reconstruction
    counts = [stat.shots for degree, stat in strict.statistics if degree < 2*rec.krylov_dimension-1]
    expected = 2*rec.krylov_dimension*np.sqrt(2*np.log(2*len(counts)/.1)/min(counts))
    assert strict.cutoff == pytest.approx(expected, rel=2e-15)
    assert strict.analysis_failure_probability == .1 and strict.cutoff_source == "hoeffding_gram_bound"
    assert run.rng.snapshot() == old_state and len(run.trace.events) == 4
    from nwqlib.algorithms.lanczos import numerical
    monkeypatch.setattr(numerical, "_projected_matrices",
                        lambda *a, **k: pytest.fail("cutoff validation rebuilt a matrix"))
    strict.validate_plan(result.plan)
    with pytest.raises(ValueError, match="cutoff"):
        strict.revise(cutoff=strict.cutoff*1.01).validate_plan(result.plan)


# Projected analysis envelope of numerical.reconstruct at krylov_dimension m=2:
# 512*m*m+128*m bytes and 32*m*m+16*m**3 work.
ANALYSIS_BYTES = 512 * 2 * 2 + 128 * 2
ANALYSIS_WORK = 32 * 2 * 2 + 16 * 2**3
# The Pauli census and the Gershgorin frame each admit a fixed 65,536-byte
# allowance before allocating (ENGINEERING_CONSTANTS "Chebyshev Lanczos
# defaults" and "Eigen input conversion and classical preparation"), more than
# the analysis envelope. A byte limit equal to the envelope therefore passes
# the analysis admission and stops at that later phase, and the byte case
# completes under the default limit.
FIXED_PHASE = "Lanczos Pauli census|Gershgorin frame workspace"


def at_envelope(limit, envelope, plan_method, method):
    """Return the Method that completes: the envelope itself, or the default byte limit."""
    if limit != "max_bytes":
        return method(envelope)
    with pytest.raises(ValueError, match=FIXED_PHASE):
        plan_method(method(envelope))
    return method(None)


@pytest.mark.parametrize("mode,shots", [("quantum", None), ("quantum", 100), ("classical", None)])
@pytest.mark.parametrize(
    "limit,envelope,message",
    [
        ("max_bytes", ANALYSIS_BYTES, f"Lanczos projected analysis needs {ANALYSIS_BYTES} data bytes"),
        ("max_analysis_work", ANALYSIS_WORK, "Lanczos projected analysis exceeds max_analysis_work"),
    ],
)
def test_lanczos_plan_admits_projected_analysis_before_acquisition(mode, shots, limit, envelope, message):
    """The projected analysis runs after every moment is acquired, but its envelope depends
    only on m and the Method's limits. One unit below the envelope must stop plan(), so no
    Run and no acquisition exist. At the envelope itself the analysis admission passes: a
    max_analysis_work limit plans and completes, and a max_bytes limit stops at the fixed
    census or Gershgorin allowance (FIXED_PHASE), so that case completes under the default
    byte limit.
    """
    problem = Eigenproblem(A=[[1, 1], [1, 1]])

    def plan_method(method):
        return method.plan(
            problem, output=problem.default_output(), execution=mode, shots=shots, rng=RandomStreams(4)
        )

    def lanczos(value):
        return Lanczos(initial_state=[1, 0], krylov_dimension=2, **({} if value is None else {limit: value}))

    with pytest.raises(ValueError, match=message):
        plan_method(lanczos(envelope - 1))
    method = at_envelope(limit, envelope, plan_method, lanczos)
    result, run = execute(problem, method, mode, shots)
    assert result.eigenvalue is not None
    assert len(run.trace.events) == len(result.plan.experiments)


@pytest.mark.parametrize(
    "limit,envelope,message",
    [
        ("max_bytes", ANALYSIS_BYTES, f"Lanczos projected analysis needs {ANALYSIS_BYTES} data bytes"),
        ("max_analysis_work", 2 * ANALYSIS_WORK, "sensitivity pilot exceeds max_analysis_work"),
    ],
)
def test_sensitivity_plan_admits_pilot_solve_and_derivative_before_acquisition(limit, envelope, message):
    """The pilot solves the projected pencil and then differentiates it, within the byte
    envelope of numerical.reconstruct and twice its work. One unit below either must stop
    plan() before the pilot stage is acquired. At the envelope itself the analysis admission
    passes: a max_analysis_work limit runs both stages, and a max_bytes limit stops at the
    fixed census or Gershgorin allowance (FIXED_PHASE), so that case runs both stages under
    the default byte limit.
    """
    problem = Eigenproblem(A=[[1, 1], [1, 1]])

    def method(value):
        return Lanczos(
            initial_state=[1, 0],
            krylov_dimension=2,
            sampling=SensitivitySampling(total_shots=100, pilot_fraction=0.2),
            **({} if value is None else {limit: value}),
        )

    def plan_method(selected):
        return selected.plan(
            problem, output=problem.default_output(), execution="quantum", shots=None, rng=RandomStreams(4)
        )

    with pytest.raises(ValueError, match=message):
        plan_method(method(envelope - 1))
    result, run = execute(problem, at_envelope(limit, envelope, plan_method, method))
    assert result.eigenvalue is not None
    assert len(run.trace.events) == 2 * len(result.plan.experiments)


def test_sensitivity_scalar_operator_needs_no_pilot_work():
    """A scalar operator has every moment known, so the controller acquires nothing and
    never runs the pilot. One solve's work must therefore suffice for its Plan.
    """
    method = Lanczos(
        initial_state=[1, 0],
        krylov_dimension=2,
        sampling=SensitivitySampling(total_shots=100, pilot_fraction=0.2),
        max_analysis_work=ANALYSIS_WORK,
    )
    result, run = execute(Eigenproblem(A=[[2, 0], [0, 2]]), method)
    assert result.eigenvalue == 2
    assert not run.trace.events


def test_lanczos_analysis_checks_the_analyzing_method_limits():
    """Lanczos.analyze takes its limits from the Method it is called on, which need not be the
    one that planned the data. The projected analysis must check those limits before its solve.
    Reanalysis under the planning Method's own limits, here exactly the work envelope, must
    pass. The planning Method keeps the default byte limit, which the fixed census allowance
    needs (FIXED_PHASE).
    """
    problem = Eigenproblem(A=[[1, 1], [1, 1]])
    planned = Lanczos(
        initial_state=[1, 0],
        krylov_dimension=2,
        max_analysis_work=ANALYSIS_WORK,
    )
    result, _ = execute(problem, planned)
    again = result.analyze(overlap_cutoff=1e-9)
    assert again.eigenvalue == pytest.approx(result.eigenvalue, abs=1e-12, rel=0)
    for limit, envelope, message in (
        ("max_bytes", ANALYSIS_BYTES, f"Lanczos projected analysis needs {ANALYSIS_BYTES} data bytes"),
        ("max_analysis_work", ANALYSIS_WORK, "Lanczos projected analysis exceeds max_analysis_work"),
    ):
        small = Lanczos(initial_state=[1, 0], krylov_dimension=2, **{limit: envelope - 1})
        with pytest.raises(ValueError, match=message):
            small.analyze(result.plan, result.data, settings={})


@pytest.mark.parametrize("mode", ["quantum", "classical"])
def test_adapt_commutator_actual_decision_and_reanalysis(mode):
    from nwqlib.algorithms.gcim.adapt import ADAPT
    from nwqlib.operators import ingest_pauli
    from nwqlib.algorithms.gcim.adapt_verification import AdaptVerificationOptions

    p = Eigenproblem(A=[[1.0, 0], [0, -1.0]])
    method = ADAPT(initial_state=[1, 1], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),))
    plan = method.plan(
        p, output=p.default_output(), execution=mode, shots=None, rng=RandomStreams(3)
    )
    if mode == "quantum":
        assert plan.reconstruction.pool[0].commutator.rows() == (("X", 2.0),)
    else:
        assert plan.reconstruction.processed_hamiltonian == p.A.reference
    run = Run(plan)
    method.prepare(plan, run=run)
    assert not run.trace.events
    result = method.execute(plan, run=run)
    assert result.eigenvalue == pytest.approx(-1, abs=3e-12, rel=0)
    assert result.history[0].gradient_norm == pytest.approx(2, abs=1e-12, rel=0)
    assert result.selected == (0,) and result.stop_reason == "pool_exhausted"
    changed = result.analyze(overlap_cutoff=1e-8)
    assert changed.history == result.history and changed.data is result.data
    receipt, facts = result.verify(
        checks=AdaptVerificationOptions(name="residual", comparisons=("residual",))
    )
    metric = next(f for f in facts if f.fact.quantity == "residual.residual_norm")
    assert metric.fact.value.value < 1e-11
    assert receipt.result_id == result.content_id


def test_saved_eigen_results_restore_selection_without_analysis(tmp_path, monkeypatch):
    from nwqlib.saved_evidence import load_result
    from nwqlib.algorithms.gcim.adapt import ADAPT
    from nwqlib.operators import ingest_pauli

    p = Eigenproblem(A=[[1.0, 0.3], [0.3, -1.0]])
    methods = (
        Lanczos(initial_state=[1, 0], krylov_dimension=2),
        FixedGCIM(basis=([1, 0], [0, 1])),
        ADAPT(initial_state=[1, 1], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),)),
    )
    for index, method in enumerate(methods):
        plan = method.plan(
            p, output=p.default_output(), execution="classical", shots=None, rng=RandomStreams(3)
        )
        run = Run(plan)
        result = method.execute(plan, run=run)
        location = tmp_path / str(index)
        result.save(location)

        def forbidden(*args, **kwargs):
            raise AssertionError("loading reanalyzed or replanned scientific data")

        with monkeypatch.context() as patch:
            patch.setattr(type(method), "analyze", forbidden)
            patch.setattr(type(method), "plan", forbidden)
            loaded = load_result(location, method=type(method))
        assert loaded.content_id == result.content_id and loaded.plan.content_id == plan.content_id
        assert loaded.analyze(overlap_cutoff=1e-9).eigenvalue == pytest.approx(
            result.eigenvalue, abs=2e-12, rel=0
        )


@pytest.mark.parametrize("family", ["adapt", "sensitivity"])
def test_interrupted_analysis_retry_keeps_physical_population_and_rng(monkeypatch, family):
    """Explicit analysis recovery adds one analysis attempt while preserving acquired populations
    and the RNG trajectory.
    """
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.operators import ingest_pauli

    if family == "adapt":
        from nwqlib.algorithms.gcim import adapt_acquisition as owner

        function = "_pencil_from_observations"
        method = ADAPT(initial_state=[1, 1], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),))
        problem = Eigenproblem(A=[[1.0, 0], [0, -1.0]])
    else:
        from nwqlib.algorithms.lanczos import workflow as owner

        function = "_sensitivity_weights"
        method = Lanczos(
            initial_state=[1, 0],
            krylov_dimension=2,
            sampling=SensitivitySampling(total_shots=100, pilot_fraction=0.2),
        )
        problem = Eigenproblem(A=[[1.0, 1], [1, 1.0]])
    plan = method.plan(
        problem,
        output=problem.default_output(),
        execution="quantum",
        shots=None,
        rng=RandomStreams(31),
    )
    control = Run(plan)
    expected = control.wait(timeout=5, poll_interval=0)
    run = Run(plan)
    original = getattr(owner, function)
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("injected interrupted analysis")

    monkeypatch.setattr(owner, function, interrupted)
    with pytest.raises(RuntimeError, match="injected interrupted analysis"):
        run.resume()
    acquired = tuple(e.attempt for e in run.trace.events)
    before = run.rng.snapshot()
    with pytest.raises(RuntimeError, match="interrupted"):
        run.resume()
    assert calls == [1] and run.rng.snapshot() == before
    monkeypatch.setattr(owner, function, original)
    run.resume(reanalyze=True)
    actual = run.wait(timeout=5, poll_interval=0)
    assert expected.eigenvalue is not None
    assert actual.eigenvalue == expected.eigenvalue
    assert len(run.trace.events) == len(control.trace.events)
    assert sum(e.shots for e in run.trace.events) == sum(e.shots for e in control.trace.events)
    assert run.rng.snapshot() == control.rng.snapshot()
    assert set(acquired) <= set(e.attempt for e in run.trace.events)
    assert (
        run.checkpoint_state["analysis_attempts"]
        == control.checkpoint_state["analysis_attempts"] + 1
    )
    assert len(run.checkpoint_state["analysis_retries"]) == 1
    with pytest.raises(ValueError, match="result.analyze"):
        run.resume(reanalyze=True)


@pytest.mark.parametrize("storage", ["dense", "csr", "csc", "pauli"])
def test_original_matvec_affine_frame_and_saved_ritz(storage, monkeypatch, tmp_path):
    from scipy.sparse import csr_matrix, csc_matrix
    from nwqlib.operators import ingest_pauli
    from nwqlib.algorithms import _eigen_inputs
    from nwqlib.saved_evidence import load_result

    # A²=2.09 I gives an independent spectrum. |0>, A|0> span C²;
    # Gershgorin and Pauli L1 frames have different raw moments but same Ritz roots.
    matrix = np.array([[0.3, 1 - 1j], [1 + 1j, -0.3]])
    inputs = {
        "dense": matrix,
        "csr": csr_matrix(matrix),
        "csc": csc_matrix(matrix),
        "pauli": ingest_pauli((("X", 1), ("Y", 1), ("Z", 0.3)), num_qubits=1),
    }
    monkeypatch.setattr(
        _eigen_inputs, "pauli_coefficients", lambda *a: pytest.fail("implicit conversion")
    )
    problem = Eigenproblem(A=inputs[storage])
    result, run = execute(
        problem,
        Lanczos(initial_state=[1, 0], krylov_dimension=2, overlap_cutoff=1e-14),
        "classical",
    )
    assert result.kept_rank == 2
    np.testing.assert_allclose(
        result.eigenvalues, [-np.sqrt(2.09), np.sqrt(2.09)], rtol=0, atol=2e-12
    )
    rec = result.plan.reconstruction
    assert rec.operator == problem.A.reference
    expected_alpha = 2.3 if storage == "pauli" else 0.3 + np.sqrt(2)
    assert rec.alpha == pytest.approx(expected_alpha, rel=2e-14, abs=0)
    assert result.moments[1] == pytest.approx(0.3 / expected_alpha, rel=2e-14, abs=0)
    assert result.plan._native["reference"].basis.dimension == 2
    result.save(tmp_path / storage)
    loaded = load_result(tmp_path / storage, method=Lanczos)
    assert loaded.plan.reconstruction == rec
    assert loaded.analyze(overlap_cutoff=1e-14).eigenvalue == pytest.approx(
        -np.sqrt(2.09), rel=0, abs=2e-12
    )
    run.close()


def test_eigen_scalar_shots_and_conversion_admission(monkeypatch):
    from scipy.sparse import csr_matrix, csc_matrix, diags
    from nwqlib.algorithms import _eigen_inputs
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.operators import ingest_pauli

    for scalar in (0.0, 2.5):
        result, run = execute(
            Eigenproblem(A=scalar * np.eye(2)),
            Lanczos(initial_state=[1, 0], krylov_dimension=2),
            shots=17,
        )
        assert result.eigenvalue == scalar
        assert not run.trace.events and not result.contribution_ids
        assert not result.plan.construction.program.bindings
        run.close()
    for sparse in (csr_matrix, csc_matrix):
        problem = Eigenproblem(A=sparse([[1.0, 0.2], [0.2, -1.0]]))
        with pytest.raises(ApplicabilityError, match="input_conversion"):
            execute(problem, Lanczos(initial_state=[1, 0], krylov_dimension=2))
        result, run = execute(
            problem,
            Lanczos(initial_state=[1, 0], krylov_dimension=2, input_conversion="dense_pauli"),
        )
        assert result.eigenvalue == pytest.approx(-np.sqrt(1.04), rel=0, abs=2e-12)
        run.close()
    calls = []
    original = _eigen_inputs.pauli_coefficients
    monkeypatch.setattr(
        _eigen_inputs, "pauli_coefficients", lambda a: (calls.append(1), original(a))[1]
    )
    p = Eigenproblem(A=np.eye(2))
    for method, mode, shots in (
        (Lanczos(krylov_dimension=3), "quantum", None),
        (Lanczos(degrees=(1, 1)), "quantum", None),
        (Lanczos(), "classical", 2),
        (FixedGCIM(basis=([1, 0], [0, 1]), max_basis_size=1), "quantum", None),
        (Lanczos(max_conversion_work=3), "quantum", None),
    ):
        with pytest.raises(ValueError):
            method.plan(
                p, output=p.default_output(), execution=mode, shots=shots, rng=RandomStreams(4)
            )
    assert calls == []
    # Large sparse planning exercises original access only, no state expansion/solve.
    p = Eigenproblem(A=diags(np.arange(1024, dtype=float), format="csr"))
    plan = Lanczos(krylov_dimension=2).plan(
        p, output=p.default_output(), execution="classical", shots=None, rng=RandomStreams(4)
    )
    assert plan._native["operator"] is p.A and calls == []
    # Compact quantum input is unaffected by the small dense convenience limit.
    p = Eigenproblem(A=ingest_pauli((("Z" + "I" * 5, 1),), num_qubits=6))
    plan = Lanczos(krylov_dimension=2).plan(
        p, output=p.default_output(), execution="quantum", shots=None, rng=RandomStreams(4)
    )
    assert plan._native["operator"] is p.A


def test_preparation_admitted_before_recurrence_and_duplicate_basis(monkeypatch):
    from nwqlib.algorithms import _eigen_inputs
    from nwqlib.algorithms.lanczos import method as lanczos_method
    from nwqlib.algorithms.gcim import fixed_basis
    from nwqlib.problems.inputs import ingest_product, ingest_vector

    problem = Eigenproblem(A=np.diag([0.0, 1.0, 2.0, 3.0]))
    seed = ingest_product(([2**-0.5, 2**-0.5], [2**-0.5, 1j * 2**-0.5]))
    method = Lanczos(initial_state=seed, krylov_dimension=2)
    plan = method.plan(
        problem,
        output=problem.default_output(),
        execution="classical",
        shots=None,
        rng=RandomStreams(4),
    )
    _, work = method._classical_requirements(plan._native["operator"], seed, plan.reconstruction)
    calls = []
    original = _eigen_inputs.state_direction
    monkeypatch.setattr(
        lanczos_method, "state_direction", lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    )
    limited = method.revise(max_classical_products=work - 1)
    with pytest.raises(ValueError, match="max_classical_products"):
        execute(problem, limited, "classical")
    assert not calls
    result, run = execute(problem, method.revise(max_classical_products=work), "classical")
    assert len(calls) == 1 and result.eigenvalue is not None
    run.close()

    a = np.array([[0.7, 0.3 + 0.2j], [0.3 - 0.2j, -0.4]])
    one, two = ingest_vector([1, 0]), ingest_vector([1, 1j])
    for basis, expected in (((one, one), 1), ((one, two), 2)):
        calls.clear()
        actions = []
        # (A - cI) acts once per distinct state, whichever route removes c.
        action = fixed_basis._offset_free_actions
        with monkeypatch.context() as guard:
            guard.setattr(
                fixed_basis,
                "state_direction",
                lambda *a, **k: (calls.append(1), original(*a, **k))[1],
            )
            guard.setattr(
                fixed_basis, "_offset_free_actions",
                lambda operator, columns, *a, **k: (actions.extend(columns.T),
                                                    action(operator, columns, *a, **k))[1],
            )
            result, run = execute(Eigenproblem(A=a), FixedGCIM(basis=basis), "classical")
        assert len(calls) == len(actions) == expected
        v = np.column_stack([original(s) for s in basis])
        h = [[complex(z.real, z.imag) for z in row] for row in result.pencil.hamiltonian]
        s = [[complex(z.real, z.imag) for z in row] for row in result.pencil.overlap]
        np.testing.assert_allclose(h, v.conj().T @ a @ v, rtol=0, atol=2e-14)
        np.testing.assert_allclose(s, v.conj().T @ v, rtol=0, atol=2e-14)
        assert result.pencil.kept_rank == expected
        run.close()


def test_sparse_three_dimensional_adapt_preserves_embedding(monkeypatch):
    from scipy.sparse import csr_matrix, csc_matrix
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.algorithms import _eigen_inputs

    a = np.diag([1.0, -1.0, 2.0])
    generator = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for sparse in (csr_matrix, csc_matrix):
        monkeypatch.setattr(sparse, "toarray", lambda *a, **k: pytest.fail("sparse densification"))
        method = ADAPT(
            initial_state=[1, 1, 0],
            pool=(sparse(generator),),
            input_conversion="dense_pauli",
            max_iterations=1,
        )
        result, run = execute(Eigenproblem(A=sparse(a)), method, "classical")
        assert result.eigenvalue == pytest.approx(-1, rel=0, abs=3e-12)
        assert result.plan.problem.dimension == 3
        assert result.plan._native["inputs"].hamiltonian.reference.representation in ("csr", "csc")
        assert result.plan._native["reference"].basis.dimension == 4
        run.close()
    # Exact zero pruning cannot become a coefficient cutoff.
    terms = dict(_eigen_inputs.pauli_coefficients(np.array([[0.0, 1e-30], [1e-30, 0.0]])))
    assert terms == {"X": 1e-30}


def test_supplied_preparation_products_and_declared_alias_preservation(monkeypatch):
    from qiskit import QuantumCircuit
    from test_gcim_native_preparations import reference_kwargs
    from nwqlib.algorithms._eigen_inputs import preparation_requirements
    from nwqlib.algorithms.lanczos import method as lanczos_method

    circuit = QuantumCircuit(1)
    circuit.h(0)
    seed = reference_kwargs(circuit)["reference"]
    assert preparation_requirements(seed)[1] == 4  # 2 amplitudes x 2 gate columns.
    problem = Eigenproblem(A=[[1.0, 0.0], [0.0, -1.0]])
    method = Lanczos(initial_state=seed, krylov_dimension=2)
    plan = method.plan(
        problem,
        output=problem.default_output(),
        execution="classical",
        shots=None,
        rng=RandomStreams(4),
    )
    _, work = method._classical_requirements(problem.A, seed, plan.reconstruction)
    with monkeypatch.context() as guard:
        guard.setattr(
            lanczos_method, "state_direction", lambda *a, **k: pytest.fail("early simulation")
        )
        with pytest.raises(ValueError, match="max_classical_products"):
            execute(problem, method.revise(max_classical_products=work - 1), "classical")
    result, run = execute(problem, method.revise(max_classical_products=work), "classical")
    assert result.eigenvalue == pytest.approx(-1.0, rel=0, abs=2e-12)
    run.close()
    # The same declared InputRef does not establish two distinct circuits equal.
    other = QuantumCircuit(1)
    other.x(0)
    other_seed = reference_kwargs(other)["reference"]
    result, run = execute(problem, FixedGCIM(basis=(seed, other_seed)), "classical")
    assert result.pencil.kept_rank == 2
    assert result.eigenvalue == pytest.approx(-1.0, rel=0, abs=2e-12)
    run.close()
    composite = QuantumCircuit(1)
    composite.append(circuit.to_gate(), [0])
    assert preparation_requirements(reference_kwargs(composite)["reference"])[1] is None
    from qiskit.circuit import Gate

    named = QuantumCircuit(1)
    custom = Gate("h", 1, [])
    custom.definition = circuit
    named.append(custom, [0])
    assert preparation_requirements(reference_kwargs(named)["reference"])[1] is None


def test_matrix_energy_shift_preserves_original_input_relation(monkeypatch, tmp_path):
    from scipy.sparse import csr_matrix, csc_matrix
    from nwqlib.evidence.energy_shift import EnergyShiftOptions
    from nwqlib.saved_evidence import load_result
    from nwqlib.algorithms import _eigen_inputs

    # Binary-exact diagonal shift and complex off-diagonal preservation.
    matrix = np.array([[1.0, 0.25 + 0.5j], [0.25 - 0.5j, -1.0]])
    for index, storage in enumerate((np.array, csr_matrix, csc_matrix)):
        method = Lanczos(initial_state=[1, 0], krylov_dimension=2, overlap_cutoff=1e-14)
        first, first_run = execute(Eigenproblem(A=storage(matrix)), method, "classical")
        second, second_run = execute(
            Eigenproblem(A=storage(matrix + 2 * np.eye(2))), method, "classical"
        )
        assert first.plan.reconstruction.center != second.plan.reconstruction.center
        # No default endpoint table is cached in either solve.
        assert "matrix_entries" not in first.plan._native
        location = second.save(tmp_path / str(index))
        second = load_result(location, method=Lanczos)

        def forbidden(*args, **kwargs):
            pytest.fail("explicit input relation triggered new science")

        with monkeypatch.context() as guard:
            guard.setattr(_eigen_inputs, "pauli_coefficients", forbidden)
            guard.setattr(np.linalg, "eigh", forbidden)
            guard.setattr(np.linalg, "eigvalsh", forbidden)
            options = EnergyShiftOptions.for_result(first, name="shift", shift=2.0, tolerance=2e-12)
            assert options.relation == "matrix_entries" and options.baseline.terms is None
            options = EnergyShiftOptions.model_validate_json(options.model_dump_json())
            receipt, facts = second.verify(checks=options)
        assert receipt.result_id == second.content_id
        value = facts[0].fact.value
        assert value.numerator / value.denominator < 2e-12
        # Same spectrum is insufficient: the exact original input relation must hold.
        changed = matrix + 2 * np.eye(2)
        changed[0, 1] *= -1
        changed[1, 0] *= -1
        wrong, wrong_run = execute(Eigenproblem(A=storage(changed)), method, "classical")
        with pytest.raises(ValueError, match="exact H\\+cI relation"):
            wrong.verify(checks=options)
        for run in (first_run, second_run, wrong_run):
            run.close()
    # Original 3D and implicit sparse diagonal zeros participate as coordinates,
    # including FixedGCIM's same explicit endpoint consumer.
    first, run1 = execute(
        Eigenproblem(A=csr_matrix(np.diag([0.0, 1.0, 2.0]))),
        FixedGCIM(basis=([1, 0, 0],)),
        "classical",
    )
    second, run2 = execute(
        Eigenproblem(A=csc_matrix(np.diag([2.0, 3.0, 4.0]))),
        FixedGCIM(basis=([1, 0, 0],)),
        "classical",
    )
    options = EnergyShiftOptions.for_result(
        first, name="three-dimensional", shift=2.0, tolerance=0.0
    )
    _, facts = second.verify(checks=options)
    assert facts[0].fact.value.numerator == 0
    run1.close()
    run2.close()


def test_energy_endpoint_byte_laws_count_table_scalars_and_relation_workspace():
    import json
    from scipy.sparse import csr_matrix
    from nwqlib.evidence import energy_shift
    from nwqlib.operators import ingest_dense, ingest_pauli, ingest_sparse

    # Capture (energy_shift._operator_payload): borrowed storage plus, per
    # scanned entry, four 8-byte scalars and two copies of at most
    # 6 + 2*digits + 48 bytes of identity JSON, or 3q + 68 bytes per Pauli
    # term, plus 2 bytes of brackets per JSON copy; a matrix adds the
    # extraction workspace 41n + 24D + 65536 (dense input omits 24D).
    # -1.2345678901234567e-308 has the longest binary64 repr, so the entries
    # and the terms below fill the JSON allowance up to one byte.
    x = -1.2345678901234567e-308
    dense = ingest_dense(np.array([[x + 1j * x, x + 1j * x], [x + 1j * x, x + 1j * x]]))
    diagonal = ingest_sparse(csr_matrix(np.diag([0.0, x + 1j * x, x + 1j * x, 0.0])))
    for operator, scanned, index_bytes in ((dense, 4, 0), (diagonal, 2, 24 * 4)):
        text = 6 + 2 * len(str(operator.basis.dimension - 1)) + 2 * 24
        law = operator.manifest.payload_bytes + scanned * (32 + 2 * text) + 4 + 41 * scanned + index_bytes + 65536
        entries = energy_shift._operator_payload(operator, max_bytes=law)["matrix_entries"]
        # Record.content_id encodes with compact separators and hashes UTF-8 bytes.
        encoded = json.dumps([list(entry) for entry in entries], separators=(",", ":"), ensure_ascii=False)
        assert len(encoded.encode("utf-8")) == scanned * text + 1
        with pytest.raises(ValueError, match="endpoint entries needs"):
            energy_shift._operator_payload(operator, max_bytes=law - 1)
    for terms, q in (((("XZ", x), ("II", x)), 2), (((("X" + "I" * 19), x),), 20)):
        pauli = ingest_pauli(terms, num_qubits=q)
        law = pauli.manifest.payload_bytes + len(terms) * (3 * q + 68) + 4
        captured = energy_shift._operator_payload(pauli, max_bytes=law)["terms"]
        assert captured == terms
        encoded = json.dumps([list(term) for term in captured], separators=(",", ":"), ensure_ascii=False)
        assert len(encoded.encode("utf-8")) == len(terms) * (q + 30) + 1
        with pytest.raises(ValueError, match="endpoint terms needs"):
            energy_shift._operator_payload(pauli, max_bytes=law - 1)

    # Relation workspace (energy_shift._relation_bytes): 192U + 65536 for a
    # matrix relation of U stored entries, and [16 max(1,q) + 256]N + 65536
    # for a Pauli relation of N = U + 1 contributions.
    matrix = np.array([[1.0, 0.25 + 0.5j], [0.25 - 0.5j, -1.0]])
    method = Lanczos(initial_state=[1, 0], krylov_dimension=2, overlap_cutoff=1e-14)
    first, first_run = execute(Eigenproblem(A=matrix), method, "classical")
    second, second_run = execute(Eigenproblem(A=matrix + 2 * np.eye(2)), method, "classical")
    baseline = energy_shift.EnergyShiftOptions.for_result(first, name="shift", shift=2.0, tolerance=0.0).baseline
    target = energy_shift._endpoint(second)
    law = 192 * 8 + 65536  # Four entries in each endpoint.
    energy_shift._matrix_relation(baseline, target, 2.0, max_bytes=law)
    with pytest.raises(ValueError, match="matrix relation entries needs"):
        energy_shift._matrix_relation(baseline, target, 2.0, max_bytes=law - 1)
    pauli_base = baseline.revise(matrix_entries=None, terms=(("I", 1.0), ("Z", 0.5)))
    pauli_target = baseline.revise(matrix_entries=None, terms=(("I", 3.0), ("Z", 0.5)))
    law = (16 + 256) * 5 + 65536
    energy_shift._pauli_relation(pauli_base, pauli_target, 2.0, max_bytes=law)
    with pytest.raises(ValueError, match="Pauli relation entries needs"):
        energy_shift._pauli_relation(pauli_base, pauli_target, 2.0, max_bytes=law - 1)
    first_run.close()
    second_run.close()


def test_energy_shift_relations_decide_the_exact_binary64_relation():
    """The stored-table relation equals exact Fraction arithmetic on the recorded values.

    Owners: energy_shift._triple_zero, _pauli_relation and _matrix_relation.
    The cases separate the exact relation from rounded comparisons:
    t == b + c accepts t = b = 1, c = eta.
    """
    from collections import defaultdict
    from fractions import Fraction
    from nwqlib.evidence import energy_shift

    eta, big = 5e-324, float(np.finfo(float).max)
    method = Lanczos(initial_state=[1, 0], krylov_dimension=2, overlap_cutoff=1e-14)
    result, run = execute(Eigenproblem(A=np.diag([1.0, 2.0])), method, "classical")
    endpoint = energy_shift._endpoint(result)
    run.close()

    def exact_pauli(base, target, shift):
        table = defaultdict(Fraction)
        for label, value in target:
            table[label] += Fraction(value)
        for label, value in base:
            table[label] -= Fraction(value)
        table["I"] -= Fraction(shift)
        return all(value == 0 for value in table.values())

    def exact_matrix(base, target, shift):
        table = defaultdict(Fraction)
        for index in range(2):
            table[index, index, "real"] -= Fraction(shift)
        for sign, entries in ((1, target), (-1, base)):
            for row, column, real, imag in entries:
                table[row, column, "real"] += sign * Fraction(real)
                table[row, column, "imag"] += sign * Fraction(imag)
        return all(value == 0 for value in table.values())

    def decided(relation, base, target, shift):
        try:
            relation(base, target, shift)
        except ValueError as error:
            assert "do not establish" in str(error)
            return False
        return True

    pauli = lambda base, target, shift: energy_shift._pauli_relation(  # noqa: E731
        endpoint.revise(matrix_entries=None, terms=base), endpoint.revise(matrix_entries=None, terms=target),
        shift, max_bytes=10**6)
    matrix = lambda base, target, shift: energy_shift._matrix_relation(  # noqa: E731
        endpoint.revise(matrix_entries=base), endpoint.revise(matrix_entries=target), shift, max_bytes=10**6)
    pauli_cases = [
        ((("I", 1.0),), (("I", 1.0),), eta, False),  # t = b = 1, c = eta: residual -eta.
        ((("I", 1.0),), (("I", 3.0),), 2.0, True),
        ((("Z", 0.5),), (("Z", 0.5),), 0.0, True),  # Pair (-0.5, 0.5): -0.5 == -0.5.
        ((("Z", 0.5),), (("Z", 0.25),), 0.0, False),  # Pair (-0.5, 0.25): -0.5 != -0.25.
        ((("I", 1.0),), (), -1.0, True),  # Identity pair (-1, 1): -1 == -1.
        ((), (("Z", 0.5),), 0.0, False),
        ((("Z", -0.0),), (), 0.0, True),
    ]
    matrix_cases = [
        (((0, 0, eta, 0.0), (1, 1, eta, 0.0)), ((0, 0, 2 * eta, 0.0), (1, 1, 2 * eta, 0.0)), eta, True),
        (((0, 0, eta, 0.0), (1, 1, eta, 0.0)), ((0, 0, 2 * eta, 0.0), (1, 1, 2 * eta, 0.0)), 2 * eta, False),
        (((0, 0, -big, 0.0), (1, 1, -big, 0.0)), ((0, 0, big, 0.0), (1, 1, big, 0.0)), big, False),
        (((0, 0, 1.0, 0.0),), ((0, 0, 2.0, 0.0),), 1.0, False),  # Missing diagonal (1, 1).
        (((0, 1, 0.5, 0.25),), ((0, 1, 0.5, 0.25), (0, 0, 1.0, 0.0), (1, 1, 1.0, 0.0)), 1.0, True),
        (((0, 1, 0.5, 0.25),), ((0, 1, 0.5, -0.25),), 0.0, False),
        ((), (), 0.0, True),
        ((), (), 1.0, False),
        # Zero shift with equal imaginary parts and no off-diagonal keys, so
        # only the diagonal real parts decide.
        (((0, 0, 1.0, 0.0), (1, 1, 1.0, 0.0)), ((0, 0, 2.0, 0.0), (1, 1, 1.0, 0.0)), 0.0, False),
    ]
    for base, target, shift, expected in pauli_cases:
        assert exact_pauli(base, target, shift) is expected
        assert decided(pauli, base, target, shift) is expected
    for base, target, shift, expected in matrix_cases:
        assert exact_matrix(base, target, shift) is expected
        assert decided(matrix, base, target, shift) is expected
    # Coordinates beyond int64 are merged as the endpoint's own Python integers.
    wide = endpoint.basis.revise(dimension=2**70)
    big = ((2**69, 2**69 + 1, 0.5, 0.0), (2**69, 2**69, 1.0, 0.0))
    wide_relation = lambda base, target, shift: energy_shift._matrix_relation(  # noqa: E731
        endpoint.revise(basis=wide, matrix_entries=base), endpoint.revise(basis=wide, matrix_entries=target),
        shift, max_bytes=10**6)
    # 2**69 + 1 and 2**69 round to one binary64 value, so float64 coordinates
    # would merge the last pair's distinct keys and accept it.
    for base, target, shift, expected in ((big, big, 0.0, True), (big, big, 1.0, False),
                                          (big[:1], ((2**69, 2**69, 0.5, 0.0),), 0.0, False)):
        assert decided(wide_relation, base, target, shift) is expected
