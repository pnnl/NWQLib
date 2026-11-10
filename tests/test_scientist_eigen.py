"""Independent two-qubit eigenvalue witnesses on the selected public Method path."""

import math

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


def test_same_target_different_initialization():
    p = Eigenproblem(A=[[1, 1], [1, 1]])
    for method in (
        Lanczos(initial_state=[1, 0], krylov_dimension=2),
        FixedGCIM(basis=([1, 1], [1, -1])),
    ):
        result, run = execute(p, method, "quantum")
        assert result.eigenvalue == pytest.approx(0, abs=1e-12, rel=0)
        assert not all(e.execution == "host_kernel" for e in run.trace.events)
        again = result.analyze(overlap_cutoff=1e-9)
        assert again.data is result.data and again.plan is result.plan
        assert again.eigenvalue == pytest.approx(result.eigenvalue, abs=1e-12, rel=0)
        assert len(run.trace.events) == len(result.data.trace.events)


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
    strict = result.analyze(cutoff_policy="confidence", overlap_fail_prob=.1)
    rec = result.plan.reconstruction
    counts = [stat.shots for degree, stat in strict.statistics if degree < 2*rec.krylov_dimension-1]
    expected = 2*rec.krylov_dimension*np.sqrt(2*np.log(2*len(counts)/.1)/min(counts))
    assert strict.cutoff == pytest.approx(expected, rel=2e-15)
    assert strict.analysis_failure_probability == .1 and strict.cutoff_source == "hoeffding_gram_bound"
    # The guide's default regularization max(1e-12, noise_multiplier*RMS) at a changed multiplier.
    looser = result.analyze(noise_multiplier=3.)
    assert looser.analysis_noise_multiplier == 3. and looser.cutoff_source == "empirical_gram_rms"
    assert looser.cutoff == max(1e-12, 3.*result.empirical_gram_noise_frobenius_rms)
    with pytest.raises(ValueError, match="accepts overlap_cutoff, cutoff_policy, noise_multiplier and overlap_fail_prob"):
        result.analyze(overlap_cutoff_policy="confidence")
    assert run.rng.snapshot() == old_state and len(run.trace.events) == 4
    from nwqlib.algorithms.lanczos import numerical
    monkeypatch.setattr(numerical, "_projected_matrices",
                        lambda *a, **k: pytest.fail("cutoff validation rebuilt a matrix"))
    strict.validate_plan(result.plan)
    with pytest.raises(ValueError, match="cutoff"):
        strict.revise(cutoff=strict.cutoff*1.01).validate_plan(result.plan)


def test_saved_sampled_lanczos_result_reanalyzes_with_its_shot_counts(tmp_path):
    """A reloaded sampled Result gives the original Ritz values under changed cutoff criteria.

    The confidence cutoff 2 m sqrt(2 log(2 r/delta)/n_min) (Hoeffding, see
    the Lanczos guide) reads the shot counts of the r = 2m - 2 = 2 sampled Gram
    moments, so it reveals counts lost or changed by saving.
    """
    import nwqlib
    from nwqlib.operators import ingest_pauli

    problem = Eigenproblem(A=ingest_pauli((("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)), num_qubits=2))
    result = nwqlib.solve(nwqlib.plan(problem, method=Lanczos(initial_state=[1, 0, 0, 0], krylov_dimension=2),
                                      shots=4096, seed=7))
    loaded = nwqlib.load_result(result.save(tmp_path / "sampled"))
    counts = [stat.shots for degree, stat in loaded.statistics if degree < 3]
    assert counts == [4096, 4096]
    delta = 0.05
    for criterion in (dict(cutoff_policy="confidence", overlap_fail_prob=delta),
                      dict(overlap_cutoff=1e-9), dict()):
        original, reopened = result.analyze(**criterion), loaded.analyze(**criterion)
        assert (reopened.eigenvalue, reopened.kept_rank, reopened.cutoff) == (
            original.eigenvalue, original.kept_rank, original.cutoff)
    strict = loaded.analyze(cutoff_policy="confidence", overlap_fail_prob=delta)
    # Machine precision: the same closed form evaluated in binary64.
    assert strict.cutoff == pytest.approx(2*2*np.sqrt(2*np.log(2*2/delta)/4096), rel=2e-15)


# Projected analysis bytes of numerical.reconstruct at krylov_dimension m=2.
ANALYSIS_BYTES = 512 * 2 * 2 + 128 * 2


def test_lanczos_plan_admits_projected_analysis_before_acquisition():
    """The projected analysis runs after every moment is acquired, but its bytes depend only on m.

    One byte below them stops plan(), so no Run and no acquisition exist. At
    the bound the analysis admission passes and planning stops at the fixed
    65,536-byte allowance of the Pauli census or the Gershgorin frame
    (ENGINEERING_CONSTANTS "Chebyshev Lanczos defaults" and "Eigen input
    conversion and classical preparation"), so the run completes under the
    default byte limit.
    """
    problem = Eigenproblem(A=[[1, 1], [1, 1]])

    def plan_method(max_bytes):
        method = Lanczos(initial_state=[1, 0], krylov_dimension=2, max_bytes=max_bytes)
        return method.plan(
            problem, output=problem.default_output(), execution="quantum", shots=None, rng=RandomStreams(4)
        )

    with pytest.raises(ValueError, match=f"Lanczos projected analysis needs about {ANALYSIS_BYTES} bytes"):
        plan_method(ANALYSIS_BYTES - 1)
    with pytest.raises(ValueError, match="Lanczos Pauli census|Gershgorin frame workspace"):
        plan_method(ANALYSIS_BYTES)
    result, run = execute(problem, Lanczos(initial_state=[1, 0], krylov_dimension=2), "quantum", None)
    assert result.eigenvalue is not None
    assert len(run.trace.events) == len(result.plan.experiments)


def test_classical_recurrence_is_checked_once_at_planning():
    """A classical Lanczos Plan is refused at planning when its recurrence exceeds a limit, and runs at the limit.

    For H = Z_0 + 0.5 X_1 + 0.25 Z_0 Z_1 on 12 qubits from |0...0> with
    three Krylov directions, the bytes and products of the reference and the
    Chebyshev recurrence on 4,096 amplitudes are known before any Run and
    exceed the other planning checks. Half the bytes or half the products
    refuses the Plan, and Plans at exactly those limits run. From |0...0>
    the trial space holds |0...0> and qubit 1 flipped, where H is
    [[1.25, 0.5], [0.5, 0.75]], so the projected value is 1 - sqrt(0.3125).
    """
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation

    n = 12
    terms = (("I" * (n - 1) + "Z", 1.0), ("I" * (n - 2) + "XI", 0.5), ("I" * (n - 2) + "ZZ", 0.25))
    problem = Eigenproblem(A=ingest_pauli(terms, num_qubits=n))

    def plan(**limits):
        method = Lanczos(initial_state=ingest_occupation("0" * n, num_qubits=n), krylov_dimension=3, **limits)
        return method.plan(problem, output=problem.default_output(), execution="classical", shots=None,
                           rng=RandomStreams(4))

    free = plan()
    need_bytes, need_work = free.method._classical_requirements(free._native["operator"], free._native["reference"],
                                                                free.reconstruction)
    with pytest.raises(ValueError, match=f"classical Chebyshev preparation and recurrence needs about {need_bytes}"):
        plan(max_bytes=need_bytes // 2)
    with pytest.raises(ValueError, match=rf"require {need_work} classical products, above "
                                         rf"max_products={need_work // 2}\. Raise max_products "
                                         rf"to at least {need_work} or set it to None\."):
        plan(max_products=need_work // 2)
    for limits in ({"max_bytes": need_bytes}, {"max_products": need_work}):
        result = Run(plan(**limits)).wait(timeout=30, poll_interval=0)
        assert result.eigenvalue == pytest.approx(1 - np.sqrt(0.3125), abs=1e-12, rel=0)


def test_adapt_commutator_actual_decision_and_reanalysis():
    from nwqlib.algorithms.gcim.adapt import ADAPT
    from nwqlib.operators import ingest_pauli
    from nwqlib.algorithms.gcim.adapt_verification import AdaptVerificationOptions

    p = Eigenproblem(A=[[1.0, 0], [0, -1.0]])
    method = ADAPT(initial_state=[1, 1], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),))
    plan = method.plan(
        p, output=p.default_output(), execution="quantum", shots=None, rng=RandomStreams(3)
    )
    assert plan.reconstruction.pool[0].commutator.rows() == (("X", 2.0),)
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


def test_interrupted_analysis_retry_keeps_physical_population_and_rng(monkeypatch):
    """Explicit analysis recovery adds one analysis attempt while preserving acquired populations
    and the RNG trajectory.
    """
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


@pytest.mark.parametrize("storage", ["dense", "pauli"])
def test_original_matvec_affine_frame_and_saved_ritz(storage, monkeypatch, tmp_path):
    from nwqlib.operators import ingest_pauli
    from nwqlib.algorithms import _eigen_inputs
    from nwqlib.saved_evidence import load_result

    # A²=2.09 I gives an independent spectrum. |0>, A|0> span C²;
    # Gershgorin and Pauli L1 frames have different raw moments but same Ritz roots.
    matrix = np.array([[0.3, 1 - 1j], [1 + 1j, -0.3]])
    inputs = {
        "dense": matrix,
        "pauli": ingest_pauli((("X", 1), ("Y", 1), ("Z", 0.3)), num_qubits=1),
    }
    monkeypatch.setattr(
        _eigen_inputs, "pauli_coefficient_masks", lambda *a: pytest.fail("implicit conversion")
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
    x, z, coefficients = _eigen_inputs.pauli_coefficient_masks(np.array([[0.0, 1e-30], [1e-30, 0.0]]),
                                                               max_bytes=None)
    assert (x.tolist(), z.tolist(), coefficients.tolist()) == ([[1]], [[0]], [1e-30])


def test_dense_custom_adapt_generator_keeps_the_record_of_its_projected_labels(monkeypatch):
    """A dense custom ADAPT generator is projected on its packed words and keeps the label route's records.

    The expected member projects the labels of ``pauli_coefficients`` with
    ``_project_terms`` and ingests them: the stored terms, the processed
    operator and the symmetry evidence agree, for a generator with Y terms
    and a small Hermitian part removed by the projection. A term with a
    negative imaginary coefficient has a negative-zero real part, which the
    member record keeps and the processed operator's content hash does not.
    Planning sums no identical words, since the generator's words are
    distinct.
    """
    import nwqlib
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.algorithms._eigen_inputs import dense_pauli_matrix
    from nwqlib.algorithms.gcim.adapt_inputs import _project_terms
    from nwqlib.algorithms.gcim.adapt_records import AdaptSymmetry
    from nwqlib.core.records import Complex128
    from nwqlib.operators import ingest_dense, ingest_pauli, inputs
    from nwqlib.operators._pauli import pauli_coefficients

    rng = np.random.default_rng(11)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hermitian = rng.normal(size=(4, 4))
    generator = ingest_dense((a - a.conj().T) + 1e-13 * (hermitian + hermitian.T))
    method = ADAPT(initial_state=[1, 0, 0, 0], pool=(generator,), input_conversion="dense_pauli",
                   max_iterations=1, symmetry_tol=1e-10)
    monkeypatch.setattr(inputs, "_coalesced_input", lambda table: pytest.fail("distinct words summed again"))
    plan = nwqlib.plan(Eigenproblem(A=ingest_dense(np.diag([1.0, -1.0, 0.5, 2.0]))), method=method,
                       execution="classical", seed=3)
    member = plan._native["inputs"].members[0]
    monkeypatch.undo()
    matrix = dense_pauli_matrix(generator, input_conversion="dense_pauli", max_bytes=None)
    rows, evidence = _project_terms(pauli_coefficients(matrix), antihermitian=True, tolerance=1e-10, qubits=2)
    assert any("Y" in label for label, _ in rows) and evidence["correction_applied"]
    assert any(c.imag < 0 and math.copysign(1.0, c.real) < 0 for _, c in rows)
    assert member.terms == tuple((label, Complex128(real=c.real, imag=c.imag)) for label, c in rows)
    assert member.processed_input == ingest_pauli(rows, num_qubits=2).reference
    assert member.symmetry == AdaptSymmetry(**evidence)


@pytest.mark.parametrize("d", [5, 8])
def test_dense_pauli_conversion_ingests_masks_with_the_operator_of_its_labels(d):
    """Converting a dense Hermitian matrix through packed masks gives the operator of its labels.

    A random complex Hermitian d-by-d matrix, padded to 8 with its mean
    diagonal for d = 5, is converted with ``dense_pauli``. The operator has
    the content hash and the terms of ingesting the labelled block
    transform, Y terms and their signs included. The transform's words are
    distinct, so the conversion sums no identical words.
    """
    from nwqlib.algorithms._eigen_inputs import dense_pauli_matrix, eigen_operator
    from nwqlib.operators import ingest_dense, ingest_pauli, inputs
    from nwqlib.operators._pauli import pauli_coefficients

    rng = np.random.default_rng(d)
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    problem = Eigenproblem(A=ingest_dense(a + a.conj().T))
    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(inputs, "_coalesced_input", lambda table: pytest.fail("distinct words summed again"))
        operator = eigen_operator(problem, problem.default_output(), input_conversion="dense_pauli")
    matrix = dense_pauli_matrix(problem.A, input_conversion="dense_pauli", max_bytes=None, padding="mean")
    labels = pauli_coefficients(matrix)
    expected = ingest_pauli(((label, c.real) for label, c in labels), num_qubits=3)
    assert any("Y" in label for label, _ in labels)
    assert operator.reference == expected.reference
    assert operator.pauli_terms().labels() == expected.pauli_terms().labels()


def test_dense_pauli_conversion_checks_the_transform_then_the_packed_words():
    """The conversion bytes are the transform's 64*D**2, then 32*D**2 + 64*R once R words are known.

    A single-Z dense Hamiltonian on 6 qubits has R = 1, so the transform
    binds: the public quantum Lanczos Plan is built at 64*D**2 bytes and
    refused one byte below. A random dense Hermitian matrix on 3 qubits has
    R = D**2 words, so packing them binds above the transform. R comes from
    the labelled block transform.
    """
    import nwqlib
    from nwqlib.operators import ingest_dense
    from nwqlib.operators._pauli import pauli_coefficient_masks, pauli_coefficients
    from nwqlib.problems import ingest_occupation

    q = 6
    d = 1 << q
    problem = Eigenproblem(A=ingest_dense(np.diag(np.tile([1.0, -1.0], d // 2))))
    state = ingest_occupation("0" * q, num_qubits=q)

    def build(cap):
        return nwqlib.plan(problem, method=Lanczos(initial_state=state, krylov_dimension=2,
                                                   input_conversion="dense_pauli", max_bytes=cap),
                           execution="quantum", seed=6)

    build(64 * d * d)
    with pytest.raises(ValueError, match="eigenvalue Pauli conversion"):
        build(64 * d * d - 1)

    rng = np.random.default_rng(3)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    matrix = a + a.conj().T
    words = len(pauli_coefficients(matrix))
    need = 32 * 8 * 8 + 64 * words
    assert words == 64 and need > 64 * 8 * 8
    x, z, coefficients = pauli_coefficient_masks(matrix, max_bytes=need)
    assert len(coefficients) == words
    with pytest.raises(ValueError, match="Pauli mask packing"):
        pauli_coefficient_masks(matrix, max_bytes=need - 1)


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
        with pytest.raises(ValueError, match="max_products"):
            execute(problem, method.revise(max_products=work - 1), "classical")
    result, run = execute(problem, method.revise(max_products=work), "classical")
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
            guard.setattr(_eigen_inputs, "pauli_coefficient_masks", forbidden)
            guard.setattr(np.linalg, "eigh", forbidden)
            guard.setattr(np.linalg, "eigvalsh", forbidden)
            options = EnergyShiftOptions.for_result(first, name="shift", shift=2.0, tolerance=2e-12)
            assert options.relation == "matrix_entries"
            # A dense input with no zero entry keeps all entries without coordinates.
            assert options.baseline.table == ("dense_matrix" if storage is np.array else "matrix_entries")
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


def test_energy_shift_relations_decide_the_exact_binary64_relation():
    """The stored-table relation decides each hand-derived exact case on the recorded values.

    Owners: energy_shift._triple_zero and _relation.
    The cases separate the exact relation from rounded comparisons:
    t == b + c accepts t = b = 1, c = eta.
    """
    from nwqlib.core.records import FrozenArray
    from nwqlib.evidence import energy_shift

    eta, big = 5e-324, float(np.finfo(float).max)
    method = Lanczos(initial_state=[1, 0], krylov_dimension=2, overlap_cutoff=1e-14)
    result, run = execute(Eigenproblem(A=np.diag([1.0, 2.0])), method, "classical")
    endpoint = energy_shift._endpoint(result)
    run.close()

    def decided(relation, base, target, shift):
        try:
            relation(base, target, shift)
        except ValueError as error:
            assert "do not establish" in str(error)
            return False
        return True

    def tables(rows, table):
        """Two-column keys and (real, imag) values. A one-qubit Pauli key is its X and Z mask.

        A dense table holds the four entries of the 2-by-2 matrix in row-major order.
        """
        if table == "dense_matrix":
            values = np.zeros((4, 2))
            for row, column, real, imag in rows:
                values[2 * row + column] = real, imag
            return endpoint.revise(table=table, table_keys=None, table_values=FrozenArray(values))
        masks = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
        keys = [masks[row[0]] for row in rows] if table == "pauli_terms" else [row[:2] for row in rows]
        values = [(row[1], 0.0) for row in rows] if table == "pauli_terms" else [row[2:] for row in rows]
        return endpoint.revise(table=table, table_keys=FrozenArray(np.array(keys, dtype=np.int64).reshape(-1, 2)),
                               table_values=FrozenArray(np.array(values, dtype=np.float64).reshape(-1, 2)))

    def relation(table, base_table=None, target_table=None):
        return lambda base, target, shift: energy_shift._relation(
            tables(base, base_table or table), tables(target, target_table or table), shift, table, max_bytes=10**6)

    pauli, matrix = relation("pauli_terms"), relation("matrix_entries")
    # Dense tables, alone or against stored entries, decide every matrix case alike.
    dense = [relation("matrix_entries", *kinds) for kinds in (
        ("dense_matrix", None), (None, "dense_matrix"), ("dense_matrix", "dense_matrix"))]
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
        assert decided(pauli, base, target, shift) is expected
    for base, target, shift, expected in matrix_cases:
        assert decided(matrix, base, target, shift) is expected
        assert all(decided(variant, base, target, shift) is expected for variant in dense)


def test_packed_hamiltonian_conversion_charges_its_arrays_not_records_of_dropped_words():
    """The packed conversion charges its coexisting arrays, and the records charge only the terms they keep.

    A 5-qubit diagonal Hamiltonian with a 2**-28 Hermitian perturbation has
    nearly all 1024 Pauli words nonzero, and ADAPT's coefficient cutoff
    keeps one nonidentity word, beside the identity coefficient that stays
    whole. The public quantum Plan is built at 300000 bytes with those terms.
    One byte below the packing of the nonzero words, counted from the
    labelled block transform, refuses at the packing stage.
    """
    import nwqlib
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.operators import ingest_dense, ingest_pauli
    from nwqlib.operators._pauli import pauli_coefficients
    from nwqlib.problems import ingest_occupation

    q, d = 5, 32
    rng = np.random.default_rng(19)
    v = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    matrix = np.diag(np.tile([1.0, -1.0], d // 2)) + 2.0**-28 * (v + v.conj().T)
    problem = Eigenproblem(A=ingest_dense(matrix))
    generator = ingest_pauli((("I" * (q - 1) + "X", 1j),), num_qubits=q)

    def build(cap):
        return nwqlib.plan(problem, method=ADAPT(
            initial_state=ingest_occupation("0" * q, num_qubits=q), pool=(generator,), input_conversion="dense_pauli",
            max_iterations=1, max_bytes=cap, pauli_cutoff=1e-5), execution="quantum", seed=2)

    inputs = build(300_000)._native["inputs"]
    assert len(inputs.nonidentity) == 1 and inputs.dropped_l1 > 0
    words = len(pauli_coefficients(matrix.astype(complex)))
    packing = 32 * d * d + 64 * words
    assert words > 1000 and packing < 300_000
    with pytest.raises(ValueError, match="Pauli mask packing"):
        build(packing - 1)
