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
    for criterion in (dict(overlap_cutoff_policy="confidence", overlap_failure_probability=delta),
                      dict(overlap_cutoff=1e-9), dict()):
        original, reopened = result.analyze(**criterion), loaded.analyze(**criterion)
        assert (reopened.eigenvalue, reopened.kept_rank, reopened.cutoff) == (
            original.eigenvalue, original.kept_rank, original.cutoff)
    strict = loaded.analyze(overlap_cutoff_policy="confidence", overlap_failure_probability=delta)
    # Machine precision: the same closed form evaluated in binary64.
    assert strict.cutoff == pytest.approx(2*2*np.sqrt(2*np.log(2*2/delta)/4096), rel=2e-15)


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


@pytest.mark.parametrize("mode,shots", [("quantum", None)])
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
