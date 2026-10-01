"""Natural-input classical QLS, independent scale/phase/domain and reuse witnesses."""

from dataclasses import replace
from fractions import Fraction
import json
import math
import numpy as np
import pytest
import nwqlib
from nwqlib import (
    LinearSystem,
    Solution,
    StateVector,
    NormSquared,
    QuadraticForm,
    NormalizedExpectation,
)
from nwqlib.algorithms.qls import QLS, QLSVerification
from nwqlib.algorithms.qls import method as owner, numerical
from nwqlib.algorithms.qls.primary_records import InversePolynomial, validate_selection
from nwqlib.core.planning import Plan
from nwqlib.core.records import Float64
from nwqlib.operators import ingest_dense

METHODS = ("qsvt_inverse", "shortcut_native_svp", "shortcut_dilation")
MATRIX = np.array([[1.0, 0.2j, 0.0], [0.1, 1.5, 0.3j], [0.2j, 0.0, 2.0]])
RHS = np.array([1.0, 1j, 0.3])


def forbidden(*args, **kwargs):
    raise AssertionError("unrequested numerical or reference work")


def test_descriptive_rhs_rejects_before_original_svd_or_fit(monkeypatch):
    from nwqlib.problems.inputs import StateInput, ingest_vector

    rhs=StateInput.from_record(ingest_vector(RHS).to_record())
    problem=LinearSystem(A=MATRIX,b=rhs)
    # Classical planning of this non-Hermitian A takes its original SVD here.
    monkeypatch.setattr(np.linalg,"svd",forbidden)
    monkeypatch.setattr(numerical,"_extreme_singular_values",forbidden)
    monkeypatch.setattr(owner,"select_polynomial",forbidden)
    with pytest.raises(ValueError,match="descriptive record"):
        nwqlib.plan(problem,method=QLS(),execution="classical",seed=7)


def selected(*, solver="qsvt_inverse", A=MATRIX, b=RHS, output=None, t=1.5, **fields):
    if output is None:
        output = (
            Solution()
            if solver == "qsvt_inverse"
            else StateVector(normalization="unit", global_phase="modulo_global_phase")
        )
    return nwqlib.plan(
        LinearSystem(A=A, b=b),
        method=QLS(
            solver=solver,
            encoded_solution_norm_estimate=None if solver == "qsvt_inverse" else t,
            **fields,
        ),
        output=output,
        execution="classical",
        seed=7,
    )


def aligned(actual, expected):
    phase = np.vdot(expected, actual)
    return actual * np.conj(phase) / abs(phase)


def facts(receipt):
    return {
        f.fact.quantity: None if f.fact.value is None else f.fact.value.value
        for f in receipt.applications[0].facts
    }


def arguments(receipt):
    return {
        a.parameter: getattr(a.value, "value", a.value) for a in receipt.applications[0].arguments
    }


@pytest.mark.parametrize("solver", METHODS)
def test_original_complex_three_coordinate_models_and_archive(tmp_path, monkeypatch, solver):
    reference = np.linalg.solve(MATRIX, RHS)
    plan = selected(solver=solver)
    assert type(plan) is Plan and plan.method.epsilon_inv == 0.01
    assert plan.problem.dimension == 3 and plan.reconstruction.padded_dimension == 4
    result = nwqlib.solve(plan)
    if solver == "qsvt_inverse":
        assert np.linalg.norm(result.x - reference) / np.linalg.norm(reference) < 0.01
    else:
        expected = reference / np.linalg.norm(reference)
        assert np.linalg.norm(aligned(result.value, expected) - expected) < 0.01
        assert result.physical_scale is result.norm_squared is None
    result.save(tmp_path / "result")
    monkeypatch.setattr(QLS, "plan", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(owner, "select_polynomial", forbidden)
    loaded = nwqlib.load_result(tmp_path / "result")
    assert loaded.plan.content_id == plan.content_id and loaded.content_id == result.content_id
    np.testing.assert_array_equal(loaded.value, result.value)
    # Saved original factors must be those the recorded spectral selection acquired.
    saved = tmp_path / "result" / "result.json"
    content = json.loads(saved.read_text())
    if content["selection"]["selected"]["factors"] is not None:
        content["selection"]["selected"]["factors"] = None
        saved.write_text(json.dumps(content))
        with pytest.raises(ValueError, match="recorded spectral selection"):
            nwqlib.load_result(tmp_path / "result")


def test_original_svd_scale_no_second_padded_decomposition(monkeypatch):
    from nwqlib.algorithms.qls.host_planning import select_inputs
    from nwqlib.algorithms.qls.quantum import BaseEncoding, _native_encoding
    from nwqlib.subroutines.block_encoding import block_encoding_top_left
    from qiskit.quantum_info import Operator

    original = np.linalg.svd
    seen = []

    def svd(matrix, *args, **kwargs):
        seen.append(matrix.shape)
        np.testing.assert_array_equal(matrix, MATRIX / 2)
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", svd)
    problem, method = LinearSystem(A=MATRIX, b=RHS), QLS()
    encoding, operator, rhs, cache, spectrum = select_inputs(problem, method, execution="quantum")
    assert seen == [(3, 3)]
    np.testing.assert_allclose(
        (cache.left * cache.singular) @ cache.right_h, MATRIX, rtol=1e-14, atol=1e-14
    )
    assert spectrum["alpha"] == cache.singular[0] and spectrum["sigma_min"] == cache.singular[-1]
    assert spectrum["kappa_be"] == spectrum["alpha"] / spectrum["sigma_min"]
    padded = operator.dense_array()
    assert padded[3, 3] == spectrum["alpha"] and np.count_nonzero(padded[:3, 3]) == 0
    np.testing.assert_array_equal(rhs.physical_vector()[:3], RHS)
    assert rhs.physical_vector()[3] == 0
    cache.require_source(problem.A)
    base = BaseEncoding(encoding, problem.A, operator, cache, method.max_bytes, method.max_work)
    native = _native_encoding(base)
    np.testing.assert_allclose(block_encoding_top_left(Operator(native.circuit).data,
        num_ancillas=1), padded/spectrum["alpha"], atol=2e-14, rtol=2e-14)
    assert seen == [(3, 3)]
    with pytest.raises(ValueError, match="original input binding"):
        cache.require_source(ingest_dense(MATRIX + np.eye(3)))
    writable = replace(cache, left=np.array(cache.left))
    with pytest.raises(ValueError, match="original input binding"):
        writable.require_source(problem.A)


def test_exact_hermiticity_and_identity_condition_floor_are_separate():
    hermitian = np.array([[1.0, 0.2j], [-0.2j, 2.0]])
    near = hermitian.copy()
    near[0, 1] += 1e-15
    assert selected(A=hermitian, b=[1.0, 1.0]).reconstruction.embedding == "none"
    r = selected(A=near, b=[1.0, 1.0]).reconstruction
    assert r.embedding == "hermitian_dilation" and r.spectral_method == "original_svd"
    r = selected(A=3 * np.eye(2), b=[1.0, 1.0]).reconstruction
    assert (r.alpha, r.kappa_be, r.condition_number, r.polynomial_kappa) == (3.0, 1.0, 1.0, 1.01)


def test_classical_spectral_selection_builds_frames_only_for_their_consumers(monkeypatch):
    from nwqlib.algorithms.qls.host_planning import select_inputs

    original = np.linalg.svd
    calls = []

    def recorded(matrix, *args, **kwargs):
        calls.append(kwargs.get("compute_uv", True))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recorded)
    problem = LinearSystem(A=MATRIX, b=RHS)
    # A shortcut with a numeric t has no factor consumer: singular values only.
    method = QLS(solver="shortcut_native_svp", encoded_solution_norm_estimate=1.0)
    assert select_inputs(problem, method, execution="classical")[3] is None
    assert calls == [False]
    # The inverse keeps the original frames it reuses; its evaluation
    # decomposes nothing more.
    calls.clear()
    plan = selected()
    assert plan._native["svd"].method == "original_svd" and calls == [True]
    nwqlib.solve(plan)
    assert calls == [True]


def test_supplied_premise_disclosure_preserves_classical_no_spectrum_contract(monkeypatch):
    matrix = np.array([[1.1, .1], [.1, .9]])
    problem = LinearSystem(A=matrix, b=[1, .25])
    with monkeypatch.context() as context:
        context.setattr(np.linalg, "svd", forbidden)
        for kappa in (1.05, 1.4):
            result = nwqlib.solve(problem, method=QLS(alpha=1.2, kappa=kappa), execution="classical")
            assert result.plan.reconstruction.sigma_min is None
            assert "supplied spectral premise was not independently checked" in str(result)
    # The original eigenvalues are 1 +/- sqrt(.02), so alpha/sigma_min > 1.05.
    # Dense quantum selection already owns these endpoints and must use them.
    with pytest.raises(ValueError, match="supplied kappa"):
        nwqlib.plan(problem, method=QLS(alpha=1.2, kappa=1.05), execution="quantum")


@pytest.mark.parametrize("scale", [1e-200, 1., 1e200])
def test_numerical_spectral_window_keeps_insufficient_premises_rejected(scale):
    from nwqlib.algorithms.qls.host_planning import _spectrum
    from nwqlib.operators import ingest_dense

    # Diagonal singular values are scale and scale/2: alpha=scale, kappa=2
    # exactly cover the mathematical spectrum at every tested physical scale.
    operator = ingest_dense(scale * np.diag([1., .5]))
    admitted = _spectrum(operator, QLS(alpha=scale, kappa="auto"))
    assert admitted["alpha"] == scale
    assert admitted["kappa_be"] == pytest.approx(2., rel=2e-15, abs=0)
    rounded_kappa = float(np.nextafter(2., 0.))
    supplied = _spectrum(operator, QLS(alpha="auto", kappa=rounded_kappa))
    assert supplied["kappa_be"] == rounded_kappa  # Preserve the selected premise.
    # 1e-8 is well outside the explicit rounding window, but would pass a
    # default np.isclose. Tiny scale also rules out an absolute error floor.
    with pytest.raises(ValueError, match="supplied alpha"):
        _spectrum(operator, QLS(alpha=(1.-1e-8)*scale, kappa="auto"))
    with pytest.raises(ValueError, match="supplied kappa"):
        _spectrum(operator, QLS(alpha="auto", kappa=2.*(1.-1e-8)))


def test_general_spectrum_admits_the_singular_values_without_vectors(monkeypatch):
    from nwqlib.algorithms.qls.host_planning import _spectrum

    d = 64
    matrix = 4*np.eye(d) + .1*np.random.default_rng(3).standard_normal((d, d))
    operator = ingest_dense(matrix)
    assert operator.structure == "general"
    # The SVD without vectors, ceil(4 d**3/3) = 349,526 units for the
    # bidiagonal reduction and 32 d (d + 1) = 133,120 for the rest, and
    # 8 d**2 = 32,768 for the scaling, the Hermitian test and the magnitudes
    # make 515,414 units. A full SVD with both frames would be charged
    # 8 d**3 = 2,097,152.
    law = 515_414
    spectrum = _spectrum(operator, QLS(max_work=law))
    singular = np.linalg.svd(matrix, compute_uv=False)
    assert (spectrum["sigma_min"], spectrum["sigma_max"]) == pytest.approx(
        (singular[-1], singular[0]), rel=1e-13, abs=0)
    assert dict(spectrum["spectral_calls"])["svd_calls"] == 1
    monkeypatch.setattr(numerical, "_extreme_singular_values", forbidden)
    with pytest.raises(ValueError, match="original QLS spectral selection exceeds max_work"):
        _spectrum(operator, QLS(max_work=law - 1))
    # The bidiagonal reduction alone makes ceil(4 d**3/3) = 349,526 multiply-adds.
    with pytest.raises(ValueError, match="original QLS spectral selection exceeds max_work"):
        _spectrum(operator, QLS(max_work=300_000))


def test_reference_spectrum_admits_the_singular_values_it_computes(monkeypatch):
    d = 16
    rng = np.random.default_rng(5)
    matrix = 4*np.eye(d) + .1*(rng.standard_normal((d, d)) + 1j*rng.standard_normal((d, d)))
    singular = np.linalg.svd(matrix, compute_uv=False)
    alpha = 1.01*singular[0]
    # Supplied alpha and kappa leave the Plan without singular endpoints.
    result = nwqlib.solve(LinearSystem(A=matrix, b=np.ones(d)),
        method=QLS(alpha=alpha, kappa=1.01*alpha/singular[-1]), execution="classical", seed=7)
    assert result.plan.reconstruction.sigma_min is None
    # The SVD without vectors, ceil(4 d**3/3) = 5462 units for the bidiagonal
    # reduction and 32 d (d + 1) = 8704 for the rest, and 3 d**2 = 768 units
    # of margin around the call.
    law = 14_934
    receipt = result.verify(checks=QLSVerification(comparisons=("spectral_domain",), max_work=law))[0]
    assert arguments(receipt)["svd_completed"] == 1 and arguments(receipt)["known_work"] == law
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    with pytest.raises(ValueError, match="max_work"):
        result.verify(checks=QLSVerification(comparisons=("spectral_domain",), max_work=law - 1))
    # The bidiagonal reduction alone makes ceil(4 d**3/3) = 5462
    # multiply-adds. A charge of d**3 + 3 d**2 = 4864 for the singular values
    # would admit this cap.
    with pytest.raises(ValueError, match="max_work"):
        result.verify(checks=QLSVerification(comparisons=("spectral_domain",), max_work=5000))


@pytest.mark.parametrize(("solver", "t", "hermitian", "cap"), [("qsvt_inverse", None, False, 800_000),
                                                               ("qsvt_inverse", None, True, 400_000),
                                                               ("shortcut_native_svp", 1.0, False, 6_500_000),
                                                               ("shortcut_dilation", 1.0, False, 25_000_000)])
def test_classical_model_admits_its_decompositions_with_vectors(monkeypatch, solver, t, hermitian, cap):
    # For a 64-square A each classical model makes a decomposition that
    # forms vectors, and the recorded counters name the actual calls. The
    # inverse reuses one original factorization from planning. For a general
    # A it is the SVD with both frames (dgesdd), whose bidiagonal reduction,
    # ceil(4 * 64**3 / 3) = 349,526, and two back-transformations,
    # 2 * 64**2 * 63 = 516,096, make more than 865,000. For a Hermitian A it
    # is one eigh (dsyevd), whose tridiagonal reduction, about
    # (2/3) 64**3 = 174,763, and back-transformation of the eigenvectors,
    # 64**2 * 63 = 258,048, make more than 432,000. The native SVP shortcut
    # computes the SVD of the 128-square G_t with both frames by zgesdd,
    # whose bidiagonal reduction, ceil(4 * 128**3 / 3), and two
    # back-transformations, 2 * 128**2 * 127, bring the total above
    # 6,957,000. The dilation shortcut diagonalizes the 256-square dilation
    # of G_t with zheevd, (2/3) 256**3 and 256**2 * 255, more than
    # 27,896,000. Each cap lies below that work and must refuse the Plan. A
    # charge of one cubic term, 64**3, 128**3 or 256**3, for each
    # decomposition would admit it. The default limit admits each Plan.
    d = 64
    rng = np.random.default_rng(64)
    noise = .3*rng.standard_normal((d, d))/np.sqrt(d)
    problem = LinearSystem(A=4*np.eye(d) + (noise + noise.T if hermitian else noise), b=rng.standard_normal(d))
    assert (problem.A.structure == "hermitian") == hermitian
    fields = dict(solver=solver) if t is None else dict(solver=solver, encoded_solution_norm_estimate=t)
    output = None if t is None else StateVector(normalization="unit", global_phase="modulo_global_phase")
    counts = {}
    for name in ("svd", "eigh", "eigvalsh"):
        def counted(*args, _name=name, _call=getattr(np.linalg, name), **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _call(*args, **kwargs)
        monkeypatch.setattr(np.linalg, name, counted)
    plan = nwqlib.plan(problem, method=QLS(**fields), output=output, execution="classical", seed=7)
    spectral = dict(plan.reconstruction.spectral_calls)
    assert counts == {name: spectral[name + "_calls"] for name in ("svd", "eigh", "eigvalsh")
                      if spectral[name + "_calls"]}
    counts.clear()
    nwqlib.solve(plan)
    work = plan.reconstruction.work
    assert counts == {name: getattr(work, name + "_calls") for name in ("svd", "eigh")
                      if getattr(work, name + "_calls")}
    assert work.matrix_products == 0
    refusal = ("classical action" if t is not None else "original QLS eigensystem" if hermitian
               else "original QLS SVD") + " exceeds max_work"
    with pytest.raises(ValueError, match=refusal):
        nwqlib.plan(problem, method=QLS(max_work=cap, **fields), output=output, execution="classical", seed=7)


def test_compact_reference_admission_is_owned_and_never_densifies(monkeypatch):
    from nwqlib.operators import ingest_pauli
    problem = LinearSystem(A=ingest_pauli((("I", 1.), ("X", .1), ("Z", .1)), num_qubits=1),
                           b=[1, .25])
    # alpha=1.2 and kappa=1.4 covers the exact spectrum 1 +/- sqrt(.02).
    result = nwqlib.solve(problem, method=QLS(alpha=1.2, kappa=1.4), execution="quantum")
    assert "supplied spectral premise was not independently checked" in str(result)
    monkeypatch.setattr(type(problem.A), "dense_array", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    # Without a numeric kappa, for classical execution of a Pauli A and for a
    # sparse A, planning refuses with the user's remedy and without forming a
    # dense matrix.
    from scipy import sparse
    from nwqlib.algorithms.protocol import ApplicabilityError
    monkeypatch.setattr(sparse.csr_array, "toarray", forbidden)
    with pytest.raises(ApplicabilityError, match=r"QLS\(kappa="):
        nwqlib.plan(problem, method=QLS(), seed=7)
    with pytest.raises(ApplicabilityError, match=r"A\.to_matrix\(\)"):
        nwqlib.plan(problem, method=QLS(alpha=1.2, kappa=1.4), execution="classical", seed=7)
    for execution in ("quantum", "classical"):
        with pytest.raises(ApplicabilityError, match=r"A\.toarray\(\)"):
            nwqlib.plan(LinearSystem(A=sparse.csr_array([[1.1, .1], [.1, .9]]), b=[1, .25]),
                        method=QLS(), execution=execution, seed=7)
    with pytest.raises(ValueError, match="QLS verification requires original dense input access"):
        result.verify(checks=QLSVerification(comparisons=("spectral_domain",)))
    with pytest.raises(ValueError, match="max_work"):
        result.verify(checks=QLSVerification(comparisons=("spectral_domain",), max_work=1))


@pytest.mark.parametrize("factor", [1.0, 1e-12, 1e-13, 1e200, 1e-200])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_original_matrix_scaling_preserves_selected_inverse_and_mass(factor, sign):
    matrix = factor * np.diag([1.0, sign*2.0])
    low, high, _ = numerical._extreme_singular_values(matrix, hermitian=True)
    np.testing.assert_allclose((low/factor, high/factor), (1., 2.), rtol=2e-15, atol=0.)
    result = nwqlib.solve(selected(A=matrix, b=factor*np.ones(2)))
    np.testing.assert_allclose(result.x, [1., sign*.5], rtol=.01, atol=.01)
    assert 0 <= result.algorithm_success_mass <= 1
    assert result.plan.reconstruction.alpha/factor == pytest.approx(2., rel=2e-15, abs=0)
    assert result.plan.reconstruction.kappa_be == pytest.approx(2., rel=2e-15, abs=0)


def test_classical_mass_rejects_probability_outside_domain():
    result = nwqlib.solve(selected(A=np.diag([1., 2.]), b=[1., 1.]))
    with pytest.raises(ValueError, match="mass"):
        result.revise(algorithm_success_mass=-0.1)


@pytest.mark.parametrize(
    "output", [Solution(), QuadraticForm(observable=np.diag([0.4, -0.2, 0.1]))], ids=["solution", "quadratic"]
)
def test_saved_classical_result_loads_with_its_identity(tmp_path, output):
    """A saved classical QLS Result loads with its identity."""
    result = nwqlib.solve(selected(output=output))
    result.save(tmp_path / "result")
    assert nwqlib.load_result(tmp_path / "result").content_id == result.content_id


@pytest.mark.parametrize("factor", [1j, 1e200, 1e-200])
def test_inverse_preserves_physical_complex_rhs_scale(factor):
    plan = selected(A=np.diag([1.0, -2.0]), b=np.array([1.0, 1j]) * factor)
    result = nwqlib.solve(plan)
    expected = np.array([1.0, -0.5j])
    np.testing.assert_allclose(result.x / factor, expected, rtol=0.01, atol=0.01)
    assert (result.norm_squared is None) == (abs(factor) in (1e200, 1e-200))
    unit = nwqlib.solve(
        selected(
            A=np.diag([1.0, -2.0]),
            b=np.array([1.0, 1j]) * factor,
            output=StateVector(normalization="unit", global_phase="physical"),
        )
    )
    np.testing.assert_allclose(
        unit.value,
        expected / np.linalg.norm(expected) * (factor / abs(factor)),
        rtol=0.01,
        atol=0.01,
    )


@pytest.mark.parametrize(
    "output,expected",
    [
        (NormSquared(), 1.25),
        (QuadraticForm(observable=np.diag([2.0, -1.0])), 1.75),
        (NormalizedExpectation(observable=np.diag([2.0, -1.0])), 1.4),
    ],
)
def test_physical_and_normalized_scalars_are_actual_original_reductions(output, expected):
    result = nwqlib.solve(selected(A=np.diag([1.0, -2.0]), b=[1.0, 1j], output=output))
    assert result.value == pytest.approx(expected, rel=0.03, abs=0)


@pytest.mark.parametrize("solver", METHODS[1:])
def test_shortcut_physical_outputs_and_implicit_t_reject_before_fit(monkeypatch, solver):
    monkeypatch.setattr(owner, "select_inputs", forbidden)
    for output in (
        Solution(),
        NormSquared(),
        QuadraticForm(observable=np.eye(3)),
        StateVector(normalization="physical"),
        StateVector(normalization="unit", global_phase="physical"),
    ):
        with pytest.raises(ValueError, match="physical x scale"):
            selected(solver=solver, output=output)
    with pytest.raises(ValueError, match="explicit numeric t"):
        selected(solver=solver, t=None)


@pytest.mark.parametrize("source", ["grid", "noisy_binary_search", "linear_kappa_sequence"])
def test_named_classical_norm_model_records_actual_work_and_user_replay(monkeypatch, source):
    plan = selected(solver="shortcut_native_svp", A=np.diag([1.0, 0.25]), b=[1.0, 1j], t=source)
    result = nwqlib.solve(plan)
    args = {
        a.parameter: getattr(a.value, "value", a.value) for a in result.applications[0].arguments
    }
    assert result.execution == "classical" and result.physical_scale is None
    assert plan.reconstruction.t_source == source
    assert plan.reconstruction.work.target_solves == int(source != "linear_kappa_sequence")
    assert args["search_rows"] > 0 and args["planned_trials"] > 0
    application = result.applications[0]
    declared = {binding.parameter: binding.value for binding in owner.application_arguments(plan.reconstruction)}
    for quantity in ("search_rows", "planned_trials", "planned_queries"):
        limit = declared[quantity]
        at_limit = application.revise(arguments=tuple(
            binding.revise(value=limit) if binding.parameter == quantity else binding
            for binding in application.arguments))
        owner._validate_application(plan, (at_limit,))
        exceeded = at_limit.revise(arguments=tuple(
            binding.revise(value=limit+1) if binding.parameter == quantity else binding
            for binding in at_limit.arguments))
        with pytest.raises(ValueError, match="accounting exceeds selected envelope"):
            owner._validate_application(plan, (exceeded,))
    assert len(result.data.trace.events) > 0 and all(
        e.execution != "quantum_circuit" for e in result.data.trace.events
    )
    if plan.reconstruction.work.target_solves:
        # eq17 consumes only ||(A/alpha)^-1 b_hat||, which the model's own
        # solve recorded: verification reuses it without a solve and names it
        # as that intermediate. The direction comparison still solves once.
        reference = args["encoded_reference_norm"]
        assert math.isfinite(reference) and reference > 0
        with pytest.raises(ValueError, match="finite and positive"):
            owner._validate_application(plan, (application.revise(arguments=tuple(
                binding.revise(value=Float64(value=0.0)) if binding.parameter == "encoded_reference_norm"
                else binding
                for binding in application.arguments)),))
        choice = QLSVerification(comparisons=("eq17", "spectral_domain"))
        checks = {check.name: check.reference_work for check in choice.verification_checks(result)}
        assert checks["reference.eq17"].startswith("encoded reference norm recorded")
        assert checks["reference.spectral_domain"] == choice.source.reference
        with monkeypatch.context() as patched:
            patched.setattr(np.linalg, "solve", forbidden)
            reused = arguments(result.verify(checks=choice)[0])
        assert reused["recorded_reference_norm_reuses"] == 1 and reused["solve_attempts"] == 0
        assert reused["encoded_reference_norm"] == reference
        # The recorded norm is the independently solved ||(A/alpha)^-1 b_hat||.
        rec = plan.reconstruction
        rhs = np.array([1.0, 1j])
        independent = np.linalg.norm(np.linalg.solve(np.diag([1.0, 0.25]) / rec.alpha, rhs / np.linalg.norm(rhs)))
        assert reference == pytest.approx(independent, rel=1e-12, abs=0)
        solved = arguments(result.verify(checks=QLSVerification(comparisons=("shortcut_direction", "eq17")))[0])
        assert solved["solve_attempts"] == 1 and solved["recorded_reference_norm_reuses"] == 0
    replay = nwqlib.solve(
        selected(
            solver="shortcut_native_svp", A=np.diag([1.0, 0.25]), b=[1.0, 1j], t=args["t_value"]
        )
    )
    np.testing.assert_allclose(replay.value, result.value, rtol=1e-13, atol=1e-13)
    assert replay.algorithm_success_mass == pytest.approx(result.algorithm_success_mass, abs=1e-14)


def _rotated_condition_sixteen():
    """Return a rotation of diag(1, 1/16), whose exact condition number is 16.

    With the validated NumPy and LAPACK its computed polynomial_kappa is
    16.000000000000004, one ulp above 16.
    """
    c, s = math.cos(0.002), math.sin(0.002)
    rotation = np.array([[c, -s], [s, c]])
    matrix = rotation @ np.diag([1.0, 0.0625]) @ rotation.T
    return 0.5 * (matrix + matrix.T)


# sigma_max = 1 + 2**-52 and sigma_min = 2**-e are exact singular values, so
# kappa = (1 + 2**-52) * 2**e = nextafter(2**e, inf) exactly.
POWER_OF_TWO_EDGES = [
    pytest.param(np.diag([math.nextafter(1.0, math.inf), 2.0**-e]),
                 math.nextafter(2.0**e, math.inf), id=f"diag_2^{e}")
    for e in range(4, 8)
] + [pytest.param(_rotated_condition_sixteen(), None, id="rotated_16")]


@pytest.mark.parametrize("source", ["grid", "noisy_binary_search", "linear_kappa_sequence"])
@pytest.mark.parametrize("matrix,edge_kappa", POWER_OF_TWO_EDGES)
def test_named_norm_model_runs_one_ulp_above_a_power_of_two(matrix, edge_kappa, source):
    """Planning and execution count the same norm-search ladder at a power-of-two edge."""
    # The ladder depends on kappa only. The kernel-reflection degree
    # 2*ceil(kappa*ln(2/eta)/2), eta = epsilon_inv/sqrt(2), is 428 just above
    # kappa = 128 at epsilon_inv = 0.1, above the default max_degree of 256.
    plan = selected(solver="shortcut_native_svp", A=matrix, b=[1.0, 1j], t=source,
                    epsilon_inv=0.1, max_degree=512)
    kappa = plan.reconstruction.polynomial_kappa
    if edge_kappa is not None:
        assert kappa == edge_kappa
    steps = next(k for k in range(64) if 2**k >= Fraction(kappa))
    result = nwqlib.solve(plan)
    realized = {
        a.parameter: getattr(a.value, "value", a.value) for a in result.applications[0].arguments
    }
    declared = {
        b.parameter: getattr(b.value, "value", b.value)
        for b in owner.application_arguments(plan.reconstruction)
    }
    rows = {
        "grid": steps + 1,
        "noisy_binary_search": math.ceil(math.log(steps + 1) / math.log(1.5)),
        "linear_kappa_sequence": 4 * steps,
    }[source]
    assert realized["search_rows"] == declared["search_rows"] == rows
    for quantity in ("planned_trials", "planned_queries"):
        assert realized[quantity] == declared[quantity]


def test_selected_polynomial_exact_shape_independent_eigensystem(monkeypatch):
    def polynomial(method, kappa):
        return InversePolynomial(
            coefficients=(0.0, 1.0, 0.0, -0.5),
            rescale=2.0,
            candidate_degrees=(1, 3),
            certificate=0.0,
            certificate_grid_points=75,
            certificate_basis="affine_chebyshev_residual_norming",
            lsq_node_count=8,
        )

    monkeypatch.setattr(owner, "select_polynomial", polynomial)
    plan = selected(A=np.diag([1.0, -2.0]), b=[1.0, 1j], alpha=2.0, kappa=2.0)

    def eigensystem(matrix):
        np.testing.assert_array_equal(matrix, np.diag([0.5, -1.0]))
        return np.array([0.5, -1.0]), np.eye(2, dtype=complex)

    monkeypatch.setattr(np.linalg, "eigh", eigensystem)
    result = nwqlib.solve(plan)
    # The signed eigenvalue -1 gives P(-1) = -1/2; P(|-1|) would give +1/2.
    np.testing.assert_allclose(result.x, [1.0, -0.5j], rtol=0.0, atol=2e-15)
    assert result.algorithm_success_mass == pytest.approx(5 / 32, rel=0.0, abs=2e-16)
    monkeypatch.undo()
    monkeypatch.setattr(owner, "select_polynomial", polynomial)
    # Complex non-Hermitian A, padded to four coordinates: an independent
    # ascending Chebyshev recurrence on the explicit dilation
    # H = [[0, A/alpha], [A^dagger/alpha, 0]] applied to (b_hat, 0) gives
    # (0, y) with y = V P(Sigma/alpha) U^dagger b_hat (numerical.inverse_polynomial_action).
    # Starting from (0, b_hat) would give U P V^dagger b_hat.
    # Scoped regression tolerance: unit-scale dimension three
    # and degree three, rtol=0, atol=2e-12, compared without phase alignment.
    alpha, kappa = 3.0, 5.0
    plan = selected(alpha=alpha, kappa=kappa)
    assert plan.reconstruction.padded_dimension == 4 and plan._native["svd"] is None
    hermitian = np.zeros((6, 6), dtype=complex)
    hermitian[:3, 3:], hermitian[3:, :3] = MATRIX / alpha, MATRIX.conj().T / alpha
    b_hat = RHS / np.linalg.norm(RHS)
    previous, current = np.concatenate((b_hat, np.zeros(3))), None
    current = hermitian @ previous
    branch = 0.0 * previous + 1.0 * current
    previous, current = current, 2 * hermitian @ current - previous
    previous, current = current, 2 * hermitian @ current - previous
    branch = branch - 0.5 * current
    np.testing.assert_allclose(branch[:3], 0, rtol=0, atol=2e-12)
    expected = np.linalg.norm(RHS) * max(kappa, 1.01) / alpha * branch[3:]
    result = nwqlib.solve(plan)
    np.testing.assert_allclose(result.x, expected, rtol=0, atol=2e-12)
    assert result.algorithm_success_mass == pytest.approx(
        float(np.vdot(branch, branch).real) / 4, rel=0, abs=2e-12)


def test_polynomial_degree_work_and_bytes_are_admitted_before_each_candidate(monkeypatch):
    from nwqlib.subroutines.qsp import inverse

    fit = inverse._fit_inverse_chebyshev(4.0, 0.01)
    assert inverse._fit_inverse_chebyshev(4.0, 0.01, max_degree=fit.degree).degree == fit.degree
    with pytest.raises(ValueError, match="max_degree"):
        inverse._fit_inverse_chebyshev(4.0, 0.01, max_degree=fit.degree - 1)
    monkeypatch.setattr(inverse, "_fit_candidate", forbidden)
    for kwargs in ({"max_work": 1}, {"max_bytes": 1}):
        with pytest.raises(ValueError, match="max_"):
            inverse._fit_inverse_chebyshev(4.0, 0.01, **kwargs)


def test_invalid_scale_or_singular_original_rejects():
    for fields in (
        dict(alpha=True),
        dict(kappa=0.0),
        dict(epsilon_inv=np.inf),
        dict(max_degree=2.5),
    ):
        with pytest.raises(ValueError):
            QLS(**fields)
    with pytest.raises(ValueError, match="invertible"):
        selected(A=np.diag([0.0, 1.0]), b=[1.0, 1.0])
    with pytest.raises(ValueError, match="nonzero"):
        selected(b=np.zeros(3))
    with pytest.raises(ValueError, match="alpha"):
        selected(alpha=0.1)
    with pytest.raises(ValueError, match="kappa"):
        selected(kappa=1.0)


def test_explicit_reference_bundle_shares_solve_preserves_physical_phase(monkeypatch):
    result = nwqlib.solve(selected(A=np.diag([1.0, -2.0]), b=[1.0, 1j]))
    original = np.linalg.solve
    calls = []

    def solve(matrix, rhs):
        calls.append(1)
        return original(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", solve)
    choice = QLSVerification(
        comparisons=("spectral_domain", "inverse_relative_error", "inverse_success"),
        relative_tolerance=0.02,
    )
    receipt = result.verify(checks=choice)[0]
    assert calls == [1] and arguments(receipt)["selected_spectrum_reuses"] == 1
    assert facts(receipt)["reference.inverse_relative_error"] < 0.01
    monkeypatch.setattr(np.linalg, "solve", lambda a, b: -original(a, b))
    assert facts(result.verify(checks=choice)[0])["reference.inverse_relative_error"] > 1.9


def test_default_inverse_error_is_raw_error_over_polynomial_and_phase_allowance():
    from types import SimpleNamespace
    from nwqlib.algorithms.qls.verification import _comparisons

    A, b, name = np.diag([1.0, -2.0]), np.array([1.0, 1j]), "reference.inverse_relative_error"
    result = nwqlib.solve(selected(A=A, b=b))
    receipt = result.verify(checks=QLSVerification(comparisons=("inverse_relative_error",)))[0]
    rows = {f.fact.quantity: f for f in receipt.applications[0].facts}
    raw = rows[name + ".raw_relative_error"].fact.value.value
    allowance = rows[name + ".method_allowance"].fact.value.value
    x = np.linalg.solve(A, b)
    expected = np.linalg.norm(result.x - x) / np.linalg.norm(x)
    # Encoded and physical solves of this kappa-2 system differ by O(1e-16) relative,
    # far below the approximately 3e-3 inverse-fit error being compared.
    assert raw == pytest.approx(expected, rel=1e-10, abs=0)
    # The componentwise relative inverse residual gives ||error|| <= epsilon*||y||.
    # Classical execution contributes no phase-synthesis error.
    assert allowance == pytest.approx(0.01, rel=1e-14, abs=0)
    # Eigenvalues .5 and -1 of A/alpha lie in the fit domain, where
    # |k*lambda*p(lambda)-1| <= epsilon bounds the relative vector error by epsilon.
    assert rows[name].frame.metric == "relative_error_over_method_allowance"
    assert rows[name].fact.value.value == raw / allowance <= 1.0
    # A branch error delta contributes k*rescale*delta/||y|| after recovery.
    rec = SimpleNamespace(alpha=2.0, kappa_be=2.0, polynomial_kappa=2.0, phase_error=0.004,
                          polynomial=SimpleNamespace(rescale=1.25))
    values = _comparisons(
        SimpleNamespace(reconstruction=rec, method=QLS(), execution="quantum"), None,
        QLSVerification(comparisons=("inverse_relative_error",)),
        bounds=(1.0, 2.0), norm=1.0, discrepancy=0.003, parameters={},
    )
    # On the largest-singular-value direction, ||y||=1, so .01 + 2*1.25*.004=.02.
    assert values[name + ".method_allowance"] == pytest.approx(0.02, rel=1e-15, abs=0)
    assert values[name] == pytest.approx(0.15, rel=1e-15, abs=0)

    # With k=10, s=1 and ||y||=1 the exact branch amplitude is .1.
    # Relative polynomial error .01 permits amplitude error .001, hence
    # probability error at most (.101)^2-(.1)^2=.000201.
    rec = SimpleNamespace(alpha=1.0, kappa_be=10.0, polynomial_kappa=10.0, phase_error=0.0,
                          polynomial=SimpleNamespace(rescale=1.0))
    values = _comparisons(
        SimpleNamespace(reconstruction=rec, method=QLS(), execution="quantum"),
        SimpleNamespace(returned_shots=None, algorithm_success_mass=0.011),
        QLSVerification(comparisons=("inverse_success",)),
        bounds=(0.1, 1.0), norm=1.0, discrepancy=None, parameters={},
    )
    assert values["reference.inverse_success.deterministic_allowance"] == pytest.approx(
        0.000201, rel=1e-14, abs=0)
    assert values["reference.inverse_success"] > 1


def test_reference_missing_vector_and_finite_limits_precede_solve(monkeypatch):
    vector = nwqlib.solve(selected())
    scalar = nwqlib.solve(selected(output=NormSquared()))
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    with pytest.raises(ValueError, match="existing physical-phase"):
        scalar.verify(checks=QLSVerification(comparisons=("inverse_relative_error",)))
    with pytest.raises(ValueError, match="max_work"):
        vector.verify(checks=QLSVerification(comparisons=("inverse_relative_error",), max_work=1))
    receipt = vector.verify(checks=QLSVerification(comparisons=("spectral_domain",)))[0]
    assert arguments(receipt)["solve_attempts"] == 0 and arguments(receipt)["svd_attempts"] == 0


def test_shortcut_explicit_direction_allows_declared_global_phase(monkeypatch):
    result = nwqlib.solve(selected(solver="shortcut_native_svp"))
    original = np.linalg.solve
    monkeypatch.setattr(np.linalg, "solve", lambda a, b: 1j * original(a, b))
    receipt = result.verify(checks=QLSVerification(comparisons=("shortcut_direction", "eq17")))[0]
    assert facts(receipt)["reference.shortcut_direction"] < 0.01
    assert facts(receipt)["reference.eq17"] == 0


def test_selection_rejects_a_plan_with_other_inputs():
    result = nwqlib.solve(selected())
    altered = result.plan.revise(problem=LinearSystem(A=MATRIX, b=RHS * 2))
    with pytest.raises(ValueError, match="input association"):
        validate_selection(altered)


def test_result_artifact_must_belong_to_selection_and_requested_frame():
    result = nwqlib.solve(selected(A=np.diag([1.0, -2.0]), b=[1.0, 1j]))
    artifact = result.artifact
    result.validate_plan(result.plan)
    for changed, message in (
        (artifact.revise(plan_id="sha256:" + "0" * 64),
         "QLS artifact differs from its actual selection"),
        (artifact.revise(construction_id="sha256:" + "1" * 64),
         "QLS artifact differs from its actual selection"),
        (artifact.revise(output=artifact.output.revise(global_phase="modulo_global_phase")),
         "QLS artifact differs from its requested original-coordinate frame"),
    ):
        with pytest.raises(ValueError, match=message):
            result.revise(artifact=changed).validate_plan(result.plan)


def test_binary_recovery_products_validate_operands_and_preserve_components():
    from math import ldexp
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery

    tiny = ldexp(1.0, -1074)
    recovery = PhysicalScale(mantissa=0.5, exponent=1000)
    vector = recovery.apply_vector(np.array([complex(tiny, -tiny)]))
    np.testing.assert_array_equal(vector, [complex(ldexp(1.0, -75), -ldexp(1.0, -75))])
    assert (
        PhysicalScale(mantissa=0.5, exponent=0).apply_vector(np.array([1.0 + 0j, tiny + 0j]))
        is None
    )
    np.testing.assert_array_equal(recovery.apply_vector(np.array([0.0 + 0j])), [0.0 + 0j])
    # Compose the exact pair tiny/(2*tiny) before another factor2: recovery1.
    composed = compose_recovery(PhysicalScale(mantissa=0.5, exponent=-1073), (1.0, 2 * tiny), 2.0)
    assert composed.mantissa == 0.5 and composed.exponent == 1
    for factors in (
        ((-2.0, -1.0),),
        ((1.0, 0.0),),
        ((1.0, float("inf")),),
        (0.0,),
        (True,),
        (PhysicalScale(mantissa=0.0, exponent=0),),
    ):
        with pytest.raises(ValueError):
            compose_recovery(*factors)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_subnormal_signed_physical_moment_recovers_before_mantissa_product(sign):
    from math import ldexp
    from nwqlib.problems.inputs import PhysicalScale
    from nwqlib.problems.scalars import physical_vector_statistics

    scale, _, statistics = physical_vector_statistics(
        np.array([1.0 + 0j]),
        observable=ingest_dense([[sign * ldexp(1.0, -1072)]]),
        recovery=PhysicalScale(mantissa=0.5, exponent=501),
    )
    # Framed vector1/2 produces moment +/-2^-1074. Recovery2^500 yields +/-2^-72.
    assert statistics[1].value == sign * ldexp(1.0, -72)
    assert scale.mantissa == 0.5 and scale.exponent == 501


def test_safe_matrix_normalization_and_identity_path():
    from math import ldexp
    from nwqlib._numerics import normalized_matrix

    tiny = ldexp(1.0, -1074)
    np.testing.assert_array_equal(
        normalized_matrix(np.array([[complex(tiny, -tiny)]]), 2 * tiny), [[0.5 - 0.5j]]
    )
    np.testing.assert_array_equal(normalized_matrix(np.array([[tiny]]), 2 * tiny), [[0.5]])
    encoded = np.array([[0.5 + 0j]])
    assert normalized_matrix(encoded, 1.0) is encoded


@pytest.mark.parametrize('factor', [1., 2j])
@pytest.mark.parametrize('a,r', [(1e308, 1e-160), (-1e308, 1e-160), (1e-320, 1e160)])
def test_inverse_scalar_recovers_observable_scale_before_physical_float(a, r, factor):
    output = QuadraticForm(observable=np.full((2, 2), a))
    choice = selected(A=np.eye(2), b=np.full(2, factor*r), output=output)
    result = nwqlib.solve(choice)
    # A=I gives x=b. QLS's selected inverse residual <= epsilon contributes
    # at most 2*epsilon+epsilon² to this rank-one quadratic form's relative error.
    expected = 4 * (a*r)*r * abs(factor)**2
    epsilon = choice.method.epsilon_inv
    assert result.value == pytest.approx(expected, rel=2*epsilon+epsilon**2+1e-13, abs=0.)
    assert result.plan.reconstruction.work.observable_actions == 1


@pytest.mark.parametrize('solver', METHODS)
def test_unit_observable_scale_preserves_signed_cancellation(solver):
    choice = selected(solver=solver, A=np.eye(2), b=[1.,1.],
                      output=NormalizedExpectation(observable=np.diag([1e308,-1e308])), t=1.)
    result = nwqlib.solve(choice)
    # Equal components cancel exactly, including their selected scalar polynomial.
    # A vdot reduction at this scale need not be correctly rounded after cancellation,
    # so use the analytic zero with a componentwise roundoff bound in operator units.
    assert result.value is not None
    assert abs(result.value) <= 64*np.finfo(float).eps*1e308
