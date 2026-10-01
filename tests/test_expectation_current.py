"""Finite Pauli scientific values, actual counts and selected criterion lineage."""

from math import log
import pytest
import nwqlib
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.problems import Accuracy, Expectation
from nwqlib.operators.inputs import ingest_pauli
from nwqlib.problems.inputs import ingest_occupation


def problem(terms=(("I", 1.0), ("Z", 1.0)), state=None):
    return Expectation(
        state=ingest_occupation("0", num_qubits=1) if state is None else state,
        observable=ingest_pauli(terms, num_qubits=1),
    )


def test_exact_physical_operator_and_normalized_state_share_one_result():
    p = problem(state=[2.0, 2.0])
    result = nwqlib.solve(p, method=ExpectationMethod(), seed=7)
    # <+|(I+Z)|+> = 1 independently of the physical state norm 8.
    assert result.value == pytest.approx(1.0, abs=1e-13, rel=0)
    assert result.physical_scale.squared_as_float() == pytest.approx(8.0, abs=1e-13, rel=0)
    assert result.data.trace.jobs == 1
    # The default exact route reads one grouped Pauli expectation, records no
    # sampled shots and keeps no array for this scalar output.
    assert [e.observation.kind for e in result.plan.experiments] == ["pauli_expectation"]
    assert [event.shots for event in result.data.trace.events] == [0]
    assert result.data.artifacts == ()
    assert result.plan.problem is p


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_supplied_rotations_preserve_public_expectation_and_archive(execution, tmp_path):
    from math import cos, sin
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, global_phase=.31)
    circuit.ry(.9, 0)
    circuit.ry(-.4, 1)
    circuit.rz(1.3, 0)
    p = Expectation(state=circuit,
        observable=ingest_pauli((("ZI", 1.), ("IZ", .5), ("XI", .25)), num_qubits=2))
    expected = cos(.4) + .5*cos(.9) - .25*sin(.4)
    result = nwqlib.solve(p, method=ExpectationMethod(), execution=execution)
    assert result.value == pytest.approx(expected, rel=0, abs=2e-13)
    restored = nwqlib.load_result(result.save(tmp_path / execution))
    assert restored.analyze().value == pytest.approx(expected, rel=0, abs=2e-13)


@pytest.mark.parametrize("label, expected", [("XX", 1.), ("YY", -1.)])
def test_bell_state_multisite_parity_has_the_physical_sign(label, expected):
    # Both single-site means vanish. Rotating only the parity pivot or losing
    # a Y phase cannot recover these opposite correlations. Counts select the
    # parity circuit; Bell's XX/YY outcomes are deterministic, so no sampling
    # tolerance is needed for these eigenstate witnesses.
    p = Expectation(state=[1., 0., 0., 1.],
        observable=ingest_pauli(((label, 1.),), num_qubits=2))
    result = nwqlib.solve(p, method=ExpectationMethod(), execution="quantum", shots=32, seed=7)
    assert all(chunk.returned_shots == 32 for chunk in result.data.observations.chunks)
    assert result.value == pytest.approx(expected, rel=0, abs=2e-13)


def test_grouped_sampled_readout_shares_group_shots_and_their_covariance():
    from fractions import Fraction

    from nwqlib.evidence.binary import BinaryInferenceOptions, BinaryReadoutMitigation

    # Bell labels ZI and IZ share one QWC group, XX forms the other. The
    # weighted score ZI+IZ has per-shot variance 4 while the individual
    # variances sum to 2, so the group variance must use the joint histogram
    # (ExpectationMethod grouped readout, _group_variance).
    p = Expectation(state=[2**-.5, 0., 0., 2**-.5],
                    observable=ingest_pauli((("ZI", 1.), ("IZ", 1.), ("XX", 1.)), num_qubits=2))
    options = BinaryInferenceOptions(sampling_model="iid_bernoulli", independent_populations=True)
    result = nwqlib.solve(p, method=ExpectationMethod(inference=options), execution="quantum",
                          shots=200, seed=7)
    settings = result.plan.reconstruction.settings
    assert [(s.name, s.labels, s.basis, s.qubits) for s in settings] == [
        ("group_0", ("ZI", "IZ"), "ZZ", (0, 1)), ("group_1", ("XX",), "XX", (0, 1))]
    assert [chunk.returned_shots for chunk in result.data.observations.chunks] == [200, 200]
    zi, iz, xx = result.statistics.populations
    assert zi.population.source_ids == iz.population.source_ids != xx.population.source_ids
    assert (zi.population.zeros, zi.population.ones) == (iz.population.zeros, iz.population.ones)
    n0, n = zi.population.zeros, 200
    first, second = 2 * (2 * n0 - n), 4 * n
    covariant = Fraction(n * second - first**2, n * n * (n - 1))
    variance = result.statistics.variance.value.fact.value
    assert Fraction(variance.numerator, variance.denominator) == covariant
    marginal = 2 * Fraction(4 * n0 * (n - n0), n * n * (n - 1))
    assert covariant == 2 * marginal and covariant != marginal
    beta = nwqlib.solve(p, method=ExpectationMethod(inference=options.revise(method="beta")),
                        execution="quantum", shots=200, seed=7).statistics
    assert beta.variance is None and "joint posterior" in beta.variance_unavailable
    assert all(item.point is not None and item.interval.family_size == 3 for item in beta.populations)
    assert beta.interval.family_size == 3 and any("joint credibility" in a for a in beta.interval.assumptions)
    mitigated = nwqlib.solve(
        p, method=ExpectationMethod(mitigation=BinaryReadoutMitigation(calibration_shots=100)),
        execution="quantum", shots=200, seed=7)
    assert [s.role for s in mitigated.plan.reconstruction.settings] == [
        "science", "science", "science", "calibration_zero", "calibration_one",
        "calibration_zero", "calibration_one"]
    # Qubit 0 is |->, qubit 1 the +1 eigenstate of Y and qubit 2 is |0>, so
    # the one group ZYX, which reads differently reversed, has deterministic
    # parities that differ by site. Each group experiment is Allocate, PREP,
    # the group's basis block, one whole-register Measure and Release
    # (_measured_program), and each label's decoded mean equals its exact
    # expectation, which fixes the qubits of the basis rotations, the bit
    # order and the physical Y sign.
    import numpy as np
    from qiskit.quantum_info import SparsePauliOp, Statevector

    state = np.kron(np.kron([1, 0], np.array([1, 1j]) / 2**.5), np.array([1, -1]) / 2**.5)
    labels = ("IIX", "IYI", "ZII", "ZYX")
    local = nwqlib.solve(
        Expectation(state=state, observable=ingest_pauli([(label, 1.) for label in labels], num_qubits=3)),
        method=ExpectationMethod(), execution="quantum", shots=16, seed=7)
    program = local.plan.construction.program
    nodes = {item.id: item.node for item in program.definitions}
    assert [(r.name, r.width) for r in program.registers] == [("system", 3)]
    assert [(c.name, c.width) for c in program.classical] == [("readout", 3)]
    assert [s.basis for s in local.plan.reconstruction.settings] == ["ZYX"]
    assert nodes["body_group_0"].children == ("allocate", "prepare_science", "basis_0", "measure", "release")
    assert (nodes["measure"].wire, nodes["measure"].result) == ("system", "readout")
    exact = Statevector(state)
    means = {item.population.name.split(":")[1]: (item.population.zeros - item.population.ones) / 16
             for item in local.statistics.populations}
    assert means == pytest.approx(
        {label: exact.expectation_value(SparsePauliOp(label)).real for label in labels}, rel=0, abs=1e-12)
    assert means["IIX"] == means["ZYX"] == -1 and means["IYI"] == means["ZII"] == 1


def test_nearly_singleton_groups_keep_sampled_program_admission():
    import numpy as np

    # Random 12-qubit labels form nearly one QWC group per term. With one
    # whole-register body per group, the 61-term prefix (58 groups) plans at
    # the default Program admission limits (ir/expressions.py::AdmissionLimits).
    rng = np.random.default_rng(1)
    labels = [s for s in dict.fromkeys("".join(rng.choice(list("IXYZ"), 12)) for _ in range(400)) if set(s) != {"I"}]
    coefficients = np.random.default_rng(2).normal(size=len(labels))
    p = Expectation(state=ingest_occupation("0" * 12, num_qubits=12),
                    observable=ingest_pauli(list(zip(labels[:61], map(float, coefficients[:61]))), num_qubits=12))
    plan = nwqlib.plan(p, method=ExpectationMethod(), shots=100, seed=7)
    assert len(plan.experiments) == 58 and plan.shots == 100


def test_grouping_refusal_names_a_value_that_admits_on_the_first_retry():
    """A grouping refusal names the Method field and a value that completes grouping.

    The nine two-qubit labels without I are pairwise not QWC, so first-fit
    grouping needs 0+1+...+8 = 36 comparisons, the L(L-1)/2 envelope that
    PauliTerms.group names. A next-tile minimum (3 at cap 1) would be refused
    again. One QWC group of seven Z labels needs L-1 = 6 comparisons and is
    admitted at 6, far below its envelope 21.
    """
    import re

    def selected(terms, cap):
        n = len(terms[0][0])
        p = Expectation(state=[1] + [0] * ((1 << n) - 1), observable=ingest_pauli(terms, num_qubits=n))
        return nwqlib.plan(p, method=ExpectationMethod(max_classical_products=cap), shots=64, seed=1)

    distinct = tuple((a + b, 1.0 / (1 + i)) for i, (a, b) in enumerate(
        (a, b) for a in "XYZ" for b in "XYZ"))
    with pytest.raises(ValueError, match=r"ExpectationMethod\.max_classical_products=1\b") as refusal:
        selected(distinct, 1)
    named = int(re.search(r"Raise ExpectationMethod\.max_classical_products to (\d+), which is sufficient",
                          str(refusal.value)).group(1))
    assert len(selected(distinct, named).experiments) == 9
    diagonal = tuple((format(k, "03b").replace("0", "I").replace("1", "Z"), 1.0) for k in range(1, 8))
    assert len(selected(diagonal, 6).experiments) == 1


def test_sampling_accuracy_selects_actual_shots_and_keeps_original_criterion(tmp_path):
    p = problem((("Z", 2.0), ("X", -1.0)))
    accuracy = Accuracy(component="sampling", absolute_tolerance=0.5, confidence=0.9)
    selected = nwqlib.plan(p, method=ExpectationMethod(), accuracy=accuracy, seed=7)
    required = 2 * (3 / 0.5) ** 2 * log(4 / 0.1)
    assert selected.shots - 1 < required <= selected.shots
    # compare selects each candidate's acquisition through the same connector,
    # and a Method without one becomes a blocked row instead of an error.
    from _hadamard_method import HadamardPauliExpectation

    compared = nwqlib.compare(
        p, methods=(ExpectationMethod(), HadamardPauliExpectation()), accuracy=accuracy, seed=7
    )
    assert compared.rows[0].plan.shots == selected.shots
    assert compared.rows[0].plan.selection_accuracy == accuracy
    assert compared.rows[1].plan is None and "accuracy connector" in compared.rows[1].reason
    result = nwqlib.solve(selected)
    assert sum(event.shots for event in result.data.trace.events) == 2 * selected.shots
    assert all(chunk.returned_shots == selected.shots for chunk in result.data.observations.chunks)
    assert result.plan.selection_accuracy == accuracy
    path = result.save(tmp_path / "result")
    restored = nwqlib.load_result(path)
    assert restored.plan.selection_accuracy == accuracy and restored.content_id == result.content_id
    other = accuracy.revise(absolute_tolerance=0.50001)
    changed = nwqlib.plan(p, method=ExpectationMethod(), accuracy=other, seed=7)
    assert changed.shots == selected.shots and changed.content_id != selected.content_id


@pytest.mark.parametrize(
    "criterion", [Accuracy(absolute_tolerance=0.1), Accuracy(component="sampling", relative_tolerance=0.1)]
)
def test_unsupported_accuracy_rejects_before_execution(criterion):
    with pytest.raises(ValueError, match="sampling accuracy"):
        nwqlib.plan(problem(), method=ExpectationMethod(), accuracy=criterion)


def test_identity_needs_no_acquisition_and_explicit_shots_conflict():
    accuracy = Accuracy(component="sampling", absolute_tolerance=0.1)
    selected = nwqlib.plan(problem((("I", 3.0),)), method=ExpectationMethod(), accuracy=accuracy)
    result = nwqlib.solve(selected)
    assert result.value == 3.0 and result.data.trace.jobs == 0
    with pytest.raises(ValueError, match="conflict"):
        nwqlib.plan(problem(), method=ExpectationMethod(), shots=10, accuracy=accuracy)


@pytest.fixture
def injected(monkeypatch):
    from types import SimpleNamespace
    from nwqlib.backends import qiskit_aer
    from nwqlib.blocks import lowering

    data = SimpleNamespace(payloads={}, calls=[])
    lower = lowering._lower_qiskit

    def selected_lower(construction, **kwargs):
        value = lower(construction, **kwargs)
        value.circuit.metadata = {"test_setting": construction.program.root.removeprefix("batch_")}
        return value

    def acquire(prepared):
        name = prepared.circuit.metadata["test_setting"]
        data.calls.append(name)
        return SimpleNamespace(
            raw_output={"counts": data.payloads[name]}, metadata={"native_job_id": str(len(data.calls))}
        )

    monkeypatch.setattr(lowering, "_lower_qiskit", selected_lower)
    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", acquire)
    return data


@pytest.mark.parametrize("quadratic", [False, True])
def test_calibration_shared_variance_signed_result_and_reanalysis(injected, quadratic):
    from fractions import Fraction
    from nwqlib.evidence.binary import BinaryInferenceOptions, BinaryReadoutMitigation

    inference = BinaryInferenceOptions(
        method="hoeffding", sampling_model="iid_bernoulli", independent_populations=True
    )
    method = ExpectationMethod(
        inference=inference, mitigation=BinaryReadoutMitigation(calibration_shots=1000)
    )
    injected.payloads = {
        "science_0": {"0": 80, "1": 20},
        "science_1": {"0": 80, "1": 20},
        "zero_0": {"0": 900, "1": 100},
        "one_0": {"0": 200, "1": 800},
    }
    p = problem((("Z", 1.0), ("X", 2.0), ("I", 3.0)), state=[2.0, 0.0])
    output = nwqlib.QuadraticForm(observable=p.observable) if quadratic else p.default_output()
    scale = 4 if quadratic else 1
    result = nwqlib.solve(p, method=method, output=output, shots=100, seed=17)
    assert result.value == pytest.approx(scale * (3 + 15 / 7), abs=1e-13, rel=0)
    assert result.statistics.raw_value == pytest.approx(scale * 4.8, abs=1e-13, rel=0)
    # Shared calibration derivatives add before squaring: -180/49, -30/49.
    expected = (
        5 * Fraction(10, 7) ** 2 * Fraction(16, 25 * 99)
        + Fraction(180, 49) ** 2 * Fraction(9, 25 * 999)
        + Fraction(30, 49) ** 2 * Fraction(16, 25 * 999)
    )
    value = result.statistics.variance.value.fact.value
    assert Fraction(value.numerator, value.denominator) == pytest.approx(
        scale**2 * float(expected), abs=1e-13, rel=0
    )
    assert result.statistics.variance_kind == "delta_method"
    assert result.statistics.interval.status == "conditional"
    # Two science settings and two calibration pivots, each prepared once.
    assert len(result.data.trace.events) == result.data.trace.preparations == 4
    assert sum(event.shots for event in result.data.trace.events) == 2200
    assert result.data.artifacts == ()
    calls = injected.calls.copy()
    revised = result.analyze(inference=inference.revise(method="anytime_hoeffding"))
    assert revised.statistics.interval.kind == "time_uniform" and revised.plan is result.plan
    assert result.statistics.interval.kind == "fixed_time" and injected.calls == calls


def test_sampling_accuracy_rejects_unused_frozen_repeat_before_new_work(injected):
    from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
    from nwqlib.core.planning import RuntimeOptions
    from nwqlib.evidence.binary import BinaryInferenceOptions

    selected = nwqlib.plan(
        problem((("Z", 1.0),)),
        method=ExpectationMethod(),
        accuracy=Accuracy(component="sampling", absolute_tolerance=1.0),
        seed=7,
    )
    injected.payloads = {"group_0": {"0": selected.shots}}
    run = Run(selected)
    handle = prepare_experiment(selected.resolve("group_0"), run=run, runtime=RuntimeOptions(seed=11))
    submit_experiment(handle, run=run)  # Deliberately successful and uncollected.
    with pytest.raises(ValueError, match="fixed sampling stream"):
        submit_experiment(handle, run=run)
    assert run.trace.jobs == 1 and len(injected.calls) == 1
    with pytest.raises(ValueError, match="sampling accuracy"):
        nwqlib.plan(
            problem(),
            method=ExpectationMethod(
                inference=BinaryInferenceOptions(method="beta", sampling_model="iid_bernoulli")
            ),
            accuracy=Accuracy(component="sampling", absolute_tolerance=0.1),
        )


def test_small_dense_observable_preserves_complex_pauli_mean_and_archive(tmp_path, monkeypatch):
    import numpy as np
    from nwqlib.operators import _pauli

    # Y expectation on (|0>+i|1>)/sqrt(2) is +1; the tiny Z coefficient survives.
    observable = np.array([[2.0 + 1e-12, -1j], [1j, 2.0 - 1e-12]])
    p = Expectation(state=[1.0, 1j], observable=observable)
    result = nwqlib.solve(p, method=ExpectationMethod(), seed=9)
    assert result.value == pytest.approx(3.0, abs=1e-13, rel=0)
    assert result.plan.problem is p and result.plan.output.observable is p.observable
    assert dict((x.label, x.coefficient) for x in result.plan.reconstruction.terms)["Z"] != 0
    path = result.save(tmp_path / "dense")
    monkeypatch.setattr(_pauli, "pauli_coefficients", lambda *a: pytest.fail("load transformed dense matrix"))
    assert nwqlib.load_result(path).value == result.value
    with pytest.raises(ValueError, match="dimension <=16"):
        nwqlib.plan(
            Expectation(state=np.eye(1, 32, 0).reshape(-1), observable=np.eye(32)), method=ExpectationMethod()
        )
    with pytest.raises(ValueError, match="Hermitian"):
        Expectation(state=[1.0, 0.0], observable=[[1.0, 1e-16], [0.0, 1.0]])


def test_sampling_integer_edge_uses_exact_input_confidence():
    from decimal import Decimal, localcontext
    from math import nextafter, sqrt

    for epsilon in (nextafter(sqrt(2 * log(40) / 1000), 0.0), nextafter(sqrt(2 * log(40) / 1000), 1.0)):
        accuracy = Accuracy(component="sampling", absolute_tolerance=epsilon)
        selected = nwqlib.plan(problem((("Z", 1.0),)), method=ExpectationMethod(), accuracy=accuracy)
        with localcontext() as context:
            context.prec = 160
            required = 2 / Decimal(epsilon) ** 2 * (Decimal(2) / (1 - Decimal(accuracy.confidence))).ln()
            assert selected.shots - 1 < required <= selected.shots


def test_actual_sampling_evidence_keeps_confidence_through_assessment_and_save(injected, tmp_path):
    from nwqlib.evidence.binary import BinaryInferenceOptions

    selected = nwqlib.plan(
        problem((("Z", 1.0),)),
        method=ExpectationMethod(),
        accuracy=Accuracy(component="sampling", absolute_tolerance=0.5, confidence=0.9),
        seed=7,
    )
    injected.payloads = {"group_0": {"0": selected.shots}}
    result = nwqlib.solve(selected)
    (fact,) = result.facts
    assert fact.fact.quantity == "sampling" and fact.fact.value.value <= 0.5
    assert fact.failure_probability == 1 - selected.selection_accuracy.confidence
    requested = result.assess(component="sampling", absolute_tolerance=0.5, confidence=0.99)
    assert requested.status == "INCONCLUSIVE"
    assert requested.facts[0].failure_probability == fact.failure_probability
    calls = injected.calls.copy()
    revised = result.analyze(
        inference=BinaryInferenceOptions(
            method="hoeffding", sampling_model="iid_bernoulli", failure_probability=0.2
        )
    )
    assert revised.facts[0].failure_probability == 0.2 and injected.calls == calls
    assert revised.statistics.interval.probability == 0.8 and result.facts == (fact,)
    restored = nwqlib.load_result(revised.save(tmp_path / "evidence"))
    assert restored.facts == revised.facts and restored.plan.selection_accuracy.confidence == 0.9
    # Bell ZI and IZ share one group and XX forms another, so G = 2 < L = 3.
    # The exported radius sum_j |c_j| sqrt(2 log(2L/delta)/n_g(j)) takes its
    # union bound over labels (_sampling_facts), with n_g(j) the injected
    # size of label j's group.
    from math import sqrt

    coefficients = (("ZI", 1.0), ("IZ", 0.5), ("XX", -2.0))
    bell = nwqlib.plan(
        Expectation(state=[2**-.5, 0., 0., 2**-.5], observable=ingest_pauli(coefficients, num_qubits=2)),
        method=ExpectationMethod(),
        accuracy=Accuracy(component="sampling", absolute_tolerance=1.0, confidence=0.9),
        seed=7,
    )
    injected.payloads = {"group_0": {"00": bell.shots - 1, "11": 1}, "group_1": {"00": bell.shots}}
    result = nwqlib.solve(bell)
    (fact,) = result.facts
    sizes = {item.population.name.split(":")[1]: item.population.zeros + item.population.ones
             for item in result.statistics.populations}
    groups = {item.population.name.split(":")[0] for item in result.statistics.populations}
    assert sorted(sizes) == ["IZ", "XX", "ZI"] and set(sizes.values()) == {bell.shots} and len(groups) == 2
    delta = fact.failure_probability
    radius = sum(abs(c) * sqrt(2 * log(2 * 3 / delta) / sizes[label]) for label, c in coefficients)
    assert fact.fact.value.value == pytest.approx(radius, rel=1e-12, abs=0)


@pytest.mark.parametrize("mode", ["classical", "quantum"])
def test_quadratic_form_preserves_magnitude_phase_and_zero(mode, tmp_path):
    import numpy as np
    from nwqlib.problems import QuadraticForm

    for index, (state, sign, expected) in enumerate(
        (([2, 2], 1, 8), ([2j, 2j], -1, -8), ([6 + 8j, 6 + 8j], 1, 200), ([0, 0], -1, 0))
    ):
        p = Expectation(state=state, observable=sign * np.diag([2.0, 0.0]))
        result = nwqlib.solve(
            p,
            method=ExpectationMethod(),
            output=QuadraticForm(observable=p.observable),
            execution=mode,
            seed=7,
        )
        assert result.value == pytest.approx(expected, rel=0, abs=1e-11)
        assert result.plan.output.frame(p).quantity == "quadratic_form"
        if expected == 0:
            assert result.data.trace.jobs == 0 and not result.contribution_ids
            with pytest.raises(ValueError, match="zero state"):
                nwqlib.plan(p, method=ExpectationMethod(), execution=mode)
        else:
            normalized = nwqlib.solve(p, method=ExpectationMethod(), execution=mode, seed=7)
            assert normalized.value == pytest.approx(sign, rel=0, abs=2e-13)
        restored = nwqlib.load_result(result.save(tmp_path / f"{mode}-{index}"))
        assert restored.value == result.value and restored.analyze().value == result.value
        assert "Quadratic form" in str(restored)


def test_classical_original_dimension_sparse_access_and_early_admission(monkeypatch):
    import numpy as np
    from scipy.sparse import csr_matrix, csc_matrix
    from nwqlib.problems import QuadraticForm
    from nwqlib.algorithms import _eigen_inputs
    from nwqlib.operators import _pauli

    def forbidden(*args, **kwargs):
        pytest.fail("classical input converted or materialized before admission")

    monkeypatch.setattr(_pauli, "pauli_coefficients", forbidden)
    for constructor in (np.array, csr_matrix, csc_matrix):
        p = Expectation(state=[2, 2, 0], observable=constructor(np.diag([2.0, 0.0, -1.0])))
        result = nwqlib.solve(
            p,
            method=ExpectationMethod(),
            output=QuadraticForm(observable=p.observable),
            execution="classical",
        )
        assert result.value == pytest.approx(8, rel=0, abs=2e-13)
        assert result.plan.problem.dimension == 3
        from math import ldexp
        assert ldexp(result.data.observations.chunks[0].values[0].value,
                     result.plan.reconstruction.operator_exponent) == pytest.approx(
            1, rel=0, abs=2e-13
        )
        with pytest.raises(ValueError, match="power-of-two"):
            nwqlib.plan(p, method=ExpectationMethod())
        # A constant diagonal with nonzero off-diagonal entries is not c*I
        # (_classical_constant): <v|O|v> = 2 for v = (1, 1)/sqrt(2), not 1.
        coupled = Expectation(state=[1, 1], observable=constructor(np.array([[1.0, 1.0], [1.0, 1.0]])))
        result = nwqlib.solve(coupled, method=ExpectationMethod(), execution="classical")
        assert result.plan.reconstruction.constant is None
        assert result.value == pytest.approx(2, rel=0, abs=2e-13)
    p = Expectation(state=[1, 1], observable=[[1.0, 0.0], [0.0, -1.0]])
    with monkeypatch.context() as guard:
        guard.setattr(_eigen_inputs, "state_direction", forbidden)
        for method, shots in (
            (ExpectationMethod(max_classical_products=1), None),
            (ExpectationMethod(), 4),
            (ExpectationMethod(estimate_precision=0.1), None),
        ):
            with pytest.raises(ValueError, match="(max_classical_products|classical expectation)"):
                nwqlib.solve(p, method=method, execution="classical", shots=shots)


def test_quadratic_counts_scale_point_variance_interval_and_sampling(injected, tmp_path):
    from decimal import Decimal, localcontext
    from fractions import Fraction
    from nwqlib.problems import QuadraticForm, Accuracy
    from nwqlib.evidence.binary import BinaryInferenceOptions

    inference = BinaryInferenceOptions(
        method="hoeffding", sampling_model="iid_bernoulli", independent_populations=True
    )
    p = problem((("Z", 1.0),), state=[2.0, 2.0])
    output = QuadraticForm(observable=p.observable)
    injected.payloads = {"group_0": {"0": 75, "1": 25}}
    normal = nwqlib.solve(p, method=ExpectationMethod(inference=inference), shots=100, seed=7)
    physical = nwqlib.solve(
        p, method=ExpectationMethod(inference=inference), output=output, shots=100, seed=7
    )
    assert physical.value == pytest.approx(4, rel=0, abs=2e-13)
    variance = physical.statistics.variance.value.fact.value
    assert Fraction(variance.numerator, variance.denominator) == pytest.approx(
        64 * 0.75 / 99, rel=0, abs=2e-13
    )
    npair, ppair = normal.statistics.interval, physical.statistics.interval
    assert (ppair.lower, ppair.upper) == pytest.approx(
        (8 * npair.lower, 8 * npair.upper), rel=0, abs=2e-13
    )
    assert physical.facts[0].frame.quantity == "quadratic_form"
    old_calls = injected.calls.copy()
    revised = physical.analyze(inference=inference.revise(method="anytime_hoeffding"))
    assert injected.calls == old_calls and revised.value == physical.value
    assert nwqlib.load_result(revised.save(tmp_path / "qf-counts")).value == physical.value

    p = problem((("Z", 1.0),), state=[2.0, 0.0])  # Exact input norm 2, s=4.
    output = QuadraticForm(observable=p.observable)
    accuracy = Accuracy(component="sampling", absolute_tolerance=0.5, confidence=0.9)
    selected = nwqlib.plan(p, method=ExpectationMethod(), output=output, accuracy=accuracy, seed=7)
    with localcontext() as context:
        context.prec = 100
        required = 2 * (Decimal(4) / Decimal(0.5)) ** 2 * (Decimal(2) / (1 - Decimal(0.9))).ln()
        assert selected.shots - 1 < required <= selected.shots
    from nwqlib.core.planning import RandomStreams

    with pytest.raises(ValueError, match="actual shots"):
        selected.method.plan(
            p,
            output=output,
            execution="quantum",
            shots=selected.shots - 1,
            accuracy=accuracy,
            rng=RandomStreams(7),
        )
    injected.payloads = {"group_0": {"0": selected.shots}}
    actual = nwqlib.solve(selected)
    assert actual.value == 4
    assert actual.data.trace.events[0].shots == selected.shots
    assert actual.facts[0].fact.value.value <= 0.5
    restored = nwqlib.load_result(actual.save(tmp_path / "qf-accuracy"))
    assert restored.plan.selection_accuracy == accuracy
    assert restored.plan.output.kind == "quadratic_form"


def test_quadratic_scaling_does_not_overflow_intermediate_norm():
    from nwqlib.problems import QuadraticForm

    p = problem((("I", 1e-200),), state=[1e200, 0.0])
    result = nwqlib.solve(
        p, method=ExpectationMethod(), output=QuadraticForm(observable=p.observable)
    )
    assert result.physical_scale.squared_as_float() is None
    assert result.value == pytest.approx(1e200, rel=5e-15, abs=0)
    assert result.data.trace.jobs == 0


def test_quantum_four_dimensional_and_classical_supplied_preparation_scale(monkeypatch, tmp_path):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from test_gcim_native_preparations import reference_kwargs
    from nwqlib.problems import QuadraticForm

    p = Expectation(state=[2, 0, 0, 0], observable=np.diag([-2.0, 1.0, 3.0, 4.0]))
    result = nwqlib.solve(
        p, method=ExpectationMethod(), output=QuadraticForm(observable=p.observable), seed=7
    )
    assert result.value == -8
    wrong = result.physical_scale.revise(exponent=result.physical_scale.exponent + 1)
    with pytest.raises(ValueError, match="(scale|reconstruction)"):
        result.revise(physical_scale=wrong)._attach(result.plan, result.data)
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.s(0)
    seed = reference_kwargs(circuit)["reference"]
    p = Expectation(state=seed, observable=[[0.0, -1j], [1j, 0.0]])
    with monkeypatch.context() as guard:
        guard.setattr(
            Statevector,
            "from_instruction",
            lambda *a, **k: pytest.fail("planning simulated circuit"),
        )
        selected = nwqlib.plan(p, method=ExpectationMethod(), execution="classical")
        with pytest.raises(ValueError, match="max_classical_products"):
            nwqlib.plan(
                p, method=ExpectationMethod(max_classical_products=1), execution="classical"
            )
    result = nwqlib.solve(selected)
    assert result.value == pytest.approx(1.0, rel=0, abs=2e-13)  # Y(1,i)=(1,i).
    with monkeypatch.context() as guard:
        guard.setattr(
            Statevector,
            "from_instruction",
            lambda *a, **k: pytest.fail("saved/reanalysis simulated circuit"),
        )
        loaded = nwqlib.load_result(result.save(tmp_path / "supplied"))
        assert loaded.analyze().value == result.value


def test_classical_quadratic_matvec_recovers_extreme_observable_scale(tmp_path, monkeypatch):
    import numpy as np
    from scipy.sparse import csr_matrix, csc_matrix
    from nwqlib.operators import inputs as owner
    from nwqlib.problems import QuadraticForm

    # Every entry contributes a*r*r: four terms, without forming r² or 2*a.
    for index, (a, r, expected) in enumerate(((1e308, 1e-160, 4e-12),
                                            (1e-320, 1e160, 4 * (1e-320 * 1e160) * 1e160),
                                            (-1e308, 1e-160, -4e-12),
                                            (2., 3., 72.))):
        for name, observable in (("dense", np.full((2, 2), a)),
                                 ("csr", csr_matrix(np.full((2, 2), a))),
                                 ("csc", csc_matrix(np.full((2, 2), a))),
                                 ("pauli", ingest_pauli((("I", a), ("X", a)), num_qubits=1))):
            p = Expectation(state=[r, r], observable=observable)
            result = nwqlib.solve(p, method=ExpectationMethod(), execution="classical",
                                  output=QuadraticForm(observable=p.observable))
            assert result.value == pytest.approx(expected, rel=1e-13, abs=0)
            assert len(result.data.trace.events) == 1
            assert result.data.trace.events[0].execution == "host_kernel"
            scalar = result.data.observations.chunks[0].values[0]
            assert scalar.label == "scaled_observable" and scalar.value is not None
            if name == "dense":
                path = result.save(tmp_path / str(index))
                with monkeypatch.context() as guard:
                    guard.setattr(owner, "_scaled_observable", lambda *a: pytest.fail("load replayed action"))
                    loaded = nwqlib.load_result(path)
                    assert loaded.analyze().value == result.value
                bad = result.plan.revise(reconstruction=result.plan.reconstruction.revise(
                    operator_exponent=result.plan.reconstruction.operator_exponent + 1))
                with pytest.raises(ValueError, match="(Plan|binding|reconstruction|exponent)"):
                    result.validate_plan(bad)
            if a == 1e308:
                normalized = nwqlib.solve(p, method=ExpectationMethod(), execution="classical")
                assert normalized.value is None and normalized.unavailable
