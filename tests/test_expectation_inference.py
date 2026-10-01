"""Independent binary algebra and injected public expectation acquisitions."""

from dataclasses import replace
from fractions import Fraction
from math import comb, log, sqrt
from types import SimpleNamespace
import pytest
import nwqlib
from nwqlib.algorithms.expectation import ExpectationAnalysis, ExpectationMethod
from nwqlib.core import Source
from nwqlib.core.planning import RuntimeOptions
from nwqlib.evidence.binary import (
    BinaryInferenceOptions,
    BinaryPopulation,
    BinaryReadoutMitigation,
    correct_binary,
    infer_binary,
)
from nwqlib.execution import ObservationView, ExecutionLimits
from nwqlib.operators.inputs import ingest_pauli
from nwqlib.problems.inputs import ingest_occupation

from nwqlib._prepared_execution import Run, prepare_experiment as prepare, submit_experiment as submit
from nwqlib.core.analysis import RunData
from nwqlib.problems import Expectation


def make_choice(
    *,
    terms=(("Z", 1.0),),
    shots=100,
    inference=None,
    mitigation=None,
    q=1,
    preparation="native",
    occupations=None,
):
    state = ingest_occupation((0,) * q if occupations is None else occupations, num_qubits=q)
    problem = Expectation(state=state, observable=ingest_pauli(terms, num_qubits=q))
    return nwqlib.plan(
        problem,
        method=ExpectationMethod(
            preparation_choice=preparation,
            inference=inference or BinaryInferenceOptions(),
            mitigation=mitigation,
        ),
        shots=shots,
        seed=7,
    )


@pytest.fixture
def injected(monkeypatch):
    from nwqlib.blocks import lowering
    from nwqlib.backends import qiskit_aer as aer

    state = SimpleNamespace(payloads={}, calls=[], lowered=[])
    original = lowering._lower_qiskit

    def logical(construction, **kwargs):
        value = original(construction, **kwargs)
        name = construction.program.root.removeprefix("batch_")
        value.circuit.metadata = {"test_setting": name}
        state.lowered.append(name)
        return value

    def acquire(prepared):
        name = prepared.circuit.metadata["test_setting"]
        state.calls.append(name)
        return SimpleNamespace(
            raw_output={"counts": state.payloads[name]},
            metadata={"native_job_id": f"injected-{len(state.calls)}"},
        )

    monkeypatch.setattr(lowering, "_lower_qiskit", logical)
    monkeypatch.setattr(aer, "_submit_aer_execution", acquire)
    return state


def acquire(choice, injected, *, only=None, repeat=None):
    run = Run(choice)
    handles = {}
    for i, experiment in enumerate(choice.experiments):
        if only is not None and experiment.name not in only:
            continue
        handle = prepare(choice.resolve(experiment.name), run=run, runtime=RuntimeOptions(seed=7 + i))
        handles[experiment.name] = handle
        run.collect(submit(handle, run=run))
    if repeat is not None:
        run.collect(submit(handles[repeat], run=run))
    return run


def analyze(choice, run):
    return choice.method.analyze(choice, run.data, settings={})


def population(name, zeros, ones):
    source = Source(
        name=name, version="1", domain="explicit supplied binary data", reference="hand-derived test"
    )
    return BinaryPopulation(
        name=name,
        zeros=zeros,
        ones=ones,
        source_ids=(source.content_id,),
        observation_ids=(source.revise(name=name + ".observation").content_id,),
    )


def infer(data, method="point", *, family=1, fixed=True, **options):
    selected = BinaryInferenceOptions(method=method, sampling_model="iid_bernoulli", **options)
    result = infer_binary(
        data,
        options=selected,
        family_size=family,
        fixed_time=fixed,
        fixed_time_reason="explicit stopped acquisition",
        applicability_reason=None,
    )
    return (selected, result)


def test_binary_mean_variance_and_finite_interval_are_different():
    options, observed = infer(population("balanced", 3, 1), "hoeffding")
    assert observed.raw_mean == 0.5
    assert Fraction(
        observed.empirical_variance.numerator, observed.empirical_variance.denominator
    ) == Fraction(1, 4)
    _, identical = infer(population("identical", 20, 0), "hoeffding")
    assert identical.empirical_variance.numerator == 0
    radius = sqrt(2 * log(40) / 20)
    assert identical.interval.lower == pytest.approx(1 - radius, rel=1e-14, abs=1e-14)
    assert identical.interval.upper == 1
    _, stopped = infer(population("stopped", 3, 1), "hoeffding", fixed=False)
    assert stopped.point == 0.5 and stopped.interval.kind == "fixed_time"
    assert stopped.interval.status == "unavailable" and "stopped" in stopped.interval.reason
    _, anytime = infer(population("stopped", 30, 10), "anytime_hoeffding", fixed=False)
    radius = sqrt(2 * log(2 * 40 * 41 / options.failure_probability) / 40)
    assert anytime.interval.kind == "time_uniform"
    assert anytime.interval.lower == pytest.approx(max(-1, 0.5 - radius), rel=1e-14, abs=1e-14)
    assert anytime.interval.status == "conditional"


def test_beta_special_case_prior_domain_and_unknown_model(monkeypatch):
    """A Beta(2,1) posterior gives analytic square-root quantiles, while unsupported priors must
    not reach inversion.
    """
    _, result = infer(population("one zero", 1, 0), "beta")
    assert result.point == pytest.approx(1 / 3, rel=1e-14, abs=1e-14)
    assert result.posterior_variance == pytest.approx(2 / 9, rel=1e-14, abs=1e-14)
    assert result.interval.lower == pytest.approx(2 * sqrt(0.025) - 1, rel=1e-13, abs=1e-13)
    assert result.interval.upper == pytest.approx(2 * sqrt(0.975) - 1, rel=1e-13, abs=1e-13)
    assert result.interval.kind == "bayesian"
    for prior in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            BinaryInferenceOptions(method="beta", prior_alpha=prior)
    import scipy.special

    for kernel in ("betaincinv", "betainccinv"):
        monkeypatch.setattr(
            scipy.special, kernel, lambda *a: pytest.fail("unsupported Beta reached kernel")
        )
    selected = BinaryInferenceOptions(method="beta")
    unknown = infer_binary(
        population("unknown", 1, 0),
        options=selected,
        family_size=1,
        fixed_time=True,
        fixed_time_reason="complete",
        applicability_reason=None,
    )
    assert unknown.raw_mean == 1 and unknown.point is None
    assert unknown.interval.status == "unavailable"
    extreme = BinaryInferenceOptions(method="beta", sampling_model="iid_bernoulli", prior_alpha=1e300)
    guarded = infer_binary(
        population("unrepresentable posterior", 0, 1),
        options=extreme,
        family_size=1,
        fixed_time=True,
        fixed_time_reason="complete",
        applicability_reason=None,
    )
    assert guarded.raw_mean == -1 and guarded.point is None and (guarded.interval.status == "unavailable")
    with pytest.raises(ValueError, match="no selected Beta"):
        ExpectationMethod(inference=selected, mitigation=BinaryReadoutMitigation(calibration_shots=100))


def _beta_tail_masses(interval, a, b):
    """Exact Beta(a, b) posterior masses below and above an interval in 2p-1.

    For integer shapes, I_x(a, b) = P(Binomial(a+b-1, x) >= a), a finite sum
    that is an exact rational at the binary64 endpoint x = (endpoint+1)/2.
    """
    m = a + b - 1

    def below(x):
        return sum(comb(m, j) * x**j * (1 - x) ** (m - j) for j in range(a, m + 1))

    lower = (Fraction(interval.lower) + 1) / 2
    upper = (Fraction(interval.upper) + 1) / 2
    return below(lower), 1 - below(upper)


@pytest.mark.parametrize("failure", [0.05, 1.2e-16])
def test_beta_endpoints_leave_the_selected_tail_mass(failure):
    """Both endpoints of Beta(4, 6) leave posterior mass alpha/2 outside.

    The window of 1 percent of alpha/2 is the tail-mass accuracy stated with
    MAX_BETA_SHAPE_TOTAL in ENGINEERING_CONSTANTS. At alpha/2 = 6e-17,
    1-alpha/2 rounds to 1-2**-53, and inverting I_x there would leave 2**-53
    above the upper endpoint, 85 percent more than alpha/2.
    """
    _, result = infer(population("tail", 3, 5), "beta", failure_probability=failure)
    assert result.interval.status == "conditional"
    tail = Fraction(failure) / 2
    below, above = _beta_tail_masses(result.interval, 4, 6)
    assert abs(below / tail - 1) <= Fraction(1, 100)
    assert abs(above / tail - 1) <= Fraction(1, 100)


def _cornish_fisher_beta_quantile(a, b, z):
    """Beta(a, b) quantile at standard normal quantile z, and the posterior sd.

    Cornish-Fisher to second order in the skewness g1 and first order in the
    excess kurtosis g2. The omitted terms are of order (a+b)**-1.5 standard
    deviations, about 3e-18 at a+b = 5e11.
    """
    s = a + b
    mean, sd = a / s, sqrt(a * b / (s * s * (s + 1)))
    g1 = 2 * (b - a) * sqrt(s + 1) / ((s + 2) * sqrt(a * b))
    g2 = 6 * ((a - b) ** 2 * (s + 1) - a * b * (s + 2)) / (a * b * (s + 2) * (s + 3))
    w = z + (z * z - 1) * g1 / 6 + (z**3 - 3 * z) * g2 / 24 - (2 * z**3 - 5 * z) * g1 * g1 / 36
    return mean + w * sd, sd


@pytest.mark.parametrize(
    "zeros,ones",
    [
        (150_000_000_000, 349_999_999_998),  # a+b = 5e11, the validated limit
        (150_000_000_000, 349_999_999_999),  # a+b = 5e11 + 1
        (2_702_159_776_422_297, 6_305_039_478_318_694),  # n = 2**53-1
    ],
)
def test_beta_interval_is_unavailable_above_the_validated_shape_total(zeros, ones, monkeypatch):
    """A huge count keeps its posterior mean and variance but loses its interval.

    SciPy's endpoint error grows about in proportion to a+b and reaches 0.32
    posterior standard deviations at n = 2**53-1 with n0/n = 0.3. At
    a+b = 5e11 the endpoints stay within 1e-3 standard deviations of a
    Cornish-Fisher reference.
    """
    from scipy.special import ndtri

    a, b = 1 + zeros, 1 + ones
    available = a + b <= 5e11
    if not available:
        import scipy.special

        for kernel in ("betaincinv", "betainccinv"):
            monkeypatch.setattr(scipy.special, kernel, lambda *x: pytest.fail("kernel called"))
    _, result = infer(population("large", zeros, ones), "beta")
    assert result.point == pytest.approx(float(Fraction(a - b, a + b)), rel=1e-15, abs=0)
    variance = Fraction(4 * a * b, (a + b) ** 2 * (a + b + 1))
    assert result.posterior_variance == pytest.approx(float(variance), rel=1e-14, abs=0)
    if not available:
        assert result.interval.status == "unavailable" and "shape total" in result.interval.reason
        return
    assert result.interval.status == "conditional"
    for endpoint, z in ((result.interval.lower, ndtri(0.025)), (result.interval.upper, -ndtri(0.025))):
        reference, sd = _cornish_fisher_beta_quantile(a, b, z)
        assert abs((endpoint + 1) / 2 - reference) <= 1e-3 * sd


@pytest.mark.parametrize(
    "zero,one,science,expected",
    [
        ((9, 1), (2, 8), (8, 2), Fraction(5, 7)),
        ((2, 8), (9, 1), (8, 2), Fraction(-5, 7)),
        ((6, 4), (4, 6), (10, 0), Fraction(5)),
    ],
)
def test_affine_asymmetry_negative_contrast_and_signed_point(zero, one, science, expected):
    inference, x = infer(population("science", *science))
    _, z0 = infer(population("zero", *zero))
    _, z1 = infer(population("one", *one))
    correction = correct_binary(
        x,
        z0,
        z1,
        options=BinaryReadoutMitigation(calibration_shots=10),
        inference=inference,
        applicability_reason=None,
    )
    assert correction.point == pytest.approx(float(expected), rel=1e-14, abs=1e-14)
    d = (zero[0] - zero[1]) / 10 - (one[0] - one[1]) / 10
    assert correction.contrast == pytest.approx(d / 2, rel=1e-14, abs=1e-14)
    assert correction.derivatives == pytest.approx(
        (2 / d, -(1 + float(expected)) / d, (float(expected) - 1) / d), rel=1e-14, abs=1e-14
    )
    assert type(correction).model_validate_json(correction.model_dump_json()).point == correction.point


def test_unusable_contrast_and_joint_rectangle_do_not_fallback():
    inference, x = infer(population("science", 800, 200), "hoeffding", family=3)
    _, zero = infer(population("zero", 900, 100), "hoeffding", family=3)
    _, one = infer(population("one", 200, 800), "hoeffding", family=3)
    option = BinaryReadoutMitigation(calibration_shots=1000)
    fitted = correct_binary(x, zero, one, options=option, inference=inference, applicability_reason=None)
    assert fitted.interval.status == "conditional" and fitted.interval_image is not None
    assert fitted.interval.lower <= 5 / 7 <= fitted.interval.upper
    _, same = infer(population("same", 500, 500), "hoeffding", family=3)
    unusable = correct_binary(x, same, same, options=option, inference=inference, applicability_reason=None)
    assert unusable.point is None and unusable.interval.status == "unavailable"
    assert "contrast" in unusable.unavailable[0]
    _, weak0 = infer(population("weak0", 6, 4), "hoeffding", family=3)
    _, weak1 = infer(population("weak1", 4, 6), "hoeffding", family=3)
    uncertain = correct_binary(
        x, weak0, weak1, options=option, inference=inference, applicability_reason=None
    )
    assert uncertain.point is not None and uncertain.interval.status == "unavailable"
    assert "denominator" in uncertain.interval.reason
    _, high = infer(population("incompatible science", 10000, 0), "hoeffding", family=3)
    _, tight0 = infer(population("tight zero", 600000, 400000), "hoeffding", family=3)
    _, tight1 = infer(population("tight one", 400000, 600000), "hoeffding", family=3)
    empty = correct_binary(
        high, tight0, tight1, options=option, inference=inference, applicability_reason=None
    )
    assert empty.point == pytest.approx(5, rel=1e-14, abs=1e-14)
    assert empty.interval.status == "empty" and empty.interval_image[0] > 1


@pytest.mark.parametrize("mismatch", ["upgrade", "beta", "family", "probability", "kind"])
def test_correction_rejects_incompatible_saved_inference_before_arithmetic(monkeypatch, mismatch):
    from nwqlib.evidence._work import ExactArithmetic

    inference, science = infer(population("science", 600, 400), "hoeffding", family=3)
    _, zero = infer(population("zero", 900, 100), "hoeffding", family=3)
    _, one = infer(population("one", 100, 900), "hoeffding", family=3)
    if mismatch == "upgrade":
        inference = inference.revise(method="anytime_hoeffding")
    elif mismatch == "beta":
        _, zero = infer(population("zero", 900, 100), "beta", family=3, independent_populations=True)
    elif mismatch == "family":
        _, zero = infer(population("zero", 900, 100), "hoeffding", family=1)
    elif mismatch == "probability":
        zero = zero.revise(interval=zero.interval.revise(probability=0.9))
    else:
        zero = zero.revise(interval=zero.interval.revise(kind="time_uniform"))
    monkeypatch.setattr(
        ExactArithmetic, "fraction", lambda *a: pytest.fail("incompatible inference reached arithmetic")
    )
    with pytest.raises(ValueError, match="inference"):
        correct_binary(
            science,
            zero,
            one,
            options=BinaryReadoutMitigation(calibration_shots=1000),
            inference=inference,
            applicability_reason=None,
        )


def test_frozen_reuse_partial_and_missing_receipts_keep_honest_inference(injected):
    """Repeated frozen samples consume execution work but do not create independent statistical
    observations.
    """
    options = BinaryInferenceOptions(method="hoeffding", sampling_model="iid_bernoulli")
    choice = make_choice(inference=options)
    injected.payloads = {"group_0": {"0": 75, "1": 25}}
    run = acquire(choice, injected, repeat="group_0")
    result = analyze(choice, run)
    sample = result.statistics.populations[0].population
    assert sample.zeros + sample.ones == 100
    assert len(sample.reused_observation_ids) == 1
    assert result.value == 0.5 and (not result.statistics.fixed_time)
    assert (
        result.statistics.interval.kind == "fixed_time" and result.statistics.interval.status == "unavailable"
    )
    assert len(run.trace.events) == 2 and sum((e.shots for e in run.trace.events)) == 200
    first_receipt = run.prepared_artifacts[0]
    repeated = prepare(
        choice.resolve("group_0"),
        run=run,
        runtime=first_receipt.runtime.revise(seed=first_receipt.runtime.seed),
    )
    assert repeated.record.content_id != first_receipt.content_id
    run.collect(submit(repeated, run=run))
    again = analyze(choice, run)
    data = again.statistics.populations[0].population
    assert data.zeros + data.ones == 100 and len(data.reused_observation_ids) == 2
    assert data.preparation_ids == ((first_receipt.content_id, repeated.record.content_id),)
    assert set(again.statistics.receipt_ids) == set(data.preparation_ids[0])
    assert len(run.trace.events) == 3 and sum((e.shots for e in run.trace.events)) == 300
    restored = ExpectationAnalysis.model_validate_json(again.model_dump_json())
    restored.validate_plan(choice)
    assert restored.statistics == again.statistics
    absent = choice.method.analyze(choice, RunData(run.observations, trace=run.trace), settings={})
    assert absent.value == 0.5 and absent.statistics.applicability_reason
    partial_choice = make_choice(
        inference=options, mitigation=BinaryReadoutMitigation(calibration_shots=1000)
    )
    injected.payloads = {"science_0": {"0": 75, "1": 25}}
    partial = acquire(partial_choice, injected, only={"science_0"})
    unavailable = analyze(partial_choice, partial)
    assert unavailable.value is None and unavailable.statistics.raw_value == 0.5
    assert "calibration" in unavailable.unavailable


def test_identity_options_do_not_create_measurement_or_calibration(monkeypatch):
    import nwqlib.algorithms.expectation as owner

    monkeypatch.setattr(
        owner, "_measured_program", lambda *a, **k: pytest.fail("identity created a measured path")
    )
    choice = make_choice(
        terms=(("I", -2.0),),
        inference=BinaryInferenceOptions(method="anytime_hoeffding"),
        mitigation=BinaryReadoutMitigation(calibration_shots=1000),
    )
    run = Run(choice)
    result = analyze(choice, run)
    assert result.value == -2 and result.statistics is None
    assert choice.experiments == choice.blocks == ()
    assert run.trace.events == () and run.trace.preparations == 0


@pytest.mark.parametrize("independent", [False, True])
def test_independent_equal_histograms_keep_distinct_variables(injected, independent):
    choice = make_choice(
        terms=(("Z", 1.0), ("X", 1.0)),
        inference=BinaryInferenceOptions(sampling_model="iid_bernoulli", independent_populations=independent),
    )
    injected.payloads = {"group_0": {"0": 75, "1": 25}, "group_1": {"0": 75, "1": 25}}
    run = acquire(choice, injected)
    result = analyze(choice, run)
    populations = result.statistics.populations
    assert populations[0].population.source_ids != populations[1].population.source_ids
    assert result.value == 1
    variance = result.statistics.variance.value.fact
    if independent:
        assert float(Fraction(variance.value.numerator, variance.value.denominator)) == pytest.approx(
            1.5 / 99, rel=1e-14, abs=1e-14
        )
        assert variance.evidence.kind == "user_assertion"
    else:
        assert variance.availability == "unknown"
    shared = Run(choice)
    for experiment in choice.experiments:
        handle = prepare(choice.resolve(experiment.name), run=shared, runtime=RuntimeOptions(seed=7))
        shared.collect(submit(handle, run=shared))
    correlated = analyze(choice, shared)
    assert correlated.value == 1
    assert len({item.population.source_ids[0] for item in correlated.statistics.populations}) == 2
    assert all(
        (item.population.zeros + item.population.ones == 100 for item in correlated.statistics.populations)
    )
    assert correlated.statistics.variance is None
    assert "fixed sampling stream" in correlated.statistics.variance_unavailable


@pytest.mark.parametrize("method", ["hoeffding", "anytime_hoeffding", "beta"])
def test_shared_stream_preserves_marginal_union_bound_but_not_product_independence(injected, method):
    choice = make_choice(
        terms=(("Z", 1.0), ("X", 1.0)),
        inference=BinaryInferenceOptions(
            method=method, sampling_model="iid_bernoulli", independent_populations=True
        ),
    )
    injected.payloads = {"group_0": {"0": 75, "1": 25}, "group_1": {"0": 75, "1": 25}}
    run = Run(choice)
    for experiment in choice.experiments:
        handle = prepare(choice.resolve(experiment.name), run=run, runtime=RuntimeOptions(seed=7))
        run.collect(submit(handle, run=run))
    result = analyze(choice, run)
    assert result.statistics.fixed_time and result.statistics.raw_value == 1
    assert result.statistics.independence_reason and result.statistics.variance is None
    if method == "beta":
        assert result.value is None and result.statistics.interval.status == "unavailable"
    else:
        assert result.value == 1 and result.statistics.interval.status == "conditional"
        assert all((item.interval.status == "conditional" for item in result.statistics.populations))
        alpha = 0.05 / 2
        radius = sqrt(2 * log(2 / alpha * (100 * 101 if method == "anytime_hoeffding" else 1)) / 100)
        assert result.statistics.interval.lower == pytest.approx(
            2 * max(-1, 0.5 - radius), rel=1e-14, abs=1e-14
        )


def test_within_setting_shared_stream_is_detected_after_another_setting(injected, monkeypatch):
    from nwqlib.backends.connection import AerBackend

    choice = make_choice(
        q=2,
        terms=(("IZ", 1.0), ("IX", 1.0)),
        inference=BinaryInferenceOptions(
            method="anytime_hoeffding", sampling_model="iid_bernoulli", independent_populations=True
        ),
    )
    injected.payloads = {"group_0": {"00": 75, "01": 25}, "group_1": {"00": 75, "01": 25}}
    run = Run(choice)
    native_prepare = AerBackend.prepare
    mapping = (0, 1)

    def prepare_mapping(self, *args, **kwargs):
        return replace(native_prepare(self, *args, **kwargs), logical_to_native=mapping)

    monkeypatch.setattr(AerBackend, "prepare", prepare_mapping)
    for name, mapping in (("group_0", (0, 1)), ("group_1", (0, 1)), ("group_1", (1, 0))):
        handle = prepare(choice.resolve(name), run=run, runtime=RuntimeOptions(seed=7))
        run.collect(submit(handle, run=run))
    result = analyze(choice, run)
    first, second = result.statistics.populations
    assert first.interval.status == "conditional"
    assert second.population.zeros == 150 and second.population.ones == 50
    assert len(second.population.source_ids) == 2
    assert second.interval.status == "unavailable" and "fixed sampling stream" in second.interval.reason
    assert result.statistics.interval.status == "unavailable" and result.statistics.raw_value == 1.0


@pytest.mark.parametrize("fresh", [False, True])
def test_new_populations_accumulate_but_stopped_fixed_time_stays_unavailable(injected, monkeypatch, fresh):
    """Contrast complete prefixes with filtered or data-stopped populations before allowing
    coverage claims.
    """
    from nwqlib.backends.connection import AerBackend
    from nwqlib.execution import CountsSampling

    if fresh:
        actual = AerBackend.prepare
        monkeypatch.setattr(
            AerBackend,
            "prepare",
            lambda self, *args, **kwargs: replace(
                actual(self, *args, **kwargs), counts_sampling=CountsSampling(kind="fresh")
            ),
        )
    choice = make_choice(
        inference=BinaryInferenceOptions(method="anytime_hoeffding", sampling_model="iid_bernoulli")
    )
    injected.payloads = {"group_0": {"0": 75, "1": 25}}
    run = Run(choice)
    first = prepare(choice.resolve("group_0"), run=run, runtime=RuntimeOptions(seed=7))
    run.collect(submit(first, run=run))
    if fresh:
        receipt, next_handle = (first.record, first)
    else:
        next_handle = prepare(choice.resolve("group_0"), run=run, runtime=RuntimeOptions(seed=8))
    injected.payloads = {"group_0": {"0": 25, "1": 75}}
    run.collect(submit(next_handle, run=run))
    result = analyze(choice, run)
    population = result.statistics.populations[0].population
    assert population.zeros + population.ones == 200 and len(population.source_ids) == 2
    assert result.value == 0
    assert len(run.prepared_artifacts) == (1 if fresh else 2)
    if fresh:
        assert population.preparation_ids == ((receipt.content_id,), (receipt.content_id,))
    assert (
        result.statistics.interval.status == "conditional"
        and result.statistics.interval.kind == "time_uniform"
    )
    filtered = choice.method.analyze(
        choice,
        RunData(
            ObservationView(chunks=(run.observations.chunks[-1],)),
            trace=run.trace,
            receipts=run.prepared_artifacts,
        ),
        settings={},
    )
    assert filtered.value == -0.5 and filtered.statistics.interval.status == "unavailable"
    assert "earlier" in filtered.statistics.interval.reason
    prefix = choice.method.analyze(
        choice,
        RunData(
            ObservationView(chunks=(run.observations.chunks[0],)),
            trace=run.trace,
            receipts=run.prepared_artifacts,
        ),
        settings={},
    )
    assert prefix.value == 0.5 and prefix.statistics.interval.status == "conditional"
    assert (
        prefix.statistics.populations[0].population.zeros + prefix.statistics.populations[0].population.ones
        == 100
    )
    fixed_choice = make_choice(
        inference=BinaryInferenceOptions(method="hoeffding", sampling_model="iid_bernoulli")
    )
    fixed_run = acquire(fixed_choice, injected)
    stopped = fixed_choice.method.analyze(
        fixed_choice,
        RunData(
            fixed_run.observations,
            trace=fixed_run.trace.revise(termination_reason="observed-data stop"),
            receipts=fixed_run.prepared_artifacts,
        ),
        settings={},
    )
    assert stopped.value == -0.5 and stopped.statistics.interval.status == "unavailable"
    assert stopped.statistics.interval.kind == "fixed_time"


@pytest.mark.parametrize("method", ["anytime_hoeffding", "beta", "point"])
@pytest.mark.parametrize("fresh", [False, True])
def test_imported_observed_population_does_not_start_a_fresh_statistical_prefix(
    injected, monkeypatch, method, fresh
):
    """Importing an already observed frozen preparation must not restart a fresh statistical
    prefix.
    """
    from nwqlib.backends.connection import AerBackend
    from nwqlib.execution import CountsSampling

    if fresh:
        actual = AerBackend.prepare
        monkeypatch.setattr(
            AerBackend,
            "prepare",
            lambda self, *args, **kwargs: replace(
                actual(self, *args, **kwargs), counts_sampling=CountsSampling(kind="fresh")
            ),
        )
    choice = make_choice(inference=BinaryInferenceOptions(method=method, sampling_model="iid_bernoulli"))
    old = Run(choice)
    for seed, payload in ((7, {"1": 100}), (8, {"0": 100})):
        injected.payloads = {"group_0": payload}
        handle = prepare(choice.resolve("group_0"), run=old, runtime=RuntimeOptions(seed=seed))
        old.collect(submit(handle, run=old))
    full = analyze(choice, old)
    assert full.statistics.raw_value == 0
    if method != "point":
        assert full.statistics.interval.status == "conditional"
    restored = ExpectationAnalysis.model_validate_json(full.model_dump_json())
    restored.validate_plan(choice)
    assert restored == full
    new = Run(choice)
    new.collect(submit(handle, run=new))
    result = analyze(choice, new)
    assert result.statistics.raw_value == 1
    assert new.trace.preparations == 0 and new.trace.local_prepared_ids == ()
    assert len(new.trace.events) == 1 and new.trace.events[0].shots == 100
    if method == "point":
        assert result.value == 1 and result.statistics.interval is None
    elif fresh:
        assert result.statistics.interval.status == "conditional"
        assert result.value == (pytest.approx(50 / 51, rel=1e-14, abs=1e-14) if method == "beta" else 1)
    else:
        assert result.statistics.interval.status == "unavailable"
        assert "imported" in result.statistics.interval.reason
        assert result.value == (None if method == "beta" else 1)


def test_partial_count_denominator_and_public_signed_correction(injected):
    options = BinaryInferenceOptions(method="hoeffding", sampling_model="iid_bernoulli")
    choice = make_choice(inference=options)
    injected.payloads = {"group_0": {"0": 3, "1": 1}}
    partial = analyze(choice, acquire(choice, injected))
    assert partial.value == 0.5
    assert not partial.statistics.fixed_time and partial.statistics.interval.status == "unavailable"
    fitted = make_choice(mitigation=BinaryReadoutMitigation(calibration_shots=10))
    injected.payloads = {"science_0": {"0": 100}, "zero_0": {"0": 6, "1": 4}, "one_0": {"0": 4, "1": 6}}
    solved = nwqlib.solve(fitted)
    assert solved.value == pytest.approx(5.0, rel=1e-14, abs=1e-14)
    restored = ExpectationAnalysis.model_validate_json(solved.model_dump_json())
    assert restored.value == solved.value and restored.statistics.raw_value == 1.0


def test_whole_register_hzh_reuses_actual_occupation_preparation(injected):
    choice = make_choice(preparation="hzh", occupations=(1,))
    selected = next((block for block in choice.blocks if block.record.signature.name == "state"))
    assert selected._payload is choice.problem.state
    assert [(port.name, port.width) for port in selected.record.signature.quantum] == [("system", 1)]
    assert [(gate.gate, gate.qubits) for gate in selected.record.decomposition] == [
        ("h", (0,)),
        ("z", (0,)),
        ("h", (0,)),
    ]
    injected.payloads = {"group_0": {"1": 100}}
    result = analyze(choice, acquire(choice, injected))
    assert result.value == -1 and result.physical_scale == choice.problem.state.preparation.physical_scale


@pytest.mark.parametrize("context", ["compiler", "configuration", "pivot"])
def test_receipt_context_and_inference_associations_reject_before_reconstruction(
    injected, monkeypatch, context
):
    """Forge compiler, configuration and pivot associations while poisoning inference to require
    earlier rejection.
    """
    import nwqlib.algorithms.expectation as owner
    from nwqlib.execution import RegisterMap

    choice = make_choice(
        q=2, terms=(("XY", 1.0), ("IZ", 1.0)), mitigation=BinaryReadoutMitigation(calibration_shots=1000)
    )
    assert {setting.pivot for setting in choice.reconstruction.settings if setting.role == "science"} == {
        0,
        1,
    }
    injected.payloads = {
        "science_0": {"0": 80, "1": 20},
        "science_1": {"0": 80, "1": 20},
        "zero_0": {"0": 1000},
        "one_0": {"1": 1000},
        "zero_1": {"0": 1000},
        "one_1": {"1": 1000},
    }
    run = acquire(choice, injected)
    result = analyze(choice, run)
    parity = next((block for block in choice.blocks if block.record.choice == "XY"))
    assert [(p.gate, p.qubits) for p in parity.record.decomposition] == [
        ("sdg", (0,)),
        ("h", (0,)),
        ("h", (1,)),
        ("cx", (0, 1)),
    ]
    assert len(choice.experiments) == 6 and result.value == 1.2
    monkeypatch.setattr(
        owner, "infer_binary", lambda *a, **k: pytest.fail("forged context reached inference")
    )
    bad_trace = run.trace.revise(events=(run.trace.events[0].revise(shots=99), *run.trace.events[1:]))
    with pytest.raises(ValueError, match="pivot/readout"):
        choice.method.analyze(
            choice, RunData(run.observations, trace=bad_trace, receipts=run.prepared_artifacts), settings={}
        )
    receipts = list(run.prepared_artifacts)
    foreign = receipts[-1].revise(
        **{
            "compiler": dict(compiler=receipts[-1].compiler.revise(version="other")),
            "configuration": dict(backend_configuration_id=choice.content_id),
            "pivot": dict(logical_to_native=(1, 0)),
        }[context]
    )
    chunk = run.observations.chunks[-1].revise(prepared_id=foreign.content_id)
    events = (
        *run.trace.events[:-1],
        run.trace.events[-1].revise(prepared_id=foreign.content_id, observation_id=chunk.content_id),
    )
    submission = run.trace.submissions[-1]
    submission = submission.revise(items=(submission.items[0].revise(prepared_id=foreign.content_id),))
    trace = run.trace.revise(
        events=events,
        local_prepared_ids=(*run.trace.local_prepared_ids[:-1], foreign.content_id),
        submissions=(*run.trace.submissions[:-1], submission),
    )
    with pytest.raises(ValueError, match="target/compiler|native parity pivot"):
        choice.method.analyze(
            choice,
            RunData(
                ObservationView(chunks=(*run.observations.chunks[:-1], chunk)),
                trace=trace,
                receipts=(*receipts[:-1], foreign),
            ),
            settings={},
        )
    wrong_layout = run.observations.chunks[0].revise(
        quantum_layout=(RegisterMap(name="bit_0", bits=(1,)), RegisterMap(name="bit_1", bits=(0,)))
    )
    trace = run.trace.revise(
        events=(run.trace.events[0].revise(observation_id=wrong_layout.content_id), *run.trace.events[1:])
    )
    with pytest.raises(ValueError, match="pivot/readout"):
        choice.method.analyze(
            choice,
            RunData(
                ObservationView(chunks=(wrong_layout, *run.observations.chunks[1:])),
                trace=trace,
                receipts=run.prepared_artifacts,
            ),
            settings={},
        )
    wrong = result.revise(statistics=result.statistics.revise(inference_options_id=choice.method.content_id))
    with pytest.raises(ValueError, match="inference/setting"):
        wrong.validate_plan(choice)


def test_declared_workload_caps_precede_all_preparation(injected):
    selected = make_choice(mitigation=BinaryReadoutMitigation(calibration_shots=1000))
    with pytest.raises(ValueError, match="shot"):
        nwqlib.prepare(selected, limits=ExecutionLimits(max_total_shots=2099))
    assert injected.lowered == [] and injected.calls == []


def test_result_save_load_and_reanalysis_share_original_counts(injected, tmp_path, monkeypatch):
    selected = make_choice(
        inference=BinaryInferenceOptions(method="hoeffding", sampling_model="iid_bernoulli"),
        mitigation=BinaryReadoutMitigation(calibration_shots=1000),
    )
    injected.payloads = {"science_0": {"0": 80, "1": 20}, "zero_0": {"0": 1000}, "one_0": {"1": 1000}}
    result = nwqlib.solve(selected)
    assert result.value == 0.6 and result.statistics.fixed_time
    path = result.save(tmp_path / "result")
    monkeypatch.setattr(ExpectationMethod, "plan", lambda *a, **k: pytest.fail("load replanned"))
    loaded = nwqlib.load_result(path)
    assert loaded.content_id == result.content_id and loaded.statistics == result.statistics
    before = injected.calls.copy()
    revised = loaded.analyze()
    assert revised.value == result.value and revised.statistics == result.statistics
    assert revised.origin.invocation_id != result.origin.invocation_id and injected.calls == before


def test_result_with_source_preparation_groups_loads_with_its_identity(injected, tmp_path):
    selected = make_choice()
    injected.payloads = {"group_0": {"0": 75, "1": 25}}
    run = acquire(selected, injected)
    handle = prepare(selected.resolve("group_0"), run=run, runtime=RuntimeOptions(seed=8))
    run.collect(submit(handle, run=run))
    result = analyze(selected, run)._attach(selected, run.data)
    path = result.save(tmp_path / "good")
    assert nwqlib.load_result(path).content_id == result.content_id
