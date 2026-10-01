"""Finite Pauli expectation with one selected preparation and actual data.

For O=c0 I+sum(cj Pj), the normalized observable is c0+sum(cj <Pj>).
With v=psi/||psi||, <Pj> means <v|Pj|v>, and the quadratic form multiplies
the normalized value by s=||psi||^2. Four acquisition paths produce this
value: exact grouped Pauli readout (the default), measured counts with
binary inference (one experiment per qubit-wise commuting group, or with
affine readout calibration one parity experiment per nonidentity term),
one provider-managed estimate of the whole weighted sum, and a classical
matvec in the original dimension.

The shots selected from an absolute sampling tolerance follow Hoeffding
(1963), doi:10.1080/01621459.1963.10500830, Theorem 2, for each label's
parity, with a union bound over the measured labels and a triangle
inequality (ExpectationMethod.sampling_shots). Grouping unmitigated counts and one
parity experiment per term under mitigation are NWQLib's design choices.
The binary inference and the affine calibration
model are derived in evidence/binary.py and docs/algorithms/expectation.md.
Sampling, selected binary inference, calibration and provider uncertainty remain
separate from preparation, native and physical-model accuracy.

Reading order: ExpectationMethod.plan selects the path, _measured_program
builds the group or parity experiments, and ExpectationMethod.analyze
dispatches to _exact_value, _analyze_measured, _analyze_estimated or
_analyze_classical. ExpectationAnalysis.validate_plan checks a stored Result
against its Plan without repeating inference.
"""

from math import fsum
from typing import Annotated, ClassVar, Literal, NamedTuple

from pydantic import Field, StrictBool, StrictInt, model_validator

from nwqlib.algorithms.protocol import Method, AlgorithmDescriptor, ApplicabilityError
from nwqlib.blocks import SelectedBlock, SelectedConstruction, select_preparation
from nwqlib.core.analysis import Result, capture_analysis_origin
from nwqlib.core.planning import (
    Experiment,
    ObservableEstimateSpec,
    ObservationSpec,
    Plan,
    Realization,
)
from nwqlib.core.records import ContentID, InputRef, PositiveInt, Rational, Real, Record, Source, Text, Unit
from nwqlib.evidence.binary import (
    BINARY_SOURCE,
    MAX_BINARY_COUNT,
    BinaryCorrection,
    BinaryEstimate,
    BinaryInterval,
    BinaryPopulation,
    BinaryInferenceOptions,
    BinaryReadoutMitigation,
    _float,
    correct_binary,
    infer_binary,
)
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.evidence.statistics import (
    EstimatorContribution,
    IndependenceLaw,
    VarianceAssessment,
    linear_variance,
)
from nwqlib.evidence.error_model import ErrorFrame, ErrorModel, ErrorTerm, FramedFact, exact_readout_sampling
from nwqlib.evidence._work import ExactArithmetic
from nwqlib.execution import EstimateValue, PauliValue, PreparedArtifact
from nwqlib.ir import (
    Allocate,
    BlockCall,
    ClassicalValue,
    Definition,
    Measure,
    MeasurementBatch,
    MetadataRef,
    PortMap,
    Register,
    Release,
    Sequence,
    Setting,
)
from nwqlib.ir.validation import _Admission, admitted_program
from nwqlib.operators.access import Count, _check_bytes
from nwqlib._limits import DEFAULT_MAX_BYTES
from nwqlib.problems.inputs import PhysicalScale, ingest_occupation
from nwqlib.problems.records import Accuracy, Expectation, NormalizedExpectation, QuadraticForm
from .expectation_metadata import _DESCRIPTOR, _METHOD

_ANALYSIS_SOURCE = Source(
    name="finite_pauli_expectation.analysis",
    version="4",
    domain="stored observable reduction or binary inference/calibration in the requested normalized or physical frame",
    reference="nwqlib.algorithms.expectation.ExpectationMethod.analyze",
)
_ERROR_SOURCES = (
    "native_preparation",
    "native_simulation",
    "physical_model",
    "sampling",
    "reconstruction",
)


def _error_model(problem, output, construction, *, mitigated=False):
    """Declare every expectation error source as unknown in the requested output frame.

    The Method supplies no bound for preparation, native simulation, physical
    model, sampling or reconstruction error, and adds calibration transfer
    when mitigation is selected. A sampling fact from _sampling_facts can
    fill only the sampling term, so a total-error assessment stays
    INCONCLUSIVE while the other terms remain unknown.
    """
    frame = output.frame(problem)
    names = (*_ERROR_SOURCES, "calibration_transfer") if mitigated else _ERROR_SOURCES
    return ErrorModel(
        output_id=output.content_id,
        subject_id=problem.content_id,
        construction_id=construction.content_id,
        frame=frame,
        required_sources=names,
        terms=tuple(
            ErrorTerm(
                name=name,
                stage="execution",
                source=_METHOD,
                formula="already propagated absolute error in the requested output frame",
                coverage=name,
                fact=FramedFact(
                    frame=frame,
                    bindings=(),
                    fact=Fact(
                        quantity=name,
                        unit=frame.unit,
                        scope=frame.scope,
                        availability="unknown",
                        reason=f"no supplied {name} bound for the requested {output.kind}",
                    ),
                ),
            )
            for name in names
        ),
        source=_METHOD,
    )


def _classical_constant(operator, *, max_bytes):
    """Return c when the stored observable is exactly c*I, else None.

    Classical planning recognizes an observable equal to cI by testing its
    stored nonzero support and constant diagonal, or its identity-only Pauli
    table. Such an observable has normalized expectation c for every nonzero
    state, so it needs no acquisition. A general observable's mean diagonal
    does not establish this property.

    A Pauli table qualifies when every row with a nonzero coefficient has
    ``x|z == 0``; the value is the fsum of the real coefficients. Separate
    cancelling nonidentity rows are deliberately not recognized. For
    admitted canonical dense/CSR/CSC storage every coordinate is unique, so
    the number of nonzero stored values equals the number of nonzero
    diagonal entries exactly when every off-diagonal value is zero.
    Explicit sparse zeros do not affect the counts, and a missing sparse
    diagonal coordinate reads as zero. Then the diagonal must be constant,
    and ``diagonal[0].real`` is returned, not the trace mean: diag(1, 2)
    is not a scalar observable. The work is O(D²) dense or O(nnz + D)
    compressed. A raw duplicate-bearing sparse matrix before canonical
    admission is outside this test. No eigensolve or format conversion is
    performed.
    """
    import numpy as np

    if "pauli_terms" in operator.manifest.access:
        table = operator.pauli_terms()
        if np.any((table.x | table.z)[table.coefficients != 0]):
            return None
        return fsum(float(c.real) for c in table.coefficients)
    d = operator.basis.dimension
    # The payload reserve and 32*d bytes cover storage access, a complex
    # sparse diagonal and its Boolean comparison scratch while numerical
    # nonzero counts scan the stored values. Dense diagonal access is a view.
    _check_bytes(
        operator.manifest.payload_bytes + 32 * d, max_bytes, "expectation scalar recognition"
    )
    compressed = operator.reference.representation in ("csr", "csc")
    matrix = operator._data if compressed else operator.dense_array()
    diagonal = matrix.diagonal()
    values = matrix.data if compressed else matrix
    if np.count_nonzero(values) != np.count_nonzero(diagonal):
        return None
    return float(diagonal[0].real) if np.all(diagonal == diagonal[0]) else None


def _classical_action_requirements(operator, state, *, max_bytes):
    """Admit the classical kernel and return its (bytes, work) before any vector exists.

    bytes adds the scaled observable copy with its matvec envelope
    (_scaled_observable_requirements) to the state preparation arrays
    (preparation_requirements). work adds their counted products. A custom
    circuit's unknown preparation work is left out. Raises ValueError when
    bytes exceeds max_bytes.
    """
    from nwqlib.algorithms._eigen_inputs import preparation_requirements
    from nwqlib.operators.inputs import _scaled_observable_requirements

    action_bytes, action_work = _scaled_observable_requirements(operator)
    prep_bytes, prep_work = preparation_requirements(state)
    size = prep_bytes + action_bytes
    _check_bytes(size, max_bytes, "classical expectation scaled action and preparation")
    return size, action_work + (prep_work if prep_work is not None else 0)


class PauliCoefficient(Record):
    """One physical observable coefficient; no encoding normalization is applied."""

    label: Text
    coefficient: Real


class ExpectationSetting(Record):
    """One selected QWC group, parity or calibration acquisition.

    Unmitigated sampled readout acquires one setting of role ``group`` per
    qubit-wise commuting group. labels lists the group's nonzero
    nonidentity labels in first-fit member order, basis is the group's
    accumulated local Pauli basis (one I/X/Y/Z per qubit, qubit zero
    rightmost), and qubits lists the group's support, the non-I sites of
    basis, in increasing order; its labels read only these sites. The
    experiment measures the whole width-q register ``system`` into the
    width-q classical value ``readout``, bit i holding qubit i, and each
    label's parity is decoded from the support bits of that joint outcome.

    With mitigation, a ``science`` setting measures one label's parity on
    its pivot, the qubit whose single measured bit carries the parity, the
    highest qubit index at which the label is not I (its leftmost non-I
    character, since labels list the highest qubit first). Calibration
    settings measure the same pivot so that their fitted channel describes
    that readout.
    """

    name: Text
    role: Literal["group", "science", "calibration_zero", "calibration_one"]
    pivot: Count | None = None
    label: Text | None = None
    labels: tuple[Text, ...] = ()
    basis: Text | None = None
    qubits: tuple[Count, ...] = ()

    @model_validator(mode="after")
    def _role(self):
        group = self.role == "group"
        if group != bool(self.labels) or group != (self.basis is not None) or group != bool(self.qubits):
            raise ValueError("group settings require their labels, basis and support qubits; others do not")
        if group == (self.pivot is not None):
            raise ValueError("parity and calibration settings require a pivot; group settings do not")
        if (self.role == "science") != (self.label is not None):
            raise ValueError("science settings require a Pauli label; other settings do not")
        if group and (tuple(sorted(set(self.qubits))) != self.qubits or len(set(self.labels)) != len(self.labels)):
            raise ValueError("group settings require distinct labels and increasing support qubits")
        # Analysis decodes each label from the support bits of the whole-register
        # outcome, so every label must use the group basis on its non-I sites,
        # and the support qubits must be exactly the non-I sites of that basis.
        if group and (
            set(self.basis) - set("IXYZ")
            or tuple(bit for bit, axis in enumerate(reversed(self.basis)) if axis != "I") != self.qubits
            or any(len(label) != len(self.basis) or any(a not in ("I", b) for a, b in zip(label, self.basis))
                   for label in self.labels)
        ):
            raise ValueError("group labels must match the group basis on their non-I sites, which are the support qubits")
        return self


class ExpectationReconstruction(Record):
    """Physical observable and state-scale metadata for sum c_P <P>.

    terms keep the physical coefficients without encoding normalization.
    settings are the measured QWC group experiments, or with mitigation the
    parity and calibration experiments, empty on the exact, provider and
    classical paths. A classical Plan stores either
    constant, when the observable is a scalar matrix or the state is zero, or
    operator_exponent, the binary exponent e by which the kernel scaled O
    before its matvec. Analysis restores 2**e in exact arithmetic.
    """

    schema_version: Literal[4] = 4
    terms: tuple[PauliCoefficient, ...]
    physical_scale: PhysicalScale
    settings: tuple[ExpectationSetting, ...] = ()
    output_kind: Literal["normalized_expectation", "quadratic_form"] = "normalized_expectation"
    classical: bool = False
    constant: Real | None = None
    operator_exponent: StrictInt = 0
    convention: Literal["normalized: v†Ov; quadratic: norm(psi)^2 v†Ov; v=psi/norm(psi)"] = (
        "normalized: v†Ov; quadratic: norm(psi)^2 v†Ov; v=psi/norm(psi)"
    )


class ExpectationStatistics(Record):
    """Performed binary inference and data dependencies, with conditional scope.

    variance is empirical or delta-method under its kept covariance premises,
    not total physical variance. interval has the selected statistical meaning
    and no independent physical-model guarantee.

    Attributes:
        inference_options_id: Content identity of the BinaryInferenceOptions this analysis used.
        populations: Without mitigation, one BinaryEstimate per nonzero nonidentity label, named ``<group setting>:<label>``, in group and member order; its marginal parity counts come from its group's population, and the labels of one group share its sources, observations and receipts, which count as one exposure. With mitigation, one BinaryEstimate per predeclared science and calibration setting, in setting order. Each keeps raw counts and any posterior result.
        corrections: One affine BinaryCorrection per science term, with its signed fitted point, when mitigation is selected, otherwise empty.
        receipt_ids: Preparation receipts associated with the counted sources.
        raw_value: Uncorrected empirical value in the requested output frame.
        variance_kind: ``empirical``, ``delta_method`` for fitted calibration, or ``posterior`` for Beta inference.
        variance: Output-frame variance under the stored covariance premises, or None. The empirical variance of grouped readout has one contribution per group, the empirical variance of its weighted score mean, so covariance between labels of one group is included.
        variance_unavailable: Reason variance is None.
        interval: Linear image of the simultaneous per-population intervals in the output frame, or None for point inference.
        fixed_time: Whether the data qualify for a fixed-time interval: exactly one complete original acquisition per setting (group, parity or calibration) with its receipts, no blocking population reason, and no cancellation or early termination.
        fixed_time_reason: Why fixed_time holds or fails.
        applicability_reason: Why missing receipts leave target, compiler or layout applicability unverified, or None.
        independence_reason: Why independence across settings is unverified because they share one fixed sampling stream, or None. Labels of one QWC group share their group's shots by design; independence concerns distinct groups.
        source: Implementation source of the binary inference.
    """

    inference_options_id: ContentID
    populations: tuple[BinaryEstimate, ...]
    corrections: tuple[BinaryCorrection, ...]
    receipt_ids: tuple[ContentID, ...]
    raw_value: Real | None
    variance_kind: Literal["empirical", "delta_method", "posterior"]
    variance: VarianceAssessment | None
    variance_unavailable: Text | None
    interval: BinaryInterval | None
    fixed_time: StrictBool
    fixed_time_reason: Text
    applicability_reason: Text | None
    independence_reason: Text | None = None
    source: Source = BINARY_SOURCE

    @model_validator(mode="after")
    def _variance_availability(self):
        if (self.variance is None) != (self.variance_unavailable is not None):
            raise ValueError("unavailable variance requires its explicit reason")
        return self


class ExpectationAnalysis(Result):
    """Requested normalized or physical value; raw populations preserve their own frame.

    Attributes:
        value: Requested normalized expectation or quadratic form, or None when data are missing or the value is unrepresentable.
        physical_scale: Norm of the supplied state as a mantissa and binary exponent.
        missing: Nonidentity Pauli labels without data.
        statistics: Count inference record of measured parity data, otherwise None.
        estimates: Original provider estimates in the normalized-state observable frame.
        unavailable: Reason value is None when no label is missing.
        inference: BinaryInferenceOptions used by a measured analysis, otherwise None.
    """

    value: Real | None
    physical_scale: PhysicalScale
    missing: tuple[Text, ...]
    statistics: ExpectationStatistics | None = None
    estimates: tuple[EstimateValue, ...] = ()
    unavailable: Text | None = None
    inference: BinaryInferenceOptions | None = None

    def _summary_lines(self):
        """Return the printed summary: the value, missing data and the uncertainty scope."""
        title = (
            "Expectation"
            if self._plan is None
            else "Quadratic form"
            if self.plan.output.kind == "quadratic_form"
            else "Normalized expectation"
        )
        lines = [f"{title}: {self._scalar_text(self.value)}{self._unit_text()}"]
        if self.missing:
            lines.append(f"Partial data: {len(self.missing)} missing Pauli expectations")
        if self.unavailable:
            lines.append(self.unavailable)
        if self.statistics is not None:
            lines.append("Count inference uses its recorded sampling and calibration premises")
        elif self.estimates:
            lines.append(
                "Provider-estimated observable; uncertainty has the provider's stated scope"
            )
            if len(self.estimates) == 1 and self._plan is not None:
                estimate = self.provider_output_estimates[0]
                if estimate["standard_error"] is not None:
                    lines.append(
                        f"Provider standard error in output frame: {estimate['standard_error']:.6g}"
                    )
                else:
                    lines.append(estimate["uncertainty_unavailable"])
        return lines

    @property
    def provider_output_estimates(self):
        """Cheap output-frame views; estimates keeps immutable original provider values.

        SEM keeps the provider's original meaning. Repeated estimates have no
        implied covariance or combined SEM. No output view is persisted twice.
        """
        rec = self.plan.reconstruction
        views = []
        for item in self.estimates:
            error = _output_value(item.standard_error, rec)
            views.append(
                dict(
                    source_id=item.content_id,
                    value=_output_value(item.value, rec),
                    standard_error=error,
                    uncertainty_unavailable=(
                        None
                        if error is not None
                        else item.uncertainty_unavailable
                        or "physical provider uncertainty is unrepresentable"
                    ),
                )
            )
        return tuple(views)

    def validate_plan(self, plan):
        """Bind kept term availability and scale to the exact reconstruction.

        A stored Result must agree with its Plan's selected output, state
        scale, identity constant and provider reduction. For measured data it
        must keep the complete predeclared setting family, the selected
        interval kind, family size and probability, the fixed-time and
        independence status, the variance role and each correction's exact
        science and calibration populations. Only the stored per-setting
        points are recombined. No inference is rerun and nothing is acquired.
        """
        self._validate_common_plan(plan, Plan)
        if (
            plan.reconstruction.output_kind != plan.output.kind
            or plan.reconstruction.classical != (plan.execution == "classical")
            or plan.reconstruction.physical_scale != plan.problem.state.preparation.physical_scale
        ):
            raise ValueError(
                "expectation reconstruction differs from the original output/state scale"
            )
        inference = self.inference or plan.method.inference
        labels = {
            term.label
            for term in plan.reconstruction.terms
            if term.coefficient != 0.0 and set(term.label) != {"I"}
        }
        if (
            self.physical_scale != plan.reconstruction.physical_scale
            or len(set(self.missing)) != len(self.missing)
            or (not set(self.missing) <= labels)
            or ((self.value is None) != bool(self.missing or self.unavailable))
        ):
            raise ValueError("expectation result differs from its reconstruction/availability")
        if plan.reconstruction.classical:
            if plan.reconstruction.operator_exponent != plan._native.get("operator_exponent"):
                raise ValueError("classical observable exponent differs from original input binding")
            if self.statistics is not None or self.estimates or self.missing:
                raise ValueError(
                    "classical expectation cannot contain quantum acquisition statistics"
                )
            return
        if not labels:
            constant = fsum(
                (term.coefficient for term in plan.reconstruction.terms if term.coefficient != 0.0)
            )
            constant = _output_value(constant, plan.reconstruction)
            if self.value != constant:
                raise ValueError(
                    "identity expectation differs from its exact constant reconstruction"
                )
        if plan.method.estimate_precision is not None and plan.experiments:
            estimate_id = plan.experiments[0].observation.estimate.content_id
            if (
                self.statistics is not None
                or any((item.estimate_id != estimate_id for item in self.estimates))
                or len(self.estimates) != len(self.contribution_ids)
                or (
                    self.value != _output_value(_provider_mean(self.estimates), plan.reconstruction)
                )
            ):
                raise ValueError(
                    "provider estimates differ from the requested observable or kept point reduction"
                )
        elif self.estimates:
            raise ValueError("provider estimates require the selected provider acquisition")
        # Measured inference must keep its complete declared science/calibration
        # family, original contribution identities and selected coverage convention.
        if plan.reconstruction.settings:
            _validate_measured_family(self, plan, inference)
            _validate_interval_meaning(self.statistics, plan, inference)
            _validate_corrections_and_value(self, plan)
        elif self.statistics is not None:
            raise ValueError("exact or identity-only expectation has no sampled inference record")


def _variance_kind(plan, inference):
    """Return the statistical role of the output variance for this inference.

    ``posterior`` for Beta inference, where the variance is that of the
    posterior. ``delta_method`` with readout mitigation, where the fitted
    correction is linearized. ``empirical`` otherwise, the sample-mean
    variance of the raw counts.
    """
    return (
        "posterior"
        if inference.method == "beta"
        else "delta_method"
        if plan.method.mitigation is not None
        else "empirical"
    )


def _science_parities(settings):
    """Return (label, population name, setting) per measured science parity, in population order.

    A QWC group setting supplies one parity per label, named
    ``<setting>:<label>``. With mitigation each science setting is one
    parity named by its setting.
    """
    parities = []
    for setting in settings:
        if setting.role == "group":
            parities.extend((label, f"{setting.name}:{label}", setting) for label in setting.labels)
        elif setting.role == "science":
            parities.append((setting.label, setting.name, setting))
    return tuple(parities)


def _population_names(settings):
    """Return the stored population names: group labels, or every parity and calibration setting.

    Their number is the family size F of the simultaneous intervals: L
    labels for grouped readout, all science and calibration settings with
    mitigation.
    """
    if settings and settings[0].role == "group":
        return tuple(name for _, name, _ in _science_parities(settings))
    return tuple(item.name for item in settings)


def _validate_measured_family(result, plan, inference):
    """Check that stored measured statistics cover exactly the Plan's setting family.

    The statistics must name the inference options, list one population per
    predeclared label (grouped readout) or setting (mitigation) in order,
    and use this Plan's construction. The labels of one group must share
    its source, observation and preparation identities, which are counted
    once. The distinct observation, reused-observation and receipt
    identities must equal the Result's contributions and the populations'
    preparation groups. A combined interval must keep the family size.
    Raises ValueError on the first disagreement.
    """
    statistics = result.statistics
    settings = plan.reconstruction.settings
    names = _population_names(settings)
    if (
        statistics is None
        or statistics.inference_options_id != inference.content_id
        or tuple((item.population.name for item in statistics.populations)) != names
        or any((item.options_id != inference.content_id for item in statistics.populations))
        or (result.construction_id != plan.construction.content_id)
    ):
        raise ValueError(
            "measured expectation statistics differ from their selected inference/setting family"
        )
    # One group exposure is counted once even though several labels use it.
    shared = {}
    for item in statistics.populations:
        population = item.population
        key = population.name.split(":", 1)[0] if settings[0].role == "group" else population.name
        exposure = (
            population.source_ids,
            population.observation_ids,
            population.reused_observation_ids,
            population.preparation_ids,
            population.zeros + population.ones,
        )
        if shared.setdefault(key, exposure) != exposure:
            raise ValueError("labels of one measured group differ in their shared acquisition")
    observed = tuple(
        identity for exposure in shared.values() for identity in (*exposure[1], *exposure[2])
    )
    sources = {
        identity
        for item in statistics.populations
        for group in item.population.preparation_ids
        for identity in group
    }
    if (
        len(set(observed)) != len(observed)
        or set(observed) != set(result.contribution_ids)
        or len(set(statistics.receipt_ids)) != len(statistics.receipt_ids)
        or (set(statistics.receipt_ids) != sources)
    ):
        raise ValueError(
            "binary inference data differ from their actual result contributions"
        )
    if statistics.interval is not None and statistics.interval.family_size != len(names):
        raise ValueError(
            "binary interval changed the predeclared label/science/calibration family"
        )


def _validate_interval_meaning(statistics, plan, inference):
    """Check that every stored interval keeps the meaning of the selected inference.

    Each interval must have the selected kind (fixed-time, time-uniform or
    Bayesian), family size and probability 1-delta. An unknown sampling
    model, incomplete fixed-time data or settings that share a fixed stream
    must leave the corresponding intervals, variance or posterior
    unavailable. The variance must have the role _variance_kind selects.
    Raises ValueError on the first disagreement.
    """
    family_size = len(_population_names(plan.reconstruction.settings))
    # Fixed-time, time-uniform and Bayesian intervals have distinct meanings.
    # Missing sampling assumptions cannot be repaired by relabeling an interval.
    method = inference.method
    interval_kind = {
        "point": None,
        "hoeffding": "fixed_time",
        "anytime_hoeffding": "time_uniform",
        "beta": "bayesian",
    }[method]
    intervals = (statistics.interval, *(item.interval for item in statistics.populations))
    if any(
        (
            (interval is None) != (interval_kind is None)
            or (
                interval is not None
                and (
                    interval.kind != interval_kind
                    or interval.family_size != family_size
                    or interval.probability != 1.0 - inference.failure_probability
                )
            )
            for interval in intervals
        )
    ):
        raise ValueError(
            "binary interval differs from its selected statistical kind/family"
        )
    if (
        interval_kind is not None
        and inference.sampling_model == "unknown"
        and any((interval.status != "unavailable" for interval in intervals))
    ):
        raise ValueError("unspecified sampling model cannot supply a statistical interval")
    if (
        method == "hoeffding"
        and (not statistics.fixed_time)
        and any((interval.status != "unavailable" for interval in intervals))
    ):
        raise ValueError("incomplete/stopped acquisition cannot keep a fixed-time interval")
    if statistics.independence_reason is not None and (
        statistics.variance is not None
        or (
            method == "beta"
            and any((interval.status != "unavailable" for interval in intervals))
        )
    ):
        raise ValueError(
            "known dependent settings cannot claim independent covariance or a product posterior"
        )
    variance_kind = _variance_kind(plan, inference)
    if statistics.variance_kind != variance_kind:
        raise ValueError(
            "binary variance differs from the selected estimator's statistical role"
        )


def _validate_corrections_and_value(result, plan):
    """Check fitted corrections against their data and recombine the stored value.

    With mitigation, each science term needs one correction built from its
    own science population and the zero and one calibration populations of
    its pivot, with the Plan's mitigation options. The Result's value must
    equal _weighted_sum of the stored per-label points (the corrected
    points with mitigation), and statistics.raw_value the _weighted_sum of
    the raw means. Nothing is inferred again. Raises ValueError on the first
    disagreement.
    """
    statistics = result.statistics
    settings = plan.reconstruction.settings
    # Associate each fitted correction with the actual science and pivot
    # calibration populations, even when two numeric estimates happen to agree.
    populations = {item.population.name: item for item in statistics.populations}
    science = tuple((item for item in settings if item.role == "science"))
    if plan.method.mitigation is None:
        if statistics.corrections:
            raise ValueError("uncalibrated expectation cannot contain fitted corrections")
    elif len(statistics.corrections) != len(science):
        raise ValueError(
            "calibrated expectation requires one kept correction per science term"
        )
    else:
        calibration = {
            (item.role, item.pivot): populations[item.name].population.content_id
            for item in settings
            if item.role != "science"
        }
        for item, correction in zip(science, statistics.corrections, strict=True):
            if (
                correction.science_id != populations[item.name].population.content_id
                or correction.zero_id != calibration["calibration_zero", item.pivot]
                or correction.one_id != calibration["calibration_one", item.pivot]
                or (correction.options_id != plan.method.mitigation.content_id)
            ):
                raise ValueError(
                    "fitted correction changed its exact science/calibration data or options"
                )
    selected = {
        label: (
            populations[name].point
            if plan.method.mitigation is None
            else statistics.corrections[index].point
        )
        for index, (label, name, _) in enumerate(_science_parities(settings))
    }
    raw = {label: populations[name].raw_mean for label, name, _ in _science_parities(settings)}
    if result.value != _weighted_sum(
        plan.reconstruction.terms, selected, plan.reconstruction
    ) or statistics.raw_value != _weighted_sum(
        plan.reconstruction.terms, raw, plan.reconstruction
    ):
        raise ValueError(
            "physical expectation value differs from its selected inference and scale"
        )


class _BinaryData(NamedTuple):
    """Admitted count populations and the facts that decide which inference is legal.

    Attributes:
        populations: One BinaryPopulation per stored population name (_population_names): per label for grouped readout, per setting with mitigation.
        receipt_ids: Content ids of the preparation receipts that were checked and used.
        contribution_ids: Content ids of every observed chunk, in observation order.
        applicability_reason: Why missing receipts leave target, compiler or layout applicability unverified, or None.
        population_reasons: Setting name to the reason its population cannot support an interval: unknown sampling semantics, a stream reused within the setting, imported data or a broken chronological prefix.
        independence_reason: Why a fixed sampling stream shared across settings leaves their independence unverified, or None.
        fixed_time: Whether the data are one complete original acquisition per setting with its receipts, no blocking population reason, and no cancellation or early termination.
        fixed_time_reason: Why fixed_time holds or fails.
        histograms: Group setting name to its joint histogram over distinct sources, outcome index to count, for grouped readout; empty with mitigation.
    """

    populations: tuple
    receipt_ids: tuple
    contribution_ids: tuple
    applicability_reason: str | None
    population_reasons: dict
    independence_reason: str | None
    fixed_time: bool
    fixed_time_reason: str
    histograms: dict


def _expected_observations(plan, admission):
    """Return setting name to (Realization, ObservationSpec) of each predeclared setting.

    A group or science setting must request plan.shots counts and a calibration
    setting the mitigation's calibration_shots, both as unconditional
    populations. Raises ValueError otherwise.
    """
    settings = plan.reconstruction.settings
    experiments = {item.name: item for item in plan.experiments}
    expected = {}
    for item in settings:
        experiment = experiments[item.name]
        setting, observation = experiment._resolved_observation(admission)
        requested = (
            plan.shots if item.role in ("group", "science") else plan.method.mitigation.calibration_shots
        )
        if (
            setting != item.name
            or observation.kind != "counts"
            or observation.shots != requested
            or (observation.population != "unconditional")
        ):
            raise ValueError(
                "measured expectation acquisition differs from its declared science/calibration role"
            )
        realization = Realization(plan_id=plan.content_id, experiment=item.name, bindings=())
        expected[item.name] = (realization, observation)
    return expected


def _label_supports(labels):
    """Return each label's support mask x|z, bit i for qubit i (the rightmost letter is qubit 0)."""
    return tuple(
        sum(1 << bit for bit, axis in enumerate(reversed(label)) if axis != "I") for label in labels
    )


def _accumulated_populations(settings, groups, reused, associations, *, width):
    """Return the stored populations and each group's joint histogram from distinct sources.

    groups maps each setting to its distinct sources and their counts: an
    (n0, n1) pair for a parity or calibration setting, and the (outcome
    index, count) pairs of the joint histogram for a QWC group. reused
    maps a setting to the further observations of those sources and
    associations to the receipts of each source. Only distinct sources add
    counts, and the accumulated total must stay within MAX_BINARY_COUNT.

    A group's label j is read from its joint histogram: n1 counts the
    outcomes with odd ``popcount(outcome & support_j)``, where ``support_j =
    x_j | z_j`` in measured bit order (classical bit i reads qubit i), and
    n0 the rest. After H on the X sites and S† then H on the Y sites, the
    parity of the support bits is zero exactly for the +1 eigenvalue of the
    label. Every label of a group keeps the group's sources, observations
    and receipts, so one group shot supplies every parity in that group.

    Returns:
        (populations, histograms): the populations in _population_names
        order, and group setting name to its outcome-index counts.
    """
    from itertools import compress

    import numpy as np

    populations, histograms = [], {}
    for item in settings:
        unique = groups[item.name]
        fields = dict(
            source_ids=tuple(unique),
            observation_ids=tuple((value[0] for value in unique.values())),
            reused_observation_ids=tuple(reused[item.name]),
            preparation_ids=tuple((tuple(associations[item.name][key]) for key in unique)),
        )
        if item.role != "group":
            n0, n1 = (0, 0)
            for _, pair in unique.values():
                if pair[0] + pair[1] > MAX_BINARY_COUNT - n0 - n1:
                    raise ValueError("accumulated binary count exceeds its exact binary64 integer domain")
                n0, n1 = (n0 + pair[0], n1 + pair[1])
            populations.append(BinaryPopulation(name=item.name, zeros=n0, ones=n1, **fields))
            continue
        joint, total = {}, 0
        for _, pairs in unique.values():
            for key, count in pairs:
                if count > MAX_BINARY_COUNT - total:
                    raise ValueError("accumulated binary count exceeds its exact binary64 integer domain")
                total += count
                joint[key] = joint.get(key, 0) + count
        histograms[item.name] = joint
        keys, counts = tuple(joint), tuple(joint.values())
        indices = np.array(keys, dtype=np.uint64) if width <= 64 else None
        for label, support in zip(item.labels, _label_supports(item.labels), strict=True):
            if indices is not None:
                odd = (np.bitwise_count(indices & np.uint64(support)) & 1).astype(bool).tolist()
            else:
                odd = [(key & support).bit_count() & 1 for key in keys]
            ones = sum(compress(counts, odd))
            populations.append(
                BinaryPopulation(name=f"{item.name}:{label}", zeros=total - ones, ones=ones, **fields)
            )
    return tuple(populations), histograms


def _prefix_reasons(trace, receipt_map, sources, groups, expected):
    """Return setting name to the reason its data are not a complete chronological prefix.

    Statistical inference needs every earlier prepared population of a
    setting, in trace order. A reason is recorded when the data come from an
    imported preparation without fresh history in this run, omit an earlier
    population, or include an incomplete or uncertain acquisition. A later
    event overwrites an earlier reason for the same setting.
    """
    updates = {}
    seen, gaps = (set(), set())
    unknown_gap = False
    local_sources = set(trace.local_prepared_ids)
    for event in trace.events:
        receipt = receipt_map.get(event.prepared_id)
        if receipt is None:
            unknown_gap = True
            continue
        name = receipt.realization.experiment
        if name not in groups:
            raise ValueError("prepared source is not a selected measured expectation setting")
        source = sources.resolve(event.prepared_id, event.attempt)
        if source.identity in seen:
            continue
        seen.add(source.identity)
        if source.identity not in groups[name]:
            gaps.add(name)
        elif source.kind != "fresh" and event.prepared_id not in local_sources:
            updates[name] = (
                "imported prepared population has no validated fresh chronological history in this run"
            )
        elif unknown_gap or name in gaps:
            updates[name] = (
                "observed data omit an earlier prepared population; statistical inference requires a chronological prefix"
            )
        elif event.status != "completed" or event.returned_shots != expected[name][1].shots:
            updates[name] = (
                "incomplete or uncertain acquisition cannot establish a complete observed prefix"
            )
    return updates


def _parity_counts(chunk, plan, trace, events, expected, layouts, support=None):
    """Admit one observed chunk as a parity or QWC group population and return its counts.

    trace.validate_observation first joins the chunk to its submission
    item. The chunk must then belong to this Plan and Run and to a
    predeclared setting (a key of expected), with that setting's
    realization and observation, no bindings, the unconditional population
    and quantum_circuit execution. Its trace event must be a completed
    quantum_circuit attempt with zero evaluations and the requested shots,
    and must record the chunk's preparation, identity and returned shots.
    The chunk's layouts must equal layouts, the logical registers and the
    classical readout that _measured_program declares.
    For a parity or calibration setting (support is None) the values may
    hold only "0" and "1" count bins; n0 counts outcome 0 of the measured
    pivot bit, which is the +1 eigenvalue for a science term's Pauli, and
    n1 counts outcome 1, and (n0, n1) is returned. For a QWC group the
    histogram, read through ObservationChunk.histogram (indices, or packed
    indices above 64 bits), spans the whole q-bit readout, and support is
    the mask of the group's non-I sites. Each outcome is restricted to
    ``outcome & support``: summing the counts over the other measured sites
    gives the marginal on the group's support, which is all that its labels'
    parities read. The (restricted outcome, count) pairs are returned in
    increasing outcome order. Raises ValueError when a check fails or the
    returned shots exceed MAX_BINARY_COUNT.
    """
    trace.validate_observation(chunk)
    event = events.get(chunk.attempt)
    if (
        chunk.experiment not in expected
        or chunk.plan_id != plan.content_id
        or chunk.run_id != trace.run_id
        or (event is None)
        or (event.status != "completed")
        or (event.prepared_id != chunk.prepared_id)
        or (event.observation_id != chunk.content_id)
        or (event.returned_shots != chunk.returned_shots)
        or (chunk.execution != "quantum_circuit")
        or (chunk.population != "unconditional")
    ):
        raise ValueError(
            "binary contribution has incompatible actual Plan, attempt, population or setting"
        )
    realization, observation = expected[chunk.experiment]
    quantum_layout, classical_layout = layouts
    if (
        chunk.realization_id != realization.content_id
        or chunk.bindings
        or chunk.setting != chunk.experiment
        or (chunk.observation != observation)
        or (event.execution != "quantum_circuit")
        or (event.evaluations != 0)
        or (event.shots != observation.shots)
        or (tuple(((r.name, r.bits) for r in chunk.quantum_layout)) != quantum_layout)
        or (tuple(((r.name, r.bits) for r in chunk.classical_layout)) != classical_layout)
    ):
        raise ValueError("binary contribution differs from its fixed actual parity pivot/readout layout")
    # chunk.observation equals the selected counts observation here.
    histogram = chunk.histogram()
    width = sum(len(bits) for _, bits in classical_layout)
    if histogram.entries and histogram.width != width:
        raise ValueError("binary contribution differs from its fixed actual parity pivot/readout layout")
    if chunk.returned_shots > MAX_BINARY_COUNT:
        raise ValueError("binary population exceeds the exact binary64 integer domain")
    if histogram.width <= 64:
        indices = histogram.indices().tolist()
    else:
        indices = [int.from_bytes(row.astype("<u8").tobytes(), "little") for row in histogram.packed_indices()]
    pairs = tuple(zip(indices, histogram.weights.tolist(), strict=True))
    if support is None:
        return (
            sum(count for index, count in pairs if index == 0),
            sum(count for index, count in pairs if index == 1),
        )
    counts = {}
    for index, count in pairs:
        index &= support
        counts[index] = counts.get(index, 0) + count
    return tuple(sorted(counts.items()))


class _ReceiptChecks:
    """Admit each preparation receipt once and keep what all receipts must share.

    All admitted receipts must share one producer context, with or without
    mitigation. A fitted readout correction relies on this, because it
    applies the channel measured by the calibration settings to the science
    settings. With mitigation, a science term and the calibrations of its
    pivot must also read the same physical qubit. The attributes record
    those shared values as receipts are admitted.

    Attributes:
        admitted: prepared_id to its checked receipt, in first-use order.
        missing: Whether some observed preparation has no receipt, which
            leaves target, compiler and layout applicability unverified.
        construction_ids: Setting name to its selected construction id,
            looked up once per setting.
        context: (target, (compiler name, version, domain), native basis,
            backend configuration id) of the first admitted receipt. Every
            later receipt must have the same context.
        native_pivots: Logical pivot to the native qubit that the first
            receipt using it maps it to, checked only with mitigation.
    """

    def __init__(self, plan, receipt_map, descriptors, q):
        self.plan, self.receipt_map, self.descriptors, self.q = plan, receipt_map, descriptors, q
        self.admitted, self.construction_ids, self.native_pivots = {}, {}, {}
        self.missing = False
        self.context = None

    def admit(self, chunk, realization, observation):
        """Check the receipt of chunk's preparation the first time that preparation appears.

        The receipt must record the chunk's realization, observation, layouts,
        execution kind and unconditional population, and the setting's
        selected construction. A preparation without a receipt sets missing.
        Raises ValueError for a differing receipt, a different producer
        context, or with mitigation a native mapping of the wrong width or a
        second native qubit for one logical pivot.
        """
        if chunk.prepared_id in self.admitted:
            return
        receipt = self.receipt_map.get(chunk.prepared_id)
        if receipt is None:
            self.missing = True
            return
        if (
            receipt.realization != realization
            or receipt.observation != observation
            or receipt.quantum_layout != chunk.quantum_layout
            or (receipt.classical_layout != chunk.classical_layout)
            or (receipt.execution != "quantum_circuit")
            or (receipt.population != "unconditional")
        ):
            raise ValueError(
                "binary preparation receipt differs from its actual observation and selected layout"
            )
        if chunk.experiment not in self.construction_ids:
            _, construction = realization._selected_construction(self.plan)
            self.construction_ids[chunk.experiment] = construction.content_id
        if receipt.construction_id != self.construction_ids[chunk.experiment]:
            raise ValueError("binary preparation receipt changed its selected construction")
        # Remote compilation receipts can have different job IDs with the
        # same effective compiler configuration. Each receipt keeps its job
        # provenance, and only the effective context is compared here.
        compiler = receipt.compiler
        context = (
            receipt.target,
            (compiler.name, compiler.version, compiler.domain),
            receipt.native_basis,
            receipt.backend_configuration_id,
        )
        if self.context is None:
            self.context = context
        elif self.context != context:
            raise ValueError(
                "science and calibration require matching actual target/compiler/readout context"
            )
        if self.plan.method.mitigation is not None:
            pivot = self.descriptors[chunk.experiment].pivot
            if len(receipt.logical_to_native) != self.q:
                raise ValueError("binary calibration requires the actual native parity pivot mapping")
            native_pivot = receipt.logical_to_native[pivot]
            if self.native_pivots.setdefault(pivot, native_pivot) != native_pivot:
                raise ValueError(
                    "science and calibration require the same actual native parity pivot"
                )
        self.admitted[chunk.prepared_id] = receipt


def _note_source_limits(chunk, source, population_reasons, first_population, first_setting):
    """Record why one observation's counts may not be fresh independent samples.

    source is the chunk's CountsSource. A fixed sampling stream is a seeded
    sampler in one producer context, which returns the same counts for the
    same circuit. The setting's entry in population_reasons is set, which
    makes its interval unavailable, when the source has unknown sampling
    semantics or when this setting already drew a different population
    from the same fixed stream. first_population maps (stream, setting) to
    the first population identity seen, and first_setting maps a stream to
    the first setting that used it.

    Returns:
        The shared-stream reason when a different setting already used this
        fixed stream, so that independence across settings is unverified,
        otherwise None.
    """
    if source.kind == "unknown":
        population_reasons[chunk.experiment] = (
            "actual sampling semantics are unavailable; independent counts are unverified"
        )
    if source.stream is None:
        return None
    reason = (
        "different queried populations reuse one fixed sampling stream; independence is unverified"
    )
    previous = first_population.setdefault((source.stream, chunk.experiment), source.identity)
    if previous != source.identity:
        population_reasons[chunk.experiment] = reason
    if first_setting.setdefault(source.stream, chunk.experiment) != chunk.experiment:
        return reason
    return None


def _count_observation(chunk, source, pair, groups, reused, associations, receipt_map):
    """Add one observed chunk's counts (_parity_counts) to its setting, counting each source once.

    groups[setting] maps a source identity to its first observation id and
    counts. A later observation of a known source, such as a rerun of a
    fixed-seed preparation, joins reused[setting] and adds no sample size.
    Its counts must equal the first ones, since a frozen source reproduces
    its population. associations[setting][source] collects, in first-use
    order, the preparations with a receipt in receipt_map (prepared_id to
    receipt) that produced the source. Raises ValueError for conflicting
    counts from one source.
    """
    name = chunk.experiment
    prior = groups[name].get(source.identity)
    if prior is not None:
        if prior[1] != pair:
            raise ValueError("one frozen prepared counts source returned conflicting populations")
        reused[name].append(chunk.content_id)
    else:
        groups[name][source.identity] = (chunk.content_id, pair)
    associated = associations[name].setdefault(source.identity, {})
    if chunk.prepared_id in receipt_map:
        associated[chunk.prepared_id] = None


def _fixed_time(trace, settings, populations, expected, applicable, population_reasons):
    """Decide whether the data support a fixed-time (Hoeffding) interval.

    The fixed-time bound holds for a sample size fixed before acquisition.
    The data qualify when the Run has one completed event per predeclared
    setting and was neither cancelled nor terminated early, every
    preparation has its receipt (applicable is None), no setting has a
    population reason, and every setting has exactly one observation that
    returned its requested shots. A label population ``<group>:<label>``
    refers to its group setting's acquisition.

    Returns:
        (fixed_time, reason), with reason stating why fixed_time holds or fails.
    """
    fixed = (
        applicable is None
        and (not trace.cancel_requested)
        and (not trace.termination_reason)
        and (len(trace.events) == len(settings))
        and (not population_reasons)
        and all((event.status == "completed" for event in trace.events))
        and all(
            (
                len(p.observation_ids) + len(p.reused_observation_ids) == 1
                and p.zeros + p.ones == expected[p.name.split(":", 1)[0]][1].shots
                for p in populations
            )
        )
    )
    reason = (
        "one completed original acquisition for every predeclared setting"
        if fixed
        else "fixed-time inference requires one complete original acquisition per setting; partial, repeated, reused or stopped data do not qualify"
    )
    return fixed, reason


def _binary_data(plan, observations, trace, receipts):
    """Admit measured parity data and tally one count population per setting.

    After checking the receipts tuple and the Plan's fixed Program, the
    function derives each setting's expected observation
    (_expected_observations). Each observed chunk is then admitted as a
    parity population (_parity_counts), its preparation receipt is admitted
    once (_ReceiptChecks), its sampling source is classified
    (_note_source_limits) and its counts are added to its setting's
    distinct sources (_count_observation). Finally the populations are
    summed (_accumulated_populations), the chronological-prefix rule is
    applied (_prefix_reasons) and fixed-time eligibility is decided
    (_fixed_time).

    Sources are identified by _counts.CountsSources. Same-context fixed-seed
    repetitions are one counted source even across preparations, and fresh
    hardware acquisitions are identified by their provider coordinates.
    Repeated observations of a frozen source therefore never add sample size.

    Returns:
        _BinaryData with the per-setting populations, the used receipt and
        contribution ids, and the facts that decide which inference is legal.

    Raises:
        TypeError: receipts is not a tuple of PreparedArtifact records.
        ValueError: An identity, observation, receipt or count check fails.
    """
    settings = plan.reconstruction.settings
    if type(receipts) is not tuple:
        raise TypeError("measured expectation receipts require an explicit tuple of PreparedArtifact records")
    if any((type(receipt) is not PreparedArtifact for receipt in receipts)):
        raise TypeError("measured expectation receipts require an explicit tuple of PreparedArtifact records")
    events = {event.attempt: event for event in trace.events}
    receipt_map = {receipt.content_id: receipt for receipt in receipts}
    from nwqlib._counts import CountsSources

    sources = CountsSources(receipt_map.get, trace._acquisition)
    if len(events) != len(trace.events) or len(receipt_map) != len(receipts):
        raise ValueError("measured analysis requires distinct actual attempt and preparation identities")
    if any((receipt.plan_id != plan.content_id for receipt in receipts)):
        raise ValueError("measured analysis receipt belongs to another Plan")
    descriptors = {item.name: item for item in settings}
    q = plan.problem.state.manifest.basis.dimension.bit_length() - 1
    # _measured_program: a QWC group measures the width-q register "system"
    # into the width-q classical value "readout", bit i holding qubit i. With
    # mitigation it allocates q one-qubit registers bit_0..bit_(q-1) and
    # measures one pivot bit into the one-bit classical value "readout".
    if plan.method.mitigation is None:
        layouts = ((("system", tuple(range(q))),), (("readout", tuple(range(q))),))
    else:
        layouts = (tuple(((f"bit_{bit}", (bit,)) for bit in range(q))), (("readout", (0,)),))
    supports = {
        item.name: sum(1 << bit for bit in item.qubits) for item in settings if item.role == "group"
    }
    admission = _Admission(plan.construction.program)
    readiness = admission.check()
    readiness.require_ready()
    if plan.construction.program.parameters or plan.construction.program.bindings:
        raise ValueError(
            "measured expectation uses its fixed predeclared setting family, without effective parameter rebinding"
        )
    expected = _expected_observations(plan, admission)
    receipt_checks = _ReceiptChecks(plan, receipt_map, descriptors, q)
    groups = {item.name: {} for item in settings}
    reused = {item.name: [] for item in settings}
    associations = {item.name: {} for item in settings}
    contribution_ids, population_reasons = [], {}
    first_population, first_setting = {}, {}
    independence_reason = None
    for chunk in observations.chunks:
        pair = _parity_counts(
            chunk, plan, trace, events, expected, layouts, supports.get(chunk.experiment)
        )
        receipt_checks.admit(chunk, *expected[chunk.experiment])
        source = sources.observation(chunk)
        shared = _note_source_limits(
            chunk, source, population_reasons, first_population, first_setting
        )
        if shared is not None:
            independence_reason = shared
        contribution_ids.append(chunk.content_id)
        _count_observation(chunk, source, pair, groups, reused, associations, receipt_map)
    populations, histograms = _accumulated_populations(
        settings, groups, reused, associations, width=q
    )
    population_reasons.update(_prefix_reasons(trace, receipt_map, sources, groups, expected))
    applicable = (
        "actual preparation receipts are unavailable; target/compiler/layout applicability is unverified"
        if receipt_checks.missing
        else None
    )
    fixed, fixed_reason = _fixed_time(
        trace, settings, populations, expected, applicable, population_reasons
    )
    return _BinaryData(
        populations=populations,
        receipt_ids=tuple(receipt_checks.admitted),
        contribution_ids=tuple(contribution_ids),
        applicability_reason=applicable,
        population_reasons=population_reasons,
        independence_reason=independence_reason,
        fixed_time=fixed,
        fixed_time_reason=fixed_reason,
        histograms=histograms,
    )


def _classical_value(plan, data):
    """Recover the classical output from its one scaled-observable scalar.

    The kernel returned v^dagger (O*2**(-e)) v for the normalized direction v.
    Exact arithmetic applies 2**e and, for a quadratic form, ||psi||^2 before
    one rounding to binary64.
    """
    from nwqlib.algorithms._eigen_support import matched_chunks
    from nwqlib.execution import ScalarValue

    rec = plan.reconstruction
    chunks = tuple(chunk for chunk, _ in matched_chunks(plan, data))
    if rec.constant is not None:
        if chunks:
            raise ValueError("algebraic expectation cannot contain an acquisition")
        return _output_value(rec.constant, rec), ()
    if not chunks:
        return None, ()
    if len(chunks) != 1:
        raise ValueError("classical expectation requires its one selected matvec acquisition")
    chunk = chunks[0]
    if (
        len(chunk.values) != 1
        or not isinstance(chunk.values[0], ScalarValue)
        or chunk.values[0].label != "scaled_observable"
        or chunk.values[0].frame != "unit"
        or chunk.physical_scale != rec.physical_scale
    ):
        raise ValueError("classical expectation scalar differs from its selected frame and scale")
    value = chunk.values[0].value
    if value is not None:
        arithmetic = ExactArithmetic()
        value = _output_fraction(arithmetic.fraction(value), rec, arithmetic,
                                 exponent=rec.operator_exponent)
    return value, (chunk.content_id,)


def _exact_value(plan, data):
    """Reduce exact grouped Pauli readout to the requested value.

    Every chunk must come from the one selected experiment and carry
    PauliValue entries only for the nonzero nonidentity labels. Repeated
    acquisitions of a label are averaged exactly by _mean, and the value is
    _weighted_sum of those means in the output frame.

    Returns:
        (value, missing): value is None when a label has no data, and missing
        lists those labels.
    """
    from nwqlib.saved_evidence import _validate_data

    _validate_data(plan, data)
    labels = tuple(
        term.label
        for term in plan.reconstruction.terms
        if term.coefficient and set(term.label) != {"I"}
    )
    groups = {label: [] for label in labels}
    for chunk in data.observations.chunks:
        if len(plan.experiments) != 1 or chunk.experiment != plan.experiments[0].name:
            raise ValueError("exact Pauli contribution has a foreign experiment")
        point = plan.resolve(chunk.experiment)
        _, observation = point.resolved_observation(plan)
        if (
            chunk.realization_id != point.content_id
            or chunk.observation != observation
            or chunk.population != "unconditional"
            or chunk.bindings != point.bindings
        ):
            raise ValueError("exact Pauli contribution has a foreign point/readout population")
        for item in chunk.values:
            if not isinstance(item, PauliValue) or item.label not in groups:
                raise ValueError("exact Pauli contribution has an unrequested value or label")
            groups[item.label].append(item.value)
    missing = tuple(label for label, values in groups.items() if not values)
    means = {label: _mean(values) for label, values in groups.items() if values}
    value = (
        None if missing else _weighted_sum(plan.reconstruction.terms, means, plan.reconstruction)
    )
    return value, missing


def _scale_factor(scale, arithmetic):
    """Nominal norm squared as bounded binary arithmetic, never a float norm square."""
    from fractions import Fraction

    exponent = 2 * scale.exponent
    arithmetic._check(abs(exponent) + 1)
    power = Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)
    return arithmetic.multiply(
        arithmetic.multiply(
            arithmetic.fraction(scale.mantissa), arithmetic.fraction(scale.mantissa)
        ),
        power,
    )


def _output_fraction(value, reconstruction, arithmetic, *, side=None, exponent=0):
    """Map an exact normalized-frame value to the requested output, then round once.

    A quadratic form multiplies by ||psi||^2, formed from the PhysicalScale
    mantissa and binary exponent, so no floating norm square can overflow
    first. The keyword exponent multiplies by a further 2**exponent. side
    rounds an interval endpoint outward.
    Returns None when the result is not representable in binary64.
    """
    if value == 0:
        return 0.0
    if reconstruction.output_kind == "quadratic_form":
        scale = reconstruction.physical_scale
        value = arithmetic.multiply(value, arithmetic.fraction(scale.mantissa))
        value = arithmetic.multiply(value, arithmetic.fraction(scale.mantissa))
        exponent += 2 * scale.exponent
    if abs(exponent) + 1 > arithmetic.max_integer_bits:
        return None
    if exponent:
        from fractions import Fraction

        power = Fraction(1 << exponent) if exponent > 0 else Fraction(1, 1 << -exponent)
        value = arithmetic.multiply(value, power)
    return _float(value, arithmetic, side=side)


def _output_value(value, reconstruction):
    """Map a binary64 normalized-frame value to the output frame, keeping None."""
    if value is None:
        return None
    arithmetic = ExactArithmetic()
    return _output_fraction(arithmetic.fraction(value), reconstruction, arithmetic)


def _weighted_sum(terms, values, reconstruction):
    """c0+sum_j c_j*value_j in exact rationals, rounded once in the output frame.

    Identity labels contribute their coefficient. None when a selected term
    value is unavailable.
    """
    arithmetic = ExactArithmetic()
    total = arithmetic.fraction(0)
    for term in terms:
        if term.coefficient == 0:
            continue
        value = 1.0 if set(term.label) == {"I"} else values[term.label]
        if value is None:
            return None
        total = arithmetic.add(
            total,
            arithmetic.multiply(arithmetic.fraction(term.coefficient), arithmetic.fraction(value)),
        )
    return _output_fraction(total, reconstruction, arithmetic)


def _combined_interval(terms, selected, options, family_size, reconstruction, *, extra_assumptions=()):
    """Map the simultaneous binary interval family through the signed physical Pauli sum.

    With delta=failure_probability, each of the F predeclared populations
    (the L labels of grouped readout, or every setting with mitigation) was
    evaluated at delta/F, so all its intervals hold jointly with probability
    at least 1-delta by the union bound, a posterior probability for Beta
    inference. On that event the weighted sum lies in the interval whose
    endpoints take each term's lower or upper endpoint by the sign of its
    coefficient. Terms that share calibration data make this an outer
    enclosure. Labels of one QWC group use the same shots; the union bound
    needs no independence between them. extra_assumptions are appended to
    the per-term assumptions.
    """
    from nwqlib.evidence.binary import _interval

    if options.method == "point":
        return None
    assumptions = tuple(
        dict.fromkeys(
            (
                *(note for item in selected.values() if item is not None for note in item.assumptions),
                *extra_assumptions,
            )
        )
    )
    if any((item is None or item.status == "unavailable" for item in selected.values())):
        reason = next(
            (
                item.reason
                for item in selected.values()
                if item is not None and item.status == "unavailable"
            ),
            "a selected term interval is unavailable",
        )
        return _interval(options, family_size, reason=reason, assumptions=assumptions)
    if any((item.status == "empty" for item in selected.values())):
        return _interval(
            options,
            family_size,
            reason="joint physical parameter set is empty",
            assumptions=assumptions,
            empty=True,
        )
    # Apply each coefficient with the appropriate endpoint order. Exact
    # rational accumulation precedes outward rounding to binary64 endpoints.
    arithmetic = ExactArithmetic()
    low = high = arithmetic.fraction(0)
    for term in terms:
        if term.coefficient == 0:
            continue
        coefficient = arithmetic.fraction(term.coefficient)
        if set(term.label) == {"I"}:
            low, high = (arithmetic.add(low, coefficient), arithmetic.add(high, coefficient))
            continue
        item = selected[term.label]
        a = arithmetic.multiply(coefficient, arithmetic.fraction(item.lower))
        b = arithmetic.multiply(coefficient, arithmetic.fraction(item.upper))
        low, high = (arithmetic.add(low, min(a, b)), arithmetic.add(high, max(a, b)))
    endpoints = (
        _output_fraction(low, reconstruction, arithmetic, side="lower"),
        _output_fraction(high, reconstruction, arithmetic, side="upper"),
    )
    if any((value is None for value in endpoints)):
        return _interval(
            options,
            family_size,
            reason="physical observable interval is unrepresentable in binary64",
            assumptions=assumptions,
        )
    return _interval(
        options,
        family_size,
        bounds=endpoints,
        reason="linear image of the predeclared simultaneous binary family; no cross-Plan winner guarantee",
        assumptions=assumptions,
    )


_JOINT_BETA_UNAVAILABLE = (
    "the joint posterior model of parities that share a QWC group's shots is unspecified; "
    "each label keeps its marginal Beta point and interval"
)
_JOINT_BETA_INTERVAL = (
    "Without a compatible joint posterior, a weighted sum of marginal Beta points remains a "
    "selected estimator, but a joint credibility statement lacks its model"
)


def _shares_group_shots(plan):
    """Whether some QWC group of an unmitigated Plan supplies more than one label."""
    return plan.method.mitigation is None and any(
        len(item.labels) > 1 for item in plan.reconstruction.settings
    )


def _group_scores(outcomes, weights, supports, width):
    """Return the exact integer score ``sum_j (-1)**popcount(s & supports[j]) * weights[j]`` of each outcome s.

    weights are integers. Up to 64 measured bits the parities are formed
    by NumPy population counts over all outcomes at once. The scores
    accumulate in int64 when the sum of the absolute weights stays below
    2**63, which bounds every partial sum, and as Python integers
    otherwise. Work is one pass over the outcomes per label.
    """
    import numpy as np

    if width > 64:
        return [
            sum(-weight if (outcome & support).bit_count() & 1 else weight
                for weight, support in zip(weights, supports, strict=True))
            for outcome in outcomes
        ]
    indices = np.array(outcomes, dtype=np.uint64)
    small = sum(abs(weight) for weight in weights) < 2**63
    scores = np.zeros(len(outcomes), dtype=np.int64 if small else object)
    for weight, support in zip(weights, supports, strict=True):
        odd = (np.bitwise_count(indices & np.uint64(support)) & 1).astype(bool)
        scores[odd] -= weight
        scores[~odd] += weight
    return scores.tolist()


def _group_variance(plan, estimates, histograms, frame):
    """Return one EstimatorContribution per QWC group, or a reason when one is unavailable.

    For group g with output-frame coefficients c_j and decoded parities
    ``X_j(s) = (-1)**popcount(s & support_j)`` of each joint outcome s, the
    weighted score is ``Y_g(s) = sum_(j in g) c_j X_j(s)``. With ``S1 =
    sum_s n_s Y_g(s)`` and ``S2 = sum_s n_s Y_g(s)**2`` over the group's
    n > 1 shots, the empirical variance of its mean is ``(mean(Y**2) -
    mean(Y)**2)/(n-1) = (n*S2 - S1**2)/(n**2*(n-1))``, evaluated as an
    exact rational. Every binary64 weight is an integer over a power of
    two, so with the largest denominator D each ``D*c_j`` is an integer,
    ``D*Y_g(s)`` is an exact integer sum (_group_scores), and the quotient
    is ``(n*S2_D - S1_D**2)/(D**2*n**2*(n-1))`` with ``S1_D = D*S1`` and
    ``S2_D = D**2*S2``; its width is checked against the exact-arithmetic
    limit. The work is one pass over the group's outcomes per label.
    Within-group covariance is essential and per-label
    means alone cannot recover it: for the Bell labels ZI and IZ with
    coefficients one, the weighted score has per-shot variance 4 while the
    individual variances sum to 2. The variable is keyed by the population
    of the group's first label and enters with coefficient one.
    """
    from fractions import Fraction

    arithmetic = ExactArithmetic()
    coefficients = {
        term.label: _output_value(term.coefficient, plan.reconstruction)
        for term in plan.reconstruction.terms
    }
    contributions = []
    for setting in plan.reconstruction.settings:
        weights = tuple(coefficients[label] for label in setting.labels)
        if any(value is None for value in weights):
            return None, "physical coefficient-weighted variance is unrepresentable"
        histogram = histograms[setting.name]
        n = sum(histogram.values())
        fields = dict(
            quantity="qwc_group_score_variance",
            unit=frame.unit,
            scope=frame.scope,
            assumptions=(
                "sample-mean variance formula assumes iid shots within this actual group population",
            ),
        )
        if n <= 1:
            fields.update(availability="unknown", reason="selected population variance is unavailable")
        else:
            ratios = tuple(arithmetic.fraction(value) for value in weights)
            scale = max(ratio.denominator for ratio in ratios)
            outcomes, counts = tuple(histogram), tuple(histogram.values())
            scores = _group_scores(
                outcomes,
                tuple(ratio.numerator * (scale // ratio.denominator) for ratio in ratios),
                _label_supports(setting.labels),
                len(setting.basis),
            )
            first = sum(count * score for count, score in zip(counts, scores, strict=True))
            second = sum(count * score * score for count, score in zip(counts, scores, strict=True))
            value = arithmetic.fraction(Fraction(n * second - first * first, scale * scale * n * n * (n - 1)))
            fields.update(
                availability="concrete",
                value=Rational(numerator=value.numerator, denominator=value.denominator),
                evidence=Evidence(kind="numerical_estimate", source=BINARY_SOURCE),
            )
        data_id = estimates[f"{setting.name}:{setting.labels[0]}"].population.content_id
        contributions.append(
            EstimatorContribution(
                data_id=data_id,
                coefficient=1.0,
                variance=FramedFact(frame=frame, bindings=(), fact=Fact(**fields)),
            )
        )
    return tuple(contributions), None


def _binary_variance(
    plan, estimates, corrections, *, histograms, joint_id, applicability_reason, inference
):
    """Return (kind, assessment, reason) for the variance of the output estimator.

    Grouped readout without mitigation enters one variable per QWC group,
    the mean of its weighted score (_group_variance), so covariance between
    labels measured in the same shots is included. With Beta inference, a
    group with several labels leaves the aggregate posterior variance
    unavailable because the joint posterior model of its parities is
    unspecified. Each label keeps its marginal Beta point and interval.
    With mitigation each science term enters as its three correction
    derivatives in (z, z0, z1) times its output-frame coefficient, keyed
    by data identity, so terms that share a pivot's calibration add their
    derivatives before squaring. This first-order (delta-method) variance
    is Var(sum_j c_j mu_j) with mu_j linearized at the fitted channel.
    Covariance across distinct groups or populations is taken as zero only
    through an IndependenceLaw built from independent_populations, recorded
    as a user assertion.

    Returns:
        kind is _variance_kind. assessment is the linear_variance
        VarianceAssessment in the squared output unit, or None. reason
        explains a None assessment.
    """
    posterior = inference.method == "beta"
    kind = _variance_kind(plan, inference)
    if applicability_reason is not None:
        return (kind, None, applicability_reason)
    if posterior and _shares_group_shots(plan):
        return (kind, None, _JOINT_BETA_UNAVAILABLE)
    unit = plan.output.frame(plan.problem).unit
    if plan.method.mitigation is None and not posterior:
        frame = ErrorFrame(
            quantity=plan.output.kind + "_estimator",
            metric="variance",
            unit=Unit(symbol=f"({unit.symbol})^2", dimension="custom"),
            scope=plan.problem.evidence_scope,
            conditioning="variables are QWC group weighted-score means in the output unit; "
            "empirical sample-mean variance of each group's joint histogram",
        )
        contributions, reason = _group_variance(plan, estimates, histograms, frame)
        if contributions is None:
            return kind, None, reason
        independence = None
        if inference.independent_populations:
            independence = IndependenceLaw(
                joint_id=joint_id,
                frame=frame,
                bindings=(),
                evidence=Evidence(kind="user_assertion", source=BINARY_SOURCE),
            )
        return (
            kind,
            linear_variance(
                joint_id=joint_id, frame=frame, contributions=contributions, independence=independence
            ),
            None,
        )
    frame = ErrorFrame(
        quantity=plan.output.kind + "_estimator",
        metric="variance",
        unit=Unit(symbol=f"({unit.symbol})^2", dimension="custom"),
        scope=plan.problem.evidence_scope,
        conditioning="variables are binary means times one input observable unit; coefficients are unit-scale numeric coordinates; "
        + (
            "product Beta posterior"
            if posterior
            else "first-order fitted calibration Jacobian"
            if corrections
            else "empirical sample-mean variance"
        ),
    )
    statements = {}
    for estimate in estimates.values():
        value = estimate.posterior_variance if posterior else estimate.empirical_variance
        assumptions = (
            "selected iid likelihood/prior are unverified"
            if posterior
            else "sample-mean variance formula assumes iid data within this actual population",
        )
        if corrections:
            assumptions += (
                "first-order delta-method at the fitted channel; calibration transfer and fit bias remain unverified",
            )
        fields = dict(
            quantity="binary_population_variance",
            unit=frame.unit,
            scope=frame.scope,
            assumptions=assumptions,
        )
        if value is None:
            fields.update(
                availability="unknown", reason="selected population variance is unavailable"
            )
        else:
            from nwqlib.core.records import Float64

            fields.update(
                availability="concrete",
                value=Float64(value=value) if posterior else value,
                evidence=Evidence(kind="numerical_estimate", source=BINARY_SOURCE),
            )
        fact = Fact(**fields)
        statements[estimate.population.content_id] = FramedFact(frame=frame, bindings=(), fact=fact)
    # Use data identity to express shared calibration dependence. Multiplying
    # the fitted Jacobian by each Pauli coefficient preserves that covariance.
    coefficients = {
        term.label: _output_value(term.coefficient, plan.reconstruction)
        for term in plan.reconstruction.terms
    }
    if any(value is None for value in coefficients.values()):
        return kind, None, "physical coefficient-weighted variance is unrepresentable"
    contributions = []
    arithmetic = ExactArithmetic()
    for index, (label, name, _) in enumerate(_science_parities(plan.reconstruction.settings)):
        estimate = estimates[name]
        if plan.method.mitigation is None:
            inputs = ((estimate.population.content_id, coefficients[label]),)
        else:
            correction = corrections[index]
            if correction.derivatives is None:
                return (kind, None, "selected fitted point/Jacobian is unavailable")
            weighted = tuple(
                (
                    _float(
                        arithmetic.multiply(
                            arithmetic.fraction(coefficients[label]),
                            arithmetic.fraction(derivative),
                        ),
                        arithmetic,
                    )
                    for derivative in correction.derivatives
                )
            )
            if any((value is None for value in weighted)):
                return (
                    kind,
                    None,
                    "coefficient-weighted calibration Jacobian is unrepresentable in binary64",
                )
            inputs = tuple(
                zip(
                    (correction.science_id, correction.zero_id, correction.one_id),
                    weighted,
                    strict=True,
                )
            )
        for source_id, coefficient in inputs:
            contributions.append(
                EstimatorContribution(
                    data_id=source_id, coefficient=coefficient, variance=statements[source_id]
                )
            )
    independence = None
    if inference.independent_populations:
        independence = IndependenceLaw(
            joint_id=joint_id,
            frame=frame,
            bindings=(),
            evidence=Evidence(kind="user_assertion", source=BINARY_SOURCE),
        )
    result = linear_variance(
        joint_id=joint_id,
        frame=frame,
        contributions=tuple(contributions),
        independence=independence,
    )
    return (kind, result, None)


def _analyze_measured(plan, observations, *, trace, receipts, inference):
    """Infer and optionally calibrate the actual binary populations before combining their
    observables.

    Each stored population is inferred with infer_binary at delta/F for the
    F predeclared populations: the L labels of grouped readout, whose
    marginal parities share their group's shots, or every science and
    calibration setting with mitigation. Each science term is then
    corrected with its pivot's calibration when mitigation is selected. The
    value is c0+sum_j c_j*point_j in the output frame. The variance, the
    linear image of the joint interval family and the sampling fact are
    separate results with their own availability. Returns the validated
    ExpectationAnalysis.
    """
    (
        populations,
        receipt_ids,
        contribution_ids,
        applicability,
        population_reasons,
        independence_reason,
        fixed,
        fixed_reason,
        histograms,
    ) = _binary_data(plan, observations, trace, receipts)
    # Infer the predeclared family with its actual completeness and dependence
    # status, rather than starting a fresh interval after an interrupted run.
    family_size = len(populations)
    options = inference
    estimates = {
        population.name: infer_binary(
            population,
            options=options,
            family_size=family_size,
            fixed_time=fixed,
            fixed_time_reason=fixed_reason,
            applicability_reason=applicability
            or population_reasons.get(population.name.split(":", 1)[0])
            or (independence_reason if options.method == "beta" else None),
        )
        for population in populations
    }
    selected, raw, intervals, corrections = ({}, {}, {}, [])
    # Share zero/one calibration by physical readout pivot, then keep both
    # raw and corrected science estimates for the same acquired population.
    calibration = {
        (item.role, item.pivot): estimates[item.name]
        for item in plan.reconstruction.settings
        if item.role in ("calibration_zero", "calibration_one")
    }
    missing = []
    reasons = []
    for label, name, setting in _science_parities(plan.reconstruction.settings):
        estimate = estimates[name]
        raw[label] = estimate.raw_mean
        if estimate.raw_mean is None:
            missing.append(label)
        if plan.method.mitigation is None:
            selected[label], intervals[label] = (estimate.point, estimate.interval)
            reasons.extend(estimate.unavailable)
        else:
            correction = correct_binary(
                estimate,
                calibration["calibration_zero", setting.pivot],
                calibration["calibration_one", setting.pivot],
                options=plan.method.mitigation,
                inference=options,
                applicability_reason=applicability,
            )
            corrections.append(correction)
            selected[setting.label], intervals[setting.label] = (
                correction.point,
                correction.interval,
            )
            reasons.extend(correction.unavailable)
    value = _weighted_sum(plan.reconstruction.terms, selected, plan.reconstruction)
    raw_value = _weighted_sum(plan.reconstruction.terms, raw, plan.reconstruction)
    unavailable = (
        None
        if value is not None
        else "; ".join(dict.fromkeys(reasons)) or "selected physical estimator is unrepresentable"
    )
    # Compose the weighted point, covariance and simultaneous interval as
    # separate results, each with its own availability and assumptions.
    observation_id = observations.content_id
    variance_kind, variance, variance_unavailable = _binary_variance(
        plan,
        estimates,
        corrections,
        histograms=histograms,
        inference=inference,
        joint_id=observation_id,
        applicability_reason=applicability
        or next(iter(population_reasons.values()), None)
        or independence_reason,
    )
    interval = _combined_interval(
        plan.reconstruction.terms,
        intervals,
        options,
        family_size,
        plan.reconstruction,
        extra_assumptions=(
            (_JOINT_BETA_INTERVAL,) if options.method == "beta" and _shares_group_shots(plan) else ()
        ),
    )
    statistics = ExpectationStatistics(
        inference_options_id=options.content_id,
        populations=tuple(estimates.values()),
        corrections=tuple(corrections),
        receipt_ids=receipt_ids,
        raw_value=raw_value,
        variance_kind=variance_kind,
        variance=variance,
        variance_unavailable=variance_unavailable,
        interval=interval,
        fixed_time=fixed,
        fixed_time_reason=fixed_reason,
        applicability_reason=applicability,
        independence_reason=independence_reason,
    )
    origin = capture_analysis_origin(
        analyzer=_ANALYSIS_SOURCE,
        method_id=plan.method.content_id,
        dependencies=("scipy",) if options.method == "beta" else (),
    )
    result = ExpectationAnalysis(
        origin=origin,
        plan_id=plan.content_id,
        construction_id=plan.construction.content_id,
        observation_id=observation_id,
        contribution_ids=contribution_ids,
        value=value,
        physical_scale=plan.reconstruction.physical_scale,
        missing=tuple(missing),
        statistics=statistics,
        unavailable=unavailable,
        inference=inference,
        facts=_sampling_facts(plan, statistics, value, inference),
    )
    result.validate_plan(plan)
    return result


def _sampling_facts(plan, statistics, value, inference):
    """Export the sampling radius evaluated from the analyzed counts at its own confidence.

    For eligible fixed-time point inference with an accuracy-selected
    allocation, the normalized-frame radius is
    ``sum_j abs(c_j)*sqrt(2*log(2*L/delta)/n_g(j))``, where L counts labels
    and ``n_g(j)`` is the actual returned population of label j's QWC group.
    Multiply this radius by the input's squared physical norm for a
    quadratic form. Shared parities refer to one acquisition exposure. A
    conditional selected interval supplies the larger distance from the
    reported output-frame point to its endpoints. Mitigated and Beta
    analyses have no raw-count sampling fact. The requested tolerance is
    not substituted for the achieved radius.

    Each label's parity mean satisfies Hoeffding's bound at failure
    probability delta/L with its group's n_g(j) shots; a union bound over
    labels and the triangle inequality need no independence within or
    across groups (ExpectationMethod.sampling_shots, Hoeffding (1963),
    doi:10.1080/01621459.1963.10500830, Theorem 2). The fact is a numerical
    estimate, so an assessment lists it as unverified.
    """
    if value is None or plan.method.mitigation is not None or inference.method == "beta":
        return ()
    interval = statistics.interval
    if interval is not None and interval.status == "conditional":
        arithmetic = ExactArithmetic()
        point = arithmetic.fraction(value)
        radius = _float(
            max(
                abs(point - arithmetic.fraction(interval.lower)),
                abs(arithmetic.fraction(interval.upper) - point),
            ),
            arithmetic,
            side="upper",
        )
        failure = inference.failure_probability
        assumptions = interval.assumptions
    elif (
        plan.selection_accuracy is not None
        and inference.method == "point"
        and statistics.fixed_time
        and statistics.applicability_reason is None
        and statistics.independence_reason is None
    ):
        # The selected sampling design also accompanies a plain empirical point.
        # Use actual returned population sizes, never the requested epsilon as
        # an achieved error value. This scalar calculation acquires no data.
        from fractions import Fraction
        from math import isfinite, log, nextafter, sqrt

        exact_failure = 1 - Fraction(plan.selection_accuracy.confidence)
        failure = float(exact_failure)
        if Fraction(failure) < exact_failure:
            failure = nextafter(failure, 1.0)
        if not 0 < failure < 1:
            return ()
        coefficients = {term.label: abs(term.coefficient) for term in plan.reconstruction.terms}
        # One marginal population per label; n_g(j) is its group's size.
        sizes = {
            item.population.name: item.population.zeros + item.population.ones
            for item in statistics.populations
        }
        parities = _science_parities(plan.reconstruction.settings)
        logarithm = log(2 * len(parities)) - log(failure)
        terms = [
            coefficients[label] * sqrt(2 * logarithm / sizes[name]) for label, name, _ in parities
        ]
        try:
            radius = _output_value(nextafter(fsum(terms), float("inf")), plan.reconstruction)
        except OverflowError:
            return ()
        if radius is None or not isfinite(radius):
            return ()
        assumptions = plan.assumptions
    else:
        return ()
    if radius is None:
        return ()
    from nwqlib.core.records import Float64

    frame = plan.output.frame(plan.problem)
    return (
        FramedFact(
            frame=frame,
            bindings=(),
            failure_probability=failure,
            fact=Fact(
                quantity="sampling",
                unit=frame.unit,
                scope=frame.scope,
                availability="concrete",
                value=Float64(value=radius),
                evidence=Evidence(kind="numerical_estimate", source=BINARY_SOURCE),
                assumptions=(
                    *assumptions,
                    "binary64 bound evaluation is not a certified numerical enclosure",
                    "this bounds acquired raw-count sampling only; native, preparation and physical-model bias are separate",
                ),
            ),
        ),
    )


def _provider_mean(estimates):
    """Round the mean once; avoid overflow or premature underflow in scaling."""
    if not estimates:
        return None
    if len(estimates) == 1:
        return estimates[0].value
    return _mean(tuple(item.value for item in estimates))


def _mean(values):
    """Arithmetic mean of binary64 values in exact rationals, rounded once."""
    arithmetic = ExactArithmetic()
    total = arithmetic.fraction(0)
    for value in values:
        total = arithmetic.add(total, arithmetic.fraction(value))
    return float(arithmetic.divide(total, arithmetic.fraction(len(values))))


def _analyze_estimated(plan, observations, *, trace, receipts):
    """Preserve provider uncertainty for the weighted sum; do no count inference.

    Repeated acquired estimates have an empirical arithmetic-mean point only.
    Their original standard errors remain separate; no covariance or combined
    standard error is manufactured for that mean.
    """
    events = {event.attempt: event for event in trace.events}
    receipt_map = {receipt.content_id: receipt for receipt in receipts}
    if len(receipt_map) != len(receipts):
        raise ValueError("provider analysis requires distinct actual preparation receipts")
    (experiment,) = plan.experiments
    _, construction = plan._selected_construction(experiment.name, ())
    realization = Realization(
        plan_id=plan.content_id, experiment=experiment.name, bindings=construction.program.bindings
    )
    admission = _Admission(construction.program)
    readiness = admission.check()
    readiness.require_ready()
    estimates, contribution_ids = ([], [])
    for chunk in observations.chunks:
        trace.validate_observation(chunk)
        event = events.get(chunk.attempt)
        if (
            chunk.plan_id != plan.content_id
            or chunk.run_id != trace.run_id
            or chunk.realization_id != realization.content_id
            or (chunk.experiment != experiment.name)
            or (chunk.setting != experiment.setting)
            or (chunk.bindings != realization.bindings)
            or (chunk.observation != experiment.observation)
            or (chunk.execution != "quantum_circuit")
            or (event is None)
            or (event.status != "completed")
            or (event.prepared_id != chunk.prepared_id)
            or (event.observation_id != chunk.content_id)
            or event.shots
            or (event.evaluations != 1)
            or (not event.provider_managed_sampling)
        ):
            raise ValueError(
                "provider estimate differs from its original acquisition/observable association"
            )
        receipt = receipt_map.get(chunk.prepared_id)
        if receipt is not None and (
            receipt.realization != realization
            or receipt.construction_id != construction.content_id
            or receipt.observation != chunk.observation
            or (receipt.quantum_layout != chunk.quantum_layout)
            or (receipt.classical_layout != chunk.classical_layout)
        ):
            raise ValueError("provider estimate receipt differs from its original point/readout")
        estimates.append(chunk.values[0])
        contribution_ids.append(chunk.content_id)
    value = _output_value(_provider_mean(estimates), plan.reconstruction)
    missing = (
        ()
        if estimates
        else tuple(
            (
                term.label
                for term in plan.reconstruction.terms
                if term.coefficient != 0 and set(term.label) != {"I"}
            )
        )
    )
    result = ExpectationAnalysis(
        origin=capture_analysis_origin(analyzer=_ANALYSIS_SOURCE, method_id=plan.method.content_id),
        plan_id=plan.content_id,
        construction_id=construction.content_id,
        observation_id=observations.content_id,
        contribution_ids=tuple(contribution_ids),
        value=value,
        physical_scale=plan.reconstruction.physical_scale,
        missing=missing,
        estimates=tuple(estimates),
        unavailable="physical provider estimate is unrepresentable"
        if estimates and value is None
        else None,
    )
    result.validate_plan(plan)
    return result


def _measured_program(problem, labels, method, shots):
    """Prepare the state, rotate to each measured basis and measure its readout qubits.

    Without mitigation, the nonzero nonidentity labels are partitioned into
    qubit-wise commuting groups by deterministic first fit in term order
    (PauliTerms.group with strategy "qwc"), with max_classical_products as
    the running comparison budget. The Program declares one width-q
    register ``system`` and one width-q classical value ``readout``, and
    each group gets one science experiment with the five-child body
    ``Allocate(system), PREP(system), BASIS_g(system), Measure(system,
    readout), Release(system)``. PREP is the whole-register preparation
    (select_preparation with per_bit=False). BASIS_g is the group's own
    selected block (select_pauli_group_basis on the accumulated basis,
    signature ``basis_<g>``), which applies H on the X sites and S† then H
    on the Y sites with no parity network. Classical bit i of readout holds
    qubit i, and one group count table supplies every label's parity
    ``(-1)**popcount(outcome & support_j)``. Measuring the sites outside
    the group's support adds q - s_g physical measurements per shot but no
    basis rotation or CX gate, and summing over their outcomes gives the
    support marginal. shots keeps its meaning of shots per setting, now per
    group, so the science workload is G*shots.

    With mitigation, each nonidentity term gets its own science experiment,
    which applies the state PREP, the label's basis rotations and CX parity
    network, and then measures one bit on the pivot. That bit is the +/-1
    population consumed by binary inference and by pivot calibration. One
    all-zero preparation is shared, each distinct pivot gets its own
    one-preparation, and each pivot acquires separate zero and one
    calibration populations. A group histogram with local measurements has
    a different readout channel, which these pivot calibrations do not
    describe.
    """
    from nwqlib.blocks.selection import select_pauli_group_basis, select_pauli_parity
    from nwqlib.core.planning import ReadoutDetails

    state = problem.state
    q = (state.manifest.basis.dimension - 1).bit_length()
    grouped = method.mitigation is None
    settings, experiments, batches = [], [], []

    def body(setting, children, count):
        """Add one single-setting counts batch whose body Sequence runs children.

        children allocates the registers, runs the preparation and any basis
        or parity blocks, measures and releases the registers. The batch
        repeats count times. setting and its Experiment are recorded.
        """
        name = setting.name
        definitions.append(Definition(id="body_" + name, node=Sequence(children=children)))
        batch = "batch_" + name
        metadata = MetadataRef(
            format=_METHOD,
            data=InputRef(identity=problem.content_id, representation="scalar", source=_METHOD),
        )
        definitions.append(
            Definition(
                id=batch,
                node=MeasurementBatch(
                    body="body_" + name,
                    settings=(Setting(label=name, metadata=metadata),),
                    repetitions=count,
                    observation_kind="counts",
                ),
            )
        )
        batches.append(batch)
        settings.append(setting)
        experiments.append(Experiment(name=name, batch=batch, setting_index=0, readout=ReadoutDetails()))

    if grouped:
        from nwqlib.operators.inputs import pauli_table

        registers = (Register(name="system", width=q),)
        classical = (ClassicalValue(name="readout", dtype="bits", width=q),)
        port = (PortMap(port="system", wire="system"),)
        blocks = [select_preparation("state", state, choice=method.preparation_choice, per_bit=False)]
        definitions = [
            Definition(id="allocate", node=Allocate(wire="system")),
            Definition(id="release", node=Release(wire="system")),
            Definition(id="measure", node=Measure(wire="system", result="readout")),
            Definition(id="prepare_science", node=BlockCall(signature="state", ports=port)),
        ]
        table = pauli_table(((label, 1.0) for label in labels), num_qubits=q, max_bytes=method.max_bytes)
        grouping = table.group(
            strategy="qwc", max_bytes=method.max_bytes, max_comparisons=method.max_classical_products,
            limit_name="ExpectationMethod.max_classical_products",
        )
        for index, members in enumerate(grouping.groups):
            members = tuple(labels[member] for member in members)
            # A QWC group has at most one nonidentity axis on each qubit.
            axes = ["I"] * q
            for label in members:
                for bit, axis in enumerate(reversed(label)):
                    if axis != "I":
                        axes[bit] = axis
            basis = "".join(reversed(axes))
            # One selected basis block per group; first fit gives each group a
            # distinct accumulated basis.
            signature = f"basis_{index}"
            blocks.append(select_pauli_group_basis(signature, basis, basis=state.manifest.basis))
            definitions.append(Definition(id=signature, node=BlockCall(signature=signature, ports=port)))
            setting = ExpectationSetting(
                name=f"group_{index}",
                role="group",
                labels=members,
                basis=basis,
                qubits=tuple(bit for bit in range(q) if axes[bit] != "I"),
            )
            body(setting, ("allocate", "prepare_science", signature, "measure", "release"), shots)
    else:
        registers = tuple(Register(name=f"bit_{bit}", width=1) for bit in range(q))
        classical = (ClassicalValue(name="readout", dtype="bits", width=1),)
        allocate = tuple(f"allocate_{bit}" for bit in range(q))
        release = tuple(f"release_{bit}" for bit in range(q))
        definitions = [Definition(id=name, node=Allocate(wire=f"bit_{bit}")) for bit, name in enumerate(allocate)]
        definitions += [Definition(id=name, node=Release(wire=f"bit_{bit}")) for bit, name in enumerate(release)]
        ports = tuple(PortMap(port=f"system_{bit}", wire=f"bit_{bit}") for bit in range(q))
        blocks = [select_preparation("state", state, choice=method.preparation_choice, per_bit=True)]
        definitions.append(Definition(id="prepare_science", node=BlockCall(signature="state", ports=ports)))

        def pivot_body(setting, calls, count):
            """Add a batch that runs calls and measures the setting's pivot into readout."""
            measure = "measure_" + setting.name
            definitions.append(Definition(id=measure, node=Measure(wire=f"bit_{setting.pivot}", result="readout")))
            body(setting, (*allocate, *calls, measure, *release), count)

        pivots = tuple(
            sorted({max(bit for bit, axis in enumerate(reversed(label)) if axis != "I") for label in labels})
        )
        for index, label in enumerate(labels):
            name = f"parity_{index}"
            parity = select_pauli_parity(name, label, basis=state.manifest.basis)
            blocks.append(parity)
            definitions.append(Definition(id=name, node=BlockCall(signature=name, ports=ports)))
            pivot = max(bit for bit, axis in enumerate(reversed(label)) if axis != "I")
            setting = ExpectationSetting(name=f"science_{index}", role="science", pivot=pivot, label=label)
            pivot_body(setting, ("prepare_science", name), shots)
        # Use a common zero preparation and a separate one preparation per pivot.
        # Each pivot still acquires its own zero and one calibration populations.
        for pivot in (None, *pivots):
            state = ingest_occupation(tuple(int(bit == pivot) for bit in range(q)), num_qubits=q)
            name = "calibration_zero" if pivot is None else f"calibration_one_{pivot}"
            blocks.append(select_preparation(name, state, per_bit=True))
            definitions.append(Definition(id="prepare_" + name, node=BlockCall(signature=name, ports=ports)))
        for pivot in pivots:
            calibration_shots = method.mitigation.calibration_shots
            pivot_body(
                ExpectationSetting(name=f"zero_{pivot}", role="calibration_zero", pivot=pivot),
                ("prepare_calibration_zero",),
                calibration_shots,
            )
            pivot_body(
                ExpectationSetting(name=f"one_{pivot}", role="calibration_one", pivot=pivot),
                (f"prepare_calibration_one_{pivot}",),
                calibration_shots,
            )
    definitions.append(Definition(id="root", node=Sequence(children=tuple(batches))))
    return (
        admitted_program(
            "ExpectationMethod.max_admission_steps",
            method.max_admission_steps,
            root="root",
            definitions=tuple(definitions),
            registers=registers,
            classical=classical,
            signatures=tuple(block.record.signature for block in blocks),
        ),
        tuple(blocks),
        tuple(experiments),
        tuple(settings),
    )


def _validate_inputs(problem, output, *, quantum):
    """Admit an Expectation problem and its output before any planning work.

    The output must be a NormalizedExpectation or QuadraticForm of the
    Problem's own observable, and the state scale must be known. A normalized
    expectation of the zero state is undefined. quantum additionally requires
    a power-of-two dimension of at least 2 and, for a nonzero state, a
    preparation without a blocker. Raises ApplicabilityError for an
    unsupported combination and ValueError for an invalid value.
    """
    if not isinstance(problem, Expectation) or not isinstance(
        output, (NormalizedExpectation, QuadraticForm)
    ):
        raise ApplicabilityError(
            "ExpectationMethod requires Expectation and a normalized or quadratic output"
        )
    if output.observable.manifest != problem.observable.manifest:
        raise ValueError("output observable must be the original Expectation observable")
    scale = problem.state.preparation.physical_scale
    if scale is None:
        raise ApplicabilityError("expectation requires a known physical state scale")
    if scale.mantissa == 0 and isinstance(output, NormalizedExpectation):
        raise ValueError("normalized expectation is undefined for a zero state")
    if quantum and (problem.dimension < 2 or problem.dimension.bit_count() != 1):
        raise ApplicabilityError(
            "quantum Expectation requires a power-of-two dimension; use classical for other dimensions"
        )
    if quantum and scale.mantissa != 0 and problem.state.preparation.blocker:
        raise ApplicabilityError(problem.state.preparation.blocker)


def _terms(problem, output):
    """Return the observable as PauliCoefficient terms for the quantum paths.

    Pauli access is used as stored. An explicit dense observable of dimension
    at most AUTO_DENSE_PAULI_DIMENSION (16, the limit the eigenvalue methods
    share in algorithms/_eigen_inputs.py) is expanded by pauli_coefficients
    into c_P=tr(P O)/D (ENGINEERING_CONSTANTS "Eigen input conversion and
    classical preparation"). Other access raises ApplicabilityError. A
    nonreal coefficient raises ValueError, since a Hermitian observable has
    real Pauli coefficients. A zero state returns no terms. Its quadratic
    form is zero, and _validate_inputs has already rejected its normalized
    expectation.
    """
    from nwqlib.algorithms._eigen_inputs import AUTO_DENSE_PAULI_DIMENSION

    _validate_inputs(problem, output, quantum=True)
    if problem.state.preparation.physical_scale.mantissa == 0:
        return ()
    if "pauli_terms" in problem.observable.manifest.access:
        terms = problem.observable.pauli_terms().labels()
    elif (
        problem.observable.manifest.reference.representation == "dense"
        and problem.dimension <= AUTO_DENSE_PAULI_DIMENSION
    ):
        # This is the admitted original matrix, already exactly Hermitian at
        # the Problem owner. At D=16, q=4, the block transform visits
        # q*D²=1024 entries. No eigensolve, symmetrization or pruning.
        from nwqlib.operators._pauli import pauli_coefficients

        terms = pauli_coefficients(problem.observable.dense_array())
    else:
        raise ApplicabilityError(
            "finite Pauli access or an explicit dense observable of dimension "
            f"<={AUTO_DENSE_PAULI_DIMENSION} is required"
        )
    if any(value.imag != 0 for _, value in terms):
        raise ValueError("Hermitian expectation requires real Pauli coefficients")
    return tuple(
        PauliCoefficient(label=label, coefficient=float(value.real)) for label, value in terms
    )


class ExpectationMethod(Method):
    """Select exact readout, counts inference or an explicit weighted provider estimate.

    Exact Expectation readout acquires all selected nonidentity labels in
    one experiment. Sampled grouping uses joint measurement populations
    whose uncertainty calculation includes within-group covariance.
    Positive shots without mitigation select one measured experiment per
    qubit-wise commuting group, with shots per group and the configured
    binary inference on each label's marginal parity. With mitigation,
    positive shots select one measured parity experiment per nonidentity
    term and its pivot calibrations. estimate_precision selects one
    provider-managed estimate of the whole weighted sum. Classical
    execution uses the original matvec access. Identity terms are algebraic
    constants on the exact and measured paths, and an identity-only
    observable needs no acquisition on any path. The Expectation guide
    derives the statistical quantities.

    Attributes:
        preparation_choice: ``native`` state PREP or equivalent ``hzh`` wiring for admitted occupation preparations only.
        inference: Binary count-inference configuration, including its conditional interval model.
        mitigation: Optional zero/one readout calibration per measured pivot; selects one parity experiment per term instead of QWC groups and adds actual calibration acquisitions.
        estimate_precision: Positive managed Estimator precision request; None keeps exact/counts selection. It does not specify a raw shot population.
        max_bytes: Bound on known operator/state and classical workspace bytes.
        max_classical_products: Cap on counted classical observable applications, and the comparison budget of the QWC grouping of sampled readout.
        max_admission_steps: Admission ceiling of the Plan's Programs, their ``AdmissionLimits.max_steps``: the kept field slots and the structural and lifecycle work units of one admission check. The resource fold of a Program may use up to 24 times this value. The default, 1,000,000, is the QLS default, ten times the shared ``AdmissionLimits`` default. Raising it admits a larger metadata check and fold. It changes no selected quantum work.
    """

    schema_version: Literal[2] = 2
    result_type: ClassVar[type[Result]] = ExpectationAnalysis
    preparation_choice: Literal["native", "hzh"] = "native"
    inference: BinaryInferenceOptions = BinaryInferenceOptions()
    mitigation: BinaryReadoutMitigation | None = None
    estimate_precision: Annotated[Real, Field(gt=0)] | None = None
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_classical_products: PositiveInt = 100_000_000
    max_admission_steps: PositiveInt = 1_000_000

    descriptor: ClassVar[AlgorithmDescriptor] = _DESCRIPTOR

    @model_validator(mode="after")
    def _configuration(self):
        if self.estimate_precision is not None and (
            self.inference != BinaryInferenceOptions() or self.mitigation is not None
        ):
            raise ValueError("provider estimation excludes direct binary inference and calibration")
        if self.mitigation is not None and self.inference.method == "beta":
            raise ValueError("binary readout mitigation has no selected Beta posterior model")
        return self

    def sampling_shots(self, problem, *, output, accuracy, execution, shots):
        """Return the shots per QWC group selected by an absolute sampling tolerance.

        For L nonzero nonidentity labels and ``C=sum_j abs(c_j)``, use
        ``ceil(2*(C/epsilon)**2*log(2*L/delta))``, where
        ``delta=1-confidence``. Each label uses the shots of its group, and a
        union bound over labels permits their parities to be correlated
        within a shot. A quadratic form uses C in its physical output frame.
        An identity-only observable needs no acquisition. This selection
        supports raw quantum counts without mitigation, Beta inference or
        provider estimation and conflicts with an explicit shot count.

        For fixed counts, Hoeffding's inequality on each parity gives
        ``Pr(abs(mean_j - mu_j) > r_j) <= delta/L`` with ``r_j =
        sqrt(2*log(2*L/delta)/n_g(j))``; a union over labels and the triangle
        inequality bound the weighted error by ``sum_j abs(c_j)*r_j`` with
        probability at least 1-delta, without within-group or cross-group
        independence. Equal n per group and ``C*sqrt(2*log(2*L/delta)/n) <=
        epsilon`` give the count above, and the science workload is G*n
        shots for G groups. Returns None when no label needs measurement.
        """
        if (
            not isinstance(accuracy, Accuracy)
            or accuracy.component != "sampling"
            or accuracy.absolute_tolerance is None
            or accuracy.relative_tolerance is not None
            or execution != "quantum"
            or self.estimate_precision is not None
            or self.mitigation is not None
            or self.inference.method == "beta"
        ):
            raise ApplicabilityError(
                "sampling accuracy supports only absolute finite-Pauli raw counts without mitigation"
            )
        if shots is not None:
            raise ValueError(
                "explicit shots conflict with an accuracy-selected sampling allocation"
            )
        terms = tuple(
            term
            for term in _terms(problem, output)
            if term.coefficient and set(term.label) != {"I"}
        )
        if not terms:
            return None
        # Hoeffding (1963), doi:10.1080/01621459.1963.10500830, Theorem 2:
        # n outcomes in [-1,1] with a fixed mean satisfy
        # P(|mean-mu|>r) <= 2 exp(-n*r^2/2). With C=sum|cj| over the L labels,
        # a union bound at delta/L and the triangle inequality set r=eps/C,
        # so the labels of one QWC group may share its shots. This conditions
        # on independent stationary successive shots of each group; it bounds
        # sampling alone, not native/PREP/physical bias.
        # Preserve exact binary64 inputs before the transcendental evaluation.
        # Decimal.ln is correctly rounded; next_plus and upward arithmetic keep
        # the admitted integer above the analytical bound near an integer edge.
        from decimal import Decimal, ROUND_CEILING, localcontext
        from fractions import Fraction

        coefficient = sum((Fraction(abs(term.coefficient)) for term in terms), Fraction())
        if isinstance(output, QuadraticForm):
            coefficient *= _scale_factor(
                problem.state.preparation.physical_scale, ExactArithmetic()
            )
        factor = 2 * (coefficient / Fraction(accuracy.absolute_tolerance)) ** 2
        logarithm_argument = Fraction(2 * len(terms)) / (1 - Fraction(accuracy.confidence))
        with localcontext() as context:
            # 80 significant digits keep the upward-rounded value within a
            # relative 1e-78 of the exact one (ENGINEERING_CONSTANTS
            # "Expectation classical work and physical output").
            context.prec = 80
            context.rounding = ROUND_CEILING
            factor_upper = Decimal(factor.numerator) / Decimal(factor.denominator)
            argument_upper = Decimal(logarithm_argument.numerator) / Decimal(
                logarithm_argument.denominator
            )
            required = factor_upper * argument_upper.ln().next_plus()
            if required > MAX_BINARY_COUNT:
                raise ValueError(
                    "selected sampling count exceeds the supported exact integer population"
                )
            return max(1, int(required.to_integral_value(rounding=ROUND_CEILING)))

    def before_submit(self, plan, prepared, *, run):
        """An accuracy-selected sample cannot reuse a known fixed population."""
        if plan.selection_accuracy is None:
            return
        run.admit_count_preparations(prepared, require_all=True)

    def plan(self, problem, *, output, execution, shots, rng, accuracy=None):
        """Select direct Pauli, measured group or parity, or provider-managed acquisition for the
        normalized observable.
        """
        if execution == "classical":
            if (
                shots is not None
                or self.mitigation is not None
                or self.estimate_precision is not None
                or self.inference != BinaryInferenceOptions()
                or accuracy is not None
            ):
                raise ValueError(
                    "classical expectation excludes shots, binary inference, mitigation and provider precision"
                )
            return self._plan_classical(problem, output, rng)
        if self.estimate_precision is not None and shots is not None:
            raise ValueError("provider estimates cannot honor an explicit raw shot count")
        if shots is None and (
            self.inference != BinaryInferenceOptions() or self.mitigation is not None
        ):
            raise ValueError("binary inference and calibration require selected positive shots")
        terms = _terms(problem, output)
        if accuracy is not None:
            selected = self.sampling_shots(
                problem, output=output, accuracy=accuracy, execution=execution, shots=None
            )
            if selected != shots:
                raise ValueError("actual shots differ from the accuracy-selected allocation")
        # Identity contributions are known algebraically. Only nonzero
        # nonidentity terms require state preparation and actual acquisition.
        labels = tuple(
            term.label for term in terms if term.coefficient and set(term.label) != {"I"}
        )
        settings = ()
        if labels and shots is not None:
            program, blocks, experiments, settings = _measured_program(problem, labels, self, shots)
        elif labels:
            q = (problem.dimension - 1).bit_length()
            block = select_preparation("state", problem.state, choice=self.preparation_choice)
            blocks = (block,)
            program = admitted_program(
                "ExpectationMethod.max_admission_steps",
                self.max_admission_steps,
                root="root",
                registers=(Register(name="system", width=q),),
                signatures=(block.record.signature,),
                definitions=(
                    Definition(id="allocate", node=Allocate(wire="system")),
                    Definition(
                        id="prepare",
                        node=BlockCall(
                            signature="state", ports=(PortMap(port="system", wire="system"),)
                        ),
                    ),
                    Definition(id="root", node=Sequence(children=("allocate", "prepare"))),
                ),
            )
            observation = (
                ObservationSpec(kind="pauli_expectation", labels=labels)
                if self.estimate_precision is None
                else ObservationSpec(
                    kind="estimated_observable",
                    estimate=ObservableEstimateSpec(
                        observable_id=problem.observable.manifest.content_id,
                        labels=tuple(term.label for term in terms),
                        coefficients=tuple(term.coefficient for term in terms),
                        precision=self.estimate_precision,
                    ),
                )
            )
            experiments = (
                Experiment(
                    name="expectation", setting="physical_pauli_sum", observation=observation
                ),
            )
        else:
            blocks, experiments = (), ()
            program = admitted_program(
                "ExpectationMethod.max_admission_steps",
                self.max_admission_steps,
                root="root",
                definitions=(Definition(id="root", node=Sequence()),),
            )
        construction = SelectedConstruction(
            program=program, selections=tuple(block.record for block in blocks)
        )
        model = _error_model(
            problem, output, construction, mitigated=bool(settings and self.mitigation is not None)
        )
        return Plan(
            problem=problem,
            method=self,
            output=output,
            execution=execution,
            shots=shots,
            randomness=rng.snapshot(),
            selection_accuracy=accuracy,
            construction=construction,
            reconstruction=ExpectationReconstruction(
                terms=terms,
                physical_scale=problem.state.preparation.physical_scale,
                settings=settings,
                output_kind=output.kind,
            ),
            experiments=experiments,
            error_model=model,
            assumptions=(
                ()
                if accuracy is None
                else (
                    "sampling design assumes independent stationary raw counts within each setting",
                    "the selected sampling criterion excludes native, preparation, physical-model and calibration bias",
                    "resolved PCG64 seeds do not establish statistical independence",
                )
            ),
        )._bind(blocks=blocks)

    def _plan_classical(self, problem, output, rng):
        """Select one original-dimensional matvec and scalar reduction; no PREP work."""
        from nwqlib.algorithms._eigen_inputs import preparation_requirements
        from nwqlib.algorithms._eigen_support import host_construction
        from nwqlib.operators.inputs import _observable_exponent

        _validate_inputs(problem, output, quantum=False)
        state, operator = problem.state, problem.observable
        zero = state.preparation.physical_scale.mantissa == 0
        constant = 0.0 if zero else _classical_constant(operator, max_bytes=self.max_bytes)
        _, preparation_work = preparation_requirements(state)
        exponent = 0
        if constant is None:
            _, work = _classical_action_requirements(operator, state, max_bytes=self.max_bytes)
            if work > self.max_classical_products:
                raise ValueError("classical expectation exceeds max_classical_products before preparation")
            exponent = _observable_exponent(operator)
            construction, experiments = host_construction(
                _METHOD,
                (state.reference, operator.reference),
                ("scaled_observable",),
                work=work,
                admission=("ExpectationMethod.max_admission_steps", self.max_admission_steps),
                description="original-dimensional observable scaled by recorded binary exponent; "
                + (
                    "custom preparation work unknown"
                    if preparation_work is None
                    else "known preparation products included"
                ),
            )
        else:
            construction = SelectedConstruction(
                program=admitted_program(
                    "ExpectationMethod.max_admission_steps",
                    self.max_admission_steps,
                    root="root",
                    definitions=(Definition(id="root", node=Sequence()),),
                ),
                selections=(),
            )
            experiments = ()
        reconstruction = ExpectationReconstruction(
            terms=(),
            physical_scale=state.preparation.physical_scale,
            output_kind=output.kind,
            classical=True,
            constant=constant,
            operator_exponent=exponent,
        )
        plan = Plan(
            problem=problem,
            method=self,
            output=output,
            execution="classical",
            shots=None,
            randomness=rng.snapshot(),
            construction=construction,
            experiments=experiments,
            reconstruction=reconstruction,
            error_model=_error_model(problem, output, construction),
            assumptions=(
                "classical normalized-direction matvec; floating-point reduction is not a certified bound",
            ),
        )
        return plan._bind(blocks=self._host_blocks(plan) if experiments else (),
                          operator_exponent=exponent)

    def _host_blocks(self, plan):
        """Bind the classical kernel that evaluates v^dagger (O*2**(-e)) v.

        v is the normalized state direction and e the Plan's
        operator_exponent. The kernel checks the byte and product caps again
        before it materializes v. It returns one unit-frame scalar
        ``scaled_observable``, or no value with a reason when scaling lost a
        nonzero component or the reduction is not finite. _classical_value
        restores 2**e and, for a quadratic form, ||psi||^2 in exact arithmetic.
        """
        from nwqlib.algorithms._eigen_inputs import state_direction
        from nwqlib.blocks.kernels import BoundKernel, KernelOutput
        from nwqlib.execution import ScalarValue

        (kernel,) = plan.construction.kernels

        def invoke():
            """Evaluate the scaled quadratic form once and return it as a KernelOutput."""
            import numpy as np
            from nwqlib.operators.inputs import _scaled_observable

            state, operator = plan.problem.state, plan.problem.observable
            _, work = _classical_action_requirements(operator, state, max_bytes=self.max_bytes)
            if work > self.max_classical_products:
                raise ValueError("classical expectation exceeds max_classical_products before preparation")
            # O*2**(-e) keeps the matvec finite for extreme coefficients.
            # _classical_value restores 2**e in exact arithmetic.
            scaled = _scaled_observable(operator, plan.reconstruction.operator_exponent)
            value = float("nan")
            if scaled is not None:
                vector = state_direction(state, max_bytes=self.max_bytes)
                if "pauli_terms" in operator.manifest.access:
                    from nwqlib.operators._pauli import apply_terms

                    action = apply_terms(scaled, vector, num_qubits=scaled.num_qubits)
                else:
                    action = scaled @ vector
                value = float(np.vdot(vector, action).real)
            finite = bool(np.isfinite(value))
            return KernelOutput(
                plan_id=plan.content_id,
                selected_kernel_id=kernel.content_id,
                scalars=(
                    ScalarValue(
                        label="scaled_observable",
                        value=value if finite else None,
                        frame="unit",
                        unavailable=None
                        if finite
                        else "scaled observable lost a nonzero component or reduction is unrepresentable",
                    ),
                ),
                physical_scale=state.preparation.physical_scale,
            )

        return (BoundKernel._bind(plan, kernel, invoke),)

    def _analyze_classical(self, plan, data):
        """Return the ExpectationAnalysis of a classical Plan from its one scalar or its constant."""
        value, ids = _classical_value(plan, data)
        result = ExpectationAnalysis(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=ids,
            origin=capture_analysis_origin(analyzer=_ANALYSIS_SOURCE, method_id=self.content_id),
            value=value,
            physical_scale=plan.reconstruction.physical_scale,
            missing=(),
            unavailable=None
            if value is not None
            else "classical physical observable is unrepresentable",
            facts=exact_readout_sampling(plan, data.observations, random_draws=False),
        )
        result.validate_plan(plan)
        return result

    def analyze(self, plan, data, *, settings):
        """Dispatch reconstruction from the acquired readout kind while preserving its
        statistical assumptions.
        """
        if settings.keys() - {"inference"}:
            raise ValueError("unsupported Expectation analysis setting")
        inference = settings.get("inference", self.inference)
        if not isinstance(inference, BinaryInferenceOptions):
            raise TypeError("inference must be BinaryInferenceOptions")
        if self.mitigation is not None and inference.method == "beta":
            raise ValueError("binary mitigation has no Beta posterior model")
        if settings and not plan.reconstruction.settings:
            raise ValueError("binary inference requires existing measured counts")
        if (
            type(plan) is not Plan
            or plan.method is not self
            or data.trace.plan_id != plan.content_id
        ):
            raise ValueError("analysis requires the selected Expectation Plan and actual data")
        if plan.reconstruction.classical:
            return self._analyze_classical(plan, data)
        if plan.reconstruction.settings:
            return _analyze_measured(
                plan,
                data.observations,
                trace=data.trace,
                receipts=data.receipts,
                inference=inference,
            )
        if self.estimate_precision is not None and plan.experiments:
            return _analyze_estimated(
                plan, data.observations, trace=data.trace, receipts=data.receipts
            )
        value, missing = _exact_value(plan, data)
        result = ExpectationAnalysis(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=tuple(chunk.content_id for chunk in data.observations.chunks),
            origin=capture_analysis_origin(analyzer=_ANALYSIS_SOURCE, method_id=self.content_id),
            value=value,
            physical_scale=plan.reconstruction.physical_scale,
            missing=missing,
            unavailable="physical weighted sum is unrepresentable"
            if value is None and not missing
            else None,
            facts=exact_readout_sampling(plan, data.observations, random_draws=False),
        )
        result.validate_plan(plan)
        return result

    def save_archive(self, plan, files):
        """Return the archive index that load_archive rebinds without replanning.

        A classical Plan needs only its Plan, Problem, output and Method
        records. A quantum Plan also stores the state payload of each
        preparation block and names each block's built-in constructor. A
        block built any other way raises ValueError.
        """
        from nwqlib.blocks.selection import _preparation_circuit, _primitive_circuit

        if plan.execution == "classical":
            return dict(
                format="expectation/5",
                plan=files.write_plan(plan),
                problem=files.write_problem(plan.problem),
                output=files.write_output(plan.output),
                method=self.model_dump(mode="json", exclude_computed_fields=True),
            )
        inputs, payloads, constructors = {}, [], []
        for index, block in enumerate(plan.blocks):
            if block._base is not None or block._constructor not in (
                _preparation_circuit,
                _primitive_circuit,
            ):
                raise ValueError(
                    "Expectation archive requires its actual built-in preparation/parity bindings"
                )
            name = None
            if block._payload is not None:
                name = f"preparation-{index}"
                inputs[name] = files.write_state(name, block._payload)
            payloads.append(name)
            constructors.append(
                "preparation" if block._constructor is _preparation_circuit else "primitives"
            )
        return dict(
            format="expectation/5",
            plan=files.write_plan(plan),
            problem=files.write_problem(plan.problem),
            output=files.write_output(plan.output),
            method=self.model_dump(mode="json", exclude_computed_fields=True),
            inputs=inputs,
            payloads=payloads,
            constructors=constructors,
        )

    @classmethod
    def load_archive(cls, saved, files):
        """Rebuild and bind a Plan written by save_archive, without replanning.

        A classical Plan recomputes the observable's binary exponent from the
        stored operator and requires it to equal the saved one before binding
        the host kernel. A quantum Plan checks every stored preparation
        payload against its selected block record and rebinds the blocks.
        Raises ValueError for another format or a changed exponent.
        """
        from nwqlib.blocks.selection import _preparation_circuit, _primitive_circuit
        from nwqlib.blocks._archive import validate_preparation_binding

        if saved.get("format") != "expectation/5":
            raise ValueError(
                f"unsupported Expectation archive format {saved.get('format')!r}. "
                "This NWQLib reads only 'expectation/5'. Open it with the NWQLib release that "
                "wrote it, or plan and run the problem again."
            )
        method = cls.model_validate(saved["method"])
        plan = files.read_plan(
            saved["plan"],
            problem=files.read_problem(saved["problem"]),
            method=method,
            output=files.read_output(saved["output"]),
            reconstruction=ExpectationReconstruction.model_validate(
                saved["plan"]["reconstruction"]
            ),
        )
        if plan.execution == "classical":
            from nwqlib.operators.inputs import _observable_exponent
            exponent = 0
            if plan.experiments:
                _classical_action_requirements(plan.problem.observable, plan.problem.state,
                                               max_bytes=method.max_bytes)
                exponent = _observable_exponent(plan.problem.observable)
            if plan.reconstruction.operator_exponent != exponent:
                raise ValueError("saved observable exponent differs from original input storage")
            return plan._bind(blocks=method._host_blocks(plan) if plan.experiments else (),
                              operator_exponent=exponent)
        inputs = {name: files.read_state(value) for name, value in saved["inputs"].items()}
        constructors = dict(preparation=_preparation_circuit, primitives=_primitive_circuit)
        for record, name, constructor in zip(
            plan.construction.selections, saved["payloads"], saved["constructors"], strict=True
        ):
            if name is not None:
                validate_preparation_binding(record, inputs[name], constructors[constructor])
        blocks = tuple(
            SelectedBlock.bind(
                record,
                payload=None if name is None else inputs[name],
                constructor=constructors[constructor],
            )
            for record, name, constructor in zip(
                plan.construction.selections, saved["payloads"], saved["constructors"], strict=True
            )
        )
        return plan._bind(blocks=blocks)
