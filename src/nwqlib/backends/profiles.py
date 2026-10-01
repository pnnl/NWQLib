"""Stored machine, allocation and finite time-model data; no machine discovery.

JSON/schema/identity use the shared Record owner. Sources and coefficients are
supplied declarations. Loading any reference never opens it or loads code.
"""

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, Field, model_validator

from nwqlib.backends.capabilities import BackendTarget
from nwqlib.core.planning import RuntimeOptions
from nwqlib.core.records import ContentID, Float64, Limit, Nonnegative, Real, Record, Source, Text
from nwqlib.evidence.records import Evidence
from nwqlib.execution import TimeScope
from nwqlib.ir.expressions import Binding
from nwqlib.operators.access import Count
from nwqlib.resources.records import MetricBasis

Timestamp = Annotated[AwareDatetime, AfterValidator(lambda value: value.astimezone(timezone.utc))]
TimeFeature = Literal["invocations", "sampled_shots", "exact_evaluations", "logical_operations"]


def _dates(recorded_at: datetime, valid_until: datetime) -> None:
    """Reject a validity window that ends before it was recorded."""
    if valid_until < recorded_at:
        raise ValueError("valid_until precedes recorded_at")


def _limits(limits: tuple[Limit, ...], locations: tuple[str, ...] | None = None) -> None:
    """Reject two limits on one (stage, metric, scope), and a capacity stock outside the granted locations.

    A capacity stock names exactly one location, because capacities are never
    pooled across locations. ``locations`` is None for a device profile, which
    has no grants.
    """
    keys = [(item.stage, item.metric, item.scope) for item in limits]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate stage/metric/scope limit")
    if locations is not None and any(
        item.kind == "capacity_stock" and item.scope not in locations for item in limits
    ):
        raise ValueError("capacity stocks must name one granted location")


class DeviceConfiguration(Record):
    """The machine a device profile describes: target, hardware, runtime, build, compiler and numerical precision.

    Build it with keyword arguments and pass it as `configuration=` to
    [`DeviceProfile`][nwqlib.backends.profiles.DeviceProfile]. `name`, `version`,
    `target`, `hardware`, `runtime` and `build` are required. Its content hash
    (`content_id`) is what an [`Allocation`][nwqlib.backends.profiles.Allocation]
    and a [`ModelDomain`][nwqlib.backends.profiles.ModelDomain] name as their
    `configuration_id`, so a time model can refer to the machine without referring
    to the profile that contains it. An optional field left `None` stays unknown
    in an assessment.

    Attributes:
        name: Required. Name of the stored specification.
        version: Required. Version of the stored specification.
        target: Required. The backend target the machine supports, the same
            target record that backend adapters declare.
        hardware: Required. Source describing the hardware.
        runtime: Required. Source of the intended runtime. Its `name` must equal
            `target.name`. It describes what preparation is expected to use, not an
            observed preparation record.
        build: Required. Source of the implementation and build configuration.
        compiler: Default `None`. Source of the intended compiler.
        precision: Default `None`. `"complex64"`, `"complex128"` or `"float64"`.
        representation: Default `None`. `"statevector"`, `"density_matrix"`,
            `"tensor"`, `"hardware"` or `"classical"`.

    Raises:
        ValueError: If `runtime.name` differs from `target.name`.
    """

    name: Text
    version: Text
    target: BackendTarget
    hardware: Source
    runtime: Source
    build: Source
    compiler: Source | None = None
    precision: Literal["complex64", "complex128", "float64"] | None = None
    representation: Literal["statevector", "density_matrix", "tensor", "hardware", "classical"] | None = None

    @model_validator(mode="after")
    def _target(self):
        """Require the runtime Source to name this configuration's own target."""
        if self.runtime.name != self.target.name:
            raise ValueError("runtime Source must name this actual BackendTarget")
        return self


class Allocation(Record):
    """The resources granted to one workload: locations, their placement and their capacity limits.

    Build it with keyword arguments, for example
    `Allocation(name=..., configuration_id=configuration.content_id, locations=("logical_device",), topology="single_device", limits=(...), recorded_at=..., valid_until=..., evidence=...)`,
    and pass it as `allocation=` to `nwqlib.estimate`, `nwqlib.compare` or
    [`assess`][nwqlib.backends.assessment.assess]. Every field is required. The
    record states a grant that the caller supplies. NWQLib never queries a
    scheduler or device for it, and it never configures ranks, threads or devices.
    Capacities are never pooled across locations. For example, two 8-GiB devices
    cannot hold a 10-GiB peak on one device. The
    [device-domain section of the profiles guide](../profiles.md#device-and-physical-model-domains)
    explains how capacities are compared.

    Attributes:
        name: Required. Name of the grant.
        configuration_id: Required. `content_id` of the
            [`DeviceConfiguration`][nwqlib.backends.profiles.DeviceConfiguration]
            the grant is for. An assessment applies the grant's limits only when it
            matches the profile's configuration.
        locations: Required. Distinct names of the granted locations, the names
            the Program's registers and workspaces use, such as
            `"logical_device"`.
        topology: Required. `"single_device"` (exactly one location),
            `"independent_devices"`, `"distributed"`, or `None` when the placement
            is unknown.
        limits: Required. Capacity, consumption and deadline limits as `Limit`
            records, at most one per stage, metric and scope. A limit of kind
            `"capacity_stock"` names exactly one granted location as its `scope`.
        recorded_at: Required. Time zone-aware time the grant was recorded,
            stored in UTC.
        valid_until: Required. Time zone-aware end of the grant's validity, not
            before `recorded_at`.
        evidence: Required. Evidence describing the grant, not its use.

    Raises:
        ValueError: If the locations are empty or repeated, `"single_device"`
            has more than one location, two limits share a stage, metric and
            scope, a `"capacity_stock"` limit names a location outside the grant, or
            `valid_until` precedes `recorded_at`.
    """

    name: Text
    configuration_id: ContentID
    locations: tuple[Text, ...]
    topology: Literal["single_device", "independent_devices", "distributed"] | None
    limits: tuple[Limit, ...]
    recorded_at: Timestamp
    valid_until: Timestamp
    evidence: Evidence

    @model_validator(mode="after")
    def _grants(self):
        """Require distinct locations, one location for ``single_device``, valid limits and dates."""
        if not self.locations or len(set(self.locations)) != len(self.locations):
            raise ValueError("allocation requires distinct granted locations")
        if self.topology == "single_device" and len(self.locations) != 1:
            raise ValueError("single_device topology requires one location")
        _limits(self.limits, self.locations)
        _dates(self.recorded_at, self.valid_until)
        return self


class ModelDomain(Record):
    """The workloads a time model is valid for: machine, allocation, readout, operation mix and size ranges.

    Build it with keyword arguments and pass it as `domain=` to
    [`TimeModel`][nwqlib.backends.profiles.TimeModel]. Every field except
    `rotation_precision`, `synthesis` and `bindings` is required. An assessment
    evaluates the model only for a workload inside every range and list here,
    and reports the prediction as unavailable otherwise. The record selects,
    fits and extrapolates nothing. An empty list of selections or gates supports
    none, never any. Size ranges are inclusive, and `max_resets`,
    `max_measurements`, `max_classical_work` and `max_adaptive_rounds` each give
    the inclusive range from 0 to the maximum. A model calibrated without resets,
    for example, does not apply once resets are added, even when the gate counts
    match. The [time-model section of the profiles guide](../profiles.md#finite-time-models)
    lists every domain check.

    Attributes:
        configuration_id: Required. `content_id` of the
            [`DeviceConfiguration`][nwqlib.backends.profiles.DeviceConfiguration]:
            hardware, runtime, build, precision and target.
        allocation_id: Required. `content_id` of the
            [`Allocation`][nwqlib.backends.profiles.Allocation]: every granted
            location, limit and topology.
        basis: Required. Gate basis in which logical operations are counted:
            `"selected_logical"`, `"cx"`, `"clifford_t"` or `"toffoli"`.
        rotation_precision: Default `None`. Positive rotation-synthesis precision
            the operation counts assume.
        synthesis: Default `None`. Source of the synthesis rule the operation
            counts assume.
        batch_schedule: Required. `"unspecified"` or `"serial"`.
        acquisition: Required. `"direct_observation"` or `"measurement_batch"`.
        runtime: Required. The runtime seed the model was measured with, as
            `RuntimeOptions(seed=...)`, or `"seed_independent"`. A whole-Plan
            forecast does not guess future seeds, so only a seed-independent model
            applies there.
        population: Required. Readout population, `"unconditional"` or
            `"native_conditioned"`.
        readouts: Required. Distinct supported readouts: `"pauli_expectation"`,
            `"counts"`, `"probabilities"`, `"host_scalars"`, `"amplitudes"` or
            `"estimated_observable"`.
        selection_ids: Required. Distinct content hashes of the supported block
            selections.
        primitive_gates: Required. Distinct supported primitive gates: `"x"`,
            `"h"`, `"z"`, `"sdg"`, `"cx"`, `"mc_z"` or `"phase"`.
        min_qubits: Required. Smallest supported qubit count.
        max_qubits: Required. Largest supported qubit count, at least
            `min_qubits`.
        min_operations: Required. Smallest supported logical operation count.
        max_operations: Required. Largest supported logical operation count, at
            least `min_operations`.
        max_shots: Required. Largest supported shot count.
        max_exact_evaluations: Required. Largest supported number of exact
            evaluations.
        max_readout_items: Required. Largest supported number of returned readout
            items.
        max_resets: Required. Largest supported reset count.
        max_measurements: Required. Largest supported measurement count.
        max_classical_work: Required. Largest supported classical work.
        max_adaptive_rounds: Required. Largest supported number of adaptive
            rounds.
        bindings: Default `()`. Parameter values the model is restricted to, at
            most one per parameter.

    Raises:
        ValueError: If a minimum exceeds its maximum, `rotation_precision` is not
            positive, or a binding, readout, selection or gate is repeated.
    """

    configuration_id: ContentID
    allocation_id: ContentID
    basis: MetricBasis
    rotation_precision: Float64 | None = None
    synthesis: Source | None = None
    batch_schedule: Literal["unspecified", "serial"]
    acquisition: Literal["direct_observation", "measurement_batch"]
    runtime: RuntimeOptions | Literal["seed_independent"]
    population: Literal["unconditional", "native_conditioned"]
    readouts: tuple[Literal["pauli_expectation", "counts", "probabilities", "host_scalars", "amplitudes", "estimated_observable"], ...]
    selection_ids: tuple[ContentID, ...]
    primitive_gates: tuple[Literal["x", "h", "z", "sdg", "cx", "mc_z", "phase"], ...]
    min_qubits: Count
    max_qubits: Count
    min_operations: Count
    max_operations: Count
    max_shots: Count
    max_exact_evaluations: Count
    max_readout_items: Count
    max_resets: Count
    max_measurements: Count
    max_classical_work: Count
    max_adaptive_rounds: Count
    bindings: tuple[Binding, ...] = ()

    @model_validator(mode="after")
    def _domain(self):
        """Require ordered scale intervals, a positive rotation precision and no duplicate entries."""
        if self.min_qubits > self.max_qubits or self.min_operations > self.max_operations:
            raise ValueError("model scale domain is reversed")
        if self.rotation_precision is not None and self.rotation_precision.value <= 0:
            raise ValueError("model rotation precision must be positive")
        if len({binding.parameter for binding in self.bindings}) != len(self.bindings):
            raise ValueError("duplicate model domain binding")
        for name in ("readouts", "selection_ids", "primitive_gates"):
            values = getattr(self, name)
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate model {name}")
        return self


class TimeCoefficient(Record):
    """One coefficient of a linear time model: seconds per invocation, shot, exact evaluation or logical operation.

    Build it with keyword arguments, for example
    `TimeCoefficient(feature="sampled_shots", seconds_per_unit=0.001, unit="s/shot")`,
    and pass it in `coefficients=` of
    [`TimeModel`][nwqlib.backends.profiles.TimeModel]. Every field is required.
    The unit is kept exactly as supplied and never converted. A zero coefficient
    states that the feature contributes nothing, which differs from leaving the
    feature out.

    Attributes:
        feature: Required. `"invocations"`, `"sampled_shots"`,
            `"exact_evaluations"` or `"logical_operations"`.
        seconds_per_unit: Required. Nonnegative coefficient in `unit`.
        unit: Required. The one unit of the feature: `"s/invocation"`,
            `"s/shot"`, `"s/exact_evaluation"` or `"s/logical_operation"`
            respectively.

    Raises:
        ValueError: If `unit` is not the unit of `feature`.
    """

    feature: TimeFeature
    seconds_per_unit: Nonnegative
    unit: Literal["s/invocation", "s/shot", "s/exact_evaluation", "s/logical_operation"]

    @model_validator(mode="after")
    def _units(self):
        """Require the one unit that belongs to the coefficient's feature. Units are never converted."""
        expected = {
            "invocations": "s/invocation", "sampled_shots": "s/shot",
            "exact_evaluations": "s/exact_evaluation", "logical_operations": "s/logical_operation",
        }[self.feature]
        if self.unit != expected:
            raise ValueError("coefficient feature/unit mismatch")
        return self


class ModelUncertainty(Record):
    """The supplied uncertainty of a time model, as an additive interval in seconds around its prediction.

    Build it with keyword arguments, for example
    `ModelUncertainty(kind="future_run_prediction", lower_residual_seconds=-0.25, upper_residual_seconds=0.5, coverage=0.9, source=...)`,
    and pass it as `uncertainty=` to
    [`TimeModel`][nwqlib.backends.profiles.TimeModel]. Every field except
    `coverage` is required, and `coverage` is required for the two statistical
    kinds. The residuals are added to the central prediction, so the example
    gives the interval `[prediction - 0.25, prediction + 0.5]` seconds. The
    interval keeps the meaning its source states and is never an execution
    guarantee.

    Attributes:
        kind: Required. `"future_run_prediction"` concerns one future run.
            `"sample_mean_confidence"` concerns a mean and does not bound an
            individual run. `"model_error_envelope"` states an engineering
            uncertainty without a probability.
        lower_residual_seconds: Required. Zero or negative residual in seconds.
        upper_residual_seconds: Required. Nonnegative residual in seconds.
        coverage: Default `None`. Probability in (0, 1] that the interval holds,
            required for `"future_run_prediction"` and `"sample_mean_confidence"`
            and not allowed for `"model_error_envelope"`, because a coverage would
            claim more than an engineering uncertainty supports.
        source: Required. Source of the stated interval.

    Raises:
        ValueError: If `coverage` is missing for a statistical kind or given for
            `"model_error_envelope"`.
    """

    kind: Literal["future_run_prediction", "sample_mean_confidence", "model_error_envelope"]
    lower_residual_seconds: Annotated[Real, Field(le=0)]
    upper_residual_seconds: Nonnegative
    coverage: Annotated[Real, Field(gt=0, le=1)] | None = None
    source: Source

    @model_validator(mode="after")
    def _coverage(self):
        """Require a coverage for a statistical interval and none for an engineering envelope.

        An engineering envelope states an uncertainty without a probability, so
        a coverage would claim more than its source supports.
        """
        if (self.kind == "model_error_envelope") != (self.coverage is None):
            raise ValueError("statistical intervals require coverage; engineering envelopes do not")
        return self


class CalibrationReference(Record):
    """Where a calibrated time model's coefficients came from: training data, validation data and validation errors.

    Build it with keyword arguments and pass it as `calibration=` to
    [`TimeModel`][nwqlib.backends.profiles.TimeModel]. Every field is required.
    Each Source names the exact data or procedure version. NWQLib never opens or
    fetches a referenced source. A link to a parent record alone does not
    establish which workloads trained the model.

    Attributes:
        source: Required. Source of the calibration procedure.
        training: Required. Source of the training workloads.
        validation: Required. Source of the held-out validation workloads.
        validation_errors: Required. Source of the held-out error report.
    """

    source: Source
    training: Source
    validation: Source
    validation_errors: Source


class TimeModel(Record):
    """A named linear model of time in seconds, the sum of coefficient times feature count, for one scope.

    Build it with keyword arguments and pass it in `models=` of
    [`DeviceProfile`][nwqlib.backends.profiles.DeviceProfile]. Every field except
    `form`, `calibration` and `uncertainty` is required. The one supported form,
    `acquisition_linear/1`, is

    ```text
    seconds = c_invocation * invocations + c_shot * sampled_shots
            + c_exact * exact_evaluations + c_operation * logical_operations
    ```

    over the feature counts of the resource estimate, so it is evaluated without
    running anything ([Finite time models](../profiles.md#finite-time-models)).
    An engineering model states assumed coefficients and gives conditional
    numbers under its assumptions. A calibrated model keeps its supplied
    calibration sources and uncertainty unchanged. Neither kind collects
    calibration data. An expired, future-dated or out-of-domain model gives an
    unavailable prediction with its reason.

    Attributes:
        form: Default `"acquisition_linear/1"`, the only accepted value.
            Arbitrary expressions and plugins are rejected.
        name: Required. Model name, unique within the profile.
        kind: Required. `"engineering"` for assumed coefficients or
            `"calibrated"` for coefficients from a supplied calibration.
        scope: Required. What the seconds cover: `"selected_acquisition"`,
            `"acquisition_overhead"` or `"native_call_wall"`. The first two
            exclude planning, compilation, queue delay, analysis and whole-run
            cost. `"native_call_wall"` covers the synchronous backend call,
            including waiting and result extraction, and excludes preparation and
            saving. The scopes overlap, so their predictions must not be added,
            and none covers a whole run.
        coefficients: Required. One to four
            [`TimeCoefficient`][nwqlib.backends.profiles.TimeCoefficient] records
            with distinct features.
        domain: Required. The [`ModelDomain`][nwqlib.backends.profiles.ModelDomain]
            of workloads the model is valid for.
        recorded_at: Required. Time zone-aware time the model was recorded,
            stored in UTC.
        valid_until: Required. Time zone-aware end of validity, not before
            `recorded_at`.
        evidence: Required. Evidence of kind `"numerical_estimate"` for an
            engineering model or `"empirical_prediction"` for a calibrated one.
        assumptions: Required. At least one stated assumption.
        calibration: Default `None`. The
            [`CalibrationReference`][nwqlib.backends.profiles.CalibrationReference],
            required for a calibrated model and not allowed for an engineering one.
        uncertainty: Default `None`. The
            [`ModelUncertainty`][nwqlib.backends.profiles.ModelUncertainty],
            required for a calibrated model.

    Raises:
        ValueError: If `valid_until` precedes `recorded_at`, a feature repeats,
            `calibration` does not match `kind`, the evidence kind does not match
            `kind`, or a calibrated model has no `uncertainty`.
    """

    form: Literal["acquisition_linear/1"] = "acquisition_linear/1"
    name: Text
    kind: Literal["engineering", "calibrated"]
    scope: TimeScope
    coefficients: Annotated[tuple[TimeCoefficient, ...], Field(min_length=1, max_length=4)]
    domain: ModelDomain
    recorded_at: Timestamp
    valid_until: Timestamp
    evidence: Evidence
    assumptions: Annotated[tuple[Text, ...], Field(min_length=1)]
    calibration: CalibrationReference | None = None
    uncertainty: ModelUncertainty | None = None

    @model_validator(mode="after")
    def _model(self):
        """Keep engineering and calibrated models distinct.

        A calibrated model needs its calibration provenance, empirical-prediction
        evidence and a stated uncertainty. An engineering model has no
        calibration provenance and carries numerical-estimate evidence. Features
        are distinct.
        """
        _dates(self.recorded_at, self.valid_until)
        if len({c.feature for c in self.coefficients}) != len(self.coefficients):
            raise ValueError("duplicate model feature coefficient")
        if (self.kind == "calibrated") != (self.calibration is not None):
            raise ValueError("calibrated model requires existing calibration provenance only")
        expected = "numerical_estimate" if self.kind == "engineering" else "empirical_prediction"
        if self.evidence.kind != expected:
            raise ValueError("model evidence must preserve engineering versus calibrated meaning")
        if self.kind == "calibrated" and self.uncertainty is None:
            raise ValueError("calibrated model requires its supplied uncertainty meaning")
        return self


class DeviceProfile(Record):
    """A stored description of one machine: its configuration, advertised limits and named time models.

    Build it with keyword arguments, for example
    `DeviceProfile(configuration=..., recorded_at=..., valid_until=..., evidence=..., models=(model,))`,
    and pass it as `profile=` to `nwqlib.estimate`, `nwqlib.compare` or
    [`assess`][nwqlib.backends.assessment.assess]. `configuration`,
    `recorded_at`, `valid_until` and `evidence` are required. The profile is data
    the caller supplies. NWQLib never discovers a machine or refreshes a
    profile. New calibration makes a new profile with a new content hash, so an
    earlier assessment is never rewritten. The [profiles guide](../profiles.md)
    explains the assessment, and the example under
    [Forecast cost and feasibility](backends.md#forecast-cost-and-feasibility) builds a
    small profile.

    Attributes:
        configuration: Required. The
            [`DeviceConfiguration`][nwqlib.backends.profiles.DeviceConfiguration].
        limits: Default `()`. Advertised limits as `Limit` records, at most one
            per stage, metric and scope.
        recorded_at: Required. Time zone-aware time the profile was recorded,
            stored in UTC.
        valid_until: Required. Time zone-aware end of validity, not before
            `recorded_at`.
        evidence: Required. The specification the profile rests on, not a record
            of an execution.
        models: Default `()`. [`TimeModel`][nwqlib.backends.profiles.TimeModel]
            records with distinct names.

    Raises:
        ValueError: If `valid_until` precedes `recorded_at`, two limits share a
            stage, metric and scope, or two models share a name.
    """

    configuration: DeviceConfiguration
    limits: tuple[Limit, ...] = ()
    recorded_at: Timestamp
    valid_until: Timestamp
    evidence: Evidence
    models: tuple[TimeModel, ...] = ()

    @model_validator(mode="after")
    def _profile(self):
        """Check the dates, the advertised limits and that model names are distinct."""
        _dates(self.recorded_at, self.valid_until)
        _limits(self.limits)
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("stored models require distinct names")
        return self
