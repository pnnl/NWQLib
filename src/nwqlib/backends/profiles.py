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
    """The immutable hardware/runtime/build/precision domain of one profile.

    name/version identify the stored specification. target is the actual common
    BackendTarget. runtime is the intended PreparedArtifact.target Source, not
    an observed preparation receipt; compiler is the corresponding intended
    compiler Source. build preserves implementation/build configuration. None
    precision/representation/compiler remains missing information.
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
    """Actual resource grants to this workload, never a scheduler or device query.

    configuration_id names the exact intended machine configuration. locations
    use the selected Program/Workspace names; capacities are never pooled across
    them. topology describes the supplied placement, and None is unknown.
    limits reuse core.Limit; scope on a capacity stock is its one location.
    recorded_at/valid_until and evidence describe the grant, not its consumption.
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
    """Finite validation envelope for this model, including exact allocation.

    configuration_id binds hardware, runtime/build, precision and the actual
    target. allocation_id binds all granted locations/resources/topology.
    selection_ids and primitive_gates describe the supported operation mix;
    a missing inventory is not a wildcard. Logical operations are in basis and
    the explicit synthesis/rotation_precision context. Scale intervals are
    inclusive. No model is selected, fitted or extrapolated by this record.
    max_resets, max_measurements, max_classical_work and max_adaptive_rounds
    give inclusive finite [0, maximum] domains for the separate folded non-gate
    populations. bindings restrict the law's actual parameter point; population
    owns its readout population. Neither restriction belongs to Evidence.
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
    """Nonnegative seconds per one named acquisition/work feature.

    unit keeps the source coefficient unit exactly; there is no conversion.
    Zero coefficients explicitly omit a contribution, never an unknown feature.
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
    """Supplied additive residual interval in seconds, not an inferred guarantee.

    future_run_prediction describes future individual acquisitions;
    sample_mean_confidence describes a mean and does not bound an individual run.
    model_error_envelope keeps a stated engineering uncertainty without claiming
    probabilistic coverage. Residuals are added to the central prediction.
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
    """Existing calibration provenance; references are inert and never fetched.

    training/validation name the supplied workloads, validation_errors preserves
    the held-out error report. References name the actual data/procedure version;
    a record parent link alone does not establish its training population.
    """

    source: Source
    training: Source
    validation: Source
    validation_errors: Source


class TimeModel(Record):
    """One named, versioned scalar relation: sum coefficient * feature, in seconds.

    The supported form ``acquisition_linear/1`` is defined in docs/profiles.md,
    "Finite time models". A linear law over counted features can be evaluated
    from the resource fold without executing anything, and its coefficients
    keep explicit units.

    form has one supported version; arbitrary expressions/plugins are rejected.
    scope separates acquisition, overhead and native-call wall time. The last
    includes native waiting/extraction, excluding preparation and publication.
    These scopes overlap and must not be summed. None covers an entire run.
    engineering parameters are explicit assumptions, not measured calibration.
    calibrated parameters keep supplied calibration and uncertainty unchanged.
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
    """Versioned stored configuration, advertised limits and supplied named models.

    configuration has a separate identity so a coefficient can bind the machine
    without a circular reference to its containing model/profile. New calibration
    changes the Profile identity and never rewrites an earlier assessment.
    evidence is the specification basis; it is not an execution receipt.
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
