"""Immutable shared Program DAG and declared interfaces, independent of SDKs."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from nwqlib.core.records import InputRef, Record, Source, Text
from .expressions import (
    AdmissionLimits, Binding, ExprRef, Expression, Integer, NonnegativeInt, Parameter,
)


class Register(Record):
    """A whole quantum register in declaration order. Allocation is explicit.

    Attributes:
        name: Register name used by ports, allocation and measurement.
        width: Number of qubits, exact or a local expression reference.
        location: Logical device on which the resource fold places this width.
        role: ``system``, ``clean_ancilla`` or ``dirty_ancilla``, reported separately in footprints.
    """

    name: Text
    width: Integer
    location: Text = "logical_device"
    role: Literal["system", "clean_ancilla", "dirty_ancilla"] = "system"


class ClassicalValue(Record):
    """A named classical value; bits have a positive width, scalars have none."""

    name: Text
    dtype: Literal["bits", "bool", "integer", "real"]
    width: Integer | None = None

    @model_validator(mode="after")
    def _width(self):
        if (self.dtype == "bits") != (self.width is not None):
            raise ValueError("only bits require a width")
        if type(self.width) is int and self.width == 0:
            raise ValueError("classical bits width must be positive")
        return self


class QuantumPort(Record):
    """Ordered exclusive whole-register port and promised input/output state effect.

    Zero is a coherent state. Unitary keeps the epoch but clears known zero;
    preserve promises exact state restoration. Coherent/zero outputs assert new
    preparation obligations for the semantic owner to check.
    Unknown effects block readiness and cannot satisfy later state promises.

    Attributes:
        name: Formal port name, mapped once per call in signature order.
        width: Port width, exact or a local expression reference.
        requires: Input state the port needs: any ``live`` state, ``zero`` or ``coherent``.
        ensures: Declared effect on the register's state and coherence epoch.
    """

    name: Text
    width: Integer
    requires: Literal["live", "zero", "coherent"] = "live"
    ensures: Literal["unitary", "preserve", "zero", "coherent", "unknown"] = "unitary"


class BlockSignature(Record):
    """Versioned semantic target and declared interface, not an action proof.

    Attributes:
        name: Signature name referenced by BlockCall and by its selected definition.
        target: Versioned Source of the semantic action a selection must implement.
        quantum: Ordered exclusive quantum ports.
        parameters: Formal scalar parameters with their domains.
        interface: ``declared``, or ``unknown`` to keep an undeclared block representable with blockers.
        coupling: ``joint`` when the call may correlate its ports, ``independent`` when it promises not to.
        obligations: Semantic promises for the selected implementation's owner to check.
    """

    name: Text
    target: Source
    quantum: tuple[QuantumPort, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    interface: Literal["declared", "unknown"] = "declared"
    coupling: Literal["joint", "independent"] = "joint"
    obligations: tuple[Text, ...] = ()


class PortMap(Record):
    """Map one named formal port to an exclusive whole register."""

    port: Text
    wire: Text


class Argument(Record):
    """Map a formal parameter to a local expression definition."""

    parameter: Text
    value: ExprRef


class Sequence(Record):
    """Ordered serial children, even on disjoint registers."""

    kind: Literal["sequence"] = "sequence"
    children: tuple[Text, ...] = ()


class Parallel(Record):
    """Concurrent disjoint declared unitary/preserve BlockCalls, with a join.

    Children must be direct calls; shared wires, classical work and lifetime
    changes are rejected. Sequence remains serial even on disjoint registers.
    """

    kind: Literal["parallel"] = "parallel"
    children: tuple[Text, ...] = ()


class Repeat(Record):
    """Nonnegative repetitions; admission checks an iteration contract, never unrolls."""

    kind: Literal["repeat"] = "repeat"
    body: Text
    count: Integer


class BlockCall(Record):
    """Invoke a declared signature on exclusive whole registers, ports in signature order.

    The call names its signature, not an implementation. The selected
    definition bound to that signature supplies the action, cost laws and
    native constructor.
    """

    kind: Literal["block_call"] = "block_call"
    signature: Text
    ports: tuple[PortMap, ...] = ()
    arguments: tuple[Argument, ...] = ()


class StateClaim(Record):
    """Claim a current coherent register epoch. Measurement and reset advance it.

    An epoch numbers the coherence generations of one register. The first
    allocation starts it at 0. Every measurement, reset, release,
    reallocation and block port that promises a newly prepared zero or
    coherent output adds one, to the register it acts on and to every
    register correlated with it. A claim therefore names one specific coherent
    state, and a claim written before a measurement no longer matches after
    it. docs/ir.md, "Declared quantum and classical lifecycle", gives the
    full rules.

    Attributes:
        wire: Register whose coherence is claimed.
        epoch: Epoch the register must currently have.
    """

    wire: Text
    epoch: NonnegativeInt


class CoherentRegion(Record):
    """The named coherent epochs must remain unbroken throughout the body."""

    kind: Literal["coherent_region"] = "coherent_region"
    body: Text
    claims: tuple[StateClaim, ...]


class Allocate(Record):
    """Begin a fresh zero-state register lifetime in the current experiment."""

    kind: Literal["allocate"] = "allocate"
    wire: Text


class Release(Record):
    """Discard a wire explicitly; its quantum state cannot cross later boundaries."""

    kind: Literal["release"] = "release"
    wire: Text


class Measure(Record):
    """Measure a whole register into typed bits; wire stays live in a new epoch."""

    kind: Literal["measure"] = "measure"
    wire: Text
    result: Text
    basis: Text = "computational"


class Reset(Record):
    """Prepare zero on a live wire, beginning a new coherent epoch."""

    kind: Literal["reset"] = "reset"
    wire: Text


class Branch(Record):
    """Both paths are checked; later availability is their typed intersection."""

    kind: Literal["branch"] = "branch"
    condition: Text
    when_true: Text
    when_false: Text


class ClassicalStage(Record):
    """In-job processing or explicit host boundary. Admission executes no callable.

    Attributes:
        kind: Node discriminator.
        implementation: Versioned Source of the classical computation.
        inputs: Classical values read, which must be available on every path.
        outputs: Classical values defined.
        arguments: Scalar arguments bound to local expressions.
        boundary: ``in_job`` for processing inside the quantum job, or ``host``, which requires every register released first.
        kernel: Name of the SelectedKernel that executes a host stage, or None.
    """

    kind: Literal["classical_stage"] = "classical_stage"
    implementation: Source
    inputs: tuple[Text, ...] = ()
    outputs: tuple[Text, ...] = ()
    arguments: tuple[Argument, ...] = ()
    boundary: Literal["in_job", "host"] = "in_job"
    kernel: Text | None = None


class AdaptiveLoop(Record):
    """Zero to max_rounds iterations, with a declared policy and termination rule.

    Attributes:
        kind: Node discriminator.
        body: Definition ID of one round.
        max_rounds: Upper bound on the number of rounds.
        policy: Versioned classical stage admitted after each round, which may read that round's results.
        termination: Declared termination rule, kept as text and never executed.
        resource_envelope: Asserted uniform per-round cost bound over the history, or None, which leaves the loop's costs unknown in the resource fold.
    """

    kind: Literal["adaptive_loop"] = "adaptive_loop"
    body: Text
    max_rounds: Integer
    policy: ClassicalStage
    termination: Text
    resource_envelope: Text | None = None


class MetadataRef(Record):
    """Algorithm-owned metadata schema and immutable input reference; never decoded."""

    format: Source
    data: InputRef


class Setting(Record):
    """One experiment point: label plus bindings and metadata determine identity."""

    label: Text
    bindings: tuple[Binding, ...] = ()
    metadata: MetadataRef


class RangeAxis(Record):
    """Compact integer range [start, stop) with positive step; never expanded."""

    parameter: Text
    start: NonnegativeInt
    stop: NonnegativeInt
    step: Annotated[NonnegativeInt, Field(gt=0)] = 1

    @model_validator(mode="after")
    def _range(self):
        if self.stop <= self.start:
            raise ValueError("range axis must be nonempty and increasing")
        return self


ObservationKind = Literal["counts", "pauli_expectation", "probabilities", "estimated_observable", "trajectory"]


class MeasurementBatch(Record):
    """Independent experiments sharing a body; no quantum/classical state escapes.

    Settings combine with compact axes. Axis-dependent unresolved leaf/count
    requirements remain readiness blockers until one point is selected/bound.
    repetitions counts independent body invocations. Terminal observation_kind
    distinguishes sampled counts from exact-statistic evaluations; outer batches
    have no observation kind. None keeps an unknown planning requirement.
    A terminal ``trajectory`` kind is one exact evaluation of the selected body;
    its observation points belong to the selected Experiment's readout details,
    never to settings. As for the other exact kinds, preparation requires
    ``repetitions`` to be one.
    Logical lowering materializes one selected body, not these repetitions.
    """

    schema_version: Literal[2] = 2
    kind: Literal["measurement_batch"] = "measurement_batch"
    body: Text
    settings: Annotated[tuple[Setting, ...], Field(min_length=1)]
    axes: tuple[RangeAxis, ...] = ()
    scope: Literal["independent_experiments"] = "independent_experiments"
    repetitions: Integer | None = None
    observation_kind: ObservationKind | None = None


Node = Annotated[
    Sequence | Repeat | BlockCall | CoherentRegion | Allocate | Release | Measure
    | Reset | Branch | ClassicalStage | AdaptiveLoop | MeasurementBatch | Parallel,
    Field(discriminator="kind"),
]


class Definition(Record):
    """One local node definition. References preserve sharing in persisted JSON."""

    id: Text
    node: Node


class Readiness(Record):
    """Structural admission result, not semantic conformance or execution approval.

    Attributes:
        blockers: Sorted unresolved requirements, such as an unbound width or an unknown block effect.
        expression_evaluations: Expression evaluations performed by this check.
        lifecycle_steps: Other admission work units of this check, not quantum events or algorithm costs.
    """

    blockers: tuple[Text, ...]
    expression_evaluations: NonnegativeInt
    lifecycle_steps: NonnegativeInt

    @property
    def ready(self) -> bool:
        return not self.blockers

    def require_ready(self):
        """Refuse executable structural admission while obligations are unresolved."""
        if self.blockers:
            raise ValueError("Program is not ready: " + "; ".join(self.blockers))
        return self


class Program(Record):
    """Finite shared structure with exact bindings and explicit experiment lifetimes.

    The root starts with no allocated wires or available classical values. Root
    wires may remain live for a quantum-output consumer; independent batch bodies
    must release theirs. This layer checks declared effects only.

    One Program is the single description of a selected construction that
    every consumer reads: structural admission, resource folding, logical
    lowering and acquisition resolution. Keeping one graph means a cost, a
    circuit and a readout cannot describe different constructions. Bodies
    are referenced by local ID and Repeat keeps a count, so a shared body is
    stored once however often it runs, and admission work grows with its
    distinct contexts rather than its run count. Construction runs the
    full admission check, so an illegal lifecycle or an over-limit graph
    rejects before any consumer receives the Program. Content identity
    covers every table, binding and limit.

    The Readiness that construction computes is stored in the slot
    ``_readiness``, which is neither a field nor a private attribute, so
    equality, hashing and identity never see it. An admitted Program
    embedded in another record is reused unchanged (see ``Record``), so
    ``check_readiness`` returns the stored result and every consumer reads
    one admission per Program object. Copies and pickles start without it
    and compute it again on first use.
    """

    __slots__ = ("_readiness",)
    schema_version: Literal[2] = 2
    root: Text
    definitions: tuple[Definition, ...]
    expressions: tuple[Expression, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    constraints: tuple[ExprRef, ...] = ()
    registers: tuple[Register, ...] = ()
    classical: tuple[ClassicalValue, ...] = ()
    signatures: tuple[BlockSignature, ...] = ()
    bindings: tuple[Binding, ...] = ()
    limits: AdmissionLimits = AdmissionLimits()
    premises: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def _admit(self):
        self.check_readiness()
        return self

    def check_readiness(self) -> Readiness:
        """Return the Readiness of this Program's full admission check.

        The check runs once per Program object, at construction, and its
        result is stored; it covers concrete legality and returns blockers for
        unresolved requirements. ``Readiness.expression_evaluations +
        Readiness.lifecycle_steps`` is the admission work it measured.
        """
        try:
            readiness = object.__getattribute__(self, "_readiness")
        except AttributeError:
            readiness = None
        if readiness is None:
            from .validation import check_program
            readiness = check_program(self)
            object.__setattr__(self, "_readiness", readiness)
        return readiness

    def bind(self, **values):
        """Return a validated revision with concrete assignments; keep shared bodies."""
        bound = {item.parameter: item for item in self.bindings}
        bound.update({key: Binding(parameter=key, value=value) for key, value in values.items()})
        return self.revise(bindings=tuple(bound[key] for key in sorted(bound)))

    def select_experiment(self, batch_id: str, setting_index: int, **axis_values):
        """Select one independent experiment without expanding any range/product.

        The returned root is the selected batch with one setting and no axes.
        Keep its dependency closure and every global constraint and register
        layout. Selected values also bind global constraints. This creates no
        observation/execution ID; the parent identifies the entire source graph.
        """
        from .selection import select_experiment
        from .validation import _Admission
        admission = _Admission(self)
        admission.setup()
        return select_experiment(admission, batch_id, setting_index, axis_values)

    def iter_definitions(self):
        """Visit kept definitions once in declaration order, with no dynamic expansion."""
        return iter(self.definitions)
