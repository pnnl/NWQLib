"""Immutable shared Program DAG and declared interfaces, independent of SDKs."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from nwqlib.core.records import InputRef, Record, Source, Text
from .expressions import (
    AdmissionLimits, Binding, ExprRef, Expression, Integer, NonnegativeInt, Parameter,
)


class Register(Record):
    """A quantum register of a Program, allocated explicitly by an `Allocate` node.

    Build it as `Register(name="q", width=2)` and pass it in `registers=` of a
    [`Program`][nwqlib.ir.records.Program]. `name` and `width` are required.

    Attributes:
        name: Required. Register name, used by ports, allocation and measurement.
        width: Required. Number of qubits, an integer or an
            [`ExprRef`][nwqlib.ir.expressions.ExprRef].
        location: Default `"logical_device"`. Device location on which the
            resource estimate places the register.
        role: Default `"system"`. `"system"`, `"clean_ancilla"` or
            `"dirty_ancilla"`, reported separately in memory and qubit counts.
    """

    name: Text
    width: Integer
    location: Text = "logical_device"
    role: Literal["system", "clean_ancilla", "dirty_ancilla"] = "system"


class ClassicalValue(Record):
    """A named classical value of a Program, such as measured bits or a computed scalar.

    Build it as `ClassicalValue(name="bits", dtype="bits", width=2)` and pass it
    in `classical=` of a [`Program`][nwqlib.ir.records.Program]. `name` and
    `dtype` are required. Bits have a positive width, and the other types have
    none.

    Attributes:
        name: Required. Value name.
        dtype: Required. `"bits"`, `"bool"`, `"integer"` or `"real"`.
        width: Default `None`. Number of bits, positive and required for
            `"bits"`, `None` otherwise.

    Raises:
        ValueError: If `width` is missing or not positive for bits, or given for
            another type.
    """

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
    """A port of a block signature: a whole register the block acts on, with the state it needs and the effect it promises.

    Build it as `QuantumPort(name="system", width=2)` and pass it in `quantum=`
    of a [`BlockSignature`][nwqlib.ir.records.BlockSignature]. `name` and `width`
    are required. A call maps each port, in signature order, to a register that
    no other port of the call uses. The zero state counts as coherent. A
    `"unitary"` effect keeps the register's coherence epoch (the generation
    counter that `StateClaim` defines) but clears a known zero state, and
    `"preserve"` promises that the
    state is restored exactly. A `"zero"` or `"coherent"` output asserts a newly
    prepared state, which the block's implementation must guarantee. An
    `"unknown"` effect makes the Program not ready and cannot satisfy a later
    state requirement.

    Attributes:
        name: Required. Port name.
        width: Required. Number of qubits, an integer or an
            [`ExprRef`][nwqlib.ir.expressions.ExprRef].
        requires: Default `"live"`. Input state the port needs: any allocated
            (`"live"`) state, `"zero"` or `"coherent"`.
        ensures: Default `"unitary"`. Effect on the register: `"unitary"`,
            `"preserve"`, `"zero"`, `"coherent"` or `"unknown"`.
    """

    name: Text
    width: Integer
    requires: Literal["live", "zero", "coherent"] = "live"
    ensures: Literal["unitary", "preserve", "zero", "coherent", "unknown"] = "unitary"


class BlockSignature(Record):
    """The declared interface of a block that a Program calls: its target action, ports and parameters.

    Build it with keyword arguments, for example
    `BlockSignature(name="h", target=source, quantum=(QuantumPort(name="system", width=1),))`,
    and pass it in `signatures=` of a [`Program`][nwqlib.ir.records.Program].
    `name` and `target` are required. A
    [`BlockCall`][nwqlib.ir.records.BlockCall] names the signature, and the
    selected definition bound to it supplies the action, its cost rules and the
    circuit constructor. The signature declares an interface. It does not prove
    that an implementation performs the action.

    Attributes:
        name: Required. Signature name, used by block calls and by the selected
            definition.
        target: Required. Versioned Source of the action a selection must
            implement.
        quantum: Default `()`. Ordered [`QuantumPort`][nwqlib.ir.records.QuantumPort]
            records.
        parameters: Default `()`. Scalar [`Parameter`][nwqlib.ir.expressions.Parameter]
            records with their domains.
        interface: Default `"declared"`. `"unknown"` keeps an undeclared block in
            the Program, which is then not ready.
        coupling: Default `"joint"`, when the call may correlate its ports, or
            `"independent"` when it promises not to.
        obligations: Default `()`. Promises that the selected implementation must
            meet.
    """

    name: Text
    target: Source
    quantum: tuple[QuantumPort, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    interface: Literal["declared", "unknown"] = "declared"
    coupling: Literal["joint", "independent"] = "joint"
    obligations: tuple[Text, ...] = ()


class PortMap(Record):
    """Maps one port of a block signature to a whole register in a block call.

    Build it as `PortMap(port="system", wire="q")`. Both arguments are required.

    Attributes:
        port: Required. Port name of the signature.
        wire: Required. Register name.
    """

    port: Text
    wire: Text


class Argument(Record):
    """Gives a scalar parameter of a block signature the value of an expression in a block call.

    Build it as `Argument(parameter="angle", value=ExprRef(expression="theta"))`.
    Both arguments are required.

    Attributes:
        parameter: Required. Parameter name of the signature.
        value: Required. The [`ExprRef`][nwqlib.ir.expressions.ExprRef] of the
            value.
    """

    parameter: Text
    value: ExprRef


class Sequence(Record):
    """Runs its child nodes in order, even when they act on disjoint registers.

    Build it as `Sequence(children=("allocate", "measure"))`, with the IDs of
    other definitions.

    Attributes:
        children: Default `()`. Definition IDs, in order.
        kind: Fixed `"sequence"`. Names the node type in the saved record.
    """

    kind: Literal["sequence"] = "sequence"
    children: tuple[Text, ...] = ()


class Parallel(Record):
    """Runs block calls concurrently on disjoint registers and joins them.

    Build it as `Parallel(children=("call_a", "call_b"))`. Every child must be a
    direct block call whose ports declare `"unitary"` or `"preserve"`. Children that
    share registers, and classical work or lifetime changes, are rejected. Use
    [`Sequence`][nwqlib.ir.records.Sequence] for serial order.

    Attributes:
        children: Default `()`. Definition IDs of the block calls.
        kind: Fixed `"parallel"`. Names the node type in the saved record.
    """

    kind: Literal["parallel"] = "parallel"
    children: tuple[Text, ...] = ()


class Repeat(Record):
    """Repeats a body a fixed number of times, without unrolling it.

    Build it as `Repeat(body="step", count=4)`. Both arguments are required. The
    check verifies that the body can be repeated and never expands it, so a
    large count costs no more checking work than a small one.

    Attributes:
        body: Required. Definition ID of the body.
        count: Required. Nonnegative repetition count, an integer or an
            [`ExprRef`][nwqlib.ir.expressions.ExprRef].
        kind: Fixed `"repeat"`. Names the node type in the saved record.
    """

    kind: Literal["repeat"] = "repeat"
    body: Text
    count: Integer


class BlockCall(Record):
    """Calls a declared block signature on whole registers, with ports in signature order.

    Build it as `BlockCall(signature="h", ports=(PortMap(port="system", wire="q"),))`.
    `signature` is required. Each register is used by one port only. The call
    names the signature, not an implementation. The selected definition bound to
    that signature supplies the action, cost rules and circuit constructor.

    Attributes:
        signature: Required. Signature name.
        ports: Default `()`. One [`PortMap`][nwqlib.ir.records.PortMap] per port,
            in signature order.
        arguments: Default `()`. [`Argument`][nwqlib.ir.records.Argument] values
            of the signature's parameters.
        kind: Fixed `"block_call"`. Names the node type in the saved record.
    """

    kind: Literal["block_call"] = "block_call"
    signature: Text
    ports: tuple[PortMap, ...] = ()
    arguments: tuple[Argument, ...] = ()


class StateClaim(Record):
    """Claims that a register is still in a given coherent epoch. Measurement and reset start a new epoch.

    Use it in `claims=` of a [`CoherentRegion`][nwqlib.ir.records.CoherentRegion].
    An epoch numbers the coherence generations of one register. The first
    allocation starts it at 0. Every measurement, reset, release, reallocation
    and block port that promises a newly prepared zero or coherent output adds
    one, to the register it acts on and to every register correlated with it. A
    claim therefore names one specific coherent state, and a claim written
    before a measurement no longer matches after it.
    [Program checks](../development/program_checks.md) gives the full rules.

    Attributes:
        wire: Required. Register whose coherence is claimed.
        epoch: Required. Epoch the register must currently have.
    """

    wire: Text
    epoch: NonnegativeInt


class CoherentRegion(Record):
    """Requires the claimed registers to stay in their coherent epochs throughout the body.

    Build it as `CoherentRegion(body="kernel", claims=(StateClaim(wire="q", epoch=0),))`.
    Both arguments are required. A measurement, reset or other new preparation
    of a claimed register inside the body is rejected.

    Attributes:
        body: Required. Definition ID of the body.
        claims: Required. The [`StateClaim`][nwqlib.ir.records.StateClaim]
            records.
        kind: Fixed `"coherent_region"`. Names the node type in the saved
            record.
    """

    kind: Literal["coherent_region"] = "coherent_region"
    body: Text
    claims: tuple[StateClaim, ...]


class Allocate(Record):
    """Allocates a register in the zero state, starting its lifetime in the current experiment.

    Build it as `Allocate(wire="q")`.

    Attributes:
        wire: Required. Register name.
        kind: Fixed `"allocate"`. Names the node type in the saved record.
    """

    kind: Literal["allocate"] = "allocate"
    wire: Text


class Release(Record):
    """Discards a register. Its quantum state cannot be used after this point.

    Build it as `Release(wire="q")`.

    Attributes:
        wire: Required. Register name.
        kind: Fixed `"release"`. Names the node type in the saved record.
    """

    kind: Literal["release"] = "release"
    wire: Text


class Measure(Record):
    """Measures a whole register in the computational basis into a classical bits value. The register stays allocated in a new epoch.

    Build it as `Measure(wire="q", result="bits")`. `wire` and `result` are
    required.

    Attributes:
        wire: Required. Register name.
        result: Required. Name of a classical bits value of the same width.
        kind: Fixed `"measure"`. Names the node type in the saved record.
    """

    kind: Literal["measure"] = "measure"
    wire: Text
    result: Text


class Reset(Record):
    """Resets an allocated register to zero, starting a new coherent epoch.

    Build it as `Reset(wire="q")`.

    Attributes:
        wire: Required. Register name.
        kind: Fixed `"reset"`. Names the node type in the saved record.
    """

    kind: Literal["reset"] = "reset"
    wire: Text


class Branch(Record):
    """Chooses between two bodies on a classical value. Both paths are checked.

    Build it as `Branch(condition="flag", when_true="a", when_false="b")`. Every
    argument is required. After the branch, a register state or classical value
    is available only as far as both paths make it available.

    Attributes:
        condition: Required. Name of an available classical value.
        when_true: Required. Definition ID of the body for true.
        when_false: Required. Definition ID of the body for false.
        kind: Fixed `"branch"`. Names the node type in the saved record.
    """

    kind: Literal["branch"] = "branch"
    condition: Text
    when_true: Text
    when_false: Text


class ClassicalStage(Record):
    """A classical computation inside the quantum job or outside it (`boundary="host"`).

    The check runs no code. Build it with keyword arguments, for example
    `ClassicalStage(implementation=source, inputs=("bits",), outputs=("value",))`.
    `implementation` is required. A stage with `boundary="host"` requires
    every register to be released first and can name the selected kernel
    that runs it.

    Attributes:
        implementation: Required. Versioned Source of the computation.
        inputs: Default `()`. Classical values read, which must be available on
            every path.
        outputs: Default `()`. Classical values defined.
        arguments: Default `()`. Scalar [`Argument`][nwqlib.ir.records.Argument]
            values.
        boundary: Default `"in_job"`, processing inside the quantum job, or
            `"host"`, a classical computation that NWQLib runs outside the
            quantum job.
        kernel: Default `None`. Name of the selected kernel that runs a
            `"host"` stage.
        kind: Fixed `"classical_stage"`. Names the node type in the saved
            record.
    """

    kind: Literal["classical_stage"] = "classical_stage"
    implementation: Source
    inputs: tuple[Text, ...] = ()
    outputs: tuple[Text, ...] = ()
    arguments: tuple[Argument, ...] = ()
    boundary: Literal["in_job", "host"] = "in_job"
    kernel: Text | None = None


class AdaptiveLoop(Record):
    """Repeats a body zero to `max_rounds` times, with a classical policy after each round and a declared stopping rule.

    Build it with keyword arguments. `body`, `max_rounds`, `policy` and
    `termination` are required. The stopping rule is kept as text and never
    executed.

    Attributes:
        body: Required. Definition ID of one round.
        max_rounds: Required. Upper bound on the rounds, an integer or an
            [`ExprRef`][nwqlib.ir.expressions.ExprRef].
        policy: Required. Versioned [`ClassicalStage`][nwqlib.ir.records.ClassicalStage]
            run after each round, which may read that round's results.
        termination: Required. Declared stopping rule.
        resource_envelope: Default `None`. Stated bound on the cost of every
            round. With `None`, the resource estimate leaves the loop's cost
            unknown.
        kind: Fixed `"adaptive_loop"`. Names the node type in the saved
            record.
    """

    kind: Literal["adaptive_loop"] = "adaptive_loop"
    body: Text
    max_rounds: Integer
    policy: ClassicalStage
    termination: Text
    resource_envelope: Text | None = None


class MetadataRef(Record):
    """A reference to method metadata attached to a setting, in a format the method defines. It is never decoded by the check.

    Build it as `MetadataRef(format=source, data=input_ref)`. Both arguments are
    required.

    Attributes:
        format: Required. Source of the metadata format.
        data: Required. `InputRef` of the stored metadata.
    """

    format: Source
    data: InputRef


class Setting(Record):
    """One experiment of a measurement batch: a label, parameter values and method metadata.

    Build it as `Setting(label="z_basis", bindings=(...), metadata=MetadataRef(...))`.
    `label` and `metadata` are required. The label, bindings and metadata
    together determine the experiment's content hash.

    Attributes:
        label: Required. Experiment label.
        bindings: Default `()`. [`Binding`][nwqlib.ir.expressions.Binding] values
            of this experiment.
        metadata: Required. The [`MetadataRef`][nwqlib.ir.records.MetadataRef].
    """

    label: Text
    bindings: tuple[Binding, ...] = ()
    metadata: MetadataRef


class RangeAxis(Record):
    """An integer parameter range `[start, stop)` with a positive step, kept compact and never expanded.

    Build it as `RangeAxis(parameter="k", start=0, stop=8)` and pass it in
    `axes=` of a [`MeasurementBatch`][nwqlib.ir.records.MeasurementBatch], which
    combines every setting with every value. `parameter`, `start` and `stop` are
    required.

    Attributes:
        parameter: Required. Parameter name.
        start: Required. Nonnegative first value.
        stop: Required. Nonnegative end, excluded and greater than `start`.
        step: Default `1`. Positive step.

    Raises:
        ValueError: If the range is empty.
    """

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
    """Independent experiments that share one body. No quantum or classical state passes between them.

    Build it with keyword arguments, for example
    `MeasurementBatch(body="experiment", settings=(setting,), repetitions=1, observation_kind="counts")`.
    `body` and `settings` are required. The settings are combined with the range
    axes. A requirement that depends on an axis value keeps the Program not ready
    until one experiment is selected and bound
    ([`Program.select_experiment`][nwqlib.ir.records.Program.select_experiment]).
    [`lower_qiskit`][nwqlib.blocks.lowering.lower_qiskit] builds one circuit of the body, not its repetitions. Every
    register the body allocates must be released at its end.

    Attributes:
        body: Required. Definition ID of the shared body.
        settings: Required. At least one [`Setting`][nwqlib.ir.records.Setting].
        axes: Default `()`. [`RangeAxis`][nwqlib.ir.records.RangeAxis] ranges.
        scope: Default `"independent_experiments"`, the only accepted value.
        repetitions: Default `None`. Independent runs of the body, an integer or
            an [`ExprRef`][nwqlib.ir.expressions.ExprRef]. `None` leaves the
            planning requirement unknown. Preparing an exact readout kind requires 1.
        observation_kind: Default `None`. For the innermost batch, `"counts"` for
            sampled counts, `"pauli_expectation"` or `"probabilities"` for exact
            statistics, `"estimated_observable"` for a provider estimate, or
            `"trajectory"`.
            A `"trajectory"` is one exact evaluation of the body whose
            observation points belong to the selected experiment's readout, not
            to the settings. An outer batch has `None`.
        schema_version: Fixed `2`. Version of the saved record format.
        kind: Fixed `"measurement_batch"`. Names the node type in the saved
            record.
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
    """One node of a Program with its ID. Nodes refer to each other by ID, so a shared body is stored once.

    Build it as `Definition(id="main", node=Sequence(children=(...)))` and pass it
    in `definitions=` of a [`Program`][nwqlib.ir.records.Program]. Both arguments
    are required.

    Attributes:
        id: Required. Unique ID within the Program.
        node: Required. One node: `Sequence`, `Repeat`, `BlockCall`,
            `CoherentRegion`, `Allocate`, `Release`, `Measure`, `Reset`,
            `Branch`, `ClassicalStage`, `AdaptiveLoop`, `MeasurementBatch` or
            `Parallel`.
    """

    id: Text
    node: Node


class Readiness(Record):
    """The result of a Program's structural check: what is still unresolved, and the checking work it took.

    [`Program.check_readiness`][nwqlib.ir.records.Program.check_readiness] returns
    it. The fields below are read-only. A ready Program is structurally valid. It
    is not a check of the blocks' promised actions or an approval to run.

    Attributes:
        blockers: Sorted unresolved requirements, such as an unbound width or an
            unknown block effect. Empty when the Program is ready.
        expression_evaluations: Expression evaluations of this check.
        lifecycle_steps: Other checking work units of this check, not quantum
            events or algorithm costs.
    """

    blockers: tuple[Text, ...]
    expression_evaluations: NonnegativeInt
    lifecycle_steps: NonnegativeInt

    @property
    def ready(self) -> bool:
        """Whether no requirement is unresolved, that is, `blockers` is empty."""
        return not self.blockers

    def require_ready(self):
        """Return this result, or raise when a requirement is still unresolved.

        Returns:
            readiness (Readiness): This result.

        Raises:
            ValueError: If `blockers` is not empty. The message lists them.
        """
        if self.blockers:
            raise ValueError("Program is not ready: " + "; ".join(self.blockers))
        return self


class Program(Record):
    """A circuit described as named steps: registers, block calls, measurements and their order, checked when it is built.

    Build it with keyword arguments and use it in a
    [`SelectedConstruction`][nwqlib.blocks.records.SelectedConstruction].
    `root` and `definitions` are required. Every node is a
    [`Definition`][nwqlib.ir.records.Definition] with an ID, and nodes refer to
    each other by ID, so a shared body is stored once however often it runs, and
    the checking work grows with its distinct contexts rather than its run
    count. The root starts with no allocated registers or classical values.
    Registers still allocated at the end of the root stay available to
    code that uses the quantum output, while the body of a measurement batch must
    release its registers. Construction runs the full structural check of the
    declared effects, so an illegal lifecycle or a graph over its limits is
    rejected before anything uses the Program. Unresolved requirements, such as
    an unbound width, do not reject it but make it not ready.

    One Program is the single description of a construction that the structural
    check, resource estimate, circuit building and readout all read, so a cost, a
    circuit and a readout cannot describe different constructions. Its content
    hash covers every table, binding and limit. [Run your own circuit](../own_circuit.md)
    builds a Program around a supplied circuit, and
    [Describe a circuit as a Program](../ir.md) explains the nodes.

    Attributes:
        root: Required. Definition ID of the root node.
        definitions: Required. The [`Definition`][nwqlib.ir.records.Definition]
            records.
        expressions: Default `()`. [`Expression`][nwqlib.ir.expressions.Expression]
            records that widths, counts and arguments refer to.
        parameters: Default `()`. Declared [`Parameter`][nwqlib.ir.expressions.Parameter]
            records.
        constraints: Default `()`. Bool expressions that every bound point must
            satisfy.
        registers: Default `()`. Quantum [`Register`][nwqlib.ir.records.Register]
            records.
        classical: Default `()`. [`ClassicalValue`][nwqlib.ir.records.ClassicalValue]
            records.
        signatures: Default `()`. [`BlockSignature`][nwqlib.ir.records.BlockSignature]
            records of the blocks called.
        bindings: Default `()`. Parameter values bound so far.
        limits: Default `AdmissionLimits()`. The
            [`AdmissionLimits`][nwqlib.ir.expressions.AdmissionLimits] on the
            stored structure and on the work of checking it.
        premises: Default `()`. Stated assumptions of the construction.
        schema_version: Fixed `2`. Version of the saved record format.

    Raises:
        ValueError: If the structure breaks a lifecycle rule, refers to an
            unknown ID, or exceeds `limits`.

    Examples:
        Allocate a two-qubit register and measure it.

        >>> from nwqlib.ir import (Allocate, ClassicalValue, Definition, Measure,
        ...                        Program, Register, Sequence)
        >>> program = Program(
        ...     root="main",
        ...     definitions=(
        ...         Definition(id="allocate", node=Allocate(wire="q")),
        ...         Definition(id="measure", node=Measure(wire="q", result="bits")),
        ...         Definition(id="main",
        ...                    node=Sequence(children=("allocate", "measure"))),
        ...     ),
        ...     registers=(Register(name="q", width=2),),
        ...     classical=(ClassicalValue(name="bits", dtype="bits", width=2),),
        ... )
        >>> program.check_readiness().ready
        True
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
        """Return the result of this Program's structural check.

        The check runs once per Program object, when it is built, and its result is
        kept for later calls. Copies and unpickled Programs check again on first
        use. It covers every concrete rule and lists unresolved requirements as
        blockers. `expression_evaluations + lifecycle_steps` of the result is the
        checking work it measured.

        Returns:
            readiness (Readiness): The result of the check.
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
        """Return a copy of the Program with parameter values bound, keeping shared bodies.

        The copy runs the structural check again.

        Args:
            **values (int | Float64): Parameter values by name, an exact integer or a `Float64`.

        Returns:
            program (Program): The bound Program.

        Raises:
            ValueError: If the bound Program fails the structural check.
        """
        bound = {item.parameter: item for item in self.bindings}
        bound.update({key: Binding(parameter=key, value=value) for key, value in values.items()})
        return self.revise(bindings=tuple(bound[key] for key in sorted(bound)))

    def select_experiment(self, batch_id: str, setting_index: int, **axis_values):
        """Return a Program for one experiment of a measurement batch, without expanding any range or product.

        The returned Program's root is the batch with the one selected setting and
        no axes. It keeps the dependency closure of that batch, every global
        constraint and the register layout, and the selected values also bind the
        global constraints. It creates no observation or execution ID, because the
        original Program identifies the whole source graph. Use it before
        [`lower_qiskit`][nwqlib.blocks.lowering.lower_qiskit], which builds the
        circuit of one static experiment, a Program that describes exactly one
        circuit.

        Args:
            batch_id (str): Definition ID of the measurement batch.
            setting_index (int): Index of the setting.
            **axis_values (int): One value for each range axis of the batch.

        Returns:
            program (Program): The Program of the selected experiment.

        Raises:
            ValueError: If the batch, setting or axis values are invalid.
        """
        from .selection import select_experiment
        from .validation import _Admission
        admission = _Admission(self)
        admission.setup()
        return select_experiment(admission, batch_id, setting_index, axis_values)

    def iter_definitions(self):
        """Return an iterator over the stored definitions, each once in declaration order, without expanding anything.

        Returns:
            definitions (Iterator[Definition]): The definitions.
        """
        return iter(self.definitions)
