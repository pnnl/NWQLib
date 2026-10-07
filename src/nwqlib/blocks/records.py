"""Portable selected block records, bound to the shared Program graph."""

from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from nwqlib.core.records import Basis, ContentID, InputRef, Real, Record, Source, Text
from nwqlib.ir import Binding, BlockCall, BlockSignature, ClassicalStage, Program, Sequence
from nwqlib.artifacts import ArrayOutput
from nwqlib.operators.access import Count
from nwqlib.problems.inputs import StatePreparationSpec
from nwqlib.resources.records import ResourceLaw, Workspace


class Primitive(Record):
    """One exact gate in a block's recipe. A global phase has no target qubits and becomes observable under control.

    [`SelectedDefinition.decomposition`][nwqlib.blocks.records.SelectedDefinition]
    lists these in order. Build it with keyword arguments, for example
    `Primitive(gate="cx", qubits=(0, 1))`. `gate` is required.

    Attributes:
        gate: Required. `"x"`, `"h"`, `"z"` or `"sdg"` on one qubit, `"cx"` on
            (control, target), `"mc_z"` on two or more qubits, or a global
            `"phase"` with no qubits.
        qubits: Default `()`. Distinct local qubit indices in the block's ports
            concatenated in signature order, index 0 being the least significant.
        angle: Default `0.0`. Phase angle in radians, nonzero only for `"phase"`,
            which multiplies the block by `exp(i*angle)`.

    Raises:
        ValueError: If the qubits repeat, their number does not match the gate,
            or a gate other than `"phase"` has an angle.
    """

    gate: Literal["x", "h", "z", "sdg", "cx", "mc_z", "phase"]
    qubits: tuple[Count, ...] = ()
    angle: Real = 0.0

    @model_validator(mode="after")
    def _arity(self):
        size = len(self.qubits)
        if len(set(self.qubits)) != size:
            raise ValueError("primitive targets must be distinct")
        if ((self.gate == "phase" and size != 0)
                or (self.gate in {"x", "h", "z", "sdg"} and size != 1)
                or (self.gate == "cx" and size != 2)
                or (self.gate == "mc_z" and size < 2)):
            raise ValueError("primitive target arity does not match its gate")
        if self.gate != "phase" and self.angle != 0:
            raise ValueError("only a phase primitive has an angle")
        return self


class BlockSemantics(Record):
    """The action a circuit block promises: its relation, projectors, normalization, error and phase convention.

    `SelectedDefinition.semantics` holds it, and
    `SelectedConstruction.encoding_semantics` derives it for a Pauli
    encoding.
    `epsilon` is algorithmic error only. This record gives no bound on
    floating-point roundoff of the synthesis.
    No QSP, evolution, or hardware accuracy is implied by this record.

    It states what the block's circuit must do and what code that uses the
    block may rely on. Keeping it with the block's selection lets a Method
    derive, for example, an encoding's ``A/alpha`` relation from the chosen
    SELECT block instead of storing a second copy that could disagree.

    Attributes:
        kind: Block kind: `"preparation"`, `"signed_pauli_select"`,
            `"pauli_readout"`, `"zero_reflection"`, `"block_encoding"`,
            `"unitary_transform"` (a control or adjoint of another block), or
            `"unknown"` for a promise of none of these kinds.
        input: Reference (`InputRef`) to the input the block encodes, or None.
        basis: System basis, dimension and bit ordering of the action.
        relation: The promised mathematical relation, as text.
        alpha: Positive physical normalization for Pauli SELECT and block encodings.
        index_qubits: Width of the index or encoding ancilla register.
        input_projector: Input subspace on which the relation holds.
        output_projector: Output subspace that carries the relation.
        success: Postselection event and its probability, or deterministic.
        workspace: Extra quantum workspace in qubits, or None when unknown.
        restoration: What happens to index or workspace registers.
        epsilon: Algorithmic error of the relation in approximation_metric, or None when unknown.
        approximation_metric: Norm in which epsilon is stated.
        approximation_evidence: Why epsilon has its value.
        inverse_legal: Whether an adjoint may be selected.
        control_legal: Whether one coherent control may be added.
        phase: Global-phase convention, which becomes observable under control.
        preparation: Specification (`StatePreparationSpec`) of the input
            state the block prepares: the whole action of a preparation
            block, required for that kind, or the first step of a composite
            block such as an ADAPT query.
        base_semantics: For a `"unitary_transform"`, and only for it, the
            promised action of the block before the control or adjoint.
    """

    kind: Literal["preparation", "signed_pauli_select", "pauli_readout", "zero_reflection", "block_encoding", "unitary_transform", "unknown"]
    input: InputRef | None
    basis: Basis
    relation: Text
    alpha: Annotated[Real, Field(gt=0)] | None = None
    index_qubits: Count = 0
    input_projector: Text
    output_projector: Text
    success: Text
    workspace: Count | None
    restoration: Text
    epsilon: Annotated[Real, Field(ge=0)] | None
    approximation_metric: Text
    approximation_evidence: Text
    inverse_legal: StrictBool
    control_legal: StrictBool
    phase: Text
    preparation: StatePreparationSpec | None = None
    base_semantics: "BlockSemantics | None" = None

    @model_validator(mode="after")
    def _normalization(self):
        if self.kind in ("block_encoding", "signed_pauli_select") and self.alpha is None:
            raise ValueError("Pauli semantics keep the positive physical normalization")
        if self.kind == "preparation" and self.preparation is None:
            raise ValueError("preparation semantics require the input-bound preparation specification")
        if (self.kind == "unitary_transform") != (self.base_semantics is not None):
            raise ValueError("only a unitary transform keeps a separate base semantic contract")
        if self.base_semantics is not None and self.base_semantics.kind == "unitary_transform":
            raise ValueError("nested semantic transformations are outside the selected subset")
        return self


class SelectedDefinition(Record):
    """The portable half of a selected block: its interface, promised action, chosen recipe and cost rules.

    Read it as `block.record` of a [`SelectedBlock`][nwqlib.blocks.selection.SelectedBlock],
    or from `SelectedConstruction.selections`. The `select_*` functions build it.
    It holds everything resource estimates, saved archives and reports need, and
    its content hash covers every field. The executable half is the
    `SelectedBlock` that binds trusted code to this exact record. Two recipes for
    the same full operator, such as X and HZH, are different selections with
    different content hashes and gate counts. `decomposition` gives the ordered
    exact gate recipe when one exists. Otherwise `cost_law` names the cost rule
    of the circuit construction. An unknown cost stays unresolved and never
    becomes zero. `construction_work` counts the documented size rule, not CPU
    time or memory. The fields below are read-only.

    Attributes:
        signature: Declared interface, equal to the Program's signature of the
            same name.
        semantics: The promised action and its approximation: kind, input,
            relation, normalization `alpha`, projectors, error `epsilon` and its
            norm, whether control and adjoint are allowed, and the phase
            convention.
        implementation: Versioned Source of the implementation.
        choice: Recipe label within the implementation. A supplied circuit gets a
            fresh unique label.
        decomposition: Exact ordered [`Primitive`][nwqlib.blocks.records.Primitive]
            recipe, or `None` when a cost rule describes the cost.
        cost_law: Source of the size or CX rule, or `None` when there is none.
        cost_parameters: Exact arguments of that rule for this selection.
        cost_context: Scope and limits of the rule, carried into resource
            quantities.
        construction_work: Work units of building the block's circuit definition by its
            size rule, or `None` when unknown.
        controlled: Whether this selection adds one coherent control to its base.
        adjoint: Whether this selection is the adjoint of its base.
        base_selection_id: Content hash of the base selection of a control or
            adjoint.
        coefficient_selection_id: Content hash of the SELECT whose coefficients
            this PREP prepares.
        blocker: Why the selection cannot run, for a block with a cost rule only
            or an unsupported block.
        resource_laws: Per-call resource rules.
        workspace: Workspace bytes per call, by location. An empty tuple means
            unknown, not zero.
    """

    signature: BlockSignature
    semantics: BlockSemantics
    implementation: Source
    choice: Text
    decomposition: tuple[Primitive, ...] | None
    cost_law: Source | None
    cost_parameters: tuple[Binding, ...] = ()
    cost_context: Text
    construction_work: Count | None
    controlled: StrictBool = False
    adjoint: StrictBool = False
    base_selection_id: Text | None = None
    coefficient_selection_id: ContentID | None = None
    blocker: Text | None = None
    resource_laws: tuple[ResourceLaw, ...] = ()
    workspace: tuple[Workspace, ...] = ()

    @model_validator(mode="after")
    def _primitive_width(self):
        parameters = {parameter.name for parameter in self.signature.parameters}
        if any(not set(law.unbound_parameters) <= parameters for law in self.resource_laws):
            raise ValueError("resource law coverage requires declared formal parameters")
        widths = tuple(port.width for port in self.signature.quantum)
        if self.decomposition is not None and all(type(width) is int for width in widths):
            # Transforms keep the base recipe's target numbering. Their outer
            # control is represented by controlled, not by rewriting primitives.
            width = sum(widths) - int(self.controlled)
            if any(index >= width for operation in self.decomposition for index in operation.qubits):
                raise ValueError("selected primitive target is outside its register width")
        return self


class PauliEncoding(Record):
    """Names the PREP, SELECT and PREP-adjoint calls that form one Pauli LCU block encoding in a Program.

    Build it with keyword arguments, for example
    `PauliEncoding(root="encoding", select="select", prepare="prep", unprepare="unprep")`,
    and pass it in `encodings=` of
    [`SelectedConstruction`][nwqlib.blocks.records.SelectedConstruction], which
    checks the structure. `root` and `select` are required. The normalization,
    the physical operator A and the error come from the checked SELECT, so this
    record holds no second copy of them. With the label register in `|0>` the
    subgraph acts as `A/alpha`.

    Attributes:
        root: Required. Definition ID of the Sequence that forms the encoding.
        select: Required. Signature name of the signed Pauli SELECT.
        prepare: Default `None`. Signature name of the coefficient PREP, `None`
            for a single-term encoding.
        unprepare: Default `None`. Signature name of the adjoint of that PREP,
            `None` for a single-term encoding.
    """

    root: Text
    select: Text
    prepare: Text | None = None
    unprepare: Text | None = None


class SelectedKernel(Record):
    """Declaration of a classical computation that a Program runs on this computer: its inputs, outputs, work and dependencies.

    `SelectedConstruction.kernels` holds the declarations, and a
    `ClassicalStage` with `boundary="host"` names one to run it. The
    declaration is saved with the Plan. The callable that reads the input
    data is not saved: the Method attaches it to the Plan as a
    ``BoundKernel`` with `Plan._bind`.
    ``construction_work`` and ``invocation_work`` are declared size units
    counted against the Run, not CPU-instruction counts. The ``resource_laws``
    entries specify the numerical work covered, its assumptions and whether
    the count is exact, an upper bound or an estimate. Without an applicable
    entry, the total numerical work is unknown. Workspace declarations keep
    known bytes and unknown components separately.

    Attributes:
        name: Name of the computation, referenced by the Program's
            `ClassicalStage` with `boundary="host"`.
        implementation: Versioned `Source` of the classical algorithm.
        inputs: References (`InputRef`) to the inputs the computation reads.
        scalars: Ordered labels of the scalar statistics it returns.
        scalar_frames: Frame of each scalar, ``physical`` for physical units, ``unit`` for a normalized direction, or ``encoded_branch`` for a branch mass or norm in the encoded frame that the Method declares.
        outputs: Declared named arrays it may return.
        resource_laws: Fixed ``classical_work`` formulas (`ResourceLaw`) of
            the computation, without parameter bindings, control or adjoint.
        workspace: Per-invocation memory declarations (`Workspace`). An
            entry with unknown bytes keeps that component unknown.
        construction_work: Size units counted against the Run when the
            computation is bound at preparation.
        invocation_work: Size units counted against the Run for each
            invocation.
        dependencies: Distribution names, such as ``numpy``, whose installed
            versions each preparation of the computation records.
        application_bytes: Upper bound, using the Run's JSON byte accounting, on
            the stored size of the sequence of ``KernelApplication`` records
            returned by each invocation when the Method fixes them at selection.
            The Run sets aside this many bytes along with the declared outputs.
            At completion, the smaller of this value and the returned sequence's
            JSON byte bound is deducted from the metadata byte count before it is
            compared with ``ExecutionLimits.max_completion_metadata_bytes``.
            Bytes beyond the declared bound count toward that metadata allowance.
            The default, ``0``, leaves all these bytes to the metadata allowance.
    """

    schema_version: Literal[3] = 3
    name: Text
    implementation: Source
    inputs: tuple[InputRef, ...]
    scalars: tuple[Text, ...]
    scalar_frames: tuple[Literal["physical", "unit", "encoded_branch"], ...]
    outputs: tuple[ArrayOutput, ...] = ()
    resource_laws: tuple[ResourceLaw, ...]
    workspace: tuple[Workspace, ...]
    construction_work: Count
    invocation_work: Count
    dependencies: tuple[Text, ...] = ()
    application_bytes: Count = 0

    @model_validator(mode="after")
    def _selected_outputs(self):
        names = self.scalars + tuple(output.name for output in self.outputs)
        if not self.scalars or len(set(names)) != len(names):
            raise ValueError("host output names must be distinct with declared scalar statistics")
        if len(self.scalar_frames) != len(self.scalars):
            raise ValueError("each selected scalar requires its declared vector frame")
        if len({item.identity for item in self.inputs}) != len(self.inputs):
            raise ValueError("host selected input identities must be distinct")
        if any(law.metric != "classical_work" or law.bindings or law.unbound_parameters or law.controlled or law.adjoint
               for law in self.resource_laws):
            raise ValueError("host laws describe untransformed fixed classical work")
        return self

    @property
    def data_bytes(self):
        return sum(output.data_bytes for output in self.outputs)


class SelectedConstruction(Record):
    """A Program with the selected definition of every block it calls: what estimates, circuit building and OpenQASM export read.

    Build it with keyword arguments, for example
    `SelectedConstruction(program=program, selections=tuple(b.record for b in blocks))`,
    or read it as `plan.construction`. `program` and `selections` are required.
    It is not a Plan, and holds no problem, method or data. Saved JSON keeps the
    sharing of `Program.definitions`, and executable code is bound again by
    exact record when the circuit is built, never from saved code.
    The checks require exactly one selected definition per Program signature,
    with an identical signature, so every block call resolves to one selection,
    and every stage that names a kernel is a `"host"` stage that names its
    exact selected kernel. A Pauli encoding with
    a label register must be the ordered PREP, SELECT, PREP-adjoint Sequence on
    that register, where the PREP was selected from this SELECT's coefficients
    and the adjoint is of that same PREP. A single-term encoding has no label
    register and is the SELECT alone. That structure is what makes
    `<0|U|0> = A/alpha` hold for the subgraph.

    Attributes:
        program: Required. The [`Program`][nwqlib.ir.records.Program] that estimates,
            circuit building and export read.
        selections: Required. One
            [`SelectedDefinition`][nwqlib.blocks.records.SelectedDefinition] per
            Program signature.
        encodings: Default `()`. [`PauliEncoding`][nwqlib.blocks.records.PauliEncoding]
            subgraphs, checked as PREP, SELECT and PREP-adjoint.
        kernels: Default `()`. Classical kernel declarations, each named by
            at least one `ClassicalStage` with `boundary="host"`.
        schema_version: Fixed `2`. Version of the saved record format.

    Raises:
        ValueError: If a signature lacks exactly one matching selection, a
            `"host"` stage and its kernel do not match, or an encoding breaks the
            PREP, SELECT, PREP-adjoint structure.
    """

    schema_version: Literal[2] = 2
    program: Program
    selections: tuple[SelectedDefinition, ...]
    encodings: tuple[PauliEncoding, ...] = ()
    kernels: tuple[SelectedKernel, ...] = ()

    @model_validator(mode="after")
    def _selection_bindings(self):
        """Require one exact selection per Program signature, exact kernel references and the PREP, SELECT and inverse order of each Pauli encoding."""
        signatures = {signature.name: signature for signature in self.program.signatures}
        selected = {item.signature.name: item for item in self.selections}
        if len(selected) != len(self.selections) or selected.keys() != signatures.keys():
            raise ValueError("provide exactly one selected definition per Program signature")
        if any(item.signature != signatures[name] for name, item in selected.items()):
            raise ValueError("selected implementation signature differs from Program declaration")
        nodes = {definition.id: definition.node for definition in self.program.definitions}
        kernels = {item.name: item for item in self.kernels}
        if len(kernels) != len(self.kernels):
            raise ValueError("selected kernel names must be unique")
        references = set()
        for node in nodes.values():
            if isinstance(node, ClassicalStage) and node.kernel is not None:
                kernel = kernels.get(node.kernel)
                if (node.boundary != "host" or kernel is None or node.implementation != kernel.implementation):
                    raise ValueError("host stage must reference its exact selected kernel declaration")
                references.add(node.kernel)
        if references != kernels.keys():
            raise ValueError("selected kernels require exact Program stage references")
        for encoding in self.encodings:
            sequence = nodes.get(encoding.root)
            if not isinstance(sequence, Sequence):
                raise ValueError("Pauli encoding root must be its ordered shared Sequence")
            calls = [nodes[child] for child in sequence.children]
            names = ([encoding.prepare, encoding.select, encoding.unprepare]
                     if encoding.prepare is not None else [encoding.select])
            if (not all(isinstance(call, BlockCall) for call in calls)
                    or [call.signature for call in calls] != names):
                raise ValueError("Pauli encoding must keep selected PREP/SELECT/inverse order")
            select = selected[encoding.select]
            if (select.implementation.name != "pauli.signed_select" or select.controlled or select.adjoint):
                raise ValueError("encoding relation requires an untransformed signed SELECT")
            if select.semantics.index_qubits:
                if encoding.prepare is None or encoding.unprepare is None:
                    raise ValueError("Pauli encoding requires coefficient PREP and its selected inverse")
                prep, inverse = selected[encoding.prepare], selected[encoding.unprepare]
                if (prep.coefficient_selection_id != select.content_id or prep.controlled or prep.adjoint
                        or inverse.base_selection_id != prep.content_id or not inverse.adjoint or inverse.controlled):
                    raise ValueError("encoding PREP and inverse must bind this SELECT's coefficient state")
                index_wire = calls[1].ports[0].wire
                if calls[0].ports[0].wire != index_wire or calls[2].ports[0].wire != index_wire:
                    raise ValueError("encoding coefficient operations must map the SELECT index register")
            elif encoding.prepare is not None or encoding.unprepare is not None:
                raise ValueError("single-term encoding has no coefficient register")
        return self

    def encoding_semantics(self, root: str) -> BlockSemantics:
        """Return the block-encoding promise of one checked Pauli encoding: A, alpha, label convention and error.

        The relation is the LCU block encoding of An, Childs and Lin,
        arXiv:2312.03916v2, Appendix A.3, Lemma 24, Eq. (178), in the
        real-coefficient form where both preparation oracles equal PREP and
        `sign(c_j)` sits in SELECT. With `alpha = sum_j |c_j|` the label-zero block is
        `A/alpha`. This is an exact block encoding in the sense of their
        Definition 23 (Appendix A.2). Measuring the label register in `|0>`
        succeeds with probability `||A psi||^2 / alpha^2`, and the label register is
        not restored in general.

        Args:
            root (str): Definition ID of the encoding's Sequence.

        Returns:
            semantics (BlockSemantics): The SELECT's promise, restated as a block encoding.

        Raises:
            ValueError: If no checked encoding has this root.
        """
        encoding = next((item for item in self.encodings if item.root == root), None)
        if encoding is None:
            raise ValueError("no checked Pauli encoding at this root")
        select = next(item for item in self.selections if item.signature.name == encoding.select)
        return select.semantics.revise(
            kind="block_encoding", relation="<0_index|U|0_index>=A/alpha",
            input_projector="index all-zero projector tensor system identity",
            output_projector="index all-zero projector tensor system identity",
            success="measure index all-zero; probability ||A psi||^2/alpha^2",
            restoration="index is not unconditionally restored; no additional workspace",
        )
