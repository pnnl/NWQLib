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
    """Exact ordered logical operation. A phase has no targets and is observable under control.

    Attributes:
        gate: ``x``, ``h``, ``z`` or ``sdg`` on one qubit, ``cx`` on (control, target), ``mc_z`` on two or more qubits, or a global ``phase``.
        qubits: Local qubit indices in the block's ports concatenated in signature order, index 0 being the least significant.
        angle: Phase angle in radians, nonzero only for ``phase``, which multiplies the block by ``exp(i*angle)``.
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
    """Action, projectors, normalization and phase of one selected block kind.

    epsilon is algorithmic error only; floating synthesis roundoff is unbounded.
    No QSP, evolution, or hardware accuracy is implied by this record.

    This is the contract a native constructor must implement and the one a
    consumer may rely on. Keeping it beside the selection lets a Method
    derive, for example, an encoding's ``A/alpha`` relation from the selected
    SELECT instead of storing a second copy that could disagree.

    Attributes:
        kind: Block kind. ``unknown`` keeps an unsupported contract representable.
        input: Identity of the admitted input the block encodes, or None.
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
        preparation: Admitted specification of the input state the block prepares, its whole action for a preparation block or the first step of a composite native block such as an ADAPT query.
        base_semantics: Contract of the untransformed block for a control or adjoint.
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
    """Selected implementation and its law context, never an executable callable.

    decomposition gives the ordered exact primitive recipe when available;
    otherwise law points to the selected native kernel. Program owns composition.
    construction_work bounds the documented size law, not CPU time or RSS.
    Unknown cost remains an unresolved law; it never becomes a zero inventory.

    A selected definition is the portable half of a block. It holds everything
    the resource fold, archives and reports need, and its identity covers
    every field. The executable half is a ``SelectedBlock`` that binds trusted
    code to this exact identity. Two recipes with the same full operator,
    such as X and HZH, are different selections with different identities
    and inventories.

    Attributes:
        signature: Declared interface, equal to the Program's signature of the same name.
        semantics: The promised action and its approximation.
        implementation: Versioned implementation Source.
        choice: Recipe label within the implementation. A supplied native circuit gets a fresh unique label.
        decomposition: Exact ordered primitive recipe, or None when a law describes the cost.
        cost_law: Source of the selected size or CX law, or None when no law exists.
        cost_parameters: Exact law arguments for this selection.
        cost_context: Scope and limits of that law, carried into resource quantities.
        construction_work: Size-law work units to build the native definition, or None when unknown.
        controlled: Whether this selection adds one coherent control to its base.
        adjoint: Whether this selection is the adjoint of its base.
        base_selection_id: Identity of the base selection for a control or adjoint.
        coefficient_selection_id: Identity of the SELECT whose coefficients this PREP prepares.
        blocker: Reason the selection cannot execute, for a law-only or unsupported leaf.
        resource_laws: Per-call ResourceLaw summaries.
        workspace: Per-invocation workspace byte declarations by location. An empty tuple means unknown, not zero.
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
            # Transforms keep the base recipe's target numbering; their outer
            # control is represented by controlled, not by rewriting primitives.
            width = sum(widths) - int(self.controlled)
            if any(index >= width for operation in self.decomposition for index in operation.qubits):
                raise ValueError("selected primitive target is outside its register width")
        return self


class PauliEncoding(Record):
    """A named PREP/SELECT/PREP-inverse subgraph and its index-zero projection.

    Normalization, physical A, and epsilon come from the checked selected SELECT;
    this record does not own another independent copy of those quantities.

    Attributes:
        root: Definition ID of the Sequence that forms the encoding.
        select: Signature name of the signed Pauli SELECT.
        prepare: Signature name of the coefficient PREP, or None for a single-term encoding.
        unprepare: Signature name of the adjoint of that PREP, or None for a single-term encoding.
    """

    root: Text
    select: Text
    prepare: Text | None = None
    unprepare: Text | None = None


class SelectedKernel(Record):
    """Selected host algorithm declaration. Native input access is bound by the Method's factory.

    A host kernel is a classical computation that a Program runs at a host
    ClassicalStage. Its declaration is portable, while the callable is bound
    separately by ``BoundKernel`` from the Method's live factory.
    construction_work and invocation_work are declared size
    units charged to the Run, not CPU instructions. Resource laws state
    whether these bound the total numerical work. Absent laws leave the total
    unknown. Workspace declarations keep known bytes and unknown components
    separately.

    Attributes:
        name: Kernel name, referenced by the Program's host ClassicalStage.
        implementation: Versioned Source of the host algorithm.
        inputs: Identities of the admitted inputs the kernel reads.
        scalars: Ordered labels of the scalar statistics it returns.
        scalar_frames: Frame of each scalar, ``physical`` for physical units, ``unit`` for a normalized direction, or ``encoded_branch`` for a branch mass or norm in the encoded frame that the Method declares.
        outputs: Declared named arrays it may return.
        resource_laws: Fixed ``classical_work`` laws of the untransformed kernel.
        workspace: Per-invocation host workspace declarations. An entry with unknown bytes keeps that component unknown.
        construction_work: Size units charged when the kernel is bound at preparation.
        invocation_work: Size units charged for each invocation.
        dependencies: Distribution names, such as ``numpy``, whose installed versions each host preparation records.
        application_bytes: JSON size bound, from the run journal's size owner, of the application receipts that every invocation returns when the Method fixes them at selection. The Run reserves it with the declared outputs. Completion admission subtracts the smaller of this value and the returned receipts' bound before checking the variable metadata allowance, so undeclared receipt bytes still spend that allowance. Zero declares none.
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
    """One directly consumable Program and its selected definitions; not a Plan.

    JSON preserves sharing in Program.definitions. Native access is rebound by
    exact selected-record identity at explicit lowering, never from persisted code.

    The validator requires exactly one selected definition per Program
    signature, with an identical signature, so every BlockCall resolves to
    one selection. A host stage must name its exact selected kernel. A Pauli
    encoding with an index register must be the ordered PREP, SELECT,
    PREP-inverse Sequence on that register, where the PREP was selected from
    this SELECT's coefficients and the inverse is the adjoint of that same
    PREP. A single-term encoding has no index register and is SELECT alone,
    with no PREP. That structure is what makes ``<0|U|0> = A/alpha`` hold for
    the subgraph.

    Attributes:
        program: The Program that every consumer reads.
        selections: One SelectedDefinition per Program signature.
        encodings: Pauli encodings checked as PREP, SELECT and PREP-inverse subgraphs.
        kernels: Host kernel declarations, each named by one host ClassicalStage.
    """

    schema_version: Literal[2] = 2
    program: Program
    selections: tuple[SelectedDefinition, ...]
    encodings: tuple[PauliEncoding, ...] = ()
    kernels: tuple[SelectedKernel, ...] = ()

    @model_validator(mode="after")
    def _selection_bindings(self):
        """Require exact Program bindings and the ordered PREP/SELECT/inverse
        coefficient-register relation.
        """
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
        """Derive (A, alpha, index convention, epsilon) from one checked subgraph.

        The relation is the LCU block encoding of An, Childs and Lin,
        arXiv:2312.03916v2, Appendix A.3, Lemma 24, Eq. (178), in the
        real-coefficient form where both preparation oracles equal PREP and
        sign(c_j) sits in SELECT. With ``alpha = sum_j |c_j|`` the index-zero
        block is ``A/alpha``. This is an exact block encoding in the sense of
        their Definition 23 (Appendix A.2).
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
