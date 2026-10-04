"""QLS as actual selected preparation, query, projector and readout bodies.

The inverse route runs the real-part QSVT pass ([GSLW] arXiv:1806.01838v1,
Corollary 18) of the rescaled odd polynomial on the selected encoding of
``A/alpha``. A non-Hermitian ``A`` enters through the Hermitian dilation
``[[0, A], [A^dagger, 0]] / alpha`` with ``b`` in the upper half, and the
solution is read from the lower half. The shortcut routes apply the even
kernel-reflection polynomial to Dalzell's ``G_t = Q_b' A_t``
(arXiv:2406.12086v2, Eqs. (8)-(11)), directly or through its Hermitian
dilation. Their circuits follow the pattern of Dalzell App. A: ``A_t`` by
one controlled query plus a predicate-controlled ``RY`` with amplitude
``1/t`` (App. A.2, Fig. 4), ``b'`` with one extra coordinate qubit
(App. A.3 and A.6), and ``Q_b'`` by unpreparing ``b'``, flagging the
all-zero state and preparing again (App. A.1 and A.4). Projector phases
use ``e^{i phi (2 Pi - 1)}`` of [MRTC] arXiv:2105.02859v5, Eq. (27).
Resource estimation, inspection and native lowering all read the one
Program built here.
"""

from dataclasses import dataclass
from math import sqrt
import numpy as np
from nwqlib.blocks.encoding import construct_block_encoding
from nwqlib.blocks.records import (
    BlockSemantics,
    Primitive,
    SelectedConstruction,
    SelectedDefinition,
)
from nwqlib.blocks.selection import SelectedBlock, select_preparation, select_pauli_parity
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.core.planning import Plan, Experiment, ObservationPoint, ObservationSpec, ReadoutDetails
from nwqlib.core.records import Float64, InputRef, Source
from nwqlib.evidence import Evidence
from nwqlib.ir import (
    Allocate,
    Argument,
    Binding,
    BlockCall,
    BlockSignature,
    ClassicalValue,
    Constant,
    Definition,
    ExprRef,
    Expression,
    Measure,
    MeasurementBatch,
    MetadataRef,
    Parameter,
    PortMap,
    QuantumPort,
    Register,
    Release,
    Sequence,
    Setting,
)
from nwqlib.ir.validation import admitted_program
from nwqlib.operators.access import _check_bytes
from nwqlib.problems.inputs import PhysicalScale, compose_recovery
from nwqlib.problems.records import (
    Solution,
    StateVector,
    NormSquared,
    Samples,
    QuadraticForm,
    NormalizedExpectation,
)
from nwqlib.resources.records import ResourceLaw
from nwqlib.subroutines._mcx_counts import MCX_CX_BY_CONTROLS
from nwqlib._quantum_readout import (
    PROJECTED_KERNEL,
    PROJECTED_MASS_EXCLUSIONS,
    PROJECTED_MOMENTS,
    ReadoutSetting,
    pair_float,
    pair_ratio,
    physical_moment,
    projected_parameters,
    projected_populations,
    projected_statistics,
    qwc_groups,
    recover_scaled_pair,
    reduce_sample_arrays,
    validate_readout_layout,
    weighted_group_moments,
)
from .primary_records import (
    METHOD,
    QLSAnalysis,
    QLSGroupMoments,
    QLSProjectedMoments,
    QLSSamples,
    validate_selection,
)


@dataclass(frozen=True)
class BaseEncoding:
    """Selected encoding of ``A/alpha`` bound to its exact inputs for native construction.

    ``validate`` rejects a payload whose operator, dense source or SVD no
    longer belongs to the selected original input, so a native query is
    never built from different data than the Plan selected.
    """

    selected: SelectedBlock
    original: object
    encoded: object
    svd: object
    max_bytes: int
    max_work: int

    def validate(self):
        if self.selected.record.semantics.input != self.encoded.reference:
            raise ValueError("QLS encoding payload differs from its selected actual operator")
        if (
            self.selected.record.implementation.name == "encoding.planned"
            and self.selected._payload.implementation == "dense_dilation"
            and self.selected._payload.source is not self.encoded._data
        ):
            raise ValueError("QLS dense completion differs from its exact selected input data")
        if self.svd is not None:
            self.svd.require_source(self.original)


def _native_encoding(base):
    """Construct the native base encoding once per Run.

    A dense-dilation selection with a selected original SVD reuses those
    frames. For a padded system the dummy coordinates get identity frames
    and singular value ``alpha``, which is the exact SVD of the padded
    ``alpha * I`` block, so no padded decomposition is computed.
    """
    base.validate()
    if base.svd is None or base.selected.record.implementation.name == "encoding.native":
        return construct_block_encoding(base.selected, max_bytes=base.max_bytes, max_work=base.max_work)
    from nwqlib.subroutines.block_encoding.core import _dense_dilation_encoding, _admit_dense_input

    selected = base.selected._payload
    if selected.implementation != "dense_dilation":
        return construct_block_encoding(base.selected, max_bytes=base.max_bytes, max_work=base.max_work)
    d, padded = base.original.basis.dimension, base.encoded.basis.dimension
    _admit_dense_input(selected.source, max_bytes=base.max_bytes, max_work=base.max_work,
        construction=True, selected_svd=True)
    # The padded frames built below are two complex p x p identity-extended
    # frames (32 p^2 bytes) and the padded singular values (8 p bytes). The
    # source passed the power-of-two admission, so its dimension is p, and
    # the completion law admitted above (384 p^2 + 32 p bytes) already
    # covers these arrays.
    cache = base.svd
    if d == padded:
        frames = cache.left, cache.singular, cache.right_h
    else:
        left = np.eye(padded, dtype=complex)
        right = np.eye(padded, dtype=complex)
        singular = np.full(padded, selected.alpha)
        left[:d, :d], right[:d, :d], singular[:d] = cache.left, cache.right_h, cache.singular
        frames = left, singular, right
    return _dense_dilation_encoding(
        selected.source,
        requested_implementation=selected.requested_implementation,
        normalization=selected.alpha,
        planned_error_bound=selected.error_bound,
        selected_svd=frames,
    )


@dataclass(frozen=True)
class Query:
    """Payload of one query leaf of the QLS Program.

    Attributes:
        base: The selected encoding of ``A/alpha`` or the selected RHS
            preparation that the leaf calls.
        control_count: Number of valued controls added to that call. The
            control state and the adjoint flag are call arguments.
        route: Dense control route of those controls
            (``qiskit_compat.controlled``), the QLS Method's
            ``dense_control_route`` for a planned dense-dilation encoding and
            ``gatewise`` otherwise.
    """

    base: BaseEncoding | SelectedBlock
    control_count: int
    route: str = "gatewise"


def _query_circuit(block, arguments, method_context):
    """Realize the bound base encoding with its actual valued controls and adjoint flag."""
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.qiskit_compat import controlled, inverse_realized_gate

    payload = block._payload
    arguments = dict(arguments)
    if set(arguments) != {"adjoint", "control_state"}:
        raise ValueError("QLS query requires its actual adjoint/control-state arguments")
    if isinstance(payload.base, BaseEncoding):
        payload.base.validate()
        base_id = payload.base.selected.record.content_id
    else:
        base_id = payload.base.record.content_id
    if block.record.choice != base_id:
        raise ValueError("QLS native query differs from its exact base selection")
    selected = payload.base.selected if isinstance(payload.base, BaseEncoding) else payload.base
    expected_width = (
        sum(port.width for port in selected.record.signature.quantum) + payload.control_count
    )
    if (
        sum(port.width for port in block.record.signature.quantum) != expected_width
        or block.record.semantics.base_semantics != selected.record.semantics
    ):
        raise ValueError("QLS native query differs from its base semantic contract or ports")
    state, adjoint = arguments["control_state"], arguments["adjoint"]
    if (
        type(adjoint) is not int
        or adjoint not in (0, 1)
        or type(state) is not int
        or not 0 <= state < 1 << payload.control_count
    ):
        raise ValueError("QLS query has invalid valued controls or adjoint flag")
    # Cache the realized base gate once per run, then derive its inverse
    # without repeating base synthesis or discarding its controlled phase.
    context = method_context(dict)
    gates = context.setdefault("qls_base_gates", {})
    if base_id not in gates:
        if isinstance(payload.base, BaseEncoding):
            circuit = _native_encoding(payload.base).circuit
        else:
            circuit = payload.base._constructor(payload.base, {}, method_context)
        gates[base_id] = circuit.to_gate()
    gate = gates[base_id]
    if adjoint:
        native_ucg = payload.control_count == 0
        key = (base_id, "adjoint", native_ucg)
        if key not in gates:
            gates[key] = inverse_realized_gate(gate, native_ucg=native_ucg)
        gate = gates[key]
    if payload.control_count:
        gate = controlled(gate, payload.control_count, ctrl_state=state, route=payload.route)
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, range(gate.num_qubits))
    return circuit


def _gate_circuit(block, arguments, method_context):
    """Realize one primitive leaf as a circuit on its ports, controls first.

    The payload is ``(kind, control_count, control_state)``. ``kind`` is
    ``x``, ``h``, ``z``, ``ry``, ``rz`` or ``global_phase``. Rotations and
    the global phase take their angle from the call's ``angle`` argument.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate, XGate, ZGate, RYGate, RZGate
    from nwqlib.subroutines.qiskit_compat import controlled

    kind, controls, values = block._payload
    arguments = dict(arguments)
    width = len(block.record.signature.quantum)
    circuit = QuantumCircuit(width)
    if kind == "global_phase":
        if set(arguments) != {"angle"}:
            raise ValueError("QLS global phase requires its selected angle")
        circuit.global_phase = arguments["angle"]
        return circuit
    if kind in {"ry", "rz"}:
        if set(arguments) != {"angle"}:
            raise ValueError("QLS controlled rotation requires its selected angle")
        gate = (RYGate if kind == "ry" else RZGate)(arguments["angle"])
    else:
        if arguments:
            raise ValueError("fixed QLS primitive has no scalar arguments")
        gate = {"x": XGate, "h": HGate, "z": ZGate}[kind]()
    if controls:
        gate = controlled(gate, controls, ctrl_state=values)
    circuit.append(gate, range(width))
    return circuit


def _primitive_cx_law(kind, control_count):
    """Return ``(cx, exact)`` for one primitive leaf, or ``None`` when no CX law applies.

    ``cx`` is the CX count of the leaf and ``exact`` tells whether it is
    the leaf's exact inventory or an estimate. An uncontrolled primitive is
    a one-qubit gate or a global phase and uses no CX. A singly controlled
    X is exactly one CX. A multi-controlled X with ``2 <= k <= 64`` controls
    takes the CX count of Qiskit 2.5.2's synthesis of that gate from
    ``MCX_CX_BY_CONTROLS``, recorded as an estimate because another
    Qiskit version can synthesize it differently. A larger k, and controlled
    H, Z, RY, RZ and global phase, have no CX law here, so a CX fold that
    meets one reports the total as unknown.
    """
    if not control_count:
        return 0, True
    if kind != "x":
        return None
    if control_count == 1:
        return 1, True
    if control_count > len(MCX_CX_BY_CONTROLS):
        return None
    return MCX_CX_BY_CONTROLS[control_count - 1], False


def _leaf(
    name,
    *,
    basis,
    reference,
    width,
    kind,
    payload,
    constructor,
    parameters=(),
    decomposition=None,
    laws=(),
    cost_parameters=(),
    choice=None,
    base_semantics=None,
    construction_work=None,
):
    """Bind one QLS primitive with its port order, phase contract and scoped cost metadata."""
    source = Source(
        name="qls." + kind,
        version="2",
        domain="actual selected QLS native operation",
        reference="nwqlib.algorithms.qls.quantum; actual ports/arguments are in the shared Program",
    )
    signature = BlockSignature(
        name=name,
        target=source,
        quantum=tuple(QuantumPort(name=f"q{i}", width=1) for i in range(width)),
        parameters=parameters,
    )
    record = SelectedDefinition(
        signature=signature,
        implementation=source,
        semantics=BlockSemantics(
            kind="unknown" if base_semantics is None else "unitary_transform",
            input=reference,
            basis=basis,
            base_semantics=base_semantics,
            relation="actual selected " + kind,
            input_projector="selected valued controls in actual port order",
            output_projector="same control values; selected target action",
            success="unitary operation before solution readout",
            workspace=0,
            restoration="no unreported work qubits",
            epsilon=None,
            approximation_metric="native unitary action",
            approximation_evidence="native arithmetic error unknown",
            inverse_legal=False,
            control_legal=False,
            phase="global phase and its inverse sign stay in the controlled action",
        ),
        choice=name if choice is None else choice,
        decomposition=decomposition,
        cost_law=None,
        cost_parameters=cost_parameters,
        resource_laws=laws,
        cost_context="selected native operation; hardware routing and SDK workspace unknown",
        construction_work=max(1, width) * 16 if construction_work is None else construction_work,
    )
    return SelectedBlock.bind(record, payload=payload, constructor=constructor)


def _query_leaf(base, *, kind, controls, basis, reference, route="gatewise"):
    """Select a parameterized base query and reuse only applicable per-query resource evidence.

    Only an uncontrolled query of the encoding of ``A/alpha`` carries a CX
    law. It is the planned family's ``block_encoding_per_query_cx``, or the
    uncontrolled, non-adjoint CX law recorded by a supplied native encoding,
    and the adjoint query has the same inventory. Controlled queries and
    RHS-preparation queries carry no CX law, because NWQLib has no model of
    their controlled synthesis.
    """
    selected = base.selected if isinstance(base, BaseEncoding) else base
    width = sum(port.width for port in selected.record.signature.quantum) + controls
    name = f"{kind}_query_c{controls}"
    parameters = (
        Parameter(name="adjoint", domain="integer", lower=0, upper=1),
        Parameter(name="control_state", domain="integer", lower=0, upper=(1 << controls) - 1),
    )
    laws = ()
    if isinstance(base, BaseEncoding) and not controls:
        from nwqlib.backends.resources import block_encoding_per_query_cx

        plan = selected._payload
        child = next(
            (
                law
                for law in selected.record.resource_laws
                if law.metric == "cx"
                and law.basis == "cx"
                and not law.controlled
                and not law.adjoint
            ),
            None,
        )
        query = (
            block_encoding_per_query_cx(
                plan.detail, implementation=plan.implementation, system_qubits=plan.system_qubits
            )
            if selected.record.implementation.name == "encoding.planned"
            else None
            if child is None
            else child.value.value
        )
        if query is not None:
            laws = (
                ResourceLaw(
                    metric="cx",
                    basis="cx",
                    value=Float64(value=float(query)),
                    interpretation="estimate",
                    unbound_parameters=("adjoint", "control_state"),
                    evidence=Evidence(kind="numerical_estimate", source=METHOD)
                    if child is None
                    else child.evidence,
                    assumptions=(
                        "existing selected family per-query synthesis model; inverse has the same inventory",
                    )
                    + (() if child is None else child.assumptions),
                ),
            )
    return _leaf(
        name,
        basis=basis,
        reference=selected.record.semantics.input,
        width=width,
        kind=kind + "_query",
        payload=Query(base, controls, route),
        constructor=_query_circuit,
        parameters=parameters,
        laws=laws,
        choice=selected.record.content_id,
        base_semantics=selected.record.semantics,
        construction_work=selected.record.construction_work,
    )


def _observable_terms(problem, output, *, max_bytes):
    """Return the observable of a scalar output as real Pauli terms ``((label, c), ...)``.

    Vector and sample outputs return ``()``. Finite Pauli input is used as
    stored. A dense observable ``O`` is embedded as ``P O P`` in the padded
    coordinate register, where ``P`` projects onto the original
    coordinates, and decomposed by the Pauli block transform
    ``pauli_coefficients``, which omits only exact zeros. The readout
    evaluates these terms on the coordinate qubits. A Hermitian
    observable has real Pauli coefficients, so a nonzero imaginary part is
    rejected. Planning uses these rows for the exact reduction's bound
    terms and the sampled groups only; the reconstruction keeps the
    reference or table of ``_observable_fields``.
    """
    if not isinstance(output, (QuadraticForm, NormalizedExpectation)):
        return ()
    operator = output.observable
    if "pauli_terms" in operator.manifest.access:
        rows = operator.pauli_terms().labels(max_bytes=max_bytes)
    else:
        from nwqlib.operators._pauli import pauli_coefficients

        if operator.reference.representation != "dense":
            raise ValueError("quantum QLS observable requires dense or finite Pauli access")
        d = problem.dimension
        padded = 1 << max(1, (d - 1).bit_length())
        # 64 p^2 bytes: the padded complex matrix, the two live levels of the
        # I/X/Y/Z block transform and its half-sum temporaries.
        _check_bytes(64 * padded * padded, max_bytes, "QLS P O P observable conversion")
        matrix = np.zeros((padded, padded), dtype=complex)
        matrix[:d, :d] = operator.dense_array()
        rows = pauli_coefficients(matrix)
    if any(c.imag != 0 for _, c in rows):
        raise ValueError("QLS observable Pauli coefficients must be real")
    return tuple((label, float(c.real)) for label, c in rows)


def _observable_fields(output, terms):
    """Return the reconstruction fields that let analysis rebuild the observable rows.

    A finite Pauli observable is kept by reference (``observable_input``);
    its packed table stays with the observable input. A dense observable's
    ``pauli_coefficients`` table has distinct labels: its identity
    coefficient is stored as ``observable_identity``, and its non-identity
    labels with nonzero coefficient, in table order, as
    ``observable_labels`` with their coefficients in a ``FrozenArray``.
    """
    if not isinstance(output, (QuadraticForm, NormalizedExpectation)):
        return {}
    if "pauli_terms" in output.observable.manifest.access:
        return dict(observable_input=output.observable.reference)
    identity = [c for label, c in terms if all(axis == "I" for axis in label)]
    if len(identity) > 1:
        raise ValueError("a dense observable's Pauli table has one identity row")
    kept = [(label, c) for label, c in terms if c != 0 and any(axis != "I" for axis in label)]
    return dict(
        observable_identity=identity[0] if identity else 0.0,
        observable_labels=tuple(label for label, _ in kept),
        observable_coefficients=np.array([c for _, c in kept], dtype=np.float64),
    )


def _canonical_terms(terms, width):
    """Return observable rows as the reduction binds them: the summed identity first, then the
    nonzero non-identity rows in table order.

    Planning and the stored-observable rebuild both pass through this one
    form, so the reduction parameters of a Plan and of its reload agree.
    """
    identity = sum(c for label, c in terms if all(axis == "I" for axis in label))
    head = [("I" * width, float(identity))] if identity != 0 else []
    return tuple(head + [(label, float(c)) for label, c in terms
                         if c != 0 and any(axis != "I" for axis in label)])


def _stored_terms(rec, output, *, max_bytes):
    """Return the canonical rows of the stored scalar observable, without a new decomposition.

    A Pauli observable is read from its referenced packed table (one
    ``labels()`` expansion) and a dense observable from its stored identity
    coefficient, labels and coefficients. NormSquared and non-scalar outputs
    have no rows.
    """
    if not isinstance(output, (QuadraticForm, NormalizedExpectation)):
        return ()
    width = len(rec.coordinates)
    if rec.observable_input is None:
        coefficients = () if rec.observable_coefficients is None else rec.observable_coefficients.array.tolist()
        return _canonical_terms(
            (("I" * width, rec.observable_identity), *zip(rec.observable_labels, coefficients, strict=True)), width)
    operator = output.observable
    if operator.reference != rec.observable_input:
        raise ValueError("QLS observable differs from its selected reference")
    rows = operator.pauli_terms().labels(max_bytes=max_bytes)
    if any(c.imag != 0 for _, c in rows):
        raise ValueError("QLS observable Pauli coefficients must be real")
    return _canonical_terms(tuple((label, c.real) for label, c in rows), width)


def _readout(method, output, terms, rec, *, shots):
    """Select the QLS readout acquisitions and charge QWC grouping to ``max_work``.

    Returns ``(settings, groups, comparisons)``. A vector output reads its
    amplitudes and Samples one counts setting. With ``shots=None`` a scalar
    output has one exact ``projected_moments`` reduction. With shots,
    NormSquared measures its physical mass; a Pauli observable measures one
    counts setting per first-fit qubit-wise commuting group of its nonzero
    non-identity labels, whose label is the group's accumulated basis. A
    padded normalized output adds the unrotated ``physical_mass`` setting
    unless a group basis without X or Y already measures every coordinate
    unrotated, in which case that group supplies the physical-prefix mass.
    An observable without non-identity terms has one identity group for its
    selected population. ``groups`` lists each group setting's member
    labels. The grouping comparisons are admitted tile by tile against
    ``max_work``, which QLS applies to each planning phase, such as the
    polynomial fit and the encoding construction, separately.
    ``comparisons`` is the actual evaluated count.
    """
    identity = "I" * len(rec.coordinates) or "I"
    projection = rec.original_dimension != 1 << len(rec.coordinates)
    if isinstance(output, (Solution, StateVector, Samples)):
        return (ReadoutSetting(name="setting_0", label=identity),), (), 0
    if shots is None:
        return (ReadoutSetting(name=PROJECTED_MOMENTS, label=identity),), (), 0
    if isinstance(output, NormSquared):
        return (ReadoutSetting(name="physical_mass", label=identity, physical_projection=projection),), (), 0
    groups, bases, comparisons = qwc_groups(
        terms, len(rec.coordinates), max_comparisons=method.max_work, max_bytes=method.max_bytes,
        limit_name="QLS.max_work")
    if not groups:
        groups, bases = ((),), (identity,)
    settings = tuple(ReadoutSetting(name=f"group_{index}", label=basis) for index, basis in enumerate(bases))
    if (isinstance(output, NormalizedExpectation) and projection
            and all(set(basis) & set("XY") for basis in bases)):
        settings += (ReadoutSetting(name="physical_mass", label=identity, physical_projection=True),)
    return settings, groups, comparisons


def _layout(rec, solver):
    """Return the circuit width, coordinate qubits and readout predicates.

    Wire order: 0 is the real-pass pair qubit, 1 the QSP signal qubit, then
    the encoding ancillas (and, for shortcuts, the ``A_t`` flag and the
    ``Q_b'`` flag), then the original coordinate qubits, then the dilation
    coordinate of a non-Hermitian inverse, the Dalzell augmented coordinate
    and the ``G_t`` dilation qubit when present. Algorithm success requires
    the pair, signal and every ancilla at zero, the augmented coordinate at
    zero (Dalzell arXiv:2406.12086v2, Algorithm 1 step 3), and the ``G_t``
    dilation qubit at one. The physical slice of a non-Hermitian inverse
    also requires its dilation coordinate at one, the lower half that holds
    the solution.
    """
    inverse = solver == "qsvt_inverse"
    native = (rec.padded_dimension - 1).bit_length()
    system = native + int(rec.embedding != "none")
    ancillas = rec.encoding_ancillas + (0 if inverse else 2)
    offset = ancillas + 2
    width = offset + system + (0 if inverse else 1 + int(solver == "shortcut_dilation"))
    success = tuple((bit, 0) for bit in range(offset))
    if not inverse:
        success += ((offset + system, 0),)
        if solver == "shortcut_dilation":
            success += ((width - 1, 1),)
    conditions = ((offset + system - 1, 1),) if rec.embedding != "none" else ()
    return dict(
        width=width,
        coordinates=tuple(range(offset, offset + native)),
        success=success,
        conditions=conditions,
    )


def _counts_bits(rec, setting):
    """Physical qubits measured into a setting's compact classical prefix.

    Success and condition bits come first in their declared order. A group
    then measures the coordinates in its basis support, in coordinate order.
    Samples, physical_mass and the first unrotated group of padded input
    measure every coordinate. That unrotated group supplies the prefix mass.
    First-fit QWC grouping has at most one group with an I/Z-only basis.
    The Program declares the largest such prefix once. Bits beyond this
    setting's prefix stay at their initial classical zero value.
    """
    selectors = tuple(bit for bit, _ in rec.success + rec.conditions)
    padded = rec.original_dimension != 1 << len(rec.coordinates)
    full = (setting.name in {"setting_0", "physical_mass"}
            or (padded and not set(setting.label) & set("XY")))
    coordinates = (rec.coordinates if full else tuple(
        bit for bit, axis in zip(rec.coordinates, reversed(setting.label), strict=True)
        if axis != "I"))
    return selectors + tuple(coordinates)


def _counts_width(rec):
    """Width of the shared classical layout, the largest measured prefix."""
    return max(len(_counts_bits(rec, setting)) for setting in rec.settings)


def _counts_program_bytes(rec):
    """Extra logical Program allowance for distinct measurement maps.

    One 1024-byte qualified record allowance covers each distinct Measure,
    each layout Sequence and each child reference. The existing Program
    allowance already covers its shared classical declarations and wires.
    This is a CPython record allowance, not a process RSS bound.
    """
    layouts = set(_counts_bits(rec, setting) for setting in rec.settings)
    pairs = set(pair for observed in layouts for pair in enumerate(observed))
    return 1024 * (len(pairs) + len(layouts) + sum(map(len, layouts)))


def _program(method, problem, output, shots, rec, base, preparation, terms=None):
    """Emit the one shared actual composition, including every valued projector.

    The body prepares the input (``b`` for the inverse, ``|e_n>`` and the
    dilation basis state for the shortcuts), applies the real-part QSVT
    pass with the selected reflection-convention phases, and ends with the
    readout. Every child is a named call with explicit ports and
    arguments, so resource folding, inspection, lowering and archive
    comparison all read the same Program. The wire map is in ``_layout``.

    With ``shots=None``, a scalar observable is evaluated from one
    simulation of the selected coherent circuit. The readout fixes the
    method's success and physical-condition bits and excludes dummy
    coordinates. A Pauli observable with L terms then needs O(LN) classical
    work on the N-coordinate encoded system. The normalized expectation
    divides the projected quadratic moment by the physical-slice mass. A
    physical quadratic form also applies the method's recovery scale.
    Exact readout has no finite-shot error, while circuit, observable-action
    and normalization rounding remain separate error contributions.
    Normalized outputs are unavailable at zero physical mass. That one
    experiment is the coherent body without parity tails and one
    ``projected_moments`` reduction at its end, whose parameters bind the
    success bits, condition bits, coordinate order, original dimension and
    stored observable rows (``terms``, or the stored observable when None).

    Sampled observable readout groups qubit-wise-commuting Pauli terms into
    shared local bases. ``shots`` is the number of shots per selected
    group. Terms in a group share outcomes, so uncertainty calculations use
    the weighted group outcome and its covariance. Grouping reduces the
    number of settings. Its variance at fixed total shots depends on the
    state, coefficients and allocation. A padded normalized output also
    acquires its physical-coordinate mass in an unrotated basis. QLS admits
    each QWC candidate tile against ``max_work``, which caps each QLS
    planning phase separately, and separately admits grouping arrays and
    Program payloads. The shared coherent body is stored once, and
    ``shots`` counts shots per selected group. Each group setting rotates X
    coordinates by H and Y coordinates by S† then H, one selected one-site
    block per coordinate and axis shared by all groups, with no parity
    network, and measures its support and valued selectors into a compact
    prefix of one shared classical layout. Samples, physical_mass and the
    unrotated group supplying a padded prefix mass measure all coordinates.
    Unmeasured suffix bits stay at their initial zero value.
    """
    from nwqlib.amplitudes import AmplitudeReadout
    from .method import _array_output
    from nwqlib.subroutines.qsp.phases import wx_phases_to_reflection

    definitions, expressions, blocks = {}, {}, {}
    basis, reference = base.encoded.basis, problem.A.reference
    a = rec.encoding_ancillas
    n = (rec.padded_dimension - 1).bit_length()
    offset = rec.coordinates[0]
    original_ancillas = tuple(range(2, 2 + a))
    original_system = rec.coordinates
    encoded_bits = original_ancillas + original_system
    pair, signal = 0, 1
    ancillas = tuple(range(2, offset))
    embedding = offset + n if rec.embedding != "none" else None
    system = original_system + (() if embedding is None else (embedding,))
    augment = offset + len(system) if method.solver != "qsvt_inverse" else None
    dilation = rec.width - 1 if method.solver == "shortcut_dilation" else None

    # The helpers below record Program nodes in first-definition order, which
    # fixes the Program's content identity. Reordering calls changes that
    # identity and makes archived Plans fail their load-time comparison.

    def define(name, node):
        """Record ``node`` under ``name``. A name can never denote two different nodes."""
        if name in definitions and definitions[name] != node:
            raise ValueError("QLS definition name reused for a different selected operation")
        definitions[name] = node
        return name

    def sequence(name, children):
        """Define a named ``Sequence`` of already defined children."""
        return define(name, Sequence(children=tuple(children)))

    def expression(name, value):
        """Define a named constant argument (int, or float stored as ``Float64``)."""
        node = Constant(value=value if type(value) is int else Float64(value=float(value)))
        if name in expressions and expressions[name] != node:
            raise ValueError("QLS parameter expression changed under a selected name")
        expressions[name] = node
        return ExprRef(expression=name)

    def call(name, block, bits, arguments=()):
        """Define a call of ``block`` on wires ``bits`` in its port order, controls first.

        Each scalar argument becomes the constant expression ``{name}_{key}``.
        """
        if len(bits) != len(block.record.signature.quantum):
            raise ValueError("QLS call ports differ from the actual selected register width")
        blocks[block.record.signature.name] = block
        return define(
            name,
            BlockCall(
                signature=block.record.signature.name,
                ports=tuple(
                    PortMap(port=port.name, wire=f"bit_{bit}")
                    for port, bit in zip(block.record.signature.quantum, bits, strict=True)
                ),
                arguments=tuple(
                    Argument(parameter=key, value=expression(name + "_" + key, value))
                    for key, value in arguments
                ),
            ),
        )

    gate_blocks = {}

    def gate(name, kind, target, *, controls=(), angle=None):
        """Call a primitive ``kind`` on ``target`` with valued controls.

        ``controls`` is a tuple of ``(wire, value)`` pairs, and the control
        state packs value ``i`` into bit ``i``. One leaf is selected per
        ``(kind, control count, control state)``, so the Program lists each
        distinct primitive once, and ``angle`` is bound per call. The leaf
        carries the CX law of ``_primitive_cx_law``.
        """
        values = sum(value << i for i, (_, value) in enumerate(controls))
        key = kind, len(controls), values
        if key not in gate_blocks:
            width = len(controls) + len(target)
            parameters = (Parameter(name="angle", domain="real"),) if angle is not None else ()
            decomposition = (
                (Primitive(gate=kind, qubits=(0,)),)
                if kind in {"x", "h", "z"} and not controls
                else None
            )
            costs = (
                Binding(parameter="control_count", value=len(controls)),
                Binding(parameter="control_state", value=values),
            )
            laws = ()
            # The CX law belongs to the leaf. The Program supplies how often
            # each leaf is called, including the two pair CNOTs of every
            # projector phase.
            cx_law = _primitive_cx_law(kind, len(controls))
            if cx_law is not None:
                cx, exact = cx_law
                laws = (
                    ResourceLaw(
                        metric="cx",
                        basis="cx",
                        value=cx,
                        interpretation="exact" if exact else "estimate",
                        bindings=costs,
                        unbound_parameters=tuple(parameter.name for parameter in parameters),
                        evidence=Evidence(
                            kind="proved_relation" if exact else "numerical_estimate", source=METHOD
                        ),
                        assumptions=()
                        if exact
                        else (
                            # Stored text, part of Plan content identity.
                            "Qiskit 2.5.2 multi-controlled X synthesis without ancillas, "
                            "lowered to cx,u at optimization level 0 before routing; "
                            "another Qiskit version can differ",
                        ),
                    ),
                )
            gate_blocks[key] = _leaf(
                f"{kind}_c{len(controls)}_v{values}",
                basis=basis,
                reference=reference,
                width=width,
                kind=kind,
                payload=key,
                constructor=_gate_circuit,
                parameters=parameters,
                decomposition=decomposition,
                cost_parameters=costs,
                laws=laws,
            )
        return call(
            name,
            gate_blocks[key],
            tuple(bit for bit, _ in controls) + tuple(target),
            () if angle is None else (("angle", float(angle)),),
        )

    query_blocks = {}
    # The Method's dense control route applies to a planned dense-dilation
    # encoding. A supplied encoding, whose matrix is not known and which may
    # hold operations without one, and the RHS preparation are controlled
    # gate-wise.
    selected_base = base.selected if isinstance(base, BaseEncoding) else base
    implementation = selected_base.record.implementation
    route = (method.dense_control_route if implementation.domain == "dense_dilation"
             and implementation.name != "encoding.native" else "gatewise")

    def query(name, kind, bits, *, controls=(), adjoint=False):
        """Call the encoding of ``A/alpha`` (``original_A``) or the RHS preparation (``rhs``).

        One leaf is selected per ``(kind, control count)``. The control
        state and the adjoint flag are call arguments.
        """
        key = kind, len(controls)
        if key not in query_blocks:
            query_blocks[key] = _query_leaf(
                base if kind == "original_A" else preparation,
                kind=kind,
                controls=len(controls),
                basis=basis,
                reference=reference if kind == "original_A" else preparation.record.semantics.input,
                route=route if kind == "original_A" else "gatewise",
            )
        values = sum(value << i for i, (_, value) in enumerate(controls))
        return call(
            name,
            query_blocks[key],
            tuple(bit for bit, _ in controls) + tuple(bits),
            (("adjoint", int(adjoint)), ("control_state", values)),
        )

    def original_query(name, *, controls=(), adjoint=False):
        """Query the encoding of ``A/alpha``, or of its Hermitian dilation.

        Without an embedding this is one ``original_A`` call. For the
        dilation of a non-Hermitian inverse, with ``U`` the selected encoding
        of ``A/alpha``, the calls ``U^dagger`` controlled on the embedding
        qubit at 0, then ``U`` controlled on it at 1, then an X on it give
        ``W = |0><1| (x) U + |1><0| (x) U^dagger``. With the embedding qubit
        as the highest coordinate bit, the block of ``W`` is
        ``[[0, A], [A^dagger, 0]] / alpha``. ``W`` is Hermitian as a full
        unitary, so its adjoint is the same body, relative phase included,
        and ``adjoint`` is ignored.
        """
        if embedding is None:
            return query(name, "original_A", encoded_bits, controls=controls, adjoint=adjoint)
        return sequence(
            name,
            (
                query(
                    name + "_zero",
                    "original_A",
                    encoded_bits,
                    controls=controls + ((embedding, 0),),
                    adjoint=True,
                ),
                query(
                    name + "_one", "original_A", encoded_bits, controls=controls + ((embedding, 1),)
                ),
                gate(name + "_swap", "x", (embedding,), controls=controls),
            ),
        )

    def b_prime(name, *, controls=(), adjoint=False):
        """Prepare ``b' = (b + e_n)/sqrt(2)`` (Dalzell Eq. (9)), or unprepare it.

        H on the augmented coordinate, then ``U_b`` where that coordinate is
        zero. Dalzell arXiv:2406.12086v2, App. A.3 (Fig. 5), uses one extra
        ancilla to label the two branches and a multi-controlled Toffoli to
        uncompute it, because ``e_m`` lies in the same ``s``-qubit register
        as ``b``. Here the augmented coordinate is its own qubit (App. A.6)
        and itself labels the branches. ``e_n`` is the all-zero system state
        with that qubit at one, and ``U_b`` acts only where it is zero, so no
        ancilla is needed.
        """
        h = gate(name + "_h", "h", (augment,), controls=controls)
        b = query(
            name + "_b",
            "rhs",
            original_system,
            controls=controls + ((augment, 0),),
            adjoint=adjoint,
        )
        return sequence(name, (b, h) if adjoint else (h, b))

    def complement(name, *, controls=()):
        """Block-encode ``Q_b' = I - |b'><b'|`` on the ``Q_b'`` flag (wire ``2 + a + 1``).

        The body is ``U_b'`` times an X on the flag, controlled on the
        all-zero state of the system and augmented coordinates, times
        ``U_b'^dagger`` (Dalzell arXiv:2406.12086v2, App. A.1, Fig. 3, with
        ``b`` replaced by ``b'`` as in App. A.4). It flips the flag exactly on
        ``|b'>``, so the flag-zero block is ``Q_b'``. The full unitary is
        Hermitian.
        """
        projector = 2 + a + 1
        return sequence(
            name,
            (
                b_prime(name + "_unprepare", controls=controls, adjoint=True),
                gate(
                    name + "_zero",
                    "x",
                    (projector,),
                    controls=controls + tuple((bit, 0) for bit in system + (augment,)),
                ),
                b_prime(name + "_prepare", controls=controls),
            ),
        )

    def augmented_query(name, *, controls=(), adjoint=False):
        """Block-encode ``A_t = A/alpha + (1/t)|e_n><e_n|`` (Dalzell Eq. (8)) with the ``A_t`` flag.

        Dalzell's ``A`` is already normalized, which here is ``A/alpha``.
        ``A/alpha`` acts where the augmented coordinate is zero. Where it is
        one, the flag (wire ``2 + a``) is set, and on ``|e_n>`` it is reset
        and rotated by ``RY(2 arccos(1/t))``, whose zero amplitude is
        ``cos(arccos(1/t)) = 1/t`` (Dalzell arXiv:2406.12086v2, App. A.2,
        Fig. 4). Other states of that half keep the flag at one, outside the
        block. In the square case ``e_m = e_n``, so Fig. 4's CNOT series is
        not needed. The adjoint reverses the parts and negates the angle.
        """
        aux = 2 + a
        predicate = tuple((bit, 0) for bit in system) + ((augment, 1),)
        angle = 2 * float(np.arccos(1 / rec.t)) * (-1 if adjoint else 1)
        parts = (
            original_query(name + "_A", controls=controls + ((augment, 0),), adjoint=adjoint),
            gate(name + "_cx", "x", (aux,), controls=controls + ((augment, 1),)),
            gate(name + "_predicate", "x", (aux,), controls=controls + predicate),
            gate(name + "_rotation", "ry", (aux,), controls=controls + predicate, angle=angle),
        )
        return sequence(name, tuple(reversed(parts)) if adjoint else parts)

    def g_query(name, *, controls=(), adjoint=False):
        """Block-encode ``G_t = Q_b' A_t`` (Dalzell arXiv:2406.12086v2, Eq. (11),
        App. A.4, Fig. 6), or its adjoint.
        """
        action = augmented_query(name + "_A_t", controls=controls, adjoint=adjoint)
        projection = complement(name + "_projector", controls=controls)
        return sequence(name, (projection, action) if adjoint else (action, projection))

    if method.solver == "qsvt_inverse":
        forward = original_query("encoding_forward")
        backward = original_query("encoding_adjoint", adjoint=True)
    elif method.solver == "shortcut_native_svp":
        forward = g_query("encoding_forward")
        backward = g_query("encoding_adjoint", adjoint=True)
    else:
        # Hermitian dilation of G_t, built like the dilation in original_query.
        # Its block is [[0, G_t], [G_t^dagger, 0]] with the dilation qubit
        # highest, and the body is Hermitian, so it is its own adjoint.
        forward = sequence(
            "encoding_forward",
            (
                g_query("dilation_zero", controls=((dilation, 0),), adjoint=True),
                g_query("dilation_one", controls=((dilation, 1),)),
                gate("dilation_swap", "x", (dilation,)),
            ),
        )
        backward = sequence("encoding_adjoint", (forward,))

    if method.solver == "qsvt_inverse":
        blocks[preparation.record.signature.name] = preparation
        prep = call("prepare_rhs", preparation, original_system)
    else:
        # Dalzell arXiv:2406.12086v2, Algorithm 1, step 1: prepare |e_n> by
        # setting the augmented coordinate. With the G_t dilation the start
        # vector lies in the dilation-one half, the lower block of
        # [[0, G_t], [G_t^dagger, 0]].
        children = [gate("prepare_augmented_basis", "x", (augment,))]
        if dilation is not None:
            children.append(gate("prepare_dilation_basis", "x", (dilation,)))
        prep = sequence("prepare_rhs", children)
    # Real-part QSVT pass ([GSLW] arXiv:1806.01838v1, Corollary 18): H on
    # the pair, the Wx phases in reflection form with their global phase, a
    # Z on the pair for odd degree, projector phases whose sign the pair
    # CNOT flips on the -Phi branch, alternating forward/adjoint queries,
    # then H on the pair.
    # This is the circuit of subroutines/qsp/evolution.py::build_real_chebyshev_encoding.
    phases, global_phase = wx_phases_to_reflection(rec.phase_solution)
    children = [
        prep,
        gate("pair_h_before", "h", (pair,)),
        gate("qsp_global_phase", "global_phase", (), angle=global_phase),
    ]
    if rec.degree % 2:
        children.append(gate("pair_odd_sign", "z", (pair,)))
    flip = gate("projector_flip", "x", (signal,), controls=tuple((bit, 0) for bit in ancillas))
    conjugate = gate("pair_conjugation", "x", (signal,), controls=((pair, 1),))
    for index, phase in enumerate(phases):
        if index:
            children.append(forward if index % 2 else backward)
        rotation = gate(f"projector_rz_{index}", "rz", (signal,), angle=2 * float(phase))
        children.append(
            sequence(f"projector_phase_{index}", (flip, conjugate, rotation, conjugate, flip))
        )
    children.append(gate("pair_h_after", "h", (pair,)))
    coherent = sequence("solution", children)
    # Allocation and release are shared. Each distinct measured prefix has
    # one measurement sequence, so each counts body has at most five children.
    allocate = sequence(
        "allocate", tuple(define(f"allocate_{i}", Allocate(wire=f"bit_{i}")) for i in range(rec.width)))
    vector = isinstance(output, (Solution, StateVector))
    classical = ()
    experiments = []
    if shots is None:
        # The coherent body ends the one amplitude or reduction acquisition.
        root = sequence("body_0", (allocate, coherent))
    else:
        release = sequence(
            "release", tuple(define(f"release_{i}", Release(wire=f"bit_{i}")) for i in range(rec.width)))
        classical = tuple(
            ClassicalValue(name=f"readout_{i}", dtype="bits", width=1)
            for i in range(_counts_width(rec))
        )
        measurements = {}
        batches = []
        for index, setting in enumerate(rec.settings):
            observed = _counts_bits(rec, setting)
            measure = measurements.get(observed)
            if measure is None:
                measure = sequence(f"measure_layout_{len(measurements)}", tuple(
                    define(f"measure_{i}_{bit}", Measure(wire=f"bit_{bit}", result=f"readout_{i}"))
                    for i, bit in enumerate(observed)
                ))
                measurements[observed] = measure
            rotations = []
            for bit, axis in enumerate(reversed(setting.label)):
                if axis in "XY":
                    name = f"rotate_{axis.lower()}_{bit}"
                    label = "".join(axis if j == bit else "I" for j in reversed(range(len(setting.label))))
                    block = blocks.get(name) or select_pauli_parity(name, label, basis=basis)
                    rotations.append(call(f"readout_{name}", block, rec.coordinates))
            basis_change = (sequence(f"basis_{index}", rotations),) if rotations else ()
            body = sequence(f"body_{index}", (allocate, coherent, *basis_change, measure, release))
            batch = define(
                f"batch_{index}",
                MeasurementBatch(
                    body=body,
                    settings=(
                        Setting(
                            label=setting.name,
                            metadata=MetadataRef(
                                format=METHOD,
                                data=InputRef(
                                    identity=problem.content_id, representation="scalar", source=METHOD
                                ),
                            ),
                        ),
                    ),
                    repetitions=shots,
                    observation_kind="counts",
                ),
            )
            batches.append(batch)
            experiments.append(
                Experiment(name=setting.name, batch=batch, setting_index=0, readout=ReadoutDetails())
            )
        root = batches[0] if len(batches) == 1 else sequence("settings", batches)
    registers = tuple(
        Register(
            name=f"bit_{i}", width=1, role="system" if i in rec.coordinates else "clean_ancilla"
        )
        for i in range(rec.width)
    )
    program = admitted_program(
        "QLS.max_admission_steps", method.max_admission_steps,
        root=root,
        definitions=tuple(Definition(id=name, node=node) for name, node in definitions.items()),
        expressions=tuple(Expression(id=name, value=value) for name, value in expressions.items()),
        registers=registers,
        classical=classical,
        signatures=tuple(block.record.signature for block in blocks.values()),
    )
    construction = SelectedConstruction(
        program=program, selections=tuple(block.record for block in blocks.values())
    )
    if vector:
        observation = ObservationSpec(
            kind="amplitudes",
            amplitudes=AmplitudeReadout(
                construction_id=construction.content_id,
                source=METHOD,
                width=rec.width,
                coordinates=rec.coordinates,
                success=rec.success,
                conditions=rec.conditions,
                output=_array_output(problem, output),
                recovery=rec.recovery,
                keep_masses=True,
            ),
        )
        experiments = [
            Experiment(
                name=rec.settings[0].name, setting=rec.settings[0].name, observation=observation
            )
        ]
    elif shots is None:
        moment = not isinstance(output, NormSquared)
        if terms is None:
            terms = _stored_terms(rec, output, max_bytes=method.max_bytes)
        point = ObservationPoint(
            id="end",
            kind="reduction",
            reducer=PROJECTED_MOMENTS,
            parameters=projected_parameters(
                coordinates=rec.coordinates,
                success=rec.success,
                conditions=rec.conditions,
                dimension=rec.original_dimension,
                terms=terms if moment else (),
                moment=moment,
            ),
        )
        experiments = [
            Experiment(
                name=PROJECTED_MOMENTS,
                setting=PROJECTED_MOMENTS,
                observation=ObservationSpec(kind="trajectory", positions=(point,)),
            )
        ]
    return construction, tuple(experiments), tuple(blocks.values())


def plan_quantum(
    method, problem, *, output, shots, rng, reconstruction, encoding, encoded_operator, rhs, svd
):
    """Solve the selected QSP phases and build the matching RHS preparation, query body and
    readout.

    The phases realize ``Re P = polynomial / rescale``. Their residual norming
    bound becomes ``phase_error``, which explicit verification adds to the
    polynomial allowance. Phase solving is the only numerical optimization
    here and runs once, within ``max_qsp_evaluations`` and ``max_bytes``.
    """
    from .method import _error_model
    from nwqlib.subroutines.qsp.phases import solve_symmetric_qsp_phases

    rec = reconstruction
    rec = rec.revise(**_layout(rec, method.solver))
    # Choose output-specific readout acquisitions on the original physical
    # slice after fixing the algorithm and encoding ancilla layout. The one
    # labels() expansion of a Pauli observable names the reduction terms and
    # the grouping candidates.
    terms = _observable_terms(problem, output, max_bytes=method.max_bytes)
    rows = _canonical_terms(terms, len(rec.coordinates))
    settings, groups, comparisons = _readout(method, output, rows, rec, shots=shots)
    rec = rec.revise(
        settings=settings,
        groups=groups,
        grouping_comparisons=comparisons,
        **_observable_fields(output, terms),
    )
    if shots is not None and _counts_width(rec) > 64:
        raise ValueError(
            f"sampled QLS readout needs {_counts_width(rec)} classical bits, "
            "more than its 64-bit count decoder supports")
    phase = solve_symmetric_qsp_phases(
        tuple(c / rec.polynomial.rescale for c in rec.polynomial.coefficients),
        max_degree=method.max_degree,
        max_evaluations=method.max_qsp_evaluations,
        max_bytes=method.max_bytes,
        limit_name="QLS.max_qsp_evaluations",
    )
    preparation = select_preparation("rhs", rhs, per_bit=True)
    rec = rec.revise(phase_solution=phase.phases, phase_error=phase.residual_sup_bound,
                     phase_evaluations=phase.evaluations,
                     preparation_id=preparation.record.content_id)
    # 1 KiB per unit of polynomial degree, circuit wire and observable term,
    # plus 32 KiB, as an allowance for the Program records that _program
    # builds below, plus the stored group bases and member labels.
    _check_bytes(
        1024 * (rec.degree + rec.width + len(terms) + 32)
        + (_counts_program_bytes(rec) if shots is not None else 0)
        + sum(len(setting.label) for setting in settings)
        + sum(len(label) for group in groups for label in group),
        method.max_bytes,
        "QLS selected body metadata",
    )
    base = BaseEncoding(encoding, problem.A, encoded_operator, svd, method.max_bytes, method.max_work)
    construction, experiments, blocks = _program(
        method, problem, output, shots, rec, base, preparation, terms=rows
    )
    plan = Plan(
        problem=problem,
        method=method,
        output=output,
        execution="quantum",
        shots=shots,
        randomness=rng.snapshot(),
        construction=construction,
        experiments=experiments,
        reconstruction=rec,
        error_model=_error_model(problem, method, output, "quantum", shots, construction, rec),
        assumptions=(
            "original spectral estimates and native oracle relations are premises, not a total solution error certificate",
        ),
    )
    plan._bind(
        blocks=blocks,
        base=base,
        preparation=preparation,
        encoding=encoding,
        encoded_operator=encoded_operator,
        rhs=rhs,
        svd=svd,
    )
    validate_selection(plan)
    return plan


def _acquisitions(plan, data):
    """Join each selected experiment to its one completed acquisition before reduction.

    Every chunk must match its Plan, Run, realization, setting, bindings and
    a completed trace event, so analysis never combines data from another
    selection or reuses one acquisition for two settings. The exact
    reduction's one point chunk joins its producing receipt's trajectory
    declaration, and its completed event names that one-point acquisition.
    A counts chunk holds the shared layout whose width is the largest
    measured prefix. _selection checks its setting's zero suffix and remaps
    physical selectors and parity support into that prefix.
    """
    from nwqlib.execution import ObservationView

    validate_selection(plan)
    receipts = {receipt.content_id: receipt for receipt in data.receipts}
    if data.trace.plan_id != plan.content_id or len(data.observations.chunks) != len(
        plan.experiments
    ):
        raise ValueError("QLS requires one actual completed acquisition per selected experiment")
    chunks = {chunk.experiment: chunk for chunk in data.observations.chunks}
    if set(chunks) != {experiment.name for experiment in plan.experiments}:
        raise ValueError("QLS acquisitions differ from the selected settings")
    completed = {}
    for event in data.trace.events:
        if event.status == "completed" and event.execution == "quantum_circuit":
            key = (event.attempt, event.prepared_id, event.observation_id)
            if key in completed:
                raise ValueError("QLS acquisition requires one unique completion event")
            completed[key] = event
    classical_width = _counts_width(plan.reconstruction)
    for experiment in plan.experiments:
        realization = plan.resolve(experiment.name)
        setting, observation = realization.resolved_observation(plan)
        chunk = chunks[experiment.name]
        data.trace.validate_observation(chunk)
        if observation.kind == "trajectory":
            receipt = receipts.get(chunk.prepared_id)
            same = (receipt is not None and receipt.observation == observation
                    and chunk.declares_readout_of(receipt))
            event = completed.get((chunk.attempt, chunk.prepared_id, ObservationView(chunks=(chunk,)).content_id))
        else:
            same = chunk.observation == observation and chunk.population == "unconditional"
            event = completed.get((chunk.attempt, chunk.prepared_id, chunk.content_id))
            if event is not None and event.shots != observation.shots:
                event = None
        if (
            not same
            or chunk.plan_id != plan.content_id
            or chunk.run_id != data.trace.run_id
            or chunk.realization_id != realization.content_id
            or chunk.bindings != realization.bindings
            or chunk.execution != "quantum_circuit"
            or chunk.setting != setting
        ):
            raise ValueError("QLS acquisition differs from its actual selected point")
        if event is None or event.returned_shots != chunk.returned_shots:
            raise ValueError("QLS acquisition differs from its completed population exposure")
        if observation.kind == "counts":
            validate_readout_layout(chunk, observed=tuple(range(classical_width)),
                                    classical=plan.construction.program.classical)
    return chunks


def projected_moments(data, chunk):
    """Saved statistics of the exact reduction, validated against its producing receipt.

    The native width is that of the bound declaration (its coordinates and
    valued selectors partition the native wires), so the populations come
    from the declaration and not from the payload.

    The projected kernel bounds the masses it forms from saved amplitudes with
    its scaled two-pass allowance. It resolves only the receipt's
    amplitude-derived masses label. Mass validation uses the resulting
    qualified saved-state budget (``PreparedArtifact.saved_state_error``,
    host phase product included), or the producing receipt's propagated
    window (``saved_state_probability_window``) when another premise remains
    unavailable. Acquisition and publication use this same pair.
    """
    import json

    receipt = {receipt.content_id: receipt for receipt in data.receipts}.get(chunk.prepared_id)
    if receipt is None or not chunk.declares_readout_of(receipt):
        raise ValueError("QLS exact reduction needs its producing preparation receipt")
    point = receipt.observation.point(chunk.point)
    if point.kind != "reduction" or point.reducer != PROJECTED_MOMENTS:
        raise ValueError("projected statistics require the projected_moments reducer")
    parameters = json.loads(point.parameters)
    width = len(parameters["coordinates"]) + len(parameters["success"]) + len(parameters["conditions"])
    stats = projected_statistics(
        chunk, parameters, width,
        delta=receipt.saved_state_error(PROJECTED_MASS_EXCLUSIONS)[0],
        window=receipt.saved_state_probability_window,
    )
    return QLSProjectedMoments(
        kernel=PROJECTED_KERNEL,
        **stats,
        populations=projected_populations(parameters, width),
        contribution_id=chunk.content_id,
    )


def _exact_outputs(reduction, recovery, output):
    """Publish an exact scalar output from its saved reduction pairs.

    Returns ``(scalar_value, numerator, norm_squared, physical_scale,
    unavailable)``. With p the saved physical mass and q the saved moment,
    NormalizedExpectation publishes q/p (``pair_ratio``), QuadraticForm
    Gamma² q and NormSquared Gamma² p (``recover_scaled_pair``), each with
    ``norm_squared`` Gamma² p. The physical scale is sqrt(p) composed with
    the recovery Gamma: for p = (m, e) the root is sqrt(m) * 2**(e/2) when
    e is even and sqrt(2m)/2 * 2**((e+1)/2) when e is odd, so Gamma² is
    never formed in binary64. With zero physical mass a normalized quantity
    is unavailable, and a physical quadratic form or norm is zero. Analysis
    calls this.
    """
    p, q = reduction.physical_mass, reduction.numerator
    scalar = numerator = scale = unavailable = None
    recovered = recover_scaled_pair(p, recovery)
    norm = None if recovered is None else pair_float(recovered)
    if recovery is not None:
        if p[0] != 0:
            m, e = p
            root = sqrt(m * 2) if e % 2 else sqrt(m)
            scale = compose_recovery(
                PhysicalScale(mantissa=root / 2 if e % 2 else root, exponent=(e + 1) // 2 if e % 2 else e // 2),
                recovery,
            )
        else:
            scale = PhysicalScale(mantissa=0.0, exponent=0)
    if isinstance(output, NormSquared):
        scalar = norm
        if scalar is None:
            unavailable = "physical norm squared is unavailable or unrepresentable"
    elif isinstance(output, NormalizedExpectation):
        if p[0] == 0:
            unavailable = "normalized quantity is undefined for zero physical mass"
        else:
            scalar = pair_ratio(q, p)
            unavailable = None if scalar is not None else "normalized expectation is not representable"
        numerator = scalar
    else:
        recovered = recover_scaled_pair(q, recovery)
        scalar = None if recovered is None else pair_float(recovered)
        unavailable = None if scalar is not None else "physical quadratic form is not representable"
        numerator = scalar
    return scalar, numerator, norm, scale, unavailable


def _selection(rec, chunk, setting, *, classical_width):
    """Decode a compact prefix using its physical-to-classical position map.

    Returns (bits, counts, success, selected, physical). bits are classical
    outcome indices. physical is None when padded coordinates were rotated
    or were not all measured. Unpadded physical selection equals selected.
    Unused suffix bits must be zero. The single-word limit applies to the
    shared classical width, not to physical qubit numbers.
    The caller supplies its known setting and the width computed once for
    the whole acquisition family.
    """
    if chunk.setting != setting.name:
        raise ValueError("QLS counts differ from their selected setting")
    observed = _counts_bits(rec, setting)
    positions = {bit: place for place, bit in enumerate(observed)}
    width = classical_width
    histogram = chunk.histogram()
    if histogram.entries and histogram.width != width:
        raise ValueError("QLS counts differ from their shared classical layout")
    if width > 64:
        raise ValueError("QLS sampled readout decodes at most 64 observed bits")
    bits = histogram.indices() if histogram.entries else np.zeros(0, dtype=np.uint64)
    if len(observed) < width and np.any(bits >> np.uint64(len(observed))):
        raise ValueError("QLS unmeasured classical suffix must be zero")
    counts = histogram.weights

    def matches(pairs):
        mask = np.uint64(sum(1 << positions[bit] for bit, _ in pairs))
        value = np.uint64(sum(v << positions[bit] for bit, v in pairs))
        return (bits & mask) == value

    success = matches(rec.success)
    selected = success & matches(rec.conditions)
    if rec.original_dimension == 1 << len(rec.coordinates):
        physical = selected
    elif set(setting.label) & set("XY") or any(bit not in positions for bit in rec.coordinates):
        physical = None
    else:
        index = np.zeros(bits.shape, dtype=np.uint64)
        for place, bit in enumerate(rec.coordinates):
            index |= ((bits >> np.uint64(positions[bit])) & np.uint64(1)) << np.uint64(place)
        physical = selected & (index < np.uint64(rec.original_dimension))
    return bits, counts, success, selected, physical


def _counts_statistics(plan, chunks, rows=None):
    """Reduce the counts acquisitions of a sampled Plan into its published statistics.

    Returns the ``QLSAnalysis`` fields of the mass acquisition, the group
    moments of a Pauli observable and the sample arrays of a Samples
    output. The mass acquisition is the unrotated ``physical_mass`` setting
    when present, otherwise for padded input the first group whose basis
    has no X or Y, otherwise the first setting. The physical-slice mass is
    not formed from a rotated coordinate outcome of padded input. ``rows``
    are the canonical observable rows when the caller already holds them.

    One group count table supplies every term parity: the parity of label j
    is ``(-1)**popcount(outcome & support_j)`` after support_j is remapped
    from physical coordinates to this setting's measured-bit positions,
    and the success and condition flags are separate from the coordinate
    rotations. For unpadded normalized output, each group's Pauli means are
    formed over its positive number of selected shots, and the identity
    contribution is then exactly its coefficient. Otherwise every returned
    shot is kept with its selected indicator, the identity term ``c_I p_B``
    is assigned once to the first group, and a padded normalized output
    divides by the physical-prefix mass of the mass acquisition.
    """
    from math import fsum
    from nwqlib.core.records import FrozenArray

    rec, output = plan.reconstruction, plan.output
    padded = rec.original_dimension != 1 << len(rec.coordinates)
    unrotated = [s for s in rec.settings if not set(s.label) & set("XY")]
    mass_setting = next((s for s in rec.settings if s.name == "physical_mass"),
                        unrotated[0] if padded and unrotated else rec.settings[0])
    mass_chunk = chunks[mass_setting.name]
    classical_width = _counts_width(rec)
    _, counts, success, _, physical = _selection(
        rec, mass_chunk, mass_setting, classical_width=classical_width)
    returned = mass_chunk.returned_shots
    algorithm_selected = sum(map(int, counts[success]))
    physical_selected = (
        None if physical is None else sum(map(int, counts[physical]))
    )
    fields = dict(
        submitted_shots=mass_chunk.observation.shots,
        returned_shots=returned,
        algorithm_selected_shots=algorithm_selected,
        physical_selected_shots=physical_selected,
        algorithm_success_mass=algorithm_selected / returned if returned else None,
        physical_slice_mass=physical_selected / returned if returned and physical_selected is not None else None,
        mass_contribution_id=mass_chunk.content_id,
    )
    if isinstance(output, Samples):
        indices, sample_counts, _, _ = reduce_sample_arrays(
            mass_chunk, observed=_counts_bits(rec, mass_setting), coordinates=rec.coordinates,
            success=rec.success, conditions=rec.conditions, dimension=rec.original_dimension)
        # The reducer returns fresh, C-contiguous int64 arrays that nothing else
        # references, so the record adopts them without copying.
        fields["samples"] = QLSSamples(indices=FrozenArray._from_owned_canonical_array(indices),
                                       counts=FrozenArray._from_owned_canonical_array(sample_counts))
        del indices, sample_counts
        return fields
    if not isinstance(output, (QuadraticForm, NormalizedExpectation)):
        return fields
    conditional = isinstance(output, NormalizedExpectation) and not padded
    rows = _stored_terms(plan.reconstruction, plan.output, max_bytes=plan.method.max_bytes) if rows is None else rows
    identity = fsum(c for label, c in rows if all(axis == "I" for axis in label))
    coefficients = {}
    for label, c in rows:
        if any(axis != "I" for axis in label):
            coefficients[label] = coefficients.get(label, 0.0) + c
    records = []
    for index, setting in enumerate(rec.settings):
        chunk = chunks[setting.name]
        bits, counts, _, selected, physical = _selection(
            rec, chunk, setting, classical_width=classical_width)
        returned = chunk.returned_shots
        if setting.name == "physical_mass":
            count = sum(map(int, counts[physical]))
            records.append(QLSGroupMoments(
                name=setting.name, basis=setting.label, population="physical_prefix",
                returned_shots=returned, selected_shots=count,
                mean=count / returned if returned else None, contribution_id=chunk.content_id))
            continue
        labels = rec.groups[index]
        positions = {bit: place for place, bit in enumerate(_counts_bits(rec, setting))}
        masks = [sum(1 << positions[rec.coordinates[bit]]
                     for bit, axis in enumerate(reversed(label)) if axis != "I")
                 for label in labels]
        weights = [coefficients[label] for label in labels]
        count = sum(map(int, counts[selected]))
        if not conditional and index == 0 and identity != 0:
            masks, weights = [*masks, 0], [*weights, identity]
        mean = second = variance = None
        if conditional and count:
            mean, second, variance = weighted_group_moments(bits[selected], counts[selected], 1.0, masks, weights)
        elif not conditional and returned:
            mean, second, variance = weighted_group_moments(bits, counts, selected.astype(float), masks, weights)
        records.append(QLSGroupMoments(
            name=setting.name, labels=labels, basis=setting.label,
            population="success_conditional" if conditional else "unconditional",
            returned_shots=returned, selected_shots=count, mean=mean, second_moment=second,
            variance=variance, contribution_id=chunk.content_id))
    fields["groups"] = tuple(records)
    return fields


def _sampled_outputs(plan, rows, physical, groups, physical_selected):
    """Publish a sampled output from its counts statistics.

    Returns ``(scalar_value, numerator, norm_squared, unavailable)``.
    ``physical`` is the physical-slice mass of the mass acquisition,
    ``groups`` the ``QLSGroupMoments`` of ``_counts_statistics`` and
    ``physical_selected`` its selected physical-prefix count. The norm
    squared is Gamma² times the physical mass (``physical_moment``). A
    success-conditional sum adds the identity coefficient to the group
    means. An unconditional sum is divided by the physical-prefix mass for
    NormalizedExpectation and recovered by Gamma² for QuadraticForm.
    Analysis calls this.
    """
    from math import fsum

    rec, output = plan.reconstruction, plan.output
    scalar = numerator = norm = unavailable = None
    if physical is not None:
        norm = physical_moment(physical, rec.recovery)
    if isinstance(output, Samples):
        if not physical_selected:
            unavailable = "conditional samples unavailable: no observed physical-selection shots"
    elif isinstance(output, NormSquared):
        scalar = norm
        if scalar is None:
            unavailable = "physical norm squared is unavailable or unrepresentable"
    else:
        parts = [item for item in groups if item.population != "physical_prefix"]
        identity = fsum(c for label, c in rows if all(axis == "I" for axis in label))
        if any(item.mean is None for item in parts):
            unavailable = "no observed physical-selection shots"
        elif parts[0].population == "success_conditional":
            scalar = fsum([*(item.mean for item in parts), identity])
        else:
            total = fsum(item.mean for item in parts)
            if isinstance(output, NormalizedExpectation):
                # The physical-prefix fraction of the mass acquisition.
                if not physical:
                    unavailable = "no observed physical-selection shots"
                else:
                    scalar = total / physical
            else:
                scalar = physical_moment(total, rec.recovery)
                if scalar is None:
                    unavailable = "physical quadratic form is not representable"
        numerator = scalar
    return scalar, numerator, norm, unavailable


def _published_frames(plan, scale):
    """Return a quantum result's ``(numerator_frame, physical_scale_unavailable)``.

    The numerator of NormalizedExpectation is in the unit frame and that of
    QuadraticForm in the physical frame; other outputs have none. Counts
    yield an empirical success mass, not an exact numerical vector norm. A
    shortcut has no physical reconstruction scale in either case. Analysis
    calls this.
    """
    output, rec = plan.output, plan.reconstruction
    frame = ("unit" if isinstance(output, NormalizedExpectation)
             else "physical" if isinstance(output, QuadraticForm) else None)
    reason = (
        None
        if scale is not None
        else (
            "shortcut supplies no physical reconstruction scale"
            if rec.recovery is None
            else "counts provide an empirical mass rather than a numerical vector norm"
            if plan.shots is not None
            else "physical numerical vector norm is unavailable"
        )
    )
    return frame, reason


def _vector_publication(chunk):
    """Return a quantum Solution's or StateVector's ``(artifact, unavailable)``.

    The Result publishes the first artifact of its acquisition chunk and no
    unavailability reason. Without an artifact it publishes the chunk's
    first unavailability reason, if any. Analysis calls this.
    """
    if chunk.artifacts:
        return chunk.artifacts[0], None
    return None, chunk.unavailable[0].reason if chunk.unavailable else None


def analyze_quantum(plan, data):
    """Reduce acquired QLS populations into the requested vector, physical scalar or conditional
    samples.

    Exact scalar QLS outputs use one acquisition's success mass,
    physical-slice mass and projected observable moment. Their comparison
    with per-setting readout accounts for both arithmetic routes and, for
    padded full-block numerators, the dummy-coordinate coupling. Binary
    recovery is applied after the scalar statistics are formed.

    Exact scalar readout evaluates the stored Pauli observable on the
    success-and-physical slice v, returning ``p=v†v`` and
    ``q=v† O_tilde v``. Normalized output is q/p for positive p. A dense
    observable is zero extended before coefficient conversion. Its
    conversion error contributes separately to error against the original
    dense observable. Projected and full-block Pauli sums coincide for an
    exact zero-extension representation, while rounded coefficients can
    change the cancellation of dummy-coordinate contributions. Recovery and
    numerical reduction errors follow their separate owners. The saved
    pairs are recovered by their binary exponents (``pair_ratio``,
    ``recover_scaled_pair``) without forming Gamma² in binary64; with zero
    physical mass a normalized quantity is unavailable, and a physical
    quadratic form or norm is zero.

    Sampled Pauli outputs combine the group moments of
    ``_counts_statistics`` through ``_sampled_outputs``; Samples keep the
    selected original-coordinate indices and counts as arrays. The numerator
    frame and the scale unavailability reason come from
    ``_published_frames``.
    """

    chunks = _acquisitions(plan, data)
    rec, output = plan.reconstruction, plan.output
    first = chunks[plan.experiments[0].name]
    scalar = norm = numerator = scale = artifact = algorithm = physical = reduction = None
    unavailable = None
    fields = {}
    mass_id = first.content_id
    # Vector readout already carries the acquired artifact and recovery scale.
    # Analysis resolves its identity without reconstructing a second vector.
    if isinstance(output, (Solution, StateVector)):
        artifact, unavailable = _vector_publication(first)
        scale = first.physical_scale
        norm = None if scale is None else scale.squared_as_float()
        values = {value.label: value.value for value in first.values}
        algorithm, physical = values["algorithm_success_mass"], values["physical_slice_mass"]
        if artifact is not None:
            data.artifact(artifact)
    elif plan.shots is None:
        if len(chunks) != 1 or first.point is None:
            raise ValueError("QLS exact scalar output needs its one selected reduction")
        reduction = projected_moments(data, first)
        algorithm, physical = pair_float(reduction.success_mass), pair_float(reduction.physical_mass)
        scalar, numerator, norm, scale, unavailable = _exact_outputs(reduction, rec.recovery, output)
    else:
        rows = _stored_terms(plan.reconstruction, plan.output, max_bytes=plan.method.max_bytes)
        fields = _counts_statistics(plan, chunks, rows)
        algorithm = fields.pop("algorithm_success_mass")
        physical = fields.pop("physical_slice_mass")
        mass_id = fields.pop("mass_contribution_id")
        scalar, numerator, norm, unavailable = _sampled_outputs(
            plan, rows, physical, fields.get("groups", ()), fields["physical_selected_shots"])
    frame, reason = _published_frames(plan, scale)
    result = QLSAnalysis(
        plan_id=plan.content_id,
        construction_id=plan._construction_id,
        observation_id=data.observations.content_id,
        contribution_ids=tuple(c.content_id for c in data.observations.chunks),
        origin=capture_analysis_origin(
            analyzer=METHOD, method_id=plan.method.content_id, dependencies=("numpy",)
        ),
        scalar_value=scalar,
        unavailable=unavailable,
        norm_squared=norm,
        physical_scale=scale,
        physical_scale_unavailable=reason,
        numerator=numerator,
        numerator_frame=frame,
        algorithm_success_mass=algorithm,
        physical_slice_mass=physical,
        artifact=artifact,
        execution="quantum",
        reduction=reduction,
        mass_contribution_id=mass_id,
        **fields,
    )
    result.validate_plan(plan)
    return result
