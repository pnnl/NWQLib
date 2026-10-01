"""Explicit SDK-free selection work, with immutable native input binding."""

from dataclasses import dataclass, field
from math import fsum, isfinite
from uuid import uuid4

import numpy as np

from nwqlib.core.records import Basis, Source
from nwqlib.ir import Binding, BlockSignature, QuantumPort
from nwqlib._preparation_laws import direct_preparation_controlled_cx_bound
from nwqlib.operators.access import DEFAULT_INPUT_BYTES
from nwqlib.operators.inputs import OperatorInput
from nwqlib.problems.inputs import StateInput, ingest_vector
from .records import BlockSemantics, Primitive, SelectedDefinition


def _source(name):
    """Versioned Source of a builtin block implementation or law.

    Every field, including the fixed domain text, is part of each selected
    identity and of the registered law identities that ``resources.fold``
    recognizes, so changing it would change every saved selection.
    """
    return Source(name=name, version="1", domain="M1 logical blocks", reference="nwqlib.blocks versioned selected kernels")


def signed_pauli_cx_bound(index_qubits: int, system_qubits: int, *, controlled=False) -> int:
    """Native CX slot upper bound, including UCG completion and sign diagonal.

    L=2**a labels give <=3(L-1) CX per system-qubit UCG, plus <=L-2
    sign-diagonal CX. One extra control costs <=6 CX per existing CX and
    <=2 CX per one-qubit operation (per UCG: L core slots and 2L-1
    completion-diagonal rotations; also L-1 sign-diagonal rotations).
    Global phase becomes a one-qubit phase under control: zero CX, not zero work.
    Exact dependency projection can reduce these bounds; hardware cost is unknown.

    This is NWQLib's own count of ``subroutines._semantic.signed_pauli_select``,
    with a = index_qubits, L = 2**a and q = system_qubits:

    - Each system qubit receives one uniformly controlled one-qubit gate
      (UCG) with a controls. Qiskit's exact ``UCGate`` construction
      (``up_to_diagonal=False``) uses L one-qubit core gates and L-1 CX,
      followed by an exact completion diagonal on a+1 qubits. This core is
      the decomposition of Bergholm, Vartiainen, Mottonen and Salomaa,
      arXiv:quant-ph/0410066v2, Sec. III, pp. 3-4, drawn in Fig. 6(a),
      p. 5, which implements a UCG with k controls by 2**k one-qubit gates
      and 2**k - 1 CNOTs up to one diagonal (k+1)-qubit gate. In Qiskit 2.5.2,
      the lower bound of the declared dependency, a random UCGate with
      a = 1, ..., 4 lowers to exactly 3(L-1) CX at optimization level 0,
      matching the per-UCG total below.
    - A diagonal on m qubits costs 2**m - 2 CX and 2**m - 1 Rz rotations.
      Shende, Bullock and Markov, arXiv:quant-ph/0406176v5, Theorem 7 (p. 10)
      splits a diagonal into a multiplexed Rz with m-1 select bits and a
      diagonal on the remaining m-1 qubits. Their Theorem 8 (p. 11) gives
      2**k CX for a multiplexed rotation with k select bits. Summing over
      k = 1, ..., m-1 gives the count. The completion diagonal (m = a+1)
      therefore adds 2L-2 CX, so each UCG costs 3(L-1) CX.
    - The coefficient sign diagonal on the index register (m = a) adds at
      most L-2 CX and L-1 rotations.

    The uncontrolled total is ``3q(L-1) + max(0, L-2)``. With one control,
    each CX becomes a Toffoli of 6 CX and one-qubit gates, and each of the
    ``q(3L-1) + L-1`` one-qubit operations becomes a controlled one-qubit
    gate of at most 2 CX (the two-CX circuit of Shende, Bullock and Markov,
    arXiv:quant-ph/0406176v5, Sec. 3.1, p. 9). The six-CX Toffoli is the
    textbook circuit reproduced
    by Shende and Markov, arXiv:0803.2316v1, Fig. 1, p. 3. Their Theorem 1
    shows that no Toffoli circuit of CX and one-qubit gates uses fewer CX.
    """
    if any(type(value) is not int or value < 0 for value in (index_qubits, system_qubits)) or type(controlled) is not bool:
        raise ValueError("CX law requires nonnegative widths and a boolean control flag")
    size = 1 << index_qubits
    cx = system_qubits * 3 * (size - 1) + max(0, size - 2)
    return 6 * cx + 2 * (system_qubits * (3 * size - 1) + size - 1) if controlled else cx


def pauli_readout_cx_bound(index_qubits: int, system_qubits: int, *, controlled=False) -> int:
    """Same UCG core/completion bound as SELECT, with no coefficient sign diagonal."""
    bound = signed_pauli_cx_bound(index_qubits, system_qubits, controlled=controlled)
    size = 1 << index_qubits
    sign_cx = max(0, size - 2)
    return bound - (6 * sign_cx + 2 * (size - 1) if controlled else sign_cx)


def _signature(name, implementation, widths):
    """Declare a joint-coupling signature with one port per nonzero ``(name, width)`` pair.

    A zero-width port is omitted, so a single-term SELECT has no index port.
    """
    ports = tuple(QuantumPort(name=key, width=value) for key, value in widths if value)
    return BlockSignature(name=name, target=_source(implementation), quantum=ports, coupling="joint")


def _no_arguments(arguments):
    if arguments:
        raise ValueError("static native implementation does not accept scalar arguments")


def direct_preparation_amplitudes(block) -> int:
    """Return the amplitude count of the direct state preparation that a block declares, or 0.

    A block declares the input state it prepares in ``semantics.preparation``,
    or in ``semantics.base_semantics`` for a control or adjoint of such a
    block. A ``qiskit.direct`` specification means that the constructor
    synthesizes that state with the direct magnitude/phase tree of
    ``subroutines/state_preparation/direct.py``, whose classical work grows
    as q*2**q in the 2**q amplitudes. That holds for the native PREP leaf
    (through ``prepare_qiskit``), the ADAPT query reference and the LCHS
    vector PREP. Lowering and the Run compare this count with their
    ``max_direct_amplitudes`` before any native constructor runs. A tree
    built inside another constructor without a declaration is bounded by
    that constructor's own admission. The LCU coefficient PREP is bounded
    by ``_admit_lcu``, the band-address PREP of the banded block encoding by
    ``block_encoding.core._banded_plan`` and the LCHS source-branch
    preparations by the dense LCHS input admission.
    """
    semantics = block.record.semantics
    for declared in (semantics, semantics.base_semantics):
        spec = None if declared is None else declared.preparation
        if spec is not None and spec.implementation == "qiskit.direct":
            return spec.basis.dimension
    return 0


# Native constructors bound by SelectedBlock.bind. Lowering calls each one as
# constructor(block, arguments, context) after its admission checks, where
# context(factory) returns the method context and context.max_direct_amplitudes
# is the direct-preparation limit that lowering admitted the block against. The
# returned circuit acts on the block's ports in signature order.


def _preparation_circuit(block, arguments, method_context):
    _no_arguments(arguments)
    from nwqlib.problems.inputs import prepare_qiskit
    return prepare_qiskit(block._payload, max_direct_amplitudes=method_context.max_direct_amplitudes).circuit


def _primitive_circuit(block, arguments, method_context):
    """Apply the stored recipe gate by gate through the QuantumCircuit method of each name.

    The HZH occupation preparation, the Pauli parity network and the Pauli
    group readout basis bind this constructor. Their recipes use only h, z,
    sdg and cx primitives, each a QuantumCircuit method of the same name. A
    recipe with a phase or mc_z primitive, such as the zero reflection, has
    its own constructor.
    """
    _no_arguments(arguments)
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(sum(port.width for port in block.record.signature.quantum))
    for operation in block.record.decomposition:
        getattr(circuit, operation.gate)(*operation.qubits)
    return circuit


def _select_circuit(block, arguments, method_context):
    _no_arguments(arguments)
    from nwqlib.subroutines._semantic import signed_pauli_select
    operator, coefficients, _ = block._payload
    return signed_pauli_select(tuple(label for label, _ in operator.pauli_terms().labels()), coefficients,
                              block.record.semantics.index_qubits, operator.manifest.basis.dimension.bit_length() - 1)


def _reflection_circuit(block, arguments, method_context):
    _no_arguments(arguments)
    from nwqlib.subroutines._semantic import zero_reflection
    return zero_reflection(block._payload, positive_zero=True)


def _readout_circuit(block, arguments, method_context):
    _no_arguments(arguments)
    from nwqlib.subroutines._semantic import pauli_readout_basis
    operator, _, _ = block._payload
    return pauli_readout_basis(tuple(label for label, _ in operator.pauli_terms().labels()),
                              block.record.semantics.index_qubits, operator.manifest.basis.dimension.bit_length() - 1)


@dataclass(frozen=True, eq=False, init=False, slots=True)
class SelectedBlock:
    """Immutable selected record and explicitly bound trusted native code.

    The binding is what connects a portable selection to its native lowering.
    Lowering accepts a block only when its record is the exact selected
    definition named by the Program, so a record loaded from JSON, or an
    equal-looking input, cannot attach a different circuit. A control or
    adjoint keeps the live base block instead of its own constructor, so the
    base's realized gate and global phase remain the single source of the
    transformed action.

    Attributes:
        record: Portable selected definition.
        _payload: Actual admitted native input or immutable construction tuple.
        _base: Selected base for one control and/or adjoint; no global cache.
        _constructor: Live leaf constructor; never loaded from portable provenance.
    """

    record: SelectedDefinition
    _payload: object = field(repr=False)
    _base: object = field(repr=False)
    _constructor: object = field(repr=False)

    def __init__(self, *args, **kwargs):
        raise TypeError("use selected block factories or SelectedBlock.bind")

    @classmethod
    def bind(cls, record, *, payload=None, base=None, constructor=None):
        """Bind trusted code to its exact versioned selection and admitted inputs.

        A leaf constructor receives (block, arguments, method_context) and returns
        a circuit after common lowering admission. It must implement this record's
        semantics/work law and must not keep the context getter or Run.
        Control/adjoint uses the actual base instead of a second constructor.
        Loading a SelectedDefinition alone never recreates executable access.
        """
        if type(record) is not SelectedDefinition:
            raise TypeError("native binding requires a SelectedDefinition")
        if constructor is not None and not callable(constructor):
            raise TypeError("native constructor must be callable")
        if base is not None:
            if (not isinstance(base, SelectedBlock) or constructor is not None
                    or record.base_selection_id != base.record.content_id):
                raise ValueError("transformed binding requires its exact base and no leaf constructor")
        elif record.base_selection_id is not None:
            raise ValueError("transformed selection requires its exact live base")
        elif constructor is None and record.blocker is None:
            raise ValueError("selected leaf requires a bound native constructor")
        instance = object.__new__(cls)
        object.__setattr__(instance, "record", record)
        object.__setattr__(instance, "_payload", payload)
        object.__setattr__(instance, "_base", base)
        object.__setattr__(instance, "_constructor", constructor)
        return instance


def select_preparation(name: str, state: StateInput, *,
                       choice: str = "native", per_bit: bool = False) -> SelectedBlock:
    """Select the native preparation of an admitted StateInput, or HZH for an occupation input.

    Selection scans only the admitted representation;
    it never normalizes, hashes the payload again, synthesizes, or simulates.
    X/HZH are equal as full operators. Generic preparation promises only U|0>.
    per_bit exposes the same ordered native system wires as one-bit ports for
    consumers that explicitly measure or connect individual sites.

    The input is classified so the resource fold gets the tightest law the
    construction supports. Occupation, basis and full-uniform states get an
    exact primitive recipe (with the physical global phase as a phase
    primitive), while prefix-uniform, general and product states point to
    the direct-preparation CX law. A ``qiskit.product`` state is priced as q
    applications of the one-qubit law, not as a generic q-qubit vector. A
    ``qiskit.supplied`` circuit becomes ``preparation.supplied`` with a fresh
    choice label. Its content cannot be recognized, so each supplied circuit
    is its own selection with unknown synthesis cost and error.
    """
    spec = state.preparation
    if spec.blocker:
        raise ValueError(spec.blocker)
    if spec.implementation not in {"qiskit.direct", "qiskit.product", "qiskit.occupation", "qiskit.supplied"}:
        raise ValueError("this StateInput preparation has no native constructor and cost law at select_preparation")
    if choice not in ("native", "hzh"):
        raise ValueError("known preparation choices are native and hzh")
    if choice == "hzh" and spec.implementation != "qiskit.occupation":
        raise ValueError("hzh is defined only for admitted occupation preparation")
    q = (spec.basis.dimension - 1).bit_length()
    if type(per_bit) is not bool:
        raise TypeError("per_bit preparation ports require a boolean choice")
    primitives = None
    classification = "product"
    if spec.implementation == "qiskit.occupation":
        classification = "occupation"
        axes = ("h", "z", "h") if choice == "hzh" else ("x",)
        operations = []
        for j, bit in enumerate(state._physical):
            if bit:
                operations.extend(Primitive(gate=axis, qubits=(j,)) for axis in axes)
        primitives = tuple(operations)
    elif spec.implementation == "qiskit.direct":
        vector = state._direction
        # Counts and views only: at most one N-byte Boolean temporary is live.
        count = int(np.count_nonzero(vector))
        index = int(np.argmax(vector != 0))  # The first nonzero amplitude.
        first = vector[index]
        phase = float(np.angle(first))
        if count == 1:
            classification = "basis"
            primitives = tuple(Primitive(gate="x", qubits=(j,)) for j in range(q) if index & (1 << j))
        elif count == len(vector) == 1 << q and np.all(vector == first):
            classification = "uniform"
            primitives = tuple(Primitive(gate="h", qubits=(j,)) for j in range(q))
        # All count nonzero amplitudes lie in the first count slots and are equal.
        elif np.count_nonzero(vector[:count]) == count and np.all(vector[:count] == first):
            classification = "prefix_uniform"
        else:
            classification = "magnitude_phase"
        if primitives is not None and phase:
            primitives += (Primitive(gate="phase", angle=phase),)
    supplied = spec.implementation == "qiskit.supplied"
    implementation = "preparation.supplied" if supplied else "preparation.hzh" if choice == "hzh" else "preparation.native"
    work = max(spec.work, len(primitives) if primitives is not None else 0)
    semantics = BlockSemantics(
        kind="preparation", input=spec.input, basis=spec.basis,
        relation="U|0>=physical input / positive norm; selected extension fixes U inverse and controlled U",
        input_projector="all-zero state", output_projector="prepared normalized state",
        success="deterministic from all-zero input", workspace=0,
        restoration="no additional workspace; system is transformed", epsilon=None if supplied else 0.0,
        approximation_metric="phase-sensitive state L2 on zero input",
        approximation_evidence=("supplied unitary premise; numerical/synthesis error unknown" if supplied else spec.approximation),
        inverse_legal=True, control_legal=True,
        phase="physical global phase kept, including under coherent control", preparation=spec,
    )
    record = SelectedDefinition(
        signature=_signature(name, implementation,
            tuple((f"system_{bit}", 1) for bit in range(q)) if per_bit else (("system", q),)), semantics=semantics,
        implementation=_source(implementation), choice=f"native:supplied:{uuid4()}" if supplied else f"{choice}:{classification}",
        decomposition=primitives, cost_law=None if supplied else _source("direct_preparation_cx_bound" if primitives is None else "ordered_primitives"),
        cost_parameters=(tuple(Binding(parameter=key, value=value) for key, value in (
            ("num_qubits", q), ("multiplicity", 1), ("snapshot_items", spec.items),
            ("snapshot_logical_payload", spec.payload_bytes), ("snapshot_work", spec.work))) if supplied else
                         (Binding(parameter="num_qubits", value=1 if classification == "product" else q),
                         Binding(parameter="multiplicity", value=q if classification == "product" else 1))),
        cost_context=("stored-data snapshot census only; native synthesis/control cost and allocator overhead unknown" if supplied else
                      "exact ordered logical primitives; control acts on every operation including phase" if primitives is not None else
                      f"{classification}; {spec.work_law}; native CX slot bound is not an exact inventory; workspace/synthesis context required"),
        construction_work=work,
    )
    return SelectedBlock.bind(record, payload=state,
                              constructor=_primitive_circuit if choice == "hzh" else _preparation_circuit)


def _signed_pauli_requirements(m, q):
    """Index width and byte and work laws of a signed Pauli SELECT with m terms on q qubits.

    Returns ``(a, padded, requirements)`` with ``a = ceil(log2 m)`` index
    qubits and ``padded = 2**a`` label slots. The requirement terms are:

    - ``payload_bytes = m*(q + 16) + 32*padded``. The first term is the
      per-term label law of ``PauliTerms.labels``: a q-character label and a
      16-byte complex coefficient. The native constructor ``_select_circuit``
      materializes these labels during lowering, and charging them here
      admits them under the caller's ``max_bytes``. The 32 bytes per slot
      bound the peak of the selection itself, which holds m binary64
      coefficients and the float64 amplitude array together with either at
      most two float64 temporaries of length m while forming
      ``sqrt(|c_j| / alpha)`` or the bytes copy made by ``tobytes``, since
      m <= padded.
    - ``work = m*(q + 1) + (q + 1)*(a + 1)*padded``, the size law reported
      as ``construction_work`` of the native constructor, which lowering
      sums but does not check against a limit. Its unit is one read or
      write of a table entry. The first term reads m labels of q characters
      and their coefficients. The second term covers the q per-qubit UCG
      tables and the sign diagonal. Each is visited in one pass that builds
      the table padded to ``padded`` entries and in one pass per address
      bit. For a UCG table that pass is the dependency test of
      ``project_unitary_table_dependencies``, which compares at most
      padded/2 entry pairs differing in that bit. For the diagonal it is one
      level of ``append_control_diagonal_phases``, which halves the table
      and lowers the differences as an RZ multiplexor whose Walsh butterfly
      makes one pass per remaining control. The projected copy, the
      ``UCGate`` padding and checks, the diagonal's phase wrapping and the
      butterflies add a bounded number of passes of at most ``padded``
      entries per table and per address bit, so the law counts these
      visits up to a small constant factor, not exactly.
    - ``items = max(m, padded)``.

    These are logical size laws, not allocator or RSS measurements.
    """
    a = (max(1, m) - 1).bit_length()
    padded = 1 << a
    return a, padded, dict(items=max(m, padded), payload_bytes=m * (q + 16) + padded * 32,
                           work=m * (q + 1) + (q + 1) * (a + 1) * padded)


def signed_pauli_coefficients(operator):
    """Return the read-only float64 coefficients c_j of a Hermitian Pauli operator, in term order.

    They are the real parts of the stored complex128 coefficients, copied
    bit for bit. Selection derives them here, and loading a saved SELECT or
    readout block derives them again from the restored operator with this
    same expression, so alpha, the coefficients and the PREP amplitudes are
    equal before saving and after loading.
    """
    coefficients = np.array(operator.pauli_terms().coefficients.real, dtype=np.float64)
    coefficients.flags.writeable = False
    return coefficients


def select_signed_pauli(name: str, operator: OperatorInput, *, max_bytes=DEFAULT_INPUT_BYTES) -> SelectedBlock:
    """Build signed Pauli construction data in O(M*q + 2**a) bounded work.

    Includes identity terms; no centering or coefficient threshold. Padding SELECT
    is identity and PREP amplitude zero. The operator input owner admits the zero
    operator, but it cannot define this positive-alpha encoding. Coefficient
    normalization is never metadata work.

    For ``A = sum_j c_j P_j`` with real c_j, ``alpha = sum_j |c_j|`` and
    ``|G> = PREP|0> = sum_j sqrt(|c_j| / alpha) |j>``, SELECT applies
    ``sign(c_j) P_j`` on label j. Then
    ``(<G| tensor I) SELECT (|G> tensor I) = A / alpha``, written with the
    label register first as the papers do. The block places the label
    register on the low-order qubits, so a Qiskit dense matrix of it, whose
    qubit 0 is the least significant index, reads
    ``(I tensor <G|) SELECT (I tensor |G>)``. The recorded relation string
    keeps the paper order. This is the LCU relation of An, Childs and
    Lin, arXiv:2312.03916v2, Appendix A.3, Lemma 24, Eq. (178), with each
    coefficient's sign moved from the two preparation oracles into SELECT.
    Both oracles then coincide, so the unpreparation is the exact adjoint of
    PREP and a single real preparation serves both sides. Kirby, Motta and
    Mezzacapo, arXiv:2208.00567v4, Eqs. (2)-(4), p. 3, use the same form,
    with nonnegative weights and each Pauli carrying its sign. Their weights
    have unit L1 norm, so their alpha_i, P_i and H are ``|c_j|/alpha``,
    ``sign(c_j) P_j`` and ``A/alpha`` here. Every branch ``sign(c_j) P_j`` is
    Hermitian and unitary, so SELECT squares to the identity. That is their
    Eq. (5), p. 4, the premise of the Chebyshev walk in their Lemma 1.

    Selection admits the read-only coefficient array, padded amplitudes and
    their construction temporaries. The archive stores the shared operator
    and amplitude array once and derives coefficients from the restored
    operator. The byte law ``m*(q+16) + 32P`` of
    ``_signed_pauli_requirements``, P = 2**a >= m, bounds the documented
    logical labels and selection arrays because the coefficients are frozen
    before the amplitudes are built: coefficients use 8m, amplitudes 8P,
    and at most two float64 temporaries 16m, giving 24m + 8P <= 32P.
    Coefficient freezing peaks at 16m before amplitudes exist, amplitude
    freezing peaks at 8m + 16P <= 24P, and these phases are sequential. The
    coefficients are the stored real components in order and bits (see
    ``signed_pauli_coefficients``); alpha is ``fsum(abs(c))`` over them and
    each amplitude keeps the operand order ``sqrt(abs(c_j)) / sqrt(alpha)``.
    One term gives a = 0 and P = 1; identity terms are valid; padding has
    zero amplitudes and identity SELECT action; empty or zero-alpha input
    fails the positive-alpha encoding.

    Args:
        name: Signature name of the SELECT block.
        operator: Admitted Hermitian Pauli operator A with M terms on q qubits.
        max_bytes: Limit on the known selection arrays of ``_signed_pauli_requirements``.

    Returns:
        The SelectedBlock with ports ``index`` (a = ceil(log2 M) qubits,
        omitted for M = 1) and ``system`` (q qubits). Its
        ``record.semantics.alpha`` is ``sum_j |c_j|`` in the units of A.
    """
    if operator.structure != "hermitian" or "pauli_terms" not in operator.manifest.access:
        raise ValueError("signed Pauli selection requires admitted Hermitian Pauli access")
    m = operator.manifest.shape[0]
    q = operator.manifest.basis.dimension.bit_length() - 1
    a, padded, requirements = _signed_pauli_requirements(m, q)
    work = requirements["work"]
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("Pauli selection max_bytes must be a nonnegative integer")
    if requirements["payload_bytes"] > max_bytes:
        raise ValueError("signed Pauli coefficient selection exceeds max_bytes")
    # Frozen before the amplitudes exist; alpha and the amplitudes read it in term order.
    coefficients = signed_pauli_coefficients(operator)
    try:
        alpha = fsum(np.abs(coefficients))
    except OverflowError as error:
        raise ValueError("Pauli alpha must be finite and positive") from error
    if not isfinite(alpha) or alpha <= 0:
        raise ValueError("zero Pauli operator has no positive-alpha signed encoding")
    amplitudes = np.zeros(padded, dtype=np.float64)
    amplitudes[:m] = np.sqrt(np.abs(coefficients)) / np.sqrt(alpha)
    # Immutable native tuple; no second input hashing/normalization at lowering.
    amplitudes = np.frombuffer(amplitudes.tobytes(), dtype=np.float64)
    semantics = BlockSemantics(
        kind="signed_pauli_select", input=operator.manifest.reference, basis=operator.manifest.basis,
        relation="SELECT=sum_j |j><j| tensor sign(c_j) P_j; <G|SELECT|G>=A/alpha",
        alpha=alpha, index_qubits=a, input_projector="|G>=sum_j sqrt(abs(c_j)/alpha)|j>; zero padding amplitudes",
        output_projector="same |G>; projected system action A/alpha",
        success="project index onto G; success probability ||A psi||^2/alpha^2",
        workspace=0, restoration="index label preserved; index may entangle with system; padding action is identity",
        epsilon=0.0, approximation_metric="projected block operator norm, ideal gates",
        approximation_evidence="no algorithmic approximation; floating synthesis roundoff not bounded",
        inverse_legal=True, control_legal=True, phase="negative coefficients use phase pi; global phase observable under control",
    )
    record = SelectedDefinition(
        signature=_signature(name, "pauli.signed_select", (("index", a), ("system", q))),
        semantics=semantics, implementation=_source("pauli.signed_select"), choice="signed_identity_padding",
        decomposition=None, cost_law=_source("signed_pauli_cx_bound"),
        cost_parameters=(Binding(parameter="index_qubits", value=a), Binding(parameter="system_qubits", value=q)),
        cost_context="one dependency-projected single-qubit UCG per system qubit plus coefficient sign diagonal; effective label controls determine CX bound; outer control includes all rotations and phases; hardware cost unknown",
        construction_work=work,
    )
    return SelectedBlock.bind(record, payload=(operator, coefficients, amplitudes), constructor=_select_circuit)


def select_pauli_preparation(name: str, select: SelectedBlock, *, max_bytes=DEFAULT_INPUT_BYTES) -> SelectedBlock:
    """Admit the small coefficient vector as a StateInput, then select its PREP.

    Its input identity is the coefficient vector, while SELECT keeps physical A.
    Single-term encodings have no index register and require no PREP call.
    """
    if select.record.implementation.name != "pauli.signed_select" or select._base is not None:
        raise ValueError("coefficient preparation requires an untransformed native signed SELECT")
    if select.record.semantics.index_qubits == 0:
        raise ValueError("single-term SELECT has no index register or PREP")
    state = ingest_vector(select._payload[2], max_bytes=max_bytes)
    prep = select_preparation(name, state)
    return SelectedBlock.bind(prep.record.revise(coefficient_selection_id=select.record.content_id),
                              payload=state, constructor=prep._constructor)


def select_pauli_readout(name: str, select: SelectedBlock) -> SelectedBlock:
    """Select the existing label-controlled readout using admitted data.

    This choice performs no new payload scan. Actual basis-table synthesis is
    charged by explicit logical lowering. The classical diagonal outcome uses
    coefficient sign times non-I system parity; padding outcomes contribute zero.
    """
    if select.record.implementation.name != "pauli.signed_select" or select._base is not None:
        raise ValueError("readout requires an untransformed native signed SELECT")
    record = select.record
    semantics = record.semantics.revise(
        kind="pauli_readout", relation="V: label-controlled H for X, H S^dagger for Y, identity for Z/I/padding; V^dagger D V = Pi_used SELECT Pi_used, not the full identity-padded SELECT",
        input_projector="Pi_used for equivalence with SELECT; readout remains defined with zero weight on its complement", output_projector="computational measurement after V; padding remains in the denominator with zero weight",
        success="no postselection; D=coefficient sign times non-I parity on physical labels, zero on padding",
        approximation_metric="readout observable operator norm, ideal gates",
        restoration="index labels preserved, may remain entangled; no additional workspace",
        phase="Y readout applies S^dagger before H; phases kept under control/inverse",
    )
    signature = _signature(name, "pauli.readout_basis", tuple((port.name, port.width) for port in record.signature.quantum))
    return SelectedBlock.bind(record.revise(
        signature=signature, semantics=semantics, implementation=_source("pauli.readout_basis"),
        choice="signed_label_readout", cost_law=_source("pauli_readout_cx_bound"),
        cost_context="dependency-projected single-qubit UCG per system qubit; no coefficient sign diagonal; classical signed parity and zero padding convention; native CX bound only, hardware cost unknown",
    ), payload=select._payload, constructor=_readout_circuit)


def select_zero_reflection(name: str, num_qubits: int) -> SelectedBlock:
    """Select the Lanczos sign 2|0><0|-I, with no hidden workspace.

    Conjugated by the coefficient PREP G, this block is the reflection
    ``R = (2|G><G| - I) (x) I`` of Kirby, Motta and Mezzacapo,
    arXiv:2208.00567v4, Lemma 1, Eq. (6), p. 4, with the label register
    written first as in ``select_signed_pauli``. With the self-inverse signed
    SELECT U of an operator A, their Eq. (7) gives
    ``(<G| (x) I)(R U)**k (|G> (x) I) = T_k(A/alpha)``. Their H has
    coefficients of unit L1 norm, which is A/alpha here. The Lanczos Method
    reads expectations of these Chebyshev polynomials. The sign convention
    matters under control, because control makes the global phase relative.
    The recipe applies X on every qubit, a multi-controlled Z, X again and a
    global phase pi, so zero gets +1 and every other basis state -1.
    """
    if type(num_qubits) is not int or num_qubits < 0:
        raise ValueError("reflection width must be a nonnegative integer")
    # Multi-controlled Z synthesis has width-dependent cost, not one free gate.
    # This conservative construction envelope is registered in
    # ENGINEERING_CONSTANTS.md. Revisit it when the reflection constructor changes.
    work = (num_qubits + 1) * (1 << num_qubits)
    semantics = BlockSemantics(
        kind="zero_reflection", input=None,
        basis=Basis(identity="computational", dimension=1 << num_qubits, ordering="qubit 0 is rightmost tensor/bit position"),
        relation="2|0><0|-I", input_projector="all-zero projector", output_projector="same projector",
        success="deterministic unitary", workspace=0, restoration="no additional workspace", epsilon=0.0,
        approximation_metric="operator norm, ideal gates", approximation_evidence="exact reflection sign; synthesis roundoff unbounded",
        inverse_legal=True, control_legal=True, phase="positive on zero, negative on every nonzero basis state",
    )
    operations = ()
    if num_qubits:
        # The recipe remains O(q); native multi-control synthesis owns its cost.
        flips = tuple(Primitive(gate="x", qubits=(j,)) for j in range(num_qubits))
        middle = Primitive(gate="z" if num_qubits == 1 else "mc_z", qubits=tuple(range(num_qubits)))
        operations = flips + (middle,) + flips + (Primitive(gate="phase", angle=float(np.pi)),)
    record = SelectedDefinition(
        signature=_signature(name, "reflection.positive_zero", (("system", num_qubits),)),
        semantics=semantics, implementation=_source("reflection.positive_zero"), choice="positive_zero",
        decomposition=operations, cost_law=_source("ordered_primitives"),
        cost_parameters=(Binding(parameter="num_qubits", value=num_qubits),),
        cost_context="2*q X plus (q-1)-controlled Z and global pi; zero width is identity; controlled global pi is a phase gate; native synthesis and hardware costs separate",
        construction_work=work,
    )
    return SelectedBlock.bind(record, payload=num_qubits, constructor=_reflection_circuit)


def transform_block(name: str, block: SelectedBlock, *, control=False, adjoint=False) -> SelectedBlock:
    """One coherent control and/or inverse of the selected extension, never phase-free equivalence.

    A block's semantic contract usually fixes only part of the operator (for
    a preparation, only ``U|0>``). Control and inversion act on the whole
    selected extension, including its global phase, which control makes
    observable. The transformed record therefore keeps the base contract in
    ``base_semantics`` and declares its own approximation unknown, because a
    state or projected-block error of the base does not bound the
    transformed operator. A transformed block cannot be transformed again, so every
    transform refers to one concrete base. When control is added, the
    construction work grows by the controlled law of the base implementation,
    and a direct preparation's CX law switches to its controlled form. The
    Pauli SELECT and readout laws read the ``controlled`` flag instead. A
    dense-dilation encoding adds the work of the exact synthesis of its
    unitary (``_dense_synthesis.dense_synthesis_size``) and of Qiskit's
    control of the synthesized gates (``gatewise_control_size``). A
    supplied preparation or native encoding adds the same for each dense
    unitary in its circuit (``qiskit_compat.dense_synthesis_widths`` and
    ``dense_control_counts``). Lowering charges these syntheses before they
    start, against its ``max_synthesis_work`` or, in a Run, against the
    Run's ``ExecutionLimits.max_synthesis_work``.
    """
    if type(control) is not bool or type(adjoint) is not bool or not (control or adjoint):
        raise ValueError("select at least one boolean control/adjoint transformation")
    record = block.record
    if block._base is not None:
        raise ValueError("transform the original selection once; nested control is outside this subset")
    if record.signature.parameters:
        raise ValueError("parameterized control/adjoint requires its compact native selected implementation")
    if (control and not record.semantics.control_legal) or (adjoint and not record.semantics.inverse_legal):
        raise ValueError("selected semantic contract does not permit this transformation")
    widths = tuple((port.name, port.width) for port in record.signature.quantum)
    ports = (QuantumPort(name="control", width=1),) if control else ()
    ports += tuple(QuantumPort(name=port, width=width) for port, width in widths if width)
    signature = BlockSignature(name=name, target=record.signature.target, quantum=ports, coupling="joint")
    work = record.construction_work
    law = record.cost_law
    if control and work is not None:
        if record.implementation.name == "reflection.positive_zero":
            dimension = record.semantics.basis.dimension
            if record.semantics.kind != "zero_reflection" or dimension.bit_count() != 1:
                raise ValueError("positive-zero reflection law requires a qubit zero-reflection basis")
            q = dimension.bit_length() - 1
            if all(type(width) is int for _, width in widths) and sum(width for _, width in widths) != q:
                raise ValueError("reflection port widths differ from its declared basis")
            # The control adds one qubit to the diagonal reflection, so the
            # registered law (q+1)*2**q of select_zero_reflection is evaluated
            # at width q+1 and added to the base's own construction work.
            work += (q + 2) * (1 << (q + 1))
        elif record.decomposition is not None:
            # Each stored primitive gets one controlled logical operation;
            # controlled CX and S-dagger still need native synthesis laws.
            work += len(record.decomposition)
        elif record.implementation.name == "preparation.native":
            parameters = {binding.parameter: binding.value for binding in record.cost_parameters}
            work += parameters["multiplicity"] * direct_preparation_controlled_cx_bound(parameters["num_qubits"])
            law = _source("direct_preparation_controlled_cx_bound")
        elif record.implementation.name in ("pauli.signed_select", "pauli.readout_basis"):
            a, q = record.semantics.index_qubits, record.semantics.basis.dimension.bit_length() - 1
            bound = signed_pauli_cx_bound if record.implementation.name == "pauli.signed_select" else pauli_readout_cx_bound
            # The controlled CX bound plus one unit for the global phase, which
            # control turns into a one-qubit phase gate with zero CX.
            work += bound(a, q, controlled=True) + 1
        elif record.implementation.name in ("preparation.supplied", "encoding.native"):
            from nwqlib.subroutines._dense_synthesis import dense_synthesis_size, gatewise_control_size
            from nwqlib.subroutines.qiskit_compat import dense_control_counts, dense_synthesis_widths
            # Lowering controls the supplied circuit once per Run through
            # qiskit_compat.controlled, which synthesizes each distinct dense
            # unitary in it exactly before Qiskit controls every occurrence of
            # the synthesized gates.
            circuit = (block._payload._native if record.implementation.name == "preparation.supplied"
                       else block._payload.circuit)
            work += sum(dense_synthesis_size(width)[0] for width in dense_synthesis_widths(circuit))
            counts = dense_control_counts(circuit, 1)
            work += gatewise_control_size(*counts)[0] if counts[0] else 0
        elif (record.implementation.name == "encoding.planned"
              and record.implementation.domain == "dense_dilation"):
            from nwqlib.subroutines._dense_synthesis import (dense_synthesis_size, gatewise_control_counts,
                                                             gatewise_control_size)
            # Lowering controls the base once per Run through
            # qiskit_compat.controlled, which synthesizes the dilation unitary
            # on the ancilla and system qubits exactly before Qiskit controls
            # each synthesized gate.
            qubits = record.semantics.index_qubits + record.semantics.basis.dimension.bit_length() - 1
            work += dense_synthesis_size(qubits)[0]
            work += gatewise_control_size(*gatewise_control_counts(qubits, 1))[0]
    action = "U^dagger" if adjoint else "U"
    if control:
        action = f"|0><0| tensor I + |1><1| tensor {action}"
    semantics = BlockSemantics(
        kind="unitary_transform", input=record.semantics.input, basis=record.semantics.basis,
        relation=f"{action}, where U is the exact selected extension identified by base_selection_id",
        index_qubits=record.semantics.index_qubits,
        input_projector="identity on the full selected register space",
        output_projector="identity on the full selected register space",
        success="deterministic unitary; no postselection; inverse acts on the selected extension",
        workspace=record.semantics.workspace,
        restoration="base workspace obligation kept; selected system/control registers undergo the effective action",
        epsilon=None, approximation_metric="operator norm for the effective transformed action",
        approximation_evidence="unknown: a base zero-input state or projected-block error does not bound the full transformed operator",
        inverse_legal=False, control_legal=False,
        phase="base global phase kept with its adjoint sign; coherent control makes that phase relative",
        base_semantics=record.semantics,
    )
    return SelectedBlock.bind(record.revise(
        signature=signature, controlled=control, adjoint=adjoint,
        semantics=semantics,
        base_selection_id=record.content_id,
        cost_context=record.cost_context + "; transformation: " + ("selected inverse" if adjoint else "selected forward extension") + (" with one coherent control (all phase operations included); controlled cost requires law context" if control else ""),
        construction_work=work, cost_law=law,
    ), base=block)


def select_pauli_parity(name: str, label: str, *, basis: Basis) -> SelectedBlock:
    """Rotate every active site, then collect its Pauli parity on the last site.

    Labels are Qiskit/MSB-first; local qubit coordinates are LSB-first. The
    caller measures the returned pivot and keeps flags outside these ports.
    """
    q = basis.dimension.bit_length() - 1
    if basis.dimension != 1 << q or len(label) != q or set(label) - set("IXYZ"):
        raise ValueError("Pauli parity requires an IXYZ label matching the system basis")
    active = tuple((bit, axis) for bit, axis in enumerate(reversed(label)) if axis != "I")
    size = sum(2 if axis == "Y" else int(axis == "X") for _, axis in active) + max(0, len(active)-1)
    recipe = []
    for bit, axis in active:
        if axis == "Y":
            recipe.append(Primitive(gate="sdg", qubits=(bit,)))
        if axis in ("X", "Y"):
            recipe.append(Primitive(gate="h", qubits=(bit,)))
    if active:
        recipe.extend(Primitive(gate="cx", qubits=(bit, active[-1][0])) for bit, _ in active[:-1])
    signature = _signature(name, "pauli.parity", tuple((f"system_{bit}", 1) for bit in range(q)))
    semantics = BlockSemantics(kind="pauli_readout", input=None, basis=basis,
        relation=f"computational pivot parity after full-site basis change for {label}",
        input_projector="identity", output_projector="identity", success="deterministic basis/parity change",
        workspace=0, restoration="system remains in the selected measurement frame", epsilon=0.,
        approximation_metric="ideal unitary action", approximation_evidence="exact primitive algebra; native roundoff unbounded",
        inverse_legal=True, control_legal=True, phase="full unitary phase preserved")
    return SelectedBlock.bind(SelectedDefinition(signature=signature, semantics=semantics,
        implementation=_source("pauli.parity"), choice=label, decomposition=tuple(recipe), cost_law=None,
        cost_context="ordered full-site basis changes and support-minus-one CX parity network",
        construction_work=size), constructor=_primitive_circuit)


def select_pauli_group_basis(name: str, label: str, *, basis: Basis) -> SelectedBlock:
    """Rotate each site of a qubit-wise commuting group basis to the computational basis.

    label is the group's accumulated basis, Qiskit/MSB-first with physical
    qubit zero at its right end. The block applies H on each X site and S†
    then H on each Y site, and acts as the identity on I and Z sites,
    through one whole-register port ``system`` of width q. It has no CX
    parity network: measuring the whole register afterwards gives the joint
    outcome z, from which a caller decodes each member label j as
    ``(-1)**popcount(z & M_j)`` with M_j the label's support mask. The block
    applies exactly the same tensor product of H and S†/H primitives on
    their physical coordinates as per-site rotations would, through one
    whole-register port. Measuring sites outside the group's support leaves
    the required marginal distribution unchanged: summing the outcomes on
    the additional sites gives the original projective measurement
    probabilities. Its identity is its own ``pauli.group_basis`` Source,
    distinct from ``pauli.parity``, whose relation promises a pivot parity
    rather than a local basis change. construction_work counts the
    primitives, x + 2y for x X sites and y Y sites.
    """
    q = basis.dimension.bit_length() - 1
    if basis.dimension != 1 << q or len(label) != q or not label or set(label) - set("IXYZ"):
        raise ValueError("a group basis must be a nonempty IXYZ label matching the system basis")
    recipe = []
    for bit, axis in enumerate(reversed(label)):
        if axis == "Y":
            recipe.append(Primitive(gate="sdg", qubits=(bit,)))
        if axis in ("X", "Y"):
            recipe.append(Primitive(gate="h", qubits=(bit,)))
    signature = _signature(name, "pauli.group_basis", (("system", q),))
    semantics = BlockSemantics(kind="pauli_readout", input=None, basis=basis,
        relation=f"local readout basis change for {label}: H on X sites, S† then H on Y sites, identity elsewhere",
        input_projector="identity", output_projector="identity", success="deterministic local basis change",
        workspace=0, restoration="system remains in the selected measurement frame", epsilon=0.,
        approximation_metric="ideal unitary action", approximation_evidence="exact primitive algebra; native roundoff unbounded",
        inverse_legal=True, control_legal=True, phase="full unitary phase preserved")
    return SelectedBlock.bind(SelectedDefinition(signature=signature, semantics=semantics,
        implementation=_source("pauli.group_basis"), choice=label, decomposition=tuple(recipe), cost_law=None,
        cost_context="ordered local basis changes on the X and Y sites of a group basis, no CX",
        construction_work=len(recipe)), constructor=_primitive_circuit)
