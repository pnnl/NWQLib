"""Input-bound block-encoding selection with lazy native construction."""

from dataclasses import replace
from math import isfinite
from uuid import uuid4

from nwqlib.core.records import Float64, Source
from nwqlib.evidence import Evidence
from nwqlib.ir import Binding
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes
from nwqlib.resources.records import ResourceLaw
from .records import BlockSemantics, SelectedDefinition
from .selection import SelectedBlock, _no_arguments, _signature


def _validate_encoding(operator, alpha, error, ancillas, system_qubits):
    """Reject an encoding whose widths, alpha or error bound cannot describe ``operator``.

    The system width must be log2 of the operator's power-of-two dimension,
    alpha must be finite and positive, and a supplied error bound finite and
    nonnegative. None keeps an unknown error.
    """
    dimension = operator.manifest.basis.dimension
    if (type(ancillas) is not int or ancillas < 0 or type(system_qubits) is not int or system_qubits < 0
            or dimension & (dimension-1) or dimension.bit_length()-1 != system_qubits):
        raise ValueError("selected encoding width must match its original operator basis")
    if not isfinite(alpha) or alpha <= 0 or error is not None and (not isfinite(error) or error < 0):
        raise ValueError("selected encoding alpha/error must be finite and in their stated domains")


def _encoding_record(name, *, operator, alpha, error, ancillas, system_qubits,
                     family, implementation, work, snapshot_bytes=None, census=(), query=None):
    """Build the SelectedDefinition of one block encoding of the original operator A.

    The promised relation is the (alpha, a, epsilon) block encoding of
    Gilyén, Su, Low and Wiebe, arXiv:1806.01838v1, Sec. 4.1, Definition 43
    (p. 41): ``||A - alpha (<0_a| tensor I) U (|0_a> tensor I)|| <= epsilon``.
    The formula writes the ancillas first, as the paper does. NWQLib's
    encodings place them on the low-order qubits, so in a Qiskit dense matrix,
    whose qubit 0 is the least significant index, the block is
    ``(I tensor <0_a|) U (I tensor |0_a>)``.
    An, Childs and Lin, arXiv:2312.03916v2, Appendix A.2, Definition 23
    (p. 36), is the exact case epsilon = 0 with alpha >= ||A||. Alpha and
    epsilon are in the units of the original A. A supplied native oracle
    asserts this relation. Selection does not extract or verify its dense
    block.

    Args:
        name: Signature name of the block.
        operator: Admitted operator input that the encoding represents.
        alpha: Positive normalization in the units of A.
        error: Operator-norm error bound epsilon in the units of A, or None when unknown.
        ancillas: Number a of encoding ancillas.
        system_qubits: System width, log2 of the operator dimension.
        family: Encoding family, such as ``dense_dilation``, ``banded`` or ``pauli_lcu``.
        implementation: ``encoding.native`` for a supplied circuit or ``encoding.planned`` for a deferred plan.
        work: Construction work units of the native definition.
        snapshot_bytes: Stored-data census of a supplied circuit, or None.
        census: Named integer gate counts read by the family CX law.
        query: CX count per uncontrolled forward query from that law, or None when no law applies.

    Returns:
        The SelectedDefinition. A known ``query`` becomes one CX ResourceLaw
        whose evidence is a user assertion, because the encoding family and
        its gate counts are the caller's metadata, not a verified inventory.
    """
    source = Source(name=implementation, version="1", domain=family,
        reference="nwqlib.blocks.encoding; projected action bound to the original operator")
    semantics = BlockSemantics(kind="block_encoding", input=operator.reference,
        basis=operator.basis, relation="||A-alpha <0_a|U|0_a>|| <= epsilon",
        alpha=float(alpha), index_qubits=ancillas, input_projector="all encoding ancillas zero",
        output_projector="all encoding ancillas zero", success="projected original operator action",
        workspace=0, restoration="encoding ancillas participate in the unitary; projection selects the block",
        epsilon=None if error is None else float(error), approximation_metric="physical original-A operator norm ||A-alpha B||",
        approximation_evidence="error certificate unavailable" if error is None else
            "supplied native oracle assertion; no operator extraction" if snapshot_bytes is not None else
            "selected block-encoding constructor error in the original-A frame",
        inverse_legal=True, control_legal=True, phase="selected native global phase kept under control and inverse")
    fields = dict(signature=_signature(name, implementation, (("ancillas",ancillas),("system",system_qubits))),
        semantics=semantics, implementation=source, choice=str(uuid4()), decomposition=None, cost_law=None,
        cost_parameters=(Binding(parameter="system_qubits",value=system_qubits),
            Binding(parameter="encoding_ancillas",value=ancillas)),
        cost_context="selected native definition assembly; synthesis/control expansion and native object memory are separate",
        construction_work=work)
    if snapshot_bytes is not None:
        fields["cost_parameters"] += (Binding(parameter="snapshot_data_bytes",value=snapshot_bytes),)
    if query is not None:
        fields["cost_parameters"] += tuple(Binding(parameter=key,value=value) for key,value in census)
        law_source=Source(name="encoding.native.family_law",version="2" if family=="dense_dilation" else "1",domain=family,
            reference="caller-supplied family census; nwqlib.backends.resources.block_encoding_per_query_cx")
        fields["resource_laws"]=(ResourceLaw(metric="cx",basis="cx",
            value=Float64(value=query),interpretation="estimate",bindings=fields["cost_parameters"],
            evidence=Evidence(kind="user_assertion",source=law_source),
            assumptions=("per-query synthesis model of the caller-asserted family; native length not verified",
                "SBM arXiv:quant-ph/0406176v5 Table 1 QSD count, which the exact synthesis can exceed" if family=="dense_dilation" else "four caller-supplied exact synthesis counts",
                "uncontrolled forward query only; controlled/adjoint synthesis remains unknown")),)
    return SelectedDefinition(**fields)


def _native_family_census(encoding):
    """Read only the four named integer counts used by the actual resource law.

    Returns ``(census, query)``: the ``(name, count)`` pairs read from the
    encoding metadata and the CX count per query from
    ``block_encoding_per_query_cx``. A family without a law, missing counts
    or a law value that overflows or is not finite give ``((), None)``. A
    count that is not an ordinary integer raises. The dense dilation reads no
    counts because its law depends only on the system width.
    """
    import numpy as np
    from nwqlib._validation import integer
    from nwqlib.backends.resources import block_encoding_per_query_cx

    family=encoding.implementation
    if family=="dense_dilation":
        keys=()
    elif family=="banded":
        keys=("ucrz_multiplexor_gates","ucrz_basis_cx_per_gate","control_diagonal_basis_cx","prep_pair_cx")
    elif family in ("pauli_lcu","multiplexed_pauli"):
        keys=("ucg_core_cx","ucg_completion_diagonal_cx","coefficient_diagonal_cx","prep_pair_cx")
    else:
        return (),None
    counts={}
    if keys:
        source=encoding.metadata
        if type(source) is not dict:
            return (),None
        if family=="banded":
            source=source.get("periodic_gate_counts",source)
        if type(source) is not dict or any(key not in source for key in keys):
            return (),None
        for key in keys:
            value=source[key]
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} requires an ordinary integer scalar")
            counts[key]=integer(value,key,0)
    try:
        query=block_encoding_per_query_cx(counts,implementation=family,system_qubits=encoding.system_qubits)
    except OverflowError:
        return (),None
    if query is None or not isfinite(query):
        return (),None
    return tuple(counts.items()),query


def select_block_encoding(name, encoding, *, operator, max_bytes=DEFAULT_INPUT_BYTES):
    """Bind a supplied native BlockEncoding using one private execution snapshot.

    The operator equation/error are caller assertions, not a dense validation.
    Equal shapes or metadata cannot reattach another native oracle. SDK import
    occurs only at this explicit native intake.
    """
    from nwqlib.subroutines.block_encoding.core import BlockEncoding
    from ._qiskit_intake import snapshot_circuit

    if type(encoding) is not BlockEncoding:
        raise TypeError("select_block_encoding requires a native BlockEncoding")
    if encoding.circuit.num_qubits != encoding.num_ancillas+encoding.system_qubits:
        raise ValueError("native encoding circuit differs from its declared widths")
    _validate_encoding(operator, encoding.alpha, encoding.error_bound, encoding.num_ancillas, encoding.system_qubits)
    census,query = _native_family_census(encoding)
    circuit,data_bytes = snapshot_circuit(encoding.circuit, max_bytes=max_bytes)
    record = _encoding_record(name, operator=operator, alpha=encoding.alpha, error=encoding.error_bound,
        ancillas=encoding.num_ancillas, system_qubits=encoding.system_qubits,
        family=encoding.implementation, implementation="encoding.native", work=len(circuit.data),
        snapshot_bytes=data_bytes,census=census,query=query)
    return SelectedBlock.bind(record, payload=replace(encoding, circuit=circuit, metadata={}),
                              constructor=_encoding_circuit)


def construct_block_encoding(block, *, max_bytes=DEFAULT_INPUT_BYTES, max_work=1_000_000_000):
    """Construct the exact selected encoding only at the native consumer.

    Known construction arrays are checked before dense completion or multiplexor
    table allocation. This does not claim to bound undocumented SDK workspace.
    """
    from nwqlib.subroutines.block_encoding.core import build_block_encoding_from_plan

    if block.record.implementation.name == "encoding.native":
        return block._payload
    if block.record.implementation.name == "encoding.planned":
        plan=block._payload
        n,a=plan.system_qubits,plan.num_ancillas
        size=(2 << n)**2 if plan.implementation=="dense_dilation" else (1 << a)*max(1,n)
        # A cheaper necessary check before the builder runs. The 64 bytes are
        # one 2x2 complex128 matrix per (label, system qubit) table entry, or
        # four complex128 values per entry of the 2**(n+1)-dimensional dense
        # dilation. The builder then checks its own complete byte law.
        _check_bytes(64*size,max_bytes,"block-encoding construction arrays")
        return build_block_encoding_from_plan(plan, max_bytes=max_bytes, max_work=max_work)
    raise ValueError("selected block is not a registered native/deferred encoding")


def _encoding_circuit(block, arguments, method_context):
    _no_arguments(arguments)
    return construct_block_encoding(block).circuit


def _select_planned_encoding(name, plan, *, operator, dense_svd_selected=False):
    """Bind the method's privately selected plan; no native circuit is built."""
    _validate_encoding(operator, plan.alpha, plan.error_bound, plan.num_ancillas, plan.system_qubits)
    n, a = plan.system_qubits, plan.num_ancillas
    size = (2 << n)**2 if plan.implementation=="dense_dilation" else (1 << a)*max(1,n)
    # Dense completion counts 8D^3 for the SVD, D^3 for the complement and 8D^3
    # for the unitarity product, D = 2**n. A method that already selected the
    # SVD frames omits the first term. ENGINEERING_CONSTANTS.md, DEFAULT_MAX_BLOCK_WORK.
    # Table families charge 32 work units for each of the 2**a*max(1,n)
    # (label, system qubit) table entries, times max(1,n) system qubits, that
    # is 32*2**a*max(1,n)**2, the same law the block-encoding builder checks
    # against max_work. The untuned coefficient 32 is registered in
    # ENGINEERING_CONSTANTS.md.
    work = (9 if dense_svd_selected else 17)*(1 << n)**3 if plan.implementation=="dense_dilation" else 32*size*max(1,n)
    record = _encoding_record(name, operator=operator, alpha=plan.alpha, error=plan.error_bound,
        ancillas=a, system_qubits=n, family=plan.implementation, implementation="encoding.planned",work=work)
    return SelectedBlock.bind(record, payload=plan, constructor=_encoding_circuit)
