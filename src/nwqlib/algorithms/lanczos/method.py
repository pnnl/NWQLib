"""Signed Chebyshev moments from one selected input and shared Program.

Write H=center*I+alpha*K, where the spectrum of K lies in [-1,1], and recover
physical Ritz values as center+alpha*x (the shift and rescaling of Oumarou et
al., arXiv 2603.15552v1, Section 2, Eqs. (1) and (4)). The trial space
span{T_k(K)|psi>, k<m} equals the power Krylov space of H, and its Gram and
projected matrices need only the moments mu_k=<psi|T_k(K)|psi>, k<2m
(Kirby, Motta and Mezzacapo, arXiv 2208.00567v4, Section 3.1, Eqs. (13), (14)
and (20)). numerical.py assembles and solves them.

For Pauli input, center is the identity coefficient and alpha=sum_j abs(c_j)
over the nonidentity terms, so K=sum_j (c_j/alpha) P_j is a signed Pauli sum
with unit L1 norm, the encoding of Kirby et al., Section 2.1, Eqs. (2)-(4).
A moment of degree k is read on the walk state after floor(k/2) walk steps
RU, where U is the signed SELECT and R=G(2|0><0|-I)G^dagger reflects about
the coefficient state G|0> (Lemma 1, Eqs. (6)-(7), and Eq. (24)). Even
degrees measure R and odd degrees measure U (Eq. (26)). Exact probabilities
record every requested moment along one walk trajectory
(_trajectory_program). Sampled counts prepare each moment setting
independently.

The odd readout is NWQLib's coherent SELECT-basis readout. A label-controlled
basis change turns every P_i into a Z-type parity, so one sampled setting per
odd degree, or one trajectory point, measures the whole of U, and each shot
returns one signed outcome. Kirby's Section 3.1, step 3 instead measures one
term, or one group of compatible Pauli bases, per setting. Both estimate the
same <U>. The cost is the label-controlled basis change, whose table-size
contract is in ENGINEERING_CONSTANTS "Chebyshev Lanczos defaults".
"""

from nwqlib._limits import DEFAULT_MAX_BYTES

from math import fsum
from typing import Annotated, ClassVar, Literal
from pydantic import Field

from nwqlib.algorithms.protocol import Method, AlgorithmDescriptor, ApplicabilityError
from nwqlib.algorithms._eigen_inputs import (
    eigen_operator,
    eigen_state,
    state_direction,
    preparation_requirements,
    gershgorin_frame,
    DEFAULT_CONVERSION_WORK,
)
from nwqlib.blocks import (
    SelectedConstruction,
    select_preparation,
    select_signed_pauli,
    select_pauli_preparation,
    select_pauli_readout,
    select_zero_reflection,
    transform_block,
)
from nwqlib.core.planning import (
    Experiment,
    ObservationPoint,
    ObservationSpec,
    Plan,
    ReadoutDetails,
    ReadoutView,
    point_items,
    readout_shape,
)
from nwqlib.core.analysis import capture_analysis_origin
from nwqlib.core.records import InputRef, Record, Source, PositiveInt, Real
from nwqlib.ir import (
    Allocate,
    Binding,
    BlockCall,
    ClassicalValue,
    Definition,
    ExprRef,
    Expression,
    Measure,
    MeasurementBatch,
    MetadataRef,
    Parameter,
    ParameterRef,
    PortMap,
    Program,
    Register,
    Release,
    Repeat,
    Sequence,
    Setting,
)
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _pauli_identity_bytes, ingest_pauli_masks
from nwqlib.problems.records import StateData
from .records import LanczosReconstruction, LanczosResult, MomentSetting, MomentStatistics
from .readout import LanczosReadout

METHOD = Source(
    name="chebyshev_lanczos",
    version="5",
    domain="signed Chebyshev projected eigenvalue",
    reference="Kirby, Motta and Mezzacapo, Quantum 7, 1018 (2023), arXiv:2208.00567v4, Eqs. 24-27",
)
DESCRIPTOR = AlgorithmDescriptor(
    method=METHOD.name,
    version=METHOD.version,
    problem_families=("eigenproblem",),
    output_families=("eigenvalue",),
    access_families=("pauli", "dense", "csr", "csc"),
    references=(METHOD,),
    limitations=("projected Ritz estimate does not establish ground identification",),
)

# Planning admits at most 2**20 entries in one exact probability marginal or
# in one histogram of a fixed per-setting shot count. Exact probability
# readout is therefore limited to 20 measured qubits. SensitivitySampling
# plans at its two-shot stage minimum, so its stage histograms are bounded
# later by the Run's data limit. This caps returned data size and is not an
# accuracy parameter. Revisit for a wider exact readout
# (ENGINEERING_CONSTANTS "Chebyshev Lanczos defaults").
MAX_READOUT_ENTRY_BITS = 20


class ChebyshevSubspace(Record):
    """Content identity of one selected trial space span{T_k(K)|psi>, k<m}.

    The operator, reference state, dimension m, affine frame (center, alpha)
    and operator access identify the space and its Chebyshev coordinates. A
    Problem that names another subspace is rejected during planning rather
    than silently reinterpreted.
    """

    operator: InputRef
    reference: InputRef
    m: PositiveInt
    center: Real
    alpha: Real
    access: InputRef


class SensitivitySampling(Record):
    """Two-stage shot allocation for `Lanczos`, weighted by each moment's effect on the energy.

    Build it with keyword arguments, for example
    `SensitivitySampling(total_shots=1400, pilot_fraction=0.2)`, and pass it
    as `Lanczos(sampling=...)`. Both arguments are required. It replaces
    `shots=`, which must then be omitted, and it conflicts with classical
    execution. A pilot stage spends about `pilot_fraction` of the shots
    uniformly. One allocation decision then weights the main stage by each
    moment's pilot energy sensitivity, following the measured-pilot
    suggestion of Oumarou et al., arXiv:2603.15552v1, Section 3.3.2. Only
    main-stage moments enter the final estimate. The weights
    `abs(g_k)*sqrt(v_k)`, the pilot floor and the fallback rules are NWQLib's
    choices, described in the [Lanczos guide](../../algorithms/lanczos.md).
    The allocation is an empirical rule, not a proven optimum. Because the
    main stage allocates its shots from the pilot,
    `prepare(plan, settings="all")` is refused, and `prepare(plan)` prepares
    the next setting.

    Attributes:
        total_shots: Required. Positive number of shots across both stages
            and all settings, at least four per setting.
        pilot_fraction: Required, strictly between 0 and 1. Requested pilot
            share of `total_shots`, adjusted so that each stage keeps two
            shots per setting.
    """

    total_shots: PositiveInt
    pilot_fraction: Annotated[Real, Field(gt=0, lt=1)]


# Fixed bookkeeping allowance of one simultaneous phase, and the fixed
# allowance per newly owned rank-one or rank-two ndarray, on the qualified
# native stack (centered_operator_bytes; ENGINEERING_CONSTANTS "Chebyshev
# Lanczos defaults").
H0 = 65536
ARRAY_FIXED_BYTES = 256


def centered_operator_bytes(table, readout):
    """Return (caller, ingestion) bytes of the SELECT/K helper beyond the supplied table and readout.

    With q qubits, W = ceil(q/64) words per mask and J = len(readout.rows)
    selected rows, the four caller-owned buffers are the physical
    coefficients (8J), the selected x and z words (8JW each) and the
    complex128 coefficient copy (16J), J*(16*W+24) data bytes, plus a
    256-byte fixed allowance each. The ingester's logical envelope beyond
    its caller-owned arrays is J*(4*(16*W+16)+S(q))+(q+8)//8 with
    S(q) = _pauli_identity_bytes(q): the input snapshot and the later
    coalescing/output populations of _coalesced_input, and the identity
    serialization share. Integer products stay Python integers.
    """
    q = table.num_qubits
    W = table.x.shape[1]
    J = len(readout.rows)
    caller = J*(16*W+24) + 4*ARRAY_FIXED_BYTES
    ingestion = J*(4*(16*W+16)+_pauli_identity_bytes(q)) + (q+8)//8
    return caller, ingestion


def _centered_operator(table, readout, *, scaled, max_bytes):
    """Ingest the used nonidentity rows of a packed Pauli table in SELECT address order.

    The x/z words of the rows readout.rows are ingested unchanged, without
    label strings, with their physical coefficients c_j (scaled=False, the
    signed SELECT) or c_j/alpha (scaled=True, the classical K). Source term
    order is kept, no term is pruned and the identity is left out.

    The complete SELECT/K helper admits the caller's selected mask and
    coefficient copies together with the mask-ingestion envelope before
    allocating them. With J selected rows and W words per mask, the four
    caller-owned arrays contain J*(16*W+24) data bytes and receive a
    256-byte fixed allowance each. They remain live throughout ingestion,
    whose existing charge is beyond caller-owned arrays. A further
    65,536-byte allowance covers the phase's other fixed bookkeeping. The
    supplied table and readout are outside this additional-helper charge.
    The allowances describe the qualified native stack and do not bound
    process RSS or every dependency-internal allocation.

    The one check B_centered = caller + ingestion + H0
    = J*(80*W+88+S(q)) + (q+8)//8 + 1024 + 65536 bytes runs before
    ``physical`` is allocated. The ingester then receives
    max_bytes - caller - H0 as its own limit, which keeps its "beyond
    caller-owned arrays" contract without reserving the copies twice.
    Separate checks of the copies and of ingestion against max_bytes would
    never establish their sum. The scaled branch keeps ``physical`` live
    although it is unused, so the four-buffer envelope covers both branches.
    """
    caller, ingestion = centered_operator_bytes(table, readout)
    _check_bytes(caller+ingestion+H0, max_bytes, "Lanczos SELECT/K and mask ingestion")
    physical = table.coefficients.real[readout.rows]
    coefficients = readout.coefficients if scaled else physical
    return ingest_pauli_masks(
        table.x[readout.rows],
        table.z[readout.rows],
        coefficients.astype(complex),
        num_qubits=table.num_qubits,
        max_bytes=max_bytes-caller-H0,
    )


def _plan_readout(operator, reconstruction, *, max_bytes):
    """Derive the readout table of a bound Pauli-input Plan from its selected operator.

    Returns None for matrix input. The operator must be the one the Plan
    names, and the census of its table must reproduce the stored frame and
    index width. Raises ValueError otherwise.
    """
    if reconstruction.enclosure_source != "pauli_l1":
        return None
    readout = LanczosReadout(operator.pauli_terms(), max_bytes=max_bytes)
    if operator.reference != reconstruction.operator or (
        readout.center, readout.alpha, readout.index_width
    ) != (reconstruction.center, reconstruction.alpha, reconstruction.num_index_qubits):
        raise ValueError("Lanczos Pauli table differs from the Plan's selected operator and frame")
    return readout


def _selected_blocks(reference, select_operator, *, a, degrees, trajectory, max_bytes):
    """Select reference PREP, signed SELECT U and, when needed, R and the odd readout.

    With more than one term, R=G(2|0><0|-I)G^dagger uses the coefficient PREP
    G, its inverse and the zero reflection (Kirby et al., arXiv:2208.00567v4,
    Eq. (6)). A single term has no index register, so R is the identity and
    the walk is U alone. An exact trajectory with odd degrees also selects
    the adjoint of the odd readout, which undoes it before the walk
    continues. It is the transform of the same selected readout block, so it
    reverses and adjoints that block's actual operation sequence.
    """
    state = select_preparation("reference", reference)
    select = select_signed_pauli("select", select_operator, max_bytes=max_bytes)
    blocks = [state, select]
    if a:
        prep = select_pauli_preparation("coefficient_prep", select, max_bytes=max_bytes)
        blocks.extend(
            (
                prep,
                transform_block("coefficient_inverse", prep, adjoint=True),
                select_zero_reflection("positive_zero", a),
            )
        )
    if any(k % 2 for k in degrees):
        readout = select_pauli_readout("signed_readout", select)
        blocks.append(readout)
        if trajectory:
            blocks.append(transform_block("signed_readout_inverse", readout, adjoint=True))
    return tuple(blocks)


_READOUT_PADDING = "unused SELECT labels contribute zero; all returned observations kept"


def _moment_program(reference, select_operator, *, m, n, a, degrees, acquisition, shots, sampling, max_bytes):
    """Build the quantum Program that reads the requested moments.

    Record requested even and odd moments along the exact selected walk.
    After j applications of RU, the positive reflection gives mu_(2j) and
    the signed SELECT readout gives mu_(2j+1), by Kirby et al.,
    arXiv:2208.00567v4, Eq. (26). Readout snapshots leave the continuation
    state unchanged. The signed decoder assigns zero to unused SELECT
    addresses and does not renormalize probability mass. Sampled settings
    use independent preparations.

    Exact probabilities build one trajectory experiment
    (_trajectory_program). Counts build one MeasurementBatch per degree
    parity with one setting per degree k. Each setting prepares the
    reference on the system register and G|0> on the index register and
    applies floor(k/2) walk steps RU (Kirby et al., arXiv 2208.00567v4,
    Section 3.1, step 1, p. 6). An even k then applies PREP^dagger and reads
    the index register (step 2). An odd k applies NWQLib's coherent SELECT
    readout and reads the index and system registers, in place of Kirby's
    per-term step 3. With counts, each required moment setting needs fresh
    state preparation: the exact continuation does not furnish a way to
    measure both noncommuting observables nondestructively on one hardware
    sample.

    Args:
        reference: Reference StateInput, zero-padded to 2**n coordinates.
        select_operator: Pauli input of the used nonidentity terms in physical
            units, in SELECT address order.
        m: Trial dimension. Walk steps range over 0..m-1.
        n: System qubits.
        a: Index qubits, ceil(log2(number of terms)), zero for a single term.
        degrees: Nonempty sorted moment degrees to acquire.
        acquisition: ``counts`` or ``probabilities``.
        shots: Shots per setting for counts, 2 under SensitivitySampling, or None.
        sampling: SensitivitySampling, or None for scalar shots and exact
            probabilities.
        max_bytes: Byte limit passed to block selection.

    Returns:
        (program, blocks, settings, experiments): the Program, its selected
        blocks, one MomentSetting per degree, and one Experiment per degree
        for counts or the one trajectory Experiment for exact probabilities.
    """
    if acquisition == "probabilities":
        return _trajectory_program(reference, select_operator, n=n, a=a, degrees=degrees, max_bytes=max_bytes)
    settings, experiments, definitions, registers, classical = [], [], [], [], []
    parameters, expressions = [], []

    def define(name, node):
        definitions.append(Definition(id=name, node=node))
        return name

    def call(name, block, ports):
        return define(
            name,
            BlockCall(
                signature=block.record.signature.name,
                ports=tuple(PortMap(port=port, wire=wire) for port, wire in ports),
            ),
        )

    # Admit each readout layout, including the capped count population,
    # before any block is selected.
    for odd in {k % 2 for k in degrees}:
        readout_shape(
            kind=acquisition,
            details=ReadoutDetails(padding="unused SELECT labels have weight zero; no postselection"),
            width=a + n,
            classical_width=a + n if odd else a,
            repetitions=shots,
            max_items=1 << min(MAX_READOUT_ENTRY_BITS, a + n),
        )
    blocks = _selected_blocks(reference, select_operator, a=a, degrees=degrees, trajectory=False,
                              max_bytes=max_bytes)
    state, select = blocks[:2]
    if a:
        registers.append(Register(name="index", width=a))
    registers.append(Register(name="system", width=n))
    initial = [define("allocate_" + r.name, Allocate(wire=r.name)) for r in registers]
    initial.append(call("reference", state, (("system", "system"),)))
    ports = (("index", "index"), ("system", "system")) if a else (("system", "system"),)
    select_call = call("select", select, ports)
    if a:
        coefficient_prep, inverse, reflection = blocks[2:5]
        p = call("coefficient_prep", coefficient_prep, (("system", "index"),))
        inv = call("coefficient_inverse", inverse, (("system", "index"),))
        refl = call("positive_zero", reflection, (("system", "index"),))
        initial.append(p)
        # In circuit order SELECT, PREP^dagger, zero reflection, PREP:
        # the operator RU with R=G(2|0><0|-I)G^dagger.
        walk_children = (select_call, inv, refl, p)
    else:
        walk_children = (select_call,)
    define("initial", Sequence(children=tuple(initial)))
    define("walk", Sequence(children=walk_children))
    parameters.append(Parameter(name="walks", domain="integer", lower=0, upper=m - 1))
    expressions.append(Expression(id="walk_count", value=ParameterRef(parameter="walks")))
    define(
        "walk_power",
        Repeat(body="walk", count=ExprRef(expression="walk_count")),
    )
    # Stage labels distinguish sampled pilot/main data.
    parameters.append(Parameter(name="stage", domain="integer", lower=0, upper=1))
    parameters.append(
        Parameter(
            name="shots",
            domain="integer",
            lower=1,
            upper=sampling.total_shots if sampling else shots,
        )
    )
    expressions.append(
        Expression(id="shot_count", value=ParameterRef(parameter="shots"))
    )
    repetitions = ExprRef(expression="shot_count")
    if any(k % 2 for k in degrees):
        readout = blocks[-1]
        read = call("signed_readout", readout, ports)
    roots = []
    releases = tuple(define("release_" + r.name, Release(wire=r.name)) for r in registers)
    for wire, width in (("index", a), ("system", n)):
        if width:
            classical.append(
                ClassicalValue(name=wire + "_bits", dtype="bits", width=width)
            )
            define("measure_" + wire, Measure(wire=wire, result=wire + "_bits"))
    for odd in (1, 0):
        selected_degrees = tuple(k for k in degrees if k % 2 == odd)
        if not selected_degrees:
            continue
        tag = "odd" if odd else "even"
        # For even k, PREP^dagger followed by the all-zero index test
        # reads R (Kirby et al., arXiv:2208.00567v4, Section 3.1, step 2).
        tail = (read,) if odd else (inv,)
        children = ["initial", "walk_power", *tail]
        qubits = tuple(range(a + n if odd else a))
        for wire in (["index"] if a else []) + (["system"] if odd else []):
            children.append("measure_" + wire)
        define(tag + "_body", Sequence(children=(*children, *releases)))
        batch_settings = []
        for index, degree in enumerate(selected_degrees):
            name = f"moment_{degree}"
            setting = MomentSetting(
                degree=degree,
                experiment=name,
                walk_steps=degree // 2,
                readout="select" if odd else "reflection",
                qubits=qubits,
                histogram_width=a + n,
            )
            settings.append(setting)
            metadata = MetadataRef(
                format=METHOD,
                data=InputRef(
                    identity=setting.content_id,
                    representation="signed moment readout",
                    source=METHOD,
                ),
            )
            batch_settings.append(
                Setting(
                    label=name,
                    bindings=(Binding(parameter="walks", value=degree // 2),),
                    metadata=metadata,
                )
            )
            experiments.append(
                Experiment(
                    name=name,
                    batch=tag,
                    setting_index=index,
                    readout=ReadoutDetails(padding=_READOUT_PADDING),
                )
            )
        roots.append(
            define(
                tag,
                MeasurementBatch(
                    body=tag + "_body",
                    settings=tuple(batch_settings),
                    observation_kind=acquisition,
                    repetitions=repetitions,
                ),
            )
        )
    define("root", Sequence(children=tuple(roots)))
    # Scalar shots bind stage 0 and the per-setting count once.
    # SensitivitySampling leaves both open for its controller.
    bindings = [] if sampling else [
        Binding(parameter="stage", value=0),
        Binding(parameter="shots", value=shots),
    ]
    program = Program(
        root="root",
        definitions=tuple(definitions),
        registers=tuple(registers),
        classical=tuple(classical),
        parameters=tuple(parameters),
        expressions=tuple(expressions),
        bindings=tuple(bindings),
        signatures=tuple(b.record.signature for b in blocks),
    )
    return program, blocks, settings, experiments


def _trajectory_program(reference, select_operator, *, n, a, degrees, max_bytes):
    """Build the one exact trajectory that reads every requested moment along one walk.

    The body is the initial preparation, the reference on the system
    register and G|0> on the index register, followed by
    max floor(k/2) walk steps RU, so a degree subset stops at its
    largest requested floor(k/2). A single exact trajectory needs one initial
    preparation and m-1 walks to obtain mu_0,...,mu_(2m-1), where the
    independent circuits of the sampled route use
    sum_(k=1)^(2m-1) floor(k/2) = m(m-1) walks (7 and 56 at m=8). The ratio
    m is a walk-count ratio, not a measured wall-time speedup.

    After j walks the schedule declares one point per requested degree,
    the even view for 2j and the odd view for 2j+1, each a probability
    marginal read through a reversible readout view. With
    |psi_j> = W^j |G>|phi>, W = RU, R W R = W^dagger and R psi_0 = psi_0
    give <psi_j|R|psi_j> = mu_(2j) and <psi_j|U|psi_j> = mu_(2j+1)
    (Kirby, Motta and Mezzacapo, arXiv:2208.00567v4, Eq. (26), printed
    page 6, following Eqs. (23)-(25)).

    Even view: apply PREP^dagger (``coefficient_inverse``) to the a index
    qubits, save their probabilities, then apply the same PREP
    (``coefficient_prep``) before continuing. The decoder gives
    2*p_zero - p_total, which measures R because PREP|0> = |G>. Odd view:
    apply the label-controlled basis change B = sum_l |l><l| (x) B_l
    (``signed_readout``), save the index and system probabilities, then
    apply B^dagger (``signed_readout_inverse``). The observed operator is
    U_read = sum_(l<L) |l><l| (x) sign(c_l) P_l, which equals U on the ideal
    active subspace. A padded address returns zero, and its raw mass is
    kept rather than converted to the whole unitary SELECT expectation.

    Views at one boundary each start from the restored continuation state,
    and the final observation needs no inverse when nothing follows it. The
    odd view's inverse is the adjoint of the forward table, target by target
    (``qiskit_compat.inverse_realized_gate``): the multiplexer of the
    adjoint matrices, or the adjoint matrix unitary for a target without
    controls, so the forward tail and its inverse have the same native
    realization. For the
    represented pair write E = B_inv B - I and eps_B >= ||E||_2, the
    operator norm on all a+n wires including padded addresses. With x the
    state before the tail, e_f the error made while applying B and e_b the
    error while applying B_inv, the restored computed state satisfies
    x_after = B_inv(B x + e_f) + e_b = (I + E) x + B_inv e_f + e_b and
    ||x_after - x||_2 <= eps_B ||x||_2 + ||B_inv||_2 ||e_f||_2 + ||e_b||_2.
    The inverse norm factor is necessary when the stored inverse matrices
    are not exactly unitary, and saving the marginal contributes no term.
    For the native adjoint table the remaining represented residual is the
    stored table's Gram defect. With h = fl(1/fl(sqrt(2))) and s = 2 h^2,
    the two-step computation of ``_semantic.pauli_readout_basis``, the
    stored matrices H and H S^dagger have Gram matrix s I and the identity
    has Gram matrix I, so an address whose label has p_c X/Y factors has
    B_c^dagger B_c = s^(p_c) I and
    ||B^dagger B - I||_2 = max_c |s^(p_c) - 1| = 1 - s^(p_max), since s < 1,
    with ||B|| <= 1 and p_c = 0 on padding. These terms bound the
    represented matrices, not Aer's state-vector execution.

    Relative to the same selected walk with the readout excursions removed,
    with intervening ideal steps of norm at most one and a reference state
    of norm at most one, the construction-only difference b obeys
    b_(i+1) <= (1 + eps_i) b_i + eps_i, b_0 = 0, so
    b_k <= prod_(i in O(k)) (1 + eps_i) - 1, where O(k) is the set of odd
    views whose inverses execute before the save for moment k. The
    recurrence follows from (I + E_i)(psi + v) - psi = v + E_i psi + E_i v.
    Other nonidentity readout pairs, including any nonzero represented
    even-pair residual, enter the same product. For the full degree
    schedule |O(k)| = floor(k/2), and a subset schedule counts the views
    that actually precede the save. The diagonal signed score D_k of either
    view has ||D_k|| <= 1, so for a unit-norm or contractive reference state
    psi and state error v the saved moment changes by at most
    2 ||v||_2 + ||v||_2^2, and the construction-only contribution of the
    preceding odd views is at most 2 b_k + b_k^2 when the current tail is
    contractive. ``readout.reduce_moments`` states how these terms enter
    the comparison with separately prepared moments.

    A tail with d native operations and an inverse with the same count add
    2d operations to the receipt's G, and d_forward + d_inverse when the
    inverse lowers to another count.

    Each point's position is the number of bound logical operations before
    it, one operation per bound call (the lowering convention of
    PreparedArtifact.body_length). The body has no measurement register,
    and the trajectory declares zero shots. Planning admits the whole
    declared schedule with readout_shape before block selection, and each
    point's marginal is one exact probability marginal capped by
    MAX_READOUT_ENTRY_BITS. An over-budget schedule is rejected, never
    replaced by fewer points or another route. A backend without trajectory
    views rejects the Plan before work.

    Returns:
        (program, blocks, settings, experiments), with one MomentSetting per
        degree naming its point and the one direct trajectory Experiment.
    """
    walks = max(k // 2 for k in degrees)
    requested = set(degrees)
    width = a + n
    even_wires, odd_wires = tuple(range(a)), tuple(range(width))
    # Bound calls of the initial preparation and of one walk step, in circuit
    # order. The walk applies SELECT, PREP^dagger, the zero reflection and PREP:
    # the operator RU with R=G(2|0><0|-I)G^dagger.
    initial_calls = ("reference", "coefficient_prep") if a else ("reference",)
    walk_calls = ("select", "coefficient_inverse", "positive_zero", "coefficient_prep") if a else ("select",)
    even_view = ReadoutView(tail="coefficient_inverse", inverse="coefficient_prep", wires=even_wires) if a else None
    odd_view = ReadoutView(tail="signed_readout", inverse="signed_readout_inverse", wires=odd_wires)
    points = []
    for j in range(walks + 1):
        position = len(initial_calls) + j * len(walk_calls)
        if 2 * j in requested:
            points.append(ObservationPoint(id=f"moment_{2 * j}", position=position, kind="probabilities",
                                           qubits=even_wires, view=even_view))
        if 2 * j + 1 in requested:
            points.append(ObservationPoint(id=f"moment_{2 * j + 1}", position=position, kind="probabilities",
                                           qubits=odd_wires, view=odd_view))
    observation = ObservationSpec(kind="trajectory", positions=tuple(points), padding=_READOUT_PADDING)
    # Admit the whole schedule, and each point's marginal against the
    # per-marginal cap, before any block is selected.
    readout_shape(kind="trajectory", details=observation, width=width, classical_width=0, repetitions=1)
    for point in points:
        point_items(point, width=width, max_items=1 << min(MAX_READOUT_ENTRY_BITS, width),
                    max_items_source=(f"exact probability readout is limited to {MAX_READOUT_ENTRY_BITS} measured qubits "
                                      "per marginal (MAX_READOUT_ENTRY_BITS); request sampled readout "
                                      "with shots=N"))
    blocks = _selected_blocks(reference, select_operator, a=a, degrees=degrees, trajectory=True,
                              max_bytes=max_bytes)
    selected = {block.record.signature.name: block for block in blocks}
    definitions, registers = [], []

    def define(name, node):
        definitions.append(Definition(id=name, node=node))
        return name

    if a:
        registers.append(Register(name="index", width=a))
    registers.append(Register(name="system", width=n))
    allocations = tuple(define("allocate_" + r.name, Allocate(wire=r.name)) for r in registers)
    ports = {"reference": (("system", "system"),)}
    for name in ("select", "signed_readout", "signed_readout_inverse"):
        ports[name] = (("index", "index"), ("system", "system")) if a else (("system", "system"),)
    for name in ("coefficient_prep", "coefficient_inverse", "positive_zero"):
        ports[name] = (("system", "index"),)
    for name, block in selected.items():
        define(name, BlockCall(signature=block.record.signature.name,
                               ports=tuple(PortMap(port=port, wire=wire) for port, wire in ports[name])))
    define("initial", Sequence(children=(*allocations, *initial_calls)))
    define("walk", Sequence(children=walk_calls))
    define("walk_power", Repeat(body="walk", count=walks))
    releases = tuple(define("release_" + r.name, Release(wire=r.name)) for r in registers)
    define("root", Sequence(children=("initial", "walk_power", *releases)))
    program = Program(
        root="root",
        definitions=tuple(definitions),
        registers=tuple(registers),
        classical=(),
        parameters=(),
        expressions=(),
        bindings=(),
        signatures=tuple(b.record.signature for b in blocks),
    )
    settings = [
        MomentSetting(
            degree=k,
            experiment="trajectory",
            point=f"moment_{k}",
            walk_steps=k // 2,
            readout="select" if k % 2 else "reflection",
            qubits=odd_wires if k % 2 else even_wires,
            histogram_width=len(odd_wires if k % 2 else even_wires),
        )
        for k in degrees
    ]
    experiment = Experiment(name="trajectory", setting="trajectory", observation=observation)
    return program, blocks, settings, [experiment]


class Lanczos(Method):
    """Chebyshev Lanczos method for the smallest eigenvalue of an `Eigenproblem`.

    Build it with keyword arguments and pass it as `method=`, for example
    `solve(Eigenproblem(A=matrix), method=Lanczos(krylov_dimension=4))`.
    Every argument is optional. The result is a
    [`LanczosResult`][nwqlib.algorithms.lanczos.records.LanczosResult] whose
    `eigenvalue` is the lowest Ritz value of the target in the trial space
    `span{T_k(K)|psi>, k<m}` (Kirby, Motta and Mezzacapo, arXiv:2208.00567v4,
    Section 3.1). Here `|psi>` is the initial state, and
    `K = (A - center*I) / alpha` is `A` shifted and scaled so that its
    spectrum lies in [-1, 1] (Oumarou et al., arXiv:2603.15552v1, Section 2,
    Eqs. (1) and (4)). A projected Ritz value does not identify the ground
    state. Quantum execution obtains the moments through the
    qubitized walk, and classical execution evaluates the same moments with
    the three-term recurrence. Both feed one projected solve. The four
    `overlap_*` settings can be changed after the run with
    `result.analyze(...)`, which reuses the same moments. The
    [Lanczos guide](../../algorithms/lanczos.md) describes the construction,
    the cutoff analysis and sensitivity sampling.

    Attributes:
        initial_state: Default `None`. Reference state `|psi>`. `None` draws
            the Method's default reference from the planning random
            generator. A Problem with an explicit `sector` requires a
            supplied state.
        krylov_dimension: Default `None`, which selects `min(8, d)` for
            problem dimension d. Trial dimension m, a positive integer that
            must not exceed d.
        degrees: Default `None`, which requests every degree 1 to 2m-1.
            Distinct positive moment degrees below 2m.
        overlap_cutoff: Default `None`. Positive cutoff on the Gram (overlap)
            eigenvalues. `None` uses `overlap_cutoff_policy`, or the
            numerical floor `1e-12` for exact moments.
        overlap_cutoff_policy: Default `"empirical"`, empirical noise
            filtering. `"confidence"` selects explicit confidence
            regularization. Neither establishes the accuracy of the Ritz
            energy.
        overlap_noise_multiplier: Default `1.0`. Positive scale of the
            empirical Gram RMS noise, a tunable exploratory threshold.
        overlap_failure_probability: Default `0.05`, strictly between 0 and
            1. Tail probability of the reported Gram sampling bound under
            independent bounded shots.
        sampling: Default `None`. A two-stage
            [`SensitivitySampling`][nwqlib.algorithms.lanczos.method.SensitivitySampling]
            shot allocation. It conflicts with `shots=` and with classical
            execution.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Limit on
            the known bytes of input, moments and projected numerical
            workspace.
        max_analysis_work: Default `1e8`. Limit on the work of the projected
            analysis, including matrix assembly and solve. Planning checks it
            before any moment is obtained.
        max_classical_products: Default `1e8`. Limit on the counted
            classical Chebyshev operator applications.
        input_conversion: Default `"auto"`, which keeps the accepted input
            access. `"dense_pauli"` permits explicit dense-to-Pauli
            conversion, which quantum execution needs for sparse input and
            for dense input of dimension above 16.
        max_conversion_work: Default `1e8`. Limit on the work of operator
            conversion, counted as `q*D²` transform work, where D includes
            the quantum embedding. It is not a time estimate.

    Examples:
        `H = ZZ + 0.5*(XI + IX)` on two qubits has lowest eigenvalue
        `-sqrt(2) = -1.4142135623...`, which a three-dimensional trial
        space built from `|00>` recovers on Aer with exact readout:

        >>> from qiskit.quantum_info import SparsePauliOp
        >>> from nwqlib import Eigenproblem, solve
        >>> from nwqlib.algorithms import Lanczos
        >>> H = SparsePauliOp.from_list([("ZZ", 1), ("XI", 0.5), ("IX", 0.5)])
        >>> method = Lanczos(initial_state=[1, 0, 0, 0], krylov_dimension=3)
        >>> result = solve(Eigenproblem(A=H), method=method, seed=7)
        >>> print(round(result.eigenvalue, 10))
        -1.4142135624
    """

    schema_version: Literal[3] = 3
    initial_state: StateData | None = None
    krylov_dimension: PositiveInt | None = None
    degrees: tuple[PositiveInt, ...] | None = None
    overlap_cutoff: Annotated[Real, Field(gt=0)] | None = None
    overlap_cutoff_policy: Literal["empirical", "confidence"] = "empirical"
    overlap_noise_multiplier: Annotated[Real, Field(gt=0)] = 1.0
    overlap_failure_probability: Annotated[Real, Field(gt=0, lt=1)] = 0.05
    sampling: SensitivitySampling | None = None
    max_bytes: PositiveInt = DEFAULT_MAX_BYTES
    max_analysis_work: PositiveInt = 100_000_000
    max_classical_products: PositiveInt = 100_000_000
    input_conversion: Literal["auto", "dense_pauli"] = "auto"
    max_conversion_work: PositiveInt = DEFAULT_CONVERSION_WORK
    result_type: ClassVar[type] = LanczosResult

    descriptor: ClassVar[AlgorithmDescriptor] = DESCRIPTOR

    def plan(self, problem, *, output, execution, shots, rng):
        """Choose the Chebyshev frame, the moments to obtain and the circuits that read them.

        The Plan fixes the frame `H = center*I + alpha*K` once, with the
        source of its spectral enclosure. Pauli input uses the L1 norm of the
        non-identity coefficients (`"pauli_l1"`), which gives K the unit L1
        norm of the encoding of Kirby et al., arXiv:2208.00567v4,
        Section 2.1, after the identity term is removed. For Pauli input,
        Lanczos uses the accepted packed Pauli table to define its centered
        operator and SELECT address order. The stored reconstruction keeps
        the scalar frame and the table's content hash. Readout signs, support
        masks and normalized coefficients come from that same table, so the
        classical action and the quantum decoder use one term order. Dense,
        CSR and CSC input on the classical path uses scaled Gershgorin row
        intervals, or the equal column intervals of Hermitian CSC storage
        (`"scaled_gershgorin"`). The frame determines every moment, so later
        analysis never changes it. Moments known algebraically are recorded
        instead of measured.

        The requested even and odd moments are read along one walk. After j
        applications of `RU`, the positive reflection gives `mu_(2j)` and
        the signed SELECT readout gives `mu_(2j+1)` (Kirby et al.,
        arXiv:2208.00567v4, Eq. (26)). Readout snapshots leave the state that
        the walk continues from unchanged. The signed decoder assigns zero to
        unused SELECT addresses and does not renormalize probability mass.
        Sampled settings use independent preparations.

        With exact probabilities the Plan declares one trajectory, whose
        points read each requested degree through a reversible view. A
        backend without exact trajectory views rejects it before any work,
        with no fallback to independent per-degree simulations. With counts,
        one measurement batch per degree parity holds one setting per degree.
        A setting prepares the reference and the coefficient state, applies
        `floor(k/2)` walk steps, then applies `PREP^dagger` and reads the
        index register (even k), or applies the coherent SELECT readout and
        reads the index and system registers (odd k). Classical execution
        uses one recurrence kernel instead.

        `shots` is the count per setting, or None for exact probabilities,
        and must be None with `SensitivitySampling` or classical execution.
        `rng` draws the default reference when `initial_state` is None. The
        returned Plan carries the reconstruction and the eigenvalue error
        model.

        Raises:
            ValueError: If `krylov_dimension` exceeds the problem dimension,
                if `degrees` are not distinct members of 1 to `2m - 1`, or if
                `shots` conflicts. Also if the projected analysis, or a
                sensitivity pilot with its solve and derivative, would exceed
                `max_bytes` or `max_analysis_work`. These analyses run after
                the moments are measured, so planning rejects them before any
                moment is measured.
            ApplicabilityError: If the Problem requests a different trial
                subspace.
        """
        m = self.krylov_dimension or min(8, problem.dimension)
        if m > problem.dimension:
            raise ValueError("krylov_dimension exceeds the original problem dimension")
        if self.degrees is not None and (
            len(set(self.degrees)) != len(self.degrees) or any(k >= 2 * m for k in self.degrees)
        ):
            raise ValueError("degrees must be distinct members of 1..2m-1")
        if self.sampling is not None and (shots is not None or execution != "quantum"):
            raise ValueError(
                "SensitivitySampling selects quantum counts and conflicts with scalar shots"
            )
        if execution == "classical" and shots is not None:
            raise ValueError("classical Chebyshev moments do not use shots")
        from .numerical import admit_projected_analysis

        # The projected analysis runs after every moment is acquired, but its
        # envelope depends only on m, so it is admitted before any acquisition.
        admit_projected_analysis(m, max_bytes=self.max_bytes, max_work=self.max_analysis_work)
        operator = eigen_operator(
            problem,
            output,
            max_bytes=self.max_bytes,
            as_pauli=execution == "quantum",
            input_conversion=self.input_conversion,
            max_conversion_work=self.max_conversion_work,
        )
        reference = eigen_state(
            self.initial_state,
            problem,
            rng=rng,
            max_bytes=self.max_bytes,
            pad=execution == "quantum",
        )
        acquisition = "counts" if shots is not None or self.sampling else "probabilities"
        if self.sampling:
            # Plan.shots marks a counts Plan at the two-shot stage minimum. The
            # controller in workflow.py binds each stage's actual shots.
            shots = 2
        # System qubits n=ceil(log2 D) of the operator dimension D, at least one.
        n = max(1, (operator.basis.dimension - 1).bit_length())
        pauli = "pauli_terms" in operator.manifest.access
        table = operator.pauli_terms() if pauli else None
        # For Pauli input the census of the packed table fixes the frame and
        # the SELECT address order (readout.packed_lanczos_census).
        readout = LanczosReadout(table, max_bytes=self.max_bytes) if pauli else None
        if pauli:
            center, alpha = readout.center, readout.alpha
            low, high, enclosure = center - alpha, center + alpha, "pauli_l1"
        else:
            low, high, center, alpha = gershgorin_frame(operator, max_bytes=self.max_bytes)
            enclosure = "scaled_gershgorin"
        terms = len(readout.signs) if pauli else 0
        subspace_record = ChebyshevSubspace(
            operator=problem.A.reference,
            reference=reference.reference,
            m=m,
            center=center,
            alpha=alpha,
            access=operator.reference,
        )
        subspace = InputRef(
            identity=subspace_record.content_id, representation="Chebyshev subspace", source=METHOD
        )
        if problem.subspace is not None and problem.subspace != subspace:
            raise ApplicabilityError(
                "requested subspace differs from the selected Chebyshev trial subspace"
            )

        # mu_0=<psi|psi>=1 for the normalized reference.
        known = [(0, 1.0)]
        # Index qubits a=ceil(log2 L) address the L SELECT labels. One term needs none.
        a = readout.index_width if pauli else 0
        degrees = tuple(range(1, 2 * m)) if self.degrees is None else tuple(sorted(self.degrees))
        if alpha == 0:
            # T_k(0)=cos(k*pi/2), exactly; no fake SELECT normalization.
            known = [(k, (1.0, 0.0, -1.0, 0.0)[k % 4]) for k in range(2 * m)]
            degrees = ()
        elif terms == 1:
            # K=+/-P satisfies K^2=I, so every even moment is one and only
            # the odd moments <K> need acquisition.
            known.extend((k, 1.0) for k in range(2, 2 * m, 2))
            degrees = tuple(k for k in degrees if k % 2)
        if self.sampling and degrees:
            # The pilot solve and its derivative run after the pilot stage is
            # acquired. A Plan with no acquired moment never runs the pilot.
            admit_projected_analysis(
                m,
                max_bytes=self.max_bytes,
                max_work=self.max_analysis_work,
                solves=2,
                operation="sensitivity pilot",
            )
        if degrees and execution == "quantum":
            program, blocks, settings, experiments = _moment_program(
                reference,
                _centered_operator(table, readout, scaled=False, max_bytes=self.max_bytes),
                m=m,
                n=n,
                a=a,
                degrees=degrees,
                acquisition=acquisition,
                shots=shots,
                sampling=self.sampling,
                max_bytes=self.max_bytes,
            )
        else:
            # Nothing is read on a quantum register when execution is classical
            # or every requested moment is known (a scalar operator, or only
            # even degrees of a single Pauli term). The classical host
            # construction replaces this empty Program below when degrees remain.
            blocks, experiments, settings = (), [], []
            if execution == "classical":
                settings = [
                    MomentSetting(
                        degree=k,
                        experiment=f"moment_{k}",
                        walk_steps=k // 2,
                        readout="classical",
                        qubits=(),
                        histogram_width=0,
                    )
                    for k in degrees
                ]
            program = Program(
                root="root",
                definitions=(Definition(id="root", node=Sequence()),),
                registers=(),
                classical=(),
                parameters=(),
                expressions=(),
                bindings=(),
                signatures=(),
            )
        construction = SelectedConstruction(
            program=program, selections=tuple(b.record for b in blocks)
        )
        reconstruction = LanczosReconstruction(
            center=center,
            alpha=alpha,
            krylov_dimension=m,
            num_system_qubits=n,
            num_index_qubits=a,
            known=tuple(known),
            settings=tuple(sorted(settings, key=lambda s: s.degree)),
            subspace=subspace,
            physical_scale=reference.preparation.physical_scale,
            operator=operator.reference,
            spectral_lower=low,
            spectral_upper=high,
            enclosure_source=enclosure,
        )
        plan_fields = dict(
            problem=problem,
            method=self,
            output=output,
            execution=execution,
            shots=shots,
            randomness=rng.snapshot(),
            reconstruction=reconstruction,
            assumptions=(
                "raw Chebyshev Gram; its diagonal need not equal one",
                "projected estimate does not establish smallest full-space eigenvalue",
            ),
        )
        if execution == "classical" and degrees:
            from nwqlib.algorithms._eigen_support import host_construction

            labels = tuple(f"moment_{degree}" for degree in reconstruction.moment_indices)
            construction, experiments = host_construction(
                METHOD,
                (operator.reference, reference.reference),
                labels,
                work=self._classical_requirements(operator, reference, reconstruction)[1],
                description="selected classical Chebyshev recurrence, not an independent reference",
            )
        if execution == "quantum" and acquisition == "counts":
            # One sampled experiment per degree, in degree order.
            experiments = sorted(experiments, key=lambda e: int(e.name.split("_")[-1]))
        plan = Plan(**plan_fields, construction=construction, experiments=tuple(experiments))
        from nwqlib.algorithms._eigen_support import eigen_error_model

        plan = plan.revise(error_model=eigen_error_model(plan, METHOD))
        if execution == "classical" and degrees:
            blocks = self._host_blocks(plan, operator, reference, readout)
        return plan._bind(blocks=blocks, operator=operator, reference=reference, readout=readout)

    def _host_blocks(self, plan, operator, reference, readout):
        """Bind the classical recurrence kernel to the selected frame.

        Pauli input is rescaled term by term to K=(H-center*I)/alpha, so the
        kernel runs with center 0 and alpha 1. K is ingested from the x/z
        words of the Plan's readout table and its scaled coefficients, in
        SELECT address order, without label strings. Matrix input keeps its
        original storage and chebyshev_moments applies the affine map. Byte and product
        caps are checked when the kernel is invoked, before any recurrence.
        """
        from nwqlib.blocks.kernels import BoundKernel, KernelOutput
        from nwqlib.execution import ScalarValue
        from .numerical import chebyshev_moments

        rec = plan.reconstruction
        centered = (
            _centered_operator(operator.pauli_terms(), readout, scaled=True, max_bytes=self.max_bytes)
            if rec.enclosure_source == "pauli_l1"
            else operator
        )
        (kernel,) = plan.construction.kernels

        def invoke():
            """Run the recurrence once and return the acquired moments as scalars.

            The Run calls this for the one host acquisition. It admits the
            byte and product totals before materializing the reference, then
            returns one unit-frame ScalarValue ``moment_k`` per acquired
            degree, together with the reference's physical scale.
            """
            required_bytes, required_work = self._classical_requirements(operator, reference, rec)
            _check_bytes(
                required_bytes, self.max_bytes, "classical Chebyshev preparation and recurrence"
            )
            if required_work > self.max_classical_products:
                raise ValueError(
                    "classical Chebyshev preparation and recurrence exceed max_classical_products"
                )
            moments = chebyshev_moments(
                centered,
                state_direction(reference, max_bytes=self.max_bytes),
                m=rec.krylov_dimension,
                center=rec.center if rec.enclosure_source != "pauli_l1" else 0.0,
                alpha=rec.alpha if rec.enclosure_source != "pauli_l1" else 1.0,
                max_bytes=self.max_bytes,
                max_products=self.max_classical_products,
            )
            return KernelOutput(
                plan_id=plan.content_id,
                selected_kernel_id=kernel.content_id,
                scalars=tuple(
                    ScalarValue(
                        label=f"moment_{degree}", value=float(moments[degree]), frame="unit"
                    )
                    for degree in rec.moment_indices
                ),
                physical_scale=reference.preparation.physical_scale,
            )

        return (BoundKernel._bind(plan, kernel, invoke),)

    def _classical_requirements(self, operator, reference, rec):
        """Return (bytes, work) of reference preparation plus the recurrence.

        bytes adds recurrence_requirements to the preparation arrays of
        preparation_requirements. work adds the recurrence count to the known
        preparation products. A custom circuit's unknown preparation work is
        left out, so this total covers known work only.
        """
        from .numerical import recurrence_requirements

        required_bytes, work = recurrence_requirements(
            operator, rec.krylov_dimension, affine=rec.enclosure_source != "pauli_l1"
        )
        prep_bytes, prep_work = preparation_requirements(reference)
        # Unknown (None) preparation work is not counted; this total covers known work only.
        return required_bytes + prep_bytes, work + (prep_work if prep_work is not None else 0)

    def prepare_all_refusal(self):
        """SensitivitySampling refuses ``settings="all"``, because the main stage allocates its shots from the pilot."""
        if self.sampling:
            return (
                "Lanczos with SensitivitySampling cannot prepare every setting in advance, because "
                "the main stage allocates the shots of each setting from the pilot-stage observations. "
                "prepare(plan) with the default settings='first' prepares the setting the "
                "controller submits next, and estimate(plan) folds the Plan's resource laws "
                "without building circuits"
            )
        return super().prepare_all_refusal()

    def prepare(self, plan, *, run, settings="first"):
        """Prepare the controller's next pilot or main setting under SensitivitySampling, else use the static path.

        Under SensitivitySampling, ``settings`` is always ``"first"``, because
        ``prepare_all_refusal`` refuses ``"all"`` before a Run exists, so
        ``prepare_sensitivity`` does not take it.
        """
        if self.sampling:
            from .workflow import prepare_sensitivity

            return prepare_sensitivity(plan, run=run)
        return super().prepare(plan, run=run, settings=settings)

    def execute(self, plan, *, run):
        """Run the two-stage controller of workflow.py under SensitivitySampling, else the static path."""
        if self.sampling:
            from .workflow import execute_sensitivity

            return execute_sensitivity(plan, run=run)
        return super().execute(plan, run=run)

    def recover_analysis(self, plan, *, run):
        """Clear an interrupted sensitivity allocation or final analysis for an explicit retry."""
        from .workflow import recover_sensitivity_analysis

        return recover_sensitivity_analysis(plan, run=run)

    def statistics(self, plan, data, *, stage=None):
        """Aggregate actual moment populations, optionally restricting them to one sampling
        stage.

        Count chunks of one degree are pooled with weights equal to their
        returned shots. Exact-probability point chunks and host chunks count
        once each. An exact moment is associated with its setting by the
        chunk's experiment and trajectory point ID, never by the chunk's
        order, so a missing point leaves only its own degree unacquired and
        the result partial. Analysis, re-analysis and loading read the
        supplied chunks and never acquire. stage selects the pilot (0) or
        main (1) SensitivitySampling stage.
        A quantum chunk's histogram is decoded by readout.decode_histogram
        with the bound Plan's LanczosReadout, which keeps the result by chunk
        and setting identity, so a pilot, an analysis and a re-analysis of
        the same chunk decode it once. Decode the signed marginal with
        denominator one and zero weight for unused SELECT addresses.
        Comparison adds probability-evaluation and signed-sum errors to both
        routes' state contributions. Unknown execution error is not replaced
        by the population floor.
        The return value is (statistics by degree, contributing chunk ids,
        their selected construction ids).
        """
        from nwqlib.algorithms._eigen_support import matched_chunks

        grouped, ids, construction_ids = {}, [], []
        # Sampled settings name no point, so their key ends in None.
        settings = {(s.experiment, s.point): s for s in plan.reconstruction.settings}
        readout = plan._native.get("readout")
        for chunk, construction in matched_chunks(plan, data):
            if (
                stage is not None
                and dict((b.parameter, b.value) for b in chunk.bindings).get("stage") != stage
            ):
                continue
            if chunk.execution == "host_kernel":
                for item in chunk.values:
                    degree = int(item.label.split("_")[-1])
                    if degree not in plan.reconstruction.moment_indices or item.value is None:
                        raise ValueError("classical moment differs from the selected degree")
                    grouped.setdefault(degree, []).append(
                        MomentStatistics(
                            mean=item.value, second_moment=item.value * item.value, shots=None
                        )
                    )
            else:
                setting = settings[chunk.experiment, chunk.point]
                counts = chunk.observation.kind == "counts"
                if not chunk.histogram().entries or counts and not chunk.returned_shots:
                    continue
                if readout is None:
                    raise ValueError("Lanczos quantum analysis needs the bound Plan's Pauli readout table")
                grouped.setdefault(setting.degree, []).append(
                    readout.chunk_moments(setting, chunk, counts=counts)
                )
            ids.append(chunk.content_id)
            construction_ids.append(construction.content_id)
        result = {}
        for degree, rows in grouped.items():
            sampled = rows[0].shots is not None
            population = sum(s.shots for s in rows) if sampled else len(rows)
            result[degree] = MomentStatistics(
                mean=fsum(s.mean * (s.shots if sampled else 1) for s in rows) / population,
                second_moment=fsum(s.second_moment * (s.shots if sampled else 1) for s in rows)
                / population,
                shots=population if sampled else None,
            )
        return result, tuple(ids), tuple(construction_ids)

    def analyze(self, plan, data, *, settings):
        """Solve the Chebyshev projected problem from acquired moments and the chosen overlap
        cutoff.

        settings may override overlap_cutoff, overlap_cutoff_policy,
        overlap_noise_multiplier and overlap_failure_probability for this
        analysis only. Under SensitivitySampling only main-stage (stage 1)
        moments enter the estimate. Returns a LanczosResult attached to the
        Plan and data.
        """
        from .numerical import reconstruct

        from nwqlib._validation import finite_real
        from nwqlib.evidence.error_model import exact_readout_sampling

        unknown = set(settings) - {"overlap_cutoff", "overlap_cutoff_policy",
                                   "overlap_noise_multiplier", "overlap_failure_probability"}
        if unknown:
            raise ValueError(f"unsupported Lanczos analysis settings: {sorted(unknown)}")
        cutoff = settings.get("overlap_cutoff", self.overlap_cutoff)
        policy = settings.get("overlap_cutoff_policy", self.overlap_cutoff_policy)
        multiplier = finite_real(settings.get("overlap_noise_multiplier", self.overlap_noise_multiplier),
                                 "overlap_noise_multiplier")
        delta = finite_real(settings.get("overlap_failure_probability", self.overlap_failure_probability),
                            "overlap_failure_probability")
        if (not isinstance(policy, str) or policy not in {"empirical", "confidence"}
                or multiplier <= 0 or not 0 < delta < 1):
            raise ValueError("invalid Lanczos overlap cutoff policy, multiplier or failure probability")
        if cutoff is not None and (
            isinstance(cutoff, bool)
            or not isinstance(cutoff, (int, float))
            or not 0 < cutoff < float("inf")
        ):
            raise ValueError("overlap_cutoff must be finite and positive")
        # Conditional on the pilot, the main allocations are fixed. Fresh,
        # independent main observations then have fixed-population moments.
        # Pooling pilot data with the sample counts it selected would need a
        # different statistical analysis. This does not remove Ritz-value bias.
        statistics, ids, selected_ids = self.statistics(
            plan, data, stage=1 if self.sampling else None
        )
        solved = reconstruct(
            plan.reconstruction,
            statistics,
            cutoff=cutoff,
            sampled=plan.shots is not None,
            max_bytes=self.max_bytes,
            max_work=self.max_analysis_work,
            overlap_failure_probability=delta,
            overlap_cutoff_policy=policy,
            overlap_noise_multiplier=multiplier,
        )
        return LanczosResult(
            plan_id=plan.content_id,
            construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id,
            contribution_ids=ids,
            origin=capture_analysis_origin(
                analyzer=METHOD, method_id=self.content_id, dependencies=("numpy", "scipy")
            ),
            physical_scale=plan.reconstruction.physical_scale,
            subspace=plan.reconstruction.subspace,
            selected_construction_ids=selected_ids,
            statistics=tuple(sorted(statistics.items())),
            analysis_cutoff=cutoff,
            analysis_cutoff_policy=policy,
            analysis_noise_multiplier=multiplier,
            analysis_failure_probability=delta,
            facts=exact_readout_sampling(plan, data.observations, random_draws=False),
            **solved,
        )._attach(plan, data)

    def save_archive(self, plan, files):
        from .archive import save

        return save(plan, files)

    def verify(self, plan, result, *, checks):
        """Run one explicit check. EnergyShiftOptions compares two saved results, and
        ProjectedVerificationOptions read the stored projected diagnostics.
        Any other options type raises TypeError in verify_projected.
        """
        from nwqlib.evidence.energy_shift import EnergyShiftOptions, verify_energy_shift
        from nwqlib.evidence.verification import verify_projected

        result.validate_plan(plan)
        if type(checks) is EnergyShiftOptions:
            return verify_energy_shift(result, options=checks)
        return verify_projected(result, options=checks)

    @classmethod
    def load_archive(cls, data, files):
        from .archive import load

        return load(data, files)
