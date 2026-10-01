"""Controlled powers of U for the Hadamard test: step selection and circuit constructors.

select_powers chooses, once per Plan, how controlled powers of U are
built: exact dense powers of one base (the polar factor of a supplied
unitary, or the spectral exponential of a Hamiltonian), or repetitions of
controlled second-order Suzuki steps selected from the commutator bound of
Childs et al. doi:10.1103/PhysRevX.11.011020
(subroutines/trotterization/error_budget.py). An exact static trajectory
repeats one shared step on a common integer grid and applies gap blocks
between its power positions. Sampled static settings and RWPE select a
step count for each power independently. The ``construct_*``
functions build the Qiskit circuits only when a native circuit is requested.
For product-formula powers the Hamiltonian identity coefficient is split off
and acts as a separate phase gate on the control (method._quantum_program).
A dense power contains it in its matrix.
"""

from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import frexp, isfinite

from nwqlib._linalg_laws import hermitian_eigensystem_work
from nwqlib.blocks.records import BlockSemantics, Primitive, SelectedDefinition
from nwqlib.blocks.selection import SelectedBlock
from nwqlib.core.records import Float64, Source
from nwqlib.evidence import Evidence
from nwqlib.ir import Binding, BlockSignature, Parameter, QuantumPort
from nwqlib.resources import ResourceLaw
from nwqlib.subroutines._dense_synthesis import controlled_synthesis_size
from nwqlib.subroutines.trotterization.error_budget import (
    COEFFICIENT_ARITHMETIC,
    PruningBudgetExhausted,
    TrotterStepSelection,
    _BoundCoefficientEvaluation,
    _emitted_parts,
    _finite_upper,
    _pauli_bound_coefficient_from_terms,
    _remaining_budget_after_pruning,
    _select_common_step,
    _selection_from_evaluation,
    census_bytes,
    census_sizes,
    census_work,
    choose_census_block,
)
from .records import QPECommonStep, QPEPower


def source(name):
    """Return the implementation Source ``qpe.<name>`` of a QPE block kind."""
    return Source(
        name="qpe." + name,
        version="1",
        domain="selected physical controlled QPE construction",
        reference="nwqlib.algorithms.qpe.powers; identity phase stored under control",
    )


# Sources and qualifications of the per-block CX laws. The step law is the
# decomposition count derived in suzuki_step_cx. A one-qubit P or RZ gate has
# no CX in any lowering to CX and one-qubit gates.
_CX_SOURCES = {
    "suzuki_step": Source(
        name="qpe.suzuki_step_cx",
        version="1",
        domain="controlled second-order Suzuki step",
        reference="4*sum_j w_j CX with 2*w per controlled Pauli leaf and two leaves per term, "
        "for Qiskit 2.5.2 PauliEvolutionGate.control at optimization level 0 without routing",
    ),
    "phase": Source(
        name="qpe.phase_cx",
        version="1",
        domain="one-qubit P or RZ",
        reference="a one-qubit gate has no CX",
    ),
}
_CX_ASSUMPTIONS = {
    # Stored text, part of Plan content identity.
    "suzuki_step": (
        "Qiskit 2.5.2 lowering to cx and one-qubit gates at optimization level 0, "
        "before cancellation, rotation merging and routing, and another Qiskit version can differ",
    ),
    "phase": (),
}


def _make(
    name,
    kind,
    *,
    basis,
    reference,
    payload,
    width,
    work,
    epsilon=None,
    evidence="selected second-order/pruning relation where given; native numerical error unknown",
    constructor,
    parameters=(),
    cost_parameters=(),
    operations=None,
    cx=None,
    decomposition=None,
    context="",
):
    """Return a SelectedBlock for one QPE block kind with its payload and constructor.

    ``kind`` is ``suzuki_step`` or ``dense_power`` (ports ``control`` and,
    when ``width > 1``, ``system``) or ``hadamard`` or ``phase`` (one
    ``register`` port). ``payload`` is the data the constructor needs,
    ``work`` the admitted construction size and ``epsilon`` the per-block
    operator-norm error bound, when one exists, with ``evidence`` stating
    why. ``operations`` gives the exact number of stored instructions per
    block and ``cx`` its exact CX count in the qualified lowering
    (``suzuki_step_cx`` for a step, zero for a one-qubit phase). Each is
    recorded as a resource law only for blocks without a primitive
    ``decomposition``.

    The fold applies a law only when its fixed bindings equal the call's
    cost parameters (``resources/fold.py``, ``_Fold.leaf``), so both laws
    bind exactly the ``cost_parameters`` of their block, for example the
    ``power``, ``steps`` and ``step_time`` of one Suzuki step. The formal
    ``parameters`` (the ``angle`` of a phase leaf) stay unbound and the laws
    cover their whole admitted domain, since one P or RZ instruction with no
    CX is stored whatever the angle. The laws keep ``controlled=False``,
    because a Suzuki step already contains its QPE control and the flag
    describes an additional control transformation of the selection.
    """
    implementation = source(kind)
    costs = tuple(Binding(parameter=key, value=value) for key, value in cost_parameters)
    ports = (
        (QuantumPort(name="control", width=1),)
        + ((QuantumPort(name="system", width=width - 1),) if width > 1 else ())
        if kind in ("suzuki_step", "dense_power")
        else (QuantumPort(name="register", width=width),)
    )
    signature = BlockSignature(
        name=name, target=implementation, quantum=ports, parameters=parameters
    )
    semantics = BlockSemantics(
        kind="unknown",
        input=reference,
        basis=basis,
        relation=context,
        input_projector="identity on the selected register",
        output_projector="identity on the selected register",
        success="deterministic selected unitary action",
        workspace=0,
        restoration="no additional workspace",
        epsilon=epsilon,
        approximation_metric="unitary operator norm before native floating synthesis",
        approximation_evidence=evidence,
        inverse_legal=False,
        control_legal=False,
        phase="physical phase stored; identity acts on the QPE control as P(-power*tau*identity)",
    )
    # Logical constructor slots are exact at this basis only. They do not
    # predict SDK decomposition or target-native gate counts.
    laws = ()
    if operations is not None and decomposition is None:
        law_source = Source(
            name="qpe.selected_slots",
            version="1",
            domain=kind,
            reference="one stored instruction per selected phase or controlled Pauli rotation; no native basis prediction",
        )
        laws = (
            ResourceLaw(
                metric="operations",
                basis="selected_logical",
                value=operations,
                interpretation="exact",
                bindings=costs,
                unbound_parameters=tuple(parameter.name for parameter in parameters),
                evidence=Evidence(kind="proved_relation", source=law_source),
                assumptions=("logical constructor slots; HLS/native basis inventory is separate",),
            ),
        )
    if cx is not None and decomposition is None:
        laws += (
            ResourceLaw(
                metric="cx",
                basis="cx",
                value=cx,
                interpretation="exact",
                bindings=costs,
                unbound_parameters=tuple(parameter.name for parameter in parameters),
                evidence=Evidence(kind="proved_relation", source=_CX_SOURCES[kind]),
                assumptions=_CX_ASSUMPTIONS[kind],
            ),
        )
    record = SelectedDefinition(
        signature=signature,
        semantics=semantics,
        implementation=implementation,
        choice=name,
        decomposition=decomposition,
        cost_law=None,
        resource_laws=laws,
        cost_parameters=costs,
        cost_context=context
        + "; logical construction size excludes unknown SDK synthesis/allocator overhead",
        construction_work=work,
    )
    return SelectedBlock.bind(record, payload=payload, constructor=constructor)


def select_auxiliary(*, basis, reference):
    """Select the Hadamard and the two one-qubit phase blocks of the QPE control, with exact angles.

    The phase blocks take their angle as a call argument and apply it without
    truncation. Returns the Hadamard, the identity-phase block P(angle) and
    the feedback block RZ(angle). P(angle) = diag(1, exp(i*angle)) is exactly the controlled
    scalar phase of the Hamiltonian identity term. RZ(angle) equals P(angle)
    up to a global phase of the whole circuit, which is all a quadrature or
    feedback rotation on the uncontrolled ancilla needs.
    """
    hadamard = _make(
        "hadamard",
        "hadamard",
        basis=basis,
        reference=reference,
        payload=None,
        constructor=construct_hadamard,
        width=1,
        work=1,
        decomposition=(Primitive(gate="h", qubits=(0,)),),
        context="Hadamard on the QPE control",
    )
    parameter = Parameter(name="angle", domain="real")
    phase = _make(
        "identity_phase",
        "phase",
        basis=basis,
        reference=reference,
        payload="p",
        constructor=construct_phase,
        width=1,
        work=1,
        parameters=(parameter,),
        operations=1,
        cx=0,
        context="P(angle) on the control; original Hamiltonian identity phase",
    )
    feedback = _make(
        "feedback",
        "phase",
        basis=basis,
        reference=reference,
        payload="rz",
        constructor=construct_phase,
        width=1,
        work=1,
        parameters=(parameter,),
        operations=1,
        cx=0,
        context="RZ(angle) on the control; real/imaginary quadrature or RWPE feedback",
    )
    return hadamard, phase, feedback


def _census_envelope(count, width, config, variant, pairs, nested, triples):
    """Return (work, bytes, block) of one second-order QPE census envelope.

    Work is ``2*p*q + G + V`` (``error_budget.census_work``): p*q label reads
    for the packed masks and p*q for the support sum of ``suzuki_step_cx``,
    plus the structure and contraction charges of one selected term table.
    The census does not repeat over powers. Bytes are
    ``error_budget.census_bytes`` at the largest checked block that fits
    ``max_bytes`` (``choose_census_block``). ``_admit`` is a per-step
    gate, and ``max_bytes`` does not promise a total simultaneous cap, so
    B_held = 0 here although the QPE term records remain live. When no
    checked block fits, ``block`` is None and the bytes are those of
    block 1, a checked candidate above ``max_bytes``.
    """
    work = 2 * count * width + census_work(
        count, width, order=2, variant=variant, pairs=pairs, nested=nested, triples=triples
    )
    chosen = choose_census_block(count, width, pairs, triples, config.max_bytes,
                                 order=2, variant=variant)
    if chosen is None:
        return work, census_bytes(count, width, pairs, triples, 1, order=2, variant=variant), None
    return work, chosen[1], chosen[0]


def _census(labels, coefficients, width, config, *, reservation=(0, 0)):
    """Admit one census together with reserved common-step work and bytes.

    reservation is (R, B), computed by _common_recheck_law with W=None
    from the input shapes before any masks or pair structure exist.
    Every census candidate C requires C.work+R <= max_work and separately
    C.bytes <= max_bytes and B <= max_bytes. The census arrays are released
    before the exact stage, so their byte envelopes are not added.

    The initial admission uses E=P, N=F=J for the full variant and E=P for
    the relaxation. After pairs, price the actual E,N with F=N. Prefer
    the full expression when the combined charge fits, otherwise select
    the admitted relaxation. These are successive bounds on one census,
    not two charges. The fixed R covers either coefficient representation
    and every checked contraction block. No later common-step work gate
    can refuse an admitted candidate. Step-count and error checks remain.

    A refusal reports the kept nonidentity term count p and one sufficient
    combined work/byte limit for this admission. It names reducing p, for
    example with an explicit ``pauli_pruning_rtol`` whose dropped mass
    enters the controlled-evolution error budget, as the main way to reduce
    the p(p-1)/2 pair scan and its storage. On the exact trajectory, raising
    those limits admits a census and its exact stage, subject to the
    independent input, step-count and error requirements. Sampled and RWPE
    Plans admit their exact recheck separately at the entry of
    _independent_powers.
    The remedy names only limits below the selected candidate's sufficient
    combined allowance. Limits already meeting that allowance can stay unchanged.

    Returns:
        (evaluation, block, work), with work counting the census only.
    """
    count = len(labels)
    reserved_work, reserved_bytes = reservation

    def choose_envelope(pairs, nested):
        full = _census_envelope(count, width, config, "exact_census",
                                pairs, nested, nested)
        relaxed = _census_envelope(count, width, config, "relaxed_prefix", pairs, 0, 0)
        candidates = (("exact_census", full), ("relaxed_prefix", relaxed))
        for variant, envelope in candidates:
            work, size, block = envelope
            if (block is not None and work + reserved_work <= config.max_work
                    and max(size, reserved_bytes) <= config.max_bytes):
                config._admit(work + reserved_work, max(size, reserved_bytes))
                return variant, envelope
        byte_feasible = [item for item in candidates
                         if item[1][2] is not None
                         and max(item[1][1], reserved_bytes) <= config.max_bytes]
        variant, (work, size, _) = min(byte_feasible or candidates, key=lambda item: item[1][0])
        need_work, need_bytes = work + reserved_work, max(size, reserved_bytes)
        remedies = []
        if need_work > config.max_work:
            remedies.append(f"max_work to at least {need_work}")
        if need_bytes > config.max_bytes:
            remedies.append(f"max_bytes to at least {need_bytes}")
        header = ("QPE census and common-step admission" if reservation != (0, 0)
                  else "QPE census admission")
        raise ValueError(
            f"{header}: variant={variant}, p={count}, census_work={work}, "
            f"reserved_common_work={reserved_work}, requested_work={need_work}, "
            f"requested_bytes={need_bytes}, max_work={config.max_work}, "
            f"max_bytes={config.max_bytes}. The selected Pauli census does not fit the "
            "planning limits for p kept nonidentity terms. Its current algorithm still "
            "scans p(p−1)/2 pairs. Reducing the kept term count is the main way to reduce "
            "this work and pair storage. An explicit `pauli_pruning_rtol` can remove small "
            "terms, and the Plan includes their dropped coefficient mass in the "
            "controlled-evolution error budget. A larger pruning tolerance can exhaust that "
            "budget. The reported byte candidate is sufficient for the selected block choice "
            "and is not a minimum over every possible block size. "
            "Raise the Method's " + " and ".join(remedies)
        )

    total_pairs, nested_envelope = census_sizes(count)
    _, initial = choose_envelope(total_pairs, nested_envelope)
    chosen = {}

    def choose(pairs, nested):
        variant, envelope = choose_envelope(pairs, nested)
        chosen.update(block=envelope[2], work=envelope[0])
        return variant, envelope[2]

    evaluation = _pauli_bound_coefficient_from_terms(
        tuple(zip(labels, coefficients, strict=True)), 2, choose=choose
    )
    if not chosen:
        chosen.update(block=65536, work=initial[0])
    return evaluation, chosen["block"], chosen["work"]


def _float_shape(value):
    """Return (denominator exponent, strict magnitude exponent) for a native finite float."""
    if type(value) is not float or not isfinite(value):
        raise TypeError("common-step admission requires native finite binary64 inputs")
    numerator, denominator = value.as_integer_ratio()
    return denominator.bit_length() - 1, frexp(abs(value))[1] if numerator else 0


def _rational_bits(value):
    """Return the larger numerator or denominator bit length of a native int, float or Fraction."""
    if type(value) is int:
        return max(1, abs(value).bit_length())
    if type(value) is Fraction:
        return max(abs(value.numerator).bit_length(), value.denominator.bit_length())
    if type(value) is float and isfinite(value):
        n, d = value.as_integer_ratio()
        return max(abs(n).bit_length(), d.bit_length())
    raise TypeError("exact admission requires a native int, finite float or Fraction")


def _census_coefficient_bits(count, lower_exponent, scale_exponent):
    """Bound the numerator/denominator width of either second-order census W.

    The nonzero input magnitudes obey 2**(lower_exponent-1) <= |c| <
    2**scale_exponent. Normalization uses e=scale_exponent. Each positive
    cubic contribution, and each positive final reduction, is at least
    2**v, v=max(-1074, 3*(lower_exponent-1-e)). Thus the two final binary64
    reductions have denominator exponents at most s=min(1074,max(0,52-v)).

    For every reduction length <= 2**50, sum_up is at most four times
    the exact nonnegative sum. Normalized magnitudes are <= 1. Products
    for the pair term are <= 4, the relaxed suffix <= 4L, and its cubic
    contributions <= 16L. The block and outer sums therefore give each
    final reduction <= 256L(P+1), P=L(L-1)/2. Their strict magnitude
    exponent is U=8+bit_length(L)+bit_length(P+1). For larger populations
    use the finite-binary64 exponent U=1024. Nonfinite reductions already
    fail coefficient formation. W=(2*A12+A24)*2**(3e)/(6*2**s), so its
    reduced component widths do not exceed the returned integer.
    """
    if count < 2 or lower_exponent is None:
        return 1
    pairs, nested = census_sizes(count)
    v = max(-1074, 3 * (lower_exponent - 1 - scale_exponent))
    s = min(1074, max(0, 52 - v))
    u = (8 + count.bit_length() + (pairs + 1).bit_length()
         if max(count, pairs, nested) <= 1 << 50 else 1024)
    e3 = 3 * scale_exponent
    return max(1, s + u + 2 + max(e3, 0), s + 3 + max(-e3, 0))


def _common_recheck_law(W, loss, powers, coefficients, identity, tau, allowances,
                       max_steps):
    """Return (work, logical peak bytes, visits, integer bits) for one QPE common-step attempt.

    Inputs are the native float values already held by the QPE caller, native
    nonnegative distinct integer powers, and native rational W and loss.
    W=None reserves the stage before the census, using the component-width
    envelope of _census_coefficient_bits for either variant and every block.
    The coefficient exponent extrema reuse this function's shape scan. The
    denominator and magnitude envelopes cover the actual input exponents and
    all finite emitted values after the max_steps check. Candidate-count widths
    separately cover a candidate refused by that check. With D=2**s all ordinary
    dyadics are a/D with abs(a)<2**a_bits. The integer envelope follows rational
    selection, the five subtotal parts, residual scaling and publication.

    Premises. Binary64 round-to-nearest with gradual underflow; coefficients,
    identity, tau and allowances are finite native floats, powers are
    distinct native nonnegative integers, W and loss are native ints, floats
    or Fractions, and the allowance dictionary has precisely the power keys.
    This is not a bound for arbitrary conversion callbacks or a public
    ``common_steps`` call with arbitrary rational times.

    Widths. K counts the distinct powers including zero, L the kept
    nonidentity coefficients, P = max(1, bit_length(p_max)), w and d the
    larger numerator/denominator bit lengths of W and of the loss,
    ell = bit_length(max(1, L)), k = bit_length(max(1, K)) and
    R_M = bit_length(M) for M = max_steps. For a finite nonzero float x,
    s_x is the exponent of its reduced denominator 2**s_x and
    u_x = frexp(abs(x))[1], so abs(x) < 2**u_x; zero has both shapes zero.
    s_in, u_in bound all ordinary inputs (u_in >= 1), s_c, u_c the
    coefficients (u_c >= 0), with separate shapes for tau and the identity.
    An accepted candidate has r_max = (p_max/g)m <= M, hence h = g*tau/m >=
    tau/M, and a positive nearest-rounded h_hat is at least h/2, so
    s_h = min(1074, max(0, 54 - u_tau + R_M)) is a sufficient denominator
    exponent for the emitted step. A correctly rounded finite product of
    dyadics needs no denominator finer than the exact product's lattice
    (capped at exponent 1074), so 0.5*h_hat*c_j has exponent at most
    min(1074, s_h+s_c+1) and -(p-p_prev)*tau*identity at most
    min(1074, s_tau+s_I). With
    s = max(s_in, s_h, min(1074, s_h+s_c+1), min(1074, s_tau+s_I)) and
    u = max(u_in, min(1024, P+u_tau+1), min(1024, P+u_tau+u_c+2),
    min(1024, P+u_tau+u_I+3)), every ordinary input and every finite emitted
    value is a/D with D = 2**s and abs(a) < 2**a_bits, a_bits = max(1, s+u);
    s >= 1, so 2D**2 divides D**3. A candidate refused before emission uses
    only the inputs, which fit the same envelope: the candidate square fits
    Q = w+d+3P+3a_bits+2, and the floor, integer square root, increment and
    multiplication by p/g give R = max(P+ceil(Q/2)+2, R_M), which covers a
    count refused by the step limit. The five subtotal parts share a
    denominator dividing B_w*B_d*D**3 and their sum fits
    T = w+d+3a_bits+P+R+ell+k+5; rescaling the residual to r_max adds at
    most R+2. Hence H = w+d+3P+4a_bits+2R+2ell+2k+32 and
    b = max(2H+2, H+1076): the first term covers unreduced cross-products,
    the second comparisons with a general binary64 boundary during
    conversion and publication.

    Visits. With the shared reductions of ``error_budget._common_reductions``
    and one exact running phase prefix, V = 32L+320K+128+32K**2+4Kk: 32L for
    coefficient validation, shape inspection, half-angle construction,
    finiteness checks and the single exact mass/angle reduction; 320K+128
    for input and allowance checks, ideal selection, one phase conversion
    and exact prefix addition per power, the five scalar parts and their
    sum, residuals, publication and grid validation; 4Kk for the two sorts;
    32K**2 for integer-key hash-collision comparisons. The arithmetic
    reductions are O(L+K); the admission stays O(L+K**2) because it prices
    hostile collisions.

    Work is V*c*c*(1+ceil(log2(c))), c=ceil(bits/1075), a logical proxy rather
    than a timing guarantee. Bytes keep the prior convention
    I(b) = 32+4*ceil(b/30) and B = 65536+192L+2048K+(2L+13K+64)*I(b): 2L
    integer slots for the coefficient Fractions while the shared reductions
    are formed, 13K for six Fractions per completed prefix and its count, and
    a 64-slot frontier. The byte envelope has the qualified 64-bit CPython
    object allowance and excludes opaque allocator/bigint implementation
    scratch and process RSS. Source: NWQLib's derivation for the common-step
    selector, not a paper theorem.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive native integer")
    if any(type(p) is not int or p < 0 for p in powers):
        raise ValueError("powers must be nonnegative native integers")
    K, L = len(powers), len(coefficients)
    P = max(1, max((p.bit_length() for p in powers), default=1))
    s_tau, u_tau = _float_shape(tau)
    s_id, u_id = _float_shape(identity)
    if tau <= 0:
        raise ValueError("tau must be positive")
    s_input, u_input, s_c, u_c = max(s_tau, s_id), max(1, u_tau, u_id), 0, 0
    lower_exponent = scale_exponent = None
    for value in coefficients:
        s_value, u_value = _float_shape(value)
        if value:
            lower_exponent = u_value if lower_exponent is None else min(lower_exponent, u_value)
            scale_exponent = u_value if scale_exponent is None else max(scale_exponent, u_value)
        s_c, u_c = max(s_c, s_value), max(u_c, u_value)
        s_input, u_input = max(s_input, s_value), max(u_input, u_value)
    for value in allowances.values():
        s_value, u_value = _float_shape(value)
        s_input, u_input = max(s_input, s_value), max(u_input, u_value)
    # h >= tau/max_steps. A positive rounded h is at least h/2.
    # u_tau-1 is floor(log2(tau)), including subnormal tau.
    s_h = min(1074, max(0, 54 - u_tau + max_steps.bit_length()))
    s = max(s_input, s_h, min(1074, s_h + s_c + 1),
            min(1074, s_tau + s_id))
    u = max(u_input, min(1024, P + u_tau + 1),
            min(1024, P + u_tau + u_c + 2),
            min(1024, P + u_tau + u_id + 3))
    a_bits = max(1, s + u)
    w = (_census_coefficient_bits(L, lower_exponent, scale_exponent)
         if W is None else _rational_bits(W))
    d = _rational_bits(loss)
    ell, k = max(1, L).bit_length(), max(1, K).bit_length()
    Q = w + d + 3*P + 3*a_bits + 2
    R = max(P + (Q+1)//2 + 2, max_steps.bit_length())
    H = w + d + 3*P + 4*a_bits + 2*R + 2*ell + 2*k + 32
    # Binary64 boundary comparison/publication can use a 1075-bit denominator.
    bits = max(2*H + 2, H + 1076)
    visits = 32*L + 320*K + 128 + 32*K*K + 4*K*k
    chunks = (bits + 1074)//1075
    work = visits * chunks**2 * (1 + (chunks-1).bit_length())
    integer_bytes = 32 + 4*((bits + 29)//30)
    peak = 65536 + 192*L + 2048*K + (2*L + 13*K + 64)*integer_bytes
    return work, peak, visits, bits


def _admit_common_recheck(W, loss, powers, coefficients, identity, tau, allowances, config):
    """Admit one common-step attempt of an empty kept generator.

    select_powers calls it only when no Pauli term is kept, so there is no
    census and no census reservation (``_census``). The input-exponent
    width envelope covers candidate selection and the shared exact
    reductions before the selector runs. max_trotter_steps separately
    refuses an oversized candidate. Bytes describe the exact stage's
    logical live objects.
    """
    work, peak, visits, bits = _common_recheck_law(
        W, loss, powers, coefficients, identity, tau, allowances,
        config.max_trotter_steps,
    )
    remedies = []
    if work > config.max_work:
        remedies.append(f"max_work to at least {work}")
    if peak > config.max_bytes:
        remedies.append(f"max_bytes to at least {peak}")
    if remedies:
        raise ValueError(
            f"common-step exact arithmetic: requested_work={work}, requested_bytes={peak}, "
            f"max_work={config.max_work}, max_bytes={config.max_bytes}, "
            f"visits={visits}, integer_bits<={bits}; raise the Method's "
            + " and ".join(remedies)
        )
    return work, peak


def _chunk_cost(bits):
    """Return C(b) = c(b)**2 * (1 + ceil(log2(c(b)))), c(b) = ceil(b/1075), for one visit of b bits."""
    chunks = (bits + 1074) // 1075
    return chunks * chunks * (1 + (chunks - 1).bit_length())


def independent_recheck_work(W, loss, powers, coefficients, identity, tau,
                             allowance, max_steps):
    """Return (work, visits, integer bits) of the exact recheck in _independent_powers.

    K is the number of powers actually passed to this invocation, including
    zero. L is the number of kept nonidentity terms. The existing code forms
    half angles and performs the exact angle reduction even for power zero,
    so that entry must not be omitted from KL. Repeated shots and the two
    interference quadratures do not multiply K when they share one selected
    power. RWPE planning passes all of its max(1, max_steps) relative times
    in one call.

    Premises: native finite binary64 coefficients, tau>0, finite identity
    and allowance, native integer or finite binary64 powers, an admitted
    rational census coefficient W, and the represented pruning mass. All
    powers have already passed the schedule's range checks. A selected
    count r is checked against M=max_trotter_steps before its angles are
    emitted. A candidate refused by that check still performs exact
    inversion and must be priced. Arbitrary conversion callbacks and
    arbitrary rational public powers are outside this law.

    Work units. Each power forms and checks L half angles and evaluates the
    exact reduction A_p = 2 sum_j |val(a_pj)-val(h_p)c_j/2|, then the five
    emitted-error parts, their comparison with the allowance and the
    upward-rounded publications; the candidate inversion performs a fixed
    number of rational operations and one integer square root per active
    power. These are O(KL+K+L) logical arithmetic visits. With the
    integer-width weighting of _common_recheck_law,
    c(b)=ceil(b/1075), C(b)=c(b)^2 [1+ceil(log2(c(b)))], a visit involving
    integers of at most b bits receives C(b) work units. This is an
    engineering arithmetic proxy, not a CPU-time or exact bigint-instruction
    bound. The sufficient visit inventory is
    V_T=32L(K+1) (coefficients once plus each power's coefficient pass),
    V_S=320K+128 (per-power scalar path) and V_I=128K_a (candidate
    inversion, including a candidate later refused by M), where K_a counts
    nonzero powers if L>0 and is zero for an empty kept generator.

    Widths. p, tau, the identity coefficient, the allowance, coefficients,
    displayed time and the finite emitted h, half angles and phase are
    n/D with D=2^s, s>=1, |n|<2^a, a=s+u, derived from the native float
    shapes. For positive displayed time with frexp(time)[1]=u_t, a nonzero
    emitted h=RN(time/r) with r<=M has denominator exponent at most
    s_h = min(1074, max(0, 54-min_positive(u_t)+bit_length(M))); half
    angles at most min(1074, s_h+s_c+1) and phases at most
    min(1074, s_p+s_tau+s_I). With ell=bit_length(max(1,L)),
    r_M=bit_length(M), d the pruning mass width and w the larger of W's
    width and the _census_coefficient_bits envelope of these coefficients:
    H_T = 3a+ell+4, b_T = 2H_T+2,
    H_S = w+d+3a+ell+r_M+8, b_S = max(2H_S+2, H_S+1076, 2152),
    Q = w+3a+1075+4, R = ceil(Q/2)+2,
    H_I = w+3a+2R+8, b_I = max(2H_I+2, H_I+1076),
    W_independent = V_T C(b_T) + V_S C(b_S) + V_I C(b_I).
    H_T places the angle difference on denominator D^3 (two terms below
    2^(3a), one bit for the difference, ell for the sum, one for the factor
    two); doubling covers unreduced cross-products. The five emitted parts
    share a denominator dividing den(W)*den(loss)*D^3, the count adds at
    most r_M bits, the coefficient sum ell and the five-part sum fewer than
    three; H_S+1076 covers comparison/publication against a finite binary64
    boundary and the 2,152-bit floor scalar operations on two binary64
    values. The selector divides W*time^3 by a positive binary64 remaining
    allowance, whose numerator and denominator fit Q; the square-root
    candidate has at most R bits even above M; publishing W*time^3/r^2
    needs H_I. The law covers the work before both scientific refusals and
    the step-count refusal. It is not a wall-clock guarantee, a Plan-wide
    work total or a heap/RSS envelope.

    Source: NWQLib's derivation of the independent exact recheck
    admission; the widths reuse the coefficient-width premise of
    _census_coefficient_bits and the chunk convention of
    _common_recheck_law.
    """
    K, L = len(powers), len(coefficients)
    s_tau, u_tau = _float_shape(tau)
    s_id, u_id = _float_shape(identity)
    s_budget, u_budget = _float_shape(allowance)
    s_input = max(1, s_tau, s_id, s_budget)
    u_input = max(1, u_tau, u_id, u_budget)
    s_c = u_c = 0
    lower_exponent = scale_exponent = None
    for value in coefficients:
        s_value, u_value = _float_shape(value)
        if value:
            lower_exponent = u_value if lower_exponent is None else min(lower_exponent, u_value)
            scale_exponent = u_value if scale_exponent is None else max(scale_exponent, u_value)
        s_c, u_c = max(s_c, s_value), max(u_c, u_value)
        s_input, u_input = max(s_input, s_value), max(u_input, u_value)
    s_power, u_power, u_time = 0, 0, 0
    smallest_time_exponent = None
    active = 0
    for power in powers:
        if type(power) is int:
            sp, up = 0, abs(power).bit_length()
        else:
            sp, up = _float_shape(power)
        s_power, u_power = max(s_power, sp), max(u_power, up)
        s_input, u_input = max(s_input, sp), max(u_input, up)
        time = abs(power) * tau
        if not isfinite(time):
            raise ValueError("selected signed-power evolution time must be finite")
        st, ut = _float_shape(time)
        s_input, u_input = max(s_input, st), max(u_input, ut)
        u_time = max(u_time, ut)
        if time:
            smallest_time_exponent = (ut if smallest_time_exponent is None
                                      else min(smallest_time_exponent, ut))
        active += bool(power) and bool(L)
    r_bits = max_steps.bit_length()
    s_h = (0 if smallest_time_exponent is None else
           min(1074, max(0, 54 - smallest_time_exponent + r_bits)))
    s = max(s_input, s_h, min(1074, s_h + s_c + 1),
            min(1074, s_power + s_tau + s_id))
    u = max(u_input, min(1024, u_time + u_c + 2),
            min(1024, u_power + u_tau + u_id + 3))
    a = s + u
    ell = max(1, L).bit_length()
    w = max(_rational_bits(W), _census_coefficient_bits(L, lower_exponent, scale_exponent))
    d = _rational_bits(loss)
    h_terms = 3 * a + ell + 4
    b_terms = 2 * h_terms + 2
    h_scalar = w + d + 3 * a + ell + r_bits + 8
    b_scalar = max(2 * h_scalar + 2, h_scalar + 1076, 2152)
    candidate = w + 3 * a + 1075 + 4
    candidate_count = (candidate + 1) // 2 + 2
    h_select = w + 3 * a + 2 * candidate_count + 8
    b_select = max(2 * h_select + 2, h_select + 1076)
    visits = (32 * L * (K + 1), 320 * K + 128, 128 * active)
    bits = (b_terms, b_scalar, b_select)
    work = sum(v * _chunk_cost(b) for v, b in zip(visits, bits, strict=True))
    return work, visits, bits


def polar_base(matrix, config):
    """Return V = polar(A) = U @ Vh from one NumPy SVD of the admitted near-unitary A.

    Dense unitary selection forms one polar base and constructs controlled
    gap powers of that base. For a nonsingular admitted A the ideal target
    is V = polar(A), a unitary. If eta_A = ||A^dagger A - I|| < 1, singular
    values give ||A - V||_2 = max_j |sigma_j(A) - 1|
    <= eta_A/[1 + sqrt(1 - eta_A)], from |sigma-1| = |sigma**2-1|/(sigma+1);
    the entrywise admission implies eta_A <= D*entry_tolerance. Numerical
    polar computation contributes its own error relative to V.

    Admission. The polar stage charges 9*D**3 + 8*D**2 work units and
    192*D**2 + 160*D known-buffer bytes for complex128 input under the
    NumPy 2.5.2 wrapper inventory. The byte allowance includes the caller's
    matrix, singular-vector frames, singular values, RWORK and IWORK,
    using eight bytes per LAPACK integer, plus one conservative D-square
    matrix allowance. NumPy's separately allocated, implementation-dependent
    complex WORK array is outside this charge. Thus max_bytes is a
    known-buffer admission and does not bound the complete SVD allocation
    or process RSS. The selected factor occupies 16*D**2 persistent bytes.
    This runs once per Plan for unitary dense input. Hamiltonian input keeps
    its Hermitian spectral route.
    """
    import numpy as np

    d = len(matrix)
    config._admit(9 * d**3 + 8 * d * d,
                  max(16 * d * d + 160 * d * d + (96 + 8 * 8) * d, 64 * d * d + 8 * d) + 16 * d * d)
    left, _, right = np.linalg.svd(matrix)
    base = left @ right
    base.flags.writeable = False
    return base


def gap_powers(powers):
    """Return the positive gaps p_i - p_(i-1) of sorted distinct nonnegative powers, with p_0 = 0.

    Between powers p_(i-1) and p_i the trajectory applies controlled V
    exactly p_i - p_(i-1) times; since the two control projectors are
    orthogonal, controlled(A) controlled(B) = controlled(AB), so the
    boundary after the gap to p_i holds (|0>|phi> + |1> V**p_i |phi>)/sqrt2.
    Repeated powers have zero gaps and reuse their point, and power zero is
    algebraic.
    """
    gaps, previous = {}, 0
    for power in sorted(powers):
        gaps[power] = power - previous
        previous = power
    return gaps


@dataclass(frozen=True)
class PowerSelection:
    """The selected constructions of one Plan's powers.

    Attributes:
        powers: One QPEPower per distinct power, sorted.
        blocks: Every selected power block, each once.
        evaluation: The one Pauli census evaluation, or None.
        census_block: Contraction block actually passed to that census.
        common: The shared step record of exact-trajectory product-formula powers.
        step: The shared controlled step block, or None.
        increments: Certified binary64 identity phases. The exact
            trajectory stores the increment before each power position,
            and sampled and RWPE Plans store the whole phase for each
            independent power.
        dense: Dense power blocks by their nonnegative exponent.
    """

    powers: tuple
    blocks: tuple
    evaluation: object = None
    census_block: int | None = None
    common: object = None
    step: object = None
    increments: dict = field(default_factory=dict)
    dense: dict = field(default_factory=dict)


def select_powers(*, target, base, tau, labels, coefficients, identity, pruned_mass, powers,
                  backend, config, route):
    """Census the kept terms once and select every construction of the Plan's powers once.

    ``route`` is ``trajectory`` (the exact static quantum trajectory),
    ``sampled`` (separate static Hadamard tests with counts), ``rwpe`` or
    ``classical``.

    Product formula, trajectory, sampled and RWPE routes. One Pauli commutator census gives an
    upper bound W_up on the second-order coefficient of Childs, Su, Tran,
    Wiebe and Zhu, Phys. Rev. X 11, 011020 (2021),
    doi:10.1103/PhysRevX.11.011020, Prop. 10, Eq. (121) (arXiv:1912.08854v3
    Prop. 16, Eq. (152)), from the Pauli-triangle expansion in
    trotterization.error_budget. The census uses the full Pauli-triangle
    expression when its work plus any reserved common-step work and its
    storage fit, and otherwise the second-order suffix relaxation
    (``_census``).

    Product formula, exact static trajectory. The exact static trajectory
    uses one propagated state and keeps the common grid; only this route
    reserves common-step work. The reservation bounds the census
    coefficient width from the input exponents before any pair work. After
    this combined admission,
    ``error_budget.select_common_step`` selects one ordered second-order
    step for the nonnegative integer power positions. The ideal rational
    grid uses ``t_p=p*val(tau)`` and integer prefix counts. Persist the
    emitted binary64 step and the cumulative count at every point. Admit
    each prefix using its pruning bound, ``W*r_p*abs(val(h_hat))**3``,
    kept-generator time-displacement bound and applicable identity-phase and
    rotation-angle formation bounds. The counts are sufficient for the
    shared trajectory and need not minimize each prefix independently. Every
    power's allowance is ``controlled_power_error_budget``. A failed
    subtotal refuses planning, naming the power, the allowance and the
    contributions; no retry is made. Target increments are
    ``(p_k-p_(k-1))*val(tau)``.

    Product formula, sampled static settings and RWPE. A sampled setting
    starts from its own preparation and runs its own Hadamard circuit. It
    does not share the propagated state at earlier powers. Therefore it can
    use an independent Suzuki step count, and so does each continuous RWPE
    power. Each power has exact signed target t=val(p)*val(tau). Its
    candidate count is independently selected at the displayed positive
    time, then its emitted parameters are checked against the complete
    subtotal in _independent_powers. The step block carries its intrinsic
    formula-plus-angle bound at the represented step time. Pruning, time
    displacement and identity phase enter only the complete power bound.
    These routes reserve no common-step work and record no common step;
    their exact recheck is admitted separately at its entry.

    Dense powers. A Hamiltonian power is the exact spectral power. For a
    dense unitary input A, the selected base is its unitary polar factor V
    (``base``, ``polar_base``). Every nominal power and trajectory gap uses
    this same base. A gap of k powers is constructed by repeated squaring of
    V and realized as a controlled dense block. The represented target is
    V^p at power p. Polar selection, floating-point power formation and
    controlled synthesis contribute numerical error. Each gap is admitted
    for its actual matrix work, synthesis and live cache storage. The
    trajectory constructs one block per distinct positive gap and a sampled
    or RWPE Plan one block per distinct nonzero power.

    Returns:
        PowerSelection.
    """

    dimension = target.manifest.basis.dimension
    width = dimension.bit_length() - 1
    common_grid = route == "trajectory"
    records, blocks, dense = [], [], {}
    if backend == "dense_exact":
        gaps = gap_powers(powers) if route == "trajectory" else {}
        exponents = sorted({k for k in gaps.values() if k} if route == "trajectory"
                           else {abs(p) for p in powers if p})
        # J from every exponent the Plan constructs, admitted before the cache exists.
        squares = max((k.bit_length() - 1 for k in exponents if type(k) is int), default=0)
        matrix = (target if base is None else base).dense_array()
        reference = (target if base is None else base).manifest.reference
        for exponent in exponents:
            dense[exponent] = _dense_block(target, reference, matrix, tau, exponent, config,
                                           squares=squares)
            blocks.append(dense[exponent])
        for power in powers:
            key = gaps.get(power) if route == "trajectory" else abs(power)
            block = dense.get(key) if power else None
            records.append(QPEPower(
                power=power, backend=backend, evolution_time=_display_time(power, tau),
                steps=0, pruning_error=None, evolution_error=None, total_error=None,
                selected_definition=None if block is None else block.record.content_id))
        return PowerSelection(powers=tuple(records), blocks=tuple(blocks), dense=dense)
    if backend != "trotter_error_budgeted":
        for power in powers:
            records.append(QPEPower(
                power=power, backend=backend, evolution_time=_display_time(power, tau),
                steps=0, pruning_error=None, evolution_error=None, total_error=None))
        return PowerSelection(powers=tuple(records), blocks=())
    terms = tuple(zip(labels, coefficients, strict=True))
    evaluation = census_block = None
    allowances = ({power: config.controlled_power_error_budget for power in powers}
                  if common_grid else {})
    reservation = (_common_recheck_law(
        None, pruned_mass, tuple(allowances), coefficients, identity, tau, allowances,
        config.max_trotter_steps,
    )[:2] if common_grid and terms else (0, 0))
    if terms:
        evaluation, census_block, _ = _census(
            labels, coefficients, width, config, reservation=reservation)
    if not common_grid:
        return _independent_powers(target, tau, terms, identity, pruned_mass, powers, config,
                                   evaluation, census_block)
    if evaluation is None:
        # An empty kept generator: W = 0 and no census work.
        evaluation = _BoundCoefficientEvaluation(
            coefficient=Fraction(0), pauli_term_count=0, pair_commutation_checks=0,
            nested_commutation_checks=0, bound_variant="exact_census",
            coefficient_arithmetic=COEFFICIENT_ARITHMETIC)
    if not terms:
        _admit_common_recheck(
            evaluation.coefficient, pruned_mass, tuple(allowances), coefficients,
            identity, tau, allowances, config,
        )
    selection, prefixes, certified = _select_common_step(
        evaluation, coefficients=coefficients, identity=identity, tau=tau,
        allowances=allowances, dropped_mass=pruned_mass, max_steps=config.max_trotter_steps)
    common = step = None
    # The recheck certified these emitted increments; without a positive
    # power the only position is power 0, with no phase.
    increments = certified or {power: 0.0 for power in powers}
    if selection is not None:
        common = QPECommonStep(
            tau=selection.common_tau, g=selection.common_g, m=selection.common_m,
            step_time=selection.common_step_time, step_count=selection.step_count,
            evolution_time=selection.evolution_time, error_budget=selection.error_budget,
            bound_value=selection.bound_value)
        if terms:
            angle_per_step = prefixes[max(prefixes)][1]["angles"] / selection.step_count
            step = _step_block(
                target, terms, selection.common_step_time, evaluation, config,
                angle_per_step=angle_per_step,
            )
            blocks.append(step)
    counts = {} if selection is None else dict(selection.common_prefix_steps)
    published = {} if selection is None else dict(selection.common_prefix_bounds)
    for power in powers:
        if selection is None:
            pruning = evolution = total = 0.0
        else:
            exact, parts = prefixes[power]
            pruning = _finite_upper(parts["pruning"], "pruning error")
            evolution = _finite_upper(exact - parts["pruning"], "evolution error")
            total = published[power]
        records.append(QPEPower(
            power=power, backend=backend, evolution_time=_display_time(power, tau),
            steps=counts.get(power, 0), pruning_error=pruning, evolution_error=evolution,
            total_error=total,
            selected_definition=None if step is None or not power else step.record.content_id))
    return PowerSelection(powers=tuple(records), blocks=tuple(blocks), evaluation=evaluation,
                          census_block=census_block, common=common, step=step,
                          increments=increments)


def _display_time(power, tau):
    """Return the binary64 display of the target time |p|*val(tau), or None for unitary input.

    For an integer power the display is the correctly rounded exact product;
    an RWPE relative time keeps its binary64 product.
    """
    if tau is None:
        return None
    time = float(abs(power) * Fraction(tau)) if type(power) is int else abs(power) * tau
    if not isfinite(time):
        raise ValueError("selected power evolution time must be finite")
    return time


def _independent_powers(target, tau, terms, identity, pruned_mass, powers, config,
                        evaluation, census_block):
    """Select and certify independent QPE powers against t=val(p)*val(tau).

    The powers are the integer powers of sampled QCELS, SPE and RFE
    settings or the continuous powers of RWPE. Each is selected on its own,
    because its circuit starts from its own preparation and does not share
    a propagated state with other powers.

    Select one candidate count at the displayed positive time using the
    outward pruning subtotal. With t_p=p*val(tau), d_up the dropped
    coefficient mass and epsilon_p=controlled_power_error_budget, the ideal
    remaining allowance is epsilon_p-|t_p|*d_up. At order two the formula
    term is W_up*|t_p|**3/r_p**2, so for a positive remaining allowance the
    smallest sufficient integer count is
    r_p = max(1, ceil(sqrt(W_up*|t_p|**3/(epsilon_p-|t_p|*d_up)))),
    evaluated by the exact rational inversion in _selection_from_evaluation,
    not by floating ceil(sqrt(...)). Recheck the emitted signed step h and
    both leaf sweeps exactly. With M=sum|c_j| and
    A=2*sum|val(a_j)-val(h)*c_j/2|, its intrinsic error is W*|val(h)|**3+A.
    The complete power subtotal is
    B_p = |t|*d_up + r*(W*|val(h)|**3+A) + M*|r*val(h)-t| + |val(phi)+t*c_I|,
    with every quantity an exact rational of the stored binary64
    parameters. B_p <= epsilon_p is required and its upward rounding is
    published. phi=-p*tau*c_I is emitted separately on the control and is
    stored for the Program to use. A failed complete subtotal refuses
    without retry. The initial smallest-r claim concerns only the ideal
    formula bound at the displayed time, not the complete emitted-parameter
    subtotal. Source: the independent minimum-count rule of Childs et al.
    doi:10.1103/PhysRevX.11.011020, Sec. V B, with the second-order
    coefficient of Prop. 10, Eq. (121); docs/algorithms/qpe.md states the
    per-power selection.

    Before forming the coefficient Fractions, the recheck is admitted
    against max_work as a next-stage check (``_QPEMethod._admit``'s
    contract, not a cumulative ledger): independent_recheck_work prices
    its K powers, including zero, and L kept terms by a linear metadata
    scan without any KL Fraction reduction or commutation scan. The
    admission changes no step-selection, emitted-parameter or
    error-comparison formula.
    """
    coefficients = tuple(value for _, value in terms)
    required, visits, bits = independent_recheck_work(
        Fraction(0) if evaluation is None else evaluation.coefficient,
        pruned_mass, powers, coefficients, identity, tau,
        config.controlled_power_error_budget, config.max_trotter_steps,
    )
    if required > config.max_work:
        raise ValueError(
            f"QPE independent exact recheck needs work={required}, "
            f"K={len(powers)}, L={len(coefficients)}, visits={visits}, "
            f"integer_bits={bits}, max_work={config.max_work}. "
            f"Raise the Method's max_work to at least {required} for this recheck stage. "
            "The selected Trotter count and emitted error subtotal must "
            "still satisfy their own limits."
        )
    step_cx = suzuki_step_cx(terms) if terms else None
    current = evaluation
    records, blocks, increments = [], [], {}
    exact_coefficients = tuple(Fraction(value) for value in coefficients)
    mass = sum(map(abs, exact_coefficients), Fraction(0))
    for power in powers:
        time = abs(power) * tau
        if not isfinite(time):
            raise ValueError("selected signed-power evolution time must be finite")
        target_time = Fraction(power) * Fraction(tau)
        pruning_exact = abs(target_time) * Fraction(pruned_mass)
        pruning = _finite_upper(pruning_exact, "pruning error")
        try:
            remaining = _remaining_budget_after_pruning(
                total_budget=config.controlled_power_error_budget, pruning_error=pruning
            )
            selection = None
            if power and current is not None:
                if remaining == 0:
                    # Only an exactly zero bound fits a zero remaining allowance.
                    if current.coefficient != 0:
                        raise PruningBudgetExhausted
                    selection = TrotterStepSelection(
                        formula_order=2, evolution_time=time, error_budget=0.0,
                        bound_value=0.0, step_count=1,
                        pauli_term_count=current.pauli_term_count,
                        pair_commutation_checks=current.pair_commutation_checks,
                        nested_commutation_checks=current.nested_commutation_checks,
                        bound_variant=current.bound_variant,
                        coefficient_arithmetic=current.coefficient_arithmetic,
                    )
                else:
                    selection = _selection_from_evaluation(
                        current, time=time, error_budget=remaining, order=2
                    )
                current = replace(current, pair_commutation_checks=0, nested_commutation_checks=0)
        except PruningBudgetExhausted as error:
            raise ValueError(
                "controlled-power allowance is exhausted by the actual pruning perturbation"
            ) from error
        steps = 0 if selection is None else selection.step_count
        if steps > config.max_trotter_steps:
            raise ValueError("selected Trotter steps exceed the finite native construction envelope")
        step_time = (1 if power > 0 else -1) * time / steps if steps else 0.0
        phase = -power * tau * identity
        half_angles = tuple(0.5 * step_time * value for value in coefficients)
        if not all(isfinite(value) for value in (step_time, phase, *half_angles)):
            raise ValueError("emitted phases and angles must be finite")
        h = Fraction(step_time)
        angle_per_step = 2 * sum(
            (abs(Fraction(angle) - h * value / 2)
             for angle, value in zip(half_angles, exact_coefficients, strict=True)), Fraction(0)
        )
        W = Fraction(0) if evaluation is None else evaluation.coefficient
        exact, parts = _emitted_parts(
            W, Fraction(pruned_mass), mass, angle_per_step, Fraction(phase),
            target_time, h, Fraction(identity), steps,
        )
        if exact > Fraction(config.controlled_power_error_budget):
            raise ValueError(
                f"QPE power {power!r}: emitted subtotal {_finite_upper(exact, 'power error')!r} "
                f"exceeds controlled_power_error_budget={config.controlled_power_error_budget!r}"
            )
        evolution = _finite_upper(exact - parts["pruning"], "evolution error")
        total = _finite_upper(exact, "power error")
        step_error = _finite_upper(W * abs(h)**3 + angle_per_step, "step error")
        block = bind_power(
            target=target, terms=terms, power=power, backend="trotter_error_budgeted",
            steps=steps, step_time=step_time, step_error=step_error, config=config,
            step_cx=step_cx,
        )
        if block is not None:
            blocks.append(block)
        increments[power] = phase
        records.append(QPEPower(
            power=power, backend="trotter_error_budgeted", evolution_time=time, steps=steps,
            pruning_error=pruning, evolution_error=evolution, total_error=total,
            selected_definition=None if block is None else block.record.content_id))
    return PowerSelection(powers=tuple(records), blocks=tuple(blocks), evaluation=evaluation,
                          census_block=census_block, increments=increments)


def _step_block(target, terms, step_time, evaluation, config, *, angle_per_step):
    """Return the one controlled second-order step shared by every exact-trajectory power position.

    The step applies every kept term for step_time/2 in the given order and
    again in reverse order (``rotation_schedule``), with leaf times
    ``0.5*step_time*c_j``, the values the common-step recheck certifies.
    Two controlled Pauli rotations per term, each on at most width+1 qubits,
    give 2*L*(width+1) work units for L terms; bytes are an allowance of 32
    per term and qubit. The block's structural operator-norm error is the
    upward rounding of W_up*abs(val(step_time))**3 + angle_per_step, where
    angle_per_step is twice the exact sum of represented half-angle
    discrepancies. It bounds the emitted forward/reverse step against
    evolution of the kept nonidentity generator at the represented step
    time. Pruning, prefix time displacement and the separately emitted
    identity phase enter the complete prefix bound.
    The Program repeats the block by the
    difference of the cumulative counts between adjacent power positions.
    """
    width = target.manifest.basis.dimension.bit_length() - 1
    size = 2 * len(terms) * (width + 1)
    config._admit(size, 32 * len(terms) * (width + 1))
    return _make(
        "common_step",
        "suzuki_step",
        basis=target.manifest.basis,
        constructor=construct_suzuki_step,
        reference=target.manifest.reference,
        payload=(terms, step_time),
        width=width + 1,
        cost_parameters=(("step_time", Float64(value=step_time)),),
        work=size,
        epsilon=_finite_upper(
            evaluation.coefficient * abs(Fraction(step_time))**3 + angle_per_step,
            "step error",
        ),
        operations=2 * len(terms),
        cx=suzuki_step_cx(terms),
        context=f"shared controlled second-order step at time {step_time!r}; repeated between "
                "power positions; identity handled separately",
    )


def _dense_block(target, reference, matrix, tau, exponent, config, *, squares):
    """Return the controlled dense block of one nonnegative exponent k of the selected base.

    Construct a controlled gap power of the Plan's selected polar base. For
    a computed gap matrix B with operator-norm unitarity defect eta below
    one, its polar factor differs from B by at most eta/(1+sqrt(1-eta)).
    Error against the selected ideal power additionally includes B's
    formation error and circuit-synthesis error.

    Work counts D**3 units per product of two D-square matrices, the unit of
    _linalg_laws. For a unitary the per-exponent envelope is
    W_k = [bitlength(k) + popcount(k) - 2]*D**3 + 8*D**2 + W_synthesis; it
    remains valid with shared squares, since a cache miss cannot require
    more squares than an independent construction. For a Hamiltonian,
    V*exp(-i*E*tau*k)*V^dagger is one product, and the eigendecomposition it
    reads is computed by whichever block is constructed first, so every
    block is charged _linalg_laws.hermitian_eigensystem_work as well.
    Bytes: for a Hamiltonian an allowance of six D x D complex128 arrays
    (96*D**2); for a unitary the 96*D**2 allowance does not cover the
    persistent square cache, and the sufficient replacement is
    (96+16J)*D**2 + B_synthesis,working + B_synthesis,kept + H_cache with
    H_cache = 256(J+1) + H0, H0 = 65536, and J the largest square index of
    any exponent the Plan constructs, admitted before the cache is
    populated (``construct_dense_power``). The synthesis of the controlled
    matrix on width + 1 qubits adds the work, working bytes and kept circuit
    of _dense_synthesis.controlled_synthesis_size. Each block is admitted on
    its own; turning the reported total into a Plan-wide rejection gate is
    a separate policy (docs/algorithms/qpe.md).
    """
    dimension = target.manifest.basis.dimension
    width = dimension.bit_length() - 1
    synthesis_work, working_bytes, kept_bytes = controlled_synthesis_size(width + 1)
    if tau is None:
        matrix_work = (exponent.bit_length() + exponent.bit_count() - 2) * dimension**3
        size = ((96 + 16 * squares) * dimension**2 + working_bytes + kept_bytes
                + 256 * (squares + 1) + 65536)
    else:
        matrix_work = hermitian_eigensystem_work(dimension) + dimension**3
        size = 96 * dimension**2 + working_bytes + kept_bytes
    work = matrix_work + 8 * dimension**2 + synthesis_work
    config._admit(work, size)
    return _make(
        _power_name(exponent),
        "dense_power",
        basis=target.manifest.basis,
        constructor=construct_dense_power,
        reference=reference,
        payload=(matrix, tau, exponent),
        width=width + 1,
        cost_parameters=(("power", exponent if type(exponent) is int else Float64(value=exponent)),)
        + (() if tau is None else (("tau", Float64(value=tau)),)),
        work=work,
        epsilon=0.0,
        evidence=("exact spectral power" if tau is not None else
                  "Exact-real-arithmetic power of the selected polar base. The computed gap matrix "
                  "is realized through its unitary polar factor, with numerical polar-selection, "
                  "power-formation and synthesis errors separate."),
        context=f"explicit small dense controlled nominal power {exponent}, tau={tau!r}; cubic numerical size and exact controlled synthesis",
    )


def bind_power(*, target, terms, power, backend, steps, step_time, step_error, config,
               step_cx=None):
    """Return one independently selected sampled or RWPE power's controlled step at its step_time.

    step_time is the checked signed binary64 step time of that power.

    The Program repeats this block steps times. step_error bounds one
    emitted forward/reverse sweep against the kept nonidentity generator
    at val(step_time). Pruning, whole-power time displacement and the
    separately emitted identity phase belong to QPEPower.total_error.
    step_cx is computed once from the shared terms. Zero power and an
    empty kept generator need no block.
    """
    dimension = target.manifest.basis.dimension
    width = dimension.bit_length() - 1
    if backend == "trotter_error_budgeted" and steps:
        # Two controlled Pauli rotations per term (rotation_schedule), each on
        # at most width+1 qubits, give 2*L*(width+1) work units for L terms.
        # Bytes: an allowance of 32 per term and qubit.
        size = 2 * len(terms) * (width + 1)
        config._admit(size, 32 * len(terms) * (width + 1))
        return _make(
            _power_name(power), "suzuki_step", basis=target.manifest.basis,
            constructor=construct_suzuki_step, reference=target.manifest.reference,
            payload=(terms, step_time), width=width + 1,
            cost_parameters=(
                ("power", power if type(power) is int else Float64(value=power)),
                ("steps", steps), ("step_time", Float64(value=step_time)),
            ),
            work=size, epsilon=step_error, operations=2 * len(terms), cx=step_cx,
            context=f"controlled second-order step at signed time {step_time!r}; repeat {steps} "
                    f"times for power {power}; identity handled separately",
        )
    return None


def _power_name(power):
    return f"power_{'m' if power < 0 else 'p'}{abs(power)}"


def construct_hadamard(block, arguments, method_context):
    """Build the one-qubit Hadamard circuit of the ancilla."""
    if arguments:
        raise ValueError("Hadamard has no scalar arguments")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1)
    circuit.h(0)
    return circuit


def construct_phase(block, arguments, method_context):
    """Build P(angle) or RZ(angle) on one qubit, as named by the block payload ``p`` or ``rz``."""
    if len(arguments) != 1 or arguments[0][0] != "angle":
        raise ValueError("QPE phase requires its exact admitted angle")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1)
    getattr(circuit, block._payload)(float(arguments[0][1]), 0)
    return circuit


def rotation_schedule(terms, step_time):
    """Ordered second-order rotations consumed directly by native construction.

    One step of the second-order Suzuki formula S2(dt) applies every term for
    dt/2 in the given order and then again in reverse order. Term 1 acts
    first, which is the ordering for which Childs et al., Phys. Rev. X 11,
    011020 (2021), doi:10.1103/PhysRevX.11.011020, state Prop. 10
    (arXiv:1912.08854v3 Prop. 16). Their gamma = 1 factor is rightmost.
    PauliEvolutionGate(P, time=a) applies exp(-i*a*P), so the entry
    (label, dt*c/2) applies exp(-i*dt*c*P/2).

    Returns:
        Tuple of 2*L (label, angle) pairs for L terms, in application order.
    """
    half_step = tuple((label, 0.5 * step_time * coefficient) for label, coefficient in terms)
    return (*half_step, *reversed(half_step))


def suzuki_step_cx(terms):
    """Count CX gates in one selected QPE controlled Suzuki step.

    Let w_j be the number of nonidentity symbols in nonidentity Pauli term
    j, and W_support = sum_j w_j over the terms after splitting and pruning.
    rotation_schedule emits each term twice. Write R_P(theta) =
    exp(-i*theta*P/2). A single-qubit basis change B maps each active factor
    of P to Z, and a parity chain A of w - 1 CX gates maps the product of
    those Z factors to one target Z, so with V = A*B,
    R_P(theta) = V^dagger R_Z(theta) V. For one control,
    C(R_P(theta)) = (I x V^dagger) CR_Z(theta) (I x V): on the inactive
    control branch V^dagger V = I, so the basis changes and parity chains
    stay uncontrolled. The central CR_Z(theta) takes two CX, as
    R_Z(theta/2), CX, R_Z(-theta/2), CX, since X R_Z(-theta/2) X =
    R_Z(theta/2). Thus the basis changes cost zero CX, the parity chain and
    its inverse 2*(w_j - 1), the central CRZ two, the controlled leaf
    2*w_j, and the controlled second-order step 4*W_support.

    construct_suzuki_step uses qiskit_compat.controlled on a one-term
    PauliEvolutionGate. Qiskit 2.5.2 controls its central rotation through
    the projector representation (``PauliEvolutionGate.control`` in
    qiskit/circuit/library/pauli_evolution.py, lowered by
    crates/synthesis/src/pauli_evolution.rs), leaving the parity and basis
    networks uncontrolled. The corresponding uncontrolled leaf costs
    2*(w_j - 1), and the uncontrolled schedule 4*(W_support - L).

    These are exact decomposition counts for CX plus arbitrary single-qubit
    gates, qualified with Qiskit 2.5.2 and optimization level 0 without a
    coupling map or backend target. They exclude cancellation, rotation
    merging, alternative synthesis and routing. Angles and power signs do
    not change the emitted count. Identity evolution is a separate ancilla
    P(-power*tau*identity_coefficient), which costs zero CX, as do the
    one-qubit feedback RZ and Hadamard gates.

    select_powers computes W_support once in O(L*q) label reads and uses
    constant scalar work to instantiate each power's law. The value is per
    step. Repeat supplies the steps multiplier, and the workload supplies
    repetitions. The law binds the exact selected power, steps and
    step_time cost parameters (``_make``). The law does not apply to the
    separately selected dense-power synthesis.
    """
    return 4 * sum(len(label) - label.count("I") for label, _ in terms)


def construct_suzuki_step(block, arguments, method_context):
    """Build one controlled S2 step as a sequence of controlled Pauli exponentials.

    Controlling every factor equals controlling their product, so the step
    needs no controlled global phase. The identity phase is applied
    separately on the control. The reverse sweep of ``rotation_schedule``
    repeats the forward (label, angle) entries, so each of the L controlled
    leaves is constructed once and appended in both sweeps.
    """
    if arguments:
        raise ValueError("selected QPE step has no unresolved scalar arguments")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.subroutines.qiskit_compat import controlled

    terms, step_time = block._payload
    circuit = QuantumCircuit(sum(port.width for port in block.record.signature.quantum))
    schedule = rotation_schedule(terms, step_time)
    leaves = [
        controlled(PauliEvolutionGate(SparsePauliOp.from_list([(label, 1.0)]), time=time), 1)
        for label, time in schedule[: len(terms)]
    ]
    for leaf in (*leaves, *reversed(leaves)):
        circuit.append(leaf, circuit.qubits, copy=False)
    return circuit


def cached_unitary_power(matrix, power, squares):
    """Return matrix**power from the binary squares in ``squares``, appending missing squares.

    Reuse binary squares of the same admitted unitary within the Plan
    context, preserving NumPy's multiplication order including its
    power-three case. Admission includes every cached square that can remain
    live during native construction.

    The cache holds Z_0 = U (aliasing the owned matrix) and
    Z_j = Z_(j-1) @ Z_(j-1). For n = abs(power) > 0 with j = bit_length(n)-1
    and j_old the highest square cached before the call, it needs
    [max(0, j-j_old) + popcount(n) - 1]*D**3 of matrix work, and across calls
    [J + sum_calls(popcount(abs(p)) - 1)]*D**3 with J the largest square
    index. For n = 3 NumPy's special product Z_1 @ Z_0 is kept, since the
    general least-significant-bit accumulation Z_0 @ Z_1 need not round
    identically. The multiplication order is that of NumPy 2.5.2's
    ``matrix_power`` (numpy/linalg/_linalg.py), so every result keeps its
    bits. Negative powers follow the conjugate-transpose convention, not
    NumPy's inverse path, and power zero is algebraic.
    """
    import numpy as np

    n = abs(power)
    if n == 0:
        return np.eye(len(matrix), dtype=matrix.dtype)
    while len(squares) < n.bit_length():
        squares.append(squares[-1] @ squares[-1])
    if n == 3:
        result = squares[1] @ squares[0]
    else:
        result = None
        for j in range(n.bit_length()):
            if (n >> j) & 1:
                result = squares[j] if result is None else result @ squares[j]
    return result if power > 0 else result.conj().T


def construct_dense_power(block, arguments, method_context):
    """Build the controlled dense power of one exponent of the selected base.

    Construct a controlled gap power of the Plan's selected polar base. For
    a computed gap matrix B with operator-norm unitarity defect eta below
    one, its polar factor differs from B by at most eta/(1+sqrt(1-eta)).
    Error against the selected ideal power additionally includes B's
    formation error and circuit-synthesis error.

    For a Hamiltonian, U^p = V diag(exp(-i*E*tau*p)) V^dagger from the
    eigensystem kept in the method context, so every power and the host
    signal reuse one eigendecomposition. For unitary input the payload
    matrix is the selected polar base, and its power comes from the binary
    squares kept in the method context under the base's identity
    (``cached_unitary_power``); the squares are a recomputable construction
    cache and are not saved with a Result. ``qiskit_compat.controlled``
    synthesizes the unitary polar factor of the controlled matrix to
    binary64 rounding (``_dense_synthesis.controlled_unitary_circuit``). On
    m = n + 1 >= 3 qubits for n system qubits that takes at most
    ``(25/96) 4**m - 2**m + 4/3`` CX.
    """
    if arguments:
        raise ValueError("selected dense QPE power has no unresolved scalar arguments")
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from nwqlib.subroutines.qiskit_compat import controlled
    from .method import _new_context, _eigensystem

    matrix, tau, power = block._payload
    if tau is None:
        context = method_context(_new_context)
        key = block.record.semantics.input
        if context["squares"] is None or context["squares"][0] != key:
            context["squares"] = (key, [matrix])
        unitary = cached_unitary_power(matrix, power, context["squares"][1])
    else:
        (eigenvalues, eigenvectors), _ = _eigensystem(
            block.record.semantics.input,
            matrix,
            kind="hamiltonian",
            context=method_context(_new_context),
        )
        unitary = (eigenvectors * np.exp(-1j * eigenvalues * tau * power)) @ eigenvectors.conj().T
    # A computed gap matrix B need not be exactly unitary, and check_input=False
    # does not change it into one: it only skips Qiskit's constructor check of
    # B^dagger B (numpy.allclose with the identity, atol 1e-8 and rtol 1e-5),
    # and the controlled synthesis realizes the polar factor of B.
    gate = controlled(UnitaryGate(unitary, label=f"U^{power}", check_input=False), 1)
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, circuit.qubits, copy=False)
    return circuit
