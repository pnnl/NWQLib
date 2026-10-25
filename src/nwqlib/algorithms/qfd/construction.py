"""The fixed product-formula step V of QFD, its evolution blocks and the base-power pair rule.

V is one frozen ordered sequence of Pauli rotations ``R_P(theta) =
exp(-i theta P/2)`` built once per Plan. By default its factors are the
nonidentity rows of the centered Hamiltonian ``A0 = A - c_I I``. With an
explicit fermionic generator B they are the Jordan-Wigner leaves of the
complete Hermitian blocks of the centered ``B0 = B - c_B I``
(``fermionic_blocks`` and ``fermionic_step``). Every basis state is an
integer power of the same V applied to the reference, and a negative power
applies the exact adjoint of that sequence: the factors in reverse order
with negated angles. Circuits are built only when a Qiskit circuit is
requested.
"""

from fractions import Fraction

import numpy as np

from nwqlib.blocks.records import BlockSemantics, SelectedDefinition
from nwqlib.blocks.selection import SelectedBlock
from nwqlib.core.records import Source
from nwqlib.evidence import Evidence
from nwqlib.ir import BlockSignature, QuantumPort
from nwqlib.operators.inputs import _basis
from nwqlib.resources import ResourceLaw

EVOLUTION = Source(
    name="qfd.pauli_evolution",
    version="1",
    domain="fixed product-formula step of a centered Pauli generator, uncontrolled or controlled",
    reference="nwqlib.algorithms.qfd.construction; R_P(theta) = exp(-i theta P/2) per factor",
)
# The block families of a nonzero canonical monomial, keyed by its number of
# ladder operators and of distinct modes, and the power kappa of the common
# leaf weight 2**-kappa of its unit Hermitian components under Jordan-Wigner:
# number n_p and hopping (1), density n_p n_q and hopping with one spectator
# number (2), double excitation on four distinct modes (3).
_FAMILY_KAPPA = {(2, 1): 1, (2, 2): 1, (4, 2): 2, (4, 3): 2, (4, 4): 3}
# A unit block maps to at most 2**4 = 16 distinct Pauli words, and
# 16 * 15 / 2 = 120 pair tests check that they commute.
_BLOCK_ROWS = 16
# The CX law is the decomposition count of apply_pauli_rotation: a basis
# change and parity ladder of 2(w - 1) CX around RZ(theta), with the ladder
# left uncontrolled under control, where RZ becomes a two-CX CRZ. RX, RY and
# RZ cost no CX, and RXX, RYY and RZZ two each.
_CX_SOURCE = Source(
    name="qfd.pauli_rotation_cx",
    version="1",
    domain="uncontrolled and controlled Pauli rotations of apply_pauli_rotation",
    reference="2(w - 1) CX per uncontrolled and 2w per controlled rotation of a weight-w Pauli word, "
              "for Qiskit 2.6.0 at optimization level 0 without routing",
)
# Stored text, part of Plan content identity.
_CX_ASSUMPTIONS = (
    "Qiskit 2.6.0 lowering to cx and one-qubit gates at optimization level 0, "
    "before cancellation, rotation merging and routing, and another Qiskit version can differ",
)
_SLOTS_SOURCE = Source(
    name="qfd.selected_slots",
    version="1",
    domain="Pauli rotation instructions of apply_pauli_rotation",
    reference="one instruction per appended basis change, parity CX and rotation; no native basis prediction",
)


def step_factors(rows, substep, order):
    """Return the row indices and emitted angles of one product-formula substep, in execution order.

    ``rows`` are the nonidentity ``(label, coefficient)`` rows of the centered
    Hamiltonian in stored order and ``substep`` the emitted binary64 step
    ``tau/r``. Order 2 is the S2 schedule of
    ``qpe.powers.rotation_schedule``: every row for half the step, then
    every row again in reverse order. Its exponent parameter
    ``beta = dt c/2`` of ``exp(-i beta P)`` is the half angle of
    ``R_P(theta)``, so ``theta = 2 beta = c dt``. Order 1 applies every row
    once for the full step, ``theta = 2 c dt``. Each angle is rounded once,
    and the doubling is exact.
    """
    if order == 2:
        from nwqlib.algorithms.qpe.powers import rotation_schedule

        count = len(rows)
        indices = (*range(count), *reversed(range(count)))
        angles = tuple(2.0 * beta for _, beta in rotation_schedule(rows, substep))
    else:
        indices = tuple(range(len(rows)))
        angles = tuple(2.0 * (substep * coefficient) for _, coefficient in rows)
    return np.array(indices, dtype=np.int64), np.array(angles, dtype=np.float64)


def canonical_order(key):
    """Return the sort key of a normal-ordered ladder monomial: its length, then its ``(mode, action)`` tuple."""
    return len(key), key


def canonical_partner(key):
    """Return ``(T_bar, s_T)`` with ``T^dagger = s_T T_bar`` for a canonical normal-ordered monomial T.

    The adjoint reverses the ladder word and exchanges creation and
    annihilation (``fermionic_pool._hermitian_conjugate_ops``), and
    ``fermionic_pool._normal_ordered_term`` restores the canonical order
    with its anticommutation sign. For the scalar, one-body and two-body
    monomials of ``fermionic_blocks`` each group of k = 1 or 2 operators of
    one kind comes back in ascending mode order and needs k(k-1)/2 swaps, so
    s_T = +1, and no creation passes an annihilation, so no contraction
    term arises. The sign is still taken from the owner, not assumed.
    """
    from nwqlib.subroutines.fermionic_pool import _hermitian_conjugate_ops, _normal_ordered_term

    ((partner, sign),) = _normal_ordered_term(_hermitian_conjugate_ops(key), 1).items()
    return partner, int(sign.real)


def _commute(left, right):
    """Return whether two Pauli labels commute: an even number of positions with different non-identity letters."""
    return sum(a != "I" != b != a for a, b in zip(left, right, strict=True)) % 2 == 0


def fermionic_blocks(operator, *, max_bytes):
    """Return the generator offset ``c_B`` and the complete number-conserving Hermitian blocks of a fermionic generator B.

    ``operator`` is a fermion-representation ``OperatorInput`` on q modes,
    read in the occupation convention of ``operators._fermion``: mode j is
    qubit j, ``|1>`` is occupied, and the Jordan-Wigner images are
    ``a_j^dagger = (prod_{k<j} Z_k)(X_j - i Y_j)/2`` and
    ``a_j = (prod_{k<j} Z_k)(X_j + i Y_j)/2`` (P. Jordan and E. Wigner,
    Z. Phys. 47, 631 (1928), doi:10.1007/BF01331938). Its coefficients are
    read in the energy unit of the target and used as given.

    Domain. Every row is a scalar, a one-body ``a_p^dagger a_q`` or a
    two-body ``a_p^dagger a_q^dagger a_r a_s``, written with its creations
    left of its annihilations. Any other ladder word is refused. The rows
    are normal ordered with exact anticommutation signs and identical words
    are summed by ``fermionic_pool._normal_ordered_operator`` with
    ``coefficient_cutoff=0``, so only exact zeros are removed. Within each
    kind the canonical order puts higher modes first, so the canonical
    monomials T are the scalar, ``a_p^dagger a_q`` and
    ``a_p^dagger a_q^dagger a_r a_s`` with p > q and r > s, and a repeated
    creation or annihilation gives zero. The normal-order signs carry the
    repeated-index reductions, for example
    ``a_p^dagger a_q^dagger a_p a_s = -n_p a_q^dagger a_s`` and
    ``a_p^dagger a_q^dagger a_p a_q = -n_p n_q``.

    Hermitian pairing. Write ``T^dagger = s_T T_bar`` with T_bar canonical
    (``canonical_partner``) and let g_T be the summed coefficient of T. The
    operator is Hermitian at the coefficient level exactly when
    ``g_bar = s_T conj(g_T)`` for every T, with ``g_bar = 0`` when T_bar is
    absent, which for a self-adjoint T (T_bar = T, s_T = +1) is a real
    coefficient. Any other table is refused without repair. Of a distinct
    pair the side first in ``canonical_order`` is taken once as T, and with
    ``g_T = u + i v``

        g_T T + g_bar T_bar = u (T + T^dagger) + v i (T - T^dagger),

    two Hermitian components that do not commute in general. Each with a
    nonzero amplitude is a block of its own. A self-adjoint T gives the one
    block ``Re(g_T) T``, without a factor 2, and the scalar row is added to
    c_B.

    Leaves. Each unit component, ``T + T^dagger``, ``i (T - T^dagger)`` or a
    self-adjoint T, is mapped once by ``FermionTerms.to_pauli`` to
    ``c_0 I + sum_k eps_k 2**-kappa P_k`` with eps_k = +1 or -1 and the
    family's kappa: 1 for a number term ``n_p = (I - Z_p)/2`` and a hopping
    term, 2 for a density ``n_p n_q`` and a hopping term times a spectator
    number, 3 for a double excitation on four distinct modes. Every word
    must have the real coefficient +-2**-kappa exactly. The identity part
    c_0, nonzero only for number and density terms, adds ``amplitude c_0``
    to c_B. The words of a block commute: in a hopping or double-excitation
    component the active modes carry X or Y and the other modes a fixed Z
    string, the real and the imaginary component each keep one parity of
    the Y count, so two words differ by X/Y on an even number of modes, and
    a spectator number ``(I - Z_r)/2`` and a density term add only Z
    letters. Each block is checked: at most 16 distinct words and, with at
    most 120 symplectic pair tests, pairwise commutation. This
    fragment-level grouping by an Abelian symmetry is that of Leal-Sanchez
    et al., arXiv:2607.01560v1, Fig. 1b. The signs and weights above come
    from the anticommutation relations and the ladder images, not from that
    paper's Majorana expansion. Each block conserves the particle number,
    since every T has as many creations as annihilations.

    Work. Each component maps at most two words of at most four ladder
    operators, through the bounded mapping of ``to_pauli`` with
    ``max_bytes``. No dense operator or commutator count over B is formed.

    Returns:
        ``(c_B, blocks)``. c_B is the exact rational identity coefficient of
        B. ``blocks`` lists ``(kappa, amplitude, leaves)`` in the canonical
        block order, T in ``canonical_order`` and its real component before
        its imaginary one, with the binary64 amplitude u, v or Re(g_T) and
        the leaves ``(label, eps)`` in increasing label order.

    Raises:
        ValueError: For a row outside the domain, a table that is not
            exactly Hermitian, or a block that fails its checks.
    """
    from nwqlib.operators._fermion import fermion_table
    from nwqlib.subroutines.fermionic_pool import _normal_ordered_operator

    terms = operator.fermion_terms()
    q, offsets, modes = terms.num_modes, terms.offsets.tolist(), terms.modes.tolist()
    actions = terms.actions.tolist()
    rows = []
    for k, coefficient in enumerate(terms.coefficients.tolist()):
        word = tuple(zip(modes[offsets[k]:offsets[k + 1]], actions[offsets[k]:offsets[k + 1]], strict=True))
        if tuple(action for _, action in word) not in ((), (1, 0), (1, 1, 0, 0)):
            raise ValueError(
                f"QFD evolution='fermionic_blocks' accepts scalars, a_p^dagger a_q and a_p^dagger a_q^dagger a_r a_s "
                f"with creations left of annihilations, and evolution_terms row {k} is {word} as (mode, action) "
                "pairs. Use evolution='pauli_terms' for other operators, without a particle-number qualification.")
        rows.append((coefficient, word))
    table = _normal_ordered_operator(rows, coefficient_cutoff=0)
    shift, blocks, paired = Fraction(0), [], set()
    for key in sorted(table, key=canonical_order):
        if key in paired:
            continue
        partner, sign = canonical_partner(key)
        coefficient, other = table[key], table.get(partner, 0j)
        if other != sign * coefficient.conjugate():
            raise ValueError(
                f"QFD evolution_terms are not exactly Hermitian: the term {key} has the coefficient {coefficient!r} "
                f"and its adjoint term {partner}, with sign {sign}, has {other!r} instead of "
                f"{sign * coefficient.conjugate()!r}. Supply exactly conjugate coefficients.")
        paired.add(partner)
        if not key:
            shift += Fraction(coefficient.real)
            continue
        components = ((coefficient.real, ((1, key),)),) if partner == key else (
            (coefficient.real, ((1, key), (sign, partner))),
            (coefficient.imag, ((1j, key), (-1j * sign, partner))))
        kappa = _FAMILY_KAPPA[len(key), len({mode for mode, _ in key})]
        for amplitude, unit in components:
            if amplitude == 0:
                continue
            mapped = fermion_table(unit, num_modes=q, max_bytes=max_bytes).to_pauli(mapping="jw", max_bytes=max_bytes)
            leaves = []
            for label, value in mapped.pauli_terms().labels():
                if value.imag != 0 or label.count("I") < len(label) and abs(value.real) * 2**kappa != 1:
                    raise ValueError(f"QFD fermionic block of {key} maps to {label} with {value!r}, not a real "
                                     f"coefficient of magnitude 2**-{kappa}")
                if label.count("I") == len(label):
                    shift += Fraction(amplitude) * Fraction(value.real)
                else:
                    leaves.append((label, 1 if value.real > 0 else -1))
            leaves.sort()
            if len(leaves) > _BLOCK_ROWS or not all(
                    _commute(a, b) for i, (a, _) in enumerate(leaves) for b, _ in leaves[i + 1:]):
                raise ValueError(f"QFD fermionic block of {key} does not map to at most {_BLOCK_ROWS} commuting "
                                 "Pauli words")
            blocks.append((kappa, amplitude, tuple(leaves)))
    return shift, blocks


def fermionic_step(shift, blocks, substep, order, num_qubits):
    """Return B's evolution table and the rows, angles and block starts of one product-formula substep.

    ``shift`` and ``blocks`` come from ``fermionic_blocks``, ``substep`` is
    the emitted binary64 substep ``tau/r`` and ``order`` is 1 or 2. A block
    ``B_b = u sum_k eps_k 2**-kappa P_k`` is a sum of commuting words, so
    its exponential for the time s is exactly the product of the rotations
    ``R_P(theta) = exp(-i theta P/2)`` of its words with
    ``theta_bk = eps_k theta_b``,

        theta_b = RN(2 s val(u) 2**-kappa),

    rounded once from the exact rational product with
    ``s = val(substep)`` for order 1 and ``val(substep)/2`` for order 2. All
    words of a block share the one rounded magnitude, so the emitted block
    is again a scalar multiple of the same Hermitian component and keeps
    the particle number. An angle that rounds to 0 gives the identity block
    and keeps its place. No angle is reduced modulo 2 pi, because
    ``R_P(theta + 2 pi) = -R_P(theta)`` changes a controlled phase. A
    product too large for binary64 refuses the step.

    Order 1 applies every block once in the canonical block order. Order 2
    applies every block for half the substep and then the reverse of that
    factor sequence, so the blocks come again in reverse block order, each
    with its words in reverse order. The words of a block are contiguous and
    commute, so either word order gives the same block exponential, and no
    block is split or interleaved with another. One substep has L factors
    for order 1 and 2L for order 2, with ``L = sum_b m_b`` the words of all
    blocks, a word counted once per block that contains it.

    Returns:
        ``(table, step_rows, step_angles, step_blocks, generator)``.
        ``table`` is the ``PauliArrays`` of the identity row c_B and of every
        distinct word of a block, in increasing label order, with the
        coefficient of B rounded once from its exact sum. A word whose sum
        is exactly zero keeps its row, because a factor refers to it.
        ``step_rows`` are int64 positions in the nonidentity rows of
        ``table``, ``step_angles`` float64 angles and ``step_blocks`` the
        int64 position in ``step_rows`` where each block of the substep
        starts, all in execution order. ``generator`` maps each label of
        ``table`` to its exact rational coefficient.

    Raises:
        ValueError: If an angle overflows binary64.
    """
    from nwqlib.algorithms.gcim.fixed_basis import PauliArrays

    s = Fraction.from_float(substep) / (2 if order == 2 else 1)
    generator, sweep = {"I" * num_qubits: shift}, []
    for kappa, amplitude, leaves in blocks:
        weight = Fraction.from_float(amplitude) / 2**kappa
        try:
            angle = float(2 * s * weight)
        except OverflowError:
            raise ValueError(f"QFD fermionic block angle 2 s u 2**-{kappa} overflows binary64 for the substep "
                             f"{substep!r} and the amplitude {amplitude!r}") from None
        sweep.append([(label, eps * angle) for label, eps in leaves])
        for label, eps in leaves:
            generator[label] = generator.get(label, 0) + eps * weight
    if order == 2:
        sweep += [block[::-1] for block in reversed(sweep)]
    labels = sorted(generator)
    position = {label: index - 1 for index, label in enumerate(labels)}
    factors = [factor for block in sweep for factor in block]
    starts = np.cumsum([0] + [len(block) for block in sweep[:-1]], dtype=np.int64) if sweep else ()
    table = PauliArrays.from_rows(((label, float(generator[label])) for label in labels), num_qubits)
    return (table, np.array([position[label] for label, _ in factors], dtype=np.int64),
            np.array([angle for _, angle in factors], dtype=np.float64), np.array(starts, dtype=np.int64),
            generator)


def generator_difference(target, generator):
    """Return the upper bounds ``(delta_gen_0, delta_gen_full)`` on the distance of a generator B from the target A.

    ``target`` holds the ``(label, coefficient)`` rows of A, its identity
    row included, and ``generator`` maps the labels of B, its identity
    label included, to exact rational coefficients. Over the union of the
    nonidentity labels

        delta_gen_0 = sum_{P != I} |a_P - b_P| >= ||A0 - B0||,
        delta_gen_full = |c_I - c_B| + delta_gen_0 >= ||A - B||,

    by the triangle inequality, since every Pauli word has norm 1. Both
    sums are exact rationals of the binary64 values, each rounded up once
    (``pair_reducer.upward``), so the returned numbers are upper bounds.
    The work is one pass over the two sparse tables. They bound coefficient
    differences of A and B only. They are not the distance of V from
    ``exp(-i tau A0)`` and not an error bar of an energy.
    """
    from nwqlib.algorithms.gcim.pair_reducer import upward

    identity = "I" * len(next(iter(generator)))
    a = {label: Fraction(coefficient) for label, coefficient in target}
    zero = Fraction(0)
    nonidentity = sum((abs(a.get(label, zero) - generator.get(label, zero))
                       for label in set(a) | set(generator) if label != identity), zero)
    offset = abs(a.get(identity, zero) - generator[identity])
    return upward(nonidentity), upward(offset + nonidentity)


def _rotation_counts(label, controlled):
    """Return the ``(instructions, cx)`` that ``apply_pauli_rotation`` appends for one Pauli word.

    Without control a weight-one word is one RX, RY or RZ and a weight-two
    word of one repeated letter one RXX, RYY or RZZ. Otherwise an X needs
    two H, a Y an S^dagger, two H and an S, and the parity ladder and its
    inverse 2(w - 1) CX around one RZ or CRZ.
    """
    letters = [letter for letter in label if letter != "I"]
    w = len(letters)
    if not controlled and (w == 1 or (w == 2 and letters[0] == letters[1])):
        return 1, 2 * (w - 1)
    basis = sum(2 if letter == "X" else 4 if letter == "Y" else 0 for letter in letters)
    return basis + 2 * (w - 1) + 1, 2 * (w - 1) + 2 * int(controlled)


def evolution_block(kind, rows, step_rows, step_angles, steps, operator, num_qubits):
    """Return the selected block of V, V^dagger or one of their controlled forms.

    ``kind`` is ``"forward"``, ``"adjoint"``, ``"controlled_forward"`` or
    ``"controlled_adjoint"``, ``rows`` the nonidentity rows of the
    centered generator (``QFDReconstruction.evolution_rows``), ``step_rows``
    and ``step_angles`` one substep's factors (``step_factors`` or
    ``fermionic_step``), ``steps`` the substep count r of V and
    ``operator`` the operator input the factors come from, on
    ``num_qubits`` system qubits: the processed target A, or the fermionic
    ``evolution_terms`` B. The adjoint reverses the factor order of
    the whole step and negates every angle. A controlled block has a first
    one-qubit port ``control`` and controls each rotation's RZ, which is
    the controlled V because every factor is the exact rotation, without a
    global phase. Its cost rules give the exact instruction and CX counts
    of the constructor (``_rotation_counts``) for all r substeps.
    """
    controlled, adjoint = kind.startswith("controlled"), kind.endswith("adjoint")
    from nwqlib.subroutines.hamiltonian_evolution.pauli_labels import make_pauli_label

    offset = int(controlled)
    labels, angles, instructions, cx = [], [], 0, 0
    for row, angle in zip(step_rows.tolist(), step_angles.tolist(), strict=True):
        label = rows[row][0]
        labels.append(make_pauli_label({k + offset: a for k, a in enumerate(reversed(label)) if a != "I"}))
        angles.append(angle)
        slots, gates = _rotation_counts(label, controlled)
        instructions += slots
        cx += gates
    if adjoint:
        labels, angles = labels[::-1], [-angle for angle in angles[::-1]]
    ports = ((QuantumPort(name="control", width=1),) if controlled else ()) + (
        QuantumPort(name="system", width=num_qubits),)
    semantics = BlockSemantics(
        kind="unknown",
        input=operator.reference,
        basis=_basis(1 << num_qubits),
        relation=f"{'controlled ' if controlled else ''}{'adjoint of ' if adjoint else ''}"
                 f"V = S({steps} substeps), the stored ordered Pauli rotations exp(-i theta P/2)",
        input_projector="identity on the selected register",
        output_projector="identity on the selected register",
        success="deterministic selected unitary action",
        workspace=0,
        restoration="no additional workspace",
        epsilon=None,
        approximation_metric="unitary operator norm before native floating synthesis",
        approximation_evidence=(
            "the product formula of the complete fermionic blocks of B0 = B - c_B I, saved with the Plan, defines the "
            "basis, and its distance from exp(-i tau A0) is not evaluated"
            if operator.reference.representation == "fermion" else
            "the stored product formula defines the basis; its distance from exp(-i tau A0) is not evaluated"),
        inverse_legal=False,
        control_legal=False,
        phase="no global phase: every factor is exactly exp(-i theta P/2); a control acts on each central RZ",
    )
    laws = (
        ResourceLaw(metric="operations", basis="selected_logical", value=steps * instructions,
                    interpretation="exact", evidence=Evidence(kind="proved_relation", source=_SLOTS_SOURCE),
                    assumptions=("logical constructor slots; HLS/native basis inventory is separate",)),
        ResourceLaw(metric="cx", basis="cx", value=steps * cx, interpretation="exact",
                    evidence=Evidence(kind="proved_relation", source=_CX_SOURCE), assumptions=_CX_ASSUMPTIONS),
    )
    name = ("controlled_" if controlled else "") + "evolution" + ("_adjoint" if adjoint else "")
    record = SelectedDefinition(
        signature=BlockSignature(name=name, target=EVOLUTION, quantum=ports),
        semantics=semantics,
        implementation=EVOLUTION,
        choice=kind,
        decomposition=None,
        cost_law=None,
        resource_laws=laws,
        cost_context="QFD fixed product-formula step; logical construction size excludes unknown SDK "
                     "synthesis/allocator overhead",
        construction_work=steps * instructions,
    )
    return SelectedBlock.bind(record, payload=(tuple(labels), tuple(angles), steps, controlled),
                              constructor=construct_evolution)


def construct_evolution(block, arguments, method_context):
    """Build the circuit of one evolution block: its r substeps of stored Pauli rotations.

    The payload holds the compact labels and angles in execution order,
    already reversed and negated for an adjoint and shifted past the control
    qubit 0 for a controlled block, the substep count and whether qubit 0
    controls each rotation. An empty factor list is V = I, from a generator
    whose centered part has no block, and the controlled form is I too,
    since the scalar c_B is not a phase of V. It returns the empty circuit
    without visiting the r substeps.
    """
    if arguments:
        raise ValueError("selected QFD evolution has no unresolved scalar arguments")
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.hamiltonian_evolution import apply_pauli_rotation

    labels, angles, steps, controlled = block._payload
    circuit = QuantumCircuit(sum(port.width for port in block.record.signature.quantum))
    if not labels:
        return circuit
    control = 0 if controlled else None
    for _ in range(steps):
        for label, angle in zip(labels, angles, strict=True):
            apply_pauli_rotation(circuit, label, angle, control=control)
    return circuit


def power_pair_recipes(powers, rule, positions):
    """Return the joint-state recipe of every upper-triangle pair, diagonal pairs included.

    ``powers`` are the requested integer powers in basis order, ``rule`` the
    resolved ``pair_preparation`` and ``positions`` maps +1 and -1 to the
    positions of V and V^dagger in the prepared block list, whose position
    0 is the reference preparation B. A recipe ``(common, left, right)``
    lists prepared positions (``fixed_basis._joint_state_blocks``). The
    state of power p is B, then V repeated p times or V^dagger repeated -p
    times. Under ``"prefix"`` a pair takes its end of smaller absolute
    power as base b, the smaller basis index on a tie, and runs B and
    ``V**b`` once without control. The remaining ``V**(q - b)`` of the other
    end q is controlled on the ancilla's one branch when the base is i and
    on its zero branch when the base is j. Integer powers of one V commute,
    so the branches hold ``phi_i`` and ``phi_j``. Under ``"controlled"`` both
    complete preparations, B included, are controlled. A diagonal pair
    prepares its state without control.
    """
    def uses(power):
        return (positions[1],) * power if power > 0 else (positions[-1],) * -power if power < 0 else ()

    recipes = {(i, i): ((0, *uses(p)), (), ()) for i, p in enumerate(powers)}
    for i, left in enumerate(powers):
        for j in range(i + 1, len(powers)):
            right = powers[j]
            if rule == "prefix":
                at_i = abs(left) <= abs(right)
                base, other = (left, right) if at_i else (right, left)
                remainder = uses(other - base)
                recipes[i, j] = ((0, *uses(base)), (), remainder) if at_i else ((0, *uses(base)), remainder, ())
            else:
                recipes[i, j] = ((), (0, *uses(left)), (0, *uses(right)))
    return recipes


def macro_uses(powers, rule):
    """Return the ``PairPreparation`` of every off-diagonal pair and the V and V^dagger calls of every pair circuit.

    Both come from the powers alone, without the recipes of
    ``power_pair_recipes``. The calls map each upper-triangle pair to
    ``|b| + |q - b|`` for a prefix pair with base b and other end q, to
    ``|p_i| + |p_j|`` for a controlled pair and to ``|p_i|`` for a
    diagonal pair. A prefix ``PairPreparation`` has
    ``shared_blocks = 1 + |b|``, the reference and the ``|b|`` uses of V or
    V^dagger of its base, the end of smaller absolute power.
    """
    from nwqlib.algorithms.gcim.fixed_basis import PairPreparation

    pairs, counts = [], {}
    for i, left in enumerate(powers):
        counts[i, i] = abs(left)
        for j in range(i + 1, len(powers)):
            right = powers[j]
            if rule == "prefix":
                base, other = (left, right) if abs(left) <= abs(right) else (right, left)
                counts[i, j] = abs(base) + abs(other - base)
                pairs.append(PairPreparation(pair=(i, j), route="prefix", shared_blocks=1 + abs(base)))
            else:
                counts[i, j] = abs(left) + abs(right)
                pairs.append(PairPreparation(pair=(i, j), route="controlled"))
    return tuple(pairs), counts
