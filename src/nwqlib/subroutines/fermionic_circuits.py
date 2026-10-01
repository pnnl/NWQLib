"""Exact compact circuits for normalized fermionic generator exponentials.

``build_generator_circuit`` returns ``exp(theta*A)`` for one admitted
anti-Hermitian generator ``A``. ``_plan_generator_circuit`` chooses one of
four routes. Each is exact in real arithmetic and is evaluated in binary64.

- ``commuting``: all Pauli terms of ``A`` commute, so ``exp(theta*A)`` is a
  product of one Pauli rotation per term. Spin-adapted singles,
  perfect-pairing doubles (``p == q`` and ``r == s``) and every member of the
  UCCSD, QEB and CEO pools take this route.
- ``pair_split``: a singlet double between one spatial orbital pair and two
  split orbitals. It uses the five-factor Wei-Norman product of Magoulas and
  Evangelista, arXiv:2511.13485v2, Sec. V, Eq. (15), with the closed forms
  of Eqs. (21), (22) and (25)-(27) (pp. 6-7).
- ``four_distinct``: a singlet or triplet double on four distinct spatial
  orbitals. It uses the closed forms of the same paper, Sec. VI, Table I
  (p. 13) and Table II (p. 14), with the assignment of parameters to factors
  in Supplemental Tables SI and SII (printed as S1 and S2, supplement
  pp. S16-S18, PDF pp. 46-48).
- ``shared_index``: every other default double. It uses NWQLib's own exact
  block synthesis, described in ``_append_shared_index``.

The following conventions connect the paper's operators to the library
generators. The test ``test_compact_gsd_circuits_cover_default_index_classes``
compares the circuit of every default index class with ``expm(theta*A)``,
where ``A`` is built independently in the occupation basis from the stored
fermion terms, so it checks the four items together.

1. Orientation. The paper writes
   ``A^{rs}_{pq} = a_r^dagger a_s^dagger a_q a_p - h.c.`` (Sec. III, Eq. (2),
   p. 4). The adjoint of the first term is
   ``a_p^dagger a_q^dagger a_s a_r = -a_p^dagger a_q^dagger a_r a_s``, so the
   same operator is ``a_p^dagger a_q^dagger a_r a_s - h.c.`` and swapping the
   upper and lower index pairs negates it, ``A^{pq}_{rs} = -A^{rs}_{pq}``.
   A library double with spatial indices ``(p, q, r, s)`` is the paper's
   ``A^{[0] RS}_{PQ}`` (Eq. (6), singlet) or ``A^{[1] RS}_{PQ}`` (Eq. (7),
   triplet) with ``(P, Q, R, S) = (p, q, r, s)``, up to the normalization of
   item 2. It is also the operator of Zheng et al., arXiv:2312.07691v3,
   Appendix E, Eq. (E2) or (E3), written there with creation indices
   ``(p, q)``. A pair/split double with ``p == q`` is the paper's
   ``A^{QR}_{PP}`` of Eq. (5) with ``P = p`` and ``(Q, R) = (r, s)``. With
   ``r == s`` the paired orbital is in the upper pair, so the library
   generator is the negative of ``A^{PQ}_{RR}``, and ``_append_pair_split``
   negates the angle.
2. Normalization. A library generator has unit Euclidean norm over its
   normal-ordered fermionic coefficients (``_normalized_anti_hermitian_terms``).
   The paper's operators have norm ``sqrt(2)`` in that measure for the
   index classes that use them, Eq. (5) with ``Q != R`` and Eqs. (6)-(7)
   with four distinct orbitals. Hence ``A_library = A_paper / sqrt(2)`` and
   ``exp(theta * A_library) = exp((theta / sqrt(2)) * A_paper)``, and every
   closed form is evaluated at ``theta / sqrt(2)``.
3. Factor order. The paper orders products with ascending factor index from
   left to right (text after Eq. (15)), so the rightmost factor acts first
   on a state. A ``QuantumCircuit`` applies instructions in append order,
   so the factors are appended in reverse.
4. Fermionic signs. ``_append_excitation`` compiles each factor. It takes
   the sign of the ordered ladder product on the occupation it excites and
   attaches the Jordan-Wigner parity of the inactive modes that its ladder
   operators pass over.
   Mode ``j`` is qubit ``j``, spin orbital ``(g, sigma)`` is mode
   ``2*g + sigma`` with ``sigma = 0`` for up and 1 for down (Zheng et al.,
   arXiv:2312.07691v3, Appendix E.1, text after Eq. (E3)), and the
   Jordan-Wigner images are
   those of Eq. (E8).

Factors that share one excitation commute, so the four-distinct route
combines them into one occupation-dependent angle before synthesis. The
paper's tabulated gate counts do not include the routing and outer-control
cost of these library circuits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan, atan2, cos, hypot, isfinite, pi, sin, sqrt
from typing import TYPE_CHECKING

import scipy.linalg as la
from sys import int_info
from nwqlib._linalg_laws import MAX_EXPONENTIAL_NORM
from nwqlib.subroutines.fermionic_pool import (
    FermionicGenerator,
    _active_occupation_blocks,
    _double_singlet_base_terms,
    _double_triplet_base_terms,
    _ladder_expansion_terms,
    _normalized_anti_hermitian_terms,
    _check_generator_work,
    _snapshot_generator,
)
from nwqlib.operators._pauli import _pauli_masks
from nwqlib.operators._fermion import mapping_requirements
from nwqlib.operators.access import DEFAULT_INPUT_BYTES

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


@dataclass(frozen=True)
class _GeneratorCircuitPlan:
    """Theta-independent circuit route and local blocks for one admitted generator.

    Attributes:
        kind: ``commuting``, ``pair_split``, ``four_distinct`` or
            ``shared_index`` (see the module docstring).
        generator: The admitted generator snapshot this plan compiles.
        active_modes: Sorted spin-orbital modes the generator acts on. Only
            the ``shared_index`` route sets it.
        occupation_blocks: ``(occupation indices, matrix)`` pairs from
            ``_active_occupation_blocks``, the connected blocks of ``A`` on
            the ``2**len(active_modes)`` active occupations. Only the
            ``shared_index`` route sets it.
    """
    kind: str
    generator: FermionicGenerator
    active_modes: tuple = ()
    occupation_blocks: tuple = ()


class _UnsupportedDefaultGenerator(ValueError):
    """The stored generator does not implement the normalized default family."""


def _plan_generator_circuit(generator, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000):
    """Classify one admitted generator into the exact circuit route it will use.

    ``commuting`` applies when all Pauli terms pairwise commute. Two Pauli
    words with bit masks ``(x_i, z_i)`` and ``(x_j, z_j)`` commute exactly when
    the symplectic product ``popcount(x_i & z_j) + popcount(z_i & x_j)`` is
    even. Otherwise the generator must equal a default normalized GSD double,
    and its spatial indices select ``four_distinct``, ``pair_split`` (index
    pairs ``(p, q)`` and ``(r, s)`` with no common orbital, one of them
    repeated) or ``shared_index``. Perfect-pairing doubles have commuting
    terms and never reach ``pair_split``, so that route always has two split
    orbitals. Only the shared-index route needs its occupation blocks now.
    The plan does not depend on theta, so one plan serves every angle. ADAPT
    builds it when its generator is first selected, and the ADAPT archive
    saves only the plans built.

    The active-mode cap 6 and block cap 5 bound the shared-index synthesis
    (see ``docs/ENGINEERING_CONSTANTS.md``, "Generator occupation planning
    limits"). The default GSD shared-index doubles fit these caps.
    """
    generator = _snapshot_generator(generator, max_bytes=max_bytes, max_products=max_products)
    rows = generator.pauli_terms
    if any(complex(value).real != 0.0 for _, value in rows):
        raise ValueError("generator Pauli coefficients must be anti-Hermitian")
    m, q = len(rows), generator.num_qubits
    digits = max(1, (q + int_info.bits_per_digit - 1) // int_info.bits_per_digit)
    pairs = m * (m - 1) // 2
    _check_generator_work(max_bytes, max_products, payload_bytes=m * (32 + 8 * digits), work=m * (q * digits + 1) + pairs * (digits + 1))
    masks = tuple(_pauli_masks(label)[:2] for label, _ in rows)
    if all(((x & masks[j][1]).bit_count() + (z & masks[j][0]).bit_count()) % 2 == 0
           for i, (x, z) in enumerate(masks) for j in range(i)):
        return _GeneratorCircuitPlan("commuting", generator)
    _require_default_double(generator, max_bytes=max_bytes, max_products=max_products)
    p, q, r, s = generator.spatial_indices
    if len({p, q, r, s}) == 4:
        return _GeneratorCircuitPlan("four_distinct", generator)
    if not set((p, q)).intersection((r, s)):
        return _GeneratorCircuitPlan("pair_split", generator)
    active, blocks = _active_occupation_blocks(generator, max_bytes=max_bytes, max_products=max_products,
        max_active_modes=6, max_block_dimension=5)
    return _GeneratorCircuitPlan("shared_index", generator, active, blocks)


def build_generator_circuit(
    generator: FermionicGenerator,
    theta: float,
    *,
    controlled: bool = False,
    inverse: bool = False,
    _plan: _GeneratorCircuitPlan | None = None,
    max_bytes: int = DEFAULT_INPUT_BYTES,
) -> QuantumCircuit:
    """Build the circuit of ``exp(theta*A)`` for one admitted generator.

    Inversion evaluates the complete parameter map at ``-theta``, including
    the even angle functions such as ``alpha_5`` of Magoulas and
    Evangelista, arXiv:2511.13485v2, Eq. (27), because
    ``exp(theta*A)^dagger = exp(-theta*A)`` for anti-Hermitian ``A``. The
    multi-controlled gates are therefore never passed through a generic
    circuit inverse, which
    ``test_compact_inverse_evaluates_full_map_without_reinverting_controls``
    checks. Custom commuting Pauli sums are supported. A
    noncommuting custom operator has no exact compact decomposition here and
    is rejected. Construction never expands a problem state.

    For the commuting route ``A = sum_k i c_k P_k`` with real ``c_k``, so
    ``exp(theta*A) = prod_k exp(i theta c_k P_k)``.
    ``apply_pauli_rotation(P, phi)`` implements ``exp(-i phi P / 2)``, hence
    ``phi = -2 theta c_k``. An identity term gives the phase
    ``exp(i theta c_k)``. It is a global phase without control and a phase
    gate on the control qubit under control, where the phase is physical.

    Args:
        generator: Admitted anti-Hermitian ``FermionicGenerator``.
        theta: Finite rotation angle in radians. On the ``shared_index``
            route, ``theta`` times each occupation block must have a 1-norm
            of at most ``_linalg_laws.MAX_EXPONENTIAL_NORM``, the limit of
            the matrix exponential.
        controlled: Add one control qubit. The control is qubit 0 and system
            qubit ``j`` becomes qubit ``j + 1``.
        inverse: Build ``exp(-theta*A)``.
        _plan: Precomputed ``_GeneratorCircuitPlan`` of this exact generator
            object, reused across angles.
        max_bytes: Byte cap for planning when ``_plan`` is omitted.

    Returns:
        A ``QuantumCircuit`` on ``generator.num_qubits + int(controlled)``
        qubits.
    """

    theta = float(theta)
    if not isfinite(theta):
        raise ValueError("generator angle must be finite")
    if inverse:
        theta = -theta
    if _plan is None:
        plan = _plan_generator_circuit(generator, max_bytes=max_bytes)
        generator = plan.generator
    else:
        if type(_plan) is not _GeneratorCircuitPlan or _plan.generator is not generator:
            raise ValueError("native compiler plan requires its exact admitted generator")
        plan = _plan
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.hamiltonian_evolution import apply_pauli_rotation, make_pauli_label
    off = int(controlled)
    circuit = QuantumCircuit(
        generator.num_qubits + off,
        name=f"exp_A{generator.pool_index}{'_dg' if inverse else ''}",
    )
    commuting = plan.kind == "commuting"
    if commuting:
        for label, value in generator.pauli_terms:
            coefficient = complex(value).imag
            if coefficient == 0.0:
                continue
            if set(label) == {"I"}:
                if controlled:
                    circuit.p(theta * coefficient, 0)
                else:
                    circuit.global_phase += theta * coefficient
            else:
                compact = make_pauli_label(
                    {q + off: c for q, c in enumerate(reversed(label)) if c != "I"}
                )
                apply_pauli_rotation(
                    circuit, compact, -2.0 * theta * coefficient,
                    control=0 if controlled else None,
                )
    else:
        if plan.kind == "four_distinct":
            _append_four_distinct(circuit, generator, theta, off)
        elif plan.kind == "pair_split":
            _append_pair_split(circuit, generator, theta, off)
        else:
            _append_shared_index(circuit, generator, theta, off, plan)
    return circuit


def _require_default_double(generator: FermionicGenerator, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000) -> None:
    """Require the generator to be exactly the normalized default GSD double for its indices.

    The closed-form tables hold only for that operator. A custom generator
    that merely shares a family label would otherwise be compiled into a
    different unitary. Both the fermion terms and their Jordan-Wigner Pauli
    terms must match.
    """
    if (
        generator.family not in ("double_singlet", "double_triplet")
        or len(generator.spatial_indices) != 4
    ):
        raise _UnsupportedDefaultGenerator("noncommuting custom generator has no exact compact decomposition")
    indices = generator.spatial_indices
    if any(not isinstance(i, int) or i < 0 or 2 * i + 1 >= generator.num_qubits for i in indices):
        raise _UnsupportedDefaultGenerator("GSD spatial indices must fit the system register")
    base = (
        _double_singlet_base_terms
        if generator.family == "double_singlet"
        else _double_triplet_base_terms
    )
    # Default four-operator normalization is bounded locally; each reached map
    # still pays its full-register raw frontier and label construction below.
    # The fixed 8192 bytes and 4096 work are an untuned charge for normalizing
    # at most six base terms of four ladder operators, checked against the
    # caller's limits (docs/ENGINEERING_CONSTANTS.md).
    _check_generator_work(max_bytes, max_products, payload_bytes=8192, work=4096)
    expected = _normalized_anti_hermitian_terms(base(*indices))
    if (
        len(generator.fermion_terms) != len(expected)
        or dict((term, coefficient) for coefficient, term in generator.fermion_terms) != expected
    ):
        raise _UnsupportedDefaultGenerator(
            "noncommuting custom generator does not match the normalized default GSD family"
        )
    law = mapping_requirements((len(ops) for ops in expected), num_modes=generator.num_qubits,
                               max_bytes=max_bytes, labels=True)
    _check_generator_work(max_bytes, max_products, payload_bytes=law[1], work=law[2])
    expected_pauli = _ladder_expansion_terms(tuple((c, ops) for ops, c in expected.items()),
        generator.num_qubits, mapping="jw", max_bytes=max_bytes)
    if dict(expected_pauli) != dict(generator.pauli_terms):
        raise _UnsupportedDefaultGenerator("generator fermion terms and Pauli representation must agree")


def _append_excitation(
    circuit, ops, angle, off, *, equality=None, sign_mode=None, spectators=(), angles=None
):
    """Compile ``exp(angle * (E - E^dagger))`` for one ordered ladder product ``E``.

    ``ops`` lists ``(mode, creation)`` pairs left to right, so the rightmost
    operator acts first. On the active modes, ``E - E^dagger`` connects only
    the occupation ``initial`` (annihilated modes occupied, created modes
    empty) with the occupation ``final`` obtained by flipping every active
    bit, and its exponential is a plane rotation by ``angle`` between them.
    This is the fermionic-excitation-based (FEB) form used throughout
    Magoulas and Evangelista, arXiv:2511.13485v2, Secs. V-VI.

    The circuit maps that pair of occupations onto one pivot qubit.

    - ``sign`` is the fermionic sign of ``E`` on ``initial``, so that
      ``E |initial> = sign |final>``. Each ladder operator contributes
      ``(-1)**m``, where ``m`` counts the occupied active modes below it at
      the moment it acts.
    - ``parity`` is the XOR of the Jordan-Wigner strings ``prod_{k<j} Z_k`` of
      all operators with the active modes removed. It is the parity of the
      inactive modes that ``E`` passes over. CZ gates between those modes
      and the pivot conjugate the rotation, and ``Z RY(phi) Z = RY(-phi)``
      makes its sign follow that parity.
    - After the CNOT ladder from the pivot, X gates turn every other active
      qubit into a positive control that is 1 only on the two coupled
      occupations.
    - ``RY(phi) = exp(-i phi Y / 2)`` rotates by ``phi / 2``, so
      ``scale = 2 * sign * (1 - 2 * pivot_bit)`` converts ``angle`` to the RY
      angle. The last factor flips the direction when the pivot is occupied
      in ``initial``.

    ``equality`` adds the positive control ``n_left == n_right``, which
    realizes a factor ``h_left h_right + n_left n_right`` of Eq. (14).
    ``sign_mode`` adds one Z to the parity, so the rotation changes sign when
    that mode is occupied. Combined with an equality control that includes
    that mode, as in the spin-flip factor ``E_5`` (sign mode ``P up``,
    equality of ``P up`` and ``P dn``), it realizes ``h h - n n``. With
    ``angles``, the angle depends on the
    occupations of ``spectators`` through a uniformly controlled RY. Entry
    ``k`` of ``angles`` is the angle for spectator occupation ``k``, whose
    bit ``i`` is the occupation of ``spectators[i]``.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate
    from nwqlib.subroutines.qiskit_compat import controlled as controlled_gate
    active = sorted(mode for mode, _ in ops)
    initial = sum(1 << mode for mode, creation in ops if not creation)
    state, sign, parity = initial, 1, 0
    for mode, _ in reversed(ops):
        sign *= -1 if (state & ((1 << mode) - 1)).bit_count() % 2 else 1
        state ^= 1 << mode
        parity ^= (1 << mode) - 1
    for mode in active:
        parity &= ~(1 << mode)
    if sign_mode is not None:
        parity ^= 1 << sign_mode
    pivot = active[0]
    before = QuantumCircuit(circuit.num_qubits)
    for mode in active[1:]:
        before.cx(pivot + off, mode + off)
    while parity:
        mode = (parity & -parity).bit_length() - 1
        before.cz(mode + off, pivot + off)
        parity &= parity - 1
    controls = []
    for mode in active[1:]:
        if ((initial >> mode) ^ (initial >> pivot)) & 1 == 0:
            before.x(mode + off)
        controls.append(mode + off)
    if equality is not None:
        left, right = equality
        before.cx(left + off, right + off)
        before.x(right + off)
        controls.append(right + off)
    scale = 2 * sign * (1 - 2 * ((initial >> pivot) & 1))
    circuit.compose(before, inplace=True)
    if angles is None:
        if off:
            controls.append(0)
        circuit.append(
            controlled_gate(RYGate(scale * angle), len(controls)),
            controls + [pivot + off],
        )
    else:
        # At most four spectators and three excitation controls, plus one outer control.
        # Multiplexer index bits, low to high: the excitation controls (all 1
        # on the coupled pair), the spectator occupation, then the outer
        # control. Every other index keeps angle 0, the identity.
        count = len(controls)
        controls += [mode + off for mode in spectators]
        values = [0.0] * (1 << (len(controls) + off))
        for occupation, value in enumerate(angles):
            index = (occupation << count) | ((1 << count) - 1)
            if off:
                index |= 1 << len(controls)
            values[index] = scale * value
        if off:
            controls.append(0)
        _append_multiplexed_ry(circuit, pivot + off, controls, values)
    circuit.compose(before.inverse(), inplace=True)


def _append_multiplexed_ry(circuit, target, controls, angles):
    """Apply the shared non-pruning synthesis to integer-indexed excitation qubits."""
    from nwqlib.subroutines._multiplexors import append_uniformly_controlled_ry
    append_uniformly_controlled_ry(
        circuit, circuit.qubits[target], [circuit.qubits[index] for index in controls], angles
    )


def _pair_parameters(theta):
    """Return alpha_1..alpha_5 of Magoulas and Evangelista, arXiv:2511.13485v2, Sec. V, Eqs. (21), (25), (22), (26), (27)."""
    ratio = 3 - 2 * sqrt(2)
    return (
        theta / sqrt(2),
        (1 - 1 / sqrt(2)) * theta - atan2(ratio * sin(2 * theta), 1 + ratio * cos(2 * theta)),
        -theta / sqrt(2),
        theta / sqrt(2) - asin(sin(theta) / sqrt(2)),
        pi / 4 - atan(cos(theta)),
    )


def _append_pair_split(circuit, generator, theta, off):
    """Append ``exp(theta A)`` for a double between one orbital pair and two split orbitals.

    The five factors are ``E_1``-``E_5`` of Magoulas and Evangelista,
    arXiv:2511.13485v2, Sec. V, Eq. (14), in the product of Eq. (15),
    appended in reverse so the
    rightmost factor acts first. With paired orbital ``P`` and split orbitals
    ``Q, R``, the rows of ``factors`` are

    - ``E_1 = A^{Q up R dn}_{P up P dn}``,
    - ``E_2 = E_1 (h_{Q dn} h_{R up} + n_{Q dn} n_{R up})``,
    - ``E_3 = A^{Q dn R up}_{P up P dn}``,
    - ``E_4 = E_3 (h_{Q up} h_{R dn} + n_{Q up} n_{R dn})``,
    - ``E_5 = A^{Q dn R up}_{Q up R dn} (h_{P up} h_{P dn} - n_{P up} n_{P dn})``,

    where ``n`` is a number operator and ``h = 1 - n``. Each ladder tuple
    lists ``a_r^dagger a_s^dagger a_q a_p`` in the order of Eq. (2). In
    Eq. (5) the paired orbital is in the lower index pair. When a library
    double has the pair in its upper index pair (``r == s``), it is the
    negative of that operator (module docstring, item 1), so the angle
    changes sign. The library angle is divided by ``sqrt(2)`` (item 2).
    """
    p, q, r, s = generator.spatial_indices
    if p == q:
        pair, left, right = p, r, s
    else:
        pair, left, right = r, p, q
        theta = -theta
    pu, pd, qu, qd, ru, rd = (
        2 * pair,
        2 * pair + 1,
        2 * left,
        2 * left + 1,
        2 * right,
        2 * right + 1,
    )
    factors = (
        (((qu, 1), (rd, 1), (pd, 0), (pu, 0)), None, None),
        (((qu, 1), (rd, 1), (pd, 0), (pu, 0)), (qd, ru), None),
        (((qd, 1), (ru, 1), (pd, 0), (pu, 0)), None, None),
        (((qd, 1), (ru, 1), (pd, 0), (pu, 0)), (qu, rd), None),
        (((qd, 1), (ru, 1), (rd, 0), (qu, 0)), (pu, pd), pu),
    )
    parameters = _pair_parameters(theta / sqrt(2))
    for (ops, equality, sign_mode), angle in reversed(tuple(zip(factors, parameters, strict=True))):
        _append_excitation(circuit, ops, angle, off, equality=equality, sign_mode=sign_mode)


def _singlet_parameters(theta):
    """Return b_1..b_6 of Magoulas and Evangelista, arXiv:2511.13485v2, Table I (singlet four-distinct double)."""
    ratio = 3 - 2 * sqrt(2)

    def b1(t):
        return (1 / sqrt(2) - 0.5) * t - atan2(
            ratio * sin(sqrt(2) * t), 1 + ratio * cos(sqrt(2) * t)
        )

    def b3(t):
        return t / 2 - asin(sin(t / sqrt(2)) / sqrt(2))

    def b5(t):
        return pi / 4 - atan(cos(t / sqrt(2)))

    return (
        b1(theta),
        b1(2 * theta) / 2 - 2 * b1(theta),
        b3(theta),
        b3(2 * theta) / 2 - 2 * b3(theta),
        b5(theta),
        b5(2 * theta) / 2 - 2 * b5(theta),
    )


def _triplet_parameters(theta):
    """Evaluate the twelve trigonometric angle functions consumed by the triplet factor tables.

    These are c_1, c_2, c_3, c_4, c_7, c_8 and c_11-c_16 of Magoulas and
    Evangelista, arXiv:2511.13485v2, Table II. The table's c_5, c_6, c_9 and
    c_10 are the
    combinations ``c4 - c3``, ``-c3 - c4``, ``c8 - c7`` and ``-c7 - c8`` formed
    in ``_four_distinct_factors``.
    """
    ratio = 5 - 2 * sqrt(6)

    def c1(t):
        return -(1 / sqrt(2) - 1 / sqrt(3)) * t + atan2(
            ratio * sin(sqrt(2) * t), 1 + ratio * cos(sqrt(2) * t)
        )

    a = sin(theta / sqrt(2))
    b = sin(sqrt(2) * theta)
    c3 = -theta / sqrt(3) + 3 * asin(a / sqrt(6 - a * a)) - asin(b / sqrt(6 - b * b)) / 2
    c4 = -theta / (2 * sqrt(3)) + asin(a / sqrt(6 - a * a))
    c7 = -theta / sqrt(3) + 3 * asin(a / sqrt(6)) - asin(b / sqrt(6)) / 2
    c8 = -theta / (2 * sqrt(3)) + asin(a / sqrt(6))

    def t11(t):
        c = cos(t / sqrt(2))
        return atan(2 * sqrt(3) * (c - 1) / ((c + 2) * sqrt(cos(sqrt(2) * t) + 11)))

    def t13(t):
        c = cos(t / sqrt(2))
        return atan(sqrt(2) * (c - 1) / sqrt(c * c + 4 * c + 13))

    def t15(t):
        c = cos(t / sqrt(2))
        return atan((c - 1) / (c + 5))

    return (
        c1(theta),
        4 * c1(theta) - c1(2 * theta) / 2,
        c3,
        c4,
        c7,
        c8,
        2 * t11(theta) - t11(2 * theta) / 4,
        -t11(theta) + t11(2 * theta) / 4,
        2 * t13(theta) - t13(2 * theta) / 4,
        -t13(theta) + t13(2 * theta) / 4,
        2 * t15(theta) - t15(2 * theta) / 4,
        -t15(theta) + t15(2 * theta) / 4,
    )


def _four_distinct_factors(family, theta):
    """Return the factors of Supplemental Tables S1/S2 in the paper's product order.

    Modes are local labels ``P up, P dn, Q up, Q dn, R up, R dn, S up, S dn =
    0, ..., 7`` for the generator's spatial indices ``(p, q, r, s)``.
    ``_append_four_distinct`` maps them to spin orbitals. ``theta`` is the
    paper's angle, the library angle divided by ``sqrt(2)``.

    Each row is ``(source pair, target pair, constant, occupation terms)`` and
    stands for one exponential factor. Its excitation moves the two
    electrons of ``source`` into ``target`` in the order of Eq. (2). Its
    angle is ``constant`` plus the occupation terms evaluated on the four
    spectator modes. An occupation term ``(coefficient, occupied, empty,
    sign)`` means ``coefficient * (n_occupied h_empty + sign * h_occupied
    n_empty)``, where ``n_occupied`` is the product of number operators on
    ``occupied`` and ``h = 1 - n``. The two products are particle-hole
    conjugates, as in the basis of Eq. (14). Table rows whose parameter is
    identically zero are omitted. Adjacent rows with the same excitation
    commute, so they are combined into one row without changing the factor
    order. Table I (singlet) and Table II (triplet) give the parameters, and
    ``_singlet_parameters`` and ``_triplet_parameters`` evaluate them.
    """
    pu, pd, qu, qd, ru, rd, su, sd = range(8)
    # The constant of each row is theta times that excitation's coefficient in
    # Eq. (6) or (7): 1/2 and -1/2 for the singlet, 1/sqrt(3) and
    # 1/(2 sqrt(3)) for the triplet. Sec. VI.B, p. 10, reports that the
    # Wei-Norman parameters of the defining generators equal these
    # coefficients.
    if family == "double_singlet":
        b1, b2, b3, b4, b5, b6 = _singlet_parameters(theta)
        factors = []
        for source, target, a, b, base, x, y in (
            ((pu, qd), (ru, sd), (pd, qu), (rd, su), theta / 2, b1, b2),
            ((pd, qu), (rd, su), (pu, qd), (ru, sd), theta / 2, b1, b2),
            ((pu, qd), (rd, su), (pd, qu), (ru, sd), -theta / 2, b3, b4),
            ((pd, qu), (ru, sd), (pu, qd), (rd, su), -theta / 2, b3, b4),
        ):
            factors.append(
                (
                    source,
                    target,
                    base,
                    ((x, (), a, 1), (x, (), b, 1), (-2 * x, (), a + b, 1), (y, b, a, 1)),
                )
            )
        factors += [
            (
                (pu, qd),
                (pd, qu),
                0,
                ((b5, (), (ru, sd), -1), (-b5, (), (rd, su), -1), (b6, (rd, su), (ru, sd), -1)),
            ),
            (
                (ru, sd),
                (rd, su),
                0,
                ((b5, (), (pu, qd), -1), (-b5, (), (pd, qu), -1), (b6, (pd, qu), (pu, qd), -1)),
            ),
        ]
        return factors
    c1, c2, c3, c4, c7, c8, c11, c12, c13, c14, c15, c16 = _triplet_parameters(theta)
    factors = [
        (
            (pu, qu),
            (ru, su),
            theta / sqrt(3),
            (
                (c1, (sd,), (qd,), 1),
                (-c1, (rd,), (pd,), 1),
                (-2 * c1, (rd, sd), (qd,), 1),
                (-2 * c1, (sd,), (pd, qd), 1),
                (c2, (rd, sd), (pd, qd), 1),
            ),
        ),
        (
            (pd, qd),
            (rd, sd),
            theta / sqrt(3),
            (
                (c1, (ru,), (pu,), 1),
                (-c1, (su,), (qu,), 1),
                (-2 * c1, (ru, su), (pu,), 1),
                (-2 * c1, (ru,), (pu, qu), 1),
                (c2, (ru, su), (pu, qu), 1),
            ),
        ),
        (
            (pu, qd),
            (ru, sd),
            theta / (2 * sqrt(3)),
            (
                (c3, (su,), (qu,), 1),
                (c4, (rd,), (pd,), 1),
                (c4 - c3, (rd, su), (qu,), 1),
                (c4 - c3, (su,), (pd, qu), 1),
                (-c3 - c4, (qu, rd), (pd, su), 1),
            ),
        ),
        (
            (pd, qu),
            (rd, su),
            theta / (2 * sqrt(3)),
            (
                (c3, (ru,), (pu,), 1),
                (c4, (sd,), (qd,), 1),
                (c4 - c3, (ru,), (pu, qd), 1),
                (c4 - c3, (ru, sd), (pu,), 1),
                (-c3 - c4, (qd, ru), (pu, sd), 1),
            ),
        ),
        (
            (pu, qd),
            (rd, su),
            theta / (2 * sqrt(3)),
            (
                (c7, (ru,), (qu,), 1),
                (c8, (sd,), (pd,), 1),
                (c8 - c7, (ru, sd), (qu,), 1),
                (c8 - c7, (ru,), (pd, qu), 1),
                (-c7 - c8, (qu, sd), (pd, ru), 1),
            ),
        ),
        (
            (pd, qu),
            (ru, sd),
            theta / (2 * sqrt(3)),
            (
                (c7, (su,), (pu,), 1),
                (c8, (rd,), (qd,), 1),
                (c8 - c7, (su,), (pu, qd), 1),
                (c8 - c7, (rd, su), (pu,), 1),
                (-c7 - c8, (qd, su), (pu, rd), 1),
            ),
        ),
    ]
    # The remaining six table blocks are spin exchanges conditioned on four untouched modes.
    for source, target, a, b, c, d, x, y in (
        ((qu, sd), (qd, su), pu, pd, ru, rd, c11, c12),
        ((pu, rd), (pd, ru), qu, qd, su, sd, c11, c12),
        ((qu, rd), (qd, ru), pu, pd, su, sd, c13, c14),
        ((pu, sd), (pd, su), qu, qd, ru, rd, c13, c14),
    ):
        factors.append(
            (
                source,
                target,
                0,
                (
                    (x, (c,), (a,), -1),
                    (-x, (d,), (b,), -1),
                    (y, (c,), (a, d), -1),
                    (y, (b, c), (a,), -1),
                    (y, (b,), (a, d), -1),
                    (-y, (d,), (b, c), -1),
                ),
            )
        )
    for source, target, a, b, c, d in (
        ((ru, sd), (rd, su), pu, pd, qu, qd),
        ((pu, qd), (pd, qu), ru, rd, su, sd),
    ):
        factors.append(
            (
                source,
                target,
                0,
                (
                    (c15, (), (a, d), -1),
                    (-c15, (), (b, c), -1),
                    (c16, (c,), (a, d), -1),
                    (c16, (b, c), (a,), -1),
                    (c16, (b,), (a, d), -1),
                    (-c16, (d,), (b, c), -1),
                ),
            )
        )
    return factors


def _append_four_distinct(circuit, generator, theta, off):
    """Append the Wei-Norman product for a four-distinct singlet or triplet double.

    Factors are appended in reverse product order (module docstring, item 3).
    Each factor's angle is tabulated over the 16 occupations of its four
    spectator spin orbitals, so one uniformly controlled rotation realizes
    every number-operator condition of that factor. The ladder tuple
    ``(target[0]^dagger, target[1]^dagger, source[1], source[0])`` is the
    order ``a_r^dagger a_s^dagger a_q a_p`` of Eq. (2).
    """
    modes = tuple(2 * orbital + spin for orbital in generator.spatial_indices for spin in (0, 1))
    for source, target, constant, terms in reversed(
        _four_distinct_factors(generator.family, theta / sqrt(2))
    ):
        spectators = tuple(i for i in range(8) if i not in source + target)
        angles = []
        for occupation in range(16):
            bits = dict((mode, (occupation >> j) & 1) for j, mode in enumerate(spectators))
            angle = constant
            for coefficient, occupied, empty, sign in terms:
                direct = all(bits[i] for i in occupied) and all(not bits[i] for i in empty)
                conjugate = all(not bits[i] for i in occupied) and all(bits[i] for i in empty)
                angle += coefficient * (int(direct) + sign * int(conjugate))
            angles.append(angle)
        ops = (
            (modes[target[0]], 1),
            (modes[target[1]], 1),
            (modes[source[1]], 0),
            (modes[source[0]], 0),
        )
        _append_excitation(
            circuit, ops, 0.0, off, spectators=tuple(modes[i] for i in spectators), angles=angles
        )


def _append_two_level(circuit, width, a, b, angle, off):
    """Rotate by RY(angle) in the plane of occupation states ``a`` and ``b``.

    CNOTs from a pivot bit where ``a`` and ``b`` differ reduce the pair to one
    qubit, and X gates make every other active qubit a positive control.
    Other occupation states are left unchanged.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate
    from nwqlib.subroutines.qiskit_compat import controlled as controlled_gate
    diff = a ^ b
    pivot = (diff & -diff).bit_length() - 1
    before = QuantumCircuit(circuit.num_qubits)
    encoded = a
    for j in range(width):
        if j != pivot and ((diff >> j) & 1):
            before.cx(pivot + off, j + off)
            if (a >> pivot) & 1:
                encoded ^= 1 << j
    controls = []
    for j in range(width):
        if j != pivot:
            if not ((encoded >> j) & 1):
                before.x(j + off)
            controls.append(j + off)
    if off:
        controls.append(0)
    circuit.compose(before, inplace=True)
    circuit.append(
        controlled_gate(RYGate(angle * (1 - 2 * ((a >> pivot) & 1))), len(controls)),
        controls + [pivot + off],
    )
    circuit.compose(before.inverse(), inplace=True)


def _append_shared_index(circuit, generator, theta, off, plan):
    """Append ``exp(theta A)`` for a shared-index double by exact block synthesis.

    This is NWQLib's own construction rather than a closed form from
    Magoulas and Evangelista. Fermionic swaps (SWAP followed by CZ) move the
    active modes to positions ``0..m-1`` and back. The CZ gives the sign
    ``-1`` when both exchanged modes are occupied, which is the
    anticommutation sign of exchanging two occupied fermionic modes, so the
    Jordan-Wigner signs of spectators are carried along. In these positions
    the generator is the direct sum of its occupation blocks
    (``_active_occupation_blocks``).

    Each block ``B`` is real and antisymmetric, because the default
    generators have real coefficients and are anti-Hermitian. Hence
    ``expm(theta B)`` lies in SO(k). Givens rotations reduce it column by
    column to an upper triangular orthogonal matrix whose first ``k - 1``
    diagonal entries are positive. That matrix is diagonal with entries
    ``+-1``, and determinant 1 forces the last entry to be 1, so it is the
    identity. Applying the transposed rotations in reverse order therefore
    reproduces ``expm(theta B)``. Each rotation is one two-level RY on the
    active qubits. The result is exact up to floating-point error. The
    planning caps (six active modes, blocks of dimension five) bound its
    size.
    """
    active = plan.active_modes
    width = len(active)
    # Route fermionic modes, including their parity on occupied spectators.
    order, swaps = list(range(generator.num_qubits)), []
    for destination, mode in enumerate(active):
        position = order.index(mode)
        while position > destination:
            swaps.append((position - 1, position))
            order[position - 1], order[position] = order[position], order[position - 1]
            position -= 1
    for left, right in swaps:
        circuit.swap(left + off, right + off)
        circuit.cz(left + off, right + off)
    for component, block in plan.occupation_blocks:
        if len(component) == 1:
            continue
        exponent = theta * block.real
        # Above this 1-norm a power norm that SciPy's expm computes can
        # overflow, and SciPy 1.18.1 then does not return (_linalg_laws).
        if not la.norm(exponent, 1) <= MAX_EXPONENTIAL_NORM:
            raise ValueError(
                f"generator angle {theta!r} times an occupation block has a 1-norm above "
                f"{MAX_EXPONENTIAL_NORM!r}, the matrix exponential limit"
            )
        current = la.expm(exponent)
        rotations = []
        for col in range(len(component) - 1):
            for row in range(len(component) - 1, col, -1):
                x, y = current[col, col], current[row, col]
                if y == 0.0 and x >= 0.0:
                    continue
                radius = hypot(x, y)
                if radius == 0.0:
                    continue
                c, s = x / radius, y / radius
                first, second = current[col].copy(), current[row].copy()
                current[col] = c * first + s * second
                current[row] = -s * first + c * second
                rotations.append((component[col], component[row], 2 * atan2(s, c)))
        for a, b, angle in reversed(rotations):
            _append_two_level(circuit, width, a, b, angle, off)
    for left, right in reversed(swaps):
        circuit.swap(left + off, right + off)
        circuit.cz(left + off, right + off)
