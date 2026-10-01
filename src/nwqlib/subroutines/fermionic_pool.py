"""Fermionic excitation pools for ADAPT-GCiM and their matrix-free actions.

Every pool member is an anti-Hermitian generator ``A`` stored as its Pauli
sum and, for the fermionic families, also as normal-ordered fermionic
terms. ADAPT-GCiM prepares ``exp(theta A)`` from it.

Conventions. Spin orbital ``(g, up)`` is mode ``2*g`` and ``(g, down)`` is
mode ``2*g + 1`` for zero-based spatial orbital ``g``, as in Zheng et al.
(2024), arXiv:2312.07691v3, Appendix E.1 (text after Eq. (E3), p. 13). Mode
``j`` is qubit ``j``, and the Jordan-Wigner images are those of Eq. (E8),
``a_j = (X_j + i Y_j)/2 prod_{k<j} Z_k``. In a Qiskit Pauli label qubit 0 is
the rightmost character, so qubit ``j`` is label position
``num_qubits - 1 - j``. Equation labels
E1-E8 below refer to that appendix.

Default pool (``enumerate_spin_adapted_gsd_pool``). The generalized
spin-adapted singles and doubles of Appendix E.1, Eqs. (E1)-(E3). Singles
come first, over spatial pairs ``p < q``. Doubles follow over spatial pairs
``(p, q)`` and ``(r, s)`` with ``p <= q``, ``r <= s`` and
``rank(p, q) <= rank(r, s)`` in lexicographic order. For each index
quadruple the triplet of Eq. (E3) precedes the singlet of Eq. (E2), and
an operator that vanishes identically is skipped. Pool indices follow this
order. The singlet and triplet couplings are those of Magoulas and
Evangelista, arXiv:2511.13485v2, Sec. III, Eqs. (6) and (7).

Orientation and sign. The builders write each generator as ``B - B^dagger``,
normal-order it with the canonical anticommutation relations and scale it
to unit Euclidean norm over the normal-ordered coefficients. A single
``(p, q)`` is ``a_{p up}^dagger a_{q up} + a_{p dn}^dagger a_{q dn} - h.c.``,
Eq. (E1). The double base terms are interleaved products
``a_r^dagger a_p a_s^dagger a_q`` with spin labels. Since
``a_p a_s^dagger = delta_ps - a_s^dagger a_p``, the base operator is
``B = -X`` plus one-body terms proportional to ``delta_ps``, where
``X = a_r^dagger a_s^dagger a_p a_q`` (spin labels implied). The generator
is then ``B - B^dagger = X^dagger - X`` with
``X^dagger = a_q^dagger a_p^dagger a_s a_r = a_p^dagger a_q^dagger a_r a_s``,
which creates in ``(p, q)`` and annihilates in ``(r, s)``. So the library
double ``(p, q, r, s)`` is
Eq. (E2) or (E3) with the same index order and a positive normalization
factor. Reading only the first creation index of a base term gives the
opposite sign. The one-body terms never enter the pool, because
``p == s`` with the ordering above forces ``p == q == r == s``, where
``B`` is Hermitian and the operator is skipped. This agreement holds for
every enumerated double, including repeated indices. The test
``test_generalized_pool_has_the_signed_operators_of_zheng_table_5`` checks
all operators of Table V (Table 5 in the npj version,
doi:10.1038/s41534-024-00916-8). The ``q < p``
condition of Eq. (E4) belongs to the elementary circuit decomposition and
does not restrict this enumeration.

Reversing an orientation replaces ``A`` by ``-A``. At a fixed ADAPT angle
this changes the product states and the adaptive trajectory. Flipping the
sign of a molecular orbital negates every generator whose ladder operators
act on that orbital an odd number of times, which is why the chemistry
builder fixes the sign of every orbital.

UCCSD-SD pool (``enumerate_uccsd_sd_pool``). A smaller, reference-dependent
comparison pool. Occupied and virtual spin orbitals are read from a
reference occupation string, indexed by spin orbital as above. Singles
``a_a^dagger a_i - h.c.`` run over occupied ``i`` (outer loop, ascending)
and virtual ``a`` (inner loop, ascending) with equal spin. Doubles
``a_a^dagger a_b^dagger a_i a_j - h.c.`` run over occupied pairs ``i < j``
(outer) and virtual pairs ``a < b`` (inner) with equal total ``S_z``. These
are the excitation operators of Romero et al., arXiv:1701.02691v2,
Eqs. (8)-(9), in the unitary form of Eq. (17). When occupied orbitals
precede virtual ones they are also Zheng et al., arXiv:2312.07691v3,
Eqs. (E4)-(E5). Members
are normalized like the default pool. ``spin_orbital_indices`` holds
``(i, a)`` or ``(i, j, a, b)``. The occupied-to-virtual restriction with a
fixed angle need not span the target sector, so an adaptive run on this
pool can stop above the sector minimum.

QEB-SD and OVP-CEO pools. The QEB-SD pool uses the same occupied/virtual
enumeration with qubit excitations, which are Jordan-Wigner excitations
without parity strings (Yordanov et al., arXiv:2011.10540v2,
Eqs. (17)-(18)). The OVP-CEO pool holds sums and differences of qubit
excitations on one spin-orbital set, each with a single rotation angle
(Ramôa et al., arXiv:2407.08696v3, Sec. II.B.3, Eqs. (23)-(24)). ADAPT-GCiM
assigns one angle to each selected operator, so the multi-parameter
MVP-CEOs of that paper are not represented.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
import json
from math import hypot, isfinite, sqrt
from numbers import Complex
from typing import TYPE_CHECKING, Iterable, Literal, get_args

import numpy as np

from nwqlib._validation import integer
from nwqlib._numerics import stable_complex_sum, _occupation_bits
from nwqlib.operators._pauli import apply_terms, combine_terms
from nwqlib.operators import FactorizedOperatorProduct, fermion_table
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes, _check_products
from nwqlib.operators._fermion import mapping_requirements, _check
from nwqlib.core.records import InputRef, Source

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

Spin = Literal["up", "down"]
FermionAction = Literal[0, 1]  # 0 = annihilation, 1 = creation
PoolFamily = Literal[
    "single",
    "double_singlet",
    "double_triplet",
    "uccsd_single",
    "uccsd_double",
    "qeb_single",
    "qeb_double",
    "ceo_ovp_single",
    "ceo_ovp_plus",
    "ceo_ovp_minus",
    "custom",
]
FermionOp = tuple[int, FermionAction]
NormalTerm = tuple[FermionOp, ...]

# Absolute cutoff on a coalesced coefficient of a pool generator, applied
# after every contribution to that term has been summed. It removes binary64
# residues of terms that cancel exactly in real arithmetic. Unit-normalized
# pool members keep coefficients far above it. The smallest kept Pauli
# coefficient is 0.0255 for the six-orbital default pool and at least 0.088
# for the UCCSD, QEB and CEO pools of six occupied and six virtual spin
# orbitals. Revisit for a
# generator family whose genuine coefficients can approach 1e-12, or for a
# different precision (docs/ENGINEERING_CONSTANTS.md).
GENERATOR_COEFFICIENT_CUTOFF = 1.0e-12

# Taylor degree of each step of apply_generator_exponential. Steps are chosen
# so that ||step * A|| <= 1/2, and the degree-18 remainder is then below
# exp(1/2) * (1/2)**19 / 19! < 2.6e-23 per step, far below binary64 unit
# roundoff. ADAPT's classical work laws charge this many Pauli-sum actions per
# step. Revisit for another precision or a requested total action-error bound
# (docs/ENGINEERING_CONSTANTS.md).
GENERATOR_TAYLOR_DEGREE = 18


@dataclass(frozen=True, kw_only=True)
class FermionicGenerator:
    """One anti-Hermitian ADAPT-GCiM pool generator ``A``.

    Attributes:
        family: Generator family, which fixes its excitation pattern and
            orientation (module docstring).
        spatial_indices: Ordered spatial-orbital indices defining the generator,
            ``(p, q)`` for singles and ``(p, q, r, s)`` for doubles.
        pool_index: Position in the enumerated pool.
        num_qubits: Spin-orbital register width of the Pauli representation.
        fermion_terms: ``(coefficient, operators)`` pairs of the normal-ordered
            generator. Each operator is ``(mode, action)`` with action 1 for
            creation and 0 for annihilation, and the rightmost operator acts
            first. Empty for the qubit-excitation families.
        pauli_terms: ``(label, coefficient)`` pairs of the Pauli sum of ``A``.
            Pauli words are Hermitian and ``A`` is anti-Hermitian, so every
            coefficient is purely imaginary.
        spin_orbital_indices: Spin orbitals of a reference-dependent member.
            UCCSD and QEB members and CEO singles store the occupied, then the
            virtual indices, ascending within each group. Opposite-spin CEO
            doubles store (occupied alpha, occupied beta, virtual alpha,
            virtual beta), and same-spin CEO doubles store their four modes
            sorted. None for the spin-adapted families, whose spatial indices
            identify them.
    """

    family: PoolFamily
    spatial_indices: tuple[int, ...]
    pool_index: int
    num_qubits: int
    fermion_terms: tuple[tuple[complex, NormalTerm], ...]
    pauli_terms: tuple[tuple[str, complex], ...]
    spin_orbital_indices: tuple[int, ...] | None = None


RAW_GENERATOR_SOURCE = Source(
    name="fermionic_generator.raw", version="1", domain="ordered finite generator metadata",
    reference="nwqlib.subroutines.fermionic_pool generator/1 compact sorted-key UTF-8 JSON",
)


def _check_generator_work(max_bytes, max_products, *, payload_bytes=0, work=0):
    """Check one named generator kernel's known arrays and scalar products."""
    _check_bytes(payload_bytes, max_bytes, "fermionic generator")
    _check_products(work, max_products)


def _snapshot_generator(generator, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000):
    """Admit named raw fields before traversal/copy, returning an immutable snapshot.

    Only builtin finite tuples/lists are accepted; iterators are never consumed.
    Arbitrary instance attributes are not scientific input fields and are
    neither read nor included in raw identity.
    """
    stored_bytes = products = 0

    def charge(*, payload_bytes=0, work=0):
        nonlocal stored_bytes, products
        stored_bytes += payload_bytes
        products += work
        _check_generator_work(max_bytes, max_products, payload_bytes=stored_bytes, work=products)

    if type(generator) is not FermionicGenerator:
        raise TypeError("generator admission requires FermionicGenerator")
    q = integer(generator.num_qubits, "num_qubits", minimum=0)
    index = integer(generator.pool_index, "pool_index", minimum=0)
    if type(generator.family) is not str:
        raise TypeError("generator family must be a string")
    charge(payload_bytes=len(generator.family) + 64, work=len(generator.family) + q + index.bit_length() + 8)
    if generator.family not in get_args(PoolFamily):
        raise ValueError("unknown fermionic generator family")
    immutable = type(generator.num_qubits) is int and type(generator.pool_index) is int

    def sequence(value):
        nonlocal immutable
        if type(value) not in (tuple, list):
            raise TypeError("generator metadata requires finite builtin tuples or lists")
        charge(payload_bytes=16 * len(value), work=len(value))
        immutable &= type(value) is tuple
        return value

    def coefficient(value):
        nonlocal immutable
        if isinstance(value, bool) or not isinstance(value, Complex):
            raise TypeError("generator coefficients must be finite numeric scalars")
        if isinstance(value, int):
            charge(payload_bytes=max(1, (value.bit_length() + 7) // 8), work=value.bit_length())
        try:
            result = complex(value)
        except (OverflowError, ValueError) as error:
            raise ValueError("generator coefficients must be finite binary64 scalars") from error
        if not isfinite(result.real) or not isfinite(result.imag):
            raise ValueError("generator coefficients must be finite")
        immutable &= type(value) is complex
        return result

    def indices(values, bound):
        output = []
        for value in sequence(values):
            if type(value) is not int or not 0 <= value < bound:
                raise ValueError("generator indices must be exact integers within the register")
            output.append(value)
        return tuple(output)

    spatial = indices(generator.spatial_indices, (q + 1) // 2)
    spin = None if generator.spin_orbital_indices is None else indices(generator.spin_orbital_indices, q)
    fermions = []
    for row in sequence(generator.fermion_terms):
        if len(sequence(row)) != 2:
            raise ValueError("fermion row must contain coefficient and operators")
        value = coefficient(row[0])
        operators = []
        for pair in sequence(row[1]):
            if len(sequence(pair)) != 2:
                raise ValueError("fermion operator must contain mode and action")
            mode, action = pair
            if type(mode) is not int or not 0 <= mode < q or type(action) is not int or action not in (0, 1):
                raise ValueError("fermion operator requires an in-register integer mode and action 0/1")
            operators.append((mode, action))
        fermions.append((value, tuple(operators)))
    rows = []
    for row in sequence(generator.pauli_terms):
        if len(sequence(row)) != 2:
            raise ValueError("Pauli row must contain label and coefficient")
        label, value = row
        if type(label) is not str:
            raise TypeError("generator Pauli labels must be strings")
        charge(payload_bytes=len(label) + 16, work=len(label) + 1)
        if len(label) != q or any(axis not in "IXYZ" for axis in label):
            raise ValueError("generator and Pauli qubit counts must agree")
        rows.append((label, coefficient(value)))
    if immutable:
        return generator
    return FermionicGenerator(family=generator.family, spatial_indices=spatial,
        spin_orbital_indices=spin, pool_index=index, num_qubits=q,
        fermion_terms=tuple(fermions), pauli_terms=tuple(rows))


def _generator_reference(generator, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000):
    """Hash an admitted snapshot in generator/1 encoding; signed zeros become +0.

    Sorted-key compact UTF-8 JSON keeps row/operator order, exact indices,
    and finite binary64 [real, imaginary] coefficients. It asserts raw content
    identity; it does not prove the later projection/mapping scientifically.
    """
    slots = (len(generator.pauli_terms) * (generator.num_qubits + 64)
             + sum(64 + len(ops) * (32 + generator.num_qubits.bit_length())
                   for _, ops in generator.fermion_terms)
             + 32 * (len(generator.spatial_indices) + len(generator.spin_orbital_indices or ()))
             + len(generator.family) + 256)
    _check_generator_work(max_bytes, max_products, payload_bytes=8 * slots, work=8 * slots)
    def parts(value):
        return (value.real or 0.0, value.imag or 0.0)
    raw = dict(encoding="nwqlib.fermionic_generator/1", family=generator.family,
        spatial_indices=generator.spatial_indices, spin_orbital_indices=generator.spin_orbital_indices,
        pool_index=generator.pool_index, num_qubits=generator.num_qubits,
        fermion_terms=tuple((parts(c), ops) for c, ops in generator.fermion_terms),
        pauli_terms=tuple((p, parts(c)) for p, c in generator.pauli_terms))
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return InputRef(identity="sha256:" + sha256(encoded).hexdigest(),
                    representation="fermionic_generator", source=RAW_GENERATOR_SOURCE)


def spin_orbital(spatial_orbital: int, spin: Spin) -> int:
    """Return the spin-orbital index for the paper's convention."""

    if spatial_orbital < 0:
        raise ValueError("spatial_orbital must be non-negative")
    return 2 * spatial_orbital + (0 if spin == "up" else 1)


def enumerate_spin_adapted_gsd_pool(n_spatial_orbitals: int) -> tuple[FermionicGenerator, ...]:
    """Enumerate the generalized spin-adapted singles and doubles of Zheng et al., arXiv:2312.07691v3, Eqs. (E1)-(E3).

    The order and orientation are described in the module docstring. With
    ``P = n (n + 1) / 2`` spatial pairs for ``n`` orbitals, the pool has at
    most ``P`` singles and ``P (P + 1)`` doubles, because each pair of pairs
    contributes at most one triplet and one singlet. Operators that vanish
    identically are skipped, so the actual count is smaller.

    Args:
        n_spatial_orbitals: Number of spatial orbitals ``n``. The generators
            act on ``2 n`` interleaved spin-orbital qubits.

    Returns:
        The generators in pool-index order.
    """

    n_spatial_orbitals = integer(n_spatial_orbitals, "n_spatial_orbitals", 1)
    num_qubits = 2 * n_spatial_orbitals
    generators: list[FermionicGenerator] = []

    for p in range(n_spatial_orbitals):
        for q in range(p, n_spatial_orbitals):
            terms = _single_base_terms(p, q)
            if terms:
                generators.append(
                    _generator(
                        family="single",
                        spatial_indices=(p, q),
                        pool_index=len(generators),
                        num_qubits=num_qubits,
                        base_terms=terms,
                    )
                )

    pair_rank = -1
    for p in range(n_spatial_orbitals):
        for q in range(p, n_spatial_orbitals):
            pair_rank += 1
            other_rank = -1
            for r in range(n_spatial_orbitals):
                for s in range(r, n_spatial_orbitals):
                    other_rank += 1
                    if pair_rank > other_rank:
                        continue
                    # Emit triplet-like before singlet-like for each (p, q, r, s).
                    for family, terms in (
                        ("double_triplet", _double_triplet_base_terms(p, q, r, s)),
                        ("double_singlet", _double_singlet_base_terms(p, q, r, s)),
                    ):
                        if terms:
                            generators.append(
                                _generator(
                                    family=family,
                                    spatial_indices=(p, q, r, s),
                                    pool_index=len(generators),
                                    num_qubits=num_qubits,
                                    base_terms=terms,
                                )
                            )

    return tuple(generators)


def enumerate_uccsd_sd_pool(reference_occupations) -> tuple[FermionicGenerator, ...]:
    """Enumerate the occupied-to-virtual UCCSD singles and doubles of a reference.

    This smaller, reference-dependent pool is a comparison baseline beside
    :func:`enumerate_spin_adapted_gsd_pool`. The module docstring gives the
    ordering, index meaning and spin selection, and the source equations
    (Romero et al., arXiv:1701.02691v2, Eqs. (8)-(9) and (17)).
    """

    bits, occupied, virtual = _occupation_sets(reference_occupations)
    num_qubits = len(bits)
    generators: list[FermionicGenerator] = []

    for i in occupied:
        for a in virtual:
            if i % 2 != a % 2:
                continue
            generators.append(
                _generator(
                    family="uccsd_single",
                    spatial_indices=(i // 2, a // 2),
                    spin_orbital_indices=(i, a),
                    pool_index=len(generators),
                    num_qubits=num_qubits,
                    base_terms=((1.0, ((a, 1), (i, 0))),),
                )
            )

    for i_position, i in enumerate(occupied):
        for j in occupied[i_position + 1 :]:
            for a_position, a in enumerate(virtual):
                for b in virtual[a_position + 1 :]:
                    if (i % 2) + (j % 2) != (a % 2) + (b % 2):
                        continue
                    generators.append(
                        _generator(
                            family="uccsd_double",
                            spatial_indices=(i // 2, j // 2, a // 2, b // 2),
                            spin_orbital_indices=(i, j, a, b),
                            pool_index=len(generators),
                            num_qubits=num_qubits,
                            # The annihilation order (i, j) fixes the generator sign, since
                            # a_i a_j = -a_j a_i.
                            base_terms=((1.0, ((a, 1), (b, 1), (i, 0), (j, 0))),),
                        )
                    )

    return tuple(generators)


def enumerate_qeb_sd_pool(reference_occupations) -> tuple[FermionicGenerator, ...]:
    """Enumerate reference-dependent QEB singles/doubles.

    The occupied/virtual enumeration mirrors :func:`enumerate_uccsd_sd_pool`,
    but the operators are the qubit excitations of Yordanov et al.,
    arXiv:2011.10540v2, Eqs. (17)-(18), p. 6:
    ``Q_a^dagger Q_i - Q_i^dagger Q_a`` and
    ``Q_a^dagger Q_b^dagger Q_i Q_j - Q_i^dagger Q_j^dagger Q_a Q_b``, with
    ``Q_j = (X_j + i Y_j)/2`` and no Jordan-Wigner parity string. Eqs. (21)-(22)
    of the same paper give their exponentials as Pauli rotations.
    """

    bits, occupied, virtual = _occupation_sets(reference_occupations)
    num_qubits = len(bits)
    generators: list[FermionicGenerator] = []

    for i in occupied:
        for a in virtual:
            if i % 2 != a % 2:
                continue
            generators.append(
                _pauli_generator(
                    family="qeb_single",
                    spatial_indices=(i // 2, a // 2),
                    spin_orbital_indices=(i, a),
                    pool_index=len(generators),
                    num_qubits=num_qubits,
                    pauli=_qubit_single_excitation_generator(a, i, num_qubits=num_qubits),
                )
            )

    for i_position, i in enumerate(occupied):
        for j in occupied[i_position + 1 :]:
            for a_position, a in enumerate(virtual):
                for b in virtual[a_position + 1 :]:
                    if (i % 2) + (j % 2) != (a % 2) + (b % 2):
                        continue
                    generators.append(
                        _pauli_generator(
                            family="qeb_double",
                            spatial_indices=(i // 2, j // 2, a // 2, b // 2),
                            spin_orbital_indices=(i, j, a, b),
                            pool_index=len(generators),
                            num_qubits=num_qubits,
                            pauli=_qubit_double_excitation_generator(
                                a, b, i, j, num_qubits=num_qubits
                            ),
                        )
                    )

    return tuple(generators)


def enumerate_ceo_ovp_pool(reference_occupations) -> tuple[FermionicGenerator, ...]:
    """Enumerate one-variational-parameter coupled exchange operators.

    MVP-CEOs are deliberately not represented: they need per-component
    parameters inside one selected generator, while ADAPT-GCiM records one
    rotation parameter per selected operator.

    The pool holds the single qubit excitations, then for each opposite-spin
    quadruple the plus and minus combinations of the direct and crossed
    double qubit excitations (Ramôa et al., arXiv:2407.08696v3, Sec. II.B.3,
    Eqs. (23)-(24), p. 8). For each same-spin quadruple it holds the sums and
    differences of every pair of its three qubit excitations, the six
    OVP-CEOs described on p. 9.
    """

    bits, occupied, virtual = _occupation_sets(reference_occupations)
    num_qubits = len(bits)
    generators: list[FermionicGenerator] = []

    for single in enumerate_qeb_sd_pool(bits):
        if single.family != "qeb_single":
            continue
        generators.append(
            _pauli_generator(
                family="ceo_ovp_single",
                spatial_indices=single.spatial_indices,
                spin_orbital_indices=single.spin_orbital_indices,
                pool_index=len(generators),
                num_qubits=num_qubits,
                pauli=single.pauli_terms,
            )
        )

    occupied_by_spin = {
        spin: tuple(mode for mode in occupied if mode % 2 == spin) for spin in (0, 1)
    }
    virtual_by_spin = {spin: tuple(mode for mode in virtual if mode % 2 == spin) for spin in (0, 1)}

    for occ_alpha in occupied_by_spin[0]:
        for occ_beta in occupied_by_spin[1]:
            for virt_alpha in virtual_by_spin[0]:
                for virt_beta in virtual_by_spin[1]:
                    direct = _qubit_double_excitation_generator(
                        virt_alpha,
                        virt_beta,
                        occ_alpha,
                        occ_beta,
                        num_qubits=num_qubits,
                    )
                    crossed = _qubit_double_excitation_generator(
                        occ_alpha,
                        virt_beta,
                        virt_alpha,
                        occ_beta,
                        num_qubits=num_qubits,
                    )
                    for family, sign in (("ceo_ovp_plus", 1.0), ("ceo_ovp_minus", -1.0)):
                        # The crossed excitation (virt_alpha, occ_beta) -> (occ_alpha, virt_beta)
                        # enters with + or - sign (Ramôa et al.,
                        # arXiv:2407.08696v3, Eqs. (23)-(24)).
                        pauli = _sum_paulis((direct, tuple((p, sign * c) for p, c in crossed)))
                        indices = (occ_alpha, occ_beta, virt_alpha, virt_beta)
                        generators.append(
                            _pauli_generator(
                                family=family,
                                spatial_indices=tuple(index // 2 for index in indices),
                                spin_orbital_indices=indices,
                                pool_index=len(generators),
                                num_qubits=num_qubits,
                                pauli=pauli,
                            )
                        )

    for spin in (0, 1):
        for occ_pos, occ_a in enumerate(occupied_by_spin[spin]):
            for occ_b in occupied_by_spin[spin][occ_pos + 1 :]:
                for virt_pos, virt_a in enumerate(virtual_by_spin[spin]):
                    for virt_b in virtual_by_spin[spin][virt_pos + 1 :]:
                        modes = tuple(sorted((occ_a, occ_b, virt_a, virt_b)))
                        qes = _same_spin_qubit_excitations(modes, num_qubits=num_qubits)
                        for left_pos, left in enumerate(qes):
                            for right in qes[left_pos + 1 :]:
                                for family, sign in (
                                    ("ceo_ovp_plus", 1.0),
                                    ("ceo_ovp_minus", -1.0),
                                ):
                                    generators.append(
                                        _pauli_generator(
                                            family=family,
                                            spatial_indices=tuple(mode // 2 for mode in modes),
                                            spin_orbital_indices=modes,
                                            pool_index=len(generators),
                                            num_qubits=num_qubits,
                                            pauli=_sum_paulis((left, tuple((p, sign * c) for p, c in right))),
                                        )
                                    )

    return tuple(generators)


def _occupation_sets(
    reference_occupations,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return ``(bits, occupied, virtual)`` of a reference occupation string.

    ``bits[j]`` is the occupation of spin orbital ``j``. ``occupied`` and
    ``virtual`` list the spin orbitals with occupation 1 and 0 in ascending
    order.
    """
    bits = _occupation_bits(reference_occupations)
    if not bits:
        raise ValueError("reference_occupations must not be empty")
    occupied = tuple(mode for mode, bit in enumerate(bits) if bit == 1)
    virtual = tuple(mode for mode, bit in enumerate(bits) if bit == 0)
    return bits, occupied, virtual


def _reference_pool_size(family, reference_occupations) -> int:
    """Return how many members a reference-dependent pool has, without building any.

    ``family`` is ``uccsd_sd``, ``qeb_sd`` or ``ceo_ovp``. Let ``o_s`` and
    ``v_s`` count the occupied and virtual spin orbitals of spin ``s``, where
    even modes are up and odd modes down. Every family has the
    ``o_up v_up + o_dn v_dn`` same-spin singles. The UCCSD and QEB
    enumerators emit one double per same-spin quadruple,
    ``C(o_s, 2) C(v_s, 2)`` for each spin, and one per opposite-spin
    quadruple, ``o_up o_dn v_up v_dn``. The CEO enumerator emits a plus and a
    minus member per opposite-spin quadruple and six per same-spin quadruple,
    the sums and differences of the three pairs of its three qubit
    excitations. ``test_reference_pool_size_matches_enumeration`` compares
    these counts with the enumerators.
    """
    _, occupied, virtual = _occupation_sets(reference_occupations)
    o = tuple(sum(1 for mode in occupied if mode % 2 == spin) for spin in (0, 1))
    v = tuple(sum(1 for mode in virtual if mode % 2 == spin) for spin in (0, 1))
    singles = o[0] * v[0] + o[1] * v[1]
    same_spin = sum(o[s] * (o[s] - 1) // 2 * (v[s] * (v[s] - 1) // 2) for s in (0, 1))
    opposite_spin = o[0] * o[1] * v[0] * v[1]
    if family == "ceo_ovp":
        return singles + 2 * opposite_spin + 6 * same_spin
    if family in ("uccsd_sd", "qeb_sd"):
        return singles + opposite_spin + same_spin
    raise ValueError("unknown reference-dependent ADAPT pool")


def _pauli_generator(
    *,
    family: PoolFamily,
    spatial_indices: tuple[int, ...],
    pool_index: int,
    num_qubits: int,
    pauli: tuple[tuple[str, complex], ...],
    spin_orbital_indices: tuple[int, ...] | None,
) -> FermionicGenerator:
    """Wrap a qubit-excitation Pauli sum as a pool member without fermionic terms.

    QEB and CEO generators have no Jordan-Wigner parity strings, so they are
    stored only as their coalesced Pauli sum, with coefficients at or below
    ``GENERATOR_COEFFICIENT_CUTOFF`` removed. Unlike ``_generator``, no
    normalization is applied. The coefficients are those of Yordanov et al.,
    arXiv:2011.10540v2, Eqs. (17)-(18), or of their sums and differences.
    """
    return FermionicGenerator(
        family=family,
        spatial_indices=spatial_indices,
        pool_index=pool_index,
        num_qubits=num_qubits,
        fermion_terms=(),
        pauli_terms=tuple(
            (p, c) for p, c in combine_terms(pauli) if abs(c) > GENERATOR_COEFFICIENT_CUTOFF
        ),
        spin_orbital_indices=spin_orbital_indices,
    )


def _qubit_single_excitation_generator(
    create: int, annihilate: int, *, num_qubits: int
) -> tuple[tuple[str, complex], ...]:
    """Pauli sum of ``Q_create^dagger Q_annihilate - h.c.``, Yordanov et al., arXiv:2011.10540v2, Eq. (17)."""
    return _ladder_expansion_terms(
        (
            (1.0, ((create, 1), (annihilate, 0))),
            (-1.0, ((annihilate, 1), (create, 0))),
        ),
        num_qubits, mapping="z_free",
    )


def _qubit_double_excitation_generator(
    create_a: int,
    create_b: int,
    annihilate_a: int,
    annihilate_b: int,
    *,
    num_qubits: int,
) -> tuple[tuple[str, complex], ...]:
    """Pauli sum of the double qubit excitation ``T_ijkl`` of Yordanov et al., arXiv:2011.10540v2, Eq. (18).

    ``T_ijkl = Q_i^dagger Q_j^dagger Q_k Q_l - h.c.`` with
    ``(i, j, k, l) = (create_a, create_b, annihilate_a, annihilate_b)``.
    """
    return _ladder_expansion_terms(
        (
            (1.0, ((create_a, 1), (create_b, 1), (annihilate_a, 0), (annihilate_b, 0))),
            (-1.0, ((annihilate_a, 1), (annihilate_b, 1), (create_a, 0), (create_b, 0))),
        ),
        num_qubits, mapping="z_free",
    )


def _same_spin_qubit_excitations(
    modes: tuple[int, int, int, int],
    *,
    num_qubits: int,
) -> tuple[tuple[tuple[str, complex], ...], ...]:
    """Return the three qubit double excitations among four same-spin modes.

    For sorted modes ``a < b < c < d`` these are the three ways to split
    them into an annihilated pair that contains ``a`` and a created pair:
    ``(a, b) -> (c, d)``, ``(a, c) -> (b, d)`` and ``(a, d) -> (b, c)``. The
    OVP-CEO pool forms sums and differences of every two of them (Ramôa
    et al., arXiv:2407.08696v3, p. 9).
    """
    a, b, c, d = modes
    return (
        _qubit_double_excitation_generator(c, d, a, b, num_qubits=num_qubits),
        _qubit_double_excitation_generator(b, d, a, c, num_qubits=num_qubits),
        _qubit_double_excitation_generator(b, c, a, d, num_qubits=num_qubits),
    )


def _native_ladder_sum(terms, num_qubits, *, mapping, max_bytes=DEFAULT_INPUT_BYTES, labels=False):
    """Map ordered ladder terms to a coalesced Pauli ``OperatorInput``.

    ``mapping`` is ``jw`` (Jordan-Wigner, Eq. (E8)) or ``z_free`` (qubit
    ladders without parity strings). The complete mapping byte law of
    ``mapping_requirements`` is checked before the first product. Returns
    ``(operator, max_bytes)``.
    """
    table = fermion_table(terms, num_modes=num_qubits, max_bytes=max_bytes)
    lengths = (int(table.offsets[i+1])-int(table.offsets[i]) for i in range(len(table)))
    _check(max_bytes, mapping_requirements(lengths, num_modes=num_qubits, max_bytes=max_bytes, labels=labels))
    return table.to_pauli(mapping=mapping, max_bytes=max_bytes), max_bytes


def _ladder_expansion_to_pauli(
    terms: Iterable[tuple[complex, NormalTerm]],
    num_qubits: int,
    *,
    mapping: Literal["jw", "z_free"],
    coefficient_cutoff: float = GENERATOR_COEFFICIENT_CUTOFF,
    max_bytes: int = DEFAULT_INPUT_BYTES,
) -> SparsePauliOp:
    """Map ordered terms natively, then apply the caller cutoff after the sum.

    Full raw contributions reach one stable coalescing boundary, so the
    cutoff acts on each label's summed coefficient and never on an individual
    contribution. Conversion to a Qiskit ``SparsePauliOp`` happens only here.
    """
    from qiskit.quantum_info import SparsePauliOp

    kept = _ladder_expansion_terms(terms, num_qubits, mapping=mapping,
                                      coefficient_cutoff=coefficient_cutoff, max_bytes=max_bytes)
    return SparsePauliOp.from_list(kept or [("I" * num_qubits, 0.0)])


def _ladder_expansion_terms(terms, num_qubits, *, mapping, coefficient_cutoff=GENERATOR_COEFFICIENT_CUTOFF,
                            max_bytes=DEFAULT_INPUT_BYTES):
    """Return the ``(label, coefficient)`` Pauli terms of ordered ladder terms.

    All contributions to a label are summed before the absolute
    ``coefficient_cutoff`` is applied to the sum, as in
    ``_ladder_expansion_to_pauli``, which adds the SDK conversion.
    """
    native, max_bytes = _native_ladder_sum(terms, num_qubits, mapping=mapping, max_bytes=max_bytes, labels=True)
    labels = native.pauli_terms().labels(max_bytes=max_bytes)
    return tuple((label, value) for label, value in labels
                 if hypot(value.real, value.imag) > coefficient_cutoff)


def _sum_paulis(paulis):
    """Coalesce several Pauli sums label by label and drop coefficients at or below the cutoff."""
    return tuple((p, c) for p, c in combine_terms(term for pauli in paulis for term in pauli)
                 if abs(c) > GENERATOR_COEFFICIENT_CUTOFF)


def _active_occupation_blocks(generator, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000, max_active_modes=8, max_block_dimension=8):
    """Return bounded active-mode connected blocks of the stored fermion action.

    The generator is written as a matrix on the ``2**m`` occupation states of
    its m active modes, as if those modes were reordered to positions
    ``0..m-1``. Jordan-Wigner signs then come only from occupied active modes
    below each acted mode. That fermionic reordering is unitary on Fock
    space, and every term has an even number of ladder operators, so the
    full generator is equivalent to these blocks times the identity on
    spectators. Its spectrum is the block spectrum used by the shift rules.
    The shared-index circuit applies the same reordering with fermionic
    swaps. Connected components of the matrix are the blocks.
    The active-mode cap bounds the enumeration before it starts, and the
    block-dimension cap is checked for each component
    (docs/ENGINEERING_CONSTANTS.md, "Generator occupation planning limits").

    Returns:
        ``(active_modes, blocks)``. ``active_modes`` is the sorted tuple of
        acted modes. Each block is ``(occupations, matrix)``, where
        ``occupations`` lists the local occupation integers of one connected
        component (bit ``i`` is ``active_modes[i]``) and ``matrix[i, j]`` is
        ``<occupations[i]| A |occupations[j]>``.
    """
    terms = len(generator.fermion_terms)
    operators = sum(len(ops) for _, ops in generator.fermion_terms)
    _check_generator_work(max_bytes, max_products, payload_bytes=32 * operators, work=(operators + 1) * (operators.bit_length() + generator.num_qubits.bit_length() + 1))
    active = tuple(sorted({mode for _, ops in generator.fermion_terms for mode, _ in ops}))
    if len(active) > max_active_modes:
        raise ValueError(f"local occupation planning supports at most {max_active_modes} active modes")
    occupations = 1 << len(active)
    # Bytes: an allowance of 64 per column entry, charged for
    # min(terms, 2**m) + 1 entries in each of the 2**m columns. A column
    # holds at most min(terms, 2**m) nonzero entries.
    # Work: for each of the 2**m occupations, 4 units per term and per stored
    # ladder operator (the sign loop below) plus 2**m for the
    # connected-component search.
    _check_generator_work(max_bytes, max_products, payload_bytes=64 * occupations * (min(terms, occupations) + 1), work=occupations * (4 * (terms + operators + 1) + occupations))
    mapping = {mode: index for index, mode in enumerate(active)}
    columns = {}
    for initial in range(1 << len(active)):
        column = {}
        for coefficient, ops in generator.fermion_terms:
            state, value = initial, complex(coefficient)
            for mode, creation in reversed(ops):
                mode = mapping[mode]
                if ((state >> mode) & 1) == creation:
                    break
                if (state & ((1 << mode) - 1)).bit_count() % 2:
                    value = -value
                state ^= 1 << mode
            else:
                column[state] = column.get(state, 0.0) + value
        columns[initial] = {row: value for row, value in column.items() if value != 0}
    unseen = set(columns)
    blocks = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, pending = [start], [start]
        while pending:
            adjacent = set(columns[pending.pop()]).intersection(unseen)
            unseen.difference_update(adjacent)
            component.extend(adjacent)
            pending.extend(adjacent)
        component.sort()
        if len(component) > max_block_dimension:
            raise ValueError(f"local occupation block exceeds dimension {max_block_dimension}")
        _check_generator_work(max_bytes, max_products, payload_bytes=32 * len(component) ** 2, work=len(component) ** 2)
        block = np.array([[columns[col].get(row, 0.0) for col in component] for row in component], dtype=complex)
        block.flags.writeable = False
        blocks.append((tuple(component), block))
    return active, tuple(blocks)


def apply_generator(generator: FermionicGenerator, state: np.ndarray) -> np.ndarray:
    """Apply a generator through its matrix-free Pauli action."""

    return apply_terms(generator.pauli_terms, state, num_qubits=generator.num_qubits)


def _pauli_one_norm(coefficients) -> float:
    """Return the Pauli 1-norm ``sum_k |c_k|``, summed in binary64 in the given order."""
    return sum(abs(complex(value)) for value in coefficients)


def _taylor_steps(coefficients, theta) -> int:
    """Return the number of scaled Taylor steps for ``exp(theta * A)``.

    ``coefficients`` are the Pauli coefficients ``c_k`` of ``A`` in stored
    order. The count is ``max(1, ceil(2 * |theta| * sum_k |c_k|))``, or 0
    when ``|theta| * sum_k |c_k|`` is zero. The Pauli 1-norm bounds ``||A||``,
    because each Pauli word has norm 1, so the step ``h = theta / steps``
    has ``||hA|| <= 1/2``, the premise of the remainder bound beside
    ``GENERATOR_TAYLOR_DEGREE``. The coefficient sum avoids a state-space
    operator-norm computation. ``apply_generator_exponential`` and ADAPT's
    work law (``adapt_records._exponential_action_work``) both call this
    function with the same stored coefficients, so they count the same
    steps.
    """
    scaled_norm_bound = abs(float(theta)) * _pauli_one_norm(coefficients)
    if scaled_norm_bound == 0.0:
        return 0
    return max(1, int(np.ceil(2.0 * scaled_norm_bound)))


def apply_generator_exponential(
    generator: FermionicGenerator,
    state: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Apply ``exp(theta * A)`` to a state vector by scaled Taylor steps.

    The state is not normalized, and a new vector of the same length is
    returned. The work is ``steps * GENERATOR_TAYLOR_DEGREE`` matrix-free
    Pauli-sum actions, with ``steps`` from ``_taylor_steps``. ADAPT's
    classical work laws take their step count from the same function.

    Args:
        generator: Anti-Hermitian generator ``A``.
        state: Complex vector of length ``2**generator.num_qubits``.
        theta: Rotation angle in radians.

    Returns:
        ``exp(theta * A) @ state`` up to the Taylor and rounding errors
        described in the code.
    """

    vector = np.asarray(state, dtype=complex).reshape(-1)
    steps = _taylor_steps((value for _, value in generator.pauli_terms), theta)
    if steps == 0:
        return vector.copy()
    step = float(theta) / steps
    for _ in range(steps):
        vector = _scaled_taylor_step(generator, vector, step)
    return vector


def _scaled_taylor_step(
    operator: FermionicGenerator,
    state: np.ndarray,
    step: float,
) -> np.ndarray:
    """Apply one degree-18 Taylor step with two reusable state buffers.

    Horner form of ``sum_{k=0}^{18} (step*A)**k / k! @ state``: starting from
    ``r = state``, ``r <- state + (step / k) * A r`` for ``k = 18, ..., 1``.
    """

    result = np.empty_like(state)
    work = np.empty_like(state)
    result[:] = state
    for degree in range(GENERATOR_TAYLOR_DEGREE, 0, -1):
        apply_terms(operator.pauli_terms, result, num_qubits=operator.num_qubits, out=work)
        work *= step / degree
        work += state
        result, work = work, result
    return result


def apply_spin_squared(state: np.ndarray, *, num_qubits: int) -> np.ndarray:
    """Apply ``S^2`` matrix-free using the same JW ladder convention.

    Uses ``S^2 = S_- S_+ + S_z (S_z + 1)`` with
    ``S_+ = sum_p a_{p up}^dagger a_{p down}`` and ``S_- = S_+^dagger``. The
    ladder product is a two-factor matrix-free action, and ``S_z`` is
    diagonal in the interleaved occupation basis.
    """

    if num_qubits % 2:
        raise ValueError("apply_spin_squared requires an even number of spin orbitals")
    vector = np.asarray(state, dtype=complex).reshape(-1)
    if vector.size != 2**num_qubits:
        raise ValueError("state length must match num_qubits")
    if num_qubits == 0:
        return np.zeros_like(vector)
    minus_terms, plus_terms = [], []
    for p in range(num_qubits // 2):
        down_p = spin_orbital(p, "down")
        up_p = spin_orbital(p, "up")
        minus_terms.append((1.0, ((down_p, 1), (up_p, 0))))
        plus_terms.append((1.0, ((up_p, 1), (down_p, 0))))
    minus, _ = _native_ladder_sum(minus_terms, num_qubits, mapping="jw")
    plus, _ = _native_ladder_sum(plus_terms, num_qubits, mapping="jw")
    # Only the two already admitted references and dimension metadata are owned.
    metadata_bytes = 16 + ((2**num_qubits).bit_length() + 7) // 8
    factors = FactorizedOperatorProduct((minus, plus), max_bytes=metadata_bytes)
    law = factors.action_requirements()
    # This function takes no caller limit, so the default input byte and
    # product ceilings apply (docs/ENGINEERING_CONSTANTS.md).
    _check_bytes(law[1], DEFAULT_INPUT_BYTES, "spin-squared action")
    if law[2] > 1_000_000_000:
        raise ValueError(f"spin-squared action needs {law[2]} scalar products, above the fixed "
                         "bound of 1,000,000,000 products for apply_spin_squared")
    indices = np.arange(vector.size, dtype=np.uint64)
    spin_z_values = np.zeros(vector.size, dtype=float)
    for spatial in range(num_qubits // 2):
        up = (indices >> np.uint64(2 * spatial)) & np.uint64(1)
        down = (indices >> np.uint64(2 * spatial + 1)) & np.uint64(1)
        spin_z_values += 0.5 * (up.astype(float) - down.astype(float))
    return factors.matvec(vector, max_bytes=DEFAULT_INPUT_BYTES) + (
        spin_z_values * (spin_z_values + 1.0) * vector
    )


def _generator(
    *,
    family: PoolFamily,
    spatial_indices: tuple[int, ...],
    pool_index: int,
    num_qubits: int,
    base_terms: Iterable[tuple[complex, NormalTerm]],
    spin_orbital_indices: tuple[int, ...] | None = None,
) -> FermionicGenerator:
    """Build one pool member ``A`` from the base operator ``B`` of a fermionic family.

    ``A`` is ``B - B^dagger``, normal-ordered and scaled to unit Euclidean
    norm over its fermionic coefficients (``_normalized_anti_hermitian_terms``).
    Its Pauli terms are the Jordan-Wigner image of those normal-ordered
    terms, so the two stored representations are the same operator.
    Fermionic terms are stored sorted by operator tuple, which makes the
    record independent of the order in which the base terms were written.
    """
    normal_terms = _normalized_anti_hermitian_terms(base_terms)
    pauli = _ladder_expansion_terms(tuple((coefficient, ops) for ops, coefficient in normal_terms.items()),
                                   num_qubits, mapping="jw")
    return FermionicGenerator(
        family=family,
        spatial_indices=spatial_indices,
        pool_index=pool_index,
        num_qubits=num_qubits,
        fermion_terms=tuple(
            (coefficient, ops)
            for ops, coefficient in sorted(normal_terms.items(), key=lambda item: item[0])
        ),
        pauli_terms=pauli,
        spin_orbital_indices=spin_orbital_indices,
    )


def _single_base_terms(p: int, q: int) -> tuple[tuple[complex, NormalTerm], ...]:
    """Base terms of the spin-adapted single of Eq. (E1), before ``- h.c.``.

    ``a_{p up}^dagger a_{q up} + a_{p dn}^dagger a_{q dn}``, or ``()`` when
    ``B - B^dagger`` vanishes, which happens for ``p == q``.
    """
    up_p, down_p = spin_orbital(p, "up"), spin_orbital(p, "down")
    up_q, down_q = spin_orbital(q, "up"), spin_orbital(q, "down")
    return _skip_zero_base(
        (
            (1.0, ((up_p, 1), (up_q, 0))),
            (1.0, ((down_p, 1), (down_q, 0))),
        )
    )


def _double_singlet_base_terms(
    p: int,
    q: int,
    r: int,
    s: int,
) -> tuple[tuple[complex, NormalTerm], ...]:
    """Base terms of the singlet double of Eq. (E2), before ``- h.c.``.

    Four interleaved products ``a_r^dagger a_p a_s^dagger a_q`` with weights
    ``1/2, 1/2, -1/2, -1/2`` and the spin labels written below. The two
    pairings that keep each electron's spin come first, and the two that
    exchange spins follow. The module docstring derives why
    ``B - B^dagger`` equals Eq. (E2) with creation indices
    ``(p, q)``, or Magoulas and Evangelista, arXiv:2511.13485v2, Eq. (6), with
    ``(P, Q, R, S) = (p, q, r, s)``. The factor 1/2 is overwritten by the
    unit normalization. Returns ``()`` when ``B - B^dagger`` vanishes.
    """
    up_p, down_p = spin_orbital(p, "up"), spin_orbital(p, "down")
    up_q, down_q = spin_orbital(q, "up"), spin_orbital(q, "down")
    up_r, down_r = spin_orbital(r, "up"), spin_orbital(r, "down")
    up_s, down_s = spin_orbital(s, "up"), spin_orbital(s, "down")
    return _skip_zero_base(
        (
            (0.5, ((up_r, 1), (up_p, 0), (down_s, 1), (down_q, 0))),
            (0.5, ((down_r, 1), (down_p, 0), (up_s, 1), (up_q, 0))),
            (-0.5, ((up_r, 1), (down_p, 0), (down_s, 1), (up_q, 0))),
            (-0.5, ((down_r, 1), (up_p, 0), (up_s, 1), (down_q, 0))),
        )
    )


def _double_triplet_base_terms(
    p: int,
    q: int,
    r: int,
    s: int,
) -> tuple[tuple[complex, NormalTerm], ...]:
    """Base terms of the triplet double of Eq. (E3), before ``- h.c.``.

    Six interleaved products ``a_r^dagger a_p a_s^dagger a_q`` with weights
    ``2, 2, 1, 1, 1, 1`` over ``sqrt(12)``, the same-spin patterns first.
    These are the six terms and relative weights of Eq. (E3). As for the
    singlet, ``B - B^dagger`` equals Eq. (E3) with creation indices
    ``(p, q)`` (module docstring) and Magoulas and Evangelista,
    arXiv:2511.13485v2, Eq. (7). The
    overall factor is overwritten by the unit normalization. Returns ``()``
    when ``B - B^dagger`` vanishes, for example when ``p == q``.
    """
    up_p, down_p = spin_orbital(p, "up"), spin_orbital(p, "down")
    up_q, down_q = spin_orbital(q, "up"), spin_orbital(q, "down")
    up_r, down_r = spin_orbital(r, "up"), spin_orbital(r, "down")
    up_s, down_s = spin_orbital(s, "up"), spin_orbital(s, "down")
    return _skip_zero_base(
        (
            (2.0 / sqrt(12.0), ((up_r, 1), (up_p, 0), (up_s, 1), (up_q, 0))),
            (2.0 / sqrt(12.0), ((down_r, 1), (down_p, 0), (down_s, 1), (down_q, 0))),
            (1.0 / sqrt(12.0), ((up_r, 1), (up_p, 0), (down_s, 1), (down_q, 0))),
            (1.0 / sqrt(12.0), ((down_r, 1), (down_p, 0), (up_s, 1), (up_q, 0))),
            (1.0 / sqrt(12.0), ((up_r, 1), (down_p, 0), (down_s, 1), (up_q, 0))),
            (1.0 / sqrt(12.0), ((down_r, 1), (up_p, 0), (up_s, 1), (down_q, 0))),
        )
    )


def _skip_zero_base(
    base_terms: tuple[tuple[complex, NormalTerm], ...],
) -> tuple[tuple[complex, NormalTerm], ...]:
    """Return ``base_terms``, or ``()`` when ``B - B^dagger`` normal-orders to zero.

    The pool enumerations skip an empty result, so an operator that vanishes
    identically, such as a triplet with ``p == q``, never gets a pool index.
    """
    normal_terms = _normalized_anti_hermitian_terms(base_terms, normalize=False)
    return () if not normal_terms else base_terms


def _normalized_anti_hermitian_terms(
    base_terms: Iterable[tuple[complex, NormalTerm]],
    *,
    normalize: bool = True,
) -> dict[NormalTerm, complex]:
    """Return the normal-ordered terms of ``B - B^dagger``, optionally at unit coefficient 2-norm.

    The unit norm counts both ``B`` and ``-B^dagger`` terms. A paper operator
    whose terms have norm sqrt(2) in this measure, such as the spin-adapted
    doubles of Magoulas and Evangelista, arXiv:2511.13485v2, therefore
    corresponds to the library
    generator times sqrt(2), which is why the compact circuits divide the
    library angle by sqrt(2).
    """
    raw_terms: list[tuple[complex, NormalTerm]] = []
    for coefficient, ops in base_terms:
        raw_terms.append((coefficient, ops))
        raw_terms.append((-complex(coefficient).conjugate(), _hermitian_conjugate_ops(ops)))
    normal_terms = _normal_ordered_operator(raw_terms)
    if not normal_terms or not normalize:
        return normal_terms
    coefficient_norm = sqrt(sum(abs(coefficient) ** 2 for coefficient in normal_terms.values()))
    return {ops: coefficient / coefficient_norm for ops, coefficient in normal_terms.items()}


def _hermitian_conjugate_ops(ops: NormalTerm) -> NormalTerm:
    """Return the ladder product of the adjoint: order reversed and each action flipped."""
    return tuple((mode, 1 - action) for mode, action in reversed(ops))  # type: ignore[misc]


def _normal_ordered_operator(
    terms: Iterable[tuple[complex, NormalTerm]],
    *,
    coefficient_cutoff: float = GENERATOR_COEFFICIENT_CUTOFF,
) -> dict[NormalTerm, complex]:
    """Coalesce before pruning, keeping O(contributions) scalar workspace."""
    coefficient_buckets: dict[NormalTerm, array] = {}
    for coefficient, ops in terms:
        for normal_ops, normal_coefficient in _normal_ordered_term(
            ops, coefficient
        ).items():
            if normal_ops not in coefficient_buckets:
                coefficient_buckets[normal_ops] = array("d")
            coefficient_buckets[normal_ops].extend((normal_coefficient.real, normal_coefficient.imag))
    combined = {}
    for ops, values in coefficient_buckets.items():
        coeff = stable_complex_sum(islice(values, 0, None, 2), islice(values, 1, None, 2))
        if hypot(coeff.real, coeff.imag) > coefficient_cutoff:
            combined[ops] = coeff
    return combined


def _normal_ordered_term(ops: NormalTerm, coefficient: complex) -> dict[NormalTerm, complex]:
    """Normal order one product with the canonical anticommutation relations.

    Normal order puts creations left of annihilations and, within each kind,
    higher modes first (``_out_of_normal_order``). Each swap of adjacent
    operators flips the sign, and swapping ``a_p a_p^dagger`` also adds the
    contracted term. A repeated operator makes the product vanish.
    """
    terms: dict[NormalTerm, complex] = {tuple(ops): complex(coefficient)}
    while True:
        changed = False
        updated: defaultdict[NormalTerm, complex] = defaultdict(complex)
        for current_ops, current_coefficient in terms.items():
            for index in range(len(current_ops) - 1):
                left, right = current_ops[index], current_ops[index + 1]
                if left == right:
                    changed = True
                    break
                if _out_of_normal_order(left, right):
                    changed = True
                    before = current_ops[:index]
                    after = current_ops[index + 2 :]
                    left_mode, left_action = left
                    right_mode, right_action = right
                    if left_action == 0 and right_action == 1 and left_mode == right_mode:
                        updated[before + after] += current_coefficient
                    updated[before + (right, left) + after] -= current_coefficient
                    break
            else:
                updated[current_ops] += current_coefficient
        terms = {
            ops_key: coeff
            for ops_key, coeff in updated.items()
            if coeff != 0
        }
        if not changed:
            return terms


def _out_of_normal_order(left: FermionOp, right: FermionOp) -> bool:
    """Return whether adjacent ``left, right`` must swap: creations first, then higher modes first."""
    left_mode, left_action = left
    right_mode, right_action = right
    left_key = (0 if left_action == 1 else 1, -left_mode)
    right_key = (0 if right_action == 1 else 1, -right_mode)
    return left_key > right_key


def _fermion_terms_to_pauli(
    terms: dict[NormalTerm, complex],
    num_qubits: int,
    *,
    coefficient_cutoff: float = GENERATOR_COEFFICIENT_CUTOFF,
) -> SparsePauliOp:
    """Return the Jordan-Wigner ``SparsePauliOp`` of ``{operators: coefficient}`` terms.

    Summed Pauli coefficients at or below ``coefficient_cutoff`` are removed.
    The XACC reader passes its own cutoff, zero by default.
    """
    return _ladder_expansion_to_pauli(
        tuple((coefficient, ops) for ops, coefficient in terms.items()),
        num_qubits,
        mapping="jw",
        coefficient_cutoff=coefficient_cutoff,
    )


__all__ = [
    "FermionicGenerator",
    "apply_generator",
    "apply_generator_exponential",
    "apply_spin_squared",
    "enumerate_ceo_ovp_pool",
    "enumerate_qeb_sd_pool",
    "enumerate_spin_adapted_gsd_pool",
    "enumerate_uccsd_sd_pool",
    "spin_orbital",
]
