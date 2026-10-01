"""Validation-scale fermionic references for pool, sector and generator tests.

Conventions match ``nwqlib.subroutines.fermionic_pool``: spin orbital
``g_up = 2*g`` and ``g_down = 2*g + 1`` for zero-based spatial orbital ``g``,
and big-endian Qiskit labels with spin orbital ``j`` at label position
``n_qubits - 1 - j``. Both conventions and the equation labels E5-E7 follow
the appendix of the ADAPT-GCiM paper, arXiv:2312.07691v3 (listed under GCIM in
``docs/references.md``).

The Eq. E6/E7 closed forms are hand-written Pauli strings. The ``jw_*``
references apply the production Jordan-Wigner mapping to the literal ladder
terms of the single excitation and of Eq. E5, so comparing them with the
closed forms checks that mapping against the paper. ``spin_squared_operator``
also maps its ``S_- S_+`` ladder terms with the production mapping. The
particle-number and ``S_z`` operators are diagonal in the occupation basis and
use no mapping. The dense and sparse builders are for small registers only.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sps
from qiskit.quantum_info import SparsePauliOp

from nwqlib.subroutines.fermionic_pool import (
    FermionicGenerator,
    _fermion_terms_to_pauli,
    _normal_ordered_operator,
    spin_orbital,
)


def generator_pauli(generator: FermionicGenerator) -> SparsePauliOp:
    """Return the stored anti-Hermitian Pauli terms as a ``SparsePauliOp``."""

    return SparsePauliOp.from_list(
        generator.pauli_terms or (("I" * generator.num_qubits, 0.0),)
    )


def generator_matrix(generator: FermionicGenerator) -> np.ndarray:
    """Return the dense matrix of the stored Pauli terms."""

    return np.asarray(generator_pauli(generator).to_matrix(), dtype=complex)


def generator_unitary(generator: FermionicGenerator, theta: float) -> np.ndarray:
    """Return ``exp(theta * A)`` by a dense matrix exponential."""

    return la.expm(theta * generator_matrix(generator))


def jw_single_excitation_generator(
    p: int, q: int, *, num_qubits: int | None = None
) -> SparsePauliOp:
    """Return the JW image of ``a_p^dagger a_q - a_q^dagger a_p``."""

    if p == q:
        raise ValueError("single excitation requires distinct spin orbitals")
    qubits = max(p, q) + 1 if num_qubits is None else num_qubits
    return _fermion_terms_to_pauli(
        _normal_ordered_operator(
            (
                (1.0, ((p, 1), (q, 0))),
                (-1.0, ((q, 1), (p, 0))),
            )
        ),
        qubits,
    )


def jw_double_excitation_generator(
    p: int,
    r: int,
    q: int,
    s: int,
    *,
    num_qubits: int | None = None,
) -> SparsePauliOp:
    """Return the JW image of Eq. E5's four-index skew-Hermitian generator."""

    if len({p, r, q, s}) != 4:
        raise ValueError("closed-form double excitation requires four distinct spin orbitals")
    qubits = max(p, r, q, s) + 1 if num_qubits is None else num_qubits
    base_terms = (
        (1.0, ((p, 1), (r, 1), (q, 0), (s, 0))),
        (-1.0, ((s, 1), (q, 1), (r, 0), (p, 0))),
    )
    return _fermion_terms_to_pauli(_normal_ordered_operator(base_terms), qubits)


def closed_form_single_generator(p: int, q: int, *, num_qubits: int | None = None) -> SparsePauliOp:
    """Return the paper Eq. E6 Pauli generator for ``q < p``."""

    if q >= p:
        raise ValueError("Eq. E6 requires q < p")
    qubits = max(p, q) + 1 if num_qubits is None else num_qubits
    terms = [
        (_label_from_ops(qubits, {q: "X", p: "Y", **{k: "Z" for k in range(q + 1, p)}}), -0.5j),
        (_label_from_ops(qubits, {q: "Y", p: "X", **{k: "Z" for k in range(q + 1, p)}}), 0.5j),
    ]
    return SparsePauliOp.from_list(terms).simplify(atol=1.0e-14)


def closed_form_double_generator(
    p: int,
    r: int,
    q: int,
    s: int,
    *,
    num_qubits: int | None = None,
) -> SparsePauliOp:
    """Return the paper Eq. E7 Pauli generator for ``q < s < p < r``."""

    if not (q < s < p < r):
        raise ValueError("Eq. E7 requires q < s < p < r")
    qubits = max(p, r, q, s) + 1 if num_qubits is None else num_qubits
    ladders = {k: "Z" for k in range(p + 1, r)}
    ladders.update({k: "Z" for k in range(q + 1, s)})
    strings = (
        ({q: "X", s: "Y", p: "X", r: "X"}, 1),
        ({q: "Y", s: "X", p: "X", r: "X"}, 1),
        ({q: "Y", s: "Y", p: "Y", r: "X"}, 1),
        ({q: "Y", s: "Y", p: "X", r: "Y"}, 1),
        ({q: "X", s: "X", p: "Y", r: "X"}, -1),
        ({q: "X", s: "X", p: "X", r: "Y"}, -1),
        ({q: "Y", s: "X", p: "Y", r: "Y"}, -1),
        ({q: "X", s: "Y", p: "Y", r: "Y"}, -1),
    )
    terms = [(_label_from_ops(qubits, {**ladders, **ops}), -0.125j * sign) for ops, sign in strings]
    return SparsePauliOp.from_list(terms).simplify(atol=1.0e-14)


def total_particle_number_operator(num_qubits: int) -> sps.csr_matrix:
    """Return the diagonal total particle-number operator."""

    return _diagonal_mode_operator(num_qubits, lambda bits: sum(bits))


def spin_z_operator(num_qubits: int) -> sps.csr_matrix:
    """Return the diagonal ``S_z`` operator for the paper spin convention."""

    if num_qubits % 2:
        raise ValueError("spin_z_operator requires an even number of spin orbitals")
    return _diagonal_mode_operator(
        num_qubits,
        lambda bits: 0.5 * sum(bits[2 * g] - bits[2 * g + 1] for g in range(num_qubits // 2)),
    )


def spin_squared_operator(num_qubits: int) -> sps.csr_matrix:
    """Return ``S^2 = S_- S_+ + S_z(S_z + 1)`` from its four-operator JW expansion."""

    if num_qubits <= 0 or num_qubits % 2:
        raise ValueError("spin_squared_operator requires a positive even number of spin orbitals")
    s_minus_s_plus_terms = []
    for p in range(num_qubits // 2):
        down_p = spin_orbital(p, "down")
        up_p = spin_orbital(p, "up")
        for q in range(num_qubits // 2):
            up_q = spin_orbital(q, "up")
            down_q = spin_orbital(q, "down")
            s_minus_s_plus_terms.append((1.0, ((down_p, 1), (up_p, 0), (up_q, 1), (down_q, 0))))
    s_minus_s_plus = (
        _fermion_terms_to_pauli(
            _normal_ordered_operator(s_minus_s_plus_terms),
            num_qubits,
        )
        .to_matrix(sparse=True)
        .tocsr()
    )
    spin_z = spin_z_operator(num_qubits)
    identity = sps.identity(2**num_qubits, format="csr")
    return (s_minus_s_plus + spin_z @ (spin_z + identity)).tocsr()


def _label_from_ops(num_qubits: int, ops: dict[int, str]) -> str:
    label = ["I"] * num_qubits
    for qubit, pauli in ops.items():
        label[num_qubits - 1 - qubit] = pauli
    return "".join(label)


def _diagonal_mode_operator(num_qubits: int, value_fn) -> sps.csr_matrix:
    values = []
    for basis_index in range(2**num_qubits):
        bits = [(basis_index >> mode) & 1 for mode in range(num_qubits)]
        values.append(float(value_fn(bits)))
    return sps.diags(values, format="csr")
