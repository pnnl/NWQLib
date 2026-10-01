"""Exact sector-expectation QA helpers for GCiM states."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nwqlib._numerics import stable_vector_norm
from nwqlib.subroutines.fermionic_pool import apply_spin_squared


def sector_expectations(
    state: np.ndarray,
    *,
    num_qubits: int | None = None,
    reference_spin: float = 0.0,
) -> dict[str, Any]:
    """Return exact ``<N>``, ``<S_z>``, ``<S^2>`` QA metadata for a state.

    Qubit ``j`` is spin orbital ``j`` in the interleaved order of
    ``fermionic_pool.spin_orbital``. Even qubits are alpha (``S_z = +1/2``)
    and odd qubits beta. ``<N>`` and ``<S_z>`` are diagonal in the occupation
    basis and come from bit probabilities. ``<S^2>`` uses the matrix-free
    action of ``apply_spin_squared``.

    Particle number and spin-z expectations sum bit-one probability
    marginals from reshape views. Their values have the same exact
    population as the occupation-diagonal dots, with reduction-order error
    charged separately. Spin-squared keeps its own action and workspace.
    For the computed probability vector p, ``m_j = sum_(x: x_j = 1) p_x``
    gives ``<N> = sum_j m_j`` and ``<S_z> = (1/2) sum_j (-1)**j m_j`` by
    exchanging two finite sums. The view ``p.reshape(-1, 2, 1 << j)[:, 1, :]``
    supplies the bit-one slice without an index vector, and summing all its
    entries needs no contiguous flattened copy. At most q marginal scalars
    are stored, then ``fsum`` forms the two weighted sums. Multiplication by
    one half is exact when it does not underflow. With ``P = sum p_x``,
    ``M = N/2`` and ``gamma(r) = r*u/(1 - r*u)``, each marginal error is at
    most ``gamma(M - 1)*m_j`` for normal arithmetic, and the comparison with
    the former occupation-diagonal dots on the same p is
    ``T_N <= q*P*[gamma(N) + gamma(M - 1) + u*(1 + gamma(M - 1))]`` for
    ``<N>`` and ``T_{S_z} <= T_N/2`` (Higham, Accuracy and Stability of
    Numerical Algorithms, 2nd ed., Lemma 3.1 and Chapter 3,
    doi:10.1137/1.9780898718027). Absolute underflow terms are added if
    reached. ``spin_contamination`` is
    ``<S^2> - s(s+1)`` for ``s = reference_spin`` and keeps its sign. These
    expectations describe the state. A value equal to a sector's eigenvalue
    does not prove membership in that sector.
    """

    vector = np.asarray(state, dtype=complex).reshape(-1)
    if num_qubits is None:
        if vector.size < 1 or vector.size & (vector.size - 1):
            raise ValueError("state length must be a power of two")
        num_qubits = int(round(math.log2(vector.size)))
    if vector.size != 2**num_qubits:
        raise ValueError("state length must match num_qubits")
    norm = stable_vector_norm(vector)
    if norm == 0.0:
        raise ValueError("state must be nonzero")
    vector = vector / norm
    probabilities = np.abs(vector) ** 2
    marginals = [
        float(probabilities.reshape(-1, 2, 1 << qubit)[:, 1, :].sum())
        for qubit in range(num_qubits)
    ]
    particle_number = math.fsum(marginals)
    spin_z = 0.5 * math.fsum(m if qubit % 2 == 0 else -m for qubit, m in enumerate(marginals))
    spin_squared = _expectation_action(apply_spin_squared(vector, num_qubits=num_qubits), vector)
    spin_contamination = spin_squared - reference_spin * (reference_spin + 1.0)
    return {
        "label": "QA metadata",
        "evaluation": "exact bit probabilities and matrix-free Pauli action",
        "validation_gate": False,
        "particle_number": particle_number,
        "spin_z": spin_z,
        "spin_squared": spin_squared,
        "reference_spin": float(reference_spin),
        "spin_contamination": spin_contamination,
    }


def _expectation_action(action: np.ndarray, state: np.ndarray) -> float:
    # S^2 is Hermitian, so on a unit state its expectation is real up to
    # roundoff. The absolute 1e-10 window is an engineering choice for these
    # O(num_qubits**2)-sized values and does not scale with the state
    # dimension. A larger imaginary part is treated as a wrong action or input.
    value = complex(np.vdot(state, action))
    if abs(value.imag) > 1.0e-10:
        raise ValueError(f"sector expectation has non-real component {value.imag}")
    return float(value.real)


def _sector_work(num_qubits: int) -> int:
    """Return the work charged for one ``sector_expectations`` call on ``2**num_qubits`` amplitudes.

    The unit is one amplitude visited by one elementwise pass or reduction.
    With ``q = num_qubits`` (even) and ``N = 2**q``:

    - ``q*N/2 + 4q + 2``: the occupation statistics, q bit-one marginal
      slices of ``N/2`` entries, their q stored scalars and the two final
      ``fsum`` passes.
    - ``9 (q/2)`` passes: the ``S_z`` diagonal of ``apply_spin_squared``,
      nine passes per spatial orbital.
    - ``15`` passes: the norm, normalization, squared moduli, the two finite
      scans of the factor actions, the ``S_z (S_z + 1)`` combination and the
      final inner product.
    - ``S_- S_+`` as two sequential Pauli-sum actions, each with the ``2q``
      Jordan-Wigner terms of ``S_+`` or ``S_-``, charged by the shared
      action owner ``pauli_action_requirements`` at one column with the
      count-only flip-group bound ``g = min(2q, N)``. Each charge covers
      the attempted grouped pass and a possible ordered retry. The factors
      are never multiplied out into a product of ``(2q)**2`` terms.

    So ``W_sector = N*[9*(q/2) + 15] + q*N/2 + 4q + 2 + 2*W_action``.
    Mapping ``S_+`` and ``S_-`` to Pauli sums does not scale with ``N`` and is
    admitted by its own law in ``fermionic_pool._native_ladder_sum``. The
    occupation-statistic scratch is ``8q + 65536`` bytes beyond the existing
    probabilities, with no additional N-vector. ``apply_spin_squared`` keeps
    its own index and diagonal workspace for ``S^2``.
    """
    from nwqlib.operators._pauli import pauli_action_requirements

    q = num_qubits
    n = 1 << q
    terms = 2 * q
    action = pauli_action_requirements(n, terms, min(terms, n))[1]
    return n * (9 * (q // 2) + 15) + q * n // 2 + 4 * q + 2 + 2 * action


__all__ = ["sector_expectations"]
