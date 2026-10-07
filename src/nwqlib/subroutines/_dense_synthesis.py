"""Synthesis of dense unitaries into CX and one-qubit gates, exact to rounding.

NWQLib's error models treat a dense unitary that a construction appends, such
as an LCHS ``dense_exact`` branch or a dense block-encoding dilation, as the
exact matrix. When a construction adds controls, or a backend lowers the
circuit to a gate basis, the circuit that replaces the unitary must
therefore equal it to binary64 rounding. Qiskit 2.5.2 does
not guarantee this. Its ``TwoQubitBasisDecomposer`` always decomposes the
target with ``TwoQubitWeylDecomposition`` at fidelity ``1 - 1e-9``, which
replaces the Weyl coordinates by those of a special class whenever the
average gate fidelity of the replacement reaches that value
(``crates/synthesis/src/two_qubit_decompose/basis_decomposer.rs``,
``call_inner``, and ``weyl_decomposition.rs``, ``new_inner``). A two-qubit
unitary near the identity thus loses its entangling part, with entry
errors measured up to 4.2e-5. Qiskit's quantum Shannon decomposition also
omits multiplexed rotations of at most 1e-10 rad (``qsd.rs``,
``get_ucrz``). ``UnitaryGate.control`` keeps that decomposition of the
controlled matrix when ``numpy.allclose`` with ``atol=1e-7`` and
``rtol=1e-5`` accepts it (``generalized_gates/unitary.py``, ``control``).
``qiskit_compat.controlled`` therefore synthesizes a controlled
``UnitaryGate`` with :func:`controlled_unitary_circuit` and replaces every
dense ``UnitaryGate`` in a definition by :func:`dense_unitary_circuit`
before it adds controls. ``qiskit_compat.exact_dense_unitaries`` replaces
them before a backend lowers a circuit to a gate basis, and the MPS state
preparation synthesizes its two-qubit gates with
:func:`dense_unitary_circuit`.

The construction:

- One qubit: one ``U`` gate with the ZYZ angles of Qiskit's
  ``OneQubitEulerDecomposer("U")`` (:func:`_euler_u`).
- Two qubits: the KAK decomposition ``U = e^{i g} (A1 (x) C1) N(a, b, c)
  (A2 (x) C2)`` with ``N(a, b, c) = exp(i (a XX + b YY + c ZZ))``, computed in
  the magic basis. Shende, Markov and Bullock, arXiv:quant-ph/0308033v3,
  Proposition IV.3 and its proof, give the basis ``E``, the local gates as
  ``E SO(4) E^dagger`` and the shared real eigenbasis of the real and
  imaginary parts of the symmetric unitary ``u u^T``. ``N`` then needs three
  CX by Vatan and Williams, arXiv:quant-ph/0308006v3, Sec. V and Fig. 6.
  Fewer suffice when a coordinate is a multiple of pi/2 (two CX), when two
  are and the third is pi/4 modulo pi/2 (one CX), and when all three are
  (none).
- Three or more qubits: the block-ZXZ decomposition of Krol and Al-Ars,
  arXiv:2403.13692v2, Secs. 4.1 and 4.2, Eqs. (5)-(10), applied
  recursively down to two-qubit blocks. Their Sec. 5.2 and Eq. (11) merge
  the closing CX of the two outer multiplexors into the central block as CZ
  gates, which is similar to optimization A.1 of Shende, Bullock and
  Markov, arXiv:quant-ph/0406176v5, Appendix A. Optimization A.2 of the
  same appendix (Krol and Al-Ars, arXiv:2403.13692v2, Sec. 5.1) moves a
  diagonal out of each two-qubit block into the next, which leaves a two-CX
  block when the product is realizable with two CX (Shende, Markov and
  Bullock, quant-ph/0308033v3, Proposition V.2). A generic n-qubit unitary
  takes ``(22/48) 4**n - (3/2) 2**n + 5/3`` CX (Krol and Al-Ars,
  arXiv:2403.13692v2, abstract and Sec. 5.3), the same count as Qiskit
  2.5.2's ``qs_decomposition``. A
  two-qubit block with a small second Weyl coordinate, near the
  controlled-rotation class or a local gate, can keep three CX, because the
  trace condition of A.2 then fixes the diagonal too poorly to remove the
  last coordinate exactly. With one more CX in every block except the last,
  any input on n >= 2 qubits takes at most
  ``(25/48) 4**n - (3/2) 2**n + 2/3`` CX, the count of the recursion
  without A.2. A matrix that is block diagonal with respect to
  one qubit, such as a controlled unitary, is demultiplexed directly and
  does not use A.2, as in Qiskit. On m >= 3 qubits it takes one
  multiplexor of ``2**(m-1)`` CX and two (m-1)-qubit blocks without A.2,
  each with at most ``(25/48) 4**(m-1) - (3/2) 2**(m-1) + 2/3`` CX, so at
  most ``(25/96) 4**m - 2**m + 4/3`` CX. That is the largest count that
  Qiskit 2.5.2 reached for the same controlled matrices in tests.
  :func:`controlled_synthesis_size` derives the instruction count and the
  classical cost of a controlled matrix, and :func:`dense_synthesis_size`
  those of any input. The multiplexed rotations use the Gray-code schedule
  of ``_multiplexors.gray_code_rotation_schedule``, through its parts
  ``gray_code_rotation_angles`` and ``gray_code_controls``.

Every numerical step is a unitary-preserving factorization (singular value
decomposition, complex Schur form, symmetric eigendecomposition) or exact
gate algebra, so the circuit differs from the input matrix by rounding. The
only other change is deliberate. A Weyl coordinate within
``ROUNDING_WINDOW`` of zero, or of the special value that saves gates, is
replaced by that value, which changes the operator by at most the window in
operator norm. A one-qubit gate within the window of the identity, in
rotation angle and in phase, is omitted, which changes the operator by at
most 1.5 times the window. An off-diagonal block, or the deviation of a
block from a multiple of the identity, whose entries are all within the
window is dropped, which changes each entry by at most the window. For
``U = expm(-i t G)`` with random Hermitian G, t from 1e-10 to 10 and one to
five qubits, the largest entry error measured was 14 unit roundoffs times
the dimension, and ``tests/test_dense_synthesis.py`` checks a bound of 64
unit roundoffs times the dimension. When t times the norm of G is itself of
the order of the window, the deliberate replacements dominate and the
error is of the order of the window, about 1e-14.

The recursion is evaluated one depth at a time (:func:`_qsd`). The blocks
of one depth all have the same size, and those that take the same step, a
block-ZXZ step or a demultiplexing on the same qubit, go through each
factorization, product and elementwise operation of that step as one
stacked array. Most blocks are 4-by-4 or 8-by-8, and for them the Python
overhead of a separate NumPy call exceeds its arithmetic. Each block goes
through the same floating-point operations as in a depth-first recursion,
so the circuit does not depend on the order. The complex Schur
factorization has no stacked form and is called once per block
(:func:`_schur`). Optimization A.2 couples the two-qubit blocks, because a
block can receive the diagonal of the block before it, and
:func:`_apply_a2` evaluates them in speculative batches. The one-qubit
angles of all blocks come from one stacked evaluation (:func:`_euler_u`),
and :func:`_standard_gate_appender` appends the gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nwqlib.subroutines._multiplexors import gray_code_controls, gray_code_rotation_angles

# Largest deviation that the synthesis treats as zero or as an exact special
# value: a Weyl coordinate from a multiple of pi/4, a one-qubit rotation from
# the identity, or a block entry from zero or from a multiple of the
# identity. It is 128 binary64 unit roundoffs (2**-53 each), far above the
# rounding noise of these quantities for exactly special inputs (about
# 1e-16), and each replacement changes the operator by rounding only.
# Registered in docs/ENGINEERING_CONSTANTS.md.
ROUNDING_WINDOW = 2.0**-46

_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.diag([1, 1j])
_SDG = np.diag([1, -1j])
_PAULIS = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.diag([1.0 + 0j, -1.0]),
)
# Magic basis E of Shende, Markov and Bullock, quant-ph/0308033v3, proof of
# Proposition IV.3, as columns. E^dagger (A (x) C) E is real orthogonal for
# A, C in SU(2), and E^dagger (P (x) P) E is diagonal for each Pauli P.
_MAGIC = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
_MAGIC_ADJOINT = _MAGIC.conj().T
# E^dagger N(a, b, c) E = diag(exp(i theta)) with theta = _WEYL_MAP @ (a, b, c),
# read off the diagonals of E^dagger XX E, E^dagger YY E and E^dagger ZZ E.
# The columns are orthogonal with squared norm 4 and each sums to zero.
_WEYL_MAP = np.array([[1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1]], dtype=float)


def _euler_u(matrices: np.ndarray):
    """Return (theta, phi, lam, phase) with ``matrices[k] = exp(i phase) U(theta, phi, lam)`` for a (K, 2, 2) stack.

    The formulas are those of Qiskit 2.5.2's ``OneQubitEulerDecomposer("U")``
    (``params_zyz_inner`` and ``params_u3_inner`` in
    ``crates/synthesis/src/euler_one_qubit_decomposer.rs``). With
    ``det = m00 m11 - m01 m10``, ``theta = 2 atan2(|m10|, |m00|)``,
    ``phi = arg(m11) + arg(m10) - arg(det)``, ``lam = arg(m11) - arg(m10)``
    and ``phase = arg(det)/2 - (phi + lam)/2``. The complex products are
    written as the real products of that Rust code and the moduli as
    ``hypot``, so that the angles equal Qiskit's, and for 20,000 random and
    special unitaries all four agreed in every bit (macOS arm64). One NumPy
    evaluation covers the whole stack, where Qiskit's decomposer takes one
    call per gate.
    """
    a, b, c, d = matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 0], matrices[:, 1, 1]
    det_real = (a.real * d.real - a.imag * d.imag) - (b.real * c.real - b.imag * c.imag)
    det_imag = (a.real * d.imag + a.imag * d.real) - (b.real * c.imag + b.imag * c.real)
    det_arg = np.arctan2(det_imag, det_real)
    theta = 2.0 * np.arctan2(np.hypot(c.real, c.imag), np.hypot(a.real, a.imag))
    ang1 = np.arctan2(d.imag, d.real)
    ang2 = np.arctan2(c.imag, c.real)
    phi = ang1 + ang2 - det_arg
    lam = ang1 - ang2
    return theta, phi, lam, 0.5 * det_arg - 0.5 * (phi + lam)


def _one_qubit_gates(matrices: np.ndarray):
    """Return (theta, phi, lam, keep, phase) of the U gates for a (K, 2, 2) stack of unitaries.

    ``keep`` is False for a rotation within ROUNDING_WINDOW of the identity,
    whose gate is omitted. Its global phase still counts.
    """
    theta, phi, lam, phase = _euler_u(matrices)
    # U(0, phi, lam) = diag(1, exp(i (phi + lam))).
    residual = (phi + lam + np.pi) % (2 * np.pi) - np.pi
    keep = (np.abs(theta) > ROUNDING_WINDOW) | (np.abs(residual) > ROUNDING_WINDOW)
    return theta, phi, lam, keep, phase


def _rz(angle: Any) -> np.ndarray:
    """Return RZ(angle), or a stack of them for an array of angles."""
    angle = np.asarray(angle, dtype=float)
    matrix = np.zeros(angle.shape + (2, 2), dtype=complex)
    matrix[..., 0, 0] = np.exp(-0.5j * angle)
    matrix[..., 1, 1] = np.exp(0.5j * angle)
    return matrix


def _ry(angle: Any) -> np.ndarray:
    """Return RY(angle), or a stack of them for an array of angles."""
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    matrix = np.empty(angle.shape + (2, 2), dtype=complex)
    matrix[..., 0, 0] = matrix[..., 1, 1] = c
    matrix[..., 0, 1] = -s
    matrix[..., 1, 0] = s
    return matrix


def _rx(angle: Any) -> np.ndarray:
    """Return RX(angle), or a stack of them for an array of angles."""
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    matrix = np.empty(angle.shape + (2, 2), dtype=complex)
    matrix[..., 0, 0] = matrix[..., 1, 1] = c
    matrix[..., 0, 1] = matrix[..., 1, 0] = -1j * s
    return matrix


_RZ_MINUS_HALF_PI = _rz(-np.pi / 2)
_RZ_HALF_PI = _rz(np.pi / 2)
_RX_HALF_PI = _rx(np.pi / 2)
_RX_MINUS_HALF_PI = _rx(-np.pi / 2)
# The six pairs (j, k) with j < k of the four eigenvalues of a symmetric unitary.
_PAIRS = np.triu_indices(4, 1)


def _polar(matrix: np.ndarray) -> np.ndarray:
    """Return the unitary polar factor of a matrix or of each matrix of a stack, the closest unitary in every unitarily invariant norm."""
    left, _, right = np.linalg.svd(matrix)
    return left @ right


def _schur(products: np.ndarray):
    """Return the Schur diagonals and Schur vectors of each matrix of a complex (K, h, h) stack.

    ``scipy.linalg.schur(a, output="complex")`` calls LAPACK ``zgees``
    after a workspace query for every matrix. The query depends only on h,
    so it is made once here, and each matrix then takes the same ``zgees``
    call with the same workspace, which gives the same factors.
    """
    from scipy.linalg import lapack

    count, size = products.shape[:2]
    eigenvalues = np.empty((count, size), dtype=complex)
    vectors = np.empty((count, size, size), dtype=complex)
    if not count:
        return eigenvalues, vectors
    lwork = int(lapack.zgees(_unsorted, products[0], lwork=-1)[-2][0].real)
    for index in range(count):
        triangular, _, _, schur_vectors, _, info = lapack.zgees(_unsorted, products[index], lwork=lwork)
        if info:
            raise np.linalg.LinAlgError(f"complex Schur factorization failed with LAPACK info {info}")
        eigenvalues[index] = np.diagonal(triangular)
        vectors[index] = schur_vectors
    return eigenvalues, vectors


def _unsorted(value):
    """Eigenvalue selector that ``zgees`` requires and never calls when it does not sort."""
    return None


def _real_eigenbasis(symmetric_unitary: np.ndarray) -> np.ndarray:
    """Return real orthogonal P with P^T M P diagonal for each symmetric unitary M of a (K, 4, 4) stack.

    The real and imaginary parts of M commute, so they share a real
    orthonormal eigenbasis, which is the eigenbasis of every real
    combination Re(exp(-i beta) M) (Shende, Markov and Bullock,
    quant-ph/0308033v3, end of the proof of Proposition IV.3). The
    combination separates eigenvalues l_j and l_k of M by
    |l_j - l_k| |cos(arg(l_j - l_k) - beta)|. Choosing beta in the middle of
    the widest gap between the directions arg(l_j - l_k) + pi/2 modulo pi
    keeps every cosine at least sin(pi/12), so the eigenvector errors of the
    symmetric eigensolver leave off-diagonal residuals of order the unit
    roundoff. Qiskit's ``TwoQubitWeylDecomposition`` instead draws the
    combination at random and accepts the basis when the reconstruction
    matches to 1e-13, which bounds its two-qubit accuracy. Equal eigenvalues
    give no direction, and beta is zero when all four are equal.
    """
    eigenvalues = np.linalg.eigvals(symmetric_unitary)
    first, second = _PAIRS
    distinct = eigenvalues[:, first] != eigenvalues[:, second]
    directions = (np.angle(eigenvalues[:, first] - eigenvalues[:, second]) + np.pi / 2) % np.pi
    beta = np.zeros(len(eigenvalues))
    all_distinct = distinct.all(axis=1)
    beta[all_distinct] = _widest_gap_middle(directions[all_distinct])
    for row in np.flatnonzero(distinct.any(axis=1) & ~all_distinct):
        beta[row] = _widest_gap_middle(directions[row, distinct[row]][None])[0]
    combination = (np.exp(-1j * beta)[:, None, None] * symmetric_unitary).real
    return np.linalg.eigh((combination + np.swapaxes(combination, 1, 2)) / 2)[1]


def _widest_gap_middle(directions: np.ndarray) -> np.ndarray:
    """Return, for each row of angles modulo pi, the middle of the widest gap between cyclically adjacent angles."""
    ordered = np.sort(directions, axis=1)
    # The last gap wraps around from the largest direction to the smallest
    # one plus pi.
    cyclic = np.concatenate([ordered, ordered[:, :1] + np.pi], axis=1)
    gaps = np.diff(cyclic, axis=1)
    widest = np.argmax(gaps, axis=1)
    rows = np.arange(len(widest))
    return cyclic[rows, widest] + gaps[rows, widest] / 2


def _product_factors(local: np.ndarray):
    """Return stacks (A, C) of 2-by-2 unitaries with A (x) C equal to each local 4-by-4 unitary of a (K, 4, 4) stack.

    Entry (2i + j, 2k + l) of A (x) C is A[i, k] C[j, l], so the rearranged
    matrix R[(i, k), (j, l)] is the rank-one product vec(A) vec(C)^T. Its
    leading singular pair gives both factors, each scaled to unit columns.
    """
    rearranged = local.reshape(-1, 2, 2, 2, 2).transpose(0, 1, 3, 2, 4).reshape(-1, 4, 4)
    left, _, right = np.linalg.svd(rearranged)
    return left[:, :, 0].reshape(-1, 2, 2) * np.sqrt(2), right[:, 0].reshape(-1, 2, 2) * np.sqrt(2)


def _weyl(unitaries: np.ndarray):
    """Return the Weyl frame (g, magic, basis, theta) of the KAK decomposition of each two-qubit unitary of a (K, 4, 4) stack.

    The decomposition is U = e^{i g} (A1 (x) C1) N(a, b, c) (A2 (x) C2), and
    in Qiskit's little-endian order A acts on qubit 1 and C on qubit 0.
    With V = U exp(-i g) in SU(4) and ``magic`` = V_E = E^dagger V E, the
    symmetric unitary V_E^T V_E equals P D P^T for a real orthogonal P,
    ``basis``, and a diagonal D = diag(exp(2 i theta)). Then
    V_E = K D^(1/2) P^T with K real orthogonal, E K E^dagger and
    E P^T E^dagger are local, and E D^(1/2) E^dagger = N(a, b, c). The last
    theta is set to minus the sum of the others, which picks a square root
    of D with unit determinant and makes theta lie in the span of the
    columns of _WEYL_MAP. :func:`_reduced_weyl` gives the coordinates
    (a, b, c) and :func:`_local_factors` the local gates. They are separate
    steps because :func:`_apply_a2` needs only the coordinates to choose
    between two matrices, and :func:`_two_qubit_plans` computes the local
    gates once, for the chosen frame.
    """
    g = np.angle(np.linalg.det(unitaries)) / 4
    magic = _MAGIC_ADJOINT @ (unitaries * np.exp(-1j * g)[:, None, None]) @ _MAGIC
    symmetric = np.swapaxes(magic, 1, 2) @ magic
    basis = _real_eigenbasis(symmetric)
    flip = np.linalg.det(basis) < 0
    basis[flip, :, 3] = -basis[flip, :, 3]
    theta = np.angle(np.diagonal(np.swapaxes(basis, 1, 2) @ symmetric @ basis, axis1=1, axis2=2)) / 2
    theta[:, 3] = -theta[:, :3].sum(axis=1)
    return g, magic, basis, theta


def _local_factors(magic: np.ndarray, basis: np.ndarray, theta: np.ndarray):
    """Return the local gates ((A1, C1), (A2, C2)) from the output of :func:`_weyl`."""
    left = _product_factors(_MAGIC @ ((magic @ basis) * np.exp(-1j * theta)[:, None, :]) @ _MAGIC_ADJOINT)
    right = _product_factors(_MAGIC @ np.swapaxes(basis, 1, 2) @ _MAGIC_ADJOINT)
    return left, right


def _reduced_weyl(theta: np.ndarray):
    """Return (turns, coords, zero, quarter) for each row of theta from :func:`_weyl`.

    The Weyl coordinates (a, b, c) = _WEYL_MAP^T theta / 4 are reduced into
    [-pi/4, pi/4] by ``turns`` quarter turns each, using
    exp(i x PP) = i^k (P (x) P)^k exp(i (x - k pi/2) PP). ``zero`` and
    ``quarter`` mark the reduced coordinates within ROUNDING_WINDOW of 0
    and of +-pi/4.
    """
    coords = (_WEYL_MAP.T @ theta[:, :, None])[:, :, 0] / 4
    turns = np.round(coords / (np.pi / 2)).astype(int)
    coords = coords - turns * np.pi / 2
    zero = np.abs(coords) <= ROUNDING_WINDOW
    quarter = np.abs(np.abs(coords) - np.pi / 4) <= ROUNDING_WINDOW
    return turns, coords, zero, quarter


def _cx_counts(zero: np.ndarray, quarter: np.ndarray) -> np.ndarray:
    """Return the CX count of each fewest-CX circuit from the flags of :func:`_reduced_weyl`.

    It is zero when all three reduced coordinates are zero, one when two are
    zero and the third is +-pi/4, two in every other case with a zero
    coordinate and three otherwise (module docstring).
    """
    counts = np.where(zero.any(axis=1), 2, 3)
    counts[(zero.sum(axis=1) == 2) & (quarter & ~zero).any(axis=1)] = 1
    counts[zero.all(axis=1)] = 0
    return counts


# CX gates (control, target) of the fewest-CX circuit with zero to three CX.
_CX_PATTERNS = ((), ((0, 1),), ((0, 1), (0, 1)), ((0, 1), (1, 0), (0, 1)))


def _two_qubit_plans(g: np.ndarray, magic: np.ndarray, basis: np.ndarray, theta: np.ndarray):
    """Return the fewest-CX exact circuits of the two-qubit unitaries whose :func:`_weyl` output is given.

    The result is (phase, offsets, layers, cx). Circuit k has global phase
    ``phase[k]``, the CX gates ``cx[k]``, each given as (control, target),
    and the 2-by-2 matrices ``layers[offsets[k]:offsets[k + 1]]`` on qubits
    0 and 1 alternately, one pair before, between and after the CX gates.
    A coordinate reduced by t quarter turns in :func:`_reduced_weyl` adds
    t pi/2 to the global phase, an odd t multiplies A1 and C1 by the Pauli
    of that coordinate, and the outer layers absorb the local gates. The
    three-CX and two-CX circuits are built for all unitaries of their class
    at once, the rarer ones by :func:`_fewer_cx_plan`.

    A local Clifford W with W P W^dagger = +-Q and W Q W^dagger = +-P
    exchanges the coordinates of PP and QQ, since (W (x) W) N (W (x) W)^dagger
    permutes the three commuting terms. H exchanges X and Z and RX(-pi/2)
    exchanges Y and Z, which moves the first zero coordinate to ZZ for the
    two-CX circuit.
    """
    turns, coords, zero, quarter = _reduced_weyl(theta)
    counts = _cx_counts(zero, quarter)
    (a1, c1), (a2, c2) = _local_factors(magic, basis, theta)
    g = g.copy()
    for index in range(3):
        g += turns[:, index] * np.pi / 2
        odd = turns[:, index] % 2 == 1
        a1[odd] = a1[odd] @ _PAULIS[index]
        c1[odd] = c1[odd] @ _PAULIS[index]
    # A circuit with n CX has n + 1 layers of two one-qubit matrices.
    offsets = np.concatenate([[0], np.cumsum(2 * counts + 2)])
    layers = np.empty((offsets[-1], 2, 2), dtype=complex)
    phase = np.empty(len(g))
    rows = np.flatnonzero(counts == 3)
    if len(rows):
        a, b, c = coords[rows].T
        # Vatan and Williams, quant-ph/0308006v3, Fig. 6, with every rotation
        # angle negated for Qiskit's RZ(t) = exp(-i t Z / 2). The circuit
        # equals exp(-i pi/4) N(a, b, c).
        three = np.empty((len(rows), 4, 2, 2, 2), dtype=complex)
        three[:, 0, 0] = _RZ_MINUS_HALF_PI @ c2[rows]
        three[:, 0, 1] = _I2 @ a2[rows]
        three[:, 1, 0] = _ry(2 * a - np.pi / 2)
        three[:, 1, 1] = _rz(np.pi / 2 - 2 * c)
        three[:, 2, 0] = _ry(np.pi / 2 - 2 * b)
        three[:, 2, 1] = _I2
        three[:, 3, 0] = c1[rows] @ _I2
        three[:, 3, 1] = a1[rows] @ _RZ_HALF_PI
        layers[offsets[rows][:, None] + np.arange(8)] = three.reshape(len(rows), 8, 2, 2)
        phase[rows] = g[rows] + np.pi / 4
    rows = np.flatnonzero(counts == 2)
    if len(rows):
        # Local gates and coordinates after the first zero coordinate moves to ZZ.
        a1_ex, c1_ex, a2_ex, c2_ex, coords_ex = a1[rows], c1[rows], a2[rows], c2[rows], coords[rows]
        first_zero = np.argmax(zero[rows], axis=1)
        for index, w in ((0, _H), (1, _RX_MINUS_HALF_PI)):
            swapped = np.flatnonzero(first_zero == index)
            a1_ex[swapped] = a1_ex[swapped] @ w.conj().T
            c1_ex[swapped] = c1_ex[swapped] @ w.conj().T
            a2_ex[swapped] = w @ a2_ex[swapped]
            c2_ex[swapped] = w @ c2_ex[swapped]
            coords_ex[swapped, index] = coords_ex[swapped, 2]
        # CX (exp(i a X) (x) exp(i b Z)) CX = exp(i (a XX + b ZZ)) on control
        # 0 and target 1, and RX(-pi/2) maps Z to Y, so conjugating by
        # RX(-pi/2) on both qubits turns ZZ into YY.
        two = np.empty((len(rows), 3, 2, 2, 2), dtype=complex)
        two[:, 0, 0] = _RX_HALF_PI @ c2_ex
        two[:, 0, 1] = _RX_HALF_PI @ a2_ex
        two[:, 1, 0] = _rx(-2 * coords_ex[:, 0])
        two[:, 1, 1] = _rz(-2 * coords_ex[:, 1])
        two[:, 2, 0] = c1_ex @ _RX_MINUS_HALF_PI
        two[:, 2, 1] = a1_ex @ _RX_MINUS_HALF_PI
        layers[offsets[rows][:, None] + np.arange(6)] = two.reshape(len(rows), 6, 2, 2)
        phase[rows] = g[rows] + 0.0
    for row in np.flatnonzero(counts < 2).tolist():
        phase[row], row_layers = _fewer_cx_plan(float(g[row]), list(coords[row]), list(zero[row]),
                                                a1[row], c1[row], a2[row], c2[row])
        layers[offsets[row]:offsets[row + 1]] = [matrix for layer in row_layers for matrix in layer]
    return phase, offsets, layers, [_CX_PATTERNS[count] for count in counts.tolist()]


def _fewer_cx_plan(g: float, coords: list, zero: list, a1, c1, a2, c2):
    """Return (phase, layers) of a two-qubit unitary that needs no CX or one CX.

    ``layers`` holds one (qubit 0, qubit 1) pair of 2-by-2 matrices before
    and after the CX gate, or a single pair. For one CX the non-zero
    coordinate is +-pi/4 and moves to XX, with S exchanging X and Y and H
    exchanging X and Z (:func:`_two_qubit_plans`).
    """
    if all(zero):
        layers, phase = [(_I2, _I2)], 0.0
    else:
        index = zero.index(False)
        if index:
            w = _S if index == 1 else _H
            a1, c1 = a1 @ w.conj().T, c1 @ w.conj().T
            a2, c2 = w @ a2, w @ c2
            coords[0] = coords[index]
        if coords[0] < 0:
            # exp(-i pi/4 XX) = -i (X (x) X) exp(i pi/4 XX).
            a1, c1 = a1 @ _PAULIS[0], c1 @ _PAULIS[0]
            g -= np.pi / 2
        # exp(i pi/4 XX) = (H (x) H) exp(i pi/4 ZZ) (H (x) H) and
        # exp(i pi/4 ZZ) = exp(i pi/4) (S^dagger (x) S^dagger) CZ.
        layers, phase = [(_H, _I2), (_H @ _SDG, _H @ _SDG @ _H)], np.pi / 4
    layers = [list(layer) for layer in layers]
    layers[0] = [layers[0][0] @ c2, layers[0][1] @ a2]
    layers[-1] = [c1 @ layers[-1][0], a1 @ layers[-1][1]]
    return g + phase, layers


def _two_qubit_up_to_diagonal(unitary: np.ndarray):
    """Return (delta, V) with ``U = diag(delta) V`` and tr(gamma(V)) real.

    The diagonal is Delta = diag(1, 1, e^{i psi}, e^{-i psi}) applied after
    V, with psi chosen so that tr(gamma(Delta^dagger U)) is real, where
    gamma(V) = V (Y (x) Y) V^T (Y (x) Y). This makes Delta^dagger U
    realizable with two CX (Shende, Markov and Bullock, quant-ph/0308033v3,
    Definition IV.1 and Proposition V.2). Qiskit's
    ``two_qubit_decompose_up_to_diagonal`` uses the same psi. Near the
    controlled-rotation class, where the second Weyl coordinate is small,
    the coefficients that fix psi are small, and V can keep a coordinate
    outside ``ROUNDING_WINDOW`` and three CX. :func:`_apply_a2` then keeps U.
    """
    m = unitary * np.exp(-1j * float(np.angle(np.linalg.det(unitary))) / 4)
    a1 = -m[1, 3] * m[2, 0] + m[1, 2] * m[2, 1] + m[1, 1] * m[2, 2] - m[1, 0] * m[2, 3]
    a2 = m[0, 3] * m[3, 0] - m[0, 2] * m[3, 1] - m[0, 1] * m[3, 2] + m[0, 0] * m[3, 3]
    psi = float(np.arctan2(a1.imag + a2.imag, a1.real - a2.real))
    delta = np.exp(1j * np.array([0.0, 0.0, psi, -psi]))
    return delta, delta.conj()[:, None] * unitary


# Largest number of consecutive two-qubit blocks whose Weyl frames _apply_a2
# computes in one stacked call. Registered in docs/ENGINEERING_CONSTANTS.md.
_A2_BATCH = 32


def _apply_a2(blocks: np.ndarray):
    """Return the Weyl frames of the matrices from which the two-qubit blocks of an A.2 tree are synthesized.

    An A.2 tree is the recursion of a matrix whose top is not block
    diagonal (:func:`_split`), and ``blocks`` are its two-qubit blocks in
    circuit order. Optimization A.2 of Shende, Bullock and Markov,
    quant-ph/0406176v5, Appendix A, moves a diagonal from each block into
    the next. The product of a block and the diagonal Delta that it
    receives, with Delta applied first, is U. When the circuit of U needs three CX and V of
    :func:`_two_qubit_up_to_diagonal` needs at most two, the block is
    synthesized as V and passes its own Delta on. Every other block, the
    last one included, is synthesized as U and passes nothing. Every block
    of such a tree acts on qubits 0 and 1, and every gate between two blocks
    is a multiplexed RZ, a CX or an H on a higher qubit, or a CX controlled
    by qubit 0 or 1, so it commutes with a diagonal on qubits 0 and 1.

    Each block depends on the one before only through the Delta it
    receives, but the decision to pass Delta needs the Weyl frames of U and
    V. The loop therefore takes the blocks in batches of consecutive
    blocks. It first assumes that every block of a batch passes, which needs
    only the polar factor and Delta of each block in turn, and then computes
    the frames of the whole batch in one stacked call. The blocks up to and
    including the first one that does not pass are final, and the next
    batch starts after it without a diagonal. With r the number of blocks
    accepted since the last block that did not pass, a batch holds
    max(1, r // 2) blocks, but no more than ``_A2_BATCH`` or the blocks that
    remain. A batch that fails therefore discards at most max(0, r // 2 - 1)
    blocks, fewer than half of the r + 1 or more blocks accepted by then,
    and the discarded work stays below half of the accepted work. Every
    accepted block goes through the same operations as in a block-by-block
    loop.
    """
    count = len(blocks)
    g = np.empty(count)
    magic = np.empty((count, 4, 4), dtype=complex)
    basis = np.empty((count, 4, 4))
    theta = np.empty((count, 4))
    start, delta, run = 0, None, 0
    while start < count:
        size = min(max(1, run // 2), _A2_BATCH, count - start)
        own = np.empty((size, 4, 4), dtype=complex)
        moved = np.empty((size, 4, 4), dtype=complex)
        for offset in range(size):
            block = blocks[start + offset]
            own[offset] = _polar(block if delta is None else block @ np.diag(delta))
            delta, moved[offset] = _two_qubit_up_to_diagonal(own[offset])
        frames = _weyl(np.concatenate([own, moved]))
        counts = _cx_counts(*_reduced_weyl(frames[3])[2:])
        passes = (counts[:size] > 2) & (counts[size:] <= 2)
        if start + size == count:
            passes[-1] = False
        failed = np.flatnonzero(~passes)
        accepted = int(failed[0]) + 1 if len(failed) else size
        chosen = np.arange(accepted) + np.where(passes[:accepted], size, 0)
        stop = start + accepted
        g[start:stop], magic[start:stop], basis[start:stop], theta[start:stop] = (part[chosen] for part in frames)
        if len(failed):
            delta, run = None, 0
        else:
            run += accepted
        start = stop
    return g, magic, basis, theta


def _multiplex_blocks(matrices: np.ndarray, axis: int, num_qubits: int):
    """Return the blocks (u00, u11, u01, u10) of each matrix of a stack for qubit ``num_qubits - 1 - axis``.

    ``uij`` maps that qubit from value j to value i, and the other qubits
    keep their order.
    """
    count, dim = matrices.shape[:2]
    half = dim // 2
    tensor = matrices.reshape((count,) + (2,) * (2 * num_qubits))
    if axis:
        tensor = np.moveaxis(np.moveaxis(tensor, 1 + axis, 1), 1 + axis + num_qubits, 1 + num_qubits)
    blocks = tensor.reshape(count, 2, half, 2, half)
    return blocks[:, 0, :, 0, :], blocks[:, 1, :, 1, :], blocks[:, 0, :, 1, :], blocks[:, 1, :, 0, :]


def _block_zxz(unitaries: np.ndarray):
    """Return (A1, A2, B, C) of Krol and Al-Ars, arXiv:2403.13692v2, Eqs. (5)-(9), for each matrix of a stack.

    U = (1/2) [[A1, 0], [0, A2]] [[I + B, I - B], [I - B, I + B]] [[I, 0], [0, C]]
    with the polar decompositions X = S_X U_X and Y = S_Y U_Y of the upper
    blocks, C^dagger = i U_Y^dagger U_X, A1 = (S_X + i S_Y) U_X,
    A2 = U21 + U22 (i U_Y^dagger U_X) and B = 2 A1^dagger X - I.
    """
    half = unitaries.shape[1] // 2
    x, y = unitaries[:, :half, :half], unitaries[:, :half, half:]
    lower_left, lower_right = unitaries[:, half:, :half], unitaries[:, half:, half:]
    vx, sx, wx = np.linalg.svd(x)
    vy, sy, wy = np.linalg.svd(y)
    ux, uy = vx @ wx, vy @ wy
    positive_x = (vx * sx[:, None, :]) @ np.swapaxes(vx.conj(), 1, 2)
    positive_y = (vy * sy[:, None, :]) @ np.swapaxes(vy.conj(), 1, 2)
    rotation = 1j * np.swapaxes(uy.conj(), 1, 2) @ ux
    a1 = (positive_x + 1j * positive_y) @ ux
    a2 = lower_left + lower_right @ rotation
    b = 2 * (np.swapaxes(a1.conj(), 1, 2) @ x) - np.eye(half)
    return a1, a2, b, np.swapaxes(rotation.conj(), 1, 2)


def _demultiplex(first: np.ndarray, second: np.ndarray):
    """Return (d, V, W) with diag(U0, U1) = (I (x) V) diag(D, D^dagger) (I (x) W) for stacks of U0 = ``first`` and U1 = ``second``.

    U0 U1^dagger = V D^2 V^dagger and W = D V^dagger U1 (Shende, Bullock and
    Markov, quant-ph/0406176v5, Theorem 12), with D = diag(d). Both blocks
    are first replaced by their unitary polar factors. The complex Schur
    form of the normal matrix U0 U1^dagger gives a unitary V and a
    triangular factor whose off-diagonal part is of the order of the
    rounding error, also for repeated or clustered eigenvalues.
    diag(D, D^dagger) is the Z rotation multiplexor with angles -2 arg(d) on
    the demultiplexed qubit. ``first`` may be a single matrix shared by the
    whole stack.
    """
    first, second = _polar(first), _polar(second)
    eigenvalues, v = _schur(first @ np.swapaxes(second.conj(), -1, -2))
    d = np.sqrt(eigenvalues / np.abs(eigenvalues))
    w = (d[:, :, None] * np.swapaxes(v.conj(), 1, 2)) @ second
    return d, v, w


class _Level:
    """The blocks of one recursion depth in circuit order and what :func:`_append_gates` emits for each.

    ``qubits[k]`` lists the circuit qubits of block k, lowest first.
    ``nodes[k]`` is None for a block within ROUNDING_WINDOW of a multiple of
    the identity, which emits no gate, ``("leaf", start, cx, qubits)`` for a
    two-qubit circuit whose one-qubit gates start at row ``start`` of the
    ``gates`` lists of :func:`_qsd`, ``("one", start, qubit)`` for a
    one-qubit input, ``("demux", child, target, controls, angles)`` for a
    demultiplexed block and ``("zxz", child, target, controls, first,
    middle, last)`` for a block-ZXZ step, whose children start at row
    ``child`` of the next level and whose multiplexors have the listed
    Gray-code angles. ``demultiplexed`` and ``zxz`` hold the rows of the
    blocks of each kind and the rows of their first children. ``phase[k]``
    is the global phase of block k, filled after the leaves, without the
    phases of A.2 leaves, which :func:`_qsd` sums separately.
    """

    def __init__(self, qubits: list):
        self.qubits = qubits
        self.nodes: list = [None] * len(qubits)
        self.phase = np.zeros(len(qubits))
        self.demultiplexed = (np.empty(0, dtype=int), np.empty(0, dtype=int))
        self.zxz = (np.empty(0, dtype=int), np.empty(0, dtype=int))


def _split(level: _Level, matrices: np.ndarray, active: np.ndarray, num_qubits: int, a2: bool | None):
    """Decompose the active blocks of one depth and return (children, their qubits, a2).

    A block that is block diagonal with respect to a qubit is demultiplexed
    into two children. The search runs only where A.2 is off, and ``a2``,
    None at the top, becomes False for a block-diagonal top and True
    otherwise. Every other block takes a block-ZXZ step, whose four
    children are, in circuit order, the W of the first demultiplexing, the
    W and V of the middle one and the V of the last one.
    """
    count = len(matrices)
    half = matrices.shape[1] // 2
    axis_of = np.full(count, -1)
    if a2 is not True:
        candidates = active.copy()
        for axis in range(num_qubits):
            if not candidates.any():
                break
            # Searching the whole stack costs the two passes per qubit that
            # the work law of dense_synthesis_size counts. Each block keeps
            # the first qubit it is block diagonal for, as in a block-by-block
            # search.
            _, _, upper, lower = _multiplex_blocks(matrices, axis, num_qubits)
            off = np.maximum(np.abs(upper).max(axis=(1, 2)), np.abs(lower).max(axis=(1, 2)))
            found = candidates & (off <= ROUNDING_WINDOW)
            axis_of[found] = axis
            candidates &= ~found
        if a2 is None:
            # A.2 stays off below a block-diagonal level, as in Qiskit's
            # qs_decomposition, and is on for a top-level matrix without one.
            a2 = bool(axis_of[0] < 0)
    demultiplexed = np.flatnonzero(axis_of >= 0)
    zxz = np.flatnonzero(active & (axis_of < 0))
    children = np.zeros(count, dtype=int)
    children[demultiplexed] = 2
    children[zxz] = 4
    child = np.cumsum(children) - children
    level.demultiplexed = (demultiplexed, child[demultiplexed])
    level.zxz = (zxz, child[zxz])
    next_matrices = np.empty((children.sum(), half, half), dtype=complex)
    next_qubits: list = [None] * int(children.sum())
    for axis in np.unique(axis_of[demultiplexed]):
        rows = demultiplexed[axis_of[demultiplexed] == axis]
        u00, u11, _, _ = _multiplex_blocks(matrices if len(rows) == count else matrices[rows], axis, num_qubits)
        d, v, w = _demultiplex(u00, u11)
        next_matrices[child[rows]] = w
        next_matrices[child[rows] + 1] = v
        control = num_qubits - 1 - axis
        for row, angles in zip(rows.tolist(), gray_code_rotation_angles(-2 * np.angle(d)).tolist()):
            qubits = level.qubits[row]
            others = qubits[:control] + qubits[control + 1:]
            level.nodes[row] = ("demux", int(child[row]), qubits[control], others, angles)
            next_qubits[child[row]] = next_qubits[child[row] + 1] = others
    if len(zxz):
        a_first, a_second, b, c = _block_zxz(matrices if len(zxz) == count else matrices[zxz])
        d_first, v_first, w_first = _demultiplex(np.eye(half), c)
        d_last, v_last, w_last = _demultiplex(a_first, a_second)
        del a_first, a_second, c
        # The two CX that the outer multiplexors omit become CZ gates next to
        # the Hadamards and are absorbed into the middle block as Z on the
        # top remaining qubit (Krol and Al-Ars, arXiv:2403.13692v2, Sec. 5.2
        # and Eq. (11), similar to optimization A.1 of Shende, Bullock and
        # Markov, quant-ph/0406176v5).
        z = np.diag([1.0] * (half // 2) + [-1.0] * (half // 2))
        d_middle, v_middle, w_middle = _demultiplex(w_last @ v_first, z @ w_last @ b @ v_first @ z)
        next_matrices[child[zxz]] = w_first
        next_matrices[child[zxz] + 1] = w_middle
        next_matrices[child[zxz] + 2] = v_middle
        next_matrices[child[zxz] + 3] = v_last
        tables = gray_code_rotation_angles(-2 * np.angle(np.stack([d_first, d_middle, d_last], axis=1)))
        for row, (first, middle, last) in zip(zxz.tolist(), tables.tolist()):
            qubits = level.qubits[row]
            level.nodes[row] = ("zxz", int(child[row]), qubits[-1], qubits[:-1], first, middle, last)
            next_qubits[child[row]:child[row] + 4] = [qubits[:-1]] * 4
    return next_matrices, next_qubits, a2


def _qsd(matrix: np.ndarray):
    """Return (phase, levels, gates) of the recursive block-ZXZ decomposition of a 2**m unitary.

    The recursion runs one depth at a time. ``levels[d]`` is the
    :class:`_Level` of the blocks of depth d, all of size 2**(m - d), and
    ``gates`` the (theta, phi, lam, keep) lists of the U gates of the
    leaves. A block within ROUNDING_WINDOW of a multiple of the identity
    contributes its phase and no gate. The decomposition ends at two-qubit
    blocks, which are synthesized directly, or, when A.2 is on, after
    :func:`_apply_a2`. The global phase then combines the phases of the
    blocks from the leaves upwards, in the order of a depth-first recursion.
    With A.2 the phases of the leaves are summed in circuit order into a
    separate total, which is added to the phase of the top block last.
    """
    num_qubits = matrix.shape[0].bit_length() - 1
    levels: list[_Level] = []
    matrices, qubits = matrix[None], [tuple(range(num_qubits))]
    a2 = None
    while True:
        n = num_qubits - len(levels)
        level = _Level(qubits)
        levels.append(level)
        deviation = np.abs(matrices - matrices[:, :1, :1] * np.eye(1 << n)).max(axis=(1, 2))
        active = deviation > ROUNDING_WINDOW
        level.phase[~active] = np.angle(matrices[~active, 0, 0])
        if n <= 2 or not active.any():
            break
        matrices, qubits, a2 = _split(level, matrices, active, n, a2)
        del deviation, active
    rows = np.flatnonzero(active) if n <= 2 else np.empty(0, dtype=int)
    a2_phase = 0.0
    gates: tuple = ([], [], [], [])
    if n == 1 and len(rows):
        theta, phi, lam, keep, phase = _one_qubit_gates(matrices)
        gates = (theta.tolist(), phi.tolist(), lam.tolist(), keep.tolist())
        level.nodes[0] = ("one", 0, qubits[0][0])
        level.phase[0] = phase[0]
    elif n == 2 and len(rows):
        deferred = bool(a2) and len(levels) > 1
        leaves = matrices if len(rows) == len(matrices) else matrices[rows]
        del matrices
        frames = _apply_a2(leaves) if deferred else _weyl(_polar(leaves))
        del leaves
        plan_phase, offsets, layers, cx = _two_qubit_plans(*frames)
        del frames
        theta, phi, lam, keep, phase = _one_qubit_gates(layers)
        del layers
        gates = (theta.tolist(), phi.tolist(), lam.tolist(), keep.tolist())
        del theta, phi, lam, keep
        total = plan_phase.copy()
        sizes = np.diff(offsets)
        for position in range(int(sizes.max())):
            has = np.flatnonzero(sizes > position)
            total[has] += phase[offsets[has] + position]
        for index, row in enumerate(rows.tolist()):
            level.nodes[row] = ("leaf", int(offsets[index]), cx[index], level.qubits[row])
        if deferred:
            for value in total.tolist():
                a2_phase += value
        else:
            level.phase[rows] = total
    # The same additions in the same order as a depth-first recursion, which
    # sums the phases of the children of each block from 0.0.
    for depth in range(len(levels) - 2, -1, -1):
        parent, child = levels[depth].phase, levels[depth + 1].phase
        rows, first = levels[depth].demultiplexed
        parent[rows] = (0.0 + child[first]) + child[first + 1]
        rows, first = levels[depth].zxz
        parent[rows] = ((0.0 + child[first]) + ((0.0 + child[first + 1]) + child[first + 2])) + (0.0 + child[first + 3])
    phase = float(levels[0].phase[0])
    if a2:
        phase += a2_phase
    return phase, levels, gates


def _standard_gate_appender(circuit: Any):
    """Return functions (u, cx, rz, h) that append one gate to ``circuit`` on qubits given by index.

    ``QuantumCircuit.u`` and its siblings check and broadcast their
    arguments for every gate, which took about five times as long as
    building the instructions directly for the gates of a 512-square
    controlled matrix. These functions build each instruction with Qiskit's
    ``CircuitInstruction.from_standard`` and ``StandardGate``, as
    ``QuantumCircuit._append_standard_gate`` does in Qiskit 2.5.2, and add
    it with ``QuantumCircuit._append``, which Qiskit
    documents as a fast path for callers that have checked their
    arguments. ``CircuitInstruction.from_standard`` is undocumented and
    ``StandardGate`` is defined in the private module
    ``qiskit._accelerate.circuit`` (docs/dependency_issues.md,
    "Standard-gate instructions"), and the exactness tests in
    ``tests/test_dense_synthesis.py`` fail when either interface changes.
    The circuit must not be inside a control-flow builder.
    """
    from qiskit._accelerate.circuit import StandardGate
    from qiskit.circuit import CircuitInstruction

    bits = circuit.qubits
    append = circuit._append
    make = CircuitInstruction.from_standard
    u_gate, rz_gate = StandardGate.U, StandardGate.RZ
    fixed: dict = {}

    def u(theta, phi, lam, qubit):
        append(make(u_gate, (bits[qubit],), (theta, phi, lam)))

    def cx(control, target):
        key = (control, target)
        if key not in fixed:
            fixed[key] = make(StandardGate.CX, (bits[control], bits[target]), ())
        append(fixed[key])

    def rz(angle, qubit):
        append(make(rz_gate, (bits[qubit],), (angle,)))

    def h(qubit):
        if qubit not in fixed:
            fixed[qubit] = make(StandardGate.H, (bits[qubit],), ())
        append(fixed[qubit])

    return u, cx, rz, h


def _append_gates(circuit: Any, levels: list, gates: tuple) -> None:
    """Append the gates of the decomposition returned by :func:`_qsd` to ``circuit`` in circuit order.

    A block-ZXZ step emits its first child, the first multiplexor without
    its closing CX, an H, the middle child W, the middle multiplexor, the
    middle child V, an H, the last multiplexor without its closing CX in
    reverse order and the last child. A demultiplexed block emits its W,
    its multiplexor and its V.
    """
    u, cx, rz, h = _standard_gate_appender(circuit)
    theta, phi, lam, keep = gates

    def multiplexor(angles, target, controls, *, closing=True, reverse=False):
        selectors = gray_code_controls(len(angles))
        last = len(angles) - 1
        if reverse:
            for index in range(last, -1, -1):
                if index < last:
                    cx(controls[selectors[index]], target)
                if angles[index] != 0.0:
                    rz(angles[index], target)
            return
        for index, angle in enumerate(angles):
            if angle != 0.0:
                rz(angle, target)
            if closing or index < last:
                cx(controls[selectors[index]], target)

    def emit(depth, row):
        node = levels[depth].nodes[row]
        if node is None:
            return
        kind = node[0]
        if kind == "leaf":
            _, start, pattern, qubits = node
            for layer in range(len(pattern) + 1):
                for position, qubit in ((start + 2 * layer, qubits[0]), (start + 2 * layer + 1, qubits[1])):
                    if keep[position]:
                        u(theta[position], phi[position], lam[position], qubit)
                if layer < len(pattern):
                    control, target = pattern[layer]
                    cx(qubits[control], qubits[target])
        elif kind == "one":
            _, start, qubit = node
            if keep[start]:
                u(theta[start], phi[start], lam[start], qubit)
        elif kind == "demux":
            _, child, target, controls, angles = node
            emit(depth + 1, child)
            multiplexor(angles, target, controls)
            emit(depth + 1, child + 1)
        else:
            _, child, target, controls, first, middle, last = node
            emit(depth + 1, child)
            multiplexor(first, target, controls, closing=False)
            h(target)
            emit(depth + 1, child + 1)
            multiplexor(middle, target, controls)
            emit(depth + 1, child + 2)
            h(target)
            multiplexor(last, target, controls, reverse=True)
            emit(depth + 1, child + 3)

    emit(0, 0)


def dense_unitary_circuit(matrix: Any):
    """Return a circuit of U, CX, RZ and H gates equal to a 2**n unitary to rounding.

    The module docstring gives the construction, its CX count and its
    accuracy. For n >= 3 the RZ and H gates come from the multiplexors and
    the block-ZXZ step, as in Qiskit's ``qs_decomposition``, which keeps
    controlled versions of the circuit comparable in cost. The matrix must
    be unitary. :func:`dense_synthesis_size` gives the classical work, of
    order ``8**n``, and the bytes of one call.
    """
    from qiskit import QuantumCircuit

    array = np.asarray(matrix, dtype=complex)
    dim = array.shape[0] if array.ndim == 2 else 0
    if array.shape != (dim, dim) or dim < 2 or dim & (dim - 1):
        raise ValueError("dense synthesis needs a square matrix of power-of-two dimension >= 2")
    if not np.all(np.isfinite(array)):
        raise ValueError("dense synthesis needs finite matrix entries")
    num_qubits = dim.bit_length() - 1
    phase, levels, gates = _qsd(array)
    circuit = QuantumCircuit(num_qubits, global_phase=phase)
    _append_gates(circuit, levels, gates)
    return circuit


def controlled_unitary_circuit(matrix: Any, num_controls: int):
    """Return the exact synthesis of ``matrix`` controlled on qubits ``0`` to ``num_controls - 1``.

    The controls are closed and come first, as in Qiskit's
    ``ControlledGate``. In Qiskit's little-endian order they are the
    low-order bits, so the controlled matrix is the identity except on the
    indices whose low ``num_controls`` bits are all one, where it acts as
    ``matrix``. That matrix is block diagonal with respect to every control
    qubit, and :func:`dense_unitary_circuit` demultiplexes it at the top
    (module docstring). :func:`controlled_synthesis_size` gives the cost.

    The demultiplexing step replaces each diagonal block by its unitary
    polar factor, the nearest unitary, and every later step works on exact
    unitaries. A ``matrix`` that is unitary to rounding is therefore
    realized to rounding. One with a larger unitarity defect, such as a high
    power of an input admitted within a tolerance, is realized as its polar
    factor, which differs from it by about half that defect.
    """
    array = np.asarray(matrix, dtype=complex)
    step = 1 << num_controls
    closed = np.eye(array.shape[0] * step, dtype=complex)
    closed[step - 1::step, step - 1::step] = array
    return dense_unitary_circuit(closed)


def controlled_synthesis_size(num_qubits: int) -> tuple[int, int, int]:
    """Return (work, working bytes, kept bytes) of one :func:`controlled_unitary_circuit` on ``num_qubits`` qubits.

    Here m = ``num_qubits`` >= 1 counts the control and system qubits
    together, and M = 2**m. The laws follow the recursion of :func:`_qsd`
    for a matrix that is block diagonal at the top (module docstring). For
    m = 1, the control of a scalar phase, the circuit is one U gate. A work
    unit is one complex multiply-add. As for the dense completion of
    ``block_encoding/core.py``, a product of two h-square matrices counts
    h**3 units and an SVD or complex Schur factorization 8 h**3.

    - Work ``11 M**3 + (m**2 + 5 m + 256) M**2``. The top demultiplexing
      step makes two SVDs, one Schur factorization and four products of
      size h = M/2, 28 h**3 units. Each generic block-ZXZ step on an
      s-square block makes at most eight SVDs, three Schur factorizations
      and 25 products of size s/2, 113 (s/2)**3 units, and passes four
      blocks of size s/2 on. The blocks of one depth share the SVD of the
      identity. Summing (113/8) s**3 over the recursion, whose terms halve
      at each depth, a generic s-square block costs less than
      (113/4) s**3. The two (m-1)-qubit blocks therefore cost less than
      2 (113/4) (M/2)**3, and the total is below 10.6 M**3. A block that is
      block diagonal again is demultiplexed, which costs less than a
      block-ZXZ step. At most M**2 / 16 two-qubit blocks remain, and 4096
      units cover the 4-square factorizations, products and eight one-qubit
      decompositions of each, about 3800. The identity check of an l-qubit
      block (s = 2**l) makes five elementwise passes over its s**2 entries,
      the search for a block-diagonal qubit two per qubit tried, and a
      block-ZXZ step about ten more (sums, scalings, conjugate transposes,
      the input copies of its factorizations, the dense Z of the middle
      block and the copy of its children into the stack of the next depth).
      The blocks at each recursion depth below the top hold M**2 / 2 entries
      together, so these passes, with building the controlled matrix and
      checking that it is finite, stay below (m**2 + 5 m) M**2.
    - Kept bytes ``176 M**2 + 16384``. A two-qubit block takes at most
      seven U gates and three CX, since one of its four one-qubit layers
      has an identity on one qubit. A demultiplexor on l qubits takes
      ``2**(l-1)`` RZ and ``2**(l-1)`` CX, one CX fewer in the two outer
      ones, and a block-ZXZ step adds two H. A generic l-qubit block without
      A.2 therefore has at most ``(11/8) 4**l - 3 2**l`` instructions, and
      the controlled matrix at most ``(11/16) 4**m - 2**(m+1)`` for m >= 3,
      which a generic base unitary reaches, and ten for m = 2. Qiskit stores
      a standard gate in Rust. The law allows 256 bytes for each of
      ``(11/16) M**2`` slots and 16384 bytes for the circuit, gate and
      register objects. The circuits of a random base unitary with one
      control, rebuilt from their gate lists in a fresh process, held 115
      to 165 bytes of resident memory per instruction for m = 3 to 9 and
      2.8 KB in total for m = 2 (Apple-silicon Mac, Qiskit 2.5.2, averaged
      over up to 256 copies for m <= 5). The law was 1.7 to 7.0 times that
      memory, and 1.95 and 2.03 times for m = 8 and 9.
    - Working bytes ``256 M**2 + 65536``, alive for one synthesis at a
      time. The controlled matrix and the reordered copy that the top
      demultiplexing step reads hold 32 M**2. That step holds its two
      blocks, their polar factors, singular vectors and product, and the
      Schur factors and W, each M**2 / 4 entries, with the copies and
      workspace of one SVD, about 70 M**2. Each deeper depth holds its
      blocks, M**2 / 2 entries, the stack of the next depth and a block-ZXZ
      step on them, below 70 M**2, and the two-qubit blocks, at most
      M**2 / 32 for m >= 3, hold their Weyl frames and the arrays that build
      their circuits, below 60 M**2. The multiplexor and U angles kept as Python floats for
      appending take below 35 M**2. The 65536 bytes cover the fixed Python
      objects of a call. For m = 2 to 9 the peak that tracemalloc measured,
      which excludes the circuit held in Rust, was at most 0.48 of the law,
      and 0.40 for m = 6 to 9.
    """
    size = 1 << num_qubits
    work = 11 * size**3 + (num_qubits**2 + 5 * num_qubits + 256) * size**2
    return work, 256 * size**2 + 65536, 176 * size**2 + 16384


def dense_synthesis_size(num_qubits: int) -> tuple[int, int, int]:
    """Return (work, working bytes, kept bytes) of one :func:`dense_unitary_circuit` on ``num_qubits`` qubits.

    Here m = ``num_qubits`` >= 1 and M = 2**m. The laws hold for any input
    and follow the recursion of :func:`_qsd` (module docstring). The units
    are those of :func:`controlled_synthesis_size`, in which a product of
    two h-square matrices counts h**3 units and an SVD or complex Schur
    factorization 8 h**3.

    - Work ``113 M**3 / 4 + (m**2 + 16 m + 512) M**2``. A generic s-square
      block with s >= 8 takes one block-ZXZ step of at most eight SVDs,
      three Schur factorizations and 25 products of size s/2,
      113 (s/2)**3 units, and passes four blocks of size s/2 on. The terms
      of the recursion halve at each depth, so the blocks cost less than
      (113/4) M**3 in total. A block that is block diagonal with respect to
      a qubit is demultiplexed at 28 (s/2)**3 units into two blocks of size
      s/2, which costs less. At most M**2 / 16 two-qubit blocks remain.
      With optimization A.2 each takes the product that moves the diagonal
      on, its polar factor, the diagonal, the Weyl frames of the block and
      of the block without its diagonal (two determinants, one eigenvalue
      decomposition and one symmetric eigendecomposition each), the local
      gates of one of them, the layers of its circuit and eight one-qubit
      decompositions, about 5300 units. :func:`_apply_a2` discards fewer
      than half as many blocks as it accepts, each after at most its first
      3700 units, so a block costs below 7200 units, and 8192 units per
      block give the 512 M**2. The identity test of a block makes five
      elementwise passes over its entries, and a block-ZXZ step about ten
      more (sums, scalings, conjugate transposes, the input copies of its
      factorizations, the dense Z of the middle block and the copy of its
      children into the stack of the next depth). The blocks at one
      recursion depth hold at most M**2 entries together. The identity test
      runs at the m - 1 depths with blocks of two or more qubits and the
      block-ZXZ step at the m - 2 depths with three or more, so these
      passes stay below (15 m - 25) M**2. The search for a block-diagonal
      qubit makes two passes over a depth per qubit tried,
      2 (m - d) M**2 at depth d and below (m**2 + m) M**2 in total.
      Converting the input and checking that it is finite add two passes.
    - Kept bytes ``352 M**2 + 16384``. For m >= 2 the circuit has at most
      ``7 M**2 / 16`` U, ``25 M**2 / 48`` CX, ``3 M**2 / 8`` RZ and
      ``M**2 / 24`` H gates (:func:`dense_synthesis_gate_census`), at most
      ``11 M**2 / 8`` instructions. The law allows 256 bytes for each of
      these slots and 16384 bytes for the circuit and register objects. The
      circuits of Haar-random inputs, rebuilt from their gate lists in a
      fresh process, held 124 to 173 bytes of resident memory per
      instruction for m = 3 to 9 and 3.9 KB in total for m = 2
      (Apple-silicon Mac, Qiskit 2.5.2, averaged over up to 256 copies for
      m <= 5). The law was 1.7 to 5.6 times that memory, and 1.98 and 2.17
      times for m = 8 and 9.
    - Working bytes ``320 M**2 + 65536``, alive during one synthesis. The
      blocks of one depth hold M**2 entries, 16 M**2 bytes, the input at
      the top. A block-ZXZ step on them holds about 20 arrays of a quarter
      of their entries (singular vectors, polar factors, A1, A2, B and C
      and the factors and operands of the three demultiplexings) with the
      copies and workspace of one SVD, about 110 M**2, and builds the
      stack of the next depth, 16 M**2 more. At the two-qubit depth the
      M**2 / 16 blocks keep their Weyl frames, 424 bytes each, and building
      their circuits holds their local gates, the stacked factorizations of
      one local-gate computation and their one-qubit matrices twice, below
      1.9 KB per block with the frames, or 120 M**2. :func:`_apply_a2`
      holds at most ``_A2_BATCH`` blocks at a time. The multiplexor and U
      angles kept as Python floats for appending take below 70 M**2, and
      the Python records of the blocks below 20 M**2. The 65536 bytes cover
      the fixed Python objects of a call. For m = 1 to 9 the peak that
      tracemalloc measured, which excludes the circuit held in Rust, was at
      most 0.54 of the law, and 0.45 for m = 6 to 9, for Haar-random,
      near-identity and block-diagonal inputs.

    A Haar-random input took 0.027 s at m = 6, 0.12 s at m = 7, 0.57 s at
    m = 8 and 2.6 s at m = 9 on the same machine with one thread, 0.65 to
    1.1 s per 10**9 units of the work law at m = 8 and 9.
    """
    size = 1 << num_qubits
    work = 113 * size**3 // 4 + (num_qubits**2 + 16 * num_qubits + 512) * size**2
    return work, 320 * size**2 + 65536, 352 * size**2 + 16384


def admit_dense_syntheses(qubit_counts, *, max_work: int, max_bytes: int, operation: str,
                          controls=(), controlled_qubit_counts=()) -> None:
    """Raise before any synthesis when the exact syntheses of dense unitaries on ``qubit_counts`` qubits, and Qiskit's control of them, would exceed ``max_work`` or ``max_bytes``.

    Each entry of ``qubit_counts`` stands for one call of
    :func:`dense_unitary_circuit`, sized by :func:`dense_synthesis_size`.
    Each entry of ``controls`` is a triple ``(gates, instructions, heavy)``
    for one call of Qiskit's ``control`` on a gate that holds such
    syntheses, sized by :func:`gatewise_control_size`, and
    :func:`gatewise_control_counts` gives the triple for one synthesis. A
    triple without gates stands for a control call that holds no synthesis
    and is skipped. Each entry of ``controlled_qubit_counts`` is the qubit
    count, controls included, of one call of
    :func:`controlled_unitary_circuit`, sized by
    :func:`controlled_synthesis_size`, which the whole-matrix route makes
    (:func:`select_dense_control_route`). The calls run one at a time, and
    the circuits they return stay alive together, so the bytes are the
    largest working allowance plus every kept allowance. ``operation``
    names the construction in the error message.
    """
    from nwqlib.operators.access import _check_bytes

    sizes = [dense_synthesis_size(count) for count in qubit_counts]
    sizes += [gatewise_control_size(*counts) for counts in controls if counts[0]]
    sizes += [controlled_synthesis_size(count) for count in controlled_qubit_counts]
    if not sizes:
        return
    _check_bytes(max(size[1] for size in sizes) + sum(size[2] for size in sizes), max_bytes, operation)
    if sum(size[0] for size in sizes) > max_work:
        raise ValueError(f"{operation} exceeds max_work")


def dense_synthesis_gate_census(num_qubits: int) -> dict[str, int]:
    """Return upper bounds on the U, CX, RZ and H gates of :func:`dense_unitary_circuit` on ``num_qubits`` qubits.

    One qubit: one U gate, which is also Qiskit's own definition of a
    one-qubit ``UnitaryGate``. Two qubits: the KAK circuit has at most three
    CX and, around them, at most seven U gates, because in the three-CX
    circuit of Vatan and Williams (quant-ph/0308006v3, Fig. 6) the layer
    between the second and third CX is the identity on one qubit and
    :func:`_one_qubit_gates` omits it (the fewer-CX circuits have at most six).
    Three or more qubits: each block-ZXZ level emits four (m-1)-qubit
    children, three multiplexed Z rotations of 2**(m-1) RZ gates each, two
    of them without their closing CX (3*2**(m-1) - 2 CX), and two H gates.
    The recursion ends at 4**(m-2) two-qubit blocks of at most three CX and
    seven U gates. Solving it gives, for m >= 2,

        cx <= (25/48) 4**m - (3/2) 2**m + 2/3,
        u  <= 7 * 4**(m-2),
        rz <= (3/8) 4**m - (3/2) 2**m,
        h  <= (4**m - 16)/24.

    The CX bound is the "any input" count of the module docstring. A
    block-diagonal input takes the demultiplexing path, whose counts are
    smaller at every level, and a block within ``ROUNDING_WINDOW`` of a
    multiple of the identity emits no gate.
    """
    m = num_qubits
    if m == 1:
        return {"cx": 0, "u": 1, "rz": 0, "h": 0}
    return {"cx": (25*4**m - 72*2**m + 32)//48, "u": 7*4**(m-2),
            "rz": (3*4**m - 12*2**m)//8, "h": (4**m - 16)//24}


# Top-level instructions that Qiskit 2.5.2's add_control emits for one U gate
# and for one RZ gate with k controls, stored at index k - 1 for k = 1 to 64
# (qiskit/circuit/_add_control.py, apply_basic_controlled_gate). With one
# control a U becomes one CU and an RZ one CRZ. With k >= 2 an RZ becomes
# QuantumCircuit.mcrz, which composes the circuit of Qiskit's
# _mcsu2_real_diagonal inline: four dirty-ancilla X syntheses on ceil(k/2)
# and floor(k/2) controls around four RZ. A general U becomes MCRZ(lambda),
# MCRY(theta), MCRZ(phi) and an MCPhase on k - 1 controls. The MCRY is two
# RY and two Toffoli gates for k = 2, the Gray-code circuit of seven CU and
# six CX for k = 3, and the MCRZ circuit from k = 4 on, so the U entry is
# three times the RZ entry plus one from k = 4. Qiskit synthesizes the
# dirty-ancilla X gates in Rust (synth_mcx_n_dirty_i15) without a published
# closed-form size, so the table stores the values, which grow by 40 and 120
# per control from k = 8. A U with special angles (theta = phi = 0, or
# phi = lambda = 0, or phi = -pi/2 and lambda = pi/2) emits fewer. The other
# gates of the synthesis emit a fixed number for every k: a CX one X with
# k + 1 controls, an H seven (S, H, T, an X with k controls, Tdg, H and Sdg)
# and the global phase of the controlled gate one P or MCPhase.
# _controlled_heavy_instructions counts the emitted instructions that hold
# more memory than a parameterless standard gate. The tests
# test_gatewise_control_tables_match_installed_qiskit and
# test_gatewise_heavy_counts_match_installed_qiskit recompute every entry.
# Revisit when either fails after a Qiskit upgrade.
CONTROLLED_U_INSTRUCTIONS = (
    1, 21, 86, 193, 289, 385, 607, 829, 949, 1069, 1189, 1309, 1429, 1549, 1669, 1789,
    1909, 2029, 2149, 2269, 2389, 2509, 2629, 2749, 2869, 2989, 3109, 3229, 3349, 3469, 3589, 3709,
    3829, 3949, 4069, 4189, 4309, 4429, 4549, 4669, 4789, 4909, 5029, 5149, 5269, 5389, 5509, 5629,
    5749, 5869, 5989, 6109, 6229, 6349, 6469, 6589, 6709, 6829, 6949, 7069, 7189, 7309, 7429, 7549,
)
CONTROLLED_RZ_INSTRUCTIONS = (
    1, 8, 36, 64, 96, 128, 202, 276, 316, 356, 396, 436, 476, 516, 556, 596,
    636, 676, 716, 756, 796, 836, 876, 916, 956, 996, 1036, 1076, 1116, 1156, 1196, 1236,
    1276, 1316, 1356, 1396, 1436, 1476, 1516, 1556, 1596, 1636, 1676, 1716, 1756, 1796, 1836, 1876,
    1916, 1956, 1996, 2036, 2076, 2116, 2156, 2196, 2236, 2276, 2316, 2356, 2396, 2436, 2476, 2516,
)


def _controlled_heavy_instructions(kind: str, num_controls: int) -> int:
    """Return how many of the instructions that Qiskit emits for one gate of ``kind`` with k controls hold angles or are Python objects.

    A parameterless standard gate is stored in Rust. A standard gate with
    angles also stores them, and an MCX gate with three or more controls and
    an MCPhase gate are Python objects. An RZ gives one CRZ with one
    control. With more, its multi-controlled RZ holds four RZ, and each of
    its dirty-ancilla X blocks on three controls fifteen P gates, which
    occur for k = 5, 6 and 7. A general U gives one CU with one control.
    With more, it gives two such multi-controlled RZ, an MCRY and an
    MCPhase, where the MCRY holds two RY for k = 2, seven CU for k = 3 and
    the gates of the multi-controlled RZ from k = 4. A CX gives one MCX from
    k = 2, an H one from k = 3, and the global phase one P or MCPhase.
    """
    k = num_controls
    if kind in ("u", "rz"):
        if k == 1:
            return 1
        rz = 4 + 30 * (((k + 1) // 2 == 3) + (k // 2 == 3))
        return rz if kind == "rz" else 2 * rz + (2 if k == 2 else 7 if k == 3 else rz) + 1
    return {"cx": int(k >= 2), "h": int(k >= 3), "phase": 1}[kind]


def controlled_synthesis_gate_census(num_qubits: int) -> dict[str, int]:
    """Return upper bounds on the U, CX, RZ and H gates of :func:`controlled_unitary_circuit` on ``num_qubits`` qubits, controls included.

    The controlled matrix is block diagonal with respect to each control
    qubit, so on m >= 3 qubits the synthesis demultiplexes it at the top
    into one multiplexed RZ of ``2**(m-1)`` rotations and ``2**(m-1)`` CX
    and two (m-1)-qubit blocks without optimization A.2 (module
    docstring). Each block stays within :func:`dense_synthesis_gate_census`
    on m - 1 qubits, because a block that is block diagonal again is
    demultiplexed, which emits no more gates of any kind than a block-ZXZ
    step. The CX bound is therefore ``(25/96) 4**m - 2**m + 4/3``. On two
    qubits the synthesis is the two-qubit circuit of
    :func:`dense_synthesis_gate_census`, and a controlled scalar phase on
    one qubit is one U gate.
    """
    m = num_qubits
    if m <= 2:
        return dense_synthesis_gate_census(m)
    block = dense_synthesis_gate_census(m - 1)
    half = 1 << (m - 1)
    return {"cx": 2 * block["cx"] + half, "u": 2 * block["u"], "rz": 2 * block["rz"] + half,
            "h": 2 * block["h"]}


def census_control_counts(census: dict[str, int], num_controls: int) -> tuple[int, int, int]:
    """Return (gates, instructions, heavy) of Qiskit's control of a circuit with the gate census ``census``.

    ``census`` has the keys u, cx, rz and h of :func:`dense_synthesis_gate_census`
    and :func:`controlled_synthesis_gate_census`, and the circuit also has a
    global phase. The counts are those that :func:`gatewise_control_counts`
    derives for one synthesis.
    """
    if not 1 <= num_controls <= len(CONTROLLED_U_INSTRUCTIONS):
        raise ValueError(f"the gate-wise control law covers 1 to {len(CONTROLLED_U_INSTRUCTIONS)} controls")
    gates = sum(census.values()) + 1
    instructions = (census["u"] * CONTROLLED_U_INSTRUCTIONS[num_controls - 1] + census["cx"]
                    + census["rz"] * CONTROLLED_RZ_INSTRUCTIONS[num_controls - 1] + 7 * census["h"] + 1)
    heavy = (sum(count * _controlled_heavy_instructions(kind, num_controls) for kind, count in census.items())
             + _controlled_heavy_instructions("phase", num_controls))
    return gates, instructions, heavy


def gatewise_control_counts(num_qubits: int, num_controls: int) -> tuple[int, int, int]:
    """Return (gates, instructions, heavy) of Qiskit's control of one :func:`dense_unitary_circuit` on ``num_qubits`` qubits.

    ``qiskit_compat.controlled`` replaces a dense ``UnitaryGate`` inside a
    composite gate by its exact synthesis and then calls Qiskit's
    ``control``, which unrolls the composite to its basis and controls each
    gate with ``num_controls`` controls (``qiskit/circuit/_add_control.py``).
    ``gates`` bounds the gates it unrolls, the census of
    :func:`dense_synthesis_gate_census` plus the global phase.
    ``instructions`` bounds the instructions it emits, each census gate
    priced by ``CONTROLLED_U_INSTRUCTIONS``, ``CONTROLLED_RZ_INSTRUCTIONS``
    or the fixed counts beside them, plus one for the phase. ``heavy``
    counts those of them that hold angles or are Python objects
    (:func:`_controlled_heavy_instructions`). A composite that holds several
    syntheses, or one synthesis several times, adds the triples of its
    occurrences. The global phase is counted for each, which only
    overestimates.
    """
    return census_control_counts(dense_synthesis_gate_census(num_qubits), num_controls)


def gatewise_control_size(gates: int, instructions: int, heavy: int) -> tuple[int, int, int]:
    """Return (work, working bytes, kept bytes) of one call of Qiskit's ``control`` with the counts of :func:`gatewise_control_counts`.

    The call unrolls ``gates`` gates and emits ``instructions``
    instructions, ``heavy`` of which hold angles or are Python objects.
    Qiskit's control step does Python and Rust bookkeeping, no
    floating-point linear algebra, so its work has no count of
    multiply-adds. The law sets its units by measured time instead. Per
    unit, the call took 0.45 to 3.3 times as long as
    :func:`dense_synthesis_size` on 4 qubits and 2.6 to 21 times as long on
    7 qubits in the measurements below, and the limits that add both laws
    count work units, not time.

    - Work ``2048 gates + 16 instructions``. Each unrolled gate costs one
      Python call of ``apply_basic_controlled_gate``, 6 to 16 us with one
      control. With more controls a U gate took up to 320 us and an RZ up
      to 98 us for k <= 8, most of it in building the multi-controlled
      rotations, while a CX or H took at most 27 us. Each further emitted
      instruction added 0.1 to 0.2 us (Apple-silicon Mac, one thread,
      Qiskit 2.5.2). For Haar-random syntheses on m = 4 to 7 qubits with
      one to eight controls the call took 4.4 to 35 ns per unit of this
      law, and the synthesis itself 1.6 to 15 ns per unit of its own law.
    - Kept bytes ``96 instructions + 1024 heavy + 16384``. The
      ``ControlledGate`` keeps a deep copy of the emitted definition and of
      the base gate. The copy of the base holds the synthesized circuit,
      which replaces the circuit that :func:`dense_synthesis_size` keeps
      once ``controlled`` returns, so it is not counted again. In circuits
      of one instruction kind, a parameterless standard gate held 40 to 59
      bytes of resident memory, a standard gate with angles 123 to 177, an
      MCX gate 355 and an MCPhase gate 733. The law allows 96 bytes for
      every instruction, 1024 more for each heavy one and 16384 bytes for
      the gate and circuit objects.
    - Working bytes ``1024 gates + 96 instructions + 1024 heavy + 65536``,
      alive during the call. Qiskit unrolls the definition through a DAG, a
      second circuit and the definition of the unrolled gate
      (``_unroll_gate``), and the ``ControlledGate`` constructor deep-copies
      the base gate and the emitted circuit, so four copies of the unrolled
      gates, allowed 256 bytes a slot, and one of the emitted circuit exist
      besides the kept copy. The 65536 bytes cover the fixed Python objects
      of the call.

    For m = 4 to 7 and one to eight controls, kept controlled gates with the
    copy of their base held 67 to 267 bytes of resident memory per emitted
    instruction. The kept laws of the synthesis and of this step together
    were 1.7 to 7.7 times that memory, and this law, working and kept bytes
    together, was 2.1 to 11 times the peak increase of resident memory
    during the call, wherever that increase exceeded 0.1 MB.
    """
    return (2048 * gates + 16 * instructions, 1024 * gates + 96 * instructions + 1024 * heavy + 65536,
            96 * instructions + 1024 * heavy + 16384)


# Engineering constant K of the dense control route, registered in
# docs/ENGINEERING_CONSTANTS.md under "Circuit-free synthesis laws". "auto" synthesizes
# the whole controlled matrix for at most K controls and controls each
# synthesized gate for more. The whole-matrix synthesis on n system qubits
# and k controls costs 11 (2**k M)**3 work units against 28 M**3 for the
# synthesis of the unitary alone, M = 2**n, so its work grows about eight
# times and its bytes four times per added control. With one control it
# costs about 88 M**3 and took 3.2 to 4.0 times fewer CX than the
# gate-wise route for Haar-random unitaries on 2 to 5 qubits, with two
# about 704 M**3 for 3.6 to 3.8 times fewer CX. Two controls take
# 256 (4 M)**2 working and 176 (4 M)**2 kept bytes against 320 M**2 and
# 352 M**2 for the synthesis alone, about 12.8 and 8 times as many.
AUTO_WHOLE_MATRIX_MAX_CONTROLS = 1
DENSE_CONTROL_ROUTES = ("gatewise", "whole_matrix", "auto")


def select_dense_control_route(route: str, num_controls: int) -> str:
    """Return the route, ``"gatewise"`` or ``"whole_matrix"``, by which a construction adds ``num_controls`` controls to a dense unitary.

    ``"gatewise"`` synthesizes the unitary with :func:`dense_unitary_circuit`
    and lets Qiskit control every synthesized gate
    (:func:`gatewise_control_counts`). ``"whole_matrix"`` synthesizes the
    controlled matrix with :func:`controlled_unitary_circuit`
    (:func:`controlled_synthesis_size`). ``"auto"`` takes the whole-matrix
    route for at most ``AUTO_WHOLE_MATRIX_MAX_CONTROLS`` controls and the
    gate-wise route for more, from the number of controls alone.

    Raises:
        ValueError: ``route`` is not one of ``DENSE_CONTROL_ROUTES`` or
            ``num_controls`` is not a positive integer.
    """
    if route not in DENSE_CONTROL_ROUTES:
        raise ValueError(f"dense_control_route must be one of {DENSE_CONTROL_ROUTES}, not {route!r}")
    if type(num_controls) is not int or num_controls < 1:
        raise ValueError("a dense control route needs a positive number of controls")
    if route == "auto":
        return "whole_matrix" if num_controls <= AUTO_WHOLE_MATRIX_MAX_CONTROLS else "gatewise"
    return route


__all__ = [
    "AUTO_WHOLE_MATRIX_MAX_CONTROLS",
    "CONTROLLED_RZ_INSTRUCTIONS",
    "CONTROLLED_U_INSTRUCTIONS",
    "DENSE_CONTROL_ROUTES",
    "ROUNDING_WINDOW",
    "admit_dense_syntheses",
    "census_control_counts",
    "controlled_synthesis_gate_census",
    "controlled_synthesis_size",
    "controlled_unitary_circuit",
    "dense_synthesis_gate_census",
    "dense_synthesis_size",
    "dense_unitary_circuit",
    "gatewise_control_counts",
    "gatewise_control_size",
    "select_dense_control_route",
]
