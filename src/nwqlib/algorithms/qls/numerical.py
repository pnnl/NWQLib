"""Dense spectral estimates and polynomial actions of the classical QLS model.

Planning, the explicit ``execution="classical"`` model and explicit
verification call these kernels. Each acts on the encoded matrix
``A/alpha``, through the original spectral factors or through a shortcut's
augmented ``G_t`` and its Hermitian dilation, with the selected Chebyshev
coefficients, the classical counterpart of the QSVT circuit that
``algorithms/qls/quantum.py`` builds.
"""

from __future__ import annotations
from math import ldexp
import numpy as np
from nwqlib._numerics import binary_scaled_matrix, normalized_vector, stable_vector_norm
from nwqlib.subroutines.dense_matrices import dalzell_augmented_matrix

# Row-tile length r = min(a, PROJECTOR_ROW_TILE) of the rank-one projector
# update in projected_augmented_inplace. It bounds the outer-product
# temporary at 16 r a bytes (the ``16 r a`` summand of
# ``augmentation`` in host_planning.selected_work). The value is a storage choice. Tiling changes
# no per-entry arithmetic.
PROJECTOR_ROW_TILE = 64


def _extreme_singular_values(matrix: np.ndarray, *, hermitian: bool, events) -> tuple[float, float, str]:
    """Return ``(sigma_min, sigma_max, method)`` for square dense input.

    The matrix is first scaled by a power of two, which is exact in binary64
    and keeps extreme magnitudes away from overflow and underflow. The
    endpoints are scaled back exactly. ``hermitian`` is the caller's exact
    admitted Hermiticity metadata (``operator.structure == "hermitian"``),
    so no conjugate-transpose copy is formed to test it again.

    For Hermitian input, ``eigvalsh`` computes all eigenvalues of the
    binary-scaled matrix, and the endpoints are their minimum and maximum
    absolute values. All magnitudes are needed because the smallest
    singular value of an indefinite matrix is an interior eigenvalue
    magnitude. Squaring A would lose the small eigenvalues and square its
    condition number. Any other matrix uses a singular-values-only SVD.
    The caller, ``host_planning._spectrum``, admits the work and bytes.
    """

    scaled, exponent = binary_scaled_matrix(matrix)
    if hermitian:
        events["eigvalsh_calls"] += 1
        magnitudes = np.abs(np.linalg.eigvalsh(scaled))
        low, high, method = np.min(magnitudes), np.max(magnitudes), "eigvalsh"
    else:
        events["svd_calls"] += 1
        singular_values = np.linalg.svd(scaled, compute_uv=False)
        low, high, method = singular_values[-1], singular_values[0], "svd_validation_bounds"
    return ldexp(float(low), exponent), ldexp(float(high), exponent), method


def inverse_polynomial_action(left, values, rhs_direction, *, alpha, coefficients, right_h=None):
    """Return ``(y, ||y||**2)`` for the selected odd inverse polynomial from the original factors.

    Apply the selected inverse polynomial through the original spectral
    factors. For A = U Sigma V† and the dilation [[0,A],[A†,0]], an
    odd polynomial acting on (b,0) gives V P(Sigma) U†b in the lower block.
    Positive dummy coordinates extend the factors analytically and have
    zero RHS weight. Hermitian input uses signed eigenvalues. Shortcut
    polynomial action still needs spectral information for its distinct
    matrix G_t.

    Relation: with encoding scale alpha, the library's dilation is
    ``H = [[0, A/alpha], [A†/alpha, 0]]`` and
    ``P(H) = [[0, U P(Sigma/alpha) V†], [V P(Sigma/alpha) U†, 0]]``.
    Conjugating by ``diag(U, V)`` gives ``[[0, S], [S, 0]]``, whose even
    powers are diagonal and whose odd powers carry S's odd powers in the
    off-diagonal blocks. Linear combination proves the identity for odd
    Chebyshev sums. The input is ``(b_hat, 0)``, so the lower output block
    is ``y = V P(Sigma/alpha) U† b_hat``. Starting from ``(0, b_hat)`` would
    give ``U P(Sigma/alpha) V† b_hat`` in the upper block instead. The full
    dilation branch is ``(0, y)``, so the algorithm-branch mass and the
    physical-slice mass both equal ``||y||**2``, and the d-entry branch is
    returned directly. For a Hermitian original, one ``eigh`` supplies
    signed eigenvalues ``lambda`` and eigenvectors ``V``, and the action is
    ``V P(lambda/alpha) V† b_hat``. ``P(|lambda|)`` would lose the sign of
    negative eigenvalues. Degenerate values and arbitrary eigenvector
    phases leave these matrix functions unchanged. The basis changes
    conjugate vectors, never the d-by-d factors.

    Args:
        left: ``U`` of the original SVD, or the eigenvectors ``V`` of a
            Hermitian original, read-only.
        values: Singular values or signed eigenvalues, in the original scale.
        rhs_direction: Unit RHS direction ``b_hat`` on the d original
            coordinates.
        alpha: Selected encoding normalization.
        coefficients: Selected odd Chebyshev coefficients divided by the
            polynomial's rescale.
        right_h: ``V†`` of the original SVD, or None for a Hermitian
            eigensystem, whose right frame is ``left`` itself.
    """
    coordinates = np.conjugate(left.T @ np.conjugate(rhs_direction))
    transformed = np.polynomial.chebyshev.chebval(values / alpha, coefficients) * coordinates
    if right_h is None:
        branch = left @ transformed
    else:
        branch = np.conjugate(right_h.T @ np.conjugate(transformed))
    return branch, float(np.vdot(branch, branch).real)


def linear_model_weights(left, rhs_direction):
    """Return the left-singular weights ``w_j = |u_j† b_hat|**2`` of the linear norm model.

    Dalzell arXiv:2406.12086v2, Eq. (36):
    ``||x_bar_sigma||**2 = sum_j w_j / (f(sigma)**2 + [1 - f(sigma)**2] (sigma_j/alpha)**2)``.
    ``left`` is ``U`` of the original SVD, or the eigenvectors ``V`` of a
    Hermitian original, whose weights ``|V† b_hat|**2`` have the same
    magnitudes as those of the signed SVD frame ``U = V diag(sign lambda)``.
    The direction is normalized with ``stable_vector_norm``, and the frame is
    not conjugated as a matrix.
    """
    normalized = normalized_vector(rhs_direction, stable_vector_norm(rhs_direction))
    coordinates = np.conjugate(left.T @ np.conjugate(normalized))
    return np.abs(coordinates) ** 2


def projected_augmented_inplace(augmented, b_prime, tile=PROJECTOR_ROW_TILE):
    """Overwrite an owned augmented ``A_t`` with ``G_t = A_t - b'(b'† A_t)``.

    Form G_t by subtracting the rank-one matrix b′(b′†A_t) from A_t, using
    the normalized augmented RHS b′. This evaluates the same projector
    action with quadratic dense work and avoids constructing the complement
    projector.

    Relation: for the normalized
    ``b'``, ``G_t = (I - b'b'†) A_t = A_t - b'(b'† A_t)`` by distributivity,
    with no Hermiticity or invertibility requirement on ``A_t``. ``b'`` is
    normalized again, as ``projector_complement_matrix`` does, so the
    selected projector of the rounded input is preserved. The update is
    applied in row tiles of length ``r = min(a, tile)``, so the
    outer-product temporary holds ``16 r a`` bytes. Tiling changes storage,
    not the per-entry multiply and subtract. Rounding differs from the
    projector-matrix product, so the equivalence is algebraic, not bitwise.
    A zero or nonfinite projector vector rejects.
    """
    b = np.asarray(b_prime, dtype=np.complex128)
    norm = np.linalg.norm(b)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("the projector vector needs finite positive norm")
    b = b / norm
    row = np.conjugate(b) @ augmented
    for start in range(0, len(b), tile):
        stop = min(len(b), start + tile)
        correction = b[start:stop, None] * row[None, :]
        augmented[start:stop] -= correction
        del correction
    return augmented


def hermitian_dilation(matrix):
    """Return ``[[0, M], [M†, 0]]`` written into one preallocated array.

    One square output, zero fill and two block writes. ``np.conjugate``
    writes the conjugated block into the transposed lower-left view, so no
    separate conjugated block is formed. The assembly is bitwise equal to
    the ``np.block`` layout, including the zero quadrants.
    """
    d = len(matrix)
    result = np.zeros((2 * d, 2 * d), dtype=np.complex128)
    result[:d, d:] = matrix
    np.conjugate(matrix, out=result[d:, :d].T)
    return result


def shortcut_matrix(matrix, rhs_direction, *, t_value):
    """Return Dalzell's ``G_t = Q_b' A_t`` for the encoded matrix (Eqs. (8), (9), (11)).

    ``A_t`` doubles the dimension and places ``1/t`` at the first new
    coordinate ``e_n`` (the one-qubit padding of Dalzell arXiv:2406.12086v2,
    App. A.6), and ``b' = (b + e_n)/sqrt(2)``. ``projected_augmented_inplace``
    applies the projector to the freshly built ``A_t``, which this function
    owns, so neither a copy nor the complement projector matrix is formed.
    No circuit is built and no target solve is performed.
    """
    augmented = dalzell_augmented_matrix(matrix, t_value)
    b_prime = np.zeros(augmented.shape[0], dtype=complex)
    b_prime[: len(rhs_direction)] = rhs_direction
    b_prime[len(rhs_direction)] = 1.0
    b_prime /= np.sqrt(2.0)
    return projected_augmented_inplace(augmented, b_prime)


def shortcut_polynomial_action(g_t, coefficients, *, system_dimension, method):
    """Evaluate steps 2-3 of Dalzell Algorithm 1 (arXiv:2406.12086v2, p. 5).

    Applies the even kernel-reflection polynomial to the right singular
    vectors of ``G_t``, ``V K(Sigma) V^dagger e_n`` (Dalzell App. B.1,
    Eq. (50)), starting from ``e_n``. ``shortcut_native_svp`` uses the SVD
    of ``G_t``. ``shortcut_dilation`` applies the even polynomial to the
    Hermitian dilation ``[[0, G_t], [G_t^dagger, 0]]``, written into one
    array by ``hermitian_dilation``, and reads its lower block, which gives
    the same vector. With ``G_t = W Sigma V^dagger`` and
    ``K(x) = q(x^2)``, ``K(H) = q(H^2) = diag(W K(Sigma) W^dagger,
    V K(Sigma) V^dagger)``, whose lower block is the ``V K(Sigma) V^dagger``
    of Eq. (50). The branch is then projected onto the original coordinates.
    """
    if method == "shortcut_native_svp":
        singular_values, right_h = np.linalg.svd(g_t)[1:]
        values = np.polynomial.chebyshev.chebval(singular_values, coefficients)
        start = np.zeros(g_t.shape[1], dtype=complex)
        start[system_dimension] = 1.0
        transformed = values * (right_h @ start)
        # right_h is V^dagger, so right_h.T = conj(V) and
        # conj(conj(V) conj(w)) = V w. Conjugating the vectors instead of
        # right_h avoids a conjugated copy of the G_t-sized matrix, which the
        # workspace law in host_planning.selected_work does not count.
        branch = (right_h.T @ transformed.conj()).conj()
    else:
        hermitian = hermitian_dilation(g_t)
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        del hermitian
        values = np.polynomial.chebyshev.chebval(eigenvalues, coefficients)
        start = np.zeros(eigenvectors.shape[0], dtype=complex)
        start[g_t.shape[0] + system_dimension] = 1.0
        # V^dagger e = conj(V^T conj(e)): the basis change conjugates the
        # vectors instead of forming a conjugated copy of the eigenvectors.
        coordinates = np.conjugate(eigenvectors.T @ np.conjugate(start))
        branch = eigenvectors @ (values * coordinates)
    return project_shortcut_branch(
        branch,
        system_dimension=system_dimension,
        dilation_offset=g_t.shape[0] if method == "shortcut_dilation" else 0,
    )


def project_shortcut_branch(branch, *, system_dimension, dilation_offset=0):
    """Project a shortcut branch onto the original coordinates (Algorithm 1 step 3).

    Keeps the ``system_dimension`` entries after ``dilation_offset`` and
    returns them with their squared norm, the physical-slice mass.
    """
    projected = np.array(
        branch[dilation_offset : dilation_offset + system_dimension], dtype=complex
    )
    return projected, float(np.sum(np.abs(projected) ** 2))


def _rhs_direction(rhs):
    """Return the unit RHS direction ``b/||b||`` as a dense vector.

    A stored vector is used as is. Occupation and product preparations are
    expanded once for the selected classical work.
    """
    if rhs.reference.representation == "vector":
        return rhs._direction
    if rhs.preparation.implementation == "qiskit.occupation":
        direction = np.zeros(rhs.manifest.basis.dimension, dtype=complex)
        direction[sum(bit << index for index, bit in enumerate(rhs._physical))] = 1.0
        return direction
    if rhs.preparation.implementation == "qiskit.product":
        direction = np.ones(1, dtype=complex)
        for pair in rhs._direction:
            direction = np.multiply.outer(pair, direction).reshape(-1)
        return direction
    raise ValueError("reference solve requires live nonzero RHS direction access")
