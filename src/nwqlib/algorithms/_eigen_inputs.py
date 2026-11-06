"""Original-coordinate inputs shared by Hermitian eigenvalue methods."""

from math import frexp, fsum, isfinite

import numpy as np

from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _distinct_pauli_input
from nwqlib.operators._pauli import pauli_coefficient_masks
from nwqlib.problems.inputs import ingest_vector, ingest_product, state_input
from nwqlib.problems.records import Eigenproblem, Eigenvalue


# Largest dense dimension converted to Pauli terms without an explicit
# conversion request. ENGINEERING_CONSTANTS.md, "Eigen input conversion and
# classical preparation", lists the methods that share it.
AUTO_DENSE_PAULI_DIMENSION = 16

# Qiskit standard gates whose statevector action is a fixed dense k-qubit
# matrix, so preparation_requirements can count d*2**k products per gate. A
# barrier is admitted and counts no work. An instruction outside this set, or
# from another module, leaves the preparation work unknown.
_STANDARD_GATE_NAMES = frozenset((
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "p", "u", "u1", "u2", "u3",
    "cx", "cy", "cz", "ch", "cp", "crx", "cry", "crz",
    "swap", "iswap", "rxx", "ryy", "rzz", "rzx", "ccx", "cswap", "barrier",
))


def conversion_requirements(operator, *, input_conversion, max_bytes):
    """Check one dense-to-Pauli conversion against max_bytes and return (q, D) before allocation.

    q=max(1, ceil(log2 d)) qubits embed the original dimension d in D=2**q.
    The bytes checked here are those of the block transform of
    operators._pauli.pauli_coefficient_masks, 64*D*D: the padded complex128
    matrix, the two active transform levels and the componentwise
    temporaries of one level. The number R of nonzero words is known only
    after the transform, so pauli_coefficient_masks checks the packed masks,
    and a caller that forms labels checks them with its label decoding.
    Raises ApplicabilityError for input other than dense, CSR or CSC and
    when automatic conversion does not apply. Raises ValueError for an
    unknown input_conversion and for bytes above max_bytes.
    """
    d = operator.basis.dimension
    representation = operator.reference.representation
    if representation not in ("dense", "csr", "csc"):
        raise ApplicabilityError("requires finite Pauli or explicit dense/sparse input")
    if input_conversion not in ("auto", "dense_pauli"):
        raise ValueError("input_conversion must be auto or dense_pauli")
    if input_conversion == "auto" and (representation != "dense" or d > AUTO_DENSE_PAULI_DIMENSION):
        raise ApplicabilityError(
            f"automatic Pauli conversion requires explicit dense dimension <={AUTO_DENSE_PAULI_DIMENSION}; "
            "choose input_conversion='dense_pauli', "
            "or classical execution for original matvec access"
        )
    width = max(1, (d - 1).bit_length())
    dimension = 1 << width
    # The padded matrix, the two active transform levels and the
    # componentwise temporaries: four complex128 D-by-D arrays.
    _check_bytes(64 * dimension * dimension, max_bytes, "eigenvalue Pauli conversion")
    return width, dimension


def _mean_diagonal(operator, *, max_bytes):
    """Return ``c = trace(A)/d``, the value of mean padding and the identity component a classical projection removes.

    For Pauli input this is the identity coefficient, exactly the identity
    component of the Pauli expansion. For dense and CSR/CSC input it is the
    mean of the diagonal. Other representations give zero and keep their
    action. Any c gives the same Ritz values in exact arithmetic, since the
    solve adds it back. The mean diagonal is the mean eigenvalue, so it lies in
    [lambda_min, lambda_max] and ``||A - cI||_2 <= lambda_max - lambda_min``,
    a bound that an identity offset of A does not change.

    An exactly constant diagonal is its own mean. The rounded sum of d
    quotients can miss it by a unit in the last place (c = .9, d = 3), and then
    the padded or shifted form of H = cI has a nonzero nonidentity part made of
    rounding. Each a_ii is otherwise divided by d before the fsum, because
    fsum(diag/d) and fsum(diag)/d do not generally round identically and the
    latter can overflow before division. fsum of one entry returns it exactly,
    apart from -0.0, which becomes the 0.0 that the sum of quotients gives.
    """
    representation = operator.reference.representation
    d = operator.basis.dimension
    if representation == "pauli":
        identity = "I" * (d.bit_length() - 1)
        return fsum(c.real for label, c in operator.pauli_terms().labels(max_bytes=max_bytes)
                    if label == identity)
    if representation in ("dense", "csr", "csc"):
        diagonal = operator._data.diagonal().real
        return fsum(diagonal[:1] if diagonal.min() == diagonal.max() else diagonal / d)
    return 0.0


def dense_pauli_matrix(operator, *, input_conversion, max_bytes, padding="zero"):
    """Return A embedded in a complex128 D-by-D matrix, D=2**max(1, ceil(log2 d)).

    The conversion is checked against the limits before the matrix is allocated. Dense input is
    copied and CSR/CSC input is scattered row by row or column by column.
    padding="zero" leaves the dummy diagonal at zero. padding="mean" writes
    trace(A)/d there, the choice eigen_operator explains.
    """
    _, dimension = conversion_requirements(operator, input_conversion=input_conversion, max_bytes=max_bytes)
    d = operator.basis.dimension
    representation = operator.reference.representation
    matrix = np.zeros((dimension, dimension), dtype=complex)
    if representation == "dense":
        matrix[:d, :d] = operator.dense_array()
    else:
        data, indices, indptr = operator.sparse_entries(max_bytes=max_bytes)
        for i in range(d):
            if representation == "csr":
                matrix[i, indices[indptr[i] : indptr[i + 1]]] = data[indptr[i] : indptr[i + 1]]
            else:
                matrix[indices[indptr[i] : indptr[i + 1]], i] = data[indptr[i] : indptr[i + 1]]
    if dimension != d and padding == "mean":
        matrix[np.arange(d, dimension), np.arange(d, dimension)] = _mean_diagonal(operator, max_bytes=max_bytes)
    return matrix


def eigen_operator(
    problem,
    output,
    *,
    max_bytes=None,
    as_pauli=True,
    input_conversion="auto",
    pad_classical=False,
):
    """Choose Pauli access.

    Explicit dense padding adds trace(A)/d on dummy states. The input object
    already checked exact Hermiticity. This finite transform has
    O(q D²) numerical traffic and O(D²) known array storage. It
    never diagonalizes A. Structured Pauli access stays compact.

    Padding applies on the dense-to-Pauli conversion and, for classical
    access, only when ``pad_classical`` is set. Otherwise a classical
    operator is returned unpadded. A dimension d that is not a power of two
    is then embedded in D = 2**q, and each padding state gets the value
    trace(A)/d, the mean eigenvalue of A, which lies in
    [lambda_min, lambda_max]. The padded operator therefore has the same
    lambda_min and lambda_max, and the original coordinates keep their
    action. The padding states join the ground eigenspace only when
    trace(A)/d equals lambda_min, that is, when A is a multiple of the
    identity.
    """
    if type(problem) is not Eigenproblem or type(output) is not Eigenvalue:
        raise ApplicabilityError("this method requires Eigenproblem and Eigenvalue")
    operator = problem.A
    if not as_pauli:
        if operator.reference.representation not in ("pauli", "dense", "csr", "csc"):
            raise ApplicabilityError("classical execution of this Method needs dense, CSR, CSC or Pauli input, "
                                     f"and this input is {operator.reference.representation}")
        d = problem.dimension
        dimension = 1 << max(1, (d - 1).bit_length())
        if pad_classical and dimension != d:
            from nwqlib.operators.inputs import ingest_dense, ingest_sparse

            if operator.reference.representation in ("csr", "csc"):
                from scipy.sparse import block_diag, diags

                # 4*payload_bytes allow four copies of the stored sparse arrays
                # during block_diag assembly and ingestion. 64*D covers the
                # padding diagonal and the enlarged pointer arrays.
                _check_bytes(
                    4 * operator.manifest.payload_bytes + 64 * dimension,
                    max_bytes,
                    "classical ADAPT sparse embedding",
                )
                padded = block_diag(
                    (operator._data, diags(np.full(dimension - d, _mean_diagonal(operator, max_bytes=max_bytes)))),
                    format=operator.reference.representation,
                )
                return ingest_sparse(padded, max_bytes=max_bytes)
            # Four complex128 D-by-D arrays: the padded matrix, the dense copy
            # of A, and ingest_dense's snapshot and conjugate for its Hermitian check.
            _check_bytes(64 * dimension * dimension, max_bytes, "classical ADAPT embedding")
            matrix = np.zeros((dimension, dimension), dtype=complex)
            matrix[:d, :d] = operator.dense_array()
            matrix[np.arange(d, dimension), np.arange(d, dimension)] = _mean_diagonal(operator, max_bytes=max_bytes)
            return ingest_dense(matrix, max_bytes=max_bytes)
        return operator
    if "pauli_terms" in operator.manifest.access:
        return operator
    d = problem.dimension
    width = max(1, (d - 1).bit_length())
    matrix = dense_pauli_matrix(
        operator,
        input_conversion=input_conversion,
        max_bytes=max_bytes,
        padding="mean",
    )
    # The words go to the operator as packed masks, never as one Python
    # label and coefficient per word. The transform's words are distinct
    # and nonzero, so they need no coalescing.
    x, z, coefficients = pauli_coefficient_masks(matrix, max_bytes=max_bytes)
    del matrix
    if np.any(coefficients.imag != 0):
        raise ValueError("Hermitian Pauli transform produced nonreal coefficients")
    # The masks and the real coefficients, then the operator's copy of both.
    # The records that later hold kept terms check their own bytes.
    _check_bytes(len(x) * 2 * (16 * x.shape[1] + 16), max_bytes, "eigenvalue Pauli operator")
    values = coefficients.real.astype(np.complex128)
    del coefficients
    return _distinct_pauli_input(width, x, z, values)


def eigen_state(value, problem, *, rng=None, max_bytes=None, pad=True):
    """Resolve initialization in original coordinates, then zero-pad if necessary.

    Without a supplied state, the method RNG draws a random product state
    (power-of-two d) or a random complex Gaussian vector. Either has nonzero
    overlap with any fixed eigenvector with probability one. The draw comes
    from the method stream, so it is reproduced by the root seed recorded with
    the Plan. An explicit symmetry sector
    requires a supplied state, since a random draw would leave the sector.
    With pad set and d below the padded dimension, the Gaussian draw is made
    by ``draw_padded_state``.
    """
    d = problem.dimension
    target = 1 << max(1, (d - 1).bit_length())
    if value is None:
        if problem.sector is not None:
            raise ApplicabilityError(
                "an explicit sector requires a supplied preparation in that sector"
            )
        if rng is None:
            raise ValueError("default initialization requires the selected method RNG")
        if d == target:
            factors = []
            for _ in range((d - 1).bit_length()):
                p, phase = rng.method.uniform(), rng.method.uniform(0, 2 * np.pi)
                factors.append((np.sqrt(1 - p), np.exp(1j * phase) * np.sqrt(p)))
            state = ingest_product(factors, max_bytes=max_bytes)
        elif pad:
            # The draw has the original dimension d by construction. The
            # returned state already has the padded dimension.
            state = draw_padded_state(d, target, rng.method, max_bytes)
        else:
            # With d = problem.dimension, 64d covers four complex128 arrays during
            # ingestion: input, physical snapshot, direction and direction snapshot.
            _check_bytes(64 * d, max_bytes, "eigenvalue initial vector")
            vector = rng.method.normal(size=d) + 1j * rng.method.normal(size=d)
            vector /= np.linalg.norm(vector)
            state = ingest_vector(vector, max_bytes=max_bytes)
    else:
        state = state_input(value, max_bytes=max_bytes)
    padded = value is None and pad and d != target
    if state.basis.dimension != (target if padded else d):
        raise ValueError("method preparation must use the original Eigenproblem dimension")
    if state.preparation.physical_scale is None:
        raise ApplicabilityError(state.preparation.blocker or "preparation scale is unresolved")
    if state.preparation.physical_scale.mantissa == 0:
        raise ValueError("eigenvalue initialization must be nonzero")
    if pad and d != target and not padded:
        # With T = target, ingestion holds four length-T complex128 buffers
        # while the supplied state's physical and direction buffers remain live.
        # Admit 64*T + state.manifest.payload_bytes + 65536 bytes. The state
        # payload is at most 32*d bytes, and 65536 is the fixed bookkeeping allowance.
        _check_bytes(
            64 * target + state.manifest.payload_bytes + 65536,
            max_bytes,
            "eigenvalue padded preparation",
        )
        vector = np.zeros(target, dtype=complex)
        vector[:d] = state.physical_vector()
        state = ingest_vector(vector, max_bytes=max_bytes)
    if pad and state.preparation.blocker:
        raise ApplicabilityError(state.preparation.blocker)
    return state


def draw_padded_state(d, target, rng, max_bytes):
    """Draw a Gaussian initial state directly into its zero-padded vector.

    Draw the original-dimensional Gaussian state into the nonzero prefix of
    one padded complex128 vector, normalize that prefix with the established
    reduction, and count four simultaneous padded complex arrays against the
    byte limit for input ingestion.

    Byte accounting: float64 entries take eight bytes and complex128 entries
    sixteen. With T = target, drawing holds the 16T-byte padded vector and
    two 8d-byte normal arrays, at most 32T. The contiguous [:d] prefix is
    normalized, preserving the reduction length and arithmetic of a
    length-d draw, and the draws are released. ingest_vector can then hold
    the padded vector, its immutable physical copy, the mutable normalized
    direction and its immutable copy at once, four arrays of 16T bytes, 64T
    numerical bytes. The digest reads a memoryview and
    creates no full hash buffer. H0 = 65536 bytes is the engineering
    allowance for scalar bookkeeping, ndarray headers, iterators and fixed
    sort stacks on the checked 64-bit CPython/NumPy stack. It is not a
    universal interpreter or process-RSS bound. The counted size is therefore
    64*T + 65536 bytes. The multiply and add below reproduce real+1j*imag,
    including signed-zero operations, without a complex draw buffer.
    Replacing the prefix norm by norm(vector) or another norm routine can
    change the resulting bits. This branch covers 0 < d < target, including
    d = 1 with target 2. A zero or nonfinite draw still fails normalization
    or the input check.

    Args:
        d: Original dimension.
        target: Padded power-of-two dimension, above d.
        rng: NumPy Generator of the method stream.
        max_bytes: Byte limit.

    Returns:
        The ingested padded StateInput.
    """
    if not 0 < d < target:
        raise ValueError("this branch requires a positive padded dimension")
    _check_bytes(64 * target + 65536, max_bytes, "eigenvalue padded random preparation")
    vector = np.zeros(target, dtype=np.complex128)
    real = rng.normal(size=d)
    imag = rng.normal(size=d)
    np.multiply(1j, imag, out=vector[:d])
    np.add(real, vector[:d], out=vector[:d])
    del real, imag
    vector[:d] /= np.linalg.norm(vector[:d])
    return ingest_vector(vector, max_bytes=max_bytes)


def preparation_requirements(state):
    """Return (bytes, work) of computing a state's length-d vector classically.

    A k-qubit dense gate applies 2**k products per full-state amplitude.
    Only standard gates with known fixed action use this bound. Custom,
    opaque or composite instruction simulation work remains unknown (None).
    The classical call is synchronous. This is not a bound on Qiskit run
    time or RSS.

    Returns:
        (bytes, work). A product state needs 32*d bytes, the final and
        previous vectors of the Kronecker expansion, and 2+4+...+d=2d-2
        products. A vector or occupation state needs 32*d bytes and no
        products. A Qiskit circuit of standard gates needs 48*d bytes, three
        complex128 vectors for the statevector and its update buffers, plus
        one dense 2**k-by-2**k complex128 matrix of the widest gate, and
        d*2**k products per k-qubit gate (d more for a global phase). A
        circuit with any other instruction returns (48*d, None), and any other
        representation (32*d, None).
    """
    d = state.basis.dimension
    representation = state.reference.representation
    if representation == "product":
        return 32 * d, 2 * d - 2
    if representation in ("vector", "occupation"):
        return 32 * d, 0
    if state._native is not None:
        work = d if state._native.global_phase != 0 else 0
        gate_bytes = 0
        for instruction in state._native.data:
            operation = instruction.operation
            base = getattr(operation, "base_class", type(operation))
            module = base.__module__ or ""
            if operation.name not in _STANDARD_GATE_NAMES or not (
                module.startswith("qiskit.circuit.library.standard_gates.")
                or module == "qiskit.circuit.barrier"
            ):
                return 48 * d, None
            if operation.name != "barrier":
                work += d * (1 << operation.num_qubits)
                gate_bytes = max(gate_bytes, 16 * (1 << (2 * operation.num_qubits)))
        return 48 * d + gate_bytes, work
    return 32 * d, None


def classical_constant_matrix(a, *, compressed):
    """Return c when a finite canonical Hermitian matrix is exactly c*I, else None.

    a is dense, or CSR/CSC when compressed, with dimension at least one. The
    test reads the original values before any scaling, because a very small
    off-diagonal value can underflow to zero during scaling.
    """
    diagonal = a.diagonal()
    # With distinct sparse coordinates, the excess counts exactly the
    # nonzero off-diagonal entries, including implicit-zero diagonals.
    values = a.data if compressed else a
    if np.count_nonzero(values) != np.count_nonzero(diagonal):
        return None
    return float(diagonal[0].real) if np.all(diagonal == diagonal[0]) else None


# Below this largest row extent max_i(|a_ii| + r_i), the absolute term
# tiny = 2**-1022 of the Gershgorin margin exceeds 2**-53 times the extent,
# so gershgorin_frame scales the entries up by a power of two first.
_SMALL_FRAME_EXTENT = 2.0**-969


def _gershgorin_rows(a, d, compressed, exponent):
    """Return (radius, center, off_count) of a dense, CSR or CSC Hermitian matrix with entries scaled by 2**exponent.

    radius holds the off-diagonal magnitude sums of the rows (the columns
    for CSC), center the real diagonal and off_count the stored
    off-diagonal entries per row as int64. ``exponent`` is 0 or positive.
    Multiplying a binary64 component by a positive power of two is exact as
    long as the result does not overflow, so the scaled entries carry the
    stored bits of the matrix.
    """
    values = a.data if compressed else a
    if exponent:
        if np.iscomplexobj(values):
            magnitude = np.ldexp(values.real, exponent)
            np.hypot(magnitude, np.ldexp(values.imag, exponent), out=magnitude)
        else:
            magnitude = np.abs(np.ldexp(values, exponent))
        center = np.ldexp(a.diagonal().real, exponent)
    else:
        magnitude = np.abs(values)
        center = a.diagonal().real
    if compressed:
        counts = np.diff(a.indptr).astype(np.int64)
        major = np.repeat(np.arange(d, dtype=np.intp), counts)
        is_diagonal = a.indices == major
        diag_positions = np.flatnonzero(is_diagonal)
        off_count = counts.copy()
        off_count[major[diag_positions]] -= 1
        magnitude[diag_positions] = 0.0
        del major, is_diagonal, diag_positions
        nonempty = np.flatnonzero(counts)
        radius = np.zeros(d)
        if nonempty.size:
            radius[nonempty] = np.add.reduceat(magnitude, a.indptr[nonempty])
    else:
        np.fill_diagonal(magnitude, 0.0)
        radius = magnitude.sum(axis=1)
        off_count = np.full(d, max(0, d - 1), dtype=np.int64)
    return radius, center, off_count


def gershgorin_frame(operator, *, max_bytes):
    """Return a Hermitian Gershgorin frame without an eigensolve.

    Magnitudes |a_ij| are computed directly, diagonal magnitudes are zeroed
    before reduction, and empty compressed rows or columns have zero radius.
    The outward allowance uses ``(8*n+16)*eps``, where n counts stored
    off-diagonal entries and eps is 2**-52. It covers sequential or pairwise
    positive summation under round-to-nearest binary64 arithmetic, gradual
    underflow and a one-ulp hypot assumption. This is a conditional
    numerical enclosure, not a directed-rounding certificate. Scalar
    operators return their exact scalar value and zero half-width.

    Assumptions. The stored matrix is Hermitian with a real diagonal, finite
    entries and canonical CSR/CSC coordinates when compressed. IEEE binary64
    with gradual underflow, round-to-nearest, and hypot relative error at
    most 2u on normal results. By the Gershgorin circle theorem every
    eigenvalue of a Hermitian A lies in the union of the real intervals
    ``[a_ii - r_i, a_ii + r_i]`` with the exact row radius
    ``r_i = sum_{j != i} |a_ij|``. For a Hermitian matrix the column radii
    are equal, so CSC reduces columns without conversion.

    Steps. With n_i stored off-diagonal entries in row i, including explicit
    zeros, the diagonal positions of the magnitude workspace are zeroed
    before the reduction, so every nonzero magnitude has at most n_i
    effective addition roundings under either sequential or pairwise
    summation, and no diagonal magnitude is subtracted from a row sum. With

        q_i = (8 n_i + 16) eps,  eps = 2**-52,  q_i < 1,
        M_i = q_i / (1 - q_i) (|c_i| + r_i) + tiny,  tiny = 2**-1022,

    c_i the diagonal entry and r_i the computed radius, row i widens to
    [c_i - r_i - M_i, c_i + r_i + M_i], rounded outward. tiny covers the
    absolute rounding of results below the normal range. Row counts are
    int64 before 8*n_i+16 is evaluated, since int32 arithmetic can wrap.
    For CSR/CSC only nonempty segments are reduced and every empty segment's
    sum is zero, because duplicate ``indptr`` starts do not denote empty
    sums to ``reduceat`` and a start equal to ``len(data)`` raises. Sparse
    input is never densified. The final alpha covers both outward-rounded
    endpoints around the computed center, which protects the affine frame
    after midpoint rounding. It can enlarge alpha beyond the half-width
    (high - low)/2 by a last bit.

    Small matrices. When the largest row extent max_i(|c_i| + r_i) is below
    2**-969, tiny exceeds 2**-53 times that extent and would widen the
    frame beyond the matrix's own scale, which shrinks the normalized
    spectrum and can change a later rank decision. The entries are then
    multiplied by the power of two 2**k that brings the extent into
    [1/2, 1), which is exact because no scaled value overflows, and the rows
    and margin are formed in those coordinates, where tiny is negligible.
    The endpoints are multiplied back by 2**-k. That product rounds when it
    falls below the normal range, so each endpoint is rounded outward once
    more. Every other matrix takes the direct path alone.

    A frame whose endpoints, center or half-width are not finite is refused.
    The padded endpoint of row i has magnitude |c_i| + r_i + M_i, and M_i
    exceeds the rounding error of the computed radius, so an overflowing
    radius or margin means that the padded endpoint exceeds the binary64
    maximum in exact arithmetic. Magnitudes formed from scaled components
    would then give no finite frame either. The exact Gershgorin endpoint,
    or the spectrum itself, can still be representable, and such a matrix
    is refused. For example the eigenvalues of [[1e308, 1e308], [1e308,
    -1e308]] are about +/-1.414e308, and its Gershgorin endpoints are
    +/-2e308.

    Workspace: O(nnz+D) for compressed and O(D²) for dense storage. The
    magnitudes are one float array per stored value, two while a scaled
    complex input forms them, and the sparse major-coordinate and mask
    arrays are released before the row endpoints are assembled. Gershgorin
    frame construction counts 32*N+256*D additional array-data bytes and a
    65,536-byte fixed bookkeeping allowance. Here N includes every stored
    entry for CSR/CSC, including explicit zeros, and equals D squared for
    dense storage. The input matrix is excluded. The array formula covers
    the simultaneous magnitude, coordinate, reduction and endpoint arrays.
    The fixed allowance is qualified for the declared software stack and is
    separate from process RSS and external library workspace. Derivation.
    The live array schedule has the peaks 16N for the scaled components,
    17N+40D for compressed diagonal/index handling, 8N+56D for segmented
    reduction and 8N+80D for final endpoints, each at most 32N+256D
    (ENGINEERING_CONSTANTS.md, "Eigen input conversion and classical
    preparation").

    Returns:
        ``(low, high, center, alpha)`` with low <= every eigenvalue <= high
        under the assumptions, ``center`` the midpoint and ``alpha`` the
        half-width, all in the operator's units. A scalar operator returns
        alpha = 0.
    """
    d = operator.basis.dimension
    compressed = operator.reference.representation in ("csr", "csc")
    a = operator._data
    values = a.data if compressed else a
    _check_bytes(32*values.size + 256*d + 65536, max_bytes, "Gershgorin frame workspace")
    scalar = classical_constant_matrix(a, compressed=compressed)
    if scalar is not None:
        return scalar, scalar, scalar, 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        radius, center, off_count = _gershgorin_rows(a, d, compressed, 0)
        extent = float(np.max(np.abs(center) + radius))
        exponent = -frexp(extent)[1] if 0 < extent < _SMALL_FRAME_EXTENT else 0
        if exponent:
            radius, center, off_count = _gershgorin_rows(a, d, compressed, exponent)
        k_eps = (8 * off_count + 16) * np.finfo(float).eps
        if np.any(k_eps >= 1):
            raise ValueError("Gershgorin roundoff allowance is unresolved at this dimension")
        margin = k_eps / (1 - k_eps) * (np.abs(center) + radius) + np.finfo(float).tiny
        low = float(np.min(np.nextafter(center - radius - margin, -np.inf)))
        high = float(np.max(np.nextafter(center + radius + margin, np.inf)))
        if exponent:
            low = float(np.nextafter(np.ldexp(low, -exponent), -np.inf))
            high = float(np.nextafter(np.ldexp(high, -exponent), np.inf))
        midpoint = low / 2 + high / 2
        alpha = float(np.nextafter(max(midpoint - low, high - midpoint), np.inf))
    if not all(isfinite(v) for v in (low, high, midpoint, alpha)) or alpha <= 0:
        raise ValueError("finite Gershgorin affine frame is outside binary64 range")
    return low, high, midpoint, alpha


def state_direction(state, *, max_bytes=None):
    """Return the normalized complex length-d vector of a state for classical execution.

    Only explicit classical execution calls this. A product state is expanded
    by Kronecker products, an occupation state becomes a basis vector, and a
    Qiskit circuit is simulated by Statevector.from_instruction.
    """
    d = state.basis.dimension
    # The complex128 result and the previous Kronecker factor product.
    _check_bytes(32 * d, max_bytes, "classical eigenvalue preparation")
    state._require_data()
    if state.reference.representation == "vector":
        return state._direction
    if state.reference.representation == "product":
        vector = np.ones(1, dtype=complex)
        for factor in state._direction:
            vector = np.kron(factor, vector)
        return vector
    if state.reference.representation == "occupation":
        vector = np.zeros(d, dtype=complex)
        index = sum(int(bit) << i for i, bit in enumerate(state._physical))
        vector[index] = 1
        return vector
    if state._native is not None:
        from qiskit.quantum_info import Statevector

        return Statevector.from_instruction(state._native).data
    raise ApplicabilityError("classical eigenvalue acquisition needs explicit state access")
