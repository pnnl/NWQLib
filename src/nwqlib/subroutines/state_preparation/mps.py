"""NumPy TT-SVD of a state vector into MPS cores, and scalar analysis of those cores.

The decomposition follows Algorithm 1 (TT-SVD, p. 2301) of Oseledets,
"Tensor-train decomposition", SIAM J. Sci. Comput. 33(5), 2295-2317 (2011),
doi:10.1137/090752286, with the per-value truncation rule stated in
``decompose_state_to_mps``. The truncation estimates of
``analyze_mps_state_compression`` rest on the orthogonality step in the
proof of Theorem 2.2 (p. 2299). The work and byte limits are checked for
the whole sweep before the first SVD. ``layered_construction_size`` bounds
the work and bytes of the layered circuit that ``mps_circuit`` builds from
these cores.
"""

from dataclasses import dataclass
from math import fsum, sqrt

import numpy as np

from nwqlib._numerics import (
    normalize_state_vector,
    normalized_fidelity_with_window,
    stable_vector_norm,
)
from nwqlib._validation import integer
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes, _check_products
from nwqlib.operators.inputs import _array_input, _digest, _freeze_array

# Dense economy SVD is performed before truncation. This logical work ceiling
# bounds sum(rows*columns*min(rows,columns)); it is not FLOPs, RSS or CPU time.
# Raise it explicitly only when the selected dense decomposition is intended.
# Registered in docs/ENGINEERING_CONSTANTS.md.
DEFAULT_MAX_SVD_WORK = 100_000_000


def _input(vector, max_bytes):
    """Admit a one-dimensional vector and return ``(normalized_state, input_norm)``.

    Admission reserves 48 bytes per entry beyond the stored input, three
    complex128 state-sized arrays for the conversion and normalization.
    """
    array = _array_input(vector, ndim=1, max_bytes=max_bytes, extra_bytes_per_item=48)
    return normalize_state_vector(array)


@dataclass(frozen=True, kw_only=True)
class MPSDecomposition:
    """The tensor-train (MPS) cores of one normalized state vector, most significant qubit first.

    [`decompose_state_to_mps`][nwqlib.subroutines.state_preparation.mps.decompose_state_to_mps]
    returns it. `discarded_weight` is the sum of squared singular values
    discarded in the sweep. Floating-point truncation metrics are estimates,
    not bounds on SVD roundoff or fidelity of a synthesized layered circuit.
    Construction checks that there are `max(1, n)` cores of shape
    `(r_i, 2, 1, r_{i+1})` with boundary ranks one, finite and read-only,
    that `discarded_weight` lies in `[0, 1]` and that `input_norm` is
    positive. The fields below are read-only.

    Attributes:
        cores: Immutable MSB-first TT cores with shape (left rank, physical size, 1, right rank).
        num_qubits: Number of binary physical sites; zero represents a scalar input.
        original_dimension: Input length, exactly 2**num_qubits.
        bond_dimensions: Boundary and internal TT ranks; the first and last equal one.
        max_bond_dim: Requested retained-rank cap, or None for no explicit rank cap.
        threshold: Individual singular-value truncation threshold, not a total error tolerance.
        discarded_weight: Sum of discarded squared singular values for the normalized TT-SVD input.
        input_id: Content hash of the normalized input that was compressed.
        input_norm: Positive norm removed from the original input before TT-SVD.
        provenance: How the cores were computed and ordered, with the source of the algorithm.
    """

    cores: tuple[np.ndarray, ...]
    num_qubits: int
    original_dimension: int
    bond_dimensions: tuple[int, ...]
    max_bond_dim: int | None
    threshold: float
    discarded_weight: float
    input_id: str
    input_norm: float = 1.0
    provenance: str = "Oseledets (2011), doi:10.1137/090752286, Algorithm 1 and Theorem 2.2 proof"

    def __post_init__(self):
        """Reject a record whose cores cannot be the TT-SVD of a 2**n vector.

        There must be ``max(1, n)`` cores and one more bond dimension than
        cores, with ones at both ends. Core ``i`` must have shape
        ``(r_i, 2, 1, r_{i+1})``, or ``(1, 1, 1, 1)`` for the scalar input,
        and be finite and read-only. The discarded weight is a sum of squared
        singular values of a unit-norm input, so it lies in ``[0, 1]``, and
        the removed input norm is positive.
        """
        integer(self.num_qubits, "num_qubits", 0)
        if self.original_dimension != 1 << self.num_qubits:
            raise ValueError("MPS dimension must match its qubit count")
        if len(self.cores) != max(1, self.num_qubits) or len(self.bond_dimensions) != len(self.cores) + 1:
            raise ValueError("MPS cores and bond dimensions must agree")
        if self.bond_dimensions[0] != 1 or self.bond_dimensions[-1] != 1:
            raise ValueError("MPS boundary ranks must equal one")
        for i, core in enumerate(self.cores):
            shape = (self.bond_dimensions[i], 2 if self.num_qubits else 1, 1, self.bond_dimensions[i+1])
            if core.shape != shape or core.flags.writeable or not np.all(np.isfinite(core)):
                raise ValueError("MPS cores require matching finite immutable arrays")
        if not np.isfinite(self.discarded_weight) or not 0 <= self.discarded_weight <= 1:
            raise ValueError("normalized TT-SVD discarded weight must lie in [0,1]")
        if not np.isfinite(self.input_norm) or self.input_norm <= 0:
            raise ValueError("MPS input norm must be finite and positive")

    @property
    def core_bytes(self):
        """Total bytes of the stored core arrays."""
        return sum(core.nbytes for core in self.cores)

    # max_products=1_000_000_000 here and in the contraction functions below is
    # an untuned ceiling on output entries times contracted rank, registered
    # in docs/ENGINEERING_CONSTANTS.md. Revisit for a deliberately larger
    # requested materialization.
    def to_statevector(self, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000):
        """Contract the stored cores into the raw compressed vector, without renormalizing it or computing an SVD.

        The vector approximates the normalized input. It is not rescaled by
        `input_norm`.

        Args:
            max_bytes (int): Default 10 GB (decimal, `10_000_000_000`
                bytes). Limit on the stored cores plus the known arrays of
                each contraction step and of the output vector, checked
                before each step. It does not measure process memory.
            max_products (int): Default `1_000_000_000`. Limit on the
                scalar multiplications of the whole contraction, the sum
                over steps of the step's output entries times the
                contracted bond rank, checked before each step.

        Returns:
            vector (numpy.ndarray): The `original_dimension = 2**num_qubits`
                amplitudes, in the order of the compressed input.

        Raises:
            ValueError: If a contraction step or the output would exceed
                `max_bytes` or `max_products`, or if either limit is not a
                positive integer.
        """
        return _reconstruct_cores(self.cores, max_bytes=max_bytes, max_products=max_products)

    def to_dict(self):
        """Scalar metadata only; no expansion, reference computation or copying cores."""
        return dict(num_qubits=self.num_qubits, original_dimension=self.original_dimension,
                    bond_dimensions=list(self.bond_dimensions), max_bond_dim=self.max_bond_dim,
                    threshold=self.threshold, discarded_weight=self.discarded_weight,
                    input_id=self.input_id, input_norm=self.input_norm, core_bytes=self.core_bytes,
                    core_shapes=[list(c.shape) for c in self.cores], provenance=self.provenance)


@dataclass(frozen=True, kw_only=True)
class MPSCompressionAnalysis:
    """Truncation estimates of an MPS decomposition, and an optional comparison with a reference vector.

    [`analyze_mps_state_compression`][nwqlib.subroutines.state_preparation.mps.analyze_mps_state_compression]
    returns it. The estimates follow from the discarded weight alone. The
    `reference_*`, `measured_*`, `raw_measured_fidelity` and
    `fidelity_roundoff_window` fields are filled only when a reference was
    supplied. The fields below are read-only.

    Attributes:
        decomposition: The analyzed decomposition, shared, not copied.
        estimated_raw_l2: Floating estimate sqrt(discarded_weight) under the TT-SVD exact-arithmetic relation.
        estimated_norm_squared: Floating estimate 1-discarded_weight for the raw compressed tensor.
        estimated_normalized_fidelity: Estimated normalized fidelity from that relation, or None for an undefined zero tensor.
        reference_id: Content hash of the supplied comparison reference after normalization, or None without one.
        reference_matches_input: Whether that hash matches the compressed input, or None without a reference.
        reference_norm: Norm of the supplied reference before normalization; None without one.
        measured_raw_l2: Explicitly measured raw-tensor distance to the normalized reference, or None if not computed.
        measured_normalized_fidelity: Measured normalized fidelity to the supplied reference, clipped to [0, 1], or None if unavailable.
        raw_measured_fidelity: Computed ``|<t,r>|**2 / (<t,t> <r,r>)`` for the normalized reference t
            and the raw tensor r before clipping to [0, 1], or None if unmeasured.
        fidelity_roundoff_window: Bound on how far rounding can push that value above one, from
            NWQLib's rounding analysis of this evaluation (see the
            [constants registry](../../ENGINEERING_CONSTANTS.md)). None if unmeasured. It is never
            an approximation allowance.
        method: Name of the analysis, `"mps_ttsvd_analysis"`.
    """

    decomposition: MPSDecomposition
    estimated_raw_l2: float
    estimated_norm_squared: float
    estimated_normalized_fidelity: float | None
    reference_id: str | None = None
    reference_matches_input: bool | None = None
    reference_norm: float | None = None
    measured_raw_l2: float | None = None
    measured_normalized_fidelity: float | None = None
    raw_measured_fidelity: float | None = None
    fidelity_roundoff_window: float | None = None
    method: str = "mps_ttsvd_analysis"

    def to_dict(self):
        """Return a JSON-like record of the analysis, with the decomposition's scalar metadata."""
        result = {key: value for key, value in vars(self).items() if key != "decomposition"}
        result["decomposition"] = self.decomposition.to_dict()
        result["metric_basis"] = "TT-SVD exact-arithmetic relation evaluated as floating estimates; explicit reference metrics only when requested"
        return result


def decompose_state_to_mps(vector, *, max_bond_dim=None, threshold=1e-14,
                           max_bytes=DEFAULT_INPUT_BYTES, max_svd_work=DEFAULT_MAX_SVD_WORK):
    """Normalize a state vector and compute its tensor-train (MPS) cores by TT-SVD, without reconstructing the state.

    Each site follows steps 4 to 7 of Oseledets (2011),
    doi:10.1137/090752286, Algorithm 1
    (p. 2301): reshape the remaining tensor into a
    ``(left_rank*2, right_size)`` matrix, take its truncated SVD, keep the
    left singular vectors ``U`` as the core and carry ``S Vh`` to the next
    site. The truncation rule differs from the paper. Step 5 keeps the
    delta-rank, the smallest rank whose discarded tail has Frobenius norm
    at most ``delta = eps ||A||_F / sqrt(d - 1)`` for a prescribed relative
    accuracy eps and d tensor modes, here the qubit count. This function
    instead drops every singular value at or below ``threshold``
    individually, keeps at least one, and then caps the rank at
    ``max_bond_dim``. NumPy C-order makes the cores MSB-first in the
    original flattened ordering. The whole economy-SVD work of the sweep,
    including factors computed before truncation, is checked before the
    first SVD.

    Args:
        vector (array_like): Nonzero state amplitudes of power-of-two
            length. The input is normalized first.
        max_bond_dim (int | None): Default `None` (no cap). Largest kept
            rank at each bond.
        threshold (float): Default `1e-14`. Singular values at or below it
            are dropped individually. It is not a total error tolerance.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on the known arrays. It does not measure process memory.
        max_svd_work (int): Default `100_000_000`. Limit on the declared
            dense-SVD work of the whole sweep, the sum of rows times columns
            times the smaller dimension over all economy SVDs.

    Returns:
        decomposition (MPSDecomposition): The cores, bond dimensions and
            `discarded_weight`.
    """
    target, norm = _input(vector, max_bytes)
    return _decompose_normalized_state(target, max_bond_dim=max_bond_dim, threshold=threshold,
                                      max_bytes=max_bytes, max_svd_work=max_svd_work, input_norm=norm)


def _decompose_normalized_state(normalized_state, *, max_bond_dim, threshold,
                                max_bytes=DEFAULT_INPUT_BYTES, max_svd_work=DEFAULT_MAX_SVD_WORK,
                                input_norm=1.0):
    """Consume the actual already-normalized input once; no automatic full output."""
    if max_bond_dim is not None:
        integer(max_bond_dim, "max_bond_dim", 1)
    if isinstance(threshold, (bool, np.bool_)) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    integer(max_svd_work, "max_svd_work", 1)
    dimension = normalized_state.size
    if normalized_state.ndim != 1 or dimension < 1 or dimension & (dimension-1):
        raise ValueError("normalized tensor requires a power-of-two vector")
    qubits = dimension.bit_length() - 1
    # Bound the entire prospective sweep before its first SVD. Prior rank caps
    # reduce later matrices, never the full economy factors computed this step.
    rank, core_bytes, work = 1, 0, 0
    # Peak live bytes: an untuned floor of three state-sized arrays, then per
    # step the input and the frozen cores so far plus 16 bytes per complex128
    # entry of the matrix being factored and one working copy
    # (2*rows*columns), the economy U and Vh, the next remaining matrix
    # (kept*columns), the truncated U copy and its frozen core (2*rows*kept),
    # and 8 bytes per float64 singular value.
    peak = 3 * normalized_state.nbytes
    for site in range(qubits-1):
        rows, columns = 2*rank, 1 << (qubits-site-1)
        computed = min(rows, columns)
        kept = computed if max_bond_dim is None else min(computed, max_bond_dim)
        work += rows*columns*computed
        peak = max(peak, normalized_state.nbytes + core_bytes + 16*(
            2*rows*columns + rows*computed + computed*columns + kept*columns + 2*rows*kept) + 8*computed)
        core_bytes += 16*rows*kept
        rank = kept
    if work > max_svd_work:
        raise ValueError(f"TT-SVD needs at most {work} logical SVD work, exceeding max_svd_work={max_svd_work}")
    _check_bytes(peak, max_bytes, "TT-SVD known arrays")
    input_id = _digest("mps.normalized.C-order", (dimension,), (np.ascontiguousarray(normalized_state),))
    remaining = normalized_state.reshape(1, -1)
    rank, ranks, cores, losses = 1, [1], [], []
    for site in range(qubits-1):
        columns = 1 << (qubits-site-1)
        u, s, vh = np.linalg.svd(remaining.reshape(2*rank, columns), full_matrices=False)
        kept = max(1, int(np.count_nonzero(s > threshold)))
        if max_bond_dim is not None:
            kept = min(kept, max_bond_dim)
        losses.append(float(np.dot(s[kept:], s[kept:])))
        # A view of truncated U would keep the entire economy factor alive.
        cores.append(_freeze_array(u[:, :kept].reshape(rank, 2, 1, kept)))
        remaining = s[:kept, None] * vh[:kept, :]
        rank = kept
        ranks.append(rank)
        del u, s, vh
    cores.append(_freeze_array(remaining.reshape(rank, 2 if qubits else 1, 1, 1)))
    ranks.append(1)
    return MPSDecomposition(cores=tuple(cores), num_qubits=qubits, original_dimension=dimension,
                            bond_dimensions=tuple(ranks), max_bond_dim=max_bond_dim,
                            threshold=float(threshold), discarded_weight=fsum(losses),
                            input_id=input_id, input_norm=input_norm)


def analyze_mps_state_compression(decomposition, *, reference=None,
                                  max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000):
    """Estimate the truncation error of MPS cores, and compare them with a reference vector on request.

    The estimates use only the stored discarded weight D: raw L2 distance
    `sqrt(D)`, raw norm squared `1 - D` and normalized fidelity `1 - D`,
    from the orthogonality step in the proof of Oseledets (2011),
    doi:10.1137/090752286, Theorem 2.2 (p. 2299). They are floating-point
    estimates, not measured circuit fidelity or certificates. No SVD or
    contraction runs unless `reference` is given, which requests one
    contraction of the cores and a numerical comparison.

    Args:
        decomposition (MPSDecomposition): Cores from `decompose_state_to_mps`.
        reference (array_like | None): Default `None`. Vector to compare
            with, normalized first. The result says whether it matches the
            compressed input.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on the arrays of the comparison.
        max_products (int): Default `1_000_000_000`. Limit on the scalar
            products of the contraction.

    Returns:
        analysis (MPSCompressionAnalysis): The estimates, and the measured
            comparison when a reference was given.
    """
    if not isinstance(decomposition, MPSDecomposition):
        raise TypeError("analysis requires an existing MPSDecomposition")
    d = decomposition.discarded_weight
    # Proof of Oseledets (2011), doi:10.1137/090752286, Theorem 2.2 (p. 2299):
    # a truncated SVD
    # A_1 = U_1 B_1 + E_1 has U_1^dagger E_1 = 0, so the result U_1 B_1' of
    # the sweep satisfies ||A - B||^2 = ||E_1||^2 + ||B_1 - B_1'||^2, which
    # Eq. (2.5) states as an upper bound. Applied at every site of this
    # sweep, where ||E_k||^2 is the sum of the squared singular values that
    # site discarded, the equality telescopes to ||G - T||^2 = D for the
    # normalized input G and the raw compressed tensor T. The same
    # orthogonality gives <G, T> = ||T||^2 (NWQLib's step, not in the paper),
    # so ||T||^2 = 1 - D and the fidelity of G with T/||T|| is 1 - D.
    # Floating-point SVD makes these estimates, not certificates.
    fields = dict(decomposition=decomposition, estimated_raw_l2=sqrt(d),
                  estimated_norm_squared=1-d, estimated_normalized_fidelity=1-d if d < 1 else None)
    if reference is not None:
        dim = decomposition.original_dimension
        # Cores plus six complex128 state-sized arrays: three for admitting
        # and normalizing the reference (as in _input), the contracted raw
        # tensor, and the two max-scaled copies that the fidelity kernel
        # forms. The difference target - raw is formed after those are freed.
        _check_bytes(decomposition.core_bytes + 6*16*dim, max_bytes, "MPS explicit reference arrays")
        target, norm = _input(reference, max_bytes)
        if target.size != dim:
            raise ValueError("reference dimension differs from the decomposed input")
        target_id = _digest("mps.normalized.C-order", (dim,), (target,))
        raw = decomposition.to_statevector(max_bytes=max_bytes, max_products=max_products)
        raw_norm = stable_vector_norm(raw)
        fidelity, window = None, None
        if raw_norm > 0:
            # |<t, r>|**2 / (<t, t> <r, r>) does not assume that the target or
            # the raw tensor has unit computed norm, so its window, derived in
            # the kernel's docstring, covers only the evaluation. It bounds
            # roundoff, not truncation or SVD error.
            fidelity, window = normalized_fidelity_with_window(target, raw)
        fields.update(reference_id=target_id, reference_matches_input=target_id == decomposition.input_id,
                      reference_norm=norm, measured_raw_l2=stable_vector_norm(target-raw),
                      measured_normalized_fidelity=None if fidelity is None else min(1., max(0., fidelity)),
                      raw_measured_fidelity=fidelity, fidelity_roundoff_window=window)
    return MPSCompressionAnalysis(**fields)


def layered_construction_size(bond_dimensions, num_layers) -> tuple[int, int]:
    """Return upper bounds `(work, bytes)` of building the layered circuit from MPS cores.

    The law follows the scikit_tt calls of
    [`mps_to_circuit`][nwqlib.subroutines.state_preparation.mps_circuit.mps_to_circuit],
    and of the conversion of the cores to a scikit_tt tensor train, for
    cores with these bond dimensions, with an upper bound on each TT rank in
    place of the rank a run reaches. It lives here, without Qiskit, so that
    LCHS planning can check the construction without importing an SDK.

    Ranks. Write ``b_j`` for the bond dimensions of n sites and
    ``S_j = min(2**j, 2**(n-j))``. A left sweep of ``TT.ortho`` sets rank
    j+1 to at most twice rank j, and a right sweep sets rank j to at most
    twice rank j+1. Neither sweep raises a rank, and a truncation only
    lowers one, so after ``ortho`` every rank is at most ``S_j`` and at most
    its value before. The MPO of an extracted two-qubit gate on sites s and s+1 has
    rank 4 at bond s+1 and 1 elsewhere, and ``mpo.dot`` multiplies the ranks
    of the working MPS by those of the MPO. Each bond carries at most one
    such gate per residual pass, so the working ranks at layer l are at
    most ``min(S_j, 4**l b_j)``.

    Work, in the units of the TT-SVD work limit of `decompose_state_to_mps`:

    - An SVD of an m x c core matrix, ``m c min(m, c)``, charged twice
      because scikit_tt repeats a failed ``gesdd`` with ``gesvd``.
    - A left-sweep step with ``k = min(m, c)``: ``diag(s) @ vh``, ``k k c``,
      and its ``tensordot`` into the next core of right rank r, ``2 k c r``.
    - A right-sweep step: the previous core, a ``2 a x m`` matrix, times U,
      ``2 a m k``, and times ``diag(s)``, ``2 a k k``.
    - ``mpo.dot``: four products for each entry pair of MPO and MPS cores,
      twice the entries of the product TT.
    - One unit per entry for each copy or scaling of a TT and 64 units for
      each 4 x 4 SVD of a gate's MPO and each unitary completion.

    Each layer copies the working MPS and applies ``ortho(max_rank=2)``,
    ``norm`` (a copy and a right sweep of the rank-2 result) and the scalar
    multiplication (two copies), and completes n local unitaries. Each of the
    ``num_layers - 1`` residual passes applies n gates, each an MPO SVD,
    ``mpo.dot`` and ``ortho(threshold=1e-12)`` of the product.

    Bytes, 16 per entry: the target vector, the stored cores, the
    right-canonical input TT and the rank-2 copies of a truncation are held
    throughout. On top of them comes the largest of four steps: the initial
    ``ortho``, a truncation (working and truncated TT and their sweep),
    ``mpo.dot`` (the old working TT, the product TT and one core's broadcast
    copy) and the ``ortho`` of the product (the product TT and its sweep).
    The peak of a sweep is the larger of an SVD step and a product step. An
    SVD step holds the reshaped core and the Fortran copy that LAPACK
    factors, U, s and Vh and zgesdd's work arrays, at most ``k k + 194 k``
    complex entries for LAPACK block sizes up to 64 (SciPy 1.18.1's optimal
    size stayed at or below 0.95 of it for m and c up to 4096),
    ``k max(5 k + 7, 2 max(m, c) + 2 k + 1)`` reals and ``8 k`` integers. A
    product step holds U, Vh, ``diag(s)`` with its complex copy and the
    product arrays. An untuned allowance of ``32768 + 4096 n`` bytes covers
    the gate MPOs, the Python objects of the TTs and the headers of their
    core arrays. Under tracemalloc these exceeded the array bytes by up to
    31 kB for ten sites (SciPy 1.18.1). Qiskit circuit objects
    are outside the law, and the two-qubit syntheses have their own
    limit check. Cores whose bond dimensions are all one form a product state,
    which the builder prepares site by site without scikit_tt, 64 units per
    site.
    """
    b = [int(rank) for rank in bond_dimensions]
    n = len(b) - 1

    def entries(ranks):
        return sum(2 * ranks[j] * ranks[j + 1] for j in range(n))

    def svd(m, c):
        k = min(m, c)
        return 2 * m * c * k, (16 * (2 * m * c + m * k + k * c + k * k + 194 * k)
                               + 8 * k * max(5 * k + 7, 2 * max(m, c) + 2 * k + 1) + 40 * k)

    def left(ranks):
        a, work, peak = [1] * (n + 1), 0, 0
        for i in range(n - 1):
            m, c, r = 2 * a[i], ranks[i + 1], ranks[i + 2]
            k = min(m, c)
            svd_work, svd_bytes = svd(m, c)
            work += svd_work + k * k * c + 2 * k * c * r
            peak = max(peak, svd_bytes, 16 * (m * k + 2 * k * c + 2 * c * r + 2 * k * r) + 24 * k * k)
            a[i + 1] = k
        return a, work, peak

    def right(ranks, cap):
        a, work, peak = list(ranks), 0, 0
        for i in range(n - 1, 0, -1):
            m, c = a[i], 2 * a[i + 1]
            full = min(m, c)
            k = min(full, cap)
            svd_work, svd_bytes = svd(m, c)
            rows = 2 * a[i - 1]
            work += svd_work + rows * m * k + rows * k * k
            peak = max(peak, svd_bytes, 16 * (m * full + full * c + rows * m + 2 * rows * k) + 24 * k * k)
            a[i] = k
        return a, work, peak

    def ortho(ranks, cap=1 << 62):
        a, left_work, left_peak = left(ranks)
        a, right_work, right_peak = right(a, cap)
        return a, left_work + right_work, max(left_peak, right_peak)

    if n == 0 or max(b) == 1:
        # The builder prepares a product state site by site without scikit_tt.
        return 64 * max(1, n), 16 * ((1 << n) + entries(b)) + 32768 + 4096 * n
    two = [min(2, 1 << j, 1 << (n - j)) for j in range(n + 1)]
    held = 16 * ((1 << n) + 2 * entries(b) + 4 * entries(two)) + 32768 + 4096 * n
    working, work, peak = ortho(b)
    work += 2 * entries(b)
    for layer in range(num_layers):
        truncated, ortho_work, ortho_peak = ortho(working, 2)
        _, norm_work, norm_peak = right(truncated, 1 << 62)
        work += entries(working) + ortho_work + 3 * entries(truncated) + norm_work + 64 * n
        peak = max(peak, 32 * entries(working) + ortho_peak, 16 * entries(working) + norm_peak)
        if layer + 1 == num_layers:
            break
        for site in range(n - 1, -1, -1):
            product = list(working)
            if site < n - 1:
                product[site + 1] *= 4
            broadcast = max(2 * working[j] * working[j + 1] for j in range(n))
            after, ortho_work, ortho_peak = ortho(product)
            work += 64 + 2 * entries(product) + ortho_work
            peak = max(peak, 16 * (entries(working) + entries(product) + broadcast),
                       16 * entries(product) + ortho_peak)
            working = after
    return work, held + peak


def admit_layered_construction(bond_dimensions, num_layers, *, max_svd_work, max_bytes) -> tuple[int, int]:
    """Raise before a layered construction that exceeds ``max_svd_work`` or ``max_bytes``.

    Returns:
        The ``(work, bytes)`` of ``layered_construction_size``.
    """
    work, data_bytes = layered_construction_size(bond_dimensions, num_layers)
    if work > max_svd_work:
        raise ValueError(f"layered MPS construction needs {work} work, exceeding max_svd_work={max_svd_work}")
    _check_bytes(data_bytes, max_bytes, "layered MPS construction")
    return work, data_bytes


def _reconstruct_cores(cores, *, max_bytes=DEFAULT_INPUT_BYTES, max_products=1_000_000_000):
    """Contract adjacent TT ranks after admitting known arrays and scalar products."""
    core_bytes = sum(core.nbytes for core in cores)
    tensor = cores[0][:, :, 0, :]
    products = 0
    for core in cores[1:]:
        size = tensor.size // tensor.shape[-1] * core.shape[1] * core.shape[-1]
        products += size*tensor.shape[-1]
        _check_products(products, max_products)
        # Current and output tensors plus possible contiguous BLAS input copies.
        _check_bytes(core_bytes + 2*tensor.nbytes + core.nbytes + 16*size, max_bytes, "MPS contraction")
        tensor = np.tensordot(tensor, core[:, :, 0, :], axes=([-1], [0]))
    _check_bytes(core_bytes + 2*tensor.nbytes, max_bytes, "MPS vector output")
    return np.squeeze(tensor, axis=(0, -1)).reshape(-1)


__all__ = ["MPSCompressionAnalysis", "MPSDecomposition", "analyze_mps_state_compression", "decompose_state_to_mps"]
