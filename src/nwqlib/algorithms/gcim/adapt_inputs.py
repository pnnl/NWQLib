"""Finite ADAPT input processing and the choice of commutator terms."""

from dataclasses import dataclass, field
from math import fsum
import math
from typing import Any
import numpy as np
from nwqlib._numerics import stable_complex_sum, stable_vector_norm
from nwqlib.core.records import Complex128, FrozenArray
from nwqlib.operators.inputs import (
    OperatorInput, ingest_pauli, pauli_table, _distinct_pauli_input, _matvec_requirements,
)
from nwqlib.operators.access import _check_bytes, _check_products
from nwqlib.operators._fermion import mapping_requirements
from nwqlib.operators._pauli import PauliTerms, combine_terms
from nwqlib.subroutines.fermionic_pool import (
    FermionicGenerator,
    enumerate_spin_adapted_gsd_pool,
    enumerate_uccsd_sd_pool,
    enumerate_qeb_sd_pool,
    enumerate_ceo_ovp_pool,
    _snapshot_generator,
    _generator_reference,
    _ladder_expansion_terms,
    _reference_pool_size,
)
from .adapt_actions import build_action_tables
from .adapt_records import AdaptPoolMember, AdaptSymmetry, PROCESSING_SOURCE, screen_labels
from nwqlib.algorithms._eigen_inputs import _mean_diagonal
from nwqlib.algorithms.protocol import ApplicabilityError
from .fixed_basis import PauliArrays, PauliTerm, _offset_free_requirements

# Default thresholds of ADAPT as multiples of the energy scale Omega (``energy_scale``). Heuristic choices
# registered in docs/ENGINEERING_CONSTANTS.md. Through Omega each default has the energy unit of H.
DEFAULT_PAULI_CUTOFF_PER_SCALE = 1e-14
DEFAULT_GRADIENT_FLOOR_PER_SCALE = 1e-8
DEFAULT_ENERGY_CHANGE_PER_SCALE = 1e-6


def energy_scale(operator, *, max_bytes, max_products):
    """Return ``Omega = ||H - c I||_F / sqrt(d)`` with ``c = tr(H)/d``, the scale of ADAPT's default thresholds.

    ``operator`` is the Problem's Hermitian H on its original dimension d.
    Omega is a defined scale for a default heuristic. It is neither an
    energy-error certificate nor an optimal stopping tolerance.

    On the full qubit space d = 2**q, Pauli words satisfy
    ``tr(P^dagger Q) = d delta_PQ``, so a sum ``H = sum_P c_P P`` with duplicate
    words combined has ``||H - c I||_F**2 = d sum_{P != I} |c_P|**2`` and
    ``Omega = sqrt(sum_{P != I} |c_P|**2)``, an O(J) scan of its J coefficients.
    With k stored diagonal entries, whose index set is D, a dense matrix
    (k = d) or canonical CSR/CSC matrix has

        d Omega**2 = sum_{i != j} |H_ij|**2 + sum_{i in D} |H_ii - c|**2 + (d - k) |c|**2,

    an O(d**2) or O(d + nnz) scan: the formula is linear in the stored entries,
    and the line pointers and the length-d array of ``_mean_diagonal`` add the
    d. The last term is counted by its multiplicity, with no zeros added to
    the sum and no identity formed. The parts are combined by scaled norms (BLAS NRM2 and ``hypot``), because ``tr(H**2)/d - c**2``
    would cancel. No matrix is densified, and no Pauli decomposition, eig or
    SVD is made. ``_mean_diagonal`` returns a constant diagonal itself, so H = cI
    gives Omega = 0 exactly. Omega = 0 holds if and only if H = cI.

    For ``H' = aH + bI`` with a > 0, ``c' = ac + b`` and ``Omega' = a Omega``.
    The Pauli, dense and sparse forms of one operator have the same exact
    Omega, up to rounding in each scan. Equal Omega does not give equal
    processed Hamiltonians, because the Pauli path prunes and the classical
    matrix path does not. With J nonidentity coefficients,
    ``max|c_P| <= Omega <= sqrt(J) max|c_P|`` and ``Omega <= sum|c_P|``, so
    Omega can grow with the size of the system: J equal orthogonal terms give
    ``Omega = |c| sqrt(J)``.

    Mean padding to dimension D > d (``eigen_operator``) keeps
    ``||H_pad - c I||_F = ||H - c I||_F``, so the padded operator with D in
    place of d would give a smaller value. This function reads the original
    operator, and padded Pauli coefficients would need
    ``Omega = sqrt((D/d) sum_{P != I} |c_P|**2)``. That relation is specific
    to mean padding.

    The known work and scratch are checked before the scan, which each
    planning makes once. One work unit is one visited entry in one pass. The
    Pauli scan tests the two mask words of each row and makes three passes over
    the rows to combine the tests, select the coefficients and take their norm,
    with three mask bytes and one 16-byte coefficient copy per row. The dense
    scan reads d**2 entries and makes seven passes over length-d arrays, with
    up to 32d bytes for row norms and diagonal arrays. The sparse scan makes
    seven passes over its z stored entries and nine over the lines, with I
    bytes of line index and two mask bytes per entry, an off-diagonal copy of up
    to 16z and up to 48d bytes for diagonal arrays.
    """
    d = operator.basis.dimension
    if "pauli_terms" in operator.manifest.access:
        table = operator.pauli_terms()
        _check_products(len(table) * (2 * table.x.shape[1] + 3), max_products)
        _check_bytes(19 * len(table), max_bytes, "ADAPT energy scale")
        return stable_vector_norm(table.coefficients[table.x.any(axis=1) | table.z.any(axis=1)])
    if operator.reference.representation == "dense":
        matrix = operator.dense_array()
        _check_products(d * d + 7 * d, max_products)
        _check_bytes(32 * d, max_bytes, "ADAPT energy scale")
        # Row slices need no copy of H and no second array for H - cI.
        off_diagonal = stable_vector_norm(np.fromiter(
            (math.hypot(stable_vector_norm(matrix[i, :i]), stable_vector_norm(matrix[i, i + 1:])) for i in range(d)),
            dtype=np.float64, count=d))
        stored = matrix.diagonal().real
    else:
        data, indices, indptr = operator.sparse_entries(max_bytes=max_bytes)
        _check_products(7 * len(data) + 9 * d, max_products)
        _check_bytes((indices.dtype.itemsize + 18) * len(data) + 48 * d, max_bytes, "ADAPT energy scale")
        lines = np.repeat(np.arange(d, dtype=indices.dtype), np.diff(indptr))
        on_diagonal = indices == lines
        # ponytail: this copy doubles the stored values for the length of the scan, as _shifted_sparse does.
        # Scan tiles of lines if a sparse matrix near the memory limit needs this path.
        off_diagonal = stable_vector_norm(data[~on_diagonal])
        stored = data[on_diagonal].real
    mean = _mean_diagonal(operator, max_bytes=max_bytes)
    parts = (off_diagonal, stable_vector_norm(stored - mean), math.sqrt(d - len(stored)) * abs(mean))
    return stable_vector_norm(np.array(parts)) / math.sqrt(d)


def _symmetry_evidence(
    original: np.ndarray,
    correction: np.ndarray | None,
    *,
    tolerance: float,
    target: str,
    exact_symmetry: bool,
    pauli_qubits: int = 0,
) -> dict[str, Any]:
    """Record the evidence of the input check, using ``correction=None`` for verified exact symmetry.

    The relative correction ``||correction||_F / ||original||_F`` is compared
    with ``tolerance`` through ``frexp`` mantissas and exponents, so neither
    overflow nor underflow of an intermediate can change the outcome of the
    check. For a Pauli sum on ``pauli_qubits`` qubits the absolute
    Frobenius norm is ``2**(pauli_qubits/2)`` times the coefficient 2-norm,
    because ``Tr(P_i^dagger P_j) = 2**pauli_qubits delta_ij``. The ratio needs
    no such factor.
    """

    if not np.all(np.isfinite(original)):
        raise ValueError(f"{target} input must be finite")
    tolerance = float(tolerance)
    nonzero = correction is not None and bool(
        np.any(correction.real != 0.0) or np.any(correction.imag != 0.0)
    )
    correction_scale = 0.0
    relative = 0.0
    exceeds_tolerance = False
    if nonzero:
        original_scale = float(
            np.max(np.maximum(np.abs(original.real), np.abs(original.imag)), initial=0.0)
        )
        correction_scale = float(
            np.max(np.maximum(np.abs(correction.real), np.abs(correction.imag)), initial=0.0)
        )
        correction_mantissa, correction_exponent = math.frexp(correction_scale)
        original_mantissa, original_exponent = math.frexp(original_scale)
        relative_mantissa, relative_exponent = math.frexp(
            (correction_mantissa / original_mantissa)
            * math.hypot(
                stable_vector_norm(correction.real / correction_scale),
                stable_vector_norm(correction.imag / correction_scale),
            )
            / math.hypot(
                stable_vector_norm(original.real / original_scale),
                stable_vector_norm(original.imag / original_scale),
            )
        )
        relative_exponent += correction_exponent - original_exponent
        tolerance_mantissa, tolerance_exponent = math.frexp(tolerance)
        exceeds_tolerance = tolerance == 0.0 or (
            (relative_exponent, relative_mantissa) > (tolerance_exponent, tolerance_mantissa)
        )
        relative = math.ldexp(relative_mantissa, relative_exponent)
    # Exact mode checks components: an actual nonzero ratio may underflow.
    if (tolerance == 0.0 and not exact_symmetry) or exceeds_tolerance:
        raise ValueError(f"input must be {target} within symmetry_tol")
    correction_norm = stable_vector_norm(correction) if nonzero else 0.0
    try:
        correction_norm = math.ldexp(correction_norm, pauli_qubits // 2)
        if pauli_qubits % 2:
            correction_norm *= math.sqrt(2.0)
    except OverflowError:
        correction_norm = math.inf
    return {
        "target": target,
        "projection": "(H + H.adjoint)/2" if target == "Hermitian" else "(A - A.adjoint)/2",
        "symmetry_tol": tolerance,
        "correction_applied": nonzero,
        "correction_component_max": correction_scale,
        "correction_frobenius_norm": correction_norm if math.isfinite(correction_norm) else None,
        "correction_frobenius_norm_status": "finite"
        if math.isfinite(correction_norm)
        else "overflow",
        "relative_frobenius_correction": None if nonzero and relative == 0.0 else relative,
        "relative_frobenius_correction_status": "underflow"
        if nonzero and relative == 0.0
        else "finite",
        "certificate_target": "processed_operator",
    }


def _project_terms(terms, *, antihermitian, tolerance, qubits):
    """Project combined Pauli terms onto their (anti-)Hermitian part with evidence.

    Pauli strings are Hermitian, so ``(H + H^dagger)/2`` keeps the real part of
    each coefficient and ``(A - A^dagger)/2`` keeps ``i`` times the imaginary
    part. Terms that vanish after projection are dropped.
    """
    terms = combine_terms(terms)
    processed, evidence = _project_coefficients(
        np.array([c for _, c in terms], dtype=complex), antihermitian=antihermitian, tolerance=tolerance,
        qubits=qubits)
    return tuple(
        (p, complex(c)) for (p, _), c in zip(terms, processed, strict=True) if c != 0
    ), evidence


def _project_coefficients(original, *, antihermitian, tolerance, qubits):
    """Return the projected coefficients of distinct Pauli words and the evidence of ``_project_terms``."""
    processed = 1j * original.imag if antihermitian else original.real.astype(complex)
    evidence = _symmetry_evidence(
        original,
        processed - original,
        tolerance=tolerance,
        target="anti-Hermitian" if antihermitian else "Hermitian",
        exact_symmetry=bool(np.array_equal(original, processed)),
        pauli_qubits=qubits,
    )
    return processed, evidence


@dataclass(frozen=True)
class AdaptInputs:
    """Processed ADAPT inputs bound to a Plan and restored by archive loading.

    ``hamiltonian`` and ``pool`` are the processed operators that queries act
    on, and ``members`` keep each generator's raw identity, symmetry evidence
    and ``[H, A_i]`` terms. ``terms`` are the kept Hamiltonian Pauli terms and
    ``dropped_l1`` is the sum of the absolute values of the removed Hamiltonian coefficients.
    ``groups`` and ``energy_groups`` are the QWC measurement groups for screening and energy
    queries, as indices into ``adapt_records.screen_labels`` and
    ``adapt_records.energy_labels``. For a classical plan that keeps a dense or sparse matrix,
    ``host_identity_shift`` is its mean diagonal c and
    ``offset_free_requirements`` the (bytes, work) that removing c adds to one
    pair action. Both are zero for a Pauli Hamiltonian.

    ``nonidentity`` holds the ``(label, coefficient)`` rows of the nonidentity
    Hamiltonian terms, formed once here for every pair circuit and pair
    action. ``cache`` holds labels decoded from the member records. Neither
    is saved.
    """

    hamiltonian: OperatorInput
    pool: tuple[FermionicGenerator, ...]
    members: tuple[AdaptPoolMember, ...]
    terms: tuple[PauliTerm, ...]
    groups: tuple[tuple[int, ...], ...]
    energy_groups: tuple[tuple[int, ...], ...]
    dropped_l1: float
    processing: tuple[str, ...]
    hamiltonian_work: int
    hamiltonian_symmetry: AdaptSymmetry
    processing_source: object = PROCESSING_SOURCE
    host_identity_shift: float = 0.0
    offset_free_requirements: tuple[int, int] = (0, 0)
    nonidentity: tuple = field(init=False, repr=False, compare=False)
    cache: dict = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self):
        if self.terms:
            q = len(self.terms[0].label)
            rows = tuple((t.label, t.coefficient) for t in self.terms if t.label != "I" * q)
        else:
            rows = ()
        object.__setattr__(self, "nonidentity", rows)


# Candidate Pauli pairs tested per commutator tile. An engineering choice of
# tile length, not a scientific threshold. Changing it changes storage, not
# rows, order or coefficients.
_COMMUTATOR_TILE = 1 << 14


def pauli_array_conversion_requirements(rows, q, *, tile=1024):
    """Return ``(work, peak bytes, raw bytes)`` of converting J packed rows with distinct labels to ``PauliArrays``.

    Here J = ``rows``, q is the number of qubits, W = ceil(q/64) is the
    number of 64-bit words per mask, and C = ``tile``.
    With ``P = (16W+16)J``, ``B = 8J(q+1)``, ``t = min(J, C)`` and
    ``h = ceil(log2 max(1, J))``, the live phases are decoding (packed input
    P, code and coefficient outputs B, two uint64 tile vectors 16t), sorting
    and permuting (P, unsorted and sorted outputs 2B, indirect permutation
    8J) and immutable output and validation (P, working outputs and their
    immutable copies 2B, validator scratch at most ``J*q + 40J``), so

        B_conv = P + max{B + 16t, 2B + 8J, 2B + Jq + 40J} + 65536.

    One work unit is one scalar code, word or index visit, with an inspected
    byte of a row-key comparison counted as one unit. The decoder has eight
    vector operations per character and a coefficient pass, and heap
    construction and removal have fewer than 4Jh key comparisons of at most
    8q bytes and 4Jh index moves, so

        W_conv = J(16q + 32) + (32q + 8)Jh.

    These are defined accounting units, not literal NumPy instruction counts.
    ``65536`` is the CPython/NumPy engineering allowance for fixed overhead,
    not a bound that holds for every memory allocator.
    """
    if type(rows) is not int or rows < 0 or type(q) is not int or q < 1:
        raise ValueError("invalid Pauli array dimensions")
    if type(tile) is not int or tile < 1:
        raise ValueError("tile must be positive")
    words = (q+63)//64
    packed = (16*words+16)*rows
    raw = 8*rows*(q+1)
    depth = (max(1, rows)-1).bit_length()
    peak = packed + max(raw+16*min(rows, tile),
                        2*raw+8*rows, 2*raw+rows*q+40*rows) + 65536
    work = rows*(16*q+32) + (32*q+8)*rows*depth
    return work, peak, raw


def _packed_to_pauli_arrays(x, z, coefficients, q, *, tile=1024):
    """Return packed rows with distinct labels and real coefficients as a sorted ``PauliArrays`` record.

    Each printed character reads its bit from high qubit to low qubit and
    stores the int64 code ``c = (x_bit XOR z_bit) + 2*z_bit``, so the pairs
    ``(x, z) = (0, 0), (1, 0), (1, 1), (0, 1)`` give 0, 1, 2, 3, that is
    I, X, Y, Z. Rows are then sorted lexicographically with their
    coefficients by a heapsort of fixed-size row keys, whose bytes compare
    in code-row order because every code is one of 0 to 3. The input rows
    are distinct canonical keys, so sort stability is irrelevant here. No
    label string is formed. The caller checks
    ``pauli_array_conversion_requirements`` against the limits before this call.
    """
    rows = len(coefficients)
    labels = np.empty((rows, q), dtype=np.int64)
    values = np.ascontiguousarray(coefficients, dtype=np.float64).copy()
    for start in range(0, rows, tile):
        stop = min(rows, start+tile)
        xx = np.empty(stop-start, dtype=np.uint64)
        zz = np.empty(stop-start, dtype=np.uint64)
        for col, bit in enumerate(range(q-1, -1, -1)):
            np.right_shift(x[start:stop, bit//64], np.uint64(bit % 64), out=xx)
            np.bitwise_and(xx, np.uint64(1), out=xx)
            np.right_shift(z[start:stop, bit//64], np.uint64(bit % 64), out=zz)
            np.bitwise_and(zz, np.uint64(1), out=zz)
            np.bitwise_xor(xx, zz, out=xx)
            np.left_shift(zz, np.uint64(1), out=zz)
            np.bitwise_or(xx, zz, out=xx)
            np.copyto(labels[start:stop, col], xx, casting="unsafe")
        del xx, zz
    keys = labels.view(np.dtype((np.void, 8*q))).reshape(rows)
    order = np.argsort(keys, kind="heapsort")
    del keys
    sorted_labels, sorted_coefficients = labels[order], values[order]
    del labels, values, order
    return PauliArrays(labels=FrozenArray(sorted_labels),
                       coefficients=FrozenArray(sorted_coefficients))


def _commutator_arrays(h, a, *, cutoff, ledger, held, max_bytes, max_products):
    """Return ``[H, A]`` as ``PauliArrays`` and the sum of its dropped coefficient magnitudes, from packed tables.

    For canonical Pauli strings ``P_a P_h = (-1)**s_ha P_h P_a``, hence
    ``[P_h, P_a] = (1 - (-1)**s_ha) P_h P_a``, zero when ``s_ha = 0`` and
    ``2 P_h P_a`` when ``s_ha = 1``. In ``PauliTerms``,
    ``P(x, z) = i**|x & z| X**x Z**z``, and with ``e = x_h XOR x_a`` and
    ``f = z_h XOR z_a``, ``P_h P_a = i**nu P(e, f)`` with
    ``nu = |x_h & z_h| + |x_a & z_a| + 2|z_h & x_a| - |e & f| (mod 4)``
    (``operators._pauli.product_words``). Every surviving pair contributes
    ``2 c_h c_a i**nu`` to its output label. For example XZ = -iY and
    ZX = +iY, and for H = X and A = iZ, [H, A] = 2Y, which fixes the ADAPT
    sign. Its derivative convention is
    ``d<exp(theta A) psi|H|exp(theta A) psi>/d theta at 0 = <[H, A]>``,
    because A is anti-Hermitian. All surviving pairs with the same output
    label are accumulated before the cutoff, which is algebraically
    identical to the two ordered products and their subtraction by
    bilinearity. After the ADAPT input checks, H has real coefficients and A
    purely imaginary ones, so each contribution is real. The Hermiticity
    check and ``commutator_dropped_l1`` apply to the summed output. Empty
    H or A, commuting pairs, identity strings and complete cancellation all
    give zero without an invented residual. ``stable_complex_sum`` uses
    ``fsum``, not arbitrary-precision addition. Changing multiplication,
    doubling and accumulation order relative to the product-and-cancel
    route can change the last bits or an intermediate overflow decision, so
    agreement holds where the finite-arithmetic assumptions of both
    constructions hold. Finite pair products and finite summed outputs are
    required, and no individual pair contribution is pruned (Zheng et al.,
    arXiv:2312.07691v3, commutator screen, and Higham, Accuracy and Stability
    of Numerical Algorithms, 2nd ed., doi:10.1137/1.9780898718027).

    Commutator construction tests bounded tiles of packed pairs, stores only
    anticommuting products and sums all surviving products with equal labels
    before returning them. One work total covers every pool member, and the
    byte check includes completed member tables, the current surviving
    products and their sorting and copying workspace. With L rows of H, T
    rows of A, ``P = L*T`` tested pairs, C anticommuting surviving products,
    J output rows after summing, W packed words and tiles of at most b pairs
    (the pair index is flattened, so a tile need not hold a whole row),

        B_tile = (64W + 128)b,    B_coalesce(C) = (80W + 128)C + 65536,
        W_j = P(W + 1) + C(12W + 16) + (2W + 2)C ceil(log2 max(2, C)).

    ``B_coalesce`` allows five copies of packed X/Z data (raw, sortable keys,
    gathered keys, output and immutable output copy) and 128C for
    complex coefficients, their gathered and output copies, sort, inverse
    and first-position indices, head masks and group order. ``ledger`` is
    the one running work total shared by all pool members, a one-element
    list. Each pair tile is checked before it is tested, the bytes of each
    tile's worst-case surviving products are set aside before they are
    appended, and the sort and output stages are checked once C is known.
    ``held`` is the bytes of the earlier output tables ``8J_i(q+1)``. The
    packed surviving products then convert to the output int64 code table
    (``pauli_array_conversion_requirements``), with the conversion peak
    ``B_conv`` added under the same maximum.
    """
    from nwqlib.operators._pauli import anticommutes, product_words

    q = h.num_qubits
    if a.num_qubits != q:
        raise ValueError("commutator requires equal Pauli widths")
    words = h.x.shape[1]
    left, right = len(h), len(a)
    pairs = left * right

    def charge(work):
        ledger[0] += work
        _check_products(ledger[0], max_products)

    def reserve(size):
        _check_bytes(held + size, max_bytes, "ADAPT commutators")

    xs, zs, cs = [], [], []
    survivors = 0
    for start in range(0, pairs, _COMMUTATOR_TILE):
        stop = min(pairs, start + _COMMUTATOR_TILE)
        count = stop - start
        charge(count * (words + 1))
        raw = survivors * (16 * words + 16)
        reserve(max(raw + (64 * words + 128) * count,
                    (80 * words + 128) * (survivors + count) + 65536))
        flat = np.arange(start, stop, dtype=np.intp)
        hi, ai = flat // right, flat % right
        del flat
        keep = anticommutes(h.x[hi], h.z[hi], a.x[ai], a.z[ai])
        hi, ai = hi[keep], ai[keep]
        del keep
        if not len(hi):
            continue
        charge(len(hi) * (12 * words + 16))
        x, z, phase = product_words(h.x[hi], h.z[hi], a.x[ai], a.z[ai])
        with np.errstate(over="raise", invalid="raise"):
            try:
                c = (h.coefficients[hi] * a.coefficients[ai]) * phase
                c = 2.0 * c
            except FloatingPointError:
                raise ValueError("commutator coefficients must be finite") from None
        if not np.isfinite(c).all():
            raise ValueError("commutator coefficients must be finite")
        xs.append(x)
        zs.append(z)
        cs.append(c)
        survivors += len(hi)
    depth = math.ceil(math.log2(max(2, survivors)))
    charge((2 * words + 2) * survivors * depth)
    reserve((80 * words + 128) * survivors + 65536)
    if survivors:
        x = np.concatenate(xs)
        z = np.concatenate(zs)
        c = np.concatenate(cs)
    else:
        x = np.zeros((0, words), dtype=np.uint64)
        z = np.zeros((0, words), dtype=np.uint64)
        c = np.zeros(0, dtype=np.complex128)
    del xs, zs, cs
    # Stable packed-key sort, then one componentwise stable_complex_sum per
    # key in pair order.
    order = np.lexsort(tuple(np.concatenate((z, x), axis=1).T[::-1])) if survivors else np.zeros(0, np.intp)
    x, z, c = x[order], z[order], c[order]
    del order
    if survivors:
        heads = np.ones(survivors, dtype=bool)
        heads[1:] = np.any(x[1:] != x[:-1], axis=1) | np.any(z[1:] != z[:-1], axis=1)
        starts = np.flatnonzero(heads)
    else:
        starts = np.zeros(0, dtype=np.intp)
    # stable_complex_sum per key. A sum of one or two components is one
    # correctly rounded addition, the value fsum returns (+0.0 for a zero
    # sum), so only longer runs call fsum. Every overflow raises its error.
    sizes = np.diff(np.append(starts, survivors))
    real, imag = c.real[starts] + 0.0, c.imag[starts] + 0.0
    pairs = sizes == 2
    with np.errstate(over="ignore", invalid="ignore"):
        real[pairs] = (c.real[starts[pairs]] + c.real[starts[pairs] + 1]) + 0.0
        imag[pairs] = (c.imag[starts[pairs]] + c.imag[starts[pairs] + 1]) + 0.0
    if not (np.isfinite(real).all() and np.isfinite(imag).all()):
        raise ValueError("coefficient accumulation overflowed or is nonfinite; coefficients must be finite")
    for index in np.flatnonzero(sizes > 2).tolist():
        start, stop = int(starts[index]), int(starts[index] + sizes[index])
        value = stable_complex_sum(c.real[start:stop], c.imag[start:stop])
        real[index], imag[index] = value.real, value.imag
    if np.any(imag != 0):
        raise ValueError("[H,A] must be Hermitian after coalescing its surviving pairs")
    magnitude = np.abs(real)
    # fsum is correctly rounded, so the order of the removed magnitudes does not matter.
    removed = fsum(magnitude[magnitude <= cutoff].tolist())
    kept = magnitude > cutoff
    x, z, real = x[starts][kept], z[starts][kept], real[kept]
    del c, starts, sizes, pairs, imag, magnitude, kept
    work, peak, _ = pauli_array_conversion_requirements(len(real), q)
    charge(work)
    reserve(peak)
    return _packed_to_pauli_arrays(x, z, real, q), removed


def _pool_enumeration_requirements(candidates, num_qubits):
    """Return ``(bytes, work)`` for enumerating ``candidates`` built-in pool members on ``num_qubits`` qubits.

    Every built-in member has at most 12 normal-ordered ladder strings of at
    most four operators and at most 48 Pauli rows after equal labels are
    summed. A triplet double
    of Zheng et al., arXiv:2312.07691v3, Eq. (E3), on four distinct orbitals
    reaches both. Its six base terms and their adjoints are 12 strings on six
    distinct four-mode sets, and the Jordan-Wigner image of the
    anti-Hermitian pair on one set has 8 Pauli rows.

    Bytes. The enumerated pool stays in memory while each member is checked,
    so each candidate is counted at the size of the largest member, ``q + 64``
    per Pauli row, ``48 + 48 * 4 = 240`` per four-operator string, and 206
    for the family name and index tuples
    (64 plus the at most 14 characters of a built-in family name, and 16
    per index of at most four spatial and four spin-orbital indices). The
    mapping arrays of the one member being built, the byte term of
    ``mapping_requirements`` for its 12 strings, are added once.

    Work. Per candidate, the work term of ``mapping_requirements`` for 12
    four-operator strings with label output, the formula whose byte term
    ``_ladder_expansion_terms`` checks for each member, and 168 comparisons
    for normal ordering. A spin-adapted candidate is normal-ordered twice,
    in ``_skip_zero_base`` and in ``_generator``, with at most 84
    comparisons each time. The count depends only on the order pattern of
    the spatial indices, and every pattern of four indices occurs with four
    spatial orbitals, so an exhaustive count over four spatial orbitals gives
    the maximum.
    """
    q = num_qubits
    _, mapping_bytes, mapping_work = mapping_requirements((4,) * 12, num_modes=q, labels=True)
    member_bytes = 48 * (q + 64) + 12 * 240 + 206
    return candidates * member_bytes + mapping_bytes, candidates * (mapping_work + 168)


def prepare_adapt_inputs(hamiltonian, reference, method, *, execution, shots, original_dimension, cutoff):
    """Check the finite generator pool and choose the Hamiltonian and commutator terms to measure.

    A named pool is enumerated only after its candidate count passes the byte
    and product limits. Every generator is projected onto its anti-Hermitian
    part within ``symmetry_tol`` and must agree with its stored
    fermion terms. A Hamiltonian with Pauli access, which every quantum plan
    has, must have exactly real Pauli coefficients. ``cutoff`` is the
    effective ``pauli_cutoff``, resolved once by ``ADAPT.plan``
    from the original Problem. Nonidentity coefficients at or below it are
    removed, and the sum of their absolute values is recorded. The identity
    coefficient stays whole, because removing it when it crosses the cutoff
    would break ``E(H + bI) = E(H) + b``. Only a quantum plan of the default
    growth forms each ``[H, A_i]``, whose labels its screening queries measure
    and whose coefficients the same ``cutoff`` prunes, and only a quantum plan
    with ``shots`` forms the QWC measurement groups. A
    classical plan screens with generator actions on the state and forms
    neither, and ``BoundedRitzGrowth`` screens no gradient and forms no
    commutator. A classical plan with
    dense or sparse access keeps the original matrix and forms no Pauli
    decomposition. Its pair queries then remove the mean diagonal
    ``c = trace(H)/d`` from the matrix before they project, as classical
    FixedGCIM does, so the projected entries and their rounding do not grow
    with an identity offset of H. The solve adds c back.

    Returns:
        The ``AdaptInputs`` bound to the Plan.
    """
    q = reference.basis.dimension.bit_length() - 1
    # hamiltonian_action_work counts the full-state action, which reads each
    # Pauli mask as one 64-bit word (operators.inputs._matvec_requirements).
    if q > 64:
        raise ApplicabilityError(f"ADAPT supports at most 64 qubits because its Hamiltonian action uses single-word "
                                 f"Pauli masks, and this input has {q}. No ADAPT option changes this width")
    pool = method.pool
    growth = method.basis_growth
    # Bound built-in pool enumeration before expanding orbital-index combinations.
    if isinstance(pool, str):
        n = method.n_spatial_orbitals
        if type(n) is not int or n < 1 or 2 * n != q:
            raise ValueError(
                "built-in pool needs n_spatial_orbitals matching the interleaved width"
            )
        families = {
            "spin_adapted_sd": enumerate_spin_adapted_gsd_pool,
            "uccsd_sd": enumerate_uccsd_sd_pool,
            "qeb_sd": enumerate_qeb_sd_pool,
            "ceo_ovp": enumerate_ceo_ovp_pool,
        }
        if pool not in families:
            raise ValueError("unknown built-in ADAPT pool")
        if pool != "spin_adapted_sd" and reference.reference.representation != "occupation":
            raise ValueError("reference-dependent pool requires an occupation preparation")
        # Candidate members: at most P singles and P(P + 1) doubles for the
        # P = n(n+1)/2 spatial pairs of the spin-adapted pool, whose
        # enumeration skips members that vanish. An occupied-virtual pool has
        # an exact member count fixed by the reference occupation.
        if pool == "spin_adapted_sd":
            pairs = n * (n + 1) // 2
            candidates = pairs + pairs * (pairs + 1)
        else:
            candidates = _reference_pool_size(pool, reference._physical)
        required_bytes, required_work = _pool_enumeration_requirements(candidates, q)
        _check_bytes(required_bytes, method.max_bytes, "ADAPT pool enumeration")
        _check_products(required_work, method.max_products)
        pool = families[pool](n if pool == "spin_adapted_sd" else reference._physical)
    if type(pool) is not tuple or not pool:
        raise ValueError("ADAPT needs a nonempty finite pool")
    generators = []
    members = []
    qualifiers = []
    # Convert each supported input to the same anti-Hermitian Pauli convention
    # while keeping its original identity and any symmetry correction evidence.
    for index, generator in enumerate(pool):
        if isinstance(generator, OperatorInput):
            if generator.basis.dimension != original_dimension:
                raise ValueError("custom generators must use the original Eigenproblem dimension")
            original = generator.reference
            if "pauli_terms" in generator.manifest.access:
                table = generator.pauli_terms()
                x, z, values = table.x, table.z, table.coefficients
            else:
                from nwqlib.algorithms._eigen_inputs import dense_pauli_matrix
                from nwqlib.operators._pauli import pauli_coefficient_masks

                matrix = dense_pauli_matrix(
                    generator,
                    input_conversion=method.input_conversion,
                    max_bytes=method.max_bytes,
                )
                x, z, values = pauli_coefficient_masks(matrix, max_bytes=method.max_bytes)
                del matrix
            # The words of a Pauli table or a block transform are distinct, so
            # the projection reads the arrays and the processed operator takes
            # the kept rows without coalescing. Labels are decoded once, for
            # the kept words that the member and generator records store.
            values, evidence = _project_coefficients(
                values, antihermitian=True, tolerance=method.symmetry_tol, qubits=q)
            kept = values != 0
            x, z, values = x[kept], z[kept], values[kept]
            # The kept rows, their zero-signed copy and the operator's copy.
            _check_bytes(len(x) * (2 * (16 * x.shape[1] + 16) + 16), method.max_bytes,
                         "ADAPT generator Pauli operator")
            # i*Im(c) has a negative-zero real part for Im(c) < 0. Adding +0.0
            # gives the positive zero that summing identical words produces,
            # so the operator has the content hash of ingesting these rows.
            processed = _distinct_pauli_input(q, x, z, values + 0.0)
            rows = PauliTerms._snapshot(q, x, z, values).labels(max_bytes=method.max_bytes)
            family, spatial, spin, fermions = "custom", (), None, ()
        elif isinstance(generator, FermionicGenerator):
            generator = _snapshot_generator(generator)
            if (
                generator.num_qubits != q
                or 1 << q != original_dimension
                or generator.pool_index != index
            ):
                raise ValueError(
                    "pool generators require matching width and consecutive stable indices"
                )
            rows = generator.pauli_terms
            family, spatial, spin, fermions = (
                generator.family,
                generator.spatial_indices,
                generator.spin_orbital_indices,
                generator.fermion_terms,
            )
            original = _generator_reference(generator)
            rows, evidence = _project_terms(
                rows, antihermitian=True, tolerance=method.symmetry_tol, qubits=q
            )
            processed = ingest_pauli(rows, num_qubits=q, max_bytes=method.max_bytes)
        else:
            raise TypeError("pool entries must be Pauli OperatorInput or FermionicGenerator")
        # A named pool's members come from fermionic_pool._generator, whose
        # Pauli terms are the Jordan-Wigner image of the same fermion terms,
        # so only a supplied generator is mapped again.
        if fermions and not isinstance(method.pool, str):
            expected = _ladder_expansion_terms(
                fermions, q, mapping="jw", max_bytes=method.max_bytes
            )
            if dict(expected) != dict(rows):
                raise ValueError("generator fermion and Pauli terms differ")
        generators.append(
            FermionicGenerator(
                family=family,
                spatial_indices=spatial,
                spin_orbital_indices=spin,
                pool_index=index,
                num_qubits=q,
                fermion_terms=fermions,
                pauli_terms=rows,
            )
        )
        members.append(
            AdaptPoolMember(
                original_input=original,
                processed_input=processed.reference,
                symmetry=AdaptSymmetry(**evidence),
                pool_index=index,
                family=family,
                spatial_indices=spatial,
                spin_orbital_indices=spin,
                terms=tuple((p, Complex128(real=c.real, imag=c.imag)) for p, c in rows),
                fermion_terms=tuple(
                    (Complex128(real=c.real, imag=c.imag), ops) for c, ops in fermions
                ),
            )
        )
        qualifiers.append(
            f"pool {index}: symmetry correction applied={evidence['correction_applied']}"
        )
    # The classical matrix route can apply the original Hermitian operator
    # directly, without paying for or silently pruning a Pauli decomposition.
    if "pauli_terms" not in hamiltonian.manifest.access:
        if execution != "classical":
            raise ValueError("quantum ADAPT requires the selected Pauli representation")
        original = (
            hamiltonian.dense_array()
            if hamiltonian.reference.representation == "dense"
            else hamiltonian.sparse_entries(max_bytes=method.max_bytes)[0]
        )
        evidence = _symmetry_evidence(
            original, None, tolerance=0.0, target="Hermitian", exact_symmetry=True
        )
        shift = _mean_diagonal(hamiltonian, max_bytes=method.max_bytes)
        offset = _offset_free_requirements(hamiltonian, 1, shift)
        work = _matvec_requirements(hamiltonian)[2]
        symmetry = AdaptSymmetry(**evidence)
        notes = tuple(qualifiers) + (
            "exact original Hermitian matrix; no Pauli decomposition or pruning",
        )
        inputs = AdaptInputs(
            hamiltonian,
            tuple(generators),
            tuple(members),
            (),
            (),
            (),
            0.0,
            notes,
            work,
            symmetry,
            host_identity_shift=shift,
            offset_free_requirements=offset,
        )
        build_action_tables(inputs, q, max_bytes=method.max_bytes, max_products=method.max_products)
        return inputs
    raw = hamiltonian.pauli_terms().labels(max_bytes=method.max_bytes)
    rows, evidence = _project_terms(raw, antihermitian=False, tolerance=0.0, qubits=q)
    identity = "I" * q
    dropped = fsum(abs(c.real) for p, c in rows if p != identity and abs(c.real) <= cutoff)
    rows = tuple((p, c.real) for p, c in rows if p == identity or abs(c.real) > cutoff)
    processed = ingest_pauli(rows, num_qubits=q, max_bytes=method.max_bytes)
    terms = tuple(PauliTerm(label=p, coefficient=c) for p, c in rows)
    qualifiers.append(f"Hamiltonian Pauli cutoff {cutoff}; dropped L1 {dropped}")
    work = _matvec_requirements(processed)[2]
    symmetry = AdaptSymmetry(**evidence)
    if execution != "quantum":
        inputs = AdaptInputs(
            processed,
            tuple(generators),
            tuple(members),
            terms,
            (),
            (),
            dropped,
            tuple(qualifiers),
            work,
            symmetry,
        )
        build_action_tables(inputs, q, max_bytes=method.max_bytes, max_products=method.max_products)
        return inputs
    # Form [H,A_i] from the selected Hamiltonian and record dropped coefficient
    # mass separately from the generator symmetry corrections.
    updated = []
    table = processed.pauli_terms()
    # Bytes of the commutator tables built so far, 8J_i(q+1) each.
    held = 0
    # One running work ledger covers the commutator construction of every
    # pool member (_commutator_arrays).
    ledger = [0]

    if growth is None:
        for member, generator in zip(members, generators, strict=True):
            a = pauli_table(generator.pauli_terms, num_qubits=q, max_bytes=method.max_bytes)
            # d/dtheta <exp(theta A)psi|H|exp(theta A)psi> at theta = 0 is
            # <psi|H A - A H|psi> = <[H,A]>, because A^dagger = -A.
            commutator, removed = _commutator_arrays(
                table, a, cutoff=cutoff, ledger=ledger,
                held=held,
                max_bytes=method.max_bytes, max_products=method.max_products,
            )
            held += 8 * len(commutator.coefficients.array) * (q + 1)
            updated.append(
                member.revise(
                    commutator=commutator,
                    commutator_dropped_l1=removed,
                )
            )
        members = tuple(updated)
    if shots is None:
        return AdaptInputs(
            processed,
            tuple(generators),
            members,
            terms,
            (),
            (),
            dropped,
            tuple(qualifiers),
            work,
            symmetry,
        )

    # Group qubit-wise commuting labels so sampled queries share a valid basis.
    # This is the greedy QWC grouping, not the pool-commutator partitioning of
    # Anastasiou et al., arXiv:2306.03227v3, Sec. III.A-III.C.
    def groups(ordered):
        """Return first-fit QWC groups of the sorted distinct ``ordered`` labels as index tuples."""
        if not ordered:
            return ()
        table = pauli_table(((p, 1.0) for p in ordered), num_qubits=q, max_bytes=method.max_bytes)
        grouped = table.group(
            strategy="qwc", max_bytes=method.max_bytes, max_comparisons=method.max_products,
            limit_name="ADAPT.max_products",
        )
        return tuple(tuple(group) for group in grouped.groups)

    # The label index formed here serves the bound Plan's queries too.
    decoded = {}
    screen = screen_labels(members, decoded)
    energy = tuple(sorted(t.label for t in terms if t.label != "I" * q))
    screen_groups, energy_groups = groups(screen), groups(energy)
    inputs = AdaptInputs(
        processed,
        tuple(generators),
        members,
        terms,
        screen_groups,
        energy_groups,
        dropped,
        tuple(qualifiers),
        work,
        symmetry,
    )
    inputs.cache.update(decoded)
    return inputs
