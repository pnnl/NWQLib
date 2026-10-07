"""Finite ADAPT input processing and actual commutator selection."""

from dataclasses import dataclass, field
from functools import lru_cache
import json
from math import fsum
import math
from typing import Any
import numpy as np
from nwqlib._numerics import stable_complex_sum, stable_vector_norm
from nwqlib.core.records import Complex128, FrozenArray
from nwqlib.operators.inputs import OperatorInput, ingest_pauli, pauli_table, _matvec_requirements
from nwqlib.operators.access import _check_bytes, _check_products
from nwqlib.operators._fermion import mapping_requirements
from nwqlib.operators._pauli import combine_terms
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
from .adapt_records import AdaptPoolMember, AdaptSymmetry, PROCESSING_SOURCE
from .fixed_basis import PauliArrays, PauliTerm


def _symmetry_evidence(
    original: np.ndarray,
    correction: np.ndarray | None,
    *,
    tolerance: float,
    target: str,
    exact_symmetry: bool,
    pauli_qubits: int = 0,
) -> dict[str, Any]:
    """Record admission evidence, using ``correction=None`` for verified exact symmetry.

    The relative correction ``||correction||_F / ||original||_F`` is compared
    with ``tolerance`` through ``frexp`` mantissas and exponents, so neither
    overflow nor underflow of an intermediate can change the admission
    decision. For a Pauli sum on ``pauli_qubits`` qubits the absolute
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
        raise ValueError(f"input must be {target} within input_symmetry_tolerance")
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
        "input_symmetry_tolerance": tolerance,
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
    original = np.array([c for _, c in terms], dtype=complex)
    processed = 1j * original.imag if antihermitian else original.real.astype(complex)
    evidence = _symmetry_evidence(
        original,
        processed - original,
        tolerance=tolerance,
        target="anti-Hermitian" if antihermitian else "Hermitian",
        exact_symmetry=bool(np.array_equal(original, processed)),
        pauli_qubits=qubits,
    )
    return tuple(
        (p, complex(c)) for (p, _), c in zip(terms, processed, strict=True) if c != 0
    ), evidence


@dataclass(frozen=True)
class AdaptInputs:
    """Processed ADAPT inputs bound to a Plan and restored by archive loading.

    ``hamiltonian`` and ``pool`` are the processed operators that queries act
    on, and ``members`` keep each generator's raw identity, symmetry evidence
    and ``[H, A_i]`` terms. ``terms`` are the kept Hamiltonian Pauli terms and
    ``dropped_l1`` is the L1 mass of the pruned Hamiltonian coefficients. ``groups`` and
    ``energy_groups`` are the QWC readout groups for screening and energy
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


def array_record_bytes(raw_lengths, header_bounds, *, fixed_json,
                       index_entries=0, label_count=0, group_count=0,
                       other_live=0):
    """Admit the live array payloads, immutable snapshots, group indices and the selected identity encoding of the ADAPT reconstruction.

    Each array contributes its dtype-dependent byte length and dtype/shape
    header, including empty arrays. Fields stored as individual Pauli records
    are charged separately and enter here as other live bytes.

    Let ``R_j`` be the rows of array table j and ``b_j`` its raw bytes per
    row, ``B_j = R_j*b_j`` and ``B = sum_j B_j``. With binary arrays encoded
    as base64 strings (``FrozenArray``'s JSON form), one source/working array
    and one immutable snapshot at most, and ordinary record serialization
    holding the base64 strings, complete JSON text and UTF-8 bytes,

        J = J_fixed + sum_j [4 ceil(B_j/3) + H_j] + J_indices,
        B_record <= B_other_live + 2B + 4J + 16I,

    where ``J_fixed`` includes the enclosing reconstruction's remaining
    JSON fields and collection delimiters, including empty collections,
    as well as each array-bearing table record's fixed fields.
    ``H_j`` is an upper bound for that array field's actual
    dtype/shape/key JSON and ``J_indices <= I*(digits(max(U-1,0))+1)+2G+2``
    for ``I`` nonnegative indices into ``U`` labels stored in ``G`` nested
    tuples. Two raw copies contribute ``2B``. Three simultaneous
    JSON/base64 text forms and one transient encoding form are covered by
    ``4J``. The ``16I`` term covers two logical index sequences while
    immutable tuples are formed. The bound does not include Python object
    headers, so it is not a ``tracemalloc`` or RSS bound. Hashing canonical
    bytes incrementally, transferring an already immutable array, or
    streaming serialization can lower this peak, but requires the
    corresponding owner to establish the smaller schedule. A hex codec
    replaces the base64 length by ``2B_j``.
    """
    raw = sum(raw_lengths)
    digits = len(str(max(label_count - 1, 0)))
    indices_json = index_entries * (digits + 1) + 2 * group_count + 2
    encoded = fixed_json + indices_json + sum(
        4 * ((n + 2) // 3) + h
        for n, h in zip(raw_lengths, header_bounds, strict=True)
    )
    return other_live + 2 * raw + 4 * encoded + 16 * index_entries


def _array_header_bytes(key, dtype, shape):
    """Return the JSON length of an array field's key, dtype and shape with empty data.

    ``json.dumps`` writes the separators with spaces, so this is at least
    the compact record form, whose data string adds exactly the base64 length.
    """
    return len(json.dumps({key: {"dtype": dtype, "shape": list(shape), "data": ""}}))


@lru_cache(maxsize=None)
def _commutator_record_fixed_json(num_qubits):
    """Return the fixed JSON of one ``[H, A_i]`` table record beyond its two array headers.

    This is one table's share of ``J_fixed`` in ``array_record_bytes``, the
    actual fixed JSON including empty tables: the member's ``commutator``
    key and the ``PauliArrays`` record's own fields (schema version, parent
    identity and the sha256 content identity), measured on the empty table
    with the same spaced ``json.dumps`` as ``_array_header_bytes``. These
    fields do not depend on the row count. The two array headers, which do,
    are ``_array_header_bytes`` at the actual shapes.
    """
    empty = PauliArrays.from_rows((), num_qubits)
    return len(json.dumps({"commutator": empty.model_dump(mode="json")})) - (
        _array_header_bytes("labels", "<i8", (0, num_qubits))
        + _array_header_bytes("coefficients", "<f8", (0,))
    )


def _commutator_array_charges(tables, num_qubits):
    """Return the raw lengths, header bounds and fixed JSON of ``[H, A_i]`` tables with the given row counts.

    Each table is an int64 ``(rows, q)`` letter-code array and a float64
    ``(rows,)`` coefficient array, ``8q + 8`` raw bytes per row
    (``PauliArrays``), and its record adds
    ``_commutator_record_fixed_json(q)``.
    """
    raw, headers = [], []
    for rows in tables:
        raw += [8 * rows * num_qubits, 8 * rows]
        headers += [
            _array_header_bytes("labels", "<i8", (rows, num_qubits)),
            _array_header_bytes("coefficients", "<f8", (rows,)),
        ]
    return raw, headers, len(tables) * _commutator_record_fixed_json(num_qubits)


def label_cache_bytes(q, row_counts, pool_size, energy_count, max_selections,
                      *, sampled):
    """Return the qualified object-memory allowance of one Plan's decoded ADAPT label cache.

    The cache (``adapt_records.label_cache``) holds the rows decoded by
    ``commutator_rows`` (new strings, floats, row tuples, member tuples and
    an outer tuple), the sorted ``screen_labels`` and ``energy_labels``
    tuples, which reference those strings, and the active-screen entries
    (a ``frozenset(removed)`` key, a value pair, a sorted label tuple and a
    label frozenset). It is never serialized, so no base64 or JSON
    multiplier applies, and it is added once, as other live storage, to the
    reconstruction's array and encoding charge (``array_record_bytes``).
    This is a qualified CPython object-memory allowance on the checked
    64-bit CPython 3.12.14 build, not a bound on all Python heap allocations
    or process RSS.

    Dimensions: q Pauli characters, K = ``pool_size`` members, R and r_max
    the sum and maximum of ``row_counts``, J = min(max_selections, K), and
    L = ``energy_count`` nonidentity Hamiltonian labels. R bounds the
    distinct commutator labels even when several rows share a label. The
    active-cache capacity A is J + 1 for exact plans and K + 1 for sampled
    plans: clearing before insertion only when the stored count exceeds
    the limit permits limit + 1 entries.

    Engineering envelopes are ``T(n) = 64 + 8n`` for a tuple,
    ``H(n) = 256 + 128n`` for a set, frozenset or active dictionary, and
    ``I_K = 32 + 4 ceil(max(1, bit_length(K))/30)`` for a selected pool
    index. A decoded row costs q + 168 bytes, including its label, float,
    row tuple and member-tuple slot. The kept cache and construction
    allowances are

        B_kept = 4096 + T(K) + 64K + (q + 168)R + T(R) + T(L) + H(A)
                 + A[T(R) + H(R) + H(J) + T(2) + J I_K],
        X_decode = r_max(8q + 128) + 16q,
        X_build = 320R + 32L + (160 + I_K)J + 128A,
        B_cache = B_kept + 65536 + max(X_decode, X_build).

    ``X_decode`` prices ``PauliArrays.rows()``'s member-sized ``tolist``
    lists, builder capacity and a label join. ``X_build`` prices set
    formation, sorting, tuple and frozenset construction, key construction
    and dictionary growth. The 65,536 bytes cover fixed Python and NumPy
    call and iterator bookkeeping. docs/ENGINEERING_CONSTANTS.md lists this
    allowance and its revisit condition.
    """
    counts = tuple(row_counts)
    K, R, rmax = pool_size, sum(counts), max(counts, default=0)
    J = min(max_selections, K)
    A = (K if sampled else J) + 1
    if min(q, K, energy_count, max_selections, *counts) < 0 or q < 1 or len(counts) > K:
        raise ValueError("invalid cache dimensions")
    # CPython 3.12, 64-bit engineering envelopes, including variable headers.
    integer = 32 + 4 * ((max(1, K.bit_length()) + 29) // 30)
    tup = lambda n: 64 + 8 * n  # noqa: E731
    hashed = lambda n: 256 + 128 * n  # noqa: E731
    kept = (4096 + tup(K) + 64 * K + (q + 168) * R
            + tup(R) + tup(energy_count) + hashed(A)
            + A * (tup(R) + hashed(R) + hashed(J) + tup(2) + J * integer))
    decode = rmax * (8 * q + 128) + 16 * q
    build = 320 * R + 32 * energy_count + (160 + integer) * J + 128 * A
    return kept + 65536 + max(decode, build)


def restored_cache_reservation(rec, *, sampled):
    """Return the count-based reservation a restored quantum Plan passes before its first cache decode.

    It is reconstructed from the stored commutator row shapes, members and
    group sizes of ``rec``, with no rerun of commutators, chemistry or
    selection: the commutator tables' array and encoding charge
    (``array_record_bytes`` with the table records' fixed JSON) plus the
    decoded label cache (``label_cache_bytes``) as other live storage, and
    the group-index charges of a sampled Plan. The pool-member, Pauli-row
    and reconstruction-metadata allowances that preparation also charged
    are not reconstructed here.
    """
    q = rec.num_qubits
    kept_rows = [len(member.commutator.coefficients.array) for member in rec.pool
                 if member.commutator is not None]
    raw, headers, tables = _commutator_array_charges(kept_rows, q)
    cache = label_cache_bytes(
        q, kept_rows, len(rec.pool), sum(t.label != "I" * q for t in rec.terms),
        rec.max_selections, sampled=sampled,
    )
    total = array_record_bytes(raw, headers, fixed_json=tables, other_live=cache)
    for grouped in (rec.groups, rec.energy_groups):
        total += array_record_bytes(
            (), (), fixed_json=0, index_entries=sum(map(len, grouped)),
            label_count=sum(map(len, grouped)), group_count=len(grouped),
        )
    return total


def _reconstruction_fixed_json(fields, *, pool_size, term_count):
    """Bound the fresh reconstruction's JSON outside pool, term and group payloads.

    ``fields`` contains its actual remaining constructor fields. Record
    metadata is serialized once, including arbitrary supplied text. The
    empty lists supply their keys and brackets. Spaced list separators and
    the Record identity wrapper are added without visiting their payloads.
    ASCII escaping bounds both the compact export and its UTF-8 encoding.
    """
    from .adapt_records import AdaptReconstruction

    shell = {
        **fields,
        "schema_version": AdaptReconstruction.model_fields["schema_version"].default,
        "parent_id": None,
        "content_id": "sha256:" + "0" * 64,
        "pool": [],
        "terms": [],
        "groups": [],
        "energy_groups": [],
    }
    encoder = json.JSONEncoder(
        ensure_ascii=True,
        allow_nan=False,
        default=lambda record: record.model_dump(mode="json"),
    )
    shell_chars = sum(len(part) for part in encoder.iterencode(shell))
    wrapper = {
        "identity_encoding": "nwqlib.record/1",
        "type": f"{AdaptReconstruction.__module__}.{AdaptReconstruction.__qualname__}",
        "fields": {},
    }
    wrapper_chars = sum(len(part) for part in encoder.iterencode(wrapper)) - 2
    separators = 2 * (max(pool_size - 1, 0) + max(term_count - 1, 0))
    return shell_chars + wrapper_chars + separators


# Candidate Pauli pairs tested per commutator tile. An engineering choice of
# tile length, not a scientific threshold. Changing it changes storage, not
# rows, order or coefficients.
_COMMUTATOR_TILE = 1 << 14


def pauli_array_conversion_requirements(rows, q, *, tile=1024):
    """Return ``(work, peak bytes, raw bytes)`` of converting J coalesced packed rows to ``PauliArrays``.

    Here J = ``rows``, q is the number of qubits, W = ceil(q/64) is the
    number of 64-bit words per mask, and C = ``tile``.
    With ``P = (16W+16)J``, ``B = 8J(q+1)``, ``t = min(J, C)`` and
    ``h = ceil(log2 max(1, J))``, the live phases are decoding (packed input
    P, code and coefficient outputs B, two uint64 tile vectors 16t), sorting
    and permuting (P, unsorted and sorted outputs 2B, indirect permutation
    8J) and immutable publication and validation (P, working outputs and
    snapshots 2B, validator scratch at most ``J*q + 40J``), so

        B_conv = P + max{B + 16t, 2B + 8J, 2B + Jq + 40J} + 65536.

    One work unit is one scalar code, word or index visit, with an inspected
    byte of a row-key comparison counted as one unit. The decoder has eight
    vector operations per character and a coefficient pass, and heap
    construction and removal have fewer than 4Jh key comparisons of at most
    8q bytes and 4Jh index moves, so

        W_conv = J(16q + 32) + (32q + 8)Jh.

    These are defined accounting units, not literal NumPy instruction counts.
    ``65536`` is the CPython/NumPy engineering allowance for fixed overhead,
    not an arbitrary-allocator theorem.
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
    """Publish coalesced packed rows with real coefficients as a sorted ``PauliArrays`` record.

    Each printed character reads its bit from high qubit to low qubit and
    stores the int64 code ``c = (x_bit XOR z_bit) + 2*z_bit``: the pairs
    ``(x, z) = (0, 0), (1, 0), (1, 1), (0, 1)`` give 0, 1, 2, 3, that is
    I, X, Y, Z. Rows are then sorted lexicographically with their
    coefficients by a heapsort of fixed-size row keys, whose bytes compare
    in code-row order because every code is one of 0 to 3. The input rows
    are distinct canonical keys, so sort stability is irrelevant here. No
    label string is materialized. The caller admits
    ``pauli_array_conversion_requirements`` before this call.
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


def _commutator_arrays(h, a, *, cutoff, ledger, held, max_bytes, max_products, admit_rows):
    """Return ``[H, A]`` as ``PauliArrays`` and its dropped coefficient mass, from packed tables.

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
    bilinearity. In the accepted ADAPT route H has real coefficients and A
    purely imaginary ones, so each contribution is real. The Hermiticity
    check and ``commutator_dropped_l1`` stay at the coalesced output. Empty
    H or A, commuting pairs, identity strings and complete cancellation all
    give zero without an invented residual. ``stable_complex_sum`` uses
    ``fsum``, not arbitrary-precision addition. Changing multiplication,
    doubling and accumulation order relative to the product-and-cancel
    route can change the last bits or an intermediate overflow decision, so
    agreement holds where both constructions' finite arithmetic premises
    hold. Finite pair products and coalesced sums are required, and no
    individual pair contribution is pruned (Zheng et al.,
    arXiv:2312.07691v3, commutator screen; Higham, Accuracy and Stability of
    Numerical Algorithms, 2nd ed., doi:10.1137/1.9780898718027).

    Commutator construction tests bounded packed pair tiles, stores only
    anticommuting products and coalesces the complete survivor population
    before publication. One work ledger covers every pool member, and byte
    admission includes completed member tables, the current survivor
    population and its sort/snapshot workspace. With L rows of H, T rows of
    A, ``P = L*T`` tested pairs, C anticommuting survivors, J coalesced
    rows, W packed words and tiles of at most b pairs (the pair index is
    flattened, so a tile need not hold a whole row),

        B_tile = (64W + 128)b,    B_coalesce(C) = (80W + 128)C + 65536,
        W_j = P(W + 1) + C(12W + 16) + (2W + 2)C ceil(log2 max(2, C)).

    ``B_coalesce`` reserves five populations of packed X/Z data (raw,
    sortable keys, gathered keys, output and output snapshot) and 128C for
    complex coefficients, their gathered and output copies, sort, inverse
    and first-position indices, head masks and group order. ``ledger`` is
    the one running work total shared by all pool members, a one-element
    list. ``admit_rows(J)`` runs the caller's record admission
    for the J published rows before their conversion. Each pair tile is
    admitted before it is tested, each tile's
    worst-case survivors are reserved before they are appended, and the sort
    and output stages are admitted once C is known. ``held`` is the byte
    count already live, including earlier published tables ``8J_i(q+1)``.
    The packed survivors then convert to the published int64 code table
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
    ends = np.append(starts[1:], survivors)
    values = [stable_complex_sum(c.real[s:e], c.imag[s:e]) for s, e in zip(starts.tolist(), ends.tolist())]
    if any(value.imag != 0 for value in values):
        raise ValueError("[H,A] must be Hermitian after coalescing its surviving pairs")
    real = np.array([value.real for value in values], dtype=np.float64)
    removed = fsum(abs(value) for value in real.tolist() if abs(value) <= cutoff)
    kept = np.abs(real) > cutoff
    x, z, real = x[starts][kept], z[starts][kept], real[kept]
    del c, starts, ends, values, kept
    admit_rows(len(real))
    work, peak, _ = pauli_array_conversion_requirements(len(real), q)
    charge(work)
    reserve(peak)
    return _packed_to_pauli_arrays(x, z, real, q), removed


def _pool_enumeration_requirements(candidates, num_qubits):
    """Return ``(bytes, work)`` for enumerating ``candidates`` built-in pool members on ``num_qubits`` qubits.

    Every built-in member has at most 12 normal-ordered ladder strings of at
    most four operators and at most 48 coalesced Pauli rows. A triplet double
    of Zheng et al., arXiv:2312.07691v3, Eq. (E3), on four distinct orbitals
    reaches both. Its six base terms and their adjoints are 12 strings on six
    distinct four-mode sets, and the Jordan-Wigner image of the
    anti-Hermitian pair on one set has 8 Pauli rows.
    ``test_pool_enumeration_law_covers_every_builtin_member`` measures the
    bytes, strings and comparisons of every member of the four families and
    checks that this law covers them.

    Bytes. The enumerated pool stays alive while each member is admitted, so
    each candidate is charged what ``_snapshot_generator`` charges for the
    largest member: ``q + 64`` per Pauli row, ``48 + 48 * 4 = 240`` per
    four-operator string, and 206 for the family name and index tuples
    (64 plus the at most 14 characters of a built-in family name, and 16
    per index of at most four spatial and four spin-orbital indices). The
    mapping arrays of the one member being built, the byte term of
    ``mapping_requirements`` for its 12 strings, are added once.

    Work. Per candidate, the work term of ``mapping_requirements`` for 12
    four-operator strings with label output, the law whose byte term
    ``_ladder_expansion_terms`` checks for each member, and 168 comparisons
    for normal ordering. A spin-adapted candidate is normal-ordered twice,
    in ``_skip_zero_base`` and in ``_generator``, with at most 84
    comparisons each time. The count depends only on the order pattern of
    the spatial indices, and every pattern of four indices occurs with four
    spatial orbitals, so the test's exhaustive count there is the maximum.
    """
    q = num_qubits
    _, mapping_bytes, mapping_work = mapping_requirements((4,) * 12, num_modes=q, labels=True)
    member_bytes = 48 * (q + 64) + 12 * 240 + 206
    return candidates * member_bytes + mapping_bytes, candidates * (mapping_work + 168)


def _generator_record_bytes(generators):
    """Return the generator-record term of the ADAPT reconstruction record law.

    Planning (``prepare_adapt_inputs``) and the classical archive reload
    (``adapt_actions.restored_record_allowance``) both charge this term.
    """
    return 3 * sum(2048 + 128 * len(generator.fermion_terms)
                   + 16 * sum(len(ops) for _, ops in generator.fermion_terms)
                   + 16 * (len(generator.spatial_indices) + len(generator.spin_orbital_indices or ()))
                   for generator in generators)


def prepare_adapt_inputs(hamiltonian, reference, method, *, execution, shots,
                         original_dimension, original_hamiltonian, reference_requirements):
    """Admit the finite generator pool and select Hamiltonian/commutator readout terms.

    A named pool is enumerated only after its candidate count passes the byte
    and product caps. Every generator is projected onto its anti-Hermitian
    part within ``input_symmetry_tolerance`` and must agree with its stored
    fermion terms. A Hamiltonian with Pauli access, which every quantum plan
    has, must have exactly real Pauli coefficients. Coefficients at or below
    ``pauli_coefficient_cutoff`` are removed with their L1 mass recorded.
    Only a quantum plan forms each ``[H, A_i]``, whose labels its screening
    queries read, and only a quantum plan with ``shots`` forms the QWC
    readout groups. A classical plan screens with generator actions on the
    state and forms neither. A classical plan with
    dense or sparse access keeps the original matrix and forms no Pauli
    decomposition. Its pair queries then remove the mean diagonal
    ``c = trace(H)/d`` from the matrix before they project, as classical
    FixedGCIM does, so the projected entries and their rounding do not grow
    with an identity offset of H. The solve adds c back.

    Returns:
        The ``AdaptInputs`` bound to the Plan.
    """
    q = reference.basis.dimension.bit_length() - 1
    pool = method.pool
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
    if len(pool) > method.max_pool_size:
        raise ValueError(f"ADAPT pool has {len(pool)} members, more than max_pool_size={method.max_pool_size}")
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
                rows = generator.pauli_terms().labels(max_bytes=method.max_bytes)
            else:
                from nwqlib.operators._pauli import pauli_coefficients

                from nwqlib.algorithms._eigen_inputs import dense_pauli_matrix

                matrix = dense_pauli_matrix(
                    generator,
                    input_conversion=method.input_conversion,
                    max_conversion_work=method.max_conversion_work,
                    max_bytes=method.max_bytes,
                )
                rows = pauli_coefficients(matrix)
            family, spatial, spin, fermions = "custom", (), None, ()
        elif isinstance(generator, FermionicGenerator):
            generator = _snapshot_generator(
                generator, max_bytes=method.max_bytes, max_products=method.max_products
            )
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
            original = _generator_reference(
                generator, max_bytes=method.max_bytes, max_products=method.max_products
            )
        else:
            raise TypeError("pool entries must be Pauli OperatorInput or FermionicGenerator")
        rows, evidence = _project_terms(
            rows, antihermitian=True, tolerance=method.input_symmetry_tolerance, qubits=q
        )
        processed = ingest_pauli(rows, num_qubits=q, max_bytes=method.max_bytes)
        if fermions:
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
    # The reconstruction records the Hamiltonian terms and every member's
    # terms as Pauli rows. Each of these is charged the identity JSON share
    # of operators/inputs.py::_pauli_identity_bytes in a running total,
    # because Pauli ingestion admits one operator at a time. Each member's
    # other fields are charged three copies of the sum of 2048 bytes for its
    # input identities, symmetry evidence and processing note (at most about
    # 1.5 KB of JSON with 24-character floats), 128 bytes per fermionic term
    # and 16 per ladder entry or orbital index. The reconstruction's own
    # fields are measured once for each route, with four JSON copies,
    # including supplied preparation metadata. The pool is charged before the
    # classical matrix route returns. A quantum plan's commutator tables and
    # a sampled plan's group indices are arrays and index tuples, charged by
    # array_record_bytes with these Pauli-row charges as its other live
    # bytes and the reconstruction and table records' own fields as fixed JSON
    # (_reconstruction_fixed_json and _commutator_record_fixed_json), each table before its
    # record is built.
    def reconstruction_json(processed, term_count, work, dropped, notes, symmetry,
                            *, shift=0.0, offset=(0, 0)):
        return _reconstruction_fixed_json(
            {
                "reference": reference.preparation,
                "original_hamiltonian": original_hamiltonian,
                "processed_hamiltonian": processed.reference,
                "pool_name": method.pool if isinstance(method.pool, str) else "explicit",
                "max_selections": min(method.max_iterations, len(generators)),
                "num_qubits": q,
                "hamiltonian_action_work": work,
                "hamiltonian_dropped_l1": dropped,
                "processing": notes,
                "hamiltonian_symmetry": symmetry,
                "processing_source": PROCESSING_SOURCE,
                "reference_action_bytes": reference_requirements[0],
                "reference_action_work": reference_requirements[1],
                "host_identity_shift": shift,
                "offset_free_action_bytes": offset[0],
                "offset_free_action_work": offset[1],
            },
            pool_size=len(generators),
            term_count=term_count,
        )

    from nwqlib.operators.inputs import _pauli_identity_bytes
    share = _pauli_identity_bytes(q)
    fixed = _generator_record_bytes(generators)
    records = sum(len(generator.pauli_terms) for generator in generators)
    _check_bytes(fixed + records * share, method.max_bytes, "ADAPT reconstruction records")
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
        from .fixed_basis import _mean_diagonal, _offset_free_requirements

        shift = _mean_diagonal(hamiltonian, max_bytes=method.max_bytes)
        offset = _offset_free_requirements(hamiltonian, 1, shift)
        work = _matvec_requirements(hamiltonian)[2]
        symmetry = AdaptSymmetry(**evidence)
        notes = tuple(qualifiers) + (
            "exact original Hermitian matrix; no Pauli decomposition or pruning",
        )
        record_fixed = reconstruction_json(
            hamiltonian, 0, work, 0.0, notes, symmetry, shift=shift, offset=offset,
        )
        _check_bytes(fixed + records * share + 4 * record_fixed, method.max_bytes,
                     "ADAPT reconstruction records")
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
        build_action_tables(inputs, q, max_bytes=method.max_bytes, max_products=method.max_products,
                            other_live=fixed + records * share + 4 * record_fixed)
        return inputs
    raw = hamiltonian.pauli_terms().labels(max_bytes=method.max_bytes)
    rows, evidence = _project_terms(raw, antihermitian=False, tolerance=0.0, qubits=q)
    dropped = fsum(abs(c.real) for _, c in rows if abs(c.real) <= method.pauli_coefficient_cutoff)
    rows = tuple((p, c.real) for p, c in rows if abs(c.real) > method.pauli_coefficient_cutoff)
    processed = ingest_pauli(rows, num_qubits=q, max_bytes=method.max_bytes)
    terms = tuple(PauliTerm(label=p, coefficient=c) for p, c in rows)
    qualifiers.append(
        f"Hamiltonian Pauli cutoff {method.pauli_coefficient_cutoff}; dropped L1 {dropped}"
    )
    records += len(terms)
    work = _matvec_requirements(processed)[2]
    symmetry = AdaptSymmetry(**evidence)
    record_fixed = reconstruction_json(
        processed, len(terms), work, dropped, tuple(qualifiers), symmetry,
    )
    _check_bytes(fixed + records * share + 4 * record_fixed, method.max_bytes,
                 "ADAPT reconstruction records")
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
        build_action_tables(inputs, q, max_bytes=method.max_bytes, max_products=method.max_products,
                            other_live=fixed + records * share + 4 * record_fixed)
        return inputs
    # Form [H,A_i] from the selected Hamiltonian and record dropped coefficient
    # mass separately from the generator symmetry corrections.
    updated = []
    table = processed.pauli_terms()
    kept_rows = []
    # One running work ledger covers the commutator construction of every
    # pool member (_commutator_arrays).
    ledger = [0]
    # The decoded label cache (label_cache_bytes) is charged with the full
    # pool size and selection count while tables are added, so each record
    # check below reserves the cache the finished Plan can build.
    energy_count = sum(t.label != "I" * q for t in terms)
    selections = min(method.max_iterations, len(generators))

    def cache_bytes(rows):
        return label_cache_bytes(q, rows, len(generators), energy_count, selections,
                                 sampled=shots is not None)

    def admit_tables(rows):
        raw, headers, tables = _commutator_array_charges(rows, q)
        _check_bytes(
            array_record_bytes(raw, headers, fixed_json=tables + record_fixed,
                               other_live=fixed + records * share + cache_bytes(rows)),
            method.max_bytes,
            "ADAPT reconstruction records and decoded label cache",
        )

    for member, generator in zip(members, generators, strict=True):
        a = pauli_table(generator.pauli_terms, num_qubits=q, max_bytes=method.max_bytes)
        # d/dtheta <exp(theta A)psi|H|exp(theta A)psi> at theta = 0 is
        # <psi|H A - A H|psi> = <[H,A]>, because A^dagger = -A.
        commutator, removed = _commutator_arrays(
            table, a, cutoff=method.pauli_coefficient_cutoff, ledger=ledger,
            held=fixed + records * share + sum(8 * rows * (q + 1) for rows in kept_rows),
            max_bytes=method.max_bytes, max_products=method.max_products,
            admit_rows=lambda rows: admit_tables([*kept_rows, rows]),
        )
        kept_rows.append(len(commutator.coefficients.array))
        updated.append(
            member.revise(
                commutator=commutator,
                commutator_dropped_l1=removed,
            )
        )
    if not generators:
        admit_tables(kept_rows)
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

    screen = tuple(sorted({p for member in members for p, _ in member.commutator.rows()}))
    energy = tuple(sorted(t.label for t in terms if t.label != "I" * q))
    screen_groups, energy_groups = groups(screen), groups(energy)
    raw, headers, tables = _commutator_array_charges(kept_rows, q)
    total = array_record_bytes(raw, headers, fixed_json=tables + record_fixed,
                               other_live=fixed + records * share + cache_bytes(kept_rows))
    for ordered, grouped in ((screen, screen_groups), (energy, energy_groups)):
        total += array_record_bytes(
            (), (), fixed_json=0, index_entries=sum(map(len, grouped)),
            label_count=len(ordered), group_count=len(grouped),
        )
    _check_bytes(total, method.max_bytes, "ADAPT reconstruction records")
    return AdaptInputs(
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
