"""SDK-free coefficient accumulation, packed Pauli kernels and tiled little-endian Pauli action."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from nwqlib._numerics import stable_complex_sum
from .access import DEFAULT_INPUT_BYTES, _check_bytes

# Tile length C of the label decoder, the product and grouping candidate
# blocks and the coordinate tiles of the action. The default C = 1024 is an
# engineering choice, not a scientific threshold. Changing C changes
# storage, not labels, order or scientific records.
_PAULI_TILE = 1024
# i**e for e = 0..3, the same binary64 values the scalar product used.
_PHASES = np.array([1j ** power for power in range(4)], dtype=np.complex128)


def _half_sum(left, right):
    """Finite componentwise average without overflowing or halving tiny peers."""
    result = np.empty_like(left)
    safe = np.maximum(np.abs(left), np.abs(right)) <= np.finfo(float).max/2
    result[safe] = (left[safe]+right[safe])*.5
    result[~safe] = left[~safe]*.5+right[~safe]*.5
    return result


def pauli_coefficients(matrix):
    """SDK-free I/X/Y/Z block transform of an already admitted dense matrix.

    Labels proceed from the most significant qubit to the least significant.
    This function checks no limit: each caller admits the transform's bytes
    and work under its own law before entry. Only exact zero coefficients
    are omitted. The caller applies its own explicit pruning policy
    afterward.

    Returns the coefficients ``c_P = tr(P M)/D`` of ``M = sum_P c_P P``.
    Splitting M on its most significant qubit into blocks
    ``[[a, b], [c, d]]`` gives ``M = I (x) (a+d)/2 + X (x) (b+c)/2 +
    Y (x) i(b-c)/2 + Z (x) (a-d)/2``, applied recursively to the half-size
    blocks. All blocks at one level share a vectorized pass, avoiding a
    Python call for each of the 4**q leaves. The two active levels occupy
    O(D²) storage and the arithmetic is O(q*D²). Each component average uses
    ``_half_sum`` so finite entries near the binary64 maximum do not overflow.
    """
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] < 1 or matrix.shape[0] & (matrix.shape[0]-1)):
        raise ValueError("Pauli block transform requires a square power-of-two matrix")
    width = matrix.shape[0].bit_length()-1
    if not np.any(matrix != 0):
        return ()
    blocks = matrix[None, :, :]
    for _ in range(width):
        half = blocks.shape[-1]//2
        a, b = blocks[:, :half, :half], blocks[:, :half, half:]
        c, d = blocks[:, half:, :half], blocks[:, half:, half:]
        next_level = np.empty((len(blocks), 4, half, half), dtype=complex)
        for component in ("real", "imag"):
            aa, bb, cc, dd = (getattr(block, component) for block in (a, b, c, d))
            output = getattr(next_level, component)
            output[:, 0] = _half_sum(aa, dd)
            output[:, 1] = _half_sum(bb, cc)
            output[:, 3] = _half_sum(aa, -dd)
        # i(b-c)/2 rotates real and imaginary parts without a complex product.
        next_level[:, 2].real = _half_sum(c.imag, -b.imag)
        next_level[:, 2].imag = _half_sum(b.real, -c.real)
        blocks = next_level.reshape(-1, half, half)
    coefficients = blocks.reshape(-1)
    if not np.isfinite(coefficients).all():
        raise ValueError("Pauli block transform produced a nonfinite coefficient")
    # Half-label tables need O(q*D) characters. Joining two halves avoids
    # decoding q base-four digits in Python for each of the possible D² words.
    low_width = width//2
    low = tuple(map("".join, product("IXYZ", repeat=low_width)))
    high = low if width % 2 == 0 else tuple(map("".join, product("IXYZ", repeat=width-low_width)))
    low_bits, mask = 2*low_width, (1 << (2*low_width))-1
    return tuple(
        (high[int(index) >> low_bits]+low[int(index) & mask], complex(coefficients[index]))
        for index in np.flatnonzero(coefficients != 0)
    )



def pc(words):
    """Population count summed over the final packed uint64-word axis, as int64.

    ``np.bitwise_count`` returns uint8 per word. The sum over all W words is
    accumulated in int64 before any subtraction, so multiword sums include
    every word and signed differences cannot wrap. Padding bits above the
    qubit width are zero by admission.
    """
    return np.bitwise_count(words).sum(axis=-1, dtype=np.int64)


def product_words(xa, za, xb, zb):
    """Ordered product ``P_a P_b = i**e P(xa ^ xb, za ^ zb)`` of packed rows.

    Inputs broadcast on all but their final uint64-word axis. Bit j of a
    mask describes qubit j, the rightmost printed letter; x_j is set for
    X/Y and z_j for Z/Y. With ``y(x, z) = popcount(x & z)`` a row denotes
    ``P(x, z) = i**y X**x Z**z`` and ``P(x, z)|j> = i**y
    (-1)**popcount(j & z) |j XOR x>``. Moving the left Z factors through the
    right X factors gives ``e = y_a + y_b + 2*popcount(za & xb)
    - popcount((xa ^ xb) & (za ^ zb))`` modulo four, where the last term
    converts ``X**x Z**z = i**(-popcount(x & z)) P(x, z)``. Popcounts are
    added over all packed words before the reduction. The orientation
    matters: XZ = -iY, ZX = iY, XY = iZ and YX = -iZ.

    Returns:
        (x, z, phase): product masks and the complex128 factors ``i**e``,
        the same binary64 values as the Python powers ``1j**e``.
    """
    x, z = xa ^ xb, za ^ zb
    exponent = (pc(xa & za) + pc(xb & zb) + 2 * pc(za & xb) - pc(x & z)) & 3
    return x, z, _PHASES[exponent]


def anticommutes(xa, za, xb, zb):
    """Whether packed rows anticommute: odd ``popcount(xa & zb) + popcount(za & xb)``."""
    return ((pc(xa & zb) + pc(za & xb)) & 1).astype(bool)


def qwc(xa, za, xb, zb):
    """Whether packed rows are qubit-wise compatible, over the final word axis.

    With ``overlap = (xa | za) & (xb | zb)`` the rows are QWC exactly when
    every packed word of ``overlap & ((xa ^ xb) | (za ^ zb))`` is zero. For a
    QWC group, the OR of its member masks is its accumulated basis, and a
    new term need only pass this test against that basis.
    """
    overlap = (xa | za) & (xb | zb)
    return ~np.any(overlap & ((xa ^ xb) | (za ^ zb)), axis=-1)


def _product_tile(left, right):
    """Candidate block length C = min(max(M, K), 1024) of an M-by-K Pauli product, 0 when empty."""
    return min(max(left, right), _PAULI_TILE) if left * right else 0


def pauli_product_requirements(left, right, words):
    """Bytes that PauliTerms.product admits for an ordered product of M = left and K = right rows.

    With N = MK output terms, W = words packed words per mask and the block
    length ``C = min(max(M, K), 1024)``, the output plus its immutable
    snapshot take ``2N(16W+16)`` bytes, and a conservative explicit scratch
    envelope of ``64CW + 128C`` bytes covers two result masks,
    intermediate masks and popcounts, signed exponent vectors and
    coefficient temporaries of one candidate block, released before the
    next. The reservation is ``2N(16W+16) + 64CW + 128C``; an empty product
    reserves 0. PauliTerms.product checks this law, and
    FactorizedOperatorProduct.expansion_requirements charges it for each
    Pauli chain step, so the two cannot differ.
    """
    count = left * right
    tile = _product_tile(left, right)
    return 2 * count * (16 * words + 16) + 64 * tile * words + 128 * tile


@dataclass(frozen=True, eq=False, init=False, slots=True)
class PauliTerms:
    """Immutable native table; word j covers qubits 64*j through 64*j+63.

    Raw layout is M*(16*W+16) bytes, W=ceil(q/64). Python object overhead
    is outside logical byte limits. Product tables keep pair order and
    duplicates, so a later sum can accumulate all contributions only once.

    Row ``(x, z)`` denotes the word ``i**popcount(x & z) X**x Z**z``, so a
    qubit with both bits set carries Y itself and the stored coefficient
    multiplies the canonical I/X/Y/Z label.

    The packed fields are the public access for packed consumers, which
    should not call ``labels()``. The module-level kernels ``pc``,
    ``product_words``, ``anticommutes`` and ``qwc`` operate on rows of
    ``x`` and ``z`` and broadcast over leading axes; ``apply_terms`` acts
    with a single-word table on a state or a block of state columns.

    Attributes:
        num_qubits: Positive physical width, with qubit zero rightmost.
        x: Native uint64 flip words, shape (M,W).
        z: Native uint64 phase words, shape (M,W).
        coefficients: Native complex128 values, shape (M,).
    """

    num_qubits: int
    x: np.ndarray
    z: np.ndarray
    coefficients: np.ndarray

    def __init__(self, *args, **kwargs):
        raise TypeError("Pauli tables require bounded input admission")

    @classmethod
    def _snapshot(cls, num_qubits, x, z, coefficients):
        """Build a table from read-only byte copies, so later changes to the caller's arrays cannot reach it."""
        instance = object.__new__(cls)
        object.__setattr__(instance, "num_qubits", num_qubits)
        for name, array in (("x", x), ("z", z), ("coefficients", coefficients)):
            frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
            object.__setattr__(instance, name, frozen)
        return instance

    def __len__(self):
        return len(self.coefficients)

    def labels(self, *, max_bytes=DEFAULT_INPUT_BYTES):
        """Materialize labels in stored term order using tiled ASCII columns.

        In addition to the returned logical payload ``M*(q+16)``, a tile of t
        rows reserves ``t*(q+25)+q+4`` bytes on a 64-bit host for the
        character table, word/code buffers and one transient encoded row.
        Python object headers are outside this logical byte law.

        A tile of ``t = min(M, C)`` rows holds one (t, q) uint8 character
        table, two t-entry uint64 word buffers, one t-entry intp code buffer
        and one t-entry uint8 output column, ``tq + 16t + 8t + t = t(q+25)``
        bytes with an eight-byte intp. The alphabet takes four bytes and one
        row converted to ``bytes`` adds q transient bytes, so the admission
        is ``M(q+16) + t(q+25) + q + 4`` for M > 0. The tile is chosen from
        the bytes that remain after the payload, before the first tile is
        allocated. An empty table returns ``()`` and charges nothing. The
        code ``x_bit + 2*z_bit`` indexes ``b"IXZY"`` (I, X, Z, Y for 0..3)
        and bits are emitted from q-1 down to 0, so qubit zero stays the
        rightmost letter. Decoding visits M*(q+1) entries. The returned
        tuple is caller-owned, never cached by the input.
        """
        m, q = len(self), self.num_qubits
        if m == 0:
            return ()
        fixed = m * (q + 16) + q + 4
        tile = min(m, _PAULI_TILE)
        if type(max_bytes) is int:
            tile = max(1, min(tile, (max_bytes - fixed) // (q + 25)))
        _check_bytes(fixed + tile * (q + 25), max_bytes, "Pauli labels")
        alphabet = np.frombuffer(b"IXZY", dtype=np.uint8)
        result = []
        for start in range(0, m, tile):
            stop = min(m, start + tile)
            t = stop - start
            letters = np.empty((t, q), dtype=np.uint8)
            xx = np.empty(t, dtype=np.uint64)
            zz = np.empty(t, dtype=np.uint64)
            index = np.empty(t, dtype=np.intp)
            char = np.empty(t, dtype=np.uint8)
            for col, bit in enumerate(range(q - 1, -1, -1)):
                np.right_shift(self.x[start:stop, bit // 64], np.uint64(bit % 64), out=xx)
                np.bitwise_and(xx, np.uint64(1), out=xx)
                np.right_shift(self.z[start:stop, bit // 64], np.uint64(bit % 64), out=zz)
                np.bitwise_and(zz, np.uint64(1), out=zz)
                np.left_shift(zz, np.uint64(1), out=zz)
                np.bitwise_or(xx, zz, out=xx)
                np.copyto(index, xx, casting="unsafe")
                # Every index is in 0..3, so clip leaves it unchanged and
                # avoids take's exception-buffering route.
                np.take(alphabet, index, out=char, mode="clip")
                letters[:, col] = char
            for row, coefficient in zip(letters, self.coefficients[start:stop], strict=True):
                result.append((row.tobytes().decode("ascii"), complex(coefficient)))
            del letters, xx, zz, index, char, row
        return tuple(result)

    def product(self, other, *, max_bytes=DEFAULT_INPUT_BYTES):
        """Ordered P*Q, without pruning or intermediate bucket accumulation.

        A packed row denotes ``i**popcount(x & z) X**x Z**z``. Ordered
        multiplication XORs the masks and multiplies the coefficients by
        ``i**e``, where ``e = popcount(xa & za) + popcount(xb & zb) +
        2*popcount(za & xb) - popcount((xa ^ xb) & (za ^ zb))`` modulo four.
        Popcounts include every packed word (``product_words``).

        Output row ``i*K+j`` holds left row i times right row j, without
        combining duplicate products. One row of the shorter table is
        multiplied by blocks of at most C rows of the longer one, with
        ``C = min(max(M, K), 1024)`` (``_PAULI_TILE``), and the products of
        a block are written to their rows ``i*K+j``. N=M*K output terms are
        admitted before enumeration with ``pauli_product_requirements(M, K,
        W)``, the byte law stated there. Work law N*(W+1) counts word-pair
        visits and coefficients (constant arithmetic per visit), not CPU
        instructions. Each coefficient is ``(c_i*c_j)*1j**e`` in that order,
        as a scalar loop would form it.
        """
        if not isinstance(other, PauliTerms) or self.num_qubits != other.num_qubits:
            raise ValueError("Pauli product requires equal qubit widths")
        w, m, k = self.x.shape[1], len(self), len(other)
        count = m * k
        tile = _product_tile(m, k)
        _check_bytes(pauli_product_requirements(m, k, w), max_bytes, "Pauli product")
        x, z = np.empty((count, w), dtype=np.uint64), np.empty((count, w), dtype=np.uint64)
        coefficients = np.empty(count, dtype=np.complex128)
        if count == 0:
            return self._snapshot(self.num_qubits, x, z, coefficients)
        if k >= m:
            for i in range(m):
                for start in range(0, k, tile):
                    stop = min(k, start + tile)
                    rows = slice(i * k + start, i * k + stop)
                    x[rows], z[rows], phase = product_words(
                        self.x[i], self.z[i], other.x[start:stop], other.z[start:stop])
                    coefficients[rows] = self.coefficients[i] * other.coefficients[start:stop] * phase
        else:
            for j in range(k):
                for start in range(0, m, tile):
                    stop = min(m, start + tile)
                    rows = slice(start * k + j, stop * k, k)
                    x[rows], z[rows], phase = product_words(
                        self.x[start:stop], self.z[start:stop], other.x[j], other.z[j])
                    coefficients[rows] = self.coefficients[start:stop] * other.coefficients[j] * phase
        if not np.isfinite(coefficients).all():
            raise ValueError("Pauli product coefficients must be finite")
        return PauliTerms._snapshot(self.num_qubits, x, z, coefficients)

    def group(self, *, strategy, max_bytes=DEFAULT_INPUT_BYTES, max_comparisons=1_000_000,
              limit_name="max_comparisons"):
        """Deterministic first-fit QWC or general commuting algebra partition.

        Grouping keeps deterministic first-fit and original member order.
        QWC compares each term with the group's accumulated basis, while
        general commutation compares it with each group member. Two words
        commute exactly when ``popcount(x1 & z2) + popcount(z1 & x2)`` is
        even (``anticommutes``). A word is qubit-wise compatible with a QWC
        group when ``overlap & ((x ^ bx) | (z ^ bz))`` is zero in every word,
        with ``overlap = (x | z) & (bx | bz)`` (``qwc``); on success the
        basis ``(bx, bz)`` is updated by OR. General groups are algebraic
        partitions, not QWC measurement settings.

        Grouping stores ordered member indices and QWC basis masks, then
        tests bounded candidate tiles. Its logical buffer reserve is
        ``M*(16*W+16)+64*t*W+64*t+32*W`` bytes. QWC compares group masks,
        while general commuting groups compare members. ``comparison_count``
        counts every candidate actually evaluated.

        Grouping admits each candidate tile before evaluating it.
        ``comparison_count`` counts all candidates in evaluated tiles. For L
        input terms it is at most L(L-1)/2. The next tile is admitted only if
        ``C_so_far + charge <= max_comparisons``, so the next-tile minimum
        ``C_so_far + charge`` is necessary to continue, while ``L(L-1)/2`` is
        sufficient for complete grouping. A refusal names the actual grouping
        work, the next charge and the next-tile minimum, then reports
        ``L(L-1)/2`` as the value sufficient for this phase. Grouping can
        succeed below that envelope when its actual comparisons fit. Changing
        only the comparison cap leaves the byte-selected tile, traversal
        order and resulting partition unchanged, so retrying at ``L(L-1)/2``
        completes grouping when the other limits are unchanged. The
        triangular quantity is printed on refusal only; it is neither
        reserved in advance nor a reason to continue past the cap. Later
        phases using the same option are admitted separately. General
        commuting grouping charges actual member-pair tests and can reach the
        quadratic count even with one group; its evaluated members form a
        disjoint partition of the preceding terms, so its total is also at
        most ``L(L-1)/2``.

        The reserve covers two M-by-W basis arrays and two M-entry
        membership sequences (``M(16W+16)``); four t-by-W uint64 candidate
        arrays, their t-entry reductions and, for commuting groups, the
        gathered member rows, masks and popcounts (``64tW + 64t``); and the
        scalar term masks (``32W``). The tile ``t = min(M, C)`` is chosen
        from ``max_bytes - M(16W+16) - 32W`` at ``64W + 64`` bytes per
        candidate before any candidate is allocated, and a table that needs
        a comparison is refused when one candidate cannot be funded. If term
        i meets ``g_i`` groups (``g_0 = 0``), a full QWC pass costs
        ``sum_i g_i`` comparisons. Forming all G groups as early as possible
        maximizes every ``g_i``, giving ``0+1+...+(G-1)+(L-G)G``, so the
        count is at most ``LG - G(G+1)/2 < LG``; ``g_i <= i`` gives
        ``L(L-1)/2``. When every term fits one group the count is ``L-1``.
        A block that contains the first compatible group is still counted
        in full, so the count can exceed a scalar early-exit count: with
        ``j_i`` the zero-based first compatible group and t the tile, term i
        costs ``min(g_i, t*(floor(j_i/t)+1))`` when a compatible group
        exists and ``g_i`` otherwise. A caller that passes a Method field as
        ``max_comparisons`` passes that field's name as ``limit_name``, such
        as ``"FixedGCIM.max_analysis_work"``, and the refusal names it. The
        default ``max_comparisons`` is registered in
        ENGINEERING_CONSTANTS.md, "Explicit input operation defaults".
        """
        if strategy not in ("qwc", "commuting"):
            raise ValueError("grouping strategy must be qwc or commuting")
        m, w = len(self), self.x.shape[1]
        fixed = m * (16 * w + 16) + 32 * w if m else 0
        _check_bytes(fixed, max_bytes, "Pauli grouping")
        if type(max_comparisons) is not int or max_comparisons < 1:
            raise ValueError("max_comparisons must be a positive integer")
        tile = max(1, min(m, _PAULI_TILE, (max_bytes - fixed) // (64 * w + 64)))
        if m > 1:
            _check_bytes(fixed + 64 * tile * w + 64 * tile, max_bytes, "Pauli grouping")
        groups = []
        comparisons = 0

        def admit(charge):
            nonlocal comparisons
            total = comparisons + charge
            if total > max_comparisons:
                raise ValueError(
                    f"Pauli grouping exceeds {limit_name}={max_comparisons}: "
                    f"grouping used={comparisons}, next charge={charge}. "
                    f"The next tile needs {limit_name} of at least {total}. "
                    f"Raise {limit_name} to {m * (m - 1) // 2}, which is sufficient "
                    "for complete grouping. Later phases using this limit can need more.")
            comparisons += charge

        admit(0)
        if strategy == "qwc":
            bx, bz = np.empty((m, w), dtype=np.uint64), np.empty((m, w), dtype=np.uint64)
            for i in range(m):
                found = None
                for start in range(0, len(groups), tile):
                    stop = min(len(groups), start + tile)
                    admit(stop - start)
                    compatible = qwc(self.x[i], self.z[i], bx[start:stop], bz[start:stop])
                    if compatible.any():
                        found = start + int(np.argmax(compatible))
                        break
                if found is None:
                    found = len(groups)
                    groups.append([])
                    bx[found], bz[found] = self.x[i], self.z[i]
                else:
                    bx[found] |= self.x[i]
                    bz[found] |= self.z[i]
                groups[found].append(i)
        else:
            for i in range(m):
                for group in groups:
                    for start in range(0, len(group), tile):
                        members = group[start:start + tile]
                        admit(len(members))
                        if anticommutes(self.x[i], self.z[i], self.x[members], self.z[members]).any():
                            break
                    else:
                        group.append(i)
                        break
                else:
                    groups.append([i])
        return PauliGrouping(tuple(tuple(group) for group in groups), comparisons)


@dataclass(frozen=True, slots=True)
class PauliGrouping:
    """Ordered algebra partition; comparison count is not measurement count.

    Attributes:
        groups: Original table indices, preserving first-fit and member order.
        comparison_count: Evaluated QWC group masks or commuting member pairs, including every candidate of an evaluated tile.
    """

    groups: tuple[tuple[int, ...], ...]
    comparison_count: int


def _pauli_masks(label: str) -> tuple[int, int, int]:
    """Return flip mask, phase mask, and Y count in Qiskit little-endian order."""
    flip_mask = phase_mask = y_count = 0
    for qubit, factor in enumerate(reversed(label)):
        if factor in "XY":
            flip_mask |= 1 << qubit
        if factor in "YZ":
            phase_mask |= 1 << qubit
        if factor == "Y":
            y_count += 1
    return flip_mask, phase_mask, y_count


def combine_terms(terms):
    """Combine a bounded term stream without thresholding nonzero residues.

    Terms with identical labels are summed with ``stable_complex_sum``, which
    keeps a small nonzero residue of near cancellation. Only a sum that is
    exactly zero is removed.
    """
    buckets = {}
    for label, coefficient in terms:
        value = complex(coefficient)
        if not np.isfinite(value):
            raise ValueError("Pauli coefficients must be finite")
        buckets.setdefault(label, []).append(value)
    combined = []
    for label, values in buckets.items():
        value = stable_complex_sum((v.real for v in values), (v.imag for v in values))
        if value != 0.0:
            combined.append((label, value))
    return tuple(combined)


def _rotate(value, exponent):
    """Return ``value * i**exponent`` as an exact component permutation or negation."""
    return (value, complex(-value.imag, value.real), -value, complex(value.imag, -value.real))[exponent & 3]


def pauli_action_requirements(rows, terms, flips, columns=1, *, complex_input=False, label_width=None):
    """Return logical ``(bytes, work)`` allowances for one ``apply_terms`` call.

    N = rows, b = columns >= 1, L = terms (including stored zeros),
    g = flips >= the number of distinct flip masks, and t = min(N, C),
    where C = _PAULI_TILE. The caller supplies admitted dimensions and
    a valid group-count upper bound. A single-word N-coordinate action
    can use g = min(L, N) from counts alone. This helper reads no table.

    Bytes are ``(33 if complex_input else 49)*N*b + 24*L + 16*g + 8
    + max(9*L, (64+32*b)*t)``. Input, output and a finite-mask reserve
    occupy 33Nb. A possible conversion to complex128 adds 16Nb, whose
    result stays live throughout the action. Set complex_input=True
    only for an already complex128 input. An existing out buffer counts
    as the resident output. The law reserves at most 16 bytes per input
    or output entry, as used by the admitted binary64 callers.

    Logical metadata is 8L member indices, 16L rotated coefficients,
    8g flip keys and 8(g+1) group offsets. Forming the Y counts holds
    an 8L AND array beside an L-byte popcount array. Both are released
    before the grouped tiles. The larger of the grouped (64+16b)t and
    ordered-retry (64+32b)t buffers therefore shares a maximum with 9L.
    If label_width=q is supplied, add 32L for the packed x, z and
    coefficient arrays. Those arrays stay live through both phases.

    Work is ``(L+g) + 3*L + N*(L+g*b+b) + N*b + L*N*b``. Metadata
    visits L+g and the AND, popcount and rotation visits 3L precede
    the grouped term-phase and group-column work N*(L+g*b), plus its
    output zero fill Nb. The possible ordered retry adds Nb+LNb.
    The 3L preprocessing work is independent of b. Labels add L*(q+1) work.
    The retry is reserved in full because grouping can fail late.

    These are conservative kernel/pass units and logical buffer
    payloads, including an overestimate for empty tables. They exclude
    the caller-owned packed table or label payload, Python object
    overhead, opaque allocation workspace and RSS.
    """
    if type(columns) is not int or columns < 1:
        raise ValueError("a Pauli action requires at least one state column")
    n, b, tile = rows, columns, min(rows, _PAULI_TILE)
    size = (
        (33 if complex_input else 49) * n * b
        + 24 * terms + 16 * flips + 8
        + max(9 * terms, (64 + 32 * b) * tile)
    )
    work = (
        (terms + flips) + 3 * terms
        + n * (terms + flips * b + b)
        + n * b + terms * n * b
    )
    if label_width is not None:
        size += 32 * terms
        work += terms * (label_width + 1)
    return size, work


def _grouped_pass(groups, rotated, phase_masks, columns, result, tile):
    """Accumulate each flip group's tiled diagonal action into result; False after overflow.

    For each group and coordinate tile the diagonal ``sum_t rotated[t] *
    (-1)**popcount(k & z_t)`` is summed in member order, the source rows
    ``k XOR f`` are gathered once for all columns, multiplied by it and added
    to the output rows k. Overflow or an invalid operation stops the pass
    and returns False, and the tile arrays are released with this frame.
    """
    dimension = len(result)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for flip, members in groups.items():
                for start in range(0, dimension, tile):
                    k = np.arange(start, min(dimension, start + tile), dtype=np.uint64)
                    diagonal = np.zeros(k.size, dtype=np.complex128)
                    for t in members:
                        signs = 1 - 2 * (np.bitwise_count(k & phase_masks[t]) & 1).astype(np.int8)
                        diagonal += rotated[t] * signs
                    block = columns[(k ^ np.uint64(flip)).astype(np.intp)]
                    block *= diagonal[:, None]
                    result[start:start + k.size] += block
                    del block, diagonal, k
    except FloatingPointError:
        return False
    return True


def apply_terms(
    terms,
    state: np.ndarray,
    *,
    num_qubits: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a Pauli sum to one state or a block of state columns.

    Terms with the same flip mask share a diagonal phase sum and one gather
    of the state rows. For N coordinates, b columns, L terms and G distinct
    flip masks, the arithmetic costs O(LN + GNb). Coordinate tiles bound
    the temporary arrays. Grouping changes summation order, so results
    agree with the termwise formula within floating-point error rather than
    necessarily bit for bit. An overflow in grouped arithmetic retries the
    original term order.

    ``terms`` is a single-word ``PauliTerms`` table or an iterable of
    ``(label, value)`` pairs, which is packed once (32L bytes); each label
    must be a string of exactly num_qubits letters from IXYZ, qubit zero
    rightmost, and any other label raises ValueError naming it. ``state``
    has shape (N,) or (N, b) with ``N = 2**num_qubits`` and b >= 1; a
    float input is converted to complex128 once. ``out`` must have the
    same shape, a complex dtype and no overlap with the input; it is
    zero-filled and returned.

    With ``P(x, z) = i**y X**x Z**z``, ``y = popcount(x & z)``, a term acts
    as ``P|j> = i**y (-1)**popcount(j & z) |j XOR x>``. For each distinct
    flip f the destination-coordinate diagonal is ``d_f(k) = sum_{t: x_t=f}
    c_t (-i)**y_t (-1)**popcount(k & z_t)`` and ``(A V)[k, :] = sum_f
    d_f(k) V[k XOR f, :]``, because the parity of ``(k XOR f) & z`` is that
    of ``k & z`` plus ``f & z`` and ``popcount(f & z) = y``. A Y term on
    ``|0>`` gives +i at destination 1. Groups are built once in
    first-occurrence order and keep member order; each diagonal entry is
    computed once per tile and shared by all b columns. For b = 1 the phase
    sum still does O(LN) work; the saving is fewer state gathers and their
    reuse across columns. Empty tables return zero; terms with f = 0 share
    the identity permutation.

    Rounding: with u = 2**-53, ``gamma_r = r*u/(1-r*u)``, ``C1 = sum
    abs(c_t)``, m* the largest group, ``mu = sqrt(2)*gamma_2``, finite
    normal intermediate products, round to nearest and ``(L+G+m*)u < 1``,
    ``||Y_old - AV||_F <= e_o*C1*||V||_F`` with ``e_o = (1+mu)(1+gamma_(L-1))
    - 1`` and ``||Y_group - AV||_F <= e_g*C1*||V||_F`` with ``e_g =
    (1+gamma_(m*-1))(1+mu)(1+gamma_(G-1)) - 1``, so the grouped and
    termwise results differ by at most ``(e_o+e_g)*C1*||V||_F`` (to first
    order ``[L + m* + G - 3 + 4*sqrt(2)]u C1 ||V||``). Each group's diagonal
    sum errs by at most ``gamma_(m_f-1)*sum_group abs(c_t)``, a permutation
    preserves the state norm, the complex product adds mu, and the group
    errors add by the triangle inequality; quarter turns and sign changes
    are exact. Since ``m* + G - 2 <= L - 1``, ``e_g <= e_o``. With gradual
    underflow add ``4*(L+G)*eta*sqrt(N*b)/(1-(L+G+4)*u)``, eta = 2**-1074,
    per action (flush-to-zero excluded). Summing coefficients first can
    overflow where the termwise products are finite (two ``DBL_MAX``
    identity terms on amplitudes near 1e-308), so an overflow or invalid
    result in grouped arithmetic retries the original ordered action once;
    an ordered result that also overflows stays nonfinite.

    Memory: the input and output of 16Nb bytes each, O(L) group metadata
    and at most ``(64+16b)C`` explicit scratch bytes per tile of length C
    (``(64+32b)C`` on the ordered retry), never a full diagonal.
    ``pauli_action_requirements`` states the admitted envelope.
    """
    if (isinstance(num_qubits, (bool, np.bool_)) or not isinstance(num_qubits, (int, np.integer))
            or not 0 <= num_qubits < 63):
        raise ValueError("full-state Pauli action requires 0 <= num_qubits < 63")
    num_qubits = int(num_qubits)
    vector = np.asarray(state, dtype=np.complex128)
    dimension = 1 << num_qubits
    if vector.ndim not in (1, 2) or vector.shape[0] != dimension:
        raise ValueError("state length must match the Pauli operator qubit count")
    if vector.ndim == 2 and vector.shape[1] < 1:
        raise ValueError("a state block requires at least one column")
    if out is None:
        output = np.zeros(vector.shape, dtype=np.complex128)
    else:
        output = np.asarray(out)
        if output.shape != vector.shape or output.dtype.kind != "c":
            raise ValueError("out must be a complex state-sized buffer")
        if np.may_share_memory(output, vector):
            raise ValueError("out must not overlap the input state")
        output.fill(0.0)
    if isinstance(terms, PauliTerms):
        if terms.num_qubits != num_qubits or terms.x.shape[1] != 1:
            raise ValueError("full-state action requires a matching single-word Pauli table")
        flips, phase_masks, coefficients = terms.x[:, 0], terms.z[:, 0], terms.coefficients
    else:
        terms = tuple(terms)
        flips = np.empty(len(terms), dtype=np.uint64)
        phase_masks = np.empty(len(terms), dtype=np.uint64)
        coefficients = np.empty(len(terms), dtype=np.complex128)
        for index, (label, value) in enumerate(terms):
            # _pauli_masks reads any other letter as I and a short label as
            # acting on the low qubits, so the label is checked here first.
            if not isinstance(label, str) or len(label) != num_qubits or set(label) - set("IXYZ"):
                raise ValueError(f"Pauli label {label!r} must have num_qubits={num_qubits} IXYZ letters")
            flip, phase, _ = _pauli_masks(label)
            flips[index], phase_masks[index], coefficients[index] = flip, phase, complex(value)
    if len(coefficients) == 0:
        return output
    columns = vector if vector.ndim == 2 else vector[:, None]
    result = output if output.ndim == 2 else output[:, None]
    # Group metadata: member indices (8L), rotated coefficients (16L) and
    # the flip keys (8G). Each term's Y count is formed again on the retry.
    groups = {}
    for index, flip in enumerate(flips):
        groups.setdefault(int(flip), []).append(index)
    rotated = [_rotate(complex(c), -int(count))
               for c, count in zip(coefficients, np.bitwise_count(flips & phase_masks), strict=True)]
    tile = min(dimension, _PAULI_TILE)
    if _grouped_pass(groups, rotated, phase_masks, columns, result, tile):
        return output
    # Grouped coefficient sums can overflow even when c_t*v is finite. The
    # grouped tile arrays were released with the pass before this retry.
    result.fill(0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(len(coefficients)):
            phase = _rotate(complex(coefficients[t]), int(np.bitwise_count(flips[t] & phase_masks[t])))
            for start in range(0, dimension, tile):
                k = np.arange(start, min(dimension, start + tile), dtype=np.uint64)
                signs = 1 - 2 * (np.bitwise_count(k & phase_masks[t]) & 1).astype(np.int8)
                block = phase * columns[start:start + k.size]
                block *= signs[:, None]
                result[(k ^ flips[t]).astype(np.intp)] += block
                del block, k
    return output
