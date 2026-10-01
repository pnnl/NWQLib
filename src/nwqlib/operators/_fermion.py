"""Ordered ladder tables and explicit SDK-free Jordan–Wigner/Z-free mapping."""

from collections.abc import Sized
from dataclasses import dataclass
from numbers import Complex
import sys

import numpy as np

from .access import DEFAULT_INPUT_BYTES, _check_bytes
from ._pauli import PauliTerms, combine_terms


def _count_product(a, b, limit):
    """Reject before multiplication can form an inadmissible count."""
    limit = min(limit, sys.maxsize)
    if b and a > limit // b:
        raise ValueError("operator expansion exceeds its finite size limit")
    return a * b


def _fermion_law(m, p, q):
    """Return ``(items, bytes, work)`` for a Fermion table of m strings and p ladder entries.

    One layout is ``24 m + 9 p + 8`` bytes: per string an int64 offset and a
    complex128 coefficient, per entry an int64 mode and a uint8 action, and
    the closing offset.
    """
    # Six layouts cover raw conversion/snapshot and exact-string coalescing.
    # The coalesced handle also owns dimension=2**q (q+1 bits).
    payload = 6 * (24 * m + 9 * p + 8) + (q + 8) // 8
    if m + p > sys.maxsize or payload > sys.maxsize:
        raise ValueError("Fermion envelope exceeds native index representability")
    return m + p, payload, max(q, m + p)


def _check(max_bytes, law):
    """Check the byte component of an ``(items, bytes, work)`` law against ``max_bytes``."""
    _check_bytes(law[1], max_bytes, "operator expansion")


@dataclass(frozen=True, eq=False, init=False, slots=True)
class FermionTerms:
    """Immutable ragged strings, left factor first; action is right to left.

    Attributes:
        num_modes: Nonnegative width; action 0 annihilates and 1 creates.
        offsets: Native int64 boundaries, length M+1.
        modes: Native int64 modes, length P.
        actions: Native uint8 ladder actions, length P.
        coefficients: Native complex128 coefficients, length M.
    """

    num_modes: int
    offsets: np.ndarray
    modes: np.ndarray
    actions: np.ndarray
    coefficients: np.ndarray

    def __init__(self, *args, **kwargs):
        raise TypeError("Fermion tables require bounded input admission")

    @classmethod
    def _snapshot(cls, q, offsets, modes, actions, coefficients):
        from .inputs import _freeze_array
        instance = object.__new__(cls)
        object.__setattr__(instance, "num_modes", q)
        for name, value in (("offsets", offsets), ("modes", modes),
                            ("actions", actions), ("coefficients", coefficients)):
            object.__setattr__(instance, name, _freeze_array(value))
        return instance

    def __len__(self):
        return len(self.coefficients)

    def product(self, other, *, max_bytes=DEFAULT_INPUT_BYTES):
        """Raw ordered concatenation: MK strings, K*P_left+M*P_right entries."""
        if not isinstance(other, FermionTerms) or self.num_modes != other.num_modes:
            raise ValueError("Fermion product requires equal mode widths")
        m = _count_product(len(self), len(other), max_bytes)
        p = (_count_product(len(other), len(self.modes), max_bytes)
             + _count_product(len(self), len(other.modes), max_bytes))
        _check(max_bytes, _fermion_law(m, p, self.num_modes))
        offsets = np.empty(m + 1, dtype=np.int64)
        modes, actions = np.empty(p, dtype=np.int64), np.empty(p, dtype=np.uint8)
        coefficients = np.empty(m, dtype=np.complex128)
        position = 0
        offsets[0] = 0
        if m == 0:
            return self._snapshot(self.num_modes, offsets, modes, actions, coefficients)
        for i in range(len(self)):
            for j in range(len(other)):
                k = i * len(other) + j
                for table, row in ((self, i), (other, j)):
                    start, end = map(int, table.offsets[row:row + 2])
                    length = end - start
                    modes[position:position + length] = table.modes[start:end]
                    actions[position:position + length] = table.actions[start:end]
                    position += length
                offsets[k + 1] = position
                coefficients[k] = self.coefficients[i] * other.coefficients[j]
        if not np.isfinite(coefficients).all():
            raise ValueError("Fermion product coefficients must be finite")
        return self._snapshot(self.num_modes, offsets, modes, actions, coefficients)

    def to_pauli(self, *, mapping, max_bytes=DEFAULT_INPUT_BYTES):
        """Map explicitly, keeping all candidate contributions until final sum.

        mapping is 'jw' (lower-mode parity) or 'z_free' (qubit ladders).
        max_bytes bounds raw candidate tables and their documented conversion
        frontier. No SDK, pruning or full-state action is used.
        """
        if mapping not in ("jw", "z_free"):
            raise ValueError("mapping must be jw or z_free")
        if self.num_modes == 0:
            raise ValueError("Pauli mapping requires at least one mode")
        # Offset traversal itself is admitted before calculating the chain.
        _check(max_bytes, _fermion_law(len(self), len(self.modes), self.num_modes))
        lengths = (int(self.offsets[i + 1]) - int(self.offsets[i]) for i in range(len(self)))
        law = mapping_requirements(lengths, num_modes=self.num_modes, max_bytes=max_bytes)
        _check(max_bytes, law)
        from .inputs import ingest_pauli_masks
        w = (self.num_modes + 63) // 64
        # The mapped row count is the first component (also bounds raw input).
        rows = sum(1 << (int(self.offsets[i + 1]) - int(self.offsets[i])) for i in range(len(self)))
        xs, zs = np.empty((rows, w), dtype=np.uint64), np.empty((rows, w), dtype=np.uint64)
        coefficients = np.empty(rows, dtype=np.complex128)
        cursor = 0
        for i, coefficient in enumerate(self.coefficients):
            start, end = map(int, self.offsets[i:i + 2])
            if start == end:
                xs[cursor], zs[cursor], coefficients[cursor] = 0, 0, coefficient
                cursor += 1
                continue
            partial = None
            for j in range(start, end):
                local = _ladder_image(int(self.modes[j]), creation=bool(self.actions[j]),
                                      num_modes=self.num_modes, mapping=mapping)
                partial = local if partial is None else partial.product(local, max_bytes=max_bytes)
            count = len(partial)
            xs[cursor:cursor + count], zs[cursor:cursor + count] = partial.x, partial.z
            coefficients[cursor:cursor + count] = coefficient * partial.coefficients
            cursor += count
        return ingest_pauli_masks(xs, zs, coefficients, num_qubits=self.num_modes, max_bytes=max_bytes)


def fermion_table(terms, *, num_modes, max_bytes=DEFAULT_INPUT_BYTES):
    """Admit (coefficient, ordered (mode,action) iterable) rows.

    Raw duplicates/order/zeros survive. M rows/P operators use 24M+9P+8
    bytes; reserve six layouts plus ceil((q+1)/8) dimension bytes for
    conversion/snapshots/coalescing. Unsized rows and strings use one lookahead,
    rejecting before storing/traversing an over-limit item. No normal order.
    """
    if type(num_modes) is not int or not 0 <= num_modes <= np.iinfo(np.int64).max:
        raise ValueError("num_modes must be a nonnegative representable int64 width")
    _check(max_bytes, _fermion_law(0, 0, num_modes))
    if isinstance(terms, Sized):
        _check(max_bytes, _fermion_law(len(terms), 0, num_modes))
    offsets, modes, actions, coefficients = [0], [], [], []
    for coefficient, ops in terms:
        m = len(coefficients) + 1
        _check(max_bytes, _fermion_law(m, len(modes), num_modes))
        if isinstance(coefficient, (bool, np.bool_)) or not isinstance(coefficient, Complex):
            raise TypeError("Fermion coefficients must be numerical scalars")
        value = complex(coefficient)
        if not np.isfinite(value):
            raise ValueError("Fermion coefficients must be finite")
        if isinstance(ops, Sized):
            _check(max_bytes, _fermion_law(m, len(modes) + len(ops), num_modes))
        for mode, action in ops:
            _check(max_bytes, _fermion_law(m, len(modes) + 1, num_modes))
            if type(mode) is not int or not 0 <= mode < num_modes:
                raise ValueError("Fermion mode must lie within num_modes")
            if type(action) is not int or action not in (0, 1):
                raise ValueError("Fermion action must be 0 or 1")
            modes.append(mode)
            actions.append(action)
        coefficients.append(value)
        offsets.append(len(modes))
    return FermionTerms._snapshot(num_modes, np.array(offsets, dtype=np.int64),
                                 np.array(modes, dtype=np.int64), np.array(actions, dtype=np.uint8),
                                 np.array(coefficients, dtype=np.complex128))


def _coalesced_fermion(table, *, max_bytes=DEFAULT_INPUT_BYTES):
    """Admit a Fermion table after coalescing identical complete strings.

    Only strings with the same ordered ``(mode, action)`` sequence combine,
    and only an exact-zero sum is removed. No anticommutation or normal
    ordering is applied, so two orderings of one operator stay separate rows.
    Hermiticity would need that algebra, so ``structure`` stays general.
    """
    from .inputs import OperatorInput, _digest, _manifest
    combined = combine_terms((tuple(zip(table.modes[int(table.offsets[i]):int(table.offsets[i + 1])].tolist(),
                                       table.actions[int(table.offsets[i]):int(table.offsets[i + 1])].tolist(), strict=True)), c)
                             for i, c in enumerate(table.coefficients))
    # Admission was made against raw M/P, so these smaller exact-string rows fit.
    data = fermion_table(((c, key) for key, c in combined), num_modes=table.num_modes, max_bytes=max_bytes)
    arrays = (data.offsets, data.modes, data.actions, data.coefficients)
    manifest = _manifest("fermion", 1 << data.num_modes, (len(data),), "int64[offsets,modes]+uint8[actions]+complex128",
                         sum(a.nbytes for a in arrays),
                         _digest("fermion.ordered.left-factor-first.action-right-to-left", (data.num_modes,), arrays),
                         ("fermion_terms",), ("finite", "ordered; no normal ordering"),
                         "M rows/P operators: 24M+9P+8 bytes; admission reserves six layouts plus dimension bytes; work=max(q,M+P)")
    return OperatorInput._from_admitted(manifest, "general", data)


def ingest_fermion(terms, *, num_modes, max_bytes=DEFAULT_INPUT_BYTES):
    """Coalesce identical complete ordered strings only, dropping exact zero."""
    return _coalesced_fermion(fermion_table(terms, num_modes=num_modes, max_bytes=max_bytes), max_bytes=max_bytes)


def mapping_requirements(lengths, *, num_modes, max_bytes=DEFAULT_INPUT_BYTES, labels=False):
    """Cumulative mapping law, including raw admission and final conversion.

    Count arithmetic is capped by intp representability and data bytes before
    shifts/arrays. Counts and scalar visits remain descriptive cost laws.
    Local ladders reserve 4L bytes each (two rows plus snapshots), and products
    reserve 2*N*L+32W each. Final arrays/scaling/coalescing reserve 6*R*L, plus
    R*S for the identity JSON share S of ``inputs._pauli_identity_bytes`` that
    the final mask ingestion charges.
    Explicit SDK conversion additionally reserves R*(3q+48) bytes and
    R*(3q+3) work, bounding labels, SDK boolean tables, scalars and conversion.

    Here ``q = num_modes``, ``W = ceil(q/64)`` mask words, and
    ``L = 16 W + 16`` bytes is one Pauli row (two W-word uint64 masks and a
    complex128 coefficient). A string of ``p`` ladder operators maps to
    ``2**p`` rows, and ``R`` is their total over all strings. The
    left-to-right products of one string have ``4, 8, ..., 2**p`` rows, so a
    string with ``p >= 2`` forms ``N = 2**(p+1) - 4`` candidate rows.
    """
    if type(num_modes) is not int or not 1 <= num_modes <= sys.maxsize:
        raise ValueError("mapping requires a positive representable width")
    _check_bytes((num_modes + 8) // 8, max_bytes, "mapping width metadata")
    cap = min(sys.maxsize, max_bytes)
    from .inputs import _pauli_identity_bytes
    w, m, p_total, rows, payload, work = (num_modes + 63) // 64, 0, 0, 0, 0, 0
    layout = 16 * w + 16
    final = 6 * layout + _pauli_identity_bytes(num_modes)
    for p in lengths:
        if type(p) is not int or p < 0:
            raise ValueError("operator lengths must be nonnegative integers")
        m += 1
        p_total += p
        if m + p_total > cap or p >= cap.bit_length():
            raise ValueError("mapping exceeds max_bytes")
        count = 1 << p
        rows += count
        if rows > cap:
            raise ValueError("mapping exceeds max_bytes")
        payload += 4 * p * layout
        work += 2 * p * (w + 1)
        if p == 0:
            payload += 2 * layout
            work += w + 1
        # Geometric candidate sum, after the capped final power is admitted.
        if p >= 2:
            candidates = 2 * count - 4
            payload += 2 * candidates * layout + (p - 1) * 32 * w
            work += candidates * (w + 1)
        raw = _fermion_law(m, p_total, num_modes)
        total = (max(raw[0], rows), raw[1] + payload + rows * final,
                 raw[2] + work + max(num_modes, rows * (w + 1)))
        if labels:
            total = (max(total[0], 1), total[1] + max(rows, 1) * (3 * num_modes + 48),
                     total[2] + max(rows, 1) * (3 * num_modes + 3))
        if max(total) > sys.maxsize:
            raise ValueError("mapping envelope exceeds native index representability")
        _check(max_bytes, total)
    raw = _fermion_law(m, p_total, num_modes)
    total = (max(raw[0], rows), raw[1] + payload + rows * final,
             raw[2] + work + max(num_modes, rows * (w + 1)))
    if labels:
        total = (max(total[0], 1), total[1] + max(rows, 1) * (3 * num_modes + 48),
                 total[2] + max(rows, 1) * (3 * num_modes + 3))
    if max(total) > sys.maxsize:
        raise ValueError("mapping envelope exceeds native index representability")
    _check(max_bytes, total)
    return total


def _ladder_image(mode, *, creation, num_modes, mapping):
    """Return the two-row Pauli image of one ladder operator on mode j.

    Jordan-Wigner (``jw``) maps ``a_j^dagger`` to
    ``(prod_{k<j} Z_k) (X_j - i Y_j)/2`` and ``a_j`` to
    ``(prod_{k<j} Z_k) (X_j + i Y_j)/2``, with mode j on qubit j and
    ``|1>`` occupied. ``z_free`` omits the parity string. Row 0 is the
    X-type word with coefficient 1/2, and row 1 sets both bits on qubit j,
    which is Y in the ``PauliTerms`` encoding.
    """
    w = (num_modes + 63) // 64
    x, z = np.zeros((2, w), dtype=np.uint64), np.zeros((2, w), dtype=np.uint64)
    x[:, mode // 64] = np.uint64(1 << (mode % 64))
    # Mutation probe qeb_z_ladder_injected targets the actual parity choice.
    parity = mapping == "jw"
    if parity:
        for word in range(mode // 64):
            z[:, word] = np.uint64(2**64 - 1)
        z[:, mode // 64] = np.uint64((1 << (mode % 64)) - 1)
    z[1, mode // 64] |= x[1, mode // 64]
    # Mutation probe jw_phase_string_sign_flip targets creation phase.
    y_coefficient = -0.5j if creation else 0.5j
    return PauliTerms._snapshot(num_modes, x, z, np.array([0.5, y_coefficient], dtype=np.complex128))
