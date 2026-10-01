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
    """Table of M ordered ladder-operator strings with complex coefficients.

    Each string is a product of creation and annihilation operators, written
    with the left factor first. The rightmost operator acts first. No normal
    ordering is applied. [`fermion_table`][nwqlib.operators._fermion.fermion_table]
    and `OperatorInput.fermion_terms()` return it, and constructing it
    directly raises `TypeError`. `len(table)` is M. The fields below are
    read-only.

    Attributes:
        num_modes: Number of fermionic modes, nonnegative.
        offsets: int64 array of length M+1. String k uses entries
            `offsets[k]` to `offsets[k+1] - 1` of `modes` and `actions`.
        modes: int64 array of length P, the mode of each ladder operator.
        actions: uint8 array of length P. 0 annihilates and 1 creates.
        coefficients: complex128 array of length M.
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
        """Return the ordered product table, every string of this table times every string of `other`.

        String `(i, j)` is string i of this table followed by string j of
        `other`, with coefficient `c_i * d_j`. With M and K strings and
        `P_left` and `P_right` ladder entries, the product has `M*K` strings
        and `K*P_left + M*P_right` entries. No term is combined or dropped.

        Args:
            other (FermionTerms): Table with the same `num_modes`.
            max_bytes (int): Byte limit, checked before the product is
                formed. Default 10 GB (decimal, `10_000_000_000`).

        Returns:
            table (FermionTerms): The product table.

        Raises:
            ValueError: If the mode counts differ, a product coefficient is
                not finite, or the product exceeds `max_bytes`.
        """
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
        """Map the strings to qubits and return the Pauli operator as an OperatorInput.

        Mode j goes to qubit j, with `|1>` meaning occupied. With
        `mapping="jw"` (Jordan–Wigner, with the parity of the lower modes),
        `a_j† = (prod_{k<j} Z_k) (X_j - i Y_j)/2` and
        `a_j = (prod_{k<j} Z_k) (X_j + i Y_j)/2`. With `mapping="z_free"` the
        parity string is omitted, which gives qubit ladder operators. Every
        product term is kept until the final sum, which adds identical Pauli
        words and removes only exact zeros. No coefficient is pruned, and no
        state is formed. The mapping cost is checked against `max_bytes`
        before the first product (see [Input cost
        controls](../development/input_contracts.md)).

        Args:
            mapping (str): `"jw"` or `"z_free"`.
            max_bytes (int): Byte limit. Default 10 GB (decimal,
                `10_000_000_000`).

        Returns:
            operator (OperatorInput): The Pauli operator on `num_modes`
                qubits.

        Raises:
            ValueError: If `mapping` is another value, the table has no
                modes, or the mapping exceeds `max_bytes`.
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
    """Return the raw table of ordered fermionic strings, keeping duplicates, order and zeros.

    Each row is `(coefficient, ops)`, where `ops` lists `(mode, action)`
    pairs with the left factor first. Action 0 annihilates and 1 creates,
    and the rightmost operator acts first. No normal ordering is applied.
    Use [`ingest_fermion`][nwqlib.operators._fermion.ingest_fermion] for an
    operator with identical strings summed. The table size is checked
    against `max_bytes` before data is stored (see [Input cost
    controls](../development/input_contracts.md)). An iterator is read at
    most one item past the limit, and an over-limit item is rejected before
    it is stored.

    Args:
        terms (Iterable[tuple[complex, Iterable[tuple[int, int]]]]): Rows of
            a finite coefficient and its ordered `(mode, action)` pairs.
        num_modes (int): Number of modes q, a nonnegative int64 value.
        max_bytes (int): Byte limit. Default 10 GB (decimal,
            `10_000_000_000`).

    Returns:
        table (FermionTerms): The raw table.

    Raises:
        ValueError: If a mode is outside `0 <= mode < num_modes`, an action
            is not 0 or 1, a coefficient is not finite, or the table exceeds
            `max_bytes`.
        TypeError: If a coefficient is not a number.
    """
    # Size law. M rows with P ladder operators use 24M+9P+8 bytes. The check
    # reserves six layouts plus ceil((q+1)/8) dimension bytes for
    # conversion, snapshots and coalescing. Unsized rows and strings use one
    # lookahead and reject before storing or traversing an over-limit item.
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
    """Return an OperatorInput for a fermionic operator given as ordered ladder strings.

    Rows are read as in [`fermion_table`][nwqlib.operators._fermion.fermion_table].
    Only strings with the same complete ordered `(mode, action)` sequence are
    summed, and only an exactly zero sum is removed. No anticommutation or
    normal ordering is applied, so two orderings of one operator stay
    separate rows, and the `structure` of the result is `"general"`. Map it
    to qubits with `operator.fermion_terms().to_pauli(mapping="jw")`.

    Args:
        terms (Iterable[tuple[complex, Iterable[tuple[int, int]]]]): Rows of
            a finite coefficient and its ordered `(mode, action)` pairs.
        num_modes (int): Number of modes q.
        max_bytes (int): Byte limit. Default 10 GB (decimal,
            `10_000_000_000`).

    Returns:
        operator (OperatorInput): The summed fermionic operator.

    Raises:
        ValueError: As for `fermion_table`.
        TypeError: If a coefficient is not a number.
    """
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
