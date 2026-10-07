"""Decode one-hot or binary QHD outputs into grid-point distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nwqlib.algorithms.qhd.grid import OneHotGrid


# Register probabilities per streamed mass pass: 4096 float64 values reuse
# 32 KiB of scratch while the total and the one-hot invalid mass are summed
# over the full register. The chunk bounds memory only and never changes which
# probabilities enter a mass. Revisit if measured decoding throughput justifies
# another bounded size. Registered in docs/ENGINEERING_CONSTANTS.md.
QHD_DECODING_PROBABILITY_CHUNK_SIZE = 4096


@dataclass(frozen=True, kw_only=True)
class DecodedOneHotPoint:
    """One decoded valid grid point of either encoding and the probability of the basis outcome that encodes it."""

    grid_indices: tuple[int, ...]
    probability: float


@dataclass(frozen=True, kw_only=True)
class DecodedOneHotDistribution:
    """Decoded distribution and invalid-subspace probability mass.

    ``points`` lists the valid grid points with a nonzero decoded
    probability, in lexicographic order of their grid indices.
    ``invalid_probability`` is the mass of one-hot outcomes
    without exactly one excitation per variable register, and it is zero for
    the binary encoding, whose every outcome is a grid point.
    ``total_probability`` is the mass of every decoded outcome.
    """

    points: tuple[DecodedOneHotPoint, ...]
    invalid_probability: float
    total_probability: float


def decode_onehot_basis_index(index: int, grid: OneHotGrid) -> tuple[int, ...] | None:
    """Return active grid indices if ``index`` is valid one-hot, else ``None``.

    ``index`` is a little-endian computational-basis index whose bit
    ``j*K + i`` is qubit ``OneHotGrid.qubit(j, i)``. A register with zero or
    several excitations makes the whole outcome invalid, and callers keep its
    probability in the invalid mass.
    """

    active_indices: list[int] = []
    for var_index in range(grid.num_variables):
        active = [
            grid_index
            for grid_index in range(grid.num_grid_points)
            if (index >> grid.qubit(var_index, grid_index)) & 1
        ]
        if len(active) != 1:
            return None
        active_indices.append(active[0])
    return tuple(active_indices)


def decode_basis_index(index: int, grid: OneHotGrid, bits: int | None = None) -> tuple[int, ...] | None:
    """Return the grid tuple of a full-register basis index, or None for an invalid one-hot outcome.

    ``bits`` is None for the one-hot encoding (``decode_onehot_basis_index``)
    and b for the binary encoding, whose every index decodes to
    ``n_j = (index >> j b) mod 2**b`` (``binary.decode_register_index``).
    """
    if bits is None:
        return decode_onehot_basis_index(index, grid)
    from .binary import decode_register_index

    return decode_register_index(index, grid.num_variables, bits)


def onehot_register_indices(k: int, d: int) -> np.ndarray:
    """Return the full-register index of every valid one-hot point as an int64 array of shape ``(K,)*d``.

    Valid one-hot register indices are ``z(i_0, ..., i_(d-1)) = sum_j
    2**(j K + i_j)``. Each summand sets one bit in a disjoint variable
    register, so addition and bitwise OR agree exactly. There are D
    distinct indices. Broadcasting one one-dimensional vector per axis
    enumerates them in C order, which is lexicographic grid order. A
    materialized full register must already fit ``np.intp`` and its byte
    limit. The array costs 8D bytes and its
    coordinate/local-index scratch at most 24K.

    Raises:
        ValueError: ``d*K >= 63``, so a register index would exceed signed
            int64.
    """
    if d * k >= 63:
        raise ValueError("dense one-hot register index exceeds signed int64")
    indices = np.zeros((k,) * d, dtype=np.int64)
    local = np.arange(k, dtype=np.int64)
    for j in range(d):
        shape = [1] * d
        shape[j] = k
        indices |= (np.int64(1) << (j * k + local)).reshape(shape)
    return indices


def onehot_valid_words(words: np.ndarray, k: int, d: int) -> np.ndarray:
    """Return the Boolean mask of valid one-hot register words (``uint64``, ``d*K <= 64``).

    For each variable register r of K bits, one-hot validity is
    ``r != 0 and (r & (r-1)) == 0``, and a word is
    valid when every register is.
    """
    words = np.asarray(words, dtype=np.uint64)
    register_mask = np.uint64((1 << k) - 1)
    valid = np.ones(words.shape, dtype=bool)
    for j in range(d):
        register = (words >> np.uint64(j * k)) & register_mask
        valid &= (register != 0) & ((register & (register - np.uint64(1))) == 0)
    return valid


def onehot_local_indices(words: np.ndarray, k: int, d: int) -> np.ndarray:
    """Return the ``(entries, d)`` grid tuples of valid one-hot words (``uint64``, ``d*K <= 64``).

    On a valid word, ``bit_count(w - 1)`` of its variable register w is its
    local point index.
    """
    words = np.asarray(words, dtype=np.uint64)
    register_mask = np.uint64((1 << k) - 1)
    columns = []
    for j in range(d):
        register = (words >> np.uint64(j * k)) & register_mask
        columns.append(np.bitwise_count(register - np.uint64(1)).astype(np.int64))
    return np.stack(columns, axis=-1) if columns else np.zeros((words.size, 0), dtype=np.int64)


def decode_statevector_probabilities(
    statevector: Any, grid: OneHotGrid, bits: int | None = None
) -> DecodedOneHotDistribution:
    """Decode a full-register statevector into its valid grid population and masses.

    The register has ``d K`` qubits for the one-hot encoding (``bits`` None)
    and ``d b`` qubits for the binary encoding with ``bits = b``. Each
    probability is ``np.square(np.absolute(z))`` of one amplitude, the
    evaluation whose relative rounding ``_validation.ABSOLUTE_SQUARE_ROUNDOFF``
    bounds.

    One-hot indices set the bit ``j*K+i_j`` for every variable, and binary
    indices use the variable-axis permutation. Valid, invalid and total
    masses describe their respective observed populations.

    The valid amplitudes are gathered once (``onehot_register_indices``, or
    ``binary.lexicographic_register_array``) before their probabilities are
    formed. The total and invalid masses are each summed directly with
    ``math.fsum`` over their own population, the total over every register
    probability and the invalid mass over the complement of the valid set,
    streamed ``QHD_DECODING_PROBABILITY_CHUNK_SIZE`` entries at a time with
    the one-hot word test (``onehot_valid_words``), so that no full-register
    Boolean mask is formed. ``invalid = total - fsum(valid)`` is an identity
    in exact arithmetic, not fsum semantics in floating point: a valid
    weight 1 and an invalid weight ``2**-54`` give correctly rounded total 1,
    and subtraction returns zero although the directly summed invalid mass
    is ``5.551115123125783e-17``. These accurate reductions can differ from a
    sequential ``+=`` sum in the last bits, which can change threshold
    decisions. The analysis of
    the same populations uses the same reduction. ``points`` keeps each
    valid point whose probability is not ``<= 0``, so a NaN reaches the mass
    checks instead of disappearing as a zero probability.
    """
    valid, invalid, total = decode_statevector_arrays(statevector, grid, bits)
    flat = valid.reshape(-1)
    d, k = grid.num_variables, grid.num_grid_points
    positions = np.flatnonzero(~(flat <= 0.0))
    tuples = np.stack(np.unravel_index(positions, (k,) * d), axis=-1).tolist()
    points = tuple(
        DecodedOneHotPoint(grid_indices=tuple(point), probability=probability)
        for point, probability in zip(tuples, flat[positions].tolist(), strict=True)
    )
    return DecodedOneHotDistribution(points=points, invalid_probability=float(invalid),
                                     total_probability=float(total))


def decode_statevector_arrays(statevector: Any, grid: OneHotGrid, bits: int | None = None):
    """Return ``(valid, invalid, total)``: the ``(K,)*d`` valid probability array and the two direct masses.

    The arrays and masses of ``decode_statevector_probabilities``, formed by
    the same operations, without its per-point records: ``valid`` holds
    ``np.square(np.absolute(z))`` of every valid amplitude in lexicographic
    grid order, and ``invalid`` and ``total`` are the ``math.fsum`` sums of
    their own populations, streamed over the register.
    """
    from math import fsum

    amplitudes = np.asarray(statevector, dtype=complex).reshape(-1)
    d, k = grid.num_variables, grid.num_grid_points
    width = grid.num_qubits if bits is None else d * bits
    if amplitudes.shape[0] != 2**width:
        raise ValueError("statevector dimension does not match the QHD grid")
    chunk = QHD_DECODING_PROBABILITY_CHUNK_SIZE
    scratch = np.empty(min(amplitudes.size, chunk))

    def probabilities(values):
        out = scratch[: values.size]
        np.absolute(values, out=out)
        np.square(out, out=out)
        return out

    def streamed(select):
        for start in range(0, amplitudes.size, chunk):
            values = probabilities(amplitudes[start: start + chunk])
            yield from (values if select is None else values[select(start, values.size)]).tolist()

    total = fsum(streamed(None))
    if bits is None:
        valid = np.square(np.absolute(amplitudes[onehot_register_indices(k, d)]))

        def invalid_words(start, size):
            words = np.arange(start, start + size, dtype=np.uint64)
            return ~onehot_valid_words(words, k, d)

        invalid = fsum(streamed(invalid_words))
    else:
        from .binary import lexicographic_register_array

        valid = np.square(np.absolute(lexicographic_register_array(amplitudes, d, bits))).reshape((k,) * d)
        invalid = 0.0
    return valid, float(invalid), float(total)


def decode_histogram(histogram, grid: OneHotGrid, bits: int | None = None):
    """Return ``(valid, points)`` of an observation histogram: which entries encode a grid point, and which point.

    ``valid`` is the Boolean mask of the histogram's stored entries, in
    stored order, that encode a grid point, and ``points`` the
    ``(entries, d)`` int64 grid-index tuples of those entries, in the same
    order. A readout up to 64 bits wide is decoded by array operations on
    ``Histogram.indices()``: the one-hot word test and local indices
    (``onehot_valid_words``, ``onehot_local_indices``) or the base-K digits
    ``(index >> j b) mod K`` of a binary index. A wider readout is decoded
    entry by entry (``decode_basis_index``). No ``2**width`` array and no
    flat index of the ``K**d`` grid is formed.
    """
    d, k = grid.num_variables, grid.num_grid_points
    if histogram.width > 64:
        decoded = [decode_basis_index(index, grid, bits) for index in histogram.index_list()]
        valid = np.array([point is not None for point in decoded], dtype=bool)
        points = np.array([point for point in decoded if point is not None], dtype=np.int64).reshape(-1, d)
        return valid, points
    words = histogram.indices()
    if bits is None:
        valid = onehot_valid_words(words, k, d)
        return valid, onehot_local_indices(words[valid], k, d).reshape(-1, d)
    mask = np.uint64(k - 1)
    points = np.stack([((words >> np.uint64(j * bits)) & mask).astype(np.int64) for j in range(d)],
                      axis=-1).reshape(-1, d)
    return np.ones(words.shape, dtype=bool), points
