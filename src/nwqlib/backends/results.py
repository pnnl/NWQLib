"""Backend execution result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from nwqlib.backends.capabilities import BackendTarget
from nwqlib.execution import ExecutionMode


@dataclass(frozen=True, kw_only=True)
class BackendRunResult:
    """Native output returned by a backend adapter.

    Args:
        execution_mode: Transport/readout mode of the native execution.
        backend_target: Backend target metadata.
        raw_output: Backend-specific raw output, such as counts or statevector.
        metadata: Additional execution metadata.
    """

    execution_mode: ExecutionMode
    backend_target: BackendTarget
    raw_output: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def remap_counts(states, counts, sources):
    """Map native count outcomes to classical outcome indices and sum equal outcomes.

    Classical bit ``b`` takes native bit ``sources[b]``; a ``None`` source
    leaves it zero. Bit zero is the least significant bit on both sides, so an
    outcome integer ``j`` is the bit string ``format(j, f"0{len(sources)}b")``
    (rightmost character bit zero). Each count is checked against
    ``execution.MAX_COUNT`` while it is still a Python or provider integer,
    before the cast to int64. Their exact Python total must not exceed
    ``MAX_COUNT`` either, so no int64 sum of equal outcomes can wrap.

    Args:
        states: Native outcome integers, one per count, each below ``2**64``
            (a sequence or a ``uint64`` array).
        counts: Nonnegative integer counts.
        sources: Native bit of each classical bit, in classical order, at most
            64 classical bits.

    Returns:
        ``(indices, counts)``: strictly increasing ``uint64`` indices and
        positive ``int64`` counts of equal length, the layout
        ``ObservationChunk.from_histogram`` accepts for a width of at most 64.
        The identity map (``sources == range(len(sources))``) keeps the native
        integers and skips the bit gather.
    """
    import numpy as np
    from nwqlib.execution import MAX_COUNT
    if any(type(count) is not int or not 0 <= count <= MAX_COUNT for count in counts):
        raise ValueError(f"counts must be integers in [0, {MAX_COUNT}]")
    if sum(counts) > MAX_COUNT:
        raise ValueError(f"the counts sum to more than {MAX_COUNT}")
    indices, inverse = np.unique(gather_bits(np.asarray(states, dtype=np.uint64), sources),
                                 return_inverse=True)
    totals = np.zeros(len(indices), dtype=np.int64)
    np.add.at(totals, inverse, np.fromiter(counts, dtype=np.int64, count=len(counts)))
    keep = totals > 0
    return indices[keep], totals[keep]


def gather_bits(native, sources):
    """Classical outcome indices of native outcome integers (``uint64`` in, ``uint64`` out).

    Classical bit ``b`` takes native bit ``sources[b]`` (``None``: zero), with
    bit zero least significant on both sides, at most 64 classical bits.
    Native bits that no classical bit takes, such as unmeasured qubits of a
    provider histogram over every qubit, are dropped. The identity map keeps
    the low ``len(sources)`` native bits with one mask; any other map gathers
    one bit per classical position with vectorized shifts.
    """
    import numpy as np
    if len(sources) > 64:
        raise ValueError("count remapping supports at most 64 classical bits")
    if tuple(sources) == tuple(range(len(sources))):
        return native if len(sources) == 64 else native & np.uint64((1 << len(sources)) - 1)
    indices = np.zeros(len(native), dtype=np.uint64)
    for bit, source in enumerate(sources):
        if source is not None:
            indices |= ((native >> np.uint64(source)) & np.uint64(1)) << np.uint64(bit)
    return indices



