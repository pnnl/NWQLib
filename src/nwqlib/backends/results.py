"""Backend execution result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from nwqlib.backends.capabilities import BackendTarget
from nwqlib.execution import ExecutionMode


@dataclass(frozen=True, kw_only=True)
class BackendRunResult:
    """Raw output returned by a backend adapter.

    Args:
        execution_mode: Transport and readout mode of the execution.
        backend_target: Backend target metadata.
        raw_output: Backend-specific raw output, such as counts or statevector.
        metadata: Additional execution metadata.
    """

    execution_mode: ExecutionMode
    backend_target: BackendTarget
    raw_output: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def remap_counts(states, counts, sources):
    """Map the backend's count outcomes to classical outcome indices and sum equal outcomes.

    Classical bit ``b`` takes backend bit ``sources[b]``, and a ``None`` source
    leaves it zero. Bit zero is the least significant bit on both sides, so an
    outcome integer ``j`` is the bit string ``format(j, f"0{len(sources)}b")``
    (rightmost character bit zero). The caller reads the provider's counts
    and checks them first. They are nonnegative integers whose total is at
    most ``execution.MAX_COUNT``, as ``IonQBackend._decode`` checks against
    the job's shots, so their int64 sums cannot wrap.

    Args:
        states: Backend outcome integers, one per count, each below ``2**64``
            (a sequence or a ``uint64`` array).
        counts: The checked nonnegative integer counts (Python or NumPy integers).
        sources: Backend bit of each classical bit, in classical order, at most
            64 classical bits.

    Returns:
        ``(indices, counts)``: strictly increasing ``uint64`` indices and
        positive ``int64`` counts of equal length, the layout
        ``ObservationChunk.from_histogram`` accepts for a width of at most 64.
        The identity map (``sources == range(len(sources))``) keeps the backend
        integers and skips the bit gather.
    """
    import numpy as np
    indices, inverse = np.unique(gather_bits(np.asarray(states, dtype=np.uint64), sources),
                                 return_inverse=True)
    totals = np.zeros(len(indices), dtype=np.int64)
    np.add.at(totals, inverse, np.asarray(counts, dtype=np.int64))
    keep = totals > 0
    return indices[keep], totals[keep]


def gather_bits(native, sources):
    """Classical outcome indices of backend outcome integers (``uint64`` in, ``uint64`` out).

    Classical bit ``b`` takes backend bit ``sources[b]`` (``None``: zero), with
    bit zero least significant on both sides, at most 64 classical bits.
    Backend bits that no classical bit takes, such as unmeasured qubits of a
    provider histogram over every qubit, are dropped. The identity map keeps
    the low ``len(sources)`` backend bits with one mask. Any other map gathers
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



