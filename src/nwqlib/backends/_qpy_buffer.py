"""Byte-limited in-memory output for prepared backend QPY payloads."""

from io import BytesIO

from nwqlib._limits import ResourceLimitError, byte_allowance, check_memory, exceeds, memory_limit


class QpyBuffer(BytesIO):
    """Check the buffer's next length against the limit before BytesIO allocates or copies a write.

    For cursor p, input length n and logical length L, the next length is
    ``max(L, p + n)``, because after a seek a write can end inside the buffer.
    ``limit`` None sets no limit. ``error`` is the exception type of the refusal.
    """

    def __init__(self, limit, *, message, error=ValueError):
        super().__init__()
        self.limit = limit
        self.message = message
        self.error = error
        self.logical_length = 0

    def write(self, data):
        next_length = max(self.logical_length, self.tell() + memoryview(data).nbytes)
        if exceeds(next_length, self.limit):
            raise self.error(self.message)
        self.logical_length = next_length
        return super().write(data)


def run_qpy_buffer(run):
    """Return the buffer of one QPY payload of ``run``, limited by the Method's memory limit and by the Run's data allowance.

    These are two different quantities, and each has its own refusal and remedy.

    With ``b_next`` the buffer's next length (``QpyBuffer.write``):

    Memory. A payload is held twice, in the buffer and in the bytes that ``getvalue``
    returns, so ``2 * b_next`` is an allowance for a possible copy, not an identity or
    a law of process memory. With what the Run already holds (``_held_bytes``), the
    dominant payload that the existing policy has identified, it goes against the
    Method's ``max_bytes``, or the physical memory when that is None, so
    ``max_data_bytes=None`` does not remove this check. SDK temporaries and the spare
    capacity of BytesIO are not in it.

    Stored data. ``b_next`` goes against the bytes left of ``max_data_bytes``
    (``Run._available_data_bytes``), which is never the physical memory. The buffer
    itself is transient and is not charged. All of its bytes are new durable data,
    because the payload's ``PayloadRef`` takes a new UUID as its key where the record
    is made, so the journal cannot find them already saved and nothing here refuses
    bytes that would be shared. ``Run._write`` still counts the actual bytes, with the
    rows saved beside them, when the preparation record commits.

    The two limits are independent and each is read once, for the writes of this one
    payload. Whichever is smaller is the first that a growing payload reaches, so one
    comparison per write checks both, before the buffer takes the bytes. This cannot
    stop the SDK building what it writes: Qiskit 2.6.0's ``qpy.dump`` hands over a
    whole circuit in one write, measured on circuits of 60 and 6000 gates, so on that
    version the comparison saves only the buffer's copy. A refusal comes after the
    preparation's charge, which stays counted.
    ``Run.resume(rebuild_preparations=True)`` prepares it again once the limit allows.
    """
    from nwqlib._prepared_execution import _held_bytes, _limit_remedy
    max_bytes = getattr(run.plan.method, "max_bytes", None)
    memory, held = memory_limit(max_bytes), _held_bytes(run)
    payload = None if memory is None else (memory - held) // 2
    data = run._available_data_bytes()
    if data is not None and (payload is None or data < payload):
        return QpyBuffer(data, message=(
            f"saving this prepared circuit as QPY needs more than the {data} bytes left of this Run's "
            f"max_data_bytes={run.limits.max_data_bytes}, and its preparation stays counted. "
            f"{_limit_remedy('max_data_bytes')}"))
    computer = "Reopen the Run on a computer with more memory and use run.resume(rebuild_preparations=True), or plan " \
        if max_bytes is None else "Plan "
    return QpyBuffer(payload, error=ResourceLimitError, message=(
        f"saving this prepared circuit as QPY needs more than {payload} bytes, held twice while it is written, which "
        f"with the {held} bytes this Run holds exceeds {byte_allowance(max_bytes)}, and its preparation stays counted. "
        f"{computer}again with a larger max_bytes in the Method. A Run keeps the limit of its Method, so that is a new "
        "Run, which sends its jobs again"))


def check_qpy_read(size, run):
    """Compare a saved QPY payload of ``size`` bytes with the memory limit before it is decoded.

    The saved bytes and the stream that ``qpy.load`` reads are at most one
    payload-sized copy each, an untuned allowance. With what the Run holds, this goes
    against the same limit as ``run_qpy_buffer``. It is a memory quantity, so it is
    not compared with ``max_data_bytes``, and the bytes were counted as stored data
    when they were saved.
    """
    from nwqlib._prepared_execution import _held_bytes
    check_memory(_held_bytes(run) + 2 * size, getattr(run.plan.method, "max_bytes", None), "reading a saved QPY circuit")
