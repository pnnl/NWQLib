"""Optional user limits and the coarse memory check of dominant arrays."""

import os

from nwqlib._validation import integer


class ResourceLimitError(ValueError):
    """A step needs more memory or work than a limit allows, whether a default limit or one you set.

    The check raises it before the step allocates or computes. A phase synthesis
    that doubles its FFT length raises it before an attempt that does not fit. It
    is a `ValueError`, so a direct caller sees the same refusal. A Method raises it
    from `plan` for its own limits, as `ApplicabilityError` is raised for a false
    premise. A refusal shows that planning stopped at this limit. It shows neither
    that more capacity will succeed nor that every remaining premise of the Method
    holds for this input.

    A Method's selection among routes or grids lets it propagate instead of
    recording a refused alternative, so capacity decides whether a Plan exists and
    never the Method's mathematical domain or which scientific choice it makes.
    [`compare`][nwqlib.scientist.compare] records it on that Method's row
    (`ComparisonRow.resource_limited`), with no Plan, and the other rows keep their
    positions and child seeds. Running such a row raises this error. A numerical or
    accuracy failure, for example a phase synthesis that fails its residual test at
    a length its own rule calls sufficient, is not a limit and still raises from
    `compare`.
    """


def exceeds(need, cap):
    """Return whether ``need`` is above ``cap``, where ``cap=None`` sets no limit."""
    return cap is not None and need > cap


def least(*caps):
    """Return the smallest of ``caps`` that are not None, or None when all are None."""
    return min((cap for cap in caps if cap is not None), default=None)


def known_sum(*counts):
    """Return the sum of ``counts``, or None when any is None, so an unknown part never leaves a partial total."""
    return None if any(count is None for count in counts) else sum(counts)


def optional_limit(value, name):
    """Return ``value`` as a positive integer, NumPy integers included and booleans excluded, or None for no limit."""
    return None if value is None else integer(value, name, 1)


def fits_power(bits, per_entry, cap):
    """Return whether ``per_entry * 2**bits <= cap``, where ``cap=None`` sets no limit.

    ``c 2**s <= L`` holds exactly when ``s < (L // c).bit_length()``, so the
    test never forms ``2**bits``, which for many qubits is itself huge.
    """
    return cap is None or bits < (cap // per_entry).bit_length()


def physical_memory_bytes():
    """Return the physical memory of this computer in bytes, or None where the platform does not report it."""
    try:
        page, pages = os.sysconf("SC_PAGE_SIZE"), os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    # sysconf returns -1 for a value the platform cannot determine.
    return page * pages if page > 0 and pages > 0 else None


def memory_limit(max_bytes):
    """Return ``max_bytes``, or the physical memory of this computer when it is None.

    The result is None only when ``max_bytes`` is None and the platform does
    not report its memory; no memory check is then made.
    """
    return physical_memory_bytes() if max_bytes is None else max_bytes


def byte_allowance(max_bytes):
    """Name the byte limit that ``max_bytes`` sets, for a refusal: its value, or the physical memory when it is None."""
    return f"max_bytes={max_bytes}" if max_bytes is not None else f"the physical memory ({physical_memory_bytes()} bytes)"


def check_memory(size, max_bytes, operation="input"):
    """Refuse an operation whose dominant arrays would exceed the memory limit, before it allocates.

    ``size`` is a coarse byte estimate of the dominant arrays alive together
    in one operation (dense matrices, states, FFT buffers, tables), meant to
    be within about a factor of two of the actual allocation. It is not a
    count of Python objects, library workspace or process memory.
    ``max_bytes=None`` uses the physical memory of this computer.
    """
    if max_bytes is not None:
        try:
            max_bytes = integer(max_bytes, "max_bytes", 1)
        except ValueError:
            raise ValueError("max_bytes must be a positive integer or None") from None
    limit = memory_limit(max_bytes)
    if limit is not None and size > limit:
        source = f"max_bytes={limit}" if max_bytes is not None else f"the physical memory ({limit} bytes)"
        raise ResourceLimitError(f"{operation} needs about {size} bytes, exceeding {source}")
