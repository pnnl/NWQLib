"""Shared power-of-two integer helpers.

Leaf module: imports nothing from ``nwqlib``, so algorithms, subroutines
and backends can import it without import cycles. Shared integer tests and
padding sizes avoid inconsistent floating-point logarithm rounding.
"""

from __future__ import annotations

__all__ = ["is_power_of_two", "next_power_of_two"]


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""

    return value > 0 and (value & (value - 1)) == 0


def next_power_of_two(value: int) -> int:
    """Return the next power of two greater than or equal to ``value``."""

    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()
