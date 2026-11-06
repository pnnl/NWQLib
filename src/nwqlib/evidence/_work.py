"""Exact scalar conversion for evidence, and the bookkeeping allowance of the byte-accounting conventions.

Certificates, outward bounds and exact checks compare their values with
thresholds, so they compute with ``fractions.Fraction`` instead of rounded
binary64. Their integers grow with the binary64 exponents and the number of
inputs, and a value saved in a record follows the record's representability
rule (``core.records.storable_integer``).
"""

from fractions import Fraction
from math import isfinite

from nwqlib.core.records import Float64, Rational

# H0 of the byte-accounting conventions, registered in ENGINEERING_CONSTANTS.md,
# "Byte-accounting conventions", with its revisit condition, and restated by the
# QHD Pauli grouping law (algorithms/qhd/resources.py). It is an engineering
# allowance for scalar bookkeeping, ndarray headers, iterators and fixed sort
# stacks on the checked 64-bit CPython/NumPy stack, not a process-RSS bound.
# Variable populations of Python objects, allocator fragmentation, library
# thread pools and native synthesis are counted separately.
BOOKKEEPING_BYTES = 65536


def exact(value):
    """Return the exact ``Fraction`` of a finite real scalar: a ``Rational``, ``Fraction``, ``Float64``, float or int.

    Raises:
        ValueError: ``value`` is not a finite real scalar.
    """
    if isinstance(value, (Rational, Fraction)):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, Float64):
        value = value.value
    if type(value) is int:
        return Fraction(value)
    if type(value) is not float or not isfinite(value):
        raise ValueError("exact arithmetic requires a finite real scalar")
    return Fraction(value)
