"""Exact scalar arithmetic with a finite integer representation boundary.

The guard bounds prospective integer components before an operation. It is
conservative about unreduced fractions, and is not a CPU or memory guarantee.
"""

from fractions import Fraction
from math import frexp, isfinite, ldexp

from nwqlib.core.records import Float64, Rational

DEFAULT_MAX_INTEGER_BITS = 4096

# Byte-accounting convention H0 of the admission laws that name it (the energy
# endpoint capture and relation laws in energy_shift.py and the Pauli plan
# classification gate in subroutines/block_encoding/core.py). "H0=65536 is an
# explicit engineering allowance for scalar bookkeeping, ndarray headers,
# iterators and fixed sort stacks on the checked 64-bit CPython/NumPy stack.
# Variable populations of Python objects are charged separately. This is not a
# universal interpreter or process-RSS theorem. Allocator fragmentation,
# library thread pools and native synthesis have separate owners."
# Registered in ENGINEERING_CONSTANTS.md, "Byte-accounting conventions".
BOOKKEEPING_BYTES = 65536


def integer_object_bytes(bits):
    """Return L(b), a conservative byte charge for one Python integer of at most b magnitude bits.

    ``L(b) = 32 + 4 ceil(max(1, b)/30)`` on the checked 64-bit CPython stack:
    CPython uses thirty magnitude bits per four-byte digit, and the 32 covers
    its header above the observed 28-byte minimum. The byte-accounting
    conventions that use it: numerical arrays use eight-byte float64, int64,
    uint64 and intp entries, sixteen-byte complex128 entries and one-byte
    Boolean entries; counts and size products are Python integers admitted
    before allocation or intp conversion; stored payload, incremental
    workspace, peak simultaneous storage and archive size are distinct
    quantities. ``B_held`` denotes other already admitted allocations that
    remain live in the phase being checked, a shared buffer counted once; a
    gate limiting total simultaneous storage adds ``B_held`` to its local
    formula or subtracts it from the available limit, and a local function
    cannot infer unrelated live child plans from its term count.
    Registered in ENGINEERING_CONSTANTS.md, "Byte-accounting conventions".
    """
    return 32 + 4 * ((max(1, bits) + 29) // 30)


def ratio_bits(value):
    """Bound exact numerator/denominator widths without constructing a Fraction.

    Returns ``(numerator_bits, denominator_bits)`` of the reduced fraction.
    A nonzero finite binary64 value is ``s * 2**p`` with an odd integer
    significand s once the trailing zero bits of its 53-bit significand move
    into p. For p >= 0 the reduced fraction is ``(s * 2**p) / 1``, and for
    p < 0 it is ``s / 2**-p``, whose denominator has ``1 - p`` bits.
    """
    if isinstance(value, (Rational, Fraction)):
        return max(1, value.numerator.bit_length()), value.denominator.bit_length()
    if isinstance(value, Float64):
        value = value.value
    if type(value) is int:
        return max(1, value.bit_length()), 1
    if type(value) is not float or not isfinite(value):
        raise ValueError("exact arithmetic requires a finite real scalar")
    if value == 0.:
        return 1, 1
    mantissa, exponent = frexp(abs(value))
    significand = int(ldexp(mantissa, 53))
    trailing = (significand & -significand).bit_length() - 1
    power = exponent - 53 + trailing
    return significand.bit_length() - trailing + max(power, 0), 1 + max(-power, 0)


class ExactArithmetic:
    """Perform exact rational operations within max_integer_bits per component.

    Default 4096 bits covers every finite binary64 input and ordinary composed
    scalar checks. A finite binary64 value needs at most a 1024-bit numerator
    and a 1075-bit denominator (the smallest subnormal is 2**-1074). See
    ENGINEERING_CONSTANTS "Exact evidence integer representation".
    Cross products are bounded before Fraction can reduce them;
    a rejected operation may have a smaller reduced result. Raise this explicit
    representation limit only for an intended larger exact-arithmetic domain.

    With numerator and denominator widths ``(ln, ld)`` and ``(rn, rd)``, the
    unreduced results are ``a/b + c/d = (a*d + c*b) / (b*d)``, at most
    ``max(ln + rd, rn + ld) + 1`` and ``ld + rd`` bits, ``(a*c) / (b*d)``,
    at most ``ln + rn`` and ``ld + rd`` bits, and ``(a*d) / (b*c)``, at
    most ``ln + rd`` and ``ld + rn`` bits. A comparison forms the two cross
    products ``a*d`` and ``c*b``. Each method checks these widths first.
    """

    def __init__(self, *, max_integer_bits=DEFAULT_MAX_INTEGER_BITS):
        if type(max_integer_bits) is not int or max_integer_bits < 1:
            raise ValueError("max_integer_bits must be a positive integer")
        self.max_integer_bits = max_integer_bits

    def _check(self, *widths):
        if max(widths) > self.max_integer_bits:
            raise ValueError(f"exact arithmetic exceeds max_integer_bits={self.max_integer_bits} before operation")

    def fraction(self, value):
        self._check(*ratio_bits(value))
        if isinstance(value, Rational):
            return Fraction(value.numerator, value.denominator)
        return Fraction(value.value if isinstance(value, Float64) else value)

    def add(self, left, right):
        ln, ld = ratio_bits(left)
        rn, rd = ratio_bits(right)
        self._check(max(ln + rd, rn + ld) + 1, ld + rd)
        return left + right

    def subtract(self, left, right):
        ln, ld = ratio_bits(left)
        rn, rd = ratio_bits(right)
        self._check(max(ln + rd, rn + ld) + 1, ld + rd)
        return left - right

    def multiply(self, left, right):
        ln, ld = ratio_bits(left)
        rn, rd = ratio_bits(right)
        self._check(ln + rn, ld + rd)
        return left * right

    def divide(self, left, right):
        ln, ld = ratio_bits(left)
        rn, rd = ratio_bits(right)
        self._check(ln + rd, ld + rn)
        return left / right

    def le(self, left, right):
        ln, ld = ratio_bits(left)
        rn, rd = ratio_bits(right)
        self._check(ln + rd, rn + ld)
        return left <= right

    def absolute(self, value):
        self._check(*ratio_bits(value))
        return abs(value)

    def integer_product(self, left, right):
        if type(left) is not int or type(right) is not int:
            raise TypeError("integer_product requires exact integers")
        self._check(max(1, left.bit_length()) + max(1, right.bit_length()))
        return left * right
