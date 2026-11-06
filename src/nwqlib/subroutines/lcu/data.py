"""Coefficient amplitudes of an LCU, computed without a quantum SDK and shared by planning and analysis."""

from math import fsum, isfinite
import numpy as np
from nwqlib._validation import finite_real, integer
from nwqlib._limits import ResourceLimitError, exceeds, optional_limit
from nwqlib.operators.access import _check_bytes
from nwqlib.operators.inputs import _array_input
from nwqlib.subroutines._power_of_two import next_power_of_two as _next_power_of_two


def _lcu_prep_stage_requirements(padded_term_count, num_control_qubits):
    """Return the bytes and work of the coefficient and PREP stage of an LCU.

    With P padded addresses on a = log2(P) qubits, the stage is counted at
    an untuned 128 bytes per address and ``16 P (a + 1)`` work for the a-level
    preparation tree. The bytes cover the converted complex128 coefficients
    (16 per term) and the 80 bytes per address of the bookkeeping arrays
    (``_coefficient_bookkeeping``). ``lcu.core._admit_lcu`` adds the dense
    SELECT stage to these terms.
    """
    return 128*padded_term_count, 16*padded_term_count*(num_control_qubits+1)


def lcu_coefficient_intake(coefficients, *, system_dimension, coefficient_atol=0.0, max_bytes, max_work):
    """Check the coefficients of a gate-level LCU SELECT against the limits and return their bookkeeping.

    This is the coefficient check of ``lcu.core.prepare_lcu_gate_data``. It
    imports no quantum SDK, so LCHS planning can reject a branch-controlled
    SELECT that construction would reject. The PREP-stage formula is checked
    once, from the length of the input sequence, before the coefficients are
    converted to an array. A one-dimensional input has as many entries as
    its length, so the padded address count after conversion is the same.

    Returns:
        ``(coefficient_array, dimension, bookkeeping)``. ``bookkeeping`` is the
        tuple that :func:`_coefficient_bookkeeping` returns.
    """
    max_work = optional_limit(max_work, "max_work")
    if isinstance(coefficients, (np.ndarray, list, tuple)):
        width = max(len(coefficients)-1, 0).bit_length()
        prep_bytes, prep_work = _lcu_prep_stage_requirements(1 << width, width)
        _check_bytes(prep_bytes, max_bytes, "LCU coefficient intake arrays")
        if exceeds(prep_work, max_work):
            raise ResourceLimitError(f"LCU coefficient intake exceeds max_work: it needs {prep_work} work units, "
                                     f"max_work={max_work}")
    # Any other type is refused by the conversion, before its size is read.
    coefficient_array = _array_input(coefficients, ndim=1, max_bytes=max_bytes)
    if coefficient_array.size == 0:
        raise ValueError("LCU requires at least one coefficient")
    dimension = integer(system_dimension, "system_dimension", 1)
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("system dimension must be a positive power of two")
    bookkeeping = _coefficient_bookkeeping(coefficient_array, dimension=dimension, coefficient_atol=coefficient_atol)
    return coefficient_array, dimension, bookkeeping

def _coefficient_bookkeeping(
    coefficient_array: np.ndarray,
    *,
    dimension: int,
    coefficient_atol: float,
) -> tuple[float, np.ndarray, np.ndarray, int, int, int]:
    """Return the LCU coefficient bookkeeping shared by dense and gate-level data.

    Returns ``(coefficient_l1_norm, positive, prep_amplitudes,
    padded_term_count, num_control_qubits, num_system_qubits)``. The PREP
    amplitudes are ``sqrt(|c_j| / alpha)`` with ``alpha`` the coefficient
    1-norm (Low and Chuang, arXiv:1610.06546v3, Lemma 5, Eq. (10), p. 8),
    padded with zeros to a power-of-two address register.

    This is the one finiteness and 1-norm pass over the coefficients. The
    caller admits its arrays first, an untuned 80 bytes per padded address
    for the float64 magnitude, padded, square-root and quotient arrays and
    the complex128 amplitudes.
    """

    coefficient_atol = finite_real(coefficient_atol, "coefficient_atol")
    if coefficient_atol < 0:
        raise ValueError("coefficient_atol must be nonnegative")
    if not np.all(np.isfinite(coefficient_array)):
        raise ValueError("LCU coefficients must be finite")
    magnitudes = np.abs(coefficient_array)
    try:
        coefficient_l1_norm = fsum(magnitudes)
    except OverflowError:
        raise ValueError("LCU coefficient 1-norm must be finite") from None
    if not isfinite(coefficient_l1_norm):
        raise ValueError("LCU coefficient 1-norm must be finite")
    if coefficient_l1_norm <= coefficient_atol:
        raise ValueError("LCU coefficient 1-norm must be nonzero")

    padded_term_count = _next_power_of_two(coefficient_array.size)
    num_control_qubits = int(np.log2(padded_term_count))
    num_system_qubits = dimension.bit_length() - 1
    positive = np.zeros(padded_term_count, dtype=float)
    positive[: coefficient_array.size] = magnitudes
    # The probability ratio can underflow while its square-root amplitude is
    # representable. Taking square roots first preserves that amplitude.
    prep_amplitudes = (np.sqrt(positive) / np.sqrt(coefficient_l1_norm)).astype(complex)
    return (
        coefficient_l1_norm,
        positive,
        prep_amplitudes,
        padded_term_count,
        num_control_qubits,
        num_system_qubits,
    )

