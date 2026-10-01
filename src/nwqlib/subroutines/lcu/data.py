"""SDK-free coefficient amplitudes shared by LCU selection and analysis."""

from math import fsum, isfinite
import numpy as np
from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes
from nwqlib.operators.inputs import _array_input
from nwqlib.subroutines._power_of_two import next_power_of_two as _next_power_of_two


def _lcu_prep_stage_requirements(padded_term_count, num_control_qubits):
    """Return the bytes and work of the coefficient and PREP stage of an LCU.

    With P padded addresses on a = log2(P) qubits, the stage is charged an
    untuned 128 bytes per address and ``16 P (a + 1)`` work for the a-level
    preparation tree. ``lcu.core._admit_lcu`` adds the dense SELECT stage to
    these terms.
    """
    return 128*padded_term_count, 16*padded_term_count*(num_control_qubits+1)


def lcu_coefficient_intake(coefficients, *, system_dimension, coefficient_atol=0.0, max_bytes, max_work):
    """Admit the coefficients of a gate-level LCU SELECT and return their bookkeeping.

    This is the coefficient intake of ``lcu.core.prepare_lcu_gate_data``. It
    imports no SDK, so LCHS planning can refuse a branch-controlled SELECT
    that construction would refuse. The PREP-stage law is checked from the
    container length before the coefficients are converted to an array, and
    again from the padded address count after conversion.

    Returns:
        ``(coefficient_array, dimension, bookkeeping)``. ``bookkeeping`` is the
        tuple that :func:`_coefficient_bookkeeping` returns.
    """
    integer(max_work, "max_work", 1)
    if isinstance(coefficients, (np.ndarray, list, tuple)):
        width = max(len(coefficients)-1, 0).bit_length()
        prep_bytes, prep_work = _lcu_prep_stage_requirements(1 << width, width)
        _check_bytes(prep_bytes, max_bytes, "LCU coefficient intake arrays")
        if prep_work > max_work:
            raise ValueError("LCU coefficient intake exceeds max_work")
    coefficient_array = _array_input(coefficients, ndim=1, max_bytes=max_bytes, extra_bytes_per_item=80)
    if coefficient_array.size == 0:
        raise ValueError("LCU requires at least one coefficient")
    dimension = integer(system_dimension, "system_dimension", 1)
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("system dimension must be a positive power of two")
    bookkeeping = _coefficient_bookkeeping(
        coefficient_array, dimension=dimension, coefficient_atol=coefficient_atol, max_bytes=max_bytes)
    prep_bytes, prep_work = _lcu_prep_stage_requirements(bookkeeping[3], bookkeeping[4])
    _check_bytes(prep_bytes, max_bytes, "LCU selected arrays")
    if prep_work > max_work:
        raise ValueError("LCU selected work exceeds max_work")
    return coefficient_array, dimension, bookkeeping

def _coefficient_bookkeeping(
    coefficient_array: np.ndarray,
    *,
    dimension: int,
    coefficient_atol: float,
    max_bytes: int = DEFAULT_INPUT_BYTES,
) -> tuple[float, np.ndarray, np.ndarray, int, int, int]:
    """Shared LCU coefficient bookkeeping for dense and gate-level data.

    Returns ``(coefficient_l1_norm, positive, prep_amplitudes,
    padded_term_count, num_control_qubits, num_system_qubits)``. The PREP
    amplitudes are ``sqrt(|c_j| / alpha)`` with ``alpha`` the coefficient
    1-norm (Low and Chuang, arXiv:1610.06546v3, Lemma 5, Eq. (10), p. 8),
    padded with zeros to a power-of-two address register.
    """

    # An untuned 80 bytes per padded address covers the float64 magnitude,
    # padded, square-root and quotient arrays and the complex128 amplitudes.
    _check_bytes(80 * _next_power_of_two(coefficient_array.size), max_bytes, "LCU coefficient arrays")
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
    num_system_qubits = int(np.log2(dimension))
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

