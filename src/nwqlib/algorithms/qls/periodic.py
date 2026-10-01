"""Compact QLS for the positive periodic stencil ``A = m I + d (2I - S - S^dagger)``.

``S`` is the cyclic shift on ``2**q`` sites. The banded block encoding
represents ``A`` from its three coefficients, so planning never forms a
dense matrix and stays compact for wide metadata-only inputs.
"""

from dataclasses import replace
from fractions import Fraction
from math import inf, nextafter
from nwqlib._validation import finite_real
from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.blocks.encoding import _select_planned_encoding
from .constants import QLS_POLYNOMIAL_KAPPA_FLOOR


def _ceil_float(value):
    """Return the smallest binary64 value that is not below the exact rational ``value``."""
    try:
        result = finite_real(float(value), "periodic scalar output")
    except OverflowError as error:
        raise ValueError("periodic QLS requires representable scalar outputs") from error
    if Fraction.from_float(result) < value:
        result = finite_real(nextafter(result, inf), "outward periodic scalar output")
    return result


def select_periodic_encoding(operator, method):
    """Select the compact positive periodic encoding with an outward-covered encoded gap.

    NWQLib's derivation. For ``m > 0`` and ``d >= 0`` the exact spectrum is
    ``m + 2d (1 - cos(2 pi k / 2**q))``, so ``sigma_min = m`` and
    ``sigma_max = m + 4d``. For one qubit ``S = S^dagger = X`` and the two
    wrapped edges merge into ``(m + 2d) I - 2d X``. The encoding stores the
    rounded diagonal ``c0 = fl(m + 2d)`` and reports a rounded
    normalization ``alpha``. In exact rational arithmetic on those stored
    floats, ``|c0 - m - 2d| + |alpha - c0 - 2d|`` bounds
    ``||A - alpha * block||`` and becomes the encoding error bound, and
    ``c0 - 2d - |alpha - c0 - 2d|`` bounds the smallest eigenvalue of
    ``alpha * block`` from below. The polynomial domain parameter is then
    ``alpha / min(m, that gap)``, rounded outward, so the polynomial covers
    both the original spectrum and the actual encoded gap. The reported
    ``kappa_be`` stays ``alpha / m`` and the condition number
    ``(m + 4d) / m``. A supplied ``kappa`` must cover the same requirement.
    """
    from nwqlib.subroutines.block_encoding.banded import BandSpecification
    from nwqlib.subroutines.block_encoding.core import plan_block_encoding

    parameters = operator.periodic_stencil()
    q, m, d = parameters.num_qubits, parameters.mass, parameters.diffusion
    if parameters.potential != 0:
        raise ApplicabilityError(
            "compact QLS supports the Hermitian positive periodic family with zero potential"
        )
    if m == 0:
        raise ValueError("periodic QLS requires a positive original mass and encoded gap")
    if method.block_encoding_implementation not in ("auto", "banded"):
        raise ApplicabilityError("compact periodic QLS requires its banded encoding")
    # This function holds at most three bands and a few Fractions of binary64
    # values, whose size does not depend on q. The work that grows with q (the
    # integer 2**q and the per-band sums over q system qubits of the address
    # correction phases) happens inside plan_block_encoding, which admits it
    # with the band-table byte and work laws of _banded_plan before doing it.
    c0 = finite_real(m + 2 * d, "periodic diagonal coefficient")
    if d == 0:
        offsets, coefficients = (0,), (m,)
    elif q == 1:
        offsets, coefficients = (0, 1), (c0, -2 * d)
    else:
        offsets, coefficients = (-1, 0, 1), (-d, c0, -d)
    selected = plan_block_encoding(
        BandSpecification(offsets=offsets, coefficients=coefficients, num_qubits=q),
        implementation="banded",
        max_bytes=method.max_bytes, max_work=method.max_work,
    )
    alpha = finite_real(selected.alpha, "periodic normalization")
    if method.alpha != "auto" and method.alpha != alpha:
        raise ValueError("periodic encoding differs from requested alpha")
    # Use exact rational values of the stored floats to account for diagonal
    # and normalization rounding before claiming a positive encoded gap.
    mass, diffusion, diagonal, normalization = (
        Fraction.from_float(float(value)) for value in (m, d, c0, alpha)
    )
    twice = 2 * diffusion
    delta = diagonal - mass - twice
    normalization_error = abs(normalization - diagonal - twice)
    error = abs(delta) + normalization_error
    gap = diagonal - twice - normalization_error
    if gap <= 0:
        raise ValueError("selected periodic encoding has no representable positive gap")
    required = normalization / min(mass, gap)
    actual = alpha / m
    if method.kappa == "auto":
        kappa, polynomial, kappa_source = (
            actual,
            max(_ceil_float(required), QLS_POLYNOMIAL_KAPPA_FLOOR),
            "analytic_periodic",
        )
    else:
        if Fraction.from_float(float(method.kappa)) < required:
            raise ValueError(
                "supplied kappa does not cover the original spectrum and actual encoded gap"
            )
        kappa, polynomial, kappa_source = (
            method.kappa,
            max(method.kappa, QLS_POLYNOMIAL_KAPPA_FLOOR),
            "user",
        )
    selected = replace(selected, error_bound=_ceil_float(error))
    upper = _ceil_float(mass + 4 * diffusion)
    spectrum = dict(
        alpha=alpha,
        alpha_source="selected_encoding",
        kappa_be=kappa,
        polynomial_kappa=polynomial,
        kappa_source=kappa_source,
        condition_number=upper / m,
        sigma_min=m,
        sigma_max=upper,
        spectral_method="analytic_periodic",
        spectral_calls=(),
    )
    return _select_planned_encoding("base_encoding", selected, operator=operator), spectrum
