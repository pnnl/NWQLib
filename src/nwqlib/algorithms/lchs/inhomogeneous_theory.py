"""Constant-source LCHS numerical kernels and explicit remainder bounds."""

from __future__ import annotations

from nwqlib._limits import DEFAULT_MAX_BYTES

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from nwqlib._linalg_laws import singular_values_work
from nwqlib._validation import integer
from nwqlib.algorithms.lchs.native import (
    resolve_hamiltonian_evolution_backend,
)
from nwqlib.algorithms.lchs.time_independent_common import validate_final_time
from nwqlib.algorithms.lchs.time_independent_terms import (
    LCHSQuadratureData,
    generate_lchs_quadrature,
    lchs_quadrature_summary,
)
from nwqlib.algorithms.lchs.providers import (
    LCHS_PROVIDER_RECORD_KEYS,
)
from nwqlib._numerics import componentwise_exact_zero, stable_vector_norm

if TYPE_CHECKING:
    from nwqlib.algorithms.lchs.method import LCHS


@dataclass(frozen=True, kw_only=True)
class _KernelContext:
    """Prepared k-kernel quadrature data for elapsed-time propagators.

    Attributes:
        final_time: Final time T for which the k-grid was selected.
        quadrature: Shared quadrature summary, including psd_shift.
    """

    final_time: float
    quadrature: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class _DuhamelBoundEvaluation:
    """Duhamel remainder bound and whether it may be used.

    Attributes:
        value: Physical L2 bound on the source-time quadrature error. It is
            nan when the norm of a nonzero A or b underflows to zero, and inf
            when a norm or the bound overflows.
        unusable_nonzero: True when the true remainder may be nonzero but the
            evaluated value is nan, inf or rounded to zero.
    """

    value: float
    unusable_nonzero: bool


def duhamel_quadrature(final_time: float, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gaussian-Legendre nodes and weights for the source-time integral.

    One Gauss-Legendre panel on [0, T] discretizes the Duhamel integral of
    ACL arXiv:2312.03916v2 Eq. (2), the single-panel case of Eq. (72). The
    rule is exact for polynomials of degree 2*node_count-1, and its
    remainder is bounded by _duhamel_quadrature_error_bound.
    """

    final_time = validate_final_time(final_time)
    node_count = integer(node_count, "node_count", 1)
    nodes, weights = np.polynomial.legendre.leggauss(node_count)
    mapped_nodes = 0.5 * final_time * (nodes + 1.0)
    mapped_weights = 0.5 * final_time * weights
    return mapped_nodes, mapped_weights


def _scaling_safe_spectral_norm(matrix: np.ndarray) -> float:
    """Evaluate a matrix spectral norm without squaring at the input scale."""

    array = np.asarray(matrix, dtype=complex)
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scaled_norm = float(np.linalg.norm(array / maximum, ord=2))
        return float(maximum * scaled_norm)


def _duhamel_quadrature_error_bound(
    *,
    matrix: np.ndarray,
    source_term: np.ndarray,
    final_time: float,
    node_count: int,
    lambda_min_before_psd_conversion: float,
    matrix_norm: float | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_dense_work: int = 100_000_000,
) -> _DuhamelBoundEvaluation:
    """Evaluate the constant-source Gauss-Legendre remainder in log form.

    DLMF 3.5.19 gives the m-point Gauss remainder gamma_m*f**(2m)(xi)/(2m)!,
    and DLMF 3.5.21 gives gamma_m = 2**(2m+1)*(m!)**4/((2m+1)*((2m)!)**2) for
    Legendre on [-1,1]. The map s = T*(x+1)/2 multiplies the remainder by
    (T/2)**(2m+1). Together they give
    T**(2m+1)*(m!)**4/((2m+1)*((2m)!)**3) times sup||f**(2m)||.
    For ACL arXiv:2312.03916v2 Eq.(2), f(s)=exp(-A*(T-s))*b.
    ||f**(2m)|| <= ||A||**(2m)*exp(max(0,-lambda_min(L))*T)*||b||, because
    ||exp(-A*t)|| <= ||exp(-L*t)|| (ACL Lemma 21, Eq. (162)). lambda_min is
    taken before the PSD shift, so the bound describes the physical growth.
    For complex vector error E, choose v=E/||E|| and the real scalar function
    Re(v†f). Its scalar remainder equals ||E||, with derivatives bounded by
    ||f**(2m)||; no dimension factor or common complex mean-value point is
    assumed. This is a numerical
    evaluation of the ideal bound, not an outward-rounded interval.
    """

    from nwqlib.operators.access import _check_bytes
    final_time = validate_final_time(final_time)
    node_count = integer(node_count, "node_count", 1)
    max_dense_work = integer(max_dense_work, "max_dense_work", 0)
    d = len(source_term)
    # Bytes: 64*d*d, four complex128 d-by-d arrays, covers the complex copy,
    # its absolute values, the scaled copy and the SVD input of
    # _scaling_safe_spectral_norm. 64*d covers four complex128 d-vectors.
    _check_bytes(64*d*d+64*d, max_bytes, "Duhamel remainder arrays")
    # Work: 8*d for the stable source norm, 32 for the scalar log formula,
    # and, when ||A|| is not supplied, the singular values of
    # _scaling_safe_spectral_norm (_linalg_laws.singular_values_work) plus
    # 4*d**2 entrywise.
    work = 8*d+32+(singular_values_work(d)+4*d*d if matrix_norm is None else 0)
    if work > max_dense_work:
        raise ValueError("Duhamel remainder exceeds max_dense_work before norm work")

    if final_time == 0 or componentwise_exact_zero(matrix) or componentwise_exact_zero(source_term):
        return _DuhamelBoundEvaluation(value=0.0, unusable_nonzero=False)

    if matrix_norm is None:
        matrix_norm = _scaling_safe_spectral_norm(matrix)
    source_norm = stable_vector_norm(source_term)
    if matrix_norm <= 0.0 or source_norm <= 0.0:
        return _DuhamelBoundEvaluation(value=float("nan"), unusable_nonzero=True)
    if not isfinite(matrix_norm) or not isfinite(source_norm):
        return _DuhamelBoundEvaluation(value=float("inf"), unusable_nonzero=True)

    rho_plus = max(0.0, -float(lambda_min_before_psd_conversion))
    m = int(node_count)
    log_bound = (
        (2 * m + 1) * log(float(final_time))
        + 4.0 * lgamma(m + 1)
        - log(2 * m + 1)
        - 3.0 * lgamma(2 * m + 1)
        + 2.0 * m * log(matrix_norm)
        + rho_plus * float(final_time)
        + log(source_norm)
    )
    try:
        value = float(exp(log_bound))
    except OverflowError:
        value = float("inf")
    return _DuhamelBoundEvaluation(
        value=value,
        unusable_nonzero=not isfinite(value) or value == 0.0,
    )


def _kernel_context(
    *,
    matrix: np.ndarray | None = None,
    final_time: float,
    method: LCHS,
    _quadrature_plan: LCHSQuadratureData | None = None,
) -> _KernelContext:
    """Reuse one selected k-kernel quadrature for all host elapsed-time actions.

    The grid is selected for the full time T. The tail bounds do not depend
    on time, and the quadrature conditions (the ATAP ISBN 978-1-61197-239-9
    ellipse bound and the Low-Somma arXiv:2508.19238v2 step condition) only
    relax as T*||L|| decreases, so the same grid keeps its recorded bounds
    for every shorter Duhamel elapsed time T-s. ``matrix`` is needed only
    when no selected quadrature plan is supplied.
    """

    he_backend = resolve_hamiltonian_evolution_backend(method)
    if matrix is None and _quadrature_plan is None:
        raise ValueError("the kernel context needs a matrix or a selected quadrature plan")
    quadrature_data = _quadrature_plan or generate_lchs_quadrature(
        matrix=matrix, final_time=final_time, method=method,
        max_bytes=method.max_bytes, max_spectral_work=method.max_spectral_work,
        max_quadrature_work=method.max_quadrature_work,
    )
    shared_quadrature = lchs_quadrature_summary(
        quadrature_data,
        method=method,
        he_backend=he_backend,
        final_time=final_time,
    )
    quadrature: dict[str, Any] = {
        "beta": shared_quadrature["beta"],
        "epsilon": shared_quadrature["epsilon"],
        "truncation_multiplier": shared_quadrature["truncation_multiplier"],
        "K": shared_quadrature["K"],
        "effective_K": shared_quadrature["effective_K"],
        "h1": shared_quadrature["h1"],
        "Q": shared_quadrature["Q"],
        "M": shared_quadrature["M"],
        "interval_count_each_side": shared_quadrature["interval_count_each_side"],
        "coefficient_l1_norm": shared_quadrature["coefficient_l1_norm"],
        "max_lcu_coefficient_l1_norm": shared_quadrature["lcu_coefficient_l1_norm"],
        "matrix_norm": shared_quadrature["matrix_norm"],
        "prepared_matrix_norm": shared_quadrature["prepared_matrix_norm"],
        "l_norm": shared_quadrature["l_norm"],
        "h_norm": shared_quadrature["h_norm"],
        "hamiltonian_evolution_backend": shared_quadrature["hamiltonian_evolution_backend"],
        "trotter_steps": shared_quadrature["trotter_steps"],
        "max_operator_scale": shared_quadrature["operator_scale"],
        "min_l_eigenvalue_before_psd_conversion": shared_quadrature[
            "min_l_eigenvalue_before_psd_conversion"
        ],
        "min_l_eigenvalue_after_psd_conversion": shared_quadrature[
            "min_l_eigenvalue_after_psd_conversion"
        ],
        "psd_correction_applied": shared_quadrature["psd_correction_applied"],
        "psd_shift": shared_quadrature["psd_shift"],
        "make_l_psd": shared_quadrature["make_l_psd"],
        "psd_tolerance": shared_quadrature["psd_tolerance"],
        "approximate_lchs_error_bound": shared_quadrature["approximate_lchs_error_bound"],
        "quadrature_error_bound": shared_quadrature["quadrature_error_bound"],
        **{
            key: shared_quadrature[key]
            for key in LCHS_PROVIDER_RECORD_KEYS
            if key != "quadrature_error_bound"
        },
    }
    return _KernelContext(final_time=final_time, quadrature=quadrature)


__all__ = ["duhamel_quadrature"]
