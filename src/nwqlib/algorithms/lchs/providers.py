"""Kernel, quadrature, and pair-resolution providers for LCHS.

ACL denotes An, Childs, and Lin, "Quantum algorithm for linear non-unitary
dynamics with near-optimal dependence on all parameters", arXiv:2312.03916,
https://arxiv.org/abs/2312.03916. ACL equation numbers in this module refer to
arXiv:2312.03916v2. Low-Somma denotes Low and Somma, "Optimal quantum
simulation of linear non-unitary dynamics", arXiv:2508.19238v2. ATAP denotes
Trefethen, Approximation Theory and Approximation Practice,
ISBN 978-1-61197-239-9, Chapter 19 (https://www.chebfun.org/ATAP/chap19.m).

A provider pair turns an LCHS representation of exp(-A*T) as an integral of
g(k)*exp(-i*T*(k*L+H)) over real k, with A = L + iH, into finite real nodes
k_j and complex coefficients c_j. The representation is exact for the ACL
kernels (Eqs. (6) and (60)) and approximate for Low-Somma (their Eq. (8)).
The pair owns two error components, the kernel-integral error and the finite
k-quadrature error. For Eq. (7) the kernel-integral error is the tail beyond
the cutoff. For Low-Somma it is epsilon_lchs of their Theorem 2, which also
covers the approximate representation and so is not a pure tail. Branch
synthesis, state preparation and physical recovery have separate owners and
separate bounds.

The default pair is the Eq. (7) kernel with beta = 0.75 and composite Gauss
quadrature. Its construction tolerance epsilon is split equally between two
components. The cutoff K makes the tail bound from Eq. (186) at most
epsilon/2, and the Gauss panels make the ATAP Theorem 19.3 quadrature bound
at most epsilon/2. These two numbers bound only the kernel-integral and k-quadrature
components of the operator error. They are not a bound on the physical
solution error, which solution_error_budget assembles from further stages.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import asinh, ceil, cos, exp, expm1, fsum, isfinite, isnan, lgamma, log, nextafter, pi, sqrt
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from operator import index as integer_index

from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes
from nwqlib.algorithms.lchs.provider_config import (
    ProviderConfig,
    ProviderParameter,
    ResolvedProviderConfig,
)


LOW_SOMMA_FORMULA_ID = "low_somma_fhat_2_1_v1"
LOW_SOMMA_PROFILE_ID = "paper_closed_form_f2_1"
LOW_SOMMA_UNBUDGETED_STAGES = (
    "trotter_synthesis",
    "lcu_coefficient_preparation",
    "initial_state_preparation",
)

# The Eq. (60) integrand g(k) has a pole at k = -i, from 1/(1 - i*k), and the
# branch point of (1 + i*k)**beta at k = i, so it is analytic on |Im k| < 1.
# The Gauss panel bound uses the closed strip |Im k| <= 1 - d, on which
# |g(k)| <= 1/(C_beta*d) (see _ellipse_rule). The margin d is an engineering
# choice registered in ENGINEERING_CONSTANTS, not a fit to observed errors.
# A smaller d widens the Bernstein ellipse but raises the bound on |g|.
LCHS_ELLIPSE_STRIP_MARGIN = .1
# Scalar operations charged for each panel-count candidate in _ellipse_rule.
# They cover a few logarithms, one asinh, one ceil and at most two bound
# evaluations. Registered in ENGINEERING_CONSTANTS.
_ELLIPSE_CANDIDATE_WORK = 32
# Largest fraction of the construction tolerance that the binary64 rounding
# scale u*alpha of the finite coefficient sum may reach
# (_admit_coefficient_rounding). It equals the allowance that the Method
# gives each realization stage between the finite sum and the returned
# vector (LCHS_QSP_EPSILON_FRACTION and LCHS_TROTTER_EPSILON_FRACTION in
# native.py), and binary64 evaluation is one more such stage. Registered in
# ENGINEERING_CONSTANTS.
LCHS_ROUNDING_EPSILON_FRACTION = .1
_UNIT_ROUNDOFF = 2.0**-53  # binary64 unit roundoff u


def _immutable_map(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: values[key] for key in sorted(values)})


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze JSON-like metadata without changing key order."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _detached_plain(value: Any) -> Any:
    """Return detached plain dictionaries and lists for serialization."""

    if isinstance(value, Mapping):
        return {key: _detached_plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_detached_plain(item) for item in value]
    return value


KernelResolver = Callable[[Mapping[str, ProviderParameter]], Mapping[str, ProviderParameter]]
KernelCoefficient = Callable[[float, Mapping[str, Any]], complex]


@dataclass(frozen=True, kw_only=True)
class LCHSKernelProvider:
    """One registered LCHS coefficient-kernel implementation.

    Attributes:
        name: Registry key accepted in the kernel slot of a ProviderConfig.
        formula_id: Versioned identifier recorded with the selected coefficient formula.
        resolve_parameters: Callable validating explicit parameter values and returning
            the complete resolved parameter mapping.
        coefficient: Callable evaluating the complex coefficient kernel at one real k
            using the resolved pair context; quadrature weights are multiplied separately.
        fixed_parameters: Read-only formula constants, such as Low-Somma j=2 and y=1,
            which are not user-adjustable provider parameters.
    """

    name: str
    formula_id: str
    resolve_parameters: KernelResolver
    coefficient: KernelCoefficient
    fixed_parameters: Mapping[str, ProviderParameter]

    def __post_init__(self) -> None:
        if not self.name or not self.formula_id:
            raise ValueError("kernel provider name and formula_id must be non-empty")
        object.__setattr__(
            self,
            "fixed_parameters",
            _immutable_map(self.fixed_parameters),
        )


@dataclass(frozen=True, kw_only=True)
class LCHSProblemContext:
    """Problem scalars available to provider planning.

    Attributes:
        final_time: Evolution duration T in the units paired with A in exp(-T A).
        epsilon: Requested finite-construction tolerance. The default Gauss selector
            splits it equally between kernel truncation and quadrature. Other
            physical-output error contributions have separate owners.
        l_norm: Numerical spectral-norm estimate of the prepared Hermitian L used in
            exp(-i T (k L + H)); T*l_norm is dimensionless.
    """

    final_time: float
    epsilon: float
    l_norm: float


@dataclass(frozen=True, kw_only=True)
class LCHSQuadraturePlan:
    """Immutable node, weight, and address data from a quadrature provider.

    Component bounds have the selected provider pair's scope. They are not a
    bound on state preparation, Hamiltonian synthesis or total physical output.

    Attributes:
        nodes: Ordered real k values for the mathematical quadrature, without LCU
            address padding. This order also indexes base_weights.
        base_weights: Real integration weights before multiplication by the kernel.
        range_k: Selected finite cutoff scale K, or the signed-grid maximum magnitude.
        effective_range_k: Cutoff used by this realized rule and its component-bound
            calculation; it is not necessarily the largest interior Gauss node.
        interval_count_each_side: Gauss panel count on each side of zero; the symmetric
            uniform rule uses its integer half-range J, and signed/unitary rules use zero.
        node_count: Gauss-Legendre points per panel for composite Gauss; total points
            for the signed or symmetric-uniform rule, and one for unitary reduction.
        node_order: Description of the actual node-to-address ordering, including
            computational-basis wraparound for a signed two's-complement grid.
        address_structure: Frozen structural data used by SELECT admission and cost
            laws, such as affine bit coefficients or an explicit-table declaration.
        physical_node_count: Number of mathematical nodes, excluding address padding.
        padding_count: Padding explicitly reported by this provider. Zero does not
            mean a later power-of-two LCU realization needs no additional slots.
        approximate_lchs_error_bound: Available kernel-integral approximation bound;
            for the default Eq.(7) pair this is the finite-cutoff tail bound. None
            means unavailable, while unitary reduction has an algebraic zero.
        quadrature_error_bound: Available finite quadrature component bound under the
            selected pair's premises; None is unavailable, not zero or a failure verdict.
        h1: Actual composite-Gauss panel width in k. None for rules with no Gauss panels,
            including the exact single-unitary reduction.
    """

    nodes: tuple[float, ...]
    base_weights: tuple[float, ...]
    range_k: float
    effective_range_k: float
    interval_count_each_side: int
    node_count: int
    node_order: str
    address_structure: Mapping[str, Any]
    physical_node_count: int
    padding_count: int
    approximate_lchs_error_bound: float | None
    quadrature_error_bound: float | None
    h1: float | None = None  # Actual Gauss panel width; other rules have no Gauss panels.

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                field_name,
                _deep_freeze(getattr(self, field_name)),
            )


QuadratureResolver = Callable[[Mapping[str, ProviderParameter]], Mapping[str, ProviderParameter]]
QuadratureBuilder = Callable[
    [LCHSProblemContext, ResolvedProviderConfig, Mapping[str, Any]],
    LCHSQuadraturePlan,
]


@dataclass(frozen=True, kw_only=True)
class LCHSQuadratureProvider:
    """One registered LCHS k-quadrature implementation.

    Attributes:
        name: Registry key accepted in the quadrature slot of a ProviderConfig.
        formula_id: Versioned identifier for the selected node/weight construction.
        resolve_parameters: Callable validating the requested parameters and filling
            the defaults applicable to this quadrature rule.
        build_nodes_and_weights: Callable receiving problem scalars, resolved settings
            and kernel-pair context and returning the actual LCHSQuadraturePlan.
    """

    name: str
    formula_id: str
    resolve_parameters: QuadratureResolver
    build_nodes_and_weights: QuadratureBuilder

    def __post_init__(self) -> None:
        if not self.name or not self.formula_id:
            raise ValueError("quadrature provider name and formula_id must be non-empty")


@dataclass(frozen=True, kw_only=True)
class LCHSPairCompatibility:
    """Compatibility and certification policy for one registered pair.

    Attributes:
        allowed: Whether the registered kernel/quadrature combination may be selected.
        certificate_status: Pair-scope status: certified, uncertified, rejected or
            incomplete; exact_algebraic identifies the no-k-integral unitary reduction.
            None of these by itself certifies a complete physical solution.
        reason: Explanation of the allowed domain or absent/rejected pair guarantee.
        unbudgeted_error_stages: Named approximation stages absent from this pair's
            budget, such as Trotter synthesis or state preparation.
    """

    allowed: bool
    certificate_status: str
    reason: str
    unbudgeted_error_stages: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class LCHSCoefficientPlan:
    """Nodes, weights and coefficients of the finite LCHS sum, shared by every backend.

    [`resolve_lchs_coefficient_plan`][nwqlib.algorithms.lchs.providers.resolve_lchs_coefficient_plan]
    returns it. The finite sum is `sum_j c_j exp(-i T (k_j L + H)) u0`
    over the `nodes` k_j and `coefficients` c_j. The fields below are
    read-only.

    Attributes:
        requested_lchs_kernel: Kernel choice as given, showing which
            parameters the caller supplied.
        resolved_lchs_kernel: Kernel used, with its defaults filled in.
        requested_k_quadrature: Quadrature choice as given.
        resolved_k_quadrature: Quadrature used, or the algebraic reduction
            for an exactly zero L.
        lchs_kernel_formula_id: Versioned identifier of the kernel formula.
        k_quadrature_formula_id: Versioned identifier of the node and weight
            formula.
        lchs_kernel_fixed_parameters: Constants of the kernel formula that
            are not user parameters, such as Low–Somma `j=2` and `y=1`.
        kernel_parameter_selection: Where each parameter value came from,
            normally `"user"` or `"provider_default"`, with a profile
            identifier where one applies.
        derived_parameters: Quantities derived for the pair, such as the
            Low–Somma `gamma`, `R`, `h` and `J`. A problem without a source
            also records its PSD-shift compensation here.
        nodes: Ordered real k values, as in `quadrature`.
        base_weights: Integration weights before multiplication by the
            kernel.
        coefficients: Complex coefficients c_j in node order. For a problem
            without a source they include the PSD growth factor
            `exp(shift*T)`, divided by `2**e` when the factor exceeds
            binary64, and the physical recovery carries `2**e`. Their phases
            are applied in the SELECT circuit.
        coefficient_l1_norm: `sum_j |c_j|`, the normalization of the
            coefficient state. It is neither the physical output norm nor an
            error estimate.
        quadrature: Node and address data of the rule, with its component
            bounds `approximate_lchs_error_bound` (kernel integral, the
            cutoff tail for the default kernel) and `quadrature_error_bound`,
            each for a unit input before PSD growth. `None` means
            unavailable, not zero.
        compatibility: Pair status, `certificate_status` `"certified"`,
            `"uncertified"`, `"incomplete"` or `"exact_algebraic"` for the
            zero-L reduction, with its `reason` and `unbudgeted_error_stages`.
            None of these alone certifies a complete physical solution.
        mps_decomposition: Attached TT-SVD (MPS) decomposition of the
            normalized, padded coefficient-state amplitudes, or `None`.
            `decompose_mps` reuses it only for the same tensor ordering and
            truncation settings.
    """

    requested_lchs_kernel: ProviderConfig
    resolved_lchs_kernel: ResolvedProviderConfig
    requested_k_quadrature: ProviderConfig
    resolved_k_quadrature: ResolvedProviderConfig
    lchs_kernel_formula_id: str
    k_quadrature_formula_id: str
    lchs_kernel_fixed_parameters: Mapping[str, ProviderParameter]
    kernel_parameter_selection: Mapping[str, str]
    derived_parameters: Mapping[str, ProviderParameter]
    nodes: tuple[float, ...]
    base_weights: tuple[float, ...]
    coefficients: tuple[complex, ...]
    coefficient_l1_norm: float
    quadrature: LCHSQuadraturePlan
    compatibility: LCHSPairCompatibility
    mps_decomposition: object | None = None

    def prep_amplitudes(self, *, max_bytes=DEFAULT_INPUT_BYTES):
        """Return the coefficient-state amplitudes `sqrt(|c_j|/alpha)`, zero-padded.

        alpha is `coefficient_l1_norm`, and the array length is the next
        power of two at or above `len(coefficients)`. The complex phases of
        c_j are not in this state. The SELECT circuit applies them (An,
        Childs and Lin, arXiv:2312.03916v2, Lemma 24, Eq. (178), with the
        phases moved from the two preparation oracles into SELECT).

        Args:
            max_bytes (int): Default 10 GB (decimal, `10_000_000_000`
                bytes). Upper limit on the arrays, 96 bytes per padded
                address, checked first.

        Returns:
            amplitudes (numpy.ndarray): Read-only amplitudes, one per padded
                address.
        """
        from nwqlib.subroutines.lcu.data import _coefficient_bookkeeping
        from nwqlib.operators.inputs import _freeze_array
        # Per padded slot: 16 bytes for the complex128 coefficient copy and the
        # 80 bytes that _coefficient_bookkeeping admits for its own arrays.
        _check_bytes(96 * (1 << max(0, len(self.coefficients)-1).bit_length()), max_bytes,
                     "coefficient PREP arrays")
        values = np.asarray(self.coefficients, dtype=np.complex128)
        _, _, amplitudes, _, _, _ = _coefficient_bookkeeping(
            values, dimension=1, coefficient_atol=0, max_bytes=max_bytes)
        return _freeze_array(amplitudes)

    def decompose_mps(self, max_bond_dim=None, threshold=1e-14, *, reuse=None,
                      max_bytes=DEFAULT_INPUT_BYTES, max_svd_work=100_000_000):
        """Return the TT-SVD (MPS) decomposition of the `prep_amplitudes` tensor.

        A stored or supplied decomposition is reused only when it was made
        from this exact normalized tensor with the same `max_bond_dim` and
        `threshold`. Otherwise a new TT-SVD runs under `max_bytes` and
        `max_svd_work`. No LCHS solve or circuit is run.

        Args:
            max_bond_dim (int | None): Upper limit on the kept bond
                dimension. `None` means no limit.
            threshold (float): Singular-value truncation threshold.
            reuse (MPSDecomposition | None): Decomposition to reuse instead
                of the attached `mps_decomposition`.
            max_bytes (int): Default 10 GB (decimal, `10_000_000_000`
                bytes). Upper limit on the known arrays.
            max_svd_work (int): Upper limit on the counted TT-SVD work.

        Returns:
            decomposition (MPSDecomposition): The TT cores, bond dimensions
                and discarded weight, the sum of the squared singular values
                discarded.

        Raises:
            TypeError: If `reuse` is not an `MPSDecomposition`.
            ValueError: If `reuse` does not match the tensor or the settings,
                instead of being replaced silently.
        """
        from nwqlib.operators.inputs import _digest
        from nwqlib.subroutines.state_preparation.mps import MPSDecomposition, _decompose_normalized_state
        candidate = self.mps_decomposition if reuse is None else reuse
        target = self.prep_amplitudes(max_bytes=max_bytes)
        if candidate is not None:
            if not isinstance(candidate, MPSDecomposition):
                raise TypeError("reuse must be an MPSDecomposition")
            matches = (candidate.input_id == _digest("mps.normalized.C-order", (target.size,), (target,))
                       and candidate.max_bond_dim == max_bond_dim and candidate.threshold == threshold)
            if matches:
                return candidate
            if reuse is not None:
                raise ValueError("MPS reuse requires the same normalized tensor, ordering and truncation settings")
        return _decompose_normalized_state(target, max_bond_dim=max_bond_dim, threshold=threshold,
                                           max_bytes=max_bytes, max_svd_work=max_svd_work)

    def record(self) -> dict[str, Any]:
        """Return the kernel, quadrature and pair choices with their bounds as a plain dict.

        The keys name the requested and resolved kernel and quadrature,
        their parameters and formula identifiers, the node order, panel
        width and address structure, the pair's certificate status, the
        component bounds `approximate_lchs_error_bound` and
        `quadrature_error_bound`, and the `unbudgeted_error_stages`.
        """

        record = {
            "requested_lchs_kernel": self.requested_lchs_kernel.to_dict(),
            "resolved_lchs_kernel": self.resolved_lchs_kernel.to_dict(),
            "lchs_kernel_formula_id": self.lchs_kernel_formula_id,
            "lchs_kernel_parameters": dict(self.resolved_lchs_kernel.parameters),
            "lchs_kernel_fixed_parameters": dict(self.lchs_kernel_fixed_parameters),
            "requested_k_quadrature": self.requested_k_quadrature.to_dict(),
            "resolved_k_quadrature": self.resolved_k_quadrature.to_dict(),
            "k_quadrature_formula_id": self.k_quadrature_formula_id,
            "k_quadrature_parameters": dict(self.resolved_k_quadrature.parameters),
            "k_quadrature_node_order": self.quadrature.node_order,
            "k_quadrature_panel_width": self.quadrature.h1,
            "k_quadrature_address_structure": self.quadrature.address_structure,
            "lchs_kernel_quadrature_compatibility": "allowed",
            "lchs_kernel_quadrature_certificate_status": (self.compatibility.certificate_status),
            "kernel_parameter_selection": dict(self.kernel_parameter_selection),
            "lchs_pair_derived_parameters": dict(self.derived_parameters),
            "approximate_lchs_error_bound": (self.quadrature.approximate_lchs_error_bound),
            "quadrature_error_bound": self.quadrature.quadrature_error_bound,
            "solution_error_certificate_status": self.compatibility.certificate_status,
            "unbudgeted_error_stages": list(self.compatibility.unbudgeted_error_stages),
        }
        detached = _detached_plain(record)
        assert isinstance(detached, dict)
        return detached


def eq7_cbeta(beta: float) -> float:
    """Return C_beta = 2*pi*exp(-2**beta), the normalization of ACL Eqs. (7) and (60).

    ACL arXiv:2312.03916v2, Eq. (33), evaluates this constant by the residue
    theorem. It makes the Eq. (7) kernel satisfy the normalization condition
    of Theorem 6.
    """

    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in the open interval (0, 1)")
    return 2.0 * pi * exp(-(2.0**beta))


def eq7_kernel_function(beta: float, z_value: float) -> complex:
    """Evaluate f(z) = exp(-(1+i*z)**beta)/C_beta, the near-optimal kernel of ACL
    arXiv:2312.03916v2, Eq. (7).
    """

    return np.exp(-(1.0 + 1.0j * z_value) ** beta) / eq7_cbeta(beta)


def eq7_coefficient(beta: float, k_value: float) -> complex:
    """Evaluate g(k) = f(k)/(1-i*k), the integrand weight of ACL arXiv:2312.03916v2,
    Eqs. (6) and (60).
    """

    return eq7_kernel_function(beta, k_value) / (1.0 - 1.0j * k_value)


def eq7_truncation_range(beta: float, epsilon: float) -> float:
    """Invert the integrated ACL pointwise decay for the allocated tail tolerance.

    ACL arXiv:2312.03916v2, Eqs. (185)-(186), bound the density by
    exp(-c*k**beta)/(C_beta*abs(k)), where c=cos(beta*pi/2). The two tails
    integrate to 2*E1(c*K**beta)/(beta*C_beta). Applying E1(x)<=exp(-x)/x gives
    the finite bound inverted here. This integration is NWQLib's derivation.

    With x=cos(beta*pi/2)*K**beta, solve x+log(x) >=
    log(2/(beta*C_beta*epsilon)). Scalar bisection in x avoids forming the
    potentially overflowing argument of Lambert W. The returned cutoff also
    passes the actual binary64 bound evaluation used in the selected record.

    Args:
        beta: Eq. (7) exponent in (0, 1).
        epsilon: Tail allowance in (0, 1), in the operator norm of exp(-A*T).
            The default provider passes half the construction tolerance.

    Returns:
        A binary64 cutoff K (k is dimensionless) at or just above the root of
        the closed-form inequality, for which eq7_tail_bound(beta, K) <= epsilon.
        The Eq. (62) alternative inside eq7_tail_bound can only lower the
        recorded bound. The search does not use it to shrink K.
    """

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in the open interval (0, 1)")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be in the open interval (0, 1)")
    target = log(2.) - log(beta) - log(eq7_cbeta(beta)) - log(epsilon)
    # x + log(x) increases on x > 0. It tends to -inf as x -> 0, and at
    # x = max(1, target) it is at least target because log(x) >= 0 there, so
    # this interval brackets the root.
    lower, upper = 0., max(1., target)
    while True:
        middle = lower + (upper - lower) / 2
        if middle == lower or middle == upper:
            break
        if middle + log(middle) >= target:
            upper = middle
        else:
            lower = middle
    try:
        cutoff = exp((log(upper) - log(cos(beta * pi / 2))) / beta)
    except OverflowError as error:
        raise ValueError("LCHS tail cutoff exceeds the finite scalar range") from error
    # K = (x/cos(beta*pi/2))**(1/beta) is rounded, and eq7_tail_bound is a
    # separate binary64 evaluation. Step K upward one ulp at a time until that
    # recorded evaluation itself meets epsilon.
    cutoff = finite_real(nextafter(cutoff, float("inf")), "LCHS cutoff")
    while eq7_tail_bound(beta, cutoff) > epsilon:
        cutoff = finite_real(nextafter(cutoff, float("inf")), "LCHS cutoff")
    return cutoff


def composite_gauss_grid(
    *,
    interval_count_each_side: int,
    h1: float,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all ACL Eq. (61) k-nodes and mapped weights on the symmetric grid.

    Single source for the ``2 * interval_count_each_side`` interval loop over
    ``[-K, K]`` used by the composite-Gauss provider. The Gauss-Legendre base
    rule is evaluated once and affinely remapped to each interval.

    With m = interval_count_each_side, the panels are [j*h1, (j+1)*h1] for
    j = -m, ..., m-1, as in Eq. (61). The map x -> h1*(x+1)/2 + j*h1 takes a
    base node on [-1, 1] into panel j and scales its weight by h1/2 (ACL
    arXiv:2312.03916v2, Appendix A.4).

    Returns:
        Two float arrays of length 2*m*node_count. The nodes are ordered panel
        by panel from the most negative panel, and within a panel in the
        ascending order of the Legendre roots. The weights are in the same order.
    """

    interval_count_each_side = integer(interval_count_each_side, "interval_count_each_side", 1)
    h1 = finite_real(h1, "h1")
    if h1 <= 0.0:
        raise ValueError("h1 must be positive")
    node_count = integer(node_count, "node_count", 1)
    base_nodes, base_weights = np.polynomial.legendre.leggauss(node_count)
    mapped_weights = 0.5 * h1 * base_weights
    k_nodes: list[float] = []
    for shifted_index in range(2 * interval_count_each_side):
        interval_index = -interval_count_each_side + shifted_index
        lower_end = interval_index * h1
        mapped_nodes = 0.5 * h1 * base_nodes + lower_end + 0.5 * h1
        k_nodes.extend(mapped_nodes)
    weights = np.tile(mapped_weights, 2 * interval_count_each_side)
    return np.asarray(k_nodes, dtype=float), weights


def _exp_with_nonfinite_overflow(log_value: float) -> float:
    """Return exp(log_value), or +inf where math.exp would raise OverflowError."""
    try:
        return exp(log_value)
    except OverflowError:
        return float("inf")


def _positive_bound(log_value):
    # If a positive analytic bound is below the representable range, round it
    # up to the smallest positive binary64 value; never advertise exact zero.
    if isnan(log_value):
        raise ValueError("analytic bound evaluation is undefined")
    return max(nextafter(0., 1.), _exp_with_nonfinite_overflow(log_value))


def eq7_tail_bound(beta, range_k):
    """Bound the operator-norm truncation error of the Eq. (7) integral outside [-K, K].

    ACL arXiv:2312.03916v2, Eq. (185), bounds the discarded tail by the
    integral of |g(k)| over |k| > K, and Eq. (186) gives
    |g(k)| <= exp(-|k|**beta*cos(beta*pi/2))/(C_beta*|k|).
    The substitution u = cos(beta*pi/2)*k**beta turns each side into
    E1(x)/(beta*C_beta) with x = cos(beta*pi/2)*K**beta, and
    E1(x) <= exp(-x)/x (bound 1/u by 1/x inside the E1 integral) gives
    2*exp(-x)/(beta*C_beta*x). This closed form is NWQLib's derivation from
    Eq. (186). For K >= 1 the smaller of it and ACL Lemma 10, Eq. (62), is
    returned. These are binary64 evaluations of analytic bounds, not an
    interval proof.

    Both bounds are evaluated as logarithms so that exp(-x) and the factorial
    in Eq. (62) cannot underflow or overflow before they are combined.
    """
    if not 0 < beta < 1 or not isfinite(range_k) or range_k <= 0:
        raise ValueError("Eq7 tail requires 0<beta<1 and finite positive K")
    cosine = cos(beta * pi * .5)
    log_x = log(cosine) + beta*log(range_k)
    x = exp(log_x)
    # log of 2*exp(-x)/(beta*C_beta*x).
    log_bound = log(2.) - log(beta) - log(eq7_cbeta(beta)) - x - log_x
    # ACL Lemma 10, Eq. (62), with p = ceil(1/beta):
    # 2**(p+1)*p!/(C_beta*cos(beta*pi/2)**p) * exp(-x/2)/K, in log form.
    # Its proof (Eqs. (187)-(189)) needs |k| >= 1, so it applies only to a
    # cutoff K >= 1.
    if range_k >= 1:
        p = ceil(1. / beta)
        lemma10_log_bound = ((p + 1) * log(2.) + lgamma(p + 1) - .5*x
                             - log(eq7_cbeta(beta)) - p*log(cosine) - log(range_k))
        log_bound = min(log_bound, lemma10_log_bound)
    return _positive_bound(log_bound)


def _check_quadrature_work(required, max_quadrature_work, subject):
    """Check one complete quadrature work count against ``LCHS.max_quadrature_work``.

    ``required`` is the complete charge of the selected rule, known before
    its node arrays are built. A refusal reports this charge and the limit.
    """
    if type(max_quadrature_work) is not int or max_quadrature_work < 1:
        raise ValueError("max_quadrature_work must be a positive integer")
    if required > max_quadrature_work:
        raise ValueError(
            f"{subject} {required} work units, exceeding "
            f"LCHS.max_quadrature_work={max_quadrature_work}. "
            f"Raise LCHS.max_quadrature_work to at least {required}.")


def _ellipse_rule(problem, beta, range_k):
    """Select the least-node Q-point Gauss rule using ATAP Theorem 19.3.

    ATAP ISBN 978-1-61197-239-9, Theorem 19.3, Eq. (19.8), bounds (n+1)-point
    Gauss quadrature for n >= 1, so n = Q-1 and Q >= 2. Apply its scalar result
    to v*F(k)u for arbitrary unit vectors u,v, then take their supremum to
    obtain the same operator-norm bound without a dimension factor.
    On |Im k| <= b = 1-d, |1-i*k| >= d and
    Re((1+i*k)**beta) >= 0, so |g(k)| <= 1/(C_beta*d). The evolution factor
    satisfies ||exp(-i*T*(k*L+H))|| <= exp(b*T*||L||), also for noncommuting
    Hermitian L and H. Indeed the Hermitian part of -i*T*(k*L+H) is
    T*Im(k)*L, so the logarithmic-norm bound controls the exponential even
    when L and H do not commute. A panel of width h has the Bernstein ellipse
    log(rho) = asinh(2*b/h) inside that strip. Summing the mapped bound (h/2
    per panel) over the 2*m panels of [-K, K] gives
    64*K*exp(b*T*||L||)/(15*C_beta*d*(rho**2-1)*rho**(2*(Q-1))).

    NWQLib uses this bound instead of the panel prescription of ACL
    arXiv:2312.03916v2, Lemma 11 (Eqs. (64)-(65)), because that lemma assumes
    T*max||L|| >= 32/e, which short-time and weakly dissipative problems
    violate. The search tries m = 1, 2, ... panels, finds the least Q meeting
    epsilon for each m, and stops once 4*m exceeds the best node count, since
    any rule with m panels has at least 2*m*Q >= 4*m nodes. Ties prefer fewer
    points per panel. No matrix operations or sampled maximum are used by this
    scalar selection.

    The search has a finite termination bound independent of
    ``max_quadrature_work``. After its first numerically valid candidate Q1,
    the best total is at most 2*Q1. Later candidates need at least 4*m nodes
    because Q >= 2. Once 4*m exceeds the best total, no later candidate can
    improve it, so the number of candidates tried is M <= floor(Q1/2).
    Numeric failure in forming Q or its bound raises rather than extending
    the search indefinitely. The search completes before the caller checks
    its work, so ``max_quadrature_work`` admits the search charge plus the
    selected rule's construction, not every scalar probe; finite
    termination is not a uniform cheap-runtime guarantee on all input
    scales.

    Args:
        problem: Scalars T, ||L|| and the quadrature allowance epsilon (the
            default provider passes half the construction tolerance).
        beta: Eq. (7) exponent in (0, 1).
        range_k: Cutoff K. The panels tile [-K, K] exactly.

    Returns:
        A pair (choice, work). choice is (2*m*Q, Q, m, h, bound): the total
        node count, Gauss points per panel, panels per half-axis, panel width
        K/m, and the quadrature bound in operator norm. work is the scalar
        work charged for the candidates tried.
    """
    d = LCHS_ELLIPSE_STRIP_MARGIN
    b = 1. - d
    scaled_time = finite_real(problem.final_time * problem.l_norm, "T*||L||")
    if (problem.final_time < 0 or problem.l_norm < 0 or scaled_time < 0
            or not 0 < problem.epsilon < 1 or range_k <= 0):
        raise ValueError("invalid LCHS ellipse selection scalars")
    best, panels = None, 1
    while best is None or 4*panels <= best[0]:
        h = range_k / panels
        # The affine map of a panel onto [-1, 1] scales Im k by 2/h. The
        # Bernstein ellipse E_rho has semi-minor axis sinh(log rho), so this
        # rho is the largest ellipse inside the strip |Im k| <= b.
        log_rho = finite_real(asinh(2*b/h), "ellipse log radius")
        if log_rho <= 0:
            raise ValueError("LCHS ellipse radius is not numerically distinguishable from one")
        # log(rho**2 - 1) = 2*log(rho) + log(1 - rho**-2), accurate for rho near one.
        log_den = 2*log_rho + log(-expm1(-2*log_rho))
        log_prefactor = log(range_k) + log(64/(15*eq7_cbeta(beta)*d)) + b*scaled_time - log_den
        # Continuous solution of prefactor*rho**(-2*(Q-1)) = epsilon for Q-1.
        order = finite_real((log_prefactor-log(problem.epsilon))/(2*log_rho), "Gauss order")
        # Q >= 2 keeps n = Q-1 >= 1, the premise of ATAP Eq. (19.8). The
        # increment below absorbs rounding in the continuous order estimate.
        q = max(2, 1 + ceil(order))
        bound = _positive_bound(log_prefactor - 2*(q-1)*log_rho)
        if bound > problem.epsilon:
            q += 1
            bound = _positive_bound(log_prefactor - 2*(q-1)*log_rho)
        if not isfinite(bound) or bound > problem.epsilon:
            raise ValueError("LCHS ellipse rule cannot establish its quadrature tolerance numerically")
        candidate = (2*panels*q, q, panels, h, bound)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
        panels += 1
    return best, _ELLIPSE_CANDIDATE_WORK * (panels - 1)


def _resolve_eq7(
    supplied: Mapping[str, ProviderParameter],
) -> Mapping[str, ProviderParameter]:
    """Resolve the Eq. (7) exponent beta in (0, 1), default 0.75.

    ACL arXiv:2312.03916v2, Section 2.2 (p. 14), reports [0.7, 0.8] as the
    numerically best range for truncation. Its midpoint is a working choice
    within that range, not a universal optimum. The finite tail and
    quadrature bounds are evaluated at the selected beta before choosing the
    grid.
    """
    beta = float(supplied.get("beta", 0.75))
    if not 0.0 < beta < 1.0:
        raise ValueError("near_optimal_eq7 parameter 'beta' must lie in (0, 1)")
    return {"beta": beta}


def _resolve_no_parameters(
    supplied: Mapping[str, ProviderParameter],
) -> Mapping[str, ProviderParameter]:
    """Return the empty configuration of a rule with no free parameters."""
    return {}


def _resolve_low_somma_f2(
    supplied: Mapping[str, ProviderParameter],
) -> Mapping[str, ProviderParameter]:
    """Resolve the shift c > 0 of the Low-Somma kernel (arXiv:2508.19238v2), default 1.

    Theorem 2 holds for any c > 0. Low-Somma use c = 1 when bounding the
    query prefactor (Figure 1, discussed on p. 6). The coefficient 1-norm
    grows with c, and _admit_coefficient_rounding refuses a table whose
    binary64 finite sum cannot resolve the construction tolerance.
    """
    c_value = float(supplied.get("c", 1.0))
    if not np.isfinite(c_value) or c_value <= 0.0:
        raise ValueError("low_somma_f2 parameter 'c' must be finite and positive")
    return {"c": c_value}


def _eq7_provider_coefficient(k_value: float, context: Mapping[str, Any]) -> complex:
    """Evaluate the Eq. (60) integrand weight g(k) at the resolved beta."""
    return eq7_coefficient(float(context["kernel_parameters"]["beta"]), k_value)


def _cauchy_provider_coefficient(
    k_value: float,
    _context: Mapping[str, Any],
) -> complex:
    """Evaluate the Cauchy density 1/(pi*(1+k**2)) as the coefficient kernel.

    This is the original LCHS identity of An, Liu and Lin (PRL 131, 150603,
    2023, doi:10.1103/PhysRevLett.131.150603), which ACL arXiv:2312.03916v2
    restates as Eq. (5) and, as f(z) = 1/(pi*(1+i*z)), as Eq. (181). The
    composite-Gauss cutoff and node laws are specific to Eq. (7), so this
    kernel pairs only with the uncertified signed-binary grid.
    """
    return complex(1.0 / (pi * (1.0 + k_value**2)))


def low_somma_fhat_2(k_value: float, *, gamma: float, c_value: float) -> complex:
    """Evaluate the fixed ``j=2, y=1`` Low-Somma Fourier profile.

    The profile is the kernel f_hat_{j,y} of Low and Somma,
    arXiv:2508.19238v2, Eq. (6), at j=2 and y=1, which Theorems 2 and 3 use.
    It reduces to sqrt(2/pi)*exp(c-(k**2+1)/(4*gamma**2))*exp(-i*k*c)/(1+k**2).
    The 1/sqrt(2*pi) prefactor of their LCHS integral (Eq. (7)) is applied by
    the provider coefficient, not here.

    exp(c) and the Gaussian factor share one exponent, so a representable
    value does not overflow in exp(c) first. An unrepresentable magnitude
    raises OverflowError.
    """

    return complex(
        sqrt(2.0 / pi)
        * exp(c_value - (k_value**2 + 1.0) / (4.0 * gamma**2))
        * np.exp(-1.0j * k_value * c_value)
        / (1.0 + k_value**2)
    )


def _low_somma_provider_coefficient(
    k_value: float,
    context: Mapping[str, Any],
) -> complex:
    # Low-Somma arXiv:2508.19238v2, Eq. (11), weights each node by
    # h/sqrt(2*pi). The quadrature supplies h, and this kernel supplies
    # f_hat_2/sqrt(2*pi).
    c_value = float(context["kernel_parameters"]["c"])
    gamma = float(context["derived_parameters"]["gamma"])
    return low_somma_fhat_2(
        k_value,
        gamma=gamma,
        c_value=c_value,
    ) / sqrt(2.0 * pi)


LCHS_KERNEL_IMPLEMENTATIONS: dict[str, LCHSKernelProvider] = {
    "near_optimal_eq7": LCHSKernelProvider(
        name="near_optimal_eq7",
        formula_id="acl_eq7_v1",
        resolve_parameters=_resolve_eq7,
        coefficient=_eq7_provider_coefficient,
        fixed_parameters=_immutable_map({}),
    ),
    "cauchy_density": LCHSKernelProvider(
        name="cauchy_density",
        formula_id="cauchy_density_v1",
        resolve_parameters=_resolve_no_parameters,
        coefficient=_cauchy_provider_coefficient,
        fixed_parameters=_immutable_map({}),
    ),
    "low_somma_f2": LCHSKernelProvider(
        name="low_somma_f2",
        formula_id=LOW_SOMMA_FORMULA_ID,
        resolve_parameters=_resolve_low_somma_f2,
        coefficient=_low_somma_provider_coefficient,
        fixed_parameters=_immutable_map({"j": 2, "y": 1}),
    ),
}


def _resolve_composite_gauss(
    supplied: Mapping[str, ProviderParameter],
) -> Mapping[str, ProviderParameter]:
    """Resolve truncation_multiplier, a real number at least one, default 1.

    The builder multiplies the tail-certified cutoff by this factor. The tail
    bound decreases in K, so any factor at least one keeps the tail within its
    allowance. A factor below one could break that allowance and is rejected.
    """
    multiplier = float(supplied.get("truncation_multiplier", 1.0))
    if not np.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError(
            "composite_gauss parameter 'truncation_multiplier' must be finite and positive (at least one)"
        )
    return {"truncation_multiplier": multiplier}


def _resolve_signed_binary(
    supplied: Mapping[str, ProviderParameter],
) -> Mapping[str, ProviderParameter]:
    """Resolve the two required signed-binary grid parameters.

    num_qubits is the address width n >= 1, which gives 2**n nodes.
    lsb_position is the integer exponent e of the node spacing 2**e, so the
    nodes cover [-2**(n-1+e), (2**(n-1)-1)*2**e]. The rule carries no tail
    or quadrature bound, so there is no tolerance from which to derive
    default values.
    """
    num_qubits = integer(supplied["num_qubits"], "num_qubits", 1)
    lsb_position = integer_index(supplied["lsb_position"])
    return {"lsb_position": lsb_position, "num_qubits": num_qubits}


def _build_composite_gauss(
    problem: LCHSProblemContext,
    resolved: ResolvedProviderConfig,
    pair_context: Mapping[str, Any],
    max_bytes=DEFAULT_INPUT_BYTES,
    max_quadrature_work=100_000_000,
) -> LCHSQuadraturePlan:
    """Build the selected symmetric composite Gauss rule and its available component bounds.

    ACL arXiv:2312.03916v2, Eq. (61), defines the panelled k quadrature and
    Eq. (186) the finite tail bound used to select the cutoff. Each
    component receives half of the construction tolerance. The Q-point panel
    rule follows Trefethen, Approximation Theory and Approximation Practice,
    ISBN 978-1-61197-239-9, Theorem 19.3, with n=Q-1. The panel search
    divides the tail-certified K into whole panels (h1 = K/m) and selects
    the fewest nodes meeting the separate quadrature budget. Byte and work
    limits are checked before the node arrays are built. Other
    physical-output components remain outside this tolerance.

    The composite-Gauss provider first completes its scalar panel search. It
    stops when the minimum node count of every later panel count exceeds the
    best rule found. It then checks ``max_quadrature_work`` against the
    search charge, 32 units per selected node and the cubic Gauss-rule
    construction charge, before allocating the node arrays. A refusal names
    the complete charge of that selected rule. The selected rule has
    ``total = 2*m*Q`` nodes after M candidate panel counts, each charged 32
    units, so ``W_quad = 32*M + 32*(2*m*Q) + Q**3``: scalar selection, node,
    weight and kernel evaluation, and the Q-point Legendre eigensystem.
    These are engineering work units, not measured equal-cost operations.
    """
    beta = float(pair_context["kernel_parameters"]["beta"])
    multiplier = float(resolved.parameters["truncation_multiplier"])
    component_tolerance = finite_real(problem.epsilon / 2, "LCHS component tolerance")
    if component_tolerance == 0:
        raise ValueError("LCHS component tolerance is below the positive binary64 range")
    range_k = finite_real(multiplier * eq7_truncation_range(beta, component_tolerance), "LCHS cutoff")
    choice, selection_work = _ellipse_rule(replace(problem, epsilon=component_tolerance), beta, range_k)
    total, node_count, interval_count_each_side, h1, quadrature_bound = choice
    effective_range_k = range_k
    # Bytes: 256 per selected node is an envelope for the float64 node and
    # weight arrays, their tuple copies in LCHSQuadraturePlan, and the complex
    # coefficient list and tuple formed later by _resolve_coefficient_context.
    # 32*Q**2 covers the Q-by-Q float64 Jacobi matrix and eigensolver arrays
    # inside numpy's leggauss.
    _check_bytes(256*total + 32*node_count**2, max_bytes, "LCHS quadrature arrays")
    # Work: the candidate search, 32 scalar operations per node for the panel
    # map and kernel evaluation, and Q**3 for the Q-point Legendre eigenproblem.
    # The per-node 256 bytes and 32 operations are untuned envelopes registered
    # in ENGINEERING_CONSTANTS.
    _check_quadrature_work(selection_work + 32*total + node_count**3, max_quadrature_work,
                           "LCHS quadrature selection and construction require")
    nodes, weights = composite_gauss_grid(
        interval_count_each_side=interval_count_each_side,
        h1=h1,
        node_count=node_count,
    )
    approximate_bound = eq7_tail_bound(beta, effective_range_k)
    return LCHSQuadraturePlan(
        nodes=tuple(float(value) for value in nodes),
        base_weights=tuple(float(value) for value in weights),
        range_k=float(range_k),
        effective_range_k=effective_range_k,
        interval_count_each_side=interval_count_each_side,
        node_count=node_count,
        node_order="symmetric_panel_then_gauss_legendre",
        address_structure=_immutable_map({"affine": False, "kind": "explicit_node_table"}),
        physical_node_count=len(nodes),
        padding_count=0,
        approximate_lchs_error_bound=approximate_bound,
        quadrature_error_bound=quadrature_bound,
        h1=h1,
    )


def _build_signed_binary(
    _problem: LCHSProblemContext,
    resolved: ResolvedProviderConfig,
    _pair_context: Mapping[str, Any],
    max_bytes=DEFAULT_INPUT_BYTES,
    max_quadrature_work=100_000_000,
) -> LCHSQuadraturePlan:
    """Enumerate two's-complement coefficient nodes in little-endian address order.

    This uniform rectangle rule, k = signed(address)*2**lsb_position with
    weight 2**lsb_position, is NWQLib's own grid, not a construction from ACL
    or Low-Somma. It exists for its structure. Because k is affine in the
    address bits, structured SELECT can apply one affine rotation per Pauli
    occurrence. It carries no tail or quadrature bound.
    """
    num_qubits = int(resolved.parameters["num_qubits"])
    lsb_position = int(resolved.parameters["lsb_position"])
    # Validate max_bytes itself, then reject a width whose 2**n node count
    # already exceeds it, so 2**num_qubits is never formed for a huge n.
    _check_bytes(0, max_bytes)
    if num_qubits >= max_bytes.bit_length():
        raise ValueError("signed-binary coefficient grid exceeds max_bytes")
    state_count = 2**num_qubits
    # Same per-node byte envelope and per-node scalar work as the composite
    # Gauss builder, with no Legendre eigenproblem.
    _check_bytes(256*state_count, max_bytes, "signed-binary coefficient grid")
    _check_quadrature_work(32*state_count, max_quadrature_work,
                           "LCHS signed-binary coefficient grid requires")
    # The highest address bit has negative weight. Computational-basis order
    # therefore wraps from positive to negative nodes at the sign transition.
    sign_transition = 2 ** (num_qubits - 1)
    delta_k = float(2.0**lsb_position)
    nodes = tuple(
        float((state if state < sign_transition else state - state_count) * delta_k)
        for state in range(state_count)
    )
    weights = (delta_k,) * state_count
    unsigned_terms = tuple(
        {"qubit": qubit, "coefficient": float(2.0 ** (lsb_position + qubit))}
        for qubit in range(max(0, num_qubits - 1))
    )
    address_structure = _immutable_map(
        {
            "affine": True,
            "bit_order": "little_endian_computational_basis",
            "kind": "twos_complement_affine",
            "num_qubits": num_qubits,
            "sign_bit": num_qubits - 1,
            "sign_coefficient": float(-(2.0 ** (lsb_position + num_qubits - 1))),
            "unsigned_terms": unsigned_terms,
        }
    )
    return LCHSQuadraturePlan(
        nodes=nodes,
        base_weights=weights,
        range_k=float(max(abs(value) for value in nodes)),
        effective_range_k=float(max(abs(value) for value in nodes)),
        interval_count_each_side=0,
        node_count=state_count,
        node_order="computational_basis",
        address_structure=address_structure,
        physical_node_count=state_count,
        padding_count=0,
        approximate_lchs_error_bound=None,
        quadrature_error_bound=None,
    )


def _build_symmetric_uniform_trapezoid(
    _problem: LCHSProblemContext,
    _resolved: ResolvedProviderConfig,
    pair_context: Mapping[str, Any],
    max_bytes=DEFAULT_INPUT_BYTES,
    max_quadrature_work=100_000_000,
) -> LCHSQuadraturePlan:
    """Build the selected symmetric finite rule and record its power-of-two padding.

    The grid and its component caps follow Low and Somma, arXiv:2508.19238v2, Theorem 3.
    Nodes are h*j for j = -J..J with equal weight h, the uniform sum I_h of
    Eq. (11). The 2*J+1 nodes pad to a power of two with identity branches of
    zero coefficient.
    """
    derived = pair_context["derived_parameters"]
    radius = float(derived["R"])
    index_limit = int(derived["J"])
    step = float(derived["h"])
    # Same per-node byte envelope and scalar work as the other rules, for 2*J+1 nodes.
    _check_bytes(256*(2*index_limit+1), max_bytes, "symmetric coefficient grid")
    _check_quadrature_work(32*(2*index_limit+1), max_quadrature_work,
                           "LCHS symmetric coefficient grid requires")
    nodes = tuple(float(index * step) for index in range(-index_limit, index_limit + 1))
    node_count = 2 * index_limit + 1
    padded_count = 1 << max(0, node_count - 1).bit_length()
    return LCHSQuadraturePlan(
        nodes=nodes,
        base_weights=(step,) * node_count,
        range_k=radius,
        effective_range_k=radius,
        interval_count_each_side=index_limit,
        node_count=node_count,
        node_order="ascending_symmetric_integer",
        address_structure=_immutable_map(
            {
                "affine": False,
                "kind": "symmetric_uniform_trapezoid_with_identity_padding",
                "padding_count": padded_count - node_count,
                "physical_node_count": node_count,
            }
        ),
        physical_node_count=node_count,
        padding_count=padded_count - node_count,
        approximate_lchs_error_bound=float(derived["epsilon_lchs"]),
        quadrature_error_bound=float(derived["epsilon_quad"]),
    )


LCHS_K_QUADRATURE_IMPLEMENTATIONS: dict[str, LCHSQuadratureProvider] = {
    "composite_gauss": LCHSQuadratureProvider(
        name="composite_gauss",
        formula_id="acl_composite_gauss_finite_tail_v3",
        resolve_parameters=_resolve_composite_gauss,
        build_nodes_and_weights=_build_composite_gauss,
    ),
    "signed_binary_uniform": LCHSQuadratureProvider(
        name="signed_binary_uniform",
        formula_id="twos_complement_uniform_v1",
        resolve_parameters=_resolve_signed_binary,
        build_nodes_and_weights=_build_signed_binary,
    ),
    "symmetric_uniform_trapezoid": LCHSQuadratureProvider(
        name="symmetric_uniform_trapezoid",
        formula_id="low_somma_symmetric_uniform_trapezoid_v1",
        resolve_parameters=_resolve_no_parameters,
        build_nodes_and_weights=_build_symmetric_uniform_trapezoid,
    ),
}


# Registered kernel-quadrature pairs and the scope of their component bounds.
# Only (Eq. (7), composite Gauss) is "certified". Its cutoff comes from the
# Eq. (7) decay of Eqs. (185)-(186) and its panels from the strip bound
# |g| <= 1/(C_beta*d), so neither law transfers to the Cauchy kernel, whose
# tail decays only like 1/k**2 (ACL arXiv:2312.03916v2, Eq. (5)). The
# signed-binary grid is a plain rectangle rule chosen for its affine address
# structure and carries no bound. Low-Somma's Theorems 2-3
# (arXiv:2508.19238v2) give both component bounds for their own kernel and
# trapezoid rule, but the pair does not budget the synthesis and preparation
# stages, hence "incomplete". An unlisted pair is rejected.
LCHS_KERNEL_QUADRATURE_COMPATIBILITY: Mapping[tuple[str, str], LCHSPairCompatibility] = (
    MappingProxyType(
        {
            ("near_optimal_eq7", "composite_gauss"): LCHSPairCompatibility(
                allowed=True,
                certificate_status="certified",
                reason="ACL arXiv:2312.03916v2 Eq.(186) tail and ATAP ISBN 978-1-61197-239-9 Theorem19.3 Q-point ellipse quadrature bounds. Each receives half the construction tolerance. Other physical error components require separate bounds.",
            ),
            ("near_optimal_eq7", "signed_binary_uniform"): LCHSPairCompatibility(
                allowed=True,
                certificate_status="uncertified",
                reason="finite-grid truncation and quadrature error are not certified",
                unbudgeted_error_stages=("kernel_approximation", "k_quadrature"),
            ),
            ("cauchy_density", "signed_binary_uniform"): LCHSPairCompatibility(
                allowed=True,
                certificate_status="uncertified",
                reason="finite-grid truncation and quadrature error are not certified",
                unbudgeted_error_stages=("kernel_approximation", "k_quadrature"),
            ),
            ("cauchy_density", "composite_gauss"): LCHSPairCompatibility(
                allowed=False,
                certificate_status="rejected",
                reason=(
                    "composite_gauss automatic range and node-count laws are "
                    "near_optimal_eq7-specific; choose signed_binary_uniform"
                ),
            ),
            ("low_somma_f2", "symmetric_uniform_trapezoid"): LCHSPairCompatibility(
                allowed=True,
                certificate_status="incomplete",
                reason=(
                    "Low-Somma arXiv:2508.19238v2 closed-form LCHS and quadrature caps apply, "
                    "but synthesis stages are not jointly budgeted"
                ),
                unbudgeted_error_stages=LOW_SOMMA_UNBUDGETED_STAGES,
            ),
        }
    )
)


def _resolve_low_somma_pair_profile(
    problem: LCHSProblemContext,
    resolved_kernel: ResolvedProviderConfig,
) -> Mapping[str, ProviderParameter]:
    """Derive gamma, R, h and J from Low and Somma, arXiv:2508.19238v2, Theorems 2-3.

    The tolerance splits as epsilon_lchs = epsilon_quad = epsilon/3, the
    example split stated before Theorem 4 (arXiv:2508.19238v2, p. 6).
    Theorem 2 (p. 5) gives
    gamma = sqrt(c + log((1+1/(2*pi))/epsilon_lchs))/c and R = 2*c*gamma**2.
    Theorem 3 (p. 6) needs epsilon_quad <= 4/15, that is epsilon <= 4/5 as in
    Theorem 4, and a step h <= pi/(||L||_L1/2 + log(64*exp(3c/2)/(15*epsilon_quad)))
    with R/h an integer. For constant L, ||L||_L1 = T*||L||. Taking
    J = ceil(R/h_max) and h = R/J meets both conditions.

    Returns:
        A read-only mapping with J (nodes -J..J), R (truncation radius in k),
        epsilon_lchs and epsilon_quad (the two component caps), gamma, the
        realized step h = R/J, and the Theorem 3 step cap h_max.
    """
    epsilon = float(problem.epsilon)
    if not 0.0 < epsilon <= 4.0 / 5.0:
        raise ValueError(
            "low_somma_f2 + symmetric_uniform_trapezoid requires epsilon in (0, 4/5]"
        )
    c_value = float(resolved_kernel.parameters["c"])
    epsilon_lchs = epsilon / 3.0
    epsilon_quad = epsilon / 3.0
    gamma = (1.0 / c_value) * sqrt(
        c_value + log((1.0 + 1.0 / (2.0 * pi)) / epsilon_lchs)
    )
    radius = 2.0 * c_value * gamma**2
    # log(64*exp(3c/2)/(15*epsilon_quad)) in log form: exp(3c/2) alone
    # overflows for c > 473 while the step bound stays representable.
    h_max = pi / (
        problem.final_time * problem.l_norm / 2.0
        + 3.0 * c_value / 2.0
        + log(64.0 / (15.0 * epsilon_quad))
    )
    index_limit = int(ceil(radius / h_max))
    realized_step = radius / index_limit
    return _immutable_map(
        {
            "J": index_limit,
            "R": radius,
            "epsilon_lchs": epsilon_lchs,
            "epsilon_quad": epsilon_quad,
            "gamma": gamma,
            "h": realized_step,
            "h_max": h_max,
        }
    )


LCHS_PAIR_PROFILE_RESOLVERS: Mapping[
    tuple[str, str],
    Callable[[LCHSProblemContext, ResolvedProviderConfig], Mapping[str, ProviderParameter]],
] = MappingProxyType(
    {
        ("low_somma_f2", "symmetric_uniform_trapezoid"): (
            _resolve_low_somma_pair_profile
        )
    }
)

LCHS_PROVIDER_RECORD_KEYS = (
    "requested_lchs_kernel",
    "resolved_lchs_kernel",
    "lchs_kernel_formula_id",
    "lchs_kernel_parameters",
    "lchs_kernel_fixed_parameters",
    "requested_k_quadrature",
    "resolved_k_quadrature",
    "k_quadrature_formula_id",
    "k_quadrature_parameters",
    "k_quadrature_node_order",
    "k_quadrature_panel_width",
    "k_quadrature_address_structure",
    "lchs_kernel_quadrature_compatibility",
    "lchs_kernel_quadrature_certificate_status",
    "kernel_parameter_selection",
    "lchs_pair_derived_parameters",
    "approximate_lchs_error_bound",
    "quadrature_error_bound",
    "solution_error_certificate_status",
    "unbudgeted_error_stages",
)


def resolve_provider_config(
    request: ProviderConfig,
    registry: Mapping[str, LCHSKernelProvider | LCHSQuadratureProvider],
    *,
    slot: str,
) -> tuple[LCHSKernelProvider | LCHSQuadratureProvider, ResolvedProviderConfig]:
    """Resolve one provider request using its defaults and domain checks.

    Returns:
        The registered provider and a ResolvedProviderConfig holding every
        parameter after the provider's defaults and domain checks.
    """

    if not isinstance(request, ProviderConfig):
        raise TypeError(f"{slot} must be a ProviderConfig")
    try:
        provider = registry[request.implementation]
    except KeyError:
        accepted = ", ".join(repr(name) for name in sorted(registry))
        raise ValueError(
            f"unknown {slot} implementation {request.implementation!r}; "
            f"available implementations: {accepted}"
        ) from None
    resolved_parameters = provider.resolve_parameters(request.parameters)
    return provider, ResolvedProviderConfig(
        implementation=provider.name,
        parameters=resolved_parameters,
    )


def resolve_lchs_provider_requests(
    lchs_kernel: ProviderConfig,
    k_quadrature: ProviderConfig,
) -> tuple[
    LCHSKernelProvider,
    ResolvedProviderConfig,
    LCHSQuadratureProvider,
    ResolvedProviderConfig,
    LCHSPairCompatibility,
]:
    """Resolve both requests and enforce the registered pair matrix.

    The LCHS Method calls this at construction. Each provider resolves its
    supported parameters and checks their domains before any problem data
    is read. An unknown provider name or a rejected pair raises ValueError.

    Returns:
        (kernel provider, resolved kernel config, quadrature provider,
        resolved quadrature config, pair compatibility record).
    """

    kernel, resolved_kernel = resolve_provider_config(
        lchs_kernel,
        LCHS_KERNEL_IMPLEMENTATIONS,
        slot="lchs_kernel",
    )
    quadrature, resolved_quadrature = resolve_provider_config(
        k_quadrature,
        LCHS_K_QUADRATURE_IMPLEMENTATIONS,
        slot="k_quadrature",
    )
    key = (resolved_kernel.implementation, resolved_quadrature.implementation)
    compatibility = LCHS_KERNEL_QUADRATURE_COMPATIBILITY.get(key)
    if compatibility is None:
        raise ValueError(
            f"unlisted LCHS kernel-quadrature pair rejects initially: {key[0]!r} + {key[1]!r}"
        )
    if not compatibility.allowed:
        raise ValueError(
            f"LCHS kernel-quadrature pair {key[0]!r} + {key[1]!r} is rejected: "
            f"{compatibility.reason}"
        )
    return (
        kernel,
        resolved_kernel,
        quadrature,
        resolved_quadrature,
        compatibility,
    )


def _admit_coefficient_rounding(coefficient_l1_norm: float, epsilon: float, pair_label: str) -> None:
    """Refuse a coefficient table whose binary64 finite sum cannot resolve epsilon.

    The finite sum sum_j c_j U_j u0 has terms whose norms add up to
    alpha*||u0||, with alpha = sum_j |c_j| the coefficient 1-norm, while
    the exact solution of the (shifted) problem has norm at most ||u0||,
    because its L is PSD (ACL arXiv:2312.03916v2, Lemma 21, Eq. (162)). A
    large alpha therefore means heavy cancellation. Evaluating the sum in
    binary64 rounds every term at least once, which leaves an error of order
    u*alpha*||u0|| with u = 2**-53, and recursive summation of N terms can
    add up to (N - 1)*u*alpha*||u0|| (Higham, Accuracy and Stability of
    Numerical Algorithms, 2nd ed., doi:10.1137/1.9780898718027, Eq. (4.4)).
    The quantum route has the same scale, because it multiplies the
    success-projected amplitude by alpha. For the Low-Somma kernel with
    shift c, alpha is close to exp(c)*erfc(1/(2*gamma)) (Low and Somma,
    arXiv:2508.19238v2, Theorem 2, and Theorem 3, Eq. (12), for the discrete
    sum), about 1.2e16 at c = 50.

    The kernel and quadrature bounds describe exact arithmetic, and the LCHS
    error model publishes no floating-point value, so it cannot report this
    rounding next to them. Planning therefore refuses the table when
    u*alpha exceeds LCHS_ROUNDING_EPSILON_FRACTION*epsilon. Both sides are
    operator-level quantities for a unit input, and the input norm, Duhamel
    weight and PSD recovery scale them alike. Equality is admitted.
    """
    rounding = _UNIT_ROUNDOFF*coefficient_l1_norm
    allowance = LCHS_ROUNDING_EPSILON_FRACTION*epsilon
    if rounding > allowance:
        raise ValueError(
            f"LCHS provider pair {pair_label} has coefficient 1-norm {coefficient_l1_norm:.3g}, "
            f"so binary64 evaluation of its finite sum carries rounding of order u*alpha = {rounding:.3g} "
            f"of the input norm, above {LCHS_ROUNDING_EPSILON_FRACTION:g}*approximation_tolerance "
            f"= {allowance:.3g}. Choose kernel parameters with a smaller coefficient 1-norm, such as a "
            "smaller low_somma_f2 shift c, or a larger approximation_tolerance."
        )


def _resolve_coefficient_context(
    *,
    lchs_kernel: ProviderConfig,
    k_quadrature: ProviderConfig,
    problem_context: LCHSProblemContext,
    max_bytes=DEFAULT_INPUT_BYTES,
    max_quadrature_work=100_000_000,
) -> LCHSCoefficientPlan:
    """Finalize one provider-independent immutable coefficient plan.

    Coefficients are c_j = w_j*g(k_j) in node order. A non-finite node, weight,
    coefficient or running L1 norm raises at the first offending index, so an
    overflowed kernel never reaches PREP normalization or SELECT phases.
    """

    (
        kernel,
        resolved_kernel,
        quadrature,
        resolved_quadrature,
        compatibility,
    ) = resolve_lchs_provider_requests(lchs_kernel, k_quadrature)
    pair_key = (resolved_kernel.implementation, resolved_quadrature.implementation)
    profile_resolver = LCHS_PAIR_PROFILE_RESOLVERS.get(pair_key)
    pair_label = (
        f"{resolved_kernel.implementation!r} + "
        f"{resolved_quadrature.implementation!r}"
    )
    try:
        derived_parameters = (
            MappingProxyType({})
            if profile_resolver is None
            else profile_resolver(problem_context, resolved_kernel)
        )
        pair_context = MappingProxyType(
            {
                "kernel_parameters": resolved_kernel.parameters,
                "quadrature_parameters": resolved_quadrature.parameters,
                "derived_parameters": derived_parameters,
            }
        )
        quadrature_plan = quadrature.build_nodes_and_weights(
            problem_context,
            resolved_quadrature,
            pair_context,
            max_bytes=max_bytes,
            max_quadrature_work=max_quadrature_work,
        )
    except ArithmeticError as exc:
        raise ValueError(
            f"invalid inputs for LCHS provider pair {pair_label}: {exc}"
        ) from exc
    # Combine base quadrature weights with the selected kernel. The coefficient
    # L1 norm supplies LCU normalization, while complex phases belong to SELECT.
    coefficients: list[complex] = []
    coefficient_l1_norm = 0.0
    for index, (node, weight) in enumerate(zip(
        quadrature_plan.nodes,
        quadrature_plan.base_weights,
        strict=True,
    )):
        if not np.isfinite(node):
            raise ValueError(
                f"LCHS provider pair {pair_label} produced non-finite node "
                f"at index {index}: {node!r}"
            )
        if not np.isfinite(weight):
            raise ValueError(
                f"LCHS provider pair {pair_label} produced non-finite weight "
                f"at node {node!r} (index {index}): {weight!r}"
            )
        try:
            coefficient = weight * kernel.coefficient(node, pair_context)
        except ArithmeticError as exc:
            raise ValueError(
                f"invalid inputs for LCHS provider pair {pair_label} at node "
                f"{node!r} (index {index}): {exc}"
            ) from exc
        if not np.isfinite(coefficient):
            raise ValueError(
                f"LCHS provider pair {pair_label} produced non-finite coefficient "
                f"at node {node!r} (index {index}): {coefficient!r}"
            )
        coefficient_l1_norm += float(abs(coefficient))
        if not np.isfinite(coefficient_l1_norm):
            raise ValueError(
                f"LCHS provider pair {pair_label} produced non-finite coefficient "
                f"L1 norm at node {node!r} (index {index}): "
                f"{coefficient_l1_norm!r}"
            )
        coefficients.append(complex(coefficient))
    _admit_coefficient_rounding(coefficient_l1_norm, float(problem_context.epsilon), pair_label)
    # Keep user/default parameter provenance alongside the immutable grid so
    # reporting and later realization use the same coefficient selection.
    selection = {
        name: ("user" if name in lchs_kernel.parameters else "provider_default")
        for name in resolved_kernel.parameters
    }
    if profile_resolver is not None:
        selection["profile"] = LOW_SOMMA_PROFILE_ID
    return LCHSCoefficientPlan(
        requested_lchs_kernel=lchs_kernel,
        resolved_lchs_kernel=resolved_kernel,
        requested_k_quadrature=k_quadrature,
        resolved_k_quadrature=resolved_quadrature,
        lchs_kernel_formula_id=kernel.formula_id,
        k_quadrature_formula_id=quadrature.formula_id,
        lchs_kernel_fixed_parameters=kernel.fixed_parameters,
        kernel_parameter_selection=_immutable_map(selection),
        derived_parameters=derived_parameters,
        nodes=quadrature_plan.nodes,
        base_weights=quadrature_plan.base_weights,
        coefficients=tuple(coefficients),
        coefficient_l1_norm=coefficient_l1_norm,
        quadrature=quadrature_plan,
        compatibility=compatibility,
    )


def _unitary_coefficient_plan(options):
    """L=0 eliminates the k integral: one H-evolution branch of weight one.

    This is an algebraic reduction of A=L+iH, not a one-node approximation to
    the selected kernel. Requested provider settings remain provenance only.
    """
    resolved = ResolvedProviderConfig(implementation='unitary_reduction', parameters={})
    # One node k = 0 with weight 1. The single branch is exp(-i*T*H) itself,
    # and both component bounds are exact zeros.
    quadrature = LCHSQuadraturePlan(
        nodes=(0.,),
        base_weights=(1.,),
        range_k=0.,
        effective_range_k=0.,
        interval_count_each_side=0,
        node_count=1,
        node_order='single_unitary_branch',
        address_structure={},
        physical_node_count=1,
        padding_count=0,
        approximate_lchs_error_bound=0.,
        quadrature_error_bound=0.,
    )
    return LCHSCoefficientPlan(
        requested_lchs_kernel=options.lchs_kernel,
        resolved_lchs_kernel=resolved,
        requested_k_quadrature=options.k_quadrature,
        resolved_k_quadrature=resolved,
        lchs_kernel_formula_id='A=iH',
        k_quadrature_formula_id='no_k_integral',
        lchs_kernel_fixed_parameters={},
        kernel_parameter_selection={},
        derived_parameters={},
        nodes=(0.,),
        base_weights=(1.,),
        coefficients=(1.+0j,),
        coefficient_l1_norm=1.,
        quadrature=quadrature,
        compatibility=LCHSPairCompatibility(
            allowed=True,
            certificate_status='exact_algebraic',
            reason='L is exactly zero; exp(-AT)=exp(-iHT) with no k quadrature',
        ),
    )


def _homogeneous_coefficient_plan(quadrature, elapsed):
    """Return the coefficient plan of the physical propagator exp(-A*elapsed).

    Each coefficient is multiplied by exp(shift*elapsed), the recovery of the
    selected PSD shift, and coefficient_l1_norm becomes the sum of the
    compensated magnitudes. When binary64 cannot represent that factor, the
    coefficients carry exp(shift*elapsed)*2**-e instead, with e from
    solution_error_budget.psd_recovery_exponent, and every consumer composes
    2**e into its binary recovery. derived_parameters records the shift, the
    elapsed time and the factor applied to the coefficients. The nodes and
    base weights are unchanged.
    """
    from dataclasses import replace
    from .solution_error_budget import psd_recovery_exponent, psd_recovery_part
    shift = float(quadrature.conversion["psd_shift"])
    # ACL arXiv:2312.03916v2, Eq. (4)/(60), specialized to constant A:
    # replacing A by A+sI shifts L by s*I, and
    # exp(-A*T) = exp(s*T)*exp(-(A+s*I)*T). The LCHS integral is applied to
    # the shifted A, whose L is PSD, and exp(s*T) is applied once to the
    # homogeneous coefficients. Duhamel source branches use their own elapsed
    # time T-t_j instead.
    compensation = psd_recovery_part(shift, elapsed, psd_recovery_exponent(shift, elapsed))
    coefficients = tuple(complex(c * compensation) for c in quadrature.coefficients)
    alpha = fsum(abs(c) for c in coefficients)
    if not isfinite(alpha) or alpha <= 0:
        raise ValueError("PSD-compensated LCHS coefficients require a finite positive 1-norm")
    raw = quadrature.coefficient_plan
    return replace(raw, coefficients=coefficients, coefficient_l1_norm=alpha,
        derived_parameters=_immutable_map({**raw.derived_parameters,
            "psd_shift": shift, "elapsed_time": elapsed,
            "homogeneous_coefficient_compensation": compensation}))


def resolve_lchs_coefficient_plan(problem_or_plan, *, lchs_kernel=None, k_quadrature=None,
                                  approximation_tolerance=None, max_bytes=DEFAULT_INPUT_BYTES):
    """Return the LCHS coefficient table of a problem without a source, or of an LCHS Plan.

    For a `LinearDynamics` without a source, it computes the k nodes, the
    weights and the coefficients `c_j` of the finite sum
    `sum_j c_j exp(-i T (k_j L + H)) u0`, with L after any PSD shift, from
    the given kernel, quadrature and tolerance. It builds no SELECT circuit
    or QSP phases, computes no output vector and runs nothing on a backend.
    For a Plan made by `plan(problem, method=LCHS(...))`, it returns the
    table that the Plan executes, so an analysis of the coefficient tensor
    always describes those coefficients. Call `decompose_mps` on the result
    to study the MPS compression of the coefficient state, as in the
    [coefficient MPS analysis](../../algorithms/lchs.md#coefficient-mps-analysis)
    of the LCHS guide.

    Args:
        problem_or_plan (LinearDynamics | Plan): A `LinearDynamics` without
            a source and with positive elapsed time, or an LCHS Plan with a
            coefficient table for an input without a source.
        lchs_kernel (ProviderConfig | None): Kernel choice. `None` selects
            `"near_optimal_eq7"`, as in `LCHS`. Must be `None` for a Plan.
        k_quadrature (ProviderConfig | None): Quadrature choice. `None`
            selects `"composite_gauss"`, as in `LCHS`. Must be `None` for a
            Plan.
        approximation_tolerance (float | None): Construction tolerance,
            strictly between 0 and 1. `None` selects `0.01`, as in `LCHS`.
            Must be `None` for a Plan.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Upper limit on the arrays of the Cartesian parts and the PSD
            check, tested before they are formed. A `PeriodicStencil` A
            forms no matrix and uses the bound `||L|| <= mass + 4*diffusion`.

    Returns:
        coefficients (LCHSCoefficientPlan): The nodes, weights, coefficients
            and component bounds. For a dense A the coefficients include the
            PSD growth factor `exp(shift*T)`, divided by `2**e` when the
            factor exceeds binary64, as recorded in
            `derived_parameters["homogeneous_coefficient_compensation"]`.

    Raises:
        ValueError: If the problem has a source or zero elapsed time, if
            `approximation_tolerance` is outside (0, 1), or if a Plan is
            given with any of the three choices or holds no coefficient
            table for an input without a source.
        TypeError: If `problem_or_plan` is neither a `LinearDynamics` nor a
            Plan.
    """
    from nwqlib.core.planning import Plan
    from nwqlib.problems.records import LinearDynamics
    if isinstance(problem_or_plan, Plan):
        if any(value is not None for value in (lchs_kernel, k_quadrature, approximation_tolerance)):
            raise ValueError("selected coefficient access cannot override the existing Plan")
        selected = problem_or_plan._native.get("coefficient_plan")
        if problem_or_plan.method.descriptor.method != "lchs" or not isinstance(selected, LCHSCoefficientPlan):
            raise ValueError("this Plan has no available homogeneous LCHS coefficient selection")
        return selected
    if not isinstance(problem_or_plan, LinearDynamics):
        raise TypeError("coefficient selection requires LinearDynamics or its selected LCHS Plan")
    problem = problem_or_plan
    if problem.source is not None:
        raise ValueError("coefficient MPS analysis currently supports homogeneous dynamics only")
    if problem.elapsed_time <= 0:
        raise ValueError("zero-time dynamics needs no coefficient construction")
    kernel = lchs_kernel or ProviderConfig(implementation="near_optimal_eq7")
    quadrature = k_quadrature or ProviderConfig(implementation="composite_gauss")
    # Same default as LCHS.approximation_tolerance (registered in ENGINEERING_CONSTANTS).
    tolerance = .01 if approximation_tolerance is None else finite_real(approximation_tolerance, "approximation_tolerance")
    if not 0 < tolerance < 1:
        raise ValueError("approximation_tolerance must lie in (0,1)")
    from .time_independent_terms import _prepare_decomposition, _quadrature_data_from_plan
    if problem.A.manifest.reference.representation == "periodic_stencil":
        parameters = problem.A.periodic_stencil()
        # Analytic bound ||L|| <= mass + 4*diffusion (see periodic.py), so no
        # 2**q array or eigensolve is needed.
        norm = parameters.mass + 4*parameters.diffusion
        return _resolve_coefficient_context(lchs_kernel=kernel, k_quadrature=quadrature,
            problem_context=LCHSProblemContext(final_time=problem.elapsed_time, epsilon=tolerance,
                l_norm=norm), max_bytes=max_bytes)
    matrix = problem.A.dense_array()
    # Encoded dimension: the next power of two, at least 2 (one system qubit).
    dimension = 1 << max(1, (problem.dimension-1).bit_length())
    # Coefficient-only selection computes the unpadded Hermitian endpoints
    # with the planner's spectral owner (_prepare_decomposition, parts=False)
    # and adds padding zeros analytically. It forms one adjoint and L,
    # releases the adjoint, and forms no H, padded or shifted array. With d
    # the physical dimension, B_coefficient = max(48 d**2, 48 d**2 + 16 d):
    # A, L and the adjoint, then A, L, the solver's internal copy and both
    # eigenvalue vectors. The queried eigensolver workspace Q_N(d) is not
    # charged (a declared known-array workspace, not a complete cap). PSD
    # compensation remains part of the selected coefficients.
    d = problem.dimension
    _check_bytes(48*d*d + 16*d, max_bytes, "LCHS Cartesian/PSD coefficient selection")
    from .method import LCHS
    method = LCHS(lchs_kernel=kernel, k_quadrature=quadrature, approximation_tolerance=tolerance)
    prepared = _prepare_decomposition(matrix=matrix, method=method, padded_dimension=dimension,
                                      max_spectral_work=100_000_000, parts=False)
    selected = _quadrature_data_from_plan(prepared=prepared, final_time=problem.elapsed_time, method=method,
                                          max_bytes=max_bytes)
    return _homogeneous_coefficient_plan(selected, problem.elapsed_time)
