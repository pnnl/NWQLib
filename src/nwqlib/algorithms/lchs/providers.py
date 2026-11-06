"""Kernel and quadrature rules of LCHS and the resolution of their pairs.

ACL denotes An, Childs, and Lin, "Quantum algorithm for linear non-unitary
dynamics with near-optimal dependence on all parameters", arXiv:2312.03916,
https://arxiv.org/abs/2312.03916. ACL equation numbers in this module refer to
arXiv:2312.03916v2. Low-Somma denotes Low and Somma, "Optimal quantum
simulation of linear non-unitary dynamics", arXiv:2508.19238v2. Apel-Berntson
denotes Apel and Berntson, arXiv:2610.03669v1, a preprint whose peer-review
status is unconfirmed. ATAP denotes
Trefethen, Approximation Theory and Approximation Practice,
ISBN 978-1-61197-239-9, Chapter 19 (https://www.chebfun.org/ATAP/chap19.m).

A kernel-quadrature pair turns an LCHS representation of exp(-A*T) as an integral of
g(k)*exp(-i*T*(k*L+H)) over real k, with A = L + iH, into finite real nodes
k_j and complex coefficients c_j. The representation is exact for the ACL
kernels (Eqs. (6) and (60)) and approximate for Low-Somma (their Eq. (8))
and for the compact Apel-Berntson kernel on (-R, R).
The pair accounts for its error components, the kernel-integral error and the
finite k-quadrature error. For Eq. (7) the kernel-integral error is the tail beyond
the cutoff. For Low-Somma it is epsilon_lchs of their Theorem 2, which also
covers the approximate representation and so is not a pure tail. For
Apel-Berntson it is the continuous allowance of their Theorem 4 and
Proposition 10, and the tanh-sinh rule bounds its step and truncation
separately (their Theorem 8 at the default radius, which
docs/algorithms/lchs_derivations.md adapts to a supplied radius). Weights prepared
at fixed nodes (coefficient_artifact.py) replace both stages by one residual
bound of the whole finite sum. Branch
synthesis, state preparation and physical recovery are bounded separately.

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
from fractions import Fraction
from math import (asin, asinh, atan, ceil, cos, exp, expm1, frexp, fsum, hypot, inf, isfinite, isnan, ldexp, lgamma,
                  log, log1p, nextafter, pi, sin, sinh, sqrt, tan)
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from scipy.special import roots_legendre

from operator import index as integer_index

from nwqlib._limits import ResourceLimitError, byte_allowance, exceeds, memory_limit, optional_limit
from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from nwqlib.algorithms.lchs.coefficient_artifact import LCHSDiscreteWeightArtifact
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


def _read_only_vector(values: Any, dtype: Any) -> np.ndarray:
    """Return values as a read-only array of dtype whose bytes cannot change, sharing an already frozen array."""
    from nwqlib.operators.inputs import _freeze_array

    return _freeze_array(np.asarray(values, dtype=dtype))


def _fields_equal(left: Any, right: Any) -> bool:
    """Compare two records of the same dataclass field by field, an array field by its dtype and exact elements."""
    if left.__class__ is not right.__class__:
        return NotImplemented
    for field_name in left.__dataclass_fields__:
        first, second = getattr(left, field_name), getattr(right, field_name)
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            if not (isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
                    and first.dtype == second.dtype and np.array_equal(first, second)):
                return False
        elif first is not second and first != second:
            return False
    return True


def _detached_plain(value: Any) -> Any:
    """Return detached plain dictionaries and lists for serialization."""

    if isinstance(value, Mapping):
        return {key: _detached_plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_detached_plain(item) for item in value]
    return value


KernelResolver = Callable[[Mapping[str, ProviderParameter]], Mapping[str, ProviderParameter]]
KernelCoefficient = Callable[[float, Mapping[str, Any]], complex]
KernelCoefficients = Callable[[np.ndarray, Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True, kw_only=True)
class LCHSKernelProvider:
    """One registered LCHS coefficient-kernel implementation.

    Attributes:
        name: Name accepted as the implementation of a kernel ProviderConfig.
        formula_id: Versioned identifier recorded with the coefficient formula.
        resolve_parameters: Callable validating explicit parameter values and returning
            the complete resolved parameter mapping.
        coefficient: Callable evaluating the complex coefficient kernel at one real k
            using the resolved pair context. Quadrature weights are multiplied separately.
        fixed_parameters: Read-only formula constants, such as Low-Somma j=2 and y=1,
            which are not user-adjustable provider parameters.
        coefficients: Default None. Callable evaluating the same kernel at a
            float64 array of k values with the same pair context and
            returning their complex values in order. When it is None, a
            coefficient plan calls `coefficient` once per node.
    """

    name: str
    formula_id: str
    resolve_parameters: KernelResolver
    coefficient: KernelCoefficient
    fixed_parameters: Mapping[str, ProviderParameter]
    coefficients: KernelCoefficients | None = None

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

    Providers read only these scalars, never an operator, state or circuit.

    Attributes:
        final_time: Largest elapsed time T_e of any application of the finite
            sum, in the units paired with A in exp(-T A). The selected rule
            serves every elapsed time in [0, T_e], because each pair's bound
            conditions at T_e also hold at every smaller elapsed time.
        epsilon: Requested finite-construction tolerance. Without
            `component_allowances` each pair divides it among its error
            components by its own rule: the Eq. (7) and composite-Gauss pair
            gives each of its two components epsilon/2, Low-Somma gives each
            of its two epsilon/3, Apel-Berntson gives each of its three
            epsilon/3, and a discrete-weight artifact receives all of it.
            Other physical-output error contributions are bounded
            separately. The binary64 rounding admission of the finite sum
            also uses this tolerance.
        l_norm: Bound B_Ls on the spectral norm of the processed Hermitian
            L_s = L + s I that enters exp(-i t (k L_s + H)). The strip and
            support conditions of the providers need B_Ls >= ||L_s||. The
            dense planner supplies the eigensolver's spectral norm, whose
            rounding is not enclosed. T_e*l_norm is dimensionless.
        component_allowances: Default `None`. Operator-norm allowance of each
            error component of the chosen pair, by component name, each
            finite and positive with a sum at most `epsilon`. The Eq. (7)
            and Low-Somma pairs name `"kernel_approximation"` and
            `"k_quadrature"`, Apel-Berntson names `"kernel_approximation"`,
            `"k_discretization"` and `"k_truncation"`, and the discrete-weight
            pair names `"discrete_residual"`. Pairs without bounds accept
            none. `None` applies the division described under `epsilon`.
    """

    final_time: float
    epsilon: float
    l_norm: float
    component_allowances: Mapping[str, float] | None = None


@dataclass(frozen=True, kw_only=True, eq=False)
class LCHSQuadraturePlan:
    """The k-quadrature rule of an `LCHSCoefficientPlan`: its nodes, weights, address layout and component error bounds.

    `LCHSCoefficientPlan.quadrature` holds it. The chosen quadrature rule
    builds it, except for the exact unitary reduction of an exactly zero L,
    whose one-node rule LCHS builds directly. The fields are read-only.
    The component bounds cover only the error components of the chosen
    kernel-quadrature pair. They are not a bound on state preparation,
    Hamiltonian synthesis or total physical output.

    Attributes:
        nodes: Ordered real k values for the mathematical quadrature, without the
            padding address slots of the LCU, as a read-only float64 array.
            This order also indexes base_weights.
        base_weights: Real integration weights before multiplication by the
            kernel, as a read-only float64 array.
        range_k: Cutoff K of the rule as built and of its available
            component bounds, or the maximum node magnitude for a signed-binary
            grid (zero for the exact unitary reduction). In composite Gauss, K
            specifies the panel interval [-K, K]. The Gauss nodes are interior.
        interval_count_each_side: Gauss panel count on each side of zero. The symmetric
            uniform rule uses its integer half-range J, tanh-sinh its half-range
            N, and signed, discrete-weight and unitary rules use zero.
        node_count: Gauss-Legendre points per panel for composite Gauss. Total points
            for the signed, symmetric-uniform, tanh-sinh and discrete-weight
            rules, and one for unitary reduction.
        node_order: Description of the node-to-address ordering, including
            computational-basis wraparound for a signed two's-complement grid.
        address_structure: Read-only structural data that the SELECT check and
            cost formulas use, such as affine bit coefficients or an
            explicit-table declaration.
        physical_node_count: Number of mathematical nodes, excluding padding address slots.
        padding_count: Number of padding address slots that this quadrature rule
            reports. An LCU circuit may need further address slots to reach a
            power-of-two total, even when this value is ``0``.
        approximate_lchs_error_bound: Available kernel-integral approximation bound.
            For the default Eq.(7) pair this is the finite-cutoff tail bound. None
            means unavailable, while unitary reduction has an algebraic zero.
        quadrature_error_bound: Finite bound on the quadrature component of the
            error, under the assumptions of the chosen kernel and quadrature
            rule. ``None`` means that this bound is unavailable, not that
            the error is zero or that the quadrature rule has failed.
        h1: Width of each composite-Gauss panel in the integration variable k,
            equal to ``range_k / interval_count_each_side``. ``None`` (the default)
            for rules without Gauss panels, including the exact single-unitary
            reduction.
        kernel_values: Default ``None``. Kernel factor of each node, in node
            order, when the rule determines it together with the node. The
            tanh-sinh rule evaluates the Apel-Berntson phase from its
            parameter xi_n, because an evaluation at the rounded node k_n
            loses that phase near the endpoints ±R. For an
            LCHSDiscreteWeightArtifact these are its complex weights, and the
            base weights are one. A read-only complex128 array, or ``None``
            when the kernel is evaluated at each node k_j.
    """

    nodes: np.ndarray
    base_weights: np.ndarray
    range_k: float
    interval_count_each_side: int
    node_count: int
    node_order: str
    address_structure: Mapping[str, Any]
    physical_node_count: int
    padding_count: int
    approximate_lchs_error_bound: float | None
    quadrature_error_bound: float | None
    h1: float | None = None
    kernel_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if field_name in ("nodes", "base_weights"):
                value = _read_only_vector(value, np.float64)
            elif field_name == "kernel_values":
                value = None if value is None else _read_only_vector(value, np.complex128)
            else:
                value = _deep_freeze(value)
            object.__setattr__(self, field_name, value)

    __eq__ = _fields_equal


QuadratureResolver = Callable[[Mapping[str, ProviderParameter]], Mapping[str, ProviderParameter]]
QuadratureBuilder = Callable[
    [LCHSProblemContext, ResolvedProviderConfig, Mapping[str, Any]],
    LCHSQuadraturePlan,
]


@dataclass(frozen=True, kw_only=True)
class LCHSQuadratureProvider:
    """One registered LCHS k-quadrature implementation.

    Attributes:
        name: Name accepted as the implementation of a quadrature ProviderConfig.
        formula_id: Versioned identifier for the node and weight construction.
        resolve_parameters: Callable validating the requested parameters and filling
            the defaults applicable to this quadrature rule.
        build_nodes_and_weights: Callable receiving problem scalars, resolved settings
            and kernel-pair context and returning the LCHSQuadraturePlan.
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
    """Whether an LCHS kernel and quadrature pair may be used together, and what its error certificate covers.

    `LCHSCoefficientPlan.compatibility` holds it.

    Attributes:
        allowed: Whether this kernel and quadrature combination may be
            chosen.
        certificate_status: Status of the pair's component bounds, one of
            certified, uncertified, rejected or incomplete, or exact_algebraic
            for the unitary reduction that needs no k integral.
            None of these by itself certifies a complete physical solution.
        reason: Explanation of the domain in which the pair is allowed, or of
            why its guarantee is absent or rejected.
        unbudgeted_error_stages: Named approximation stages that receive no
            share of `approximation_tolerance` from this pair. A pair with
            component bounds gives each bounded component a share, and it can
            name other stages, such as Trotter synthesis or state preparation.
            A pair on `signed_binary_uniform` bounds neither its kernel nor its
            k quadrature, so it names those two. The planned construction can
            separately give a synthesis stage an allowance, as
            `trotter_error_budgeted` does for each node. The tuple is not a
            complete list: an empty tuple does not mean that every other stage
            has an allowance. A planned construction that has its own bound for
            such a stage still reports that bound.
    """

    allowed: bool
    certificate_status: str
    reason: str
    unbudgeted_error_stages: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True, eq=False)
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
            Low–Somma `gamma`, `R`, `h` and `J`, the Apel–Berntson `mu`,
            `R`, `Delta`, `eta`, `h`, `N` and component bounds, or the
            support and residual terms of a discrete-weight artifact.
            Each error component of the pair records its allowance here as
            `allowance_<component>`: the explicit `component_allowances`,
            or by default the share of `epsilon` that `LCHSProblemContext`
            states. A problem without a source also records here the factor
            `homogeneous_coefficient_compensation` applied to its
            coefficients.
        nodes: Ordered real k values, as in `quadrature`, as a read-only
            float64 array.
        base_weights: Integration weights before multiplication by the
            kernel, as a read-only float64 array.
        coefficients: Complex coefficients in node order, as a read-only
            complex128 array. A provider table
            holds the raw coefficients c_j. For a problem without a source
            they are `exp(s*T)*c_j*exp(i*phi_j)` for the signed shift s and
            any extracted scalar phase phi_j, divided by
            `2**recovery_exponent`, and the physical recovery carries
            `2**recovery_exponent`. Their phases are applied in the SELECT
            circuit.
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
        recovery_exponent: Default `0`. The power of two `2**e` that the
            physical recovery carries outside `coefficients`. For a problem
            without a source, `e = e' - 1 - k`, with `e'` the binary exponent
            of `exp(s*T)`, about `floor(s*T/ln 2) + 1`, and `k` (0 to 52) the
            exact power of two that lifts every nonzero real and imaginary
            part of the raw table into the normal binary64 range. The
            coefficients then carry the factor `exp(s*T)*2**-e`, between 1
            and `2**53`, so none underflows under strong known decay and
            their magnitudes and phases keep binary64 relative precision.
            For a table without subnormal parts, `e` is `0` when `exp(s*T)`
            lies in [1, 2).
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
    nodes: np.ndarray
    base_weights: np.ndarray
    coefficients: np.ndarray
    coefficient_l1_norm: float
    quadrature: LCHSQuadraturePlan
    compatibility: LCHSPairCompatibility
    mps_decomposition: object | None = None
    recovery_exponent: int = 0

    def __post_init__(self) -> None:
        for field_name, dtype in (("nodes", np.float64), ("base_weights", np.float64), ("coefficients", np.complex128)):
            object.__setattr__(self, field_name, _read_only_vector(getattr(self, field_name), dtype))

    __eq__ = _fields_equal

    def prep_amplitudes(self, *, max_bytes=None):
        """Return the coefficient-state amplitudes `sqrt(|c_j|/alpha)`, zero-padded.

        alpha is `coefficient_l1_norm`, and the array length is the next
        power of two at or above `len(coefficients)`. The complex phases of
        c_j are not in this state. The SELECT circuit applies them (An,
        Childs and Lin, arXiv:2312.03916v2, Lemma 24, Eq. (178), with the
        phases moved from the two preparation oracles into SELECT).

        Args:
            max_bytes (int | None): Default `None`, the physical memory of this
                computer. Upper limit on the arrays, 96 bytes per address
                slot, checked first.

        Returns:
            amplitudes (numpy.ndarray): Read-only amplitudes, one per address
                slot.
        """
        from nwqlib.subroutines.lcu.data import _coefficient_bookkeeping
        from nwqlib.operators.inputs import _freeze_array
        # Per padded slot: 16 bytes for the complex128 coefficient copy and the
        # 80 bytes of the _coefficient_bookkeeping arrays. The 18 bytes of the
        # lift scan are released before those arrays exist.
        _check_bytes(96 * (1 << max(0, len(self.coefficients)-1).bit_length()), max_bytes,
                     "coefficient PREP arrays")
        # The exact lift 2**k (coefficient_lift) brings subnormal parts into
        # the normal range before the magnitudes are formed. The amplitudes
        # sqrt(2**k*|c_j|/(2**k*alpha)) do not depend on it.
        values = np.array(self.coefficients, dtype=np.complex128)
        values *= 2.0**coefficient_lift(values)
        _, _, amplitudes, _, _, _ = _coefficient_bookkeeping(values, dimension=1, coefficient_atol=0)
        return _freeze_array(amplitudes)

    def decompose_mps(self, max_bond_dim=None, threshold=1e-14, *, reuse=None,
                      max_bytes=None, max_svd_work=None):
        """Return the TT-SVD (MPS) decomposition of the `prep_amplitudes` tensor.

        An attached or supplied decomposition is reused only when it was made
        from this exact normalized tensor with the same `max_bond_dim` and
        `threshold`. Otherwise a new TT-SVD runs under `max_bytes` and
        `max_svd_work`. No LCHS solve or circuit is run.

        Args:
            max_bond_dim (int | None): Upper limit on the kept bond
                dimension. `None` means no limit.
            threshold (float): Singular-value truncation threshold.
            reuse (MPSDecomposition | None): Decomposition to reuse instead
                of the attached `mps_decomposition`.
            max_bytes (int | None): Default `None`, the physical memory of this
                computer. Upper limit on the known arrays.
            max_svd_work (int | None): Default `None`, no limit. Upper limit
                on the counted TT-SVD work.

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
        width and address structure, the pair's certificate status
        `solution_error_certificate_status`, the
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
    """Evaluate f(z) = exp(-(1+i*z)**beta)/C_beta, the near-optimal kernel of ACL arXiv:2312.03916v2, Eq. (7).
    """

    return np.exp(-(1.0 + 1.0j * z_value) ** beta) / eq7_cbeta(beta)


def eq7_coefficient(beta: float, k_value: float) -> complex:
    """Evaluate g(k) = f(k)/(1-i*k), the integrand weight of ACL arXiv:2312.03916v2, Eqs. (6) and (60).
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
    satisfies the binary64 evaluation of the bound that is recorded.

    Args:
        beta: Eq. (7) exponent in (0, 1).
        epsilon: Tail allowance in (0, 1), in the operator norm of exp(-A*T).
            The default pair passes half the construction tolerance.

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
    ``[-K, K]`` used by the composite Gauss rule. The Gauss-Legendre base
    rule is evaluated once and affinely remapped to each interval.

    With m = interval_count_each_side, the panels are [j*h1, (j+1)*h1] for
    j = -m, ..., m-1, as in Eq. (61). The map x -> h1*(x+1)/2 + j*h1 takes a
    base node on [-1, 1] into panel j and scales its weight by h1/2 (ACL
    arXiv:2312.03916v2, Appendix A.4).

    The base nodes and weights are SciPy's binary64 ``roots_legendre`` values,
    and the map rounds again. The quadrature bound of ``_ellipse_rule`` is for
    the exact Gauss rule on these panels, so the errors of this construction
    are reported under ``native_floating_point``, not ``k_quadrature``.

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
    # The tridiagonal Jacobi rule uses linear workspace, followed by Newton refinement.
    base_nodes, base_weights = roots_legendre(node_count)
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
    # up to the smallest positive binary64 value. Never advertise exact zero.
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


def _check_quadrature_work(required, max_quadrature_work, subject, ledger=None):
    """Check one complete quadrature work count against ``LCHS.max_quadrature_work``, where None sets no limit.

    ``required`` is the complete work count of the chosen rule, known before
    its node arrays are built. A refusal reports this count and the limit.
    An admitted count is stored as ``ledger["quadrature_work"]`` when the
    caller supplied a ledger (``_resolve_coefficient_context``), so provider
    enumeration can record each candidate's admitted work.
    """
    max_quadrature_work = optional_limit(max_quadrature_work, "max_quadrature_work")
    if exceeds(required, max_quadrature_work):
        raise ResourceLimitError(
            f"{subject} {required} work units, exceeding "
            f"LCHS.max_quadrature_work={max_quadrature_work}. "
            f"Raise LCHS.max_quadrature_work to at least {required}.")
    if ledger is not None:
        ledger["quadrature_work"] = required


def dense_select_slot_refusal(*, time, l_norm, tolerance, branches, limit, total_qubits=None):
    """Return the refusal of a dense SELECT whose padded address slots exceed ``LCHS.max_dense_select_slots``.

    ``branches`` is the final branch count when ``total_qubits`` is given
    (``parameters.select_parameters``), and otherwise a lower bound on it
    known before the grid exists (``_build_composite_gauss``). Padding rounds
    branches up to a power of two.

    The final check serves every k grid. Raising the limit or using
    product-formula branches applies to all of them. A shorter time or a looser
    tolerance can shrink only a grid chosen from those controls, so the message
    scopes that advice: an explicit grid keeps its size at any time or tolerance.
    """
    slots = 1 << (branches-1).bit_length()
    size = (f"at least {branches} branches and {slots} padded SELECT slots" if total_qubits is None
            else f"{branches} branches, {slots} padded SELECT slots and {total_qubits} total qubits")
    return ResourceLimitError(
        f"LCHS dense SELECT at time={time}, ||L||={l_norm:g}, approximation_tolerance={tolerance:g} "
        f"needs {size}; max_dense_select_slots={limit}. Raise max_dense_select_slots to at least {slots}, "
        "select supported multiplexed product-formula evolution with its different approximation and "
        "cost, or, on a k grid chosen from the time and tolerance, choose a shorter time or a looser "
        "approximation_tolerance. An explicit k grid such as signed_binary_uniform sets its own size.")


def _ellipse_rule(problem, beta, range_k, strip_margin=LCHS_ELLIPSE_STRIP_MARGIN):
    """Select the least-node Q-point Gauss rule using ATAP Theorem 19.3.

    ATAP ISBN 978-1-61197-239-9, Theorem 19.3, Eq. (19.8), bounds (n+1)-point
    Gauss quadrature for n >= 1, so n = Q-1 and Q >= 2. The bound is for the
    exact Gauss rule, with exact nodes and weights. Apply its scalar result
    to v*F(k)u for arbitrary unit vectors u,v, then take their supremum to
    obtain the same operator-norm bound without a dimension factor.
    The strip margin d = ``strip_margin`` lies in (0, 1). For k = x + i*y with
    |y| <= b = 1-d, Re(1+i*k) = 1-y >= d > 0 and |1-i*k| >= Re(1-i*k) =
    1+y >= d. The principal power of a number with positive real part has
    nonnegative real part for 0 < beta < 1, so Re((1+i*k)**beta) >= 0 and
    |g(k)| <= 1/(C_beta*d). This holds for every d in (0, 1), and d enters b,
    the ellipse radius and the prefactor together. The evolution factor
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
    points per panel. The search minimizes the node count at this beta, d and K
    only. It does not order the complete coherent cost: the rightmost node of m
    panels of Q points is K - (K/(2*m))*(1 - x_max(Q)), with x_max(Q) the
    largest Gauss-Legendre node, so (m, Q) = (3, 2) has 12 nodes below 14*K/15
    while (1, 5) has 10 nodes above 0.95*K, and the coherent joint generator
    is normalized by that largest node magnitude. No matrix operations or sampled maximum are used by this
    scalar selection.

    The search has a finite termination bound independent of
    ``max_quadrature_work``. After its first numerically valid candidate Q1,
    the best total is at most 2*Q1. Later candidates need at least 4*m nodes
    because Q >= 2. Once 4*m exceeds the best total, no later candidate can
    improve it, so the number of candidates tried is M <= floor(Q1/2).
    Numeric failure in forming Q or its bound raises rather than extending
    the search indefinitely. The search completes before the caller checks
    its work, so ``max_quadrature_work`` limits the search work plus the
    chosen rule's construction, not every scalar probe. Finite
    termination is not a uniform cheap-runtime guarantee on all input
    scales.

    Args:
        problem: Scalars T, ||L|| and the quadrature allowance epsilon (the
            default pair passes half the construction tolerance).
        beta: Eq. (7) exponent in (0, 1).
        range_k: Cutoff K. The panels tile [-K, K] exactly.
        strip_margin: The margin d in (0, 1) of the strip |Im k| <= 1 - d,
            default LCHS_ELLIPSE_STRIP_MARGIN. Provider enumeration
            (`selection.PROVIDER_FRONTIER`) also tries d = 0.5.

    Returns:
        A pair (choice, work). choice is (2*m*Q, Q, m, h, bound): the total
        node count, Gauss points per panel, panels per half-axis, panel width
        K/m, and the quadrature bound in operator norm. work is the scalar
        work counted for the candidate panel counts tried.
    """
    d = finite_real(strip_margin, "strip margin d")
    if not 0 < d < 1:
        raise ValueError("the Gauss panel strip margin d must lie in (0, 1)")
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
    quadrature bounds are evaluated at the resolved beta before choosing the
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


def _eq7_provider_coefficients(k_values: np.ndarray, context: Mapping[str, Any]) -> np.ndarray:
    """Evaluate g(k) of ``_eq7_provider_coefficient`` at an array of k."""
    return eq7_coefficient(float(context["kernel_parameters"]["beta"]), k_values)


def _cauchy_provider_coefficient(
    k_value: float,
    _context: Mapping[str, Any],
) -> complex:
    """Evaluate the Cauchy density 1/(pi*(1+k**2)) as the coefficient kernel.

    This is the original LCHS identity of An, Liu and Lin (PRL 131, 150603,
    2023, doi:10.1103/PhysRevLett.131.150603), which ACL arXiv:2312.03916v2
    restates as Eq. (5) and, as f(z) = 1/(pi*(1+i*z)), as Eq. (181). The
    composite-Gauss cutoff and node-count formulas are specific to Eq. (7), so this
    kernel pairs only with the uncertified signed-binary grid.
    """
    return complex(1.0 / (pi * (1.0 + k_value**2)))


def _cauchy_provider_coefficients(k_values: np.ndarray, _context: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the Cauchy density of ``_cauchy_provider_coefficient`` at an array of k."""
    return (1.0 / (pi * (1.0 + k_values**2))).astype(np.complex128)


def low_somma_fhat_2(k_value: float, *, gamma: float, c_value: float) -> complex:
    """Evaluate the fixed ``j=2, y=1`` Low-Somma Fourier profile.

    The profile is the kernel f_hat_{j,y} of Low and Somma,
    arXiv:2508.19238v2, Eq. (6), at j=2 and y=1, which Theorems 2 and 3 use.
    It reduces to sqrt(2/pi)*exp(c-(k**2+1)/(4*gamma**2))*exp(-i*k*c)/(1+k**2).
    The 1/sqrt(2*pi) prefactor of their LCHS integral (Eq. (7)) is applied by
    _low_somma_provider_coefficient, not here.

    exp(c) and the Gaussian factor share one exponent, so a representable
    value does not overflow in exp(c) first. An unrepresentable magnitude
    raises OverflowError.
    """

    value = complex(_low_somma_profile(k_value, gamma=gamma, c_value=c_value))
    if not (isfinite(value.real) and isfinite(value.imag)):
        raise OverflowError("the Low-Somma profile magnitude is not representable")
    return value


def _low_somma_profile(k_values, *, gamma: float, c_value: float):
    """Evaluate the profile of ``low_somma_fhat_2`` at a float or a float array of k, nonfinite where it overflows."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (sqrt(2.0 / pi) * np.exp(c_value - (k_values**2 + 1.0) / (4.0 * gamma**2))
                * np.exp(-1.0j * k_values * c_value) / (1.0 + k_values**2))


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


def _low_somma_provider_coefficients(k_values: np.ndarray, context: Mapping[str, Any]) -> np.ndarray:
    """Evaluate ``_low_somma_provider_coefficient`` at an array of k, nonfinite where the profile overflows."""
    return _low_somma_profile(k_values, gamma=float(context["derived_parameters"]["gamma"]),
                              c_value=float(context["kernel_parameters"]["c"])) / sqrt(2.0 * pi)


def apel_berntson_kernel(k_value: float, *, radius: float, mu: float, delta: float) -> complex:
    """Evaluate the compact Apel-Berntson kernel g_R(k) at a real node with |k| < R.

    With R, mu and Delta from the kernel's radius rule,
    g_R(k) = exp((1-mu)*Delta)/(pi*(1+k**2))
    * exp(-1j*(Delta/pi)*log((R+k)/(R-k))) on (-R, R), and its integral
    against exp(-i*t*(k*L_s+H)) approximates exp(-t*(L_s+iH)) within the
    continuous allowance epsilon_c (Apel and Berntson, arXiv:2610.03669v1,
    Theorem 4 and Proposition 10). Its magnitude is beta_R times the truncated Cauchy density
    1/(2*atan(R)*(1+k**2)), with beta_R = (2*atan(R)/pi)*exp((1-mu)*Delta)
    = mu*exp((1-mu)*Delta).

    The phase uses the real logarithm log1p(k/R) - log1p(-k/R) of the
    positive ratio (R+k)/(R-k). The lower-half-plane branch of the paper's
    kernel gives log((k+R)/(k-R)) = pi*sinh(xi) + i*pi under
    k = R*tanh((pi/2)*sinh(xi)), so the principal complex logarithm of the
    negative ratio (k+R)/(k-R) must not be used. Near ±R a rounded k loses
    this phase, and the tanh-sinh rule evaluates it from xi instead.

    Raises:
        ValueError: If |k| >= R, where the kernel is not defined.
    """

    if not abs(k_value) < radius:
        raise ValueError("the Apel-Berntson kernel is defined only for |k| < R")
    phase = (delta / pi) * (log1p(k_value / radius) - log1p(-k_value / radius))
    magnitude = exp((1.0 - mu) * delta) / (pi * (1.0 + k_value * k_value))
    return complex(magnitude * cos(phase), -magnitude * sin(phase))


def _apel_provider_coefficient(k_value: float, context: Mapping[str, Any]) -> complex:
    """Evaluate g_R(k) at the derived R, mu and Delta of the Apel-Berntson pair."""
    derived = context["derived_parameters"]
    return apel_berntson_kernel(k_value, radius=float(derived["R"]), mu=float(derived["mu"]),
                                delta=float(derived["Delta"]))


def _resolve_apel_berntson(
    supplied: Mapping[str, ProviderParameter],
) -> Mapping[str, ProviderParameter]:
    """Resolve the kernel radius, `None` (default) for the root rule of Theorem 6 or a finite positive R.

    Without a radius the root of Apel and Berntson, arXiv:2610.03669v1,
    Theorem 6, gives the radius that minimizes their R*beta. A supplied R selects the
    minimum-beta kernel of that radius (their Proposition 10), so a
    planner can choose R for its own cost.
    """
    radius = supplied.get("radius")
    if radius is None:
        return {"radius": None}
    radius = float(radius)
    if not isfinite(radius) or radius <= 0.0:
        raise ValueError("apel_berntson parameter 'radius' must be finite and positive")
    return {"radius": radius}


def _apel_radius(epsilon_c: float, radius: float | None) -> tuple[float, float, float, int]:
    """Return mu, R, Delta and the bisection step count of the Apel-Berntson kernel.

    Apel and Berntson, arXiv:2610.03669v1, Theorem 6, with the continuous
    allowance epsilon_c in (0, 1). Without a radius, mu is the root on
    (0, 1) of log((1-mu)/epsilon_c) = pi*mu**2/sin(pi*mu) and
    R = tan(pi*mu/2). The left side decreases from log(1/epsilon_c) > 0 and
    the right side increases from 0 to infinity, so the root is unique and
    bisection finds it. With a supplied R, mu = (2/pi)*atan(R), the
    minimum-beta kernel of that radius (Proposition 10). Both use
    Delta = log((1-mu)/epsilon_c)/mu, which must be positive, so a supplied
    R must have 1-mu > epsilon_c. These are binary64 evaluations, and the
    rounding of mu, R and Delta is not enclosed.
    """
    if not 0.0 < epsilon_c < 1.0:
        raise ValueError("the Apel-Berntson continuous allowance must lie in (0, 1)")
    log_allowance = log(epsilon_c)
    steps = 0
    if radius is None:
        lower, upper = 0.0, 1.0
        while True:
            middle = lower + (upper - lower) / 2
            if middle == lower or middle == upper:
                break
            steps += 1
            # sin(pi*mu) = sin(pi*(1-mu)) keeps relative accuracy near mu = 1.
            excess = (log1p(-middle) - log_allowance
                      - pi * middle * middle / sin(pi * min(middle, 1.0 - middle)))
            if excess >= 0.0:
                lower = middle
            else:
                upper = middle
        mu = lower
        radius = tan(pi * mu / 2)
    else:
        mu = 2.0 * atan(radius) / pi
    delta = (log1p(-mu) - log_allowance) / mu if mu > 0.0 else float("nan")
    if not (isfinite(radius) and radius > 0.0 and isfinite(delta) and delta > 0.0):
        raise ValueError(
            f"Apel-Berntson radius R={radius!r} gives no positive Delta for the continuous allowance "
            f"{epsilon_c!r}. A supplied R needs (2/pi)*atan(R) < 1 - epsilon_c")
    return mu, radius, delta, steps


def _apel_strip_limit(mu: float, radius: float, delta: float) -> float:
    """Return the supremum of the strip half-widths eta that the discretization bound allows.

    For a supplied radius, the strip argument behind the bound needs all of
    eta < pi/6, eta < asin(1-mu), D_eta = 1 - (pi**2/4)*(1+R**2)*sin(eta)**2 > 0
    and Delta*sin(eta) < pi*cos(eta). For the root-rule radius the paper
    states the range eta < min(pi/6, asin(2/(pi*sqrt(1+R**2)))), which is the
    first and third conditions. Each condition holds on an interval
    [0, limit), so the minimum of the four limits admits only strip widths
    valid under both statements.
    """
    return min(pi / 6, asin(1.0 - mu), asin(2.0 / (pi * hypot(1.0, radius))), atan(pi / delta))


def _apel_log_strip_constant(eta: float, mu: float, radius: float, delta: float, scaled_time: float) -> float:
    """Return log M_eta of the Apel-Berntson discretization bound.

    M_eta = 4*R*exp((1-mu)*Delta + Delta*sin(eta))
    / (D_eta*(pi*cos(eta) - Delta*sin(eta)))
    * exp(R*pi*sin(eta)/(2 - pi**2*sin(eta)**2/2) * T_e*B_Ls), from Apel
    and Berntson, arXiv:2610.03669v1, Theorem 8, with the time and norm of
    the dissipative part entering through scaled_time = T_e*B_Ls. At a
    supplied radius this reuses their proof under the conditions of
    `_apel_strip_limit`.
    """
    sine = sin(eta)
    d_eta = 1.0 - (pi / 2 * hypot(1.0, radius) * sine) ** 2
    margin = pi * cos(eta) - delta * sine
    if not (d_eta > 0.0 and margin > 0.0):
        raise ValueError("Apel-Berntson strip half-width violates D_eta > 0 or Delta*sin(eta) < pi*cos(eta)")
    return (log(4.0 * radius) + (1.0 - mu) * delta + delta * sine - log(d_eta) - log(margin)
            + radius * pi * sine / (2.0 - pi * pi * sine * sine / 2.0) * scaled_time)


def _log_expm1(value: float) -> float:
    """Return log(exp(value) - 1) for value > 0 without overflow or cancellation."""
    return value + log(-expm1(-value))


# Each rounding guard below shrinks the tanh-sinh step by this relative
# amount, or adds one node per side, and gives up after this many tries.
# They absorb binary64 rounding in the inverted step and truncation
# formulas, which is a few ulps of a quantity of order one.
_APEL_ROUNDING_STEP = 2.0**-44
_APEL_ROUNDING_TRIES = 64
# The strip half-widths tried are eta_limit*j/_APEL_STRIP_FRACTIONS for
# j = 1.._APEL_STRIP_FRACTIONS-1. Each try is one scalar inversion of the
# step and truncation bounds. The resolution of this set is a working
# choice, not a tuned optimum.
# ponytail: fixed strip fractions; a bounded scalar minimization of the node count over eta would find its optimum.
_APEL_STRIP_FRACTIONS = 16
# Scalar work units charged per radius bisection step and per strip width tried.
_APEL_SCALAR_STEP_WORK = 32


def _apel_tanh_sinh_grid(eta, mu, radius, delta, scaled_time, epsilon_disc, epsilon_trunc):
    """Return the step h, half-range N and both quadrature bounds of the tanh-sinh rule at one eta.

    Apel and Berntson, arXiv:2610.03669v1, Theorem 8, at the default radius,
    and its adaptation under `_apel_strip_limit` at a supplied radius, bound
    the operator error of the finite sum over n = -N..N by
    2*M_eta/(exp(2*pi*eta/h) - 1) for the step plus (4*R*exp((1-mu)*Delta)/pi)*exp(-pi*sinh(N*h)) for the
    truncation. Each term is inverted against its allowance:
    h = 2*pi*eta/log(1 + 2*M_eta/epsilon_disc) and the least N with
    N*h >= asinh(log(4*R*exp((1-mu)*Delta)/(pi*epsilon_trunc))/pi). Both bounds
    are then evaluated in log form at the chosen h and N, and the guards
    make those binary64 values meet their allowances.
    """
    log_m = _apel_log_strip_constant(eta, mu, radius, delta, scaled_time)
    step = 2.0 * pi * eta / float(np.logaddexp(0.0, log(2.0) + log_m - log(epsilon_disc)))

    def discretization(h):
        return _positive_bound(log(2.0) + log_m - _log_expm1(2.0 * pi * eta / h))

    disc = discretization(step)
    for _ in range(_APEL_ROUNDING_TRIES):
        if disc <= epsilon_disc:
            break
        step *= 1.0 - _APEL_ROUNDING_STEP
        disc = discretization(step)
    log_prefactor = log(4.0 * radius / pi) + (1.0 - mu) * delta
    half_count = max(0, ceil(asinh((log_prefactor - log(epsilon_trunc)) / pi) / step))
    trunc = _positive_bound(log_prefactor - pi * sinh(half_count * step))
    for _ in range(_APEL_ROUNDING_TRIES):
        if trunc <= epsilon_trunc:
            break
        half_count += 1
        trunc = _positive_bound(log_prefactor - pi * sinh(half_count * step))
    if disc > epsilon_disc or trunc > epsilon_trunc:
        raise ValueError("Apel-Berntson tanh-sinh rule cannot establish its allowances numerically")
    return step, half_count, disc, trunc


def _resolve_discrete_weights(
    supplied: Mapping[str, ProviderParameter],
) -> Mapping[str, ProviderParameter]:
    """Resolve `artifact_id`, the content hash of the LCHSDiscreteWeightArtifact whose weights are used.

    The identity is part of the resolved kernel parameters, so a coefficient
    plan names the prepared weights it was built from. The artifact itself
    is passed as `coefficient_artifact` when the coefficients are formed.
    """
    artifact_id = supplied.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.startswith("sha256:"):
        raise ValueError("discrete_weights parameter 'artifact_id' must be the content_id of an "
                         "LCHSDiscreteWeightArtifact")
    return {"artifact_id": artifact_id}


def _discrete_weight_coefficient(k_value: float, _context: Mapping[str, Any]) -> complex:
    """Refuse an evaluation off the artifact nodes, where prepared weights have no kernel."""
    raise ValueError("discrete weights are defined only at the nodes of their artifact")


LCHS_KERNEL_IMPLEMENTATIONS: dict[str, LCHSKernelProvider] = {
    "near_optimal_eq7": LCHSKernelProvider(
        name="near_optimal_eq7",
        formula_id="acl_eq7_v1",
        resolve_parameters=_resolve_eq7,
        coefficient=_eq7_provider_coefficient,
        coefficients=_eq7_provider_coefficients,
        fixed_parameters=_immutable_map({}),
    ),
    "cauchy_density": LCHSKernelProvider(
        name="cauchy_density",
        formula_id="cauchy_density_v1",
        resolve_parameters=_resolve_no_parameters,
        coefficient=_cauchy_provider_coefficient,
        coefficients=_cauchy_provider_coefficients,
        fixed_parameters=_immutable_map({}),
    ),
    "low_somma_f2": LCHSKernelProvider(
        name="low_somma_f2",
        formula_id=LOW_SOMMA_FORMULA_ID,
        resolve_parameters=_resolve_low_somma_f2,
        coefficient=_low_somma_provider_coefficient,
        coefficients=_low_somma_provider_coefficients,
        fixed_parameters=_immutable_map({"j": 2, "y": 1}),
    ),
    "apel_berntson": LCHSKernelProvider(
        name="apel_berntson",
        formula_id="apel_berntson_compact_v1",
        resolve_parameters=_resolve_apel_berntson,
        coefficient=_apel_provider_coefficient,
        fixed_parameters=_immutable_map({}),
    ),
    # Weights of an LCHSDiscreteWeightArtifact, which the artifact_nodes rule
    # supplies as kernel values. They exist only at the artifact nodes.
    "discrete_weights": LCHSKernelProvider(
        name="discrete_weights",
        formula_id="fixed_node_discrete_weights_v1",
        resolve_parameters=_resolve_discrete_weights,
        coefficient=_discrete_weight_coefficient,
        fixed_parameters=_immutable_map({}),
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
    max_bytes=None,
    max_quadrature_work=None,
) -> LCHSQuadraturePlan:
    """Build the symmetric composite Gauss rule and its available component bounds.

    ACL arXiv:2312.03916v2, Eq. (61), defines the panelled k quadrature and
    Eq. (186) the finite tail bound used to select the cutoff. The tail
    meets the `kernel_approximation` allowance and the panels the
    `k_quadrature` allowance, by default half of the construction tolerance
    each. The Q-point panel
    rule follows Trefethen, Approximation Theory and Approximation Practice,
    ISBN 978-1-61197-239-9, Theorem 19.3, with n=Q-1. The panel search
    divides the tail-certified K into whole panels (h1 = K/m) and selects
    the fewest nodes meeting the separate quadrature budget. Byte and work
    limits are checked before the node arrays are built. Other
    physical-output components remain outside this tolerance.

    For the composite Gauss rule, planning first completes the scalar panel
    search. It stops when the minimum node count of every later panel count exceeds the
    best rule found. It then checks ``max_quadrature_work`` against the
    search work, 32 units per chosen node and the quadratic Gauss-rule
    construction work, before allocating the node arrays. A refusal names
    the complete work of that rule. The chosen rule has
    ``total = 2*m*Q`` nodes after M candidate panel counts, each counted as 32
    units, so ``W_quad = 32*M + 32*(2*m*Q) + Q**2``: scalar selection, node,
    weight and kernel evaluation, and the tridiagonal Q-point Legendre rule.
    These are engineering work units, not measured equal-cost operations.

    On a quantum dense_exact route the caller passes
    ``dense_select_slot_limit`` in ``pair_context``, and a rule whose node
    count alone already proves the SELECT too large is refused before the
    Gauss rule is built. The homogeneous SELECT has exactly ``total`` branches.
    A constant-source dense_exact SELECT uses the compact layout of
    ``source_selection.select_source_branches``, in which each retained
    application holds ``total`` slots, zero amplitudes included, and omission
    removes only whole applications. At least one application remains, so the
    final count is n = total*A >= total for A >= 1 retained applications.
    Padding is monotone in n, so 2**ceil(log2(total)) lower-bounds the padded
    slots that ``parameters.select_parameters`` compares with the same limit. A
    provisional application count is not a valid factor, because omission can
    reduce it, so a layout that exceeds the limit only through its applications
    is left to that later exact check. Classical execution, the product-formula
    backends and ``resolve_lchs_coefficient_plan`` synthesize no controlled
    dense exponential per address slot, so they pass None and no slot check
    applies.
    """
    beta = float(pair_context["kernel_parameters"]["beta"])
    multiplier = float(resolved.parameters["truncation_multiplier"])
    allowances = pair_context["component_allowances"]
    range_k = finite_real(multiplier * eq7_truncation_range(beta, allowances["kernel_approximation"]),
                          "LCHS cutoff")
    # The strip margin d of the Gauss panel bound: the default, or the d = 0.5 of provider enumeration.
    strip_margin = pair_context["derived_parameters"].get("strip_margin", LCHS_ELLIPSE_STRIP_MARGIN)
    choice, selection_work = _ellipse_rule(replace(problem, epsilon=allowances["k_quadrature"]), beta, range_k,
                                           strip_margin)
    total, node_count, interval_count_each_side, h1, quadrature_bound = choice
    # Bytes: 128 per selected node is an envelope for the float64 node and
    # weight arrays and their read-only copies in LCHSQuadraturePlan, and the
    # complex128 kernel values, coefficients and read-only copy that
    # _resolve_coefficient_context forms later, with their temporaries.
    # The base rule uses O(Q) arrays, covered by this node envelope with total >= 2Q.
    _check_bytes(128*total, max_bytes, "LCHS quadrature arrays")
    # Work: the candidate search, 32 scalar operations per node for the panel
    # map and kernel evaluation, and Q**2 for the tridiagonal Legendre rule.
    # The per-node 128 bytes and 32 operations are untuned envelopes registered
    # in ENGINEERING_CONSTANTS.
    _check_quadrature_work(selection_work + 32*total + node_count**2, max_quadrature_work,
                           "LCHS quadrature selection and construction require", pair_context.get("work_ledger"))
    slot_limit = pair_context.get("dense_select_slot_limit")
    if slot_limit is not None and 1 << (total-1).bit_length() > slot_limit:
        raise dense_select_slot_refusal(time=problem.final_time, l_norm=problem.l_norm, tolerance=problem.epsilon,
                                        branches=total, limit=slot_limit)
    nodes, weights = composite_gauss_grid(
        interval_count_each_side=interval_count_each_side,
        h1=h1,
        node_count=node_count,
    )
    approximate_bound = eq7_tail_bound(beta, range_k)
    return LCHSQuadraturePlan(
        nodes=nodes,
        base_weights=weights,
        range_k=float(range_k),
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
    max_bytes=None,
    max_quadrature_work=None,
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
    byte_limit = memory_limit(max_bytes)
    if byte_limit is not None and num_qubits >= byte_limit.bit_length():
        raise ResourceLimitError(f"signed-binary coefficient grid of {num_qubits} qubits needs at least "
                                 f"2**{num_qubits} bytes, exceeding {byte_allowance(max_bytes)}")
    state_count = 2**num_qubits
    # Same per-node byte envelope and per-node scalar work as the composite
    # Gauss builder, with no Legendre eigenproblem.
    _check_bytes(128*state_count, max_bytes, "signed-binary coefficient grid")
    _check_quadrature_work(32*state_count, max_quadrature_work,
                           "LCHS signed-binary coefficient grid requires", _pair_context.get("work_ledger"))
    # The highest address bit has negative weight. Computational-basis order
    # therefore wraps from positive to negative nodes at the sign transition.
    sign_transition = 2 ** (num_qubits - 1)
    delta_k = float(2.0**lsb_position)
    states = np.arange(state_count)
    nodes = np.where(states < sign_transition, states, states - state_count) * delta_k
    weights = np.full(state_count, delta_k)
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
        range_k=float(np.max(np.abs(nodes))),
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
    max_bytes=None,
    max_quadrature_work=None,
) -> LCHSQuadraturePlan:
    """Build the symmetric finite rule and record its power-of-two padding.

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
    _check_bytes(128*(2*index_limit+1), max_bytes, "symmetric coefficient grid")
    _check_quadrature_work(32*(2*index_limit+1), max_quadrature_work,
                           "LCHS symmetric coefficient grid requires", pair_context.get("work_ledger"))
    nodes = np.arange(-index_limit, index_limit + 1) * step
    node_count = 2 * index_limit + 1
    padded_count = 1 << max(0, node_count - 1).bit_length()
    return LCHSQuadraturePlan(
        nodes=nodes,
        base_weights=np.full(node_count, step),
        range_k=radius,
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


def _build_tanh_sinh(
    problem: LCHSProblemContext,
    _resolved: ResolvedProviderConfig,
    pair_context: Mapping[str, Any],
    max_bytes=None,
    max_quadrature_work=None,
) -> LCHSQuadraturePlan:
    """Build the Apel-Berntson tanh-sinh nodes, weights and kernel values.

    With phi_R(xi) = R*tanh((pi/2)*sinh(xi)), the nodes are
    k_n = phi_R(n*h) for n = -N..N with weights h*phi_R'(n*h), and each
    node's kernel value is g_R(k_n) = exp((1-mu)*Delta)/(pi*(1+k_n**2))
    * exp(-1j*Delta*sinh(n*h)). The phase uses
    log((R+k)/(R-k)) = pi*sinh(xi) and so needs no logarithm of the
    rounded node. Their product is the coefficient c_n of Apel and
    Berntson, arXiv:2610.03669v1, with the branch of apel_berntson_kernel,
    and c_{-n} = conj(c_n). phi_R' = R*(pi/2)*cosh(xi)/cosh(u)**2 with
    u = (pi/2)*sinh(xi) is evaluated in log form, so cosh(u)**2 cannot
    overflow first.

    The work charge is the profile's scalar selection plus 32 units per
    node. The byte charge is 160 per node: the 128 of the other rules plus
    the complex128 kernel values and their read-only copy. Both are checked before any
    node array is formed.
    """
    derived = pair_context["derived_parameters"]
    half_count, step = int(derived["N"]), float(derived["h"])
    radius, mu, delta = float(derived["R"]), float(derived["mu"]), float(derived["Delta"])
    count = 2 * half_count + 1
    _check_bytes(160 * count, max_bytes, "LCHS tanh-sinh coefficient grid")
    _check_quadrature_work(int(derived["selection_work"]) + 32 * count, max_quadrature_work,
                           "LCHS tanh-sinh selection and grid require", pair_context.get("work_ledger"))
    xi = step * np.arange(-half_count, half_count + 1, dtype=float)
    inner = (pi / 2) * np.sinh(xi)
    nodes = radius * np.tanh(inner)
    # log cosh(x) = logaddexp(x, -x) - log(2).
    log_weights = (log(step * radius * pi / 2) + np.logaddexp(xi, -xi)
                   - 2.0 * np.logaddexp(inner, -inner) + log(2.0))
    weights = np.exp(log_weights)
    values = (exp((1.0 - mu) * delta) / (pi * (1.0 + nodes * nodes))) * np.exp(-1j * delta * np.sinh(xi))
    return LCHSQuadraturePlan(
        nodes=nodes,
        base_weights=weights,
        range_k=radius,
        interval_count_each_side=half_count,
        node_count=count,
        node_order="ascending_tanh_sinh",
        address_structure=_immutable_map({"affine": False, "kind": "explicit_node_table"}),
        physical_node_count=count,
        padding_count=0,
        approximate_lchs_error_bound=float(derived["kernel_allowance"]),
        quadrature_error_bound=float(derived["discretization_bound"]) + float(derived["truncation_bound"]),
        kernel_values=values,
    )


def _build_artifact_nodes(
    problem: LCHSProblemContext,
    _resolved: ResolvedProviderConfig,
    pair_context: Mapping[str, Any],
    max_bytes=None,
    max_quadrature_work=None,
) -> LCHSQuadraturePlan:
    """Use the nodes and weights of an LCHSDiscreteWeightArtifact after checking that they apply.

    The artifact's residual bound holds for exp(-t*(L_s+iH)) whenever
    t*||L_s|| lies in its support [0, b], so every elapsed time up to T_e is
    covered when T_e*B_Ls <= b. The product is compared exactly in rational
    arithmetic. Its bound plus rounding estimate must also meet the
    `discrete_residual` allowance. Neither check repeats the optimization
    or the residual evaluation. The bound covers the complete finite sum, so
    the quadrature component is recorded as an exact zero and the base
    weights are one.
    """
    artifact = pair_context["coefficient_artifact"]
    residual = artifact.residual
    if problem.final_time < 0 or problem.l_norm < 0:
        raise ValueError("discrete weights require nonnegative T_e and B_Ls")
    if Fraction(problem.final_time) * Fraction(problem.l_norm) > Fraction(residual.support_endpoint):
        raise ValueError(
            f"discrete-weight artifact {artifact.content_id} covers T_e*B_Ls <= {residual.support_endpoint!r}, "
            f"but this problem has T_e*B_Ls = {problem.final_time * problem.l_norm!r}. Prepare weights "
            "for a support endpoint at least this product")
    bound = residual.residual_bound + residual.rounding_estimate
    allowance = pair_context["component_allowances"]["discrete_residual"]
    if bound > allowance:
        raise ValueError(
            f"discrete-weight artifact {artifact.content_id} has residual bound {bound!r} including its "
            f"rounding estimate, above the discrete_residual allowance {allowance!r}")
    count = len(artifact.nodes)
    # Same per-node byte envelope and scalar work as the composite-Gauss rule.
    _check_bytes(128 * count, max_bytes, "LCHS discrete-weight coefficient table")
    _check_quadrature_work(32 * count, max_quadrature_work, "LCHS discrete-weight coefficient table requires",
                           pair_context.get("work_ledger"))
    return LCHSQuadraturePlan(
        nodes=artifact.nodes,
        base_weights=np.ones(count),
        range_k=max(abs(node) for node in artifact.nodes),
        interval_count_each_side=0,
        node_count=count,
        node_order="artifact_order",
        address_structure=_immutable_map({"affine": False, "kind": "explicit_node_table"}),
        physical_node_count=count,
        padding_count=0,
        approximate_lchs_error_bound=bound,
        quadrature_error_bound=0.0,
        kernel_values=artifact.weights,
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
    "tanh_sinh": LCHSQuadratureProvider(
        name="tanh_sinh",
        formula_id="apel_berntson_tanh_sinh_kk3_v1",
        resolve_parameters=_resolve_no_parameters,
        build_nodes_and_weights=_build_tanh_sinh,
    ),
    "artifact_nodes": LCHSQuadratureProvider(
        name="artifact_nodes",
        formula_id="discrete_weight_artifact_nodes_v1",
        resolve_parameters=_resolve_no_parameters,
        build_nodes_and_weights=_build_artifact_nodes,
    ),
}


# Registered kernel-quadrature pairs and the scope of their component bounds.
# (Eq. (7), composite Gauss) is "certified". Its cutoff comes from the
# Eq. (7) decay of Eqs. (185)-(186) and its panels from the strip bound
# |g| <= 1/(C_beta*d), so neither law transfers to the Cauchy kernel, whose
# tail decays only like 1/k**2 (ACL arXiv:2312.03916v2, Eq. (5)). The
# signed-binary grid is a plain rectangle rule chosen for its affine address
# structure and carries no bound. Low-Somma's Theorems 2-3
# (arXiv:2508.19238v2) give both component bounds for their own kernel and
# trapezoid rule, but the pair's split of the tolerance between those two
# components does not cover the synthesis and preparation stages, hence
# "incomplete". This describes the split only: a route can give such a stage its
# own allowance, and a route that supplies a bound for such a stage reports it
# (solution_error_budget.circuit_stage_bounds). Apel-Berntson's Theorem 4 and Proposition 10
# bound its continuous component, Theorem 8 its discretization and truncation
# components at the default radius, and the
# discrete-weight residual bounds its whole finite sum, so both are
# "certified" in the same component sense as Eq. (7). An unlisted pair is
# rejected.
LCHS_KERNEL_QUADRATURE_COMPATIBILITY: Mapping[tuple[str, str], LCHSPairCompatibility] = (
    MappingProxyType(
        {
            ("near_optimal_eq7", "composite_gauss"): LCHSPairCompatibility(
                allowed=True,
                certificate_status="certified",
                reason="ACL arXiv:2312.03916v2 Eq.(186) tail and ATAP ISBN 978-1-61197-239-9 Theorem19.3 Q-point ellipse quadrature bounds, for exact Gauss nodes and weights. Each receives half the construction tolerance. Their binary64 construction and evaluation are reported under native_floating_point. Other physical error components require separate bounds.",
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
            ("apel_berntson", "tanh_sinh"): LCHSPairCompatibility(
                allowed=True,
                certificate_status="certified",
                reason=(
                    "Theorem 4 and Proposition 10 of Apel-Berntson arXiv:2610.03669v1 give the continuous "
                    "operator-error bound at a fixed radius. Theorem 8 gives the discretization and truncation "
                    "bounds at the default radius, and NWQLib reuses that proof at a supplied radius under the "
                    "strip conditions on the LCHS derivations page. Each of these three bounds meets its own "
                    "allowance. Corollary 9 bounds the coefficient mass of the finite rule, not an operator "
                    "error. The preprint's peer-review status is unconfirmed. Its coefficient formula and "
                    "branch were derived independently and checked on two noncommuting 2x2 exponentials. "
                    "These are component bounds, and other physical error components require separate bounds."
                ),
            ),
            ("discrete_weights", "artifact_nodes"): LCHSPairCompatibility(
                allowed=True,
                certificate_status="certified",
                reason=(
                    "Local Wiener-norm residual on the artifact support [0, b] (finite cosine sum plus "
                    "integration-by-parts tail; Ni-Ying arXiv:2606.29848v1 Section 6.1), evaluated in "
                    "binary64 from the prepared weights with a standard-model rounding estimate. It "
                    "applies while T_e*B_Ls <= b and covers the whole finite sum, so there is no separate "
                    "quadrature component. Other physical error components require separate bounds."
                ),
            ),
        }
    )
)


# Error components that each pair bounds, by allowance name, and the
# divisor of the construction tolerance that gives each component its
# default allowance. A pair without components carries no bound.
_PAIR_ERROR_COMPONENTS: Mapping[tuple[str, str], tuple[tuple[str, ...], int]] = MappingProxyType({
    ("near_optimal_eq7", "composite_gauss"): (("kernel_approximation", "k_quadrature"), 2),
    ("low_somma_f2", "symmetric_uniform_trapezoid"): (("kernel_approximation", "k_quadrature"), 3),
    ("apel_berntson", "tanh_sinh"): (("kernel_approximation", "k_discretization", "k_truncation"), 3),
    ("discrete_weights", "artifact_nodes"): (("discrete_residual",), 1),
})


def _component_allowances(problem: LCHSProblemContext, pair_key: tuple[str, str],
                          pair_label: str) -> Mapping[str, float]:
    """Return the allowance of each error component of the pair.

    Without explicit `component_allowances`, each component receives the
    binary64 quotient epsilon/divisor. Explicit allowances must name
    exactly the pair's components, be finite and positive, and sum to at
    most epsilon.
    """
    names, divisor = _PAIR_ERROR_COMPONENTS.get(pair_key, ((), 1))
    supplied = problem.component_allowances
    if supplied is None:
        if not names:
            return _immutable_map({})
        allowance = finite_real(float(problem.epsilon) / divisor, "LCHS component tolerance")
        if allowance == 0:
            raise ValueError("LCHS component tolerance is below the positive binary64 range")
        return _immutable_map({name: allowance for name in names})
    if set(supplied) != set(names):
        raise ValueError(f"LCHS provider pair {pair_label} has error components {list(names)}, "
                         f"but component_allowances names {sorted(supplied)}")
    allowances = {name: finite_real(supplied[name], f"component allowance {name!r}") for name in names}
    if any(value <= 0 for value in allowances.values()) or fsum(allowances.values()) > problem.epsilon:
        raise ValueError(f"LCHS component allowances must be positive with a sum at most epsilon={problem.epsilon!r}")
    return _immutable_map(allowances)


def _resolve_low_somma_pair_profile(
    problem: LCHSProblemContext,
    resolved_kernel: ResolvedProviderConfig,
    allowances: Mapping[str, float],
) -> Mapping[str, ProviderParameter]:
    """Derive gamma, R, h and J from Low and Somma, arXiv:2508.19238v2, Theorems 2-3.

    epsilon_lchs and epsilon_quad are the `kernel_approximation` and
    `k_quadrature` allowances. By default both are epsilon/3, the
    example split stated before Theorem 4 (arXiv:2508.19238v2, p. 6).
    Explicit allowances must each lie in (0, 4/15], the range the default
    split admits.
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
    if problem.component_allowances is None and not 0.0 < epsilon <= 4.0 / 5.0:
        raise ValueError(
            "low_somma_f2 + symmetric_uniform_trapezoid requires epsilon in (0, 4/5]"
        )
    c_value = float(resolved_kernel.parameters["c"])
    epsilon_lchs = allowances["kernel_approximation"]
    epsilon_quad = allowances["k_quadrature"]
    if not (epsilon_lchs <= 4.0 / 15.0 and epsilon_quad <= 4.0 / 15.0):
        raise ValueError("low_somma_f2 + symmetric_uniform_trapezoid requires component allowances in (0, 4/15]")
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


def _resolve_apel_pair_profile(
    problem: LCHSProblemContext,
    resolved_kernel: ResolvedProviderConfig,
    allowances: Mapping[str, float],
) -> Mapping[str, ProviderParameter]:
    """Derive the Apel-Berntson kernel and tanh-sinh grid from their three allowances.

    Apel and Berntson, arXiv:2610.03669v1: Theorem 8 for the discretization
    and truncation bounds at the default radius, with the checked reuse of its
    proof at a supplied radius (docs/algorithms/lchs_derivations.md), and
    Corollary 9 for the coefficient-mass bound, which is not an operator
    error. The
    `kernel_approximation` allowance epsilon_c fixes mu, R and Delta
    (_apel_radius). The strip half-width eta is then chosen among
    eta_limit*j/16 for j = 1..15, all strictly inside the four strip
    conditions, as the one whose grid has the fewest nodes (smallest j
    on a tie). Any such eta gives a valid bound. The choice minimizes the node
    count, not the complete coherent cost, which h and N also change through
    the nodes and coefficients. T_e*B_Ls enters M_eta.

    The returned mapping holds mu, R, Delta, the continuous coefficient
    mass beta_R = mu*exp((1-mu)*Delta), eta, eta_limit, the step h, the
    half-range N, the three allowances, the discretization and truncation
    bounds, the bound 2*C_beta/(exp(2*pi*eta/h) - 1) + truncation on the
    difference between the finite coefficient mass and beta_R (the
    discretization bound with M_eta replaced by
    C_beta = 4*R*exp((1-mu)*Delta)/(pi*cos(eta)*D_eta)), the radius rule and
    the scalar selection work: 32 units per bisection step and per strip
    width tried.
    """
    scaled_time = finite_real(problem.final_time * problem.l_norm, "T_e*B_Ls")
    if problem.final_time < 0 or problem.l_norm < 0:
        raise ValueError("Apel-Berntson selection requires nonnegative T_e and B_Ls")
    epsilon_c = allowances["kernel_approximation"]
    epsilon_disc = allowances["k_discretization"]
    epsilon_trunc = allowances["k_truncation"]
    supplied_radius = resolved_kernel.parameters["radius"]
    mu, radius, delta, steps = _apel_radius(epsilon_c, supplied_radius)
    limit = _apel_strip_limit(mu, radius, delta)
    best = None
    for fraction in range(1, _APEL_STRIP_FRACTIONS):
        eta = limit * fraction / _APEL_STRIP_FRACTIONS
        grid = _apel_tanh_sinh_grid(eta, mu, radius, delta, scaled_time, epsilon_disc, epsilon_trunc)
        if best is None or grid[1] < best[1][1]:
            best = eta, grid
    eta, (step, half_count, disc, trunc) = best
    sine = sin(eta)
    d_eta = 1.0 - (pi / 2 * hypot(1.0, radius) * sine) ** 2
    beta_error = _positive_bound(log(2.0) + log(4.0 * radius / pi) + (1.0 - mu) * delta - log(cos(eta))
                                 - log(d_eta) - _log_expm1(2.0 * pi * eta / step)) + trunc
    beta_continuous = _exp_with_nonfinite_overflow(log(mu) + (1.0 - mu) * delta)
    if not isfinite(beta_continuous):
        raise ValueError("Apel-Berntson coefficient mass exceeds the finite binary64 range")
    return _immutable_map(
        {
            "mu": mu,
            "R": radius,
            "Delta": delta,
            "beta_R": beta_continuous,
            "eta": eta,
            "eta_limit": limit,
            "h": step,
            "N": half_count,
            "kernel_allowance": epsilon_c,
            "discretization_allowance": epsilon_disc,
            "truncation_allowance": epsilon_trunc,
            "discretization_bound": disc,
            "truncation_bound": trunc,
            "beta_discretization_bound": beta_error,
            "radius_rule": "kk1_root" if supplied_radius is None else "supplied",
            "selection_work": _APEL_SCALAR_STEP_WORK * (steps + _APEL_STRIP_FRACTIONS - 1),
        }
    )


def kk6_radius(epsilon_c: float, *, form: str, a: float | None = None, b: float | None = None,
               alpha_a: float | None = None, max_quadrature_work: int | None) -> dict[str, Any]:
    """Propose one Apel-Berntson radius R from the p = 2 stationary relation of KK6, or say why none is generated.

    The proposal is a heuristic finite candidate for provider enumeration
    (`selection.provider_frontier`): the fixed-R kernel at the returned R is
    built and priced by its actual finite grid and coherent row like every
    other candidate. Nothing here claims an optimum of the complete cost, a
    unique root, a p = 4 result or an enclosure of the root.

    With mu = (2/pi) atan R, L(mu) = log((1-mu)/epsilon_c), Delta = L/mu and
    beta_R = mu exp((1-mu) Delta) (KK5, `_apel_radius`), the continuous mass
    satisfies d log(beta_R)/dmu = -L/mu**2, because L' = -1/(1-mu) cancels the
    1/mu terms, so d log(beta_R)/dR = -2L/[pi (1+R**2) mu**2]. Two leading-cost
    surrogates of the coherent route's query normalization use it:

    - Separate parts (K1): J_sep(R) = (a R + b) beta_R**2 with a = alpha_L > 0
      and b = alpha_H >= 0 the query normalizations of the actual compiled
      children (`joint.access_summary(child).alpha`, not a target norm). The
      joint owner normalizes by lambda_q (lambda_k alpha_L + alpha_H); omitting
      the R-independent lambda_q is legitimate for a fixed support, and the
      surrogate equals the table normalization only when lambda_k = R (for
      the Apel nonuniform table lambda_k = max|k_j| <= R). The stationary
      condition d log J_sep/dR = 0 is the zero of
      g_sep(mu) = L - pi mu**2 (1+R**2) a / [4 (a R + b)].
    - Single A (K2): HermitianPart[(k - i) A_0] = k L_0 + H_0, and the
      single-A generator (`plan_single_a_generator`) has alpha_S =
      lambda_q r alpha_A with r = hypot(max|k_j|, 1), so the surrogate is
      J_A(R) = alpha_A sqrt(1+R**2) beta_R**2. Its stationary condition is
      the zero of g_A(mu) = L - (pi/4) mu**2 R; the positive alpha_A cancels
      but must be the known query normalization of the supplied encoding.

    Existence (K3): for 0 < epsilon_c < 1 and U = 1 - epsilon_c, L > 0 on
    0 < mu < U. As mu -> 0+, L -> log(1/epsilon_c) > 0 while the subtracted
    term is O(mu**2) for b > 0, mu/2 + o(mu) for b = 0 and O(mu**3) for
    single A. As mu -> U-, L -> 0 while the subtracted term stays strictly
    positive. Each residual is continuous, so it has an interior root. The
    separate form with a = 0 < b is strictly decreasing in R (no interior
    stationary point), and a = b = 0, zero time or the unitary reduction use
    their exact shortcuts; neither generates a proposal. An unknown or
    nonpositive normalization is not treated as zero: no proposal.

    Procedure (K3, binary64 numerical evidence only): bisect on (0, U),
    never evaluating the limiting endpoints. Each midpoint visit, including
    a domain shrink where the rounded L is not positive and a failing
    evaluation, is charged the 32 scalar units of an Apel radius step
    before it is evaluated, against the per-construction
    ``max_quadrature_work``. The loop stops at a computed zero or when the
    midpoint no longer moves. Without a zero, both residual signs must have
    been observed inside the domain, and the observed endpoint of smaller
    |g| (then smaller mu) is returned; otherwise no proposal. One further
    32-unit block covers R = tan(pi mu/2) and the mandatory fixed-R check
    `_apel_radius(epsilon_c, R)`, a numerical admission, not an enclosure.
    A numerical failure leaves the proposal ungenerated with its spent work;
    no neighbor or second bracket is tried. A block beyond
    ``max_quadrature_work`` raises `nwqlib._limits.ResourceLimitError`
    instead, so the limit refuses the planning rather than removing this
    entry from an enumeration.

    Args:
        epsilon_c: The Apel-Berntson continuous allowance of this Problem
            (the `kernel_approximation` allowance of the AB pair).
        form: ``"separate_parts"`` or ``"single_a"``.
        a: alpha_L of the separate form.
        b: alpha_H of the separate form, 0 for an absent H child.
        alpha_a: alpha_A of the single-A form.
        max_quadrature_work: The per-construction limit
            ``LCHS.max_quadrature_work``.

    Returns:
        A mapping with ``radius`` (None when not generated), ``reason``,
        ``form``, ``epsilon_c``, ``visits``, ``work_units`` (charged scalar
        units), ``residual`` ((mu, g) of the returned point or None),
        ``positive`` and ``negative`` (the last observed (mu, g) of each sign
        or None) and ``numerical_zero``.

    Raises:
        ResourceLimitError: If the next 32-unit block would exceed
            ``max_quadrature_work``.
    """
    record = dict(radius=None, form=form, epsilon_c=epsilon_c, visits=0, work_units=0, residual=None,
                  positive=None, negative=None, numerical_zero=False)

    def refused(reason):
        return {**record, "reason": reason}

    if form == "separate_parts":
        if a is None or b is None or not (isfinite(a) and isfinite(b) and a >= 0 and b >= 0):
            return refused(f"separate-parts query normalizations a={a!r}, b={b!r} are not known nonnegative values")
        if a == 0:
            return refused("a = alpha_L = 0: b*beta_R**2 decreases strictly in R, so there is no interior "
                           "stationary proposal")
    elif form == "single_a":
        if alpha_a is None or not (isfinite(alpha_a) and alpha_a > 0):
            return refused(f"single-A query normalization alpha_A={alpha_a!r} is not a known positive value")
    else:
        raise ValueError(f"KK6 form must be 'separate_parts' or 'single_a', not {form!r}")
    if not 0.0 < epsilon_c < 1.0:
        return refused(f"continuous allowance epsilon_c={epsilon_c!r} is not in (0, 1)")
    lower, upper = 0.0, 1.0 - epsilon_c
    if not 0.0 < upper <= 1.0:
        return refused("no represented domain 0 < mu < 1 - epsilon_c")
    log_epsilon = log(epsilon_c)
    hit = None

    def charge(step):
        need = record["work_units"] + _APEL_SCALAR_STEP_WORK
        if exceeds(need, max_quadrature_work):
            raise ResourceLimitError(f"the KK6 radius search needs at least {need} work units for {step}, exceeding "
                                     f"LCHS.max_quadrature_work={max_quadrature_work}")
        record["work_units"] = need

    while True:
        middle = lower + (upper - lower) / 2
        if middle == lower or middle == upper:
            break
        charge("its next residual visit")
        record["visits"] += 1
        if not 0.0 < middle < 1.0:
            return refused(f"invalid interior point mu={middle!r}")
        log_ratio = log1p(-middle) - log_epsilon
        if not isfinite(log_ratio):
            return refused(f"nonfinite L(mu) at mu={middle!r}")
        if log_ratio <= 0.0:
            # A rounded upper boundary: shrink the domain without calling it an observed negative residual.
            upper = middle
            continue
        radius = tan(pi * middle / 2)
        if not (isfinite(radius) and radius > 0.0):
            return refused(f"nonfinite or nonpositive R at mu={middle!r}")
        if form == "separate_parts":
            residual = log_ratio - pi * middle * middle * (1.0 + radius * radius) * a / (4.0 * (a * radius + b))
        else:
            residual = log_ratio - pi / 4 * middle * middle * radius
        if not isfinite(residual):
            return refused(f"nonfinite residual at mu={middle!r}")
        if residual == 0.0:
            hit = (middle, residual)
            record["numerical_zero"] = True
            break
        if residual > 0.0:
            lower, record["positive"] = middle, (middle, residual)
        else:
            upper, record["negative"] = middle, (middle, residual)
    if hit is None:
        if record["positive"] is None or record["negative"] is None:
            return refused("no representable interior sign bracket was observed")
        hit = min((record["positive"], record["negative"]), key=lambda item: (abs(item[1]), item[0]))
    record["residual"] = hit
    charge("the final conversion and fixed-R check")
    radius = tan(pi * hit[0] / 2)
    if not (isfinite(radius) and radius > 0.0):
        return refused(f"nonfinite or nonpositive R at the returned mu={hit[0]!r}")
    try:
        _apel_radius(epsilon_c, radius)
    except ValueError as error:
        return refused(f"the fixed-R check refuses R={radius!r}: {error}")
    return {**record, "radius": radius, "reason": "p = 2 stationary radius of the leading-cost surrogate"}


APEL_BERNTSON_PROFILE_ID = "apel_berntson_kk1_kk4_v1"

# Pairs whose parameters are derived together from the problem scalars and
# allowances, each with the profile identifier recorded in
# kernel_parameter_selection.
LCHS_PAIR_PROFILE_RESOLVERS: Mapping[
    tuple[str, str],
    tuple[str, Callable[[LCHSProblemContext, ResolvedProviderConfig, Mapping[str, float]],
                        Mapping[str, ProviderParameter]]],
] = MappingProxyType(
    {
        ("low_somma_f2", "symmetric_uniform_trapezoid"): (LOW_SOMMA_PROFILE_ID, _resolve_low_somma_pair_profile),
        ("apel_berntson", "tanh_sinh"): (APEL_BERNTSON_PROFILE_ID, _resolve_apel_pair_profile),
    }
)


def resolve_provider_config(
    request: ProviderConfig,
    registry: Mapping[str, LCHSKernelProvider | LCHSQuadratureProvider],
    *,
    slot: str,
) -> tuple[LCHSKernelProvider | LCHSQuadratureProvider, ResolvedProviderConfig]:
    """Resolve one kernel or quadrature request using its defaults and domain checks.

    Returns:
        The registered rule and a ResolvedProviderConfig holding every
        parameter after the rule's defaults and domain checks.
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
    *,
    coefficient_artifact: LCHSDiscreteWeightArtifact | None = None,
) -> tuple[
    LCHSKernelProvider,
    ResolvedProviderConfig,
    LCHSQuadratureProvider,
    ResolvedProviderConfig,
    LCHSPairCompatibility,
]:
    """Resolve both requests and check that the pair is allowed.

    The LCHS Method calls this at construction. Each rule resolves its
    supported parameters and checks their domains before any problem data
    is read. An unknown rule name or a rejected pair raises ValueError.
    The `"discrete_weights"` + `"artifact_nodes"` pair needs the
    LCHSDiscreteWeightArtifact whose content_id equals its `artifact_id`,
    and the artifact is accepted by no other pair.

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
    if key == ("discrete_weights", "artifact_nodes"):
        if not isinstance(coefficient_artifact, LCHSDiscreteWeightArtifact):
            raise ValueError("the discrete_weights kernel needs its prepared LCHSDiscreteWeightArtifact "
                             "(prepare_lchs_discrete_weights), and this caller supplies none. "
                             "The LCHS Method has no input for this pair")
        if coefficient_artifact.content_id != resolved_kernel.parameters["artifact_id"]:
            raise ValueError(f"the supplied artifact {coefficient_artifact.content_id} is not the "
                             f"discrete_weights artifact_id {resolved_kernel.parameters['artifact_id']}")
    elif coefficient_artifact is not None:
        raise ValueError("a coefficient artifact is used only by the discrete_weights + artifact_nodes pair")
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

    The kernel and quadrature bounds describe the ideal rule in exact
    arithmetic, and the LCHS error model reports no floating-point value, so
    it cannot report this rounding next to them. Planning therefore refuses
    the table when u*alpha exceeds LCHS_ROUNDING_EPSILON_FRACTION*epsilon.
    Both sides are operator-level quantities for a unit input, and the input
    norm, Duhamel weight and PSD recovery scale them alike. Equality is
    accepted. This is a scale check of the sum's evaluation. It does not
    bound the binary64 construction of the nodes, weights and coefficients,
    which ``native_floating_point`` covers with the other floating-point
    errors.
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


def coefficient_lift(coefficients) -> int:
    """Return k, the power of two 2**k that lifts every nonzero Cartesian part of a coefficient table into the normal range.

    With q the smallest ``frexp`` exponent among the nonzero real and
    imaginary parts, k = max(0, -1021 - q). A positive binary64 number is at
    least 2**-1074, whose exponent is -1073, so 0 <= k <= 52, and every
    nonzero part of 2**k*c_j is at least 2**-1022. Scaling by 2**k is exact.
    The admission ``_admit_coefficient_rounding`` requires u*beta <=
    .1*epsilon with u = 2**-53 and epsilon < 1, so every admitted provider
    table has raw one-norm beta < .1/u, about 9.01e14 or 2**49.7. Its
    lifted parts therefore stay below 2**102, a further factor g < 2 (the
    homogeneous compensation) keeps them below 2**103, and a finite
    application count or table length adds far less than the remaining
    range, so no lifted sum overflows. The argument covers the whole table,
    not each term alone, and it holds only for admitted provider tables or
    tables derived from them by such bounded factors. A table without a
    nonzero part has k = 0.

    The lift serves magnitudes and rotations. abs((1+1j)*2**-1074) rounds
    to 2**-1074 instead of sqrt(2)*2**-1074, and multiplying a subnormal
    part by a non-power-of-two factor or a unit phase rounds it by up to
    order one relative to its value. After the lift, hypot and such
    products round with relative error of order u. A normal table has
    k = 0 and is unchanged. Normalizing by the largest part instead can
    push a small retained part below the normal range: for
    (1e14 + 2**-1074 i, (1+i)*2**-1074), dividing by 2**47 loses the
    second coefficient, while the lift by 2**52 keeps its amplitude
    2**.25*2**-537/1e7.
    """
    parts = np.abs(np.asarray(coefficients, dtype=np.complex128).view(np.float64))
    smallest = float(np.min(parts, where=parts > 0, initial=inf))
    return 0 if smallest == inf else max(0, -1021 - frexp(smallest)[1])


def lifted_magnitudes(coefficients):
    """Return the magnitudes 2**k*|c_j| of a coefficient table and its lift k (``coefficient_lift``).

    The raw magnitude of c_j is magnitudes[j]*2**-k, which a caller keeps as
    a PhysicalScale factor, never as a binary64 product that could
    underflow. Normalized quantities such as sqrt(K_j/sum_i K_i) do not
    depend on k. The table is not modified. For k = 0 the magnitudes are
    NumPy's complex magnitudes of the c_j themselves, which can differ from
    Python's ``abs`` in the last binary place. Bytes per coefficient: the
    16-byte complex copy with the 18-byte lift scan at the peak, then the
    copy with the 8-byte result, 8 retained.
    """
    values = np.array(coefficients, dtype=np.complex128)
    lift = coefficient_lift(values)
    if lift:
        values *= 2.0**lift
    return np.abs(values), lift


def _resolve_coefficient_context(
    *,
    lchs_kernel: ProviderConfig,
    k_quadrature: ProviderConfig,
    problem_context: LCHSProblemContext,
    max_bytes=None,
    max_quadrature_work=None,
    coefficient_artifact: LCHSDiscreteWeightArtifact | None = None,
    free_half_width: int | None = None,
    strip_margin: float | None = None,
    work_ledger: dict | None = None,
    dense_select_slot_limit: int | None = None,
) -> LCHSCoefficientPlan:
    """Build one immutable coefficient plan, in the same form for every kernel-quadrature pair.

    Coefficients are c_j = w_j*g(k_j) in node order, with g(k_j) taken from
    the rule's kernel_values when it supplies them. A non-finite node, weight,
    coefficient or running L1 norm raises at the first offending index, so an
    overflowed kernel never reaches PREP normalization or SELECT phases.

    `coefficient_artifact` is the LCHSDiscreteWeightArtifact of the
    `"discrete_weights"` pair and `None` for every other pair. Its weights
    are used after the support and allowance checks of the artifact rule.
    Nothing here solves an optimization or reevaluates its residual.

    `free_half_width` is the free uniform grid of PLAN M3 for the Low–Somma
    pair, which provider enumeration (`selection.provider_frontier`) uses:
    J' at least the profile's J and h = R/J'. Then h <= R/J <= h_max, so the
    Theorem 3 condition and the recorded component caps still hold, with the
    same R and gamma. The derived J and h are replaced by J' and R/J'.

    `strip_margin` is the strip margin d of the composite-Gauss panel bound
    (`_ellipse_rule`) for the default Eq. (7) pair, which provider
    enumeration sets to 0.5. It is recorded as the derived parameter
    `strip_margin`; None keeps LCHS_ELLIPSE_STRIP_MARGIN and records nothing,
    so the default plan is unchanged.

    A supplied `work_ledger` mapping receives the admitted quadrature work
    of the rule under `"quadrature_work"` (`_check_quadrature_work`).

    `dense_select_slot_limit` is `LCHS.max_dense_select_slots` on a quantum
    dense_exact route (`selection._route`) and None elsewhere. The composite
    Gauss rule reads it (`_build_composite_gauss`).
    """

    (
        kernel,
        resolved_kernel,
        quadrature,
        resolved_quadrature,
        compatibility,
    ) = resolve_lchs_provider_requests(lchs_kernel, k_quadrature, coefficient_artifact=coefficient_artifact)
    pair_key = (resolved_kernel.implementation, resolved_quadrature.implementation)
    profile_id, profile_resolver = LCHS_PAIR_PROFILE_RESOLVERS.get(pair_key, (None, None))
    pair_label = (
        f"{resolved_kernel.implementation!r} + "
        f"{resolved_quadrature.implementation!r}"
    )
    allowances = _component_allowances(problem_context, pair_key, pair_label)
    try:
        derived_parameters = (
            {}
            if profile_resolver is None
            else dict(profile_resolver(problem_context, resolved_kernel, allowances))
        )
        derived_parameters.update({f"allowance_{name}": value for name, value in allowances.items()})
        if free_half_width is not None:
            if pair_key != ("low_somma_f2", "symmetric_uniform_trapezoid") or free_half_width < derived_parameters["J"]:
                raise ValueError("a free uniform grid needs the Low–Somma pair and a half width of at least its "
                                 f"profile's J={derived_parameters.get('J')}")
            derived_parameters.update(J=free_half_width, h=derived_parameters["R"] / free_half_width)
        if strip_margin is not None:
            if pair_key != ("near_optimal_eq7", "composite_gauss") or not 0 < strip_margin < 1:
                raise ValueError("a Gauss panel strip margin needs the near_optimal_eq7 + composite_gauss pair and "
                                 f"0 < d < 1, not d={strip_margin!r}")
            derived_parameters["strip_margin"] = float(strip_margin)
        derived_parameters = _immutable_map(derived_parameters)
        pair_context = MappingProxyType(
            {
                "kernel_parameters": resolved_kernel.parameters,
                "quadrature_parameters": resolved_quadrature.parameters,
                "derived_parameters": derived_parameters,
                "component_allowances": allowances,
                "coefficient_artifact": coefficient_artifact,
                "work_ledger": work_ledger,
                "dense_select_slot_limit": dense_select_slot_limit,
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
    # The arrays are evaluated together, and the error at the first offending
    # index is reported in the order of the checks at that index: node,
    # weight, kernel evaluation, coefficient, running L1 norm. The kernel is
    # evaluated only before the first non-finite node or weight, and a scalar
    # kernel only up to its first failure or non-finite value.
    nodes, weights = quadrature_plan.nodes, quadrature_plan.base_weights
    finite = np.isfinite(nodes) & np.isfinite(weights)
    stop, failure = (len(nodes) if finite.all() else int(np.argmin(finite))), None
    if quadrature_plan.kernel_values is not None:
        values = quadrature_plan.kernel_values[:stop]
    elif kernel.coefficients is not None:
        try:
            with np.errstate(all="ignore"):
                values = np.asarray(kernel.coefficients(nodes[:stop], pair_context), dtype=np.complex128)
        except ArithmeticError as exc:
            raise ValueError(f"invalid inputs for LCHS provider pair {pair_label}: {exc}") from exc
    else:
        values = np.empty(stop, dtype=np.complex128)
        for index in range(stop):
            try:
                values[index] = kernel.coefficient(float(nodes[index]), pair_context)
            except ArithmeticError as exc:
                values, failure = values[:index], exc
                break
            if not np.isfinite(values[index]):
                values = values[:index + 1]
                break
    evaluated = len(values)
    with np.errstate(all="ignore"):
        coefficients = weights[:evaluated] * values
        running = np.cumsum(np.abs(coefficients))
    if not np.isfinite(running).all():
        # A non-finite coefficient makes the running norm non-finite at its index.
        index = int(np.argmin(np.isfinite(running)))
        node = float(nodes[index])
        if not np.isfinite(coefficients[index]):
            raise ValueError(
                f"LCHS provider pair {pair_label} produced non-finite coefficient "
                f"at node {node!r} (index {index}): {complex(coefficients[index])!r}"
            )
        raise ValueError(
            f"LCHS provider pair {pair_label} produced non-finite coefficient "
            f"L1 norm at node {node!r} (index {index}): {float(running[index])!r}"
        )
    if failure is not None:
        raise ValueError(
            f"invalid inputs for LCHS provider pair {pair_label} at node "
            f"{float(nodes[evaluated])!r} (index {evaluated}): {failure}"
        ) from failure
    if stop < len(nodes):
        node, weight = float(nodes[stop]), float(weights[stop])
        if not np.isfinite(node):
            raise ValueError(
                f"LCHS provider pair {pair_label} produced non-finite node "
                f"at index {stop}: {node!r}"
            )
        raise ValueError(
            f"LCHS provider pair {pair_label} produced non-finite weight "
            f"at node {node!r} (index {stop}): {weight!r}"
        )
    coefficient_l1_norm = float(running[-1]) if len(running) else 0.0
    _admit_coefficient_rounding(coefficient_l1_norm, float(problem_context.epsilon), pair_label)
    # Keep user/default parameter provenance alongside the immutable grid so
    # reporting and later realization use the same coefficient selection.
    selection = {
        name: ("user" if name in lchs_kernel.parameters else "provider_default")
        for name in resolved_kernel.parameters
    }
    if profile_id is not None:
        selection["profile"] = profile_id
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
        coefficients=coefficients,
        coefficient_l1_norm=coefficient_l1_norm,
        quadrature=quadrature_plan,
        compatibility=compatibility,
    )


@dataclass(frozen=True, kw_only=True)
class LCHSContinuousKernelPlan:
    """A compact LCHS kernel on (-R, R) with the guarded truncated Cauchy proposal for sampled nodes.

    [`resolve_lchs_continuous_kernel`][nwqlib.algorithms.lchs.providers.resolve_lchs_continuous_kernel]
    returns it. The truncated Cauchy density q_R(k) = 1/(2*atan(R)*(1+k**2))
    on |k| < R has the inverse distribution function
    k = tan((2v-1)*atan(R)) for uniform v. The endpoint guard keeps v in
    [guard, 1-guard]: for U uniform on [0, 1], `node(U)` has the guarded
    density q_g(k) = q_R(k)/Z on |k| <= R_g, with Z = 1 - 2*guard and
    R_g = tan(Z*atan(R)). `weight` returns g(k)/q_g(k) = Z*g(k)/q_R(k), so
    the mean of weight(k)*f(k) over these draws is the integral of g*f
    over [-R_g, R_g], with no further normalization. That integral differs
    from the one over (-R, R) by at most `guard_omission_bound` for
    |f| <= 1. A sampled LCHS estimate uses these draws in place of a k
    quadrature. The fields are read-only. Finite random bits and the
    rounding of k, of the weight and of its phase are not bounded here.

    Attributes:
        requested_lchs_kernel: Kernel choice as given.
        resolved_lchs_kernel: Kernel used, with its defaults filled in.
        lchs_kernel_formula_id: Versioned identifier of the kernel formula.
        derived_parameters: For Apel-Berntson `mu`, `R`, `Delta` and
            `beta_R`; for Low-Somma `gamma`, `R` and `epsilon_lchs`.
        radius: Support radius R of the kernel and the proposal.
        proposal_normalization: 2*atan(R), the normalization of q_R.
        importance_weight_bound: Z*W, the upper bound on |weight(k)| for
            the guarded proposal, with W = `unguarded_weight_bound`.
        unguarded_weight_bound: Upper bound W on |g(k)/q_R(k)| over
            |k| < R. Apel-Berntson has |g/q_R| = beta_R =
            (2*atan(R)/pi)*exp((1-mu)*Delta) at every k, because |g_R| is
            proportional to 1/(1+k**2). Low-Somma has
            |g/q_R| = (2*atan(R)/pi)*exp(c - (1+k**2)/(4*gamma**2)), largest at k = 0.
        importance_weight_constant: Whether |weight(k)| equals
            `importance_weight_bound` at every k.
        kernel_error_bound: Operator-norm error of the continuous integral
            over (-R, R) for a unit input: the Apel-Berntson continuous
            allowance epsilon_c, or epsilon_lchs of Low-Somma Theorem 2.
        guard: Endpoint guard in (0, 1/2). The two omitted v-intervals
            carry proposal mass 2*guard.
        guard_omission_bound: 2*guard*W, the operator-norm bound on the
            kernel mass that the guard omits, at most the guard allowance.
    """

    requested_lchs_kernel: ProviderConfig
    resolved_lchs_kernel: ResolvedProviderConfig
    lchs_kernel_formula_id: str
    derived_parameters: Mapping[str, ProviderParameter]
    radius: float
    proposal_normalization: float
    importance_weight_bound: float
    unguarded_weight_bound: float
    importance_weight_constant: bool
    kernel_error_bound: float
    guard: float
    guard_omission_bound: float

    def node(self, uniform: float) -> float:
        """Return the node k = tan((2v-1)*atan(R)) for v = guard + (1-2*guard)*uniform.

        Every `uniform` in the closed interval [0, 1] maps to a binary64
        node with |k| < R, where the weight is finite. The resolver
        establishes this from the two ends of the interval, assuming a
        faithful tan.
        """
        guarded = self.guard + (1.0 - 2.0 * self.guard) * uniform
        return tan((2.0 * guarded - 1.0) * atan(self.radius))

    def weight(self, k_value: float) -> complex:
        """Return the importance weight (1-2*guard)*g(k)/q_R(k) of the guarded proposal at a node with |k| < R."""
        derived = self.derived_parameters
        retained = 1.0 - 2.0 * self.guard
        if self.resolved_lchs_kernel.implementation == "apel_berntson":
            return apel_berntson_kernel(k_value, radius=self.radius, mu=float(derived["mu"]),
                                        delta=float(derived["Delta"])) * (retained * self.proposal_normalization
                                                                          * (1.0 + k_value * k_value))
        c_value = float(self.resolved_lchs_kernel.parameters["c"])
        gamma = float(derived["gamma"])
        magnitude = (retained * self.proposal_normalization / pi) * exp(
            c_value - (1.0 + k_value * k_value) / (4.0 * gamma**2))
        return complex(magnitude * cos(k_value * c_value), -magnitude * sin(k_value * c_value))


def resolve_lchs_continuous_kernel(*, lchs_kernel: ProviderConfig, kernel_allowance: float,
                                   guard_allowance: float) -> LCHSContinuousKernelPlan:
    """Return a compact kernel and its guarded truncated Cauchy proposal for sampled LCHS nodes.

    The Apel-Berntson kernel uses the radius rule of `apel_berntson` with
    `kernel_allowance` as its continuous allowance epsilon_c (Apel and
    Berntson, arXiv:2610.03669v1, Theorems 4 and 6 and Proposition 10). The
    Low-Somma kernel uses
    epsilon_lchs = `kernel_allowance` in Theorem 2 of arXiv:2508.19238v2,
    gamma = sqrt(c + log((1+1/(2*pi))/epsilon_lchs))/c and R = 2*c*gamma**2.
    With W the bound on |g/q_R|, the guard is guard_allowance/(2*W), so the
    omitted kernel mass 2*guard*W meets the guard allowance. Neither choice
    depends on the problem, because the continuous error bound holds for
    every PSD L_s and Hermitian H. The Eq. (7) and Cauchy kernels have no
    admitted finite-interval weight bound on this proposal and are refused.

    The guard must also keep every binary64 node strictly inside (-R, R).
    The map from the uniform variable to the argument of tan uses only
    correctly rounded, hence monotone, operations. If tan is faithful
    (error below one unit in the last place), then every node lies within
    one unit in the last place of the range spanned by node(0) and
    node(1). The resolver therefore requires both extreme nodes to lie at
    least one unit in the last place inside (-R, R), which keeps the
    weight finite for every uniform in [0, 1]. A tiny guard fails this
    check and is refused, not widened, since a wider guard would omit more
    kernel mass than its allowance.

    Args:
        lchs_kernel (ProviderConfig): `"apel_berntson"` or `"low_somma_f2"`
            with their parameters.
        kernel_allowance (float): Operator-norm allowance of the continuous
            kernel integral, in (0, 1) for Apel-Berntson and in (0, 4/15]
            for Low-Somma.
        guard_allowance (float): Positive operator-norm allowance of the
            endpoint guard, small enough that the guard is below 1/2 and
            large enough that its extreme nodes stay inside (-R, R).

    Returns:
        kernel (LCHSContinuousKernelPlan): The kernel, proposal, weight
            bounds and both error bounds.

    Raises:
        ValueError: For another kernel, an allowance outside its range, or
            a guard whose extreme binary64 nodes do not stay inside
            (-R, R), the numerical range limit of the guard.
    """
    kernel, resolved = resolve_provider_config(lchs_kernel, LCHS_KERNEL_IMPLEMENTATIONS, slot="lchs_kernel")
    kernel_allowance = finite_real(kernel_allowance, "kernel_allowance")
    guard_allowance = finite_real(guard_allowance, "guard_allowance")
    if resolved.implementation == "apel_berntson":
        mu, radius, delta, _ = _apel_radius(kernel_allowance, resolved.parameters["radius"])
        # beta_R as (2*atan(R)/pi)*exp((1-mu)*Delta), the product that `weight`
        # evaluates, so that |weight| and this bound agree to rounding.
        bound = _exp_with_nonfinite_overflow(log(2.0 * atan(radius) / pi) + (1.0 - mu) * delta)
        derived = {"mu": mu, "R": radius, "Delta": delta, "beta_R": bound}
    elif resolved.implementation == "low_somma_f2":
        if not 0.0 < kernel_allowance <= 4.0 / 15.0:
            raise ValueError("the continuous Low-Somma kernel requires kernel_allowance in (0, 4/15]")
        c_value = float(resolved.parameters["c"])
        gamma = sqrt(c_value + log((1.0 + 1.0 / (2.0 * pi)) / kernel_allowance)) / c_value
        radius = 2.0 * c_value * gamma**2
        bound = _exp_with_nonfinite_overflow(log(2.0 * atan(radius) / pi) + c_value - 1.0 / (4.0 * gamma**2))
        derived = {"gamma": gamma, "R": radius, "epsilon_lchs": kernel_allowance}
    else:
        raise ValueError(f"lchs_kernel {resolved.implementation!r} has no admitted weight bound on the "
                         "truncated Cauchy proposal; use 'apel_berntson' or 'low_somma_f2'")
    guard = guard_allowance / (2.0 * bound) if isfinite(bound) else 0.0
    if not 0.0 < guard < 0.5:
        raise ValueError(f"guard_allowance {guard_allowance!r} gives endpoint guard {guard!r} for weight bound "
                         f"{bound!r}; the guard must lie in (0, 1/2)")
    omission = 2.0 * guard * bound
    if omission > guard_allowance:
        guard = nextafter(guard, 0.0)
        omission = 2.0 * guard * bound
    plan = LCHSContinuousKernelPlan(
        requested_lchs_kernel=lchs_kernel, resolved_lchs_kernel=resolved, lchs_kernel_formula_id=kernel.formula_id,
        derived_parameters=_immutable_map(derived), radius=radius, proposal_normalization=2.0 * atan(radius),
        importance_weight_bound=(1.0 - 2.0 * guard) * bound, unguarded_weight_bound=bound,
        importance_weight_constant=resolved.implementation == "apel_berntson",
        kernel_error_bound=kernel_allowance, guard=guard, guard_omission_bound=omission)
    lowest, highest = plan.node(0.0), plan.node(1.0)
    if not (nextafter(lowest, -float("inf")) > -radius and nextafter(highest, float("inf")) < radius):
        raise ValueError(
            f"guard_allowance {guard_allowance!r} gives endpoint guard {guard!r}, whose extreme nodes {lowest!r} "
            f"and {highest!r} do not stay one unit in the last place inside (-R, R) with R = {radius!r}. "
            "This is the numerical range limit of the binary64 guard; use a larger guard_allowance")
    return plan


def _unitary_coefficient_plan(options):
    """L=0 eliminates the k integral: one H-evolution branch of weight one.

    This is an algebraic reduction of A=L+iH, not a one-node approximation to
    the requested kernel. The requested kernel and quadrature settings are
    kept only as a record of the request.
    """
    resolved = ResolvedProviderConfig(implementation='unitary_reduction', parameters={})
    # One node k = 0 with weight 1. The single branch is exp(-i*T*H) itself,
    # and both component bounds are exact zeros.
    quadrature = LCHSQuadraturePlan(
        nodes=(0.,),
        base_weights=(1.,),
        range_k=0.,
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


def _homogeneous_coefficient_plan(quadrature, elapsed, scalar_phase=(0.0, 0.0), *, max_bytes):
    """Return the coefficient plan of the physical propagator exp(-A*elapsed).

    Coefficient j becomes d_j = g*2**k*c_j*exp(i*phi_j), with the stored
    raw coefficient c_j, and every routine that uses the coefficients
    composes 2**E, the plan's recovery_exponent, into its binary recovery,
    so that 2**E*d_j = exp(s*elapsed)*c_j*exp(i*phi_j) for the signed
    shift s of solution_error_budget.signed_shift, the recovery of the PSD
    shift or known decay. coefficient_l1_norm becomes sum_j |d_j|. The
    factor g*2**k is common to all coefficients, so it changes the physical
    recovery and leaves the normalized coefficient state unchanged.

    Scale. exp(s*elapsed) is the PhysicalScale mu*2**e' of
    solution_error_budget.admitted_exp, g = 2*mu lies in [1, 2), and k is
    the exact lift of ``coefficient_lift``, at most 52, which brings every
    nonzero Cartesian part of the raw table into the normal range. Then
    E = e' - 1 - k, an exact integer however large s*elapsed is, and
    derived_parameters["homogeneous_coefficient_compensation"] records the
    applied factor g*2**k in [1, 2**53). The exact product of each nonzero
    raw part with g*2**k lies in the normal range, so its binary64 value
    has relative error at most u = 2**-53, and no coefficient shrinks or
    underflows however strong the known decay. A table without subnormal
    parts has k = 0, and then E = 0 exactly when exp(s*elapsed) lies in
    [1, 2). Under provider admission the one-norm sum_j |d_j| stays below
    2*2**52*.1/u, about 8.1e30, so no complete sum has to leave binary64
    (``coefficient_lift``). A nonfinite s*elapsed raises
    ApplicabilityError.

    ``scalar_phase = (a, b)`` is the scalar a recipe takes out of its
    evolved children: with children L_c = L - c_L I and
    H_c = H - c_H I, a = c_L + s and b = c_H, and
    exp(-i t (k L_s + H)) = exp(-i t (k a + b)) exp(-i t (k L_c + H_c)).
    phi_j = -elapsed*(k_j*a + b) enters as the unit phase of
    solution_error_budget.extracted_phase, which refuses a nonfinite
    argument. The magnitude it multiplies is at least 2**-1022, so the
    rotation rounds by order u relative to |d_j| and keeps the phase of
    c_j, although a rotated Cartesian part can still cancel to a tiny or
    zero value with an absolute effect of order u*|d_j|. The default
    (0, 0) leaves the coefficients unrotated.

    derived_parameters records the shift, the elapsed time, the factor
    applied to the coefficients and any scalar phase. The nodes, base
    weights and raw quadrature coefficients are unchanged.

    Bytes, checked against max_bytes before the lift scan and the new table,
    as a coarse estimate of the dominant arrays. For M nodes this stage
    holds the raw node, weight and coefficient arrays of the plan and of the
    quadrature data (56 bytes per node), and forms the new complex128
    coefficients, their scalar phases and their read-only copy (64 bytes per
    node at most), so 128*M, plus 16*K for a rule that keeps its kernel
    values, K = M, else K = 0. The lift scan (18 bytes per node,
    ``coefficient_lift``) ends before the new array is formed.
    """
    from dataclasses import replace
    from .solution_error_budget import admitted_exp, extracted_phase, signed_shift
    m = len(quadrature.coefficients)
    kernel_count = 0 if quadrature.coefficient_plan.quadrature.kernel_values is None else m
    _check_bytes(128*m + 16*kernel_count, max_bytes, "LCHS homogeneous coefficient table")
    shift = signed_shift(quadrature.conversion)
    # ACL arXiv:2312.03916v2, Eq. (4)/(60), specialized to constant A:
    # replacing A by A+sI shifts L by s*I, and
    # exp(-A*T) = exp(s*T)*exp(-(A+s*I)*T). The LCHS integral is applied to
    # the shifted A, whose L is PSD, and exp(s*T) is applied once to the
    # homogeneous coefficients. Duhamel source branches use their own elapsed
    # time T-t_j instead.
    scale = admitted_exp(shift*elapsed)
    lift = coefficient_lift(quadrature.coefficients)
    compensation, exponent = ldexp(2*scale.mantissa, lift), scale.exponent - 1 - lift
    dissipative, hamiltonian = scalar_phase
    coefficients = np.asarray(quadrature.coefficients, dtype=np.complex128) * compensation
    if dissipative or hamiltonian:
        coefficients *= np.fromiter((extracted_phase(elapsed, k, scalar_phase) for k in quadrature.k_nodes),
                                    dtype=np.complex128, count=m)
    alpha = fsum(np.abs(coefficients))
    if not isfinite(alpha) or alpha <= 0:
        raise ValueError("PSD-compensated LCHS coefficients require a finite positive 1-norm")
    raw = quadrature.coefficient_plan
    extracted = {} if not (dissipative or hamiltonian) else {
        "scalar_phase_dissipative": float(dissipative), "scalar_phase_hamiltonian": float(hamiltonian)}
    return replace(raw, coefficients=coefficients, coefficient_l1_norm=alpha, recovery_exponent=exponent,
        derived_parameters=_immutable_map({**raw.derived_parameters,
            "psd_shift": float(quadrature.conversion["psd_shift"]),
            "scalar_decay": float(quadrature.conversion["scalar_decay"]), "elapsed_time": elapsed,
            "homogeneous_coefficient_compensation": compensation, **extracted}))


def resolve_lchs_coefficient_plan(problem_or_plan, *, lchs_kernel=None, k_quadrature=None,
                                  approximation_tolerance=None, max_bytes=None):
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
        max_bytes (int | None): Default `None`, the physical memory of this computer.
            Upper limit on the arrays of the Cartesian parts and the PSD
            check, and on the node grid and coefficient table data, each
            tested before it is formed. A `PeriodicStencil` A
            forms no matrix and uses the bound `||L|| <= mass + 4*diffusion`.

    Returns:
        coefficients (LCHSCoefficientPlan): The nodes, weights, coefficients
            and component bounds. For a dense A the coefficients include the
            PSD growth factor `exp(shift*T)` divided by
            `2**recovery_exponent`, and
            `derived_parameters["homogeneous_coefficient_compensation"]`
            records the factor applied to the raw coefficients.

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
    # the physical dimension, the known-array allowance is
    # B_coefficient = 48 d**2 + 16 d. Forming L holds A, L and the adjoint,
    # or 48 d**2 bytes. The eigensolve allowance holds A, L, the solver's
    # internal copy and both eigenvalue vectors, or 48 d**2 + 16 d bytes.
    # The queried eigensolver workspace Q_N(d) is not charged (a declared
    # known-array workspace, not a complete cap). PSD compensation remains
    # part of the selected coefficients.
    d = problem.dimension
    _check_bytes(48*d*d + 16*d, max_bytes, "LCHS Cartesian/PSD coefficient selection")
    from .method import LCHS
    method = LCHS(lchs_kernel=kernel, k_quadrature=quadrature, approximation_tolerance=tolerance)
    prepared = _prepare_decomposition(matrix=matrix, method=method, padded_dimension=dimension,
                                      max_spectral_work=method.max_spectral_work, parts=False)
    selected = _quadrature_data_from_plan(l_norm=prepared.l_norm, conversion=prepared.conversion,
        psd_premise_satisfied=prepared.numerical_psd_premise_satisfied, final_time=problem.elapsed_time,
        method=method, max_bytes=max_bytes)
    return _homogeneous_coefficient_plan(selected, problem.elapsed_time, max_bytes=max_bytes)
