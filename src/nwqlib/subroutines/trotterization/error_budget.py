"""Deterministic Trotter error budgets with commutator scaling.

Implements the tight low-order product-formula error bounds of Childs, Su,
Tran, Wiebe, and Zhu, "Theory of Trotter error with commutator scaling",
Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020,
arXiv:1912.08854 [CSTWZ]. Unless marked as arXiv:1912.08854v3, every
proposition, equation, section and page number in this module
refers to the Phys. Rev. X version. The arXiv:1912.08854v3 preprint, titled
"A Theory of Trotter Error", numbers the same results Prop. 15, Eq. (145)
(p. 38) and Prop. 16, Eq. (152) (p. 39) in Sec. 5.1, with the step-count
rule in Sec. 5.2 (p. 40).

- First-order Lie-Trotter bound: Proposition 9, Eq. (120), Sec. V A,
  p. 011020-22.
- Second-order Suzuki bound: Proposition 10, Eq. (121), Sec. V A,
  p. 011020-22.
- Step-count selection: the Sec. V B rule (p. 011020-23). Divide the
  evolution into ``r`` Trotter steps, apply the per-step bound within each
  step, and take the smallest ``r`` for which the total
  ``r * bound(t / r)`` is at most the requested budget. The paper finds this
  ``r`` by binary search. Both bounds here are closed-form monomials in
  ``1 / r``, so the smallest ``r`` is exact ceil arithmetic.

Ordering convention. Both propositions are stated for t >= 0 with the
``gamma = 1`` factor rightmost, so ``H_1`` acts first in ``S_1(t)``, and
``S_2(t)`` applies every term for ``t/2`` in order and then in reverse order.
The circuits built here apply terms in that order. The first-order
Pauli-triangle coefficient does not depend on the term order, but the
second-order one does.

Production commutator sums use Pauli commutation relations and coefficient
magnitudes.  Norm subadditivity plus ``||[P, Q]|| = 2`` and
``||[P, [Q, R]]|| = 4`` for surviving Pauli commutators makes these
triangle expansions certified upper bounds on the tail-sum spectral norms
in Props. 9-10. At order 2 a caller can select the full triangle
expression or its suffix relaxation, which needs pair tests only
(``_pauli_bound_coefficient_from_terms``).

Outward evaluation. The production coefficient is evaluated with scaled
binary64 upper products and sums, followed by rational rescaling. Its value
W_up bounds the selected triangle expression. For formula order o in {1,2},
time multiplication, division by r**o and integer step inversion use exact
rationals formed from W_up and the stored binary64 time. Published bounds
are rounded upward (``_upward_float``). This preserves a positive bound
below the binary64 range and reports a range error above it. The selected
count is minimal for W_up, which can exceed the exact triangle coefficient.
The record names the full or relaxed expression and its coefficient
arithmetic. Pruning consumes its allowance before step selection, and a
zero remaining allowance admits only a zero coefficient. The explicitly
named ``dense_trotter_bound_coefficient`` helper evaluates the tail-sum
norms from dense ``2**n`` by ``2**n`` matrices, for small-instance
validation only.

Census admission. ``census_work`` and ``census_bytes`` with
``choose_census_block`` price one shared structure before it is built, so a
caller can select the full or relaxed expression, or refuse, before the
unadmitted stage. Census admission includes packed masks, simultaneous pair
and triple storage, chunk-object overhead, row tests and bounded
coefficient-reduction workspace. The block size is selected from the
remaining byte allowance, including the partial sums that accumulate
between blocks. The public helpers admit their census in two stages against
``max_work`` and ``max_bytes`` (``_bound_coefficient_evaluation``).

Common steps. ``common_steps``, ``emitted_subtotal`` and
``select_common_step`` select one shared second-order step for several
nonnegative integer powers of a time unit. A common selection binds one
ordered step to every requested power by its integer grid and cumulative
counts. Its exact targets are derived from the stored time unit, and its
acceptance records the completed emitted-parameter recheck at every prefix.

Circuit construction delegates to the ``hamiltonian_evolution`` helpers,
``build_sparse_pauli_product_circuit`` over ``build_pauli_evolution_circuit``,
which apply the summands in the term order that the bounds assume
(``build_trotter_evolution_circuit``).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import ceil, frexp, gcd, inf, isfinite, isqrt, nextafter
from sys import float_info
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

import numpy as np

from nwqlib._limits import DEFAULT_MAX_BYTES


# These strings leave the module in TrotterStepSelection.to_dict output and in
# the recorded LCHS budget terms, so each one names the paper version whose
# numbering it uses.
TROTTER_BOUND_REFERENCE = (
    "Childs, Su, Tran, Wiebe, and Zhu, Theory of Trotter error with "
    "commutator scaling, Phys. Rev. X 11, 011020 (2021), "
    "doi:10.1103/PhysRevX.11.011020, arXiv:1912.08854v3. "
    "Proposition, equation and section numbers follow Phys. Rev. X"
)

BOUND_VARIANTS = ("exact_census", "relaxed_prefix")
COEFFICIENT_ARITHMETIC = "outward_float64_scaled"
# Default census work allowance of the public helpers, matching the QPE
# planning-work default. An engineering default, not a mathematical ceiling
# (docs/ENGINEERING_CONSTANTS.md).
DEFAULT_CENSUS_MAX_WORK = 1_000_000_000

_PROP9 = "[CSTWZ doi:10.1103/PhysRevX.11.011020 Prop. 9, Eq. (120), Phys. Rev. X numbering]"
_PROP10 = "[CSTWZ doi:10.1103/PhysRevX.11.011020 Prop. 10, Eq. (121), Phys. Rev. X numbering]"
_SEC_VB = "[CSTWZ doi:10.1103/PhysRevX.11.011020 Sec. V B smallest-r rule, Phys. Rev. X numbering]"
_FIRST_ORDER_TAIL = "(t^2 / 2) * sum_gamma1 ||sum_{gamma2 > gamma1} [H_gamma2, H_gamma1]||"
_SECOND_ORDER_TAIL = (
    "(t^3 / 12) * sum_gamma1 ||[T_gamma1, [T_gamma1, H_gamma1]]|| + "
    "(t^3 / 24) * sum_gamma1 ||[H_gamma1, [H_gamma1, T_gamma1]]|| with "
    "T_gamma1 = sum_{gamma2 > gamma1} H_gamma2"
)
# Selected coefficient expressions. A = {(i,j): i < j, P_i P_j = -P_j P_i}.
_VARIANT_BOUND_FORMULAS = {
    (1, "exact_census"): (
        "W_up * t^2 / r with W_up >= W_1 = sum_{(i,j) in A} |c_i*c_j|, "
        "A = {(i,j): i < j, P_i*P_j = -P_j*P_i}, by the Pauli-triangle inequality "
        f"with ||[P, Q]|| = 2, an upper bound on {_FIRST_ORDER_TAIL} {_PROP9}"
    ),
    (2, "exact_census"): (
        "W_up * t^3 / r^2 with W_up >= W_2 = sum_{(i,j,k) in T} |c_i*c_j*c_k| / 3 "
        "+ sum_{(i,j) in A} c_i^2*|c_j| / 6, A = {(i,j): i < j, P_i*P_j = -P_j*P_i}, "
        "T = {(i,j,k): (i,j) in A, k > i, P_k anticommuting with P_i*P_j}, k = j included, "
        "by the Pauli-triangle inequality with ||[P, Q]|| = 2 and ||[P, [Q, R]]|| = 4, "
        f"an upper bound on {_SECOND_ORDER_TAIL} {_PROP10}"
    ),
    (2, "relaxed_prefix"): (
        "W_up * t^3 / r^2 with W_up >= W_2,rel = sum_{(i,j) in A} |c_i*c_j|*S_i / 3 "
        "+ sum_{(i,j) in A} c_i^2*|c_j| / 6, S_i = sum_{k > i} |c_k|, "
        "A = {(i,j): i < j, P_i*P_j = -P_j*P_i}; NWQLib's suffix inequality W_2 <= W_2,rel "
        "replaces the nested anticommutation indicator of the Pauli-triangle expansion "
        f"by one, so W_up is an upper bound on {_SECOND_ORDER_TAIL} {_PROP10}"
    ),
}
_VARIANT_STEP_FORMULAS = {
    1: (
        "r = max(1, ceil(t^2 * W_up / epsilon)), integer inversion of "
        "W_up * t^2 / r against the remaining error budget, with W_up the recorded "
        f"upper coefficient {_SEC_VB}"
    ),
    2: (
        "r = max(1, ceil(sqrt(ceil(t^3 * W_up / epsilon)))), integer inversion of "
        "W_up * t^3 / r^2 against the remaining error budget, with W_up the recorded "
        f"upper coefficient {_SEC_VB}"
    ),
}
_COMMON_BOUND_FORMULA = (
    "Unitary composition of the ordered second-order step gives "
    f"r*W_up*abs(val(h_hat))**3 {_PROP10}"
)
_COMMON_STEP_FORMULA = (
    "Select the ideal gcd grid with the exact rational common-grid rule, then admit "
    "the represented step against every prefix's completed structural allowance."
)

_SYNTHESIS_METHODS = {1: "lie_trotter", 2: "suzuki_trotter"}


@dataclass(frozen=True, kw_only=True)
class TrotterStepSelection:
    """Step-count selection record for one product-formula error budget.

    Fields mirror the [CSTWZ] (doi:10.1103/PhysRevX.11.011020) quantities:
    ``bound_value`` is the certified
    total additive Trotter error bound ``step_count * bound(evolution_time /
    step_count)`` at the selected step count (at most ``error_budget``), with
    the per-application bound and the smallest-r arithmetic recorded as the
    ``bound_formula`` and ``step_formula`` source strings.

    A common selection binds one ordered step to every requested power by
    its integer grid and cumulative counts. Its exact targets are derived
    from the stored time unit, and its acceptance records the completed
    emitted-parameter recheck at every prefix (``select_common_step``). The
    ``common_*`` fields are None ("not recorded") on an independent
    selection. The maximum target time ``p_max*val(common_tau)`` and each
    represented prefix time ``r(p)*val(common_step_time)`` are derived, not
    stored. Validation derives g from the positive powers, requires every
    power to be divisible by it and checks ``r(p) = (p/g)*m``. The all-zero
    schedule has no g, m or step and no common selection.

    Attributes:
        formula_order: Selected Lie order 1 or symmetric Suzuki order 2.
            On a common selection: order of the shared ordered
            product-formula step. Common QPE trajectories use the symmetric
            second-order Suzuki formula.
        evolution_time: Total evolution time t > 0 of the product formula.
            On a common selection: binary64 display of the largest
            common-grid target time, whose exact value is defined by the
            maximum power and stored tau.
        error_budget: Requested operator-error allowance for this evolution.
            On a common selection: effective product-formula allowance for
            the full common trajectory after the other structural
            contributions are reserved at every queried power, that is
            ``min_(p>0) (epsilon_p-P_p-T_p-I_p-A_p)*r_max/r_p`` rounded
            downward to binary64.
        bound_value: Upward-rounded total operator-error bound at
            step_count, computed from the supplied upper coefficient and
            evolution_time. On a common selection: upward product-formula
            bound ``W_up*r_max*abs(val(h_hat))**3`` for the actually emitted
            common step.
        step_count: Smallest positive integer admitted by the supplied upper
            coefficient and remaining error_budget. On a common selection:
            number r(p_max) of shared steps through the last requested
            power. This is a sufficient common-grid count and need not be
            the independently smallest count for that power.
        pauli_term_count: Kept nonidentity terms of this node or Hamiltonian.
        pair_commutation_checks: New logical pair tests incurred by this
            selection, zero when the coefficient or structure was reused.
        nested_commutation_checks: New logical nested tests incurred by this
            selection, zero for relaxed_prefix or reused structure.
        bound_variant: Selected Pauli-triangle expression, "exact_census" for
            the full pair/triple indicator structure or "relaxed_prefix" for
            the second-order suffix relaxation. Order 1 uses "exact_census".
        coefficient_arithmetic: Numerical evaluation used for the supplied
            coefficient, "outward_float64_scaled" for scaled binary64 upper
            products and sums with rational rescaling.
        common_tau: Stored finite positive binary64 time unit. The exact
            target time at integer power p is p times its represented value.
        common_g: Greatest common divisor of the positive requested integer
            powers. It identifies the exact target-time grid.
        common_m: Positive integer subdivisions of g times the represented
            time unit. The ideal common step is g*val(tau)/m.
        common_step_time: Stored binary64 parameter h_hat used by the
            emitted shared step. Its represented value can differ from the
            ideal rational step.
        common_prefix_steps: Integer cumulative counts r(p)=(p/g)*m at each
            requested power, with r(0)=0, as ``(power, count)`` pairs.
            Execution between adjacent points uses the difference of these
            counts.
        common_power_allowances: Per-power structural operator-error
            allowances used by the emitted-step recheck, as
            ``(power, allowance)`` pairs. Stored binary64 allowances are
            interpreted as exact represented values for comparison.
        common_recheck_outcome: Outcome of comparing every complete
            emitted-prefix subtotal with its corresponding allowance.
            Accepted common selections have all powers passed. An absent
            outcome means the recheck was not recorded.
        common_prefix_bounds: Upward-published values of the exact pruning,
            product-formula, time-displacement, identity-phase and
            leaf-angle subtotals for the actual emitted prefixes, as
            ``(power, bound)`` pairs.
    """

    formula_order: int
    evolution_time: float
    error_budget: float
    bound_value: float
    step_count: int
    pauli_term_count: int = 0
    pair_commutation_checks: int = 0
    nested_commutation_checks: int = 0
    bound_variant: str
    coefficient_arithmetic: str
    common_tau: float | None = None
    common_g: int | None = None
    common_m: int | None = None
    common_step_time: float | None = None
    common_prefix_steps: tuple[tuple[int, int], ...] | None = None
    common_power_allowances: tuple[tuple[int, float], ...] | None = None
    common_recheck_outcome: str | None = None
    common_prefix_bounds: tuple[tuple[int, float], ...] | None = None

    def __post_init__(self):
        if self.bound_variant not in BOUND_VARIANTS:
            raise ValueError(f"unknown Pauli-triangle bound variant {self.bound_variant!r}")
        if self.formula_order == 1 and self.bound_variant == "relaxed_prefix":
            raise ValueError("order 1 records the exact_census expression")
        if self.coefficient_arithmetic != COEFFICIENT_ARITHMETIC:
            raise ValueError(f"unknown coefficient arithmetic {self.coefficient_arithmetic!r}")
        common = (
            self.common_tau, self.common_g, self.common_m, self.common_step_time,
            self.common_prefix_steps, self.common_power_allowances,
            self.common_recheck_outcome, self.common_prefix_bounds,
        )
        if all(value is None for value in common):
            return
        if self.formula_order != 2:
            raise ValueError("a common selection uses the second-order step")
        if (self.common_tau is None or self.common_prefix_steps is None
                or self.common_power_allowances is None or self.common_prefix_bounds is None
                or self.common_recheck_outcome != "accepted"):
            raise ValueError("a common selection records tau, prefix counts, allowances, "
                             "bounds and an accepted recheck together")
        if not (isfinite(self.common_tau) and self.common_tau > 0):
            raise ValueError("common_tau must be finite and positive")
        steps = dict(self.common_prefix_steps)
        if (len(steps) != len(self.common_prefix_steps)
                or len(self.common_power_allowances) != len(steps)
                or len(self.common_prefix_bounds) != len(steps)
                or any(type(p) is not int or p < 0 for p in steps)
                or {p for p, _ in self.common_power_allowances} != set(steps)
                or {p for p, _ in self.common_prefix_bounds} != set(steps)):
            raise ValueError("common prefix counts, allowances and bounds need the same "
                             "distinct nonnegative integer powers")
        positive = [p for p in steps if p]
        if not positive:
            raise ValueError("the all-zero schedule has no common selection")
        g = reduce(gcd, positive)
        m = self.common_m
        if self.common_g != g or type(m) is not int or m < 1:
            raise ValueError("common_g must be the gcd of the positive powers and common_m positive")
        if any(steps[p] != p // g * m for p in steps):
            raise ValueError("common prefix counts must be r(p) = (p/g)*m with r(0) = 0")
        if self.step_count != steps[max(positive)]:
            raise ValueError("step_count must be the common count at the largest power")
        h = self.common_step_time
        if h is None or not isfinite(h) or h <= 0:
            raise ValueError("common_step_time must be a finite positive binary64 step")

    @property
    def bound_formula(self) -> str:
        """Selected coefficient expression, source theorem and inequality.

        A common selection names the emitted-step composition.
        """
        order = 1 if self.formula_order == 1 else 2
        if self.common_recheck_outcome is not None:
            return _COMMON_BOUND_FORMULA
        return _VARIANT_BOUND_FORMULAS[order, self.bound_variant]

    @property
    def step_formula(self) -> str:
        """Integer inversion of W_up*abs(t)**(order+1)/r**order against the remaining error budget."""
        order = 1 if self.formula_order == 1 else 2
        if self.common_recheck_outcome is not None:
            return _COMMON_STEP_FORMULA
        return _VARIANT_STEP_FORMULAS[order]

    @property
    def synthesis_method(self) -> str:
        """Delegated synthesis name, structurally tied to ``formula_order``."""

        return _SYNTHESIS_METHODS[self.formula_order]

    @property
    def bound_method(self) -> str:
        """Production certificate method."""

        return "pauli_triangle"

    @property
    def bound_value_status(self) -> str:
        """Production certificate interpretation."""

        return "structural_upper_bound"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like record of the selection."""

        def pairs(values):
            return None if values is None else [list(item) for item in values]

        return {
            "formula_order": self.formula_order,
            "evolution_time": self.evolution_time,
            "error_budget": self.error_budget,
            "bound_formula": self.bound_formula,
            "step_formula": self.step_formula,
            "bound_value": self.bound_value,
            "step_count": self.step_count,
            "synthesis_method": self.synthesis_method,
            "bound_method": self.bound_method,
            "bound_value_status": self.bound_value_status,
            "bound_variant": self.bound_variant,
            "coefficient_arithmetic": self.coefficient_arithmetic,
            "algebraic_work_counts": {
                "pauli_terms": self.pauli_term_count,
                "pair_commutation_checks": self.pair_commutation_checks,
                "nested_commutation_checks": self.nested_commutation_checks,
            },
            "common_step": None if self.common_recheck_outcome is None else {
                "tau": self.common_tau,
                "g": self.common_g,
                "m": self.common_m,
                "step_time": self.common_step_time,
                "prefix_steps": pairs(self.common_prefix_steps),
                "power_allowances": pairs(self.common_power_allowances),
                "recheck_outcome": self.common_recheck_outcome,
                "prefix_bounds": pairs(self.common_prefix_bounds),
            },
        }


class PruningBudgetExhausted(ValueError):
    """Signal that pruning leaves no budget for a nonzero formula bound."""


# Outward composition of a pruning charge with a product-formula bound.
#
# Let epsilon be the exact value of the requested finite nonnegative binary64
# allowance and d the exact sum of magnitudes of the dropped coefficients of
# the Hermitian target. Every Pauli string has unit norm, so
# ||H - H_kept|| <= d, and Duhamel's formula between two unitary evolutions
# bounds the pruning error by |t|*d. With a stored upper mass d_up >= d,
# publish P = up(|t|*d_up) from the exact product of the stored values. If
# P > epsilon pruning alone is inadmissible. Otherwise R = down(epsilon - P),
# the step inversion selects E_r = W_up*|t|**(o+1)/r**o <= R, and E = up(E_r)
# is also at most R because R is a binary64 number above E_r. The published
# total T = up(P + E) is at most epsilon, which is representable, so
#   |t|*d + E_r <= P + E <= T <= epsilon.
# Rounding the pruning and formula contributions upward only after selecting
# against their unrounded sum can publish contributions whose sum exceeds the
# allowance; reserving the published P first avoids that. With ordered terms
# X:0.5, Z:0.5, order one, time one, epsilon the published coefficient and
# pruning 2**-80, the nearest-rounded remainder equals epsilon and accepts
# r = 1, while the exact published sum exceeds epsilon by 2**-80; the
# downward remainder selects r = 2. Since R_down <= R_near for the same W_up
# and time, the selected count cannot decrease; it can grow by more than one
# step when the count is large, and a remainder that rounds to zero refuses a
# nonzero formula bound. A mass upper bound can be reused across times; the
# pruning charge, remainder and total belong to each application.
# Source: NWQLib's composition derived for this owner (triangle inequality
# and Duhamel's formula for unitary evolutions), not a paper theorem.


def _finite_upper(value, name):
    """Return the least binary64 number at or above the nonnegative rational ``value``.

    Raises ValueError for a negative value or one above the finite binary64
    range, so an infinite upper bound never enters a finite record.
    """
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    result = _upward_float(value)
    if not isfinite(result):
        raise ValueError(f"{name} exceeds the finite binary64 range")
    return result


def _add_nonnegative_up(a, b):
    """Return a binary64 upper bound on a + b for finite nonnegative binary64 a and b.

    If either operand is zero the exact sum is the other one. Otherwise the
    exact sum lies no higher than the next binary64 number above the
    round-to-nearest sum r, so nextafter(r, inf) bounds it, also for
    subnormal results. A nonfinite result raises.
    """
    if a == 0.0:
        return b
    if b == 0.0:
        return a
    result = nextafter(a + b, inf)
    if not isfinite(result):
        raise ValueError("dropped coefficient mass enclosure exceeds binary64 range")
    return result


def upper_dropped_mass(dropped_coefficients):
    """Return mass_up >= sum_D |c_j| over the dropped stored complex coefficients c_j.

    D is exactly the set selected by the caller's cutoff test. For
    c = x + iy with binary64 components, |c| = sqrt(x**2 + y**2) <= |x| + |y|,
    and each |x|, |y| is exact. The accumulator starts at zero and adds every
    real and imaginary absolute component upward (``_add_nonnegative_up``),
    so by induction it bounds the exact component sum S >= d = sum_D |c_j|;
    S = d for real coefficients and can exceed d by at most a factor sqrt(2)
    for complex residues before accumulation rounding. All-zero input stays
    exactly zero, and exact cancellation contributes zero. The result can
    exceed the least binary64 upper publication of S after repeated upward
    additions, an explicit accuracy/cost choice of a constant-work streamed
    enclosure; it may increase selected steps or exhaust an allowance. A
    nearest-rounded fsum of moduli is not an upper bound: for 2**-41 and
    2**-100 it returns 2**-41. Final budget composition keeps the
    least-upward rational publication (``upper_pruning_error``,
    ``upper_combined_error``).

    Work, in scalar visits: at most 16 per dropped coefficient (one complex
    conversion, two finiteness checks and their combination, two absolute
    values and two upward additions of five visits each) and four per call,
    so 16*d + 4 for d dropped coefficients. Scratch is 128 logical bytes
    once. These units are admission proxies, not timings or equal-cost CPU
    operations. Source: NWQLib's derivation for this owner (Python's
    math.nextafter directed step), not a paper theorem.
    """
    mass = 0.0
    for coefficient in dropped_coefficients:
        c = complex(coefficient)
        if not (isfinite(c.real) and isfinite(c.imag)):
            raise ValueError("dropped coefficients must be finite")
        mass = _add_nonnegative_up(mass, abs(c.real))
        mass = _add_nonnegative_up(mass, abs(c.imag))
    return mass


def upper_pruning_error(time, dropped_mass):
    """Return P = up(|t| * d_up), the published pruning bound at one evolution time.

    ``time`` and ``dropped_mass`` must be finite binary64 values. Their
    exact product is published upward, so a positive product below the least
    subnormal is published as the least subnormal, not zero.
    """
    t = _binary64(time, "evolution time")
    d = _binary64(dropped_mass, "dropped mass upper bound")
    if d < 0:
        raise ValueError("dropped mass upper bound must be nonnegative")
    return _finite_upper(abs(Fraction(t))*Fraction(d), "pruning error")


def _remaining_budget_after_pruning(*, total_budget, pruning_error):
    """Return R = down(epsilon - P), the allowance left for the product formula.

    ``pruning_error`` is the caller's published upward pruning bound P
    (``upper_pruning_error``). The exact remainder is rounded downward, so a
    formula bound published upward within R keeps the exact sum of the two
    published contributions at or below ``total_budget``. A negative
    remainder raises PruningBudgetExhausted. A zero remainder admits only an
    exactly zero formula bound.
    """
    budget = _binary64(total_budget, "total budget")
    pruning = _binary64(pruning_error, "pruning error upper bound")
    if budget < 0 or pruning < 0:
        raise ValueError("budgets and pruning bounds must be nonnegative")
    remainder = Fraction(budget)-Fraction(pruning)
    if remainder < 0:
        raise PruningBudgetExhausted
    return _downward_float(remainder)


def upper_combined_error(pruning, formula):
    """Return T = up(P + E), the published total of a pruning and a formula bound.

    An empty generator passes zero for the formula bound. A fixed
    certificate uses the same upward total at its saved steps.
    """
    p = _binary64(pruning, "pruning bound")
    e = _binary64(formula, "formula bound")
    if p < 0 or e < 0:
        raise ValueError("error contributions must be nonnegative")
    return _finite_upper(Fraction(p)+Fraction(e), "combined error")


def _validated_terms(hamiltonian: SparsePauliOp) -> list[tuple[str, float]]:
    """Return ``(label, real coefficient)`` summands in the operator's term order.

    Rejects non-real coefficients, because the bounds and the delegated
    rotations require Hermitian summands. Also rejects identity terms. An
    identity summand commutes with everything, so every commutator in
    Eqs. (120)-(121) vanishes and it contributes only the global phase
    ``exp(-i t c)``, which the compact Pauli-evolution representation cannot
    carry. Callers of the public functions remove it and track that phase
    themselves. The QPE Methods split it off in the same way
    (algorithms/qpe/method.py, _split_pauli_terms) and apply it as a control
    phase.
    """

    terms: list[tuple[str, float]] = []
    for label, coefficient in hamiltonian.to_list():
        value = complex(coefficient)
        # NaN or infinity must not read as an absent or ordinary contribution.
        if not (isfinite(value.real) and isfinite(value.imag)):
            raise ValueError(f"Hamiltonian coefficient of term {label!r} is {value}, not finite")
        # Same tolerance as the delegated native rotation path
        # (hamiltonian_evolution/pauli_evolution.py): library-native
        # constructions (from_operator, operator products) carry ~1e-17j
        # residues on exactly Hermitian inputs and stay accepted. The
        # threshold is absolute, not relative to the coefficient scale.
        # Registered in docs/ENGINEERING_CONSTANTS.md. Revisit for operators
        # whose coefficients are far from unit scale.
        if abs(value.imag) > 1.0e-12:
            raise ValueError(
                "Hamiltonian coefficients must be real for Trotter error bounds; "
                f"term {label!r} has coefficient {value} (build the operator from "
                "real coefficients or simplify away numerical imaginary parts)"
            )
        if set(label) == {"I"}:
            raise ValueError(
                "Hamiltonian contains an identity term, which only shifts the "
                "global phase and does not change any Trotter error bound "
                "(all its commutators vanish); remove it from the operator "
                "and track exp(-i * time * coefficient) classically"
            )
        terms.append((label, value.real))
    return terms


def _validate_order(order: int) -> int:
    """Return ``order`` as int if it is 1 (Lie-Trotter) or 2 (second-order Suzuki), else raise ValueError."""
    if order not in _SYNTHESIS_METHODS:
        raise ValueError(
            "formula order must be 1 (first-order Lie-Trotter, CSTWZ Phys. Rev. X "
            "doi:10.1103/PhysRevX.11.011020 "
            f"Prop. 9) or 2 (second-order Suzuki, Prop. 10); got {order!r}"
        )
    return int(order)


def _validate_variant(variant: str, order: int) -> str:
    """Return the recorded variant: ``variant`` at order 2, "exact_census" at order 1.

    At order 1 both variants are the same pair-only expression W_1, which
    the record names "exact_census".
    """
    if variant not in BOUND_VARIANTS:
        raise ValueError(f"bound_variant must be one of {BOUND_VARIANTS}; got {variant!r}")
    return "exact_census" if order == 1 else variant


def _validate_time(time: float) -> float:
    """Return ``time`` as a positive finite float. Props. 9-10 are stated for t >= 0."""
    value = float(time)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"evolution time must be positive and finite; got {time!r} "
            "(the Prop. 9-10 bounds and the step selection are stated for t > 0)"
        )
    return value


def _term_matrices(terms: list[tuple[str, float]]) -> list[np.ndarray]:
    """Dense validation-scale matrices ``c_gamma * P_gamma`` (cost ``4**n``)."""
    from qiskit.quantum_info import SparsePauliOp

    return [
        np.asarray(SparsePauliOp.from_list([(label, coefficient)]).to_matrix())
        for label, coefficient in terms
    ]


def _suffix_pairs(matrices: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pair each summand with its later-term sum, preserving summand order."""

    tail = np.zeros_like(matrices[0])
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for head in reversed(matrices):
        pairs.append((head, tail.copy()))
        tail = tail + head
    pairs.reverse()
    return pairs


def _spectral_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, 2))


def _first_order_commutator_sum(matrices: list[np.ndarray]) -> float:
    """``C1 = sum_gamma1 ||[T_gamma1, H_gamma1]||`` with the Eq. (120) tail sum.

    Bilinearity gives ``sum_{gamma2 > gamma1} [H_gamma2, H_gamma1] =
    [T_gamma1, H_gamma1]`` for the suffix sum ``T_gamma1``, maintained in
    reverse so the pass costs two matrix products per summand.
    """

    total = 0.0
    for head, tail_sum in _suffix_pairs(matrices):
        total += _spectral_norm(tail_sum @ head - head @ tail_sum)
    return total


def _second_order_commutator_sums(matrices: list[np.ndarray]) -> tuple[float, float]:
    """``(C12, C24)`` of Eq. (121), with ``T_gamma1 = sum_{gamma2 > gamma1} H_gamma2``.

    ``C12 = sum_gamma1 ||[T_gamma1, [T_gamma1, H_gamma1]]||`` and
    ``C24 = sum_gamma1 ||[H_gamma1, [H_gamma1, T_gamma1]]||``.
    """

    c12 = 0.0
    c24 = 0.0
    for head, tail_sum in _suffix_pairs(matrices):
        inner_t = tail_sum @ head - head @ tail_sum  # [T_gamma1, H_gamma1]
        c12 += _spectral_norm(tail_sum @ inner_t - inner_t @ tail_sum)
        inner_h = -inner_t  # [H_gamma1, T_gamma1] is the exact negation
        c24 += _spectral_norm(head @ inner_h - inner_h @ head)
    return c12, c24


@dataclass(frozen=True)
class _BoundCoefficientEvaluation:
    """Prefactor W_up of bound(t) = W_up*t**(order+1) with the census work that produced it.

    Attributes:
        coefficient: W_up, a rational upper bound on the selected
            Pauli-triangle expression (``coefficient_up``).
        pauli_term_count: Number of Pauli terms in the census.
        pair_commutation_checks: Logical pair tests performed, P = p(p-1)/2.
        nested_commutation_checks: Logical nested tests performed, N for
            the exact order-2 structure and zero otherwise.
        bound_variant: "exact_census" or "relaxed_prefix".
        coefficient_arithmetic: "outward_float64_scaled".
    """

    coefficient: Fraction
    pauli_term_count: int
    pair_commutation_checks: int
    nested_commutation_checks: int
    bound_variant: str
    coefficient_arithmetic: str


def _upward_float(value):
    """Return the least binary64 number at or above the nonnegative rational ``value``.

    It is zero only for a zero value, ``2**-1074`` for a positive value below
    that, and ``inf`` above the largest finite number, which callers turn
    into an explicit range error.
    """
    if value > Fraction(float_info.max):
        return inf
    result = float(value)
    return result if Fraction(result) >= value else nextafter(result, inf)


def _downward_float(value):
    """Return the greatest binary64 number at or below the nonnegative rational ``value``."""
    result = float(value) if value <= Fraction(float_info.max) else float_info.max
    return result if Fraction(result) <= value else nextafter(result, -inf)


def _binary64(value, name):
    """Return ``value`` as a finite binary64 number, refusing a value that it does not represent exactly.

    The common-step recheck compares exact values of stored binary64
    numbers, so a value that would round on conversion is refused rather
    than replaced. A NumPy floating scalar is compared by its exact value,
    and a string or a value without a real rational value raises the same
    ValueError.
    """
    try:
        if isinstance(value, (str, bytes)):
            raise TypeError
        if isinstance(value, np.floating):
            exact = Fraction(*value.as_integer_ratio())
        else:
            exact = Fraction(value)
        result = float(exact)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite binary64 value") from None
    if not isfinite(result) or Fraction(result) != exact:
        raise ValueError(f"{name} must be a finite binary64 value")
    return result


# Pauli-triangle kernels. Callers admit work and arrays before entry
# (census_work, census_bytes).
#
# Structure. For masks x_i, z_i define the symplectic parity
# s_ij = (popcount(x_i & z_j) + popcount(z_i & x_j)) mod 2. The masks encode
# only labels, so s_ij is independent of time and coefficient sign. The masks
# of P_i P_j, ignoring its scalar phase, are (x_i XOR x_j, z_i XOR z_j), and
# bilinearity gives s_(k,ij) = s_ki XOR s_kj. Define
#   A = {(i,j): 0 <= i < j < p, s_ij = 1},
#   T = {(i,j,k): (i,j) in A, i < k < p, s_ki XOR s_kj = 1}.
# In particular (i,j,j) is in T for every (i,j) in A: there is no condition
# k != j or k > j, and both orders of distinct tail indices occur when both
# inner pairs survive. Then
#   W_1 = sum_A a_i a_j,   W_2 = (1/3) sum_T a_i a_j a_k + (1/6) sum_A a_i^2 a_j.
# Put E = |A|, F = |T|, w = ceil(q/64), P(p) = p(p-1)/2,
# N(A) = sum_((i,j) in A) (p-i-1) and J(p) = p(p-1)(2p-1)/6. There are
# exactly P(p) pair tests in the direct triangular scan and exactly N(A) <= J(p)
# nested tests, and F <= N(A). The inequalities are safe envelopes, not
# simultaneously attainable maxima for every qubit width. Masks cost O(pq)
# from strings and store 16pw bytes; pair structure takes O(wP) work and
# O(E) index storage; triple structure takes O(wN) further work and O(F)
# index storage (16E and 24F bytes with 64-bit indices).
#
# Relaxation. With S_i = sum_(k=i+1)^(p-1) a_k, all summands are
# nonnegative, so replacing the nested indicator by one gives
#   C12_triangle <= C12_rel = 4 sum_A a_i a_j S_i,
#   W_2,rel = (1/3) sum_A a_i a_j S_i + (1/6) sum_A a_i^2 a_j >= W_2,
# over exactly the range above, including k = j. The theorem's tail-norm
# coefficient is at most W_2, so it is also at most W_2,rel, under the same
# ordered Hermitian Pauli decomposition and pruning choice. Order one uses
# only pair tests and needs no relaxation. Suffix sums are a reversed
# cumulative sum, not total-prefix, which would cancel. For the same positive
# time and allowance, r_exact <= r_rel <= ceil(alpha*r_exact) with
# alpha = sqrt(W_2,rel/W_2) when W_2 > 0; the order X,Y,Z with unit
# magnitudes gives W_2 = 3/2 and W_2,rel = 13/6, and at t = 1, epsilon = 3/2
# the counts are 1 and 2. For zero W both counts are one.
#
# Source: the Pauli-triangle expansions of Childs et al.,
# doi:10.1103/PhysRevX.11.011020, Eqs. (120)-(121).


def pack_labels(labels, q):
    """Pack label character b into bit b, matching the leftmost-character-first masks.

    Returns uint64 X/Z arrays of shape (len(labels), ceil(q/64)).
    Inputs are validated, equally wide, ordered canonical Pauli labels.
    This convention reverses the physical-bit order of PauliTerms, where
    physical qubit zero, the rightmost character, is bit zero. Never mix
    tables from the two conventions.
    """
    x = np.zeros((len(labels), (q + 63) // 64), dtype=np.uint64)
    z = np.zeros_like(x)
    for i, label in enumerate(labels):
        for b, symbol in enumerate(label):
            bit = np.uint64(1) << np.uint64(b % 64)
            if symbol in "XY":
                x[i, b // 64] |= bit
            if symbol in "YZ":
                z[i, b // 64] |= bit
    return x, z


def anti(x1, z1, x2, z2):
    """Broadcast Pauli pairs and return their symplectic parity.

    The last axis consists of uint64 words in one consistent convention.
    Two strings anticommute exactly when
    popcount(x1 & z2) + popcount(z1 & x2) is odd.
    """
    bits = (x1 & z2) ^ (z1 & x2)
    return (np.bitwise_count(bits).sum(axis=-1, dtype=np.uint64) & 1) != 0


def pair_structure(x, z):
    """Return anticommuting i<j pairs and the p*(p-1)//2 logical tests.

    Work is O(p^2*w), stored output O(|A|), workspace O(p*w).
    """
    chunks = []
    p = len(x)
    for i in range(p - 1):
        js = np.flatnonzero(anti(x[i], z[i], x[i + 1:], z[i + 1:])) + i + 1
        if js.size:
            chunks.append(np.column_stack((np.full(js.size, i, dtype=np.intp), js)))
    pairs = np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.intp)
    return pairs, p * (p - 1) // 2


def triple_structure(x, z, pairs):
    """Return (i,j,k) with (i,j) in A, k>i, and P_k anti with P_i P_j.

    The range includes k=j. Work is O(w*sum_A(p-i-1)), storage O(|T|).
    Chunk concatenation can temporarily double the stored index bytes.
    """
    chunks = []
    checks = 0
    for i, j in pairs:
        ks = np.flatnonzero(anti(x[i] ^ x[j], z[i] ^ z[j], x[i + 1:], z[i + 1:])) + i + 1
        checks += len(x) - int(i) - 1
        if ks.size:
            chunks.append(np.column_stack((np.full(ks.size, i, dtype=np.intp),
                                           np.full(ks.size, j, dtype=np.intp), ks)))
    triples = np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.intp)
    return triples, checks


def _sum_factor(n):
    """Upper binary64 value of 1/(1-gamma_(n-1)), with gamma<1.

    With u = 2**-53 and gamma_s = s*u/(1-s*u), a valid upper factor is
    S <= S_hat/(1-gamma_s) = S_hat*(1-s*u)/(1-2*s*u) for gamma_s < 1.
    Multiplying by 1+gamma_s alone is insufficient.
    """
    k = max(0, int(n) - 1)
    d = 1 << 53
    if 2 * k >= d:
        raise ValueError("sum too long for this rounding bound")
    return _upward_float(Fraction(d - k, d - 2 * k))


def mul_up(a, b):
    """Outward nonnegative multiplication, including gradual underflow.

    Finite nonnegative inputs and no overflow are required. Zero factors
    give exact zero. Positive underflow gives the smallest subnormal.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        c = a * b
    return np.where((a > 0) & (b > 0), np.nextafter(c, np.inf), 0.0)


def sum_up(values):
    """Bound the sum of nonnegative binary64 values for any addition tree.

    Uses the sequential gamma bound even when NumPy sums pairwise.
    Nonnegative subnormal additions are exact, so they need no extra term.
    With u = 2**-53, round-to-nearest binary64 and no overflow, underflow or
    nonfinite input, a sum of N terms satisfies
    |S_hat - S| <= gamma_(N-1)*S, so S <= S_hat/(1 - gamma_(N-1))
    (``_sum_factor``).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values.flat[0])
    return float(mul_up(values.sum(dtype=np.float64), _sum_factor(values.size)))


def scaled_magnitudes(magnitudes):
    """Return a>=magnitudes/2**e and exponent e without materializing 2**e.

    Inputs are finite nonnegative binary64 values. Exact zeros stay zero.
    Scaling and products use IEEE binary64 with gradual underflow.
    """
    m = np.asarray(magnitudes, dtype=np.float64)
    maximum = float(m.max(initial=0.0))
    if maximum == 0:
        return m.copy(), 0
    e = frexp(maximum)[1]
    with np.errstate(under="ignore"):
        a = np.ldexp(m, -e)
    return np.where(m > 0, np.nextafter(a, np.inf), 0.0), e


def coefficient_up(magnitudes, pairs, *, order, variant, triples=None, block=65536):
    """Return a rational upper bound on the selected Pauli-triangle W.

    variant is exact_census or relaxed_prefix. At order 1 both use A only.
    At order 2 exact_census uses T, and relaxed_prefix uses all k>i.
    Weights and reductions are outward. The power-of-two scale and the
    divisors 3 and 6 are combined as Fractions once after each reduction.
    This certifies W, without claiming exact evaluation of that W.

    Premises: binary64 round-to-nearest, finite nonnegative inputs and the
    stated reduction tree (``mul_up``, ``sum_up``). The result W_up is a
    rational upper bound, not the exact rational Pauli-triangle
    coefficient, so step selection uses the rational W_up*t**(o+1).
    Underflow-aware outward rounding preserves validity over the finite
    input range but has no useful uniform relative-error guarantee when
    monomials underflow after normalization. ``block`` is a scratch-size
    choice (``choose_census_block``), not a tolerance; the bound holds for
    every positive block size, and no p*p*p array is allocated.
    """
    a, e = scaled_magnitudes(magnitudes)
    if order not in (1, 2) or variant not in ("exact_census", "relaxed_prefix"):
        raise ValueError("invalid formula or bound variant")
    if order == 2 and variant == "exact_census" and triples is None:
        raise ValueError("exact census requires its triple structure")
    suffix = np.zeros(len(a), dtype=np.float64)
    if order == 2 and variant == "relaxed_prefix" and len(a) > 1:
        raw = np.cumsum(a[:0:-1], dtype=np.float64)[::-1]
        suffix[:-1] = mul_up(raw, _sum_factor(len(a) - 1))
    part1, part24, part12 = [], [], []
    for start in range(0, len(pairs), block):
        i, j = pairs[start:start + block].T
        ij = mul_up(a[i], a[j])
        if order == 1:
            part1.append(sum_up(ij))
        else:
            part24.append(sum_up(mul_up(ij, a[i])))
            if variant == "relaxed_prefix":
                part12.append(sum_up(mul_up(ij, suffix[i])))
    if order == 1:
        return Fraction(sum_up(part1)) * Fraction(2) ** (2 * e)
    if variant == "exact_census":
        for start in range(0, len(triples), block):
            i, j, k = triples[start:start + block].T
            part12.append(sum_up(mul_up(mul_up(a[i], a[j]), a[k])))
    w = Fraction(sum_up(part12)) / 3 + Fraction(sum_up(part24)) / 6
    return w * Fraction(2) ** (3 * e)


def census_sizes(p):
    """Return ``(P, J) = (p(p-1)/2, p(p-1)(2p-1)/6)`` for p kept terms.

    P is the exact number of pair tests of the triangular scan, and J bounds
    the nested test population N(A) = sum_((i,j) in A)(p-i-1) <= J.
    """
    return p * (p - 1) // 2, p * (p - 1) * (2 * p - 1) // 6


def nested_test_count(pairs, p):
    """Return N(A) = sum_((i,j) in A)(p-i-1), the nested tests of the exact structure.

    Once A has been admitted and constructed, N is an O(E) count obtained
    without further tests.
    """
    return int(np.sum(p - 1 - np.asarray(pairs)[:, 0], dtype=np.int64)) if len(pairs) else 0


def census_work(p, q, *, order, variant, pairs, nested=0, triples=0):
    """Return G + V, the work charge of one shared census structure.

    A packed symplectic test costs w = ceil(q/64) constant-size word visits,
    and one logical magnitude-contribution visit is charged for each term
    evaluated, with linear passes charged separately:

        G_1 = wP,  G_C = wP,  G_D = w(P+N);
        V_1 = E,   V_C = p+2E, V_D = E+F.

    Here C is relaxed_prefix and D exact_census at order 2, E = ``pairs``,
    N = ``nested`` and F = ``triples``. Before any structure exists use E = P
    and N = F = J (``census_sizes``); before T exists use V_D <= E+N, that
    is F = N. The fixed number of float products, outward operations and
    additions per contribution is bundled in a visit. This is an explicit
    engineering convention, an admission proxy and not a runtime
    measurement. A caller adds its own linear passes, for QPE ``2*p*q``.
    """
    if min(p, q, pairs, nested, triples) < 0:
        raise ValueError("invalid census sizes")
    if order not in (1, 2) or variant not in BOUND_VARIANTS:
        raise ValueError("invalid census kind")
    w = (q + 63) // 64
    total_pairs = p * (p - 1) // 2
    if order == 1:
        return w * total_pairs + pairs
    if variant == "relaxed_prefix":
        return w * total_pairs + p + 2 * pairs
    return w * (total_pairs + nested) + pairs + triples


def census_bytes(p, q, pairs, triples, block, *, order, variant, held=0):
    """Return the byte envelope of one census, including B_held (``held``).

    With w = ceil(q/64), E = ``pairs``, F = ``triples``, b = ``block`` and
    H0 = 65536 bytes of scalar bookkeeping, ndarray headers, iterators and
    fixed sort stacks on the checked 64-bit CPython/NumPy stack:

        B = B_held + 80p + 16pw + 32E + 48F + H_chunks
            + max(R(p,w)+H0, C(p,E,F,b)),
        R(p,w) = 32pw+64p+32w,
        H_chunks = 384[p+1 + I_D(E+1)],
        C(p,E,F,b) = 56p+64b+48H(b)+H0,

    with H(b) = ceil(E/b) at order 1, 2 ceil(E/b) for relaxed_prefix and
    ceil(E/b)+ceil(F/b) for exact_census at order 2, and I_D = 1 only for
    the order-2 exact_census builder. F = 0 for order one or relaxed_prefix.
    80p prices the census inputs formed from borrowed labels and
    coefficients, 16pw the masks, 32E and 48F pair and triple storage during
    concatenation, R one row test (never a p^3 broadcast cube), 384 bytes a
    chunk object, 56p the linear contraction scratch, 64b one block's
    gathers and products, and 48 bytes each partial sum kept between blocks.
    Before any structure work use E = P and, for D, F = J; after the pairs
    exist, F = N reserves the triple scan. Census admission includes packed
    masks, simultaneous pair and triple storage, chunk-object overhead, row
    tests and bounded coefficient-reduction workspace. The block size is
    selected from the remaining byte allowance, including the partial sums
    that accumulate between blocks. H0 is an engineering allowance, not a
    universal interpreter or process-RSS theorem.
    """
    if min(p, q, pairs, triples, held) < 0 or block < 1:
        raise ValueError("invalid census sizes")
    if order not in (1, 2) or variant not in ("exact_census", "relaxed_prefix"):
        raise ValueError("invalid census kind")
    exact2 = order == 2 and variant == "exact_census"
    F = triples if exact2 else 0
    w = (q+63)//64
    pair_blocks = (pairs+block-1)//block
    if order == 1:
        partials = pair_blocks
    elif exact2:
        partials = pair_blocks+(F+block-1)//block
    else:
        partials = 2*pair_blocks
    objects = 384*(p+1+(pairs+1 if exact2 else 0))
    row = 32*p*w+64*p+32*w+65536
    contraction = 56*p+64*block+48*partials+65536
    return (held+80*p+16*p*w+32*pairs+48*F+objects
            +max(row, contraction))


def choose_census_block(p, q, E, F, max_bytes, *, order, variant, held=0):
    """Return ``(block, bytes)`` for the largest fitting candidate block, or None.

    The candidates are max(1, E, F) and the powers of two below it, at most
    64 integer-law evaluations for intp-sized populations. None says that
    none of the checked choices is admitted, not that every integer block is
    impossible.
    """
    # Largest fitting member of a finite geometric candidate set.
    top = max(1, E, F if order == 2 and variant == "exact_census" else 0)
    choices = {top}
    choices.update(1 << k for k in range(top.bit_length()))
    for block in sorted(choices, reverse=True):
        size = census_bytes(p, q, E, F, block, order=order,
                            variant=variant, held=held)
        if size <= max_bytes:
            return block, size
    return None


def _pauli_bound_coefficient_from_terms(
    terms: tuple[tuple[str, float], ...],
    order: int,
    *,
    variant: str = "exact_census",
    block: int = 65536,
    choose=None,
) -> _BoundCoefficientEvaluation:
    """Bound the ordered Pauli-triangle coefficient W from validated terms.

    For order 1, W=sum_A |c_i*c_j|, where A contains anticommuting i<j.
    For order 2, the exact_census variant returns an outward bound on
    sum_T |c_i*c_j*c_k|/3 + sum_A |c_i|**2*|c_j|/6. Here T contains
    (i,j,k) with (i,j) in A, k>i, and P_k anticommuting with P_i*P_j.
    The range includes k=j. The relaxed_prefix variant replaces the
    inner sum over such k by the full suffix sum of coefficient magnitudes.
    It therefore gives an upper bound using pair structure alone.

    These are triangle expansions of Childs et al.,
    doi:10.1103/PhysRevX.11.011020, Eqs. (120)-(121). The selected
    formula obeys ||S_o(t/r)**r-exp(-i*t*H)|| <= W*abs(t)**(o+1)/r**o.
    The ordered labels determine the shared structure, and the supplied
    real coefficients determine its weights. Identity phases and pruning
    are accounted for by the caller.

    Scaled binary64 products and sums are rounded outward. W is returned
    as a rational upper coefficient so its exponent can be combined with
    time without intermediate binary64 overflow or loss to zero. Step
    minimality is relative to that coefficient. A full indicator census
    does not imply exact arithmetic or the exact simulation error.

    A caller with an automatic variant policy may select exact_census when
    it fits, or relaxed_prefix when that expression fits. The public
    helpers in this module preserve an explicitly requested variant and
    refuse when its admission fails.
    The full structure takes O(w*p**3) worst-case word work, and the
    relaxed structure takes O(w*p**2), with w=ceil(q/64). Reused structure
    still requires node-specific magnitude contractions. The result
    records its bound variant, arithmetic and newly incurred checks.

    ``choose(E, N)``, when given, is called after the pair structure is
    built and before any triple test, with the actual pair count E and
    nested test population N (``nested_test_count``). It returns the
    ``(variant, block)`` to use, after admitting that stage, or raises to
    refuse it. Without it the census uses ``variant`` and ``block``
    unadmitted. The public helpers of this module admit it through
    ``_bound_coefficient_evaluation``. At order 1 the variant is
    "exact_census".

    Storage: the masks take 16pw bytes and the pair and triple index
    tables 16E and 24F bytes with 64-bit indices, and concatenation can
    temporarily double the index bytes. F <= N <= J = p(p-1)(2p-1)/6, so
    the only bound on the triple table of an unadmitted exact_census at
    order 2 is that cubic envelope. Order 1 and relaxed_prefix build no
    triple table.
    """
    labels = [label for label, _ in terms]
    magnitudes = np.array([abs(float(coefficient)) for _, coefficient in terms], dtype=np.float64)
    x, z = pack_labels(labels, len(labels[0]) if labels else 0)
    pairs, pair_checks = pair_structure(x, z)
    if choose is not None:
        variant, block = choose(len(pairs), nested_test_count(pairs, len(labels)))
    variant = _validate_variant(variant, order)
    triples, nested_checks = None, 0
    if order == 2 and variant == "exact_census":
        triples, nested_checks = triple_structure(x, z, pairs)
    coefficient = coefficient_up(
        magnitudes, pairs, order=order, variant=variant, triples=triples, block=block
    )
    return _BoundCoefficientEvaluation(
        coefficient=coefficient,
        pauli_term_count=len(terms),
        pair_commutation_checks=pair_checks,
        nested_commutation_checks=nested_checks,
        bound_variant=variant,
        coefficient_arithmetic=COEFFICIENT_ARITHMETIC,
    )


def _dense_bound_coefficient_from_terms(
    terms: tuple[tuple[str, float], ...],
    order: int,
) -> float:
    """Return W from the exact spectral norms of Eqs. (120)-(121), using dense matrices.

    Validation only. Each dense matrix has 4**n entries for n qubits, and
    each commutator product and spectral norm costs O(8**n) operations.
    """
    matrices = _term_matrices(list(terms))
    if order == 1:
        return _first_order_commutator_sum(matrices) / 2.0
    c12, c24 = _second_order_commutator_sums(matrices)
    return c12 / 12.0 + c24 / 24.0


def _validate_limit(value, name, *, minimum):
    """Return ``value`` as an int of at least ``minimum``, else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        kind = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {kind} integer; got {value!r}")
    return int(value)


def _bound_coefficient_evaluation(
    hamiltonian: SparsePauliOp,
    *,
    order: int,
    bound_variant: str = "exact_census",
    max_work: int = DEFAULT_CENSUS_MAX_WORK,
    max_bytes: int = DEFAULT_MAX_BYTES,
    held: int = 0,
) -> _BoundCoefficientEvaluation:
    """Admit, validate and evaluate the operator's Pauli-triangle W_up.

    The sparse Pauli census checks its work and byte allowances before
    converting labels or constructing its index tables. The full
    second-order expression stores three intp indices per surviving triple
    and can require cubic storage in the number of terms. The contraction
    block limits coefficient-reduction scratch and does not limit the
    stored index population. If the requested expression does not fit, the
    helper raises before the unadmitted stage. At order two, the explicitly
    selected suffix relaxation uses pair structure and a potentially larger
    error coefficient.

    Admission is staged. With p terms on q qubits, the operator supplies p
    and q before ``hamiltonian.to_list()``. The label/term conversion
    envelope ``128*p*(q+1)+65536`` bytes stays held while the census borrows
    the validated labels, and ``16*p*(q+1)`` logical preparation work plus
    ``p*q`` label visits for packing are added to ``census_work``. The pair
    phase is admitted first with ``E = P = p*(p-1)//2`` and no triples.
    After the pair scan gives the actual E and N, the requested variant is
    admitted before any triple test. The full second-order expression
    reserves F = N. Order one and relaxed_prefix charge no nested tests or
    triples. Each complete census_work expression includes the pair work
    already spent once. A requested bound_variant is preserved. held is
    caller-owned live bytes outside the census and its label-conversion
    envelope.

    The pair stage is admitted before label conversion. The requested
    expression is admitted again using the actual pair count before nested
    tests or contraction. A refusal identifies the failed stage and reports
    a sufficient complete envelope for the requested order and variant. Its
    block-one byte value is a checked candidate, rather than a minimum over
    all block sizes. A byte-fit failure is reported only when no checked
    candidate fits the current byte allowance. No triple table or
    coefficient is created on refusal.

    The complete envelope is computed before any work. With w = ceil(q/64),
    ``P = p(p-1)/2``, ``J = p(p-1)(2p-1)/6`` and ``L0 = 16p(q+1)+pq``, the
    complete sufficient work before label conversion is
    ``R_1 = L0 + (w+1)P`` at order one, ``R_relaxed = L0 + wP + p + 2P``
    for relaxed_prefix and ``R_exact2 = L0 + w(P+J) + P + J`` for the
    order-two exact census. The pair-stage charge is R_1, whatever
    order-two expression was requested. The complete order-two bounds
    substitute E = P, N = F = J for exact_census and E = P, N = F = 0 for
    relaxed_prefix; the actual post-pair requirements cannot exceed them
    because E <= P, N <= J and F <= N. The label-conversion held bytes are
    ``held + 128p(q+1) + 65536``. The complete work remedy is printed
    whenever the complete conservative work exceeds ``max_work``, qualified
    as sufficient rather than necessary, even when the refused stage failed
    only on bytes. Raising work alone does not admit an independently
    insufficient byte allowance, shape capacity, nonfinite coefficient or
    non-Hermitian term; the intp shape capacity is a separate refusal that
    no resource cap cures.

    The block search first selects the largest fitting candidate from
    choose_census_block. If that block exceeds 65536 and the byte envelope
    at 65536 also fits, this helper uses 65536. Otherwise it uses the
    selected block. Admission includes the partial sums for the block
    actually used. Changing the block can change the outward coefficient
    through the grouping of reductions, while preserving its upper-bound
    property. A relative rounding window must use that actual block.
    """
    order = _validate_order(order)
    variant = _validate_variant(bound_variant, order)
    max_work = _validate_limit(max_work, "max_work", minimum=1)
    max_bytes = _validate_limit(max_bytes, "max_bytes", minimum=1)
    held = _validate_limit(held, "held", minimum=0)
    p, q = int(hamiltonian.size), int(hamiltonian.num_qubits)
    P, J = census_sizes(p)
    label_bytes = 128*p*(q+1) + 65536
    linear = 16*p*(q+1) + p*q
    census_held = held + label_bytes
    capacity = np.iinfo(np.intp).max
    complete_F = J if order == 2 and variant == "exact_census" else 0
    complete_work = linear + census_work(
        p, q, order=order, variant=variant, pairs=P,
        nested=complete_F, triples=complete_F,
    )
    complete_bytes = census_bytes(
        p, q, P, complete_F, 1, order=order, variant=variant,
        held=census_held,
    )

    def admit(E, N, chosen_order, chosen_variant, stage):
        exact2 = chosen_order == 2 and chosen_variant == "exact_census"
        F = N if exact2 else 0
        if max(p*((q+63)//64), 2*E, 3*F) > capacity:
            raise ValueError(
                f"Pauli census {chosen_variant} with p={p}, q={q}, E={E}, F={F} "
                "exceeds the intp array shape capacity"
            )
        work = linear + census_work(
            p, q, order=chosen_order, variant=chosen_variant,
            pairs=E, nested=N if exact2 else 0, triples=F,
        )
        fit = choose_census_block(
            p, q, E, F, max_bytes, order=chosen_order,
            variant=chosen_variant, held=census_held,
        )
        if work > max_work or fit is None:
            message = (
                f"Pauli census {stage} stage needs work={work}, "
                f"max_work={max_work}, max_bytes={max_bytes}. "
                f"The complete requested order-{order} {variant} envelope needs "
                f"max_work={complete_work} and has the sufficient checked "
                f"block-one byte candidate {complete_bytes}."
            )
            if complete_work > max_work:
                message += f" Raise max_work to at least {complete_work}."
            if fit is None:
                message += (
                    " No checked block fits the current byte allowance. "
                    f"Raise max_bytes to at least {complete_bytes} to fund the "
                    "complete conservative byte envelope."
                )
            if order == 2 and variant == "exact_census":
                message += ' bound_variant="relaxed_prefix" selects the pair-only order-two bound.'
            raise ValueError(message)
        # A chosen block above the kernel default 65536 is replaced by 65536
        # when its envelope fits. An admitted call then returns the W_up of
        # the unadmitted kernel call, as it also does when max(1, E, F) <=
        # 65536 and one chunk covers every pair and triple.
        block = fit[0]
        if block > 65536 and census_bytes(
            p, q, E, F, 65536, order=chosen_order,
            variant=chosen_variant, held=census_held,
        ) <= max_bytes:
            block = 65536
        return work, block

    # The order-one envelope bounds masks/pairs before the callback.
    # It also reserves a contraction that has not yet run, conservatively.
    admit(P, 0, 1, "exact_census", "pair")
    terms = tuple(_validated_terms(hamiltonian))

    def choose(E, N):
        return variant, admit(E, N, order, variant, "requested expression")[1]

    return _pauli_bound_coefficient_from_terms(terms, order, choose=choose)


def trotter_bound_coefficient(
    hamiltonian: SparsePauliOp,
    *,
    order: int = 1,
    bound_variant: str = "exact_census",
    max_work: int = DEFAULT_CENSUS_MAX_WORK,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> float:
    """Return the time-independent Trotter bound prefactor ``W_up``.

    ``W_up`` is an outward upper bound on the Pauli-triangle expression
    selected by ``bound_variant``, which bounds the commutator sum of [CSTWZ]
    doi:10.1103/PhysRevX.11.011020, Prop. 9 (``order=1``) or Prop. 10
    (``order=2``), so one product-formula
    application obeys ``bound(t) = W_up * t**(order + 1)`` and the Sec. V B
    ``r``-step total is ``W_up * t**(order + 1) / r**order``. Exposed for
    direct evaluation without dense matrices. W_up can exceed the exact
    triangle coefficient, and it is returned rounded upward to binary64
    (``_upward_float``).
    The sparse Pauli census checks its work and byte allowances
    (``max_work``, ``max_bytes``) before converting labels or constructing
    its index tables, and a requested expression that does not fit raises
    before the unadmitted stage (``_bound_coefficient_evaluation``).

    Raises:
        ValueError: W_up exceeds the largest binary64 number, a limit is not
            a positive integer, or the requested census does not fit
            ``max_work`` and ``max_bytes``.
    """

    coefficient = _upward_float(
        _bound_coefficient_evaluation(
            hamiltonian, order=order, bound_variant=bound_variant, max_work=max_work, max_bytes=max_bytes,
        ).coefficient
    )
    if coefficient == inf:
        raise ValueError(f"the order-{order} Trotter bound coefficient exceeds the binary64 range")
    return coefficient


def dense_trotter_bound_coefficient(
    hamiltonian: SparsePauliOp,
    *,
    order: int = 1,
) -> float:
    """Return the dense tail-sum coefficient of Props. 9-10 for explicit validation.

    This evaluates the spectral norms in Eqs. (120)-(121) from dense matrices,
    so it is exponential in the qubit count. The production coefficient is
    the Pauli-triangle upper bound, which is never smaller.
    """

    return _dense_bound_coefficient_from_terms(
        tuple(_validated_terms(hamiltonian)), _validate_order(order)
    )


def _finite_bound(coefficient, time: float, order: int, steps: int = 1) -> float:
    """Return ``coefficient * time**(order + 1) / steps**order`` rounded upward to binary64.

    ``coefficient`` is W_up as a rational or a nonnegative binary64 value.
    The product and quotient are exact rationals, so a positive bound never
    rounds to zero, and one above the largest binary64 number raises the
    module's ValueError.
    """

    bound = _upward_float(Fraction(coefficient) * Fraction(time) ** (order + 1) / steps**order)
    if bound == inf:
        raise ValueError(
            f"the order-{order} Trotter bound overflows at evolution time "
            f"{time!r}; split the evolution into shorter segments before "
            "selecting step counts"
        )
    return bound


def trotter_error_bound(
    hamiltonian: SparsePauliOp,
    *,
    time: float,
    order: int = 1,
    max_work: int = DEFAULT_CENSUS_MAX_WORK,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> float:
    """Additive Trotter error bound for one product-formula application.

    Evaluates ``||S_order(time) - exp(-i * time * H)||`` bounds in spectral
    norm: [CSTWZ] doi:10.1103/PhysRevX.11.011020, Prop. 9, Eq. (120) at
    ``order=1`` (Lie-Trotter) and
    Prop. 10, Eq. (121) at ``order=2`` (Suzuki), with the commutator norms
    upper-bounded by the outward full Pauli-triangle coefficient. The summand
    order is ``hamiltonian``'s term order.
    At order two, ``evaluate_trotter_bound(..., steps=1,
    bound_variant="relaxed_prefix")`` gives the relaxed bound.
    The sparse Pauli census checks its work and byte allowances
    (``max_work``, ``max_bytes``) before converting labels or constructing
    its index tables, and a requested expression that does not fit raises
    before the unadmitted stage (``_bound_coefficient_evaluation``).
    """

    return evaluate_trotter_bound(
        hamiltonian,
        time=time,
        steps=1,
        order=order,
        max_work=max_work,
        max_bytes=max_bytes,
    )


def evaluate_trotter_bound(
    hamiltonian: SparsePauliOp,
    *,
    time: float,
    steps: int,
    order: int = 1,
    bound_variant: str = "exact_census",
    max_work: int = DEFAULT_CENSUS_MAX_WORK,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> float:
    """Evaluate the total product-formula bound at a fixed step count.

    For ``r=steps``, the [CSTWZ] (doi:10.1103/PhysRevX.11.011020)
    triangle-inequality composition gives
    ``r * bound(time / r) = W_up * time**(order + 1) / r**order`` with the
    outward coefficient W_up of the ``bound_variant`` expression. At fixed
    steps the relaxed expression changes no step count, only a possibly
    larger certified bound.
    The sparse Pauli census checks its work and byte allowances
    (``max_work``, ``max_bytes``) before converting labels or constructing
    its index tables, and a requested expression that does not fit raises
    before the unadmitted stage (``_bound_coefficient_evaluation``).
    """

    order = _validate_order(order)
    time = _validate_time(time)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError(f"steps must be a positive integer; got {steps!r}")
    evaluation = _bound_coefficient_evaluation(
        hamiltonian, order=order, bound_variant=bound_variant, max_work=max_work, max_bytes=max_bytes,
    )
    return _finite_bound(evaluation.coefficient, time, order, int(steps))


def select_trotter_step_count(
    hamiltonian: SparsePauliOp,
    *,
    time: float,
    error_budget: float,
    order: int = 1,
    bound_variant: str = "exact_census",
    max_work: int = DEFAULT_CENSUS_MAX_WORK,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TrotterStepSelection:
    """Select the minimal step count whose total bound meets the budget.

    Applies the rule of [CSTWZ] doi:10.1103/PhysRevX.11.011020, Sec. V B:
    with ``r`` steps the total additive
    error is at most ``r * bound(time / r)`` (triangle inequality over the
    unitary steps), and the smallest admissible ``r`` follows in closed
    form because ``bound(t) = W_up * t**(order + 1)``:

    - order 1: total ``= W_up * t^2 / r``, so
      ``r = max(1, ceil(t^2 * W_up / epsilon))``;
    - order 2: total ``= W_up * t^3 / r^2``, so
      ``r = max(1, ceil(sqrt(ceil(t^3 * W_up / epsilon))))``.

    W_up is the outward upper bound of the ``bound_variant`` expression
    (full ``exact_census`` by default, ``relaxed_prefix`` on request), so
    the count is minimal for W_up, not for the exact triangle coefficient.
    With the relaxed expression at order 2,
    ``r_exact <= r_rel <= ceil(alpha * r_exact)`` for
    ``alpha = sqrt(W_2,rel / W_2)`` in exact arithmetic.
    The sparse Pauli census checks its work and byte allowances
    (``max_work``, ``max_bytes``) before converting labels or constructing
    its index tables, and a requested expression that does not fit raises
    before the unadmitted stage (``_bound_coefficient_evaluation``).

    Returns a :class:`TrotterStepSelection` whose ``bound_value`` is the
    certified total bound at the selected step count (``<= error_budget``)
    and whose ``synthesis_method`` names the delegated Qiskit synthesis
    (``"lie_trotter"`` or ``"suzuki_trotter"``).
    """

    order = _validate_order(order)
    time = _validate_time(time)
    return _selection_from_evaluation(
        _bound_coefficient_evaluation(
            hamiltonian, order=order, bound_variant=bound_variant, max_work=max_work, max_bytes=max_bytes,
        ),
        time=time,
        error_budget=error_budget,
        order=order,
    )


def _smallest_step_count(one_step_bound: Fraction, budget: float, order: int) -> int:
    """Return the smallest r >= 1 with ``one_step_bound / r**order <= budget``.

    ``one_step_bound`` is an exact nonnegative rational (the bound of one
    application over the whole time, formed exactly from the supplied
    coefficient) and ``budget`` a
    positive binary64 value. Let n = ceil(one_step_bound / budget). Order 1 needs r >= n, so
    r = max(1, n). Order 2 needs r**2 >= one_step_bound / budget, and since
    r**2 is an integer this is r**2 >= n, so r = ceil(sqrt(n)) =
    1 + isqrt(n - 1) for n >= 1, and r = 1 for n = 0. The ratio is exact, so
    rounded division or sqrt cannot move r across an integer boundary, and
    incrementing a huge integer needs no float comparison. Minimality refers
    to the supplied bound, not to the actual product-formula error. Used by
    ``_selection_from_evaluation`` and by the periodic LCHS Strang selection
    (``algorithms/lchs/periodic.select_periodic_parameters``).
    """
    required_power = ceil(one_step_bound / Fraction(budget))
    if order == 1:
        return max(1, required_power)
    return 1 + isqrt(max(0, required_power - 1))


def _selection_from_evaluation(
    evaluation: _BoundCoefficientEvaluation,
    *,
    time: float,
    error_budget: float,
    order: int,
) -> TrotterStepSelection:
    """Return the smallest step count whose total bound fits the budget.

    With one-application bound B = W_up * t**(order + 1), formed exactly
    from the rational W_up, r steps give B/r**order, and
    ``_smallest_step_count`` inverts that
    relation exactly. The recorded bound B/r**order is at most the budget in
    exact arithmetic and is rounded upward (``_upward_float``). Since the
    budget is itself a binary64 number, the least binary64 number above
    B/r**order is still at most the budget, and it is positive whenever B is.
    The selection records the evaluation's variant, arithmetic and checks;
    a caller that reuses one evaluation passes zero checks for later
    selections.
    """
    budget = float(error_budget)
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError(
            f"error budget must be positive and finite; got {error_budget!r} "
            "(no finite step count meets a nonpositive additive error budget)"
        )

    exact_bound = evaluation.coefficient * Fraction(time) ** (order + 1)
    step_count = _smallest_step_count(exact_bound, budget, order)
    bound_value = _upward_float(exact_bound / step_count**order)
    return TrotterStepSelection(
        formula_order=order,
        evolution_time=time,
        error_budget=budget,
        bound_value=bound_value,
        step_count=step_count,
        pauli_term_count=evaluation.pauli_term_count,
        pair_commutation_checks=evaluation.pair_commutation_checks,
        nested_commutation_checks=evaluation.nested_commutation_checks,
        bound_variant=evaluation.bound_variant,
        coefficient_arithmetic=evaluation.coefficient_arithmetic,
    )


def common_steps(coefficient, tau, powers, budgets, dropped_mass=0):
    """Return the ideal rational common step h and counts r(p) for nonnegative integer powers.

    Let ``H'=c_I I+sum_j c_jP_j`` be the one shared pruned Hamiltonian with
    one ordered term list, and d a bound on the dropped coefficient L1 mass.
    The second-order step S_2(h) has a sufficient one-step error W|h|^3 by
    Childs et al., arXiv:1912.08854v3, Proposition 16, Eq. (152), journal
    DOI 10.1103/PhysRevX.11.011020, Proposition 10, Eq. (121). For time
    ``t_p=p*tau`` and integer ``r(p)=t_p/h``, telescoping the product of
    unitaries gives ``||S_2(h)^r(p) - exp(-i t_p (H'-c_I I))|| <= r(p) W |h|^3
    = W h^2 t_p = W t_p^3/r(p)^2``, and pruning gives the total sufficient
    bound ``B_p=(d+Wh^2)t_p``. With varying allowances the condition is
    ``d+Wh^2 <= min_(p>0) epsilon_p/t_p``: after pruning,
    ``Wh^2 t_p <= epsilon_p-d*t_p`` for every power.

    Let g be the gcd of the positive requested integer powers. Every boundary
    lies on a common grid if ``h=g*tau/m`` with integer ``m>=1``, and then
    ``r(p)=(p/g)*m``. The smallest integer m >= 1 satisfying
    ``m^2 >= max_(p>0) W(g tau)^2 t_p/(epsilon_p-d t_p)`` is chosen, with a
    positive denominator whenever W > 0. W = 0 accepts a zero remaining
    allowance, and a negative remaining allowance rejects. For W = 1,
    powers 1 and 3, tau = 1 and allowance 6/5 this gives h = 1/2,
    r(1) = 2, r(3) = 6 and bounds 1/4 and 3/4. The exact-rational kernel
    avoids a floating square root at an integer boundary and returns a
    rational ideal step, not a floating-point enclosure of gate
    construction; ``select_common_step`` rechecks the emitted step. Reusing
    one numerical trajectory is a consequence of selecting the same step,
    not an additional result claimed by that paper.
    """
    ps = tuple(sorted(set(powers)))
    if any(type(p) is not int or p < 0 for p in ps):
        raise ValueError("use nonnegative integer powers")
    w, dt, loss = map(Fraction, (coefficient, tau, dropped_mass))
    if w < 0 or dt <= 0 or loss < 0:
        raise ValueError("require W >= 0, tau > 0 and dropped mass >= 0")
    positive = tuple(p for p in ps if p)
    if not positive:
        return Fraction(0), {p: 0 for p in ps}
    g = reduce(gcd, positive)
    required_square = Fraction(0)
    for p in positive:
        time = p * dt
        remaining = Fraction(budgets[p]) - time * loss
        if remaining < 0 or remaining == 0 and w:
            raise ValueError("pruning exhausts a power's allowance")
        if w:
            required_square = max(
                required_square, w * (g * dt)**2 * time / remaining)
    m = max(1, isqrt(required_square.numerator // required_square.denominator))
    if m * m < required_square:
        m += 1
    return g * dt / m, {p: p // g * m for p in ps}


def _common_reductions(coefficients, h, half_angles):
    """Return exact kept mass and angle error of one forward/reverse step."""
    c = tuple(Fraction(float(value)) for value in coefficients)
    mass = sum(map(abs, c), Fraction(0))
    angle_per_step = 2 * sum(
        (abs(Fraction(float(angle)) - h*value/2)
         for angle, value in zip(half_angles, c, strict=True)), Fraction(0))
    return mass, angle_per_step


def _emitted_parts(W, loss, mass, angle_per_step, phase, t, h, identity, steps):
    """Compose the five exact nonnegative parts for one emitted prefix."""
    parts = dict(trotter=W*steps*abs(h)**3, pruning=loss*abs(t),
                 time=mass*abs(steps*h-t), identity=abs(phase+t*identity),
                 angles=steps*angle_per_step)
    if any(value < 0 for value in parts.values()):
        raise ValueError("error contributions must be nonnegative")
    return sum(parts.values(), Fraction(0)), parts


def emitted_subtotal(W, dropped_mass, coefficients, identity, tau, power, steps,
                     step_time, half_angles, identity_phase_increments):
    """Return the exact structural subtotal B_p of one emitted prefix and its parts.

    All times and coefficients denote exact values of their stored binary64
    numbers. For a nonnegative power p, t_p = p*val(tau), r = ``steps``,
    h = val(h_hat), kept generator A = sum_j c_j P_j (the kept nonidentity
    terms after pruning, with their actual selected coefficients) and exact
    identity coefficient c_I:

        B_p = W r abs(h)^3 + d_drop abs(t_p) + C_kept abs(r h - t_p)
              + B_identity,p + B_angles,p,
        C_kept = sum_j abs(c_j),
        B_identity,p = abs(Phi_p + t_p*c_I),
        B_angles,p = 2r sum_j abs(val(a_hat_j) - h*c_j/2).

    The product-formula term is Childs et al., arXiv:1912.08854v3,
    Proposition 16, Eq. (152), DOI 10.1103/PhysRevX.11.011020. Each
    canonical Pauli has norm one, so ||A|| <= C_kept, and the spectral
    theorem gives ||exp(-isA)-exp(-itA)|| <= C_kept |s-t| for the time
    displacement. Phi_p is the actual prefix sum of the emitted identity
    phases; P(phi) = diag(1, exp(i phi)) differs from P(-t c_I) by at most
    abs(phi + t c_I), and the identity control phase commutes with the
    controlled Pauli evolution. a_hat_j are the actual half-step
    PauliEvolutionGate times of the leaves exp(-i a_hat_j P_j), not doubled
    RZ parameters; integrating the derivative of exp(-iaP) bounds a leaf
    difference by abs(a-b), telescoping adds the leaf discrepancies, r
    repetitions multiply the sum by r, and the factor 2 counts forward and
    reverse occurrences even when they share a constructed gate. The leaf
    reference uses h, not t_p/r, since the time term already prices their
    difference. These contributions are not generally zero when formed
    from the same exact target times.

    The shared-step time correction is the coefficient L1 mass of the kept
    nonidentity generator multiplied by the exact discrepancy between the
    cumulative represented step time and the selected target time. Both
    quantities are evaluated without a downward floating-point reduction.
    This correction joins the product-formula and pruning terms before
    per-power admission. Identity-phase and rotation-angle formation errors
    use their own contributions. Identity error compares the cumulative
    emitted control phase with the exact target identity phase. Angle error
    sums each emitted Pauli-evolution parameter's discrepancy from its exact
    product at the represented common step time, with its actual occurrence
    count.

    This is for the common nonnegative-power convention, order two and an
    actually repeated shared half schedule; the caller validates those
    construction invariants, and publishes B_p upward only after composing
    it (``_upward_float``).
    """
    if (type(power) is not int or power < 0 or type(steps) is not int
            or steps < 0 or len(coefficients) != len(half_angles)):
        raise ValueError("invalid emitted prefix")
    vals = [tau, step_time, identity, *coefficients, *half_angles]
    if not all(isfinite(float(v)) for v in vals) or tau <= 0:
        raise ValueError("parameters must be finite and tau positive")
    t = power * Fraction(float(tau))
    h = Fraction(float(step_time))
    mass, angle_per_step = _common_reductions(coefficients, h, half_angles)
    phase = Fraction(0)
    for value in identity_phase_increments:
        if not isfinite(float(value)):
            raise ValueError("phase parameters must be finite")
        phase += Fraction(float(value))
    return _emitted_parts(
        Fraction(W), Fraction(dropped_mass), mass, angle_per_step, phase,
        t, h, Fraction(float(identity)), steps,
    )



def select_common_step(
    evaluation: _BoundCoefficientEvaluation,
    *,
    coefficients,
    identity: float,
    tau: float,
    allowances,
    dropped_mass: float,
    max_steps: int,
) -> TrotterStepSelection | None:
    """Select one ordered second-order step for the nonnegative integer power positions.

    The ideal rational grid uses ``t_p=p*val(tau)`` and integer prefix
    counts. Persist the emitted binary64 step and the cumulative count at
    every point. Admit each prefix using its pruning bound,
    ``W*r_p*abs(val(h_hat))**3``, kept-generator time-displacement bound and
    applicable identity-phase and rotation-angle formation bounds. The
    counts are sufficient for the shared trajectory and need not minimize
    each prefix independently.

    Rule. Validate finite positive tau, nonnegative W and dropped mass,
    nonnegative integer powers and finite nonnegative per-power allowances
    (``allowances`` maps each power to its allowance). ``tau``,
    ``identity``, the coefficients and the allowances are binary64 values
    whose exact represented values the recheck uses, and a value that
    would round on conversion is refused. Run the ideal
    rational common-grid selector ``common_steps`` once and admit its
    largest count against ``max_steps``. Form the actual binary64 step
    ``h_hat = float(h)``, the half-step leaf times ``0.5 * h_hat * c_j``
    (the formation of ``algorithms/qpe/powers.rotation_schedule``) and one
    identity phase increment ``-(p - p_prev) * tau * identity`` per point in
    increasing power order. Compute the exact kept mass and one-step leaf
    discrepancy once, and accumulate the exact phase prefix once per point.
    Compose the five parts of ``emitted_subtotal`` at each prefix from these
    shared reductions. The recheck certifies exactly
    these parameters, so an emitter applies the leaf times that
    ``rotation_schedule`` forms from h_hat and these identity increments
    between adjacent points; other emitted values need their own recheck.
    Accept only if each exact
    ``B_p <= val(epsilon_p)``, then publish each B_p upward. A failure
    refuses with the power, target allowance and offending contribution
    values; no retry is made.

    Branches. No positive powers: no evolution, no phase increments,
    r(0) = 0 and zero subtotal after validating the zero-power allowance,
    and no common selection (None). Pruning alone above any allowance
    refuses. Pruning equal to the allowance refuses for W > 0 at a positive
    time; for W = 0 the emitted phase, angle and time terms must still fit.
    An empty kept generator has zero time-displacement and angle terms, and
    only pruning and identity-phase representation remain. A positive
    subnormal h_hat is checked with its actual represented angles, so a
    product 0.5*h_hat lost to rounding is counted in B_angles,p. An h_hat
    that rounds to zero for a positive-time nonempty step refuses before
    emission. Nonfinite phases or angles refuse. W (``evaluation``) must
    bound the selected ordered kept generator's triangle coefficient, and
    ``dropped_mass`` must be the exact sum of discarded magnitudes or an
    outward upper bound on it; a previously rounded-down mass does not meet
    that premise.

    Accept a common-step construction only after every emitted-prefix
    subtotal fits its allowance. A zero represented step for positive-time
    evolution is refused, and a subnormal step is checked with its actual
    represented angles. Any refinement policy has finite candidate and
    resource limits.

    The returned selection carries the evaluation's census counts; a caller
    that already recorded them passes an evaluation with zero checks.
    """
    return _select_common_step(
        evaluation, coefficients=coefficients, identity=identity, tau=tau,
        allowances=allowances, dropped_mass=dropped_mass, max_steps=max_steps)[0]


def _select_common_step(evaluation, *, coefficients, identity, tau, allowances, dropped_mass,
                        max_steps):
    """Return (``select_common_step``'s selection, the exact prefix subtotals, the certified increments).

    The second value maps every requested power to ``emitted_subtotal``'s
    exact (total, parts) of its accepted emitted prefix, and the third maps
    it to the binary64 identity phase increment the recheck certified before
    that power position. Both are empty when no positive power exists. A
    caller publishes the components of each prefix and emits exactly these
    increments without evaluating them again.
    """
    coefficients = tuple(_binary64(value, "each coefficient") for value in coefficients)
    if len(coefficients) != evaluation.pauli_term_count:
        raise ValueError("coefficients must be the evaluation's kept terms")
    identity = _binary64(identity, "identity")
    tau = _binary64(tau, "tau")
    if tau <= 0:
        raise ValueError("tau must be finite and positive")
    # The dropped mass is an exact value or an outward upper bound, so any real
    # rational within the binary64 range is kept exactly; a complex, string or
    # out-of-range value is refused like the other intake values.
    try:
        if isinstance(dropped_mass, (complex, np.complexfloating, str, bytes)):
            raise TypeError
        exact_mass = (Fraction(*dropped_mass.as_integer_ratio())
                      if isinstance(dropped_mass, np.floating) else Fraction(dropped_mass))
        admitted = exact_mass >= 0 and isfinite(float(exact_mass))
    except (TypeError, ValueError, OverflowError):
        admitted = False
    if not admitted:
        raise ValueError("dropped mass must be finite and nonnegative")
    dropped_mass = exact_mass
    if evaluation.coefficient < 0:
        raise ValueError("W must be nonnegative")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    powers = tuple(sorted(allowances))
    if any(type(p) is not int or p < 0 for p in powers):
        raise ValueError("use nonnegative integer powers")
    allowances = {p: _binary64(allowances[p], "each per-power allowance") for p in powers}
    if any(allowances[p] < 0 for p in powers):
        raise ValueError("per-power allowances must be finite and nonnegative")
    positive = tuple(p for p in powers if p)
    if not positive:
        return None, {}, {}
    h, counts = common_steps(evaluation.coefficient, tau, powers, allowances, dropped_mass)
    g = reduce(gcd, positive)
    p_max = positive[-1]
    m = counts[p_max] // (p_max // g)
    if counts[p_max] > max_steps:
        raise ValueError(
            f"common step count {counts[p_max]} at power {p_max} exceeds max_steps={max_steps}"
        )
    # float(Fraction) raises OverflowError instead of returning inf.
    if h > Fraction(float_info.max):
        raise ValueError("the common step is not a finite binary64 number")
    h_hat = float(h)
    if h_hat == 0 and coefficients:
        raise ValueError("the common step rounds to zero for positive-time evolution")
    half_angles = tuple(0.5 * h_hat * value for value in coefficients)
    if not all(isfinite(value) for value in half_angles):
        raise ValueError("emitted phases and angles must be finite")
    dt, hi, ci = Fraction(tau), Fraction(h_hat), Fraction(identity)
    mass, angle_per_step = _common_reductions(coefficients, hi, half_angles)
    phase = Fraction(0)
    increments, previous = [], 0
    bounds = {}
    for p in powers:
        try:
            increments.append(-(p - previous) * tau * identity)
        except OverflowError:
            raise ValueError("emitted phases and angles must be finite") from None
        previous = p
        if not isfinite(increments[-1]):
            raise ValueError("emitted phases and angles must be finite")
        phase += Fraction(increments[-1])
        total, parts = _emitted_parts(
            evaluation.coefficient, dropped_mass, mass, angle_per_step, phase,
            p*dt, hi, ci, counts[p],
        )
        if total > Fraction(allowances[p]):
            raise ValueError(
                f"common step h_hat={h_hat!r} does not meet the allowance {allowances[p]!r} "
                f"at power {p}: emitted subtotal {_upward_float(total)!r} with contributions "
                + ", ".join(f"{name}={_upward_float(value)!r}" for name, value in parts.items())
            )
        bounds[p] = (total, parts)
    # Scalar allowance of the whole trajectory: the least per-power residual
    # after the non-formula contributions, rescaled to r_max steps.
    r_max = counts[p_max]
    residual = min(
        (Fraction(allowances[p]) - (bounds[p][0] - bounds[p][1]["trotter"])) * r_max / counts[p]
        for p in positive
    )
    if residual < 0:
        raise ValueError("the non-formula contributions exceed a power's allowance")
    target = p_max * Fraction(tau)
    if target > Fraction(float_info.max):
        raise ValueError("the largest target time exceeds the binary64 range")
    evolution_time = float(target)
    published = {p: _upward_float(bounds[p][0]) for p in powers}
    if inf in published.values():
        raise ValueError("an emitted-prefix subtotal exceeds the binary64 range")
    return TrotterStepSelection(
        formula_order=2,
        evolution_time=evolution_time,
        error_budget=_downward_float(residual),
        bound_value=_upward_float(evaluation.coefficient * r_max * Fraction(h_hat) ** 3),
        step_count=r_max,
        pauli_term_count=evaluation.pauli_term_count,
        pair_commutation_checks=evaluation.pair_commutation_checks,
        nested_commutation_checks=evaluation.nested_commutation_checks,
        bound_variant=evaluation.bound_variant,
        coefficient_arithmetic=evaluation.coefficient_arithmetic,
        common_tau=float(tau),
        common_g=g,
        common_m=m,
        common_step_time=h_hat,
        common_prefix_steps=tuple((p, counts[p]) for p in powers),
        common_power_allowances=tuple((p, float(allowances[p])) for p in powers),
        common_recheck_outcome="accepted",
        common_prefix_bounds=tuple(published.items()),
    ), bounds, dict(zip(powers, increments, strict=True))


def build_trotter_evolution_circuit(
    hamiltonian: SparsePauliOp,
    selection: TrotterStepSelection,
) -> QuantumCircuit:
    """Build the budgeted evolution circuit by delegating the synthesis.

    Validates ``hamiltonian`` (the operator ``selection`` was computed for)
    and delegates to ``build_sparse_pauli_product_circuit`` with the
    selection's synthesis method and ``reps=step_count``, which reaches
    Qiskit's product-formula synthesis through ``make_evolution_synthesis``.
    The returned circuit realizes
    ``S_order(time / step_count)**step_count`` with the summands applied in
    term order, matching the ordering the recorded bounds are stated for.
    """
    from nwqlib.subroutines.hamiltonian_evolution.sparse_pauli_product import (
        build_sparse_pauli_product_circuit,
    )

    return build_sparse_pauli_product_circuit(
        _validated_terms(hamiltonian),
        num_qubits=int(hamiltonian.num_qubits),
        time_step=selection.evolution_time,
        evolution_synthesis=selection.synthesis_method,
        reps=selection.step_count,
        order=2 if selection.formula_order == 2 else None,
    )


__all__ = [
    "TROTTER_BOUND_REFERENCE",
    "TrotterStepSelection",
    "build_trotter_evolution_circuit",
    "dense_trotter_bound_coefficient",
    "evaluate_trotter_bound",
    "select_trotter_step_count",
    "trotter_bound_coefficient",
    "trotter_error_bound",
]
