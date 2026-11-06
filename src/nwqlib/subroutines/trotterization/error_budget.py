"""Deterministic Trotter error budgets with commutator scaling.

Implements the tight low-order product-formula error bounds of Childs, Su,
Tran, Wiebe, and Zhu, "Theory of Trotter error with commutator scaling",
Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020,
arXiv:1912.08854v3 [CSTWZ]. Unless marked as arXiv:1912.08854v3, every
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

The commutator sums used outside dense validation take Pauli commutation
relations and coefficient magnitudes. Norm subadditivity plus
``||[P, Q]|| = 2`` and ``||[P, [Q, R]]|| = 4`` for nonzero Pauli
commutators makes these triangle expansions certified upper bounds on the
tail-sum spectral norms in Props. 9-10. At order 2 the full triangle
expression ("exact_census") or its degree-capped suffix relaxation
("relaxed_prefix"), which needs pair tests only, can be requested, and
"auto" takes the full expression when its selection count N(A) max(w, K)
is at most EXACT_CENSUS_WORD_TESTS (``choose_variant``). N(A) is the
number of nested tests, w the mask words per test and K the number of
weightings contracted with the same masks, so the count is the larger of
the w N(A) nested word tests and the K N(A) weight products.

Outward evaluation. The Pauli-triangle coefficient is evaluated with scaled
binary64 upper products and sums, followed by rational rescaling. Its value
W_up bounds the chosen triangle expression. For formula order o in {1,2},
time multiplication, division by r**o and integer step inversion use exact
rationals formed from W_up and the stored binary64 time. Returned bounds
are rounded upward (``_upward_float``). This keeps a positive bound
below the binary64 range positive and reports a range error above it. The
chosen count is minimal for W_up, which can exceed the exact triangle
coefficient. The record names the full or relaxed expression. Pruning uses
its share of the allowance before the step count is chosen, and a zero
remaining allowance accepts only a zero coefficient. The function
``dense_trotter_bound_coefficient`` evaluates the tail-sum
norms from dense ``2**n`` by ``2**n`` matrices, for small-instance
validation only.

Commutation census. The census, called the Pauli commutation table in the
user documentation, of an ordered Pauli term list is its rows of
anticommuting pairs (``pair_rows``). The full second-order expression
forms, for each pair, the mask of the later terms that anticommute with
the pair product and contracts it with the magnitudes, a tile of
_PAIR_TILE pairs at a time, so only one tile of masks is held
(``coefficient_up``). The same masks serve every weighting of one label
table, such as the LCHS quadrature nodes. The expression is chosen from
counts alone (``census_structure``), so memory never changes the
expression or the arithmetic grouping. ``census_work`` counts the work for
an explicit work limit, and ``census_bytes`` gives a coarse estimate of the
dominant arrays, which are checked against the memory limit before the
contraction. Its nested-mask term charges the largest tile that the pair
rows can form, zero when no pair anticommutes.

Common steps. ``common_steps`` chooses an ideal rational step whose
integer repetitions reach the target times ``p*val(tau)`` for nonnegative
integer powers ``p``, where ``val(tau)`` is the exact value of the stored
binary64 time unit. The emitted step, angles and phases are the binary64
values that the circuit uses. ``select_common_step`` accepts the
corresponding shared ordered second-order step only after the subtotal
B_p defined by ``_emitted_parts`` fits every requested prefix's allowance.
This subtotal includes product-formula, pruning, time-displacement,
identity-phase and rotation-angle bounds evaluated with the emitted
binary64 step, angles and phases. A first count that misses it by rounding
gets a bounded correction on the same grid (``_common_envelope_count``).

Circuit construction delegates to the ``hamiltonian_evolution`` helpers,
``build_sparse_pauli_product_circuit`` over ``build_pauli_evolution_circuit``,
which apply the summands in the term order that the bounds assume
(``build_trotter_evolution_circuit``).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import ceil, frexp, fsum, gcd, inf, isfinite, isqrt, nextafter
from sys import float_info
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

import numpy as np

from nwqlib._limits import ResourceLimitError, check_memory, exceeds


# These strings leave the module in TrotterStepSelection.to_dict output and in
# the recorded LCHS budget terms, so each one names the paper version whose
# numbering it uses.
TROTTER_BOUND_REFERENCE = (
    "Childs, Su, Tran, Wiebe, and Zhu, Theory of Trotter error with "
    "commutator scaling, Phys. Rev. X 11, 011020 (2021), "
    "doi:10.1103/PhysRevX.11.011020, arXiv:1912.08854v3. "
    "Proposition, equation and section numbers follow Phys. Rev. X"
)
"""Citation text of the Trotter error bounds, Childs et al., Phys. Rev. X 11, 011020 (2021), doi:10.1103/PhysRevX.11.011020, with the note that cited proposition, equation and section numbers follow that journal version."""

BOUND_VARIANTS = ("exact_census", "relaxed_prefix")
# "auto" chooses the exact second-order census when N(A)*max(w, K) is at
# most this count, and the relaxed one above it. With one weighting (K = 1)
# the count is the w*N(A) logical nested word tests. K weightings contracted
# with the same masks also make K*N(A) weight products, and the count keeps
# the nested work w*N(A) + K*N(A) within twice this value, the work that one
# weighting at w = 1 is allowed. The weight products are counted like word
# tests by choice, without a measured cost ratio. The threshold sits
# just above the 1,280,549,200 word tests of a 2500-term table, which an
# earlier triple-table implementation took 7.5 s to count (0.41 s and 1.56 s
# for 80,918,050 and 270,132,409 tests at 1000 and 1500 terms) on 20-qubit
# 2-4-local Pauli sums. A choice of expression, not a limit: both variants
# plan when their arrays fit memory.
# Registered in docs/ENGINEERING_CONSTANTS.md. Revisit when the exact
# contraction or its measured cost changes.
EXACT_CENSUS_WORD_TESTS = 1_300_000_000
# Pairs of one row whose nested masks are formed together. Fixed, so that
# memory never changes the arithmetic grouping.
_PAIR_TILE = 32

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
        "W_up * t^3 / r^2 with W_up >= W_2,rel = sum_{(i,j) in A} |c_i*c_j|*U_ij / 3 "
        "+ sum_{(i,j) in A} c_i^2*|c_j| / 6, U_ij = min(S_i, D_i + D_j - |c_i|), "
        "S_i = sum_{k > i} |c_k|, D_v = sum_{k: P_v*P_k = -P_k*P_v} |c_k|, "
        "A = {(i,j): i < j, P_i*P_j = -P_j*P_i}; the nested mass is at most S_i and "
        "D_i + D_j - |c_i| by symplectic XOR <= the sum of endpoint parities, so "
        f"W_2 <= W_2,rel bounds {_SECOND_ORDER_TAIL} {_PROP10}"
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
    """The step count chosen for one product-formula error budget, with its error bound.

    [`select_trotter_step_count`][nwqlib.subroutines.trotterization.select_trotter_step_count]
    returns it. Its `step_count` field holds the result, and `bound_value` is
    its certified error bound. The fields below are read-only.

    The fields follow the quantities of Childs, Su, Tran, Wiebe and Zhu
    (doi:10.1103/PhysRevX.11.011020). ``bound_value`` is the certified
    total additive Trotter error bound ``step_count * bound(evolution_time /
    step_count)`` at the chosen step count (at most ``error_budget``), and
    the properties ``bound_formula`` and ``step_formula`` give the
    per-application bound and the smallest-r arithmetic as text with their
    sources.

    The Pauli-triangle coefficient used for the bound is evaluated with
    scaled binary64 products and sums rounded upward, followed by exact
    rational rescaling. ``bound_variant`` identifies the expression that
    this coefficient bounds.

    A common selection, made by ``select_common_step`` for QPE powers on a
    common time grid, uses one ordered product-formula step throughout
    the trajectory. At each requested nonnegative integer power ``p``, the
    target time is ``p*val(common_tau)``, where ``val`` denotes the exact
    real value of a stored binary64 number. Its circuit prefix consists of
    the first ``r(p)`` steps, with ``r(p)`` recorded in
    ``common_prefix_steps``. ``select_common_step`` accepts the selection
    only after the product-formula, pruning, time-displacement,
    identity-phase and rotation-angle bounds for every prefix fit that
    power's allowance, evaluated with the binary64 step and angles that the
    circuit uses. The ``common_*`` fields are None ("not recorded") on a
    selection for one evolution time. The maximum target time
    ``p_max*val(common_tau)``, where ``p_max`` is the largest requested
    power, and each represented prefix time ``r(p)*val(common_step_time)``
    are computed from the fields and not recorded. ``select_common_step``
    sets ``common_g`` to the greatest common divisor g of the positive
    powers and ``common_m`` to the positive subdivision count m, so that
    ``r(p) = (p/g)*m``. The all-zero schedule has no g, m or step and no
    common selection.

    Attributes:
        formula_order: Lie-Trotter order 1 or symmetric Suzuki order 2.
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
            contributions are subtracted at every requested power. At each
            positive requested power ``p``, let ``epsilon_p`` be its
            allowance in ``common_power_allowances``, and let ``P_p``,
            ``T_p``, ``I_p`` and ``A_p`` be its pruning, time-displacement,
            identity-phase and rotation-angle error bounds, respectively.
            Let ``r_p`` be its cumulative step count in
            ``common_prefix_steps`` and ``r_max`` the largest of those
            counts. The allowance is
            ``min_(p>0) (epsilon_p-P_p-T_p-I_p-A_p)*r_max/r_p`` rounded
            downward to binary64.
        bound_value: Upward-rounded total operator-error bound at
            step_count, computed from the supplied upper coefficient and
            evolution_time. On a common selection: upward-rounded
            product-formula bound ``W_up*r_max*abs(val(h_hat))**3`` for the
            common step that the circuit uses.
        step_count: Smallest positive integer allowed by the supplied upper
            coefficient and remaining error_budget. On a common selection:
            number r(p_max) of shared steps through the last requested
            power. This is a sufficient common-grid count and need not be
            the independently smallest count for that power.
        pauli_term_count: Number of nonidentity terms kept, after pruning, in
            this Hamiltonian or in the generator of this LCHS quadrature
            node.
        pair_commutation_checks: Number of new pair commutation tests that
            this selection performed, zero when the coefficient or the Pauli
            commutation table (its anticommuting pairs) was reused.
        nested_commutation_checks: Number of new nested commutation tests
            that this selection performed, zero for relaxed_prefix or a
            reused Pauli commutation table.
        bound_variant: Pauli-triangle expression used, "exact_census" for
            the full expression over the anticommuting pairs and, at second
            order, every later term anticommuting with each pair product, or
            "relaxed_prefix" for the second-order degree-capped suffix
            relaxation. Order 1 uses "exact_census".
        common_tau: Finite positive binary64 time unit. The exact target
            time at integer power p is p times its represented value.
        common_g: Greatest common divisor of the positive requested integer
            powers. It identifies the exact target-time grid.
        common_m: Positive integer subdivisions of g times the represented
            time unit. The ideal common step is g*val(tau)/m.
        common_step_time: Binary64 step h_hat that the circuit uses for
            every shared step. Its represented value can differ from the
            ideal rational step.
        common_prefix_steps: Integer cumulative counts r(p)=(p/g)*m at each
            requested power, with r(0)=0, as ``(power, count)`` pairs.
            Execution between adjacent points uses the difference of these
            counts.
        common_power_allowances: Per-power structural operator-error
            allowances against which the bounds of the circuit's step were
            rechecked, as ``(power, allowance)`` pairs. The binary64
            allowances are compared through their exact represented values.
        common_recheck_outcome: Outcome of comparing the complete subtotal
            of every circuit prefix with its allowance. An accepted common
            selection passed at every power. None means the recheck was not
            recorded.
        common_prefix_bounds: Upward-rounded values of the exact pruning,
            product-formula, time-displacement, identity-phase and
            rotation-angle subtotals for the circuit prefixes, as
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

    @property
    def bound_formula(self) -> str:
        """Coefficient expression used, with its source theorem and inequality.

        For a common selection, it names the composition of the step that
        the circuit uses.
        """
        order = 1 if self.formula_order == 1 else 2
        if self.common_recheck_outcome is not None:
            return _COMMON_BOUND_FORMULA
        return _VARIANT_BOUND_FORMULAS[order, self.bound_variant]

    @property
    def step_formula(self) -> str:
        """Integer inversion of ``W_up*abs(t)**(order+1)/r**order`` against the remaining error budget."""
        order = 1 if self.formula_order == 1 else 2
        if self.common_recheck_outcome is not None:
            return _COMMON_STEP_FORMULA
        return _VARIANT_STEP_FORMULAS[order]

    @property
    def synthesis_method(self) -> str:
        """Qiskit product formula used for the circuit: `"lie_trotter"` at order 1, `"suzuki_trotter"` at order 2."""

        return _SYNTHESIS_METHODS[self.formula_order]

    @property
    def bound_method(self) -> str:
        """Name of the bound, `"pauli_triangle"`: commutator norms bounded by the Pauli triangle inequality."""

        return "pauli_triangle"

    @property
    def bound_value_status(self) -> str:
        """Meaning of `bound_value`, `"structural_upper_bound"`: an upper bound on the product-formula operator error."""

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
# allowance. Reserving the published P first avoids that. With ordered terms
# X:0.5, Z:0.5, order one, time one, epsilon the published coefficient and
# pruning 2**-80, the nearest-rounded remainder equals epsilon and accepts
# r = 1, while the exact published sum exceeds epsilon by 2**-80. The
# downward remainder selects r = 2. Since R_down <= R_near for the same W_up
# and time, the selected count cannot decrease. It can grow by more than one
# step when the count is large, and a remainder that rounds to zero refuses a
# nonzero formula bound. A mass upper bound can be reused across times. The
# pruning charge, remainder and total belong to each application.
# Source: NWQLib's composition derived for this module (triangle inequality
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


def upper_dropped_mass(dropped_coefficients):
    """Return mass_up >= sum_D |c_j| over the dropped stored complex coefficients c_j.

    D is exactly the set selected by the caller's cutoff test. For
    c = x + iy with binary64 components, |c| = sqrt(x**2 + y**2) <= |x| + |y|,
    and each |x|, |y| is exact, so the exact component sum S bounds
    d = sum_D |c_j|. S = d for real coefficients and can exceed d by at most
    a factor sqrt(2) for complex residues. The coefficients are read once,
    as a stream, and their nonzero absolute components go to math.fsum.
    With round-to-nearest, which Python does not change, fsum returns S
    rounded to nearest, so one step up with nextafter bounds S. Python
    documents that on some builds whose C library adds in extended precision
    the result can be off by one unit in the last place, so two steps up
    bound S on every build, at most about 2.5 units in the last place above
    it. A sum of no or one nonzero component is exact and is returned
    unchanged, +0.0 for no component. The final budget composition rounds
    the exact rational value upward once (``upper_pruning_error``,
    ``upper_combined_error``).

    Work, in scalar visits: at most 10 per dropped coefficient (one complex
    conversion, two finiteness checks and their combination, two absolute
    values, two zero tests and two fsum inputs, one visit each) and five per
    call (the fsum result, the count test, two steps up and their finiteness
    check), so 10*d + 5 for d dropped coefficients. Workspace is the list of
    nonoverlapping partials that fsum keeps. Its length depends on the
    spread of the binary64 exponents, not on d, and it is counted once as
    512 logical bytes. These units are proxies used to check work limits, not
    timings or equal-cost CPU operations. Source: NWQLib's derivation for
    this module (Python's math.fsum and math.nextafter), not a paper theorem.
    """
    nonzero = 0

    def components():
        nonlocal nonzero
        for coefficient in dropped_coefficients:
            c = complex(coefficient)
            if not (isfinite(c.real) and isfinite(c.imag)):
                raise ValueError("dropped coefficients must be finite")
            for part in (abs(c.real), abs(c.imag)):
                if part:
                    nonzero += 1
                    yield part

    try:
        mass = fsum(components())
    except OverflowError as error:
        raise ValueError("dropped coefficient mass enclosure exceeds binary64 range") from error
    if nonzero > 1:
        mass = nextafter(nextafter(mass, inf), inf)
        if not isfinite(mass):
            raise ValueError("dropped coefficient mass enclosure exceeds binary64 range")
    return mass


def upper_pruning_error(time, dropped_mass):
    """Return P = up(|t| * d_up), the upward-rounded pruning bound at one evolution time.

    ``time`` and ``dropped_mass`` must be finite binary64 values. Their
    exact product is rounded upward, so a positive product below the least
    subnormal becomes the least subnormal, not zero.
    """
    t = _binary64(time, "evolution time")
    d = _binary64(dropped_mass, "dropped mass upper bound")
    if d < 0:
        raise ValueError("dropped mass upper bound must be nonnegative")
    return _finite_upper(abs(Fraction(t))*Fraction(d), "pruning error")


def _remaining_budget_after_pruning(*, total_budget, pruning_error):
    """Return R = down(epsilon - P), the allowance left for the product formula.

    ``pruning_error`` is the caller's upward-rounded pruning bound P
    (``upper_pruning_error``). The exact remainder is rounded downward, so a
    formula bound rounded upward within R keeps the exact sum of the two
    rounded contributions at or below ``total_budget``. A negative
    remainder raises PruningBudgetExhausted. A zero remainder accepts only an
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
    """Return T = up(P + E), the upward-rounded total of a pruning and a formula bound.

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


def _validate_time(time: float) -> float:
    """Return ``time`` as a positive finite float. Props. 9-10 are stated for t >= 0."""
    value = float(time)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"evolution time must be positive and finite; got {time!r} "
            "(the Prop. 9-10 bounds and the step selection are stated for t > 0)"
        )
    return value


def _spectral_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, 2))


def _tail_commutator_norms(terms, order: int) -> list[tuple[float, ...]]:
    """Return, in term order, the spectral norms that Eqs. (120)-(121) sum over the terms H_gamma1 = c P.

    With the suffix sum ``T_gamma1 = sum_{gamma2 > gamma1} H_gamma2``,
    bilinearity gives ``sum_{gamma2 > gamma1} [H_gamma2, H_gamma1] =
    [T_gamma1, H_gamma1]``. Order 1 returns ``||[T_gamma1, H_gamma1]||`` of
    C1, and order 2 the pair ``||[T_gamma1, [T_gamma1, H_gamma1]]||`` of C12
    and ``||[H_gamma1, [H_gamma1, T_gamma1]]||`` of C24. The terms are
    visited last to first and each dense matrix (``4**n`` entries) is
    formed only when visited, so the term matrix, T and the commutator
    products of one term are live at a time, not every term matrix and
    every suffix sum. T starts at zero and adds each visited term, the same
    additions in the same order as a forward pairing of precomputed
    suffix sums, so the norms are bit for bit those of that pairing, and
    the callers sum them first to last.
    """
    from qiskit.quantum_info import SparsePauliOp

    norms = []
    tail = None
    for label, coefficient in reversed(terms):
        head = np.asarray(SparsePauliOp.from_list([(label, coefficient)]).to_matrix())
        if tail is None:
            tail = np.zeros_like(head)
        inner_t = tail @ head - head @ tail  # [T_gamma1, H_gamma1]
        if order == 1:
            norms.append((_spectral_norm(inner_t),))
        else:
            inner_h = -inner_t  # [H_gamma1, T_gamma1] is the exact negation
            norms.append((_spectral_norm(tail @ inner_t - inner_t @ tail),
                          _spectral_norm(head @ inner_h - inner_h @ head)))
        tail += head
    norms.reverse()
    return norms


@dataclass(frozen=True)
class _BoundCoefficientEvaluation:
    """Prefactor W_up of bound(t) = W_up*t**(order+1) with the census work that produced it.

    The coefficient is the rational upper enclosure produced by scaled
    binary64 products and sums rounded upward, followed by exact rational
    rescaling in ``coefficient_up``.

    Attributes:
        coefficient: W_up, a rational upper bound on the chosen
            Pauli-triangle expression (``coefficient_up``).
        pauli_term_count: Number of Pauli terms in the census.
        pair_commutation_checks: Logical pair tests performed, P = p(p-1)/2.
        nested_commutation_checks: Logical nested tests performed, N(A) for
            the exact order-2 census and zero otherwise.
        bound_variant: "exact_census" or "relaxed_prefix".
        nested_word_tests: w*N(A), the logical nested word tests of the
            full expression, whether or not the tests were made. For one
            weighting it is the count that chose the expression at order 2
            (``choose_variant``).
    """

    coefficient: Fraction
    pauli_term_count: int
    pair_commutation_checks: int
    nested_commutation_checks: int
    bound_variant: str
    nested_word_tests: int = 0


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
# from strings and store 16pw bytes. Pair structure takes O(wP) work and
# O(E) index storage. Triple structure takes O(wN) further work and O(F)
# index storage (16E and 24F bytes with 64-bit indices).
#
# Relaxation. Put S_i = sum_(k=i+1)^(p-1) a_k and
# D_v = sum_(k:s_vk=1) a_k. The exact nested mass
# M_ij = sum_(k>i) a_k*(s_ki XOR s_kj) is at most S_i. Symplectic
# bilinearity also gives s_ki XOR s_kj <= s_ki+s_kj, hence M_ij <=
# D_i+D_j-a_i: the second degree sum excludes i and s_ij=1. Its remainder
# is nonnegative, j is still in the nested range, and no a_j is subtracted.
# With U_ij = min(S_i,D_i+D_j-a_i),
#   W_2,rel = (1/3) sum_A a_i a_j U_ij + (1/6) sum_A a_i^2 a_j,
#   tail-norm coefficient <= W_2 <= W_2,rel <= W_2,suffix.
# This uses one scalar degree vector and the pair table alone, under the
# same ordered Hermitian Pauli decomposition and pruning choice. Order one
# needs no relaxation. Degrees include earlier neighbors and the union
# includes shared neighbors whose XOR cancels, so this remains a relaxation.
# Suffix sums use a reversed cumulative sum, not total-prefix subtraction,
# which would cancel. For the same positive
# time and allowance, r_exact <= r_rel <= ceil(alpha*r_exact) with
# alpha = sqrt(W_2,rel/W_2) when W_2 > 0. The order X,Y,Z with unit
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
    tables from the two conventions. Besides the tables, the ASCII codes of
    the labels, one difference array and one Boolean mask, 3*len(labels)*q
    bytes, are held at once.
    """
    p, words = len(labels), (q + 63) // 64
    # Row i holds the ASCII codes of label i, column b its character b.
    chars = np.frombuffer("".join(labels).encode("ascii"), dtype=np.uint8).reshape(p, q)
    tables = []
    # X, Y and Z have consecutive codes, so an unsigned difference of at most
    # one from X marks X or Y, and from Y marks Y or Z.
    for first in ("X", "Y"):
        packed = np.zeros((p, 8 * words), dtype=np.uint8)
        # Little-endian bit order within bytes and words puts character b at bit b % 64 of word b // 64.
        packed[:, :(q + 7) // 8] = np.packbits(chars - np.uint8(ord(first)) <= 1, axis=1, bitorder="little")
        tables.append(packed.view("<u8").astype(np.uint64, copy=False))
    return tables[0], tables[1]


def anti(x1, z1, x2, z2):
    """Broadcast Pauli pairs and return their symplectic parity.

    The last axis consists of uint64 words in one consistent convention.
    Two strings anticommute exactly when
    popcount(x1 & z2) + popcount(z1 & x2) is odd.
    """
    bits = (x1 & z2) ^ (z1 & x2)
    return (np.bitwise_count(bits).sum(axis=-1, dtype=np.uint64) & 1) != 0


def pair_rows(x, z, *, max_bytes=None):
    """Return ``(rows, P)``: rows[i] holds the j > i whose Pauli anticommutes with P_i, ascending.

    P = p(p-1)/2 logical pair tests are made. Work is O(p^2 w), and the
    rows hold O(E) indices without a concatenated copy (16pw bytes of masks
    and about 8E + 128p bytes of rows, checked against ``max_bytes``, None
    for the physical memory, as they grow).
    """
    p, w = x.shape if x.ndim == 2 else (len(x), 0)
    rows, pairs = [], 0
    for i in range(p):
        js = np.flatnonzero(anti(x[i], z[i], x[i + 1:], z[i + 1:])) + i + 1
        rows.append(js.astype(np.intp, copy=False))
        pairs += len(js)
        check_memory(16 * p * w + 8 * pairs + 128 * (i + 1), max_bytes, "Pauli census pair rows")
    return rows, p * (p - 1) // 2


def nested_test_count(rows):
    """Return N(A) = sum_i e_i (p-i-1), the nested tests of the exact census, from the row sizes e_i.

    For every anticommuting pair (i, j) the exact census tests every k > i,
    including k = j. With w = ceil(q/64), it makes w*N(A) logical nested
    word tests, and K weightings make K*N(A) weight products
    (``choose_variant``).
    """
    p = len(rows)
    return sum(len(row) * (p - 1 - i) for i, row in enumerate(rows))


def choose_variant(order, requested, selection_count):
    """Return the census expression: order 1 has one, an explicit request is kept, and "auto" compares N(A)*max(w, K) with EXACT_CENSUS_WORD_TESTS.

    ``selection_count`` is N(A)*max(w, K) (``census_structure``). A count
    equal to the threshold chooses the full expression.
    """
    if requested not in (*BOUND_VARIANTS, "auto"):
        raise ValueError(f"bound_variant must be 'auto' or one of {BOUND_VARIANTS}; got {requested!r}")
    if order == 1:
        return "exact_census"
    if requested == "auto":
        return "exact_census" if selection_count <= EXACT_CENSUS_WORD_TESTS else "relaxed_prefix"
    return requested


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
    A product of two positive factors is at least +0, and binary64 values
    from +0 upward are ordered like their bit patterns, so adding one to
    the bit pattern is exactly nextafter(c, inf).
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(under="ignore", over="raise", invalid="raise"):
        c = a * b
    return np.where(np.minimum(a, b) > 0, (c.view(np.int64) + 1).view(np.float64), 0.0)


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


def _row_sums(values):
    """Outward sum of each row of nonnegative ``values`` (``sum_up`` along the last axis)."""
    if values.shape[-1] == 1:
        return values[..., 0].copy()
    return mul_up(values.sum(axis=-1, dtype=np.float64), _sum_factor(values.shape[-1]))


def coefficient_up(magnitudes, rows, *, order, variant, x=None, z=None):
    """Return a rational upper bound on the chosen Pauli-triangle W, or one bound per row of ``magnitudes``.

    ``magnitudes`` holds |c_i| in the census order, one row per weighting of
    the same labels, such as LCHS nodes. A zero entry removes a label exactly.
    ``rows`` comes from ``pair_rows``. variant is exact_census or
    relaxed_prefix. At order 1 both use the pairs A only. At order 2
    exact_census sums a_i a_j n_ij over A, with the nested mass
    n_ij = sum_{k>i} a_k [P_k anticommutes with P_i P_j], k = j included,
    formed from the masks ``x``, ``z`` one tile of pairs at a time.
    relaxed_prefix uses U_ij = min(S_i, D_i+D_j-a_i), with S_i = sum_{k>i} a_k
    and D_v = sum_{k: P_v P_k = -P_k P_v} a_k. Symplectic bilinearity gives
    s_{k,ij} = s_ki XOR s_kj <= s_ki+s_kj. Thus the exact nested mass over
    k>i is at most S_i and D_i+D_j-a_i, since (i,j) anticommutes and the
    range excludes i but includes j. The chosen coefficient is
    sum_A a_i*a_j*n_ij/3 + sum_A a_i**2*a_j/6 (CSTWZ
    doi:10.1103/PhysRevX.11.011020, Prop. 10, Eq. (121)), with U_ij in place
    of n_ij for the relaxation.

    A mask of zeros and ones selects nonnegative magnitudes exactly, and a
    binary64 sum of s values is enclosed by the sequential gamma factor of
    ``_sum_factor`` for every addition tree. Products and the per-row and
    final reductions are rounded outward (``mul_up``, ``sum_up``). The
    grouping is fixed: one reduction per row of A, then one over the rows,
    and nested masks are formed _PAIR_TILE pairs at a time. The power-of-two
    scale and the divisors 3 and 6 are combined as Fractions once. This
    bounds W from above, without claiming exact evaluation of that W.

    Assumptions: binary64 round-to-nearest and finite nonnegative inputs.
    Underflow-aware outward rounding preserves validity over the finite
    input range but has no useful uniform relative-error guarantee when
    monomials underflow after normalization. For relaxed_prefix a negative
    D_j - a_i remainder violates the assumption that (i, j) anticommutes.
    """
    m = np.asarray(magnitudes, dtype=np.float64)
    single = m.ndim == 1
    m = m.reshape(1, -1) if single else m
    if order not in (1, 2) or variant not in BOUND_VARIANTS:
        raise ValueError("invalid formula or bound variant")
    exact2 = order == 2 and variant == "exact_census"
    if exact2 and x is None:
        raise ValueError("the exact census requires the Pauli masks")
    scaled = [scaled_magnitudes(row) for row in m]
    a = np.array([row for row, _ in scaled]).reshape(m.shape)
    nodes, p = a.shape
    if order == 2 and variant == "relaxed_prefix":
        suffix = np.zeros_like(a)
        if p > 1:
            raw = np.cumsum(a[:, :0:-1], axis=1, dtype=np.float64)[:, ::-1]
            suffix[:, :-1] = mul_up(raw, _sum_factor(p - 1))
            del raw
        degree = np.zeros_like(a)
        for i, js in enumerate(rows):
            if len(js):
                degree[:, i] += a[:, js].sum(axis=1)
                degree[:, js] += a[:, i:i + 1]
        degree = mul_up(degree, _sum_factor(max(1, p - 1)))
    # Per-row partial sums, zero for a row without pairs.
    part12, part24 = np.zeros_like(a), np.zeros_like(a) if order == 2 else None
    for i, js in enumerate(rows):
        if not len(js):
            continue
        ai = a[:, i:i + 1]
        ij = mul_up(ai, a[:, js])
        if order == 1:
            part12[:, i] = _row_sums(ij)
            continue
        part24[:, i] = _row_sums(mul_up(ij, ai))
        if variant == "relaxed_prefix":
            remainder = np.nextafter(degree[:, js] - ai, np.inf)
            union = np.nextafter(degree[:, i:i + 1] + remainder, np.inf)
            nested = np.minimum(suffix[:, i:i + 1], union)
            del remainder, union
        else:
            tail = a[:, i + 1:]
            nested = np.empty((nodes, len(js)))
            for start in range(0, len(js), _PAIR_TILE):
                tile = js[start:start + _PAIR_TILE]
                mask = anti((x[i] ^ x[tile])[:, None, :], (z[i] ^ z[tile])[:, None, :],
                            x[None, i + 1:], z[None, i + 1:])
                nested[:, start:start + len(tile)] = np.einsum(
                    "ts,ks->kt", mask.astype(np.float64), tail, optimize=False)
                del mask
            nested = mul_up(nested, _sum_factor(p - i - 1))
        part12[:, i] = _row_sums(mul_up(ij, nested))
        del ij, nested
    bounds = []
    for k, (_, e) in enumerate(scaled):
        if order == 1:
            bounds.append(Fraction(sum_up(part12[k])) * Fraction(2) ** (2 * e))
            continue
        w = Fraction(sum_up(part12[k])) / 3 + Fraction(sum_up(part24[k])) / 6
        bounds.append(w * Fraction(2) ** (3 * e))
    return bounds[0] if single else bounds


def census_work(p, q, *, order, variant, pairs, nested=0, nodes=1):
    """Return G + K*V, the work counted for one census structure and K node contractions.

    A packed symplectic test costs w = ceil(q/64) constant-size word visits,
    and one logical magnitude-contribution visit is counted for each term
    evaluated, with linear passes counted separately:

        G_1 = G_C = wP,  G_D = w(P+N);
        V_1 = E,   V_C = 2p+4E,   V_D = E+N.

    Here C is relaxed_prefix and D exact_census at order 2, P = p(p-1)/2,
    E = ``pairs`` and N = ``nested``, the nested tests of the exact census.
    The count is a proxy for an explicit work limit, not a runtime
    measurement. A caller adds its own linear passes, for QPE ``2*p*q``.
    """
    if min(p, q, pairs, nested) < 0 or nodes < 1:
        raise ValueError("invalid census sizes")
    if order not in (1, 2) or variant not in BOUND_VARIANTS:
        raise ValueError("invalid census kind")
    w = (q + 63) // 64
    total_pairs = p * (p - 1) // 2
    if order == 1:
        return w * total_pairs + nodes * pairs
    if variant == "relaxed_prefix":
        return w * total_pairs + nodes * (2 * p + 4 * pairs)
    return w * (total_pairs + nested) + nodes * (pairs + nested)


def census_bytes(p, q, pairs, *, order, variant, nodes=1, tile=None):
    """Return a coarse estimate of the dominant arrays of one census and its K node contractions.

    With w = ceil(q/64) and E = ``pairs``: 80p for the census inputs, 16pw
    for the masks, 3pq for the label codes and the two temporaries that
    ``pack_labels`` holds while it forms them, 8E + 128p for the pair rows
    and their array headers, 64Kp for the scaled magnitudes, partial sums
    and per-row products of K weightings, 16Kp more for the suffix and degree vectors of
    relaxed_prefix, and 40Tw for the largest tile of nested masks of
    exact_census, a tile of T mask entries. ``tile`` is T from the pair rows
    (``_CensusStructure.nested_tile``), and without it the upper bound
    T = min(_PAIR_TILE, E)(p - 1). T is zero when no pair anticommutes. The
    estimate is meant to be within about a factor of two of the arrays held
    at once, not the process memory.
    """
    if min(p, q, pairs) < 0 or nodes < 1:
        raise ValueError("invalid census sizes")
    w = (q + 63) // 64
    size = 80 * p + 16 * p * w + 3 * p * q + 8 * pairs + 128 * p + 64 * nodes * p
    if order == 2 and variant == "relaxed_prefix":
        size += 16 * nodes * p
    if order == 2 and variant == "exact_census":
        size += 40 * w * (min(_PAIR_TILE, pairs) * max(p - 1, 0) if tile is None else tile)
    return size


def _check_census_memory(p, q, pairs, *, order, variant, max_bytes, nodes=1, held=0, tile=None,
                         operation="Pauli census"):
    """Refuse a census whose dominant arrays exceed the memory limit, naming the relaxed alternative for the exact one."""
    need = held + census_bytes(p, q, pairs, order=order, variant=variant, nodes=nodes, tile=tile)
    try:
        check_memory(need, max_bytes, f"{operation} ({variant})")
    except ResourceLimitError as error:
        if order == 2 and variant == "exact_census":
            relaxed = held + census_bytes(p, q, pairs, order=2, variant="relaxed_prefix", nodes=nodes)
            raise ResourceLimitError(f"{error}. bound_variant='relaxed_prefix' needs about {relaxed} bytes") from None
        raise


@dataclass(frozen=True)
class _CensusStructure:
    """The ordered pair rows of one Pauli label table and the census expression chosen for it.

    ``word_tests`` is the w*N(A) logical nested word tests and
    ``selection_count`` the N(A)*max(w, K) that chose ``variant``. They are
    equal for one weighting. ``nested_tile`` is the largest nested tile in
    mask entries, the maximum of min(_PAIR_TILE, e_i)(p - i - 1) over the
    pair rows, zero without pairs (``census_bytes``). The evaluation records
    ``nested_tests`` only when it performs them.
    """
    x: Any
    z: Any
    rows: list
    pair_tests: int
    pairs: int
    nested_tests: int
    word_tests: int
    selection_count: int
    nested_tile: int
    variant: str


def census_structure(labels, q, order, bound_variant="auto", *, weightings=1, max_bytes=None):
    """Pack ``labels``, scan their anticommuting pairs and choose the census expression.

    The pair scan makes p(p-1)/2 tests and stores the rows (``pair_rows``).
    Before any nested test, the expression is "exact_census" at order 1, the
    requested one, or for "auto" the one that ``choose_variant`` gives from
    N(A)*max(w, K), with K = ``weightings`` the number of magnitude rows
    that the contraction will process. Memory never selects the expression.
    """
    if type(weightings) is not int or weightings < 1:
        raise ValueError("weightings must be a positive integer")
    x, z = pack_labels(labels, q)
    rows, pair_tests = pair_rows(x, z, max_bytes=max_bytes)
    nested = nested_test_count(rows) if order == 2 else 0
    w = (q + 63) // 64
    selection = nested * max(w, weightings)
    p = len(rows)
    tile = max((min(_PAIR_TILE, len(row)) * (p - 1 - i) for i, row in enumerate(rows)), default=0)
    return _CensusStructure(x=x, z=z, rows=rows, pair_tests=pair_tests, pairs=sum(len(row) for row in rows),
                            nested_tests=nested, word_tests=w * nested, selection_count=selection,
                            nested_tile=tile, variant=choose_variant(order, bound_variant, selection))


def _pauli_bound_coefficient_from_terms(
    terms: tuple[tuple[str, float], ...],
    order: int,
    *,
    bound_variant="auto",
    max_bytes=None,
    admit=None,
) -> _BoundCoefficientEvaluation:
    """Bound the ordered Pauli-triangle coefficient W from validated terms.

    For order 1, W=sum_A |c_i*c_j|, where A contains anticommuting i<j.
    For order 2, the exact_census variant returns an outward bound on
    sum_A |c_i*c_j| n_ij/3 + sum_A |c_i|**2*|c_j|/6, with n_ij the mass of
    the k > i, k = j included, whose P_k anticommutes with P_i*P_j. The
    relaxed_prefix variant replaces n_ij by U_ij = min(S_i, D_i+D_j-|c_i|),
    with S_i the full suffix mass and D_v the mass of all anticommuting
    neighbors of v (``coefficient_up``).

    These are triangle expansions of Childs et al.,
    doi:10.1103/PhysRevX.11.011020, Eqs. (120)-(121). The chosen
    formula obeys ||S_o(t/r)**r-exp(-i*t*H)|| <= W*abs(t)**(o+1)/r**o.
    The ordered labels determine the shared structure, and the supplied
    real coefficients determine its weights. Identity phases and pruning
    are accounted for by the caller. W is returned as a rational upper
    coefficient so its exponent can be combined with time without
    intermediate binary64 overflow or loss to zero. A full indicator census
    does not imply exact arithmetic or the exact simulation error.

    ``bound_variant`` is "auto" or an explicit expression
    (``census_structure``). ``admit(census)``, when given, receives the
    ``_CensusStructure``, checks the caller's explicit limits after the pair
    scan and before any nested test, and raises to refuse. The dominant arrays are checked against
    ``max_bytes`` (None: the physical memory) before the contraction. The
    exact census takes O(w*(p**2 + N)) word work and O(p*w + E) storage, the
    relaxed one O(w*p**2) and the same storage, with w=ceil(q/64).
    """
    labels = [label for label, _ in terms]
    magnitudes = np.array([abs(float(coefficient)) for _, coefficient in terms], dtype=np.float64)
    q = len(labels[0]) if labels else 0
    census = census_structure(labels, q, order, bound_variant, max_bytes=max_bytes)
    if admit is not None:
        admit(census)
    _check_census_memory(len(labels), q, census.pairs, order=order, variant=census.variant, max_bytes=max_bytes,
                         tile=census.nested_tile)
    exact2 = order == 2 and census.variant == "exact_census"
    coefficient = coefficient_up(magnitudes, census.rows, order=order, variant=census.variant,
                                 x=census.x, z=census.z)
    return _BoundCoefficientEvaluation(
        coefficient=coefficient,
        pauli_term_count=len(terms),
        pair_commutation_checks=census.pair_tests,
        nested_commutation_checks=census.nested_tests if exact2 else 0,
        bound_variant=census.variant,
        nested_word_tests=census.word_tests,
    )


def _dense_bound_coefficient_from_terms(
    terms: tuple[tuple[str, float], ...],
    order: int,
) -> float:
    """Return W from the exact spectral norms of Eqs. (120)-(121), using dense matrices.

    Validation only. Each dense matrix has 4**n entries for n qubits, and
    each commutator product and spectral norm costs O(8**n) operations.
    The arrays of one term are live at a time (``_tail_commutator_norms``).
    """
    norms = _tail_commutator_norms(terms, order)
    if order == 1:
        c1 = 0.0
        for (value,) in norms:
            c1 += value
        return c1 / 2.0
    c12 = 0.0
    c24 = 0.0
    for value_12, value_24 in norms:
        c12 += value_12
        c24 += value_24
    return c12 / 12.0 + c24 / 24.0


def _validate_limit(value, name, *, minimum):
    """Return ``value`` as an int of at least ``minimum``, else raise ValueError."""
    if not isinstance(value, (int, np.integer)) or value < minimum:
        kind = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {kind} integer; got {value!r}")
    return int(value)


def _bound_coefficient_evaluation(
    hamiltonian: SparsePauliOp,
    *,
    order: int,
    bound_variant: str = "auto",
    max_work: int | None = None,
    max_bytes: int | None = None,
) -> _BoundCoefficientEvaluation:
    """Validate the terms, choose the census expression and evaluate the operator's Pauli-triangle W_up.

    The label conversion, ``128*p*(q+1)`` bytes for p terms on q qubits,
    is checked against ``max_bytes`` (None: the physical memory) before
    ``hamiltonian.to_list()``. ``bound_variant`` is "auto" or an explicit
    expression (``census_structure``). With an explicit ``max_work``, the
    work ``16*p*(q+1) + p*q`` of label preparation and packing plus the
    ``w*p(p-1)/2`` pair tests is checked before the labels are converted,
    and that work plus ``census_work`` of the chosen expression after the
    pair scan and before any nested test. A refusal states the requirement
    and, for the exact census, the work of the relaxed one.
    """
    order = _validate_order(order)
    if max_work is not None:
        max_work = _validate_limit(max_work, "max_work", minimum=1)
    p, q = int(hamiltonian.size), int(hamiltonian.num_qubits)
    check_memory(128 * p * (q + 1), max_bytes, "Pauli label conversion of the census")
    linear = 16 * p * (q + 1) + p * q
    pair_scan = linear + (q + 63) // 64 * (p * (p - 1) // 2)
    if exceeds(pair_scan, max_work):
        raise ResourceLimitError(f"Pauli census pair scan needs work={pair_scan}, above max_work={max_work}")
    terms = tuple(_validated_terms(hamiltonian))

    def admit(census):
        variant = census.variant
        need = linear + census_work(p, q, order=order, variant=variant, pairs=census.pairs,
                                    nested=census.nested_tests if variant == "exact_census" else 0)
        if exceeds(need, max_work):
            message = f"Pauli census ({variant}) needs work={need}, above max_work={max_work}"
            if order == 2 and variant == "exact_census":
                relaxed = linear + census_work(p, q, order=2, variant="relaxed_prefix", pairs=census.pairs)
                message += f". bound_variant='relaxed_prefix' needs work={relaxed}"
            raise ResourceLimitError(message)

    return _pauli_bound_coefficient_from_terms(terms, order, bound_variant=bound_variant, max_bytes=max_bytes,
                                               admit=admit)


def trotter_bound_coefficient(
    hamiltonian: SparsePauliOp,
    *,
    order: int = 1,
    bound_variant: str = "auto",
    max_work: int | None = None,
    max_bytes: int | None = None,
) -> float:
    """Return the time-independent prefactor `W_up` of the Trotter error bound `W_up * t**(order + 1)`.

    ``W_up`` is an outward upper bound on the Pauli-triangle expression
    chosen by ``bound_variant``, which bounds the commutator sum of Childs
    et al., doi:10.1103/PhysRevX.11.011020, Prop. 9 (``order=1``) or
    Prop. 10 (``order=2``), so one product-formula
    application obeys ``bound(t) = W_up * t**(order + 1)`` and the Sec. V B
    ``r``-step total is ``W_up * t**(order + 1) / r**order``. The function
    evaluates the coefficient without dense matrices. W_up can exceed the
    exact triangle coefficient, and it is returned rounded upward to
    binary64.
    The census of anticommuting Pauli pairs, and at order 2 of the later
    terms anticommuting with each pair product, is checked against
    `max_bytes` before its arrays are formed and, when set, against
    `max_work` before any nested test.

    Args:
        hamiltonian (SparsePauliOp): `H = sum_j c_j P_j` in the term order of
            the product formula. Coefficients must be finite and real (an
            imaginary part of at most 1e-12 in absolute value is accepted),
            and identity terms are rejected, so remove them and keep their
            global phase `exp(-i t c)` yourself.
        order (int): Default `1`. Product-formula order, 1 (Lie-Trotter) or
            2 (second-order Suzuki).
        bound_variant (str): Default `"auto"`. At order 2, `"exact_census"`
            selects the full Pauli-triangle expression and `"relaxed_prefix"`
            the degree-capped suffix relaxation, which needs pair tests only
            and can give a larger bound. `"auto"` uses the full expression
            when it needs at most 1,300,000,000 nested word tests (one per
            64 qubits of each test). Order 1 has one expression.
        max_work (int | None): Default `None`, no limit. Optional work limit
            of label preparation, pair and nested tests and coefficient
            contraction.
        max_bytes (int | None): Default `None`, the physical memory. Limit
            on the label conversion and the dominant census arrays.

    Returns:
        coefficient (float): `W_up`, rounded upward to binary64.

    Raises:
        ValueError: W_up exceeds the largest binary64 number, a limit is not
            a positive integer, or the census does not fit ``max_work`` or
            ``max_bytes``.
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
    """Return the dense tail-sum coefficient of Props. 9-10 for validation on small instances.

    This evaluates the spectral norms in Eqs. (120)-(121) from dense matrices,
    so its cost is exponential in the qubit count. The other functions of
    this module use the Pauli-triangle upper bound, which is never smaller.
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


def evaluate_trotter_bound(
    hamiltonian: SparsePauliOp,
    *,
    time: float,
    steps: int,
    order: int = 1,
    bound_variant: str = "auto",
    max_work: int | None = None,
    max_bytes: int | None = None,
) -> float:
    """Return the error bound `W_up * time**(order + 1) / steps**order` of a product formula with a given step count.

    For ``r=steps``, the triangle-inequality composition of Childs et al.
    (doi:10.1103/PhysRevX.11.011020) gives
    ``r * bound(time / r) = W_up * time**(order + 1) / r**order`` with the
    outward coefficient W_up of the ``bound_variant`` expression. With the
    step count fixed, the relaxed expression changes only the certified
    bound, which can be larger.
    The census of anticommuting Pauli pairs, and at order 2 of the later
    terms anticommuting with each pair product, is checked against
    `max_bytes` before its arrays are formed and, when set, against
    `max_work` before any nested test.

    Args:
        hamiltonian (SparsePauliOp): `H = sum_j c_j P_j` with finite real
            coefficients and no identity term, in the term order of the
            product formula, as for `select_trotter_step_count`.
        time (float): Total evolution time.
        steps (int): Positive number of product-formula steps.
        order (int): Default `1`. Product-formula order, 1 (Lie-Trotter) or
            2 (second-order Suzuki).
        bound_variant (str): Default `"auto"`. At order 2, `"exact_census"`
            selects the full Pauli-triangle expression and `"relaxed_prefix"`
            the degree-capped suffix relaxation, which needs pair tests only
            and can give a larger bound. `"auto"` uses the full expression
            when it needs at most 1,300,000,000 nested word tests (one per
            64 qubits of each test). Order 1 has one expression.
        max_work (int | None): Default `None`, no limit. Optional work limit
            of label preparation, pair and nested tests and coefficient
            contraction.
        max_bytes (int | None): Default `None`, the physical memory. Limit
            on the label conversion and the dominant census arrays.

    Returns:
        bound (float): The total bound, rounded upward to binary64.

    Raises:
        ValueError: If `time` is not positive and finite, `steps` is not a
            positive integer, `order` is not 1 or 2, `bound_variant` is
            unknown, a coefficient is not finite or has an imaginary part
            above 1e-12, `hamiltonian` has an identity term, or as for
            `trotter_bound_coefficient`.
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
    bound_variant: str = "auto",
    max_work: int | None = None,
    max_bytes: int | None = None,
) -> TrotterStepSelection:
    """Return the smallest Trotter step count whose error bound meets an error budget.

    Applies the rule of Childs et al., doi:10.1103/PhysRevX.11.011020,
    Sec. V B. With ``r`` steps the total additive
    error is at most ``r * bound(time / r)`` (triangle inequality over the
    unitary steps), and the smallest admissible ``r`` follows in closed
    form because ``bound(t) = W_up * t**(order + 1)``:

    - order 1: total ``= W_up * t^2 / r``, so
      ``r = max(1, ceil(t^2 * W_up / epsilon))``.
    - order 2: total ``= W_up * t^3 / r^2``, so
      ``r = max(1, ceil(sqrt(ceil(t^3 * W_up / epsilon))))``.

    W_up is the outward upper bound of the ``bound_variant`` expression
    (full ``exact_census`` by default, ``relaxed_prefix`` on request), so
    the count is minimal for W_up, not for the exact triangle coefficient.
    At order 2, let ``W_2`` and ``W_2,rel`` be the full and relaxed
    Pauli-triangle coefficients before outward numerical evaluation, and
    let ``r_exact`` and ``r_rel`` be their smallest positive step counts
    for the same time and positive error allowance. If ``W_2 > 0``, then
    ``r_exact <= r_rel <= ceil(alpha * r_exact)`` for
    ``alpha = sqrt(W_2,rel / W_2)`` in exact arithmetic.
    The census of anticommuting Pauli pairs, and at order 2 of the later
    terms anticommuting with each pair product, is checked against
    `max_bytes` before its arrays are formed and, when set, against
    `max_work` before any nested test.

    Args:
        hamiltonian (SparsePauliOp): `H = sum_j c_j P_j` in the term order of
            the product formula. Coefficients must be finite and real (an
            imaginary part of at most 1e-12 in absolute value is accepted),
            and identity terms are rejected, so remove them and keep their
            global phase `exp(-i t c)` yourself.
        time (float): Positive finite evolution time `t`.
        error_budget (float): Positive finite allowed operator-norm error
            `epsilon` of the whole evolution.
        order (int): Default `1`. Product-formula order, 1 (Lie-Trotter) or
            2 (second-order Suzuki).
        bound_variant (str): Default `"auto"`. At order 2, `"exact_census"`
            selects the full Pauli-triangle expression and `"relaxed_prefix"`
            the degree-capped suffix relaxation, which needs pair tests only
            and can give a larger bound. `"auto"` uses the full expression
            when it needs at most 1,300,000,000 nested word tests (one per
            64 qubits of each test). Order 1 has one expression.
        max_work (int | None): Default `None`, no limit. Optional work limit
            of label preparation, pair and nested tests and coefficient
            contraction.
        max_bytes (int | None): Default `None`, the physical memory. Limit
            on the label conversion and the dominant census arrays.

    Returns:
        selection (TrotterStepSelection): `selection.step_count` is the
            step count, and `selection.bound_value` is the certified total
            bound at that count (`<= error_budget`).
            `selection.synthesis_method` names the Qiskit product formula
            (`"lie_trotter"` or `"suzuki_trotter"`).

    Raises:
        ValueError: If `time` or `error_budget` is not positive and finite,
            `order` is not 1 or 2, `bound_variant` is unknown, a coefficient
            is not finite or has an imaginary part above 1e-12, `hamiltonian`
            has an identity term, or as for `trotter_bound_coefficient`.

    Examples:
        For `H = X + Z` the second-order coefficient is
        `||[Z, [Z, X]]||/12 + ||[X, [X, Z]]||/24 = 1/2`, so at `t = 1` and
        `epsilon = 0.01` the smallest `r` with `(1/2)/r**2 <= 0.01` is 8:

        >>> from qiskit.quantum_info import SparsePauliOp
        >>> from nwqlib.subroutines.trotterization import (
        ...     select_trotter_step_count)
        >>> H = SparsePauliOp(["X", "Z"], [1.0, 1.0])
        >>> selection = select_trotter_step_count(
        ...     H, time=1.0, error_budget=0.01, order=2)
        >>> print(selection.step_count, round(selection.bound_value, 10))
        8 0.0078125
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


def _smallest_step_count(one_step_bound: Fraction, budget: float | Fraction, order: int) -> int:
    """Return the smallest r >= 1 with ``one_step_bound / r**order <= budget``.

    ``one_step_bound`` is an exact nonnegative rational (the bound of one
    application over the whole time, formed exactly from the supplied
    coefficient) and ``budget`` a
    positive binary64 value or exact rational. Let n = ceil(one_step_bound / budget). Order 1 needs r >= n, so
    r = max(1, n). Order 2 needs r**2 >= one_step_bound / budget, and since
    r**2 is an integer this is r**2 >= n, so r = ceil(sqrt(n)) =
    1 + isqrt(n - 1) for n >= 1, and r = 1 for n = 0. The ratio is exact, so
    rounded division or sqrt cannot move r across an integer boundary, and
    incrementing a huge integer needs no float comparison. Minimality refers
    to the supplied bound, not to the product-formula error itself. Used by
    ``_selection_from_evaluation``, ``algorithms/qpe/powers._corrected_steps``
    and the periodic LCHS Strang selection
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
    The selection records the evaluation's variant and checks.
    A caller that reuses one evaluation passes zero checks for later
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
    ``d+Wh^2 <= min_(p>0) epsilon_p/t_p``, that is, after pruning,
    ``Wh^2 t_p <= epsilon_p-d*t_p`` for every power.

    Let g be the gcd of the positive requested integer powers. Every boundary
    lies on a common grid if ``h=g*tau/m`` with integer ``m>=1``, and then
    ``r(p)=(p/g)*m``. The smallest integer m >= 1 satisfying
    ``m^2 >= max_(p>0) W(g tau)^2 t_p/(epsilon_p-d t_p)`` is chosen, with a
    positive denominator whenever W > 0. W = 0 accepts a zero remaining
    allowance, and a negative remaining allowance rejects. For W = 1,
    powers 1 and 3, tau = 1 and allowance 6/5 this gives h = 1/2,
    r(1) = 2, r(3) = 6 and bounds 1/4 and 3/4. The exact rational
    computation avoids a floating square root at an integer boundary and
    returns a
    rational ideal step, not a floating-point enclosure of gate
    construction. ``select_common_step`` rechecks the emitted step. Reusing
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
    """Return the exact structural subtotal B_p of one emitted prefix and its five parts.

    All times and coefficients denote exact values of their stored binary64
    numbers. For a nonnegative power p, t_p = ``t`` = p*val(tau),
    r = ``steps``, h = ``h`` = val(h_hat), kept generator A = sum_j c_j P_j
    (the nonidentity terms kept after pruning, with their coefficients),
    exact identity coefficient c_I = ``identity`` and dropped mass
    d_drop = ``loss``:

        B_p = W r abs(h)^3 + d_drop abs(t_p) + C_kept abs(r h - t_p)
              + B_identity,p + B_angles,p,
        C_kept = sum_j abs(c_j),
        B_identity,p = abs(Phi_p + t_p*c_I),
        B_angles,p = 2r sum_j abs(val(a_hat_j) - h*c_j/2).

    ``_common_reductions`` forms C_kept (``mass``) and the one-step leaf
    discrepancy 2 sum_j abs(val(a_hat_j) - h*c_j/2) (``angle_per_step``)
    once per checked count, and ``phase`` is Phi_p.

    The product-formula term is Childs et al., arXiv:1912.08854v3,
    Proposition 16, Eq. (152), DOI 10.1103/PhysRevX.11.011020. Each
    canonical Pauli has norm one, so ||A|| <= C_kept, and the spectral
    theorem gives ||exp(-isA)-exp(-itA)|| <= C_kept |s-t| for the time
    displacement. Phi_p is the prefix sum of the emitted identity
    phases. P(phi) = diag(1, exp(i phi)) differs from P(-t c_I) by at most
    abs(phi + t c_I), and the identity control phase commutes with the
    controlled Pauli evolution. a_hat_j are the emitted half-step
    PauliEvolutionGate times of the leaves exp(-i a_hat_j P_j), not doubled
    RZ parameters. Integrating the derivative of exp(-iaP) bounds a leaf
    difference by abs(a-b), telescoping adds the leaf discrepancies, r
    repetitions multiply the sum by r, and the factor 2 counts forward and
    reverse occurrences even when they share a constructed gate. The leaf
    reference uses h, not t_p/r, since the time term already accounts for
    their difference. These contributions are not generally zero when formed
    from the same exact target times.

    The shared-step time correction is the coefficient L1 mass of the kept
    nonidentity generator multiplied by the exact discrepancy between the
    cumulative represented step time and the selected target time. Both
    quantities are evaluated without a downward floating-point reduction.
    This correction joins the product-formula and pruning terms before the
    per-power check. Identity-phase and rotation-angle formation errors
    use their own contributions. Identity error compares the cumulative
    emitted control phase with the exact target identity phase. Angle error
    sums each emitted Pauli-evolution parameter's discrepancy from its exact
    product at the represented common step time, times its number of
    occurrences.

    This is for the common nonnegative-power convention, order two and a
    shared half-step schedule that is actually repeated.
    ``_select_common_step`` validates those construction invariants and
    rounds B_p upward only after composing it (``_upward_float``).
    """
    parts = dict(trotter=W*steps*abs(h)**3, pruning=loss*abs(t),
                 time=mass*abs(steps*h-t), identity=abs(phase+t*identity),
                 angles=steps*angle_per_step)
    if any(value < 0 for value in parts.values()):
        raise ValueError("error contributions must be nonnegative")
    return sum(parts.values(), Fraction(0)), parts


def _is_normal(value):
    """Return whether the binary64 ``value`` has a finite magnitude in the normal range."""
    return float_info.min <= abs(value) <= float_info.max


def _common_envelope_count(W, coefficients, dt, identity, dropped_mass, g, allowances, phases):
    """Return ``(m_lo, None)``, the count that a rounding envelope proposes for a common grid, or ``(None, why)``.

    The exact subtotal B_p(m) of ``_emitted_parts`` stays the only acceptance
    criterion. This rule is a sufficient envelope that proposes one count for
    it, not another comparator, and that count need not be the smallest emitted
    count that fits. The single-step bound ``||S_2(h)-exp(-i h H_0)|| <= W
    abs(h)**3`` for Hermitian summands and nonnegative time comes from Childs et
    al., arXiv:1912.08854v3, Proposition 16, Eq. (152), whose two classes of
    nested commutator norms have coefficients 1/12 and 1/24. W, the upper bound
    from the current Pauli commutator count, is a premise here. That paper does
    not treat binary64 parameter formation. The grid, the roundoff envelope and
    this rule are this library's own derivation for its emitted schedule.

    Let tau > 0 denote its exact represented value, g the gcd of the positive
    integer powers, n_p = p/g and T_0 = g val(tau). The factors of repeated
    steps are unitary, so telescoping gives r W abs(h)**3 on the grid

        h_* = T_0/m,  r_p = n_p m,  t_p = n_p T_0,  integer m >= 1.                   (T1)

    Every binary64 value denotes its exact rational, and only the composed
    total is rounded outward for reporting. With h_hat = RN(h_*), a_j =
    RN((h_hat/2) c_j), M = sum_j abs(c_j), d_drop the bound on the dropped
    coefficient mass and Phi_p the exact sum of the emitted identity phase
    increments,

        B_p(m) = W r_p abs(h_hat)**3 + d_drop t_p + M abs(r_p h_hat - t_p)
                 + abs(Phi_p + t_p c_I) + 2 r_p sum_j abs(val(a_j) - h_hat c_j/2).    (T2)

    (T2) bounds these structural contributions of the ordered half-step leaves
    and the sorted-prefix identity increments. It does not bound later gate
    synthesis, simulation roundoff or hardware noise.

    The envelope assumes binary64 round-to-nearest with gradual underflow,
    h_hat and h_hat/2 normal, exact halving, finite products and the same
    parameter schedule as (T2) (``_select_common_step`` checks the proposed
    count before using it). Then, with unit roundoff u = 2**-53, smallest
    subnormal eta = 2**-1074 and J kept terms, ``abs(h_hat - h_*) <= u h_*`` and
    ``2 sum_j abs(a_j - h_hat c_j/2) <= u abs(h_hat) M + J eta``. The second
    relation uses relative roundoff for a normal product and an absolute half
    ULP for a subnormal one, and the sum of the two is their common conservative
    envelope. The eta term includes products that round to zero, and it does not
    cover an inexact half step. Therefore

        B_p(m) <= C_p/m**2 + D_p + F_p m,                                             (T3)
        C_p = W n_p (1+u)**3 T_0**3,
        D_p = d_drop t_p + abs(Phi_p + t_p c_I) + u (2+u) M t_p,
        F_p = n_p J eta.                                                              (T4)

    C_p, D_p and F_p are exact rationals here, since unchecked rounded-down
    binary64 arithmetic could lose the bound. Phi_p does not depend on m, since
    the adjacent-power identity schedule is reused unchanged. Applying the
    independent-power correction of ``algorithms/qpe/powers._corrected_steps``
    to each power would break (T1). Only positive powers enter the interval,
    because power 0 has zero count and zero subtotal and its allowance may be
    zero. If Q_p = epsilon_p - D_p > 0 at every positive power, let

        m_lo = max(1, max_(p>0) ceil(sqrt(2 C_p/Q_p))),
        m_hi = min_(p>0: F_p>0) floor(Q_p/(2 F_p)),                                   (T5)

    with an unbounded m_hi when no F_p is positive. C_p = 0 asks nothing of
    m_lo and F_p = 0 asks nothing of m_hi. For a rational R = N/D >= 0, the
    exact ceiling of sqrt(R) is k = isqrt(N//D), incremented when k*k*D < N,
    including R = 0 and exact squares. A rounded float square root could decide
    the boundary wrongly. Every m with m_lo <= m <= m_hi gives C_p/m**2 <= Q_p/2
    and F_p m <= Q_p/2, so (T3) stays within the allowance at every prefix, and
    this rule chooses m_lo.

    ``Q_p <= 0``, an empty interval or a failed normal or finite premise means
    only that this conservative rule confirmed no count, not that the target
    error is unattainable. A ``Q_p <= 0`` refusal reports the largest D_p among
    the powers with no room, since every one of them must have room before the
    rule proposes a count. The rule repairs the failures caused by the small
    margin of the first ideal count and need not find a count on arbitrary
    input. It does not make success monotone in the allowance, because the first
    exact check can pass where this more conservative envelope has no evidence.
    The first count and m+1 need only (T2), so a subnormal case that passes its
    exact check is still accepted and the envelope adds no refusal.

    The work is one pass of O(J+K) scalar reductions over J terms and K powers,
    with no commutator count and no circuit. It counts reductions, not the bit
    cost of the exact rational arithmetic.
    """
    u, eta = Fraction(1, 2**53), Fraction(1, 2**1074)
    magnitudes = [abs(Fraction(value)) for value in coefficients]
    M, J = sum(magnitudes, Fraction(0)), len(magnitudes)
    grown_cube = (1 + u)**3 * (g * dt)**3
    m_lo, m_hi, no_room = 1, None, None
    for p in filter(None, allowances):
        n, t = p // g, p * dt
        D = dropped_mass * t + abs(phases[p] + t * identity) + u * (2 + u) * M * t
        Q = Fraction(allowances[p]) - D
        if Q <= 0:
            no_room = (D, p) if no_room is None or D > no_room[0] else no_room
            continue
        ratio = 2 * W * n * grown_cube / Q
        k = isqrt(ratio.numerator // ratio.denominator)
        m_lo = max(m_lo, k + 1 if k * k * ratio.denominator < ratio.numerator else k)
        if J:
            top = Q // (2 * n * J * eta)
            m_hi = top if m_hi is None else min(m_hi, top)
    if no_room is not None:
        D, p = no_room
        return None, (f"The part of the conservative rounding bound that does not depend on the step count leaves no "
                      f"room below an allowance, and its largest such value is {_upward_float(D)!r} at power {p}, "
                      f"where the allowance is {allowances[p]!r}, so the rule proposes no further count.")
    if m_hi is not None and m_lo > m_hi:
        return None, (f"The conservative rounding bound needs m at least {m_lo} and at most {m_hi}, which no count "
                      "satisfies, so the rule proposes no further count.")
    return m_lo, None


def select_common_step(
    evaluation: _BoundCoefficientEvaluation,
    *,
    coefficients,
    identity: float,
    tau: float,
    allowances,
    dropped_mass: float,
    max_steps: int | None,
) -> TrotterStepSelection | None:
    """Select one ordered second-order step for the nonnegative integer power positions.

    The ideal rational grid uses ``t_p=p*val(tau)`` and integer prefix
    counts. The selection records the emitted binary64 step and the
    cumulative count at every point. Each prefix is checked against its
    allowance with its pruning bound, ``W*r_p*abs(val(h_hat))**3``,
    kept-generator time-displacement bound and applicable identity-phase
    and rotation-angle formation bounds. The counts are sufficient for the
    shared trajectory and need not minimize each prefix independently.

    Rule. Validate finite positive tau, nonnegative W and dropped mass,
    nonnegative integer powers and finite nonnegative per-power allowances
    (``allowances`` maps each power to its allowance). ``tau``,
    ``identity``, the coefficients and the allowances are binary64 values
    whose exact represented values the recheck uses, and a value that
    would round on conversion is refused. Run the ideal
    rational common-grid selector ``common_steps`` once and check its
    largest count against ``max_steps``, where None sets no limit. Form the binary64 step
    ``h_hat = float(h)``, the half-step leaf times ``0.5 * h_hat * c_j``
    (the formation of ``algorithms/qpe/powers.rotation_schedule``) and one
    identity phase increment ``-(p - p_prev) * tau * identity`` per point in
    increasing power order. Compute the exact kept mass and one-step leaf
    discrepancy once per checked count, and accumulate the exact phase prefix
    once per point, since the phases do not depend on the count.
    Compose the five parts of B_p (``_emitted_parts``) at each prefix from
    these shared reductions. The recheck certifies exactly
    these parameters, so the code that builds the circuit applies the leaf
    times that ``rotation_schedule`` forms from h_hat and these identity
    increments between adjacent points. Other emitted values need their own
    recheck. Accept only if each exact
    ``B_p <= val(epsilon_p)``, then round each B_p upward.

    Correction. The first count inverts only the trotter and pruning terms, so
    the rounding of h_hat, the phases and the angles can exceed the margin they
    leave. After a failure, recheck the subdivision m+1 exactly, and then the
    count m_lo that ``_common_envelope_count`` proposes. Every checked count is
    compared with ``max_steps`` through its largest prefix count r(p_max), and
    only m_lo needs the normal premise of that rule. Each recheck forms the step
    and leaf times of its own m on the same grid and takes O(L+K) scalar
    reductions for L kept terms and K powers, so at most two are added. There is
    no commutator count, no circuit and no search over m, and proposing a count
    never enlarges an allowance. When neither count is confirmed, the refusal
    gives the power, target allowance, contribution values and margin of each
    failed count. It says that this rule confirmed no count, not that the target
    error is unattainable, and it does not promise that a larger allowance
    passes.

    Special cases. Without positive powers there is no evolution and no
    phase increment, r(0) = 0 and the subtotal is zero after the zero-power
    allowance is validated, and no common selection is returned (None).
    Pruning alone above any allowance
    refuses. Pruning equal to the allowance refuses for W > 0 at a positive
    time. For W = 0 the emitted phase, angle and time terms must still fit.
    An empty kept generator has zero time-displacement and angle terms, and
    only pruning and identity-phase representation remain. A positive
    subnormal h_hat is checked with its represented angles, so a
    product 0.5*h_hat lost to rounding is counted in B_angles,p. An h_hat
    that rounds to zero for a positive-time nonempty step refuses before
    the circuit is built. Nonfinite phases or angles refuse. W
    (``evaluation``) must bound the triangle coefficient of the ordered kept
    generator, and ``dropped_mass`` must be the exact sum of discarded
    magnitudes or an outward upper bound on it. A previously rounded-down
    mass does not meet that assumption.

    The returned selection carries the evaluation's census counts. A caller
    that already recorded them passes an evaluation with zero checks.
    """
    return _select_common_step(
        evaluation, coefficients=coefficients, identity=identity, tau=tau,
        allowances=allowances, dropped_mass=dropped_mass, max_steps=max_steps)[0]


def _select_common_step(evaluation, *, coefficients, identity, tau, allowances, dropped_mass,
                        max_steps):
    """Return (``select_common_step``'s selection, the exact prefix subtotals, the certified increments).

    The second value maps every requested power to the exact (total, parts)
    of its accepted emitted prefix (``_emitted_parts``), and the third maps
    it to the binary64 identity phase increment the recheck certified before
    that power position. Both are empty when no positive power exists. A
    caller reports the components of each prefix and places exactly these
    increments in the circuit without evaluating them again.
    """
    coefficients = tuple(_binary64(value, "each coefficient") for value in coefficients)
    if len(coefficients) != evaluation.pauli_term_count:
        raise ValueError("coefficients must be the evaluation's kept terms")
    identity = _binary64(identity, "identity")
    tau = _binary64(tau, "tau")
    if tau <= 0:
        raise ValueError("tau must be finite and positive")
    # The dropped mass is an exact value or an outward upper bound, so any real
    # rational within the binary64 range is kept exactly. A complex, string or
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
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None")
    powers = tuple(sorted(allowances))
    if any(type(p) is not int or p < 0 for p in powers):
        raise ValueError("use nonnegative integer powers")
    allowances = {p: _binary64(allowances[p], "each per-power allowance") for p in powers}
    if any(allowances[p] < 0 for p in powers):
        raise ValueError("per-power allowances must be finite and nonnegative")
    positive = tuple(p for p in powers if p)
    if not positive:
        return None, {}, {}
    _, ideal = common_steps(evaluation.coefficient, tau, powers, allowances, dropped_mass)
    g = reduce(gcd, positive)
    p_max = positive[-1]
    m = ideal[p_max] // (p_max // g)

    def admit(subdivisions, remedy):
        count = p_max // g * subdivisions
        if exceeds(count, max_steps):
            raise ResourceLimitError(
                f"common step count {count} at power {p_max} exceeds the step limit {max_steps}, the "
                f"max_trotter_steps of a QPE method or the max_steps of select_common_step. Raise it to at least "
                f"{count}{remedy}"
            )

    admit(m, ", or raise the allowance to need fewer steps")
    dt, ci = Fraction(tau), Fraction(identity)
    # The identity schedule does not depend on m, so every checked count emits these increments.
    increments, phases, phase, previous = {}, {}, Fraction(0), 0
    for p in powers:
        try:
            increments[p] = -(p - previous) * tau * identity
        except OverflowError:
            raise ValueError("emitted phases and angles must be finite") from None
        previous = p
        if not isfinite(increments[p]):
            raise ValueError("emitted phases and angles must be finite")
        phase += Fraction(increments[p])
        phases[p] = phase

    def emit(subdivisions):
        """Return h_hat, the exact (total, parts) of every prefix, and the first power over its allowance, or None."""
        h = g * dt / subdivisions
        # float(Fraction) raises OverflowError instead of returning inf.
        if h > Fraction(float_info.max):
            raise ValueError("the common step is not a finite binary64 number")
        h_hat = float(h)
        if h_hat == 0 and coefficients:
            raise ValueError("the common step rounds to zero for positive-time evolution")
        half_angles = tuple(0.5 * h_hat * value for value in coefficients)
        if not all(isfinite(value) for value in half_angles):
            raise ValueError("emitted phases and angles must be finite")
        hi = Fraction(h_hat)
        mass, angle_per_step = _common_reductions(coefficients, hi, half_angles)
        totals = {p: _emitted_parts(
            evaluation.coefficient, dropped_mass, mass, angle_per_step, phases[p],
            p*dt, hi, ci, p // g * subdivisions,
        ) for p in powers}
        return h_hat, totals, next((p for p in powers if totals[p][0] > Fraction(allowances[p])), None)

    def missed(subdivisions, h_hat, totals, p):
        # B_p > epsilon_p is the same as time + identity + angles > epsilon_p - trotter - pruning.
        # common_steps inverts only the right side, and the left side is the rounding of h_hat, the
        # phases and the angles, which is zero on the ideal rational grid.
        total, parts = totals[p]
        margin = Fraction(allowances[p]) - parts["trotter"] - parts["pruning"]
        return (
            f"common step count {p // g * subdivisions} (m={subdivisions}, h_hat={h_hat!r}) does not meet the "
            f"allowance {allowances[p]!r} at power {p}: emitted subtotal {_upward_float(total)!r} with contributions "
            + ", ".join(f"{name}={_upward_float(value)!r}" for name, value in parts.items())
            + f", and the trotter and pruning terms leave a margin of {float(margin)!r} below the allowance"
        )

    def corrected(subdivisions, first):
        """Return the count and emission that pass after ``first`` failed, or refuse (``_common_envelope_count``)."""
        tried = [missed(subdivisions, *first)]

        def unconfirmed(why):
            return ValueError(
                "; ".join(tried) + f". {why} This conservative rule for proposing step counts confirmed no feasible "
                "count, which does not show that the target error is unattainable. A larger allowance, the "
                "controlled_power_error_budget of a QPE method or the allowances of select_common_step, can be tried, "
                "but none is guaranteed to pass, because each allowance has its own counts and its own exact check"
            )

        def admit_proposed(proposed):
            admit(proposed, " so that the exact check can test this count. Smaller counts failed it: "
                  + "; ".join(tried))

        subdivisions += 1
        admit_proposed(subdivisions)
        emission = emit(subdivisions)
        if emission[2] is None:
            return subdivisions, emission
        tried.append(missed(subdivisions, *emission))
        proposed, why = _common_envelope_count(
            evaluation.coefficient, coefficients, dt, ci, dropped_mass, g, allowances, phases)
        if proposed is None:
            raise unconfirmed(why)
        admit_proposed(proposed)
        step = float(g * dt / proposed)
        if not (_is_normal(step) and _is_normal(0.5 * step)):
            raise unconfirmed(f"The proposed m={proposed} gives the step {step!r}, whose half is below the normal "
                              "binary64 range, so the rounding bound that proposes it does not apply.")
        emission = emit(proposed)
        if emission[2] is None:
            return proposed, emission
        tried.append(missed(proposed, *emission))
        raise unconfirmed("That last count is the one the conservative rounding bound proposes.")

    emission = emit(m)
    if emission[2] is not None:
        m, emission = corrected(m, emission)
    h_hat, bounds, _ = emission
    counts = {p: p // g * m for p in powers}
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
        common_tau=float(tau),
        common_g=g,
        common_m=m,
        common_step_time=h_hat,
        common_prefix_steps=tuple((p, counts[p]) for p in powers),
        common_power_allowances=tuple((p, float(allowances[p])) for p in powers),
        common_recheck_outcome="accepted",
        common_prefix_bounds=tuple(published.items()),
    ), bounds, increments


def build_trotter_evolution_circuit(
    hamiltonian: SparsePauliOp,
    selection: TrotterStepSelection,
) -> QuantumCircuit:
    """Build the product-formula circuit `S_order(time / step_count)**step_count` of a step-count selection.

    Checks that `hamiltonian` has finite real coefficients and no identity
    term, as `select_trotter_step_count` requires, and builds the circuit
    with Qiskit's product-formula synthesis (`LieTrotter` or
    `SuzukiTrotter`, as `selection.synthesis_method` names) repeated
    `step_count` times. It does not check that `hamiltonian` is the operator
    `selection` was computed for, and `selection.bound_value` holds only for
    that operator in the same term order. The summands are applied in term
    order, matching the ordering the recorded bounds are stated for.

    Args:
        hamiltonian (SparsePauliOp): The Hamiltonian the selection was
            computed for.
        selection (TrotterStepSelection): Output of
            `select_trotter_step_count`.

    Returns:
        circuit (QuantumCircuit): The evolution circuit on
            `hamiltonian.num_qubits` qubits.

    Raises:
        ValueError: If a coefficient is not finite, has an imaginary part
            above 1e-12 in absolute value, or `hamiltonian` has an identity
            term.
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
]
