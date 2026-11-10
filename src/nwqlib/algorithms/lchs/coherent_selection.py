r"""Static choice of the coherent LCHS evolution route, and preparation of the chosen route only.

Equation labels refer to the LCHS mathematical specification and its
supplements (MATHEMATICAL_SPEC E6-E15, R1-R3; transformation T1-T28;
encoding G12-G17, G24-G27; alternative transformations AT8-AT15).

Selection. One coherent SELECT applies ``exp(-i T_e S)`` to the joint
generator ``S`` of `plan_joint_generator` (G12) or `plan_single_a_generator`
(G13), with ``T_e`` the largest active elapsed time and time factors
``q_a = t_a / T_e`` read from the factorized PREP layout
(`plan_factorized_source_preparation`, G14-G17). Every row of the result is a
node/joint generator together with one transformation: shared real QSVT raw
or with three-use OAA (T7-T13) for each common cap in `SHARED_TARGET_CAPS`,
ordinary Laurent (T14/T15) and one-sided GQET (T16/T17) for each margin in
`GENERAL_MARGINS`, the imaginary-time filter (AT8-AT13) of an H = 0
alternative, or, when the representation gives an exact [L, H] = 0, the
product of that filter with an H evolution (AT14/AT15,
`CommutingProductAlternative`). Shared QSVT reads the generator directly when its encoded
block is recorded Hermitian; the qubitized walk reads it directly when the
whole unitary is a recorded Hermitian involution; otherwise both use the
generic lift (T3/T4). Berry, outer amplitude amplification, Martyn one-shot
and coherent MPF are listed as refused with the M0 reasons. An exactly zero
generator or ``T_e = 0`` gives the identity evolution and no polynomial row
(T1 boundary); no nonzero application gives the exact zero solution.

Allowance (E15). In the physical-vector frame with outer recovery ``gamma``
(G14) and the qualified other stages ``Delta_rest``,

    \[\epsilon_{\rm alg,allow}=(\epsilon_{\rm phys}-\Delta_{\rm rest})/\gamma
      -T_e\delta_S-\zeta\epsilon_{\rm native}.\]

``T_e delta_S`` is the generator term of E6 (``beta delta_X`` for the
filter, AT13). In the algorithmic frame the caller supplies
``epsilon_alg,allow`` directly and the generator and native terms are
reported beside it. A tail allowance ``tail_share * epsilon_alg,allow``
selects the truncation degree; the OAA rows use the conservative inverse of
``Phi_{1/2}``, since ``Phi_a <= Phi_{1/2}`` for ``a <= 1/2`` (T13). Then

* Laurent/GQET: ``epsilon_synth = (epsilon_alg,allow - delta_T) / sigma``,
  rounded down, is the phase allowance (E15, T15, T17);
* shared raw/OAA: the phase owner fixes the residual tolerance ``tol`` per
  parity, so ``delta <= delta_T + 2 s tol`` (T10) and the row is admitted
  only when ``delta`` (raw) or ``Phi_a(delta)`` (OAA) is within the
  allowance;
* imaginary time: ``epsilon_synth = (epsilon_I,allow - delta_m) / s``, with
  ``s = nextafter(1/(1-eta))`` the scale that `imaginary_time_expansion`
  returns when the stored coefficient sum, rounded upward, is at most one; a
  larger returned scale refuses the row before synthesis.

A forecast is conditional on the synthesis meeting its tolerance; binary64
Bessel, phase and PREP arithmetic stay UNKNOWN, never zero.

Cost (T24, R1, G27). One attempt costs ``C = C_evol + P_out`` CX. ``C_evol``
adds the original-U calls of the transformation census, priced by the
generator's own ``forward_cx`` (no control) and ``controlled_cx`` (one
control), and the CX the emitters add around those calls. Both come from
the emitter owners in ``qsp.evolution``, which the plan records of the
built circuit use too: ``_shared_census`` and ``_shared_gate_cx`` for
shared QSVT, ``_qubitized_census`` and ``_qubitized_gate_cx`` for the
qubitized walk (M7.2 of the M7 mathematics handoff).

``P_out`` is the factorized PREP and its inverse
(`direct_preparation_cx_bound` for each magnitude tree, or the layered MPS
estimate ``3 L max(0, q - 1)`` of an explicit MPS factor), the slot phase
diagonal (`multiplexor_resource_law` for each diagonal that
`build_coefficient_phase` appends) and the caller's CX count of the
kind-state preparation R. Child laws carry their own upper-bound
qualification; an unknown child or R cost leaves the row's cost unknown.

Ranking (R2, T25, T26). All rows of one alternative share one finite
target, so rejection acquisition ranks ``zeta**2 C`` and single-circuit
acquisition ranks ``C``. Single-circuit acquisition covers exact readout and
fixed-shot moments; under fixed shots ranking ``C`` is a fixed-acquisition
comparison, not matched precision (M7 mathematics handoff, Q6). Admitted
rows with exactly the same objective go to the lowest total degree of the
phase polynomials to synthesize, a classical cost the objective does not
contain, and remaining ties keep the row order. Under fixed shots equal ``C`` is not equal precision:
the Hoeffding range radius of an unconditional moment scales as
``Gamma**2 = (zeta gamma)**2`` (M7.13-M7.14) at fixed shots and statistical
model, and the accepted probability ``q/Gamma**2`` of a success-conditional
output, with ``q`` the squared norm of the approximate vector, as ``1/Gamma**2``
only when ``q`` is the same; the reason names the
consequence for the requested readout with the tied rows' recoveries and
phase degrees. ``LCHSFixedShotPrecision`` records the same relation against the
admitted row of smallest ``Gamma`` (`smallest_recovery_row`), because the
ranking of ``C`` does not change and no other record shows it. Rows of unknown cost are
not ranked, and the reason says the minimum is not certified over them. When
no admitted row has a known cost, the deterministic choice takes the
smallest recovery ``zeta`` (rejection and accuracy) or the fewest original-U calls
(single circuit), then the lowest total phase degree, the classical
tie-break of known-cost rows applied to that stand-in objective; this order
is a principal and counselor decision, not a certified minimum. Different alternatives have different finite targets
and different gamma. Their chosen rows are compared by the nominal
``C (zeta gamma)**2`` for rejection (M7.8, PLAN M0.1) and by ``C`` for a
single circuit, the fixed-acquisition comparison; both are nominal, since
a strict order needs every total error against one reference solution and
a positive norm lower bound (T25, M7.9). The caller's alternative order
breaks ties and decides only when no chosen row has a known nominal value.

Accuracy (ACCURACY_HOOK_POLICY AH1-AH3). A requested absolute sampling
radius of a physical scalar output (`SamplingTarget`) gives each row the
sufficient shots N per setting of its own recovery ``Gamma = zeta gamma``
(``analysis.sufficient_counts``), and the row objective is the quantum work
``J = N G C`` over the G settings. N already contains ``Gamma**4``, so known
J values compare directly within and across alternatives. They are
sufficient-work forecasts from upper bounds, not a certified minimum of the
realized cost. A row whose N exceeds the exact integer population is not
eligible; rows of unknown cost fall back to the rejection order above.

Work. Selection runs Jacobi-Anger and modified-Bessel degree selection,
the real-target normalization of `_qsp_phase_targets`, the imaginary-time
coefficient acquisition of `imaginary_time_expansion` and the
PREP/generator plans. No phase is fitted and no circuit is built.
`prepare_selected_coherent_route` then synthesizes the selected target
once, with the frozen expansion, cap or margin and tolerance; a synthesis
failure is never followed by another row.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Literal, Mapping, NamedTuple, Sequence

import numpy as np

from nwqlib._limits import ResourceLimitError
from nwqlib._preparation_laws import direct_preparation_cx_bound
from nwqlib._validation import finite_real, integer
from nwqlib.subroutines._multiplexors import multiplexor_resource_law
from nwqlib.subroutines.block_encoding.joint import (
    DiagonalTable,
    plan_convex_contraction,
    plan_joint_generator,
    plan_single_a_generator,
)
from nwqlib.subroutines.block_encoding.nodes import uniform_node_labels, uniform_node_recipes
from nwqlib.subroutines.block_encoding.source_prep import plan_factorized_source_preparation
from nwqlib.subroutines.qsp.evolution import (
    _qsp_phase_targets,
    _qubitized_census,
    _qubitized_gate_cx,
    _shared_census,
    _shared_gate_cx,
    _shared_helper_x,
    _solve_qsp_targets,
    imaginary_time_filter_error_terms,
    imaginary_time_target,
    jacobi_anger_expansion,
    plan_jacobi_anger_gqet,
    plan_jacobi_anger_laurent,
    plan_qubitized_evolution,
    plan_shared_qsvt_evolution,
    prepare_qubitized_target,
    with_phase_allowance,
)
from nwqlib.subroutines.qsp.imaginary_time import (
    imaginary_time_expansion,
    prepare_imaginary_time_filter,
    select_imaginary_time_degree,
)

# Common caps of the shared real targets: 1 - eta for the general margins,
# plus 1/2 (M0.2; PHASE_POLICY_REVISION section 10 compares exactly these four).
SHARED_TARGET_CAPS = (0.999, 0.99, 0.9375, 0.5)
# The general margin set of M0.2, also the margins `imaginary_time_expansion` accepts.
GENERAL_MARGINS = (1e-3, 1e-2, 1 / 16)

NOT_ADMITTED_ROUTES = {
    "berry": "Berry prescribed-pair synthesis is not admitted (M0 ADMISSIONS section 2): no qualified bounded "
             "factorization of the T21 pair preserves T22 and the d+2 signal count. This is not a "
             "nonexistence result.",
    "outer_amplitude_amplification": "Outer amplitude amplification is not admitted (M0 ADMISSIONS section 5): "
                                     "no finite schedule with a positive success premise and original-amplitude "
                                     "recovery exists; amplified mass is not physical mass.",
    "martyn_one_shot": "Martyn one-shot is not admitted (M0 ADMISSIONS section 7): the complex-even endpoint "
                       "completion and realizable degree (AT6) are not closed.",
    "coherent_mpf": "Coherent amplified MPF is not admitted (M0 ADMISSIONS section 7): (4V-U)/3 is not a norm-one "
                    "compression (AT24) and the robust-amplification error chain is not closed.",
}

_REAL_TIME_ROUTES = ("shared_oaa", "shared_raw", "laurent", "gqet")
_EXACT_ROUTES = ("zero_solution", "identity")
_ARITHMETIC_UNKNOWN = ("bessel_or_ive_arithmetic", "phase_arithmetic", "prep_angle_rounding")


@dataclass(frozen=True, kw_only=True)
class SamplingTarget:
    """A requested absolute sampling radius of one physical scalar output and its counts settings.

    Attributes:
        output: ``"quadratic_form"`` or ``"norm_squared"``.
        ranges: ``A_g`` of each qubit-wise commuting group setting
            (``analysis.score_ranges``), empty for the norm squared, whose
            one setting is the physical mass.
        tolerance: The radius epsilon, positive, in the output's unit.
        confidence: ``1 - delta`` of the radius.
    """

    output: Literal["quadratic_form", "norm_squared"]
    ranges: tuple[float, ...]
    tolerance: float
    confidence: float

    @property
    def settings(self) -> int:
        """G, the number of counts settings."""
        return len(self.ranges) if self.output == "quadratic_form" else 1

    def shots(self, recovery) -> int | None:
        """N per setting for the physical recovery ``Gamma`` (``analysis.sufficient_counts``), None beyond what a saved Plan holds."""
        from .analysis import sufficient_counts
        return sufficient_counts(recovery, kind=self.output, ranges=self.ranges, tolerance=self.tolerance,
                                 confidence=self.confidence)


@dataclass(frozen=True, kw_only=True)
class CoherentObjective:
    """The requested output acquisition and error frame of one coherent selection.

    Attributes:
        acquisition: ``"rejection"`` ranks ``zeta**2 C`` (state or Born
            samples by postselection, R2/T26). ``"single_circuit"`` ranks
            ``C``, the cost of one coherent circuit: for exact readout, and
            for moments measured with fixed shots, where it is a
            fixed-acquisition comparison and not matched precision (the
            radius scales as ``Gamma**2``). ``"accuracy"`` ranks the
            sufficient quantum work ``N G C`` of ``sampling`` (AH1-AH3).
        allowance_frame: ``"physical_vector"`` applies E15 to ``allowance``
            as ``epsilon_phys``; ``"algorithmic"`` takes ``allowance`` as
            ``epsilon_alg,allow`` in the recovered evolution frame, a
            component tolerance whose physical total is not budgeted here.
        allowance: Positive allowance in the frame above.
        tail_share: Share of the algorithmic allowance given to the
            truncation tail, strictly between zero and one. The remainder
            funds synthesis.
        native_error: ``epsilon_native`` of one complete evolution circuit,
            required in the physical frame (0.0 states exact native gates);
            None is unknown.
        readout: How the output is read, for the reasons of a fixed-shot
            choice: ``"exact"``, ``"unconditional_counts"`` (norm squared,
            quadratic form, padded normalized output), ``"conditional_counts"``
            (unpadded normalized output) or ``"samples"``; None is not stated.
        sampling: The `SamplingTarget` of ``"accuracy"``, else None.
    """

    acquisition: Literal["rejection", "single_circuit", "accuracy"]
    allowance_frame: Literal["physical_vector", "algorithmic"]
    allowance: float
    tail_share: float
    native_error: float | None = None
    readout: Literal["exact", "unconditional_counts", "conditional_counts", "samples"] | None = None
    sampling: SamplingTarget | None = None


@dataclass(frozen=True, kw_only=True)
class CoherentControls:
    """Limits of selection and of the selected synthesis.

    Attributes:
        max_bytes: Memory limit of each numerical step. None uses the
            physical memory.
        max_node_work: Planning work limit of the Pauli node recipes. None
            sets no limit.
    """

    max_bytes: int | None = None
    max_node_work: int | None = None


@dataclass(frozen=True, kw_only=True)
class CoherentRequest:
    """Explicit constraints on the selected row; a refused requested row is never replaced by another.

    Attributes:
        route: Route name, including the refused ones in `NOT_ADMITTED_ROUTES`.
        level: Shared cap or Laurent/GQET/filter margin.
        joint: ``"separate_parts"``, ``"single_a"`` or ``"convex_contraction"``.
        node: Node encoding after the ``/`` of a generator label, for
            example ``"comparator"`` or ``"table"``.
        alternative: Alternative label.
    """

    route: str | None = None
    level: float | None = None
    joint: str | None = None
    node: str | None = None
    alternative: str | None = None


@dataclass(frozen=True, kw_only=True)
class KernelGrid:
    """Kernel nodes and coefficients ``c_j`` of one finite LCHS sum, in node order.

    Attributes:
        nodes: Physical nodes ``k_j``.
        coefficients: Complex coefficients ``c_j`` including k-quadrature weights.
        half_width: J of a uniform grid ``k_j = h j``, ``j = -J..J``, else None.
        spacing: h of a uniform grid, else None.
    """

    nodes: tuple[float, ...]
    coefficients: tuple[complex, ...]
    half_width: int | None = None
    spacing: float | None = None


@dataclass(frozen=True, kw_only=True)
class LCHSCoherentAlternative:
    """One kernel/source alternative of the coherent LCHS sum with its already qualified facts.

    Attributes:
        label: Name used in rows and requests.
        applications: `LCHSApplicationFacts` of kind ``"homogeneous"`` or
            ``"discrete_source"``; masses include ``exp(s t_a)``.
        kernel: The kernel grid and coefficients.
        l_child: Encoding of the centered ``L_0``, or None for an exact zero.
        h_child: Encoding of the centered ``H_0``, or None for an exact zero.
        a_child: Encoding of ``A_0 = L_0 + i H_0`` for the single-A form, or None.
        rest_error: Qualified sum of the other physical-vector stages of
            E11 (kernel, k-quadrature, source quadrature/window, PREP), or
            None when unknown. An omitted-application bound of the
            factorized PREP is added here by the selection.
        state_preparation_cx: CX count of the kind-state preparation R as
            `build_kind_state_preparation` builds it, or None when unknown.
        drop_allowance: Physical allowance for applications whose amplitude
            underflows, or None.
        omitted_part_bound: Operator-norm bound of a nonzero centered part
            that the child owner left out of the generator, such as an H
            whose Pauli terms relative pruning removed entirely, else 0. It
            is added to the generator error ``delta_S``, because the omitted
            ``D_Q (x) I (x) X`` has norm at most ``||X||`` for ``q_a <= 1``.
        coefficient_mps_layers: Layer count L when both magnitude factors of
            the coefficient PREP are explicit MPS circuits, priced by
            ``3 L max(0, q - 1)`` each; None for direct trees.
        analytic_prep: Keyword arguments of
            ``analytic_prep.plan_analytic_kernel_preparation`` except the
            recipe and ``max_rounds`` (``gamma``, ``approx_tol``,
            ``margin`` and limits) when the kernel factor must be the analytic
            Low–Somma PREP of the uniform grid; None for a direct or MPS
            kernel factor. The caller states that the kernel coefficients
            have the Low–Somma magnitudes ``|c_j|/beta_k = f(k_j)**2/(M p_f)``
            of that gamma (G18).
    """

    label: str
    applications: Any
    kernel: KernelGrid
    l_child: Any = None
    h_child: Any = None
    a_child: Any = None
    rest_error: float | None = None
    state_preparation_cx: int | None = None
    drop_allowance: float | None = None
    omitted_part_bound: float = 0.0
    coefficient_mps_layers: int | None = None
    analytic_prep: Mapping[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class ImaginaryTimeAlternative:
    """The H = 0 imaginary-time alternative (AT8-AT13), a finite target without a kernel.

    The Hamiltonian must be an exact scalar ``c_H`` (``applications.hamiltonian_scalar``).
    The masses must be ``w_a nu_a kappa_a`` with ``kappa_a = exp[-t_a (c - alpha)]``,
    i.e. built by `application_facts` with ``signed_shift = alpha - c_L``.

    Attributes:
        label: Name used in rows and requests.
        applications: `LCHSApplicationFacts` with the masses above.
        l_child: Encoding of the centered ``L_0``; its error bound must be known.
        alpha: Contraction normalization, at least ``alpha_L + delta_L``.
        rest_error: As for `LCHSCoherentAlternative`, without kernel stages.
        state_preparation_cx: As for `LCHSCoherentAlternative`.
        drop_allowance: As for `LCHSCoherentAlternative`.
        coefficient_mps_layers: As for `LCHSCoherentAlternative`.
    """

    label: str
    applications: Any
    l_child: Any
    alpha: float
    rest_error: float | None = None
    state_preparation_cx: int | None = None
    drop_allowance: float | None = None
    coefficient_mps_layers: int | None = None


@dataclass(frozen=True, kw_only=True)
class CommutingProductAlternative:
    r"""The conditional commuting product (AT14/AT15): an L filter times an H evolution, a finite target without a kernel.

    For Hermitian parts with an exact ``[L, H] = 0`` in one basis, ``L = c_L I + L_0`` and
    ``H = c_H I + H_0``, every application obeys (M7.5 of the M7 mathematics handoff)

        \[e^{-t_a(L+iH)}=\kappa_a e^{-it_ac_H}F_{I,a}e^{-it_aH_0},\quad
          F_{I,a}=e^{-\beta(1-X_a)},\quad X_a=(1-q_a)I-q_aL_0/\alpha,\]

    with ``beta = T_e alpha`` and ``kappa_a = exp[t_a (alpha - c_L)]``, so the masses are those of
    `ImaginaryTimeAlternative` and the scalar H phase enters the slot phase once. The H factor
    evolves ``D_Q (x) H_0`` for ``T_e`` on the same application register. Approximate factors
    ``A_I = sigma_I B_I`` and ``A_H = zeta_H B_H`` within ``e_I, e_H`` of the contractions
    ``F_I, F_H`` compose, without any commutation of the approximants and with independent success
    workspaces, to (M7.6)

        \[\|A_IA_H-F_IF_H\|\le e_I+e_H+e_Ie_H,\qquad \zeta=\sigma_I\zeta_H.\]

    An allowance B of the product is split equally, ``b = sqrt(1 + B) - 1 = B/(sqrt(1 + B) + 1)``,
    so that ``2 b + b**2 = B`` (M7.7). Unitary OAA is never applied to the filter.

    Attributes:
        label: Name used in rows and requests.
        applications: `LCHSApplicationFacts` with the masses above (``signed_shift = alpha - c_L``).
        l_child: Encoding of the centered ``L_0``; its error bound must be known.
        h_child: Encoding of the centered ``H_0``.
        alpha: Contraction normalization, at least ``alpha_L + delta_L``.
        commutation: Provenance of the exact ``[L, H] = 0`` fact, such as ``"periodic_bands"``.
        rest_error: As for `LCHSCoherentAlternative`, without kernel stages.
        state_preparation_cx: As for `LCHSCoherentAlternative`.
        drop_allowance: As for `LCHSCoherentAlternative`.
        coefficient_mps_layers: As for `LCHSCoherentAlternative`.
    """

    label: str
    applications: Any
    l_child: Any
    h_child: Any
    alpha: float
    commutation: str
    rest_error: float | None = None
    state_preparation_cx: int | None = None
    drop_allowance: float | None = None
    coefficient_mps_layers: int | None = None


@dataclass(frozen=True, kw_only=True)
class CoherentCandidate:
    """One generator-transformation row with its admission, forecast and cost.

    Attributes:
        alternative: Alternative label.
        generator: ``"separate_parts/<node>"``, ``"single_a/<labels>"``,
            ``"convex_contraction"`` or ``"none"``.
        route: Transformation or exact shortcut.
        level: Shared cap or Laurent/GQET/filter margin, else None.
        status: ``"admitted"``, ``"refused"`` or ``"unknown"`` (a structural
            premise without evidence).
        reasons: Why the row has its status, and qualifications.
        requested: Whether the row satisfies the explicit request.
        signal_access: ``"direct"`` or ``"generic_lift"`` (shared),
            ``"direct_involution"`` or ``"generic_lift"`` (qubitized).
        recovery: zeta of the route.
        polynomial_time: ``tau = alpha_S T_e`` or beta.
        degrees: Polynomial degrees and signal counts.
        allowance: Algorithmic, tail and synthesis allowances.
        error_forecast: Forecast terms in the recovered frame, and the
            physical evolution term; None is unknown.
        census: Original-U call rows and the emitter's other gate counts.
        cost: CX of the evolution, the PREP and one attempt, width,
            original-U calls and the ranking objective; None is unknown.
        unknown_terms: Error or cost terms whose value is unknown.
        handles: Frozen data for `prepare_selected_coherent_route`.
    """

    alternative: str
    generator: str
    route: str
    level: float | None
    status: Literal["admitted", "refused", "unknown"]
    reasons: tuple[str, ...]
    requested: bool = True
    signal_access: str | None = None
    recovery: float | None = None
    polynomial_time: float | None = None
    degrees: Mapping[str, int] = field(default_factory=dict)
    allowance: Mapping[str, float | None] = field(default_factory=dict)
    error_forecast: Mapping[str, float | None] = field(default_factory=dict)
    census: Mapping[str, Any] = field(default_factory=dict)
    cost: Mapping[str, Any] = field(default_factory=dict)
    unknown_terms: tuple[str, ...] = ()
    handles: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def label(self) -> str:
        """Return ``alternative:generator:route[level]``."""
        level = "" if self.level is None else f"[{self.level!r}]"
        return f"{self.alternative}:{self.generator}:{self.route}{level}"


@dataclass(frozen=True, kw_only=True)
class CoherentSelection:
    """All coherent rows, the selected one and the reason, fixed before any phase solve.

    Attributes:
        objective: The objective used.
        controls: The limits used.
        request: The explicit request, or None.
        candidates: Every row in deterministic order.
        selected: The selected row, or None when no requested row is admitted.
        reason: Why this row, or why none.
    """

    objective: CoherentObjective
    controls: CoherentControls
    request: CoherentRequest | None
    candidates: tuple[CoherentCandidate, ...]
    selected: CoherentCandidate | None
    reason: str


@dataclass(frozen=True, kw_only=True)
class PreparedCoherentRoute:
    """The selected route after its one synthesis, ready for the M35 circuit builders.

    Attributes:
        selection: The frozen selection.
        candidate: Its selected row.
        generator: The `JointEncodingPlan` of the evolved generator, or None
            for an exact shortcut. For a commuting product, the pair
            (contraction plan of the filter, H-only joint generator).
        preparation: The `FactorizedSourcePreparation`, or None for the zero solution.
        phases: `QSPPreparedEvolution`, `QubitizedTarget` or None; the
            (filter, H) pair for a commuting product.
        evolution_plan: `SharedQSVTEvolutionPlan`, `QubitizedEvolutionPlan`
            or None; the (filter, H) pair for a commuting product.
        evolution_time: ``T_e`` for real time and the H factor of a product,
            None for the filter and shortcuts.
        recovery: zeta of the prepared route.
        physical_recovery: gamma as ``(mantissa, exponent)``, or None.
        error: Actual ``algorithmic``, ``child``, ``native``, ``evolution``
            (recovered frame) and ``physical_evolution`` terms; None is unknown.
        kernel_preparation: The selected ``analytic_prep.AnalyticKernelPreparation``
            of an analytic kernel PREP (its coefficients, MP1 probabilities,
            G22 schedule, phases and error terms), or None for a direct or MPS
            kernel factor.
    """

    selection: CoherentSelection
    candidate: CoherentCandidate
    generator: Any
    preparation: Any
    phases: Any
    evolution_plan: Any
    evolution_time: float | None
    recovery: float
    physical_recovery: tuple[float, int] | None
    error: Mapping[str, float | None]
    kernel_preparation: Any = None


class CoherentPreparationError(ValueError):
    """The selected synthesis failed; the owner exception is ``__cause__`` and no other target is tried.

    Attributes:
        candidate: Label of the selected row.
        evidence: Phase evidence of the failed attempt, when reported.
    """

    def __init__(self, message: str, *, candidate: str, evidence: Any = None):
        super().__init__(message)
        self.candidate = candidate
        self.evidence = evidence


class _CoherentPreparationLimitError(CoherentPreparationError, ResourceLimitError):
    """The selected synthesis stopped at a capacity limit; it keeps the row, the evidence and the cause."""


def _preparation_failure(message, *, candidate, cause):
    """Return the ``CoherentPreparationError`` of ``cause``, a capacity one when ``cause`` is a limit refusal."""
    kind = _CoherentPreparationLimitError if isinstance(cause, ResourceLimitError) else CoherentPreparationError
    return kind(message, candidate=candidate, evidence=getattr(cause, "evidence", None))


# Scalar relations ---------------------------------------------------------------------------------


def _phi(a: float, delta: float) -> float:
    """T13: ``Phi_a(delta) = delta + 12a^2/(3-4a^2) delta^2 + 4a^2/(3-4a^2) delta^3``."""
    c = 4.0 * a * a / (3.0 - 4.0 * a * a)
    return delta + c * delta * delta * (3.0 + delta)


def _phi_half_inverse(allowance: float) -> float:
    """Largest bisected ``delta`` with ``Phi_{1/2}(delta) <= allowance``; ``Phi_a <= Phi_{1/2}`` for ``a <= 1/2``.

    ``Phi`` increases and ``Phi(delta) >= delta``, so the root lies in
    ``[0, allowance]``; 64 halvings keep the lower end, whose value passes.
    """
    low, high = 0.0, allowance
    for _ in range(64):
        middle = 0.5 * (low + high)
        if _phi(0.5, middle) <= allowance:
            low = middle
        else:
            high = middle
    return low


def _tail_upper(bound: float | None, log_bound: float | None) -> float:
    """A float upper bound of a tail: the bound, or the least subnormal for a positive tail that underflowed."""
    if bound is not None:
        return bound
    return 0.0 if log_bound is None else math.ulp(0.0)


def _scale_value(mantissa: float, exponent: int) -> float:
    """``mantissa * 2**exponent`` as a float, infinity on overflow."""
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return math.inf


def _sum_known(values) -> int | None:
    values = tuple(values)
    return None if any(value is None for value in values) else sum(values)


def _query_cx(rows, census) -> int | None:
    """CX of the original-U calls ``(name, controls, adjoint, count)`` priced by the generator census."""
    prices = {0: census.get("forward_cx"), 1: census.get("controlled_cx")}
    return _sum_known(None if prices.get(controls) is None else count * prices[controls]
                      for _, controls, _, count in rows)


def _preparation_cx(prep, state_preparation_cx, mps_layers=None, analytic=None) -> int | None:
    """CX of P_c, P_c^dagger, the slot phase diagonal and R for one attempt (G27), as `source_prep` builds them.

    With ``mps_layers`` L each nontrivial magnitude factor on q qubits is an
    L-layer MPS circuit priced by the existing estimate ``3 L max(0, q - 1)``
    (at most q - 1 two-qubit unitaries of at most three CX per layer,
    `quantum.select_vector_preparation`; MPS_PREP_POLICY of the M7
    mathematics handoff); otherwise each is a direct magnitude tree. With an
    ``analytic`` plan (``analytic_prep.AnalyticKernelPlan``) the kernel factor
    is the analytic PREP, priced by its MP8 law at the acquisition degree n
    and the round cap K, an upper bound of the selected d <= n and K; its
    adjoint, the reversed realized circuit, has the same CX. An unknown
    multi-controlled X count leaves the cost unknown.
    """
    if state_preparation_cx is None:
        return None
    k, a = prep.kernel_qubits, prep.application_qubits

    def diagonal(qubits):
        return multiplexor_resource_law("diagonal", control_qubits=qubits)["basis_cx_gates"]

    if analytic is not None:
        kernel = analytic.census["forward"]["cx"]
        if kernel is None:
            return None
        trees = kernel + direct_preparation_cx_bound(a, complex_phases=False)
    elif mps_layers is None:
        trees = (direct_preparation_cx_bound(k, complex_phases=False)
                 + direct_preparation_cx_bound(a, complex_phases=False))
    else:
        trees = 3 * mps_layers * (max(0, k - 1) + max(0, a - 1))
    phase = diagonal(k) + diagonal(a)
    if prep.phase_recipe == "twos_complement":
        phase += len(prep.bit_phases) * diagonal(a + 1)
    elif prep.phase_recipe == "signed_magnitude":
        phase += len(prep.bit_phases) * diagonal(a + 2)
    elif prep.phase_recipe == "table":
        phase += diagonal(k + a)
    return 2 * trees + phase + integer(state_preparation_cx, "state_preparation_cx", 0)


# Rows ---------------------------------------------------------------------------------------------


def _base_allowance(objective, rest, prep, child):
    """Return ``(A0, reasons)``: E15 without the route's ``zeta epsilon_native``, or the algorithmic allowance."""
    if objective.allowance_frame == "algorithmic":
        return objective.allowance, ()
    missing = [name for name, value in (("rest_error", rest), ("native_error", objective.native_error),
                                        ("generator error", child)) if value is None]
    if missing:
        return None, (f"E15 needs {', '.join(missing)}; unknown terms make the physical allowance "
                      "unqualified",)
    if prep.dropped_bound is not None:
        rest = rest + _scale_value(*prep.dropped_bound)
    remaining = objective.allowance - rest
    if not remaining > 0:
        return None, (f"other stages {rest!r} exhaust the physical allowance {objective.allowance!r}",)
    gamma_scaled = math.ldexp(remaining / prep.recovery_mantissa, -prep.recovery_exponent)
    base = gamma_scaled - child
    if not base > 0:
        return None, (f"(epsilon_phys - Delta_rest)/gamma = {gamma_scaled!r} does not exceed the generator "
                      f"term {child!r} (E15)",)
    return base, ()


def _route_allowance(objective, base, zeta):
    native = objective.native_error
    if objective.allowance_frame == "physical_vector":
        return base - zeta * native
    return base


def _forecast(objective, prep, algorithmic, child, zeta):
    """E6 terms in the recovered frame and E8 ``gamma * epsilon_evol``; any unknown term leaves the sums unknown."""
    native = None if objective.native_error is None else zeta * objective.native_error
    parts = (algorithmic, child, native)
    evolution = None if any(part is None for part in parts) else math.fsum(parts)
    physical = None if evolution is None else evolution * _scale_value(prep.recovery_mantissa,
                                                                         prep.recovery_exponent)
    return {"algorithmic": algorithmic, "child": child, "native": native, "evolution": evolution,
            "physical_evolution": physical}


def _gamma_log2(prep) -> float:
    """``log2 gamma`` of the factorized PREP, from its exact binary exponent, without forming gamma."""
    return math.log2(prep.recovery_mantissa) + prep.recovery_exponent


def _physical_recovery(prep, zeta):
    """``Gamma = zeta gamma`` as a PhysicalScale, composed as `quantum.plan_coherent_lcu` composes the Plan's recovery."""
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery
    return compose_recovery(PhysicalScale(mantissa=prep.recovery_mantissa, exponent=prep.recovery_exponent), zeta)


def _ranked(objective, zeta, attempt, prep):
    """Return the ranking objective of one row and, for accuracy, its sufficient shots N and settings G.

    Rejection ranks ``zeta**2 C``, single circuit ``C`` and accuracy the
    sufficient quantum work ``N G C`` (AH3), N from the row's own recovery
    ``Gamma = zeta gamma``; an unknown cost, recovery or count leaves the
    objective unknown.
    """
    if objective.acquisition == "accuracy":
        target = objective.sampling
        shots = None if zeta is None or prep is None else target.shots(_physical_recovery(prep, zeta))
        value = None if attempt is None or shots is None else float(shots * target.settings * attempt)
        return {"objective": value, "shots": shots, "settings": target.settings}
    if attempt is None:
        return {"objective": None}
    return {"objective": zeta * zeta * attempt if objective.acquisition == "rejection" else float(attempt)}


def _cost(objective, zeta, rows, generator, gate_cx, prep, alternative, width, analytic=None):
    evolution = _sum_known((_query_cx(rows, generator.query_census), gate_cx))
    preparation = _preparation_cx(prep, alternative.state_preparation_cx, alternative.coefficient_mps_layers, analytic)
    # The analytic kernel PREP keeps its work register through P_c and P_c^dagger.
    width += 0 if analytic is None else analytic.work_qubits
    attempt = _sum_known((evolution, preparation))
    calls = sum(count for *_, count in rows)
    return {"evolution_cx": evolution, "preparation_cx": preparation, "attempt_cx": attempt,
            "width": width, "original_u_calls": calls, **_ranked(objective, zeta, attempt, prep),
            "gamma_log2": _gamma_log2(prep), "cx_qualification": generator.query_census.get("cx_qualification", "")}


def _unknowns(objective, alternative, cost, child):
    terms = list(_ARITHMETIC_UNKNOWN)
    if objective.native_error is None:
        terms.append("native_error")
    if child is None:
        terms.append("generator_error")
    if cost.get("attempt_cx") is None:
        terms.append("cost")
    if alternative.state_preparation_cx is None:
        terms.append("state_preparation_cx")
    return tuple(terms)


def _refused_rows(label, generator, routes, reason):
    return [CoherentCandidate(alternative=label, generator=generator, route=route, level=level,
                              status="refused", reasons=(reason,)) for route, level in routes]


def _real_time_route_levels():
    rows = [("shared_oaa", cap) for cap in SHARED_TARGET_CAPS]
    rows += [("shared_raw", cap) for cap in SHARED_TARGET_CAPS]
    rows += [("laurent", margin) for margin in GENERAL_MARGINS]
    rows += [("gqet", margin) for margin in GENERAL_MARGINS]
    return rows


def _real_time_rows(alternative, generator_label, generator, prep, evolution_time, objective, controls,
                    analytic=None):
    """Forecast every real-time transformation of one generator without solving a phase."""
    label = alternative.label
    if generator.target_operator_is_hermitian is False:
        return _refused_rows(label, generator_label, _real_time_route_levels(),
                             "the generator records a non-Hermitian target; T1 needs a Hermitian S")
    premise = ("Hermitian target recorded by construction" if generator.target_operator_is_hermitian
               else "Hermitian target is a caller assumption; the error forecast is conditional on it")
    child = None if generator.error_bound is None else evolution_time * generator.error_bound
    base, reasons = _base_allowance(objective, alternative.rest_error, prep, child)
    if base is None:
        return _refused_rows(label, generator_label, _real_time_route_levels(), reasons[0])
    tau = generator.alpha * evolution_time
    expansions = {}

    def expansion_for(allowance):
        epsilon = min(objective.tail_share * allowance, math.nextafter(1.0, 0.0))
        if epsilon not in expansions:
            try:
                expansions[epsilon] = jacobi_anger_expansion(tau, epsilon, min_degree=1, max_bytes=controls.max_bytes)
            except ResourceLimitError:
                raise
            except ValueError as error:
                expansions[epsilon] = error
        return expansions[epsilon]

    targets = {}

    def targets_for(expansion, cap):
        # The shared targets are a deterministic function of the series
        # coefficients and the cap alone, so rows whose expansions have equal
        # coefficients, such as the OAA and raw rows, form them once.
        key = (expansion.cos_coefficients, expansion.sin_coefficients, cap)
        if key not in targets:
            try:
                targets[key] = _qsp_phase_targets(expansion, cap, controls.max_bytes)
            except ResourceLimitError:
                raise
            except ValueError as error:
                targets[key] = error
        return targets[key]

    shared_lift = generator.encoded_operator_is_hermitian is not True
    walk_lift = generator.full_unitary_involution is not True
    a_u = generator.num_ancillas
    rows = []
    for route, level in _real_time_route_levels():
        common = dict(alternative=label, generator=generator_label, route=route, level=level, polynomial_time=tau)
        expansion = expansion_for(_phi_half_inverse(base) if route == "shared_oaa" else base)
        if isinstance(expansion, ValueError):
            rows.append(CoherentCandidate(status="refused", reasons=(f"degree selection: {expansion}",), **common))
            continue
        tail = _tail_upper(expansion.tail_bound, expansion.tail_log_bound)
        if route.startswith("shared"):
            rows.append(_shared_row(common, alternative, generator, prep, expansion, targets_for(expansion, level),
                                    tail, evolution_time, objective, controls, base, child, shared_lift, a_u,
                                    premise, analytic))
        else:
            rows.append(_qubitized_row(common, alternative, generator, prep, expansion, tail, evolution_time,
                                       objective, controls, base, child, walk_lift, a_u, premise, analytic))
    return rows


def _attach_analytic(row, analytic):
    """Add the analytic kernel PREP to an admitted row: its plan handle and its reasons.

    Its cost and width are already in the row (`_cost`). Its coefficient
    acquisition, MP1 probabilities and phase solve run for the selected row
    only.
    """
    if analytic is None or row.status != "admitted":
        return row
    reason = (f"analytic Low-Somma kernel PREP (M3P) on the {analytic.recipe.kind} labels: acquisition degree "
              f"{analytic.degree}, at most {analytic.max_rounds} amplification rounds and {analytic.work_qubits} "
              "work qubits kept through PREP and its inverse; its CX is the MP8 bound at that degree and round cap, "
              "an upper forecast, and its coefficients, probabilities and phases are computed for the selected row "
              "only")
    return replace(row, reasons=row.reasons + (reason,),
                   unknown_terms=row.unknown_terms + ("kernel_preparation_error",),
                   handles={**row.handles, "analytic": analytic})


def _analytic_plan(recipe, settings):
    """``analytic_prep.plan_analytic_kernel_preparation`` of one uniform recipe, with its round cap from the center-term law.

    A first scalar plan with cap 1 gives the conditional G22 count of the
    center-term lower bound ``p_q >= (1 - Delta)**2/(M mu**2)``
    (``AnalyticKernelPlan.conditional_round_bound``); the plan used has that
    count as its cap. The cap is an admission limit: a selected preparation
    whose actual K exceeds it is refused before the repetition, never
    retried. Delta >= 1 gives no such count and refuses the recipe.
    """
    from nwqlib.subroutines.block_encoding.analytic_prep import plan_analytic_kernel_preparation
    first = plan_analytic_kernel_preparation(recipe, max_rounds=1, **settings)
    if first.conditional_round_bound is None:
        raise ValueError("its approximation bound Delta >= 1 gives no center-term probability bound, so no round cap")
    if first.conditional_round_bound <= 1:
        return first
    return plan_analytic_kernel_preparation(recipe, max_rounds=first.conditional_round_bound, **settings)


def _shared_row(common, alternative, generator, prep, expansion, targets, tail, evolution_time, objective, controls,
                base, child, lift, a_u, premise, analytic=None):
    """Forecast one shared-QSVT route, with the per-parity phase tolerance its own allowance leaves after the tail.

    Both parity circuits approximate ``C/s`` and ``S/s`` with the common
    rescale s, so a shared tolerance ``eps`` per parity gives the half block
    ``B = a(F + E)``, ``a = 1/(2s)``, with
    ``||E|| <= d = tail + s(eps + eps)`` (``evolution.qsp_evolution_error_terms``).
    Raw recovery multiplies by ``zeta = 2s`` and leaves error at most d, so
    ``eps = (A - tail)/(2s)`` for the route allowance A in the recovered
    frame (`_route_allowance`). OAA is not linear in d: its error is at most
    ``Phi_a(d) <= Phi_{1/2}(d)``, so ``eps = (Phi_{1/2}^{-1}(A) - tail)/(2s)``.
    The inverse at ``a = 1/2`` is conservative, and `_phi_half_inverse` serves
    every rescale.

    The child and native terms are already out of A, and in the algorithmic
    frame A bounds only this component. ``eps`` is one step below the rounded
    quotient, which does not make the earlier arithmetic outward-rounded:
    `prepare_selected_coherent_route` recomputes the error from the actual
    residuals and compares it with A. All of this is nominal numerical
    evidence, since the residuals have no arithmetic enclosure.
    """
    amplification = "oaa" if common["route"] == "shared_oaa" else "raw"
    if isinstance(targets, ValueError):
        return CoherentCandidate(status="refused", reasons=(f"shared phase targets: {targets}",), **common)
    s, _, problems = targets
    d_c, d_s = (len(problem.coefficients) - 1 for problem in problems)
    if (d_c - d_s) % 2 == 0:
        return CoherentCandidate(status="refused", reasons=(
            f"cosine/sine degrees {d_c}/{d_s} after exact-zero trimming lack opposite parity",), **common)
    a = 0.5 / s
    zeta = 2.0 * s if amplification == "raw" else 1.0 / (3.0 * a - 4.0 * a ** 3)
    allowance = _route_allowance(objective, base, zeta)
    if not allowance > 0:
        return CoherentCandidate(status="refused", recovery=zeta, reasons=(
            f"tail {tail!r} and allowance {allowance!r}: zeta * epsilon_native consumes the allowance",), **common)
    delta_limit = allowance if amplification == "raw" else _phi_half_inverse(allowance)
    if not delta_limit > tail:
        return CoherentCandidate(status="refused", recovery=zeta, reasons=(
            f"tail {tail!r} and allowance {allowance!r}: no synthesis allowance",), **common)
    epsilon_phase = math.nextafter((delta_limit - tail) / (2.0 * s), 0.0)
    delta = tail + 2.0 * s * epsilon_phase
    algorithmic = delta if amplification == "raw" else _phi(a, delta)
    n = max(d_c, d_s) - 1
    rows = _shared_census(n, lift, amplification, 0)
    uses = 3 if amplification == "oaa" else 1
    census = {"rows": rows, "phase_slots": uses * (n + 2), "flag_toggles": 2 * uses * (n + 2),
              "full_success_reflections": uses - 1, "helper_x": _shared_helper_x(n, lift, amplification, 0)}
    base_qubits = a_u + int(lift)
    cost = _cost(objective, zeta, rows, generator, _shared_gate_cx(n, base_qubits, lift, amplification), prep,
                 alternative, base_qubits + 2 + generator.system_qubits, analytic)
    return _attach_analytic(CoherentCandidate(
        status="admitted", reasons=(premise,), signal_access="generic_lift" if lift else "direct", recovery=zeta,
        degrees={"cos": d_c, "sin": d_s, "ladder": n}, census=census, cost=cost,
        allowance={"algorithmic": allowance, "tail": expansion.epsilon, "synthesis_per_parity": epsilon_phase},
        error_forecast={"tail": tail, **_forecast(objective, prep, algorithmic, child, zeta)},
        unknown_terms=_unknowns(objective, alternative, cost, child),
        handles={"generator": generator, "preparation": prep, "expansion": expansion, "targets": targets, "scale": s,
                 "amplification": amplification, "evolution_time": evolution_time, "allowance": allowance}, **common),
        analytic)


def _qubitized_row(common, alternative, generator, prep, expansion, tail, evolution_time, objective, controls,
                   base, child, lift, a_u, premise, analytic=None):
    # T15 (Laurent) or T17 (GQET): the target, its scale and circle cap come from the phase owner's planning
    # entry, which preparation then solves unchanged.
    planner = plan_jacobi_anger_laurent if common["route"] == "laurent" else plan_jacobi_anger_gqet
    target = planner(expansion, margin=common["level"])
    zeta, degree, negative = target.scale, target.signals, target.negative_signals
    allowance = _route_allowance(objective, base, zeta)
    if not allowance > tail:
        cause = "zeta * epsilon_native consumes the allowance" if not allowance > 0 else "no synthesis allowance"
        return CoherentCandidate(status="refused", recovery=zeta, reasons=(
            f"tail {tail!r} and allowance {allowance!r}: {cause}",), **common)
    epsilon_phase = math.nextafter((allowance - tail) / zeta, 0.0)
    if not epsilon_phase > 0:
        return CoherentCandidate(status="refused", recovery=zeta, reasons=(
            "the synthesis allowance underflows",), **common)
    target = with_phase_allowance(target, epsilon_phase=epsilon_phase, max_bytes=controls.max_bytes)
    reasons = [premise]
    rows = _qubitized_census(degree, lift, 0)
    census = {"rows": rows, "walk_reflections": degree, "processing_rotations": 3 * (degree + 1) + 1,
              "helper_x": ((1, degree),) if lift and degree else ()}
    walk = a_u + int(lift)
    cost = _cost(objective, zeta, rows, generator, _qubitized_gate_cx(degree, walk, lift), prep, alternative,
                 1 + walk + generator.system_qubits, analytic)
    algorithmic = tail + zeta * epsilon_phase
    return _attach_analytic(CoherentCandidate(
        status="admitted", reasons=tuple(reasons), signal_access="generic_lift" if lift else "direct_involution",
        recovery=zeta, degrees={"signals": degree, "negative_signals": negative, "expansion": expansion.degree},
        census=census, cost=cost,
        allowance={"algorithmic": allowance, "tail": expansion.epsilon, "synthesis": epsilon_phase},
        error_forecast={"tail": tail, **_forecast(objective, prep, algorithmic, child, zeta)},
        unknown_terms=_unknowns(objective, alternative, cost, child),
        handles={"generator": generator, "preparation": prep, "target": target, "evolution_time": evolution_time,
                 "allowance": allowance}, **common), analytic)


def _imaginary_rows(alternative, generator, prep, evolution_time, objective, controls):
    """Forecast the AT9-AT13 filter for each margin; coefficients are acquired once per margin."""
    label = alternative.label
    levels = [("imaginary_gqet", margin) for margin in GENERAL_MARGINS]
    beta = evolution_time * alternative.alpha
    child = None if generator.error_bound is None else beta * generator.error_bound
    base, reasons = _base_allowance(objective, alternative.rest_error, prep, child)
    if base is None:
        return _refused_rows(label, "convex_contraction", levels, reasons[0])
    try:
        selection = select_imaginary_time_degree(beta, objective.tail_share * base)
    except ResourceLimitError:
        raise
    except ValueError as error:
        return _refused_rows(label, "convex_contraction", levels, f"filter degree selection: {error}")
    tail = _tail_upper(selection.tail_bound, selection.tail_log_bound)
    lift = generator.full_unitary_involution is not True
    walk = generator.num_ancillas + int(lift)
    rows = []
    for route, margin in levels:
        common = dict(alternative=label, generator="convex_contraction", route=route, level=margin,
                      polynomial_time=beta)
        predicted = math.nextafter(1.0 / (1.0 - margin), math.inf)
        allowance = _route_allowance(objective, base, predicted)
        if not allowance > tail:
            rows.append(CoherentCandidate(status="refused", reasons=(
                f"tail {tail!r} leaves no synthesis allowance within {allowance!r}",), **common))
            continue
        epsilon_phase = math.nextafter((allowance - tail) / predicted, 0.0)
        try:
            expansion = imaginary_time_expansion(selection, epsilon_phase=epsilon_phase, margin=margin,
                                                 max_bytes=controls.max_bytes)
        except ResourceLimitError:
            raise
        except ValueError as error:
            rows.append(CoherentCandidate(status="refused", reasons=(f"filter expansion: {error}",), **common))
            continue
        zeta = expansion.scale
        if zeta > predicted:
            rows.append(CoherentCandidate(status="refused", recovery=zeta, reasons=(
                f"stored coefficient sum gives scale {zeta!r} above the predeclared {predicted!r}; the phase "
                "allowance would exceed the filter allocation",), **common))
            continue
        degree = selection.degree
        query_rows = _qubitized_census(degree, lift, 0)
        cost = _cost(objective, zeta, query_rows, generator, _qubitized_gate_cx(degree, walk, lift), prep,
                     alternative, 1 + walk + generator.system_qubits)
        algorithmic = tail + zeta * epsilon_phase
        reasons = ["AT13 contraction-filter law; unitary OAA is not used for this filter"]
        rows.append(CoherentCandidate(
            status="admitted", reasons=tuple(reasons),
            signal_access="generic_lift" if lift else "direct_involution", recovery=zeta,
            degrees={"signals": degree, "negative_signals": 0, "expansion": degree},
            census={"rows": query_rows, "walk_reflections": degree, "processing_rotations": 3 * (degree + 1) + 1,
                    "helper_x": ((1, degree),) if lift and degree else ()},
            cost=cost, allowance={"algorithmic": allowance, "tail": objective.tail_share * base,
                                  "synthesis": epsilon_phase},
            error_forecast={"tail": tail, **_forecast(objective, prep, algorithmic, child, zeta)},
            unknown_terms=_unknowns(objective, alternative, cost, child),
            handles={"generator": generator, "preparation": prep, "expansion": expansion,
                     "contraction_error": generator.error_bound, "allowance": allowance}, **common))
    return rows


# Generators ---------------------------------------------------------------------------------------


def _preparation(applications, kernel_magnitudes, lift, arguments, *, nodes=None, recipe=None, scalar_phase,
                 alternative, controls):
    return plan_factorized_source_preparation(
        applications=applications, kernel_magnitudes=kernel_magnitudes, kernel_lift=lift,
        kernel_arguments=arguments, kernel_nodes=nodes, node_recipe=recipe, scalar_phase=scalar_phase,
        drop_allowance=alternative.drop_allowance)


def _layout_time(prep):
    """``T_e`` and the time factors ``q_a = t_a / T_e`` of the PREP application labels (None for one label)."""
    times = np.asarray(prep.application_times, dtype=float)
    evolution_time = float(np.max(times))
    if prep.application_qubits == 0 or evolution_time == 0:
        return evolution_time, None
    return evolution_time, tuple(float(t) / evolution_time for t in times)


def _exact_row(label, route, reason, prep=None, alternative=None, objective=None):
    preparation = None if prep is None else _preparation_cx(prep, alternative.state_preparation_cx,
                                                            alternative.coefficient_mps_layers)
    ranked = ({"objective": None if preparation is None else float(preparation)} if objective is None
              or objective.acquisition != "accuracy" else _ranked(objective, 1.0, preparation, prep))
    cost = {"evolution_cx": 0, "preparation_cx": preparation, "attempt_cx": preparation, "original_u_calls": 0,
            **ranked, "width": None, "gamma_log2": None if prep is None else _gamma_log2(prep)}
    return CoherentCandidate(alternative=label, generator="none", route=route, level=None, status="admitted",
                             reasons=(reason,), recovery=1.0 if route == "identity" else None, cost=cost,
                             error_forecast={"algorithmic": 0.0, "child": 0.0, "native": 0.0, "evolution": 0.0,
                                             "physical_evolution": 0.0},
                             handles={"preparation": prep})


def _active_applications(alternative):
    applications = alternative.applications
    if applications.kind not in ("homogeneous", "discrete_source"):
        raise ValueError(f"coherent selection needs discrete applications, not {applications.kind!r}; a "
                         "continuous source belongs to the sampled route")
    return tuple(item for item in applications.applications if item.mass.mantissa > 0)


def _lchs_rows(alternative, objective, controls):
    label = alternative.label
    applications = _active_applications(alternative)
    if not applications:
        return [_exact_row(label, "zero_solution", "no nonzero application: the physical solution is exactly zero")]
    kernel = alternative.kernel
    coefficients = np.asarray(kernel.coefficients, dtype=complex)
    from nwqlib.algorithms.lchs.providers import lifted_magnitudes
    magnitudes, lift = lifted_magnitudes(coefficients)
    arguments = np.angle(coefficients)
    facts = alternative.applications
    scalar_phase = (facts.dissipative_scalar + facts.signed_shift, facts.hamiltonian_scalar)
    uniform = kernel.half_width is not None
    analytic_settings = alternative.analytic_prep
    if (all(child is None for child in (alternative.l_child, alternative.h_child, alternative.a_child))
            and not alternative.omitted_part_bound):
        if analytic_settings is not None:
            return [CoherentCandidate(alternative=label, generator="none", route="identity", level=None,
                                      status="refused", reasons=(
                "the analytic kernel PREP prepares the kernel on a uniform node recipe of the evolution; an exact "
                "identity evolution has none, so use the direct coefficient PREP",))]
        try:
            prep = _preparation(applications, magnitudes, lift, arguments, nodes=kernel.nodes,
                                scalar_phase=scalar_phase, alternative=alternative, controls=controls)
        except ResourceLimitError:
            raise
        except ValueError as error:
            return [CoherentCandidate(alternative=label, generator="none", route="identity", level=None,
                                      status="refused", reasons=(f"factorized PREP: {error}",))]
        return [_exact_row(label, "identity", "every centered child is an exact zero: the evolution is the "
                           "identity, applied before any polynomial (T1 boundary)", prep, alternative, objective)]
    recipes = ()
    if uniform and (alternative.l_child is not None or alternative.a_child is not None):
        recipes = uniform_node_recipes(kernel.half_width, kernel.spacing, max_bytes=controls.max_bytes,
                                       max_work=controls.max_node_work)
    # (generator label, kind, node encoding of G12, uniform recipe that fixes the PREP labels)
    plans = []
    if alternative.l_child is not None:
        if uniform:
            plans += [(f"separate_parts/{recipe.kind}", "separate", recipe, recipe) for recipe in recipes]
        else:
            values = tuple(float(k) for k in kernel.nodes)
            plans.append(("separate_parts/table", "separate",
                          DiagonalTable(values=values, normalization=max(abs(k) for k in values)), None))
    elif alternative.h_child is not None:
        plans.append(("separate_parts/none", "separate" if len(kernel.nodes) == 1 else "inconsistent", None, None))
    if alternative.a_child is not None:
        # The single-A coefficient table uses the two's-complement labels of the rotation-table recipe.
        plans.append(("single_a/twos_complement_labels" if uniform else "single_a/table", "single_a", None,
                      recipes[0] if uniform else None))
    rows = []
    for generator_label, kind, nodes, recipe in plans:
        if kind == "inconsistent":
            rows += _refused_rows(label, generator_label, _real_time_route_levels(),
                                  "kernel nodes without a dissipative child leave the node register unused; give "
                                  "the scalar shortcut its one-node kernel")
            continue
        analytic = None
        if analytic_settings is not None:
            # The analytic PREP prepares the kernel on the labels of the evolution's own uniform recipe, so the
            # node map of PREP, phase and generator is one (MATH_TRIAGE Q3); other generators have no such recipe.
            if kind != "separate" or recipe is None:
                rows += _refused_rows(label, generator_label, _real_time_route_levels(),
                                      "the analytic kernel PREP needs the uniform Low-Somma grid and its node recipe "
                                      "in the separate-parts generator")
                continue
            try:
                analytic = _analytic_plan(recipe, analytic_settings)
            except ResourceLimitError:
                raise
            except ValueError as error:
                rows += _refused_rows(label, generator_label, _real_time_route_levels(),
                                      f"analytic kernel PREP: {error}")
                continue
        try:
            prep = _preparation(applications, magnitudes, lift, arguments,
                                nodes=None if recipe is not None else kernel.nodes, recipe=recipe,
                                scalar_phase=scalar_phase, alternative=alternative, controls=controls)
        except ResourceLimitError:
            raise
        except ValueError as error:
            rows += _refused_rows(label, generator_label, _real_time_route_levels(), f"factorized PREP: {error}")
            continue
        evolution_time, time_factors = _layout_time(prep)
        if evolution_time == 0:
            return [_exact_row(label, "identity", "T_e = 0: the evolution is the identity, applied before any "
                               "polynomial (T1 boundary)", prep, alternative, objective)]
        try:
            if kind == "single_a":
                generator = _single_a_plan(alternative.a_child, kernel, recipe, time_factors)
            else:
                generator = plan_joint_generator(nodes=nodes, l_child=alternative.l_child,
                                                 h_child=alternative.h_child, time_factors=time_factors)
        except ResourceLimitError:
            raise
        except (TypeError, ValueError) as error:
            rows += _refused_rows(label, generator_label, _real_time_route_levels(), f"generator plan: {error}")
            continue
        if alternative.omitted_part_bound:
            generator = replace(generator, error_bound=None if generator.error_bound is None
                                else generator.error_bound + alternative.omitted_part_bound,
                                error_terms={**generator.error_terms, "omitted_part": alternative.omitted_part_bound})
        rows += _real_time_rows(alternative, generator_label, generator, prep, evolution_time, objective,
                                controls, analytic)
    for route, reason in NOT_ADMITTED_ROUTES.items():
        rows.append(CoherentCandidate(alternative=label, generator="none", route=route, level=None,
                                      status="refused", reasons=(reason,)))
    if alternative.l_child is not None and alternative.h_child is not None:
        rows.append(CoherentCandidate(
            alternative=label, generator="none", route="commuting_product", level=None, status="unknown",
            reasons=("the kernel sum does not use [L, H] = 0; the product rows belong to a "
                     "CommutingProductAlternative, which needs an exact commutation fact of the representation "
                     "and is absent otherwise; no commutator is computed",)))
    return rows


def _single_a_plan(a_child, kernel, recipe, time_factors):
    """G13 on the PREP labels: ``c_j = k_j - i`` with padding 0, so ``r = sqrt(R^2 + 1)`` bounds the completion."""
    nodes = np.asarray(kernel.nodes, dtype=float)
    if recipe is not None:
        labels = np.asarray(uniform_node_labels(recipe))
        width = recipe.address_qubits
    else:
        labels = np.arange(nodes.size)
        width = (nodes.size - 1).bit_length() if nodes.size > 1 else 0
    values = np.zeros(1 << width, dtype=complex)
    values[labels] = nodes - 1j
    radius = float(np.nextafter(math.hypot(float(np.max(np.abs(nodes))), 1.0), np.inf))
    return plan_single_a_generator(a_child=a_child, node_coefficients=values, radius=radius,
                                   time_factors=time_factors)


def _imaginary_alternative_rows(alternative, objective, controls):
    label = alternative.label
    facts = alternative.applications
    applications = _active_applications(alternative)
    if not applications:
        return [_exact_row(label, "zero_solution", "no nonzero application: the physical solution is exactly zero")]
    alpha = finite_real(alternative.alpha, "alpha")
    if facts.signed_shift != alpha - facts.dissipative_scalar:
        raise ValueError("imaginary-time masses must use signed_shift = alpha - c_L so that they carry kappa_a")
    levels = [("imaginary_gqet", margin) for margin in GENERAL_MARGINS]
    try:
        prep = _preparation(applications, np.ones(1), 0, np.zeros(1), nodes=(0.0,),
                            scalar_phase=(0.0, facts.hamiltonian_scalar), alternative=alternative,
                            controls=controls)
    except ResourceLimitError:
        raise
    except ValueError as error:
        return _refused_rows(label, "convex_contraction", levels, f"factorized PREP: {error}")
    evolution_time, time_factors = _layout_time(prep)
    if evolution_time == 0:
        return [_exact_row(label, "identity", "T_e = 0: the filter is the identity (AT9 at beta = 0)", prep,
                           alternative, objective)]
    try:
        generator = plan_convex_contraction(l_child=alternative.l_child, alpha=alpha, time_factors=time_factors)
    except ResourceLimitError:
        raise
    except (TypeError, ValueError) as error:
        return _refused_rows(label, "convex_contraction", levels, f"AT12 contraction plan: {error}")
    if generator.target_operator_is_hermitian is False:
        return _refused_rows(label, "convex_contraction", levels, "AT13 needs a Hermitian L_0")
    return _imaginary_rows(alternative, generator, prep, evolution_time, objective, controls)


def _factor_terms(row):
    """``(algorithmic, child, evolution)`` forecast of one factor row in its recovered frame."""
    forecast = row.error_forecast
    return forecast.get("algorithmic"), forecast.get("child"), forecast.get("evolution")


def _composed(first, second):
    """M7.6: ``e_1 + e_2 + e_1 e_2``, unknown when either term is."""
    return None if first is None or second is None else first + second + first * second


def _composed_child(alg_i, child_i, alg_h, child_h):
    """The child share of the product's M7.6 bound, so that its algorithmic and child terms sum to the whole bound.

    Each factor's error is ``e = a + c`` (algorithmic plus child), and M7.6
    bounds the product by ``e_I + e_H + e_I e_H``. The algorithmic term is
    M7.6 of the algorithmic parts alone, ``a_I + a_H + a_I a_H``, and the
    child term is the rest, ``c_I (1 + a_H + c_H) + c_H (1 + a_I)``, which
    keeps the cross products of a child error with the other factor.
    Unknown when any term is.
    """
    if None in (alg_i, child_i, alg_h, child_h):
        return None
    return child_i * (1.0 + alg_h + child_h) + child_h * (1.0 + alg_i)


def _product_row(label, prep, objective, filter_row, h_row, system_qubits, budget):
    """Compose one admitted filter row and one admitted H row into a product row (M7.6), PREP counted once."""
    zeta = filter_row.recovery * h_row.recovery
    evolution = _sum_known((filter_row.cost.get("evolution_cx"), h_row.cost.get("evolution_cx")))
    preparation = filter_row.cost.get("preparation_cx")
    attempt = _sum_known((evolution, preparation))
    (alg_i, child_i, evol_i), (alg_h, child_h, evol_h) = _factor_terms(filter_row), _factor_terms(h_row)
    evolution_error = _composed(evol_i, evol_h)
    cost = {"evolution_cx": evolution, "preparation_cx": preparation, "attempt_cx": attempt,
            "width": filter_row.cost["width"] + h_row.cost["width"] - system_qubits,
            "original_u_calls": filter_row.cost["original_u_calls"] + h_row.cost["original_u_calls"],
            **_ranked(objective, zeta, attempt, prep), "gamma_log2": _gamma_log2(prep),
            "cx_qualification": h_row.cost.get("cx_qualification", "")}
    return CoherentCandidate(
        alternative=label, generator=f"filter[{filter_row.level!r}]*{h_row.route}[{h_row.level!r}]",
        route="commuting_product", level=None, status="admitted",
        reasons=(f"exact [L, H] = 0 from {budget['commutation']}; each factor's algorithmic allowance is "
                 f"b = {budget['factor']!r} of B = {budget['product']!r} (2b + b**2 = B), and the factor errors compose "
                 "as e_I + e_H + e_I e_H (M7.6)", *filter_row.reasons, *h_row.reasons),
        signal_access=f"{filter_row.signal_access}*{h_row.signal_access}", recovery=zeta,
        polynomial_time=None, degrees={"filter": dict(filter_row.degrees), "hamiltonian": dict(h_row.degrees)},
        allowance={"algorithmic": budget["product"], "factor": budget["factor"]},
        error_forecast={"algorithmic": _composed(alg_i, alg_h),
                        "child": _composed_child(alg_i, child_i, alg_h, child_h),
                        "native": None, "evolution": evolution_error,
                        "physical_evolution": None if evolution_error is None else evolution_error * _scale_value(
                            prep.recovery_mantissa, prep.recovery_exponent)},
        census={"filter": filter_row.census, "hamiltonian": h_row.census,
                "factors": tuple((item.route, item.level, item.signal_access, dict(item.degrees))
                                 for item in (filter_row, h_row))}, cost=cost,
        unknown_terms=tuple(dict.fromkeys(filter_row.unknown_terms + h_row.unknown_terms)),
        handles={"factors": (filter_row, h_row), "preparation": prep, "allowance": budget["product"]})


def _product_alternative_rows(alternative, objective, controls):
    """Rows of the commuting product: every admitted filter margin with every admitted H transformation."""
    from nwqlib.subroutines.block_encoding.joint import plan_joint_generator
    label = alternative.label
    facts = alternative.applications
    applications = _active_applications(alternative)
    if not applications:
        return [_exact_row(label, "zero_solution", "no nonzero application: the physical solution is exactly zero")]
    alpha = finite_real(alternative.alpha, "alpha")
    if facts.signed_shift != alpha - facts.dissipative_scalar:
        raise ValueError("commuting-product masses must use signed_shift = alpha - c_L so that they carry kappa_a")
    if objective.allowance_frame != "algorithmic":
        return [CoherentCandidate(alternative=label, generator="none", route="commuting_product", level=None,
                                  status="refused", reasons=(
            "the product split is implemented in the algorithmic frame; in the physical frame each factor's child "
            "and native terms must also fit its share (M7.7)",))]
    try:
        prep = _preparation(applications, np.ones(1), 0, np.zeros(1), nodes=(0.0,),
                            scalar_phase=(0.0, facts.hamiltonian_scalar), alternative=alternative,
                            controls=controls)
    except ResourceLimitError:
        raise
    except ValueError as error:
        return [CoherentCandidate(alternative=label, generator="none", route="commuting_product", level=None,
                                  status="refused", reasons=(f"factorized PREP: {error}",))]
    evolution_time, time_factors = _layout_time(prep)
    if evolution_time == 0:
        return [_exact_row(label, "identity", "T_e = 0: both factors are the identity", prep, alternative, objective)]
    try:
        filter_generator = plan_convex_contraction(l_child=alternative.l_child, alpha=alpha, time_factors=time_factors)
        h_generator = plan_joint_generator(nodes=None, l_child=None, h_child=alternative.h_child,
                                           time_factors=time_factors)
    except ResourceLimitError:
        raise
    except (TypeError, ValueError) as error:
        return [CoherentCandidate(alternative=label, generator="none", route="commuting_product", level=None,
                                  status="refused", reasons=(f"factor generator plan: {error}",))]
    product = objective.allowance
    factor = product / (math.sqrt(1.0 + product) + 1.0)
    budget = {"product": product, "factor": factor, "commutation": alternative.commutation}
    share = replace(objective, allowance=factor)
    filters = [row for row in _imaginary_rows(alternative, filter_generator, prep, evolution_time, share,
                                              controls) if row.status == "admitted"]
    hamiltonians = [row for row in _real_time_rows(alternative, "separate_parts/none", h_generator, prep,
                                                   evolution_time, share, controls)
                    if row.status == "admitted"]
    if not filters or not hamiltonians:
        missing = "filter" if not filters else "H evolution"
        return [CoherentCandidate(alternative=label, generator="none", route="commuting_product", level=None,
                                  status="refused", reasons=(f"no admitted {missing} factor at the factor allowance "
                                                             f"{factor!r}",))]
    rows = []
    for filter_row in filters:
        for h_row in hamiltonians:
            row = _product_row(label, prep, objective, filter_row, h_row, filter_generator.system_qubits, budget)
            row = replace(row, handles={**row.handles, "generators": (filter_generator, h_generator),
                                        "evolution_time": evolution_time})
            rows.append(row)
    return rows


# Selection ----------------------------------------------------------------------------------------


def _matches(candidate, request):
    if request is None:
        return True
    joint, _, node = candidate.generator.partition("/")
    return ((request.route is None or candidate.route == request.route)
            and (request.level is None or candidate.level == request.level)
            and (request.joint is None or joint == request.joint)
            and (request.node is None or node == request.node)
            and (request.alternative is None or candidate.alternative == request.alternative))


def _phase_degree(row):
    """Total degree of the phase polynomials a row synthesizes, a classical cost its quantum objective omits."""
    def total(degrees):
        return degrees["cos"] + degrees["sin"] if "cos" in degrees else degrees.get("signals", 0)
    degrees = row.degrees
    return total(degrees["filter"]) + total(degrees["hamiltonian"]) if "filter" in degrees else total(degrees)


def _squared_ratio(numerator, denominator):
    """``(numerator/denominator)**2`` of two PhysicalScale values, as a PhysicalScale that binary64 need not hold."""
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery
    mantissas = (numerator.mantissa, denominator.mantissa)
    return compose_recovery(mantissas, mantissas,
                            PhysicalScale(mantissa=0.5, exponent=2 * (numerator.exponent - denominator.exponent) + 1))


def _scale_text(scale):
    """The unrounded decimal of a PhysicalScale, or its binary form when binary64 cannot hold it."""
    value = scale.as_float()
    return f"{scale.mantissa!r}*2**{scale.exponent}" if value is None else repr(value)


def _fixed_shot_consequence(readout, gamma, smallest):
    """The precision consequence of recovery ``gamma`` against the smallest ``smallest`` under fixed shots, or None.

    ``gamma`` and ``smallest`` are the PhysicalScale recoveries ``Gamma = zeta
    gamma`` of the two rows, fixed before independent stationary shots. At
    fixed shots and statistical model the Hoeffding radius of a moment scales
    exactly as ``Gamma**2``. For the same absolute radius, failure probability,
    observable range and setting allocation, the unrounded sufficient count
    scales as ``Gamma**4``, and the integer ceiling prevents an exact ratio of
    final counts. The accepted probability of a success-conditional output is
    ``p = q/Gamma**2`` with ``q`` the squared norm of the approximate vector, so
    it scales as ``1/Gamma**2`` only if ``q`` is the same, and different kernel
    or filter approximations can change ``q``. These control sampling about the
    finite circuit's moment, not its error against the exact solution.
    """
    if readout == "unconditional_counts":
        return (f"at equal shots the Hoeffding radius of the moment is proportional to Gamma**2 = (zeta gamma)**2, so "
                f"Gamma {_scale_text(gamma)} against the smallest {_scale_text(smallest)} widens it by "
                f"{_scale_text(_squared_ratio(gamma, smallest))}, and for one radius, failure probability, observable "
                "range and settings the sufficient shots before rounding up grow by the square of that factor")
    if readout == "conditional_counts":
        return (f"the accepted probability is q/Gamma**2 for the squared norm q of the circuit's approximate vector, so "
                f"at equal q, Gamma {_scale_text(gamma)} against the smallest {_scale_text(smallest)} keeps "
                f"{_scale_text(_squared_ratio(smallest, gamma))} of the accepted shots, and different kernels or filters "
                "can change q")
    return None


def _row_recovery(row):
    """``Gamma = zeta gamma`` of a row with a factorized PREP, as a PhysicalScale."""
    return _physical_recovery(row.handles["preparation"], row.recovery)


def row_on_grid(label, grid):
    """Return a row's label, followed by the frontier entry of its grid when one planning pass compares several grids."""
    return label if grid is None else f"{label} on grid {grid}"


def smallest_recovery_row(selection, incumbent=None, grid=None):
    """Return ``(Gamma, label)`` of the admitted requested row of smallest recovery, starting from ``incumbent``.

    A row without a recovery or a factorized PREP, such as an exact shortcut,
    has no ``Gamma`` to compare. ``incumbent`` is the result for an earlier
    selection of the same planning pass, or None, and ``grid`` names this
    selection's frontier entry in the label (`row_on_grid`). Different rows and
    grids have different finite targets, so this compares their actual
    recoveries, not their errors. A normalized mantissa orders a PhysicalScale
    by ``(exponent, mantissa)``.
    """
    best = incumbent
    for row in selection.candidates:
        if row.status == "admitted" and row.requested and row.recovery is not None and row.handles.get("preparation"):
            gamma = _row_recovery(row)
            if best is None or (gamma.exponent, gamma.mantissa) < (best[0].exponent, best[0].mantissa):
                best = (gamma, row_on_grid(row.label, grid))
    return best


def _choose(rows, objective):
    """Return ``(row, reason)`` among one alternative's admitted requested rows."""
    exact = [row for row in rows if row.route in _EXACT_ROUTES and row.status == "admitted" and row.requested]
    accuracy = objective.acquisition == "accuracy"
    if exact and not (accuracy and exact[0].cost.get("shots") is None):
        return exact[0], "exact shortcut: " + exact[0].reasons[0]
    pool = [(index, row) for index, row in enumerate(rows) if row.status == "admitted" and row.requested]
    # A row whose sufficient count has no finite value within the exact integer population cannot meet the
    # requested radius; it stays in the table with its forecast.
    uncounted = [row.label for _, row in pool if accuracy and row.cost.get("shots") is None]
    pool = [(index, row) for index, row in pool if not accuracy or row.cost.get("shots") is not None]
    if not pool:
        requested = [row for row in rows if row.requested]
        if uncounted:
            details = (f"{len(uncounted)} admitted rows (first: {uncounted[0]}) have no sufficient count of shots per "
                       "setting that a saved Plan can hold for the requested radius")
        elif requested:
            # Each distinct first reason once, in row order, with its rows: many rows can share one cause, such as
            # their node recipe, which would otherwise hide a different binding cause of later rows.
            causes = {}
            for row in requested:
                first, count = causes.get(row.reasons[0], (row.label, 0))
                causes[row.reasons[0]] = (first, count + 1)
            shown = list(causes.items())[:6]
            details = "; ".join(f"{first} and {count - 1} other rows: {reason}" if count > 1 else f"{first}: {reason}"
                                for reason, (first, count) in shown)
            if len(causes) > len(shown):
                details += f"; {len(causes) - len(shown)} other distinct reasons"
        else:
            details = "no row matches the request"
        return None, f"no admitted requested row ({details})"
    metric = {"rejection": "zeta**2 * attempt CX", "single_circuit": "attempt CX",
              "accuracy": "sufficient quantum work N G C (AH3)"}[objective.acquisition]
    known = [(index, row) for index, row in pool if row.cost.get("objective") is not None]
    unknown = [row.label for _, row in pool if row.cost.get("objective") is None]
    if known:
        best = min(row.cost["objective"] for _, row in known)
        tied = [(index, row) for index, row in known if row.cost["objective"] == best]
        # Exact ties on the quantum objective go to the lowest total phase degree. Remaining ties keep row order.
        index, row = min(tied, key=lambda item: (_phase_degree(item[1]), item[0]))
        reason = (f"minimum {metric} = {best!r} among {len(known)} admitted rows of one finite "
                  "target with known cost (R2/T26)")
        if accuracy:
            reason += (f"; N = {row.cost['shots']} shots on each of {row.cost['settings']} settings, the "
                       "sufficient count of the row's recovery; the CX are upper bounds, so this is a "
                       "sufficient-work forecast, not a certified minimum of the realized cost")
        if len(tied) > 1:
            reason += (f"; {len(tied)} rows tie on it and the lowest total phase degree decides ("
                       + ", ".join(f"{other.label}: zeta {other.recovery!r}, phase degree "
                                   f"{_phase_degree(other)}" for _, other in tied) + ")")
            smallest = min(other.recovery for _, other in tied)
            consequence = (_fixed_shot_consequence(objective.readout, _row_recovery(row), _row_recovery(
                               next(other for _, other in tied if other.recovery == smallest)))
                           if objective.acquisition == "single_circuit" and row.recovery > smallest else None)
            if consequence:
                reason += f"; equal attempt CX is not equal precision: {consequence}"
        if unknown:
            reason += (f"; {len(unknown)} admitted rows of unknown cost (first: {unknown[0]}) were not ranked, so "
                       "this is not a certified minimum over them")
        if uncounted:
            reason += f"; {len(uncounted)} rows without a sufficient count were not eligible"
        return row, reason
    # Without known costs the original-U calls (with zeta for rejection and accuracy) stand in for the quantum
    # objective; rows equal on that stand-in go to the lowest total phase degree, as exact objective ties do.
    if objective.acquisition in ("rejection", "accuracy"):
        index, row = min(pool, key=lambda item: (item[1].recovery, item[1].cost["original_u_calls"],
                                                 _phase_degree(item[1]), item[0]))
        rule = "smallest recovery zeta, then fewest original-U calls, then the lowest total phase degree"
    else:
        index, row = min(pool, key=lambda item: (item[1].cost["original_u_calls"], _phase_degree(item[1]),
                                                 item[1].recovery, item[0]))
        rule = "fewest original-U calls, then the lowest total phase degree, then smallest recovery zeta"
        smallest = min(other.recovery for _, other in pool)
        consequence = (_fixed_shot_consequence(objective.readout, _row_recovery(row), _row_recovery(
                           next(other for _, other in pool if other.recovery == smallest)))
                       if row.recovery > smallest else None)
        if consequence:
            rule += f" ({consequence})"
    reason = (f"cost unknown for every admitted row (first: {unknown[0]}); deterministic choice by {rule}; "
              "not a certified cost minimum")
    if accuracy:
        reason += f"; N = {row.cost['shots']} shots on each of {row.cost['settings']} settings"
    return row, reason


def _nominal_log2(row, objective):
    """``log2`` of the nominal cross-alternative objective of a chosen row, or None when unknown.

    Rejection compares ``J = C Gamma**2 = C (zeta gamma)**2`` (M7.8, PLAN
    M0.1): different alternatives have different finite targets and so
    different gamma. Single-circuit acquisition keeps every shot and compares
    the cost C of one circuit, the fixed-acquisition comparison of the M7
    mathematics handoff (Q6). Both are nominal: a strict order needs every
    alternative's total physical error against one reference solution and a
    positive lower bound on its norm (T25, M7.9), which are not available.
    """
    value = row.cost.get("objective")
    if value is None:
        return None
    base = math.log2(value) if value > 0 else -math.inf
    if objective.acquisition != "rejection":
        return base
    gamma = row.cost.get("gamma_log2")
    return None if gamma is None else base + 2.0 * gamma


def _choose_alternative(winners, objective):
    """Return ``(row, reason)`` across the chosen rows of several alternatives by the nominal model."""
    model = {"rejection": "C (zeta gamma)**2", "single_circuit": "attempt CX",
             "accuracy": "sufficient quantum work N G C"}[objective.acquisition]
    known = [(index, row, why, _nominal_log2(row, objective)) for index, (row, why) in enumerate(winners)]
    unknown = [row.alternative for _, row, _, value in known if value is None]
    known = [item for item in known if item[3] is not None]
    if not known:
        row, why = winners[0]
        return row, (f"{row.alternative}: {why}; no alternative has a known {model}, so the caller's "
                     "alternative order decides")
    index, row, why, best = min(known, key=lambda item: (item[3], item[0]))
    listed = ", ".join(f"{other.alternative} log2 {value!r}" for _, other, _, value in known)
    if objective.acquisition == "accuracy":
        reason = (f"{row.alternative}: {why}; across alternatives the {model} (AH3) is smallest ({listed}); each N "
                  "already contains its own (zeta gamma)**4, so this compares sufficient-work forecasts at the "
                  "requested radius, not total errors against one reference solution")
    else:
        reason = (f"{row.alternative}: {why}; across alternatives the nominal {model} (M7.8) is smallest ({listed}); "
                  "this is not a strict order, which needs the total errors against one reference solution and a "
                  "positive norm lower bound (T25)")
    if objective.acquisition == "single_circuit" and objective.readout in ("unconditional_counts", "conditional_counts"):
        reason += ("; under fixed shots the range-based confidence radius of a moment scales as (zeta gamma)**2, "
                   "which this cost comparison does not include")
    if unknown:
        reason += f"; alternatives of unknown nominal cost were not ranked ({', '.join(unknown)})"
    return row, reason


def select_coherent_route(alternatives: Sequence[Any], *, objective: CoherentObjective,
                          controls: CoherentControls | None = None,
                          request: CoherentRequest | None = None) -> CoherentSelection:
    """Choose the coherent node/joint generator and transformation once, by scalar work only.

    Args:
        alternatives: `LCHSCoherentAlternative`, `ImaginaryTimeAlternative`
            and `CommutingProductAlternative` records in the caller's order
            of preference.
        objective (CoherentObjective): Acquisition, error frame and allocation.
        controls (CoherentControls | None): Default ``CoherentControls()``.
        request (CoherentRequest | None): Explicit constraints, default None.

    Returns:
        selection (CoherentSelection): Every row, the selected row or None,
        and the reason.

    Raises:
        ValueError: For an invalid objective, an empty or duplicated
            alternative list, a continuous-source application record, or
            imaginary-time masses without ``kappa_a``.
        ResourceLimitError: If evaluating any row needs more bytes than
            ``controls.max_bytes`` (None: the physical memory) or more work
            than ``controls.max_node_work``. Rows are refused only for
            mathematical or structural reasons, so every computer with
            enough memory compares the same rows.
    """
    controls = controls or CoherentControls()
    if objective.acquisition not in ("rejection", "single_circuit", "accuracy"):
        raise ValueError("acquisition must be 'rejection', 'single_circuit' or 'accuracy'; other estimators need "
                         "their own law")
    if (objective.acquisition == "accuracy") != isinstance(objective.sampling, SamplingTarget):
        raise ValueError("an 'accuracy' acquisition needs its SamplingTarget, and only it takes one")
    if objective.allowance_frame not in ("physical_vector", "algorithmic"):
        raise ValueError("allowance_frame must be 'physical_vector' or 'algorithmic'")
    if not finite_real(objective.allowance, "allowance") > 0:
        raise ValueError("allowance must be positive")
    if not 0 < finite_real(objective.tail_share, "tail_share") < 1:
        raise ValueError("tail_share must lie strictly between zero and one")
    if objective.native_error is not None and finite_real(objective.native_error, "native_error") < 0:
        raise ValueError("native_error must be nonnegative")
    labels = [alternative.label for alternative in alternatives]
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("alternatives must be a nonempty sequence with distinct labels")
    for alternative in alternatives:
        if alternative.rest_error is not None and finite_real(alternative.rest_error, "rest_error") < 0:
            raise ValueError("rest_error must be nonnegative")
    candidates = []
    winners, notes = [], []
    for alternative in alternatives:
        if isinstance(alternative, ImaginaryTimeAlternative):
            rows = _imaginary_alternative_rows(alternative, objective, controls)
        elif isinstance(alternative, CommutingProductAlternative):
            rows = _product_alternative_rows(alternative, objective, controls)
        else:
            rows = _lchs_rows(alternative, objective, controls)
        # An exact shortcut answers any transformation request of its alternative: no approximation remains.
        rows = [row if _matches(row, request) or (
                    row.route in _EXACT_ROUTES and (request is None or request.alternative in (None, row.alternative)))
                else replace(row, requested=False) for row in rows]
        candidates += rows
        row, why = _choose(rows, objective)
        if row is None:
            notes.append(f"{alternative.label}: {why}")
        else:
            winners.append((row, why))
    if not winners:
        chosen, reason = None, "no admitted requested coherent row; " + " | ".join(notes)
    elif len(winners) == 1:
        chosen, reason = winners[0][0], f"{winners[0][0].alternative}: {winners[0][1]}"
    else:
        chosen, reason = _choose_alternative(winners, objective)
    if winners and notes:
        reason += "; " + " | ".join(notes)
    return CoherentSelection(objective=objective, controls=controls, request=request, candidates=tuple(candidates),
                             selected=chosen, reason=reason)


# Preparation --------------------------------------------------------------------------------------


def _prepare_factor(row, controls, objective):
    """Synthesize one transformation row; return ``(phases, plan, algorithmic, child)``.

    The same expansion, cap or margin, tolerance and memory limit as at
    selection. A shared route solves the phase targets that its row formed
    at selection (_shared_row). Owner failures propagate as their ValueError.
    """
    handles = row.handles
    generator, evolution_time = handles["generator"], handles.get("evolution_time")
    if row.route.startswith("shared"):
        phases = _solve_qsp_targets(handles["expansion"], handles["targets"], row.level,
                                    residual_tolerance=row.allowance["synthesis_per_parity"],
                                    max_bytes=controls.max_bytes)
        plan = plan_shared_qsvt_evolution(generator, phases, evolution_time=evolution_time,
                                          amplification=handles["amplification"])
        return phases, plan, plan.algorithmic_error_bound, plan.child_error
    if row.route in ("laurent", "gqet"):
        phases = prepare_qubitized_target(handles["target"])
        plan = plan_qubitized_evolution(generator, phases, evolution_time=evolution_time)
        return phases, plan, plan.algorithmic_error_bound, plan.child_error
    filtered = prepare_imaginary_time_filter(handles["expansion"])
    phases = imaginary_time_target(filtered)
    plan = plan_qubitized_evolution(generator, phases)
    native = 0.0 if objective.native_error is None else objective.native_error
    terms = imaginary_time_filter_error_terms(phases, contraction_error=handles["contraction_error"],
                                              native_error=native)
    return phases, plan, _sum_known((terms["tail"], terms["synthesis"])), terms["child"]


def prepare_selected_coherent_route(selection: CoherentSelection) -> PreparedCoherentRoute:
    """Synthesize only the selected target with its frozen expansion, level, tolerance and limits.

    A commuting-product row synthesizes its filter factor and then its H
    factor, each once with its frozen factor allowance; the prepared
    ``generator``, ``phases`` and ``evolution_plan`` are then the pairs
    ``(filter, H)``, the recovery is their product and the error terms are
    composed by M7.6.

    Args:
        selection (CoherentSelection): From `select_coherent_route`.

    Returns:
        prepared (PreparedCoherentRoute): Phases, the M35 plan and actual
        error terms.

    Raises:
        ValueError: When the selection has no selected row.
        CoherentPreparationError: When the selected synthesis fails or its
            actual error exceeds the frozen allowance; no other row is tried.
    """
    row = selection.selected
    if row is None:
        raise ValueError(f"no coherent row was selected: {selection.reason}")
    handles = row.handles
    prep = handles.get("preparation")
    gamma = None if prep is None else (prep.recovery_mantissa, prep.recovery_exponent)
    if row.route in _EXACT_ROUTES:
        return PreparedCoherentRoute(selection=selection, candidate=row, generator=None, preparation=prep,
                                     phases=None, evolution_plan=None, evolution_time=None, recovery=1.0,
                                     physical_recovery=gamma, error=dict(row.error_forecast))
    controls, objective = selection.controls, selection.objective
    factors = handles["factors"] if row.route == "commuting_product" else (row,)
    prepared = []
    kernel_preparation = None
    if handles.get("analytic") is not None:
        # The analytic kernel PREP of the selected row: one coefficient acquisition, one MP1 pass, the G22
        # schedule and one phase solve (analytic_prep.prepare_analytic_kernel), before the evolution phases.
        from nwqlib.subroutines.block_encoding.analytic_prep import prepare_analytic_kernel
        try:
            kernel_preparation = prepare_analytic_kernel(handles["analytic"])
        except ValueError as error:
            raise _preparation_failure(f"selected coherent row {row.label} failed in its analytic kernel PREP: {error}",
                                       candidate=row.label, cause=error) from error
    for factor in factors:
        try:
            result = _prepare_factor(factor, controls, objective)
        except ValueError as error:
            raise _preparation_failure(f"selected coherent row {row.label} failed: {error}", candidate=row.label,
                                       cause=error) from error
        # None stays UNKNOWN (for example a positive tail below binary64 range in the owner's record).
        if result[2] is not None and not result[2] <= factor.handles["allowance"]:
            raise CoherentPreparationError(
                f"selected coherent row {row.label} realized algorithmic error {result[2]!r} against its allowance "
                f"{factor.handles['allowance']!r}", candidate=row.label)
        prepared.append(result)
    if len(prepared) == 1:
        phases, plan, algorithmic, child = prepared[0]
        generator, recovery = handles["generator"], plan.recovery
        evolution_time = None if row.route == "imaginary_gqet" else handles.get("evolution_time")
        error_terms = _forecast(objective, prep, algorithmic, child, recovery)
    else:
        (filter_phases, filter_plan, alg_i, child_i), (h_phases, h_plan, alg_h, child_h) = prepared
        phases, plan, generator = (filter_phases, h_phases), (filter_plan, h_plan), handles["generators"]
        recovery, evolution_time = filter_plan.recovery * h_plan.recovery, handles["evolution_time"]
        factor_errors = [_forecast(objective, prep, alg, child, factor_plan.recovery)
                         for alg, child, factor_plan in ((alg_i, child_i, filter_plan), (alg_h, child_h, h_plan))]
        evolution = _composed(factor_errors[0]["evolution"], factor_errors[1]["evolution"])
        error_terms = {"algorithmic": _composed(alg_i, alg_h),
                       "child": _composed_child(alg_i, child_i, alg_h, child_h),
                       "native": None if objective.native_error is None else recovery * objective.native_error,
                       "evolution": evolution,
                       "physical_evolution": None if evolution is None else evolution * _scale_value(*gamma)}
        if error_terms["algorithmic"] is not None and not error_terms["algorithmic"] <= handles["allowance"]:
            raise CoherentPreparationError(
                f"selected coherent row {row.label} realized algorithmic error {error_terms['algorithmic']!r} "
                f"against its allowance {handles['allowance']!r}", candidate=row.label)
    return PreparedCoherentRoute(
        selection=selection, candidate=row, generator=generator, preparation=prep, phases=phases,
        evolution_plan=plan, evolution_time=evolution_time, recovery=recovery, physical_recovery=gamma,
        error=error_terms, kernel_preparation=kernel_preparation)


class EvolutionFactor(NamedTuple):
    """One evolution of a prepared route, as the circuit, cost and sampling consumers read it."""

    generator: Any
    phases: Any
    plan: Any
    census: Mapping[str, Any]
    route: str
    level: float | None
    signal_access: str | None
    degrees: Mapping[str, Any]
    evolution_time: float | None


def evolution_factors(prepared: PreparedCoherentRoute) -> tuple[EvolutionFactor, ...]:
    """Return the evolutions of a prepared route: none for an exact shortcut, one, or (filter, H) for a product.

    The product's factors keep their own generators, phases, plans,
    censuses and transformation facts; its filter has no evolution time and
    its H factor evolves for ``T_e``.
    """
    if prepared.evolution_plan is None:
        return ()
    row = prepared.candidate
    if row.route != "commuting_product":
        return (EvolutionFactor(prepared.generator, prepared.phases, prepared.evolution_plan, row.census, row.route,
                                row.level, row.signal_access, row.degrees, prepared.evolution_time),)
    census = row.census
    return tuple(EvolutionFactor(generator, phases, plan, census[key], route, level, access, degrees, time)
                 for generator, phases, plan, key, (route, level, access, degrees), time in zip(
                     prepared.generator, prepared.phases, prepared.evolution_plan, ("filter", "hamiltonian"),
                     census["factors"], (None, prepared.evolution_time), strict=True))


def without_preparation_handles(prepared: PreparedCoherentRoute) -> PreparedCoherentRoute:
    """Return ``prepared`` with every row's ``handles`` emptied, the form a Plan keeps.

    The handles hold the expansions, generator plans and PREP plans of every
    row so that `prepare_selected_coherent_route` can synthesize the selected
    one. After that synthesis the circuit builders read ``generator``,
    ``preparation``, ``phases`` and ``evolution_plan`` of the prepared route,
    so the handles of all rows only enlarge the saved Plan. Rows keep their
    status, reasons, forecasts and costs.
    """
    rows = tuple(replace(row, handles={}) for row in prepared.selection.candidates)
    selected = replace(prepared.candidate, handles={})
    selection = replace(prepared.selection, candidates=rows, selected=selected)
    return replace(prepared, selection=selection, candidate=selected)


__all__ = [
    "GENERAL_MARGINS",
    "NOT_ADMITTED_ROUTES",
    "SHARED_TARGET_CAPS",
    "CoherentCandidate",
    "CoherentControls",
    "CoherentObjective",
    "CoherentPreparationError",
    "CoherentRequest",
    "CoherentSelection",
    "CommutingProductAlternative",
    "EvolutionFactor",
    "ImaginaryTimeAlternative",
    "KernelGrid",
    "LCHSCoherentAlternative",
    "PreparedCoherentRoute",
    "SamplingTarget",
    "evolution_factors",
    "prepare_selected_coherent_route",
    "row_on_grid",
    "select_coherent_route",
    "smallest_recovery_row",
    "without_preparation_handles",
]
