r"""Analytic Low–Somma kernel-magnitude PREP: a real Chebyshev polynomial of the node diagonal, then clean amplitude amplification.

The constant-source LCHS kernel register needs the state

    \[|\psi_f\rangle=\frac{\sum_{j=-J}^{J}f(k_j)|\ell(j)\rangle}{\sqrt{Mp_f}},\qquad
      f(k)=\frac{e^{-k^2/(8\gamma^2)}}{\sqrt{1+k^2}},\quad k_j=hj,\quad M=2J+1\]

(G18). Its squares ``f(k_j)**2 = exp(-k_j**2/(4 gamma**2))/(1 + k_j**2)``
are the magnitudes of the Low–Somma ``j=2, y=1`` coefficients on a grid of
uniform weights h (Low and Somma, arXiv:2508.19238v2, Eqs. (6) and (11);
``algorithms.lchs.providers.low_somma_fhat_2``), so ``|psi_f>`` is the
kernel factor ``sum_j sqrt(|c_j|/beta_k)|j>`` of
``source_prep.FactorizedSourcePreparation``. The phase ``arg c_j`` stays with
``source_prep.build_coefficient_phase``. Low and Somma, arXiv:2508.19238v2,
Lemmas 19–22, prepare such a state from a uniform state, a polynomial
transformation and amplitude amplification; the finite parameters and errors
below replace their asymptotic cost.

Node encoding and polynomial. ``lambda`` is the ``alpha`` of the selected
``UniformNodeRecipe`` (its padding completion included), so the node block is
``x_j = k_j/lambda`` on physical labels. With ``F(x) = f(lambda x)``, F is
even and analytic. The Bernstein ellipse ``E_rho`` with imaginary semiaxis
``y = 1/(2 lambda)`` has ``rho = y + sqrt(1 + y**2)``; on it
``|Im(lambda z)| <= 1/2``, so ``|1 + (lambda z)**2| >= 3/4`` and
``|F| <= M_rho = (2/sqrt(3)) exp(1/(32 gamma**2))``. The Chebyshev
coefficients of F then obey ``|a_k| <= 2 M_rho rho**-k`` (Trefethen,
Approximation Theory and Approximation Practice, extended ed., SIAM 2019,
doi:10.1137/1.9781611975949, Theorem 8.1), and the degree-n truncation
error is at most

    \[\epsilon_{\rm tail}\le 2M_\rho\rho^{-n}/(\rho-1)\qquad\text{(G19)}.\]

The coefficients come from the 2N cosine samples ``theta_l = pi l/N``:
``c_0 = (1/2N) sum_l F(cos theta_l)`` and
``c_m = (1/N) sum_l F(cos theta_l) cos(m theta_l)``, ``1 <= m <= n < N``.
Orthogonality of ``cos`` on these points gives
``c_m = a_m + sum_{l>=1} (a_{2Nl+m} + a_{2Nl-m})``; summing the geometric
alias tails over ``m = 0..n`` gives

    \[\epsilon_{\rm alias}\le\frac{2M_\rho(1+2n)\rho^{-(2N-n)}}{1-\rho^{-2N}}
      \qquad\text{(G20)}.\]

The samples obey ``g_l = g_{N-l}`` because ``F(-x) = F(x)``; they are
evaluated for ``l <= N/2`` and mirrored, so the exact transform of the stored
samples has zero odd coefficients, and the odd slots are set to exactly zero.
This uses the analytic symmetry, not a threshold on computed values. With
``p = sum_m c_m T_m`` and ``Delta = G19 + G20``, ``|p - F| <= Delta`` in
exact arithmetic. The binary64 transform has no enclosure and is not in
Delta. Since ``|F| <= 1`` on [-1, 1], the fixed scale
``mu = (1 + Delta)/(1 - eta)``, ``0 < eta < 1``, gives the target
``q = p/mu`` with ``|q| <= 1 - eta``. n and N are chosen from the logarithms
of G19 and G20 before any sample exists, and do not change when the phase
solve fails.

Block and probabilities. ``solve_wx_qsp_phases`` returns Wx phases whose
real part ``Re P`` approximates q with degree d (n after exact trailing-zero
trimming), and ``qsp.evolution._build_real_chebyshev_encoding`` realizes the
block ``Re P(x)`` (Gilyen, Su, Low and Wiebe, arXiv:1806.01838v1,
Corollary 18), not ``Re P/2``, on the projector of the pair, signal and every
node ancilla. On the uniform input the target good amplitudes are
``q(x_j)/sqrt(M)``; padding labels carry no amplitude. One streamed pass on
the selected candidate gives

    \[p_q=M^{-1}\sum_jq(k_j/\lambda)^2,\qquad p_f=M^{-1}\sum_jf(k_j)^2
      \qquad\text{(MP1)},\]

by Clenshaw in batches of ``b = max(d + 1, 4096)`` nodes, so the
workspace is ``O(b + M/b)`` (``_mp1_bytes``) and the batch size, which
fixes the ``fsum`` grouping, depends only on d. MP1 is
the probability of the target polynomial; the realized block differs by the
terms below. With ``T_m' = m U_{m-1}`` and ``|U_{m-1}| <= m`` on [-1, 1], a
node error ``|x~_j - x_j| <= e_x`` changes q by at most ``L_q e_x``, with
``L_q = sum_{m>=1} m**2 |q_m|`` (MP2), a scalar statement on [-1, 1]. Summed
in squares against the uniform amplitudes it bounds the good vector without
a factor ``sqrt(M)``. The comparator and Pauli recipes have the exact ideal
block ``j/N``, so ``e_x = |h N - lambda|/lambda`` (``nodes`` module
docstring); the rotation table is refused because its stored-angle residual
excludes the rounding of ``h j``. With ``e_phi`` the solver's coefficient
residual of the saved phases and ``e = e_phi + L_q e_x + e_emit + e_u``,
projection and unitarity give for the actual good vector a and the target
``b_q = (q(x_j)/sqrt(M))_j``

    \[|\sqrt{p_a}-\sqrt{p_q}|\le e,\qquad
      p_a\in[\max(0,\sqrt{l_q}-e)^2,\ \min(1,(\sqrt{u_q}+e)^2)]
      \qquad\text{(MP3, MP4)},\]

and with ``E = Delta/mu + e`` and ``b_f = (f(k_j)/(mu sqrt(M)))_j``

    \[\Big\|\frac a{\|a\|}-\frac{b_f}{\|b_f\|}\Big\|\le\frac{2\mu E}{\sqrt{l_f}},
      \qquad \mu E<\sqrt{l_f}\qquad\text{(MP5)}.\]

The emitter term ``e_emit`` (binary64 angle conversion and gate parameters)
and the uniform-support term ``e_u`` (rounded rotation angles of Qiskit's
``UniformSuperpositionGate``) have no enclosure and are recorded as unknown.
MP1 also has no interval: ``l_q = u_q = p_q`` and ``l_f = p_f`` are binary64
estimates. Every bound is therefore a numerical estimate conditional on
those terms, never a certified probability or a certified unit success.

Uniform support. Qiskit's ``UniformSuperpositionGate`` prepares the prefix
``t = 0..M-1`` on ``r = (2J).bit_length()`` address bits; adding
``(-J) mod 2**r`` turns t into the two's-complement label of ``j = t - J``.
An increment of bit s flips the targets ``r-1, ..., s`` in that order, the
target ``s + t`` controlled by the t bits below it, which have not changed
yet. For signed-magnitude labels, on sign one the low ``p = r - 1`` bits are
complemented and incremented, which maps ``2**p + (2**p - m)`` to
``2**p + m`` and the padding word ``2**p`` to negative zero. Both maps are
permutations of the whole register, so the inverse is the reversed circuit
and no scratch is needed. No M-amplitude table is formed.

Amplification. With ``A_p`` the support and the ladder, and ``p_0 = p_q``,

    \[\theta_0=\arcsin\sqrt{p_0},\quad K=\max(0,\lceil\pi/(4\theta_0)-1/2\rceil),
      \quad\theta_*=\pi/(4K+2),\quad r_d=\sin^2\theta_*/p_0\le1
      \qquad\text{(G22)}.\]

K is raised by one if rounding gives ``sin(theta_*)**2 > p_0``, which keeps
``r_d <= 1``. ``RY(2 arccos sqrt(r_d))`` on a dilution qubit gives its zero
state amplitude ``sqrt(r_d)``, and ``A_d = D_r A_p``. The good projector is
all work qubits (dilution, pair, signal, node ancillas) zero, with the
address free. With ``R_g = I - 2 Pi_good`` and ``R_0 = 2|0><0| - I`` on every
input of ``A_d``, the address included,

    \[P_{LS}=(A_dR_0A_d^\dagger R_g)^KA_d\qquad\text{(MP6)},\]

appended as ``A_d`` and then ``R_g, A_d^dagger, R_0, A_d`` per round. In the
good/bad plane ``A_d R_0 A_d^dagger R_g`` rotates the angle by ``2 theta_*``,
so ideal output is the good vector with every work qubit zero and a positive
sign (exact amplitude amplification with a known probability, Brassard,
Hoyer, Mosca and Tapp, arXiv:quant-ph/0005055v1, Sec. 2; the reflection
signs are those of ``qsp.evolution._append_reflection``). The stored
angle realizes ``r_d' = cos(angle/2)**2``, and for the MP4 endpoints

    \[\Delta_\theta=\max_{p\in\{p_-,p_+\}}|\arcsin\sqrt{p\,r_d'}-\theta_*|,\qquad
      \delta_k\le\frac{2\mu E}{\sqrt{l_f}}+(2K+1)\Delta_\theta+e_{\rm exec}
      \qquad\text{(MP7)}.\]

Only the endpoints are needed because the angle is monotone in p; the
full-state distance to the good vector is ``2|sin((Phi - pi/2)/2)| <=
|Phi - pi/2|``, so MP7 bounds the clean state including all garbage, without
postselection. The polynomial block error enters once, through MP4/MP5. A
block error is not a full-unitary error, so it is not multiplied by
``2K + 1``; ``e_exec`` (full-gate synthesis of ``A_d`` and the reflections)
is unknown and recorded as such. ``||P_LS|0> - |0_work>psi_f|| <= delta_k``
also gives ``||P_LS^dagger|0_work>psi_f - |0>|| <= delta_k``.

Control and adjoint. Each phase slot of the real ladder is
``F C RZ C F``, with F the X on the signal controlled by the node ancillas in
zero and C the pair-to-signal CNOT. Controlling only the RZ makes an
inactive slot ``F C C F = I`` on every input. For even d the inactive ladder
is then ``H U U^dagger ... U U^dagger H = I`` on the whole register, the
node queries being one realized forward/adjoint pair. The global phase
``g_d = d pi/2`` of ``wx_phases_to_reflection`` becomes the active-control
sign ``(-1)**(d/2)``, exact by ``core._append_phase_flip``. The support map,
the dilution rotation and both reflections carry the controls, so the
controlled PREP is ``|s><s| (x) P_LS + (I - |s><s|) (x) I`` on every input.
The adjoint is ``qiskit_compat.inverse_realized_gate`` of the realized
forward gate, with no second phase target.

Counts (MP8). One complete PREP costs
``(2K+1)(C_U_M + C_map + C_B + C_dil) + K C_R_g + K C_R_0``. The ladder has
``d/2`` forward and ``d/2`` adjoint node queries, ``2(d+1)`` flag toggles
with ``a`` controls, ``2(d+1)`` pair CNOTs, ``d+1`` RZ and two H; one
external control turns each RZ into a controlled RZ (two CX) and leaves the
queries uncontrolled. The prefix costs ``3r - 5`` CX, ``41r - 59`` with one
control (``_preparation_laws._uniform_superposition_gate_slots`` and
``direct_preparation_controlled_cx_bound``). The subtractor costs
``sum_{s in bits((-J) mod 2**r)} sum_{t=0}^{r-s-1} c_X(t+k)`` and the
signed-magnitude conversion ``p c_X(k+1) + sum_{t<p} c_X(t+k+1)``, with
``c_X = core._mcx_cx`` and k external controls. ``R_g`` flips one state of
the ``w = a + 3`` work qubits and ``R_0`` one state of all ``r + w`` inputs,
plus its exact sign. These are CX upper bounds before cancellation; the node
query counts are the recipe's ``query_census``.

Stages and limits. ``plan_analytic_kernel_preparation`` evaluates only
scalar bounds and finite laws. ``prepare_analytic_kernel`` acquires the
coefficients, streams MP1, chooses the G22 schedule and refuses when
``K > max_rounds`` before the phase solve, then makes one
``solve_wx_qsp_phases`` call. Each stage checks its dominant arrays against
the memory limit before allocating. ``max_rounds`` limits work; it
asserts nothing about success. ``build_analytic_kernel_query`` builds only
from the saved selected data, so rebuilding a saved preparation never
solves phases, recomputes probabilities or reselects.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Mapping

import numpy as np
from scipy.fft import dct

from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from nwqlib.subroutines.block_encoding.core import (
    _UnknownMCXCount,
    _append_mcx,
    _append_phase_flip,
    _control_state,
    _mcx_cx,
)
from nwqlib.subroutines.block_encoding.nodes import (
    UniformNodeRecipe,
    _normalization_error,
    _uniform_preparation_cx,
)
from nwqlib.subroutines.qsp.phases import (
    QSP_SOLVER_RESIDUAL_TOLERANCE,
    WxQSPPhases,
    _QSPPhaseConvergenceError,
    solve_wx_qsp_phases,
)

_SUPPORTED_RECIPES = ("comparator", "twos_complement_pauli", "signed_magnitude_pauli")
# The smallest positive binary64 value: a positive bound evaluated from its
# logarithm is never reported as exactly zero.
_LOG_TINY = math.log(float(np.nextafter(0.0, 1.0)))


# Nodes per Clenshaw batch of the MP1 probability pass, at least d + 1. A
# fixed engineering choice, so memory never changes the summation grouping.
# Registered in docs/ENGINEERING_CONSTANTS.md. Revisit with another pass.
_NODE_BATCH = 4096


@dataclass(frozen=True, kw_only=True)
class AnalyticKernelPlan:
    """The analytic kernel PREP of one grid, described by scalar bounds and finite laws before any array or circuit exists.

    Attributes:
        recipe: The node encoding of the PREP. Its ``alpha`` is the
            normalization lambda, its ``half_width`` J and ``spacing`` h
            define the grid, and its mapping fixes the address labels.
        gamma: Low–Somma gamma of ``f``, positive.
        degree: Even acquisition degree n of G19; the solved degree d is at
            most n.
        samples: N, a power of two above n; 2N cosine samples.
        tail_bound: G19 at n, in the units of F.
        alias_bound: G20 at (n, N).
        approximation_bound: Delta = G19 + G20, rounded up; an exact-arithmetic
            bound that excludes the binary64 transform.
        margin: eta of the norm cap ``|q| <= 1 - eta``.
        scale: mu = (1 + Delta)/(1 - eta), rounded up; q = p/mu.
        max_rounds: Admission limit on the amplification rounds K.
        conditional_round_bound: The G22 count of the center-term lower bound
            ``p_q >= (1 - Delta)**2/(M mu**2)``, which bounds the selected K
            when Delta covers the stored coefficients; None when Delta >= 1.
        residual_tolerance, max_fft: Controls passed to
            ``solve_wx_qsp_phases``.
        max_bytes: Memory limit of each preparation stage; None uses the
            physical memory.
        census: MP8 counts per PREP at d = n and K = ``max_rounds``, without
            and with one external control (``_census``).
    """

    recipe: UniformNodeRecipe
    gamma: float
    degree: int
    samples: int
    tail_bound: float
    alias_bound: float
    approximation_bound: float
    margin: float
    scale: float
    max_rounds: int
    conditional_round_bound: int | None
    residual_tolerance: float
    max_fft: int | None
    max_bytes: int | None
    census: Mapping[str, Any]

    @property
    def nodes(self) -> int:
        """M = 2J + 1."""
        return 2 * self.recipe.half_width + 1

    @property
    def work_qubits(self) -> int:
        """w = a + 3: dilution, pair, signal and the node ancillas."""
        return self.recipe.encoding_ancillas + 3

    @property
    def kernel_qubits(self) -> int:
        """r + w, the width of the PREP without external controls."""
        return self.recipe.address_qubits + self.work_qubits

    @property
    def wires(self) -> dict[str, tuple[int, ...]]:
        """Qubit indices of the PREP after its external controls."""
        a, r = self.recipe.encoding_ancillas, self.recipe.address_qubits
        return {"dilution": (0,), "pair": (1,), "signal": (2,), "node_ancillas": tuple(range(3, 3 + a)),
                "address": tuple(range(3 + a, 3 + a + r))}


@dataclass(frozen=True, kw_only=True)
class AnalyticKernelPreparation:
    """The selected analytic kernel PREP: coefficients, phases, probabilities, schedule and errors.

    Attributes:
        plan: The plan it was prepared from.
        phases: The one phase solve, with the trimmed target coefficients of
            q in ``target_chebyshev_coefficients`` and the solver's evidence.
        sensitivity: L_q of MP2.
        target_probability: p_q of MP1.
        physical_probability: p_f of MP1.
        probability_qualification: What the MP1 values are.
        rounds: K of G22.
        dilution_angle: The stored RY angle ``2 arccos sqrt(r_d)``.
        dilution_ratio: ``cos(angle/2)**2``, the ratio the stored angle
            realizes, used in MP7.
        error_terms: Values of the MP2–MP7 terms, None where unknown:
            ``approximation`` (Delta), ``phase`` (e_phi), ``node`` (e_x),
            ``emitter``, ``uniform``, ``block`` (the known part of e),
            ``probability_lower``/``probability_upper`` (MP4),
            ``direction`` (MP5), ``angle_spread``, ``amplification``
            ((2K+1) Delta_theta) and ``execution``.
        error_evidence: What each term covers.
        kernel_error: delta_k of MP7 without the unknown terms, a numerical
            estimate.
        qualification: The evidence class of ``kernel_error``.
        census: MP8 counts at the actual d and K (``_census``).
    """

    plan: AnalyticKernelPlan
    phases: WxQSPPhases
    sensitivity: float
    target_probability: float
    physical_probability: float
    probability_qualification: str
    rounds: int
    dilution_angle: float
    dilution_ratio: float
    error_terms: Mapping[str, float | None]
    error_evidence: Mapping[str, str]
    kernel_error: float
    qualification: str
    census: Mapping[str, Any]

    @property
    def degree(self) -> int:
        """d, the degree of the solved target."""
        return self.phases.degree

    @property
    def theta_star(self) -> float:
        """pi/(4K + 2)."""
        return math.pi / (4 * self.rounds + 2)


def _magnitude(k: np.ndarray, gamma: float) -> np.ndarray:
    """G18: ``f(k) = exp(-k**2/(8 gamma**2))/sqrt(1 + k**2)``."""
    square = k * k
    return np.exp(-square / (8.0 * gamma * gamma)) / np.sqrt(1.0 + square)


def _log_bounds(radius: float, gamma: float):
    """Return ``log rho``, ``log(rho - 1)`` and ``log M_rho`` of the ellipse with imaginary semiaxis 1/(2 radius).

    ``rho = y + sqrt(1 + y**2) = exp(asinh y)``, so ``rho - 1 = expm1(asinh y)``
    without cancellation for small y.
    """
    log_rho = math.asinh(0.5 / radius)
    return log_rho, math.log(math.expm1(log_rho)), math.log(2.0 / math.sqrt(3.0)) + 1.0 / (32.0 * gamma * gamma)


def _log_tail(n: int, log_rho: float, log_gap: float, log_m: float) -> float:
    """log of G19, ``2 M_rho rho**-n/(rho - 1)``."""
    return math.log(2.0) + log_m - n * log_rho - log_gap


def _log_alias(n: int, samples: int, log_rho: float, log_m: float) -> float:
    """log of G20, ``2 M_rho (1 + 2n) rho**-(2N - n)/(1 - rho**-2N)``."""
    return (math.log(2.0) + log_m + math.log(1 + 2 * n) - (2 * samples - n) * log_rho
            - math.log(-math.expm1(-2 * samples * log_rho)))


def _bound(log_value: float) -> float:
    return math.exp(max(log_value, _LOG_TINY))


def _acquisition_bytes(degree: int, samples: int) -> int:
    """Coarse bytes of the coefficient acquisition: the evaluated and mirrored samples, the DCT output and n + 1 coefficients."""
    return 8 * (8 * (samples // 2 + 1) + 2 * (samples + 1) + degree + 1)


def _mp1_bytes(terms: int, stored: int, nodes: int) -> int:
    """Coarse bytes of the MP1 pass over ``nodes`` nodes with ``terms`` target coefficients.

    A batch holds ``min(nodes, b)`` nodes, ``b = max(terms, _NODE_BATCH)``,
    the last or only batch included. Ten float64 arrays of that length cover
    the nodes, their scaled values, the two Clenshaw recurrences and one
    step's temporaries, q, its square and the temporaries of ``_magnitude``.
    The ``stored`` coefficients of the target array stay alive, and the two
    lists of partial sums take about 72 bytes per batch.
    """
    batch = max(terms, _NODE_BATCH)
    return 8 * (10 * min(nodes, batch) + stored) + 72 * -(-nodes // batch)


def _rounds(probability: float) -> int:
    """G22: ``K = max(0, ceil(pi/(4 theta) - 1/2))`` with ``theta = asin sqrt(p)``, raised once if rounding puts theta_* above theta."""
    theta = math.asin(math.sqrt(probability))
    rounds = max(0, math.ceil(math.pi / (4.0 * theta) - 0.5))
    if math.sin(math.pi / (4 * rounds + 2)) ** 2 > probability:
        rounds += 1
    return rounds


def _census(recipe: UniformNodeRecipe, *, degree: int, rounds: int) -> dict[str, Any]:
    """Return the MP8 counts of one PREP, without and with one external control.

    Each row holds the operation counts and the CX upper bounds per
    amplitude circuit ``A_d`` and per reflection, and the total
    ``(2K+1) A_d + K (R_g + R_0)``. A CX count of an X with more controls
    than ``_mcx_counts`` tabulates is unknown, and that row's CX values are
    None.
    """
    r, a, J = recipe.address_qubits, recipe.encoding_ancillas, recipe.half_width
    w, p, d, K = a + 3, r - 1, degree, rounds
    shift = (-J) % (1 << r)
    query = recipe.query_census
    rows = {}
    for name, k in (("forward", 0), ("controlled", 1)):
        operations = {
            "amplitude_calls": 2 * K + 1, "good_reflections": K, "zero_reflections": K,
            "forward_node_queries": (2 * K + 1) * (d // 2), "adjoint_node_queries": (2 * K + 1) * (d // 2),
            "phase_slots": (2 * K + 1) * (d + 1), "flag_toggles": 2 * (2 * K + 1) * (d + 1),
            "pair_cnots": 2 * (2 * K + 1) * (d + 1), "pair_hadamards": 2 * (2 * K + 1),
            "uniform_prefixes": 2 * K + 1, "dilution_rotations": 2 * K + 1,
        }
        try:
            subtract = sum(_mcx_cx(t + k) for s in range(r) if shift >> s & 1 for t in range(r - s))
            negate = (p * _mcx_cx(k + 1) + sum(_mcx_cx(t + k + 1) for t in range(p))
                      if recipe.mapping == "signed_magnitude" else 0)
            # d/2 forward and d/2 adjoint queries, 2(d+1) flag toggles (a controls) and pair CNOTs, and
            # with one control d+1 controlled RZ of two CX each. A controlled dilution RY adds two CX.
            ladder = ((d // 2) * (query["forward_cx"] + query["adjoint_cx"]) + 2 * (d + 1) * (_mcx_cx(a) + 1)
                      + 2 * k * (d + 1))
            prefix = _uniform_preparation_cx(2 * J + 1) if k == 0 else 41 * r - 59
            amplitude = prefix + subtract + negate + ladder + 2 * k
            good, zero = _mcx_cx(w + k - 1), _mcx_cx(r + w + k - 1) + (_mcx_cx(k - 1) if k else 0)
            cx = {"amplitude_cx": amplitude, "uniform_prefix_cx": prefix, "map_cx": subtract + negate,
                  "ladder_cx": ladder, "good_reflection_cx": good, "zero_reflection_cx": zero,
                  "cx": (2 * K + 1) * amplitude + K * (good + zero)}
        except _UnknownMCXCount:
            cx = dict.fromkeys(("amplitude_cx", "uniform_prefix_cx", "map_cx", "ladder_cx", "good_reflection_cx",
                                "zero_reflection_cx", "cx"))
        rows[name] = {"qubits": r + w + k, **operations, **cx}
    rows["cx_qualification"] = ("upper bounds before cancellation: Qiskit's no-ancilla multi-controlled X counts, "
                                "the recipe's node-query census and the uniform-prefix slot bounds; not a transpiled "
                                "or hardware count")
    return rows


def plan_analytic_kernel_preparation(
    recipe: UniformNodeRecipe, *, gamma: float, approximation_tolerance: float, margin: float, max_rounds: int,
    residual_tolerance: float = QSP_SOLVER_RESIDUAL_TOLERANCE, max_fft: int | None = None,
    max_bytes: int | None = None,
) -> AnalyticKernelPlan:
    """Choose the degree, samples and scale of the analytic kernel PREP from G19/G20, without arrays or circuits.

    n is the smallest even degree with G19 at most half of
    ``approximation_tolerance``, and N the smallest power of two above n
    with G20 at most the rest, both from their logarithms; both bounds
    decrease to zero, so the searches end. See the module docstring for the
    relations.

    Args:
        recipe (UniformNodeRecipe): ``"comparator"``,
            ``"twos_complement_pauli"`` or ``"signed_magnitude_pauli"`` from
            ``uniform_node_recipes``. The rotation table is refused: its
            node error excludes the rounding of ``h j``.
        gamma (float): Low–Somma gamma, finite and positive.
        approximation_tolerance (float): Budget for Delta = G19 + G20 in the
            frame of F, finite and positive.
        margin (float): eta in (0, 1).
        max_rounds (int): Admission limit on K, at least 1.
        residual_tolerance (float): Default 1e-12; phase residual tolerance.
        max_fft (int | None): Default None; FFT cap of the phase solver.
        max_bytes (int | None): Default None, the physical memory; memory
            limit of each preparation stage.

    Returns:
        plan (AnalyticKernelPlan): The scalar facts and finite laws.

    Raises:
        ValueError: For an unsupported recipe or a value outside its domain.
    """
    if not isinstance(recipe, UniformNodeRecipe) or recipe.kind not in _SUPPORTED_RECIPES:
        raise ValueError(f"the analytic kernel PREP needs a {', '.join(_SUPPORTED_RECIPES)} node recipe; the "
                         "rotation table's node error excludes the rounding of h j")
    gamma = finite_real(gamma, "gamma")
    tolerance = finite_real(approximation_tolerance, "approximation_tolerance")
    eta = finite_real(margin, "margin")
    if gamma <= 0 or tolerance <= 0 or not 0 < eta < 1:
        raise ValueError("gamma and approximation_tolerance must be positive and margin must lie in (0, 1)")
    rounds_cap = integer(max_rounds, "max_rounds", 1)
    if max_fft is not None:
        max_fft = integer(max_fft, "max_fft", 1)
    if max_bytes is not None:
        max_bytes = integer(max_bytes, "max_bytes", 1)

    log_rho, log_gap, log_m = _log_bounds(recipe.alpha, gamma)
    if not math.isfinite(log_m):
        raise ValueError(f"M_rho = (2/sqrt(3)) exp(1/(32 gamma**2)) is not finite for gamma={gamma!r}")
    target = math.log(tolerance / 2.0)
    n = max(0, math.ceil((math.log(2.0) + log_m - log_gap - target) / log_rho))
    n += n % 2
    while _log_tail(n, log_rho, log_gap, log_m) > target:
        n += 2
    tail = _bound(_log_tail(n, log_rho, log_gap, log_m))
    remaining = math.log(tolerance - tail)
    samples = 1 << n.bit_length()
    while _log_alias(n, samples, log_rho, log_m) > remaining:
        samples *= 2
    alias = _bound(_log_alias(n, samples, log_rho, log_m))
    delta = float(np.nextafter(tail + alias, np.inf))
    scale = float(np.nextafter((1.0 + delta) / (1.0 - eta), np.inf))

    nodes = 2 * recipe.half_width + 1
    conditional = None
    if delta < 1.0:
        center = (1.0 - delta) ** 2 / (nodes * scale * scale)
        conditional = _rounds(center)
    return AnalyticKernelPlan(
        recipe=recipe, gamma=gamma, degree=n, samples=samples, tail_bound=tail, alias_bound=alias,
        approximation_bound=delta, margin=eta, scale=scale, max_rounds=rounds_cap,
        conditional_round_bound=conditional, residual_tolerance=finite_real(residual_tolerance, "residual_tolerance"),
        max_fft=max_fft, max_bytes=max_bytes,
        census=_census(recipe, degree=n, rounds=rounds_cap))


def _acquire_coefficients(plan: AnalyticKernelPlan) -> np.ndarray:
    """Return the even ascending coefficients ``c_0..c_n`` of the 2N-point cosine transform of ``F(x) = f(lambda x)``.

    The DCT-I of ``g_0..g_N`` is ``g_0 + (-1)**m g_N + 2 sum_{l=1}^{N-1} g_l
    cos(pi m l/N)``, the sum over all 2N points because ``g_{2N-l} = g_l``.
    The samples are evaluated for ``l <= N/2`` and mirrored by
    ``F(-x) = F(x)``, and the odd slots are exactly zero.
    """
    n, samples = plan.degree, plan.samples
    half = samples // 2
    values = np.empty(samples + 1)
    values[:half + 1] = _magnitude(plan.recipe.alpha * np.cos(np.pi * np.arange(half + 1) / samples), plan.gamma)
    values[samples - np.arange(half + 1)] = values[:half + 1]
    transform = dct(values, type=1)
    coefficients = np.zeros(n + 1)
    coefficients[0::2] = transform[:n + 1:2] / samples
    coefficients[0] /= 2.0
    return coefficients


def _streamed_probabilities(coefficients: np.ndarray, plan: AnalyticKernelPlan) -> tuple[float, float]:
    """MP1: ``p_q`` and ``p_f`` on the physical nodes ``k_j = fl(h j)``, ``x_j = fl(k_j/lambda)``, by Clenshaw in batches.

    A batch holds ``max(d + 1, _NODE_BATCH)`` nodes, so its arrays stay small
    while the Python loop over coefficients runs once per batch. Each batch
    sum and the total use ``math.fsum``. The values are binary64 estimates
    without an enclosure.
    """
    J, h, lam = plan.recipe.half_width, plan.recipe.spacing, plan.recipe.alpha
    batch = max(len(coefficients), _NODE_BATCH)
    q_sums, f_sums = [], []
    for start in range(-J, J + 1, batch):
        k = h * np.arange(start, min(start + batch, J + 1), dtype=float)
        x = k / lam
        later = current = np.zeros_like(x)
        for coefficient in coefficients[:0:-1]:
            current, later = 2.0 * x * current - later + coefficient, current
        q = x * current - later + coefficients[0]
        q_sums.append(math.fsum(q * q))
        f_sums.append(math.fsum(_magnitude(k, plan.gamma) ** 2))
    nodes = 2 * J + 1
    return math.fsum(q_sums) / nodes, math.fsum(f_sums) / nodes


def _node_error(recipe: UniformNodeRecipe) -> float:
    """e_x of MP2: ``|h N - lambda|/lambda`` for the exact ideal block ``j/N`` of the comparator and Pauli recipes, rounded up.

    N is J for the comparator and the integer alpha of the Pauli plan, as in
    ``nodes.build_uniform_node_encoding``.
    """
    extent = recipe.half_width if recipe.plan is None else int(recipe.plan.alpha)
    exact = Fraction(_normalization_error(recipe.spacing, extent, recipe.alpha)) / Fraction(recipe.alpha)
    bound = float(exact)
    return bound if Fraction(bound) >= exact else float(np.nextafter(bound, np.inf))


def prepare_analytic_kernel(plan: AnalyticKernelPlan) -> AnalyticKernelPreparation:
    """Acquire, evaluate, schedule and solve the one selected analytic kernel PREP.

    The stages run once each, in this order: coefficient acquisition, the
    MP1 pass, the G22 schedule (refused when K exceeds ``plan.max_rounds``),
    one ``solve_wx_qsp_phases`` call, and the MP2–MP7 terms. See the module docstring for the relations and their
    qualification.

    Args:
        plan (AnalyticKernelPlan): From ``plan_analytic_kernel_preparation``.

    Returns:
        preparation (AnalyticKernelPreparation): The selected data.

    Raises:
        ValueError: For a zero target polynomial, K above ``max_rounds``, a
            failed MP5 premise ``mu E < sqrt(p_f)`` or a zero MP4 lower
            endpoint, a memory refusal, or a phase-solver refusal, which
            propagates as raised. A phase-solver convergence failure is
            raised again in its own class, a ``_QSPPhaseConvergenceError``
            that is also a ``ResourceLimitError`` when a limit stopped the
            doubling, with its evidence and cause.
    """
    _check_bytes(_acquisition_bytes(plan.degree, plan.samples), plan.max_bytes, "analytic kernel coefficients")
    coefficients = _acquire_coefficients(plan)
    target = coefficients / plan.scale
    del coefficients
    support = np.flatnonzero(target)
    # Only exact zeros change the degree, as in the solver.
    degree = int(support[-1]) if support.size else 0
    target = target[:degree + 1]
    _check_bytes(_mp1_bytes(target.size, plan.degree + 1, 2 * plan.recipe.half_width + 1), plan.max_bytes,
                 "analytic kernel MP1 pass")
    p_q, p_f = _streamed_probabilities(target, plan)
    if p_q <= 0.0:
        raise ValueError("the target polynomial vanishes on every node (p_q = 0), so it gives no amplification "
                         "schedule")
    rounds = _rounds(p_q)
    if rounds > plan.max_rounds:
        raise ValueError(f"G22 needs K={rounds} amplification rounds for p_q={p_q!r}, above max_rounds="
                         f"{plan.max_rounds}")
    try:
        phases = solve_wx_qsp_phases(target, residual_tolerance=plan.residual_tolerance,
                                     max_bytes=plan.max_bytes, max_fft=plan.max_fft)
    except _QSPPhaseConvergenceError as error:
        raise type(error)(f"analytic kernel PREP phase solve failed: {error}", evidence=error.evidence) from error

    orders = np.arange(1, degree + 1, dtype=float)
    sensitivity = math.fsum(orders * orders * np.abs(target[1:]))
    node = _node_error(plan.recipe)
    phase = phases.residual_sup_bound
    block = phase + sensitivity * node
    mu = plan.scale
    total = plan.approximation_bound / mu + block
    lower = max(0.0, math.sqrt(p_q) - block) ** 2
    upper = min(1.0, (math.sqrt(p_q) + block) ** 2)
    if lower <= 0.0 or mu * total >= math.sqrt(p_f):
        raise ValueError(f"the direction bound needs mu E < sqrt(p_f) and a positive MP4 lower endpoint; mu E="
                         f"{mu * total!r}, sqrt(p_f)={math.sqrt(p_f)!r}, lower endpoint={lower!r}")
    direction = 2.0 * mu * total / math.sqrt(p_f)
    theta_star = math.pi / (4 * rounds + 2)
    angle = 2.0 * math.acos(math.sqrt(math.sin(theta_star) ** 2 / p_q))
    ratio = math.cos(angle / 2.0) ** 2
    spread = max(abs(math.asin(math.sqrt(min(1.0, value * ratio))) - theta_star) for value in (lower, upper))
    amplification = (2 * rounds + 1) * spread
    terms = {
        "approximation": plan.approximation_bound, "phase": phase, "node": node, "emitter": None, "uniform": None,
        "block": block, "probability_lower": lower, "probability_upper": upper, "direction": direction,
        "angle_spread": spread, "amplification": amplification, "execution": None,
    }
    evidence = {
        "approximation": "G19 + G20 in exact arithmetic; the binary64 cosine transform has no enclosure",
        "phase": f"solver coefficient residual of the saved phases, {phases.evidence.qualification}",
        "node": "|h N - lambda|/lambda for the exact ideal block j/N, rational arithmetic, rounded up",
        "emitter": "unknown: binary64 reflection-angle conversion and gate parameters of the real ladder",
        "uniform": "unknown: rounded rotation angles of UniformSuperpositionGate",
        "block": "phase + sensitivity * node; excludes the unknown emitter and uniform terms",
        "probability_lower": "MP4 lower endpoint from the binary64 p_q estimate and block",
        "probability_upper": "MP4 upper endpoint from the binary64 p_q estimate and block",
        "direction": "MP5 2 mu E/sqrt(p_f) with the binary64 p_f estimate",
        "angle_spread": "MP7 Delta_theta at the MP4 endpoints and the stored dilution angle",
        "amplification": "(2K + 1) angle_spread, the clean-state garbage of the fixed dilution",
        "execution": "unknown: full-gate synthesis error of A_d and the reflections",
    }
    return AnalyticKernelPreparation(
        plan=plan, phases=phases, sensitivity=sensitivity, target_probability=p_q, physical_probability=p_f,
        probability_qualification=("binary64 Clenshaw and f evaluation on k_j = fl(h j), x_j = fl(k_j/lambda), "
                                   "fsum accumulation; numerical estimates, no enclosure"),
        rounds=rounds, dilution_angle=angle, dilution_ratio=ratio, error_terms=terms, error_evidence=evidence,
        kernel_error=direction + amplification,
        qualification=("numerical: binary64 coefficient, probability and phase arithmetic without enclosures; the "
                       "emitter, uniform-support and full-gate errors are unknown and not included"),
        census=_census(plan.recipe, degree=degree, rounds=rounds))


def _append_increment(circuit, bits, low: int, controls, state: int) -> None:
    """Add ``2**low`` modulo ``2**len(bits)`` to the little-endian register ``bits`` on the controls' ``state``.

    The targets run from the top bit down to ``low``; target ``low + t`` is
    controlled by the t bits below it, all one, which have not changed yet.
    """
    k = len(controls)
    for target in range(len(bits) - 1, low - 1, -1):
        carry = target - low
        _append_mcx(circuit, [*controls, *bits[low:target]], state | (((1 << carry) - 1) << k), bits[target])


def _controlled_ladder(ladder, degree: int, num_controls: int, state: int):
    """Return the realized real ladder with only its RZ gates and its degree sign on the external controls (module docstring).

    The ladder's float global phase ``d pi/2`` is replaced by the exact
    active-control sign ``(-1)**(d/2)`` of even d.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZGate

    circuit = QuantumCircuit(num_controls + ladder.num_qubits, name="c_" + ladder.name)
    controls, wires = circuit.qubits[:num_controls], circuit.qubits[num_controls:]
    for item in ladder.data:
        qubits = [wires[ladder.find_bit(qubit).index] for qubit in item.qubits]
        if type(item.operation) is RZGate:
            circuit.append(RZGate(item.operation.params[0]).control(num_controls, ctrl_state=state, annotated=False),
                           [*controls, *qubits])
        else:
            circuit.append(item.operation, qubits)
    if degree % 4 == 2:
        _append_phase_flip(circuit, controls, state, None)
    return circuit


def _amplitude_circuit(prepared: AnalyticKernelPreparation, num_controls: int, state: int, max_bytes, max_work):
    """Return ``A_d = D_r B_phi U_M`` on [controls, dilution, pair, signal, node ancillas, address], controlled on ``state``."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate, UniformSuperpositionGate
    from nwqlib.subroutines.block_encoding.nodes import build_uniform_node_encoding
    from nwqlib.subroutines.qiskit_compat import controlled
    from nwqlib.subroutines.qsp.evolution import _block_gate_pair, _build_real_chebyshev_encoding

    plan, k = prepared.plan, num_controls
    recipe, d = plan.recipe, prepared.degree
    r, a = recipe.address_qubits, recipe.encoding_ancillas
    encoding = build_uniform_node_encoding(recipe, max_bytes=max_bytes, max_work=max_work)
    pair = (None, None)
    if d:
        pair = _block_gate_pair(encoding, native_ucg=False)
    ladder = _build_real_chebyshev_encoding(encoding, prepared.phases.phases, block_gates=pair)

    circuit = QuantumCircuit(k + plan.kernel_qubits, name="c_analytic_kernel_amplitude" if k else
                             "analytic_kernel_amplitude")
    controls = list(circuit.qubits[:k])
    dilution, work = circuit.qubits[k], list(circuit.qubits[k + 1:k + 3 + a])
    address = list(circuit.qubits[k + 3 + a:])
    uniform = UniformSuperpositionGate(plan.nodes, num_qubits=r)
    circuit.append(controlled(uniform, k, ctrl_state=state) if k else uniform, [*controls, *address])
    shift = (-recipe.half_width) % (1 << r)
    for bit in range(r):
        if shift >> bit & 1:
            _append_increment(circuit, address, bit, controls, state)
    if recipe.mapping == "signed_magnitude":
        signed = [*controls, address[-1]]
        for bit in address[:-1]:
            _append_mcx(circuit, signed, state | (1 << k), bit)
        _append_increment(circuit, address[:-1], 0, signed, state | (1 << k))
    if k:
        circuit.compose(_controlled_ladder(ladder, d, k, state), qubits=[*controls, *work, *address], inplace=True)
        circuit.append(RYGate(prepared.dilution_angle).control(k, ctrl_state=state, annotated=False),
                       [*controls, dilution])
    else:
        circuit.compose(ladder, qubits=[*work, *address], inplace=True)
        circuit.ry(prepared.dilution_angle, dilution)
    return circuit


def build_analytic_kernel_query(prepared: AnalyticKernelPreparation, *, num_controls: int = 0,
                                ctrl_state: int | str | None = None, adjoint: bool = False,
                                max_bytes: int | None = None, max_work: int | None = None):
    """Build the analytic kernel PREP ``P_LS`` (MP6), its adjoint, or either with external controls, as a gate.

    The gate acts on [controls, dilution, pair, signal, node ancillas,
    address] (``AnalyticKernelPlan.wires``). From all zero it prepares
    ``|0_work> (x) |psi_f>`` up to ``prepared.kernel_error``, the address in
    the recipe's label mapping. With controls the inactive branch is the
    identity on every qubit (module docstring), and the adjoint is the
    inverse of the realized forward definition. Only the saved data are
    used: no phase solve, probability pass or reselection.

    Args:
        prepared (AnalyticKernelPreparation): From
            ``prepare_analytic_kernel`` or a saved copy.
        num_controls (int): Default 0. Number of external controls.
        ctrl_state (int | str | None): Default None, all controls one.
        adjoint (bool): Default False. Return the adjoint.
        max_bytes (int | None): Default None, the physical memory. Limit on
            the dominant arrays of the node encoding, its multiplexor tables.
        max_work (int | None): Default None, no limit. Construction work
            limit of the node encoding.

    Returns:
        gate (qiskit.circuit.Gate): On ``num_controls + plan.kernel_qubits``
            qubits.

    Raises:
        ValueError: If ``ctrl_state`` does not fit or a limit is exceeded.
    """
    from qiskit import QuantumCircuit
    from nwqlib.subroutines.qiskit_compat import inverse_realized_gate
    from nwqlib.subroutines.qsp.evolution import _append_reflection

    k = integer(num_controls, "num_controls", 0)
    state = _control_state(ctrl_state, k)
    amplitude = _amplitude_circuit(prepared, k, state, max_bytes, max_work)
    forward = amplitude.to_gate(label="analytic_kernel_amplitude")
    backward = inverse_realized_gate(forward, native_ucg=False)
    circuit = QuantumCircuit(amplitude.num_qubits, name="c_analytic_kernel_prep" if k else "analytic_kernel_prep")
    wires = list(circuit.qubits)
    controls, inputs = wires[:k], wires[k:]
    work = inputs[:prepared.plan.work_qubits]
    circuit.append(forward, wires)
    for _ in range(prepared.rounds):
        _append_reflection(circuit, work, controls=controls, ctrl_state=state, positive=False)
        circuit.append(backward, wires)
        _append_reflection(circuit, inputs, controls=controls, ctrl_state=state, positive=True)
        circuit.append(forward, wires)
    gate = circuit.to_gate(label=circuit.name)
    return inverse_realized_gate(gate, native_ucg=False) if adjoint else gate


__all__ = [
    "AnalyticKernelPlan",
    "AnalyticKernelPreparation",
    "build_analytic_kernel_query",
    "plan_analytic_kernel_preparation",
    "prepare_analytic_kernel",
]
