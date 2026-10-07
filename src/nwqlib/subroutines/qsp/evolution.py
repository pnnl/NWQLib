"""QSVT circuits and Jacobi-Anger Hamiltonian-evolution synthesis.

[GSLW] is Gilyen, Su, Low, and Wiebe, arXiv:1806.01838v1, and [MRTC] is
Martyn, Rossi, Tan, and Chuang, arXiv:2105.02859v5. Equation, lemma and
theorem numbers refer to these arXiv versions.

Implements Jacobi-Anger evolution using the shared block-encoding records
and the
[block-encoding and QSP conventions](../../conventions.md#block-encoding-and-qsp-conventions):

- ``build_qsvt_circuit``: projector-controlled phase QSVT ([MRTC]
  arXiv:2105.02859v5, Sec. II.C-II.D, Eq. (27), Fig. 3, Theorems 3-4, and
  [GSLW] arXiv:1806.01838v1, Theorem 17 and
  Lemma 19): alternate ``U_BE`` / ``U_BE^dagger`` with
  ``e^{i phi (2 Pi - 1)}`` phases, where ``Pi`` projects onto the all-zero
  block-encoding ancillas. The phase operator is realized by the
  X-conjugation trick: flip a signal qubit onto the ``Pi`` subspace (``X``
  controlled on the ancillas being all zero), rotate ``RZGate(+2 phi)``
  (``= e^{-i phi Z}``, the Qiskit sign
  ``RZGate(theta) = exp(-i theta Z / 2)``), and flip back. Wx phases, in
  the convention dictionary of [MRTC] arXiv:2105.02859v5, App. A, are
  converted with ``wx_phases_to_reflection`` and the ``i^d`` factor becomes
  circuit global phase, so the encoded block is exactly the Wx polynomial
  ``P(A / alpha)``.
- ``build_real_chebyshev_encoding``: block-encodes ``Re P(A / alpha)`` by
  averaging the ``+Phi`` and ``-Phi`` passes over one pair ancilla
  ([GSLW] arXiv:1806.01838v1, Corollary 18, Eq. (33)). Both branches
  share the same ``U_BE``
  ladder: a CNOT from the pair qubit onto the signal qubit conjugates every
  ``RZ``, negating the reflection phases on the pair-one branch
  (``X RZ(theta) X = RZ(-theta)``). A ``Z`` on the pair qubit for odd
  degree fixes the ``(-1)^d`` of ``i^d <0|U_R(-Phi')|0> = (-1)^d conj(P)``.
  The shared ladder is NWQLib's circuit for Corollary 18's controlled
  ``U_Phi`` / ``U_-Phi`` pair.
- ``build_qsp_evolution_encoding``: Jacobi-Anger split ([GSLW]
  arXiv:1806.01838v1, Lemma 57, with the Bessel tails of Eqs. (53)-(56))
  ``cos(tau x) = J_0(tau) + 2 sum (-1)^k J_2k(tau) T_2k(x)`` (even) and
  ``sin(tau x) = 2 sum (-1)^k J_2k+1(tau) T_2k+1(x)`` (odd), truncation
  degree from the Bessel tail bound ``2 sum_{k>d} |J_k(tau)| <= eps`` with
  the recorded slack, one parity ancilla combining the passes with
  coefficients ``(1/2, -i/2)`` (the construction in the proof of [GSLW]
  arXiv:1806.01838v1, Theorem 58), and 3-step oblivious amplitude amplification
  ``A R A^dagger R A`` ([GSLW] arXiv:1806.01838v1, Theorem 28 with
  ``n = 3``), which realizes
  ``-(3B - 4 B B^dagger B)``, where ``B`` is the all-zero-ancilla block
  before amplification. The circuit adds a compensating ``pi`` global
  phase. At block amplitude ``a = 1/(2s)`` the OAA output ``(3a - 4a^3)`` is
  quadratically insensitive to the recorded target rescale ``s``, so the
  margin costs only the recorded amplitude deficit.
- ``build_control_diagonal_generator_encoding``: block encoding of the LCHS
  effective Hamiltonian, adapted from Pocrnic, Johnson, Katabarwa, and
  Wiebe, arXiv:2506.20760v2, Lemma 7 (p. 15).

The qubitized walk of Low and Chuang, arXiv:1610.06546v3, is an alternative
route to Hamiltonian evolution that this module does not implement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Any, TYPE_CHECKING

import numpy as np
import scipy.special
from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes
from nwqlib.subroutines.qsp.phases import (
    _QSPEvaluationsExhausted,
    _QSPPhaseConvergenceError,
    SymmetricQSPPhases,
    chebyshev_polynomial_sup_bound,
    solve_symmetric_qsp_phases,
    wx_phases_to_reflection,
)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit, QuantumRegister
    from nwqlib.subroutines.block_encoding import BlockEncoding

# Engineering constants (registered in docs/ENGINEERING_CONSTANTS.md):
# - QSP_EVOLUTION_TARGET_MARGINS (1e-3, 3e-3): the truncated cosine reaches |f| = 1 at
#   x = 0, and the sine does for tau >= pi/2. Dong, Meng, Whaley and Lin (arXiv:2002.11649v2, Sec. IV.5 and
#   Fig. 13) report that the Hessian condition number at the optimum grows
#   like eta**-gamma with gamma > 1 as max|f| = 1 - eta approaches 1.
#   Dividing by s = max(1, cos bound, sin bound) * (1 + margin) (see
#   _target_scale) keeps each target's max|f| at most 1/(1 + margin). In the
#   tau x epsilon sweep registered in docs/ENGINEERING_CONSTANTS.md (tau up
#   to 10 at five tolerances from 1e-2 to 1e-9, and up to 100 at 1e-3 and
#   1e-9), every target converged at 1e-3 with the Newton start of
#   phases.py, so 3e-3 is an unused fallback there. The cost is only the
#   recorded quadratic OAA amplitude deficit (~6*(margin/2)^2). Revisit if
#   the solver or the target scale changes.
# - QSP_EVOLUTION_BESSEL_TAIL_TERMS 200: the adaptive finite-suffix search may extend
#   through ceil(tau) + 200.  It stops at the first terminal that certifies a
#   degree, keeping the analytic remainder representable whenever binary64
#   permits.  The constant is a preprocessing search horizon, not a tail
#   truncation.
QSP_EVOLUTION_TARGET_MARGINS = (1.0e-3, 3.0e-3)
QSP_EVOLUTION_BESSEL_TAIL_TERMS = 200
# Paper constant, not an engineering choice: 3-step robust OAA multiplies
# block-encoding queries by exactly 3 ([GSLW] arXiv:1806.01838v1,
# Theorem 28 with n = 3, whose
# T_3 sequence applies A, A^dagger, A).
QSP_OAA_QUERY_MULTIPLIER = 3

_ABSTRACT_OPERATOR_ERROR_PROVENANCE = (
    "abstract_operator_bound_physical_generator_frame"
)


def _combined_operator_certificate_metadata(
    *children: BlockEncoding,
) -> dict[str, Any]:
    """Propagate abstract-error provenance and the Hermitian premise."""

    provenance_available = all(
        child.metadata.get("error_bound_provenance")
        == _ABSTRACT_OPERATOR_ERROR_PROVENANCE
        for child in children
    )
    def premise(name):
        values = tuple(child.metadata.get(name) for child in children)
        if all(value is True for value in values):
            return True
        # With nonzero real weights, one non-Hermitian summand cannot lose
        # its anti-Hermitian part by adding only Hermitian summands.
        if sum(value is False for value in values) == 1 and all(type(value) is bool for value in values):
            return False
        return None
    return {
        "error_bound_provenance": (
            _ABSTRACT_OPERATOR_ERROR_PROVENANCE if provenance_available else None
        ),
        # A real linear combination preserves a proved premise. Failure to
        # prove each child says nothing about cancellations in their sum.
        **{
            name: premise(name)
            for name in ("target_operator_is_hermitian", "encoded_operator_is_hermitian")
        },
    }


_GeneratorBranch = tuple[
    "BlockEncoding", float, "QuantumCircuit", str, tuple[Any, ...]
]


def _combine_generator_children(
    *,
    branches: tuple[_GeneratorBranch, ...],
    num_ancillas: int,
    system_qubits: int,
    implementation: str,
    metadata: dict[str, Any],
    circuit: QuantumCircuit | None = None,
    combine_qubit: Any | None = None,
    max_work: int = 1_000_000_000,
    max_bytes: int = DEFAULT_INPUT_BYTES,
    dense_control_route: str = "auto",
) -> BlockEncoding:
    """Combine the child block encodings as an LCU with real weights and return the combined ``BlockEncoding``.

    Each branch is ``(child, weight, circuit, label, qubits)`` with real weight.
    The combined normalization is the LCU subnormalization
    ``sum_j w_j alpha_j`` (docs/conventions.md, "Block encodings and QSP"),
    and the operator-error bound is ``sum_j w_j epsilon_j`` over nonzero
    weights, unknown when any such child bound is unknown. For two branches
    the combine qubit is prepared by ``RY(theta)`` with
    ``tan(theta/2) = sqrt(w_1 alpha_1 / (w_0 alpha_0))``, then each branch is
    applied under its control value and the rotation is undone. Controlling
    a branch synthesizes the dense unitaries it holds on the route that
    ``dense_control_route`` selects for one control
    (``qiskit_compat.controlled``), which ``max_work`` and ``max_bytes``
    bound before the first branch (``qiskit_compat.dense_control_charges``).
    """

    from nwqlib.subroutines.block_encoding import BlockEncoding
    from nwqlib.subroutines.qiskit_compat import controlled

    children = tuple(branch[0] for branch in branches)
    alpha_terms = tuple(branch[1] * branch[0].alpha for branch in branches)
    alpha = float(sum(alpha_terms))
    error_bound = (
        None if any(branch[1] != 0 and branch[0].error_bound is None for branch in branches)
        else float(sum(branch[1] * branch[0].error_bound for branch in branches if branch[1] != 0))
    )
    if len(branches) == 1:
        circuit = branches[0][2]
    else:
        assert circuit is not None and combine_qubit is not None
        from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses
        from nwqlib.subroutines.qiskit_compat import dense_control_charges
        amplitudes = np.sqrt(np.asarray(alpha_terms) / alpha)
        theta = 2.0 * float(np.arctan2(amplitudes[1], amplitudes[0]))
        branch_gates = [branch[2].to_gate(label=branch[3]) for branch in branches]
        charges = [dense_control_charges(gate, 1, dense_control_route) for gate in branch_gates]
        admit_dense_syntheses([width for charge in charges for width in charge[0]],
                              controlled_qubit_counts=[width for charge in charges for width in charge[1]],
                              controls=[charge[2] for charge in charges],
                              max_work=max_work, max_bytes=max_bytes,
                              operation="joint-generator dense synthesis")
        circuit.ry(theta, combine_qubit)
        for branch_index, (branch_gate, branch) in enumerate(zip(branch_gates, branches, strict=True)):
            circuit.append(
                controlled(branch_gate, 1, ctrl_state=branch_index, route=dense_control_route),
                [combine_qubit, *branch[4]],
            )
        circuit.ry(-theta, combine_qubit)
    combined_metadata = dict(metadata)
    if len(children) == 1:
        combined_metadata["degenerate_single_term"] = True
    combined_metadata["child_implementations"] = [
        child.implementation for child in children
    ]
    combined_metadata.update(
        _combined_operator_certificate_metadata(*(branch[0] for branch in branches if branch[1] != 0))
    )
    assert circuit is not None
    return BlockEncoding(
        circuit=circuit,
        alpha=alpha,
        num_ancillas=num_ancillas,
        system_qubits=system_qubits,
        error_bound=error_bound,
        implementation=implementation,
        metadata=combined_metadata,
    )


@dataclass(frozen=True, kw_only=True)
class JacobiAngerExpansion:
    """Truncated Jacobi-Anger expansion of `exp(-i tau x)` with its Bessel tail bounds.

    [`jacobi_anger_expansion`][nwqlib.subroutines.qsp.evolution.jacobi_anger_expansion]
    returns it. The polynomial is in `cos_coefficients` and
    `sin_coefficients`, and its truncation error bound is `tail_bound`. The
    fields below are read-only.

    Attributes:
        tau: Effective evolution time `alpha * t`.
        epsilon: Requested truncation tolerance for the combined expansion.
        degree: Smallest degree at or above the `min_degree` supplied to
            `jacobi_anger_expansion` whose computed combined tail meets
            `epsilon` at the selected `analysis_terminal`, defined below.
            This is not a global minimum over all possible terminals.
        cos_degree: Even-part polynomial degree.
        sin_degree: Odd-part polynomial degree.
        cos_coefficients: Chebyshev coefficients of the truncated cosine.
        sin_coefficients: Chebyshev coefficients of the truncated sine.
        tail_bound: Upper bound on the combined Bessel tail
            ``2 sum_{k>degree} |J_k|``, computed as the finite sum through
            ``analysis_terminal`` plus ``analytic_infinite_tail_bound``.
        tail_slack: ``epsilon / tail_bound``, the margin to the truncation
            boundary. A degree selected with slack near one can change under
            a different Bessel-function implementation, so recorded integer
            degrees are meaningful only together with this slack.
        cos_tail_bound: Complete even-part finite-plus-infinite tail bound.
        sin_tail_bound: Complete odd-part finite-plus-infinite tail bound.
        analysis_terminal: Last Bessel order included in the finite sums
            used for the reported tail bounds.
        analytic_infinite_tail_bound: Combined analytic infinite-tail term
            added across both parity bounds, or ``None`` when that
            mathematically nonzero term is not representable in binary64.
            On that path, all tail bounds and tail_slack are also None.
    """

    tau: float
    epsilon: float
    degree: int
    cos_degree: int
    sin_degree: int
    cos_coefficients: tuple[float, ...]
    sin_coefficients: tuple[float, ...]
    tail_bound: float | None
    tail_slack: float | None
    cos_tail_bound: float | None
    sin_tail_bound: float | None
    analysis_terminal: int
    analytic_infinite_tail_bound: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like expansion record."""

        return {
            "tau": self.tau,
            "epsilon": self.epsilon,
            "degree": self.degree,
            "cos_degree": self.cos_degree,
            "sin_degree": self.sin_degree,
            "tail_bound": self.tail_bound,
            "tail_slack": self.tail_slack,
            "cos_tail_bound": self.cos_tail_bound,
            "sin_tail_bound": self.sin_tail_bound,
            "analysis_terminal": self.analysis_terminal,
            "analytic_infinite_tail_bound": self.analytic_infinite_tail_bound,
        }


def _analytic_jacobi_anger_remainder(tau: float, terminal: int) -> float:
    """Return the positive-order suffix bound after ``terminal``.

    Returns ``R >= sum_{k > T} |J_k(tau)|`` with ``T = terminal``. NWQLib
    derives it from the power series of ``J_k``: ``(m+k)!/k! >= (k+1)^m``
    gives ``|J_k(tau)| <= (tau/2)^k exp(tau^2/(4(k+1))) / k!``, and
    consecutive terms of that bound shrink by at most
    ``q = tau/(2(T+2)) < 1`` for ``k > T``. Summing the geometric series,
    ``R = (tau/2)^(T+1) exp(tau^2/(4(T+2))) / ((T+1)! (1-q))``.
    [GSLW] arXiv:1806.01838v1, Eq. (55), gives the smaller real-argument bound
    ``|J_m(t)| <= (t/2)^m / m!``. The same geometric sum built from Eq. (55)
    is smaller than ``R`` by the factor ``exp(tau^2/(4(T+2)))``, so ``R`` is
    a conservative choice.

    The logarithm stays internal so a binary64 underflow is never presented
    as a usable logarithmic certificate alongside a zero ordinary bound.
    Callers keep the zero as an unusable-certificate marker and do not
    replace it by an arbitrary tiny value.
    """

    remainder_ratio = tau / (2.0 * (terminal + 2))
    if not remainder_ratio < 1.0:
        raise ValueError("Jacobi-Anger analysis terminal must satisfy q < 1")
    first_analytic_order = terminal + 1
    log_remainder = (
        first_analytic_order * math.log(tau / 2.0)
        - math.lgamma(first_analytic_order + 1)
        + tau**2 / (4.0 * (first_analytic_order + 1))
        - math.log1p(-remainder_ratio)
    )
    try:
        return math.exp(log_remainder)
    except OverflowError:
        return float("inf")


def jacobi_anger_expansion(
    tau: float,
    epsilon: float,
    *,
    min_degree: int = 1,
    max_degree: int = 256,
    max_bytes: int = DEFAULT_INPUT_BYTES,
) -> JacobiAngerExpansion:
    """Truncate the Jacobi-Anger expansion by the Bessel tail bound.

    ``exp(-i tau x) = cos(tau x) - i sin(tau x)`` with
    ``cos(tau x) = J_0 + 2 sum_{k>=1} (-1)^k J_2k T_2k`` and
    ``sin(tau x) = 2 sum_{k>=0} (-1)^k J_2k+1 T_2k+1`` ([GSLW] Lemma 57,
    arXiv:1806.01838v1, pp. 49-50). Truncating both parities at ``k <= d``
    leaves error
    at most ``2 sum_{k>d} |J_k(tau)|`` because ``|T_k| <= 1`` on ``[-1, 1]``
    (Eqs. (53)-(54)).
    The bound adds the Bessel magnitudes through the selected terminal
    ``T``, the last order included in its finite sum, and bounds the rest by
    an analytic remainder (NWQLib's power-series bound, for which [GSLW]
    Eq. (55) is the sharper real-argument form). For each tested terminal,
    the search considers degrees from ``min_degree`` through
    ``min(max_degree, T - 1)``. It chooses the first tested terminal whose
    computed combined tail permits one of those degrees at tolerance ``epsilon``.
    Each parity tail carries the whole analytic suffix, so the combined
    ``tail_bound`` counts it twice, which is conservative.
    The scaling of [GSLW] arXiv:1806.01838v1, Cor. 60,
    ``d = Theta(tau + log(1/eps)/log(e + log(1/eps)/tau))``, is the
    documented reference form for the resulting degree. ``min_degree`` floors the selected
    degree. The phase solver needs ``min_degree=2`` so the even (cos) part
    is never a constant.
    ``max_degree`` bounds the returned polynomial degree, while the finite
    search evaluates at most ``max_degree + 201`` distinct Bessel orders,
    from zero through at most ``max_degree + 200``. ``max_bytes``
    bounds known numerical arrays, not special-function workspaces or
    process memory.

    Args:
        tau (float): Effective evolution time `alpha * t`.
        epsilon (float): Truncation tolerance for the combined expansion.
        min_degree (int): Default `1`. Smallest returned degree.
        max_degree (int): Default `256`. Largest returned degree.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on the known numerical arrays.

    Returns:
        expansion (JacobiAngerExpansion): Cosine and sine Chebyshev
            coefficients with their tail bounds and `tail_slack`.
    """

    tau = finite_real(tau, "tau")
    epsilon = finite_real(epsilon, "epsilon")
    min_degree = integer(min_degree, "min_degree", 1)
    max_degree = integer(max_degree, "max_degree", min_degree)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in the open interval (0, 1)")
    terminal_limit = min(math.ceil(tau), max_degree) + QSP_EVOLUTION_BESSEL_TAIL_TERMS
    # 96 bytes (twelve 8-byte slots) per Bessel order 0..terminal_limit. The
    # arrays alive together need fewer slots: order indices, Bessel values and
    # magnitudes, the reversed cumulative sums and doubled suffix tails of two
    # consecutive terminals, the two coefficient tables and the parity gathers.
    _check_bytes(96 * (terminal_limit + 1), max_bytes, "Jacobi-Anger selection")
    # The analytic remainder needs q = tau/(2(T+2)) < 1.
    first_terminal = min_degree + 1
    while first_terminal <= terminal_limit and tau / (2.0 * (first_terminal + 2)) >= 1.0:
        first_terminal += 1
    if first_terminal > terminal_limit:
        raise ValueError(f"no Jacobi-Anger degree within max_degree={max_degree} can be certified by the bounded search")
    all_magnitudes = np.abs(
        scipy.special.jv(np.arange(terminal_limit + 1), tau)
    )
    degree = None
    analysis_terminal = None
    analytic_remainder = None
    suffix_tails = None
    magnitudes = None
    for candidate_terminal in range(first_terminal, terminal_limit + 1):
        candidate_remainder = _analytic_jacobi_anger_remainder(
            tau,
            candidate_terminal,
        )
        # Every candidate degree is below this terminal T, so its combined tail
        # contains at least the order-T term 2|J_T| plus the analytic term 4R.
        # No degree can pass at T when that lower bound already exceeds eps.
        if (
            2.0 * float(all_magnitudes[candidate_terminal])
            + 4.0 * candidate_remainder
            > epsilon
        ):
            continue
        candidate_magnitudes = all_magnitudes[: candidate_terminal + 1]
        # suffix_tails[k] = 2 sum_{j=k..T} |J_j|, the finite part of the
        # combined tail of degree k - 1 (Eqs. (53)-(54) summed over parities).
        candidate_suffix_tails = (
            2.0 * np.cumsum(candidate_magnitudes[::-1])[::-1]
        )
        # Each parity tail carries 2R for the whole analytic suffix R, so the
        # combined tail carries 4R.
        combined_candidate_tail = 4.0 * candidate_remainder
        for candidate in range(min_degree, min(candidate_terminal, max_degree + 1)):
            complete_tail = (
                float(candidate_suffix_tails[candidate + 1])
                + combined_candidate_tail
            )
            if complete_tail <= epsilon:
                degree = candidate
                break
        if degree is None:
            continue
        analysis_terminal = candidate_terminal
        analytic_remainder = candidate_remainder
        suffix_tails = candidate_suffix_tails
        magnitudes = candidate_magnitudes
        break
    if degree is None:
        raise ValueError(
            f"no Jacobi-Anger degree within max_degree={max_degree} reaches eps={epsilon} in the bounded search"
        )
    assert analysis_terminal is not None
    assert analytic_remainder is not None
    assert suffix_tails is not None
    assert magnitudes is not None
    parity_analytic_tail = 2.0 * analytic_remainder
    combined_analytic_tail = 2.0 * parity_analytic_tail
    analytic_tail_usable = combined_analytic_tail > 0.0
    tail_bound = (
        float(suffix_tails[degree + 1]) + combined_analytic_tail
        if analytic_tail_usable
        else None
    )

    cos_degree = degree if degree % 2 == 0 else degree - 1
    sin_degree = degree if degree % 2 == 1 else degree - 1
    cos_coefficients = np.zeros(cos_degree + 1)
    cos_coefficients[0] = float(scipy.special.jv(0, tau))
    for k in range(1, cos_degree // 2 + 1):
        cos_coefficients[2 * k] = 2.0 * (-1.0) ** k * float(scipy.special.jv(2 * k, tau))
    sin_coefficients = np.zeros(sin_degree + 1)
    for k in range((sin_degree - 1) // 2 + 1):
        sin_coefficients[2 * k + 1] = (
            2.0 * (-1.0) ** k * float(scipy.special.jv(2 * k + 1, tau))
        )
    even_indices = np.arange(cos_degree + 2, analysis_terminal + 1, 2)
    odd_indices = np.arange(sin_degree + 2, analysis_terminal + 1, 2)
    cos_tail = (
        2.0 * float(np.sum(magnitudes[even_indices])) + parity_analytic_tail
        if analytic_tail_usable
        else None
    )
    sin_tail = (
        2.0 * float(np.sum(magnitudes[odd_indices])) + parity_analytic_tail
        if analytic_tail_usable
        else None
    )
    return JacobiAngerExpansion(
        tau=float(tau),
        epsilon=float(epsilon),
        degree=int(degree),
        cos_degree=int(cos_degree),
        sin_degree=int(sin_degree),
        cos_coefficients=tuple(float(value) for value in cos_coefficients),
        sin_coefficients=tuple(float(value) for value in sin_coefficients),
        tail_bound=tail_bound,
        tail_slack=None if tail_bound is None else float(epsilon / tail_bound),
        cos_tail_bound=cos_tail,
        sin_tail_bound=sin_tail,
        analysis_terminal=analysis_terminal,
        analytic_infinite_tail_bound=(
            combined_analytic_tail if analytic_tail_usable else None
        ),
    )


def _append_projector_phase(
    circuit: QuantumCircuit,
    signal: Any,
    block_ancillas: list[Any],
    angle: float,
    *,
    pair_qubit: Any = None,
) -> None:
    """Append ``e^{i angle (2 Pi - 1)}`` with ``Pi`` = ancillas all zero.

    This is the projector-controlled phase of [MRTC] arXiv:2105.02859v5,
    Eq. (27) and Fig. 3, realized as in [GSLW] arXiv:1806.01838v1 (text
    before Lemma 19) by ``C_Pi NOT``, a
    single-qubit ``e^{-i angle Z}`` on the flag, and ``C_Pi NOT`` again.
    With ``pair_qubit`` set, a CNOT conjugation negates the angle on the
    pair-one branch (the shared ladder of ``build_real_chebyshev_encoding``,
    which runs the ``+Phi`` and ``-Phi`` passes together).
    """

    from nwqlib.subroutines.qiskit_compat import controlled
    from qiskit.circuit.library import RZGate, XGate

    if not block_ancillas:
        # Pi is the whole space: the phase is global (pair branch handled by
        # conjugating a plain X flip of the signal qubit).
        if pair_qubit is None:
            circuit.global_phase += angle
            return
        circuit.x(signal)
        circuit.cx(pair_qubit, signal)
        circuit.append(RZGate(2.0 * angle), [signal])
        circuit.cx(pair_qubit, signal)
        circuit.x(signal)
        return
    flip = controlled(XGate(), len(block_ancillas), ctrl_state=0)
    circuit.append(flip, [*block_ancillas, signal])
    if pair_qubit is not None:
        circuit.cx(pair_qubit, signal)
    # e^{-i angle Z} on the flipped signal: the Pi subspace (signal one)
    # acquires e^{+i angle}, its complement e^{-i angle}.
    circuit.append(RZGate(2.0 * angle), [signal])
    if pair_qubit is not None:
        circuit.cx(pair_qubit, signal)
    circuit.append(flip, [*block_ancillas, signal])


def _block_gate_pair(
    encoding: BlockEncoding, *, native_ucg: bool = True
) -> tuple[Any, Any]:
    """Return the child gate and its inverse for one construction.

    Controlled parity passes use ``native_ucg=False`` so the child UCG
    definitions are realized before the passes copy their occurrences.
    Uncontrolled QSVT and real-polynomial circuits can keep native tables.
    """
    from nwqlib.subroutines.qiskit_compat import inverse_realized_gate

    forward = encoding.circuit.to_gate(label="U_BE")
    return forward, inverse_realized_gate(forward, native_ucg=native_ucg)


def build_qsvt_circuit(encoding: BlockEncoding, wx_phases: Any) -> QuantumCircuit:
    """Build the projector-controlled-phase QSVT circuit for Wx phases.

    The all-zero block (signal + block-encoding ancillas) equals the Wx
    polynomial ``P(A / alpha)`` exactly (for Hermitian encoded blocks and
    definite-parity-compatible phases), as in [MRTC] arXiv:2105.02859v5,
    Theorem 3. Trivial
    (all-zero) Wx phases realize the Chebyshev polynomial ``T_d(A / alpha)``
    ([MRTC] arXiv:2105.02859v5, Sec. II.A, p. 3). That known answer checks
    the phase, sign and
    ordering conventions of this module at circuit level.

    Args:
        encoding: Block encoding of the operator to transform, including its normalization and ancilla layout.
        wx_phases: Wx-convention phase vector ``(phi_0, ..., phi_d)``.

    Returns:
        circuit (QuantumCircuit): Circuit on ``1 + num_ancillas + system_qubits``
            qubits with the signal qubit first (register order: signal, block
            ancillas, system).
    """

    from qiskit import QuantumCircuit, QuantumRegister

    block_gate, inverse_block_gate = _block_gate_pair(encoding)
    reflection_phases, global_phase = wx_phases_to_reflection(wx_phases)
    degree = len(reflection_phases) - 1
    signal = QuantumRegister(1, "qsp_signal")
    registers = [signal]
    ancillas = None
    if encoding.num_ancillas:
        ancillas = QuantumRegister(encoding.num_ancillas, "be_ancilla")
        registers.append(ancillas)
    system = QuantumRegister(encoding.system_qubits, "system")
    registers.append(system)
    circuit = QuantumCircuit(*registers, name="qsvt")
    ancilla_list = [*ancillas] if ancillas else []
    block_qubits = ancilla_list + [*system]
    circuit.global_phase += global_phase
    _append_projector_phase(circuit, signal[0], ancilla_list, float(reflection_phases[0]))
    for j in range(1, degree + 1):
        circuit.append(block_gate if j % 2 == 1 else inverse_block_gate, block_qubits)
        _append_projector_phase(circuit, signal[0], ancilla_list, float(reflection_phases[j]))
    return circuit


def _build_real_chebyshev_encoding(
    encoding: BlockEncoding,
    wx_phases: Any,
    *,
    block_gates: tuple[Any, Any],
) -> QuantumCircuit:
    """Assemble the circuit of ``build_real_chebyshev_encoding`` from a realized child-gate pair.

    ``build_qsp_evolution_encoding`` calls this for its cosine and sine
    passes with one ``_block_gate_pair``, so the child encoding is converted
    to a gate once per evolution.
    """

    from qiskit import QuantumCircuit, QuantumRegister

    block_gate, inverse_block_gate = block_gates
    reflection_phases, global_phase = wx_phases_to_reflection(wx_phases)
    degree = len(reflection_phases) - 1
    pair = QuantumRegister(1, "qsp_pair")
    signal = QuantumRegister(1, "qsp_signal")
    registers = [pair, signal]
    ancillas = None
    if encoding.num_ancillas:
        ancillas = QuantumRegister(encoding.num_ancillas, "be_ancilla")
        registers.append(ancillas)
    system = QuantumRegister(encoding.system_qubits, "system")
    registers.append(system)
    circuit = QuantumCircuit(*registers, name="qsp_real_pass")
    ancilla_list = [*ancillas] if ancillas else []
    block_qubits = ancilla_list + [*system]

    circuit.h(pair[0])
    circuit.global_phase += global_phase
    if degree % 2 == 1:
        circuit.z(pair[0])
    _append_projector_phase(
        circuit, signal[0], ancilla_list, float(reflection_phases[0]), pair_qubit=pair[0]
    )
    for j in range(1, degree + 1):
        circuit.append(block_gate if j % 2 == 1 else inverse_block_gate, block_qubits)
        _append_projector_phase(
            circuit, signal[0], ancilla_list, float(reflection_phases[j]), pair_qubit=pair[0]
        )
    circuit.h(pair[0])
    return circuit


def build_real_chebyshev_encoding(encoding: BlockEncoding, wx_phases: Any) -> QuantumCircuit:
    """Block-encode ``Re P(A / alpha)`` via the shared-ladder ``+-Phi`` pair.

    This is [GSLW] arXiv:1806.01838v1, Corollary 18, Eq. (33):
    ``Re P = (P + P^*)/2``, and the
    phases ``-Phi`` realize ``P^*``.
    One pair ancilla in ``H .. H`` averages the ``+Phi`` and ``-Phi`` QSVT
    passes. Both branches share the ``U_BE`` ladder because a CNOT from the
    pair qubit negates every reflection phase on the pair-one branch, and a
    ``Z`` on the pair qubit for odd degree absorbs the ``(-1)^d`` of the
    conjugated pass. Register order: pair, signal, block ancillas, system.

    Args:
        encoding (BlockEncoding): Block encoding of the operator to
            transform, as for `build_qsvt_circuit`.
        wx_phases (array_like): Wx-convention phase vector
            `(phi_0, ..., phi_d)`.

    Returns:
        circuit (QuantumCircuit): The block encoding of ``Re P(A / alpha)``,
            with register order pair, signal, block ancillas, system.
    """

    return _build_real_chebyshev_encoding(
        encoding,
        wx_phases,
        block_gates=_block_gate_pair(encoding),
    )


def _as_control_diagonal(values: Any, *, name: str) -> np.ndarray:
    """Admit real coefficients of a Hermitian generator without projecting inputs.

    A nonzero imaginary component changes the generator. Removing it would
    need a perturbation budget and would not establish the Hermitian target
    premise required by GSLW arXiv:1806.01838v1, Lemma 61. This checks the
    stored scalar table.
    """

    diagonal = np.asarray(values, dtype=complex).reshape(-1)
    if diagonal.size < 1 or diagonal.size & (diagonal.size - 1):
        raise ValueError(f"{name} must have a positive power-of-two length")
    if not np.all(np.isfinite(diagonal)):
        raise ValueError(f"{name} entries must be finite")
    if np.any(diagonal.imag != 0.0):
        raise ValueError(f"{name} entries must be real (the joint generator must stay Hermitian)")
    return diagonal.real.astype(float)


def _diagonal_rotation_circuit(
    diagonal: np.ndarray,
    max_abs: float,
    *,
    control_qubits: int,
) -> QuantumCircuit:
    """One-ancilla block encoding of ``diag(d_b) / max_abs`` by multiplexed RY.

    For control state ``b`` the rotation ancilla acquires
    ``cos(theta_b / 2) = d_b / max_abs`` on ``|0>``, so the all-zero
    rotation-ancilla block is the rescaled diagonal, signs included because
    ``arccos`` covers ``[-1, 1]``. Register order: rotation ancilla first,
    then the control register (the multiplexer's select lines).
    """

    from nwqlib.subroutines._multiplexors import append_uniformly_controlled_ry
    from qiskit import QuantumCircuit, QuantumRegister

    angles = 2.0 * np.arccos(np.clip(diagonal / max_abs, -1.0, 1.0))
    rotation = QuantumRegister(1, "diag_rotation")
    registers = [rotation]
    control = None
    if control_qubits:
        control = QuantumRegister(control_qubits, "generator_control")
        registers.append(control)
    circuit = QuantumCircuit(*registers, name="control_diagonal_rotation")
    append_uniformly_controlled_ry(
        circuit,
        rotation[0],
        tuple(control or ()),
        angles,
    )
    return circuit


def _control_diagonal_generator_layout(
    encoding_l: Any,
    encoding_h: Any | None,
    *,
    l_diagonal: Any,
    h_diagonal: Any | None,
) -> dict[str, Any]:
    """Resolve the joint-generator registers and normalization without circuits.

    Returns a dict with ``control_qubits`` (``log2`` of the diagonal length),
    ``l_diagonal_max_abs`` and ``h_diagonal_max_abs`` (``max |D_L|`` and
    ``max |D_H|``), ``alpha = max|D_L| alpha_L + max|D_H| alpha_H`` over the
    nonvanishing branches, ``child_count`` (one or two nonvanishing
    branches) and ``num_ancillas`` (the larger child ancilla count plus the
    shared rotation ancilla, plus the LCU combine qubit when both branches
    are active). ``build_control_diagonal_generator_encoding`` and
    ``algorithms/lchs/compiled_selection.py`` read the same values.
    """

    d_l = _as_control_diagonal(l_diagonal, name="l_diagonal")
    if (encoding_h is None) != (h_diagonal is None):
        raise ValueError("h_diagonal must be supplied exactly when encoding_h is supplied")
    d_h = None if h_diagonal is None else _as_control_diagonal(h_diagonal, name="h_diagonal")
    if d_h is not None and d_h.size != d_l.size:
        raise ValueError("l_diagonal and h_diagonal must have the same length")
    if encoding_h is not None and encoding_l.system_qubits != encoding_h.system_qubits:
        raise ValueError("L and H encodings must share the system size")
    maxima = (float(np.max(np.abs(d_l))), 0.0 if d_h is None else float(np.max(np.abs(d_h))))
    active = tuple(
        (child, scale)
        for child, scale in zip((encoding_l, encoding_h), maxima, strict=True)
        if child is not None and scale > 0.0
    )
    if not active:
        raise ValueError("joint generator D_L (x) L + D_H (x) H vanished (both diagonals zero)")
    return {
        "control_qubits": int(np.log2(d_l.size)),
        "l_diagonal_max_abs": maxima[0],
        "h_diagonal_max_abs": maxima[1],
        "alpha": float(sum(child.alpha * scale for child, scale in active)),
        "child_count": len(active),
        "num_ancillas": max(child.num_ancillas for child, _ in active) + len(active),
    }


def build_control_diagonal_generator_encoding(
    encoding_l: BlockEncoding,
    encoding_h: BlockEncoding | None,
    *,
    l_diagonal: Any,
    h_diagonal: Any | None = None,
    max_work: int = 1_000_000_000,
    max_bytes: int = DEFAULT_INPUT_BYTES,
    dense_control_route: str = "auto",
) -> BlockEncoding:
    """Block-encode the joint generator ``D_L (x) L + D_H (x) H``.

    Following the effective-Hamiltonian SELECT route of Pocrnic et al.,
    arXiv:2506.20760v2, Lemma 7 (p. 15), adapted to the library's separate ``BE(L)``/``BE(H)``
    children, the LCHS node coefficients ``(k, 1)`` become real diagonals ``D_L``,
    ``D_H`` on a control register that joins the encoded operator's system
    space. Length-one diagonals give the scalar generator ``k L + H``. Each
    child branch is a one-ancilla multiplexed-RY encoding of its rescaled
    diagonal tensored with the child encoding, and one LCU combine qubit sums
    the branches, so

    ``alpha = max|D_L| * alpha_L + max|D_H| * alpha_H``.

    A child with fewer ancillas uses a prefix of the shared ancilla register,
    and the idle ancillas keep its all-zero block. Encoded system order: control
    register first (low-order), then the child system, so the encoded matrix
    is ``kron(L, D_L) + kron(H, D_H)`` in numpy index convention. Zero
    diagonal entries make their control state evolve trivially under
    downstream Hamiltonian simulation (identity padding).

    Args:
        encoding_l: Block encoding of the Hermitian part ``L``.
        encoding_h: Block encoding of ``H``, or ``None`` when ``H = 0``.
        l_diagonal: Real diagonal for the ``L`` branch, power-of-two length
            ``2**control_qubits`` indexed by the control-register basis.
        h_diagonal: Real diagonal for the ``H`` branch (same length),
            required exactly when ``encoding_h`` is supplied.
        max_work: Default `1_000_000_000`. Limit on the work of the exact
            dense syntheses that controlling the two branches makes, and of
            Qiskit's control of the synthesized gates, in the work units of
            [Exact dense synthesis](../../development/dense_synthesis.md#admission-of-the-exact-synthesis).
            A ``dense_dilation`` child is synthesized and controlled once. A
            single active branch is not controlled here.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Limit on
            the working and kept bytes of those syntheses and control steps.
        dense_control_route: Default ``"auto"``. ``"gatewise"``,
            ``"whole_matrix"`` or ``"auto"``, the route by which the combine
            qubit controls a ``dense_dilation`` child
            ([dense control route](../../development/dense_synthesis.md#dense-control-route)).
            ``"auto"`` takes the whole-matrix route for this one control.
            The child's dilation is then synthesized with the control, and
            the branch's diagonal rotation is controlled gate-wise. A child
            whose circuit is not one dense unitary is controlled gate-wise on
            either route.

    Returns:
        encoding (BlockEncoding): BlockEncoding of the joint generator, with
            implementation ``"control_diagonal_generator_lcu"``. An all-zero
            diagonal on one branch degenerates to the other branch alone.
    """

    from qiskit import QuantumCircuit, QuantumRegister

    d_l = _as_control_diagonal(l_diagonal, name="l_diagonal")
    d_h = None if h_diagonal is None else _as_control_diagonal(h_diagonal, name="h_diagonal")
    layout = _control_diagonal_generator_layout(
        encoding_l, encoding_h, l_diagonal=d_l, h_diagonal=d_h
    )
    control_qubits = layout["control_qubits"]
    d_l_max = layout["l_diagonal_max_abs"]
    d_h_max = layout["h_diagonal_max_abs"]

    def _single_term(
        child: BlockEncoding,
        diagonal: np.ndarray,
        max_abs: float,
        branch_name: str,
    ) -> BlockEncoding:
        """Tensor the normalized diagonal rotation with one child encoding when the other branch vanishes."""
        rotation_block = _diagonal_rotation_circuit(
            diagonal, max_abs, control_qubits=control_qubits
        )
        registers: list[QuantumRegister] = [QuantumRegister(1, "diag_rotation")]
        ancillas = None
        if child.num_ancillas:
            ancillas = QuantumRegister(child.num_ancillas, "be_ancilla")
            registers.append(ancillas)
        control = None
        if control_qubits:
            control = QuantumRegister(control_qubits, "generator_control")
            registers.append(control)
        system = QuantumRegister(child.system_qubits, "system")
        registers.append(system)
        circuit = QuantumCircuit(*registers, name="control_diagonal_generator")
        rotation_qubit = registers[0][0]
        circuit.compose(
            rotation_block,
            qubits=[rotation_qubit, *(control or [])],
            inplace=True,
        )
        circuit.append(
            child.circuit.to_gate(label=f"U_{branch_name}"),
            [*(ancillas or []), *system],
        )
        num_ancillas = 1 + child.num_ancillas
        return _combine_generator_children(
            branches=(
                (
                    child,
                    max_abs,
                    circuit,
                    "",
                    (),
                ),
            ),
            num_ancillas=num_ancillas,
            system_qubits=control_qubits + child.system_qubits,
            implementation="control_diagonal_generator_lcu",
            metadata={
                "control_qubits": control_qubits,
                "l_diagonal_max_abs": d_l_max,
                "h_diagonal_max_abs": d_h_max,
            },
        )

    l_vanished = d_l_max == 0.0
    h_vanished = encoding_h is None or d_h_max == 0.0
    if h_vanished:
        return _single_term(encoding_l, d_l, d_l_max, "L")
    assert encoding_h is not None and d_h is not None
    if l_vanished:
        return _single_term(encoding_h, d_h, d_h_max, "H")

    num_child_ancillas = max(encoding_l.num_ancillas, encoding_h.num_ancillas)
    combine = QuantumRegister(1, "generator_lcu")
    rotation = QuantumRegister(1, "diag_rotation")
    registers = [combine, rotation]
    ancillas = None
    if num_child_ancillas:
        ancillas = QuantumRegister(num_child_ancillas, "be_ancilla")
        registers.append(ancillas)
    control = None
    if control_qubits:
        control = QuantumRegister(control_qubits, "generator_control")
        registers.append(control)
    system = QuantumRegister(encoding_l.system_qubits, "system")
    registers.append(system)
    circuit = QuantumCircuit(*registers, name="control_diagonal_generator")
    ancilla_list = [*ancillas] if ancillas else []
    control_list = [*control] if control else []

    # Construct the L and H tensor branches with the same control/system
    # ordering before their coefficient-weighted LCU combination.
    branches = []
    for child, diagonal, max_abs, branch_name in (
        (encoding_l, d_l, d_l_max, "L"),
        (encoding_h, d_h, d_h_max, "H"),
    ):
        branch_circuit = QuantumCircuit(
            rotation,
            *([ancillas] if ancillas else []),
            *([control] if control else []),
            system,
            name=f"generator_branch_{branch_name.lower()}",
        )
        branch_circuit.compose(
            _diagonal_rotation_circuit(diagonal, max_abs, control_qubits=control_qubits),
            qubits=[rotation[0], *control_list],
            inplace=True,
        )
        branch_circuit.append(
            child.circuit.to_gate(label=f"U_{branch_name}"),
            [*ancilla_list[: child.num_ancillas], *system],
        )
        branches.append(
            (
                child,
                max_abs,
                branch_circuit,
                f"U_D{branch_name}",
                tuple([rotation[0], *ancilla_list, *control_list, *system]),
            )
        )

    return _combine_generator_children(
        branches=tuple(branches),
        circuit=circuit,
        combine_qubit=combine[0],
        max_work=max_work,
        max_bytes=max_bytes,
        dense_control_route=dense_control_route,
        num_ancillas=num_child_ancillas + 2,
        system_qubits=control_qubits + encoding_l.system_qubits,
        implementation="control_diagonal_generator_lcu",
        metadata={
            "control_qubits": control_qubits,
            "l_diagonal_max_abs": d_l_max,
            "h_diagonal_max_abs": d_h_max,
            "alpha_l": encoding_l.alpha,
            "alpha_h": encoding_h.alpha,
        },
    )


def compiled_select_resource_law(
    *,
    control_qubits: int,
    cos_degree: int,
    sin_degree: int,
    child_count: int = 2,
) -> dict[str, int]:
    """Count the queries, child calls and rotation angles of LCHS's compiled QSP SELECT before gate synthesis.

    The compiled SELECT is one QSP Hamiltonian evolution of the joint
    generator built by
    [`build_control_diagonal_generator_encoding`][nwqlib.subroutines.qsp.evolution.build_control_diagonal_generator_encoding]:
    ``3 * (cos_degree + sin_degree)`` queries to the joint block encoding
    (the 3-step OAA over both parity ladders), and each query costs one call
    per child encoding plus one multiplexed RY of ``2**control_qubits``
    rotation angles per child. Counts are structural (rotation angles and
    child calls before gate synthesis), so they are qiskit-version stable.

    In Pocrnic et al., arXiv:2506.20760v2, Lemma 7, p. 15, each
    effective-Hamiltonian query uses one ``U_A`` plus one ``U_A^dagger`` call
    and ``2M`` multi-controlled rotations, where ``M`` is the number of
    quadrature nodes. Here each query uses ``child_count`` child calls and
    ``child_count * P`` multiplexed rotation angles. ``child_count`` is 2 for
    the ``L`` and ``H`` branches and 1 when ``H=0``, and
    ``P = 2**control_qubits`` is the number of address slots after
    power-of-two padding, so ``P >= M``. With two children, the ``2P``
    multiplexed rotation angle entries per joint-generator query are at least
    the paper's ``2M`` multi-controlled rotation calls per
    effective-Hamiltonian query. With one child, NWQLib uses one child call
    and ``P`` angle entries per query. These are counts of different rotation
    constructions and child encodings, so the comparison does not order their
    synthesized gate costs. The paper's GQSP and qubitization query count and
    amplification constants are not part of this count.
    """

    if control_qubits < 0:
        raise ValueError("control_qubits must be nonnegative")
    if cos_degree < 0 or sin_degree < 0:
        raise ValueError("polynomial degrees must be nonnegative")
    if child_count not in (1, 2):
        raise ValueError("child_count must be 1 (H = 0) or 2 (L and H branches)")
    queries = QSP_OAA_QUERY_MULTIPLIER * (cos_degree + sin_degree)
    # Per application of the half block, which OAA applies three times, the
    # cos and sin real passes have d_cos + 1 and d_sin + 1 projector phases,
    # two pair H gates each and one scalar phase each. The parity qubit adds
    # two H gates, and the odd sin pass adds one pair Z. Two OAA reflections
    # separate the three applications. The generator adds one rotation
    # ancilla, and one LCU combine qubit when both branches are present.
    return {
        "block_encoding_queries": queries,
        "child_encoding_calls": child_count * queries,
        "multiplexed_rotation_gates": child_count * queries,
        "multiplexed_rotation_angles": child_count * queries * 2**control_qubits,
        "generator_ancillas_beyond_children": 2 if child_count == 2 else 1,
        "projector_phase_count": queries + 2 * QSP_OAA_QUERY_MULTIPLIER,
        "pair_boundary_h_count": 4 * QSP_OAA_QUERY_MULTIPLIER,
        "parity_scalar_phase_count": 2 * QSP_OAA_QUERY_MULTIPLIER,
        "parity_boundary_h_count": 2 * QSP_OAA_QUERY_MULTIPLIER,
        "odd_pair_sign_count": QSP_OAA_QUERY_MULTIPLIER,
        "oaa_reflection_count": QSP_OAA_QUERY_MULTIPLIER - 1,
    }


def _target_scale(
    expansion: JacobiAngerExpansion,
    margin: float,
    *, max_degree: int, max_bytes: int,
) -> tuple[float, float, float]:
    """Return ``(s, cos_bound, sin_bound)`` for the truncated parity targets.

    ``cos_bound`` and ``sin_bound`` are the Chebyshev norming bounds on
    ``sup_{[-1, 1]}`` of the truncated cosine and sine series, and
    ``s = max(1, cos_bound, sin_bound) (1 + margin)`` is the common scale that
    divides both targets. The floor at one gives ``s >= 1``, hence
    ``a = 1/(2s) <= 1/2`` for the half-block amplitude, the range assumed by
    the OAA error expansion in ``qsp_evolution_error_terms``.
    """

    cos_bound = chebyshev_polynomial_sup_bound(expansion.cos_coefficients, max_degree=max_degree, max_bytes=max_bytes)
    sin_bound = chebyshev_polynomial_sup_bound(expansion.sin_coefficients, max_degree=max_degree, max_bytes=max_bytes)
    scale = max(1.0, cos_bound, sin_bound) * (1.0 + margin)
    return scale, cos_bound, sin_bound


@dataclass(frozen=True, kw_only=True)
class QSPPreparedEvolution:
    """The solved phases of both parity parts of a QSP evolution, with the common target scale.

    [`prepare_qsp_evolution`][nwqlib.subroutines.qsp.evolution.prepare_qsp_evolution]
    returns it, and
    [`qsp_evolution_error_terms`][nwqlib.subroutines.qsp.evolution.qsp_evolution_error_terms]
    reads it. It holds numbers only, no circuit. The coefficient and phase
    tuples are immutable. The fields below are read-only.

    Attributes:
        expansion: The Jacobi-Anger truncation whose parity targets were solved.
        cos_solution: Phases realizing ``Re P = cos-series / scale``.
        sin_solution: Phases realizing ``Re P = sin-series / scale``.
        scale: Common target divisor
            `s = max(1, cos bound, sin bound) * (1 + margin) >= 1`.
        target_norming_bounds: Norming bounds on the cos and sin series
            before division by ``scale``.
        margin: The target margin, from `QSP_EVOLUTION_TARGET_MARGINS` =
            (1e-3, 3e-3), for which both phase solves converged.
        attempted_margins: Every margin tried, in order, including failures.
    """

    expansion: JacobiAngerExpansion
    cos_solution: SymmetricQSPPhases
    sin_solution: SymmetricQSPPhases
    scale: float
    target_norming_bounds: tuple[float, float]
    margin: float
    attempted_margins: tuple[float, ...]

    def __post_init__(self) -> None:
        scale = finite_real(self.scale, "scale")
        if scale < 1:
            raise ValueError("scale must be at least one")
        if finite_real(self.margin, "margin") < 0:
            raise ValueError("margin must be nonnegative")
        if len(self.target_norming_bounds) != 2:
            raise ValueError("exactly two target norming bounds are required")
        bounds = tuple(finite_real(value, "target norming bound") for value in self.target_norming_bounds)
        if any(value < 0 or value > scale for value in bounds):
            raise ValueError("target norming bounds must be nonnegative and no greater than scale")
        object.__setattr__(self, "target_norming_bounds", bounds)

    @property
    def recovery_scale(self):
        """Return ``1 / (3a - 4a^3)`` with ``a = 1/(2 scale)``.

        Multiplying recovered postselected amplitudes by this value removes
        the known OAA amplitude deficit.
        """
        amplitude = .5 / self.scale
        return 1.0 / (3.0 * amplitude - 4.0 * amplitude**3)


def prepare_qsp_evolution(*, tau, epsilon, expansion=None, max_degree=256,
                          max_evaluations=20000, max_bytes=DEFAULT_INPUT_BYTES,
                          limit_name="max_evaluations"):
    """Solve the QSP phases of both parity parts of a Jacobi-Anger expansion within one evaluation limit.

    Both parity targets are divided by one common scale
    ``s = max(1, cos bound, sin bound) * (1 + margin)``, so the parity
    passes keep the relative weights of the Jacobi-Anger expansion. Margins
    are tried in the order of `QSP_EVOLUTION_TARGET_MARGINS`, 1e-3 then 3e-3,
    and the first one for which both phase solves converge is kept. Only a
    phase-convergence failure moves to the next margin. `max_evaluations`
    covers both parities and every attempted target margin, including
    failed solves and both starts of each solve.
    Sup-norm and Bessel preprocessing are bounded by degree and known
    numerical-array bytes. A supplied expansion is used directly, without
    another Bessel selection. This function and `jacobi_anger_expansion` do
    not import Qiskit.

    Args:
        tau (float): Positive effective evolution time `alpha * t`.
        epsilon (float): Truncation tolerance, strictly between 0 and 1.
        expansion (JacobiAngerExpansion | None): Default `None`, which
            computes `jacobi_anger_expansion(tau, epsilon, min_degree=2)`.
            A supplied expansion must have the same `tau` and `epsilon`.
        max_degree (int): Default `256`. Largest accepted degree.
        max_evaluations (int): Default `20000`. Limit on the phase-solver
            evaluations summed over both parities, both starts of each
            solve and every attempted margin.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on known numerical arrays.
        limit_name (str): Default `"max_evaluations"`. Name of the caller's
            option that sets `max_evaluations`, used in the error messages.

    Returns:
        prepared (QSPPreparedEvolution): Both phase solutions, the scale and
            the margins tried.

    Raises:
        ValueError: If `tau <= 0`, `epsilon` is not in `(0, 1)`, the degree
            exceeds `max_degree`, the evaluations are exhausted, or the
            phases do not converge at any registered margin.
    """
    tau = finite_real(tau, "tau")
    epsilon = finite_real(epsilon, "epsilon")
    if tau <= 0 or not 0 < epsilon < 1:
        raise ValueError("QSP evolution requires tau > 0 and 0 < epsilon < 1")
    max_degree = integer(max_degree, "max_degree", 2)
    max_evaluations = integer(max_evaluations, limit_name, 1)
    # A zero-byte check validates max_bytes itself before any work. Each
    # later step admits its own arrays.
    _check_bytes(0, max_bytes, "QSP preparation")
    # min_degree=2: at small tau the tail bound alone would select degree 1,
    # whose even (cos) part is a degree-0 constant. The symmetric phase
    # solver needs a target of degree >= 1, and no target margin changes
    # that, so the floor keeps small-tau synthesis solvable.
    expansion = expansion or jacobi_anger_expansion(tau, epsilon, min_degree=2,
                                                    max_degree=max_degree, max_bytes=max_bytes)
    if expansion.tau != tau or expansion.epsilon != epsilon:
        raise ValueError("QSP preprocessing expansion differs from its actual time/tolerance")
    if expansion.degree > max_degree:
        raise ValueError(f"selected QSP degree exceeds max_degree={max_degree}")
    solutions: tuple[SymmetricQSPPhases, SymmetricQSPPhases] | None = None
    scale = None
    target_norming_bounds: tuple[float, float] | None = None
    margin_used = None
    last_error: Exception | None = None
    attempted_margins = []
    evaluations_used = 0

    def solve_target(coefficients, scale):
        """Solve phases for ``coefficients / scale`` within the remaining evaluation allowance.

        Evaluations of a failed solve are charged before its error propagates.
        The solver receives exactly the remainder, so its own exhaustion means
        the whole allowance is used. It is reported under ``limit_name`` and the
        full cap.
        """
        nonlocal evaluations_used
        if evaluations_used >= max_evaluations:
            raise ValueError(f"QSP evolution exhausted {limit_name}={max_evaluations}")
        try:
            solved = solve_symmetric_qsp_phases(
                np.asarray(coefficients) / scale, max_degree=max_degree,
                max_evaluations=max_evaluations - evaluations_used, max_bytes=max_bytes,
            )
        except _QSPEvaluationsExhausted as error:
            evaluations_used = max_evaluations
            raise ValueError(f"QSP evolution exhausted {limit_name}={max_evaluations}") from error
        except _QSPPhaseConvergenceError as error:
            evaluations_used += error.evaluations
            raise
        evaluations_used += solved.evaluations
        return solved

    for margin in QSP_EVOLUTION_TARGET_MARGINS:
        attempted_margins.append(margin)
        candidate_scale, cos_target_bound, sin_target_bound = _target_scale(
            expansion,
            margin, max_degree=max_degree, max_bytes=max_bytes,
        )
        try:
            cos_solution = solve_target(expansion.cos_coefficients, candidate_scale)
            sin_solution = solve_target(expansion.sin_coefficients, candidate_scale)
        except _QSPPhaseConvergenceError as error:
            last_error = error
            continue
        solutions = (cos_solution, sin_solution)
        scale = candidate_scale
        target_norming_bounds = (cos_target_bound, sin_target_bound)
        margin_used = margin
        break
    if solutions is None:
        raise ValueError(
            f"QSP phase solving failed for tau={tau:.4g}, eps={epsilon:.2g} at all "
            f"registered target margins {QSP_EVOLUTION_TARGET_MARGINS}"
        ) from last_error
    cos_solution, sin_solution = solutions
    assert scale is not None and target_norming_bounds is not None

    return QSPPreparedEvolution(expansion=expansion, cos_solution=cos_solution, sin_solution=sin_solution,
        scale=scale, target_norming_bounds=target_norming_bounds, margin=margin_used,
        attempted_margins=tuple(attempted_margins))


def qsp_evolution_error_terms(prepared, *, evolution_time, child_error_bound):
    """Return the error terms of a prepared QSP evolution without building a circuit.

    NWQLib's derivation. Before amplification the combined half block is
    ``B = a (V + E)`` with ``a = 1/(2s)`` and ``V = exp(-i t A)``, the
    target at effective time ``tau = alpha t``. The polynomial part obeys
    ``||E|| <= tail_c + tail_s + s (resid_c + resid_s)``, where ``tail_c``
    and ``tail_s`` are the parity Bessel tail bounds and ``resid_c`` and
    ``resid_s`` the phase-solver residual bounds. The child
    encoding represents ``A`` only within ``child_error_bound``.
    For Hermitian A and A', [GSLW] arXiv:1806.01838v1, Lemma 61, gives
    ``||exp(-i t A) - exp(-i t A')|| <= abs(t) ||A - A'||``.
    That physical-time term is added to ``e_B = a ||E||`` without the factor
    ``a``, which is conservative because ``a <= 1/2``. OAA maps ``B`` to
    ``3B - 4 B B^dagger B``. Expanding around ``aV`` gives
    ``||block - (3a - 4a^3) V|| <= 3 e_B (1 + 4a^2) + 12 a e_B^2 + 4 e_B^3``,
    which is at most ``6 e_B + 6 e_B^2 + 4 e_B^3`` for ``a <= 1/2``.

    ``error_bound`` is the raw ``BlockEncoding`` bound against
    ``exp(-i t A)``: the amplitude deficit ``1 - (3a - 4a^3)`` plus that
    residual. Code that divides recovered amplitudes by the known
    factor ``3a - 4a^3`` uses ``compensated_recovery_error_bound``, the
    residual bound times ``recovery_scale``, instead.
    A missing child bound or an unrepresentable Bessel tail makes the
    dependent terms ``None``.

    Args:
        prepared (QSPPreparedEvolution): Output of `prepare_qsp_evolution`.
        evolution_time (float): Physical evolution time `t`.
        child_error_bound (float | None): Operator-norm error bound of the
            child encoding of `A`, or `None` when unknown.

    Returns:
        terms (dict): `amplitude` (`a`), `oaa_amplitude_factor`
            (`3a - 4a^3`), `deficit`, `polynomial_part_error`,
            `child_error`, `oaa_residual_error_bound`, `recovery_scale`,
            `compensated_recovery_error_bound` and `error_bound`.
    """
    expansion = prepared.expansion
    cos_solution, sin_solution = prepared.cos_solution, prepared.sin_solution
    scale = finite_real(prepared.scale, "scale")
    if scale < 1:
        raise ValueError("scale must be at least one")
    unavailable_tail = (
        isinstance(expansion, JacobiAngerExpansion)
        and expansion.analytic_infinite_tail_bound is None
        and expansion.cos_tail_bound is None and expansion.sin_tail_bound is None
        and expansion.tail_bound is None
    )
    for name, value in (
        ("cos_tail_bound", expansion.cos_tail_bound),
        ("sin_tail_bound", expansion.sin_tail_bound),
        ("cos_residual_sup_bound", cos_solution.residual_sup_bound),
        ("sin_residual_sup_bound", sin_solution.residual_sup_bound),
        ("child_error_bound", child_error_bound),
    ):
        # An expansion whose tail bounds underflowed keeps its polynomial but has no finite
        # tail bound, so its None tail bounds are not checked. An unknown child bound is
        # skipped too.
        if (unavailable_tail and name in ("cos_tail_bound", "sin_tail_bound")
                or name == "child_error_bound" and value is None):
            continue
        if finite_real(value, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    evolution_time = finite_real(evolution_time, "evolution_time")
    amplitude = .5 / scale
    oaa_amplitude_factor = 3.0 * amplitude - 4.0 * amplitude**3
    deficit = 1.0 - oaa_amplitude_factor
    polynomial_part_error = None if unavailable_tail else amplitude * (
        expansion.cos_tail_bound
        + expansion.sin_tail_bound
        + scale
        * (cos_solution.residual_sup_bound + sin_solution.residual_sup_bound)
    )
    # GSLW arXiv:1806.01838v1 Lemma 61, p. 53, assumes both A and A_encoded
    # Hermitian. It gives ||exp(-itA)-exp(-itA_encoded)|| <= |t|*||A-A_encoded||.
    # This scalar helper propagates a conditional bound. The public circuit
    # builder records whether construction metadata or the caller supplies
    # that premise, without a circuit-matrix expansion.
    child_error = (0.0 if evolution_time == 0 else None if child_error_bound is None
                   else abs(float(evolution_time)) * float(child_error_bound))
    part_error = None if polynomial_part_error is None or child_error is None else polynomial_part_error + child_error
    oaa_residual_error_bound = None if part_error is None else (
        6.0 * part_error
        + 6.0 * part_error**2
        + 4.0 * part_error**3
    )
    recovery_scale = 1.0 / oaa_amplitude_factor
    compensated_recovery_error_bound = None if oaa_residual_error_bound is None else oaa_residual_error_bound * recovery_scale
    # The raw bound is evaluated from its own formula rather than from
    # oaa_residual_error_bound, so the LCHS recovery terms above cannot
    # change the binary64 value of the BlockEncoding contract.
    error_bound = None if part_error is None else (
        deficit
        + 6.0 * part_error
        + 6.0 * part_error**2
        + 4.0 * part_error**3
    )
    return {
        "amplitude": amplitude,
        "oaa_amplitude_factor": oaa_amplitude_factor,
        "deficit": deficit,
        "polynomial_part_error": polynomial_part_error,
        "child_error": child_error,
        "oaa_residual_error_bound": oaa_residual_error_bound,
        "recovery_scale": recovery_scale,
        "compensated_recovery_error_bound": compensated_recovery_error_bound,
        "error_bound": error_bound,
    }


def build_qsp_evolution_encoding(
    encoding: BlockEncoding,
    *,
    evolution_time: float,
    epsilon: float,
    require_hermitian_evidence: bool = False,
    max_work: int = 1_000_000_000,
    max_bytes: int = DEFAULT_INPUT_BYTES,
    _prepared: QSPPreparedEvolution | None = None,
) -> BlockEncoding:
    """Synthesize a block encoding of ``exp(-i t A)`` from ``BE(A)``.

    Follows the proof of [GSLW] arXiv:1806.01838v1, Theorem 58: Jacobi-Anger
    parity split
    (Lemma 57) at effective time ``tau = alpha * t``, one real-Chebyshev
    pass per parity (each a shared ladder over the ``+-Phi`` pair), a parity
    ancilla combining them with coefficients ``(1/2, -i/2)``, and 3-step
    OAA (Theorem 28, ``n = 3``) raising the amplitude from ``a = 1/(2s)`` to
    ``3a - 4a^3``, close to one because ``s`` is close to one. The error
    terms are derived in ``qsp_evolution_error_terms``.

    GSLW, arXiv:1806.01838v1, Lemma 61 (p. 53), bounds the change in
    evolution by ``abs(t) * ||A - A_encoded||`` when both generators are
    Hermitian. The QSVT polynomial calculus also needs a Hermitian encoded
    block. For an externally constructed encoding, the caller can record
    these separately as ``target_operator_is_hermitian`` and
    ``encoded_operator_is_hermitian`` after establishing them. A dense
    numerical check belongs to the source representation and its resource
    budget, since a circuit matrix requires space exponential in its width.

    Dense Hermitian evidence uses exact entry equality, so even an assembly
    roundoff asymmetry is rejected. If the intended target is Hermitian, the
    caller can explicitly choose ``(A + A.conj().T)/2`` before encoding.
    That changes the supplied target and belongs to the caller's input model.
    The check never performs this projection implicitly.

    Args:
        encoding: Block encoding of a Hermitian operator ``A`` (the QSVT
            eigenvalue calculus assumes Hermitian encoded blocks).
        evolution_time: Evolution time ``t > 0``.
        epsilon: Jacobi-Anger truncation tolerance for ``exp(-i tau x)``.
        require_hermitian_evidence: Require construction metadata declaring
            both ``A`` and the encoded block Hermitian. By default, missing
            evidence is treated as an assumption that the caller has checked,
            recorded in the output with a warning. Explicitly non-Hermitian
            inputs are rejected in either mode. This option reads metadata
            only and never materializes a circuit matrix or applies an
            operator to a state.
        max_work: Default `1_000_000_000`, the default `max_work` of the
            block-encoding builders. Limit on the work of the exact dense
            syntheses that controlling the two passes makes, and of Qiskit's
            control of the synthesized gates, in the work units of
            [Exact dense synthesis](../../development/dense_synthesis.md#admission-of-the-exact-synthesis).
            A child that holds a dense
            ``UnitaryGate``, such as a ``dense_dilation`` encoding, has that
            unitary synthesized once for each pass, and its adjoint once
            for each pass of degree two or more, and the parity control
            unrolls the synthesized gates at every query of the pass. The parity control
            takes the gate-wise route on every route of the joint
            generator, because its controlled object is the whole pass. A gate that an earlier control already synthesized, such
            as a branch of a two-child joint generator, is controlled again
            outside this count, and LCHS planning counts it.
        max_bytes: Default 10 GB (decimal, `10_000_000_000` bytes). Limit on
            the working and kept bytes of those syntheses and control steps.
        _prepared: Default `None`. Phase solution that LCHS planning computes
            with `prepare_qsp_evolution` and supplies, solved for
            ``tau = encoding.alpha * evolution_time`` and the same `epsilon`.
            A caller leaves it at its default.

    Returns:
        encoding (BlockEncoding): BlockEncoding of ``exp(-i t A)`` with
            subnormalization 1, ancillas ``num_ancillas + 3`` (parity, pair,
            signal + block ancillas), the documented ``error_bound``, and full
            synthesis metadata (degrees, recorded tail slack, rescale, solver
            residuals, amplitude deficit, raw OAA residual bound,
            compensated-recovery scale and bound, and query counts).

    Raises:
        ValueError: If `evolution_time` is not positive, `epsilon` is not in
            the open interval (0, 1), the metadata declares the target or
            encoded operator non-Hermitian, `require_hermitian_evidence` is
            True and either declaration is missing, `_prepared` was solved
            for another `tau` or `epsilon`, or the dense syntheses would
            exceed `max_work` or `max_bytes`. With `_prepared` left at None,
            also when `prepare_qsp_evolution` fails for
            ``tau = encoding.alpha * evolution_time``: a degree above 256,
            exhausted evaluations, or no target margin that converges.
        TypeError: If `require_hermitian_evidence` is not a bool.

    Warns:
        UserWarning: When either Hermitian declaration is missing and
            `require_hermitian_evidence` is False.
    """

    from nwqlib.subroutines._semantic import zero_reflection
    from nwqlib.subroutines.block_encoding import BlockEncoding
    from nwqlib.subroutines.qiskit_compat import controlled
    from nwqlib.subroutines.qiskit_compat import inverse_realized_gate
    from qiskit import QuantumCircuit, QuantumRegister

    evolution_time = finite_real(evolution_time, "evolution_time")
    epsilon = finite_real(epsilon, "epsilon")
    if evolution_time <= 0.0:
        raise ValueError("evolution_time must be positive")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in the open interval (0, 1)")
    if not isinstance(require_hermitian_evidence, bool):
        raise TypeError("require_hermitian_evidence must be a bool")
    input_metadata = getattr(encoding, "metadata", {})
    premises = tuple(input_metadata.get(name) for name in (
        "target_operator_is_hermitian", "encoded_operator_is_hermitian"
    ))
    if any(value is False for value in premises):
        raise ValueError("QSP Hamiltonian evolution requires Hermitian target and encoded operators. "
                         "For an intended Hermitian dense target, explicitly symmetrize it before encoding.")
    hermitian_evidence = all(value is True for value in premises)
    if not hermitian_evidence:
        if require_hermitian_evidence:
            raise ValueError("Hermitian construction evidence is unavailable for the target or encoded operator")
        warnings.warn(
            "QSP evolution assumes the caller has established that both the target "
            "and encoded operators are Hermitian. The error bound is conditional "
            "on this premise. No circuit matrix is materialized to check it.",
            UserWarning, stacklevel=2,
        )
    tau = encoding.alpha * float(evolution_time)
    prepared = _prepared or prepare_qsp_evolution(tau=tau, epsilon=epsilon)
    if prepared.expansion.tau != tau or prepared.expansion.epsilon != epsilon:
        raise ValueError("QSP construction differs from its frozen numerical selection")
    expansion = prepared.expansion
    cos_solution, sin_solution = prepared.cos_solution, prepared.sin_solution
    scale, target_norming_bounds, margin_used = prepared.scale, prepared.target_norming_bounds, prepared.margin

    parity = QuantumRegister(1, "qsp_parity")
    pair = QuantumRegister(1, "qsp_pair")
    signal = QuantumRegister(1, "qsp_signal")
    registers = [parity, pair, signal]
    ancillas = None
    if encoding.num_ancillas:
        ancillas = QuantumRegister(encoding.num_ancillas, "be_ancilla")
        registers.append(ancillas)
    system = QuantumRegister(encoding.system_qubits, "system")
    registers.append(system)
    combined = QuantumCircuit(*registers, name="qsp_evolution_half")
    rest = [pair[0], signal[0], *([*ancillas] if ancillas else []), *system]

    block_gates = _block_gate_pair(encoding, native_ucg=False)
    cos_gate = _build_real_chebyshev_encoding(
        encoding,
        cos_solution.phases,
        block_gates=block_gates,
    ).to_gate(
        label="qsp_cos_pass"
    )
    sin_pass = _build_real_chebyshev_encoding(
        encoding,
        sin_solution.phases,
        block_gates=block_gates,
    )
    sin_pass.global_phase -= np.pi / 2.0  # LCU coefficient -i absorbed (lcu/core convention)
    sin_gate = sin_pass.to_gate(label="qsp_sin_pass")
    # Each controlled call below synthesizes the dense unitaries of its pass
    # again, so both passes are counted before either starts.
    from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses
    from nwqlib.subroutines.qiskit_compat import dense_control_counts, dense_synthesis_widths
    admit_dense_syntheses(dense_synthesis_widths(cos_gate) + dense_synthesis_widths(sin_gate),
                          controls=(dense_control_counts(cos_gate, 1), dense_control_counts(sin_gate, 1)),
                          max_work=max_work, max_bytes=max_bytes, operation="QSP evolution dense synthesis")
    combined.h(parity[0])
    combined.append(controlled(cos_gate, 1, ctrl_state=0), [parity[0], *rest])
    combined.append(controlled(sin_gate, 1, ctrl_state=1), [parity[0], *rest])
    combined.h(parity[0])

    num_ancillas_total = encoding.num_ancillas + 3
    circuit = QuantumCircuit(*registers, name="qsp_evolution")
    # A R A^dagger R A realizes -(3B - 4 B B^dagger B). The global phase pi below
    # compensates the minus.
    circuit.global_phase += np.pi
    all_qubits = [parity[0], pair[0], signal[0], *([*ancillas] if ancillas else []), *system]
    ancilla_qubits = all_qubits[: num_ancillas_total]
    combined_gate = combined.to_gate(label="qsp_evolution_half")
    reflection_gate = zero_reflection(num_ancillas_total).to_gate(label="refl_0")
    circuit.append(combined_gate, all_qubits)
    circuit.append(reflection_gate, ancilla_qubits)
    circuit.append(inverse_realized_gate(combined_gate, native_ucg=False), all_qubits)
    circuit.append(reflection_gate, ancilla_qubits)
    circuit.append(combined_gate, all_qubits)

    # Error bound: see qsp_evolution_error_terms for the derivation. The raw
    # BlockEncoding contract includes the amplitude deficit 1 - (3a - 4a^3).
    # LCHS may divide the recovered postselected amplitudes by the known
    # factor 3a - 4a^3 and then uses the residual-only bound in that frame.
    bounds = qsp_evolution_error_terms(prepared, evolution_time=evolution_time, child_error_bound=encoding.error_bound)
    amplitude = bounds["amplitude"]
    deficit = bounds["deficit"]
    polynomial_part_error = bounds["polynomial_part_error"]
    child_error = bounds["child_error"]
    oaa_residual_error_bound = bounds["oaa_residual_error_bound"]
    recovery_scale = bounds["recovery_scale"]
    compensated_recovery_error_bound = bounds["compensated_recovery_error_bound"]
    error_bound = bounds["error_bound"]
    queries = QSP_OAA_QUERY_MULTIPLIER * (expansion.cos_degree + expansion.sin_degree)

    metadata: dict[str, Any] = {
        "hermitian_premise": "construction_metadata" if hermitian_evidence else "caller_assumption",
        "evolution_time": float(evolution_time),
        "effective_time_tau": tau,
        "jacobi_anger": expansion.to_dict(),
        "target_rescale": scale,
        "target_rescale_source": "chebyshev_grid_norming_bound",
        "target_cos_norming_bound_before_rescale": target_norming_bounds[0],
        "target_sin_norming_bound_before_rescale": target_norming_bounds[1],
        "target_norming_bound": max(target_norming_bounds) / scale,
        "target_margin": margin_used,
        "oaa_amplitude": amplitude,
        "oaa_amplitude_deficit": deficit,
        "polynomial_part_error_bound": polynomial_part_error,
        "child_operator_error_contribution": child_error,
        "oaa_residual_error_bound": oaa_residual_error_bound,
        "recovery_scale": recovery_scale,
        "compensated_recovery_error_bound": compensated_recovery_error_bound,
        "oaa_query_multiplier": QSP_OAA_QUERY_MULTIPLIER,
        "block_encoding_queries": queries,
        "query_count_reference_form": (
            "3 * d(tau, eps), d = Theta(tau + log(1/eps)/log(e + log(1/eps)/tau)) "
            "[GSLW arXiv:1806.01838v1 Cor. 60]; this circuit uses 3 * (d_cos + d_sin) queries"
        ),
        # Full phase vectors (degree-many floats) are omitted. The residuals
        # and diagnostics carry the scientifically checkable content.
        "phase_solver": {
            "cos": {k: v for k, v in cos_solution.to_dict().items() if k != "phases"},
            "sin": {k: v for k, v in sin_solution.to_dict().items() if k != "phases"},
        },
        "classical_preprocessing_scope": "qsp_phase_factor_optimization",
        "child_block_encoding": {
            "implementation": encoding.implementation,
            "alpha": encoding.alpha,
            "error_bound": encoding.error_bound,
            "error_bound_provenance": encoding.metadata.get(
                "error_bound_provenance"
            ),
            "target_operator_is_hermitian": input_metadata.get("target_operator_is_hermitian"),
            "encoded_operator_is_hermitian": input_metadata.get("encoded_operator_is_hermitian"),
            "physical_time_error_bound": child_error,
            "num_ancillas": encoding.num_ancillas,
        },
    }
    return BlockEncoding(
        circuit=circuit,
        alpha=1.0,
        num_ancillas=num_ancillas_total,
        system_qubits=encoding.system_qubits,
        error_bound=error_bound,
        implementation="qsp_jacobi_anger_evolution",
        metadata=metadata,
    )


__all__ = [
    "JacobiAngerExpansion",
    "QSP_OAA_QUERY_MULTIPLIER",
    "build_control_diagonal_generator_encoding",
    "build_qsp_evolution_encoding",
    "build_qsvt_circuit",
    "build_real_chebyshev_encoding",
    "compiled_select_resource_law",
    "jacobi_anger_expansion",
]
