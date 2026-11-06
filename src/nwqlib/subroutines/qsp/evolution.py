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
  block-encoding ancillas. The phase operator is implemented by
  conjugation with X. A signal qubit is flipped onto the ``Pi`` subspace
  (``X`` controlled on the ancillas being all zero), rotated by
  ``RZGate(+2 phi)`` (``= e^{-i phi Z}``, the Qiskit sign
  ``RZGate(theta) = exp(-i theta Z / 2)``), and flipped back. Wx phases, in
  the convention dictionary of [MRTC] arXiv:2105.02859v5, App. A, are
  converted with ``wx_phases_to_reflection`` and the ``i^d`` factor becomes
  circuit global phase, so the encoded block is exactly the Wx polynomial
  ``P(A / alpha)``.
- ``build_real_chebyshev_encoding``: block-encodes ``Re P(A / alpha)`` by
  averaging the ``+Phi`` and ``-Phi`` passes over one pair ancilla
  ([GSLW] arXiv:1806.01838v1, Corollary 18, Eq. (33)). Both branches
  share the same ladder of alternating ``U_BE`` and ``U_BE^dagger``
  applications, because a CNOT from the pair qubit onto the signal qubit
  conjugates every ``RZ`` and so negates the reflection phases on the
  pair-one branch (``X RZ(theta) X = RZ(-theta)``). A ``Z`` on the pair qubit for odd
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
- ``build_control_diagonal_generator_encoding``: block encoding of the
  effective Hamiltonian of the linear combination of Hamiltonian simulation
  (LCHS) method, adapted from Pocrnic, Johnson, Katabarwa, and
  Wiebe, arXiv:2506.20760v2, Lemma 7 (p. 15).
- ``build_shared_qsvt_evolution``: the shared real-QSVT evolution, with one
  flag for the ``+Phi/-Phi`` average, one parity qubit, one alternating
  ladder whose first n queries both parities share and one final controlled
  query ([GSLW] arXiv:1806.01838v1, Corollary 18, Lemma 19, Theorems
  56/58), raw or with three-use OAA.
- ``build_qubitized_evolution``: the canonical qubitized GQSP circuit of a
  general complex Laurent or one-sided target on the walk ``W = (2 Pi - I) U``
  of a full involution or ``W = (2 Pi_0 - I) V_0`` of the generic lift (Low and
  Chuang, arXiv:1610.06546v3; Motlagh and Wiebe, arXiv:2308.01501v2,
  Theorem 6 and Corollary 8; Suenderhauf, arXiv:2312.00723v1, Appendix C.2),
  for the Jacobi-Anger Laurent and GQET targets and the imaginary-time
  filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
import warnings
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import scipy.special
from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from nwqlib.subroutines.qsp.phases import (
    _QSPPhaseConvergenceError,
    WxQSPPhases,
    chebyshev_polynomial_sup_bound,
    _controls,
    _prepare_wx_problem,
    _solve_wx_problem,
    QSP_SOLVER_RESIDUAL_TOLERANCE,
    wx_phases_to_reflection,
)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit, QuantumRegister
    from nwqlib.subroutines.block_encoding import BlockEncoding

# The common cap is selected before either parity solve, and the FFT
# doubling of each target stops at its accepted residual.
# Numerical qualification and the finite cap comparison are recorded in
# docs/ENGINEERING_CONSTANTS.md.
QSP_EVOLUTION_TARGET_CAP = 0.999
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
    """Propagate the source of the abstract error bound and the Hermitian assumption."""

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
    max_work: int | None = None,
    max_bytes: int | None = None,
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
    """Cosine and sine Chebyshev series for ``exp(-i tau x)``.

    The finite Bessel suffixes and analytic remainders bound truncation in
    exact arithmetic. Their binary64 evaluations are numerical estimates,
    without outward rounding of Bessel or libm operations.

    Attributes:
        tau: Nonnegative effective time ``alpha * t``.
        epsilon: Requested combined truncation tolerance in ``(0, 1)``.
        degree: Smallest degree at least ``min_degree`` that meets the
            computed bound at ``analysis_terminal``; zero when ``tau=0``.
            This is not a global minimum over all possible tail bounds.
        cos_degree: Even polynomial degree, or zero for constant cosine.
        sin_degree: Odd polynomial degree, or zero for the zero polynomial.
        cos_coefficients: Cosine Chebyshev coefficients in ascending order.
        sin_coefficients: Sine Chebyshev coefficients in ascending order.
        tail_bound: Combined finite-plus-analytic tail, or None when positive
            but below binary64 range. Exact zero occurs only at ``tau=0``.
        tail_slack: ``epsilon / tail_bound`` when finite and representable.
        cos_tail_bound: Even finite suffix plus its analytic remainder.
        sin_tail_bound: Odd finite suffix plus its analytic remainder.
        analysis_terminal: Last Bessel order used in the finite suffixes.
        analytic_infinite_tail_bound: Sum of the even and odd analytic
            remainders, with each omitted order counted once.
        tail_log_bound: Natural logarithm of the positive combined bound,
            retained when its ordinary value underflows; None at exact zero.
        cos_tail_log_bound: Natural logarithm of the positive even bound.
        sin_tail_log_bound: Natural logarithm of the positive odd bound.
        analytic_infinite_tail_log_bound: Natural logarithm of the positive
            combined analytic remainder. Log bounds are numerical estimates
            under the same arithmetic qualification as ordinary bounds.
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
    tail_log_bound: float | None
    cos_tail_log_bound: float | None
    sin_tail_log_bound: float | None
    analytic_infinite_tail_log_bound: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return the scalar truncation quantities, including log bounds."""
        from dataclasses import fields
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in ("cos_coefficients", "sin_coefficients")
        }


def _jacobi_anger_log_remainder(tau: float, first: int, stride: int) -> float:
    """Bound ``log(2 sum_{j>=0}|J_(first+j*stride)(tau)|)``.

    For r=first>tau>0, set theta=acosh(r/tau),
    Delta=sqrt((r-tau)(r+tau)), and psi=r*theta-Delta.
    Kapteyn and Watson give |J_r| <= exp(-psi) and
    |J_r| <= exp(-psi)/sqrt(2*pi*Delta), respectively; see
    Pocrnic--Motlagh, arXiv:2608.02734v1, Eqs. (8), (37)--(39).
    Since psi'(r)=acosh(r/tau) increases, psi(r+j*h)>=psi(r)+j*h*theta.
    The Watson prefactor decreases with r. Summing either geometric
    majorant therefore gives 2*exp(-psi)/(1-exp(-h*theta)), with the
    Watson prefactor when it improves the bound. Here h is 1 or 2.

    For theta<=1, the positive Taylor series
    psi=tau*sum_{k>=1} 2k*theta**(2k+1)/(2k+1)!
    avoids cancellation. Successive terms have ratio
    theta**2/(2k*(2k+3)). We keep ten terms; the omitted remainder is
    between zero and term_10*q/(1-q), q=theta**2/(20*23).
    Using the partial sum lowers psi and thus enlarges the bound. This
    does not provide an outward-rounded binary64 certificate.

    The independent factorial bound follows from the Poisson integral
    (DLMF 10.9.4): |J_r(tau)| <= (tau/2)**r/r!. The integral of sin**(2r)
    cancels its gamma prefactor. Successive h-step ratios decrease with
    order, so the first ratio q<1 gives a geometric remainder. Compare
    both bounds in log space; h=2 keeps the parity remainders disjoint.
    """
    log_first = first * (math.log(tau) - math.log(2.0)) - math.lgamma(first + 1)
    ratio = tau / (2.0 * (first + 1))
    if stride == 2:
        ratio *= tau / (2.0 * (first + 2))
    factorial = (math.log(2.0) + log_first - math.log1p(-ratio)
                 if ratio < 1.0 else math.inf)
    if first <= tau:
        return factorial
    relative_gap = (first - tau) / tau
    # log1p preserves r/tau-1 near the turning point. The log-ratio
    # expression also handles positive subnormal tau without overflowing r/tau.
    if relative_gap <= 1.0:
        theta = math.log1p(relative_gap + math.sqrt(relative_gap * (2.0 + relative_gap)))
    else:
        theta = math.log(first) - math.log(tau) + math.log1p(math.sqrt(1.0 - (tau / first)**2))
    delta = math.sqrt(first - tau) * math.sqrt(first + tau)
    if theta <= 1.0:
        term = theta**3 / 3.0
        terms = [term]
        for k in range(1, 10):
            term *= theta**2 / (2 * k * (2 * k + 3))
            terms.append(term)
        psi = tau * math.fsum(terms)
    else:
        psi = first * theta - delta
    kapteyn = math.log(2.0) - psi - math.log(-math.expm1(-stride * theta))
    watson = kapteyn - 0.5 * (math.log(2.0 * math.pi) + math.log(delta))
    return min(factorial, kapteyn, watson)


def _bound_from_log(log_bound: float) -> float | None:
    """Keep a positive bound's underflow distinct from exact zero."""
    value = math.exp(log_bound)
    return value if value > 0.0 else None


def jacobi_anger_expansion(
    tau: float,
    epsilon: float,
    *,
    min_degree: int = 1,
    max_bytes: int | None = None,
) -> JacobiAngerExpansion:
    """Truncate ``exp(-i tau x)`` on ``[-1, 1]``.

    The Jacobi--Anger identity gives
    C=J_0+2*sum_{k>=1}(-1)**k*J_(2k)*T_(2k) and
    S=2*sum_{k>=0}(-1)**k*J_(2k+1)*T_(2k+1), with target C-i*S.
    Since |T_k|<=1, the combined truncation error is at most
    2*sum_{k>d}|J_k(tau)| (GSLW, arXiv:1806.01838v1, Lemma 57).

    Kapteyn/Watson and factorial geometric bounds, derived in
    ``_jacobi_anger_log_remainder``, first select a terminal by exponential
    bracketing and binary search; both remainders decrease to zero with the
    order, so the search ends. Even and
    odd remainders use stride two, so their sum counts each order once.
    A single Bessel table through that terminal then tightens the degree
    using finite suffixes plus those analytic remainders. All tolerance
    decisions use logarithms. Positive underflow retains a log bound and
    an ordinary bound of None. At zero time the exact polynomial is one.

    Selection uses O(log(analysis_terminal+1)) scalar bound evaluations and
    O(analysis_terminal+1) Bessel values, sums and coefficient storage.
    The known numerical arrays require at most 96*(analysis_terminal+2)
    bytes, checked before allocation; Python objects, special-function
    workspace and process memory are outside that count. No phase solve
    is performed, and a scalar degree does not qualify a phase solver.
    Binary64 Bessel values, logarithms and sums are numerical estimates;
    these inequalities alone do not certify their arithmetic errors.

    Args:
        tau (float): Finite nonnegative effective time ``alpha * t``.
        epsilon (float): Combined truncation tolerance, strictly in (0, 1).
        min_degree (int): Default 1. Minimum nonzero-time truncation degree.
        max_bytes (int | None): Default None, the physical memory. Limit on
            the Bessel and coefficient arrays.

    Returns:
        expansion (JacobiAngerExpansion): Parity coefficients and finite
            tail estimates, including logarithmic evidence.

    Raises:
        ValueError: For invalid parameters or an exceeded memory limit.
    """
    tau = finite_real(tau, "tau")
    epsilon = finite_real(epsilon, "epsilon")
    min_degree = integer(min_degree, "min_degree", 1)
    if tau < 0.0:
        raise ValueError("tau must be nonnegative")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in the open interval (0, 1)")
    if tau == 0.0:
        return JacobiAngerExpansion(
            tau=tau, epsilon=epsilon, degree=0, cos_degree=0, sin_degree=0,
            cos_coefficients=(1.0,), sin_coefficients=(0.0,),
            tail_bound=0.0, tail_slack=None, cos_tail_bound=0.0, sin_tail_bound=0.0,
            analysis_terminal=0, analytic_infinite_tail_bound=0.0,
            tail_log_bound=None, cos_tail_log_bound=None, sin_tail_log_bound=None,
            analytic_infinite_tail_log_bound=None,
        )
    log_epsilon = math.log(epsilon)

    def parity_remainders(terminal):
        even_first = terminal + 1 + (terminal % 2 == 0)
        odd_first = terminal + 1 + (terminal % 2 == 1)
        return (_jacobi_anger_log_remainder(tau, even_first, 2),
                _jacobi_anger_log_remainder(tau, odd_first, 2))

    def log_remainder(terminal):
        return float(np.logaddexp(*parity_remainders(terminal)))

    # Both analytic remainders decrease to minus infinity with the terminal.
    lower = min_degree - 1
    upper = max(min_degree, math.floor(tau))
    step = 1
    while log_remainder(upper) > log_epsilon:
        lower = upper
        upper += step
        step *= 2
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if log_remainder(middle) <= log_epsilon:
            upper = middle
        else:
            lower = middle
    terminal = upper
    _check_bytes(96 * (terminal + 2), max_bytes, "Jacobi-Anger selection")
    log_even, log_odd = parity_remainders(terminal)
    log_analytic = float(np.logaddexp(log_even, log_odd))
    bessel = scipy.special.jv(np.arange(terminal + 1), tau)
    magnitudes = np.abs(bessel)
    suffix = np.zeros((2, terminal + 2))
    suffix[0, :terminal + 1:2] = 2.0 * magnitudes[::2]
    suffix[1, 1:terminal + 1:2] = 2.0 * magnitudes[1::2]
    suffix = np.cumsum(suffix[:, ::-1], axis=1)[:, ::-1]
    with np.errstate(divide="ignore"):
        parity_logs = np.logaddexp(np.log(suffix), np.array([log_even, log_odd])[:, None])
    complete_logs = np.logaddexp(parity_logs[0], parity_logs[1])
    degree = min_degree + int(np.flatnonzero(complete_logs[min_degree + 1:] <= log_epsilon)[0])
    cos_degree = degree - degree % 2
    sin_degree = degree - (degree % 2 == 0)
    log_even, log_odd = (float(value) for value in parity_logs[:, degree + 1])
    log_tail = float(complete_logs[degree + 1])
    cos_coefficients = np.zeros(cos_degree + 1)
    cos_coefficients[0] = bessel[0]
    cos_indices = np.arange(2, cos_degree + 1, 2)
    cos_coefficients[cos_indices] = 2.0 * (-1.0)**(cos_indices // 2) * bessel[cos_indices]
    sin_coefficients = np.zeros(sin_degree + 1)
    sin_indices = np.arange(1, sin_degree + 1, 2)
    sin_coefficients[sin_indices] = 2.0 * (-1.0)**((sin_indices - 1) // 2) * bessel[sin_indices]
    log_slack = log_epsilon - log_tail
    return JacobiAngerExpansion(
        tau=tau, epsilon=epsilon, degree=degree,
        cos_degree=cos_degree, sin_degree=sin_degree,
        cos_coefficients=tuple(float(v) for v in cos_coefficients),
        sin_coefficients=tuple(float(v) for v in sin_coefficients),
        tail_bound=_bound_from_log(log_tail),
        tail_slack=math.exp(log_slack) if log_slack <= math.log(np.finfo(float).max) else None,
        cos_tail_bound=_bound_from_log(log_even), sin_tail_bound=_bound_from_log(log_odd),
        analysis_terminal=terminal,
        analytic_infinite_tail_bound=_bound_from_log(log_analytic),
        tail_log_bound=log_tail, cos_tail_log_bound=log_even, sin_tail_log_bound=log_odd,
        analytic_infinite_tail_log_bound=log_analytic,
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
    """Return the gate of the encoding and its inverse for one construction.

    Controlled parity passes use ``native_ucg=False`` so that the
    definitions of the uniformly controlled gates (UCG) of the encoding are
    expanded before the passes copy them. Uncontrolled QSVT and
    real-polynomial circuits can keep Qiskit's UCG tables.
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

    reflection_phases, global_phase = wx_phases_to_reflection(wx_phases)
    degree = len(reflection_phases) - 1
    block_gate, inverse_block_gate = _block_gate_pair(encoding) if degree else (None, None)
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
    """Assemble the circuit of ``build_real_chebyshev_encoding`` from a built pair of encoding gates.

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
    """Block-encode ``Re P(A / alpha)`` with the ``+Phi`` and ``-Phi`` passes on one shared ladder.

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

    phases = np.asarray(wx_phases, float).reshape(-1)
    return _build_real_chebyshev_encoding(
        encoding,
        phases,
        block_gates=_block_gate_pair(encoding) if phases.size > 1 else (None, None),
    )


def _as_control_diagonal(values: Any, *, name: str) -> np.ndarray:
    """Check that the coefficients of a Hermitian generator are real, without projecting the input.

    A nonzero imaginary component changes the generator. Removing it would
    need a perturbation budget and would not establish the Hermitian target
    assumption required by GSLW arXiv:1806.01838v1, Lemma 61. This checks
    the supplied table of scalars.
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
    are active). ``build_control_diagonal_generator_encoding`` builds its
    circuit from these values.
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
    max_work: int | None = None,
    max_bytes: int | None = None,
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
    diagonal entries make their control state evolve trivially under the
    subsequent Hamiltonian simulation (identity padding).

    Args:
        encoding_l: Block encoding of the Hermitian part ``L``.
        encoding_h: Block encoding of ``H``, or ``None`` when ``H = 0``.
        l_diagonal: Real diagonal for the ``L`` branch, power-of-two length
            ``2**control_qubits`` indexed by the control-register basis.
        h_diagonal: Real diagonal for the ``H`` branch (same length),
            required exactly when ``encoding_h`` is supplied.
        max_work: Default None, no limit. Limit on the work of the exact
            dense syntheses that controlling the two branches makes, and of
            Qiskit's control of the synthesized gates, in the work units of
            [Exact dense synthesis](../../development/dense_synthesis.md#admission-of-the-exact-synthesis).
            A ``dense_dilation`` child is synthesized and controlled once. A
            single active branch is not controlled here.
        max_bytes: Default `None`, the physical memory of this computer. Limit on
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
    """Count the queries, child calls and rotation angles of a QSP evolution of a joint generator before gate synthesis.

    The counted construction is
    [`build_qsp_evolution_encoding`][nwqlib.subroutines.qsp.evolution.build_qsp_evolution_encoding]
    applied to the joint generator that
    [`build_control_diagonal_generator_encoding`][nwqlib.subroutines.qsp.evolution.build_control_diagonal_generator_encoding]
    builds. It makes ``3 * (cos_degree + sin_degree)`` queries to the joint
    block encoding (the 3-step OAA over both parity ladders), and each query
    costs one call per child encoding plus one multiplexed RY of
    ``2**control_qubits`` rotation angles per child. Counts are structural
    (rotation angles and child calls before gate synthesis), so they do not
    depend on the Qiskit version. They are not CX or native-depth counts,
    a minimum cost or a success-probability factor. The generator's
    `dense_control_route` changes how its children are synthesized, not
    these counts. The evolutions of LCHS's `qsp_block_encoding` backend are
    other constructions with their own query counts.

    In Pocrnic et al., arXiv:2506.20760v2, Lemma 7, p. 15, each
    effective-Hamiltonian query uses one ``U_A`` plus one ``U_A^dagger`` call
    and ``2M`` multi-controlled rotations, where ``M`` is the number of
    quadrature nodes. Here each query uses ``child_count`` child calls and
    ``child_count * P`` multiplexed rotation angles. ``child_count`` is the
    number of branches with a nonzero diagonal, 2 for the ``L`` and ``H``
    branches and 1 when only one branch has one, as for ``H=0``, and
    ``P = 2**control_qubits`` is the number of control-register addresses after
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


@dataclass(frozen=True, kw_only=True)
class QSPPreparedEvolution:
    """The two real-Wx phase solutions and their common scale.

    ``scale=max(1, cos_bound, sin_bound)/target_cap`` preserves the relative
    cosine and sine weights. Each phase result includes its actual FFT
    attempts, numerical residual and unknown arithmetic enclosure.
    """

    expansion: JacobiAngerExpansion
    cos_solution: WxQSPPhases
    sin_solution: WxQSPPhases
    scale: float
    target_norming_bounds: tuple[float, float]
    target_cap: float

    @property
    def recovery_scale(self):
        """Return 1/(3a-4a³), a=1/(2 scale), for the OAA success amplitude."""
        amplitude = .5 / self.scale
        return 1.0 / (3.0 * amplitude - 4.0 * amplitude**3)


def _qsp_phase_targets(expansion, target_cap, max_bytes=None):
    """Return the common scale, both norming bounds and both normalized Wx problems, without a phase solve.

    ``scale = max(1, B_cos, B_sin)/target_cap`` from the Chebyshev-zero
    norming estimates B of the raw cosine and sine series. Each problem
    keeps its exact-zero trimmed degree and its own normalized norm bound.
    """
    cap = finite_real(target_cap, "target_cap")
    if not 0 < cap < 1:
        raise ValueError("target_cap must lie strictly between zero and one")
    arrays = (expansion.cos_coefficients, expansion.sin_coefficients)
    bounds = tuple(chebyshev_polynomial_sup_bound(c, max_bytes=max_bytes) for c in arrays)
    scale = max(1.0, *bounds) / cap
    problems = []
    for coefficients in arrays:
        # Division by scale rounds each coefficient by at most half an ulp. A
        # subnormal result can round to zero, and _prepare_wx_problem then
        # trims it as an exact zero, as the imaginary-time path does.
        problems.append(_prepare_wx_problem(np.asarray(coefficients, float) / scale, max_bytes))
    return scale, bounds, tuple(problems)


def prepare_qsp_evolution(*, tau, epsilon, expansion=None, max_bytes=None,
                          target_cap=QSP_EVOLUTION_TARGET_CAP, max_fft=None,
                          residual_tolerance=QSP_SOLVER_RESIDUAL_TOLERANCE):
    """Prepare the two real parity targets of exp(-i tau x) and their Wx phases.

    ``tau`` is positive alpha*t, and ``epsilon`` in (0,1) controls the
    Jacobi-Anger truncation. ``expansion`` optionally supplies that already
    selected expansion. ``max_bytes`` (default None, the physical memory)
    limits the dominant arrays of each step.

    ``target_cap`` defaults to 0.999 and fixes the common normalization before
    synthesis. Each target's FFT length doubles until its actual residual is
    at most ``residual_tolerance`` (`solve_wx_qsp_phases`), in the normalized
    target frame and for each parity; ``max_fft`` optionally caps the doubling.
    A caller without a total allowance keeps the solver's default 1e-12. A
    caller with one derives the tolerance from it (``qsp_evolution_error_terms``
    gives the relation; ``coherent_selection._shared_row`` is the LCHS owner).
    Failure changes neither target, scale nor polynomial degree.

    Returns numerical phase evidence and the shared OAA recovery scale.
    Arithmetic enclosures remain unknown. Invalid premises, a memory
    refusal or a failed actual-angle residual check raises ValueError.
    """
    tau = finite_real(tau, "tau")
    epsilon = finite_real(epsilon, "epsilon")
    if tau <= 0 or not 0 < epsilon < 1:
        raise ValueError("QSP evolution requires tau > 0 and 0 < epsilon < 1")
    residual_tolerance, max_fft = _controls(residual_tolerance, max_fft)
    expansion = expansion or jacobi_anger_expansion(tau, epsilon, min_degree=1, max_bytes=max_bytes)
    if expansion.tau != tau or expansion.epsilon != epsilon:
        raise ValueError("QSP preprocessing expansion differs from its time or tolerance")
    return _solve_qsp_targets(expansion, _qsp_phase_targets(expansion, target_cap, max_bytes), target_cap,
                              residual_tolerance=residual_tolerance, max_fft=max_fft, max_bytes=max_bytes)


def _solve_qsp_targets(expansion, targets, target_cap, *, residual_tolerance, max_fft=None, max_bytes=None):
    """Solve the Wx phases of the ``(scale, bounds, problems)`` that ``_qsp_phase_targets`` planned for ``expansion``.

    A route selector that already formed the targets to choose the route
    passes them here, so the selected route is synthesized from the same
    targets without forming them again. ``residual_tolerance`` is the
    normalized acceptance threshold of each parity.
    """
    scale, bounds, problems = targets
    solutions = []
    for parity, problem in zip(("cosine", "sine"), problems):
        try:
            solutions.append(_solve_wx_problem(problem, residual_tolerance,
                                               max_fft=max_fft, max_bytes=max_bytes))
        except _QSPPhaseConvergenceError as error:
            raise type(error)(f"QSP {parity} synthesis failed: {error}", evidence=error.evidence) from error
    cos_solution, sin_solution = solutions
    return QSPPreparedEvolution(
        expansion=expansion, cos_solution=cos_solution, sin_solution=sin_solution,
        scale=scale, target_norming_bounds=bounds, target_cap=float(target_cap))


def qsp_evolution_error_terms(prepared, *, evolution_time, child_error_bound):
    """Return raw and recovered error bounds without constructing a circuit.

    The parity combination has half block B=a(F+E), a=1/(2s)<=1/2,
    where F is the unitary evolution of the Hermitian encoded operator.
    Its unscaled polynomial error obeys
    ``||E||<=delta=tail_c+tail_s+s*(resid_c+resid_s)``. The returned
    ``polynomial_part_error`` is the raw half-block bound ``a*delta``.

    Three-use OAA gives T(B)=3B-4BB^dagger B (GSLW,
    arXiv:1806.01838v1, Theorem 28). Set r=3a-4a^3. Direct expansion with
    F^dagger F=I gives

    ```text
    T(B)-rF = a(3-8a^2)E - 4a^3 F E^dagger F
    - 4a^3*(E E^dagger F + E F^dagger E + F E^dagger E + E E^dagger E).
    ```

    For 0<a<=1/2, the linear norm coefficients sum to r. The three
    quadratic terms and one cubic term thus imply
    ``||T(B)/r-F||<=Phi_a(delta)``, with

    ```text
    Phi_a(delta)=delta+12a^2/(3-4a^2)*delta^2+4a^2/(3-4a^2)*delta^3.
    ```

    For Hermitian target A and encoded A', Duhamel's identity integrates
    exp(-i(t-v)A)(A-A')exp(-ivA') over v. Both exponentials have norm one,
    so their evolution difference is at most ``abs(t)*child_error_bound``
    (GSLW, Lemma 61). Add this once after Phi_a, in the recovered frame.
    The raw bound against exp(-itA) is ``(1-r)+r*(Phi_a(delta)+child_error)``.
    Native full-circuit error, if assessed separately, needs its own
    multiplication by 1/r and is not included in these bounds.

    The bounds inherit the Bessel and phase-solver arithmetic qualification.
    A missing child bound or underflowed combined tail leaves dependent
    ordinary bounds unknown; logarithmic tail evidence remains available
    in ``prepared.expansion``.

    Args:
        prepared (QSPPreparedEvolution): Output of ``prepare_qsp_evolution``.
        evolution_time (float): Physical evolution time t.
        child_error_bound (float | None): Operator error of the child
            encoding in physical generator units, or None if unknown.

    Returns:
        terms (dict): Half-block ``amplitude`` a, ``oaa_amplitude_factor`` r,
            ``deficit`` 1-r, ``polynomial_error`` delta (the recovered error
            of the raw route, zeta=2s), ``algorithmic_error``
            Phi_a(delta) (the recovered OAA error before the child term),
            ``polynomial_part_error`` a*delta,
            ``child_error`` abs(t)*child_error_bound,
            ``oaa_residual_error_bound`` in the raw frame,
            ``recovery_scale`` 1/r, ``compensated_recovery_error_bound``
            in the recovered frame, and raw ``error_bound``.
    """
    expansion = prepared.expansion
    cos_solution, sin_solution = prepared.cos_solution, prepared.sin_solution
    scale = finite_real(prepared.scale, "scale")
    if scale < 1:
        raise ValueError("scale must be at least one")
    for name, value in (
        ("tail_bound", expansion.tail_bound),
        ("cos_residual_sup_bound", cos_solution.residual_sup_bound),
        ("sin_residual_sup_bound", sin_solution.residual_sup_bound),
        ("child_error_bound", child_error_bound),
    ):
        if value is None and name in ("tail_bound", "child_error_bound"):
            continue
        if finite_real(value, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    evolution_time = finite_real(evolution_time, "evolution_time")
    amplitude = .5 / scale
    oaa_amplitude_factor = 3.0 * amplitude - 4.0 * amplitude**3
    deficit = 1.0 - oaa_amplitude_factor
    delta = None if expansion.tail_bound is None else (
        expansion.tail_bound
        + scale * (cos_solution.residual_sup_bound + sin_solution.residual_sup_bound)
    )
    polynomial_part_error = None if delta is None else amplitude * delta
    # Both target and encoded generators must be Hermitian. The circuit
    # builder records construction evidence or the caller's assumption.
    child_error = (0.0 if evolution_time == 0 else None if child_error_bound is None
                   else abs(float(evolution_time)) * float(child_error_bound))
    amplitude_squared = amplitude * amplitude
    cubic_coefficient = 4.0 * amplitude_squared / (3.0 - 4.0 * amplitude_squared)
    algorithmic_error = None if delta is None else delta + cubic_coefficient * delta**2 * (3.0 + delta)
    recovery_scale = 1.0 / oaa_amplitude_factor
    compensated_recovery_error_bound = (None if algorithmic_error is None or child_error is None
                                        else algorithmic_error + child_error)
    oaa_residual_error_bound = (None if compensated_recovery_error_bound is None
                               else oaa_amplitude_factor * compensated_recovery_error_bound)
    error_bound = None if oaa_residual_error_bound is None else deficit + oaa_residual_error_bound
    return {
        "amplitude": amplitude,
        "oaa_amplitude_factor": oaa_amplitude_factor,
        "deficit": deficit,
        "polynomial_error": delta,
        "algorithmic_error": algorithmic_error,
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
    max_work: int | None = None,
    max_bytes: int | None = None,
    residual_tolerance: float = QSP_SOLVER_RESIDUAL_TOLERANCE,
    _prepared: QSPPreparedEvolution | None = None,
) -> BlockEncoding:
    """Synthesize a block encoding of ``exp(-i t A)`` from ``BE(A)``.

    This construction controls each whole parity pass and so uses
    ``3 * (d_cos + d_sin)`` queries. The shared real QSVT routes of LCHS's
    `qsp_block_encoding` backend use `build_shared_qsvt_evolution` instead,
    which shares one ladder and uses n uncontrolled queries and one
    controlled query per use.

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
    numerical check is left to the caller, in the representation of the
    source and within its resource budget, because a circuit matrix needs
    memory exponential in the circuit width.

    Dense Hermitian evidence uses exact entry equality, so even an asymmetry
    from rounding during assembly is rejected. If the intended target is Hermitian, the
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
            only and never forms a circuit matrix or applies an operator to
            a state.
        max_work: Default None, no limit. Limit on the work of the exact
            dense syntheses that controlling the two passes makes, and of Qiskit's
            control of the synthesized gates, in the work units of
            [Exact dense synthesis](../../development/dense_synthesis.md#admission-of-the-exact-synthesis).
            A child that holds a dense
            ``UnitaryGate``, such as a ``dense_dilation`` encoding, has that
            unitary synthesized once for each pass, and its adjoint once
            for each pass of degree two or more, and the parity control
            unrolls the synthesized gates at every query of the pass. The
            parity control takes the gate-wise route whatever route the
            joint generator uses, because its controlled object is the whole
            pass. A gate that an earlier control already synthesized, such
            as a branch of a two-child joint generator, is controlled again
            outside this count. Neither this builder nor the joint-generator
            builder counts that second control, which is left to the caller
            that composes them.
        max_bytes: Default `None`, the physical memory of this computer. Limit on
            the working and kept bytes of those syntheses and control steps.
        residual_tolerance: Default `1e-12`. Positive limit on the actual
            coefficient residual of each parity's phases, relative to its
            normalized target. Before amplification the residual adds
            ``s * (residual_cos + residual_sin)`` to the error ``d`` of the
            block, for the common rescale ``s``. This builder always applies
            the three-use OAA, whose error ``Phi_a(d)`` is nonlinear in ``d``,
            so ``(allowance - tail_bound) / (2 s)`` is only an upper limit for
            a known total allowance and can overshoot it. The sufficient
            choice is ``(Phi^-1(allowance) - tail_bound) / (2 s)`` with
            ``Phi(d) = d + d**2 (3 + d) / 2``
            ([phase targets](../../algorithms/lchs.md#shared-route-phase-target)).
            It is ignored when `_prepared` is given.
        _prepared: Default `None`. A `prepare_qsp_evolution` result for
            ``tau = encoding.alpha * evolution_time`` and the same `epsilon`,
            used in place of a new phase solve. A caller leaves it at its
            default.

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
            ``tau = encoding.alpha * evolution_time``: a memory refusal, or a
            phase residual above `residual_tolerance`.
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
    prepared = _prepared or prepare_qsp_evolution(tau=tau, epsilon=epsilon, max_bytes=max_bytes,
                                                  residual_tolerance=residual_tolerance)
    if prepared.expansion.tau != tau or prepared.expansion.epsilon != epsilon:
        raise ValueError("QSP construction differs from its frozen numerical selection")
    expansion = prepared.expansion
    cos_solution, sin_solution = prepared.cos_solution, prepared.sin_solution
    scale, target_norming_bounds = prepared.scale, prepared.target_norming_bounds

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
    from nwqlib.subroutines.qiskit_compat import dense_gatewise_charges
    cos_widths, cos_counts, _ = dense_gatewise_charges(cos_gate, 1)
    sin_widths, sin_counts, _ = dense_gatewise_charges(sin_gate, 1)
    admit_dense_syntheses(cos_widths + sin_widths, controls=(cos_counts, sin_counts),
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
    # A caller may divide the recovered postselected amplitudes by the known
    # factor 3a - 4a^3 and then use the residual-only bound in that frame.
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
        "target_cap": prepared.target_cap,
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
        "classical_preprocessing_scope": "canonical_real_Wx_phase_synthesis",
        "phase_error_qualification": "numerical_uniform_error_estimate",
        "phase_arithmetic_error_bound": None,
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


# Shared real QSVT and qubitized GQSP emitters.
#
# Both emitters read their encoding through block_encoding.joint.access_summary and build_access_query, so
# native plans, node recipes, joint generators, contractions and supplied circuits share one control and
# adjoint convention. Each builder states the identities it uses (GSLW arXiv:1806.01838v1 Corollary 18,
# Lemma 19, Theorems 56/58; Motlagh-Wiebe arXiv:2308.01501v2 Theorem 6 and Corollary 8; Suenderhauf
# arXiv:2312.00723v1 Appendix C.2), including the flag, sign and recovery conventions.


def _append_reflection(circuit, targets, *, controls=(), ctrl_state=0, positive=True, spare=None):
    """Append ``2 Pi - I`` (``positive``) or ``I - 2 Pi``, ``Pi = |0..0><0..0|`` on ``targets``, on ``ctrl_state``.

    ``I - 2 Pi`` is an exact phase flip of the all-zero target state with the
    controls on their value; ``2 Pi - I`` adds the exact -1 on the active
    control value (a global sign of the uncontrolled reflection is a
    relative phase once controlled). With no targets ``Pi = I``, so the
    positive reflection is the identity and is omitted.
    """
    from nwqlib.subroutines.block_encoding.core import _append_phase_flip

    targets, controls = list(targets), list(controls)
    if targets:
        _append_phase_flip(circuit, [*targets, *controls], ctrl_state << len(targets), None)
    if positive == bool(targets):
        _append_phase_flip(circuit, controls, ctrl_state, spare if spare is not None else
                           (targets[0] if targets else None))


def _access_premises(summary, *, emitter):
    """Return the Hermitian-target premise; refuse a target known to be non-Hermitian."""
    if summary.target_operator_is_hermitian is False:
        raise ValueError(f"{emitter} requires a Hermitian target generator; this encoding records a "
                         "non-Hermitian target")
    if summary.target_operator_is_hermitian is True:
        return "construction"
    warnings.warn(f"{emitter} assumes a Hermitian target generator; the encoding records no evidence. The "
                  "error bound is conditional on this premise.", UserWarning, stacklevel=3)
    return "caller_assumption"


def _rotation(circuit, gate_class, angle, target, controls, state):
    if controls:
        circuit.append(gate_class(angle).control(len(controls), ctrl_state=state, annotated=False), [*controls, target])
    else:
        circuit.append(gate_class(angle), [target])


@dataclass(frozen=True, kw_only=True)
class SharedQSVTEvolutionPlan:
    """The shared real-QSVT evolution of one encoding, described without a circuit.

    Attributes:
        evolution_time: T_e, the physical time of the encoded generator.
        tau: ``alpha_S T_e``, the polynomial time of ``prepared``.
        signal_access: ``"direct"`` when the encoded block is recorded as
            Hermitian, otherwise ``"generic_lift"``, which uses
            ``V_0 = H_h diag(U, I) X_h diag(U, I)^dagger H_h``.
        amplification: ``"raw"`` (one use) or ``"oaa"`` (three uses and two
            full-success reflections).
        hermitian_target_premise: ``"construction"`` or ``"caller_assumption"``.
        cos_degree: Degree of the even cosine polynomial.
        sin_degree: Degree of the odd sine polynomial.
        ladder_degree: n, the shared uncontrolled queries before the one
            controlled final query; the short branch is padded with
            cancelling query pairs to degree n when the degrees differ by
            more than one.
        long_branch: ``"cosine"`` or ``"sine"``, the branch with the
            controlled final query.
        polynomial_scale: s, the common rescale of both polynomials.
        raw_amplitude: ``a = 1/(2s)``.
        oaa_factor: ``r = 3a - 4a^3``.
        recovery: zeta, ``2s`` (raw) or ``1/r`` (OAA). The evolution block
            times zeta approximates ``exp(-i T_e S)``.
        algorithmic_error_bound: ``delta`` (raw) or ``Phi_a(delta)`` (OAA),
            None when a tail underflowed or is unknown.
        child_error: ``T_e delta_S``, None when ``delta_S`` is unknown.
        error_bound: Their sum, the bound on
            ``||zeta B - exp(-i T_e S)||`` for the success block B, without
            native gate error.
        num_ancillas: Flag, parity and base ancillas (``a_U + 2`` direct,
            ``a_U + 3`` lift).
        system_qubits: Width of the encoded system.
        query_census: Calls of the original encoding per forward evolution
            and per evolution with one external control, as
            ``(query, controls, adjoint, count)`` rows, plus phase slots,
            flag toggles, reflections, the lift's ``X_h`` gates as
            ``(controls, count)`` rows (``helper_x``) and the CX the emitter
            adds around the queries of the uncontrolled evolution
            (``forward_gate_cx``, `_shared_gate_cx`).
        phase_qualification: The numerical status of the phases and tails.
    """

    evolution_time: float
    tau: float
    signal_access: str
    amplification: str
    hermitian_target_premise: str
    cos_degree: int
    sin_degree: int
    ladder_degree: int
    long_branch: str
    polynomial_scale: float
    raw_amplitude: float
    oaa_factor: float
    recovery: float
    algorithmic_error_bound: float | None
    child_error: float | None
    error_bound: float | None
    num_ancillas: int
    system_qubits: int
    query_census: Mapping[str, Any]
    phase_qualification: str = "numerical_uniform_error_estimate"

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import fields
        return {item.name: (dict(getattr(self, item.name)) if item.name == "query_census"
                            else getattr(self, item.name)) for item in fields(self)}


def _shared_census(n, lift, amplification, k):
    """Rows ``(query, controls, adjoint, count)`` of the original U for one evolution with k external controls.

    OAA uses the raw circuit twice and its reverse once; the reverse swaps
    U and U^dagger.
    """
    first_controlled = int(k > 0 and n % 2 == 1)
    if lift:
        # Every V_0 uses C_{h=0}(U^dagger) and C_{h=0}(U); only X_h carries p and the external controls.
        raw = [("U", 1, False, n + 1), ("U", 1, True, n + 1)]
    else:
        raw = [("U", 0, False, (n + 1) // 2 - first_controlled), ("U", 0, True, n // 2),
               ("U", 1 + k, (n + 1) % 2 == 0, 1), ("U", k, False, first_controlled)]
    totals: dict[tuple, int] = {}
    for reverse, weight in ((False, 2), (True, 1)) if amplification == "oaa" else ((False, 1),):
        for name, controls, adjoint, count in raw:
            key = (name, controls, adjoint != reverse)
            totals[key] = totals.get(key, 0) + weight * count
    return tuple((*key, count) for key, count in sorted(totals.items()) if count)


def _shared_helper_x(n, lift, amplification, k):
    """Rows ``(controls, count)`` of the lift's ``X_h`` gates for one evolution with k external controls.

    Each ``V_0`` holds one ``X_h``. On the ladder ``j = 1..n`` it is plain,
    except that with external controls and odd n the first carries them;
    the final query's ``X_h`` carries p and the controls. Without a lift
    there is no ``X_h``.
    """
    if not lift:
        return ()
    uses = 3 if amplification == "oaa" else 1
    first = int(k > 0 and n % 2 == 1)
    totals: dict[int, int] = {}
    for controls, count in ((0, n - first), (k, first), (1 + k, 1)):
        totals[controls] = totals.get(controls, 0) + uses * count
    return tuple((controls, count) for controls, count in sorted(totals.items()) if count)


def _shared_gate_cx(n, base, lift, amplification):
    r"""CX that `build_shared_qsvt_evolution` adds around the U calls of one uncontrolled evolution, or None.

    With ``M(c)`` the CX of an X with c controls (`_mcx_cx`, Qiskit's count
    without ancillas) and ``b = a_U + [lift]`` base ancillas, the raw circuit
    has ``n + 2`` phase slots of two ``C_Pi X_f`` toggles (``M(b)`` each) and
    one RZ multiplexed on p (2 CX), the odd branch's ``Z_f`` controlled on p
    (``M(1)``) and the final ``X_h`` controlled on p (``M(1)``, lift only);
    the other ``X_h`` are plain. OAA repeats the raw circuit three times and
    adds two success reflections on ``b + 2`` qubits, ``M(b + 1)`` each, and
    its correcting global sign costs no CX:

    \[G_{\rm raw}=(n+2)[2M(b)+2]+M(1)+[{\rm lift}]M(1),\qquad
      G_{\rm OAA}=3G_{\rm raw}+2M(b+1).\]

    These counts are read from the gates the emitter appends. None when an
    MCX count is outside the tabulated range.
    """
    from nwqlib.subroutines._multiplexors import multiplexor_resource_law
    from nwqlib.subroutines.block_encoding.core import _mcx_cx, _UnknownMCXCount

    try:
        toggle, reflection, single = _mcx_cx(base), _mcx_cx(1 + base), _mcx_cx(1)
    except _UnknownMCXCount:
        return None
    ucrz = multiplexor_resource_law("ucrz", control_qubits=1)["basis_cx_gates"]
    raw = (n + 2) * (2 * toggle + ucrz) + single + int(lift) * single
    return raw if amplification == "raw" else 3 * raw + 2 * reflection


def plan_shared_qsvt_evolution(source, prepared: QSPPreparedEvolution, *, evolution_time: float,
                               amplification: str = "oaa") -> SharedQSVTEvolutionPlan:
    r"""Describe the shared real-QSVT evolution of an encoding without building a circuit.

    The construction merges the ``+Phi/-Phi`` pair qubit and the
    signal into one flag f: with ``R = 2 Pi - I`` for the base success
    projector ``Pi``,
    ``G_phi = C_Pi X_f RZ_f(2 phi) C_Pi X_f = |0><0|_f e^{i phi R} +
    |1><1|_f e^{-i phi R}``, so preparing and projecting f on ``|+>``
    averages ``P`` and its conjugate. A parity qubit p multiplexes the
    cosine and sine phases on one alternating ladder ``U, U^dagger, ...``;
    the first n queries are shared and uncontrolled and the longer branch
    ends with one p-controlled query. The Wx-to-reflection conversion
    keeps ``g_d = d pi/2`` per branch as a p-relative phase, ``Z_f`` on
    the odd branch, and the sine coefficient ``-i``. The raw block is
    ``a (C - i S)``, ``a = 1/(2s)``, so ``zeta = 2s``; three-use OAA gives
    ``3B - 4B B^dagger B`` with ``zeta = 1/(3a - 4a^3)``. The error
    in the recovered frame is ``delta`` or ``Phi_a(delta)`` (from
    ``qsp_evolution_error_terms``) plus ``T_e delta_S`` added once after the
    transformation.

    Args:
        source (BlockEncodingPlan | BlockEncoding | OperatorInput | JointEncodingPlan | ScaledBranchAccess | UniformNodeRecipe | DiagonalTable | ComplexDiagonalTable): Encoding of the Hermitian generator S, any source accepted
            by ``block_encoding.joint.access_summary``.
        prepared (QSPPreparedEvolution): Phases from
            ``prepare_qsp_evolution`` for ``tau = alpha_S T_e``.
        evolution_time (float): T_e > 0.
        amplification (str): Default ``"oaa"``; or ``"raw"``.

    Returns:
        plan (SharedQSVTEvolutionPlan): Degrees, recovery, error and query
        census.

    Raises:
        ValueError: For a non-Hermitian target, a ``prepared`` solved for a
            different ``tau``, or an unknown amplification.

    Warns:
        UserWarning: When the target's Hermiticity is not recorded.
    """
    from nwqlib.subroutines.block_encoding.joint import access_summary

    summary = access_summary(source)
    premise = _access_premises(summary, emitter="shared QSVT evolution")
    evolution_time = finite_real(evolution_time, "evolution_time")
    if evolution_time <= 0:
        raise ValueError("evolution_time must be positive")
    if amplification not in ("raw", "oaa"):
        raise ValueError("amplification must be 'raw' or 'oaa'")
    tau = summary.alpha * evolution_time
    if prepared.expansion.tau != tau:
        raise ValueError(f"prepared phases have tau={prepared.expansion.tau}, but alpha_S*T_e={tau}")
    lift = summary.encoded_operator_is_hermitian is not True
    d_c, d_s = prepared.cos_solution.degree, prepared.sin_solution.degree
    if (d_c - d_s) % 2 == 0:
        raise ValueError("the cosine and sine polynomials must have opposite parity")
    n = max(d_c, d_s) - 1
    terms = qsp_evolution_error_terms(prepared, evolution_time=evolution_time,
                                      child_error_bound=summary.error_bound)
    if amplification == "raw":
        recovery, algorithmic = 2.0 * prepared.scale, terms["polynomial_error"]
    else:
        recovery, algorithmic = terms["recovery_scale"], terms["algorithmic_error"]
    child = terms["child_error"]
    error = None if algorithmic is None or child is None else algorithmic + child
    uses = 3 if amplification == "oaa" else 1
    base = summary.num_ancillas + int(lift)
    census = {
        "forward": _shared_census(n, lift, amplification, 0),
        "controlled": _shared_census(n, lift, amplification, 1),
        "phase_slots": uses * (n + 2),
        "flag_toggles": 2 * uses * (n + 2),
        "full_success_reflections": uses - 1,
        "helper_x": {"forward": _shared_helper_x(n, lift, amplification, 0),
                     "controlled": _shared_helper_x(n, lift, amplification, 1)},
        "forward_gate_cx": _shared_gate_cx(n, base, lift, amplification),
    }
    return SharedQSVTEvolutionPlan(
        evolution_time=evolution_time, tau=tau, signal_access="generic_lift" if lift else "direct",
        amplification=amplification, hermitian_target_premise=premise, cos_degree=d_c, sin_degree=d_s,
        ladder_degree=n, long_branch="cosine" if d_c > d_s else "sine", polynomial_scale=prepared.scale,
        raw_amplitude=terms["amplitude"], oaa_factor=terms["oaa_amplitude_factor"], recovery=recovery,
        algorithmic_error_bound=algorithmic, child_error=child, error_bound=error, num_ancillas=base + 2,
        system_qubits=summary.system_qubits, query_census=census)


def _shared_evolution_circuit(source, prepared, plan, *, num_controls, ctrl_state, max_bytes, max_work):
    """Assemble the shared evolution on [controls, f, p, (h), base ancillas, system]."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XGate
    from nwqlib.subroutines._multiplexors import append_control_diagonal_phases, append_uniformly_controlled_rz
    from nwqlib.subroutines.block_encoding.core import _append_phase_flip, _control_state
    from nwqlib.subroutines.block_encoding.joint import _complete_dense_children, access_summary, build_access_query
    from nwqlib.subroutines.qiskit_compat import controlled, inverse_realized_gate

    k = integer(num_controls, "num_controls", 0)
    state = _control_state(ctrl_state, k)
    source = _complete_dense_children(source, max_bytes=max_bytes, max_work=max_work)
    summary = access_summary(source)
    lift = plan.signal_access == "generic_lift"
    n, a_u = plan.ladder_degree, summary.num_ancillas

    def query(**options):
        return build_access_query(source, max_bytes=max_bytes, max_work=max_work, **options)

    width = 2 + int(lift) + a_u + summary.system_qubits
    raw = QuantumCircuit(k + width, name="shared_qsvt_raw")
    controls = list(raw.qubits[:k])
    f, p = raw.qubits[k], raw.qubits[k + 1]
    helper = [raw.qubits[k + 2]] if lift else []
    ancillas = list(raw.qubits[k + 2 + int(lift):k + 2 + int(lift) + a_u])
    system = list(raw.qubits[k + 2 + int(lift) + a_u:])
    base = [*helper, *ancillas]

    branches = []
    for solution in (prepared.cos_solution, prepared.sin_solution):
        reflection, _ = wx_phases_to_reflection(solution.phases)
        padded = np.zeros(n + 2)
        padded[:len(reflection)] = reflection
        branches.append((padded, solution.degree))
    (cos_phases, d_c), (sin_phases, d_s) = branches
    long_value = 0 if d_c > d_s else 1

    if lift:
        forward_c, adjoint_c = query(num_controls=1, ctrl_state=0), query(num_controls=1, ctrl_state=0,
                                                                         adjoint=True)

        def base_query(extra, extra_state):
            # V_0 = H_h D_U X_h D_U^dagger H_h with D_U = diag(U, I)_h; only X_h carries controls.
            raw.h(helper[0])
            raw.append(adjoint_c, [helper[0], *ancillas, *system])
            if extra:
                raw.append(XGate().control(len(extra), ctrl_state=extra_state, annotated=False), [*extra, helper[0]])
            else:
                raw.x(helper[0])
            raw.append(forward_c, [helper[0], *ancillas, *system])
            raw.h(helper[0])
    else:
        gates = {}

        def base_query(extra, extra_state, dagger):
            key = (len(extra), extra_state, dagger)
            if key not in gates:
                gates[key] = query(num_controls=len(extra), ctrl_state=extra_state, adjoint=dagger)
            raw.append(gates[key], [*extra, *ancillas, *system])

    def active(values):
        """Table over [p, controls]: the parity-indexed value on the active control value, else 0."""
        return [values[index & 1] if index >> 1 == state else 0.0 for index in range(1 << (k + 1))]

    flip = controlled(XGate(), len(base), ctrl_state=0) if base else XGate()

    def phase_slot(j):
        # G_phi = C_Pi X_f RZ_f(2 phi) C_Pi X_f = |0><0|_f e^{i phi R} + |1><1|_f e^{-i phi R}.
        raw.append(flip, [*base, f])
        append_uniformly_controlled_rz(raw, f, [p, *controls], active((2.0 * cos_phases[j], 2.0 * sin_phases[j])))
        raw.append(flip, [*base, f])

    raw.h(f)
    raw.h(p)
    # Z_f on each odd-degree branch, exactly, on the active control value.
    for value, degree in ((0, d_c), (1, d_s)):
        if degree % 2:
            _append_phase_flip(raw, [f, p, *controls], 1 | (value << 1) | (state << 2), None)
    # Branch phases e^{i g_d} (g_d = d pi/2) and the sine coefficient -i, as a p-diagonal on the active value.
    append_control_diagonal_phases(raw, [p, *controls], active((d_c * np.pi / 2.0, d_s * np.pi / 2.0 - np.pi / 2.0)))
    phase_slot(0)
    for j in range(1, n + 1):
        extra, extra_state = ((controls, state) if k and j == 1 and n % 2 else ((), 0))
        if lift:
            base_query(extra, extra_state)
        else:
            base_query(extra, extra_state, j % 2 == 0)
        phase_slot(j)
    final = [p, *controls]
    final_state = long_value | (state << 1)
    if lift:
        base_query(final, final_state)
    else:
        base_query(final, final_state, (n + 1) % 2 == 0)
    phase_slot(n + 1)
    raw.h(f)
    raw.h(p)

    if plan.amplification == "raw":
        circuit = raw
    else:
        gate = raw.to_gate(label="shared_qsvt_raw")
        inverse = inverse_realized_gate(gate, native_ucg=False)
        circuit = QuantumCircuit(k + width, name="shared_qsvt_oaa")
        success = list(circuit.qubits[k:k + 2 + len(base)])
        wires = list(circuit.qubits)
        circuit.append(gate, wires)
        _append_reflection(circuit, success, controls=wires[:k], ctrl_state=state, positive=False)
        circuit.append(inverse, wires)
        _append_reflection(circuit, success, controls=wires[:k], ctrl_state=state, positive=False)
        circuit.append(gate, wires)
        # U R U^dagger R U with R = I - 2 Pi has block -(3B - 4 B B^dagger B); restore the sign exactly.
        from nwqlib.subroutines.block_encoding.core import _append_phase_flip as flip_sign
        flip_sign(circuit, wires[:k], state, wires[k])
    return circuit


def build_shared_qsvt_evolution(source, prepared: QSPPreparedEvolution, *, evolution_time: float,
                                amplification: str = "oaa", max_bytes: int | None = None,
                                max_work: int | None = None) -> BlockEncoding:
    """Build the shared real-QSVT evolution ``exp(-i T_e S)`` of an encoding of S.

    See `plan_shared_qsvt_evolution` for the construction. The returned
    encoding has ``alpha = zeta`` and ``error_bound`` the recovered-frame
    bound ``||zeta B - exp(-i T_e S)||``, both from that plan, so it follows
    the block-encoding convention: the success block times alpha
    approximates the evolution. Native gate error is not included.

    Args:
        source (BlockEncodingPlan | BlockEncoding | OperatorInput | JointEncodingPlan | ScaledBranchAccess | UniformNodeRecipe | DiagonalTable | ComplexDiagonalTable): Encoding of the Hermitian generator S.
        prepared (QSPPreparedEvolution): Phases for ``tau = alpha_S T_e``.
        evolution_time (float): T_e > 0.
        amplification (str): Default ``"oaa"``; or ``"raw"``.
        max_bytes (int | None): Default None, the physical memory. Limit on
            the dominant arrays of each child query, which each query checks
            before it builds them.
        max_work (int | None): Default None, no limit. Construction work
            limit of each child query.

    Returns:
        encoding (BlockEncoding): Implementation ``"shared_qsvt_evolution"``;
        ancillas flag, parity, (helper), child ancillas; metadata holds the
        plan record and the phase evidence.

    Raises:
        ValueError: As for `plan_shared_qsvt_evolution`, or for an exceeded
            limit.
    """
    plan = plan_shared_qsvt_evolution(source, prepared, evolution_time=evolution_time,
                                      amplification=amplification)
    circuit = _shared_evolution_circuit(source, prepared, plan, num_controls=0, ctrl_state=None,
                                           max_bytes=max_bytes, max_work=max_work)
    from nwqlib.subroutines.block_encoding import BlockEncoding
    return BlockEncoding(circuit=circuit, alpha=plan.recovery, num_ancillas=plan.num_ancillas,
                         system_qubits=plan.system_qubits, error_bound=plan.error_bound,
                         implementation="shared_qsvt_evolution",
                         metadata={"plan": plan.to_dict(),
                                   "phase_solver": {
                                       "cos": {k: v for k, v in prepared.cos_solution.to_dict().items()
                                               if k != "phases"},
                                       "sin": {k: v for k, v in prepared.sin_solution.to_dict().items()
                                               if k != "phases"}},
                                   "phase_arithmetic_error_bound": None})


def build_shared_qsvt_evolution_query(source, prepared: QSPPreparedEvolution, *, evolution_time: float,
                                      amplification: str = "oaa", num_controls: int = 0,
                                      ctrl_state: int | str | None = None, adjoint: bool = False,
                                      max_bytes: int | None = None, max_work: int | None = None):
    """Build the shared evolution, its adjoint, or either with external controls, as a gate.

    With controls, the inactive branch is the identity on every qubit: the
    flag and parity Hadamards are left uncontrolled because everything
    between them is the identity there (every phase angle, branch phase and
    odd correction carries the controls, and the ``C_Pi X_f`` pairs cancel),
    and when n is odd the first ladder query also carries the controls, so
    the uncontrolled queries cancel in pairs. The final query carries p and
    the controls. The OAA reflections and the restored sign carry the
    controls. Census: `SharedQSVTEvolutionPlan.query_census`.

    Args:
        source (BlockEncodingPlan | BlockEncoding | OperatorInput | JointEncodingPlan | ScaledBranchAccess | UniformNodeRecipe | DiagonalTable | ComplexDiagonalTable): Encoding of the Hermitian generator S.
        prepared (QSPPreparedEvolution): Phases for ``tau = alpha_S T_e``.
        evolution_time (float): T_e > 0.
        amplification (str): Default ``"oaa"``; or ``"raw"``.
        num_controls (int): Default `0`. Number of external controls.
        ctrl_state (int | str | None): Default `None`, all controls one.
        adjoint (bool): Default `False`. Reverse the whole construction.
        max_bytes (int | None): Default None, the physical memory. As for
            `build_shared_qsvt_evolution`.
        max_work (int | None): Default None, no limit.

    Returns:
        gate (qiskit.circuit.Gate): On the controls, then the evolution's
            ancillas and system, in that order.
    """
    from nwqlib.subroutines.qiskit_compat import inverse_realized_gate

    plan = plan_shared_qsvt_evolution(source, prepared, evolution_time=evolution_time,
                                      amplification=amplification)
    circuit = _shared_evolution_circuit(source, prepared, plan, num_controls=num_controls,
                                        ctrl_state=ctrl_state, max_bytes=max_bytes, max_work=max_work)
    gate = circuit.to_gate(label=circuit.name)
    return inverse_realized_gate(gate, native_ucg=False) if adjoint else gate


@dataclass(frozen=True, kw_only=True)
class QubitizedTarget:
    """A Laurent or one-sided target with its canonical phases, cap, recovery and error terms.

    The qubitized circuit realizes ``J^dagger f(W) J = f_target(x)`` with
    ``x`` the normalized encoded operator; the recovered operator is
    ``scale`` times the success block.

    Attributes:
        kind: ``"jacobi_anger_laurent"`` (`prepare_jacobi_anger_laurent`),
            ``"jacobi_anger_gqet"`` (`prepare_jacobi_anger_gqet`) or
            ``"imaginary_time_gqet"`` (`imaginary_time_target`), or
            ``"laurent_polynomial"`` from a supplied polynomial.
        solution: Canonical phases (``LaurentQSPPhases``), or None for the
            exact identity of a zero imaginary-time beta.
        scale: sigma, the recovery zeta of the route.
        polynomial_time: ``tau = alpha_S T_e`` for real time, beta for the
            imaginary-time filter.
        margin: eta, the normalized target cap is at most ``1 - eta``.
        norm_cap: The circle bound B of the polynomial before division by
            scale.
        norm_cap_qualification: What B rests on.
        tail_bound: ``delta_T`` (or ``delta_m``), None when positive but
            below binary64 range (see ``tail_log_bound``).
        tail_log_bound: Natural logarithm of the positive tail, or None.
        synthesis_residual: Coefficient one-norm residual of the realized
            normalized target, a numerical estimate.
        expansion: The Jacobi-Anger or imaginary-time expansion used.
        approximation_error_bound: Recovered-frame individual polynomial
            error when supplied. None means unknown for ``"laurent_polynomial"``;
            other kinds then retain their expansion's truncation error.
    """

    kind: str
    solution: Any
    scale: float
    polynomial_time: float
    margin: float
    norm_cap: float
    norm_cap_qualification: str
    tail_bound: float | None
    tail_log_bound: float | None
    synthesis_residual: float | None
    expansion: Any
    approximation_error_bound: float | None = None


@dataclass(frozen=True, kw_only=True)
class QubitizedTargetPlan:
    """A Laurent or GQET target and its phase allowance, before any phase solve.

    `plan_jacobi_anger_laurent` and `plan_jacobi_anger_gqet` own the target
    coefficients, support, scale and circle cap; `with_phase_allowance`
    adds the residual allowance and solve limits; `prepare_qubitized_target`
    solves exactly this target with exactly these controls.

    Attributes:
        kind: ``"jacobi_anger_laurent"``, ``"jacobi_anger_gqet"`` or
            ``"laurent_polynomial"``.
        expansion: The Jacobi-Anger expansion, or None for a supplied polynomial.
        polynomial_time: Effective real time of the approximated exponential.
        approximation_error_bound: Recovered-frame individual polynomial error,
            including coefficient construction when applicable; None means unknown.
        tail_bound: Reference truncation error, separate from an individual
            stochastic member's error; None means unknown.
        tail_log_bound: Logarithm of the reference truncation error, if available.
        margin: eta, the normalized target cap is at most ``1 - eta``.
        coefficients: Normalized target ``f = p/sigma``; entry j multiplies
            ``z**(minimum_power + j)``.
        minimum_power: Lowest retained power after removing exact endpoint zeros.
        signals: Analytic degree d of the walk sequence: ``2m`` (Laurent) or
            m (GQET), with m the expansion degree.
        negative_signals: m (Laurent) or 0 (GQET).
        scale: sigma, the recovery zeta of the route.
        norm_cap: The circle bound B of the polynomial before division by sigma.
        norm_cap_qualification: What B rests on.
        norm_bound: Strict cap of ``|f|`` on the unit circle passed to the solver.
        solver_qualification: The cap's qualification as the solver records it.
        full_ja: ``(sigma, delta_T)`` of the near-flat aliasing option (Laurent
            only, a full complex approximation of a unit-modulus target), else None.
        epsilon_phase: Normalized-frame residual allowance, None before
            `with_phase_allowance`.
        max_bytes: Memory limit of the solve; None uses the physical memory.
        max_fft: Optional FFT length cap of the solve.
    """

    kind: str
    expansion: JacobiAngerExpansion | None
    polynomial_time: float
    approximation_error_bound: float | None
    tail_bound: float | None
    tail_log_bound: float | None
    margin: float
    coefficients: tuple[complex, ...]
    minimum_power: int
    signals: int
    negative_signals: int
    scale: float
    norm_cap: float
    norm_cap_qualification: str
    norm_bound: float
    solver_qualification: str
    full_ja: tuple[float, float] | None
    epsilon_phase: float | None = None
    max_bytes: int | None = None
    max_fft: int | None = None

    @property
    def maximum_power(self):
        """Highest retained power; internal zero coefficients remain present."""
        return self.minimum_power + len(self.coefficients) - 1


def plan_laurent_target(coefficients, *, minimum_power: int, polynomial_time: float,
                        scale: float, margin: float, norm_bound: float,
                        norm_cap_qualification: str, approximation_error_bound: float | None,
                        tail_bound: float | None = None, tail_log_bound: float | None = None) -> QubitizedTargetPlan:
    """Describe a supplied normalized Laurent polynomial before phase synthesis.

    ``coefficients[j]`` multiplies ``z**(minimum_power+j)``. ``scale`` is
    its positive physical recovery, and ``polynomial_time`` is the nonnegative
    effective time of the approximated exponential. ``norm_bound`` is a supplied
    circle cap in [0, 1-margin], qualified by ``norm_cap_qualification``.
    ``approximation_error_bound`` bounds this individual recovered polynomial's
    error, not an ensemble mean; None preserves an unknown error. The optional
    tail and log-tail describe a separate reference truncation.

    Only exact endpoint zeros are removed. The signal degree is
    max(0,-minimum_power)+max(0,maximum_power), including zero in the walk's
    interval. No full-JA or real-Wx premise is inferred. Invalid scalar domains or coefficients raise ValueError.
    The returned plan is consumed by ``with_phase_allowance``.
    """
    values = tuple(complex(value) for value in coefficients)
    if not values or any(not (math.isfinite(v.real) and math.isfinite(v.imag)) for v in values):
        raise ValueError("Laurent coefficients must be nonempty and finite")
    try:
        lower = operator.index(minimum_power)
    except TypeError:
        raise ValueError("minimum_power must be an integer") from None
    if isinstance(minimum_power, (bool, np.bool_)):
        raise ValueError("minimum_power must be an integer")
    start, stop = 0, len(values)
    while start < stop and values[start] == 0:
        start += 1
    while stop > start and values[stop - 1] == 0:
        stop -= 1
    if start == stop:
        values, lower = (0j,), 0
    else:
        values, lower = values[start:stop], lower + start
    margin = _checked_margin(margin)
    scale = finite_real(scale, "scale")
    time = finite_real(polynomial_time, "polynomial_time")
    cap = finite_real(norm_bound, "norm_bound")
    if scale <= 0 or time < 0 or not 0 <= cap <= 1 - margin:
        raise ValueError("scale must be positive, time nonnegative and cap in [0,1-margin]")
    for name, value in (("approximation_error_bound", approximation_error_bound), ("tail_bound", tail_bound)):
        if value is not None and finite_real(value, name) < 0:
            raise ValueError(f"{name} must be nonnegative or None")
    upper = lower + len(values) - 1
    negative = max(0, -lower)
    return QubitizedTargetPlan(
        kind="laurent_polynomial", expansion=None, polynomial_time=time,
        approximation_error_bound=approximation_error_bound, tail_bound=tail_bound,
        tail_log_bound=tail_log_bound, margin=margin, coefficients=values,
        minimum_power=lower, signals=negative + max(0, upper), negative_signals=negative,
        scale=scale, norm_cap=scale * cap, norm_cap_qualification=str(norm_cap_qualification),
        norm_bound=cap, solver_qualification=str(norm_cap_qualification), full_ja=None)


def _checked_margin(margin):
    margin = finite_real(margin, "margin")
    if not 0 < margin < 1:
        raise ValueError("margin must lie strictly between zero and one")
    return margin


def _chebyshev_arrays(expansion, name):
    if expansion.tau <= 0:
        raise ValueError(f"a zero-time evolution is the identity; no {name} target is needed")
    a = np.zeros(expansion.degree + 1)
    b = np.zeros(expansion.degree + 1)
    a[:len(expansion.cos_coefficients)] = expansion.cos_coefficients
    b[:len(expansion.sin_coefficients)] = expansion.sin_coefficients
    return a, b


def plan_jacobi_anger_laurent(expansion: JacobiAngerExpansion, *, margin: float) -> QubitizedTargetPlan:
    r"""Plan the symmetric Jacobi-Anger Laurent target, without a phase solve.

    From the expansion's Chebyshev arrays ``C = sum a_k T_k`` and
    ``S = sum b_k T_k``, ``f_0 = a_0 - i b_0`` and
    ``f_{+-k} = (a_k - i b_k)/2``, since
    ``T_k((z + 1/z)/2) = (z^k + z^{-k})/2``. On the circle
    ``|p_m| <= 1 + delta_T`` in exact arithmetic, so
    ``sigma = (1 + delta_T)/(1 - eta)``, rounded up, caps the normalized
    target at ``1 - eta``; the binary64 Bessel coefficients carry no
    enclosure, which the recorded qualification states. The analytic degree
    is ``d = 2m`` and the walk uses 2m signals, m of them inverse
    (``D_-``), not extra walks. The near-flat aliasing option of the phase
    owner applies with ``full_ja_scale = sigma`` and ``full_ja_error =
    delta_T`` (T15, A3/A4; a positive tail below binary64 range counts as 0
    in sigma, which rounding up absorbs).

    Args:
        expansion (JacobiAngerExpansion): From `jacobi_anger_expansion`.
        margin (float): eta in (0, 1).

    Returns:
        plan (QubitizedTargetPlan): Kind ``"jacobi_anger_laurent"``, without an allowance.

    Raises:
        ValueError: For an invalid margin or a zero time.
    """
    margin = _checked_margin(margin)
    a, b = _chebyshev_arrays(expansion, "Laurent")
    tail = 0.0 if expansion.tail_bound is None else expansion.tail_bound
    sigma = float(np.nextafter((1.0 + tail) / (1.0 - margin), np.inf))
    m = expansion.degree
    positive = (a[1:] - 1j * b[1:]) / 2.0
    coefficients = np.concatenate([positive[::-1], [a[0] - 1j * b[0]], positive]) / sigma
    return QubitizedTargetPlan(
        kind="jacobi_anger_laurent", expansion=expansion, margin=margin,
        polynomial_time=expansion.tau, approximation_error_bound=expansion.tail_bound,
        tail_bound=expansion.tail_bound, tail_log_bound=expansion.tail_log_bound,
        coefficients=tuple(complex(c) for c in coefficients), minimum_power=-m, signals=2 * m,
        negative_signals=m, scale=sigma, norm_cap=1.0 + tail, norm_cap_qualification="1 + delta_T (A4)",
        norm_bound=1.0 - margin,
        solver_qualification="jacobi_anger_A4: |p_m| <= 1 + delta_T on the unit circle in exact arithmetic; "
                             "binary64 Bessel coefficients without enclosure",
        full_ja=(sigma, tail))


def plan_jacobi_anger_gqet(expansion: JacobiAngerExpansion, *, margin: float) -> QubitizedTargetPlan:
    r"""Plan the one-sided GQET target ``P_E(z) = sum_k c_k z^k`` of ``C - i S``, without a phase solve.

    ``c_k = a_k - i b_k`` for ``k = 0..m``, so ``J^dagger P_E(W) J =
    sum_k c_k T_k(x)``, the same truncated series as the Laurent target
    with the same tail ``delta_T``. The circle cap is the finite coefficient
    one-norm ``B_E = sum |c_k|``, which bounds ``|P_E(z)|`` on the whole
    unit circle by the triangle inequality; neither the Chebyshev interval
    norm nor a sampled circle maximum does. ``sigma_E = max(1, B_E)/(1 -
    eta)``, rounded up, and the stored normalized one-norm, rounded up, is
    the cap passed to the phase owner. m signals, none inverse (T17). The
    full Jacobi-Anger near-flat premise does not apply to this target.

    Args:
        expansion (JacobiAngerExpansion): From `jacobi_anger_expansion`.
        margin (float): eta in (0, 1).

    Returns:
        plan (QubitizedTargetPlan): Kind ``"jacobi_anger_gqet"``, without an allowance.

    Raises:
        ValueError: For an invalid margin or a zero time.
    """
    margin = _checked_margin(margin)
    a, b = _chebyshev_arrays(expansion, "GQET")
    coefficients = a - 1j * b
    cap = math.fsum(np.abs(coefficients))
    sigma = float(np.nextafter(max(1.0, cap) / (1.0 - margin), np.inf))
    normalized = coefficients / sigma
    return QubitizedTargetPlan(
        kind="jacobi_anger_gqet", expansion=expansion, margin=margin,
        polynomial_time=expansion.tau, approximation_error_bound=expansion.tail_bound,
        tail_bound=expansion.tail_bound, tail_log_bound=expansion.tail_log_bound,
        coefficients=tuple(complex(c) for c in normalized), minimum_power=0, signals=expansion.degree,
        negative_signals=0, scale=sigma, norm_cap=cap,
        norm_cap_qualification="coefficient one-norm sum |c_k| (triangle inequality)",
        norm_bound=float(np.nextafter(math.fsum(np.abs(normalized)), np.inf)),
        solver_qualification="stored normalized coefficient one-norm, rounded up", full_ja=None)


def with_phase_allowance(plan: QubitizedTargetPlan, *, epsilon_phase: float, max_bytes: int | None = None,
                         max_fft: int | None = None) -> QubitizedTargetPlan:
    """Attach the phase residual allowance and solve limits to one target, without solving.

    Args:
        plan (QubitizedTargetPlan): From `plan_jacobi_anger_laurent`,
            `plan_jacobi_anger_gqet` or `plan_laurent_target`.
        epsilon_phase (float): Positive normalized-frame residual allowance:
            ``epsilon_synth <= (epsilon_alg_allow - delta_T)/sigma`` for an
            algorithmic allowance ``epsilon_alg_allow`` (E15, T15, T17).
        max_bytes (int | None): Default None, the physical memory. Memory
            limit of each FFT attempt of the solve.
        max_fft (int | None): Default None. Optional FFT length cap.

    Returns:
        plan (QubitizedTargetPlan): ``plan`` with these controls.

    Raises:
        ValueError: For a nonpositive allowance or an invalid limit.
    """
    from dataclasses import replace

    epsilon_phase = finite_real(epsilon_phase, "epsilon_phase")
    if not epsilon_phase > 0:
        raise ValueError("epsilon_phase must be positive")
    max_fft = None if max_fft is None else integer(max_fft, "max_fft", 1)
    max_bytes = None if max_bytes is None else integer(max_bytes, "max_bytes", 1)
    return replace(plan, epsilon_phase=epsilon_phase, max_bytes=max_bytes, max_fft=max_fft)


def _check_target_start(plan: QubitizedTargetPlan) -> None:
    """Check the first FFT attempt that `prepare_qubitized_target` would make for ``plan``, without solving.

    The degree is that of `solve_laurent_qsp_phases`, the interval holding
    the target's powers and zero, and the check is
    `phases._check_synthesis_start` under the plan's ``max_fft`` and
    ``max_bytes``.
    """
    from nwqlib.subroutines.qsp.phases import _check_synthesis_start

    lower = plan.minimum_power
    degree = max(0, -lower) + max(0, lower + len(plan.coefficients) - 1)
    _check_synthesis_start(degree, max_fft=plan.max_fft, max_bytes=plan.max_bytes)


def prepare_qubitized_target(plan: QubitizedTargetPlan) -> QubitizedTarget:
    """Solve the canonical phases of one target with its frozen coefficients, cap and controls.

    Args:
        plan (QubitizedTargetPlan): From `with_phase_allowance`.

    Returns:
        target (QubitizedTarget): The kind of ``plan``.

    Raises:
        ValueError: For a plan without an allowance, a memory refusal of
            `solve_laurent_qsp_phases`, or its residual failure, which keeps
            its class and evidence and names the allowance to change.
    """
    from nwqlib.subroutines.qsp.phases import solve_laurent_qsp_phases

    if plan.epsilon_phase is None:
        raise ValueError("attach the phase allowance with with_phase_allowance before preparing the target")
    options = {} if plan.full_ja is None else {"full_ja_scale": plan.full_ja[0], "full_ja_error": plan.full_ja[1]}
    try:
        solution = solve_laurent_qsp_phases(
            plan.coefficients, minimum_power=plan.minimum_power, norm_bound=plan.norm_bound,
            norm_bound_qualification=plan.solver_qualification, residual_tolerance=plan.epsilon_phase,
            max_fft=plan.max_fft, max_bytes=plan.max_bytes, **options)
    except _QSPPhaseConvergenceError as error:
        raise type(error)(
            f"{error}. with_phase_allowance(epsilon_phase=...) chooses another target, and its documentation gives "
            "the effect on the error", evidence=error.evidence) from error
    return QubitizedTarget(kind=plan.kind, solution=solution, scale=plan.scale, polynomial_time=plan.polynomial_time,
                           margin=plan.margin, norm_cap=plan.norm_cap,
                           norm_cap_qualification=plan.norm_cap_qualification, tail_bound=plan.tail_bound,
                           tail_log_bound=plan.tail_log_bound,
                           synthesis_residual=solution.evidence.coefficient_residual, expansion=plan.expansion,
                           approximation_error_bound=plan.approximation_error_bound)


def prepare_jacobi_anger_laurent(expansion: JacobiAngerExpansion, *, margin: float, epsilon_phase: float,
                                 max_bytes: int | None = None, max_fft: int | None = None) -> QubitizedTarget:
    """Prepare the symmetric Jacobi-Anger Laurent target and its canonical phases.

    The target is `plan_jacobi_anger_laurent`, its controls
    `with_phase_allowance` and the solve `prepare_qubitized_target`.

    Args:
        expansion (JacobiAngerExpansion): From `jacobi_anger_expansion`.
        margin (float): eta in (0, 1).
        epsilon_phase (float): Normalized-frame residual allowance of the
            phase solve, from the caller's error allocation:
            ``epsilon_synth <= (epsilon_alg_allow - delta_T)/sigma`` for an
            algorithmic error allowance ``epsilon_alg_allow``.
        max_bytes (int | None): Default None, the physical memory. Memory
            limit of the phase solve.
        max_fft (int | None): Default None. Optional FFT length cap.

    Returns:
        target (QubitizedTarget): Kind ``"jacobi_anger_laurent"``.

    Raises:
        ValueError: For an invalid margin or allowance, or a memory refusal
            or residual failure of `solve_laurent_qsp_phases`, which
            propagates unchanged with its evidence.
    """
    return prepare_qubitized_target(with_phase_allowance(
        plan_jacobi_anger_laurent(expansion, margin=margin), epsilon_phase=epsilon_phase,
        max_bytes=max_bytes, max_fft=max_fft))


def prepare_jacobi_anger_gqet(expansion: JacobiAngerExpansion, *, margin: float, epsilon_phase: float,
                              max_bytes: int | None = None, max_fft: int | None = None) -> QubitizedTarget:
    """Prepare the one-sided GQET target ``P_E(z) = sum_k c_k z^k`` of ``C - i S`` and its phases.

    The target is `plan_jacobi_anger_gqet`, its controls
    `with_phase_allowance` and the solve `prepare_qubitized_target`.

    Args:
        expansion (JacobiAngerExpansion): From `jacobi_anger_expansion`.
        margin (float): eta in (0, 1).
        epsilon_phase (float): Normalized-frame residual allowance.
        max_bytes (int | None): Default None. As for
            `prepare_jacobi_anger_laurent`.
        max_fft (int | None): Default None. Optional FFT length cap.

    Returns:
        target (QubitizedTarget): Kind ``"jacobi_anger_gqet"``.

    Raises:
        ValueError: As for `prepare_jacobi_anger_laurent`.
    """
    return prepare_qubitized_target(with_phase_allowance(
        plan_jacobi_anger_gqet(expansion, margin=margin), epsilon_phase=epsilon_phase,
        max_bytes=max_bytes, max_fft=max_fft))


def imaginary_time_target(prepared) -> QubitizedTarget:
    """Wrap an imaginary-time filter from `prepare_imaginary_time_filter` as a qubitized target.

    The analytic ``Q_m(z) = e^{-beta} [I_0 + 2 sum I_k z^k]`` has
    nonnegative coefficients, so its circle maximum is ``Q_m(1) = w_m <=
    1``. The filter owner already fixed the scale, the margin and
    the phases; this record reuses them and its cumulative work without a
    new fit. ``scale`` multiplies the normalized residual once. The walk
    must encode ``X = -L_0/alpha`` (or, with time factors, the
    contraction ``X_a`` of `block_encoding.joint.plan_convex_contraction`),
    so the child term is ``beta delta_X``.

    Args:
        prepared (PreparedImaginaryTimeFilter): The prepared filter.

    Returns:
        target (QubitizedTarget): Kind ``"imaginary_time_gqet"``;
        ``solution`` is None for a structural beta of zero.
    """
    expansion = prepared.expansion
    selection = expansion.selection
    solution = prepared.solution
    return QubitizedTarget(
        kind="imaginary_time_gqet", solution=solution, scale=1.0 if solution is None else expansion.scale,
        polynomial_time=selection.beta, margin=expansion.margin,
        norm_cap=expansion.stored_norm_bound * expansion.scale,
        norm_cap_qualification=expansion.norm_bound_qualification,
        tail_bound=selection.tail_bound if solution is not None else 0.0,
        tail_log_bound=selection.tail_log_bound,
        synthesis_residual=None if solution is None else solution.evidence.coefficient_residual,
        expansion=expansion)


@dataclass(frozen=True, kw_only=True)
class QubitizedEvolutionPlan:
    """The qubitized GQSP circuit of one target on one encoding, described without a circuit.

    Attributes:
        kind: The target kind.
        signal_access: ``"direct_involution"`` (``W = (2 Pi - I) U``, only
            for a recorded full involution) or ``"generic_lift"``
            (``W = (2 Pi_0 - I) V_0``).
        hermitian_target_premise: ``"construction"`` or ``"caller_assumption"``.
        degree: Number of signals d.
        negative_signals: Signals ``D_-(W) = diag(I, W^dagger)``, the first
            ones of the mathematical sequence.
        recovery: zeta = the target scale sigma.
        algorithmic_error_bound: Individual polynomial approximation error plus
            ``sigma epsilon_synth``; the approximation is ``delta_T`` for a
            full expansion without an explicit override. None when unknown.
        child_error: ``(tau/alpha) delta`` of the encoding: ``T_e delta_S``
            or ``beta delta_X``; None when unknown.
        error_bound: Their sum (no native gate error).
        num_ancillas: Processing qubit plus walk ancillas (``a_U + 1``
            direct, ``a_U + 2`` lift).
        system_qubits: Width of the encoded system.
        query_census: Calls of the original encoding per forward circuit and
            per circuit with one external control, as
            ``(query, controls, adjoint, count)`` rows (`_qubitized_census`),
            the controlled walk reflections, processing rotations, the lift's
            ``X_h`` gates as ``(controls, count)`` rows (``helper_x``) and
            the CX the emitter adds around the queries of the uncontrolled
            circuit (``forward_gate_cx``, `_qubitized_gate_cx`).
    """

    kind: str
    signal_access: str
    hermitian_target_premise: str
    degree: int
    negative_signals: int
    recovery: float
    algorithmic_error_bound: float | None
    child_error: float | None
    error_bound: float | None
    num_ancillas: int
    system_qubits: int
    query_census: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import fields
        return {item.name: (dict(getattr(self, item.name)) if item.name == "query_census"
                            else getattr(self, item.name)) for item in fields(self)}


def _qubitized_census(degree, lift, k):
    """Rows ``(query, controls, adjoint, count)`` of the original U for d signals and k external controls (T24).

    A direct full-involution walk calls U once per signal, controlled by the
    processing qubit and the external controls; ``W^dagger = U R`` uses the
    same forward call. The generic lift calls ``C_{h=0}(U^dagger)`` and
    ``C_{h=0}(U)`` once each per signal, whatever the external controls,
    which go to ``X_h`` and the reflection only. Inverse signals change the
    order, not the count.
    """
    rows = [("U", 1, False, degree), ("U", 1, True, degree)] if lift else [("U", 1 + k, False, degree)]
    return tuple(row for row in rows if row[3])


def _qubitized_gate_cx(degree, walk, lift):
    r"""CX that `build_qubitized_evolution` adds around the U calls of one uncontrolled circuit, or None.

    Each signal has one walk reflection on ``a_w = a_U + [lift]`` qubits
    controlled by the processing qubit, ``M(a_w)``, and for the lift one
    ``X_h`` controlled by it, ``M(1)``; the processing rotations need no CX:
    ``G_walk = d [M(a_w) + [lift] M(1)]``, read from the gates the emitter
    appends. None when an MCX count is outside the tabulated range.
    """
    from nwqlib.subroutines.block_encoding.core import _mcx_cx, _UnknownMCXCount

    try:
        return degree * (_mcx_cx(walk) + int(lift) * _mcx_cx(1))
    except _UnknownMCXCount:
        return None


def plan_qubitized_evolution(source, target: QubitizedTarget, *,
                             evolution_time: float | None = None) -> QubitizedEvolutionPlan:
    r"""Describe the qubitized circuit of a target on an encoding, without building it.

    The walk is ``W = (2 Pi - I) U`` when the whole unitary is a
    recorded Hermitian involution, otherwise ``W = (2 Pi_0 - I) V_0`` with
    ``V_0 = H_h D_U X_h D_U^dagger H_h``, ``D_U = diag(U, I)_h``, whose
    controlled form uses two singly controlled calls of the original U per
    signal and controls only ``X_h``. For both,
    ``J^dagger W^{+-k} J = T_k(S_tilde/alpha)`` with ``S_tilde`` the
    Hermitian part of the encoded block, which is within ``delta_S`` of S
    even for an anti-Hermitian encoded defect. The GQSP sequence is
    ``A_0 D(W) A_1 ... D(W) A_d iX`` with ``D(W) = diag(W, I)_g`` and the
    first ``ell`` signals replaced by ``D_-(W) = diag(I, W^dagger)_g``,
    which give the negative powers of a Laurent target.

    Args:
        source (BlockEncodingPlan | BlockEncoding | OperatorInput | JointEncodingPlan | ScaledBranchAccess | UniformNodeRecipe | DiagonalTable | ComplexDiagonalTable): The encoding, any source accepted by
            ``block_encoding.joint.access_summary``.
        target (QubitizedTarget): Phases, recovery and individual approximation error.
        evolution_time (float | None): T_e for a real-time target, which
            must satisfy ``target.polynomial_time == alpha T_e``; None for an
            imaginary-time target, whose encoding has normalization one.

    Returns:
        plan (QubitizedEvolutionPlan): The description above.

    Raises:
        ValueError: For a non-Hermitian target generator, a time that does
            not match the target, or an imaginary-time encoding whose
            normalization is not one.

    Warns:
        UserWarning: When the target's Hermiticity is not recorded.
    """
    from nwqlib.subroutines.block_encoding.joint import access_summary

    summary = access_summary(source)
    premise = _access_premises(summary, emitter="qubitized evolution")
    if target.kind == "imaginary_time_gqet":
        if evolution_time is not None:
            raise ValueError("an imaginary-time filter uses beta from its target, not evolution_time")
        if summary.alpha != 1.0:
            raise ValueError("the imaginary-time filter needs an encoding of X with normalization one")
        time = target.polynomial_time
    else:
        time = finite_real(evolution_time, "evolution_time")
        if (time < 0 or (time == 0 and target.kind != "laurent_polynomial")
                or summary.alpha * time != target.polynomial_time):
            raise ValueError(f"target tau={target.polynomial_time} differs from alpha*T_e={summary.alpha * time}")
    lift = summary.full_unitary_involution is not True
    solution = target.solution
    degree = 0 if solution is None else solution.degree
    negative = 0 if solution is None else solution.negative_signals
    residual = target.synthesis_residual
    approximation = (target.tail_bound if target.approximation_error_bound is None
                     and target.kind != "laurent_polynomial" else target.approximation_error_bound)
    algorithmic = (0.0 if solution is None else
                   None if approximation is None or residual is None else approximation + target.scale * residual)
    child = None if summary.error_bound is None else time * summary.error_bound
    if solution is None:
        child = 0.0
    error = None if algorithmic is None or child is None else algorithmic + child
    walk = summary.num_ancillas + int(lift)
    census = {"forward": _qubitized_census(degree, lift, 0), "controlled": _qubitized_census(degree, lift, 1),
              "walk_reflections": degree, "processing_rotations": 3 * (degree + 1) + 1,
              "helper_x": {"forward": ((1, degree),) if lift and degree else (),
                           "controlled": ((2, degree),) if lift and degree else ()},
              "forward_gate_cx": _qubitized_gate_cx(degree, walk, lift)}
    return QubitizedEvolutionPlan(
        kind=target.kind, signal_access="generic_lift" if lift else "direct_involution",
        hermitian_target_premise=premise, degree=degree, negative_signals=negative, recovery=target.scale,
        algorithmic_error_bound=algorithmic, child_error=child, error_bound=error,
        num_ancillas=(0 if solution is None else 1 + int(lift) + summary.num_ancillas),
        system_qubits=summary.system_qubits, query_census=census)


def _qubitized_circuit(source, target, plan, *, num_controls, ctrl_state, max_bytes, max_work):
    """Assemble ``A_0 D A_1 ... D A_d iX`` on [controls, g, (h), ancillas, system]."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RXGate, RZGate, XGate
    from nwqlib.subroutines.block_encoding.core import _control_state
    from nwqlib.subroutines.block_encoding.joint import _complete_dense_children, access_summary, build_access_query

    k = integer(num_controls, "num_controls", 0)
    state = _control_state(ctrl_state, k)
    summary = access_summary(source)
    if target.solution is None:
        return QuantumCircuit(k + summary.system_qubits, name="identity_filter")
    lift = plan.signal_access == "generic_lift"
    a_u = summary.num_ancillas
    circuit = QuantumCircuit(k + 1 + int(lift) + a_u + summary.system_qubits, name="qubitized_" + target.kind)
    controls = list(circuit.qubits[:k])
    g = circuit.qubits[k]
    helper = [circuit.qubits[k + 1]] if lift else []
    ancillas = list(circuit.qubits[k + 1 + int(lift):k + 1 + int(lift) + a_u])
    system = list(circuit.qubits[k + 1 + int(lift) + a_u:])
    walk = [*helper, *ancillas]
    gates = {}
    source = _complete_dense_children(source, max_bytes=max_bytes, max_work=max_work)

    def query(**options):
        key = tuple(sorted(options.items()))
        if key not in gates:
            gates[key] = build_access_query(source, max_bytes=max_bytes, max_work=max_work, **options)
        return gates[key]

    def signal(negative):
        # D(W) = diag(W, I)_g, active on g = 0; D_-(W) = diag(I, W^dagger)_g, active on g = 1.
        wires = [g, *controls]
        value = int(negative) | (state << 1)
        if lift:
            def v0():
                circuit.h(helper[0])
                circuit.append(query(num_controls=1, ctrl_state=0, adjoint=True), [helper[0], *ancillas, *system])
                circuit.append(XGate().control(len(wires), ctrl_state=value, annotated=False), [*wires, helper[0]])
                circuit.append(query(num_controls=1, ctrl_state=0), [helper[0], *ancillas, *system])
                circuit.h(helper[0])
            if negative:
                # W^dagger = V_0 R_0: the reflection acts first.
                _append_reflection(circuit, walk, controls=wires, ctrl_state=value, spare=g)
                v0()
            else:
                v0()
                _append_reflection(circuit, walk, controls=wires, ctrl_state=value, spare=g)
        else:
            # A full involution has U^dagger = U, so W^dagger = U R uses the forward query.
            gate = query(num_controls=1 + k, ctrl_state=value)
            if negative:
                _append_reflection(circuit, walk, controls=wires, ctrl_state=value, spare=g)
                circuit.append(gate, [*wires, *ancillas, *system])
            else:
                circuit.append(gate, [*wires, *ancillas, *system])
                _append_reflection(circuit, walk, controls=wires, ctrl_state=value, spare=g)

    solution = target.solution
    _rotation(circuit, RXGate, solution.terminal_rx, g, controls, state)
    angles = solution.rotation_angles
    for index in range(solution.degree, -1, -1):
        left, rx, right = angles[index]
        _rotation(circuit, RZGate, right, g, controls, state)
        _rotation(circuit, RXGate, rx, g, controls, state)
        _rotation(circuit, RZGate, left, g, controls, state)
        if index:
            signal(index <= solution.negative_signals)
    return circuit


def build_qubitized_evolution(source, target: QubitizedTarget, *, evolution_time: float | None = None,
                              max_bytes: int | None = None, max_work: int | None = None) -> BlockEncoding:
    """Build the qubitized GQSP circuit of a Laurent, GQET or imaginary-time target on an encoding.

    See `plan_qubitized_evolution`. The returned encoding has ``alpha =
    target.scale`` and ``error_bound`` the recovered-frame bound of the
    plan: for real time ``||sigma B - exp(-i T_e S)||``, for the filter
    ``||sigma B - exp[-beta(1 - X)]||``. Native gate error is not included.

    Args:
        source (BlockEncodingPlan | BlockEncoding | OperatorInput | JointEncodingPlan | ScaledBranchAccess | UniformNodeRecipe | DiagonalTable | ComplexDiagonalTable): The encoding.
        target (QubitizedTarget): From `prepare_jacobi_anger_laurent`,
            `prepare_jacobi_anger_gqet` or `imaginary_time_target`.
        evolution_time (float | None): As for `plan_qubitized_evolution`.
        max_bytes (int | None): Default None, the physical memory. As for
            `build_shared_qsvt_evolution`.
        max_work (int | None): Default None, no limit.

    Returns:
        encoding (BlockEncoding): Implementation ``"qubitized_<kind>"``,
        ancillas g, (h), child ancillas, metadata with the plan record and
        the phase evidence. A zero imaginary-time beta returns the identity
        with no ancilla.
    """
    from nwqlib.subroutines.block_encoding import BlockEncoding

    plan = plan_qubitized_evolution(source, target, evolution_time=evolution_time)
    circuit = _qubitized_circuit(source, target, plan, num_controls=0, ctrl_state=None, max_bytes=max_bytes,
                                    max_work=max_work)
    evidence = None if target.solution is None else target.solution.evidence.to_dict()
    return BlockEncoding(circuit=circuit, alpha=plan.recovery, num_ancillas=plan.num_ancillas,
                         system_qubits=plan.system_qubits, error_bound=plan.error_bound,
                         implementation="qubitized_" + target.kind,
                         metadata={"plan": plan.to_dict(), "phase_evidence": evidence,
                                   "norm_cap": target.norm_cap,
                                   "norm_cap_qualification": target.norm_cap_qualification,
                                   "phase_arithmetic_error_bound": None})


def build_qubitized_evolution_query(source, target: QubitizedTarget, *, evolution_time: float | None = None,
                                    num_controls: int = 0, ctrl_state: int | str | None = None,
                                    adjoint: bool = False, max_bytes: int | None = None,
                                    max_work: int | None = None):
    """Build the qubitized circuit, its adjoint, or either with external controls, as a gate.

    External controls are added to every processing rotation and to the
    signal controls: for the direct walk the encoding query carries g and
    the controls, for the lift only ``X_h`` and the reflection do, while
    ``D_U`` keeps its single control on h. The inactive branch is then the
    identity on all qubits.

    Args:
        source (BlockEncodingPlan | BlockEncoding | OperatorInput | JointEncodingPlan | ScaledBranchAccess | UniformNodeRecipe | DiagonalTable | ComplexDiagonalTable): The encoding.
        target (QubitizedTarget): The prepared target.
        evolution_time (float | None): As for `plan_qubitized_evolution`.
        num_controls (int): Default `0`.
        ctrl_state (int | str | None): Default `None`, all controls one.
        adjoint (bool): Default `False`. Reverse the whole construction.
        max_bytes (int | None): Default None, the physical memory.
        max_work (int | None): Default None, no limit.

    Returns:
        gate (qiskit.circuit.Gate): On the controls, then the circuit's
            ancillas and system.
    """
    from nwqlib.subroutines.qiskit_compat import inverse_realized_gate

    plan = plan_qubitized_evolution(source, target, evolution_time=evolution_time)
    circuit = _qubitized_circuit(source, target, plan, num_controls=num_controls, ctrl_state=ctrl_state,
                                 max_bytes=max_bytes, max_work=max_work)
    gate = circuit.to_gate(label=circuit.name)
    return inverse_realized_gate(gate, native_ucg=False) if adjoint else gate


def imaginary_time_filter_error_terms(target: QubitizedTarget, *, contraction_error: float | None,
                                      native_error: float = 0.0) -> dict[str, Any]:
    r"""Return the error bound of the imaginary-time filter and its parts, without a circuit.

    For Hermitian ``X`` and ``X_tilde`` with spectra at most one, both
    ``beta (I - X)`` are positive semidefinite, and the Duhamel identity
    with contraction semigroups gives

        \[\epsilon_I\le\delta_m+\sigma_I\epsilon_{\rm synth}
          +\beta\delta_X+\sigma_I\epsilon_{\rm native}.\]

    Unitary-target OAA is not available for this nonunitary filter (it
    transforms singular values nonlinearly), so the recovery is
    ``sigma_I`` alone; the physical scalars ``kappa_a`` belong to the
    application weights.

    Args:
        target (QubitizedTarget): An ``"imaginary_time_gqet"`` target.
        contraction_error (float | None): ``delta_X`` of the encoding.
        native_error (float): Default 0. Full-circuit native error, if
            assessed.

    Returns:
        terms (dict): ``tail``, ``synthesis`` (``sigma_I epsilon_synth``),
        ``child`` (``beta delta_X``), ``native`` and ``filter_error`` (None
        when a term is unknown), and ``recovery`` sigma_I.
    """
    if target.kind != "imaginary_time_gqet":
        raise ValueError("AT13 applies to the imaginary-time filter only")
    if target.solution is None:
        return {"tail": 0.0, "synthesis": 0.0, "child": 0.0, "native": 0.0, "filter_error": 0.0, "recovery": 1.0}
    synthesis = None if target.synthesis_residual is None else target.scale * target.synthesis_residual
    child = None if contraction_error is None else target.polynomial_time * contraction_error
    native = target.scale * finite_real(native_error, "native_error")
    parts = (target.tail_bound, synthesis, child)
    total = None if any(part is None for part in parts) else math.fsum((*parts, native))
    return {"tail": target.tail_bound, "synthesis": synthesis, "child": child, "native": native,
            "filter_error": total, "recovery": target.scale}


__all__ = [
    "JacobiAngerExpansion",
    "QSP_OAA_QUERY_MULTIPLIER",
    "QubitizedEvolutionPlan",
    "QubitizedTarget",
    "QubitizedTargetPlan",
    "SharedQSVTEvolutionPlan",
    "build_qubitized_evolution",
    "build_qubitized_evolution_query",
    "build_shared_qsvt_evolution",
    "build_shared_qsvt_evolution_query",
    "imaginary_time_filter_error_terms",
    "imaginary_time_target",
    "plan_jacobi_anger_gqet",
    "plan_jacobi_anger_laurent",
    "plan_laurent_target",
    "plan_qubitized_evolution",
    "plan_shared_qsvt_evolution",
    "prepare_jacobi_anger_gqet",
    "prepare_jacobi_anger_laurent",
    "prepare_qubitized_target",
    "with_phase_allowance",
    "build_control_diagonal_generator_encoding",
    "build_qsp_evolution_encoding",
    "build_qsvt_circuit",
    "build_real_chebyshev_encoding",
    "compiled_select_resource_law",
    "jacobi_anger_expansion",
]
